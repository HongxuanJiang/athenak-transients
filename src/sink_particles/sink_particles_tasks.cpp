//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_tasks.cpp
//! \brief task registration and the once-per-cycle sink operator sequence.

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "driver/driver.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::AssembleSinkTasks
//! \brief register the single sink task.  Everything the sinks do is a once-per-cycle
//! operator split against the fluid update.  It lives in "after_timeintegrator"
//! (end of cycle) because that IS ORION2's operator order (advance MHD, then sinks,
//! AMRLevelOrion.cpp:863-890) and it makes the non-LAT cadence identical to the LAT
//! window-boundary invocation, so the {lat on, all factors 1} arm is bitwise-comparable
//! to the lat-off arm.  The reverse coupling (sink -> gas gravity) is a SourceTerms
//! extension applied per stage and is owned elsewhere.
//!
//! The task is registered unconditionally, LAT or not (amendment A3: masked no-op tasks,
//! never conditionally unenqueued ones).  Under LAT the driver executes the task lists
//! per BIN per tick with a rank-dependent active list, so the body below turns itself
//! into a no-op there and the driver drives the module directly from a synchronized
//! point instead -- see the LAT section of sink_particles.hpp.

void SinkParticles::AssembleSinkTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  id.step = tl["after_timeintegrator"]->AddTask(&SinkParticles::SinkStep, this, none);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::SinkStep
//! \brief one full sink cycle in ORION2's order (AMRLevelOrion.cpp:863-890):
//! gas->sink kick, sink-sink N-body, creation, merge, accretion, then dt/mirror/log.
//! Every rank runs this unconditionally with the same replicated list so the collectives
//! inside the grid stages are always matched.
//!
//! CADENCE.  The interval is `StepInterval()`: the driver's one-shot SetStepInterval
//! override when one is pending, otherwise pmesh->dt.  That makes this function correct
//! for the ordinary once-per-cycle path AND for a LAT driver that calls it once per
//! window with the window duration, or once per fine tick with the fine dt, without the
//! module having to guess which pmesh->dt it is looking at.  The override is consumed
//! here, so a driver that forgets to set it falls back to pmesh->dt rather than reusing a
//! stale window length.
//!
//! LAT NO-OP.  Under `pdrive->hydro_lat` this body runs only for the driver's own
//! synchronized-point invocation, recognized either by the sentinel stage
//! `kLATSyncStage` (the radiation_m1 RefreshLATBoundaries convention) or by a pending
//! SetStepInterval.  A task-list invocation (stage >= 0, no override) is a NO-OP: that
//! call site is inside a LAT bin, where the active list is rank dependent and the four
//! collectives below would deadlock.  The task still exists and still completes
//! (amendment A3).

TaskStatus SinkParticles::SinkStep(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  const bool driver_invoked = (stage == kLATSyncStage) || (step_interval_ > 0.0);
  if (pdrive != nullptr && pdrive->hydro_lat && !driver_invoked) {
    return TaskStatus::complete;
  }
  const Real dt = StepInterval();
  step_all_blocks_sync_ = step_interval_sync_;
  ClearStepInterval();
  const SinkMeshGeom g = MeshGeometry();

  diag = SinkDiagnostics{};

  // Structural LAT contract: factor-1 pinning of every block the accretion kernel writes,
  // and a legal cadence for the Jeans surgery.  No-op without LAT.
  CheckLATInvariants(pdrive, g);
  if (!step_all_blocks_sync_ && !lat_stale_read_warned_ &&
      global_variable::my_rank == 0) {
    lat_stale_read_warned_ = true;
    std::cout << "### WARNING: SinkStep was invoked at a LAT fine-tick boundary. The "
              << "cells it WRITES are pinned to factor 1, but the gas->sink kick and the "
              << "accretion ambient shell READ every local block, and blocks in slower "
              << "bins are still at an earlier time. Prefer invoking SinkStep at window "
              << "boundaries with the window duration." << std::endl;
  }

  // (1) gas -> sink gravitational kick, first order in dt with the start-of-cycle gas.
  // It reads the {G*M, pos, soften^2} mirror AS PUBLISHED at the end of the previous
  // SinkStep, which is exactly what SourceTerms::SinkGravity pushed the gas with all
  // cycle -- that is what makes the pair force equal and opposite.  There used to be a
  // RefreshSinkGMPos(g) here; the list cannot have changed since the last publication,
  // so its only effect was to install THIS cycle's finest spacing in soften^2, and after
  // an AMR level change the kick then integrated the reaction of a potential four times
  // more concentrated than the one that did the pushing (see RefreshSinkGMPos).
  if (!sinks.empty()) { GasSinkKick(dt, g); }

  // (2) sink-sink N-body over the full cycle (Bulirsch-Stoer + Stoermer)
  if (!sinks.empty()) { AdvanceNBody(dt, g); }

  // (3) creation from Jeans-violating finest-level cells
  if (create) { CreateSinks(g); }

  // (4) friends-of-friends merge; suppresses the swarm that (3) can produce
  if (sinks.size() >= 2) { MergeSinks(g); }

  // (5) Bondi-Hoyle accretion.  mdot is sized here so it always matches the post-merge
  // list even when accretion is disabled.
  mdot.assign(sinks.size(), 0.0);
  const std::size_t nsinks_before_accrete = sinks.size();
  if (accrete && !sinks.empty()) { Accrete(dt, g); }

  // Accretion and creation rewrite interior conserved gas state AFTER this cycle's
  // exchange+C2P, so the primitives and ghost zones the next cycle's stage-1
  // reconstruction reads would be stale (pre-surgery) without a republication.  The LAT
  // driver already reruns RefreshHydroLATBoundaries(force_c2p=true) after its
  // synchronized-point invocation; the ordinary path must republish here or the two
  // cadences diverge at the second sink step (found by the {lat}x{sink} bitwise A/B).
  const bool gas_modified =
      (accrete && nsinks_before_accrete > 0) || (create && !sinks.empty());
  if (gas_modified && pdrive != nullptr && !pdrive->hydro_lat) {
    pdrive->InitBoundaryValuesAndPrimitives(pm);
  }

  UpdateTimeStep(g);
  RefreshSinkGMPos(g);
  WriteLog(pm->time, pm->ncycle);

  // Floor activations are conservation leaks (map R17): a nonzero count is a physics
  // problem, so it is reported rather than absorbed.
  if (global_variable::my_rank == 0 &&
      (diag.n_rho_floor > 0.0 || diag.n_eint_floor > 0.0 || diag.dl_leak > 0.0)) {
    std::cout << "### WARNING: sink accretion floors activated at cycle " << pm->ncycle
              << ": n_rho=" << diag.n_rho_floor << " n_eint=" << diag.n_eint_floor
              << " dm_leak=" << diag.dm_leak << " de_leak=" << diag.de_leak
              << " dl_leak=" << diag.dl_leak << std::endl;
  }
  // A sink whose kernel reaches coarser blocks samples cells of unequal volume, which
  // biases the ambient average and the kernel normalization.  ORION2 skips such sinks
  // outright; here it is a once-per-run warning to refine around the sinks.
  if (global_variable::my_rank == 0 && !coarse_warned_ && diag.n_coarse_cell > 0.0) {
    coarse_warned_ = true;
    std::cout << "### WARNING: a sink accretion kernel overlaps blocks coarser than the "
              << "finest level (cycle " << pm->ncycle << "); refine around the sinks"
              << std::endl;
  }
  return TaskStatus::complete;
}

}  // namespace sinkparticles
