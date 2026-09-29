//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//! \brief functions that control Hydro tasks stored in tasklists in MeshBlockPack

#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "pgen/pgen.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "driver/lat_weights.hpp"

namespace {

bool SkipFinalLATBoundaryExchange(Driver *pdrive, hydro::Hydro *phydro, int stage) {
  return pdrive != nullptr && pdrive->hydro_lat_skip_final_exchange_this_bin &&
         stage == pdrive->nexp_stages &&
         phydro->porb_u == nullptr &&
         phydro->psbox_u == nullptr;
}

bool SkipLATUnionStage1StateExchange(Driver *pdrive, int stage) {
  return pdrive != nullptr && pdrive->hydro_lat_union_stage1_active && stage == 1;
}
}  // namespace

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasks
//! \brief Adds hydro tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after Hydro constructor.
//! Many of the functions in the task list are implemented in this file because they are
//! simple, or they are wrappers that call one or more other functions.
//!
//! "before_stagen" tasks are those that must be completed over all MeshBlocks BEFORE each
//! stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//!
//! "stagen" tasks are those performed DURING each stage
//!
//! "after_stagen" tasks are those that can only be completed AFTER all the "stagen" tasks
//! are completed over ALL MeshBlocks for each stage, such as clearing all MPI calls, etc.
//!
//! In addition there are "before_timeintegrator" and "after_timeintegrator" task lists
//! in the tl map, which are generally used for operator split tasks.

void Hydro::AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  // Composed from the two halves rather than spelled out a third time, exactly as
  // MHD::AssembleMHDTasks is.  TaskIDs are assigned from the AddTask call order, so the
  // composition reproduces the monolithic graph BY CONSTRUCTION -- the previous
  // duplicate-and-keep-in-step arrangement was identical only by inspection and could
  // drift silently, which is precisely the failure this file's LAT dependencies (the
  // rk_flux_dep selection) would make hard to see.  The gate handed to part B is
  // id.statefix, the task the monolithic list attached SendU_OA to.
  AssembleHydroTasksPartA(tl);
  AssembleHydroTasksPartB(tl, id.statefix);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasksPartA
//! \brief First half of the hydro stage graph.
//!
//! AssembleHydroTasks is this function followed by part B, so the radiation-free graph
//! is identical to the split one by construction.  Part A stops at id.statefix: from
//! there the radiation graph runs its transport and its implicit couple, which writes
//! u0(IEN) and u0(IM1..3) before part B publishes the gas through the one exchange of
//! this stage.

void Hydro::AssembleHydroTasksPartA(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&Hydro::InitRecv, this, none);

  // assemble first half of "stagen" task list
  id.statefix_pre = tl["stagen"]->AddTask(&Hydro::HydroStateFixup, this, none);
  id.copyu     = tl["stagen"]->AddTask(&Hydro::CopyCons, this, id.statefix_pre);
  id.flux      = tl["stagen"]->AddTask(&Hydro::Fluxes,this,id.copyu);
  id.sendf     = tl["stagen"]->AddTask(&Hydro::SendFlux, this, id.flux);
  id.recvf     = tl["stagen"]->AddTask(&Hydro::RecvFlux, this, id.sendf);
  // Multilevel LAT can replace a coarse stage flux, including in a common-time
  // unequal-bin predictor, so RKUpdate must wait for that receive.  Same-level-only LAT
  // remains delayed.
  const TaskID rk_flux_dep = (lat_requested_ && !(pmy_pack->pmesh->multilevel)) ?
      id.sendf : id.recvf;
  id.rkupdt    = tl["stagen"]->AddTask(&Hydro::RKUpdate, this, rk_flux_dep);
  id.duale     = tl["stagen"]->AddTask(&Hydro::DualEnergyStep, this, id.rkupdt);
  id.srctrms   = tl["stagen"]->AddTask(&Hydro::HydroSrcTerms, this, id.duale);
  id.statefix  = tl["stagen"]->AddTask(&Hydro::HydroStateFixup, this, id.srctrms);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasksPartB
//! \brief Second half of the hydro stage graph.
//!
//! "rad_gate" is the task everything here waits on: the radiation couple when
//! a radiation module is on, and id.statefix (what the monolithic list used) otherwise.
//! The gate is attached to the first task of the chain rather than to RestrictU so that
//! orbital advection -- which itself ships owned U -- cannot publish the pre-couple
//! state; with no orbital advection configured SendU_OA/RecvU_OA complete immediately
//! and RestrictU inherits the gate unchanged.

void Hydro::AssembleHydroTasksPartB(
    std::map<std::string, std::shared_ptr<TaskList>> tl, TaskID rad_gate) {
  TaskID none(0);

  // assemble second half of "stagen" task list
  id.sendu_oa  = tl["stagen"]->AddTask(&Hydro::SendU_OA, this, rad_gate);
  id.recvu_oa  = tl["stagen"]->AddTask(&Hydro::RecvU_OA, this, id.sendu_oa);
  id.restu     = tl["stagen"]->AddTask(&Hydro::RestrictU, this, id.recvu_oa);
  id.sendu     = tl["stagen"]->AddTask(&Hydro::SendU, this, id.restu);
  id.recvu     = tl["stagen"]->AddTask(&Hydro::RecvU, this, id.sendu);
  id.sendu_shr = tl["stagen"]->AddTask(&Hydro::SendU_Shr, this, id.recvu);
  id.recvu_shr = tl["stagen"]->AddTask(&Hydro::RecvU_Shr, this, id.sendu_shr);
  id.prol      = tl["stagen"]->AddTask(&Hydro::Prolongate, this, id.recvu_shr);
  id.bcs       = tl["stagen"]->AddTask(&Hydro::ApplyPhysicalBCs, this, id.prol);
  id.statefix_post = tl["stagen"]->AddTask(&Hydro::HydroStateFixup, this, id.bcs);
  id.c2p       = tl["stagen"]->AddTask(&Hydro::ConToPrim, this, id.statefix_post);
  id.newdt     = tl["stagen"]->AddTask(&Hydro::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&Hydro::ClearSend, this, none);
  // although RecvFlux/U functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&Hydro::ClearRecv, this, id.csend);
  id.latdense = tl["after_stagen"]->AddTask(&Hydro::SaveLATDenseOutput, this, id.crecv);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).

TaskStatus Hydro::InitRecv(Driver *pdrive, int stage) {
  flux_recv_complete_ = false;
  const bool lat_same_level = pmy_pack->pmesh->hydro_lat_same_level &&
                              pmy_pack->lat_active_mask_enabled;
  const bool delayed_lat_fluxes = pmy_pack->pmesh->multilevel || lat_same_level;
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return delayed_lat_fluxes ?
        pbval_u->InitFluxRecv(nvars + (dual_energy_pdv ? 1 : 0)) :
        TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    if (delayed_lat_fluxes && (stage >= 0)) {
      return pbval_u->InitFluxRecv(nvars + (dual_energy_pdv ? 1 : 0));
    }
    return TaskStatus::complete;
  }

  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nvars);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (delayed_lat_fluxes && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nvars + (dual_energy_pdv ? 1 : 0));
  }
  if (tstat != TaskStatus::complete) return tstat;

  // with orbital advection post receives for U
  // only execute for (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->InitRecv();
    }
  }
  if (tstat != TaskStatus::complete) return tstat;

  // with shearing box boundaries calculate x2-distance x1-boundaries have sheared and
  // with MPI post receives for U.
  // only execute if (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      Real time = pmy_pack->pmesh->time;
      if (stage == pdrive->nexp_stages) {
        time += pmy_pack->pmesh->dt;
      }
      tstat = psbox_u->InitRecv(time);
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStateFixup
//! \brief Optional problem hook that repairs owned hydro cells before boundary exchange.

TaskStatus Hydro::HydroStateFixup(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  if (pm != nullptr && pm->pgen != nullptr &&
      pm->pgen->user_hydro_state_fixup_func != nullptr) {
    Real t = pm->time;
    if (stage > 0 && pm->dt > 0.0) {
      const Real stage_time = pdrive->hydro_lat ?
          pdrive->stage_time_frac[stage - 1] : pdrive->gam0[stage - 1];
      t += stage_time * pm->dt;
    }
    const Real old_user_srcs_time = pm->pgen->user_srcs_time;
    const bool old_user_srcs_time_valid = pm->pgen->user_srcs_time_valid;
    pm->pgen->user_srcs_time = t;
    pm->pgen->user_srcs_time_valid =
        (stage > 0) || problem_runtime::HasHydroStageTime();
    pm->pgen->user_hydro_state_fixup_func(pmy_pack, t);
    pm->pgen->user_srcs_time = old_user_srcs_time;
    pm->pgen->user_srcs_time_valid = old_user_srcs_time_valid;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyCons
//! \brief Simple task list function that copies u0 --> u1 in first stage.  Extended to
//!  handle RK register logic at given stage

TaskStatus Hydro::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    if (!(pmy_pack->lat_active_mask_enabled)) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*indcs.ng;
      int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
      int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
      int nactive1 = pmy_pack->lat_nactive_thispack - 1;
      if (nactive1 < 0) return TaskStatus::complete;
      int nvar = nvars;
      auto &u0 = pmy_pack->phydro->u0;
      auto &u1 = pmy_pack->phydro->u1;
      auto active_indices = pmy_pack->lat_active_indices.d_view;
      par_for("copy_cons_lat", DevExeSpace(), 0, nactive1, 0, nvar-1, 0, ncells3-1,
              0, ncells2-1, 0, ncells1-1,
      KOKKOS_LAMBDA(int a, int n, int k, int j, int i) {
        const int m = active_indices(a);
        u1(m,n,k,j,i) = u0(m,n,k,j,i);
      });
    }
  } else {
    if (pdrive->integrator == "rk4") {
      // parallel loop to update u1 with u0 at later stages, only for rk4
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie;
      int js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int nvar = nvars;
      auto &u0 = pmy_pack->phydro->u0;
      auto &u1 = pmy_pack->phydro->u1;
      const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
      auto active = pmy_pack->lat_active_mb.d_view;
      Real &delta = pdrive->delta[stage-1];
      par_for("rk4_copy_cons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        if (lat_enabled && active(m) == 0) return;
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus Hydro::Fluxes(Driver *pdrive, int stage) {
  Real flux_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    const Real stage_time = pdrive->hydro_lat ?
        pdrive->stage_time_frac[stage-1] : pdrive->gam0[stage-1];
    flux_time += stage_time * pmy_pack->pmesh->dt;
  }

  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == Hydro_RSolver::advect) {
    CalculateFluxes<Hydro_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf) {
    CalculateFluxes<Hydro_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle) {
    CalculateFluxes<Hydro_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc) {
    CalculateFluxes<Hydro_RSolver::hllc>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::roe) {
    CalculateFluxes<Hydro_RSolver::roe>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_sr) {
    CalculateFluxes<Hydro_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_sr) {
    CalculateFluxes<Hydro_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc_sr) {
    CalculateFluxes<Hydro_RSolver::hllc_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_gr) {
    CalculateFluxes<Hydro_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_gr) {
    CalculateFluxes<Hydro_RSolver::hlle_gr>(pdrive, stage);
  }

  // Add diffusion fluxes
  if (pcond != nullptr) {
    pcond->AddHeatFluxes(w0, peos->eos_data, uflx);
  }
  if (pvisc != nullptr) {
    pvisc->AddViscousFluxes(w0, peos->eos_data, uflx);
  }

  // A slower bin's corrector with a pending finer neighbour runs on the fine flux
  // estimate on that face (MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC).  It is
  // installed here, ahead of FOFC and of the excision flux boundary, as MHD::Fluxes and
  // DynGRMHDPS::CalcFluxes install it: added afterwards it is a forcing the FOFC trial
  // state never saw, and it would overwrite the fluxes FOFC and the sink boundary had
  // just imposed.  On a union predictor the mismatch is captured in RecvFlux and is
  // measured against this flux, before FOFC can swap in a first-order one.  The edge
  // cells a same-level neighbour tests without it are tested first on the fluxes it sees
  // (Hydro::StashFOFCEdgeFlags).
  if (pmy_pack->lat_active_mask_enabled &&
      (pmy_pack->pmesh->multilevel || pmy_pack->pmesh->hydro_lat_same_level)) {
    if (use_fofc && pbval_u->PendingFineFluxMismatchDue()) {
      StashFOFCEdgeFlags(pdrive, stage);
    }
    pbval_u->AddPendingFineFluxMismatchCC(uflx, dual_energy_pdv ? &dual_vf : nullptr);
    if (pdrive->hydro_lat_union_stage1_active && stage == 1) {
      pbval_u->SnapshotPendingOwnFluxCC(
          lat_reflux, dual_energy_pdv ? &lat_dual_vf_reflux : nullptr,
          uflx, dual_energy_pdv ? &dual_vf : nullptr);
    }
  }

  // call FOFC if necessary
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(flux_time, excise_enabled,
      excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  } else if (excise_enabled) {
    FOFC(pdrive, stage);
  }

  if (excise_enabled) {
    ApplyExcisionSinkBoundary(flux_time);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  const bool lat_same_level = pmy_pack->pmesh->hydro_lat_same_level &&
                              pmy_pack->lat_active_mask_enabled;
  // A fine face between two excised cells enters the restricted flux as zero (see
  // MHD::SendFlux): near the Kerr-Schild ring its first-order solve is not bounded.
  const DvceArray4D<bool> *excised = pmy_pack->pcoord->coord_data.bh_excise ?
                                     &(pmy_pack->pcoord->excision_floor) : nullptr;
  // Normal flux correction is for SMR/AMR; LAT also uses it for mixed same-level faces.
  if (pmy_pack->pmesh->multilevel || lat_same_level) {
    if (pmy_pack->lat_active_mask_enabled) {
      // The pending finer neighbour's fine flux estimate, installed by Hydro::Fluxes,
      // is booked here with the rest of the stage flux.
      AccumulateLATCoarseFluxes(pdrive, stage);
      const bool time_integrate = pmy_pack->lat_per_block_timestep;
      tstat = pbval_u->PackAndSendFluxCC(
          uflx, dual_energy_pdv ? &dual_vf : nullptr, time_integrate,
          lat::FinalFluxWeight(pdrive, stage), {}, excised);
    } else {
      tstat = pbval_u->PackAndSendFluxCC(uflx, dual_energy_pdv ? &dual_vf : nullptr,
                                         false, 1.0, {}, excised);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  const bool lat_same_level = pmy_pack->pmesh->hydro_lat_same_level &&
                              pmy_pack->lat_active_mask_enabled;
  // Normal flux correction is for SMR/AMR; LAT also uses it for mixed same-level faces.
  if (pmy_pack->pmesh->multilevel || lat_same_level) {
    if (pmy_pack->lat_active_mask_enabled) {
      const Real scale = pmy_pack->lat_per_block_timestep ?
          static_cast<Real>(1.0) :
          lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt;
      const Real sync_integrated_weight = pmy_pack->lat_per_block_timestep ?
          lat::FinalFluxWeight(pdrive, stage) : static_cast<Real>(0.0);
      tstat = pbval_u->RecvAndAccumulateFluxCC(
          lat_reflux, scale, true,
          dual_energy_pdv ? &lat_dual_vf_reflux : nullptr, &uflx,
          dual_energy_pdv ? &dual_vf : nullptr, sync_integrated_weight, true, {}, true);
    } else {
      tstat = pbval_u->RecvAndUnpackFluxCC(uflx, dual_energy_pdv ? &dual_vf : nullptr);
    }
  }
  flux_recv_complete_ = (tstat == TaskStatus::complete);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::HydroSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus Hydro::HydroSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  Real source_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    const Real stage_time = pdrive->hydro_lat ?
        pdrive->stage_time_frac[stage-1] : pdrive->gam0[stage-1];
    source_time += stage_time * pmy_pack->pmesh->dt;
  }

  // Add physics source terms (must be computed from primitives)
  const bool ledger_debug = pmy_pack->pgrav != nullptr && pmy_pack->pgrav->lat_ledger_debug &&
      pmy_pack->pgrav->energy_window_open;
  const Real energy_before = ledger_debug ? pmy_pack->pgrav->TotalGasEnergy() : 0.0;
  if (ledger_debug && beta_dt != 0.0) {
    pmy_pack->pgrav->debug_stage_weight =
        lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt/beta_dt;
  }
  if (psrc != nullptr) psrc->ApplySrcTerms(w0, peos->eos_data,  beta_dt, u0);
  if (ledger_debug && beta_dt != 0.0) {
    // Effective contribution of this stage's sources to the completed step.
    pmy_pack->pgrav->debug_source_energy_local +=
        (pmy_pack->pgrav->TotalGasEnergy()-energy_before)*
        (lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt/beta_dt);
  }
  if (pmy_pack->pgrav != nullptr) {
    pmy_pack->pgrav->AccumulateLATBoundaryFlux(
        lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt);
  }
  if (pmy_pack->pmesh->pgen->user_hydro_gravity_ledger_func != nullptr) {
    (pmy_pack->pmesh->pgen->user_hydro_gravity_ledger_func)(
        pmy_pack->pmesh, lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt);
  }

  // Add shearing box source terms for cell-centered hydro variables
  if (psbox_u != nullptr) psbox_u->SourceTermsCC(w0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, peos->eos_data, beta_dt, u0);
  }
  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    const Real old_user_srcs_time = pmy_pack->pmesh->pgen->user_srcs_time;
    const bool old_user_srcs_time_valid =
        pmy_pack->pmesh->pgen->user_srcs_time_valid;
    const Real old_user_srcs_final_weight =
        pmy_pack->pmesh->pgen->user_srcs_final_weight;
    const Real old_user_srcs_step_dt = pmy_pack->pmesh->pgen->user_srcs_step_dt;
    const int old_user_srcs_stage = pmy_pack->pmesh->pgen->user_srcs_stage;
    const int old_user_srcs_nstages = pmy_pack->pmesh->pgen->user_srcs_nstages;
    const bool old_user_srcs_stage_valid =
        pmy_pack->pmesh->pgen->user_srcs_stage_valid;
    pmy_pack->pmesh->pgen->user_srcs_time = source_time;
    pmy_pack->pmesh->pgen->user_srcs_time_valid = true;
    pmy_pack->pmesh->pgen->user_srcs_final_weight =
        lat::FinalFluxWeight(pdrive, stage);
    pmy_pack->pmesh->pgen->user_srcs_step_dt = pmy_pack->pmesh->dt;
    pmy_pack->pmesh->pgen->user_srcs_stage = stage;
    pmy_pack->pmesh->pgen->user_srcs_nstages = pdrive->nexp_stages;
    pmy_pack->pmesh->pgen->user_srcs_stage_valid = true;
    const Real user_energy_before = ledger_debug ? pmy_pack->pgrav->TotalGasEnergy() : 0.0;
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
    if (ledger_debug && beta_dt != 0.0) {
      pmy_pack->pgrav->debug_user_source_energy_local +=
          (pmy_pack->pgrav->TotalGasEnergy()-user_energy_before)*
          (lat::FinalFluxWeight(pdrive, stage)*pmy_pack->pmesh->dt/beta_dt);
    }
    pmy_pack->pmesh->pgen->user_srcs_time = old_user_srcs_time;
    pmy_pack->pmesh->pgen->user_srcs_time_valid = old_user_srcs_time_valid;
    pmy_pack->pmesh->pgen->user_srcs_final_weight =
        old_user_srcs_final_weight;
    pmy_pack->pmesh->pgen->user_srcs_step_dt = old_user_srcs_step_dt;
    pmy_pack->pmesh->pgen->user_srcs_stage = old_user_srcs_stage;
    pmy_pack->pmesh->pgen->user_srcs_nstages = old_user_srcs_nstages;
    pmy_pack->pmesh->pgen->user_srcs_stage_valid =
        old_user_srcs_stage_valid;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus Hydro::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute if (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->PackAndSendCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection
//! Orbital remap is performed in this step.

TaskStatus Hydro::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute if (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->RecvAndUnpackCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus Hydro::RestrictU(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    if (pmy_pack->pmesh->multilevel) {
      // The ordered corrector waves may need the common-time start state from a
      // still-pending faster neighbor.  Build that restricted register once for
      // the entire due union; u0 restriction is deferred to each factor refresh.
      pmy_pack->pmesh->pmr->RestrictCC(u1, coarse_u1);
    }
    return TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    return TaskStatus::complete;
  }
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
    // In RK2 (and imex2, whose explicit tableau is identical), u1 is written only by
    // CopyCons at stage 1.  Its restricted state is therefore unchanged during stage 2
    // and during the stage-0 LAT refreshes.  RK1 retains the conservative behavior
    // because its final exchange can be skipped.
    const bool restrict_lat_start = pmy_pack->lat_active_mask_enabled &&
        (!pdrive->integrator_rk2_equiv || stage == 1);
    if (restrict_lat_start) {
      pmy_pack->pmesh->pmr->RestrictCC(u1, coarse_u1);
    }
    if (use_dual_energy) {
      // Keep the primitive-side auxiliary field in lockstep with the restricted
      // conserved auxiliary state used during coarse-to-fine prolongation.
      SynchronizeRestrictedDualEnergyField();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus Hydro::SendU(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    return TaskStatus::complete;
  }
  DvceArray5D<Real> *u_start = nullptr;
  DvceArray5D<Real> *coarse_start = nullptr;
  DvceArray5D<Real> *u_stage1 = nullptr;
  Real target_time = 0.0;
  if (pmy_pack->lat_active_mask_enabled) {
    u_start = &u1;
    coarse_start = pmy_pack->pmesh->multilevel ? &coarse_u1 : nullptr;
    if (lat_dense_output_enabled && pdrive->integrator_rk2_equiv) {
      u_stage1 = &lat_u_stage1;
    }
    if (pdrive->hydro_lat_exchange_time_set) {
      target_time = pdrive->hydro_lat_exchange_time;
    } else if (stage > 0 && stage < pdrive->nexp_stages) {
      target_time = pmy_pack->pmesh->time +
                    pdrive->stage_time_frac[stage]*pmy_pack->pmesh->dt;
    } else {
      target_time = pmy_pack->pmesh->time + pmy_pack->pmesh->dt;
    }
  }
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0, u_start, target_time,
                                            coarse_start, u_stage1);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus Hydro::RecvU(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    return TaskStatus::complete;
  }
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus Hydro::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute if (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->PackAndSendCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus Hydro::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute if (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->RecvAndUnpackCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ApplyPhysicalBCs
//! \brief Wrapper task list function to call functions that set physical and user BCs,

TaskStatus Hydro::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    return TaskStatus::complete;
  }
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Hydro::Prolongate(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return TaskStatus::complete;
  }
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    return TaskStatus::complete;
  }
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    if (pmy_pack->lat_active_mask_enabled) {
      // The stencil reads the receiver's OWN first coarse layer (ProlongCC takes ca(i-1)
      // and ca(i+1) about every coarse cell it fills), which only restriction writes.
      // Under LAT RestrictU covers the blocks that send; a due block whose neighbours
      // are all inactive or pending sends nothing and would prolongate against the
      // coarse copy of its previous step (the second step of a factor-2 bin inside a
      // factor-4 window: first order in time).
      pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, false, true);
      if (use_dual_energy) SynchronizeRestrictedDualEnergyField(true);
    }
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      pbval_u->HydroBCsCoarse(pmy_pack, pbval_u->u_in, coarse_u0);
      if (pmy_pack->pmesh->pgen->user_bcs) {
        pbval_u->FillCoarseUserBndryCC(u0, coarse_u0);
      }
    }
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_u->PrimToConsFineBndry(w0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage) {
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // Floor energy bookkeeping: this stage's state enters the completed step with the
  // weight FinalFluxWeight/beta (rk2: 0.5 after stage 1, 1 after stage 2).
  struct WeightScope {
    Hydro *hydro;
    explicit WeightScope(Hydro *h, Real w) : hydro(h) { hydro->floor_energy_weight = w; }
    ~WeightScope() { hydro->floor_energy_weight = 1.0; }
  } weight_scope(this, (stage >= 1 && pdrive->beta[stage-1] > 0.0) ?
                 lat::FinalFluxWeight(pdrive, stage)/pdrive->beta[stage-1] : 1.0);
  if (SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    // The next common-time refresh repairs ghost primitives before they are consumed.
    // Owned C2P cannot be deferred: it applies floors to u0 and prepares w0 for the
    // block's next update.  Dual-energy runs never enter this path.
    peos->ConsToPrim(u0, w0, false, indcs.is, indcs.ie, indcs.js, indcs.je,
                    indcs.ks, indcs.ke);
    return TaskStatus::complete;
  }
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  if (use_dual_energy) {
    // The eta2 overwrite test needs the current neighbor state across MeshBlock and
    // coarse-fine boundaries, so perform it only after RecvU/BCs/Prolongate are done.
    SynchronizeDualEnergyFieldFromTotal();
  }
  peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ConToPrimGhostBands
//! \brief Ghost-band-only C2P for LAT boundary refreshes.  Interior primitives are
//! already current from the owning block's last stage or post-correction recovery, and
//! the refresh only replaced ghost zones (exchange/prolongation/physical BCs).  Callers
//! must use the full ConToPrim whenever interior conserved state may have changed.
//! With dual energy the eta2 overwrite test is a one-cell stencil, so the band is
//! widened by one interior layer: that layer is the only interior that can see the
//! refreshed ghosts, every deeper cell was converted from unchanged conserved state by
//! the last full pass.  Orbital/shearing configurations still take the full path.

TaskStatus Hydro::ConToPrimGhostBands(Driver *pdrive, int stage) {
  if (psbox_u != nullptr || porb_u != nullptr) {
    return ConToPrim(pdrive, stage);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  const int bw = dual_energy_pdv ? 1 : 0;
  const int is = indcs.is + bw, ie = indcs.ie - bw;
  const int js = (indcs.nx2 > 1) ? (indcs.js + bw) : indcs.js;
  const int je = (indcs.nx2 > 1) ? (indcs.je - bw) : indcs.je;
  const int ks = (indcs.nx3 > 1) ? (indcs.ks + bw) : indcs.ks;
  const int ke = (indcs.nx3 > 1) ? (indcs.ke - bw) : indcs.ke;
  if (use_dual_energy) {
    SynchronizeDualEnergyFieldFromTotal(true);
  }
  if (indcs.nx3 > 1) {
    peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, ks-1);
    peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, ke+1, n3m1);
  }
  if (indcs.nx2 > 1) {
    peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, js-1, ks, ke);
    peos->ConsToPrim(u0, w0, false, 0, n1m1, je+1, n2m1, ks, ke);
  }
  peos->ConsToPrim(u0, w0, false, 0, is-1, js, je, ks, ke);
  peos->ConsToPrim(u0, w0, false, ie+1, n1m1, js, je, ks, ke);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears sends of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears sends of U, Flx_U,       U_Shr
//! If stage=(-1):              clears sends of U
//! If stage=(-4):              clears sends of                 U_Shr

TaskStatus Hydro::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  const bool lat_same_level = pmy_pack->pmesh->hydro_lat_same_level &&
                              pmy_pack->lat_active_mask_enabled;
  const bool delayed_lat_fluxes = pmy_pack->pmesh->multilevel || lat_same_level;
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return delayed_lat_fluxes ? pbval_u->ClearFluxSend() : TaskStatus::complete;
  }
  // check sends of U complete
  if (((stage >= 0) || (stage == -1)) &&
      !SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends of restricted fluxes of U complete
  // do not check flux send for ICs (stage < 0)
  if (delayed_lat_fluxes && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check sends of U complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check sends of U complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears recvs of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears recvs of U, Flx_U,       U_Shr
//! If stage=(-1):              clears recvs of U
//! If stage=(-4):              clears recvs of                 U_Shr

TaskStatus Hydro::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  const bool lat_same_level = pmy_pack->pmesh->hydro_lat_same_level &&
                              pmy_pack->lat_active_mask_enabled;
  const bool delayed_lat_fluxes = pmy_pack->pmesh->multilevel || lat_same_level;
  if (SkipLATUnionStage1StateExchange(pdrive, stage)) {
    return delayed_lat_fluxes ? pbval_u->ClearFluxRecv() : TaskStatus::complete;
  }
  // check receives of U complete
  if (((stage >= 0) || (stage == -1)) &&
      !SkipFinalLATBoundaryExchange(pdrive, this, stage)) {
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check receives of restricted fluxes of U complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (delayed_lat_fluxes && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check receives of U complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check receives of U complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SaveLATDenseOutput
//! \brief Save the RK2 stage-1 endpoint used for same-level LAT dense interpolation.

TaskStatus Hydro::SaveLATDenseOutput(Driver *pdrive, int stage) {
  // Union predictors are saved explicitly after each factor's boundary/C2P refresh,
  // when u0 matches the stage-1 endpoint stored by the legacy factor path.
  if (!lat_dense_output_enabled || !(pmy_pack->lat_active_mask_enabled) ||
      !pdrive->integrator_rk2_equiv || stage != 1 ||
      pdrive->hydro_lat_union_stage1_active ||
      pdrive->hydro_lat_active_factor_this_bin <= 1) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  int nactive1 = pmy_pack->lat_nactive_thispack - 1;
  if (nactive1 < 0) return TaskStatus::complete;
  int nvar = nvars;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto src = u0;
  auto dst = lat_u_stage1;
  par_for("lat_dense_stage1", DevExeSpace(), 0, nactive1, 0, nvar-1,
          0, ncells3-1, 0, ncells2-1, 0, ncells1-1,
  KOKKOS_LAMBDA(int a, int n, int k, int j, int i) {
    const int m = active_indices(a);
    dst(m,n,k,j,i) = src(m,n,k,j,i);
  });
  return TaskStatus::complete;
}

} // namespace hydro
