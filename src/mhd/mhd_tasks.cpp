//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//! \brief functions that control MHD tasks stored in tasklists in MeshBlockPack

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
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "pgen/pgen.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/lat_weights.hpp"

namespace {

//! \brief The coordinate excision mask when excision is enabled, else null.
inline const DvceArray4D<bool> *CoordinateExcisionMask(MeshBlockPack *pmbp) {
  return pmbp->pcoord->coord_data.bh_excise ? &(pmbp->pcoord->excision_floor) : nullptr;
}
}  // namespace

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasks
//! \brief Adds mhd tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after MHD constructor
//! See comments Hydro::AssembleHydroTasks() function for more details.
//!
//! This is the COMPOSITION of the two halves used by the radiation bindings
//! (mhd design B4): the radiation-free graph is emitted by exactly the same AddTask
//! sequence as before the split, so its TaskID bits and its schedule are identical by
//! construction rather than by inspection.  Passing id.srctrms as the part-B gate
//! reproduces the original `SendU_OA(id.srctrms)` / `SendB_OA(id.srctrms)` dependency.

void MHD::AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  AssembleMHDTasksPartA(tl);
  AssembleMHDTasksPartB(tl, id.srctrms);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::TransportStageWeights
//! \brief See mhd.hpp.  The FOFC candidates are built from the same weights so the probe
//! tests the state RKUpdate builds.

void MHD::TransportStageWeights(const Driver *d, int stage, Real &gam0,
                                Real &gam1) const {
  gam0 = d->gam0[stage-1];
  gam1 = d->gam1[stage-1];
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasksPartA
//! \brief Head of the MHD stage graph, through id.srctrms (mhd design B4).
//!
//! Everything that establishes the stage-consistent (u0, b0) pair lives here: the RK
//! update, the corner-E exchange, the CT update of the face fields, the dual-energy step
//! and the source terms.  With a radiation module bound to MHD the radiation graph hangs
//! off id.srctrms and its implicit couple rewrites u0(IEN)/u0(IM1..3) -- reading b0 only,
//! never writing it -- before part B publishes the gas through the single exchange of the
//! stage.  The optional FOFC preview/re-test chain stays here: it sits between the flux
//! evaluation and the RK update, entirely upstream of the couple.

void MHD::AssembleMHDTasksPartA(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_timeintegrator" task list
  id.savest = tl["before_timeintegrator"]->AddTask(&MHD::SaveMHDState, this, none);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&MHD::InitRecv, this, none);

  // assemble "stagen" task list
  id.copyu     = tl["stagen"]->AddTask(&MHD::CopyCons, this, none);
  id.flux      = tl["stagen"]->AddTask(&MHD::Fluxes, this, id.copyu);
  TaskID final_flux_dep = id.flux;
  const bool fofc_topology_sync = use_fofc && pmy_pack->pmesh->multilevel;
  if (fofc_topology_sync) {
    // Ordinary synchronized AMR needs only the conservative order-mask exchange.  Keep
    // the expensive LAT preview/re-test loop exclusive to an active LAT configuration.
    id.prep_fofc_flags = tl["stagen"]->AddTask(
        &MHD::PrepareFOFCFlags, this, id.flux);
    id.send_fofc_flags = tl["stagen"]->AddTask(
        &MHD::SendFOFCFlags, this, id.prep_fofc_flags);
    id.recv_fofc_flags = tl["stagen"]->AddTask(
        &MHD::RecvFOFCFlags, this, id.send_fofc_flags);
    id.replace_fofc = tl["stagen"]->AddTask(
        &MHD::ReplaceCanonicalFOFC, this, id.recv_fofc_flags);
    final_flux_dep = id.replace_fofc;
  }
  id.sendf     = tl["stagen"]->AddTask(&MHD::SendFlux, this, final_flux_dep);
  id.recvf     = tl["stagen"]->AddTask(&MHD::RecvFlux, this, id.sendf);
  id.rkupdt    = tl["stagen"]->AddTask(&MHD::RKUpdate, this, id.recvf);
  id.efld      = tl["stagen"]->AddTask(&MHD::EField, this, id.rkupdt);
  id.sende     = tl["stagen"]->AddTask(&MHD::SendE, this, id.efld);
  id.recve     = tl["stagen"]->AddTask(&MHD::RecvE, this, id.sende);
  id.ct        = tl["stagen"]->AddTask(&MHD::CT, this, id.recve);
  id.duale     = tl["stagen"]->AddTask(&MHD::DualEnergyStep, this, id.ct);
  id.srctrms   = tl["stagen"]->AddTask(&MHD::MHDSrcTerms, this, id.duale);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasksPartB
//! \brief Tail of the MHD stage graph, from the state exchange onward (mhd design B4).
//!
//! "rad_gate" is id.srctrms in the radiation-free composition, and the radiation couple
//! task when a radiation module is bound to MHD.  BOTH SendU_OA and SendB_OA carry that
//! gate: orbital advection itself ships owned U and B, so gating only restriction would
//! let the pre-couple gas be published (the ordering argument of hydro_tasks.cpp:148-53).
//! B is never touched by any radiation task, but SendB_OA must not run ahead of the
//! couple either, or the U and B halves of the orbital-advection shift would straddle it.
//! With no orbital advection configured both tasks complete immediately and the
//! restriction chain inherits the gate unchanged.  Task order is otherwise identical to
//! the tail of the pre-split AssembleMHDTasks.

void MHD::AssembleMHDTasksPartB(std::map<std::string, std::shared_ptr<TaskList>> tl,
                                TaskID rad_gate) {
  TaskID none(0);

  // assemble second half of "stagen" task list
  id.sendu_oa  = tl["stagen"]->AddTask(&MHD::SendU_OA, this, rad_gate);
  id.sendb_oa  = tl["stagen"]->AddTask(&MHD::SendB_OA, this, rad_gate);
  id.recvu_oa  = tl["stagen"]->AddTask(&MHD::RecvU_OA, this, id.sendu_oa);
  id.recvb_oa  = tl["stagen"]->AddTask(&MHD::RecvB_OA, this, id.sendb_oa);
  id.restu     = tl["stagen"]->AddTask(&MHD::RestrictU, this, id.recvu_oa);
  id.restb     = tl["stagen"]->AddTask(&MHD::RestrictB, this, id.recvb_oa);
  TaskID sendu_dep = id.restu;
  if (use_dual_energy) {
    sendu_dep = id.restu | id.restb;
  }
  id.sendu     = tl["stagen"]->AddTask(&MHD::SendU, this, sendu_dep);
  id.sendb     = tl["stagen"]->AddTask(&MHD::SendB, this, id.restb);
  id.recvu     = tl["stagen"]->AddTask(&MHD::RecvU, this, id.sendu);
  id.recvb     = tl["stagen"]->AddTask(&MHD::RecvB, this, id.sendb);
  id.sendu_shr = tl["stagen"]->AddTask(&MHD::SendU_Shr, this, id.recvu);
  id.sendb_shr = tl["stagen"]->AddTask(&MHD::SendB_Shr, this, id.recvb);
  id.recvu_shr = tl["stagen"]->AddTask(&MHD::RecvU_Shr, this, id.sendu_shr);
  id.recvb_shr = tl["stagen"]->AddTask(&MHD::RecvB_Shr, this, id.sendb_shr);
  id.prol      = tl["stagen"]->AddTask(&MHD::Prolongate, this,
                                       id.recvu_shr | id.recvb_shr);
  id.bcs       = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, this, id.prol);
  id.c2p = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.bcs);
  id.newdt     = tl["stagen"]->AddTask(&MHD::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&MHD::ClearSend, this, none);
  // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&MHD::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SaveMHDState
//! \brief Copy primitives and bcc before step to enable computation of time derivatives,
//! for example to compute jcon in GRMHD.

TaskStatus MHD::SaveMHDState(Driver *pdrive, int stage) {
  if (wbcc_saved) {
    const int nwork = pmy_pack->nmb_thispack;
    if (nwork <= 0) return TaskStatus::complete;

    auto saved_dt = wbcc_saved_dt.d_view;
    const Real pack_dt = pmy_pack->pmesh->dt;
    Kokkos::parallel_for("mhd_save_state_dt",
        athenak_lw(Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork)),
        KOKKOS_LAMBDA(const int m) {
          saved_dt(m) = pack_dt;
        });

    Kokkos::deep_copy(DevExeSpace(), wsaved, w0);
    Kokkos::deep_copy(DevExeSpace(), bccsaved, bcc0);
    wbcc_saved_dt.template modify<DevExeSpace>();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).  Note
//! this must be done for communication of BOTH conserved (cell-centered) and
//! face-centered fields AND their fluxes (with SMR/AMR).

TaskStatus MHD::InitRecv(Driver *pdrive, int stage) {
  if (pbval_fofc != nullptr) {
    // Keep the aggregate topology warm at the same unmasked synchronization points used
    // by the ordinary boundary objects.  A masked LAT phase cannot build it collectively.
    pbval_fofc->PrepareRankPackedVarMetadata(1);
  }
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nvars);
  if (tstat != TaskStatus::complete) return tstat;
  // post receives for B
  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U, always post receives for fluxes of B
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of U
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->InitFluxRecv(nvars + (dual_energy_pdv ? 1 : 0));
      if (tstat != TaskStatus::complete) return tstat;
    }
    // post receives for fluxes of B, which are used even with uniform grids
    tstat = pbval_b->InitFluxRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
    if (FOFCMaskExchangeEnabled()) {
      tstat = pbval_fofc->InitRecv(1);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection post receives for U and B
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->InitRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->InitRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries calculate x2-distance x1-boundaries have sheared and
  // with MPI post receives for U and B
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      Real time = pmy_pack->pmesh->time;
      if (stage == pdrive->nexp_stages) {
        time += pmy_pack->pmesh->dt;
      }
      tstat = psbox_u->InitRecv(time);
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->InitRecv(time);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::CopyCons
//! \brief Simple task list function that copies u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  } else if (pdrive->integrator == "rk4") {
    // RK4()4[2S] accumulates the second low-storage register at every
    // later stage.  MHD must update both the cell-centered conserved state
    // and the face-centered magnetic field used by CT.
    const Real delta_stage = pdrive->delta[stage-1];
    if (delta_stage == 0.0) return TaskStatus::complete;

    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, ie = indcs.ie;
    const int js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int nmb1 = pmy_pack->nmb_thispack - 1;

    auto u0_ = u0;
    auto u1_ = u1;
    par_for("rk4_copy_mhd_cons", DevExeSpace(), 0, nmb1, 0, nvars-1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      u1_(m,n,k,j,i) += delta_stage*u0_(m,n,k,j,i);
    });

    if (pmy_pack->pmesh->multi_d) {
      auto b01 = b0.x1f;
      auto b11 = b1.x1f;
      par_for("rk4_copy_mhd_b1", DevExeSpace(), 0, nmb1,
              ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b11(m,k,j,i) += delta_stage*b01(m,k,j,i);
      });
    }

    auto b02 = b0.x2f;
    auto b12 = b1.x2f;
    par_for("rk4_copy_mhd_b2", DevExeSpace(), 0, nmb1,
            ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b12(m,k,j,i) += delta_stage*b02(m,k,j,i);
    });

    auto b03 = b0.x3f;
    auto b13 = b1.x3f;
    par_for("rk4_copy_mhd_b3", DevExeSpace(), 0, nmb1,
            ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b13(m,k,j,i) += delta_stage*b03(m,k,j,i);
    });
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus MHD::Fluxes(Driver *pdrive, int stage) {
  if (FOFCMaskExchangeEnabled()) {
    // FOFC clears its cell flags after the local replacement.  Preserve this stage's
    // first-pass failures in the shared scratch mask before topology synchronization.
    Kokkos::deep_copy(lat_correction_mask, 0.0);
  }

  Real flux_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    flux_time += pdrive->stage_time_frac[stage-1] * pmy_pack->pmesh->dt;
  }

  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == MHD_RSolver::advect) {
    CalculateFluxes<MHD_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf) {
    CalculateFluxes<MHD_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle) {
    CalculateFluxes<MHD_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld) {
    CalculateFluxes<MHD_RSolver::hlld>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_sr) {
    CalculateFluxes<MHD_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_sr) {
    CalculateFluxes<MHD_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld_sr) {
    CalculateFluxes<MHD_RSolver::hlld_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_gr) {
    CalculateFluxes<MHD_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_gr) {
    CalculateFluxes<MHD_RSolver::hlle_gr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld_gr) {
    CalculateFluxes<MHD_RSolver::hlld_gr>(pdrive, stage);
  }

  // Add diffusive fluxes
  if (pcond != nullptr) {
    pcond->AddHeatFluxes(w0, peos->eos_data, FluxBand(uflx));
  }
  if (pvisc != nullptr) {
    pvisc->AddViscousFluxes(w0, peos->eos_data, FluxBand(uflx));
  }
  if ((presist != nullptr) && (peos->eos_data.is_ideal)) {
    presist->AddResistiveFluxes(b0, FluxBand(uflx));
  }

  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(flux_time, excise_enabled, excise_radius,
      excise_density, excise_eint, sink_x, sink_y, sink_z);

  // call FOFC if necessary
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

// The FOFC order-mask exchange is a matched quartet of tasks -- PrepareFOFCFlags,
// SendFOFCFlags, RecvFOFCFlags, ReplaceCanonicalFOFC -- queued by AssembleMHDTasksPartA
// and, for its own MHD chain, by Radiation::AssembleRadTasks.  QueueDynGRMHDTasks queues
// none of them: DynGRMHD runs its own FOFC (DynGRMHD::FOFC) and never writes
// lat_correction_mask, so the mask carries no information there.  MHD::InitRecv and
// MHD::ClearRecv are shared by every task list, though, and they post and then block on
// pbval_fofc receives purely on this predicate.
// Leaving it true under DynGRMHD posted receives that nothing would ever send, so every
// multilevel MPI run with <mhd>/fofc=true deadlocked in ClearRecv on the first stage.
// Gate on the fluid actually owning the producer side.
bool MHD::FOFCMaskExchangeEnabled() const {
  return pbval_fofc != nullptr && use_fofc && pmy_pack->pdyngr == nullptr &&
      pmy_pack->pmesh->multilevel;
}

//----------------------------------------------------------------------------------------
//! \brief Publish the post-preview ordinary-MHD failure mask through CC topology ownership.

TaskStatus MHD::PrepareFOFCFlags(Driver *pdrive, int stage) {
  if (!FOFCMaskExchangeEnabled()) return TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    Kokkos::deep_copy(coarse_fofc_mask, 0.0);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nwork = pmy_pack->nmb_thispack;
  if (nwork > 0) {
    auto ordinary_flag = fofc;
    auto mask = lat_correction_mask;
    par_for("mhd_canonical_fofc_mask", DevExeSpace(), 0, nwork-1,
            ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (ordinary_flag(m,k,j,i)) {
        mask(m,0,k,j,i) = static_cast<Real>(1.0);
      }
    });
  }
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(lat_correction_mask, coarse_fofc_mask);
  }
  return TaskStatus::complete;
}

TaskStatus MHD::SendFOFCFlags(Driver *pdrive, int stage) {
  if (!FOFCMaskExchangeEnabled()) return TaskStatus::complete;
  return pbval_fofc->PackAndSendCC(lat_correction_mask, coarse_fofc_mask);
}

TaskStatus MHD::RecvFOFCFlags(Driver *pdrive, int stage) {
  if (!FOFCMaskExchangeEnabled()) return TaskStatus::complete;
  return pbval_fofc->RecvAndUnpackCC(lat_correction_mask, coarse_fofc_mask);
}

TaskStatus MHD::ReplaceCanonicalFOFC(Driver *pdrive, int stage) {
  if (!FOFCMaskExchangeEnabled()) return TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    pbval_fofc->FillCoarseInBndryCC(lat_correction_mask, coarse_fofc_mask);
    pbval_fofc->ProlongateCC(lat_correction_mask, coarse_fofc_mask);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int il = indcs.is - 1, iu = indcs.ie + 1;
  const int jl = pmy_pack->pmesh->multi_d ? indcs.js - 1 : indcs.js;
  const int ju = pmy_pack->pmesh->multi_d ? indcs.je + 1 : indcs.je;
  const int kl = pmy_pack->pmesh->three_d ? indcs.ks - 1 : indcs.ks;
  const int ku = pmy_pack->pmesh->three_d ? indcs.ke + 1 : indcs.ke;
  const int nwork = pmy_pack->nmb_thispack;
  if (nwork > 0) {
    auto mask = lat_correction_mask;
    auto ordinary_flag = fofc;
    par_for("mhd_import_canonical_fofc_mask", DevExeSpace(), 0, nwork-1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (mask(m,0,k,j,i) > 0.0) ordinary_flag(m,k,j,i) = true;
    });
  }

  fofc_replacement_only_ = true;
  FOFC(pdrive, stage);
  fofc_replacement_only_ = false;

  // The replacement-only FOFC pass also installs its generic excision LLF flux.  Restore
  // the ordinary runtime sink's one-way/zero-flux rule before final flux communication.
  Real flux_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    flux_time += pdrive->stage_time_frac[stage-1] * pmy_pack->pmesh->dt;
  }
  ApplyExcisionSinkBoundary(flux_time);

  Kokkos::deep_copy(DevExeSpace(), lat_correction_mask, 0.0);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel)  {
    tstat = pbval_u->PackAndSendFluxCC(uflx, dual_energy_pdv ? &dual_vf : nullptr,
                                       false, 1.0, FluxOrigin(),
                                       CoordinateExcisionMask(pmy_pack));
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(
        uflx, dual_energy_pdv ? &dual_vf : nullptr, FluxOrigin());
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::MHDSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus MHD::MHDSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  Real source_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    source_time += pdrive->stage_time_frac[stage-1] * pmy_pack->pmesh->dt;
  }

  // Add physics source terms (must be computed from primitives)
  if (psrc != nullptr) psrc->ApplySrcTerms(w0, peos->eos_data,  beta_dt, u0);

  // Add shearing box source terms for CC MHD variables
  if (psbox_u != nullptr) psbox_u->SourceTermsCC(w0, bcc0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic &&
      !pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, bcc0, peos->eos_data, beta_dt, u0);
  } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pdyngr->AddCoordTerms(w0, bcc0, beta_dt, u0, pmy_pack->pmesh->mb_indcs.ng);
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
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
    pmy_pack->pmesh->pgen->user_srcs_time = old_user_srcs_time;
    pmy_pack->pmesh->pgen->user_srcs_time_valid = old_user_srcs_time_valid;
    pmy_pack->pmesh->pgen->user_srcs_final_weight = old_user_srcs_final_weight;
    pmy_pack->pmesh->pgen->user_srcs_step_dt = old_user_srcs_step_dt;
    pmy_pack->pmesh->pgen->user_srcs_stage = old_user_srcs_stage;
    pmy_pack->pmesh->pgen->user_srcs_nstages = old_user_srcs_nstages;
    pmy_pack->pmesh->pgen->user_srcs_stage_valid = old_user_srcs_stage_valid;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->PackAndSendCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->RecvAndUnpackCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus MHD::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
    if (use_dual_energy) {
      SynchronizeRestrictedDualEnergyField();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->PackAndSendCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->RecvAndUnpackCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::EField
//! \brief Wrapper task list function to compute electric field

TaskStatus MHD::EField(Driver *pdrive, int stage) {
  // Use CT to compute corner E
  CornerE(pdrive, stage);

  // Add resistive electric field (if needed)
  if (presist != nullptr) {
    presist->AddResistiveEMFs(b0, efld);
  }
  // TODO(@user): Add more resistive effects here

  if (psbox_b != nullptr) {
    // only execute when (2D)
    if (pmy_pack->pmesh->two_d) {
      psbox_b->SourceTermsFC(b0, efld);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendE
//! \brief Wrapper task list function to pack/send fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries. This is performed both
//! at MeshBlock boundaries at the same level (to keep magnetic flux in-sync on different
//! MeshBlocks), and at fine/coarse boundaries with SMR/AMR using restricted values of E.

TaskStatus MHD::SendE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->PackAndSendFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvE
//! \brief Wrapper task list function to recv/unpack fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries

TaskStatus MHD::RecvE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->RecvAndUnpackFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_b != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_b->shearing_box_r_phi)) {
      tstat = porb_b->PackAndSendFC(b0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_b != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_b->shearing_box_r_phi)) {
      tstat = porb_b->RecvAndUnpackFC(b0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendB
//! \brief Wrapper task list function to pack/send face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvB
//! \brief Wrapper task list function to recv/unpack face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_b != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_b->shearing_box_r_phi) {
      tstat = psbox_b->PackAndSendFC(b0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_b != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_b->shearing_box_r_phi) {
      tstat = psbox_b->RecvAndUnpackFC(b0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ApplyPhysicalBCs
//! \brief Wrapper task list function to call functions that set physical and user BCs

TaskStatus MHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
  pbval_b->BFieldBCs((pmy_pack), (pbval_b->b_in), b0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ProlongateFaceFieldsFC_
//! \brief The face-field operations of MHD::Prolongate on a multilevel mesh: the due
//! blocks' own coarse copy under LAT, the coarse boundary fill, the coarse physical and
//! user boundaries, and ProlongateFC.  None of them reads u0 or coarse_u0.

void MHD::ProlongateFaceFieldsFC_() {
  pbval_b->FillCoarseInBndryFC(b0, coarse_b0);
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_b->BFieldBCsCoarse(pmy_pack, pbval_b->b_in, coarse_b0);
    if (pmy_pack->pmesh->pgen->user_bcs) {
      pbval_b->FillCoarseUserBndryFC(b0, coarse_b0);
    }
  }
  pbval_b->ProlongateFC(b0, coarse_b0);
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse bundaries with SMR/AMR

TaskStatus MHD::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_b->FillCoarseInBndryFC(b0, coarse_b0);
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      pbval_u->HydroBCsCoarse(pmy_pack, pbval_u->u_in, coarse_u0);
      pbval_b->BFieldBCsCoarse(pmy_pack, pbval_b->b_in, coarse_b0);
      if (pmy_pack->pmesh->pgen->user_bcs) {
        pbval_u->FillCoarseUserBndryCC(u0, coarse_u0);
        pbval_b->FillCoarseUserBndryFC(b0, coarse_b0);
      }
    }
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_b0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_b->ProlongateFC(b0, coarse_b0);
      pbval_u->PrimToConsFineBndry(w0, b0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
      pbval_b->ProlongateFC(b0, coarse_b0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  if (use_dual_energy && dual_energy_pdv) {
    SynchronizeDualEnergyFieldFromTotal();
  }
  peos->ConsToPrim(u0, b0, w0, bcc0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check sends of U and B complete
    tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = pbval_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends for fluxes of U complete.  Always check sends of E complete
  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check sends of restricted fluxes of B complete even for uniform grids
    tstat = pbval_b->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
    if (FOFCMaskExchangeEnabled()) {
      tstat = pbval_fofc->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection check sends for U and B complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check sends of U and B complete
  if (psbox_u != nullptr) {
    // only execute when (stage>=0 or -4) AND (3D OR 2d_r_phi)
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check receives of U and B complete
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = pbval_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check recvs for fluxes of U complete.  Always check recvs of E complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // with SMR/AMR check receives of restricted fluxes of B complete
    tstat = pbval_b->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
    if (FOFCMaskExchangeEnabled()) {
      tstat = pbval_fofc->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with orbital advection check receives of U and B are complete
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check receives of U and B complete
  if (psbox_u != nullptr) {
    // only execute when (stage>=0 or -4) AND (3D OR 2d_r_phi)
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictB
//! \brief Wrapper function that restricts face-centered variables (magnetic field)

TaskStatus MHD::RestrictB(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictFC(b0, coarse_b0);
  }
  return TaskStatus::complete;
}

} // namespace mhd
