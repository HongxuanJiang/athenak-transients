#ifndef DYN_GRMHD_DYN_GRMHD_PS_IMPL_HPP_
#define DYN_GRMHD_DYN_GRMHD_PS_IMPL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_ps_impl.hpp
//! \brief The members of DynGRMHDPS that its explicit class instantiation compiles: the
//! task list, the primitive/conserved conversions and AddCoordTerms (whose
//! AddCoordTermsEOS<NGHOST> is in dyn_grmhd_coord_terms_impl.hpp).  Included only by the
//! dyn_grmhd_ps_<eos>.cpp units, each of which instantiates the class for one EOS policy
//! or family, so each policy's C2P kernels are compiled in exactly one unit.

#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "z4c/tmunu.hpp"
#include "dyn_grmhd.hpp"
#include "tasklist/numerical_relativity.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::QueueDynGRMHDTasks() {
  using namespace mhd;  // NOLINT(build/namespaces)
  using namespace z4c;  // NOLINT(build/namespaces)
  using namespace numrel; // NOLINT(build/namespaces))
  Z4c *pz4c = pmy_pack->pz4c;
  adm::ADM *padm = pmy_pack->padm;
  MHD *pmhd = pmy_pack->pmhd;
  NumericalRelativity *pnr = pmy_pack->pnr;

  // Interior-first ConsToPrim (see the block comment above
  // DynGRMHD::PhysicalBCInteriorReadDepth).  Only the graph SHAPE is decided here; the
  // per-call legality test lives in DynGRMHD::InteriorFirstC2PUsable, so a configuration
  // that falls outside the audit still gets exactly one full-extent recovery, in the same
  // DAG slot as before.  Z4c is excluded already here because MHD_C2P carries an optional
  // Z4c_Excise dependency that this split has not been audited against.
  const bool interior_first_c2p = (pz4c == nullptr);

  // Start task list
  pnr->QueueTask(&MHD::InitRecv, pmhd, MHD_Recv, "MHD_Recv", Task_Start);

  // A prescribed time-dependent metric must be installed before any stage-local
  // fluid or radiation kernel uses it.  Conserved-to-primitive recovery is
  // metric-dependent, so refresh primitives before the RK register copy and
  // flux calculation as well.  The driver exposes the same RK/IMEX stage time
  // used by fluid flux/source hooks through problem_runtime, including LAT bin
  // and refresh overrides.
  const bool prescribed_dynamic_adm =
      (pz4c == nullptr && padm->is_dynamic);
  std::vector<TaskName> copy_cons_deps;
  if (prescribed_dynamic_adm) {
    pnr->QueueTask(&DynGRMHD::SetADMVariables, this, MHD_SetADM, "MHD_SetADM",
                   Task_Run);
    pnr->QueueTask(&DynGRMHD::UpdateExcisionMasks, this, MHD_Excise, "MHD_Excise",
                   Task_Run, {MHD_SetADM});
    pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this,
                   MHD_PreFluxC2P, "MHD_PreFluxC2P", Task_Run, {MHD_Excise});
    copy_cons_deps.push_back(MHD_PreFluxC2P);
  }
  pnr->QueueTask(&MHD::CopyCons, pmhd, MHD_CopyU, "MHD_CopyU", Task_Run,
                 copy_cons_deps);

  // Select which CalculateFlux function to add based on rsolver_method.
  // CalcFlux requires metric in flux - must happen before z4ctoadm updates the metric
  std::vector<TaskName> flux_deps = {MHD_CopyU};
  if (rsolver_method == DynGRMHD_RSolver::llf_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::llf_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, flux_deps);
  } else if (rsolver_method == DynGRMHD_RSolver::hlle_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, flux_deps);
  } else if (rsolver_method == DynGRMHD_RSolver::hlld_dyngr) {
    pnr->QueueTask(
           &DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes<DynGRMHD_RSolver::hlld_dyngr>,
           this, MHD_Flux, "MHD_Flux", Task_Run, flux_deps);
  } else { // put more rsolvers here
    abort();
  }

  // Now the rest of the MHD run tasks
  if (pz4c != nullptr) {
    pnr->QueueTask(&DynGRMHD::SetTmunu, this, MHD_SetTmunu, "MHD_SetTmunu",
                   Task_Run, {MHD_CopyU});
  }
  pnr->QueueTask(&MHD::SendFlux, pmhd, MHD_SendFlux, "MHD_SendFlux",
                 Task_Run, {MHD_Flux});
  pnr->QueueTask(&MHD::RecvFlux, pmhd, MHD_RecvFlux, "MHD_RecvFlux",
                 Task_Run, {MHD_SendFlux});
  std::vector<TaskName> explrk_deps = {MHD_RecvFlux};
  if (pz4c != nullptr) explrk_deps.push_back(MHD_SetTmunu);
  pnr->QueueTask(&MHD::RKUpdate, pmhd, MHD_ExplRK, "MHD_ExplRK", Task_Run, explrk_deps);
  // MHD::CornerE picks the upwind side of every edge from the mass flux uflx(IDN), which
  // MHD_RecvFlux replaces on coarse/fine faces by the restricted fine flux, so it waits
  // for that receive as well as for the face EMFs of MHD_Flux.
  std::vector<TaskName> efield_deps = {MHD_Flux, MHD_RecvFlux};
  auto queue_emf_chain = [&]() {
    pnr->QueueTask(&MHD::CornerE, pmhd, MHD_EField, "MHD_EField", Task_Run, efield_deps);
    pnr->QueueTask(&MHD::SendE, pmhd, MHD_SendE, "MHD_SendE", Task_Run, {MHD_EField});
    pnr->QueueTask(&MHD::RecvE, pmhd, MHD_RecvE, "MHD_RecvE", Task_Run, {MHD_SendE});
    pnr->QueueTask(&MHD::CT, pmhd, MHD_CT, "MHD_CT", Task_Run, {MHD_RecvE});
  };
  // Every recovery of the updated state waits for the metric of its time level
  // (DynGRMHD::SetADMVariablesAtStageEnd), installed once the EMF and the source terms
  // are done with the stage's own.
  const bool stage_end_metric = prescribed_dynamic_adm && padm->IsAnalyticBBH();
  auto queue_stage_end_metric = [&]() {
    if (!stage_end_metric) return;
    std::vector<TaskName> deps = {MHD_AddSrc, MHD_DualE, MHD_EField};
    pnr->QueueTask(&DynGRMHD::SetADMVariablesAtStageEnd, this, MHD_SetADMEnd,
                   "MHD_SetADMEnd", Task_Run, deps);
  };
  pnr->QueueTask(&MHD::MHDSrcTerms, pmhd, MHD_AddSrc, "MHD_AddSrc", Task_Run,
                 {MHD_ExplRK});
  // The dual-energy resynchronization reads the updated D and the advected D*kappa and
  // writes only the auxiliary column, so it runs beside the source terms and ahead of
  // the restriction and the send that carry its result to the coarse copies and the
  // neighbours.  A no-op when the auxiliary is not carried.
  pnr->QueueTask(&MHD::DualEnergyStep, pmhd, MHD_DualE, "MHD_DualE", Task_Run,
                 {MHD_ExplRK});
  pnr->QueueTask(&MHD::RestrictU, pmhd, MHD_RestU, "MHD_RestU", Task_Run,
                 {MHD_AddSrc, MHD_DualE});
  pnr->QueueTask(&MHD::SendU, pmhd, MHD_SendU, "MHD_SendU", Task_Run, {MHD_RestU});
  pnr->QueueTask(&MHD::RecvU, pmhd, MHD_RecvU, "MHD_RecvU", Task_Run, {MHD_SendU});
  // GPU/MPI OVERLAP (task-DAG relaxation).
  // The EMF half of the stage does not read the conserved-variable exchange, so it is not
  // chained behind it.
  // MHD::CornerE reads the face-centred EMFs e1x2/e1x3/e2x1/e2x3/e3x1/e3x2 (written by
  // MHD_Flux, FOFC included -- CalcFluxes calls FOFC), the mass flux uflx(IDN) that picks
  // the upwind side of each edge (written by MHD_Flux and, on coarse/fine faces, replaced
  // by the restricted fine flux in MHD_RecvFlux), the stage-initial primitives w0/bcc0
  // and the ADM metric (MHD_SetADM).  It never reads u0.  w0/bcc0 have exactly two
  // writers in a stage: MHD_PreFluxC2P, which precedes MHD_Flux, and -- with the sidecar
  // configured -- MHD_HybridPreC2P, which is why that task carries an MHD_CT dependency
  // and CornerE provably runs first.  The tasks between MHD_RecvFlux and MHD_RecvU write
  // only u0/coarse_u0 (RKUpdate: mhd_update.cpp; AddCoordTerms:
  // dyn_grmhd_coord_terms_impl.hpp AddCoordTermsEOSImpl writes `rhs` only; RestrictU/
  // SendU/RecvU:
  // u0/u1/coarse_u0/coarse_u1 via pbval_u).  So CornerE waits for MHD_Flux and
  // MHD_RecvFlux (efield_deps) and not for MHD_RecvU: the EMF exchange
  // (pbval_b->comm_flux) and the face-field exchange (pbval_b->comm_vars) run while the
  // conserved-variable exchange (pbval_u->comm_vars) is in flight.  All four
  // communicators are separate MPI_Comm_dup handles (bvals.cpp) with disjoint buffers, so
  // the two chains are safe to have in flight together.
  queue_emf_chain();
  queue_stage_end_metric();
  pnr->QueueTask(&MHD::RestrictB, pmhd, MHD_RestB, "MHD_RestB", Task_Run, {MHD_CT});
  pnr->QueueTask(&MHD::SendB, pmhd, MHD_SendB, "MHD_SendB", Task_Run, {MHD_RestB});
  // INTERIOR-FIRST C2P.  Queued here, between the last pack of the stage and the first
  // MPI wait, deliberately: TaskList::DoAvailable walks the list in order, so the deep-
  // interior recovery is issued right after MHD_SendB's pack and before MHD_RecvB starts
  // polling, and it runs on the GPU while the messages are in flight.  Dependencies:
  // MHD_SendU because the u0 pack and the coarse restriction must have read the
  // pre-recovery u0 first, MHD_SendB because the active-cell face fields it reads are
  // only final after MHD_CT.  MHD_Prolong is not delayed: it still depends on
  // {MHD_RecvB, MHD_RecvU} only.
  //
  // The MHD_SendU edge means "the pack has finished reading u0" only because
  // PackAndSendCC fences its pack, on the default execution-space instance, before
  // MPI_Startall.  Moving that pack to its own stream with an event would turn this edge
  // into a race against the early pass's u0 write-back and must convert it into an
  // explicit wait on the pack's completion event first.
  if (interior_first_c2p) {
    std::vector<TaskName> c2p_int_deps = {MHD_SendU, MHD_SendB};
    if (stage_end_metric) c2p_int_deps.push_back(MHD_SetADMEnd);
    pnr->QueueTask(&DynGRMHD::ConToPrimInteriorFirst, this, MHD_C2P_Int, "MHD_C2P_Int",
                   Task_Run, c2p_int_deps);
  }
  pnr->QueueTask(&MHD::RecvB, pmhd, MHD_RecvB, "MHD_RecvB", Task_Run, {MHD_SendB});
  // MHD_RecvU is now an EXPLICIT dependency: with MHD_EField moved off MHD_RecvU above,
  // the conserved-variable exchange is no longer an ancestor of MHD_RecvB, and
  // MHD::Prolongate consumes both halves -- FillCoarseInBndryCC(u0,coarse_u0) plus
  // FillCoarseInBndryFC(b0,coarse_b0), and with prolong_prims also
  // ConsToPrimCoarseBndry(coarse_u0,coarse_b0,coarse_w0) (mhd_tasks.cpp:1885-1907).  This
  // is the join of the two chains; everything after it (BCS, C2P, Newdt) inherits it.
  std::vector<TaskName> prolong_deps = {MHD_RecvB, MHD_RecvU};
  if (stage_end_metric) prolong_deps.push_back(MHD_SetADMEnd);
  // MHD_C2P keeps its name, its DAG slot and every downstream dependent; with the split
  // on it recovers the complement of the deep-interior box instead of the full extent.
  auto queue_ordinary_c2p = [&]() {
    std::vector<TaskName> c2p_deps = {MHD_BCS};
    if (interior_first_c2p) c2p_deps.push_back(MHD_C2P_Int);
    if (interior_first_c2p) {
      pnr->QueueTask(&DynGRMHD::ConToPrimAfterExchange, this, MHD_C2P,
                     "MHD_C2P", Task_Run, c2p_deps, {Z4c_Excise});
    } else {
      pnr->QueueTask(&DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim, this, MHD_C2P,
                     "MHD_C2P", Task_Run, c2p_deps, {Z4c_Excise});
    }
  };
  pnr->QueueTask(&MHD::Prolongate, pmhd, MHD_Prolong, "MHD_Prolong", Task_Run,
                 prolong_deps);
  pnr->QueueTask(&MHD::ApplyPhysicalBCs, pmhd, MHD_BCS, "MHD_BCS", Task_Run,
                 {MHD_Prolong});
  //pnr->QueueTask(&DynGRMHD::ApplyPhysicalBCs, this, MHD_BCS, "MHD_BCS", Task_Run,
  //                 {MHD_RecvB});
  queue_ordinary_c2p();
  pnr->QueueTask(&MHD::NewTimeStep, pmhd, MHD_Newdt, "MHD_Newdt", Task_Run,
                 {MHD_C2P});

  // End task list
  pnr->QueueTask(&MHD::ClearSend, pmhd, MHD_ClearS, "MHD_ClearS", Task_End);
  pnr->QueueTask(&MHD::ClearRecv, pmhd, MHD_ClearR, "MHD_ClearR",
                 Task_End, {MHD_ClearS});
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::PrimToConInit(int is, int ie, int js, int je,
                                                    int ks, int ke) {
  eos.PrimToCons(pmy_pack->pmhd->w0, pmy_pack->pmhd->bcc0, pmy_pack->pmhd->u0,
                 is, ie, js, je, ks, ke);
  if (pmy_pack->ptmunu != nullptr) {
    bool fixed = fixed_evolution;
    fixed_evolution = false;
    SetTmunu(nullptr, 0);
    fixed_evolution = fixed;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRMHD::ConvertInternalEnergyToPressure
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConvertInternalEnergyToPressure(int is, int ie,
    int js, int je, int ks, int ke) {
  int nmb = pmy_pack->nmb_thispack;
  auto &prim = pmy_pack->pmhd->w0;
  auto &eos_ = eos.ps.GetEOS();
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos_.GetBaryonMass();

  par_for("coord_src", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real n = prim(m, IDN, k, j, i) / mb;
    Real egas = mb*n + prim(m, IEN, k, j, i);
    Real Y[MAX_SPECIES] = {0.};
    for (int s = 0; s < nscal; s++) {
      Y[s] = prim(m, nmhd + s, k, j, i);
    }
    Real T;
    // Note that this is done explicitly rather than with a flooring policy because we
    // don't have the temperature yet, and it's probable that the energy is bunk if the
    // density is. There may be a cleaner way to do this elsewhere.
    if (n < eos_.GetMinimumDensity()) {
      n = eos_.GetMinimumDensity();
      T = eos_.GetMinimumTemperature();
    } else {
      T = eos_.GetTemperatureFromE(n, egas, Y);
    }
    prim(m, IPR, k, j, i) = eos_.GetPressure(n, T, Y);
  });
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrim(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }

  // The dual-energy adiabat this inversion may read was resynchronized by
  // MHD::DualEnergyStep ahead of the conserved send, so every recovery -- this full
  // pass, the interior-first split and the ghost bands -- sees the column the
  // neighbours were sent, and none of them needs a pass of its own.

  // Extract the indices
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  if (store_temperature) {
    eos.template ConsToPrim<true>(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0,
                                  pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                                  temperature, 0, n1m1, 0, n2m1, 0, n3m1, false);
  } else {
    eos.template ConsToPrim<false>(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0,
                                   pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                                   temperature, 0, n1m1, 0, n2m1, 0, n3m1, false);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHDPS::ConToPrimBC(int is, int ie, int js, int je, int ks, int ke)
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::ConToPrimBC(int is, int ie, int js, int je,
                                                int ks, int ke) {
  if (fixed_evolution) {
    return;
  }
  if (store_temperature) {
    eos.template ConsToPrim<true>(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0,
                                  pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                                  temperature, is, ie, js, je, ks, ke, false);
  } else {
    eos.template ConsToPrim<false>(pmy_pack->pmhd->u0, pmy_pack->pmhd->b0,
                                   pmy_pack->pmhd->bcc0, pmy_pack->pmhd->w0,
                                   temperature, is, ie, js, je, ks, ke, false);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ADMMatterSource_(Driver *pdrive, int stage) {
//  \brief
template<class EOSPolicy, class ErrorPolicy>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTerms(const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs, int nghost) {
  switch (nghost) {
    case 2: AddCoordTermsEOS<2>(prim, bcc, dt, rhs);
            break;
    case 3: AddCoordTermsEOS<3>(prim, bcc, dt, rhs);
            break;
    case 4: AddCoordTermsEOS<4>(prim, bcc, dt, rhs);
            break;
  }
}

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_PS_IMPL_HPP_
