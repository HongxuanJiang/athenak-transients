//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//! \brief Performs explicit update of Hydro conserved variables (u0) for each stage of
//! the SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted
//! average and partial time step update of flux divergence. Source terms are added in
//! the HydroSrcTerms() function.

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "hydro.hpp"
#include "gravity/gravity.hpp"
#include "pgen/pgen.hpp"
#include "pgen/bh_force_pair.hpp"
#include "sink_particles/sink_particles.hpp"
#include "srcterms/srcterms.hpp"
#include "driver/lat_weights.hpp"
#include "utils/lat_reflux_limiter.hpp"

// LAT flux-bookkeeping helpers now live in driver/lat_weights.hpp (review A2/F19).

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus Hydro::RKUpdate(Driver *pdriver, int stage) {
  // Multilevel LAT can replace coarse stage fluxes in RecvFlux. Unmasked fallback steps
  // use the same receive-before-update ordering.
  if (lat_requested_ &&
      (!(pmy_pack->lat_active_mask_enabled) || pmy_pack->pmesh->multilevel) &&
      !flux_recv_complete_) {
    return TaskStatus::incomplete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  const Real beta_stage = pdriver->beta[stage-1];
  const Real beta_dt = beta_stage*pmy_pack->pmesh->dt;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nvars;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = FluxBand(uflx.x1f);
  auto flx2 = FluxBand(uflx.x2f);
  auto flx3 = FluxBand(uflx.x3f);
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return TaskStatus::complete;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Vector inner loop used for good performance on cpus
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("h_update",DevExeSpace(),scr_size,scr_level,0,nwork1,0,nvar-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int a, const int n, const int k, const int j) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_beta_dt = lat_per_block_dt ? beta_stage*lat_step_dt(m) : beta_dt;
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    // compute dF1/dx1
    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    });
    member.team_barrier();

    // Add dF2/dx2
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
      });
      member.team_barrier();
    }

    // Add dF3/dx3
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) =
          gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - block_beta_dt*divf(i);
    });
  });
  return TaskStatus::complete;
}

void Hydro::ResetLATFluxCorrection() {
  const bool same_level_lat = pmy_pack->pmesh->hydro_lat_same_level;
  if (!(pmy_pack->pmesh->multilevel) && !same_level_lat) return;

  // The gravity gate record is a companion of the mass mismatch and is discarded with it.
  if (lat_grav_reflux_allocated) {
    Kokkos::deep_copy(lat_grav_reflux.x1f, 0.0);
    Kokkos::deep_copy(lat_grav_reflux.x2f, 0.0);
    Kokkos::deep_copy(lat_grav_reflux.x3f, 0.0);
  }

  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar = nvars;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto rbuf = pbval_u->recvbuf_device;
  auto acc1 = lat_reflux.x1f;
  auto acc2 = lat_reflux.x2f;
  auto acc3 = lat_reflux.x3f;
  auto vfacc1 = lat_dual_vf_reflux.x1f;
  auto vfacc2 = lat_dual_vf_reflux.x2f;
  auto vfacc3 = lat_dual_vf_reflux.x3f;
  const bool dual_enabled = dual_energy_pdv;

  const int nflux_nghbr = lat::FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nflux_nghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("lat_reflux_reset", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - m*(nflux_nghbr*nvar))/nvar;
    const int nb = lat::FluxFaceNeighborIndex(slot, nnghbr);
    const int v = (tmember.league_rank() - m*(nflux_nghbr*nvar) - slot*nvar);
    if (!(nghbr.d_view(m,nb).gid >= 0 &&
          (nghbr.d_view(m,nb).lev > mblev.d_view(m) ||
           (same_level_lat && nghbr.d_view(m,nb).lev == mblev.d_view(m))))) return;

    const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
    const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
    int il = iflux.bis;
    int iu = iflux.bie;
    int jl = iflux.bjs;
    int ju = iflux.bje;
    int kl = iflux.bks;
    int ku = iflux.bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji = nj*ni;
    const int nkj = nk*nj;
    const int nki = nk*ni;

    if (nb < 8) {
      const int ia = (il == is) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        acc1(m,v,k,j,ia) = 0.0;
        if (dual_enabled && v == 0) vfacc1(m,0,k,j,ia) = 0.0;
      });
    } else if (nb < 16) {
      const int ja = (jl == js) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        acc2(m,v,k,ja,i) = 0.0;
        if (dual_enabled && v == 0) vfacc2(m,0,k,ja,i) = 0.0;
      });
    } else if ((nb >= 24) && (nb < 32)) {
      const int ka = (kl == ks) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        acc3(m,v,ka,j,i) = 0.0;
        if (dual_enabled && v == 0) vfacc3(m,0,ka,j,i) = 0.0;
      });
    }
    tmember.team_barrier();
  });
}

void Hydro::AccumulateLATCoarseFluxes(Driver *pdriver, int stage) {
  if (!(pmy_pack->lat_active_mask_enabled)) return;
  if (!(pmy_pack->pmesh->multilevel) && !(pmy_pack->pmesh->hydro_lat_same_level)) return;
  const int nactive = pmy_pack->lat_nactive_thispack;
  if (nactive <= 0) return;
  const Real flux_weight = lat::FinalFluxWeight(pdriver, stage);
  if (flux_weight == 0.0) return;

  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = nvars;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  auto lat_flux_accum = pmy_pack->lat_flux_accum_nghbr.d_view;
  auto lat_step_factor = pmy_pack->lat_step_factor.d_view;
  auto lat_nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  auto rbuf = pbval_u->recvbuf_device;
  auto flx1 = FluxBand(uflx.x1f);
  auto flx2 = FluxBand(uflx.x2f);
  auto flx3 = FluxBand(uflx.x3f);
  auto acc1 = lat_reflux.x1f;
  auto acc2 = lat_reflux.x2f;
  auto acc3 = lat_reflux.x3f;
  auto vfacc1 = lat_dual_vf_reflux.x1f;
  auto vfacc2 = lat_dual_vf_reflux.x2f;
  auto vfacc3 = lat_dual_vf_reflux.x3f;
  auto vf1 = FluxBand(dual_vf.x1f);
  auto vf2 = FluxBand(dual_vf.x2f);
  auto vf3 = FluxBand(dual_vf.x3f);
  const bool dual_enabled = dual_energy_pdv;

  const int nflux_nghbr = lat::FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nactive*nflux_nghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("lat_coarse_flux", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = (tmember.league_rank())/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - a*(nflux_nghbr*nvar))/nvar;
    const int nb = lat::FluxFaceNeighborIndex(slot, nnghbr);
    const int v = (tmember.league_rank() - a*(nflux_nghbr*nvar) - slot*nvar);
    const int m = active_indices(a);
    if (lat_flux_accum(m,nb) == 0) return;
    if (nghbr.d_view(m,nb).lev > mblev.d_view(m) &&
        lat_step_factor(m) == lat_nghbr_factor(m,nb)) return;
    const Real scale = flux_weight*(lat_per_block_dt ? lat_step_dt(m) : mesh_dt);

    const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
    const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
    int il = iflux.bis;
    int iu = iflux.bie;
    int jl = iflux.bjs;
    int ju = iflux.bje;
    int kl = iflux.bks;
    int ku = iflux.bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji = nj*ni;
    const int nkj = nk*nj;
    const int nki = nk*ni;

    if (nb < 8) {
      const int ia = (il == is) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        acc1(m,v,k,j,ia) += scale*flx1(m,v,k,j,il);
        if (dual_enabled && v == 0) vfacc1(m,0,k,j,ia) += scale*vf1(m,0,k,j,il);
      });
    } else if (nb < 16) {
      const int ja = (jl == js) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        acc2(m,v,k,ja,i) += scale*flx2(m,v,k,jl,i);
        if (dual_enabled && v == 0) vfacc2(m,0,k,ja,i) += scale*vf2(m,0,k,jl,i);
      });
    } else if ((nb >= 24) && (nb < 32)) {
      const int ka = (kl == ks) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        acc3(m,v,ka,j,i) += scale*flx3(m,v,kl,j,i);
        if (dual_enabled && v == 0) vfacc3(m,0,ka,j,i) += scale*vf3(m,0,kl,j,i);
      });
    }
    tmember.team_barrier();
  });
}

bool Hydro::ApplyLATFluxCorrection(Real sync_time, int sync_factor, int completed_tick) {
  const bool same_level_lat = pmy_pack->pmesh->hydro_lat_same_level;
  if (!(pmy_pack->pmesh->multilevel) && !same_level_lat) return false;

  const int max_factor = std::max(1, sync_factor);

  const int nrefluxed =
      pmy_pack->SetLATFluxCorrectionByCompletionPhase(max_factor, completed_tick);
  if (nrefluxed <= 0) {
    pmy_pack->lat_active_mask_enabled = false;
    pmy_pack->lat_nactive_thispack = 0;
    pmy_pack->lat_nflux_recv_thispack = 0;
    return false;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = nvars;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto rbuf = pbval_u->recvbuf_device;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto lat_flux_recv = pmy_pack->lat_flux_recv_nghbr.d_view;
  auto u = u0;
  auto acc1 = lat_reflux.x1f;
  auto acc2 = lat_reflux.x2f;
  auto acc3 = lat_reflux.x3f;
  const bool external_bh_energy_correction =
      (psrc != nullptr) && psrc->external_bh_gravity && peos->eos_data.use_e &&
      problem_runtime::ExternalBHGravitySourceCouplingEnabled();
  const bool self_gravity_energy_correction =
      (psrc != nullptr) && psrc->self_gravity && peos->eos_data.use_e &&
      pmy_pack->pgrav != nullptr && pmy_pack->pgrav->phi_valid &&
      pmy_pack->pgrav->self_phi_time_valid;
  DvceArray5D<Real> self_phi;
  if (self_gravity_energy_correction) {
    self_phi = pmy_pack->pgrav->phi;
  }
  bool external_bh_enabled = false;
  Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
  Real bh_mass = 0.0, bh_softening = 0.0, newton_g = 0.0;
  if (external_bh_energy_correction) {
    problem_runtime::GetExternalBHPotential(sync_time, external_bh_enabled,
                                            bhx, bhy, bhz,
                                            bh_mass, bh_softening, newton_g);
  }
  bool bh_sink_mask_enabled = false;
  Real bh_sink_mask_radius = 0.0;
  Real bh_sink_mask_x = 0.0;
  Real bh_sink_mask_y = 0.0;
  Real bh_sink_mask_z = 0.0;
  if (external_bh_energy_correction) {
    problem_runtime::GetBHSinkGravityMask(sync_time, bh_sink_mask_enabled,
                                          bh_sink_mask_radius,
                                          bh_sink_mask_x, bh_sink_mask_y,
                                          bh_sink_mask_z);
  }
  // The work is owed for the mass the source-term gate was allowed to act on when each
  // stage flux was taken, which SourceTerms::Gravity recorded face by face, and not for
  // whatever the post-step density of the face cell now happens to be.  Refunding the
  // admitted part alone also removes the density test from these kernels: the refund is
  // now a pure addition and no longer depends on the mass the reflux below is about to
  // take away.
  const bool grav_gate_recorded = lat_grav_reflux_allocated;
  const bool grav_self_refund = self_gravity_energy_correction && grav_gate_recorded;
  const bool grav_bh_refund =
      external_bh_energy_correction && external_bh_enabled && grav_gate_recorded;
  const bool bh_adds_to_applied = grav_self_refund;
  // Sink particles do the same Mullen-Hanawa-Gammie work with their own point-mass
  // potential (SourceTerms::SinkGravity), gated only on rho > 0, which the density floor
  // keeps open, so the whole mass mismatch is owed.  The list mirror sink_gm_pos is
  // republished only by the driver's window-end SinkStep, after every correction of the
  // window, so the potential differenced here is the one every stage of the window used.
  sinkparticles::SinkParticles *psink = pmy_pack->psink;
  const bool grav_sink_refund =
      (psrc != nullptr) && psrc->sink_gravity && peos->eos_data.use_e &&
      (psink != nullptr) && (psink->nsinks > 0) && grav_gate_recorded;
  const bool sink_adds_to_applied = grav_self_refund || grav_bh_refund;
  const bool grav_work_applied = grav_self_refund || grav_bh_refund || grav_sink_refund;
  const int nsinks = grav_sink_refund ? psink->nsinks : 0;
  DvceArray2D<Real> sink;
  if (grav_sink_refund) {
    sink = psink->sink_gm_pos.d_view;
  }
  // Mesh extents and periodicity exactly as SourceTerms::SinkGravity builds them.
  const RegionSize &msize = pmy_pack->pmesh->mesh_size;
  const Real slx1 = msize.x1max - msize.x1min;
  const Real slx2 = msize.x2max - msize.x2min;
  const Real slx3 = msize.x3max - msize.x3min;
  const BoundaryFlag *mbcs = pmy_pack->pmesh->mesh_bcs;
  const bool sper1 = (mbcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
  const bool sper2 = pmy_pack->pmesh->multi_d &&
      (mbcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
  const bool sper3 = pmy_pack->pmesh->three_d &&
      (mbcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  const int cseen = lat_reflux::kGravWorkSeen;
  const int cself = lat_reflux::kGravWorkBlockedSelf;
  const int cbh = lat_reflux::kGravWorkBlockedBH;
  const int capp = lat_reflux::kGravWorkApplied;
  auto gacc1 = lat_grav_reflux.x1f;
  auto gacc2 = lat_grav_reflux.x2f;
  auto gacc3 = lat_grav_reflux.x3f;
  DevExeSpace reflux_exec;
  const int nface = 8;
  const int nface1 = std::min(nface, nnghbr);
  if (grav_self_refund && nface1 > 0) {
    Kokkos::TeamPolicy<> policy_sg_x1(reflux_exec, nrefluxed*nface1, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_selfgrav_reflux_x1", policy_sg_x1,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface1;
      const int nb = tmember.league_rank() - a*nface1;
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int ju = iflux.bje;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj = nk*nj;

      const bool inner = (il == is);
      const int ic = inner ? is : ie;
      const int in = inner ? (ic - 1) : (ic + 1);
      const int ia = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        Real dphi = inner ? -(self_phi(m,0,k,j,ic) - self_phi(m,0,k,j,in))
                          : -(self_phi(m,0,k,j,in) - self_phi(m,0,k,j,ic));
        const Real dm = acc1(m,IDN,k,j,ia) - gacc1(m,cself,k,j,ia);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx1;
        u(m,IEN,k,j,ic) += work;
        gacc1(m,capp,k,j,ia) = work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_bh_refund && nface1 > 0) {
    Kokkos::TeamPolicy<> policy_bh_x1(reflux_exec, nrefluxed*nface1, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_bh_reflux_x1", policy_bh_x1,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface1;
      const int nb = tmember.league_rank() - a*nface1;
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int ju = iflux.bje;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj = nk*nj;

      const bool inner = (il == is);
      const int ic = inner ? is : ie;
      const int ia = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        const Real x = CellCenterX(ic - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real xn = CellCenterX((inner ? ic - 1 : ic + 1) - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(j - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(k - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = problem_runtime::ExternalBHPotentialSinkMasked(
            x, y, z, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real phi_n = problem_runtime::ExternalBHPotentialSinkMasked(
            xn, y, z, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc1(m,IDN,k,j,ia) - gacc1(m,cbh,k,j,ia);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx1;
        u(m,IEN,k,j,ic) += work;
        gacc1(m,capp,k,j,ia) =
            (bh_adds_to_applied ? gacc1(m,capp,k,j,ia) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_sink_refund && nface1 > 0) {
    Kokkos::TeamPolicy<> policy_sk_x1(reflux_exec, nrefluxed*nface1, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_sink_reflux_x1", policy_sk_x1,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface1;
      const int nb = tmember.league_rank() - a*nface1;
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int ju = iflux.bje;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj = nk*nj;

      const bool inner = (il == is);
      const int ic = inner ? is : ie;
      const int ia = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        const Real x = CellCenterX(ic - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real xn = CellCenterX((inner ? ic - 1 : ic + 1) - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(j - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(k - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = sinkparticles::SinkPotentialSum(sink, nsinks, x, y, z,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real phi_n = sinkparticles::SinkPotentialSum(sink, nsinks, xn, y, z,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc1(m,IDN,k,j,ia);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx1;
        u(m,IEN,k,j,ic) += work;
        gacc1(m,capp,k,j,ia) =
            (sink_adds_to_applied ? gacc1(m,capp,k,j,ia) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  const int nface2 = (nnghbr > 8) ? std::min(nface, nnghbr - 8) : 0;
  if (grav_self_refund && nface2 > 0) {
    Kokkos::TeamPolicy<> policy_sg_x2(reflux_exec, nrefluxed*nface2, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_selfgrav_reflux_x2", policy_sg_x2,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface2;
      const int nb = 8 + (tmember.league_rank() - a*nface2);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int iu = iflux.bie;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int ni = iu - il + 1;
      const int nk = ku - kl + 1;
      const int nki = nk*ni;

      const bool inner = (jl == js);
      const int jc = inner ? js : je;
      const int jn = inner ? (jc - 1) : (jc + 1);
      const int ja = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        Real dphi = inner ? -(self_phi(m,0,k,jc,i) - self_phi(m,0,k,jn,i))
                          : -(self_phi(m,0,k,jn,i) - self_phi(m,0,k,jc,i));
        const Real dm = acc2(m,IDN,k,ja,i) - gacc2(m,cself,k,ja,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx2;
        u(m,IEN,k,jc,i) += work;
        gacc2(m,capp,k,ja,i) = work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_bh_refund && nface2 > 0) {
    Kokkos::TeamPolicy<> policy_bh_x2(reflux_exec, nrefluxed*nface2, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_bh_reflux_x2", policy_bh_x2,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface2;
      const int nb = 8 + (tmember.league_rank() - a*nface2);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int iu = iflux.bie;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int ni = iu - il + 1;
      const int nk = ku - kl + 1;
      const int nki = nk*ni;

      const bool inner = (jl == js);
      const int jc = inner ? js : je;
      const int ja = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        const Real x = CellCenterX(i - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(jc - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real yn = CellCenterX((inner ? jc - 1 : jc + 1) - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(k - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = problem_runtime::ExternalBHPotentialSinkMasked(
            x, y, z, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real phi_n = problem_runtime::ExternalBHPotentialSinkMasked(
            x, yn, z, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc2(m,IDN,k,ja,i) - gacc2(m,cbh,k,ja,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx2;
        u(m,IEN,k,jc,i) += work;
        gacc2(m,capp,k,ja,i) =
            (bh_adds_to_applied ? gacc2(m,capp,k,ja,i) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_sink_refund && nface2 > 0) {
    Kokkos::TeamPolicy<> policy_sk_x2(reflux_exec, nrefluxed*nface2, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_sink_reflux_x2", policy_sk_x2,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface2;
      const int nb = 8 + (tmember.league_rank() - a*nface2);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int iu = iflux.bie;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ku = iflux.bke;
      const int ni = iu - il + 1;
      const int nk = ku - kl + 1;
      const int nki = nk*ni;

      const bool inner = (jl == js);
      const int jc = inner ? js : je;
      const int ja = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        const Real x = CellCenterX(i - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(jc - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real yn = CellCenterX((inner ? jc - 1 : jc + 1) - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(k - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = sinkparticles::SinkPotentialSum(sink, nsinks, x, y, z,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real phi_n = sinkparticles::SinkPotentialSum(sink, nsinks, x, yn, z,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc2(m,IDN,k,ja,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx2;
        u(m,IEN,k,jc,i) += work;
        gacc2(m,capp,k,ja,i) =
            (sink_adds_to_applied ? gacc2(m,capp,k,ja,i) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  const int nface3 = (nnghbr > 24) ? std::min(nface, nnghbr - 24) : 0;
  if (grav_self_refund && nface3 > 0) {
    Kokkos::TeamPolicy<> policy_sg_x3(reflux_exec, nrefluxed*nface3, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_selfgrav_reflux_x3", policy_sg_x3,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface3;
      const int nb = 24 + (tmember.league_rank() - a*nface3);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ni = iflux.bie - il + 1;
      const int nj = iflux.bje - jl + 1;
      const int nji = nj*ni;

      const bool inner = (kl == ks);
      const int kc = inner ? ks : ke;
      const int kn = inner ? (kc - 1) : (kc + 1);
      const int ka = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        Real dphi = inner ? -(self_phi(m,0,kc,j,i) - self_phi(m,0,kn,j,i))
                          : -(self_phi(m,0,kn,j,i) - self_phi(m,0,kc,j,i));
        const Real dm = acc3(m,IDN,ka,j,i) - gacc3(m,cself,ka,j,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx3;
        u(m,IEN,kc,j,i) += work;
        gacc3(m,capp,ka,j,i) = work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_bh_refund && nface3 > 0) {
    Kokkos::TeamPolicy<> policy_bh_x3(reflux_exec, nrefluxed*nface3, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_bh_reflux_x3", policy_bh_x3,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface3;
      const int nb = 24 + (tmember.league_rank() - a*nface3);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ni = iflux.bie - il + 1;
      const int nj = iflux.bje - jl + 1;
      const int nji = nj*ni;

      const bool inner = (kl == ks);
      const int kc = inner ? ks : ke;
      const int ka = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        const Real x = CellCenterX(i - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(j - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(kc - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real zn = CellCenterX((inner ? kc - 1 : kc + 1) - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = problem_runtime::ExternalBHPotentialSinkMasked(
            x, y, z, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real phi_n = problem_runtime::ExternalBHPotentialSinkMasked(
            x, y, zn, bhx, bhy, bhz, bh_mass, bh_softening, newton_g,
            bh_sink_mask_radius);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc3(m,IDN,ka,j,i) - gacc3(m,cbh,ka,j,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx3;
        u(m,IEN,kc,j,i) += work;
        gacc3(m,capp,ka,j,i) =
            (bh_adds_to_applied ? gacc3(m,capp,ka,j,i) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  if (grav_sink_refund && nface3 > 0) {
    Kokkos::TeamPolicy<> policy_sk_x3(reflux_exec, nrefluxed*nface3, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_sink_reflux_x3", policy_sk_x3,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface3;
      const int nb = 24 + (tmember.league_rank() - a*nface3);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      const int il = iflux.bis;
      const int jl = iflux.bjs;
      const int kl = iflux.bks;
      const int ni = iflux.bie - il + 1;
      const int nj = iflux.bje - jl + 1;
      const int nji = nj*ni;

      const bool inner = (kl == ks);
      const int kc = inner ? ks : ke;
      const int ka = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        const Real x = CellCenterX(i - is, indcs.nx1,
            mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        const Real y = CellCenterX(j - js, indcs.nx2,
            mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real z = CellCenterX(kc - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real zn = CellCenterX((inner ? kc - 1 : kc + 1) - ks, indcs.nx3,
            mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
        const Real phi_c = sinkparticles::SinkPotentialSum(sink, nsinks, x, y, z,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real phi_n = sinkparticles::SinkPotentialSum(sink, nsinks, x, y, zn,
            slx1, slx2, slx3, sper1, sper2, sper3);
        const Real dphi = 0.0 - (inner ? (phi_c - phi_n) : (phi_n - phi_c));
        const Real dm = acc3(m,IDN,ka,j,i);
        const Real work = -0.5*dm*dphi/mbsize.d_view(m).dx3;
        u(m,IEN,kc,j,i) += work;
        gacc3(m,capp,ka,j,i) =
            (sink_adds_to_applied ? gacc3(m,capp,ka,j,i) : 0.0) + work;
      });
      tmember.team_barrier();
    });
  }

  // A coarse cell the atmosphere floor reseeded partway through the window no longer
  // holds what the fine neighbor's flux history moved through this face.  Scale the whole
  // pending mismatch by the largest fraction the cell can still invert
  // (src/utils/lat_reflux_limiter.hpp) and hand the rest to the interior cell behind the
  // face, so the pair's sum is the unlimited correction.  An admissible cell takes the
  // fraction exactly 1.0 and is left bit-for-bit alone.  An excised cell is the horizon's
  // sink, which the next recovery resets: it is not tested, takes whatever reaches it
  // (fraction 1.0) and hands nothing on to the live gas behind it.
  const bool reflux_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto reflux_excised = pmy_pack->pcoord->excision_floor;
  auto &reflux_eos = peos->eos_data;
  const lat_reflux::EnergyForm reflux_form = lat_reflux::EnergyFormOf(
      pmy_pack->pcoord->is_dynamical_relativistic,
      pmy_pack->pcoord->is_general_relativistic,
      pmy_pack->pcoord->is_special_relativistic);
  const int nadm = (nvar > IEN) ? (IEN + 1) : nvar;
  auto theta1 = lat_reflux_theta.x1f;
  auto theta2 = lat_reflux_theta.x2f;
  auto theta3 = lat_reflux_theta.x3f;

  if (nface1 > 0) {
    Kokkos::TeamPolicy<> policy_x1(reflux_exec, nrefluxed*nface1, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_reflux_x1", policy_x1,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface1;
      const int nb = tmember.league_rank() - a*nface1;
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      int il = iflux.bis;
      int jl = iflux.bjs;
      int ju = iflux.bje;
      int kl = iflux.bks;
      int ku = iflux.bke;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkj = nk*nj;

      const bool inner = (il == is);
      const int ic = inner ? is : ie;
      const int ip = inner ? (ic + 1) : (ic - 1);
      const Real sgn = inner ? -1.0 : 1.0;
      const int ia = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k*nj) + jl;
        k += kl;
        Real cons[IEN+1], dcons[IEN+1];
        for (int n=0; n<nadm; ++n) {
          cons[n] = u(m,n,k,j,ic);
          dcons[n] = sgn*acc1(m,n,k,j,ia)/mbsize.d_view(m).dx1;
        }
        const Real theta = (reflux_excise && reflux_excised(m,k,j,ic)) ? 1.0 :
            lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, cons, dcons, nvar,
                                           0.0);
        theta1(m,0,k,j,ia) = theta;
        if (theta != 1.0 && grav_work_applied) {
          // The refund paid for the whole pending mismatch, but only theta of it is
          // moved into this cell, so the work has to be cut back by the same fraction.
          u(m,IEN,k,j,ic) -= (1.0 - theta)*gacc1(m,capp,k,j,ia);
        }
        if (grav_gate_recorded) {
          gacc1(m,cseen,k,j,ia) = 0.0;
          gacc1(m,cself,k,j,ia) = 0.0;
          gacc1(m,cbh,k,j,ia) = 0.0;
          gacc1(m,capp,k,j,ia) = 0.0;
        }
        if (theta == 1.0) {
          for (int v=0; v<nvar; ++v) {
            u(m,v,k,j,ic) += sgn*acc1(m,v,k,j,ia)/mbsize.d_view(m).dx1;
            acc1(m,v,k,j,ia) = 0.0;
          }
        } else {
          Real pcons[IEN+1], prem[IEN+1];
          for (int n=0; n<nadm; ++n) {
            pcons[n] = u(m,n,k,j,ip);
            prem[n] = dcons[n] - theta*dcons[n];
          }
          const Real ptheta = (reflux_excise && reflux_excised(m,k,j,ip)) ? 1.0 :
              lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, pcons, prem, nvar,
                                             0.0);
          for (int v=0; v<nvar; ++v) {
            const Real correction = sgn*acc1(m,v,k,j,ia)/mbsize.d_view(m).dx1;
            const Real applied = theta*correction;
            u(m,v,k,j,ic) += applied;
            u(m,v,k,j,ip) += ptheta*(correction - applied);
            acc1(m,v,k,j,ia) = 0.0;
          }
        }
      });
      tmember.team_barrier();
    });
  }

  if (nface2 > 0) {
    Kokkos::TeamPolicy<> policy_x2(reflux_exec, nrefluxed*nface2, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_reflux_x2", policy_x2,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface2;
      const int nb = 8 + (tmember.league_rank() - a*nface2);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      int il = iflux.bis;
      int iu = iflux.bie;
      int jl = iflux.bjs;
      int kl = iflux.bks;
      int ku = iflux.bke;
      const int ni = iu - il + 1;
      const int nk = ku - kl + 1;
      const int nki = nk*ni;

      const bool inner = (jl == js);
      const int jc = inner ? js : je;
      const int jp = inner ? (jc + 1) : (jc - 1);
      const Real sgn = inner ? -1.0 : 1.0;
      const int ja = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
        int k = idx / ni;
        int i = (idx - k*ni) + il;
        k += kl;
        Real cons[IEN+1], dcons[IEN+1];
        for (int n=0; n<nadm; ++n) {
          cons[n] = u(m,n,k,jc,i);
          dcons[n] = sgn*acc2(m,n,k,ja,i)/mbsize.d_view(m).dx2;
        }
        const Real theta = (reflux_excise && reflux_excised(m,k,jc,i)) ? 1.0 :
            lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, cons, dcons, nvar,
                                           0.0);
        theta2(m,0,k,ja,i) = theta;
        if (theta != 1.0 && grav_work_applied) {
          // The refund paid for the whole pending mismatch, but only theta of it is
          // moved into this cell, so the work has to be cut back by the same fraction.
          u(m,IEN,k,jc,i) -= (1.0 - theta)*gacc2(m,capp,k,ja,i);
        }
        if (grav_gate_recorded) {
          gacc2(m,cseen,k,ja,i) = 0.0;
          gacc2(m,cself,k,ja,i) = 0.0;
          gacc2(m,cbh,k,ja,i) = 0.0;
          gacc2(m,capp,k,ja,i) = 0.0;
        }
        if (theta == 1.0) {
          for (int v=0; v<nvar; ++v) {
            u(m,v,k,jc,i) += sgn*acc2(m,v,k,ja,i)/mbsize.d_view(m).dx2;
            acc2(m,v,k,ja,i) = 0.0;
          }
        } else {
          Real pcons[IEN+1], prem[IEN+1];
          for (int n=0; n<nadm; ++n) {
            pcons[n] = u(m,n,k,jp,i);
            prem[n] = dcons[n] - theta*dcons[n];
          }
          const Real ptheta = (reflux_excise && reflux_excised(m,k,jp,i)) ? 1.0 :
              lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, pcons, prem, nvar,
                                             0.0);
          for (int v=0; v<nvar; ++v) {
            const Real correction = sgn*acc2(m,v,k,ja,i)/mbsize.d_view(m).dx2;
            const Real applied = theta*correction;
            u(m,v,k,jc,i) += applied;
            u(m,v,k,jp,i) += ptheta*(correction - applied);
            acc2(m,v,k,ja,i) = 0.0;
          }
        }
      });
      tmember.team_barrier();
    });
  }

  if (nface3 > 0) {
    Kokkos::TeamPolicy<> policy_x3(reflux_exec, nrefluxed*nface3, Kokkos::AUTO);
    Kokkos::parallel_for("lat_apply_reflux_x3", policy_x3,
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int a = (tmember.league_rank())/nface3;
      const int nb = 24 + (tmember.league_rank() - a*nface3);
      const int m = active_indices(a);
      if (lat_flux_recv(m,nb) == 0) return;

      const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
      const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
      int il = iflux.bis;
      int jl = iflux.bjs;
      int kl = iflux.bks;
      const int ni = iflux.bie - il + 1;
      const int nj = iflux.bje - jl + 1;
      const int nji = nj*ni;

      const bool inner = (kl == ks);
      const int kc = inner ? ks : ke;
      const int kp = inner ? (kc + 1) : (kc - 1);
      const Real sgn = inner ? -1.0 : 1.0;
      const int ka = inner ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
        int j = idx / ni;
        int i = (idx - j*ni) + il;
        j += jl;
        Real cons[IEN+1], dcons[IEN+1];
        for (int n=0; n<nadm; ++n) {
          cons[n] = u(m,n,kc,j,i);
          dcons[n] = sgn*acc3(m,n,ka,j,i)/mbsize.d_view(m).dx3;
        }
        const Real theta = (reflux_excise && reflux_excised(m,kc,j,i)) ? 1.0 :
            lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, cons, dcons, nvar,
                                           0.0);
        theta3(m,0,ka,j,i) = theta;
        if (theta != 1.0 && grav_work_applied) {
          // The refund paid for the whole pending mismatch, but only theta of it is
          // moved into this cell, so the work has to be cut back by the same fraction.
          u(m,IEN,kc,j,i) -= (1.0 - theta)*gacc3(m,capp,ka,j,i);
        }
        if (grav_gate_recorded) {
          gacc3(m,cseen,ka,j,i) = 0.0;
          gacc3(m,cself,ka,j,i) = 0.0;
          gacc3(m,cbh,ka,j,i) = 0.0;
          gacc3(m,capp,ka,j,i) = 0.0;
        }
        if (theta == 1.0) {
          for (int v=0; v<nvar; ++v) {
            u(m,v,kc,j,i) += sgn*acc3(m,v,ka,j,i)/mbsize.d_view(m).dx3;
            acc3(m,v,ka,j,i) = 0.0;
          }
        } else {
          Real pcons[IEN+1], prem[IEN+1];
          for (int n=0; n<nadm; ++n) {
            pcons[n] = u(m,n,kp,j,i);
            prem[n] = dcons[n] - theta*dcons[n];
          }
          const Real ptheta = (reflux_excise && reflux_excised(m,kp,j,i)) ? 1.0 :
              lat_reflux::AdmissibleFraction(reflux_eos, reflux_form, pcons, prem, nvar,
                                             0.0);
          for (int v=0; v<nvar; ++v) {
            const Real correction = sgn*acc3(m,v,ka,j,i)/mbsize.d_view(m).dx3;
            const Real applied = theta*correction;
            u(m,v,kc,j,i) += applied;
            u(m,v,kp,j,i) += ptheta*(correction - applied);
            acc3(m,v,ka,j,i) = 0.0;
          }
        }
      });
      tmember.team_barrier();
    });
  }

  if (dual_energy_pdv) {
    auto vfacc1 = lat_dual_vf_reflux.x1f;
    auto vfacc2 = lat_dual_vf_reflux.x2f;
    auto vfacc3 = lat_dual_vf_reflux.x3f;
    auto &eos = peos->eos_data;
    const int de_idx = dual_energy_idx;
    auto apply_dual_correction = KOKKOS_LAMBDA(const int m, const int k, const int j,
                                               const int i, const Real chi) {
      if (chi == 0.0) return;
      const Real dens = fmax(u(m, IDN, k, j, i), eos.dfloor);
      Real eint = u(m, de_idx, k, j, i);
      bool efloor_used = false, tfloor_used = false;
      eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint,
                                                  efloor_used, tfloor_used);
      if (eos.is_gamma_law) {
        eint *= exp((eos.gamma - 1.0)*chi);
      } else {
        const Real pressure = eos.PressureFromRhoEint(dens, eint);
        eint += pressure*chi;
      }
      efloor_used = false;
      tfloor_used = false;
      eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint,
                                                  efloor_used, tfloor_used);
      u(m, de_idx, k, j, i) = eint;
    };

    if (nface1 > 0) {
      Kokkos::TeamPolicy<> policy_x1_de(reflux_exec, nrefluxed*nface1, Kokkos::AUTO);
      Kokkos::parallel_for("lat_apply_dual_reflux_x1", policy_x1_de,
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int a = (tmember.league_rank())/nface1;
        const int nb = tmember.league_rank() - a*nface1;
        const int m = active_indices(a);
        if (lat_flux_recv(m,nb) == 0) return;

        const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
        const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
        const int il = iflux.bis;
        const int jl = iflux.bjs;
        const int ju = iflux.bje;
        const int kl = iflux.bks;
        const int ku = iflux.bke;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkj = nk*nj;

        const bool inner = (il == is);
        const int ic = inner ? is : ie;
        const Real sgn = inner ? -1.0 : 1.0;
        const int ia = inner ? 0 : 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k*nj) + jl;
          k += kl;
          const Real chi =
            theta1(m,0,k,j,ia)*(sgn*vfacc1(m,0,k,j,ia)/mbsize.d_view(m).dx1);
          apply_dual_correction(m, k, j, ic, chi);
          vfacc1(m,0,k,j,ia) = 0.0;
        });
        tmember.team_barrier();
      });
    }

    if (nface2 > 0) {
      Kokkos::TeamPolicy<> policy_x2_de(reflux_exec, nrefluxed*nface2, Kokkos::AUTO);
      Kokkos::parallel_for("lat_apply_dual_reflux_x2", policy_x2_de,
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int a = (tmember.league_rank())/nface2;
        const int nb = 8 + (tmember.league_rank() - a*nface2);
        const int m = active_indices(a);
        if (lat_flux_recv(m,nb) == 0) return;

        const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
        const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
        const int il = iflux.bis;
        const int iu = iflux.bie;
        const int jl = iflux.bjs;
        const int kl = iflux.bks;
        const int ku = iflux.bke;
        const int ni = iu - il + 1;
        const int nk = ku - kl + 1;
        const int nki = nk*ni;

        const bool inner = (jl == js);
        const int jc = inner ? js : je;
        const Real sgn = inner ? -1.0 : 1.0;
        const int ja = inner ? 0 : 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k*ni) + il;
          k += kl;
          const Real chi =
            theta2(m,0,k,ja,i)*(sgn*vfacc2(m,0,k,ja,i)/mbsize.d_view(m).dx2);
          apply_dual_correction(m, k, jc, i, chi);
          vfacc2(m,0,k,ja,i) = 0.0;
        });
        tmember.team_barrier();
      });
    }

    if (nface3 > 0) {
      Kokkos::TeamPolicy<> policy_x3_de(reflux_exec, nrefluxed*nface3, Kokkos::AUTO);
      Kokkos::parallel_for("lat_apply_dual_reflux_x3", policy_x3_de,
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int a = (tmember.league_rank())/nface3;
        const int nb = 24 + (tmember.league_rank() - a*nface3);
        const int m = active_indices(a);
        if (lat_flux_recv(m,nb) == 0) return;

        const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
        const auto &iflux = same_level ? rbuf[nb].iflux_same[0] : rbuf[nb].iflux_coar[0];
        const int il = iflux.bis;
        const int iu = iflux.bie;
        const int jl = iflux.bjs;
        const int kl = iflux.bks;
        const int ni = iu - il + 1;
        const int nj = iflux.bje - jl + 1;
        const int nji = nj*ni;

        const bool inner = (kl == ks);
        const int kc = inner ? ks : ke;
        const Real sgn = inner ? -1.0 : 1.0;
        const int ka = inner ? 0 : 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j*ni) + il;
          j += jl;
          const Real chi =
            theta3(m,0,ka,j,i)*(sgn*vfacc3(m,0,ka,j,i)/mbsize.d_view(m).dx3);
          apply_dual_correction(m, kc, j, i, chi);
          vfacc3(m,0,ka,j,i) = 0.0;
        });
        tmember.team_barrier();
      });
    }
  }
  // All correction kernels use this execution-space instance, so launch order supplies
  // their data dependencies.  Fence once before u0 and the accumulator masks are reused.
  reflux_exec.fence();
  return true;
}
} // namespace hydro
