//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_accrete.cpp
//! \brief Krumholz, Klein & McKee (2004) + Lee et al. (2014) sink accretion, ported from
//! ORION2 FORT_SINK_ACCRETE_EXTENDED (SINKPARTICLE_3D.F:171-663) with the design's fixes.
//!
//! Fixes relative to ORION2 (design D-e):
//!  R1  the ambient pressure is (GAMMA-1)*(E - E_kin); ORION2 mis-parenthesized it.
//!      Here it comes from the EOS via eint = E - KE, so tabulated LTE works too.
//!  R3  the accretion-kernel support radius is `accrete_radius_cells`, not a hard-coded
//!      4 cells.  The kernel WIDTH clamp [1,2] cells is kept.
//!  R5  only the Fielding (2015) angular-momentum treatment is ported.
//!  R7  the Mach number uses |v_mean - v_sink|, not the grid-frame |v_mean|.
//!  R17 every floor activation is counted and the leaked mass/energy accumulated.
//! R2/R8 (the Alfven-speed limiter and its dimensionally inconsistent Mach^2 term) are
//! dormant: this phase is hydro-only, so b2 == 0, beta short-circuits to `beta_max`, and
//! the limiter reduces to the identity.  Restoring MHD is a data change, not a rewrite.
//!
//! LAT (phase S2).  Two invariants hold here and are verified, not assumed:
//!  - INTERIOR CELLS ONLY.  Every one of the three passes below indexes
//!    i in [is, is+nx1), j in [js, js+nx2), k in [ks, ks+nx3), i.e. no ghost zone is ever
//!    read or written.  This is a hard LAT rule: under a mask a neighbour's ghost band is
//!    time-interpolated, not owned, so a write there would be discarded or would corrupt
//!    a block belonging to a different time.
//!  - FACTOR-1 CELLS ONLY.  Every block this kernel writes lies inside a sink's accretion
//!    sphere and is therefore pinned to LAT factor 1 (design N13).  The assertion is
//!    SinkParticles::CheckLATInvariants, run from SinkStep before any of this: it scans
//!    the local blocks whose bounding box reaches within (accrete_radius_cells + 1/2)
//!    finest cells of a sink -- a superset of the cells the `r2 > rad2` test admits --
//!    and reports (fully synchronized call) or fatals (fine-tick call) on any of them
//!    that is not in the factor-1 bin.  It is a host-side scan over replicated metadata
//!    rather than a device-side per-cell check, so it costs nothing in the kernels.
//! `dt` is the interval SinkStep was told to cover (SetStepInterval, else pmesh->dt), so
//! `mdot = dm/dt` and the removed mass stay consistent at either LAT cadence.
//!
//! ITERATION RANGE.  All three passes run over the LOCAL BLOCKS THE SPHERES REACH INTO,
//! not over the whole local grid (review A2/F14); the per-cell predicate is untouched, so
//! the contributing set is identical.  The list is built on the host from the same
//! (accrete_radius_cells + 1/2) finest-cell radius CheckLATInvariants uses.
//!
//! Parallel structure.  A sink's kernel spanning several ranks is the NORMAL case, so
//! nothing here may depend on the decomposition: each pass reduces per-sink partial sums
//! locally and combines them with ONE batched collective over all sinks, and the cell
//! updates are then applied locally with the globally reduced normalization.  Four
//! collectives per accretion step: shell sums (SUM), shell density minimum (MIN, a
//! different operator so it cannot share the call), kernel normalization (SUM), and the
//! actually-removed totals (SUM).  Only the ~(2*ngrow+1)^3 cells inside a sink's sphere
//! contribute, so the per-sink accumulators are written with atomics rather than with a
//! full-grid reduction.
//!
//! DETERMINISM (task-B rule 3, review A2/F12+F14).  These atomics are why the sink log
//! and the history file are not bitwise reproducible run to run (SINK_TESTS.md D-5), and
//! battery calibrates a measured noise floor instead of a bitwise gate.  Making them
//! deterministic was evaluated and NOT done: a tree reduction over ns*13 accumulators
//! needs an ARRAY-valued reducer, and Kokkos's built-in Sum/Min explicitly static_assert
//! array value types away (kokkos/core/src/Kokkos_Parallel_Reduce.hpp:39), so each pass
//! would have to become a hand-written functor with `value_type = Real[]` and
//! `value_count` -- three rewritten kernels, and pass 1 needs a MIN reducer beside the
//! SUM one.  It would still only buy determinism at FIXED decomposition (each rank sums
//! the cells it owns, so a 1-vs-2-rank comparison stays a tolerance gate either way),
//! which is what the existing noise-floor gates already cover.  The gas->sink kick
//! (sink_particles_gravity.cpp) is a different story: it already uses deterministic
//! Kokkos::Sum tree reductions, so nothing there needed upgrading.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "hydro/hydro.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

namespace {

// slot layouts of the per-sink accumulators
enum ShellSlot {SH_CNT=0, SH_RHO=1, SH_M1=2, SH_M2=3, SH_M3=4, SH_RCS2=5, SH_COARSE=6,
                SH_NVAL=7};
// RM_R* and RM_L* are the two moments the angular-momentum ledger needs:
//   RM_R = sum m_acc d           (d = TRUE minimum-image cell centre - sink position)
//   RM_L = sum m_acc (d x v_gas) (grid-frame angular momentum of the accreted material
//                                 about the sink)
// See the apply loop for why the pair closes the ledger exactly where the old
// snapped-lever-arm Fielding increment alone could not.
enum RemSlot {RM_DM=0, RM_P1=1, RM_P2=2, RM_P3=3, RM_E=4, RM_L1=5, RM_L2=6, RM_L3=7,
              RM_R1=8, RM_R2=9, RM_R3=10, RM_NRHO=11, RM_NEINT=12, RM_NCAP=13,
              RM_MLEAK=14, RM_ELEAK=15, RM_NVAL=16};

//----------------------------------------------------------------------------------------
//! \fn BondiAlpha
//! \brief alpha(x) = rho(r)/rho_inf for the critical Bondi solution, x = r/r_Bondi.
//! 51-point log-spaced table on [0.01, 2] plus the asymptotic branches, transcribed
//! verbatim from SINKPARTICLE_3D.F:946-990.  Host only.

Real BondiAlpha(const Real x) {
  constexpr int ntable = 51;
  constexpr Real xmin = 0.01;
  constexpr Real xmax = 2.0;
  static const Real alphatable[ntable] = {
    820.254, 701.882, 600.752, 514.341, 440.497, 377.381, 323.427,
    277.295, 237.845, 204.1, 175.23, 150.524, 129.377, 111.27, 95.7613,
    82.4745, 71.0869, 61.3237, 52.9498, 45.7644, 39.5963, 34.2989,
    29.7471, 25.8338, 22.4676, 19.5705, 17.0755, 14.9254, 13.0714,
    11.4717, 10.0903, 8.89675, 7.86467, 6.97159, 6.19825, 5.52812,
    4.94699, 4.44279, 4.00497, 3.6246, 3.29395, 3.00637, 2.75612,
    2.53827, 2.34854, 2.18322, 2.03912, 1.91344, 1.80378, 1.70804,
    1.62439};

  if (x <= xmin) {
    return sinkfloor::bondi_lambda/std::sqrt(2.0*x*x*x);
  } else if (x >= xmax) {
    return std::exp(1.0/x);
  }
  const Real logratio = std::log(xmax/xmin);
  int idx = static_cast<int>(std::floor((ntable - 1)*std::log(x/xmin)/logratio));
  idx = std::max(0, std::min(ntable - 2, idx));
  const Real xt = std::exp(std::log(xmin) + idx*logratio/(ntable - 1));
  const Real xt1 = std::exp(std::log(xmin) + (idx + 1)*logratio/(ntable - 1));
  const Real e = std::log(x/xt)/std::log(xt1/xt);
  return alphatable[idx]*std::pow(alphatable[idx+1]/alphatable[idx], e);
}

//! \struct BondiRate
//! \brief host-side result of the (rho_inf, beta) fixed point for one sink

struct BondiRate {
  bool ok;
  Real dm;         // mass to accrete this step, before the per-cell caps
  Real r_kernel;   // Gaussian width of the deposition kernel, in cells
  Real rho_keep;   // per-cell density floor guard of the removal cap
};

//----------------------------------------------------------------------------------------
//! \fn ComputeBondiRate
//! \brief the KKM04+Lee14 rate from the reduced ambient shell state.

BondiRate ComputeBondiRate(const Real mass, const Real *vsink, const Real *shell,
                           const Real rho_min, const Real dt, const Real dxmean,
                           const int ngrow, const Real gconst, const Real dfloor) {
  BondiRate out{false, 0.0, sinkfloor::accrete_rad_min, dfloor};
  const Real count = shell[SH_CNT];
  const Real sum_rho = shell[SH_RHO];
  const Real sum_rho_cs2 = shell[SH_RCS2];
  if (!(count > 0.0) || !(sum_rho > 0.0) || !(mass > 0.0)) { return out; }

  const Real rho_mean = sum_rho/count;
  const Real c_mean = std::sqrt(std::max(sum_rho_cs2/sum_rho, static_cast<Real>(0.0)));
  if (!(c_mean > 0.0)) { return out; }

  Real vrel2 = 0.0;
  for (int d=0; d<3; ++d) {
    vrel2 += SQR(shell[SH_M1+d]/sum_rho - vsink[d]);
  }
  // R7: Mach is measured in the sink's frame.  ORION2 used the grid-frame |v_mean|,
  // which inflates Mach and suppresses mdot for a sink comoving with a bulk flow.
  const Real mach = std::sqrt(vrel2)/c_mean;
  const Real machbh = std::pow(1.0 + SQR(SQR(mach)), 1.0/3.0)/
                      std::pow(1.0 + SQR(mach/sinkfloor::bondi_lambda), 1.0/6.0);
  const Real r_bh = gconst*mass/SQR(machbh*c_mean);
  if (!(r_bh > 0.0)) { return out; }

  // Hydro-only: b2 == 0, so beta = 2 rho c^2 / max(b2,1e-100) overflows.  Clamp to
  // beta_max, which is above the 1e9 convergence gate, so the loop runs once and every
  // Lee14 magnetic term evaluates to its beta -> inf limit.
  const Real beta0 = sinkfloor::beta_max;
  Real beta = beta0;
  // With B == 0 the field direction is undefined and ORION2's expression degenerates to
  // cos = 0, i.e. the purely perpendicular branch.  The hydro limit of the KKM04
  // interpolation is mdot_B/Mach_BH^3, which is the PARALLEL branch, so take cos = 1.
  const Real cos_bv = 1.0;
  const Real alpha_scale = static_cast<Real>(ngrow - 1);
  const Real nb = sinkfloor::lee14_nb;
  const Real betab = sinkfloor::lee14_betab;

  Real rho_inf = rho_mean;
  Real r_abh = r_bh;
  Real machabh = machbh;
  for (int it=0; it<sinkfloor::beta_max_iter; ++it) {
    const Real bterm = std::pow(betab/beta, 0.5*nb);
    machabh = std::pow(std::pow(machbh, nb) + bterm, 1.0/nb);
    r_abh = gconst*mass/SQR(machabh*c_mean);
    const Real alpha_pow = std::max(std::min(0.5/std::pow(beta, 0.13) - 0.27, 1.0), 0.0);
    const Real xrho = alpha_scale*std::max(dxmean, std::min(0.125*r_bh, 0.5*r_abh))/r_bh;
    rho_inf = rho_mean/std::pow(BondiAlpha(xrho), 1.0 - alpha_pow);
    const Real beta_last = beta;
    // |B| ~ rho during the Bondi infall outside r_alfven/4, |B| ~ B_0 (reconnection)
    // when the Alfven radius is resolved; this interpolates the two limits.
    beta = std::min(beta0*std::pow(BondiAlpha(alpha_scale*dxmean/r_bh), 1.0 - alpha_pow),
                    static_cast<Real>(sinkfloor::beta_max));
    if (beta > sinkfloor::beta_converged ||
        std::fabs(beta - beta_last)/std::fabs(beta_last + beta) < sinkfloor::beta_tol) {
      break;
    }
  }

  const Real mdot_b = 4.0*M_PI*rho_inf*sinkfloor::bondi_lambda*SQR(gconst*mass)/
                      (c_mean*c_mean*c_mean);
  const Real bterm = std::pow(betab/beta, 0.5*nb);
  const Real mdot_par = mdot_b/SQR(machbh)*
                        std::pow(std::pow(machbh, nb) + bterm, -1.0/nb);
  const Real mdot_perp = mdot_b/machbh*std::min(1.0/SQR(machbh), 1.0/machabh);
  const Real mdot = mdot_par*SQR(cos_bv) + mdot_perp*(1.0 - SQR(cos_bv));

  out.ok = true;
  out.dm = std::max(mdot*dt, static_cast<Real>(0.0));
  out.r_kernel = std::min(std::max(r_abh/dxmean, sinkfloor::accrete_rad_min),
                          sinkfloor::accrete_rad_max);
  out.rho_keep = std::max(std::min(rho_min, 0.1*rho_inf), dfloor);
  return out;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::Accrete
//! \brief three-pass accretion: ambient shell state, kernel normalization, removal.

void SinkParticles::Accrete(Real dt, const SinkMeshGeom &g) {
  const int ns = nsinks;
  if (ns == 0 || dt <= 0.0) { return; }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmb = pmy_pack->nmb_thispack;

  auto u0_ = pmy_pack->phydro->u0;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto mblev = pmy_pack->pmb->mb_lev.d_view;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  const int de_idx = pmy_pack->phydro->dual_energy_idx;
  const bool dual = pmy_pack->phydro->use_dual_energy;
  const int nhyd = pmy_pack->phydro->nhydro;
  const int nscal = pmy_pack->phydro->nscalars;
  const bool is_ideal = eos.is_ideal;

  const SinkMeshGeom gg = g;
  const int nsink = ns;
  const int ngrow = accrete_radius_cells;
  const Real rad2 = SQR(static_cast<Real>(ngrow));
  const Real shell_in2 = SQR(static_cast<Real>(std::max(0, ngrow - 2)));
  const Real feps = std::numeric_limits<Real>::epsilon();
  const int maxlev = g.max_level;

  // sink state on device: {m, x, y, z, vx, vy, vz}
  DualArray2D<Real> sk("sink_acc_state", ns, 7);
  for (int s=0; s<ns; ++s) {
    const SinkData &p = sinks[s];
    sk.h_view(s, 0) = p.m;
    for (int d=0; d<3; ++d) {
      sk.h_view(s, 1+d) = p.pos[d];
      sk.h_view(s, 4+d) = (p.m > 0.0) ? p.mom[d]/p.m : 0.0;
    }
  }
  sk.template modify<HostMemSpace>();
  sk.template sync<DevExeSpace>();
  auto sk_d = sk.d_view;

  // ---- the blocks any accretion sphere reaches into (review A2/F14).  All three passes
  // below used to sweep the WHOLE local grid -- RangePolicy(0, nmb*nkji), (2 + ns) times
  // per accretion step -- for a stencil of at most (2*ngrow+1)^3 cells per sink: with the
  // default ngrow = 4 that is ~729 useful cells out of ~2e7 per rank, a factor ~3e4 of
  // wasted iteration.  The cell predicate is unchanged, so exactly the same cells
  // contribute; only empty blocks are skipped.
  //
  // (ngrow + 1/2)*dxf_max is the same radius CheckLATInvariants uses, and the file header
  // above records why it is a strict superset of the `r2 > rad2` test: the per-direction
  // offsets are measured in FINEST cells and snapped to the nearest half cell, so a cell
  // the kernels admit is at most (ngrow + 1/4)*dxf_max from the sink.  A cell centre lies
  // inside its block's bounding box, so testing the box is a superset again.
  const Real near_rad2 = SQR((static_cast<Real>(ngrow) + 0.5)*g.dxf_max);
  DualArray1D<int> near_mb("sink_accrete_near_mb", std::max(1, nmb));
  int nnear = 0;
  {
    auto &mbsize_h = pmy_pack->pmb->mb_size;
    for (int m=0; m<nmb; ++m) {
      const Real bmin[3] = {mbsize_h.h_view(m).x1min, mbsize_h.h_view(m).x2min,
                            mbsize_h.h_view(m).x3min};
      const Real bmax[3] = {mbsize_h.h_view(m).x1max, mbsize_h.h_view(m).x2max,
                            mbsize_h.h_view(m).x3max};
      bool hit = false;
      for (int s=0; s<ns && !hit; ++s) {
        Real d2 = 0.0;
        for (int d=0; d<3; ++d) {
          const Real ctr = 0.5*(bmin[d] + bmax[d]);
          const Real hlf = 0.5*(bmax[d] - bmin[d]);
          const Real sep = std::fabs(SinkWrapDelta(sinks[s].pos[d] - ctr, g.len[d],
                                                   g.periodic[d])) - hlf;
          d2 += SQR(std::max(sep, static_cast<Real>(0.0)));
        }
        hit = (d2 <= near_rad2);
      }
      if (hit) { near_mb.h_view(nnear++) = m; }
    }
  }
  near_mb.template modify<HostMemSpace>();
  near_mb.template sync<DevExeSpace>();
  auto near_d = near_mb.d_view;
  const int nnkji = nnear*nkji;

  // ---------------------------------------------------------------- pass 1: shell state
  DvceArray2D<Real> shell_d("sink_shell", ns, SH_NVAL);
  DvceArray1D<Real> rhomin_d("sink_rhomin", ns);
  Kokkos::deep_copy(shell_d, 0.0);
  Kokkos::deep_copy(rhomin_d, static_cast<Real>(sinkfloor::huge));

  Kokkos::parallel_for("sink_accrete_shell",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, nnkji),
  KOKKOS_LAMBDA(const int idx) {
    const int a = idx/nkji;
    const int m = near_d(a);
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/nx1;
    const int i = (idx - a*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    const Real xc = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                                mbsize.d_view(m).x1max);
    const Real yc = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                                mbsize.d_view(m).x2max);
    const Real zc = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                                mbsize.d_view(m).x3max);
    const Real coarse = (mblev(m) != maxlev) ? 1.0 : 0.0;
    const Real dens = u0_(m, IDN, k, j, i);

    for (int s=0; s<nsink; ++s) {
      // cell-offset distances in FINEST cells, snapped to the nearest half cell so the
      // stencil is exactly symmetric about the sink (SINKPARTICLE_3D.F:246-251)
      const Real q1 = 0.5*SinkNint(2.0*SinkWrapDelta(xc - sk_d(s, 1), gg.len[0],
                                                     gg.periodic[0])/gg.dxf[0]);
      const Real q2 = 0.5*SinkNint(2.0*SinkWrapDelta(yc - sk_d(s, 2), gg.len[1],
                                                     gg.periodic[1])/gg.dxf[1]);
      const Real q3 = 0.5*SinkNint(2.0*SinkWrapDelta(zc - sk_d(s, 3), gg.len[2],
                                                     gg.periodic[2])/gg.dxf[2]);
      const Real r2 = SQR(q1) + SQR(q2) + SQR(q3);
      if (r2 > rad2) { continue; }

      Kokkos::atomic_min(&rhomin_d(s), dens);
      if (coarse > 0.0) { Kokkos::atomic_add(&shell_d(s, SH_COARSE), coarse); }
      if (!(r2 - feps > shell_in2)) { continue; }

      Real cs2 = SQR(eos.iso_cs);
      if (is_ideal) {
        const Real dsafe = fmax(dens, eos.dfloor);
        const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                              SQR(u0_(m, IM3, k, j, i)))/dsafe;
        bool ef = false, tf = false;
        const Real eint = eos_general::ApplyHydroThermalFloors(eos, dsafe,
                              u0_(m, IEN, k, j, i) - e_k, ef, tf);
        Real pres = 0.0;
        eos.EvalPressureCs2FromRhoEint(dsafe, eint, pres, cs2);
      }
      Kokkos::atomic_add(&shell_d(s, SH_CNT), 1.0);
      Kokkos::atomic_add(&shell_d(s, SH_RHO), dens);
      Kokkos::atomic_add(&shell_d(s, SH_M1), u0_(m, IM1, k, j, i));
      Kokkos::atomic_add(&shell_d(s, SH_M2), u0_(m, IM2, k, j, i));
      Kokkos::atomic_add(&shell_d(s, SH_M3), u0_(m, IM3, k, j, i));
      // rho*cs^2 sums so that c_mean^2 = sum(rho cs^2)/sum(rho); for an ideal gas this is
      // identically ORION2's gamma*p_mean/rho_mean, and it generalizes to tabulated LTE.
      Kokkos::atomic_add(&shell_d(s, SH_RCS2), dens*fmax(cs2, 0.0));
    }
  });

  std::vector<Real> shell(SH_NVAL*ns, 0.0), rhomin(ns, sinkfloor::huge);
  {
    auto h_shell = Kokkos::create_mirror_view(shell_d);
    auto h_rhomin = Kokkos::create_mirror_view(rhomin_d);
    Kokkos::deep_copy(h_shell, shell_d);
    Kokkos::deep_copy(h_rhomin, rhomin_d);
    for (int s=0; s<ns; ++s) {
      for (int n=0; n<SH_NVAL; ++n) { shell[SH_NVAL*s+n] = h_shell(s, n); }
      rhomin[s] = h_rhomin(s);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, shell.data(), SH_NVAL*ns, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, rhomin.data(), ns, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif

  // -------------------------------------------------------- host: rate and kernel width
  std::vector<BondiRate> rate(ns);
  for (int s=0; s<ns; ++s) {
    Real vsink[3];
    for (int d=0; d<3; ++d) {
      vsink[d] = (sinks[s].m > 0.0) ? sinks[s].mom[d]/sinks[s].m : 0.0;
    }
    rate[s] = ComputeBondiRate(sinks[s].m, vsink, &shell[SH_NVAL*s], rhomin[s], dt,
                               g.dxf_mean, ngrow, newton_g, eos.dfloor);
    diag.n_coarse_cell += shell[SH_NVAL*s + SH_COARSE];
  }

  DualArray2D<Real> par("sink_acc_par", ns, 3);   // {r_kernel, dm, rho_keep}
  for (int s=0; s<ns; ++s) {
    par.h_view(s, 0) = rate[s].r_kernel;
    par.h_view(s, 1) = rate[s].ok ? rate[s].dm : 0.0;
    par.h_view(s, 2) = rate[s].rho_keep;
  }
  par.template modify<HostMemSpace>();
  par.template sync<DevExeSpace>();
  auto par_d = par.d_view;

  // ------------------------------------------------- pass 2: kernel normalization (wtot)
  DvceArray1D<Real> wtot_d("sink_wtot", ns);
  Kokkos::deep_copy(wtot_d, 0.0);

  Kokkos::parallel_for("sink_accrete_wtot",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, nnkji),
  KOKKOS_LAMBDA(const int idx) {
    const int a = idx/nkji;
    const int m = near_d(a);
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/nx1;
    const int i = (idx - a*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    const Real xc = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                                mbsize.d_view(m).x1max);
    const Real yc = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                                mbsize.d_view(m).x2max);
    const Real zc = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                                mbsize.d_view(m).x3max);
    // Exactly pass 3's admission test.  Without it a vacuum or excised cell contributed
    // its full weight to the normalization and was then skipped by the removal, so every
    // real cell in the kernel received dm*w/wtot LESS than intended and the realized rate
    // fell below the KKM04 rate the log reports, with nothing to indicate it.
    if (!(u0_(m, IDN, k, j, i) > 0.0)) { return; }

    for (int s=0; s<nsink; ++s) {
      const Real q1 = 0.5*SinkNint(2.0*SinkWrapDelta(xc - sk_d(s, 1), gg.len[0],
                                                     gg.periodic[0])/gg.dxf[0]);
      const Real q2 = 0.5*SinkNint(2.0*SinkWrapDelta(yc - sk_d(s, 2), gg.len[1],
                                                     gg.periodic[1])/gg.dxf[1]);
      const Real q3 = 0.5*SinkNint(2.0*SinkWrapDelta(zc - sk_d(s, 3), gg.len[2],
                                                     gg.periodic[2])/gg.dxf[2]);
      const Real r2 = SQR(q1) + SQR(q2) + SQR(q3);
      if (r2 > rad2) { continue; }
      // flat core of one cell plus a Gaussian of scale r_kernel cells
      const Real w = (r2 <= 1.0) ? 1.0 : exp(-(r2 - 1.0)/SQR(par_d(s, 0)));
      Kokkos::atomic_add(&wtot_d(s), w);
    }
  });

  std::vector<Real> wtot(ns, 0.0);
  {
    auto h_wtot = Kokkos::create_mirror_view(wtot_d);
    Kokkos::deep_copy(h_wtot, wtot_d);
    for (int s=0; s<ns; ++s) { wtot[s] = h_wtot(s); }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, wtot.data(), ns, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // ------------------------------------------------------------- pass 3: remove and sum
  // One kernel per sink, launched in ascending sink order on every rank.  Sinks are
  // processed sequentially rather than concurrently so that overlapping kernels mutate
  // the gas in a well-defined order (ORION2's behaviour, map R16) instead of racing.
  DvceArray2D<Real> rem_d("sink_removed", ns, RM_NVAL);
  Kokkos::deep_copy(rem_d, 0.0);

  for (int s=0; s<ns; ++s) {
    if (!rate[s].ok || !(rate[s].dm > 0.0) || !(wtot[s] > 0.0)) { continue; }
    const int ss = s;
    const Real dm_target = rate[s].dm;
    const Real wnorm = wtot[s];
    const Real rho_keep = rate[s].rho_keep;
    const Real rk2 = SQR(rate[s].r_kernel);

    Kokkos::parallel_for("sink_accrete_apply",
                         Kokkos::RangePolicy<>(DevExeSpace(), 0, nnkji),
    KOKKOS_LAMBDA(const int idx) {
      const int a = idx/nkji;
      const int m = near_d(a);
      int k = (idx - a*nkji)/nji;
      int j = (idx - a*nkji - k*nji)/nx1;
      const int i = (idx - a*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      const Real xc = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                                  mbsize.d_view(m).x1max);
      const Real yc = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                                  mbsize.d_view(m).x2max);
      const Real zc = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                                  mbsize.d_view(m).x3max);
      // TRUE minimum-image separation of the cell centre from the sink.  The snapped
      // offsets q* below are derived from it; they select the stencil (that is what the
      // NINT snapping is for) and must not be used as a lever arm.
      const Real dsx = SinkWrapDelta(xc - sk_d(ss, 1), gg.len[0], gg.periodic[0]);
      const Real dsy = SinkWrapDelta(yc - sk_d(ss, 2), gg.len[1], gg.periodic[1]);
      const Real dsz = SinkWrapDelta(zc - sk_d(ss, 3), gg.len[2], gg.periodic[2]);
      const Real q1 = 0.5*SinkNint(2.0*dsx/gg.dxf[0]);
      const Real q2 = 0.5*SinkNint(2.0*dsy/gg.dxf[1]);
      const Real q3 = 0.5*SinkNint(2.0*dsz/gg.dxf[2]);
      const Real r2 = SQR(q1) + SQR(q2) + SQR(q3);
      if (r2 > rad2) { return; }

      const Real rho_old = u0_(m, IDN, k, j, i);
      if (!(rho_old > 0.0)) { return; }
      const Real dx1 = mbsize.d_view(m).dx1;
      const Real dx2 = mbsize.d_view(m).dx2;
      const Real dx3 = mbsize.d_view(m).dx3;
      const Real vol = dx1*dx2*dx3;

      const Real w = (r2 <= 1.0) ? 1.0 : exp(-(r2 - 1.0)/rk2);
      Real m_acc = dm_target*w/wnorm;
      const Real m_uncapped = m_acc;
      m_acc = fmin(m_acc, sinkfloor::maxaccfac*vol*rho_old);
      m_acc = fmin(m_acc, (rho_old - rho_keep)*vol);
      // Lee14 Alfven-speed cap: with b2 == 0 the limiting density equals rho_old, so
      // this reduces to the identity.  Kept explicit so an MHD extension is local.
      m_acc = fmax(m_acc, 0.0);
      if (m_acc < m_uncapped) { Kokkos::atomic_add(&rem_d(ss, RM_NCAP), 1.0); }
      if (!(m_acc > 0.0)) { return; }

      const Real f = m_acc/rho_old;              // has units of volume
      const Real frac = 1.0 - f/vol;             // fraction of the cell that remains
      const Real vgx = u0_(m, IM1, k, j, i)/rho_old;
      const Real vgy = u0_(m, IM2, k, j, i)/rho_old;
      const Real vgz = u0_(m, IM3, k, j, i)/rho_old;
      const Real p_acc1 = u0_(m, IM1, k, j, i)*f;
      const Real p_acc2 = u0_(m, IM2, k, j, i)*f;
      const Real p_acc3 = u0_(m, IM3, k, j, i)*f;
      const Real e_acc = is_ideal ? u0_(m, IEN, k, j, i)*f : 0.0;

      // The two moments of the accreted material about the sink.  Both use the TRUE
      // separation, and both are frame-independent sums: the Fielding (2015) increment
      // dL = m_acc*(d x (v_gas - v_sink)) is assembled from them on the host with the
      // POST-update sink velocity, which is what makes the total ledger close.
      Kokkos::atomic_add(&rem_d(ss, RM_R1), m_acc*dsx);
      Kokkos::atomic_add(&rem_d(ss, RM_R2), m_acc*dsy);
      Kokkos::atomic_add(&rem_d(ss, RM_R3), m_acc*dsz);
      Kokkos::atomic_add(&rem_d(ss, RM_L1), m_acc*(dsy*vgz - dsz*vgy));
      Kokkos::atomic_add(&rem_d(ss, RM_L2), m_acc*(dsz*vgx - dsx*vgz));
      Kokkos::atomic_add(&rem_d(ss, RM_L3), m_acc*(dsx*vgy - dsy*vgx));

      // gas update, interior cells only (ghosts are never touched)
      Real rho_new = rho_old*frac;
      Real m1 = u0_(m, IM1, k, j, i) - p_acc1/vol;
      Real m2 = u0_(m, IM2, k, j, i) - p_acc2/vol;
      Real m3 = u0_(m, IM3, k, j, i) - p_acc3/vol;
      Real etot = is_ideal ? (u0_(m, IEN, k, j, i) - e_acc/vol) : 0.0;

      if (rho_new < eos.dfloor) {
        // ORION2 resets rho AND rebuilds the momentum as dfloor*v (map R17a), which
        // manufactures dm_leak*v of gas momentum that the sink also keeps -- and only the
        // mass half was ever counted.  The momentum is no longer rebuilt: the cell keeps
        // the momentum the removal left it, so the ONLY manufactured quantity is the mass
        // already booked below.  This branch is in any case a rounding-scale repair, not
        // a physics path: the cap m_acc <= (rho_old - rho_keep)*vol with
        // rho_keep >= dfloor (ComputeBondiRate) already bounds rho_new from below.
        Kokkos::atomic_add(&rem_d(ss, RM_NRHO), 1.0);
        Kokkos::atomic_add(&rem_d(ss, RM_MLEAK), (eos.dfloor - rho_new)*vol);
        rho_new = eos.dfloor;
      }
      u0_(m, IDN, k, j, i) = rho_new;
      u0_(m, IM1, k, j, i) = m1;
      u0_(m, IM2, k, j, i) = m2;
      u0_(m, IM3, k, j, i) = m3;
      // scalar mass fractions are held fixed across the change.  sfac, not frac: on the
      // floor path the two differ, and the dual-energy auxiliary below used to be scaled
      // by the unfloored frac while its partner density was floored, so the (eaux, rho)
      // pair no longer encoded the same specific internal energy as (etot, rho).
      const Real sfac = rho_new/rho_old;
      for (int n=0; n<nscal; ++n) { u0_(m, nhyd+n, k, j, i) *= sfac; }
      const Real eaux = dual ? (u0_(m, de_idx, k, j, i)*sfac) : 0.0;

      if (is_ideal) {
        const Real e_k = 0.5*(SQR(m1) + SQR(m2) + SQR(m3))/rho_new;
        bool ef = false, tf = false;
        const Real eint_fl = eos_general::ApplyHydroThermalFloors(eos, rho_new,
                                 etot - e_k, ef, tf);
        if (ef || tf) {
          Kokkos::atomic_add(&rem_d(ss, RM_NEINT), 1.0);
          Kokkos::atomic_add(&rem_d(ss, RM_ELEAK), (eint_fl - (etot - e_k))*vol);
          etot = eint_fl + e_k;
        }
        u0_(m, IEN, k, j, i) = etot;
        if (dual) {
          bool ef2 = false, tf2 = false;
          u0_(m, de_idx, k, j, i) = eos_general::ApplyHydroThermalFloors(eos, rho_new,
                                        eaux, ef2, tf2);
        }
      }

      Kokkos::atomic_add(&rem_d(ss, RM_DM), m_acc);
      Kokkos::atomic_add(&rem_d(ss, RM_P1), p_acc1);
      Kokkos::atomic_add(&rem_d(ss, RM_P2), p_acc2);
      Kokkos::atomic_add(&rem_d(ss, RM_P3), p_acc3);
      Kokkos::atomic_add(&rem_d(ss, RM_E), e_acc);
    });
  }

  std::vector<Real> rem(RM_NVAL*ns, 0.0);
  {
    auto h_rem = Kokkos::create_mirror_view(rem_d);
    Kokkos::deep_copy(h_rem, rem_d);
    for (int s=0; s<ns; ++s) {
      for (int n=0; n<RM_NVAL; ++n) { rem[RM_NVAL*s+n] = h_rem(s, n); }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, rem.data(), RM_NVAL*ns, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  // ----------------------------------------------------------------- apply to the sinks
  //
  // ANGULAR MOMENTUM.  Write M, p, x for the sink before the step, dm = sum m_acc,
  // dp = sum m_acc v_gas, R = sum m_acc d and L = sum m_acc (d x v_gas), with d the
  // MINIMUM-IMAGE separation -- i.e. the unwrapped configuration MergeInto already books
  // its own conservation in.  The gas gives up  x x dp + L; the sink takes
  // x x dp + (R/M') x p' + dS  once
  // its centre of mass is displaced to x + R/M', so the ledger closes IDENTICALLY iff
  //
  //     dS = L - R x v',        v' = p'/M' = the POST-update sink velocity.
  //
  // That expression is Fielding (2015)'s dL = m_acc (d x (v_gas - v_sink)) evaluated
  // consistently, and it is assembled here from two reduced moments rather than per cell
  // because v' is not known until dm and dp are.  What used to be done -- the Fielding
  // increment with the PRE-update velocity and a snapped lever arm, and no displacement
  // of `pos` at all -- left a residual of -m_acc (d x v_sink) per cell.  For a supersonic
  // sink, sum m_acc d is systematically anti-parallel to v_sink instead of cancelling, so
  // that residual accumulated monotonically in a fixed direction, every step, unreported.
  for (int s=0; s<ns; ++s) {
    const Real *r = &rem[RM_NVAL*s];
    const Real dm = r[RM_DM];
    sinks[s].m += dm;
    for (int d=0; d<3; ++d) { sinks[s].mom[d] += r[RM_P1+d]; }

    // centre of mass of (old sink + accreted material), then the exact spin increment
    const Real minv = (sinks[s].m > 0.0) ? 1.0/sinks[s].m : 0.0;
    Real vnew[3];
    for (int d=0; d<3; ++d) {
      sinks[s].pos[d] += r[RM_R1+d]*minv;
      vnew[d] = sinks[s].mom[d]*minv;
    }
    WrapPosition(&sinks[s], g);
    const Real dl[3] = {r[RM_L1] - (r[RM_R2]*vnew[2] - r[RM_R3]*vnew[1]),
                        r[RM_L2] - (r[RM_R3]*vnew[0] - r[RM_R1]*vnew[2]),
                        r[RM_L3] - (r[RM_R1]*vnew[1] - r[RM_R2]*vnew[0])};

    // Keplerian cap (Fielding).  ORION2 evaluates j_kep with the POST-update mass
    // (SINKPARTICLE_3D.F:550 precedes :568); r_angmom = 1e100 leaves it inert.  What the
    // cap discards is real angular momentum leaving the gas+sink system, so it is the one
    // term of this ledger that can still be nonzero and it is counted.
    const Real dlmag = std::sqrt(SQR(dl[0]) + SQR(dl[1]) + SQR(dl[2]));
    const Real j_kep = (r_angmom > 0.0 && sinks[s].m > 0.0)
                     ? std::sqrt(newton_g*sinks[s].m*r_angmom) : 0.0;
    if (r_angmom > 0.0 && dlmag > 0.0 && dm*j_kep < dlmag) {
      for (int d=0; d<3; ++d) { sinks[s].angmom[d] += dm*j_kep*dl[d]/dlmag; }
      diag.dl_leak += dlmag - dm*j_kep;
    } else {
      for (int d=0; d<3; ++d) { sinks[s].angmom[d] += dl[d]; }
    }

    mdot[s] = dm/dt;
    // Gas total energy the sink removed.  A sink has no energy member, so this leaves the
    // total-energy ledger entirely; RM_E used to be summed here and then never read.
    diag.de_gas += r[RM_E];
    diag.n_rho_floor += r[RM_NRHO];
    diag.n_eint_floor += r[RM_NEINT];
    diag.n_cell_cap += r[RM_NCAP];
    diag.dm_leak += r[RM_MLEAK];
    diag.de_leak += r[RM_ELEAK];
  }
}

}  // namespace sinkparticles
