//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_gravity.cpp
//! \brief gas -> sink gravitational kick (ORION2 FORT_GRAV_GAS_PARTICLE,
//! SINKPARTICLE_3D.F:668-839): a first-order explicit momentum kick from a direct sum
//! over every local interior cell.
//!
//! DISCRETE FORCE CONVENTION (changed while triaging D-8; see
//! tst/analysis/SINK_TESTS.md).
//! What this pass sums is the exact NEGATIVE of the per-cell force that
//! SourceTerms::SinkGravity hands the gas, namely the centred difference of the softened
//! sink potential,
//!
//!     a_gas,d = -( phi(x + dx_d e_d) - phi(x - dx_d e_d) ) / (2 dx_d),
//!
//! with phi = sinkparticles::SinkPotentialOne, the same function, evaluated at the same
//! points, in the same order.  The pair force is therefore equal and opposite CELL BY
//! CELL, so d(p_gas + p_sink)/dt = 0 holds discretely instead of only to O(dx^2).
//!
//! ORION2 (and this module until the D-8 triage) summed the ANALYTIC force
//! a = G M r_hat/(|r|^2 + s^2) here while pushing the gas with the potential difference.
//! That mismatch does not cancel: the sink's own gravity builds a strongly peaked,
//! co-moving gas atmosphere inside the softening radius, whose discrete sum exerts a
//! spurious self-force on the sink, and with mismatched conventions that self-force had
//! no reaction on the gas at all, i.e. it MANUFACTURED total momentum.  Measured in the
//! bondi configuration at 64^3: 87% of the sink's momentum had no counterpart in the gas.
//! The analytic sum additionally carries two artifacts the potential-difference form is
//! exactly free of (both quantified on SinkPotentialOne): a nonzero force from a uniform
//! periodic medium, and a nonzero residual at lattice symmetry points.
//!
//! The sink-sink force in AdvanceNBody is unaffected and stays analytic: there is no grid
//! there, so `a = G M r_hat/(r^2 + s^2)` is the exact force of the same potential and the
//! N-body pair is already equal and opposite by construction.
//!
//! LAT (phase S2).  This pass READS every local interior cell and writes no gas at all,
//! so it needs no factor-1 pinning; what it does need is that every block it reads be at
//! the current mesh time, which is why SinkStep belongs at a synchronized point.  At a
//! fully synchronized invocation that is exact.  At a fine-tick invocation the blocks in
//! slower bins contribute a state from an earlier time -- a first-order error in an
//! already first-order explicit kick, reported once by SinkStep, not silent.  The single
//! batched Allreduce below is legal only because SinkStep is never called inside a LAT
//! bin (no collective may run there).

#include <algorithm>
#include <cmath>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::GasSinkKick
//! \brief accumulate the reaction of the sink->gas source term over every local interior
//! cell and apply it to the sinks.  AthenaK leaf blocks do not overlap, so the
//! composite-grid masking ORION2 needs (SinkParticleList.cpp:1092-1113) disappears and
//! the sum runs over all local interior cells directly -- the same set of cells
//! SourceTerms::SinkGravity writes, which is what makes the pairing complete as well as
//! per-cell exact.
//!
//! The accumulator `dp` holds the momentum per unit time that the source term GIVES the
//! gas; the sinks then take `mom -= dt*dp`.  Sign conventions are therefore pinned by the
//! source term and cannot drift from it independently.
//!
//! This is the one O(N_cell * N_sink) sum in the module, so it is written as one flat
//! reduction per sink with four scalar reducers (three force components plus the
//! sink<->gas interaction energy) rather than as atomics over every cell.

void SinkParticles::GasSinkKick(Real dt, const SinkMeshGeom &g) {
  const int ns = nsinks;
  if (ns == 0 || dt <= 0.0) { return; }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmb = pmy_pack->nmb_thispack;
  const int nmkji = nmb*nkji;

  auto u0_ = pmy_pack->phydro->u0;
  auto &mbsize = pmy_pack->pmb->mb_size;
  // {G*M, x, y, z, soften^2} of every sink, frozen for this cycle.  Reusing the same
  // table the sink->gas source term reads is what makes the pair force consistent.
  auto gmpos = sink_gm_pos.d_view;

  // The source term only pushes the directions the mesh actually has, so the reaction
  // must be summed over exactly those directions too.
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // Mesh extents / periodicity for the minimum-image convention.  These are the same
  // quantities SourceTerms::SinkGravity derives from pmesh->mesh_size and mesh_bcs;
  // SinkMeshGeom caches them per cycle.
  const SinkMeshGeom gg = g;

  // 3 force components per sink plus ONE interaction-energy accumulator per sink; the
  // latter is the term that closes the gravitational energy channel (see below) and it
  // rides in the same buffer so it costs no extra collective.
  std::vector<Real> dp(4*ns, 0.0);
  for (int s=0; s<ns; ++s) {
    const int ss = s;
    Real fx = 0.0, fy = 0.0, fz = 0.0, fu = 0.0;
    Kokkos::parallel_reduce("sink_gas_kick",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &sx, Real &sy, Real &sz, Real &su) {
      const int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      const int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // For hydro the conserved and primitive densities are the same variable, so this is
      // the `w0(m,IDN,...)` the source term reads; the guard mirrors its `rho_fl <= 0`.
      // The one input on which the two predicates disagree is NaN: the source term
      // proceeds (and pushes the gas with NaN), this pass skips.  That asymmetry is
      // deliberate -- by then the pairing is already broken by the gas, and propagating
      // one NaN cell into every sink's momentum would kill the run in AdvanceNBody with
      // a message naming the softening instead of the gas.
      const Real rho = u0_(m, IDN, k, j, i);
      if (!(rho > 0.0)) { return; }

      const Real gms = gmpos(ss, 0);
      const Real px = gmpos(ss, 1), py = gmpos(ss, 2), pz = gmpos(ss, 3);
      const Real soft2 = gmpos(ss, 4);
      const Real dx1 = mbsize.d_view(m).dx1;
      const Real dx2 = mbsize.d_view(m).dx2;
      const Real dx3 = mbsize.d_view(m).dx3;
      const Real cellvol = dx1*dx2*dx3;

      const Real x1min = mbsize.d_view(m).x1min, x1max = mbsize.d_view(m).x1max;
      const Real x2min = mbsize.d_view(m).x2min, x2max = mbsize.d_view(m).x2max;
      const Real x3min = mbsize.d_view(m).x3min, x3max = mbsize.d_view(m).x3max;
      const Real xc = CellCenterX(i-is, nx1, x1min, x1max);
      const Real yc = CellCenterX(j-js, nx2, x2min, x2max);
      const Real zc = CellCenterX(k-ks, nx3, x3min, x3max);

      // phi at the cell centre and at the two neighbouring centres in each direction.
      // The neighbour coordinates come from CellCenterX(i +/- 1) rather than xc +/- dx1,
      // and the two one-sided differences are formed and added exactly as the source term
      // forms them, so the reaction is the same floating-point quantity and not merely
      // the same expression algebraically.
      const Real phi_c = SinkPotentialOne(gms, px, py, pz, soft2, xc, yc, zc,
                                          gg.len[0], gg.len[1], gg.len[2],
                                          gg.periodic[0], gg.periodic[1], gg.periodic[2]);
      // The sink<->gas INTERACTION ENERGY, sum_cells rho dV phi.  The source term does
      // real work on the gas and the reaction below does real work on the sink; neither
      // is a leak, but nothing stores the mutual potential energy that pays for the
      // difference, so an energy-conservation check on a sink run had a drift with no
      // attributable term.  phi_c is already in hand, so this is free.
      su += cellvol*rho*phi_c;
      {
        const Real xl = CellCenterX(i-1-is, nx1, x1min, x1max);
        const Real xr = CellCenterX(i+1-is, nx1, x1min, x1max);
        const Real dpl = -(phi_c - SinkPotentialOne(gms, px, py, pz, soft2, xl, yc, zc,
                                     gg.len[0], gg.len[1], gg.len[2],
                                     gg.periodic[0], gg.periodic[1], gg.periodic[2]));
        const Real dpr = -(SinkPotentialOne(gms, px, py, pz, soft2, xr, yc, zc,
                             gg.len[0], gg.len[1], gg.len[2],
                             gg.periodic[0], gg.periodic[1], gg.periodic[2]) - phi_c);
        sx += cellvol*(0.5/dx1)*rho*(dpl + dpr);
      }
      if (multi_d) {
        const Real yl = CellCenterX(j-1-js, nx2, x2min, x2max);
        const Real yr = CellCenterX(j+1-js, nx2, x2min, x2max);
        const Real dpl = -(phi_c - SinkPotentialOne(gms, px, py, pz, soft2, xc, yl, zc,
                                     gg.len[0], gg.len[1], gg.len[2],
                                     gg.periodic[0], gg.periodic[1], gg.periodic[2]));
        const Real dpr = -(SinkPotentialOne(gms, px, py, pz, soft2, xc, yr, zc,
                             gg.len[0], gg.len[1], gg.len[2],
                             gg.periodic[0], gg.periodic[1], gg.periodic[2]) - phi_c);
        sy += cellvol*(0.5/dx2)*rho*(dpl + dpr);
      }
      if (three_d) {
        const Real zl = CellCenterX(k-1-ks, nx3, x3min, x3max);
        const Real zr = CellCenterX(k+1-ks, nx3, x3min, x3max);
        const Real dpl = -(phi_c - SinkPotentialOne(gms, px, py, pz, soft2, xc, yc, zl,
                                     gg.len[0], gg.len[1], gg.len[2],
                                     gg.periodic[0], gg.periodic[1], gg.periodic[2]));
        const Real dpr = -(SinkPotentialOne(gms, px, py, pz, soft2, xc, yc, zr,
                             gg.len[0], gg.len[1], gg.len[2],
                             gg.periodic[0], gg.periodic[1], gg.periodic[2]) - phi_c);
        sz += cellvol*(0.5/dx3)*rho*(dpl + dpr);
      }
    }, Kokkos::Sum<Real>(fx), Kokkos::Sum<Real>(fy), Kokkos::Sum<Real>(fz),
       Kokkos::Sum<Real>(fu));

    dp[4*s]   = fx;
    dp[4*s+1] = fy;
    dp[4*s+2] = fz;
    dp[4*s+3] = fu;
  }

  // ONE Allreduce for all sinks (ORION2 issued 3*N separate scalar reductions, map R15)
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, dp.data(), 4*ns, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // `dp` is the momentum per unit time the source term gives the gas, so the sinks take
  // exactly its negative.  The residual in d(p_gas + p_sink)/dt is now only the
  // difference between the density this pass reads (end of cycle, u0) and the per-stage
  // primitive the source term integrated against, plus the fact that the source term is
  // applied stage by stage with the RK weights while this kick is one Lie-split push of
  // the full interval -- i.e. O(dt * d(rho)/dt), not O(rho * dx^2) as before.
  //
  // LAT: under per-block timestepping the source term weights each block by its own
  // lat_step_dt, while this reduction weights every block by the one SinkStep interval.
  // Blocks in the sink's neighbourhood are pinned to factor 1 (design N13), so the terms
  // that carry the force are weighted consistently; distant slower blocks contribute a
  // force that is negligible by construction, and SinkStep is only legal at synchronized
  // points anyway.
  for (int s=0; s<ns; ++s) {
    for (int d=0; d<3; ++d) { sinks[s].mom[d] -= dt*dp[4*s+d]; }
    diag.e_sink_gas += dp[4*s+3];
  }
}

}  // namespace sinkparticles
