//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_create.cpp
//! \brief Truelove/Jeans sink creation (ORION2 FORT_SINK_CREATE,
//! SINKPARTICLE_3D.F:21-168).
//!
//! A sink is created in every finest-level interior cell whose density exceeds
//!   rho_J = pi c_s^2 / (G (dx_max/jeans_no)^2).
//! There are deliberately NO secondary checks (no div v < 0, no boundedness, no
//! minimum mass, no "is a sink already nearby") - exactly as in ORION2; the FOF merge
//! that runs immediately afterwards is what suppresses swarms.  The magnetic Jeans
//! correction (1 + 0.74/beta) is inert here because this phase is hydro-only
//! (beta -> inf).
//!
//! The surgery conserves mass and momentum exactly and preserves the cell's SPECIFIC
//! (thermal+kinetic) energy; total energy is not conserved because the sink carries no
//! energy variable (map R17c).
//!
//! LAT (phase S2).  Unlike accretion, creation is NOT confined to the pinned
//! neighbourhood of an existing sink: it scans and writes every finest-level block, which
//! is legal only where that block is at the current mesh time.  At a fully synchronized
//! SinkStep invocation (a LAT window boundary, and the ordinary non-LAT cycle) that holds
//! for every block; at a fine-tick invocation it holds only for factor-1 blocks, which
//! SinkParticles::CheckLATInvariants turns into a fatal before any of this runs.  Only
//! interior cells are touched, as everywhere else in the module.
//!
//! A sink created here enters the list before SinkStep's closing RefreshSinkGMPos, so its
//! pin sphere is published by LATPinRegions() as soon as this function returns -- i.e.
//! before the driver's next Mesh::UpdateHydroLATMetadata, which is what the next window
//! is built from.  Creation cannot change the pin set mid-window because SinkStep never
//! runs mid-window.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
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

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::CreateSinks
//! \brief device flag+count scan over finest-level cells, then a deterministic host
//! finalize.  Candidate slots are claimed with an atomic counter (so the local order is
//! arbitrary) and the global list is sorted by (gid, cell index) before ids are handed
//! out, which makes the ids and the list order independent of the rank count.

void SinkParticles::CreateSinks(const SinkMeshGeom &g) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmb = pmy_pack->nmb_thispack;
  const int nmkji = nmb*nkji;

  auto u0_ = pmy_pack->phydro->u0;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto mblev = pmy_pack->pmb->mb_lev.d_view;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  const int de_idx = pmy_pack->phydro->dual_energy_idx;
  const bool dual = pmy_pack->phydro->use_dual_energy;
  const int nhyd = pmy_pack->phydro->nhydro;
  const int nscal = pmy_pack->phydro->nscalars;

  const Real gconst = newton_g;
  const Real jno = jeans_no;
  const int maxlev = g.max_level;
  const int gids = pmy_pack->gids;

  // (1) count Jeans-violating cells on this rank
  int nloc = 0;
  Kokkos::parallel_reduce("sink_create_count",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, int &count) {
    const int m = idx/nkji;
    if (mblev(m) != maxlev) { return; }
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    const Real dens = u0_(m, IDN, k, j, i);
    if (!(dens > 0.0)) { return; }
    Real cs2 = SQR(eos.iso_cs);
    if (eos.is_ideal) {
      const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                            SQR(u0_(m, IM3, k, j, i)))/dens;
      bool ef = false, tf = false;
      const Real eint = eos_general::ApplyHydroThermalFloors(eos, dens,
                            u0_(m, IEN, k, j, i) - e_k, ef, tf);
      Real pres = 0.0;
      eos.EvalPressureCs2FromRhoEint(dens, eint, pres, cs2);
    }
    const Real dxmax = fmax(mbsize.d_view(m).dx1,
                            fmax(mbsize.d_view(m).dx2, mbsize.d_view(m).dx3));
    const Real rho_j = M_PI*fmax(cs2, 0.0)/(gconst*SQR(dxmax/jno));
    if (dens > rho_j) { count += 1; }
  }, Kokkos::Sum<int>(nloc));

  // (2) claim slots, build the records and perform the gas surgery
  DualArray2D<int> ibuf("sink_create_ibuf", std::max(1, nloc), 2);
  // 8 Reals per record: {mass, x, y, z, px, py, pz, energy removed from the gas}.  The
  // last one has no recipient -- a sink has no energy member -- so it is a ledger term,
  // not sink state, and it is summed into diag.de_gas below rather than being dropped.
  DualArray2D<Real> rbuf("sink_create_rbuf", std::max(1, nloc), 8);
  if (nloc > 0) {
    DvceArray1D<int> counter("sink_create_counter", 1);
    Kokkos::deep_copy(counter, 0);
    auto ibuf_d = ibuf.d_view;
    auto rbuf_d = rbuf.d_view;
    const int nslot = nloc;

    Kokkos::parallel_for("sink_create_apply",
                         Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx) {
      const int m = idx/nkji;
      if (mblev(m) != maxlev) { return; }
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      const int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      const Real dens = u0_(m, IDN, k, j, i);
      if (!(dens > 0.0)) { return; }
      Real cs2 = SQR(eos.iso_cs);
      if (eos.is_ideal) {
        const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                              SQR(u0_(m, IM3, k, j, i)))/dens;
        bool ef = false, tf = false;
        const Real eint = eos_general::ApplyHydroThermalFloors(eos, dens,
                              u0_(m, IEN, k, j, i) - e_k, ef, tf);
        Real pres = 0.0;
        eos.EvalPressureCs2FromRhoEint(dens, eint, pres, cs2);
      }
      const Real dx1 = mbsize.d_view(m).dx1;
      const Real dx2 = mbsize.d_view(m).dx2;
      const Real dx3 = mbsize.d_view(m).dx3;
      const Real dxmax = fmax(dx1, fmax(dx2, dx3));
      const Real rho_j = M_PI*fmax(cs2, 0.0)/(gconst*SQR(dxmax/jno));
      if (!(dens > rho_j)) { return; }

      const int slot = Kokkos::atomic_fetch_add(&counter(0), 1);
      if (slot >= nslot) { return; }

      const Real vol = dx1*dx2*dx3;
      const Real frac = rho_j/dens;
      ibuf_d(slot, 0) = gids + m;
      ibuf_d(slot, 1) = ((k - ks)*nx2 + (j - js))*nx1 + (i - is);
      rbuf_d(slot, 0) = (dens - rho_j)*vol;
      rbuf_d(slot, 1) = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                                    mbsize.d_view(m).x1max);
      rbuf_d(slot, 2) = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                                    mbsize.d_view(m).x2max);
      rbuf_d(slot, 3) = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                                    mbsize.d_view(m).x3max);
      rbuf_d(slot, 4) = u0_(m, IM1, k, j, i)*(1.0 - frac)*vol;
      rbuf_d(slot, 5) = u0_(m, IM2, k, j, i)*(1.0 - frac)*vol;
      rbuf_d(slot, 6) = u0_(m, IM3, k, j, i)*(1.0 - frac)*vol;
      rbuf_d(slot, 7) = eos.is_ideal ? (u0_(m, IEN, k, j, i)*(1.0 - frac)*vol) : 0.0;

      // gas surgery: density down to rho_J, velocity and specific (thermal+kinetic)
      // energy unchanged, scalar mass fractions unchanged
      u0_(m, IDN, k, j, i) = rho_j;
      u0_(m, IM1, k, j, i) *= frac;
      u0_(m, IM2, k, j, i) *= frac;
      u0_(m, IM3, k, j, i) *= frac;
      if (eos.is_ideal) { u0_(m, IEN, k, j, i) *= frac; }
      for (int n=0; n<nscal; ++n) { u0_(m, nhyd+n, k, j, i) *= frac; }
      // E and KE both scale by frac, so E_int does too: scaling the auxiliary internal
      // energy by the same factor keeps the dual-energy pair consistent.
      if (dual) { u0_(m, de_idx, k, j, i) *= frac; }
    });
    ibuf.template modify<DevExeSpace>();
    rbuf.template modify<DevExeSpace>();
    ibuf.template sync<HostMemSpace>();
    rbuf.template sync<HostMemSpace>();
  }

  // (3) gather the records from every rank.  One int Allgather runs every cycle; the two
  // Allgathervs only when a sink was actually created anywhere.
  const int nranks = global_variable::nranks;
  const int myrank = global_variable::my_rank;
  std::vector<int> counts(nranks, 0);
  counts[myrank] = nloc;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif
  int ntot = 0;
  for (int r=0; r<nranks; ++r) { ntot += counts[r]; }
  if (ntot == 0) { return; }

  std::vector<int> idisp(nranks, 0), rdisp(nranks, 0), icnt(nranks, 0), rcnt(nranks, 0);
  int ioff = 0, roff = 0;
  for (int r=0; r<nranks; ++r) {
    icnt[r] = 2*counts[r];
    rcnt[r] = 8*counts[r];
    idisp[r] = ioff;
    rdisp[r] = roff;
    ioff += icnt[r];
    roff += rcnt[r];
  }
  std::vector<int> iall(2*ntot, 0);
  std::vector<Real> rall(8*ntot, 0.0);
  std::vector<int> isnd(2*std::max(1, nloc), 0);
  std::vector<Real> rsnd(8*std::max(1, nloc), 0.0);
  for (int s=0; s<nloc; ++s) {
    isnd[2*s]   = ibuf.h_view(s, 0);
    isnd[2*s+1] = ibuf.h_view(s, 1);
    for (int n=0; n<8; ++n) { rsnd[8*s+n] = rbuf.h_view(s, n); }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allgatherv(isnd.data(), 2*nloc, MPI_INT, iall.data(), icnt.data(), idisp.data(),
                 MPI_INT, MPI_COMM_WORLD);
  MPI_Allgatherv(rsnd.data(), 8*nloc, MPI_ATHENA_REAL, rall.data(), rcnt.data(),
                 rdisp.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
#else
  for (int n=0; n<2*ntot; ++n) { iall[n] = isnd[n]; }
  for (int n=0; n<8*ntot; ++n) { rall[n] = rsnd[n]; }
#endif

  // (4) deterministic ordering by (gid, cell index) -> rank-count-independent ids
  std::vector<int> order(ntot);
  for (int n=0; n<ntot; ++n) { order[n] = n; }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (iall[2*a] != iall[2*b]) { return iall[2*a] < iall[2*b]; }
    return iall[2*a+1] < iall[2*b+1];
  });

  for (int n=0; n<ntot; ++n) {
    const int c = order[n];
    SinkData s;
    s.id = nsinks_created++;
    s.m = rall[8*c];
    for (int d=0; d<3; ++d) {
      s.pos[d] = rall[8*c + 1 + d];
      s.mom[d] = rall[8*c + 4 + d];
      s.angmom[d] = 0.0;
    }
    diag.de_gas += rall[8*c + 7];
    sinks.push_back(s);
  }
  SyncSinkCount();
  if (global_variable::my_rank == 0) {
    std::cout << "### Sink creation: " << ntot << " new sink(s), " << sinks.size()
              << " total" << std::endl;
  }
}

}  // namespace sinkparticles
