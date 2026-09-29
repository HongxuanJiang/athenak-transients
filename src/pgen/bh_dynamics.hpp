#ifndef PGEN_BH_DYNAMICS_HPP_
#define PGEN_BH_DYNAMICS_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bh_dynamics.hpp
//! \brief Live-BH helpers shared by the analytic-BH problem generators
//! (tde_external): the gas self-gravity sample at the BH, the
//! reciprocal (volume-weighted opposite gas) force, the orbit step, the rank-local stage
//! impulse ledger and the energy-ledger history columns.  Functions marked as
//! containing collectives must only run at a rank-synchronized point (the LAT window
//! hooks or a non-LAT cycle); the stage ledger never communicates.

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "pgen/bh_force_pair.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/gravity_weight.hpp"

namespace bh_dynamics {

KOKKOS_INLINE_FUNCTION
bool CoordinateTouchesBlock(const Real x, const Real xmin, const Real xmax,
                            const Real dx, const Real mesh_xmax) {
  const Real tol = 1.0e-12 * fmax(dx, static_cast<Real>(1.0));
  return (x >= xmin - tol) &&
         (x < xmax + tol || (xmax >= mesh_xmax && x <= xmax + tol));
}

KOKKOS_INLINE_FUNCTION
bool PositionTouchesBlock(const Real x, const Real y, const Real z,
                          const RegionSize mb, const Real mesh_x1max,
                          const Real mesh_x2max, const Real mesh_x3max) {
  return CoordinateTouchesBlock(x, mb.x1min, mb.x1max, mb.dx1, mesh_x1max) &&
         CoordinateTouchesBlock(y, mb.x2min, mb.x2max, mb.dx2, mesh_x2max) &&
         CoordinateTouchesBlock(z, mb.x3min, mb.x3max, mb.dx3, mesh_x3max);
}

KOKKOS_INLINE_FUNCTION
void InterpolateSelfGravityAccelerationAtPosition(
    const DvceArray5D<Real> self_phi, const RegionSize mb, const RegionIndcs indcs,
    const int m, const Real x, const Real y, const Real z,
    const bool multi_d, const bool three_d, Real &ax, Real &ay, Real &az) {
  // Interpolate face-centered potential differences. Each component is face
  // centered in its own direction and cell centered in the transverse ones.
  // One valid potential ghost layer (including edges/corners) is sufficient.
  const Real pos[3] = {x, y, z};
  const Real lower[3] = {mb.x1min, mb.x2min, mb.x3min};
  const Real dx[3] = {mb.dx1, mb.dx2, mb.dx3};
  const int first[3] = {indcs.is, indcs.js, indcs.ks};
  const int count[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  const bool active[3] = {true, multi_d, three_d};
  Real accel[3] = {0.0, 0.0, 0.0};
  for (int component = 0; component < 3; ++component) {
    if (!active[component]) continue;
    int index[3];
    Real weight[3];
    for (int d = 0; d < 3; ++d) {
      if (!active[d]) {
        index[d] = first[d];
        weight[d] = 0.0;
        continue;
      }
      // Block-ownership tolerance can admit a point just beyond the face.
      const Real local = fmin(fmax((pos[d] - lower[d])/dx[d], 0.0),
                              static_cast<Real>(count[d]));
      const Real q = local - ((d == component) ? 0.0 : 0.5);
      int offset = static_cast<int>(floor(q));
      if (d == component && offset == count[d]) --offset;
      index[d] = first[d] + offset;
      weight[d] = q - static_cast<Real>(offset);
    }
    for (int kside = 0; kside < (three_d ? 2 : 1); ++kside) {
      for (int jside = 0; jside < (multi_d ? 2 : 1); ++jside) {
        for (int iside = 0; iside < 2; ++iside) {
          const int side[3] = {iside, jside, kside};
          int hi[3], lo[3];
          Real w = 1.0;
          for (int d = 0; d < 3; ++d) {
            hi[d] = lo[d] = index[d] + side[d];
            w *= side[d] ? weight[d] : (1.0 - weight[d]);
          }
          if (w == 0.0) continue;
          --lo[component];
          accel[component] -= w*(self_phi(m, 0, hi[2], hi[1], hi[0]) -
                                  self_phi(m, 0, lo[2], lo[1], lo[0]))/dx[component];
        }
      }
    }
  }
  ax = accel[0];
  ay = accel[1];
  az = accel[2];
}

// The cells SourceTerms::Gravity gives no source (the excision zone and the BH sink
// gravity mask) and the radius it clips the BH potential at.  Both spheres are centred
// on the BH, the position the caller passes to Excludes.
struct GravitySourceMask {
  bool excise = false;
  Real excise_r2 = 0.0;
  bool sink = false;
  Real sink_radius = 0.0;
  KOKKOS_INLINE_FUNCTION
  bool Excludes(const Real x, const Real y, const Real z, const Real bhx,
                const Real bhy, const Real bhz) const {
    return (excise &&
            problem_runtime::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2)) ||
           (sink &&
            problem_runtime::InsideBHSinkGravityMask(x, y, z, bhx, bhy, bhz,
                                                     sink_radius));
  }
};

inline GravitySourceMask GetGravitySourceMask(const Real t) {
  GravitySourceMask mask;
  Real radius = 0.0, density = 0.0, eint = 0.0;
  Real cx = 0.0, cy = 0.0, cz = 0.0;
  problem_runtime::GetExcisionState(t, mask.excise, radius, density, eint, cx, cy, cz);
  mask.excise_r2 = radius*radius;
  problem_runtime::GetBHSinkGravityMask(t, mask.sink, mask.sink_radius, cx, cy, cz);
  return mask;
}

// Experimental instantaneous reciprocal force. This integrates the SAME spatial
// stencil, density weight and cell masks as SourceTerms::Gravity. It does not replace
// the missing RK/LAT stage-impulse reconciliation, and is not an energy fix alone.
inline bool SampleReciprocalAcceleration(Mesh *pm, const Real bhx, const Real bhy,
                                         const Real bhz, const Real bh_mass,
                                         const Real bh_softening, const Real bh_newton_g,
                                         const bool analytic_pair,
                                         Real &ax, Real &ay, Real &az) {
  ax = ay = az = 0.0;
  if (pm == nullptr || pm->pmb_pack == nullptr || pm->pmb_pack->phydro == nullptr ||
      pm->pmb_pack->phydro->psrc == nullptr || !(bh_mass > 0.0)) return false;
  auto *pack = pm->pmb_pack;
  const auto indcs = pm->mb_indcs;
  const auto size = pack->pmb->mb_size.d_view;
  const auto prim = pack->phydro->w0;
  const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
  const int per_block = nx*ny*nz;
  const int nwork = pack->nmb_thispack*per_block;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  const Real mass = bh_mass, softening = bh_softening;
  const Real newton_g = bh_newton_g;
  const Real rho_gate = pack->phydro->psrc->rho_external_bh_min;
  const Real rho_floor = pack->phydro->peos->eos_data.dfloor;
  const GravitySourceMask mask =
      GetGravitySourceMask(problem_runtime::HydroStageTimeOr(pm->time));
  const Real sink_radius = mask.sink_radius;
  Real local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("bh_reciprocal_force", Kokkos::RangePolicy<>(DevExeSpace(), 0,
      nwork),
  KOKKOS_LAMBDA(const int &index, Real &fx, Real &fy, Real &fz) {
    const int m = index/per_block;
    const int cell = index - m*per_block;
    const int k = cell/(nx*ny) + indcs.ks;
    const int j = (cell/nx)%ny + indcs.js;
    const int i = cell%nx + indcs.is;
    const Real rho = prim(m, IDN, k, j, i);
    if (!(rho > 0.0)) return;
    // The gas feels w(rho) of the BH pull, so it pulls back with w(rho) of its mass.
    const Real w_bh = gravity_weight::Weight(rho, rho_floor, rho_gate);
    if (!(w_bh > 0.0)) return;
    const auto mb = size(m);
    const Real x = CellCenterX(i-indcs.is, nx, mb.x1min, mb.x1max);
    const Real y = CellCenterX(j-indcs.js, ny, mb.x2min, mb.x2max);
    const Real z = CellCenterX(k-indcs.ks, nz, mb.x3min, mb.x3max);
    if (mask.Excludes(x, y, z, bhx, bhy, bhz)) return;
    const Real cell_mass = w_bh*rho*mb.dx1*mb.dx2*mb.dx3;
    if (analytic_pair) {
      const auto a = bh_force_pair::Evaluate(x,y,z,bhx,bhy,bhz,mass,softening,newton_g,
          sink_radius);
      fx -= cell_mass*a.ax;
      if (multi_d) fy -= cell_mass*a.ay;
      if (three_d) fz -= cell_mass*a.az;
      return;
    }
    const Real xl = CellCenterX(i-1-indcs.is, nx, mb.x1min, mb.x1max);
    const Real xr = CellCenterX(i+1-indcs.is, nx, mb.x1min, mb.x1max);
    const auto gx = bh_force_pair::Stencil(0,x,y,z,xl,xr,bhx,bhy,bhz,mass,softening,
        newton_g,sink_radius);
    fx -= cell_mass*gx.Acceleration(mb.dx1);
    if (multi_d) {
      const Real yl = CellCenterX(j-1-indcs.js, ny, mb.x2min, mb.x2max);
      const Real yr = CellCenterX(j+1-indcs.js, ny, mb.x2min, mb.x2max);
      const auto gy = bh_force_pair::Stencil(1,x,y,z,yl,yr,bhx,bhy,bhz,mass,softening,
          newton_g,sink_radius);
      fy -= cell_mass*gy.Acceleration(mb.dx2);
    }
    if (three_d) {
      const Real zl = CellCenterX(k-1-indcs.ks, nz, mb.x3min, mb.x3max);
      const Real zr = CellCenterX(k+1-indcs.ks, nz, mb.x3min, mb.x3max);
      const auto gz = bh_force_pair::Stencil(2,x,y,z,zl,zr,bhx,bhy,bhz,mass,softening,
          newton_g,sink_radius);
      fz -= cell_mass*gz.Acceleration(mb.dx3);
    }
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]), Kokkos::Sum<Real>(local[2]));
  Real global[3] = {local[0], local[1], local[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local, global, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  ax = global[0]/mass; ay = global[1]/mass; az = global[2]/mass;
  return std::isfinite(ax) && std::isfinite(ay) && std::isfinite(az);
}

// Gas self-gravity acceleration at a point, interpolated from the finest blocks that
// touch it.  Contains MPI collectives: call only at a rank-synchronized point.
inline bool SampleSelfGravityAcceleration(Mesh *pm, const Real bhx, const Real bhy,
                                          const Real bhz,
                                          Real &ax, Real &ay, Real &az) {
  ax = 0.0;
  ay = 0.0;
  az = 0.0;
  if (pm == nullptr || pm->pmb_pack == nullptr || pm->pmb_pack->pgrav == nullptr) {
    return false;
  }
  auto *pgrav = pm->pmb_pack->pgrav;
  if (!pgrav->phi_valid) return false;

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;

  // The BH is advanced by the gas self-gravity only.  The external softened BH
  // potential is applied separately to the gas and is not part of pgrav->phi.
  auto self_phi = pgrav->phi;
  auto &size = pmbp->pmb->mb_size;
  auto &level = pmbp->pmb->mb_lev;
  const int nmb = pmbp->nmb_thispack;
  const Real mesh_x1max = pm->mesh_size.x1max;
  const Real mesh_x2max = pm->mesh_size.x2max;
  const Real mesh_x3max = pm->mesh_size.x3max;

  int local_finest_level = -1;
  Kokkos::parallel_reduce("sample_bh_selfgrav_level",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
  KOKKOS_LAMBDA(const int &m, int &max_level) {
    const auto mb = size.d_view(m);
    if (PositionTouchesBlock(bhx, bhy, bhz, mb, mesh_x1max, mesh_x2max,
                             mesh_x3max)) {
      const int block_level = level.d_view(m);
      if (block_level > max_level) max_level = block_level;
    }
  }, Kokkos::Max<int>(local_finest_level));

  int global_finest_level = -1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_finest_level, &global_finest_level, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#else
  global_finest_level = local_finest_level;
#endif

  if (global_finest_level < 0) return false;

  Real local_buf[4] = {0.0, 0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("sample_bh_selfgrav_accel",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
  KOKKOS_LAMBDA(const int &m, Real &count, Real &sum_ax,
                                  Real &sum_ay, Real &sum_az) {
    if (level.d_view(m) != global_finest_level) return;

    const auto mb = size.d_view(m);
    if (!PositionTouchesBlock(bhx, bhy, bhz, mb, mesh_x1max, mesh_x2max,
                              mesh_x3max)) {
      return;
    }

    Real interp_ax, interp_ay, interp_az;
    InterpolateSelfGravityAccelerationAtPosition(self_phi, mb, indcs, m,
                                                 bhx, bhy, bhz,
                                                 multi_d, three_d,
                                                 interp_ax, interp_ay,
                                                 interp_az);

    count += 1.0;
    sum_ax += interp_ax;
    sum_ay += interp_ay;
    sum_az += interp_az;
  }, Kokkos::Sum<Real>(local_buf[0]), Kokkos::Sum<Real>(local_buf[1]),
     Kokkos::Sum<Real>(local_buf[2]), Kokkos::Sum<Real>(local_buf[3]));

  Real global_buf[4] = {0.0, 0.0, 0.0, 0.0};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local_buf, global_buf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int n = 0; n < 4; ++n) global_buf[n] = local_buf[n];
#endif

  bool ok = false;
  if (global_buf[0] > 0.0) {
    ax = global_buf[1] / global_buf[0];
    ay = global_buf[2] / global_buf[0];
    az = global_buf[3] / global_buf[0];
    ok = std::isfinite(ax) && std::isfinite(ay) && std::isfinite(az);
  }
  return ok;
}

// Accumulate the actual RK-weighted gas source impulse locally. HydroSrcTerms
// calls this immediately after Gravity while w0 still holds the same stage state.
// Deliberately no MPI here: rank-local LAT schedules need not call it equally.
// Adds this stage's contribution to the rank-local impulse accumulator `impulse`.
inline void AccumulateStageGasImpulse(Mesh *pm, const Real final_dt,
                                      const bool analytic_pair, Real impulse[3]) {
  auto *pack = pm->pmb_pack;
  const auto indcs = pm->mb_indcs;
  const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
  const int per_block = nx*ny*nz;
  const bool masked = pack->lat_active_mask_enabled;
  const bool block_dt = pack->lat_per_block_timestep;
  const int nwork = (masked ? pack->lat_nactive_thispack : pack->nmb_thispack)*per_block;
  if (nwork == 0 || final_dt == 0.0) return;
  const auto active = pack->lat_active_indices.d_view;
  const auto step_dt = pack->lat_step_dt.d_view;
  const Real stage_weight = final_dt/pm->dt;
  const auto size = pack->pmb->mb_size.d_view;
  const auto prim = pack->phydro->w0;
  const Real rho_gate = pack->phydro->psrc->rho_external_bh_min;
  const Real rho_floor = pack->phydro->peos->eos_data.dfloor;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  bool enabled;
  Real bhx, bhy, bhz, mass, softening, newton_g;
  const Real t_src = problem_runtime::HydroStageTimeOr(pm->time);
  problem_runtime::GetExternalBHPotential(t_src, enabled, bhx, bhy, bhz, mass,
                                          softening, newton_g);
  if (!enabled) {
    std::cerr << "BH impulse ledger requires the external potential\n";
    std::exit(EXIT_FAILURE);
  }
  const GravitySourceMask mask = GetGravitySourceMask(t_src);
  const Real sink_radius = mask.sink_radius;
  Real local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("bh_stage_impulse", Kokkos::RangePolicy<>(DevExeSpace(), 0,
      nwork),
  KOKKOS_LAMBDA(const int &index, Real &px, Real &py, Real &pz) {
    const int a = index/per_block;
    const int m = masked ? active(a) : a;
    const int cell = index-a*per_block;
    const int k = cell/(nx*ny)+indcs.ks;
    const int j = (cell/nx)%ny+indcs.js;
    const int i = cell%nx+indcs.is;
    const Real rho = prim(m, IDN, k, j, i);
    if (!(rho > 0.0)) return;
    // The same weight SourceTerms::Gravity gives this cell's BH momentum source.
    const Real w_bh = gravity_weight::Weight(rho, rho_floor, rho_gate);
    if (!(w_bh > 0.0)) return;
    const auto mb = size(m);
    const Real x = CellCenterX(i-indcs.is, nx, mb.x1min, mb.x1max);
    const Real y = CellCenterX(j-indcs.js, ny, mb.x2min, mb.x2max);
    const Real z = CellCenterX(k-indcs.ks, nz, mb.x3min, mb.x3max);
    if (mask.Excludes(x, y, z, bhx, bhy, bhz)) return;
    const Real dt = block_dt ? stage_weight*step_dt(m) : final_dt;
    const Real dm_dt = w_bh*rho*mb.dx1*mb.dx2*mb.dx3*dt;
    if (analytic_pair) {
      const auto a = bh_force_pair::Evaluate(x,y,z,bhx,bhy,bhz,mass,softening,newton_g,
          sink_radius);
      px += dm_dt*a.ax;
      if (multi_d) py += dm_dt*a.ay;
      if (three_d) pz += dm_dt*a.az;
      return;
    }
    const Real xl = CellCenterX(i-1-indcs.is, nx, mb.x1min, mb.x1max);
    const Real xr = CellCenterX(i+1-indcs.is, nx, mb.x1min, mb.x1max);
    px += dm_dt*bh_force_pair::Stencil(0,x,y,z,xl,xr,bhx,bhy,bhz,mass,softening,
                                      newton_g,sink_radius).Acceleration(mb.dx1);
    if (multi_d) {
      const Real yl = CellCenterX(j-1-indcs.js, ny, mb.x2min, mb.x2max);
      const Real yr = CellCenterX(j+1-indcs.js, ny, mb.x2min, mb.x2max);
      py += dm_dt*bh_force_pair::Stencil(1,x,y,z,yl,yr,bhx,bhy,bhz,mass,softening,
                                        newton_g,sink_radius).Acceleration(mb.dx2);
    }
    if (three_d) {
      const Real zl = CellCenterX(k-1-indcs.ks, nz, mb.x3min, mb.x3max);
      const Real zr = CellCenterX(k+1-indcs.ks, nz, mb.x3min, mb.x3max);
      pz += dm_dt*bh_force_pair::Stencil(2,x,y,z,zl,zr,bhx,bhy,bhz,mass,softening,
                                        newton_g,sink_radius).Acceleration(mb.dx3);
    }
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]), Kokkos::Sum<Real>(local[2]));
  for (int d = 0; d < 3; ++d) impulse[d] += local[d];
}

//----------------------------------------------------------------------------------------
//! \fn LedgerHistory
//! \brief The energy ledger written from column `base` on:
//! BH kinetic energy, gated BH-gas interaction energy, gated self-gravity energy
//! 1/2 sum(rho phi dV), the cumulative window-end time-centering work, the four
//! cumulative boundary-transport terms, the BH position/velocity, and the age of
//! the self-potential relative to the history time.  Everything is double precision
//! and evaluated on the synchronized state, so with a closed box and no nuclear
//! feedback  tot-E + K_BH + U_bhgas + U_self  must be constant; with open boundaries
//! add B_E + B_Wself + B_Wbh; with feedback subtract E_CNO + E_3alpha.  E_floor (column
//! 18) is the energy the gas floors and excision resets added.

inline void LedgerHistory(HistoryData *pdata, Mesh *pm, const int base,
                          const Real bh_mass, const Real bh_x, const Real bh_y,
                          const Real bh_z, const Real bh_vx, const Real bh_vy,
                          const Real bh_vz) {
  const int ncol = 20;
  pdata->nhist = base + ncol;
  const char *labels[ncol] = {"K_BH", "U_bhgas", "U_self", "W_cent", "B_mass", "B_E",
                              "B_Wself", "B_Wbh", "E_remap", "bh_x", "bh_y", "bh_z",
                              "bh_vx", "bh_vy", "bh_vz", "phi_age", "phi_int", "mesh_vol",
                              "E_floor", "E_asym"};
  for (int n = 0; n < ncol; ++n) {
    pdata->label[base+n] = labels[n];
    pdata->hdata[base+n] = 0.0;
  }
  if (pm == nullptr || pm->pmb_pack == nullptr || pm->pmb_pack->phydro == nullptr) {
    return;
  }
  // The history writer sums hdata across ranks; rank-replicated scalars are
  // therefore divided by nranks on every rank.
  const Real share = 1.0/static_cast<Real>(global_variable::nranks);
  gravity::Gravity *pgrav = pm->pmb_pack->pgrav;
  if (pgrav != nullptr) {
    Real u_self = 0.0, u_bh = 0.0;
    pgrav->LedgerEnergies(u_self, u_bh);
    pdata->hdata[base+1] = share*u_bh;
    pdata->hdata[base+2] = share*u_self;
    pdata->hdata[base+3] = share*pgrav->centered_work_total;
    for (int n = 0; n < 4; ++n) pdata->hdata[base+4+n] = share*pgrav->boundary_flux_total[n];
    pdata->hdata[base+8] = share*pgrav->remap_energy_total;
    if (pgrav->self_phi_time_valid) {
      pdata->hdata[base+15] = share*(pm->time - pgrav->self_phi_time);
    }
    Real phi_int = 0.0, mass = 0.0, volume = 0.0;
    pgrav->LedgerPotentialMoments(phi_int, mass, volume);
    pdata->hdata[base+16] = share*phi_int;
    pdata->hdata[base+17] = share*volume;
    pdata->hdata[base+18] = share*pgrav->floor_energy_total;
    pdata->hdata[base+19] = share*pgrav->asym_energy_total;
  }
  pdata->hdata[base+0] = share*0.5*bh_mass*
      (bh_vx*bh_vx + bh_vy*bh_vy + bh_vz*bh_vz);
  pdata->hdata[base+9] = share*bh_x;
  pdata->hdata[base+10] = share*bh_y;
  pdata->hdata[base+11] = share*bh_z;
  pdata->hdata[base+12] = share*bh_vx;
  pdata->hdata[base+13] = share*bh_vy;
  pdata->hdata[base+14] = share*bh_vz;
}

}  // namespace bh_dynamics

#endif  // PGEN_BH_DYNAMICS_HPP_
