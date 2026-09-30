//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.cpp
//! \brief Implements constructor and functions in MeshRefinement class.
//! Note while restriction functions for CC and FC data are implemented in this file,
//! prolongation operators are implemented as INLINE functions in prolongation.hpp (and
//! are used both here for AMR and in the BVals class at fine/coarse boundaries).
//!
//! Because refinement criteria and buffer metadata depend on physics, this constructor
//! is called in main() after physics modules are added and before the problem generator.

#include <cstdint>   // int32_t
#include <iostream>
#include <cmath>     // abs
#include <algorithm> // sort
#include <utility>   // pair
#include <vector>
#include <limits>
#include <memory>
#include <unordered_map>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "mesh/mb_storage.hpp"
#include "refinement_criteria.hpp"

#include "hydro/hydro.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "radiation/radiation.hpp"
#include "gravity/gravity.hpp"
#include "coordinates/adm.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "prolongation.hpp"
#include "restriction.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

struct AMRLogicalLocationKey {
  std::int32_t lx1, lx2, lx3, level;

  explicit AMRLogicalLocationKey(const LogicalLocation &loc)
      : lx1(loc.lx1), lx2(loc.lx2), lx3(loc.lx3), level(loc.level) {}

  bool operator==(const AMRLogicalLocationKey &other) const {
    return lx1 == other.lx1 && lx2 == other.lx2 && lx3 == other.lx3 &&
           level == other.level;
  }
};

struct AMRLogicalLocationKeyHash {
  std::size_t operator()(const AMRLogicalLocationKey &key) const {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint32_t value) {
      h ^= static_cast<std::uint64_t>(value);
      h *= 1099511628211ull;
    };
    mix(static_cast<std::uint32_t>(key.lx1));
    mix(static_cast<std::uint32_t>(key.lx2));
    mix(static_cast<std::uint32_t>(key.lx3));
    mix(static_cast<std::uint32_t>(key.level));
    return static_cast<std::size_t>(h);
  }
};

AMRLogicalLocationKey AMRParentKey(AMRLogicalLocationKey key) {
  key.lx1 >>= 1;
  key.lx2 >>= 1;
  key.lx3 >>= 1;
  --key.level;
  return key;
}

bool LogicalLocationLATDerefineOrder(const LogicalLocation &left,
                                     const LogicalLocation &right) {
  if (left.level != right.level) return left.level < right.level;
  const std::int32_t lpx1 = left.lx1 >> 1;
  const std::int32_t rpx1 = right.lx1 >> 1;
  const std::int32_t lpx2 = left.lx2 >> 1;
  const std::int32_t rpx2 = right.lx2 >> 1;
  const std::int32_t lpx3 = left.lx3 >> 1;
  const std::int32_t rpx3 = right.lx3 >> 1;
  if (lpx3 != rpx3) return lpx3 < rpx3;
  if (lpx2 != rpx2) return lpx2 < rpx2;
  if (lpx1 != rpx1) return lpx1 < rpx1;
  if ((left.lx3 & 1) != (right.lx3 & 1)) return (left.lx3 & 1) < (right.lx3 & 1);
  if ((left.lx2 & 1) != (right.lx2 & 1)) return (left.lx2 & 1) < (right.lx2 & 1);
  return left.lx1 < right.lx1;
}

struct LATBalanceScore {
  double sum_tick_max_work{0.0};
  double max_tick_work{0.0};
};

// work_eachmb is the measured per-step work of each block (Mesh::hydro_lat_work_eachmb,
// in the gid order of gids_eachrank); nullptr counts every block as one unit, which is
// exactly the integer count score this function returned before work was measured.
LATBalanceScore ScoreLATPartition(const int *factor_eachmb, int sync_factor,
                                  const int *gids_eachrank, const int *nmb_eachrank,
                                  int nranks, const Real *work_eachmb) {
  LATBalanceScore score;
  if (factor_eachmb == nullptr || gids_eachrank == nullptr ||
      nmb_eachrank == nullptr || nranks <= 0) {
    return score;
  }

  const int sync = std::max(1, sync_factor);
  int nlevels = 1;
  for (int factor=1; factor<sync; factor*=2) {
    ++nlevels;
    if (factor > std::numeric_limits<int>::max()/2) break;
  }
  std::vector<std::vector<double>> bin_work(
      nranks, std::vector<double>(nlevels, 0.0));
  std::vector<double> all_work(nranks, 0.0);
  for (int rank=0; rank<nranks; ++rank) {
    const int begin = gids_eachrank[rank];
    const int end = begin + nmb_eachrank[rank];
    for (int gid=begin; gid<end; ++gid) {
      int factor = std::max(1, std::min(sync, factor_eachmb[gid]));
      int level = 0;
      while (factor > 1 && level < nlevels - 1) {
        factor >>= 1;
        ++level;
      }
      const double work =
          (work_eachmb != nullptr) ? static_cast<double>(work_eachmb[gid]) : 1.0;
      bin_work[rank][level] += work;
      all_work[rank] += work;
    }
  }

  // Tick zero advances every factor.  Other ticks fall into log2(sync) phase classes:
  // sync/(2*factor) ticks advance exactly the bins up through that factor.
  double all_factor_max = 0.0;
  for (int rank=0; rank<nranks; ++rank) {
    all_factor_max = std::max(all_factor_max, all_work[rank]);
  }
  score.sum_tick_max_work = all_factor_max;
  score.max_tick_work = all_factor_max;

  std::vector<double> cumulative(nranks, 0.0);
  int factor = 1;
  for (int level=0; level<nlevels - 1; ++level) {
    double phase_max = 0.0;
    for (int rank=0; rank<nranks; ++rank) {
      cumulative[rank] += bin_work[rank][level];
      phase_max = std::max(phase_max, cumulative[rank]);
    }
    const int multiplicity = sync/(2*factor);
    score.sum_tick_max_work += static_cast<double>(multiplicity)*phase_max;
    score.max_tick_work = std::max(score.max_tick_work, phase_max);
    factor *= 2;
  }
  return score;
}

KOKKOS_INLINE_FUNCTION
void NewBlockBounds(const DvceArray1D<LogicalLocation> &lloc, const int gid,
                    const RegionSize &mesh_size, const int root_level,
                    const int nmb_rootx1, const int nmb_rootx2, const int nmb_rootx3,
                    const bool multi_d, const bool three_d,
                    Real &x1min, Real &x1max,
                    Real &x2min, Real &x2max,
                    Real &x3min, Real &x3max) {
  const auto loc = lloc(gid);
  const int lev = loc.level;

  const int nmbx1 = nmb_rootx1 << (lev - root_level);
  x1min = (loc.lx1 == 0) ? mesh_size.x1min
                         : LeftEdgeX(loc.lx1, nmbx1, mesh_size.x1min, mesh_size.x1max);
  x1max = (loc.lx1 == nmbx1 - 1) ? mesh_size.x1max
                                 : LeftEdgeX(loc.lx1 + 1, nmbx1, mesh_size.x1min,
                                             mesh_size.x1max);

  if (!multi_d) {
    x2min = mesh_size.x2min;
    x2max = mesh_size.x2max;
  } else {
    const int nmbx2 = nmb_rootx2 << (lev - root_level);
    x2min = (loc.lx2 == 0) ? mesh_size.x2min
                           : LeftEdgeX(loc.lx2, nmbx2, mesh_size.x2min, mesh_size.x2max);
    x2max = (loc.lx2 == nmbx2 - 1) ? mesh_size.x2max
                                   : LeftEdgeX(loc.lx2 + 1, nmbx2, mesh_size.x2min,
                                               mesh_size.x2max);
  }

  if (!three_d) {
    x3min = mesh_size.x3min;
    x3max = mesh_size.x3max;
  } else {
    const int nmbx3 = nmb_rootx3 << (lev - root_level);
    x3min = (loc.lx3 == 0) ? mesh_size.x3min
                           : LeftEdgeX(loc.lx3, nmbx3, mesh_size.x3min, mesh_size.x3max);
    x3max = (loc.lx3 == nmbx3 - 1) ? mesh_size.x3max
                                   : LeftEdgeX(loc.lx3 + 1, nmbx3, mesh_size.x3min,
                                               mesh_size.x3max);
  }
}

KOKKOS_INLINE_FUNCTION
Real CellCenteredBx(const DvceFaceFld4D<Real> &b, const int m, const int k,
                    const int j, const int i) {
  return 0.5 * (b.x1f(m, k, j, i) + b.x1f(m, k, j, i + 1));
}

KOKKOS_INLINE_FUNCTION
Real CellCenteredBy(const DvceFaceFld4D<Real> &b, const int m, const int k,
                    const int j, const int i) {
  return 0.5 * (b.x2f(m, k, j, i) + b.x2f(m, k, j + 1, i));
}

KOKKOS_INLINE_FUNCTION
Real CellCenteredBz(const DvceFaceFld4D<Real> &b, const int m, const int k,
                    const int j, const int i) {
  return 0.5 * (b.x3f(m, k, j, i) + b.x3f(m, k + 1, j, i));
}

KOKKOS_INLINE_FUNCTION
bool InsideCoordinateExcision(const bool enabled, const int scheme, const bool flat,
                              const Real spin, const Real rexcise,
                              const Real excise_lapse,
                              const Real x1, const Real x2, const Real x3) {
  if (!enabled) return false;
  if (scheme == static_cast<int>(ExcisionScheme::lapse)) {
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1, x2, x3, flat, spin, glower, gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);
    return alpha < excise_lapse;
  }
  return CartesianKerrSchildRadius(x1, x2, x3, spin) <= rexcise;
}

KOKKOS_INLINE_FUNCTION
Real SafePassiveScalar(const Real scalar) {
  return (isfinite(scalar) && scalar > 0.0) ? scalar : 0.0;
}

void RefinedHydroCoarseConsToPrim(const DvceArray1D<int> &n2o,
    const DvceArray1D<int> &rflag,
                                  const DvceArray1D<LogicalLocation> &lloc,
                                  const int new_gids, const int new_nmb_local,
                                  const DvceArray5D<Real> &cu, DvceArray5D<Real> &cw,
                                  const RegionSize &mesh_size, const int root_level,
                                  const int nmb_rootx1, const int nmb_rootx2,
                                  const int nmb_rootx3, const bool multi_d,
                                  const bool three_d,
                                  const EOS_Data &eos, const int nhyd, const int nscal,
                                  const bool use_dual, const int dual_idx,
                                  const Real dual_eta1, const bool is_sr,
                                  const bool is_gr, const CoordData coord,
                                  const bool excise_enabled, const Real excise_r2,
                                  const Real excise_density, const Real excise_eint,
                                  const Real bhx, const Real bhy, const Real bhz,
                                  const int coarse_is, const int coarse_js,
                                  const int coarse_ks, const int coarse_nx1,
                                  const int coarse_nx2, const int coarse_nx3,
                                  const int il, const int iu,
                                  const int jl, const int ju,
                                  const int kl, const int ku) {
  if (new_nmb_local <= 0) return;

  const bool coord_excise = is_gr && coord.bh_excise;
  const int coord_excise_scheme = static_cast<int>(coord.excision_scheme);
  const bool coord_flat = coord.is_minkowski;
  const Real coord_spin = coord.bh_spin;
  const Real coord_rexcise = coord.rexcise;
  const Real coord_dexcise = coord.dexcise;
  const Real coord_pexcise = coord.pexcise;
  const Real coord_excise_lapse = coord.excise_lapse;

  par_for("RefinedHydroCoarseConsToPrim", DevExeSpace(), 0, new_nmb_local - 1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag(n2o(m + new_gids)) <= 0) return;

    Real x = 0.0, y = 0.0, z = 0.0;
    if (is_gr || excise_enabled || coord_excise) {
      Real x1min, x1max, x2min, x2max, x3min, x3max;
      NewBlockBounds(lloc, m + new_gids, mesh_size, root_level, nmb_rootx1, nmb_rootx2,
                     nmb_rootx3, multi_d, three_d, x1min, x1max, x2min, x2max, x3min, x3max);
      x = CellCenterX(i - coarse_is, coarse_nx1, x1min, x1max);
      y = CellCenterX(j - coarse_js, coarse_nx2, x2min, x2max);
      z = CellCenterX(k - coarse_ks, coarse_nx3, x3min, x3max);
    }
    bool excised = excise_enabled &&
        problem_runtime::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2);
    Real floor_density = excise_density;
    Real floor_eint = excise_eint;
    if (coord_excise &&
        InsideCoordinateExcision(coord_excise, coord_excise_scheme, coord_flat,
                                 coord_spin, coord_rexcise, coord_excise_lapse,
                                 x, y, z)) {
      excised = true;
      floor_density = coord_dexcise;
      floor_eint = coord_pexcise/(eos.gamma - 1.0);
    }

    if (excised) {
      cw(m, IDN, k, j, i) = floor_density;
      cw(m, IVX, k, j, i) = 0.0;
      cw(m, IVY, k, j, i) = 0.0;
      cw(m, IVZ, k, j, i) = 0.0;
      if (eos.use_e) cw(m, IEN, k, j, i) = floor_eint;
      for (int n = nhyd; n < (nhyd + nscal); ++n) {
        cw(m, n, k, j, i) = 0.0;
      }
      if (use_dual) {
        cw(m, dual_idx, k, j, i) = floor_eint;
      }
      return;
    }

    HydCons1D u;
    u.d = cu(m, IDN, k, j, i);
    u.mx = cu(m, IM1, k, j, i);
    u.my = cu(m, IM2, k, j, i);
    u.mz = cu(m, IM3, k, j, i);
    u.e = eos.use_e ? cu(m, IEN, k, j, i) : 0.0;

    HydPrim1D w;
    bool dfloor_used = false, efloor_used = false, tfloor_used = false;
    bool vceiling_used = false;
    Real dual_prim_e = 0.0;
    if (is_gr) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x, y, z, coord_flat, coord_spin, glower, gupper);
      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u, glower, gupper, s2, u_sr);
      bool c2p_failure = false;
      int iter_used = 0;
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w, dfloor_used, efloor_used,
                           c2p_failure, iter_used);
      const Real v2 = glower[1][1]*SQR(w.vx) + glower[2][2]*SQR(w.vy) +
                      glower[3][3]*SQR(w.vz) + 2.0*glower[1][2]*w.vx*w.vy +
                      2.0*glower[1][3]*w.vx*w.vz + 2.0*glower[2][3]*w.vy*w.vz;
      const Real lor = sqrt(1.0 + v2);
      if (lor > eos.gamma_max) {
        const Real factor = sqrt((SQR(eos.gamma_max) - 1.0)/(SQR(lor) - 1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else if (is_sr) {
      const Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
      bool c2p_failure = false;
      int iter_used = 0;
      SingleC2P_IdealSRHyd(u, eos, s2, w, dfloor_used, efloor_used,
                           c2p_failure, iter_used);
      const Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
      if (lor > eos.gamma_max) {
        const Real factor = sqrt((SQR(eos.gamma_max) - 1.0)/(SQR(lor) - 1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else if (use_dual) {
      eos_general::SingleC2P_GeneralHydDual(
          u, eos, cu(m, dual_idx, k, j, i), dual_eta1, w, dual_prim_e,
          dfloor_used, efloor_used, tfloor_used, vceiling_used);
    } else {
      eos_general::SingleC2P_GeneralHyd(
          u, eos, w, dfloor_used, efloor_used, tfloor_used,
          vceiling_used);
    }

    cw(m, IDN, k, j, i) = w.d;
    cw(m, IVX, k, j, i) = w.vx;
    cw(m, IVY, k, j, i) = w.vy;
    cw(m, IVZ, k, j, i) = w.vz;
    if (eos.use_e) cw(m, IEN, k, j, i) = w.e;
    for (int n = nhyd; n < (nhyd + nscal); ++n) {
      const Real scalar_cons = (cu(m, n, k, j, i) < 0.0) ? 0.0 : cu(m, n, k, j, i);
      cw(m, n, k, j, i) = scalar_cons/u.d;
    }
    if (use_dual) {
      cw(m, dual_idx, k, j, i) = dual_prim_e;
    }
  });
}

void RefinedHydroPrimToCons(const DvceArray1D<int> &n2o, const DvceArray1D<int> &rflag,
                            const DvceArray1D<LogicalLocation> &lloc,
                            const int new_gids, const int new_nmb_local,
                            DvceArray5D<Real> &w, DvceArray5D<Real> &u,
                            const RegionSize &mesh_size, const int root_level,
                            const int nmb_rootx1, const int nmb_rootx2,
                            const int nmb_rootx3, const bool multi_d,
                            const bool three_d,
                            const EOS_Data &eos,
                            const int nhyd, const int nscal, const bool use_dual,
                            const int dual_idx, const bool is_sr, const bool is_gr,
                            const CoordData coord, const bool excise_enabled,
                            const Real excise_r2, const Real excise_density,
                            const Real excise_eint, const Real bhx, const Real bhy,
                            const Real bhz, const int mesh_is, const int mesh_js,
                            const int mesh_ks, const int mesh_nx1, const int mesh_nx2,
                            const int mesh_nx3, const int il, const int iu,
                            const int jl, const int ju,
                            const int kl, const int ku) {
  if (new_nmb_local <= 0) return;

  const bool coord_excise = is_gr && coord.bh_excise;
  const int coord_excise_scheme = static_cast<int>(coord.excision_scheme);
  const bool coord_flat = coord.is_minkowski;
  const Real coord_spin = coord.bh_spin;
  const Real coord_rexcise = coord.rexcise;
  const Real coord_dexcise = coord.dexcise;
  const Real coord_pexcise = coord.pexcise;
  const Real coord_excise_lapse = coord.excise_lapse;

  par_for("RefinedHydroPrimToCons", DevExeSpace(), 0, new_nmb_local - 1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag(n2o(m + new_gids)) <= 0) return;

    Real x = 0.0, y = 0.0, z = 0.0;
    if (is_gr || excise_enabled || coord_excise) {
      Real x1min, x1max, x2min, x2max, x3min, x3max;
      NewBlockBounds(lloc, m + new_gids, mesh_size, root_level, nmb_rootx1, nmb_rootx2,
                     nmb_rootx3, multi_d, three_d, x1min, x1max, x2min, x2max, x3min, x3max);
      x = CellCenterX(i - mesh_is, mesh_nx1, x1min, x1max);
      y = CellCenterX(j - mesh_js, mesh_nx2, x2min, x2max);
      z = CellCenterX(k - mesh_ks, mesh_nx3, x3min, x3max);
    }
    bool excised = excise_enabled &&
        problem_runtime::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2);
    Real floor_density = excise_density;
    Real floor_eint = excise_eint;
    if (coord_excise &&
        InsideCoordinateExcision(coord_excise, coord_excise_scheme, coord_flat,
                                 coord_spin, coord_rexcise, coord_excise_lapse,
                                 x, y, z)) {
      excised = true;
      floor_density = coord_dexcise;
      floor_eint = coord_pexcise/(eos.gamma - 1.0);
    }

    HydPrim1D wp;
    if (excised) {
      wp.d = floor_density;
      wp.vx = 0.0;
      wp.vy = 0.0;
      wp.vz = 0.0;
      wp.e = eos.use_e ? floor_eint : 0.0;
    } else {
      wp.d = w(m, IDN, k, j, i);
      wp.vx = w(m, IVX, k, j, i);
      wp.vy = w(m, IVY, k, j, i);
      wp.vz = w(m, IVZ, k, j, i);
      wp.e = eos.use_e ? w(m, IEN, k, j, i) : 0.0;
    }
    bool efloor_used = false, tfloor_used = false;
    if (!excised && !is_sr && !is_gr && eos.use_e) {
      wp.e = eos_general::ApplyHydroThermalFloors(eos, wp.d, wp.e, efloor_used,
                                                  tfloor_used);
    }

    HydCons1D uc;
    if (is_gr) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x, y, z, coord_flat, coord_spin, glower, gupper);
      SingleP2C_IdealGRHyd(glower, gupper, wp, eos.gamma, uc);
    } else if (is_sr) {
      SingleP2C_IdealSRHyd(wp, eos.gamma, uc);
    } else {
      eos_general::SingleP2C_GeneralHyd(wp, uc);
    }

    if (eos.use_e) w(m, IEN, k, j, i) = wp.e;
    u(m, IDN, k, j, i) = uc.d;
    u(m, IM1, k, j, i) = uc.mx;
    u(m, IM2, k, j, i) = uc.my;
    u(m, IM3, k, j, i) = uc.mz;
    if (eos.use_e) u(m, IEN, k, j, i) = uc.e;
    for (int n = nhyd; n < (nhyd + nscal); ++n) {
      u(m, n, k, j, i) = excised ? 0.0 : uc.d*w(m, n, k, j, i);
    }
    if (use_dual) {
      Real eint_aux = excised ? floor_eint : w(m, dual_idx, k, j, i);
      if (!excised) {
        efloor_used = false;
        tfloor_used = false;
        eint_aux = eos_general::ApplyHydroThermalFloors(eos, wp.d, eint_aux,
                                                        efloor_used, tfloor_used);
      }
      w(m, dual_idx, k, j, i) = eint_aux;
      u(m, dual_idx, k, j, i) = eint_aux;
    }
  });
}

void RefinedMHDCoarseConsToPrim(const DvceArray1D<int> &n2o,
    const DvceArray1D<int> &rflag,
                                const DvceArray1D<LogicalLocation> &lloc,
                                const int new_gids, const int new_nmb_local,
                                const DvceArray5D<Real> &cu,
                                const DvceFaceFld4D<Real> &cb,
                                DvceArray5D<Real> &cw,
                                const RegionSize &mesh_size, const int root_level,
                                const int nmb_rootx1, const int nmb_rootx2,
                                const int nmb_rootx3, const bool multi_d,
                                const bool three_d,
                                const EOS_Data &eos, const int nmhd, const int nscal,
                                const bool use_dual, const int dual_idx,
                                const Real dual_eta1, const bool is_sr, const bool is_gr,
                                const CoordData coord,
                                const bool excise_enabled, const Real excise_r2,
                                const Real excise_density, const Real excise_eint,
                                const Real bhx, const Real bhy, const Real bhz,
                                const int coarse_is, const int coarse_js,
                                const int coarse_ks, const int coarse_nx1,
                                const int coarse_nx2, const int coarse_nx3,
                                const int il, const int iu,
                                const int jl, const int ju,
                                const int kl, const int ku) {
  if (new_nmb_local <= 0) return;

  const bool coord_excise = is_gr && coord.bh_excise;
  const int coord_excise_scheme = static_cast<int>(coord.excision_scheme);
  const bool coord_flat = coord.is_minkowski;
  const Real coord_spin = coord.bh_spin;
  const Real coord_rexcise = coord.rexcise;
  const Real coord_dexcise = coord.dexcise;
  const Real coord_pexcise = coord.pexcise;
  const Real coord_excise_lapse = coord.excise_lapse;

  par_for("RefinedMHDCoarseConsToPrim", DevExeSpace(), 0, new_nmb_local - 1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag(n2o(m + new_gids)) <= 0) return;

    Real x = 0.0, y = 0.0, z = 0.0;
    bool have_position = false;
    if (is_gr || excise_enabled || coord_excise) {
      Real x1min, x1max, x2min, x2max, x3min, x3max;
      NewBlockBounds(lloc, m + new_gids, mesh_size, root_level, nmb_rootx1, nmb_rootx2,
                     nmb_rootx3, multi_d, three_d, x1min, x1max, x2min, x2max, x3min,
                     x3max);
      x = CellCenterX(i - coarse_is, coarse_nx1, x1min, x1max);
      y = CellCenterX(j - coarse_js, coarse_nx2, x2min, x2max);
      z = CellCenterX(k - coarse_ks, coarse_nx3, x3min, x3max);
      have_position = true;
    }

    bool excised = false;
    Real floor_density = excise_density;
    Real floor_eint = excise_eint;
    if (excise_enabled) {
      excised = problem_runtime::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2);
    }
    if (coord_excise && have_position &&
        InsideCoordinateExcision(coord_excise, coord_excise_scheme, coord_flat,
                                 coord_spin, coord_rexcise, coord_excise_lapse,
                                 x, y, z)) {
      excised = true;
      floor_density = coord_dexcise;
      floor_eint = coord_pexcise / (eos.gamma - 1.0);
    }

    if (excised) {
      cw(m, IDN, k, j, i) = floor_density;
      cw(m, IVX, k, j, i) = 0.0;
      cw(m, IVY, k, j, i) = 0.0;
      cw(m, IVZ, k, j, i) = 0.0;
      if (eos.use_e) cw(m, IEN, k, j, i) = floor_eint;
      for (int n = nmhd; n < (nmhd + nscal); ++n) {
        cw(m, n, k, j, i) = 0.0;
      }
      if (use_dual) {
        cw(m, dual_idx, k, j, i) = floor_eint;
      }
      return;
    }

    MHDCons1D u;
    u.d = cu(m, IDN, k, j, i);
    u.mx = cu(m, IM1, k, j, i);
    u.my = cu(m, IM2, k, j, i);
    u.mz = cu(m, IM3, k, j, i);
    u.e = eos.use_e ? cu(m, IEN, k, j, i) : 0.0;
    Real scalar_cons_density = u.d;
    u.bx = CellCenteredBx(cb, m, k, j, i);
    u.by = CellCenteredBy(cb, m, k, j, i);
    u.bz = CellCenteredBz(cb, m, k, j, i);

    HydPrim1D w;
    bool dfloor_used = false, efloor_used = false, tfloor_used = false;
    bool vceiling_used = false, sigceiling_used = false, c2p_failure = false;
    Real dual_prim_e = 0.0;
    if (is_gr) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x, y, z, coord_flat, coord_spin, glower, gupper);

      EOS_Data cell_eos = eos;

      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u, glower, gupper, s2, b2, rpar, u_sr);
      int iter_used = 0;
      SingleC2P_IdealSRMHD(u_sr, cell_eos, s2, b2, rpar, w, dfloor_used,
                           efloor_used, c2p_failure, iter_used);

      const auto lorentz_limit = mhd::forcefree::ApplyIsotropicLorentzLimit(
          glower, cell_eos.gamma_max, w.vx, w.vy, w.vz);
      if (!lorentz_limit.valid) {
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        vceiling_used = true;
      } else if (lorentz_limit.limited) {
        vceiling_used = true;
      }
      sigceiling_used = ApplySigmaCeiling_IdealGRMHD(cell_eos, glower, gupper, u, w);
    } else if (is_sr) {
      const Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
      const Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
      const Real rpar = (u.bx*u.mx + u.by*u.my + u.bz*u.mz)/u.d;
      int iter_used = 0;
      SingleC2P_IdealSRMHD(u, eos, s2, b2, rpar, w, dfloor_used,
                           efloor_used, c2p_failure, iter_used);
      const Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        const Real factor = sqrt((SQR(eos.gamma_max) - 1.0)/(SQR(lor) - 1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else if (use_dual) {
      // Primitives only: the coarse conserved state is restricted, never read back.
      bool eint_from_aux = false;
      eos_general::SingleC2P_GeneralMHDDual(
          u, eos, cu(m, dual_idx, k, j, i), dual_eta1, w, dual_prim_e, eint_from_aux,
          dfloor_used, efloor_used, tfloor_used, vceiling_used);
    } else {
      eos_general::SingleC2P_GeneralMHD(
          u, eos, w, dfloor_used, efloor_used, tfloor_used, vceiling_used);
    }

    cw(m, IDN, k, j, i) = w.d;
    cw(m, IVX, k, j, i) = w.vx;
    cw(m, IVY, k, j, i) = w.vy;
    cw(m, IVZ, k, j, i) = w.vz;
    if (eos.use_e) cw(m, IEN, k, j, i) = w.e;

    for (int n = nmhd; n < (nmhd + nscal); ++n) {
      Real scalar = 0.0;
      if (scalar_cons_density > 0.0) {
        scalar = SafePassiveScalar(cu(m, n, k, j, i) / scalar_cons_density);
      }
      cw(m, n, k, j, i) = scalar;
    }
    if (use_dual) {
      cw(m, dual_idx, k, j, i) = dual_prim_e;
    }
  });
}

void RefinedMHDPrimToCons(const DvceArray1D<int> &n2o, const DvceArray1D<int> &rflag,
                          const DvceArray1D<LogicalLocation> &lloc,
                          const int new_gids, const int new_nmb_local,
                          DvceArray5D<Real> &w, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &u,
                          const RegionSize &mesh_size, const int root_level,
                          const int nmb_rootx1, const int nmb_rootx2,
                          const int nmb_rootx3, const bool multi_d,
                          const bool three_d,
                          const EOS_Data &eos, const int nmhd, const int nscal,
                          const bool use_dual, const int dual_idx, const bool is_sr,
                          const bool is_gr, const CoordData coord,
                          const bool excise_enabled, const Real excise_r2,
                          const Real excise_density, const Real excise_eint,
                          const Real bhx, const Real bhy, const Real bhz,
                          const int mesh_is, const int mesh_js,
                          const int mesh_ks, const int mesh_nx1,
                          const int mesh_nx2, const int mesh_nx3,
                          const int il, const int iu,
                          const int jl, const int ju,
                          const int kl, const int ku) {
  if (new_nmb_local <= 0) return;

  const bool coord_excise = is_gr && coord.bh_excise;
  const int coord_excise_scheme = static_cast<int>(coord.excision_scheme);
  const bool coord_flat = coord.is_minkowski;
  const Real coord_spin = coord.bh_spin;
  const Real coord_rexcise = coord.rexcise;
  const Real coord_dexcise = coord.dexcise;
  const Real coord_pexcise = coord.pexcise;
  const Real coord_excise_lapse = coord.excise_lapse;

  par_for("RefinedMHDPrimToCons", DevExeSpace(), 0, new_nmb_local - 1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag(n2o(m + new_gids)) <= 0) return;

    Real x = 0.0, y = 0.0, z = 0.0;
    bool have_position = false;
    if (is_gr || excise_enabled || coord_excise) {
      Real x1min, x1max, x2min, x2max, x3min, x3max;
      NewBlockBounds(lloc, m + new_gids, mesh_size, root_level, nmb_rootx1, nmb_rootx2,
                     nmb_rootx3, multi_d, three_d, x1min, x1max, x2min, x2max, x3min,
                     x3max);
      x = CellCenterX(i - mesh_is, mesh_nx1, x1min, x1max);
      y = CellCenterX(j - mesh_js, mesh_nx2, x2min, x2max);
      z = CellCenterX(k - mesh_ks, mesh_nx3, x3min, x3max);
      have_position = true;
    }

    bool excised = false;
    Real floor_density = excise_density;
    Real floor_eint = excise_eint;
    if (excise_enabled) {
      excised = problem_runtime::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2);
    }
    if (coord_excise && have_position &&
        InsideCoordinateExcision(coord_excise, coord_excise_scheme, coord_flat,
                                 coord_spin, coord_rexcise, coord_excise_lapse,
                                 x, y, z)) {
      excised = true;
      floor_density = coord_dexcise;
      floor_eint = coord_pexcise / (eos.gamma - 1.0);
    }

    MHDPrim1D wp;
    if (excised) {
      wp.d = floor_density;
      wp.vx = 0.0;
      wp.vy = 0.0;
      wp.vz = 0.0;
      wp.e = eos.use_e ? floor_eint : 0.0;
    } else {
      wp.d = w(m, IDN, k, j, i);
      wp.vx = w(m, IVX, k, j, i);
      wp.vy = w(m, IVY, k, j, i);
      wp.vz = w(m, IVZ, k, j, i);
      wp.e = eos.use_e ? w(m, IEN, k, j, i) : 0.0;
    }
    wp.bx = CellCenteredBx(b, m, k, j, i);
    wp.by = CellCenteredBy(b, m, k, j, i);
    wp.bz = CellCenteredBz(b, m, k, j, i);

    bool efloor_used = false, tfloor_used = false;
    Real local_gamma = eos.gamma;
    EOS_Data cell_eos = eos;
    if (is_gr) {
      cell_eos.gamma = local_gamma;
      if (!isfinite(wp.d) || wp.d < cell_eos.dfloor) {
        wp.d = cell_eos.dfloor;
      }
      Real eint_floor = IdealMHDEintFloor(cell_eos, wp.d);
      if (!isfinite(wp.e) || wp.e < eint_floor) {
        wp.e = eint_floor;
        efloor_used = true;
      }
    } else if (!is_sr && eos.use_e) {
      wp.e = eos_general::ApplyMHDThermalFloors(eos, wp.d, wp.e, efloor_used,
                                                tfloor_used);
    }

    HydCons1D uc;
    if (is_gr) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x, y, z, coord_flat, coord_spin, glower, gupper);
      // Component-wise primitive prolongation does not preserve a metric Lorentz ball,
      // even when every coarse value was capped.  Exact-zero hybrid cells keep this
      // ordinary W/U pair authoritative, so close the invariant before GR P2C.
      const auto lorentz_limit = mhd::forcefree::ApplyIsotropicLorentzLimit(
          glower, cell_eos.gamma_max, wp.vx, wp.vy, wp.vz);
      if (!lorentz_limit.valid) {
        wp.vx = 0.0;
        wp.vy = 0.0;
        wp.vz = 0.0;
      }
      SingleP2C_IdealGRMHD(glower, gupper, wp, local_gamma, uc);
    } else if (is_sr) {
      SingleP2C_IdealSRMHD(wp, eos.gamma, uc);
    } else {
      MHDCons1D uc_mhd;
      eos_general::SingleP2C_GeneralMHD(wp, uc_mhd);
      uc.d = uc_mhd.d;
      uc.mx = uc_mhd.mx;
      uc.my = uc_mhd.my;
      uc.mz = uc_mhd.mz;
      uc.e = uc_mhd.e;
    }

    w(m, IDN, k, j, i) = wp.d;
    w(m, IVX, k, j, i) = wp.vx;
    w(m, IVY, k, j, i) = wp.vy;
    w(m, IVZ, k, j, i) = wp.vz;
    if (eos.use_e) w(m, IEN, k, j, i) = wp.e;
    u(m, IDN, k, j, i) = uc.d;
    u(m, IM1, k, j, i) = uc.mx;
    u(m, IM2, k, j, i) = uc.my;
    u(m, IM3, k, j, i) = uc.mz;
    if (eos.use_e) u(m, IEN, k, j, i) = uc.e;
    for (int n = nmhd; n < (nmhd + nscal); ++n) {
      if (excised) {
        Real scalar = 0.0;
        w(m, n, k, j, i) = scalar;
        u(m, n, k, j, i) = uc.d * scalar;
      } else {
        Real scalar = SafePassiveScalar(w(m, n, k, j, i));
        w(m, n, k, j, i) = scalar;
        u(m, n, k, j, i) = uc.d * scalar;
      }
    }
    if (use_dual) {
      Real eint_aux = excised ? floor_eint : w(m, dual_idx, k, j, i);
      efloor_used = false;
      tfloor_used = false;
      eint_aux = eos_general::ApplyMHDThermalFloors(eos, wp.d, eint_aux, efloor_used,
                                                    tfloor_used);
      w(m, dual_idx, k, j, i) = eint_aux;
      u(m, dual_idx, k, j, i) = eint_aux;
    }
  });
}

void ApplyUserRefinementCriterion(Mesh *pm, MeshBlockPack* pmbp) {
  if (pm == nullptr || pm->pgen == nullptr || pm->pgen->user_ref_func == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    Kokkos::abort("AMR criterion 'method = user' requested, but no user_ref_func"
                  " was enrolled by the problem generator");
  }
  pm->pgen->user_ref_func(pmbp);
}

}  // namespace

//----------------------------------------------------------------------------------------
// MeshRefinement constructor:
// called from Mesh::BuildTree (before physics modules are enrolled)

MeshRefinement::MeshRefinement(Mesh *pm, ParameterInput *pin) :
  pmy_mesh(pm),
  refine_flag("rflag",pm->nmb_total),
  refine_hold("rhold",pm->nmb_total),
  fc_amr_repair("fc_amr_repair",pm->nmb_total),
  ncyc_since_ref("cyc_since_ref",pm->nmb_total),
  nmb_created(0),
  nmb_deleted(0),
  nmb_sent_thisrank(0),
  ncyc_check_amr(1),
  refinement_interval(5),
  last_amr_call_cycle(pm->ncycle),
  ncyc_since_amr_check(0),
#if MPI_PARALLEL_ENABLED
  lb_chunk_elements(static_cast<std::size_t>(64LL*1024LL*1024LL)/sizeof(Real)),
  sendbuf("lb send buff",1),
  recvbuf("lb recv buff",1),
  send_data("lb send data",1),
  recv_data("lb recv data",1),
  lb_stage("lb amr stage",1),
#endif
  prolong_prims(false),
  sticky_load_balance(false),
  lb_transfer_gravity(true) {
  if (pin->DoesBlockExist("mesh_refinement")) {
#if MPI_PARALLEL_ENABLED
    const int lb_chunk_mb =
        pin->GetOrAddInteger("mesh_refinement", "lb_chunk_mb", 64);
    const std::int64_t lb_chunk_bytes =
        static_cast<std::int64_t>(lb_chunk_mb)*1024LL*1024LL;
    if (lb_chunk_mb <= 0 || lb_chunk_bytes < static_cast<std::int64_t>(sizeof(Real))) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "mesh_refinement/lb_chunk_mb must specify a positive staging size."
                  << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    lb_chunk_elements = static_cast<std::size_t>(lb_chunk_bytes)/sizeof(Real);
#endif
    // read interval (in cycles) between check of AMR and derefinement
    ncyc_check_amr = pin->GetOrAddInteger("mesh_refinement", "ncycle_check", 1);
    refinement_interval = pin->GetOrAddInteger("mesh_refinement", "refinement_interval",
        5);
    // refinement cooldown finer than the global AMR check cadence has no effect and
    // encourages threshold chatter. Enforce the cooldown on the same or coarser cadence.
    refinement_interval = std::max(refinement_interval, ncyc_check_amr);
    // read prolongate primitives flag
    if (pin->DoesParameterExist("mesh_refinement", "prolong_primitives")) {
      prolong_prims = pin->GetBoolean("mesh_refinement", "prolong_primitives");
    }
    const bool sticky_load_balance_requested =
        pin->GetOrAddBoolean("mesh_refinement", "sticky_load_balance", false);
    const bool hydro_lat_enabled = pin->IsLATEnabled();
    sticky_load_balance = sticky_load_balance_requested && hydro_lat_enabled;
    if (sticky_load_balance_requested && !hydro_lat_enabled &&
        global_variable::my_rank == 0) {
      std::cout << "MeshRefinement: sticky AMR load balance ignored because HD LAT "
                << "is disabled" << std::endl;
    }
    if (sticky_load_balance && global_variable::my_rank == 0) {
      std::cout << "MeshRefinement: sticky AMR load balance enabled" << std::endl;
      std::cout << "MeshRefinement: sticky AMR load balance is used only when no "
                << "HD LAT factor rebalance is active"
                << std::endl;
    }
  }

  // Physics modules are already constructed at this point.  Preserve the AMR policy
  // that their constructors used when MeshRefinement was created earlier: thermal
  // closures with an auxiliary/internal state must prolong primitives and rebuild a
  // consistent conserved state on the fine mesh.
  if (pm->pmb_pack->phydro != nullptr) {
    auto *phydro = pm->pmb_pack->phydro;
    if (phydro->dual_energy_pdv || phydro->peos->eos_data.UsesTabulatedLTE()) {
      prolong_prims = true;
    }
  }
  if (pm->pmb_pack->pmhd != nullptr) {
    auto *pmhd = pm->pmb_pack->pmhd;
    // dual_energy_pdv, not use_dual_energy: only the non-relativistic auxiliary is an
    // internal-energy density whose primitive form has to survive the prolongation.
    // The GR adiabat is per unit mass and conserved as D*kappa, so it crosses a
    // coarse-fine boundary the way a passive scalar already does, and requiring
    // primitive prolongation for it would forbid LAT on the dynamical-GR path.
    if (pmhd->dual_energy_pdv || pmhd->peos->eos_data.UsesTabulatedLTE()) {
      prolong_prims = true;
    }
  }

  // allocate arrays for AMR, add RefinementCriteria object
  if (pm->adaptive) {
    nref_eachrank = new int[global_variable::nranks];
    nderef_eachrank = new int[global_variable::nranks];
    nref_rsum = new int[global_variable::nranks];
    nderef_rsum = new int[global_variable::nranks];
    pmrc = new RefinementCriteria(pm, pin);
  }

  // be sure Views are initialized to zero
  for (int m=0; m<(pm->nmb_total); ++m) {
    refine_flag.h_view(m) = 0;
    fc_amr_repair.h_view(m) = 0;
    ncyc_since_ref(m) = 0;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  fc_amr_repair.template modify<HostMemSpace>();
  fc_amr_repair.template sync<DevExeSpace>();

  // initialize interpolation weights for prolongation and restriction
  InitInterpWghts();

#if MPI_PARALLEL_ENABLED
  // create unique communicators for AMR
  MPI_Comm_dup(MPI_COMM_WORLD, &amr_comm);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

MeshRefinement::~MeshRefinement() {
  if (pmy_mesh->adaptive) { // deallocate arrays for AMR
    delete [] nref_eachrank;
    delete [] nderef_eachrank;
    delete [] nref_rsum;
    delete [] nderef_rsum;
  }
#if MPI_PARALLEL_ENABLED
  if (amr_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&amr_comm);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::AdaptiveMeshRefinement()
//! \brief Simple driver function for adaptive mesh refinement

void MeshRefinement::AdaptiveMeshRefinement(Driver *pdriver, ParameterInput *pin) {
  const int elapsed_cycles = std::max(1, pmy_mesh->ncycle - last_amr_call_cycle);
  last_amr_call_cycle = pmy_mesh->ncycle;
  for (int m=0; m<(pmy_mesh->nmb_total); ++m) {
    ncyc_since_ref(m) += elapsed_cycles;
  }
  ncyc_since_amr_check += elapsed_cycles;
  if (((pmy_mesh->ncycle)%(ncyc_check_amr) != 0) &&
      (ncyc_since_amr_check < ncyc_check_amr)) {
    return;
  }
  ncyc_since_amr_check = 0;

  // first check refinement criteria
  CheckForRefinement(pmy_mesh->pmb_pack);
  LimitRefinementToMemoryCap();

  // then update mesh tree if MeshBlock anywhere (on any rank) is flagged for refinement
  // indicated by the refine_flag[m] value being true (1) for any m.
  int nnew = 0, ndel = 0;
  UpdateMeshBlockTree(nnew, ndel);

  if ((nnew != 0 || ndel != 0) && pmy_mesh->nmb_maxperrank > 0) {
    const int hard_cap = pmy_mesh->nmb_maxperrank * std::max(1, global_variable::nranks);
    int actual_nmb = 0;
    pmy_mesh->ptree->CountMeshBlocks(actual_nmb);
    if (actual_nmb > hard_cap) {
      if (global_variable::my_rank == 0) {
        std::cout << "MeshRefinement: AMR memory cap rejected tree closure with "
                  << actual_nmb << " MeshBlocks > cap " << hard_cap
                  << "; canceling this refinement pass" << std::endl;
      }
      pmy_mesh->ptree = std::make_unique<MeshBlockTree>(pmy_mesh);
      pmy_mesh->ptree->CreateRootGrid();
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        pmy_mesh->ptree->AddNodeWithoutRefinement(pmy_mesh->lloc_eachmb[gid]);
      }
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        pmy_mesh->ptree->SetLeafGID(pmy_mesh->lloc_eachmb[gid], gid);
      }
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        refine_flag.h_view(gid) = 0;
      }
      refine_flag.template modify<HostMemSpace>();
      refine_flag.template sync<DevExeSpace>();
      nnew = 0;
      ndel = 0;
    }
  }
  if ((nnew != 0 || ndel != 0) && pmy_mesh->nmb_maxperrank > 0) {
    const int hard_cap = pmy_mesh->nmb_maxperrank * std::max(1, global_variable::nranks);
    const int projected_nmb = pmy_mesh->nmb_total + nnew - ndel;
    if (projected_nmb > hard_cap) {
      if (global_variable::my_rank == 0) {
        std::cout << "MeshRefinement: AMR memory cap rejected realized closure with "
                  << projected_nmb << " MeshBlocks > cap " << hard_cap
                  << "; canceling this refinement pass" << std::endl;
      }
      pmy_mesh->ptree = std::make_unique<MeshBlockTree>(pmy_mesh);
      pmy_mesh->ptree->CreateRootGrid();
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        pmy_mesh->ptree->AddNodeWithoutRefinement(pmy_mesh->lloc_eachmb[gid]);
      }
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        pmy_mesh->ptree->SetLeafGID(pmy_mesh->lloc_eachmb[gid], gid);
      }
      for (int gid=0; gid<pmy_mesh->nmb_total; ++gid) {
        refine_flag.h_view(gid) = 0;
      }
      refine_flag.template modify<HostMemSpace>();
      refine_flag.template sync<DevExeSpace>();
      nnew = 0;
      ndel = 0;
    }
  }

  // Refine/derefine mesh and evolved data, set boundary conditions/timestep on new mesh
  if (nnew != 0 || ndel != 0) { // at least one (de)refinement flagged
    RedistAndRefineMeshBlocks(pin, nnew, ndel);

    MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
    pdriver->InitBoundaryValuesAndPrimitives(pmy_mesh, true);

    if (pmbp->phydro != nullptr) {
      (void) pmbp->phydro->NewTimeStep(pdriver, pdriver->nexp_stages);
    }
    if (pmbp->pmhd != nullptr) {
      (void) pmbp->pmhd->NewTimeStep(pdriver, pdriver->nexp_stages);
    }
    if (pmbp->prad != nullptr) {
      (void) pmbp->prad->NewTimeStep(pdriver, pdriver->nexp_stages);
    }
    if (pmbp->pz4c != nullptr) {
      (void) pmbp->pz4c->NewTimeStep(pdriver, pdriver->nexp_stages);
    }

    nmb_created += nnew;
    nmb_deleted += ndel;

    int ncomm_event = nmb_sent_thisrank;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ncomm_event, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "AMR event: cycle=" << pmy_mesh->ncycle
                << " time=" << pmy_mesh->time << std::endl
                << nnew << " MeshBlocks created, " << ndel << " deleted by AMR"
                << std::endl
                << ncomm_event << " communicated for load balancing"
                << std::endl
                << "Current number of MeshBlocks = " << pmy_mesh->nmb_total
                << std::endl;
    }
  }
  return;
}

//! \fn void RefinementCriteria::CheckForRefinement()
//! \brief Checks for refinement/de-refinement and sets refine_flag(m) for all
//! MeshBlocks within a MeshBlockPack.  Increments number of cycles since last refinement
//! counter for all MeshBlocks

void MeshRefinement::CheckForRefinement(MeshBlockPack* pmbp) {
  const int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  const int nmb = pmbp->nmb_thispack;

  // Reallocate when the mesh hierarchy size changes, but only clear the local flag slice.
  if (static_cast<int>(refine_flag.extent(0)) != pmy_mesh->nmb_total) {
    Kokkos::realloc(refine_flag, pmy_mesh->nmb_total);
  }
  if (static_cast<int>(refine_hold.extent(0)) != pmy_mesh->nmb_total) {
    Kokkos::realloc(refine_hold, pmy_mesh->nmb_total);
  }
  for (int m = 0; m < nmb; ++m) {
    refine_flag.h_view(m + mbs) = 0;
    refine_hold.h_view(m + mbs) = 0;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  refine_hold.template modify<HostMemSpace>();
  refine_hold.template sync<DevExeSpace>();

  pmrc->ResizeMeshBlockStorage(nmb);
  pmrc->SetRefinementData(pmbp, false, false);

  // calculate derived refinement variables
  if (pmrc->nderived > 0) {
    pmrc->SetRefinementData(pmbp, false, true);
  }

  // iterate through list of refinement criteria and apply methods
  bool user_criterion_requested = false;
  bool generic_after_user = false;
  for (auto it = pmrc->rcrit.begin(); it != pmrc->rcrit.end(); ++it) {
    switch (it->rmethod) {
      case RefCritMethod::min_max:
        generic_after_user = generic_after_user || user_criterion_requested;
        pmrc->CheckMinMax(pmbp, *it);
        break;
      case RefCritMethod::slope:
        generic_after_user = generic_after_user || user_criterion_requested;
        pmrc->CheckSlope(pmbp, *it);
        break;
      case RefCritMethod::second_deriv:
        generic_after_user = generic_after_user || user_criterion_requested;
        pmrc->CheckSecondDeriv(pmbp, *it);
        break;
      case RefCritMethod::location:
        generic_after_user = generic_after_user || user_criterion_requested;
        pmrc->CheckLocation(pmbp, *it);
        break;
      case RefCritMethod::user:
        user_criterion_requested = true;
        ApplyUserRefinementCriterion(pmy_mesh, pmbp);
        break;
      default:
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
        Kokkos::abort("Unknown refinement method requested in a <refinement_criteria>");
        break;
    }
  }
  if (generic_after_user) {
    // Reapply the user criterion last so problem-specific refinement locks can
    // override derefine requests made by later generic criteria.
    ApplyUserRefinementCriterion(pmy_mesh, pmbp);
  }
  // A criterion that would refine a block but cannot, because the block is already at
  // that criterion's maximum level, leaves the flag at 0.  Do not let another criterion
  // or the user hook turn that into a derefinement.
  refine_flag.template sync<HostMemSpace>();
  refine_hold.template sync<HostMemSpace>();
  for (int m = 0; m < nmb; ++m) {
    if (refine_hold.h_view(m + mbs) != 0 && refine_flag.h_view(m + mbs) < 0) {
      refine_flag.h_view(m + mbs) = 0;
    }
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  // Every criterion has been evaluated; drop the derived-variable array until the next
  // check instead of keeping 0.37 MB per block resident.  Kept resident when AMR runs
  // every few cycles (per-cycle device free/alloc churn is what the grow-only policies in
  // this file guard against).
  if (refinement_interval >= 16 && MeshBlockStorageReserve() == 0) {
    pmrc->ReleaseMeshBlockStorage();
  }

  long long pre_filter_refine = 0;
  long long pre_filter_derefine = 0;
  for (int m = 0; m < nmb; ++m) {
    const int flag = refine_flag.h_view(m + mbs);
    if (flag > 0) ++pre_filter_refine;
    if (flag < 0) ++pre_filter_derefine;
  }

  long long max_level_refine_cancel = 0;
  long long root_level_derefine_cancel = 0;
  // Turn off (on host) refine/derefine flag for MeshBlocks at max/root level
  for (int m=0; m<nmb; ++m) {
    if (pmy_mesh->lloc_eachmb[m+mbs].level == pmy_mesh->max_level) {
      if (refine_flag.h_view(m+mbs) > 0) {
        refine_flag.h_view(m+mbs) = 0;
        ++max_level_refine_cancel;
      }
    }
    if (pmy_mesh->lloc_eachmb[m+mbs].level == pmy_mesh->root_level) {
      if (refine_flag.h_view(m+mbs) < 0) {
        refine_flag.h_view(m+mbs) = 0;
        ++root_level_derefine_cancel;
      }
    }
  }
  long long cooldown_refine_cancel = 0;
  long long cooldown_derefine_cancel = 0;
  // Turn off (on host) refine/derefine flag for any MB that has been recently refined
  for (int m=0; m<nmb; ++m) {
    if (ncyc_since_ref(m+mbs) < refinement_interval) {
      if (refine_flag.h_view(m+mbs) > 0) ++cooldown_refine_cancel;
      if (refine_flag.h_view(m+mbs) < 0) ++cooldown_derefine_cancel;
      refine_flag.h_view(m+mbs) = 0;
    }
  }
  long long final_refine = 0;
  long long final_derefine = 0;
  for (int m = 0; m < nmb; ++m) {
    const int flag = refine_flag.h_view(m + mbs);
    if (flag > 0) ++final_refine;
    if (flag < 0) ++final_derefine;
  }
  if (pmy_mesh->hydro_lat_diagnostics) {
    long long local_diag[8] = {
        pre_filter_refine, pre_filter_derefine,
        max_level_refine_cancel, root_level_derefine_cancel,
        cooldown_refine_cancel, cooldown_derefine_cancel,
        final_refine, final_derefine};
    long long global_diag[8];
    for (int n = 0; n < 8; ++n) global_diag[n] = local_diag[n];
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(local_diag, global_diag, 8, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "MeshRefinement: AMR flags cycle=" << pmy_mesh->ncycle
                << " pre_refine=" << global_diag[0]
                << " pre_derefine=" << global_diag[1]
                << " max_refine_cancel=" << global_diag[2]
                << " root_derefine_cancel=" << global_diag[3]
                << " cooldown_refine_cancel=" << global_diag[4]
                << " cooldown_derefine_cancel=" << global_diag[5]
                << " final_refine=" << global_diag[6]
                << " final_derefine=" << global_diag[7]
                << std::endl;
    }
  }
  // The criteria above can update refine_flag on either device or host.  The final
  // max/root/recent-refinement and LAT cleanup edits are host-side, so publish them
  // before later AMR-cap checks call sync<HostMemSpace>().  Otherwise a stale device
  // copy can resurrect refinement requests that were just canceled here.
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  // Keep refine_flag local here. UpdateMeshBlockTree() gathers only the flagged logical
  // locations it actually needs, so an all-ranks copy of the full flag array is redundant
  // and expensive on large AMR meshes.
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::LimitRefinementToMemoryCap()
//! \brief Enforce the configured MeshBlock memory cap before modifying the AMR tree.

int MeshRefinement::PredictMeshBlockCountAfterAMR() {
  Mesh *pm = pmy_mesh;
  if (pm == nullptr) return 0;

  int nleaf = 2;
  if (pm->two_d) {nleaf = 4;}
  if (pm->three_d) {nleaf = 8;}

  refine_flag.template sync<HostMemSpace>();

  const int nranks = std::max(1, global_variable::nranks);
  std::vector<int> nref(nranks, 0), nderef(nranks, 0);
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pm->nmb_thisrank;
  for (int i=0; i<nmb; ++i) {
    const int gid = mbs + i;
    if (refine_flag.h_view(gid) == 1) ++nref[global_variable::my_rank];
    if (refine_flag.h_view(gid) == -1) ++nderef[global_variable::my_rank];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif

  int tnref = 0, tnderef = 0;
  for (int r=0; r<nranks; ++r) {
    tnref += nref[r];
    tnderef += nderef[r];
  }
  if (tnref == 0 && tnderef < nleaf) {
    return pm->nmb_total;
  }

  std::vector<int> nref_rsum(nranks, 0), nderef_rsum(nranks, 0);
  for (int r=1; r<nranks; ++r) {
    nref_rsum[r] = nref_rsum[r-1] + nref[r-1];
    nderef_rsum[r] = nderef_rsum[r-1] + nderef[r-1];
  }

  std::vector<LogicalLocation> llref(std::max(1, tnref));
  std::vector<LogicalLocation> llderef(std::max(1, tnderef));
  {
    int iref = nref_rsum[global_variable::my_rank];
    int ideref = nderef_rsum[global_variable::my_rank];
    for (int i=0; i<nmb; ++i) {
      const int gid = pm->pmb_pack->pmb->mb_gid.h_view(i);
      if (refine_flag.h_view(gid) == 1) {
        llref[iref++] = pm->lloc_eachmb[gid];
      } else if (refine_flag.h_view(gid) == -1 && tnderef >= nleaf) {
        llderef[ideref++] = pm->lloc_eachmb[gid];
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Datatype lloc_type;
  MPI_Type_contiguous(4, MPI_INT32_T, &lloc_type);
  MPI_Type_commit(&lloc_type);
  if (tnref > 0) {
    MPI_Allgatherv(MPI_IN_PLACE, nref[global_variable::my_rank], lloc_type,
                   llref.data(), nref.data(), nref_rsum.data(), lloc_type, MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf) {
    MPI_Allgatherv(MPI_IN_PLACE, nderef[global_variable::my_rank], lloc_type,
                   llderef.data(), nderef.data(), nderef_rsum.data(), lloc_type,
                   MPI_COMM_WORLD);
  }
  MPI_Type_free(&lloc_type);
#endif

  std::vector<LogicalLocation> cllderef(std::max(1, tnderef/nleaf));
  int ctnd = 0;
  if (tnderef >= nleaf) {
    if (pm->hydro_lat_gid_reordered) {
      // LAT GID interleaving breaks the original Z-order assumption that sibling
      // MeshBlocks are consecutive in GID order.  Group by parent before testing
      // complete derefine families.  Non-LAT meshes keep the original Z-order path.
      std::sort(llderef.begin(), llderef.begin() + tnderef,
                LogicalLocationLATDerefineOrder);
    }
    int lk = 0, lj = 0;
    if (pm->multi_d) lj = 1;
    if (pm->three_d) lk = 1;
    for (int n=0; n<tnderef; n++) {
      if ((llderef[n].lx1 & 1) == 0 &&
          (llderef[n].lx2 & 1) == 0 &&
          (llderef[n].lx3 & 1) == 0) {
        int r = n, rr = 0;
        for (std::int32_t k=0; k<=lk; k++) {
          for (std::int32_t j=0; j<=lj; j++) {
            for (std::int32_t i=0; i<=1; i++) {
              if (r < tnderef) {
                if ((llderef[n].lx1+i) == llderef[r].lx1 &&
                    (llderef[n].lx2+j) == llderef[r].lx2 &&
                    (llderef[n].lx3+k) == llderef[r].lx3 &&
                     llderef[n].level  == llderef[r].level) {
                  rr++;
                }
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          cllderef[ctnd].lx1   = llderef[n].lx1 >> 1;
          cllderef[ctnd].lx2   = llderef[n].lx2 >> 1;
          cllderef[ctnd].lx3   = llderef[n].lx3 >> 1;
          cllderef[ctnd].level = llderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  if (ctnd > 1) {
    std::sort(cllderef.begin(), cllderef.begin() + ctnd, Mesh::GreaterLevel);
  }

  MeshBlockTree trial(pm);
  trial.CreateRootGrid();
  for (int gid=0; gid<pm->nmb_total; ++gid) {
    trial.AddNodeWithoutRefinement(pm->lloc_eachmb[gid]);
  }

  int nnew = 0, ndel = 0;
  for (int n=0; n<tnref; ++n) {
    MeshBlockTree *bt = trial.FindMeshBlock(llref[n]);
    if (bt != nullptr) bt->Refine(nnew);
  }
  for (int n=0; n<ctnd; ++n) {
    MeshBlockTree *bt = trial.FindMeshBlock(cllderef[n]);
    if (bt != nullptr) bt->Derefine(ndel);
  }

  int predicted = 0;
  trial.CountMeshBlocks(predicted);
  pm->ptree->ActivateRoot();
  return predicted;
}

void MeshRefinement::LimitRefinementToMemoryCap() {
  Mesh *pm = pmy_mesh;
  if (pm == nullptr || pm->nmb_maxperrank <= 0) return;
  const int hard_cap = pm->nmb_maxperrank * std::max(1, global_variable::nranks);
  if (pm->nmb_total >= hard_cap) {
    refine_flag.template sync<HostMemSpace>();
    const int mbs = pm->gids_eachrank[global_variable::my_rank];
    const int nmb = pm->nmb_thisrank;
    int local_refine = 0;
    for (int m = 0; m < nmb; ++m) {
      const int gid = mbs + m;
      if (refine_flag.h_view(gid) == 1) {
        refine_flag.h_view(gid) = 0;
        ++local_refine;
      }
    }
    int global_refine = local_refine;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &global_refine, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_refine > 0) {
      refine_flag.template modify<HostMemSpace>();
      refine_flag.template sync<DevExeSpace>();
      if (global_variable::my_rank == 0) {
        std::cout << "MeshRefinement: AMR memory cap blocked " << global_refine
                  << " refinement requests because current MeshBlocks="
                  << pm->nmb_total << " already reaches cap " << hard_cap
                  << " (" << pm->nmb_maxperrank << "/rank)" << std::endl;
      }
    }
    return;
  }

  int nleaf = 2;
  if (pm->two_d) {nleaf = 4;}
  if (pm->three_d) {nleaf = 8;}
  const int refine_delta = nleaf - 1;
  if (refine_delta <= 0) return;

  refine_flag.template sync<HostMemSpace>();

  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pm->nmb_thisrank;
  int local_refine = 0;
  int local_derefine = 0;
  for (int m = 0; m < nmb; ++m) {
    const int gid = mbs + m;
    if (refine_flag.h_view(gid) == 1) {
      ++local_refine;
    } else if (refine_flag.h_view(gid) == -1) {
      ++local_derefine;
    }
  }

  int global_refine = local_refine;
  int global_derefine = local_derefine;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_refine, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_derefine, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Derefine flags are only realized when every sibling in a group is flagged, and
  // UpdateMeshBlockTree() may reject derefines to preserve the AMR level jump.  Do not
  // spend unproven derefine credit here; the cap must remain conservative.
  const int conservative_next = pm->nmb_total + global_refine * refine_delta;
  if (global_refine <= 0) return;

  const int direct_allowed_refine =
      std::max(0, (hard_cap - pm->nmb_total) / refine_delta);
  const int direct_to_cancel =
      (conservative_next > hard_cap) ? std::max(0,
          global_refine - direct_allowed_refine) : 0;

  struct RefineCandidate {
    int gid;
    int level;
    int ncyc;
  };
  std::vector<RefineCandidate> local_candidates;
  local_candidates.reserve(local_refine);
  for (int m = 0; m < nmb; ++m) {
    const int gid = mbs + m;
    if (refine_flag.h_view(gid) != 1) continue;
    local_candidates.push_back(
        {gid, pm->lloc_eachmb[gid].level, ncyc_since_ref(gid)});
  }

  std::vector<int> counts(global_variable::nranks, 0);
  counts[global_variable::my_rank] = static_cast<int>(local_candidates.size());
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif
  std::vector<int> displs(global_variable::nranks + 1, 0);
  for (int r = 0; r < global_variable::nranks; ++r) {
    displs[r + 1] = displs[r] + counts[r];
  }
  std::vector<int> sendbuf(3 * local_candidates.size(), 0);
  for (std::size_t n = 0; n < local_candidates.size(); ++n) {
    sendbuf[3*n    ] = local_candidates[n].gid;
    sendbuf[3*n + 1] = local_candidates[n].level;
    sendbuf[3*n + 2] = local_candidates[n].ncyc;
  }
  std::vector<int> recvcounts(global_variable::nranks, 0);
  std::vector<int> recvdispls(global_variable::nranks, 0);
  for (int r = 0; r < global_variable::nranks; ++r) {
    recvcounts[r] = 3 * counts[r];
    recvdispls[r] = 3 * displs[r];
  }
  std::vector<int> recvbuf(3 * displs.back(), 0);
#if MPI_PARALLEL_ENABLED
  MPI_Allgatherv(sendbuf.empty() ? nullptr : sendbuf.data(),
                 static_cast<int>(sendbuf.size()), MPI_INT,
                 recvbuf.empty() ? nullptr : recvbuf.data(),
                 recvcounts.data(), recvdispls.data(), MPI_INT,
                 MPI_COMM_WORLD);
#else
  recvbuf = sendbuf;
#endif
  std::vector<RefineCandidate> global_candidates;
  global_candidates.reserve(displs.back());
  for (int n = 0; n < displs.back(); ++n) {
    global_candidates.push_back(
        {recvbuf[3*n], recvbuf[3*n + 1], recvbuf[3*n + 2]});
  }
  std::sort(global_candidates.begin(), global_candidates.end(),
            [](const RefineCandidate &a, const RefineCandidate &b) {
              if (a.level != b.level) return a.level < b.level;
              if (a.ncyc != b.ncyc) return a.ncyc < b.ncyc;
              return a.gid > b.gid;
            });

  auto apply_cancel_count = [&](const int cancel_count) {
    for (const auto &candidate : local_candidates) {
      refine_flag.h_view(candidate.gid) = 1;
    }
    const int ncancel = std::min(cancel_count,
                                 static_cast<int>(global_candidates.size()));
    for (int n = 0; n < ncancel; ++n) {
      const int gid = global_candidates[n].gid;
      if (gid >= mbs && gid < mbs + nmb) {
        refine_flag.h_view(gid) = 0;
      }
    }
    refine_flag.template modify<HostMemSpace>();
  };

  int cancel_count = std::min(direct_to_cancel,
                              static_cast<int>(global_candidates.size()));
  apply_cancel_count(cancel_count);
  int predicted_nmb = PredictMeshBlockCountAfterAMR();
  if (predicted_nmb > hard_cap) {
    int lo = cancel_count;
    int hi = static_cast<int>(global_candidates.size());
    apply_cancel_count(hi);
    const int all_cancel_predicted = PredictMeshBlockCountAfterAMR();
    if (all_cancel_predicted > hard_cap) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "AMR memory cap cannot hold mesh even with all refinements canceled: "
                << "cap=" << hard_cap << ", predicted=" << all_cancel_predicted
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    predicted_nmb = all_cancel_predicted;
    while (lo + 1 < hi) {
      const int mid = lo + (hi - lo)/2;
      apply_cancel_count(mid);
      const int mid_predicted = PredictMeshBlockCountAfterAMR();
      if (mid_predicted > hard_cap) {
        lo = mid;
      } else {
        hi = mid;
        predicted_nmb = mid_predicted;
      }
    }
    cancel_count = hi;
    apply_cancel_count(cancel_count);
    predicted_nmb = PredictMeshBlockCountAfterAMR();
  }

  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  if (cancel_count > 0 && global_variable::my_rank == 0) {
    const int kept_refine = global_refine - cancel_count;
    std::cout << "MeshRefinement: AMR memory cap limited refinement requests from "
              << global_refine << " to " << kept_refine
              << " to keep predicted MeshBlocks=" << predicted_nmb
              << " <= " << hard_cap << " (" << pm->nmb_maxperrank << "/rank)"
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel)
//! \brief collect refinement flags and manipulate the MeshBlockTree with AMR
//! Returns total number of MBs refined/derefined in arguments.

void MeshRefinement::UpdateMeshBlockTree(int &nnew, int &ndel) {
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) {nleaf = 4;}
  if (pmy_mesh->three_d) {nleaf = 8;}

  // count the number of the blocks to be (de)refined on this rank
  nref_eachrank[global_variable::my_rank] = 0;
  nderef_eachrank[global_variable::my_rank] = 0;
  int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
    if (refine_flag.h_view(i+mbs) ==  1) nref_eachrank[global_variable::my_rank]++;
    if (refine_flag.h_view(i+mbs) == -1) nderef_eachrank[global_variable::my_rank]++;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref_eachrank,   1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#endif

  // count the number of the blocks to be (de)refined over all ranks
  int tnref = 0, tnderef = 0;
  for (int n=0; n<global_variable::nranks; n++) {
    tnref  += nref_eachrank[n];
    tnderef += nderef_eachrank[n];
  }
  // nothing to do (only derefine if all MeshBlocks within a leaf are flagged)
  if (tnref == 0 && tnderef < nleaf) {
    return;
  }

  // allocate memory for logical location arrays over total number MBs refined/derefined
  LogicalLocation *llref, *llderef, *cllderef;
  if (tnref > 0) {
    llref = new LogicalLocation[tnref];
  }
  if (tnderef >= nleaf) {
    llderef = new LogicalLocation[tnderef];
    cllderef = new LogicalLocation[tnderef/nleaf];
  }

  // calculate running sum of number of MBs to be refined/de-refined
  nref_rsum[0] = 0;
  nderef_rsum[0] = 0;
  for (int n=1; n<global_variable::nranks; n++) {
    nref_rsum[n] = nref_rsum[n-1] + nref_eachrank[n-1];
    nderef_rsum[n] = nderef_rsum[n-1] + nderef_eachrank[n-1];
  }

  // collect logical locations of MBs to be refined/derefined into arrays
  {
    int iref = nref_rsum[global_variable::my_rank];
    int ideref = nderef_rsum[global_variable::my_rank];
    for (int i=0; i<(pmy_mesh->nmb_thisrank); ++i) {
      int gid = pmy_mesh->pmb_pack->pmb->mb_gid.h_view(i);
      if (refine_flag.h_view(gid) ==  1) {
        llref[iref++] = pmy_mesh->lloc_eachmb[gid];;
      } else if (refine_flag.h_view(gid) == -1 && tnderef >= nleaf) {
        llderef[ideref++] = pmy_mesh->lloc_eachmb[gid];
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  // Now pass Logical Locations of MBs updated between all ranks.
  MPI_Datatype lloc_type;
  MPI_Type_contiguous(4, MPI_INT32_T, &lloc_type);
  MPI_Type_commit(&lloc_type);
  if (tnref > 0) {
    MPI_Allgatherv(MPI_IN_PLACE, nref_eachrank[global_variable::my_rank], lloc_type,
                   llref, nref_eachrank, nref_rsum, lloc_type, MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf) {
    MPI_Allgatherv(MPI_IN_PLACE, nderef_eachrank[global_variable::my_rank], lloc_type,
                   llderef, nderef_eachrank, nderef_rsum, lloc_type, MPI_COMM_WORLD);
  }
  MPI_Type_free(&lloc_type);
#endif

  // Each rank now has a complete list of the LLs of MBs refined/derefined on other ranks
  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    if (pmy_mesh->hydro_lat_gid_reordered) {
      // See PredictMeshBlockCountAfterAMR(): only LAT-reordered meshes need this sort.
      std::sort(llderef, llderef + tnderef, LogicalLocationLATDerefineOrder);
    }
    int lk = 0, lj = 0;
    if (pmy_mesh->multi_d) lj = 1;
    if (pmy_mesh->three_d) lk = 1;
    for (int n=0; n<tnderef; n++) {
      if ((llderef[n].lx1 & 1) == 0 &&
          (llderef[n].lx2 & 1) == 0 &&
          (llderef[n].lx3 & 1) == 0) {
        int r = n, rr = 0;
        for (std::int32_t k=0; k<=lk; k++) {
          for (std::int32_t j=0; j<=lj; j++) {
            for (std::int32_t i=0; i<=1; i++) {
              if (r < tnderef) {
                if ((llderef[n].lx1+i) == llderef[r].lx1 &&
                    (llderef[n].lx2+j) == llderef[r].lx2 &&
                    (llderef[n].lx3+k) == llderef[r].lx3 &&
                     llderef[n].level  == llderef[r].level) {
                  rr++;
                }
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          cllderef[ctnd].lx1   = llderef[n].lx1 >> 1;
          cllderef[ctnd].lx2   = llderef[n].lx2 >> 1;
          cllderef[ctnd].lx3   = llderef[n].lx3 >> 1;
          cllderef[ctnd].level = llderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) {
    std::sort(cllderef, cllderef + ctnd, Mesh::GreaterLevel);
  }

  if (tnderef >= nleaf) {
    delete [] llderef;
  }

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation.  Note all ranks manipulate entire tree, so each rank has
  // a complete and updated copy of the entire tree.
  // Step 1. perform refinement
  for (int n=0; n<tnref; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(llref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0) {
    delete [] llref;
  }

  // Step 2. perform derefinement
  for (int n=0; n<ctnd; n++) {
    MeshBlockTree *bt = pmy_mesh->ptree->FindMeshBlock(cllderef[n]);
    bt->Derefine(ndel);
  }

  if (pmy_mesh->hydro_lat_diagnostics && global_variable::my_rank == 0) {
    std::cout << "MeshRefinement: AMR tree cycle=" << pmy_mesh->ncycle
              << " raw_refine=" << tnref
              << " raw_derefine=" << tnderef
              << " complete_derefine_groups=" << ctnd
              << " realized_created=" << nnew
              << " realized_deleted=" << ndel
              << std::endl;
  }

  if (tnderef >= nleaf) {
    delete [] cllderef;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RedistAndRefineMeshBlocks()
//! \brief redistribute MeshBlocks according to the new load balance
//! This requires moving data within the evolved variable arrays for each Physics (e.g.,
//! hydro, mhd, radiation) both within a rank (using deep copies) and potentially between
//! ranks (using MPI calls), and applying restriction and prolongation operators as
//! required. It also requires rebuilding the MB data arrays, coordinates, and neighbors.
//! Boundary values and primitives are set in calling function: AdaptiveMeshRefinement()

void MeshRefinement::RedistAndRefineMeshBlocks(ParameterInput *pin, int nnew, int ndel) {
  Mesh* pm = pmy_mesh;
  int old_nmb = pm->nmb_total;
  int new_nmb = old_nmb + nnew - ndel;
  const bool pure_rebalance = (nnew == 0 && ndel == 0);
  lb_full_block_transfer = pure_rebalance;
  const bool preserve_lat_metadata =
      pure_rebalance && pm->hydro_lat_metadata_valid &&
      pm->hydro_lat_factor_eachmb != nullptr &&
      pm->hydro_lat_dt_eachmb != nullptr &&
      pm->hydro_lat_metadata_nmb == old_nmb;
  const int old_hydro_lat_sync_factor = pm->hydro_lat_sync_factor_current;
  const bool old_hydro_lat_dt_limited = pm->hydro_lat_dt_limited_by_hydro;
  std::vector<int> old_hydro_lat_factor;
  std::vector<Real> old_hydro_lat_dt;
  std::vector<Real> old_hydro_lat_work;
  if (preserve_lat_metadata) {
    old_hydro_lat_factor.assign(pm->hydro_lat_factor_eachmb,
                                pm->hydro_lat_factor_eachmb + old_nmb);
    old_hydro_lat_dt.assign(pm->hydro_lat_dt_eachmb,
                            pm->hydro_lat_dt_eachmb + old_nmb);
    if (pm->hydro_lat_work_eachmb != nullptr) {
      old_hydro_lat_work.assign(pm->hydro_lat_work_eachmb,
                                pm->hydro_lat_work_eachmb + old_nmb);
    } else {
      old_hydro_lat_work.assign(old_nmb, 1.0);
    }
  }
  bool target_gid_reordered = pm->hydro_lat_gid_reordered;
  if (!pure_rebalance && pm->nmb_maxperrank > 0) {
    const int hard_cap = pm->nmb_maxperrank * std::max(1, global_variable::nranks);
    if (new_nmb > hard_cap) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "AMR memory cap would be exceeded before data migration: cap="
                  << hard_cap << ", requested MeshBlocks=" << new_nmb << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }
  // compute nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pm->two_d) nleaf = 4;
  if (pm->three_d) nleaf = 8;

  // Step 1. Create the candidate logical-location list and new->old GID map.
  // Real AMR starts from tree traversal order, then LAT can interleave that candidate
  // before load balance and state-transfer maps are built.
  new_lloc_eachmb = new LogicalLocation[new_nmb];
  newtoold = new int[new_nmb];
  int new_nmb_total;
  std::vector<float> candidate_cost_eachmb;
  std::vector<int> candidate_lat_factor_eachmb;
  int candidate_lat_sync_factor = 1;
  int candidate_max_blocks_per_rank = 0;
  bool candidate_lat_costs_valid = false;
  // A regrid with no LAT metadata is the owed pass of a restart (Driver::Initialize
  // Step 1b), which runs before this run has built any: every Execute pass follows a
  // RebuildHydroLATMetadata and a pure rebalance requires valid metadata.  Tell the
  // estimator so, as BuildTreeFromRestart does, so it keeps a flat ladder instead of
  // inventing 2^(max_level - level) factors; the first metadata rebuild installs the real
  // ladder.
  const bool lat_costs_measured = !pm->hydro_lat_metadata_valid;
  if (pure_rebalance) {
    std::vector<int> requested_gid_map(new_nmb, 0);
    const int *lat_factors = (pm->hydro_lat_metadata_valid &&
                              pm->hydro_lat_factor_eachmb != nullptr &&
                              pm->hydro_lat_metadata_nmb == old_nmb) ?
        pm->hydro_lat_factor_eachmb : nullptr;
    const int lat_sync_factor = std::max(1, pm->hydro_lat_sync_factor_current);
    const Real *lat_work =
        (lat_factors != nullptr) ? pm->hydro_lat_work_eachmb : nullptr;
    const bool use_gid_map =
        pm->BuildHydroLATGIDMap(pin, pm->lloc_eachmb, old_nmb, lat_factors,
                                lat_sync_factor, requested_gid_map.data(),
                                false, lat_work);
    if (use_gid_map) target_gid_reordered = true;
    for (int n=0; n<new_nmb; ++n) {
      const int old_gid = use_gid_map ? requested_gid_map[n] : n;
      new_lloc_eachmb[n] = pm->lloc_eachmb[old_gid];
      newtoold[n] = old_gid;
    }
    new_nmb_total = new_nmb;
  } else {
    pm->ptree->CreateZOrderedLLList(new_lloc_eachmb, newtoold, new_nmb_total);
    target_gid_reordered = false;
    const bool hydro_lat_enabled = (pin != nullptr && pin->IsLATEnabled());
    const bool gid_reorder_enabled = hydro_lat_enabled &&
        pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true);
    if (gid_reorder_enabled) {
      candidate_cost_eachmb.assign(new_nmb_total, 1.0f);
      candidate_lat_factor_eachmb.assign(new_nmb_total, 0);
      candidate_max_blocks_per_rank =
          pm->ApplyHydroLATLoadBalanceCosts(pin, new_lloc_eachmb,
                                            candidate_cost_eachmb.data(),
                                            new_nmb_total,
                                            candidate_lat_factor_eachmb.data(),
                                            &candidate_lat_sync_factor,
                                            lat_costs_measured);
      candidate_lat_costs_valid = true;
      std::vector<int> requested_gid_map(new_nmb_total, 0);
      std::vector<Real> candidate_work(new_nmb_total, 1.0);
      for (int gid=0; gid<new_nmb_total; ++gid) {
        candidate_work[gid] = static_cast<Real>(candidate_cost_eachmb[gid])*
            static_cast<Real>(std::max(1, candidate_lat_factor_eachmb[gid]))/
            static_cast<Real>(std::max(1, candidate_lat_sync_factor));
      }
      const bool use_gid_map =
          pm->BuildHydroLATGIDMap(pin, new_lloc_eachmb, new_nmb_total,
                                  candidate_lat_factor_eachmb.data(),
                                  candidate_lat_sync_factor, requested_gid_map.data(),
                                  false, candidate_work.data());
      if (use_gid_map) {
        std::vector<LogicalLocation> old_lloc(new_lloc_eachmb,
                                              new_lloc_eachmb + new_nmb_total);
        std::vector<int> old_newtoold(newtoold, newtoold + new_nmb_total);
        const std::vector<float> old_candidate_cost(candidate_cost_eachmb);
        const std::vector<int> old_candidate_factor(candidate_lat_factor_eachmb);
        for (int gid=0; gid<new_nmb_total; ++gid) {
          const int old_gid = requested_gid_map[gid];
          if (old_gid < 0 || old_gid >= new_nmb_total) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl
                      << "LAT AMR GID map references invalid old gid=" << old_gid
                      << " at new gid=" << gid << std::endl;
            std::exit(EXIT_FAILURE);
          }
          new_lloc_eachmb[gid] = old_lloc[old_gid];
          newtoold[gid] = old_newtoold[old_gid];
          candidate_cost_eachmb[gid] = old_candidate_cost[old_gid];
          candidate_lat_factor_eachmb[gid] = old_candidate_factor[old_gid];
        }
        target_gid_reordered = true;
      }
    }
  }
  if (new_nmb_total != new_nmb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of MeshBlocks in new tree = " << new_nmb_total << " but expected "
        << "value = " << new_nmb << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Step 2.  Create oldtonew list mapping the previous gid to the current one for all MBs
  // Index of array is old gid, value is new gid.
  oldtonew = new int[old_nmb];
  for (int oldm=0; oldm<old_nmb; ++oldm) oldtonew[oldm] = -1;
  if (!pure_rebalance) {
    std::unordered_map<AMRLogicalLocationKey, int, AMRLogicalLocationKeyHash>
        new_gid_by_location;
    new_gid_by_location.reserve(static_cast<std::size_t>(new_nmb)*2);
    for (int newm=0; newm<new_nmb; ++newm) {
      new_gid_by_location.emplace(AMRLogicalLocationKey(new_lloc_eachmb[newm]), newm);
      const int oldm = newtoold[newm];
      if (oldm < 0 || oldm >= old_nmb) continue;
      if (oldtonew[oldm] < 0 || newm < oldtonew[oldm]) oldtonew[oldm] = newm;
    }
    for (int oldm=0; oldm<old_nmb; ++oldm) {
      if (oldtonew[oldm] >= 0) continue;
      AMRLogicalLocationKey ancestor(pm->lloc_eachmb[oldm]);
      while (ancestor.level > pm->root_level) {
        ancestor = AMRParentKey(ancestor);
        const auto it = new_gid_by_location.find(ancestor);
        if (it != new_gid_by_location.end()) {
          oldtonew[oldm] = it->second;
          break;
        }
      }
    }
    for (int oldm=0; oldm<old_nmb; ++oldm) {
      if (oldtonew[oldm] < 0 || oldtonew[oldm] >= new_nmb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "AMR old-to-new GID map is incomplete for old gid=" << oldm
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  deref_old_gid_each_new.assign(static_cast<std::size_t>(new_nmb)*nleaf, -1);
  refine_new_gid_each_old.assign(static_cast<std::size_t>(old_nmb)*nleaf, -1);
  auto child_slot = [&](const LogicalLocation &lloc) {
    const int ox1 = ((lloc.lx1 & 1) == 1);
    const int ox2 = (pm->multi_d && ((lloc.lx2 & 1) == 1));
    const int ox3 = (pm->three_d && ((lloc.lx3 & 1) == 1));
    return ox1 + (ox2 << 1) + (ox3 << 2);
  };
  for (int newm=0; newm<new_nmb; ++newm) {
    const int oldm = newtoold[newm];
    if (oldm < 0 || oldm >= old_nmb) continue;
    const LogicalLocation &old_lloc = pm->lloc_eachmb[oldm];
    const LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {
      refine_new_gid_each_old[oldm*nleaf + child_slot(new_lloc)] = newm;
    }
  }
  for (int oldm=0; oldm<old_nmb; ++oldm) {
    const int newm = oldtonew[oldm];
    if (newm < 0 || newm >= new_nmb) continue;
    const LogicalLocation &old_lloc = pm->lloc_eachmb[oldm];
    const LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {
      deref_old_gid_each_new[newm*nleaf + child_slot(old_lloc)] = oldm;
    }
  }
  // Step 3.
  // Calculate new load balance. Start from equal per-block costs; LAT, when enabled,
  // overwrites these with timestep-bin costs below.
  new_cost_eachmb = new float[new_nmb];
  new_rank_eachmb = new int[new_nmb];
  new_gids_eachrank = new int[global_variable::nranks];
  new_nmb_eachrank = new int[global_variable::nranks];

  for (int i=0; i<new_nmb; i++) {new_cost_eachmb[i] = 1.0;}
  std::vector<int> lat_factor_eachmb(new_nmb_total, 0);
  int lat_sync_factor = 1;
  int max_blocks_per_rank = 0;
  if (candidate_lat_costs_valid) {
    std::copy(candidate_cost_eachmb.begin(), candidate_cost_eachmb.end(),
        new_cost_eachmb);
    lat_factor_eachmb = std::move(candidate_lat_factor_eachmb);
    lat_sync_factor = candidate_lat_sync_factor;
    max_blocks_per_rank = candidate_max_blocks_per_rank;
  } else {
    max_blocks_per_rank =
        pm->ApplyHydroLATLoadBalanceCosts(pin, new_lloc_eachmb, new_cost_eachmb,
                                          new_nmb_total, lat_factor_eachmb.data(),
                                          &lat_sync_factor, lat_costs_measured);
  }
  if (pure_rebalance) {
    for (int newm=0; newm<new_nmb_total; ++newm) {
      const int oldm = newtoold[newm];
      if (oldm < 0 || oldm >= old_nmb || oldtonew[oldm] >= 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "Pure rebalance GID map is not a permutation at new gid="
                  << newm << ", old gid=" << oldm << std::endl;
        std::exit(EXIT_FAILURE);
      }
      oldtonew[oldm] = newm;
    }
  }

  bool used_sticky_load_balance = false;
  const bool hydro_lat_rebalance =
      (max_blocks_per_rank > 0 && lat_sync_factor > 1);
  if (sticky_load_balance && !(hydro_lat_rebalance)) {
    for (int n=0; n<global_variable::nranks; ++n) {
      new_nmb_eachrank[n] = 0;
      new_gids_eachrank[n] = 0;
    }
    bool valid_sticky = true;
    for (int m=0; m<new_nmb_total; ++m) {
      const int oldm = newtoold[m];
      if (oldm < 0 || oldm >= old_nmb) {
        valid_sticky = false;
        break;
      }
      const int rank = pm->rank_eachmb[oldm];
      if (rank < 0 || rank >= global_variable::nranks) {
        valid_sticky = false;
        break;
      }
      new_rank_eachmb[m] = rank;
      new_nmb_eachrank[rank]++;
      if (m > 0 && new_rank_eachmb[m] < new_rank_eachmb[m - 1]) {
        valid_sticky = false;
        break;
      }
    }
    for (int n=0; n<global_variable::nranks && valid_sticky; ++n) {
      const int sticky_cap =
          (max_blocks_per_rank > 0) ? max_blocks_per_rank : pm->nmb_maxperrank;
      if (new_nmb_eachrank[n] <= 0 || new_nmb_eachrank[n] > sticky_cap) {
        valid_sticky = false;
      }
    }
    if (valid_sticky) {
      new_gids_eachrank[0] = 0;
      for (int n=1; n<global_variable::nranks; ++n) {
        new_gids_eachrank[n] = new_gids_eachrank[n - 1] + new_nmb_eachrank[n - 1];
      }
      used_sticky_load_balance = true;
    }
  }
  if (!used_sticky_load_balance) {
    pm->LoadBalance(new_cost_eachmb, new_rank_eachmb, new_gids_eachrank, new_nmb_eachrank,
                    new_nmb_total, max_blocks_per_rank, lat_factor_eachmb.data(),
                    lat_sync_factor);
  }

  if (pure_rebalance && hydro_lat_rebalance && preserve_lat_metadata) {
    pm->hydro_lat_lb_last_attempt_cycle = pm->ncycle;
    // Both partitions are scored with the same measured work, the candidate's read
    // back from its window cost (cost*factor/sync = the per-step work in the new order).
    std::vector<Real> candidate_work(new_nmb_total, 1.0);
    for (int gid=0; gid<new_nmb_total; ++gid) {
      candidate_work[gid] = static_cast<Real>(new_cost_eachmb[gid])*
          static_cast<Real>(std::max(1, lat_factor_eachmb[gid]))/
          static_cast<Real>(std::max(1, lat_sync_factor));
    }
    const LATBalanceScore current_score =
        ScoreLATPartition(old_hydro_lat_factor.data(), old_hydro_lat_sync_factor,
                          pm->gids_eachrank, pm->nmb_eachrank,
                          global_variable::nranks, old_hydro_lat_work.data());
    const LATBalanceScore candidate_score =
        ScoreLATPartition(lat_factor_eachmb.data(), lat_sync_factor,
                          new_gids_eachrank, new_nmb_eachrank,
                          global_variable::nranks, candidate_work.data());
    bool current_cap_violated = false;
    for (int rank=0; rank<global_variable::nranks; ++rank) {
      current_cap_violated = current_cap_violated ||
          (max_blocks_per_rank > 0 && pm->nmb_eachrank[rank] > max_blocks_per_rank);
    }
    const double sum_scale = std::max(1.0, current_score.sum_tick_max_work);
    const double peak_scale = std::max(1.0, current_score.max_tick_work);
    const double sum_improvement =
        (current_score.sum_tick_max_work - candidate_score.sum_tick_max_work)/sum_scale;
    const double peak_improvement =
        (current_score.max_tick_work - candidate_score.max_tick_work)/peak_scale;
    const double eps = 64.0*std::numeric_limits<double>::epsilon()*sum_scale;
    const bool candidate_no_worse =
        candidate_score.sum_tick_max_work <= current_score.sum_tick_max_work + eps;
    const bool migration_worthwhile = current_cap_violated ||
        (candidate_no_worse &&
         std::max(sum_improvement, peak_improvement) >=
             Mesh::kHydroLATRebalanceMinRelativeGain);

    if (!migration_worthwhile) {
      pm->StampHydroLATLoadBalanceVersions();
      delete [] new_lloc_eachmb;
      delete [] newtoold;
      delete [] oldtonew;
      delete [] new_cost_eachmb;
      delete [] new_rank_eachmb;
      delete [] new_gids_eachrank;
      delete [] new_nmb_eachrank;
      new_lloc_eachmb = nullptr;
      newtoold = nullptr;
      oldtonew = nullptr;
      new_cost_eachmb = nullptr;
      new_rank_eachmb = nullptr;
      new_gids_eachrank = nullptr;
      new_nmb_eachrank = nullptr;
      deref_old_gid_each_new.clear();
      refine_new_gid_each_old.clear();
      return;
    }
  }

  if (pm->pmb_pack != nullptr) {
    pm->pmb_pack->ReleaseLATCacheMemory();
  }
  if (new_nmb_eachrank[global_variable::my_rank] > pm->nmb_maxperrank) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Number of MeshBlocks in this rank on new tree = "
        << new_nmb_eachrank[global_variable::my_rank] << " on rank = "
        << global_variable::my_rank <<" exceeds <mesh_refinement>/max_nmb_per_rank = "
        << pm->nmb_maxperrank << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // UpdateMeshBlockTree function can refine/de-refine MBs to ensure resolution jump is
  // no more than 2x at boundaries, even if refine flag not set in these MBs.  So loop
  // over entire list of MBs on all ranks, reset refine_flag
  if (!pure_rebalance) {
    for (int oldm=0; oldm<old_nmb; oldm++) {
      int newm = oldtonew[oldm];
      LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
      LogicalLocation &new_lloc = new_lloc_eachmb[newm];
      if (old_lloc.level > new_lloc.level) {          // old MB was de-refined
        refine_flag.h_view(oldm) = -nleaf;
      } else if (old_lloc.level < new_lloc.level) {   // old MB was refined
        refine_flag.h_view(oldm) = 1;
      } else {
        refine_flag.h_view(oldm) = 0;
      }
    }
    // All ranks have copy of refine_flag over all MBs. So just sync host view with device
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
  }

  const int new_nmb_local = new_nmb_eachrank[global_variable::my_rank];
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  gravity::Gravity* pgrav = pm->pmb_pack->pgrav;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  adm::ADM* padm = pm->pmb_pack->padm;
  if (ndel > 0 && pmhd != nullptr) {
    // Refresh every coarse FC snapshot before migration.  A topology transaction is a
    // global synchronization point, so this must not inherit the current LAT subset.
    RestrictFC(pmhd->b0, pmhd->coarse_b0);
  }
  if (phydro != nullptr) {
    phydro->ResizeMeshBlockStorage(new_nmb_local);
  }
  if (pmhd != nullptr) {
    pmhd->ResizeMeshBlockStorage(new_nmb_local);
  }
  if (prad != nullptr) {
    prad->ResizeMeshBlockStorage(new_nmb_local);
  }
  if (pgrav != nullptr) {
    pgrav->ResizeMeshBlockStorage(new_nmb_local);
  }

  // Step 4.
  // Allocate send/recv buffers for load balancing, post receives.
  // Pack send buffers for load blancing and send data
#if MPI_PARALLEL_ENABLED
  InitRecvAMR(nleaf);
  PackAndSendAMR(nleaf);
  nmb_sent_thisrank += nmb_send;
#endif

  // Step 5.
  // De-refine (restrict) evolved physics variables for MeshBlocks within this rank.
  // Simply copies data from coarse arrays in source MBs to appropriate octant of fine
  // array in target MB.
  // derefine (if needed)
  if (ndel > 0) {
    if (phydro != nullptr) {
      DerefineCCSameRank(phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      DerefineCCSameRank(pmhd->u0, pmhd->coarse_u0);
      DerefineFCSameRank(pmhd->b0, pmhd->coarse_b0);
    }
    if (prad != nullptr) {
      DerefineCCSameRank(prad->i0, prad->coarse_i0);
    }
    if (pgrav != nullptr) {
      DerefineCCSameRank(pgrav->phi, pgrav->coarse_phi);
    }
    if (pz4c != nullptr) {
      DerefineCCSameRank(pz4c->u0, pz4c->coarse_u0);
    }
  }

  // Step 6.
  // Copy evolved physics variables to new MB index within View for MeshBlocks that stay
  // within this rank
  if (phydro != nullptr) {
    CopyCC(phydro->u0);
    if (lb_full_block_transfer) CopyCC(phydro->w0);
  }
  if (pmhd != nullptr) {
    CopyCC(pmhd->u0);
    CopyFC(pmhd->b0);
  }
  if (prad != nullptr) {
    CopyCC(prad->i0);
  }
  if (pgrav != nullptr && lb_transfer_gravity) {
    CopyCC(pgrav->phi);
  }
  if (pz4c != nullptr) {
    CopyCC(pz4c->u0);
  } else if (padm != nullptr && padm->StoresMetricGrid()) {
    CopyCC(padm->u_adm);
  }
  // Step 7.
  // Copy evolved physics variables for MBs flagged for refinement from source fine array
  // to target coarse array, when both are on same rank.
  if (nnew > 0) {
    if (phydro != nullptr) {
      CopyForRefinementCC(phydro->u0, phydro->coarse_u0);
    }
    if (pmhd != nullptr) {
      CopyForRefinementCC(pmhd->u0, pmhd->coarse_u0);
      CopyForRefinementFC(pmhd->b0, pmhd->coarse_b0);
    }
    if (prad != nullptr) {
      CopyForRefinementCC(prad->i0, prad->coarse_i0);
    }
    if (pgrav != nullptr) {
      CopyForRefinementCC(pgrav->phi, pgrav->coarse_phi);
    }
    if (pz4c != nullptr) {
      CopyForRefinementCC(pz4c->u0, pz4c->coarse_u0);
    }
  }

  // Step 8.
  // Wait for all MPI load balancing communications to finish.  Unpack data.
#if MPI_PARALLEL_ENABLED
  if (nmb_recv > 0) {ClearRecvAndUnpackAMR();}
  if (nmb_send > 0) {ClearSendAMR();}
#endif

  // copy newtoold array to DualView so that it can be accessed in kernel
  DualArray1D<int> new_to_old("newtoold",new_nmb_total);
  for (int m=0; m<new_nmb_total; ++m) {
    new_to_old.h_view(m) = newtoold[m];
  }
  new_to_old.template modify<HostMemSpace>();
  new_to_old.template sync<DevExeSpace>();

  DualArray1D<LogicalLocation> new_lloc("new_lloc", new_nmb_total);
  for (int m=0; m<new_nmb_total; ++m) {
    new_lloc.h_view(m) = new_lloc_eachmb[m];
  }
  new_lloc.template modify<HostMemSpace>();
  new_lloc.template sync<DevExeSpace>();

  const auto &indcs = pmy_mesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int cis = indcs.cis;
  const int cie = indcs.cie;
  const int cjs = indcs.cjs;
  const int cje = indcs.cje;
  const int cks = indcs.cks;
  const int cke = indcs.cke;
  const bool multi_d = pmy_mesh->multi_d;
  const bool three_d = pmy_mesh->three_d;
  const int cprol_il = cis - 1;
  const int cprol_iu = cie + 1;
  const int cprol_jl = multi_d ? (cjs - 1) : cjs;
  const int cprol_ju = multi_d ? (cje + 1) : cje;
  const int cprol_kl = three_d ? (cks - 1) : cks;
  const int cprol_ku = three_d ? (cke + 1) : cke;
  const int new_gids_local = new_gids_eachrank[global_variable::my_rank];
  const int mesh_cis = indcs.cis;
  const int mesh_cjs = indcs.cjs;
  const int mesh_cks = indcs.cks;
  const int mesh_cnx1 = indcs.cnx1;
  const int mesh_cnx2 = indcs.cnx2;
  const int mesh_cnx3 = indcs.cnx3;
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real bhx = 0.0;
  Real bhy = 0.0;
  Real bhz = 0.0;
  problem_runtime::GetExcisionState(pmy_mesh->time, excise_enabled, excise_radius,
                                    excise_density, excise_eint, bhx, bhy, bhz);
  const Real excise_r2 = excise_radius * excise_radius;
  const bool regrid_is_sr = (pm->pmb_pack->pcoord != nullptr) &&
      pm->pmb_pack->pcoord->is_special_relativistic;
  const bool regrid_is_gr = (pm->pmb_pack->pcoord != nullptr) &&
      pm->pmb_pack->pcoord->is_general_relativistic;
  const bool regrid_is_dynamical_gr = (pm->pmb_pack->pcoord != nullptr) &&
      pm->pmb_pack->pcoord->is_dynamical_relativistic;
  const CoordData regrid_coord = (pm->pmb_pack->pcoord != nullptr) ?
      pm->pmb_pack->pcoord->coord_data : CoordData{};
  if (nnew > 0 && prolong_prims && regrid_is_dynamical_gr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "mesh_refinement/prolong_primitives is not implemented for "
              << "dynamical GRMHD AMR regrids." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (nnew > 0 && prolong_prims && phydro != nullptr) {
    // AMR regrids must refine the same primitive state used at fine/coarse boundaries.
    // Restrict this conversion to the newly refined blocks. A whole-pack u->w->u
    // round-trip is not identity for dual-energy hydro and can imprint AMR structure onto
    // unchanged MeshBlocks during regrid.
    RefinedHydroCoarseConsToPrim(new_to_old.d_view, refine_flag.d_view, new_lloc.d_view,
                                 new_gids_local, new_nmb_local, phydro->coarse_u0,
                                 phydro->coarse_w0, pm->mesh_size, pm->root_level,
                                 pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3, multi_d,
                                 three_d, phydro->peos->eos_data, phydro->nhydro,
                                 phydro->nscalars, phydro->dual_energy_pdv,
                                 phydro->dual_energy_idx, phydro->dual_energy_eta1,
                                 regrid_is_sr, regrid_is_gr, regrid_coord,
                                 excise_enabled, excise_r2, excise_density, excise_eint,
                                 bhx, bhy, bhz, mesh_cis, mesh_cjs, mesh_cks, mesh_cnx1,
                                 mesh_cnx2, mesh_cnx3, cprol_il, cprol_iu,
                                 cprol_jl, cprol_ju, cprol_kl, cprol_ku);
  }
  if (nnew > 0 && prolong_prims && pmhd != nullptr) {
    // Match the MHD AMR regrid path to the existing primitive-boundary prolongation path.
    // Without this, dual-energy and Saha EOS MHD runs refine in primitives at boundaries
    // but in conserved variables during regrid, which breaks AMR consistency.
    RefinedMHDCoarseConsToPrim(new_to_old.d_view, refine_flag.d_view, new_lloc.d_view,
                               new_gids_local, new_nmb_local, pmhd->coarse_u0,
                               pmhd->coarse_b0, pmhd->coarse_w0, pm->mesh_size,
                               pm->root_level, pm->nmb_rootx1, pm->nmb_rootx2,
                               pm->nmb_rootx3, multi_d, three_d, pmhd->peos->eos_data,
                               pmhd->nmhd, pmhd->nscalars, pmhd->dual_energy_pdv,
                               pmhd->dual_energy_idx, pmhd->dual_energy_eta1,
                               regrid_is_sr, regrid_is_gr, regrid_coord,
                               excise_enabled, excise_r2, excise_density, excise_eint,
                               bhx, bhy, bhz, mesh_cis, mesh_cjs, mesh_cks, mesh_cnx1,
                               mesh_cnx2, mesh_cnx3, cprol_il, cprol_iu,
                               cprol_jl, cprol_ju, cprol_kl, cprol_ku);
  }

  // Step 9.
  // Coarse arrays are now up-to-date, either through copies on same rank or MPI calls
  // So prolongate (refine) evolved physics variables for all MBs flagged for refinement.

  if (nnew > 0) {
    if (phydro != nullptr) {
      if (prolong_prims) {
        RefineCC(new_to_old, phydro->w0, phydro->coarse_w0);
        RefinedHydroPrimToCons(new_to_old.d_view, refine_flag.d_view, new_lloc.d_view,
                               new_gids_local, new_nmb_local, phydro->w0, phydro->u0,
                               pm->mesh_size, pm->root_level, pm->nmb_rootx1,
                               pm->nmb_rootx2, pm->nmb_rootx3, multi_d, three_d,
                               phydro->peos->eos_data,
                               phydro->nhydro, phydro->nscalars, phydro->dual_energy_pdv,
                               phydro->dual_energy_idx, regrid_is_sr, regrid_is_gr,
                               regrid_coord, excise_enabled, excise_r2,
                               excise_density, excise_eint, bhx, bhy, bhz, is,
                               js, ks, indcs.nx1, indcs.nx2, indcs.nx3,
                               is, ie, js, je, ks, ke);
      } else {
        RefineCC(new_to_old, phydro->u0, phydro->coarse_u0);
      }
      if (phydro->use_dual_energy) {
        phydro->RepairRefinedDualEnergyState(new_to_old, refine_flag,
                                            new_gids_local, new_nmb_local);
      }
    }
    if (pmhd != nullptr) {
      RefineFC(new_to_old, pmhd->b0, pmhd->coarse_b0);
      if (prolong_prims) {
        RefineCC(new_to_old, pmhd->w0, pmhd->coarse_w0);
        RefinedMHDPrimToCons(new_to_old.d_view, refine_flag.d_view, new_lloc.d_view,
                             new_gids_local, new_nmb_local, pmhd->w0, pmhd->b0, pmhd->u0,
                             pm->mesh_size, pm->root_level, pm->nmb_rootx1, pm->nmb_rootx2,
                             pm->nmb_rootx3, multi_d, three_d, pmhd->peos->eos_data,
                             pmhd->nmhd, pmhd->nscalars, pmhd->use_dual_energy,
                             pmhd->dual_energy_idx, regrid_is_sr, regrid_is_gr,
                             regrid_coord,
                             excise_enabled, excise_r2,
                             excise_density, excise_eint, bhx, bhy, bhz,
                             is, js, ks, indcs.nx1, indcs.nx2, indcs.nx3,
                             is, ie, js, je, ks, ke);
      } else {
        RefineCC(new_to_old, pmhd->u0, pmhd->coarse_u0);
      }
      if (pmhd->use_dual_energy) {
        pmhd->RepairRefinedDualEnergyState(new_to_old, refine_flag,
                                          new_gids_eachrank[global_variable::my_rank],
                                          new_nmb_eachrank[global_variable::my_rank]);
      }
    }
    if (prad != nullptr) {
      RefineCC(new_to_old, prad->i0, prad->coarse_i0);
    }
    if (pgrav != nullptr) {
      RefineCC(new_to_old, pgrav->phi, pgrav->coarse_phi);
    }
    if (pz4c != nullptr) {
      RefineCC(new_to_old, pz4c->u0, pz4c->coarse_u0, true);
    }
  }

  // Step 10.
  // General housekeeping
  // Update new number of cycles since refinement
  HostArray1D<int> new_ncyc_since_ref("nnref",new_nmb_total);
  for (int m=0; m<(new_nmb_total); ++m) {
    int oldm = newtoold[m];
    if (!pure_rebalance && refine_flag.h_view(oldm) != 0) {
      new_ncyc_since_ref(m) = 0;
    } else {
      new_ncyc_since_ref(m) = ncyc_since_ref(oldm);
    }
  }
  Kokkos::realloc(ncyc_since_ref, new_nmb_total);
  Kokkos::deep_copy(ncyc_since_ref, new_ncyc_since_ref);

  // Update data in Mesh/MeshBlockPack/MeshBlock classes with new grid properties
  delete [] pm->lloc_eachmb;
  delete [] pm->rank_eachmb;
  delete [] pm->cost_eachmb;
  delete [] pm->gids_eachrank;
  delete [] pm->nmb_eachrank;
  pm->lloc_eachmb = new_lloc_eachmb;
  pm->rank_eachmb = new_rank_eachmb;
  pm->cost_eachmb = new_cost_eachmb;
  pm->gids_eachrank = new_gids_eachrank;
  pm->nmb_eachrank  = new_nmb_eachrank;
  pm->nmb_total = new_nmb_total;
  pm->nmb_thisrank = pm->nmb_eachrank[global_variable::my_rank];
  pm->hydro_lat_gid_reordered = target_gid_reordered;
  pm->InvalidateHydroLATMetadata();
  if (preserve_lat_metadata && pm->hydro_lat_factor_eachmb != nullptr &&
      pm->hydro_lat_dt_eachmb != nullptr && new_nmb_total == old_nmb) {
    pm->hydro_lat_metadata_nmb = new_nmb_total;
    pm->hydro_lat_sync_factor_current = std::max(1, old_hydro_lat_sync_factor);
    pm->hydro_lat_dt_limited_by_hydro = old_hydro_lat_dt_limited;
    if (pm->hydro_lat_work_eachmb == nullptr) {
      pm->hydro_lat_work_eachmb = new Real[new_nmb_total];
    }
    for (int newm=0; newm<new_nmb_total; ++newm) {
      const int oldm = newtoold[newm];
      pm->hydro_lat_factor_eachmb[newm] = old_hydro_lat_factor[oldm];
      pm->hydro_lat_dt_eachmb[newm] = old_hydro_lat_dt[oldm];
      pm->hydro_lat_work_eachmb[newm] = old_hydro_lat_work[oldm];
    }
    for (int i=0; i<Mesh::kMaxLATBinLevels; ++i) pm->hydro_lat_bin_count[i] = 0;
    for (int gid=0; gid<new_nmb_total; ++gid) {
      int factor = std::max(1, pm->hydro_lat_factor_eachmb[gid]);
      int bin = 0;
      while (factor > 1 && bin < Mesh::kMaxLATBinLevels - 1) {
        factor >>= 1;
        ++bin;
      }
      ++pm->hydro_lat_bin_count[bin];
    }
    pm->hydro_lat_metadata_valid = true;
  }
  for (int gid=0; gid<pm->nmb_total; ++gid) {
    pm->ptree->SetLeafGID(pm->lloc_eachmb[gid], gid);
  }
  // +1 on the MeshBlocks this transaction refined, -1 on those it derefined.
  Kokkos::realloc(fc_amr_repair, new_nmb_total);
  for (int m=0; m<new_nmb_total; ++m) {
    const int flag = pure_rebalance ? 0 : refine_flag.h_view(newtoold[m]);
    fc_amr_repair.h_view(m) = (flag > 0) ? 1 : ((flag < 0) ? -1 : 0);
  }
  fc_amr_repair.template modify<HostMemSpace>();
  fc_amr_repair.template sync<DevExeSpace>();
  if (static_cast<int>(refine_flag.extent(0)) != pm->nmb_total) {
    Kokkos::realloc(refine_flag, pm->nmb_total);
  }
  Kokkos::deep_copy(refine_flag.h_view, 0);
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();

  pm->pmb_pack->gids = pm->gids_eachrank[global_variable::my_rank];
  pm->pmb_pack->gide = pm->pmb_pack->gids + pm->nmb_eachrank[global_variable::my_rank]-1;
  pm->pmb_pack->nmb_thispack = pm->pmb_pack->gide - pm->pmb_pack->gids + 1;
  // The topology transaction has finished placing data, so blocks at or above
  // nmb_thispack are dead and the field arrays may now give memory back.  This is the
  // ONLY resize allowed to shrink: the pre-migration call at the top of this function
  // must keep the outgoing blocks addressable for PackAndSendAMR, which still reads them
  // at their OLD local ids.
  constexpr bool kAllowShrink = true;
  if (phydro != nullptr) {
    phydro->ResizeMeshBlockStorage(pm->pmb_pack->nmb_thispack, false, kAllowShrink);
  }
  if (pmhd != nullptr) {
    pmhd->ResizeMeshBlockStorage(pm->pmb_pack->nmb_thispack, false, kAllowShrink);
  }
  if (prad != nullptr) {
    prad->ResizeMeshBlockStorage(pm->pmb_pack->nmb_thispack, false, kAllowShrink);
  }
  if (pgrav != nullptr) {
    pgrav->ResizeMeshBlockStorage(pm->pmb_pack->nmb_thispack, false, kAllowShrink);
  }

  // Delete old then allocate new MeshBlocks and Coordinates (latter includes masks in GR)
  delete (pm->pmb_pack->pmb);
  delete (pm->pmb_pack->pcoord);
  pm->pmb_pack->AddMeshBlocks(pin);
  pm->pmb_pack->AddCoordinates(pin);
  pm->pmb_pack->pmb->SetNeighbors(pm->ptree, pm->rank_eachmb);
  // Standard boundary buffers reserve nmb_maxperrank capacity. Rank-packed metadata is
  // rebuilt lazily after topology_version changes below, so neither path needs resizing.
  if (pgrav != nullptr && pgrav->pmgd != nullptr) {
    // The multigrid boundary object owns per-MeshBlock communication buffers and
    // persistent MPI/cache state keyed to the old MeshBlock pack. Rebuild it on the
    // new post-AMR pack before the next gravity solve touches same-level or FC ghosts.
    pgrav->pmgd->RefreshMeshblockBoundaryValues(pin);
  }

  // clean-up
  delete [] newtoold;
  delete [] oldtonew;

  // Step 11.
  // Initialize quantities stored on the mesh associated with each physics, if necessary
  // ADM arrays are not redistributed. Recompute prescribed metrics after both
  // refinement changes and pure LAT load-balancing moves.
  if ((pz4c == nullptr) && (padm != nullptr)) {
    padm->SetADMVariables(pm->pmb_pack);
  }
  if ((nnew > 0) || (ndel > 0)) {
    // With radiation, compute tetrads and associated mesh arrays
    if (prad != nullptr) {
      prad->SetOrthonormalTetrad();
    }
    if (pgrav != nullptr) {
      // By default, force a post-AMR solve because the transferred potential is only
      // an interpolation onto the new mesh. Opt-in runs may reuse it as a predictor
      // until the next configured gravity cadence point.
      if (!(pgrav->reuse_phi_after_amr && pgrav->phi_valid)) {
        pgrav->MarkPhiInvalid();
      }
    }
  }
  ++pm->topology_version;
  pm->topology_last_change_cycle = pm->ncycle;
  // It carried phi the same way, ghosts included, so a potential that was current on the
  // old topology is still the solution for the same cells.  Re-stamp only such a one: an
  // AMR-interpolated potential kept by reuse_phi_after_amr stays off-topology.
  if (pure_rebalance && pgrav != nullptr && pgrav->phi_valid &&
      pgrav->self_phi_time_valid &&
      pgrav->self_phi_topology == pm->topology_version - 1) {
    pgrav->self_phi_topology = pm->topology_version;
  }
  if (preserve_lat_metadata && pm->hydro_lat_metadata_valid) {
    pm->hydro_lat_metadata_topology_version = pm->topology_version;
    pm->StampHydroLATLoadBalanceVersions();
  }
  if (pgrav != nullptr && pgrav->phi_valid) {
    if (!pgrav->RefreshPhiGhosts()) {
      pgrav->MarkPhiInvalid();
    }
  }

  // The pack/unpack staging buffer is live only inside this transaction.  Keep it only
  // when AMR runs every few cycles (its allocation churn was the original reason for a
  // persistent buffer); otherwise return its >= 64 MiB to the device.  It only exists in
  // an MPI build -- a serial AMR transaction moves blocks without ever packing them.
#if MPI_PARALLEL_ENABLED
  if (refinement_interval >= 16 && MeshBlockStorageReserve() == 0) {
    lb_stage = DvceArray1D<Real>();
  }
#endif

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineCCSameRank
//! \brief For any MeshBlock m flagged for derefinment (refine_flag = -nleaf), copies
//! cell-centered variables in input coarse array for the nleaf MeshBlock indices that are
//! immediately following to the appropriate quadrant of the MeshBlock m in the input
//! fine array,overwriting any data located there.  Only operates on MBs on the same rank

void MeshRefinement::DerefineCCSameRank(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  // nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) nleaf = 4;
  if (pmy_mesh->three_d) nleaf = 8;

  auto &indcs = pmy_mesh->mb_indcs;
  auto &is  = indcs.is,  &js  = indcs.js,  &ks  = indcs.ks;
  auto &cis = indcs.cis, &cjs = indcs.cjs, &cks = indcs.cks;
  auto &cie = indcs.cie, &cje = indcs.cje, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of source (coarse) array
  std::pair<int,int> isrc = std::make_pair(cis,cie+1);
  std::pair<int,int> jsrc = std::make_pair(cjs,cje+1);
  std::pair<int,int> ksrc = std::make_pair(cks,cke+1);

  // Loop over new local coarse MBs. Gather same-rank old fine children into the
  // canonical old child slot; CopyCC later moves that slot to the new local gid.
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int root_oldm = newtoold[newm];
    if (root_oldm < ombs || root_oldm > ombe) continue;
    LogicalLocation &root_old_lloc = pmy_mesh->lloc_eachmb[root_oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (root_old_lloc.level <= new_lloc.level) continue;
    for (int l=0; l<nleaf; l++) {
      const int old_child = DerefOldChildGID(newm, l, nleaf);
      if (old_child < ombs || old_child > ombe) continue;
      LogicalLocation &lloc = pmy_mesh->lloc_eachmb[old_child];
      int ox1 = ((lloc.lx1 & 1) == 1);
      int ox2 = ((lloc.lx2 & 1) == 1);
      int ox3 = ((lloc.lx3 & 1) == 1);
      std::pair<int,int> idst = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1));
      std::pair<int,int> jdst = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2));
      std::pair<int,int> kdst = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3));

      auto src = Kokkos::subview(ca,(old_child-ombs),Kokkos::ALL,ksrc,jsrc,isrc);
      auto dst = Kokkos::subview( a,(root_oldm-ombs),Kokkos::ALL,kdst,jdst,idst);
      Kokkos::deep_copy(DevExeSpace(), dst, src);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::DerefineFCSameRank
//! \brief Same as DerefineCCSameRank, except for face-centered variables

void MeshRefinement::DerefineFCSameRank(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  // nleaf = number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (pmy_mesh->two_d) nleaf = 4;
  if (pmy_mesh->three_d) nleaf = 8;

  auto &indcs = pmy_mesh->mb_indcs;
  auto &is  = indcs.is,  &js  = indcs.js,  &ks  = indcs.ks;
  auto &cis = indcs.cis, &cjs = indcs.cjs, &cks = indcs.cks;
  auto &cie = indcs.cie, &cje = indcs.cje, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of source (coarse) array
  std::pair<int,int> isrc  = std::make_pair(cis,cie+1);
  std::pair<int,int> isrc1 = std::make_pair(cis,cie+2);
  std::pair<int,int> jsrc  = std::make_pair(cjs,cje+1);
  std::pair<int,int> jsrc1 = std::make_pair(cjs,cje+2);
  std::pair<int,int> ksrc  = std::make_pair(cks,cke+1);
  std::pair<int,int> ksrc1 = std::make_pair(cks,cke+2);

  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int root_oldm = newtoold[newm];
    if (root_oldm < ombs || root_oldm > ombe) continue;
    LogicalLocation &root_old_lloc = pmy_mesh->lloc_eachmb[root_oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (root_old_lloc.level <= new_lloc.level) continue;
    for (int l=0; l<nleaf; l++) {
      const int old_child = DerefOldChildGID(newm, l, nleaf);
      if (old_child < ombs || old_child > ombe) continue;
      LogicalLocation &lloc = pmy_mesh->lloc_eachmb[old_child];
      int ox1 = ((lloc.lx1 & 1) == 1);
      int ox2 = ((lloc.lx2 & 1) == 1);
      int ox3 = ((lloc.lx3 & 1) == 1);
      std::pair<int,int> idst  = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1  ));
      std::pair<int,int> idst1 = std::make_pair((is+ox1*cnx1),(is+(ox1+1)*cnx1+1));
      std::pair<int,int> jdst  = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2  ));
      std::pair<int,int> jdst1 = std::make_pair((js+ox2*cnx2),(js+(ox2+1)*cnx2+1));
      std::pair<int,int> kdst  = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3  ));
      std::pair<int,int> kdst1 = std::make_pair((ks+ox3*cnx3),(ks+(ox3+1)*cnx3+1));

      auto src1 = Kokkos::subview(cb.x1f,(old_child-ombs),ksrc,jsrc,isrc1);
      auto dst1 = Kokkos::subview( b.x1f,(root_oldm-ombs),kdst,jdst,idst1);
      Kokkos::deep_copy(DevExeSpace(), dst1, src1);
      auto src2 = Kokkos::subview(cb.x2f,(old_child-ombs),ksrc,jsrc1,isrc);
      auto dst2 = Kokkos::subview( b.x2f,(root_oldm-ombs),kdst,jdst1,idst);
      Kokkos::deep_copy(DevExeSpace(), dst2, src2);
      auto src3 = Kokkos::subview(cb.x3f,(old_child-ombs),ksrc1,jsrc,isrc);
      auto dst3 = Kokkos::subview( b.x3f,(root_oldm-ombs),kdst1,jdst,idst);
      Kokkos::deep_copy(DevExeSpace(), dst3, src3);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyCC
//! \brief Copy cell-centered variables to new MB index within View for MeshBlocks that
//! stay within this rank

void MeshRefinement::CopyCC(DvceArray5D<Real> &a) {
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int old_nmb_local = pmy_mesh->nmb_eachrank[global_variable::my_rank];
  int new_nmb_local = new_nmb_eachrank[global_variable::my_rank];
  int block_slot_count = std::max(old_nmb_local, new_nmb_local);

  std::vector<int> map_dst(old_nmb_local, -1);
  std::vector<int> src_for_slot(block_slot_count, -1);
  for (int oldm=ombs; oldm<=ombe; ++oldm) {
    int newm = oldtonew[oldm];
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;

    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level && newtoold[newm] != oldm) continue;

    int src = oldm - ombs;
    int dst = newm - nmbs;
    if (src < 0 || src >= old_nmb_local || dst < 0 || dst >= new_nmb_local) continue;
    map_dst[src] = dst;
    if (dst < block_slot_count) src_for_slot[dst] = src;
  }

  DevExeSpace exec;
  auto copy_block = [&](const int src_lid, const int dst_lid) {
    if (src_lid == dst_lid) return;
    auto src = Kokkos::subview(a, src_lid, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                               Kokkos::ALL);
    auto dst = Kokkos::subview(a, dst_lid, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                               Kokkos::ALL);
    Kokkos::deep_copy(exec, dst, src);
  };

  std::vector<char> done(old_nmb_local, 0);
  std::vector<int> stack;
  stack.reserve(old_nmb_local);
  for (int src=0; src<old_nmb_local; ++src) {
    const int dst = map_dst[src];
    if (dst >= 0 && (dst >= old_nmb_local || map_dst[dst] < 0)) stack.push_back(src);
  }
  while (!stack.empty()) {
    const int src = stack.back();
    stack.pop_back();
    if (src < 0 || src >= old_nmb_local || done[src] || map_dst[src] < 0) continue;
    const int dst = map_dst[src];
    copy_block(src, dst);
    done[src] = 1;
    map_dst[src] = -1;
    if (src < block_slot_count) {
      const int pred = src_for_slot[src];
      if (pred >= 0 && pred < old_nmb_local && !done[pred] && map_dst[pred] >= 0) {
        stack.push_back(pred);
      }
    }
  }

  DvceArray5D<Real> tmp;
  for (int start=0; start<old_nmb_local; ++start) {
    if (done[start] || map_dst[start] < 0) continue;
    if (tmp.extent_int(0) == 0) {
      tmp = DvceArray5D<Real>("amr_copycc_tmp", 1, a.extent_int(1), a.extent_int(2),
                              a.extent_int(3), a.extent_int(4));
    }
    auto start_view = Kokkos::subview(a, start, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                      Kokkos::ALL);
    auto tmp_view = Kokkos::subview(tmp, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                    Kokkos::ALL);
    Kokkos::deep_copy(exec, tmp_view, start_view);

    int dst = start;
    int src = (dst < block_slot_count) ? src_for_slot[dst] : -1;
    while (src >= 0 && src != start) {
      copy_block(src, dst);
      done[src] = 1;
      map_dst[src] = -1;
      dst = src;
      src = (dst < block_slot_count) ? src_for_slot[dst] : -1;
    }
    auto final_dst = Kokkos::subview(a, dst, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
    Kokkos::deep_copy(exec, final_dst, tmp_view);
    done[start] = 1;
    map_dst[start] = -1;
  }
  exec.fence();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyFC
//! \brief Copy face-centered variables to new MB index within View for MeshBlocks that
//! stay within this rank

void MeshRefinement::CopyFC(DvceFaceFld4D<Real> &b) {
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int old_nmb_local = pmy_mesh->nmb_eachrank[global_variable::my_rank];
  int new_nmb_local = new_nmb_eachrank[global_variable::my_rank];
  int block_slot_count = std::max(old_nmb_local, new_nmb_local);

  std::vector<int> base_map_dst(old_nmb_local, -1);
  std::vector<int> base_src_for_slot(block_slot_count, -1);
  for (int oldm=ombs; oldm<=ombe; ++oldm) {
    int newm = oldtonew[oldm];
    if (new_rank_eachmb[newm] != global_variable::my_rank) continue;

    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level && newtoold[newm] != oldm) continue;

    int src = oldm - ombs;
    int dst = newm - nmbs;
    if (src < 0 || src >= old_nmb_local || dst < 0 || dst >= new_nmb_local) continue;
    base_map_dst[src] = dst;
    if (dst < block_slot_count) base_src_for_slot[dst] = src;
  }

  auto permute_face = [&](DvceArray4D<Real> &face, const char *label) {
    std::vector<int> map_dst = base_map_dst;
    std::vector<int> src_for_slot = base_src_for_slot;
    std::vector<char> done(old_nmb_local, 0);
    DevExeSpace exec;
    auto copy_block = [&](const int src_lid, const int dst_lid) {
      if (src_lid == dst_lid) return;
      auto src = Kokkos::subview(face, src_lid, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto dst = Kokkos::subview(face, dst_lid, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      Kokkos::deep_copy(exec, dst, src);
    };

    std::vector<int> stack;
    stack.reserve(old_nmb_local);
    for (int src=0; src<old_nmb_local; ++src) {
      const int dst = map_dst[src];
      if (dst >= 0 && (dst >= old_nmb_local || map_dst[dst] < 0)) stack.push_back(src);
    }
    while (!stack.empty()) {
      const int src = stack.back();
      stack.pop_back();
      if (src < 0 || src >= old_nmb_local || done[src] || map_dst[src] < 0) continue;
      const int dst = map_dst[src];
      copy_block(src, dst);
      done[src] = 1;
      map_dst[src] = -1;
      if (src < block_slot_count) {
        const int pred = src_for_slot[src];
        if (pred >= 0 && pred < old_nmb_local && !done[pred] && map_dst[pred] >= 0) {
          stack.push_back(pred);
        }
      }
    }

    DvceArray4D<Real> tmp;
    for (int start=0; start<old_nmb_local; ++start) {
      if (done[start] || map_dst[start] < 0) continue;
      if (tmp.extent_int(0) == 0) {
        tmp = DvceArray4D<Real>(std::string(label), 1, face.extent_int(1),
                                face.extent_int(2), face.extent_int(3));
      }
      auto start_view = Kokkos::subview(face, start, Kokkos::ALL, Kokkos::ALL,
                                        Kokkos::ALL);
      auto tmp_view = Kokkos::subview(tmp, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      Kokkos::deep_copy(exec, tmp_view, start_view);

      int dst = start;
      int src = (dst < block_slot_count) ? src_for_slot[dst] : -1;
      while (src >= 0 && src != start) {
        copy_block(src, dst);
        done[src] = 1;
        map_dst[src] = -1;
        dst = src;
        src = (dst < block_slot_count) ? src_for_slot[dst] : -1;
      }
      auto final_dst = Kokkos::subview(face, dst, Kokkos::ALL, Kokkos::ALL,
                                       Kokkos::ALL);
      Kokkos::deep_copy(exec, final_dst, tmp_view);
      done[start] = 1;
      map_dst[start] = -1;
    }
    exec.fence();
  };

  permute_face(b.x1f, "amr_copyfc_x1_tmp");
  permute_face(b.x2f, "amr_copyfc_x2_tmp");
  permute_face(b.x3f, "amr_copyfc_x3_tmp");
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyForRefinementCC
//! \brief For any MeshBlock m flagged for refinment (refine_flag = 1), copies
//! cell-centered variables in octants of input fine array to the input coarse arrays at
//! the nleaf-index locations that are immediately following (overwriting any data located
//! there).  Only operates on MBs on the same rank.

void MeshRefinement::CopyForRefinementCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca) {
  auto &indcs = pmy_mesh->mb_indcs;
  auto &ng = indcs.ng;
  int il = indcs.cis - ng, iu = indcs.cie + ng;
  int jl = indcs.cjs,      ju = indcs.cje;
  int kl = indcs.cks,      ku = indcs.cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of destination (coarse) array
  std::pair<int,int> idst = std::make_pair(il,iu+1);
  std::pair<int,int> jdst = std::make_pair(jl,ju+1);
  std::pair<int,int> kdst = std::make_pair(kl,ku+1);

  // loop over new MBs on this rank
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int oldm = newtoold[newm];
    if (refine_flag.h_view(oldm) > 0) {
      // only copy if old and new location of MB on this rank
      if ((new_rank_eachmb[oldtonew[oldm]] == global_variable::my_rank) &&
          (new_rank_eachmb[newm] == global_variable::my_rank)) {
        int msrc = oldtonew[oldm] - nmbs;
        int mdst = newm - nmbs;
        LogicalLocation &lloc = new_lloc_eachmb[newm];
        int ox1 = ((lloc.lx1 & 1) == 1);
        int ox2 = ((lloc.lx2 & 1) == 1);
        int ox3 = ((lloc.lx3 & 1) == 1);
        std::pair<int,int> isrc = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1));
        std::pair<int,int> jsrc = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2));
        std::pair<int,int> ksrc = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3));

        // copy data in MBs to be refined to coarse arrays in target MBs
        auto src = Kokkos::subview( a,msrc,Kokkos::ALL,ksrc,jsrc,isrc);
        auto dst = Kokkos::subview(ca,mdst,Kokkos::ALL,kdst,jdst,idst);
        Kokkos::deep_copy(DevExeSpace(), dst, src);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CopyForRefinementFC
//! \brief Same as CopyForRefinementCC, but for face-centered arrays

void MeshRefinement::CopyForRefinementFC(DvceFaceFld4D<Real> &b,DvceFaceFld4D<Real> &cb) {
  auto &indcs = pmy_mesh->mb_indcs;
  auto &ng = indcs.ng;
  int il = indcs.cis - ng, iu = indcs.cie + ng;
  int jl = indcs.cjs,      ju = indcs.cje;
  int kl = indcs.cks,      ku = indcs.cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;

  // Set indices of destination (coarse) array
  std::pair<int,int> idst  = std::make_pair(il,iu+1);
  std::pair<int,int> idst1 = std::make_pair(il,iu+2);
  std::pair<int,int> jdst  = std::make_pair(jl,ju+1);
  std::pair<int,int> jdst1 = std::make_pair(jl,ju+2);
  std::pair<int,int> kdst  = std::make_pair(kl,ku+1);
  std::pair<int,int> kdst1 = std::make_pair(kl,ku+2);

  // loop over new MBs on this rank
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; ++newm) {
    int oldm = newtoold[newm];
    if (refine_flag.h_view(oldm) > 0) {
      // only copy if old and new location of MB on this rank
      if ((new_rank_eachmb[oldtonew[oldm]] == global_variable::my_rank) &&
          (new_rank_eachmb[newm] == global_variable::my_rank)) {
        int msrc = oldtonew[oldm] - nmbs;
        int mdst = newm - nmbs;
        LogicalLocation &lloc = new_lloc_eachmb[newm];
        int ox1 = ((lloc.lx1 & 1) == 1);
        int ox2 = ((lloc.lx2 & 1) == 1);
        int ox3 = ((lloc.lx3 & 1) == 1);
        std::pair<int,int> isrc  = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1  ));
        std::pair<int,int> isrc1 = std::make_pair((il + ox1*cnx1),(iu+1 + ox1*cnx1+1));
        std::pair<int,int> jsrc  = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2  ));
        std::pair<int,int> jsrc1 = std::make_pair((jl + ox2*cnx2),(ju+1 + ox2*cnx2+1));
        std::pair<int,int> ksrc  = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3  ));
        std::pair<int,int> ksrc1 = std::make_pair((kl + ox3*cnx3),(ku+1 + ox3*cnx3+1));

        // copy data in MBs to be refined to coarse arrays in target MBs
        auto src1 = Kokkos::subview( b.x1f,msrc,ksrc,jsrc,isrc1);
        auto dst1 = Kokkos::subview(cb.x1f,mdst,kdst,jdst,idst1);
        Kokkos::deep_copy(DevExeSpace(), dst1, src1);
        auto src2 = Kokkos::subview( b.x2f,msrc,ksrc,jsrc1,isrc);
        auto dst2 = Kokkos::subview(cb.x2f,mdst,kdst,jdst1,idst);
        Kokkos::deep_copy(DevExeSpace(), dst2, src2);
        auto src3 = Kokkos::subview( b.x3f,msrc,ksrc1,jsrc,isrc);
        auto dst3 = Kokkos::subview(cb.x3f,mdst,kdst1,jdst,idst);
        Kokkos::deep_copy(DevExeSpace(), dst3, src3);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineCC
//! \brief Refines cell-centered variables in input view at any MeshBlock index m that is
//! flagged for refinement to the m-index locations which are immediately following,
//! overwriting any data located there. The data in these locations must already have been
//! copied to another location or sent to another rank via MPI.

void MeshRefinement::RefineCC(DualArray1D<int> &n2o, DvceArray5D<Real> &a,
                              DvceArray5D<Real> &ca, bool is_z4c) {
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &new_nmb = new_nmb_eachrank[global_variable::my_rank];
  auto &indcs = pmy_mesh->mb_indcs;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &nx1 = indcs.nx1;
  auto &nx2 = indcs.nx2;
  auto &nx3 = indcs.nx3;
  auto& prolong_2nd = weights.prolong_2nd;
  auto& prolong_4th = weights.prolong_4th;

  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;
  auto &ngids_ = new_gids_eachrank[global_variable::my_rank];
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nmv = new_nmb*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - m*nvar);

    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      const int ni = cie - cis + 1;
      const int nj = cje - cjs + 1;
      const int nk = cke - cks + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + cis;
        k += cks;
        j += cjs;

        // fine indices refer to target array
        int fi = 2*i - cis;  // correct when cis=is
        int fj = 2*j - cjs;  // correct when cjs=js
        int fk = 2*k - cks;  // correct when cks=ks

        // call inlined prolongation operator for CC variables
        if (!is_z4c) {
          ProlongCC(m,v,k,j,i,fk,fj,fi,multi_d,three_d,ca,a);
        } else {
          switch (indcs.ng) {
            case 2: HighOrderProlongCC<2>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_2nd);
                    break;
            case 4: HighOrderProlongCC<4>(m,v,k,j,i,fk,fj,fi,nx1,nx2,nx3,
                                          ca,a,prolong_4th);
                    break;
          }
        }
      });
    }
  });

  return;
}

namespace {

//! Tangential extents of a MeshBlock boundary plane of normal dir (0,1,2 = x1,x2,x3):
//! (k,j) for x1, (k,i) for x2, (j,i) for x3, in the fine or in the coarse array.
KOKKOS_INLINE_FUNCTION
void BoundaryPlaneExtent(const int dir, const bool coarse, const RegionIndcs &indcs,
                         int &na, int &nc) {
  if (coarse) {
    na = (dir == 2) ? indcs.cnx2 : indcs.cnx3;
    nc = (dir == 0) ? indcs.cnx2 : indcs.cnx1;
  } else {
    na = (dir == 2) ? indcs.nx2 : indcs.nx3;
    nc = (dir == 0) ? indcs.nx2 : indcs.nx1;
  }
}

//! The normal face field on that plane at tangential offsets (a,c) from its first active
//! face.  outer selects the ie+1 (je+1, ke+1) plane, otherwise the is (js, ks) plane;
//! coarse addresses a coarse array with the coarse indices.
KOKKOS_INLINE_FUNCTION
Real &BoundaryPlaneFace(const DvceFaceFld4D<Real> &b, const int m, const int dir,
                        const bool outer, const bool coarse, const int a, const int c,
                        const RegionIndcs &indcs) {
  const int is = coarse ? indcs.cis : indcs.is, ie = coarse ? indcs.cie : indcs.ie;
  const int js = coarse ? indcs.cjs : indcs.js, je = coarse ? indcs.cje : indcs.je;
  const int ks = coarse ? indcs.cks : indcs.ks, ke = coarse ? indcs.cke : indcs.ke;
  if (dir == 0) {
    return b.x1f(m, ks + a, js + c, outer ? ie + 1 : is);
  } else if (dir == 1) {
    return b.x2f(m, ks + a, outer ? je + 1 : js, is + c);
  }
  return b.x3f(m, outer ? ke + 1 : ks, js + a, is + c);
}

//----------------------------------------------------------------------------------------
//! \fn void AdoptNeighborSharedFaces()
//! \brief Each MeshBlock the current transaction creates by refinement takes, on each of
//! its boundary planes, the face field of the MeshBlock of its own level across that
//! plane: the current values of a MeshBlock that is unchanged, derefined, or created by
//! refinement with a smaller gid, or, when the transaction refined that MeshBlock, its
//! pre-transaction values, which its children on the plane hold in their coarse arrays.
//! The adopted values of each aligned group of 2x2 faces (2 in 2D) are shifted by one
//! constant so that the group keeps the flux the refined block prolonged from its coarse
//! parent (the group tiles one face of a coarse parent cell), and the Toth & Roe internal
//! faces built from the group afterwards stay divergence free.  All values are read
//! before any is written.  Runs on every rank after all transaction data has arrived,
//! since it may communicate.

void AdoptNeighborSharedFaces(Mesh *pm, MeshRefinement *pmr, DvceFaceFld4D<Real> &b,
                              DvceFaceFld4D<Real> &cb) {
  // In 1D a plane is one face, which is its own group and so keeps its value.
  if (pm->one_d) return;
  const int nranks = global_variable::nranks;
  const int my_rank = global_variable::my_rank;
  int new_nmb_total = 0;
  for (int r=0; r<nranks; ++r) new_nmb_total += pmr->new_nmb_eachrank[r];
  auto flag = [&](const int gid) {
    return pmr->refine_flag.h_view(pmr->newtoold[gid]);
  };
  std::unordered_map<AMRLogicalLocationKey, int, AMRLogicalLocationKeyHash> gid_at;
  gid_at.reserve(static_cast<std::size_t>(new_nmb_total)*2);
  for (int g=0; g<new_nmb_total; ++g) {
    gid_at.emplace(AMRLogicalLocationKey(pmr->new_lloc_eachmb[g]), g);
  }
  auto find_gid = [&](const LogicalLocation &loc) {
    const auto it = gid_at.find(AMRLogicalLocationKey(loc));
    return (it == gid_at.end()) ? -1 : it->second;
  };
  auto &indcs = pm->mb_indcs;
  auto active = [&](const int d) {
    return (d == 0) || (d == 1 && pm->multi_d) || (d == 2 && pm->three_d);
  };

  // Every rank lists the same planes and pieces in gid order.  A piece fills the part of
  // its plane at tangential offsets (a0,c0) from the facing plane of block src, read from
  // src's coarse array when coarse != 0.
  struct Plane {int dst, dir, outer;};
  struct Piece {int plane, src, coarse, a0, c0;};
  std::vector<Plane> planes;
  std::vector<Piece> pieces;
  const int ndir = pm->three_d ? 3 : 2;
  const int nroot[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  for (int g=0; g<new_nmb_total; ++g) {
    if (flag(g) <= 0) continue;
    const LogicalLocation &loc = pmr->new_lloc_eachmb[g];
    for (int dir=0; dir<ndir; ++dir) {
      const std::int32_t nlev =
          static_cast<std::int32_t>(nroot[dir]) << (loc.level - pm->root_level);
      const int ta = (dir == 2) ? 1 : 2;  // direction of plane offset a
      const int tc = (dir == 0) ? 1 : 0;  // direction of plane offset c
      for (int outer=0; outer<2; ++outer) {
        LogicalLocation nloc = loc;
        std::int32_t *nl[3] = {&nloc.lx1, &nloc.lx2, &nloc.lx3};
        *nl[dir] += outer ? 1 : -1;
        if (*nl[dir] < 0 || *nl[dir] >= nlev) {
          if (pm->mesh_bcs[2*dir + outer] != BoundaryFlag::periodic) continue;
          *nl[dir] = (*nl[dir] < 0) ? nlev - 1 : 0;
        }
        const int plane = static_cast<int>(planes.size());
        const int src = find_gid(nloc);
        if (src >= 0) {
          // Of two refined blocks, the one of larger gid adopts.
          if (flag(src) > 0 && src >= g) continue;
          planes.push_back({g, dir, outer});
          pieces.push_back({plane, src, 0, 0, 0});
          continue;
        }
        // No same-level block: its children facing the face, if the transaction created
        // them by refining it.
        LogicalLocation cloc = nloc;
        cloc.level += 1;
        cloc.lx1 *= 2;
        cloc.lx2 *= 2;
        cloc.lx3 *= 2;
        std::int32_t *cl[3] = {&cloc.lx1, &cloc.lx2, &cloc.lx3};
        if (!outer) *cl[dir] += 1;
        int ca, cc;
        BoundaryPlaneExtent(dir, true, indcs, ca, cc);
        std::vector<Piece> kids;
        bool all_refined = true;
        for (int oa=0; oa<(active(ta) ? 2 : 1) && all_refined; ++oa) {
          for (int oc=0; oc<(active(tc) ? 2 : 1) && all_refined; ++oc) {
            LogicalLocation q = cloc;
            std::int32_t *ql[3] = {&q.lx1, &q.lx2, &q.lx3};
            *ql[ta] += oa;
            *ql[tc] += oc;
            const int kid = find_gid(q);
            all_refined = (kid >= 0) && (flag(kid) > 0);
            if (all_refined) kids.push_back({plane, kid, 1, oa*ca, oc*cc});
          }
        }
        if (!all_refined) continue;
        planes.push_back({g, dir, outer});
        pieces.insert(pieces.end(), kids.begin(), kids.end());
      }
    }
  }
  if (planes.empty()) return;

  auto rank_of = [&](const int gid) { return pmr->new_rank_eachmb[gid]; };
  auto lid_of = [&](const int gid) {
    return gid - pmr->new_gids_eachrank[rank_of(gid)];
  };
  auto piece_size = [&](const Piece &q) {
    int pa, pc;
    BoundaryPlaneExtent(planes[q.plane].dir, q.coarse != 0, indcs, pa, pc);
    return pa*pc;
  };
  // Staging area of the planes whose block is on this rank.
  std::vector<int> stage_off(planes.size(), -1), my_planes;
  int nstage = 0;
  for (std::size_t p=0; p<planes.size(); ++p) {
    if (rank_of(planes[p].dst) != my_rank) continue;
    int na, nc;
    BoundaryPlaneExtent(planes[p].dir, false, indcs, na, nc);
    stage_off[p] = nstage;
    nstage += na*nc;
    my_planes.push_back(static_cast<int>(p));
  }
  std::vector<int> local;
  std::vector<std::vector<int>> send_to(nranks), recv_from(nranks);
  for (std::size_t n=0; n<pieces.size(); ++n) {
    const int dst_rank = rank_of(planes[pieces[n].plane].dst);
    const int src_rank = rank_of(pieces[n].src);
    if (dst_rank == my_rank && src_rank == my_rank) {
      local.push_back(static_cast<int>(n));
    } else if (src_rank == my_rank) {
      send_to[dst_rank].push_back(static_cast<int>(n));
    } else if (dst_rank == my_rank) {
      recv_from[src_rank].push_back(static_cast<int>(n));
    }
  }
  // Piece descriptor columns: src lid, dir, outer, coarse, a0, c0, stage offset of the
  // plane, message offset.
  auto piece_desc = [&](const std::vector<int> &list, const char *label) {
    DualArray2D<int> d(label, std::max<std::size_t>(list.size(), 1), 8);
    int off = 0;
    for (std::size_t n=0; n<list.size(); ++n) {
      const Piece &q = pieces[list[n]];
      const Plane &p = planes[q.plane];
      d.h_view(n, 0) = (rank_of(q.src) == my_rank) ? lid_of(q.src) : -1;
      d.h_view(n, 1) = p.dir;
      d.h_view(n, 2) = p.outer;
      d.h_view(n, 3) = q.coarse;
      d.h_view(n, 4) = q.a0;
      d.h_view(n, 5) = q.c0;
      d.h_view(n, 6) = stage_off[q.plane];
      d.h_view(n, 7) = off;
      off += piece_size(q);
    }
    d.template modify<HostMemSpace>();
    d.template sync<DevExeSpace>();
    return d;
  };
  int na_max = 1, nc_max = 1;
  for (int dir=0; dir<ndir; ++dir) {
    int na, nc;
    BoundaryPlaneExtent(dir, false, indcs, na, nc);
    na_max = std::max(na_max, na);
    nc_max = std::max(nc_max, nc);
  }
  DvceArray1D<Real> stage("amr_adopt_stage", std::max(nstage, 1));

  if (!local.empty()) {
    auto d = piece_desc(local, "amr_adopt_local").d_view;
    par_for("AdoptFaces-gather", DevExeSpace(), 0, static_cast<int>(local.size())-1,
            0, na_max-1, 0, nc_max-1,
    KOKKOS_LAMBDA(const int n, const int a, const int c) {
      const int dir = d(n, 1);
      const bool coarse = (d(n, 3) != 0);
      int pa, pc, na, nc;
      BoundaryPlaneExtent(dir, coarse, indcs, pa, pc);
      if (a >= pa || c >= pc) return;
      BoundaryPlaneExtent(dir, false, indcs, na, nc);
      const bool outer = (d(n, 2) != 0);
      stage(d(n, 6) + (d(n, 4) + a)*nc + d(n, 5) + c) =
          BoundaryPlaneFace(coarse ? cb : b, d(n, 0), dir, !outer, coarse, a, c, indcs);
    });
  }

#if MPI_PARALLEL_ENABLED
  // Messages between two ranks list their pieces in the same order on both sides.
  std::vector<int> send_list, recv_list, send_off(nranks, 0), send_cnt(nranks, 0),
                   recv_off(nranks, 0), recv_cnt(nranks, 0);
  int nsend = 0, nrecv = 0;
  for (int r=0; r<nranks; ++r) {
    send_off[r] = nsend;
    for (const int n : send_to[r]) {
      send_list.push_back(n);
      send_cnt[r] += piece_size(pieces[n]);
    }
    nsend += send_cnt[r];
    recv_off[r] = nrecv;
    for (const int n : recv_from[r]) {
      recv_list.push_back(n);
      recv_cnt[r] += piece_size(pieces[n]);
    }
    nrecv += recv_cnt[r];
  }
  if (nsend > 0 || nrecv > 0) {
    // amr_comm otherwise carries only tag-0 load-balance messages, all complete by now.
    constexpr int kAdoptFacesTag = 1;
    DvceArray1D<Real> recv_dev("amr_adopt_recv", std::max(nrecv, 1));
    auto recv_host = Kokkos::create_mirror_view(recv_dev);
    DvceArray1D<Real> send_dev("amr_adopt_send", std::max(nsend, 1));
    auto send_host = Kokkos::create_mirror_view(send_dev);
    std::vector<MPI_Request> requests;
    for (int r=0; r<nranks; ++r) {
      if (recv_cnt[r] == 0) continue;
      requests.emplace_back();
      MPI_Irecv(recv_host.data() + recv_off[r], recv_cnt[r], MPI_ATHENA_REAL, r,
                kAdoptFacesTag, pmr->amr_comm, &requests.back());
    }
    if (nsend > 0) {
      auto d = piece_desc(send_list, "amr_adopt_send_desc").d_view;
      par_for("AdoptFaces-pack", DevExeSpace(), 0, static_cast<int>(send_list.size())-1,
              0, na_max-1, 0, nc_max-1,
      KOKKOS_LAMBDA(const int n, const int a, const int c) {
        const int dir = d(n, 1);
        const bool coarse = (d(n, 3) != 0);
        int pa, pc;
        BoundaryPlaneExtent(dir, coarse, indcs, pa, pc);
        if (a >= pa || c >= pc) return;
        const bool outer = (d(n, 2) != 0);
        send_dev(d(n, 7) + a*pc + c) =
            BoundaryPlaneFace(coarse ? cb : b, d(n, 0), dir, !outer, coarse, a, c, indcs);
      });
      Kokkos::deep_copy(send_host, send_dev);
      for (int r=0; r<nranks; ++r) {
        if (send_cnt[r] == 0) continue;
        requests.emplace_back();
        MPI_Isend(send_host.data() + send_off[r], send_cnt[r], MPI_ATHENA_REAL, r,
                  kAdoptFacesTag, pmr->amr_comm, &requests.back());
      }
    }
    if (MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "AMR shared-face exchange failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (nrecv > 0) {
      Kokkos::deep_copy(recv_dev, recv_host);
      auto d = piece_desc(recv_list, "amr_adopt_recv_desc").d_view;
      par_for("AdoptFaces-unpack", DevExeSpace(), 0, static_cast<int>(recv_list.size())-1,
              0, na_max-1, 0, nc_max-1,
      KOKKOS_LAMBDA(const int n, const int a, const int c) {
        const int dir = d(n, 1);
        int pa, pc, na, nc;
        BoundaryPlaneExtent(dir, d(n, 3) != 0, indcs, pa, pc);
        if (a >= pa || c >= pc) return;
        BoundaryPlaneExtent(dir, false, indcs, na, nc);
        stage(d(n, 6) + (d(n, 4) + a)*nc + d(n, 5) + c) = recv_dev(d(n, 7) + a*pc + c);
      });
    }
  }
#endif

  if (my_planes.empty()) return;
  DualArray2D<int> pd("amr_adopt_planes", my_planes.size(), 4);
  for (std::size_t n=0; n<my_planes.size(); ++n) {
    const Plane &p = planes[my_planes[n]];
    pd.h_view(n, 0) = lid_of(p.dst);
    pd.h_view(n, 1) = p.dir;
    pd.h_view(n, 2) = p.outer;
    pd.h_view(n, 3) = stage_off[my_planes[n]];
  }
  pd.template modify<HostMemSpace>();
  pd.template sync<DevExeSpace>();
  auto pdv = pd.d_view;
  par_for("AdoptFaces-apply", DevExeSpace(), 0, static_cast<int>(my_planes.size())-1,
          0, (na_max+1)/2-1, 0, (nc_max+1)/2-1,
  KOKKOS_LAMBDA(const int n, const int ga, const int gc) {
    const int dir = pdv(n, 1);
    int na, nc;
    BoundaryPlaneExtent(dir, false, indcs, na, nc);
    const int sa = (na > 1) ? 2 : 1;
    const int sc = (nc > 1) ? 2 : 1;
    const int a0 = ga*sa, c0 = gc*sc;
    if (a0 >= na || c0 >= nc) return;
    const int m = pdv(n, 0);
    const bool outer = (pdv(n, 2) != 0);
    const int st = pdv(n, 3);
    Real own = 0.0, adopted = 0.0;
    for (int a=a0; a<a0+sa; ++a) {
      for (int c=c0; c<c0+sc; ++c) {
        own += BoundaryPlaneFace(b, m, dir, outer, false, a, c, indcs);
        adopted += stage(st + a*nc + c);
      }
    }
    const Real shift = (own - adopted)/static_cast<Real>(sa*sc);
    for (int a=a0; a<a0+sa; ++a) {
      for (int c=c0; c<c0+sc; ++c) {
        BoundaryPlaneFace(b, m, dir, outer, false, a, c, indcs) =
            stage(st + a*nc + c) + shift;
      }
    }
  });
  return;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RefineFC
//! \brief Same as RefineCC, except for face-centered arrays

void MeshRefinement::RefineFC(DualArray1D<int> &n2o, DvceFaceFld4D<Real> &b,
                              DvceFaceFld4D<Real> &cb) {
  auto &new_nmb = new_nmb_eachrank[global_variable::my_rank];;
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is;
  auto &js = indcs.js;
  auto &ks = indcs.ks;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;

  // First prolongate face-centered fields at shared faces betwen fine and coarse cells
  auto &refine_flag_ = refine_flag;
  bool &multi_d = pmy_mesh->multi_d;
  bool &three_d = pmy_mesh->three_d;
  auto &ngids_ = new_gids_eachrank[global_variable::my_rank];

  // Prolongate x1f
  par_for("RefineFC1",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje, cis,cie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX1Face(m,k,j,i,fk,fj,fi,multi_d,three_d,cb.x1f,b.x1f);
    }
  });

  // Prolongate x2f
  par_for("RefineFC2",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje+1, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX2Face(m,k,j,i,fk,fj,fi,three_d,cb.x2f,b.x2f);
    }
  });

  // Prolongate x3f
  par_for("RefineFC3",DevExeSpace(), 0,(new_nmb-1), cks,cke+1, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;                   // fine i
      int fj = (multi_d)? ((j - cjs)*2 + js) : j;  // fine j
      int fk = (three_d)? ((k - cks)*2 + ks) : k;  // fine k
      ProlongFCSharedX3Face(m,k,j,i,fk,fj,fi,multi_d,cb.x3f,b.x3f);
    }
  });

  // The faces shared with same-level MeshBlocks take their values before the internal
  // faces below are built from the boundary faces.
  AdoptNeighborSharedFaces(pmy_mesh, this, b, cb);

  // Second prolongate face-centered fields at internal faces of fine cells using
  // divergence-preserving operator of Toth & Roe (2002)
  bool &one_d = pmy_mesh->one_d;
  par_for("RefineFC-int",DevExeSpace(), 0,(new_nmb-1), cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (refine_flag_.d_view(n2o.d_view(m+ngids_)) > 0) {
      // fine indices refer to target array
      int fi = (i - cis)*2 + is;   // fine i
      int fj = (j - cjs)*2 + js;   // fine j
      int fk = (k - cks)*2 + ks;   // fine k

      if (one_d) {
        // In 1D, interior face field is trivial
        b.x1f(m,fk,fj,fi+1) = 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
      } else {
        // in multi-D call inlined prolongation operator for FC fields at internal faces
        ProlongFCInternal(m,fk,fj,fi,three_d,b);
      }
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RepairAMRFC
//! \brief Recompute internal FC fields after exterior faces are finalized post-AMR, on
//! the MeshBlocks the last transaction refined.  A derefined MeshBlock keeps the exact
//! restriction of its children's faces, which the exchange before this call has already
//! copied into its neighbours' ghost faces.

void MeshRefinement::RepairAMRFC(DvceFaceFld4D<Real> &b) {
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is;
  auto &js = indcs.js;
  auto &ks = indcs.ks;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;

  const int nmb = pmy_mesh->pmb_pack->nmb_thispack;
  const int mbs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  bool &one_d = pmy_mesh->one_d;
  bool &three_d = pmy_mesh->three_d;
  auto &repair = fc_amr_repair;

  par_for("RepairAMRFC",DevExeSpace(), 0,(nmb-1), cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (repair.d_view(m + mbs) > 0) {
      const int fi = (i - cis)*2 + is;
      const int fj = (j - cjs)*2 + js;
      const int fk = (k - cks)*2 + ks;

      if (one_d) {
        b.x1f(m,fk,fj,fi+1) = 0.5*(b.x1f(m,fk,fj,fi) + b.x1f(m,fk,fj,fi+2));
      } else {
        ProlongFCInternal(m,fk,fj,fi,three_d,b);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCC
//!  \brief Restricts cell-centered variables to coarse mesh

void MeshRefinement::RestrictCC(DvceArray5D<Real> &u, DvceArray5D<Real> &cu,
    bool is_z4c, bool lat_active_only, bool owned_source) {
  int nmb  = u.extent_int(0);  // TODO(@user): 1st index from L of in array must be NMB
  int nvar = u.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  // The block extent is the storage capacity, which can exceed the live block count.
  if (pmy_mesh->pmb_pack != nullptr) {
    nmb = std::min(nmb, pmy_mesh->pmb_pack->nmb_thispack);
  }

  auto &indcs = pmy_mesh->mb_indcs;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  // Origin of the fine source: an owned-cell source starts at (ks,js,is).  The stencil
  // below reads owned fine cells only, and a collapsed axis has origin 0 either way.
  const int fio = owned_source ? indcs.is : 0;
  const int fjo = owned_source ? indcs.js : 0;
  const int fko = owned_source ? indcs.ks : 0;
  auto &nx1 = indcs.nx1;
  auto &nx2 = indcs.nx2;
  auto &nx3 = indcs.nx3;
  auto& restrict_2nd = weights.restrict_2nd;
  auto& restrict_4th = weights.restrict_4th;
  auto& restrict_4th_edge = weights.restrict_4th_edge;
  auto *pmbp = pmy_mesh->pmb_pack;
  const bool lat_enabled = (pmbp != nullptr) && pmbp->lat_active_mask_enabled;
  DvceArray1D<int> lat_work_indices;
  if (lat_enabled) {
    lat_work_indices = lat_active_only ? pmbp->lat_active_indices.d_view :
                                        pmbp->lat_boundary_send_indices.d_view;
  }
  const int nwork = lat_enabled ?
      (lat_active_only ? pmbp->lat_nactive_thispack :
                         pmbp->lat_nboundary_send_thispack) : nmb;
  if (nwork <= 0) return;
  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrictCC-1D",DevExeSpace(), 0,nwork-1, 0,nvar-1, cis,cie,
    KOKKOS_LAMBDA(const int a, const int n, const int i) {
      const int m = lat_enabled ? lat_work_indices(a) : a;
      int finei = 2*i - cis - fio;  // correct when cis=is
      cu(m,n,cks,cjs,i) = 0.5*(u(m,n,cks,cjs,finei) + u(m,n,cks,cjs,finei+1));
    });
  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrictCC-2D",DevExeSpace(), 0,nwork-1, 0,nvar-1, cjs,cje, cis,cie,
    KOKKOS_LAMBDA(const int a, const int n, const int j, const int i) {
      const int m = lat_enabled ? lat_work_indices(a) : a;
      int finei = 2*i - cis - fio;  // correct when cis=is
      int finej = 2*j - cjs - fjo;  // correct when cjs=js
      cu(m,n,cks,j,i) = 0.25*(u(m,n,cks,finej  ,finei) + u(m,n,cks,finej  ,finei+1)
                            + u(m,n,cks,finej+1,finei) + u(m,n,cks,finej+1,finei+1));
    });

  // restrict in 3D
  } else {
    par_for("restrictCC-3D",DevExeSpace(), 0,nwork-1, 0,nvar-1, cks,cke, cjs,cje, cis,cie,
    KOKKOS_LAMBDA(const int a, const int n, const int k, const int j, const int i) {
      const int m = lat_enabled ? lat_work_indices(a) : a;
      int finei = 2*i - cis - fio;  // correct when cis=is
      int finej = 2*j - cjs - fjo;  // correct when cjs=js
      int finek = 2*k - cks - fko;  // correct when cks=ks
      if (!is_z4c) {
        cu(m,n,k,j,i) =
            0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
                + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
                + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
                + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
      } else {
        switch (indcs.ng) {
          case 2: cu(m,n,k,j,i) = RestrictInterpolation<2>(m,n,finek,finej,finei,
                          nx1,nx2,nx3,u,restrict_2nd,restrict_4th,restrict_4th_edge);
                  break;
          case 4: cu(m,n,k,j,i) = RestrictInterpolation<4>(m,n,finek,finej,finei,
                          nx1,nx2,nx3,u,restrict_2nd,restrict_4th,restrict_4th_edge);
                  break;
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFC
//! \brief Restricts face-centered variables to coarse mesh

void MeshRefinement::RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  int nmb  = b.x1f.extent_int(0);  // TODO(@user): 1st idx from L of in array must be NMB
  // The block extent is the storage capacity, which can exceed the live block count.
  if (pmy_mesh->pmb_pack != nullptr) {
    nmb = std::min(nmb, pmy_mesh->pmb_pack->nmb_thispack);
  }

  auto &cis = pmy_mesh->mb_indcs.cis;
  auto &cie = pmy_mesh->mb_indcs.cie;
  auto &cjs = pmy_mesh->mb_indcs.cjs;
  auto &cje = pmy_mesh->mb_indcs.cje;
  auto &cks = pmy_mesh->mb_indcs.cks;
  auto &cke = pmy_mesh->mb_indcs.cke;

  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrictFC-1D",DevExeSpace(), 0,nmb-1, cis,cie,
    KOKKOS_LAMBDA(const int m, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      // restrict B1
      cb.x1f(m,cks,cjs,i) = b.x1f(m,cks,cjs,finei);
      if (i==cie) {
        cb.x1f(m,cks,cjs,i+1) = b.x1f(m,cks,cjs,finei+2);
      }
      // restrict B2
      Real b2coarse = 0.5*(b.x2f(m,cks,cjs,finei) + b.x2f(m,cks,cjs,finei+1));
      cb.x2f(m,cks,cjs  ,i) = b2coarse;
      cb.x2f(m,cks,cjs+1,i) = b2coarse;
      // restrict B3
      Real b3coarse = 0.5*(b.x3f(m,cks,cjs,finei) + b.x3f(m,cks,cjs,finei+1));
      cb.x3f(m,cks  ,cjs,i) = b3coarse;
      cb.x3f(m,cks+1,cjs,i) = b3coarse;
    });

  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrictFC-2D",DevExeSpace(), 0,nmb-1, cjs,cje, cis,cie,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      // restrict B1
      cb.x1f(m,cks,j,i) = 0.5*(b.x1f(m,cks,finej,finei) + b.x1f(m,cks,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,cks,j,i+1) =
          0.5*(b.x1f(m,cks,finej,finei+2) + b.x1f(m,cks,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,cks,j,i) = 0.5*(b.x2f(m,cks,finej,finei) + b.x2f(m,cks,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,cks,j+1,i) =
          0.5*(b.x2f(m,cks,finej+2,finei) + b.x2f(m,cks,finej+2,finei+1));
      }
      // restrict B3
      Real b3coarse = 0.25*(b.x3f(m,cks,finej  ,finei) + b.x3f(m,cks,finej  ,finei+1)
                          + b.x3f(m,cks,finej+1,finei) + b.x3f(m,cks,finej+1,finei+1));
      cb.x3f(m,cks  ,j,i) = b3coarse;
      cb.x3f(m,cks+1,j,i) = b3coarse;
    });

  // restrict in 3D
  } else {
    par_for("restrictFC-3D",DevExeSpace(), 0,nmb-1, cks,cke, cjs,cje, cis,cie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      // restrict B1
      cb.x1f(m,k,j,i) =
        0.25*(b.x1f(m,finek  ,finej,finei) + b.x1f(m,finek  ,finej+1,finei)
            + b.x1f(m,finek+1,finej,finei) + b.x1f(m,finek+1,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,k,j,i+1) =
          0.25*(b.x1f(m,finek  ,finej,finei+2) + b.x1f(m,finek  ,finej+1,finei+2)
              + b.x1f(m,finek+1,finej,finei+2) + b.x1f(m,finek+1,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,k,j,i) =
        0.25*(b.x2f(m,finek  ,finej,finei) + b.x2f(m,finek  ,finej,finei+1)
            + b.x2f(m,finek+1,finej,finei) + b.x2f(m,finek+1,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,k,j+1,i) =
          0.25*(b.x2f(m,finek  ,finej+2,finei) + b.x2f(m,finek  ,finej+2,finei+1)
              + b.x2f(m,finek+1,finej+2,finei) + b.x2f(m,finek+1,finej+2,finei+1));
      }
      // restrict B3
      cb.x3f(m,k,j,i) =
        0.25*(b.x3f(m,finek,finej  ,finei) + b.x3f(m,finek,finej  ,finei+1)
            + b.x3f(m,finek,finej+1,finei) + b.x3f(m,finek,finej+1,finei+1));
      if (k==cke) {
        cb.x3f(m,k+1,j,i) =
          0.25*(b.x3f(m,finek+2,finej  ,finei) + b.x3f(m,finek+2,finej  ,finei+1)
              + b.x3f(m,finek+2,finej+1,finei) + b.x3f(m,finek+2,finej+1,finei+1));
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitInterpWghts()
//! \brief interpolation weights for prolongation and restriction
//
void MeshRefinement::InitInterpWghts() {
  auto &pro_2nd = weights.prolong_2nd;
  auto &res_2nd = weights.restrict_2nd;
  auto &pro_4th = weights.prolong_4th;
  auto &res_4th = weights.restrict_4th;
  auto &res_4th_e = weights.restrict_4th_edge;

  // Allocate memory for the arrays
  Kokkos::realloc(pro_2nd,3,3,3);
  Kokkos::realloc(res_2nd,3);
  Kokkos::realloc(pro_4th,5,5,5);
  Kokkos::realloc(res_4th,5);
  Kokkos::realloc(res_4th_e,5);

  // 2nd order prolongation weights
  const Real wght2[3] = {0.15625, 0.9375, -0.09375};
  for (int k = 0; k < 3; k++) {
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 3; i++) {
        pro_2nd.h_view(k,j,i) = wght2[k]*wght2[j]*wght2[i];
      }
    }
  }
  /*pro_2nd.h_view(0) = 0.15625;
  pro_2nd.h_view(1) = 0.9375;
  pro_2nd.h_view(2) = -0.09375;*/

  // 2nd order restriction weights
  res_2nd.h_view(0) = 0.375;
  res_2nd.h_view(1) = 0.75;
  res_2nd.h_view(2) = -0.125;

  // 4th order prolongation weights
  const Real wght4[5] = {-0.02197265625, 0.205078125, 0.9228515625,
                         -0.123046875, 0.01708984375};
  for (int k = 0; k < 5; k++) {
    for (int j = 0; j < 5; j++) {
      for (int i = 0; i < 5; i++) {
        pro_4th.h_view(k,j,i) = wght4[k]*wght4[j]*wght4[i];
      }
    }
  }
  /*pro_4th.h_view(0) = -0.02197265625;
  pro_4th.h_view(1) = 0.205078125;
  pro_4th.h_view(2) = 0.9228515625;
  pro_4th.h_view(3) = -0.123046875;
  pro_4th.h_view(4) = 0.01708984375;*/

  // 4th order restriction weights
  res_4th.h_view(0) = -0.0390625;
  res_4th.h_view(1) = 0.46875;
  res_4th.h_view(2) = 0.703125;
  res_4th.h_view(3) = -0.15625;
  res_4th.h_view(4) = 0.0234375;

  // 4th order restriction weights at edge
  res_4th_e.h_view(0) = 0.2734375;
  res_4th_e.h_view(1) = 1.09375;
  res_4th_e.h_view(2) = -0.546875;
  res_4th_e.h_view(3) = 0.21875;
  res_4th_e.h_view(4) = -0.0390625;

  // sync dual arrays
  pro_2nd.template modify<HostMemSpace>();
  pro_2nd.template sync<DevExeSpace>();
  res_2nd.template modify<HostMemSpace>();
  res_2nd.template sync<DevExeSpace>();
  pro_4th.template modify<HostMemSpace>();
  pro_4th.template sync<DevExeSpace>();
  res_4th.template modify<HostMemSpace>();
  res_4th.template sync<DevExeSpace>();
  res_4th_e.template modify<HostMemSpace>();
  res_4th_e.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void SeedGhostsFromInteriorCC()
//! \brief Every ghost cell of the first nmb MeshBlocks of a cell-centred array takes the
//! value of the nearest active cell of its own MeshBlock.  Active cells are only read, so
//! the pass has no ordering hazard.

void SeedGhostsFromInteriorCC(DvceArray5D<Real> &a, const RegionIndcs &indcs,
                              const int nmb, const bool multi_d, const bool three_d) {
  if (nmb <= 0 || a.size() == 0) return;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nvar = a.extent_int(1);
  const int n3 = a.extent_int(2), n2 = a.extent_int(3), n1 = a.extent_int(4);
  par_for("seed_ghosts_cc", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1,
          0, n1-1, KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    const int ic = (i < is) ? is : ((i > ie) ? ie : i);
    const int jc = !multi_d ? j : ((j < js) ? js : ((j > je) ? je : j));
    const int kc = !three_d ? k : ((k < ks) ? ks : ((k > ke) ? ke : k));
    if (ic != i || jc != j || kc != k) a(m,n,k,j,i) = a(m,n,kc,jc,ic);
  });
}
