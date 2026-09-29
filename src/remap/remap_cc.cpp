//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_cc.cpp
//! \brief cell-centered remap engine.
//!
//! The gas-state sampling machinery (source AMR hierarchy lookup, trilinear cell-center
//! interpolation, old-domain boundary transition band) is moved VERBATIM from
//! src/pgen/tde_external.cpp so the historical TDE remap behavior is preserved bitwise;
//! see docs/remap_module_design.md section 4.
//!
//! Two apply passes share that machinery, selected by RemapOptions::band_mode:
//!   kFloorFade  (ApplyRemapCCFloorFade) — the historical Newtonian pass: every active
//!               target cell is written, the old-domain boundary band is reconstructed
//!               from the interior profile and tapered to the ambient hydro floor.
//!   kKeepTarget (ApplyRemapCCKeep)      — the GR pass, and the only one valid there: the
//!               target's pgen state is the ambient, cells outside the source's reach are
//!               left untouched, and band cells linearly blend the plainly interpolated
//!               source conserved state with the pgen state.  Handles the gas group plus
//!               the radiation i0 group.
//!
//! The gas group carries the THERMAL energy across the interpolation and rebuilds the
//! total energy from it at the destination (BuildSourceThermalEnergy, sample_full_state):
//! interpolating E instead would turn the grid-scale kinetic-energy variance the target
//! mesh cannot represent into heat.  Total energy is therefore not conserved by
//! construction -- the dropped variance is a loss, not a transfer.
//!
//! NOT CONSERVATIVE, deliberately.  Every target cell is the plain mean of a few point
//! samples of an interpolant: there is no cell-volume weight and no donor/acceptor
//! intersection volume anywhere in the accumulators below, so mass, momentum and energy
//! move by O(h^2) of the resolved profile, and by O(1) wherever the source structure is
//! not representable on the target grid.  Fixing that is not a weighting change: it needs
//! the exact intersection volumes of two arbitrary AMR hierarchies (a supermesh),
//! sqrt(gamma)-weighted in GR, plus the matching flux-conservative restriction on the FC
//! path to keep div(B) = 0 — so the module transfers a state, not a budget.  See
//! docs/remap_usage.md "Accuracy and conservation".  The one exact case is a coincident
//! grid, which is copied (TargetCellMatchesSourceGrid).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

namespace remap {
namespace impl {

RegionSize LogicalLocationToRegionSize(const RegionSize &mesh_size,
                                       const RegionIndcs &mesh_indcs,
                                       const RegionIndcs &mb_indcs,
                                       const int root_level,
                                       const LogicalLocation &lloc) {
  RegionSize size;
  std::int32_t nmbx1 = (mesh_indcs.nx1/mb_indcs.nx1) << (lloc.level - root_level);
  std::int32_t nmbx2 = (mesh_indcs.nx2/mb_indcs.nx2) << (lloc.level - root_level);
  std::int32_t nmbx3 = (mesh_indcs.nx3/mb_indcs.nx3) << (lloc.level - root_level);
  size.x1min = (lloc.lx1 == 0) ? mesh_size.x1min
                               : LeftEdgeX(lloc.lx1, nmbx1, mesh_size.x1min,
                                   mesh_size.x1max);
  size.x1max = (lloc.lx1 == nmbx1 - 1) ? mesh_size.x1max
                                       : LeftEdgeX(lloc.lx1 + 1, nmbx1,
                                                   mesh_size.x1min, mesh_size.x1max);
  if (mesh_indcs.nx2 > 1) {
    size.x2min = (lloc.lx2 == 0) ? mesh_size.x2min
                                 : LeftEdgeX(lloc.lx2, nmbx2,
                                             mesh_size.x2min, mesh_size.x2max);
    size.x2max = (lloc.lx2 == nmbx2 - 1) ? mesh_size.x2max
                                         : LeftEdgeX(lloc.lx2 + 1, nmbx2,
                                                     mesh_size.x2min, mesh_size.x2max);
  } else {
    size.x2min = mesh_size.x2min;
    size.x2max = mesh_size.x2max;
  }
  if (mesh_indcs.nx3 > 1) {
    size.x3min = (lloc.lx3 == 0) ? mesh_size.x3min
                                 : LeftEdgeX(lloc.lx3, nmbx3,
                                             mesh_size.x3min, mesh_size.x3max);
    size.x3max = (lloc.lx3 == nmbx3 - 1) ? mesh_size.x3max
                                         : LeftEdgeX(lloc.lx3 + 1, nmbx3,
                                                     mesh_size.x3min, mesh_size.x3max);
  } else {
    size.x3min = mesh_size.x3min;
    size.x3max = mesh_size.x3max;
  }
  size.dx1 = (size.x1max - size.x1min) / static_cast<Real>(mb_indcs.nx1);
  size.dx2 = (size.x2max - size.x2min) / static_cast<Real>(mb_indcs.nx2);
  size.dx3 = (size.x3max - size.x3min) / static_cast<Real>(mb_indcs.nx3);
  return size;
}

bool RegionOverlaps(const RegionSize &a, const RegionSize &b) {
  bool x1 = (a.x1min < b.x1max) && (a.x1max > b.x1min);
  bool x2 = (a.x2min < b.x2max) && (a.x2max > b.x2min);
  bool x3 = (a.x3min < b.x3max) && (a.x3max > b.x3min);
  return x1 && x2 && x3;
}

RegionSize LocalPackBounds(Mesh *pm) {
  RegionSize bounds;
  bounds.x1min = std::numeric_limits<Real>::max();
  bounds.x1max = -std::numeric_limits<Real>::max();
  bounds.x2min = std::numeric_limits<Real>::max();
  bounds.x2max = -std::numeric_limits<Real>::max();
  bounds.x3min = std::numeric_limits<Real>::max();
  bounds.x3max = -std::numeric_limits<Real>::max();

  auto &mb_size = pm->pmb_pack->pmb->mb_size;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    bounds.x1min = std::min(bounds.x1min, mb_size.h_view(m).x1min);
    bounds.x1max = std::max(bounds.x1max, mb_size.h_view(m).x1max);
    bounds.x2min = std::min(bounds.x2min, mb_size.h_view(m).x2min);
    bounds.x2max = std::max(bounds.x2max, mb_size.h_view(m).x2max);
    bounds.x3min = std::min(bounds.x3min, mb_size.h_view(m).x3min);
    bounds.x3max = std::max(bounds.x3max, mb_size.h_view(m).x3max);
  }
  return bounds;
}

RegionSize ExpandRegionSize(const RegionSize &in, const Real pad_x1,
                            const Real pad_x2, const Real pad_x3) {
  RegionSize out = in;
  out.x1min -= pad_x1;
  out.x1max += pad_x1;
  out.x2min -= pad_x2;
  out.x2max += pad_x2;
  out.x3min -= pad_x3;
  out.x3max += pad_x3;
  return out;
}

namespace {

bool SourcePointInsideMesh(const RemapSourceData &src, const Real x, const Real y,
                           const Real z) {
  return (x >= src.mesh_size.x1min && x <= src.mesh_size.x1max &&
          y >= src.mesh_size.x2min && y <= src.mesh_size.x2max &&
          z >= src.mesh_size.x3min && z <= src.mesh_size.x3max);
}

Real SourcePointBoundaryTransitionWeight(const RemapSourceData &src,
                                         const RemapSourceBlock &block,
                                         const Real x, const Real y, const Real z,
                                         Real &xref, Real &yref, Real &zref,
                                         Real &xdeep, Real &ydeep, Real &zdeep) {
  // Build a thin transition layer that straddles the old-domain face: deep
  // inside the old box the remap uses the untouched source state, far outside
  // it stays on the true floor, and only the interface band is reconstructed.
  const Real inner_guard_source_cells =
      static_cast<Real>(std::max(src.ngh + 1, 3));
  const Real outer_guard_source_cells =
      static_cast<Real>(std::max(src.ngh + 2, 4));
  auto axis_transition_weight = [&](const Real q, const Real qmin, const Real qmax,
                                    const Real dq, Real &qref, Real &qdeep) {
    qref = q;
    qdeep = q;
    if (!(dq > 0.0) || !(qmax > qmin)) return static_cast<Real>(1.0);
    const Real inner_guard = inner_guard_source_cells * dq;
    const Real outer_guard = outer_guard_source_cells * dq;
    const Real center = static_cast<Real>(0.5) * (qmin + qmax);
    const Real inner_min = std::min(qmin + inner_guard, center);
    const Real inner_max = std::max(qmax - inner_guard, center);
    qref = std::clamp(q, inner_min, inner_max);
    if (q <= inner_min) {
      qdeep = std::clamp(inner_min + dq, inner_min, inner_max);
      const Real outer_min = qmin - outer_guard;
      if (q <= outer_min) return static_cast<Real>(0.0);
      const Real s = std::clamp((q - outer_min) /
                                    std::max(inner_min - outer_min,
                                             static_cast<Real>(1.0e-30)),
                                static_cast<Real>(0.0), static_cast<Real>(1.0));
      return s * s * s *
             (static_cast<Real>(10.0) + s *
              (static_cast<Real>(-15.0) + static_cast<Real>(6.0) * s));
    }
    if (q >= inner_max) {
      qdeep = std::clamp(inner_max - dq, inner_min, inner_max);
      const Real outer_max = qmax + outer_guard;
      if (q >= outer_max) return static_cast<Real>(0.0);
      const Real s = std::clamp((outer_max - q) /
                                    std::max(outer_max - inner_max,
                                             static_cast<Real>(1.0e-30)),
                                static_cast<Real>(0.0), static_cast<Real>(1.0));
      return s * s * s *
             (static_cast<Real>(10.0) + s *
              (static_cast<Real>(-15.0) + static_cast<Real>(6.0) * s));
    }
    return static_cast<Real>(1.0);
  };

  Real weight = axis_transition_weight(x, src.mesh_size.x1min, src.mesh_size.x1max,
                                       block.size.dx1, xref, xdeep);
  if (src.mesh_indcs.nx2 > 1 && block.size.dx2 > 0.0) {
    weight = std::min(weight, axis_transition_weight(y, src.mesh_size.x2min,
                                                     src.mesh_size.x2max, block.size.dx2,
                                                     yref, ydeep));
  } else {
    yref = y;
    ydeep = y;
  }
  if (src.mesh_indcs.nx3 > 1 && block.size.dx3 > 0.0) {
    weight = std::min(weight, axis_transition_weight(z, src.mesh_size.x3min,
                                                     src.mesh_size.x3max, block.size.dx3,
                                                     zref, zdeep));
  } else {
    zref = z;
    zdeep = z;
  }
  return weight;
}

int FindContainingSourceBlock(const RemapSourceData &src, const Real x, const Real y,
                              const Real z) {
  if (!SourcePointInsideMesh(src, x, y, z)) return -1;
  for (int lev = src.max_level; lev >= src.root_level; --lev) {
    std::int32_t nmbx1 = src.nmb_rootx1 << (lev - src.root_level);
    std::int32_t nmbx2 = src.nmb_rootx2 << (lev - src.root_level);
    std::int32_t nmbx3 = src.nmb_rootx3 << (lev - src.root_level);
    auto logical_index = [](Real xcoord, Real xmin, Real xmax, std::int32_t nmb) {
      if (nmb <= 1) return static_cast<std::int32_t>(0);
      Real frac = (xcoord - xmin) / (xmax - xmin);
      if (frac <= 0.0) return static_cast<std::int32_t>(0);
      if (frac >= 1.0) return static_cast<std::int32_t>(nmb - 1);
      std::int32_t idx = static_cast<std::int32_t>(std::floor(frac * nmb));
      if (idx < 0) idx = 0;
      if (idx >= nmb) idx = nmb - 1;
      return idx;
    };
    RemapBlockKey key;
    key.level = lev;
    key.lx1 = logical_index(x, src.mesh_size.x1min, src.mesh_size.x1max, nmbx1);
    key.lx2 = logical_index(y, src.mesh_size.x2min, src.mesh_size.x2max, nmbx2);
    key.lx3 = logical_index(z, src.mesh_size.x3min, src.mesh_size.x3max, nmbx3);
    auto it = src.block_map.find(key);
    if (it != src.block_map.end()) return it->second;
  }
  return -1;
}

// Same-grid detection.  Every sampler below takes two sub-samples per axis, at
// x_c +/- h/4 of a trilinear interpolant.  On a coincident grid that is exactly the
// separable filter [1/8, 3/4, 1/8] per axis -- transfer 0.75 + 0.25*cos(k*h), i.e. 0.5 at
// the Nyquist wavenumber and 0.125 in 3D -- so a same-grid remap used to erase 87.5% of
// the grid-scale gas power, while the FC/B path, which is nodal-exact by construction,
// kept all of it: one remap filtered the gas and the field differently.  Where the source
// cell containing the target cell center has the same width and the same center, a single
// sub-sample AT that center reproduces the source cell bit-for-bit (the trilinear weights
// collapse to [1, 0]), so the samplers drop to one sub-sample there and a same-grid remap
// is the identity again.
//
// Tolerance: 1e-10 of a cell width.  The two centers are computed by different arithmetic
// (LeftEdgeX on the source's stored mesh_size vs this run's MeshBlock bounds), so they
// agree to rounding rather than bitwise; 1e-10 sits far above that ~1e-16 and far below
// the smallest offset that could ever be intended, which is half a cell.
bool TargetCellMatchesSourceGrid(const RemapSourceData &src, const RegionSize &cell_box) {
  const Real xc = 0.5 * (cell_box.x1min + cell_box.x1max);
  const Real yc = 0.5 * (cell_box.x2min + cell_box.x2max);
  const Real zc = 0.5 * (cell_box.x3min + cell_box.x3max);
  const int iblk = FindContainingSourceBlock(src, xc, yc, zc);
  if (iblk < 0) return false;
  const RemapSourceBlock &block = src.blocks[iblk];
  constexpr Real ktol = 1.0e-10;
  auto axis_matches = [](const Real q, const Real h, const Real qmin, const Real dq,
                         const bool active) {
    if (!active) return true;                       // degenerate axis: nothing to align
    if (!(dq > 0.0)) return false;
    if (std::abs(h - dq) > ktol * dq) return false;
    const Real t = (q - qmin) / dq - static_cast<Real>(0.5);
    return (std::abs(t - std::round(t)) <= ktol);
  };
  return axis_matches(xc, cell_box.x1max - cell_box.x1min, block.size.x1min,
                      block.size.dx1, (src.mesh_indcs.nx1 > 1)) &&
         axis_matches(yc, cell_box.x2max - cell_box.x2min, block.size.x2min,
                      block.size.dx2, (src.mesh_indcs.nx2 > 1)) &&
         axis_matches(zc, cell_box.x3max - cell_box.x3min, block.size.x3min,
                      block.size.dx3, (src.mesh_indcs.nx3 > 1));
}

// generic cell-centered samplers over an arbitrary per-block CC array (nvars_arr wide);
// same interpolation logic as the gas versions below.  `undens` divides every node value
// by sqrt(gamma) AT THAT NODE (RemapSourceBlock::sqrt_gamma), which is what keeps the
// metric out of the interpolation stencil; the caller re-densitizes at the destination.
Real SampleSourceActiveCellContainingArr(const RemapSourceData &src,
                                         const std::vector<Real> RemapSourceBlock::*arr,
                                         const int nvars_arr, const int var,
                                         const Real x, const Real y, const Real z,
                                         const bool undens = false) {
  (void)nvars_arr;
  const Real xc = std::clamp(x, src.mesh_size.x1min, src.mesh_size.x1max);
  const Real yc = std::clamp(y, src.mesh_size.x2min, src.mesh_size.x2max);
  const Real zc = std::clamp(z, src.mesh_size.x3min, src.mesh_size.x3max);
  const int iblk = FindContainingSourceBlock(src, xc, yc, zc);
  if (iblk < 0) return 0.0;
  const auto &block = src.blocks[iblk];

  auto containing_index = [](const Real q, const Real qmin, const Real dq, const int nx) {
    if (nx <= 1 || !(dq > 0.0)) return 0;
    int i = static_cast<int>(std::floor((q - qmin) / dq));
    if (i < 0) i = 0;
    if (i >= nx) i = nx - 1;
    return i;
  };

  const int i = containing_index(xc, block.size.x1min, block.size.dx1, src.nx1);
  const int j = containing_index(yc, block.size.x2min, block.size.dx2, src.nx2);
  const int k = containing_index(zc, block.size.x3min, block.size.dx3, src.nx3);
  const std::size_t cell =
      ((static_cast<std::size_t>(src.ngh + k) * src.nout2 + (src.ngh + j)) * src.nout1)
      + (src.ngh + i);
  const std::size_t idx =
      static_cast<std::size_t>(var) * src.nout3 * src.nout2 * src.nout1 + cell;
  return undens ? (block.*arr)[idx] / block.sqrt_gamma[cell] : (block.*arr)[idx];
}

Real SampleSourceCellCenteredArr(const RemapSourceData &src,
                                 const RemapSourceBlock &block,
                                 const std::vector<Real> RemapSourceBlock::*arr,
                                 const int nvars_arr, const int var,
                                 const Real x, const Real y, const Real z,
                                 const bool undens = false) {
  auto interpolation_pair = [](const Real q, const Real qmin, const Real dq, const int nx,
                               int indices[2], Real centers[2], Real weights[2]) {
    if (nx <= 1 || !(dq > 0.0)) {
      indices[0] = 0;
      indices[1] = 0;
      centers[0] = qmin + static_cast<Real>(0.5) * dq;
      centers[1] = centers[0];
      weights[0] = 1.0;
      weights[1] = 0.0;
      return;
    }
    int i0 = static_cast<int>(std::floor((q - qmin) / dq - static_cast<Real>(0.5)));
    Real f = (q - qmin) / dq - static_cast<Real>(0.5) - static_cast<Real>(i0);
    if (i0 < -1) {
      i0 = -1;
      f = 0.0;
    } else if (i0 > nx - 1) {
      i0 = nx - 1;
      f = 1.0;
    }
    indices[0] = i0;
    indices[1] = i0 + 1;
    centers[0] = qmin + (static_cast<Real>(i0) + static_cast<Real>(0.5)) * dq;
    centers[1] = qmin + (static_cast<Real>(i0) + static_cast<Real>(1.5)) * dq;
    weights[0] = 1.0 - f;
    weights[1] = f;
  };

  const int nx = (src.nx1 > 1) ? 2 : 1;
  const int ny = (src.nx2 > 1) ? 2 : 1;
  const int nz = (src.nx3 > 1) ? 2 : 1;
  int iidx[2], jidx[2], kidx[2];
  Real xcenters[2], ycenters[2], zcenters[2];
  Real wx[2], wy[2], wz[2];
  interpolation_pair(x, block.size.x1min, block.size.dx1, src.nx1, iidx, xcenters, wx);
  interpolation_pair(y, block.size.x2min, block.size.dx2, src.nx2, jidx, ycenters, wy);
  interpolation_pair(z, block.size.x3min, block.size.dx3, src.nx3, kidx, zcenters, wz);

  Real sum = 0.0;
  for (int dk = 0; dk < nz; ++dk) {
    const Real wk = (nz == 1) ? 1.0 : wz[dk];
    for (int dj = 0; dj < ny; ++dj) {
      const Real wj = (ny == 1) ? 1.0 : wy[dj];
      for (int di = 0; di < nx; ++di) {
        const Real wi = (nx == 1) ? 1.0 : wx[di];
        const int ii = iidx[di];
        const int jj = jidx[dj];
        const int kk = kidx[dk];
        if (ii >= 0 && ii < src.nx1 &&
            jj >= 0 && jj < src.nx2 &&
            kk >= 0 && kk < src.nx3) {
          const std::size_t cell =
              ((static_cast<std::size_t>(src.ngh + kk) * src.nout2 + (src.ngh + jj))
               * src.nout1) + (src.ngh + ii);
          const std::size_t idx =
              static_cast<std::size_t>(var) * src.nout3 * src.nout2 * src.nout1 + cell;
          const Real node = undens ? (block.*arr)[idx] / block.sqrt_gamma[cell]
                                   : (block.*arr)[idx];
          sum += wi * wj * wk * node;
        } else {
          sum += wi * wj * wk *
                 SampleSourceActiveCellContainingArr(src, arr, nvars_arr, var,
                                                     xcenters[di], ycenters[dj],
                                                     zcenters[dk], undens);
        }
      }
    }
  }
  return sum;
}

Real SampleSourceActiveCellContaining(const RemapSourceData &src, const int var,
                                      const Real x, const Real y, const Real z) {
  return SampleSourceActiveCellContainingArr(src, &RemapSourceBlock::hydro, src.nvars,
                                             var, x, y, z);
}

Real SampleSourceCellCentered(const RemapSourceData &src, const RemapSourceBlock &block,
                              const int var, const Real x, const Real y, const Real z) {
  return SampleSourceCellCenteredArr(src, block, &RemapSourceBlock::hydro, src.nvars,
                                     var, x, y, z);
}

// The source's thermal energy density, through the SAME trilinear sampler as the
// conserved columns (one column wide, so var == 0).
Real SampleSourceThermalEnergy(const RemapSourceData &src, const RemapSourceBlock &block,
                               const Real x, const Real y, const Real z) {
  return SampleSourceCellCenteredArr(src, block, &RemapSourceBlock::eint, 1, 0, x, y, z);
}

}  // namespace

void FloorOuterSourceGhostZones(RemapSourceData &src, const int dual_energy_idx,
                                const Real floor_rho, const Real floor_eint) {
  auto floor_idx = [&](std::vector<Real> &hydro, const std::size_t idx) {
    hydro[idx] = 0.0;
  };

  auto apply_floor_state = [&](std::vector<Real> &hydro, const int k, const int j,
      const int i) {
    for (int n = 0; n < src.nvars; ++n) {
      std::size_t idx =
          ((((static_cast<std::size_t>(n) * src.nout3 + k) * src.nout2 + j)
            * src.nout1) + i);
      floor_idx(hydro, idx);
    }
    hydro[((((static_cast<std::size_t>(IDN) * src.nout3 + k) * src.nout2 + j)
            * src.nout1) + i)] = floor_rho;
    hydro[((((static_cast<std::size_t>(IEN) * src.nout3 + k) * src.nout2 + j)
            * src.nout1) + i)] = floor_eint;
    if (dual_energy_idx >= 0 && dual_energy_idx < src.nvars) {
      hydro[((((static_cast<std::size_t>(dual_energy_idx) * src.nout3 + k) * src.nout2 + j)
              * src.nout1) + i)] = floor_eint;
    }
  };

  for (auto &block : src.blocks) {
    const int level_shift = block.lloc.level - src.root_level;
    const std::int32_t nmbx1 = src.nmb_rootx1 << level_shift;
    const std::int32_t nmbx2 = src.nmb_rootx2 << level_shift;
    const std::int32_t nmbx3 = src.nmb_rootx3 << level_shift;
    const bool on_x1min = (block.lloc.lx1 == 0);
    const bool on_x1max = (block.lloc.lx1 == nmbx1 - 1);
    const bool on_x2min = (block.lloc.lx2 == 0);
    const bool on_x2max = (block.lloc.lx2 == nmbx2 - 1);
    const bool on_x3min = (block.lloc.lx3 == 0);
    const bool on_x3max = (block.lloc.lx3 == nmbx3 - 1);

    for (int k = 0; k < src.nout3; ++k) {
      for (int j = 0; j < src.nout2; ++j) {
        if (on_x1min) {
          for (int i = 0; i < src.ngh; ++i) apply_floor_state(block.hydro, k, j, i);
        }
        if (on_x1max) {
          for (int i = src.ngh + src.nx1; i < src.nout1; ++i) {
            apply_floor_state(block.hydro, k, j, i);
          }
        }
      }
    }

    if (src.mesh_indcs.nx2 > 1) {
      for (int k = 0; k < src.nout3; ++k) {
        if (on_x2min) {
          for (int j = 0; j < src.ngh; ++j) {
            for (int i = 0; i < src.nout1; ++i) apply_floor_state(block.hydro, k, j, i);
          }
        }
        if (on_x2max) {
          for (int j = src.ngh + src.nx2; j < src.nout2; ++j) {
            for (int i = 0; i < src.nout1; ++i) apply_floor_state(block.hydro, k, j, i);
          }
        }
      }
    }

    if (src.mesh_indcs.nx3 > 1) {
      if (on_x3min) {
        for (int k = 0; k < src.ngh; ++k) {
          for (int j = 0; j < src.nout2; ++j) {
            for (int i = 0; i < src.nout1; ++i) apply_floor_state(block.hydro, k, j, i);
          }
        }
      }
      if (on_x3max) {
        for (int k = src.ngh + src.nx3; k < src.nout3; ++k) {
          for (int j = 0; j < src.nout2; ++j) {
            for (int i = 0; i < src.nout1; ++i) apply_floor_state(block.hydro, k, j, i);
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// The thermal energy density of every source cell, in one extra host column (one Real per
// source cell).  This is what the CC engine interpolates in place of the total energy:
// see sample_full_state below for why.  The auxiliary is preferred where the source
// carries one, because it is the channel the source run itself trusted in the cold cells
// this fix is about; elsewhere the two agree, the source having synchronized them.
//
// Newtonian path only.  The loader has already subtracted the source's magnetic energy
// from the IEN column ("magnetic-energy swap", remap_load.cpp), so IEN is the GAS total
// energy here and the FC pass adds the new magnetic energy back after the transfer; the
// same contract, with no extra term to account for.
void BuildSourceThermalEnergy(RemapSourceData &src, const int base_nvars,
                              const Real floor_rho, const Real floor_eint) {
  const int src_dual_idx = (src.nvars > base_nvars) ? base_nvars : -1;
  const std::size_t ncells =
      static_cast<std::size_t>(src.nout3) * src.nout2 * src.nout1;
  for (auto &block : src.blocks) {
    block.eint.assign(ncells, floor_eint);
    if (block.hydro.size() < static_cast<std::size_t>(src.nvars) * ncells) continue;
    for (std::size_t c = 0; c < ncells; ++c) {
      Real eint;
      if (src_dual_idx >= 0) {
        eint = block.hydro[static_cast<std::size_t>(src_dual_idx) * ncells + c];
      } else {
        const Real rho =
            std::max(block.hydro[static_cast<std::size_t>(IDN) * ncells + c], floor_rho);
        const Real m1 = block.hydro[static_cast<std::size_t>(IM1) * ncells + c];
        const Real m2 = block.hydro[static_cast<std::size_t>(IM2) * ncells + c];
        const Real m3 = block.hydro[static_cast<std::size_t>(IM3) * ncells + c];
        eint = block.hydro[static_cast<std::size_t>(IEN) * ncells + c] -
               0.5 * (SQR(m1) + SQR(m2) + SQR(m3)) / rho;
      }
      block.eint[c] = std::max(eint, floor_eint);
    }
  }
}

// Average a target cell by sampling a few subcell points from the source AMR
// hierarchy. Restart ghost zones are not trusted as interpolation data: they
// are checkpoint payload, not guaranteed physical source states. Each trilinear
// corner instead samples the active cell containing the corresponding physical
// cell-center location, so the remap follows the stored source solution rather
// than stale internal ghosts. Far outside the old source box the larger-domain
// restart remains on the true hydro floor. Deep in the interior of the old
// box, the full source state is interpolated directly. Only within a thin band
// that straddles the old-domain interface do we reconstruct a one-sided
// primitive state from the interior reference profile and taper it smoothly
// toward the ambient floor.
void SampleRemapCellAverage(const RemapSourceData &src, const RegionSize &cell_box,
                            const std::vector<Real> &floor_state,
                            const int base_nvars,
                            const int dual_energy_idx,
                            const EOS_Data &eos,
                            const Real floor_rho, const Real floor_eint,
                            const Real floor_p,
                            const bool use_band,
                            std::vector<Real> &cell_average) {
  // On a coincident grid one sub-sample AT the cell center is an exact copy; two per axis
  // would apply the [1/8, 3/4, 1/8] filter to it (see TargetCellMatchesSourceGrid).
  const bool coincident = TargetCellMatchesSourceGrid(src, cell_box);
  const int nxs = (src.mesh_indcs.nx1 > 1 && !coincident) ? 2 : 1;
  const int nys = (src.mesh_indcs.nx2 > 1 && !coincident) ? 2 : 1;
  const int nzs = (src.mesh_indcs.nx3 > 1 && !coincident) ? 2 : 1;
  const int nsamp = nxs * nys * nzs;
  const Real dx = (cell_box.x1max - cell_box.x1min) / static_cast<Real>(nxs);
  const Real dy = (cell_box.x2max - cell_box.x2min) / static_cast<Real>(nys);
  const Real dz = (cell_box.x3max - cell_box.x3min) / static_cast<Real>(nzs);

  std::vector<Real> accum(cell_average.size(), 0.0);
  Real accum_eint = 0.0;
  std::vector<Real> sample_state(cell_average.size(), 0.0);
  std::vector<Real> full_state(cell_average.size(), 0.0);
  std::vector<Real> reference_state(cell_average.size(), 0.0);
  std::vector<Real> deeper_state(cell_average.size(), 0.0);

  auto set_floor_state = [&](std::vector<Real> &state) {
    state = floor_state;
  };

  auto set_floor_state_with_velocity = [&](const Real vx, const Real vy, const Real vz,
                                           std::vector<Real> &state) {
    state = floor_state;
    state[IM1] = floor_rho * vx;
    state[IM2] = floor_rho * vy;
    state[IM3] = floor_rho * vz;
    state[IEN] = floor_eint +
        0.5 * floor_rho * (SQR(vx) + SQR(vy) + SQR(vz));
    if (dual_energy_idx >= 0) {
      state[dual_energy_idx] = floor_eint;
    }
  };

  // Carry the THERMAL energy, not the total energy.  The interpolated E and the
  // interpolated (rho, m) do not share a kinetic energy: E_interp - ekin(m_interp,
  // rho_interp) = <e_int> + [<ekin> - ekin(<m>, <rho>)], and that bracket is non-negative
  // by Jensen -- it is the grid-scale kinetic-energy variance the target mesh cannot
  // represent.  Interpolating E hands the variance to the thermal channel as heat (on a
  // cold supersonic flow, e_int/KE ~ 1e-3, one TDE remap raised the global internal
  // energy by 32%); rebuilding E from the carried e_int drops it, which is what a remap
  // that cannot represent the fluctuation should do.
  auto sample_full_state = [&](const RemapSourceBlock &block,
                               const Real sx, const Real sy, const Real sz,
                               std::vector<Real> &state, Real &eint_state) {
    state = floor_state;
    eint_state = floor_eint;
    const Real rho_src = SampleSourceCellCentered(src, block, IDN, sx, sy, sz);
    if (!(rho_src > 0.0)) return;

    for (int n = 0; n < base_nvars && n < src.nvars; ++n) {
      state[n] = SampleSourceCellCentered(src, block, n, sx, sy, sz);
    }
    eint_state = std::max(SampleSourceThermalEnergy(src, block, sx, sy, sz), floor_eint);
    if (dual_energy_idx >= 0) {
      state[dual_energy_idx] = eint_state;
    }
    // A coincident cell is a copy (TargetCellMatchesSourceGrid), and a copy keeps the
    // source's own E: a rebuilt E reproduces it only to a rounding, and for a
    // dual-energy source not even that, E and the auxiliary being separate channels.
    if (coincident) return;
    const Real rho_state = std::max(state[IDN], floor_rho);
    state[IEN] = eint_state + 0.5 *
        (SQR(state[IM1]) + SQR(state[IM2]) + SQR(state[IM3])) / rho_state;
  };

  auto sample_full_state_at_point = [&](const Real sx, const Real sy, const Real sz,
                                        std::vector<Real> &state, Real &eint_state) {
    const int src_block_idx = FindContainingSourceBlock(src, sx, sy, sz);
    if (src_block_idx < 0) {
      state = floor_state;
      eint_state = floor_eint;
      return false;
    }
    const auto &src_block = src.blocks[src_block_idx];
    sample_full_state(src_block, sx, sy, sz, state, eint_state);

    constexpr Real kAmbientBoundaryFloorMult = 20.0;
    const Real rho_state = std::max(state[IDN], floor_rho);
    const Real p_state =
        std::max(eos.HostPressureFromRhoEint(rho_state, eint_state), floor_p);

    Real nearest_face_dist = std::min(std::abs(sx - src.mesh_size.x1min),
                                      std::abs(src.mesh_size.x1max - sx));
    int nearest_axis = 1;
    int nearest_sign = (std::abs(sx - src.mesh_size.x1min)
                        <= std::abs(src.mesh_size.x1max - sx)) ? 1 : -1;
    Real inward_shift = src_block.size.dx1;
    if (src.mesh_indcs.nx2 > 1) {
      const Real dist_ymin = std::abs(sy - src.mesh_size.x2min);
      const Real dist_ymax = std::abs(src.mesh_size.x2max - sy);
      const Real face_dist_y = std::min(dist_ymin, dist_ymax);
      if (face_dist_y < nearest_face_dist) {
        nearest_face_dist = face_dist_y;
        nearest_axis = 2;
        nearest_sign = (dist_ymin <= dist_ymax) ? 1 : -1;
        inward_shift = src_block.size.dx2;
      }
    }
    if (src.mesh_indcs.nx3 > 1) {
      const Real dist_zmin = std::abs(sz - src.mesh_size.x3min);
      const Real dist_zmax = std::abs(src.mesh_size.x3max - sz);
      const Real face_dist_z = std::min(dist_zmin, dist_zmax);
      if (face_dist_z < nearest_face_dist) {
        nearest_face_dist = face_dist_z;
        nearest_axis = 3;
        nearest_sign = (dist_zmin <= dist_zmax) ? 1 : -1;
        inward_shift = src_block.size.dx3;
      }
    }

    const bool ambient_like = (rho_state <= kAmbientBoundaryFloorMult * floor_rho &&
                               p_state <= kAmbientBoundaryFloorMult * floor_p);
    if (ambient_like &&
        nearest_face_dist <= inward_shift + static_cast<Real>(1.0e-12)) {
      Real xshift = sx;
      Real yshift = sy;
      Real zshift = sz;
      if (nearest_axis == 1) {
        xshift = std::clamp(sx + nearest_sign * inward_shift,
                            src.mesh_size.x1min, src.mesh_size.x1max);
      } else if (nearest_axis == 2) {
        yshift = std::clamp(sy + nearest_sign * inward_shift,
                            src.mesh_size.x2min, src.mesh_size.x2max);
      } else {
        zshift = std::clamp(sz + nearest_sign * inward_shift,
                            src.mesh_size.x3min, src.mesh_size.x3max);
      }
      const int shifted_block_idx = FindContainingSourceBlock(src, xshift, yshift,
          zshift);
      if (shifted_block_idx >= 0) {
        sample_full_state(src.blocks[shifted_block_idx], xshift, yshift, zshift, state,
                          eint_state);
      }
    }
    return true;
  };

  auto cubic_hermite = [&](const Real s, const Real y0, const Real y1,
                           const Real m0, const Real m1) {
    const Real s2 = s * s;
    const Real s3 = s2 * s;
    return (static_cast<Real>(2.0) * s3 - static_cast<Real>(3.0) * s2
            + static_cast<Real>(1.0)) * y0
         + (s3 - static_cast<Real>(2.0) * s2 + s) * m0
         + (static_cast<Real>(-2.0) * s3 + static_cast<Real>(3.0) * s2) * y1
         + (s3 - s2) * m1;
  };

  auto limited_endpoint_slope = [&](const Real y0, const Real y1, const Real y2) {
    const Real delta_face = y1 - y0;
    const Real delta_inner = y2 - y1;
    if (!(delta_face * delta_inner > 0.0)) return static_cast<Real>(0.0);
    const Real raw = static_cast<Real>(std::max(src.ngh + 1, 2)) * delta_inner;
    const Real limit = static_cast<Real>(2.0) * std::abs(delta_face);
    return std::clamp(raw, -limit, limit);
  };

  auto sample_boundary_transition_state = [&](const Real refx, const Real refy,
      const Real refz,
                                              const Real deepx, const Real deepy,
                                              const Real deepz, const Real interior_weight,
                                              std::vector<Real> &state, Real &eint_out) {
    constexpr Real kAmbientFaceFloorMult = 20.0;
    constexpr Real kAmbientFlatnessMult = 4.0;
    state = floor_state;
    eint_out = floor_eint;
    Real eint_ref = floor_eint;
    Real eint_deep = floor_eint;
    if (!sample_full_state_at_point(refx, refy, refz, reference_state, eint_ref)) return;
    if (!sample_full_state_at_point(deepx, deepy, deepz, deeper_state, eint_deep)) {
      deeper_state = reference_state;
      eint_deep = eint_ref;
    }

    const Real s = std::clamp(interior_weight, static_cast<Real>(0.0),
        static_cast<Real>(1.0));
    const Real rho_ref = std::max(reference_state[IDN], floor_rho);
    const Real rho_deep = std::max(deeper_state[IDN], floor_rho);
    const Real vx_ref = reference_state[IM1] / rho_ref;
    const Real vy_ref = reference_state[IM2] / rho_ref;
    const Real vz_ref = reference_state[IM3] / rho_ref;
    const Real vx_deep = deeper_state[IM1] / rho_deep;
    const Real vy_deep = deeper_state[IM2] / rho_deep;
    const Real vz_deep = deeper_state[IM3] / rho_deep;
    (void)vx_deep;
    (void)vy_deep;
    (void)vz_deep;
    const Real rho_floor_state = std::max(floor_state[IDN], floor_rho);
    const Real vx_floor = floor_state[IM1] / rho_floor_state;
    const Real vy_floor = floor_state[IM2] / rho_floor_state;
    const Real vz_floor = floor_state[IM3] / rho_floor_state;
    const Real p_ref = std::max(eos.HostPressureFromRhoEint(rho_ref, eint_ref), floor_p);
    const Real p_deep = std::max(eos.HostPressureFromRhoEint(rho_deep, eint_deep),
        floor_p);
    const Real rho_ratio = std::max(rho_ref, rho_deep) /
                           std::max(std::min(rho_ref, rho_deep), floor_rho);
    const Real p_ratio = std::max(p_ref, p_deep) /
                         std::max(std::min(p_ref, p_deep), floor_p);
    if (rho_ref <= kAmbientFaceFloorMult * floor_rho &&
        rho_deep <= kAmbientFaceFloorMult * floor_rho &&
        p_ref <= kAmbientFaceFloorMult * floor_p &&
        p_deep <= kAmbientFaceFloorMult * floor_p &&
        rho_ratio <= kAmbientFlatnessMult &&
        p_ratio <= kAmbientFlatnessMult) {
      set_floor_state(state);
      return;
    }

    const Real log_floor_rho = std::log(floor_rho);
    const Real log_rho_ref = std::log(std::max(rho_ref, floor_rho));
    const Real log_rho_deep = std::log(std::max(rho_deep, floor_rho));
    const Real rho_state = std::max(
        std::exp(cubic_hermite(s, log_floor_rho, log_rho_ref, static_cast<Real>(0.0),
                               limited_endpoint_slope(log_floor_rho, log_rho_ref,
                                                      log_rho_deep))),
        floor_rho);
    const Real log_floor_p = std::log(floor_p);
    const Real log_p_ref = std::log(std::max(p_ref, floor_p));
    const Real log_p_deep = std::log(std::max(p_deep, floor_p));
    Real p_state = std::max(
        std::exp(cubic_hermite(s, log_floor_p, log_p_ref, static_cast<Real>(0.0),
                               limited_endpoint_slope(log_floor_p, log_p_ref,
                                                      log_p_deep))),
        floor_p);
    // Keep thermal and velocity excess coupled to density excess in the
    // interface band. This preserves the smooth remap for resolved gas, but
    // suppresses pressure or momentum shoulders where the reconstructed
    // density has already fallen back close to the ambient floor.
    const Real rho_excess_ref = std::max(rho_ref - floor_rho, static_cast<Real>(0.0));
    const Real rho_excess_state = std::max(rho_state - floor_rho, static_cast<Real>(0.0));
    Real thermal_excess_frac = static_cast<Real>(0.0);
    if (rho_excess_ref > static_cast<Real>(0.0)) {
      thermal_excess_frac = std::clamp(rho_excess_state / rho_excess_ref,
                                       static_cast<Real>(0.0), static_cast<Real>(1.0));
    }
    thermal_excess_frac *= s;
    const Real velocity_excess_frac = thermal_excess_frac;
    const Real vx_state = vx_floor + velocity_excess_frac * (vx_ref - vx_floor);
    const Real vy_state = vy_floor + velocity_excess_frac * (vy_ref - vy_floor);
    const Real vz_state = vz_floor + velocity_excess_frac * (vz_ref - vz_floor);
    const Real floor_eint_state = eos.HostClampHydroInternalEnergyDensity(
        rho_state, eos.HostHydroInternalEnergyDensityFloor(rho_state));
    const Real p_floor_state = std::max(eos.HostPressureFromRhoEint(rho_state,
        floor_eint_state),
                                        floor_p);
    p_state = p_floor_state +
              thermal_excess_frac * std::max(p_state - p_floor_state,
                  static_cast<Real>(0.0));
    Real eint_state = eos.HostInternalEnergyDensityFromRhoP(rho_state, p_state);
    eint_state =
        eos.HostClampHydroInternalEnergyDensity(rho_state, std::max(eint_state,
            floor_eint));

    state[IDN] = rho_state;
    state[IM1] = state[IDN] * vx_state;
    state[IM2] = state[IDN] * vy_state;
    state[IM3] = state[IDN] * vz_state;

    const Real ekin_state = 0.5 *
        (SQR(state[IM1]) + SQR(state[IM2]) + SQR(state[IM3])) / rho_state;
    state[IEN] = eint_state + ekin_state;
    eint_out = eint_state;
    if (dual_energy_idx >= 0) {
      state[dual_energy_idx] = eint_state;
    }

    for (int n = IEN + 1; n < base_nvars && n < src.nvars; ++n) {
      state[n] = cubic_hermite(s, floor_state[n], reference_state[n],
                               static_cast<Real>(0.0),
                               limited_endpoint_slope(floor_state[n], reference_state[n],
                                                      deeper_state[n]));
    }
  };

  for (int kz = 0; kz < nzs; ++kz) {
    const Real z = cell_box.x3min + (static_cast<Real>(kz) + 0.5) * dz;
    for (int jy = 0; jy < nys; ++jy) {
      const Real y = cell_box.x2min + (static_cast<Real>(jy) + 0.5) * dy;
      for (int ix = 0; ix < nxs; ++ix) {
        const Real x = cell_box.x1min + (static_cast<Real>(ix) + 0.5) * dx;
        set_floor_state(sample_state);
        Real sample_eint = floor_eint;
        const Real xsrc = std::clamp(x, src.mesh_size.x1min, src.mesh_size.x1max);
        const Real ysrc = std::clamp(y, src.mesh_size.x2min, src.mesh_size.x2max);
        const Real zsrc = std::clamp(z, src.mesh_size.x3min, src.mesh_size.x3max);
        const int iblk = FindContainingSourceBlock(src, xsrc, ysrc, zsrc);
        if (iblk >= 0 && !use_band) {
          // pure-interpolation mode: full source state inside the old box, floor
          // outside; sample_full_state directly (no ambient near-face inward shift,
          // which is a transition-band heuristic)
          if (SourcePointInsideMesh(src, x, y, z)) {
            const int pib = FindContainingSourceBlock(src, x, y, z);
            if (pib >= 0) {
              sample_full_state(src.blocks[pib], x, y, z, full_state, sample_eint);
              sample_state = full_state;
            }
          }
        } else if (iblk >= 0) {
          const auto &src_block = src.blocks[iblk];
          Real xref = xsrc;
          Real yref = ysrc;
          Real zref = zsrc;
          Real xdeep = xsrc;
          Real ydeep = ysrc;
          Real zdeep = zsrc;
          // the exterior branch below keeps the floor thermal state and borrows only the
          // reference cell's velocity, so the e_int it samples is not carried anywhere
          Real eint_exterior = floor_eint;
          const Real transition_weight =
              SourcePointBoundaryTransitionWeight(src, src_block, x, y, z,
                                                  xref, yref, zref,
                                                  xdeep, ydeep, zdeep);
          if (transition_weight >= static_cast<Real>(1.0) &&
              SourcePointInsideMesh(src, x, y, z)) {
            if (sample_full_state_at_point(x, y, z, full_state, sample_eint)) {
              sample_state = full_state;
            }
          } else if (transition_weight > static_cast<Real>(0.0)) {
            sample_boundary_transition_state(xref, yref, zref,
                                             xdeep, ydeep, zdeep,
                                             transition_weight, sample_state,
                                             sample_eint);
          } else if (!SourcePointInsideMesh(src, x, y, z) &&
                     sample_full_state_at_point(xref, yref, zref, reference_state,
                                                eint_exterior)) {
            const Real rho_ref = std::max(reference_state[IDN], floor_rho);
            const Real vx_ref = reference_state[IM1] / rho_ref;
            const Real vy_ref = reference_state[IM2] / rho_ref;
            const Real vz_ref = reference_state[IM3] / rho_ref;
            const Real rho_floor_state = std::max(floor_state[IDN], floor_rho);
            const Real vx_floor = floor_state[IM1] / rho_floor_state;
            const Real vy_floor = floor_state[IM2] / rho_floor_state;
            const Real vz_floor = floor_state[IM3] / rho_floor_state;
            constexpr Real kAmbientExteriorFloorMult = 20.0;
            const Real rho_excess_ref = std::max(rho_ref - floor_rho,
                static_cast<Real>(0.0));
            const Real ambient_velocity_frac = std::clamp(
                rho_excess_ref /
                    std::max((kAmbientExteriorFloorMult - static_cast<Real>(1.0)) * floor_rho,
                             static_cast<Real>(1.0e-30)),
                static_cast<Real>(0.0), static_cast<Real>(1.0));
            set_floor_state_with_velocity(
                                          vx_floor + ambient_velocity_frac * (vx_ref - vx_floor),
                                          vy_floor + ambient_velocity_frac * (vy_ref - vy_floor),
                                          vz_floor + ambient_velocity_frac * (vz_ref - vz_floor),
                                          sample_state);
          }
        }
        for (std::size_t n = 0; n < cell_average.size(); ++n) {
          accum[n] += sample_state[n];
        }
        accum_eint += sample_eint;
      }
    }
  }

  const Real denom = static_cast<Real>(nsamp);
  for (std::size_t n = 0; n < cell_average.size(); ++n) {
    cell_average[n] = accum[n] / denom;
  }
  // The cell carries the averaged (rho, m, e_int); its total energy is rebuilt from them,
  // so the kinetic energy it holds is the one its own averaged momentum can carry and the
  // sub-sample kinetic variance is dropped rather than averaged in as heat (the same
  // Jensen argument as sample_full_state, one level down).  A coincident cell is a copy
  // and keeps the source's own E.
  if (!coincident) {
    const Real eint_average = std::max(accum_eint / denom, floor_eint);
    const Real rho_average = std::max(cell_average[IDN], floor_rho);
    cell_average[IEN] = eint_average + 0.5 *
        (SQR(cell_average[IM1]) + SQR(cell_average[IM2]) + SQR(cell_average[IM3]))
        / rho_average;
    if (dual_energy_idx >= 0) {
      cell_average[dual_energy_idx] = eint_average;
    }
  }
}

//----------------------------------------------------------------------------------------
// The kKeepTarget support weight at ONE point.  Factored out of the sampler below because
// remap_fc.cpp needs the identical predicate: B and the gas must agree, cell for cell, on
// where the target's own pgen state survives the remap.
Real KeepSampleWeight(const RemapSourceData &src, const Real x, const Real y,
                      const Real z, const bool use_band, const int taper_root_cells) {
  if (SourcePointInsideMesh(src, x, y, z)) return 1.0;
  if (!use_band || taper_root_cells <= 0) return 0.0;
  // normalized per-axis excess beyond the source box, in taper lengths
  const Real lx = static_cast<Real>(taper_root_cells) *
      (src.mesh_size.x1max - src.mesh_size.x1min) /
      static_cast<Real>(src.mesh_indcs.nx1);
  const Real ly = static_cast<Real>(taper_root_cells) *
      (src.mesh_size.x2max - src.mesh_size.x2min) /
      static_cast<Real>(src.mesh_indcs.nx2);
  const Real lz = static_cast<Real>(taper_root_cells) *
      (src.mesh_size.x3max - src.mesh_size.x3min) /
      static_cast<Real>(src.mesh_indcs.nx3);
  const Real ex = std::max(std::max(src.mesh_size.x1min - x, x - src.mesh_size.x1max),
                           static_cast<Real>(0.0)) / lx;
  const Real ey = std::max(std::max(src.mesh_size.x2min - y, y - src.mesh_size.x2max),
                           static_cast<Real>(0.0)) / ly;
  const Real ez = std::max(std::max(src.mesh_size.x3min - z, z - src.mesh_size.x3max),
                           static_cast<Real>(0.0)) / lz;
  const Real dist = std::max(ex, std::max(ey, ez));
  if (dist >= 1.0) return 0.0;
  const Real t = static_cast<Real>(1.0) - dist;
  return t * t * t * (t * (static_cast<Real>(6.0)*t - static_cast<Real>(15.0)) +
                      static_cast<Real>(10.0));  // quintic smoothstep
}

//----------------------------------------------------------------------------------------
// kKeepTarget CC engine.  There is no ambient floor state to fade to: the TARGET's own
// ProblemGenerator state plays that role, so this routine reports the band weight and
// lets the caller blend.
//
//   w == 0      the cell has no source support (outside the old box, or past the fade)
//               -> the caller must leave the target cell untouched
//   0 < w < 1   exterior fade cell -> caller writes w*cell_average + (1-w)*u_existing
//   w == 1      inside the source box -> caller overwrites with cell_average
//
// Unlike the Newtonian floor-fade band (which starts INSIDE the source boundary because
// the loader floors the source's outer ghost zones there), keep mode carries full weight
// all the way to the source boundary: the source data is valid up to its last active
// cell, and blending interior cells with the target's t=0 pgen state would resurrect
// initial-condition material inside a fully covered domain (measured: +11% total mass on
// a same-domain BBH remap).  The fade lives entirely OUTSIDE the source box -- the
// clamped boundary value decays to zero weight across taper_root_cells source root
// cells, the exact geometry of the FC vector-potential taper.
//
// cell_average is the w_sub-weighted mean of the sub-sample source values, so the
// caller's single blend reproduces the per-sub-sample blend averaged over the cell
// exactly.  With use_band == false the weights collapse to the 0/1 indicator of the old
// box, i.e. pure interpolation where the source covers the cell and no write elsewhere.
Real SampleKeepCellAverage(const RemapSourceData &src, const RegionSize &cell_box,
                           const std::vector<Real> RemapSourceBlock::*arr,
                           const int nvars_arr, const bool use_band,
                           const int taper_root_cells, const bool undensitize,
                           std::vector<Real> &cell_average) {
  // A coincident cell is copied, not filtered (TargetCellMatchesSourceGrid), and the
  // densitization round trip is dropped with it: source node and destination are then the
  // same point, so the two sqrt(gamma) factors are the same number and omitting both is
  // exact where dividing and re-multiplying is only exact to a rounding.
  const bool coincident = TargetCellMatchesSourceGrid(src, cell_box);
  const bool undens = undensitize && src.metric.Active() && !coincident;
  const int nxs = (src.mesh_indcs.nx1 > 1 && !coincident) ? 2 : 1;
  const int nys = (src.mesh_indcs.nx2 > 1 && !coincident) ? 2 : 1;
  const int nzs = (src.mesh_indcs.nx3 > 1 && !coincident) ? 2 : 1;
  const int nsamp = nxs * nys * nzs;
  const Real dx = (cell_box.x1max - cell_box.x1min) / static_cast<Real>(nxs);
  const Real dy = (cell_box.x2max - cell_box.x2min) / static_cast<Real>(nys);
  const Real dz = (cell_box.x3max - cell_box.x3min) / static_cast<Real>(nzs);

  std::vector<Real> accum(cell_average.size(), 0.0);
  Real wsum = 0.0;

  for (int kz = 0; kz < nzs; ++kz) {
    const Real z = cell_box.x3min + (static_cast<Real>(kz) + 0.5) * dz;
    for (int jy = 0; jy < nys; ++jy) {
      const Real y = cell_box.x2min + (static_cast<Real>(jy) + 0.5) * dy;
      for (int ix = 0; ix < nxs; ++ix) {
        const Real x = cell_box.x1min + (static_cast<Real>(ix) + 0.5) * dx;
        const Real xsrc = std::clamp(x, src.mesh_size.x1min, src.mesh_size.x1max);
        const Real ysrc = std::clamp(y, src.mesh_size.x2min, src.mesh_size.x2max);
        const Real zsrc = std::clamp(z, src.mesh_size.x3min, src.mesh_size.x3max);
        const int iblk = FindContainingSourceBlock(src, xsrc, ysrc, zsrc);
        if (iblk < 0) continue;

        const Real w = KeepSampleWeight(src, x, y, z, use_band, taper_root_cells);
        if (!(w > 0.0)) continue;
        // sample at the clamped point: identical to (x,y,z) inside the box, the held
        // boundary value in the exterior fade
        for (int n = 0; n < nvars_arr; ++n) {
          accum[n] += w * SampleSourceCellCenteredArr(src, src.blocks[iblk], arr,
                                                      nvars_arr, n, xsrc, ysrc, zsrc,
                                                      undens);
        }
        wsum += w;
      }
    }
  }

  if (!(wsum > 0.0)) {
    for (std::size_t n = 0; n < cell_average.size(); ++n) cell_average[n] = 0.0;
    return 0.0;
  }
  for (std::size_t n = 0; n < cell_average.size(); ++n) {
    cell_average[n] = accum[n] / wsum;
  }
  // Re-densitize with the metric at the DESTINATION cell center: what was interpolated is
  // the undensitized state, so what comes back out is the conserved state this target
  // cell would have carried had the source been evolved on this mesh.
  if (undens) {
    const Real sqrt_gamma = src.metric.SqrtGamma(0.5*(cell_box.x1min + cell_box.x1max),
                                                 0.5*(cell_box.x2min + cell_box.x2max),
                                                 0.5*(cell_box.x3min + cell_box.x3max));
    for (std::size_t n = 0; n < cell_average.size(); ++n) {
      cell_average[n] *= sqrt_gamma;
    }
  }
  return std::min(wsum / static_cast<Real>(nsamp), static_cast<Real>(1.0));
}

//----------------------------------------------------------------------------------------
// kFloorFade apply pass: the historical Newtonian engine, unchanged.  Every active target
// cell is written; cells with no source support get the ambient hydro floor.
namespace {

void ApplyRemapCCFloorFade(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                           const RemapSourceData &src) {
  hydro::Hydro *phyd = pmbp->phydro;
  mhd::MHD *pmhd = pmbp->pmhd;
  auto &u0 = (pmhd != nullptr) ? pmhd->u0 : phyd->u0;
  auto u0_h = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(u0_h, 0.0);

  auto &indcs = pm->mb_indcs;
  auto &mb_size = pmbp->pmb->mb_size;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nhyd = (pmhd != nullptr) ? pmhd->nmhd : phyd->nhydro;
  const int nscalars = (pmhd != nullptr) ? pmhd->nscalars : phyd->nscalars;
  const int nhydro = (pmhd != nullptr) ? pmhd->nvars : phyd->nvars;
  const int base_nvars = nhyd + nscalars;
  const bool use_dual = (pmhd != nullptr) ? pmhd->use_dual_energy : phyd->use_dual_energy;
  const int dual_energy_idx = use_dual ?
      ((pmhd != nullptr) ? pmhd->dual_energy_idx : phyd->dual_energy_idx) : -1;
  auto eos = (pmhd != nullptr) ? pmhd->peos->eos_data : phyd->peos->eos_data;
  Real floor_rho = eos.dfloor;
  // Host-side remap setup cannot query the device table directly.
  Real floor_eint = eos.HostClampHydroInternalEnergyDensity(
      floor_rho, eos.HostHydroInternalEnergyDensityFloor(floor_rho));
  Real floor_p = eos.HostPressureFromRhoEint(floor_rho, floor_eint);
  std::vector<Real> floor_state(nhydro, 0.0);
  floor_state[IDN] = floor_rho;
  floor_state[IEN] = floor_eint +
      0.5 * (SQR(floor_state[IM1]) + SQR(floor_state[IM2]) + SQR(floor_state[IM3]))
          / floor_rho;
  if (dual_energy_idx >= 0) floor_state[dual_energy_idx] = floor_eint;

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    std::vector<Real> cell_average(nhydro, 0.0);
    for (int k = ks; k <= ke; ++k) {
      Real z = CellCenterX(k - ks, indcs.nx3, mb_size.h_view(m).x3min,
          mb_size.h_view(m).x3max);
      Real zmin = mb_size.h_view(m).x3min + static_cast<Real>(k - ks) * mb_size.h_view(m).dx3;
      Real zmax = zmin + mb_size.h_view(m).dx3;
      for (int j = js; j <= je; ++j) {
        Real y = CellCenterX(j - js, indcs.nx2, mb_size.h_view(m).x2min,
            mb_size.h_view(m).x2max);
        Real ymin = mb_size.h_view(m).x2min + static_cast<Real>(j - js) * mb_size.h_view(m).dx2;
        Real ymax = ymin + mb_size.h_view(m).dx2;
        for (int i = is; i <= ie; ++i) {
          Real x = CellCenterX(i - is, indcs.nx1, mb_size.h_view(m).x1min,
                               mb_size.h_view(m).x1max);
          Real xmin = mb_size.h_view(m).x1min + static_cast<Real>(i - is) * mb_size.h_view(m).dx1;
          Real xmax = xmin + mb_size.h_view(m).dx1;

          for (int n = 0; n < nhydro; ++n) u0_h(m, n, k, j, i) = floor_state[n];

          if (opts.skip_cell && opts.skip_cell(x, y, z)) {
            continue;
          }

          RegionSize cell_box;
          cell_box.x1min = xmin;
          cell_box.x1max = xmax;
          cell_box.x2min = ymin;
          cell_box.x2max = ymax;
          cell_box.x3min = zmin;
          cell_box.x3max = zmax;
          SampleRemapCellAverage(src, cell_box,
                                 floor_state,
                                 base_nvars,
                                 dual_energy_idx, eos, floor_rho, floor_eint,
                                 floor_p,
                                 opts.use_transition_band,
                                 cell_average);
          for (int n = 0; n < nhydro; ++n) {
            u0_h(m, n, k, j, i) = cell_average[n];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u0, u0_h);
}

//----------------------------------------------------------------------------------------
// kKeepTarget apply pass.  The target's ProblemGenerator state is already in place and is
// the ambient: cells the source does not reach keep it verbatim, band cells are a linear
// blend in the CONSERVED variables, and only fully covered cells are overwritten.  No
// Newtonian floors, no pressure reconstruction, no ambient inward-shift heuristic: in GR
// the conserved variables are densitized and the primitive inversion (C2P), FOFC and the
// excision task own every physical bound that used to be enforced here.
void ApplyRemapCCKeep(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                      const RemapSourceData &src) {
  hydro::Hydro *phyd = pmbp->phydro;
  mhd::MHD *pmhd = pmbp->pmhd;
  auto &u0 = (pmhd != nullptr) ? pmhd->u0 : phyd->u0;
  auto u0_h = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(u0_h, u0);   // start from the pgen state, ghosts included

  auto &indcs = pm->mb_indcs;
  auto &mb_size = pmbp->pmb->mb_size;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nhydro = (pmhd != nullptr) ? pmhd->nvars : phyd->nvars;
  // Columns the source does not carry (e.g. a target-only dual-energy aux column; the
  // loader already refused the opposite mismatch) keep their pgen value.
  const int ngas_blend = std::min(nhydro, src.nvars);
  const bool use_band = opts.use_transition_band;

  // ---- optional aux groups -----------------------------------------------------------
  radiation::Radiation *prad = pmbp->prad;
  const bool do_i0 = src.load_i0 && (prad != nullptr);

  decltype(Kokkos::create_mirror_view(u0)) ui0_h;
  if (do_i0) {
    ui0_h = Kokkos::create_mirror_view(prad->i0);
    Kokkos::deep_copy(ui0_h, prad->i0);
  }

  std::vector<Real> cell_average(std::max(nhydro, src.nvars), 0.0);
  std::vector<Real> i0_average(do_i0 ? src.nvars_i0 : 0, 0.0);

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = ks; k <= ke; ++k) {
      Real z = CellCenterX(k - ks, indcs.nx3, mb_size.h_view(m).x3min,
                           mb_size.h_view(m).x3max);
      Real zmin = mb_size.h_view(m).x3min +
                  static_cast<Real>(k - ks) * mb_size.h_view(m).dx3;
      Real zmax = zmin + mb_size.h_view(m).dx3;
      for (int j = js; j <= je; ++j) {
        Real y = CellCenterX(j - js, indcs.nx2, mb_size.h_view(m).x2min,
                             mb_size.h_view(m).x2max);
        Real ymin = mb_size.h_view(m).x2min +
                    static_cast<Real>(j - js) * mb_size.h_view(m).dx2;
        Real ymax = ymin + mb_size.h_view(m).dx2;
        for (int i = is; i <= ie; ++i) {
          Real x = CellCenterX(i - is, indcs.nx1, mb_size.h_view(m).x1min,
                               mb_size.h_view(m).x1max);
          Real xmin = mb_size.h_view(m).x1min +
                      static_cast<Real>(i - is) * mb_size.h_view(m).dx1;
          Real xmax = xmin + mb_size.h_view(m).dx1;

          // skip_cell keeps the pgen state (excision interiors, masked regions, ...)
          if (opts.skip_cell && opts.skip_cell(x, y, z)) continue;

          RegionSize cell_box;
          cell_box.x1min = xmin;
          cell_box.x1max = xmax;
          cell_box.x2min = ymin;
          cell_box.x2max = ymax;
          cell_box.x3min = zmin;
          cell_box.x3max = zmax;

          // The dyn-GR gas columns are densitized (PrimitiveSolver stores cons*sdetg),
          // so they are interpolated with the metric divided out.  src.metric is inactive
          // for Newtonian and fixed-GR sources -- neither densitizes -- and for a stored
          // ADM backend, and then this is the historical verbatim interpolation.
          const Real wg = SampleKeepCellAverage(src, cell_box, &RemapSourceBlock::hydro,
                                                src.nvars, use_band,
                                                opts.b_taper_root_cells, true,
                                                cell_average);
          if (wg > 0.0) {
            for (int n = 0; n < ngas_blend; ++n) {
              u0_h(m, n, k, j, i) = wg * cell_average[n] +
                                    (1.0 - wg) * u0_h(m, n, k, j, i);
            }
          }

          if (do_i0) {
            const Real w = SampleKeepCellAverage(src, cell_box, &RemapSourceBlock::i0,
                                                 src.nvars_i0, use_band,
                                                 opts.b_taper_root_cells, false,
                                                 i0_average);
            if (w > 0.0) {
              for (int n = 0; n < src.nvars_i0; ++n) {
                const Real v = w * i0_average[n] + (1.0 - w) * ui0_h(m, n, k, j, i);
                ui0_h(m, n, k, j, i) = std::max(v, static_cast<Real>(0.0));
              }
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u0, u0_h);
  if (do_i0) Kokkos::deep_copy(prad->i0, ui0_h);
}

}  // namespace

void ApplyRemapCC(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                  const RemapSourceData &src) {
  if (opts.band_mode == RemapBandMode::kKeepTarget) {
    ApplyRemapCCKeep(pm, pmbp, opts, src);
  } else {
    ApplyRemapCCFloorFade(pm, pmbp, opts, src);
  }
}

}  // namespace impl
}  // namespace remap
