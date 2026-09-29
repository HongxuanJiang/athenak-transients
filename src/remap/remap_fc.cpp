//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_fc.cpp
//! \brief divergence-free face-centered B remap via vector-potential interpolation.
//!
//! Pipeline (docs/remap_module_design.md section 5): the loader restricted the source
//! B faces exactly onto the source-root covering grid.  Here we (1) invert the discrete
//! curl on that grid in the gauge A1 == 0 (exact prefix sums, so curl(A) reproduces the
//! covering B; the redundant component is a telescoped identity iff the source was
//! divergence-free — its residual is reported), (2) normalize the gauge by subtracting
//! the boundary-shell mean of each component, (3) sample a separable Catmull-Rom (C1)
//! interpolant of A at every target edge — averaging two half-edge samples where the edge
//! abuts a finer target neighbor, mirroring the canonical MHD pgen contract
//! (gr_torus.cpp) so shared fine/coarse face fluxes are restriction-consistent — with a
//! smoothstep taper of A outside the source box, and (4) take the discrete curl per
//! target block.  Every target block is therefore divergence-free to machine precision.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

namespace remap {
namespace impl {

namespace {

// 1D Catmull-Rom on four consecutive nodes; f in [0,1] between p0 and p1.
inline Real CatmullRom(const Real f, const Real pm1, const Real p0,
                       const Real p1, const Real p2) {
  return p0 + 0.5*f*((p1 - pm1) + f*((2.0*pm1 - 5.0*p0 + 4.0*p1 - p2)
                     + f*(3.0*(p0 - p1) + p2 - pm1)));
}

// One separable Catmull-Rom lattice: nodes at q0N + idx*dqN, idx in [0, nN-1].
struct EdgeLattice {
  const std::vector<Real> *data = nullptr;
  int n1 = 0, n2 = 0, n3 = 0;                    // node counts (i fastest)
  Real x0 = 0.0, y0 = 0.0, z0 = 0.0;
  Real dx = 0.0, dy = 0.0, dz = 0.0;

  Real Node(const int k, const int j, const int i) const {
    return (*data)[(static_cast<std::size_t>(k) * n2 + j) * n1 + i];
  }

  Real Eval(const Real x, const Real y, const Real z) const {
    auto locate = [](const Real q, const Real q0, const Real dq, const int n,
                     int &i0, Real &f) {
      Real t = (q - q0) / dq;
      Real tf = std::floor(t);
      i0 = static_cast<int>(tf);
      f = t - tf;
      if (i0 < 0) { i0 = 0; f = 0.0; }
      if (i0 > n - 2) { i0 = n - 2; f = 1.0; }
    };
    auto cl = [](int idx, const int n) {
      return std::min(std::max(idx, 0), n - 1);
    };
    int i0, j0, k0;
    Real fx, fy, fz;
    locate(x, x0, dx, n1, i0, fx);
    locate(y, y0, dy, n2, j0, fy);
    locate(z, z0, dz, n3, k0, fz);
    Real cz[4];
    for (int dk = -1; dk <= 2; ++dk) {
      const int kk = cl(k0 + dk, n3);
      Real cy[4];
      for (int dj = -1; dj <= 2; ++dj) {
        const int jj = cl(j0 + dj, n2);
        const Real pm1 = Node(kk, jj, cl(i0 - 1, n1));
        const Real p0 = Node(kk, jj, cl(i0, n1));
        const Real p1 = Node(kk, jj, cl(i0 + 1, n1));
        const Real p2 = Node(kk, jj, cl(i0 + 2, n1));
        cy[dj + 1] = CatmullRom(fx, pm1, p0, p1, p2);
      }
      cz[dk + 1] = CatmullRom(fy, cy[0], cy[1], cy[2], cy[3]);
    }
    return CatmullRom(fz, cz[0], cz[1], cz[2], cz[3]);
  }
};

inline Real SmoothStep5(const Real u) {
  const Real s = std::clamp(u, static_cast<Real>(0.0), static_cast<Real>(1.0));
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}

// Interpolated, gauge-normalized, tapered vector potential of the covering grid.
struct PotentialSampler {
  EdgeLattice a2, a3;
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  Real taper1, taper2, taper3;   // taper widths (<= 0 means hard cutoff)
  Real off2, off3;               // shell means to subtract

  Real Window(const Real x, const Real y, const Real z) const {
    auto axis = [](const Real q, const Real qmin, const Real qmax, const Real taper) {
      Real d = 0.0;
      if (q < qmin) d = qmin - q;
      if (q > qmax) d = q - qmax;
      if (d <= 0.0) return static_cast<Real>(1.0);
      if (!(taper > 0.0)) return static_cast<Real>(0.0);
      return 1.0 - SmoothStep5(d / taper);
    };
    return axis(x, x1min, x1max, taper1) * axis(y, x2min, x2max, taper2) *
           axis(z, x3min, x3max, taper3);
  }

  Real A2(const Real x, const Real y, const Real z) const {
    const Real w = Window(x, y, z);
    if (!(w > 0.0)) return 0.0;
    const Real xc = std::clamp(x, x1min, x1max);
    const Real yc = std::clamp(y, x2min, x2max);
    const Real zc = std::clamp(z, x3min, x3max);
    return w * (a2.Eval(xc, yc, zc) - off2);
  }

  Real A3(const Real x, const Real y, const Real z) const {
    const Real w = Window(x, y, z);
    if (!(w > 0.0)) return 0.0;
    const Real xc = std::clamp(x, x1min, x1max);
    const Real yc = std::clamp(y, x2min, x2max);
    const Real zc = std::clamp(z, x3min, x3max);
    return w * (a3.Eval(xc, yc, zc) - off3);
  }
};

}  // namespace

void BuildCoveringPotential(RemapSourceData &src, const RemapOptions &opts) {
  RemapCoveringField &cov = src.cov;
  const int n1 = cov.n1, n2 = cov.n2, n3 = cov.n3;
  const Real dx = cov.dx1, dy = cov.dx2, dz = cov.dx3;

  auto b1 = [&](int k, int j, int i) -> Real {
    return cov.b1f[(static_cast<std::size_t>(k) * n2 + j) * (n1 + 1) + i];
  };
  auto b2 = [&](int k, int j, int i) -> Real {
    return cov.b2f[(static_cast<std::size_t>(k) * (n2 + 1) + j) * n1 + i];
  };
  auto b3 = [&](int k, int j, int i) -> Real {
    return cov.b3f[(static_cast<std::size_t>(k) * n2 + j) * n1 + i];
  };

  // a2(k in [0..n3], j in [0..n2-1], i in [0..n1]); a3(k in [0..n3-1], j in [0..n2],
  // i in [0..n1]).
  cov.a2.assign(static_cast<std::size_t>(n3 + 1) * n2 * (n1 + 1), 0.0);
  cov.a3.assign(static_cast<std::size_t>(n3) * (n2 + 1) * (n1 + 1), 0.0);
  auto a2i = [&](int k, int j, int i) -> Real & {
    return cov.a2[(static_cast<std::size_t>(k) * n2 + j) * (n1 + 1) + i];
  };
  auto a3i = [&](int k, int j, int i) -> Real & {
    return cov.a3[(static_cast<std::size_t>(k) * (n2 + 1) + j) * (n1 + 1) + i];
  };

  // i = 0 plane: a2 = 0; a3 integrates B1 along y so the redundant (b1) component holds
  // there; the x-sweeps then extend both while reproducing b2/b3 exactly.
  for (int k = 0; k < n3; ++k) {
    for (int j = 0; j < n2; ++j) {
      a3i(k, j + 1, 0) = a3i(k, j, 0) + dy * b1(k, j, 0);
    }
  }
  for (int k = 0; k <= n3; ++k) {
    for (int j = 0; j < n2; ++j) {
      for (int i = 0; i < n1; ++i) {
        a2i(k, j, i + 1) = a2i(k, j, i) + dx * b3(k, j, i);
      }
    }
  }
  for (int k = 0; k < n3; ++k) {
    for (int j = 0; j <= n2; ++j) {
      for (int i = 0; i < n1; ++i) {
        a3i(k, j, i + 1) = a3i(k, j, i) - dx * b2(k, j, i);
      }
    }
  }

  // consistency of the redundant component: curl(A)_1 vs the covering b1
  Real b1_err = 0.0;
  for (int k = 0; k < n3; ++k) {
    for (int j = 0; j < n2; ++j) {
      for (int i = 0; i <= n1; ++i) {
        const Real rec = (a3i(k, j + 1, i) - a3i(k, j, i)) / dy -
                         (a2i(k + 1, j, i) - a2i(k, j, i)) / dz;
        b1_err = std::max(b1_err, std::abs(rec - b1(k, j, i)));
      }
    }
  }
  cov.inv_curl_b1_err = b1_err;

  // gauge normalization: subtract each component's boundary-shell mean
  auto shell_mean = [&](const std::vector<Real> &a, const int an1, const int an2,
                        const int an3) {
    Real sum = 0.0;
    std::size_t cnt = 0;
    for (int k = 0; k < an3; ++k) {
      for (int j = 0; j < an2; ++j) {
        for (int i = 0; i < an1; ++i) {
          if (k != 0 && k != an3 - 1 && j != 0 && j != an2 - 1 &&
              i != 0 && i != an1 - 1) {
            continue;
          }
          sum += a[(static_cast<std::size_t>(k) * an2 + j) * an1 + i];
          ++cnt;
        }
      }
    }
    return (cnt > 0) ? sum / static_cast<Real>(cnt) : 0.0;
  };
  cov.a2_shell_mean = shell_mean(cov.a2, n1 + 1, n2, n3 + 1);
  cov.a3_shell_mean = shell_mean(cov.a3, n1 + 1, n2 + 1, n3);

  // boundary-field diagnostic for the taper warning
  Real bmax = 0.0, bshell = 0.0;
  for (int k = 0; k < n3; ++k) {
    for (int j = 0; j < n2; ++j) {
      for (int i = 0; i <= n1; ++i) {
        const Real v = std::abs(b1(k, j, i));
        bmax = std::max(bmax, v);
        if (i == 0 || i == n1 || j == 0 || j == n2 - 1 || k == 0 || k == n3 - 1) {
          bshell = std::max(bshell, v);
        }
      }
    }
  }
  for (int k = 0; k < n3; ++k) {
    for (int j = 0; j <= n2; ++j) {
      for (int i = 0; i < n1; ++i) {
        const Real v = std::abs(b2(k, j, i));
        bmax = std::max(bmax, v);
        if (j == 0 || j == n2 || i == 0 || i == n1 - 1 || k == 0 || k == n3 - 1) {
          bshell = std::max(bshell, v);
        }
      }
    }
  }
  for (int k = 0; k <= n3; ++k) {
    for (int j = 0; j < n2; ++j) {
      for (int i = 0; i < n1; ++i) {
        const Real v = std::abs(b3(k, j, i));
        bmax = std::max(bmax, v);
        if (k == 0 || k == n3 || i == 0 || i == n1 - 1 || j == 0 || j == n2 - 1) {
          bshell = std::max(bshell, v);
        }
      }
    }
  }
  cov.interior_bmax = bmax;
  cov.boundary_bmax = bshell;

  if (opts.b_report && global_variable::my_rank == 0) {
    std::cout << "--- remap FC (vector-potential) diagnostics ---" << std::endl
              << "covering grid            = " << n1 << " x " << n2 << " x " << n3
              << " (source root)" << std::endl
              << "inverse-curl b1 residual = " << cov.inv_curl_b1_err
              << "  (source div(B) proxy; ~1e-15 * |B|/dx for CT sources)" << std::endl
              << "max |B| interior/shell   = " << cov.interior_bmax << " / "
              << cov.boundary_bmax << std::endl
              << "taper width (root cells) = " << opts.b_taper_root_cells << std::endl
              << std::endl;
  }
  // The sheet-current warning is not silenceable by b_report.  1e-2 relative is a
  // deliberate choice: compact fields have Gaussian-tail boundary values around 1e-3
  // of the peak, which is cosmetically harmless in the taper band.
  if (global_variable::my_rank == 0 &&
      cov.boundary_bmax > 1.0e-2 * std::max(cov.interior_bmax, 1.0e-300)) {
    std::cout << "WARNING (remap FC): the source B does not vanish at the old domain "
              << "boundary (shell max " << cov.boundary_bmax << " vs interior max "
              << cov.interior_bmax << "); the taper band outside the old box will "
              << "carry sheet currents." << std::endl << std::endl;
  }

  // The inverse-curl residual used to be PRINTED and never acted on, so with
  // b_report = false a source whose B1 cannot be recovered from A at all was accepted in
  // silence and the target got a different field.  Gate it, and gate it outside b_report.
  // Scale: the prefix sums accumulate n_root roundoffs of size eps*|A| and are then
  // divided by one cell width, so a discretely divergence-free (CT) source lands near
  // eps*n_root*|B| ~ 1e-13*|B| even on a 512-cell root grid.  1e-8 is five orders above
  // that floor and still catches a source that was never CT-evolved; 1e-2 means curl(A)
  // simply is not the source's B1 and the remap would install a fabricated field.
  const Real b1_rel = cov.inv_curl_b1_err / std::max(cov.interior_bmax, 1.0e-300);
  if (b1_rel > 1.0e-2) {
    std::cout << "### FATAL ERROR in remap module (FC)" << std::endl
              << "The source B field is not discretely divergence-free: the exact "
              << "inverse curl reproduces b1 only to " << cov.inv_curl_b1_err << " ("
              << b1_rel << " of max |B|).  The vector-potential remap would install a "
              << "field that is divergence-free but is NOT the source's field."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0 && b1_rel > 1.0e-8) {
    std::cout << "WARNING (remap FC): inverse-curl b1 residual " << cov.inv_curl_b1_err
              << " is " << b1_rel << " of max |B| -- far above the ~1e-13 roundoff floor "
              << "of a CT-evolved source.  The remapped B reproduces b2/b3 exactly but "
              << "b1 only to that residual." << std::endl << std::endl;
  }
}

void ApplyRemapFC(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                  const RemapSourceData &src) {
  mhd::MHD *pmhd = pmbp->pmhd;
  const RemapCoveringField &cov = src.cov;

  PotentialSampler pot;
  pot.a2.data = &cov.a2;
  pot.a2.n1 = cov.n1 + 1;
  pot.a2.n2 = cov.n2;
  pot.a2.n3 = cov.n3 + 1;
  pot.a2.x0 = cov.x1min;
  pot.a2.y0 = cov.x2min + 0.5 * cov.dx2;
  pot.a2.z0 = cov.x3min;
  pot.a2.dx = cov.dx1;
  pot.a2.dy = cov.dx2;
  pot.a2.dz = cov.dx3;
  pot.a3.data = &cov.a3;
  pot.a3.n1 = cov.n1 + 1;
  pot.a3.n2 = cov.n2 + 1;
  pot.a3.n3 = cov.n3;
  pot.a3.x0 = cov.x1min;
  pot.a3.y0 = cov.x2min;
  pot.a3.z0 = cov.x3min + 0.5 * cov.dx3;
  pot.a3.dx = cov.dx1;
  pot.a3.dy = cov.dx2;
  pot.a3.dz = cov.dx3;
  pot.x1min = cov.x1min;
  pot.x1max = cov.x1max;
  pot.x2min = cov.x2min;
  pot.x2max = cov.x2max;
  pot.x3min = cov.x3min;
  pot.x3max = cov.x3max;
  const Real tw = static_cast<Real>(opts.b_taper_root_cells);
  pot.taper1 = tw * cov.dx1;
  pot.taper2 = tw * cov.dx2;
  pot.taper3 = tw * cov.dx3;
  pot.off2 = cov.a2_shell_mean;
  pot.off3 = cov.a3_shell_mean;

  auto &indcs = pm->mb_indcs;
  auto &mb_size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  // Magnetic-energy add-back is the exact inverse of the loader's subtraction, and both
  // are Newtonian-only: see remap_load.cpp.  In GR the conserved energy column was never
  // decomposed, so nothing is added back and u0 is left as the CC engine wrote it.
  const bool add_emag = !src.gr_mode;
  // kKeepTarget: the CC pass deliberately left the TARGET's own pgen state standing
  // wherever the source has no support, so the FC pass must respect the same region.  v1
  // zeroed b0/bcc0 over the whole mesh and rebuilt it from A everywhere, which drove B to
  // exactly zero outside the source box plus taper while the gas there still carried the
  // pgen's magnetized conserved state -- in GR that state then decodes against B = 0, so
  // the electromagnetic energy and momentum are read back as fluid.  Start from the pgen
  // field instead and write only the faces the remap actually reaches.
  const bool keep_mode = (opts.band_mode == RemapBandMode::kKeepTarget);
  auto &b0 = pmhd->b0;
  auto &bcc0 = pmhd->bcc0;
  auto &u0 = pmhd->u0;
  auto x1f_h = Kokkos::create_mirror_view(b0.x1f);
  auto x2f_h = Kokkos::create_mirror_view(b0.x2f);
  auto x3f_h = Kokkos::create_mirror_view(b0.x3f);
  auto bcc_h = Kokkos::create_mirror_view(bcc0);
  auto u0_h = Kokkos::create_mirror_view(u0);
  if (keep_mode) {
    Kokkos::deep_copy(x1f_h, b0.x1f);
    Kokkos::deep_copy(x2f_h, b0.x2f);
    Kokkos::deep_copy(x3f_h, b0.x3f);
    Kokkos::deep_copy(bcc_h, bcc0);
  } else {
    Kokkos::deep_copy(x1f_h, 0.0);
    Kokkos::deep_copy(x2f_h, 0.0);
    Kokkos::deep_copy(x3f_h, 0.0);
    Kokkos::deep_copy(bcc_h, 0.0);
  }
  if (add_emag) {
    Kokkos::deep_copy(u0_h, u0);  // CC remap already installed the gas state
  }
  // Seam diagnostic (keep mode only): a face-wise mixture of two divergence-free fields
  // is not itself divergence-free where they meet, so the one cell layer at the outer
  // edge of the taper can carry div(B) ~ |B_pgen|/h.  Report it -- not suppressible by
  // b_report, because it is the price of keeping the pgen field and must not be silent.
  Real seam_divb = 0.0;    // max |div(B)| * h over the local blocks
  Real seam_bmax = 0.0;    // max |B| over the same, for the relative report

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  const int ne1 = nx1 + 1, ne2 = nx2 + 1, ne3 = nx3 + 1;
  std::vector<Real> a2b(static_cast<std::size_t>(ne3) * ne2 * ne1);
  std::vector<Real> a3b(a2b.size());
  // Per-edge support flags: is the taper window nonzero where this edge potential was
  // sampled?  The test is on the WINDOW, not on A itself -- A can be exactly 0.0 for
  // reasons that are not "the remap has nothing to say here" (a source with B == 0 makes
  // A vanish identically), and those cells must still be overwritten.
  std::vector<char> s2b(a2b.size()), s3b(a3b.size());
  auto A2B = [&](int kk, int jj, int ii) -> Real & {
    return a2b[(static_cast<std::size_t>(kk) * ne2 + jj) * ne1 + ii];
  };
  auto A3B = [&](int kk, int jj, int ii) -> Real & {
    return a3b[(static_cast<std::size_t>(kk) * ne2 + jj) * ne1 + ii];
  };
  auto S2B = [&](int kk, int jj, int ii) -> char & {
    return s2b[(static_cast<std::size_t>(kk) * ne2 + jj) * ne1 + ii];
  };
  auto S3B = [&](int kk, int jj, int ii) -> char & {
    return s3b[(static_cast<std::size_t>(kk) * ne2 + jj) * ne1 + ii];
  };

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const Real x1min = mb_size.h_view(m).x1min;
    const Real x1max = mb_size.h_view(m).x1max;
    const Real x2min = mb_size.h_view(m).x2min;
    const Real x2max = mb_size.h_view(m).x2max;
    const Real x3min = mb_size.h_view(m).x3min;
    const Real x3max = mb_size.h_view(m).x3max;
    const Real dx1 = mb_size.h_view(m).dx1;
    const Real dx2 = mb_size.h_view(m).dx2;
    const Real dx3 = mb_size.h_view(m).dx3;
    const int lev = mblev.h_view(m);

    auto finer = [&](int idx) { return nghbr.h_view(m, idx).lev > lev; };

    for (int kk = 0; kk < ne3; ++kk) {
      const Real x3fc = LeftEdgeX(kk, nx3, x3min, x3max);
      const Real x3v = CellCenterX(kk, nx3, x3min, x3max);
      for (int jj = 0; jj < ne2; ++jj) {
        const Real x2fc = LeftEdgeX(jj, nx2, x2min, x2max);
        const Real x2v = CellCenterX(jj, nx2, x2min, x2max);
        for (int ii = 0; ii < ne1; ++ii) {
          const Real x1fc = LeftEdgeX(ii, nx1, x1min, x1max);

          Real a2v = pot.A2(x1fc, x2v, x3fc);
          Real a3v = pot.A3(x1fc, x2fc, x3v);
          char s2v = (pot.Window(x1fc, x2v, x3fc) > 0.0) ? 1 : 0;
          char s3v = (pot.Window(x1fc, x2fc, x3v) > 0.0) ? 1 : 0;

          // Where the edge abuts a FINER target neighbor, replace the midpoint sample
          // with the mean of the two half-edge midpoints so the coarse edge line
          // integral equals the sum of the fine ones (canonical SMR contract,
          // gr_torus.cpp:596-691).
          // a2 lives along x2: corrected on x1 faces, x3 faces, and x1x3 edges.
          if ((ii == 0 && (finer(0) || finer(1) || finer(2) || finer(3))) ||
              (ii == ne1 - 1 && (finer(4) || finer(5) || finer(6) || finer(7))) ||
              (kk == 0 && (finer(24) || finer(25) || finer(26) || finer(27))) ||
              (kk == ne3 - 1 && (finer(28) || finer(29) || finer(30) || finer(31))) ||
              (ii == 0 && kk == 0 && (finer(32) || finer(33))) ||
              (ii == ne1 - 1 && kk == 0 && (finer(34) || finer(35))) ||
              (ii == 0 && kk == ne3 - 1 && (finer(36) || finer(37))) ||
              (ii == ne1 - 1 && kk == ne3 - 1 && (finer(38) || finer(39)))) {
            a2v = 0.5 * (pot.A2(x1fc, x2v - 0.25 * dx2, x3fc) +
                         pot.A2(x1fc, x2v + 0.25 * dx2, x3fc));
            s2v = (pot.Window(x1fc, x2v - 0.25 * dx2, x3fc) > 0.0 ||
                   pot.Window(x1fc, x2v + 0.25 * dx2, x3fc) > 0.0) ? 1 : 0;
          }
          // a3 lives along x3: corrected on x1 faces, x2 faces, and x1x2 edges.
          if ((ii == 0 && (finer(0) || finer(1) || finer(2) || finer(3))) ||
              (ii == ne1 - 1 && (finer(4) || finer(5) || finer(6) || finer(7))) ||
              (jj == 0 && (finer(8) || finer(9) || finer(10) || finer(11))) ||
              (jj == ne2 - 1 && (finer(12) || finer(13) || finer(14) || finer(15))) ||
              (ii == 0 && jj == 0 && (finer(16) || finer(17))) ||
              (ii == ne1 - 1 && jj == 0 && (finer(18) || finer(19))) ||
              (ii == 0 && jj == ne2 - 1 && (finer(20) || finer(21))) ||
              (ii == ne1 - 1 && jj == ne2 - 1 && (finer(22) || finer(23)))) {
            a3v = 0.5 * (pot.A3(x1fc, x2fc, x3v - 0.25 * dx3) +
                         pot.A3(x1fc, x2fc, x3v + 0.25 * dx3));
            s3v = (pot.Window(x1fc, x2fc, x3v - 0.25 * dx3) > 0.0 ||
                   pot.Window(x1fc, x2fc, x3v + 0.25 * dx3) > 0.0) ? 1 : 0;
          }

          A2B(kk, jj, ii) = a2v;
          A3B(kk, jj, ii) = a3v;
          S2B(kk, jj, ii) = s2v;
          S3B(kk, jj, ii) = s3v;
        }
      }
    }

    // Which faces does the remap own?  Exactly those whose curl stencil touches an edge
    // where the taper window is still alive; beyond the taper the window is EXACTLY zero
    // (SmoothStep5 clamps its argument, so 1 - SmoothStep5(1) is 0.0, not an epsilon),
    // and the face keeps whatever the target's pgen put there.  In floor-fade mode every
    // face is owned, which is the historical behavior bit for bit.
    auto owns_x1f = [&](const int kk, const int jj, const int ii) {
      if (!keep_mode) return true;
      return (S3B(kk, jj + 1, ii) || S3B(kk, jj, ii) ||
              S2B(kk + 1, jj, ii) || S2B(kk, jj, ii));
    };
    auto owns_x2f = [&](const int kk, const int jj, const int ii) {
      if (!keep_mode) return true;
      return (S3B(kk, jj, ii + 1) || S3B(kk, jj, ii));
    };
    auto owns_x3f = [&](const int kk, const int jj, const int ii) {
      if (!keep_mode) return true;
      return (S2B(kk, jj, ii + 1) || S2B(kk, jj, ii));
    };

    // discrete curl (gauge A1 == 0: the a1 terms drop)
    for (int kk = 0; kk < nx3; ++kk) {
      for (int jj = 0; jj < nx2; ++jj) {
        for (int ii = 0; ii <= nx1; ++ii) {
          if (!owns_x1f(kk, jj, ii)) continue;
          x1f_h(m, ks + kk, js + jj, is + ii) =
              (A3B(kk, jj + 1, ii) - A3B(kk, jj, ii)) / dx2 -
              (A2B(kk + 1, jj, ii) - A2B(kk, jj, ii)) / dx3;
        }
      }
    }
    for (int kk = 0; kk < nx3; ++kk) {
      for (int jj = 0; jj <= nx2; ++jj) {
        for (int ii = 0; ii < nx1; ++ii) {
          if (!owns_x2f(kk, jj, ii)) continue;
          x2f_h(m, ks + kk, js + jj, is + ii) =
              -(A3B(kk, jj, ii + 1) - A3B(kk, jj, ii)) / dx1;
        }
      }
    }
    for (int kk = 0; kk <= nx3; ++kk) {
      for (int jj = 0; jj < nx2; ++jj) {
        for (int ii = 0; ii < nx1; ++ii) {
          if (!owns_x3f(kk, jj, ii)) continue;
          x3f_h(m, ks + kk, js + jj, is + ii) =
              (A2B(kk, jj, ii + 1) - A2B(kk, jj, ii)) / dx1;
        }
      }
    }

    // cell-centered field + magnetic energy back onto the remapped total energy
    const Real hmin = std::min(dx1, std::min(dx2, dx3));
    for (int kk = 0; kk < nx3; ++kk) {
      for (int jj = 0; jj < nx2; ++jj) {
        for (int ii = 0; ii < nx1; ++ii) {
          const int k = ks + kk, j = js + jj, i = is + ii;
          const Real emag_old = 0.5 * (SQR(bcc_h(m, IBX, k, j, i)) +
                                       SQR(bcc_h(m, IBY, k, j, i)) +
                                       SQR(bcc_h(m, IBZ, k, j, i)));
          const bool touched = owns_x1f(kk, jj, ii) || owns_x1f(kk, jj, ii + 1) ||
                               owns_x2f(kk, jj, ii) || owns_x2f(kk, jj + 1, ii) ||
                               owns_x3f(kk, jj, ii) || owns_x3f(kk + 1, jj, ii);
          const Real bx = 0.5 * (x1f_h(m, k, j, i) + x1f_h(m, k, j, i + 1));
          const Real by = 0.5 * (x2f_h(m, k, j, i) + x2f_h(m, k, j + 1, i));
          const Real bz = 0.5 * (x3f_h(m, k, j, i) + x3f_h(m, k + 1, j, i));
          bcc_h(m, IBX, k, j, i) = bx;
          bcc_h(m, IBY, k, j, i) = by;
          bcc_h(m, IBZ, k, j, i) = bz;
          if (add_emag && touched) {
            // Gate and weight the add-back with the CC pass's own blend weight.  The keep
            // pass wrote w*E_source_gas + (1-w)*E_pgen_total, and that pgen fraction
            // ALREADY carries the pgen's magnetic energy, so only the remapped fraction
            // may take the new one; a skipped cell (w = 0) keeps the pgen energy whole
            // and just swaps its magnetic part.  In floor-fade mode w == 1 and
            // emag_old == 0 everywhere, so this is the historical add-back bit for bit.
            // The CC pass averages this weight over the cell's sub-samples; the value at
            // the cell center agrees exactly where it matters (1 under full coverage, 0
            // beyond the taper) and differs only by O(h*dw/dx) inside the collar itself.
            const Real x1v = CellCenterX(ii, nx1, x1min, x1max);
            const Real x2v = CellCenterX(jj, nx2, x2min, x2max);
            const Real x3v = CellCenterX(kk, nx3, x3min, x3max);
            Real w = keep_mode ? KeepSampleWeight(src, x1v, x2v, x3v,
                                                  opts.use_transition_band,
                                                  opts.b_taper_root_cells)
                               : 1.0;
            if (keep_mode && opts.skip_cell && opts.skip_cell(x1v, x2v, x3v)) w = 0.0;
            u0_h(m, IEN, k, j, i) += 0.5 * (bx*bx + by*by + bz*bz) -
                                     (1.0 - w) * emag_old;
          }
          if (keep_mode) {
            const Real divb = (x1f_h(m, k, j, i + 1) - x1f_h(m, k, j, i)) / dx1 +
                              (x2f_h(m, k, j + 1, i) - x2f_h(m, k, j, i)) / dx2 +
                              (x3f_h(m, k + 1, j, i) - x3f_h(m, k, j, i)) / dx3;
            seam_divb = std::max(seam_divb, std::abs(divb) * hmin);
            seam_bmax = std::max(seam_bmax, std::sqrt(bx*bx + by*by + bz*bz));
          }
        }
      }
    }
  }

  Kokkos::deep_copy(b0.x1f, x1f_h);
  Kokkos::deep_copy(b0.x2f, x2f_h);
  Kokkos::deep_copy(b0.x3f, x3f_h);
  Kokkos::deep_copy(bcc0, bcc_h);
  if (add_emag) {
    Kokkos::deep_copy(u0, u0_h);
  }

  if (keep_mode) {
#if MPI_PARALLEL_ENABLED
    Real seam[2] = {seam_divb, seam_bmax};
    MPI_Allreduce(MPI_IN_PLACE, seam, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    seam_divb = seam[0];
    seam_bmax = seam[1];
#endif
    // 1e-10 relative: the curl of the interpolated potential is divergence-free to
    // roundoff, so anything above that is the seam, not arithmetic.  A same-domain remap
    // (the production case) has no seam at all and never prints.
    if (global_variable::my_rank == 0 &&
        seam_divb > 1.0e-10 * std::max(seam_bmax, 1.0e-300)) {
      std::cout << "WARNING (remap FC): keep mode left the target's own B standing "
                << "outside the source box plus taper, and the two divergence-free "
                << "fields do not match at that seam: max |div(B)|*h = " << seam_divb
                << " ("
                << (seam_divb / std::max(seam_bmax, 1.0e-300)) << " of max |B|), confined"
                << " to the one cell layer at the outer edge of the taper.  The "
                << "alternative -- zeroing B there -- leaves the gas holding magnetized "
                << "conserved variables with no field to decode them against."
                << std::endl << std::endl;
    }
  }
}

}  // namespace impl
}  // namespace remap
