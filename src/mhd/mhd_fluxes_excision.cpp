//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_excision.cpp
//! \brief MHD::ApplyExcisionSinkBoundary() and the sink-flux helpers it alone uses.
//!
//! Split out of mhd_fluxes.cpp (see mhd_fluxes_impl.hpp for the rationale of that split):
//! unlike MHD::CalculateFluxes(), this function is NOT templated on the Riemann solver,
//! so it must not be defined in a header included by all nine solver translation units
//! -- that would either violate ODR (as a plain non-inline member function) or, if made
//! inline, needlessly recompile and re-run ptxas on its three sink-flux kernels in every
//! one of them.  It depends on nothing else in mhd_fluxes_impl.hpp (confirmed: it uses
//! only the file-local helpers below, FinalizeExcisionSinkGRMHDState, and symbols from
//! the headers included here), so it gets its own translation unit instead, compiled
//! exactly once.

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/grmhd_face_floors.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/thermal_floors.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"
#include "mhd/rsolvers/hlld_grmhd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
void ZeroSinkFaceFluxX1(const bool dual_enabled, const int nvars,
                        BandView5D<Real> flx, BandView4D<Real> e31,
                        BandView4D<Real> e21, BandView5D<Real> vf,
                        const int m, const int k, const int j, const int i) {
  for (int n = 0; n < nvars; ++n) {
    flx(m, n, k, j, i) = 0.0;
  }
  e31(m, k, j, i) = 0.0;
  e21(m, k, j, i) = 0.0;
  if (dual_enabled) {
    vf(m, 0, k, j, i) = 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
void ZeroSinkFaceFluxX2(const bool dual_enabled, const int nvars,
                        BandView5D<Real> flx, BandView4D<Real> e12,
                        BandView4D<Real> e32, BandView5D<Real> vf,
                        const int m, const int k, const int j, const int i) {
  for (int n = 0; n < nvars; ++n) {
    flx(m, n, k, j, i) = 0.0;
  }
  e12(m, k, j, i) = 0.0;
  e32(m, k, j, i) = 0.0;
  if (dual_enabled) {
    vf(m, 0, k, j, i) = 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
void ZeroSinkFaceFluxX3(const bool dual_enabled, const int nvars,
                        BandView5D<Real> flx, BandView4D<Real> e13,
                        BandView4D<Real> e23, BandView5D<Real> vf,
                        const int m, const int k, const int j, const int i) {
  for (int n = 0; n < nvars; ++n) {
    flx(m, n, k, j, i) = 0.0;
  }
  e13(m, k, j, i) = 0.0;
  e23(m, k, j, i) = 0.0;
  if (dual_enabled) {
    vf(m, 0, k, j, i) = 0.0;
  }
}

inline bool BlockMayIntersectExcisionSphere(const RegionSize &size, const Real x,
                                            const Real y, const Real z,
                                            const Real r2) {
  const Real xmin = size.x1min - 0.5*size.dx1;
  const Real xmax = size.x1max + 0.5*size.dx1;
  const Real ymin = size.x2min - 0.5*size.dx2;
  const Real ymax = size.x2max + 0.5*size.dx2;
  const Real zmin = size.x3min - 0.5*size.dx3;
  const Real zmax = size.x3max + 0.5*size.dx3;

  Real dx = 0.0;
  if (x < xmin) {
    dx = xmin - x;
  } else if (x > xmax) {
    dx = x - xmax;
  }

  Real dy = 0.0;
  if (y < ymin) {
    dy = ymin - y;
  } else if (y > ymax) {
    dy = y - ymax;
  }

  Real dz = 0.0;
  if (z < zmin) {
    dz = zmin - z;
  } else if (z > zmax) {
    dz = z - zmax;
  }

  return (SQR(dx) + SQR(dy) + SQR(dz)) <= r2;
}

}  // namespace

namespace mhd {

void MHD::ApplyExcisionSinkBoundary(const Real t) {
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(t, excise_enabled, excise_radius, excise_density,
      excise_eint, sink_x, sink_y, sink_z);
  if (!excise_enabled) return;

  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  const int is = indcs_.is;
  const int ie = indcs_.ie;
  const int js = indcs_.js;
  const int je = indcs_.je;
  const int ks = indcs_.ks;
  const int ke = indcs_.ke;
  const int nvars_ = nvars;
  const int nmhd_ = nmhd;
  const bool dual_enabled = use_dual_energy;
  const int dual_idx = dual_energy_idx;
  const Real excise_r2 = excise_radius * excise_radius;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto &b0_ = b0;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &eos = peos->eos_data;
  const bool is_sr = pmy_pack->pcoord->is_special_relativistic;
  const bool is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto flx1_ = FluxBand(uflx.x1f);
  auto flx2_ = FluxBand(uflx.x2f);
  auto flx3_ = FluxBand(uflx.x3f);
  auto e31_ = EmfBand(e3x1);
  auto e21_ = EmfBand(e2x1);
  auto e12_ = EmfBand(e1x2);
  auto e32_ = EmfBand(e3x2);
  auto e23_ = EmfBand(e2x3);
  auto e13_ = EmfBand(e1x3);
  auto vf1_ = FluxBand(dual_vf.x1f);
  auto vf2_ = FluxBand(dual_vf.x2f);
  auto vf3_ = FluxBand(dual_vf.x3f);
  const int nscan = pmy_pack->nmb_thispack;

  if (sink_block_indices.extent_int(0) < std::max(1, pmy_pack->nmb_thispack)) {
    // AMR may replace this allocation after asynchronous sink kernels have used it.
    DevExeSpace().fence();
    sink_block_indices = DualArray1D<int>("mhd_sink_blocks",
                                          std::max(1, pmy_pack->nmb_thispack));
  }
  int nsink = 0;
  for (int a = 0; a < nscan; ++a) {
    const int m = a;
    if (BlockMayIntersectExcisionSphere(size_.h_view(m), sink_x, sink_y, sink_z,
                                        excise_r2)) {
      sink_block_indices.h_view(nsink++) = m;
    }
  }
  if (nsink == 0) return;

  auto &sink_blocks = sink_block_indices;
  sink_blocks.template modify<HostMemSpace>();
  sink_blocks.template sync<DevExeSpace>();
  const int nsink_blocks = nsink - 1;

  par_for("mhd_sink_flux_x1", DevExeSpace(), 0, nsink_blocks, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int n, int k, int j, int i) {
    const int m = sink_blocks.d_view(n);
    const Real y = CellCenterX(j - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                               size_.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                               size_.d_view(m).x3max);
    const Real xl = CellCenterX(i - 1 - indcs_.is, indcs_.nx1, size_.d_view(m).x1min,
                                size_.d_view(m).x1max);
    const Real xr = CellCenterX(i - indcs_.is, indcs_.nx1, size_.d_view(m).x1min,
                                size_.d_view(m).x1max);
    const bool left_inside =
        problem_runtime::InsideExcisionZone(xl, y, z, sink_x, sink_y, sink_z, excise_r2);
    const bool right_inside =
        problem_runtime::InsideExcisionZone(xr, y, z, sink_x, sink_y, sink_z, excise_r2);
    if (!(left_inside || right_inside)) return;
    if (left_inside && right_inside) {
      ZeroSinkFaceFluxX1(dual_enabled, nvars_, flx1_, e31_, e21_, vf1_, m, k, j, i);
      return;
    }

    const int ilive = left_inside ? i : (i - 1);
    MHDPrim1D wlive;
    wlive.d = w0_(m, IDN, k, j, ilive);
    wlive.vx = w0_(m, IVX, k, j, ilive);
    wlive.vy = w0_(m, IVY, k, j, ilive);
    wlive.vz = w0_(m, IVZ, k, j, ilive);
    if (eos.use_e) wlive.e = w0_(m, IEN, k, j, ilive);
    wlive.by = bcc0_(m, IBY, k, j, ilive);
    wlive.bz = bcc0_(m, IBZ, k, j, ilive);

    const Real x1v = LeftEdgeX(i - indcs_.is, indcs_.nx1, size_.d_view(m).x1min,
                               size_.d_view(m).x1max);

    const Real bxi = b0_.x1f(m, k, j, i);
    MHDCons1D flux;
    if (is_gr) {
      // The cell-centered state is capped at its own metric location.  Reapply the
      // common ceiling at the sink face before the single-state GRMHD flux; otherwise
      // the metric shift alone can put this face state above gamma_max when the full
      // hybrid face finalizer is disabled.
      LimitDirectionalGRMHDFaceStateLorentzAtCartesian(
          1, coord, x1v, y, z, eos.gamma_max, wlive);
      SingleStateLLF_GRMHD(wlive, wlive, bxi, x1v, y, z, IVX, coord, eos, flux);
    } else if (is_sr) {
      SingleStateLLF_SRMHD(wlive, wlive, bxi, eos, flux);
    } else {
      SingleStateLLF_MHD(wlive, wlive, bxi, eos, flux);
    }

    const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
    if (!allow_outflow) {
      ZeroSinkFaceFluxX1(dual_enabled, nvars_, flx1_, e31_, e21_, vf1_, m, k, j, i);
      return;
    }

    flx1_(m, IDN, k, j, i) = flux.d;
    flx1_(m, IM1, k, j, i) = flux.mx;
    flx1_(m, IM2, k, j, i) = flux.my;
    flx1_(m, IM3, k, j, i) = flux.mz;
    if (eos.use_e) flx1_(m, IEN, k, j, i) = flux.e;
    if (nvars_ > nmhd_) {
      for (int ivar = nmhd_; ivar < nvars_; ++ivar) {
        if (dual_enabled && ivar == dual_idx) continue;
        const Real scalar_value = w0_(m,ivar,k,j,ilive);
        flx1_(m, ivar, k, j, i) = flux.d * scalar_value;
      }
    }
    e31_(m, k, j, i) = flux.by;
    e21_(m, k, j, i) = flux.bz;
    if (dual_enabled) {
      flx1_(m, dual_idx, k, j, i) = w0_(m, dual_idx, k, j, ilive) * wlive.vx;
      vf1_(m, 0, k, j, i) = wlive.vx;
    }
  });

  if (pmy_pack->pmesh->multi_d) {
    par_for("mhd_sink_flux_x2", DevExeSpace(), 0, nsink_blocks, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int n, int k, int j, int i) {
      const int m = sink_blocks.d_view(n);
      const Real x = CellCenterX(i - indcs_.is, indcs_.nx1, size_.d_view(m).x1min,
                                 size_.d_view(m).x1max);
      const Real z = CellCenterX(k - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                                 size_.d_view(m).x3max);
      const Real yl = CellCenterX(j - 1 - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                                  size_.d_view(m).x2max);
      const Real yr = CellCenterX(j - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                                  size_.d_view(m).x2max);
      const bool left_inside =
          problem_runtime::InsideExcisionZone(x, yl, z, sink_x, sink_y, sink_z,
                                              excise_r2);
      const bool right_inside =
          problem_runtime::InsideExcisionZone(x, yr, z, sink_x, sink_y, sink_z,
                                              excise_r2);
      if (!(left_inside || right_inside)) return;
      if (left_inside && right_inside) {
        ZeroSinkFaceFluxX2(dual_enabled, nvars_, flx2_, e12_, e32_, vf2_, m, k, j, i);
        return;
      }

      const int jlive = left_inside ? j : (j - 1);
      MHDPrim1D wlive;
      wlive.d = w0_(m, IDN, k, jlive, i);
      wlive.vx = w0_(m, IVY, k, jlive, i);
      wlive.vy = w0_(m, IVZ, k, jlive, i);
      wlive.vz = w0_(m, IVX, k, jlive, i);
      if (eos.use_e) wlive.e = w0_(m, IEN, k, jlive, i);
      wlive.by = bcc0_(m, IBZ, k, jlive, i);
      wlive.bz = bcc0_(m, IBX, k, jlive, i);

      const Real x2v = LeftEdgeX(j - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                                 size_.d_view(m).x2max);

      const Real bxi = b0_.x2f(m, k, j, i);
      MHDCons1D flux;
      if (is_gr) {
        LimitDirectionalGRMHDFaceStateLorentzAtCartesian(
            2, coord, x, x2v, z, eos.gamma_max, wlive);
        SingleStateLLF_GRMHD(wlive, wlive, bxi, x, x2v, z, IVY, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRMHD(wlive, wlive, bxi, eos, flux);
      } else {
        SingleStateLLF_MHD(wlive, wlive, bxi, eos, flux);
      }

      const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
      if (!allow_outflow) {
        ZeroSinkFaceFluxX2(dual_enabled, nvars_, flx2_, e12_, e32_, vf2_, m, k, j, i);
        return;
      }

      flx2_(m, IDN, k, j, i) = flux.d;
      flx2_(m, IM2, k, j, i) = flux.mx;
      flx2_(m, IM3, k, j, i) = flux.my;
      flx2_(m, IM1, k, j, i) = flux.mz;
      if (eos.use_e) flx2_(m, IEN, k, j, i) = flux.e;
      if (nvars_ > nmhd_) {
        for (int ivar = nmhd_; ivar < nvars_; ++ivar) {
          if (dual_enabled && ivar == dual_idx) continue;
          const Real scalar_value = w0_(m,ivar,k,jlive,i);
          flx2_(m, ivar, k, j, i) = flux.d * scalar_value;
        }
      }
      e12_(m, k, j, i) = flux.by;
      e32_(m, k, j, i) = flux.bz;
      if (dual_enabled) {
        flx2_(m, dual_idx, k, j, i) =
            w0_(m, dual_idx, k, jlive, i) * wlive.vx;
        vf2_(m, 0, k, j, i) = wlive.vx;
      }
    });
  }

  if (pmy_pack->pmesh->three_d) {
    par_for("mhd_sink_flux_x3", DevExeSpace(), 0, nsink_blocks, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int n, int k, int j, int i) {
      const int m = sink_blocks.d_view(n);
      const Real x = CellCenterX(i - indcs_.is, indcs_.nx1, size_.d_view(m).x1min,
                                 size_.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                                 size_.d_view(m).x2max);
      const Real zl = CellCenterX(k - 1 - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                                  size_.d_view(m).x3max);
      const Real zr = CellCenterX(k - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                                  size_.d_view(m).x3max);
      const bool left_inside =
          problem_runtime::InsideExcisionZone(x, y, zl, sink_x, sink_y, sink_z,
                                              excise_r2);
      const bool right_inside =
          problem_runtime::InsideExcisionZone(x, y, zr, sink_x, sink_y, sink_z,
                                              excise_r2);
      if (!(left_inside || right_inside)) return;
      if (left_inside && right_inside) {
        ZeroSinkFaceFluxX3(dual_enabled, nvars_, flx3_, e13_, e23_, vf3_, m, k, j, i);
        return;
      }

      const int klive = left_inside ? k : (k - 1);
      MHDPrim1D wlive;
      wlive.d = w0_(m, IDN, klive, j, i);
      wlive.vx = w0_(m, IVZ, klive, j, i);
      wlive.vy = w0_(m, IVX, klive, j, i);
      wlive.vz = w0_(m, IVY, klive, j, i);
      if (eos.use_e) wlive.e = w0_(m, IEN, klive, j, i);
      wlive.by = bcc0_(m, IBX, klive, j, i);
      wlive.bz = bcc0_(m, IBY, klive, j, i);

      const Real x3v = LeftEdgeX(k - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                                 size_.d_view(m).x3max);

      const Real bxi = b0_.x3f(m, k, j, i);
      MHDCons1D flux;
      if (is_gr) {
        LimitDirectionalGRMHDFaceStateLorentzAtCartesian(
            3, coord, x, y, x3v, eos.gamma_max, wlive);
        SingleStateLLF_GRMHD(wlive, wlive, bxi, x, y, x3v, IVZ, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRMHD(wlive, wlive, bxi, eos, flux);
      } else {
        SingleStateLLF_MHD(wlive, wlive, bxi, eos, flux);
      }

      const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
      if (!allow_outflow) {
        ZeroSinkFaceFluxX3(dual_enabled, nvars_, flx3_, e13_, e23_, vf3_, m, k, j, i);
        return;
      }

      flx3_(m, IDN, k, j, i) = flux.d;
      flx3_(m, IM3, k, j, i) = flux.mx;
      flx3_(m, IM1, k, j, i) = flux.my;
      flx3_(m, IM2, k, j, i) = flux.mz;
      if (eos.use_e) flx3_(m, IEN, k, j, i) = flux.e;
      if (nvars_ > nmhd_) {
        for (int ivar = nmhd_; ivar < nvars_; ++ivar) {
          if (dual_enabled && ivar == dual_idx) continue;
          const Real scalar_value = w0_(m,ivar,klive,j,i);
          flx3_(m, ivar, k, j, i) = flux.d * scalar_value;
        }
      }
      e23_(m, k, j, i) = flux.by;
      e13_(m, k, j, i) = flux.bz;
      if (dual_enabled) {
        flx3_(m, dual_idx, k, j, i) =
            w0_(m, dual_idx, klive, j, i) * wlive.vx;
        vf3_(m, 0, k, j, i) = wlive.vx;
      }
    });
  }
}

} // namespace mhd
