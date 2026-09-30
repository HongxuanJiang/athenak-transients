//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes_excision.cpp
//! \brief Hydro::ApplyExcisionSinkBoundary(), the excision-sink face fluxes, with the
//! helpers only it reads.  It does not depend on the Riemann solver, so it is compiled
//! once, apart from the hydro_fluxes_<solver>.cpp units (see hydro_fluxes_impl.hpp).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/specific_energy_recon.hpp"
#include "reconstruct/thermal_floors.hpp"
#include "hydro/rsolvers/advect_hyd.hpp"
#include "hydro/rsolvers/llf_hyd.hpp"
#include "hydro/rsolvers/hlle_hyd.hpp"
#include "hydro/rsolvers/hllc_hyd.hpp"
#include "hydro/rsolvers/roe_hyd.hpp"
#include "hydro/rsolvers/llf_srhyd.hpp"
#include "hydro/rsolvers/hlle_srhyd.hpp"
#include "hydro/rsolvers/hllc_srhyd.hpp"
#include "hydro/rsolvers/llf_grhyd.hpp"
#include "hydro/rsolvers/hlle_grhyd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
void ZeroSinkFaceFlux(const bool dual_enabled, const int nvars,
                      DvceArray5D<Real> flx, DvceArray5D<Real> vf,
                      const int m, const int k, const int j, const int i) {
  for (int n = 0; n < nvars; ++n) {
    flx(m, n, k, j, i) = 0.0;
  }
  if (dual_enabled) {
    vf(m, 0, k, j, i) = 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
void ComputeSinkStateFlux(const HydPrim1D &w, const EOS_Data &eos,
                          HydCons1D &flux) {
  const Real mass_flux = w.d*w.vx;
  flux.d = mass_flux;
  flux.mx = mass_flux*w.vx;
  flux.my = mass_flux*w.vy;
  flux.mz = mass_flux*w.vz;

  if (eos.use_e) {
    const Real pressure = eos.PressureFromRhoEint(w.d, w.e);
    const Real etot = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
    flux.mx += pressure;
    flux.e = (etot + pressure)*w.vx;
  } else {
    flux.mx += SQR(eos.iso_cs)*w.d;
    flux.e = 0.0;
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

namespace hydro {

void Hydro::ApplyExcisionSinkBoundary(const Real t) {
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(t, excise_enabled, excise_radius,
      excise_density, excise_eint, sink_x, sink_y, sink_z);
  if (!excise_enabled) return;

  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  const int is = indcs_.is;
  const int ie = indcs_.ie;
  const int js = indcs_.js;
  const int je = indcs_.je;
  const int ks = indcs_.ks;
  const int ke = indcs_.ke;
  const int nvars_ = nvars;
  // Only the Newtonian p dV auxiliary has its own face-velocity array and flux here; a
  // GR auxiliary (D*kappa) is advected with the mass flux like a passive scalar.
  const bool dual_enabled = use_dual_energy && dual_energy_pdv;
  const int dual_idx = dual_energy_idx;
  const Real excise_r2 = excise_radius * excise_radius;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &w0_ = w0;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &eos = peos->eos_data;
  const bool is_sr = pmy_pack->pcoord->is_special_relativistic;
  const bool is_gr = pmy_pack->pcoord->is_general_relativistic;
  const int nhyd_ = nhydro;
  auto &flx1_ = uflx.x1f;
  auto &flx2_ = uflx.x2f;
  auto &flx3_ = uflx.x3f;
  auto &vf1_ = dual_vf.x1f;
  auto &vf2_ = dual_vf.x2f;
  auto &vf3_ = dual_vf.x3f;

  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &active_indices = pmy_pack->lat_active_indices;
  if (lat_enabled) active_indices.template sync<HostMemSpace>();
  const int nscan = lat_enabled ? pmy_pack->lat_nactive_thispack : pmy_pack->nmb_thispack;
  int nsink = 0;
  for (int a = 0; a < nscan; ++a) {
    const int m = lat_enabled ? active_indices.h_view(a) : a;
    if (BlockMayIntersectExcisionSphere(size_.h_view(m), sink_x, sink_y, sink_z,
                                        excise_r2)) {
      sink_block_indices.h_view(nsink++) = m;
    }
  }
  if (nsink == 0) return;

  // Stream-ordered upload: DualView::sync would fence the whole device queue here,
  // in the middle of every RK stage.
  Kokkos::deep_copy(DevExeSpace(), sink_block_indices.d_view, sink_block_indices.h_view);
  sink_block_indices.clear_sync_state();
  auto sink_blocks = sink_block_indices.d_view;
  const int nsink_blocks = nsink - 1;

  par_for("sink_flux_x1", DevExeSpace(), 0, nsink_blocks, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int n, int k, int j, int i) {
    const int m = sink_blocks(n);
    const Real y = CellCenterX(j - indcs_.js, indcs_.nx2, size_.d_view(m).x2min,
                               size_.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs_.ks, indcs_.nx3, size_.d_view(m).x3min,
                               size_.d_view(m).x3max);
    const Real xl = CellCenterX(i - 1 - indcs_.is, indcs_.nx1,
                                size_.d_view(m).x1min, size_.d_view(m).x1max);
    const Real xr = CellCenterX(i - indcs_.is, indcs_.nx1,
                                size_.d_view(m).x1min, size_.d_view(m).x1max);
    const bool left_inside = problem_runtime::InsideExcisionZone(
        xl, y, z, sink_x, sink_y, sink_z, excise_r2);
    const bool right_inside = problem_runtime::InsideExcisionZone(
        xr, y, z, sink_x, sink_y, sink_z, excise_r2);
    if (!(left_inside || right_inside)) return;
    if (left_inside && right_inside) {
      ZeroSinkFaceFlux(dual_enabled, nvars_, flx1_, vf1_, m, k, j, i);
      return;
    }

    const int ilive = left_inside ? i : (i - 1);
    HydPrim1D wlive;
    wlive.d = w0_(m, IDN, k, j, ilive);
    wlive.vx = w0_(m, IVX, k, j, ilive);
    wlive.vy = w0_(m, IVY, k, j, ilive);
    wlive.vz = w0_(m, IVZ, k, j, ilive);
    if (eos.use_e) wlive.e = w0_(m, IEN, k, j, ilive);

    HydCons1D flux;
    if (is_gr) {
      const Real x1v = LeftEdgeX(i - indcs_.is, indcs_.nx1,
                                 size_.d_view(m).x1min, size_.d_view(m).x1max);
      SingleStateLLF_GRHyd(wlive, wlive, x1v, y, z, IVX, coord, eos, flux);
    } else if (is_sr) {
      SingleStateLLF_SRHyd(wlive, wlive, eos, flux);
    } else {
      ComputeSinkStateFlux(wlive, eos, flux);
    }

    const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
    if (!allow_outflow) {
      ZeroSinkFaceFlux(dual_enabled, nvars_, flx1_, vf1_, m, k, j, i);
      return;
    }

    flx1_(m, IDN, k, j, i) = flux.d;
    flx1_(m, IM1, k, j, i) = flux.mx;
    flx1_(m, IM2, k, j, i) = flux.my;
    flx1_(m, IM3, k, j, i) = flux.mz;
    if (eos.use_e) flx1_(m, IEN, k, j, i) = flux.e;
    if (nvars_ > nhyd_) {
      for (int n = nhyd_; n < nvars_; ++n) {
        if (dual_enabled && n == dual_idx) continue;
        flx1_(m, n, k, j, i) = flux.d * w0_(m, n, k, j, ilive);
      }
    }
    if (dual_enabled) {
      flx1_(m, dual_idx, k, j, i) = w0_(m, dual_idx, k, j, ilive) * wlive.vx;
      vf1_(m, 0, k, j, i) = wlive.vx;
    }
  });

  if (pmy_pack->pmesh->multi_d) {
    par_for("sink_flux_x2", DevExeSpace(), 0, nsink_blocks, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int n, int k, int j, int i) {
      const int m = sink_blocks(n);
      const Real x = CellCenterX(i - indcs_.is, indcs_.nx1,
                                 size_.d_view(m).x1min, size_.d_view(m).x1max);
      const Real z = CellCenterX(k - indcs_.ks, indcs_.nx3,
                                 size_.d_view(m).x3min, size_.d_view(m).x3max);
      const Real yl = CellCenterX(j - 1 - indcs_.js, indcs_.nx2,
                                  size_.d_view(m).x2min, size_.d_view(m).x2max);
      const Real yr = CellCenterX(j - indcs_.js, indcs_.nx2,
                                  size_.d_view(m).x2min, size_.d_view(m).x2max);
      const bool left_inside = problem_runtime::InsideExcisionZone(
          x, yl, z, sink_x, sink_y, sink_z, excise_r2);
      const bool right_inside = problem_runtime::InsideExcisionZone(
          x, yr, z, sink_x, sink_y, sink_z, excise_r2);
      if (!(left_inside || right_inside)) return;
      if (left_inside && right_inside) {
        ZeroSinkFaceFlux(dual_enabled, nvars_, flx2_, vf2_, m, k, j, i);
        return;
      }

      const int jlive = left_inside ? j : (j - 1);
      HydPrim1D wlive;
      wlive.d = w0_(m, IDN, k, jlive, i);
      wlive.vx = w0_(m, IVY, k, jlive, i);
      wlive.vy = w0_(m, IVZ, k, jlive, i);
      wlive.vz = w0_(m, IVX, k, jlive, i);
      if (eos.use_e) wlive.e = w0_(m, IEN, k, jlive, i);

      HydCons1D flux;
      if (is_gr) {
        const Real x2v = LeftEdgeX(j - indcs_.js, indcs_.nx2,
                                   size_.d_view(m).x2min, size_.d_view(m).x2max);
        SingleStateLLF_GRHyd(wlive, wlive, x, x2v, z, IVY, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wlive, wlive, eos, flux);
      } else {
        ComputeSinkStateFlux(wlive, eos, flux);
      }

      const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
      if (!allow_outflow) {
        ZeroSinkFaceFlux(dual_enabled, nvars_, flx2_, vf2_, m, k, j, i);
        return;
      }

      flx2_(m, IDN, k, j, i) = flux.d;
      flx2_(m, IM2, k, j, i) = flux.mx;
      flx2_(m, IM3, k, j, i) = flux.my;
      flx2_(m, IM1, k, j, i) = flux.mz;
      if (eos.use_e) flx2_(m, IEN, k, j, i) = flux.e;
      if (nvars_ > nhyd_) {
        for (int n = nhyd_; n < nvars_; ++n) {
          if (dual_enabled && n == dual_idx) continue;
          flx2_(m, n, k, j, i) = flux.d * w0_(m, n, k, jlive, i);
        }
      }
      if (dual_enabled) {
        flx2_(m, dual_idx, k, j, i) = w0_(m, dual_idx, k, jlive, i) * wlive.vx;
        vf2_(m, 0, k, j, i) = wlive.vx;
      }
    });
  }

  if (pmy_pack->pmesh->three_d) {
    par_for("sink_flux_x3", DevExeSpace(), 0, nsink_blocks, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int n, int k, int j, int i) {
      const int m = sink_blocks(n);
      const Real x = CellCenterX(i - indcs_.is, indcs_.nx1,
                                 size_.d_view(m).x1min, size_.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs_.js, indcs_.nx2,
                                 size_.d_view(m).x2min, size_.d_view(m).x2max);
      const Real zl = CellCenterX(k - 1 - indcs_.ks, indcs_.nx3,
                                  size_.d_view(m).x3min, size_.d_view(m).x3max);
      const Real zr = CellCenterX(k - indcs_.ks, indcs_.nx3,
                                  size_.d_view(m).x3min, size_.d_view(m).x3max);
      const bool left_inside = problem_runtime::InsideExcisionZone(
          x, y, zl, sink_x, sink_y, sink_z, excise_r2);
      const bool right_inside = problem_runtime::InsideExcisionZone(
          x, y, zr, sink_x, sink_y, sink_z, excise_r2);
      if (!(left_inside || right_inside)) return;
      if (left_inside && right_inside) {
        ZeroSinkFaceFlux(dual_enabled, nvars_, flx3_, vf3_, m, k, j, i);
        return;
      }

      const int klive = left_inside ? k : (k - 1);
      HydPrim1D wlive;
      wlive.d = w0_(m, IDN, klive, j, i);
      wlive.vx = w0_(m, IVZ, klive, j, i);
      wlive.vy = w0_(m, IVX, klive, j, i);
      wlive.vz = w0_(m, IVY, klive, j, i);
      if (eos.use_e) wlive.e = w0_(m, IEN, klive, j, i);

      HydCons1D flux;
      if (is_gr) {
        const Real x3v = LeftEdgeX(k - indcs_.ks, indcs_.nx3,
                                   size_.d_view(m).x3min, size_.d_view(m).x3max);
        SingleStateLLF_GRHyd(wlive, wlive, x, y, x3v, IVZ, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wlive, wlive, eos, flux);
      } else {
        ComputeSinkStateFlux(wlive, eos, flux);
      }

      const bool allow_outflow = left_inside ? (flux.d < 0.0) : (flux.d > 0.0);
      if (!allow_outflow) {
        ZeroSinkFaceFlux(dual_enabled, nvars_, flx3_, vf3_, m, k, j, i);
        return;
      }

      flx3_(m, IDN, k, j, i) = flux.d;
      flx3_(m, IM3, k, j, i) = flux.mx;
      flx3_(m, IM1, k, j, i) = flux.my;
      flx3_(m, IM2, k, j, i) = flux.mz;
      if (eos.use_e) flx3_(m, IEN, k, j, i) = flux.e;
      if (nvars_ > nhyd_) {
        for (int n = nhyd_; n < nvars_; ++n) {
          if (dual_enabled && n == dual_idx) continue;
          flx3_(m, n, k, j, i) = flux.d * w0_(m, n, klive, j, i);
        }
      }
      if (dual_enabled) {
        flx3_(m, dual_idx, k, j, i) = w0_(m, dual_idx, klive, j, i) * wlive.vx;
        vf3_(m, 0, k, j, i) = wlive.vx;
      }
    });
  }
}

} // namespace hydro
