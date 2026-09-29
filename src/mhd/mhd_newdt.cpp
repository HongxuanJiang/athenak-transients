//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_newdt.cpp
//! \brief function to compute MHD timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "eos/grmhd_magnetization.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "mhd.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "srcterms/srcterms.hpp"

namespace mhd {

namespace {

// Compute the maximum coordinate fast-magnetosonic signal in each Cartesian
// direction for one fixed-spacetime GRMHD cell.  The state construction and
// pressure/gamma convention intentionally match HLLE_GR and the ordinary flux
// kernels.  Invalid or unrecoverable states use the conservative legacy signal
// (one) rather than allowing a NaN into a timestep reduction.
KOKKOS_INLINE_FUNCTION
void FixedGRMHDLocalFastSignalsWithGamma(
    const EOS_Data &eos, const Real gamma_gas,
    const Real rho, const Real eint,
    const Real vx, const Real vy, const Real vz,
    const Real bx, const Real by, const Real bz,
    const Real x1v, const Real x2v, const Real x3v,
    const bool minkowski, const Real spin,
    Real &max_dv1, Real &max_dv2, Real &max_dv3) {
  // A finite unit-speed fallback is both safe for invalid primitive/metric data
  // and exactly the disabled GR timestep's conservative signal.
  max_dv1 = 1.0;
  max_dv2 = 1.0;
  max_dv3 = 1.0;

  if (!isfinite(rho) || !(rho > 0.0) || !isfinite(eint) || !(eint >= 0.0) ||
      !isfinite(vx) || !isfinite(vy) || !isfinite(vz) ||
      !isfinite(bx) || !isfinite(by) || !isfinite(bz)) {
    return;
  }

  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v, x2v, x3v, minkowski, spin, glower, gupper);
  for (int n = 0; n < 4; ++n) {
    for (int q = 0; q < 4; ++q) {
      if (!isfinite(glower[n][q]) || !isfinite(gupper[n][q])) return;
    }
  }
  if (!(gupper[0][0] < 0.0)) return;

  const auto field = grmhd::ComputeMagneticFieldState(
      glower, gupper, vx, vy, vz, bx, by, bz);
  if (!field.valid || !isfinite(field.alpha) || !(field.alpha > 0.0) ||
      !isfinite(field.lorentz_factor) || !(field.lorentz_factor >= 1.0) ||
      !isfinite(field.ucon[0]) || !(field.ucon[0] > 0.0) ||
      !isfinite(field.ucon[1]) || !isfinite(field.ucon[2]) ||
      !isfinite(field.ucon[3]) || !isfinite(field.bcon[0]) ||
      !isfinite(field.bcon[1]) || !isfinite(field.bcon[2]) ||
      !isfinite(field.bcon[3]) || !isfinite(field.bsq) || field.bsq < 0.0) {
    return;
  }

  if (!isfinite(gamma_gas) || !(gamma_gas > 1.0)) {
    return;
  }
  const Real p = (gamma_gas - 1.0)*eint;
  if (!isfinite(p) || !(p >= 0.0)) return;

  EOS_Data local_eos = eos;
  local_eos.gamma = gamma_gas;
  Real lp = 0.0, lm = 0.0;
  local_eos.IdealGRMHDFastSpeeds(
      rho, p, field.ucon[0], field.ucon[1], field.bsq,
      gupper[0][0], gupper[0][1], gupper[1][1], lp, lm);
  max_dv1 = fmax(fabs(lp), fabs(lm));
  if (!isfinite(max_dv1) || !(max_dv1 > 0.0)) max_dv1 = 1.0;

  local_eos.IdealGRMHDFastSpeeds(
      rho, p, field.ucon[0], field.ucon[2], field.bsq,
      gupper[0][0], gupper[0][2], gupper[2][2], lp, lm);
  max_dv2 = fmax(fabs(lp), fabs(lm));
  if (!isfinite(max_dv2) || !(max_dv2 > 0.0)) max_dv2 = 1.0;

  local_eos.IdealGRMHDFastSpeeds(
      rho, p, field.ucon[0], field.ucon[3], field.bsq,
      gupper[0][0], gupper[0][3], gupper[3][3], lp, lm);
  max_dv3 = fmax(fabs(lp), fabs(lm));
  if (!isfinite(max_dv3) || !(max_dv3 > 0.0)) max_dv3 = 1.0;
}

}  // namespace

//----------------------------------------------------------------------------------------
// \!fn void MHD::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for MHD problems

TaskStatus MHD::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != (pdriver->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  // Every MeshBlock's timestep includes its ghost cells, with or without LAT (as in
  // Hydro::NewTimeStep); the pack timestep is the minimum over the MeshBlocks.
  Mesh *pmesh = pmy_pack->pmesh;
  auto &indcs = pmesh->mb_indcs;
  const int il = indcs.is - indcs.ng;
  const int iu = indcs.ie + indcs.ng;
  const int jl = pmesh->multi_d ? (indcs.js - indcs.ng) : indcs.js;
  const int ju = pmesh->multi_d ? (indcs.je + indcs.ng) : indcs.je;
  const int kl = pmesh->three_d ? (indcs.ks - indcs.ng) : indcs.ks;
  const int ku = pmesh->three_d ? (indcs.ke + indcs.ng) : indcs.ke;
  const int nx1 = iu - il + 1;
  const int nx2 = ju - jl + 1;
  const int nx3 = ku - kl + 1;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int mesh_nx1 = indcs.nx1;
  const int mesh_nx2 = indcs.nx2;
  const int mesh_nx3 = indcs.nx3;

  const int nmb = pmy_pack->nmb_thispack;
  const Real big_dt = std::numeric_limits<float>::max();
  dtnew = big_dt;

  // capture class variables for kernel
  auto &w0_ = w0;
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &is_special_relativistic_ = pmy_pack->pcoord->is_special_relativistic;
  auto &is_general_relativistic_ = pmy_pack->pcoord->is_general_relativistic;
  auto &is_dynamical_relativistic_ = pmy_pack->pcoord->is_dynamical_relativistic;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  const bool multi_d = pmesh->multi_d;
  const bool three_d = pmesh->three_d;

  if (pdriver->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("MHDNudt1", Kokkos::TeamPolicy<>(DevExeSpace(), nmb,
                            Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + il;
        k += kl;
        j += jl;

        Real cell_dt = mbsize.d_view(m).dx1/fabs(w0_(m,IVX,k,j,i));
        if (multi_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx2/
                                             fabs(w0_(m,IVY,k,j,i)));
        if (three_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx3/
                                             fabs(w0_(m,IVZ,k,j,i)));
        local_min = fmin(cell_dt, local_min);
      }, Kokkos::Min<Real>(mb_dt));
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  } else if (is_general_relativistic_ && gr_dt) {
    auto &bcc0_ = bcc0;
    const auto coord = pmy_pack->pcoord->coord_data;
    const bool use_excision = coord.bh_excise;
    auto excision_floor = pmy_pack->pcoord->excision_floor;

    Kokkos::parallel_reduce(
    "MHDNudtGRLocal", Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + il;
        k += kl;
        j += jl;
        if (use_excision && excision_floor(m,k,j,i)) return;

        const Real x1v = CellCenterX(
            i-is, mesh_nx1, mbsize.d_view(m).x1min,
            mbsize.d_view(m).x1max);
        const Real x2v = CellCenterX(
            j-js, mesh_nx2, mbsize.d_view(m).x2min,
            mbsize.d_view(m).x2max);
        const Real x3v = CellCenterX(
            k-ks, mesh_nx3, mbsize.d_view(m).x3min,
            mbsize.d_view(m).x3max);
        Real max_dv1, max_dv2, max_dv3;
        FixedGRMHDLocalFastSignalsWithGamma(
            eos, eos.gamma, w0_(m,IDN,k,j,i), w0_(m,IEN,k,j,i),
            w0_(m,IVX,k,j,i), w0_(m,IVY,k,j,i), w0_(m,IVZ,k,j,i),
            bcc0_(m,IBX,k,j,i), bcc0_(m,IBY,k,j,i), bcc0_(m,IBZ,k,j,i),
            x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
            max_dv1, max_dv2, max_dv3);
        Real cell_dt = mbsize.d_view(m).dx1/max_dv1;
        if (multi_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx2/max_dv2);
        if (three_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx3/max_dv3);
        local_min = fmin(cell_dt, local_min);
      }, Kokkos::Min<Real>(mb_dt));
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  } else if (is_general_relativistic_ || is_dynamical_relativistic_) {
    // GR/DynGR currently uses a unit maximum signal speed for the CFL estimate, so
    // the timestep depends only on meshblock spacing, not cell primitive values.
    Kokkos::parallel_reduce("MHDNudtGR", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
    KOKKOS_LAMBDA(const int &m, Real &pack_min) {
      Real mb_dt = mbsize.d_view(m).dx1;
      if (multi_d) mb_dt = fmin(mb_dt, mbsize.d_view(m).dx2);
      if (three_d) mb_dt = fmin(mb_dt, mbsize.d_view(m).dx3);
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  } else {
    // find smallest dx/(v +/- Cf) in each direction for mhd problems
    auto &bcc0_ = bcc0;

    Kokkos::parallel_reduce("MHDNudt2", Kokkos::TeamPolicy<>(DevExeSpace(), nmb,
                            Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + il;
        k += kl;
        j += jl;
        Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

        // timestep in SR MHD
        if (is_special_relativistic_) {
          Real &wd = w0_(m,IDN,k,j,i);
          Real &ux = w0_(m,IVX,k,j,i);
          Real &uy = w0_(m,IVY,k,j,i);
          Real &uz = w0_(m,IVZ,k,j,i);
          Real &bcc1 = bcc0_(m,IBX,k,j,i);
          Real &bcc2 = bcc0_(m,IBY,k,j,i);
          Real &bcc3 = bcc0_(m,IBZ,k,j,i);

          Real v2 = SQR(ux) + SQR(uy) + SQR(uz);
          Real lor = sqrt(1.0 + v2);
          // FIXME ERM: Ideal fluid for now
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
          // Calculate 4-magnetic field in left state
          Real b_0 = bcc1*ux + bcc2*uy + bcc3*uz;
          Real b_1 = (bcc1 + b_0 * ux) / lor;
          Real b_2 = (bcc2 + b_0 * uy) / lor;
          Real b_3 = (bcc3 + b_0 * uz) / lor;
          Real b_sq = -SQR(b_0) + SQR(b_1) + SQR(b_2) + SQR(b_3);

          Real lm, lp;
          eos.IdealSRMHDFastSpeeds(wd, p, ux, lor, b_sq, lp, lm);
          max_dv1 = fmax(fabs(lm), lp);

          eos.IdealSRMHDFastSpeeds(wd, p, uy, lor, b_sq, lp, lm);
          max_dv2 = fmax(fabs(lm), lp);

          eos.IdealSRMHDFastSpeeds(wd, p, uz, lor, b_sq, lp, lm);
          max_dv3 = fmax(fabs(lm), lp);
        // timestep in Newtonian MHD
        } else {
          Real &w_d = w0_(m,IDN,k,j,i);
          Real &w_bx = bcc0_(m,IBX,k,j,i);
          Real &w_by = bcc0_(m,IBY,k,j,i);
          Real &w_bz = bcc0_(m,IBZ,k,j,i);
          Real cf;
          if (eos.is_ideal) {
            Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
            cf = eos.IdealMHDFastSpeed(w_d, p, w_bx, w_by, w_bz);
            max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cf;
            cf = eos.IdealMHDFastSpeed(w_d, p, w_by, w_bz, w_bx);
            max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cf;
            cf = eos.IdealMHDFastSpeed(w_d, p, w_bz, w_bx, w_by);
            max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cf;
          } else {
            cf = eos.IdealMHDFastSpeed(w_d, w_bx, w_by, w_bz);
            max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cf;
            cf = eos.IdealMHDFastSpeed(w_d, w_by, w_bz, w_bx);
            max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cf;
            cf = eos.IdealMHDFastSpeed(w_d, w_bz, w_bx, w_by);
            max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cf;
          }
        }

        Real cell_dt = mbsize.d_view(m).dx1/max_dv1;
        if (multi_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx2/max_dv2);
        if (three_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx3/max_dv3);
        local_min = fmin(cell_dt, local_min);
      }, Kokkos::Min<Real>(mb_dt));
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  // compute timestep for diffusion
  if (pcond != nullptr) {
    pcond->NewTimeStep(w0, peos->eos_data);
  }
  if (pvisc != nullptr) {
    pvisc->NewTimeStep(w0, peos->eos_data);
  }
  if (presist != nullptr) {
    presist->NewTimeStep(w0, peos->eos_data);
  }
  // compute source terms timestep
  if (psrc != nullptr) {
    psrc->NewTimeStep(w0, peos->eos_data);
  }

  return TaskStatus::complete;
}
} // namespace mhd
