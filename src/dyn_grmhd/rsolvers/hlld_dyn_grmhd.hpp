#ifndef DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_dyn_grmhd.hpp
//! \brief Five-wave HLLD solver for dynamical-spacetime GRMHD.
//!
//! Relativistic HLLD is derived in Minkowski space, so the fan has to be solved in a
//! locally flat frame.  Here that frame is built from the face's ADM variables: the
//! timelike leg is the normal observer n^mu = (1/alpha, -beta^i/alpha), and the spatial
//! triad is a Gram-Schmidt orthonormalisation of (dx^normal, dx^t1, dx^t2) in the inverse
//! three-metric, so that its first leg is along the face normal.  The interface then
//! moves with speed beta^x/(alpha sqrt(gamma^xx)) in that frame, exactly as in Fields,
//! Wong & Stone (2026, arXiv:2609.06150) Sec. II, and the fan itself is the shared solve
//! of mhd/rsolvers/hlld_local.hpp.
//!
//! Nothing here assumes an ideal gas.  The two states reach the fan as (rho, p, rho h),
//! all three taken from the PrimitiveSolver EOS policy, and the extremal speeds come from
//! the policy's own magnetosonic-speed routine evaluated in the flat frame.  A state the
//! fan declines falls back to HLLE on the same reconstructed states, as in the paper.

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "mhd/rsolvers/hlld_local.hpp"
#include "flux_dyn_grmhd.hpp"
#include "hlle_dyn_grmhd.hpp"

#include "eos/drift_frame_floor.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn bool BuildADMTriad
//! \brief Orthonormal spatial triad of gamma_ij whose first leg is the face normal.
//! `omega[a][i]` are the one-form components, so a vector's frame components are vhat^a =
//! omega[a][i] v^i, and `e[a][i] = gamma^{ij} omega[a][j]` are the triad vectors.  Built
//! by Gram-Schmidt on (dx^n, dx^t1, dx^t2) in the inverse metric, which makes omega[0]
//! proportional to dx^n and hence e[1] and e[2] free of any normal component -- the
//! property the flux transform below relies on.  `normal` is 0, 1 or 2.

KOKKOS_INLINE_FUNCTION
bool BuildADMTriad(const Real g3u[NSPMETRIC], const int normal,
                   Real omega[3][3], Real e[3][3]) {
  constexpr int kSym[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};
  const int idx[3] = {normal, (normal + 1)%3, (normal + 2)%3};

  for (int a = 0; a < 3; ++a) {
    // Start from the coordinate one-form dx^{idx[a]} and remove the earlier legs.
    for (int i = 0; i < 3; ++i) omega[a][i] = (i == idx[a]) ? 1.0 : 0.0;
    for (int b = 0; b < a; ++b) {
      Real dot = 0.0;
      for (int i = 0; i < 3; ++i) dot += e[b][i]*omega[a][i];
      for (int i = 0; i < 3; ++i) omega[a][i] -= dot*omega[b][i];
    }
    Real norm2 = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) norm2 += g3u[kSym[i][j]]*omega[a][i]*omega[a][j];
    }
    if (!(norm2 > 0.0) || !isfinite(norm2)) return false;
    const Real inorm = 1.0/sqrt(norm2);
    for (int i = 0; i < 3; ++i) omega[a][i] *= inorm;
    for (int i = 0; i < 3; ++i) {
      e[a][i] = 0.0;
      for (int j = 0; j < 3; ++j) e[a][i] += g3u[kSym[i][j]]*omega[a][j];
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void HLLD_DYNGR
//! \brief Five-wave HLLD flux for dynamical GRMHD, one face.

template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLD_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int mbuf, const int k, const int j, const int i,
     const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
     const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
     const DvceArray4D<Real> &bx,
     const int nhyd, const int nscal,
     const adm::ADMMetricView& metric,
     const BandView5D<Real> &flx, const BandView4D<Real> &ey,
     const BandView4D<Real> &ez) {
  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;
  constexpr int kSym[3][3] = {{S11, S12, S13}, {S12, S22, S23}, {S13, S23, S33}};

  auto fallback = [&]() {
    HLLE_DYNGR<ivx>(eos, indcs, size, coord, m, mbuf, k, j, i,
                    wl, wr, bl, br, bx, nhyd, nscal, metric, flx, ey, ez);
  };

  // Every way the fan can decline ends in the same HLLE call on the same reconstructed
  // states, so the solve is one block with a single exit rather than four returns: each
  // call site inlines the whole dynamical-GR HLLE solver, and that -- not the fan -- is
  // what this kernel's frame was made of.
  const bool solved = [&]() -> bool {
    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (ivx == IVX) {
      metric.template FaceMetric<1>(m, k, j, i, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      metric.template FaceMetric<2>(m, k, j, i, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      metric.template FaceMetric<3>(m, k, j, i, g3d, beta_u, alpha);
    }
    const Real detg = Primitive::GetDeterminant(g3d);
    if (!(detg > 0.0) || !(alpha > 0.0)) {
      return false;
    }
    const Real sdetg = sqrt(detg);
    const Real isdetg = 1.0/sdetg;

    Real g3u[NSPMETRIC];
    Primitive::InvertMatrix(g3u, g3d, detg);
    Real omega[3][3], triad[3][3];
    if (!BuildADMTriad(g3u, ivx - IVX, omega, triad)) {
      return false;
    }

    // Extract left and right primitives, as HLLE_DYNGR does.
    Real prim_l[NPRIM]{}, prim_r[NPRIM]{};
    Real Bu_l[NMAG], Bu_r[NMAG];
    const Real mb = eos.ps.GetEOS().GetBaryonMass();

    prim_l[PRH] = wl(mbuf, IDN, k, j, i)/mb;
    prim_l[PVX] = wl(mbuf, IVX, k, j, i);
    prim_l[PVY] = wl(mbuf, IVY, k, j, i);
    prim_l[PVZ] = wl(mbuf, IVZ, k, j, i);
    for (int n = 0; n < nscal; n++) prim_l[PYF + n] = wl(mbuf, nhyd + n, k, j, i);
    const auto &eos_l = eos;
    eos_l.ps.GetEOS().ApplyDensityLimits(prim_l[PRH]);
    eos_l.ps.GetEOS().ApplySpeciesLimits(&prim_l[PYF]);
    prim_l[PPR] = wl(mbuf, IPR, k, j, i);
    prim_l[PTM] = eos_l.ps.GetEOS().GetTemperatureFromP(
                  prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
    Bu_l[ibx] = bx(m, k, j, i)*isdetg;
    Bu_l[iby] = bl(mbuf, iby, k, j, i)*isdetg;
    Bu_l[ibz] = bl(mbuf, ibz, k, j, i)*isdetg;

    prim_r[PRH] = wr(mbuf, IDN, k, j, i)/mb;
    prim_r[PVX] = wr(mbuf, IVX, k, j, i);
    prim_r[PVY] = wr(mbuf, IVY, k, j, i);
    prim_r[PVZ] = wr(mbuf, IVZ, k, j, i);
    for (int n = 0; n < nscal; n++) prim_r[PYF + n] = wr(mbuf, nhyd + n, k, j, i);
    const auto &eos_r = eos;
    eos_r.ps.GetEOS().ApplyDensityLimits(prim_r[PRH]);
    eos_r.ps.GetEOS().ApplySpeciesLimits(&prim_r[PYF]);
    prim_r[PPR] = wr(mbuf, IPR, k, j, i);
    prim_r[PTM] = eos_r.ps.GetEOS().GetTemperatureFromP(
                  prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
    Bu_r[ibx] = bx(m, k, j, i)*isdetg;
    Bu_r[iby] = br(mbuf, iby, k, j, i)*isdetg;
    Bu_r[ibz] = br(mbuf, ibz, k, j, i)*isdetg;

    eos_floor::DriftFrameApplyPrimitiveFloor(eos_l.ps.GetEOS(), prim_l, Bu_l, g3d);
    eos_floor::DriftFrameApplyPrimitiveFloor(eos_r.ps.GetEOS(), prim_r, Bu_r, g3d);

    // Rotate the four-velocity and the field into the triad.  Both are spatial vectors
    // relative to the normal observer, so this is the whole transformation: the Lorentz
    // factor W = -u.n is the same number in either basis.
    mhd::hlld::LocalState sl{}, sr{};
    Real uhat_l[3], uhat_r[3], bhat_l[3], bhat_r[3];
    for (int a = 0; a < 3; ++a) {
      uhat_l[a] = 0.0; uhat_r[a] = 0.0; bhat_l[a] = 0.0; bhat_r[a] = 0.0;
      for (int c = 0; c < 3; ++c) {
        uhat_l[a] += omega[a][c]*prim_l[PVX + c];
        uhat_r[a] += omega[a][c]*prim_r[PVX + c];
        bhat_l[a] += omega[a][c]*Bu_l[c];
        bhat_r[a] += omega[a][c]*Bu_r[c];
      }
    }
    // B^normal is continuous across the face, so the two sides give the same number up to
    // round-off; average them as the fixed-metric solver does.
    const Real bx_hat = 0.5*(bhat_l[0] + bhat_r[0]);

    sl.d = prim_l[PRH]*mb;
    sl.p = prim_l[PPR];
    sl.wgas = sl.d*eos_l.ps.GetEOS().GetEnthalpy(prim_l[PRH], prim_l[PTM], &prim_l[PYF]);
    sl.vx = uhat_l[0]; sl.vy = uhat_l[1]; sl.vz = uhat_l[2];
    sl.by = bhat_l[1]; sl.bz = bhat_l[2];
    sr.d = prim_r[PRH]*mb;
    sr.p = prim_r[PPR];
    sr.wgas = sr.d*eos_r.ps.GetEOS().GetEnthalpy(prim_r[PRH], prim_r[PTM], &prim_r[PYF]);
    sr.vx = uhat_r[0]; sr.vy = uhat_r[1]; sr.vz = uhat_r[2];
    sr.by = bhat_r[1]; sr.bz = bhat_r[2];

    // Extremal speeds in the flat frame: the policy's own magnetosonic routine, evaluated
    // with a Minkowski three-metric and the rotated velocities.
    Real flat_g3d[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    Real flat_beta[3] = {0.0, 0.0, 0.0};
    Real prim_hat_l[NPRIM], prim_hat_r[NPRIM];
    for (int n = 0; n < NPRIM; ++n) {
      prim_hat_l[n] = prim_l[n];
      prim_hat_r[n] = prim_r[n];
    }
    for (int a = 0; a < 3; ++a) {
      prim_hat_l[PVX + a] = uhat_l[a];
      prim_hat_r[PVX + a] = uhat_r[a];
    }
    Real u0_l, u0_r, b4_l[4], b4_r[4];
    const Real bsq_l = mhd::hlld::FourVectors(sl, bx_hat, u0_l, b4_l);
    const Real bsq_r = mhd::hlld::FourVectors(sr, bx_hat, u0_r, b4_r);
    Real lp_l, lm_l, lp_r, lm_r;
    eos_l.GetGRFastMagnetosonicSpeeds(lp_l, lm_l, prim_hat_l, bsq_l,
                                      flat_g3d, flat_beta, 1.0, 1.0, PVX);
    eos_r.GetGRFastMagnetosonicSpeeds(lp_r, lm_r, prim_hat_r, bsq_r,
                                      flat_g3d, flat_beta, 1.0, 1.0, PVX);

    // The interface sits at fixed coordinate x, so in this frame it moves with the
    // triad-projected shift, beta^x/(alpha sqrt(gamma^xx)).
    Real v_interface = 0.0;
    for (int c = 0; c < 3; ++c) v_interface += omega[0][c]*beta_u[c];
    v_interface /= alpha;

    MHDCons1D uhat{}, fhat{};
    if (!mhd::hlld::SolveLocalHLLDStates(sl, sr, bx_hat, fmin(lm_l, lm_r),
                                         fmax(lp_l, lp_r), v_interface, uhat, fhat)) {
      return false;
    }

    // Back to the coordinate basis.  M^mu_a is the tetrad: e_0 = n, e_a = the triad.
    // Only a = 0 and a = 1 have a component along the face normal, so the normal row of
    // any tensor needs just those two.
    const Real mn0 = -beta_u[ivx - IVX]/alpha;    // n^x
    const Real mn1 = triad[0][ivx - IVX];         // e_1^x = sqrt(gamma^xx)
    Real mglob[4][4] = {};
    mglob[0][0] = 1.0/alpha;
    for (int c = 0; c < 3; ++c) {
      mglob[c + 1][0] = -beta_u[c]/alpha;
      for (int a = 0; a < 3; ++a) mglob[c + 1][a + 1] = triad[a][c];
    }
    // Four-metric from the ADM data.
    Real beta_d[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < 3; ++c) {
      for (int d = 0; d < 3; ++d) beta_d[c] += g3d[kSym[c][d]]*beta_u[d];
    }
    Real glower[4][4];
    glower[0][0] = -alpha*alpha + beta_d[0]*beta_u[0] + beta_d[1]*beta_u[1]
                   + beta_d[2]*beta_u[2];
    for (int c = 0; c < 3; ++c) {
      glower[0][c + 1] = beta_d[c];
      glower[c + 1][0] = beta_d[c];
      for (int d = 0; d < 3; ++d) glower[c + 1][d + 1] = g3d[kSym[c][d]];
    }

    const Real jn = mn0*uhat.d + mn1*fhat.d;
    const Real t0[4] = {uhat.e, uhat.mx, uhat.my, uhat.mz};
    const Real tx[4] = {fhat.e, fhat.mx, fhat.my, fhat.mz};
    Real t_con[4];
    for (int nu = 0; nu < 4; ++nu) {
      Real row0 = 0.0, rowx = 0.0;
      for (int a = 0; a < 4; ++a) {
        row0 += mglob[nu][a]*t0[a];
        rowx += mglob[nu][a]*tx[a];
      }
      t_con[nu] = mn0*row0 + mn1*rowx;
    }
    Real t_mixed[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < 3; ++c) {
      for (int nu = 0; nu < 4; ++nu) t_mixed[c] += glower[c + 1][nu]*t_con[nu];
    }

    // The dual field tensor in the frame, and its two transverse-normal components.
    Real dual[4][4] = {};
    dual[1][0] =  bx_hat;   dual[0][1] = -bx_hat;
    dual[2][0] =  uhat.by;  dual[0][2] = -uhat.by;
    dual[3][0] =  uhat.bz;  dual[0][3] = -uhat.bz;
    dual[2][1] =  fhat.by;  dual[1][2] = -fhat.by;
    dual[3][1] =  fhat.bz;  dual[1][3] = -fhat.bz;
    const int jy = 1 + iby;
    const int jz = 1 + ibz;
    Real dual_yn = 0.0, dual_zn = 0.0;
    for (int a = 0; a < 4; ++a) {
      const Real mn_a = (a == 0) ? mn0 : ((a == 1) ? mn1 : 0.0);
      if (mn_a == 0.0) continue;
      for (int b = 0; b < 4; ++b) {
        dual_yn += mglob[jy][b]*mn_a*dual[b][a];
        dual_zn += mglob[jz][b]*mn_a*dual[b][a];
      }
    }

    const Real vol = sdetg*alpha;
    const Real flux_d = vol*jn;
    const Real flux_e = vol*((mn0*uhat.e + mn1*fhat.e) - jn);
    if (!isfinite(flux_d) || !isfinite(flux_e) || !isfinite(t_mixed[0]) ||
        !isfinite(t_mixed[1]) || !isfinite(t_mixed[2]) ||
        !isfinite(dual_yn) || !isfinite(dual_zn)) {
      return false;
    }
    flx(m, IDN, k, j, i) = flux_d;
    flx(m, IEN, k, j, i) = flux_e;
    flx(m, IVX, k, j, i) = vol*t_mixed[0];
    flx(m, IVY, k, j, i) = vol*t_mixed[1];
    flx(m, IVZ, k, j, i) = vol*t_mixed[2];
    ey(m, k, j, i) = -vol*dual_yn;
    ez(m, k, j, i) =  vol*dual_zn;
    return true;
  }();
  if (!solved) {
    fallback();
  }
}

}  // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
