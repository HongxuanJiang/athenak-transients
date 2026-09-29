#ifndef EOS_GRMHD_MAGNETIZATION_HPP_
#define EOS_GRMHD_MAGNETIZATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grmhd_magnetization.hpp
//! \brief Device-safe GRMHD four-velocity, comoving magnetic field, and sigma helpers.

#include <math.h>

#include "athena.hpp"

namespace grmhd {

struct MagneticFieldState {
  Real alpha = 0.0;
  Real lorentz_factor = 0.0;
  Real ucon[4] = {};
  Real ucov[4] = {};
  Real bcon[4] = {};
  Real bcov[4] = {};
  Real bsq = 0.0;
  bool valid = false;
};

// Construct u^mu and b^mu from AthenaK's normal-frame u-tilde^i and B^i=*F^{it}.
// The explicit statement order matches the fixed-GR sigma-ceiling implementation that
// preceded this shared helper.
KOKKOS_INLINE_FUNCTION
MagneticFieldState ComputeMagneticFieldState(const Real glower[][4],
                                             const Real gupper[][4],
                                             const Real ux, const Real uy,
                                             const Real uz, const Real bx,
                                             const Real by, const Real bz) {
  MagneticFieldState state;

  Real q = glower[1][1]*SQR(ux) + 2.0*glower[1][2]*ux*uy
         + 2.0*glower[1][3]*ux*uz + glower[2][2]*SQR(uy)
         + 2.0*glower[2][3]*uy*uz + glower[3][3]*SQR(uz);
  if (!isfinite(q)) return state;

  state.alpha = sqrt(-1.0/gupper[0][0]);
  state.lorentz_factor = sqrt(1.0 + q);
  state.ucon[0] = state.lorentz_factor/state.alpha;
  state.ucon[1] = ux - state.alpha*state.lorentz_factor*gupper[0][1];
  state.ucon[2] = uy - state.alpha*state.lorentz_factor*gupper[0][2];
  state.ucon[3] = uz - state.alpha*state.lorentz_factor*gupper[0][3];

  state.ucov[1] = glower[1][0]*state.ucon[0] + glower[1][1]*state.ucon[1]
                + glower[1][2]*state.ucon[2] + glower[1][3]*state.ucon[3];
  state.ucov[2] = glower[2][0]*state.ucon[0] + glower[2][1]*state.ucon[1]
                + glower[2][2]*state.ucon[2] + glower[2][3]*state.ucon[3];
  state.ucov[3] = glower[3][0]*state.ucon[0] + glower[3][1]*state.ucon[1]
                + glower[3][2]*state.ucon[2] + glower[3][3]*state.ucon[3];

  state.bcon[0] = state.ucov[1]*bx + state.ucov[2]*by + state.ucov[3]*bz;
  state.bcon[1] = (bx + state.bcon[0]*state.ucon[1])/state.ucon[0];
  state.bcon[2] = (by + state.bcon[0]*state.ucon[2])/state.ucon[0];
  state.bcon[3] = (bz + state.bcon[0]*state.ucon[3])/state.ucon[0];

  state.bcov[0] = glower[0][0]*state.bcon[0] + glower[0][1]*state.bcon[1]
                + glower[0][2]*state.bcon[2] + glower[0][3]*state.bcon[3];
  state.bcov[1] = glower[1][0]*state.bcon[0] + glower[1][1]*state.bcon[1]
                + glower[1][2]*state.bcon[2] + glower[1][3]*state.bcon[3];
  state.bcov[2] = glower[2][0]*state.bcon[0] + glower[2][1]*state.bcon[1]
                + glower[2][2]*state.bcon[2] + glower[2][3]*state.bcon[3];
  state.bcov[3] = glower[3][0]*state.bcon[0] + glower[3][1]*state.bcon[1]
                + glower[3][2]*state.bcon[2] + glower[3][3]*state.bcon[3];
  state.bsq = state.bcon[0]*state.bcov[0] + state.bcon[1]*state.bcov[1]
            + state.bcon[2]*state.bcov[2] + state.bcon[3]*state.bcov[3];
  if (!isfinite(state.bsq)) return state;

  state.valid = true;
  return state;
}

KOKKOS_INLINE_FUNCTION
Real MagnetizationFromState(const MagneticFieldState &state, const Real rho) {
  if (!state.valid || !isfinite(rho) || !(rho > 0.0) || state.bsq < 0.0) return 0.0;
  const Real sigma = state.bsq/rho;
  return (isfinite(sigma) && sigma >= 0.0) ? sigma : 0.0;
}

// Algebraically equivalent 3+1 form used when only lapse and the spatial metric are
// available (including DynGRMHD).  This preserves the pre-existing two-temperature
// statement order and invalid-state fallback.
KOKKOS_INLINE_FUNCTION
Real ComovingMagneticFieldSquared3p1(const Real alpha, const Real gxx,
                                    const Real gxy, const Real gxz,
                                    const Real gyy, const Real gyz,
                                    const Real gzz, const Real ux,
                                    const Real uy, const Real uz,
                                    const Real bx, const Real by,
                                    const Real bz) {
  const Real usq = gxx*SQR(ux) + 2.0*gxy*ux*uy + 2.0*gxz*ux*uz
                 + gyy*SQR(uy) + 2.0*gyz*uy*uz + gzz*SQR(uz);
  const Real bsq_eulerian = gxx*SQR(bx) + 2.0*gxy*bx*by + 2.0*gxz*bx*bz
                           + gyy*SQR(by) + 2.0*gyz*by*bz + gzz*SQR(bz);
  const Real budot = bx*(gxx*ux + gxy*uy + gxz*uz)
                   + by*(gxy*ux + gyy*uy + gyz*uz)
                   + bz*(gxz*ux + gyz*uy + gzz*uz);
  const Real lorentz_sq = 1.0 + usq;
  Real bsq = (alpha > 0.0 && isfinite(alpha) && lorentz_sq > 0.0) ?
      (SQR(alpha)*(bsq_eulerian + SQR(budot))/lorentz_sq) : 0.0;
  if (!isfinite(bsq) || bsq < 0.0) bsq = 0.0;
  return bsq;
}

KOKKOS_INLINE_FUNCTION
Real MagnetizationFrom3p1(const Real alpha, const Real gxx, const Real gxy,
                          const Real gxz, const Real gyy, const Real gyz,
                          const Real gzz, const Real ux, const Real uy,
                          const Real uz, const Real bx, const Real by,
                          const Real bz, const Real rho) {
  if (!isfinite(rho) || !(rho > 0.0)) return 0.0;
  const Real bsq = ComovingMagneticFieldSquared3p1(
      alpha, gxx, gxy, gxz, gyy, gyz, gzz, ux, uy, uz, bx, by, bz);
  const Real sigma = bsq/rho;
  return (isfinite(sigma) && sigma >= 0.0) ? sigma : 0.0;
}

}  // namespace grmhd

#endif  // EOS_GRMHD_MAGNETIZATION_HPP_
