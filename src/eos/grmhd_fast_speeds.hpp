#ifndef EOS_GRMHD_FAST_SPEEDS_HPP_
#define EOS_GRMHD_FAST_SPEEDS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grmhd_fast_speeds.hpp
//! \brief free-function forms of the fast-magnetosonic wavespeed kernels that live as
//! EOS_Data member functions in eos.hpp.
//!
//! The Riemann solvers need these speeds evaluated with a *side-local* adiabatic index
//! (variable-EOS / two-temperature runs), while EOS_Data carries 12 Kokkos Views plus
//! ~40 scalars (>600 B).  Copying the whole struct twice per face just to overwrite the
//! single `gamma` scalar is pure register/stack traffic in the GR Riemann kernels, so the
//! bodies below take gamma as an argument instead.  They are exact copies of
//! EOS_Data::IdealGRMHDFastSpeeds and EOS_Data::IdealSRMHDFastSpeeds -- keep them in sync
//! with eos.hpp if those ever change.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void GRMHDFastSpeedsGamma
//! \brief GENERAL RELATIVISTIC IDEAL GAS MHD: maximal fast magnetosonic wave speeds.
//! Inputs:
//!  - gamma: adiabatic index to use for this state
//!  - d: density in comoving frame
//!  - p: gas pressure
//!  - u0, u1: contravariant components of 4-velocity
//!  - b_sq: b_\mu b^\mu
//!  - g00, g01, g11: contravariant components of metric (-1, 0, 1 in SR)
//! Outputs:
//!  - l_p/l_m: most positive/negative wavespeed
//! Notes:
//!  - Follows same general procedure as vchar() in phys.c in Harm.
//!  - Variables are named as though 1 is normal direction.

KOKKOS_INLINE_FUNCTION
void GRMHDFastSpeedsGamma(const Real gamma, const Real d, const Real p, const Real u0,
                          const Real u1, const Real b_sq, const Real g00, const Real g01,
                          const Real g11, Real& l_p, Real& l_m) {
  // Calculate comoving fast magnetosonic speed
  Real w = d + gamma*p/(gamma - 1.0);
  Real cs_sq = gamma * p / w;
  Real va_sq = b_sq / (b_sq + w);
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Set fast magnetosonic speeds in appropriate coordinates
  Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real a1 = b / a;
  Real a0 = c / a;
  Real s = fmax(SQR(a1) - 4.0 * a0, 0.0);
  s = sqrt(s);
  l_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
  l_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
}

//----------------------------------------------------------------------------------------
//! \fn void SRMHDFastSpeedsGamma
//! \brief SPECIAL RELATIVISTIC IDEAL GAS MHD: maximal fast magnetosonic wave speeds.
//! Arguments as in the GR version, with lor the Lorentz factor and ux the normal
//! 4-velocity component.
//! Reference:
//!   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)

KOKKOS_INLINE_FUNCTION
void SRMHDFastSpeedsGamma(const Real gamma, const Real d, const Real p, const Real ux,
                          const Real lor, const Real b_sq, Real& l_p, Real& l_m) {
  // Calculate comoving fast magnetosonic speed
  Real w = d + gamma*p/(gamma - 1.0);
  Real cs_sq = gamma*p/w;                            // (DZB 73)
  Real va_sq = b_sq / (b_sq + w);                    // (DZB 73)
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;       // (DZB 72)

  Real v2 = 1.0 - 1.0/(lor*lor);
  auto const p1 = (ux/lor) * (1.0 - cms_sq);
  auto const tmp = sqrt(cms_sq * ((1.0-v2*cms_sq) - p1*(ux/lor))) / lor;
  auto const invden = 1.0/(1.0 - v2*cms_sq);

  l_p = (p1 + tmp) * invden;
  l_m = (p1 - tmp) * invden;
}

#endif // EOS_GRMHD_FAST_SPEEDS_HPP_
