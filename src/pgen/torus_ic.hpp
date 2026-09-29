#ifndef PGEN_TORUS_IC_HPP_
#define PGEN_TORUS_IC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file torus_ic.hpp
//! \brief The initial condition of an accretion disk around a Kerr black hole, shared by
//! the fixed-Kerr generator (fluids/gr_torus.cpp) and the binary generator (BBH.cpp).
//!
//! Three disk models live here, selected by torus_pgen::use_nt_disk and
//! torus_pgen::use_chakrabarti_torus:
//!
//!   * Fishbone & Moncrief (1976), ApJ 207, 962 -- a constant-l equilibrium torus;
//!   * Chakrabarti (1985), ApJ 288, 1 -- l = c lambda^n instead of constant l;
//!   * Novikov & Thorne (1973) -- a thin, radiatively efficient disk on circular
//!     geodesics, with the relativistic flux function of Page & Thorne (1974) setting
//!     the radial temperature profile and a Gaussian vertical profile.
//!
//! and two initial magnetic fields, selected by torus_pgen::potential_inclined and
//! torus_pgen::is_vertical_field: the density-tied poloidal loop of the SANE/MAD family,
//! and the inclined large-scale field of Zanni et al. (2007).
//!
//! The ambient medium the disk sits in follows Vourellis, Fendt, Qian & Noble (2019),
//! ApJ 882, 2.  Their corona is hot and in pressure equilibrium with the disk surface,
//! and it is replaced within a few orbits by their floor value, so what is built here is
//! that background itself:
//!
//!   rho_amb(r) = rho0 [ (r/r0)^-3 + (r/r0)^-1 ]
//!   p_amb(r)   = p0   [ (r/r0)^-4 + (r/r0)^-2 ]
//!
//! measured from the disk's own centre.  The steep term feeds the funnel, where an open
//! field is strongest; the shallow one is the paper's "higher floor values at large radii
//! to avoid too high magnetization", and it is what keeps sigma = b^2/rho falling outward
//! against a midplane field that only falls as R^-5/4.  The amplitude is the deck's:
//! rho0 = <problem>/ambient_rho0 in units of rho_max, and p0 = rho0 T0 with T0 = p/rho at
//! r0 (<problem>/ambient_temp0).  Nothing ties it to the disk surface -- the paper's hot
//! corona collapses within an orbit anyway -- so the amplitude is set by what the field
//! outside the disk needs: light enough not to load the open lines, heavy enough to keep
//! sigma and beta bounded there.  It is the initial condition and nothing else: the
//! run-time lower bounds are dfloor/pfloor/tfloor and the magnetization ceilings, which
//! is where the upstream generator leaves them.  The sigma ceiling loading a uniform
//! floor onto the open field lines is now the wanted behaviour -- it is the bound that
//! knows about the field, and the drift-frame injection means the mass it adds arrives
//! carrying the cell's perpendicular motion instead of stopping the cell dead.
//!
//! Everything here is a closed-form function of position and of the parameters in
//! torus_pgen, so it can be evaluated inside a Kokkos kernel and, equally, sampled away
//! from the requested point (which the one-cell erosion below relies on).
//!
//! Both generators normalize lengths to the mass of the hole the disk orbits, but the
//! binary generator's hole is not of unit mass, so every formula carries torus_pgen::M
//! explicitly.  With M = 1 those factors are exact no-ops in IEEE arithmetic.
//!
//! What is NOT here, and why: the second hole and its trajectory, the superposed binary
//! metric, the excision masks, the circumbinary disk (which replaces the analytic Kerr
//! metric by a phi-averaged binary metric tabulated on a radial grid, so it reconstructs
//! its own orbit, enthalpy and vector potential), and the horizon bookkeeping -- all of
//! those are properties of the binary, not of the disk, and stay in BBH.cpp.

#include <math.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace torus_ic {

//----------------------------------------------------------------------------------------
//! \struct torus_pgen
//! \brief Everything the closed-form initial condition needs, in one value type that can
//! be captured by a device kernel.

struct torus_pgen {
  Real M;                                     // mass of the black hole the disk orbits
  Real spin;                                  // its spin a, dimensional (|a| <= M)
  Real dexcise, pexcise;                      // excision parameters
  Real gamma_adi;                             // EOS parameters
  Real arad;                                  // radiation constant of the fixed-Kerr seed
  bool prograde;                              // disk orbits with (true) or against a
  // Disk model
  bool use_chakrabarti_torus;                 // Chakrabarti torus instead of FM
  bool use_nt_disk;                           // Novikov-Thorne thin disk instead of a torus
  // Torus parameters, given and derived
  Real r_edge, r_peak, rho_max;               // inner edge, pressure maximum, peak density
  Real l_peak;                                // specific angular momentum at r_peak
  Real c_param, n_param;                      // Chakrabarti l = c lambda^n
  Real log_h_edge, log_h_peak;                // log-enthalpy at the edge and the peak
  Real ptot_over_rho_peak, rho_peak;          // p/rho and the density scale at the peak
  Real r_outer_edge;                          // outermost radius with log_h > log_h_edge
  Real cb_l_edge, cb_u_t_edge;                // BBH circumbinary edge reference state
  // Novikov-Thorne thin disk
  Real nt_r_trunc, nt_h_over_r, nt_r_out;     // truncation radius, aspect ratio, outer edge
  Real nt_inner_taper_width, nt_outer_taper_width;
  Real nt_arad;                               // LTE split of the run-time cooling target
  Real nt_lte_max_j_over_eint, nt_lte_rho_cut, nt_lte_tfloor;   // and its gates
  bool nt_radiation_pressure;                 // gas+radiation, not gas alone, supports it
  // Common parameters
  Real psi, sin_psi, cos_psi;                 // tilt of the disk against the spin axis
  Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
  bool is_vertical_field;                     // vertical initial field configuration
  bool potential_inclined;                    // inclined large-scale initial field
  Real potential_incl_m;                      // its field-line inclination parameter m
  Real potential_cutoff, potential_falloff;   // sets region of torus to magnetize
  Real potential_r_pow;                       // set how vector potential scales
  Real potential_beta_min;                    // target plasma beta of the seed field
  Real potential_r_norm;                      // radius the inclined potential scales on
  Real potential_rho_pow;                     // set vector potential dependence on rho
};

//----------------------------------------------------------------------------------------
// Forward declarations: the closed-form profiles call each other freely.

KOKKOS_INLINE_FUNCTION
void GetBoyerLindquistCoordinates(const torus_pgen &pgen, Real x1, Real x2, Real x3,
                                  Real *pr, Real *ptheta, Real *pphi);
KOKKOS_INLINE_FUNCTION
Real CalculateL(const torus_pgen &pgen, Real r, Real sin_theta);
KOKKOS_INLINE_FUNCTION
Real CalculateCovariantUT(const torus_pgen &pgen, Real r, Real sin_theta, Real l);
KOKKOS_INLINE_FUNCTION
Real LogHAux(const torus_pgen &pgen, Real r, Real sin_theta);
KOKKOS_INLINE_FUNCTION
void CalculateVelocityInTorus(const torus_pgen &pgen, Real r, Real sin_theta,
                              Real *pu0, Real *pu3);
KOKKOS_INLINE_FUNCTION
bool EvaluateNTDiskDensityProfile(const torus_pgen &pgen, const Real r_disk,
                                  const Real z_torus, Real *prho);
KOKKOS_INLINE_FUNCTION
bool NTKerrThermalProperties(const torus_pgen &pgen, Real r,
                             Real *pp_over_rho, Real *pheight);
KOKKOS_INLINE_FUNCTION
void CalculateVectorPotentialInTiltedTorus(const torus_pgen &pgen, Real r, Real theta,
                                           Real phi, Real *patheta, Real *paphi);
KOKKOS_INLINE_FUNCTION
Real A1(const torus_pgen &pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(const torus_pgen &pgen, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(const torus_pgen &pgen, Real x1, Real x2, Real x3);

//----------------------------------------------------------------------------------------
//! \fn GetBoyerLindquistCoordinates
//! \brief Boyer-Lindquist coordinates of a point given in Cartesian Kerr-Schild.
//!
//! r solves r^4 - (R^2 - a^2) r^2 - a^2 z^2 = 0 and is floored at M, which is strictly
//! inside the horizon r_+ = M + sqrt(M^2 - a^2) and so only ever reached in excised
//! material; the floor keeps Delta and 1/r finite there.  The azimuth is the Kerr-Schild
//! one: the exact Boyer-Lindquist azimuth differs from it by the integral of a/Delta dr,
//! which no caller needs because phi enters the profiles only through the tilt rotation
//! and drops out of every untilted disk.

KOKKOS_INLINE_FUNCTION
void GetBoyerLindquistCoordinates(const torus_pgen &pgen, Real x1, Real x2, Real x3,
                                  Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), pgen.M);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2 - pgen.spin*x1, pgen.spin*x2 + r*x1);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn TransformVector
//! \brief A'^i = (dx'^i/dx^a) A^a from Boyer-Lindquist to Cartesian Kerr-Schild.

KOKKOS_INLINE_FUNCTION
void TransformVector(const torus_pgen &pgen,
                     Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl,
                     Real x1, Real x2, Real x3,
                     Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  Real delta = SQR(r) - 2.0*pgen.M*r + SQR(pgen.spin);
  Real rho_cks_sq = fmax(SQR(x1) + SQR(x2), 1.0e-24);
  *pa0 = a0_bl + 2.0*pgen.M*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/rho_cks_sq) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/rho_cks_sq) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt(rho_cks_sq/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateFMLFromRPeak
//! \brief l = u^t u_phi placing the pressure maximum of a Fishbone-Moncrief torus exactly
//! at r_peak, i.e. the extremum of the effective potential LogHAux evaluates.  Implements
//! (3.8) of Fishbone & Moncrief (1976); Harm's lfish_calc() is the same expression for
//! M = 1.  Beware the many other definitions of l: this is not -u_phi/u_t.

KOKKOS_INLINE_FUNCTION
Real CalculateFMLFromRPeak(const torus_pgen &pgen, Real r) {
  // The reference expression is for M = 1, so work in units of the hole's mass.
  Real r_norm = r/pgen.M;
  Real a_norm = pgen.spin/pgen.M;
  Real sgn = (pgen.prograde) ? 1.0 : -1.0;
  Real num = sgn*(SQR(r_norm*r_norm) + SQR(a_norm*r_norm) - 2.0*SQR(a_norm)*r_norm)
           - a_norm*(r_norm*r_norm - a_norm*a_norm)*sqrt(r_norm);
  Real denom = SQR(r_norm) - 3.0*r_norm + sgn*2.0*a_norm*sqrt(r_norm);
  if (fabs(denom) < 1.0e-20) {
    return 1.0e30*sgn*pgen.M;
  }
  return (1.0/r_norm * sqrt(1.0/r_norm) * num/denom) * pgen.M;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateCN
//! \brief c and n of the Chakrabarti angular-momentum profile l = c lambda^n.  With
//! n_param = 0 both follow from Keplerian l at the inner edge and at the pressure
//! maximum; with n_param given that assumption is dropped and only c is fitted.
//! The closed form below is written for M = 1, which is why the Chakrabarti torus is
//! refused by the binary generator's mini-disk branch.

KOKKOS_INLINE_FUNCTION
void CalculateCN(const torus_pgen &pgen, Real *cparam, Real *nparam) {
  Real n_input = pgen.n_param;
  Real nn; // slope of angular momentum profile
  Real cc; // constant of angular momentum profile
  Real l_edge = ((SQR(pgen.r_edge) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_edge))/
                 (sqrt(pgen.r_edge)*(pgen.r_edge - 2.0) + pgen.spin));
  Real l_peak = ((SQR(pgen.r_peak) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_peak))/
                 (sqrt(pgen.r_peak)*(pgen.r_peak - 2.0) + pgen.spin));
  Real lambda_edge = sqrt((l_edge*(-2.0*pgen.spin*l_edge + SQR(pgen.r_edge)*pgen.r_edge
                                   + SQR(pgen.spin)*(2.0+pgen.r_edge)))/
                          (2.0*pgen.spin + l_edge*(pgen.r_edge - 2.0)));
  Real lambda_peak = sqrt((l_peak*(-2.0*pgen.spin*l_peak + SQR(pgen.r_peak)*pgen.r_peak
                                   + SQR(pgen.spin)*(2.0+pgen.r_peak)))/
                          (2.0*pgen.spin + l_peak*(pgen.r_peak - 2.0)));
  if (n_input == 0.0) {
    nn = log(l_peak/l_edge)/log(lambda_peak/lambda_edge);
    cc = l_edge*pow(lambda_edge, -nn);
  } else {
    nn = n_input;
    cc = l_peak*pow(lambda_peak, -nn);
  }
  *cparam = cc;
  *nparam = nn;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateL
//! \brief l of the Chakrabarti profile at (r, theta), by bisection on the algebraic
//! equation (l/c)^(2/n) + (l g_phiphi + l^2 g_tphi)/(g_tphi + l g_tt) = 0.

KOKKOS_INLINE_FUNCTION
Real CalculateL(const torus_pgen &pgen, Real r, Real sin_theta) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*pgen.M*r/sigma;
  Real g_03 = -2.0*pgen.M*pgen.spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*pgen.M*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Perform bisection
  Real l_min = 1.0;
  Real l_max = 100.0;
  Real l_val = 0.5*(l_min + l_max);
  for (int n=0; n<1000; ++n) {
    Real error_rel = 2.0*(l_max - l_min)/(l_max + l_min);
    Real tol_rel = 1.0e-8;
    if (error_rel < tol_rel) {
      break;
    }
    Real residual = pow(l_val/pgen.c_param, 2.0/pgen.n_param) +
                    (l_val*g_33 + SQR(l_val)*g_03)/(g_03 + l_val*g_00);
    if (residual < 0.0) {
      l_min = l_val;
      l_val = 0.5*(l_min + l_max);
    } else if (residual > 0.0) {
      l_max = l_val;
      l_val = 0.5*(l_min + l_max);
    } else if (residual == 0.0) {
      break;
    }
  }
  return l_val;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateCovariantUT
//! \brief u_t of a circular orbit of specific angular momentum l, from the normalization.

KOKKOS_INLINE_FUNCTION
Real CalculateCovariantUT(const torus_pgen &pgen, Real r, Real sin_theta, Real l) {
  // Compute BL metric components
  Real sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  Real g_00 = -1.0 + 2.0*pgen.M*r/sigma;
  Real g_03 = -2.0*pgen.M*pgen.spin*r/sigma*SQR(sin_theta);
  Real g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*pgen.M*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  Real u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
//! \fn LogHAux
//! \brief log of the enthalpy h = p_gas/rho of the equilibrium torus.  Fishbone-Moncrief
//! implements the first half of (FM 3.6); the constant offset -0.5*log(2) that the
//! algebraic simplification introduces cancels against log_h_edge.  Chakrabarti follows
//! Chakrabarti (1985) and returns -1 (outside the torus) where the enthalpy is not a
//! finite number >= 1.

KOKKOS_INLINE_FUNCTION
Real LogHAux(const torus_pgen &pgen, Real r, Real sin_theta) {
  Real logh;
  if (!pgen.use_chakrabarti_torus) {  // Fishbone-Moncrief
    Real sin_sq_theta = SQR(sin_theta);
    Real cos_sq_theta = 1.0 - sin_sq_theta;
    Real delta = SQR(r) - 2.0*pgen.M*r + SQR(pgen.spin);      // \Delta
    Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;        // \Sigma
    Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
    Real exp_2nu = sigma * delta / aa;                        // \exp(2\nu) (FM 3.5)
    Real exp_2psi = aa / sigma * sin_sq_theta;                // \exp(2\psi) (FM 3.5)
    Real exp_neg2chi = exp_2nu / exp_2psi;                    // \exp(-2\chi) (cf. FM 2.15)
    Real omega = 2.0*pgen.M*pgen.spin*r/aa;                   // \omega (FM 3.5)
    Real var_a = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
    Real var_b = 0.5 * log((1.0+var_a) / (sigma*delta/aa));
    Real var_c = -0.5 * var_a;
    Real var_d = -pgen.l_peak * omega;
    logh = var_b + var_c + var_d;                             // (FM 3.4)
  } else {  // Chakrabarti
    Real l = CalculateL(pgen, r, sin_theta);
    Real u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
    Real l_edge = CalculateL(pgen, pgen.r_edge, 1.0);
    Real u_t_edge = CalculateCovariantUT(pgen, pgen.r_edge, 1.0, l_edge);
    Real h = u_t_edge/u_t;
    if (pgen.n_param==1.0) {
      h *= pow(l_edge/l, SQR(pgen.c_param)/(SQR(pgen.c_param)-1.0));
    } else {
      Real pow_c = 2.0/pgen.n_param;
      Real pow_l = 2.0-2.0/pgen.n_param;
      Real pow_abs = pgen.n_param/(2.0-2.0*pgen.n_param);
      h *= (pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l   , pow_l)), pow_abs) *
            pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l_edge, pow_l)), -1.0*pow_abs));
    }
    if (isfinite(h) && h >= 1.0) {
      logh = log(h);
    } else {
      logh = -1.0;  // outside the torus
    }
  }
  return logh;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateT
//! \brief Temperature of a radiating torus in gas/radiation pressure and temperature
//! equilibrium: the real root of (arad/3rho) T^4 + T - p_tot/rho = 0.

KOKKOS_INLINE_FUNCTION
Real CalculateT(const torus_pgen &pgen, Real rho, Real ptot_over_rho) {
  // Calculate quartic coefficients
  Real b4 = pgen.arad / (3.0 * rho);
  Real b0 = -ptot_over_rho;

  // Gas-dominated cells, q = (64/27) p_rad/p_gas at T = p/rho below 1.  With S = A^2 +
  // A B + B^2 (A, B the cube roots below) the root is b4^(-1/3) (S^(3/2) - 1)/(S (sqrt(2
  // sqrt(S) - 1/S) + 1/sqrt(S))), and S - 1 -> 0 here: forming delta1 - 1/2 and S - 1 by
  // subtraction leaves an error ~1e-17/q in T.  Both are built from q directly instead
  // (dm = delta1 - 1/2, A - 1 = dm/(A^2 + A + 1)), so every term is positive.
  const Real q = -64.0 * b0 * b0 * b0 * b4 / 27.0;
  if (q < 1.0) {
    if (!(q > 0.0)) {
      // q < 0 (p < 0) has no positive root; at q = 0 the radiation term vanishes.
      return (q < 0.0) ? 0.0 : fmax(ptot_over_rho, 0.0);
    }
    const Real dm = q / (sqrt(0.25 + q) + 0.5);
    const Real a = Kokkos::cbrt(1.0 + dm);
    const Real b = Kokkos::cbrt(dm);
    const Real s1 = (a + 1.0) * dm / (a * a + a + 1.0) + b * (a + b);
    const Real s = 1.0 + s1;
    const Real rs = sqrt(s);
    return pow(b4, -1.0/3.0) * s1 * (s * s + s + 1.0) /
           (s * (s * rs + 1.0) * (sqrt(2.0 * rs - 1.0 / s) + 1.0 / rs));
  }

  // Radiation-dominated cells: no cancellation on this path (rcoef/delta2 ~ q^(-1/4)/4).
  // Calculate real root of z^3 - 4*b0/b4 * z - 1/b4^2 = 0
  Real delta1 = 0.25 - 64.0 * b0 * b0 * b0 * b4 / 27.0;
  if (delta1 < 0.0) {
    return 0.0;
  }
  delta1 = sqrt(delta1);
  if (delta1 < 0.5) {
    return 0.0;
  }
  // A^3 - B^3 = 1 for A = (delta1 + 1/2)^(1/3), B = (delta1 - 1/2)^(1/3), so the root
  // A - B = 1/(A^2 + A B + B^2) is evaluated without the cancellation that the direct
  // difference suffers at large delta1 (it amplified the last-bit differences between
  // host and device pow() to 1e-6 in the seeded temperature).
  const Real ca = pow(delta1 + 0.5, 1.0/3.0);
  const Real cb = pow(delta1 - 0.5, 1.0/3.0);
  Real zroot = 1.0/(ca*ca + ca*cb + cb*cb);
  zroot *= pow(b4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (b4 * rcoef);
  if (delta2 < 0.0) {
    return 0.0;
  }
  delta2 = sqrt(delta2);
  Real root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return 0.0;
  }
  return root;
}

//----------------------------------------------------------------------------------------
// The Novikov-Thorne thin disk.  Its state is built from circular geodesics of the local
// Kerr metric rather than from an equilibrium potential: the orbit fixes u^mu, the
// Page-Thorne flux function fixes T(r), hydrostatic balance in the vertical direction
// fixes the scale height, and the density follows from T(r) at fixed entropy.

//! \fn LocalKerrPhiMetricComponents
//! \brief The t-phi block of the Kerr metric in Boyer-Lindquist coordinates.

KOKKOS_INLINE_FUNCTION
void LocalKerrPhiMetricComponents(const torus_pgen &pgen, Real r, Real sin_theta,
                                  Real *pg_tt, Real *pg_tphi, Real *pg_phiphi) {
  const Real sin_sq_theta = SQR(sin_theta);
  const Real cos_sq_theta = fmax(0.0, 1.0 - sin_sq_theta);
  const Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;
  *pg_tt = -1.0 + 2.0*pgen.M*r/sigma;
  *pg_tphi = -2.0*pgen.M*pgen.spin*r/sigma * sin_sq_theta;
  *pg_phiphi = (SQR(r) + SQR(pgen.spin) +
                2.0*pgen.M*SQR(pgen.spin)*r/sigma*sin_sq_theta) * sin_sq_theta;
}

//! \fn CalculateCircularOrbitFromMetric
//! \brief Omega, l and u_t of the circular orbit of a stationary axisymmetric metric,
//! from the metric and its radial derivative alone.  Used both for the analytic Kerr
//! metric here and for the tabulated phi-averaged binary metric in BBH.cpp.

KOKKOS_INLINE_FUNCTION
bool CalculateCircularOrbitFromMetric(Real g_tt, Real g_tphi, Real g_phiphi,
                                      Real d_g_tt, Real d_g_tphi, Real d_g_phiphi,
                                      bool prograde, Real *pl, Real *pu_t, Real *pomega) {
  *pl = 0.0;
  *pu_t = 0.0;
  *pomega = 0.0;

  const Real term = SQR(d_g_tphi) - d_g_phiphi*d_g_tt;
  if ((!isfinite(term)) || term < 0.0 || fabs(d_g_phiphi) < 1.0e-20) {
    return false;
  }

  const Real orbit_sign = prograde ? 1.0 : -1.0;
  const Real omega = (-d_g_tphi + orbit_sign*sqrt(term)) / d_g_phiphi;
  const Real denom_l = g_tt + omega*g_tphi;
  const Real norm = -(g_tt + 2.0*omega*g_tphi + SQR(omega)*g_phiphi);
  if ((!isfinite(omega)) || (!isfinite(denom_l)) || fabs(denom_l) < 1.0e-20 ||
      (!isfinite(norm)) || norm <= 0.0) {
    return false;
  }

  const Real u0 = 1.0/sqrt(norm);
  const Real l = -(g_tphi + omega*g_phiphi) / denom_l;
  const Real u_t = (g_tt + omega*g_tphi) * u0;
  if ((!isfinite(l)) || (!isfinite(u_t)) || u_t >= 0.0) {
    return false;
  }

  *pl = l;
  *pu_t = u_t;
  *pomega = omega;
  return true;
}

//! \fn CalculateLocalNTCircularOrbit
//! \brief The circular orbit of the local Kerr metric, by centered differences of the
//! metric in r.

KOKKOS_INLINE_FUNCTION
bool CalculateLocalNTCircularOrbit(const torus_pgen &pgen, Real r, Real sin_theta,
                                   Real *pl, Real *pu_t, Real *pomega) {
  *pl = 0.0;
  *pu_t = 0.0;
  *pomega = 0.0;

  if (!pgen.use_nt_disk || (!isfinite(r)) || (!isfinite(sin_theta)) ||
      (!isfinite(pgen.M)) || pgen.M <= 0.0 ||
      (!isfinite(pgen.spin)) || fabs(pgen.spin) > pgen.M ||
      r <= pgen.nt_r_trunc || r >= pgen.nt_r_out ||
      fabs(sin_theta) <= 1.0e-12) {
    return false;
  }

  const Real r_max = pgen.nt_r_out;
  Real dr = fmax(1.0e-5*r, 1.0e-8*pgen.M);
  Real r_m = fmax(r - dr, pgen.nt_r_trunc);
  Real r_p = fmin(r + dr, r_max);
  dr = 0.5*(r_p - r_m);
  if (dr <= 0.0) {
    return false;
  }

  Real g_tt, g_tphi, g_phiphi;
  Real g_tt_p, g_tphi_p, g_phiphi_p;
  Real g_tt_m, g_tphi_m, g_phiphi_m;
  LocalKerrPhiMetricComponents(pgen, r, sin_theta, &g_tt, &g_tphi, &g_phiphi);
  LocalKerrPhiMetricComponents(pgen, r_p, sin_theta, &g_tt_p, &g_tphi_p, &g_phiphi_p);
  LocalKerrPhiMetricComponents(pgen, r_m, sin_theta, &g_tt_m, &g_tphi_m, &g_phiphi_m);

  return CalculateCircularOrbitFromMetric(
      g_tt, g_tphi, g_phiphi,
      (g_tt_p - g_tt_m)/(2.0*dr),
      (g_tphi_p - g_tphi_m)/(2.0*dr),
      (g_phiphi_p - g_phiphi_m)/(2.0*dr),
      pgen.prograde, pl, pu_t, pomega);
}

KOKKOS_INLINE_FUNCTION
bool CalculateLocalNTKeplerianOrbit(const torus_pgen &pgen, Real r,
                                    Real *pl, Real *pu_t, Real *pomega) {
  return CalculateLocalNTCircularOrbit(pgen, r, 1.0, pl, pu_t, pomega);
}

//! \fn NTKerrFluxFunction
//! \brief The relativistic flux function F(r) of Page & Thorne (1974) for a disk
//! truncated at r_in, in units of M = 1, with the three roots of x^3 - 3x + 2a = 0.

KOKKOS_INLINE_FUNCTION
Real NTKerrFluxFunction(Real a, Real r_norm, Real r_in_norm) {
  if ((!isfinite(a)) || fabs(a) > 1.0 || (!isfinite(r_norm)) ||
      (!isfinite(r_in_norm)) || r_norm <= r_in_norm || r_in_norm <= 0.0) {
    return 0.0;
  }

  const Real a_eval = fmax(-1.0 + 1.0e-12, fmin(1.0 - 1.0e-12, a));
  const Real x = sqrt(r_norm);
  const Real x0 = sqrt(r_in_norm);
  const Real acos_a = acos(a_eval);
  const Real s1 = 2.0*cos(acos_a/3.0 - M_PI/3.0);
  const Real s2 = 2.0*cos(acos_a/3.0 + M_PI/3.0);
  const Real s3 = -2.0*cos(acos_a/3.0);
  const Real roots[3] = {s1, s2, s3};

  Real bracket = x - x0 - 1.5*a_eval*log(x/x0);
  for (int n = 0; n < 3; ++n) {
    const Real si = roots[n];
    const Real sj = roots[(n + 1)%3];
    const Real sk = roots[(n + 2)%3];
    const Real denom = si*(si - sj)*(si - sk);
    const Real coeff_num = 3.0*SQR(si - a_eval);
    if ((!isfinite(denom)) || fabs(denom) < 1.0e-20) {
      if (isfinite(coeff_num) && coeff_num < 1.0e-24) {
        continue;
      }
      return 0.0;
    }
    const Real ratio = (x - si)/(x0 - si);
    if ((!isfinite(ratio)) || ratio <= 0.0) {
      return 0.0;
    }
    bracket -= coeff_num/denom * log(ratio);
  }

  const Real denom = 2.0*SQR(x)*(2.0*a_eval + x*x*x - 3.0*x);
  if ((!isfinite(bracket)) || (!isfinite(denom)) || fabs(denom) < 1.0e-20) {
    return 0.0;
  }
  const Real flux = 3.0*bracket/denom;
  return (isfinite(flux) && flux > 0.0) ? flux : 0.0;
}

//! \fn NTKerrTemperatureProfile
//! \brief F(r)/r, the shape of sigma_SB T^4 for the thin disk.  Only its ratio to the
//! value at r_peak is ever used, so the normalization is irrelevant.

KOKKOS_INLINE_FUNCTION
Real NTKerrTemperatureProfile(const torus_pgen &pgen, Real r) {
  if ((!isfinite(r)) || r <= pgen.nt_r_trunc || r >= pgen.nt_r_out ||
      (!isfinite(pgen.M)) || pgen.M <= 0.0) {
    return 0.0;
  }
  const Real a_norm = pgen.spin/pgen.M;
  const Real a_orbit = (pgen.prograde ? 1.0 : -1.0) * a_norm;
  const Real r_norm = r/pgen.M;
  const Real r_in_norm = pgen.nt_r_trunc/pgen.M;
  const Real flux = NTKerrFluxFunction(a_orbit, r_norm, r_in_norm);
  const Real profile = flux/r_norm;
  return (isfinite(profile) && profile > 0.0) ? profile : 0.0;
}

//! \fn NTMetricVerticalFactor
//! \brief The 1/(1 - Omega l) redshift factor of the vertical epicyclic frequency, for a
//! metric known only through Omega and l (the tabulated binary metric).

KOKKOS_INLINE_FUNCTION
Real NTMetricVerticalFactor(Real omega, Real l) {
  const Real one_minus_omega_l = 1.0 - omega*l;
  if ((!isfinite(one_minus_omega_l)) || one_minus_omega_l <= 1.0e-20) {
    return 1.0;
  }
  return 1.0/one_minus_omega_l;
}

//! \fn NTKerrVerticalFactor
//! \brief The same factor for the analytic Kerr metric, where the ratio of the vertical
//! epicyclic frequency to the orbital one is known in closed form.

KOKKOS_INLINE_FUNCTION
Real NTKerrVerticalFactor(const torus_pgen &pgen, Real r, Real omega, Real l) {
  if ((!isfinite(r)) || r <= 0.0 || (!isfinite(pgen.M)) || pgen.M <= 0.0 ||
      (!isfinite(omega)) || (!isfinite(l))) {
    return 1.0;
  }
  const Real r_norm = r/pgen.M;
  const Real a_norm = (pgen.prograde ? 1.0 : -1.0) * pgen.spin/pgen.M;
  const Real delta = SQR(r_norm) - 2.0*r_norm + SQR(a_norm);
  const Real aa = SQR(r_norm) + SQR(a_norm);
  const Real numer = SQR(aa) + 2.0*SQR(a_norm)*delta;
  const Real denom = SQR(aa) - 2.0*SQR(a_norm)*delta;
  const Real one_minus_omega_l = 1.0 - omega*l;
  if ((!isfinite(delta)) || (!isfinite(numer)) || (!isfinite(denom)) ||
      fabs(denom) < 1.0e-20 || (!isfinite(one_minus_omega_l)) ||
      one_minus_omega_l <= 1.0e-20) {
    return 1.0;
  }
  const Real factor = numer/(denom*one_minus_omega_l);
  return (isfinite(factor) && factor > 0.0) ? factor : 1.0;
}

//! \fn NTDiskPressureOverDensityFromProfile
//! \brief p/rho at r, anchored so that the disk has the requested aspect ratio at r_peak
//! and follows T ~ profile^(1/4) outwards.

KOKKOS_INLINE_FUNCTION
Real NTDiskPressureOverDensityFromProfile(const torus_pgen &pgen, Real profile,
                                          Real profile_peak, Real vertical_factor_peak) {
  if ((!isfinite(profile)) || (!isfinite(profile_peak)) ||
      profile <= 0.0 || profile_peak <= 0.0 ||
      (!isfinite(vertical_factor_peak)) || vertical_factor_peak <= 0.0 ||
      !(pgen.nt_h_over_r > 0.0) || !(pgen.r_peak > 0.0)) {
    return 0.0;
  }
  const Real mass = (pgen.M > 0.0) ? pgen.M : 1.0;
  const Real r_peak_norm = pgen.r_peak/mass;
  if ((!isfinite(r_peak_norm)) || r_peak_norm <= 0.0) {
    return 0.0;
  }
  const Real temp_ratio = pow(profile/profile_peak, 0.25);
  const Real p_peak_over_rho = SQR(pgen.nt_h_over_r) * vertical_factor_peak/r_peak_norm;
  const Real p_over_rho = p_peak_over_rho * temp_ratio;
  return (isfinite(p_over_rho) && p_over_rho > 0.0) ? p_over_rho : 0.0;
}

//! \fn NTDiskScaleHeight
//! \brief H = sqrt(p/rho) / Omega_z, with Omega_z^2 = (M/r^3) * vertical_factor.

KOKKOS_INLINE_FUNCTION
Real NTDiskScaleHeight(const torus_pgen &pgen, Real r, Real p_over_rho,
                       Real vertical_factor) {
  if ((!isfinite(r)) || r <= 0.0 || (!isfinite(p_over_rho)) || p_over_rho <= 0.0 ||
      (!isfinite(vertical_factor)) || vertical_factor <= 0.0) {
    return 0.0;
  }
  const Real mass = (pgen.M > 0.0) ? pgen.M : 1.0;
  const Real r_norm = r/mass;
  if ((!isfinite(r_norm)) || r_norm <= 0.0) {
    return 0.0;
  }
  const Real h_norm = sqrt(p_over_rho*SQR(r_norm)*r_norm/vertical_factor);
  const Real height = mass*h_norm;
  return (isfinite(height) && height > 0.0) ? height : 0.0;
}

//! \fn NTKerrThermalProperties
//! \brief p/rho and H at r for the analytic Kerr background.

KOKKOS_INLINE_FUNCTION
bool NTKerrThermalProperties(const torus_pgen &pgen, Real r,
                             Real *pp_over_rho, Real *pheight) {
  *pp_over_rho = 0.0;
  *pheight = 0.0;
  const Real profile = NTKerrTemperatureProfile(pgen, r);
  const Real profile_peak = NTKerrTemperatureProfile(pgen, pgen.r_peak);
  if (!(profile > 0.0) || !(profile_peak > 0.0)) {
    return false;
  }

  Real l = 0.0, u_t = 0.0, omega = 0.0;
  Real l_peak = 0.0, u_t_peak = 0.0, omega_peak = 0.0;
  if (!CalculateLocalNTKeplerianOrbit(pgen, r, &l, &u_t, &omega) ||
      !CalculateLocalNTKeplerianOrbit(pgen, pgen.r_peak, &l_peak, &u_t_peak,
                                      &omega_peak)) {
    return false;
  }
  const Real vertical_factor = NTKerrVerticalFactor(pgen, r, omega, l);
  const Real vertical_factor_peak =
      NTKerrVerticalFactor(pgen, pgen.r_peak, omega_peak, l_peak);
  const Real p_over_rho =
      NTDiskPressureOverDensityFromProfile(pgen, profile, profile_peak,
                                           vertical_factor_peak);
  const Real height = NTDiskScaleHeight(pgen, r, p_over_rho, vertical_factor);
  if (!(p_over_rho > 0.0) || !(height > 0.0)) {
    return false;
  }
  *pp_over_rho = p_over_rho;
  *pheight = height;
  return true;
}

//! \fn FindLocalNTProfilePeakRadius
//! \brief Where F(r)/r peaks, which is where the thin disk is hottest and densest.  A
//! coarse logarithmic scan followed by a golden-section refinement; host-side only, it
//! runs once during initialization.

inline bool FindLocalNTProfilePeakRadius(const torus_pgen &pgen, Real *pr_peak) {
  *pr_peak = 0.0;
  if (!pgen.use_nt_disk || !(pgen.M > 0.0) ||
      !(pgen.nt_r_trunc > 0.0) || !(pgen.nt_r_out > pgen.nt_r_trunc)) {
    return false;
  }

  const Real r_min = pgen.nt_r_trunc*(1.0 + 1.0e-8);
  const Real r_max = pgen.nt_r_out*(1.0 - 1.0e-8);
  if (!(r_max > r_min)) {
    return false;
  }

  constexpr int nscan = 512;
  const Real log_r_min = log(r_min);
  const Real dlogr = (log(r_max) - log_r_min)/static_cast<Real>(nscan);
  Real best_profile = 0.0;
  int best_idx = -1;
  for (int n = 0; n <= nscan; ++n) {
    const Real r = exp(log_r_min + dlogr*static_cast<Real>(n));
    const Real profile = NTKerrTemperatureProfile(pgen, r);
    if (isfinite(profile) && profile > best_profile) {
      best_profile = profile;
      best_idx = n;
    }
  }
  if (!(best_profile > 0.0) || best_idx <= 0 || best_idx >= nscan) {
    return false;
  }

  Real lo = log_r_min + dlogr*static_cast<Real>(best_idx - 1);
  Real hi = log_r_min + dlogr*static_cast<Real>(best_idx + 1);
  const Real invphi = 0.5*(sqrt(5.0) - 1.0);
  Real c = hi - invphi*(hi - lo);
  Real d = lo + invphi*(hi - lo);
  Real fc = NTKerrTemperatureProfile(pgen, exp(c));
  Real fd = NTKerrTemperatureProfile(pgen, exp(d));
  for (int iter = 0; iter < 80; ++iter) {
    if (fc < fd) {
      lo = c;
      c = d;
      fc = fd;
      d = lo + invphi*(hi - lo);
      fd = NTKerrTemperatureProfile(pgen, exp(d));
    } else {
      hi = d;
      d = c;
      fd = fc;
      c = hi - invphi*(hi - lo);
      fc = NTKerrTemperatureProfile(pgen, exp(c));
    }
  }

  const Real best_r = exp(0.5*(lo + hi));
  best_profile = NTKerrTemperatureProfile(pgen, best_r);
  if (!(best_r > pgen.nt_r_trunc) || !(best_r < pgen.nt_r_out) ||
      !(best_profile > 0.0) || !isfinite(best_r) || !isfinite(best_profile)) {
    return false;
  }
  *pr_peak = best_r;
  return true;
}

//! \fn SmoothTaper01
//! \brief The quintic smoothstep, C2 at both ends, used to taper the disk to nothing at
//! its truncation radius and at its outer edge.

KOKKOS_INLINE_FUNCTION
Real SmoothTaper01(const Real x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

//! \fn NTDiskEdgeTaper
//! \brief The fraction of the untapered thin-disk profile the two edge smoothsteps leave
//! standing at r_disk, normalized to unity at r_peak.  It is the disk's own statement of
//! where it stops being a disk, and it depends on nothing but the disk's parameters --
//! which is why the seed-field normalization uses it to decide which midplane cells are
//! disk, rather than comparing against a background that the deck can rescale.

KOKKOS_INLINE_FUNCTION
Real NTDiskEdgeTaper(const torus_pgen &pgen, const Real r_disk) {
  const Real inner_width = fmax(pgen.nt_inner_taper_width, 1.0e-6);
  const Real outer_width = fmax(pgen.nt_outer_taper_width, 1.0e-6);
  const Real inner_taper = SmoothTaper01((r_disk - pgen.nt_r_trunc) / inner_width);
  const Real inner_taper_peak =
      SmoothTaper01((pgen.r_peak - pgen.nt_r_trunc) / inner_width);
  const Real outer_taper = SmoothTaper01((pgen.nt_r_out - r_disk) / outer_width);
  const Real outer_taper_peak =
      SmoothTaper01((pgen.nt_r_out - pgen.r_peak) / outer_width);
  return fmin(inner_taper/fmax(inner_taper_peak, 1.0e-12), 1.0) *
         fmin(outer_taper/fmax(outer_taper_peak, 1.0e-12), 1.0);
}

//! \fn EvaluateNTDiskDensityFromRadialProfile
//! \brief The thin-disk density given the temperature profile and scale height already
//! evaluated at this radius.  rho ~ T^(1/(4(gamma-1))) holds the disk on one adiabat;
//! the tapers are normalized to unity at r_peak so that rho_max is the actual peak.

KOKKOS_INLINE_FUNCTION
bool EvaluateNTDiskDensityFromRadialProfile(const torus_pgen &pgen, const Real r_disk,
                                            const Real z_torus, const Real profile,
                                            const Real profile_peak,
                                            const Real scale_height, Real *prho) {
  *prho = 0.0;

  if (!pgen.use_nt_disk || !isfinite(r_disk) || !isfinite(z_torus) ||
      pgen.nt_r_trunc <= 0.0 || pgen.nt_h_over_r <= 0.0 ||
      pgen.nt_r_out <= pgen.r_peak || pgen.r_peak <= pgen.nt_r_trunc ||
      pgen.nt_inner_taper_width <= 0.0 || pgen.nt_outer_taper_width <= 0.0 ||
      pgen.rho_max <= pgen.rho_min || r_disk <= pgen.nt_r_trunc ||
      r_disk >= pgen.nt_r_out || profile <= 0.0 || profile_peak <= 0.0 ||
      scale_height <= 0.0) {
    return false;
  }

  const Real gamma_profile = (pgen.gamma_adi > 1.0) ? pgen.gamma_adi : 5.0/3.0;
  const Real exponent = 1.0/(4.0*(gamma_profile - 1.0));
  const Real radial_nt = pow(profile/profile_peak, exponent);
  if ((!isfinite(radial_nt)) || radial_nt <= 0.0) {
    return false;
  }

  const Real radial = radial_nt * NTDiskEdgeTaper(pgen, r_disk);
  if ((!isfinite(radial)) || (!isfinite(scale_height)) || scale_height <= 0.0) {
    return false;
  }

  const Real z_over_h = z_torus / scale_height;
  const Real vertical = exp(-0.5*SQR(z_over_h));
  const Real rho = pgen.rho_max * radial * vertical;
  if ((!isfinite(rho)) || rho <= pgen.rho_min) {
    return false;
  }

  *prho = rho;
  return true;
}

//! \fn EvaluateNTDiskDensityProfile
//! \brief The thin-disk density at (r, z) of the local Kerr background.

KOKKOS_INLINE_FUNCTION
bool EvaluateNTDiskDensityProfile(const torus_pgen &pgen, const Real r_disk,
                                  const Real z_torus, Real *prho) {
  const Real profile = NTKerrTemperatureProfile(pgen, r_disk);
  const Real profile_peak = NTKerrTemperatureProfile(pgen, pgen.r_peak);
  Real p_over_rho = 0.0, scale_height = 0.0;
  if (!NTKerrThermalProperties(pgen, r_disk, &p_over_rho, &scale_height)) {
    return false;
  }
  return EvaluateNTDiskDensityFromRadialProfile(pgen, r_disk, z_torus,
                                                profile, profile_peak,
                                                scale_height, prho);
}

//! \fn NTDiskInnerTaperEndRadius
//! \brief nt_r_trunc + nt_inner_taper_width, where the thin disk's inner smoothstep
//! itself reaches one.  It is the anchor potential_r_norm of the inclined seed on the
//! thin disk, where the seed's switch-on ramp ends.
//!
//! It is not where the disk reaches full density.  NTDiskEdgeTaper divides the inner
//! smoothstep by its value at r_peak and caps the ratio at one, so the density is the
//! untapered profile from min(r_peak, nt_r_trunc + nt_inner_taper_width) outward.  The
//! default width puts this radius at r_peak for a deck that sets r_peak; under the auto
//! peak the default is derived from 2.25 nt_r_trunc instead.  Every deck with
//! r_peak < nt_r_trunc + nt_inner_taper_width therefore ramps the seed on over
//! full-density disk between r_peak and this radius.

KOKKOS_INLINE_FUNCTION
Real NTDiskInnerTaperEndRadius(const torus_pgen &pgen) {
  return pgen.nt_r_trunc + pgen.nt_inner_taper_width;
}

//----------------------------------------------------------------------------------------
// Radiation pressure inside the thin disk.  With nt_radiation_pressure the profile's
// pressure is the TOTAL vertical support, gas plus LTE photons.  The builders write that
// total as the gas pressure and record which cells the disk holds (NTDiskHoldsCell);
// once the field is in place, RepartitionGasForRadiation splits exactly those cells
// between gas and J = arad T^4 and InitializeM1RadiationFromGas hangs the matching LTE
// field on them (m1_gas_seed.hpp).  The split is the seed's own, so it applies the
// seed's magnetization weight, temperature gate, j/e_int cap and, under
// <two_temperature>, T_e, and every other cell -- the ambient included -- keeps its whole
// support as gas and is seeded thin.

KOKKOS_INLINE_FUNCTION
bool NTRadiationPressureActive(const torus_pgen &pgen) {
  return pgen.use_nt_disk && pgen.nt_radiation_pressure;
}

//! \fn NTDiskHoldsCell
//! \brief Whether the thin disk, not the background it sits in, sets a cell's state: the
//! cell lies inside the Novikov-Thorne profile (in_disk) and the profile's density
//! exceeds the background density there.  The profile's Gaussian tail reaches down to
//! rho_min, so in_disk alone also covers cells whose density and pressure are the
//! background's.

KOKKOS_INLINE_FUNCTION
bool NTDiskHoldsCell(const bool in_disk, const Real rho_disk, const Real rho_bg) {
  return in_disk && isfinite(rho_disk) && rho_disk > rho_bg;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateVelocityInTorus
//! \brief u^t and u^phi of the disk at (r, sin theta), in Boyer-Lindquist coordinates.
//! The thin disk sits on a circular geodesic; the Chakrabarti torus rotates on its l
//! profile; the Fishbone-Moncrief torus follows (FM 3.3), whose u^3 in terms of
//! u_{(phi)} is tedious to derive but matches Harm's init.c.

KOKKOS_INLINE_FUNCTION
void CalculateVelocityInTorus(const torus_pgen &pgen, Real r, Real sin_theta,
                              Real *pu0, Real *pu3) {
  // Compute BL metric components
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*pgen.M*r + SQR(pgen.spin);        // \Delta
  Real sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;          // \Sigma
  Real aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
  Real g_00 = -(1.0 - 2.0*pgen.M*r/sigma);                    // g_tt
  Real g_03 = -2.0*pgen.M*pgen.spin*r/sigma * sin_sq_theta;   // g_tp
  Real g_33 = (sigma + (1.0 + 2.0*pgen.M*r/sigma) *
              SQR(pgen.spin) * sin_sq_theta) * sin_sq_theta;  // g_pp
  Real g00 = -aa/(delta*sigma);                               // g^tt
  Real g03 = -2.0*pgen.M*pgen.spin*r/(delta*sigma);           // g^tp

  Real u0 = 0.0, u3 = 0.0;
  if (pgen.use_nt_disk) {
    Real l = 0.0;
    Real u_t = 0.0;
    Real omega = 0.0;
    if (CalculateLocalNTCircularOrbit(pgen, r, fabs(sin_theta), &l, &u_t, &omega)) {
      Real g_tt_nt, g_tphi_nt, g_phiphi_nt;
      LocalKerrPhiMetricComponents(pgen, r, fabs(sin_theta),
                                   &g_tt_nt, &g_tphi_nt, &g_phiphi_nt);
      const Real det = SQR(g_tphi_nt) - g_tt_nt*g_phiphi_nt;
      if (isfinite(det) && fabs(det) > 1.0e-20) {
        const Real g_inv_tt = -g_phiphi_nt/det;
        const Real g_inv_tphi = g_tphi_nt/det;
        u0 = (g_inv_tt - l*g_inv_tphi) * u_t;
        u3 = omega * u0;
      } else {
        u0 = 1.0;
        u3 = 0.0;
      }
    } else {
      u0 = 1.0;
      u3 = 0.0;
    }
  } else if (pgen.use_chakrabarti_torus) {
    Real l = CalculateL(pgen, r, sin_theta);
    Real u_0 = CalculateCovariantUT(pgen, r, sin_theta, l);   // u_t
    Real omega = -(g_03 + l*g_00)/(g_33 + l*g_03);
    u0 = (g00 - l*g03) * u_0;                                 // u^t
    u3 = omega * u0;                                          // u^p
  } else {  // Fishbone-Moncrief
    Real exp_2nu = sigma * delta / aa;                        // \exp(2\nu) (FM 3.5)
    Real exp_2psi = aa / sigma * sin_sq_theta;                // \exp(2\psi) (FM 3.5)
    Real exp_neg2chi = exp_2nu / exp_2psi;                    // \exp(-2\chi) (cf. FM 2.15)
    Real u_phi_proj_a = 1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi;
    Real u_phi_proj_b = -1.0 + sqrt(u_phi_proj_a);
    Real u_phi_proj = sqrt(0.5 * u_phi_proj_b);               // (FM 3.3)
    u_phi_proj *= (pgen.prograde) ? 1.0 : -1.0;
    Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
    Real u3_b = 2.0*pgen.M*pgen.spin*r * sqrt(u3_a);
    Real u3_c = sqrt(sigma/aa) / sin_theta;
    u3 = u3_b + u3_c * u_phi_proj;
    Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
    if (u0_a - g_00 < 0.0) {  // cannot happen inside the torus
      u0 = 1.0;
      u3 = 0.0;
    } else {
      Real u0_b = sqrt(u0_a - g_00);
      u0 = -1.0/g_00 * (g_03*u3 + u0_b);
    }
  }
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateVelocityInTiltedTorus
//! \brief u^mu at a point of a disk whose axis is tilted by psi against the spin axis:
//! find the corresponding point of the untilted disk, evaluate the velocity there, and
//! rotate it back.

KOKKOS_INLINE_FUNCTION
void CalculateVelocityInTiltedTorus(const torus_pgen &pgen, Real r, Real theta, Real phi,
                                    Real *pu0, Real *pu1, Real *pu2, Real *pu3) {
  // Calculate corresponding location
  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_phi = sin(phi);
  Real cos_phi = cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (pgen.psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    Real vary = y;
    Real varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = atan2(vary, varx);
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
  }
  Real sin_varphi = sin(varphi);
  Real cos_varphi = cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(pgen, r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (pgen.psi != 0.0) {
    Real dtheta_dvartheta =
        (pgen.cos_psi * sin_vartheta
         + pgen.sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -pgen.sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = pgen.sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (pgen.cos_psi * sin_vartheta + pgen.sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ReconstructLocalNTDiskState
//! \brief Density, support pressure and Boyer-Lindquist 4-velocity of the thin disk at a
//! point given in Cartesian Kerr-Schild coordinates centered on the hole.  Returns false
//! where there is no disk, leaving the caller's background in place.  The pressure is the
//! profile's whole vertical support: the gas pressure, or with nt_radiation_pressure the
//! gas-plus-radiation total that RepartitionGasForRadiation splits once the field is
//! built.

KOKKOS_INLINE_FUNCTION
bool ReconstructLocalNTDiskState(const torus_pgen &pgen, Real x1, Real x2, Real x3,
                                 Real *prho, Real *ppgas,
                                 Real *pu0_bl, Real *pu1_bl,
                                 Real *pu2_bl, Real *pu3_bl) {
  *prho = 0.0;
  *ppgas = 0.0;
  *pu0_bl = 0.0;
  *pu1_bl = 0.0;
  *pu2_bl = 0.0;
  *pu3_bl = 0.0;

  if (!pgen.use_nt_disk) {
    return false;
  }

  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if ((!isfinite(r)) || (!isfinite(theta)) || r <= 0.0) {
    return false;
  }

  const Real sin_theta = sin(theta);
  const Real cos_theta = cos(theta);
  const Real sin_phi = sin(phi);
  const Real cos_phi = cos(phi);
  Real sin_vartheta, cos_vartheta;
  if (pgen.psi != 0.0) {
    const Real x = sin_theta * cos_phi;
    const Real y = sin_theta * sin_phi;
    const Real z = cos_theta;
    const Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    const Real vary = y;
    const Real varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
  }
  if ((!isfinite(sin_vartheta)) || (!isfinite(cos_vartheta)) || sin_vartheta <= 1.0e-12) {
    return false;
  }

  const Real r_disk = r;
  Real rho = 0.0;
  if (!EvaluateNTDiskDensityProfile(pgen, r_disk, r * cos_vartheta, &rho)) {
    return false;
  }

  Real p_over_rho = 0.0, scale_height = 0.0;
  if (!NTKerrThermalProperties(pgen, r_disk, &p_over_rho, &scale_height)) {
    return false;
  }

  const Real ptot_target = rho * p_over_rho;
  if ((!isfinite(ptot_target)) || ptot_target <= 0.0) {
    return false;
  }

  if (ptot_target <= pgen.pgas_min) {
    return false;
  }

  CalculateVelocityInTiltedTorus(pgen, r, theta, phi, pu0_bl, pu1_bl, pu2_bl, pu3_bl);
  if ((!isfinite(*pu0_bl)) || (!isfinite(*pu1_bl)) ||
      (!isfinite(*pu2_bl)) || (!isfinite(*pu3_bl))) {
    return false;
  }

  *prho = rho;
  *ppgas = ptot_target;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn NTDiskMidplaneBodyPressure
//! \brief The unperturbed support pressure of a cell that samples the thin disk's
//! midplane body, the set the inclined seed's minimum midplane beta is taken over.
//! Returns false for any other cell.
//!
//! (x1, x2, x3) is the cell centre relative to the hole, r_d and cos_vt its Boyer-Lindquist
//! radius and cosine of the (tilted) polar angle, dz the cell's own x3 width.  A cell
//! samples the midplane when its z-extent reaches the disk plane; the test carries a
//! rounding margin because a mesh symmetric about that plane puts the two nearest rows
//! at exactly +/- dz/2.  The body is where NTDiskEdgeTaper still leaves at least a tenth
//! of the untapered profile standing, inside nt_r_out - nt_outer_taper_width: below that
//! the cell is background wearing a vanishing disk, and the open seed, which is not cut
//! at the outer edge, keeps its R^{-5/4} b^2 across the outer taper while the disk
//! pressure falls to nothing there.  The same tenth applies vertically: the cell centre
//! must lie where the Gaussian exp(-z^2/2H^2) still leaves a tenth of the plane's
//! pressure, |z| <= sqrt(2 ln 10) H.  A midplane cell of a mesh that does not resolve H
//! there samples the Gaussian's wing, whose pressure falls with z while b^2 does not,
//! and would set the minimum on a near-empty cell.

KOKKOS_INLINE_FUNCTION
bool NTDiskMidplaneBodyPressure(const torus_pgen &pgen, const Real x1, const Real x2,
                                const Real x3, const Real r_d, const Real cos_vt,
                                const Real dz, Real *ppgas) {
  constexpr Real taper_min = 0.1;
  *ppgas = 0.0;
  if (!(r_d >= pgen.nt_r_trunc && r_d <= pgen.nt_r_out &&
        fabs(r_d*cos_vt) <= 0.5*dz*(1.0 + 1.0e-8))) {
    return false;
  }
  Real rho_d = 0.0, pgas_d = 0.0;
  Real u0_d = 0.0, u1_d = 0.0, u2_d = 0.0, u3_d = 0.0;
  ReconstructLocalNTDiskState(pgen, x1, x2, x3, &rho_d, &pgas_d,
                              &u0_d, &u1_d, &u2_d, &u3_d);
  const Real r_body_out = pgen.nt_r_out - fmax(pgen.nt_outer_taper_width, 0.0);
  if (!(isfinite(rho_d) && rho_d > 0.0 && isfinite(pgas_d) && pgas_d > 0.0 &&
        NTDiskEdgeTaper(pgen, r_d) >= taper_min && r_d <= r_body_out)) {
    return false;
  }
  Real p_over_rho_d = 0.0, height_d = 0.0;
  if (!(NTKerrThermalProperties(pgen, r_d, &p_over_rho_d, &height_d) &&
        SQR(r_d*cos_vt) <= -2.0*log(taper_min)*SQR(height_d))) {
    return false;
  }
  *ppgas = pgas_d;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn TorusDensityForPotential
//! \brief The analytic disk density at (r, sin vartheta, cos vartheta), as the vector
//! potential sees it: the thin-disk profile, or the equilibrium torus solution.  Returns
//! false outside the disk.

KOKKOS_INLINE_FUNCTION
bool TorusDensityForPotential(const torus_pgen &pgen, Real r, Real sin_vartheta,
                              Real cos_vartheta, Real *prho) {
  *prho = 0.0;
  if (pgen.use_nt_disk) {
    return EvaluateNTDiskDensityProfile(pgen, r, r*cos_vartheta, prho);
  }
  const Real gm1 = pgen.gamma_adi - 1.0;
  const Real log_h = LogHAux(pgen, r, sin_vartheta) - pgen.log_h_edge;  // (FM 3.6)
  if (!(log_h >= 0.0)) {
    return false;
  }
  const Real ptot_over_rho = gm1/pgen.gamma_adi * (exp(log_h) - 1.0);
  *prho = pow(ptot_over_rho, 1.0/gm1) / pgen.rho_peak;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn CalculateVectorPotentialInTiltedTorus
//! \brief A_theta and A_phi of the initial field in spherical Kerr-Schild coordinates.
//!
//! Three configurations.  "inclined" is the inclined large-scale poloidal field of Zanni
//! et al. (2007), in the form of eq. (1.18) of Dihingia & Fendt, "Thin Accretion Disks in
//! GR-MHD simulations" (arXiv:2404.06140):
//!
//!   A_phi  ~  (r sin(theta))^(3/4) m^(5/4) / (m^2 + (tan theta)^(-2))^(5/8)
//!
//! Multiplying numerator and denominator by sin(theta)^(5/4) clears the cotangent:
//!
//!   A_phi  ~  [(r/r_in)^(3/4) - 1] sin^2(theta) m^(5/4)
//!             / (m^2 sin^2(theta) + cos^2(theta))^(5/8),
//!
//! which never divides by sin(theta).  The denominator is 1 + (m^2-1) sin^2 and so lies
//! between min(1,m^2) and max(1,m^2); on the axis sin -> 0 and A_phi vanishes like
//! theta^2, and on the midplane sin = 1 gives A_phi = (r/r_in)^(3/4) exactly, i.e. the
//! self-similar B_z ~ R^(-5/4).  Subtracting 1 makes A_phi vanish on the gate sphere
//! r = r_in so that the truncation carries no current sheet; along the midplane the
//! subtracted term is constant and leaves B_z ~ R^(-5/4) exact.  m sets the inclination
//! of the field lines to the disk surface and with it the flux threading the disk: small
//! m bends them away from the axis, which is what makes a Blandford-Payne wind possible,
//! large m approaches a uniform vertical field.  This form is not tied to the density, so
//! unlike the other two it threads the disk with open field lines and closes nowhere in
//! the domain; potential_cutoff, potential_rho_pow and potential_r_pow are properties of
//! the density-tied loop and take no part here, while potential_falloff does, as a radial
//! truncation of the flux, applied exactly as it is below.
//!
//! "vertical" is a more-or-less vertical field falling to zero on the edges, and the
//! default is the density-tied poloidal loop of the SANE/MAD family.  potential_cutoff is
//! subtracted from rho/rho_max BEFORE the radial scaling is applied, so that it means
//! what its name says -- a density contour -- and the seed field is exactly zero in
//! low-density material whatever potential_r_pow does at large radius.

KOKKOS_INLINE_FUNCTION
void CalculateVectorPotentialInTiltedTorus(const torus_pgen &pgen, Real r, Real theta,
                                           Real phi, Real *patheta, Real *paphi) {
  // Find vector potential components, accounting for tilt
  Real atheta = 0.0, aphi = 0.0;

  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_phi = sin(phi);
  Real cos_phi = cos(phi);
  Real sin_vartheta, cos_vartheta;

  if (pgen.psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi * x - pgen.sin_psi * z;
    Real vary = y;
    Real varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
  } else {
    sin_vartheta = fabs(sin(theta));
    cos_vartheta = cos_theta;
  }

  if (pgen.potential_inclined) {
    // Dihingia & Fendt (2024) eq. 1.18, after Zanni et al. (2007):
    //   A_phi ~ (r sin theta)^{3/4} m^{5/4} / (m^2 + cot^2 theta)^{5/8},
    // written with (m^2 + cot^2)^{-5/8} = sin^{5/4} (1 + (m^2 - 1) sin^2)^{-5/8} so the
    // pole is regular.  Outside potential_r_norm this IS the paper's field: the midplane
    // B_z ~ R^{-5/4}, and the field-line inclination at the disk surface is the one m
    // promises.  The paper's field threads the hole too; here it is switched on by a C2
    // step to one at potential_r_norm (from zero on the cylinder R = nt_r_trunc for the
    // thin disk, from zero at the horizon for a torus), so that A_phi is continuous (no
    // current sheet) and the funnel starts field-free.  Outward the potential is not cut
    // at all: the field is the paper's open one everywhere beyond the switch-on.  The
    // step's own dA/dr contributes to B^theta only inside potential_r_norm.  Do NOT
    // subtract a constant instead:
    // (r/r_norm)^{3/4} - 1 leaves B_z alone but scales B_r by 1 - (r_norm/r)^{3/4}, 0.41
    // at 2 r_norm, so the inner disk sees a field far more vertical than its m.
    //
    // potential_r_norm is the disk's inner edge for the equilibrium tori and the end of
    // the thin disk's inner smoothstep (NTDiskInnerTaperEndRadius) for the NT one.  On a
    // thin disk it must lie above nt_r_trunc: at r_norm == nt_r_trunc the switch-on below
    // is a step in A_phi, i.e. a current sheet on the cylinder R = nt_r_trunc.
    const Real r_hor = pgen.M + sqrt(fmax(SQR(pgen.M) - SQR(pgen.spin), 0.0));
    if (r > r_hor) {
      const Real sin_sq = SQR(sin_vartheta);
      const Real m_incl = pgen.potential_incl_m;
      const Real r_norm = pgen.potential_r_norm;
      // Thin disk: the field is cut on the CYLINDER R = nt_r_trunc and switched on across
      // the inner taper, so no field line threads the column above the truncation radius
      // at any height and the funnel starts field-free all the way up; the flux that the
      // paper's potential would put through the hole enters at the disk's inner edge and
      // reaches the hole only by accretion.  A spherical ramp from the horizon instead
      // left the polar column magnetized (r > r_norm there) with its field lines forced
      // to bend out through the ramp shell, and put the ramp's B^theta bump in the dense
      // inner disk.  Equilibrium tori keep the spherical ramp from the horizon.
      // Nothing cuts the potential at the disk's outer edge: beyond nt_r_out the field
      // is the paper's, open and threading the ambient, and the flux the disk carries
      // leaves through the outer boundary rather than returning through an annulus.  A
      // cut there closed the structure onto the disk and, because the disk's own
      // pressure vanishes across the outer taper while the seed does not, put the
      // steepest dA/dR over the lightest gas the disk has.
      Real switch_on = 1.0;
      if (pgen.use_nt_disk) {
        const Real cyl = r*sin_vartheta;
        switch_on = (r_norm > pgen.nt_r_trunc)
            ? SmoothTaper01((cyl - pgen.nt_r_trunc)/(r_norm - pgen.nt_r_trunc))
            : ((cyl >= pgen.nt_r_trunc) ? 1.0 : 0.0);
      } else if (r_norm > r_hor) {
        switch_on = SmoothTaper01((r - r_hor)/(r_norm - r_hor));
      }
      Real aphi_tilt = switch_on * pow(r/r_norm, 0.75) * sin_sq * pow(m_incl, 1.25)
                       / pow(1.0 + (SQR(m_incl) - 1.0)*sin_sq, 0.625);
      if (pgen.potential_falloff != 0) {
        aphi_tilt *= exp(-r/pgen.potential_falloff);
      }
      if (pgen.psi != 0.0) {
        Real dvarphi_dtheta = -pgen.sin_psi * sin_phi / SQR(sin_vartheta);
        Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
            * (pgen.cos_psi * sin_theta - pgen.sin_psi * cos_theta * cos_phi);
        atheta = dvarphi_dtheta * aphi_tilt;
        aphi = dvarphi_dphi * aphi_tilt;
      } else {
        atheta = 0.0;
        aphi = aphi_tilt;
      }
    }

  } else if (pgen.is_vertical_field) {
    Real rho = 0.0;
    bool in_torus = TorusDensityForPotential(pgen, r, sin_vartheta, cos_vartheta, &rho);

    // more-or-less vertical geometry but falling to zero on edges
    Real cyl_radius = r * sin_vartheta;
    Real rcyl_in = pgen.r_edge;
    Real rcyl_falloff = pgen.potential_falloff;

    Real aphi_tilt = pow(cyl_radius/rcyl_in, pgen.potential_r_pow);
    if (pgen.potential_falloff != 0) {
      aphi_tilt *= exp(-cyl_radius/rcyl_falloff);
    }

    Real aphi_offset = exp(-rcyl_in/rcyl_falloff);
    if (cyl_radius < rcyl_in) {
      aphi_tilt = 0.0;
    } else {
      aphi_tilt -= aphi_offset;
    }

    if (pgen.potential_rho_pow != 0) {
      if (in_torus) {
        aphi_tilt *= pow(rho/pgen.rho_max, pgen.potential_rho_pow);
      } else {
        aphi_tilt = 0.0;
      }
    }

    if (pgen.psi != 0.0) {
      Real dvarphi_dtheta = -pgen.sin_psi * sin_phi / SQR(sin_vartheta);
      Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
          * (pgen.cos_psi * sin_theta - pgen.sin_psi * cos_theta * cos_phi);
      atheta = dvarphi_dtheta * aphi_tilt;
      aphi = dvarphi_dphi * aphi_tilt;
    } else {
      atheta = 0.0;
      aphi = aphi_tilt;
    }

  } else {
    if (r >= pgen.r_edge) {
      Real rho = 0.0;
      bool in_torus = TorusDensityForPotential(pgen, r, sin_vartheta, cos_vartheta, &rho);

      Real aphi_tilt = 0.0;
      if (in_torus) {
        Real scaling_param = pow((r/pgen.r_edge)*sin_vartheta, pgen.potential_r_pow);
        if (pgen.potential_falloff != 0) {
          scaling_param *= exp(-r/pgen.potential_falloff);
        }
        // Apply the density cutoff before any radial scaling.  This keeps the
        // seed field exactly zero in low-density material, independent of the
        // large-radius scaling, and therefore avoids isolated initial
        // high-sigma cells at the disk edge.
        const Real rho_weight =
            fmax(rho/pgen.rho_max - pgen.potential_cutoff, 0.0);
        if (rho_weight > 0.0) {
          aphi_tilt = pow(rho_weight, pgen.potential_rho_pow)*scaling_param;
        }
        if (pgen.psi != 0.0) {
          Real dvarphi_dtheta = -pgen.sin_psi * sin_phi / SQR(sin_vartheta);
          Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
              * (pgen.cos_psi * sin_theta - pgen.sin_psi * cos_theta * cos_phi);
          atheta = dvarphi_dtheta * aphi_tilt;
          aphi = dvarphi_dphi * aphi_tilt;
        } else {
          atheta = 0.0;
          aphi = aphi_tilt;
        }
      }
    }
  }

  *patheta = atheta;
  *paphi = aphi;

  return;
}

//----------------------------------------------------------------------------------------
// The Cartesian Kerr-Schild components of the vector potential.  Each first computes the
// spherical-KS components at the point and then transforms them.

KOKKOS_INLINE_FUNCTION
Real A1(const torus_pgen &pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-12) return 0.0;

  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A2(const torus_pgen &pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-12) return 0.0;

  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A3(const torus_pgen &pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  if (r <= 1.0e-12) return 0.0;

  Real atheta, aphi;
  CalculateVectorPotentialInTiltedTorus(pgen, r, theta, phi, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));
}

//----------------------------------------------------------------------------------------
//! \fn TorusPotentialRegion
//! \brief True where the vector potential is permitted to magnetize.  Mirrors exactly the
//! conditions CalculateVectorPotentialInTiltedTorus uses to leave aphi_tilt nonzero.  The
//! disk density is a closed-form function of position, so sampling it away from the
//! requested point is exact.

KOKKOS_INLINE_FUNCTION
bool TorusPotentialRegion(const torus_pgen &pgen, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);

  Real sin_theta = sin(theta);
  Real cos_theta = cos(theta);
  Real sin_vartheta, cos_vartheta;
  if (pgen.psi != 0.0) {
    Real cos_phi = cos(phi);
    Real sin_phi = sin(phi);
    Real x = sin_theta*cos_phi;
    Real y = sin_theta*sin_phi;
    Real z = cos_theta;
    Real varx = pgen.cos_psi*x - pgen.sin_psi*z;
    Real vary = y;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = pgen.sin_psi*x + pgen.cos_psi*z;
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
  }

  if (pgen.potential_inclined) {
    // The inclined large-scale field is independent of the density.  It is the paper's
    // field from the disk's inner edge outward (the switch-on across the plunging region
    // is not counted), and there is no cutoff surface to erode away from.  For the thin
    // disk the edge is the cylinder R = nt_r_trunc the potential is cut on; outward the
    // field is open and unbounded.
    if (pgen.use_nt_disk) {
      return (r*sin_vartheta >= pgen.nt_r_trunc);
    }
    return (r >= pgen.r_edge);
  }

  Real rho = 0.0;
  const bool in_torus = TorusDensityForPotential(pgen, r, sin_vartheta, cos_vartheta,
                                                 &rho);

  if (pgen.is_vertical_field) {
    // This branch carries no density cutoff; it magnetizes disk material outside the
    // cylindrical inner radius.
    if (r*sin_vartheta < pgen.r_edge) return false;
    return (pgen.potential_rho_pow != 0.0) ? in_torus : true;
  }

  if (r < pgen.r_edge || !in_torus) return false;
  return (rho/pgen.rho_max - pgen.potential_cutoff) > 0.0;
}

//----------------------------------------------------------------------------------------
// Vector-potential components eroded by one cell.
//
// B = curl A puts A on cell edges and B on faces, so a single edge with A != 0 magnetizes
// all four cells surrounding it.  With the bare density cutoff the curl therefore leaks
// field one to two cells past the cutoff surface into ambient-density cells.  Measured on
// the Fishbone-Moncrief torus that produced 39120 cells with b^2/rho > 20 whose densities
// were all far below the cutoff (median 4.1e-13), reaching sigma ~ 7.8e5, while genuinely
// magnetized material only reached sigma 4.8e-4 -- a pure discretization artifact that
// nevertheless activates the hybrid FFE classifier on the very first cycle.
//
// Requiring every cell that touches an edge to be inside the magnetized region removes
// it.  Zeroing A rather than B keeps the field an exact curl, so div(B) = 0 is preserved.
// The half-widths are passed in so that the fine sub-position averages used on shared
// fine/coarse faces erode with the FINE spacing; a coarse edge then still equals the
// average of the fine edges and the shared-face flux stays identical across levels.
//
// This is a property of a torus with a sharp density edge, not of every disk: a
// Novikov-Thorne thin disk is a few cells thick at the resolutions these runs use, so
// eroding it by a cell would delete most of its seed flux rather than a leaked rim.  The
// erosion therefore applies to the equilibrium tori only.

KOKKOS_INLINE_FUNCTION
bool ErodeSeedField(const torus_pgen &pgen) {
  // The inclined potential already ramps smoothly from the horizon to r_edge and
  // has no density cutoff. Eroding it would cut a nonzero A at r_edge, create a
  // spurious B ~ A/dx sheet, and make beta normalization resolution-dependent.
  return !pgen.use_nt_disk && !pgen.potential_inclined;
}

KOKKOS_INLINE_FUNCTION
Real ErodedA1(const torus_pgen &pgen, Real x1, Real x2, Real x3, Real hx2, Real hx3) {
  if (!ErodeSeedField(pgen)) return A1(pgen, x1, x2, x3);
  if (!(TorusPotentialRegion(pgen, x1, x2-hx2, x3-hx3) &&
        TorusPotentialRegion(pgen, x1, x2+hx2, x3-hx3) &&
        TorusPotentialRegion(pgen, x1, x2-hx2, x3+hx3) &&
        TorusPotentialRegion(pgen, x1, x2+hx2, x3+hx3))) {
    return 0.0;
  }
  return A1(pgen, x1, x2, x3);
}

KOKKOS_INLINE_FUNCTION
Real ErodedA2(const torus_pgen &pgen, Real x1, Real x2, Real x3, Real hx1, Real hx3) {
  if (!ErodeSeedField(pgen)) return A2(pgen, x1, x2, x3);
  if (!(TorusPotentialRegion(pgen, x1-hx1, x2, x3-hx3) &&
        TorusPotentialRegion(pgen, x1+hx1, x2, x3-hx3) &&
        TorusPotentialRegion(pgen, x1-hx1, x2, x3+hx3) &&
        TorusPotentialRegion(pgen, x1+hx1, x2, x3+hx3))) {
    return 0.0;
  }
  return A2(pgen, x1, x2, x3);
}

KOKKOS_INLINE_FUNCTION
Real ErodedA3(const torus_pgen &pgen, Real x1, Real x2, Real x3, Real hx1, Real hx2) {
  if (!ErodeSeedField(pgen)) return A3(pgen, x1, x2, x3);
  if (!(TorusPotentialRegion(pgen, x1-hx1, x2-hx2, x3) &&
        TorusPotentialRegion(pgen, x1+hx1, x2-hx2, x3) &&
        TorusPotentialRegion(pgen, x1-hx1, x2+hx2, x3) &&
        TorusPotentialRegion(pgen, x1+hx1, x2+hx2, x3))) {
    return 0.0;
  }
  return A3(pgen, x1, x2, x3);
}

//----------------------------------------------------------------------------------------
//! \fn ReadNTDiskParameters
//! \brief Read the <problem> keys that describe the Novikov-Thorne thin disk, so that the
//! same block of a deck means the same thing to both generators.
//!
//!   use_nt_disk           select the thin disk instead of an equilibrium torus
//!   nt_r_trunc            truncation radius (defaults to r_edge where a deck gives one)
//!   nt_h_over_r           aspect ratio at r_peak
//!   nt_r_out              outer edge
//!   nt_inner/outer_taper_width   widths of the smoothstep tapers at those two radii
//!   nt_radiation_pressure vertical support is gas plus radiation, not gas alone
//!
//! With auto_peak the pressure maximum is not a deck parameter: it is where the
//! Page-Thorne flux function peaks, and the caller derives it with
//! FindLocalNTProfilePeakRadius once the hole's mass and spin are known.  Otherwise
//! r_peak is required, as it is for every other disk model.
//!
//! arad_code is the radiation constant in code units, and lte_from_deck says whether the
//! deck also asked for an LTE radiation seed (<problem>/m1_init_lte), which
//! nt_radiation_pressure requires.  The initial condition does not split the disk here:
//! the builders write the total support and RepartitionGasForRadiation splits the cells
//! the disk holds (see NTDiskHoldsCell).  The seed's density cut, j/e_int cap and
//! temperature gate (with the seed's precedence between its two spellings) are read into
//! nt_lte_* for the binary generator's run-time disk-cooling target, which splits the
//! profile's total at run time (srcterms.cpp, BBHMiniDiskTargetEint).

inline void ReadNTDiskParameters(ParameterInput *pin, torus_pgen &t,
                                 const bool auto_peak, const bool lte_from_deck,
                                 const Real arad_code, const Real eos_tfloor) {
  if (t.use_nt_disk) {
    if (auto_peak && !pin->DoesParameterExist("problem", "nt_r_trunc") &&
        !pin->DoesParameterExist("problem", "r_edge")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "nt_r_trunc is required for the auto-peak NT thin disk."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    const Real legacy_r_edge = pin->DoesParameterExist("problem", "r_edge")
        ? pin->GetReal("problem", "r_edge") : (auto_peak ? 0.0 : 0.5*t.r_peak);
    t.nt_r_trunc = pin->GetOrAddReal("problem", "nt_r_trunc", legacy_r_edge);
    t.r_edge = t.nt_r_trunc;
  } else {
    t.r_edge = pin->GetReal("problem", "r_edge");
    t.nt_r_trunc = t.r_edge;
  }
  t.nt_h_over_r = pin->GetOrAddReal("problem", "nt_h_over_r", 0.05);
  const Real nt_r_peak_for_defaults =
      (!auto_peak && t.r_peak > t.nt_r_trunc) ? t.r_peak : 2.25*t.nt_r_trunc;
  t.nt_r_out = pin->GetOrAddReal("problem", "nt_r_out", 4.0*nt_r_peak_for_defaults);
  const Real nt_inner_taper_default =
      fmax(fmax(nt_r_peak_for_defaults - t.nt_r_trunc,
                4.0*t.nt_h_over_r*t.nt_r_trunc), 1.0e-6);
  const Real nt_outer_taper_default =
      fmax(fmax(0.05*t.nt_r_out, 2.0*t.nt_h_over_r*t.nt_r_out), 1.0e-6);
  t.nt_inner_taper_width =
      pin->GetOrAddReal("problem", "nt_inner_taper_width", nt_inner_taper_default);
  t.nt_outer_taper_width =
      pin->GetOrAddReal("problem", "nt_outer_taper_width", nt_outer_taper_default);
  const bool nt_radiation_pressure_requested =
      pin->GetOrAddBoolean("problem", "nt_radiation_pressure", false);
  if (nt_radiation_pressure_requested && !t.use_nt_disk) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "nt_radiation_pressure=true is only supported with use_nt_disk=true."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  t.nt_radiation_pressure = nt_radiation_pressure_requested && t.use_nt_disk;
  t.nt_arad = t.nt_radiation_pressure ? arad_code : 0.0;
  const bool lte_for_nt_pressure = t.nt_radiation_pressure && lte_from_deck;
  t.nt_lte_max_j_over_eint = lte_for_nt_pressure
      ? pin->GetOrAddReal("problem", "m1_init_lte_max_j_over_eint", -1.0)
      : -1.0;
  // Read, never inserted: the initial condition splits the disk on its own cells, so a
  // deck without the density cut should not show one in its parameter dump.
  t.nt_lte_rho_cut = !lte_for_nt_pressure ? 0.0
      : (pin->DoesParameterExist("problem", "m1_init_lte_rho_cut")
             ? pin->GetReal("problem", "m1_init_lte_rho_cut") : 1.0e-4);
  // The LTE temperature gate in the seed's two spellings and with its precedence
  // (InitializeM1RadiationFromGas): m1_init_lte_tfloor_factor times the EOS temperature
  // floor, or m1_init_lte_tfloor itself when only that key is given.  Decided before
  // either key is read, since GetOrAdd would create the factor key the seed's test asks
  // about.
  const bool lte_tfloor_absolute = lte_for_nt_pressure &&
      !pin->DoesParameterExist("problem", "m1_init_lte_tfloor_factor") &&
      pin->DoesParameterExist("problem", "m1_init_lte_tfloor");
  if (lte_tfloor_absolute) {
    t.nt_lte_tfloor = fmax(0.0, pin->GetReal("problem", "m1_init_lte_tfloor"));
  } else {
    const Real nt_lte_tfloor_factor = lte_for_nt_pressure
        ? pin->GetOrAddReal("problem", "m1_init_lte_tfloor_factor", 1.0)
        : 1.0;
    t.nt_lte_tfloor = fmax(0.0, nt_lte_tfloor_factor * eos_tfloor);
  }

  if (t.nt_radiation_pressure &&
      (!lte_for_nt_pressure || !isfinite(t.nt_arad) || t.nt_arad <= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "nt_radiation_pressure=true requires m1_init_lte=true, "
              << "and a positive units-derived arad, photons/arad, or problem/nt_arad."
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

}  // namespace torus_ic

#endif  // PGEN_TORUS_IC_HPP_
