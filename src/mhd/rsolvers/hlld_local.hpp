#ifndef MHD_RSOLVERS_HLLD_LOCAL_HPP_
#define MHD_RSOLVERS_HLLD_LOCAL_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_local.hpp
//! \brief The flat-space relativistic HLLD fan, solved once for both entry points.
//!
//! Relativistic HLLD is derived in Minkowski space, so its states and fluxes are only
//! correct in a locally flat frame.  Special relativity is already that frame; the
//! general-relativistic solver reaches it through a face-aligned orthonormal tetrad
//! (hlld_grmhd.hpp), in which the interface is no longer static but moves with speed
//! v = beta^x/(alpha sqrt(gamma^xx)).  Everything below therefore takes the interface
//! speed as an argument and is shared verbatim by HLLD_SR (v = 0) and HLLD_GR.
//!
//! The wave fan follows Mignone, Ugliano & Bodo (2009, MNRAS 393, 1141; "MUB09") with
//! the changes of Fields, Wong & Stone (2026, arXiv:2609.06150; "FWS26") Sec. II:
//! the iteration is seeded from the weighted average of the two total pressures
//! (FWS26 Eq. 11) rather than from a conserved-to-primitive inversion of the HLL state.
//! The root solve is a secant on f(P_tot) from that seed, converged on P_tot rather than
//! on the residual, and where the seeds fail it is followed by a bracketing search: f is
//! only defined on the interval of pressures the fan admits, and that interval narrows
//! as P_tot/(sigma gamma^2) with the face's magnetisation sigma = b.b/(rho h) and Lorentz
//! factor -- parts in 1e8 of P_tot on a force-free face, not the per cent an ordinary one
//! allows -- and need not contain any seed at all.  Both searches therefore take their
//! step from the face rather than from a fixed ratio.

#include <cmath>

#include "athena.hpp"
#include "eos/grmhd_fast_speeds.hpp"

namespace mhd {
namespace hlld {

//! Root-solve tolerance and iteration cap, FWS26 step 8.  The paper's bound is on
//! |f| = |lambda_cL - lambda_cR|, a difference of two velocities, and 1e-12 is an
//! absolute one.  That is the wrong scale twice over.  The contact velocity carries a
//! factor (1 - K.K) (MUB09 Eq. 49), and on a Poynting-dominated face K.K sits within a
//! part in 1e7 of unity or closer, so f carries only the digits that survive that
//! subtraction -- ContactVelocity below evaluates the factor in a form that does not make
//! it, but the residual is still a difference of two velocities and an absolute bound of
//! 1e-12 on it is a bound on nothing in particular.  At the other end, where f is flat in
//! P_tot, |f| < 1e-12 is met by pressures far enough from the root to leave the
//! induction flux of a *uniform* state wrong at 1e-9 relative while its hydrodynamic
//! fluxes are exact to 1e-13.  P_tot has an unambiguous scale of its own, so the test
//! is on it: the iteration stops when its own step is within kPressureTolerance of the
//! pressure it is refining.  An exactly vanishing residual still stops immediately.
constexpr Real kPressureTolerance = 1.0e-14;
constexpr int kMaxIterations = 15;
//! Below this ratio of (B^x)^2 to the FWS26 Eq. 11 guess the fan is only weakly
//! magnetised and the HLLC-like Eq. 14 pressure is the better start (FWS26 Sec. II.A).
constexpr Real kWeakFieldCutoff = 1.0e-2;
//! FWS26 Sec. II.A: as B -> 0 the Alfven waves become degenerate with the contact and
//! denominators such as lambda_aL - lambda_aR tend to zero.  The limits are well defined,
//! so a fudge of this size keeps the division finite.  It doubles as the slack in the
//! degenerate half of the wave-ordering test, which the root solve can only satisfy to
//! the same tolerance, and as the threshold below which the fan is treated as degenerate
//! outright.
constexpr Real kFudge = 1.0e-12;
//! Relative offset of the second secant point from the initial guess, and the relative
//! step of the bracketing walk, on an ordinary face.  FWS26 names neither; any offset
//! small against the curvature of f and large against round-off gives the same root
//! where the fan admits a per cent of P_tot either side of it.
constexpr Real kSecantSeedStep = 1.0e-4;
constexpr Real kBracketStep = 0.002;

//! The bracketing search that runs where all three seeds fail.  f(P_tot) is monotone
//! non-increasing on every one of the 25731 declining production faces measured, so the
//! search steps outward from the mean pressure and stops at the first pair of admissible
//! pressures whose residuals differ in sign.
constexpr int kBracketSteps = 8;
constexpr int kBracketIterations = 20;
//! Halvings of the gap between a side's first admissible probe and the inadmissible one
//! before it: a gap of one walk step (at most kBracketStep of P_tot) is down to
//! kPressureTolerance of P_tot after 38.
constexpr int kEdgeBisections = 48;

//! Both searches step in fractions of the window of pressures the fan admits, which is
//! ~ P_tot/(sigma gamma^2) wide: a fixed per-cent step walks over it on a force-free face
//! and finds two admissible probes with opposite residuals that have nothing but
//! inadmissible pressures between them.  kStiffKappa is that fraction, small enough that
//! the bracket the walk returns is itself narrower than the window; the walk then doubles
//! its step until it reaches the ordinary-face step, and takes exactly those extra
//! iterations, so a face with sigma gamma^2 below kStiffKappa/kBracketStep walks the same
//! sequence of trial pressures it always did.  kMinSearchStep keeps a step meaningful
//! against the convergence tolerance where the window has closed below it altogether.
constexpr Real kStiffKappa = 1.0/64.0;
constexpr Real kMinSearchStep = 1.0e-13;
constexpr int kBracketFineSteps = 20;
//! Contractions tried toward the nearer end of the bracket before a trial pressure that
//! the fan cannot be evaluated at is allowed to condemn the whole bracket.
constexpr int kContractionSteps = 6;

//! Intermediate quantities of one f(P_tot) evaluation: the velocities and transverse
//! fields between the fast and Alfven waves (subscript a), the Alfven speeds, the
//! transverse field between the Alfven waves, the two velocities at the contact, and
//! f(P_tot) = lambda_cL - lambda_cR (FWS26 step 7).
//! One side of the fan in the locally flat frame.  The equation of state enters the
//! solve only through the gas pressure and the enthalpy density rho h; the root find
//! itself is EOS-free, so a tabulated or piecewise-polytropic EOS needs nothing more
//! than these two numbers and the fast speeds.
struct LocalState {
  Real d;             // rest-mass density
  Real p;             // gas pressure
  Real wgas;          // rho h = rho + rho eps + p
  Real vx, vy, vz;    // four-velocity relative to the frame's normal observer
  Real by, bz;        // transverse magnetic field
};

struct PressureState {
  Real va_l[3];
  Real va_r[3];
  Real ba_l[2];
  Real ba_r[2];
  Real lambda_al;
  Real lambda_ar;
  Real bc[2];
  Real vc_l[3];
  Real vc_r[3];
  Real residual;
};

KOKKOS_INLINE_FUNCTION
bool Finite(const Real x) {
  return isfinite(x);
}

KOKKOS_INLINE_FUNCTION
bool FiniteState(const MHDCons1D &u) {
  return Finite(u.d) && Finite(u.mx) && Finite(u.my) && Finite(u.mz) &&
         Finite(u.e) && Finite(u.by) && Finite(u.bz);
}

KOKKOS_INLINE_FUNCTION
void LinearCombination(const Real a, const MHDCons1D &u,
                       const Real b, const MHDCons1D &v, MHDCons1D &out) {
  out.d = a*u.d + b*v.d;
  out.mx = a*u.mx + b*v.mx;
  out.my = a*u.my + b*v.my;
  out.mz = a*u.mz + b*v.mz;
  out.e = a*u.e + b*v.e;
  out.by = a*u.by + b*v.by;
  out.bz = a*u.bz + b*v.bz;
}

//! FWS26 step 3: the jump condition across a fast wave, R = lambda U - F.
KOKKOS_INLINE_FUNCTION
void JumpState(const Real lambda, const MHDCons1D &u, const MHDCons1D &f,
               MHDCons1D &r) {
  LinearCombination(lambda, u, -1.0, f, r);
}

//! Divide by lambda_aR - lambda_aL, which vanishes in the unmagnetised limit (FWS26
//! Sec. II.A).  The physical ordering is lambda_aR >= lambda_aL, so the fudge is added,
//! never subtracted: a round-off-negative difference is pushed back to positive rather
//! than away from zero on the wrong side.
KOKKOS_INLINE_FUNCTION
Real FudgedDenominator(const Real x) {
  return x + kFudge;
}

//! Four-magnetic field and Lorentz factor of one side; returns b.b.
KOKKOS_INLINE_FUNCTION
Real FourVectors(const LocalState &s, const Real bx, Real &u0, Real b[4]) {
  u0 = sqrt(1.0 + SQR(s.vx) + SQR(s.vy) + SQR(s.vz));
  b[0] = bx*s.vx + s.by*s.vy + s.bz*s.vz;
  b[1] = (bx + b[0]*s.vx)/u0;
  b[2] = (s.by + b[0]*s.vy)/u0;
  b[3] = (s.bz + b[0]*s.vz)/u0;
  return -SQR(b[0]) + SQR(b[1]) + SQR(b[2]) + SQR(b[3]);
}

//----------------------------------------------------------------------------------------
//! \fn EvaluateFastSide()
//! \brief MUB09 Eqs. 41-43 on one side of the fan: the velocity and transverse field
//! immediately behind that side's fast wave, the sign-carrying eta of Eq. 42, and the
//! vector K whose normal component is the Alfven speed.  `kSign` is -1 on the left and
//! +1 on the right, so the side is a compile-time constant and every array index here is
//! one too.  `omv2`, `gsum` and `wgas` are the three numbers ContactVelocity needs to
//! evaluate Eq. 49 without cancellation; the algebra is in its comment.

template <int kSign>
KOKKOS_INLINE_FUNCTION
bool EvaluateFastSide(const Real p, const Real bx, const Real lam,
                      const MHDCons1D &rs, Real va[3], Real ba[2],
                      Real kv[3], Real &omv2, Real &gsum, Real &wgas) {
  const Real a = rs.mx - lam*rs.e + p*(1.0 - lam*lam);
  const Real g = SQR(rs.by) + SQR(rs.bz);
  const Real c = rs.my*rs.by + rs.mz*rs.bz;
  const Real q = -a - g + SQR(bx)*(1.0 - lam*lam);
  const Real x = bx*(a*lam*bx + c) - (a + g)*(lam*p + rs.e);

  va[0] = (bx*(a*bx + lam*c) - (a + g)*(p + rs.mx))/x;
  va[1] = (q*rs.my + rs.by*(c + bx*(lam*rs.mx - rs.e)))/x;
  va[2] = (q*rs.mz + rs.bz*(c + bx*(lam*rs.mx - rs.e)))/x;
  const Real v2 = SQR(va[0]) + SQR(va[1]) + SQR(va[2]);
  if (!(v2 < 1.0)) return false;

  const Real dlv = lam - va[0];
  ba[0] = (rs.by - bx*va[1])/dlv;
  ba[1] = (rs.bz - bx*va[2])/dlv;
  const Real vdotr = va[0]*rs.mx + va[1]*rs.my + va[2]*rs.mz;
  // w is the total enthalpy density rho h + b^2 of the state behind the fast wave (MUB09
  // Eq. 41), and P_tot = p_gas + b^2/2, so w - P_tot = rho (1 + eps) + b^2/2 whatever the
  // sign of p_gas.  The test rejects only a state whose rho (1 + eps) + b^2/2 is not
  // positive.  It does not test p_gas: a fan with negative gas pressure behind a fast
  // wave is accepted.
  const Real w = p + (rs.e - vdotr)/dlv;
  if (!(w > p)) return false;
  const Real eta = static_cast<Real>(kSign)*copysign(sqrt(w), bx);

  const Real denom = lam*p + rs.e + bx*eta;
  kv[0] = (rs.mx + p + lam*bx*eta)/denom;
  kv[1] = (rs.my + rs.by*eta)/denom;
  kv[2] = (rs.mz + rs.bz*eta)/denom;

  // b.b of this state from its lab field and velocity, so that w - b.b is the gas
  // enthalpy rho h behind the fast wave rather than the total one.
  const Real vdotba = va[0]*bx + va[1]*ba[0] + va[2]*ba[1];
  omv2 = 1.0 - v2;
  wgas = w - ((SQR(bx) + SQR(ba[0]) + SQR(ba[1]))*omv2 + SQR(vdotba));
  gsum = eta + vdotba;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn ContactVelocity()
//! \brief MUB09 Eq. 49: the velocity one side of the fan takes at the contact, from that
//! side's K and the single transverse field between the Alfven waves.
//!
//! Eq. 49 is v_c = K - B_c (1 - K.K)/(eta - K.B_c), and as sigma = b.b/(rho h) grows both
//! of those go to zero together: K.K -> 1 and K.B_c -> eta like 1/sigma.  Evaluated
//! literally the quotient is a ratio of two catastrophic cancellations -- 1 - K.K loses
//! log10(sigma gamma^2) digits, and eta - K.B_c the same -- and on a force-free face it
//! keeps none of the digits that decide whether v_c is subluminal.  Both are rewritten
//! below in terms of the state behind the fast wave, which holds the same information
//! with no subtraction of nearly equal numbers.
//!
//! Eq. 48 is the same relation at the Alfven wave, where the field is B_a and the
//! velocity is the known v_a, so K = v_a + f B_a with f = (1 - K.K)/(eta - K.B_a).
//! Substituting that K back into f and cancelling the f^2 B_a^2 on both sides leaves
//! f (eta + v_a.B_a) = 1 - v_a.v_a, i.e. f = omv2/gsum with gsum = eta + v_a.B_a.  Then
//!   eta - K.B_a = eta - v_a.B_a - f B_a^2 = [eta^2 - (v_a.B_a)^2 - omv2 B_a^2]/gsum
//!               = (w - b.b)/gsum = wgas/gsum,
//! since eta^2 = w (Eq. 42) and b.b = omv2 B_a^2 + (v_a.B_a)^2; and therefore
//!   1 - K.K = f (eta - K.B_a) = omv2 wgas/gsum^2.
//! B_c and B_a share the normal component, so eta - K.B_c = wgas/gsum - K.(B_c - B_a),
//! and the quotient Eq. 49 asks for is, identically,
//!   (1 - K.K)/(eta - K.B_c) = omv2 wgas/(gsum [wgas - gsum K.(B_c - B_a)]).
//! Where the two Alfven waves carry the same field this is exact at any sigma; elsewhere
//! the only cancellation left is the one in wgas.  Against 60-digit arithmetic the
//! contact velocity this returns is 20x to 5000x closer to the truth over sigma = 5e3 to
//! 6e11, and from sigma ~ 7e7 up the literal form's error already exceeds 1 - v_c.v_c.

KOKKOS_INLINE_FUNCTION
bool ContactVelocity(const Real bx, const Real bc0, const Real bc1, const Real ba0,
                     const Real ba1, const Real kv[3], const Real omv2, const Real gsum,
                     const Real wgas, Real vc[3]) {
  const Real kdb = kv[1]*(bc0 - ba0) + kv[2]*(bc1 - ba1);
  const Real factor = omv2*wgas/(gsum*(wgas - gsum*kdb));
  vc[0] = kv[0] - bx*factor;
  vc[1] = kv[1] - bc0*factor;
  vc[2] = kv[2] - bc1*factor;
  const Real v2 = SQR(vc[0]) + SQR(vc[1]) + SQR(vc[2]);
  return v2 < 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn EvaluatePressure()
//! \brief FWS26 steps 5-7 for one trial total pressure: the Alfven speeds and the states
//! either side of the contact, ending with f(P_tot) = lambda_cL - lambda_cR.
//! Returns false where the trial P_tot is not positive and finite, where it has no
//! admissible fan (an intermediate velocity not below light speed, a total enthalpy
//! behind a fast wave not above P_tot), or where the residual is not finite; the caller
//! treats each as a failed root solve.  Algebra as in MUB09 Sec. 3.2, Eqs. 41-49.

KOKKOS_INLINE_FUNCTION
bool EvaluatePressure(const Real p, const Real bx, const Real lambda_l,
                      const Real lambda_r, const MHDCons1D &r_l,
                      const MHDCons1D &r_r, PressureState &s) {
  if (!(p > 0.0) || !Finite(p)) return false;

  Real k_l[3], k_r[3], omv2_l, omv2_r, gsum_l, gsum_r, wgas_l, wgas_r;
  if (!EvaluateFastSide<-1>(p, bx, lambda_l, r_l, s.va_l, s.ba_l, k_l,
                            omv2_l, gsum_l, wgas_l)) {
    return false;
  }
  if (!EvaluateFastSide<1>(p, bx, lambda_r, r_r, s.va_r, s.ba_r, k_r,
                           omv2_r, gsum_r, wgas_r)) {
    return false;
  }

  // The normal component of K is the Alfven speed (MUB09 Eq. 43).
  s.lambda_al = k_l[0];
  s.lambda_ar = k_r[0];
  const Real dk = FudgedDenominator(s.lambda_ar - s.lambda_al);
  s.bc[0] = (s.ba_r[0]*(s.lambda_ar - s.va_r[0]) + bx*s.va_r[1]
            -s.ba_l[0]*(s.lambda_al - s.va_l[0]) - bx*s.va_l[1])/dk;
  s.bc[1] = (s.ba_r[1]*(s.lambda_ar - s.va_r[0]) + bx*s.va_r[2]
            -s.ba_l[1]*(s.lambda_al - s.va_l[0]) - bx*s.va_l[2])/dk;

  if (!ContactVelocity(bx, s.bc[0], s.bc[1], s.ba_l[0], s.ba_l[1], k_l,
                       omv2_l, gsum_l, wgas_l, s.vc_l)) return false;
  if (!ContactVelocity(bx, s.bc[0], s.bc[1], s.ba_r[0], s.ba_r[1], k_r,
                       omv2_r, gsum_r, wgas_r, s.vc_r)) return false;
  s.residual = s.vc_l[0] - s.vc_r[0];
  return Finite(s.residual);
}

//----------------------------------------------------------------------------------------
//! \fn BracketedPressure()
//! \brief The root solve of last resort, reached only where all three seeds of the ladder
//! failed.  A seed fails because EvaluatePressure is a partial function: it is defined
//! only where the fan's intermediate velocities stay subluminal and the enthalpy behind
//! each fast wave exceeds P_tot, and that set narrows as P_tot/(sigma gamma^2) -- a per
//! cent of P_tot at sigma ~ 1, parts in 1e8 of it on a force-free face -- and frequently
//! excludes every seed.  A secant started outside it dies on its first evaluation, and
//! one started just inside it steps out on its second, which is what 99.9 % of the
//! declines on real funnel faces are.
//!
//! So bracket the root before refining it.  Walk outward from the mean pressure, keep the
//! last pressure the fan could be evaluated at on each side, and stop at the first
//! adjacent pair whose residuals differ in sign; f is monotone non-increasing in P_tot,
//! so the side the residual at the mean pressure points to is the only one worth walking.
//! `step` is the walk's first relative step and comes from the face's own stiffness; it
//! doubles until it reaches kBracketStep, and the walk is given exactly those extra
//! iterations, so a face whose step already is kBracketStep walks what it always walked.
//! Then close on the root with a secant confined to that bracket -- it cannot step out,
//! because a step that would is replaced by a bisection -- and stop when the bracket
//! itself is down to the last digits of P_tot.
//!
//! Returns false where the search finds no sign change, which is a fan with no admissible
//! solution rather than one this routine failed to find; HLLE is then the right flux.

KOKKOS_INLINE_FUNCTION
bool BracketedPressure(const Real bx, const Real lambda_l, const Real lambda_r,
                       const MHDCons1D &r_l, const MHDCons1D &r_r, const Real p_mid,
                       const Real step, Real &pstar, PressureState &star) {
  if (!(p_mid > 0.0) || !Finite(p_mid)) return false;

  // One extra iteration per doubling the face's step needs to reach the ordinary one.
  int nfine = 0;
  for (Real dq = step; dq < kBracketStep && nfine < kBracketFineSteps; dq += dq) ++nfine;
  const int nsteps = kBracketSteps + nfine;
  Real dp = step;

  // The two nearest admissible pressures above and below the mean, and the bracket.
  Real p_up = 0.0, f_up = 0.0, p_dn = 0.0, f_dn = 0.0;
  bool have_up = false, have_dn = false;
  Real lo = 0.0, hi = 0.0, f_lo = 0.0, f_hi = 0.0;
  bool bracketed = false;
  Real up = p_mid, dn = p_mid;
  // The probe before `up` and `dn` on each side.
  Real up_prev = p_mid, dn_prev = p_mid;
  for (int k = 0; k <= nsteps && !bracketed; ++k, dp = fmin(dp + dp, kBracketStep)) {
    const Real ratio = 1.0 + dp;
    for (int side = 0; side < 2; ++side) {
      Real p;
      if (k == 0) {
        if (side != 0) break;
        p = p_mid;
      } else if (side == 0) {
        // f is non-increasing, so a positive residual at the mean pressure puts the root
        // above it and a negative one below: only one side is ever worth a step.
        if (have_up && f_up < 0.0) continue;
        up_prev = up;
        up *= ratio;
        p = up;
      } else {
        if (have_dn && f_dn > 0.0) continue;
        dn_prev = dn;
        dn /= ratio;
        p = dn;
      }
      if (!(p > 0.0) || !Finite(p)) continue;
      if (!EvaluatePressure(p, bx, lambda_l, lambda_r, r_l, r_r, star)) continue;
      const Real f = star.residual;
      if (f == 0.0) {
        pstar = p;
        return true;
      }
      if (k == 0) {
        p_up = p;  f_up = f;  have_up = true;
        p_dn = p;  f_dn = f;  have_dn = true;
        continue;
      }
      if (side == 0) {
        if (have_up && f_up*f < 0.0) {
          lo = p_up;  f_lo = f_up;  hi = p;  f_hi = f;  bracketed = true;
        } else if (!have_up && have_dn && f_dn*f < 0.0) {
          lo = p_dn;  f_lo = f_dn;  hi = p;  f_hi = f;  bracketed = true;
        }
        p_up = p;  f_up = f;  have_up = true;
      } else {
        if (have_dn && f_dn*f < 0.0) {
          lo = p;  f_lo = f;  hi = p_dn;  f_hi = f_dn;  bracketed = true;
        } else if (!have_dn && have_up && f_up*f < 0.0) {
          lo = p;  f_lo = f;  hi = p_up;  f_hi = f_up;  bracketed = true;
        }
        p_dn = p;  f_dn = f;  have_dn = true;
      }
      if (bracketed) break;
    }
  }

  // The walk stops a side at its first admissible pressure whose residual points back
  // toward the mean, and where the mean pressure itself is inadmissible that pressure is
  // the side's first admissible probe: the probe before it is inadmissible and the root
  // can lie between the two.  Bisect between them, keeping the admissible end, until a
  // midpoint is admissible with the other sign of the residual or the two ends agree to
  // kPressureTolerance.
  if (!bracketed) {
    const bool from_up = have_up && (p_up > p_mid) && (f_up < 0.0);
    const bool from_dn = have_dn && (p_dn < p_mid) && (f_dn > 0.0);
    if (!from_up && !from_dn) return false;
    Real p_a = from_up ? p_up : p_dn;
    Real f_a = from_up ? f_up : f_dn;
    Real p_b = from_up ? up_prev : dn_prev;
    for (int n = 0; n < kEdgeBisections && !bracketed; ++n) {
      if (fabs(p_a - p_b) <= kPressureTolerance*p_a) return false;
      const Real p = 0.5*(p_a + p_b);
      if (!EvaluatePressure(p, bx, lambda_l, lambda_r, r_l, r_r, star)) {
        p_b = p;
        continue;
      }
      const Real f = star.residual;
      if (f == 0.0) {
        pstar = p;
        return true;
      }
      if ((f > 0.0) == (f_a > 0.0)) {
        p_a = p;
        f_a = f;
        continue;
      }
      if (from_up) {
        lo = p;  f_lo = f;  hi = p_a;  f_hi = f_a;
      } else {
        lo = p_a;  f_lo = f_a;  hi = p;  f_hi = f;
      }
      bracketed = true;
    }
    if (!bracketed) return false;
  }

  // Bracket-confined secant.  The two most recent iterates drive it, the bracket ends
  // catch it, and the loop stops on the width of the bracket rather than on |f|.
  Real p_1 = hi, f_1 = f_hi, p_0 = lo, f_0 = f_lo;
  Real p = 0.5*(lo + hi);
  bool converged = false;
  for (int n = 0; n < kBracketIterations; ++n) {
    if (hi - lo <= kPressureTolerance*hi) {
      converged = true;
      break;
    }
    p = (f_1 != f_0) ? p_1 - f_1*(p_1 - p_0)/(f_1 - f_0) : 0.5*(lo + hi);
    if (!(p > lo) || !(p < hi) || !Finite(p)) p = 0.5*(lo + hi);
    if (!EvaluatePressure(p, bx, lambda_l, lambda_r, r_l, r_r, star)) {
      p = 0.5*(lo + hi);
      if (!EvaluatePressure(p, bx, lambda_l, lambda_r, r_l, r_r, star)) {
        // Both ends of the bracket are admissible and the midpoint is not, so the
        // admissible set inside the bracket is not connected and the midpoint has landed
        // in a hole.  Contract toward the end with the smaller |f| -- the one the root is
        // nearer -- instead of condemning the bracket: on a force-free face the bracket
        // is many holes wide and the root sits in the sliver next to one end.
        const bool to_lo = (fabs(f_lo) <= fabs(f_hi));
        const Real p_bad = p;
        bool reached = false;
        for (int j = 0; j < kContractionSteps && !reached; ++j) {
          const Real theta = 1.0/static_cast<Real>(1 << (j + 1));
          p = to_lo ? lo + theta*(p_bad - lo) : hi - theta*(hi - p_bad);
          reached = (p > lo) && (p < hi) &&
                    EvaluatePressure(p, bx, lambda_l, lambda_r, r_l, r_r, star);
        }
        if (!reached) return false;
      }
    }
    const Real f = star.residual;
    if (f == 0.0) {
      converged = true;
      break;
    }
    p_0 = p_1;  f_0 = f_1;  p_1 = p;  f_1 = f;
    if (f_lo*f < 0.0) { hi = p;  f_hi = f; } else { lo = p;  f_lo = f; }
  }
  if (!converged) return false;
  // `star` must belong to the pressure the caller is handed, and the last evaluation
  // need not have been at it.
  pstar = p;
  return EvaluatePressure(pstar, bx, lambda_l, lambda_r, r_l, r_r, star);
}

//----------------------------------------------------------------------------------------
//! \fn SolveLocalHLLDStates()
//! \brief One face of the relativistic HLLD fan in a locally flat frame, FWS26 steps 2-9.
//! \param wl,wr        the two states, with the gas pressure and enthalpy the EOS gave
//! \param bx           normal magnetic field, common to both sides
//! \param lambda_l,lambda_r  the extremal fast speeds in this frame (FWS26 Eq. 8)
//! \param v_interface  speed at which the interface moves in this frame (0 in SR)
//! \param u_interface,f_interface  the selected state and its flux
//! Returns false where the fan reverts to HLLE: an input state has non-positive density
//! or gas pressure, neither the seed ladder nor the bracketing search after it finds an
//! admissible root, the wave speeds are not well ordered, or the selected state is not
//! finite or has non-positive density.  The intermediate gas pressure is not tested.

KOKKOS_INLINE_FUNCTION
bool SolveLocalHLLDStates(const LocalState &wl, const LocalState &wr, const Real bx,
                          const Real lambda_l, const Real lambda_r,
                          const Real v_interface,
                          MHDCons1D &u_interface, MHDCons1D &f_interface) {
  const Real pgas_l = wl.p;
  const Real pgas_r = wr.p;
  if (!(wl.d > 0.0) || !(wr.d > 0.0) || !(pgas_l > 0.0) || !(pgas_r > 0.0) ||
      !Finite(v_interface)) return false;

  Real u0_l, u0_r, b_l[4], b_r[4];
  const Real bsq_l = FourVectors(wl, bx, u0_l, b_l);
  const Real bsq_r = FourVectors(wr, bx, u0_r, b_r);
  if (!(bsq_l >= 0.0) || !(bsq_r >= 0.0)) return false;
  if (!Finite(lambda_l) || !Finite(lambda_r) || !(lambda_l < lambda_r)) return false;

  const Real wtot_l = wl.wgas + bsq_l;
  const Real wtot_r = wr.wgas + bsq_r;
  const Real ptot_l = pgas_l + 0.5*bsq_l;
  const Real ptot_r = pgas_r + 0.5*bsq_r;
  MHDCons1D u_l{}, u_r{}, f_l{}, f_r{};
  u_l.d = wl.d*u0_l;
  u_l.e = wtot_l*SQR(u0_l) - SQR(b_l[0]) - ptot_l;
  u_l.mx = wtot_l*wl.vx*u0_l - b_l[1]*b_l[0];
  u_l.my = wtot_l*wl.vy*u0_l - b_l[2]*b_l[0];
  u_l.mz = wtot_l*wl.vz*u0_l - b_l[3]*b_l[0];
  u_l.by = b_l[2]*u0_l - b_l[0]*wl.vy;
  u_l.bz = b_l[3]*u0_l - b_l[0]*wl.vz;
  f_l.d = wl.d*wl.vx;
  f_l.e = wtot_l*u0_l*wl.vx - b_l[0]*b_l[1];
  f_l.mx = wtot_l*SQR(wl.vx) - SQR(b_l[1]) + ptot_l;
  f_l.my = wtot_l*wl.vy*wl.vx - b_l[2]*b_l[1];
  f_l.mz = wtot_l*wl.vz*wl.vx - b_l[3]*b_l[1];
  f_l.by = b_l[2]*wl.vx - b_l[1]*wl.vy;
  f_l.bz = b_l[3]*wl.vx - b_l[1]*wl.vz;

  u_r.d = wr.d*u0_r;
  u_r.e = wtot_r*SQR(u0_r) - SQR(b_r[0]) - ptot_r;
  u_r.mx = wtot_r*wr.vx*u0_r - b_r[1]*b_r[0];
  u_r.my = wtot_r*wr.vy*u0_r - b_r[2]*b_r[0];
  u_r.mz = wtot_r*wr.vz*u0_r - b_r[3]*b_r[0];
  u_r.by = b_r[2]*u0_r - b_r[0]*wr.vy;
  u_r.bz = b_r[3]*u0_r - b_r[0]*wr.vz;
  f_r.d = wr.d*wr.vx;
  f_r.e = wtot_r*u0_r*wr.vx - b_r[0]*b_r[1];
  f_r.mx = wtot_r*SQR(wr.vx) - SQR(b_r[1]) + ptot_r;
  f_r.my = wtot_r*wr.vy*wr.vx - b_r[2]*b_r[1];
  f_r.mz = wtot_r*wr.vz*wr.vx - b_r[3]*b_r[1];
  f_r.by = b_r[2]*wr.vx - b_r[1]*wr.vy;
  f_r.bz = b_r[3]*wr.vx - b_r[1]*wr.vz;
  if (!FiniteState(u_l) || !FiniteState(u_r) || !FiniteState(f_l) || !FiniteState(f_r)) {
    return false;
  }

  // FWS26 step 2 / Eq. 9: supersonic flow across the interface takes the upwind flux.
  if (lambda_l >= v_interface) {
    u_interface = u_l;
    f_interface = f_l;
    return true;
  }
  if (lambda_r <= v_interface) {
    u_interface = u_r;
    f_interface = f_r;
    return true;
  }

  // FWS26 step 3.
  MHDCons1D r_l{}, r_r{};
  JumpState(lambda_l, u_l, f_l, r_l);
  JumpState(lambda_r, u_r, f_r, r_r);

  // FWS26 step 4 / Eq. 11: the initial guess is a weighted average of the two total
  // pressures.  This replaces the HLL-state conserved-to-primitive inversion of MUB09,
  // whose cost can exceed that of the whole root solve.
  const Real inv_dl = 1.0/(lambda_r - lambda_l);
  const Real p_eq11 = (lambda_r*ptot_l - lambda_l*ptot_r)*inv_dl;

  // FWS26 Eq. 14: in the weakly magnetised limit the intermediate pressure is close to
  // the HLLC value, which follows from the HLL state and flux alone (Eqs. 6 and 7) with
  // no inversion.  It is the positive root of P^2 + (E - F_Sx) P + (Sx F_E - E F_Sx),
  // written in the form that avoids cancellation for E > F_Sx.
  MHDCons1D u_hll{}, f_hll{};
  LinearCombination(inv_dl, r_r, -inv_dl, r_l, u_hll);
  LinearCombination(lambda_l*inv_dl, r_r, -lambda_r*inv_dl, r_l, f_hll);
  const Real a1 = u_hll.e - f_hll.mx;
  const Real a0 = u_hll.mx*f_hll.e - f_hll.mx*u_hll.e;
  const Real disc = sqrt(fmax(0.0, SQR(a1) - 4.0*a0));
  const Real p_eq14 = (a1 >= 0.0) ? -2.0*a0/(a1 + disc) : 0.5*(-a1 + disc);

  // Eq. 11 is the paper's guess, and Eq. 14 replaces it below the weak-field cutoff.  The
  // two further entries are not in the paper: Eq. 11 is a lambda-weighted difference of
  // pressures, and a fan whose speeds share a sign -- which is every fan the GR shortcut
  // lets through where the interface itself moves fast, i.e. the funnel -- can make it
  // negative.  Rather than give such a face to HLLE untried, fall through to the Eq. 14
  // root and then to the plain mean of the two pressures.  Both are cheap and already at
  // hand, and the ladder only runs where the preceding seed had no root.
  const bool weak_field = (p_eq11 > 0.0) && (SQR(bx)/p_eq11 < kWeakFieldCutoff);
  const Real seed_first = weak_field ? p_eq14 : p_eq11;
  const Real seed_second = weak_field ? p_eq11 : p_eq14;
  const Real seed_third = 0.5*(ptot_l + ptot_r);

  // The window of trial pressures the fan can be evaluated at is ~ P_tot/(sigma gamma^2)
  // wide, so both searches step in that unit rather than in a fixed ratio of P_tot.  The
  // fixed steps are the ceiling, not the floor: an ordinary face gets exactly them, and
  // only a face stiff enough to close the window below them is given anything smaller.
  const Real wgas_min = fmin(wl.wgas, wr.wgas);
  const Real sigma_face = (wgas_min > 0.0) ? fmax(bsq_l, bsq_r)/wgas_min : 0.0;
  const Real stiffness = fmax(kStiffKappa/(1.0 + sigma_face*fmax(SQR(u0_l), SQR(u0_r))),
                              kMinSearchStep);
  const Real secant_step = fmin(kSecantSeedStep, stiffness);
  const Real bracket_step = fmin(kBracketStep, stiffness);

  // FWS26 step 8: secant iteration on f(P_tot), stopping at |f| < 1e-12 or 15 iterations.
  // The ladder is walked through a lambda rather than over an indexed array of seeds:
  // an array subscripted by a loop counter cannot be held in registers, and this fan is
  // limited by the size of its local-memory frame.
  Real pstar = 0.0;
  PressureState star{};
  auto secant_from = [&](const Real seed) -> bool {
    if (!(seed > 0.0) || !Finite(seed)) return false;
    pstar = seed;
    Real p_old = 0.0, res_old = 0.0;
    bool have_old = false;
    for (int n = 0; n <= kMaxIterations; ++n) {
      if (!EvaluatePressure(pstar, bx, lambda_l, lambda_r, r_l, r_r, star)) break;
      if (star.residual == 0.0) return true;
      if (n == kMaxIterations) break;
      Real p_next;
      if (have_old) {
        p_next = pstar - star.residual*(pstar - p_old)/(star.residual - res_old);
      } else {
        p_next = pstar*(1.0 + secant_step);
      }
      if (!(p_next > 0.0) || !Finite(p_next)) break;
      // The state in `star` belongs to `pstar`, so accept that pair, not the step that
      // would have refined it: the two agree to kPressureTolerance and only one of them
      // has a fan built for it.
      if (fabs(p_next - pstar) <= kPressureTolerance*pstar) return true;
      p_old = pstar;
      res_old = star.residual;
      have_old = true;
      pstar = p_next;
    }
    return false;
  };
  if (!secant_from(seed_first) && !secant_from(seed_second) &&
      !secant_from(seed_third) &&
      !BracketedPressure(bx, lambda_l, lambda_r, r_l, r_r, seed_third, bracket_step,
                         pstar, star)) {
    return false;
  }

  // FWS26 step 9: accept only a well-ordered fan.  The two contact estimates agree to
  // the root-solve tolerance, so lambda_c is their mean; the inner pair of inequalities
  // is degenerate (all three speeds coincide) as B^x -> 0 and carries the same slack as
  // the divisions of Sec. II.A.
  const Real lambda_c = 0.5*(star.vc_l[0] + star.vc_r[0]);
  if (!(lambda_l < star.lambda_al) || !(star.lambda_al <= lambda_c + kFudge) ||
      !(lambda_c <= star.lambda_ar + kFudge) || !(star.lambda_ar < lambda_r)) {
    return false;
  }

  // MUB09 Sec. 3.4: for B^x -> 0 the two Alfven waves collapse onto the contact and the
  // three inner speeds agree to round-off.  There is then no interior state between them
  // to build -- the transverse field is discontinuous across the contact, so the single
  // B_c of Eq. 47 does not exist -- and the U_c construction would divide the round-off
  // numerator of lambda_a - lambda_c by a denominator of the same size.  The limit of the
  // fan is simply the Alfven state on the side the contact selects.
  const bool degenerate = (star.lambda_ar - star.lambda_al) <= kFudge;

  // FWS26 steps 5, 6 and 9: build only the state Eq. 10 selects, and its flux from the
  // jump condition Eq. 5 across the wave that separates it from the upwind fast state.
  MHDCons1D u_a{}, f_a{};
  const bool left_of_contact = (star.lambda_al >= v_interface) ||
      (star.lambda_ar > v_interface && lambda_c >= v_interface);
  const Real lambda_fast = left_of_contact ? lambda_l : lambda_r;
  const Real lambda_a = left_of_contact ? star.lambda_al : star.lambda_ar;
  // Selected side-by-value, not by pointer: a pointer chosen at run time forces the
  // whole PressureState and both jump states into the local frame.
  const Real va0 = left_of_contact ? star.va_l[0] : star.va_r[0];
  const Real va1 = left_of_contact ? star.va_l[1] : star.va_r[1];
  const Real va2 = left_of_contact ? star.va_l[2] : star.va_r[2];
  MHDCons1D r_up{};
  r_up.d = left_of_contact ? r_l.d : r_r.d;
  r_up.mx = left_of_contact ? r_l.mx : r_r.mx;
  r_up.my = left_of_contact ? r_l.my : r_r.my;
  r_up.mz = left_of_contact ? r_l.mz : r_r.mz;
  r_up.e = left_of_contact ? r_l.e : r_r.e;
  r_up.by = left_of_contact ? r_l.by : r_r.by;
  r_up.bz = left_of_contact ? r_l.bz : r_r.bz;
  u_a.by = left_of_contact ? star.ba_l[0] : star.ba_r[0];
  u_a.bz = left_of_contact ? star.ba_l[1] : star.ba_r[1];
  const Real vb_a = va0*bx + va1*u_a.by + va2*u_a.bz;
  u_a.d = r_up.d/(lambda_fast - va0);
  u_a.e = (r_up.e + pstar*va0 - vb_a*bx)/(lambda_fast - va0);
  u_a.mx = (u_a.e + pstar)*va0 - vb_a*bx;
  u_a.my = (u_a.e + pstar)*va1 - vb_a*u_a.by;
  u_a.mz = (u_a.e + pstar)*va2 - vb_a*u_a.bz;
  LinearCombination(lambda_fast, u_a, -1.0, r_up, f_a);

  if (degenerate || star.lambda_al >= v_interface || star.lambda_ar <= v_interface) {
    u_interface = u_a;
    f_interface = f_a;
  } else {
    const Real vc[3] = {lambda_c,
                        0.5*(star.vc_l[1] + star.vc_r[1]),
                        0.5*(star.vc_l[2] + star.vc_r[2])};
    MHDCons1D u_c{};
    u_c.by = star.bc[0];
    u_c.bz = star.bc[1];
    const Real vb_c = vc[0]*bx + vc[1]*u_c.by + vc[2]*u_c.bz;
    const Real denom = lambda_a - lambda_c;
    u_c.d = u_a.d*(lambda_a - va0)/denom;
    u_c.e = (lambda_a*u_a.e - u_a.mx + pstar*vc[0] - vb_c*bx)/denom;
    u_c.mx = (u_c.e + pstar)*vc[0] - vb_c*bx;
    u_c.my = (u_c.e + pstar)*vc[1] - vb_c*u_c.by;
    u_c.mz = (u_c.e + pstar)*vc[2] - vb_c*u_c.bz;
    MHDCons1D f_c{};
    LinearCombination(1.0, f_a, lambda_a, u_c, f_c);
    LinearCombination(1.0, f_c, -lambda_a, u_a, f_c);
    u_interface = u_c;
    f_interface = f_c;
  }
  return FiniteState(u_interface) && FiniteState(f_interface) && u_interface.d > 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn SolveLocalHLLD()
//! \brief The ideal-gas entry: builds the two states from (rho, e_int, gamma) and the
//! extremal fast speeds of Eq. 8, then hands them to the general solve above.  This is
//! what the fixed-metric SR and GR solvers use; the dynamical-GR solver builds its states
//! from the PrimitiveSolver EOS instead.

KOKKOS_INLINE_FUNCTION
bool SolveLocalHLLD(const MHDPrim1D &wl, const MHDPrim1D &wr, const Real bx,
                    const Real gamma_l, const Real gamma_r, const Real v_interface,
                    MHDCons1D &u_interface, MHDCons1D &f_interface) {
  if (!(gamma_l > 1.0) || !(gamma_r > 1.0)) return false;
  LocalState l{}, r{};
  l.d = wl.d;  l.p = (gamma_l - 1.0)*wl.e;  l.wgas = wl.d + gamma_l*wl.e;
  l.vx = wl.vx; l.vy = wl.vy; l.vz = wl.vz; l.by = wl.by; l.bz = wl.bz;
  r.d = wr.d;  r.p = (gamma_r - 1.0)*wr.e;  r.wgas = wr.d + gamma_r*wr.e;
  r.vx = wr.vx; r.vy = wr.vy; r.vz = wr.vz; r.by = wr.by; r.bz = wr.bz;
  if (!(l.d > 0.0) || !(r.d > 0.0) || !(l.p > 0.0) || !(r.p > 0.0)) return false;

  Real u0_l, u0_r, b_l[4], b_r[4];
  const Real bsq_l = FourVectors(l, bx, u0_l, b_l);
  const Real bsq_r = FourVectors(r, bx, u0_r, b_r);
  if (!(bsq_l >= 0.0) || !(bsq_r >= 0.0)) return false;

  // Free-function wavespeeds: copying all of EOS_Data (12 Kokkos Views + ~40 scalars)
  // twice per face just to overwrite the single side-local gamma is pure register traffic
  // in the GR kernel.
  Real lp_l, lm_l, lp_r, lm_r;
  SRMHDFastSpeedsGamma(gamma_l, l.d, l.p, l.vx, u0_l, bsq_l, lp_l, lm_l);
  SRMHDFastSpeedsGamma(gamma_r, r.d, r.p, r.vx, u0_r, bsq_r, lp_r, lm_r);
  return SolveLocalHLLDStates(l, r, bx, fmin(lm_l, lm_r), fmax(lp_l, lp_r),
                              v_interface, u_interface, f_interface);
}

}  // namespace hlld
}  // namespace mhd
#endif  // MHD_RSOLVERS_HLLD_LOCAL_HPP_
