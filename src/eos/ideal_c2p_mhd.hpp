#ifndef EOS_IDEAL_C2P_MHD_HPP_
#define EOS_IDEAL_C2P_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for MHD
//! with an ideal gas EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.

#include "coordinates/cartesian_ks.hpp"
#include "eos/drift_frame_floor.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real IdealMHDEintFloor()
//! \brief Returns the larger of the pressure and entropy floors in internal-energy form.

KOKKOS_INLINE_FUNCTION
Real IdealMHDEintFloor(const EOS_Data &eos, const Real dens) {
  const Real gm1 = eos.gamma - 1.0;
  const Real eint_floor = eos.pfloor/gm1;
  const Real entropy_floor = (eos.sfloor > 0.0) ?
      (eos.sfloor*pow(dens, eos.gamma)/gm1) : 0.0;
  return fmax(eint_floor, entropy_floor);
}

KOKKOS_INLINE_FUNCTION
Real IdealMHDAtmosphereInternalEnergy(const EOS_Data &eos) {
  const Real gm1 = eos.gamma - 1.0;
  Real eint = IdealMHDEintFloor(eos, eos.dfloor);
  if ((eos.tfloor > 0.0) && (gm1*eint/eos.dfloor < eos.tfloor)) {
    eint = eos.dfloor*eos.tfloor/gm1;
  }
  const Real eceil = eos.HydroInternalEnergyDensityCeiling(eos.dfloor);
  if (eint > eceil) {
    eint = eceil;
  }
  return eint;
}

KOKKOS_INLINE_FUNCTION
bool NeedsIdealMHDAtmosphereReset(const EOS_Data &eos, const MHDCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetIdealMHDAtmosphereState(const EOS_Data &eos, MHDCons1D &u, HydPrim1D &w,
                                  bool &dfloor_used) {
  const Real atmosphere_eint = IdealMHDAtmosphereInternalEnergy(eos);
  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint + emag;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  dfloor_used = true;
}

//----------------------------------------------------------------------------------------
//! \!fn void SingleC2P_IdealMHD()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Note input CONSERVED state contains cell-centered magnetic
//! fields, but PRIMITIVE state returned through arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealMHD(MHDCons1D &u, const EOS_Data &eos,
                        HydPrim1D &w,
                        bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                        bool &vceil_used) {
  Real efloor = eos.pfloor/(eos.gamma - 1.0);
  Real tfloor = eos.tfloor;
  Real gm1 = eos.gamma - 1.0;

  if (NeedsIdealMHDAtmosphereReset(eos, u)) {
    ResetIdealMHDAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  // set internal energy, apply floor, correcting total energy
  Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  Real e_m = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  w.e = (u.e - e_k - e_m);
  efloor = IdealMHDEintFloor(eos, w.d);
  if (w.e < efloor) {
    w.e = efloor;
    u.e = efloor + e_k + e_m;
    efloor_used = true;
  }
  // apply temperature floor
  if (gm1*w.e*di < tfloor) {
    w.e = w.d*tfloor/gm1;
    u.e = w.e + e_k + e_m;
    tfloor_used =true;
  }
  const Real eceil = eos.HydroInternalEnergyDensityCeiling(w.d);
  if (w.e > eceil) {
    w.e = eceil;
    u.e = w.e + e_k + e_m;
    efloor_used = true;
  }

  // Apply optional velocity ceiling on |v|.
  Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vmag = eos.vceil;
    v2 = vmag*vmag;
    vceil_used = true;
  }

  // Keep conserved variables consistent with any floor-limited primitive state.
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*v2 + e_m;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealMHD()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic MHD with an ideal gas EOS.  Note input PRIMITIVE state contains
//! cell-centered magnetic fields, but CONSERVED state returned via arguments does not.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealMHD(const MHDPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e  = w.e + 0.5*(w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
                        (SQR(w.bx) + SQR(w.by) + SQR(w.bz)) );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation49()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving Equation44

KOKKOS_INLINE_FUNCTION
Real Equation49(const Real mu, const Real b2, const Real rp, const Real r, const Real q) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  Real rbar = (x*x*r*r + mu*x*(1.0 + x)*rp*rp); // (38)
  return mu*sqrt(1.0 + rbar) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real KastaunRbar()
//! \brief Returns rbar(mu), eq. 38 of Kastaun et al., shared by eqs. 44 and 49.

KOKKOS_INLINE_FUNCTION
Real KastaunRbar(const Real mu, const Real b2, const Real rpar, const Real r) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  return (x*x*r*r + mu*x*(1.0 + x)*rpar*rpar);  // (38)
}

//----------------------------------------------------------------------------------------
//! \fn bool MuInsideKastaunBranch()
//! \brief True if mu is at or below the eq. 49 root mu+, i.e. on the branch where the
//! eq. 44 master function is single valued.  This is fa(mu)<=0 written without the
//! square root: mu*sqrt(1+rbar) <= 1 <=> mu^2*(1+rbar) <= 1 for mu >= 0.

KOKKOS_INLINE_FUNCTION
bool MuInsideKastaunBranch(const Real mu, const Real rbar) {
  return (mu*mu*(1.0 + rbar) <= 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real IdealMHDEntropyFloorDensityThreshold()
//! \brief Returns the density above which the entropy floor exceeds the pressure floor,
//! i.e. the density where sfloor*d^gamma == pfloor.  Computed once per cell so that the
//! root-find inner loop can branch on it instead of evaluating a double precision pow()
//! on every iteration.  The threshold is nudged down by a relative 1.0e-12 so that
//! round-off in this expression can only take the fmax() branch early (where fmax()
//! returns the same value the original expression did), never late.

KOKKOS_INLINE_FUNCTION
Real IdealMHDEntropyFloorDensityThreshold(const EOS_Data &eos) {
  if (!(eos.sfloor > 0.0)) {
    return static_cast<Real>(1.0e300);
  }
  return pow(eos.pfloor/eos.sfloor, 1.0/eos.gamma)*(1.0 - 1.0e-12);
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation44()
//! \brief Inline function to compute function f(mu) defined in eq. 44 of Kastaun et al.
//! The ConsToPrim algorithms finds the root of this function f(mu)=0
//! `sfloor_dthresh` is the entropy-floor density threshold returned by
//! IdealMHDEntropyFloorDensityThreshold(), computed once per cell by the caller.

KOKKOS_INLINE_FUNCTION
Real Equation44(const Real mu, const Real b2, const Real rpar, const Real r, const Real q,
                const Real u_d, const EOS_Data &eos, const Real sfloor_dthresh) {
  Real const x = 1./(1.+mu*b2);                    // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);   // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar- rpar*rpar)); // (31)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar))); // (32)
  Real w = sqrt(1.+z2);
  Real const wd = u_d/w;                           // (34)
  Real eps = w*(qbar - mu*rbar) + z2/(w+1.);
  Real const gm1 = eos.gamma - 1.0;
  // The entropy floor can only exceed the pressure floor above wd=(pfloor/sfloor)^(1/g);
  // below that threshold fmax() returns the pressure term anyway, so the returned value
  // is unchanged while a double precision pow() leaves every root-find iteration.
  Real epsmin = eos.pfloor/(wd*gm1);
  if (wd > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(wd, gm1)/gm1);
  }
  eps = fmax(eps, epsmin);
  Real const h = 1.0 + eos.gamma*eps;              // (43)
  return mu - 1./(h/w + rbar*mu);                  // (45)
}

//----------------------------------------------------------------------------------------
//! \fn Real IllinoisRootEquation44()
//! \brief False position ("Illinois") root find of Kastaun eq. 44 on the bracket
//! [zm,zp], with the tolerance and the convergence test of the original inline loop.
//! Returns the root; `iter` returns the number of iterations used, and equals
//! `max_iterations` if the convergence test was never met.

KOKKOS_INLINE_FUNCTION
Real IllinoisRootEquation44(Real zm, Real zp, Real fm, Real fp,
                            const Real b2, const Real rpar, const Real r, const Real q,
                            const Real u_d, const EOS_Data &eos,
                            const Real sfloor_dthresh, const Real tol,
                            const int max_iterations, int &iter) {
  int iterations = max_iterations;
  // If bracket within tolerances, don't bother doing any iterations
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  Real z = 0.5*(zm + zp);

  for (iter=0; iter<iterations; ++iter) {
    z = (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = Equation44(z, b2, rpar, r, q, u_d, eos, sfloor_dthresh);
    // Quit if convergence reached
    // NOTE: both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
      zm = zp;
      fm = fp;
      zp = z;
      fp = f;
    } else {  // assign zp-->z if root bracketed by [zm,z]
      fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
      zp = z;
      fp = f;
    }
  }
  return z;
}

//----------------------------------------------------------------------------------------
//! \fn Real Equation44Adiabat()
//! \brief Kastaun's eq. 44 with the specific internal energy taken from the advected
//! adiabat kappa = p/rho^gamma instead of from the conserved energy.
//!
//! The energy form computes eps = w(qbar - mu*rbar) + z2/(w+1), in which qbar carries
//! -b^2/2: in a magnetically dominated cell both terms are O(b^2/D) and the gas energy
//! is what survives the cancellation, so its relative error grows like b^2/u.  Here eps
//! is a power of the density, so its conditioning does not depend on the field at all.
//! q never enters, which is the same statement as: this branch discards the energy
//! equation.

KOKKOS_INLINE_FUNCTION
Real Equation44Adiabat(const Real mu, const Real b2, const Real rpar, const Real r,
                       const Real kappa, const Real epsmax, const Real u_d,
                       const EOS_Data &eos, const Real sfloor_dthresh) {
  Real const x = 1./(1.+mu*b2);                     // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);    // (38)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));  // (32)
  Real w = sqrt(1.+z2);
  Real const wd = u_d/w;                            // (34)
  Real const gm1 = eos.gamma - 1.0;
  Real eps = kappa*pow(wd, gm1)/gm1;
  Real epsmin = eos.pfloor/(wd*gm1);
  if (wd > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(wd, gm1)/gm1);
  }
  eps = fmax(eps, epsmin);
  // The advected adiabat is not bounded from above by anything the transport does, and
  // a cell whose density reaches its floor turns kappa = (D kappa)/D into a ratio of a
  // finite number to a floor.  The state it then implies has more internal energy than
  // the cell owns, which no fluid state can, so it is capped at the energy budget the
  // conserved variables actually carry.  Without this the pressure feeds the next flux,
  // which feeds the next kappa, and the pair runs away exponentially.
  eps = fmin(eps, epsmax);
  Real const h = 1.0 + eos.gamma*eps;               // (43)
  return mu - 1./(h/w + rbar*mu);                   // (45)
}

//----------------------------------------------------------------------------------------
//! \fn Real IllinoisRootEquation44Adiabat()
//! \brief The same false-position iteration on the adiabat form of eq. 44.  Identical
//! stopping rule and tolerance, so a cell that changes channel does not also change the
//! accuracy of the root it converged to.

KOKKOS_INLINE_FUNCTION
Real IllinoisRootEquation44Adiabat(Real zm, Real zp, Real fm, Real fp,
                                   const Real b2, const Real rpar, const Real r,
                                   const Real kappa, const Real epsmax,
                                   const Real u_d, const EOS_Data &eos,
                                   const Real sfloor_dthresh, const Real tol,
                                   const int max_iterations, int &iter) {
  int iterations = max_iterations;
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  Real z = 0.5*(zm + zp);

  for (iter=0; iter<iterations; ++iter) {
    z = (zm*fp - zp*fm)/(fp-fm);
    Real f = Equation44Adiabat(z, b2, rpar, r, kappa, epsmax, u_d, eos,
                               sfloor_dthresh);
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }
    if (f*fp < 0.0) {
      zm = zp;
      fm = fp;
      zp = z;
      fp = f;
    } else {
      fm = 0.5*fm;
      zp = z;
      fp = f;
    }
  }
  return z;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD_Adiabat()
//! \brief The auxiliary-channel inversion: same conserved momentum and density, pressure
//! from the advected adiabat.
//!
//! Deliberately smaller than the energy-channel solve.  It carries no warm start and no
//! eq. 49 pre-bracket, because it runs only on the cells the eta1 ratio test rejects --
//! a few per cent -- and because mu = 1/(hW) <= 1/h_min <= 1 for an ideal gas makes
//! [0,1] an always-valid bracket, on which f(0) = -1/h(0) < 0.  A root is accepted only
//! if it is still on the single-valued Kastaun branch, exactly as the widened bracket of
//! the energy solve requires; otherwise the caller keeps the energy-channel answer.
//!
//! The velocity, the floors and w_prefloor are computed by the same expressions as the
//! energy solve, so a cell that switches channels does not also change floor behaviour.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD_Adiabat(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2,
                                  Real rpar, const Real kappa, HydPrim1D &w,
                                  bool &dfloor_used, bool &efloor_used,
                                  bool &c2p_failure, int &max_iter,
                                  Real *w_prefloor = nullptr) {
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real gm1 = eos.gamma - 1.0;
  const Real sfloor_dthresh = IdealMHDEntropyFloorDensityThreshold(eos);

  if (!((kappa > 0.0) && isfinite(kappa))) {
    c2p_failure = true;
    return;
  }

  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // Recast as in the energy solve (eq 22-24).  The conserved energy is not read.
  Real r = sqrt(s2)/u.d;
  Real isqrtd = 1.0/sqrt(u.d);
  Real bx = u.bx*isqrtd;
  Real by = u.by*isqrtd;
  Real bz = u.bz*isqrtd;
  b2 /= u.d;
  rpar *= isqrtd;

  // The specific energy the conserved state can pay for.  u.e is tau, the energy
  // without rest mass, so tau/D bounds the specific internal energy from above however
  // large the advected adiabat has become.  A cell handed a non-positive tau -- the
  // state the energy channel has just failed on, and the one this solve exists to
  // rescue -- has no budget to apply and gets no cap: the conserved energy is not
  // rewritten from the auxiliary state, so the next inversion finds it just as
  // wanting and takes the adiabat again, which is the design.
  const Real epsmax = (u.e > 0.0 && isfinite(u.e/u.d)) ? u.e/u.d
                                                       : static_cast<Real>(1.0e300);

  Real zm = 0.0;
  Real zp = 1.0;
  Real fm = Equation44Adiabat(zm, b2, rpar, r, kappa, epsmax, u.d, eos, sfloor_dthresh);
  Real fp = Equation44Adiabat(zp, b2, rpar, r, kappa, epsmax, u.d, eos, sfloor_dthresh);
  if (!(fm*fp < 0.0)) {
    c2p_failure = true;
    return;
  }

  int iter = 0;
  Real z = IllinoisRootEquation44Adiabat(zm, zp, fm, fp, b2, rpar, r, kappa, epsmax,
                                         u.d, eos, sfloor_dthresh, tol, max_iterations,
                                         iter);
  max_iter = (iter > max_iter) ? iter : max_iter;
  if ((iter == max_iterations) ||
      !((z > 0.0) && MuInsideKastaunBranch(z, KastaunRbar(z, b2, rpar, r)))) {
    c2p_failure = true;
    return;
  }

  Real &mu = z;
  Real rbar = KastaunRbar(mu, b2, rpar, r);
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));
  Real lor = sqrt(1.0 + z2);

  Real dens = u.d/lor;
  const Real dens_raw = dens;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  Real eps = fmin(kappa*pow(dens, gm1)/gm1, epsmax);
  Real epsmin = eos.pfloor/(dens*gm1);
  if (dens > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(dens, gm1)/gm1);
  }
  const Real eps_raw = eps;
  if (eps <= epsmin) {
    eps = epsmin;
    efloor_used = true;
  }
  if (w_prefloor != nullptr) {
    *w_prefloor = dens_raw*(1.0 + eos.gamma*eps_raw);
  }

  Real const h = 1.0 + eos.gamma*eps;
  Real const conv = lor/(h*lor + b2);

  w.d  = dens;
  w.vx = conv*(u.mx/u.d + bx*rpar/(h*lor));
  w.vy = conv*(u.my/u.d + by*rpar/(h*lor));
  w.vz = conv*(u.mz/u.d + bz*rpar/(h*lor));
  w.e  = dens*eps;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic MHD with an ideal gas EOS. Note input CONSERVED state contains
//! cell-centered magnetic fields, but PRIMITIVE state returned via arguments does not.
//! `mu_guess` is an optional warm start for the Kastaun root mu; a value that is not
//! strictly inside (0,1) means "no guess" and reproduces the original cold solve.  The
//! converged root is returned through `mu_out` (0 if the inversion failed) so that a
//! caller can cache it and warm start the next inversion of the same cell.  The Kastaun
//! scheme, its 1e-12 tolerance and its convergence test are identical either way: a
//! guess only changes which bracket the unchanged Illinois iteration starts from -- which
//! still moves the accepted root within that tolerance (~1e-12 relative), so a warm
//! started inversion is not bitwise the cold one.

template <bool floor_at_equality = true>
KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                          HydPrim1D &w, bool &dfloor_used, bool &efloor_used,
                          bool &c2p_failure, int &max_iter,
                          const Real mu_guess, Real &mu_out,
                          Real *w_prefloor = nullptr) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real gm1 = eos.gamma - 1.0;
  // Density above which the entropy floor can exceed the pressure floor.  Evaluated once
  // here so that Equation44() does not evaluate a pow() on every iteration.
  const Real sfloor_dthresh = IdealMHDEntropyFloorDensityThreshold(eos);
  mu_out = 0.0;

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // apply energy floor
  if (u.e < (eos.pfloor/gm1 + 0.5*b2)) {
    u.e = eos.pfloor/gm1 + 0.5*b2;
    efloor_used = true;
  }

  // Recast all variables (eq 22-24)
  Real q = u.e/u.d;
  Real r = sqrt(s2)/u.d;
  Real isqrtd = 1.0/sqrt(u.d);
  Real bx = u.bx*isqrtd;
  Real by = u.by*isqrtd;
  Real bz = u.bz*isqrtd;

  // normalize b2 and rpar as well since they contain b
  b2 /= u.d;
  rpar *= isqrtd;

  Real z = 0.0;
  int iter = 0;
  bool solved = false;

  // Warm start.  In a smoothly evolving flow the root mu moves by well under 1% per RK
  // substage, so a cached previous root brackets the new one immediately.  The bracket
  // must satisfy two conditions: f(lo)*f(hi) < 0, and lo,hi <= mu+ (the eq. 49 root) so
  // that both ends lie on the branch where eq. 44 is single valued -- above mu+ the
  // Lorentz factor expression turns over and f can change sign spuriously.  The branch
  // test is exact and needs neither the EOS nor a pow(), so it is the guard here.
  // Since the root of eq. 44 is unique on [0,mu+], any sign-changing bracket inside that
  // interval contains exactly that root, so a stale or wrong guess costs iterations (or
  // falls through to the original path below) and cannot converge to a different root.
  // It does NOT return the same number, though: both paths stop at |f| < tol with
  // |f'| = O(1), so the accepted value is a function of the bracket -- measured over 2e5
  // random states, a warm start moves the converged mu by a median 2e-14 and up to 2e-11
  // relative.  That is the accuracy of the root find, not of the cache, but it does make
  // this inversion depend on call history; see eos.hpp's c2p_mu_cache.
  if ((mu_guess > 0.0) && (mu_guess < 1.0) && isfinite(mu_guess)) {
    const Real delta = 0.05;
    Real d = delta;
    Real lo = mu_guess*(1.0 - d);
    // mu+ is the fixed point of mu = 1/sqrt(1+rbar(mu)).  rbar varies slowly with mu, so
    // one step of that iteration from the guess already lands very close to mu+, and a
    // second step from there brackets it.  This only proposes the upper end; it is the
    // exact branch condition below that decides whether it may be used.
    Real hi = mu_guess*(1.0 + d);
    const Real mup1 = 1.0/sqrt(1.0 + KastaunRbar(mu_guess, b2, rpar, r));
    if (mup1 < hi) {
      const Real mup2 = 1.0/sqrt(1.0 + KastaunRbar(mup1, b2, rpar, r));
      hi = fmin(hi, fmax(mup1, mup2));
    }
    hi = fmin(hi, 1.0);
    bool hi_ok = (hi > lo) &&
                 MuInsideKastaunBranch(hi, KastaunRbar(hi, b2, rpar, r));
    for (int t=0; (t<3) && !hi_ok; ++t) {
      hi = 0.5*(mu_guess + hi);
      hi_ok = (hi > lo) &&
              MuInsideKastaunBranch(hi, KastaunRbar(hi, b2, rpar, r));
    }
    if (hi_ok) {
      Real flo = Equation44(lo, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
      Real fhi = Equation44(hi, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
      // f(mu) < 0 below the root and > 0 above it, so only a positive f(lo) can be
      // repaired here: widening the upper end would leave the physical branch, and that
      // case falls through to the original two-solve path instead.
      for (int t=0; (t<4) && (flo > 0.0) && (fhi > 0.0); ++t) {
        d *= 2.0;
        Real lo_new = mu_guess*(1.0 - d);
        if (!(lo_new > 0.0)) break;
        lo = lo_new;
        flo = Equation44(lo, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
      }
      if (flo*fhi < 0.0) {
        int warm_iter = max_iterations;
        Real warm_z = IllinoisRootEquation44(lo, hi, flo, fhi, b2, rpar, r, q, u.d, eos,
                                             sfloor_dthresh, tol, max_iterations,
                                             warm_iter);
        // A bracketed false position always converges, but never let a warm start that
        // did not meet the convergence test raise the failure counter: fall back to the
        // untouched cold path instead.
        if (warm_iter < max_iterations) {
          z = warm_z;
          iter = warm_iter;
          solved = true;
          max_iter = (iter > max_iter) ? iter : max_iter;
        }
      }
    }
  }

  if (!solved) {
    // Need to find initial bracket. Requires separate solve
    Real zm=0.;
    Real zp=1.; // This is the lowest specific enthalpy admitted by the EOS

    // Evaluate master function (eq 49) at bracket values
    Real fm = Equation49(zm, b2, rpar, r, q);
    Real fp = Equation49(zp, b2, rpar, r, q);

    // For simplicity on the GPU, find roots using the false position method
    int iterations = max_iterations;
    // If bracket within tolerances, don't bother doing any iterations
    if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
      iterations = -1;
    }
    z = 0.5*(zm + zp);

    for (iter=0; iter<iterations; ++iter) {
      z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
      Real f = Equation49(z, b2, rpar, r, q);
      // Quit if convergence reached
      // NOTE(@ermost): both z and f are of order unity
      if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
        break;
      }
      // assign zm-->zp if root bracketed by [z,zp]
      if (f*fp < 0.0) {
        zm = zp;
        fm = fp;
        zp = z;
        fp = f;
      } else {  // assign zp-->z if root bracketed by [zm,z]
        fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
        zp = z;
        fp = f;
      }
    }
    max_iter = (iter > max_iter) ? iter : max_iter;

    // Found brackets. Now find solution in bounded interval, again using the
    // false position method
    zm= 0.;
    zp= z;

    // Evaluate master function (eq 44) at bracket values
    fm = Equation44(zm, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
    fp = Equation44(zp, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
    // Kastaun's construction puts the eq. 44 root inside [0, mu+], so this bracket holds
    // unless the eq. 49 solve was itself degenerate (its early-out can return the
    // interval midpoint rather than a root).  When it does not hold, false position on
    // [0, z49] EXTRAPOLATES, and what it lands on is confined to no interval at all: it
    // can leave the physical range of mu and still pass the |f| < tol test.  Widen to
    // [0, 1] instead -- mu = 1/(h W) <= 1/h_min <= 1 for an ideal gas, so 1 is always a
    // valid upper bound.
    //
    // This differs from the original inline loop, in degenerate cells only.  The widened
    // interval can reach past mu+, where the Lorentz factor expression turns over and
    // eq. 44 changes sign spuriously, so a widened solve is accepted only if its root is
    // still on the single-valued branch (the exact eq. 49 test, no square root) and is
    // reported as an ordinary C2P failure otherwise.  A root off the branch is never
    // returned as converged, which is what the extrapolating original could do.
    const bool widened_bracket = !(fm*fp < 0.0);
    if (widened_bracket) {
      zp = 1.0;
      fp = Equation44(zp, b2, rpar, r, q, u.d, eos, sfloor_dthresh);
    }

    z = IllinoisRootEquation44(zm, zp, fm, fp, b2, rpar, r, q, u.d, eos, sfloor_dthresh,
                               tol, max_iterations, iter);
    if (widened_bracket &&
        !((z > 0.0) && MuInsideKastaunBranch(z, KastaunRbar(z, b2, rpar, r)))) {
      iter = max_iterations;   // hand it to the failure path below
    }
    max_iter = (iter > max_iter) ? iter : max_iter;
  }

  // check if convergence is established within max_iterations.  If not, trigger a C2P
  // failure and return floored density, pressure, and primitive velocities. Future
  // development may trigger averaging of (successfully inverted) neighbors in the event
  // of a C2P failure.
  if (max_iter==max_iterations) {
    w.d = eos.dfloor;
    w.e = eos.pfloor/gm1;
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    c2p_failure = true;
    return;
  }
  mu_out = z;

  // iterations ended, compute primitives from resulting value of z
  Real &mu = z;
  Real const x = 1./(1.+mu*b2);                               // (26)
  Real rbar = (x*x*r*r + mu*x*(1.+x)*rpar*rpar);              // (38)
  Real qbar = q - 0.5*b2 - 0.5*(mu*mu*(b2*rbar - rpar*rpar)); // (31)
  Real z2 = (mu*mu*rbar/(fabs(1.- SQR(mu)*rbar)));            // (32)
  Real lor = sqrt(1.0 + z2);

  // compute density then apply floor
  Real dens = u.d/lor;
  const Real dens_raw = dens;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  // compute specific internal energy density then apply floors
  Real eps = lor*(qbar - mu*rbar) + z2/(lor + 1.0);
  // Same guarded entropy floor as in Equation44(): below the threshold density the
  // pressure term is the larger of the two, so the value is unchanged.
  Real epsmin = eos.pfloor/(dens*gm1);
  if (dens > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(dens, gm1)/gm1);
  }
  const bool below_energy_floor = floor_at_equality ? (eps <= epsmin) : (eps < epsmin);
  const Real eps_raw = eps;
  if (below_energy_floor) {
    eps = epsmin;
    efloor_used = true;
  }
  // rho*h as the inversion recovered it, before the two floors above.  Whatever they
  // added has to be given a velocity, and the caller decides that once, after the
  // magnetization ceiling has had its say too; see eos/drift_frame_floor.hpp.
  if (w_prefloor != nullptr) {
    *w_prefloor = dens_raw*(1.0 + eos.gamma*eps_raw);
  }

  // set parameters required for velocity inversion
  Real const h = 1.0 + eos.gamma*eps;  // (43)
  Real const conv = lor/(h*lor + b2);  // (C26)

  // set primitive variables
  w.d  = dens;
  w.vx = conv*(u.mx/u.d + bx*rpar/(h*lor));  // (C26)
  w.vy = conv*(u.my/u.d + by*rpar/(h*lor));  // (C26)
  w.vz = conv*(u.mz/u.d + bz*rpar/(h*lor));  // (C26)
  w.e  = dens*eps;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRMHD()
//! \brief Overload for call sites that keep no cached Kastaun root.  Identical to the
//! function above called with no guess, i.e. the original cold two-solve path.

template <bool floor_at_equality = true>
KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRMHD(MHDCons1D &u, const EOS_Data &eos, Real s2, Real b2, Real rpar,
                          HydPrim1D &w, bool &dfloor_used, bool &efloor_used,
                          bool &c2p_failure, int &max_iter) {
  Real mu_out = 0.0;
  SingleC2P_IdealSRMHD<floor_at_equality>(u, eos, s2, b2, rpar, w, dfloor_used,
                                          efloor_used, c2p_failure, max_iter,
                                          0.0, mu_out);
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealSRMHD()
//! \brief Converts single set of primitive into conserved variables in SRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealSRMHD(const MHDPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));

  // Calculate 4-magnetic field
  Real b0 = w.bx*w.vx + w.by*w.vy + w.bz*w.vz;
  Real b1 = (w.bx + b0 * w.vx) / u0;
  Real b2 = (w.by + b0 * w.vy) / u0;
  Real b3 = (w.bz + b0 * w.vz) / u0;
  Real b_sq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3);

  // Set conserved quantities
  Real wtot_u02 = (w.d + gam * w.e + b_sq) * u0 * u0;
  u.d  = w.d * u0;
  u.e  = wtot_u02 - b0 * b0 - ((gam-1.0)*w.e + 0.5*b_sq) - u.d;  // In SR, evolve E - D
  u.mx = wtot_u02 * w.vx / u0 - b0 * b1;
  u.my = wtot_u02 * w.vy / u0 - b0 * b2;
  u.mz = wtot_u02 * w.vz / u0 - b0 * b3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool ApplySigmaCeiling_IdealGRMHD()
//! \brief Enforce an optional fixed-metric GRMHD magnetization ceiling sigma = b^2/rho
//! on a recovered primitive state while preserving specific internal energy.

KOKKOS_INLINE_FUNCTION
bool ApplySigmaCeiling_IdealGRMHD(const EOS_Data &eos, const Real glower[][4],
                                  const Real gupper[][4], const MHDCons1D &u,
                                  HydPrim1D &w) {
  if (!(isfinite(eos.sigma_max)) || !(eos.sigma_max > 0.0) || !(w.d > 0.0)) {
    return false;
  }

  Real q = glower[1][1]*SQR(w.vx) + 2.0*glower[1][2]*w.vx*w.vy
         + 2.0*glower[1][3]*w.vx*w.vz + glower[2][2]*SQR(w.vy)
         + 2.0*glower[2][3]*w.vy*w.vz + glower[3][3]*SQR(w.vz);
  if (!isfinite(q)) {
    return false;
  }

  Real alpha = sqrt(-1.0/gupper[0][0]);
  Real lor = sqrt(1.0 + q);
  Real u0 = lor / alpha;
  Real u1 = w.vx - alpha * lor * gupper[0][1];
  Real u2 = w.vy - alpha * lor * gupper[0][2];
  Real u3 = w.vz - alpha * lor * gupper[0][3];

  Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

  Real b0 = u_1*u.bx + u_2*u.by + u_3*u.bz;
  Real b1 = (u.bx + b0*u1) / u0;
  Real b2 = (u.by + b0*u2) / u0;
  Real b3 = (u.bz + b0*u3) / u0;

  Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
  Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
  Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
  Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
  Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
  if (!isfinite(bsq)) {
    return false;
  }

  Real dmin = bsq/eos.sigma_max;
  if (dmin > w.d) {
    Real ratio = dmin/w.d;
    w.d = dmin;
    w.e *= ratio;
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRMHD()
//! \brief Converts single state of conserved variables in GR MHD into conserved
//! variables for special relativistic MHD with an ideal gas EOS. This allows
//! the ConsToPrim() function in GR MHD to use SingleP2C_IdealSRMHD() function.

KOKKOS_INLINE_FUNCTION
void TransformToSRMHD(const MHDCons1D &u, Real glower[][4], Real gupper[][4],
                      Real &s2, Real &b2, Real &rpar, MHDCons1D &u_sr) {
  // Need to multiply the conserved density by alpha, so that it
  // contains a lorentz factor
  Real alpha = sqrt(-1.0/gupper[0][0]);
  u_sr.d = u.d*alpha;

  // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
  // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
  // We are also evolving T^t_t + D as conserved variable, so must convert to E
  u_sr.e = gupper[0][0]*(u.e - u.d) +
           gupper[0][1]*u.mx + gupper[0][2]*u.my + gupper[0][3]*u.mz;

  // This is only true if sqrt{-g}=1!
  u_sr.e *= (-1./gupper[0][0]);  // Multiply by alpha^2

  // Subtract density for consistency with the rest of the algorithm
  u_sr.e -= u_sr.d;

  // Need to treat the conserved momenta. Also they lack an alpha
  // This is only true if sqrt{-g}=1!
  Real m1l = u.mx*alpha;
  Real m2l = u.my*alpha;
  Real m3l = u.mz*alpha;

  // Need to raise indices on u_m1, which transforms using the spatial 3-metric.
  // Store in u_sr.  This is slightly more involved
  //
  // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
  //       g^0i = beta^i/alpha^2
  //       g^00 = -1/ alpha^2
  // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
  u_sr.mx = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.my = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.mz = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  // Compute (S^i S_i) (eqn C2)
  s2 = (m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz);

  // load magnetic fields into SR conserved state. Also they lack an alpha
  // This is only true if sqrt{-g}=1!
  u_sr.bx = alpha*u.bx;
  u_sr.by = alpha*u.by;
  u_sr.bz = alpha*u.bz;

  b2 = glower[1][1]*SQR(u_sr.bx) + glower[2][2]*SQR(u_sr.by) + glower[3][3]*SQR(u_sr.bz) +
       2.0*(u_sr.bx*(glower[1][2]*u_sr.by + glower[1][3]*u_sr.bz) +
                     glower[2][3]*u_sr.by*u_sr.bz);
  rpar = (u_sr.bx*m1l +  u_sr.by*m2l +  u_sr.bz*m3l)/u_sr.d;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn voidSingleP2C_IdealGRMHD()
//! \brief Converts single set of primitive into conserved variables in GRMHD.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRMHD(const Real glower[][4], const Real gupper[][4],
                          const MHDPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate 4-velocity (exploiting symmetry of metric)
  Real q = glower[1][1]*w.vx*w.vx +2.0*glower[1][2]*w.vx*w.vy +2.0*glower[1][3]*w.vx*w.vz
         + glower[2][2]*w.vy*w.vy +2.0*glower[2][3]*w.vy*w.vz
         + glower[3][3]*w.vz*w.vz;
  Real alpha = sqrt(-1.0/gupper[0][0]);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real u1 = w.vx - alpha * gamma * gupper[0][1];
  Real u2 = w.vy - alpha * gamma * gupper[0][2];
  Real u3 = w.vz - alpha * gamma * gupper[0][3];

  // lower vector indices
  Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

  // Calculate 4-magnetic field
  Real b0 = u_1*w.bx + u_2*w.by + u_3*w.bz;
  Real b1 = (w.bx + b0 * u1) / u0;
  Real b2 = (w.by + b0 * u2) / u0;
  Real b3 = (w.bz + b0 * u3) / u0;

  // lower vector indices
  Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
  Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
  Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
  Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
  Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  Real wtot = w.d + gam * w.e + b_sq;
  Real ptot = (gam-1.0)*w.e + 0.5 * b_sq;
  u.d  = w.d * u0;
  u.e  = wtot * u0 * u_0 - b0 * b_0 + ptot + u.d;  // evolve T^t_t + D
  u.mx = wtot * u0 * u_1 - b0 * b_1;
  u.my = wtot * u0 * u_2 - b0 * b_2;
  u.mz = wtot * u0 * u_3 - b0 * b_3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool ApplySigmaCeiling_IdealGRMHD()  [Kerr-Schild null-form overload]
//! \brief Same magnetization ceiling as the glower/gupper overload above, evaluated from
//! the four-number CKS null form (f, l_1, l_2, l_3) instead of two 4x4 matrices.
//! Identities used (all exact rank-1 consequences of g = eta + f l l, and none of them
//! assumes l.l = 1, which the r < 1e-6 floor breaks):
//!   gamma_ij v^i v^j = |v|^2 + f (l.v)^2                        (KSSpatialNormSq)
//!   alpha            = 1/sqrt(1+f)                              (KSAlpha)
//!   g^{0i}           = f l_i                                    (KSGupper0i)
//!   A_mu             = eta_mu_nu A^nu + f l_mu (l.A)            (KSLowerVec)
//!   b.b              = -(b^0)^2 + |b^i|^2 + f (l.b)^2

//----------------------------------------------------------------------------------------
//! \fn void ReinjectFlooredMass_IdealGRMHD()
//! \brief The one drift-frame velocity re-solve that answers every floor a fixed-metric
//! GRMHD cell hit: the deck's uniform density and pressure floors inside the inversion
//! and the magnetization ceiling that runs after it.  The ceiling cannot join the same
//! max as the others -- b^2 is not known until the velocity is -- so the bounds are
//! applied in two places, but the velocity they imply is decided exactly once, here.
//!
//! The metric contractions use the rank-1 Kerr-Schild identity
//! gamma_ij a^i b^j = a.b + f (l.a)(l.b), so gamma_ij is never materialized.

KOKKOS_INLINE_FUNCTION
void ReinjectFlooredMass_IdealGRMHD(const Real gamma, const KSNullForm &nf,
                                    const MHDCons1D &u, const Real w_prefloor,
                                    HydPrim1D &w) {
  if (!(w_prefloor > 0.0) || !isfinite(w_prefloor)) {
    return;
  }
  const Real w_new = w.d + gamma*w.e;
  const Real ldB = KSLowerNull(nf,1)*u.bx + KSLowerNull(nf,2)*u.by +
                   KSLowerNull(nf,3)*u.bz;
  const Real ldv = KSLowerNull(nf,1)*w.vx + KSLowerNull(nf,2)*w.vy +
                   KSLowerNull(nf,3)*w.vz;
  const Real Bsq = SQR(u.bx) + SQR(u.by) + SQR(u.bz) + nf.f*SQR(ldB);
  const Real Bdotu = u.bx*w.vx + u.by*w.vy + u.bz*w.vz + nf.f*ldB*ldv;
  const Real utilde2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz) + nf.f*SQR(ldv);
  const Real b_u[3] = {u.bx, u.by, u.bz};
  Real utilde[3] = {w.vx, w.vy, w.vz};
  eos_floor::DriftFrameReinjectFromContractions(Bsq, Bdotu, utilde2, b_u,
                                                w_prefloor, w_new, utilde);
  w.vx = utilde[0];
  w.vy = utilde[1];
  w.vz = utilde[2];
}

KOKKOS_INLINE_FUNCTION
bool ApplySigmaCeiling_IdealGRMHD(const EOS_Data &eos, const KSNullForm &nf,
                                  const MHDCons1D &u, HydPrim1D &w) {
  if (!(isfinite(eos.sigma_max)) || !(eos.sigma_max > 0.0) || !(w.d > 0.0)) {
    return false;
  }

  Real q = KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz);
  if (!isfinite(q)) {
    return false;
  }

  Real alpha = KSAlpha(nf);
  Real lor = sqrt(1.0 + q);
  Real u0 = lor / alpha;
  // u^i = v^i - alpha*lor*g^{0i}, with g^{0i} = f l_i
  Real alor_f = (alpha*lor)*nf.f;
  Real u1 = w.vx - alor_f*nf.l1;
  Real u2 = w.vy - alor_f*nf.l2;
  Real u3 = w.vz - alor_f*nf.l3;

  Real u_0, u_1, u_2, u_3;
  KSLowerVec(nf, 1, 2, 3, u0, u1, u2, u3, u_0, u_1, u_2, u_3);
  (void) u_0;

  Real b0 = u_1*u.bx + u_2*u.by + u_3*u.bz;
  Real b1 = (u.bx + b0*u1) / u0;
  Real b2 = (u.by + b0*u2) / u0;
  Real b3 = (u.bz + b0*u3) / u0;

  Real ldb = KSLDotVec(nf, 1, 2, 3, b0, b1, b2, b3);
  Real bsq = -SQR(b0) + SQR(b1) + SQR(b2) + SQR(b3) + nf.f*SQR(ldb);
  if (!isfinite(bsq)) {
    return false;
  }

  Real dmin = bsq/eos.sigma_max;
  if (dmin > w.d) {
    Real ratio = dmin/w.d;
    w.d = dmin;
    w.e *= ratio;
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRMHD()  [Kerr-Schild null-form overload]
//! \brief GR->SR conserved transform driven by the null form instead of glower/gupper.
//! Beyond dropping the 32 live doubles, raising the momentum index is done with the
//! closed form
//!     gamma^{ij} = g^{ij} - g^{0i} g^{0j}/g^{00} = delta^{ij} - [f/(1+f)] l_i l_j ,
//! which is algebraically identical to the six explicit differences the matrix version
//! evaluates, but strictly better conditioned: f/(1+f) is bounded in [0,1) for every f,
//! whereas the matrix version subtracts -f l_i l_j and +f^2 l_i l_j/(1+f), two terms that
//! individually diverge like f and cancel to O(1) as f -> infinity near the horizon.

KOKKOS_INLINE_FUNCTION
void TransformToSRMHD(const MHDCons1D &u, const KSNullForm &nf,
                      Real &s2, Real &b2, Real &rpar, MHDCons1D &u_sr) {
  // -g^{00} = 1/alpha^2 = 1 + f
  const Real inv_alpha_sq = KSInvLapseSq(nf);
  const Real alpha = KSAlpha(nf);
  u_sr.d = u.d*alpha;

  // T^{tt} = g^{0 mu} T^t_mu, with g^{00} = -(1+f) and g^{0i} = f l_i
  const Real ldm = nf.l1*u.mx + nf.l2*u.my + nf.l3*u.mz;
  u_sr.e = -inv_alpha_sq*(u.e - u.d) + nf.f*ldm;
  u_sr.e *= (1.0/inv_alpha_sq);  // multiply by alpha^2 (only true if sqrt(-g)=1)
  u_sr.e -= u_sr.d;

  // Conserved momenta lack an alpha (only true if sqrt(-g)=1)
  const Real m1l = u.mx*alpha;
  const Real m2l = u.my*alpha;
  const Real m3l = u.mz*alpha;

  // Raise with gamma^{ij} = delta^{ij} - c l_i l_j,  c = f/(1+f)
  const Real c = KSGammaUpperFactor(nf);
  const Real cldm = c*(nf.l1*m1l + nf.l2*m2l + nf.l3*m3l);
  u_sr.mx = m1l - nf.l1*cldm;    // (C26)
  u_sr.my = m2l - nf.l2*cldm;    // (C26)
  u_sr.mz = m3l - nf.l3*cldm;    // (C26)

  // Compute (S^i S_i) (eqn C2)
  s2 = (m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz);

  // Magnetic fields also lack an alpha (only true if sqrt(-g)=1)
  u_sr.bx = alpha*u.bx;
  u_sr.by = alpha*u.by;
  u_sr.bz = alpha*u.bz;

  // b2 = gamma_ij B^i B^j = |B|^2 + f (l.B)^2
  b2 = KSSpatialNormSq(nf, 1, 2, 3, u_sr.bx, u_sr.by, u_sr.bz);
  rpar = (u_sr.bx*m1l +  u_sr.by*m2l +  u_sr.bz*m3l)/u_sr.d;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealGRMHD()  [Kerr-Schild null-form overload]
//! \brief Primitive -> conserved in GRMHD from the null form.  Index lowering uses
//! A_mu = eta_{mu nu} A^nu + f l_mu (l.A), which is the full 16-multiply glower
//! contraction written as one dot product plus four fused updates.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRMHD(const KSNullForm &nf, const MHDPrim1D &w,
                          const Real gam, HydCons1D &u) {
  // Calculate 4-velocity: q = gamma_ij v^i v^j = |v|^2 + f (l.v)^2
  Real q = KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz);
  Real alpha = KSAlpha(nf);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real ag_f = (alpha*gamma)*nf.f;          // alpha*gamma*g^{0i} = ag_f * l_i
  Real u1 = w.vx - ag_f*nf.l1;
  Real u2 = w.vy - ag_f*nf.l2;
  Real u3 = w.vz - ag_f*nf.l3;

  // lower vector indices
  Real u_0, u_1, u_2, u_3;
  KSLowerVec(nf, 1, 2, 3, u0, u1, u2, u3, u_0, u_1, u_2, u_3);

  // Calculate 4-magnetic field
  Real b0 = u_1*w.bx + u_2*w.by + u_3*w.bz;
  Real b1 = (w.bx + b0 * u1) / u0;
  Real b2 = (w.by + b0 * u2) / u0;
  Real b3 = (w.bz + b0 * u3) / u0;

  // lower vector indices
  Real b_0, b_1, b_2, b_3;
  KSLowerVec(nf, 1, 2, 3, b0, b1, b2, b3, b_0, b_1, b_2, b_3);
  Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  Real wtot = w.d + gam * w.e + b_sq;
  Real ptot = (gam-1.0)*w.e + 0.5 * b_sq;
  u.d  = w.d * u0;
  u.e  = wtot * u0 * u_0 - b0 * b_0 + ptot + u.d;  // evolve T^t_t + D
  u.mx = wtot * u0 * u_1 - b0 * b_1;
  u.my = wtot * u0 * u_2 - b0 * b_2;
  u.mz = wtot * u0 * u_3 - b0 * b_3;
  return;
}

#endif // EOS_IDEAL_C2P_MHD_HPP_
