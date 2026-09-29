#ifndef EOS_DRIFT_FRAME_FLOOR_HPP_
#define EOS_DRIFT_FRAME_FLOOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drift_frame_floor.hpp
//! \brief Drift-frame mass/energy injection for the GRMHD density and pressure floors,
//! after Ressler, Tchekhovskoy, Quataert, Chandra & Gammie 2017, MNRAS 467, 3604,
//! Appendix B3.
//!
//! A floor adds rest mass and internal energy to a cell.  The question the floor cannot
//! avoid answering is what momentum comes with that mass.  Adding it at rest in the
//! coordinate frame -- which is what "set the velocity to zero when you raise the
//! density" does -- is a momentum sink: in a magnetically dominated funnel the cell sits
//! a few per cent above the floor, the floor fires every step, and the outflow is pinned
//! at rest no matter what the Riemann solver hands it.  Adding it at rest in the fluid
//! frame (leave the velocity alone) is the opposite extreme and lets the parallel
//! momentum per unit mass grow without bound as sigma rises.
//!
//! Ressler et al. inject in the *drift frame*: the frame moving with the E x B velocity,
//! in which the new material is at rest along the field but shares the perpendicular
//! drift of the plasma.  The perpendicular velocity is then untouched by construction and
//! only the field-parallel three-velocity is rescaled, by exactly the amount that
//! conserves the parallel momentum density the floor did not create.
//!
//! The rewriting below is in normal-observer (3+1) variables -- the spatial three-metric
//! gamma_ij, the normal-frame field B^i, and utilde^i = W v^i, which is what the dynamic
//! spacetime PrimitiveSolver stores in prim[PVX..PVZ] and what the fixed-metric GRMHD
//! inversion returns in w.vx..w.vz.  Every quantity that appears is built from gamma_ij
//! contractions of purely spatial vectors, so the lapse and the shift cancel identically:
//! the drift frame is defined by the field and the velocity, not by the slicing, and the
//! four-vector form of Appendix B3 carries alpha and beta^i only through factors that
//! divide out.  That is why this one routine serves both the fixed Kerr-Schild metric
//! (zero shift only at infinity) and the dynamical BSSN spacetime.
//!
//! The map is the identity when the floor added nothing: with w_new == w_old the
//! algebraic identity 1 + x^2 == (1 + 2 W^2 p^2)^2 collapses p_new back to p.  The call
//! is skipped outright in that case, so a cell that was not actually floored is bitwise
//! unchanged.  The map is written for added enthalpy, w_new > w_old, where it only ever
//! lowers the field-parallel speed and W; handed w_new < w_old it would raise both, so
//! the metric-holding entry points below leave the velocity alone unless rho*h rose.
//! Every division is guarded; on any non-finite intermediate the velocity is left exactly
//! as it was.  Leaving it alone is fluid-frame injection, which is the defensible
//! fallback -- zeroing it is not, and never happens here.

#include <math.h>

#include "athena.hpp"
#include "eos/primitive-solver/ps_types.hpp"
#include "eos/primitive-solver/geom_math.hpp"

namespace eos_floor {

//----------------------------------------------------------------------------------------
//! \fn void DriftFrameReinjectFromContractions
//! \brief The Ressler+2017 B3 map, given the three metric contractions it needs.
//!
//! Call sites that already carry a cheap rank-1 form of gamma_ij (the Kerr-Schild null
//! form, say) build the contractions themselves and avoid materializing the metric.
//! Their floors only raise rho and eps, so they hand it w_new >= w_old.
//!
//! \param[in]     Bsq     gamma_ij B^i B^j
//! \param[in]     Bdotu   gamma_ij B^i utilde^j
//! \param[in]     utilde2 gamma_ij utilde^i utilde^j
//! \param[in]     b_u     the normal-frame magnetic field B^i (undensitized)
//! \param[in]     w_old   rho*h before the floor
//! \param[in]     w_new   rho*h after the floor
//! \param[in,out] utilde  W v^i, rescaled along B in place

KOKKOS_INLINE_FUNCTION
void DriftFrameReinjectFromContractions(const Real Bsq, const Real Bdotu,
                                        const Real utilde2, const Real b_u[3],
                                        const Real w_old, const Real w_new,
                                        Real utilde[3]) {
  // Nothing was added, or the enthalpies are unusable: fluid-frame injection.  The
  // equality test is what makes an unfloored cell bitwise unchanged.
  if (!(w_old > 0.0) || !(w_new > 0.0) || !isfinite(w_old) || !isfinite(w_new) ||
      (w_old == w_new)) {
    return;
  }
  // The zero-field limit of the map below: with no field there is no parallel direction,
  // B^i/|B| is 0/0, and the rescaling has nothing to rescale.  The velocity stays.
  if (!(Bsq > 0.0) || !isfinite(Bsq) || !isfinite(Bdotu) ||
      !(utilde2 >= 0.0) || !isfinite(utilde2)) {
    return;
  }

  const Real W = sqrt(1.0 + utilde2);
  const Real Bmag = sqrt(Bsq);
  const Real iBmag = 1.0/Bmag;
  const Real iW = 1.0/W;

  // Field-parallel three-velocity, p = Bhat_i v^i.
  const Real p = (Bdotu*iBmag)*iW;

  // The E x B drift: everything the floor is forbidden to change.
  const Real bh[3] = {b_u[0]*iBmag, b_u[1]*iBmag, b_u[2]*iBmag};
  const Real vperp[3] = {utilde[0]*iW - p*bh[0],
                         utilde[1]*iW - p*bh[1],
                         utilde[2]*iW - p*bh[2]};

  // Lorentz factor of the drift frame: |vperp|^2 = 1 - 1/W^2 - p^2, so
  // 1 - |vperp|^2 = 1/W^2 + p^2 exactly, with no cancellation for either small p or
  // large W.
  const Real idr2 = iW*iW + p*p;
  if (!(idr2 > 0.0) || !isfinite(idr2)) {
    return;
  }
  const Real W_dr = 1.0/sqrt(idr2);

  // Parallel momentum per unit drift-frame enthalpy is what is conserved.  Solving the
  // resulting quadratic for the new parallel velocity gives x/(1 + sqrt(1+x^2)), the
  // cancellation-free branch of (sqrt(1+x^2) - 1)/x.
  const Real x = 2.0*(w_old/w_new)*(W*W/W_dr)*p;
  if (!isfinite(x)) {
    return;
  }
  const Real p_new = (x/(1.0 + sqrt(1.0 + x*x)))/W_dr;

  const Real den = 1.0/(W_dr*W_dr) - p_new*p_new;
  if (!(den > 0.0) || !isfinite(den)) {
    return;
  }
  const Real W_new = 1.0/sqrt(den);
  if (!isfinite(W_new)) {
    return;
  }

  utilde[0] = W_new*(vperp[0] + p_new*bh[0]);
  utilde[1] = W_new*(vperp[1] + p_new*bh[1]);
  utilde[2] = W_new*(vperp[2] + p_new*bh[2]);
}

//----------------------------------------------------------------------------------------
//! \fn void DriftFrameReinjectVelocity
//! \brief The same map for call sites that hold the full spatial three-metric.  The
//! velocity is left alone unless the floors raised rho*h (w_new > w_old).
//!
//! \param[in]     g3d    gamma_ij, in the S11,S12,S13,S22,S23,S33 order of ps_types.hpp
//! \param[in]     b_u    the normal-frame magnetic field B^i (undensitized)
//! \param[in]     w_old  rho*h before the floor
//! \param[in]     w_new  rho*h after the floor
//! \param[in,out] utilde W v^i, rescaled along B in place

KOKKOS_INLINE_FUNCTION
void DriftFrameReinjectVelocity(const Real g3d[NSPMETRIC], const Real b_u[3],
                                const Real w_old, const Real w_new, Real utilde[3]) {
  if (!(w_old > 0.0) || !(w_new > w_old)) {
    return;
  }
  Real B_d[3], u_d[3];
  Primitive::LowerVector(B_d, b_u, g3d);
  Primitive::LowerVector(u_d, utilde, g3d);
  const Real Bsq = Primitive::Contract(b_u, B_d);
  const Real Bdotu = Primitive::Contract(utilde, B_d);
  const Real utilde2 = Primitive::Contract(utilde, u_d);
  DriftFrameReinjectFromContractions(Bsq, Bdotu, utilde2, b_u, w_old, w_new, utilde);
}

//----------------------------------------------------------------------------------------
//! \fn bool DriftFrameApplyPrimitiveFloor
//! \brief The primitive floor plus its velocity answer, for call sites that hold a state
//! and nothing else -- the reconstructed face states the dynamical-GR Riemann solvers
//! floor before they build a flux.
//!
//! A face at the atmosphere density is the extreme end of the same problem the cell
//! floors have: the reconstructed state can be almost all floor, so the enthalpy ratio is
//! large and the map drives the field-parallel velocity to nearly zero while leaving the
//! E x B drift alone.  That is the bounded, physical version of what the old code
//! achieved by zeroing all three components.
//!
//! The face keeps the pressure it was reconstructed with, raised to the atmosphere
//! pressure at the floored density if below.  Its temperature is P/n of two separately
//! reconstructed variables, so where n undershot (to zero, in the limit) it is not a
//! property of the gas, and carrying it to the raised density would scale the face
//! pressure by n_new/n.  The floor then adds rest mass (and heat only up to T_atm).
//!
//! \param[in]     eos  a Primitive::EOS
//! \param[in,out] prim the face state, floored in place
//! \param[in]     b_u  the normal-frame magnetic field B^i at the face
//! \param[in]     g3d  gamma_ij at the face
//! \return whether the floor fired

template <class EOSType>
KOKKOS_INLINE_FUNCTION
bool DriftFrameApplyPrimitiveFloor(const EOSType &eos, Real prim[NPRIM],
                                   const Real b_u[3], const Real g3d[NSPMETRIC]) {
  const Real mb = eos.GetBaryonMass();
  const Real p_face = prim[PPR];
  const Real w_old = prim[PRH]*mb*eos.GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
  const bool floored = eos.ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR],
                                               prim[PTM], &prim[PYF]);
  if (floored) {
    const Real T_atm = eos.GetTemperatureFloor();
    const Real p_atm = eos.GetPressure(prim[PRH], T_atm, &prim[PYF]);
    if (isfinite(p_face) && (p_face > p_atm)) {
      prim[PPR] = p_face;
      prim[PTM] = eos.GetTemperatureFromP(prim[PRH], p_face, &prim[PYF]);
    } else {
      prim[PPR] = p_atm;
      prim[PTM] = T_atm;
    }
    const Real w_new = prim[PRH]*mb*eos.GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
    DriftFrameReinjectVelocity(g3d, b_u, w_old, w_new, &prim[PVX]);
  }
  return floored;
}

} // namespace eos_floor

#endif // EOS_DRIFT_FRAME_FLOOR_HPP_
