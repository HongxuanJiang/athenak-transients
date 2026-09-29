#ifndef UTILS_LAT_REFLUX_LIMITER_HPP_
#define UTILS_LAT_REFLUX_LIMITER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_reflux_limiter.hpp
//! \brief Admissibility limiter for the LAT window-end conservative reflux.
//!
//! The window-end reflux replaces a coarse block's own face flux by the time-summed
//! flux its finer neighbor actually used.  The two can differ by far more than the
//! coarse cell holds when the atmosphere floor reseeded that cell partway through the
//! window: the floor invented mass the flux history never carried, so subtracting the
//! true fine-side transport drives the corrected state negative.  This picks the
//! largest fraction of the pending mismatch the cell can take and still invert.
//!
//! The fraction is a single scalar per face cell, shared by every conserved variable of
//! that cell, so the correction is scaled as a whole state and never variable by
//! variable.  Whatever the cell declines is handed to the neighbor behind it, which is
//! an ordinary interior cell of the same block, and the same test is applied there: the
//! pair's sum is the unlimited correction whenever that second cell can take its share,
//! which is the case the limiter was written for.  When it cannot either, the residue is
//! dropped, because an inadmissible state has no inversion at all and a conservative
//! reflux that cannot be inverted is worth less than a small accounting loss.  A cell
//! that can take the whole mismatch returns exactly 1.0 and is left bit-for-bit alone.
//!
//! Both fractions are recorded, because the force-free sidecar refluxes its own five
//! tails across the same face and must scale them by the fraction the conserved state of
//! THAT cell took.  A tail and the conserved state it describes are one physical state:
//! the cold closure divides the transported EM momentum by the same cell's |B|^2, so
//! moving different fractions of one face mismatch into the two systems can put the pair
//! outside the drift cone even though both inputs were inside it.

#include <math.h>

#include "athena.hpp"
#include "eos/eos.hpp"

namespace lat_reflux {

//----------------------------------------------------------------------------------------
//! Components of the gravitational-work companion of the pending reflux
//! (Hydro::lat_grav_reflux, MHD::lat_grav_reflux), one register per coarse/fine face
//! cell.  The source term in SourceTerms::Gravity gates the work of each stage's mass
//! flux on that stage's primitive density, while the window-end refund sees only the
//! summed mismatch, so the stage that took the flux has to record how much of it the
//! gate refused.  Storing the refused part rather than the admitted part keeps a face
//! whose gate never closed at exactly zero, so the refund reads the mass accumulator
//! unchanged.

enum GravWorkComponent {
  kGravWorkSeen = 0,         // mass mismatch the gate has already been asked about
  kGravWorkBlockedSelf = 1,  // part of it the self-gravity gate refused
  kGravWorkBlockedBH = 2,    // part of it the external-BH gate refused
  kGravWorkApplied = 3,      // work the window-end refund paid, for the limiter to scale
  kNGravWork = 4
};

//----------------------------------------------------------------------------------------
//! \enum EnergyForm
//! \brief What the energy slot IEN of the conserved state holds on each backend.
//!
//!   kNewtonian  total energy density (kinetic + magnetic + internal)
//!   kTauFlat    SR tau = E - D, with the flat-space inversion bound E >= |S|
//!   kTauCurved  dynamical-GR tau = sqrt(gamma)(E - D), non-negative for every state the
//!               inversion accepts
//!   kTttPlusD   fixed-metric GR T^t_t + D (ideal_c2p_hyd.hpp, ideal_c2p_mhd.hpp), the
//!               rest mass less the energy at infinity: negative for gas at rest in flat
//!               space and of either sign once the gas is bound, so no sign test on it
//!               is an inversion bound and only D carries a test

enum class EnergyForm { kNewtonian, kTauFlat, kTauCurved, kTttPlusD };

//! \fn EnergyForm EnergyFormOf
//! \brief The IEN convention selected by the coordinate flags.  The dynamical backend
//! clears both the SR and the GR flag (coordinates.cpp), so it is tested first.
inline EnergyForm EnergyFormOf(const bool dynamical_rel, const bool general_rel,
                               const bool special_rel) {
  if (dynamical_rel) return EnergyForm::kTauCurved;
  if (general_rel) return EnergyForm::kTttPlusD;
  if (special_rel) return EnergyForm::kTauFlat;
  return EnergyForm::kNewtonian;
}

//----------------------------------------------------------------------------------------
//! \fn bool Admissible
//! \brief Whether one conserved state can be inverted without tripping a floor.
//!
//! Newtonian gas: rest-mass density above the density floor and, with an energy
//! variable, internal energy above the pressure floor once kinetic and magnetic energy
//! are removed.  Relativistic gas: conserved density above the density floor, and where
//! the energy slot holds tau, non-negative tau, with the flat-space inversion bound
//! added under SR.  Every test uses the whole corrected state.

KOKKOS_INLINE_FUNCTION
bool Admissible(const EOS_Data &eos, const EnergyForm form, const Real cons[],
                const int nvar, const Real emag) {
  const Real dens = cons[IDN];
  if (!(dens >= eos.dfloor)) return false;
  if (!eos.is_ideal) return true;
  if (nvar <= IEN) return true;
  if (form == EnergyForm::kTttPlusD) return true;
  const Real etot = cons[IEN];
  if (form != EnergyForm::kNewtonian) {
    // cons[IEN] is tau, the conserved energy less the conserved density.  A negative
    // tau has no primitive state behind it under any of the inversions in this code.
    if (!(etot >= 0.0)) return false;
    if (form == EnergyForm::kTauFlat) {
      const Real ssq = cons[IM1]*cons[IM1] + cons[IM2]*cons[IM2] + cons[IM3]*cons[IM3];
      if (!(etot + dens >= sqrt(dens*dens + ssq))) return false;
    }
    return true;
  }
  const Real ekin = 0.5*(cons[IM1]*cons[IM1] + cons[IM2]*cons[IM2] +
                         cons[IM3]*cons[IM3])/dens;
  const Real eint = etot - ekin - emag;
  const Real gm1 = eos.gamma - 1.0;
  const Real eint_floor = (gm1 > 0.0) ? eos.pfloor/gm1 : eos.pfloor;
  return (eint >= eint_floor);
}

//----------------------------------------------------------------------------------------
//! \fn Real AdmissibleFraction
//! \brief Largest fraction of a pending conserved correction the cell can absorb.
//!
//! Returns exactly 1.0 whenever the unlimited correction is already admissible, so a
//! run that never reseeds a coarse cell next to a fine neighbor is untouched.  When it
//! is not, the admissible set is bracketed from the uncorrected state, which the last
//! conserved-to-primitive pass left admissible, and bisected.  A state that was already
//! inadmissible before the correction is past this limiter's reach, and takes the
//! unlimited correction so the delayed flux stays exact there.

KOKKOS_INLINE_FUNCTION
Real AdmissibleFraction(const EOS_Data &eos, const EnergyForm form, const Real cons[],
                        const Real dcons[], const int nvar, const Real emag) {
  const int ntest = (nvar > IEN) ? (IEN + 1) : (IM3 + 1);
  Real trial[IEN + 1];
  for (int n = 0; n < ntest; ++n) trial[n] = cons[n] + dcons[n];
  if (Admissible(eos, form, trial, nvar, emag)) return 1.0;
  if (!Admissible(eos, form, cons, nvar, emag)) return 1.0;

  Real lo = 0.0, hi = 1.0;
  for (int it = 0; it < 50; ++it) {
    const Real mid = 0.5*(lo + hi);
    for (int n = 0; n < ntest; ++n) trial[n] = cons[n] + mid*dcons[n];
    if (Admissible(eos, form, trial, nvar, emag)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

} // namespace lat_reflux

#endif // UTILS_LAT_REFLUX_LIMITER_HPP_
