#ifndef UTILS_GRAVITY_WEIGHT_HPP_
#define UTILS_GRAVITY_WEIGHT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gravity_weight.hpp
//! \brief Continuous coupling weight that replaces the hard gravity density threshold.
//!
//! gravity/rho_grav_min (and gravity/rho_external_bh_min) used to be a step: a cell below
//! it neither sourced the Poisson equation nor felt the gravitational momentum and work
//! source, a cell above it was fully coupled, and nothing in between.  The two-sided
//! exclusion is deliberate and is kept, but the step is not: because a cell can cross the
//! threshold inside a step, the per-stage source and the LAT window-end work refund could
//! disagree about whether gravity had applied at all, and repairing that by freezing the
//! boolean on the start-of-step register only trades the disagreement for a delay of one
//! whole block step.  A continuous weight removes the discontinuity instead, so there is
//! nothing left to disagree about: whatever weight the source used, the refund uses.
//!
//! The ramp runs in log density between the two EXISTING parameters -- the hydro density
//! floor at w = 0 and the threshold at w = 1 -- with the standard C1 cubic smoothstep
//! 3t^2 - 2t^3, so no input key is added.  When the threshold sits at or below the floor
//! the interval is empty and the weight IS the step it generalises: exactly 1.0 at or
//! above the threshold and 0.0 below it, so every consumer reproduces the old cut bit for
//! bit there.  Every consumer of a cut must read this weight -- the Poisson source, the
//! gas momentum and work source, the BH reaction and its impulse ledger, the translating
//! frame, the external-BH timestep limit, and the energy ledger -- or the momentum pair
//! and the ledger stop describing the coupling that is applied.
//!
//! What no weight can give is a conserved energy.  The gas feels -w(rho) grad(phi) per
//! unit mass while phi is sourced by S = w rho (or is Phi_BH).  An energy U[rho] exists
//! only if that acceleration is a gradient, grad(dU/drho) = w grad(phi), whose curl
//! grad(w) x grad(phi) vanishes only when w' = 0.  So in the continuum
//!   d/dt [E_gas + 1/2 int S phi] = -int phi rho^2 w'(rho) div(v) dV
//! for self-gravity (and the same with 1/2 -> 1, phi -> Phi_BH for the BH), nonzero
//! wherever ramp gas is compressed: it does not converge away with dt or dx.  Freezing w
//! over a step only moves that term to the step boundary, where 1/2 int (dw) rho phi
//! appears at once; the variational force -rho grad(S'(rho) phi) conserves energy but
//! adds -rho phi grad(S'), a force up the density gradient of size |phi|/L_ramp that
//! dwarfs gravity in a deep well.  The old step is no exception: its w' is a delta
//! function at the threshold.  The ledger therefore closes only up to this term.

#include <math.h>

#include "athena.hpp"

namespace gravity_weight {

//----------------------------------------------------------------------------------------
//! \fn Real Weight
//! \brief Gravitational coupling weight in [0,1] for a cell of density rho.

KOKKOS_INLINE_FUNCTION
Real Weight(const Real rho, const Real rho_floor, const Real rho_min) {
  if (rho >= rho_min) return 1.0;
  // Empty ramp (threshold at or below the floor) or no positive floor to ramp from in
  // log density: the two-valued cut this weight generalises.
  if (!(rho_min > rho_floor) || !(rho_floor > 0.0)) return 0.0;
  if (!(rho > rho_floor)) return 0.0;
  const Real t = log(rho/rho_floor)/log(rho_min/rho_floor);
  return t*t*(3.0 - 2.0*t);
}

}  // namespace gravity_weight
#endif  // UTILS_GRAVITY_WEIGHT_HPP_
