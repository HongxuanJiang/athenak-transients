#ifndef MHD_HYBRID_FORCEFREE_ALGEBRA_HPP_
#define MHD_HYBRID_FORCEFREE_ALGEBRA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hybrid_forcefree_algebra.hpp
//! \brief Device-safe KORAL cold-parallel force-free algebra.

#include <math.h>

#include <cfloat>
#include <limits>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "eos/grmhd_magnetization.hpp"

namespace mhd {
namespace forcefree {

#if SINGLE_PRECISION_ENABLED
constexpr Real kForceFreeMachineEpsilon = FLT_EPSILON;
#else
constexpr Real kForceFreeMachineEpsilon = DBL_EPSILON;
#endif

struct IsotropicLorentzLimitResult {
  Real lorentz_factor = 0.0;
  bool valid = false;
  bool limited = false;
};

// Apply KORAL's final GAMMAMAXHD operation to a spatial relative four-velocity.
// Evaluate its metric norm after scaling by the largest component, so neither a large
// cold branch nor a large pre-cap hybrid blend can overflow while forming u_i u^i.
// When a limit is needed, construct the bounded vector directly from the normalized
// direction instead of multiplying by a potentially underflowed scale factor.
KOKKOS_INLINE_FUNCTION
IsotropicLorentzLimitResult ApplyIsotropicLorentzLimit(
    const Real glower[][4], const Real gamma_max,
    Real &u1, Real &u2, Real &u3) {
  IsotropicLorentzLimitResult result;
  if (!isfinite(gamma_max) || !(gamma_max > 1.0) ||
      !isfinite(u1) || !isfinite(u2) || !isfinite(u3)) {
    return result;
  }

  const Real component_scale = fmax(fabs(u1), fmax(fabs(u2), fabs(u3)));
  if (component_scale == 0.0) {
    result.lorentz_factor = 1.0;
    result.valid = true;
    return result;
  }
  if (!isfinite(component_scale) || !(component_scale > 0.0)) return result;

  const Real n1 = u1/component_scale;
  const Real n2 = u2/component_scale;
  const Real n3 = u3/component_scale;
  const Real normalized_norm_sq =
      glower[1][1]*n1*n1 + glower[2][2]*n2*n2 + glower[3][3]*n3*n3 +
      2.0*glower[1][2]*n1*n2 + 2.0*glower[1][3]*n1*n3 +
      2.0*glower[2][3]*n2*n3;
  if (!isfinite(normalized_norm_sq) || !(normalized_norm_sq > 0.0)) return result;
  const Real normalized_norm = sqrt(normalized_norm_sq);
  if (!isfinite(normalized_norm) || !(normalized_norm > 0.0)) return result;

  // gamma*sqrt(1 - gamma^-2) is algebraically sqrt(gamma^2 - 1), but it
  // remains finite for every finite representable gamma_max.  Place the stored
  // state slightly inside that physical sphere.  The smaller comparison band
  // then makes a limited vector a fixed point without admitting any one-sided
  // roundoff excursion above gamma_max.
  const Real inverse_gamma = 1.0/gamma_max;
  const Real physical_maximum_spatial_norm = gamma_max*sqrt(
      (1.0 - inverse_gamma)*(1.0 + inverse_gamma));
  constexpr Real inward_guard_ulps = static_cast<Real>(256.0);
  constexpr Real comparison_ulps = static_cast<Real>(64.0);
  const Real inward_guard = inward_guard_ulps*kForceFreeMachineEpsilon;
  if (!isfinite(physical_maximum_spatial_norm) ||
      !(physical_maximum_spatial_norm > 0.0) || !(inward_guard < 1.0)) {
    return result;
  }
  const Real maximum_spatial_norm =
      physical_maximum_spatial_norm*(1.0 - inward_guard);
  if (!isfinite(maximum_spatial_norm) || !(maximum_spatial_norm > 0.0)) {
    return result;
  }
  const Real maximum_component_scale = maximum_spatial_norm/normalized_norm;
  // An infinite component threshold is a valid "no representable vector is limited"
  // result when gamma_max is near the Real maximum and normalized_norm < 1.
  if (isnan(maximum_component_scale) || !(maximum_component_scale > 0.0)) {
    return result;
  }

  // Reapplying the limiter to its own output can reconstruct the component threshold a
  // few ulps below the stored vector.  Treat that closed-set roundoff neighborhood as
  // already admissible so FOFC does not repeatedly flag an exactly capped state.
  const Real comparison_tolerance = comparison_ulps*
      kForceFreeMachineEpsilon*fmax(component_scale, maximum_component_scale);
  if (component_scale - maximum_component_scale > comparison_tolerance) {
    u1 = n1*maximum_component_scale;
    u2 = n2*maximum_component_scale;
    u3 = n3*maximum_component_scale;
    if (!isfinite(u1) || !isfinite(u2) || !isfinite(u3)) return result;
    result.lorentz_factor = gamma_max;
    result.valid = true;
    result.limited = true;
    return result;
  }

  const Real spatial_norm = component_scale*normalized_norm;
  if (!isfinite(spatial_norm)) return result;
  result.lorentz_factor = Kokkos::hypot(static_cast<Real>(1.0), spatial_norm);
  if (!isfinite(result.lorentz_factor) || result.lorentz_factor < 1.0) {
    return result;
  }
  // The inward guard should already keep the comparison-tolerance neighborhood below
  // gamma_max.  Retain an explicit postcondition so a platform-specific contraction or
  // hypot rounding can never publish a finite value above the authoritative ceiling.
  if (result.lorentz_factor > gamma_max) {
    u1 = n1*maximum_component_scale;
    u2 = n2*maximum_component_scale;
    u3 = n3*maximum_component_scale;
    if (!isfinite(u1) || !isfinite(u2) || !isfinite(u3)) {
      return IsotropicLorentzLimitResult{};
    }
    result.lorentz_factor = gamma_max;
    result.valid = true;
    result.limited = true;
    return result;
  }
  result.valid = true;
  return result;
}

}  // namespace forcefree
}  // namespace mhd

#endif  // MHD_HYBRID_FORCEFREE_ALGEBRA_HPP_
