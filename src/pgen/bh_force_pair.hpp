#ifndef PGEN_BH_FORCE_PAIR_HPP_
#define PGEN_BH_FORCE_PAIR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include "athena.hpp"

namespace bh_force_pair {

struct Force {
  Real ax = 0.0, ay = 0.0, az = 0.0, phi = 0.0;
};

// Preserve the existing potential expression, including its singular limit
// when both physical softening and sink radius are zero. Do not introduce an
// undocumented numerical softening through a denominator floor.
KOKKOS_INLINE_FUNCTION
Real Potential(const Real x, const Real y, const Real z,
                const Real bhx, const Real bhy, const Real bhz,
                const Real mass, const Real softening, const Real newton_g,
                const Real sink_radius = 0.0) {
  const Real dx = x - bhx, dy = y - bhy, dz = z - bhz;
  Real r2 = dx*dx + dy*dy + dz*dz;
  if (sink_radius > 0.0) r2 = fmax(r2, sink_radius*sink_radius);
  return -newton_g*mass/sqrt(r2 + softening*softening);
}

// Analytic derivative, NOT the finite-difference force applied by Gravity().
// A radius-clipped potential is constant strictly inside the clipping sphere.
KOKKOS_INLINE_FUNCTION
Force Evaluate(const Real x, const Real y, const Real z,
                                           const Real bhx, const Real bhy, const Real bhz,
                                           const Real mass, const Real softening,
                                           const Real newton_g,
                                           const Real sink_radius = 0.0) {
  const Real dx = x - bhx, dy = y - bhy, dz = z - bhz;
  const Real r2 = dx*dx + dy*dy + dz*dz;
  Force result;
  result.phi = Potential(x, y, z, bhx, bhy, bhz, mass, softening, newton_g,
                         sink_radius);
  if (sink_radius > 0.0 && r2 <= sink_radius*sink_radius) return result;
  const Real inverse = 1.0/sqrt(r2 + softening*softening);
  const Real factor = -newton_g*mass*inverse*inverse*inverse;
  result.ax = factor*dx;
  result.ay = factor*dy;
  result.az = factor*dz;
  return result;
}

struct AxisStencil {
  Real dpl, dpr;
  KOKKOS_INLINE_FUNCTION
  Real Acceleration(const Real spacing) const { return (dpl + dpr)/(2.0*spacing); }
  KOKKOS_INLINE_FUNCTION
  Real WorkRate(const Real left_mass_flux, const Real right_mass_flux,
                const Real spacing) const {
    return (left_mass_flux*dpl + right_mass_flux*dpr)/(2.0*spacing);
  }
};

// Coordinates of the actual adjacent cell centers come from CellCenterX in the
// caller. This retains the source's grid geometry and floating-point ordering.
KOKKOS_INLINE_FUNCTION
AxisStencil Stencil(const int axis, const Real x, const Real y, const Real z,
                     const Real left, const Real right,
                     const Real bhx, const Real bhy, const Real bhz,
                     const Real mass, const Real softening, const Real newton_g,
                     const Real sink_radius = 0.0) {
  const Real center = Potential(x, y, z, bhx, bhy, bhz, mass, softening,
                                newton_g, sink_radius);
  const Real pl = Potential(axis == 0 ? left : x, axis == 1 ? left : y,
                            axis == 2 ? left : z, bhx, bhy, bhz, mass, softening,
                            newton_g, sink_radius);
  const Real pr = Potential(axis == 0 ? right : x, axis == 1 ? right : y,
                            axis == 2 ? right : z, bhx, bhy, bhz, mass, softening,
                            newton_g, sink_radius);
  return {-(center - pl), -(pr - center)};
}

}  // namespace bh_force_pair

#endif  // PGEN_BH_FORCE_PAIR_HPP_
