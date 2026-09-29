#ifndef MHD_GRMHD_FACE_FLOORS_HPP_
#define MHD_GRMHD_FACE_FLOORS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grmhd_face_floors.hpp
//! \brief Shared device-safe closure helpers for GRMHD face states.

#include <cfloat>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "reconstruct/thermal_floors.hpp"

namespace mhd {

// KORAL check_floors_mhd() CASE 4 (u2p.c:835-860): reconstructed relative
// four-velocities are scaled at fixed direction until gamma <= GAMMAMAXHD.  This is a
// velocity-only physical ceiling; callers keep density, thermal, and scalar repairs on
// their own opt-in paths.
// Every face closure in this header needs the same metric at the same face centre, so
// the ceiling is expressed in a metric the caller supplies.  The ...AtCartesian spellings
// below build it; a kernel that already holds one (the fused hybrid face) passes it in.
KOKKOS_INLINE_FUNCTION
bool LimitGRMHDFaceRelativeFourVelocityInMetric(
    const Real glower[][4],
    const Real gamma_max, Real &u1, Real &u2, Real &u3) {
  // Use the same inward, overflow-safe ceiling as cell recovery so high-order
  // reconstruction, FOFC, and sink faces all share one strict invariant.
  const auto lorentz_limit = forcefree::ApplyIsotropicLorentzLimit(
      glower, gamma_max, u1, u2, u3);
  if (!lorentz_limit.valid) {
    u1 = 0.0;
    u2 = 0.0;
    u3 = 0.0;
    return true;
  }
  return lorentz_limit.limited;
}

// MHDPrim1D rotates its velocity components so vx is face-normal.  Convert to the
// coordinate ordering required by the spatial metric, apply the KORAL limiter, and rotate
// back without changing the face-normal/tangential Riemann-solver convention.
KOKKOS_INLINE_FUNCTION
bool LimitDirectionalGRMHDFaceStateLorentzInMetric(
    const int direction, const Real glower[][4],
    const Real gamma_max, MHDPrim1D &state) {
  Real u1 = state.vx;
  Real u2 = state.vy;
  Real u3 = state.vz;
  if (direction == 2) {
    u1 = state.vz;
    u2 = state.vx;
    u3 = state.vy;
  } else if (direction == 3) {
    u1 = state.vy;
    u2 = state.vz;
    u3 = state.vx;
  }

  const bool limited = LimitGRMHDFaceRelativeFourVelocityInMetric(
      glower, gamma_max, u1, u2, u3);
  if (!limited) return false;
  if (direction == 1) {
    state.vx = u1;
    state.vy = u2;
    state.vz = u3;
  } else if (direction == 2) {
    state.vx = u2;
    state.vy = u3;
    state.vz = u1;
  } else {
    state.vx = u3;
    state.vy = u1;
    state.vz = u2;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool LimitDirectionalGRMHDFaceStateLorentzAtCartesian(
    const int direction, const CoordData &coord,
    const Real x1v, const Real x2v, const Real x3v,
    const Real gamma_max, MHDPrim1D &state) {
  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski,
                          coord.bh_spin, glower, gupper);
  return LimitDirectionalGRMHDFaceStateLorentzInMetric(
      direction, glower, gamma_max, state);
}

}  // namespace mhd

#endif  // MHD_GRMHD_FACE_FLOORS_HPP_
