#ifndef PGEN_STAR_BH_ORBIT_HPP_
#define PGEN_STAR_BH_ORBIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <array>
#include <cmath>

namespace star_bh_orbit {

// Explicit kick-drift-kick. The endpoint force callback may contain MPI
// collectives, so call only at a rank-synchronized orbit window.
template<typename T, typename Force>
bool LeapfrogKDK(const std::array<T, 3> &x0, const std::array<T, 3> &v0,
                 const std::array<T, 3> &a0, const T dt, Force force,
                 std::array<T, 3> &x1, std::array<T, 3> &v1) {
  std::array<T, 3> vhalf, a1;
  for (int d = 0; d < 3; ++d) {
    vhalf[d] = v0[d] + dt*a0[d]/2;
    x1[d] = x0[d] + dt*vhalf[d];
  }
  if (!force(x1, a1)) return false;
  for (int d = 0; d < 3; ++d) {
    v1[d] = vhalf[d] + dt*a1[d]/2;
    if (!std::isfinite(x1[d]) || !std::isfinite(v1[d])) return false;
  }
  return true;
}

}  // namespace star_bh_orbit
#endif  // PGEN_STAR_BH_ORBIT_HPP_
