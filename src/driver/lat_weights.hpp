#ifndef DRIVER_LAT_WEIGHTS_HPP_
#define DRIVER_LAT_WEIGHTS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_weights.hpp
//! \brief the three pure LAT flux-bookkeeping helpers shared by every physics module.
//!
//! These used to be file-static copies: LATFinalFluxWeight in EIGHT byte-comparable
//! copies (hydro_tasks.cpp, hydro_update.cpp, mhd_tasks.cpp, mhd_lat.cpp,
//! radiation_m1_tasks.cpp, radiation_m1_lat.cpp) and LATFluxFaceCount/LATFluxFaceNeighborIndex in four.  They
//! HAD already drifted -- the two hydro copies carried an rk3 branch the other six did
//! not -- and the failure mode of that drift is a silent 50 % conservation error at every
//! coarse/fine face, so they are hoisted here (review A2/F19).
//!
//! The rk3 branch is kept in the single surviving definition and is unreachable in
//! practice: Driver::Execute refuses time/lat for any integrator that is not rk1 or
//! rk2-equivalent (driver.cpp:673-679), and these weights are consumed only on LAT paths.
//! Keeping it means no former copy loses a case.

#include "athena.hpp"
#include "driver/driver.hpp"

namespace lat {

//----------------------------------------------------------------------------------------
//! \fn Real lat::FinalFluxWeight
//! \brief per-stage weight of the final flux in the window integral.  imex2 shares rk2's
//! EXPLICIT tableau (gam0={1,0.5}, gam1={0,0.5}, beta={1,0.5}), so the weight is 0.5 per
//! stage; the beta[stage-1] fallback would integrate 1.5*dt per window instead of dt, a
//! 50 % over-correction.

inline Real FinalFluxWeight(Driver *pdrive, int stage) {
  if (pdrive->integrator == "rk1") return 1.0;
  if (pdrive->integrator_rk2_equiv) return 0.5;
  if (pdrive->integrator == "rk3") {
    return (stage < 3) ? (1.0/6.0) : (2.0/3.0);
  }
  return pdrive->beta[stage-1];
}

//----------------------------------------------------------------------------------------
//! \fn int lat::FluxFaceCount / lat::FluxFaceNeighborIndex
//! \brief the neighbour slots that carry a face (x1: 0-7, x2: 8-15, x3: 24-31), packed
//! into a contiguous team index.  Pure functions of nnghbr; device callable.

KOKKOS_INLINE_FUNCTION
int FluxFaceCount(const int nnghbr) {
  int count = (nnghbr < 16) ? nnghbr : 16;
  if (nnghbr > 24) count += ((nnghbr - 24) < 8) ? (nnghbr - 24) : 8;
  return count;
}

KOKKOS_INLINE_FUNCTION
int FluxFaceNeighborIndex(const int slot, const int nnghbr) {
  const int first_count = (nnghbr < 16) ? nnghbr : 16;
  return (slot < first_count) ? slot : 24 + (slot - first_count);
}

}  // namespace lat

#endif  // DRIVER_LAT_WEIGHTS_HPP_
