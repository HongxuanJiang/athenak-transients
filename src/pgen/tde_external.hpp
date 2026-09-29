#ifndef PGEN_TDE_EXTERNAL_HPP_
#define PGEN_TDE_EXTERNAL_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_external.hpp
//! \brief Shared helpers for the TDE external-potential problem.

#include "athena.hpp"

class ParameterInput;

namespace tde_external {

KOKKOS_INLINE_FUNCTION
bool InsideExcisionZone(const Real x, const Real y, const Real z,
                        const Real bhx, const Real bhy, const Real bhz,
                        const Real radius2) {
  Real dx = x - bhx;
  Real dy = y - bhy;
  Real dz = z - bhz;
  return (dx*dx + dy*dy + dz*dz) <= radius2;
}

void StoreRuntimeMetadata(ParameterInput *pin);

}  // namespace tde_external

#endif  // PGEN_TDE_EXTERNAL_HPP_
