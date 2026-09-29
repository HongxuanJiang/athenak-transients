//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fofc_ideal_llf.cpp
//! \brief Explicit instantiation of the dynamical-GRMHD FOFC for the ideal gas and the
//! LLF corrector, with the trial state it shares with the HLLE unit.
//! One unit per ideal-gas first-order corrector and per other EOS policy, so each gets
//! its own ptxas pass; see dyn_grmhd_fofc_impl.hpp.

#include "dyn_grmhd_fofc_impl.hpp"

namespace dyngr {

template
void DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor>::
  BuildFOFCTrial(Driver *pdriver, int stage, int il, int iu, int jl, int ju, int kl,
                 int ku);
template
void DynGRMHDPS<Primitive::IdealGas, Primitive::ResetFloor>::
  FOFC<DynGRMHD_RSolver::llf_dyngr>(Driver *pdriver, int stage);

} // namespace dyngr
