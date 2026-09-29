//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fluxes_ideal_hlld.cpp
//! \brief Explicit instantiation of DynGRMHDPS::CalcFluxes() for the ideal gas and the
//! HLLD solver.
//! One unit per ideal-gas Riemann solver and per other EOS policy, so each gets its own
//! ptxas pass; see dyn_grmhd_fluxes_impl.hpp.

#include "dyn_grmhd_fluxes_impl.hpp"

namespace dyngr {

template
TaskStatus DynGRMHDPS<Primitive::IdealGas,
                      Primitive::ResetFloor>::
    CalcFluxes<DynGRMHD_RSolver::hlld_dyngr>(Driver *pdriver, int stage);

} // namespace dyngr
