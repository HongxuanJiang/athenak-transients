//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fluxes_compose_logs.cpp
//! \brief Explicit instantiation of DynGRMHDPS::CalcFluxes() for the CompOSE table with
//! NormalLogs and every Riemann solver.
//! One unit per ideal-gas Riemann solver and per other EOS policy, so each gets its own
//! ptxas pass; see dyn_grmhd_fluxes_impl.hpp.

#include "dyn_grmhd_fluxes_impl.hpp"

namespace dyngr {

template
TaskStatus DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                      Primitive::ResetFloor>::
    CalcFluxes<DynGRMHD_RSolver::llf_dyngr>(Driver *pdriver, int stage);
template
TaskStatus DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                      Primitive::ResetFloor>::
    CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>(Driver *pdriver, int stage);
template
TaskStatus DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                      Primitive::ResetFloor>::
    CalcFluxes<DynGRMHD_RSolver::hlld_dyngr>(Driver *pdriver, int stage);

} // namespace dyngr
