//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes_llf_gr.cpp
//! \brief Explicit instantiation of Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>()
//! (general-relativistic LLF).  One .cpp per solver so each gets its own ptxas pass; see
//! hydro_fluxes_impl.hpp.

#include "hydro_fluxes_impl.hpp"

namespace hydro {

template void Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);

} // namespace hydro
