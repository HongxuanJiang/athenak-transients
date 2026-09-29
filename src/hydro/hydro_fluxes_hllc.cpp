//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes_hllc.cpp
//! \brief Explicit instantiation of Hydro::CalculateFluxes<Hydro_RSolver::hllc>()
//! (HLLC).  One .cpp per solver so each gets its own ptxas pass; see
//! hydro_fluxes_impl.hpp.

#include "hydro_fluxes_impl.hpp"

namespace hydro {

template void Hydro::CalculateFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);

} // namespace hydro
