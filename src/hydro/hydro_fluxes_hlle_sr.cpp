//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes_hlle_sr.cpp
//! \brief Explicit instantiation of Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>()
//! (special-relativistic HLLE).  One .cpp per solver so each gets its own ptxas pass; see
//! hydro_fluxes_impl.hpp.

#include "hydro_fluxes_impl.hpp"

namespace hydro {

template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);

} // namespace hydro
