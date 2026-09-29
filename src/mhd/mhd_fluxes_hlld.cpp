//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_hlld.cpp
//! \brief Explicit instantiation of MHD::CalculateFluxes<MHD_RSolver::hlld>()
//! (Newtonian HLLD).  One .cpp per solver so each gets its own ptxas pass; see
//! mhd_fluxes_impl.hpp.

#include "mhd_fluxes_impl.hpp"

namespace mhd {

template void MHD::CalculateFluxes<MHD_RSolver::hlld>(Driver *pdriver, int stage);

} // namespace mhd
