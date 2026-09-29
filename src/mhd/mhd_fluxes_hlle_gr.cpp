//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_hlle_gr.cpp
//! \brief Explicit instantiation of MHD::CalculateFluxes<MHD_RSolver::hlle_gr>()
//! (general-relativistic HLLE).  One .cpp per solver so each gets its own ptxas pass; see
//! mhd_fluxes_impl.hpp.

#include "mhd_fluxes_impl.hpp"

namespace mhd {

// The x2 and x3 face kernels are compiled in mhd_fluxes_hlle_gr_x2.cpp and _x3.cpp.
extern template MHD_FACE_FLUX_KERNELS_DECL(MHD_RSolver::hlle_gr, IVY);
extern template MHD_FACE_FLUX_KERNELS_DECL(MHD_RSolver::hlle_gr, IVZ);

template void MHD::CalculateFluxes<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
