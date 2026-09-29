//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_llf_gr.cpp
//! \brief Explicit instantiation of MHD::CalculateFluxes<MHD_RSolver::llf_gr>()
//! (general-relativistic LLF).  One .cpp per solver so each gets its own ptxas pass; see
//! mhd_fluxes_impl.hpp.

#include "mhd_fluxes_impl.hpp"

namespace mhd {

// The x2 and x3 face kernels are compiled in mhd_fluxes_llf_gr_x2.cpp and _x3.cpp.
extern template MHD_FACE_FLUX_KERNELS_DECL(MHD_RSolver::llf_gr, IVY);
extern template MHD_FACE_FLUX_KERNELS_DECL(MHD_RSolver::llf_gr, IVZ);

template void MHD::CalculateFluxes<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);

} // namespace mhd
