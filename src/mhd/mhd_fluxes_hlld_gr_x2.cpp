//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_hlld_gr_x2.cpp
//! \brief Explicit instantiation of the x2 face kernels of
//! MHD::CalculateFluxes<MHD_RSolver::hlld_gr>() (general-relativistic HLLD),
//! LaunchMHDFaceFluxKernels<MHD_RSolver::hlld_gr, IVY>().  One .cpp per GR solver and
//! direction so each gets its own ptxas pass; see mhd_fluxes_impl.hpp and
//! mhd_fluxes_hlld_gr.cpp.

#include "mhd_fluxes_impl.hpp"

namespace mhd {

template MHD_FACE_FLUX_KERNELS_DECL(MHD_RSolver::hlld_gr, IVY);

} // namespace mhd
