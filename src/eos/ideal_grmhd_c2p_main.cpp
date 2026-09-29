//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd_c2p_main.cpp
//! \brief Explicit instantiation of IdealGRMHD::ConsToPrimImpl for the ordinary recovery
//! (not floors-only).  Two units share the specializations so each gets its own ptxas
//! pass; see ideal_grmhd_c2p_impl.hpp.

#include "ideal_grmhd_c2p_impl.hpp"

INSTANTIATE_IDEAL_GRMHD_C2P(false, false, false)
INSTANTIATE_IDEAL_GRMHD_C2P(false, false, true)
INSTANTIATE_IDEAL_GRMHD_C2P(false, true, false)
INSTANTIATE_IDEAL_GRMHD_C2P(false, true, true)
