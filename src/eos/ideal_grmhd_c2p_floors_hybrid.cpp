//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd_c2p_floors_hybrid.cpp
//! \brief Explicit instantiation of IdealGRMHD::ConsToPrimImpl for the floors-only test.
//! Two units share the specializations so each gets its own ptxas pass; see
//! ideal_grmhd_c2p_impl.hpp.

#include "ideal_grmhd_c2p_impl.hpp"

INSTANTIATE_IDEAL_GRMHD_C2P(true, false, false)
INSTANTIATE_IDEAL_GRMHD_C2P(true, false, true)
INSTANTIATE_IDEAL_GRMHD_C2P(true, true, false)
INSTANTIATE_IDEAL_GRMHD_C2P(true, true, true)
