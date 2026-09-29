//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fofc_eoshybrid_logs.cpp
//! \brief Explicit instantiation of the dynamical-GRMHD FOFC for the hybrid EOS
//! with NormalLogs.
//! One unit per ideal-gas first-order corrector and per other EOS policy, so each gets
//! its own ptxas pass; see dyn_grmhd_fofc_impl.hpp.

#include "dyn_grmhd_fofc_impl.hpp"

namespace dyngr {

INSTANTIATE_FOFC(Primitive::EOSHybrid<Primitive::NormalLogs>, Primitive::ResetFloor)

} // namespace dyngr
