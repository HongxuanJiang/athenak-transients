//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fofc_compose_nqt.cpp
//! \brief Explicit instantiation of the dynamical-GRMHD FOFC for the CompOSE table
//! with NQTLogs.
//! One unit per ideal-gas first-order corrector and per other EOS policy, so each gets
//! its own ptxas pass; see dyn_grmhd_fofc_impl.hpp.

#include "dyn_grmhd_fofc_impl.hpp"

namespace dyngr {

INSTANTIATE_FOFC(Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor)

} // namespace dyngr
