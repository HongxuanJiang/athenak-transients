//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_ps_ideal.cpp
//! \brief Explicit instantiation of DynGRMHDPS for the ideal gas: the task list and
//! the C2P and P2C kernels.  Its coordinate-source kernels are compiled in
//! dyn_grmhd_coord_terms_ideal.cpp.
//! One unit per EOS family (the ideal gas: its class and its coordinate sources apart),
//! so each gets its own ptxas pass; see dyn_grmhd_ps_impl.hpp.

#include "dyn_grmhd_ps_impl.hpp"

namespace dyngr {

template class DynGRMHDPS<Primitive::IdealGas,
                          Primitive::ResetFloor>;

} // namespace dyngr
