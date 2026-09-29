//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_ps_piecewise_poly.cpp
//! \brief Explicit instantiation of DynGRMHDPS and its AddCoordTermsEOS<NGHOST> for
//! the piecewise polytrope.
//! One unit per EOS family (the ideal gas: its class and its coordinate sources apart),
//! so each gets its own ptxas pass; see dyn_grmhd_ps_impl.hpp.

#include "dyn_grmhd_ps_impl.hpp"
#include "dyn_grmhd_coord_terms_impl.hpp"

namespace dyngr {

template class DynGRMHDPS<Primitive::PiecewisePolytrope,
                          Primitive::ResetFloor>;

INSTANTIATE_COORD_TERMS(Primitive::PiecewisePolytrope,
                        Primitive::ResetFloor);

} // namespace dyngr
