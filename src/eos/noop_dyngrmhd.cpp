//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file noop_dyngrmhd.cpp
//! \brief derived class for DynGRMHD that acts as a no-op

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

NoOpDynGRMHD::NoOpDynGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  // DynGRMHD recovers primitives with PrimitiveSolver, whose own EOS policy owns the
  // adiabatic index (PrimitiveSolverHydro::SetPolicyParams reads <mhd>/gamma for
  // dyn_eos = ideal and the pwp_* family for dyn_eos = piecewise_poly).  Nothing on the
  // recovery path reads eos_data.gamma; only shared helpers written for the fixed-metric
  // fluids do (srcterms disk cooling converts P to eps with it).  So a deck whose dyn_eos
  // is not ideal need not carry <mhd>/gamma, and the default here is the same 5/3 the
  // ideal policy defaults to, so the two agree whenever the key is absent.
  eos_data.gamma = pin->GetOrAddReal("mhd","gamma",5.0/3.0);
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
}
