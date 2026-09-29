#ifndef EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_error.hpp
//  \brief defines an enumerator struct for error types.

namespace Primitive {
enum struct Error {
  SUCCESS,
  RHO_TOO_BIG,
  RHO_TOO_SMALL,
  NANS_IN_CONS,
  MAG_TOO_BIG,
  BRACKETING_FAILED,
  NO_SOLUTION,
  CONS_FLOOR,
  PRIM_FLOOR,
  CONS_ADJUSTED,
};

struct SolverResult {
  Error error;
  int  iterations;
  bool cons_floor;
  bool prim_floor;
  bool cons_adjusted;
  //! rho*h as the raw inversion recovered it, before any floor touched the state.  The
  //! GRMHD floors downstream re-inject whatever mass and heat they add in the drift frame
  //! (eos/drift_frame_floor.hpp), and that map needs the enthalpy density it started
  //! from.  Zero means "no usable value" -- a failed or excised inversion -- and the
  //! re-injection is then skipped, which leaves the velocity alone.
  Real w_prefloor = 0.0;
  //! The conserved floor raised D, by any amount.  The temperature recovered afterwards is
  //! the kept tau spread over the raised mass: for D << D_floor that mass is the floor's
  //! and the temperature is no gas temperature.  Every further mass the recovery adds to
  //! the cell then comes cold (see PrimitiveSolver::ConToPrim), which is also the
  //! conservative choice where D was just under D_floor and the mass is mostly the cell's.
  bool cons_mass_floor = false;
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
