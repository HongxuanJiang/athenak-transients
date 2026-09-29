#ifndef EOS_PRIMITIVE_SOLVER_RESET_FLOOR_HPP_
#define EOS_PRIMITIVE_SOLVER_RESET_FLOOR_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reset_floor.hpp
//  \brief Describes an error floor that simply resets nonphysical values.
//
//  If the density or pressure fall below the atmosphere, they get floored.  We impose
//  similar limits for D and tau.  Both floors only add.  The conserved floor raises D to
//  max(D, n_atm m_b) and keeps S_i and tau (the added rest mass is at rest in the normal
//  frame), then raises tau to the atmosphere value if below.  The primitive floor raises
//  n to max(n, n_atm) at the cell's own temperature (itself raised to T_atm if below); in
//  a cell whose D the conserved floor raised, PrimitiveSolver::ConToPrim then re-solves T
//  at the internal energy density the cell had, because there T is the kept tau over the
//  floor's mass.  In both the added mass has the atmosphere composition.  The primitive
//  floor leaves the velocity alone: what momentum the floored mass carries is not this
//  policy's decision.  The caller owns it, because only the caller knows the metric and
//  the magnetic field the answer depends on -- see PrimitiveSolverHydro
//  (eos/primitive_solver_hyd.hpp), which re-injects it in the drift frame of Ressler+2017
//  (eos/drift_frame_floor.hpp), a map that is defined for added enthalpy only.  Zeroing
//  the velocity or the momentum here is a momentum sink that pins any flow resting near
//  the atmosphere at rest for good.
//  If the pressure is floored, all other quantities are ignored.
//  If the primitive solve fails, all points are set to floor.

#include <math.h>

#include "ps_types.hpp"
#include "error_policy_interface.hpp"
#include "ps_error.hpp"

namespace Primitive {

class ResetFloor : public ErrorPolicyInterface {
  // NOTE: the species loops below run to the compile-time MAX_SPECIES and predicate on
  // the runtime count, so the caller's Y[] keeps constant subscripts and can stay in
  // registers.  See the note in PrimitiveSolver::ConToPrim for the measurement.
 protected:
  /// Constructor
  ResetFloor() {
    fail_conserved_floor = false;
    fail_primitive_floor = false;
    adjust_conserved = true;
  }

  /// Floor for primitive variables
  KOKKOS_INLINE_FUNCTION bool PrimitiveFloor(Real& n, Real v[3], Real& T, Real *Y,
                                             int n_species) const {
    // v is untouched by design; see the note at the top of this file.
    (void) v;
    if (n < n_atm*n_threshold) {
      // Added mass n_new - n at the cell's temperature; the n the cell had keeps its
      // composition, the added mass has the atmosphere's (n <= 0 keeps none, so a
      // non-finite Y there cannot survive the mix).
      const Real n_new = fmax(n, n_atm);
      const Real kept = (n > 0.0) ? n/n_new : 0.0;
      #pragma unroll
      for (int i = 0; i < MAX_SPECIES; i++) {
        if (i < n_species) Y[i] = (kept > 0.0) ? Y_atm[i] + (Y[i] - Y_atm[i])*kept : Y_atm[i];
      }
      n = n_new;
      T = fmax(T, T_atm);
      return true;
    } else if (T < T_atm) {
      T = T_atm;
      return true;
    }
    return false;
  }

  /// Floor for conserved variables
  KOKKOS_INLINE_FUNCTION bool ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y,
                                Real D_floor, Real tau_floor, Real tau_abs_floor,
                                int n_species) const {
    if (D < D_floor*n_threshold) {
      // Added rest mass D_new - D at rest in the normal frame: S_i and tau are kept.  The
      // D the cell had keeps its composition, the added mass has the atmosphere's.
      // tau_floor was evaluated at the incoming D; a cell raised to D_floor takes the
      // atmosphere's tau floor, tau_abs_floor.
      const bool raised = (D < D_floor);
      const Real D_new = raised ? D_floor : D;
      const Real kept = (D > 0.0) ? D/D_new : 0.0;
      #pragma unroll
      for (int i = 0; i < MAX_SPECIES; i++) {
        if (i < n_species) {
          Y[i] = (kept > 0.0) ? Y_atm[i] + (Y[i] - Y_atm[i])*kept : Y_atm[i];
        }
      }
      D = D_new;
      tau = fmax(tau, raised ? tau_abs_floor : tau_floor);
      return true;
    } else if (tau < tau_floor) {
      tau = tau_floor;
      return true;
    }
    return false;
  }

  /// Response to excess magnetization
  KOKKOS_INLINE_FUNCTION Error MagnetizationResponse(Real& bsq, Real b_u[3]) const {
    if (bsq > max_bsq) {
      Real factor = sqrt(max_bsq/bsq);
      bsq = max_bsq;

      b_u[0] *= factor;
      b_u[1] *= factor;
      b_u[2] *= factor;

      return Error::CONS_ADJUSTED;
    }
    return Error::SUCCESS;
  }

  /// Policy for resetting density
  KOKKOS_INLINE_FUNCTION void DensityLimits(Real& n, Real n_min, Real n_max) const {
    n = fmax(n_min, fmin(n_max, n));
  }

  /// Policy for resetting temperature
  KOKKOS_INLINE_FUNCTION void TemperatureLimits(Real& T, Real T_min, Real T_max) const {
    T = fmax(T_min, fmin(T_max, T));
  }

  /// Policy for resetting species fractions
  KOKKOS_INLINE_FUNCTION bool SpeciesLimits(Real* Y, const Real* Y_min, const Real* Y_max,
                                            int n_species) const {
    bool adjusted = false;
    #pragma unroll
    for (int i = 0; i < MAX_SPECIES; i++) {
      if (i < n_species) {
        if (Y[i] < Y_min[i]) {
          adjusted = true;
          Y[i] = Y_min[i];
        } else if (Y[i] > Y_max[i]) {
          adjusted = true;
          Y[i] = Y_max[i];
        }
      }
    }
    return adjusted;
  }

  /// Policy for resetting pressure
  KOKKOS_INLINE_FUNCTION void PressureLimits(Real& P, Real P_min, Real P_max) const {
    P = fmax(P_min, fmin(P_max, P));
  }

  /// Policy for resetting energy density
  KOKKOS_INLINE_FUNCTION void EnergyLimits(Real& e, Real e_min, Real e_max) const {
    e = fmax(e_min, fmin(e_max, e));
  }

  /// Policy for dealing with failed points
  KOKKOS_INLINE_FUNCTION bool FailureResponse(Real prim[NPRIM]) const {
    prim[PRH] = n_atm;
    prim[PVX] = 0.0;
    prim[PVY] = 0.0;
    prim[PVZ] = 0.0;
    prim[PTM] = T_atm;
    for (int i = 0; i < MAX_SPECIES; i++) {
      prim[PYF + i] = Y_atm[i];
    }
    return true;
  }

 public:
  /// Set the failure mode for conserved flooring
  KOKKOS_INLINE_FUNCTION void SetConservedFloorFailure(bool failure) {
    fail_conserved_floor = failure;
  }

  /// Set the failure mode for primitive flooring
  KOKKOS_INLINE_FUNCTION void SetPrimitiveFloorFailure(bool failure) {
    fail_primitive_floor = failure;
  }

  /// Set whether or not it's okay to adjust the conserved variables.
  KOKKOS_INLINE_FUNCTION void SetAdjustConserved(bool adjust) {
    adjust_conserved = adjust;
  }
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_RESET_FLOOR_HPP_
