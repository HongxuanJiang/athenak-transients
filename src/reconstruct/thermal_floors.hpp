#ifndef RECONSTRUCT_THERMAL_FLOORS_HPP_
#define RECONSTRUCT_THERMAL_FLOORS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermal_floors.hpp
//! \brief shared helpers for applying hydro thermal floors/ceilings to reconstructed
//! face states before they are consumed by EOS closures.

#include <math.h>

#include "athena.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn ApplyReconstructedHydroThermalFloors()
//! \brief Apply the same thermal floors used by cell-centered hydro states to a
//! reconstructed internal-energy face state before it is consumed by an EOS closure.

KOKKOS_INLINE_FUNCTION
Real ApplyReconstructedHydroThermalFloors(const EOS_Data &eos, const Real dens_in,
                                          Real eint) {
  const Real dens = fmax(dens_in, eos.dfloor);
  const Real eint_floor = eos.HydroInternalEnergyDensityFloor(dens);
  if (eint < eint_floor) {
    eint = eint_floor;
  }

  // Tabulated LTE EOS floors already enforce tfloor through the cached table closure.
  if ((eos.tfloor > 0.0) && !eos.UsesTabulatedLTE()) {
    const auto thermo = eos.EvalThermoStateFromRhoEint(dens, eint);
    if (thermo.temperature < eos.tfloor) {
      eint = dens*eos.SpecificEintFromRhoT(dens, eos.tfloor);
    }
  }

  return eos.ClampHydroInternalEnergyDensity(dens, eint);
}

//----------------------------------------------------------------------------------------
//! \fn FloorReconstructedInternalEnergyPair()
//! \brief Apply primitive-state thermal floors/ceilings to a reconstructed internal-energy
//! pair so face states stay consistent with cell-centered thermal repairs.

KOKKOS_INLINE_FUNCTION
void FloorReconstructedInternalEnergyPair(const EOS_Data &eos, Real &ql, Real &qr,
                                          const Real ql_rho, const Real qr_rho) {
  ql = ApplyReconstructedHydroThermalFloors(eos, ql_rho, ql);
  qr = ApplyReconstructedHydroThermalFloors(eos, qr_rho, qr);
}

#endif // RECONSTRUCT_THERMAL_FLOORS_HPP_
