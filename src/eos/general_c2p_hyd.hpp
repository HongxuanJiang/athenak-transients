#ifndef EOS_GENERAL_C2P_HYD_HPP_
#define EOS_GENERAL_C2P_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file general_c2p_hyd.hpp
//! \brief EOS-aware nonrel hydro primitive/conserved helpers shared by gamma-law and
//! tabulated EOS paths.

#include <cmath>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace eos_general {

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
}

KOKKOS_INLINE_FUNCTION
Real ApplyHydroThermalFloors(const EOS_Data &eos, const Real dens, Real eint,
                             bool &efloor_used, bool &tfloor_used) {
  const Real eint_floor = eos.HydroInternalEnergyDensityFloor(dens);
  if (eint < eint_floor) {
    eint = eint_floor;
    efloor_used = true;
  }

  // Tabulated LTE EOS floors already enforce tfloor through the cached table closure.
  if ((eos.tfloor > 0.0) && !eos.UsesTabulatedLTE()) {
    const auto thermo = eos.EvalThermoStateFromRhoEint(dens, eint);
    if (thermo.temperature < eos.tfloor) {
      eint = dens*eos.SpecificEintFromRhoT(dens, eos.tfloor);
      tfloor_used = true;
    }
  }
  const Real eint_limited = eos.ClampHydroInternalEnergyDensity(dens, eint);
  if (eint_limited < eint) {
    eint = eint_limited;
    efloor_used = true;
  }
  return eint;
}

KOKKOS_INLINE_FUNCTION
Real HydroAtmosphereInternalEnergy(const EOS_Data &eos) {
  bool efloor_used = false;
  bool tfloor_used = false;
  return ApplyHydroThermalFloors(eos, eos.dfloor, static_cast<Real>(0.0),
                                 efloor_used, tfloor_used);
}

KOKKOS_INLINE_FUNCTION
bool NeedsHydroAtmosphereReset(const EOS_Data &eos, const HydCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetHydroAtmosphereState(const EOS_Data &eos, HydCons1D &u, HydPrim1D &w,
                               bool &dfloor_used) {
  const Real atmosphere_eint = HydroAtmosphereInternalEnergy(eos);
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ApplyHydroVelocityCeiling(const EOS_Data &eos, HydPrim1D &w, bool &vceil_used) {
  const Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  const Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    const Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vceil_used = true;
  }
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralHyd(HydCons1D &u, const EOS_Data &eos, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsHydroAtmosphereReset(eos, u)) {
    ResetHydroAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));

  w.e = ApplyHydroThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
  ApplyHydroVelocityCeiling(eos, w, vceil_used);

  SingleP2C_GeneralHyd(w, u);
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralHydDual(HydCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, HydPrim1D &w, Real &eint_aux_out,
                              bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                              bool &vceil_used) {
  if (NeedsHydroAtmosphereReset(eos, u)) {
    ResetHydroAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  Real eint_aux = ApplyHydroThermalFloors(eos, w.d, eint_aux_in, efloor_used,
                                          tfloor_used);

  const bool use_cons_e =
      (eint_cons > 0.0) &&
      ((dual_eta1 <= 0.0) || (eint_cons > dual_eta1*fmax(u.e, 1.0e-18)));
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyHydroThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);

  ApplyHydroVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralHyd(w, u);
  eint_aux_out = eint_aux;
}

}  // namespace eos_general

#endif  // EOS_GENERAL_C2P_HYD_HPP_
