#ifndef EOS_GENERAL_C2P_MHD_HPP_
#define EOS_GENERAL_C2P_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file general_c2p_mhd.hpp
//! \brief EOS-aware nonrel MHD primitive/conserved helpers shared by gamma-law and
//! tabulated EOS paths.

#include <cmath>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace eos_general {

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralMHD(const MHDPrim1D &w, MHDCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
        0.5*(SQR(w.bx) + SQR(w.by) + SQR(w.bz));
  u.by = w.by;
  u.bz = w.bz;
}

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralMHD(const MHDPrim1D &w, HydCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
        0.5*(SQR(w.bx) + SQR(w.by) + SQR(w.bz));
}

// Dual-energy selection for Newtonian MHD: the internal energy is read off the energy
// channel, eint = E - KE - B^2/2, only where it is not lost in the subtraction.  This is
// the ONE statement of that rule.  The inversions call it to choose the pressure, and
// the eta2 resynchronisation calls it too, so the auxiliary is only ever overwritten by
// a value the inversion would itself accept.
//
// Two thresholds, because E carries two truncation errors of different size.  The eta1
// test of Bryan et al. (1995) bounds the kinetic one.  The magnetic one is first order
// in the field perturbation, because the Riemann flux of E and the CT update of B are
// two discretisations of the magnetic energy that agree only to leading order in B.dB:
// per stage E - B^2/2 is off by ~ CFL*(dv/v_A)*B^2/2, and for any motion slower than
// sound (dv < c_s) that is ~ CFL*sqrt(beta)*B^2/2.  It exceeds the thermal energy,
// ~ beta*B^2/2, below beta ~ CFL^2 ~ 0.1, however quiet the flow.  A low-beta star
// atmosphere (beta ~ 1e-3) measured it at 0.3-1% of B^2/2 per stage against a thermal
// energy of 0.03-0.06%, which eta1 = 1e-3 of E ~ B^2/2 cannot reject.  Hence the second
// test, a fixed fraction of the magnetic energy: below it the energy channel is noise
// and the auxiliary, which sees only advection and compression, is the thermal state.
//
// A rejected energy channel must not be kept: the inversion that rejects it writes
// E = eint_aux + KE + B^2/2 back.  Left alone, E integrates the error of every stage,
// and since nothing else ever reads it the residual grows until it passes any fixed
// threshold, when the whole of it is released as heat at once.  E is therefore not
// conserved in a rejected cell, by exactly the residual that is not thermal energy.
inline constexpr Real kDualEnergyMagneticFraction = 0.1;

KOKKOS_INLINE_FUNCTION
bool MHDEnergyChannelResolvesEint(const Real eint_cons, const Real etot, const Real emag,
                                  const Real dual_eta1) {
  return (eint_cons > 0.0) &&
         ((dual_eta1 <= 0.0) ||
          ((eint_cons > dual_eta1*fmax(etot, 1.0e-18)) &&
           (eint_cons > kDualEnergyMagneticFraction*emag)));
}

// The one place the magnetic test is wrong is a shock.  Shock heating is entropy, which
// only the energy channel records; the auxiliary compresses adiabatically.  A low-beta
// shock (post-shock gas below 0.1*B^2/2) that the magnetic test rejects loses its
// heating to the reset, and a beta = 0.01 Brio-Wu slow shock came out at 0.64 of its
// peak specific energy.  So a cell whose flow converges supersonically keeps the energy
// channel: `converging_dv` is the velocity difference across the cell's own stencil,
// sum_d [v_d(-1) - v_d(+1)], and gas approaching itself faster than sound (Mach > 1 on
// the auxiliary's sound speed, which is not polluted) cannot be decelerated by pressure
// waves -- it shocks.  Slower convergence is adiabatic compression to the order that
// matters (shock entropy is third order in the jump), and the auxiliary follows it.
// Only the Bryan eta1 test still applies.  The price is the magnetic truncation error in
// the shocked cells, which the scheme accepted there before the magnetic test existed.
// Callers must have floored eint_aux.  Evaluated by the eta2 resynchronisation, which
// copies the accepted energy into the auxiliary, so every inversion -- pointwise ones
// included, which cannot see a neighbour -- takes the shock-heated value through the
// auxiliary channel.
KOKKOS_INLINE_FUNCTION
bool MHDEnergyChannelCarriesShockHeat(const EOS_Data &eos, const Real dens,
                                      const Real eint_cons, const Real etot,
                                      const Real eint_aux, const Real converging_dv,
                                      const Real dual_eta1) {
  if (!(eint_cons > 0.0) || !(converging_dv > 0.0)) return false;
  if ((dual_eta1 > 0.0) && !(eint_cons > dual_eta1*fmax(etot, 1.0e-18))) return false;
  Real pressure, cs2;
  eos.EvalPressureCs2FromRhoEint(dens, eint_aux, pressure, cs2);
  return SQR(converging_dv) > cs2;
}

KOKKOS_INLINE_FUNCTION
Real ApplyMHDThermalFloors(const EOS_Data &eos, const Real dens, Real eint,
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
Real MHDAtmosphereInternalEnergy(const EOS_Data &eos) {
  bool efloor_used = false;
  bool tfloor_used = false;
  return ApplyMHDThermalFloors(eos, eos.dfloor, static_cast<Real>(0.0),
                               efloor_used, tfloor_used);
}

KOKKOS_INLINE_FUNCTION
bool NeedsMHDAtmosphereReset(const EOS_Data &eos, const MHDCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetMHDAtmosphereState(const EOS_Data &eos, MHDCons1D &u, MHDPrim1D &w,
                             bool &dfloor_used) {
  const Real atmosphere_eint = MHDAtmosphereInternalEnergy(eos);
  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint + emag;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ResetMHDAtmosphereState(const EOS_Data &eos, MHDCons1D &u, HydPrim1D &w,
                             bool &dfloor_used) {
  const Real atmosphere_eint = MHDAtmosphereInternalEnergy(eos);
  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint + emag;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ApplyMHDVelocityCeiling(const EOS_Data &eos, MHDPrim1D &w, bool &vceil_used) {
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
void SingleC2P_GeneralMHD(MHDCons1D &u, const EOS_Data &eos, MHDPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))
              - 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
  ApplyMHDVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralMHD(w, u);
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHD(MHDCons1D &u, const EOS_Data &eos, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))
              - 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
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
void SingleC2P_GeneralMHDDual(MHDCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, MHDPrim1D &w, Real &eint_aux_out,
                              bool &eint_from_aux, bool &dfloor_used, bool &efloor_used,
                              bool &tfloor_used, bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    eint_from_aux = false;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;

  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz)) - emag;
  Real eint_aux = ApplyMHDThermalFloors(eos, w.d, eint_aux_in, efloor_used,
                                        tfloor_used);

  const bool use_cons_e = MHDEnergyChannelResolvesEint(eint_cons, u.e, emag, dual_eta1);
  eint_from_aux = !use_cons_e;
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);

  ApplyMHDVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralMHD(w, u);
  eint_aux_out = eint_aux;
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHDDual(MHDCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, HydPrim1D &w, Real &eint_aux_out,
                              bool &eint_from_aux, bool &dfloor_used, bool &efloor_used,
                              bool &tfloor_used, bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    eint_from_aux = false;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz)) - emag;
  Real eint_aux = ApplyMHDThermalFloors(eos, w.d, eint_aux_in, efloor_used,
                                        tfloor_used);

  const bool use_cons_e = MHDEnergyChannelResolvesEint(eint_cons, u.e, emag, dual_eta1);
  eint_from_aux = !use_cons_e;
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);

  const Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  const Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    const Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vceil_used = true;
  }
  eint_aux_out = eint_aux;
}

}  // namespace eos_general

#endif  // EOS_GENERAL_C2P_MHD_HPP_
