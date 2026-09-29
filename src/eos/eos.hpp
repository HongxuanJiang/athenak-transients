#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//! \brief Contains data and functions that implement conserved->primitive variable
//! conversion for various EOS (e.g. ideal gas, isothermal, etc.), for various fluids
//! (Hydro, MHD, etc.), and for non-relativistic and relativistic flows.

#include <math.h>
//#include <cmath>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \struct EOSData
//! \brief container for EOS variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

enum class HydroEOSModel : int {
  gamma_law = 0,
  isothermal = 1,
  saha_table = 2,
  lte_table = 3
};
enum class SahaBoundsMode : int {error = 0, clamp = 1};

struct EOS_Data {
  static constexpr int saha_nvars = 11;
  static constexpr int saha_cache_nvars = 11;
  enum SahaTableVar : int {
    saha_logpress = 0,
    saha_logeps = 1,
    saha_logcs2 = 2,
    saha_gamma1 = 3,
    saha_gamma3m1 = 4,
    saha_xh2 = 5,
    saha_xion = 6,
    saha_xhe1 = 7,
    saha_xhe2 = 8,
    saha_mu = 9,
    saha_beta_rad = 10
  };
  enum SahaCacheVar : int {
    saha_cache_temp = 0,
    saha_cache_press = 1,
    saha_cache_cs2 = 2,
    saha_cache_gamma1 = 3,
    saha_cache_gamma3m1 = 4,
    saha_cache_xh2 = 5,
    saha_cache_xion = 6,
    saha_cache_xhe1 = 7,
    saha_cache_xhe2 = 8,
    saha_cache_mu = 9,
    saha_cache_beta_rad = 10
  };

  struct ThermoState {
    Real temperature = 0.0;
    Real pressure = 0.0;
    Real cs2 = 0.0;
    Real gamma1 = 0.0;
    Real gamma3m1 = 0.0;
    Real xh2 = 0.0;
    Real xion = 0.0;
    Real xhe1 = 0.0;
    Real xhe2 = 0.0;
    Real mu = 0.0;
    Real beta_rad = 0.0;
  };

  HydroEOSModel hydro_eos = HydroEOSModel::gamma_law;
  Real gamma = 0.0;        // ratio of specific heats for ideal gas
  Real iso_cs = 0.0;       // isothermal sound speed
  bool is_ideal = false;   // flag to denote hydro/MHD with an energy variable
  bool is_gamma_law = false; // flag to denote constant-gamma ideal gas EOS
  bool use_e = false, use_t = false; // use internal energy density (e) or temperature (t)
  Real dfloor = 0.0, pfloor = 0.0, tfloor = 0.0, sfloor = 0.0;
  Real cs_ceil = 0.0;      // optional NR ceiling on sound speed
  Real gamma_max = 0.0;    // ceiling on Lorentz factor in SR/GR
  Real sigma_max = 0.0;    // optional GRMHD ceiling on sigma = b^2/rho; <= 0 disables
  Real vceil = 0.0;        // NR ceiling on |v|
  Real density_unit_cgs = 1.0;
  Real pressure_unit_cgs = 1.0;
  Real specific_eint_unit_cgs = 1.0;
  // The code temperature unit in K: the cgs value of the number this EOS calls
  // temperature.  The ideal EOS copies Units::temperature_cgs() here (eos.cpp); a
  // tabulated LTE/Saha EOS overwrites it with its own v_cgs^2*m_H/k_B scale, because
  // the table's temperature axis is in K and knows nothing about <units>/mu.  Read it
  // through MeshBlockPack::TemperatureUnitCGS() outside the EOS.
  Real temp_unit_cgs = 1.0;
  Real lte_h_mass_fraction = 1.0;
  Real lte_he_mass_fraction = 0.0;
  bool lte_has_helium = false;
  bool lte_has_radiation = false;
  bool lte_has_h2 = false;
  bool lte_zpe_subtracted = false;
  Real saha_tmin_code = 0.0;
  Real saha_tmax_code = 0.0;
  Real saha_logrho_min = 0.0;
  Real saha_logrho_max = 0.0;
  Real saha_logtemp_min = 0.0;
  Real saha_logtemp_max = 0.0;
  Real saha_logeps_min = 0.0;
  Real saha_logeps_max = 0.0;
  Real saha_inv_dlogrho = 0.0;
  Real saha_inv_dlogtemp = 0.0;
  Real saha_inv_dlogeps = 0.0;
  Real saha_floor_logrho_min = 0.0;
  Real saha_floor_logrho_max = 0.0;
  Real saha_floor_inv_dlogrho = 0.0;
  int saha_nrho = 0;
  int saha_ntemp = 0;
  int saha_neps = 0;
  int saha_floor_nrho = 0;
  bool saha_debug_checks = false;
  SahaBoundsMode saha_bounds_mode = SahaBoundsMode::error;
  DvceArray1D<Real> saha_logrho;
  DvceArray1D<Real> saha_logtemp;
  DvceArray3D<Real> saha_table;
  DvceArray3D<Real> saha_thermo_cache;
  DvceArray1D<Real> saha_logeps_floor;
  DvceArray1D<Real> saha_logeps_ceil;
  HostArray1D<Real> saha_logrho_h;
  HostArray1D<Real> saha_logtemp_h;
  HostArray3D<Real> saha_table_h;
  HostArray3D<Real> saha_thermo_cache_h;
  HostArray1D<Real> saha_logeps_floor_h;
  HostArray1D<Real> saha_logeps_ceil_h;

  // IDEAL GAS PRESSURE: converts primitive variable (either internal energy density e
  // or temperature e/d) into pressure.
  KOKKOS_INLINE_FUNCTION
  Real IdealGasPressure(const Real eint) const {
    return ((gamma-1.0)*eint);
  }

  KOKKOS_INLINE_FUNCTION
  bool UsesTabulatedLTE() const {
    return (hydro_eos == HydroEOSModel::saha_table) ||
           (hydro_eos == HydroEOSModel::lte_table);
  }

  KOKKOS_INLINE_FUNCTION
  Real PressureFromRhoEint(const Real d, const Real eint) const;

  KOKKOS_INLINE_FUNCTION
  Real PressureFromRhoT(const Real d, const Real T) const;

  KOKKOS_INLINE_FUNCTION
  Real SpecificEintFromRhoT(const Real d, const Real T) const;

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromRhoEint(const Real d, const Real eint) const;
  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromRhoEint(const Real d, const Real eint, Real &dtemp_deint) const;

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromRhoP(const Real d, const Real p) const;

  KOKKOS_INLINE_FUNCTION
  Real SpecificEintFromRhoP(const Real d, const Real p) const;

  KOKKOS_INLINE_FUNCTION
  Real InternalEnergyDensityFromRhoP(const Real d, const Real p) const;

  KOKKOS_INLINE_FUNCTION
  ThermoState EvalThermoStateFromRhoT(const Real d, const Real T) const;

  KOKKOS_INLINE_FUNCTION
  ThermoState EvalThermoStateFromRhoEint(const Real d, const Real eint) const;

  KOKKOS_INLINE_FUNCTION
  void EvalPressureCs2FromRhoEint(const Real d, const Real eint,
                                  Real &pressure, Real &cs2) const;

  KOKKOS_INLINE_FUNCTION
  Real HydroSoundSpeed2FromRhoEint(const Real d, const Real eint) const;

  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeedFromSoundSpeed2(const Real d, const Real cs2,
                                            const Real bx, const Real by,
                                            const Real bz) const;

  KOKKOS_INLINE_FUNCTION
  Real IonizationFractionFromRhoT(const Real d, const Real T) const;

  KOKKOS_INLINE_FUNCTION
  Real HydroInternalEnergyDensityFloor(const Real d) const;

  KOKKOS_INLINE_FUNCTION
  Real HydroInternalEnergyDensityCeiling(const Real d) const;

  KOKKOS_INLINE_FUNCTION
  Real ClampHydroInternalEnergyDensity(const Real d, const Real eint) const;

  Real HostHydroInternalEnergyDensityFloor(const Real d) const;

  Real HostHydroInternalEnergyDensityCeiling(const Real d) const;

  Real HostClampHydroInternalEnergyDensity(const Real d, const Real eint) const;

  Real HostPressureFromRhoEint(const Real d, const Real eint) const;

  Real HostInternalEnergyDensityFromRhoP(const Real d, const Real p) const;

  Real HostGamma1FromRhoP(const Real d, const Real p) const;

  Real HostDensityInActiveDomain(const Real d) const;

  Real HostTabulatedPressureFloor(const Real d) const;

  Real HostTabulatedPressureCeiling(const Real d) const;

  // NON-RELATIVISTIC IDEAL GAS HYDRO: inlined sound speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealHydroSoundSpeed(const Real d, const Real p) const {
    return sqrt(gamma*p/d);
  }

  // NON-RELATIVISTIC IDEAL GAS MHD: inlined fast magnetosonic speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealMHDFastSpeed(const Real d, const Real p,
                         const Real bx, const Real by, const Real bz) const {
    Real asq = gamma*p;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return sqrt(0.5*(qsq + sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // NON-RELATIVISTIC ISOTHERMAL MHD: inlined fast magnetosonic speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealMHDFastSpeed(const Real d,
                         const Real bx, const Real by, const Real bz) const {
    Real asq = (iso_cs*iso_cs)*d;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return sqrt(0.5*(qsq + sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // SPECIAL RELATIVISTIC IDEAL GAS HYDRO: inlined maximal sound wave speeds function
  // Inputs:
  //   d: density in comoving frame
  //   p: gas pressure
  //   ux: x-component of 4-velocity u^x
  //   lor: Lorentz factor \gamma
  // Outputs:
  //   l_p/m: most positive/negative wavespeed
  // Reference:
  //   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)
  KOKKOS_INLINE_FUNCTION
  void IdealSRHydroSoundSpeeds(const Real d, const Real p, const Real ux, const Real lor,
                               Real& l_p, Real& l_m) const {
    Real cs2 = gamma*p / (d + gamma*p/(gamma - 1.0));  // (DZB 73)
    Real v2 = 1.0 - 1.0/(lor*lor);
    auto const p1 = (ux/lor) * (1.0 - cs2);
    auto const tmp = sqrt(cs2 * ((1.0-v2*cs2) - p1*(ux/lor))) / lor;
    auto const invden = 1.0/(1.0 - v2*cs2);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // SPECIAL RELATIVISTIC IDEAL GAS MHD: inlined maximal fast magnetosonic wave speeds fn
  // arguments same or SR hydro version, with the addition of b_sq = b_\mu b_\mu
  // Reference:
  //   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)
  KOKKOS_INLINE_FUNCTION
  void IdealSRMHDFastSpeeds(const Real d, const Real p, const Real ux, const Real lor,
                            const Real b_sq, Real& l_p, Real& l_m) const {
    // Calculate comoving fast magnetosonic speed
    Real w = d + gamma*p/(gamma - 1.0);
    Real cs_sq = gamma*p/w;                            // (DZB 73)
    Real va_sq = b_sq / (b_sq + w);                    // (DZB 73)
    Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;       // (DZB 72)

    Real v2 = 1.0 - 1.0/(lor*lor);
    auto const p1 = (ux/lor) * (1.0 - cms_sq);
    auto const tmp = sqrt(cms_sq * ((1.0-v2*cms_sq) - p1*(ux/lor))) / lor;
    auto const invden = 1.0/(1.0 - v2*cms_sq);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // GENERAL RELATIVISTIC IDEAL GAS HYDRO: inlined maximal sound wave speeds fn
  // Inputs:
  //  - d: density in comoving frame
  //  - p: gas pressure
  //  - u0,u1: 4-velocity components u^0, u^1
  //  - g00,g01,g11: metric components g^00, g^01, g^11
  // Outputs:
  //  - l_p/l_m: most positive/negative wavespeed
  // Notes:
  //  - Follows same general procedure as vchar() in phys.c in Harm.
  //  - Variables are named as though 1 is normal direction.
  KOKKOS_INLINE_FUNCTION
  void IdealGRHydroSoundSpeeds(const Real d, const Real p, const Real u0, const Real u1,
                               const Real g00, const Real g01, const Real g11,
                               Real& l_p, Real& l_m) const {
    // Parameters and constants
    const Real discriminant_tol = -1.0e-10;  // values between this and 0 are considered 0

    // Calculate comoving sound speed
    Real cs_sq = gamma * p / (d + gamma*p/(gamma - 1.0));

    // Set sound speeds in appropriate coordinates
    Real a = SQR(u0) - (g00 + SQR(u0)) * cs_sq;
    Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cs_sq);
    Real c = SQR(u1) - (g11 + SQR(u1)) * cs_sq;
    Real dis = SQR(b) - 4.0*a*c;
    if (dis < 0.0 && dis > discriminant_tol) {
      dis = 0.0;
    }
    // TODO(@pdmullen): fmax(dis, 0.0) prevents NaNs (see Issue #7), but this should be
    // eliminated after enforcing positivity on recon L/R densities and pressures
    Real dis_sqrt = sqrt(fmax(dis, 0.0));
    Real root_1 = (-b + dis_sqrt) / (2.0*a);
    Real root_2 = (-b - dis_sqrt) / (2.0*a);
    if (root_1 > root_2) {
      l_p = root_1;
      l_m = root_2;
    } else {
      l_p = root_2;
      l_m = root_1;
    }
  }

  // GENERAL RELATIVISTIC IDEAL GAS MHD: inlined maximal fast magnetosonic wave speeds fn
  // Inputs:
  //  - d: density in comoving frame
  //  - h: enthalpy per unit volume
  //  - p: gas pressure
  //  - u0, u1: contravariant components of 4-velocity
  //  - b_sq: b_\mu b^\mu
  //  - g00, g01, g11: contravariant components of metric (-1, 0, 1 in SR)
  // Outputs:
  //  - l_p/l_m: most positive/negative wavespeed
  // Notes:
  //  - Follows same general procedure as vchar() in phys.c in Harm.
  //  - Variables are named as though 1 is normal direction.
  KOKKOS_INLINE_FUNCTION
  void IdealGRMHDFastSpeeds(const Real d, const Real p, const Real u0, const Real u1,
                            const Real b_sq, const Real g00, const Real g01,
                            const Real g11, Real& l_p, Real& l_m) const {
    // Calculate comoving fast magnetosonic speed
    Real w = d + gamma*p/(gamma - 1.0);
    Real cs_sq = gamma * p / w;
    Real va_sq = b_sq / (b_sq + w);
    Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

    // Set fast magnetosonic speeds in appropriate coordinates
    Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
    Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
    Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
    Real a1 = b / a;
    Real a0 = c / a;
    Real s = fmax(SQR(a1) - 4.0 * a0, 0.0);
    s = sqrt(s);
    l_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
    l_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
  }

 private:
  KOKKOS_INLINE_FUNCTION
  Real DensityCodeToCGS(const Real d) const {
    return d*density_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real PressureCGSToCode(const Real p) const {
    return p/pressure_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real PressureCodeToCGS(const Real p) const {
    return p*pressure_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real SpecificEintCodeToCGS(const Real eps) const {
    return eps*specific_eint_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real SpecificEintCGSToCode(const Real eps) const {
    return eps/specific_eint_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureCodeToCGS(const Real T) const {
    return T*temp_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureCGSToCode(const Real T) const {
    return T/temp_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real SafePositive(const Real x) const {
    return fmax(x, static_cast<Real>(1.0e-300));
  }

  KOKKOS_INLINE_FUNCTION
  Real SafeLog(const Real x) const {
    return log(SafePositive(x));
  }

  // Above the table's last temperature the weights extrapolate (wt1 > 1): the log
  // fields continue as the power law of the last interval, every bounded field holds
  // its top value.
  KOKKOS_INLINE_FUNCTION
  bool SahaLogField(const int iv) const {
    return (iv == saha_logpress) || (iv == saha_logeps) || (iv == saha_logcs2);
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaEvalFieldFromWeights(const int iv, const int ir, const int it,
                                const Real wr0, const Real wr1,
                                const Real wt0, const Real wt1) const {
    const Real w1 = SahaLogField(iv) ? wt1 : fmin(wt1, static_cast<Real>(1.0));
    const Real w0 = 1.0 - w1;
    return wr0*(w0*saha_table(iv, ir, it) + w1*saha_table(iv, ir, it + 1)) +
           wr1*(w0*saha_table(iv, ir + 1, it) + w1*saha_table(iv, ir + 1, it + 1));
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaFieldAtTempIndexFromRhoWeights(const int iv, const int ir,
                                          const Real wr0, const Real wr1,
                                          const int it) const {
    return wr0*saha_table(iv, ir, it) + wr1*saha_table(iv, ir + 1, it);
  }

  KOKKOS_INLINE_FUNCTION
  void NormalizeThermoFractions(ThermoState &state) const {
    state.xh2 = fmin(fmax(state.xh2, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    state.xion = fmin(fmax(state.xion, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    if (state.xh2 + state.xion > 1.0) {
      state.xion = fmax(static_cast<Real>(0.0), 1.0 - state.xh2);
    }

    state.xhe1 = fmin(fmax(state.xhe1, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    state.xhe2 = fmin(fmax(state.xhe2, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    if (state.xhe1 + state.xhe2 > 1.0) {
      state.xhe2 = fmax(static_cast<Real>(0.0), 1.0 - state.xhe1);
    }

    state.beta_rad =
        fmin(fmax(state.beta_rad, static_cast<Real>(0.0)), static_cast<Real>(1.0));
  }

  KOKKOS_INLINE_FUNCTION
  ThermoState SahaEvalThermoStateFromLogRhoTemp(const Real log_rho,
                                                const Real temp_code) const {
    ThermoState state;
    state.temperature = BoundSahaTemperatureCode(temp_code);
    const Real log_temp = SafeLog(TemperatureCodeToCGS(state.temperature));
    int ir, it;
    Real wr0, wr1, wt0, wt1;
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    SahaTempWeights(log_temp, it, wt0, wt1);
    state.pressure =
        PressureCGSToCode(exp(SahaEvalFieldFromWeights(saha_logpress, ir, it,
                                                       wr0, wr1, wt0, wt1)));
    state.cs2 = exp(SahaEvalFieldFromWeights(saha_logcs2, ir, it,
                                             wr0, wr1, wt0, wt1))/specific_eint_unit_cgs;
    state.gamma1 = SahaEvalFieldFromWeights(saha_gamma1, ir, it, wr0, wr1, wt0, wt1);
    state.gamma3m1 = SahaEvalFieldFromWeights(saha_gamma3m1, ir, it, wr0, wr1, wt0, wt1);
    state.xh2 = SahaEvalFieldFromWeights(saha_xh2, ir, it, wr0, wr1, wt0, wt1);
    state.xion = SahaEvalFieldFromWeights(saha_xion, ir, it, wr0, wr1, wt0, wt1);
    state.xhe1 = SahaEvalFieldFromWeights(saha_xhe1, ir, it, wr0, wr1, wt0, wt1);
    state.xhe2 = SahaEvalFieldFromWeights(saha_xhe2, ir, it, wr0, wr1, wt0, wt1);
    state.mu = SahaEvalFieldFromWeights(saha_mu, ir, it, wr0, wr1, wt0, wt1);
    state.beta_rad =
        SahaEvalFieldFromWeights(saha_beta_rad, ir, it, wr0, wr1, wt0, wt1);
    NormalizeThermoFractions(state);
    return state;
  }

  KOKKOS_INLINE_FUNCTION
  void SahaAbortIfOutOfBounds(const char *msg) const {
    Kokkos::abort(msg);
  }

  KOKKOS_INLINE_FUNCTION
  Real BoundSahaTemperatureCode(const Real T) const {
    if (saha_bounds_mode == SahaBoundsMode::error) {
      if ((T < saha_tmin_code) || (T > saha_tmax_code)) {
        SahaAbortIfOutOfBounds("Saha EOS temperature query is outside the loaded table.");
      }
      return fmin(fmax(T, saha_tmin_code), saha_tmax_code);
    }
    return fmax(T, saha_tmin_code);
  }

  KOKKOS_INLINE_FUNCTION
  void SahaRhoWeights(const Real log_rho, int &ir, Real &wr0, Real &wr1) const {
    if ((saha_bounds_mode == SahaBoundsMode::error) &&
        ((log_rho < saha_logrho_min) || (log_rho > saha_logrho_max))) {
      SahaAbortIfOutOfBounds("Saha EOS density query is outside the loaded table.");
    }
    const Real log_rho_clamped =
        fmin(fmax(log_rho, saha_logrho_min), saha_logrho_max);
    ir = static_cast<int>((log_rho_clamped - saha_logrho(0))*saha_inv_dlogrho);
    if (ir < 0) {
      ir = 0;
    } else if (ir > saha_nrho - 2) {
      ir = saha_nrho - 2;
    }
    wr1 = (log_rho_clamped - saha_logrho(ir))*saha_inv_dlogrho;
    wr0 = 1.0 - wr1;
  }

  KOKKOS_INLINE_FUNCTION
  void SahaTempWeights(const Real log_T, int &it, Real &wt0, Real &wt1) const {
    if ((saha_bounds_mode == SahaBoundsMode::error) &&
        ((log_T < saha_logtemp_min) || (log_T > saha_logtemp_max))) {
      SahaAbortIfOutOfBounds("Saha EOS temperature query is outside the loaded table.");
    }
    const Real log_t_bounded = (saha_bounds_mode == SahaBoundsMode::error)
        ? fmin(fmax(log_T, saha_logtemp_min), saha_logtemp_max)
        : fmax(log_T, saha_logtemp_min);
    it = static_cast<int>((log_t_bounded - saha_logtemp(0))*saha_inv_dlogtemp);
    if (it < 0) {
      it = 0;
    } else if (it > saha_ntemp - 2) {
      it = saha_ntemp - 2;
    }
    wt1 = (log_t_bounded - saha_logtemp(it))*saha_inv_dlogtemp;
    wt0 = 1.0 - wt1;
  }

  KOKKOS_INLINE_FUNCTION
  void SahaEpsWeights(const Real log_eps, int &ie, Real &we0, Real &we1) const {
    const Real log_eps_clamped =
        fmin(fmax(log_eps, saha_logeps_min), saha_logeps_max);
    const Real dlogeps = 1.0/saha_inv_dlogeps;
    ie = static_cast<int>((log_eps_clamped - saha_logeps_min)*saha_inv_dlogeps);
    if (ie < 0) {
      ie = 0;
    } else if (ie > saha_neps - 2) {
      ie = saha_neps - 2;
    }
    const Real log_eps_lo = saha_logeps_min + ie*dlogeps;
    we1 = (log_eps_clamped - log_eps_lo)*saha_inv_dlogeps;
    we0 = 1.0 - we1;
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaCacheEvalFieldFromWeights(const int iv, const int ir, const int ie,
                                     const Real wr0, const Real wr1,
                                     const Real we0, const Real we1) const {
    return wr0*(we0*saha_thermo_cache(iv, ir, ie) + we1*saha_thermo_cache(iv, ir,
        ie + 1)) +
           wr1*(we0*saha_thermo_cache(iv, ir + 1, ie) +
                we1*saha_thermo_cache(iv, ir + 1, ie + 1));
  }

  // Increment of a log field over the table's last temperature interval, interpolated
  // in rho as the forward evaluator interpolates the field itself.
  KOKKOS_INLINE_FUNCTION
  Real SahaAboveTopLogIncrement(const int iv, const int ir, const Real wr0,
                                const Real wr1) const {
    return SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, saha_ntemp - 1) -
           SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, saha_ntemp - 2);
  }

  // Factors from the top (wt1 = 1) to wt1 > 1 along the last temperature interval:
  // T scales as exp((wt1-1) dln T), a log field f as exp((wt1-1) dln f).
  KOKKOS_INLINE_FUNCTION
  Real SahaAboveTopTemperatureFactor(const Real wt1) const {
    return exp((wt1 - 1.0)*(saha_logtemp(saha_ntemp - 1) - saha_logtemp(saha_ntemp - 2)));
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaAboveTopFieldFactor(const int iv, const int ir, const Real wr0, const Real wr1,
                               const Real wt1) const {
    return exp((wt1 - 1.0)*SahaAboveTopLogIncrement(iv, ir, wr0, wr1));
  }

  // Returns true when, in clamp mode, log_eps lies above the rho-interpolated top of
  // the table. The inverse cache cannot continue past it: its eps axis ends at the
  // largest row top, and it stores linear T, p, cs2 against log eps. Along the last
  // temperature interval log eps is linear in log T (the forward continuation), so
  // wt1_hot > 1 is that interval's weight at log_eps, and ie, we0, we1 are the cache
  // weights AT the top: the continuation starts from the cache's own value there, so
  // every accessor is continuous where the query leaves the cache.
  KOKKOS_INLINE_FUNCTION
  bool SahaCacheWeightsFromLogRhoEps(const Real log_rho, const Real log_eps, int &ir,
                                     Real &wr0, Real &wr1, int &ie, Real &we0,
                                     Real &we1, Real &wt1_hot) const {
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    const Real logeps_min_local =
        SahaFieldAtTempIndexFromRhoWeights(saha_logeps, ir, wr0, wr1, 0);
    // Exact row-edge values are valid table targets. Only strictly smaller
    // energies should be treated as out of bounds.
    if (log_eps < logeps_min_local) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        SahaAbortIfOutOfBounds("Saha EOS inversion target is below the loaded table.");
      }
    }
    const Real logeps_max_local =
        SahaFieldAtTempIndexFromRhoWeights(saha_logeps, ir, wr0, wr1, saha_ntemp - 1);
    if (log_eps > logeps_max_local) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        SahaAbortIfOutOfBounds("Saha EOS inversion target is above the loaded table.");
      } else {
        const Real dv = SahaAboveTopLogIncrement(saha_logeps, ir, wr0, wr1);
        wt1_hot = 1.0 + ((dv > 0.0) ? ((log_eps - logeps_max_local)/dv) : 0.0);
        SahaEpsWeights(logeps_max_local, ie, we0, we1);
        return true;
      }
    }

    const Real log_eps_bounded = fmin(fmax(log_eps, logeps_min_local), logeps_max_local);
    SahaEpsWeights(log_eps_bounded, ie, we0, we1);
    return false;
  }

  // Above the top, T, p and cs2 continue as the power law of the last interval from
  // their cache values at the top, and the bounded fields hold those values.
  KOKKOS_INLINE_FUNCTION
  ThermoState SahaEvalThermoStateFromLogRhoEps(const Real log_rho,
                                               const Real log_eps) const {
    int ir, ie;
    Real wr0, wr1, we0, we1, wt1;
    const bool hot = SahaCacheWeightsFromLogRhoEps(log_rho, log_eps, ir, wr0, wr1, ie,
                                                   we0, we1, wt1);

    ThermoState state;
    state.temperature =
        SahaCacheEvalFieldFromWeights(saha_cache_temp, ir, ie, wr0, wr1, we0, we1);
    state.pressure =
        SahaCacheEvalFieldFromWeights(saha_cache_press, ir, ie, wr0, wr1, we0, we1);
    state.cs2 =
        SahaCacheEvalFieldFromWeights(saha_cache_cs2, ir, ie, wr0, wr1, we0, we1);
    state.gamma1 =
        SahaCacheEvalFieldFromWeights(saha_cache_gamma1, ir, ie, wr0, wr1, we0, we1);
    state.gamma3m1 =
        SahaCacheEvalFieldFromWeights(saha_cache_gamma3m1, ir, ie, wr0, wr1, we0, we1);
    state.xh2 =
        SahaCacheEvalFieldFromWeights(saha_cache_xh2, ir, ie, wr0, wr1, we0, we1);
    state.xion =
        SahaCacheEvalFieldFromWeights(saha_cache_xion, ir, ie, wr0, wr1, we0, we1);
    state.xhe1 =
        SahaCacheEvalFieldFromWeights(saha_cache_xhe1, ir, ie, wr0, wr1, we0, we1);
    state.xhe2 =
        SahaCacheEvalFieldFromWeights(saha_cache_xhe2, ir, ie, wr0, wr1, we0, we1);
    state.mu =
        SahaCacheEvalFieldFromWeights(saha_cache_mu, ir, ie, wr0, wr1, we0, we1);
    state.beta_rad =
        SahaCacheEvalFieldFromWeights(saha_cache_beta_rad, ir, ie, wr0, wr1, we0, we1);
    if (hot) {
      state.temperature *= SahaAboveTopTemperatureFactor(wt1);
      state.pressure *= SahaAboveTopFieldFactor(saha_logpress, ir, wr0, wr1, wt1);
      state.cs2 *= SahaAboveTopFieldFactor(saha_logcs2, ir, wr0, wr1, wt1);
    }
    NormalizeThermoFractions(state);
    return state;
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaEvalTemperatureFromLogRhoEps(const Real log_rho, const Real log_eps) const {
    int ir, ie;
    Real wr0, wr1, we0, we1, wt1;
    const bool hot = SahaCacheWeightsFromLogRhoEps(log_rho, log_eps, ir, wr0, wr1, ie,
                                                   we0, we1, wt1);
    const Real temp =
        SahaCacheEvalFieldFromWeights(saha_cache_temp, ir, ie, wr0, wr1, we0, we1);
    return hot ? temp*SahaAboveTopTemperatureFactor(wt1) : temp;
  }

  // Same T, plus dT/dln(eps) of the bilinear interpolant in the cell the query landed in
  // (on a query clamped at the cold edge, the slope of the edge cell); above the top,
  // T dln T/dln eps of the continuation.
  KOKKOS_INLINE_FUNCTION
  Real SahaEvalTemperatureFromLogRhoEps(const Real log_rho, const Real log_eps,
                                        Real &dtemp_dlogeps) const {
    int ir, ie;
    Real wr0, wr1, we0, we1, wt1;
    const bool hot = SahaCacheWeightsFromLogRhoEps(log_rho, log_eps, ir, wr0, wr1, ie,
                                                   we0, we1, wt1);
    const Real temp =
        SahaCacheEvalFieldFromWeights(saha_cache_temp, ir, ie, wr0, wr1, we0, we1);
    if (hot) {
      const Real temp_hot = temp*SahaAboveTopTemperatureFactor(wt1);
      const Real dv = SahaAboveTopLogIncrement(saha_logeps, ir, wr0, wr1);
      dtemp_dlogeps = (dv > 0.0)
          ? temp_hot*(saha_logtemp(saha_ntemp - 1) - saha_logtemp(saha_ntemp - 2))/dv
          : 0.0;
      return temp_hot;
    }
    dtemp_dlogeps =
        (wr0*(saha_thermo_cache(saha_cache_temp, ir, ie + 1) -
              saha_thermo_cache(saha_cache_temp, ir, ie)) +
         wr1*(saha_thermo_cache(saha_cache_temp, ir + 1, ie + 1) -
              saha_thermo_cache(saha_cache_temp, ir + 1, ie)))*saha_inv_dlogeps;
    return temp;
  }

  KOKKOS_INLINE_FUNCTION
  void SahaEvalPressureCs2FromLogRhoEps(const Real log_rho, const Real log_eps,
                                        Real &pressure, Real &cs2) const {
    int ir, ie;
    Real wr0, wr1, we0, we1, wt1;
    const bool hot = SahaCacheWeightsFromLogRhoEps(log_rho, log_eps, ir, wr0, wr1, ie,
                                                   we0, we1, wt1);
    pressure = SahaCacheEvalFieldFromWeights(saha_cache_press, ir, ie, wr0, wr1, we0,
        we1);
    cs2 = SahaCacheEvalFieldFromWeights(saha_cache_cs2, ir, ie, wr0, wr1, we0, we1);
    if (hot) {
      pressure *= SahaAboveTopFieldFactor(saha_logpress, ir, wr0, wr1, wt1);
      cs2 *= SahaAboveTopFieldFactor(saha_logcs2, ir, wr0, wr1, wt1);
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaEvalPressureFromLogRhoEps(const Real log_rho, const Real log_eps) const {
    int ir, ie;
    Real wr0, wr1, we0, we1, wt1;
    const bool hot = SahaCacheWeightsFromLogRhoEps(log_rho, log_eps, ir, wr0, wr1, ie,
                                                   we0, we1, wt1);
    const Real pressure =
        SahaCacheEvalFieldFromWeights(saha_cache_press, ir, ie, wr0, wr1, we0, we1);
    return hot ? pressure*SahaAboveTopFieldFactor(saha_logpress, ir, wr0, wr1, wt1)
               : pressure;
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaLogEpsCeilingFromLogRho(const Real log_rho) const {
    int ir;
    Real wr0, wr1;
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    return wr0*saha_logeps_ceil(ir) + wr1*saha_logeps_ceil(ir + 1);
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaLogEpsFloorFromLogRho(const Real log_rho) const {
    if (saha_floor_nrho <= 1) return saha_logeps_floor(0);
    if (log_rho <= saha_floor_logrho_min) return saha_logeps_floor(0);
    if (log_rho >= saha_floor_logrho_max) return saha_logeps_floor(saha_floor_nrho - 1);
    const Real x = (log_rho - saha_floor_logrho_min)*saha_floor_inv_dlogrho;
    const int ir = static_cast<int>(x);
    const int ilo = (ir < (saha_floor_nrho - 1)) ? ir : (saha_floor_nrho - 2);
    const Real wr1 = x - static_cast<Real>(ilo);
    const Real wr0 = 1.0 - wr1;
    return wr0*saha_logeps_floor(ilo) + wr1*saha_logeps_floor(ilo + 1);
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaFieldAtTempIndex(const int iv, const Real log_rho, const int it) const {
    int ir;
    Real wr0, wr1;
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    return SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, it);
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaEvalField(const int iv, const Real log_rho, const Real log_T) const {
    int ir, it;
    Real wr0, wr1, wt0, wt1;
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    SahaTempWeights(log_T, it, wt0, wt1);
    return SahaEvalFieldFromWeights(iv, ir, it, wr0, wr1, wt0, wt1);
  }

  KOKKOS_INLINE_FUNCTION
  Real SahaTemperatureFromMonotonicField(const int iv, const Real log_rho,
                                         const Real target) const {
    int ir;
    Real wr0, wr1;
    SahaRhoWeights(log_rho, ir, wr0, wr1);
    const Real vmin = SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, 0);
    // Hitting the table edge exactly is valid; clamp only for strictly
    // out-of-range targets.
    if (target < vmin) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        SahaAbortIfOutOfBounds("Saha EOS inversion target is below the loaded table.");
      }
      return TemperatureCGSToCode(exp(saha_logtemp(0)));
    }
    const Real vmax = SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1,
        saha_ntemp - 1);
    if (target > vmax) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        SahaAbortIfOutOfBounds("Saha EOS inversion target is above the loaded table.");
      }
      const Real vprev = SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1,
                                                            saha_ntemp - 2);
      const Real dv = vmax - vprev;
      const Real frac = (dv > 0.0) ? ((target - vmax)/dv) : 0.0;
      const Real log_t = saha_logtemp(saha_ntemp - 1) +
          frac*(saha_logtemp(saha_ntemp - 1) - saha_logtemp(saha_ntemp - 2));
      return TemperatureCGSToCode(exp(log_t));
    }

    int ilo = 0;
    int ihi = saha_ntemp - 1;
    Real vlo = vmin;
    Real vhi = vmax;
    while (ihi - ilo > 1) {
      const int imid = ilo + (ihi - ilo)/2;
      const Real vmid = SahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, imid);
      if (target <= vmid) {
        ihi = imid;
        vhi = vmid;
      } else {
        ilo = imid;
        vlo = vmid;
      }
    }

    const Real log_t_lo = saha_logtemp(ilo);
    const Real log_t_hi = saha_logtemp(ihi);
    const Real denom = vhi - vlo;
    const Real frac = (fabs(denom) > 0.0) ? ((target - vlo)/denom) : 0.0;
    const Real log_t = log_t_lo + frac*(log_t_hi - log_t_lo);
    return TemperatureCGSToCode(exp(log_t));
  }

  Real HostBoundSahaTemperatureCode(const Real T) const {
    if ((saha_bounds_mode == SahaBoundsMode::error) &&
        ((T < saha_tmin_code) || (T > saha_tmax_code))) {
      std::cout << "### FATAL ERROR in eos.hpp" << std::endl
                << "Saha EOS temperature query is outside the loaded table." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (saha_bounds_mode == SahaBoundsMode::error) {
      return std::min(std::max(T, saha_tmin_code), saha_tmax_code);
    }
    return std::max(T, saha_tmin_code);
  }

  void HostSahaRhoWeights(const Real log_rho, int &ir, Real &wr0, Real &wr1) const {
    if ((saha_bounds_mode == SahaBoundsMode::error) &&
        ((log_rho < saha_logrho_min) || (log_rho > saha_logrho_max))) {
      std::cout << "### FATAL ERROR in eos.hpp" << std::endl
                << "Saha EOS density query is outside the loaded table." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real log_rho_clamped = std::min(std::max(log_rho, saha_logrho_min),
        saha_logrho_max);
    ir = static_cast<int>((log_rho_clamped - saha_logrho_h(0))*saha_inv_dlogrho);
    if (ir < 0) {
      ir = 0;
    } else if (ir > saha_nrho - 2) {
      ir = saha_nrho - 2;
    }
    wr1 = (log_rho_clamped - saha_logrho_h(ir))*saha_inv_dlogrho;
    wr0 = 1.0 - wr1;
  }

  void HostSahaTempWeights(const Real log_T, int &it, Real &wt0, Real &wt1) const {
    if ((saha_bounds_mode == SahaBoundsMode::error) &&
        ((log_T < saha_logtemp_min) || (log_T > saha_logtemp_max))) {
      std::cout << "### FATAL ERROR in eos.hpp" << std::endl
                << "Saha EOS temperature query is outside the loaded table." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real log_t_bounded = (saha_bounds_mode == SahaBoundsMode::error)
        ? std::min(std::max(log_T, saha_logtemp_min), saha_logtemp_max)
        : std::max(log_T, saha_logtemp_min);
    it = static_cast<int>((log_t_bounded - saha_logtemp_h(0))*saha_inv_dlogtemp);
    if (it < 0) {
      it = 0;
    } else if (it > saha_ntemp - 2) {
      it = saha_ntemp - 2;
    }
    wt1 = (log_t_bounded - saha_logtemp_h(it))*saha_inv_dlogtemp;
    wt0 = 1.0 - wt1;
  }

  Real HostSahaEvalFieldFromWeights(const int iv, const int ir, const int it,
                                    const Real wr0, const Real wr1,
                                    const Real wt0, const Real wt1) const {
    const Real w1 = SahaLogField(iv) ? wt1 : std::min(wt1, static_cast<Real>(1.0));
    const Real w0 = 1.0 - w1;
    return wr0*(w0*saha_table_h(iv, ir, it) + w1*saha_table_h(iv, ir, it + 1)) +
           wr1*(w0*saha_table_h(iv, ir + 1, it) + w1*saha_table_h(iv, ir + 1, it + 1));
  }

  Real HostSahaFieldAtTempIndexFromRhoWeights(const int iv, const int ir,
                                              const Real wr0, const Real wr1,
                                              const int it) const {
    return wr0*saha_table_h(iv, ir, it) + wr1*saha_table_h(iv, ir + 1, it);
  }

  Real HostSahaFieldAtTempIndex(const int iv, const Real log_rho, const int it) const {
    int ir;
    Real wr0, wr1;
    HostSahaRhoWeights(log_rho, ir, wr0, wr1);
    return HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, it);
  }

  Real HostSahaEvalField(const int iv, const Real log_rho, const Real log_T) const {
    int ir, it;
    Real wr0, wr1, wt0, wt1;
    HostSahaRhoWeights(log_rho, ir, wr0, wr1);
    HostSahaTempWeights(log_T, it, wt0, wt1);
    return HostSahaEvalFieldFromWeights(iv, ir, it, wr0, wr1, wt0, wt1);
  }

  Real HostSahaLogEpsCeilingFromLogRho(const Real log_rho) const {
    int ir;
    Real wr0, wr1;
    HostSahaRhoWeights(log_rho, ir, wr0, wr1);
    return wr0*saha_logeps_ceil_h(ir) + wr1*saha_logeps_ceil_h(ir + 1);
  }

  Real HostSahaLogEpsFloorFromLogRho(const Real log_rho) const {
    if (saha_floor_nrho <= 1) return saha_logeps_floor_h(0);
    if (log_rho <= saha_floor_logrho_min) return saha_logeps_floor_h(0);
    if (log_rho >= saha_floor_logrho_max) return saha_logeps_floor_h(saha_floor_nrho - 1);
    const Real x = (log_rho - saha_floor_logrho_min)*saha_floor_inv_dlogrho;
    const int ir = static_cast<int>(x);
    const int ilo = (ir < (saha_floor_nrho - 1)) ? ir : (saha_floor_nrho - 2);
    const Real wr1 = x - static_cast<Real>(ilo);
    const Real wr0 = 1.0 - wr1;
    return wr0*saha_logeps_floor_h(ilo) + wr1*saha_logeps_floor_h(ilo + 1);
  }

  Real HostSahaTemperatureFromMonotonicField(const int iv, const Real log_rho,
                                             const Real target) const {
    int ir;
    Real wr0, wr1;
    HostSahaRhoWeights(log_rho, ir, wr0, wr1);
    const Real vmin = HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, 0);
    // Exact host-side edge values are valid. This matters for floor states
    // built from the precomputed logeps floor cache and then re-inverted
    // through the primary monotonic table.
    if (target < vmin) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        std::cout << "### FATAL ERROR in eos.hpp" << std::endl
                  << "Saha EOS inversion target is below the loaded table." << std::endl
                  << "  iv = " << iv << std::endl
                  << "  log_rho = " << log_rho << std::endl
                  << "  target = " << target << std::endl
                  << "  vmin = " << vmin << std::endl
                  << "  vmax = "
                  << HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1,
                      saha_ntemp - 1)
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return TemperatureCGSToCode(exp(saha_logtemp_h(0)));
    }
    const Real vmax =
        HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, saha_ntemp - 1);
    if (target > vmax) {
      if (saha_bounds_mode == SahaBoundsMode::error) {
        std::cout << "### FATAL ERROR in eos.hpp" << std::endl
                  << "Saha EOS inversion target is above the loaded table." << std::endl
                  << "  iv = " << iv << std::endl
                  << "  log_rho = " << log_rho << std::endl
                  << "  target = " << target << std::endl
                  << "  vmin = " << vmin << std::endl
                  << "  vmax = " << vmax << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real vprev = HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1,
                                                                saha_ntemp - 2);
      const Real dv = vmax - vprev;
      const Real frac = (dv > 0.0) ? ((target - vmax)/dv) : 0.0;
      const Real log_t = saha_logtemp_h(saha_ntemp - 1) +
          frac*(saha_logtemp_h(saha_ntemp - 1) - saha_logtemp_h(saha_ntemp - 2));
      return TemperatureCGSToCode(exp(log_t));
    }

    int ilo = 0;
    int ihi = saha_ntemp - 1;
    Real vlo = vmin;
    Real vhi = vmax;
    while (ihi - ilo > 1) {
      const int imid = ilo + (ihi - ilo)/2;
      const Real vmid = HostSahaFieldAtTempIndexFromRhoWeights(iv, ir, wr0, wr1, imid);
      if (target <= vmid) {
        ihi = imid;
        vhi = vmid;
      } else {
        ilo = imid;
        vlo = vmid;
      }
    }

    const Real log_t_lo = saha_logtemp_h(ilo);
    const Real log_t_hi = saha_logtemp_h(ihi);
    const Real denom = vhi - vlo;
    const Real frac = (fabs(denom) > 0.0) ? ((target - vlo)/denom) : 0.0;
    const Real log_t = log_t_lo + frac*(log_t_hi - log_t_lo);
    return TemperatureCGSToCode(exp(log_t));
  }

  Real HostSpecificEintFromRhoT(const Real d, const Real T) const {
    const Real rho_cgs = DensityCodeToCGS(SafePositive(d));
    const Real temp_cgs = TemperatureCodeToCGS(HostBoundSahaTemperatureCode(T));
    const Real log_eps = HostSahaEvalField(saha_logeps, SafeLog(rho_cgs),
        SafeLog(temp_cgs));
    return SpecificEintCGSToCode(exp(log_eps));
  }
};

KOKKOS_INLINE_FUNCTION
Real EOS_Data::PressureFromRhoT(const Real d, const Real T) const {
  return EvalThermoStateFromRhoT(d, T).pressure;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::SpecificEintFromRhoT(const Real d, const Real T) const {
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(SafePositive(d));
    const Real temp_cgs = TemperatureCodeToCGS(BoundSahaTemperatureCode(T));
    const Real log_eps = SahaEvalField(saha_logeps, SafeLog(rho_cgs), SafeLog(temp_cgs));
    return SpecificEintCGSToCode(exp(log_eps));
  }
  if (hydro_eos == HydroEOSModel::gamma_law) {
    return T/(gamma - 1.0);
  }
  return 0.0;
}

// Temperature alone: the same weights and the same cache field as
// EvalThermoStateFromRhoEint(d, eint).temperature, without the other ten fields.
KOKKOS_INLINE_FUNCTION
Real EOS_Data::TemperatureFromRhoEint(const Real d, const Real eint) const {
  const Real dens = SafePositive(d);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    return SahaEvalTemperatureFromLogRhoEps(SafeLog(rho_cgs), SafeLog(eps_cgs));
  }
  if (hydro_eos == HydroEOSModel::isothermal) return 0.0;
  return (gamma - 1.0)*eint/dens;
}

// The same T, and its derivative in the internal energy density at fixed density.
KOKKOS_INLINE_FUNCTION
Real EOS_Data::TemperatureFromRhoEint(const Real d, const Real eint, Real &dtemp_deint)
    const {
  const Real dens = SafePositive(d);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    Real dtemp_dlogeps;
    const Real temp = SahaEvalTemperatureFromLogRhoEps(SafeLog(rho_cgs), SafeLog(eps_cgs),
                                                       dtemp_dlogeps);
    dtemp_deint = (eint > 0.0) ? dtemp_dlogeps/eint : 0.0;
    return temp;
  }
  if (hydro_eos == HydroEOSModel::isothermal) {
    dtemp_deint = 0.0;
    return 0.0;
  }
  dtemp_deint = (gamma - 1.0)/dens;
  return (gamma - 1.0)*eint/dens;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::TemperatureFromRhoP(const Real d, const Real p) const {
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(SafePositive(d));
    const Real p_cgs = PressureCodeToCGS(SafePositive(p));
    const Real t_code = SahaTemperatureFromMonotonicField(saha_logpress, SafeLog(rho_cgs),
                                                          SafeLog(p_cgs));
    return BoundSahaTemperatureCode(t_code);
  }
  if (hydro_eos == HydroEOSModel::gamma_law) {
    return p/SafePositive(d);
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::SpecificEintFromRhoP(const Real d, const Real p) const {
  return SpecificEintFromRhoT(d, TemperatureFromRhoP(d, p));
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::InternalEnergyDensityFromRhoP(const Real d, const Real p) const {
  return d*SpecificEintFromRhoP(d, p);
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::PressureFromRhoEint(const Real d, const Real eint) const {
  if (UsesTabulatedLTE()) {
    const Real dens = SafePositive(d);
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    return SahaEvalPressureFromLogRhoEps(SafeLog(rho_cgs), SafeLog(eps_cgs));
  }
  return EvalThermoStateFromRhoEint(d, eint).pressure;
}

KOKKOS_INLINE_FUNCTION
void EOS_Data::EvalPressureCs2FromRhoEint(const Real d, const Real eint,
                                          Real &pressure, Real &cs2) const {
  const Real dens = SafePositive(d);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    SahaEvalPressureCs2FromLogRhoEps(SafeLog(rho_cgs), SafeLog(eps_cgs), pressure, cs2);
    return;
  }
  const auto thermo = EvalThermoStateFromRhoEint(dens, eint);
  pressure = thermo.pressure;
  cs2 = thermo.cs2;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::HydroSoundSpeed2FromRhoEint(const Real d, const Real eint) const {
  Real pressure, cs2;
  EvalPressureCs2FromRhoEint(d, eint, pressure, cs2);
  return cs2;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::FastMagnetosonicSpeedFromSoundSpeed2(const Real d, const Real cs2,
                                                    const Real bx, const Real by,
                                                    const Real bz) const {
  const Real asq = d*cs2;
  const Real ct2 = by*by + bz*bz;
  const Real qsq = bx*bx + ct2 + asq;
  const Real tmp = bx*bx + ct2 - asq;
  return sqrt(0.5*(qsq + sqrt(tmp*tmp + 4.0*asq*ct2))/d);
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::IonizationFractionFromRhoT(const Real d, const Real T) const {
  return EvalThermoStateFromRhoT(d, T).xion;
}

KOKKOS_INLINE_FUNCTION
EOS_Data::ThermoState EOS_Data::EvalThermoStateFromRhoT(const Real d,
    const Real T) const {
  ThermoState state;
  const Real dens = SafePositive(d);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    return SahaEvalThermoStateFromLogRhoTemp(SafeLog(rho_cgs), T);
  }
  if (hydro_eos == HydroEOSModel::isothermal) {
    state.temperature = T;
    state.pressure = dens*SQR(iso_cs);
    state.cs2 = SQR(iso_cs);
    state.gamma1 = 1.0;
    state.gamma3m1 = 0.0;
    state.xh2 = 0.0;
    state.xion = 0.0;
    state.xhe1 = 0.0;
    state.xhe2 = 0.0;
    state.mu = 0.0;
    state.beta_rad = 0.0;
    return state;
  }
  state.temperature = T;
  state.pressure = dens*T;
  state.cs2 = gamma*T;
  state.gamma1 = gamma;
  state.gamma3m1 = gamma - 1.0;
  state.xh2 = 0.0;
  state.xion = 0.0;
  state.xhe1 = 0.0;
  state.xhe2 = 0.0;
  state.mu = 0.0;
  state.beta_rad = 0.0;
  return state;
}

KOKKOS_INLINE_FUNCTION
EOS_Data::ThermoState EOS_Data::EvalThermoStateFromRhoEint(const Real d,
    const Real eint) const {
  ThermoState state;
  const Real dens = SafePositive(d);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    return SahaEvalThermoStateFromLogRhoEps(SafeLog(rho_cgs), SafeLog(eps_cgs));
  }
  if (hydro_eos == HydroEOSModel::isothermal) {
    state.temperature = 0.0;
    state.pressure = dens*SQR(iso_cs);
    state.cs2 = SQR(iso_cs);
    state.gamma1 = 1.0;
    state.gamma3m1 = 0.0;
    state.xh2 = 0.0;
    state.xion = 0.0;
    state.xhe1 = 0.0;
    state.xhe2 = 0.0;
    state.mu = 0.0;
    state.beta_rad = 0.0;
    return state;
  }
  state.temperature = (gamma - 1.0)*eint/dens;
  state.pressure = IdealGasPressure(eint);
  state.cs2 = gamma*state.pressure/dens;
  state.gamma1 = gamma;
  state.gamma3m1 = gamma - 1.0;
  state.xh2 = 0.0;
  state.xion = 0.0;
  state.xhe1 = 0.0;
  state.xhe2 = 0.0;
  state.mu = 0.0;
  state.beta_rad = 0.0;
  return state;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::HydroInternalEnergyDensityFloor(const Real d) const {
  if (UsesTabulatedLTE()) {
    const Real dens = SafePositive(d);
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real log_eps = SahaLogEpsFloorFromLogRho(SafeLog(rho_cgs));
    return dens*SpecificEintCGSToCode(exp(log_eps));
  }
  if (hydro_eos == HydroEOSModel::gamma_law) {
    const Real gm1 = gamma - 1.0;
    const Real eint_floor = pfloor/gm1;
    const Real entropy_floor = (sfloor > 0.0) ? (sfloor*pow(d, gamma)/gm1) : 0.0;
    return fmax(eint_floor, entropy_floor);
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::HydroInternalEnergyDensityCeiling(const Real d) const {
  if (!(use_e) || !(cs_ceil > 0.0)) {
    return 1.0e99;
  }
  const Real dens = SafePositive(d);
  Real eint_ceil = 1.0e99;
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real log_eps = SahaLogEpsCeilingFromLogRho(SafeLog(rho_cgs));
    eint_ceil = dens*SpecificEintCGSToCode(exp(log_eps));
  } else if (hydro_eos == HydroEOSModel::gamma_law) {
    eint_ceil = dens*SQR(cs_ceil)/(gamma*(gamma - 1.0));
  }
  return fmax(eint_ceil, HydroInternalEnergyDensityFloor(dens));
}

KOKKOS_INLINE_FUNCTION
Real EOS_Data::ClampHydroInternalEnergyDensity(const Real d, const Real eint) const {
  if (!(use_e) || !(cs_ceil > 0.0)) {
    return eint;
  }
  return fmin(eint, HydroInternalEnergyDensityCeiling(d));
}

inline Real EOS_Data::HostHydroInternalEnergyDensityFloor(const Real d) const {
  if (UsesTabulatedLTE()) {
    const Real dens = SafePositive(d);
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real log_eps = HostSahaLogEpsFloorFromLogRho(SafeLog(rho_cgs));
    return dens*SpecificEintCGSToCode(exp(log_eps));
  }
  return HydroInternalEnergyDensityFloor(d);
}

inline Real EOS_Data::HostHydroInternalEnergyDensityCeiling(const Real d) const {
  if (!(use_e) || !(cs_ceil > 0.0)) {
    return 1.0e99;
  }
  const Real dens = SafePositive(d);
  Real eint_ceil = 1.0e99;
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real log_eps = HostSahaLogEpsCeilingFromLogRho(SafeLog(rho_cgs));
    eint_ceil = dens*SpecificEintCGSToCode(exp(log_eps));
  } else if (hydro_eos == HydroEOSModel::gamma_law) {
    eint_ceil = dens*SQR(cs_ceil)/(gamma*(gamma - 1.0));
  }
  return std::max(eint_ceil, HostHydroInternalEnergyDensityFloor(dens));
}

inline Real EOS_Data::HostClampHydroInternalEnergyDensity(const Real d,
                                                          const Real eint) const {
  if (!(use_e) || !(cs_ceil > 0.0)) {
    return eint;
  }
  return std::min(eint, HostHydroInternalEnergyDensityCeiling(d));
}

inline Real EOS_Data::HostPressureFromRhoEint(const Real d, const Real eint) const {
  if (UsesTabulatedLTE()) {
    const Real dens = SafePositive(d);
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real eps_cgs = SpecificEintCodeToCGS(SafePositive(eint/dens));
    const Real t_code =
        HostSahaTemperatureFromMonotonicField(saha_logeps, SafeLog(rho_cgs),
                                              SafeLog(eps_cgs));
    const Real log_press =
        HostSahaEvalField(saha_logpress, SafeLog(rho_cgs),
                          SafeLog(TemperatureCodeToCGS(t_code)));
    return PressureCGSToCode(exp(log_press));
  }
  return PressureFromRhoEint(d, eint);
}

inline Real EOS_Data::HostInternalEnergyDensityFromRhoP(const Real d,
    const Real p) const {
  const Real dens = SafePositive(d);
  const Real pres = SafePositive(p);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real p_cgs = PressureCodeToCGS(pres);
    const Real t_code =
        HostSahaTemperatureFromMonotonicField(saha_logpress, SafeLog(rho_cgs),
                                              SafeLog(p_cgs));
    return dens * HostSpecificEintFromRhoT(dens, t_code);
  }
  return InternalEnergyDensityFromRhoP(dens, pres);
}

inline Real EOS_Data::HostGamma1FromRhoP(const Real d, const Real p) const {
  const Real dens = SafePositive(d);
  const Real pres = SafePositive(p);
  if (UsesTabulatedLTE()) {
    const Real rho_cgs = DensityCodeToCGS(dens);
    const Real p_cgs = PressureCodeToCGS(pres);
    const Real t_code =
        HostSahaTemperatureFromMonotonicField(saha_logpress, SafeLog(rho_cgs),
                                              SafeLog(p_cgs));
    return HostSahaEvalField(saha_gamma1, SafeLog(rho_cgs),
                             SafeLog(TemperatureCodeToCGS(t_code)));
  }
  if (hydro_eos == HydroEOSModel::gamma_law) {
    return gamma;
  }
  if (hydro_eos == HydroEOSModel::isothermal) {
    return 1.0;
  }
  return 0.0;
}

inline Real EOS_Data::HostDensityInActiveDomain(const Real d) const {
  const Real dens = SafePositive(d);
  if (!(UsesTabulatedLTE())) {
    return true;
  }
  const Real rho_cgs = DensityCodeToCGS(dens);
  const Real log_rho = SafeLog(rho_cgs);
  return (log_rho >= saha_logrho_min) && (log_rho <= saha_logrho_max);
}

inline Real EOS_Data::HostTabulatedPressureFloor(const Real d) const {
  const Real dens = SafePositive(d);
  if (!(UsesTabulatedLTE())) {
    return 0.0;
  }
  const Real rho_cgs = DensityCodeToCGS(dens);
  const Real log_press =
      HostSahaFieldAtTempIndex(saha_logpress, SafeLog(rho_cgs), 0);
  return PressureCGSToCode(exp(log_press));
}

inline Real EOS_Data::HostTabulatedPressureCeiling(const Real d) const {
  const Real dens = SafePositive(d);
  if (!(UsesTabulatedLTE())) {
    return 1.0e99;
  }
  const Real rho_cgs = DensityCodeToCGS(dens);
  const Real log_press =
      HostSahaFieldAtTempIndex(saha_logpress, SafeLog(rho_cgs), saha_ntemp - 1);
  return PressureCGSToCode(exp(log_press));
}

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//! \brief Abstract base class for EOS.

class EquationOfState {
 public:
  EquationOfState(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  virtual ~EquationOfState() = default;

  MeshBlockPack* pmy_pack;
  EOS_Data eos_data;

  // Limit a conversion launch to MeshBlocks carrying one specific user boundary face.
  // The default (-1) preserves ordinary pack-wide conversion behavior.
  void SetUserBoundaryFaceFilter(const int face) { user_boundary_face_filter_ = face; }
  void ClearUserBoundaryFaceFilter() { user_boundary_face_filter_ = -1; }
  int UserBoundaryFaceFilter() const { return user_boundary_face_filter_; }

  // virtual functions to convert cons to prim in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes
  virtual void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);
  virtual void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);

  // virtual functions to convert prim to cons in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes.
  virtual void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);
  virtual void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku);

  //! \brief Empty whatever per-cell solver history this policy carries into the next
  //! inversion, so that the recovery after it is the cold one.  Returns true if the
  //! policy keeps any (only IdealGRMHD does; see c2p_mu_cache below).  The callers are
  //! RestartOutput::LoadOutputData, which has to hand a restart the same starting point
  //! the writing run continues from, and the restart's own recovery.
  virtual bool ResetC2PWarmStart() { return false; }

 private:
  int user_boundary_face_filter_ = -1;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//! \brief Derived class for isothermal EOS in nonrelativistic Hydro

class IsothermalHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealHydro
//! \brief Derived class for ideal gas EOS in nonrelativistic hydro

class IdealHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class SahaTableHydro
//! \brief Derived class for a tabulated hydrogen LTE/Saha EOS in nonrelativistic hydro

class SahaTableHydro : public EquationOfState {
 public:
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  SahaTableHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class SahaTableMHD
//! \brief Derived class for a tabulated hydrogen LTE/Saha EOS in nonrelativistic MHD

class SahaTableMHD : public EquationOfState {
 public:
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  SahaTableMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class LTETableHydro
//! \brief Derived class for a tabulated H+He LTE EOS in nonrelativistic hydro

class LTETableHydro : public EquationOfState {
 public:
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  LTETableHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class LTETableMHD
//! \brief Derived class for a tabulated H+He LTE EOS in nonrelativistic MHD

class LTETableMHD : public EquationOfState {
 public:
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  LTETableMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealSRHydro
//! \brief Derived class for ideal gas EOS in special relativistic Hydro

class IdealSRHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealSRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealGRHydro
//! \brief Derived class for ideal gas EOS in general relativistic Hydro

class IdealGRHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalMHD
//! \brief Derived class for isothermal EOS in nonrelativistic MHD

class IsothermalMHD : public EquationOfState {
 public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealMHD
//! \brief Derived class for ideal gas EOS in nonrelativistic MHD

class IdealMHD : public EquationOfState {
 public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealSRMHD
//! \brief Derived class for ideal gas EOS in special relativistic MHD

class IdealSRMHD : public EquationOfState {
 public:
  // Following suppress warnings that hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealSRMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealGRMHD
//! \brief Derived class for ideal gas EOS in general relativistic MHD

class IdealGRMHD : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
  bool ResetC2PWarmStart() override;
  template <bool only_testfloors, bool track_event_counters, bool apply_sigma_ceiling>
  void ConsToPrimImpl(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                      DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                      const int il, const int iu, const int jl, const int ju,
                      const int kl, const int ku);

 private:
  bool track_event_counters_;
  DualArray1D<int> event_counters_;
  // Cached Kastaun root mu of the last authoritative (u0/w0) inversion of each cell,
  // used to warm start the next one, as one cell-centered component (m,0,k,j,i).  Zero
  // means "no guess", which is both the initial state and what the cache is reset to
  // whenever the mesh topology changes (after a remesh the local block index m addresses
  // a DIFFERENT block, so a retained entry is another cell's root), except after a pure
  // rebalance, which carries every entry with its block (CarriedC2PHistory).
  // Dimensioned with nmb_maxperrank, and grown before a pure rebalance that needs more.
  //
  // The warm start does NOT leave the answer unchanged.  Both paths stop at
  // |f(mu)| < 1e-12 with |f'| = O(1), so the accepted root is a function of the bracket:
  // measured over 2e5 random states, feeding back the exact previous root moves the
  // converged mu by a median 2e-14 and up to 2e-11 relative (1e-10 in the recovered
  // primitives) against the cold solve -- four orders above one ULP.  The cache is
  // therefore run history.  The ordinary checkpoint does not carry it: a restarted run
  // cold starts, which is why the run that WRITES such a checkpoint empties the cache
  // and repeats its own recovery cold at that point (ResetC2PWarmStart,
  // RestartOutput::LoadOutputData) -- without that the two trajectories part at the root
  // find's tolerance and amplify from there.  This affects IdealGRMHD decks only;
  // IdealSRMHD never
  // allocates the cache and
  // the dynamical-GR path uses PrimitiveSolverHydro, which has no such cache.
  DvceArray5D<Real> c2p_mu_cache;
  // Mesh topology the cache's contents belong to; a mismatch invalidates it.  Initialised
  // here as well as in IdealGRMHD's constructor: every other policy inherits the member
  // and none of them ever assigns it.
  std::uint64_t c2p_mu_cache_topology_version = 0;
};

//----------------------------------------------------------------------------------------
//! \class NoOpDynGRMHD
//! \brief Derived class for no-op EOS in dynamical GRMHD

class NoOpDynGRMHD : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  NoOpDynGRMHD(MeshBlockPack *pp, ParameterInput *pin);
};

#endif // EOS_EOS_HPP_
