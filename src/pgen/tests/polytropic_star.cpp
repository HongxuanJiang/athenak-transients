//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file polytropic_star.cpp
//! \brief Problem generator for collapse of a Lane-Emden polytropic sphere with AMR.
//!
//! Follows the same structure as be_collapse.cpp: self-gravity via multigrid Poisson,
//! Jeans-style user AMR criterion, and zero initial velocity field. The only physics
//! difference is the initial density profile: here it is a numerically solved
//! Lane-Emden polytrope with index n and finite surface at the first zero xi_1.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

namespace {

// AMR parameter (set from input in pgen, read in refinement function)
Real njeans_threshold;
Real iso_cs_global;
Real poly_n_global;
Real poly_k_global;
Real rho_floor_global;
Real four_pi_G_global;
bool relax_enable_global;
Real relax_tau_global;
Real relax_t_end_global;
Real relax_radius_global;
Real relax_x_center_global;
Real relax_y_center_global;
Real relax_z_center_global;

struct LaneEmdenProfile {
  std::vector<Real> xi;
  std::vector<Real> theta;
  std::vector<Real> dtheta;
  Real xi1;
};

enum class StellarStructureMode {
  legacy_polytrope,
  eos_balanced,
};

struct StellarRadialProfile {
  std::vector<Real> radius;
  std::vector<Real> density;
  std::vector<Real> pressure;
  Real rho_central = 0.0;
  Real p_central = 0.0;
  Real mass_at_surface = 0.0;
  Real pressure_at_surface = 0.0;
  Real surface_radius = 0.0;
  bool reached_target_surface = false;
};

inline Real ThetaPow(const Real theta, const Real n) {
  return (theta > 0.0) ? std::pow(theta, n) : 0.0;
}

inline void LaneEmdenRHS(const Real xi, const Real theta, const Real dtheta,
                         const Real n, Real &dtheta_dxi, Real &ddtheta_dxi) {
  dtheta_dxi = dtheta;
  // Start integration away from xi=0, so this guard is only for safety.
  if (xi <= 0.0) {
    ddtheta_dxi = -1.0/3.0;
  } else {
    ddtheta_dxi = -2.0*dtheta/xi - ThetaPow(theta, n);
  }
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteValue(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  Real t2 = t*t;
  Real t3 = t2*t;
  Real h00 = (2.0*t3 - 3.0*t2 + 1.0);
  Real h10 = (t3 - 2.0*t2 + t);
  Real h01 = (-2.0*t3 + 3.0*t2);
  Real h11 = (t3 - t2);
  return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1;
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteSlope(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  if (h <= 0.0) return m0;
  Real t2 = t*t;
  Real dh00 = (6.0*t2 - 6.0*t);
  Real dh10 = (3.0*t2 - 4.0*t + 1.0);
  Real dh01 = (-6.0*t2 + 6.0*t);
  Real dh11 = (3.0*t2 - 2.0*t);
  Real dtheta_dt = dh00*y0 + dh10*h*m0 + dh01*y1 + dh11*h*m1;
  return dtheta_dt / h;
}

inline void RK4Step(const Real xi, const Real h, const Real theta, const Real dtheta,
                    const Real n, Real &theta_out, Real &dtheta_out) {
  Real k1_t, k1_dt;
  LaneEmdenRHS(xi, theta, dtheta, n, k1_t, k1_dt);

  Real t2 = theta + 0.5*h*k1_t;
  Real dt2 = dtheta + 0.5*h*k1_dt;
  Real k2_t, k2_dt;
  LaneEmdenRHS(xi + 0.5*h, t2, dt2, n, k2_t, k2_dt);

  Real t3 = theta + 0.5*h*k2_t;
  Real dt3 = dtheta + 0.5*h*k2_dt;
  Real k3_t, k3_dt;
  LaneEmdenRHS(xi + 0.5*h, t3, dt3, n, k3_t, k3_dt);

  Real t4 = theta + h*k3_t;
  Real dt4 = dtheta + h*k3_dt;
  Real k4_t, k4_dt;
  LaneEmdenRHS(xi + h, t4, dt4, n, k4_t, k4_dt);

  theta_out = theta + (h/6.0)*(k1_t + 2.0*k2_t + 2.0*k3_t + k4_t);
  dtheta_out = dtheta + (h/6.0)*(k1_dt + 2.0*k2_dt + 2.0*k3_dt + k4_dt);
}

bool SolveLaneEmden(const Real n, LaneEmdenProfile &profile) {
  constexpr Real xi_max = 2.0e3;
  constexpr int max_steps = 5000000;
  constexpr Real atol = 1.0e-12;
  constexpr Real rtol = 1.0e-10;
  constexpr Real safety = 0.9;
  constexpr Real fac_min = 0.2;
  constexpr Real fac_max = 2.0;
  constexpr Real h_min = 1.0e-10;
  constexpr Real h_max = 5.0e-2;
  constexpr Real xi0 = 1.0e-6;

  profile.xi.clear();
  profile.theta.clear();
  profile.dtheta.clear();
  profile.xi.push_back(0.0);
  profile.theta.push_back(1.0);
  profile.dtheta.push_back(0.0);

  Real xi = xi0;
  // Regular center expansion of Lane-Emden solution.
  Real theta = 1.0 - SQR(xi)/6.0 + n*SQR(SQR(xi))/120.0;
  Real dtheta = -xi/3.0 + n*xi*SQR(xi)/30.0;

  profile.xi.push_back(xi);
  profile.theta.push_back(theta);
  profile.dtheta.push_back(dtheta);

  Real h = 1.0e-4;

  for (int step = 0; step < max_steps; ++step) {
    if (xi >= xi_max) break;
    if (h < h_min) h = h_min;
    if (h > h_max) h = h_max;
    if (xi + h > xi_max) h = xi_max - xi;
    if (h <= 0.0) break;

    Real theta_big, dtheta_big;
    RK4Step(xi, h, theta, dtheta, n, theta_big, dtheta_big);

    Real theta_half, dtheta_half;
    RK4Step(xi, 0.5*h, theta, dtheta, n, theta_half, dtheta_half);
    Real theta_half2, dtheta_half2;
    RK4Step(xi + 0.5*h, 0.5*h, theta_half, dtheta_half, n, theta_half2, dtheta_half2);

    if (!std::isfinite(theta_big) || !std::isfinite(dtheta_big) ||
        !std::isfinite(theta_half2) || !std::isfinite(dtheta_half2)) {
      return false;
    }

    Real sc_t = atol + rtol*std::fmax(std::abs(theta_half2), std::abs(theta_big));
    Real sc_dt = atol + rtol*std::fmax(std::abs(dtheta_half2), std::abs(dtheta_big));
    Real err_t = std::abs(theta_half2 - theta_big) / (15.0*sc_t);
    Real err_dt = std::abs(dtheta_half2 - dtheta_big) / (15.0*sc_dt);
    Real err = std::fmax(err_t, err_dt);

    if (err <= 1.0) {
      Real xi_next = xi + h;
      Real theta_next = theta_half2;
      Real dtheta_next = dtheta_half2;

      if (theta_next <= 0.0) {
        Real tlo = 0.0;
        Real thi = 1.0;
        for (int it = 0; it < 80; ++it) {
          Real tm = 0.5*(tlo + thi);
          Real thm = CubicHermiteValue(theta, dtheta, theta_next, dtheta_next, h, tm);
          if (thm > 0.0) {
            tlo = tm;
          } else {
            thi = tm;
          }
        }
        Real troot = 0.5*(tlo + thi);
        profile.xi1 = xi + troot*h;
        profile.xi.push_back(profile.xi1);
        profile.theta.push_back(0.0);
        profile.dtheta.push_back(
            CubicHermiteSlope(theta, dtheta, theta_next, dtheta_next, h, troot));
        return true;
      }

      profile.xi.push_back(xi_next);
      profile.theta.push_back(theta_next);
      profile.dtheta.push_back(dtheta_next);
      xi = xi_next;
      theta = theta_next;
      dtheta = dtheta_next;
    }

    Real fac = fac_max;
    if (err > 1.0e-30) {
      fac = safety*std::pow(1.0/err, 0.2);
      fac = std::fmax(fac_min, std::fmin(fac_max, fac));
    }
    h = std::fmax(h_min, std::fmin(h_max, h*fac));
  }

  return false;
}

KOKKOS_INLINE_FUNCTION
Real LaneEmdenTheta(const Real xi, const DvceArray1D<Real> xi_tab,
                    const DvceArray1D<Real> theta_tab,
                    const DvceArray1D<Real> dtheta_tab, const int npts) {
  if (xi <= 0.0) return 1.0;
  if (xi >= xi_tab(npts - 1)) return 0.0;

  int lo = 0;
  int hi = npts - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (xi_tab(mid) <= xi) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  Real x0 = xi_tab(lo);
  Real x1 = xi_tab(hi);
  if (x1 <= x0) return Kokkos::fmax(theta_tab(lo), 0.0);

  Real frac = (xi - x0) / (x1 - x0);
  Real theta = CubicHermiteValue(theta_tab(lo), dtheta_tab(lo),
                                 theta_tab(hi), dtheta_tab(hi),
                                 x1 - x0, frac);
  return Kokkos::fmax(theta, 0.0);
}

const char *StellarStructureModeName(const StellarStructureMode mode) {
  switch (mode) {
    case StellarStructureMode::legacy_polytrope:
      return "legacy_polytrope";
    case StellarStructureMode::eos_balanced:
      return "eos_balanced";
  }
  return "unknown";
}

bool ParseStellarStructureMode(const std::string &mode_name, StellarStructureMode &mode) {
  if (mode_name == "legacy_polytrope") {
    mode = StellarStructureMode::legacy_polytrope;
    return true;
  }
  if (mode_name == "eos_balanced") {
    mode = StellarStructureMode::eos_balanced;
    return true;
  }
  return false;
}

bool EOSGamma1FromRhoP(const EOS_Data &eos, const Real rho, const Real p, Real &gamma1) {
  if (!(rho > 0.0) || !(p > 0.0)) return false;
  if (eos.UsesTabulatedLTE()) {
    gamma1 = eos.HostGamma1FromRhoP(rho, p);
  } else if (eos.is_gamma_law) {
    gamma1 = eos.gamma;
  } else {
    return false;
  }
  return std::isfinite(gamma1) && (gamma1 > 0.0);
}

bool EOSPressureInActiveDomain(const EOS_Data &eos, const Real rho, const Real p) {
  if (!(rho > 0.0) || !(p > 0.0) || !std::isfinite(rho) || !std::isfinite(p)) {
    return false;
  }
  if (!(eos.UsesTabulatedLTE())) {
    return true;
  }
  if (!(eos.HostDensityInActiveDomain(rho))) {
    return false;
  }
  const Real p_floor = eos.HostTabulatedPressureFloor(rho);
  const Real p_ceil = eos.HostTabulatedPressureCeiling(rho);
  return (p >= p_floor) && (p <= p_ceil);
}

bool EOSBalancedRHS(const EOS_Data &eos, const Real grav_const, const Real r,
                    const Real mass, const Real rho, const Real p, Real &dmdr,
                    Real &drhodr, Real &dpdr) {
  Real gamma1 = 0.0;
  if (!EOSGamma1FromRhoP(eos, rho, p, gamma1)) {
    return false;
  }
  const Real r_safe = std::max(r, static_cast<Real>(1.0e-30));
  dmdr = 4.0 * M_PI * r_safe * r_safe * rho;
  dpdr = -grav_const * mass * rho / (r_safe * r_safe);
  drhodr = (rho / std::max(gamma1 * p, static_cast<Real>(1.0e-300))) * dpdr;
  return std::isfinite(dmdr) && std::isfinite(drhodr) && std::isfinite(dpdr);
}

bool IntegrateEOSBalancedStructure(const EOS_Data &eos, const Real grav_const,
                                   const Real r_surface, const Real rho_central,
                                   const Real p_central, const int nsteps,
                                   StellarRadialProfile *profile,
                                   Real &mass_at_surface, Real &pressure_at_surface,
                                   Real *surface_radius_out = nullptr,
                                   bool *reached_target_surface_out = nullptr) {
  if (!(r_surface > 0.0) || !(rho_central > 0.0) || !(p_central > 0.0) || nsteps < 4) {
    return false;
  }

  const Real dr = r_surface / static_cast<Real>(nsteps);
  Real r = 0.5 * dr;
  Real rho = rho_central;
  Real p = p_central;
  Real mass = (4.0 / 3.0) * M_PI * rho_central * r * r * r;

  auto finish_surface = [&](const Real r_prev, const Real mass_prev,
                            const Real rho_prev, const Real p_prev,
                            const Real r_term, const Real mass_term,
                            const Real rho_term, const Real p_term) {
    Real frac = 1.0;
    if ((p_prev > 0.0) && (p_term < p_prev) && std::isfinite(p_prev) &&
        std::isfinite(p_term)) {
      frac = p_prev / std::max(p_prev - p_term, static_cast<Real>(1.0e-300));
      frac = std::min(std::max(frac, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    }

    const Real r_zero = r_prev + frac * (r_term - r_prev);
    const Real mass_zero = mass_prev + frac * (mass_term - mass_prev);
    const Real rho_zero = rho_prev + frac * (rho_term - rho_prev);
    const Real r_store =
        std::min(std::max(r_zero, static_cast<Real>(0.0)), r_surface);
    const Real mass_store = std::max(mass_zero, static_cast<Real>(0.0));
    const Real rho_store = std::max(rho_zero, static_cast<Real>(0.0));

    if (profile != nullptr) {
      if (profile->radius.empty() ||
          r_store > profile->radius.back() + static_cast<Real>(1.0e-15) * r_surface) {
        profile->radius.push_back(r_store);
        profile->density.push_back(rho_store);
        profile->pressure.push_back(0.0);
      }
      profile->rho_central = rho_central;
      profile->p_central = p_central;
      profile->mass_at_surface = mass_store;
      profile->pressure_at_surface = p_term;
      profile->surface_radius = r_store;
      profile->reached_target_surface = false;
    }
    mass_at_surface = mass_store;
    pressure_at_surface = p_term;
    if (surface_radius_out != nullptr) {
      *surface_radius_out = r_store;
    }
    if (reached_target_surface_out != nullptr) {
      *reached_target_surface_out = false;
    }
    return true;
  };

  if (profile != nullptr) {
    profile->radius.clear();
    profile->density.clear();
    profile->pressure.clear();
    profile->radius.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->density.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->pressure.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->radius.push_back(0.0);
    profile->density.push_back(rho_central);
    profile->pressure.push_back(p_central);
  }

  for (int step = 0; step < nsteps; ++step) {
    if (!(rho > 0.0) || !(p > 0.0) || !std::isfinite(rho) || !std::isfinite(p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }
    if (!EOSPressureInActiveDomain(eos, rho, p)) {
      if (step == 0) {
        return false;
      }
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    Real k1_m = 0.0, k1_rho = 0.0, k1_p = 0.0;
    if (!EOSBalancedRHS(eos, grav_const, r, mass, rho, p, k1_m, k1_rho, k1_p)) {
      return false;
    }

    const Real r_k2 = r + static_cast<Real>(0.5) * dr;
    const Real mass_k2 = mass + static_cast<Real>(0.5) * dr * k1_m;
    const Real rho_k2 = rho + static_cast<Real>(0.5) * dr * k1_rho;
    const Real p_k2 = p + static_cast<Real>(0.5) * dr * k1_p;
    if (!std::isfinite(mass_k2) || !std::isfinite(rho_k2) || !std::isfinite(p_k2)) {
      return false;
    }
    if (!(rho_k2 > 0.0) || !(p_k2 > 0.0) || !EOSPressureInActiveDomain(eos, rho_k2,
        p_k2)) {
      return finish_surface(r, mass, rho, p, r_k2, mass_k2, rho_k2, p_k2);
    }

    Real k2_m = 0.0, k2_rho = 0.0, k2_p = 0.0;
    if (!EOSBalancedRHS(eos, grav_const, r_k2, mass_k2, rho_k2, p_k2,
                        k2_m, k2_rho, k2_p)) {
      return false;
    }

    const Real r_k3 = r + static_cast<Real>(0.5) * dr;
    const Real mass_k3 = mass + static_cast<Real>(0.5) * dr * k2_m;
    const Real rho_k3 = rho + static_cast<Real>(0.5) * dr * k2_rho;
    const Real p_k3 = p + static_cast<Real>(0.5) * dr * k2_p;
    if (!std::isfinite(mass_k3) || !std::isfinite(rho_k3) || !std::isfinite(p_k3)) {
      return false;
    }
    if (!(rho_k3 > 0.0) || !(p_k3 > 0.0) || !EOSPressureInActiveDomain(eos, rho_k3,
        p_k3)) {
      return finish_surface(r_k2, mass_k2, rho_k2, p_k2,
                            r_k3, mass_k3, rho_k3, p_k3);
    }

    Real k3_m = 0.0, k3_rho = 0.0, k3_p = 0.0;
    if (!EOSBalancedRHS(eos, grav_const, r_k3, mass_k3, rho_k3, p_k3,
                        k3_m, k3_rho, k3_p)) {
      return false;
    }

    const Real r_k4 = r + dr;
    const Real mass_k4 = mass + dr * k3_m;
    const Real rho_k4 = rho + dr * k3_rho;
    const Real p_k4 = p + dr * k3_p;
    if (!std::isfinite(mass_k4) || !std::isfinite(rho_k4) || !std::isfinite(p_k4)) {
      return false;
    }
    if (!(rho_k4 > 0.0) || !(p_k4 > 0.0) || !EOSPressureInActiveDomain(eos, rho_k4,
        p_k4)) {
      return finish_surface(r_k3, mass_k3, rho_k3, p_k3,
                            r_k4, mass_k4, rho_k4, p_k4);
    }

    Real k4_m = 0.0, k4_rho = 0.0, k4_p = 0.0;
    if (!EOSBalancedRHS(eos, grav_const, r_k4, mass_k4, rho_k4, p_k4,
                        k4_m, k4_rho, k4_p)) {
      return false;
    }

    const Real mass_next =
        mass + (dr / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m);
    const Real rho_next =
        rho + (dr / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho);
    const Real p_next =
        p + (dr / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
    const Real r_next = r + dr;

    if (!(mass_next >= 0.0) || !std::isfinite(mass_next) || !std::isfinite(rho_next) ||
        !std::isfinite(p_next)) {
      return false;
    }

    // Interpolate the first zero from the completed RK step using the
    // valid state and the trial endpoint as distinct states.
    const Real r_prev = r;
    const Real mass_prev = mass;
    const Real rho_prev = rho;
    const Real p_prev = p;
    if (!(rho_next > 0.0) || !(p_next > 0.0)) {
      return finish_surface(r_prev, mass_prev, rho_prev, p_prev,
                            r_next, mass_next, rho_next, p_next);
    }

    mass = mass_next;
    rho = rho_next;
    p = p_next;
    r = r_next;

    if (profile != nullptr) {
      profile->radius.push_back(std::min(r, r_surface));
      profile->density.push_back(std::max(rho, static_cast<Real>(0.0)));
      profile->pressure.push_back(std::max(p, static_cast<Real>(0.0)));
    }
  }

  mass_at_surface = mass;
  pressure_at_surface = p;
  if (surface_radius_out != nullptr) {
    *surface_radius_out = r_surface;
  }
  if (reached_target_surface_out != nullptr) {
    *reached_target_surface_out = true;
  }
  if (profile != nullptr) {
    profile->rho_central = rho_central;
    profile->p_central = p_central;
    profile->mass_at_surface = mass_at_surface;
    profile->pressure_at_surface = pressure_at_surface;
    profile->surface_radius = r_surface;
    profile->reached_target_surface = true;
  }
  return true;
}

bool RenormalizeEOSBalancedProfile(const Real grav_const, const Real target_mass,
                                   const Real r_surface, StellarRadialProfile &profile) {
  const std::size_t npts = profile.radius.size();
  if (!(target_mass > 0.0) || npts < 2 || profile.density.size() != npts ||
      profile.pressure.size() != npts) {
    return false;
  }

  if (profile.radius.back() < r_surface) {
    profile.radius.push_back(r_surface);
    profile.density.push_back(0.0);
    profile.pressure.push_back(0.0);
  } else if (profile.radius.back() > r_surface) {
    profile.radius.back() = r_surface;
  }

  const std::size_t n = profile.radius.size();
  profile.density[n - 1] = 0.0;
  profile.pressure[n - 1] = 0.0;
  std::vector<Real> enclosed_mass(n, 0.0);
  Real current_mass = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const Real r0 = profile.radius[i];
    const Real r1 = profile.radius[i + 1];
    const Real dr = r1 - r0;
    if (!(dr >= 0.0)) return false;
    const Real r_mid = static_cast<Real>(0.5) * (r0 + r1);
    const Real rho_mid = static_cast<Real>(0.5) *
                         (std::max(profile.density[i], static_cast<Real>(0.0)) +
                          std::max(profile.density[i + 1], static_cast<Real>(0.0)));
    current_mass += 4.0 * M_PI * r_mid * r_mid * rho_mid * dr;
    enclosed_mass[i + 1] = current_mass;
  }
  if (!(current_mass > 0.0) || !std::isfinite(current_mass)) {
    return false;
  }

  const Real rho_scale = target_mass / current_mass;
  for (Real &rho : profile.density) {
    rho = std::max(rho * rho_scale, static_cast<Real>(0.0));
  }

  enclosed_mass.assign(n, 0.0);
  current_mass = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const Real r0 = profile.radius[i];
    const Real r1 = profile.radius[i + 1];
    const Real dr = r1 - r0;
    const Real r_mid = static_cast<Real>(0.5) * (r0 + r1);
    const Real rho_mid = static_cast<Real>(0.5) * (profile.density[i] + profile.density[i + 1]);
    current_mass += 4.0 * M_PI * r_mid * r_mid * rho_mid * dr;
    enclosed_mass[i + 1] = current_mass;
  }

  profile.pressure.assign(n, 0.0);
  for (std::size_t ii = n - 1; ii > 0; --ii) {
    const std::size_t i = ii - 1;
    const Real r0 = profile.radius[i];
    const Real r1 = profile.radius[i + 1];
    const Real dr = r1 - r0;
    const Real r_mid = std::max(static_cast<Real>(0.5) * (r0 + r1),
                                static_cast<Real>(1.0e-30));
    const Real rho_mid = static_cast<Real>(0.5) * (profile.density[i] + profile.density[i + 1]);
    const Real mass_mid =
        static_cast<Real>(0.5) * (enclosed_mass[i] + enclosed_mass[i + 1]);
    const Real dp = grav_const * mass_mid * rho_mid * dr / (r_mid * r_mid);
    profile.pressure[i] = profile.pressure[i + 1] + dp;
    if (!std::isfinite(profile.pressure[i]) || !(profile.pressure[i] >= 0.0)) {
      return false;
    }
  }

  profile.rho_central = profile.density.front();
  profile.p_central = profile.pressure.front();
  profile.mass_at_surface = enclosed_mass.back();
  profile.pressure_at_surface = 0.0;
  profile.surface_radius = profile.radius.back();
  profile.reached_target_surface =
      std::abs(profile.surface_radius - r_surface) <=
      static_cast<Real>(1.0e-12) * std::max(r_surface, static_cast<Real>(1.0));
  return std::isfinite(profile.rho_central) && std::isfinite(profile.p_central) &&
         std::isfinite(profile.mass_at_surface) &&
         std::abs(profile.mass_at_surface - target_mass) <=
             static_cast<Real>(1.0e-12) * std::max(target_mass, static_cast<Real>(1.0));
}

bool SolveEOSBalancedCentralPressure(const EOS_Data &eos, const Real grav_const,
                                     const Real r_surface, const Real rho_central,
                                     const Real p_guess, Real &p_central,
                                     Real &mass_at_surface, Real &pressure_at_surface) {
  constexpr int kStructureSteps = 4096;
  constexpr int kMaxBracketIters = 80;
  constexpr int kMaxBisectIters = 96;
  constexpr Real kLogPressureTol = 1.0e-10;

  auto evaluate = [&](const Real trial_p, Real &trial_mass, Real &trial_psurf,
                      Real &trial_rsurf, bool &trial_reached) {
    return IntegrateEOSBalancedStructure(eos, grav_const, r_surface, rho_central, trial_p,
                                         kStructureSteps, nullptr, trial_mass, trial_psurf,
                                         &trial_rsurf, &trial_reached);
  };

  Real p_lo = std::max(p_guess * static_cast<Real>(1.0e-3), static_cast<Real>(1.0e-16));
  Real p_hi = std::max(p_guess, p_lo * static_cast<Real>(10.0));
  if (eos.UsesTabulatedLTE()) {
    const Real p_floor = eos.HostTabulatedPressureFloor(rho_central);
    const Real p_ceil = eos.HostTabulatedPressureCeiling(rho_central);
    p_lo = std::max(p_lo, p_floor);
    p_hi = std::min(std::max(p_hi, p_lo * static_cast<Real>(10.0)), p_ceil);
  }
  if (!(p_hi >= p_lo) || !std::isfinite(p_lo) || !std::isfinite(p_hi)) return false;

  Real m_lo = 0.0, psurf_lo = 0.0, rsurf_lo = 0.0;
  bool reached_lo = false;
  if (!evaluate(p_lo, m_lo, psurf_lo, rsurf_lo, reached_lo)) return false;
  for (int n = 0; n < kMaxBracketIters && reached_lo; ++n) {
    const Real old_p = p_lo;
    p_lo *= static_cast<Real>(0.1);
    if (eos.UsesTabulatedLTE()) {
      p_lo = std::max(p_lo, eos.HostTabulatedPressureFloor(rho_central));
    }
    if (!(p_lo < old_p)) break;
    if (!evaluate(p_lo, m_lo, psurf_lo, rsurf_lo, reached_lo)) return false;
  }

  if (reached_lo) {
    p_central = p_lo;
    mass_at_surface = m_lo;
    pressure_at_surface = psurf_lo;
    return true;
  }

  Real m_hi = 0.0, psurf_hi = 0.0, rsurf_hi = 0.0;
  bool reached_hi = false;
  if (!evaluate(p_hi, m_hi, psurf_hi, rsurf_hi, reached_hi)) return false;
  for (int n = 0; n < kMaxBracketIters && !reached_hi; ++n) {
    const Real old_p = p_hi;
    p_hi *= static_cast<Real>(10.0);
    if (eos.UsesTabulatedLTE()) {
      p_hi = std::min(p_hi, eos.HostTabulatedPressureCeiling(rho_central));
    }
    if (!(p_hi > old_p) || !std::isfinite(p_hi)) break;
    if (!evaluate(p_hi, m_hi, psurf_hi, rsurf_hi, reached_hi)) return false;
  }
  if (!reached_hi) return false;

  for (int n = 0; n < kMaxBisectIters; ++n) {
    const Real log_p_lo = std::log(std::max(p_lo, static_cast<Real>(1.0e-300)));
    const Real log_p_hi = std::log(std::max(p_hi, static_cast<Real>(1.0e-300)));
    if (std::abs(log_p_hi - log_p_lo) <= kLogPressureTol) break;

    const Real p_mid = std::exp(static_cast<Real>(0.5) * (log_p_lo + log_p_hi));
    Real m_mid = 0.0, psurf_mid = 0.0, rsurf_mid = 0.0;
    bool reached_mid = false;
    if (!evaluate(p_mid, m_mid, psurf_mid, rsurf_mid, reached_mid)) return false;
    if (reached_mid) {
      p_hi = p_mid;
      m_hi = m_mid;
      psurf_hi = psurf_mid;
      rsurf_hi = rsurf_mid;
    } else {
      p_lo = p_mid;
      m_lo = m_mid;
      psurf_lo = psurf_mid;
      rsurf_lo = rsurf_mid;
    }
  }

  p_central = p_hi;
  mass_at_surface = m_hi;
  pressure_at_surface = psurf_hi;
  return true;
}

bool SolveEOSBalancedProfile(const EOS_Data &eos, const Real grav_const,
                             const Real target_mass, const Real r_surface,
                             const Real rho_guess, const Real p_guess,
                             StellarRadialProfile &profile) {
  constexpr int kStructureSteps = 4096;
  constexpr int kMaxBracketIters = 80;
  constexpr int kMaxBisectIters = 96;
  constexpr Real kMassTolFrac = 1.0e-10;

  auto evaluate = [&](const Real trial_rho, Real &trial_p_c, Real &trial_mass,
                      Real &trial_psurf) {
    // At fixed radius the hydrostatic pressure scale is P_c ~ G rho_c^2 R^2.
    const Real rho_ratio =
        std::max(trial_rho, static_cast<Real>(1.0e-30)) /
        std::max(rho_guess, static_cast<Real>(1.0e-30));
    const Real scaled_p_guess = p_guess * rho_ratio * rho_ratio;
    return SolveEOSBalancedCentralPressure(eos, grav_const, r_surface, trial_rho,
                                           scaled_p_guess, trial_p_c, trial_mass,
                                           trial_psurf);
  };

  Real rho_active_min = static_cast<Real>(1.0e-12);
  Real rho_active_max = std::numeric_limits<Real>::max();
  if (eos.UsesTabulatedLTE()) {
    rho_active_min =
        std::exp(eos.saha_logrho_min) /
        std::max(eos.density_unit_cgs, static_cast<Real>(1.0e-300));
    rho_active_max =
        std::exp(eos.saha_logrho_max) /
        std::max(eos.density_unit_cgs, static_cast<Real>(1.0e-300));
  }
  if (!(rho_active_min > 0.0) || !(rho_active_max >= rho_active_min)) return false;
  const Real rho_seed = std::min(std::max(rho_guess, rho_active_min), rho_active_max);
  Real rho_lo = std::max(rho_seed * static_cast<Real>(0.5), rho_active_min);
  Real rho_hi = std::min(std::max(rho_seed, rho_lo * static_cast<Real>(2.0)),
                         rho_active_max);
  Real p_lo = 0.0, mass_lo = 0.0, psurf_lo = 0.0;
  Real p_hi = 0.0, mass_hi = 0.0, psurf_hi = 0.0;

  if (!evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_lo > target_mass; ++n) {
    const Real old_rho = rho_lo;
    rho_lo = std::max(rho_lo * static_cast<Real>(0.5), rho_active_min);
    if (!(rho_lo < old_rho)) break;
    if (!(rho_lo > 0.0) || !evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  }

  if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_hi < target_mass; ++n) {
    if (!(rho_hi < rho_active_max)) break;
    rho_hi = std::min(rho_hi * static_cast<Real>(2.0), rho_active_max);
    if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  }

  if (!(mass_lo <= target_mass && mass_hi >= target_mass)) return false;

  Real rho_mid = rho_guess;
  Real p_mid = 0.0;
  Real mass_mid = 0.0;
  Real psurf_mid = 0.0;
  for (int n = 0; n < kMaxBisectIters; ++n) {
    const Real log_rho_lo = std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi = std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (target_mass - mass_lo) / dmass;
      if (trial_frac > 0.0 && trial_frac < 1.0 && std::isfinite(trial_frac)) {
        frac = trial_frac;
      }
    }
    rho_mid = std::exp(log_rho_lo + frac * (log_rho_hi - log_rho_lo));
    if (!evaluate(rho_mid, p_mid, mass_mid, psurf_mid)) return false;
    if (std::abs(mass_mid - target_mass) <=
        kMassTolFrac * std::max(target_mass, static_cast<Real>(1.0))) {
      break;
    }
    if (mass_mid > target_mass) {
      rho_hi = rho_mid;
      p_hi = p_mid;
      mass_hi = mass_mid;
      psurf_hi = psurf_mid;
    } else {
      rho_lo = rho_mid;
      p_lo = p_mid;
      mass_lo = mass_mid;
      psurf_lo = psurf_mid;
    }
  }

  if (std::abs(mass_mid - target_mass) >
      kMassTolFrac * std::max(target_mass, static_cast<Real>(1.0))) {
    const Real log_rho_lo = std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi = std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (target_mass - mass_lo) / dmass;
      if (trial_frac > 0.0 && trial_frac < 1.0 && std::isfinite(trial_frac)) {
        frac = trial_frac;
      }
    }
    rho_mid = std::exp(log_rho_lo + frac * (log_rho_hi - log_rho_lo));
    if (!evaluate(rho_mid, p_mid, mass_mid, psurf_mid)) return false;
  }

  Real rho_profile = rho_mid;
  Real p_profile = p_mid;
  if (psurf_mid < 0.0 && psurf_hi >= 0.0) {
    rho_profile = rho_hi;
    p_profile = p_hi;
  }
  Real final_mass = 0.0;
  Real final_psurf = 0.0;
  if (!IntegrateEOSBalancedStructure(eos, grav_const, r_surface, rho_profile, p_profile,
                                     kStructureSteps, &profile, final_mass, final_psurf)) {
    return false;
  }
  if (!RenormalizeEOSBalancedProfile(grav_const, target_mass, r_surface, profile)) {
    return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
Real RadialProfileValue(const Real r, const DvceArray1D<Real> radius_tab,
                        const DvceArray1D<Real> value_tab, const int npts) {
  if (npts <= 0) return 0.0;
  if (r <= radius_tab(0)) return value_tab(0);
  if (r >= radius_tab(npts - 1)) return 0.0;

  int lo = 0;
  int hi = npts - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (radius_tab(mid) <= r) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const Real r0 = radius_tab(lo);
  const Real r1 = radius_tab(hi);
  if (!(r1 > r0)) return value_tab(lo);
  const Real frac = (r - r0) / (r1 - r0);
  return value_tab(lo) + frac * (value_tab(hi) - value_tab(lo));
}

}  // namespace

// Forward declaration
void PolyStarRefinement(MeshBlockPack *pmbp);
void PolyStarRelaxationSource(Mesh *pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PolytropicStar()
//! \brief Sets up a Lane-Emden polytropic sphere for gravitational collapse with Jeans AMR.

void ProblemGenerator::PolytropicStar(ParameterInput *pin, const bool restart) {
  // --- gravity coupling ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
    pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
    if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  // --- problem parameters ---
  Real r_star = pin->GetOrAddReal("problem", "star_radius", 0.5);
  Real rho_central = pin->GetOrAddReal("problem", "rho_central", 1.0);
  Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-8);
  Real poly_n = pin->GetOrAddReal("problem", "poly_n", 1.0);
  Real pressure_support = pin->GetOrAddReal("problem", "pressure_support", 1.0);
  std::string stellar_structure_mode_name =
      pin->GetOrAddString("problem", "stellar_structure_mode", "legacy_polytrope");
  StellarStructureMode stellar_structure_mode = StellarStructureMode::legacy_polytrope;
  bool relax_damp = pin->GetOrAddBoolean("problem", "relax_damp", false);
  Real relax_tau = pin->GetOrAddReal("problem", "relax_tau", 0.0);
  Real relax_t_end = pin->GetOrAddReal("problem", "relax_t_end", 0.0);
  Real relax_radius = pin->GetOrAddReal("problem", "relax_radius", r_star);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);  // m=2 perturbation amplitude
  Real x_center = pin->GetOrAddReal("problem", "x_center", 0.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  if (poly_n <= 0.0 || poly_n >= 5.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/poly_n must satisfy 0 < n < 5 for a finite-radius "
              << "Lane-Emden sphere." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (r_star <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/star_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rho_central <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/rho_central must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pressure_support <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/pressure_support must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!ParseStellarStructureMode(stellar_structure_mode_name, stellar_structure_mode)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/stellar_structure_mode must be one of "
              << "'legacy_polytrope' or 'eos_balanced'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (relax_damp) {
    if (relax_tau <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/relax_tau must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_t_end <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/relax_t_end must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_radius <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/relax_radius must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (four_pi_G <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "gravity/four_pi_G must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // Hydro when present, otherwise MHD with a uniform field that exerts no force on the
  // star (the same problem/b0 convention as the other self-gravity test generators).
  const bool use_mhd = (pmbp->phydro == nullptr);
  if (use_mhd && pmbp->pmhd == nullptr) return;
  const EOS_Data eos = use_mhd ? pmbp->pmhd->peos->eos_data
                               : pmbp->phydro->peos->eos_data;
  const Real b0_val = use_mhd ? pin->GetOrAddReal("problem", "b0", 0.0) : 0.0;
  const bool use_eos_balanced_profile =
      (stellar_structure_mode == StellarStructureMode::eos_balanced);
  if (use_eos_balanced_profile) {
    if (!eos.use_e || (!eos.is_gamma_law && !eos.UsesTabulatedLTE())) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/stellar_structure_mode = eos_balanced requires a "
                << "gamma-law or tabulated LTE EOS with an energy variable."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (amp != 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/amp must be 0 for eos_balanced initialization."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (std::abs(pressure_support - 1.0) > 1.0e-12) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "problem/pressure_support must be 1 for eos_balanced initialization."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  LaneEmdenProfile lane_profile;
  if (!SolveLaneEmden(poly_n, lane_profile)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "Failed to solve Lane-Emden equation for n=" << poly_n << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real lane_xi1 = lane_profile.xi1;
  Real a_scale = r_star / lane_xi1;
  Real lane_qn = -SQR(lane_xi1) * lane_profile.dtheta.back();
  if (!(lane_qn > 0.0)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "Lane-Emden mass factor is non-positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real lane_mass = 4.0 * M_PI * std::pow(a_scale, 3.0) * rho_central * lane_qn;
  if (!use_eos_balanced_profile && pin->DoesParameterExist("problem", "star_mass")) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/star_mass is supported only with "
              << "problem/stellar_structure_mode = eos_balanced." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real target_mass = pin->DoesParameterExist("problem", "star_mass") ?
      pin->GetReal("problem", "star_mass") : lane_mass;
  if (!(target_mass > 0.0)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/star_mass must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real poly_k = (four_pi_G / (poly_n + 1.0)) * SQR(a_scale)
              * std::pow(rho_central, 1.0 - 1.0/poly_n);
  Real gamma_poly = 1.0 + 1.0/poly_n;
  int lane_npts = static_cast<int>(lane_profile.xi.size());

  DvceArray1D<Real> lane_xi_d("lane_xi", lane_npts);
  DvceArray1D<Real> lane_theta_d("lane_theta", lane_npts);
  DvceArray1D<Real> lane_dtheta_d("lane_dtheta", lane_npts);
  HostArray1D<Real> lane_xi_h = Kokkos::create_mirror_view(lane_xi_d);
  HostArray1D<Real> lane_theta_h = Kokkos::create_mirror_view(lane_theta_d);
  HostArray1D<Real> lane_dtheta_h = Kokkos::create_mirror_view(lane_dtheta_d);
  for (int n = 0; n < lane_npts; ++n) {
    lane_xi_h(n) = lane_profile.xi[n];
    lane_theta_h(n) = lane_profile.theta[n];
    lane_dtheta_h(n) = lane_profile.dtheta[n];
  }
  Kokkos::deep_copy(lane_xi_d, lane_xi_h);
  Kokkos::deep_copy(lane_theta_d, lane_theta_h);
  Kokkos::deep_copy(lane_dtheta_d, lane_dtheta_h);

  StellarRadialProfile eos_profile;
  int radial_profile_npts = 1;
  DvceArray1D<Real> radial_r_d("stellar_profile_r", 1);
  DvceArray1D<Real> radial_rho_d("stellar_profile_rho", 1);
  DvceArray1D<Real> radial_p_d("stellar_profile_p", 1);
  {
    HostArray1D<Real> radial_r_h = Kokkos::create_mirror_view(radial_r_d);
    HostArray1D<Real> radial_rho_h = Kokkos::create_mirror_view(radial_rho_d);
    HostArray1D<Real> radial_p_h = Kokkos::create_mirror_view(radial_p_d);
    radial_r_h(0) = 0.0;
    radial_rho_h(0) = 0.0;
    radial_p_h(0) = 0.0;
    Kokkos::deep_copy(radial_r_d, radial_r_h);
    Kokkos::deep_copy(radial_rho_d, radial_rho_h);
    Kokkos::deep_copy(radial_p_d, radial_p_h);
  }

  if (use_eos_balanced_profile) {
    const Real grav_const = four_pi_G / (4.0 * M_PI);
    const Real p_central_guess = poly_k * std::pow(rho_central, gamma_poly);
    if (!SolveEOSBalancedProfile(eos, grav_const, target_mass, r_star, rho_central,
                                 p_central_guess, eos_profile)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
                << "Failed to construct an EOS-balanced stellar profile." << std::endl;
      if (eos.UsesTabulatedLTE()) {
        std::cout << "The requested star is not covered by the active tabulated EOS "
                  << "rho-T domain. Try a smaller problem/star_mass or a different "
                  << "unit scaling/table." << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    rho_central = eos_profile.rho_central;
    radial_profile_npts = static_cast<int>(eos_profile.radius.size());
    radial_r_d = DvceArray1D<Real>("stellar_profile_r", radial_profile_npts);
    radial_rho_d = DvceArray1D<Real>("stellar_profile_rho", radial_profile_npts);
    radial_p_d = DvceArray1D<Real>("stellar_profile_p", radial_profile_npts);
    HostArray1D<Real> radial_r_h = Kokkos::create_mirror_view(radial_r_d);
    HostArray1D<Real> radial_rho_h = Kokkos::create_mirror_view(radial_rho_d);
    HostArray1D<Real> radial_p_h = Kokkos::create_mirror_view(radial_p_d);
    for (int n = 0; n < radial_profile_npts; ++n) {
      radial_r_h(n) = eos_profile.radius[n];
      radial_rho_h(n) = eos_profile.density[n];
      radial_p_h(n) = eos_profile.pressure[n];
    }
    Kokkos::deep_copy(radial_r_d, radial_r_h);
    Kokkos::deep_copy(radial_rho_d, radial_rho_h);
    Kokkos::deep_copy(radial_p_d, radial_p_h);
  }

  // --- AMR Jeans criterion ---
  njeans_threshold = pin->GetOrAddReal("problem", "njeans", 16.0);

  // The refinement hook evaluates the Jeans number with the active EOS.
  iso_cs_global = eos.iso_cs;
  poly_n_global = poly_n;
  poly_k_global = poly_k;
  rho_floor_global = rho_floor;
  four_pi_G_global = four_pi_G;
  relax_enable_global = relax_damp;
  relax_tau_global = relax_tau;
  relax_t_end_global = relax_t_end;
  relax_radius_global = relax_radius;
  relax_x_center_global = x_center;
  relax_y_center_global = y_center;
  relax_z_center_global = z_center;

  if (global_variable::my_rank == 0 && eos.is_gamma_law && !use_eos_balanced_profile) {
    Real gamma_target = 1.0 + 1.0/poly_n;
    Real gamma_eos = eos.gamma;
    if (std::abs(gamma_eos - gamma_target) > 1.0e-8) {
      std::cout << "### WARNING in ProblemGenerator::PolytropicStar" << std::endl
                << "For consistency with poly_n=" << poly_n
                << ", set hydro/gamma = 1 + 1/n = " << gamma_target
                << " (current gamma = " << gamma_eos << ")." << std::endl;
    }
  }

  // Register Jeans refinement condition
  user_ref_func = PolyStarRefinement;
  if (relax_enable_global) {
    user_srcs = true;
    user_srcs_func = PolyStarRelaxationSource;
    pin->SetBoolean("problem", "user_srcs", true);
  }

  if (restart) return;

  // --- initialize density ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  DvceArray5D<Real> u0 = use_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  int nmb = pmbp->nmb_thispack;
  bool eos_has_energy = eos.use_e;
  Real xi_scale = lane_xi1 / r_star;
  const Real floor_eint =
      eos.HostClampHydroInternalEnergyDensity(rho_floor,
                                              eos.HostHydroInternalEnergyDensityFloor(rho_floor));
  const Real p_floor_eos = eos.HostPressureFromRhoEint(rho_floor, floor_eint);
  const Real p_floor_legacy = poly_k * std::pow(rho_floor, gamma_poly);
  const Real p_floor_init = use_eos_balanced_profile ? p_floor_eos : p_floor_legacy;

  par_for("polytropic_star_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center) + SQR(z - z_center));
    Real rho_star = 0.0;
    Real p_star = 0.0;
    if (use_eos_balanced_profile) {
      if (r < r_star) {
        rho_star = RadialProfileValue(r, radial_r_d, radial_rho_d, radial_profile_npts);
        p_star = RadialProfileValue(r, radial_r_d, radial_p_d, radial_profile_npts);
      }
    } else {
      Real theta = 0.0;
      if (r < r_star) {
        Real xi = r * xi_scale;
        theta = LaneEmdenTheta(xi, lane_xi_d, lane_theta_d, lane_dtheta_d, lane_npts);
      }
      rho_star = rho_central * Kokkos::pow(theta, poly_n);
      if (amp > 0.0 && r < r_star) {
        rho_star *= (1.0 + amp * (r * r) / (r_star * r_star)
                     * Kokkos::cos(2.0 * Kokkos::atan2(y - y_center, x - x_center)));
      }
      p_star = poly_k * Kokkos::pow(Kokkos::fmax(rho_star, 0.0), gamma_poly);
    }
    Real rho = rho_floor + rho_star;

    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (eos_has_energy) {
      Real pgas = p_floor_init + pressure_support * p_star;
      Real eint = eos.InternalEnergyDensityFromRhoP(rho, pgas);
      eint = Kokkos::fmax(eint, eos.HydroInternalEnergyDensityFloor(rho));
      eint = eos.ClampHydroInternalEnergyDensity(rho, eint);
      u0(m, IEN, k, j, i) = eint + 0.5*SQR(b0_val);
    }
  });

  if (use_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("polytropic_star_bfield", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m, k, j, i) = 0.0;
      b0.x2f(m, k, j, i) = 0.0;
      b0.x3f(m, k, j, i) = b0_val;
      if (i == ie) b0.x1f(m, k, j, i+1) = 0.0;
      if (j == je) b0.x2f(m, k, j+1, i) = 0.0;
      if (k == ke) b0.x3f(m, k+1, j, i) = b0_val;
      bcc0(m, IBX, k, j, i) = 0.0;
      bcc0(m, IBY, k, j, i) = 0.0;
      bcc0(m, IBZ, k, j, i) = b0_val;
    });
  }

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- Polytropic Star ---" << std::endl
      << "rho_central            = " << rho_central << std::endl
      << "star_mass              = " << target_mass << std::endl
      << "star_radius            = " << r_star << std::endl
      << "stellar_structure_mode = "
      << StellarStructureModeName(stellar_structure_mode) << std::endl
      << "rho_floor              = " << rho_floor << std::endl
      << "Perturbation amplitude = " << amp << std::endl
      << "poly_n                 = " << poly_n << std::endl
      << "poly_gamma             = " << gamma_poly << std::endl
      << "Lane-Emden xi1         = " << lane_xi1 << std::endl
      << "poly_K (derived)       = " << poly_k << std::endl
      << "pressure_support       = " << pressure_support << std::endl
      << "relax_damp             = " << (relax_enable_global ? "true" : "false")
      << std::endl
      << "relax_tau              = " << relax_tau_global << std::endl
      << "relax_t_end            = " << relax_t_end_global << std::endl
      << "relax_radius           = " << relax_radius_global << std::endl
      << "Jeans AMR threshold    = " << njeans_threshold << std::endl
      << "four_pi_G              = " << four_pi_G << std::endl
      << std::endl;
    if (use_eos_balanced_profile) {
      std::cout
        << "EOS-balanced p_c       = " << eos_profile.p_central << std::endl
        << "EOS-balanced M(r_*)    = " << eos_profile.mass_at_surface << std::endl
        << "EOS-balanced P(r_*)    = " << eos_profile.pressure_at_surface << std::endl
        << "EOS profile samples    = " << radial_profile_npts << std::endl
        << std::endl;
    }
  }
}

void PolyStarRelaxationSource(Mesh *pm, const Real bdt) {
  if (!relax_enable_global || bdt <= 0.0) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;

  Real stage_fraction = 0.0;
  if (pm->dt > 0.0) {
    stage_fraction = bdt/pm->dt;
  }
  if (stage_fraction < 0.0) stage_fraction = 0.0;
  Real t_src = pm->time + stage_fraction * pm->dt;
  if (t_src > relax_t_end_global) return;

  Real damp_fac = std::exp(-bdt/relax_tau_global);
  Real rmax2 = relax_radius_global * relax_radius_global;
  Real xc = relax_x_center_global;
  Real yc = relax_y_center_global;
  Real zc = relax_z_center_global;
  bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("polystar_relax_damp", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
    Real r2 = SQR(x - xc) + SQR(y - yc) + SQR(z - zc);
    if (r2 > rmax2) return;

    Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0) return;

    Real m1 = u0(m, IM1, k, j, i);
    Real m2 = u0(m, IM2, k, j, i);
    Real m3 = u0(m, IM3, k, j, i);
    Real ke_old = 0.0;
    if (eos_has_energy) {
      ke_old = 0.5 * (m1*m1 + m2*m2 + m3*m3) / rho;
    }

    m1 *= damp_fac;
    m2 *= damp_fac;
    m3 *= damp_fac;
    u0(m, IM1, k, j, i) = m1;
    u0(m, IM2, k, j, i) = m2;
    u0(m, IM3, k, j, i) = m3;

    if (eos_has_energy) {
      Real ke_new = 0.5 * (m1*m1 + m2*m2 + m3*m3) / rho;
      u0(m, IEN, k, j, i) += (ke_new - ke_old);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void PolyStarRefinement()
//! \brief Jeans-length AMR criterion for self-gravitating gas.
//!
//! For each MeshBlock, computes the minimum cell-wise Jeans number:
//!   nJ = 2*pi*cs / (dx*sqrt(4*pi*G*rho))
//! and sets the refinement flag accordingly.

void PolyStarRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int ng = indcs.ng;
  const int nkji = (nx3 + 2 * ng) * (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int nji  = (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int ni   = (nx1 + 2 * ng);
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  const bool use_mhd = (pmbp->phydro == nullptr);
  DvceArray5D<Real> u0 = use_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  DvceArray5D<Real> w0 = use_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  const auto eos = use_mhd ? pmbp->pmhd->peos->eos_data
                           : pmbp->phydro->peos->eos_data;
  auto &size = pmbp->pmb->mb_size;
  Real cs_iso = iso_cs_global;
  Real rho_floor = rho_floor_global;
  Real four_pi_G = four_pi_G_global;
  Real njeans = njeans_threshold;

  par_for_outer("PolyStarAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_njmin = 1.0e99;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &njmin) {
        int k = idx / nji;
        int j = (idx - k * nji) / ni;
        int i = (idx - k * nji - j * ni);
        Real rho = Kokkos::fmax(u0(m, IDN, k, j, i), rho_floor);
        Real cs = cs_iso;
        if (eos.use_e) {
          cs = Kokkos::sqrt(Kokkos::fmax(
              eos.HydroSoundSpeed2FromRhoEint(rho, w0(m, IEN, k, j, i)),
              static_cast<Real>(0.0)));
        }
        Real dx = size.d_view(m).dx1;
        if (nx2 > 1) dx = Kokkos::fmax(dx, size.d_view(m).dx2);
        if (nx3 > 1) dx = Kokkos::fmax(dx, size.d_view(m).dx3);
        Real nj_cell = (2.0 * M_PI * cs) / (dx * Kokkos::sqrt(four_pi_G * rho));
        njmin = Kokkos::fmin(njmin, nj_cell);
      },
      Kokkos::Min<Real>(team_njmin));

    if (team_njmin < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (team_njmin > njeans * 2.5) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
