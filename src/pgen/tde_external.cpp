//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_external.cpp
//! \brief Problem generator for a star in an external SMBH potential.
//!
//! The legacy stellar profile is a Lane-Emden polytrope with fixed total stellar mass
//! M*=1. A selectable EOS-balanced mode also constructs a 1D hydrostatic profile with
//! the active EOS closure before mapping it into 3D.
//! An external Newtonian BH gravity is added through the gravity source path:
//!   Phi_BH = -G M_BH / sqrt(|r-r_BH|^2 + eps^2).
//! Here G is inferred from gravity/four_pi_G as G = four_pi_G/(4*pi), consistent with
//! the unit system used by self-gravity setups in this code.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "driver/driver.hpp"
#include "outputs/io_wrapper.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/gravity_weight.hpp"
#include "units/units.hpp"
#include "remap/remap.hpp"
#include "pgen.hpp"
#include "pgen/bh_dynamics.hpp"
#include "pgen/star_bh_orbit.hpp"
#include "tde_external.hpp"

namespace tde_external {
void StoreRuntimeMetadata(ParameterInput *pin);
}  // namespace tde_external

namespace {

// External BH gravity source-term parameters
Real bh_mass_global;
Real bh_inertial_x_global, bh_inertial_y_global, bh_inertial_z_global;
Real bh_x_global, bh_y_global, bh_z_global;
Real bh_vx_global, bh_vy_global, bh_vz_global;
Real bh_soft_global;
Real bh_grav_rho_min_global;  // minimum density for BH gravity source term
Real frame_rho_min_global;    // minimum density used to measure frame acceleration
bool use_translating_frame_global = false;
bool hydro_lat_enabled_global = false;
Real bh_excise_radius_global;
Real bh_excise_density_global;
Real bh_excise_eint_global;
Real newton_g_global;
bool bh_inner_boundary_global = true;
bool remap_enable_global = false;
std::string remap_restart_source_global;
int remap_settle_steps_global = 0;
int remap_settle_config_passes_global = 0;
int remap_settle_remaining_passes_global = 0;
int remap_settle_target_cycle_global = -1;
bool remap_settle_active_global = false;
int remap_amr_ncycle_check_config_global = 1;
int remap_amr_refinement_interval_config_global = 1;

// Translating, non-rotating debris-frame state. When the translating frame is
// disabled, these are reset to zero and the BH is frozen in the current mesh
// coordinates.
Real orbit_center_x_global, orbit_center_y_global, orbit_center_z_global;
Real frame_vx_global, frame_vy_global, frame_vz_global;
Real frame_ax_global, frame_ay_global, frame_az_global;
Real frame_state_time_global;
int frame_step_cycle_global = -1;
Real frame_step_time_global = 0.0;
Real frame_step_dt_global = 0.0;
Real frame_step_x0_global, frame_step_y0_global, frame_step_z0_global;
Real frame_step_vx0_global, frame_step_vy0_global, frame_step_vz0_global;
Real frame_step_ax0_global, frame_step_ay0_global, frame_step_az0_global;
Real frame_step_x1_global, frame_step_y1_global, frame_step_z1_global;
Real frame_step_vx1_global, frame_step_vy1_global, frame_step_vz1_global;
Real frame_step_ax1_global, frame_step_ay1_global, frame_step_az1_global;
bool frame_step_valid_global = false;
bool relax_enable_global;
Real relax_tau_global, relax_t_end_global, relax_radius_global;
bool bh_force_refine_global = false;
int bh_force_refine_level_offset_global = 0;
bool unbound_refine_global = false;
int unbound_refine_level_offset_global = 3;
Real unbound_refine_rho_min_global = -1.0;
Real unbound_refine_fill_frac_global = 0.01;
bool stream_shell_enable_global = false;
Real stream_shell_dr_global = -1.0;
int stream_shell_level_offset_global = 0;
Real stream_shell_r_max_global = 0.0;
Real stream_shell_xsplit_radius_global = 0.0;
Real stream_shell_rho_frac_global = 0.5;
Real stream_shell_fill_frac_global = 0.05;
Real stream_shell_rho_floor_global = 0.0;
Real hydro_dfloor_global = 0.0;
Real stream_shell_derefine_dfloor_mult_global = 100.0;
std::vector<Real> stream_shell_level_radii_global;
std::vector<int> stream_shell_level_offsets_global;
std::vector<Real> stream_shell_level_drs_global;
std::vector<Real> stream_shell_level_rho_fracs_global;
std::vector<Real> stream_shell_level_fill_fracs_global;

int bh_step_cycle_global = -1;
Real bh_step_time_global = 0.0;
Real bh_step_dt_global = 0.0;
Real bh_step_x0_global, bh_step_y0_global, bh_step_z0_global;
Real bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global;
Real bh_step_x1_global, bh_step_y1_global, bh_step_z1_global;
Real bh_step_vx1_global, bh_step_vy1_global, bh_step_vz1_global;
bool bh_step_valid_global = false;

// Live BH (problem/bh_live = true).  The BH inertial state is advanced by the gas gravity
// with the kick-drift-kick leapfrog at rank-synchronized points exactly as in
// star_bh_collision (shared helpers in pgen/bh_dynamics.hpp); the simulation-frame BH is
// that state minus the translating frame.  By default the BH stays at rest in the
// inertial frame.
bool bh_live_global = false;
Real bh_inertial_vx_global = 0.0, bh_inertial_vy_global = 0.0;
Real bh_inertial_vz_global = 0.0;
Real bh_gas_ax_global = 0.0, bh_gas_ay_global = 0.0, bh_gas_az_global = 0.0;
bool bh_reciprocal_force_global = false;
bool bh_analytic_pair_global = false;
Real bh_local_gas_impulse[3] = {0.0, 0.0, 0.0};
Real bh_total_gas_impulse[3] = {0.0, 0.0, 0.0};
Real bh_last_impulse_residual = 0.0;
bool bh_impulse_window_open = false;
Real bh_window_start_v[3] = {0.0, 0.0, 0.0};   // inertial BH velocity at the step start
Real bh_window_frame_dv[3] = {0.0, 0.0, 0.0};  // frame velocity change over the step
Real bh_window_frame_v1[3] = {0.0, 0.0, 0.0};  // frame velocity at the step end
// Energy ledger (gravity/lat_time_centered_work): cumulative work of the translating
// frame's fictitious force (and relaxation damping) on the gas and on a live BH.  The
// gas part is accumulated rank-locally in the source pass and reduced only at
// synchronized points.
bool energy_ledger_global = false;
Real frame_work_local = 0.0;
Real frame_work_total = 0.0;

struct LaneEmdenProfile {
  std::vector<Real> xi;
  std::vector<Real> theta;
  std::vector<Real> dtheta;
  Real xi1;
  Real dtheta_xi1;
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

void RefineBHPosition(MeshBlockPack *pmbp);
void CapHydroLATFactorsNearBH(Mesh *pm, int max_factor, int *lat_factor_eachmb);
void SyncProblemRuntimeState();
void InvalidateFrameBHStepState();
void RestoreOptionalFrameBHStepState(ParameterInput *pin);
void ScheduleNextAutoRemapPass(Mesh *pm);
void DisableAutoRemapPasses(Mesh *pm = nullptr);
void UpdateAutoRemapAMRControls(Mesh *pm);
bool OutputsAllowedOnCurrentCycle(Mesh *pm);
void TDEExternalHydroStateFixup(MeshBlockPack *pmbp, const Real time);
void SwitchToFrozenBHInertialFrame(Mesh *pm, MeshBlockPack *pmbp);
bool RestoreFrameBHStateFromMetadata(ParameterInput *pin);
void RestoreLiveBHMetadata(ParameterInput *pin);
void LoadAndApplyRemapSource(Mesh *pm, MeshBlockPack *pmbp, ParameterInput *dst_pin,
                             bool copy_output_state, const char *banner_label);
bool AutoRemapAfterCycle(Driver *driver, ParameterInput *pin, Mesh *pm);

inline Real ClampUnitInterval(const Real x);
KOKKOS_INLINE_FUNCTION
Real CubicHermiteValue(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t);
KOKKOS_INLINE_FUNCTION
Real CubicHermiteSlope(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t);

inline Real ThetaPow(const Real theta, const Real n) {
  return (theta > 0.0) ? std::pow(theta, n) : 0.0;
}

inline void EvaluateBHAccelerationAtPoint(const Real x, const Real y, const Real z,
                                          const Real bhx, const Real bhy, const Real bhz,
                                          Real &ax, Real &ay, Real &az) {
  Real dx = x - bhx;
  Real dy = y - bhy;
  Real dz = z - bhz;
  Real eps2 = bh_soft_global * bh_soft_global;
  Real r2 = dx*dx + dy*dy + dz*dz + eps2;
  r2 = std::fmax(r2, static_cast<Real>(1.0e-30));
  Real invr = 1.0 / std::sqrt(r2);
  Real invr3 = invr*invr*invr;
  Real gm = newton_g_global * bh_mass_global;
  ax = -gm * dx * invr3;
  ay = -gm * dy * invr3;
  az = -gm * dz * invr3;
}

void UpdateAutoRemapAMRControls(Mesh *pm) {
  if (pm == nullptr || pm->pmr == nullptr) return;
  if (remap_settle_active_global) {
    pm->pmr->ncyc_check_amr = 1;
    pm->pmr->refinement_interval = 1;
  } else {
    pm->pmr->ncyc_check_amr =
        std::max(remap_amr_ncycle_check_config_global, 1);
    pm->pmr->refinement_interval =
        std::max(remap_amr_refinement_interval_config_global,
                 pm->pmr->ncyc_check_amr);
  }
}

void DisableAutoRemapPasses(Mesh *pm) {
  remap_settle_remaining_passes_global = 0;
  remap_settle_target_cycle_global = -1;
  remap_settle_active_global = false;
  UpdateAutoRemapAMRControls(pm);
}

void ScheduleNextAutoRemapPass(Mesh *pm) {
  if (pm == nullptr || !remap_enable_global || remap_settle_steps_global <= 0 ||
      remap_settle_remaining_passes_global <= 0) {
    DisableAutoRemapPasses(pm);
    return;
  }
  remap_settle_target_cycle_global = pm->ncycle + remap_settle_steps_global;
  remap_settle_active_global = true;
  UpdateAutoRemapAMRControls(pm);
}

bool OutputsAllowedOnCurrentCycle(Mesh *pm) {
  (void)pm;
  return !remap_settle_active_global;
}

void LoadAndApplyRemapSource(Mesh *pm, MeshBlockPack *pmbp, ParameterInput *dst_pin,
                             bool copy_output_state, const char *banner_label) {
  if (pm == nullptr || pmbp == nullptr || pmbp->phydro == nullptr) return;
  if (!remap_enable_global || remap_restart_source_global.empty()) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Automatic remap requested without a valid remap source path."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  remap::RemapOptions opts;
  if (dst_pin != nullptr && dst_pin->DoesBlockExist("remap")) {
    opts = remap::OptionsFromInput(dst_pin);
  }
  opts.source_path = remap_restart_source_global;
  opts.skip_cell = [](Real x, Real y, Real z) {
    const Real excise_r2 = bh_inner_boundary_global ?
        bh_excise_radius_global * bh_excise_radius_global : -1.0;
    return tde_external::InsideExcisionZone(x, y, z, bh_x_global, bh_y_global,
                                            bh_z_global, excise_r2);
  };
  remap::LoadAndApplyRemap(pm, pmbp, dst_pin, copy_output_state, banner_label, opts,
      [](ParameterInput *src_pin) {
        // The BH/frame record must be restored BEFORE the state is applied: the
        // excision skip hook above reads the restored BH position during the apply.
        if (!RestoreFrameBHStateFromMetadata(src_pin)) {
          std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                    << "remap source restart is missing the BH/frame metadata."
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        // A live BH also keeps its inertial velocity and gas acceleration across the
        // remap.  Without them the leapfrog restarts from rest after every remap and
        // SwitchToFrozenBHInertialFrame places the BH at rest in the new frame.
        RestoreLiveBHMetadata(src_pin);
      });

  if (use_translating_frame_global) {
    frame_state_time_global = pm->time;
    InvalidateFrameBHStepState();
    SyncProblemRuntimeState();
  } else {
    SwitchToFrozenBHInertialFrame(pm, pmbp);
  }

  if (global_variable::my_rank == 0) {
    std::cout << "use translating frame  = "
              << (use_translating_frame_global ? "true" : "false") << std::endl
              << std::endl;
  }
}

// Apply a uniform Galilean boost to the loaded hydro state.
void BoostHydroState(MeshBlockPack *pmbp, const Real dvx, const Real dvy, const Real dvz,
                     const Real rho_boost_min) {
  if (pmbp == nullptr || pmbp->phydro == nullptr) return;
  if (dvx == 0.0 && dvy == 0.0 && dvz == 0.0) return;

  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;
  Real dv2 = dvx*dvx + dvy*dvy + dvz*dvz;

  par_for("tde_boost_hydro_state", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = u0(m, IDN, k, j, i);
    if (!(rho > rho_boost_min)) return;

    Real mx = u0(m, IM1, k, j, i);
    Real my = u0(m, IM2, k, j, i);
    Real mz = u0(m, IM3, k, j, i);
    if (eos_has_energy) {
      u0(m, IEN, k, j, i) += mx*dvx + my*dvy + mz*dvz + 0.5*rho*dv2;
    }
    u0(m, IM1, k, j, i) = mx + rho*dvx;
    u0(m, IM2, k, j, i) = my + rho*dvy;
    u0(m, IM3, k, j, i) = mz + rho*dvz;
  });
}

// Convert a loaded translating-frame restart/remap state into the fixed BH
// inertial-frame mode without changing mesh coordinates.
void SwitchToFrozenBHInertialFrame(Mesh *pm, MeshBlockPack *pmbp) {
  const Real rho_boost_min =
      (pmbp != nullptr &&
          pmbp->phydro != nullptr) ? pmbp->phydro->peos->eos_data.dfloor : 0.0;
  BoostHydroState(pmbp, frame_vx_global, frame_vy_global, frame_vz_global, rho_boost_min);

  // Keep the current mesh coordinates and freeze the BH at its current location.
  bh_inertial_x_global = bh_x_global;
  bh_inertial_y_global = bh_y_global;
  bh_inertial_z_global = bh_z_global;
  bh_vx_global = bh_live_global ? bh_inertial_vx_global : 0.0;
  bh_vy_global = bh_live_global ? bh_inertial_vy_global : 0.0;
  bh_vz_global = bh_live_global ? bh_inertial_vz_global : 0.0;

  orbit_center_x_global = 0.0;
  orbit_center_y_global = 0.0;
  orbit_center_z_global = 0.0;
  frame_vx_global = 0.0;
  frame_vy_global = 0.0;
  frame_vz_global = 0.0;
  frame_ax_global = 0.0;
  frame_ay_global = 0.0;
  frame_az_global = 0.0;
  frame_state_time_global = pm->time;

  InvalidateFrameBHStepState();

  SyncProblemRuntimeState();
}

inline Real DegToRad(const Real deg) {
  return deg * (M_PI/180.0);
}

inline void InterpolateFrameStepState(const Real t, Real &cx, Real &cy, Real &cz,
                                      Real &cvx, Real &cvy, Real &cvz) {
  if (!frame_step_valid_global || frame_step_dt_global <= 0.0) {
    Real dt = t - frame_state_time_global;
    cx = orbit_center_x_global + frame_vx_global*dt + 0.5*frame_ax_global*dt*dt;
    cy = orbit_center_y_global + frame_vy_global*dt + 0.5*frame_ay_global*dt*dt;
    cz = orbit_center_z_global + frame_vz_global*dt + 0.5*frame_az_global*dt*dt;
    cvx = frame_vx_global + frame_ax_global*dt;
    cvy = frame_vy_global + frame_ay_global*dt;
    cvz = frame_vz_global + frame_az_global*dt;
    return;
  }

  Real s = ClampUnitInterval((t - frame_step_time_global) / frame_step_dt_global);
  cx = CubicHermiteValue(frame_step_x0_global, frame_step_vx0_global,
                         frame_step_x1_global, frame_step_vx1_global,
                         frame_step_dt_global, s);
  cy = CubicHermiteValue(frame_step_y0_global, frame_step_vy0_global,
                         frame_step_y1_global, frame_step_vy1_global,
                         frame_step_dt_global, s);
  cz = CubicHermiteValue(frame_step_z0_global, frame_step_vz0_global,
                         frame_step_z1_global, frame_step_vz1_global,
                         frame_step_dt_global, s);
  cvx = CubicHermiteSlope(frame_step_x0_global, frame_step_vx0_global,
                          frame_step_x1_global, frame_step_vx1_global,
                          frame_step_dt_global, s);
  cvy = CubicHermiteSlope(frame_step_y0_global, frame_step_vy0_global,
                          frame_step_y1_global, frame_step_vy1_global,
                          frame_step_dt_global, s);
  cvz = CubicHermiteSlope(frame_step_z0_global, frame_step_vz0_global,
                          frame_step_z1_global, frame_step_vz1_global,
                          frame_step_dt_global, s);
}

inline void FrameCenterAtTime(const Real t, Real &cx, Real &cy, Real &cz) {
  Real cvx = 0.0, cvy = 0.0, cvz = 0.0;
  InterpolateFrameStepState(t, cx, cy, cz, cvx, cvy, cvz);
}

bool MeasureSelectedFrameAcceleration(Mesh *pm,
                                      Real &frame_ax, Real &frame_ay, Real &frame_az) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp == nullptr || pmbp->phydro == nullptr) return false;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ni = ie - is + 1;
  int nj = je - js + 1;
  int nk = ke - ks + 1;
  int nmb = pmbp->nmb_thispack;
  int nmkji = nmb * nk * nj * ni;
  if (nmkji <= 0) return false;

  auto &u0 = pmbp->phydro->u0;
  auto size = pmbp->pmb->mb_size.d_view;
  DvceArray1D<Real> sums_d("tde_frame_sums", 4);
  Kokkos::deep_copy(sums_d, static_cast<Real>(0.0));
  Real bhx = bh_x_global;
  Real bhy = bh_y_global;
  Real bhz = bh_z_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real rho_min = frame_rho_min_global;
  Real soft2 = bh_soft_global * bh_soft_global;
  Real gm = newton_g_global * bh_mass_global;

  par_for("tde_frame_moments", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0 || rho < rho_min) return;

    Real x = CellCenterX(i - is, ni, size(m).x1min, size(m).x1max);
    Real y = CellCenterX(j - js, nj, size(m).x2min, size(m).x2max);
    Real z = CellCenterX(k - ks, nk, size(m).x3min, size(m).x3max);
    Real dx = x - bhx;
    Real dy = y - bhy;
    Real dz = z - bhz;
    if ((dx*dx + dy*dy + dz*dz) <= excise_r2) return;

    Real vol = size(m).dx1 * size(m).dx2 * size(m).dx3;
    Real mass = rho * vol;
    Real r2 = dx*dx + dy*dy + dz*dz + soft2;
    r2 = Kokkos::fmax(r2, static_cast<Real>(1.0e-30));
    Real invr = 1.0 / Kokkos::sqrt(r2);
    Real invr3 = invr * invr * invr;
    Real ax = -gm * dx * invr3;
    Real ay = -gm * dy * invr3;
    Real az = -gm * dz * invr3;

    Kokkos::atomic_add(&sums_d(0), mass);
    Kokkos::atomic_add(&sums_d(1), mass * ax);
    Kokkos::atomic_add(&sums_d(2), mass * ay);
    Kokkos::atomic_add(&sums_d(3), mass * az);
  });

  HostArray1D<Real> sums_h = Kokkos::create_mirror_view(sums_d);
  Kokkos::deep_copy(sums_h, sums_d);
  Real local_buf[4];
  for (int n = 0; n < 4; ++n) local_buf[n] = sums_h(n);
  Real global_buf[4];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local_buf, global_buf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int n = 0; n < 4; ++n) global_buf[n] = local_buf[n];
#endif

  Real total_mass = global_buf[0];
  if (!(total_mass > 0.0) || !std::isfinite(total_mass)) return false;

  Real cm_ax = global_buf[1] / total_mass;
  Real cm_ay = global_buf[2] / total_mass;
  Real cm_az = global_buf[3] / total_mass;
  if (!std::isfinite(cm_ax) || !std::isfinite(cm_ay) || !std::isfinite(cm_az)) {
    return false;
  }

  frame_ax = cm_ax;
  frame_ay = cm_ay;
  frame_az = cm_az;
  return true;
}

constexpr int kMaxStreamLevelTiers = 4;

void RotateAboutY(Real &x, Real &y, Real &z, const Real angle) {
  Real c = std::cos(angle);
  Real s = std::sin(angle);
  Real x_new = c*x + s*z;
  Real z_new = -s*x + c*z;
  x = x_new;
  z = z_new;
}

void RefineBHPosition(MeshBlockPack *pmbp) {
  if (!(bh_force_refine_global || unbound_refine_global ||
      stream_shell_enable_global)) return;

  Mesh *pmesh = pmbp->pmesh;
  const int bh_target_level =
      std::max(pmesh->root_level, pmesh->max_level - bh_force_refine_level_offset_global);
  const int unbound_target_level =
      std::max(pmesh->root_level, pmesh->max_level - unbound_refine_level_offset_global);
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  Real bhx = bh_x_global;
  Real bhy = bh_y_global;
  Real bhz = bh_z_global;

  // Combine the TDE-specific AMR requests with the standard criteria rather than
  // overwriting them outright.  Refinement requests dominate, while derefinement
  // only applies when no other criterion has already asked to refine the block.
  auto request_exact_level = [&](const int gid, const int level,
                                 const int target_level) {
    int &flag = refine_flag.h_view(gid);
    if (level < target_level) {
      flag = 1;
    } else if (level > target_level && flag == 0) {
      flag = -1;
    } else if (level == target_level && flag < 0) {
      flag = 0;
    }
  };
  auto request_derefine = [&](const int gid) {
    if (refine_flag.h_view(gid) == 0) {
      refine_flag.h_view(gid) = -1;
    }
  };
  auto distance_to_interval = [](const Real x, const Real xmin, const Real xmax) {
    if (x < xmin) return xmin - x;
    if (x > xmax) return x - xmax;
    return static_cast<Real>(0.0);
  };
  const Real bh_refine_r2 = (bh_force_refine_global && bh_inner_boundary_global) ?
      bh_excise_radius_global * bh_excise_radius_global : static_cast<Real>(0.0);

  refine_flag.template sync<HostMemSpace>();

  // The per-cell sweeps run on the device.  Every quantity they build is a max, an
  // integer count or a logical "any" over the cells of a block, all of which are
  // order independent, and the per-cell expressions are unchanged, so the refinement
  // flags are the same as the former host sweep over a mirror copy of w0.
  auto &indcs = pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nji = nx2*nx1;
  const int nkji = nx3*nji;
  const int nmb_alloc = std::max(1, nmb);

  std::vector<Real> zone_rmins;
  std::vector<Real> zone_rmaxs;
  std::vector<Real> zone_drs;
  std::vector<Real> zone_rho_fracs;
  std::vector<Real> zone_fill_fracs;
  std::vector<int> zone_offsets;
  std::vector<int> zone_shell_start;
  int nshell = 0;
  int nzone = 0;
  DvceArray1D<Real> zone_rmin_d, zone_rmax_d, zone_dr_d, zone_rho_frac_d;
  DvceArray1D<int> zone_nz_d, zone_start_d, zone_level_d;
  DvceArray1D<Real> shell_rho_d, shell_rho_half_d;
  if (stream_shell_enable_global) {
    Real x1len = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
    Real x2len = pmesh->mesh_size.x2max - pmesh->mesh_size.x2min;
    Real x3len = pmesh->mesh_size.x3max - pmesh->mesh_size.x3min;
    Real rmax = 0.5 * std::sqrt(x1len*x1len + x2len*x2len + x3len*x3len);
    auto zone_block_width = [&](const int target_level) {
      std::int32_t nmbx1 = pmesh->nmb_rootx1 << (target_level - pmesh->root_level);
      std::int32_t nmbx2 = pmesh->nmb_rootx2 << (target_level - pmesh->root_level);
      std::int32_t nmbx3 = pmesh->nmb_rootx3 << (target_level - pmesh->root_level);
      Real bw1 = x1len / static_cast<Real>(std::max<std::int32_t>(1, nmbx1));
      Real bw2 = (pmesh->mesh_indcs.nx2 > 1)
                   ? x2len / static_cast<Real>(std::max<std::int32_t>(1, nmbx2))
                   : bw1;
      Real bw3 = (pmesh->mesh_indcs.nx3 > 1)
                   ? x3len / static_cast<Real>(std::max<std::int32_t>(1, nmbx3))
                   : bw1;
      return std::max(bw1, std::max(bw2, bw3));
    };

    zone_rmins.push_back(0.0);
    zone_offsets.push_back(stream_shell_level_offset_global);
    zone_drs.push_back(stream_shell_dr_global);
    zone_rho_fracs.push_back(stream_shell_rho_frac_global);
    zone_fill_fracs.push_back(stream_shell_fill_frac_global);
    for (std::size_t n = 0; n < stream_shell_level_radii_global.size(); ++n) {
      zone_rmins.push_back(stream_shell_level_radii_global[n]);
      zone_offsets.push_back(stream_shell_level_offsets_global[n]);
      zone_drs.push_back(stream_shell_level_drs_global[n]);
      zone_rho_fracs.push_back(stream_shell_level_rho_fracs_global[n]);
      zone_fill_fracs.push_back(stream_shell_level_fill_fracs_global[n]);
    }
    zone_rmaxs.resize(zone_rmins.size(), rmax);
    for (std::size_t n = 0; n + 1 < zone_rmins.size(); ++n) {
      zone_rmaxs[n] = zone_rmins[n + 1];
    }
    if (stream_shell_r_max_global > 0.0) {
      zone_rmaxs.back() = std::min(zone_rmaxs.back(), stream_shell_r_max_global);
    }
    zone_shell_start.assign(zone_rmins.size(), 0);
    for (std::size_t n = 0; n < zone_rmins.size(); ++n) {
      int target_level =
          std::max(pmesh->root_level, pmesh->max_level - zone_offsets[n]);
      if (zone_drs[n] <= 0.0) {
        // A shell width of one target-block width over-samples the stream radially and
        // makes the shell-by-shell density threshold too jittery after remap. Use a few
        // target-block widths instead so the shell rule traces the stream without
        // triggering widespread fine-level rebuilds at every AMR check.
        zone_drs[n] = std::max(static_cast<Real>(1.0e-12),
            4.0 * zone_block_width(target_level));
      }
      if (n > 0) zone_shell_start[n] = nshell;
      Real span = zone_rmaxs[n] - zone_rmins[n];
      if (span <= 0.0) continue;
      nshell += std::max(1, static_cast<int>(std::ceil(span / zone_drs[n])));
    }
    nshell = std::max(1, nshell);

    nzone = static_cast<int>(zone_rmins.size());
    zone_rmin_d = DvceArray1D<Real>("tde_amr_zone_rmin", nzone);
    zone_rmax_d = DvceArray1D<Real>("tde_amr_zone_rmax", nzone);
    zone_dr_d = DvceArray1D<Real>("tde_amr_zone_dr", nzone);
    zone_rho_frac_d = DvceArray1D<Real>("tde_amr_zone_rho_frac", nzone);
    zone_nz_d = DvceArray1D<int>("tde_amr_zone_nz", nzone);
    zone_start_d = DvceArray1D<int>("tde_amr_zone_start", nzone);
    zone_level_d = DvceArray1D<int>("tde_amr_zone_level", nzone);
    HostArray1D<Real> zone_rmin_h = Kokkos::create_mirror_view(zone_rmin_d);
    HostArray1D<Real> zone_rmax_h = Kokkos::create_mirror_view(zone_rmax_d);
    HostArray1D<Real> zone_dr_h = Kokkos::create_mirror_view(zone_dr_d);
    HostArray1D<Real> zone_rho_frac_h = Kokkos::create_mirror_view(zone_rho_frac_d);
    HostArray1D<int> zone_nz_h = Kokkos::create_mirror_view(zone_nz_d);
    HostArray1D<int> zone_start_h = Kokkos::create_mirror_view(zone_start_d);
    HostArray1D<int> zone_level_h = Kokkos::create_mirror_view(zone_level_d);
    for (int n = 0; n < nzone; ++n) {
      zone_rmin_h(n) = zone_rmins[n];
      zone_rmax_h(n) = zone_rmaxs[n];
      zone_dr_h(n) = zone_drs[n];
      zone_rho_frac_h(n) = zone_rho_fracs[n];
      // Same expression the former locate_shell() evaluated per cell.
      const Real zone_span = zone_rmaxs[n] - zone_rmins[n];
      zone_nz_h(n) =
          std::max(1, static_cast<int>(std::ceil(zone_span / zone_drs[n])));
      zone_start_h(n) = zone_shell_start[n];
      zone_level_h(n) = std::max(pmesh->root_level, pmesh->max_level - zone_offsets[n]);
    }
    Kokkos::deep_copy(zone_rmin_d, zone_rmin_h);
    Kokkos::deep_copy(zone_rmax_d, zone_rmax_h);
    Kokkos::deep_copy(zone_dr_d, zone_dr_h);
    Kokkos::deep_copy(zone_rho_frac_d, zone_rho_frac_h);
    Kokkos::deep_copy(zone_nz_d, zone_nz_h);
    Kokkos::deep_copy(zone_start_d, zone_start_h);
    Kokkos::deep_copy(zone_level_d, zone_level_h);

    shell_rho_d = DvceArray1D<Real>("tde_amr_shell_rho", nshell);
    shell_rho_half_d = DvceArray1D<Real>("tde_amr_shell_rho_half", 2*nshell);
    Kokkos::deep_copy(shell_rho_d, static_cast<Real>(-1.0));
    Kokkos::deep_copy(shell_rho_half_d, static_cast<Real>(-1.0));
  }

  const Real unbound_rho_floor =
      std::max(unbound_refine_rho_min_global, static_cast<Real>(0.0));
  const Real gm_bh = newton_g_global * bh_mass_global;
  const Real bh_soft2 = bh_soft_global * bh_soft_global;
  const Real stream_r_max = stream_shell_r_max_global;
  const Real stream_xsplit = stream_shell_xsplit_radius_global;
  const Real bh_vx = bh_vx_global;
  const Real bh_vy = bh_vy_global;
  const Real bh_vz = bh_vz_global;
  const bool do_stream = stream_shell_enable_global;
  const bool do_unbound = unbound_refine_global;
  // The relative tolerance keeps a product that is an integer up to round-off (for
  // example 0.07*100 = 7.000000000000001) from rounding up to the next cell.
  const int unbound_min_cells = std::max(1, static_cast<int>(
      std::ceil(unbound_refine_fill_frac_global * static_cast<Real>(nkji) *
                (1.0 - 1.0e-12))));

  DvceArray1D<Real> block_rho_max_d("tde_amr_block_rho_max", nmb_alloc);
  DvceArray1D<int> block_unbound_d("tde_amr_block_unbound", nmb_alloc);
  DvceArray1D<int> block_peak_level_d("tde_amr_block_peak_level", nmb_alloc);
  DvceArray1D<int> block_cover_d("tde_amr_block_cover",
                                 nmb_alloc*std::max(1, nzone));
  HostArray1D<Real> block_rho_max_h = Kokkos::create_mirror_view(block_rho_max_d);
  HostArray1D<int> block_unbound_h = Kokkos::create_mirror_view(block_unbound_d);
  HostArray1D<int> block_peak_level_h = Kokkos::create_mirror_view(block_peak_level_d);
  HostArray1D<int> block_cover_h = Kokkos::create_mirror_view(block_cover_d);

  if (nmb > 0 && (do_stream || do_unbound)) {
    auto &w0_ = pmbp->phydro->w0;
    auto rho_max_ = block_rho_max_d;
    auto unbound_ = block_unbound_d;
    auto shell_rho_ = shell_rho_d;
    auto shell_half_ = shell_rho_half_d;
    auto zone_rmin_ = zone_rmin_d;
    auto zone_rmax_ = zone_rmax_d;
    auto zone_dr_ = zone_dr_d;
    auto zone_nz_ = zone_nz_d;
    auto zone_start_ = zone_start_d;
    const int nzone_ = nzone;
    Kokkos::parallel_for("tde_amr_block_stats",
    Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank();
      if (do_stream) {
        Real bmax = 0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &lmax) {
          const int kk = idx/nji;
          const int jj = (idx - kk*nji)/nx1;
          const int ii = idx - kk*nji - jj*nx1;
          const Real rho = w0_(m, IDN, kk+ks, jj+js, ii+is);
          lmax = Kokkos::fmax(lmax, rho);

          const Real x = CellCenterX(ii, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          const Real y = CellCenterX(jj, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
          const Real z = CellCenterX(kk, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
          const Real dx = x - bhx;
          const Real dy = y - bhy;
          const Real dz = z - bhz;
          const Real r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
          if (stream_r_max > 0.0 && r >= stream_r_max) return;
          int tier = 0;
          for (int n = 1; n < nzone_; ++n) {
            if (r >= zone_rmin_(n)) {
              tier = n;
            } else {
              break;
            }
          }
          if (r < zone_rmin_(tier) || r >= zone_rmax_(tier)) return;
          int local_shell = static_cast<int>(
              Kokkos::floor((r - zone_rmin_(tier)) / zone_dr_(tier)));
          local_shell = (local_shell < 0) ? 0 : local_shell;
          local_shell = (local_shell > zone_nz_(tier) - 1) ? (zone_nz_(tier) - 1)
                                                           : local_shell;
          const int ishell = zone_start_(tier) + local_shell;
          // The plain read only skips an atomic that could not raise the entry.
          if (rho > shell_rho_(ishell)) {
            Kokkos::atomic_max(&shell_rho_(ishell), rho);
          }
          if (stream_xsplit > 0.0 && r < stream_xsplit) {
            const int islot = 2*ishell + ((dx < 0.0) ? 0 : 1);
            if (rho > shell_half_(islot)) {
              Kokkos::atomic_max(&shell_half_(islot), rho);
            }
          }
        }, Kokkos::Max<Real>(bmax));
        Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
          rho_max_(m) = Kokkos::fmax(bmax, static_cast<Real>(0.0));
        });
      }
      if (do_unbound) {
        int nunbound = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, int &lcnt) {
          const int kk = idx/nji;
          const int jj = (idx - kk*nji)/nx1;
          const int ii = idx - kk*nji - jj*nx1;
          const int k = kk + ks, j = jj + js, i = ii + is;
          const Real rho = w0_(m, IDN, k, j, i);
          if (rho <= unbound_rho_floor) return;
          const Real x = CellCenterX(ii, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          const Real y = CellCenterX(jj, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
          const Real z = CellCenterX(kk, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
          const Real dx = x - bhx;
          const Real dy = y - bhy;
          const Real dz = z - bhz;
          const Real phi_bh = -gm_bh / Kokkos::sqrt(dx*dx + dy*dy + dz*dz + bh_soft2);
          const Real dvx = w0_(m, IVX, k, j, i) - bh_vx;
          const Real dvy = w0_(m, IVY, k, j, i) - bh_vy;
          const Real dvz = w0_(m, IVZ, k, j, i) - bh_vz;
          const Real eps_orb =
              static_cast<Real>(0.5) * (dvx*dvx + dvy*dvy + dvz*dvz) + phi_bh;
          if (eps_orb >= 0.0) lcnt += 1;
        }, nunbound);
        Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
          unbound_(m) = (nunbound >= unbound_min_cells) ? 1 : 0;
        });
      }
    });
    if (do_stream) Kokkos::deep_copy(block_rho_max_h, block_rho_max_d);
    if (do_unbound) Kokkos::deep_copy(block_unbound_h, block_unbound_d);
  }

  if (do_stream) {
    // Reduce the shell tables across ranks before the per-cell peak/cover test, which
    // is what the former host pass did with local_best_rho / local_best_rho_half.
    HostArray1D<Real> shell_rho_h = Kokkos::create_mirror_view(shell_rho_d);
    HostArray1D<Real> shell_half_h = Kokkos::create_mirror_view(shell_rho_half_d);
    Kokkos::deep_copy(shell_rho_h, shell_rho_d);
    Kokkos::deep_copy(shell_half_h, shell_rho_half_d);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, shell_rho_h.data(), nshell, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, shell_half_h.data(), 2*nshell, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    Kokkos::deep_copy(shell_rho_d, shell_rho_h);
    Kokkos::deep_copy(shell_rho_half_d, shell_half_h);
  }

  Real stream_floor = std::max(stream_shell_rho_floor_global, static_cast<Real>(0.0));
  Real derefine_floor = stream_shell_derefine_dfloor_mult_global *
                        std::max(hydro_dfloor_global, static_cast<Real>(0.0));

  if (nmb > 0 && do_stream) {
    auto &w0_ = pmbp->phydro->w0;
    auto peak_level_ = block_peak_level_d;
    auto cover_ = block_cover_d;
    auto shell_rho_ = shell_rho_d;
    auto shell_half_ = shell_rho_half_d;
    auto zone_rmin_ = zone_rmin_d;
    auto zone_rmax_ = zone_rmax_d;
    auto zone_dr_ = zone_dr_d;
    auto zone_nz_ = zone_nz_d;
    auto zone_start_ = zone_start_d;
    auto zone_rho_frac_ = zone_rho_frac_d;
    auto zone_level_ = zone_level_d;
    const int nzone_ = nzone;
    Kokkos::deep_copy(block_cover_d, 0);
    Kokkos::parallel_for("tde_amr_stream_cover",
    Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank();
      int blevel = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, int &llev) {
        const int kk = idx/nji;
        const int jj = (idx - kk*nji)/nx1;
        const int ii = idx - kk*nji - jj*nx1;
        const Real rho = w0_(m, IDN, kk+ks, jj+js, ii+is);
        if (rho <= stream_floor) return;

        const Real x = CellCenterX(ii, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(jj, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(kk, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
        const Real dx = x - bhx;
        const Real dy = y - bhy;
        const Real dz = z - bhz;
        const Real r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
        if (stream_r_max > 0.0 && r >= stream_r_max) return;
        int tier = 0;
        for (int n = 1; n < nzone_; ++n) {
          if (r >= zone_rmin_(n)) {
            tier = n;
          } else {
            break;
          }
        }
        if (r < zone_rmin_(tier) || r >= zone_rmax_(tier)) return;
        int local_shell = static_cast<int>(
            Kokkos::floor((r - zone_rmin_(tier)) / zone_dr_(tier)));
        local_shell = (local_shell < 0) ? 0 : local_shell;
        local_shell = (local_shell > zone_nz_(tier) - 1) ? (zone_nz_(tier) - 1)
                                                         : local_shell;
        const int ishell = zone_start_(tier) + local_shell;
        const Real rho_shell_max = shell_rho_(ishell);
        if (rho_shell_max <= stream_floor) return;

        Real rho_peak = rho_shell_max;
        if (stream_xsplit > 0.0 && r < stream_xsplit) {
          const Real rho_half_max = shell_half_(2*ishell + ((dx < 0.0) ? 0 : 1));
          if (rho_half_max > stream_floor) rho_peak = rho_half_max;
        }
        if ((rho_peak > stream_floor) && (rho >= rho_peak)) {
          llev = (llev > zone_level_(tier)) ? llev : zone_level_(tier);
        }
        if (rho >= zone_rho_frac_(tier) * rho_peak) {
          Kokkos::atomic_add(&cover_(m*nzone_ + tier), 1);
        }
      }, Kokkos::Max<int>(blevel));
      Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
        peak_level_(m) = (blevel < 0) ? -1 : blevel;
      });
    });
    Kokkos::deep_copy(block_peak_level_h, block_peak_level_d);
    Kokkos::deep_copy(block_cover_h, block_cover_d);
  }

  for (int m = 0; m < nmb; ++m) {
    int gid = mbs + m;
    int level = pmesh->lloc_eachmb[gid].level;

    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min;
    Real x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min;
    Real x3max = size.h_view(m).x3max;

    bool in_x1 = (bhx >= x1min) &&
                 (bhx < x1max ||
                  (x1max >= pmesh->mesh_size.x1max && bhx == x1max));
    bool in_x2 = (bhy >= x2min) &&
                 (bhy < x2max ||
                  (x2max >= pmesh->mesh_size.x2max && bhy == x2max));
    bool in_x3 = (bhz >= x3min) &&
                 (bhz < x3max ||
                 (x3max >= pmesh->mesh_size.x3max && bhz == x3max));
    bool contains_bh = (in_x1 && in_x2 && in_x3);
    bool intersects_bh_sink = contains_bh;
    if (bh_refine_r2 > 0.0) {
      Real dxmin = distance_to_interval(bhx, x1min, x1max);
      Real dymin = distance_to_interval(bhy, x2min, x2max);
      Real dzmin = distance_to_interval(bhz, x3min, x3max);
      intersects_bh_sink =
          (dxmin*dxmin + dymin*dymin + dzmin*dzmin) <= bh_refine_r2;
    }
    // Each rule that applies to this block proposes a target level, and the block is
    // steered to the finest of them: refined if below, derefined if above (unless another
    // criterion asked to refine it).  With stream shells on, a block no rule selects
    // derefines.
    int target_level = -1;
    if (bh_force_refine_global && intersects_bh_sink) {
      target_level = std::max(target_level, bh_target_level);
    }
    if (unbound_refine_global && block_unbound_h(m) != 0) {
      target_level = std::max(target_level, unbound_target_level);
    }
    if (stream_shell_enable_global && block_rho_max_h(m) >= derefine_floor) {
      Real dxmin = distance_to_interval(bhx, x1min, x1max);
      Real dymin = distance_to_interval(bhy, x2min, x2max);
      Real dzmin = distance_to_interval(bhz, x3min, x3max);
      Real drmin = std::sqrt(dxmin*dxmin + dymin*dymin + dzmin*dzmin);
      if (!(stream_shell_r_max_global > 0.0 && drmin >= stream_shell_r_max_global)) {
        if (block_peak_level_h(m) >= 0) {
          // The device pass already took the max of the tier target levels over the
          // shell-peak cells of this block.
          target_level = std::max(target_level,
                                  std::max(pmesh->root_level, block_peak_level_h(m)));
        } else {
          for (int tier = 0; tier < nzone; ++tier) {
            Real fill = static_cast<Real>(block_cover_h(m*nzone + tier)) /
                        static_cast<Real>(std::max(1, nkji));
            if (fill >= zone_fill_fracs[tier]) {
              target_level = std::max(target_level,
                  std::max(pmesh->root_level, pmesh->max_level - zone_offsets[tier]));
            }
          }
        }
      }
    }
    if (target_level >= 0) {
      request_exact_level(gid, level, target_level);
    } else if (stream_shell_enable_global) {
      request_derefine(gid);
    }
  }

  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

void CapHydroLATFactorsNearBH(Mesh *pm, int max_factor, int *lat_factor_eachmb) {
  if (!bh_inner_boundary_global || bh_excise_radius_global <= 0.0 ||
      pm == nullptr || pm->pmb_pack == nullptr || pm->pmb_pack->pmb == nullptr ||
      lat_factor_eachmb == nullptr || max_factor <= 1 || pm->nmb_total <= 0) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_gid = pmbp->pmb->mb_gid;
  const int nmb = pmbp->nmb_thispack;

  auto cap_gid_to_one = [&](const int gid) {
    if (gid < 0 || gid >= pm->nmb_total) return;
    lat_factor_eachmb[gid] = 1;
  };

  auto &nghbr = pmbp->pmb->nghbr;
  const int nnghbr = pmbp->pmb->nnghbr;

  // In the translating frame the BH drifts through the mesh during a window, so protect
  // the neighbourhood it sweeps before the next factor rebuild.
  Real frame_travel = 0.0;
  if (use_translating_frame_global || bh_live_global) {
    const Real window_dt = std::max(static_cast<Real>(0.0), pm->dt)*
        static_cast<Real>(std::max(1, max_factor));
    const Real speed = std::sqrt(SQR(bh_vx_global) + SQR(bh_vy_global) +
                                 SQR(bh_vz_global));
    const Real accel = std::sqrt(SQR(frame_ax_global) + SQR(frame_ay_global) +
                                 SQR(frame_az_global)) +
        (bh_live_global ? std::sqrt(SQR(bh_gas_ax_global) + SQR(bh_gas_ay_global) +
                                    SQR(bh_gas_az_global)) : static_cast<Real>(0.0));
    frame_travel = speed*window_dt + static_cast<Real>(0.5)*accel*window_dt*window_dt;
  }

  for (int m = 0; m < nmb; ++m) {
    const int gid = mb_gid.h_view(m);
    if (gid < 0 || gid >= pm->nmb_total) continue;

    const Real dx =
        (bh_x_global < size.h_view(m).x1min) ? (size.h_view(m).x1min - bh_x_global) :
        ((bh_x_global > size.h_view(m).x1max) ? (bh_x_global - size.h_view(m).x1max) :
         static_cast<Real>(0.0));
    const Real dy =
        (bh_y_global < size.h_view(m).x2min) ? (size.h_view(m).x2min - bh_y_global) :
        ((bh_y_global > size.h_view(m).x2max) ? (bh_y_global - size.h_view(m).x2max) :
         static_cast<Real>(0.0));
    const Real dz =
        (bh_z_global < size.h_view(m).x3min) ? (size.h_view(m).x3min - bh_z_global) :
        ((bh_z_global > size.h_view(m).x3max) ? (bh_z_global - size.h_view(m).x3max) :
         static_cast<Real>(0.0));

    const Real cell_diag = std::sqrt(size.h_view(m).dx1*size.h_view(m).dx1 +
                                     size.h_view(m).dx2*size.h_view(m).dx2 +
                                     size.h_view(m).dx3*size.h_view(m).dx3);
    Real guard_radius = bh_excise_radius_global +
        static_cast<Real>(std::max(1, pm->mb_indcs.ng))*cell_diag;
    if (use_translating_frame_global || bh_live_global) guard_radius += frame_travel;
    if (dx*dx + dy*dy + dz*dz <= guard_radius*guard_radius) {
      cap_gid_to_one(gid);
      for (int n = 0; n < nnghbr; ++n) {
        cap_gid_to_one(nghbr.h_view(m, n).gid);
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lat_factor_eachmb, pm->nmb_total, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
  }
#endif
}

void SyncProblemRuntimeState() {
  // Export the evolving TDE state through the generic runtime interface so core
  // modules do not need to include this pgen directly.
  problem_runtime::SetExcisionState(bh_inner_boundary_global, bh_excise_radius_global,
                                    bh_excise_density_global, bh_excise_eint_global);
  problem_runtime::SetLiveBHState(use_translating_frame_global,
                                  bh_x_global, bh_y_global, bh_z_global,
                                  bh_vx_global, bh_vy_global, bh_vz_global,
                                  orbit_center_x_global, orbit_center_y_global,
                                  orbit_center_z_global,
                                  frame_vx_global, frame_vy_global, frame_vz_global);
  problem_runtime::SetExternalBHPotential(true, bh_mass_global, bh_soft_global,
                                          newton_g_global);
  problem_runtime::SetExternalBHGravitySourceCoupling(true);
  problem_runtime::SetFrameStepState(frame_step_valid_global,
                                     frame_step_time_global, frame_step_dt_global,
                                     frame_step_x0_global, frame_step_y0_global,
                                     frame_step_z0_global,
                                     frame_step_vx0_global, frame_step_vy0_global,
                                     frame_step_vz0_global,
                                     frame_step_ax0_global, frame_step_ay0_global,
                                     frame_step_az0_global,
                                     frame_step_x1_global, frame_step_y1_global,
                                     frame_step_z1_global,
                                     frame_step_vx1_global, frame_step_vy1_global,
                                     frame_step_vz1_global,
                                     frame_step_ax1_global, frame_step_ay1_global,
                                     frame_step_az1_global);
  problem_runtime::SetBHStepState(bh_step_valid_global, bh_step_time_global,
      bh_step_dt_global,
                                  bh_step_x0_global, bh_step_y0_global, bh_step_z0_global,
                                  bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global,
                                  bh_step_x1_global, bh_step_y1_global, bh_step_z1_global,
                                  bh_step_vx1_global, bh_step_vy1_global, bh_step_vz1_global);
}

void InvalidateFrameBHStepState() {
  frame_step_valid_global = false;
  frame_step_cycle_global = -1;
  frame_step_time_global = 0.0;
  frame_step_dt_global = 0.0;
  bh_step_valid_global = false;
  bh_step_cycle_global = -1;
  bh_step_time_global = 0.0;
  bh_step_dt_global = 0.0;
}

void RestoreOptionalFrameBHStepState(ParameterInput *pin) {
  InvalidateFrameBHStepState();
  if (pin == nullptr) return;
  if (!pin->DoesParameterExist("problem", "frame_step_state_valid") ||
      !pin->DoesParameterExist("problem", "bh_step_state_valid")) {
    return;
  }
  if (!pin->GetBoolean("problem", "frame_step_state_valid") ||
      !pin->GetBoolean("problem", "bh_step_state_valid")) {
    return;
  }

  const char *frame_keys[] = {
      "frame_step_time", "frame_step_dt",
      "frame_step_x0", "frame_step_y0", "frame_step_z0",
      "frame_step_vx0", "frame_step_vy0", "frame_step_vz0",
      "frame_step_ax0", "frame_step_ay0", "frame_step_az0",
      "frame_step_x1", "frame_step_y1", "frame_step_z1",
      "frame_step_vx1", "frame_step_vy1", "frame_step_vz1",
      "frame_step_ax1", "frame_step_ay1", "frame_step_az1"};
  const char *bh_keys[] = {
      "bh_step_time", "bh_step_dt",
      "bh_step_x0", "bh_step_y0", "bh_step_z0",
      "bh_step_vx0", "bh_step_vy0", "bh_step_vz0",
      "bh_step_x1", "bh_step_y1", "bh_step_z1",
      "bh_step_vx1", "bh_step_vy1", "bh_step_vz1"};
  for (const char *key : frame_keys) {
    if (!pin->DoesParameterExist("problem", key)) return;
  }
  for (const char *key : bh_keys) {
    if (!pin->DoesParameterExist("problem", key)) return;
  }

  frame_step_time_global = pin->GetReal("problem", "frame_step_time");
  frame_step_dt_global = pin->GetReal("problem", "frame_step_dt");
  bh_step_time_global = pin->GetReal("problem", "bh_step_time");
  bh_step_dt_global = pin->GetReal("problem", "bh_step_dt");
  if (frame_step_dt_global <= 0.0 || bh_step_dt_global <= 0.0) {
    InvalidateFrameBHStepState();
    return;
  }

  frame_step_x0_global = pin->GetReal("problem", "frame_step_x0");
  frame_step_y0_global = pin->GetReal("problem", "frame_step_y0");
  frame_step_z0_global = pin->GetReal("problem", "frame_step_z0");
  frame_step_vx0_global = pin->GetReal("problem", "frame_step_vx0");
  frame_step_vy0_global = pin->GetReal("problem", "frame_step_vy0");
  frame_step_vz0_global = pin->GetReal("problem", "frame_step_vz0");
  frame_step_ax0_global = pin->GetReal("problem", "frame_step_ax0");
  frame_step_ay0_global = pin->GetReal("problem", "frame_step_ay0");
  frame_step_az0_global = pin->GetReal("problem", "frame_step_az0");
  frame_step_x1_global = pin->GetReal("problem", "frame_step_x1");
  frame_step_y1_global = pin->GetReal("problem", "frame_step_y1");
  frame_step_z1_global = pin->GetReal("problem", "frame_step_z1");
  frame_step_vx1_global = pin->GetReal("problem", "frame_step_vx1");
  frame_step_vy1_global = pin->GetReal("problem", "frame_step_vy1");
  frame_step_vz1_global = pin->GetReal("problem", "frame_step_vz1");
  frame_step_ax1_global = pin->GetReal("problem", "frame_step_ax1");
  frame_step_ay1_global = pin->GetReal("problem", "frame_step_ay1");
  frame_step_az1_global = pin->GetReal("problem", "frame_step_az1");

  bh_step_x0_global = pin->GetReal("problem", "bh_step_x0");
  bh_step_y0_global = pin->GetReal("problem", "bh_step_y0");
  bh_step_z0_global = pin->GetReal("problem", "bh_step_z0");
  bh_step_vx0_global = pin->GetReal("problem", "bh_step_vx0");
  bh_step_vy0_global = pin->GetReal("problem", "bh_step_vy0");
  bh_step_vz0_global = pin->GetReal("problem", "bh_step_vz0");
  bh_step_x1_global = pin->GetReal("problem", "bh_step_x1");
  bh_step_y1_global = pin->GetReal("problem", "bh_step_y1");
  bh_step_z1_global = pin->GetReal("problem", "bh_step_z1");
  bh_step_vx1_global = pin->GetReal("problem", "bh_step_vx1");
  bh_step_vy1_global = pin->GetReal("problem", "bh_step_vy1");
  bh_step_vz1_global = pin->GetReal("problem", "bh_step_vz1");

  frame_step_valid_global = true;
  bh_step_valid_global = true;
}

inline Real ClampUnitInterval(const Real x) {
  return std::fmax(0.0, std::fmin(1.0, x));
}

inline void LaneEmdenRHS(const Real xi, const Real theta, const Real dtheta,
                         const Real n, Real &dtheta_dxi, Real &ddtheta_dxi) {
  dtheta_dxi = dtheta;
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

void InterpolateBHStepState(const Real t, Real &x, Real &y, Real &z,
                            Real &vx, Real &vy, Real &vz) {
  if (!bh_step_valid_global || bh_step_dt_global <= 0.0) {
    x = bh_x_global;
    y = bh_y_global;
    z = bh_z_global;
    vx = bh_vx_global;
    vy = bh_vy_global;
    vz = bh_vz_global;
    return;
  }

  Real s = ClampUnitInterval((t - bh_step_time_global) / bh_step_dt_global);
  x = CubicHermiteValue(bh_step_x0_global, bh_step_vx0_global,
                        bh_step_x1_global, bh_step_vx1_global, bh_step_dt_global, s);
  y = CubicHermiteValue(bh_step_y0_global, bh_step_vy0_global,
                        bh_step_y1_global, bh_step_vy1_global, bh_step_dt_global, s);
  z = CubicHermiteValue(bh_step_z0_global, bh_step_vz0_global,
                        bh_step_z1_global, bh_step_vz1_global, bh_step_dt_global, s);
  vx = CubicHermiteSlope(bh_step_x0_global, bh_step_vx0_global,
                         bh_step_x1_global, bh_step_vx1_global, bh_step_dt_global, s);
  vy = CubicHermiteSlope(bh_step_y0_global, bh_step_vy0_global,
                         bh_step_y1_global, bh_step_vy1_global, bh_step_dt_global, s);
  vz = CubicHermiteSlope(bh_step_z0_global, bh_step_vz0_global,
                         bh_step_z1_global, bh_step_vz1_global, bh_step_dt_global, s);
}

inline void FrameAccelerationAtTime(const Real t, Real &ax, Real &ay, Real &az) {
  if (!frame_step_valid_global || frame_step_dt_global <= 0.0) {
    ax = frame_ax_global;
    ay = frame_ay_global;
    az = frame_az_global;
    return;
  }

  Real s = ClampUnitInterval((t - frame_step_time_global) / frame_step_dt_global);
  ax = (1.0 - s) * frame_step_ax0_global + s * frame_step_ax1_global;
  ay = (1.0 - s) * frame_step_ay0_global + s * frame_step_ay1_global;
  az = (1.0 - s) * frame_step_az0_global + s * frame_step_az1_global;
}

//----------------------------------------------------------------------------------------
//! \brief Prepare the frame and BH state for the current timestep.
//!
//! In translating-frame mode, the frame position/velocity are integrated
//! continuously in time, while the frame acceleration is taken to be the
//! mass-weighted BH acceleration of density-selected debris. This removes the
//! bulk COM acceleration without introducing per-cycle coordinate resets.
//!
//! In fixed-frame mode, the frame state is identically zero and the BH remains
//! fixed at its current mesh position.
//!
//! The frame state for the current timestep is:
//!   1. Use the existing continuous frame center/velocity as the step start state
//!   2. Measure the selected-gas mean BH acceleration
//!   3. Advance the frame center/velocity with that acceleration over dt
//!   4. Derive BH position in simulation frame:  bh_sim = bh_inertial - frame_center
//!
//! The start (_x0) and end (_x1) state pairs are stored so that
//! InterpolateBHStepState / InterpolateFrameStepState can provide sub-step
//! values via cubic Hermite interpolation during RK stages.
//!
//! Without LAT, called once per cycle from TDEExternalGravitySource (guarded by
//! cycle/time match).  With LAT the translating frame is advanced per window by
//! PrepareTranslatingFrameLATWindow instead.

void AdvanceTranslatingFrameStep(Mesh *pm, const Real step_dt,
                                 const bool lat_window = false);
void AdvanceFrozenFrameLiveBHStep(Mesh *pm, const Real step_dt);

// Gas acceleration of the BH at a simulation-frame position: the reciprocal
// (volume-weighted opposite gas) force or the self-gravity sample.  MPI collectives.
bool SampleBHGasAcceleration(Mesh *pm, const Real x, const Real y, const Real z,
                             Real &ax, Real &ay, Real &az) {
  if (bh_reciprocal_force_global) {
    return bh_dynamics::SampleReciprocalAcceleration(pm, x, y, z, bh_mass_global,
        bh_soft_global, newton_g_global, bh_analytic_pair_global, ax, ay, az);
  }
  return bh_dynamics::SampleSelfGravityAcceleration(pm, x, y, z, ax, ay, az);
}

//! \brief Advance the live BH inertial state over [pm->time, pm->time + step_dt] and
//! store its simulation-frame step pair.  f0/fv0 and f1/fv1 are the frame center and
//! velocity at the step ends (zero without the translating frame).  The orbit step is
//! star_bh_collision's leapfrog: acceleration sampled at the start, then kick-drift-kick
//! with the endpoint force sampled at the drifted position against the current gas.
//! Collectives: synchronized points only.
void AdvanceLiveBHStep(Mesh *pm, const Real step_dt, const Real f0[3], const Real fv0[3],
                       const Real f1[3], const Real fv1[3]) {
  const std::array<Real, 3> x0 = {bh_inertial_x_global, bh_inertial_y_global,
                                  bh_inertial_z_global};
  const std::array<Real, 3> v0 = {bh_inertial_vx_global, bh_inertial_vy_global,
                                  bh_inertial_vz_global};
  std::array<Real, 3> a0 = {bh_gas_ax_global, bh_gas_ay_global, bh_gas_az_global};
  const std::array<Real, 3> previous_a = a0;
  // The gas potential is not solved yet on a new mesh (the first step after a remap).
  // The step then keeps the gas acceleration carried by the source restart.
  const bool phi_ready = pm->pmb_pack == nullptr || pm->pmb_pack->pgrav == nullptr ||
                         pm->pmb_pack->pgrav->phi_valid;
  if (!phi_ready) {
    a0 = previous_a;
  } else if (SampleBHGasAcceleration(pm, x0[0] - f0[0], x0[1] - f0[1], x0[2] - f0[2],
                                     a0[0], a0[1], a0[2])) {
    bh_gas_ax_global = a0[0];
    bh_gas_ay_global = a0[1];
    bh_gas_az_global = a0[2];
  } else if (pm->pmb_pack != nullptr && pm->pmb_pack->pgrav != nullptr) {
    std::cerr << "### FATAL ERROR in AdvanceLiveBHStep\n"
              << "No valid gas-potential acceleration sample at the BH position.\n";
    std::exit(EXIT_FAILURE);
  } else {
    a0 = previous_a;
  }
  std::array<Real, 3> x1, v1;
  {
    const Real end_frame[3] = {f1[0], f1[1], f1[2]};
    auto force = [pm, end_frame, phi_ready, previous_a](
                     const std::array<Real, 3> &position, std::array<Real, 3> &a) {
      if (pm->pmb_pack->pgrav == nullptr) {
        a = {0.0, 0.0, 0.0};
        return true;
      }
      if (!phi_ready) {
        a = previous_a;
        return true;
      }
      return SampleBHGasAcceleration(pm, position[0] - end_frame[0],
          position[1] - end_frame[1], position[2] - end_frame[2], a[0], a[1], a[2]);
    };
    if (!star_bh_orbit::LeapfrogKDK(x0, v0, a0, step_dt, force, x1, v1)) {
      std::cerr << "### FATAL ERROR: invalid leapfrog BH endpoint force or state\n";
      std::exit(EXIT_FAILURE);
    }
  }
  bh_step_x0_global = x0[0] - f0[0];
  bh_step_y0_global = x0[1] - f0[1];
  bh_step_z0_global = x0[2] - f0[2];
  bh_step_vx0_global = v0[0] - fv0[0];
  bh_step_vy0_global = v0[1] - fv0[1];
  bh_step_vz0_global = v0[2] - fv0[2];
  bh_step_x1_global = x1[0] - f1[0];
  bh_step_y1_global = x1[1] - f1[1];
  bh_step_z1_global = x1[2] - f1[2];
  bh_step_vx1_global = v1[0] - fv1[0];
  bh_step_vy1_global = v1[1] - fv1[1];
  bh_step_vz1_global = v1[2] - fv1[2];
  for (int d = 0; d < 3; ++d) {
    bh_window_start_v[d] = v0[d];
    bh_window_frame_dv[d] = fv1[d] - fv0[d];
    bh_window_frame_v1[d] = fv1[d];
  }
  bh_inertial_x_global = x1[0];
  bh_inertial_y_global = x1[1];
  bh_inertial_z_global = x1[2];
  bh_inertial_vx_global = v1[0];
  bh_inertial_vy_global = v1[1];
  bh_inertial_vz_global = v1[2];
  bh_x_global = bh_step_x1_global;
  bh_y_global = bh_step_y1_global;
  bh_z_global = bh_step_z1_global;
  bh_vx_global = bh_step_vx1_global;
  bh_vy_global = bh_step_vy1_global;
  bh_vz_global = bh_step_vz1_global;
}

// Work of the frame's fictitious force on the live BH over the step just taken:
// Delta K_sim = M v_avg . (Delta v_inertial - Delta v_frame), and the frame part is
// -M v_avg . Delta v_frame exactly (v_avg the mean of the step-end simulation-frame
// velocities).  Call once the step-end velocity is final.
void AccountLiveBHFrameWork() {
  if (!energy_ledger_global || !bh_live_global) return;
  const Real vs0[3] = {bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global};
  const Real vs1[3] = {bh_vx_global, bh_vy_global, bh_vz_global};
  Real work = 0.0;
  for (int d = 0; d < 3; ++d) {
    work -= bh_mass_global*0.5*(vs0[d] + vs1[d])*bh_window_frame_dv[d];
  }
  frame_work_total += work;
}

void PrepareFrameBHStepState(Mesh *pm) {
  if (pm->dt <= 0.0) return;
  if (!use_translating_frame_global && bh_live_global) {
    // Live BH in the frozen (non-translating) frame.  With LAT the window hook owns it.
    if (hydro_lat_enabled_global) return;
    if (frame_step_valid_global && bh_step_valid_global &&
        bh_step_cycle_global == pm->ncycle && bh_step_time_global == pm->time &&
        bh_step_dt_global == pm->dt) {
      return;
    }
    AdvanceFrozenFrameLiveBHStep(pm, pm->dt);
    return;
  }
  if (!use_translating_frame_global) {
    if (frame_step_valid_global && bh_step_valid_global &&
        frame_step_cycle_global == pm->ncycle && bh_step_cycle_global == pm->ncycle &&
        frame_step_time_global == pm->time && bh_step_time_global == pm->time &&
        frame_step_dt_global == pm->dt && bh_step_dt_global == pm->dt) {
      return;
    }

    frame_step_cycle_global = pm->ncycle;
    frame_step_time_global = pm->time;
    frame_step_dt_global = pm->dt;
    bh_step_cycle_global = pm->ncycle;
    bh_step_time_global = pm->time;
    bh_step_dt_global = pm->dt;

    frame_step_x0_global = 0.0;
    frame_step_y0_global = 0.0;
    frame_step_z0_global = 0.0;
    frame_step_vx0_global = 0.0;
    frame_step_vy0_global = 0.0;
    frame_step_vz0_global = 0.0;
    frame_step_ax0_global = 0.0;
    frame_step_ay0_global = 0.0;
    frame_step_az0_global = 0.0;
    frame_step_x1_global = 0.0;
    frame_step_y1_global = 0.0;
    frame_step_z1_global = 0.0;
    frame_step_vx1_global = 0.0;
    frame_step_vy1_global = 0.0;
    frame_step_vz1_global = 0.0;
    frame_step_ax1_global = 0.0;
    frame_step_ay1_global = 0.0;
    frame_step_az1_global = 0.0;

    bh_step_x0_global = bh_x_global;
    bh_step_y0_global = bh_y_global;
    bh_step_z0_global = bh_z_global;
    bh_step_vx0_global = 0.0;
    bh_step_vy0_global = 0.0;
    bh_step_vz0_global = 0.0;
    bh_step_x1_global = bh_x_global;
    bh_step_y1_global = bh_y_global;
    bh_step_z1_global = bh_z_global;
    bh_step_vx1_global = 0.0;
    bh_step_vy1_global = 0.0;
    bh_step_vz1_global = 0.0;

    frame_step_valid_global = true;
    bh_step_valid_global = true;
    SyncProblemRuntimeState();
    return;
  }

  // With LAT the frame is advanced once per synchronization window by
  // PrepareTranslatingFrameLATWindow; LAT source and fixup passes only read it.
  if (hydro_lat_enabled_global) return;

  if (frame_step_valid_global && bh_step_valid_global &&
      frame_step_cycle_global == pm->ncycle && bh_step_cycle_global == pm->ncycle &&
      frame_step_time_global == pm->time && bh_step_time_global == pm->time &&
      frame_step_dt_global == pm->dt && bh_step_dt_global == pm->dt) {
    return;
  }

  AdvanceTranslatingFrameStep(pm, pm->dt);
}

//----------------------------------------------------------------------------------------
//! \brief Advance the translating frame over [pm->time, pm->time + step_dt].
//!
//! Measures the selected-gas mean BH acceleration with an MPI collective, so every rank
//! must call it at the same mesh time: once per cycle without LAT, once per LAT window
//! from the driver's common-time hook with LAT.  The acceleration is held constant over
//! the step, so the stored Hermite pair reproduces the quadratic frame trajectory exactly
//! at every block's own stage time inside it.

void AdvanceTranslatingFrameStep(Mesh *pm, const Real step_dt, const bool lat_window) {
  // A LAT window holds the frame acceleration for many fine steps, so there it varies
  // linearly, extrapolated from the previous window's sample (bootstrapped after a
  // restart or a window more than twice the previous one); the frame trajectory is then
  // second order in the window length.  The per-cycle path keeps its constant sample.
  const Real previous_time = frame_step_time_global;
  const bool previous_valid = lat_window && frame_step_valid_global;
  const Real previous_ax = frame_step_ax0_global;
  const Real previous_ay = frame_step_ay0_global;
  const Real previous_az = frame_step_az0_global;
  frame_step_cycle_global = pm->ncycle;
  frame_step_time_global = pm->time;
  frame_step_dt_global = step_dt;
  bh_step_cycle_global = pm->ncycle;
  bh_step_time_global = pm->time;
  bh_step_dt_global = step_dt;

  frame_step_x0_global = orbit_center_x_global;
  frame_step_y0_global = orbit_center_y_global;
  frame_step_z0_global = orbit_center_z_global;
  frame_step_vx0_global = frame_vx_global;
  frame_step_vy0_global = frame_vy_global;
  frame_step_vz0_global = frame_vz_global;
  frame_step_ax0_global = frame_ax_global;
  frame_step_ay0_global = frame_ay_global;
  frame_step_az0_global = frame_az_global;

  Real measured_ax = 0.0;
  Real measured_ay = 0.0;
  Real measured_az = 0.0;
  if (MeasureSelectedFrameAcceleration(pm, measured_ax, measured_ay, measured_az)) {
    frame_step_ax0_global = measured_ax;
    frame_step_ay0_global = measured_ay;
    frame_step_az0_global = measured_az;
  }

  if (!bh_live_global) {
    bh_step_x0_global = bh_inertial_x_global - frame_step_x0_global;
    bh_step_y0_global = bh_inertial_y_global - frame_step_y0_global;
    bh_step_z0_global = bh_inertial_z_global - frame_step_z0_global;
    bh_step_vx0_global = -frame_step_vx0_global;
    bh_step_vy0_global = -frame_step_vy0_global;
    bh_step_vz0_global = -frame_step_vz0_global;
  }

  // Advance the continuous translating frame over this timestep using the
  // selected-gas mean BH acceleration. This removes the selected debris COM
  // acceleration without resetting the coordinates to the instantaneous COM.
  const Real previous_dt = pm->time - previous_time;
  const Real time_tol = static_cast<Real>(1024.0)*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::abs(pm->time));
  if (previous_valid && previous_dt > time_tol && step_dt <= 2.0*previous_dt) {
    const Real dt = step_dt;
    const Real jx = (frame_step_ax0_global - previous_ax)/previous_dt;
    const Real jy = (frame_step_ay0_global - previous_ay)/previous_dt;
    const Real jz = (frame_step_az0_global - previous_az)/previous_dt;
    frame_step_ax1_global = frame_step_ax0_global + jx*dt;
    frame_step_ay1_global = frame_step_ay0_global + jy*dt;
    frame_step_az1_global = frame_step_az0_global + jz*dt;
    frame_step_vx1_global = frame_step_vx0_global + dt*frame_step_ax0_global + 0.5*dt*dt*jx;
    frame_step_vy1_global = frame_step_vy0_global + dt*frame_step_ay0_global + 0.5*dt*dt*jy;
    frame_step_vz1_global = frame_step_vz0_global + dt*frame_step_az0_global + 0.5*dt*dt*jz;
    frame_step_x1_global = frame_step_x0_global + dt*frame_step_vx0_global +
                           0.5*dt*dt*frame_step_ax0_global + dt*dt*dt*jx/6.0;
    frame_step_y1_global = frame_step_y0_global + dt*frame_step_vy0_global +
                           0.5*dt*dt*frame_step_ay0_global + dt*dt*dt*jy/6.0;
    frame_step_z1_global = frame_step_z0_global + dt*frame_step_vz0_global +
                           0.5*dt*dt*frame_step_az0_global + dt*dt*dt*jz/6.0;
  } else {
    frame_step_ax1_global = frame_step_ax0_global;
    frame_step_ay1_global = frame_step_ay0_global;
    frame_step_az1_global = frame_step_az0_global;
    frame_step_vx1_global = frame_step_vx0_global + frame_step_ax0_global*step_dt;
    frame_step_vy1_global = frame_step_vy0_global + frame_step_ay0_global*step_dt;
    frame_step_vz1_global = frame_step_vz0_global + frame_step_az0_global*step_dt;
    frame_step_x1_global = frame_step_x0_global +
                           0.5*(frame_step_vx0_global + frame_step_vx1_global)*step_dt;
    frame_step_y1_global = frame_step_y0_global +
                           0.5*(frame_step_vy0_global + frame_step_vy1_global)*step_dt;
    frame_step_z1_global = frame_step_z0_global +
                           0.5*(frame_step_vz0_global + frame_step_vz1_global)*step_dt;
  }

  // End-of-step BH in simulation frame = bh_inertial - frame_center
  if (bh_live_global) {
    const Real f0[3] = {frame_step_x0_global, frame_step_y0_global, frame_step_z0_global};
    const Real fv0[3] = {frame_step_vx0_global, frame_step_vy0_global,
                         frame_step_vz0_global};
    const Real f1[3] = {frame_step_x1_global, frame_step_y1_global, frame_step_z1_global};
    const Real fv1[3] = {frame_step_vx1_global, frame_step_vy1_global,
                         frame_step_vz1_global};
    AdvanceLiveBHStep(pm, step_dt, f0, fv0, f1, fv1);
  } else {
    bh_step_vx1_global = -frame_step_vx1_global;
    bh_step_vy1_global = -frame_step_vy1_global;
    bh_step_vz1_global = -frame_step_vz1_global;
    bh_step_x1_global = bh_inertial_x_global - frame_step_x1_global;
    bh_step_y1_global = bh_inertial_y_global - frame_step_y1_global;
    bh_step_z1_global = bh_inertial_z_global - frame_step_z1_global;
  }

  frame_step_valid_global = true;
  bh_step_valid_global = true;

  // Commit end-of-step frame state for the next cycle's start-of-step
  orbit_center_x_global = frame_step_x1_global;
  orbit_center_y_global = frame_step_y1_global;
  orbit_center_z_global = frame_step_z1_global;
  frame_vx_global = frame_step_vx1_global;
  frame_vy_global = frame_step_vy1_global;
  frame_vz_global = frame_step_vz1_global;
  frame_ax_global = frame_step_ax1_global;
  frame_ay_global = frame_step_ay1_global;
  frame_az_global = frame_step_az1_global;
  frame_state_time_global = pm->time + step_dt;

  // Commit end-of-step BH state (used by metadata, AMR, and excision)
  bh_x_global = bh_step_x1_global;
  bh_y_global = bh_step_y1_global;
  bh_z_global = bh_step_z1_global;
  bh_vx_global = bh_step_vx1_global;
  bh_vy_global = bh_step_vy1_global;
  bh_vz_global = bh_step_vz1_global;
  if (!bh_reciprocal_force_global) AccountLiveBHFrameWork();
  SyncProblemRuntimeState();
}

//! \brief Live BH step without a translating frame (frozen BH-inertial coordinates).
void AdvanceFrozenFrameLiveBHStep(Mesh *pm, const Real step_dt) {
  frame_step_cycle_global = pm->ncycle;
  frame_step_time_global = pm->time;
  frame_step_dt_global = step_dt;
  bh_step_cycle_global = pm->ncycle;
  bh_step_time_global = pm->time;
  bh_step_dt_global = step_dt;
  frame_step_x0_global = frame_step_y0_global = frame_step_z0_global = 0.0;
  frame_step_vx0_global = frame_step_vy0_global = frame_step_vz0_global = 0.0;
  frame_step_ax0_global = frame_step_ay0_global = frame_step_az0_global = 0.0;
  frame_step_x1_global = frame_step_y1_global = frame_step_z1_global = 0.0;
  frame_step_vx1_global = frame_step_vy1_global = frame_step_vz1_global = 0.0;
  frame_step_ax1_global = frame_step_ay1_global = frame_step_az1_global = 0.0;
  const Real zero[3] = {0.0, 0.0, 0.0};
  AdvanceLiveBHStep(pm, step_dt, zero, zero, zero, zero);
  frame_step_valid_global = true;
  bh_step_valid_global = true;
  SyncProblemRuntimeState();
}

//----------------------------------------------------------------------------------------
//! \brief LAT window hook: advance the translating frame over the whole window.
//!
//! The driver calls this on every rank at the window's common start time, after the
//! window length is final and before any rank-local bin runs, so the collective in
//! MeasureSelectedFrameAcceleration is safe here and nowhere inside the window.

void PrepareTranslatingFrameLATWindow(Mesh *pm, const Real window_dt) {
  if ((!use_translating_frame_global && !bh_live_global) || pm == nullptr ||
      window_dt <= 0.0) {
    return;
  }
  const Real t0 = pm->time;
  const Real t1 = t0 + window_dt;
  const Real time_tol = static_cast<Real>(1024.0)*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::max(std::abs(t0), std::abs(t1)));
  if (frame_step_valid_global && bh_step_valid_global &&
      frame_step_time_global <= t0 + time_tol &&
      frame_step_time_global + frame_step_dt_global >= t1 - time_tol) {
    return;
  }
  if (use_translating_frame_global) {
    AdvanceTranslatingFrameStep(pm, window_dt, true);
  } else {
    AdvanceFrozenFrameLiveBHStep(pm, window_dt);
  }
  if (bh_reciprocal_force_global) {
    if (bh_impulse_window_open) {
      std::cerr << "Previous BH impulse window was not reconciled\n";
      std::exit(EXIT_FAILURE);
    }
    for (Real &value : bh_local_gas_impulse) value = 0.0;
    bh_impulse_window_open = true;
  }
}

// Rank-local stage impulse of the BH force on the gas (no MPI).
void AccumulateTDEBHGasImpulse(Mesh *pm, const Real final_dt) {
  if (!bh_impulse_window_open) {
    std::cerr << "BH impulse ledger used outside its synchronization window\n";
    std::exit(EXIT_FAILURE);
  }
  bh_dynamics::AccumulateStageGasImpulse(pm, final_dt, bh_analytic_pair_global,
                                         bh_local_gas_impulse);
}

// LAT window end, after every bin and reflux and before output/AMR: reconcile the BH
// velocity with the gas impulse (as star_bh_collision) and reduce the frame work.
void CompleteTDELATWindow(Mesh *pm) {
  if (bh_reciprocal_force_global) {
    const Real scale = std::max(static_cast<Real>(1.0), std::abs(pm->time));
    const Real tolerance = 2048.0*std::numeric_limits<Real>::epsilon()*scale;
    if (!bh_impulse_window_open ||
        std::abs(pm->time-bh_step_time_global-bh_step_dt_global) > tolerance) {
      std::cerr << "BH impulse reconciliation requires a complete synchronized LAT "
                << "window\n";
      std::exit(EXIT_FAILURE);
    }
    Real impulse[3] = {bh_local_gas_impulse[0], bh_local_gas_impulse[1],
                       bh_local_gas_impulse[2]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(bh_local_gas_impulse, impulse, 3, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    Real end_v[3];
    bh_last_impulse_residual = 0.0;
    for (int d = 0; d < 3; ++d) {
      end_v[d] = bh_window_start_v[d]-impulse[d]/bh_mass_global;
      bh_total_gas_impulse[d] += impulse[d];
      const Real residual = bh_mass_global*(end_v[d]-bh_window_start_v[d])+impulse[d];
      bh_last_impulse_residual = std::max(bh_last_impulse_residual, std::abs(residual));
      bh_local_gas_impulse[d] = 0.0;
    }
    // Keep the already-used position predictor; the next window starts from the
    // corrected velocity.
    bh_inertial_vx_global = end_v[0];
    bh_inertial_vy_global = end_v[1];
    bh_inertial_vz_global = end_v[2];
    bh_vx_global = end_v[0] - bh_window_frame_v1[0];
    bh_vy_global = end_v[1] - bh_window_frame_v1[1];
    bh_vz_global = end_v[2] - bh_window_frame_v1[2];
    bh_impulse_window_open = false;
    AccountLiveBHFrameWork();
  }
  if (energy_ledger_global) {
    Real work = frame_work_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&frame_work_local, &work, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    frame_work_total += work;
    frame_work_local = 0.0;
  }
  SyncProblemRuntimeState();
}

// Sum of u0(IEN) dV over the blocks this source pass updates.
Real ActiveSourceGasEnergy(MeshBlockPack *pmbp, const int nwork1) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int per_block = nx1*nx2*nx3;
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const auto active = pmbp->lat_active_indices.d_view;
  const auto size = pmbp->pmb->mb_size.d_view;
  const auto u0 = pmbp->phydro->u0;
  Real total = 0.0;
  Kokkos::parallel_reduce("tde_source_energy",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, (nwork1 + 1)*per_block),
  KOKKOS_LAMBDA(const int index, Real &sum) {
    const int a = index/per_block;
    const int m = lat_enabled ? active(a) : a;
    const int cell = index - a*per_block;
    const int k = cell/(nx1*nx2) + ks;
    const int j = (cell/nx1)%nx2 + js;
    const int i = cell%nx1 + is;
    sum += u0(m, IEN, k, j, i)*size(m).dx1*size(m).dx2*size(m).dx3;
  }, Kokkos::Sum<Real>(total));
  return total;
}

//! \brief History: the shared energy ledger columns (simulation-frame K_BH and BH
//! state) followed by W_frame, the cumulative fictitious-force work on gas and BH.
//! Closure: tot-E + K_BH + U_bhgas + U_self + B_E + B_Wself + B_Wbh - E_remap - E_floor
//! - E_asym - W_frame is constant for a live BH with the reciprocal force.  With
void TDEExternalHistory(HistoryData *pdata, Mesh *pm) {
  bh_dynamics::LedgerHistory(pdata, pm, 0, bh_mass_global, bh_x_global, bh_y_global,
                             bh_z_global, bh_vx_global, bh_vy_global, bh_vz_global);
  const int n = pdata->nhist;
  pdata->nhist = n + 1;
  pdata->label[n] = "W_frame";
  const Real share = 1.0/static_cast<Real>(global_variable::nranks);
  pdata->hdata[n] = share*frame_work_total + frame_work_local;
}

void StoreTDEExternalMetadata(ParameterInput *pin, Mesh *pm) {
  if (pin == nullptr) return;
  if (pm != nullptr && pm->pmb_pack != nullptr && pm->pmb_pack->pgrav != nullptr) {
    pm->pmb_pack->pgrav->StoreBoundaryFluxMetadata(pin);
  }
  tde_external::StoreRuntimeMetadata(pin);
  if (bh_live_global) {
    pin->SetReal("problem", "bh_live_inertial_x", bh_inertial_x_global);
    pin->SetReal("problem", "bh_live_inertial_y", bh_inertial_y_global);
    pin->SetReal("problem", "bh_live_inertial_z", bh_inertial_z_global);
    pin->SetReal("problem", "bh_live_inertial_vx", bh_inertial_vx_global);
    pin->SetReal("problem", "bh_live_inertial_vy", bh_inertial_vy_global);
    pin->SetReal("problem", "bh_live_inertial_vz", bh_inertial_vz_global);
    pin->SetReal("problem", "bh_live_ax", bh_gas_ax_global);
    pin->SetReal("problem", "bh_live_ay", bh_gas_ay_global);
    pin->SetReal("problem", "bh_live_az", bh_gas_az_global);
    if (bh_reciprocal_force_global) {
      pin->SetReal("problem", "bh_pair_gas_impulse_x", bh_total_gas_impulse[0]);
      pin->SetReal("problem", "bh_pair_gas_impulse_y", bh_total_gas_impulse[1]);
      pin->SetReal("problem", "bh_pair_gas_impulse_z", bh_total_gas_impulse[2]);
      pin->SetReal("problem", "bh_pair_impulse_residual", bh_last_impulse_residual);
    }
  }
  if (energy_ledger_global) {
    pin->SetReal("problem", "frame_work_total", frame_work_total + frame_work_local);
  }
}

void RestoreLiveBHMetadata(ParameterInput *pin) {
  auto get = [pin](const char *key, const Real fallback) {
    return pin->DoesParameterExist("problem", key) ? pin->GetReal("problem", key) :
                                                     fallback;
  };
  if (bh_live_global && pin->DoesParameterExist("problem", "bh_live_inertial_x")) {
    bh_inertial_x_global = get("bh_live_inertial_x", bh_inertial_x_global);
    bh_inertial_y_global = get("bh_live_inertial_y", bh_inertial_y_global);
    bh_inertial_z_global = get("bh_live_inertial_z", bh_inertial_z_global);
    bh_inertial_vx_global = get("bh_live_inertial_vx", 0.0);
    bh_inertial_vy_global = get("bh_live_inertial_vy", 0.0);
    bh_inertial_vz_global = get("bh_live_inertial_vz", 0.0);
    bh_gas_ax_global = get("bh_live_ax", 0.0);
    bh_gas_ay_global = get("bh_live_ay", 0.0);
    bh_gas_az_global = get("bh_live_az", 0.0);
  } else if (bh_live_global) {
    // The source was written without a live BH (it followed the two-body orbit), so the
    // BH starts from rest in the inertial frame with no stored gas pull.  This also keeps
    // a remap settle pass from carrying the velocity gathered during the settle steps.
    bh_inertial_vx_global = 0.0;
    bh_inertial_vy_global = 0.0;
    bh_inertial_vz_global = 0.0;
    bh_gas_ax_global = 0.0;
    bh_gas_ay_global = 0.0;
    bh_gas_az_global = 0.0;
  }
  if (bh_reciprocal_force_global) {
    bh_total_gas_impulse[0] = get("bh_pair_gas_impulse_x", 0.0);
    bh_total_gas_impulse[1] = get("bh_pair_gas_impulse_y", 0.0);
    bh_total_gas_impulse[2] = get("bh_pair_gas_impulse_z", 0.0);
    for (Real &value : bh_local_gas_impulse) value = 0.0;
    bh_impulse_window_open = false;
  }
  if (energy_ledger_global) {
    frame_work_total = get("frame_work_total", 0.0);
    frame_work_local = 0.0;
  }
}

bool RestoreFrameBHStateFromMetadata(ParameterInput *pin) {
  if (pin == nullptr) return false;
  if (!pin->DoesParameterExist("problem", "bh_live_state_valid")) return false;
  if (!pin->GetBoolean("problem", "bh_live_state_valid")) return false;

  const char *required_keys[] = {
      "bh_live_x", "bh_live_y", "bh_live_z",
      "bh_live_vx", "bh_live_vy", "bh_live_vz",
      "frame_live_x", "frame_live_y", "frame_live_z",
      "frame_live_vx", "frame_live_vy", "frame_live_vz"};
  for (const char *key : required_keys) {
    if (!pin->DoesParameterExist("problem", key)) return false;
  }

  bh_x_global = pin->GetReal("problem", "bh_live_x");
  bh_y_global = pin->GetReal("problem", "bh_live_y");
  bh_z_global = pin->GetReal("problem", "bh_live_z");
  bh_vx_global = pin->GetReal("problem", "bh_live_vx");
  bh_vy_global = pin->GetReal("problem", "bh_live_vy");
  bh_vz_global = pin->GetReal("problem", "bh_live_vz");
  orbit_center_x_global = pin->GetReal("problem", "frame_live_x");
  orbit_center_y_global = pin->GetReal("problem", "frame_live_y");
  orbit_center_z_global = pin->GetReal("problem", "frame_live_z");
  frame_vx_global = pin->GetReal("problem", "frame_live_vx");
  frame_vy_global = pin->GetReal("problem", "frame_live_vy");
  frame_vz_global = pin->GetReal("problem", "frame_live_vz");
  bh_inertial_x_global = orbit_center_x_global + bh_x_global;
  bh_inertial_y_global = orbit_center_y_global + bh_y_global;
  bh_inertial_z_global = orbit_center_z_global + bh_z_global;
  RestoreOptionalFrameBHStepState(pin);
  SyncProblemRuntimeState();
  return true;
}

bool AutoRemapAfterCycle(Driver *driver, ParameterInput *pin, Mesh *pm) {
  if (driver == nullptr || pin == nullptr || pm == nullptr) return false;
  if (!remap_settle_active_global) return false;
  if (pm->ncycle < remap_settle_target_cycle_global) return false;

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp == nullptr || pmbp->phydro == nullptr) return false;

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
              << "--- TDE Auto Remap Trigger ---" << std::endl
              << "current cycle           = " << pm->ncycle << std::endl
              << "current time            = " << pm->time << std::endl
              << "target cycle            = " << remap_settle_target_cycle_global
              << std::endl
              << "remaining passes before = " << remap_settle_remaining_passes_global
              << std::endl
              << std::endl;
  }

  LoadAndApplyRemapSource(pm, pmbp, pin, true, "--- TDE Auto Remap ---");
  if (remap_settle_remaining_passes_global > 0) {
    remap_settle_remaining_passes_global--;
  }
  ScheduleNextAutoRemapPass(pm);
  driver->InitBoundaryValuesAndPrimitives(pm);
  tde_external::StoreRuntimeMetadata(pin);

  if (global_variable::my_rank == 0) {
    std::cout << "auto-remap remaining passes = "
              << remap_settle_remaining_passes_global << std::endl;
    if (remap_settle_active_global) {
      std::cout << "next auto-remap cycle       = "
                << remap_settle_target_cycle_global << std::endl;
    } else {
      std::cout << "auto-remap passes complete" << std::endl
                << "restored AMR cadence        = ncycle_check="
                << remap_amr_ncycle_check_config_global
                << ", refinement_interval="
                << remap_amr_refinement_interval_config_global << std::endl;
    }
    std::cout << std::endl;
  }
  return true;
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
        profile.dtheta_xi1 =
            CubicHermiteSlope(theta, dtheta, theta_next, dtheta_next, h, troot);
        profile.xi.push_back(profile.xi1);
        profile.theta.push_back(0.0);
        profile.dtheta.push_back(profile.dtheta_xi1);
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
                                 theta_tab(hi), dtheta_tab(hi), x1 - x0, frac);
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
    const Real mass,
                    const Real rho, const Real p, Real &dmdr, Real &drhodr, Real &dpdr) {
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

    const Real r_zero =
        r_prev + frac * (r_term - r_prev);
    const Real mass_zero =
        mass_prev + frac * (mass_term - mass_prev);
    const Real rho_zero =
        rho_prev + frac * (rho_term - rho_prev);
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

    mass = mass_next;
    rho = rho_next;
    p = p_next;
    r = r_next;

    if (profile != nullptr) {
      profile->radius.push_back(std::min(r, r_surface));
      profile->density.push_back(std::max(rho, static_cast<Real>(0.0)));
      profile->pressure.push_back(std::max(p, static_cast<Real>(0.0)));
    }

    if (!(rho > 0.0) || !(p > 0.0)) {
      return finish_surface(r - dr,
          mass - (dr / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m),
                            rho - (dr / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho),
                            p - (dr / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p),
                            r, mass, rho, p);
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

    const Real p_mid = std::exp(static_cast<Real>(0.5)*(log_p_lo + log_p_hi));
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
    const Real star_mass,
                             const Real r_surface, const Real rho_guess, const Real p_guess,
                             const Real guess_gamma, StellarRadialProfile &profile) {
  constexpr int kStructureSteps = 4096;
  constexpr int kMaxBracketIters = 80;
  constexpr int kMaxBisectIters = 96;
  constexpr Real kMassTolFrac = 1.0e-10;

  auto evaluate = [&](const Real trial_rho, Real &trial_p_c, Real &trial_mass,
                      Real &trial_psurf) {
    const Real scaled_p_guess =
        p_guess * std::pow(std::max(trial_rho, static_cast<Real>(1.0e-30)) /
                               std::max(rho_guess, static_cast<Real>(1.0e-30)),
                           guess_gamma);
    return SolveEOSBalancedCentralPressure(eos, grav_const, r_surface, trial_rho,
                                           scaled_p_guess,
                                           trial_p_c, trial_mass, trial_psurf);
  };

  Real rho_active_max = std::numeric_limits<Real>::max();
  if (eos.UsesTabulatedLTE()) {
    rho_active_max =
        std::exp(eos.saha_logrho_max) / std::max(eos.density_unit_cgs,
            static_cast<Real>(1.0e-300));
  }
  Real rho_lo = std::max(rho_guess * static_cast<Real>(0.5), static_cast<Real>(1.0e-12));
  Real rho_hi = std::min(std::max(rho_guess, rho_lo * static_cast<Real>(2.0)),
      rho_active_max);
  Real p_lo = 0.0, mass_lo = 0.0, psurf_lo = 0.0;
  Real p_hi = 0.0, mass_hi = 0.0, psurf_hi = 0.0;

  if (!evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_lo > star_mass; ++n) {
    rho_lo *= static_cast<Real>(0.5);
    if (!(rho_lo > 0.0) || !evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  }

  if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_hi < star_mass; ++n) {
    if (!(rho_hi < rho_active_max)) break;
    rho_hi = std::min(rho_hi * static_cast<Real>(2.0), rho_active_max);
    if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  }

  if (!(mass_lo <= star_mass && mass_hi >= star_mass)) return false;

  Real rho_mid = rho_guess;
  Real p_mid = 0.0;
  Real mass_mid = 0.0;
  Real psurf_mid = 0.0;
  for (int n = 0; n < kMaxBisectIters; ++n) {
    const Real log_rho_lo =
        std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi =
        std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (star_mass - mass_lo) / dmass;
      if (trial_frac > 0.0 && trial_frac < 1.0 && std::isfinite(trial_frac)) {
        frac = trial_frac;
      }
    }
    rho_mid = std::exp(log_rho_lo + frac * (log_rho_hi - log_rho_lo));
    if (!evaluate(rho_mid, p_mid, mass_mid, psurf_mid)) return false;
    if (std::abs(mass_mid - star_mass) <=
        kMassTolFrac * std::max(star_mass, static_cast<Real>(1.0))) {
      break;
    }
    if (mass_mid > star_mass) {
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

  if (std::abs(mass_mid - star_mass) >
      kMassTolFrac * std::max(star_mass, static_cast<Real>(1.0))) {
    const Real log_rho_lo =
        std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi =
        std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (star_mass - mass_lo) / dmass;
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
  if (!RenormalizeEOSBalancedProfile(grav_const, star_mass, r_surface, profile)) {
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

void TDEExternalGravitySource(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;

  PrepareFrameBHStepState(pm);

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool external_bh_gravity_source =
      pmbp->phydro->psrc != nullptr &&
      pmbp->phydro->psrc->external_bh_gravity &&
      problem_runtime::ExternalBHGravitySourceCouplingEnabled();
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmbp->lat_per_block_timestep;
  const Real mesh_dt = pm->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmbp->lat_step_dt.d_view;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;

  Real t_src = pm->time;
  if (pm->pgen != nullptr && pm->pgen->user_srcs_time_valid) {
    t_src = pm->pgen->user_srcs_time;
  } else if (pm->dt > 0.0) {
    const Real stage_fraction = std::fmax(static_cast<Real>(0.0), bdt/pm->dt);
    t_src = pm->time + stage_fraction * pm->dt;
  }
  Real bhvx = 0.0, bhvy = 0.0, bhvz = 0.0;
  Real bhx, bhy, bhz;
  InterpolateBHStepState(t_src, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  Real cx, cy, cz;
  FrameCenterAtTime(t_src, cx, cy, cz);

  Real gm = external_bh_gravity_source ? static_cast<Real>(0.0) :
      newton_g_global * bh_mass_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real eps2 = bh_soft_global * bh_soft_global;

  // Translating-frame correction using the measured selected-gas mean BH
  // acceleration. This removes the bulk selected-debris acceleration from the frame.
  Real ax_frame = 0.0;
  Real ay_frame = 0.0;
  Real az_frame = 0.0;
  FrameAccelerationAtTime(t_src, ax_frame, ay_frame, az_frame);
  bool apply_relax = relax_enable_global && (t_src <= relax_t_end_global);
  const Real relax_tau = relax_tau_global;
  const Real relax_fac = apply_relax ? std::exp(-bdt/relax_tau) : 1.0;
  if (gm == 0.0 && ax_frame == 0.0 && ay_frame == 0.0 && az_frame == 0.0 &&
      !apply_relax) {
    // Core external-BH source active, fixed frame, relaxation over: the kernel below
    // would only add exact zeros to every active cell.
    return;
  }
  const bool ledger_frame_work = energy_ledger_global && eos_has_energy &&
      bdt != 0.0 && pm->pgen != nullptr && pm->pgen->user_srcs_stage_valid;
  const Real frame_energy_before =
      ledger_frame_work ? ActiveSourceGasEnergy(pmbp, nwork1) : 0.0;
  Kokkos::Array<Real, 21> lat_relax_fac;
  for (int level = 0; level <= 20; ++level) {
    lat_relax_fac[level] = 1.0;
  }
  if (apply_relax && lat_per_block_dt) {
    for (int level = 0; level <= 20; ++level) {
      const int factor = 1 << level;
      const Real block_bdt = stage_weight*mesh_dt*static_cast<Real>(factor);
      lat_relax_fac[level] = std::exp(-block_bdt/relax_tau);
    }
  }
  auto lat_step_level = pmbp->lat_step_level.d_view;
  Real relax_r2 = relax_radius_global * relax_radius_global;
  Real rho_floor = bh_grav_rho_min_global;
  const Real rho_weight_floor = pmbp->phydro->peos->eos_data.dfloor;

  par_for("tde_bh_src", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    const Real rx = x - bhx;
    const Real ry = y - bhy;
    const Real rz = z - bhz;

    if (excise_r2 > 0.0 &&
        tde_external::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2)) return;

    Real r2 = rx*rx + ry*ry + rz*rz + eps2;
    r2 = Kokkos::fmax(r2, static_cast<Real>(1.0e-30));
    Real invr = 1.0 / Kokkos::sqrt(r2);
    Real invr3 = invr*invr*invr;

    Real ax = -gm * rx * invr3 - ax_frame;
    Real ay = -gm * ry * invr3 - ay_frame;
    Real az = -gm * rz * invr3 - az_frame;

    Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0) return;

    // The frame correction removes the BH acceleration of the gas, so it is applied
    // with the coupling weight the BH force itself carries (utils/gravity_weight.hpp).
    const Real w_bh = gravity_weight::Weight(rho, rho_weight_floor, rho_floor);
    if (w_bh > 0.0) {
      Real ke_old = 0.0;
      if (eos_has_energy) {
        Real vx_old = u0(m, IM1, k, j, i) / rho;
        Real vy_old = u0(m, IM2, k, j, i) / rho;
        Real vz_old = u0(m, IM3, k, j, i) / rho;
        ke_old = 0.5 * rho * (vx_old*vx_old + vy_old*vy_old + vz_old*vz_old);
      }

      const Real coupled_rho = w_bh * rho;
      u0(m, IM1, k, j, i) += block_bdt * coupled_rho * ax;
      u0(m, IM2, k, j, i) += block_bdt * coupled_rho * ay;
      u0(m, IM3, k, j, i) += block_bdt * coupled_rho * az;

      if (eos_has_energy) {
        // External BH gravity only changes kinetic energy here, so the dual-energy
        // auxiliary thermal field intentionally remains untouched.
        Real vx_new = u0(m, IM1, k, j, i) / rho;
        Real vy_new = u0(m, IM2, k, j, i) / rho;
        Real vz_new = u0(m, IM3, k, j, i) / rho;
        Real ke_new = 0.5 * rho * (vx_new*vx_new + vy_new*vy_new + vz_new*vz_new);
        u0(m, IEN, k, j, i) += (ke_new - ke_old);
      }
    }
    // The relaxation damping keeps its own two-valued density cut.
    if (rho < rho_floor) return;

    if (apply_relax) {
      Real rs2 = SQR(x - cx) + SQR(y - cy) + SQR(z - cz);
      if (rs2 <= relax_r2) {
        const int lat_level = lat_per_block_dt ? lat_step_level(m) : 0;
        const Real block_relax_fac =
            lat_per_block_dt ? lat_relax_fac[lat_level] : relax_fac;
        Real mom1 = u0(m, IM1, k, j, i);
        Real mom2 = u0(m, IM2, k, j, i);
        Real mom3 = u0(m, IM3, k, j, i);
        Real ke_rel_old = 0.0;
        if (eos_has_energy) {
          ke_rel_old = 0.5 * (mom1*mom1 + mom2*mom2 + mom3*mom3) / rho;
        }

        mom1 *= block_relax_fac;
        mom2 *= block_relax_fac;
        mom3 *= block_relax_fac;
        u0(m, IM1, k, j, i) = mom1;
        u0(m, IM2, k, j, i) = mom2;
        u0(m, IM3, k, j, i) = mom3;

        if (eos_has_energy) {
          Real ke_rel_new = 0.5 * (mom1*mom1 + mom2*mom2 + mom3*mom3) / rho;
          u0(m, IEN, k, j, i) += (ke_rel_new - ke_rel_old);
        }
      }
    }
  });
  if (ledger_frame_work) {
    // This stage's energy change enters the completed step with weight
    // FinalFluxWeight*dt/(beta*dt).  Without LAT every rank runs every stage, so the
    // reduction can happen here; with LAT it waits for the window end.
    const Real weight =
        pm->pgen->user_srcs_final_weight*pm->pgen->user_srcs_step_dt/bdt;
    frame_work_local +=
        (ActiveSourceGasEnergy(pmbp, nwork1) - frame_energy_before)*weight;
    if (!hydro_lat_enabled_global) {
      Real work = frame_work_local;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(&frame_work_local, &work, 1, MPI_ATHENA_REAL, MPI_SUM,
                    MPI_COMM_WORLD);
#endif
      frame_work_total += work;
      frame_work_local = 0.0;
    }
  }
}

// MeshBlocks whose cells can reach the excision shell rewritten by
// TDEExternalHydroStateFixup.  Only the block geometry is cached; the LAT active mask
// changes every substep and is re-applied on every call.
struct ExcisionShellBlocks {
  bool valid = false;
  int nmb = -1;
  std::uint64_t topology_version = 0;
  Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
  Real radius = -1.0;
  std::vector<char> candidate;
};
ExcisionShellBlocks excision_shell_blocks;

void TDEExternalHydroStateFixup(MeshBlockPack *pmbp, const Real time) {
  if (pmbp == nullptr || pmbp->pmesh == nullptr || pmbp->phydro == nullptr) return;
  // A stage-0 call outside a step (InitBoundaryValuesAndPrimitives) has no valid
  // pmesh->dt -- on a fresh start it runs before the first Mesh::NewTimeStep, and after a
  // remap it holds the source's last dt -- so it must not advance the translating frame
  // or a live BH; it reads the current frame and BH state instead.
  if ((!use_translating_frame_global && !bh_live_global) ||
      pmbp->pmesh->pgen->user_srcs_time_valid) {
    PrepareFrameBHStepState(pmbp->pmesh);
  }
  if (!bh_inner_boundary_global || bh_excise_radius_global <= 0.0) return;

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const int nscan = lat_enabled ? pmbp->lat_nactive_thispack : nmb;
  if (nscan <= 0) return;

  Real bhvx = 0.0, bhvy = 0.0, bhvz = 0.0;
  Real bhx, bhy, bhz;
  InterpolateBHStepState(time, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;
  const bool dual_enabled = pmbp->phydro->use_dual_energy;
  const int dual_energy_idx = pmbp->phydro->dual_energy_idx;
  const Real bh_excise_radius = bh_excise_radius_global;
  const Real exact_floor_rho_max = hydro_dfloor_global * static_cast<Real>(10.0);
  const Real exact_floor_eint = bh_excise_eint_global;

  // Every cell of a block whose bounding box, grown by the excision radius plus the
  // cell diagonal (the kernel's own reach) plus one cell of slack, cannot reach the BH
  // returns at the distance test below, so skipping such blocks is a no-op.
  auto &cache = excision_shell_blocks;
  if (!cache.valid || cache.nmb != nmb ||
      cache.topology_version != pmbp->pmesh->topology_version ||
      cache.bhx != bhx || cache.bhy != bhy || cache.bhz != bhz ||
      cache.radius != bh_excise_radius) {
    cache.candidate.assign(nmb, 0);
    for (int m = 0; m < nmb; ++m) {
      const Real d1 = size.h_view(m).dx1;
      const Real d2 = size.h_view(m).dx2;
      const Real d3 = size.h_view(m).dx3;
      const Real cell_diag = std::sqrt(d1*d1 + d2*d2 + d3*d3);
      const Real guard = bh_excise_radius + cell_diag + std::max(d1, std::max(d2, d3));
      const Real ox =
          (bhx < size.h_view(m).x1min) ? (size.h_view(m).x1min - bhx) :
          ((bhx > size.h_view(m).x1max) ? (bhx - size.h_view(m).x1max) :
           static_cast<Real>(0.0));
      const Real oy =
          (bhy < size.h_view(m).x2min) ? (size.h_view(m).x2min - bhy) :
          ((bhy > size.h_view(m).x2max) ? (bhy - size.h_view(m).x2max) :
           static_cast<Real>(0.0));
      const Real oz =
          (bhz < size.h_view(m).x3min) ? (size.h_view(m).x3min - bhz) :
          ((bhz > size.h_view(m).x3max) ? (bhz - size.h_view(m).x3max) :
           static_cast<Real>(0.0));
      cache.candidate[m] = (ox*ox + oy*oy + oz*oz <= guard*guard) ? 1 : 0;
    }
    cache.valid = true;
    cache.nmb = nmb;
    cache.topology_version = pmbp->pmesh->topology_version;
    cache.bhx = bhx;
    cache.bhy = bhy;
    cache.bhz = bhz;
    cache.radius = bh_excise_radius;
  }

  // Hydro's scratch list of active blocks around the sink: it is refilled before each
  // use and every consumer runs in stream order behind this upload.
  auto &block_list = pmbp->phydro->sink_block_indices;
  auto &lat_indices = pmbp->lat_active_indices;
  if (lat_enabled) lat_indices.template sync<HostMemSpace>();
  int nshell_blocks = 0;
  for (int a = 0; a < nscan; ++a) {
    const int m = lat_enabled ? lat_indices.h_view(a) : a;
    if (cache.candidate[m]) block_list.h_view(nshell_blocks++) = m;
  }
  if (nshell_blocks == 0) return;

  // Stream-ordered upload: DualView::sync would fence the whole device queue here.
  Kokkos::deep_copy(DevExeSpace(), block_list.d_view, block_list.h_view);
  block_list.clear_sync_state();
  auto shell_blocks = block_list.d_view;

  par_for("tde_bh_floor_shell_fixup", DevExeSpace(), 0, nshell_blocks-1, ks, ke, js, je,
  is, ie, KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = shell_blocks(a);
    const Real rho = u0(m, IDN, k, j, i);
    if (rho >= exact_floor_rho_max) return;

    const Real &x1min = size.d_view(m).x1min;
    const Real &x1max = size.d_view(m).x1max;
    const Real &x2min = size.d_view(m).x2min;
    const Real &x2max = size.d_view(m).x2max;
    const Real &x3min = size.d_view(m).x3min;
    const Real &x3max = size.d_view(m).x3max;

    const Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    const Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    const Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
    const Real dx = (x1max - x1min) / static_cast<Real>(indcs.nx1);
    const Real dy = (x2max - x2min) / static_cast<Real>(indcs.nx2);
    const Real dz = (x3max - x3min) / static_cast<Real>(indcs.nx3);
    const Real shell = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    const Real dx_bh = x - bhx;
    const Real dy_bh = y - bhy;
    const Real dz_bh = z - bhz;
    const Real r = Kokkos::sqrt(dx_bh*dx_bh + dy_bh*dy_bh + dz_bh*dz_bh);
    if (r > bh_excise_radius + shell) return;

    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (eos_has_energy) {
      u0(m, IEN, k, j, i) = exact_floor_eint;
    }
    if (dual_enabled && dual_energy_idx >= 0) {
      u0(m, dual_energy_idx, k, j, i) = exact_floor_eint;
    }
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Lane-Emden star (M*=1) with external Newtonian BH gravity source term.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Follow disk.cpp behavior: always enable and enroll user source terms here.
  user_srcs = true;
  user_srcs_lat_safe = false;
  user_srcs_lat_union_safe = false;
  user_hydro_state_fixup_lat_union_safe = false;
  user_srcs_func = TDEExternalGravitySource;
  user_hydro_state_fixup_func = TDEExternalHydroStateFixup;
  user_hydro_lat_factor_cap_func = CapHydroLATFactorsNearBH;

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (four_pi_G <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "gravity/four_pi_G must be > 0 (set it in <gravity>, e.g. 1.0)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  newton_g_global = four_pi_G / (4.0*M_PI);

  // Star parameters
  constexpr Real star_mass = 1.0;
  Real r_star = pin->GetOrAddReal("problem", "star_radius", 0.5);
  Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-8);
  Real poly_n = pin->GetOrAddReal("problem", "poly_n", 1.5);
  std::string stellar_structure_mode_name =
      pin->GetOrAddString("problem", "stellar_structure_mode", "legacy_polytrope");
  StellarStructureMode stellar_structure_mode = StellarStructureMode::legacy_polytrope;
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);
  Real x_center = pin->GetOrAddReal("problem", "x_center", -1.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  Real vx_star = pin->GetOrAddReal("problem", "vx_star", 0.0);
  Real vy_star = pin->GetOrAddReal("problem", "vy_star", 0.0);
  Real vz_star = pin->GetOrAddReal("problem", "vz_star", 0.0);
  bool relax_damp = pin->GetOrAddBoolean("problem", "relax_damp", false);
  Real relax_tau = pin->GetOrAddReal("problem", "relax_tau", 0.0);
  Real relax_t_end = pin->GetOrAddReal("problem", "relax_t_end", 0.0);
  Real relax_radius = pin->GetOrAddReal("problem", "relax_radius", r_star);
  // Remap is configured through the module-level <remap> block (src/remap/remap.hpp).
  // The TDE settle-pass cadence keys also live there.  Only touch the block when it
  // exists: GetOrAdd would otherwise create it and arm auto-remap unintentionally.
  //
  // Back-compat shim.  Commit ee37f855 moved these keys out of <problem> and left nothing
  // behind, so a deck written against the old spelling (problem/remap = true,
  // problem/remap_restart_source = ...) stopped remapping and quietly started a fresh
  // Lane-Emden star instead -- the run looked healthy and was a different calculation.
  // Refuse that, by name.  A deck that merely carries the retired keys with remap = false
  // means exactly what it always meant (no remap), so it only warns -- and so does a
  // RESTART, whose parameter dump can carry the old keys from the run that wrote it:
  // aborting someone's restart over a dead key in their own checkpoint would be worse
  // than the silence this shim removes, and the state comes from the file anyway.
  if (!restart && pin->DoesParameterExist("problem", "remap") &&
      pin->GetBoolean("problem", "remap")) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/remap = true is a retired key: the remap moved into the "
                 "module-level <remap> block and <problem>/remap* is no longer read, so "
                 "this run would have discarded the source restart and started a fresh "
                 "star.  Migrate the deck to:" << std::endl
              << std::endl
              << "  <remap>" << std::endl
              << "  enable        = true" << std::endl
              << "  source        = <old problem/remap_restart_source>" << std::endl
              << "  settle_steps  = <old problem/remap_settle_steps>   # optional, def 20"
              << std::endl
              << "  settle_passes = <old problem/remap_settle_passes>  # optional, def 1"
              << std::endl << std::endl
              << "problem/remap_interpolation is gone entirely (trilinear was its only "
                 "value)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    const char *retired_keys[] = {"remap", "remap_restart_source", "remap_interpolation",
                                  "remap_settle_steps", "remap_settle_passes"};
    for (const char *key : retired_keys) {
      if (!pin->DoesParameterExist("problem", key)) continue;
      std::cout << "WARNING (tde_external): <problem>/" << key << " is a retired key "
                << "and is ignored; the module-level <remap> block replaced it.  Delete "
                << "it from the input." << std::endl;
    }
  }
  bool remap_enable = remap::IsAutoRemapEnabled(pin);
  std::string remap_restart_source;
  int remap_settle_steps = 20;
  int remap_settle_passes = 1;
  if (pin->DoesBlockExist("remap")) {
    remap_restart_source = pin->GetOrAddString("remap", "source", "");
    remap_settle_steps = pin->GetOrAddInteger("remap", "settle_steps", 20);
    remap_settle_passes = pin->GetOrAddInteger("remap", "settle_passes", 1);
  }
  int configured_amr_ncycle_check =
      pin->GetOrAddInteger("mesh_refinement", "ncycle_check", 1);
  int configured_amr_refinement_interval =
      pin->GetOrAddInteger("mesh_refinement", "refinement_interval", 5);
  use_translating_frame_global =
      pin->GetOrAddBoolean("problem", "use_translating_frame", false);
  hydro_lat_enabled_global = pin->IsLATEnabled();
  user_srcs_lat_safe = true;
  user_srcs_lat_union_safe = true;
  user_hydro_state_fixup_lat_union_safe = true;
  {
    // Read without GetOrAdd so a deck that does not use these keys dumps an unchanged
    // parameter set.
    bh_live_global = pin->DoesParameterExist("problem", "bh_live") &&
        pin->GetBoolean("problem", "bh_live");
    bh_reciprocal_force_global =
        pin->DoesParameterExist("problem", "bh_reciprocal_force") &&
        pin->GetBoolean("problem", "bh_reciprocal_force");
    const std::string pair_force =
        pin->DoesParameterExist("problem", "bh_pair_force") ?
        pin->GetString("problem", "bh_pair_force") : std::string("finite_difference");
    if (pair_force != "finite_difference" && pair_force != "analytic") {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_pair_force must be finite_difference or analytic."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    bh_analytic_pair_global = pair_force == "analytic";
    MeshBlockPack *pack = pmy_mesh_->pmb_pack;
    const bool external_source = pack->phydro != nullptr &&
        pack->phydro->psrc != nullptr && pack->phydro->psrc->external_bh_gravity;
    if (bh_analytic_pair_global && !bh_reciprocal_force_global) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_pair_force = analytic requires bh_reciprocal_force = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (bh_reciprocal_force_global &&
        (!bh_live_global || !hydro_lat_enabled_global || pack->pgrav == nullptr ||
         !external_source)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_reciprocal_force requires a live BH "
                << "(bh_live = true), time/lat, self gravity, and "
                << "external_bh_gravity_source = true." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (bh_live_global && !external_source) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "a live BH needs external_bh_gravity_source = true so the gas force "
                << "and the BH force use the same potential stencil." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    analytic_external_bh_momentum = bh_analytic_pair_global;
    energy_ledger_global = pack->pgrav != nullptr && pack->pgrav->lat_time_centered_work;
    if (hydro_lat_enabled_global && (use_translating_frame_global || bh_live_global)) {
      user_hydro_lat_window_func = PrepareTranslatingFrameLATWindow;
    }
    if (hydro_lat_enabled_global &&
        (bh_reciprocal_force_global || energy_ledger_global)) {
      user_hydro_lat_window_end_func = CompleteTDELATWindow;
    }
    if (bh_reciprocal_force_global) {
      user_hydro_gravity_ledger_func = AccumulateTDEBHGasImpulse;
    }
    if (energy_ledger_global) {
      user_hist = true;
      user_hist_func = TDEExternalHistory;
    }
    if (energy_ledger_global || bh_live_global) {
      user_metadata_func = StoreTDEExternalMetadata;
    }
  }
  bool has_loaded_restart_state = restart || pmy_mesh_->ncycle > 0
      || pmy_mesh_->time != 0.0;
  remap_enable_global = remap_enable;
  if (remap_enable) {
    after_cycle_func = AutoRemapAfterCycle;
    user_output_gate_func = OutputsAllowedOnCurrentCycle;
  }
  remap_restart_source_global = remap_restart_source;
  remap_settle_steps_global = remap_settle_steps;
  remap_settle_config_passes_global = remap_settle_passes;
  remap_amr_ncycle_check_config_global =
      std::max(configured_amr_ncycle_check, 1);
  remap_amr_refinement_interval_config_global =
      std::max(configured_amr_refinement_interval,
               remap_amr_ncycle_check_config_global);
  DisableAutoRemapPasses(pmy_mesh_);

  if (poly_n <= 0.0 || poly_n >= 5.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/poly_n must satisfy 0 < n < 5." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (r_star <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/star_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rho_floor < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/rho_floor must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!ParseStellarStructureMode(stellar_structure_mode_name, stellar_structure_mode)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/stellar_structure_mode must be one of "
              << "'legacy_polytrope' or 'eos_balanced'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (relax_damp) {
    if (relax_tau <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_tau must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_t_end <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_t_end must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_radius <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_radius must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (remap_enable) {
    // The <remap> + time/lat refusal moved into the remap module itself
    // (remap.cpp), so it now fires for every pgen instead of this one.
    if (remap_restart_source.empty()) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "remap/source must be set when the <remap> block is enabled."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (remap_settle_steps < 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "remap/settle_steps must be >= 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (remap_settle_passes < 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "remap/settle_passes must be >= 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (!use_translating_frame_global && !has_loaded_restart_state && !remap_enable) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/use_translating_frame = false is only supported for "
              << "restart/remap-based TDE conversion." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  relax_enable_global = relax_damp;
  relax_tau_global = relax_tau;
  relax_t_end_global = relax_t_end;
  relax_radius_global = relax_radius;

  // External BH setup: Phantom-style orbit controls, but in star-centered coordinates.
  bh_soft_global = pin->GetOrAddReal("problem", "bh_softening", 1.0e-2);
  if (bh_soft_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_softening must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_grav_rho_min_global = pin->GetOrAddReal("problem", "bh_grav_rho_min", 0.0);
  if (bh_grav_rho_min_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_grav_rho_min must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  frame_rho_min_global = pin->GetOrAddReal("problem", "frame_rho_min", 0.0);
  if (frame_rho_min_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/frame_rho_min must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_excise_radius_global = pin->GetOrAddReal("problem", "bh_excise_radius", 0.5);
  bh_inner_boundary_global = pin->GetOrAddBoolean("problem", "bh_inner_boundary", true);
  if (bh_inner_boundary_global && bh_excise_radius_global <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_excise_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_force_refine_global = pin->GetOrAddBoolean("problem", "bh_max_amr", false);
  bh_force_refine_level_offset_global =
      pin->GetOrAddInteger("problem", "bh_max_amr_level_offset", 0);
  if (bh_force_refine_level_offset_global < 0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_max_amr_level_offset must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  unbound_refine_global = pin->GetOrAddBoolean("problem", "unbound_amr", false);
  unbound_refine_level_offset_global =
      pin->GetOrAddInteger("problem", "unbound_amr_level_offset", 3);
  if (unbound_refine_level_offset_global < 0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/unbound_amr_level_offset must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  unbound_refine_rho_min_global =
      pin->GetOrAddReal("problem", "unbound_amr_rho_min", -1.0);
  if (unbound_refine_rho_min_global < 0.0 &&
      unbound_refine_rho_min_global != static_cast<Real>(-1.0)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/unbound_amr_rho_min must be >= 0 or = -1 for auto."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  unbound_refine_fill_frac_global =
      pin->GetOrAddReal("problem", "unbound_amr_fill_frac", 0.01);
  if (!(unbound_refine_fill_frac_global >= 0.0 &&
        unbound_refine_fill_frac_global <= 1.0)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/unbound_amr_fill_frac must be in [0, 1]." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  stream_shell_enable_global =
      pin->DoesParameterExist("problem", "stream_shell_dr") ||
      pin->DoesParameterExist("problem", "stream_shell_level_offset") ||
      pin->DoesParameterExist("problem", "stream_shell_r_max") ||
      pin->DoesParameterExist("problem", "stream_shell_xsplit_radius") ||
      pin->DoesParameterExist("problem", "stream_shell_rho_frac") ||
      pin->DoesParameterExist("problem", "stream_shell_fill_frac");
  for (int n = 1; n <= kMaxStreamLevelTiers && !stream_shell_enable_global; ++n) {
    if (pin->DoesParameterExist("problem",
        "stream_shell_level_radius_" + std::to_string(n)) ||
        pin->DoesParameterExist("problem",
            "stream_shell_level_offset_" + std::to_string(n)) ||
        pin->DoesParameterExist("problem", "stream_shell_dr_" + std::to_string(n)) ||
        pin->DoesParameterExist("problem",
            "stream_shell_rho_frac_" + std::to_string(n)) ||
        pin->DoesParameterExist("problem",
            "stream_shell_fill_frac_" + std::to_string(n))) {
      stream_shell_enable_global = true;
    }
  }
  if (stream_shell_enable_global) {
    stream_shell_dr_global = pin->GetOrAddReal("problem", "stream_shell_dr", -1.0);
    if (stream_shell_dr_global == 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_dr must be < 0 for auto or > 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_level_offset_global =
        pin->GetOrAddInteger("problem", "stream_shell_level_offset", 0);
    if (stream_shell_level_offset_global < 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_level_offset must be >= 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_r_max_global = pin->GetOrAddReal("problem", "stream_shell_r_max", 0.0);
    if (stream_shell_r_max_global < 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_r_max must be >= 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_xsplit_radius_global =
        pin->GetOrAddReal("problem", "stream_shell_xsplit_radius", 0.0);
    if (stream_shell_xsplit_radius_global < 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_xsplit_radius must be >= 0." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_rho_frac_global =
        pin->GetOrAddReal("problem", "stream_shell_rho_frac", 0.5);
    if (stream_shell_rho_frac_global <= 0.0 || stream_shell_rho_frac_global > 1.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_rho_frac must satisfy 0 < value <= 1."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_fill_frac_global =
        pin->GetOrAddReal("problem", "stream_shell_fill_frac", 0.05);
    if (stream_shell_fill_frac_global <= 0.0 || stream_shell_fill_frac_global > 1.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_fill_frac must satisfy 0 < value <= 1."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_derefine_dfloor_mult_global =
        pin->GetOrAddReal("problem", "stream_shell_derefine_dfloor_mult", 100.0);
    if (stream_shell_derefine_dfloor_mult_global < 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "problem/stream_shell_derefine_dfloor_mult must be >= 0."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    stream_shell_level_radii_global.clear();
    stream_shell_level_offsets_global.clear();
    stream_shell_level_drs_global.clear();
    stream_shell_level_rho_fracs_global.clear();
    stream_shell_level_fill_fracs_global.clear();
    Real prev_radius = 0.0;
    for (int n = 1; n <= kMaxStreamLevelTiers; ++n) {
      std::string radius_name = "stream_shell_level_radius_" + std::to_string(n);
      std::string offset_name = "stream_shell_level_offset_" + std::to_string(n);
      std::string dr_name = "stream_shell_dr_" + std::to_string(n);
      std::string rho_frac_name = "stream_shell_rho_frac_" + std::to_string(n);
      std::string fill_frac_name = "stream_shell_fill_frac_" + std::to_string(n);
      bool has_radius = pin->DoesParameterExist("problem", radius_name);
      bool has_offset = pin->DoesParameterExist("problem", offset_name);
      bool has_dr = pin->DoesParameterExist("problem", dr_name);
      bool has_rho_frac = pin->DoesParameterExist("problem", rho_frac_name);
      bool has_fill_frac = pin->DoesParameterExist("problem", fill_frac_name);
      bool has_tier = has_radius || has_offset || has_dr || has_rho_frac || has_fill_frac;
      if (!has_tier) continue;
      if (!(has_radius && has_offset)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << radius_name << " and problem/" << offset_name
                  << " must both be set for each stream-shell tier." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real radius = pin->GetReal("problem", radius_name);
      int offset = pin->GetInteger("problem", offset_name);
      Real dr = pin->GetOrAddReal("problem", dr_name, -1.0);
      Real rho_frac = pin->GetOrAddReal("problem", rho_frac_name,
          stream_shell_rho_frac_global);
      Real fill_frac =
          pin->GetOrAddReal("problem", fill_frac_name, stream_shell_fill_frac_global);
      if (radius <= prev_radius) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << radius_name << " must be > previous tier radius."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (offset < 0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << offset_name << " must be >= 0." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (dr == 0.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << dr_name << " must be < 0 for auto or > 0."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (rho_frac <= 0.0 || rho_frac > 1.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << rho_frac_name << " must satisfy 0 < value <= 1."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (fill_frac <= 0.0 || fill_frac > 1.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/" << fill_frac_name << " must satisfy 0 < value <= 1."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      stream_shell_level_radii_global.push_back(radius);
      stream_shell_level_offsets_global.push_back(offset);
      stream_shell_level_drs_global.push_back(dr);
      stream_shell_level_rho_fracs_global.push_back(rho_frac);
      stream_shell_level_fill_fracs_global.push_back(fill_frac);
      prev_radius = radius;
    }
  } else {
    stream_shell_dr_global = -1.0;
    stream_shell_level_offset_global = 0;
    stream_shell_r_max_global = 0.0;
    stream_shell_xsplit_radius_global = 0.0;
    stream_shell_rho_frac_global = 0.5;
    stream_shell_fill_frac_global = 0.05;
    stream_shell_derefine_dfloor_mult_global = 100.0;
    stream_shell_level_radii_global.clear();
    stream_shell_level_offsets_global.clear();
    stream_shell_level_drs_global.clear();
    stream_shell_level_rho_fracs_global.clear();
    stream_shell_level_fill_fracs_global.clear();
  }


  bool provide_params = pin->GetOrAddBoolean("problem", "provide_params", false);
  Real mass_ratio = pin->GetOrAddReal("problem", "mass_ratio", 1000.0);
  Real beta = pin->GetOrAddReal("problem", "beta", 1.0);
  Real ecc_bh = pin->GetOrAddReal("problem", "ecc_bh", 1.0);
  Real theta_bh_deg = pin->GetOrAddReal("problem", "theta_bh", 0.0);
  Real sep_initial = pin->GetOrAddReal("problem", "sep_initial", 10.0);

  Real x1 = pin->GetOrAddReal("problem", "x1", 0.0);
  Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  Real vx1 = pin->GetOrAddReal("problem", "vx1", 0.0);
  Real vy1 = pin->GetOrAddReal("problem", "vy1", 0.0);
  Real vz1 = pin->GetOrAddReal("problem", "vz1", 0.0);

  if (mass_ratio <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/mass_ratio must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (beta <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/beta must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ecc_bh <= 0.0 || ecc_bh > 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/ecc_bh must satisfy 0 < ecc_bh <= 1." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (sep_initial <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/sep_initial must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bh_force_refine_global || unbound_refine_global || stream_shell_enable_global) {
    user_ref_func = RefineBHPosition;
  }
  bh_mass_global = mass_ratio * star_mass;
  orbit_center_x_global = x_center;
  orbit_center_y_global = y_center;
  orbit_center_z_global = z_center;
  frame_vx_global = 0.0;
  frame_vy_global = 0.0;
  frame_vz_global = 0.0;

  Real r_tidal = r_star * std::cbrt(mass_ratio);
  Real r_peri = r_tidal / beta;
  Real mu_orbit = newton_g_global * (star_mass + bh_mass_global);

  Real star_x = 0.0, star_y = 0.0, star_z = 0.0;
  Real star_vx = 0.0, star_vy = 0.0, star_vz = 0.0;

  if (provide_params) {
    // Same parameter names as Phantom setup: user gives the star state relative to BH.
    // Star-centered frame here uses the opposite sign for the BH state.
    star_x = x1;
    star_y = y1;
    star_z = z1;
    star_vx = vx1;
    star_vy = vy1;
    star_vz = vz1;
  } else {
    Real theta_bh = DegToRad(theta_bh_deg);
    if (ecc_bh < (1.0 - 1.0e-12)) {
      Real semia = r_peri/(1.0 - ecc_bh);
      Real r_apo = semia*(1.0 + ecc_bh);
      Real v_apo = std::sqrt(mu_orbit*(2.0/r_apo - 1.0/semia));

      // Start at apoapsis with pericenter along +y, then apply inclination.
      star_x = 0.0;
      star_y = -r_apo;
      star_z = 0.0;
      star_vx = v_apo;
      star_vy = 0.0;
      star_vz = 0.0;
      RotateAboutY(star_x, star_y, star_z, theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, theta_bh);
    } else {
      // Parabolic branch follows the same construction used in Phantom's TDE setup.
      Real r0 = sep_initial * r_tidal;
      Real y0 = -2.0*r_peri + r0;
      Real x2 = r0*r0 - y0*y0;
      if (x2 < -1.0e-12*std::fmax(1.0, r0*r0)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "problem/sep_initial is too small for the chosen beta." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real x0 = std::sqrt(std::fmax(x2, 0.0));
      Real denom = std::sqrt(4.0*r_peri*r_peri + x0*x0);
      if (denom <= 0.0 || r0 <= 0.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "Invalid parabolic initial condition geometry." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real vel = std::sqrt(2.0*mu_orbit/r0);

      star_x = -x0;
      star_y = y0;
      star_z = 0.0;
      star_vx = vel*(2.0*r_peri/denom);
      star_vy = vel*(-x0/denom);
      star_vz = 0.0;
      RotateAboutY(star_x, star_y, star_z, theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, theta_bh);
    }
  }

  Real bh_init_x = -star_x;
  Real bh_init_y = -star_y;
  Real bh_init_z = -star_z;
  Real bh_init_vx = -star_vx;
  Real bh_init_vy = -star_vy;
  Real bh_init_vz = -star_vz;

  if (!std::isfinite(bh_init_x) || !std::isfinite(bh_init_y) ||
      !std::isfinite(bh_init_z) || !std::isfinite(bh_init_vx) ||
      !std::isfinite(bh_init_vy) || !std::isfinite(bh_init_vz)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Non-finite BH orbit initial state." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // The BH is fixed in the inertial frame. The simulation coordinates use a
  // translating, non-rotating frame, so the BH moves only through frame motion.
  bh_inertial_x_global = orbit_center_x_global + bh_init_x;
  bh_inertial_y_global = orbit_center_y_global + bh_init_y;
  bh_inertial_z_global = orbit_center_z_global + bh_init_z;
  bh_x_global = bh_init_x;
  bh_y_global = bh_init_y;
  bh_z_global = bh_init_z;
  frame_vx_global = star_vx;
  frame_vy_global = star_vy;
  frame_vz_global = star_vz;
  bh_vx_global = -frame_vx_global;
  bh_vy_global = -frame_vy_global;
  bh_vz_global = -frame_vz_global;
  InvalidateFrameBHStepState();

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires a <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd != nullptr || pmbp->prad != nullptr || pmbp->pturb != nullptr ||
      pmbp->pz4c != nullptr || pmbp->padm != nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "tde_external remap currently supports hydro-only runs."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto eos = pmbp->phydro->peos->eos_data;
  hydro_dfloor_global = eos.dfloor;
  stream_shell_rho_floor_global = std::max(rho_floor, eos.dfloor);
  if (unbound_refine_rho_min_global < 0.0) {
    unbound_refine_rho_min_global =
        std::max(hydro_dfloor_global,
                 std::max(frame_rho_min_global, bh_grav_rho_min_global));
  }
  if (!(eos.use_e)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires a hydro EOS with an energy variable."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool self_gravity_enabled =
      (pmbp->phydro->psrc != nullptr && pmbp->phydro->psrc->self_gravity);
  Real gamma_eos = eos.gamma;
  if (eos.is_gamma_law && gamma_eos <= 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "hydro/gamma must be > 1 for ideal EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_excise_density_global = eos.dfloor;
  // Host-side startup uses the host Saha table mirror for consistent EOS floors.
  bh_excise_eint_global = eos.HostClampHydroInternalEnergyDensity(
      bh_excise_density_global,
      eos.HostHydroInternalEnergyDensityFloor(bh_excise_density_global));
  frame_state_time_global = 0.0;
  InvalidateFrameBHStepState();
  EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                orbit_center_z_global,
                                bh_inertial_x_global, bh_inertial_y_global,
                                bh_inertial_z_global,
                                frame_ax_global, frame_ay_global, frame_az_global);
  SyncProblemRuntimeState();

  if (remap_enable && !has_loaded_restart_state) {
    // The remap itself runs from remap::MaybeAutoRemap right after this function
    // returns (fresh-start ProblemGenerator constructor).  Enroll the TDE hooks and
    // skip the fresh-star initialization below.
    user_remap_skip_func = [](Real x, Real y, Real z) {
      const Real excise_r2 = bh_inner_boundary_global ?
          bh_excise_radius_global * bh_excise_radius_global : -1.0;
      return tde_external::InsideExcisionZone(x, y, z, bh_x_global, bh_y_global,
                                              bh_z_global, excise_r2);
    };
    user_remap_loaded_func = [](ParameterInput *src_pin) {
      // Restore the BH/frame record BEFORE the state is applied: the excision skip
      // hook above reads the restored BH position during the apply.
      if (!RestoreFrameBHStateFromMetadata(src_pin)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "remap source restart is missing the BH/frame metadata."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // A live BH also keeps its inertial velocity and gas acceleration across the
      // remap.  Without them the leapfrog restarts from rest after every remap and
      // SwitchToFrozenBHInertialFrame places the BH at rest in the new frame.
      RestoreLiveBHMetadata(src_pin);
    };
    Mesh *pm_cap = pmy_mesh_;
    MeshBlockPack *pmbp_cap = pmbp;
    ParameterInput *pin_cap = pin;
    user_remap_post_func = [pm_cap, pmbp_cap, pin_cap](
        const remap::RemapSummary &) {
      if (use_translating_frame_global) {
        frame_state_time_global = pm_cap->time;
        // The source state moved the frame center and the BH, so the stored frame
        // acceleration is re-evaluated at the new state, as on a restart.
        EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                      orbit_center_z_global,
                                      bh_inertial_x_global, bh_inertial_y_global,
                                      bh_inertial_z_global,
                                      frame_ax_global, frame_ay_global, frame_az_global);
        InvalidateFrameBHStepState();
        SyncProblemRuntimeState();
      } else {
        SwitchToFrozenBHInertialFrame(pm_cap, pmbp_cap);
      }
      remap_settle_remaining_passes_global = remap_settle_config_passes_global;
      ScheduleNextAutoRemapPass(pm_cap);
      tde_external::StoreRuntimeMetadata(pin_cap);

      if (global_variable::my_rank == 0) {
        std::cout << "use translating frame  = "
                  << (use_translating_frame_global ? "true" : "false") << std::endl;
        if (remap_settle_active_global) {
          std::cout << "auto-remap settle steps = " << remap_settle_steps_global
                    << std::endl
                    << "auto-remap passes       = " << remap_settle_remaining_passes_global
                    << std::endl
                    << "first auto-remap cycle  = " << remap_settle_target_cycle_global
                    << std::endl
                    << "forced AMR cadence      = ncycle_check=1, refinement_interval=1"
                    << std::endl
                    << "restored AMR cadence    = ncycle_check="
                    << remap_amr_ncycle_check_config_global
                    << ", refinement_interval="
                    << remap_amr_refinement_interval_config_global << std::endl
                    << "outputs suppressed      = true until settle/remap completes"
                    << std::endl;
        }
        std::cout << std::endl;
      }
    };
    return;
  }

  if (has_loaded_restart_state) {
    if (!RestoreFrameBHStateFromMetadata(pin)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "tde_external restart requires stored BH/frame metadata in the "
                << "restart file." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    RestoreLiveBHMetadata(pin);
    if (use_translating_frame_global) {
      frame_state_time_global = pmy_mesh_->time;
      EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                    orbit_center_z_global,
                                    bh_inertial_x_global, bh_inertial_y_global,
                                    bh_inertial_z_global,
                                    frame_ax_global, frame_ay_global, frame_az_global);
      SyncProblemRuntimeState();
    } else {
      // Boosting into the frozen BH frame rewrites the interior cells, so the
      // checkpointed ghost zones no longer describe the state and must be rebuilt.
      if (frame_vx_global != 0.0 || frame_vy_global != 0.0 || frame_vz_global != 0.0) {
        restart_state_verbatim = false;
      }
      SwitchToFrozenBHInertialFrame(pmy_mesh_, pmbp);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "Restored BH/frame state from restart metadata: "
                << "BH=(" << bh_x_global << ", " << bh_y_global << ", " << bh_z_global
                << "), frame=(" << orbit_center_x_global << ", "
                << orbit_center_y_global << ", " << orbit_center_z_global << ")"
                << std::endl;
    }
    return;
  }

  LaneEmdenProfile lane_profile;
  if (!SolveLaneEmden(poly_n, lane_profile)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Failed to solve Lane-Emden equation for n=" << poly_n << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real xi1 = lane_profile.xi1;
  Real qn = -SQR(xi1) * lane_profile.dtheta_xi1;
  if (qn <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Lane-Emden mass factor is non-positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real a_scale = r_star / xi1;
  Real rho_central = star_mass / (4.0*M_PI*std::pow(a_scale, 3.0)*qn);
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

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool eos_has_energy = eos.use_e;
  bool eos_is_gamma_law = eos.is_gamma_law;
  int nhyd = pmbp->phydro->nhydro;
  int nscalars = pmbp->phydro->nscalars;
  Real xi_scale = xi1 / r_star;
  Real excise_bhx = bh_x_global;
  Real excise_bhy = bh_y_global;
  Real excise_bhz = bh_z_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real excise_density = bh_excise_density_global;
  Real excise_eint = bh_excise_eint_global;
  const Real floor_eint =
      eos.HostClampHydroInternalEnergyDensity(rho_floor,
                                              eos.HostHydroInternalEnergyDensityFloor(rho_floor));
  const Real p_floor = eos.HostPressureFromRhoEint(rho_floor, floor_eint);
  int nmb = pmbp->nmb_thispack;
  const bool use_eos_balanced_profile =
      (stellar_structure_mode == StellarStructureMode::eos_balanced);

  if (use_eos_balanced_profile) {
    if (!eos_has_energy) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/stellar_structure_mode = eos_balanced requires an EOS "
                << "with an energy variable." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (amp != 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/amp must be 0 when problem/stellar_structure_mode = "
                << "eos_balanced. The legacy density perturbation is only defined for "
                << "the pressure-mapped polytropic initializer." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  StellarRadialProfile eos_profile;
  int radial_profile_npts = 1;
  DvceArray1D<Real> radial_r_d("stellar_profile_r", 1);
  DvceArray1D<Real> radial_rho_d("stellar_profile_rho", 1);
  DvceArray1D<Real> radial_p_d("stellar_profile_p", 1);
  {
    auto radial_r_h = Kokkos::create_mirror_view(radial_r_d);
    auto radial_rho_h = Kokkos::create_mirror_view(radial_rho_d);
    auto radial_p_h = Kokkos::create_mirror_view(radial_p_d);
    radial_r_h(0) = 0.0;
    radial_rho_h(0) = 0.0;
    radial_p_h(0) = 0.0;
    Kokkos::deep_copy(radial_r_d, radial_r_h);
    Kokkos::deep_copy(radial_rho_d, radial_rho_h);
    Kokkos::deep_copy(radial_p_d, radial_p_h);
  }

  if (use_eos_balanced_profile) {
    const Real p_central_guess = poly_k * std::pow(rho_central, gamma_poly);
    EOS_Data profile_eos = eos;
    if (!SolveEOSBalancedProfile(profile_eos, newton_g_global, star_mass, r_star,
                                 rho_central, p_central_guess, gamma_poly,
                                 eos_profile)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "Failed to construct an EOS-balanced stellar profile for "
                << "problem/stellar_structure_mode = eos_balanced." << std::endl;
      if (profile_eos.UsesTabulatedLTE()) {
        std::cout << "The requested stellar setup is not consistent with the loaded "
                  << "tabulated LTE EOS over its active rho-T domain. "
                  << "Do not rely on host-side clamping here; use a table that "
                  << "covers the intended stellar regime or revert to a lower-density "
                  << "validation scaling." << std::endl;
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

  if (!use_eos_balanced_profile && eos_is_gamma_law && global_variable::my_rank == 0) {
    if (std::abs(gamma_eos - gamma_poly) > 1.0e-8) {
      std::cout << "### WARNING in ProblemGenerator::UserProblem" << std::endl
                << "For consistent polytropic initialization, set hydro/gamma = "
                << gamma_poly << " (1 + 1/n)." << std::endl;
    }
  }

  par_for("tde_external_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
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

    if (excise_r2 > 0.0 && tde_external::InsideExcisionZone(x, y, z, excise_bhx,
                                                           excise_bhy, excise_bhz,
                                                           excise_r2)) {
      u0(m, IDN, k, j, i) = excise_density;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      if (eos_has_energy) {
        u0(m, IEN, k, j, i) = excise_eint;
      }
      for (int n = nhyd; n < (nhyd + nscalars); ++n) {
        u0(m, n, k, j, i) = 0.0;
      }
      return;
    }

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
                     * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
      }
      p_star = poly_k * Kokkos::pow(Kokkos::fmax(rho_star, 0.0), gamma_poly);
    }

    Real rho = rho_floor + rho_star;
    u0(m, IDN, k, j, i) = rho;
    // Hydro state is initialized in the comoving frame, so the star starts at
    // rest apart from any explicitly requested frame-relative bulk velocity.
    u0(m, IM1, k, j, i) = rho * vx_star;
    u0(m, IM2, k, j, i) = rho * vy_star;
    u0(m, IM3, k, j, i) = rho * vz_star;

    if (eos_has_energy) {
      // The star is stored as a star-over-floor decomposition. In legacy mode the
      // pressure comes from the Lane-Emden polytrope and is only mapped through the
      // active EOS. In eos_balanced mode p_star itself comes from the active-EOS
      // hydrostatic structure solve.
      Real pgas = p_floor + p_star;
      Real ekin = 0.5 * rho * (vx_star*vx_star + vy_star*vy_star + vz_star*vz_star);
      Real eint = eos.InternalEnergyDensityFromRhoP(rho, pgas);
      eint = eos.ClampHydroInternalEnergyDensity(rho, eint);
      u0(m, IEN, k, j, i) = eint + ekin;
    }
    for (int n = nhyd; n < (nhyd + nscalars); ++n) {
      u0(m, n, k, j, i) = 0.0;
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- TDE External Potential ---" << std::endl
      << "star_mass (fixed)      = " << star_mass << std::endl
      << "star_radius            = " << r_star << std::endl
      << "stellar_structure_mode = "
      << StellarStructureModeName(stellar_structure_mode) << std::endl
      << "rho_central (derived)  = " << rho_central << std::endl
      << "rho_floor              = " << rho_floor << std::endl
      << "poly_n                 = " << poly_n << std::endl
      << "poly_gamma             = " << gamma_poly << std::endl
      << "relax_damp             = " << (relax_enable_global ? "true" : "false")
      << std::endl
      << "relax_tau              = " << relax_tau_global << std::endl
      << "relax_t_end            = " << relax_t_end_global << std::endl
      << "relax_radius           = " << relax_radius_global << std::endl
      << "BH mass                = " << bh_mass_global << std::endl
      << "BH position            = (" << bh_x_global << ", "
      << bh_y_global << ", " << bh_z_global << ")" << std::endl
      << "BH velocity            = (" << bh_vx_global << ", "
      << bh_vy_global << ", " << bh_vz_global << ")" << std::endl
      << "Frame center           = (" << orbit_center_x_global << ", "
      << orbit_center_y_global << ", " << orbit_center_z_global << ")" << std::endl
      << "Frame velocity         = (" << frame_vx_global << ", "
      << frame_vy_global << ", " << frame_vz_global << ")" << std::endl
      << "Frame acceleration     = (" << frame_ax_global << ", "
      << frame_ay_global << ", " << frame_az_global << ")" << std::endl
      << "Use translating frame  = "
      << (use_translating_frame_global ? "true" : "false") << std::endl
      << "Self gravity enabled   = " << (self_gravity_enabled ? "true" : "false")
      << std::endl
      << "Frame rho min          = " << frame_rho_min_global << std::endl
      << "BH softening           = " << bh_soft_global << std::endl
      << "BH grav rho min        = " << bh_grav_rho_min_global << std::endl
      << "BH inner boundary      = " << (bh_inner_boundary_global ? "true" : "false")
      << std::endl
      << "BH excise radius       = " << bh_excise_radius_global << std::endl
      << "BH force refine        = " << (bh_force_refine_global ? "true" : "false")
      << std::endl
      << "BH refine target level = "
      << std::max(pmy_mesh_->root_level,
                  pmy_mesh_->max_level - bh_force_refine_level_offset_global)
      << " (max_level - " << bh_force_refine_level_offset_global << ")"
      << std::endl
      << "Unbound AMR            = " << (unbound_refine_global ? "true" : "false")
      << std::endl
      << "Unbound refine level   = "
      << std::max(pmy_mesh_->root_level,
                  pmy_mesh_->max_level - unbound_refine_level_offset_global)
      << " (max_level - " << unbound_refine_level_offset_global << ")"
      << std::endl
      << "Unbound rho min        = " << unbound_refine_rho_min_global
      << std::endl
      << "Unbound fill frac      = " << unbound_refine_fill_frac_global
      << std::endl
      << "Stream shell AMR       = " << (stream_shell_enable_global ? "true" : "false")
      << std::endl
      << "Stream shell base lvl  = "
      << std::max(pmy_mesh_->root_level,
                  pmy_mesh_->max_level - stream_shell_level_offset_global)
      << " (max_level - " << stream_shell_level_offset_global << ")"
      << std::endl
      << "Stream shell base dr   = " << stream_shell_dr_global << std::endl
      << "Stream shell r_max     = " << stream_shell_r_max_global << std::endl
      << "Stream x-split r       = " << stream_shell_xsplit_radius_global
      << " (inside: separate left/right shell peaks relative to the BH)"
      << std::endl
      << "Stream shell rho frac  = " << stream_shell_rho_frac_global
      << std::endl
      << "Stream shell fill frac = " << stream_shell_fill_frac_global
      << " (minimum block coverage away from the shell peak;"
      << " inner split region refines on direct hits)"
      << std::endl
      << "Stream derefine floor  = "
      << stream_shell_derefine_dfloor_mult_global << " * dfloor"
      << std::endl
      << "BH frame note          = "
      << (use_translating_frame_global
              ? "fixed inertial BH, continuous translating frame"
              : "fixed BH in the simulation frame after restart/remap conversion")
      << std::endl
      << "Newtonian G            = " << newton_g_global << std::endl
      << "four_pi_G              = " << four_pi_G << std::endl
      << std::endl;
    if (use_eos_balanced_profile) {
      std::cout
        << "EOS-balanced p_c       = " << eos_profile.p_central << std::endl
        << "EOS-balanced M(r_*)    = " << eos_profile.mass_at_surface << std::endl
        << "EOS-balanced P(r_*)    = " << eos_profile.pressure_at_surface << std::endl
        << "EOS profile samples    = " << radial_profile_npts << std::endl
        << std::endl;
    } else {
      std::cout
        << "poly_K (derived)       = " << poly_k << std::endl
        << "Lane-Emden xi1         = " << xi1 << std::endl
        << std::endl;
    }
    for (std::size_t n = 0; n < stream_shell_level_radii_global.size(); ++n) {
      std::cout << "Stream shell tier " << (n + 1)
                << " starts at r >= " << stream_shell_level_radii_global[n]
                << " with target level "
                << std::max(pmy_mesh_->root_level,
                            pmy_mesh_->max_level - stream_shell_level_offsets_global[n])
                << " (max_level - " << stream_shell_level_offsets_global[n] << ")"
                << ", dr = " << stream_shell_level_drs_global[n]
                << ", rho_frac = " << stream_shell_level_rho_fracs_global[n]
                << ", fill_frac = " << stream_shell_level_fill_fracs_global[n]
                << std::endl;
    }
    if (!stream_shell_level_radii_global.empty()) {
      std::cout << std::endl;
    }
    std::cout
      << "provide_params         = "
      << (provide_params ? "true" : "false") << std::endl
      << "mass_ratio             = " << mass_ratio << std::endl
      << "beta                   = " << beta << std::endl
      << "ecc_bh                 = " << ecc_bh << std::endl
      << "theta_bh (deg)         = " << theta_bh_deg << std::endl
      << "sep_initial            = " << sep_initial << std::endl
      << "r_tidal                = " << r_tidal << std::endl
      << "r_peri                 = " << r_peri << std::endl
      << "BH init rel pos        = (" << bh_init_x << ", "
      << bh_init_y << ", " << bh_init_z << ")" << std::endl
      << "BH init rel vel        = (" << bh_init_vx << ", "
      << bh_init_vy << ", " << bh_init_vz << ")" << std::endl
      << std::endl;
  }
}

namespace tde_external {

void StoreRuntimeMetadata(ParameterInput *pin) {
  SyncProblemRuntimeState();
  problem_runtime::StoreRuntimeMetadata(pin);
}

}  // namespace tde_external
