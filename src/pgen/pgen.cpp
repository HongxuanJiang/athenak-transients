//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//! \brief Implementation of constructors and functions in class ProblemGenerator.
//! Default constructor calls problem generator function, while  constructor for restarts
//! reads data from restart file, as well as re-initializing problem-specific data.

#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "sink_particles/sink_particles.hpp"
#include "srcterms/turb_driver.hpp"
#include "remap/remap.hpp"
#include "pgen.hpp"

namespace {

bool excise_enabled_ = false;
Real excise_radius_ = 0.0;
Real excise_density_ = 0.0;
Real excise_eint_ = 0.0;

bool live_bh_state_valid_ = false;
bool bh_orbit_enabled_ = false;
Real bh_x_ = 0.0, bh_y_ = 0.0, bh_z_ = 0.0;
Real bh_vx_ = 0.0, bh_vy_ = 0.0, bh_vz_ = 0.0;
Real frame_x_ = 0.0, frame_y_ = 0.0, frame_z_ = 0.0;
Real frame_vx_ = 0.0, frame_vy_ = 0.0, frame_vz_ = 0.0;

bool external_bh_potential_enabled_ = false;
bool external_bh_gravity_source_coupling_ = false;
Real external_bh_mass_ = 0.0;
Real external_bh_softening_ = 0.0;
Real external_newton_g_ = 0.0;
bool bh_sink_gravity_mask_enabled_ = false;
Real bh_sink_gravity_mask_radius_ = 0.0;
problem_runtime::BlackHoleHorizonAtTimeFnPtr bh_horizon_at_time_fn_ = nullptr;

bool bh_step_valid_ = false;
Real bh_step_time_ = 0.0;
Real bh_step_dt_ = 0.0;
Real bh_step_x0_ = 0.0, bh_step_y0_ = 0.0, bh_step_z0_ = 0.0;
Real bh_step_vx0_ = 0.0, bh_step_vy0_ = 0.0, bh_step_vz0_ = 0.0;
Real bh_step_x1_ = 0.0, bh_step_y1_ = 0.0, bh_step_z1_ = 0.0;
Real bh_step_vx1_ = 0.0, bh_step_vy1_ = 0.0, bh_step_vz1_ = 0.0;

bool frame_step_valid_ = false;
Real frame_step_time_ = 0.0;
Real frame_step_dt_ = 0.0;
Real frame_step_x0_ = 0.0, frame_step_y0_ = 0.0, frame_step_z0_ = 0.0;
Real frame_step_vx0_ = 0.0, frame_step_vy0_ = 0.0, frame_step_vz0_ = 0.0;
Real frame_step_ax0_ = 0.0, frame_step_ay0_ = 0.0, frame_step_az0_ = 0.0;
Real frame_step_x1_ = 0.0, frame_step_y1_ = 0.0, frame_step_z1_ = 0.0;
Real frame_step_vx1_ = 0.0, frame_step_vy1_ = 0.0, frame_step_vz1_ = 0.0;
Real frame_step_ax1_ = 0.0, frame_step_ay1_ = 0.0, frame_step_az1_ = 0.0;

bool hydro_stage_time_valid_ = false;
Real hydro_stage_time_ = 0.0;

KOKKOS_INLINE_FUNCTION
Real ClampUnitInterval(const Real x) {
  return (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
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
  if (!bh_orbit_enabled_ || !bh_step_valid_ || bh_step_dt_ <= 0.0) {
    x = bh_x_;
    y = bh_y_;
    z = bh_z_;
    vx = bh_vx_;
    vy = bh_vy_;
    vz = bh_vz_;
    return;
  }

  Real s = ClampUnitInterval((t - bh_step_time_) / bh_step_dt_);
  x = CubicHermiteValue(bh_step_x0_, bh_step_vx0_, bh_step_x1_, bh_step_vx1_,
                        bh_step_dt_, s);
  y = CubicHermiteValue(bh_step_y0_, bh_step_vy0_, bh_step_y1_, bh_step_vy1_,
                        bh_step_dt_, s);
  z = CubicHermiteValue(bh_step_z0_, bh_step_vz0_, bh_step_z1_, bh_step_vz1_,
                        bh_step_dt_, s);
  vx = CubicHermiteSlope(bh_step_x0_, bh_step_vx0_, bh_step_x1_, bh_step_vx1_,
                         bh_step_dt_, s);
  vy = CubicHermiteSlope(bh_step_y0_, bh_step_vy0_, bh_step_y1_, bh_step_vy1_,
                         bh_step_dt_, s);
  vz = CubicHermiteSlope(bh_step_z0_, bh_step_vz0_, bh_step_z1_, bh_step_vz1_,
                         bh_step_dt_, s);
}

Real ActiveHydroStageTimeOr(const Real t) {
  return hydro_stage_time_valid_ ? hydro_stage_time_ : t;
}

}  // namespace

namespace problem_runtime {

void SetExcisionState(bool enabled, Real radius, Real density, Real eint) {
  excise_enabled_ = enabled;
  excise_radius_ = radius;
  excise_density_ = density;
  excise_eint_ = eint;
}

void SetLiveBHState(bool orbit_enabled,
                    Real bhx, Real bhy, Real bhz,
                    Real bhvx, Real bhvy, Real bhvz,
                    Real framex, Real framey, Real framez,
                    Real framevx, Real framevy, Real framevz) {
  live_bh_state_valid_ = true;
  bh_orbit_enabled_ = orbit_enabled;
  bh_x_ = bhx;
  bh_y_ = bhy;
  bh_z_ = bhz;
  bh_vx_ = bhvx;
  bh_vy_ = bhvy;
  bh_vz_ = bhvz;
  frame_x_ = framex;
  frame_y_ = framey;
  frame_z_ = framez;
  frame_vx_ = framevx;
  frame_vy_ = framevy;
  frame_vz_ = framevz;
}

void SetFrameStepState(bool valid, Real step_time, Real step_dt,
                       Real x0, Real y0, Real z0,
                       Real vx0, Real vy0, Real vz0,
                       Real ax0, Real ay0, Real az0,
                       Real x1, Real y1, Real z1,
                       Real vx1, Real vy1, Real vz1,
                       Real ax1, Real ay1, Real az1) {
  frame_step_valid_ = valid;
  frame_step_time_ = step_time;
  frame_step_dt_ = step_dt;
  frame_step_x0_ = x0;
  frame_step_y0_ = y0;
  frame_step_z0_ = z0;
  frame_step_vx0_ = vx0;
  frame_step_vy0_ = vy0;
  frame_step_vz0_ = vz0;
  frame_step_ax0_ = ax0;
  frame_step_ay0_ = ay0;
  frame_step_az0_ = az0;
  frame_step_x1_ = x1;
  frame_step_y1_ = y1;
  frame_step_z1_ = z1;
  frame_step_vx1_ = vx1;
  frame_step_vy1_ = vy1;
  frame_step_vz1_ = vz1;
  frame_step_ax1_ = ax1;
  frame_step_ay1_ = ay1;
  frame_step_az1_ = az1;
}

void SetExternalBHPotential(bool enabled, Real mass, Real softening, Real newton_g) {
  external_bh_potential_enabled_ = enabled;
  external_bh_mass_ = mass;
  external_bh_softening_ = softening;
  external_newton_g_ = newton_g;
}

void SetExternalBHGravitySourceCoupling(bool enabled) {
  external_bh_gravity_source_coupling_ = enabled;
}

void SetBHSinkGravityMask(bool enabled, Real radius) {
  bh_sink_gravity_mask_enabled_ = enabled && radius > 0.0;
  bh_sink_gravity_mask_radius_ = bh_sink_gravity_mask_enabled_ ? radius : 0.0;
}

void SetBlackHoleHorizonAtTimeFunction(BlackHoleHorizonAtTimeFnPtr func) {
  bh_horizon_at_time_fn_ = func;
}

BlackHoleHorizonState GetBlackHoleHorizonAtTime(Real time) {
  return (bh_horizon_at_time_fn_ != nullptr) ? bh_horizon_at_time_fn_(time)
                                             : BlackHoleHorizonState{};
}

void SetBHStepState(bool valid, Real step_time, Real step_dt,
                    Real x0, Real y0, Real z0,
                    Real vx0, Real vy0, Real vz0,
                    Real x1, Real y1, Real z1,
                    Real vx1, Real vy1, Real vz1) {
  bh_step_valid_ = valid;
  bh_step_time_ = step_time;
  bh_step_dt_ = step_dt;
  bh_step_x0_ = x0;
  bh_step_y0_ = y0;
  bh_step_z0_ = z0;
  bh_step_vx0_ = vx0;
  bh_step_vy0_ = vy0;
  bh_step_vz0_ = vz0;
  bh_step_x1_ = x1;
  bh_step_y1_ = y1;
  bh_step_z1_ = z1;
  bh_step_vx1_ = vx1;
  bh_step_vy1_ = vy1;
  bh_step_vz1_ = vz1;
}

void SetHydroStageTime(Real t) {
  hydro_stage_time_ = t;
  hydro_stage_time_valid_ = true;
}

void ClearHydroStageTime() {
  hydro_stage_time_valid_ = false;
}

bool HasHydroStageTime() {
  return hydro_stage_time_valid_;
}

Real HydroStageTimeOr(Real t) {
  return ActiveHydroStageTimeOr(t);
}

bool HasExternalBHPotential() {
  return external_bh_potential_enabled_;
}

bool ExternalBHGravitySourceCouplingEnabled() {
  return external_bh_potential_enabled_ && external_bh_gravity_source_coupling_;
}

void GetExternalBHPotential(const Real t, bool &enabled,
                            Real &bhx, Real &bhy, Real &bhz,
                            Real &mass, Real &softening, Real &newton_g) {
  enabled = external_bh_potential_enabled_;
  if (!enabled) {
    bhx = 0.0;
    bhy = 0.0;
    bhz = 0.0;
    mass = 0.0;
    softening = 0.0;
    newton_g = 0.0;
    return;
  }

  Real bhvx = 0.0;
  Real bhvy = 0.0;
  Real bhvz = 0.0;
  InterpolateBHStepState(ActiveHydroStageTimeOr(t), bhx, bhy, bhz, bhvx, bhvy, bhvz);
  mass = external_bh_mass_;
  softening = external_bh_softening_;
  newton_g = external_newton_g_;
}

void GetExternalBHPotentialAtExactTime(const Real t, bool &enabled,
                                       Real &bhx, Real &bhy, Real &bhz,
                                       Real &mass, Real &softening,
                                       Real &newton_g) {
  enabled = external_bh_potential_enabled_;
  if (!enabled) {
    bhx = 0.0;
    bhy = 0.0;
    bhz = 0.0;
    mass = 0.0;
    softening = 0.0;
    newton_g = 0.0;
    return;
  }

  Real bhvx = 0.0;
  Real bhvy = 0.0;
  Real bhvz = 0.0;
  InterpolateBHStepState(t, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  mass = external_bh_mass_;
  softening = external_bh_softening_;
  newton_g = external_newton_g_;
}

void GetBHSinkGravityMask(const Real t, bool &enabled, Real &radius,
                          Real &center_x, Real &center_y, Real &center_z) {
  enabled = bh_sink_gravity_mask_enabled_ && live_bh_state_valid_;
  if (!enabled) {
    radius = 0.0;
    center_x = 0.0;
    center_y = 0.0;
    center_z = 0.0;
    return;
  }

  radius = bh_sink_gravity_mask_radius_;
  Real bhvx = 0.0;
  Real bhvy = 0.0;
  Real bhvz = 0.0;
  InterpolateBHStepState(ActiveHydroStageTimeOr(t), center_x, center_y, center_z,
                         bhvx, bhvy, bhvz);
}

void GetExcisionState(const Real t, bool &enabled, Real &radius, Real &density,
    Real &eint,
                      Real &center_x, Real &center_y, Real &center_z) {
  enabled = excise_enabled_;
  if (!enabled) {
    radius = 0.0;
    density = 0.0;
    eint = 0.0;
    center_x = 0.0;
    center_y = 0.0;
    center_z = 0.0;
    return;
  }

  radius = excise_radius_;
  density = excise_density_;
  eint = excise_eint_;
  Real bhvx = 0.0;
  Real bhvy = 0.0;
  Real bhvz = 0.0;
  InterpolateBHStepState(ActiveHydroStageTimeOr(t), center_x, center_y, center_z,
                         bhvx, bhvy, bhvz);
}

void StoreRuntimeMetadata(ParameterInput *pin) {
  if (pin == nullptr) return;
  if (!live_bh_state_valid_) {
    pin->SetBoolean("problem", "bh_live_state_valid", false);
    return;
  }
  pin->SetBoolean("problem", "bh_live_state_valid", true);
  pin->SetBoolean("problem", "bh_live_orbit_enabled", bh_orbit_enabled_);
  pin->SetReal("problem", "bh_live_x", bh_x_);
  pin->SetReal("problem", "bh_live_y", bh_y_);
  pin->SetReal("problem", "bh_live_z", bh_z_);
  pin->SetReal("problem", "bh_live_vx", bh_vx_);
  pin->SetReal("problem", "bh_live_vy", bh_vy_);
  pin->SetReal("problem", "bh_live_vz", bh_vz_);
  pin->SetReal("problem", "frame_live_x", frame_x_);
  pin->SetReal("problem", "frame_live_y", frame_y_);
  pin->SetReal("problem", "frame_live_z", frame_z_);
  pin->SetReal("problem", "frame_live_vx", frame_vx_);
  pin->SetReal("problem", "frame_live_vy", frame_vy_);
  pin->SetReal("problem", "frame_live_vz", frame_vz_);
  pin->SetBoolean("problem", "frame_step_state_valid", frame_step_valid_);
  pin->SetReal("problem", "frame_step_time", frame_step_time_);
  pin->SetReal("problem", "frame_step_dt", frame_step_dt_);
  pin->SetReal("problem", "frame_step_x0", frame_step_x0_);
  pin->SetReal("problem", "frame_step_y0", frame_step_y0_);
  pin->SetReal("problem", "frame_step_z0", frame_step_z0_);
  pin->SetReal("problem", "frame_step_vx0", frame_step_vx0_);
  pin->SetReal("problem", "frame_step_vy0", frame_step_vy0_);
  pin->SetReal("problem", "frame_step_vz0", frame_step_vz0_);
  pin->SetReal("problem", "frame_step_ax0", frame_step_ax0_);
  pin->SetReal("problem", "frame_step_ay0", frame_step_ay0_);
  pin->SetReal("problem", "frame_step_az0", frame_step_az0_);
  pin->SetReal("problem", "frame_step_x1", frame_step_x1_);
  pin->SetReal("problem", "frame_step_y1", frame_step_y1_);
  pin->SetReal("problem", "frame_step_z1", frame_step_z1_);
  pin->SetReal("problem", "frame_step_vx1", frame_step_vx1_);
  pin->SetReal("problem", "frame_step_vy1", frame_step_vy1_);
  pin->SetReal("problem", "frame_step_vz1", frame_step_vz1_);
  pin->SetReal("problem", "frame_step_ax1", frame_step_ax1_);
  pin->SetReal("problem", "frame_step_ay1", frame_step_ay1_);
  pin->SetReal("problem", "frame_step_az1", frame_step_az1_);
  pin->SetBoolean("problem", "bh_step_state_valid", bh_step_valid_);
  pin->SetReal("problem", "bh_step_time", bh_step_time_);
  pin->SetReal("problem", "bh_step_dt", bh_step_dt_);
  pin->SetReal("problem", "bh_step_x0", bh_step_x0_);
  pin->SetReal("problem", "bh_step_y0", bh_step_y0_);
  pin->SetReal("problem", "bh_step_z0", bh_step_z0_);
  pin->SetReal("problem", "bh_step_vx0", bh_step_vx0_);
  pin->SetReal("problem", "bh_step_vy0", bh_step_vy0_);
  pin->SetReal("problem", "bh_step_vz0", bh_step_vz0_);
  pin->SetReal("problem", "bh_step_x1", bh_step_x1_);
  pin->SetReal("problem", "bh_step_y1", bh_step_y1_);
  pin->SetReal("problem", "bh_step_z1", bh_step_z1_);
  pin->SetReal("problem", "bh_step_vx1", bh_step_vx1_);
  pin->SetReal("problem", "bh_step_vy1", bh_step_vy1_);
  pin->SetReal("problem", "bh_step_vz1", bh_step_vz1_);
  pin->SetBoolean("problem", "external_bh_potential_valid",
      external_bh_potential_enabled_);
  if (external_bh_potential_enabled_) {
    pin->SetReal("problem", "external_bh_mass", external_bh_mass_);
    pin->SetReal("problem", "external_bh_softening", external_bh_softening_);
    pin->SetReal("problem", "external_newton_g", external_newton_g_);
  }
  pin->SetBoolean("problem", "external_bh_gravity_source_coupling",
                  external_bh_gravity_source_coupling_);
}

}  // namespace problem_runtime

//----------------------------------------------------------------------------------------
// default constructor, calls pgen function.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm) :
    user_bcs(false),
    user_bcs_lat_safe(false),
    user_srcs(false),
    user_srcs_lat_safe(false),
    user_srcs_lat_union_safe(false),
    user_hydro_state_fixup_lat_union_safe(false),
    user_dt_lat_safe(false),
    user_hist(false),
    pmy_mesh_(pm) {
  user_srcs_time = 0.0;
  user_srcs_time_valid = false;
  user_srcs_final_weight = 1.0;
  user_srcs_step_dt = 0.0;
  user_srcs_stage = 0;
  user_srcs_nstages = 0;
  user_srcs_stage_valid = false;
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }

  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);

  // second argument false since this IS NOT a restart
  CallProblemGenerator(pin, false);

  // <remap>-block auto remap (src/remap/): runs after the pgen function so pgens can
  // enroll the user_remap_* hooks first.  No-op unless the input carries <remap>.
  remap::MaybeAutoRemap(this, pin, pm);

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// constructor for restarts
// When called, data needed to rebuild mesh has been read from restart file by
// Mesh::BuildTreeFromRestart() function. This constructor reads from the restart file and
// initializes all the dependent variables (u0,b0,etc) stored in each Physics class. It
// also calls ProblemGenerator::SetProblemData() function to set any user-defined BCs,
// and any data necessary for restart runs to continue correctly.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile,
                                   bool single_file_per_rank) :
    user_bcs(false),
    user_bcs_lat_safe(false),
    user_srcs(false),
    user_srcs_lat_safe(false),
    user_srcs_lat_union_safe(false),
    user_hydro_state_fixup_lat_union_safe(false),
    user_dt_lat_safe(false),
    user_hist(false),
    pmy_mesh_(pm) {
  user_srcs_time = 0.0;
  user_srcs_time_valid = false;
  user_srcs_final_weight = 1.0;
  user_srcs_step_dt = 0.0;
  user_srcs_stage = 0;
  user_srcs_nstages = 0;
  user_srcs_stage_valid = false;
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }
  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);

  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;
  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad=pm->pmb_pack->prad;
  sinkparticles::SinkParticles* psink=pm->pmb_pack->psink;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nrad = 0, nhydro = 0, nmhd = 0, nforce = 3, nadm = 0, nz4c = 0;
  int nhydro_file = 0;
  int nhydro_legacy = 0;
  int nhydro_dual = 0;
  int nmhd_file = 0;
  int nmhd_legacy = 0;
  int nmhd_dual = 0;
  bool hydro_restart_missing_dual = false;
  bool hydro_restart_extra_dual = false;
  bool mhd_restart_missing_dual = false;
  bool mhd_restart_extra_dual = false;
  if (phydro != nullptr) {
    nhydro = phydro->nvars;
    nhydro_file = nhydro;
    nhydro_legacy = phydro->nhydro + phydro->nscalars;
    nhydro_dual = nhydro_legacy + ((phydro->peos->eos_data.use_e) ? 1 : 0);
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nvars;
    nmhd_file = nmhd;
    nmhd_legacy = pmhd->nmhd + pmhd->nscalars;
    nmhd_dual = nmhd_legacy + ((pmhd->peos->eos_data.use_e) ? 1 : 0);
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->RestartVariableCount();
  }

  // root process reads z4c last_output_time and tracker data
  if (pz4c != nullptr) {
    Real last_output_time = 0.0;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_Reals(&last_output_time, 1,single_file_per_rank) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "z4c::last_output_time data size read from restart "
                  << "file is incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(&last_output_time, sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    pz4c->last_output_time = last_output_time;

    for (auto &pt : pz4c->ptracker) {
      Real pos[3];
      if (global_variable::my_rank == 0 || single_file_per_rank) {
        if (resfile.Read_Reals(&pos[0], 3, single_file_per_rank) != 3) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "compact object tracker data size read from restart "
                    << "file is incorrect, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (!single_file_per_rank) {
        MPI_Bcast(&pos[0], 3*sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
      }
#endif
      pt->SetPos(&pos[0]);
    }
  }

  if (pturb != nullptr) {
    // root process reads size the random seed
    char *rng_data = new char[sizeof(RNG_State)];
    // the master process reads the variables data
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(rng_data, 1, sizeof(RNG_State), single_file_per_rank)
          != sizeof(RNG_State)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "RNG data size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      // then broadcast the RNG information
      MPI_Bcast(rng_data, sizeof(RNG_State), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    std::memcpy(&(pturb->rstate), &(rng_data[0]), sizeof(RNG_State));
  }

  // Sink-particle list: symmetric to the writer's header block (count + packed POD
  // image, written once after the turbulence RNG state).  The list is replicated on
  // every rank, so root reads and broadcasts.
  if (psink != nullptr) {
    int nsinks_file = 0;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(reinterpret_cast<char*>(&nsinks_file), 1, sizeof(int),
                             single_file_per_rank) != sizeof(int)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "sink count read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(&nsinks_file, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    // Review A2/F8: bound the FILE-supplied count before it sizes an allocation.  When
    // <sink_particles> is added on restart to a run that did not have it (the ordinary
    // workflow), these four bytes are the first four of the IOWrapperSizeT variablesize
    // field, so nsinks_file is an arbitrary large integer and the vector below threw
    // std::bad_alloc / length_error instead of reporting a diagnosable condition.
    if (nsinks_file < 0 ||
        nsinks_file > sinkparticles::SinkParticles::kMaxRestartSinks) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "implausible sink count " << nsinks_file
                << " in the restart header (limit "
                << sinkparticles::SinkParticles::kMaxRestartSinks
                << "). The file was probably written by a run WITHOUT "
                << "<sink_particles>, whose header carries no sink block." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (nsinks_file > 0) {
      const std::size_t sink_bytes =
          sinkparticles::SinkParticles::RestartDataSizeForCount(nsinks_file);
      std::vector<char> sink_data(sink_bytes);
      if (global_variable::my_rank == 0 || single_file_per_rank) {
        if (resfile.Read_bytes(sink_data.data(), 1, sink_bytes, single_file_per_rank)
            != sink_bytes) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "sink data read from restart file is incorrect, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (!single_file_per_rank) {
        MPI_Bcast(sink_data.data(), static_cast<int>(sink_bytes), MPI_CHAR, 0,
                  MPI_COMM_WORLD);
      }
#endif
      psink->UnpackRestartData(sink_data.data(), sink_bytes);
    }
  }

  // Each MeshBlock's cycles since its last refinement, in the writer's gid order
  // (restart.cpp STEP 3), put back by gid over the zeros MeshRefinement starts from.
  const int nncyc_since_ref_file =
      pin->DoesParameterExist("mesh_refinement", "restart_ncyc_since_ref_count") ?
      pin->GetInteger("mesh_refinement", "restart_ncyc_since_ref_count") : 0;
  if (nncyc_since_ref_file > 0) {
    if (nncyc_since_ref_file != pm->nmb_total) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "restart_ncyc_since_ref_count = " << nncyc_since_ref_file
                << " differs from the checkpoint's " << pm->nmb_total
                << " MeshBlocks; restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    std::vector<int> ncyc_since_ref_file(nncyc_since_ref_file);
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(reinterpret_cast<char*>(ncyc_since_ref_file.data()),
                             sizeof(int), nncyc_since_ref_file, single_file_per_rank)
          != static_cast<std::size_t>(nncyc_since_ref_file)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MeshBlock refinement ages read from restart file "
                  << "are incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(ncyc_since_ref_file.data(), nncyc_since_ref_file, MPI_INT, 0,
                MPI_COMM_WORLD);
    }
#endif
    if (pm->adaptive && pm->pmr != nullptr) {
      for (int gid=0; gid<pm->nmb_total; ++gid) {
        const int file_gid = (pm->restart_gid_eachmb != nullptr) ?
            pm->restart_gid_eachmb[gid] : gid;
        pm->pmr->ncyc_since_ref(gid) = ncyc_since_ref_file[file_gid];
      }
    }
  }

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (resfile.Read_bytes(variabledata, 1, variablesize, single_file_per_rank)
        != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  if (!single_file_per_rank) {
    MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif
  IOWrapperSizeT data_size;
  std::memcpy(&data_size, &(variabledata[0]), sizeof(IOWrapperSizeT));

  // calculate total number of CC variables
  IOWrapperSizeT headeroffset=0;
  // master process gets file offset
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    headeroffset = resfile.GetPosition(single_file_per_rank);
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  if (!single_file_per_rank) {
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  IOWrapperSizeT data_size_ = 0;
  IOWrapperSizeT data_size_legacy = 0;
  if (phydro != nullptr) {
    data_size_ += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
    data_size_legacy += nout1*nout2*nout3*nhydro_legacy*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size_ += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size_legacy += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size_ += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size_legacy += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size_ += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size_legacy += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size_ += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    data_size_legacy += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
  }
  if (prad != nullptr) {
    data_size_ += nout1*nout2*nout3*nrad*sizeof(Real);   // rad i0
    data_size_legacy += nout1*nout2*nout3*nrad*sizeof(Real);   // rad i0
  }
  if (pturb != nullptr) {
    data_size_ += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    data_size_legacy += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size_ += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
    data_size_legacy += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr && nadm > 0) {
    data_size_ += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
    data_size_legacy += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
  }

  const IOWrapperSizeT cc_cell_bytes = nout1*nout2*nout3*sizeof(Real);
  const IOWrapperSizeT fixed_cc_bytes = data_size_ - cc_cell_bytes*(nhydro + nmhd);
  int hydro_file_options[2] = {nhydro, nhydro};
  int nhydro_options = 1;
  if (phydro != nullptr) {
    hydro_file_options[0] = nhydro_legacy;
    hydro_file_options[1] = nhydro_dual;
    nhydro_options = (nhydro_dual != nhydro_legacy) ? 2 : 1;
  }
  int mhd_file_options[8] = {nmhd, nmhd, nmhd, nmhd, nmhd, nmhd, nmhd, nmhd};
  constexpr int max_mhd_file_options =
      static_cast<int>(sizeof(mhd_file_options)/sizeof(mhd_file_options[0]));
  int nmhd_options = 1;
  if (pmhd != nullptr) {
    nmhd_options = 0;
    auto add_mhd_file_option = [&](const int candidate) {
      if (candidate <= 0 || nmhd_options >= max_mhd_file_options) return;
      for (int n = 0; n < nmhd_options; ++n) {
        if (mhd_file_options[n] == candidate) return;
      }
      mhd_file_options[nmhd_options++] = candidate;
    };
    add_mhd_file_option(nmhd_legacy);
    add_mhd_file_option(nmhd_dual);
  }
  if (data_size_ != data_size) {
    bool matched_restart_layout = false;
    for (int ih = 0; ih < nhydro_options && !matched_restart_layout; ++ih) {
      for (int im = 0; im < nmhd_options && !matched_restart_layout; ++im) {
        const int nhydro_candidate = (phydro != nullptr) ? hydro_file_options[ih] : 0;
        const int nmhd_candidate = (pmhd != nullptr) ? mhd_file_options[im] : 0;
        IOWrapperSizeT fixed_candidate = fixed_cc_bytes;
        const IOWrapperSizeT candidate_size =
            fixed_candidate + cc_cell_bytes*(nhydro_candidate + nmhd_candidate);
        if (candidate_size != data_size) continue;

        nhydro_file = nhydro_candidate;
        nmhd_file = nmhd_candidate;
        hydro_restart_missing_dual =
            (phydro != nullptr) && phydro->use_dual_energy && (nhydro_file < nhydro);
        hydro_restart_extra_dual =
            (phydro != nullptr) && !(phydro->use_dual_energy) && (nhydro_file > nhydro);
        // The MHD width can now be short for a second reason, so name the matched
        // layout exactly instead of testing an inequality.
        mhd_restart_missing_dual =
            (pmhd != nullptr) && pmhd->use_dual_energy && (nmhd_legacy < nmhd) &&
            (nmhd_file == nmhd_legacy);
        mhd_restart_extra_dual =
            (pmhd != nullptr) && !(pmhd->use_dual_energy) && (nmhd_dual > nmhd) &&
            (nmhd_file == nmhd_dual);
        data_size_ = candidate_size;
        matched_restart_layout = true;
      }
    }
  }
  if ((phydro != nullptr) && phydro->use_dual_energy && !hydro_restart_missing_dual) {
    phydro->dual_energy_needs_init = false;
  }
  if ((pmhd != nullptr) && pmhd->use_dual_energy && !mhd_restart_missing_dual) {
    pmhd->dual_energy_needs_init = false;
  }

  if (data_size_ != data_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro, MHD, Rad, and/or Z4c arrays, restart file is broken."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (hydro_restart_extra_dual) {
      std::cout << "Restart contains hydro dual-energy auxiliary data, but "
                << "<hydro>/dual_energy = false. Ignoring restart eint_aux field."
                << std::endl;
    }
    if (mhd_restart_extra_dual) {
      std::cout << "Restart contains MHD dual-energy auxiliary data, but "
                << "<mhd>/dual_energy = false. Ignoring restart eint_aux field."
                << std::endl;
    }
  }

  // read CC data into host array
  IOWrapperSizeT offset_myrank = headeroffset;
  if (!single_file_per_rank) {
    offset_myrank += data_size_ * pm->gids_eachrank[global_variable::my_rank];
  }
  IOWrapperSizeT myoffset = offset_myrank;

  HostArray5D<Real> ccin("rst-cc-in", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> fcin("rst-fc-in", 1, 1, 1, 1);
  const bool mapped_restart_read =
      (!single_file_per_rank && pm->restart_gid_eachmb != nullptr);
  const int restart_gid_base =
      (mapped_restart_read ? pm->gids_eachrank[global_variable::my_rank] : 0);
  std::vector<int> restart_gid_local;
  if (mapped_restart_read) {
    restart_gid_local.resize(nmb);
    for (int m=0; m<nmb; ++m) {
      const int gid = restart_gid_base + m;
      restart_gid_local[m] = pm->restart_gid_eachmb[gid];
      if (restart_gid_local[m] < 0 || restart_gid_local[m] >= pm->nmb_total) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "LAT restart GID map is invalid for gid=" << gid
                  << ", file gid=" << restart_gid_local[m] << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  auto block_offset = [&](const int m, const IOWrapperSizeT field_offset) {
    const int file_gid = mapped_restart_read ?
        restart_gid_local[m] : (pm->gids_eachrank[global_variable::my_rank] + m);
    return headeroffset + static_cast<IOWrapperSizeT>(file_gid)*data_size_ + field_offset;
  };
  auto read_cc_mapped = [&](HostArray5D<Real> &arr, const IOWrapperSizeT field_offset,
                            const char *label) {
    for (int m=0; m<nmb; ++m) {
      auto mbptr = Kokkos::subview(arr, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      const int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at(mbptr.data(), mbcnt, block_offset(m, field_offset),
                                single_file_per_rank) != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC " << label
                  << " data not read correctly from mapped rst file, "
                  << "restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  };
  auto read_fc_mapped = [&](HostFaceFld4D<Real> &fld, const IOWrapperSizeT field_offset) {
    for (int m=0; m<nmb; ++m) {
      IOWrapperSizeT off = block_offset(m, field_offset);
      auto x1fptr = Kokkos::subview(fld.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      int fldcnt = x1fptr.size();
      if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, off,
                                single_file_per_rank) != fldcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input b0.x1f field not read correctly from "
                  << "mapped rst file, restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      off += static_cast<IOWrapperSizeT>(fldcnt)*sizeof(Real);

      auto x2fptr = Kokkos::subview(fld.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x2fptr.size();
      if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, off,
                                single_file_per_rank) != fldcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input b0.x2f field not read correctly from "
                  << "mapped rst file, restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      off += static_cast<IOWrapperSizeT>(fldcnt)*sizeof(Real);

      auto x3fptr = Kokkos::subview(fld.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x3fptr.size();
      if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, off,
                                single_file_per_rank) != fldcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input b0.x3f field not read correctly from "
                  << "mapped rst file, restart file is broken." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  };

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmb_eachrank[0];
  int noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
  }

  if (phydro != nullptr) {
    Kokkos::realloc(ccin, nmb, nhydro_file, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, 0, "hydro");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    HostArray5D<Real> ccfull("rst-cc-host", nmb, phydro->nvars, nout3, nout2, nout1);
    Kokkos::deep_copy(ccfull, 0.0);
    const int nhydro_copy = std::min(nhydro, nhydro_file);
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nhydro_copy; ++n) {
        for (int k=0; k<nout3; ++k) {
          for (int j=0; j<nout2; ++j) {
            for (int i=0; i<nout1; ++i) {
              ccfull(m,n,k,j,i) = ccin(m,n,k,j,i);
            }
          }
        }
      }
    }
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccfull);
    offset_myrank += nout1*nout2*nout3*nhydro_file*sizeof(Real); // hydro u0
    myoffset = offset_myrank;
    if (hydro_restart_missing_dual) {
      phydro->dual_energy_needs_init = true;
    }
  }

  if (pmhd != nullptr) {
    Kokkos::realloc(ccin, nmb, nmhd_file, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank], "mhd");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    HostArray5D<Real> ccfull("rst-mhd-host", nmb, pmhd->nvars, nout3, nout2, nout1);
    Kokkos::deep_copy(ccfull, 0.0);
    const int nmhd_copy = std::min(nmhd, nmhd_file);
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nmhd_copy; ++n) {
        for (int k=0; k<nout3; ++k) {
          for (int j=0; j<nout2; ++j) {
            for (int i=0; i<nout1; ++i) {
              ccfull(m,n,k,j,i) = ccin(m,n,k,j,i);
            }
          }
        }
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccfull);
    offset_myrank += nout1*nout2*nout3*nmhd_file*sizeof(Real);   // mhd u0
    myoffset = offset_myrank;
    if (mhd_restart_missing_dual) {
      pmhd->dual_energy_needs_init = true;
    }

    Kokkos::realloc(fcin.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(fcin.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(fcin.x3f, nmb, nout3+1, nout2, nout1);
    // read FC data into host array, again one MeshBlock at a time
    if (mapped_restart_read) {
      read_fc_mapped(fcin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank]);
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        size_t fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        size_t fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      }
    }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x1f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x2f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x3f);
    offset_myrank += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    offset_myrank += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    offset_myrank += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    myoffset = offset_myrank;
  }

  if (prad != nullptr) {
    Kokkos::realloc(ccin, nmb, nrad, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank], "rad");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    Kokkos::deep_copy(Kokkos::subview(prad->i0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
    myoffset = offset_myrank;
  }

  if (pturb != nullptr) {
    Kokkos::realloc(ccin, nmb, nforce, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank], "turb");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    Kokkos::deep_copy(Kokkos::subview(pturb->force, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    myoffset = offset_myrank;
  }

  if (pz4c != nullptr) {
    Kokkos::realloc(ccin, nmb, nz4c, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank], "z4c");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
    myoffset = offset_myrank;

    // We also need to reinitialize the ADM data.
    pz4c->Z4cToADM(pmy_mesh_->pmb_pack);
  } else if (padm != nullptr && nadm > 0) {
    Kokkos::realloc(ccin, nmb, nadm, nout3, nout2, nout1);
    if (mapped_restart_read) {
      read_cc_mapped(ccin, offset_myrank - headeroffset -
                     data_size_ * pm->gids_eachrank[global_variable::my_rank], "adm");
    } else {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    }
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
    myoffset = offset_myrank;
  }

  if (mapped_restart_read) {
    delete [] pm->restart_gid_eachmb;
    pm->restart_gid_eachmb = nullptr;
  }

  // Every conversion above rewrites cells the checkpoint did not carry in this layout,
  // and an active hybrid-FFE payload restored without its version-6 recovery history is
  // resynchronised through the boundary exchange (the one-shot restart integration of
  // Driver::InitBoundaryValuesAndPrimitives).  A dormant sidecar -- a checkpoint written
  // before the first activation, one written without a sidecar, or a discarded legacy
  // payload -- transports nothing that exchange would rebuild, and its next synchronized
  // classification reseeds the owned tails from the ordinary state, so like an active
  // payload with its history it keeps the stored ghost zones (Driver::Initialize).
  restart_state_verbatim =
      !hydro_restart_missing_dual && !mhd_restart_missing_dual;
  // call problem generator again to re-initialize data, fn ptrs, as needed
  // second argument true since this IS a restart
  CallProblemGenerator(pin, true);

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "during restart by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, "
                << "but not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::OutputErrors()
//! \brief Generic function for computing the L1 and L-infty difference between solutions
//! stored in the u0 and u1 registers, and outputting them to an error file.  This is
//! used for linear wave convergence tests, for example.
//! Function requires appropriate solutions already stored in u0 and u1.

void ProblemGenerator::OutputErrors(ParameterInput *pin, Mesh *pm) {
  Real l1_err[16];
  Real linfty_err=0.0;
  int nvars=0,nprev=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    auto &is_ideal_ = pmbp->phydro->peos->eos_data.is_ideal;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("L1-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (is_ideal_) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
    nprev += nvars;
  }

  // compute errors for MHD  -------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + 3;  // include 3-compts of cell-centered B in errors
    auto &is_ideal_ = pmbp->pmhd->peos->eos_data.is_ideal;

    int bindx;
    if (is_ideal_) {
      bindx = 5;
    } else {
      bindx = 4;
    }

    auto &u0_ = pmbp->pmhd->u0;
    auto &u1_ = pmbp->pmhd->u1;
    auto &b0_ = pmbp->pmhd->b0;
    auto &b1_ = pmbp->pmhd->b1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("L1-err-Sums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (is_ideal_) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // cell-centered B
      Real bcc0 = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
      Real bcc1 = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      evars.the_array[bindx] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+1]);

      bcc0 = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
      bcc1 = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      evars.the_array[bindx+1] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+2]);

      bcc0 = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
      bcc1 = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
      evars.the_array[bindx+2] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+3]);

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n+nprev] = sum_this_mb.the_array[n];
    }
    nprev += nvars;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nprev, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nprev; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nprev; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle   RMS-L1       L-infty       ");
      if (pmbp->phydro != nullptr) {
        std::fprintf(pfile,"d_L1          M1_L1         M2_L1         M3_L1         ");
        if (pmbp->phydro->peos->eos_data.is_ideal) {
          std::fprintf(pfile,"E_L1          ");
        }
      }
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile,"d_L1          M1_L1         M2_L1         M3_L1         ");
        if (pmbp->pmhd->peos->eos_data.is_ideal) {
          std::fprintf(pfile,"E_L1          ");
        }
        std::fprintf(pfile,"B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nprev; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CallProblemGenerator()
//! \brief selects one of the default problem generators compiled automatically with
//! the source code depending on input string in <problem> block ELSE selects a
//! user-defined problem generator function compiled with the code.

void ProblemGenerator::CallProblemGenerator(ParameterInput *pin, bool is_restart) {
#if USER_PROBLEM_ENABLED
  UserProblem(pin, is_restart);
#else
  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, is_restart);
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, is_restart);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, is_restart);
  } else if (pgen_fun_name.compare("cshock") == 0) {
    CShock(pin, is_restart);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, is_restart);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, is_restart);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, is_restart);
  } else if (pgen_fun_name.compare("mri3d") == 0) {
    MRI3d(pin, is_restart);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, is_restart);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("rad_beam") == 0) {
    RadiationBeam(pin, is_restart);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, is_restart);
  } else if (pgen_fun_name.compare("shwave") == 0) {
    Shwave(pin, is_restart);
  } else if (pgen_fun_name.compare("z4c_boosted_puncture") == 0) {
    Z4cBoostedPuncture(pin, is_restart);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("spherical_collapse") == 0) {
    SphericalCollapse(pin, is_restart);
  } else if (pgen_fun_name.compare("gravity") == 0) {
    SelfGravity(pin, is_restart);
  } else if (pgen_fun_name.compare("binary_gravity") == 0) {
    BinaryGravity(pin, is_restart);
  } else if (pgen_fun_name.compare("be_collapse") == 0) {
    BECollapse(pin, is_restart);
  } else if (pgen_fun_name.compare("polytropic_star") == 0) {
    PolytropicStar(pin, is_restart);
  } else if (pgen_fun_name.compare("remap_test") == 0) {
    RemapTest(pin, is_restart);

  // pre-defined unit tests
  } else if (pgen_fun_name.compare("eos_compose") == 0) {
    EOSCompose(pin, is_restart);
  } else if (pgen_fun_name.compare("gauss_legendre") == 0) {
    GaussLegendre(pin, is_restart);
  } else if (pgen_fun_name.compare("hydro_plm_unit") == 0) {
    HydroPLMUnit(pin, is_restart);

  } else {
    // name not set on command line or input file, print warning and quit
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl
        << "and it was not set by -D PROBLEM option on cmake command line during build"
        << std::endl
        << "Rerun cmake with -D PROBLEM=file to specify custom problem generator file"
        << std::endl;;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}
