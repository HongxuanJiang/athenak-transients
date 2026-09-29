#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.hpp
//  \brief definitions for ProblemGenerator class

#include <functional>
#include <memory>
#include <vector>

#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"
#include "pgen/bh_force_pair.hpp"
#include "remap/remap.hpp"

class Driver;
class Mesh;
class MeshBlockPack;
struct HistoryData;

using ProblemFinalizeFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserBoundaryFnPtr = void (*)(Mesh* pm);
using UserSrctermFnPtr = void (*)(Mesh* pm, const Real bdt);
using UserTimeStepFnPtr = Real (*)(Mesh* pm, Driver *pdriver);
using UserRefinementFnPtr = void (*)(MeshBlockPack* pmbp);
using UserMetadataFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserHistoryFnPtr = void (*)(HistoryData *pdata, Mesh *pm);
using ProblemAfterCycleFnPtr = bool (*)(Driver *driver, ParameterInput *pin, Mesh *pm);
using UserOutputGateFnPtr = bool (*)(Mesh *pm);
using UserHydroStateFixupFnPtr = void (*)(MeshBlockPack *pmbp, const Real time);
using UserHydroLATFactorCapFnPtr = void (*)(Mesh *pm, int max_factor,
                                            int *lat_factor_eachmb);
using UserHydroLATWindowFnPtr = void (*)(Mesh *pm, const Real window_dt);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile,
                   bool single_file_per_rank=false);
  ~ProblemGenerator() = default;

  // true if user BCs are specified on any face
  bool user_bcs;

  // true only when the enrolled user boundary function is active-mask aware under LAT
  bool user_bcs_lat_safe;

  // true if user srcterms are specified
  bool user_srcs;

  // true only when the enrolled user source term is active-mask aware under HD LAT
  bool user_srcs_lat_safe;

  // true only when the enrolled source accepts per-MeshBlock timesteps during the
  // unioned RK2 predictor stage
  bool user_srcs_lat_union_safe;

  // true only when the enrolled state repair is idempotent under the union predictor's
  // deferred boundary refresh
  bool user_hydro_state_fixup_lat_union_safe;

  // true only when the enrolled user timestep function is safe to use under HD LAT
  bool user_dt_lat_safe;

  // true if user history outputs are specified
  bool user_hist;
  // A restart whose evolved arrays were restored exactly as the checkpoint holds them,
  // ghost zones included: no legacy-layout conversion, no module seeded from another,
  // no re-initialisation by the problem generator.  Driver::Initialize then keeps the
  // stored ghost zones instead of re-deriving them (InitBoundaryValuesAndPrimitives),
  // which is what lets a restart continue the checkpointed run bitwise.  A UserProblem
  // restart branch that rewrites the state must clear it.  False on a fresh start.
  bool restart_state_verbatim=false;

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // function pointer for final work after main loop (e.g. compute errors).  Called by
  // Driver::Finalize()
  ProblemFinalizeFnPtr pgen_final_func=nullptr;
  // function pointer for user-enrolled BCs.  Called in ApplyPhysicalBCs in task list
  UserBoundaryFnPtr user_bcs_func=nullptr;
  // Optional remap-module hooks (src/remap/).  When the input has a <remap> block, the
  // fresh-start constructor runs remap::MaybeAutoRemap right after the pgen function
  // returns; a pgen may enroll these inside its pgen function to participate:
  //   skip:   cells for which it returns true keep the ambient floor (e.g. excision)
  //   loaded: called with the SOURCE restart's ParameterInput before the state is
  //           applied (restore pgen metadata such as a BH/frame record)
  //   post:   called after the remap is applied (frame switches, reseeds, banners)
  std::function<bool(Real, Real, Real)> user_remap_skip_func;
  std::function<void(ParameterInput *)> user_remap_loaded_func;
  std::function<void(const remap::RemapSummary &)> user_remap_post_func;
  UserSrctermFnPtr user_srcs_func=nullptr;
  UserTimeStepFnPtr user_dt_func=nullptr;
  UserRefinementFnPtr user_ref_func=nullptr;
  UserMetadataFnPtr user_metadata_func=nullptr;
  UserHistoryFnPtr user_hist_func=nullptr;
  ProblemAfterCycleFnPtr after_cycle_func=nullptr;
  UserOutputGateFnPtr user_output_gate_func=nullptr;
  UserHydroStateFixupFnPtr user_hydro_state_fixup_func=nullptr;
  UserHydroLATFactorCapFnPtr user_hydro_lat_factor_cap_func=nullptr;
  // Called once from the globally synchronized LAT driver path after the current
  // window duration is known and before any rank-local bin work begins. User
  // implementations may safely use MPI collectives here.
  UserHydroLATWindowFnPtr user_hydro_lat_window_func=nullptr;
  // Optional HD-only stage ledger. final_dt is the contribution of this stage
  // to the completed RK step, not beta*dt. No MPI collectives are allowed here.
  UserSrctermFnPtr user_hydro_gravity_ledger_func=nullptr;
  // Called after all LAT bins/reflux finish, before output and AMR. May reduce
  // rank-local ledgers; does not run at intermediate LAT ticks.
  UserBoundaryFnPtr user_hydro_lat_window_end_func=nullptr;
  // Experimental paired-BH mode: use the analytic Plummer derivative for gas
  // momentum, while retaining potential-difference mass-flux work.
  bool analytic_external_bh_momentum=false;
  Real user_srcs_time=0.0;
  bool user_srcs_time_valid=false;
  Real user_srcs_final_weight=1.0;
  Real user_srcs_step_dt=0.0;
  int user_srcs_stage=0;
  int user_srcs_nstages=0;
  bool user_srcs_stage_valid=false;

  // predefined problem generator functions (default test suite)
  void CallProblemGenerator(ParameterInput *pin, bool is_restart);
  void Advection(ParameterInput *pin, const bool restart);
  void AlfvenWave(ParameterInput *pin, const bool restart);
  void BondiAccretion(ParameterInput *pin, const bool restart);
  void CShock(ParameterInput *pin, const bool restart);
  void Diffusion(ParameterInput *pin, const bool restart);
  void LinearWave(ParameterInput *pin, const bool restart);
  void LWImplode(ParameterInput *pin, const bool restart);
  void Monopole(ParameterInput *pin, const bool restart);
  void MRI3d(ParameterInput *pin, const bool restart);
  void OrszagTang(ParameterInput *pin, const bool restart);
  void ShockTube(ParameterInput *pin, const bool restart);
  void Shwave(ParameterInput *pin, const bool restart);
  void SphericalCollapse(ParameterInput *pin, const bool restart);
  void RadiationLinearWave(ParameterInput *pin, const bool restart);
  void RadiationBeam(ParameterInput *pin, const bool restart);
  void Z4cBoostedPuncture(ParameterInput *pin, const bool restart);
  void Z4cLinearWave(ParameterInput *pin, const bool restart);
  void SelfGravity(ParameterInput *pin, const bool restart);
  void BinaryGravity(ParameterInput *pin, const bool restart);
  void BECollapse(ParameterInput *pin, const bool restart);
  void PolytropicStar(ParameterInput *pin, const bool restart);
  void RemapTest(ParameterInput *pin, const bool restart);

  // predefined problem generator functions for unit tests
  void EOSCompose(ParameterInput *pin, const bool restart);
  void GaussLegendre(ParameterInput *pin, const bool restart);
  void HydroPLMUnit(ParameterInput *pin, const bool restart);

  // Generic error output function (using difference u0-u1)
  void OutputErrors(ParameterInput *pin, Mesh *pm);

  // template for user-specified problem generator
  void UserProblem(ParameterInput *pin, const bool restart);

 private:
  bool single_file_per_rank; // for restart file naming
  Mesh* pmy_mesh_;
};

namespace problem_runtime {

//! \brief Where the black holes are and how big their horizons are, at a given time.
//! Published by problem generators that carry a black hole (analytically or on a
//! trajectory) and consumed by physics that needs a distance to the nearest hole on
//! device -- the M1 radiative shear viscosity, whose mean free path is limited by it
//! (KORAL RADVISCMFPSPH), the M1 direct-field anchors and the hybrid force-free horizon
//! test.  nbh == 0 means "not published", and every consumer must have a black-hole-free
//! fallback.
struct BlackHoleHorizonState {
  int nbh = 0;
  Real x[2] = {0.0, 0.0};
  Real y[2] = {0.0, 0.0};
  Real z[2] = {0.0, 0.0};
  Real rh[2] = {0.0, 0.0};   // horizon radius r_H = M + sqrt(M^2 - a^2)
  // Coordinate velocity v the metric boosts the hole by, and cf = (gamma - 1)/v^2; a lab
  // displacement d from the hole maps to its rest frame as d + cf v (v.d), where rh is a
  // radius.  All zero for a hole at rest or a publisher that does not move it.
  Real vx[2] = {0.0, 0.0};
  Real vy[2] = {0.0, 0.0};
  Real vz[2] = {0.0, 0.0};
  Real cf[2] = {0.0, 0.0};
  // Kerr-Schild radius of the rest-frame displacement at which the publisher's excision
  // masks cut the hole under <coord>/excise (BBH.cpp cuts the heavier hole inside its
  // horizon); 0 for a hole it does not excise.  With gmax, the largest Lorentz factor
  // the publisher boosts the hole by over the whole run (1 for a hole at rest; constant
  // for the run), it places the inner radius of the M1 direct-field tables
  // (radiation_m1_direct.cpp, TableInnerRadius).
  Real rexc[2] = {0.0, 0.0};
  Real gmax[2] = {1.0, 1.0};
};

using BlackHoleHorizonAtTimeFnPtr = BlackHoleHorizonState (*)(Real time);

KOKKOS_INLINE_FUNCTION
bool InsideExcisionZone(const Real x, const Real y, const Real z,
                        const Real center_x, const Real center_y, const Real center_z,
                        const Real radius2) {
  Real dx = x - center_x;
  Real dy = y - center_y;
  Real dz = z - center_z;
  return (dx*dx + dy*dy + dz*dz) <= radius2;
}

void SetExcisionState(bool enabled, Real radius, Real density, Real eint);
void SetLiveBHState(bool orbit_enabled,
                    Real bhx, Real bhy, Real bhz,
                    Real bhvx, Real bhvy, Real bhvz,
                    Real framex, Real framey, Real framez,
                    Real framevx, Real framevy, Real framevz);
void SetFrameStepState(bool valid, Real step_time, Real step_dt,
                       Real x0, Real y0, Real z0,
                       Real vx0, Real vy0, Real vz0,
                       Real ax0, Real ay0, Real az0,
                       Real x1, Real y1, Real z1,
                       Real vx1, Real vy1, Real vz1,
                       Real ax1, Real ay1, Real az1);
void SetBlackHoleHorizonAtTimeFunction(BlackHoleHorizonAtTimeFnPtr func);
BlackHoleHorizonState GetBlackHoleHorizonAtTime(Real time);
void SetExternalBHPotential(bool enabled, Real mass, Real softening, Real newton_g);
void SetExternalBHGravitySourceCoupling(bool enabled);
void SetBHSinkGravityMask(bool enabled, Real radius);
void SetBHStepState(bool valid, Real step_time, Real step_dt,
                    Real x0, Real y0, Real z0,
                    Real vx0, Real vy0, Real vz0,
                    Real x1, Real y1, Real z1,
                    Real vx1, Real vy1, Real vz1);
void SetHydroStageTime(Real t);
void ClearHydroStageTime();
bool HasHydroStageTime();
Real HydroStageTimeOr(Real t);
bool HasExternalBHPotential();
bool ExternalBHGravitySourceCouplingEnabled();
void GetExternalBHPotential(const Real t, bool &enabled,
                            Real &bhx, Real &bhy, Real &bhz,
                            Real &mass, Real &softening, Real &newton_g);
void GetExternalBHPotentialAtExactTime(const Real t, bool &enabled,
                                       Real &bhx, Real &bhy, Real &bhz,
                                       Real &mass, Real &softening, Real &newton_g);
void GetBHSinkGravityMask(const Real t, bool &enabled, Real &radius,
                          Real &center_x, Real &center_y, Real &center_z);
KOKKOS_INLINE_FUNCTION
Real ExternalBHPotential(const Real x, const Real y, const Real z,
                         const Real bhx, const Real bhy, const Real bhz,
                         const Real mass, const Real softening,
                         const Real newton_g) {
  return bh_force_pair::Potential(x, y, z, bhx, bhy, bhz, mass, softening, newton_g);
}

KOKKOS_INLINE_FUNCTION
bool InsideBHSinkGravityMask(const Real x, const Real y, const Real z,
                             const Real center_x, const Real center_y,
                             const Real center_z, const Real radius) {
  if (radius <= 0.0) return false;
  const Real dx = x - center_x;
  const Real dy = y - center_y;
  const Real dz = z - center_z;
  return (dx*dx + dy*dy + dz*dz) <= radius*radius;
}

KOKKOS_INLINE_FUNCTION
Real ExternalBHPotentialSinkMasked(const Real x, const Real y, const Real z,
                                   const Real bhx, const Real bhy,
                                   const Real bhz, const Real mass,
                                   const Real softening,
                                   const Real newton_g,
                                   const Real sink_radius) {
  return bh_force_pair::Potential(x, y, z, bhx, bhy, bhz, mass, softening,
                                  newton_g, sink_radius);
}
void GetExcisionState(const Real t, bool &enabled, Real &radius, Real &density,
    Real &eint,
                      Real &center_x, Real &center_y, Real &center_z);
void StoreRuntimeMetadata(ParameterInput *pin);

}  // namespace problem_runtime

#endif // PGEN_PGEN_HPP_
