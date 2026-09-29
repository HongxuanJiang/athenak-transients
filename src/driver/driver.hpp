#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.hpp
//  \brief definitions for Driver class
//
// Note ProblemGenerator object is stored in Driver and is called in Initialize(). If the
// pgen class contains analysis routines that are run at end of execution, they can be
// called in Finalize().

#include <ctime>
#include <memory>
#include <string>

#include "parameter_input.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

// Forward declarations
class MeshBlockPack;
class SourceTerms;

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(ParameterInput *pin, Mesh *pmesh, Real wtlim, Kokkos::Timer* ptimer);
  ~Driver() = default;

  // data
  TimeEvolution time_evolution;
  DvceArray6D<Real> impl_src;  // stiff source terms used in ImEx integrators

  // folowing data only relevant for runs involving time evolution
  Real tlim;      // stopping time
  int nlim;       // cycle-limit
  int ndiag;      // cycles between output of diagnostic information
  // variables for various SSP and ImEx RK integrators
  std::string integrator;          // integrator name (rk1, rk2, rk3)
  // True when the explicit tableau is identical to SSPRK2 (rk2 and imex2).  Every LAT
  // weight, dense-output gate, and start-register cadence keys off the explicit tableau
  // alone, so those sites must accept both names or imex2 silently degrades them.
  bool integrator_rk2_equiv{false};
  int nimp_stages;                 // number of implicit stages (ImEx only)
  int nexp_stages;                 // number of explicit stages (both SSP-RK and ImEx)
  Real gam0[4], gam1[4], beta[4];  // weights and fractional timestep per explicit stage
  Real stage_time_frac[4];          // physical RHS time fraction for each explicit stage
  Real delta[4];                   // weights for updating the intermediate stage (u1)
  Real a_twid[4][4], a_impl;       // matrix elements for implicit stages in ImEx
  Real cfl_limit;                  // maximum CFL number for integrator
  Real gamma;                      // gamma value for the IMEX_new integrator
  Kokkos::Timer* pwall_clock_;     // timer for tracking the wall clock
  Real wall_time;
  // Wall clock Initialize read to decide whether a restart steps (-1: not read).  Execute
  // starts from the same reading, so the pass it depends on runs iff a step follows.
  Real initialize_wall_clock_{-1.0};
  bool hydro_subcycle{false};
  int hydro_subcycle_factor{1};
  bool hydro_lat{false};
  int hydro_lat_levels{1};
  bool hydro_lat_union_stage1{false};
  bool hydro_lat_union_stage1_active{false};
  bool hydro_lat_defer_final_exchange{false};
  int hydro_lat_active_factor_this_bin{1};
  bool hydro_lat_skip_final_exchange_this_bin{false};
  // The deferral's correctness conditions hold, whether or not it was requested.
  bool hydro_lat_defer_final_exchange_eligible{false};
  // This bin's final-stage ghosts are superseded by the next refresh (see
  // ConfigureHydroLATSubstep); set whenever the deferral is eligible, on both paths.
  bool hydro_lat_final_ghosts_superseded_this_bin{false};
  bool hydro_lat_exchange_time_set{false};
  Real hydro_lat_exchange_time{0.0};
  // Sink particles under LAT (design N13; the module-side contract is the LAT section of
  // sink_particles.hpp).  SinkParticles::SinkStep is registered in the
  // "after_timeintegrator" list (review A2/F24: two comments used to say "before"; the
  // registration is sink_particles_tasks.cpp:44 and before/after decides whether the
  // operator sees pre- or post-integrator gas), which Driver::Execute runs once per LAT
  // bin, only on the ranks that own a block in that bin, and with pmesh->dt equal to
  // that bin's dt.
  // The sink operator needs the opposite on all three counts: every rank must enter it
  // (it runs MPI collectives), it must see one globally time-synchronized gas state, and
  // its dt must be the interval actually advanced.  Under LAT the task therefore turns
  // itself into a no-op (amendment A3 -- it still exists and still completes, so ranks
  // whose blocks are all inactive cannot desynchronize the list) and the driver invokes
  // SinkStep at the synchronized window boundary with SinkParticles::kLATSyncStage,
  // declaring the interval through SinkParticles::SetStepInterval.
  bool lat_sink_driver_cadence{false};
  // Physical time advanced since the last driver-invoked SinkStep; the interval declared
  // to the next one.  Accumulated per completed tick, so window truncation by tlim, nlim
  // or the wall clock is carried exactly instead of assumed to be sync_factor*fine_dt.
  Real lat_sink_pending_dt{0.0};

  // functions
  void ExecuteTaskList(Mesh *pm, std::string tl, int stage);
  int ConfigureHydroLATSubstep(Mesh *pm, int substep, int active_factor,
                               int sync_factor, Real base_dt,
                               bool preserve_union_times = false);
  void ClearHydroLAT(Mesh *pm);
  // SourceTerms of the fluid that carries the gravity source: hydro when it exists,
  // otherwise MHD.  This is the same choice MGGravityDriver::Solve makes when it loads
  // the Poisson source (src/gravity/mg_gravity.cpp:717-730) and the one Mesh already
  // makes for the LAT per-block source timestep (src/mesh/mesh.cpp:903), so every
  // gravity-under-LAT rule below is stated once for whichever fluid is active.
  static SourceTerms *ActiveFluidSourceTerms(MeshBlockPack *pmbp);
  void PrepareRankPackedBoundaryMetadata(Mesh *pm);
  void CheckLATUnionStepDt(Mesh *pm);
  void RefreshHydroLATBoundaries(Mesh *pm, Real target_time,
                                 bool apply_state_fixup = true,
                                 bool force_c2p = false);
  void Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool rflag);
  void Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout);
  void InitBoundaryValuesAndPrimitives(Mesh *pm, bool repair_amr_fc=false,
                                       bool restart_verbatim=false);

 private:
  Kokkos::Timer run_time_;      // generalized timer for cpu/gpu/etc
  std::uint64_t nmb_updated_;   // running total of MB updated during run
  std::uint64_t npart_updated_; // running total of particles updated during run
  float lb_efficiency_;         // measure of how efficient was load balancing
  bool RebalanceHydroLATMesh(Mesh *pm, ParameterInput *pin);
  void RebuildHydroLATMetadata(Mesh *pm);
  void OutputCycleDiagnostics(Mesh *pm);
  Real UpdateWallClock();
};
#endif // DRIVER_DRIVER_HPP_
