#ifndef SINK_PARTICLES_SINK_PARTICLES_HPP_
#define SINK_PARTICLES_SINK_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles.hpp
//! \brief definitions for the SinkParticles class: a port of ORION2-SHAO's Krumholz-style
//! sink particles (creation / Bondi-Hoyle accretion / FOF merging / direct gravity),
//! cleaned per deviation list D-e.
//!
//! The sink list is REPLICATED: every rank holds an identical std::vector<SinkData> and
//! every host-side stage (N-body, merge, id assignment) is executed redundantly on all
//! ranks from identical inputs so no broadcast is ever needed.  Only the grid sums are
//! rank-local, and those are combined with batched MPI_Allreduce calls (one per step
//! part, never one per sink).

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

// forward declarations
class MeshBlockPack;
class Driver;

//----------------------------------------------------------------------------------------
//! \struct SinkParticlesTaskIDs
//! \brief container to hold TaskIDs of all sink-particle tasks

struct SinkParticlesTaskIDs {
  TaskID step;
};

namespace sinkparticles {

//----------------------------------------------------------------------------------------
//! \struct SinkData
//! \brief POD state of a single sink.  `mom` is momentum (M*v), not velocity, and
//! `angmom` is the accumulated spin about the sink itself (ORION2 SinkParticleData.H:52).
//! This layout is the restart record: keep it trivially copyable and never reorder it
//! without bumping the restart format.

struct SinkData {
  long int id;
  Real m;
  Real pos[3];
  Real mom[3];
  Real angmom[3];
};

//----------------------------------------------------------------------------------------
//! \struct SinkMeshGeom
//! \brief mesh geometry snapshot used by both host and device code for one cycle.
//! `dxf*` are FINEST-LEVEL spacings and are recomputed every cycle because AMR changes
//! the finest present level; they set the softening length, the FOF link length, the
//! accretion-kernel cell units and the sink timestep.

struct SinkMeshGeom {
  // actual mesh bounds; NOT the origin-at-zero shortcut of ORION2 (bug R4)
  Real xmin[3], xmax[3], len[3];
  Real dxf[3];                    // cell size on the finest level present in the mesh
  Real dxf_max, dxf_min, dxf_mean;  // dxf_mean = sqrt((dx1^2+dx2^2+dx3^2)/3)
  bool periodic[3];
  int max_level;
};

//----------------------------------------------------------------------------------------
//! \struct SinkDiagnostics
//! \brief per-cycle conservation ledger.  ORION2 applies its floors silently (map R17);
//! here every activation is counted and reported so a nonzero value is visible as the
//! physics bug it is.  The last three entries are not floors but the terms of the
//! gas<->sink ledger that no state variable can hold, so that they are MEASURED instead
//! of leaving an unattributable drift in the history file (see WriteLog's "# ledger"
//! line).  Zero-initialize with `SinkDiagnostics{}` so adding a term cannot silently
//! leave one uninitialized.

struct SinkDiagnostics {
  Real n_rho_floor;    // cells whose density hit dfloor after accretion
  Real n_eint_floor;   // cells whose internal energy hit the EOS floor (energy leak)
  Real n_cell_cap;     // cells where the per-cell accretion cap bound
  Real n_coarse_cell;  // kernel cells not on the finest level (ORION2 skips such sinks)
  Real dm_leak;        // mass manufactured by the density floor
  Real de_leak;        // energy manufactured by the internal-energy floor
  Real dl_leak;        // |angular momentum| discarded by the Keplerian (r_angmom) cap
  Real de_gas;         // gas total energy removed by accretion + Jeans creation.  A sink
                       // has no energy member, so this leaves the total-energy ledger
                       // entirely; it is the accretion channel's missing term.
  Real e_sink_gas;     // sum over cells of rho*dV*phi_sink at the step entry, i.e. the
                       // sink<->gas interaction energy.  E_gas + sum_p |p|^2/(2m) + this
                       // is what the gravitational channel conserves; without it the
                       // source term's work on the gas has no counterpart to check.
};

//----------------------------------------------------------------------------------------
// Floors / eps registry (design N10 + ORION2 map §12.4).  Absolute constants; the gas
// density and internal-energy floors come from the EOS instead (eos_data.dfloor and
// EOS_Data::HydroInternalEnergyDensityFloor) so tabulated-LTE runs floor consistently.

namespace sinkfloor {
constexpr Real huge = std::numeric_limits<Real>::max();
constexpr Real tiny = 1.0e-300;        // generic divide guard
constexpr Real small_b2 = 1.0e-100;    // ORION2's MAX(b2,1e-100); dormant while b2 == 0
constexpr Real beta_max = 1.0e30;      // b2==0 short-circuit; > the 1e9 convergence gate
constexpr Real beta_converged = 1.0e9;
constexpr Real beta_tol = 1.0e-4;
constexpr int beta_max_iter = 100;
constexpr Real bondi_lambda = 1.1204222675845161;  // exp(3/2)/4
constexpr Real lee14_nb = 1.0;
constexpr Real lee14_betab = 19.4;
constexpr Real accrete_rad_min = 1.0;  // kernel width clamp, in cells
constexpr Real accrete_rad_max = 2.0;
constexpr Real maxaccfac = 1.0;
constexpr int subcell_ndiv = 8;        // ORION2 NDIV; 8^3 sub-cell quadrature
}  // namespace sinkfloor

//----------------------------------------------------------------------------------------
//! \fn SinkNint
//! \brief Fortran NINT (round half away from zero), used to snap cell-offset distances to
//! the nearest half cell.  ORION2 SINKPARTICLE_3D.F:246-251 does this so the accretion
//! stencil is exactly symmetric about a sink sitting on a cell centre or corner.

KOKKOS_INLINE_FUNCTION
Real SinkNint(const Real x) {
  return (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

//----------------------------------------------------------------------------------------
//! \fn SinkWrapDelta
//! \brief minimum-image separation along one direction.  The wrap uses the ACTUAL mesh
//! bounds through `len`; ORION2 assumed the domain starts at x=0 (bug R4).

KOKKOS_INLINE_FUNCTION
Real SinkWrapDelta(Real d, const Real len, const bool periodic) {
  if (periodic && len > 0.0) {
    if (d >  0.5*len) { d -= len; }
    if (d < -0.5*len) { d += len; }
  }
  return d;
}

//----------------------------------------------------------------------------------------
//! \fn SinkMinImage
//! \brief branchless minimum-image separation along one axis, the form the gravitational
//! PAIR force uses on both of its sides.  Kept distinct from SinkWrapDelta (which the
//! accretion kernel and the FOF linking use) so that the one function the pair force is
//! built from is textually single-sourced: the two agree everywhere except exactly at
//! |d| = len/2, where they select the opposite -- and equivalent -- periodic image.

KOKKOS_INLINE_FUNCTION
Real SinkMinImage(const Real d, const Real len, const bool periodic) {
  if (!periodic || !(len > 0.0)) { return d; }
  // floor(x+0.5) rather than round(): available in every Kokkos device backend.
  return d - len*floor(d/len + 0.5);
}

//----------------------------------------------------------------------------------------
//! \fn SinkPotentialOne
//! \brief softened gravitational potential at (x,y,z) of a SINGLE sink of mass parameter
//! `gm = G*M` at (px,py,pz) with squared softening `s2`.  This is the exact potential of
//! the module's softening law `a = G M rhat/(r^2 + s^2)` -- it is NOT a Plummer sphere,
//! and it is finite at r = 0.
//!
//! SINGLE SOURCE OF TRUTH FOR THE SINK<->GAS PAIR FORCE.  Both directions are built from
//! differences of this one function, evaluated at the same points:
//!
//!   sink -> gas   SourceTerms::SinkGravity pushes each cell with the centred difference
//!                 -(phi(x+dx) - phi(x-dx))/(2 dx), direction by direction, and contracts
//!                 the same differences with the Godunov mass fluxes for the work term
//!                 (Mullen, Hanawa & Gammie 2020).  That form is what keeps the energy
//!                 update consistent with the momentum update, so it is the one kept.
//!   gas -> sink   SinkParticles::GasSinkKick accumulates the exact NEGATIVE of that same
//!                 per-cell force.
//!
//! Consequently the pair force is equal and opposite CELL BY CELL, not merely to O(dx^2).
//! The previous code used the analytic force `G M rhat/(r^2+s^2)` on the sink side; the
//! resulting mismatch turned the grid self-force of the sink's own induced atmosphere
//! into MANUFACTURED total momentum rather than an internal exchange (finding D-8/D-3,
//! triaged in tst/analysis/SINK_TESTS.md).  Two properties of the potential-difference
//! form that the analytic sum does not have, and that motivate keeping it on both sides:
//!   * a uniform medium on a periodic mesh exerts EXACTLY zero force on the sink at any
//!     sink position (the sum of phi over the shifted lattice telescopes), whereas the
//!     analytic direct sum leaves a spurious restoring term ~ -2.2e-3 * displacement in
//!     the bondi configuration;
//!   * it vanishes exactly at every lattice symmetry point, whereas the analytic sum
//!     retains a residual (measured 1.06e-4 at a cell centre in the bondi configuration)
//!     from the minimum-image truncation of the periodic sum.

KOKKOS_INLINE_FUNCTION
Real SinkPotentialOne(const Real gm, const Real px, const Real py, const Real pz,
                      const Real s2, const Real x, const Real y, const Real z,
                      const Real lx1, const Real lx2, const Real lx3,
                      const bool per1, const bool per2, const bool per3) {
  const Real dx = SinkMinImage(x - px, lx1, per1);
  const Real dy = SinkMinImage(y - py, lx2, per2);
  const Real dz = SinkMinImage(z - pz, lx3, per3);
  const Real r2 = dx*dx + dy*dy + dz*dz;
  if (s2 > 0.0) {
    const Real s = sqrt(s2);
    return -gm/s*atan(s/fmax(sqrt(r2), sinkfloor::tiny));
  }
  return -gm/fmax(sqrt(r2), sinkfloor::tiny);
}

//----------------------------------------------------------------------------------------
//! \fn SinkPotentialSum
//! \brief SinkPotentialOne summed over the replicated list mirror `sink_gm_pos`, row
//! layout {G*M, x, y, z, soften_len2}.  This is the potential SourceTerms::SinkGravity
//! differences for the momentum kick and the mass-flux work, and the one the LAT reflux
//! (Hydro::ApplyLATFluxCorrection) differences to settle the work of a replaced face
//! flux; both must evaluate the same sum in the same order.

KOKKOS_INLINE_FUNCTION
Real SinkPotentialSum(const DvceArray2D<Real> &sink, const int nsinks,
                      const Real x, const Real y, const Real z,
                      const Real lx1, const Real lx2, const Real lx3,
                      const bool per1, const bool per2, const bool per3) {
  Real phi = 0.0;
  for (int p=0; p<nsinks; ++p) {
    phi += SinkPotentialOne(sink(p,0), sink(p,1), sink(p,2), sink(p,3), sink(p,4),
                            x, y, z, lx1, lx2, lx3, per1, per2, per3);
  }
  return phi;
}

//----------------------------------------------------------------------------------------
//! \class SinkParticles
//! \brief globally replicated Krumholz-style sink particles (hydro only).
//!
//! LAT (phase S2) -- the whole driver-facing contract, read this before wiring S2B:
//!
//! 1. PINNING.  Every block whose bounding box intersects a sink's accretion sphere runs
//!    at factor 1 (design N13).  `LATPinRegions()` publishes one {x, y, z, r} sphere per
//!    sink for `Mesh::UpdateHydroLATMetadata` to cap with; the list is identical on every
//!    rank (the sink list is replicated and the geometry comes from the replicated tree),
//!    so consuming it needs no collective.  `r` already carries the `lat_pin_safety`
//!    margin on the kernel support; the ghost-band buffer of N13 is the CONSUMER's,
//!    because only the consumer knows each block's cell size (mesh.cpp grows each
//!    candidate box by ng of ITS OWN cells -- review A2/F4(a); it used to grow it by a
//!    whole block width, which pinned every coarse block near a sink).
//!
//! 2. CADENCE.  Without LAT, SinkStep runs once per cycle from "after_timeintegrator"
//!    (ORION2's operator order: advance MHD, then sinks -- sink_particles_tasks.cpp:44)
//!    with dt = pmesh->dt.  Under LAT that task list is executed once per BIN per tick
//!    with a rank-dependent active list -- exactly where this module's collectives may
//!    not run -- so the driver must invoke SinkStep itself at a synchronized point,
//!    unconditionally on every rank, and declare the interval that call covers:
//!
//!        psink->SetStepInterval(window_dt);   // e.g. lat_sync_factor*fine_dt
//!        (void) psink->SinkStep(pdrive, SinkParticles::kLATSyncStage);  // every rank
//!
//!    The override is ONE-SHOT: SinkStep consumes it and the next call falls back to
//!    pmesh->dt, so a missed SetStepInterval degrades to the ordinary cadence instead of
//!    silently reusing a stale window length.  The task-list entry must still exist and
//!    still be executed under LAT (amendment A3); it is a masked NO-OP there, keyed on
//!    `pdrive->hydro_lat`, never an unenqueued task.
//!
//! 3. SYNCHRONIZATION SCOPE.  The second argument of SetStepInterval says whether EVERY
//!    block is at the current mesh time at the moment of the call (true at a window
//!    boundary, false at an intra-window fine-tick boundary where only factor-1 bins have
//!    just closed a step).  It is `true` by default, which is both the ordinary non-LAT
//!    situation and the recommended LAT cadence: GasSinkKick and the accretion ambient
//!    shell READ gas from every local block, and at a fine-tick call the factor > 1
//!    blocks hold a state from an earlier time.  Passing `false` is supported -- the
//!    pinned blocks the accretion kernel WRITES are then verified to be factor 1 and the
//!    stale-read caveat is reported once -- but it also makes Jeans creation illegal
//!    unless every finest-level block is pinned, which `CheckLATInvariants` enforces.
//!
//! 4. TIMESTEP.  `dtnew` = dx_finest/|v| is computed from the replicated list on every
//!    rank at the end of SinkStep and folded into `Mesh::NewTimeStep` at the driver's
//!    synchronized recompute sites.  Because the sink neighbourhood is pinned to factor
//!    1, that global fold IS the per-block constraint for the only blocks the sinks
//!    touch; no `dtnew_eachmb` term is needed and none is provided.

class SinkParticles {
 public:
  SinkParticles(MeshBlockPack *ppack, ParameterInput *pin);
  ~SinkParticles();

  // data
  MeshBlockPack *pmy_pack;
  SinkParticlesTaskIDs id;

  // input parameters (block <sink_particles>)
  bool create;                // enable Jeans creation (requires <gravity> self-gravity)
  bool accrete;               // enable Bondi-Hoyle accretion
  Real jeans_no;              // Truelove number, default 0.25
  int accrete_radius_cells;   // ngrow; also the kernel support radius (fix R3)
  Real soften;                // softening in finest cells: a = GM/(r^2+(soften*dx_f)^2).
                              // 0 is legal (SinkPotentialOne has an unsoftened branch);
                              // negative is refused by the constructor.
  Real merge_link_cells;      // FOF link length in finest cells (fix R19)
  Real mmergemax;             // two sinks above this mass never merge (0 = always merge)
  Real r_angmom;              // Keplerian cap radius; 1e100 = uncapped
  bool use_sink_timestep;     // fold dx_finest/|v| into the global dt
  // How many finest cells a sink may traverse within ONE LAT window, in units of the
  // sink timestep dx_finest/|v| (review A2/F4(b)).  1.0 = the historical behaviour and
  // therefore bit-for-bit identical by default.
  //
  // The driver bounds the LAT window by cfl_no*dtnew*lat_window_motion_cells
  // (driver.cpp, "Design N13" block).  With the default that gives
  // lat_sync_factor <= 1 + c_s/|v_sink|, i.e. exactly 1 for any SUPERSONIC sink -- the
  // TDE / star-BH case -- so the window collapses and LAT is switched off by the mere
  // presence of a fast sink.  Raising this key trades N-body/kick accuracy inside one
  // window for the window itself; the pin spheres stay valid up to accrete_radius_cells
  // finest cells of drift (lat_pin_safety = 2), which is the hard cap enforced below.
  Real lat_window_motion_cells;
  bool subcell_gravity;       // 8^3 sub-cell quadrature in the 27-cell near block
  Real bs_tol;                // Bulirsch-Stoer tolerance (ORION2 eps = 1e-6)
  std::string log_file;       // rank-0 per-cycle sink log ("" disables)
  // Fielding (2015) only; ORION2's angmom_method=0 is dimensionally broken and is not
  // ported (design D-e / map R5).
  static constexpr int angmom_method = 1;
  // Margin applied to the accretion-kernel support when publishing the LAT pin spheres.
  // The kernel itself reaches (accrete_radius_cells + 1/2) finest cells (the extra half
  // cell is the SinkNint snapping); doubling the nominal radius covers that, one cycle of
  // sink motion, and the ambient shell, without depending on the block size.
  static constexpr Real lat_pin_safety = 2.0;

  // gravitational constant in code units
  Real newton_g;

  // replicated state (identical on every rank)
  std::vector<SinkData> sinks;
  // Cached sinks.size(); kept in sync by SyncSinkCount() at every mutation point so
  // consumers outside this module (source terms, restart, outputs) can read the count
  // without depending on the std::vector.
  int nsinks;
  long int nsinks_created;    // monotone id counter, identical on every rank

  // source-term facing mirror, rebuilt by RefreshSinkGMPos() whenever the list changes:
  // row s = {G*M, x, y, z, (soften*dx_finest)^2}.  Consumed by the sink_gravity source
  // term (implementer B) so that the srcterm kernel needs no host state.
  DualArray2D<Real> sink_gm_pos;

  // timestep and per-cycle diagnostics
  Real dtnew;
  SinkDiagnostics diag;
  std::vector<Real> mdot;     // per-sink accretion rate of the last completed cycle
  // soften*dx_finest as PUBLISHED in column 4 of sink_gm_pos, i.e. the value the
  // sink->gas source term is using right now.  AdvanceNBody reads this rather than
  // recomputing soften*dxf_max so that a mesh whose finest level changed mid-cycle
  // cannot give the sink-sink and sink-gas forces two different softening laws within
  // one cycle; it is refreshed, with the mirror, only at the END of SinkStep.
  Real soften_len;

  // functions
  void AssembleSinkTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SinkStep(Driver *pdrive, int stage);

  // per-cycle stages, in ORION2's order (map §6.2); public so tests can drive them
  void GasSinkKick(Real dt, const SinkMeshGeom &g);
  void AdvanceNBody(Real dt, const SinkMeshGeom &g);
  void CreateSinks(const SinkMeshGeom &g);
  void MergeSinks(const SinkMeshGeom &g);
  void Accrete(Real dt, const SinkMeshGeom &g);

  // helpers
  SinkMeshGeom MeshGeometry() const;
  void RefreshSinkGMPos(const SinkMeshGeom &g);
  void UpdateTimeStep(const SinkMeshGeom &g);
  void WrapPosition(SinkData *s, const SinkMeshGeom &g) const;
  void WriteLog(Real time, int ncycle);

  // ------------------------------------------------------------------- LAT (phase S2)
  //! one {x, y, z, r} sphere per sink, r = lat_pin_safety*accrete_radius_cells*dx_finest.
  //! Rebuilt by RefreshSinkGMPos (construction, restart unpack and both ends of every
  //! SinkStep) and, defensively, whenever the finest spacing has moved under it since;
  //! it is therefore valid before the first LAT window build of a fresh or restarted run.
  //! The staleness check costs one O(nmb_total) scan, so bind the reference once per
  //! metadata build rather than calling this inside the per-gid loop.
  const std::vector<std::array<Real, 4>> &LATPinRegions() const;
  //! stage code of the driver's synchronized-point SinkStep call under LAT; a task-list
  //! call (stage >= 0) is a no-op there.  Mirrors radiation_m1's stage=-1 refresh code.
  static constexpr int kLATSyncStage = -1;
  //! declare the interval the NEXT SinkStep covers, and whether every block is at the
  //! current mesh time at that point.  One-shot: SinkStep consumes it.
  void SetStepInterval(Real dt, bool all_blocks_synchronized = true);
  void ClearStepInterval();
  //! the interval the NEXT SinkStep would use (the pending override, else pmesh->dt)
  Real StepInterval() const;
  //! the scope the NEXT SinkStep would use; true again once an override is consumed
  bool StepIntervalIsFullySynchronized() const { return step_interval_sync_; }
  //! rebuild the pin spheres against `g` (called for you by RefreshSinkGMPos)
  void RefreshLATPinRegions(const SinkMeshGeom &g);
  //! validate the S2 contract at a SinkStep entry: factor-1 pinning of every block the
  //! accretion kernel writes, and a legal cadence for Jeans creation.  No-op without LAT.
  void CheckLATInvariants(Driver *pdrive, const SinkMeshGeom &g);

  // restart: one self-describing replicated block, written by one rank and broadcast on
  // read.  RestartDataSizeForCount() lets the reader size the buffer before the list
  // exists; the block header repeats the count so Unpack can validate what it was given.
  std::size_t RestartDataSize() const;
  // `long int` because that is the width the record itself stores (review A2/F25); the
  // old int signature made UnpackRestartData cast a file-supplied long to int -- UB --
  // BEFORE the validation that was supposed to catch a bad count.
  static std::size_t RestartDataSizeForCount(long int n);   // NOLINT(runtime/int)
  //! largest sink count a restart header may claim before it is treated as a corrupt or
  //! sink-free file rather than as an allocation request (review A2/F8).
  static constexpr long int kMaxRestartSinks = 1L << 20;    // NOLINT(runtime/int)
  void PackRestartData(char *pdst) const;
  void UnpackRestartData(const char *psrc, std::size_t nbytes);

 private:
  void SyncSinkCount() { nsinks = static_cast<int>(sinks.size()); }
  void BuildLATPinRegions(const SinkMeshGeom &g) const;

  bool log_header_written_;
  // Restarts reconstruct this object, so log_header_written_ is false again and the
  // truncating open of WriteLog used to destroy the whole pre-restart history on the
  // first post-restart cycle.  Set from main.cpp's "saha_runtime/restart_active".
  bool log_append_;
  bool coarse_warned_;        // the coarse-cell-in-kernel warning fires only once

  // LAT state.  `step_interval_` is negative when no override is pending; the pair
  // (interval, scope) is set together by SetStepInterval and consumed together by
  // SinkStep, which then latches the scope in step_all_blocks_sync_ for the stages.
  Real step_interval_;
  bool step_interval_sync_;
  bool step_all_blocks_sync_;
  bool lat_stale_read_warned_;
  bool lat_pin_warned_;
  // Cached pin spheres plus the finest spacing they were built with.  `mutable` so the
  // const accessor can heal a cache that AMR has invalidated between the last SinkStep
  // and the metadata build that reads it; the rebuild is O(nsinks) and host-only.
  mutable std::vector<std::array<Real, 4>> lat_pin_regions_;
  mutable Real lat_pin_dxf_max_;
};

}  // namespace sinkparticles

#endif  // SINK_PARTICLES_SINK_PARTICLES_HPP_
