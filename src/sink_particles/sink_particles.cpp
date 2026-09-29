//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles.cpp
//! \brief SinkParticles constructor, mesh-geometry snapshot, source-term mirror, sink
//! timestep, periodic position wrap, rank-0 log and restart (de)serialization.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "units/units.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

namespace {
//----------------------------------------------------------------------------------------
//! \fn ParseRealList
//! \brief split a "a, b, c" / "a b c" input string into Reals.  Returns false unless
//! every token is one whole finite number.

bool ParseRealList(const std::string &str, std::vector<Real> *out) {
  std::string buf = str;
  for (char &c : buf) {
    if (c == ',' || c == ';') { c = ' '; }
  }
  std::istringstream iss(buf);
  std::string token;
  while (iss >> token) {
    char *end = nullptr;
    const double v = std::strtod(token.c_str(), &end);
    if (*end != '\0' || !std::isfinite(v)) return false;
    out->push_back(static_cast<Real>(v));
  }
  return true;
}
}  // namespace

//----------------------------------------------------------------------------------------
// constructor: parse <sink_particles>, install the initial sink list, resolve G

SinkParticles::SinkParticles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    sink_gm_pos("sink_gm_pos", 1, 5),
    log_header_written_(false),
    log_append_(false),
    coarse_warned_(false),
    step_interval_(-1.0),
    step_interval_sync_(true),
    step_all_blocks_sync_(true),
    lat_stale_read_warned_(false),
    lat_pin_warned_(false),
    lat_pin_dxf_max_(-1.0) {
  Mesh *pm = pmy_pack->pmesh;

  // (1) module preconditions.  This phase is hydro-only; the LAT admissibility guard is
  // owned by the driver, the MHD refusal belongs here because every kernel below assumes
  // b2 == 0.
  if (pmy_pack->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles> requires a <hydro> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->DoesBlockExist("mhd")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles> does not support MHD in this phase" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pm->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles> requires a 3D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Every kernel in this module reads u0(IDN/IM/IEN) as the NEWTONIAN conserved set.
  // Under SR those slots hold (D, S_i, tau) = (rho*W, rho*h*W^2*v_i, ...), so
  // `IM/IDN` is not a velocity and `(rho_old - rho_new)*vol` is not the accreted mass;
  // under a dynamical metric they are additionally densitized by sqrt(-g), which
  // vol = dx1*dx2*dx3 does not carry.  The source term likewise adds a Newtonian
  // rho*grad(phi) impulse to a relativistic momentum density.  None of that announces
  // itself at runtime -- the numbers stay plausible -- so refuse the combination here.
  // The predicates mirror Coordinates' own (coordinates.cpp), including the rule that
  // <adm>/<z4c> beside a fluid block means dynamical GR.
  {
    const bool dyn_gr = (pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c")) &&
                        (pin->DoesBlockExist("hydro") || pin->DoesBlockExist("mhd"));
    const bool sr = pin->DoesParameterExist("coord", "special_rel") &&
                    pin->GetBoolean("coord", "special_rel");
    const bool gr = pin->DoesParameterExist("coord", "general_rel") &&
                    pin->GetBoolean("coord", "general_rel");
    if (dyn_gr || sr || gr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<sink_particles> is Newtonian-only: it reads "
                << "u0(IDN,IM,IEN) as (rho, rho*v, E), which is not what "
                << (dyn_gr ? "a dynamical <adm>/<z4c> metric" :
                    (sr ? "<coord>/special_rel" : "<coord>/general_rel"))
                << " stores there" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) parameters
  create = pin->GetOrAddBoolean("sink_particles", "create", false);
  accrete = pin->GetOrAddBoolean("sink_particles", "accrete", true);
  jeans_no = pin->GetOrAddReal("sink_particles", "jeans_no", 0.25);
  accrete_radius_cells = pin->GetOrAddInteger("sink_particles",
                                              "accrete_radius_cells", 4);
  soften = pin->GetOrAddReal("sink_particles", "soften", 1.0);
  merge_link_cells = pin->GetOrAddReal("sink_particles", "merge_link_cells",
                                       2.0*static_cast<Real>(accrete_radius_cells));
  mmergemax = pin->GetOrAddReal("sink_particles", "mmergemax", 0.0);
  r_angmom = pin->GetOrAddReal("sink_particles", "r_angmom", 1.0e100);
  use_sink_timestep = pin->GetOrAddBoolean("sink_particles", "use_sink_timestep", true);
  lat_window_motion_cells = pin->GetOrAddReal("sink_particles",
                                              "lat_window_motion_cells", 1.0);
  subcell_gravity = pin->GetOrAddBoolean("sink_particles", "subcell_gravity", false);
  bs_tol = pin->GetOrAddReal("sink_particles", "bs_tol", 1.0e-6);
  log_file = pin->GetOrAddString("sink_particles", "log", "");
  // main.cpp publishes the -r flag here before any physics is constructed; WriteLog uses
  // it to append instead of truncate, so a restart no longer destroys the pre-restart
  // accretion history the first time it writes.
  log_append_ = pin->DoesParameterExist("saha_runtime", "restart_active") &&
                pin->GetBoolean("saha_runtime", "restart_active");

  // Review A2/F3: 1 was accepted and silently produced ZERO accretion.  ComputeBondiRate
  // sets alpha_scale = ngrow-1 (the ORION2 kernel-radius convention,
  // sink_particles_accrete.cpp:160), so ngrow == 1 gives alpha_scale == 0 =>
  // BondiAlpha(0) = +inf => rho_inf = 0 => mdot_b = 0 => dm = 0 with ok = true, and pass
  // 3 skips every cell without a warning.  Two cells is the smallest radius at which the
  // Bondi-density deconvolution is defined.
  if (accrete_radius_cells < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/accrete_radius_cells must be >= 2 (the ORION2 "
              << "alpha_scale = accrete_radius_cells - 1 kernel-radius convention "
              << "degenerates at 1 and yields exactly zero accretion)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Review A2/F4(b).  Default 1.0 reproduces the previous window bound exactly.  The
  // margin the published pin spheres carry is
  // lat_pin_safety*ngrow - (ngrow + 1/2) = ngrow - 1/2 finest cells, while the driver
  // bounds one window by cfl_no*dtnew*lat_window_motion_cells and dtnew = dxf_min/|v|,
  // so the drift over a window is cfl_no*lat_window_motion_cells finest cells.  The
  // admissible bound is therefore (ngrow - 1/2)/cfl_no, NOT ngrow: the old cap let
  // e.g. ngrow = 2 with cfl_no = 0.8 drift 1.6 cells against a 1.5-cell margin, and the
  // resulting unpinned block is only a warning at the cadence the driver actually uses
  // (and, in a release build, not even scanned for unless LAT diagnostics are on).
  const Real ngrow_r = static_cast<Real>(accrete_radius_cells);
  const Real cfl_safe = std::max(pm->cfl_no, static_cast<Real>(sinkfloor::tiny));
  const Real lwmc_max = std::min(ngrow_r, (ngrow_r - 0.5)/cfl_safe);
  if (!(lat_window_motion_cells >= 1.0) || lat_window_motion_cells > lwmc_max) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/lat_window_motion_cells must lie in [1, " << lwmc_max
              << "] (= min(accrete_radius_cells, (accrete_radius_cells - 1/2)/cfl_no) "
              << "with accrete_radius_cells = " << accrete_radius_cells << ", cfl_no = "
              << pm->cfl_no << "): beyond that a sink can leave the block set its own "
              << "accretion kernel pinned to LAT factor 1 within one window" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (jeans_no <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/jeans_no must be > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // soften == 0 is legal (SinkPotentialOne has an unsoftened 1/r branch and Derivs
  // guards the softened denominator), but a negative value is not: only its SQUARE ever
  // reaches the mirror, so it used to be silently accepted as |soften|.
  if (soften < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/soften must be >= 0 (only soften^2 is ever used, so a "
              << "negative value would silently act as its absolute value)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (bs_tol <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/bs_tol must be > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The gas->sink kick now sums the exact reaction of the per-cell force
  // SourceTerms::SinkGravity applies (see sink_particles_gravity.cpp), and that source
  // term has no sub-cell branch.  Refining the quadrature on one side only is precisely
  // ORION2 bug R9 and design N12's "BOTH force directions or NEITHER", so the flag is
  // refused rather than silently breaking the pairing it was meant to protect (open
  // item O-7).  Every shipped input already sets it false.
  if (subcell_gravity) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/subcell_gravity = true is not supported: the sink->gas"
              << " source term has no matching sub-cell quadrature, so enabling it here"
              << " would make the pair force unequal cell by cell (design N12 / R9)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Creation removes the Jeans-unstable mass excess, which is meaningless without the
  // self-gravity that produced it (ORION2 gates createSinkParticles on GRAVITY).
  if (create && !pin->DoesBlockExist("gravity")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<sink_particles>/create = true requires a <gravity> block "
              << "(self-gravity)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (3) gravitational constant in code units.  With <units> the unit system is the single
  // source of truth; otherwise fall back to the pgen convention G = four_pi_G/(4 pi)
  // used by tde_external.cpp:3482-3488.
  if (pin->DoesBlockExist("units") && pmy_pack->punit != nullptr) {
    newton_g = pmy_pack->punit->grav_constant();
  } else {
    Real g_default = -1.0;
    if (pin->DoesParameterExist("gravity", "four_pi_G")) {
      Real four_pi_g = pin->GetReal("gravity", "four_pi_G");
      if (four_pi_g > 0.0) { g_default = four_pi_g/(4.0*M_PI); }
    }
    newton_g = pin->GetOrAddReal("sink_particles", "newton_g", g_default);
  }
  if (!(newton_g > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Could not determine G in code units: set <units>, or "
              << "<gravity>/four_pi_G, or <sink_particles>/newton_g" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (4) initial sinks: n_init and "initN = m, x, y, z, px, py, pz" rows
  nsinks_created = 0;
  int n_init = pin->GetOrAddInteger("sink_particles", "n_init", 0);
  for (int n=0; n<n_init; ++n) {
    std::string key = "init" + std::to_string(n);
    std::vector<Real> vals;
    const std::string row = pin->GetString("sink_particles", key);
    if (!ParseRealList(row, &vals) || (vals.size() != 7 && vals.size() != 10)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<sink_particles>/" << key << " = '" << row
                << "' needs 7 numbers: m, x, y, z, px, py, pz, or 10 with the spin "
                << "Lx, Ly, Lz" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    SinkData s;
    s.id = nsinks_created++;
    s.m = vals[0];
    for (int d=0; d<3; ++d) {
      s.pos[d] = vals[1+d];
      s.mom[d] = vals[4+d];
      s.angmom[d] = (vals.size() >= 10) ? vals[7+d] : 0.0;
    }
    if (!(s.m > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<sink_particles>/" << key << " mass must be > 0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    sinks.push_back(s);
  }

  SyncSinkCount();
  dtnew = std::numeric_limits<Real>::max();
  soften_len = 0.0;
  mdot.assign(sinks.size(), 0.0);
  diag = SinkDiagnostics{};
  // Also builds the LAT pin spheres, so LATPinRegions() is valid before the first
  // Mesh::UpdateHydroLATMetadata of the run (which precedes the first SinkStep).
  const SinkMeshGeom ctor_geom = MeshGeometry();
  RefreshSinkGMPos(ctor_geom);
  // Seed dtnew before the first SinkStep: Driver::Initialize and the first LAT window
  // bound fold psink->dtnew, and Real::max there would leave cycle/window 0 unbounded
  // by fast initial sinks.
  UpdateTimeStep(ctor_geom);
}

//----------------------------------------------------------------------------------------
// destructor

SinkParticles::~SinkParticles() {
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::MeshGeometry
//! \brief snapshot of the mesh bounds, periodicity and finest-level spacings.  The finest
//! level is read from the (globally replicated) tree, so this needs no collective and
//! tracks AMR: it must be recomputed every cycle.

SinkMeshGeom SinkParticles::MeshGeometry() const {
  Mesh *pm = pmy_pack->pmesh;
  SinkMeshGeom g;

  g.xmin[0] = pm->mesh_size.x1min;  g.xmax[0] = pm->mesh_size.x1max;
  g.xmin[1] = pm->mesh_size.x2min;  g.xmax[1] = pm->mesh_size.x2max;
  g.xmin[2] = pm->mesh_size.x3min;  g.xmax[2] = pm->mesh_size.x3max;
  for (int d=0; d<3; ++d) { g.len[d] = g.xmax[d] - g.xmin[d]; }

  g.periodic[0] = (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
  g.periodic[1] = (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
  g.periodic[2] = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);

  int maxlev = pm->root_level;
  if (pm->lloc_eachmb != nullptr) {
    for (int b=0; b<pm->nmb_total; ++b) {
      maxlev = std::max(maxlev, pm->lloc_eachmb[b].level);
    }
  }
  g.max_level = maxlev;
  const int nref = std::max(0, maxlev - pm->root_level);
  const Real fac = 1.0/static_cast<Real>(static_cast<std::int64_t>(1) << nref);
  g.dxf[0] = pm->mesh_size.dx1*fac;
  g.dxf[1] = pm->mesh_size.dx2*fac;
  g.dxf[2] = pm->mesh_size.dx3*fac;
  g.dxf_max = std::max(g.dxf[0], std::max(g.dxf[1], g.dxf[2]));
  g.dxf_min = std::min(g.dxf[0], std::min(g.dxf[1], g.dxf[2]));
  g.dxf_mean = std::sqrt((SQR(g.dxf[0]) + SQR(g.dxf[1]) + SQR(g.dxf[2]))/3.0);
  return g;
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::RefreshSinkGMPos
//! \brief rebuild the source-term mirror {G*M, x, y, z, (soften*dx_finest)^2}.  Must be
//! called whenever the list, the sink masses/positions or the finest spacing change.
//! The LAT pin spheres depend on exactly the same inputs, so they are rebuilt here too:
//! every mutation point of the list already goes through this function.

void SinkParticles::RefreshSinkGMPos(const SinkMeshGeom &g) {
  // NOTE ON CADENCE.  This publishes the parameters BOTH pair forces then use for a whole
  // cycle, so it may only be called where the two sides agree: at construction, at
  // restart ingest, and at the END of SinkStep.  Calling it at the TOP of SinkStep -- as
  // this module used to -- overwrote soften^2 with the CURRENT finest spacing after the
  // source term had already pushed the gas all cycle with the previous one, so on any
  // AMR level change GasSinkKick integrated the reaction of a differently softened
  // potential and re-opened the D-8 manufactured-momentum channel for that cycle.
  SyncSinkCount();
  const int ns = nsinks;
  const int nrow = std::max(1, ns);
  if (static_cast<int>(sink_gm_pos.extent(0)) != nrow) {
    sink_gm_pos = DualArray2D<Real>("sink_gm_pos", nrow, 5);
  }
  soften_len = soften*g.dxf_max;
  const Real soft2 = SQR(soften_len);
  for (int s=0; s<nrow; ++s) {
    if (s < ns) {
      sink_gm_pos.h_view(s, 0) = newton_g*sinks[s].m;
      sink_gm_pos.h_view(s, 1) = sinks[s].pos[0];
      sink_gm_pos.h_view(s, 2) = sinks[s].pos[1];
      sink_gm_pos.h_view(s, 3) = sinks[s].pos[2];
    } else {
      for (int n=0; n<4; ++n) { sink_gm_pos.h_view(s, n) = 0.0; }
    }
    sink_gm_pos.h_view(s, 4) = soft2;
  }
  sink_gm_pos.template modify<HostMemSpace>();
  sink_gm_pos.template sync<DevExeSpace>();
  RefreshLATPinRegions(g);
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::RefreshLATPinRegions
//! \brief rebuild the {x, y, z, r} spheres that pin every intersecting block to LAT
//! factor 1 (design N13).  One sphere per sink, radius
//! `lat_pin_safety*accrete_radius_cells*dx_finest`: the accretion kernel measures its
//! support in FINEST cells along each direction, so `dxf_max` is the bound that makes the
//! single sphere cover the anisotropic stencil, and lat_pin_safety covers the extra half
//! cell of the SinkNint snapping plus one cycle of sink motion.
//!
//! Everything here is replicated (the sink list is, and `g` comes from the replicated
//! tree), so the result is bit-identical on every rank and the consumer needs no
//! collective to agree on the pinned set.

void SinkParticles::RefreshLATPinRegions(const SinkMeshGeom &g) {
  BuildLATPinRegions(g);
}

void SinkParticles::BuildLATPinRegions(const SinkMeshGeom &g) const {
  const Real rad = lat_pin_safety*static_cast<Real>(accrete_radius_cells)*g.dxf_max;
  lat_pin_regions_.clear();
  lat_pin_regions_.reserve(sinks.size());
  for (const auto &s : sinks) {
    lat_pin_regions_.push_back({s.pos[0], s.pos[1], s.pos[2], rad});
  }
  lat_pin_dxf_max_ = g.dxf_max;
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::LATPinRegions
//! \brief the pin spheres, healed against a finest spacing that has moved since the last
//! refresh.  The metadata build that consumes this runs after AMR and after a rebalance,
//! both of which can change `dxf_max` without the sink list changing; a stale (larger)
//! radius would only over-pin, but a stale (smaller) one would leave a block the
//! accretion kernel writes in a factor > 1 bin, so the cache is checked, not trusted.

const std::vector<std::array<Real, 4>> &SinkParticles::LATPinRegions() const {
  const SinkMeshGeom g = MeshGeometry();
  if (lat_pin_regions_.size() != sinks.size() || !(lat_pin_dxf_max_ == g.dxf_max)) {
    BuildLATPinRegions(g);
  }
  return lat_pin_regions_;
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::SetStepInterval / ClearStepInterval / StepInterval
//! \brief the one-shot cadence override the LAT driver uses (see the class comment).
//! Non-positive intervals are ignored so a driver bug cannot silently freeze the sinks;
//! the pending override is dropped by SinkStep the moment it is read.

void SinkParticles::SetStepInterval(Real dt, bool all_blocks_synchronized) {
  step_interval_ = dt;
  step_interval_sync_ = all_blocks_synchronized;
}

void SinkParticles::ClearStepInterval() {
  step_interval_ = -1.0;
  step_interval_sync_ = true;
}

Real SinkParticles::StepInterval() const {
  return (step_interval_ > 0.0) ? step_interval_ : pmy_pack->pmesh->dt;
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::CheckLATInvariants
//! \brief validate the two S2 structural invariants at a SinkStep entry.  Both are pure
//! host work over replicated metadata; the only collective is the one that makes the
//! pinning verdict rank-consistent, and SinkStep is a synchronized point, so issuing it
//! here is legal (no LAT window may contain a collective).
//!
//! (1) PINNING.  Every block whose bounding box reaches inside a sink's accretion sphere
//!     must be in the factor-1 bin.  The kernel support is (accrete_radius_cells + 1/2)
//!     finest cells -- the half cell is the SinkNint snapping -- which is strictly inside
//!     the published pin sphere, so a violation here means the driver did not consume
//!     LATPinRegions(), not that the margin is too small.  At a fully synchronized
//!     invocation an unpinned block is an accuracy defect (the gas around the sink was
//!     advanced at a coarse cadence) and is reported once; at a fine-tick invocation it
//!     is a time-mixing bug -- the kernel would write a block that is not at the current
//!     time -- and is fatal.
//!
//! (2) CREATION.  The Jeans surgery writes any finest-level block, not just the pinned
//!     ones, so at a fine-tick invocation it is legal only if every finest-level block is
//!     pinned.  Checked against the replicated factor/level arrays, so the fatal is
//!     rank-consistent without a collective.

void SinkParticles::CheckLATInvariants(Driver *pdrive, const SinkMeshGeom &g) {
  Mesh *pm = pmy_pack->pmesh;
  if (pdrive == nullptr || !pdrive->hydro_lat) { return; }
  if (!pm->hydro_lat_metadata_valid || pm->hydro_lat_factor_eachmb == nullptr) { return; }

  // (2) creation cadence.  Replicated inputs -> replicated verdict.
  if (create && !step_all_blocks_sync_ && pm->lloc_eachmb != nullptr) {
    for (int gid=0; gid<pm->hydro_lat_metadata_nmb && gid<pm->nmb_total; ++gid) {
      if (pm->lloc_eachmb[gid].level == g.max_level &&
          pm->hydro_lat_factor_eachmb[gid] > 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "<sink_particles>/create = true under LAT requires SinkStep to be "
                  << "invoked at a fully synchronized point (SetStepInterval(dt, true)), "
                  << "or every finest-level block to be pinned to factor 1: gid " << gid
                  << " is on the finest level with factor "
                  << pm->hydro_lat_factor_eachmb[gid]
                  << ", and the Jeans surgery would write it off its own time."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // (1) pinning of every block the accretion kernel writes.  In release builds the scan
  // runs only when it is load bearing (a fine-tick invocation) or when LAT diagnostics
  // are on; in debug builds it always runs.  All three predicates are rank uniform, so
  // the collective below is reached by every rank or by none.
  bool run_scan = accrete && nsinks > 0;
#ifdef NDEBUG
  run_scan = run_scan && (!step_all_blocks_sync_ || pm->hydro_lat_diagnostics);
#endif
  if (!run_scan) { return; }

  const Real rad = (static_cast<Real>(accrete_radius_cells) + 0.5)*g.dxf_max;
  const Real rad2 = SQR(rad);
  auto &mbsize = pmy_pack->pmb->mb_size;
  int worst[2] = {0, -1};   // {max offending factor, first offending gid}
  for (int m=0; m<pmy_pack->nmb_thispack; ++m) {
    const int gid = pmy_pack->gids + m;
    if (gid < 0 || gid >= pm->hydro_lat_metadata_nmb) { continue; }
    const int factor = pm->hydro_lat_factor_eachmb[gid];
    if (factor <= 1) { continue; }
    const Real bmin[3] = {mbsize.h_view(m).x1min, mbsize.h_view(m).x2min,
                          mbsize.h_view(m).x3min};
    const Real bmax[3] = {mbsize.h_view(m).x1max, mbsize.h_view(m).x2max,
                          mbsize.h_view(m).x3max};
    for (int s=0; s<nsinks; ++s) {
      Real d2 = 0.0;
      for (int d=0; d<3; ++d) {
        const Real ctr = 0.5*(bmin[d] + bmax[d]);
        const Real hlf = 0.5*(bmax[d] - bmin[d]);
        const Real sep = std::fabs(SinkWrapDelta(sinks[s].pos[d] - ctr, g.len[d],
                                                 g.periodic[d])) - hlf;
        d2 += SQR(std::max(sep, static_cast<Real>(0.0)));
      }
      if (d2 <= rad2) {
        if (factor > worst[0]) { worst[0] = factor; worst[1] = gid; }
        break;
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, worst, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (worst[0] <= 1) { return; }
  if (!step_all_blocks_sync_) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "LAT pinning is not installed around the sinks: gid " << worst[1]
              << " intersects an accretion sphere but sits in the factor-" << worst[0]
              << " bin.  Feed SinkParticles::LATPinRegions() to the "
              << "Mesh::UpdateHydroLATMetadata factor cap (design N13); the accretion "
              << "kernel would otherwise write a block that is not at the current time."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!lat_pin_warned_ && global_variable::my_rank == 0) {
    lat_pin_warned_ = true;
    std::cout << "### WARNING: LAT pinning is not installed around the sinks (gid "
              << worst[1] << " intersects an accretion sphere in the factor-" << worst[0]
              << " bin).  The accretion physics then samples gas advanced at a coarser "
              << "cadence; feed SinkParticles::LATPinRegions() to the factor cap."
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::UpdateTimeStep
//! \brief dt_sink = min_p dx_min,finest/max(|v_p|,tiny)
//! (ORION2 SinkParticleList.cpp:963).
//! No CFL factor here: Mesh::NewTimeStep applies cfl_no, matching the hydro convention.
//!
//! Under LAT this stays a single global scalar and needs no `dtnew_eachmb` companion.
//! `Mesh::NewTimeStep` folds it at the driver's synchronized recompute sites (Initialize,
//! end of window, post-rebalance), which lowers the FINE dt; the only blocks the sinks
//! touch are pinned to factor 1 and therefore step at exactly that fine dt, so the global
//! fold already IS their per-block constraint.  It is evaluated from the replicated list
//! on every rank, hence rank-consistent without a collective, and it is recomputed at the
//! END of SinkStep, i.e. from the post-kick post-N-body state, which is the state the
//! next interval starts from.

void SinkParticles::UpdateTimeStep(const SinkMeshGeom &g) {
  dtnew = std::numeric_limits<Real>::max();
  if (!use_sink_timestep) { return; }
  for (const auto &s : sinks) {
    if (!(s.m > 0.0)) { continue; }
    const Real v = std::sqrt(SQR(s.mom[0]) + SQR(s.mom[1]) + SQR(s.mom[2]))/s.m;
    dtnew = std::min(dtnew, g.dxf_min/std::max(v, static_cast<Real>(sinkfloor::tiny)));
  }
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::WrapPosition
//! \brief wrap a sink back into the domain along periodic directions using the ACTUAL
//! mesh bounds.  ORION2 assumed the domain starts at x=0 (map R4) and aborted when a
//! sink left a non-periodic domain; here leaving is reported, not fatal, because outflow
//! faces make it a legal (if unusual) outcome and the grid kernels simply find no cells.

void SinkParticles::WrapPosition(SinkData *s, const SinkMeshGeom &g) const {
  for (int d=0; d<3; ++d) {
    if (g.periodic[d] && g.len[d] > 0.0) {
      // Review A2/F10: one branchless step instead of two unbounded while loops.  A very
      // large finite (or infinite) coordinate -- reachable from a Bulirsch-Stoer step
      // that overshoots before ParticleIntegrate's failure path fires, or a malformed
      // initN row -- used to spin |pos|/len times on EVERY rank at once, i.e. an
      // effective hang.  This is exactly SinkMinImage's own form, and it is bit-identical
      // for a position already inside the domain (floor() == 0 => pos - len*0.0 == pos).
      const Real nwrap = std::floor((s->pos[d] - g.xmin[d])/g.len[d]);
      if (std::isfinite(nwrap) && nwrap != 0.0) { s->pos[d] -= g.len[d]*nwrap; }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::WriteLog
//! \brief rank-0 per-cycle ASCII log: one line per sink per cycle.

void SinkParticles::WriteLog(Real time, int ncycle) {
  if (log_file.empty() || global_variable::my_rank != 0) { return; }
  std::ofstream ofs;
  if (!log_header_written_) {
    // A restart reconstructs this object, so this branch is taken again and the
    // truncating open it used to do unconditionally destroyed the whole pre-restart
    // history on the first post-restart cycle.  Append when restarting -- unless the
    // file does not exist yet, which is the case when <sink_particles>/log is added at
    // the restart, and which still needs its header.
    bool have_history = false;
    if (log_append_) {
      std::ifstream probe(log_file.c_str(), std::ios::in | std::ios::ate);
      have_history = probe.is_open() && (probe.tellg() > 0);
    }
    ofs.open(log_file.c_str(),
             have_history ? (std::ios::out | std::ios::app)
                          : (std::ios::out | std::ios::trunc));
    if (!have_history) {
      ofs << "# AthenaK sink-particle log\n"
          << "# [1]=cycle [2]=time [3]=id [4]=m [5..7]=pos [8..10]=mom [11..13]=angmom "
          << "[14]=mdot\n"
          << "# ledger lines (comments, one per cycle) carry the gas<->sink terms that "
          << "no state variable holds:\n"
          << "# ledger <cycle> <time> de_gas <E removed from the gas by accretion and "
          << "Jeans creation>\n"
          << "#   e_sink_gas <sum rho dV phi_sink>  e_kin_sink <sum |p|^2/(2m)>  "
          << "dl_leak <|dL| dropped by r_angmom>\n"
          << "#   dm_leak <mass from the density floor>  de_leak <energy from the "
          << "internal-energy floor>\n";
    }
    log_header_written_ = true;
  } else {
    ofs.open(log_file.c_str(), std::ios::out | std::ios::app);
  }
  if (!ofs.is_open()) { return; }
  ofs << std::scientific << std::setprecision(14);
  for (int s=0; s<nsinks; ++s) {
    const SinkData &p = sinks[s];
    ofs << ncycle << " " << time << " " << p.id << " " << p.m;
    for (int d=0; d<3; ++d) { ofs << " " << p.pos[d]; }
    for (int d=0; d<3; ++d) { ofs << " " << p.mom[d]; }
    for (int d=0; d<3; ++d) { ofs << " " << p.angmom[d]; }
    ofs << " " << ((s < static_cast<int>(mdot.size())) ? mdot[s] : 0.0) << "\n";
  }
  // The ledger of the terms nothing stores, written as a COMMENT so every existing
  // parser of this file keeps working.  Accretion and creation take gas total energy
  // that no sink can hold (SinkData has no energy member), the source term does work on
  // the gas whose counterpart is the sink<->gas interaction energy, and the r_angmom cap
  // discards spin: none of that used to be visible anywhere, so an energy- or
  // angular-momentum-conservation check on a sink run failed with nothing to subtract.
  Real e_kin_sink = 0.0;
  for (const SinkData &p : sinks) {
    if (p.m > 0.0) {
      e_kin_sink += 0.5*(SQR(p.mom[0]) + SQR(p.mom[1]) + SQR(p.mom[2]))/p.m;
    }
  }
  ofs << "# ledger " << ncycle << " " << time
      << " " << diag.de_gas << " " << diag.e_sink_gas << " " << e_kin_sink
      << " " << diag.dl_leak << " " << diag.dm_leak << " " << diag.de_leak << "\n";
  ofs.close();
}

//----------------------------------------------------------------------------------------
// restart serialization.  The list is replicated, so one rank writes this block and it is
// broadcast on read; the record is [nsinks][nsinks_created][SinkData...].

std::size_t SinkParticles::RestartDataSizeForCount(long int n) {  // NOLINT(runtime/int)
  const long int nn = (n > 0) ? n : 0;                             // NOLINT(runtime/int)
  return 2*sizeof(long int) + static_cast<std::size_t>(nn)*sizeof(SinkData);
}

std::size_t SinkParticles::RestartDataSize() const {
  return RestartDataSizeForCount(static_cast<long int>(sinks.size()));
}

void SinkParticles::PackRestartData(char *pdst) const {
  const long int n = static_cast<long int>(sinks.size());
  std::memcpy(pdst, &n, sizeof(long int));
  pdst += sizeof(long int);
  std::memcpy(pdst, &nsinks_created, sizeof(long int));
  pdst += sizeof(long int);
  if (n > 0) {
    std::memcpy(pdst, sinks.data(), sinks.size()*sizeof(SinkData));
  }
}

void SinkParticles::UnpackRestartData(const char *psrc, std::size_t nbytes) {
  if (nbytes < 2*sizeof(long int)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sink-particle restart block is truncated" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  long int n = 0;
  std::memcpy(&n, psrc, sizeof(long int));
  psrc += sizeof(long int);
  std::memcpy(&nsinks_created, psrc, sizeof(long int));
  psrc += sizeof(long int);
  if (n < 0 || n > kMaxRestartSinks || nbytes != RestartDataSizeForCount(n)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sink-particle restart block size mismatch" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  sinks.resize(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(sinks.data(), psrc, sinks.size()*sizeof(SinkData));
  }
  SyncSinkCount();
  mdot.assign(sinks.size(), 0.0);
  // Rebuilds the LAT pin spheres from the restored positions.  This runs during restart
  // ingest, before the driver's first Mesh::UpdateHydroLATMetadata, so LATPinRegions() is
  // already the restored list when the first window is built.
  const SinkMeshGeom unpack_geom = MeshGeometry();
  RefreshSinkGMPos(unpack_geom);
  // Seed dtnew from the restored velocities (same reason as the constructor seed).
  UpdateTimeStep(unpack_geom);
}

}  // namespace sinkparticles
