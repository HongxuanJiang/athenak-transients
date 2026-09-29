//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.cpp
//  \brief implementation of ADM class
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c/z4c.hpp"

namespace adm {
namespace {

[[noreturn]] void ADMFatal(const char *file, int line, const std::string &message) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

char const * const ADM::ADM_names[ADM::nadm] = {
  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
ADM::ADM(MeshBlockPack *ppack, ParameterInput *pin):
    u_adm(),
    is_dynamic(false),
    callback_time_override_valid(false),
    callback_time_override(0.0),
    metric_backend(ADMMetricBackend::stored),
    SetADMVariables(&ADM::SetADMVariablesToKerrSchild),
    pmy_pack(ppack),
    bbh_trajectory_sampler(nullptr),
    bbh_configured(false),
    metric_time(0.0),
    base_metric_cache_enabled(false),
    base_cache(),
    base_cache_stamp(),
    base_cache_epoch(0),
    base_cache_topology_version(0) {
  is_dynamic = pin->GetOrAddBoolean("adm" , "dynamic", false);

  // The metric backend is NOT an input parameter.  The BBH problem generator prescribes
  // an analytical superposed-Kerr-Schild binary, which is always evaluated per point from
  // a ~1 kB host-built state; every other problem generator fills a stored ADM grid
  // because it has no closed form to evaluate.
  //
  // There is deliberately no switch, because there is no configuration in which storing
  // the BBH grid wins.  Measured on a 344-block 60^3/nghost=4 BBH LAT mesh (10x V100,
  // three repeats, non-overlapping distributions): evaluating on the fly is ~16% faster
  // in wall time AND 1428 MiB/rank lighter.  Storing loses on both counts because it has
  // to re-run a full-grid 17-component fill kernel at every metric install -- once per
  // LAT bin per tick -- and then read that grid back in every consumer.  At the full
  // production resolution the stored grid does not even fit: u_adm alone asks for 2.509
  // GiB.
  if (std::string(PROBLEM_GENERATOR) == "BBH") {
    metric_backend = ADMMetricBackend::bbh_analytic_onthefly;
    is_dynamic = true;
  } else {
    metric_backend = ADMMetricBackend::stored;
  }

  if (pin->DoesParameterExist("adm", "metric_backend")) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<adm>/metric_backend is obsolete and is being ignored. The backend is "
                << "now determined by the problem generator: BBH always evaluates its "
                << "analytical binary metric on the fly, everything else stores a grid. "
                << "Remove the parameter from the input file." << std::endl;
    }
  }

  if (IsAnalyticBBH() && pmy_pack->pz4c != nullptr) {
    ADMFatal(__FILE__, __LINE__,
             "The BBH problem generator prescribes an analytical binary metric and cannot "
             "be combined with Z4c, which would evolve a second, conflicting spacetime. "
             "Use the z4c/spectre problem generator for numerical evolution.");
  }

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // The analytical backend deliberately leaves u_adm unallocated. Its compact metric
  // state is configured by the BBH problem generator before first use.
  //
  // What it may allocate instead is the BASE-metric cache, which is a different animal
  // from the stored grid the comment above rejects.  That grid holds all 17 ADM
  // components, so filling it needs the K_ij/derivative path per cell and has to be
  // redone over the WHOLE grid at every install.  This holds only the 10 base components,
  // its fill costs exactly one base evaluation per cell, and it covers only the blocks a
  // given install is actually about to step (the LAT active list).  It exists because
  // ADMMetricView::FaceMetric evaluates both cells adjacent to a face, so a three-
  // direction sweep re-derives every cell's base metric six times per RK stage in the
  // flux kernels alone, before C2P, FOFC, Tmunu, the corner-EMF and excision add theirs.
  // K_ij and the metric derivatives are deliberately NOT cached: they have far fewer
  // consumers, and holding them would cost 7 more doubles per cell and a fill that runs
  // the dual-number path instead of the plain-Real one this cache needs.
  if (!StoresMetricGrid()) {
    base_metric_cache_enabled = pin->GetOrAddBoolean("adm", "base_metric_cache", true);
    if (base_metric_cache_enabled) {
      base_cache = DvceArray5D<Real>("adm_base_cache", nmb, nadmbasecache,
                                     ncells3, ncells2, ncells1);
      // Stamps start at a value no epoch can ever take (epochs begin at 1), so that a
      // view built before the first fill cannot report a hit on an unwritten block.
      base_cache_stamp = DvceArray1D<int>("adm_base_cache_stamp", nmb);
      Kokkos::deep_copy(base_cache_stamp, -1);
    }
    if (global_variable::my_rank == 0) {
      if (base_metric_cache_enabled) {
        const double mib = static_cast<double>(nmb)*
                           static_cast<double>(nadmbasecache)*
                           static_cast<double>(ncells3)*
                           static_cast<double>(ncells2)*
                           static_cast<double>(ncells1)*
                           static_cast<double>(sizeof(Real))/(1024.0*1024.0);
        std::cout << "### ADM base-metric cache ENABLED (<adm>/base_metric_cache): "
                  << nmb << " blocks x " << nadmbasecache << " components x "
                  << ncells3 << "x" << ncells2 << "x" << ncells1 << " cells = "
                  << mib << " MiB/rank" << std::endl;
      } else {
        std::cout << "### ADM base-metric cache DISABLED "
                  << "(<adm>/base_metric_cache=false): the analytical binary metric is "
                  << "re-evaluated at every query." << std::endl;
      }
    }
    return;
  }

  if (pin->DoesParameterExist("adm", "base_metric_cache") &&
      global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<adm>/base_metric_cache only applies to the analytical BBH backend "
              << "and is being ignored: this problem generator stores an ADM grid, "
              << "which is already the cache." << std::endl;
  }

  if (pmy_pack->pz4c == nullptr) {
    u_adm = DvceArray5D<Real>("u_adm", nmb, nadm, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  } else {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    u_adm = DvceArray5D<Real>("u_adm", nmb, nadm - 4, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_ALPHA);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_BETAX, pz4c->I_Z4C_BETAZ);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_PSI4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_GXX, I_ADM_GZZ);
  adm.vK_dd.InitWithShallowSlice(u_adm, I_ADM_KXX, I_ADM_KZZ);
}

//----------------------------------------------------------------------------------------
// destructor
ADM::~ADM() {}

//----------------------------------------------------------------------------------------
void ADM::ConfigureAnalyticBBH(const analytic_bbh::KeplerParameters &params,
                               TrajectorySampler sampler) {
  if (!IsAnalyticBBH()) return;
  bbh_params = params;
  bbh_trajectory_sampler = sampler;
  bbh_configured = true;
  SetMetricTime(pmy_pack->pmesh->time);
}

//----------------------------------------------------------------------------------------
//! Build the compact metric state for an arbitrary time from whichever orbit source the
//! problem generator registered.  This is the single point at which the orbit model
//! enters the spacetime; nothing downstream (device kernels included) is aware of it.

void ADM::BuildMetricStateAtTime(const Real time,
                                 analytic_bbh::MetricState &state) const {
  if (bbh_trajectory_sampler != nullptr) {
    Real traj_minus[analytic_bbh::kTrajectorySize];
    Real traj_now[analytic_bbh::kTrajectorySize];
    Real traj_plus[analytic_bbh::kTrajectorySize];
    bbh_trajectory_sampler(time - analytic_bbh::derivative_step, traj_minus);
    bbh_trajectory_sampler(time, traj_now);
    bbh_trajectory_sampler(time + analytic_bbh::derivative_step, traj_plus);
    analytic_bbh::BuildMetricStateFromTrajectories(time, traj_minus, traj_now, traj_plus,
                                                   bbh_params, state);
    return;
  }
  analytic_bbh::BuildKeplerMetricState(time, bbh_params, state);
}

//----------------------------------------------------------------------------------------
void ADM::SetMetricTime(const Real time) {
  metric_time = time;
  if (!IsAnalyticBBH()) return;
  if (!bbh_configured) {
    ADMFatal(__FILE__, __LINE__,
             "The analytical BBH metric was used before its parameters were "
             "configured by the BBH problem generator.");
  }
  BuildMetricStateAtTime(metric_time, bbh_metric_state);
  // Installing a new state is exactly what invalidates the cache, so it is also the only
  // place that refills it.  Note this is the single writer of bbh_metric_state, and
  // ConfigureAnalyticBBH -- the single writer of bbh_params -- ends by calling us; so the
  // (state, params) pair the fill below evaluates can never change without a refill.
  FillBaseMetricCache();
}

//----------------------------------------------------------------------------------------
//! \fn void ADM::FillBaseMetricCache
//! \brief Evaluate the base metric once per cell and store the ten doubles.
//!
//! Validity protocol.  Every fill first bumps base_cache_epoch and then stamps ONLY the
//! blocks it actually wrote.  Consequences, all of which the consumer side relies on:
//!   * a block the fill skipped (LAT-inactive, or beyond nmb_thispack) keeps an older
//!     stamp, so ADMMetricView::BaseCacheHit refuses it and the caller evaluates on the
//!     fly.  Nothing ever reads an unwritten cell;
//!   * a mesh change that reshuffles which physical block sits in slot m is covered too,
//!     because the epoch bump invalidates every slot the following fill does not rewrite,
//!     and the fill always uses the mb_size in force at fill time.  The topology version
//!     recorded here is a second, independent guard on the same hazard (see
//!     ADM::GetMetricView) that does not assume the caller reinstalls the metric after
//!     AMR -- which mesh_refinement.cpp does, but which is an ordering property, not an
//!     invariant of this class;
//!   * the values stored are produced by ADMMetricView::CellMetric itself, with the cache
//!     switched off in the local copy of the view, so a cache hit necessarily returns the
//!     same bit patterns a cache miss would have computed.  No conversion happens on the
//!     way in or out.

void ADM::FillBaseMetricCache() {
  if (!base_metric_cache_enabled) return;

  // Bump the epoch BEFORE the fill.  If the fill below covers only part of the pack, the
  // uncovered blocks are left holding the previous epoch and are therefore invalid.
  if (base_cache_epoch == std::numeric_limits<int>::max()) {
    // Unreachable in any conceivable run (one increment per metric install), but a
    // wrapped epoch could alias a stale stamp, so restart the numbering from scratch.
    Kokkos::deep_copy(base_cache_stamp, -1);
    base_cache_epoch = 0;
  }
  ++base_cache_epoch;
  base_cache_topology_version = pmy_pack->pmesh->topology_version;

  const int nwork = pmy_pack->nmb_thispack;
  if (nwork <= 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*(indcs.ng);
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // The whole ghost-extended range is filled: C2P, the ghost-band C2P, excision and the
  // face metrics all reach into the ghost zones.
  ADMMetricView eval = GetMetricView();
  eval.base_cache_valid = false;

  auto cache = base_cache;
  auto stamp = base_cache_stamp;
  const int epoch = base_cache_epoch;

  par_for("adm_base_cache_fill", DevExeSpace(), 0, nwork-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = a;
    ADMMetricPoint point{};
    eval.CellMetric(m, k, j, i, point);
    cache(m,I_BCACHE_GXX,k,j,i) = point.g_dd[S11];
    cache(m,I_BCACHE_GXY,k,j,i) = point.g_dd[S12];
    cache(m,I_BCACHE_GXZ,k,j,i) = point.g_dd[S13];
    cache(m,I_BCACHE_GYY,k,j,i) = point.g_dd[S22];
    cache(m,I_BCACHE_GYZ,k,j,i) = point.g_dd[S23];
    cache(m,I_BCACHE_GZZ,k,j,i) = point.g_dd[S33];
    cache(m,I_BCACHE_ALPHA,k,j,i) = point.alpha;
    cache(m,I_BCACHE_BETAX,k,j,i) = point.beta_u[0];
    cache(m,I_BCACHE_BETAY,k,j,i) = point.beta_u[1];
    cache(m,I_BCACHE_BETAZ,k,j,i) = point.beta_u[2];
  });

  // Publish the fill.  Kernels on the default execution space instance are ordered, so
  // the stamps become visible only after every cell above has been written.
  par_for("adm_base_cache_stamp", DevExeSpace(), 0, nwork-1,
  KOKKOS_LAMBDA(const int a) {
    stamp(a) = epoch;
  });
}

//----------------------------------------------------------------------------------------
ADMMetricView ADM::GetMetricView() const {
  if (IsAnalyticBBH() && !bbh_configured) {
    ADMFatal(__FILE__, __LINE__,
             "The analytical BBH metric was requested before configuration.");
  }
  ADMMetricView view{};
  view.backend = metric_backend;
  view.stored = adm;
  view.size = pmy_pack->pmb->mb_size;
  view.indcs = pmy_pack->pmesh->mb_indcs;
  view.bbh = bbh_params;
  view.bbh_state = bbh_metric_state;
  view.base_cache = base_cache;
  view.base_cache_stamp = base_cache_stamp;
  view.base_cache_epoch = base_cache_epoch;
  // This view evaluates bbh_metric_state, which is precisely the state the last fill
  // used, so the cache is admissible.  The topology check is the independent guard
  // against a mesh change having moved a different block into a slot the cache still
  // stamps as filled: mesh_refinement.cpp reinstalls the metric right after AMR, but it
  // then increments topology_version, so that install's stamps are discarded here and
  // the cache stays off until the next install -- which for the prescribed, dynamic BBH
  // metric is the very next RK stage.  Per-block validity is decided on the device by
  // ADMMetricView::BaseCacheHit; this flag only says the cache as a whole is admissible.
  view.base_cache_valid = base_metric_cache_enabled &&
      (pmy_pack->pmesh->topology_version == base_cache_topology_version);
  return view;
}

//----------------------------------------------------------------------------------------
ADMMetricView ADM::GetMetricView(const Real time) const {
  ADMMetricView view = GetMetricView();
  if (IsAnalyticBBH()) {
    BuildMetricStateAtTime(time, view.bbh_state);
    // The cache holds the metric of the INSTALLED state only.  Outputs, AMR refinement
    // criteria and the radiation modules legitimately ask for other times; handing them a
    // view that still reported a valid cache would silently serve them the installed
    // metric instead of the one they asked for.  Rather than compare times -- which would
    // make correctness hinge on BuildMetricStateAtTime being a pure function of `time`,
    // and it reads the problem generator's trajectory table -- compare the state itself.
    // analytic_bbh::MetricState is an aggregate of nothing but Real, so it has no padding
    // and this is an exact field-by-field comparison; both objects were value-initialized
    // before being filled in any case.  Bit-identical state (the common case where the
    // requested time IS the installed time) keeps the cache, anything else drops it.
    // Note the other input to the evaluation, bbh_params, needs no such test: its only
    // writer, ConfigureAnalyticBBH, ends by calling SetMetricTime and hence refills.
    if (view.base_cache_valid &&
        std::memcmp(&view.bbh_state, &bbh_metric_state,
                    sizeof(analytic_bbh::MetricState)) != 0) {
      view.base_cache_valid = false;
    }
  }
  return view;
}

//----------------------------------------------------------------------------------------
void ADM::SetADMVariablesToKerrSchild(MeshBlockPack *pmbp) {
  if (!pmbp->padm->StoresMetricGrid()) {
    ADMFatal(__FILE__, __LINE__,
             "<adm>/metric_backend=bbh_analytic_onthefly can only be configured by "
             "the analytical BBH problem generator.");
  }
  Real a = pmbp->pcoord->coord_data.bh_spin;
  bool minkowski = pmbp->pcoord->coord_data.is_minkowski;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputeADMDecomposition(x1v, x2v, x3v, minkowski, a,
      &adm.alpha(m,k,j,i),
      &adm.beta_u(m,0,k,j,i), &adm.beta_u(m,1,k,j,i), &adm.beta_u(m,2,k,j,i),
      &adm.psi4(m,k,j,i),
      &adm.g_dd(m,0,0,k,j,i), &adm.g_dd(m,0,1,k,j,i), &adm.g_dd(m,0,2,k,j,i),
      &adm.g_dd(m,1,1,k,j,i), &adm.g_dd(m,1,2,k,j,i), &adm.g_dd(m,2,2,k,j,i),
      &adm.vK_dd(m,0,0,k,j,i), &adm.vK_dd(m,0,1,k,j,i), &adm.vK_dd(m,0,2,k,j,i),
      &adm.vK_dd(m,1,1,k,j,i), &adm.vK_dd(m,1,2,k,j,i), &adm.vK_dd(m,2,2,k,j,i));
  });
}


} // namespace adm
