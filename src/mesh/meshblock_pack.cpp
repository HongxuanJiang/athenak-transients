//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <memory>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mb_storage.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "z4c/cce/cce.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "radiation/radiation.hpp"
#include "sink_particles/sink_particles.hpp"
#include "srcterms/turb_driver.hpp"
#include "particles/particles.hpp"
#include "units/units.hpp"
#include "meshblock_pack.hpp"
#include "gravity/gravity.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn void EnsureLATCacheSlots1D(...)
//! \fn void EnsureLATCacheSlots2D(...)
//! \brief grow-only sizing for the per-slot LAT cache mask lists
//!
//! The slot Views are retained across cache invalidation: only the value/signature list
//! that names the live slots is cleared, so a rebuild refills the same allocations
//! instead of destroying and re-creating a dozen device arrays per slot.  A slot list is
//! therefore sized to the largest slot count and block capacity ever asked for, and
//! entries beyond the live slot count are simply unreferenced -- LATFactorCacheSlot() and
//! LATDueFactorCacheSlot() only ever return indices into the value/signature lists.

void EnsureLATCacheSlots1D(std::vector<DualArray1D<int>> &slots, const char *label,
                           int nslots, int nelem) {
  const int need = std::max(1, nelem);
  for (auto &slot : slots) {
    if (static_cast<int>(slot.extent(0)) < need) {
      slot = DualArray1D<int>(label, need);
    }
  }
  while (static_cast<int>(slots.size()) < nslots) {
    slots.emplace_back(label, need);
  }
}

void EnsureLATCacheSlots2D(std::vector<DualArray2D<int>> &slots, const char *label,
                           int nslots, int nelem0, int nelem1) {
  const int need0 = std::max(1, nelem0);
  const int need1 = std::max(1, nelem1);
  for (auto &slot : slots) {
    if (static_cast<int>(slot.extent(0)) < need0 ||
        static_cast<int>(slot.extent(1)) < need1) {
      slot = DualArray2D<int>(label, need0, need1);
    }
  }
  while (static_cast<int>(slots.size()) < nslots) {
    slots.emplace_back(label, need0, need1);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide) :
  pmesh(pm),
  gids(igids),
  gide(igide),
  nmb_thispack(igide - igids + 1),
  lat_active_mb("lat_active_mb", std::max(1, igide - igids + 1)),
  lat_active_indices("lat_active_indices", std::max(1, igide - igids + 1)),
  lat_boundary_send_mb("lat_boundary_send_mb", std::max(1, igide - igids + 1)),
  lat_boundary_send_indices("lat_boundary_send_indices", std::max(1, igide - igids + 1)),
  lat_boundary_send_edges("lat_boundary_send_edges", std::max(1, igide - igids + 1)),
  lat_flux_recv_indices("lat_flux_recv_indices", std::max(1, igide - igids + 1)),
  lat_send_nghbr("lat_send_nghbr", std::max(1, igide - igids + 1), 1),
  lat_flux_accum_nghbr("lat_flux_accum_nghbr", std::max(1, igide - igids + 1), 1),
  lat_flux_send_nghbr("lat_flux_send_nghbr", std::max(1, igide - igids + 1), 1),
  lat_flux_recv_nghbr("lat_flux_recv_nghbr", std::max(1, igide - igids + 1), 1),
  lat_nghbr_factor("lat_nghbr_factor", std::max(1, igide - igids + 1), 1),
  lat_time_start("lat_time_start", std::max(1, igide - igids + 1)),
  lat_time_end("lat_time_end", std::max(1, igide - igids + 1)),
  lat_step_dt("lat_step_dt", std::max(1, igide - igids + 1)),
  lat_step_factor("lat_step_factor", std::max(1, igide - igids + 1)),
  lat_step_level("lat_step_level", std::max(1, igide - igids + 1)) {
  Kokkos::deep_copy(lat_step_dt.d_view, static_cast<Real>(0.0));
  Kokkos::deep_copy(lat_step_factor.d_view, 1);
  Kokkos::deep_copy(lat_step_level.d_view, 0);
  lat_step_dt.template modify<DevExeSpace>();
  lat_step_factor.template modify<DevExeSpace>();
  lat_step_level.template modify<DevExeSpace>();
  // create map for task lists
  tl_map.insert(std::make_pair("before_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("before_stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_stagen",std::make_shared<TaskList>()));
}

void MeshBlockPack::ResizeActiveMask() {
  const int nmb = std::max(1, nmb_thispack);
  const int nnghbr = (pmb != nullptr) ? std::max(1, pmb->nnghbr) : 1;
  // Size the masks by the same rounded-up block capacity the physics field arrays use
  // (mesh/mb_storage.hpp) and never shrink them.  With refinement_interval = 1 the local
  // block count moves by a block or two on essentially every cycle, and the previous
  // exact-fit test then destroyed and re-created ~17 device arrays plus their host
  // mirrors each time -- a large share of the per-cycle allocation COUNT for well under
  // a MiB of data.  Over-sizing is safe here because every consumer bounds its loop by
  // one of the lat_n*_thispack counts written after each rebuild (or by nmb_thispack),
  // never by extent(); the trailing entries are unreachable.
  const int cap = MeshBlockStorageCapacity(nmb);
  bool resized = false;
  if (static_cast<int>(lat_active_mb.extent(0)) < nmb) {
    lat_active_mb = DualArray1D<int>("lat_active_mb", cap);
    resized = true;
  }
  if (static_cast<int>(lat_active_indices.extent(0)) < nmb) {
    lat_active_indices = DualArray1D<int>("lat_active_indices", cap);
    resized = true;
  }
  if (static_cast<int>(lat_boundary_send_mb.extent(0)) < nmb) {
    lat_boundary_send_mb = DualArray1D<int>("lat_boundary_send_mb", cap);
    resized = true;
  }
  if (static_cast<int>(lat_boundary_send_indices.extent(0)) < nmb) {
    lat_boundary_send_indices = DualArray1D<int>("lat_boundary_send_indices", cap);
    resized = true;
  }
  const int nedge = nmb*nnghbr;
  if (static_cast<int>(lat_boundary_send_edges.extent(0)) < nedge) {
    lat_boundary_send_edges = DualArray1D<int>("lat_boundary_send_edges", cap*nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_flux_recv_indices.extent(0)) < nmb) {
    lat_flux_recv_indices = DualArray1D<int>("lat_flux_recv_indices", cap);
    resized = true;
  }
  if (static_cast<int>(lat_send_nghbr.extent(0)) < nmb ||
      static_cast<int>(lat_send_nghbr.extent(1)) < nnghbr) {
    lat_send_nghbr = DualArray2D<int>("lat_send_nghbr", cap, nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_flux_accum_nghbr.extent(0)) < nmb ||
      static_cast<int>(lat_flux_accum_nghbr.extent(1)) < nnghbr) {
    lat_flux_accum_nghbr = DualArray2D<int>("lat_flux_accum_nghbr", cap, nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_flux_send_nghbr.extent(0)) < nmb ||
      static_cast<int>(lat_flux_send_nghbr.extent(1)) < nnghbr) {
    lat_flux_send_nghbr = DualArray2D<int>("lat_flux_send_nghbr", cap, nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_flux_recv_nghbr.extent(0)) < nmb ||
      static_cast<int>(lat_flux_recv_nghbr.extent(1)) < nnghbr) {
    lat_flux_recv_nghbr = DualArray2D<int>("lat_flux_recv_nghbr", cap, nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_nghbr_factor.extent(0)) < nmb ||
      static_cast<int>(lat_nghbr_factor.extent(1)) < nnghbr) {
    lat_nghbr_factor = DualArray2D<int>("lat_nghbr_factor", cap, nnghbr);
    resized = true;
  }
  if (static_cast<int>(lat_time_start.extent(0)) < nmb) {
    lat_time_start = DualArray1D<Real>("lat_time_start", cap);
    resized = true;
  }
  if (static_cast<int>(lat_time_end.extent(0)) < nmb) {
    lat_time_end = DualArray1D<Real>("lat_time_end", cap);
    resized = true;
  }
  if (static_cast<int>(lat_step_dt.extent(0)) < nmb) {
    lat_step_dt = DualArray1D<Real>("lat_step_dt", cap);
    Kokkos::deep_copy(lat_step_dt.d_view, static_cast<Real>(0.0));
    lat_step_dt.template modify<DevExeSpace>();
    resized = true;
  }
  if (static_cast<int>(lat_step_factor.extent(0)) < nmb) {
    lat_step_factor = DualArray1D<int>("lat_step_factor", cap);
    Kokkos::deep_copy(lat_step_factor.d_view, 1);
    lat_step_factor.template modify<DevExeSpace>();
    resized = true;
  }
  if (static_cast<int>(lat_step_level.extent(0)) < nmb) {
    lat_step_level = DualArray1D<int>("lat_step_level", cap);
    Kokkos::deep_copy(lat_step_level.d_view, 0);
    lat_step_level.template modify<DevExeSpace>();
    resized = true;
  }
  // A block-count change no longer necessarily reallocates, so this is no longer the
  // hook that invalidates the caches on a topology change: BuildLAT*FactorCache keys on
  // nmb_thispack/nnghbr/hydro_lat_metadata_version, and the AMR path calls
  // ReleaseLATCacheMemory() before it migrates blocks.
  if (resized) InvalidateLATFactorCache();
}

void MeshBlockPack::SetAllMeshBlocksActive() {
  ResizeActiveMask();
  lat_active_mask_enabled = false;
  lat_per_block_timestep = false;
  lat_union_pending_below_factor = 0;
  lat_nactive_thispack = nmb_thispack;
  lat_nboundary_send_thispack = nmb_thispack;
  lat_nboundary_send_edges_thispack =
      nmb_thispack*((pmb != nullptr) ? pmb->nnghbr : 1);
  lat_nflux_recv_thispack = nmb_thispack;
}

// Called immediately before a topology transaction.  Its job is the fence plus the
// detach: nothing may still be aliasing a cache slot, or reading one, while AMR moves
// blocks underneath it.  It intentionally does NOT free the slot allocations -- with
// refinement_interval = 1 this runs essentially every cycle, and freeing here is exactly
// the churn the retained slot lists exist to avoid.  A slot is a few hundred KiB and is
// refilled in place by the next BuildLAT*FactorCache.
void MeshBlockPack::ReleaseLATCacheMemory() {
  const bool has_cache =
      lat_active_mask_uses_cache ||
      !lat_factor_cache_values.empty() ||
      !lat_due_cache_signatures.empty();
  if (!has_cache) return;
  DevExeSpace().fence();
  InvalidateLATFactorCache();
}

void MeshBlockPack::ResetLATBlockTimes(Real time) {
  ResizeActiveMask();
  lat_per_block_timestep = false;
  lat_union_pending_below_factor = 0;
  Kokkos::deep_copy(lat_time_start.d_view, time);
  Kokkos::deep_copy(lat_time_end.d_view, time);
  Kokkos::deep_copy(lat_step_dt.d_view, static_cast<Real>(0.0));
  lat_time_start.template modify<DevExeSpace>();
  lat_time_end.template modify<DevExeSpace>();
  lat_step_dt.template modify<DevExeSpace>();
}

void MeshBlockPack::SetActiveLATBlockTimes(Real start_time, Real end_time) {
  ResizeActiveMask();
  lat_per_block_timestep = false;
  const int nwork = lat_active_mask_enabled ? lat_nactive_thispack : nmb_thispack;
  if (nwork <= 0) return;
  const bool use_active_indices = lat_active_mask_enabled;
  auto active_indices = lat_active_indices.d_view;
  auto time_start = lat_time_start.d_view;
  auto time_end = lat_time_end.d_view;
  Kokkos::parallel_for("SetActiveLATBlockTimes",
      athenak_lw(Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork)),
      KOKKOS_LAMBDA(const int a) {
        const int m = use_active_indices ? active_indices(a) : a;
        time_start(m) = start_time;
        time_end(m) = end_time;
      });
  lat_time_start.template modify<DevExeSpace>();
  lat_time_end.template modify<DevExeSpace>();
}

void MeshBlockPack::ConfigureLATUnionStage1(Real start_time, Real fine_dt,
                                             int max_factor) {
  ResizeActiveMask();
  lat_per_block_timestep = true;
  const int nwork = lat_active_mask_enabled ? lat_nactive_thispack : nmb_thispack;
  if (nwork <= 0) return;

  const int capped_max_factor = std::max(1, max_factor);
  const bool use_active_indices = lat_active_mask_enabled;
  auto active_indices = lat_active_indices.d_view;
  auto step_factor = lat_step_factor.d_view;
  auto step_dt = lat_step_dt.d_view;
  auto time_start = lat_time_start.d_view;
  auto time_end = lat_time_end.d_view;
  Kokkos::parallel_for("ConfigureLATUnionStage1",
      athenak_lw(Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork)),
      KOKKOS_LAMBDA(const int a) {
        const int m = use_active_indices ? active_indices(a) : a;
        const int factor = (step_factor(m) < 1) ? 1 :
                           ((step_factor(m) > capped_max_factor) ?
                            capped_max_factor : step_factor(m));
        const Real block_dt = fine_dt*static_cast<Real>(factor);
        step_dt(m) = block_dt;
        time_start(m) = start_time;
        time_end(m) = start_time + block_dt;
      });
  lat_step_dt.template modify<DevExeSpace>();
  lat_time_start.template modify<DevExeSpace>();
  lat_time_end.template modify<DevExeSpace>();
}

bool MeshBlockPack::LATActiveMaskUsesCache() const {
  return lat_active_mask_enabled && lat_active_mask_uses_cache;
}

std::uint64_t MeshBlockPack::LATCacheGeneration() const {
  return lat_cache_generation;
}

int MeshBlockPack::LATFactorForGID(int gid, int max_factor) const {
  if (pmesh == nullptr) return 1;
  return pmesh->HydroLATFactorForGID(gid, max_factor);
}

int MeshBlockPack::SetLATFluxCorrectionByCompletionPhase(int max_factor,
                                                         int completed_tick) {
  ResizeActiveMask();
  lat_per_block_timestep = false;
  lat_active_mask_enabled = true;
  const int capped_max_factor = std::max(1, max_factor);
  const int tick = std::max(0, completed_tick);

  // A positive completion tick has the same endpoint set as the corresponding
  // due phase.  Alias the due cache's receiver-only mask instead of allocating
  // and synchronizing nine scratch views after every fine tick.  Tick zero is
  // intentionally excluded: the legacy endpoint rule selects only factor 1 at
  // zero, whereas due phase zero selects every factor.
  if (tick > 0 && BuildLATDueFactorCache(capped_max_factor, tick, true)) {
    const int phase = tick % capped_max_factor;
    const std::uint64_t signature = LATDueFactorSignature(phase, true);
    const int slot = LATDueFactorCacheSlot(signature);
    if (slot >= 0) {
      lat_active_mask_uses_cache = true;
      lat_nactive_thispack = lat_due_cache_nflux_recv[slot];
      lat_nboundary_send_thispack = 0;
      lat_nboundary_send_edges_thispack = 0;
      lat_nflux_recv_thispack = lat_due_cache_nflux_recv[slot];
      lat_cached_needs_restrict = false;
      lat_cached_needs_prolongate = false;
      lat_cached_needs_physical_bc = false;
      lat_cached_needs_neighbor_recv = false;
      lat_active_mb = lat_due_cache_flux_recv_mb[slot];
      lat_active_indices = lat_due_cache_flux_recv_indices[slot];
      lat_flux_recv_indices = lat_due_cache_flux_recv_indices[slot];
      lat_flux_recv_nghbr = lat_due_cache_flux_recv_nghbr[slot];
      return lat_nactive_thispack;
    }
  }

  // Retain the explicit construction for unusual/incomplete cache states and
  // for the distinct tick-zero endpoint semantics described above.
  DetachLATActiveMask();
  lat_active_mask_uses_cache = false;
  const bool same_level_lat = (pmesh != nullptr) && pmesh->hydro_lat_same_level;

  int nrefluxed = 0;
  int nflux_recv = 0;
  for (int m = 0; m < nmb_thispack; ++m) {
    lat_active_mb.h_view(m) = 0;
    lat_boundary_send_mb.h_view(m) = 0;
    for (int n = 0; n < pmb->nnghbr; ++n) {
      lat_send_nghbr.h_view(m,n) = 0;
      lat_flux_accum_nghbr.h_view(m,n) = 0;
      lat_flux_send_nghbr.h_view(m,n) = 0;
      lat_flux_recv_nghbr.h_view(m,n) = 0;
    }

    const int lev = pmb->mb_lev.h_view(m);
    const int factor = LATFactorForGID(pmb->mb_gid.h_view(m), capped_max_factor);
    const bool endpoint = (factor <= 1) || (tick > 0 && (tick % factor) == 0);
    if (!endpoint) continue;

    bool flux_recv = false;
    for (int n = 0; n < pmb->nnghbr; ++n) {
      const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
      if (!face_nghbr) continue;
      const auto &nb = pmb->nghbr.h_view(m,n);
      if (nb.gid < 0) continue;
      const int nb_factor = LATFactorForGID(nb.gid, capped_max_factor);
      const bool receives =
          (nb.lev > lev) ||
          (same_level_lat && nb.lev == lev && factor > nb_factor);
      if (receives) {
        lat_flux_recv_nghbr.h_view(m,n) = 1;
        flux_recv = true;
      }
    }
    if (flux_recv) {
      lat_active_mb.h_view(m) = 1;
      lat_active_indices.h_view(nrefluxed++) = m;
      lat_flux_recv_indices.h_view(nflux_recv++) = m;
    }
  }

  lat_nactive_thispack = nrefluxed;
  lat_nboundary_send_thispack = 0;
  lat_nboundary_send_edges_thispack = 0;
  lat_nflux_recv_thispack = nflux_recv;
  lat_cached_needs_restrict = false;
  lat_cached_needs_prolongate = false;
  lat_cached_needs_physical_bc = false;
  lat_cached_needs_neighbor_recv = false;

  lat_active_mb.template modify<HostMemSpace>();
  lat_active_indices.template modify<HostMemSpace>();
  lat_boundary_send_mb.template modify<HostMemSpace>();
  lat_boundary_send_indices.template modify<HostMemSpace>();
  lat_flux_recv_indices.template modify<HostMemSpace>();
  lat_send_nghbr.template modify<HostMemSpace>();
  lat_flux_accum_nghbr.template modify<HostMemSpace>();
  lat_flux_send_nghbr.template modify<HostMemSpace>();
  lat_flux_recv_nghbr.template modify<HostMemSpace>();
  lat_active_mb.template sync<DevExeSpace>();
  lat_active_indices.template sync<DevExeSpace>();
  lat_boundary_send_mb.template sync<DevExeSpace>();
  lat_boundary_send_indices.template sync<DevExeSpace>();
  lat_flux_recv_indices.template sync<DevExeSpace>();
  lat_send_nghbr.template sync<DevExeSpace>();
  lat_flux_accum_nghbr.template sync<DevExeSpace>();
  lat_flux_send_nghbr.template sync<DevExeSpace>();
  lat_flux_recv_nghbr.template sync<DevExeSpace>();
  return nrefluxed;
}

void MeshBlockPack::AdvanceLATCacheGeneration() {
  if (lat_cache_generation == std::numeric_limits<std::uint64_t>::max()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "LAT cache generation exhausted; refusing to reuse cache keys."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ++lat_cache_generation;
}

void MeshBlockPack::InvalidateLATFactorCache() {
  AdvanceLATCacheGeneration();
  DetachLATActiveMask();
  lat_active_mask_uses_cache = false;
  lat_factor_cache_max_factor = 0;
  lat_factor_cache_nmb = 0;
  lat_factor_cache_nnghbr = 0;
  lat_factor_cache_metadata_version = 0;
  lat_factor_cache_values.clear();
  lat_factor_cache_nactive.clear();
  lat_factor_cache_nboundary_send.clear();
  lat_factor_cache_nboundary_send_edges.clear();
  lat_factor_cache_nflux_recv.clear();
  // The slot Views themselves are deliberately NOT cleared.  lat_factor_cache_values is
  // the only thing that names a live slot (LATFactorCacheSlot searches it, and
  // LATDueFactorSignature indexes it), so emptying it is what invalidates the cache;
  // retaining the allocations lets the next BuildLATFactorCache refill them in place.
  // The invalidation key itself is unaffected: AdvanceLATCacheGeneration() above still
  // runs, which is what the bvals rank-packed layout caches key on.
  lat_factor_cache_needs_restrict.clear();
  lat_factor_cache_needs_prolongate.clear();
  lat_factor_cache_needs_physical_bc.clear();
  lat_factor_cache_needs_neighbor_recv.clear();
  ClearLATDueFactorCache();
}

void MeshBlockPack::DetachLATActiveMask() {
  if (!lat_active_mask_uses_cache) return;
  const int nmb = std::max(1, nmb_thispack);
  const int nnghbr = (pmb != nullptr) ? std::max(1, pmb->nnghbr) : 1;
  const int cap = MeshBlockStorageCapacity(nmb);
  // Reuse one persistent scratch set instead of allocating fresh views on every
  // detach.  All consumers read these arrays only through the *_thispack counts
  // written after each host rebuild, so stale tails are never observed, and the
  // stable pointers let the rank-packed layout caches match by identity.  The test is
  // grow-only against the rounded-up block capacity for the same reason as in
  // ResizeActiveMask: an exact-fit test rebuilt the whole set on nearly every AMR cycle.
  // The two probes stand in for all ten because they are always allocated together.
  if (static_cast<int>(lat_detach_scratch_active_mb.extent(0)) < nmb ||
      static_cast<int>(lat_detach_scratch_send_nghbr.extent(0)) < nmb ||
      static_cast<int>(lat_detach_scratch_send_nghbr.extent(1)) < nnghbr) {
    // AMR may replace these allocations while asynchronous kernels still hold the
    // previous views; those views stay alive through their own reference counts.
    lat_detach_scratch_active_mb = DualArray1D<int>("lat_active_mb", cap);
    lat_detach_scratch_active_indices = DualArray1D<int>("lat_active_indices", cap);
    lat_detach_scratch_boundary_send_mb =
        DualArray1D<int>("lat_boundary_send_mb", cap);
    lat_detach_scratch_boundary_send_indices =
        DualArray1D<int>("lat_boundary_send_indices", cap);
    lat_detach_scratch_boundary_send_edges =
        DualArray1D<int>("lat_boundary_send_edges", cap*nnghbr);
    lat_detach_scratch_flux_recv_indices =
        DualArray1D<int>("lat_flux_recv_indices", cap);
    lat_detach_scratch_send_nghbr = DualArray2D<int>("lat_send_nghbr", cap, nnghbr);
    lat_detach_scratch_flux_accum_nghbr =
        DualArray2D<int>("lat_flux_accum_nghbr", cap, nnghbr);
    lat_detach_scratch_flux_send_nghbr =
        DualArray2D<int>("lat_flux_send_nghbr", cap, nnghbr);
    lat_detach_scratch_flux_recv_nghbr =
        DualArray2D<int>("lat_flux_recv_nghbr", cap, nnghbr);
  }
  lat_active_mb = lat_detach_scratch_active_mb;
  lat_active_indices = lat_detach_scratch_active_indices;
  lat_boundary_send_mb = lat_detach_scratch_boundary_send_mb;
  lat_boundary_send_indices = lat_detach_scratch_boundary_send_indices;
  lat_boundary_send_edges = lat_detach_scratch_boundary_send_edges;
  lat_flux_recv_indices = lat_detach_scratch_flux_recv_indices;
  lat_send_nghbr = lat_detach_scratch_send_nghbr;
  lat_flux_accum_nghbr = lat_detach_scratch_flux_accum_nghbr;
  lat_flux_send_nghbr = lat_detach_scratch_flux_send_nghbr;
  lat_flux_recv_nghbr = lat_detach_scratch_flux_recv_nghbr;
  lat_nactive_thispack = 0;
  lat_nboundary_send_thispack = 0;
  lat_nboundary_send_edges_thispack = 0;
  lat_nflux_recv_thispack = 0;
  lat_active_mask_uses_cache = false;
}

int MeshBlockPack::LATFactorCacheSlot(int factor) const {
  for (int n = 0; n < static_cast<int>(lat_factor_cache_values.size()); ++n) {
    if (lat_factor_cache_values[n] == factor) return n;
  }
  return -1;
}

void MeshBlockPack::ClearLATDueFactorCache() {
  if (!lat_due_cache_signatures.empty()) AdvanceLATCacheGeneration();
  lat_due_cache_max_factor = 0;
  lat_due_cache_nmb = 0;
  lat_due_cache_nnghbr = 0;
  lat_due_cache_include_factor_one = 0;
  lat_due_cache_union_stage1 = 0;
  lat_due_cache_metadata_version = 0;
  lat_due_cache_signatures.clear();
  lat_due_cache_nactive.clear();
  lat_due_cache_nboundary_send.clear();
  lat_due_cache_nboundary_send_edges.clear();
  lat_due_cache_nflux_recv.clear();
  lat_due_cache_nupdate_flux_recv.clear();
  // As in InvalidateLATFactorCache, the slot Views are retained.  The signature list is
  // the sole record of which slots are live (LATDueFactorCacheSlot searches it and the
  // next slot index is its size), so clearing it invalidates the cache while leaving the
  // allocations for BuildLATDueFactorCache to refill.  Callers that could still be
  // aliasing a slot through the public lat_* members detach first; see the key-change
  // branch of BuildLATDueFactorCache.
  lat_due_cache_needs_restrict.clear();
  lat_due_cache_needs_prolongate.clear();
  lat_due_cache_needs_physical_bc.clear();
  lat_due_cache_needs_neighbor_recv.clear();
}

std::uint64_t MeshBlockPack::LATDueFactorSignature(
    int phase, bool include_factor_one) const {
  std::uint64_t signature = 0;
  // max_factor is an int and each cache value is obtained by halving it, so
  // there are at most 31 factor slots and all of them fit in this signature.
  for (int n = 0; n < static_cast<int>(lat_factor_cache_values.size()); ++n) {
    const int factor = lat_factor_cache_values[n];
    if ((factor > 1 || include_factor_one) && ((phase % factor) == 0)) {
      signature |= (std::uint64_t{1} << n);
    }
  }
  return signature;
}

int MeshBlockPack::LATDueFactorCacheSlot(std::uint64_t signature) const {
  for (int n = 0; n < static_cast<int>(lat_due_cache_signatures.size()); ++n) {
    if (lat_due_cache_signatures[n] == signature) return n;
  }
  return -1;
}

bool MeshBlockPack::BuildLATFactorCache(int max_factor) {
  ResizeActiveMask();
  if (pmb == nullptr || nmb_thispack <= 0 || pmb->nnghbr <= 0) return false;

  const int capped_max_factor = std::max(1, max_factor);
  const std::uint64_t metadata_version =
      (pmesh != nullptr) ? pmesh->hydro_lat_metadata_version : 0;
  if (lat_factor_cache_metadata_version == metadata_version &&
      lat_factor_cache_max_factor == capped_max_factor &&
      lat_factor_cache_nmb == nmb_thispack &&
      lat_factor_cache_nnghbr == pmb->nnghbr &&
      !(lat_factor_cache_values.empty())) {
    return true;
  }

  InvalidateLATFactorCache();
  lat_factor_cache_max_factor = capped_max_factor;
  lat_factor_cache_nmb = nmb_thispack;
  lat_factor_cache_nnghbr = pmb->nnghbr;
  lat_factor_cache_metadata_version = metadata_version;

  for (int factor = capped_max_factor; factor >= 1; factor /= 2) {
    lat_factor_cache_values.push_back(factor);
    if (factor == 1) break;
  }
  lat_step_factor.template sync<HostMemSpace>();
  lat_step_level.template sync<HostMemSpace>();
  lat_nghbr_factor.template sync<HostMemSpace>();
  for (int m = 0; m < nmb_thispack; ++m) {
    const int factor = LATFactorForGID(pmb->mb_gid.h_view(m), capped_max_factor);
    lat_step_factor.h_view(m) = factor;
    int level = 0;
    for (int f=factor; f > 1 && level < 20; f >>= 1) ++level;
    lat_step_level.h_view(m) = level;
    for (int n = 0; n < pmb->nnghbr; ++n) {
      const int gid = pmb->nghbr.h_view(m,n).gid;
      lat_nghbr_factor.h_view(m,n) = (gid >= 0) ?
          LATFactorForGID(gid, capped_max_factor) : 0;
    }
  }
  lat_step_factor.template modify<HostMemSpace>();
  lat_step_level.template modify<HostMemSpace>();
  lat_nghbr_factor.template modify<HostMemSpace>();
  lat_step_factor.template sync<DevExeSpace>();
  lat_step_level.template sync<DevExeSpace>();
  lat_nghbr_factor.template sync<DevExeSpace>();
  const int nslots = static_cast<int>(lat_factor_cache_values.size());
  lat_factor_cache_nactive.assign(nslots, 0);
  lat_factor_cache_nboundary_send.assign(nslots, 0);
  lat_factor_cache_nboundary_send_edges.assign(nslots, 0);
  lat_factor_cache_nflux_recv.assign(nslots, 0);
  lat_factor_cache_needs_restrict.assign(nslots, false);
  lat_factor_cache_needs_prolongate.assign(nslots, false);
  lat_factor_cache_needs_physical_bc.assign(nslots, false);
  lat_factor_cache_needs_neighbor_recv.assign(nslots, false);

  // Size the slot lists once, to the number of factors in this window and to the
  // rounded-up block capacity, and never shrink them.  Previously every invalidation
  // destroyed and re-created ten device arrays per slot -- with a sync window that swings
  // 16/2/1/1 and an AMR transaction essentially every cycle that was the dominant LAT
  // allocation count.  Slots past nslots (left over from a longer window) are unreachable
  // because lat_factor_cache_values, not the slot list, defines which slots are live.
  const int nmb = std::max(1, nmb_thispack);
  const int nnghbr = pmb->nnghbr;
  const int cap = MeshBlockStorageCapacity(nmb);
  EnsureLATCacheSlots1D(lat_factor_cache_active_mb,
      "lat_factor_cache_active_mb", nslots, cap);
  EnsureLATCacheSlots1D(lat_factor_cache_active_indices,
      "lat_factor_cache_active_indices", nslots, cap);
  EnsureLATCacheSlots1D(lat_factor_cache_boundary_send_mb,
      "lat_factor_cache_boundary_send_mb", nslots, cap);
  EnsureLATCacheSlots1D(lat_factor_cache_boundary_send_indices,
      "lat_factor_cache_boundary_send_indices", nslots, cap);
  EnsureLATCacheSlots1D(lat_factor_cache_boundary_send_edges,
      "lat_factor_cache_boundary_send_edges", nslots, cap*nnghbr);
  EnsureLATCacheSlots1D(lat_factor_cache_flux_recv_indices,
      "lat_factor_cache_flux_recv_indices", nslots, cap);
  EnsureLATCacheSlots2D(lat_factor_cache_send_nghbr,
      "lat_factor_cache_send_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_factor_cache_flux_accum_nghbr,
      "lat_factor_cache_flux_accum_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_factor_cache_flux_send_nghbr,
      "lat_factor_cache_flux_send_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_factor_cache_flux_recv_nghbr,
      "lat_factor_cache_flux_recv_nghbr", nslots, cap, nnghbr);

  for (int slot = 0; slot < nslots; ++slot) {
    const int requested_factor = lat_factor_cache_values[slot];
    int nactive = 0;
    int nboundary_send = 0;
    int nboundary_send_edges = 0;
    int nflux_recv = 0;
    bool s_needs_restrict = false;
    bool s_needs_prolongate = false;
    bool s_needs_physical_bc = false;
    bool s_needs_neighbor_recv = false;
    const bool strictly_periodic = (pmesh != nullptr) && pmesh->strictly_periodic;
    for (int m = 0; m < nmb_thispack; ++m) {
      const int lev = pmb->mb_lev.h_view(m);
      const int factor = lat_step_factor.h_view(m);
      const int active = (factor == requested_factor) ? 1 : 0;
      const bool same_level_lat = (pmesh != nullptr) && pmesh->hydro_lat_same_level;
      lat_factor_cache_active_mb[slot].h_view(m) = active;
      if (active) lat_factor_cache_active_indices[slot].h_view(nactive) = m;
      lat_factor_cache_boundary_send_mb[slot].h_view(m) = 0;
      bool flux_recv = false;
      for (int n = 0; n < nnghbr; ++n) {
        const auto &nb = pmb->nghbr.h_view(m,n);
        const int nb_factor = (nb.gid >= 0) ?
            LATFactorForGID(nb.gid, capped_max_factor) : 0;
        const int send = (nb.gid >= 0 && nb_factor == requested_factor) ? 1 : 0;
        lat_factor_cache_send_nghbr[slot].h_view(m,n) = send;
        if (send != 0) {
          lat_factor_cache_boundary_send_edges[slot].h_view(nboundary_send_edges++) =
              m*nnghbr + n;
        }
        const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
        const bool same_level_mixed =
            same_level_lat && face_nghbr && nb.gid >= 0 && nb.lev == lev &&
            nb_factor != factor;
        lat_factor_cache_flux_accum_nghbr[slot].h_view(m,n) =
            (active && face_nghbr && nb.gid >= 0 &&
             ((nb.lev > lev) || (same_level_mixed && nb_factor < factor))) ? 1 : 0;
        lat_factor_cache_flux_send_nghbr[slot].h_view(m,n) =
            (active && face_nghbr && nb.gid >= 0 &&
             ((nb.lev < lev) || (same_level_mixed && factor < nb_factor))) ? 1 : 0;
        lat_factor_cache_flux_recv_nghbr[slot].h_view(m,n) =
            (face_nghbr && nb.gid >= 0 && nb_factor == requested_factor &&
             ((nb.lev > lev) ||
              (same_level_lat && nb.lev == lev && factor > requested_factor))) ? 1 : 0;
        lat_factor_cache_boundary_send_mb[slot].h_view(m) |= send;
        flux_recv = flux_recv ||
            (lat_factor_cache_flux_recv_nghbr[slot].h_view(m,n) != 0);
        if (send && nb.gid >= 0 && nb.lev < lev) s_needs_restrict = true;
        if (active && nb.gid >= 0 && nb.lev < lev) s_needs_prolongate = true;
        if (active && nb.gid >= 0) s_needs_neighbor_recv = true;
      }
      if (lat_factor_cache_boundary_send_mb[slot].h_view(m) != 0) {
        lat_factor_cache_boundary_send_indices[slot].h_view(nboundary_send++) = m;
      }
      if (flux_recv) lat_factor_cache_flux_recv_indices[slot].h_view(nflux_recv++) = m;
      if (active && !strictly_periodic) {
        for (int f = 0; f < 6; ++f) {
          const BoundaryFlag bc = pmb->mb_bcs.h_view(m,f);
          if (bc != BoundaryFlag::block && bc != BoundaryFlag::periodic &&
              bc != BoundaryFlag::shear_periodic) {
            s_needs_physical_bc = true;
            break;
          }
        }
      }
      nactive += active;
    }
    lat_factor_cache_nactive[slot] = nactive;
    lat_factor_cache_nboundary_send[slot] = nboundary_send;
    lat_factor_cache_nboundary_send_edges[slot] = nboundary_send_edges;
    lat_factor_cache_nflux_recv[slot] = nflux_recv;
    lat_factor_cache_needs_restrict[slot] = s_needs_restrict;
    lat_factor_cache_needs_prolongate[slot] = s_needs_prolongate;
    lat_factor_cache_needs_physical_bc[slot] = s_needs_physical_bc;
    lat_factor_cache_needs_neighbor_recv[slot] = s_needs_neighbor_recv;

    lat_factor_cache_active_mb[slot].template modify<HostMemSpace>();
    lat_factor_cache_active_indices[slot].template modify<HostMemSpace>();
    lat_factor_cache_boundary_send_mb[slot].template modify<HostMemSpace>();
    lat_factor_cache_boundary_send_indices[slot].template modify<HostMemSpace>();
    lat_factor_cache_boundary_send_edges[slot].template modify<HostMemSpace>();
    lat_factor_cache_flux_recv_indices[slot].template modify<HostMemSpace>();
    lat_factor_cache_send_nghbr[slot].template modify<HostMemSpace>();
    lat_factor_cache_flux_accum_nghbr[slot].template modify<HostMemSpace>();
    lat_factor_cache_flux_send_nghbr[slot].template modify<HostMemSpace>();
    lat_factor_cache_flux_recv_nghbr[slot].template modify<HostMemSpace>();
    lat_factor_cache_active_mb[slot].template sync<DevExeSpace>();
    lat_factor_cache_active_indices[slot].template sync<DevExeSpace>();
    lat_factor_cache_boundary_send_mb[slot].template sync<DevExeSpace>();
    lat_factor_cache_boundary_send_indices[slot].template sync<DevExeSpace>();
    lat_factor_cache_boundary_send_edges[slot].template sync<DevExeSpace>();
    lat_factor_cache_flux_recv_indices[slot].template sync<DevExeSpace>();
    lat_factor_cache_send_nghbr[slot].template sync<DevExeSpace>();
    lat_factor_cache_flux_accum_nghbr[slot].template sync<DevExeSpace>();
    lat_factor_cache_flux_send_nghbr[slot].template sync<DevExeSpace>();
    lat_factor_cache_flux_recv_nghbr[slot].template sync<DevExeSpace>();
  }
  return true;
}

bool MeshBlockPack::BuildLATDueFactorCache(int max_factor, int tick_cycle,
                                           bool include_factor_one) {
  ResizeActiveMask();
  if (pmb == nullptr || nmb_thispack <= 0 || pmb->nnghbr <= 0) return false;

  const int capped_max_factor = std::max(1, max_factor);
  if (!BuildLATFactorCache(capped_max_factor)) return false;

  const int phase = std::max(0, tick_cycle) % capped_max_factor;
  const int include_factor_one_int = include_factor_one ? 1 : 0;
  const int union_stage1_int = lat_union_stage1_enabled ? 1 : 0;
  const std::uint64_t metadata_version =
      (pmesh != nullptr) ? pmesh->hydro_lat_metadata_version : 0;
  if (lat_due_cache_metadata_version != metadata_version ||
      lat_due_cache_max_factor != capped_max_factor ||
      lat_due_cache_nmb != nmb_thispack ||
      lat_due_cache_nnghbr != pmb->nnghbr ||
      lat_due_cache_include_factor_one != include_factor_one_int ||
      lat_due_cache_union_stage1 != union_stage1_int) {
    if (!lat_due_cache_signatures.empty() && lat_active_mask_uses_cache) {
      DetachLATActiveMask();
    }
    ClearLATDueFactorCache();
    lat_due_cache_max_factor = capped_max_factor;
    lat_due_cache_nmb = nmb_thispack;
    lat_due_cache_nnghbr = pmb->nnghbr;
    lat_due_cache_include_factor_one = include_factor_one_int;
    lat_due_cache_union_stage1 = union_stage1_int;
    lat_due_cache_metadata_version = metadata_version;
  }

  const std::uint64_t signature =
      LATDueFactorSignature(phase, include_factor_one);
  if (LATDueFactorCacheSlot(signature) >= 0) return true;

  const int slot = static_cast<int>(lat_due_cache_signatures.size());
  lat_due_cache_signatures.push_back(signature);
  lat_due_cache_nactive.push_back(0);
  lat_due_cache_nboundary_send.push_back(0);
  lat_due_cache_nboundary_send_edges.push_back(0);
  lat_due_cache_nflux_recv.push_back(0);
  if (lat_union_stage1_enabled) lat_due_cache_nupdate_flux_recv.push_back(0);

  const int nmb = std::max(1, nmb_thispack);
  const int nnghbr = pmb->nnghbr;
  const int cap = MeshBlockStorageCapacity(nmb);
  // A power-of-two schedule has exactly one distinct due-factor signature per factor, so
  // lat_factor_cache_values.size() is the largest number of slots this cache can ever
  // hold; size the lists to it on the first fill so no later phase allocates.  slot+1 is
  // carried in the bound purely as a safety net -- the lists must never be shorter than
  // the slot index about to be written.
  const int nslots = std::max(static_cast<int>(lat_factor_cache_values.size()), slot + 1);
  EnsureLATCacheSlots1D(lat_due_cache_active_mb,
      "lat_due_cache_active_mb", nslots, cap);
  EnsureLATCacheSlots1D(lat_due_cache_active_indices,
      "lat_due_cache_active_indices", nslots, cap);
  EnsureLATCacheSlots1D(lat_due_cache_boundary_send_mb,
      "lat_due_cache_boundary_send_mb", nslots, cap);
  EnsureLATCacheSlots1D(lat_due_cache_boundary_send_indices,
      "lat_due_cache_boundary_send_indices", nslots, cap);
  EnsureLATCacheSlots1D(lat_due_cache_boundary_send_edges,
      "lat_due_cache_boundary_send_edges", nslots, cap*nnghbr);
  EnsureLATCacheSlots1D(lat_due_cache_flux_recv_mb,
      "lat_due_cache_flux_recv_mb", nslots, cap);
  EnsureLATCacheSlots1D(lat_due_cache_flux_recv_indices,
      "lat_due_cache_flux_recv_indices", nslots, cap);
  if (lat_union_stage1_enabled) {
    EnsureLATCacheSlots1D(lat_due_cache_update_flux_recv_indices,
        "lat_due_cache_update_flux_recv_indices", nslots, cap);
  }
  EnsureLATCacheSlots2D(lat_due_cache_send_nghbr,
      "lat_due_cache_send_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_due_cache_flux_accum_nghbr,
      "lat_due_cache_flux_accum_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_due_cache_flux_send_nghbr,
      "lat_due_cache_flux_send_nghbr", nslots, cap, nnghbr);
  EnsureLATCacheSlots2D(lat_due_cache_flux_recv_nghbr,
      "lat_due_cache_flux_recv_nghbr", nslots, cap, nnghbr);
  if (lat_union_stage1_enabled) {
    EnsureLATCacheSlots2D(lat_due_cache_update_flux_recv_nghbr,
        "lat_due_cache_update_flux_recv_nghbr", nslots, cap, nnghbr);
  }

  int nactive = 0;
  int nboundary_send = 0;
  int nboundary_send_edges = 0;
  int nflux_recv = 0;
  int nupdate_flux_recv = 0;
  bool s_needs_restrict = false;
  bool s_needs_prolongate = false;
  bool s_needs_physical_bc = false;
  bool s_needs_neighbor_recv = false;
  const bool strictly_periodic = (pmesh != nullptr) && pmesh->strictly_periodic;
  for (int m = 0; m < nmb_thispack; ++m) {
    const int lev = pmb->mb_lev.h_view(m);
    const int factor = LATFactorForGID(pmb->mb_gid.h_view(m), capped_max_factor);
    const bool same_level_lat = (pmesh != nullptr) && pmesh->hydro_lat_same_level;
    const bool due =
        ((factor > 1 || include_factor_one) && ((phase % factor) == 0));
    const int active = due ? 1 : 0;
    lat_due_cache_active_mb[slot].h_view(m) = active;
    if (active) lat_due_cache_active_indices[slot].h_view(nactive) = m;
    lat_due_cache_boundary_send_mb[slot].h_view(m) = 0;
    lat_due_cache_flux_recv_mb[slot].h_view(m) = 0;
    bool flux_recv = false;
    bool update_flux_recv = false;
    for (int n = 0; n < nnghbr; ++n) {
      const auto &nb = pmb->nghbr.h_view(m,n);
      const int nb_factor = (nb.gid >= 0) ?
          LATFactorForGID(nb.gid, capped_max_factor) : 0;
      const bool nb_due =
          (nb.gid >= 0) && (nb_factor > 1 || include_factor_one) &&
          ((phase % nb_factor) == 0);
      const int send = nb_due ? 1 : 0;
      lat_due_cache_send_nghbr[slot].h_view(m,n) = send;
      if (send != 0) {
        lat_due_cache_boundary_send_edges[slot].h_view(nboundary_send_edges++) =
            m*nnghbr + n;
      }
      const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
      const bool same_level_mixed =
          same_level_lat && face_nghbr && nb.gid >= 0 && nb.lev == lev &&
          nb_factor != factor;
      lat_due_cache_flux_accum_nghbr[slot].h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev > lev) || (same_level_mixed && nb_factor < factor))) ? 1 : 0;
      lat_due_cache_flux_send_nghbr[slot].h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev < lev) || (same_level_mixed && factor < nb_factor))) ? 1 : 0;
      // Reflux receivers are blocks whose local timestep completes at this
      // phase.  The fine/faster neighbor may have accumulated several substeps,
      // so the receiver condition intentionally does not require nb_due.
      lat_due_cache_flux_recv_nghbr[slot].h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev > lev) ||
            (same_level_lat && nb.lev == lev && factor > nb_factor))) ? 1 : 0;
      // During a union update, a slow receiver must accept flux whenever its
      // finer/faster neighbor is due, even if the receiver itself is not due.
      if (lat_union_stage1_enabled) {
        lat_due_cache_update_flux_recv_nghbr[slot].h_view(m,n) =
            (nb_due && face_nghbr &&
             ((nb.lev > lev) ||
              (same_level_lat && nb.lev == lev && factor > nb_factor))) ? 1 : 0;
      }
      lat_due_cache_boundary_send_mb[slot].h_view(m) |= send;
      flux_recv = flux_recv || (lat_due_cache_flux_recv_nghbr[slot].h_view(m,n) != 0);
      if (lat_union_stage1_enabled) {
        update_flux_recv = update_flux_recv ||
            (lat_due_cache_update_flux_recv_nghbr[slot].h_view(m,n) != 0);
      }
      if (send && nb.gid >= 0 && nb.lev < lev) s_needs_restrict = true;
      if (active && nb.gid >= 0 && nb.lev < lev) s_needs_prolongate = true;
      if (active && nb.gid >= 0) s_needs_neighbor_recv = true;
    }
    if (lat_due_cache_boundary_send_mb[slot].h_view(m) != 0) {
      lat_due_cache_boundary_send_indices[slot].h_view(nboundary_send++) = m;
    }
    if (flux_recv) {
      lat_due_cache_flux_recv_mb[slot].h_view(m) = 1;
      lat_due_cache_flux_recv_indices[slot].h_view(nflux_recv++) = m;
    }
    if (lat_union_stage1_enabled && update_flux_recv) {
      lat_due_cache_update_flux_recv_indices[slot].h_view(nupdate_flux_recv++) = m;
    }
    if (active && !strictly_periodic) {
      for (int f = 0; f < 6; ++f) {
        const BoundaryFlag bc = pmb->mb_bcs.h_view(m,f);
        if (bc != BoundaryFlag::block && bc != BoundaryFlag::periodic &&
            bc != BoundaryFlag::shear_periodic) {
          s_needs_physical_bc = true;
          break;
        }
      }
    }
    nactive += active;
  }
  lat_due_cache_nactive[slot] = nactive;
  lat_due_cache_nboundary_send[slot] = nboundary_send;
  lat_due_cache_nboundary_send_edges[slot] = nboundary_send_edges;
  lat_due_cache_nflux_recv[slot] = nflux_recv;
  if (lat_union_stage1_enabled) {
    lat_due_cache_nupdate_flux_recv[slot] = nupdate_flux_recv;
  }
  lat_due_cache_needs_restrict.push_back(s_needs_restrict);
  lat_due_cache_needs_prolongate.push_back(s_needs_prolongate);
  lat_due_cache_needs_physical_bc.push_back(s_needs_physical_bc);
  lat_due_cache_needs_neighbor_recv.push_back(s_needs_neighbor_recv);

  lat_due_cache_active_mb[slot].template modify<HostMemSpace>();
  lat_due_cache_active_indices[slot].template modify<HostMemSpace>();
  lat_due_cache_boundary_send_mb[slot].template modify<HostMemSpace>();
  lat_due_cache_boundary_send_indices[slot].template modify<HostMemSpace>();
  lat_due_cache_boundary_send_edges[slot].template modify<HostMemSpace>();
  lat_due_cache_flux_recv_mb[slot].template modify<HostMemSpace>();
  lat_due_cache_flux_recv_indices[slot].template modify<HostMemSpace>();
  if (lat_union_stage1_enabled) {
    lat_due_cache_update_flux_recv_indices[slot].template modify<HostMemSpace>();
  }
  lat_due_cache_send_nghbr[slot].template modify<HostMemSpace>();
  lat_due_cache_flux_accum_nghbr[slot].template modify<HostMemSpace>();
  lat_due_cache_flux_send_nghbr[slot].template modify<HostMemSpace>();
  lat_due_cache_flux_recv_nghbr[slot].template modify<HostMemSpace>();
  if (lat_union_stage1_enabled) {
    lat_due_cache_update_flux_recv_nghbr[slot].template modify<HostMemSpace>();
  }
  lat_due_cache_active_mb[slot].template sync<DevExeSpace>();
  lat_due_cache_active_indices[slot].template sync<DevExeSpace>();
  lat_due_cache_boundary_send_mb[slot].template sync<DevExeSpace>();
  lat_due_cache_boundary_send_indices[slot].template sync<DevExeSpace>();
  lat_due_cache_boundary_send_edges[slot].template sync<DevExeSpace>();
  lat_due_cache_flux_recv_mb[slot].template sync<DevExeSpace>();
  lat_due_cache_flux_recv_indices[slot].template sync<DevExeSpace>();
  if (lat_union_stage1_enabled) {
    lat_due_cache_update_flux_recv_indices[slot].template sync<DevExeSpace>();
  }
  lat_due_cache_send_nghbr[slot].template sync<DevExeSpace>();
  lat_due_cache_flux_accum_nghbr[slot].template sync<DevExeSpace>();
  lat_due_cache_flux_send_nghbr[slot].template sync<DevExeSpace>();
  lat_due_cache_flux_recv_nghbr[slot].template sync<DevExeSpace>();
  if (lat_union_stage1_enabled) {
    lat_due_cache_update_flux_recv_nghbr[slot].template sync<DevExeSpace>();
  }
  return true;
}

int MeshBlockPack::SetActiveMeshBlocksByLATFactor(int max_factor, int active_factor) {
  ResizeActiveMask();
  lat_per_block_timestep = false;
  lat_active_mask_enabled = true;
  const int capped_max_factor = std::max(1, max_factor);
  const int requested_factor = std::max(1, active_factor);
  if (BuildLATFactorCache(capped_max_factor)) {
    const int slot = LATFactorCacheSlot(requested_factor);
    if (slot >= 0) {
      lat_active_mask_uses_cache = true;
      lat_nactive_thispack = lat_factor_cache_nactive[slot];
      lat_nboundary_send_thispack = lat_factor_cache_nboundary_send[slot];
      lat_nboundary_send_edges_thispack =
          lat_factor_cache_nboundary_send_edges[slot];
      lat_nflux_recv_thispack = lat_factor_cache_nflux_recv[slot];
      lat_active_mb = lat_factor_cache_active_mb[slot];
      lat_active_indices = lat_factor_cache_active_indices[slot];
      lat_boundary_send_mb = lat_factor_cache_boundary_send_mb[slot];
      lat_boundary_send_indices = lat_factor_cache_boundary_send_indices[slot];
      lat_boundary_send_edges = lat_factor_cache_boundary_send_edges[slot];
      lat_flux_recv_indices = lat_factor_cache_flux_recv_indices[slot];
      lat_send_nghbr = lat_factor_cache_send_nghbr[slot];
      lat_flux_accum_nghbr = lat_factor_cache_flux_accum_nghbr[slot];
      lat_flux_send_nghbr = lat_factor_cache_flux_send_nghbr[slot];
      lat_flux_recv_nghbr = lat_factor_cache_flux_recv_nghbr[slot];
      lat_cached_needs_restrict = lat_factor_cache_needs_restrict[slot];
      lat_cached_needs_prolongate = lat_factor_cache_needs_prolongate[slot];
      lat_cached_needs_physical_bc = lat_factor_cache_needs_physical_bc[slot];
      lat_cached_needs_neighbor_recv = lat_factor_cache_needs_neighbor_recv[slot];
      return lat_nactive_thispack;
    }
  }
  DetachLATActiveMask();
  int nactive = 0;
  int nboundary_send = 0;
  int nboundary_send_edges = 0;
  int nflux_recv = 0;
  bool needs_restrict = false;
  bool needs_prolongate = false;
  bool needs_physical_bc = false;
  bool needs_neighbor_recv = false;
  const bool strictly_periodic = (pmesh != nullptr) && pmesh->strictly_periodic;
  for (int m = 0; m < nmb_thispack; ++m) {
    const int lev = pmb->mb_lev.h_view(m);
    int factor = LATFactorForGID(pmb->mb_gid.h_view(m), capped_max_factor);
    const int active = (factor == requested_factor) ? 1 : 0;
    const bool same_level_lat = (pmesh != nullptr) && pmesh->hydro_lat_same_level;
    lat_active_mb.h_view(m) = active;
    if (active) lat_active_indices.h_view(nactive) = m;
    lat_boundary_send_mb.h_view(m) = 0;
    bool flux_recv = false;
    for (int n = 0; n < pmb->nnghbr; ++n) {
      const auto &nb = pmb->nghbr.h_view(m,n);
      const int nb_factor = (nb.gid >= 0) ?
          LATFactorForGID(nb.gid, capped_max_factor) : 0;
      const int send = (nb.gid >= 0 && nb_factor == requested_factor) ? 1 : 0;
      lat_send_nghbr.h_view(m,n) = send;
      if (send != 0) {
        lat_boundary_send_edges.h_view(nboundary_send_edges++) =
            m*pmb->nnghbr + n;
      }
      const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
      const bool same_level_mixed =
          same_level_lat && face_nghbr && nb.gid >= 0 && nb.lev == lev &&
          nb_factor != factor;
      lat_flux_accum_nghbr.h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev > lev) || (same_level_mixed && nb_factor < factor))) ? 1 : 0;
      lat_flux_send_nghbr.h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev < lev) || (same_level_mixed && factor < nb_factor))) ? 1 : 0;
      lat_flux_recv_nghbr.h_view(m,n) =
          (face_nghbr && nb.gid >= 0 && nb_factor == requested_factor &&
           ((nb.lev > lev) ||
            (same_level_lat && nb.lev == lev && factor > requested_factor))) ? 1 : 0;
      lat_boundary_send_mb.h_view(m) |= send;
      flux_recv = flux_recv || (lat_flux_recv_nghbr.h_view(m,n) != 0);
      // Accumulate boundary-need flags from the neighbor scan we are already doing
      if (send && nb.gid >= 0 && nb.lev < lev) needs_restrict = true;
      if (active && nb.gid >= 0 && nb.lev < lev) needs_prolongate = true;
      if (active && nb.gid >= 0) needs_neighbor_recv = true;
    }
    if (lat_boundary_send_mb.h_view(m) != 0) lat_boundary_send_indices.h_view(nboundary_send++) = m;
    if (flux_recv) lat_flux_recv_indices.h_view(nflux_recv++) = m;
    // Check physical BCs for active blocks
    if (active && !strictly_periodic) {
      for (int f = 0; f < 6; ++f) {
        const BoundaryFlag bc = pmb->mb_bcs.h_view(m,f);
        if (bc != BoundaryFlag::block && bc != BoundaryFlag::periodic &&
            bc != BoundaryFlag::shear_periodic) {
          needs_physical_bc = true;
          break;
        }
      }
    }
    nactive += active;
  }
  lat_nactive_thispack = nactive;
  lat_nboundary_send_thispack = nboundary_send;
  lat_nboundary_send_edges_thispack = nboundary_send_edges;
  lat_nflux_recv_thispack = nflux_recv;
  lat_cached_needs_restrict = needs_restrict;
  lat_cached_needs_prolongate = needs_prolongate;
  lat_cached_needs_physical_bc = needs_physical_bc;
  lat_cached_needs_neighbor_recv = needs_neighbor_recv;
  lat_active_mb.template modify<HostMemSpace>();
  lat_active_indices.template modify<HostMemSpace>();
  lat_boundary_send_mb.template modify<HostMemSpace>();
  lat_boundary_send_indices.template modify<HostMemSpace>();
  lat_boundary_send_edges.template modify<HostMemSpace>();
  lat_flux_recv_indices.template modify<HostMemSpace>();
  lat_send_nghbr.template modify<HostMemSpace>();
  lat_flux_accum_nghbr.template modify<HostMemSpace>();
  lat_flux_send_nghbr.template modify<HostMemSpace>();
  lat_flux_recv_nghbr.template modify<HostMemSpace>();
  lat_active_mb.template sync<DevExeSpace>();
  lat_active_indices.template sync<DevExeSpace>();
  lat_boundary_send_mb.template sync<DevExeSpace>();
  lat_boundary_send_indices.template sync<DevExeSpace>();
  lat_boundary_send_edges.template sync<DevExeSpace>();
  lat_flux_recv_indices.template sync<DevExeSpace>();
  lat_send_nghbr.template sync<DevExeSpace>();
  lat_flux_accum_nghbr.template sync<DevExeSpace>();
  lat_flux_send_nghbr.template sync<DevExeSpace>();
  lat_flux_recv_nghbr.template sync<DevExeSpace>();
  return nactive;
}

int MeshBlockPack::SetActiveMeshBlocksByLATDueFactors(int max_factor, int tick_cycle,
                                                      bool include_factor_one) {
  ResizeActiveMask();
  lat_per_block_timestep = false;
  lat_active_mask_enabled = true;
  const int capped_max_factor = std::max(1, max_factor);
  const int tick = std::max(0, tick_cycle);
  const int phase = tick % capped_max_factor;
  if (BuildLATDueFactorCache(capped_max_factor, tick, include_factor_one)) {
    const std::uint64_t signature =
        LATDueFactorSignature(phase, include_factor_one);
    const int slot = LATDueFactorCacheSlot(signature);
    if (slot >= 0) {
      lat_active_mask_uses_cache = true;
      lat_nactive_thispack = lat_due_cache_nactive[slot];
      lat_nboundary_send_thispack = lat_due_cache_nboundary_send[slot];
      lat_nboundary_send_edges_thispack =
          lat_due_cache_nboundary_send_edges[slot];
      lat_nflux_recv_thispack = lat_due_cache_nflux_recv[slot];
      lat_active_mb = lat_due_cache_active_mb[slot];
      lat_active_indices = lat_due_cache_active_indices[slot];
      lat_boundary_send_mb = lat_due_cache_boundary_send_mb[slot];
      lat_boundary_send_indices = lat_due_cache_boundary_send_indices[slot];
      lat_boundary_send_edges = lat_due_cache_boundary_send_edges[slot];
      lat_flux_recv_indices = lat_due_cache_flux_recv_indices[slot];
      lat_send_nghbr = lat_due_cache_send_nghbr[slot];
      lat_flux_accum_nghbr = lat_due_cache_flux_accum_nghbr[slot];
      lat_flux_send_nghbr = lat_due_cache_flux_send_nghbr[slot];
      lat_flux_recv_nghbr = lat_due_cache_flux_recv_nghbr[slot];
      lat_cached_needs_restrict = lat_due_cache_needs_restrict[slot];
      lat_cached_needs_prolongate = lat_due_cache_needs_prolongate[slot];
      lat_cached_needs_physical_bc = lat_due_cache_needs_physical_bc[slot];
      lat_cached_needs_neighbor_recv = lat_due_cache_needs_neighbor_recv[slot];
      return lat_nactive_thispack;
    }
  }
  DetachLATActiveMask();
  int nactive = 0;
  int nboundary_send = 0;
  int nboundary_send_edges = 0;
  int nflux_recv = 0;
  bool needs_restrict = false;
  bool needs_prolongate = false;
  bool needs_physical_bc = false;
  bool needs_neighbor_recv = false;
  const bool strictly_periodic = (pmesh != nullptr) && pmesh->strictly_periodic;
  for (int m = 0; m < nmb_thispack; ++m) {
    const int lev = pmb->mb_lev.h_view(m);
    const int factor = LATFactorForGID(pmb->mb_gid.h_view(m), capped_max_factor);
    const bool same_level_lat = (pmesh != nullptr) && pmesh->hydro_lat_same_level;
    const bool due = ((factor > 1 || include_factor_one) && ((tick % factor) == 0));
    const int active = due ? 1 : 0;
    lat_active_mb.h_view(m) = active;
    if (active) lat_active_indices.h_view(nactive) = m;
    lat_boundary_send_mb.h_view(m) = 0;
    bool flux_recv = false;
    for (int n = 0; n < pmb->nnghbr; ++n) {
      const auto &nb = pmb->nghbr.h_view(m,n);
      const int nb_factor = (nb.gid >= 0) ?
          LATFactorForGID(nb.gid, capped_max_factor) : 0;
      const bool nb_due =
          (nb.gid >= 0) && (nb_factor > 1 || include_factor_one) &&
          ((tick % nb_factor) == 0);
      const int send = nb_due ? 1 : 0;
      lat_send_nghbr.h_view(m,n) = send;
      if (send != 0) {
        lat_boundary_send_edges.h_view(nboundary_send_edges++) =
            m*pmb->nnghbr + n;
      }
      const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
      const bool same_level_mixed =
          same_level_lat && face_nghbr && nb.gid >= 0 && nb.lev == lev &&
          nb_factor != factor;
      lat_flux_accum_nghbr.h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev > lev) || (same_level_mixed && nb_factor < factor))) ? 1 : 0;
      lat_flux_send_nghbr.h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev < lev) || (same_level_mixed && factor < nb_factor))) ? 1 : 0;
      lat_flux_recv_nghbr.h_view(m,n) =
          (active && face_nghbr && nb.gid >= 0 &&
           ((nb.lev > lev) ||
            (same_level_lat && nb.lev == lev && factor > nb_factor))) ? 1 : 0;
      lat_boundary_send_mb.h_view(m) |= send;
      flux_recv = flux_recv || (lat_flux_recv_nghbr.h_view(m,n) != 0);
      // Accumulate boundary-need flags from the neighbor scan we are already doing
      if (send && nb.gid >= 0 && nb.lev < lev) needs_restrict = true;
      if (active && nb.gid >= 0 && nb.lev < lev) needs_prolongate = true;
      if (active && nb.gid >= 0) needs_neighbor_recv = true;
    }
    if (lat_boundary_send_mb.h_view(m) != 0) lat_boundary_send_indices.h_view(nboundary_send++) = m;
    if (flux_recv) lat_flux_recv_indices.h_view(nflux_recv++) = m;
    // Check physical BCs for active blocks
    if (active && !strictly_periodic) {
      for (int f = 0; f < 6; ++f) {
        const BoundaryFlag bc = pmb->mb_bcs.h_view(m,f);
        if (bc != BoundaryFlag::block && bc != BoundaryFlag::periodic &&
            bc != BoundaryFlag::shear_periodic) {
          needs_physical_bc = true;
          break;
        }
      }
    }
    nactive += active;
  }
  lat_nactive_thispack = nactive;
  lat_nboundary_send_thispack = nboundary_send;
  lat_nboundary_send_edges_thispack = nboundary_send_edges;
  lat_nflux_recv_thispack = nflux_recv;
  lat_cached_needs_restrict = needs_restrict;
  lat_cached_needs_prolongate = needs_prolongate;
  lat_cached_needs_physical_bc = needs_physical_bc;
  lat_cached_needs_neighbor_recv = needs_neighbor_recv;
  lat_active_mb.template modify<HostMemSpace>();
  lat_active_indices.template modify<HostMemSpace>();
  lat_boundary_send_mb.template modify<HostMemSpace>();
  lat_boundary_send_indices.template modify<HostMemSpace>();
  lat_boundary_send_edges.template modify<HostMemSpace>();
  lat_flux_recv_indices.template modify<HostMemSpace>();
  lat_send_nghbr.template modify<HostMemSpace>();
  lat_flux_accum_nghbr.template modify<HostMemSpace>();
  lat_flux_send_nghbr.template modify<HostMemSpace>();
  lat_flux_recv_nghbr.template modify<HostMemSpace>();
  lat_active_mb.template sync<DevExeSpace>();
  lat_active_indices.template sync<DevExeSpace>();
  lat_boundary_send_mb.template sync<DevExeSpace>();
  lat_boundary_send_indices.template sync<DevExeSpace>();
  lat_boundary_send_edges.template sync<DevExeSpace>();
  lat_flux_recv_indices.template sync<DevExeSpace>();
  lat_send_nghbr.template sync<DevExeSpace>();
  lat_flux_accum_nghbr.template sync<DevExeSpace>();
  lat_flux_send_nghbr.template sync<DevExeSpace>();
  lat_flux_recv_nghbr.template sync<DevExeSpace>();
  return nactive;
}

int MeshBlockPack::SelectLATDueUpdateFluxReceivers(int max_factor, int tick_cycle,
                                                   bool include_factor_one) {
  if (!lat_union_stage1_enabled) {
    lat_nflux_recv_thispack = 0;
    return 0;
  }
  const int capped_max_factor = std::max(1, max_factor);
  const int tick = std::max(0, tick_cycle);
  const int phase = tick % capped_max_factor;
  if (!BuildLATDueFactorCache(capped_max_factor, tick, include_factor_one)) {
    lat_nflux_recv_thispack = 0;
    return 0;
  }

  const std::uint64_t signature =
      LATDueFactorSignature(phase, include_factor_one);
  const int slot = LATDueFactorCacheSlot(signature);
  if (slot < 0) {
    lat_nflux_recv_thispack = 0;
    return 0;
  }

  lat_nflux_recv_thispack = lat_due_cache_nupdate_flux_recv[slot];
  lat_flux_recv_indices = lat_due_cache_update_flux_recv_indices[slot];
  lat_flux_recv_nghbr = lat_due_cache_update_flux_recv_nghbr[slot];
  return lat_nflux_recv_thispack;
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack() {
  if (psink  != nullptr) {delete psink;}
  if (pgrav  != nullptr) {delete pgrav;}
  if (ppart  != nullptr) {delete ppart;}
  if (pnr    != nullptr) {delete pnr;}
  if (pdyngr != nullptr) {delete pdyngr;}
  if (ptmunu != nullptr) {delete ptmunu;}
  if (padm   != nullptr) {delete padm;}
  if (pz4c   != nullptr) {
    delete pz4c;
    // cce dump
    for (auto cce : pz4c_cce) {
      delete cce;
    }
    pz4c_cce.resize(0);
  }
  if (pturb  != nullptr) {delete pturb;}
  if (prad   != nullptr) {delete prad;}
  if (pionn  != nullptr) {delete pionn;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (phydro != nullptr) {delete phydro;}
  if (punit  != nullptr) {delete punit;}
  delete pcoord;
  delete pmb;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddMeshBlocks(ParameterInput *pin)
//! \brief Wrapper function for calling MeshBlock constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack.

void MeshBlockPack::AddMeshBlocks(ParameterInput *pin) {
  pmb = new MeshBlock(this, gids, nmb_thispack);
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddCoordinates(ParameterInput *pin)
//! \brief Wrapper function for calling Coordinates constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack. Must be called BEFORE AddPhysics()
//! function, since latter uses data inside Coordinates class.

void MeshBlockPack::AddCoordinates(ParameterInput *pin) {
  pcoord = new Coordinates(pin, this);
}

//----------------------------------------------------------------------------------------
//! \fn Real MeshBlockPack::TemperatureUnitCGS()
//! \brief Return the code temperature unit in K.
//!
//! THE RULE: the code temperature unit is the EOS's.  EOS_Data::temp_unit_cgs is
//! the cgs value of the number every kernel in the run calls "temperature", whichever
//! EOS is active: the ideal EOS stores Units::temperature_cgs() there, while a
//! tabulated LTE/Saha EOS builds its own scale v_cgs^2*m_H/k_B (mu = 1) because the
//! table's own temperature axis is in K and knows nothing about <units>/mu.
//! Units::temperature_cgs() is therefore the IDEAL-GAS unit, carrying the deck's mu,
//! and it agrees with the active unit only for the ideal EOS -- it is wrong by
//! mu*m_u/m_H under a tabulated EOS (0.78% in T, 3.1% in a_R T^4 at mu = 1).
//! Consumers must use this accessor and never punit->temperature_cgs().
//!
//! Fatal without a fluid: with no <hydro> and no <mhd> there is no EOS, hence no
//! temperature unit to report.  Hydro and MHD are both constructed (with their EOS)
//! before radiation, source terms, diffusion and the problem generator run, so no
//! legitimate consumer can reach this before the unit exists.

Real MeshBlockPack::TemperatureUnitCGS() const {
  const bool has_hydro = (phydro != nullptr && phydro->peos != nullptr);
  const bool has_mhd = (pmhd != nullptr && pmhd->peos != nullptr);
  if (!has_hydro && !has_mhd) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "MeshBlockPack::TemperatureUnitCGS() needs an equation of state, but "
                 "neither <hydro> nor <mhd> has been constructed. Either the deck runs "
                 "no fluid, or a physics module read the temperature unit before the "
                 "fluid was built." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (has_hydro && has_mhd) {
    // The <ion-neutral> two-fluid configuration.  Both EOS derive the unit from the
    // same <units> block, so a disagreement is unreachable; it is checked rather than
    // assumed because picking the wrong fluid's unit here would be silent.
    const Real th = phydro->peos->eos_data.temp_unit_cgs;
    const Real tm = pmhd->peos->eos_data.temp_unit_cgs;
    if (th != tm) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<hydro> and <mhd> report different code temperature units ("
                << th << " K and " << tm << " K). There is then no single unit for the "
                   "temperature the radiation and source terms exchange with the gas."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return th;
  }
  return has_hydro ? phydro->peos->eos_data.temp_unit_cgs
                   : pmhd->peos->eos_data.temp_unit_cgs;
}

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysics()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysics(ParameterInput *pin) {
  int nphysics = 0;
  TaskID none(0);
  bool assemble_single_mhd_tasks = false;

  // (1) Units.  Create first so that they can be used in other physics constructors
  // Default units are simply code units
  if (pin->DoesBlockExist("units")) {
    punit = new units::Units(pin);
  } else {
    punit = nullptr;
  }

  // (2) HYDRODYNAMICS
  // Create Hydro physics module.  Create TaskLists only for single-fluid hydro
  // (Note TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("mhd")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      phydro->AssembleHydroTasks(tl_map);
    }
  } else {
    phydro = nullptr;
  }

  // (3) MHD
  // Create MHD physics module.  Create TaskLists only for single-fluid MHD
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("hydro")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      assemble_single_mhd_tasks = true;
    }
  } else {
    pmhd = nullptr;
  }

  // (4) ION_NEUTRAL (two-fluid) MHD
  // Create Ion-Neutral physics module and TaskLists. Error if <hydro> and <mhd> are not
  // both defined as well.
  if (pin->DoesBlockExist("ion-neutral")) {
    pionn = new ion_neutral::IonNeutral(this, pin);   // construct new MHD object
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd") &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      pionn->AssembleIonNeutralTasks(tl_map);
      nphysics++;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<ion-neutral> block detected in input file, but either"
                << " <hydro> or <mhd> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // Error if both <hydro> and <mhd> defined, but not <ion-neutral>
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Both <hydro> and <mhd> blocks detected in input file, "
                << "but <ion-neutral> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pionn = nullptr;
  }

  // (5) RADIATION
  // Create radiation physics module.  Create tasklist.
  if (pin->DoesBlockExist("radiation")) {
    prad = new radiation::Radiation(this, pin);
    nphysics++;
    prad->AssembleRadTasks(tl_map);
  } else {
    prad = nullptr;
  }

  if (assemble_single_mhd_tasks) {
    pmhd->AssembleMHDTasks(tl_map);
  }

  // (6) TURBULENCE DRIVER
  // This is a special module to drive turbulence in hydro, MHD, or both. Cannot be
  // included as a source term since it requires evolving force array via O-U process.
  // Instead, TurbulenceDriver object is stored in MeshBlockPack and tasks for evolving
  // force and adding force to fluid are included in operator_split and stage_run
  // task lists respectively.
  if (pin->DoesBlockExist("turb_driving")) {
    pturb = new TurbulenceDriver(this, pin);
    pturb->IncludeInitializeModesTask(tl_map["before_timeintegrator"], none);
    pturb->IncludeAddForcingTask(tl_map["stagen"], none);
  } else {
    pturb = nullptr;
  }

  // (7) Z4c and ADM
  // Create Z4c and ADM physics module.
  if (pin->DoesBlockExist("z4c")) {
    pz4c = new z4c::Z4c(this, pin);
    padm = new adm::ADM(this, pin);
    ptmunu = nullptr;
    // init cce dump
    pz4c_cce.reserve(0);
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pz4c_cce.reserve(ncce);// 10 different components for each radius
    for(int n = 0; n < ncce; ++n) {
      // NOTE: these names are used for pittnull code, so DON'T change the convention
      pz4c_cce.push_back(new z4c::CCE(pmesh, pin,n));
    }
    nphysics++;
  } else {
    pz4c = nullptr;
    if (pin->DoesBlockExist("adm")) {
      padm = new adm::ADM(this, pin);
    } else {
      padm = nullptr;
    }
  }

  // (8) Dynamical Spacetime and Matter (MHD TODO)
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("hydro")) ) {
    std::cout << "Dynamical metric and hydro not compatible; use MHD instead  "
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("mhd")) ) {
    pdyngr = dyngr::BuildDynGRMHD(this, pin);
    // Tmunu is the matter source of the Einstein equations, so its only evolution
    // consumers are the Z4c RHS/ADM routines -- and DynGRMHD
    // queues MHD_SetTmunu only when pz4c != nullptr.  With a prescribed
    // ADM metric nothing ever reads it: DynGRMHDPS::PrimToConInit fills it once at
    // initialization and it is dead thereafter.  Ten ghosted fields sized by
    // max(nmb_thispack, nmb_maxperrank) is a large device reservation for a write-once
    // array, so allocate it only when a real consumer exists.  This used to be keyed on
    // LAT being off, which reserved it in exactly the prescribed-metric runs that cannot
    // use it.
    if (pz4c != nullptr) {
      ptmunu = new Tmunu(this, pin);
    }
  }

  if (pz4c != nullptr || padm != nullptr) {
    pnr = new numrel::NumericalRelativity(this, pin);
    pnr->AssembleNumericalRelativityTasks(tl_map);
  }

  // (8) PARTICLES
  // Create particles module.  Create tasklist.
  if (pin->DoesBlockExist("particles")) {
    ppart = new particles::Particles(this, pin);
    ppart->AssembleTasks(tl_map);
    nphysics++;
  } else {
    ppart = nullptr;
  }

  // (9) GRAVITY
  // Create gravity physics module.  Create tasklist.
  bool gravity_self_gravity =
      (pin->DoesBlockExist("gravity") &&
       pin->GetOrAddBoolean("gravity", "self_gravity", true));
  if (gravity_self_gravity) {
    // Gravity module uses Multigrid module
    pgrav = new gravity::Gravity(this, pin);
    //pgrav->AssembleTasks(tl_map);
    nphysics++;
  } else {
    pgrav = nullptr;
  }

  // (10) SINK PARTICLES
  // Constructed after Gravity so the module can validate the <gravity> requirement of
  // sink creation itself, and after Hydro so it can size its accretion kernels.
  if (pin->DoesBlockExist("sink_particles")) {
    if (phydro == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<sink_particles> requires a <hydro> block: accretion and the "
                   "sink-gravity source term act on the hydro conserved variables."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (phydro->psrc == nullptr) {
      // Hydro builds its SourceTerms object before AddPhysics reaches this block, and
      // only when <hydro_srcterms>/<gravity> asked for it.  Without it there is no place
      // to apply the sink->gas kick, so refuse rather than drop the term silently.
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<sink_particles> requires hydro source terms. Add a "
                   "<hydro_srcterms> block (it may be empty) or enable "
                   "<gravity>/self_gravity so the sink gravity kick has a carrier."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    psink = new sinkparticles::SinkParticles(this, pin);
    psink->AssembleSinkTasks(tl_map);
    nphysics++;
  } else {
    psink = nullptr;
  }

  // Check that at least ONE is requested and initialized.
  // Error if there are no physics blocks in the input file.
  if (nphysics == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "At least one physics module must be specified in input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return;
}
