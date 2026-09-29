//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file flux_correction_cc.cpp
//! \brief functions to pack/send and recv/unpack fluxes for cell-centered variables at
//! fine/coarse boundaries for the flux correction step.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
int FluxFaceCount(const int nnghbr) {
  int count = (nnghbr < 16) ? nnghbr : 16;
  if (nnghbr > 24) {
    count += ((nnghbr - 24) < 8) ? (nnghbr - 24) : 8;
  }
  return count;
}

KOKKOS_INLINE_FUNCTION
int FluxFaceNeighborIndex(const int slot, const int nnghbr) {
  const int first_count = (nnghbr < 16) ? nnghbr : 16;
  return (slot < first_count) ? slot : 24 + (slot - first_count);
}

// Entries of MeshBoundaryValuesCC::lat_pending_edge_, one per cell on two faces of the
// interior of a MeshBlock (on an edge; on a corner in 2D).  Edge e < 4 runs along x1 and
// holds the corners, 4 <= e < 8 along x2 and 8 <= e < 12 along x3; p is the position
// along it.
constexpr int kPendingEdgeCount = 12;
constexpr int kPendingEdgeTested = 1;     // a cell both kinds of face meet
constexpr int kPendingEdgeFlag = 2;       // its FOFC flag without the estimate
constexpr int kPendingEdgeWithhold = 4;   // its faces go without the estimate
constexpr int kPendingEdgeScalar = 8;     // << n: scalar n's flag without the estimate

//! The entry (e, p) of cell (i, j, k), if it is on two faces.
KOKKOS_INLINE_FUNCTION
bool PendingEdgeEntry(const int i, const int j, const int k, const int is, const int ie,
                      const int js, const int je, const int ks, const int ke,
                      const bool three_d, int &e, int &p) {
  const int bi = (i == is) ? 0 : ((i == ie) ? 1 : -1);
  const int bj = (j == js) ? 0 : ((j == je) ? 1 : -1);
  const int bk = !three_d ? -1 : ((k == ks) ? 0 : ((k == ke) ? 1 : -1));
  if (bj >= 0 && bk >= 0) {
    e = 2*bj + bk;
    p = i - is;
  } else if (bi >= 0 && bk >= 0) {
    e = 4 + 2*bi + bk;
    p = j - js;
  } else if (bi >= 0 && bj >= 0) {
    e = 8 + 2*bi + bj;
    p = k - ks;
  } else {
    return false;
  }
  return true;
}

//! The cell (i, j, k) of entry (e, p), if the entry names one.
KOKKOS_INLINE_FUNCTION
bool PendingEdgeCell(const int e, const int p, const int is, const int ie, const int js,
                     const int je, const int ks, const int ke, const bool three_d,
                     int &i, int &j, int &k) {
  const int a = (e % 4)/2, b = e % 2;
  if (e < 4) {
    i = is + p;
    j = a ? je : js;
    k = b ? ke : ks;
    return three_d && i <= ie;
  } else if (e < 8) {
    i = a ? ie : is;
    j = js + p;
    k = b ? ke : ks;
    return three_d && j > js && j < je;
  }
  i = a ? ie : is;
  j = b ? je : js;
  k = ks + p;
  return three_d ? (k > ks && k < ke) : (p == 0);
}

#if MPI_PARALLEL_ENABLED
struct LATFluxPackedEntry {
  int m;
  int n;
  int key_lid;
  int key_slot;
  int data_size;
};

bool FluxEntryKeyLess(const LATFluxPackedEntry &a, const LATFluxPackedEntry &b) {
  if (a.key_lid != b.key_lid) return a.key_lid < b.key_lid;
  return a.key_slot < b.key_slot;
}
#endif
} // namespace

#if MPI_PARALLEL_ENABLED
//----------------------------------------------------------------------------------------
//! \brief Build or select an exact per-peer aggregate layout for this LAT flux phase.
//!
//! Entries are ordered by the receiving rank's (local MeshBlock, neighbor slot).  The
//! sender obtains that key from the neighbor descriptor, while the receiver already owns
//! it.  Thus both sides construct the same payload order without exchanging headers.

void MeshBoundaryValuesCC::PrepareLATRankPackedFluxLayout(const int nvars) {
  if (!(pmy_pack->lat_active_mask_enabled) || global_variable::nranks <= 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid LAT rank-packed flux layout request" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  const int my_rank = global_variable::my_rank;
  const bool same_level_lat = pmy_pack->pmesh->hydro_lat_same_level;
  const std::uint64_t topology_version = pmy_pack->pmesh->topology_version;
  const std::uint64_t lat_metadata_version =
      pmy_pack->pmesh->hydro_lat_metadata_version;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;

  const bool rebuild =
      (rank_packed_flux_nvars_ != nvars) ||
      (rank_packed_flux_nmb_ != nmb) ||
      (rank_packed_flux_nnghbr_ != nnghbr) ||
      (rank_packed_flux_same_level_ != same_level_lat) ||
      (rank_packed_flux_topology_version_ != topology_version);
  if (rebuild) {
    rank_packed_flux_nvars_ = nvars;
    rank_packed_flux_nmb_ = nmb;
    rank_packed_flux_nnghbr_ = nnghbr;
    rank_packed_flux_same_level_ = same_level_lat;
    rank_packed_flux_topology_version_ = topology_version;
    // The persistent requests below are bound to these layouts' offsets and to the
    // aggregate buffers that may be reallocated a few lines down; release them first.
    FreeLayoutPersistentFluxReqs();
    lat_flux_layouts_.clear();
    lat_flux_layout_cache_.clear();
    lat_flux_layout_index_ = -1;

    // Allocate once for the full off-rank topology.  Exact LAT layouts are subsets, so
    // changing ticks never reallocates storage referenced by an in-flight MPI request.
    int send_capacity = 0;
    int recv_capacity = 0;
    for (int m=0; m<nmb; ++m) {
      const int lev = mblev.h_view(m);
      for (int slot=0; slot<nflux_nghbr; ++slot) {
        const int n = FluxFaceNeighborIndex(slot, nnghbr);
        const auto &nb = nghbr.h_view(m,n);
        if (nb.gid < 0 || nb.rank == my_rank) continue;
        if (nb.lev < lev) {
          send_capacity += nvars*sendbuf[n].iflxc_ndat;
        } else if (same_level_lat && nb.lev == lev) {
          send_capacity += nvars*sendbuf[n].iflxs_ndat;
        }
        if (nb.lev > lev) {
          recv_capacity += nvars*recvbuf[n].iflxc_ndat;
        } else if (same_level_lat && nb.lev == lev) {
          recv_capacity += nvars*recvbuf[n].iflxs_ndat;
        }
      }
    }
    // Grow-only, MiB-granular capacity, and Kokkos::realloc rather than `x = View(...)`:
    // the assignment form constructed the new allocation while the old one was still
    // referenced, so the old buffer, the new buffer and the peer buffer coexisted for a
    // moment (~92 MiB here).  Kokkos::realloc drops the old allocation first.
    //
    // Keeping the allocation across a rebuild is safe for the same reasons as the
    // variable buffers: every offset in every layout is recomputed below, the persistent
    // requests bound to these buffers were released by the FreeLayoutPersistentFluxReqs()
    // above (and are recreated against the new base pointer on first use), and the only
    // extent consumer is the capacity check further below.
    if (send_capacity > rank_sendbuf_flux_cap_) {
      rank_sendbuf_flux_cap_ = RankPackedBufCapacity(send_capacity);
      Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                      rank_sendbuf_flux_, rank_sendbuf_flux_cap_);
    }
    if (recv_capacity > rank_recvbuf_flux_cap_) {
      rank_recvbuf_flux_cap_ = RankPackedBufCapacity(recv_capacity);
      Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                      rank_recvbuf_flux_, rank_recvbuf_flux_cap_);
    }
  } else if (!lat_flux_layouts_.empty() &&
             lat_flux_layouts_.front().lat_metadata_version != lat_metadata_version) {
    FreeLayoutPersistentFluxReqs();
    lat_flux_layouts_.clear();
    lat_flux_layout_cache_.clear();
    lat_flux_layout_index_ = -1;
  }

  auto &lat_flux_send = pmy_pack->lat_flux_send_nghbr;
  auto &lat_flux_recv = pmy_pack->lat_flux_recv_nghbr;
  const bool cache_key_valid = pmy_pack->LATActiveMaskUsesCache();
  const std::uint64_t cache_generation = pmy_pack->LATCacheGeneration();
  if (cache_key_valid && lat_flux_layout_cache_generation_ != cache_generation) {
    lat_flux_layout_cache_.clear();
    lat_flux_layout_cache_generation_ = cache_generation;
  }
  const LATLayoutCacheKey cache_key = {cache_generation,
      reinterpret_cast<std::uintptr_t>(lat_flux_send.h_view.data()),
      reinterpret_cast<std::uintptr_t>(lat_flux_recv.h_view.data())};
  if (cache_key_valid) {
    const auto cached = lat_flux_layout_cache_.find(cache_key);
    if (cached != lat_flux_layout_cache_.end()) {
      lat_flux_layout_index_ = cached->second;
      return;
    }
  }

  std::vector<int> send_entry_ids;
  std::vector<int> recv_entry_ids;
  for (int m=0; m<nmb; ++m) {
    const int lev = mblev.h_view(m);
    for (int slot=0; slot<nflux_nghbr; ++slot) {
      const int n = FluxFaceNeighborIndex(slot, nnghbr);
      const auto &nb = nghbr.h_view(m,n);
      if (nb.gid < 0 || nb.rank == my_rank) continue;
      const int entry_id = m*nnghbr + n;
      const bool send_eligible = (nb.lev < lev) ||
          (same_level_lat && nb.lev == lev);
      const bool recv_eligible = (nb.lev > lev) ||
          (same_level_lat && nb.lev == lev);
      if (send_eligible && lat_flux_send.h_view(m,n) != 0) {
        send_entry_ids.push_back(entry_id);
      }
      if (recv_eligible && lat_flux_recv.h_view(m,n) != 0) {
        recv_entry_ids.push_back(entry_id);
      }
    }
  }

  std::uint64_t selection_hash = 1469598103934665603ULL;
  auto hash_id = [&selection_hash](const std::uint64_t value) {
    selection_hash ^= value;
    selection_hash *= 1099511628211ULL;
  };
  hash_id(static_cast<std::uint64_t>(send_entry_ids.size()));
  for (const int id : send_entry_ids) hash_id(static_cast<std::uint64_t>(id) + 1);
  hash_id(0xffffffffffffffffULL);
  hash_id(static_cast<std::uint64_t>(recv_entry_ids.size()));
  for (const int id : recv_entry_ids) hash_id(static_cast<std::uint64_t>(id) + 1);

  for (std::size_t i=0; i<lat_flux_layouts_.size(); ++i) {
    const auto &layout = lat_flux_layouts_[i];
    if (layout.topology_version == topology_version &&
        layout.lat_metadata_version == lat_metadata_version &&
        layout.selection_hash == selection_hash &&
        layout.send_entry_ids == send_entry_ids &&
        layout.recv_entry_ids == recv_entry_ids) {
      lat_flux_layout_index_ = static_cast<int>(i);
      if (cache_key_valid) {
        lat_flux_layout_cache_[cache_key] = lat_flux_layout_index_;
      }
      return;
    }
  }

  RankPackedFluxLayout layout;
  layout.topology_version = topology_version;
  layout.lat_metadata_version = lat_metadata_version;
  layout.selection_hash = selection_hash;
  layout.integrated = false;
  layout.send_entry_ids = std::move(send_entry_ids);
  layout.recv_entry_ids = std::move(recv_entry_ids);
  // Two nmb*nnghbr int maps per layout, taken as two rows of a persistent pool instead of
  // a fresh allocation pair.  The row is fixed by this layout's position in
  // lat_flux_layouts_, which only grows by push_back or is cleared wholesale, so the pool
  // can only be reshaped while no layout holds a row.  Invalidation is unchanged: the
  // clears above still key on the topology rebuild and on lat_metadata_version.
  const int map_len = std::max(1, nmb*nnghbr);
  layout.offsets_row = 2*static_cast<int>(lat_flux_layouts_.size());
  layout.send_offsets = lat_flux_offset_pool_.Acquire("lat_flux_send_offsets",
                                                      layout.offsets_row, map_len);
  layout.recv_offsets = lat_flux_offset_pool_.Acquire("lat_flux_recv_offsets",
                                                      layout.offsets_row + 1, map_len);
  auto send_offsets_h = Kokkos::create_mirror_view(layout.send_offsets);
  auto recv_offsets_h = Kokkos::create_mirror_view(layout.recv_offsets);
  for (int i=0; i<map_len; ++i) {
    send_offsets_h(i) = -1;
    recv_offsets_h(i) = -1;
  }

  std::map<int, std::vector<LATFluxPackedEntry>> send_by_rank;
  for (const int id : layout.send_entry_ids) {
    const int m = id/nnghbr;
    const int n = id - m*nnghbr;
    const int lev = mblev.h_view(m);
    const auto &nb = nghbr.h_view(m,n);
    const bool same_level = (nb.lev == lev);
    LATFluxPackedEntry entry;
    entry.m = m;
    entry.n = n;
    entry.key_lid = nb.gid - pmy_pack->pmesh->gids_eachrank[nb.rank];
    entry.key_slot = nb.dest;
    entry.data_size = nvars*(same_level ? sendbuf[n].iflxs_ndat :
                                           sendbuf[n].iflxc_ndat);
    send_by_rank[nb.rank].push_back(entry);
  }

  int send_total = 0;
  int send_entry_offset = 0;
  for (auto &peer : send_by_rank) {
    auto &entries = peer.second;
    std::sort(entries.begin(), entries.end(), FluxEntryKeyLess);
    RankPackedVarMessage msg;
    msg.rank = peer.first;
    msg.nentries = static_cast<int>(entries.size());
    msg.entry_offset = send_entry_offset;
    msg.hdr_offset = -1;
    msg.offset = send_total;
    for (const auto &entry : entries) {
      send_offsets_h(entry.m*nnghbr + entry.n) = send_total;
      send_total += entry.data_size;
    }
    msg.data_size = send_total - msg.offset;
    send_entry_offset += msg.nentries;
    layout.send_msgs.push_back(msg);
  }

  std::map<int, std::vector<LATFluxPackedEntry>> recv_by_rank;
  for (const int id : layout.recv_entry_ids) {
    const int m = id/nnghbr;
    const int n = id - m*nnghbr;
    const int lev = mblev.h_view(m);
    const auto &nb = nghbr.h_view(m,n);
    const bool same_level = (nb.lev == lev);
    LATFluxPackedEntry entry;
    entry.m = m;
    entry.n = n;
    entry.key_lid = m;
    entry.key_slot = n;
    entry.data_size = nvars*(same_level ? recvbuf[n].iflxs_ndat :
                                           recvbuf[n].iflxc_ndat);
    recv_by_rank[nb.rank].push_back(entry);
  }

  int recv_total = 0;
  int recv_entry_offset = 0;
  for (auto &peer : recv_by_rank) {
    auto &entries = peer.second;
    std::sort(entries.begin(), entries.end(), FluxEntryKeyLess);
    RankPackedVarMessage msg;
    msg.rank = peer.first;
    msg.nentries = static_cast<int>(entries.size());
    msg.entry_offset = recv_entry_offset;
    msg.hdr_offset = -1;
    msg.offset = recv_total;
    for (const auto &entry : entries) {
      recv_offsets_h(entry.m*nnghbr + entry.n) = recv_total;
      recv_total += entry.data_size;
    }
    msg.data_size = recv_total - msg.offset;
    recv_entry_offset += msg.nentries;
    layout.recv_msgs.push_back(msg);
  }

  // Compared against the live extent, which is the allocated CAPACITY (>= the full
  // off-rank total this LAT layout is a subset of), so the check is unchanged by the
  // grow-only policy above.
  if (send_total > rank_sendbuf_flux_.extent_int(0) ||
      recv_total > rank_recvbuf_flux_.extent_int(0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "LAT rank-packed flux layout exceeds buffer capacity"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(layout.send_offsets, send_offsets_h);
  Kokkos::deep_copy(layout.recv_offsets, recv_offsets_h);
  lat_flux_layouts_.push_back(std::move(layout));
  lat_flux_layout_index_ = static_cast<int>(lat_flux_layouts_.size()) - 1;
  if (cache_key_valid) {
    lat_flux_layout_cache_[cache_key] = lat_flux_layout_index_;
  }
}
#endif

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::PackAndSendFlux()
//! \brief Pack restricted fluxes of cell-centered variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors for flux-correction step.  These fluxes
//! (e.g. for the conserved hydro variables) live at cell faces.
//!
//! This routine packs ALL the buffers on ALL the faces simultaneously for ALL the
//! MeshBlocks. Buffer data are then sent (via MPI) or copied directly for periodic or
//! block boundaries.
//!
//! When `excised` is given, a fine face whose two adjacent cells are both marked in it
//! enters the restricted (fine-to-coarse) average as zero, in every variable.  Same-level
//! payloads and the fluxes in `flx`, which the sender's own update uses, are unchanged.

TaskStatus MeshBoundaryValuesCC::PackAndSendFluxCC(DvceFaceFld5D<Real> &flx,
                                                   DvceFaceFld5D<Real> *extra_flx,
                                                   bool time_integrate, Real weight,
                                                   FaceFldOrigin origin,
                                                   const DvceArray4D<bool> *excised) {
  const int ko = origin.k, jo = origin.j, io = origin.i;  // band origin of flx
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  const int nvar_base = flx.x1f.extent_int(1);
  const bool use_extra = (extra_flx != nullptr);
  const int nvar_extra = use_extra ? extra_flx->x1f.extent_int(1) : 0;
  int nvar = nvar_base + nvar_extra;
  DevExeSpace flux_exec;
  auto extra_x1f = use_extra ? extra_flx->x1f : flx.x1f;
  auto extra_x2f = use_extra ? extra_flx->x2f : flx.x2f;
  auto extra_x3f = use_extra ? extra_flx->x3f : flx.x3f;
  const bool skip_excised = (excised != nullptr);
  DvceArray4D<bool> excised_cell;
  if (skip_excised) excised_cell = *excised;

  auto &cis = pmy_pack->pmesh->mb_indcs.cis;
  auto &cjs = pmy_pack->pmesh->mb_indcs.cjs;
  auto &cks = pmy_pack->pmesh->mb_indcs.cks;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_active_indices = pmy_pack->lat_active_indices.d_view;
  auto lat_flux_send = pmy_pack->lat_flux_send_nghbr.d_view;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto sbuf = sendbuf_device;
  auto rbuf = recvbuf_device;
  auto &one_d = pmy_pack->pmesh->one_d;
  auto &two_d = pmy_pack->pmesh->two_d;
#if MPI_PARALLEL_ENABLED
  const bool rank_packed_flux = rank_packed_lat_flux_active_;
  // stub_flux_same shrank the send-side flux buffers to the coarse footprint on the
  // guarantee that a same-level payload never reaches them.  That holds because LAT
  // always routes same-level fluxes through the rank-packed buffers.  If a future task
  // ordering ever breaks the guarantee, fail loudly here instead of letting the device
  // kernel write past the end of sendbuf.flux.
  if (stub_flux_same && lat_enabled && !rank_packed_flux) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "stub_flux_same is set but the rank-packed flux path is "
              << "inactive while LAT is enabled. Same-level fluxes would be packed into "
              << "send buffers sized for coarse-level data only." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray1D<Real> rank_send_flux;
  DvceArray1D<int> rank_send_flux_offsets;
  if (rank_packed_flux) {
    rank_send_flux = rank_sendbuf_flux_;
    rank_send_flux_offsets = lat_flux_layouts_[lat_flux_layout_index_].send_offsets;
  }
#endif

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  const int nwork_send = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork_send <= 0) return TaskStatus::complete;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(flux_exec, (nwork_send*nflux_nghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = (tmember.league_rank())/(nflux_nghbr*nvar);
    const int m = lat_enabled ? lat_active_indices(a) : a;
    const int slot = (tmember.league_rank() - a*(nflux_nghbr*nvar))/nvar;
    const int n = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = (tmember.league_rank() - a*(nflux_nghbr*nvar) - slot*nvar);
    if (lat_enabled && lat_flux_send(m,n) == 0) return;
    const Real flux_time_weight = time_integrate ? weight*lat_step_dt(m) : 1.0;

    const bool send_to_coarse =
        (nghbr.d_view(m,n).gid >= 0) &&
        (nghbr.d_view(m,n).lev < mblev.d_view(m));
    const bool send_to_same =
        lat_enabled && (nghbr.d_view(m,n).gid >= 0) &&
        (nghbr.d_view(m,n).lev == mblev.d_view(m));
    if (!(send_to_coarse || send_to_same)) return;

    const bool same_level = send_to_same;
    const auto &iflux = same_level ? sbuf[n].iflux_same[0] : sbuf[n].iflux_coar[0];
    int il = iflux.bis;
    int iu = iflux.bie;
    int jl = iflux.bjs;
    int ju = iflux.bje;
    int kl = iflux.bks;
    int ku = iflux.bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji  = nj*ni;
    const int nkj  = nk*nj;
    const int nki  = nk*ni;

    // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
    // in MeshBlockPacks, so array index equals (target_id - first_id)
    int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
    int dn = nghbr.d_view(m,n).dest;

    // Pack restricted fine-to-coarse AMR fluxes or full-resolution same-level LAT fluxes.
    if (send_to_coarse || send_to_same) {
      // x1faces
      if (n<8) {
        // i-index is fixed for flux correction on x1faces
        int fi = same_level ? il : (2*il - cis);
        // fine x1-face flux of variable v entering the restriction
        auto fine = [&](const int k, const int j, const int i) -> Real {
          if (skip_excised && excised_cell(m,k,j,i-1) && excised_cell(m,k,j,i)) {
            return 0.0;
          }
          return (v < nvar_base) ? flx.x1f(m,v,(k)-ko,(j)-jo,(i)-io) :
                                   extra_x1f(m,v - nvar_base,(k)-ko,(j)-jo,(i)-io);
        };
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = same_level ? j : (2*j - cjs);
          int fk = same_level ? k : (2*k - cks);
          Real rflx;
          if (same_level) {
            if (v < nvar_base) {
              rflx = flx.x1f(m,v,(k)-ko,(j)-jo,(fi)-io);
            } else {
              rflx = extra_x1f(m,v - nvar_base,(k)-ko,(j)-jo,(fi)-io);
            }
          } else if (one_d) {
            rflx = fine(0, 0, fi);
          } else if (two_d) {
            rflx = 0.5*(fine(0, fj, fi) + fine(0, fj+1, fi));
          } else {
            rflx = 0.25*(fine(fk, fj, fi) + fine(fk, fj+1, fi) +
                         fine(fk+1, fj, fi) + fine(fk+1, fj+1, fi));
          }
          rflx *= flux_time_weight;
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            rbuf[dn].flux(dm, (j-jl + nj*(k-kl + nk*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
#if MPI_PARALLEL_ENABLED
            if (rank_packed_flux) {
              const int off = rank_send_flux_offsets(m*nnghbr + n);
              rank_send_flux(off + (j-jl + nj*(k-kl + nk*v))) = rflx;
            } else
#endif
            {
              sbuf[n].flux(m, (j-jl + nj*(k-kl + nk*v)) ) = rflx;
            }
          }
        });

      // x2faces
      } else if (n<16) {
        // j-index is fixed for flux correction on x2faces
        int fj = same_level ? jl : (2*jl - cjs);
        // fine x2-face flux of variable v entering the restriction
        auto fine = [&](const int k, const int j, const int i) -> Real {
          if (skip_excised && excised_cell(m,k,j-1,i) && excised_cell(m,k,j,i)) {
            return 0.0;
          }
          return (v < nvar_base) ? flx.x2f(m,v,(k)-ko,(j)-jo,(i)-io) :
                                   extra_x2f(m,v - nvar_base,(k)-ko,(j)-jo,(i)-io);
        };
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          int fi = same_level ? i : (2*i - cis);
          int fk = same_level ? k : (2*k - cks);
          Real rflx;
          if (same_level) {
            if (v < nvar_base) {
              rflx = flx.x2f(m,v,(k)-ko,(fj)-jo,(fi)-io);
            } else {
              rflx = extra_x2f(m,v - nvar_base,(k)-ko,(fj)-jo,(fi)-io);
            }
          } else if (two_d) {
            rflx = 0.5*(fine(0, fj, fi) + fine(0, fj, fi+1));
          } else {
            rflx = 0.25*(fine(fk, fj, fi) + fine(fk, fj, fi+1) +
                         fine(fk+1, fj, fi) + fine(fk+1, fj, fi+1));
          }
          rflx *= flux_time_weight;
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            rbuf[dn].flux(dm, (i-il + ni*(k-kl + nk*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
#if MPI_PARALLEL_ENABLED
            if (rank_packed_flux) {
              const int off = rank_send_flux_offsets(m*nnghbr + n);
              rank_send_flux(off + (i-il + ni*(k-kl + nk*v))) = rflx;
            } else
#endif
            {
              sbuf[n].flux(m, (i-il + ni*(k-kl + nk*v)) ) = rflx;
            }
          }
        });

      // x3faces
      } else if ((n>=24) && (n<32)) {
        // k-index is fixed for flux correction on x3faces
        int fk = same_level ? kl : (2*kl - cks);
        // fine x3-face flux of variable v entering the restriction
        auto fine = [&](const int k, const int j, const int i) -> Real {
          if (skip_excised && excised_cell(m,k-1,j,i) && excised_cell(m,k,j,i)) {
            return 0.0;
          }
          return (v < nvar_base) ? flx.x3f(m,v,(k)-ko,(j)-jo,(i)-io) :
                                   extra_x3f(m,v - nvar_base,(k)-ko,(j)-jo,(i)-io);
        };
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          int fi = same_level ? i : (2*i - cis);
          int fj = same_level ? j : (2*j - cjs);
          Real rflx;
          if (same_level) {
            if (v < nvar_base) {
              rflx = flx.x3f(m,v,(fk)-ko,(fj)-jo,(fi)-io);
            } else {
              rflx = extra_x3f(m,v - nvar_base,(fk)-ko,(fj)-jo,(fi)-io);
            }
          } else {
            rflx = 0.25*(fine(fk, fj, fi) + fine(fk, fj, fi+1) +
                         fine(fk, fj+1, fi) + fine(fk, fj+1, fi+1));
          }
          rflx *= flux_time_weight;
          // copy directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,n).rank == my_rank) {
            rbuf[dn].flux(dm, (i-il + ni*(j-jl + nj*v)) ) = rflx;
          // else copy into send buffer for MPI communication below
          } else {
#if MPI_PARALLEL_ENABLED
            if (rank_packed_flux) {
              const int off = rank_send_flux_offsets(m*nnghbr + n);
              rank_send_flux(off + (i-il + ni*(j-jl + nj*v))) = rflx;
            } else
#endif
            {
              sbuf[n].flux(m, (i-il + ni*(j-jl + nj*v)) ) = rflx;
            }
          }
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI.
  if (global_variable::nranks == 1) return TaskStatus::complete;
  flux_exec.fence();
  bool no_errors=true;
  if (rank_packed_flux) {
    // Persistent sends: bound once to this layout's (peer, offset, count) tuples, then
    // restarted every LAT bin-step.  See EnsurePersistentReqs in bvals.cpp for why this
    // matters with UCX_RCACHE_ENABLE=n.
    const int lay = lat_flux_layout_index_;
    auto &layout = lat_flux_layouts_[lay];
    auto &send_reqs = layout.send_preqs;
    EnsurePersistentFluxReqs(layout.send_msgs, rank_sendbuf_flux_.data(), true,
                             &send_reqs);
    if (!send_reqs.empty()) {
      no_errors = (MPI_Startall(static_cast<int>(send_reqs.size()),
                                send_reqs.data()) == MPI_SUCCESS);
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in starting rank-packed flux sends"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lat_flux_send_reqs_layout_ = lay;
    send_flux_reqs_started_ = true;
    return TaskStatus::complete;
  }
  auto &lat_flux_send_h = pmy_pack->lat_flux_send_nghbr;
  auto &lat_active_indices_h = pmy_pack->lat_active_indices;
  const int nwork_send_h = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  const int nflux_nghbr_h = FluxFaceCount(nnghbr);
  for (int a=0; a<nwork_send_h; ++a) {
    const int m = lat_enabled ? lat_active_indices_h.h_view(a) : a;
    for (int slot=0; slot<nflux_nghbr_h; ++slot) {
      const int n = FluxFaceNeighborIndex(slot, nnghbr);
      if (lat_enabled && lat_flux_send_h.h_view(m,n) == 0) continue;
      const bool send_to_coarse =
          (nghbr.h_view(m,n).gid >= 0) &&
          (nghbr.h_view(m,n).lev < mblev.h_view(m));
      const bool send_to_same =
          lat_enabled && (nghbr.h_view(m,n).gid >= 0) &&
          (nghbr.h_view(m,n).lev == mblev.h_view(m));
      if (send_to_coarse || send_to_same) {
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;

        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer for fluxes
          int data_size = nvar*(send_to_same ? sendbuf[n].iflxs_ndat :
                                                sendbuf[n].iflxc_ndat);
          auto send_ptr = Kokkos::subview(sendbuf[n].flux, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_flux, &(sendbuf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void RecvBuffers()
//! \brief Unpack boundary buffers for flux correction of CC variables.

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx,
                                                     DvceFaceFld5D<Real> *extra_flx,
                                                     FaceFldOrigin origin) {
  const int ko = origin.k, jo = origin.j, io = origin.i;  // band origin of flx
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf_h = recvbuf;
  auto rbuf = recvbuf_device;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_flux_recv = pmy_pack->lat_flux_recv_nghbr.d_view;
  const int my_rank = global_variable::my_rank;
  bool rank_packed_flux = false;
  DvceArray1D<Real> rank_recv_flux;
  DvceArray1D<int> rank_recv_flux_offsets;
#if MPI_PARALLEL_ENABLED
  rank_packed_flux = rank_packed_lat_flux_active_;
  if (rank_packed_flux) {
    rank_recv_flux = rank_recvbuf_flux_;
    rank_recv_flux_offsets = lat_flux_layouts_[lat_flux_layout_index_].recv_offsets;
  }
  bool remote_flux_recv = false;
  //----- STEP 1: check that recv boundary buffer communications have all completed
  // receives only occur for neighbors on faces at a FINER level
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    if (rank_packed_flux) {
      bflag = !TestFluxRecvComplete(&no_errors, &remote_flux_recv);
    } else {
      const int nflux_nghbr_h = FluxFaceCount(nnghbr);
      for (int m=0; m<nmb; ++m) {
        for (int slot=0; slot<nflux_nghbr_h; ++slot) {
          const int n = FluxFaceNeighborIndex(slot, nnghbr);
          if ( (nghbr.h_view(m,n).gid >=0) &&
               (nghbr.h_view(m,n).lev > mblev.h_view(m)) ) {
            if (nghbr.h_view(m,n).rank != my_rank) {
              remote_flux_recv = true;
              int test;
              int ierr = MPI_Test(&(rbuf_h[n].flux_req[m]), &test, MPI_STATUS_IGNORE);
              if (ierr != MPI_SUCCESS) {no_errors=false;}
              if (!(static_cast<bool>(test))) {
                bflag = true;
              }
            }
          }
        }
      }
    }
    // Quit if MPI error detected
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in testing non-blocking receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // exit if recv boundary buffer communications have not completed
    if (bflag) {return TaskStatus::incomplete;}
  }
#endif

  //----- STEP 2: buffers have all completed, so unpack

  const int nvar_base = flx.x1f.extent_int(1);
  const bool use_extra = (extra_flx != nullptr);
  const int nvar_extra = use_extra ? extra_flx->x1f.extent_int(1) : 0;
  int nvar = nvar_base + nvar_extra;
  auto extra_x1f = use_extra ? extra_flx->x1f : flx.x1f;
  auto extra_x2f = use_extra ? extra_flx->x2f : flx.x2f;
  auto extra_x3f = use_extra ? extra_flx->x3f : flx.x3f;

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(# of variables)
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nflux_nghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - m*(nflux_nghbr*nvar))/nvar;
    const int n = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = (tmember.league_rank() - m*(nflux_nghbr*nvar) - slot*nvar);
    if (lat_enabled && lat_flux_recv(m,n) == 0) return;

    // Recv buffer flux indices are for the regular mesh
    int il = rbuf[n].iflux_coar[0].bis;
    int iu = rbuf[n].iflux_coar[0].bie;
    int jl = rbuf[n].iflux_coar[0].bjs;
    int ju = rbuf[n].iflux_coar[0].bje;
    int kl = rbuf[n].iflux_coar[0].bks;
    int ku = rbuf[n].iflux_coar[0].bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji  = nj*ni;
    const int nkj  = nk*nj;
    const int nki  = nk*ni;

    // only unpack buffers for faces when neighbor is at finer level
    if ((nghbr.d_view(m,n).gid >=0) && (nghbr.d_view(m,n).lev > mblev.d_view(m))) {
      //x1 faces
      if (n<8) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          const int idx_buf = j-jl + nj*(k-kl + nk*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          if (v < nvar_base) {
            flx.x1f(m,v,(k)-ko,(j)-jo,(il)-io) = rval;
          } else {
            extra_x1f(m,v - nvar_base,(k)-ko,(j)-jo,(il)-io) = rval;
          }
        });
      // x2faces
      } else if (n<16) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          const int idx_buf = i-il + ni*(k-kl + nk*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          if (v < nvar_base) {
            flx.x2f(m,v,(k)-ko,(jl)-jo,(i)-io) = rval;
          } else {
            extra_x2f(m,v - nvar_base,(k)-ko,(jl)-jo,(i)-io) = rval;
          }
        });
      // x3faces
      } else if ((n>=24) && (n<32)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          const int idx_buf = i-il + ni*(j-jl + nj*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          if (v < nvar_base) {
            flx.x3f(m,v,(kl)-ko,(j)-jo,(i)-io) = rval;
          } else {
            extra_x3f(m,v - nvar_base,(kl)-ko,(j)-jo,(i)-io) = rval;
          }
        });
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

#if MPI_PARALLEL_ENABLED
  if (remote_flux_recv) MarkRecvUnpackPending();
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::ReserveLATPendingMismatch()
//! \brief Give the pending-mismatch registers the accumulator's shape.  Owners call it
//! at setup with the arguments of their first capture, so a mesh that cannot hold the
//! registers fails at startup instead of at the first LAT flux receive.  Only a capture
//! arms them (lat_pending_captured_): until then AddPendingFineFluxMismatchCC adds
//! nothing, exactly as when they did not exist yet.

void MeshBoundaryValuesCC::ReserveLATPendingMismatch(
    const DvceFaceFld5D<Real> &accum, const DvceFaceFld5D<Real> *extra_accum) {
  const auto match = [](const DvceArray5D<Real> &a, const DvceArray5D<Real> &b) {
    for (int r = 0; r < 5; ++r) {
      if (a.extent_int(r) != b.extent_int(r)) return false;
    }
    return true;
  };
  const auto shape = [&match](DvceArray5D<Real> &store, const DvceArray5D<Real> &like) {
    if (!match(store, like)) {
      store = DvceArray5D<Real>("lat_pending_mismatch", like.extent_int(0),
                                like.extent_int(1), like.extent_int(2),
                                like.extent_int(3), like.extent_int(4));
    }
  };
  shape(lat_pending_mismatch1_, accum.x1f);
  shape(lat_pending_mismatch2_, accum.x2f);
  shape(lat_pending_mismatch3_, accum.x3f);
  if (extra_accum != nullptr) {
    shape(lat_pending_extra1_, extra_accum->x1f);
    shape(lat_pending_extra2_, extra_accum->x2f);
    shape(lat_pending_extra3_, extra_accum->x3f);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::RecvAndAccumulateFluxCC()
//! \brief Receive restricted fine fluxes.  Synchronized pairs replace the coarse stage
//! flux directly; unequal-factor union predictors also retain their delayed LAT mismatch.

TaskStatus MeshBoundaryValuesCC::RecvAndAccumulateFluxCC(DvceFaceFld5D<Real> &accum,
                                                         Real scale,
                                                         bool compact_faces,
                                                         DvceFaceFld5D<Real> *extra_accum,
                                                         DvceFaceFld5D<Real> *sync_flx,
                                                         DvceFaceFld5D<Real> *extra_sync_flx,
                                                         Real sync_integrated_weight,
                                                         bool accumulate_sync_mismatch,
                                                         FaceFldOrigin origin,
                                                         bool store_pending_mismatch) {
  const int ko = origin.k, jo = origin.j, io = origin.i;  // band origin of flx
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &rbuf_h = recvbuf;
  auto rbuf = recvbuf_device;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool same_level_lat = lat_enabled && pmy_pack->pmesh->hydro_lat_same_level;
  auto &lat_flux_recv_h = pmy_pack->lat_flux_recv_nghbr;
  auto &lat_flux_recv_indices_h = pmy_pack->lat_flux_recv_indices;
  const int my_rank = global_variable::my_rank;
  bool rank_packed_flux = false;
  DvceArray1D<Real> rank_recv_flux;
  DvceArray1D<int> rank_recv_flux_offsets;
#if MPI_PARALLEL_ENABLED
  rank_packed_flux = rank_packed_lat_flux_active_;
  if (rank_packed_flux) {
    rank_recv_flux = rank_recvbuf_flux_;
    rank_recv_flux_offsets = lat_flux_layouts_[lat_flux_layout_index_].recv_offsets;
  }
  bool remote_flux_recv = false;
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    if (rank_packed_flux) {
      bflag = !TestFluxRecvComplete(&no_errors, &remote_flux_recv);
    } else {
      const int nwork_recv_h = lat_enabled ? pmy_pack->lat_nflux_recv_thispack : nmb;
      const int nflux_nghbr_h = FluxFaceCount(nnghbr);
      for (int a=0; a<nwork_recv_h; ++a) {
        const int m = lat_enabled ? lat_flux_recv_indices_h.h_view(a) : a;
        for (int slot=0; slot<nflux_nghbr_h; ++slot) {
          const int n = FluxFaceNeighborIndex(slot, nnghbr);
          if (lat_enabled && lat_flux_recv_h.h_view(m,n) == 0) continue;
          const bool recv_from_fine =
              (nghbr.h_view(m,n).gid >= 0) &&
              (nghbr.h_view(m,n).lev > mblev.h_view(m));
          const bool recv_from_same =
              same_level_lat && (nghbr.h_view(m,n).gid >= 0) &&
              (nghbr.h_view(m,n).lev == mblev.h_view(m));
          if (recv_from_fine || recv_from_same) {
            if (nghbr.h_view(m,n).rank != my_rank) {
              remote_flux_recv = true;
              int test;
              int ierr = MPI_Test(&(rbuf_h[n].flux_req[m]), &test, MPI_STATUS_IGNORE);
              if (ierr != MPI_SUCCESS) {no_errors=false;}
              if (!(static_cast<bool>(test))) {
                bflag = true;
              }
            }
          }
        }
      }
    }
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in testing non-blocking receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (bflag) {return TaskStatus::incomplete;}
  }
#endif

  const int nvar_base = accum.x1f.extent_int(1);
  const bool use_extra = (extra_accum != nullptr);
  const int nvar_extra = use_extra ? extra_accum->x1f.extent_int(1) : 0;
  const int nvar = nvar_base + nvar_extra;
  // An extra payload without its own sync register is accumulate-only: it takes no part
  // in the direct installation of a due fine flux, so its accumulator keeps the full
  // own-minus-fine difference of every stage.  No caller passes one at present; the
  // dual-energy face velocity always comes with its sync register.
  const bool use_sync = (sync_flx != nullptr);
  const bool use_extra_sync = use_extra && (extra_sync_flx != nullptr);
  const bool sync_integrated = (sync_integrated_weight != 0.0);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto lat_flux_recv = pmy_pack->lat_flux_recv_nghbr.d_view;
  auto lat_flux_recv_indices = pmy_pack->lat_flux_recv_indices.d_view;
  auto lat_active = pmy_pack->lat_active_mb.d_view;
  auto lat_step_factor = pmy_pack->lat_step_factor.d_view;
  auto lat_nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto acc1 = accum.x1f;
  auto acc2 = accum.x2f;
  auto acc3 = accum.x3f;
  auto extra_acc1 = use_extra ? extra_accum->x1f : accum.x1f;
  auto extra_acc2 = use_extra ? extra_accum->x2f : accum.x2f;
  auto extra_acc3 = use_extra ? extra_accum->x3f : accum.x3f;
  auto sync1 = use_sync ? sync_flx->x1f : accum.x1f;
  auto sync2 = use_sync ? sync_flx->x2f : accum.x2f;
  auto sync3 = use_sync ? sync_flx->x3f : accum.x3f;
  auto extra_sync1 = (use_sync && use_extra_sync) ? extra_sync_flx->x1f : accum.x1f;
  auto extra_sync2 = (use_sync && use_extra_sync) ? extra_sync_flx->x2f : accum.x2f;
  auto extra_sync3 = (use_sync && use_extra_sync) ? extra_sync_flx->x3f : accum.x3f;
  // Fine-minus-own mismatch of the union predictor, in the accumulator layout, for the
  // slower receiver's corrector stage (AddPendingFineFluxMismatchCC).
  const bool store_pending = store_pending_mismatch && use_sync && compact_faces &&
                             lat_enabled;
  // The own flux the mismatch is measured against: the snapshot this stage's flux task
  // took before its first-order flux correction (SnapshotPendingOwnFluxCC), else the
  // stage flux as it stands.
  const bool own_snapshot = store_pending && lat_pending_own_snapshot_;
  if (store_pending) {
    ReserveLATPendingMismatch(accum, use_extra_sync ? extra_accum : nullptr);
    lat_pending_captured_ = true;
    lat_pending_own_snapshot_ = false;
  }
  auto pend1 = store_pending ? lat_pending_mismatch1_ : accum.x1f;
  auto pend2 = store_pending ? lat_pending_mismatch2_ : accum.x2f;
  auto pend3 = store_pending ? lat_pending_mismatch3_ : accum.x3f;
  auto extra_pend1 = (store_pending && use_extra_sync) ? lat_pending_extra1_ : accum.x1f;
  auto extra_pend2 = (store_pending && use_extra_sync) ? lat_pending_extra2_ : accum.x2f;
  auto extra_pend3 = (store_pending && use_extra_sync) ? lat_pending_extra3_ : accum.x3f;

  const int nwork_recv = lat_enabled ? pmy_pack->lat_nflux_recv_thispack : nmb;
  if (nwork_recv <= 0) return TaskStatus::complete;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork_recv*nflux_nghbr*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("LATRecvFlux", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = (tmember.league_rank())/(nflux_nghbr*nvar);
    const int m = lat_enabled ? lat_flux_recv_indices(a) : a;
    const int slot = (tmember.league_rank() - a*(nflux_nghbr*nvar))/nvar;
    const int n = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = (tmember.league_rank() - a*(nflux_nghbr*nvar) - slot*nvar);
    if (lat_enabled && lat_flux_recv(m,n) == 0) return;

    const bool recv_from_fine =
        (nghbr.d_view(m,n).gid >= 0) &&
        (nghbr.d_view(m,n).lev > mblev.d_view(m));
    const bool recv_from_same =
        same_level_lat && (nghbr.d_view(m,n).gid >= 0) &&
        (nghbr.d_view(m,n).lev == mblev.d_view(m));
    if (!(recv_from_fine || recv_from_same)) return;
    const int block_factor = lat_step_factor(m);
    const int neighbor_factor = lat_nghbr_factor(m,n);
    const bool equal_factor_fine = recv_from_fine &&
        block_factor == neighbor_factor;
    // A union predictor exchanges time-integrated fluxes for every due factor.  Install
    // the corresponding instantaneous fine flux only on receivers that are themselves
    // due; inactive slower receivers still collect the fine integral for later reflux.
    const bool direct_fine = use_sync && recv_from_fine &&
        (!lat_enabled || lat_active(m) != 0) &&
        (equal_factor_fine || sync_integrated);
    Real sync_scale = 1.0;
    Real receiver_integrated_dt = 0.0;
    if (direct_fine && sync_integrated) {
      const Real sender_dt = (block_factor > 0) ?
          lat_step_dt(m)*static_cast<Real>(neighbor_factor)/
              static_cast<Real>(block_factor) : 0.0;
      const Real integrated_dt = sync_integrated_weight*sender_dt;
      sync_scale = (integrated_dt != 0.0) ? 1.0/integrated_dt : 0.0;
      receiver_integrated_dt = sync_integrated_weight*lat_step_dt(m);
    }

    const bool same_level = recv_from_same;
    const auto &iflux = same_level ? rbuf[n].iflux_same[0] : rbuf[n].iflux_coar[0];
    int il = iflux.bis;
    int iu = iflux.bie;
    int jl = iflux.bjs;
    int ju = iflux.bje;
    int kl = iflux.bks;
    int ku = iflux.bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;
    const int nji  = nj*ni;
    const int nkj  = nk*nj;
    const int nki  = nk*ni;

    if (recv_from_fine || recv_from_same) {
      if (n<8) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          const int idx_buf = j-jl + nj*(k-kl + nk*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          const int ia = compact_faces ? ((il == is) ? 0 : 1) : il;
          if (direct_fine) {
            if (v < nvar_base) {
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                acc1(m,v,k,j,ia) +=
                    receiver_integrated_dt*(fine_flux - sync1(m,v,(k)-ko,(j)-jo,(il)-io));
              }
              if (store_pending && !equal_factor_fine) {
                pend1(m,v,k,j,ia) = fine_flux - (own_snapshot ? pend1(m,v,k,j,ia) :
                    sync1(m,v,(k)-ko,(j)-jo,(il)-io));
              }
              sync1(m,v,(k)-ko,(j)-jo,(il)-io) = fine_flux;
            } else if (use_extra_sync) {
              const int ve = v - nvar_base;
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                extra_acc1(m,ve,k,j,ia) +=
                    receiver_integrated_dt*(fine_flux - extra_sync1(m,ve,(k)-ko,(j)-jo,
                        (il)-io));
              }
              if (store_pending && !equal_factor_fine) {
                extra_pend1(m,ve,k,j,ia) = fine_flux -
                    (own_snapshot ? extra_pend1(m,ve,k,j,ia) :
                                    extra_sync1(m,ve,(k)-ko,(j)-jo,(il)-io));
              }
              extra_sync1(m,ve,(k)-ko,(j)-jo,(il)-io) = fine_flux;
            }
          }
          if (!(direct_fine && equal_factor_fine)) {
            if (v < nvar_base) {
              acc1(m,v,k,j,ia) -= scale*rval;
            } else {
              extra_acc1(m,v - nvar_base,k,j,ia) -= scale*rval;
            }
          }
        });
      } else if (n<16) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          const int idx_buf = i-il + ni*(k-kl + nk*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          const int ja = compact_faces ? ((jl == js) ? 0 : 1) : jl;
          if (direct_fine) {
            if (v < nvar_base) {
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                acc2(m,v,k,ja,i) +=
                    receiver_integrated_dt*(fine_flux - sync2(m,v,(k)-ko,(jl)-jo,(i)-io));
              }
              if (store_pending && !equal_factor_fine) {
                pend2(m,v,k,ja,i) = fine_flux - (own_snapshot ? pend2(m,v,k,ja,i) :
                    sync2(m,v,(k)-ko,(jl)-jo,(i)-io));
              }
              sync2(m,v,(k)-ko,(jl)-jo,(i)-io) = fine_flux;
            } else if (use_extra_sync) {
              const int ve = v - nvar_base;
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                extra_acc2(m,ve,k,ja,i) +=
                    receiver_integrated_dt*(fine_flux - extra_sync2(m,ve,(k)-ko,(jl)-jo,
                        (i)-io));
              }
              if (store_pending && !equal_factor_fine) {
                extra_pend2(m,ve,k,ja,i) = fine_flux -
                    (own_snapshot ? extra_pend2(m,ve,k,ja,i) :
                                    extra_sync2(m,ve,(k)-ko,(jl)-jo,(i)-io));
              }
              extra_sync2(m,ve,(k)-ko,(jl)-jo,(i)-io) = fine_flux;
            }
          }
          if (!(direct_fine && equal_factor_fine)) {
            if (v < nvar_base) {
              acc2(m,v,k,ja,i) -= scale*rval;
            } else {
              extra_acc2(m,v - nvar_base,k,ja,i) -= scale*rval;
            }
          }
        });
      } else if ((n>=24) && (n<32)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          const int idx_buf = i-il + ni*(j-jl + nj*v);
          const bool remote = (nghbr.d_view(m,n).rank != my_rank);
          const Real rval = (rank_packed_flux && remote) ?
              rank_recv_flux(rank_recv_flux_offsets(m*nnghbr + n) + idx_buf) :
              rbuf[n].flux(m,idx_buf);
          const int ka = compact_faces ? ((kl == ks) ? 0 : 1) : kl;
          if (direct_fine) {
            if (v < nvar_base) {
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                acc3(m,v,ka,j,i) +=
                    receiver_integrated_dt*(fine_flux - sync3(m,v,(kl)-ko,(j)-jo,(i)-io));
              }
              if (store_pending && !equal_factor_fine) {
                pend3(m,v,ka,j,i) = fine_flux - (own_snapshot ? pend3(m,v,ka,j,i) :
                    sync3(m,v,(kl)-ko,(j)-jo,(i)-io));
              }
              sync3(m,v,(kl)-ko,(j)-jo,(i)-io) = fine_flux;
            } else if (use_extra_sync) {
              const int ve = v - nvar_base;
              const Real fine_flux = sync_scale*rval;
              if (!equal_factor_fine && accumulate_sync_mismatch) {
                extra_acc3(m,ve,ka,j,i) +=
                    receiver_integrated_dt*(fine_flux - extra_sync3(m,ve,(kl)-ko,(j)-jo,
                        (i)-io));
              }
              if (store_pending && !equal_factor_fine) {
                extra_pend3(m,ve,ka,j,i) = fine_flux -
                    (own_snapshot ? extra_pend3(m,ve,ka,j,i) :
                                    extra_sync3(m,ve,(kl)-ko,(j)-jo,(i)-io));
              }
              extra_sync3(m,ve,(kl)-ko,(j)-jo,(i)-io) = fine_flux;
            }
          }
          if (!(direct_fine && equal_factor_fine)) {
            if (v < nvar_base) {
              acc3(m,v,ka,j,i) -= scale*rval;
            } else {
              extra_acc3(m,v - nvar_base,ka,j,i) -= scale*rval;
            }
          }
        });
      }
    }
    tmember.team_barrier();
  });

#if MPI_PARALLEL_ENABLED
  if (remote_flux_recv) MarkRecvUnpackPending();
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesCC::InitRecvFlux
//! \brief Posts non-blocking receives (with MPI) for boundary communication of fluxes of
//! cell-centered variables, which are communicated at FACES of MeshBlocks at the SAME
//! levels.  This is different than for fluxes of face-centered vars.

TaskStatus MeshBoundaryValuesCC::InitFluxRecv(const int nvars) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  // A flux-only LAT stage can bypass InitRecv(), so protect CUDA-aware MPI receive
  // storage from reuse while the previous asynchronous unpack still reads it.
  WaitRecvUnpackComplete();
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool same_level_lat = lat_enabled && pmy_pack->pmesh->hydro_lat_same_level;
  auto &lat_flux_recv = pmy_pack->lat_flux_recv_nghbr;
  auto &lat_flux_recv_indices = pmy_pack->lat_flux_recv_indices;

  rank_packed_lat_flux_active_ = lat_enabled;
  if (rank_packed_lat_flux_active_) {
    PrepareLATRankPackedFluxLayout(nvars);
    // Persistent receives, restarted rather than re-posted; see EnsurePersistentReqs.
    const int lay = lat_flux_layout_index_;
    auto &layout = lat_flux_layouts_[lay];
    auto &recv_reqs = layout.recv_preqs;
    EnsurePersistentFluxReqs(layout.recv_msgs, rank_recvbuf_flux_.data(), false,
                             &recv_reqs);
    bool no_errors = true;
    if (!recv_reqs.empty()) {
      no_errors = (MPI_Startall(static_cast<int>(recv_reqs.size()),
                                recv_reqs.data()) == MPI_SUCCESS);
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in starting rank-packed flux receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lat_flux_recv_reqs_layout_ = lay;
    recv_flux_reqs_started_ = true;
    recv_flux_counts_verified_ = false;
    // Matches the old lat_send_flux_reqs_.clear(): nothing is started on the send side
    // until PackAndSendFlux runs, so ClearFluxSend must not drain a previous phase.
    lat_flux_send_reqs_layout_ = -1;
    send_flux_reqs_started_ = false;
    return TaskStatus::complete;
  }

  // Initialize communications of fluxes
  bool no_errors=true;
  const int nwork_recv = lat_enabled ? pmy_pack->lat_nflux_recv_thispack : nmb;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  for (int a=0; a<nwork_recv; ++a) {
    const int m = lat_enabled ? lat_flux_recv_indices.h_view(a) : a;
    for (int slot=0; slot<nflux_nghbr; ++slot) {
      const int n = FluxFaceNeighborIndex(slot, nnghbr);
      if (lat_enabled && lat_flux_recv.h_view(m,n) == 0) continue;
      const int lev = pmy_pack->pmb->mb_lev.h_view(m);
      const bool recv_from_fine =
          (nghbr.h_view(m,n).gid >= 0) &&
          (nghbr.h_view(m,n).lev > lev);
      const bool recv_from_same =
          same_level_lat && (nghbr.h_view(m,n).gid >= 0) &&
          (nghbr.h_view(m,n).lev == lev);
      if (recv_from_fine || recv_from_same) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars*(recv_from_same ? recvbuf[n].iflxs_ndat :
                                                   recvbuf[n].iflxc_ndat);
          auto recv_ptr = Kokkos::subview(recvbuf[n].flux, m, Kokkos::ALL);

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_flux, &(recvbuf[n].flux_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::SnapshotPendingOwnFluxCC()
//! \brief Union stage-1 predictor: keep this block's own flux, as its flux kernel left
//! it, on every face whose fine-minus-own mismatch RecvAndAccumulateFluxCC is about to
//! capture (store_pending_mismatch).
//!
//! The corrector adds that mismatch to its own flux BEFORE its first-order flux
//! correction (AddPendingFineFluxMismatchCC), so the estimate F_own(u1) + [F_fine(u^n) -
//! F_own(u^n)] is off by O(dt) only if both F_own are the same operator.  The capture
//! runs after the flux task, by which time FOFC may have replaced the own flux of a
//! flagged boundary cell with the first-order LLF flux (or an excision rule its flux);
//! measured against that, the corrector would carry F_HO(u1) - F_LLF(u^n), the whole
//! high-order minus first-order difference, into exactly the cells FOFC had found
//! inadmissible.  Owners call this in the flux task ahead of FOFC and of the excision
//! flux boundaries, with the arguments of their capture; the capture then measures the
//! mismatch against this snapshot.  Wherever nothing replaced a captured face the
//! snapshot equals the stage flux, so the capture's arithmetic is unchanged there.

void MeshBoundaryValuesCC::SnapshotPendingOwnFluxCC(
    const DvceFaceFld5D<Real> &accum, const DvceFaceFld5D<Real> *extra_accum,
    const DvceFaceFld5D<Real> &flx, const DvceFaceFld5D<Real> *extra_flx,
    FaceFldOrigin origin) {
  lat_pending_own_snapshot_ = false;
  // The capture stores a mismatch only on a union predictor, the one cadence whose
  // fluxes are time-integrated per block (RecvAndAccumulateFluxCC, sync_integrated).
  if (!(pmy_pack->lat_active_mask_enabled) || !(pmy_pack->lat_per_block_timestep)) {
    return;
  }
  const int nactive = pmy_pack->lat_nactive_thispack;
  if (nactive <= 0) return;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return;
  const bool use_extra = (extra_accum != nullptr) && (extra_flx != nullptr);
  ReserveLATPendingMismatch(accum, use_extra ? extra_accum : nullptr);
  const int nvar_base = lat_pending_mismatch1_.extent_int(1);
  const int nvar = nvar_base + (use_extra ? lat_pending_extra1_.extent_int(1) : 0);
  const int ko = origin.k, jo = origin.j, io = origin.i;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto lat_flux_recv = pmy_pack->lat_flux_recv_nghbr.d_view;
  auto step_factor = pmy_pack->lat_step_factor.d_view;
  auto nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  auto rbuf = recvbuf_device;
  auto f1 = flx.x1f, f2 = flx.x2f, f3 = flx.x3f;
  auto ef1 = use_extra ? extra_flx->x1f : flx.x1f;
  auto ef2 = use_extra ? extra_flx->x2f : flx.x2f;
  auto ef3 = use_extra ? extra_flx->x3f : flx.x3f;
  auto d1 = lat_pending_mismatch1_, d2 = lat_pending_mismatch2_;
  auto d3 = lat_pending_mismatch3_;
  auto ed1 = use_extra ? lat_pending_extra1_ : d1;
  auto ed2 = use_extra ? lat_pending_extra2_ : d2;
  auto ed3 = use_extra ? lat_pending_extra3_ : d3;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), nactive*nflux_nghbr*nvar, Kokkos::AUTO);
  Kokkos::parallel_for("LATPendingOwnFlux", athenak_lw(policy),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - a*nflux_nghbr*nvar)/nvar;
    const int n = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = tmember.league_rank() - a*nflux_nghbr*nvar - slot*nvar;
    const int m = active_indices(a);
    // Exactly the faces the capture writes: a receiving face of an active block whose
    // finer neighbour runs at another factor.
    if (lat_flux_recv(m,n) == 0 ||
        !(nghbr.d_view(m,n).gid >= 0 && nghbr.d_view(m,n).lev > mblev.d_view(m) &&
          nghbr_factor(m,n) != step_factor(m))) {
      return;
    }
    const bool extra = (v >= nvar_base);
    const int ve = extra ? v - nvar_base : v;
    const auto &iflux = rbuf(n).iflux_coar[0];
    const int il = iflux.bis, iu = iflux.bie;
    const int jl = iflux.bjs, ju = iflux.bje;
    const int kl = iflux.bks, ku = iflux.bke;
    const int ni = iu - il + 1, nj = ju - jl + 1, nk = ku - kl + 1;
    if (n < 8) {
      const int ia = (il == is) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*nj),
      [&](const int idx) {
        const int k = idx/nj + kl;
        const int j = idx - (k - kl)*nj + jl;
        if (extra) {
          ed1(m,ve,k,j,ia) = ef1(m,ve,k-ko,j-jo,il-io);
        } else {
          d1(m,v,k,j,ia) = f1(m,v,k-ko,j-jo,il-io);
        }
      });
    } else if (n < 16) {
      const int ja = (jl == js) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*ni),
      [&](const int idx) {
        const int k = idx/ni + kl;
        const int i = idx - (k - kl)*ni + il;
        if (extra) {
          ed2(m,ve,k,ja,i) = ef2(m,ve,k-ko,jl-jo,i-io);
        } else {
          d2(m,v,k,ja,i) = f2(m,v,k-ko,jl-jo,i-io);
        }
      });
    } else if (n >= 24 && n < 32) {
      const int ka = (kl == ks) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj*ni),
      [&](const int idx) {
        const int j = idx/ni + jl;
        const int i = idx - (j - jl)*ni + il;
        if (extra) {
          ed3(m,ve,ka,j,i) = ef3(m,ve,kl-ko,j-jo,i-io);
        } else {
          d3(m,v,ka,j,i) = f3(m,v,kl-ko,j-jo,i-io);
        }
      });
    }
    tmember.team_barrier();
  });
  lat_pending_own_snapshot_ = true;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC()
//! \brief Corrector stage of a slower bin under the union stage-1 predictor: on every
//! face of a finer neighbour that is still pending, add the fine-minus-own flux mismatch
//! captured during the fused predictor to this block's own stage flux.
//!
//! Without LAT a coarse block takes the restricted fine flux at every stage.  The union
//! predictor installs it at stage 1, but a faster neighbour sends nothing at the slower
//! bin's corrector, so that stage used to run on the coarse block's own reconstruction.
//! The difference F_fine - F_own is a spatial mismatch, O(1) in dt, so until the
//! window-end reflux the coarse boundary cells -- the states the pending neighbour's
//! substeps read back as ghosts -- were off by O(dt) per step.  The stage-2 fine flux is
//! estimated as F_own(u1) + [F_fine(u^n) - F_own(u^n)]: the mismatch changes by O(dt)
//! over the step, so the estimate is off by O(dt) times the mismatch and the boundary
//! state by O(dt^2) of it.  The mismatch is a fixed forcing, so the linear stability of
//! the stage is that of the own flux.  Callers install it before
//! AccumulateLATCoarseFluxes books the stage flux, so the window-end reflux replaces
//! exactly what was used and conservation is unchanged.  A stored stage-1 fine flux
//! alone would be off by the whole O(dt) change of the flux over the step.

bool MeshBoundaryValuesCC::PendingFineFluxMismatchDue() const {
  return pmy_pack->lat_active_mask_enabled &&
         pmy_pack->lat_union_pending_below_factor > 1 &&
         pmy_pack->lat_nactive_thispack > 0 && lat_pending_captured_ &&
         lat_pending_mismatch1_.extent_int(0) >= pmy_pack->nmb_thispack;
}

void MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC(DvceFaceFld5D<Real> &flx,
                                                        DvceFaceFld5D<Real> *extra_flx,
                                                        FaceFldOrigin origin) {
  ApplyPendingFineFluxMismatchCC(flx, extra_flx, origin, false);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::StashPendingEdgeFOFCFlags()
//! \brief First-order flux correction next to the estimate above, for a fluid whose
//! FOFC flags are not exchanged between blocks (Hydro::FOFC, DynGRMHDPS::FOFC).
//!
//! The first FOFC pass tests the ghost ring as well as the owned cells and replaces the
//! faces of every cell it flags, on the premise that the owner tests the same cell on
//! the same fluxes: a face two blocks share is then first order on both sides or on
//! neither.  The estimate breaks that premise along the edges where a face with it meets
//! a face shared with an active block of the same level and cadence.  The owner tests
//! those edge cells with the estimate, the neighbour, which cannot add it, without, and a
//! cell only one of them flags leaves the shared face first order on that side alone.
//! Both blocks update in the same stage and no reflux books that face, so the fluid lost
//! conservation: a cold shear wave on three SMR levels drifted in mass by -5.6e-7 in the
//! first window of its slowest bin (tst/inputs/hydro_lat_fofc_edges.athinput), and the
//! BBH SMR deck with M1 (bbh_smr_m1_direct_lat) left 16 same-level faces one-sided in its
//! first window.
//!
//! A flag both sides can form is the one on the fluxes without the estimate.  The fluid
//! tests its edge cells on those fluxes before the estimate is added and hands the flags
//! to this function, which keeps them for the cells both kinds of face meet and clears
//! the others' (the first pass decides those on fluxes the neighbour shares).  After the
//! first pass, ReconcilePendingEdgeFOFCFlags gives those cells their stashed flags back
//! and, where the test with the estimate flagged something the test without it did not,
//! removes the estimate from the cell's faces: its update is then the one it was tested
//! on, and the window-end reflux books the flux it used.  Where the two tests agree
//! nothing changes, so a run in which they agree on every such cell is unchanged bit for
//! bit.  Cost: one trial and one floor test per edge (12 small launches each in 3D) and
//! three small kernels in each corrector that carries the estimate, and one int per edge
//! cell.

void MeshBoundaryValuesCC::StashPendingEdgeFOFCFlags(const DvceArray4D<bool> &flag,
                                                     const DvceArray5D<bool> *scalar_flag,
                                                     const int nscalar) {
  lat_pending_edge_stashed_ = false;
  const int nwork = pmy_pack->lat_nactive_thispack;
  if (nwork <= 0 || !(pmy_pack->pmesh->multi_d)) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int nmb = pmy_pack->nmb_thispack;
  const int nmax = std::max(indcs.nx1, std::max(indcs.nx2, indcs.nx3));
  if (lat_pending_edge_.extent_int(0) < nmb) {
    Kokkos::realloc(lat_pending_edge_, nmb, kPendingEdgeCount, nmax);
  }
  auto edge = lat_pending_edge_;
  auto nghbr = pmy_pack->pmb->nghbr.d_view;
  auto mblev = pmy_pack->pmb->mb_lev.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto step_factor = pmy_pack->lat_step_factor.d_view;
  auto nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  auto scalar = (scalar_flag != nullptr) ? *scalar_flag : DvceArray5D<bool>();
  const int nscal = (scalar_flag != nullptr) ? nscalar : 0;
  par_for("lat_pending_edge_stash", DevExeSpace(), 0, nwork-1, 0, kPendingEdgeCount-1,
          0, nmax-1,
  KOKKOS_LAMBDA(int a, int e, int p) {
    int i, j, k;
    if (!PendingEdgeCell(e, p, is, ie, js, je, ks, ke, three_d, i, j, k)) return;
    const int m = active_indices(a);
    const bool on_face[6] = {i == is, i == ie, j == js, j == je,
                             three_d && k == ks, three_d && k == ke};
    const int first_slot[6] = {0, 4, 8, 12, 24, 28};
    bool shared = false, estimate = false;
    for (int f = 0; f < 6; ++f) {
      if (!on_face[f]) continue;
      const int n = first_slot[f];
      shared = shared || (nghbr(m,n).gid >= 0 && nghbr(m,n).lev == mblev(m) &&
                          nghbr_factor(m,n) == step_factor(m));
      // The predicate of ApplyPendingFineFluxMismatchCC, on any quarter of the face.
      for (int s = 0; s < 4; ++s) {
        estimate = estimate || (nghbr(m,n+s).gid >= 0 && nghbr(m,n+s).lev > mblev(m) &&
                                nghbr_factor(m,n+s) < step_factor(m));
      }
    }
    int entry = 0;
    if (shared && estimate) {
      entry = kPendingEdgeTested | (flag(m,k,j,i) ? kPendingEdgeFlag : 0);
      for (int n = 0; n < nscal; ++n) {
        if (scalar(m,n,k,j,i)) entry |= (kPendingEdgeScalar << n);
      }
    }
    edge(m,e,p) = entry;
    flag(m,k,j,i) = false;
    for (int n = 0; n < nscal; ++n) scalar(m,n,k,j,i) = false;
  });
  lat_pending_edge_stashed_ = true;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::ReconcilePendingEdgeFOFCFlags()
//! \brief After the first FOFC test of the stage whose flags StashPendingEdgeFOFCFlags
//! kept: the stashed flags replace the new ones on the cells both kinds of face meet, and
//! where the new test flagged the cell (or one of its scalars) beyond the stashed flags,
//! the estimate is taken off the cell's faces again.  A cell flagged without the estimate
//! has every face replaced, so it keeps the estimate until the replacement.  Call before
//! the faces are replaced.

void MeshBoundaryValuesCC::ReconcilePendingEdgeFOFCFlags(
    const DvceArray4D<bool> &flag, const DvceArray5D<bool> *scalar_flag,
    const int nscalar, DvceFaceFld5D<Real> &flx, DvceFaceFld5D<Real> *extra_flx,
    FaceFldOrigin origin) {
  if (!lat_pending_edge_stashed_) return;
  lat_pending_edge_stashed_ = false;
  const int nwork = pmy_pack->lat_nactive_thispack;
  if (nwork <= 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto edge = lat_pending_edge_;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto scalar = (scalar_flag != nullptr) ? *scalar_flag : DvceArray5D<bool>();
  const int nscal = (scalar_flag != nullptr) ? nscalar : 0;
  const int nmax = edge.extent_int(2);
  par_for("lat_pending_edge_reconcile", DevExeSpace(), 0, nwork-1, 0, kPendingEdgeCount-1,
          0, nmax-1,
  KOKKOS_LAMBDA(int a, int e, int p) {
    int i, j, k;
    if (!PendingEdgeCell(e, p, is, ie, js, je, ks, ke, three_d, i, j, k)) return;
    const int m = active_indices(a);
    const int entry = edge(m,e,p);
    if (!(entry & kPendingEdgeTested)) return;
    const bool stashed = (entry & kPendingEdgeFlag);
    bool beyond = !stashed && flag(m,k,j,i);
    flag(m,k,j,i) = stashed;
    for (int n = 0; n < nscal; ++n) {
      const bool stashed_n = (entry & (kPendingEdgeScalar << n));
      beyond = beyond || (!stashed && !stashed_n && scalar(m,n,k,j,i));
      scalar(m,n,k,j,i) = stashed_n;
    }
    edge(m,e,p) = beyond ? kPendingEdgeWithhold : 0;
  });
  ApplyPendingFineFluxMismatchCC(flx, extra_flx, origin, true);
}

//! The addition above (withhold = false), or its removal from the faces of the edge
//! cells ReconcilePendingEdgeFOFCFlags marked (withhold = true).
void MeshBoundaryValuesCC::ApplyPendingFineFluxMismatchCC(DvceFaceFld5D<Real> &flx,
                                                          DvceFaceFld5D<Real> *extra_flx,
                                                          FaceFldOrigin origin,
                                                          const bool withhold) {
  if (!PendingFineFluxMismatchDue()) return;
  const int nactive = pmy_pack->lat_nactive_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return;
  const int nvar_base = lat_pending_mismatch1_.extent_int(1);
  const bool use_extra = (extra_flx != nullptr) &&
                         (lat_pending_extra1_.extent_int(0) >= pmy_pack->nmb_thispack);
  const int nvar = nvar_base + (use_extra ? lat_pending_extra1_.extent_int(1) : 0);
  const int ko = origin.k, jo = origin.j, io = origin.i;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto step_factor = pmy_pack->lat_step_factor.d_view;
  auto nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  auto rbuf = recvbuf_device;
  auto f1 = flx.x1f, f2 = flx.x2f, f3 = flx.x3f;
  auto ef1 = use_extra ? extra_flx->x1f : flx.x1f;
  auto ef2 = use_extra ? extra_flx->x2f : flx.x2f;
  auto ef3 = use_extra ? extra_flx->x3f : flx.x3f;
  auto d1 = lat_pending_mismatch1_, d2 = lat_pending_mismatch2_;
  auto d3 = lat_pending_mismatch3_;
  auto ed1 = use_extra ? lat_pending_extra1_ : d1;
  auto ed2 = use_extra ? lat_pending_extra2_ : d2;
  auto ed3 = use_extra ? lat_pending_extra3_ : d3;
  auto edge = lat_pending_edge_;
  const Real sign = withhold ? -1.0 : 1.0;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), nactive*nflux_nghbr*nvar, Kokkos::AUTO);
  Kokkos::parallel_for("LATPendingFineFlux", athenak_lw(policy),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - a*nflux_nghbr*nvar)/nvar;
    const int n = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = tmember.league_rank() - a*nflux_nghbr*nvar - slot*nvar;
    const int m = active_indices(a);
    // Every faster neighbour was due at this tick's union predictor, and every due
    // coarse/fine pair captured its mismatch there.
    if (!(nghbr.d_view(m,n).gid >= 0 && nghbr.d_view(m,n).lev > mblev.d_view(m) &&
          nghbr_factor(m,n) < step_factor(m))) {
      return;
    }
    const bool extra = (v >= nvar_base);
    const int ve = extra ? v - nvar_base : v;
    const auto &iflux = rbuf(n).iflux_coar[0];
    const int il = iflux.bis, iu = iflux.bie;
    const int jl = iflux.bjs, ju = iflux.bje;
    const int kl = iflux.bks, ku = iflux.bke;
    const int ni = iu - il + 1, nj = ju - jl + 1, nk = ku - kl + 1;
    // Only the faces of the cells the reconcile marked, when withholding.
    auto skip = [&](const int i, const int j, const int k) {
      if (!withhold) return false;
      int e, p;
      return !(PendingEdgeEntry(i, j, k, is, ie, js, je, ks, ke, three_d, e, p) &&
               (edge(m,e,p) & kPendingEdgeWithhold));
    };
    if (n < 8) {
      const int ia = (il == is) ? 0 : 1;
      const int ic = (ia == 0) ? is : ie;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*nj),
      [&](const int idx) {
        const int k = idx/nj + kl;
        const int j = idx - (k - kl)*nj + jl;
        if (skip(ic, j, k)) return;
        if (extra) {
          ef1(m,ve,k-ko,j-jo,il-io) += sign*ed1(m,ve,k,j,ia);
        } else {
          f1(m,v,k-ko,j-jo,il-io) += sign*d1(m,v,k,j,ia);
        }
      });
    } else if (n < 16) {
      const int ja = (jl == js) ? 0 : 1;
      const int jc = (ja == 0) ? js : je;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*ni),
      [&](const int idx) {
        const int k = idx/ni + kl;
        const int i = idx - (k - kl)*ni + il;
        if (skip(i, jc, k)) return;
        if (extra) {
          ef2(m,ve,k-ko,jl-jo,i-io) += sign*ed2(m,ve,k,ja,i);
        } else {
          f2(m,v,k-ko,jl-jo,i-io) += sign*d2(m,v,k,ja,i);
        }
      });
    } else if (n >= 24 && n < 32) {
      const int ka = (kl == ks) ? 0 : 1;
      const int kc = (ka == 0) ? ks : ke;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj*ni),
      [&](const int idx) {
        const int j = idx/ni + jl;
        const int i = idx - (j - jl)*ni + il;
        if (skip(i, j, kc)) return;
        if (extra) {
          ef3(m,ve,kl-ko,j-jo,i-io) += sign*ed3(m,ve,ka,j,i);
        } else {
          f3(m,v,kl-ko,j-jo,i-io) += sign*d3(m,v,ka,j,i);
        }
      });
    }
    tmember.team_barrier();
  });
}
