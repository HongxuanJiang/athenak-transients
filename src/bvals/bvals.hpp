#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals.hpp
//! \brief defines classes for handling boundary values for both particles as well as all
//! types of Mesh variables. For Mesh variables, methods for cell-centered and
//! face-centered fields are currently implemented, based on derived classes from the
//! generic MeshBoundaryValue class.  A separate ParticlesBoundaryValues class is
//! implemented for particles.

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1,block, reflect, inflow, outflow, diode, user, periodic,
                         shear_periodic, vacuum, mg_zerograd, mg_zerofixed, mg_multipole};

//! identifiers for status of MPI boundary communications
enum class BoundaryStatus {waiting, arrived, completed};

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "tasklist/task_list.hpp"
//#include "particles/particles.hpp"

// Forward declarations
class MeshBlockPack;
namespace particles {
class Particles;
}

//----------------------------------------------------------------------------------------
//! \fn int CreateBvals_MPI_Tag(int lid, int bufid)
//! \brief calculate an MPI tag for boundary buffer communications.  Note maximum size of
//! lid that can be encoded is set by (NUM_BITS_LID) macro defined in athena.hpp.
//! The convention in AthenaK is lid and bufid are both for the *receiving* process.
inline int CreateBvals_MPI_Tag(int lid, int bufid) {
  return (bufid << (NUM_BITS_LID)) | lid;
}

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct MeshBufferIndcs {
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  MeshBufferIndcs() :
    bis(0), bie(0), bjs(0), bje(0), bks(0), bke(0) {}
};

//----------------------------------------------------------------------------------------
//! \struct MeshBoundaryBuffer
//! \brief container for index ranges, storage, and flags for boundary buffers

struct MeshBoundaryBuffer {
  // fixed-length-3 arrays used to store indices of each buffer for cell-centered vars, or
  // each component of a face-centered vector field ([0,1,2] --> [x1f, x2f, x3f]). For
  // cell-centered variables only first [0] component of index arrays are needed.
  MeshBufferIndcs isame[3];  // indices for pack/unpack when dest/src at same level
  MeshBufferIndcs icoar[3];  // indices for pack/unpack when dest/src at coarser level
  MeshBufferIndcs ifine[3];  // indices for pack/unpack when dest/src at finer level
  MeshBufferIndcs iprol[3];  // indices for prolongation (only used for receives)
  MeshBufferIndcs iflux_same[3];  // indices for pack/unpack for flux correction
  MeshBufferIndcs iflux_coar[3];  // indices for pack/unpack for flux correction
  // With Z4c higher-order prolongation/rstriction, must also send coarse data between
  // MeshBlocks at the same level, which requires an additional indices array
  MeshBufferIndcs isame_z4c;  // indices for pack/unpack with z4c when dst/src at same lvl

  // Maximum number of data elements (bie-bis+1) across 3 components of above
  // Zero until the owner's Init*Indices writes them, so a slot no owner ever fills
  // sizes nothing rather than sizing from whatever the allocation happened to hold.
  int isame_ndat = 0, isame_z4c_ndat = 0, icoar_ndat = 0, ifine_ndat = 0;
  int iflxs_ndat = 0, iflxc_ndat = 0;
  // 2D Views that store buffer data on device, dimensioned (nmb, ndata)
  DvceArray2D<Real> vars, flux;

#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif

  // function to allocate memory for buffers for variables and their fluxes
  // Must only be called after BufferIndcs above are initialized
  void AllocateBuffers(int nmb, int nvars, int nfluxvars, bool is_z4c,
                       bool stub_vars = false, bool coarse_only_flux = false,
                       bool no_same_vars = false, bool same_only_vars = false) {
    if (stub_vars) {
      // The owner guarantees this buffer's vars payload is never read or written
      // (see MeshBoundaryValues::stub_send_vars).  Flux storage stays full: the
      // per-buffer flux path remains live under MPI.
      Kokkos::realloc(vars, 1, 1);
    } else if (same_only_vars) {
      // The owner guarantees this buffer only ever carries a SAME-level payload, so
      // the coarse and fine footprints are dead storage.  See
      // MeshBoundaryValues::same_level_vars_only.
      Kokkos::realloc(vars, nmb, (nvars*isame_ndat));
    } else if (is_z4c) {
      // With Z4c, buffers may contain BOTH same and coarse data
      int nmax = std::max(isame_z4c_ndat, std::max(icoar_ndat, ifine_ndat) );
      Kokkos::realloc(vars, nmb, (nvars*nmax));
    } else {
      // no_same_vars: the owner guarantees this buffer never receives a SAME-level
      // payload (see MeshBoundaryValues::stub_recv_same_vars), so only the coarse/fine
      // footprint is needed.
      int nmax = no_same_vars ? std::max(icoar_ndat, ifine_ndat) :
                 std::max(isame_ndat, std::max(icoar_ndat, ifine_ndat) );
      Kokkos::realloc(vars, nmb, (nvars*nmax));
    }
    // coarse_only_flux is an owner-level guarantee that this buffer never carries a
    // SAME-level flux payload, so only the (smaller) coarse footprint is needed.  See
    // MeshBoundaryValues::stub_flux_same for the invariant and where it is checked.
    int nmax = coarse_only_flux ? iflxc_ndat : std::max(iflxs_ndat, iflxc_ndat);
    Kokkos::realloc(flux, nmb, (nfluxvars*nmax));
  }
};

//----------------------------------------------------------------------------------------
//! \struct MeshBoundaryBufferDevice
//! \brief lightweight device metadata and non-owning access to one boundary buffer

struct MeshBufferIndcsDevice {
  int bis,bie,bjs,bje,bks,bke;
};

struct MeshBufferDataDevice {
  Real *data;
  std::size_t stride0;

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int m, const int n) const {
    return data[static_cast<std::size_t>(m)*stride0 + static_cast<std::size_t>(n)];
  }
};

struct MeshBoundaryBufferDevice {
  MeshBufferIndcsDevice isame[3];
  MeshBufferIndcsDevice icoar[3];
  MeshBufferIndcsDevice ifine[3];
  MeshBufferIndcsDevice iprol[3];
  MeshBufferIndcsDevice iflux_same[3];
  MeshBufferIndcsDevice iflux_coar[3];
  MeshBufferIndcsDevice isame_z4c;
  int isame_ndat, isame_z4c_ndat, icoar_ndat, ifine_ndat, iflxs_ndat, iflxc_ndat;
  MeshBufferDataDevice vars, flux;
};

static_assert(std::is_trivially_default_constructible_v<MeshBoundaryBufferDevice>);
static_assert(std::is_trivially_copyable_v<MeshBoundaryBufferDevice>);

//----------------------------------------------------------------------------------------
//! \struct RankPackedVarEntry
//! \brief metadata for one off-rank (MeshBlock, neighbor) boundary payload

struct RankPackedVarEntry {
  int m;
  int n;
  int lid;
  int dn;
  int data_size;
  int offset;
};

//----------------------------------------------------------------------------------------
//! \struct RankPackedVarMessage
//! \brief metadata for one aggregate boundary message to or from a rank

struct RankPackedVarMessage {
  int rank;
  int nentries;
  int entry_offset;
  int hdr_offset;
  int offset;
  int data_size;
};

#if MPI_PARALLEL_ENABLED
//----------------------------------------------------------------------------------------
//! Granularity of the rank-packed aggregate payload buffers: one MiB of Reals.
//!
//! These buffers are sized from the off-rank payload total, which AMR moves by a handful
//! of entries on essentially every cycle.  Reallocating them to the exact total churned
//! tens of MiB of device memory per cycle for a change of a few KiB.  Rounding the
//! capacity up to a whole MiB and growing only turns that into a handful of allocations
//! for a whole run.  See mesh/mb_storage.hpp for the block-sized version of this policy.
constexpr std::size_t kRankPackedBufGrainReals =
    static_cast<std::size_t>(1024*1024)/sizeof(Real);

//----------------------------------------------------------------------------------------
//! \fn int RankPackedBufCapacity(int nreal_needed)
//! \brief number of Reals to allocate in order to hold nreal_needed of them
//!
//! Never returns zero: the aggregate buffers are constructed with extent 1 and their
//! .data() is taken unconditionally, so an empty allocation would be a change in behavior
//! for a rank with no off-rank partners.

inline int RankPackedBufCapacity(const int nreal_needed) {
  const int n = std::max(nreal_needed, 1);
  const int grain = static_cast<int>(kRankPackedBufGrainReals);
  // Cannot round up without overflowing the extent (a >16 GiB payload on one rank); ask
  // for exactly what is needed and let the allocation itself report the failure.
  if (n > (std::numeric_limits<int>::max() - (grain - 1))) return n;
  return ((n + grain - 1)/grain)*grain;
}

//----------------------------------------------------------------------------------------
//! \struct LATOffsetPool
//! \brief pooled row storage for the two nmb*nnghbr offset maps every LAT layout needs
//!
//! Each layout used to allocate its own pair of device int views, and a few layouts are
//! built per cycle because the topology version changes about that often.  Rows of one
//! persistent allocation are handed out instead, so a layout costs no allocation at all.
//!
//! The pool is reshaped ONLY when row 0 is requested, i.e. when the layout vector has
//! just been emptied and no layout holds a row, so a row that a layout -- or a kernel
//! that captured it -- is still using can never move.  Rows past the end of the pool fall
//! back to their own allocation exactly as before and the pool catches up at the next
//! reset, which keeps the policy correct for any number of layouts.
//!
//! Rows are handed out uninitialized: every caller fills a full-length host mirror and
//! deep_copies the whole row before anything reads it.

//! Rows reserved on the first reshape, so the usual handful of layouts is covered without
//! waiting for the high-water count to catch up.  One row is nmb*nnghbr ints, ~52 KiB at
//! 231 blocks and 56 neighbors.
constexpr int kLATOffsetPoolMinRows = 8;

struct LATOffsetPool {
  DvceArray2D<int> rows;
  int want_rows = 0;   // high-water row count; applied the next time row 0 is requested

  //! \brief row `row` of the pool as a contiguous map of `len` ints
  DvceArray1D<int> Acquire(const char *label, const int row, const int len) {
    want_rows = std::max(want_rows, row + 1);
    if ((row == 0) &&
        ((rows.extent_int(0) < want_rows) || (rows.extent_int(1) != len))) {
      rows = DvceArray2D<int>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "lat_offset_pool"),
          std::max(want_rows, kLATOffsetPoolMinRows), len);
    }
    if ((row < rows.extent_int(0)) && (rows.extent_int(1) == len)) {
      return Kokkos::subview(rows, row, Kokkos::make_pair(0, len));
    }
    // Kokkos only accepts a label as std::string or a character array, never a pointer.
    return DvceArray1D<int>(std::string(label), len);
  }
};

// Cached LAT masks are immutable, but their allocations can be recycled after AMR or a
// scheduling change.  The MeshBlockPack generation makes the pointer pair a stable,
// collision-free key across those cache lifetimes.
struct LATLayoutCacheKey {
  std::uint64_t generation;
  std::uintptr_t first_mask;
  std::uintptr_t second_mask;

  bool operator==(const LATLayoutCacheKey &other) const {
    return generation == other.generation && first_mask == other.first_mask &&
           second_mask == other.second_mask;
  }
};

struct LATLayoutCacheKeyHash {
  std::size_t operator()(const LATLayoutCacheKey &key) const {
    const std::size_t generation_hash = std::hash<std::uint64_t>{}(key.generation);
    const std::size_t first_hash = std::hash<std::uintptr_t>{}(key.first_mask);
    const std::size_t second_hash = std::hash<std::uintptr_t>{}(key.second_mask);
    return generation_hash ^ (first_hash << 1) ^ (second_hash << 2);
  }
};

// A LAT layout is a compact, exact subset of the immutable rank-packed topology.
// Keeping the selected entry IDs makes cache lookup collision-free.
struct RankPackedVarLayout {
  std::uint64_t topology_version;
  std::uint64_t lat_metadata_version;
  std::uint64_t selection_hash;
  std::vector<int> send_entry_ids;
  std::vector<int> recv_header_ids;
  std::vector<RankPackedVarMessage> send_msgs;
  std::vector<RankPackedVarMessage> recv_msgs;
#if MPI_PARALLEL_ENABLED
  // Persistent MPI_Send_init/MPI_Recv_init handles bound to send_msgs/recv_msgs and to
  // the aggregate payload buffers.  They live with the layout because a layout IS the
  // (peer, offset, count) tuple set: the dense path has one such set and each LAT mask
  // selection has its own.  Created on first use; freed wherever lat_var_layouts_ is
  // cleared, which is exactly where the offsets they encode stop being valid.
  std::vector<MPI_Request> send_preqs;
  std::vector<MPI_Request> recv_preqs;
#endif
  DvceArray1D<int> send_offsets;
  DvceArray1D<int> recv_offsets;
  // Pool row backing send_offsets; recv_offsets is the next row.  -1 when the maps were
  // never assigned.  The row is 2*(index in the layout vector), which is why the pool can
  // only ever be reshaped while that vector is empty.
  int offsets_row = -1;
};

//! \struct RankPackedFluxLayout
//! \brief exact LAT subset of off-rank flux payloads, grouped by peer

struct RankPackedFluxLayout {
  std::uint64_t topology_version;
  std::uint64_t lat_metadata_version;
  std::uint64_t selection_hash;
  bool integrated;
  std::vector<int> send_entry_ids;
  std::vector<int> recv_entry_ids;
  std::vector<RankPackedVarMessage> send_msgs;
  std::vector<RankPackedVarMessage> recv_msgs;
  // Persistent MPI_Send_init/MPI_Recv_init handles, exactly as in RankPackedVarLayout:
  // bound to send_msgs/recv_msgs and to rank_sendbuf_flux_/rank_recvbuf_flux_.  A LAT
  // bin-step re-posts the identical (peer, offset, count) tuples every time, so the one
  // registration these hold replaces a fresh MPI_Isend/MPI_Irecv per phase -- which, with
  // UCX_RCACHE_ENABLE=n, re-registered the GPU buffer on every call.  Created on first
  // use; freed wherever lat_flux_layouts_ is cleared, i.e. exactly where the offsets they
  // encode stop being valid.
  std::vector<MPI_Request> send_preqs;
  std::vector<MPI_Request> recv_preqs;
  DvceArray1D<int> send_offsets;
  DvceArray1D<int> recv_offsets;
  // Pool row backing send_offsets, as in RankPackedVarLayout.  Stays -1 for the
  // face-centered owner, which has not been moved onto the pool.
  int offsets_row = -1;
};
#endif

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class MeshBoundaryValues
//  \brief Abstract base class for boundary values for different kinds of Mesh variables

class MeshBoundaryValues {
 public:
  MeshBoundaryValues(MeshBlockPack *ppack, ParameterInput *pin, bool z4c);
  virtual ~MeshBoundaryValues();

  // data for all 56 buffers in most general 3D case. Not all elements used in most cases.
  // However each MeshBoundaryBuffer is lightweight, so the convenience of fixed array
  // sizes and index values for array elements outweighs cost of extra memory.
  MeshBoundaryBuffer sendbuf[56], recvbuf[56];
  DvceArray1D<MeshBoundaryBufferDevice> sendbuf_device, recvbuf_device;

  // Owners whose exchanges go exclusively through the standard PackAndSend*/
  // RecvAndUnpack* machinery may set this before InitializeBuffers to stub the
  // send-side per-neighbor vars payloads when nranks>1: the rank-packed path
  // (always active there for CC/FC) packs off-rank data into the aggregate
  // buffers and writes same-rank payloads directly into the destination
  // recvbuf, so sendbuf vars are never touched.  Receive-side vars must stay
  // full (same-rank direct writes and aggregate-fallback reads target them).
  // MultigridBoundaryValues packs and Isends from sendbuf vars and must NOT
  // set this.  Flux payloads are unaffected.
  bool stub_send_vars = false;
  // Receive-side companion of stub_send_vars: on multi-rank runs the rank-packed path
  // writes same-rank same-level payloads straight into the destination array and
  // off-rank payloads into the rank aggregate, so recvbuf[n].vars only ever holds
  // same-rank cross-level (coarse/fine) data.  Owners that opt in shrink every recv
  // slot to that footprint (0.4 MB per 32^3 block for 6 variables).
  bool stub_recv_same_vars = false;
  // The mirror image of the two flags above: owners that exchange nothing BUT same-level
  // payloads size their vars buffers from isame_ndat alone instead of the maximum over
  // isame/icoar/ifine.  MultigridBoundaryValues is the one such owner -- its fine/coarse
  // ghosts travel through the separate fc_remote_* path in multigrid.cpp, and icoar and
  // ifine are read nowhere under src/multigrid or src/gravity.  On a refined mesh of
  // 32^3 blocks at one multigrid ghost cell that is 13 MB per rank instead of 54 MB at
  // 128 blocks, because the subblock slots a multigrid exchange never uses still carry a
  // full coarse and fine footprint each.  The flag must be set before InitializeBuffers
  // so that the seed allocation already honours it.
  bool same_level_vars_only = false;

  // Owners may set this before InitializeBuffers to declare that their SEND-side flux
  // buffers never carry a same-level payload, so those buffers only need the coarse
  // footprint (iflxc_ndat) instead of max(iflxs_ndat, iflxc_ndat).  On a 3D AMR mesh the
  // same-level plane is ~1.7x the coarse one, so this is a large allocation.
  //
  // The invariant, for MeshBoundaryValuesCC (flux_correct_cc.cpp):
  //   * a same-level flux send requires send_to_same, which requires lat_enabled (:373);
  //   * the only writes to sendbuf.flux (:454, :507, :550) sit in the `else` arm of
  //     `if (rank_packed_flux)`, and rank_packed_flux == rank_packed_lat_flux_active_;
  //   * InitFluxRecv assigns rank_packed_lat_flux_active_ = lat_enabled (:1134) and is
  //     always queued ahead of PackAndSendFluxCC in every task list that uses it;
  //   * InvalidateRankPackedVarMetadata (the only reset, bvals.cpp:196) is reached only
  //     from EnsureRankPackedVarMetadata and InitializeBuffers, never between those two.
  // Hence lat_enabled implies rank_packed_flux at pack time, and the sendbuf.flux arm is
  // reachable only with same_level == false.  PackAndSendFluxCC re-checks this at runtime
  // and aborts rather than writing out of bounds if the ordering ever changes.
  //
  // MeshBoundaryValuesFC must NOT set this: flux_correct_fc.cpp:374-399 selects the
  // same-level EMF bounds whenever nghbr.lev == mblev with no lat_enabled gate, and
  // Isends nvar*iflxs_ndat at :786.  Recv-side buffers must always stay full size.
  bool stub_flux_same = false;

  // constant inflow states at each face, initialized in problem generator
  DualArray2D<Real> u_in, b_in, i_in;

#if MPI_PARALLEL_ENABLED
  // unique MPI communicators for each case (variables/fluxes)
  MPI_Comm comm_vars, comm_flux;

  // Rank-packed variable boundary communication. This aggregates all off-rank
  // (MeshBlock,neighbor) payloads by peer rank, reducing many small MPI messages.
  bool rank_packed_bvals_active_;
  bool rank_packed_lat_capable_;
  bool rank_packed_lat_active_;
  // RecvAndUnpack kernels consume MPI receive storage asynchronously.  Before a
  // later MPI_Irecv reuses that storage, InitRecv must establish completion.
  bool recv_unpack_pending_;
#if defined(KOKKOS_ENABLE_CUDA)
  // ...but only the UNPACK has to be complete, not the whole device queue, which is
  // what a DevExeSpace().fence() waits for.  In the LAT refresh path InitRecv is the
  // first thing called after the previous refresh's prolongation/BC/C2P kernels have
  // been submitted, so the plain fence drained all of those before a single receive
  // could be posted -- and a receive that is not posted cannot start its rendezvous,
  // so no peer traffic could overlap that work.  An event recorded on the Kokkos
  // stream right after the unpack is reached as soon as the unpack retires, with the
  // rest of the queue left running, which is exactly the guarantee the buffers need.
  //
  // Only the unpack kernels read this object's receive storage: same-rank payloads are
  // written by the pack kernel into DIFFERENT (block,neighbour) slots than any off-rank
  // Irecv target, and prolongation reads the coarse arrays, not the buffers.
  //
  // recv_unpack_event_valid_ is set ONLY by MarkRecvUnpackPending(); anything that
  // raises recv_unpack_pending_ by hand (MultigridBoundaryValues::RecvAndUnpackMG does,
  // and pairs it with its own fence) leaves it clear, and WaitRecvUnpackComplete() then
  // falls back to the full fence.
  cudaEvent_t recv_unpack_event_ = nullptr;
  bool recv_unpack_event_valid_ = false;
#endif
  bool show_rank_packed_bvals_stats_;
  int rank_packed_bvals_nvars_;
  int rank_packed_bvals_nmb_;
  int rank_packed_bvals_nnghbr_;
  std::uint64_t rank_packed_bvals_topology_version_;
  std::vector<RankPackedVarEntry> send_var_entries_, recv_var_entries_;
  std::vector<RankPackedVarMessage> send_var_msgs_, recv_var_msgs_;
  // Persistent request handles for the dense (non-LAT) rank-packed path.  The LAT path
  // keeps its own per-layout pair inside RankPackedVarLayout; VarReqs() selects between
  // them.  Nothing here is ever MPI_REQUEST_NULL between create and free.
  std::vector<MPI_Request> send_var_preqs_, recv_var_preqs_;
  // Which LAT layout's requests were last MPI_Startall'ed, so ClearSend/ClearRecv and
  // the completion test wait on the requests that were actually started rather than on
  // whichever layout the mask happens to select later in the step.  -1 = dense path.
  int lat_send_reqs_layout_ = -1;
  int lat_recv_reqs_layout_ = -1;
  bool send_var_reqs_started_ = false;
  bool recv_var_reqs_started_ = false;
  // Per-message completion, because a persistent request that has completed goes
  // INACTIVE rather than to MPI_REQUEST_NULL: waiting on it again returns an EMPTY
  // status, and the payload-size check would then compare against a count of 0.  Raised
  // for the whole set at once by TestVarRecvComplete (MPI_Testall is all-or-nothing) and
  // read by ClearRecv, which must skip what has already been verified.
  std::vector<unsigned char> recv_var_done_;
  // Audited rather than asserted: created must equal freed by the time this object dies.
  std::uint64_t persistent_var_reqs_created_ = 0;
  std::uint64_t persistent_var_reqs_freed_ = 0;
  DvceArray1D<Real> rank_sendbuf_vars_, rank_recvbuf_vars_;
  // Allocated capacity in Reals of the two aggregate payload buffers above.  Grow-only
  // and MiB-granular (RankPackedBufCapacity), so a topology change that moves the
  // off-rank payload total by a few entries no longer reallocates ~63 MiB of device
  // memory.  Only the ALLOCATION is sticky: BuildRankPackedVarMetadata still rebuilds
  // every offset and message descriptor whenever its topology key changes.
  int rank_sendbuf_vars_cap_ = 0;
  int rank_recvbuf_vars_cap_ = 0;
  HostArray1D<int> rank_sendhdr_vars_, rank_recvhdr_vars_;
  DvceArray1D<int> send_agg_offset_, recv_agg_offset_;
  std::vector<RankPackedVarLayout> lat_var_layouts_;
  LATOffsetPool lat_var_offset_pool_;
  std::unordered_map<LATLayoutCacheKey, int, LATLayoutCacheKeyHash>
      lat_var_layout_cache_;
  std::uint64_t lat_var_layout_cache_generation_{0};
  int lat_var_layout_index_;

  // LAT cell-centered flux correction uses a separate aggregate buffer because its
  // selected face set and payload sizes differ from variable boundary communication.
  bool rank_packed_lat_flux_active_;
  int rank_packed_flux_nvars_;
  int rank_packed_flux_nmb_;
  int rank_packed_flux_nnghbr_;
  bool rank_packed_flux_same_level_;
  std::uint64_t rank_packed_flux_topology_version_;
  DvceArray1D<Real> rank_sendbuf_flux_, rank_recvbuf_flux_;
  // Capacity of the two buffers above, same policy as rank_*buf_vars_cap_.  Maintained
  // by the cell-centered owner only (flux_correct_cc.cpp); MeshBoundaryValuesFC still
  // sizes its own flux buffers exactly and leaves these at 0, which is harmless because
  // a given object only ever runs one of the two paths.
  int rank_sendbuf_flux_cap_ = 0;
  int rank_recvbuf_flux_cap_ = 0;
  std::vector<RankPackedFluxLayout> lat_flux_layouts_;
  LATOffsetPool lat_flux_offset_pool_;
  std::unordered_map<LATLayoutCacheKey, int, LATLayoutCacheKeyHash>
      lat_flux_layout_cache_;
  std::uint64_t lat_flux_layout_cache_generation_{0};
  int lat_flux_layout_index_;
  // Which LAT flux layout's persistent requests were last MPI_Startall'ed, mirroring
  // lat_send_reqs_layout_/lat_recv_reqs_layout_ on the variable side.  -1 = none started,
  // which is the only "dense" state the flux path has (it is LAT-only).
  int lat_flux_send_reqs_layout_ = -1;
  int lat_flux_recv_reqs_layout_ = -1;
  bool send_flux_reqs_started_ = false;
  bool recv_flux_reqs_started_ = false;
  // Raised once RecvAndUnpackFlux*'s MPI_Testall has verified every payload size.  That
  // test is all-or-nothing -- no request is modified unless all completed -- so a single
  // flag covers the whole peer set, and ClearFluxRecv then knows the requests are already
  // INACTIVE (a completed persistent request never becomes MPI_REQUEST_NULL, and waiting
  // on it again would hand back an empty status with count 0).
  bool recv_flux_counts_verified_ = false;
  // Scratch for the MPI_Testall status array; grown to the peer count once, then reused.
  std::vector<MPI_Status> mpi_statuses_;
#endif

  //functions
  virtual void InitSendIndices(MeshBoundaryBuffer &buf,int x,int y,int z,int a,int b)=0;
  virtual void InitRecvIndices(MeshBoundaryBuffer &buf,int x,int y,int z,int a,int b)=0;
  void InitializeBuffers(const int nvar, const int nfluxvar = -1);
  // Grow the per-neighbor send/recv buffers so they can hold `nmb` MeshBlocks.  The
  // buffers are seeded from the LIVE block count in InitializeBuffers and grow only from
  // there, which is the same high-water discipline the physics field arrays already use
  // (Hydro::ResizeMeshBlockStorage).  Returns true if an allocation actually changed.
  //
  // Must be called at a synchronized point with no exchange in flight: the allocations
  // do not preserve their contents and every base pointer moves.  The topology
  // transaction (MeshRefinement::RedistAndRefineMeshBlocks, via each physics module's
  // ResizeMeshBlockStorage) is the intended caller.
  bool ResizeBuffers(const int nmb);
  void RefreshDeviceBufferMetadata();
  // Build the symmetric rank-packed variable topology at an explicitly synchronized,
  // unmasked point. Delayed auxiliary activation uses this before its first exchange.
  void PrepareRankPackedVarMetadata(const int nvars);

#if MPI_PARALLEL_ENABLED
  //! \brief persistent request vector for LAT layout `lat_layout` (-1 = dense path)
  std::vector<MPI_Request> &VarReqs(bool is_send, int lat_layout);
  //! \brief persistent flux request vector for LAT layout `lat_layout`
  //!
  //! Returns nullptr when the index addresses no layout: unlike the variable path there
  //! is no dense rank-packed flux exchange, so that state simply means "nothing started".
  std::vector<MPI_Request> *FluxReqs(bool is_send, int lat_layout);
  //! \brief create the persistent requests for `msgs` if they do not exist yet
  void EnsurePersistentReqs(const std::vector<RankPackedVarMessage> &msgs,
                            Real *base, bool is_send, MPI_Comm comm, int tag,
                            std::vector<MPI_Request> *reqs);
  //! \brief EnsurePersistentReqs on comm_vars with the variable-boundary tag
  void EnsurePersistentVarReqs(const std::vector<RankPackedVarMessage> &msgs,
                               Real *base, bool is_send,
                               std::vector<MPI_Request> *reqs);
  //! \brief EnsurePersistentReqs on comm_flux with the flux-correction tag
  void EnsurePersistentFluxReqs(const std::vector<RankPackedVarMessage> &msgs,
                                Real *base, bool is_send,
                                std::vector<MPI_Request> *reqs);
  //! \brief MPI_Request_free every handle in `reqs` and empty it
  //!
  //! `started` means an MPI_Startall was issued and not yet drained.  Freeing an ACTIVE
  //! persistent request is erroneous, so that is a hard error at a metadata rebuild --
  //! where ClearSend/ClearRecv guarantee it cannot happen -- and a drain at teardown.
  void FreePersistentVarReqs(std::vector<MPI_Request> *reqs, bool started,
                             bool drain_at_teardown);
  //! \brief free every persistent request this object owns, dense and per-layout
  void FreeAllPersistentVarReqs(bool drain_at_teardown);
  //! \brief free the per-layout requests; call immediately before lat_var_layouts_.clear()
  void FreeLayoutPersistentVarReqs(bool drain_at_teardown = false);
  //! \brief same for the flux layouts; call immediately before lat_flux_layouts_.clear()
  //! or before either aggregate flux buffer is reallocated
  void FreeLayoutPersistentFluxReqs(bool drain_at_teardown = false);
  //! \brief one MPI_Testall over the started rank-packed variable receives
  //!
  //! Returns true when every receive has completed (payload sizes checked then, and
  //! recv_var_done_ raised for ClearRecv), false for "not yet".  `*no_errors` is cleared
  //! on any MPI error or size mismatch.
  bool TestVarRecvComplete(bool *no_errors);
  //! \brief one MPI_Testall over the started rank-packed flux receives
  //!
  //! Returns true when every receive has completed (payload sizes checked then), false
  //! for "not yet".  `*no_errors` is cleared on any MPI error or size mismatch and
  //! `*remote_recv` is raised if this rank has any off-rank flux receive at all.
  bool TestFluxRecvComplete(bool *no_errors, bool *remote_recv);
#endif

  //! \brief note that an unpack kernel is in flight over this object's receive storage
  void MarkRecvUnpackPending();
  //! \brief same, for a caller that does NOT record the unpack event
  //!
  //! MultigridBoundaryValues unpacks on paths this class knows nothing about and pairs
  //! them with its own full fence.  It must clear the event validity flag as it raises
  //! the pending flag, or WaitRecvUnpackComplete() could later satisfy that pending
  //! unpack by waiting on a stale event that retired several exchanges ago.
  void MarkRecvUnpackPendingFullFence() {
#if MPI_PARALLEL_ENABLED
    recv_unpack_pending_ = true;
#if defined(KOKKOS_ENABLE_CUDA)
    recv_unpack_event_valid_ = false;
#endif
#endif
  }
  //! \brief wait for that unpack, and only that unpack, to retire
  //!
  //! Replaces `if (recv_unpack_pending_) { DevExeSpace().fence(); ... }`.  Falls back to
  //! the full-device fence when no valid event was recorded.
  void WaitRecvUnpackComplete();

  TaskStatus InitRecv(const int nvar);
  virtual TaskStatus InitFluxRecv(const int nvar)=0;
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  TaskStatus ClearFluxRecv();
  TaskStatus ClearFluxSend();

  // BCs associated with various physics modules
  static void HydroBCs(MeshBlockPack *pp, DualArray2D<Real> uin, DvceArray5D<Real> u0);
  static void HydroBCsCoarse(MeshBlockPack *pp, DualArray2D<Real> uin,
                             DvceArray5D<Real> coarse_u0);
  static void BFieldBCs(MeshBlockPack *pp, DualArray2D<Real> bin, DvceFaceFld4D<Real> b0);
  static void BFieldBCsCoarse(MeshBlockPack *pp, DualArray2D<Real> bin,
                              DvceFaceFld4D<Real> coarse_b0);
  static void RadiationBCs(MeshBlockPack *pp,DualArray2D<Real> iin,DvceArray5D<Real> i0);
  static void RadiationBCsCoarse(MeshBlockPack *pp, DualArray2D<Real> iin,
                                 DvceArray5D<Real> coarse_i0);
  static void Z4cBCs(MeshBlockPack *pp, DualArray2D<Real> uin, DvceArray5D<Real> u0);
  static void Z4cBCsCoarse(MeshBlockPack *pp, DualArray2D<Real> uin,
                           DvceArray5D<Real> coarse_u0);

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, Z4c, etc.)
  MeshBlockPack* pmy_pack;
  bool is_z4c_;   // flag to denote if this BoundaryValues is for Z4c module

  // Current per-neighbor buffer capacity in MeshBlocks, plus the component counts the
  // buffers were built with, so ResizeBuffers can rebuild them without the caller having
  // to remember what InitializeBuffers was given.
  int nmb_alloc_ = 0;
  int buf_nvar_ = 0;
  int buf_nflx_ = 0;
  // Round a required block count up to the capacity actually allocated.
  int BufferBlockCapacity(const int nmb_needed) const;
  // Reallocate every per-neighbor payload at a capacity of `nmb` MeshBlocks, honouring
  // the owner's footprint guarantees.  Allocation only: the caller updates nmb_alloc_
  // and must follow with RefreshDeviceBufferMetadata and the rank-packed invalidation,
  // because every base pointer moves.
  void AllocateAllBuffers(const int nmb);

#if MPI_PARALLEL_ENABLED
  bool UseRankPackedVars() const;
  void InvalidateRankPackedVarMetadata();
  void EnsureRankPackedVarMetadata(const int nvars);
  int GetVarDataSize(const MeshBoundaryBuffer &buf, int m, int n, int nvars) const;
  void BuildRankPackedVarMetadata(const int nvars);
  void PrepareLATRankPackedVarLayout();
#endif
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesCC
//  \brief Derived class implementing boundary values for cell-centered variables

//----------------------------------------------------------------------------------------
//! \struct FaceFldOrigin
//! \brief Index origin of a face-centred flux register stored on a sub-block band.
//! The flux-correction routines index a register as flx.x?f(m,v,k-k0,j-j0,i-i0), so a
//! module may allocate its face fluxes on the interior faces only (a cell-centred flux
//! exists on no ghost face) and pass the origin (ks,js,is).  The default (0,0,0) is the
//! ghost-extended block layout that the MHD/Hydro registers use.
struct FaceFldOrigin {
  int k = 0;
  int j = 0;
  int i = 0;
};

class MeshBoundaryValuesCC : public MeshBoundaryValues {
 public:
  MeshBoundaryValuesCC(MeshBlockPack *ppack, ParameterInput *pin, bool z4c);

  //functions
  void InitSendIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  // Whether the LAT stage-1 dense snapshot (a_stage1 below) may be stored on the owned
  // cells only, with origin (ks,js,is).  PackAndSendCC reads it only on the send ranges
  // of InitSendIndices (buffs_cc.cpp): a same-level range is an ng-wide strip of owned
  // cells, inside them when every extended axis holds nx >= ng, and a to-finer range is
  // that strip shifted by cnx - ng, inside them when nx >= 2*ng.  Sends to a coarser
  // neighbour read the coarse snapshot, which keeps its ghost-extended layout.
  static bool LATStage1OwnedOnly(const RegionIndcs &indcs, const bool multilevel) {
    const int nmin = (multilevel ? 2 : 1)*indcs.ng;
    return indcs.nx1 >= nmin && (indcs.nx2 == 1 || indcs.nx2 >= nmin) &&
           (indcs.nx3 == 1 || indcs.nx3 >= nmin);
  }
  // functions to communicate CC data
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                           DvceArray5D<Real> *a_start = nullptr,
                           Real target_time = 0.0,
                           DvceArray5D<Real> *ca_start = nullptr,
                           DvceArray5D<Real> *a_stage1 = nullptr,
                           DvceArray5D<Real> *ca_stage1 = nullptr,
                           DvceArray1D<int> *a_stage1_valid = nullptr,
                           bool limit_dense_to_endpoints = true);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  // functions to communicate fluxes of CC data.  With `excised`, a fine face whose two
  // cells are both marked in it enters the flux restricted to a coarser neighbour as zero.
  TaskStatus PackAndSendFluxCC(DvceFaceFld5D<Real> &flx,
                               DvceFaceFld5D<Real> *extra_flx = nullptr,
                               bool time_integrate = false, Real weight = 1.0,
                               FaceFldOrigin origin = {},
                               const DvceArray4D<bool> *excised = nullptr);
  TaskStatus RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx,
                                 DvceFaceFld5D<Real> *extra_flx = nullptr,
                                 FaceFldOrigin origin = {});
  TaskStatus RecvAndAccumulateFluxCC(DvceFaceFld5D<Real> &accum, Real scale,
                                     bool compact_faces = false,
                                     DvceFaceFld5D<Real> *extra_accum = nullptr,
                                     DvceFaceFld5D<Real> *sync_flx = nullptr,
                                     DvceFaceFld5D<Real> *extra_sync_flx = nullptr,
                                     Real sync_integrated_weight = 0.0,
                                     bool accumulate_sync_mismatch = true,
                                     FaceFldOrigin origin = {},
                                     bool store_pending_mismatch = false);
  // Union stage-1 predictor: the fine-minus-own face flux mismatch captured by
  // RecvAndAccumulateFluxCC(store_pending_mismatch) on every unequal-factor coarse/fine
  // face, added to the own flux of the slower bin's corrector stage while that finer
  // neighbour is still pending (flux_correct_cc.cpp).
  void AddPendingFineFluxMismatchCC(DvceFaceFld5D<Real> &flx,
                                    DvceFaceFld5D<Real> *extra_flx = nullptr,
                                    FaceFldOrigin origin = {});
  // Whether the call above adds anything in the current stage.
  bool PendingFineFluxMismatchDue() const;
  // First-order flux correction on the edge cells where a face with that estimate meets
  // a face shared with an active block of the same level and cadence, for a fluid whose
  // FOFC flags are not exchanged (Hydro, DynGRMHD; flux_correct_cc.cpp).  The fluid tests
  // its edge cells (ForEachMeshBlockEdge) before adding the estimate and stashes those
  // flags here; after its first FOFC test, which sees the estimate, the reconcile puts
  // them back and withholds the estimate from a cell whose test only it failed.
  void StashPendingEdgeFOFCFlags(const DvceArray4D<bool> &flag,
                                 const DvceArray5D<bool> *scalar_flag = nullptr,
                                 int nscalar = 0);
  void ReconcilePendingEdgeFOFCFlags(const DvceArray4D<bool> &flag,
                                     const DvceArray5D<bool> *scalar_flag, int nscalar,
                                     DvceFaceFld5D<Real> &flx,
                                     DvceFaceFld5D<Real> *extra_flx = nullptr,
                                     FaceFldOrigin origin = {});
  // Calls test(il, iu, jl, ju, kl, ku) on each edge of the interior of a MeshBlock (on
  // each corner in 2D), so that every cell on two of its faces is in exactly one call.
  template <typename EdgeTest>
  static void ForEachMeshBlockEdge(const RegionIndcs &indcs, const bool three_d,
                                   EdgeTest &&test) {
    const int iside[2] = {indcs.is, indcs.ie};
    const int jside[2] = {indcs.js, indcs.je};
    const int kside[2] = {indcs.ks, indcs.ke};
    const int kl = three_d ? indcs.ks + 1 : indcs.ks;
    const int ku = three_d ? indcs.ke - 1 : indcs.ke;
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        if (kl <= ku) test(iside[a], iside[a], jside[b], jside[b], kl, ku);
        if (three_d) {
          test(indcs.is, indcs.ie, jside[a], jside[a], kside[b], kside[b]);
          if (indcs.js < indcs.je - 1) {
            test(iside[a], iside[a], indcs.js + 1, indcs.je - 1, kside[b], kside[b]);
          }
        }
      }
    }
  }
  // Union stage-1 predictor: this block's own flux on the faces the capture above will
  // replace, stored in the pending registers before a first-order flux correction can
  // replace it, so the capture measures the mismatch against the operator the corrector
  // adds it to (flux_correct_cc.cpp).  Takes the arguments of the owner's capture.
  void SnapshotPendingOwnFluxCC(const DvceFaceFld5D<Real> &accum,
                                const DvceFaceFld5D<Real> *extra_accum,
                                const DvceFaceFld5D<Real> &flx,
                                const DvceFaceFld5D<Real> *extra_flx,
                                FaceFldOrigin origin = {});
  void ReserveLATPendingMismatch(const DvceFaceFld5D<Real> &accum,
                                 const DvceFaceFld5D<Real> *extra_accum);
  DvceArray5D<Real> lat_pending_mismatch1_, lat_pending_mismatch2_;
  DvceArray5D<Real> lat_pending_mismatch3_;
  DvceArray5D<Real> lat_pending_extra1_, lat_pending_extra2_, lat_pending_extra3_;
  bool lat_pending_captured_ = false;  // a flux receive has filled them
  bool lat_pending_own_snapshot_ = false;  // they hold this stage's own flux
  // One entry per edge cell, (nmb, 12, max nx): the stashed flags, then the withholding.
  DvceArray3D<int> lat_pending_edge_;
  bool lat_pending_edge_stashed_ = false;  // this stage's flags are in it
  void ApplyPendingFineFluxMismatchCC(DvceFaceFld5D<Real> &flx,
                                      DvceFaceFld5D<Real> *extra_flx,
                                      FaceFldOrigin origin, bool withhold);

#if MPI_PARALLEL_ENABLED
  void PrepareLATRankPackedFluxLayout(const int nvars);
#endif

  // functions to prolongate conserved and primitive CC variables
  void FillCoarseInBndryCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
       bool is_z4c=false);
  void FillCoarseUserBndryCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, bool is_z4c=false);
  void ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim);
  void PrimToConsFineBndry(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons);
  void ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                             DvceArray5D<Real> &prim);
  void PrimToConsFineBndry(DvceArray5D<Real> &prim, const DvceFaceFld4D<Real> &b,
                           DvceArray5D<Real> &cons);
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesFC
//  \brief Derived class implementing boundary values for face-centered vector fields

class MeshBoundaryValuesFC : public MeshBoundaryValues {
 public:
  MeshBoundaryValuesFC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void FillCoarseInBndryFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void FillCoarseUserBndryFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  TaskStatus PackAndSendFluxFC(DvceEdgeFld4D<Real> &flx);
  TaskStatus RecvAndUnpackFluxFC(DvceEdgeFld4D<Real> &flx);
  void SumBoundaryFluxes(DvceEdgeFld4D<Real> &flx, const bool same_level,
                         DvceArray2D<int> &nflx);
  void ZeroFluxesAtBoundaryWithFiner(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);
  void AverageBoundaryFluxes(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);

 private:
  // Reused EMF-count scratch for RecvAndUnpackFluxFC.  Every call fully rewrites
  // the rows of the blocks it works on before any consumer reads them, so the
  // buffer only needs reallocation when the pack size changes.
  DvceArray2D<int> fc_flux_count_scratch_;
};

template <int n = 56>
struct BoundaryData { // aggregate and POD (even when MPI_PARALLEL is defined)
  static constexpr int kMaxNeighbor = n;
  // KGF: "nbmax" only used in bvals_var.cpp, Init/DestroyBoundaryData()
  int nbmax;  //!> actual maximum number of neighboring MeshBlocks
  // currently, sflag[] is only used by Multgrid (send buffers are reused each stage in
  // red-black comm. pattern; need to check if they are available) and shearing box
  BoundaryStatus flag[kMaxNeighbor], sflag[kMaxNeighbor];
  Real *send[kMaxNeighbor], *recv[kMaxNeighbor];
#ifdef MPI_PARALLEL
  MPI_Request req_send[kMaxNeighbor], req_recv[kMaxNeighbor];
#endif
};

//----------------------------------------------------------------------------------------
//! \struct ParticleLocationData
//! \brief data describing location of data for particles communicated with MPI

struct ParticleLocationData {
  int prtcl_indx;   // index in particle array
  int dest_gid;     // GID of target MeshBlock
  int dest_rank;    // rank of target MeshBlock
};

// Custom operators to sort ParticleLocationData array by dest_rank or prtcl_indx
struct {
  bool operator()(ParticleLocationData a, ParticleLocationData b)
    const { return a.dest_rank < b.dest_rank; }
} SortByRank;
struct {
  bool operator()(ParticleLocationData a, ParticleLocationData b)
    const { return a.prtcl_indx < b.prtcl_indx; }
} SortByIndex;

//----------------------------------------------------------------------------------------
//! \struct ParticleMessageData
//! \brief Data describing MPI messages containing particles

struct ParticleMessageData {
  int sendrank;  // rank of sender
  int recvrank;  // rank of receiver
  int nprtcls;   // number of particles in message
  ParticleMessageData(int a, int b, int c) :
    sendrank(a), recvrank(b), nprtcls(c) {}
};

//----------------------------------------------------------------------------------------
//! \class ParticlesBoundaryValues
//  \brief Defines boundary values class for particles

namespace particles {
class ParticlesBoundaryValues {
 public:
  ParticlesBoundaryValues(particles::Particles *ppart, ParameterInput *pin);
  ~ParticlesBoundaryValues();

  int nprtcl_send, nprtcl_recv;
  DualArray1D<ParticleLocationData> sendlist;

  // Data needed to count number of messages and particles to send between ranks
  int nsends; // number of MPI sends to neighboring ranks on this rank
  int nrecvs; // number of MPI recvs from neighboring ranks on this rank
  std::vector<int> nsends_eachrank;                // length nranks
  std::vector<ParticleMessageData> sends_thisrank; // length nsends
  std::vector<ParticleMessageData> recvs_thisrank; // length nrecvs
  std::vector<ParticleMessageData> sends_allranks; // length ncounts summed over ranks

#if MPI_PARALLEL_ENABLED
  DvceArray1D<Real> prtcl_rsendbuf, prtcl_rrecvbuf;
  DvceArray1D<int>  prtcl_isendbuf, prtcl_irecvbuf;
  std::vector<MPI_Request> rrecv_req, rsend_req;  // vectors of requests for Reals
  std::vector<MPI_Request> irecv_req, isend_req;  // vectors of requests for ints
  MPI_Comm mpi_comm_part;                       // unique MPI communicators for particles
#endif

  //functions
  TaskStatus SetNewPrtclGID();
  TaskStatus CountSendsAndRecvs();
  TaskStatus InitPrtclRecv();
  TaskStatus ClearPrtclRecv();
  TaskStatus PackAndSendPrtcls();
  TaskStatus ClearPrtclSend();
  TaskStatus RecvAndUnpackPrtcls();

 protected:
  particles::Particles* pmy_part;
};
} // namespace particles

#endif // BVALS_BVALS_HPP_
