//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief constructors and initializers for both particle and Mesh variable boundary
//! classes.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <algorithm> // max
#include <map>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

namespace {
void CopyBufferIndcs(const MeshBufferIndcs &src, MeshBufferIndcsDevice &dst) {
  dst.bis = src.bis;
  dst.bie = src.bie;
  dst.bjs = src.bjs;
  dst.bje = src.bje;
  dst.bks = src.bks;
  dst.bke = src.bke;
}

void CopyBoundaryBuffer(const MeshBoundaryBuffer &src, MeshBoundaryBufferDevice &dst) {
  for (int v=0; v<3; ++v) {
    CopyBufferIndcs(src.isame[v], dst.isame[v]);
    CopyBufferIndcs(src.icoar[v], dst.icoar[v]);
    CopyBufferIndcs(src.ifine[v], dst.ifine[v]);
    CopyBufferIndcs(src.iprol[v], dst.iprol[v]);
    CopyBufferIndcs(src.iflux_same[v], dst.iflux_same[v]);
    CopyBufferIndcs(src.iflux_coar[v], dst.iflux_coar[v]);
  }
  CopyBufferIndcs(src.isame_z4c, dst.isame_z4c);
  dst.isame_ndat = src.isame_ndat;
  dst.isame_z4c_ndat = src.isame_z4c_ndat;
  dst.icoar_ndat = src.icoar_ndat;
  dst.ifine_ndat = src.ifine_ndat;
  dst.iflxs_ndat = src.iflxs_ndat;
  dst.iflxc_ndat = src.iflxc_ndat;
  dst.vars.data = src.vars.data();
  dst.vars.stride0 = src.vars.stride_0();
  dst.flux.data = src.flux.data();
  dst.flux.stride0 = src.flux.stride_0();
}
} // namespace

//----------------------------------------------------------------------------------------
// MeshBoundaryValues constructor:

MeshBoundaryValues::MeshBoundaryValues(MeshBlockPack *pp, ParameterInput *pin, bool z4c) :
  u_in("uin",1,1),
  b_in("bin",1,1),
  i_in("iin",1,1),
#if MPI_PARALLEL_ENABLED
  comm_vars(MPI_COMM_NULL),
  comm_flux(MPI_COMM_NULL),
  rank_packed_bvals_active_(false),
  rank_packed_lat_capable_(false),
  rank_packed_lat_active_(false),
  recv_unpack_pending_(false),
  show_rank_packed_bvals_stats_(pin->GetOrAddBoolean("mesh",
                                  "show_rank_packed_bvals_stats", false)),
  rank_packed_bvals_nvars_(-1),
  rank_packed_bvals_nmb_(-1),
  rank_packed_bvals_nnghbr_(-1),
  rank_packed_bvals_topology_version_(0),
  rank_sendbuf_vars_("rank_sendbuf_vars",1),
  rank_recvbuf_vars_("rank_recvbuf_vars",1),
  rank_sendhdr_vars_("rank_sendhdr_vars",1),
  rank_recvhdr_vars_("rank_recvhdr_vars",1),
  send_agg_offset_("send_agg_offset",1),
  recv_agg_offset_("recv_agg_offset",1),
  lat_var_layout_index_(-1),
  rank_packed_lat_flux_active_(false),
  rank_packed_flux_nvars_(-1),
  rank_packed_flux_nmb_(-1),
  rank_packed_flux_nnghbr_(-1),
  rank_packed_flux_same_level_(false),
  rank_packed_flux_topology_version_(0),
  rank_sendbuf_flux_("rank_sendbuf_flux",1),
  rank_recvbuf_flux_("rank_recvbuf_flux",1),
  lat_flux_layout_index_(-1),
#endif
  pmy_pack(pp),
  is_z4c_(z4c) {
  // allocate vector of status flags and MPI requests (if needed)
  int nnghbr = pmy_pack->pmb->nnghbr;

#if MPI_PARALLEL_ENABLED
  // Initialize all 56 MPI request pointers to nullptr first
  for (int n=0; n<56; ++n) {
    sendbuf[n].vars_req = nullptr;
    sendbuf[n].flux_req = nullptr;
    recvbuf[n].vars_req = nullptr;
    recvbuf[n].flux_req = nullptr;
  }
#endif

  // sendbuf and recvbuf are fixed-length [56-element] arrays
  // Initialize some of the data in appropriate elements based on dimensionality of
  // problem (indicated by value of nnghbr)
  for (int n=0; n<nnghbr; ++n) {
#if MPI_PARALLEL_ENABLED
    // allocate vector of MPI requests (if needed)
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    sendbuf[n].flux_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].flux_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      sendbuf[n].flux_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].flux_req[m] = MPI_REQUEST_NULL;
    }
#endif
    // initialize data sizes in each send/recv buffer to zero
    sendbuf[n].isame_ndat = 0;
    sendbuf[n].isame_z4c_ndat = 0;
    sendbuf[n].icoar_ndat = 0;
    sendbuf[n].ifine_ndat = 0;
    sendbuf[n].iflxs_ndat = 0;
    sendbuf[n].iflxc_ndat = 0;
    recvbuf[n].isame_ndat = 0;
    recvbuf[n].isame_z4c_ndat = 0;
    recvbuf[n].icoar_ndat = 0;
    recvbuf[n].ifine_ndat = 0;
    recvbuf[n].iflxs_ndat = 0;
    recvbuf[n].iflxc_ndat = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for variables and fluxes in this BoundaryValues object
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_vars);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_flux);
#endif
}

#if MPI_PARALLEL_ENABLED
//----------------------------------------------------------------------------------------
//! \fn std::vector<MPI_Request> &MeshBoundaryValues::VarReqs(bool, int)
//! \brief persistent request vector for a LAT layout, or for the dense path

std::vector<MPI_Request> &MeshBoundaryValues::VarReqs(bool is_send, int lat_layout) {
  if ((lat_layout >= 0) &&
      (lat_layout < static_cast<int>(lat_var_layouts_.size()))) {
    auto &layout = lat_var_layouts_[lat_layout];
    return is_send ? layout.send_preqs : layout.recv_preqs;
  }
  // Out of range means the layout vector was cleared under a recorded index; the dense
  // vectors are then the only valid storage and are empty unless the dense path is live.
  return is_send ? send_var_preqs_ : recv_var_preqs_;
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<MPI_Request> *MeshBoundaryValues::FluxReqs(bool, int)
//! \brief persistent flux request vector for a LAT flux layout

std::vector<MPI_Request> *MeshBoundaryValues::FluxReqs(bool is_send, int lat_layout) {
  if ((lat_layout >= 0) &&
      (lat_layout < static_cast<int>(lat_flux_layouts_.size()))) {
    auto &layout = lat_flux_layouts_[lat_layout];
    return is_send ? &layout.send_preqs : &layout.recv_preqs;
  }
  // The flux exchange is LAT-only, so there is no dense fallback storage: an index that
  // addresses no layout means nothing was ever started under it.
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::EnsurePersistentReqs(...)
//! \brief bind persistent MPI requests to the aggregate payload buffers, once
//!
//! The (peer, offset, count) tuples are fixed for the lifetime of the metadata that
//! produced them, and the aggregate buffers only move in BuildRankPackedVarMetadata
//! (bvals.cpp:461-469) or in PrepareLATRankPackedFluxLayout, which free these first.  So
//! one MPI_Send_init/MPI_Recv_init per message covers every subsequent step, and
//! MPI_Startall replaces the per-step MPI_Isend/MPI_Irecv -- which, with
//! UCX_RCACHE_ENABLE=n, re-registered the GPU buffer on every call.  Same idiom as
//! multigrid.cpp:2014,2026,2816.

void MeshBoundaryValues::EnsurePersistentReqs(
    const std::vector<RankPackedVarMessage> &msgs, Real *base, bool is_send,
    MPI_Comm comm, int tag, std::vector<MPI_Request> *reqs) {
  if (reqs->size() == msgs.size()) return;   // already bound (or both empty)
  // A partially built vector can only come from a failed create below, which exits.
  reqs->assign(msgs.size(), MPI_REQUEST_NULL);
  bool no_errors = true;
  for (std::size_t i=0; i<msgs.size(); ++i) {
    const auto &msg = msgs[i];
    const int ierr = is_send ?
        MPI_Send_init(base + msg.offset, msg.data_size, MPI_ATHENA_REAL, msg.rank, tag,
                      comm, &(*reqs)[i]) :
        MPI_Recv_init(base + msg.offset, msg.data_size, MPI_ATHENA_REAL, msg.rank, tag,
                      comm, &(*reqs)[i]);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error creating persistent rank-packed requests"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  persistent_var_reqs_created_ += msgs.size();
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::EnsurePersistentVarReqs(...)
//! \brief persistent variable-boundary requests (comm_vars, tag 1 as the Isends used)

void MeshBoundaryValues::EnsurePersistentVarReqs(
    const std::vector<RankPackedVarMessage> &msgs, Real *base, bool is_send,
    std::vector<MPI_Request> *reqs) {
  EnsurePersistentReqs(msgs, base, is_send, comm_vars, 1, reqs);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::EnsurePersistentFluxReqs(...)
//! \brief persistent flux-correction requests (comm_flux, tag 0 as the Isends used)
//!
//! One message per peer rank, so the single tag still matches unambiguously and the
//! persistent requests reproduce exactly the matching the per-step Isend/Irecv had.

void MeshBoundaryValues::EnsurePersistentFluxReqs(
    const std::vector<RankPackedVarMessage> &msgs, Real *base, bool is_send,
    std::vector<MPI_Request> *reqs) {
  EnsurePersistentReqs(msgs, base, is_send, comm_flux, 0, reqs);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::FreePersistentVarReqs(...)
//! \brief release persistent requests, refusing to free an active one

void MeshBoundaryValues::FreePersistentVarReqs(std::vector<MPI_Request> *reqs,
                                               bool started, bool drain_at_teardown) {
  if (reqs->empty()) return;
  if (started) {
    if (drain_at_teardown) {
      // Teardown: aborting here would be worse than waiting.
      MPI_Waitall(static_cast<int>(reqs->size()), reqs->data(), MPI_STATUSES_IGNORE);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "rank-packed metadata rebuilt with boundary requests still in flight"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  bool no_errors = true;
  for (auto &req : *reqs) {
    if (req == MPI_REQUEST_NULL) continue;
    if (MPI_Request_free(&req) != MPI_SUCCESS) no_errors = false;
    ++persistent_var_reqs_freed_;
  }
  reqs->clear();
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error freeing persistent rank-packed requests"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::FreeLayoutPersistentVarReqs()
//! \brief release the per-layout requests; call immediately before clearing the layouts

void MeshBoundaryValues::FreeLayoutPersistentVarReqs(bool drain_at_teardown) {
  for (std::size_t i=0; i<lat_var_layouts_.size(); ++i) {
    const bool send_live = send_var_reqs_started_ &&
                           (lat_send_reqs_layout_ == static_cast<int>(i));
    const bool recv_live = recv_var_reqs_started_ &&
                           (lat_recv_reqs_layout_ == static_cast<int>(i));
    FreePersistentVarReqs(&lat_var_layouts_[i].send_preqs, send_live,
                          drain_at_teardown);
    FreePersistentVarReqs(&lat_var_layouts_[i].recv_preqs, recv_live,
                          drain_at_teardown);
  }
  // The recorded indices address lat_var_layouts_, which the caller is about to empty.
  lat_send_reqs_layout_ = -1;
  lat_recv_reqs_layout_ = -1;
  recv_var_done_.clear();
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::FreeLayoutPersistentFluxReqs()
//! \brief release the per-layout flux requests; call before clearing the flux layouts
//!
//! Also the guard for a flux buffer reallocation: the requests encode base + offset, so
//! they must be released before either aggregate buffer moves.

void MeshBoundaryValues::FreeLayoutPersistentFluxReqs(bool drain_at_teardown) {
  for (std::size_t i=0; i<lat_flux_layouts_.size(); ++i) {
    const bool send_live = send_flux_reqs_started_ &&
                           (lat_flux_send_reqs_layout_ == static_cast<int>(i));
    const bool recv_live = recv_flux_reqs_started_ &&
                           (lat_flux_recv_reqs_layout_ == static_cast<int>(i));
    FreePersistentVarReqs(&lat_flux_layouts_[i].send_preqs, send_live,
                          drain_at_teardown);
    FreePersistentVarReqs(&lat_flux_layouts_[i].recv_preqs, recv_live,
                          drain_at_teardown);
  }
  // The recorded indices address lat_flux_layouts_, which the caller is about to empty.
  lat_flux_send_reqs_layout_ = -1;
  lat_flux_recv_reqs_layout_ = -1;
  send_flux_reqs_started_ = false;
  recv_flux_reqs_started_ = false;
  recv_flux_counts_verified_ = false;
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBoundaryValues::TestVarRecvComplete(bool *)
//! \brief one MPI_Testall over the started rank-packed variable receives
//!
//! See TestFluxRecvComplete below for why a single Testall replaces the per-request poll
//! loop.  Because Testall completes the whole set at once, the per-message recv_var_done_
//! flags are simply raised together here for ClearRecv.

bool MeshBoundaryValues::TestVarRecvComplete(bool *no_errors) {
  const int lay = lat_recv_reqs_layout_;
  auto &recv_reqs = VarReqs(false, lay);
  if (recv_reqs.empty()) return true;
  // Nothing to test if this phase never started these receives, or if an earlier call
  // already completed and size-checked them: either way they are INACTIVE, and Testall
  // would report them complete with an empty status whose count is 0.  recv_var_done_ is
  // now raised for the whole set at once, so its first entry marks the whole set.  This
  // reproduces what the old per-request loop did when every handle was already done.
  if (!recv_var_reqs_started_ ||
      ((recv_var_done_.size() == recv_reqs.size()) && (recv_var_done_[0] != 0))) {
    return true;
  }
  const int nreq = static_cast<int>(recv_reqs.size());
  if (static_cast<int>(mpi_statuses_.size()) < nreq) {
    mpi_statuses_.resize(nreq);
  }
  int test = 0;
  if (MPI_Testall(nreq, recv_reqs.data(), &test, mpi_statuses_.data()) != MPI_SUCCESS) {
    *no_errors = false;
    return false;
  }
  if (!static_cast<bool>(test)) return false;
  const auto &recv_msgs = (lay >= 0) ? lat_var_layouts_[lay].recv_msgs : recv_var_msgs_;
  for (int i=0; i<nreq; ++i) {
    int count = MPI_UNDEFINED;
    if (MPI_Get_count(&mpi_statuses_[i], MPI_ATHENA_REAL, &count) != MPI_SUCCESS ||
        count != recv_msgs[i].data_size) {
      *no_errors = false;
    }
  }
  // ClearRecv skips every request verified here: a completed persistent request is
  // INACTIVE, so waiting on it again would hand back an empty status with count 0.
  recv_var_done_.assign(recv_reqs.size(), 1);
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBoundaryValues::TestFluxRecvComplete(bool *, bool *)
//! \brief one MPI_Testall over the started rank-packed flux receives
//!
//! One Testall is a single progress pass over the whole peer set instead of one per peer,
//! and it is all-or-nothing: unless every receive has completed it reports flag=false and
//! modifies no request, so a persistent receive that has already arrived stays testable
//! and no per-request "already done" bookkeeping is needed.  The payload sizes are
//! verified exactly as the per-request MPI_Test did, once the statuses are valid.

bool MeshBoundaryValues::TestFluxRecvComplete(bool *no_errors, bool *remote_recv) {
  const int lay = lat_flux_recv_reqs_layout_;
  auto *recv_reqs = FluxReqs(false, lay);
  if ((recv_reqs == nullptr) || recv_reqs->empty()) return true;
  *remote_recv = true;
  // Nothing to test if this phase never started these receives, or if they were already
  // completed and size-checked: either way they are INACTIVE, and Testall would report
  // them complete with an empty status whose count is 0.  This reproduces what the old
  // per-request loop did when it found every handle already MPI_REQUEST_NULL.
  if (!recv_flux_reqs_started_ || recv_flux_counts_verified_) return true;
  const int nreq = static_cast<int>(recv_reqs->size());
  if (static_cast<int>(mpi_statuses_.size()) < nreq) {
    mpi_statuses_.resize(nreq);
  }
  int test = 0;
  if (MPI_Testall(nreq, recv_reqs->data(), &test, mpi_statuses_.data()) != MPI_SUCCESS) {
    *no_errors = false;
    return false;
  }
  if (!static_cast<bool>(test)) return false;
  const auto &recv_msgs = lat_flux_layouts_[lay].recv_msgs;
  for (int i=0; i<nreq; ++i) {
    int count = MPI_UNDEFINED;
    if (MPI_Get_count(&mpi_statuses_[i], MPI_ATHENA_REAL, &count) != MPI_SUCCESS ||
        count != recv_msgs[i].data_size) {
      *no_errors = false;
    }
  }
  recv_flux_counts_verified_ = true;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::FreeAllPersistentVarReqs(bool)
//! \brief release every persistent request this object owns

void MeshBoundaryValues::FreeAllPersistentVarReqs(bool drain_at_teardown) {
  // Sample before FreeLayoutPersistentVarReqs(), which resets both layout indices to -1
  // and would otherwise make every dense-path liveness test read as "started".
  const bool dense_send_live = send_var_reqs_started_ && (lat_send_reqs_layout_ < 0);
  const bool dense_recv_live = recv_var_reqs_started_ && (lat_recv_reqs_layout_ < 0);
  FreeLayoutPersistentVarReqs(drain_at_teardown);
  FreePersistentVarReqs(&send_var_preqs_, dense_send_live, drain_at_teardown);
  FreePersistentVarReqs(&recv_var_preqs_, dense_recv_live, drain_at_teardown);
  send_var_reqs_started_ = false;
  recv_var_reqs_started_ = false;
  recv_var_done_.clear();
}
#endif  // MPI_PARALLEL_ENABLED

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::MarkRecvUnpackPending()
//! \brief record that an unpack kernel is in flight over this object's receive storage

void MeshBoundaryValues::MarkRecvUnpackPending() {
#if MPI_PARALLEL_ENABLED
  recv_unpack_pending_ = true;
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same<DevExeSpace, Kokkos::Cuda>::value) {
    if (recv_unpack_event_ == nullptr) {
      if (cudaEventCreateWithFlags(&recv_unpack_event_, cudaEventDisableTiming) !=
          cudaSuccess) {
        recv_unpack_event_ = nullptr;
      }
    }
    if (recv_unpack_event_ != nullptr) {
      // Recorded on the same stream the unpack kernels were launched on, so the event
      // is reached exactly when they retire.  A failed create/record leaves the event
      // invalid and WaitRecvUnpackComplete falls back to the full-device fence.
      recv_unpack_event_valid_ =
          (cudaEventRecord(recv_unpack_event_, DevExeSpace().cuda_stream()) ==
           cudaSuccess);
    }
  }
#endif
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::WaitRecvUnpackComplete()
//! \brief establish that the unpack has finished reading the receive storage
//!
//! Waits on the unpack event when MarkRecvUnpackPending() recorded one, and on the whole
//! device otherwise, which is the pre-existing behaviour and the safe fallback.

void MeshBoundaryValues::WaitRecvUnpackComplete() {
#if MPI_PARALLEL_ENABLED
  if (!recv_unpack_pending_) return;
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same<DevExeSpace, Kokkos::Cuda>::value) {
    if (recv_unpack_event_valid_ && (recv_unpack_event_ != nullptr)) {
      const cudaError_t err = cudaEventSynchronize(recv_unpack_event_);
      recv_unpack_event_valid_ = false;
      recv_unpack_pending_ = false;
      if (err == cudaSuccess) return;
      // Fall through to the full fence: a failed wait must not be treated as a wait.
    }
  }
#endif
  DevExeSpace().fence();
  recv_unpack_pending_ = false;
#endif
}

#if MPI_PARALLEL_ENABLED
bool MeshBoundaryValues::UseRankPackedVars() const {
  return (global_variable::nranks > 1) &&
         (!(pmy_pack->lat_active_mask_enabled) || rank_packed_lat_capable_);
}

void MeshBoundaryValues::InvalidateRankPackedVarMetadata() {
  WaitRecvUnpackComplete();
  // The persistent requests encode offsets into metadata that is about to be replaced.
  FreeAllPersistentVarReqs(false);
  rank_packed_bvals_nvars_ = -1;
  lat_var_layout_index_ = -1;
  lat_var_layouts_.clear();
  lat_var_layout_cache_.clear();
  rank_packed_lat_flux_active_ = false;
  rank_packed_flux_nvars_ = -1;
  lat_flux_layout_index_ = -1;
  // Same reason as above: the persistent flux requests encode offsets into layouts that
  // are about to be discarded.
  FreeLayoutPersistentFluxReqs(false);
  lat_flux_layouts_.clear();
  lat_flux_layout_cache_.clear();
}

void MeshBoundaryValues::EnsureRankPackedVarMetadata(const int nvars) {
  const bool rebuild =
      (rank_packed_bvals_nvars_ != nvars) ||
      (rank_packed_bvals_nmb_ != pmy_pack->nmb_thispack) ||
      (rank_packed_bvals_nnghbr_ != pmy_pack->pmb->nnghbr) ||
      (rank_packed_bvals_topology_version_ != pmy_pack->pmesh->topology_version);
  if (!rebuild) return;

  // The full header exchange is symmetric and blocking.  A LAT refresh may invoke only a
  // sender or receiver rank, so attempting this exchange lazily under a LAT mask can
  // deadlock.  Initialization and post-AMR boundary setup build it with all blocks
  // active.
  if (pmy_pack->lat_active_mask_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Rank-packed boundary topology was not initialized before a LAT exchange"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  BuildRankPackedVarMetadata(nvars);
}
#endif

void MeshBoundaryValues::PrepareRankPackedVarMetadata(const int nvars) {
#if MPI_PARALLEL_ENABLED
  if (UseRankPackedVars()) {
    const bool rebuild =
        (rank_packed_bvals_nvars_ != nvars) ||
        (rank_packed_bvals_nmb_ != pmy_pack->nmb_thispack) ||
        (rank_packed_bvals_nnghbr_ != pmy_pack->pmb->nnghbr) ||
        (rank_packed_bvals_topology_version_ !=
         pmy_pack->pmesh->topology_version);
    if (rebuild) {
      // This entry point is used only by the lazily activated FFE sidecar.  A dormant
      // interval can skip InitRecv(), so fence any old asynchronous unpack before its
      // topology metadata is replaced.  Ordinary BoundaryValues objects retain the
      // exact latest-dev EnsureRankPackedVarMetadata() path above.
      InvalidateRankPackedVarMetadata();
    }
    EnsureRankPackedVarMetadata(nvars);
  }
#else
  (void)nvars;
#endif
}

#if MPI_PARALLEL_ENABLED

int MeshBoundaryValues::GetVarDataSize(const MeshBoundaryBuffer &buf, int m, int n,
                                       int nvars) const {
  int data_size = nvars;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  if (nghbr.h_view(m,n).lev < mblev.h_view(m)) {
    data_size *= buf.icoar_ndat;
  } else if (nghbr.h_view(m,n).lev == mblev.h_view(m)) {
    data_size *= is_z4c_ ? buf.isame_z4c_ndat : buf.isame_ndat;
  } else {
    data_size *= buf.ifine_ndat;
  }
  return data_size;
}

void MeshBoundaryValues::BuildRankPackedVarMetadata(const int nvars) {
  if (pmy_pack->lat_active_mask_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Cannot exchange full rank-packed metadata under an active LAT mask"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rank_packed_bvals_nvars_ = nvars;
  rank_packed_bvals_nmb_ = pmy_pack->nmb_thispack;
  rank_packed_bvals_nnghbr_ = pmy_pack->pmb->nnghbr;
  rank_packed_bvals_topology_version_ = pmy_pack->pmesh->topology_version;
  send_var_entries_.clear();
  recv_var_entries_.clear();
  send_var_msgs_.clear();
  recv_var_msgs_.clear();
  // Everything below is rebuilt from the new topology, and the aggregate buffers may be
  // reallocated further down, so every persistent request bound to the old offsets and
  // the old base pointers must be released first.
  FreeAllPersistentVarReqs(false);
  lat_var_layouts_.clear();
  lat_var_layout_cache_.clear();
  lat_var_layout_index_ = -1;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  int my_rank = global_variable::my_rank;

  std::map<int, std::vector<RankPackedVarEntry>> send_by_rank;
  std::map<int, std::vector<RankPackedVarEntry>> recv_by_rank;

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid < 0) continue;
      int peer_rank = nghbr.h_view(m,n).rank;
      if (peer_rank == my_rank) continue;

      RankPackedVarEntry send_entry;
      send_entry.m = m;
      send_entry.n = n;
      send_entry.lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[peer_rank];
      send_entry.dn = nghbr.h_view(m,n).dest;
      send_entry.data_size = GetVarDataSize(sendbuf[n], m, n, nvars);
      send_entry.offset = 0;
      send_by_rank[peer_rank].push_back(send_entry);

      RankPackedVarEntry recv_entry;
      recv_entry.m = m;
      recv_entry.n = n;
      recv_entry.lid = m;
      recv_entry.dn = n;
      recv_entry.data_size = GetVarDataSize(recvbuf[n], m, n, nvars);
      recv_entry.offset = 0;
      recv_by_rank[peer_rank].push_back(recv_entry);
    }
  }

  int send_total = 0;
  int send_hdr_total = 0;
  int send_entry_total = 0;
  for (auto &kv : send_by_rank) {
    const int msg_offset = send_total;
    const int hdr_offset = send_hdr_total;
    const int entry_offset = send_entry_total;
    for (auto &entry : kv.second) {
      entry.offset = send_total;
      send_var_entries_.push_back(entry);
      send_total += entry.data_size;
      ++send_entry_total;
    }
    send_hdr_total += 3*static_cast<int>(kv.second.size());
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = entry_offset;
    msg.hdr_offset = hdr_offset;
    msg.offset = msg_offset;
    msg.data_size = send_total - msg_offset;
    send_var_msgs_.push_back(msg);
  }

  int recv_total = 0;
  int recv_hdr_total = 0;
  int recv_entry_total = 0;
  for (auto &kv : recv_by_rank) {
    const int msg_offset = recv_total;
    const int hdr_offset = recv_hdr_total;
    const int entry_offset = recv_entry_total;
    for (auto &entry : kv.second) {
      entry.offset = recv_total;
      recv_var_entries_.push_back(entry);
      recv_total += entry.data_size;
      ++recv_entry_total;
    }
    recv_hdr_total += 3*static_cast<int>(kv.second.size());
    RankPackedVarMessage msg;
    msg.rank = kv.first;
    msg.nentries = static_cast<int>(kv.second.size());
    msg.entry_offset = entry_offset;
    msg.hdr_offset = hdr_offset;
    msg.offset = msg_offset;
    msg.data_size = recv_total - msg_offset;
    recv_var_msgs_.push_back(msg);
  }

  // The two aggregate payload buffers are the largest allocations in this class (~63 MiB
  // each here) and were rebuilt on every topology change, i.e. essentially every cycle at
  // refinement_interval=1, for a payload total that moves by a few entries.  Give them a
  // MiB-granular, grow-only capacity instead.  Note Kokkos::realloc zero-fills even when
  // the size is unchanged, so the old code also memset both buffers every cycle.
  //
  // Retaining the allocation across the rebuild is safe here:
  //   * every offset and message descriptor was just recomputed from the new topology
  //     above, so nothing reads this storage at an index derived from the old layout;
  //   * persistent MPI requests ARE bound to these buffers (EnsurePersistentVarReqs),
  //     but FreeAllPersistentVarReqs() at the top of this function released every one of
  //     them before any realloc below can move a base pointer, and they are re-bound
  //     lazily from the new .data() + msg.offset on the next exchange;
  //   * the only extent consumer is the capacity check in PrepareLATRankPackedVarLayout
  //     below, which capacity semantics satisfy; every payload access is bounded by a
  //     message offset plus a live data_size, never by view.extent().
  // Only the ALLOCATION becomes sticky: the metadata above still keys on
  // rank_packed_bvals_topology_version_ exactly as before.
  if (send_total > rank_sendbuf_vars_cap_) {
    rank_sendbuf_vars_cap_ = RankPackedBufCapacity(send_total);
    Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                    rank_sendbuf_vars_, rank_sendbuf_vars_cap_);
  }
  if (recv_total > rank_recvbuf_vars_cap_) {
    rank_recvbuf_vars_cap_ = RankPackedBufCapacity(recv_total);
    Kokkos::realloc(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                    rank_recvbuf_vars_, rank_recvbuf_vars_cap_);
  }
  // The header arrays are deliberately left exact: they are host-side ints (3 per
  // off-rank entry, tens of KiB), so they never touch the CUDA heap this policy exists to
  // protect, and rank_recvhdr_vars_ IS read through its extent at :659 below.
  Kokkos::realloc(rank_sendhdr_vars_, std::max(1, send_hdr_total));
  Kokkos::realloc(rank_recvhdr_vars_, std::max(1, recv_hdr_total));

  for (const auto &msg : send_var_msgs_) {
    for (int e=0; e<msg.nentries; ++e) {
      const auto &entry = send_var_entries_[msg.entry_offset + e];
      const int hidx = msg.hdr_offset + 3*e;
      rank_sendhdr_vars_(hidx    ) = entry.lid;
      rank_sendhdr_vars_(hidx + 1) = entry.dn;
      rank_sendhdr_vars_(hidx + 2) = entry.data_size;
    }
  }

  const int meta_tag = 2;
  std::vector<MPI_Request> exch_reqs(
      send_var_msgs_.size() + recv_var_msgs_.size(), MPI_REQUEST_NULL);
  std::size_t req_idx = 0;
  bool no_errors = true;
  for (const auto &msg : recv_var_msgs_) {
    const int hdr_size = 3*msg.nentries;
    int ierr = MPI_Irecv(rank_recvhdr_vars_.data() + msg.hdr_offset, hdr_size, MPI_INT,
                         msg.rank, meta_tag, comm_vars, &exch_reqs[req_idx++]);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  for (const auto &msg : send_var_msgs_) {
    const int hdr_size = 3*msg.nentries;
    int ierr = MPI_Isend(rank_sendhdr_vars_.data() + msg.hdr_offset, hdr_size, MPI_INT,
                         msg.rank, meta_tag, comm_vars, &exch_reqs[req_idx++]);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!exch_reqs.empty()) {
    int ierr = MPI_Waitall(static_cast<int>(exch_reqs.size()), exch_reqs.data(),
                           MPI_STATUSES_IGNORE);
    if (ierr != MPI_SUCCESS) no_errors = false;
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in rank-packed boundary metadata exchange"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int map_len = nmb*nnghbr;
  send_agg_offset_ = DvceArray1D<int>("send_agg_offset", std::max(1, map_len));
  recv_agg_offset_ = DvceArray1D<int>("recv_agg_offset", std::max(1, map_len));
  auto send_off_h = Kokkos::create_mirror_view(send_agg_offset_);
  auto recv_off_h = Kokkos::create_mirror_view(recv_agg_offset_);
  for (int i=0; i<map_len; ++i) {
    send_off_h(i) = -1;
    recv_off_h(i) = -1;
  }
  for (const auto &entry : send_var_entries_) {
    send_off_h(entry.m*nnghbr + entry.n) = entry.offset;
  }
  const int nmb_max = std::max(nmb, pmy_pack->pmesh->nmb_maxperrank);
  for (const auto &msg : recv_var_msgs_) {
    int off = msg.offset;
    for (int e=0; e<msg.nentries; ++e) {
      const int hidx = msg.hdr_offset + 3*e;
      const int lid = rank_recvhdr_vars_(hidx);
      const int dn = rank_recvhdr_vars_(hidx + 1);
      const int dsize = rank_recvhdr_vars_(hidx + 2);
      if ((lid < 0) || (lid >= nmb_max) || (dn < 0) || (dn >= nnghbr) || (dsize < 0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Invalid rank-packed recv header from peer "
                  << msg.rank << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (lid < nmb) {
        const auto &nb = nghbr.h_view(lid,dn);
        const int expected_size = GetVarDataSize(recvbuf[dn], lid, dn, nvars);
        if (nb.gid < 0 || nb.rank != msg.rank || dsize != expected_size) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Inconsistent rank-packed recv header from peer "
                    << msg.rank << std::endl;
          std::exit(EXIT_FAILURE);
        }
        recv_off_h(lid*nnghbr + dn) = off;
      }
      off += dsize;
    }
    if ((off - msg.offset) != msg.data_size) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Rank-packed recv payload size mismatch from peer "
                << msg.rank << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  Kokkos::deep_copy(send_agg_offset_, send_off_h);
  Kokkos::deep_copy(recv_agg_offset_, recv_off_h);

  if (show_rank_packed_bvals_stats_) {
    std::cout << "[rank " << global_variable::my_rank
              << "] persistent bvals requests: created="
              << persistent_var_reqs_created_ << " freed=" << persistent_var_reqs_freed_
              << " live=" << (persistent_var_reqs_created_ - persistent_var_reqs_freed_)
              << std::endl;
    std::cout << "[rank " << global_variable::my_rank << "] rank-packed bvals vars: "
              << "legacy_send_msgs=" << send_var_entries_.size()
              << " packed_send_msgs=" << send_var_msgs_.size()
              << " legacy_recv_msgs=" << recv_var_entries_.size()
              << " packed_recv_msgs=" << recv_var_msgs_.size()
              << std::endl;
  }
}

void MeshBoundaryValues::PrepareLATRankPackedVarLayout() {
  if (!(pmy_pack->lat_active_mask_enabled) || !rank_packed_lat_capable_) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid LAT rank-packed layout request" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nmb = pmy_pack->nmb_thispack;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const std::uint64_t topology_version = pmy_pack->pmesh->topology_version;
  const std::uint64_t lat_metadata_version =
      pmy_pack->pmesh->hydro_lat_metadata_version;
  auto &lat_send = pmy_pack->lat_send_nghbr;
  auto &lat_active = pmy_pack->lat_active_mb;

  if (!lat_var_layouts_.empty() &&
      (lat_var_layouts_.front().topology_version != topology_version ||
       lat_var_layouts_.front().lat_metadata_version != lat_metadata_version)) {
    WaitRecvUnpackComplete();
    FreeLayoutPersistentVarReqs();
    lat_var_layouts_.clear();
    lat_var_layout_cache_.clear();
    lat_var_layout_index_ = -1;
  }

  const bool cache_key_valid = pmy_pack->LATActiveMaskUsesCache();
  const std::uint64_t cache_generation = pmy_pack->LATCacheGeneration();
  if (cache_key_valid && lat_var_layout_cache_generation_ != cache_generation) {
    lat_var_layout_cache_.clear();
    lat_var_layout_cache_generation_ = cache_generation;
  }
  const LATLayoutCacheKey cache_key = {
      cache_generation,
      reinterpret_cast<std::uintptr_t>(lat_active.h_view.data()),
      reinterpret_cast<std::uintptr_t>(lat_send.h_view.data())};
  if (cache_key_valid) {
    const auto cached = lat_var_layout_cache_.find(cache_key);
    if (cached != lat_var_layout_cache_.end()) {
      lat_var_layout_index_ = cached->second;
      return;
    }
  }

  std::vector<int> send_entry_ids;
  send_entry_ids.reserve(send_var_entries_.size());
  for (std::size_t i=0; i<send_var_entries_.size(); ++i) {
    const auto &entry = send_var_entries_[i];
    if (lat_send.h_view(entry.m, entry.n) != 0) {
      send_entry_ids.push_back(static_cast<int>(i));
    }
  }

  // rank_recvhdr_vars_ is the sender's immutable ordering.  Filtering it by the local
  // target mask therefore produces exactly the compact order used by that sender.
  std::vector<int> recv_header_ids;
  recv_header_ids.reserve(recv_var_entries_.size());
  for (const auto &msg : recv_var_msgs_) {
    for (int e=0; e<msg.nentries; ++e) {
      const int hidx = msg.hdr_offset + 3*e;
      const int lid = rank_recvhdr_vars_(hidx);
      if (lid >= 0 && lid < nmb && lat_active.h_view(lid) != 0) {
        recv_header_ids.push_back(hidx);
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
  hash_id(static_cast<std::uint64_t>(recv_header_ids.size()));
  for (const int id : recv_header_ids) hash_id(static_cast<std::uint64_t>(id) + 1);

  for (std::size_t i=0; i<lat_var_layouts_.size(); ++i) {
    const auto &layout = lat_var_layouts_[i];
    if (layout.topology_version == topology_version &&
        layout.lat_metadata_version == lat_metadata_version &&
        layout.selection_hash == selection_hash &&
        layout.send_entry_ids == send_entry_ids &&
        layout.recv_header_ids == recv_header_ids) {
      lat_var_layout_index_ = static_cast<int>(i);
      if (cache_key_valid) {
        lat_var_layout_cache_[cache_key] = lat_var_layout_index_;
      }
      return;
    }
  }

  RankPackedVarLayout layout;
  layout.topology_version = topology_version;
  layout.lat_metadata_version = lat_metadata_version;
  layout.selection_hash = selection_hash;
  layout.send_entry_ids = std::move(send_entry_ids);
  layout.recv_header_ids = std::move(recv_header_ids);

  // Two nmb*nnghbr int maps per layout, and a few layouts are built per cycle.  Take them
  // as two rows of a persistent pool instead of allocating a fresh pair each time.  The
  // row index is fixed by this layout's position in lat_var_layouts_, which only ever
  // grows by push_back or is cleared wholesale, so rows are recycled but never aliased.
  // Invalidation is unchanged: the clears above still key on topology_version and
  // lat_metadata_version, and only the storage they used to free is now kept.
  const int map_len = std::max(1, nmb*nnghbr);
  layout.offsets_row = 2*static_cast<int>(lat_var_layouts_.size());
  layout.send_offsets = lat_var_offset_pool_.Acquire("lat_send_agg_offset",
                                                     layout.offsets_row, map_len);
  layout.recv_offsets = lat_var_offset_pool_.Acquire("lat_recv_agg_offset",
                                                     layout.offsets_row + 1, map_len);
  auto send_off_h = Kokkos::create_mirror_view(layout.send_offsets);
  auto recv_off_h = Kokkos::create_mirror_view(layout.recv_offsets);
  for (int i=0; i<map_len; ++i) {
    send_off_h(i) = -1;
    recv_off_h(i) = -1;
  }

  std::vector<unsigned char> send_selected(send_var_entries_.size(), 0);
  for (const int id : layout.send_entry_ids) send_selected[id] = 1;
  int send_total = 0;
  int selected_send_total = 0;
  for (const auto &dense_msg : send_var_msgs_) {
    RankPackedVarMessage msg;
    msg.rank = dense_msg.rank;
    msg.nentries = 0;
    msg.entry_offset = selected_send_total;
    msg.hdr_offset = -1;
    msg.offset = send_total;
    for (int e=0; e<dense_msg.nentries; ++e) {
      const int id = dense_msg.entry_offset + e;
      if (send_selected[id] == 0) continue;
      const auto &entry = send_var_entries_[id];
      send_off_h(entry.m*nnghbr + entry.n) = send_total;
      send_total += entry.data_size;
      ++msg.nentries;
      ++selected_send_total;
    }
    msg.data_size = send_total - msg.offset;
    if (msg.nentries > 0) layout.send_msgs.push_back(msg);
  }

  std::vector<unsigned char> recv_selected(rank_recvhdr_vars_.extent_int(0)/3, 0);
  for (const int hidx : layout.recv_header_ids) recv_selected[hidx/3] = 1;
  int recv_total = 0;
  int selected_recv_total = 0;
  for (const auto &dense_msg : recv_var_msgs_) {
    RankPackedVarMessage msg;
    msg.rank = dense_msg.rank;
    msg.nentries = 0;
    msg.entry_offset = selected_recv_total;
    msg.hdr_offset = -1;
    msg.offset = recv_total;
    for (int e=0; e<dense_msg.nentries; ++e) {
      const int hidx = dense_msg.hdr_offset + 3*e;
      if (recv_selected[hidx/3] == 0) continue;
      const int lid = rank_recvhdr_vars_(hidx);
      const int dn = rank_recvhdr_vars_(hidx + 1);
      const int dsize = rank_recvhdr_vars_(hidx + 2);
      recv_off_h(lid*nnghbr + dn) = recv_total;
      recv_total += dsize;
      ++msg.nentries;
      ++selected_recv_total;
    }
    msg.data_size = recv_total - msg.offset;
    if (msg.nentries > 0) layout.recv_msgs.push_back(msg);
  }

  // Compared against the live extent, which is the allocated CAPACITY (>= the dense total
  // this LAT layout is a subset of), so the check is unchanged by the grow-only policy.
  if (send_total > rank_sendbuf_vars_.extent_int(0) ||
      recv_total > rank_recvbuf_vars_.extent_int(0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "LAT rank-packed layout exceeds static buffer capacity"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Kokkos::deep_copy(layout.send_offsets, send_off_h);
  Kokkos::deep_copy(layout.recv_offsets, recv_off_h);
  lat_var_layouts_.push_back(std::move(layout));
  lat_var_layout_index_ = static_cast<int>(lat_var_layouts_.size()) - 1;
  if (cache_key_valid) {
    lat_var_layout_cache_[cache_key] = lat_var_layout_index_;
  }
}
#endif

//----------------------------------------------------------------------------------------
// MeshBoundaryValues destructor

MeshBoundaryValues::~MeshBoundaryValues() {
  // Device metadata contains non-owning pointers into the buffers destroyed below.
  DevExeSpace().fence();
#if MPI_PARALLEL_ENABLED
#if defined(KOKKOS_ENABLE_CUDA)
  if (recv_unpack_event_ != nullptr) {
    // The fence above has already retired anything the event could be waiting on.
    (void)cudaEventDestroy(recv_unpack_event_);
    recv_unpack_event_ = nullptr;
    recv_unpack_event_valid_ = false;
  }
#endif
  recv_unpack_pending_ = false;
  FreeAllPersistentVarReqs(true);
  FreeLayoutPersistentFluxReqs(true);
  if (persistent_var_reqs_created_ != persistent_var_reqs_freed_) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "persistent rank-packed request leak: created "
              << persistent_var_reqs_created_ << " freed " << persistent_var_reqs_freed_
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] sendbuf[n].flux_req;
    delete [] recvbuf[n].vars_req;
    delete [] recvbuf[n].flux_req;
  }
  if (comm_vars != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_vars);
  }
  if (comm_flux != MPI_COMM_NULL) {
    MPI_Comm_free(&comm_flux);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::RefreshDeviceBufferMetadata()
//! \brief copy immutable buffer bounds and non-owning data pointers to compact device views

void MeshBoundaryValues::RefreshDeviceBufferMetadata() {
  const int nnghbr = pmy_pack->pmb->nnghbr;
  DevExeSpace().fence();
  if (sendbuf_device.extent_int(0) != nnghbr) {
    sendbuf_device = DvceArray1D<MeshBoundaryBufferDevice>("sendbuf_device", nnghbr);
  }
  if (recvbuf_device.extent_int(0) != nnghbr) {
    recvbuf_device = DvceArray1D<MeshBoundaryBufferDevice>("recvbuf_device", nnghbr);
  }

  auto sendbuf_host = Kokkos::create_mirror_view(sendbuf_device);
  auto recvbuf_host = Kokkos::create_mirror_view(recvbuf_device);
  for (int n=0; n<nnghbr; ++n) {
    CopyBoundaryBuffer(sendbuf[n], sendbuf_host(n));
    CopyBoundaryBuffer(recvbuf[n], recvbuf_host(n));
  }
  Kokkos::deep_copy(sendbuf_device, sendbuf_host);
  Kokkos::deep_copy(recvbuf_device, recvbuf_host);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::InitializeBuffers
//! \brief initialize each element of send/recv MeshBoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in MeshBoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void MeshBoundaryValues::InitializeBuffers(const int nvar, const int nfluxvar) {
  const int nflx = (nfluxvar >= 0) ? nfluxvar : nvar;
  // allocate memory for inflow BCs (but only if domain not strictly periodic)
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    Kokkos::realloc(u_in, nvar, 6);
    Kokkos::realloc(b_in, 3, 6);   // always 3 components of face-fields
    Kokkos::realloc(i_in, nvar, 6);
  }

  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // initialize buffers used for uniform grid and SMR/AMR calculations

  // Send-side vars are written only by the legacy pack path, which is bypassed
  // whenever rank-packed vars are active (nranks>1 for opted-in owners), and which
  // writes same-rank payloads directly into the destination recvbuf at nranks==1.
  // Stub them for opted-in owners on multi-rank runs; see stub_send_vars in bvals.hpp.
#if MPI_PARALLEL_ENABLED
  const bool stub_send = stub_send_vars && (global_variable::nranks > 1);
  const bool no_same_recv = stub_recv_same_vars && (global_variable::nranks > 1);
#else
  const bool stub_send = false;
  const bool no_same_recv = false;
#endif

  // Seed the buffers from the LIVE block count and let ResizeBuffers grow them from
  // there, which is the high-water discipline the physics field arrays already follow
  // (see the comment at hydro.cpp:53-55).  Seeding from nmb_maxperrank instead reserves
  // the AMR hard cap on every rank from cycle 0 and never releases it: on an adaptive
  // mesh whose ranks never approach that cap, that is several GiB of device memory per
  // rank held for nothing.  The MPI_Request arrays are deliberately left at the cap --
  // they are host-side and tiny, and keeping them fixed means ResizeBuffers never has to
  // touch a request array that might still be referenced.
  buf_nvar_ = nvar;
  buf_nflx_ = nflx;
  int nmb = BufferBlockCapacity(pmy_pack->nmb_thispack);
  nmb_alloc_ = nmb;
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = NeighborIndex(n,0,0,fy,fz);
        InitSendIndices(sendbuf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recvbuf[indx],n, 0, 0, fy, fz);
        sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
        recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
        indx++;
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,0,fx,fz);
          InitSendIndices(sendbuf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recvbuf[indx],0, m, 0, fx, fz);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = NeighborIndex(n,m,0,fz,0);
          InitSendIndices(sendbuf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recvbuf[indx],n, m, 0, fz, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
          indx++;
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {
    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,0,l,fx,fy);
          InitSendIndices(sendbuf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recvbuf[indx],0, 0, l, fx, fy);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = NeighborIndex(n,0,l,fy,0);
          InitSendIndices(sendbuf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recvbuf[indx],n, 0, l, fy, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,l,fx,0);
          InitSendIndices(sendbuf[indx],0, m, l, fx, 0);
          InitRecvIndices(recvbuf[indx],0, m, l, fx, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = NeighborIndex(n,m,l,0,0);
          InitSendIndices(sendbuf[indx],n, m, l, 0, 0);
          InitRecvIndices(recvbuf[indx],n, m, l, 0, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, stub_send,
                                     stub_flux_same, false, same_level_vars_only);
          recvbuf[indx].AllocateBuffers(nmb, nvar, nflx, is_z4c_, false, false,
                                        no_same_recv, same_level_vars_only);
        }
      }
    }
  }

  RefreshDeviceBufferMetadata();

#if MPI_PARALLEL_ENABLED
  InvalidateRankPackedVarMetadata();
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn int MeshBoundaryValues::BufferBlockCapacity
//! \brief Block capacity to allocate for a required block count.

int MeshBoundaryValues::BufferBlockCapacity(const int nmb_needed) const {
  // Round up so that an AMR ramp with refinement_interval=1 does not reallocate the
  // (large) buffers on every single growth step, but never reserve past the hard
  // per-rank cap, which is what the MPI_Request arrays are sized for.
  const int cap = pmy_pack->pmesh->nmb_maxperrank;
  int nmb = ((std::max(nmb_needed, 1) + 31)/32)*32;
  if ((cap > 0) && (nmb > cap)) nmb = std::max(nmb_needed, cap);
  return nmb;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::AllocateAllBuffers
//! \brief (Re)allocate every per-neighbor payload at a capacity of `nmb` MeshBlocks.
//!
//! The owner-level footprint guarantees (stub_send_vars, stub_recv_same_vars,
//! stub_flux_same, same_level_vars_only) are resolved here, in one place, so that a
//! caller reallocating the buffers cannot silently drop one of them.  It allocates
//! only; the caller owns nmb_alloc_ and the metadata invalidation that must follow.

void MeshBoundaryValues::AllocateAllBuffers(const int nmb) {
#if MPI_PARALLEL_ENABLED
  const bool stub_send = stub_send_vars && (global_variable::nranks > 1);
  const bool no_same_recv = stub_recv_same_vars && (global_variable::nranks > 1);
#else
  const bool stub_send = false;
  const bool no_same_recv = false;
#endif

  const int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    sendbuf[n].AllocateBuffers(nmb, buf_nvar_, buf_nflx_, is_z4c_, stub_send,
                               stub_flux_same, false, same_level_vars_only);
    recvbuf[n].AllocateBuffers(nmb, buf_nvar_, buf_nflx_, is_z4c_, false, false,
                               no_same_recv, same_level_vars_only);
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBoundaryValues::ResizeBuffers
//! \brief Grow the per-neighbor send/recv buffers to hold nmb_needed MeshBlocks.
//!
//! Grow-only: a rank that sheds MeshBlocks keeps its buffers, exactly like the field
//! arrays.  Returns true if an allocation actually changed.

bool MeshBoundaryValues::ResizeBuffers(const int nmb_needed) {
  if (nmb_needed <= nmb_alloc_) return false;
  const int nmb = BufferBlockCapacity(nmb_needed);
  if (nmb <= nmb_alloc_) return false;

  // The buffer index metadata (isame/icoar/ifine/iflux_*) depends only on the mesh
  // geometry, not on the block count, so it is deliberately NOT recomputed here; only
  // the payload extents change.
  AllocateAllBuffers(nmb);
  nmb_alloc_ = nmb;

  // Every buffer base pointer just moved, so the device-side mirror must be rebuilt, and
  // any cached rank-packed layout was built against the old capacity.
  RefreshDeviceBufferMetadata();
#if MPI_PARALLEL_ENABLED
  InvalidateRankPackedVarMetadata();
#endif
  return true;
}

//----------------------------------------------------------------------------------------
// ParticlesBoundaryValues constructor:

particles::ParticlesBoundaryValues::ParticlesBoundaryValues(
  particles::Particles *pp, ParameterInput *pin) :
    sendlist("sendlist",1),
#if MPI_PARALLEL_ENABLED
    prtcl_rsendbuf("rsend",1),
    prtcl_rrecvbuf("rrecv",1),
    prtcl_isendbuf("isend",1),
    prtcl_irecvbuf("irecv",1),
#endif
    pmy_part(pp) {
#if MPI_PARALLEL_ENABLED
  //resize vectors over number of ranks
  nsends_eachrank.resize(global_variable::nranks);

  // create unique communicator for particles
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_part);
#endif
}

//----------------------------------------------------------------------------------------
// destructor

particles::ParticlesBoundaryValues::~ParticlesBoundaryValues() {
#if MPI_PARALLEL_ENABLED
  if (mpi_comm_part != MPI_COMM_NULL) {
    MPI_Comm_free(&mpi_comm_part);
  }
#endif
}
