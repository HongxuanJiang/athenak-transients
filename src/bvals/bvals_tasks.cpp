//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! Mesh variables. These are generic functions that work for both CC and FC variables.
//!
//! Note: InitFluxRecv() functions for flux correction step are specific to CC/FC vars,
//! and are implemented in flux_correct_XX.cpp files respectively. The ClearFluxRecv()
//! and ClearFluxSend() functions are generic and implemented below.
//!
//! Note2: task list functions for particle communication are all implemented in
//! bvals_part.cpp file.

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MeshBoundaryValues::InitRecv(const int nvars) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  // MPI writes directly into device receive storage.  A previous asynchronous
  // unpack must finish reading that storage before it is posted again.
  WaitRecvUnpackComplete();
  rank_packed_bvals_active_ = UseRankPackedVars();
  rank_packed_lat_active_ = rank_packed_bvals_active_ &&
                            pmy_pack->lat_active_mask_enabled &&
                            rank_packed_lat_capable_;
  if (rank_packed_bvals_active_) {
    EnsureRankPackedVarMetadata(nvars);
    if (rank_packed_lat_active_) PrepareLATRankPackedVarLayout();

    // Bind persistent receives to this layout's (peer, offset, count) tuples once, then
    // just restart them every step.  MPI_Irecv used to re-register the GPU buffer on
    // every call because UCX_RCACHE_ENABLE=n deliberately disables the registration
    // cache; MPI_Recv_init registers once and MPI_Startall reuses that registration.
    const int lay = rank_packed_lat_active_ ? lat_var_layout_index_ : -1;
    const auto &recv_msgs = (lay >= 0) ?
        lat_var_layouts_[lay].recv_msgs : recv_var_msgs_;
    auto &recv_reqs = VarReqs(false, lay);
    EnsurePersistentVarReqs(recv_msgs, rank_recvbuf_vars_.data(), false, &recv_reqs);
    if (!recv_reqs.empty()) {
      if (MPI_Startall(static_cast<int>(recv_reqs.size()), recv_reqs.data()) !=
          MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error starting rank-packed receives" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    recv_var_done_.assign(recv_reqs.size(), 0);
    lat_recv_reqs_layout_ = lay;
    recv_var_reqs_started_ = true;
    return TaskStatus::complete;
  }

  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &lat_active_indices = pmy_pack->lat_active_indices;

  // Initialize communications of variables
  bool no_errors=true;
  const int nwork_recv = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  for (int a=0; a<nwork_recv; ++a) {
    const int m = lat_enabled ? lat_active_indices.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= recvbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            if (is_z4c_) {
              data_size *= recvbuf[n].isame_z4c_ndat;
            } else {
              data_size *= recvbuf[n].isame_ndat;
            }
          } else {
            data_size *= recvbuf[n].ifine_ndat;
          }
          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(recvbuf[n].vars_req[m]));
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
//! \fn  void MeshBoundaryValues::ClearRecv
//! \brief Waits for all MPI receives associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  if (rank_packed_bvals_active_) {
    // Wait on the requests that were actually STARTED, which is the layout InitRecv
    // selected -- not whichever layout the mask selects by the time we get here.
    const int lay = lat_recv_reqs_layout_;
    auto &recv_reqs = VarReqs(false, lay);
    const auto &recv_msgs = (lay >= 0) ?
        lat_var_layouts_[lay].recv_msgs : recv_var_msgs_;
    if (recv_var_done_.size() != recv_reqs.size()) {
      recv_var_done_.assign(recv_reqs.size(), 0);
    }
    bool no_errors=true;
    for (std::size_t i=0; i<recv_reqs.size(); ++i) {
      // Already completed by RecvAndUnpack's MPI_Testall: the persistent request is now
      // INACTIVE, and waiting again would hand back an empty status with count 0.
      if (recv_var_done_[i]) continue;
      if (recv_reqs[i] == MPI_REQUEST_NULL) continue;
      MPI_Status status;
      int ierr = MPI_Wait(&recv_reqs[i], &status);
      int count = MPI_UNDEFINED;
      if (ierr == MPI_SUCCESS) ierr = MPI_Get_count(&status, MPI_ATHENA_REAL, &count);
      if (ierr != MPI_SUCCESS || count != recv_msgs[i].data_size) no_errors=false;
      recv_var_done_[i] = 1;
    }
    recv_var_reqs_started_ = false;
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "MPI error or size mismatch clearing rank-packed receives"
         << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }

  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &lat_active_indices = pmy_pack->lat_active_indices;
  // wait for all non-blocking receives for vars to finish before continuing
  const int nwork_recv = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  for (int a=0; a<nwork_recv; ++a) {
    const int m = lat_enabled ? lat_active_indices.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (recvbuf[n].vars_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearSend
//! \brief Waits for all MPI sends associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearSend() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  if (rank_packed_bvals_active_) {
    // As in ClearRecv, drain the layout whose sends were started.  MPI_Wait on an
    // inactive persistent request is legal and returns immediately, so a rank whose
    // PackAndSend never ran this step costs nothing here.
    auto &send_reqs = VarReqs(true, lat_send_reqs_layout_);
    bool no_errors=true;
    for (std::size_t i=0; i<send_reqs.size(); ++i) {
      if (send_reqs[i] == MPI_REQUEST_NULL) continue;
      int ierr = MPI_Wait(&send_reqs[i], MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) no_errors=false;
    }
    send_var_reqs_started_ = false;
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "MPI error in clearing rank-packed sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }

  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &lat_send = pmy_pack->lat_send_nghbr;
  auto &lat_boundary_send_indices = pmy_pack->lat_boundary_send_indices;
  // wait for all non-blocking sends for vars to finish before continuing
  const int nwork_send = lat_enabled ? pmy_pack->lat_nboundary_send_thispack : nmb;
  for (int a=0; a<nwork_send; ++a) {
    const int m = lat_enabled ? lat_boundary_send_indices.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if (lat_enabled && lat_send.h_view(m,n) == 0) continue;
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (sendbuf[n].vars_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearFluxRecv
//! \brief Waits for all MPI receives associated with communcation of boundary fluxes
//! to complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearFluxRecv() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  if (rank_packed_lat_flux_active_) {
    // Nothing is left to drain once RecvAndUnpackFlux's MPI_Testall succeeded: every
    // persistent receive is then INACTIVE, and waiting on one of those returns an empty
    // status whose count is 0.  A phase that never reached that test (or never started
    // the receives) still takes the wait-and-check path below.
    const int lay = lat_flux_recv_reqs_layout_;
    auto *recv_reqs = FluxReqs(false, lay);
    if (recv_flux_reqs_started_ && !recv_flux_counts_verified_ &&
        (recv_reqs != nullptr)) {
      const auto &recv_msgs = lat_flux_layouts_[lay].recv_msgs;
      for (std::size_t i=0; i<recv_reqs->size(); ++i) {
        MPI_Status status;
        int ierr = MPI_Wait(&(*recv_reqs)[i], &status);
        int count = MPI_UNDEFINED;
        if (ierr == MPI_SUCCESS) ierr = MPI_Get_count(&status, MPI_ATHENA_REAL, &count);
        if (ierr != MPI_SUCCESS || count != recv_msgs[i].data_size) no_errors = false;
      }
    }
    recv_flux_reqs_started_ = false;
    recv_flux_counts_verified_ = false;
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error or size mismatch clearing rank-packed flux "
                << "receives" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &lat_flux_recv_indices = pmy_pack->lat_flux_recv_indices;
  // wait for all non-blocking receives for fluxes to finish before continuing
  const int nwork_recv = lat_enabled ? pmy_pack->lat_nflux_recv_thispack : nmb;
  for (int a=0; a<nwork_recv; ++a) {
    const int m = lat_enabled ? lat_flux_recv_indices.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (recvbuf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(recvbuf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error clearing flux receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::ClearFluxSend
//! \brief Waits for all MPI sends associated with communcation of boundary fluxes to
//!  complete before allowing execution to continue

TaskStatus MeshBoundaryValues::ClearFluxSend() {
  bool no_errors=true;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  if (rank_packed_lat_flux_active_) {
    // One MPI_Waitall over the peer set the sends were actually started under.  A
    // persistent send that has completed simply goes inactive, never MPI_REQUEST_NULL.
    auto *send_reqs = FluxReqs(true, lat_flux_send_reqs_layout_);
    if (send_flux_reqs_started_ && (send_reqs != nullptr) && !send_reqs->empty()) {
      no_errors = (MPI_Waitall(static_cast<int>(send_reqs->size()), send_reqs->data(),
                               MPI_STATUSES_IGNORE) == MPI_SUCCESS);
    }
    send_flux_reqs_started_ = false;
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error clearing rank-packed flux sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto &lat_active_indices = pmy_pack->lat_active_indices;
  // wait for all non-blocking sends for fluxes to finish before continuing
  const int nwork_send = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  for (int a=0; a<nwork_send; ++a) {
    const int m = lat_enabled ? lat_active_indices.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if ( (nghbr.h_view(m,n).gid >= 0) &&
           (nghbr.h_view(m,n).rank != global_variable::my_rank) &&
           (sendbuf[n].flux_req[m] != MPI_REQUEST_NULL) ) {
        int ierr = MPI_Wait(&(sendbuf[n].flux_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
#endif
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error clearing flux sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}
