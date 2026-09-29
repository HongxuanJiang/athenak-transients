//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! Mesh variables.
//! Prolongation of FC variables  occurs in ProlongateFC() function called from task list

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <utility>
#include <iomanip>    // std::setprecision()

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
bool IsActiveFCFace(const int v, const int k, const int j, const int i,
                    const RegionIndcs &indcs) {
  if (v == 0) {
    return (i >= indcs.is) && (i <= indcs.ie + 1) &&
           (j >= indcs.js) && (j <= indcs.je) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  } else if (v == 1) {
    return (i >= indcs.is) && (i <= indcs.ie) &&
           (j >= indcs.js) && (j <= indcs.je + 1) &&
           (k >= indcs.ks) && (k <= indcs.ke);
  }
  return (i >= indcs.is) && (i <= indcs.ie) &&
         (j >= indcs.js) && (j <= indcs.je) &&
         (k >= indcs.ks) && (k <= indcs.ke + 1);
}

}  // namespace

//----------------------------------------------------------------------------------------
// BValFC constructor:

MeshBoundaryValuesFC::MeshBoundaryValuesFC(MeshBlockPack *pp, ParameterInput *pin) :
  MeshBoundaryValues(pp, pin, false) {
}

//----------------------------------------------------------------------------------------
//! \!fn void MeshBoundaryValuesFC::PackAndSendFC()
//! \brief Pack face-centered Mesh variables into boundary buffers and send to neighbors.
//!
//! As for cell-centered data, this routine packs ALL the buffers on ALL the faces, edges,
//! and corners simultaneously for all three components of face-fields on ALL the
//! MeshBlocks.
//!
//! Input array must be DvceFaceFld4D dimensioned (nmb, nx3, nx2, nx1)
//! DvceFaceFld4D of coarsened (restricted) fields also required with SMR/AMR

TaskStatus MeshBoundaryValuesFC::PackAndSendFC(DvceFaceFld4D<Real> &b,
                                               DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  DevExeSpace pack_exec;

#if MPI_PARALLEL_ENABLED
  // InitRecv normally selects the mode, but LAT can have sender-only ranks that do not
  // post receives. Select it again here so ClearSend waits on the requests posted below.
  rank_packed_bvals_active_ = UseRankPackedVars();
  if (rank_packed_bvals_active_) {
    const bool rebuild =
        (rank_packed_bvals_nvars_ != 3) ||
        (rank_packed_bvals_nmb_ != nmb) ||
        (rank_packed_bvals_nnghbr_ != nnghbr) ||
        (rank_packed_bvals_topology_version_ != pmy_pack->pmesh->topology_version);
    if (rebuild) BuildRankPackedVarMetadata(3);

    int my_rank = global_variable::my_rank;
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mbgid = pmy_pack->pmb->mb_gid;
    auto &mblev = pmy_pack->pmb->mb_lev;
    auto &sbuf = sendbuf;
    auto &rbuf = recvbuf;
    auto aggsbuf = rank_sendbuf_vars_;
    auto sendoff = send_agg_offset_;

    int nmnv = 3*nmb;
    Kokkos::TeamPolicy<> policy(pack_exec, nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("SendBuffRankPackedFC", athenak_lw(policy),
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank()/3;
      const int v = tmember.league_rank()%3;

      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.d_view(m,n).gid >= 0) {
          int il, iu, jl, ju, kl, ku, ndat;
          if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
            il = sbuf[n].icoar[v].bis; iu = sbuf[n].icoar[v].bie;
            jl = sbuf[n].icoar[v].bjs; ju = sbuf[n].icoar[v].bje;
            kl = sbuf[n].icoar[v].bks; ku = sbuf[n].icoar[v].bke;
            ndat = sbuf[n].icoar_ndat;
          } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            il = sbuf[n].isame[v].bis; iu = sbuf[n].isame[v].bie;
            jl = sbuf[n].isame[v].bjs; ju = sbuf[n].isame[v].bje;
            kl = sbuf[n].isame[v].bks; ku = sbuf[n].isame[v].bke;
            ndat = sbuf[n].isame_ndat;
          } else {
            il = sbuf[n].ifine[v].bis; iu = sbuf[n].ifine[v].bie;
            jl = sbuf[n].ifine[v].bjs; ju = sbuf[n].ifine[v].bje;
            kl = sbuf[n].ifine[v].bks; ku = sbuf[n].ifine[v].bke;
            ndat = sbuf[n].ifine_ndat;
          }
          const int ni = iu - il + 1;
          const int nj = ju - jl + 1;
          const int nk = ku - kl + 1;
          const int nkji = nk*nj*ni;
          const int nji = nj*ni;
          const int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
          const int dn = nghbr.d_view(m,n).dest;
          const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);
          const int base = same_rank ? 0 : sendoff(m*nnghbr + n);
          const bool coarser = (nghbr.d_view(m,n).lev < mblev.d_view(m));

          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = idx/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            Real val;
            if (v == 0) {
              val = coarser ? cb.x1f(m,k,j,i) : b.x1f(m,k,j,i);
            } else if (v == 1) {
              val = coarser ? cb.x2f(m,k,j,i) : b.x2f(m,k,j,i);
            } else {
              val = coarser ? cb.x3f(m,k,j,i) : b.x3f(m,k,j,i);
            }
            const int bi = ndat*v + i-il + ni*(j-jl + nj*(k-kl));
            if (same_rank) {
              rbuf[dn].vars(dm, bi) = val;
            } else {
              aggsbuf(base + bi) = val;
            }
          });
        }
        tmember.team_barrier();
      }
    });

    if (global_variable::nranks == 1) return TaskStatus::complete;
    pack_exec.fence();
    // This arm is reached only on the dense (non-LAT) path, so layout index -1.
    bool no_errors = true;
    auto &send_reqs = VarReqs(true, -1);
    EnsurePersistentVarReqs(send_var_msgs_, rank_sendbuf_vars_.data(), true,
                            &send_reqs);
    if (!send_reqs.empty()) {
      no_errors = (MPI_Startall(static_cast<int>(send_reqs.size()),
                                send_reqs.data()) == MPI_SUCCESS);
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in starting rank-packed sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lat_send_reqs_layout_ = -1;
    send_var_reqs_started_ = true;
    return TaskStatus::complete;
  }
#endif

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  const int nwork_send = nmb;
  if (nwork_send <= 0) return TaskStatus::complete;
  int nmnv = 3*nwork_send;
  Kokkos::TeamPolicy<> policy(pack_exec, nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int im = tmember.league_rank()/3;
    const int m = im;
    const int v = tmember.league_rank()%3;

    // scalar loop over neighbors to prevent race condition in overlapping assignments
    for (int n=0; n<nnghbr; ++n) {
      // only load buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        // if neighbor is at coarser level, use cindices to pack buffer
        // Note indices can be different for each component of face-centered field.
        int il, iu, jl, ju, kl, ku, ndat;
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          il = sbuf[n].icoar[v].bis;
          iu = sbuf[n].icoar[v].bie;
          jl = sbuf[n].icoar[v].bjs;
          ju = sbuf[n].icoar[v].bje;
          kl = sbuf[n].icoar[v].bks;
          ku = sbuf[n].icoar[v].bke;
          ndat = sbuf[n].icoar_ndat;
        // if neighbor is at same level, use sindices to pack buffer
        } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = sbuf[n].isame[v].bis;
          iu = sbuf[n].isame[v].bie;
          jl = sbuf[n].isame[v].bjs;
          ju = sbuf[n].isame[v].bje;
          kl = sbuf[n].isame[v].bks;
          ku = sbuf[n].isame[v].bke;
          ndat = sbuf[n].isame_ndat;
        // if neighbor is at finer level, use findices to pack buffer
        } else {
          il = sbuf[n].ifine[v].bis;
          iu = sbuf[n].ifine[v].bie;
          jl = sbuf[n].ifine[v].bjs;
          ju = sbuf[n].ifine[v].bje;
          kl = sbuf[n].ifine[v].bks;
          ku = sbuf[n].ifine[v].bke;
          ndat = sbuf[n].ifine_ndat;
        }
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // indices of recv'ing MB and buffer: assumes MB IDs are stored sequentially
        int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
        int dn = nghbr.d_view(m,n).dest;

        const bool coarser = (nghbr.d_view(m,n).lev < mblev.d_view(m));
        const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
        [&](const int idx) {
          const int kk = idx/nji;
          const int jj = (idx - kk*nji)/ni;
          const int ii = idx - kk*nji - jj*ni;
          const int k = kl + kk;
          const int j = jl + jj;
          const int i = il + ii;
          Real val;
          if (coarser) {
            if (v == 0) {
              val = cb.x1f(m,k,j,i);
            } else if (v == 1) {
              val = cb.x2f(m,k,j,i);
            } else {
              val = cb.x3f(m,k,j,i);
            }
          } else {
            if (v == 0) {
              val = b.x1f(m,k,j,i);
            } else if (v == 1) {
              val = b.x2f(m,k,j,i);
            } else {
              val = b.x3f(m,k,j,i);
            }
          }
          const int bi = ndat*v + ii + ni*(jj + nj*kk);
          if (same_rank) {
            rbuf[dn].vars(dm,bi) = val;
          } else {
            sbuf[n].vars(m,bi) = val;
          }
        });
      } // end if-neighbor-exists block
      tmember.team_barrier();
    }
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  if (global_variable::nranks == 1) return TaskStatus::complete;
  pack_exec.fence();
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  bool no_errors=true;
  const int nwork_send_h = nmb;
  for (int a=0; a<nwork_send_h; ++a) {
    const int m = a;
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = 3;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= sendbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= sendbuf[n].isame_ndat;
          } else {
            data_size *= sendbuf[n].ifine_ndat;
          }
          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
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
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus MeshBoundaryValuesFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b,
                                                 DvceFaceFld4D<Real> &cb) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
#if MPI_PARALLEL_ENABLED
  if (rank_packed_bvals_active_) {
    // Test the requests that were STARTED (the layout InitRecv selected), with one
    // MPI_Testall rather than a poll per peer: that is a single progress pass, and
    // because Testall modifies no request unless the whole set completed, a persistent
    // request -- which goes INACTIVE rather than to MPI_REQUEST_NULL -- can never be
    // re-tested into an EMPTY status whose count would fail the size check below.
    bool no_errors=true;
    const bool all_recvd = TestVarRecvComplete(&no_errors);
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error or size mismatch in rank-packed receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!all_recvd) return TaskStatus::incomplete;

    auto &mblev = pmy_pack->pmb->mb_lev;
    auto aggrbuf = rank_recvbuf_vars_;
    auto recvoff = recv_agg_offset_;
    const int nwork_recv_rank = nmb;
    if (nwork_recv_rank <= 0) return TaskStatus::complete;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nwork_recv_rank), Kokkos::AUTO);
    Kokkos::parallel_for("RecvBuffRankPackedFC", athenak_lw(policy),
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int im = tmember.league_rank()/3;
      const int m = im;
      const int v = tmember.league_rank()%3;

      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.d_view(m,n).gid >= 0) {
          int il, iu, jl, ju, kl, ku, ndat;
          if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
            il = rbuf[n].icoar[v].bis; iu = rbuf[n].icoar[v].bie;
            jl = rbuf[n].icoar[v].bjs; ju = rbuf[n].icoar[v].bje;
            kl = rbuf[n].icoar[v].bks; ku = rbuf[n].icoar[v].bke;
            ndat = rbuf[n].icoar_ndat;
          } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            il = rbuf[n].isame[v].bis; iu = rbuf[n].isame[v].bie;
            jl = rbuf[n].isame[v].bjs; ju = rbuf[n].isame[v].bje;
            kl = rbuf[n].isame[v].bks; ku = rbuf[n].isame[v].bke;
            ndat = rbuf[n].isame_ndat;
          } else {
            il = rbuf[n].ifine[v].bis; iu = rbuf[n].ifine[v].bie;
            jl = rbuf[n].ifine[v].bjs; ju = rbuf[n].ifine[v].bje;
            kl = rbuf[n].ifine[v].bks; ku = rbuf[n].ifine[v].bke;
            ndat = rbuf[n].ifine_ndat;
          }
          const int ni = iu - il + 1;
          const int nj = ju - jl + 1;
          const int nk = ku - kl + 1;
          const int nkji = nk*nj*ni;
          const int nji = nj*ni;
          const int base = recvoff(m*nnghbr + n);
          const bool coarser = (nghbr.d_view(m,n).lev < mblev.d_view(m));
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = idx/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            const int bi = ndat*v + i-il + ni*(j-jl + nj*(k-kl));
            const Real val = (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
            if (coarser) {
              if (v == 0) {
                cb.x1f(m,k,j,i) = val;
              } else if (v == 1) {
                cb.x2f(m,k,j,i) = val;
              } else {
                cb.x3f(m,k,j,i) = val;
              }
            } else {
              if (IsActiveFCFace(v, k, j, i, indcs)) return;
              if (v == 0) {
                b.x1f(m,k,j,i) = val;
              } else if (v == 1) {
                b.x2f(m,k,j,i) = val;
              } else {
                b.x3f(m,k,j,i) = val;
              }
            }
          });
          tmember.team_barrier();
        }
      }
    });

    if (nwork_recv_rank > 0) MarkRecvUnpackPending();
    return TaskStatus::complete;
  }

  //----- STEP 1: check that recv boundary buffer communications have all completed
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    const int nwork_recv_h = nmb;
    for (int a=0; a<nwork_recv_h; ++a) {
      const int m = a;
      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.h_view(m,n).gid >= 0) { // ID != -1, so not a physical boundary
          if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {
              bflag = true;
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

  //----- STEP 2: buffers have all completed, so unpack 3-components of field

  auto &mblev = pmy_pack->pmb->mb_lev;
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(three field components)
  const int nwork_recv = nmb;
  if (nwork_recv <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nwork_recv), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int im = tmember.league_rank()/3;
    const int m = im;
    const int v = tmember.league_rank()%3;

    // scalar loop over neighbors to prevent race condition in overlapping assignments
    for (int n=0; n<nnghbr; ++n) {
      // only unpack buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        // if neighbor is at coarser level, use cindices to unpack buffer
        int il, iu, jl, ju, kl, ku, ndat;
        if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
          il = rbuf[n].icoar[v].bis;
          iu = rbuf[n].icoar[v].bie;
          jl = rbuf[n].icoar[v].bjs;
          ju = rbuf[n].icoar[v].bje;
          kl = rbuf[n].icoar[v].bks;
          ku = rbuf[n].icoar[v].bke;
          ndat = rbuf[n].icoar_ndat;
        // if neighbor is at same level, use sindices to unpack buffer
        } else if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = rbuf[n].isame[v].bis;
          iu = rbuf[n].isame[v].bie;
          jl = rbuf[n].isame[v].bjs;
          ju = rbuf[n].isame[v].bje;
          kl = rbuf[n].isame[v].bks;
          ku = rbuf[n].isame[v].bke;
          ndat = rbuf[n].isame_ndat;
        // if neighbor is at finer level, use findices to unpack buffer
        } else {
          il = rbuf[n].ifine[v].bis;
          iu = rbuf[n].ifine[v].bie;
          jl = rbuf[n].ifine[v].bjs;
          ju = rbuf[n].ifine[v].bje;
          kl = rbuf[n].ifine[v].bks;
          ku = rbuf[n].ifine[v].bke;
          ndat = rbuf[n].ifine_ndat;
        }
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nkji = nk*nj*ni;
        const int nji  = nj*ni;

        // if neighbor is at same or finer level, load data directly into b0
        if (nghbr.d_view(m,n).lev >= mblev.d_view(m)) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = (idx)/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            if (IsActiveFCFace(v, k, j, i, indcs)) return;
            if (v==0) {
              b.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==1) {
              b.x2f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==2) {
              b.x3f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            }
          });
        // if neighbor is at coarser level, load data into coarse_b0
        } else {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji),
          [&](const int idx) {
            int k = (idx)/nji;
            int j = (idx - k*nji)/ni;
            int i = (idx - k*nji - j*ni) + il;
            k += kl;
            j += jl;
            if (v==0) {
              cb.x1f(m,k,j,i) = rbuf[n].vars(m,i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==1) {
              cb.x2f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            } else if (v==2) {
              cb.x3f(m,k,j,i) = rbuf[n].vars(m,ndat*v + i-il + ni*(j-jl + nj*(k-kl)));
            }
          });
        }
        tmember.team_barrier();
      }  // end if-neighbor-exists block
    }
  });  // end par_for_outer

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1 && nwork_recv > 0) MarkRecvUnpackPending();
#endif

  return TaskStatus::complete;
}
