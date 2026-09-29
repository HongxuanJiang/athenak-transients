//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file flux_correction_fc.cpp
//! \brief functions to pack/send and recv/unpack fluxes (emfs) for face-centered fields
//! (magnetic fields) at fine/coarse boundaries for the flux correction step.

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesFC::PackAndSendFluxFC()
//! \brief Pack restricted fluxes of face-centered fields at fine/coarse boundaries
//! into boundary buffers and send to neighbors for flux-correction step. These fluxes
//! (e.g. EMFs) live at cell edges.
//!
//! This routine packs ALL the buffers on ALL the faces simultaneously for ALL the
//! MeshBlocks. Buffer data are then sent (via MPI) or copied directly for periodic or
//! block boundaries.

TaskStatus MeshBoundaryValuesFC::PackAndSendFluxFC(DvceEdgeFld4D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  DevExeSpace flux_exec;

  auto &cis = pmy_pack->pmesh->mb_indcs.cis;
  auto &cjs = pmy_pack->pmesh->mb_indcs.cjs;
  auto &cks = pmy_pack->pmesh->mb_indcs.cks;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  auto &one_d = pmy_pack->pmesh->one_d;
  auto &two_d = pmy_pack->pmesh->two_d;
  const int nsend_interfaces = nmb*nnghbr;
  if (nsend_interfaces <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(flux_exec, (3*nsend_interfaces), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int q = tmember.league_rank()/3;
    const int mn = q;
    const int m = mn/nnghbr;
    const int n = mn - m*nnghbr;
    const int v = tmember.league_rank() - 3*q;

    // only load buffers when neighbor exists and is at same or coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev <= mblev.d_view(m))) {
      // if neighbor is at coarser level, use cindices to pack buffer
      // Note indices can be different for each component of flux
      int il, iu, jl, ju, kl, ku, ndat;
      if (nghbr.d_view(m,n).lev < mblev.d_view(m)) {
        il = sbuf[n].iflux_coar[v].bis;
        iu = sbuf[n].iflux_coar[v].bie;
        jl = sbuf[n].iflux_coar[v].bjs;
        ju = sbuf[n].iflux_coar[v].bje;
        kl = sbuf[n].iflux_coar[v].bks;
        ku = sbuf[n].iflux_coar[v].bke;
        ndat = sbuf[n].iflxc_ndat;
      // if neighbor is at same level, use sindices to pack buffer
      } else {
        il = sbuf[n].iflux_same[v].bis;
        iu = sbuf[n].iflux_same[v].bie;
        jl = sbuf[n].iflux_same[v].bjs;
        ju = sbuf[n].iflux_same[v].bje;
        kl = sbuf[n].iflux_same[v].bks;
        ku = sbuf[n].iflux_same[v].bke;
        ndat = sbuf[n].iflxs_ndat;
      }
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

      // x1faces (only load x2e and x3e)
      if (n<8) {
        // i-index is fixed for flux correction on x1faces
        const int fi = 2*il - cis;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;
          int fj = 2*j - cjs;
          int fk = 2*k - cks;
          if (v==1) {
            Real rflx;
            // if neighbor is at same level, load x2e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x2e(m,k,j,il);
            // if neighbor is at coarser level, restrict x2e
            } else {
              if (one_d) {
                rflx = flx.x2e(m,0,0,fi);
              } else if (two_d) {
                rflx = 0.5*(flx.x2e(m,0,fj,fi) + flx.x2e(m,0,fj+1,fi));
              } else {
                rflx = 0.5*(flx.x2e(m,fk,fj,fi) + flx.x2e(m,fk,fj+1,fi));
              }
            }
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + (j-jl + nj*(k-kl))) = rflx;
            // else copy into send buffer for MPI communication below
            } else {
              const int idx_buf = ndat*v + (j-jl + nj*(k-kl));
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          } else if (v==2) {
            Real rflx;
            // if neighbor is at same level, load x3e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x3e(m,k,j,il);
            // if neighbor is at coarser level, restrict x3e
            } else {
              if (one_d) {
                rflx = flx.x3e(m,0,0,fi);
              } else if (two_d) {
                rflx = flx.x3e(m,0,fj,fi);
              } else {
                rflx = 0.5*(flx.x3e(m,fk,fj,fi) + flx.x3e(m,fk+1,fj,fi));
              }
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + (j-jl + nj*(k-kl))) = rflx;
            } else {
              const int idx_buf = ndat*v + (j-jl + nj*(k-kl));
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          }
        });

      // x2faces (only load x1e and x3e)
      } else if (n<16) {
        // j-index is fixed for flux correction on x2faces
        const int j = jl;
        const int fj = 2*jl - cjs;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nki), [&](const int idx) {
          int k = idx / ni;
          int i = (idx - k * ni) + il;
          k += kl;
          int fk = 2*k - cks;
          int fi = 2*i - cis;
          if (v==0) {
            Real rflx;
            // if neighbor is at same level, load x1e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x1e(m,k,j,i);
            // if neighbor is at coarser level, restrict x1e
            } else {
              if (two_d) {
                rflx = 0.5*(flx.x1e(m,0,fj,fi) + flx.x1e(m,0,fj,fi+1));
              } else {
                rflx = 0.5*(flx.x1e(m,fk,fj,fi) + flx.x1e(m,fk,fj,fi+1));
              }
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + i-il + ni*(k-kl)) = rflx;
            } else {
              const int idx_buf = ndat*v + i-il + ni*(k-kl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          } else if (v==2) {
            Real rflx;
            // if neighbor is at same level, load x3e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x3e(m,k,j,i);
            // if neighbor is at coarser level, restrict x3e
            } else {
              if (two_d) {
                rflx = flx.x3e(m,0,fj,fi);
              } else {
                rflx = 0.5*(flx.x3e(m,fk,fj,fi) + flx.x3e(m,fk+1,fj,fi));
              }
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + i-il + ni*(k-kl)) = rflx;
            } else {
              const int idx_buf = ndat*v + i-il + ni*(k-kl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          }
        });

      // x1x2 edges (only load x3e)
      } else if (n<24) {
        // i/j-index is fixed for flux correction on x1x2 edges
        const int i = il;
        const int j = jl;
        const int fi = 2*il - cis;
        const int fj = 2*jl - cjs;
        if (v==2) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nk),[&](const int idx) {
            int k = idx + kl;
            int fk = 2*k - cks;
            Real rflx;
            // if neighbor is at same level, load x3e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x3e(m,k,j,i);
            // if neighbor is at coarser level, restrict x3e
            } else {
              if (two_d) {
                rflx = flx.x3e(m,0,fj,fi);
              } else {
                rflx = 0.5*(flx.x3e(m,fk,fj,fi) + flx.x3e(m,fk+1,fj,fi));
              }
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + (k-kl)) = rflx;
            } else {
              const int idx_buf = ndat*v + (k-kl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          });
        }

      // x3faces (only load x1e and x2e)
      } else if (n<32) {
        // k-index is fixed for flux correction on x3faces
        const int k = kl;
        const int fk = 2*kl - cks;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nji), [&](const int idx) {
          int j = idx / ni;
          int i = (idx - j * ni) + il;
          j += jl;
          int fi = 2*i - cis;
          int fj = 2*j - cjs;
          if (v==0) {
            Real rflx;
            // if neighbor is at same level, load x1e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x1e(m,k,j,i);
            // if neighbor is at coarser level, restrict x1e
            } else {
              rflx = 0.5*(flx.x1e(m,fk,fj,fi) + flx.x1e(m,fk,fj,fi+1));
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + i-il + ni*(j-jl)) = rflx;
            } else {
              const int idx_buf = ndat*v + i-il + ni*(j-jl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          } else if (v==1) {
            Real rflx;
            // if neighbor is at same level, load x2e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x2e(m,k,j,i);
            // if neighbor is at coarser level, restrict x2e
            } else {
              rflx = 0.5*(flx.x2e(m,fk,fj,fi) + flx.x2e(m,fk,fj+1,fi));
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + i-il + ni*(j-jl)) = rflx;
            } else {
              const int idx_buf = ndat*v + i-il + ni*(j-jl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          }
        });

      // x3x1 edges (only load x2e)
      } else if (n<40) {
        const int i = il;
        const int k = kl;
        const int fi = 2*il - cis;
        const int fk = 2*kl - cks;
        if (v==1) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nj),[&](const int idx) {
            int j = idx + jl;
            int fj = 2*j - cjs;
            Real rflx;
            // if neighbor is at same level, load x2e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x2e(m,k,j,i);
            // if neighbor is at coarser level, restrict x2e
            } else {
              rflx = 0.5*(flx.x2e(m,fk,fj,fi) + flx.x2e(m,fk,fj+1,fi));
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + (j-jl)) = rflx;
            } else {
              const int idx_buf = ndat*v + (j-jl);
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          });
        }

      // x2x3 edges (only load x1e)
      } else if (n<48) {
        const int j = jl;
        const int k = kl;
        const int fj = 2*jl - cjs;
        const int fk = 2*kl - cks;
        if (v==0) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,ni),[&](const int idx) {
            int i = idx + il;
            int fi = 2*i - cis;
            Real rflx;
            // if neighbor is at same level, load x1e directly
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
              rflx = flx.x1e(m,k,j,i);
            // if neighbor is at coarser level, restrict x1e
            } else {
              rflx = 0.5*(flx.x1e(m,fk,fj,fi) + flx.x1e(m,fk,fj,fi+1));
            }
            if (nghbr.d_view(m,n).rank == my_rank) {
              rbuf[dn].flux(dm, ndat*v + i-il) = rflx;
            } else {
              const int idx_buf = ndat*v + i-il;
              sbuf[n].flux(m, idx_buf) = rflx;
            }
          });
        }
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  // Sends only occur to neighbors on FACES and EDGES at COARSER or SAME level
  if (global_variable::nranks == 1) return TaskStatus::complete;
  flux_exec.fence();
  bool no_errors=true;
  const int nsend_interfaces_h = nmb*nnghbr;
  for (int q=0; q<nsend_interfaces_h; ++q) {
    const int mn = q;
    const int m = mn/nnghbr;
    const int n = mn - m*nnghbr;
    if ( (nghbr.h_view(m,n).gid >=0) &&
         (nghbr.h_view(m,n).lev <= mblev.h_view(m)) &&
         (n<48) ) {
      // index and rank of destination Neighbor
      int dn = nghbr.h_view(m,n).dest;
      int drank = nghbr.h_view(m,n).rank;

      if (drank != my_rank) {
        // create tag using local ID and buffer index of *receiving* MeshBlock
        int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
        int tag = CreateBvals_MPI_Tag(lid, dn);

        // get ptr to send buffer for fluxes
        int data_size = 3;
        if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
          data_size *= sendbuf[n].iflxc_ndat;
        } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
          data_size *= sendbuf[n].iflxs_ndat;
        }
        auto send_ptr = Kokkos::subview(sendbuf[n].flux, m, Kokkos::ALL);

        int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                             comm_flux, &(sendbuf[n].flux_req[m]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
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
//! \fn void RecvAndUnpackFluxFC()
//! \brief Unpack boundary buffers for flux correction of FC variables.  This requires
//! averaging together fluxes from MeshBlocks at the same level, or replacing the fluxes
//! with the average from MeshBlocks at finer levels.

TaskStatus MeshBoundaryValuesFC::RecvAndUnpackFluxFC(DvceEdgeFld4D<Real> &flx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
#if MPI_PARALLEL_ENABLED
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int my_rank = global_variable::my_rank;
  bool remote_flux_recv = false;
  //----- STEP 1: check that recv boundary buffer communications have all completed
  // receives only occur for neighbors on faces and edges at FINER or SAME level
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    const int nwork_recv = nmb;
    const int nedge_nghbr = std::min(48, nnghbr);
    for (int a=0; a<nwork_recv; ++a) {
      const int m = a;
      for (int n=0; n<nedge_nghbr; ++n) {
        if (nghbr.h_view(m,n).gid >= 0 &&
            nghbr.h_view(m,n).lev >= mblev.h_view(m) &&
            nghbr.h_view(m,n).rank != my_rank) {
          remote_flux_recv = true;
          int test;
          int ierr = MPI_Test(&(rbuf[n].flux_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) no_errors = false;
          if (!static_cast<bool>(test)) bflag = true;
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

  //----- STEP 2: buffers have all completed, so unpack and perform appropriate averaging

  // Face slots store fine-child seam counts; edge slots store the number of EMFs summed
  // at each coarse edge.  Face seams have no local contribution before finer buffers are
  // unpacked, while every coarse edge starts with its local EMF.
  if (fc_flux_count_scratch_.extent_int(0) != nmb) {
    fc_flux_count_scratch_ = DvceArray2D<int>("nflx", nmb, 48);
  }
  DvceArray2D<int> nflx = fc_flux_count_scratch_;
  par_for("init_nflx", DevExeSpace(), 0, (nmb-1), 0, 47,
  KOKKOS_LAMBDA(const int m, const int n) {
    const bool face_slot = n < 16 || (n >= 24 && n < 32);
    nflx(m,n) = face_slot ? 0 : 1;
  });

  SumBoundaryFluxes(flx, true, nflx);

  // A coarse edge can be covered by two different fine children along its length.  The
  // legacy bookkeeping used only the primary (even) edge slot as one count for both
  // halves, which is valid only when all fine children participate together.  LAT can
  // select the two children independently, so seed the otherwise-unused odd edge slot
  // with the same-level/local count before applying fine-level replacements below.
  const int nwork_counts = nmb;
  if (nwork_counts > 0) {
    par_for("init_fc_subedge_counts", DevExeSpace(), 0, nwork_counts-1, 0, 11,
    KOKKOS_LAMBDA(const int a, const int q) {
      const int m = a;
      int edge;
      if (q < 4) {
        edge = 16 + 2*q;
      } else if (q < 8) {
        edge = 32 + 2*(q - 4);
      } else {
        edge = 40 + 2*(q - 8);
      }
      nflx(m,edge + 1) = nflx(m,edge);
    });
  }

  // Zero EMFs at boundary that overlap with finer MeshBlocks (only use fine fluxes there)
  // Then unpack and sum fluxes from finer levels
  if (pmy_pack->pmesh->multilevel) {
    ZeroFluxesAtBoundaryWithFiner(flx, nflx);
    SumBoundaryFluxes(flx, false, nflx);
  }

  // perform appropriate averaging depending on how many fluxes contributed to sums
  AverageBoundaryFluxes(flx, nflx);

#if MPI_PARALLEL_ENABLED
  if (remote_flux_recv) MarkRecvUnpackPending();
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void BoundaryValuesFC::SumBoundaryFluxes
//! \brief Sums boundary buffer fluxes from neighboring MeshBlocks at the same level into
//! flux (e.g. EMF) array if input argument 'same_level=true', or sums boundary buffer
//! fluxes from neighboring MeshBlocks at a finer level into flux array otherwise.

void MeshBoundaryValuesFC::SumBoundaryFluxes(DvceEdgeFld4D<Real> &flx,
                                          const bool same_level, DvceArray2D<int> &nflx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;

  // Sum receive buffers into EMFs stored on MeshBlocks
  // Outer loop over (# of MeshBlocks)*(3 field components)
  const int nwork_recv = nmb;
  if (nwork_recv <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nwork_recv), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/3;
    const int m = a;
    const int v = tmember.league_rank()%3;

    // scalar loop over neighbors (except corners) to prevent race condition in sums
    for (int n=0; n<nnghbr; ++n) {
      // only unpack buffers when neighbor exists AND
      // (neighbor at same level when same_level=true on input) OR
      // (neighbor at finer level when same_level=false on input)
      if (    (nghbr.d_view(m,n).gid >= 0) &&
           (( (same_level) && (nghbr.d_view(m,n).lev == mblev.d_view(m))) ||
            (!(same_level) && (nghbr.d_view(m,n).lev >  mblev.d_view(m)))) ) {
        bool inner_bufs=false, outer_bufs=false;
        if ((n==0)||(n==16)||(n==20)||(n==32)||(n==33)||(n==36)||(n==37)) {
          inner_bufs=true;
        }
        if ((n==4)||(n==18)||(n==22)||(n==34)||(n==35)||(n==38)||(n==39)) {
          outer_bufs=true;
        }

      // only unpack buffers when
      //   (both innerx1 AND outerx1 BCs are NOT shear_periodic) OR
      //   (only innerx1 BC is shear_periodic AND n!=0,16,20,32,33,36,37) OR
      //   (only outerx1 BC is shear_periodic AND n!=4,18,22,34,35,38,39) OR
      //   (both innerx1 AND outerx1 BC are shear_periodic AND n!=...)
      if ( ((mbbcs.d_view(m,0)!=BoundaryFlag::shear_periodic) &&
            (mbbcs.d_view(m,1)!=BoundaryFlag::shear_periodic)) ||
           ((mbbcs.d_view(m,0)==BoundaryFlag::shear_periodic) && !(inner_bufs) &&
            (mbbcs.d_view(m,1)!=BoundaryFlag::shear_periodic)) ||
           ((mbbcs.d_view(m,0)!=BoundaryFlag::shear_periodic) &&
            (mbbcs.d_view(m,1)==BoundaryFlag::shear_periodic) && !(outer_bufs)) ||
           ((mbbcs.d_view(m,0)==BoundaryFlag::shear_periodic) && !(inner_bufs) &&
            (mbbcs.d_view(m,1)==BoundaryFlag::shear_periodic) && !(outer_bufs)) ) {
        int il, iu, jl, ju, kl, ku, ndat;
        // if neighbor is at same level, use same indices to unpack buffer
        if (same_level) {
          il = rbuf[n].iflux_same[v].bis;
          iu = rbuf[n].iflux_same[v].bie;
          jl = rbuf[n].iflux_same[v].bjs;
          ju = rbuf[n].iflux_same[v].bje;
          kl = rbuf[n].iflux_same[v].bks;
          ku = rbuf[n].iflux_same[v].bke;
          ndat = rbuf[n].iflxs_ndat;
        // else neighbor is at finer level, use flux_coar indices to unpack buffer
        } else {
          il = rbuf[n].iflux_coar[v].bis;
          iu = rbuf[n].iflux_coar[v].bie;
          jl = rbuf[n].iflux_coar[v].bjs;
          ju = rbuf[n].iflux_coar[v].bje;
          kl = rbuf[n].iflux_coar[v].bks;
          ku = rbuf[n].iflux_coar[v].bke;
          ndat = rbuf[n].iflxc_ndat;
        }
        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nji  = nj*ni;
        const int nkj  = nk*nj;
        const int nki  = nk*ni;

        // x1faces
        if (n<8) {
          // always use v=0 thread index in sums to avoid race condition
          if (v==0) {
            if (!same_level) {
              const int side = (n >= 4);
              const int s = n & 3;
              const int fy = s & 1;
              const int fz = s >> 1;
              const int face = side ? 4 : 0;
              const int e12 = 16 + 2*side + 4*fy + fz;
              const int e31 = 32 + 2*side + 4*fz + fy;
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                // x2e overlaps across the fz seam, segmented by fy; x3e overlaps
                // across the fy seam, segmented by fz.
                nflx(m,face + fy) += 1;
                nflx(m,face + 2 + fz) += 1;
                nflx(m,e12) += 1;
                nflx(m,e31) += 1;
              });
            } else if (n==0) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,16) += 1;
                nflx(m,20) += 1;
                nflx(m,32) += 1;
                nflx(m,36) += 1;
              });
            } else if (n==4) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,18) += 1;
                nflx(m,22) += 1;
                nflx(m,34) += 1;
                nflx(m,38) += 1;
              });
            }
          } else {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkj),
            [&](const int idx){
              int k = idx / nj;
              int j = (idx - k * nj) + jl;
              k += kl;
              if (v==1) {
                flx.x2e(m,k,j,il) += rbuf[n].flux(m,ndat*v + (j-jl + nj*(k-kl)));
              } else if (v==2) {
                flx.x3e(m,k,j,il) += rbuf[n].flux(m,ndat*v + (j-jl + nj*(k-kl)));
              }
            });
          }

        // x2faces
        } else if (n<16) {
          // always use v=0 thread index in sums to avoid race condition
          if (v==0) {
            if (!same_level) {
              const int side = (n >= 12);
              const int s = (n - 8) & 3;
              const int fx = s & 1;
              const int fz = s >> 1;
              const int face = side ? 12 : 8;
              const int e12 = 16 + 4*side + 2*fx + fz;
              const int e23 = 40 + 2*side + 4*fz + fx;
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                // x1e overlaps across the fz seam, segmented by fx; x3e overlaps
                // across the fx seam, segmented by fz.
                nflx(m,face + fx) += 1;
                nflx(m,face + 2 + fz) += 1;
                nflx(m,e12) += 1;
                nflx(m,e23) += 1;
              });
            } else if (n==8) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,16) += 1;
                nflx(m,18) += 1;
                nflx(m,40) += 1;
                nflx(m,44) += 1;
              });
            } else if (n==12) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,20) += 1;
                nflx(m,22) += 1;
                nflx(m,42) += 1;
                nflx(m,46) += 1;
              });
            }
          }
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nki),
          [&](const int idx){
            int k = idx/ni;
            int i = (idx - k * ni) + il;
            k += kl;
            if (v==0) {
              flx.x1e(m,k,jl,i) += rbuf[n].flux(m,ndat*v + i-il + ni*(k-kl));
            } else if (v==2) {
              flx.x3e(m,k,jl,i) += rbuf[n].flux(m,ndat*v + i-il + ni*(k-kl));
            }
          });

        // x1x2 edges
        } else if (n<24) {
          // always use v=0 thread index in sums to avoid race condition
          if (v==0) {
            Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
              nflx(m,n) += 1;
            });
          } else if (v==2) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nk),[&](const int idx){
              int k = idx + kl;
              flx.x3e(m,k,jl,il) += rbuf[n].flux(m,ndat*v + (k-kl));
            });
          }

        // x3faces
        } else if (n<32)  {
          // always use v=0 thread index in sums to avoid race condition
          if (v==0) {
            if (!same_level) {
              const int side = (n >= 28);
              const int s = (n - 24) & 3;
              const int fx = s & 1;
              const int fy = s >> 1;
              const int face = side ? 28 : 24;
              const int e31 = 32 + 4*side + 2*fx + fy;
              const int e23 = 40 + 4*side + 2*fy + fx;
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                // x1e overlaps across the fy seam, segmented by fx; x2e overlaps
                // across the fx seam, segmented by fy.
                nflx(m,face + fx) += 1;
                nflx(m,face + 2 + fy) += 1;
                nflx(m,e31) += 1;
                nflx(m,e23) += 1;
              });
            } else if (n==24) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,32) += 1;
                nflx(m,34) += 1;
                nflx(m,40) += 1;
                nflx(m,42) += 1;
              });
            } else if (n==28) {
              Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
                nflx(m,36) += 1;
                nflx(m,38) += 1;
                nflx(m,44) += 1;
                nflx(m,46) += 1;
              });
            }
          }
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nji),
          [&](const int idx){
            int j = idx / ni;
            int i = (idx - j * ni) + il;
            j += jl;
            if (v==0) {
              flx.x1e(m,kl,j,i) += rbuf[n].flux(m,ndat*v + i-il + ni*(j-jl));
            } else if (v==1) {
              flx.x2e(m,kl,j,i) += rbuf[n].flux(m,ndat*v + i-il + ni*(j-jl));
            }
          });

        // x3x1 edges
        } else if (n<40) {
          // always use v=0 thread index in sums to avoid race condition
          if (v==0) {
            Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
              nflx(m,n) += 1;
            });
          } else if (v==1) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nj),[&](const int idx){
              int j = idx + jl;
              flx.x2e(m,kl,j,il) += rbuf[n].flux(m,ndat*v + (j-jl));
            });
          }

        // x2x3 edges
        } else if (n<48) {
          if (v==0) {
            Kokkos::single(Kokkos::PerTeam(tmember), [&] () {
              nflx(m,n) += 1;
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,ni),[&](const int idx){
              int i = idx + il;
              flx.x1e(m,kl,jl,i) += rbuf[n].flux(m,ndat*v + i-il);
            });
          }
        }
      }  // end if-neighbor-exists block
      }
      tmember.team_barrier();
    }    // end for loop over n
  });    // end par_for_outer

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValuesFC::ZeroFluxesAtBoundaryWithFiner
//! \brief Zeroes out fluxes of face-centered variables (e.g. EMFs) at boundaries with
//! MeshBlocks at a finer level, so that boundary buffer fluxes from finer level can be
//! summed (averaged) in place.

void MeshBoundaryValuesFC::ZeroFluxesAtBoundaryWithFiner(DvceEdgeFld4D<Real> &flx,
                                                         DvceArray2D<int> &nflx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &mblev = pmy_pack->pmb->mb_lev;

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(3 field components)
  const int nwork_recv = nmb;
  if (nwork_recv <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nwork_recv*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = (tmember.league_rank())/(3*nnghbr);
    const int m = a;
    const int n = (tmember.league_rank() - a*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - a*(3*nnghbr) - 3*n);

    // only zero EMFs when neighbor exists and is at finer level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev > mblev.d_view(m))) {
      int il, iu, jl, ju, kl, ku;
      il = rbuf[n].iflux_coar[v].bis;
      iu = rbuf[n].iflux_coar[v].bie;
      jl = rbuf[n].iflux_coar[v].bjs;
      ju = rbuf[n].iflux_coar[v].bje;
      kl = rbuf[n].iflux_coar[v].bks;
      ku = rbuf[n].iflux_coar[v].bke;
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nji  = nj*ni;
      const int nkj  = nk*nj;
      const int nki  = nk*ni;

      // x1faces
      if (n<8) {
        // use idle thread index to zero number of fluxes at corners of x1faces
        if (v==0) {
          const int side = (n >= 4);
          const int s = n & 3;
          const int fy = s & 1;
          const int fz = s >> 1;
          const int e12 = 16 + 2*side + 4*fy + fz;
          const int e31 = 32 + 2*side + 4*fz + fy;
          Kokkos::atomic_exchange(&nflx(m,e12), 0);
          Kokkos::atomic_exchange(&nflx(m,e31), 0);
        // else zero fluxes
        } else {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nkj),[&](const int idx){
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;
            if (v==1) {
              if (k == kl || k == ku) {
                Kokkos::atomic_exchange(
                    &flx.x2e(m,k,j,il), static_cast<Real>(0.0));
              } else {
                flx.x2e(m,k,j,il) = 0.0;
              }
            } else if (v==2) {
              if (j == jl || j == ju) {
                Kokkos::atomic_exchange(
                    &flx.x3e(m,k,j,il), static_cast<Real>(0.0));
              } else {
                flx.x3e(m,k,j,il) = 0.0;
              }
            }
          });
        }

      // x2faces
      } else if (n<16) {
        // use idle thread index to zero number of fluxes at corners of x2faces
        if (v==1) {
          const int side = (n >= 12);
          const int s = (n - 8) & 3;
          const int fx = s & 1;
          const int fz = s >> 1;
          const int e12 = 16 + 4*side + 2*fx + fz;
          const int e23 = 40 + 2*side + 4*fz + fx;
          Kokkos::atomic_exchange(&nflx(m,e12), 0);
          Kokkos::atomic_exchange(&nflx(m,e23), 0);
        // else zero fluxes
        } else {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nki),[&](const int idx){
            int k = idx/ni;
            int i = (idx - k * ni) + il;
            k += kl;
            if (v==0) {
              if (k == kl || k == ku) {
                Kokkos::atomic_exchange(
                    &flx.x1e(m,k,jl,i), static_cast<Real>(0.0));
              } else {
                flx.x1e(m,k,jl,i) = 0.0;
              }
            } else if (v==2) {
              if (i == il || i == iu) {
                Kokkos::atomic_exchange(
                    &flx.x3e(m,k,jl,i), static_cast<Real>(0.0));
              } else {
                flx.x3e(m,k,jl,i) = 0.0;
              }
            }
          });
        }

      // x1x2 edges
      } else if (n<24) {
        if (v==2) {
          Kokkos::atomic_exchange(&nflx(m,n), 0);
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nk),[&](const int idx) {
            int k = idx + kl;
            Kokkos::atomic_exchange(
                &flx.x3e(m,k,jl,il), static_cast<Real>(0.0));
          });
        }

      // x3faces
      } else if (n<32)  {
        // use idle thread index to zero number of fluxes at corners of x2faces
        if (v==2) {
          const int side = (n >= 28);
          const int s = (n - 24) & 3;
          const int fx = s & 1;
          const int fy = s >> 1;
          const int e31 = 32 + 4*side + 2*fx + fy;
          const int e23 = 40 + 4*side + 2*fy + fx;
          Kokkos::atomic_exchange(&nflx(m,e31), 0);
          Kokkos::atomic_exchange(&nflx(m,e23), 0);
        // else zero fluxes
        } else {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nji),[&](const int idx){
            int j = idx / ni;
            int i = (idx - j * ni) + il;
            j += jl;
            if (v==0) {
              if (j == jl || j == ju) {
                Kokkos::atomic_exchange(
                    &flx.x1e(m,kl,j,i), static_cast<Real>(0.0));
              } else {
                flx.x1e(m,kl,j,i) = 0.0;
              }
            } else if (v==1) {
              if (i == il || i == iu) {
                Kokkos::atomic_exchange(
                    &flx.x2e(m,kl,j,i), static_cast<Real>(0.0));
              } else {
                flx.x2e(m,kl,j,i) = 0.0;
              }
            }
          });
        }

      // x3x1 edges
      } else if (n<40) {
        if (v==1) {
          Kokkos::atomic_exchange(&nflx(m,n), 0);
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nj),[&](const int idx){
            int j = idx + jl;
            Kokkos::atomic_exchange(
                &flx.x2e(m,kl,j,il), static_cast<Real>(0.0));
          });
        }

      // x2x3 edges
      } else if (n<48) {
        if (v==0) {
          Kokkos::atomic_exchange(&nflx(m,n), 0);
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,ni),[&](const int idx){
            int i = idx + il;
            Kokkos::atomic_exchange(
                &flx.x1e(m,kl,jl,i), static_cast<Real>(0.0));
          });
        }
      }
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValuesFC::AverageBoundaryFluxes
//! \brief Applies appropriate average to summed boundary fluxes, depending on number of
//! elements being averaged together.

void MeshBoundaryValuesFC::AverageBoundaryFluxes(DvceEdgeFld4D<Real> &flx,
                                                 DvceArray2D<int> &nflx) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  // Outer loop over (# of MeshBlocks)*(# of neighbors)*(3 field components)
  const int nwork_recv = nmb;
  if (nwork_recv <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (3*nwork_recv*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = (tmember.league_rank())/(3*nnghbr);
    const int m = a;
    const int n = (tmember.league_rank() - a*(3*nnghbr))/3;
    const int v = (tmember.league_rank() - a*(3*nnghbr) - 3*n);
    // only average when
    //   (both innerx1 AND outerx1 BCs are NOT shear_periodic) OR
    //   (only innerx1 BC is shear_periodic AND n!=0) OR
    //   (only outerx1 BC is shear_periodic AND n!=4) OR
    //   (both innerx1 AND outerx1 BC are shear_periodic AND n!=0,4)
    if ( ((mbbcs.d_view(m,0)!=BoundaryFlag::shear_periodic) &&
          (mbbcs.d_view(m,1)!=BoundaryFlag::shear_periodic)) ||
         ((mbbcs.d_view(m,0)==BoundaryFlag::shear_periodic) && (n!=0) &&
          (mbbcs.d_view(m,1)!=BoundaryFlag::shear_periodic)) ||
         ((mbbcs.d_view(m,0)!=BoundaryFlag::shear_periodic) &&
          (mbbcs.d_view(m,1)==BoundaryFlag::shear_periodic) && (n!=4)) ||
         ((mbbcs.d_view(m,0)==BoundaryFlag::shear_periodic) && (n!=0) &&
          (mbbcs.d_view(m,1)==BoundaryFlag::shear_periodic) && (n!=4)) ) {
      int il, iu, jl, ju, kl, ku;
      il = rbuf[n].iflux_same[v].bis;
      iu = rbuf[n].iflux_same[v].bie;
      jl = rbuf[n].iflux_same[v].bjs;
      ju = rbuf[n].iflux_same[v].bje;
      kl = rbuf[n].iflux_same[v].bks;
      ku = rbuf[n].iflux_same[v].bke;

      // x1faces
      if (n==0 || n==4) {
        if (v==1) {
          int nj = ju - jl + 1;
          // same level; divide EMFs on face by 2, excluding edges
          if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            if (three_d) {
              kl += 1; ku -= 1;
            }
            int nk = ku - kl + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*nj),
            [&](const int idx) {
              int k = idx / nj;
              int j = (idx - k * nj) + jl;
              k += kl;
              flx.x2e(m,k,j,il) *= 0.5;
            });
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            if (three_d) {
              int k = kl + (ku - kl + 1)/2;
              Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj),
              [&](const int idx) {
                int j = idx + jl;
                const int fy = (2*(j - jl) >= nj) ? 1 : 0;
                const int count = nflx(m,n + fy);
                if (count > 1) {
                  flx.x2e(m,k,j,il) /= static_cast<Real>(count);
                }
              });
              tmember.team_barrier();
            }
          }
        } else if (v==2) {
          int nk = ku - kl + 1;
          // same level; divide EMFs on face by 2, excluding edges
          if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            if (multi_d) {
              jl += 1; ju -= 1;
            }
            int nj = ju - jl + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*nj),
            [&](const int idx) {
              int k = idx / nj;
              int j = (idx - k * nj) + jl;
              k += kl;
              flx.x3e(m,k,j,il) *= 0.5;
            });
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            if (multi_d) {
              int j = jl + (ju - jl + 1)/2;
              Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk),
              [&](const int idx) {
                int k = idx + kl;
                const int fz = (2*(k - kl) >= nk) ? 1 : 0;
                const int count = nflx(m,n + 2 + fz);
                if (count > 1) {
                  flx.x3e(m,k,j,il) /= static_cast<Real>(count);
                }
              });
            }
          }
        }

      // x2faces
      } else if (multi_d && (n==8 || n==12)) {
        if (v==0) {
          int ni = iu - il + 1;
          // same level; divide EMFs on face by 2, excluding edges
          if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            if (three_d) {
              kl += 1; ku -= 1;
            }
            int nk = ku - kl + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*ni),
            [&](const int idx) {
              int k = idx/ni;
              int i = (idx - k * ni) + il;
              k += kl;
              flx.x1e(m,k,jl,i) *= 0.5;
            });
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            if (three_d) {
              int k = kl + (ku - kl + 1)/2;
              Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, ni),
              [&](const int idx) {
                int i = idx + il;
                const int fx = (2*(i - il) >= ni) ? 1 : 0;
                const int count = nflx(m,n + fx);
                if (count > 1) {
                  flx.x1e(m,k,jl,i) /= static_cast<Real>(count);
                }
              });
              tmember.team_barrier();
            }
          }
        } else if (v==2) {
          int nk = ku - kl + 1;
          // same level; divide EMFs on face by 2, excluding edges
          if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            il += 1; iu -= 1;
            int ni = iu - il + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*ni),
            [&](const int idx) {
              int k = idx/ni;
                int i = (idx - k * ni) + il;
              k += kl;
              flx.x3e(m,k,jl,i) *= 0.5;
            });
            tmember.team_barrier();
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            int i = il + (iu - il + 1)/2;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk),
            [&](const int idx) {
              int k = idx + kl;
              const int fz = (2*(k - kl) >= nk) ? 1 : 0;
              const int count = nflx(m,n + 2 + fz);
              if (count > 1) {
                flx.x3e(m,k,jl,i) /= static_cast<Real>(count);
              }
            });
            tmember.team_barrier();
          }
        }

      // x1x2 edges
      } else if (multi_d && (n==16 || n==18 || n==20 || n==22)) {
        if (v==2) {
          int nk = ku - kl + 1;
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nk),[&](const int idx) {
            int k = idx + kl;
            const int count = nflx(m,n + ((2*(k - kl) >= nk) ? 1 : 0));
            if (count > 1) flx.x3e(m,k,jl,il) /= static_cast<Real>(count);
          });
        }

      // x3faces
      } else if (three_d && (n==24 || n==28))  {
        if (v==0) {
          int ni = iu - il + 1;
          // same level; divide EMFs on face by 2, excluding edges
          if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            jl += 1; ju -= 1;
            int nj = ju - jl + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj*ni),
            [&](const int idx) {
              int j = idx / ni;
              int i = (idx - j * ni) + il;
              j += jl;
              flx.x1e(m,kl,j,i) *= 0.5;
            });
            tmember.team_barrier();
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            int j = jl + (ju - jl + 1)/2;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,ni),[&](const int idx){
              int i = idx + il;
              const int fx = (2*(i - il) >= ni) ? 1 : 0;
              const int count = nflx(m,n + fx);
              if (count > 1) {
                flx.x1e(m,kl,j,i) /= static_cast<Real>(count);
              }
            });
            tmember.team_barrier();
          }
        } else if (v==1) {
          int nj = ju - jl + 1;
          // same level; divide EMFs on face by 2, excluding edges
            if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            il += 1; iu -= 1;
            int ni = iu - il + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj*ni),
              [&](const int idx) {
                int j = idx / ni;
              int i = (idx - j * ni) + il;
              j += jl;
              flx.x2e(m,kl,j,i) *= 0.5;
            });
            tmember.team_barrier();
          // finer level; divide EMFs that overlap at edges of fine faces by 2
          } else if (nghbr.d_view(m,n).lev > mblev.d_view(m)) {
            int i = il + (iu - il + 1)/2;
            Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nj),[&](const int idx){
              int j = idx + jl;
              const int fy = (2*(j - jl) >= nj) ? 1 : 0;
              const int count = nflx(m,n + 2 + fy);
              if (count > 1) {
                flx.x2e(m,kl,j,i) /= static_cast<Real>(count);
              }
            });
            tmember.team_barrier();
          }
        }

      // x3x1 edges
      } else if (three_d && (n==32 || n==34 || n==36 || n==38)) {
        if (v==1) {
          int nj = ju - jl + 1;
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,nj),[&](const int idx) {
            int j = idx + jl;
            const int count = nflx(m,n + ((2*(j - jl) >= nj) ? 1 : 0));
            if (count > 1) flx.x2e(m,kl,j,il) /= static_cast<Real>(count);
          });
        }

    // x2x3 edges
    } else if (three_d && (n==40 || n==42 || n==44 || n==46)) {
      if (v==0) {
        int ni = iu - il + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember,ni),[&](const int idx) {
          int i = idx + il;
          const int count = nflx(m,n + ((2*(i - il) >= ni) ? 1 : 0));
          if (count > 1) flx.x1e(m,kl,jl,i) /= static_cast<Real>(count);
        });
      }
    }
    }
    tmember.team_barrier();
  });    // end par_for_outer

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValuesFC::InitRecvFlux
//! \brief Posts non-blocking receives (with MPI) for boundary communication of fluxes of
//! face-centered variables, which are communicated at FACES and EDGES of MeshBlocks at
//! the SAME or FINER levels.  This is different than for fluxes of cell-centered vars.

TaskStatus MeshBoundaryValuesFC::InitFluxRecv(const int nvars) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  WaitRecvUnpackComplete();
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  // Initialize communications of fluxes
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      // only post receives for neighbors on FACES and EDGES at FINER and SAME levels
      // this is the only thing different from BoundaryValuesCC::InitRecvFlux()
      if ( (nghbr.h_view(m,n).gid >=0) &&
           (nghbr.h_view(m,n).lev >= pmy_pack->pmb->mb_lev.h_view(m)) &&
           (n<48) ) {
        // rank of destination buffer
        int drank = nghbr.h_view(m,n).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (drank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, n);

          // calculate amount of data to be passed, get pointer to variables
          int data_size = nvars;
          if ( nghbr.h_view(m,n).lev > pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= recvbuf[n].iflxc_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= recvbuf[n].iflxs_ndat;
          }
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
