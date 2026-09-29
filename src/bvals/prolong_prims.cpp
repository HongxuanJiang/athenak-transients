//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file prolong_prims.cpp
//! \brief functions to convert conserved to primitive variables (and vice-versa) in
//! boundary buffers where prolongation is used at fine/coarse level boundaries.  This
//! enables prolongation in either the conserved or primitive variables.
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "bvals.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace {
template <typename NeighborView, typename LevelView, typename RecvBufferView>
KOKKOS_INLINE_FUNCTION
bool OwnsCoarsePrimitiveStencilCell(
    const NeighborView &nghbr, const LevelView &mblev,
    const RecvBufferView &rbuf, const bool multi_d, const bool three_d,
    const int m, const int n, const int k, const int j, const int i) {
  const int block_level = mblev(m);
  for (int owner = 0; owner < n; ++owner) {
    if (nghbr(m,owner).gid < 0 || nghbr(m,owner).lev >= block_level) continue;

    const auto &idx = rbuf(owner).iprol[0];
    const int il = idx.bis - 1;
    const int iu = idx.bie + 1;
    const int jl = idx.bjs - (multi_d ? 1 : 0);
    const int ju = idx.bje + (multi_d ? 1 : 0);
    const int kl = idx.bks - (three_d ? 1 : 0);
    const int ku = idx.bke + (three_d ? 1 : 0);
    if (i >= il && i <= iu && j >= jl && j <= ju && k >= kl && k <= ku) {
      return false;
    }
  }
  return true;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrimCoarseBndry()
//! \brief Converts Hydro conserved variables in coarse level boundary buffers into
//! Hydro primitive variables for prolongation. Coarse arrays should be passed in through
//! arguments.
//! Only works for hydrodynamics, the same function for MHD has different argument list.

void MeshBoundaryValuesCC::ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons,
                                                 DvceArray5D<Real> &prim) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  auto nghbr_device = nghbr.d_view;
  auto mblev_device = mblev.d_view;
  auto rbuf_device = recvbuf_device;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  const bool use_dual = pmy_pack->phydro->dual_energy_pdv;
  const int dual_idx = pmy_pack->phydro->dual_energy_idx;
  const Real dual_eta1 = pmy_pack->phydro->dual_energy_eta1;

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("Prol_C2P_CC", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/nnghbr;
    const int m = lat_enabled ? lat_active_indices(a) : a;
    const int n = tmember.league_rank() - a*nnghbr;

    // only convert coarse vars when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // use indices for prolongation on this buffer as loop limits.
      // Note that one extra cell is added to match stencil of 2nd-order prolongation
      int il = rbuf_device(n).iprol[0].bis - 1;
      int iu = rbuf_device(n).iprol[0].bie + 1;
      int jl = rbuf_device(n).iprol[0].bjs;
      int ju = rbuf_device(n).iprol[0].bje;
      if (multi_d) {
        jl -= 1;
        ju += 1;
      }
      int kl = rbuf_device(n).iprol[0].bks;
      int ku = rbuf_device(n).iprol[0].bke;
      if (three_d) {
        kl -= 1;
        ku += 1;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // Expanded face/edge/corner stencil boxes overlap.  Give every coarse cell to
        // exactly one neighbor team so C2P and scalar repair cannot race on prim/cons.
        if (!OwnsCoarsePrimitiveStencilCell(
                nghbr_device, mblev_device, rbuf_device, multi_d, three_d,
                m, n, k, j, i)) return;

        // load single state conserved variables
        HydCons1D u;
        u.d  = cons(m,IDN,k,j,i);
        u.mx = cons(m,IM1,k,j,i);
        u.my = cons(m,IM2,k,j,i);
        u.mz = cons(m,IM3,k,j,i);
        u.e  = cons(m,IEN,k,j,i);
        HydPrim1D w;
        Real dual_prim_e = 0.0;

        bool dfloor_used=false, efloor_used=false, tfloor_used=false;
        bool vceiling_used=false;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          // Note indices refer to coarse arrays, so use cis, cnx1
          Real x1v = CellCenterX(i-indcs.cis, indcs.cnx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-indcs.cjs, indcs.cnx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-indcs.cks, indcs.cnx3, x3min, x3max);

          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

          HydCons1D u_sr;
          Real s2;
          TransformToSRHyd(u,glower,gupper,s2,u_sr);
          bool c2p_failure=false;
          int iter_used=0;
          SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                               dfloor_used, efloor_used, c2p_failure, iter_used);

          // apply velocity ceiling if necessary
          Real tmp = glower[1][1]*SQR(w.vx)
                   + glower[2][2]*SQR(w.vy)
                   + glower[3][3]*SQR(w.vz)
                   + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
                   + 2.0*glower[2][3]*w.vy*w.vz;
          Real lor = sqrt(1.0+tmp);
          if (lor > eos.gamma_max) {
            Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
            w.vx *= factor;
            w.vy *= factor;
            w.vz *= factor;
          }
        } else if (is_sr) {
          // Compute (S^i S_i) (eqn C2)
          Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
          bool c2p_failure=false;
          int iter_used=0;
          SingleC2P_IdealSRHyd(u, eos, s2, w,
                               dfloor_used, efloor_used, c2p_failure, iter_used);
          // apply velocity ceiling if necessary
          Real lor = sqrt(1.0+SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
          if (lor > eos.gamma_max) {
            Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
            w.vx *= factor;
            w.vy *= factor;
            w.vz *= factor;
          }
        } else {
          if (!use_dual) {
            if (eos.UsesTabulatedLTE()) {
              eos_general::SingleC2P_GeneralHyd(u, eos, w, dfloor_used, efloor_used,
                                                tfloor_used, vceiling_used);
            } else {
              SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                                 vceiling_used);
            }
          } else {
            eos_general::SingleC2P_GeneralHydDual(
                u, eos, cons(m, dual_idx, k, j, i), dual_eta1, w, dual_prim_e,
                dfloor_used, efloor_used, tfloor_used, vceiling_used);
          }
        }

        // No need to correct conserved state in coarse boundary arrays if floors used
        // since these values will be overwritten after prolongation anyways.
        // store primitive state in 3D array
        prim(m,IDN,k,j,i) = w.d;
        prim(m,IVX,k,j,i) = w.vx;
        prim(m,IVY,k,j,i) = w.vy;
        prim(m,IVZ,k,j,i) = w.vz;
        prim(m,IEN,k,j,i) = w.e;
        // convert scalars (if any)
        for (int n=nhyd; n<(nhyd+nscal); ++n) {
          // apply scalar floor
          if (cons(m,n,k,j,i) < 0.0) {
            cons(m,n,k,j,i) = 0.0;
          }
          prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
        }
        if (use_dual) {
          prim(m,dual_idx,k,j,i) = dual_prim_e;
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToConsFineBndry()
//! \brief Converts prolongated Hydro primitive variables at fine level in boundary
//! buffers into Hydro conservative variables.
//! Note same function for MHD has different argument list.

void MeshBoundaryValuesCC::PrimToConsFineBndry(const DvceArray5D<Real> &prim,
                                               DvceArray5D<Real> &cons) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  auto rbuf_device = recvbuf_device;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  Real &gamma = pmy_pack->phydro->peos->eos_data.gamma;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  bool use_dual = pmy_pack->phydro->dual_energy_pdv;
  int dual_idx = pmy_pack->phydro->dual_energy_idx;

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", athenak_lw(policy), KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/nnghbr;
    const int m = lat_enabled ? lat_active_indices(a) : a;
    const int n = tmember.league_rank() - a*nnghbr;

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // loop over indices for prolongation on this buffer
      // Convert indices from coarse to fine arrays
      int il = (rbuf_device(n).iprol[0].bis - indcs.cis)*2 + indcs.is;
      int iu = (rbuf_device(n).iprol[0].bie - indcs.cis)*2 + indcs.is + 1;
      int jl = (rbuf_device(n).iprol[0].bjs - indcs.cjs)*2 + indcs.js;
      int ju = (rbuf_device(n).iprol[0].bje - indcs.cjs)*2 + indcs.js;
      if (multi_d) {
        ju += 1;
      }
      int kl = (rbuf_device(n).iprol[0].bks - indcs.cks)*2 + indcs.ks;
      int ku = (rbuf_device(n).iprol[0].bke - indcs.cks)*2 + indcs.ks;
      if (three_d) {
        ku += 1;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // Load single state of primitive variables
        HydPrim1D w;
        w.d  = prim(m,IDN,k,j,i);
        w.vx = prim(m,IVX,k,j,i);
        w.vy = prim(m,IVY,k,j,i);
        w.vz = prim(m,IVZ,k,j,i);
        w.e  = prim(m,IEN,k,j,i);
        if (!is_sr && !is_gr && eos.use_e) {
          bool efloor_used = false, tfloor_used = false;
          w.e = eos_general::ApplyHydroThermalFloors(eos, w.d, w.e, efloor_used,
                                                     tfloor_used);
        }
        HydCons1D u;

        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-indcs.is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-indcs.js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-indcs.ks, indcs.nx3, x3min, x3max);

          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
          SingleP2C_IdealGRHyd(glower, gupper, w, gamma, u);
        } else if (is_sr) {
          SingleP2C_IdealSRHyd(w, gamma, u);
        } else {
          if (eos.UsesTabulatedLTE()) {
            eos_general::SingleP2C_GeneralHyd(w, u);
          } else {
            SingleP2C_IdealHyd(w, u);
          }
        }

        // Set conserved quantities
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;

        // convert scalars (if any)
        for (int n=nhyd; n<(nhyd+nscal); ++n) {
          cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
        }
        if (use_dual) {
          Real eint_aux = prim(m,dual_idx,k,j,i);
          if (!is_sr && !is_gr && eos.use_e) {
            bool efloor_used = false, tfloor_used = false;
            eint_aux = eos_general::ApplyHydroThermalFloors(eos, w.d, eint_aux,
                                                            efloor_used, tfloor_used);
          }
          cons(m,dual_idx,k,j,i) = eint_aux;
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrimCoarseBndry()
//! \brief Converts MHD conserved variables in coarse level boundary buffers into
//! MHD primitive variables for prolongation. Coarse arrays should be passed in through
//! arguments.
//! Only works for MHD, the same function for hydro has different argument list.

void MeshBoundaryValuesCC::ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons,
                                 const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int nwork = nmb;
  if (nwork <= 0) return;
  auto nghbr_device = nghbr.d_view;
  auto mblev_device = mblev.d_view;
  auto rbuf_device = recvbuf_device;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  // dual_energy_pdv: the GR adiabat must cross a coarse/fine boundary through the
  // ordinary passive-scalar path, which recovers kappa as cons/D.  Skipping it here
  // published kappa = 0 in exactly the worst-conditioned cells.
  const bool use_dual = pmy_pack->pmhd->dual_energy_pdv;
  const int dual_idx = pmy_pack->pmhd->dual_energy_idx;
  const Real dual_eta1 = pmy_pack->pmhd->dual_energy_eta1;

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", athenak_lw(policy), KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/nnghbr;
    const int m = a;
    const int n = tmember.league_rank() - a*nnghbr;

    // only convert coarse vars when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // use indices for prolongation on this buffer as loop limits
      // Note that one extra cell is added to match stencil of 2nd-order prolongation
      int il = rbuf_device(n).iprol[0].bis - 1;
      int iu = rbuf_device(n).iprol[0].bie + 1;
      int jl = rbuf_device(n).iprol[0].bjs;
      int ju = rbuf_device(n).iprol[0].bje;
      if (multi_d) {
        jl -= 1;
        ju += 1;
      }
      int kl = rbuf_device(n).iprol[0].bks;
      int ku = rbuf_device(n).iprol[0].bke;
      if (three_d) {
        kl -= 1;
        ku += 1;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // Expanded face/edge/corner stencil boxes overlap.  Give every coarse cell to
        // exactly one neighbor team so C2P and scalar repair cannot race on prim/cons.
        if (!OwnsCoarsePrimitiveStencilCell(
                nghbr_device, mblev_device, rbuf_device, multi_d, three_d,
                m, n, k, j, i)) return;

        // load single state conserved variables
        MHDCons1D u;
        u.d  = cons(m,IDN,k,j,i);
        u.mx = cons(m,IM1,k,j,i);
        u.my = cons(m,IM2,k,j,i);
        u.mz = cons(m,IM3,k,j,i);
        u.e  = cons(m,IEN,k,j,i);
        Real scalar_cons_density = u.d;
        // use simple linear average of face-centered fields
        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
        HydPrim1D w;
        Real dual_prim_e = 0.0;

        bool dfloor_used=false, efloor_used=false, tfloor_used=false;
        bool vceiling_used=false;
        bool sigceiling_used=false;
        bool c2p_failure=false;
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          // Note indices refer to coarse arrays, so use cis, cnx1
          Real x1v = CellCenterX(i-indcs.cis, indcs.cnx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-indcs.cjs, indcs.cnx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-indcs.cks, indcs.cnx3, x3min, x3max);

          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

          EOS_Data cell_eos = eos;
          MHDCons1D u_sr;
          Real s2,b2,rpar;
          TransformToSRMHD(u,glower,gupper,s2,b2,rpar,u_sr);
          int iter_used=0;
          SingleC2P_IdealSRMHD(u_sr, cell_eos, s2, b2, rpar, w,
                               dfloor_used, efloor_used, c2p_failure, iter_used);

          // Coarse primitives feed every fine-boundary interpolation stencil.  Use the
          // overflow-safe KORAL ceiling here so no AMR receiver can inherit an
          // uncapped or nonfinite wall direction from its coarse GRMHD state.
          const auto lorentz_limit =
              mhd::forcefree::ApplyIsotropicLorentzLimit(
                  glower, cell_eos.gamma_max, w.vx, w.vy, w.vz);
          if (!lorentz_limit.valid) {
            w.vx = 0.0;
            w.vy = 0.0;
            w.vz = 0.0;
            vceiling_used = true;
          } else if (lorentz_limit.limited) {
            vceiling_used = true;
          }
          sigceiling_used = ApplySigmaCeiling_IdealGRMHD(cell_eos, glower, gupper, u, w);
          Real qv = glower[1][1]*SQR(w.vx)
                  + glower[2][2]*SQR(w.vy)
                  + glower[3][3]*SQR(w.vz)
                  + 2.0*glower[1][2]*w.vx*w.vy
                  + 2.0*glower[1][3]*w.vx*w.vz
                  + 2.0*glower[2][3]*w.vy*w.vz;
          Real lor = (isfinite(qv) && qv >= 0.0) ? sqrt(1.0 + qv) : 0.0;
          Real alpha = sqrt(-1.0/gupper[0][0]);
          if (scalar_cons_density > 0.0 && alpha > 0.0 && lor > 0.0) {
            u.d = w.d*lor/alpha;
          }
        } else if (is_sr) {
          // Compute (S^i S_i) (eqn C2)
          Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
          Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
          Real rpar = (u.bx*u.mx +  u.by*u.my +  u.bz*u.mz)/u.d;
          int iter_used=0;
          SingleC2P_IdealSRMHD(u, eos, s2, b2, rpar, w,
                               dfloor_used, efloor_used, c2p_failure, iter_used);
          // apply velocity ceiling if necessary
          Real lor = sqrt(1.0+SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
          if (lor > eos.gamma_max) {
            vceiling_used = true;
            Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
            w.vx *= factor;
            w.vy *= factor;
            w.vz *= factor;
          }
        } else {
          if (!use_dual) {
            if (eos.UsesTabulatedLTE()) {
              eos_general::SingleC2P_GeneralMHD(u, eos, w, dfloor_used, efloor_used,
                                                tfloor_used, vceiling_used);
            } else {
              SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                                 vceiling_used);
            }
          } else {
            bool eint_from_aux = false;  // primitives only; the coarse E is not kept
            eos_general::SingleC2P_GeneralMHDDual(
                u, eos, cons(m, dual_idx, k, j, i), dual_eta1, w, dual_prim_e,
                eint_from_aux, dfloor_used, efloor_used, tfloor_used, vceiling_used);
          }
        }

        // No need to correct conserved state in coarse boundary arrays if floors used
        // since these values will be overwritten after prolongation anyways.
        // store primitive state in 3D array
        prim(m,IDN,k,j,i) = w.d;
        prim(m,IVX,k,j,i) = w.vx;
        prim(m,IVY,k,j,i) = w.vy;
        prim(m,IVZ,k,j,i) = w.vz;
        prim(m,IEN,k,j,i) = w.e;
        // No need to store cell-centered fields since they will not be prolongated
        const bool preserve_scalars_across_kinematic_repair = is_gr &&
            vceiling_used && !dfloor_used && !efloor_used &&
            !sigceiling_used && !c2p_failure;
        // convert scalars (if any)
        for (int n=nmhd; n<(nmhd+nscal); ++n) {
          // apply scalar floor
          if (!isfinite(cons(m,n,k,j,i)) || cons(m,n,k,j,i) < 0.0) {
            cons(m,n,k,j,i) = 0.0;
          }
          const Real scalar_den =
              preserve_scalars_across_kinematic_repair ? scalar_cons_density :
              ((u.d > 0.0) ? u.d : w.d);
          Real scalar = (scalar_den > 0.0) ? cons(m,n,k,j,i)/scalar_den : 0.0;
          prim(m,n,k,j,i) = isfinite(scalar) ? scalar : 0.0;
        }
        if (use_dual) {
          prim(m,dual_idx,k,j,i) = dual_prim_e;
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToConsFineBndry()
//! \brief Converts prolongated MHD primitive variables at fine level in boundary buffers
//! into MHD conservative variables.
//! Note same function for Hydrodynamics has different argument list.

void MeshBoundaryValuesCC::PrimToConsFineBndry(DvceArray5D<Real> &prim,
                               const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &cons) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  const int nwork = nmb;
  if (nwork <= 0) return;
  auto rbuf_device = recvbuf_device;
  auto &indcs  = pmy_pack->pmesh->mb_indcs;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = pmy_pack->pmhd->peos->eos_data;
  Real &gamma = pmy_pack->pmhd->peos->eos_data.gamma;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  bool use_dual = pmy_pack->pmhd->dual_energy_pdv;
  int dual_idx = pmy_pack->pmhd->dual_energy_idx;

  // Outer loop over (# of MeshBlocks)*(# of buffers)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("ProlCC", athenak_lw(policy), KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/nnghbr;
    const int m = a;
    const int n = tmember.league_rank() - a*nnghbr;

    // only prolongate when neighbor exists and is at coarser level
    if ((nghbr.d_view(m,n).gid >= 0) && (nghbr.d_view(m,n).lev < mblev.d_view(m))) {
      // loop over indices for prolongation on this buffer
      // Convert indices from coarse to fine arrays
      int il = (rbuf_device(n).iprol[0].bis - indcs.cis)*2 + indcs.is;
      int iu = (rbuf_device(n).iprol[0].bie - indcs.cis)*2 + indcs.is + 1;
      int jl = (rbuf_device(n).iprol[0].bjs - indcs.cjs)*2 + indcs.js;
      int ju = (rbuf_device(n).iprol[0].bje - indcs.cjs)*2 + indcs.js;
      if (multi_d) {
        ju += 1;
      }
      int kl = (rbuf_device(n).iprol[0].bks - indcs.cks)*2 + indcs.ks;
      int ku = (rbuf_device(n).iprol[0].bke - indcs.cks)*2 + indcs.ks;
      if (three_d) {
        ku += 1;
      }
      const int ni = iu - il + 1;
      const int nj = ju - jl + 1;
      const int nk = ku - kl + 1;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = idx/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // Load single state of primitive variables
        MHDPrim1D w;
        w.d  = prim(m,IDN,k,j,i);
        w.vx = prim(m,IVX,k,j,i);
        w.vy = prim(m,IVY,k,j,i);
        w.vz = prim(m,IVZ,k,j,i);
        w.e  = prim(m,IEN,k,j,i);
        // use simple linear average of face-centered fields
        w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
        if (!is_sr && !is_gr && eos.use_e) {
          bool efloor_used = false, tfloor_used = false;
          w.e = eos_general::ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used,
                                                   tfloor_used);
        }
        HydCons1D u;

        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-indcs.is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-indcs.js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-indcs.ks, indcs.nx3, x3min, x3max);

          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

          EOS_Data cell_eos = eos;
          if (!isfinite(w.d) || w.d < cell_eos.dfloor) {
            w.d = cell_eos.dfloor;
          }
          Real eint_floor = IdealMHDEintFloor(cell_eos, w.d);
          if (!isfinite(w.e) || w.e < eint_floor) {
            w.e = eint_floor;
          }
          Real eint_ceiling = cell_eos.HydroInternalEnergyDensityCeiling(w.d);
          if (isfinite(eint_ceiling) && w.e > eint_ceiling) {
            w.e = eint_ceiling;
          }

          const auto lorentz_limit =
              mhd::forcefree::ApplyIsotropicLorentzLimit(
                  glower, cell_eos.gamma_max, w.vx, w.vy, w.vz);
          if (!lorentz_limit.valid) {
            w.vx = 0.0;
            w.vy = 0.0;
            w.vz = 0.0;
          }

          MHDCons1D u_for_limit;
          u_for_limit.bx = w.bx;
          u_for_limit.by = w.by;
          u_for_limit.bz = w.bz;
          HydPrim1D wh;
          wh.d = w.d;
          wh.vx = w.vx;
          wh.vy = w.vy;
          wh.vz = w.vz;
          wh.e = w.e;
          ApplySigmaCeiling_IdealGRMHD(cell_eos, glower, gupper, u_for_limit, wh);
          w.d = wh.d;
          w.vx = wh.vx;
          w.vy = wh.vy;
          w.vz = wh.vz;
          w.e = wh.e;

          Real local_gamma = gamma;

          prim(m,IDN,k,j,i) = w.d;
          prim(m,IVX,k,j,i) = w.vx;
          prim(m,IVY,k,j,i) = w.vy;
          prim(m,IVZ,k,j,i) = w.vz;
          prim(m,IEN,k,j,i) = w.e;
          SingleP2C_IdealGRMHD(glower, gupper, w, local_gamma, u);
        } else if (is_sr) {
          SingleP2C_IdealSRMHD(w, gamma, u);
        } else {
          // Non-gamma EOS AMR boundaries must rebuild conserved fields through the active
          // EOS rather than the gamma-law primitive-to-conserved helper.
          if (eos.UsesTabulatedLTE()) {
            eos_general::SingleP2C_GeneralMHD(w, u);
          } else {
            SingleP2C_IdealMHD(w, u);
          }
        }

        // Set conserved quantities
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;

        // convert scalars (if any)
        for (int n=nmhd; n<(nmhd+nscal); ++n) {
          Real scalar = prim(m,n,k,j,i);
          scalar = (isfinite(scalar) && scalar > 0.0) ? scalar : 0.0;
          cons(m,n,k,j,i) = u.d*scalar;
        }
        if (use_dual) {
          Real eint_aux = prim(m,dual_idx,k,j,i);
          if (!is_sr && !is_gr && eos.use_e) {
            bool efloor_used = false, tfloor_used = false;
            eint_aux = eos_general::ApplyMHDThermalFloors(eos, w.d, eint_aux,
                                                          efloor_used, tfloor_used);
          }
          cons(m,dual_idx,k,j,i) = eint_aux;
        }
      });
    }
    tmember.team_barrier();
  });
  return;
}
