//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid.cpp
//! \brief implementation of the functions commonly used in Multigrid

// C headers

// C++ headers
#include <algorithm>
#include <cstdint>
#include <cstdlib>    // exit
#include <cmath>
#include <cassert>    // assert
#include <cstring>    // memset, memcpy
#include <iostream>
#include <limits>     // numeric_limits
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <unordered_map>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mb_storage.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/nghbr_index.hpp"
#include "../parameter_input.hpp"
#include "../utils/env_switch.hpp"
#include "multigrid.hpp"
#include "utils/gravity_weight.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
int MGBoundsCellCount(int il, int iu, int jl, int ju, int kl, int ku) {
  return (iu - il + 1) * (ju - jl + 1) * (ku - kl + 1);
}

//! Granularity of the flat remote fine/coarse payload buffers: one MiB of Reals.
//! Rounding up costs a fraction of a MiB and keeps a change in the remote entry count
//! from moving .data(), which would invalidate every persistent MPI request bound to it.
constexpr std::int64_t kFCRemoteBufGrainReals =
    static_cast<std::int64_t>(1024*1024)/static_cast<std::int64_t>(sizeof(Real));

//----------------------------------------------------------------------------------------
//! \fn std::int64_t FCRemoteBufCapacity(std::int64_t needed)
//! \brief number of Reals to allocate in order to hold `needed` of them
//!
//! Zero stays zero: a rank with no remote fine/coarse partners must not hold a buffer.

inline std::int64_t FCRemoteBufCapacity(std::int64_t needed) {
  if (needed <= 0) return 0;
  return ((needed + kFCRemoteBufGrainReals - 1)/kFCRemoteBufGrainReals)
         * kFCRemoteBufGrainReals;
}

// Neighbor slots are seven contiguous groups of eight: x1, x2, x1x2, x3,
// x3x1, x2x3, and x1x2x3.  The group number plus one is therefore an axis bit mask.
KOKKOS_INLINE_FUNCTION
int MGNeighborFaceMask(const int n) {
  return (n >> 3) + 1;
}

KOKKOS_INLINE_FUNCTION
void AdjustMGSameSendFaceBounds(int ngh, int shift, int nx,
                                bool fx, bool fy, bool fz, int depth,
                                int &il, int &iu, int &jl, int &ju,
                                int &kl, int &ku) {
  int sh = shift;
  int nx_l = nx;
  while (sh > 0) {
    if (fx && (il == nx_l)) {
      int d = iu - il;
      il = il >> 1;
      iu = il + d;
    } else if (!fx) {
      iu = ((iu - il) >> 1) + il;
    }
    if (fy && (jl == nx_l)) {
      int d = ju - jl;
      jl = jl >> 1;
      ju = jl + d;
    } else if (!fy) {
      ju = ((ju - jl) >> 1) + jl;
    }
    if (fz && (kl == nx_l)) {
      int d = ku - kl;
      kl = kl >> 1;
      ku = kl + d;
    } else if (!fz) {
      ku = ((ku - kl) >> 1) + kl;
    }
    --sh;
    nx_l = nx_l >> 1;
  }

  // MG smoothing/defect uses a 7-point stencil, so a single same-level face halo layer
  // is sufficient even if the generic hydro boundary buffer is wider.  The exact
  // communication-avoiding pair (MultigridDriver::OneStepToCoarser) reads one layer
  // further out and asks for depth 2 instead.
  if (fx && il < iu) {
    if (il <= ngh) { if (iu > il + depth - 1) iu = il + depth - 1; }
    else           { if (il < iu - depth + 1) il = iu - depth + 1; }
  }
  if (fy && jl < ju) {
    if (jl <= ngh) { if (ju > jl + depth - 1) ju = jl + depth - 1; }
    else           { if (jl < ju - depth + 1) jl = ju - depth + 1; }
  }
  if (fz && kl < ku) {
    if (kl <= ngh) { if (ku > kl + depth - 1) ku = kl + depth - 1; }
    else           { if (kl < ku - depth + 1) kl = ku - depth + 1; }
  }
}

KOKKOS_INLINE_FUNCTION
void AdjustMGSameRecvFaceBounds(int ngh, int shift,
                                bool fx, bool fy, bool fz, int depth,
                                int &il, int &iu, int &jl, int &ju,
                                int &kl, int &ku) {
  int sh = shift;
  while (sh > 0) {
    if (fx && il > 1) {
      int d = iu - il;
      il = (il + ngh) >> 1;
      iu = il + d;
    } else if (!fx) {
      iu = ((iu - il) >> 1) + il;
    }
    if (fy && jl > 1) {
      int d = ju - jl;
      jl = (jl + ngh) >> 1;
      ju = jl + d;
    } else if (!fy) {
      ju = ((ju - jl) >> 1) + jl;
    }
    if (fz && kl > 1) {
      int d = ku - kl;
      kl = (kl + ngh) >> 1;
      ku = kl + d;
    } else if (!fz) {
      ku = ((ku - kl) >> 1) + kl;
    }
    --sh;
  }

  if (fx && il < iu) {
    if (il < ngh) { if (il < iu - depth + 1) il = iu - depth + 1; }
    else          { if (iu > il + depth - 1) iu = il + depth - 1; }
  }
  if (fy && jl < ju) {
    if (jl < ngh) { if (jl < ju - depth + 1) jl = ju - depth + 1; }
    else          { if (ju > jl + depth - 1) ju = jl + depth - 1; }
  }
  if (fz && kl < ku) {
    if (kl < ngh) { if (kl < ku - depth + 1) kl = ku - depth + 1; }
    else          { if (ku > kl + depth - 1) ku = kl + depth - 1; }
  }
}

}  // namespace

//namespace multigrid{ // NOLINT (build/namespace)
//----------------------------------------------------------------------------------------
//! \fn Multigrid::Multigrid(MultigridDriver *pmd, MeshBlock *pmb, int nghost)
//  \brief Multigrid constructor

Multigrid::Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host):
  pmy_driver_(pmd), pmy_pack_(pmbp), pmy_mesh_(pmd->pmy_mesh_), ngh_(nghost),
  nvar_(pmd->nvar_), defscale_(1.0), on_host_(on_host)  {
  if(pmy_pack_ != nullptr) {
    //Meshblock levels
    indcs_ = pmy_mesh_->mb_indcs;
    nmmb_  = pmy_pack_->nmb_thispack;
    if (global_variable::my_rank == 0) {
      std::cout << "Number of MeshBlocks in the pack: " << nmmb_ << std::endl;
      std::cout << "MeshBlock size: "
                << indcs_.nx1 << " x " << indcs_.nx2 << " x " << indcs_.nx3
                << std::endl;
    }
    if (indcs_.nx1 != indcs_.nx2 || indcs_.nx1 != indcs_.nx3) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
         << "The Multigrid solver requires logically cubic MeshBlock." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
     }

     // initialize loc/size from the first meshblock in the pack (needs to be addpated for
     // AMR)
    size_ = pmy_pack_->pmb->mb_size.h_view(0);
  } else {
    //Root levels
    indcs_.nx1 = pmy_mesh_->nmb_rootx1;
    indcs_.nx2 = pmy_mesh_->nmb_rootx2;
    indcs_.nx3 = pmy_mesh_->nmb_rootx3;
    size_ = pmy_mesh_->mesh_size;
    // Root grid should be a single meshblock
    nmmb_ = 1;
  }

  rdx_ = (size_.x1max-size_.x1min)/static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max-size_.x2min)/static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max-size_.x3min)/static_cast<Real>(indcs_.nx3);
  dvol_over_dx3_ = (rdy_ / rdx_) * (rdz_ / rdx_);

  // AMR moves the local MeshBlock count on essentially every cycle.  Size the block
  // dimension of every per-block array to a rounded-up CAPACITY (see mb_storage.hpp) so
  // that the level arrays are not reallocated each time the count moves by a block, which
  // is what fragments the device heap.  nmmb_ stays the live count that bounds every
  // kernel.  The root grid is a single block that never resizes, so it stays exact.
  nmmb_capacity_ = (pmy_pack_ != nullptr) ? MeshBlockStorageCapacity(nmmb_) : nmmb_;

  block_rdx_ = DualArray1D<Real>("block_rdx", nmmb_capacity_);
  block_color_parity_ = DualArray1D<int>("block_color_parity", nmmb_capacity_);
  Kokkos::realloc(fc_childx_, nmmb_capacity_);
  Kokkos::realloc(fc_childy_, nmmb_capacity_);
  Kokkos::realloc(fc_childz_, nmmb_capacity_);
  {
    auto brdx_h = block_rdx_.h_view;
    auto parity_h = block_color_parity_.h_view;
    if (pmy_pack_ != nullptr) {
      auto &mb_size = pmy_pack_->pmb->mb_size;
      auto &mb_gid = pmy_pack_->pmb->mb_gid;
      auto *lloc = pmy_mesh_->lloc_eachmb;
      Real rnx1 = static_cast<Real>(indcs_.nx1);
      for (int m = 0; m < nmmb_; ++m) {
        brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
        const LogicalLocation &mloc = lloc[mb_gid.h_view(m)];
        parity_h(m) = (static_cast<int>(mloc.lx1) + static_cast<int>(mloc.lx2)
                       + static_cast<int>(mloc.lx3)) & 1;
      }
    } else {
      brdx_h(0) = rdx_;
      parity_h(0) = 0;
    }
    Kokkos::deep_copy(block_rdx_.d_view, brdx_h);
    Kokkos::deep_copy(block_color_parity_.d_view, parity_h);
  }
  if (pmy_pack_ != nullptr) {
    UpdateBlockDx();
  }

  nlevel_ = 0;
  if (pmy_pack_ == nullptr) {
    // Root grid levels
    int nbx = 0, nby = 0, nbz = 0;
    for (int l = 0; l < 20; l++) {
      if (indcs_.nx1%(1<<l) == 0 && indcs_.nx2%(1<<l) == 0 && indcs_.nx3%(1<<l) == 0) {
        nbx = indcs_.nx1/(1<<l), nby = indcs_.nx2/(1<<l), nbz = indcs_.nx3/(1<<l);
        nlevel_ = l+1;
      }
    }
    int nmaxr = std::max(nbx, std::max(nby, nbz));
    if (global_variable::my_rank == 0) {
      std::cout << "Multigrid root grid levels: " << nlevel_ << std::endl;
    }
    // int nminr=std::min(nbx, std::min(nby, nbz)); // unused variable
    if (nmaxr != 1 && global_variable::my_rank == 0) {
      std::cout
          << "### Warning in Multigrid::Multigrid" << std::endl
          << "The root grid can not be reduced to a single cell." << std::endl
          << "Multigrid should still work, but this is not the"
          << " most efficient configuration"
          << " as the coarsest level is not solved exactly but iteratively." << std::endl;
    }
    if (nbx*nby*nbz>100 && global_variable::my_rank==0) {
      std::cout << "### Warning in Multigrid::Multigrid" << std::endl
                << "The degrees of freedom on the coarsest level is very large: "
                << nbx << " x " << nby << " x " << nbz << " = " << nbx*nby*nbz<< std::endl
                << "Multigrid should still work, but this is not efficient configuration "
                << "as the coarsest level solver costs considerably." << std::endl
                << "We recommend to reconsider grid configuration." << std::endl;
    }
  } else {
    // MeshBlock levels
    for (int l = 0; l < 20; l++) {
      if ((1<<l) == indcs_.nx1) {
        nlevel_=l+1;
        break;
      }
    }
    if (nlevel_ == 0) {
      std::cout << "### FATAL ERROR in Multigrid::Multigrid" << std::endl
          << "The MeshBlock size must be power of two." << std::endl;
      std::exit(EXIT_FAILURE);
      return;
    }
  }

  current_level_ = nlevel_-1;

  // allocate arrays
  u_ = new DualArray5D<Real>[nlevel_];
  src_ = new DualArray5D<Real>[nlevel_];
  def_ = new DualArray5D<Real>[nlevel_];
  uold_ = new DualArray5D<Real>[nlevel_];

  for (int l = 0; l < nlevel_; l++) {
    int ll=nlevel_-1-l;
    int ncx=(indcs_.nx1>>ll)+2*ngh_;
    int ncy=(indcs_.nx2>>ll)+2*ngh_;
    int ncz=(indcs_.nx3>>ll)+2*ngh_;
    // Built with a label rather than realloc'd from a default-constructed View: these are
    // the largest allocations in the solver and an anonymous one reports its failures
    // without a name.  Kokkos::realloc keeps the label on the AMR path below.
    u_[l]   = DualArray5D<Real>("mg_u",   nmmb_capacity_, nvar_, ncz, ncy, ncx);
    src_[l] = DualArray5D<Real>("mg_src", nmmb_capacity_, nvar_, ncz, ncy, ncx);
    def_[l] = DualArray5D<Real>("mg_def", nmmb_capacity_, nvar_, ncz, ncy, ncx);

    if (!((pmy_pack_ != nullptr) && (l == nlevel_-1)))
      uold_[l] = DualArray5D<Real>("mg_uold", nmmb_capacity_, nvar_, ncz, ncy, ncx);

    ncx=(indcs_.nx1>>(ll+1))+2*ngh_;
    ncy=(indcs_.nx2>>(ll+1))+2*ngh_;
    ncz=(indcs_.nx3>>(ll+1))+2*ngh_;
  }
}


//----------------------------------------------------------------------------------------
//! \fn Multigrid::~Multigrid
//! \brief Multigrid destroctor

Multigrid::~Multigrid() {
  delete pbval;
  delete [] u_;
  delete [] src_;
  delete [] def_;
  delete [] uold_;
}



//----------------------------------------------------------------------------------------
//! \fn void Multigrid::UpdateBlockDx()
//! \brief Refresh per-block spacing and child-octant indices used by FC ghost fills

void Multigrid::UpdateBlockDx() {
  if (pmy_pack_ == nullptr) return;
  auto &mb_size = pmy_pack_->pmb->mb_size;
  Real rnx1 = static_cast<Real>(indcs_.nx1);

  size_ = mb_size.h_view(0);
  rdx_ = (size_.x1max - size_.x1min) / static_cast<Real>(indcs_.nx1);
  rdy_ = (size_.x2max - size_.x2min) / static_cast<Real>(indcs_.nx2);
  rdz_ = (size_.x3max - size_.x3min) / static_cast<Real>(indcs_.nx3);
  dvol_over_dx3_ = (rdy_ / rdx_) * (rdz_ / rdx_);

  auto brdx_h = block_rdx_.h_view;
  auto parity_h = block_color_parity_.h_view;
  auto &mbgid = pmy_pack_->pmb->mb_gid;
  auto *lloc = pmy_mesh_->lloc_eachmb;
  for (int m = 0; m < nmmb_; ++m) {
    brdx_h(m) = (mb_size.h_view(m).x1max - mb_size.h_view(m).x1min) / rnx1;
    const LogicalLocation &loc = lloc[mbgid.h_view(m)];
    parity_h(m) = (static_cast<int>(loc.lx1) + static_cast<int>(loc.lx2)
                   + static_cast<int>(loc.lx3)) & 1;
  }
  Kokkos::deep_copy(block_rdx_.d_view, brdx_h);
  Kokkos::deep_copy(block_color_parity_.d_view, parity_h);

  int root_level = pmy_mesh_->root_level;
  auto cx_h = Kokkos::create_mirror_view(fc_childx_);
  auto cy_h = Kokkos::create_mirror_view(fc_childy_);
  auto cz_h = Kokkos::create_mirror_view(fc_childz_);
  for (int m = 0; m < nmmb_; ++m) {
    int gid = mbgid.h_view(m);
    const LogicalLocation &loc = lloc[gid];
    cx_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx1 & 1) : 0;
    cy_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx2 & 1) : 0;
    cz_h(m) = (loc.level > root_level) ? static_cast<int>(loc.lx3 & 1) : 0;
  }
  Kokkos::deep_copy(fc_childx_, cx_h);
  Kokkos::deep_copy(fc_childy_, cy_h);
  Kokkos::deep_copy(fc_childz_, cz_h);
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ReallocateForAMR()
//! \brief Reallocate MG arrays when the number of MeshBlocks changes (AMR)

void Multigrid::ReallocateForAMR() {
  if (pmy_pack_ == nullptr) return;
  const int new_nmmb = pmy_pack_->nmb_thispack;
  const bool count_changed = (new_nmmb != nmmb_);
  // Shrinking is safe HERE, and only here.  ReallocateForAMR runs from PrepareForAMR,
  // i.e. after the topology transaction has finished placing every block: MeshRefinement
  // reaches it through the post-AMR ghost refresh, after the physics arrays have
  // themselves been allowed to shrink.  The multigrid arrays carry nothing across that
  // transaction -- every solve reloads them from phi and from the conserved variables --
  // so no consumer can read them at a stale local id.
  const bool reallocate = MeshBlockStorageShouldResize(nmmb_capacity_, new_nmmb, true);
  nmmb_ = new_nmmb;
  if (reallocate) {
    nmmb_capacity_ = MeshBlockStorageCapacity(new_nmmb);
    Kokkos::realloc(block_rdx_, nmmb_capacity_);
    Kokkos::realloc(block_color_parity_, nmmb_capacity_);
    Kokkos::realloc(fc_childx_, nmmb_capacity_);
    Kokkos::realloc(fc_childy_, nmmb_capacity_);
    Kokkos::realloc(fc_childz_, nmmb_capacity_);
  }

  UpdateBlockDx();

  if (!count_changed && !reallocate) return;

  for (int l = 0; l < nlevel_; l++) {
    int ll = nlevel_ - 1 - l;
    int ncx = (indcs_.nx1 >> ll) + 2 * ngh_;
    int ncy = (indcs_.nx2 >> ll) + 2 * ngh_;
    int ncz = (indcs_.nx3 >> ll) + 2 * ngh_;
    const bool has_uold = (l != nlevel_ - 1);
    if (reallocate) {
      Kokkos::realloc(u_[l],   nmmb_capacity_, nvar_, ncz, ncy, ncx);
      Kokkos::realloc(src_[l], nmmb_capacity_, nvar_, ncz, ncy, ncx);
      Kokkos::realloc(def_[l], nmmb_capacity_, nvar_, ncz, ncy, ncx);
      if (has_uold)
        Kokkos::realloc(uold_[l], nmmb_capacity_, nvar_, ncz, ncy, ncx);
    } else {
      // The allocation is kept, so hand the solver the same zeroed arrays a fresh
      // Kokkos::realloc used to: the previous policy reallocated on every change of the
      // live count, and a reallocated View is value-initialized.  Only the side the
      // solver reads is cleared, and only over the live blocks -- every kernel here is
      // bounded by nmmb_, so the storage grain past it holds nothing anyone can read.
      // The host mirrors of the block levels need no clear: mglevels_ is never on_host_,
      // and the only host views any code touches are u_[0]/uold_[0], which
      // SetFromRootGrid and TransferFromRootToBlocksDistributed overwrite cell-for-cell
      // over [0, nmmb_) before syncing them to the device.
      if (on_host_) {
        Kokkos::deep_copy(LiveBlocks(u_[l].h_view),   0.0);
        Kokkos::deep_copy(LiveBlocks(src_[l].h_view), 0.0);
        Kokkos::deep_copy(LiveBlocks(def_[l].h_view), 0.0);
        if (has_uold) Kokkos::deep_copy(LiveBlocks(uold_[l].h_view), 0.0);
      } else {
        Kokkos::deep_copy(LiveBlocks(u_[l].d_view),   0.0);
        Kokkos::deep_copy(LiveBlocks(src_[l].d_view), 0.0);
        Kokkos::deep_copy(LiveBlocks(def_[l].d_view), 0.0);
        if (has_uold) Kokkos::deep_copy(LiveBlocks(uold_[l].d_view), 0.0);
      }
    }
  }
}


//! \fn void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh)
//! \brief Fill the inital guess in the active zone of the finest level

void Multigrid::LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh) {
  auto &dst = u_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1; je = js + indcs_.nx2 - 1; ke = ks + indcs_.nx3 - 1;

  const int lns = ns;
  const int lks = ks, ljs = js, lis = is, lngh = ngh;

  par_for("Multigrid::LoadFinestData", DevExeSpace(),0, nmmb_-1,
          0, nvar_-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = lns + v;
    const int k = mk - lks + lngh;
    const int j = mj - ljs + lngh;
    const int i = mi - lis + lngh;
    dst(m, v, mk, mj, mi) = src(m, nsrc, k, j, i);
  });

  // A new interior invalidates whatever the last solve left in the finest halo, which is
  // what the driver skips an exchange on when it is still good.
  pmy_driver_->finest_bvals_fresh_ = false;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh,
//!                                Real fac, Real src_min)
//! \brief Fill the source in the active zone of the finest level

void Multigrid::LoadSource(const DvceArray5D<Real> &src, int ns, int ngh, Real fac,
                           Real src_min, Real src_floor) {
  auto &dst = src_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1;
  je = js + indcs_.nx2 - 1;
  ke = ks + indcs_.nx3 - 1;

  // local copies for device lambda capture
  const Real lfac = fac;
  const Real lsrc_min = src_min;
  const Real lsrc_floor = src_floor;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int lks = ks, ljs = js, lis = is, lngh = ngh;

  par_for("Multigrid::LoadSource", DevExeSpace(),
          m0, m1, v0, v1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
    const int nsrc = ns + v;
    const int k = mk - lks + lngh;
    const int j = mj - ljs + lngh;
    const int i = mi - lis + lngh;
    Real s = src(m, nsrc, k, j, i);
    s *= gravity_weight::Weight(s, lsrc_floor, lsrc_min);
    if (lfac == (Real)1.0) {
      dst(m, v, mk, mj, mi) = s;
    } else {
      dst(m, v, mk, mj, mi) = s * lfac;
    }
  });

  current_level_ = nlevel_-1;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::LoadSourceAndSubtractAverage(...)
//! \brief Load the finest source and remove its global mean using one source-read pass

void Multigrid::LoadSourceAndSubtractAverage(const DvceArray5D<Real> &src, int ns,
    int ngh,
                                             Real fac, Real src_min, Real src_floor) {
  auto &dst = src_[nlevel_-1].d_view;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + indcs_.nx1 - 1;
  je = js + indcs_.nx2 - 1;
  ke = ks + indcs_.nx3 - 1;

  const Real lfac = fac;
  const Real lsrc_min = src_min;
  const Real lsrc_floor = src_floor;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int lns = ns;
  const int lks = ks, ljs = js, lis = is, lngh = ngh;
  auto brdx = block_rdx_.d_view;

  Real local_sum = 0.0;
  Kokkos::parallel_reduce("Multigrid::LoadSourceAndSum",
    Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
        {m0, v0, ks, js, is}, {m1 + 1, v1 + 1, ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj,
                  const int mi, Real &lsum) {
      const int nsrc = lns + v;
      const int k = mk - lks + lngh;
      const int j = mj - ljs + lngh;
      const int i = mi - lis + lngh;
      Real s = src(m, nsrc, k, j, i);
      s *= gravity_weight::Weight(s, lsrc_floor, lsrc_min);
      const Real loaded = (lfac == (Real)1.0) ? s : s * lfac;
      dst(m, v, mk, mj, mi) = loaded;
      if (v == 0) {
        const Real dx = brdx(m);
        lsum += loaded * dx * dx * dx;
      }
    }, Kokkos::Sum<Real>(local_sum));

  Real local_volume = 0.0;
  const Real cell_count = static_cast<Real>(indcs_.nx1) *
                          static_cast<Real>(indcs_.nx2) *
                          static_cast<Real>(indcs_.nx3);
  for (int m = 0; m < nmmb_; ++m) {
    const Real dx = block_rdx_.h_view(m);
    local_volume += cell_count * dx * dx * dx;
  }
  // Cartesian AMR preserves cell aspect ratios, so apply dy/dx * dz/dx once
  // after the x1-cubic hot-loop accumulation.
  if (dvol_over_dx3_ != static_cast<Real>(1.0)) {
    local_sum *= dvol_over_dx3_;
    local_volume *= dvol_over_dx3_;
  }

  Real global_stats[2] = {local_sum, local_volume};
#if MPI_PARALLEL_ENABLED
  Real reduced_stats[2] = {0.0, 0.0};
  MPI_Allreduce(global_stats, reduced_stats, 2, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  global_stats[0] = reduced_stats[0];
  global_stats[1] = reduced_stats[1];
#endif

  current_level_ = nlevel_ - 1;
  if (global_stats[1] > 0.0) {
    SubtractAverage(MGVariable::src, 0, global_stats[0] / global_stats[1]);
    // The sum above runs over a source that is almost all constant, and its rounding
    // leaves a mean of order 1e-12 of that constant behind.  A constant lies in the
    // nullspace of the periodic operator, so no V-cycle can remove it: the defect floors
    // there, and the coarse levels turn it into a pattern of the same size.  The residual
    // after the first pass is O(perturbation), so a second pass on it is exact to the
    // round-off of the perturbation.  (Measured on the 64^3 Jeans wave, one rank: floor
    // 5e-12 with one pass, 3e-16 with two; the June 2026 solver reached 3e-16.)
    SubtractAverage(MGVariable::src, 0, CalculateAverage(MGVariable::src));
  }
}









//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh)
//! \brief Set the result, including the ghost zone

void Multigrid::RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh) {
  // The live range only: the whole-array deep_copy below must never be handed a block
  // dimension that runs past nmmb_ into the storage padding.
  auto src = LiveBlocks(u_[nlevel_-1].d_view);
  int sngh = std::min(ngh_,ngh);

  if (ns == 0 && ngh_ == ngh && nvar_ == 1
      && src.extent(0) == dst.extent(0)
      && src.extent(2) == dst.extent(2)
      && src.extent(3) == dst.extent(3)
      && src.extent(4) == dst.extent(4)) {
    Kokkos::deep_copy(dst, src);
  } else {
    int is, ie, js, je, ks, ke;
    is = js = ks = ngh_ - sngh;
    ie = indcs_.nx1 + ngh_ + sngh - 1;
    je = indcs_.nx2 + ngh_ + sngh - 1;
    ke = indcs_.nx3 + ngh_ + sngh - 1;

    const int dst_off = ngh - ngh_;

    par_for("Multigrid::RetrieveResult", DevExeSpace(),
            0, nmmb_-1, 0, nvar_-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int mk, const int mj, const int mi) {
      const int ndst = ns + v;
      const int k = mk + dst_off;
      const int j = mj + dst_off;
      const int i = mi + dst_off;
      dst(m, ndst, k, j, i) = src(m, v, mk, mj, mi);
    });
  }

  return;
}




//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ZeroClearData()
//! \brief Clear the data array with zero

void Multigrid::ZeroClearData() {
  if (on_host_) {
    Kokkos::deep_copy(u_[current_level_].h_view, 0.0);
  } else {
    Kokkos::deep_copy(u_[current_level_].d_view, 0.0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictPack()
//! \brief Restrict the defect to the source

void Multigrid::RestrictPack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  CalculateDefectPack();
  is=js=ks= ngh_;
  ie = is + (indcs_.nx1>>ll) - 1;
  je = js + (indcs_.nx2>>ll) - 1;
  ke = ks + (indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, def_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].h_view, u_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, def_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
    Restrict(u_[current_level_-1].d_view, u_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::RestrictSourcePack()
//! \brief Restrict the source (and solution) without forming defect

void Multigrid::RestrictSourcePack() {
  int ll=nlevel_-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks= ngh_;
  ie = is+(indcs_.nx1>>ll) - 1;
  je = js+(indcs_.nx2>>ll) - 1;
  ke = ks+(indcs_.nx3>>ll) - 1;
  if (on_host_) {
    Restrict(src_[current_level_-1].h_view, src_[current_level_].h_view,
             nvar_, is, ie, js, je, ks, ke, false);
  } else {
    Restrict(src_[current_level_-1].d_view, src_[current_level_].d_view,
             nvar_, is, ie, js, je, ks, ke, false);
  }
  current_level_--;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrectPack()
//! \brief Prolongate the potential using tri-linear interpolation

void Multigrid::ProlongateAndCorrectPack() {
  ComputeCorrection();
  ProlongatePreparedCorrectionPack();
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongatePreparedCorrectionPack()
//! \brief Prolongate a correction already stored in the current-level u array

void Multigrid::ProlongatePreparedCorrectionPack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  if (on_host_) {
    ProlongateAndCorrect(u_[current_level_+1].h_view, u_[current_level_].h_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  } else {
    ProlongateAndCorrect(u_[current_level_+1].d_view, u_[current_level_].d_view,
                         is, ie, js, je, ks, ke, ngh_, ngh_, ngh_, false);
  }

  current_level_++;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongatePack()
//! \brief Prolongate the solution for FMG (direct overwrite, always tricubic)

void Multigrid::FMGProlongatePack() {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1;
  je=js+(indcs_.nx2>>ll)-1;
  ke=ks+(indcs_.nx3>>ll)-1;

  if (on_host_) {
    FMGProlongate(u_[current_level_+1].h_view, u_[current_level_].h_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  } else {
    FMGProlongate(u_[current_level_+1].d_view, u_[current_level_].d_view,
                  is, ie, js, je, ks, ke, ngh_, ngh_, ngh_);
  }

  current_level_++;
}


//----------------------------------------------------------------------------------------
//! \fn  void Multigrid::SmoothPack(int color)
//! \brief Apply Smoother on the Pack


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::SetFromRootGrid(bool folddata)
//! \brief Load the data from the root grid or octets

void Multigrid::SetFromRootGrid(bool folddata) {
  current_level_ = 0;
  auto dst_h = u_[current_level_].h_view;
  auto odst_h = uold_[current_level_].h_view;

  auto src_h = pmy_driver_->GetRootData_h();
  auto osrc_h = pmy_driver_->GetRootOldData_h();
  int padding = pmy_mesh_->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmmb_; ++m) {
    auto loc = pmy_mesh_->lloc_eachmb[m + padding];
    int lev = loc.level - pmy_driver_->locrootlevel_;
    if (lev == 0) {
      int ci = static_cast<int>(loc.lx1);
      int cj = static_cast<int>(loc.lx2);
      int ck = static_cast<int>(loc.lx3);
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = src_h(0, v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = osrc_h(0, v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    } else {
      LogicalLocation oloc;
      oloc.lx1 = (loc.lx1 >> 1);
      oloc.lx2 = (loc.lx2 >> 1);
      oloc.lx3 = (loc.lx3 >> 1);
      oloc.level = loc.level - 1;
      int olev = oloc.level - pmy_driver_->locrootlevel_;
      int oid = pmy_driver_->octetmap_[olev][oloc];
      int ci = (static_cast<int>(loc.lx1) & 1);
      int cj = (static_cast<int>(loc.lx2) & 1);
      int ck = (static_cast<int>(loc.lx3) & 1);
      const MGOctet &oct = pmy_driver_->octets_[olev][oid];
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k <= 2*ngh_; ++k) {
          for (int j = 0; j <= 2*ngh_; ++j) {
            for (int i = 0; i <= 2*ngh_; ++i) {
              dst_h(m, v, k, j, i) = oct.U(v, ck+k, cj+j, ci+i);
              if (folddata)
                odst_h(m, v, k, j, i) = oct.Uold(v, ck+k, cj+j, ci+i);
            }
          }
        }
      }
    }
  }
  u_[current_level_].template modify<HostExeSpace>();
  u_[current_level_].template sync<DevExeSpace>();
  if (folddata) {
    uold_[current_level_].template modify<HostExeSpace>();
    uold_[current_level_].template sync<DevExeSpace>();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the residual norm

Real Multigrid::CalculateDefectNorm(MGNormType nrm, int n) {
  int ll=nlevel_-1-current_level_;
  int is, ie, js, je, ks, ke;
  is=js=ks=ngh_;
  ie=is+(indcs_.nx1>>ll)-1, je=js+(indcs_.nx2>>ll)-1, ke=ks+(indcs_.nx3>>ll)-1;
  const int ll_l = ll;
  CalculateDefectPack();

  Real norm = 0.0;

  if (on_host_) {
    auto &def = def_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          const Real val = def(m, v, k, j, i);
          const Real abs_val = (val >= 0.0) ? val : -val;
          local_max = (local_max > abs_val) ? local_max : abs_val;
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1<<ll_l);
          Real dvm = dxm*dxm*dxm;
          local_sum += std::abs(def(m, v, k, j, i)) * dvm;
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1<<ll_l);
          Real dvm = dxm*dxm*dxm;
          Real val = def(m, v, k, j, i);
          local_sum += val * val * dvm;
        }, Kokkos::Sum<Real>(norm));
    }
  } else {
    auto &def = def_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::DefectNorm_Linf",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_max) {
          const Real val = def(m, v, k, j, i);
          const Real abs_val = (val >= 0.0) ? val : -val;
          local_max = (local_max > abs_val) ? local_max : abs_val;
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::DefectNorm_L1",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1<<ll_l);
          Real dvm = dxm*dxm*dxm;
          local_sum += std::abs(def(m, v, k, j, i)) * dvm;
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::DefectNorm_L2",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n+1, ke+1, je+1, ie+1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                       const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1<<ll_l);
          Real dvm = dxm*dxm*dxm;
          Real val = def(m, v, k, j, i);
          local_sum += val * val * dvm;
        }, Kokkos::Sum<Real>(norm));
    }
  }
  if (dvol_over_dx3_ != static_cast<Real>(1.0)) norm *= dvol_over_dx3_;
  norm *= defscale_;
  return norm;
}

//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateArrayNorm(MGVariable type, MGNormType nrm, int n)
//! \brief calculate a norm of the requested multigrid array on the current level

Real Multigrid::CalculateArrayNorm(MGVariable type, MGNormType nrm, int n) {
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + (indcs_.nx1 >> ll) - 1;
  je = js + (indcs_.nx2 >> ll) - 1;
  ke = ks + (indcs_.nx3 >> ll) - 1;
  const int ll_l = ll;

  Real norm = 0.0;

  if (on_host_) {
    auto data = (type == MGVariable::src) ? src_[current_level_].h_view
                                          : u_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::ArrayNorm_Linf",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_max) {
          const Real val = data(m, v, k, j, i);
          const Real abs_val = (val >= 0.0) ? val : -val;
          local_max = (local_max > abs_val) ? local_max : abs_val;
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::ArrayNorm_L1",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dvm = dxm * dxm * dxm;
          local_sum += std::abs(data(m, v, k, j, i)) * dvm;
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::ArrayNorm_L2",
        Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<5>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dvm = dxm * dxm * dxm;
          Real val = data(m, v, k, j, i);
          local_sum += val * val * dvm;
        }, Kokkos::Sum<Real>(norm));
    }
  } else {
    auto data = (type == MGVariable::src) ? src_[current_level_].d_view
                                          : u_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    if (nrm == MGNormType::max) {
      Kokkos::parallel_reduce("MG::ArrayNorm_Linf",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_max) {
          const Real val = data(m, v, k, j, i);
          const Real abs_val = (val >= 0.0) ? val : -val;
          local_max = (local_max > abs_val) ? local_max : abs_val;
        }, Kokkos::Max<Real>(norm));
      return norm;
    } else if (nrm == MGNormType::l1) {
      Kokkos::parallel_reduce("MG::ArrayNorm_L1",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dvm = dxm * dxm * dxm;
          local_sum += std::abs(data(m, v, k, j, i)) * dvm;
        }, Kokkos::Sum<Real>(norm));
    } else {
      Kokkos::parallel_reduce("MG::ArrayNorm_L2",
        Kokkos::MDRangePolicy<DevExeSpace,
            Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
            {0, n, ks, js, is}, {nmmb_, n + 1, ke + 1, je + 1, ie + 1}),
        KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                      const int i, Real &local_sum) {
          Real dxm = brdx(m) * static_cast<Real>(1 << ll_l);
          Real dvm = dxm * dxm * dxm;
          Real val = data(m, v, k, j, i);
          local_sum += val * val * dvm;
        }, Kokkos::Sum<Real>(norm));
    }
  }

  if (dvol_over_dx3_ != static_cast<Real>(1.0)) norm *= dvol_over_dx3_;

  // Return the local volume integral. The driver performs the MPI reduction and
  // applies global-volume normalization (and the L2 square root) exactly once.
  return norm;
}

//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::CalculateAverage(MGVariable type)
//! \brief Calculate volume-weighted average of variable 0 on current level

Real Multigrid::CalculateAverage(MGVariable type) {
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = ngh_;
  ie = is + (indcs_.nx1 >> ll) - 1;
  je = js + (indcs_.nx2 >> ll) - 1;
  ke = ks + (indcs_.nx3 >> ll) - 1;
  int ll_l = ll;

  Real sum = 0.0;
  if (on_host_) {
    auto data = (type == MGVariable::src) ? src_[current_level_].h_view
                                          : u_[current_level_].h_view;
    auto brdx = block_rdx_.h_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<HostExeSpace, Kokkos::Rank<4>>({0, ks, js, is},
                                                            {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  } else {
    auto data = (type == MGVariable::src) ? src_[current_level_].d_view
                                          : u_[current_level_].d_view;
    auto brdx = block_rdx_.d_view;
    Kokkos::parallel_reduce("MG::Average",
      Kokkos::MDRangePolicy<DevExeSpace,
          Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>({0, ks, js,
              is},
                                                           {nmmb_, ke+1, je+1, ie+1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &local_sum) {
        Real dx_m = brdx(m) * static_cast<Real>(1 << ll_l);
        Real dV_m = dx_m * dx_m * dx_m;
        local_sum += data(m, 0, k, j, i) * dV_m;
      }, Kokkos::Sum<Real>(sum));
  }

  Real volume = 0.0;
  {
    const Real cell_count = static_cast<Real>(indcs_.nx1) *
                            static_cast<Real>(indcs_.nx2) *
                            static_cast<Real>(indcs_.nx3);
    for (int m = 0; m < nmmb_; ++m) {
      const Real dx = block_rdx_.h_view(m);
      volume += cell_count * dx * dx * dx;
    }
  }
  if (dvol_over_dx3_ != static_cast<Real>(1.0)) {
    sum *= dvol_over_dx3_;
    volume *= dvol_over_dx3_;
  }

  #if MPI_PARALLEL_ENABLED
  const bool owner_has_global_root_data =
      (pmy_pack_ == nullptr) && (pmy_driver_ != nullptr) &&
      pmy_driver_->UseDistributedCoarseSolve() && pmy_driver_->IsCoarseSolveOwner();
  if (!owner_has_global_root_data) {
    Real local_stats[2] = {sum, volume};
    Real global_stats[2] = {0.0, 0.0};
    MPI_Allreduce(local_stats, global_stats, 2, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    sum = global_stats[0];
    volume = global_stats[1];
  }
  #endif

  return (volume > 0.0) ? (sum / volume) : 0.0;
}




//----------------------------------------------------------------------------------------
//! \fn Real Multigrid::SubtractAverage(MGVariable type, int v, Real ave)
//! \brief subtract the average value (type: 0=src, 1=u)

void Multigrid::SubtractAverage(MGVariable type, int n, Real ave) {
  int ll = nlevel_ - 1 - current_level_;
  int is, ie, js, je, ks, ke;
  is = js = ks = 0;
  ie = is + (indcs_.nx1 >> ll) + 2*ngh_ - 1;
  je = js + (indcs_.nx2 >> ll) + 2*ngh_ - 1;
  ke = ks + (indcs_.nx3 >> ll) + 2*ngh_ - 1;

  const int m0 = 0, m1 = nmmb_ - 1;
  const int vn = n;
  const Real lave = ave;

  if (on_host_) {
    auto dst = (type == MGVariable::src) ? src_[current_level_].h_view
                                         : u_[current_level_].h_view;
    par_for("Multigrid::SubtractAverage", HostExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  } else {
    auto dst = (type == MGVariable::src) ? src_[current_level_].d_view
                                         : u_[current_level_].d_view;
    par_for("Multigrid::SubtractAverage", DevExeSpace(),
            m0, m1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int mk, const int mj, const int mi) {
      dst(m, vn, mk, mj, mi) -= lave;
    });
  }
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::StoreOldData()
//! \brief store the old u data in the uold array

// The live blocks only.  This runs once per level per V-cycle and is pure bandwidth, so
// copying the storage grain past nmmb_ -- up to fifteen blocks of it, which nothing reads
// -- is the one place the over-allocation would otherwise cost something per cycle.
void Multigrid::StoreOldData() {
  if (on_host_) {
    Kokkos::deep_copy(HostExeSpace(), LiveBlocks(uold_[current_level_].h_view),
                      LiveBlocks(u_[current_level_].h_view));
  } else {
    Kokkos::deep_copy(DevExeSpace(), LiveBlocks(uold_[current_level_].d_view),
                      LiveBlocks(u_[current_level_].d_view));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::Restrict(...)
//  \brief Actual implementation of restriction (templated on view type)

template <typename ViewType>
void Multigrid::Restrict(ViewType &dst, const ViewType &src,
                int nvar, int i0, int i1, int j0, int j1, int k0, int k1, bool th) {
  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar - 1;
  const int ngh = ngh_;

  par_for("Multigrid::Restrict", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*k - ngh;
    const int fj = 2*j - ngh;
    const int fi = 2*i - ngh;
    dst(m, v, k, j, i) = 0.125 * (
        src(m, v, fk,   fj,   fi)   + src(m, v, fk,   fj,   fi+1)
      + src(m, v, fk,   fj+1, fi)   + src(m, v, fk,   fj+1, fi+1)
      + src(m, v, fk+1, fj,   fi)   + src(m, v, fk+1, fj,   fi+1)
      + src(m, v, fk+1, fj+1, fi)   + src(m, v, fk+1, fj+1, fi+1));
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ComputeCorrection(DvceArray5D<Real> &correction, int level)
//! \brief Compute the correction as u_[level] - uold_[level]

void Multigrid::ComputeCorrection() {
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  int ll = nlevel_ - 1 - current_level_;
  int is = 0, ie = is + (indcs_.nx1 >> ll) + 2*ngh_ -1;
  int js = 0, je = js + (indcs_.nx2 >> ll) + 2*ngh_ -1;
  int ks = 0, ke = ks + (indcs_.nx3 >> ll) + 2*ngh_ -1;

  if (on_host_) {
    auto u = u_[current_level_].h_view;
    auto uold = uold_[current_level_].h_view;
    par_for("Multigrid::ComputeCorrection", HostExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  } else {
    auto u = u_[current_level_].d_view;
    auto uold = uold_[current_level_].d_view;
    par_for("Multigrid::ComputeCorrection", DevExeSpace(),
            m0, m1, v0, v1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u(m, v, k, j, i) -= uold(m, v, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Multigrid::ProlongateAndCorrect(...)
//! \brief Actual implementation of prolongation and correction (templated on view type)

template <typename ViewType>
void Multigrid::ProlongateAndCorrect(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th) {
  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  const int ll = pmy_driver_->fprolongation_; // copy host flag for capture

  auto dst_ = dst;
  auto src_ = src;

  if (ll == 1) { // tricubic
    par_for("Multigrid::ProlongateAndCorrect_tricubic_factorized", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      constexpr Real wm0 = 5.0, wc = 30.0, wp0 = -3.0;
      constexpr Real wm1 = -3.0, wp1 = 5.0;
      constexpr Real inv = 1.0 / 32768.0;

      Real sx0[3][3], sx1[3][3];
      for (int kk = 0; kk < 3; ++kk) {
        const int sk = k + kk - 1;
        for (int jj = 0; jj < 3; ++jj) {
          const int sj = j + jj - 1;
          const Real sm = src_(m,v,sk,sj,i-1);
          const Real sc = src_(m,v,sk,sj,i  );
          const Real sp = src_(m,v,sk,sj,i+1);
          sx0[kk][jj] = wm0*sm + wc*sc + wp0*sp;
          sx1[kk][jj] = wm1*sm + wc*sc + wp1*sp;
        }
      }

      Real sxy00[3], sxy01[3], sxy10[3], sxy11[3];
      for (int kk = 0; kk < 3; ++kk) {
        sxy00[kk] = wm0*sx0[kk][0] + wc*sx0[kk][1] + wp0*sx0[kk][2];
        sxy01[kk] = wm0*sx1[kk][0] + wc*sx1[kk][1] + wp0*sx1[kk][2];
        sxy10[kk] = wm1*sx0[kk][0] + wc*sx0[kk][1] + wp1*sx0[kk][2];
        sxy11[kk] = wm1*sx1[kk][0] + wc*sx1[kk][1] + wp1*sx1[kk][2];
      }

      dst_(m,v,fk  ,fj  ,fi  ) +=
          (wm0*sxy00[0] + wc*sxy00[1] + wp0*sxy00[2]) * inv;
      dst_(m,v,fk  ,fj  ,fi+1) +=
          (wm0*sxy01[0] + wc*sxy01[1] + wp0*sxy01[2]) * inv;
      dst_(m,v,fk  ,fj+1,fi  ) +=
          (wm0*sxy10[0] + wc*sxy10[1] + wp0*sxy10[2]) * inv;
      dst_(m,v,fk  ,fj+1,fi+1) +=
          (wm0*sxy11[0] + wc*sxy11[1] + wp0*sxy11[2]) * inv;
      dst_(m,v,fk+1,fj  ,fi  ) +=
          (wm1*sxy00[0] + wc*sxy00[1] + wp1*sxy00[2]) * inv;
      dst_(m,v,fk+1,fj  ,fi+1) +=
          (wm1*sxy01[0] + wc*sxy01[1] + wp1*sxy01[2]) * inv;
      dst_(m,v,fk+1,fj+1,fi  ) +=
          (wm1*sxy10[0] + wc*sxy10[1] + wp1*sxy10[2]) * inv;
      dst_(m,v,fk+1,fj+1,fi+1) +=
          (wm1*sxy11[0] + wc*sxy11[1] + wp1*sxy11[2]) * inv;
    });
  } else { // trilinear
    par_for("Multigrid::ProlongateAndCorrect_trilinear", ExeSpace(),
            m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      const int fk = 2*(k-kl) + fkl;
      const int fj = 2*(j-jl) + fjl;
      const int fi = 2*(i-il) + fil;

      dst_(m,v,fk  ,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk  ,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j-1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j-1,i-1)));
      dst_(m,v,fk+1,fj+1,fi  ) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i-1)
                    +9.0*(src_(m,v,k,j,i-1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i-1)+src_(m,v,k,j+1,i-1)));
      dst_(m,v,fk+1,fj  ,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j-1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j-1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j-1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j-1,i+1)));
      dst_(m,v,fk  ,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k-1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k-1,j,i))
                    +3.0*(src_(m,v,k-1,j+1,i)+src_(m,v,k-1,j,i+1)+src_(m,v,k,j+1,i+1)));
      dst_(m,v,fk+1,fj+1,fi+1) +=
          0.015625*(27.0*src_(m,v,k,j,i) + src_(m,v,k+1,j+1,i+1)
                    +9.0*(src_(m,v,k,j,i+1)+src_(m,v,k,j+1,i)+src_(m,v,k+1,j,i))
                    +3.0*(src_(m,v,k+1,j+1,i)+src_(m,v,k+1,j,i+1)+src_(m,v,k,j+1,i+1)));
    });
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Multigrid::FMGProlongate(...)
//! \brief FMG prolongation: direct overwrite (=) with tricubic interpolation.
//! Unlike ProlongateAndCorrect (+=), this overwrites the destination array.

template <typename ViewType>
void Multigrid::FMGProlongate(ViewType &dst, const ViewType &src,
     int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl) {
  using ExeSpace = typename ViewType::execution_space;
  const int m0 = 0, m1 = nmmb_ - 1;
  const int v0 = 0, v1 = nvar_ - 1;
  const int k0 = kl, k1 = ku;
  const int j0 = jl, j1 = ju;
  const int i0 = il, i1 = iu;

  auto dst_ = dst;
  auto src_ = src;

  par_for("Multigrid::FMGProlongate", ExeSpace(),
          m0, m1, v0, v1, k0, k1, j0, j1, i0, i1,
  KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
    const int fk = 2*(k-kl) + fkl;
    const int fj = 2*(j-jl) + fjl;
    const int fi = 2*(i-il) + fil;

    dst_(m,v,fk  ,fj  ,fi  ) = (
      + 125.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,
          i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,
          i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,
          i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,
          i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,
          i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,
          i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,
          i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,
          i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )-  27.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj,  fi+1) = (
      -  75.*src_(m,v,k-1,j-1,i-1)+  750.*src_(m,v,k-1,j-1,i  )+ 125.*src_(m,v,k-1,j-1,
          i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,
          i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,
          i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,
          i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,
          i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,
          i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,
          i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,
          i+1)
      -  27.*src_(m,v,k+1,j+1,i-1)+  270.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk  ,fj+1,fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,
          i+1)
      + 750.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,
          i+1)
      + 125.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,
          i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,
          i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,
          i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,
          i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )-  27.*src_(m,v,k+1,j-1,
          i+1)
      - 450.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )+ 270.*src_(m,v,k+1,j,
          i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk,  fj+1,fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,
          i+1)
      - 450.*src_(m,v,k-1,j,  i-1)+ 4500.*src_(m,v,k-1,j,  i  )+ 750.*src_(m,v,k-1,j,
          i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)+  750.*src_(m,v,k-1,j+1,i  )+ 125.*src_(m,v,k-1,j+1,
          i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,
          i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,
          i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,
          i+1)
      -  27.*src_(m,v,k+1,j-1,i-1)+  270.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,
          i+1)
      + 270.*src_(m,v,k+1,j,  i-1)- 2700.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,
          i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi  ) = (
      -  75.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,
          i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,
          i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )-  27.*src_(m,v,k-1,j+1,
          i+1)
      + 750.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,
          i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,
          i+1)
      - 450.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )+ 270.*src_(m,v,k,  j+1,
          i+1)
      + 125.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,
          i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,
          i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )+  45.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj,  fi+1) = (
      +  45.*src_(m,v,k-1,j-1,i-1)-  450.*src_(m,v,k-1,j-1,i  )-  75.*src_(m,v,k-1,j-1,
          i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,
          i+1)
      -  27.*src_(m,v,k-1,j+1,i-1)+  270.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,
          i+1)
      - 450.*src_(m,v,k,  j-1,i-1)+ 4500.*src_(m,v,k,  j-1,i  )+ 750.*src_(m,v,k,  j-1,
          i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,
          i+1)
      + 270.*src_(m,v,k,  j+1,i-1)- 2700.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,
          i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)+  750.*src_(m,v,k+1,j-1,i  )+ 125.*src_(m,v,k+1,j-1,
          i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,
          i+1)
      +  45.*src_(m,v,k+1,j+1,i-1)-  450.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi  ) = (
      +  45.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )-  27.*src_(m,v,k-1,j-1,
          i+1)
      - 450.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )+ 270.*src_(m,v,k-1,j,
          i+1)
      -  75.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )+  45.*src_(m,v,k-1,j+1,
          i+1)
      - 450.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )+ 270.*src_(m,v,k,  j-1,
          i+1)
      +4500.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )-2700.*src_(m,v,k,  j,
          i+1)
      + 750.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )- 450.*src_(m,v,k,  j+1,
          i+1)
      -  75.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )+  45.*src_(m,v,k+1,j-1,
          i+1)
      + 750.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )- 450.*src_(m,v,k+1,j,
          i+1)
      + 125.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )-  75.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;

    dst_(m,v,fk+1,fj+1,fi+1) = (
      -  27.*src_(m,v,k-1,j-1,i-1)+  270.*src_(m,v,k-1,j-1,i  )+  45.*src_(m,v,k-1,j-1,
          i+1)
      + 270.*src_(m,v,k-1,j,  i-1)- 2700.*src_(m,v,k-1,j,  i  )- 450.*src_(m,v,k-1,j,
          i+1)
      +  45.*src_(m,v,k-1,j+1,i-1)-  450.*src_(m,v,k-1,j+1,i  )-  75.*src_(m,v,k-1,j+1,
          i+1)
      + 270.*src_(m,v,k,  j-1,i-1)- 2700.*src_(m,v,k,  j-1,i  )- 450.*src_(m,v,k,  j-1,
          i+1)
      -2700.*src_(m,v,k,  j,  i-1)+27000.*src_(m,v,k,  j,  i  )+4500.*src_(m,v,k,  j,
          i+1)
      - 450.*src_(m,v,k,  j+1,i-1)+ 4500.*src_(m,v,k,  j+1,i  )+ 750.*src_(m,v,k,  j+1,
          i+1)
      +  45.*src_(m,v,k+1,j-1,i-1)-  450.*src_(m,v,k+1,j-1,i  )-  75.*src_(m,v,k+1,j-1,
          i+1)
      - 450.*src_(m,v,k+1,j,  i-1)+ 4500.*src_(m,v,k+1,j,  i  )+ 750.*src_(m,v,k+1,j,
          i+1)
      -  75.*src_(m,v,k+1,j+1,i-1)+  750.*src_(m,v,k+1,j+1,i  )+ 125.*src_(m,v,k+1,j+1,
          i+1)
    ) / 32768.0;
  });
  return;
}


//----------------------------------------------------------------------------------------
//! \fn MultigridBoundaryValues::MultigridBoundaryValues()
//! \brief Constructor for multigrid boundary values object
//----------------------------------------------------------------------------------------

MultigridBoundaryValues::MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin,
    bool coarse, Multigrid *pmg)
  :
    MeshBoundaryValuesCC(pmbp, pin, coarse), pmy_mg(pmg),
    fc_scan_mesh_sig_(0), fc_scan_nmb_(-1), fc_scan_nnghbr_(-1),
    fc_scan_valid_(false), fc_scan_has_local_(false), fc_scan_has_remote_(false),
    fc_scan_has_remote_face_(false), fc_scan_has_remote_diag_(false),
    fc_mesh_sig_(0), fc_ngh_map_nnghbr_(-1), fc_ngh_map_valid_(false),
    fc_remote_cache_valid_(false),
    fc_remote_axis_mask_valid_(false),
    fc_remote_axis_mask_mesh_sig_(0), fc_remote_axis_mask_nnghbr_(-1),
    fc_remote_compact_axis_mask_(-1), fc_remote_send_slot_inflight_{false, false},
    fc_remote_send_slot_shift_{-1, -1},
    fc_remote_send_inflight_(false),
    fc_remote_send_pack_slot_(0),
    mg_same_mesh_sig_(0), mg_same_shift_(-1), mg_same_nvars_(-1),
    mg_same_include_diagonals_(false), mg_same_cache_valid_(false),
    mg_same_send_slot_inflight_{false, false}, mg_same_send_inflight_(false),
    mg_same_recv_inflight_(false),
    mg_same_persistent_ready_(false), mg_same_has_remote_(false),
    mg_same_poll_budget_(1), mg_same_send_poll_count_(0), mg_same_recv_poll_count_(0),
    mg_same_send_pack_slot_(0),
    mg_same_recv_post_slot_(0), mg_same_recv_data_slot_(0),
    mg_unpack_pending_ {false, false, false}
#if defined(KOKKOS_ENABLE_CUDA)
    , mg_unpack_event_{nullptr, nullptr, nullptr},
    mg_unpack_event_valid_ {false, false, false}
#endif
    {
  // Every payload these buffers carry is a same-level one: the fine/coarse ghosts go
  // through the fc_remote_* path below, and nothing under src/multigrid or src/gravity
  // reads icoar or ifine.  Declaring that before InitializeBuffers keeps the seed
  // allocation from reserving those two footprints as well.
  same_level_vars_only = true;
  // Small poll budget avoids spending excessive host time in repeated MPI_Testall
  // loops when requests are not yet ready; this improves overlap on PCIe-era systems.
  // Taken through env_switch so every rank polls with rank 0's budget: a rank-dependent
  // budget makes the MPI progress pattern rank-dependent and the run irreproducible.
  const long parsed = env_switch::Int("ATHENA_MG_POLL_BUDGET", 0);
  if (parsed > 0) {
    mg_same_poll_budget_ = static_cast<int>(parsed);
  }
}

MultigridBoundaryValues::~MultigridBoundaryValues() {
#if MPI_PARALLEL_ENABLED
  FreeFCRemotePersistentRequests();
  InvalidateMGSameLevelAggCache();
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same<DevExeSpace, Kokkos::Cuda>::value) {
    for (int store = 0; store < kMGNumRecvStores; ++store) {
      if (mg_unpack_event_[store] == nullptr) continue;
      (void)cudaEventDestroy(mg_unpack_event_[store]);
      mg_unpack_event_[store] = nullptr;
      mg_unpack_event_valid_[store] = false;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::MarkMGUnpackPending(int store)
//! \brief record that an unpack kernel is in flight over receive storage `store`

void MultigridBoundaryValues::MarkMGUnpackPending(int store) {
#if MPI_PARALLEL_ENABLED
  mg_unpack_pending_[store] = true;
  // The base class flag stays truthful for the paths that move the buffers wholesale;
  // they fall back to the full-device fence, which is correct for any storage.
  MarkRecvUnpackPendingFullFence();
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same<DevExeSpace, Kokkos::Cuda>::value) {
    if (mg_unpack_event_[store] == nullptr) {
      if (cudaEventCreateWithFlags(&mg_unpack_event_[store], cudaEventDisableTiming)
          != cudaSuccess) {
        mg_unpack_event_[store] = nullptr;
      }
    }
    if (mg_unpack_event_[store] != nullptr) {
      // Recorded on the stream the unpack was launched on, so the event is reached
      // exactly when it retires.  A failed create or record leaves the event invalid and
      // WaitMGUnpackComplete() falls back to the full-device fence.
      mg_unpack_event_valid_[store] =
          (cudaEventRecord(mg_unpack_event_[store], DevExeSpace().cuda_stream()) ==
           cudaSuccess);
    }
  }
#endif
#else
  (void)store;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::WaitMGUnpackComplete(int store)
//! \brief establish that nothing is still reading receive storage `store`

void MultigridBoundaryValues::WaitMGUnpackComplete(int store) {
#if MPI_PARALLEL_ENABLED
  if (!mg_unpack_pending_[store]) return;
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same<DevExeSpace, Kokkos::Cuda>::value) {
    if (mg_unpack_event_valid_[store] && (mg_unpack_event_[store] != nullptr)) {
      const cudaError_t err = cudaEventSynchronize(mg_unpack_event_[store]);
      mg_unpack_event_valid_[store] = false;
      mg_unpack_pending_[store] = false;
      if (err == cudaSuccess) return;
      // Fall through to the full fence: a failed wait must not be treated as a wait.
    }
  }
#endif
  WaitMGUnpackCompleteAll();
#else
  (void)store;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::WaitMGUnpackCompleteAll()
//! \brief drain every pending unpack, for the callers that move the buffers themselves

void MultigridBoundaryValues::WaitMGUnpackCompleteAll() {
#if MPI_PARALLEL_ENABLED
  bool any_pending = recv_unpack_pending_;
  for (int store = 0; store < kMGNumRecvStores; ++store) {
    any_pending = any_pending || mg_unpack_pending_[store];
  }
  if (!any_pending) return;
  DevExeSpace().fence();
  for (int store = 0; store < kMGNumRecvStores; ++store) {
    mg_unpack_pending_[store] = false;
#if defined(KOKKOS_ENABLE_CUDA)
    mg_unpack_event_valid_[store] = false;
#endif
  }
  recv_unpack_pending_ = false;
#endif
}

TaskStatus MultigridBoundaryValues::ClearRecvMG() {
#if MPI_PARALLEL_ENABLED
  if (mg_same_recv_inflight_) {
    WaitMGSameLevelRecvRequests();
  }
  return TaskStatus::complete;
#else
  return MeshBoundaryValues::ClearRecv();
#endif
}

TaskStatus MultigridBoundaryValues::ClearSendMG() {
#if MPI_PARALLEL_ENABLED
  if (mg_same_send_inflight_) {
    WaitMGSameLevelSendRequests();
  }
  if (fc_remote_send_inflight_) {
    WaitFCRemoteSendSlots();
  }
  return TaskStatus::complete;
#else
  return MeshBoundaryValues::ClearSend();
#endif
}

#if MPI_PARALLEL_ENABLED
void MultigridBoundaryValues::SwapMGSameLevelAggState(MGSameLevelAggState &state) {
  using std::swap;
  swap(mg_same_send_m_d_, state.send_m_d);
  swap(mg_same_send_off_d_, state.send_off_d);
  swap(mg_same_recv_m_d_, state.recv_m_d);
  swap(mg_same_recv_off_d_, state.recv_off_d);
  swap(mg_same_send_bounds_d_, state.send_bounds_d);
  swap(mg_same_recv_bounds_d_, state.recv_bounds_d);
  swap(mg_same_send_data_d_, state.send_data_d);
  swap(mg_same_recv_data_d_, state.recv_data_d);
  swap(mg_same_send_rank_h_, state.send_rank_h);
  swap(mg_same_send_data_off_h_, state.send_data_off_h);
  swap(mg_same_send_tag_h_, state.send_tag_h);
  swap(mg_same_recv_rank_h_, state.recv_rank_h);
  swap(mg_same_recv_data_off_h_, state.recv_data_off_h);
  swap(mg_same_recv_tag_h_, state.recv_tag_h);
  swap(mg_same_send_req_h_, state.send_req_h);
  swap(mg_same_recv_req_h_, state.recv_req_h);
  swap(mg_same_send_slot_inflight_, state.send_slot_inflight);
  swap(mg_same_send_inflight_, state.send_inflight);
  swap(mg_same_recv_inflight_, state.recv_inflight);
  swap(mg_same_persistent_ready_, state.persistent_ready);
  swap(mg_same_has_remote_, state.has_remote);
  swap(mg_same_send_poll_count_, state.send_poll_count);
  swap(mg_same_recv_poll_count_, state.recv_poll_count);
  swap(mg_same_send_pack_slot_, state.send_pack_slot);
  swap(mg_same_recv_post_slot_, state.recv_post_slot);
  swap(mg_same_recv_data_slot_, state.recv_data_slot);
}

void MultigridBoundaryValues::FreeMGSameLevelAggState(MGSameLevelAggState &state) {
  bool no_errors = true;
  for (int slot = 0; slot < 2; ++slot) {
    if (!state.send_slot_inflight[slot]) continue;
    auto &requests = state.send_req_h[slot];
    if (!requests.empty()) {
      int ierr = MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                             MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) no_errors = false;
    }
    state.send_slot_inflight[slot] = false;
  }
  state.send_inflight = false;
  if (state.recv_inflight) {
    auto &requests = state.recv_req_h[state.recv_data_slot];
    if (!requests.empty()) {
      int ierr = MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                             MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) no_errors = false;
    }
  }
  state.recv_inflight = false;

  if (state.persistent_ready) {
    for (auto &requests : state.send_req_h) {
      for (auto &request : requests) {
        if (request == MPI_REQUEST_NULL) continue;
        int ierr = MPI_Request_free(&request);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    for (auto &requests : state.recv_req_h) {
      for (auto &request : requests) {
        if (request == MPI_REQUEST_NULL) continue;
        int ierr = MPI_Request_free(&request);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in freeing cached MG requests" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  state = MGSameLevelAggState();
}

void MultigridBoundaryValues::InvalidateMGSameLevelAggCache() {
  WaitMGUnpackCompleteAll();
  // Move the resident state out first so request and buffer ownership remains unique.
  MGSameLevelAggState resident;
  SwapMGSameLevelAggState(resident);
  FreeMGSameLevelAggState(resident);
  for (auto &state : mg_same_level_cache_) {
    FreeMGSameLevelAggState(state);
  }
  mg_same_level_cache_.clear();
  mg_same_level_cache_mesh_sig_ = 0;
  mg_same_active_cache_key_ = -1;
  mg_same_mesh_sig_ = 0;
  mg_same_shift_ = -1;
  mg_same_nvars_ = -1;
  mg_same_include_diagonals_ = false;
  mg_same_cache_valid_ = false;
  mg_same_recv_post_slot_ = 0;
  mg_same_recv_data_slot_ = 0;
}

//! \fn MultigridBoundaryValues::FCRemoteSlotSet()
//! \brief persistent request set that owns the in-flight send in `slot`, or nullptr
//!
//! The two send slots are shared by all level shifts (they double-buffer one payload
//! allocation), so the shift that started a slot has to be remembered explicitly.

MultigridBoundaryValues::FCRemotePersistentSet *
MultigridBoundaryValues::FCRemoteSlotSet(int slot) {
  const int shift = fc_remote_send_slot_shift_[slot];
  if (shift < 0 || shift >= static_cast<int>(fc_remote_persistent_.size())) {
    return nullptr;
  }
  return &fc_remote_persistent_[shift];
}

//! \fn MultigridBoundaryValues::WaitFCRemoteSendSlots()
//! \brief drain both deferred FC send slots without tearing anything down
//!
//! Used before the payload slots are repacked. The persistent requests stay registered:
//! a completed persistent request is simply started again, and re-registering it would
//! reintroduce the churn this cache exists to avoid.

void MultigridBoundaryValues::WaitFCRemoteSendSlots() {
  for (int slot = 0; slot < 2; ++slot) {
    if (!fc_remote_send_slot_inflight_[slot]) continue;
    FCRemotePersistentSet *pset = FCRemoteSlotSet(slot);
    if (pset == nullptr) {
      // Nothing owns this slot's requests, so nothing can wait on them; dropping the
      // flag would let the caller repack a payload MPI is still reading.
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "orphaned in-flight FC send slot " << slot << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &slot_reqs = pset->send_req_h[slot];
    if (!slot_reqs.empty()) {
      int ierr = MPI_Waitall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in clearing deferred FC sends" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (!pset->ready) {
        std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
      }
    }
    fc_remote_send_slot_inflight_[slot] = false;
    fc_remote_send_slot_shift_[slot] = -1;
  }
  fc_remote_send_inflight_ = false;
}

//! \fn MultigridBoundaryValues::FreeFCRemotePersistentSet(int shift)
//! \brief drain and tear down one level shift's persistent FC requests
//!
//! Any send this shift still has in flight is waited on first: an MPI_Request_free on a
//! started persistent request is illegal, and the payload buffer must be quiescent
//! before it can be reused or reallocated.

void MultigridBoundaryValues::FreeFCRemotePersistentSet(int shift) {
  if (shift < 0 || shift >= static_cast<int>(fc_remote_persistent_.size())) return;
  auto &pset = fc_remote_persistent_[shift];
  for (int slot = 0; slot < 2; ++slot) {
    if (!fc_remote_send_slot_inflight_[slot]) continue;
    if (fc_remote_send_slot_shift_[slot] != shift) continue;
    auto &slot_reqs = pset.send_req_h[slot];
    if (!slot_reqs.empty()) {
      int ierr = MPI_Waitall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in waiting deferred FC sends" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    fc_remote_send_slot_inflight_[slot] = false;
    fc_remote_send_slot_shift_[slot] = -1;
  }
  fc_remote_send_inflight_ = (fc_remote_send_slot_inflight_[0]
                              || fc_remote_send_slot_inflight_[1]);
  if (pset.ready) {
    bool no_errors = true;
    for (auto &slot_reqs : pset.send_req_h) {
      for (auto &req : slot_reqs) {
        if (req == MPI_REQUEST_NULL) continue;
        int ierr = MPI_Request_free(&req);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    for (auto &req : pset.recv_req_h) {
      if (req == MPI_REQUEST_NULL) continue;
      int ierr = MPI_Request_free(&req);
      if (ierr != MPI_SUCCESS) no_errors = false;
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in freeing FC persistent requests" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  pset = FCRemotePersistentSet();
}

void MultigridBoundaryValues::FreeFCRemotePersistentRequests() {
  WaitMGUnpackComplete(kMGFCShell);
  // Every shift is torn down: this is the path taken before a payload reallocation or
  // after a topology change, and both invalidate all of them at once.
  for (int shift = 0; shift < static_cast<int>(fc_remote_persistent_.size()); ++shift) {
    FreeFCRemotePersistentSet(shift);
  }
  for (int slot = 0; slot < 2; ++slot) {
    if (!fc_remote_send_slot_inflight_[slot]) continue;
    // No shift claimed this slot, so nothing waited on its requests.  Callers free
    // requests precisely so the payload can be reallocated; dropping the flag here
    // would let that realloc race an active send instead of crashing.
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "orphaned in-flight FC send slot " << slot << std::endl;
    std::exit(EXIT_FAILURE);
  }
  fc_remote_send_inflight_ = false;
  fc_remote_send_slot_inflight_[0] = false;
  fc_remote_send_slot_inflight_[1] = false;
  fc_remote_send_slot_shift_[0] = -1;
  fc_remote_send_slot_shift_[1] = -1;
  fc_remote_send_pack_slot_ = 0;
}

void MultigridBoundaryValues::FreeMGSameLevelPersistentRequests() {
  if (mg_same_send_inflight_) {
    WaitMGSameLevelSendRequests();
  }
  if (mg_same_recv_inflight_) {
    WaitMGSameLevelRecvRequests();
  }
  if (mg_same_persistent_ready_) {
    bool no_errors = true;
    for (auto &slot_reqs : mg_same_send_req_h_) {
      for (auto &req : slot_reqs) {
        if (req == MPI_REQUEST_NULL) continue;
        int ierr = MPI_Request_free(&req);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    for (auto &slot_reqs : mg_same_recv_req_h_) {
      for (auto &req : slot_reqs) {
        if (req == MPI_REQUEST_NULL) continue;
        int ierr = MPI_Request_free(&req);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in freeing persistent MG requests" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  for (auto &slot_reqs : mg_same_send_req_h_) {
    slot_reqs.clear();
  }
  for (auto &slot_reqs : mg_same_recv_req_h_) {
    slot_reqs.clear();
  }
  mg_same_send_slot_inflight_[0] = false;
  mg_same_send_slot_inflight_[1] = false;
  mg_same_send_inflight_ = false;
  mg_same_recv_inflight_ = false;
  mg_same_persistent_ready_ = false;
  mg_same_send_poll_count_ = 0;
  mg_same_recv_poll_count_ = 0;
  mg_same_send_pack_slot_ = 0;
  mg_same_recv_post_slot_ = 0;
  mg_same_recv_data_slot_ = 0;
}

void MultigridBoundaryValues::WaitMGSameLevelSendRequests() {
  if (!mg_same_send_inflight_) return;
  for (int slot = 0; slot < 2; ++slot) {
    if (!mg_same_send_slot_inflight_[slot]) continue;
    auto &slot_reqs = mg_same_send_req_h_[slot];
    if (slot_reqs.empty()) {
      mg_same_send_slot_inflight_[slot] = false;
      continue;
    }
    int ierr = MPI_Waitall(static_cast<int>(slot_reqs.size()),
                           slot_reqs.data(), MPI_STATUSES_IGNORE);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in waiting aggregated MG sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!mg_same_persistent_ready_) {
      std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
    }
    mg_same_send_slot_inflight_[slot] = false;
  }
  mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0] ||
      mg_same_send_slot_inflight_[1]);
  mg_same_send_poll_count_ = 0;
}

void MultigridBoundaryValues::WaitMGSameLevelRecvRequests() {
  if (!mg_same_recv_inflight_) return;
  auto &slot_reqs = mg_same_recv_req_h_[mg_same_recv_data_slot_];
  if (slot_reqs.empty()) {
    mg_same_recv_inflight_ = false;
    mg_same_recv_poll_count_ = 0;
    return;
  }
  int ierr = MPI_Waitall(static_cast<int>(slot_reqs.size()),
                         slot_reqs.data(), MPI_STATUSES_IGNORE);
  if (ierr != MPI_SUCCESS) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in waiting aggregated MG receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!mg_same_persistent_ready_) {
    std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
  }
  mg_same_recv_inflight_ = false;
  mg_same_recv_poll_count_ = 0;
}
#endif

bool MultigridBoundaryValues::UseMGSameLevelAgg() const {
#if MPI_PARALLEL_ENABLED
  // Same-level aggregation only helps when communicating across ranks.
  // On single-rank runs this path just rebuilds cache metadata and adds
  // unnecessary fences/synchronization overhead.
  return global_variable::nranks > 1;
#else
  return false;
#endif
}

void MultigridBoundaryValues::BuildMGSameLevelAggCache(
    int nvars, bool include_diagonals) {
#if MPI_PARALLEL_ENABLED
  FreeMGSameLevelPersistentRequests();
#endif
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int shift_ = pmy_mg->GetLevelShift();
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;

  std::vector<MGSameEntry> send_entries_h;
  std::vector<MGSameEntry> recv_entries_h;
  send_entries_h.reserve(static_cast<std::size_t>(nmb*nnghbr));
  recv_entries_h.reserve(static_cast<std::size_t>(nmb*nnghbr));

  for (int m = 0; m < nmb; ++m) {
    const int mlev = mblev.h_view(m);
    for (int n = 0; n < nnghbr; ++n) {
      const auto &nb = nghbr.h_view(m, n);
      if (nb.gid < 0 || nb.lev != mlev || nb.rank == my_rank) continue;
      const int face_mask = MGNeighborFaceMask(n);
      const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                      + ((face_mask >> 2) & 1);
      if (!include_diagonals && nface != 1) continue;

      int sil = sendbuf[n].isame[0].bis;
      int siu = sendbuf[n].isame[0].bie;
      int sjl = sendbuf[n].isame[0].bjs;
      int sju = sendbuf[n].isame[0].bje;
      int skl = sendbuf[n].isame[0].bks;
      int sku = sendbuf[n].isame[0].bke;
      AdjustMGSameSendFaceBounds(pmy_mg->GetGhostCells(), shift_, pmy_mg->GetSize(),
                                 (face_mask & 1) != 0,
                                 (face_mask & 2) != 0,
                                 (face_mask & 4) != 0, mg_same_halo_depth_,
                                 sil, siu, sjl, sju, skl, sku);
      int send_size = nvars * MGBoundsCellCount(sil, siu, sjl, sju, skl, sku);
      if (send_size > 0) {
        int lid = nb.gid - pmy_pack->pmesh->gids_eachrank[nb.rank];
        int send_tag = CreateBvals_MPI_Tag(lid, nb.dest);
        send_entries_h.push_back({nb.rank, m, n, send_tag, send_size});
      }

      int dil = recvbuf[n].isame[0].bis;
      int diu = recvbuf[n].isame[0].bie;
      int djl = recvbuf[n].isame[0].bjs;
      int dju = recvbuf[n].isame[0].bje;
      int dkl = recvbuf[n].isame[0].bks;
      int dku = recvbuf[n].isame[0].bke;
      AdjustMGSameRecvFaceBounds(pmy_mg->GetGhostCells(), shift_,
                                 (face_mask & 1) != 0,
                                 (face_mask & 2) != 0,
                                 (face_mask & 4) != 0, mg_same_halo_depth_,
                                 dil, diu, djl, dju, dkl, dku);
      int recv_size = nvars * MGBoundsCellCount(dil, diu, djl, dju, dkl, dku);
      if (recv_size > 0) {
        int recv_tag = CreateBvals_MPI_Tag(m, n);
        recv_entries_h.push_back({nb.rank, m, n, recv_tag, recv_size});
      }
    }
  }

  auto cmp_entry = [](const MGSameEntry &a, const MGSameEntry &b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.tag < b.tag;
  };
  std::sort(send_entries_h.begin(), send_entries_h.end(), cmp_entry);
  std::sort(recv_entries_h.begin(), recv_entries_h.end(), cmp_entry);

  const int nsend_entries = static_cast<int>(send_entries_h.size());
  const int nrecv_entries = static_cast<int>(recv_entries_h.size());
  HostArray1D<int> send_m_h("mg_same_send_m_h", nsend_entries);
  HostArray1D<int> send_off_h("mg_same_send_off_h", nsend_entries);
  HostArray1D<int> recv_m_h("mg_same_recv_m_h", nrecv_entries);
  HostArray1D<int> recv_off_h("mg_same_recv_off_h", nrecv_entries);
  HostArray2D<int> send_bounds_h("mg_same_send_bounds_h", nsend_entries, 6);
  HostArray2D<int> recv_bounds_h("mg_same_recv_bounds_h", nrecv_entries, 6);

  mg_same_send_rank_h_.clear();
  mg_same_send_data_off_h_.clear();
  mg_same_send_tag_h_.clear();
  mg_same_recv_rank_h_.clear();
  mg_same_recv_data_off_h_.clear();
  mg_same_recv_tag_h_.clear();

  constexpr int kMGSameAggTagBufId = 57;

  int send_total = 0;
  for (int s = 0; s < nsend_entries; ++s) {
    const auto &e = send_entries_h[s];
    send_m_h(s) = e.m;
    send_off_h(s) = send_total;
    if (mg_same_send_rank_h_.empty() || mg_same_send_rank_h_.back() != e.rank) {
      mg_same_send_rank_h_.push_back(e.rank);
      mg_same_send_data_off_h_.push_back(send_total);
      mg_same_send_tag_h_.push_back(CreateBvals_MPI_Tag(shift_, kMGSameAggTagBufId));
    }

    int il = sendbuf[e.n].isame[0].bis;
    int iu = sendbuf[e.n].isame[0].bie;
    int jl = sendbuf[e.n].isame[0].bjs;
    int ju = sendbuf[e.n].isame[0].bje;
    int kl = sendbuf[e.n].isame[0].bks;
    int ku = sendbuf[e.n].isame[0].bke;
    const int face_mask = MGNeighborFaceMask(e.n);
    AdjustMGSameSendFaceBounds(pmy_mg->GetGhostCells(), shift_, pmy_mg->GetSize(),
                               (face_mask & 1) != 0,
                               (face_mask & 2) != 0,
                               (face_mask & 4) != 0, mg_same_halo_depth_,
                               il, iu, jl, ju, kl, ku);
    send_bounds_h(s, 0) = il;
    send_bounds_h(s, 1) = iu;
    send_bounds_h(s, 2) = jl;
    send_bounds_h(s, 3) = ju;
    send_bounds_h(s, 4) = kl;
    send_bounds_h(s, 5) = ku;
    send_total += e.size;
  }
  mg_same_send_data_off_h_.push_back(send_total);

  int recv_total = 0;
  for (int r = 0; r < nrecv_entries; ++r) {
    const auto &e = recv_entries_h[r];
    recv_m_h(r) = e.m;
    recv_off_h(r) = recv_total;
    if (mg_same_recv_rank_h_.empty() || mg_same_recv_rank_h_.back() != e.rank) {
      mg_same_recv_rank_h_.push_back(e.rank);
      mg_same_recv_data_off_h_.push_back(recv_total);
      mg_same_recv_tag_h_.push_back(CreateBvals_MPI_Tag(shift_, kMGSameAggTagBufId));
    }

    int il = recvbuf[e.n].isame[0].bis;
    int iu = recvbuf[e.n].isame[0].bie;
    int jl = recvbuf[e.n].isame[0].bjs;
    int ju = recvbuf[e.n].isame[0].bje;
    int kl = recvbuf[e.n].isame[0].bks;
    int ku = recvbuf[e.n].isame[0].bke;
    const int face_mask = MGNeighborFaceMask(e.n);
    AdjustMGSameRecvFaceBounds(pmy_mg->GetGhostCells(), shift_,
                               (face_mask & 1) != 0,
                               (face_mask & 2) != 0,
                               (face_mask & 4) != 0, mg_same_halo_depth_,
                               il, iu, jl, ju, kl, ku);
    recv_bounds_h(r, 0) = il;
    recv_bounds_h(r, 1) = iu;
    recv_bounds_h(r, 2) = jl;
    recv_bounds_h(r, 3) = ju;
    recv_bounds_h(r, 4) = kl;
    recv_bounds_h(r, 5) = ku;
    recv_total += e.size;
  }
  mg_same_recv_data_off_h_.push_back(recv_total);
  mg_same_has_remote_ =
      (!mg_same_send_rank_h_.empty() || !mg_same_recv_rank_h_.empty());

  Kokkos::realloc(mg_same_send_m_d_, nsend_entries);
  Kokkos::realloc(mg_same_send_off_d_, nsend_entries);
  Kokkos::realloc(mg_same_recv_m_d_, nrecv_entries);
  Kokkos::realloc(mg_same_recv_off_d_, nrecv_entries);
  Kokkos::realloc(mg_same_send_bounds_d_, nsend_entries, 6);
  Kokkos::realloc(mg_same_recv_bounds_d_, nrecv_entries, 6);
  Kokkos::realloc(mg_same_send_data_d_[0], send_total);
  Kokkos::realloc(mg_same_send_data_d_[1], send_total);
  Kokkos::realloc(mg_same_recv_data_d_[0], recv_total);
  Kokkos::realloc(mg_same_recv_data_d_[1], recv_total);

  if (nsend_entries > 0) {
    Kokkos::deep_copy(mg_same_send_m_d_, send_m_h);
    Kokkos::deep_copy(mg_same_send_off_d_, send_off_h);
    Kokkos::deep_copy(mg_same_send_bounds_d_, send_bounds_h);
  }
  if (nrecv_entries > 0) {
    Kokkos::deep_copy(mg_same_recv_m_d_, recv_m_h);
    Kokkos::deep_copy(mg_same_recv_off_d_, recv_off_h);
    Kokkos::deep_copy(mg_same_recv_bounds_d_, recv_bounds_h);
  }

#if MPI_PARALLEL_ENABLED
  mg_same_send_req_h_[0].assign(mg_same_send_rank_h_.size(), MPI_REQUEST_NULL);
  mg_same_send_req_h_[1].assign(mg_same_send_rank_h_.size(), MPI_REQUEST_NULL);
  mg_same_recv_req_h_[0].assign(mg_same_recv_rank_h_.size(), MPI_REQUEST_NULL);
  mg_same_recv_req_h_[1].assign(mg_same_recv_rank_h_.size(), MPI_REQUEST_NULL);
  // Aggregated MG buffers are stable between cache rebuilds, so persistent requests
  // avoid per-iteration request setup overhead.
  const bool use_persistent = true;
  if (use_persistent) {
    bool no_errors = true;
    for (int slot = 0; slot < 2; ++slot) {
      for (int sr = 0; sr < static_cast<int>(mg_same_send_rank_h_.size()); ++sr) {
        const int rank = mg_same_send_rank_h_[sr];
        const int tag = mg_same_send_tag_h_[sr];
        const int doff = mg_same_send_data_off_h_[sr];
        const int dcount = mg_same_send_data_off_h_[sr + 1] - doff;
        if (dcount <= 0) continue;
        int ierr = MPI_Send_init(mg_same_send_data_d_[slot].data() + doff, dcount,
                                 MPI_ATHENA_REAL, rank, tag, comm_vars,
                                 &(mg_same_send_req_h_[slot][sr]));
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    for (int slot = 0; slot < 2; ++slot) {
      for (int rr = 0; rr < static_cast<int>(mg_same_recv_rank_h_.size()); ++rr) {
        const int rank = mg_same_recv_rank_h_[rr];
        const int tag = mg_same_recv_tag_h_[rr];
        const int doff = mg_same_recv_data_off_h_[rr];
        const int dcount = mg_same_recv_data_off_h_[rr + 1] - doff;
        if (dcount <= 0) continue;
        int ierr = MPI_Recv_init(mg_same_recv_data_d_[slot].data() + doff, dcount,
                                 MPI_ATHENA_REAL, rank, tag, comm_vars,
                                 &(mg_same_recv_req_h_[slot][rr]));
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in creating persistent MG requests" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mg_same_persistent_ready_ = true;
  } else {
    mg_same_persistent_ready_ = false;
  }
#endif
  mg_same_send_slot_inflight_[0] = false;
  mg_same_send_slot_inflight_[1] = false;
  mg_same_send_inflight_ = false;
  mg_same_recv_inflight_ = false;
  mg_same_send_poll_count_ = 0;
  mg_same_recv_poll_count_ = 0;
  mg_same_send_pack_slot_ = 0;
  mg_same_recv_post_slot_ = 0;
  mg_same_recv_data_slot_ = 0;
}

void MultigridBoundaryValues::EnsureMGSameLevelAggReady(
    int nvars, bool include_diagonals) {
  if (!UseMGSameLevelAgg()) return;
  // Reuse multigrid driver mesh signature to avoid rebuilding a local hash in every
  // boundary call when topology has not changed.
  const std::uint64_t mesh_sig = pmy_mg->pmy_driver_->GetMeshSignature();

  const int shift = pmy_mg->GetLevelShift();
  // The halo depth changes the packed extents, so it selects a cache of its own the
  // same way the diagonal entries and the level shift already do.
  const int cache_key = 4 * shift + 2 * (mg_same_halo_depth_ > 1 ? 1 : 0)
                      + static_cast<int>(include_diagonals);
#if MPI_PARALLEL_ENABLED
  // Nothing below moves a buffer or frees a request when the resident state already is
  // the one asked for, so the common case must not pay for the drain that the rebuild
  // and swap paths below need.  An invalidating mesh signature always clears
  // mg_same_cache_valid_, so it cannot be mistaken for a hit here.
  if (mg_same_level_cache_mesh_sig_ == mesh_sig
      && mg_same_active_cache_key_ == cache_key && mg_same_cache_valid_
      && mg_same_nvars_ == nvars
      && mg_same_include_diagonals_ == include_diagonals) {
    return;
  }
  if (mg_same_level_cache_mesh_sig_ != mesh_sig) {
    InvalidateMGSameLevelAggCache();
    mg_same_level_cache_mesh_sig_ = mesh_sig;
  }

  // A level transition is a synchronization point for the MPI requests, but not for the
  // device: the payload Views are MOVED into and out of the cache, so an unpack still
  // reading one keeps its allocation, and the slot it reads is released by its own event
  // before InitRecvMG posts into that slot again.
  if (mg_same_active_cache_key_ >= 0) {
    WaitMGSameLevelSendRequests();
    WaitMGSameLevelRecvRequests();
    const int old_key = mg_same_active_cache_key_;
    if (static_cast<int>(mg_same_level_cache_.size()) <= old_key) {
      mg_same_level_cache_.resize(old_key + 1);
    }
    SwapMGSameLevelAggState(mg_same_level_cache_[old_key]);
    mg_same_active_cache_key_ = -1;
    mg_same_cache_valid_ = false;
  }

  if (static_cast<int>(mg_same_level_cache_.size()) <= cache_key) {
    mg_same_level_cache_.resize(cache_key + 1);
  }
  auto &state = mg_same_level_cache_[cache_key];
  if (state.valid && state.nvars == nvars) {
    SwapMGSameLevelAggState(state);
  } else {
    // Unlike the swap above, this drops the cached state's Views and reallocates the
    // resident payload, so no unpack of either slot may still be reading one.
    WaitMGUnpackCompleteAll();
    FreeMGSameLevelAggState(state);
    BuildMGSameLevelAggCache(nvars, include_diagonals);
    state.valid = true;
    state.nvars = nvars;
  }

  mg_same_mesh_sig_ = mesh_sig;
  mg_same_shift_ = shift;
  mg_same_nvars_ = nvars;
  mg_same_include_diagonals_ = include_diagonals;
  mg_same_cache_valid_ = true;
  mg_same_active_cache_key_ = cache_key;
#else
  (void)nvars;
  (void)include_diagonals;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::EnsureFCTopologyScan()
//! \brief refresh the cached fine/coarse neighbour census when the mesh has changed
//!
//! Whether this rank has fine/coarse neighbours, whether any of them are off-rank, and
//! whether those are faces or diagonals, are all read off the MeshBlock tree and are the
//! same at every multigrid level.  Answering them cost two nmb*nnghbr host sweeps on
//! every fine/coarse fill; keyed on the mesh signature they cost one sweep per mesh.

void MultigridBoundaryValues::EnsureFCTopologyScan(std::uint64_t mesh_sig, int nmb,
                                                   int nnghbr) {
  if (fc_scan_valid_ && fc_scan_mesh_sig_ == mesh_sig && fc_scan_nmb_ == nmb
      && fc_scan_nnghbr_ == nnghbr) {
    return;
  }
  const int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  // Without the face/diagonal map the remote classification cannot be made; leave it
  // clear and stay invalid, which is what the sweep this replaced did.
  const bool have_nface = (static_cast<int>(fc_ngh_nface_h_.size()) == nnghbr);

  fc_scan_block_mask_h_.assign(static_cast<std::size_t>(nmb), 0);
  fc_scan_has_local_ = false;
  fc_scan_has_remote_ = false;
  fc_scan_has_remote_face_ = false;
  fc_scan_has_remote_diag_ = false;
  for (int m = 0; m < nmb; ++m) {
    const int m_lev = mblev.h_view(m);
    for (int n = 0; n < nnghbr; ++n) {
      const auto &nb = nghbr.h_view(m, n);
      if (nb.gid < 0 || nb.lev == m_lev) continue;
      if (nb.rank == my_rank) {
        fc_scan_has_local_ = true;
        continue;
      }
      fc_scan_has_remote_ = true;
      fc_scan_block_mask_h_[static_cast<std::size_t>(m)] |= 0x1;
      if (!have_nface) continue;
      if (fc_ngh_nface_h_[static_cast<std::size_t>(n)] == 1) {
        fc_scan_has_remote_face_ = true;
      } else {
        fc_scan_has_remote_diag_ = true;
        fc_scan_block_mask_h_[static_cast<std::size_t>(m)] |= 0x2;
      }
    }
  }
  fc_scan_mesh_sig_ = mesh_sig;
  fc_scan_nmb_ = nmb;
  fc_scan_nnghbr_ = nnghbr;
  fc_scan_valid_ = have_nface;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::InvalidateMGTopologyCaches()
//! \brief Drop everything this object has cached about the MeshBlock topology.
//!
//! What is deliberately kept: the compact fine/coarse shell maps, which are a function
//! of the multigrid level shift alone; the flat fine/coarse payloads, which only grow;
//! and every cache gated on the driver's mesh signature, which invalidates itself.  The
//! fine/coarse persistent requests are the exception that has to be torn down by hand.
//! They are rebuilt when a rank grouping or a payload base pointer moves, and a regrid
//! can leave both of those identical while changing the MeshBlock gids that the message
//! tags are built from, which would leave a request listening on the wrong tag.

void MultigridBoundaryValues::InvalidateMGTopologyCaches() {
#if MPI_PARALLEL_ENABLED
  FreeFCRemotePersistentRequests();
  InvalidateMGSameLevelAggCache();
#endif
  mg_single_pair_m_cache_d_.clear();
  mg_single_pair_sm_cache_d_.clear();
  mg_single_pair_desc_cache_d_.clear();
  mg_single_pair_count_cache_.clear();
  mg_single_cached_nmb_cache_.clear();
  mg_single_cached_nnghbr_cache_.clear();
  mg_single_cached_mbgid0_cache_.clear();
  mg_single_cached_nx1_cache_.clear();
  mg_single_cached_mesh_sig_cache_.clear();
  mg_single_pair_valid_cache_.clear();
  fc_ngh_map_nnghbr_ = -1;
  fc_ngh_map_valid_ = false;
  // The census is classified with that map, so it goes with it.
  fc_scan_valid_ = false;
  // These two are gated on the driver's mesh signature as well, but that signature is
  // only recomputed at the head of the next solve, so at this point it can still hold the
  // value the stale caches were built under.  Clear them outright rather than trust it.
  fc_remote_cache_valid_ = false;
  fc_remote_axis_mask_valid_ = false;
  Kokkos::realloc(fc_ngh_ox1_d_, 0);
  Kokkos::realloc(fc_ngh_ox2_d_, 0);
  Kokkos::realloc(fc_ngh_ox3_d_, 0);
  Kokkos::realloc(fc_ngh_f1_d_, 0);
  Kokkos::realloc(fc_ngh_f2_d_, 0);
  Kokkos::realloc(fc_ngh_nface_d_, 0);
  Kokkos::realloc(fc_ngh_valid_d_, 0);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RefreshForNewTopology()
//! \brief Re-point an already-built boundary object at a changed MeshBlock pack.
//!
//! The index metadata is a function of the MeshBlock geometry and the multigrid ghost
//! width, and a regrid changes neither, so there is nothing to recompute.  What a regrid
//! does change is the cached topology and the number of MeshBlocks the payloads must
//! hold.

void MultigridBoundaryValues::RefreshForNewTopology() {
  InvalidateMGTopologyCaches();
  ResizeBuffers(pmy_pack->nmb_thispack);
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridBoundaryValues::RemapIndicesForMG()
//! \brief Rewrite the isame indices InitializeBuffers left in hydro coordinates (ng ghost
//! cells) into multigrid coordinates (ngh_ ghost cells). Must be called AFTER
//! InitializeBuffers.
//!
//! The slabs are rebuilt from each buffer's neighbour offset, with the same formulae
//! MeshBoundaryValuesCC::InitSendIndices / InitRecvIndices use, and not inferred from the
//! hydro index values they currently hold.  The values do not carry the offset: a send
//! slab is [is, is+ng-1] towards a lower neighbour and [ie-ng+1, ie] towards an upper
//! one, and both collapse onto the whole interior [is, ie] once ng reaches nx.  Reading
//! the side back out of them then mistakes every face, edge and corner send of such a
//! block for a whole-interior send, and the exchange packs nx1*nx2*nx3 cells into a
//! buffer the receiver reads as a one-cell-deep ghost slab.

void MultigridBoundaryValues::RemapIndicesForMG() {
  InvalidateMGTopologyCaches();
  int ngh = pmy_mg->GetGhostCells();
  const int nx1 = pmy_pack->pmesh->mb_indcs.nx1;
  const int nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  const int nx3 = pmy_pack->pmesh->mb_indcs.nx3;
  const int is = ngh, ie = ngh + nx1 - 1;
  const int js = ngh, je = ngh + nx2 - 1;
  const int ks = ngh, ke = ngh + nx3 - 1;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  auto send_slab = [ngh](int &lo, int &hi, int ox, int s, int e) {
    if (ox == 0)     { lo = s;           hi = e; }
    else if (ox > 0) { lo = e - ngh + 1; hi = e; }
    else             { lo = s;           hi = s + ngh - 1; }
  };
  auto recv_slab = [ngh](int &lo, int &hi, int ox, int s, int e) {
    if (ox == 0)     { lo = s;       hi = e; }
    else if (ox > 0) { lo = e + 1;   hi = e + ngh; }
    else             { lo = s - ngh; hi = s - 1; }
  };

  // Only the (f1,f2)=(0,0) slot of each face and edge carries a same-level payload,
  // which is exactly the slot NeighborIndex returns for subblock (0,0).  The subblock
  // slots carry none, and InitSendIndices never initialized their isame at all, so give
  // them a nominal one-cell extent before the real slots are written: AllocateAllBuffers
  // below sizes every slot's storage from isame_ndat, and it must not be sized from a
  // count nothing ever set.
  for (int n = 0; n < nnghbr; ++n) {
    for (auto *buf : {&sendbuf[n], &recvbuf[n]}) {
      auto &ib = buf->isame[0];
      ib.bis = ib.bie = is;
      ib.bjs = ib.bje = js;
      ib.bks = ib.bke = ks;
      buf->isame_ndat = 1;
    }
  }

  for (int ox3 = -1; ox3 <= 1; ++ox3) {
    if ((ox3 != 0) && !three_d) continue;
    for (int ox2 = -1; ox2 <= 1; ++ox2) {
      if ((ox2 != 0) && !multi_d) continue;
      for (int ox1 = -1; ox1 <= 1; ++ox1) {
        if ((ox1 == 0) && (ox2 == 0) && (ox3 == 0)) continue;
        const int n = NeighborIndex(ox1, ox2, ox3, 0, 0);
        if ((n < 0) || (n >= nnghbr)) continue;

        auto &si = sendbuf[n].isame[0];
        send_slab(si.bis, si.bie, ox1, is, ie);
        send_slab(si.bjs, si.bje, ox2, js, je);
        send_slab(si.bks, si.bke, ox3, ks, ke);
        sendbuf[n].isame_ndat = (si.bie-si.bis+1)*(si.bje-si.bjs+1)*(si.bke-si.bks+1);

        auto &ri = recvbuf[n].isame[0];
        recv_slab(ri.bis, ri.bie, ox1, is, ie);
        recv_slab(ri.bjs, ri.bje, ox2, js, je);
        recv_slab(ri.bks, ri.bke, ox3, ks, ke);
        recvbuf[n].isame_ndat = (ri.bie-ri.bis+1)*(ri.bje-ri.bjs+1)*(ri.bke-ri.bks+1);
      }
    }
  }

  // The payload storage is sized from isame_ndat alone (same_level_vars_only), and the
  // remap has just rewritten isame_ndat into the multigrid index space, so the storage
  // has to follow it.  The remap itself runs even when the hydro and multigrid ghost
  // widths agree: the same-level slots then come out exactly as InitSendIndices left
  // them, but the subblock slots -- which carry no same-level payload at all, and whose
  // indices are therefore never initialized -- still need their nominal one-cell extent,
  // and metadata that disagrees with the allocation is the kind of thing that goes
  // unnoticed until something finally reads it.
  AllocateAllBuffers(nmb_alloc_);
  // Every base pointer just moved.  The device-side mirror holds copies of them, and any
  // cached rank-packed layout was built against the old ones.
  RefreshDeviceBufferMetadata();
#if MPI_PARALLEL_ENABLED
  InvalidateRankPackedVarMetadata();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts()
//! \brief Fill ghost cells at fine-coarse boundaries.
//! Faces use flux-conserving prolongation/restriction.
//! Edges and corners use injection/restriction.

TaskStatus MultigridBoundaryValues::FillFineCoarseMGGhosts(
    DvceArray5D<Real> &u, bool include_diagonals) {
  if (pmy_mg == nullptr) return TaskStatus::complete;

#if MPI_PARALLEL_ENABLED
  // The previous FC fill may still be reading the persistent receive buffer.
  // Complete that consumer before MPI starts writing the buffer again.
  WaitMGUnpackComplete(kMGFCShell);
#endif

  int nvar = u.extent_int(1);
  int shift = pmy_mg->GetLevelShift();
  int ngh = pmy_mg->GetGhostCells();
  int nx = pmy_mg->GetSize();
  int ncells = nx >> shift;
  const bool use_remote_compact = (ngh == 1);

  // No fine/coarse interpolation exists once a MeshBlock MG level has only one
  // active cell. Avoid building or starting remote shell exchanges at that level.
  if (ncells < 2) return TaskStatus::complete;

  // One compact index map and one persistent-request set per multigrid level shift, so
  // that a V-cycle level change reuses them instead of rebuilding.  The number of levels
  // is fixed for the lifetime of the Multigrid, so this sizes once; it is done here,
  // before any reference into the containers is taken, because a later resize would
  // invalidate those references.
  const int nmg_shifts = std::max(pmy_mg->GetNumberOfLevels(), shift + 1);
  if (static_cast<int>(fc_remote_compact_maps_.size()) < nmg_shifts) {
    fc_remote_compact_maps_.resize(nmg_shifts);
  }
#if MPI_PARALLEL_ENABLED
  if (static_cast<int>(fc_remote_persistent_.size()) < nmg_shifts) {
    fc_remote_persistent_.resize(nmg_shifts);
  }
#endif

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &lloc = pmy_pack->pmesh->lloc_eachmb;

#if MPI_PARALLEL_ENABLED
  constexpr bool gpu_aware_remote_fc_requested = true;
  // FC sends are deferred across calls; poll completion here and only block
  // later if we must reuse send storage.
  if (fc_remote_send_inflight_) {
    fc_remote_send_inflight_ = false;
    for (int slot = 0; slot < 2; ++slot) {
      if (!fc_remote_send_slot_inflight_[slot]) continue;
      FCRemotePersistentSet *pset = FCRemoteSlotSet(slot);
      if (pset == nullptr) {
        // Clearing the flag here would let a later payload realloc race an active send.
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "orphaned in-flight FC send slot " << slot << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (pset->send_req_h[slot].empty()) {
        fc_remote_send_slot_inflight_[slot] = false;
        fc_remote_send_slot_shift_[slot] = -1;
        continue;
      }
      auto &slot_reqs = pset->send_req_h[slot];
      int completed = 0;
      int ierr = MPI_Testall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), &completed, MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in polling deferred FC sends" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (completed != 0) {
        fc_remote_send_slot_inflight_[slot] = false;
        if (!pset->ready) {
          std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
        }
        fc_remote_send_slot_shift_[slot] = -1;
      } else {
        fc_remote_send_inflight_ = true;
      }
    }
  }
#else
  constexpr bool gpu_aware_remote_fc_requested = false;
#endif

  const std::uint64_t mesh_sig = pmy_mg->pmy_driver_->GetMeshSignature();
  if (!fc_ngh_map_valid_ || fc_ngh_map_nnghbr_ != nnghbr
      || fc_ngh_valid_d_.extent_int(0) != nnghbr) {
    HostArray1D<int> ox1_h("mg_fc_ox1_h", nnghbr);
    HostArray1D<int> ox2_h("mg_fc_ox2_h", nnghbr);
    HostArray1D<int> ox3_h("mg_fc_ox3_h", nnghbr);
    HostArray1D<int> f1_h("mg_fc_f1_h", nnghbr);
    HostArray1D<int> f2_h("mg_fc_f2_h", nnghbr);
    HostArray1D<int> nface_h("mg_fc_nface_h", nnghbr);
    HostArray1D<int> valid_h("mg_fc_valid_h", nnghbr);
    for (int n = 0; n < nnghbr; ++n) {
      ox1_h(n) = 0;
      ox2_h(n) = 0;
      ox3_h(n) = 0;
      f1_h(n) = 0;
      f2_h(n) = 0;
      nface_h(n) = 0;
      valid_h(n) = 0;
      bool found = false;
      for (int ox3 = -1; ox3 <= 1 && !found; ++ox3) {
        for (int ox2 = -1; ox2 <= 1 && !found; ++ox2) {
          for (int ox1 = -1; ox1 <= 1 && !found; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            int nface = (ox1 != 0 ? 1 : 0) + (ox2 != 0 ? 1 : 0) + (ox3 != 0 ? 1 : 0);
            for (int f2 = 0; f2 <= 1 && !found; ++f2) {
              for (int f1 = 0; f1 <= 1 && !found; ++f1) {
                int ni = NeighborIndex(ox1, ox2, ox3, f1, f2);
                if (ni != n) continue;
                ox1_h(n) = ox1;
                ox2_h(n) = ox2;
                ox3_h(n) = ox3;
                f1_h(n) = f1;
                f2_h(n) = f2;
                nface_h(n) = nface;
                valid_h(n) = 1;
                found = true;
              }
            }
          }
        }
      }
    }
    Kokkos::realloc(fc_ngh_ox1_d_, nnghbr);
    Kokkos::realloc(fc_ngh_ox2_d_, nnghbr);
    Kokkos::realloc(fc_ngh_ox3_d_, nnghbr);
    Kokkos::realloc(fc_ngh_f1_d_, nnghbr);
    Kokkos::realloc(fc_ngh_f2_d_, nnghbr);
    Kokkos::realloc(fc_ngh_nface_d_, nnghbr);
    Kokkos::realloc(fc_ngh_valid_d_, nnghbr);
    Kokkos::deep_copy(fc_ngh_ox1_d_, ox1_h);
    Kokkos::deep_copy(fc_ngh_ox2_d_, ox2_h);
    Kokkos::deep_copy(fc_ngh_ox3_d_, ox3_h);
    Kokkos::deep_copy(fc_ngh_f1_d_, f1_h);
    Kokkos::deep_copy(fc_ngh_f2_d_, f2_h);
    Kokkos::deep_copy(fc_ngh_nface_d_, nface_h);
    Kokkos::deep_copy(fc_ngh_valid_d_, valid_h);
    fc_ngh_ox1_h_.resize(static_cast<std::size_t>(nnghbr));
    fc_ngh_ox2_h_.resize(static_cast<std::size_t>(nnghbr));
    fc_ngh_ox3_h_.resize(static_cast<std::size_t>(nnghbr));
    fc_ngh_nface_h_.resize(static_cast<std::size_t>(nnghbr));
    for (int n = 0; n < nnghbr; ++n) {
      fc_ngh_ox1_h_[static_cast<std::size_t>(n)] = ox1_h(n);
      fc_ngh_ox2_h_[static_cast<std::size_t>(n)] = ox2_h(n);
      fc_ngh_ox3_h_[static_cast<std::size_t>(n)] = ox3_h(n);
      fc_ngh_nface_h_[static_cast<std::size_t>(n)] = nface_h(n);
    }
    fc_ngh_map_nnghbr_ = nnghbr;
    fc_ngh_map_valid_ = true;
  }

  // The census needs the face/diagonal classification above, so it is taken once the
  // neighbour map is known good rather than at the top of the call.
  EnsureFCTopologyScan(mesh_sig, nmb, nnghbr);
  const bool has_local_fc_neighbors = fc_scan_has_local_;
  const bool has_remote_fc_neighbors = fc_scan_has_remote_;
  const bool has_remote_fc_face_neighbors = fc_scan_has_remote_face_;
  const bool has_remote_fc_diagonal_neighbors = fc_scan_has_remote_diag_;

  int remote_axis_mask = 0x7;
  int local_axis_mask = 0;
  int global_axis_mask = 0;
  const bool can_use_remote_compact =
      (use_remote_compact
       && static_cast<int>(fc_ngh_ox1_h_.size()) == nnghbr
       && static_cast<int>(fc_ngh_ox2_h_.size()) == nnghbr
       && static_cast<int>(fc_ngh_ox3_h_.size()) == nnghbr
       && static_cast<int>(fc_ngh_nface_h_.size()) == nnghbr);
  const bool reuse_cached_axis_mask =
      (use_remote_compact && fc_remote_axis_mask_valid_
       && (fc_remote_axis_mask_mesh_sig_ == mesh_sig)
       && (fc_remote_axis_mask_nnghbr_ == nnghbr)
       && (fc_remote_compact_axis_mask_ > 0));
  if (can_use_remote_compact && !reuse_cached_axis_mask) {
    for (int m = 0; m < nmb; ++m) {
      const int m_lev = mblev.h_view(m);
      for (int n = 0; n < nnghbr; ++n) {
        const auto &nb = nghbr.h_view(m, n);
        if (nb.gid < 0 || nb.lev == m_lev || nb.rank == my_rank) continue;
        if (fc_ngh_nface_h_[static_cast<std::size_t>(n)] != 1) continue;
        if (fc_ngh_ox1_h_[static_cast<std::size_t>(n)] != 0) local_axis_mask |= 0x1;
        if (fc_ngh_ox2_h_[static_cast<std::size_t>(n)] != 0) local_axis_mask |= 0x2;
        if (fc_ngh_ox3_h_[static_cast<std::size_t>(n)] != 0) local_axis_mask |= 0x4;
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  if (use_remote_compact && !reuse_cached_axis_mask) {
    // The compact shell mask is topology metadata. Reduce it once per mesh
    // signature, then reuse the cached value on later refreshes in the same
    // hierarchy.
    global_axis_mask = local_axis_mask;
    int ierr = MPI_Allreduce(&local_axis_mask, &global_axis_mask, 1, MPI_INT, MPI_BOR,
                             comm_vars);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in reducing compact FC axis mask" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#else
  if (use_remote_compact && !reuse_cached_axis_mask) {
    global_axis_mask = local_axis_mask;
  }
#endif
  if (use_remote_compact && !reuse_cached_axis_mask) {
    fc_remote_compact_axis_mask_ = (global_axis_mask != 0) ? global_axis_mask : 0x7;
    if (fc_remote_compact_axis_mask_ == 0) fc_remote_compact_axis_mask_ = 0x7;
    fc_remote_axis_mask_mesh_sig_ = mesh_sig;
    fc_remote_axis_mask_nnghbr_ = nnghbr;
    fc_remote_axis_mask_valid_ = true;
  }
  // The compact-mask reduction above must be entered by every rank in comm_vars,
  // including ranks with no local fine/coarse interfaces.
  if (!has_local_fc_neighbors && !has_remote_fc_neighbors) {
    return TaskStatus::complete;
  }
  if (use_remote_compact && has_remote_fc_face_neighbors
      && static_cast<int>(fc_ngh_ox1_h_.size()) == nnghbr
      && static_cast<int>(fc_ngh_ox2_h_.size()) == nnghbr
      && static_cast<int>(fc_ngh_ox3_h_.size()) == nnghbr
      && static_cast<int>(fc_ngh_nface_h_.size()) == nnghbr) {
    remote_axis_mask = fc_remote_compact_axis_mask_;
    if (remote_axis_mask == 0) remote_axis_mask = 0x7;
  }

  const int ni_tot = u.extent_int(4);
  const int nj_tot = u.extent_int(3);
  const int nk_tot = u.extent_int(2);
  // Remote fine/coarse exchange needs at least one halo cell for flux-conserving
  // stencils.
  const int remote_halo = (ngh > 0) ? 1 : 0;
  const int remote_is = ngh - remote_halo;
  const int remote_ni = ncells + 2*remote_halo;
  const int remote_nj = ncells + 2*remote_halo;
  const int remote_nk = ncells + 2*remote_halo;
  const int remote_full_cell_count = remote_nk * remote_nj * remote_ni;
  int remote_cell_count = remote_full_cell_count;
  int remote_block_size = nvar * remote_cell_count;
  int remote_send_slot = fc_remote_send_pack_slot_;

  DvceArray1D<int> remote_cmp2full_d;
  DvceArray1D<int> remote_full2cmp_d;
  if (use_remote_compact && has_remote_fc_face_neighbors) {
    // The map is keyed on this level shift only. It carries no topology information,
    // so a V-cycle level change swaps between cached maps instead of re-running the
    // O(ni_tot^3) host loop below.
    auto &cmap = fc_remote_compact_maps_[shift];
    bool rebuild_compact_map = (!cmap.valid
        || cmap.ngh != ngh
        || cmap.ncells != ncells
        || cmap.ni_tot != ni_tot
        || cmap.nj_tot != nj_tot
        || cmap.nk_tot != nk_tot
        || cmap.axis_mask != remote_axis_mask);
    if (rebuild_compact_map) {
      const int full_cells = nk_tot * nj_tot * ni_tot;
      HostArray1D<int> full2cmp_h("mg_fc_full2cmp_h", full_cells);
      for (int f = 0; f < full_cells; ++f) full2cmp_h(f) = -1;

      std::vector<int> cmp2full;
      cmp2full.reserve(full_cells);

      // Remote fine/coarse GPU exchange only handles face neighbors. Those stencils
      // sample the boundary-adjacent interior plane on the remote block, but not
      // the remote ghost plane, so keep only those interior planes to minimize
      // PCIe/MPI payload.
      const bool use_x = ((remote_axis_mask & 0x1) != 0);
      const bool use_y = ((remote_axis_mask & 0x2) != 0);
      const bool use_z = ((remote_axis_mask & 0x4) != 0);
      const int band_lo = ngh;
      const int band_hi = ngh + ncells - 1;
      for (int kk = 0; kk < nk_tot; ++kk) {
        for (int jj = 0; jj < nj_tot; ++jj) {
          for (int ii = 0; ii < ni_tot; ++ii) {
            bool near_x = use_x && ((ii == band_lo) || (ii == band_hi));
            bool near_y = use_y && ((jj == band_lo) || (jj == band_hi));
            bool near_z = use_z && ((kk == band_lo) || (kk == band_hi));
            if (!(near_x || near_y || near_z)) continue;

            const int full_idx = (kk*nj_tot + jj)*ni_tot + ii;
            full2cmp_h(full_idx) = static_cast<int>(cmp2full.size());
            cmp2full.push_back(full_idx);
          }
        }
      }

      HostArray1D<int> cmp2full_h("mg_fc_cmp2full_h",
                                  static_cast<int>(cmp2full.size()));
      for (int c = 0; c < static_cast<int>(cmp2full.size()); ++c) {
        cmp2full_h(c) = cmp2full[static_cast<std::size_t>(c)];
      }

      cmap.full2cmp_d = DvceArray1D<int>("mg_fc_full2cmp", full_cells);
      Kokkos::deep_copy(cmap.full2cmp_d, full2cmp_h);
      cmap.cmp2full_d = DvceArray1D<int>("mg_fc_cmp2full",
                                         static_cast<int>(cmp2full.size()));
      Kokkos::deep_copy(cmap.cmp2full_d, cmp2full_h);

      cmap.valid = true;
      cmap.ngh = ngh;
      cmap.ncells = ncells;
      cmap.ni_tot = ni_tot;
      cmap.nj_tot = nj_tot;
      cmap.nk_tot = nk_tot;
      cmap.axis_mask = remote_axis_mask;
      cmap.cell_count = static_cast<int>(cmp2full.size());
    }
    remote_cmp2full_d = cmap.cmp2full_d;
    remote_full2cmp_d = cmap.full2cmp_d;
    remote_cell_count = cmap.cell_count;
    remote_block_size = nvar * remote_cell_count;
  }

  bool gpu_aware_remote_fc = false;
  int nrecv_remote = 0;
  DvceArray1D<Real> recv_data_d = fc_recv_data_d_;
  DvceArray1D<int> send_m_d = fc_send_m_d_;
  DvceArray1D<Real> send_data_d = fc_send_data_d_[remote_send_slot];
  bool cache_same_mesh = (fc_mesh_sig_ == mesh_sig);

#if MPI_PARALLEL_ENABLED
  auto &recv_gid_h = fc_recv_gid_h_;
  auto &recv_rank_h = fc_recv_rank_h_;
  auto &send_m_h = fc_send_m_h_;
  auto &send_gid_h = fc_send_gid_h_;
  auto &send_rank_h = fc_send_rank_h_;
  // Persistent requests for THIS level shift.  fc_remote_persistent_ was sized above and
  // is never resized while this reference is live.
  auto &fc_pset = fc_remote_persistent_[shift];
  auto &gpu_remote_recv_req = fc_pset.recv_req_h;
  int posted_remote_recv = 0;
  bool posted_remote_send = false;
  bool gpu_remote_no_errors = true;

  if (has_remote_fc_face_neighbors && gpu_aware_remote_fc_requested) {
    bool rebuild_remote_cache =
        (!cache_same_mesh || !fc_remote_cache_valid_
         || fc_remote_ridx_d_.extent_int(0) != nmb
         || fc_remote_ridx_d_.extent_int(1) != nnghbr);
    if (rebuild_remote_cache) {
      FreeFCRemotePersistentRequests();
      recv_gid_h.clear();
      recv_rank_h.clear();
      send_m_h.clear();
      send_gid_h.clear();
      send_rank_h.clear();
      fc_recv_rank_list_h_.clear();
      fc_recv_rank_off_h_.clear();
      fc_send_rank_list_h_.clear();
      fc_send_rank_off_h_.clear();
      struct FCRecvEntry {
        int rank;
        int gid;
      };
      struct FCSendEntry {
        int rank;
        int gid;
        int m;
      };
      std::vector<FCRecvEntry> recv_entries_h;
      std::vector<FCSendEntry> send_entries_h;
      std::unordered_map<int, int> recv_gid_seen_h;
      recv_gid_seen_h.reserve(static_cast<std::size_t>(nmb * nnghbr));
      std::unordered_map<std::uint64_t, int> send_pair_seen;
      send_pair_seen.reserve(static_cast<std::size_t>(nmb * nnghbr));
      HostArray2D<int> remote_ridx_h("mg_fc_remote_ridx_h", nmb, nnghbr);

      for (int m = 0; m < nmb; ++m) {
        int m_lev = mblev.h_view(m);
        for (int n = 0; n < nnghbr; ++n) {
          int ridx = -1;
          const auto &nb = nghbr.h_view(m, n);
          if (nb.gid >= 0 && nb.lev != m_lev && nb.rank != my_rank
              && fc_ngh_nface_h_[static_cast<std::size_t>(n)] == 1) {
            int rk = nb.rank;
            int gid = nb.gid;
            if (recv_gid_seen_h.emplace(gid, 1).second) {
              recv_entries_h.push_back({rk, gid});
            }
            std::uint64_t send_key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(m)) << 32) |
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(rk));
            if (send_pair_seen.emplace(send_key, 1).second) {
              send_entries_h.push_back({rk, mbgid.h_view(m), m});
            }
          }
          remote_ridx_h(m, n) = ridx;
        }
      }

      std::sort(recv_entries_h.begin(), recv_entries_h.end(),
                [](const FCRecvEntry &a, const FCRecvEntry &b) {
                  if (a.rank != b.rank) return a.rank < b.rank;
                  return a.gid < b.gid;
                });
      std::sort(send_entries_h.begin(), send_entries_h.end(),
                [](const FCSendEntry &a, const FCSendEntry &b) {
                  if (a.rank != b.rank) return a.rank < b.rank;
                  return a.gid < b.gid;
                });

      recv_gid_h.reserve(recv_entries_h.size());
      recv_rank_h.reserve(recv_entries_h.size());
      for (const auto &e : recv_entries_h) {
        recv_rank_h.push_back(e.rank);
        recv_gid_h.push_back(e.gid);
      }
      send_m_h.reserve(send_entries_h.size());
      send_gid_h.reserve(send_entries_h.size());
      send_rank_h.reserve(send_entries_h.size());
      for (const auto &e : send_entries_h) {
        send_rank_h.push_back(e.rank);
        send_gid_h.push_back(e.gid);
        send_m_h.push_back(e.m);
      }

      std::unordered_map<int, int> recv_gid_to_idx_h;
      recv_gid_to_idx_h.reserve(recv_gid_h.size());
      for (int ridx = 0; ridx < static_cast<int>(recv_gid_h.size()); ++ridx) {
        recv_gid_to_idx_h.emplace(recv_gid_h[ridx], ridx);
      }
      for (int m = 0; m < nmb; ++m) {
        int m_lev = mblev.h_view(m);
        for (int n = 0; n < nnghbr; ++n) {
          int ridx = -1;
          const auto &nb = nghbr.h_view(m, n);
          if (nb.gid >= 0 && nb.lev != m_lev && nb.rank != my_rank
              && fc_ngh_nface_h_[static_cast<std::size_t>(n)] == 1) {
            auto it = recv_gid_to_idx_h.find(nb.gid);
            if (it != recv_gid_to_idx_h.end()) ridx = it->second;
          }
          remote_ridx_h(m, n) = ridx;
        }
      }

      for (int r = 0; r < static_cast<int>(recv_rank_h.size()); ++r) {
        int rk = recv_rank_h[r];
        if (fc_recv_rank_list_h_.empty() || fc_recv_rank_list_h_.back() != rk) {
          fc_recv_rank_list_h_.push_back(rk);
          fc_recv_rank_off_h_.push_back(r);
        }
      }
      fc_recv_rank_off_h_.push_back(static_cast<int>(recv_rank_h.size()));
      for (int s = 0; s < static_cast<int>(send_rank_h.size()); ++s) {
        int rk = send_rank_h[s];
        if (fc_send_rank_list_h_.empty() || fc_send_rank_list_h_.back() != rk) {
          fc_send_rank_list_h_.push_back(rk);
          fc_send_rank_off_h_.push_back(s);
        }
      }
      fc_send_rank_off_h_.push_back(static_cast<int>(send_rank_h.size()));

      Kokkos::realloc(fc_remote_ridx_d_, nmb, nnghbr);
      Kokkos::deep_copy(fc_remote_ridx_d_, remote_ridx_h);
      fc_remote_cache_valid_ = true;
      fc_mesh_sig_ = mesh_sig;
      cache_same_mesh = true;
    }

    gpu_aware_remote_fc = true;
  }

  if (gpu_aware_remote_fc && has_remote_fc_face_neighbors) {
    nrecv_remote = static_cast<int>(recv_gid_h.size());
    int nsend_remote = static_cast<int>(send_m_h.size());

    // Size the flat payload from the FINEST level so that a V-cycle level change never
    // reallocates: remote_block_size shrinks with the level, and reallocating on every
    // change moved .data() and re-registered every persistent request ~8 times per
    // V-cycle.  A level's active extent is nx>>shift, so the shift-0 totals follow from
    // this level's.
    const int fine_ni = ((ni_tot - 2*ngh) << shift) + 2*ngh;
    const int fine_nj = ((nj_tot - 2*ngh) << shift) + 2*ngh;
    const int fine_nk = ((nk_tot - 2*ngh) << shift) + 2*ngh;
    int fine_cell_count = 0;
    if (use_remote_compact) {
      // Closed form of the compact shell built above: the whole box minus the interior
      // that is left once the two kept planes are dropped on each active axis.
      const int ax = ((remote_axis_mask & 0x1) != 0) ? 2 : 0;
      const int ay = ((remote_axis_mask & 0x2) != 0) ? 2 : 0;
      const int az = ((remote_axis_mask & 0x4) != 0) ? 2 : 0;
      fine_cell_count = fine_ni*fine_nj*fine_nk
                        - (fine_ni - ax)*(fine_nj - ay)*(fine_nk - az);
    } else {
      fine_cell_count = (fine_ni - 2*ngh + 2*remote_halo)
                        * (fine_nj - 2*ngh + 2*remote_halo)
                        * (fine_nk - 2*ngh + 2*remote_halo);
    }
    // The closed form is only a sizing hint; taking the max with the level actually
    // being packed keeps the buffer correct even if it ever under-predicts.
    const int remote_block_size_max = std::max(nvar*fine_cell_count, remote_block_size);
    const std::int64_t recv_need = static_cast<std::int64_t>(nrecv_remote)
                                   * static_cast<std::int64_t>(remote_block_size_max);
    const std::int64_t send_need = static_cast<std::int64_t>(nsend_remote)
                                   * static_cast<std::int64_t>(remote_block_size_max);
    const std::int64_t recv_cap = FCRemoteBufCapacity(recv_need);
    const std::int64_t send_cap = FCRemoteBufCapacity(send_need);
    if (recv_cap > std::numeric_limits<int>::max()
        || send_cap > std::numeric_limits<int>::max()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "remote FC payload exceeds the addressable buffer size"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Grow only.  Shrinking would move .data() and force every persistent request bound
    // to it to be torn down and re-registered, which is exactly the churn these buffers
    // exist to avoid; adding a shrink path needs a large hysteresis band, see the
    // established pattern in mesh/mb_storage.hpp.
    const bool grow_recv_buf =
        (static_cast<std::int64_t>(fc_recv_data_d_.extent_int(0)) < recv_need);
    const bool resize_send_m = (fc_send_m_d_.extent_int(0) != nsend_remote);
    const bool grow_send_buf =
        (static_cast<std::int64_t>(fc_send_data_d_[0].extent_int(0)) < send_need
         || static_cast<std::int64_t>(fc_send_data_d_[1].extent_int(0)) < send_need);
    if (grow_recv_buf || resize_send_m || grow_send_buf) {
      // Drains every in-flight send and drops every persistent request BEFORE the
      // reallocations below move .data(). A persistent request left on a freed pointer
      // does not fault, it silently corrupts memory.
      FreeFCRemotePersistentRequests();
    }
    if (grow_recv_buf) {
      fc_recv_data_d_ = DvceArray1D<Real>("mg_fc_recv_data", static_cast<int>(recv_cap));
    }
    if (fc_remote_send_slot_inflight_[remote_send_slot]) {
      const int alt_slot = 1 - remote_send_slot;
      if (!fc_remote_send_slot_inflight_[alt_slot]) {
        remote_send_slot = alt_slot;
      } else {
        // Both payload slots are still being read by MPI. Drain them before repacking;
        // the persistent requests survive, a completed one is simply started again.
        // Keeping them is only legal because no reallocation can follow on this path:
        // the guard above already drained both slots whenever anything grows.  Pin that
        // reasoning down so a later edit cannot quietly invalidate it.
        assert(!grow_recv_buf && !grow_send_buf && !resize_send_m);
        WaitFCRemoteSendSlots();
        remote_send_slot = 0;
      }
    }
    if (resize_send_m) {
      Kokkos::realloc(fc_send_m_d_, nsend_remote);
    }
    if (grow_send_buf) {
      fc_send_data_d_[0] =
          DvceArray1D<Real>("mg_fc_send_data0", static_cast<int>(send_cap));
      fc_send_data_d_[1] =
          DvceArray1D<Real>("mg_fc_send_data1", static_cast<int>(send_cap));
    }
    recv_data_d = fc_recv_data_d_;
    send_m_d = fc_send_m_d_;
    send_data_d = fc_send_data_d_[remote_send_slot];

    constexpr int kCFTagBufId = 56;

    const int nrecv_rank_groups = static_cast<int>(fc_recv_rank_list_h_.size());
    const int nsend_rank_groups = static_cast<int>(fc_send_rank_list_h_.size());
    // This shift's requests are rebuilt only when something they are bound to has
    // actually moved: a payload base pointer, this level's block size, or the rank
    // grouping (which rbeg/rcount are derived from).
    const bool rebuild_remote_persistent =
        (!fc_pset.ready
         || fc_pset.block_size != remote_block_size
         || fc_pset.recv_base != fc_recv_data_d_.data()
         || fc_pset.send_base[0] != fc_send_data_d_[0].data()
         || fc_pset.send_base[1] != fc_send_data_d_[1].data()
         || static_cast<int>(gpu_remote_recv_req.size()) != nrecv_rank_groups
         || static_cast<int>(fc_pset.send_req_h[0].size()) != nsend_rank_groups
         || static_cast<int>(fc_pset.send_req_h[1].size()) != nsend_rank_groups
         || fc_pset.recv_rank_list_h != fc_recv_rank_list_h_
         || fc_pset.recv_rank_off_h != fc_recv_rank_off_h_
         || fc_pset.send_rank_list_h != fc_send_rank_list_h_
         || fc_pset.send_rank_off_h != fc_send_rank_off_h_);
    if (rebuild_remote_persistent) {
      FreeFCRemotePersistentSet(shift);
      gpu_remote_recv_req.assign(nrecv_rank_groups, MPI_REQUEST_NULL);
      fc_pset.send_req_h[0].assign(nsend_rank_groups, MPI_REQUEST_NULL);
      fc_pset.send_req_h[1].assign(nsend_rank_groups, MPI_REQUEST_NULL);
      for (int rr = 0; rr < nrecv_rank_groups; ++rr) {
        int rk = fc_recv_rank_list_h_[rr];
        int rbeg = fc_recv_rank_off_h_[rr];
        int rend = fc_recv_rank_off_h_[rr + 1];
        int rcount = rend - rbeg;
        if (rcount <= 0) continue;
        int lid = recv_gid_h[rbeg] - pmy_pack->pmesh->gids_eachrank[rk];
        int tag = CreateBvals_MPI_Tag(lid, kCFTagBufId);
        int ierr = MPI_Recv_init(recv_data_d.data() + rbeg*remote_block_size,
                                 rcount*remote_block_size, MPI_ATHENA_REAL, rk, tag,
                                 comm_vars, &gpu_remote_recv_req[rr]);
        if (ierr != MPI_SUCCESS) gpu_remote_no_errors = false;
      }
      for (int slot = 0; slot < 2; ++slot) {
        auto &slot_send_req = fc_pset.send_req_h[slot];
        for (int ss = 0; ss < nsend_rank_groups; ++ss) {
          int rk = fc_send_rank_list_h_[ss];
          int sbeg = fc_send_rank_off_h_[ss];
          int send_end = fc_send_rank_off_h_[ss + 1];
          int scount = send_end - sbeg;
          if (scount <= 0) continue;
          int lid = send_gid_h[sbeg] - pmy_pack->pmesh->gids_eachrank[my_rank];
          int tag = CreateBvals_MPI_Tag(lid, kCFTagBufId);
          int ierr = MPI_Send_init(fc_send_data_d_[slot].data() + sbeg*remote_block_size,
                                   scount*remote_block_size, MPI_ATHENA_REAL, rk, tag,
                                   comm_vars, &slot_send_req[ss]);
          if (ierr != MPI_SUCCESS) gpu_remote_no_errors = false;
        }
      }
      if (!gpu_remote_no_errors) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in creating persistent FC requests"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Record exactly what these requests are bound to, so the guard above can detect
      // any later move of the payload or regrouping of the ranks.
      fc_pset.ready = true;
      fc_pset.block_size = remote_block_size;
      fc_pset.recv_base = fc_recv_data_d_.data();
      fc_pset.send_base[0] = fc_send_data_d_[0].data();
      fc_pset.send_base[1] = fc_send_data_d_[1].data();
      fc_pset.recv_rank_list_h = fc_recv_rank_list_h_;
      fc_pset.recv_rank_off_h = fc_recv_rank_off_h_;
      fc_pset.send_rank_list_h = fc_send_rank_list_h_;
      fc_pset.send_rank_off_h = fc_send_rank_off_h_;
      gpu_remote_no_errors = true;
    }

    if (fc_pset.ready) {
      if (nrecv_rank_groups > 0) {
        int ierr = MPI_Startall(nrecv_rank_groups, gpu_remote_recv_req.data());
        if (ierr == MPI_SUCCESS) posted_remote_recv = nrecv_rank_groups;
        else
          gpu_remote_no_errors = false;
      }
    } else {
      for (int rr = 0; rr < nrecv_rank_groups; ++rr) {
        int rk = fc_recv_rank_list_h_[rr];
        int rbeg = fc_recv_rank_off_h_[rr];
        int rend = fc_recv_rank_off_h_[rr + 1];
        int rcount = rend - rbeg;
        if (rcount <= 0) continue;
        int lid = recv_gid_h[rbeg] - pmy_pack->pmesh->gids_eachrank[rk];
        int tag = CreateBvals_MPI_Tag(lid, kCFTagBufId);
        int ierr = MPI_Irecv(recv_data_d.data() + rbeg*remote_block_size,
                             rcount*remote_block_size, MPI_ATHENA_REAL, rk, tag, comm_vars,
                             &gpu_remote_recv_req[rr]);
        if (ierr != MPI_SUCCESS) {
          gpu_remote_no_errors = false;
        } else {
          ++posted_remote_recv;
        }
      }
    }

    if (nsend_remote > 0) {
      HostArray1D<int> send_m_hv("mg_fc_send_m_hv", nsend_remote);
      for (int s = 0; s < nsend_remote; ++s) send_m_hv(s) = send_m_h[s];
      Kokkos::deep_copy(send_m_d, send_m_hv);

      DevExeSpace fc_exec;
      if (use_remote_compact) {
        auto cmp2full = remote_cmp2full_d;
        const int cell_count = remote_cell_count;
        const int ni_tot_l = ni_tot;
        const int nj_tot_l = nj_tot;
        par_for("MGPackRemoteFCSendCompact", fc_exec, 0,
            nsend_remote*remote_block_size - 1,
        KOKKOS_LAMBDA(const int idx) {
          const int s = idx / remote_block_size;
          int off = idx - s*remote_block_size;
          const int v = off / cell_count;
          const int c = off - v*cell_count;
          const int full = cmp2full(c);
          const int ii = full % ni_tot_l;
          const int tmp = full / ni_tot_l;
          const int jj = tmp % nj_tot_l;
          const int kk = tmp / nj_tot_l;
          const int m = send_m_d(s);
          // idx already is the flat entry-major offset into the payload.
          send_data_d(idx) = u(m, v, kk, jj, ii);
        });
      } else {
        par_for("MGPackRemoteFCSend", fc_exec, 0, nsend_remote*remote_block_size - 1,
        KOKKOS_LAMBDA(const int idx) {
          const int s = idx / remote_block_size;
          int off = idx - s*remote_block_size;
          const int i = off % remote_ni;
          off /= remote_ni;
          const int j = off % remote_nj;
          off /= remote_nj;
          const int k = off % remote_nk;
          const int v = off / remote_nk;
          const int ii = remote_is + i;
          const int jj = remote_is + j;
          const int kk = remote_is + k;
          const int m = send_m_d(s);
          // idx already is the flat entry-major offset into the payload.
          send_data_d(idx) = u(m, v, kk, jj, ii);
        });
      }
      fc_exec.fence();
    }

    if (fc_pset.ready) {
      if (nsend_rank_groups > 0) {
        int ierr = MPI_Startall(nsend_rank_groups,
                                fc_pset.send_req_h[remote_send_slot].data());
        if (ierr != MPI_SUCCESS) gpu_remote_no_errors = false;
        else
          posted_remote_send = true;
      }
    } else {
      for (int ss = 0; ss < nsend_rank_groups; ++ss) {
        int rk = fc_send_rank_list_h_[ss];
        int sbeg = fc_send_rank_off_h_[ss];
        int send_end = fc_send_rank_off_h_[ss + 1];
        int scount = send_end - sbeg;
        if (scount <= 0) continue;
        int lid = send_gid_h[sbeg] - pmy_pack->pmesh->gids_eachrank[my_rank];
        int tag = CreateBvals_MPI_Tag(lid, kCFTagBufId);
        int ierr = MPI_Isend(send_data_d.data() + sbeg*remote_block_size,
                             scount*remote_block_size, MPI_ATHENA_REAL, rk, tag, comm_vars,
                             &fc_pset.send_req_h[remote_send_slot][ss]);
        if (ierr != MPI_SUCCESS) gpu_remote_no_errors = false;
        posted_remote_send = true;
      }
    }
    fc_remote_send_slot_inflight_[remote_send_slot] = posted_remote_send;
    // The slot is shared by all shifts, so record which shift's requests own it.
    fc_remote_send_slot_shift_[remote_send_slot] = posted_remote_send ? shift : -1;
    fc_remote_send_inflight_ = (fc_remote_send_slot_inflight_[0]
                                || fc_remote_send_slot_inflight_[1]);
    if (posted_remote_send) {
      fc_remote_send_pack_slot_ = 1 - remote_send_slot;
    }
  }
#endif

  if (has_local_fc_neighbors || gpu_aware_remote_fc) {
    auto nghbr_d = nghbr.d_view;
    auto mblev_d = mblev.d_view;
    auto child_x_d = pmy_mg->fc_childx_;
    auto child_y_d = pmy_mg->fc_childy_;
    auto child_z_d = pmy_mg->fc_childz_;
    const int mbgid0 = mbgid.h_view(0);
    const int my_rank_l = my_rank;
    const int ngh_l = ngh;
    const int ncells_l = ncells;
    const int nvar_l = nvar;
    // Ghost cells per (MeshBlock, neighbour, variable) in the fill kernels below.  Which
    // of the four branches a thread lands in is not known until it has read the
    // neighbour, so the range has to cover the widest of them -- a face, which walks the
    // half-block plane the neighbour covers, or an edge, whose tangential span is the
    // whole block -- and a thread past its own branch's extent returns.
    const int nt_l = std::max((ncells/2)*(ncells/2),
                              std::max(ncells*ngh*ngh, ngh*ngh*ngh));
    const bool gpu_remote_l = gpu_aware_remote_fc;
    const int remote_is_l = remote_is;
    const int remote_ni_l = remote_ni;
    const int remote_nj_l = remote_nj;
    const int remote_nk_l = remote_nk;
    const bool use_remote_compact_l = use_remote_compact;
    const int remote_cell_count_l = remote_cell_count;
    // Stride of one entry in the flat payload; the receive buffer may be sized for a
    // finer level, so entries are addressed explicitly rather than by view extent.
    const int remote_block_size_l = remote_block_size;
    const int ni_tot_l = ni_tot;
    const int nj_tot_l = nj_tot;
    const int nnghbr_l = nnghbr;
    auto remote_ridx = fc_remote_ridx_d_;
    auto recv_data = recv_data_d;
    auto remote_full2cmp = remote_full2cmp_d;
    auto fc_ox1_d = fc_ngh_ox1_d_;
    auto fc_ox2_d = fc_ngh_ox2_d_;
    auto fc_ox3_d = fc_ngh_ox3_d_;
    auto fc_f1_d = fc_ngh_f1_d_;
    auto fc_f2_d = fc_ngh_f2_d_;
    auto fc_nface_d = fc_ngh_nface_d_;
    auto fc_valid_d = fc_ngh_valid_d_;
    constexpr Real ot = 1.0/3.0;
    const Real tg_l = pmy_mg->pmy_driver_->FCTangentialGradient() ? 1.0 : 0.0;

    auto launch_fc_fill_local_kernel = [&]() {
      par_for("MGFillFineCoarseLocalOnly", DevExeSpace(), 0, nmb*nnghbr_l - 1,
      0, nvar_l - 1, 0, nt_l - 1,
      KOKKOS_LAMBDA(const int mn, const int v, const int t) {
      const int m = mn / nnghbr_l;
      const int n = mn - m*nnghbr_l;
      if (fc_valid_d(n) == 0) return;
      const int ox1 = fc_ox1_d(n);
      const int ox2 = fc_ox2_d(n);
      const int ox3 = fc_ox3_d(n);
      const int f1 = fc_f1_d(n);
      const int f2 = fc_f2_d(n);
      const int nface = fc_nface_d(n);
      int m_lev = mblev_d(m);
      int child_x = child_x_d(m);
      int child_y = child_y_d(m);
      int child_z = child_z_d(m);
      int half = ncells_l / 2;
      auto nb = nghbr_d(m, n);
      if (nb.gid < 0) return;
      if (nb.lev == m_lev) return;
      if (nb.rank != my_rank_l) return;
      const int dm = nb.gid - mbgid0;
      if (dm < 0 || dm >= nmb) return;

      auto nval = [&](int vv, int kk, int jj, int ii) -> Real {
        return u(dm, vv, kk, jj, ii);
      };

      if (nb.lev < m_lev && nface == 1) {
        if (ox1 != 0) {
          int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int sj0 = ngh_l + child_y * half;
          int sk0 = ngh_l + child_z * half;
          if (t >= half*half) return;
          const int sk = sk0 + t/half;
          const int sj = sj0 + t%half;
          int fj = ngh_l + 2*(sj - sj0);
          int fk = ngh_l + 2*(sk - sk0);
          Real cc = nval(v, sk, sj, si);
          int sjm = (sj > ngh_l) ? sj - 1 : sj;
          int sjp = (sj < ngh_l + ncells_l - 1) ? sj + 1 : sj;
          int skm = (sk > ngh_l) ? sk - 1 : sk;
          int skp = (sk < ngh_l + ncells_l - 1) ? sk + 1 : sk;
          Real gy = tg_l*0.125*(nval(v, sk, sjp, si) - nval(v, sk, sjm, si));
          Real gz = tg_l*0.125*(nval(v, skp, sj, si) - nval(v, skm, sj, si));
          u(m, v, fk,   fj,   fig) = ot*(2.0*(cc-gy-gz) + u(m, v, fk,   fj,   fi));
          u(m, v, fk,   fj+1, fig) = ot*(2.0*(cc+gy-gz) + u(m, v, fk,   fj+1, fi));
          u(m, v, fk+1, fj,   fig) = ot*(2.0*(cc-gy+gz) + u(m, v, fk+1, fj,   fi));
          u(m, v, fk+1, fj+1, fig) = ot*(2.0*(cc+gy+gz) + u(m, v, fk+1, fj+1, fi));
        } else if (ox2 != 0) {
          int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int si0 = ngh_l + child_x * half;
          int sk0 = ngh_l + child_z * half;
          if (t >= half*half) return;
          const int sk = sk0 + t/half;
          const int si = si0 + t%half;
          int fi = ngh_l + 2*(si - si0);
          int fk = ngh_l + 2*(sk - sk0);
          Real cc = nval(v, sk, sj, si);
          int sim = (si > ngh_l) ? si - 1 : si;
          int sip = (si < ngh_l + ncells_l - 1) ? si + 1 : si;
          int skm = (sk > ngh_l) ? sk - 1 : sk;
          int skp = (sk < ngh_l + ncells_l - 1) ? sk + 1 : sk;
          Real gx = tg_l*0.125*(nval(v, sk, sj, sip) - nval(v, sk, sj, sim));
          Real gz = tg_l*0.125*(nval(v, skp, sj, si) - nval(v, skm, sj, si));
          u(m, v, fk,   fjg, fi)   = ot*(2.0*(cc-gx-gz) + u(m, v, fk,   fj, fi));
          u(m, v, fk,   fjg, fi+1) = ot*(2.0*(cc+gx-gz) + u(m, v, fk,   fj, fi+1));
          u(m, v, fk+1, fjg, fi)   = ot*(2.0*(cc-gx+gz) + u(m, v, fk+1, fj, fi));
          u(m, v, fk+1, fjg, fi+1) = ot*(2.0*(cc+gx+gz) + u(m, v, fk+1, fj, fi+1));
        } else {
          int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int si0 = ngh_l + child_x * half;
          int sj0 = ngh_l + child_y * half;
          if (t >= half*half) return;
          const int sj = sj0 + t/half;
          const int si = si0 + t%half;
          int fi = ngh_l + 2*(si - si0);
          int fj = ngh_l + 2*(sj - sj0);
          Real cc = nval(v, sk, sj, si);
          int sim = (si > ngh_l) ? si - 1 : si;
          int sip = (si < ngh_l + ncells_l - 1) ? si + 1 : si;
          int sjm = (sj > ngh_l) ? sj - 1 : sj;
          int sjp = (sj < ngh_l + ncells_l - 1) ? sj + 1 : sj;
          Real gx = tg_l*0.125*(nval(v, sk, sj, sip) - nval(v, sk, sj, sim));
          Real gy = tg_l*0.125*(nval(v, sk, sjp, si) - nval(v, sk, sjm, si));
          u(m, v, fkg, fj,   fi)   = ot*(2.0*(cc-gx-gy) + u(m, v, fk, fj,   fi));
          u(m, v, fkg, fj,   fi+1) = ot*(2.0*(cc+gx-gy) + u(m, v, fk, fj,   fi+1));
          u(m, v, fkg, fj+1, fi)   = ot*(2.0*(cc-gx+gy) + u(m, v, fk, fj+1, fi));
          u(m, v, fkg, fj+1, fi+1) = ot*(2.0*(cc+gx+gy) + u(m, v, fk, fj+1, fi+1));
        }
      } else if (nb.lev > m_lev && nface == 1) {
        int sub_x = 0, sub_y = 0, sub_z = 0;
        if (ox1 != 0) { sub_y = f1; sub_z = f2; }
        if (ox2 != 0) { sub_x = f1; sub_z = f2; }
        if (ox3 != 0) { sub_x = f1; sub_y = f2; }

        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else { gis = ngh_l + sub_x*half; gie = ngh_l + sub_x*half + half - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else { gjs = ngh_l + sub_y*half; gje = ngh_l + sub_y*half + half - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else { gks = ngh_l + sub_z*half; gke = ngh_l + sub_z*half + half - 1; }

        int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
        int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
        int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

        if (ox1 != 0) {
          int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int njg = gje - gjs + 1;
          if (t >= njg*(gke - gks + 1)) return;
          const int gk = gks + t/njg;
          const int gj = gjs + t%njg;
          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
          Real favg = 0.25*(nval(v, fk0,   fj0,   fi) + nval(v, fk0,   fj0+1, fi)
                           +nval(v, fk0+1, fj0,   fi) + nval(v, fk0+1, fj0+1, fi));
          u(m, v, gk, gj, gis) = ot*(4.0*favg - u(m, v, gk+ok, gj+oj, gis+oi));
        } else if (ox2 != 0) {
          int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int nig = gie - gis + 1;
          if (t >= nig*(gke - gks + 1)) return;
          const int gk = gks + t/nig;
          const int gi = gis + t%nig;
          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
          Real favg = 0.25*(nval(v, fk0,   fj, fi0) + nval(v, fk0,   fj, fi0+1)
                           +nval(v, fk0+1, fj, fi0) + nval(v, fk0+1, fj, fi0+1));
          u(m, v, gk, gjs, gi) = ot*(4.0*favg - u(m, v, gk+ok, gjs+oj, gi+oi));
        } else {
          int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int nig = gie - gis + 1;
          if (t >= nig*(gje - gjs + 1)) return;
          const int gj = gjs + t/nig;
          const int gi = gis + t%nig;
          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
          Real favg = 0.25*(nval(v, fk, fj0,   fi0) + nval(v, fk, fj0,   fi0+1)
                           +nval(v, fk, fj0+1, fi0) + nval(v, fk, fj0+1, fi0+1));
          u(m, v, gks, gj, gi) = ot*(4.0*favg - u(m, v, gks+ok, gj+oj, gi+oi));
        }
      } else if (nb.lev < m_lev) {
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else              { gis = ngh_l;            gie = ngh_l + ncells_l - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else              { gjs = ngh_l;            gje = ngh_l + ncells_l - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else              { gks = ngh_l;            gke = ngh_l + ncells_l - 1; }

        const int nig = gie - gis + 1;
        const int njg = gje - gjs + 1;
        if (t >= nig*njg*(gke - gks + 1)) return;
        const int gk = gks + t/(nig*njg);
        const int gj = gjs + (t/nig)%njg;
        const int gi = gis + t%nig;
        int si, sj, sk;
        if (ox1 < 0)      si = ngh_l + ncells_l - 1;
        else if (ox1 > 0) si = ngh_l;
        else              si = ngh_l + child_x*half + (gi - ngh_l)/2;
        if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
        else if (ox2 > 0) sj = ngh_l;
        else              sj = ngh_l + child_y*half + (gj - ngh_l)/2;
        if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
        else if (ox3 > 0) sk = ngh_l;
        else              sk = ngh_l + child_z*half + (gk - ngh_l)/2;
        u(m, v, gk, gj, gi) = nval(v, sk, sj, si);
      } else {
        int sub_x = 0, sub_y = 0, sub_z = 0;
        if (nface == 2) {
          if (ox1 == 0) sub_x = f1;
          if (ox2 == 0) sub_y = f1;
          if (ox3 == 0) sub_z = f1;
        }
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else { gis = ngh_l + sub_x*half; gie = ngh_l + sub_x*half + half - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else { gjs = ngh_l + sub_y*half; gje = ngh_l + sub_y*half + half - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else { gks = ngh_l + sub_z*half; gke = ngh_l + sub_z*half + half - 1; }

        const int nig = gie - gis + 1;
        const int njg = gje - gjs + 1;
        if (t >= nig*njg*(gke - gks + 1)) return;
        const int gk = gks + t/(nig*njg);
        const int gj = gjs + (t/nig)%njg;
        const int gi = gis + t%nig;
        int fi0, fi1, fj0, fj1, fk0, fk1;
        if (ox1 < 0) { fi0 = ngh_l + ncells_l - 2; fi1 = ngh_l + ncells_l - 1; }
        else if (ox1 > 0) { fi0 = ngh_l; fi1 = ngh_l + 1; }
        else { fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half)); fi1 = fi0 + 1; }
        if (ox2 < 0) { fj0 = ngh_l + ncells_l - 2; fj1 = ngh_l + ncells_l - 1; }
        else if (ox2 > 0) { fj0 = ngh_l; fj1 = ngh_l + 1; }
        else { fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half)); fj1 = fj0 + 1; }
        if (ox3 < 0) { fk0 = ngh_l + ncells_l - 2; fk1 = ngh_l + ncells_l - 1; }
        else if (ox3 > 0) { fk0 = ngh_l; fk1 = ngh_l + 1; }
        else { fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half)); fk1 = fk0 + 1; }
        u(m, v, gk, gj, gi) = 0.125*(
          nval(v, fk0, fj0, fi0) + nval(v, fk0, fj0, fi1)
        + nval(v, fk0, fj1, fi0) + nval(v, fk0, fj1, fi1)
        + nval(v, fk1, fj0, fi0) + nval(v, fk1, fj0, fi1)
        + nval(v, fk1, fj1, fi0) + nval(v, fk1, fj1, fi1));
      }
      });
    };

    auto launch_fc_fill_kernel = [&](bool remote_phase, int remote_ridx_beg,
                                     int remote_ridx_end) {
      const bool remote_phase_l = remote_phase;
      const int remote_ridx_beg_l = remote_ridx_beg;
      const int remote_ridx_end_l = remote_ridx_end;
      par_for("MGFillFineCoarseLocal", DevExeSpace(), 0, nmb*nnghbr_l - 1,
      0, nvar_l - 1, 0, nt_l - 1,
      KOKKOS_LAMBDA(const int mn, const int v, const int t) {
      const int m = mn / nnghbr_l;
      const int n = mn - m*nnghbr_l;
      if (fc_valid_d(n) == 0) return;
      const int ox1 = fc_ox1_d(n);
      const int ox2 = fc_ox2_d(n);
      const int ox3 = fc_ox3_d(n);
      const int f1 = fc_f1_d(n);
      const int f2 = fc_f2_d(n);
      const int nface = fc_nface_d(n);
      int m_lev = mblev_d(m);
      int child_x = child_x_d(m);
      int child_y = child_y_d(m);
      int child_z = child_z_d(m);
      int half = ncells_l / 2;
      auto nb = nghbr_d(m, n);
      if (nb.gid < 0) return;
      if (nb.lev == m_lev) return;
      int dm = -1;
      int ridx = -1;
      if (nb.rank == my_rank_l) {
        if (remote_phase_l) return;
        dm = nb.gid - mbgid0;
        if (dm < 0 || dm >= nmb) return;
      } else {
        if (!gpu_remote_l || !remote_phase_l) return;
        ridx = remote_ridx(m, n);
        if (ridx < 0) return;
        if (ridx < remote_ridx_beg_l || ridx >= remote_ridx_end_l) return;
      }

      auto nval = [&](int vv, int kk, int jj, int ii) -> Real {
        if (ridx >= 0) {
          int rk = kk - remote_is_l;
          int rj = jj - remote_is_l;
          int ri = ii - remote_is_l;
          if (rk < 0) rk = 0;
          if (rj < 0) rj = 0;
          if (ri < 0) ri = 0;
          if (rk >= remote_nk_l) rk = remote_nk_l - 1;
          if (rj >= remote_nj_l) rj = remote_nj_l - 1;
          if (ri >= remote_ni_l) ri = remote_ni_l - 1;
          if (use_remote_compact_l) {
            int cidx = remote_full2cmp((rk*nj_tot_l + rj)*ni_tot_l + ri);
            if (cidx < 0) cidx = 0;
            return recv_data(ridx*remote_block_size_l + vv*remote_cell_count_l + cidx);
          }
          const int idx = ((vv*remote_nk_l + rk)*remote_nj_l + rj)*remote_ni_l + ri;
          return recv_data(ridx*remote_block_size_l + idx);
        }
        return u(dm, vv, kk, jj, ii);
      };

      if (nb.lev < m_lev && nface == 1) {
        if (ox1 != 0) {
          int fig = (ox1 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fi  = (ox1 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int si  = (ox1 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int sj0 = ngh_l + child_y * half;
          int sk0 = ngh_l + child_z * half;
          if (t >= half*half) return;
          const int sk = sk0 + t/half;
          const int sj = sj0 + t%half;
          int fj = ngh_l + 2*(sj - sj0);
          int fk = ngh_l + 2*(sk - sk0);
          Real cc = nval(v, sk, sj, si);
          int sjm = (sj > ngh_l) ? sj - 1 : sj;
          int sjp = (sj < ngh_l + ncells_l - 1) ? sj + 1 : sj;
          int skm = (sk > ngh_l) ? sk - 1 : sk;
          int skp = (sk < ngh_l + ncells_l - 1) ? sk + 1 : sk;
          Real gy = tg_l*0.125*(nval(v, sk, sjp, si) - nval(v, sk, sjm, si));
          Real gz = tg_l*0.125*(nval(v, skp, sj, si) - nval(v, skm, sj, si));
          u(m, v, fk,   fj,   fig) = ot*(2.0*(cc-gy-gz) + u(m, v, fk,   fj,   fi));
          u(m, v, fk,   fj+1, fig) = ot*(2.0*(cc+gy-gz) + u(m, v, fk,   fj+1, fi));
          u(m, v, fk+1, fj,   fig) = ot*(2.0*(cc-gy+gz) + u(m, v, fk+1, fj,   fi));
          u(m, v, fk+1, fj+1, fig) = ot*(2.0*(cc+gy+gz) + u(m, v, fk+1, fj+1, fi));
        } else if (ox2 != 0) {
          int fjg = (ox2 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fj  = (ox2 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int sj  = (ox2 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int si0 = ngh_l + child_x * half;
          int sk0 = ngh_l + child_z * half;
          if (t >= half*half) return;
          const int sk = sk0 + t/half;
          const int si = si0 + t%half;
          int fi = ngh_l + 2*(si - si0);
          int fk = ngh_l + 2*(sk - sk0);
          Real cc = nval(v, sk, sj, si);
          int sim = (si > ngh_l) ? si - 1 : si;
          int sip = (si < ngh_l + ncells_l - 1) ? si + 1 : si;
          int skm = (sk > ngh_l) ? sk - 1 : sk;
          int skp = (sk < ngh_l + ncells_l - 1) ? sk + 1 : sk;
          Real gx = tg_l*0.125*(nval(v, sk, sj, sip) - nval(v, sk, sj, sim));
          Real gz = tg_l*0.125*(nval(v, skp, sj, si) - nval(v, skm, sj, si));
          u(m, v, fk,   fjg, fi)   = ot*(2.0*(cc-gx-gz) + u(m, v, fk,   fj, fi));
          u(m, v, fk,   fjg, fi+1) = ot*(2.0*(cc+gx-gz) + u(m, v, fk,   fj, fi+1));
          u(m, v, fk+1, fjg, fi)   = ot*(2.0*(cc-gx+gz) + u(m, v, fk+1, fj, fi));
          u(m, v, fk+1, fjg, fi+1) = ot*(2.0*(cc+gx+gz) + u(m, v, fk+1, fj, fi+1));
        } else {
          int fkg = (ox3 < 0) ? ngh_l - 1 : ngh_l + ncells_l;
          int fk  = (ox3 < 0) ? ngh_l : ngh_l + ncells_l - 1;
          int sk  = (ox3 < 0) ? ngh_l + ncells_l - 1 : ngh_l;
          int si0 = ngh_l + child_x * half;
          int sj0 = ngh_l + child_y * half;
          if (t >= half*half) return;
          const int sj = sj0 + t/half;
          const int si = si0 + t%half;
          int fi = ngh_l + 2*(si - si0);
          int fj = ngh_l + 2*(sj - sj0);
          Real cc = nval(v, sk, sj, si);
          int sim = (si > ngh_l) ? si - 1 : si;
          int sip = (si < ngh_l + ncells_l - 1) ? si + 1 : si;
          int sjm = (sj > ngh_l) ? sj - 1 : sj;
          int sjp = (sj < ngh_l + ncells_l - 1) ? sj + 1 : sj;
          Real gx = tg_l*0.125*(nval(v, sk, sj, sip) - nval(v, sk, sj, sim));
          Real gy = tg_l*0.125*(nval(v, sk, sjp, si) - nval(v, sk, sjm, si));
          u(m, v, fkg, fj,   fi)   = ot*(2.0*(cc-gx-gy) + u(m, v, fk, fj,   fi));
          u(m, v, fkg, fj,   fi+1) = ot*(2.0*(cc+gx-gy) + u(m, v, fk, fj,   fi+1));
          u(m, v, fkg, fj+1, fi)   = ot*(2.0*(cc-gx+gy) + u(m, v, fk, fj+1, fi));
          u(m, v, fkg, fj+1, fi+1) = ot*(2.0*(cc+gx+gy) + u(m, v, fk, fj+1, fi+1));
        }
      } else if (nb.lev > m_lev && nface == 1) {
        int sub_x = 0, sub_y = 0, sub_z = 0;
        if (ox1 != 0) { sub_y = f1; sub_z = f2; }
        if (ox2 != 0) { sub_x = f1; sub_z = f2; }
        if (ox3 != 0) { sub_x = f1; sub_y = f2; }

        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else { gis = ngh_l + sub_x*half; gie = ngh_l + sub_x*half + half - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else { gjs = ngh_l + sub_y*half; gje = ngh_l + sub_y*half + half - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else { gks = ngh_l + sub_z*half; gke = ngh_l + sub_z*half + half - 1; }

        int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
        int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
        int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

        if (ox1 != 0) {
          int fi = (ox1 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int njg = gje - gjs + 1;
          if (t >= njg*(gke - gks + 1)) return;
          const int gk = gks + t/njg;
          const int gj = gjs + t%njg;
          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
          Real favg = 0.25*(nval(v, fk0,   fj0,   fi) + nval(v, fk0,   fj0+1, fi)
                           +nval(v, fk0+1, fj0,   fi) + nval(v, fk0+1, fj0+1, fi));
          u(m, v, gk, gj, gis) = ot*(4.0*favg - u(m, v, gk+ok, gj+oj, gis+oi));
        } else if (ox2 != 0) {
          int fj = (ox2 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int nig = gie - gis + 1;
          if (t >= nig*(gke - gks + 1)) return;
          const int gk = gks + t/nig;
          const int gi = gis + t%nig;
          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
          int fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half));
          Real favg = 0.25*(nval(v, fk0,   fj, fi0) + nval(v, fk0,   fj, fi0+1)
                           +nval(v, fk0+1, fj, fi0) + nval(v, fk0+1, fj, fi0+1));
          u(m, v, gk, gjs, gi) = ot*(4.0*favg - u(m, v, gk+ok, gjs+oj, gi+oi));
        } else {
          int fk = (ox3 > 0) ? ngh_l : ngh_l + ncells_l - 1;
          const int nig = gie - gis + 1;
          if (t >= nig*(gje - gjs + 1)) return;
          const int gj = gjs + t/nig;
          const int gi = gis + t%nig;
          int fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half));
          int fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half));
          Real favg = 0.25*(nval(v, fk, fj0,   fi0) + nval(v, fk, fj0,   fi0+1)
                           +nval(v, fk, fj0+1, fi0) + nval(v, fk, fj0+1, fi0+1));
          u(m, v, gks, gj, gi) = ot*(4.0*favg - u(m, v, gks+ok, gj+oj, gi+oi));
        }
      } else if (nb.lev < m_lev) {
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else              { gis = ngh_l;            gie = ngh_l + ncells_l - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else              { gjs = ngh_l;            gje = ngh_l + ncells_l - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else              { gks = ngh_l;            gke = ngh_l + ncells_l - 1; }

        const int nig = gie - gis + 1;
        const int njg = gje - gjs + 1;
        if (t >= nig*njg*(gke - gks + 1)) return;
        const int gk = gks + t/(nig*njg);
        const int gj = gjs + (t/nig)%njg;
        const int gi = gis + t%nig;
        int si, sj, sk;
        if (ox1 < 0)      si = ngh_l + ncells_l - 1;
        else if (ox1 > 0) si = ngh_l;
        else              si = ngh_l + child_x*half + (gi - ngh_l)/2;
        if (ox2 < 0)      sj = ngh_l + ncells_l - 1;
        else if (ox2 > 0) sj = ngh_l;
        else              sj = ngh_l + child_y*half + (gj - ngh_l)/2;
        if (ox3 < 0)      sk = ngh_l + ncells_l - 1;
        else if (ox3 > 0) sk = ngh_l;
        else              sk = ngh_l + child_z*half + (gk - ngh_l)/2;
        u(m, v, gk, gj, gi) = nval(v, sk, sj, si);
      } else {
        int sub_x = 0, sub_y = 0, sub_z = 0;
        if (nface == 2) {
          if (ox1 == 0) sub_x = f1;
          if (ox2 == 0) sub_y = f1;
          if (ox3 == 0) sub_z = f1;
        }
        int gis, gie, gjs, gje, gks, gke;
        if (ox1 < 0)      { gis = 0;               gie = ngh_l - 1; }
        else if (ox1 > 0) { gis = ngh_l + ncells_l; gie = ngh_l + ncells_l + ngh_l - 1; }
        else { gis = ngh_l + sub_x*half; gie = ngh_l + sub_x*half + half - 1; }
        if (ox2 < 0)      { gjs = 0;               gje = ngh_l - 1; }
        else if (ox2 > 0) { gjs = ngh_l + ncells_l; gje = ngh_l + ncells_l + ngh_l - 1; }
        else { gjs = ngh_l + sub_y*half; gje = ngh_l + sub_y*half + half - 1; }
        if (ox3 < 0)      { gks = 0;               gke = ngh_l - 1; }
        else if (ox3 > 0) { gks = ngh_l + ncells_l; gke = ngh_l + ncells_l + ngh_l - 1; }
        else { gks = ngh_l + sub_z*half; gke = ngh_l + sub_z*half + half - 1; }

        const int nig = gie - gis + 1;
        const int njg = gje - gjs + 1;
        if (t >= nig*njg*(gke - gks + 1)) return;
        const int gk = gks + t/(nig*njg);
        const int gj = gjs + (t/nig)%njg;
        const int gi = gis + t%nig;
        int fi0, fi1, fj0, fj1, fk0, fk1;
        if (ox1 < 0) { fi0 = ngh_l + ncells_l - 2; fi1 = ngh_l + ncells_l - 1; }
        else if (ox1 > 0) { fi0 = ngh_l; fi1 = ngh_l + 1; }
        else { fi0 = ngh_l + 2*(gi - (ngh_l + sub_x*half)); fi1 = fi0 + 1; }
        if (ox2 < 0) { fj0 = ngh_l + ncells_l - 2; fj1 = ngh_l + ncells_l - 1; }
        else if (ox2 > 0) { fj0 = ngh_l; fj1 = ngh_l + 1; }
        else { fj0 = ngh_l + 2*(gj - (ngh_l + sub_y*half)); fj1 = fj0 + 1; }
        if (ox3 < 0) { fk0 = ngh_l + ncells_l - 2; fk1 = ngh_l + ncells_l - 1; }
        else if (ox3 > 0) { fk0 = ngh_l; fk1 = ngh_l + 1; }
        else { fk0 = ngh_l + 2*(gk - (ngh_l + sub_z*half)); fk1 = fk0 + 1; }
        u(m, v, gk, gj, gi) = 0.125*(
          nval(v, fk0, fj0, fi0) + nval(v, fk0, fj0, fi1)
        + nval(v, fk0, fj1, fi0) + nval(v, fk0, fj1, fi1)
        + nval(v, fk1, fj0, fi0) + nval(v, fk1, fj0, fi1)
        + nval(v, fk1, fj1, fi0) + nval(v, fk1, fj1, fi1));
      }
      });
    };

    if (has_local_fc_neighbors) {
      launch_fc_fill_local_kernel();
    }

    bool remote_fc_applied = false;
#if MPI_PARALLEL_ENABLED
    if (gpu_aware_remote_fc && has_remote_fc_face_neighbors) {
      if (gpu_remote_no_errors && posted_remote_recv > 0) {
        int ierr = MPI_Waitall(static_cast<int>(gpu_remote_recv_req.size()),
                               gpu_remote_recv_req.data(), MPI_STATUSES_IGNORE);
        if (ierr != MPI_SUCCESS) {
          gpu_remote_no_errors = false;
        } else {
          launch_fc_fill_kernel(true, 0, nrecv_remote);
          remote_fc_applied = true;
        }
      }
      if (!gpu_remote_no_errors) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in GPU-aware remote fine/coarse MG exchange"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
#endif

    if (gpu_aware_remote_fc && !remote_fc_applied) {
      launch_fc_fill_kernel(true, 0, nrecv_remote);
    }

#if MPI_PARALLEL_ENABLED
    if (gpu_aware_remote_fc && nrecv_remote > 0) {
      MarkMGUnpackPending(kMGFCShell);
    }
#endif

#if MPI_PARALLEL_ENABLED
    if (gpu_aware_remote_fc && has_remote_fc_face_neighbors) {
      // Send completion is intentionally deferred to the next call where request
      // slots/storage are reused, allowing overlap with intervening compute.
    }
#endif
  }

  const bool need_host_remote_faces = has_remote_fc_face_neighbors && !gpu_aware_remote_fc;
  const bool need_host_remote_diagonals =
      include_diagonals && has_remote_fc_diagonal_neighbors;
  if (!need_host_remote_faces && !need_host_remote_diagonals) {
    return TaskStatus::complete;
  }

  if (fc_diag_host_mirror_.extent(0) != u.extent(0)
      || fc_diag_host_mirror_.extent(1) != u.extent(1)
      || fc_diag_host_mirror_.extent(2) != u.extent(2)
      || fc_diag_host_mirror_.extent(3) != u.extent(3)
      || fc_diag_host_mirror_.extent(4) != u.extent(4)) {
    fc_diag_host_mirror_ = HostArray5D<Real>("mg_fc_diag_host", u.extent(0), u.extent(1),
                                             u.extent(2), u.extent(3), u.extent(4));
  }
  auto u_h = fc_diag_host_mirror_;
  // Only this path reads the per-block census, and it runs on a small minority of calls,
  // so the list is assembled here rather than kept alive across the whole function.
  std::vector<int> need_pull_remote(static_cast<std::size_t>(nmb), 0);
  int pulled_blocks = 0;
  for (int m = 0; m < nmb; ++m) {
    const int mask = fc_scan_block_mask_h_[static_cast<std::size_t>(m)];
    if ((!gpu_aware_remote_fc_requested && ((mask & 0x1) != 0))
        || (include_diagonals && ((mask & 0x2) != 0))) {
      need_pull_remote[static_cast<std::size_t>(m)] = 1;
      ++pulled_blocks;
    }
  }
  if (pulled_blocks == nmb) {
    Kokkos::deep_copy(u_h, u);
  } else {
    for (int m = 0; m < nmb; ++m) {
      if (need_pull_remote[m] == 0) continue;
      Kokkos::deep_copy(Kokkos::subview(u_h, m, Kokkos::ALL(), Kokkos::ALL(),
                                        Kokkos::ALL(), Kokkos::ALL()),
                        Kokkos::subview(u, m, Kokkos::ALL(), Kokkos::ALL(),
                                        Kokkos::ALL(), Kokkos::ALL()));
    }
  }
#if MPI_PARALLEL_ENABLED
  struct RemoteRecvBlock {
    int gid;
    int rank;
    std::vector<Real> data;
    MPI_Request req;
  };
  struct RemoteSendBlock {
    int m;
    int gid;
    int rank;
    std::vector<Real> data;
    MPI_Request req;
  };

  std::vector<RemoteRecvBlock> recv_blocks;
  std::vector<RemoteSendBlock> send_blocks;

  auto has_recv = [&recv_blocks](int gid) {
    for (const auto &rb : recv_blocks) {
      if (rb.gid == gid) return true;
    }
    return false;
  };
  auto has_send = [&send_blocks](int m, int rank) {
    for (const auto &sb : send_blocks) {
      if (sb.m == m && sb.rank == rank) return true;
    }
    return false;
  };

  // Exchange full block data only for remote fine/coarse neighbors.
  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev.h_view(m);
    for (int n = 0; n < nnghbr; ++n) {
      if (nghbr.h_view(m, n).gid < 0) continue;
      if (nghbr.h_view(m, n).lev == m_lev) continue;
      int rk = nghbr.h_view(m, n).rank;
      if (rk == my_rank) continue;
      const int nface = fc_ngh_nface_h_[static_cast<std::size_t>(n)];
      if (gpu_aware_remote_fc && nface == 1) continue;
      int gid = nghbr.h_view(m, n).gid;
      if (!has_recv(gid)) {
        recv_blocks.push_back({gid, rk, {}, MPI_REQUEST_NULL});
      }
      if (!has_send(m, rk)) {
        send_blocks.push_back({m, mbgid.h_view(m), rk, {}, MPI_REQUEST_NULL});
      }
    }
  }

  if (!recv_blocks.empty() || !send_blocks.empty()) {
    // The tag must depend only on the collective exchange mode.  Whether a rank
    // also has remote face neighbors is local topology and can differ by peer.
    const int tag_buf_id = include_diagonals ? 58 : 56;
    int block_size = nvar * nk_tot * nj_tot * ni_tot;
    bool no_errors = true;

    for (auto &rb : recv_blocks) {
      rb.data.resize(block_size);
      int lid = rb.gid - pmy_pack->pmesh->gids_eachrank[rb.rank];
      int tag = CreateBvals_MPI_Tag(lid, tag_buf_id);
      int ierr = MPI_Irecv(rb.data.data(), block_size, MPI_ATHENA_REAL, rb.rank, tag,
                           comm_vars, &(rb.req));
      if (ierr != MPI_SUCCESS) no_errors = false;
    }

    for (auto &sb : send_blocks) {
      sb.data.resize(block_size);
      int idx = 0;
      for (int v = 0; v < nvar; ++v) {
        for (int k = 0; k < nk_tot; ++k) {
          for (int j = 0; j < nj_tot; ++j) {
            for (int i = 0; i < ni_tot; ++i) {
              sb.data[idx++] = u_h(sb.m, v, k, j, i);
            }
          }
        }
      }
      int lid = sb.gid - pmy_pack->pmesh->gids_eachrank[my_rank];
      int tag = CreateBvals_MPI_Tag(lid, tag_buf_id);
      int ierr = MPI_Isend(sb.data.data(), block_size, MPI_ATHENA_REAL, sb.rank, tag,
                           comm_vars, &(sb.req));
      if (ierr != MPI_SUCCESS) no_errors = false;
    }

    for (auto &rb : recv_blocks) {
      if (rb.req != MPI_REQUEST_NULL) {
        int ierr = MPI_Wait(&(rb.req), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }
    for (auto &sb : send_blocks) {
      if (sb.req != MPI_REQUEST_NULL) {
        int ierr = MPI_Wait(&(sb.req), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) no_errors = false;
      }
    }

    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in remote fine/coarse MG exchange"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  auto NeighborValue = [&](int dm, const std::vector<Real> *remote_data, int v,
                           int k, int j, int i) -> Real {
#if MPI_PARALLEL_ENABLED
    if (remote_data != nullptr) {
      int idx = ((v*nk_tot + k)*nj_tot + j)*ni_tot + i;
      return (*remote_data)[idx];
    }
#endif
    return u_h(dm, v, k, j, i);
  };

  bool modified = false;
  std::vector<int> modified_blocks(nmb, 0);
  constexpr Real ot = 1.0/3.0;
  const Real tg = pmy_mg->pmy_driver_->FCTangentialGradient() ? 1.0 : 0.0;

  for (int m = 0; m < nmb; ++m) {
    int m_lev = mblev.h_view(m);
    int m_gid = mbgid.h_view(m);
    LogicalLocation m_loc = lloc[m_gid];

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          int nface = (ox1!=0?1:0) + (ox2!=0?1:0) + (ox3!=0?1:0);

          for (int f2 = 0; f2 <= 1; ++f2) {
            for (int f1 = 0; f1 <= 1; ++f1) {
              int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
              if (n < 0 || n >= nnghbr) continue;
              if (nghbr.h_view(m, n).gid < 0) continue;

              int nlev = nghbr.h_view(m, n).lev;
              if (nlev == m_lev) continue;
              if (gpu_aware_remote_fc && nface == 1) continue;
              int ngid = nghbr.h_view(m, n).gid;
              int nrank = nghbr.h_view(m, n).rank;
              bool remote_neighbor = (nrank != my_rank);
              if (!remote_neighbor) continue;

              int dm = -1;
              const std::vector<Real> *remote_data = nullptr;
              if (remote_neighbor) {
#if MPI_PARALLEL_ENABLED
                for (const auto &rb : recv_blocks) {
                  if (rb.gid == ngid) {
                    remote_data = &(rb.data);
                    break;
                  }
                }
                if (remote_data == nullptr) continue;
#else
                continue;
#endif
              } else {
                dm = ngid - mbgid.h_view(0);
                if (dm < 0 || dm >= nmb) continue;
              }

              int child_x = m_loc.lx1 & 1;
              int child_y = m_loc.lx2 & 1;
              int child_z = m_loc.lx3 & 1;
              int half = ncells / 2;

              if (nlev < m_lev && nface == 1) {
                // ==== COARSER neighbor, FACE: flux-conserving prolongation ====
                if (ox1 != 0) {
                  int fig = (ox1 < 0) ? ngh - 1 : ngh + ncells;
                  int fi  = (ox1 < 0) ? ngh : ngh + ncells - 1;
                  int si  = (ox1 < 0) ? ngh + ncells - 1 : ngh;
                  int sj0 = ngh + child_y * half;
                  int sk0 = ngh + child_z * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sk = sk0; sk < sk0 + half; ++sk) {
                      for (int sj = sj0; sj < sj0 + half; ++sj) {
                        int fj = ngh + 2*(sj - sj0);
                        int fk = ngh + 2*(sk - sk0);
                        Real cc = NeighborValue(dm, remote_data, v, sk, sj, si);
                        int sjm = (sj > ngh) ? sj - 1 : sj;
                        int sjp = (sj < ngh + ncells - 1) ? sj + 1 : sj;
                        int skm = (sk > ngh) ? sk - 1 : sk;
                        int skp = (sk < ngh + ncells - 1) ? sk + 1 : sk;
                        Real gy = tg*0.125*(NeighborValue(dm, remote_data, v, sk, sjp, si)
                                       - NeighborValue(dm, remote_data, v, sk, sjm, si));
                        Real gz = tg*0.125*(NeighborValue(dm, remote_data, v, skp, sj, si)
                                       - NeighborValue(dm, remote_data, v, skm, sj, si));
                        u_h(m,v,fk  ,fj  ,fig)=ot*(2.0*(cc-gy-gz)+u_h(m,v,fk  ,fj  ,fi));
                        u_h(m,v,fk  ,fj+1,fig)=ot*(2.0*(cc+gy-gz)+u_h(m,v,fk  ,fj+1,fi));
                        u_h(m,v,fk+1,fj  ,fig)=ot*(2.0*(cc-gy+gz)+u_h(m,v,fk+1,fj  ,fi));
                        u_h(m,v,fk+1,fj+1,fig)=ot*(2.0*(cc+gy+gz)+u_h(m,v,fk+1,fj+1,fi));
                      }
                    }
                  }
                } else if (ox2 != 0) {
                  int fjg = (ox2 < 0) ? ngh - 1 : ngh + ncells;
                  int fj  = (ox2 < 0) ? ngh : ngh + ncells - 1;
                  int sj  = (ox2 < 0) ? ngh + ncells - 1 : ngh;
                  int si0 = ngh + child_x * half;
                  int sk0 = ngh + child_z * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sk = sk0; sk < sk0 + half; ++sk) {
                      for (int si = si0; si < si0 + half; ++si) {
                        int fi = ngh + 2*(si - si0);
                        int fk = ngh + 2*(sk - sk0);
                        Real cc = NeighborValue(dm, remote_data, v, sk, sj, si);
                        int sim = (si > ngh) ? si - 1 : si;
                        int sip = (si < ngh + ncells - 1) ? si + 1 : si;
                        int skm = (sk > ngh) ? sk - 1 : sk;
                        int skp = (sk < ngh + ncells - 1) ? sk + 1 : sk;
                        Real gx = tg*0.125*(NeighborValue(dm, remote_data, v, sk, sj, sip)
                                       - NeighborValue(dm, remote_data, v, sk, sj, sim));
                        Real gz = tg*0.125*(NeighborValue(dm, remote_data, v, skp, sj, si)
                                       - NeighborValue(dm, remote_data, v, skm, sj, si));
                        u_h(m,v,fk  ,fjg,fi  )=ot*(2.0*(cc-gx-gz)+u_h(m,v,fk  ,fj,fi  ));
                        u_h(m,v,fk  ,fjg,fi+1)=ot*(2.0*(cc+gx-gz)+u_h(m,v,fk  ,fj,fi+1));
                        u_h(m,v,fk+1,fjg,fi  )=ot*(2.0*(cc-gx+gz)+u_h(m,v,fk+1,fj,fi  ));
                        u_h(m,v,fk+1,fjg,fi+1)=ot*(2.0*(cc+gx+gz)+u_h(m,v,fk+1,fj,fi+1));
                      }
                    }
                  }
                } else {
                  int fkg = (ox3 < 0) ? ngh - 1 : ngh + ncells;
                  int fk  = (ox3 < 0) ? ngh : ngh + ncells - 1;
                  int sk  = (ox3 < 0) ? ngh + ncells - 1 : ngh;
                  int si0 = ngh + child_x * half;
                  int sj0 = ngh + child_y * half;
                  for (int v = 0; v < nvar; ++v) {
                    for (int sj = sj0; sj < sj0 + half; ++sj) {
                      for (int si = si0; si < si0 + half; ++si) {
                        int fi = ngh + 2*(si - si0);
                        int fj = ngh + 2*(sj - sj0);
                        Real cc = NeighborValue(dm, remote_data, v, sk, sj, si);
                        int sim = (si > ngh) ? si - 1 : si;
                        int sip = (si < ngh + ncells - 1) ? si + 1 : si;
                        int sjm = (sj > ngh) ? sj - 1 : sj;
                        int sjp = (sj < ngh + ncells - 1) ? sj + 1 : sj;
                        Real gx = tg*0.125*(NeighborValue(dm, remote_data, v, sk, sj, sip)
                                       - NeighborValue(dm, remote_data, v, sk, sj, sim));
                        Real gy = tg*0.125*(NeighborValue(dm, remote_data, v, sk, sjp, si)
                                       - NeighborValue(dm, remote_data, v, sk, sjm, si));
                        u_h(m,v,fkg,fj  ,fi  )=ot*(2.0*(cc-gx-gy)+u_h(m,v,fk,fj  ,fi  ));
                        u_h(m,v,fkg,fj  ,fi+1)=ot*(2.0*(cc+gx-gy)+u_h(m,v,fk,fj  ,fi+1));
                        u_h(m,v,fkg,fj+1,fi  )=ot*(2.0*(cc-gx+gy)+u_h(m,v,fk,fj+1,fi  ));
                        u_h(m,v,fkg,fj+1,fi+1)=ot*(2.0*(cc+gx+gy)+u_h(m,v,fk,fj+1,fi+1));
                      }
                    }
                  }
                }
                modified = true;
                modified_blocks[m] = 1;

              } else if (nlev > m_lev && nface == 1) {
                // ==== FINER neighbor, FACE: flux-conserving restriction ====
                // face_avg = area-average of 4 fine cells on the shared face
                // coarse_ghost = (1/3)*(4*face_avg - coarse_interior)
                int sub_x = 0, sub_y = 0, sub_z = 0;
                if (ox1 != 0) { sub_y = f1; sub_z = f2; }
                if (ox2 != 0) { sub_x = f1; sub_z = f2; }
                if (ox3 != 0) { sub_x = f1; sub_y = f2; }

                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else { gis = ngh + sub_x*half; gie = ngh + sub_x*half + half - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else { gjs = ngh + sub_y*half; gje = ngh + sub_y*half + half - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else { gks = ngh + sub_z*half; gke = ngh + sub_z*half + half - 1; }

                int oi = (ox1 < 0) ? 1 : (ox1 > 0) ? -1 : 0;
                int oj = (ox2 < 0) ? 1 : (ox2 > 0) ? -1 : 0;
                int ok = (ox3 < 0) ? 1 : (ox3 > 0) ? -1 : 0;

                if (ox1 != 0) {
                  int fi = (ox1 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gj = gjs; gj <= gje; ++gj) {
                        int fj0 = ngh + 2*(gj - (ngh + sub_y*half));
                        int fk0 = ngh + 2*(gk - (ngh + sub_z*half));
                        Real favg = 0.25*(NeighborValue(dm, remote_data, v, fk0, fj0, fi)
                                         +NeighborValue(dm, remote_data, v, fk0, fj0+1,
                                             fi)
                                         +NeighborValue(dm, remote_data, v, fk0+1, fj0,
                                             fi)
                                         +NeighborValue(dm, remote_data, v, fk0+1, fj0+1,
                                             fi));
                        u_h(m,v,gk,gj,gis) = ot*(4.0*favg - u_h(m,v,gk+ok,gj+oj,gis+oi));
                      }
                    }
                  }
                } else if (ox2 != 0) {
                  int fj = (ox2 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gk = gks; gk <= gke; ++gk) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0 = ngh + 2*(gi - (ngh + sub_x*half));
                        int fk0 = ngh + 2*(gk - (ngh + sub_z*half));
                        Real favg = 0.25*(NeighborValue(dm, remote_data, v, fk0, fj, fi0)
                                         +NeighborValue(dm, remote_data, v, fk0, fj,
                                             fi0+1)
                                         +NeighborValue(dm, remote_data, v, fk0+1, fj,
                                             fi0)
                                         +NeighborValue(dm, remote_data, v, fk0+1, fj,
                                             fi0+1));
                        u_h(m,v,gk,gjs,gi) = ot*(4.0*favg - u_h(m,v,gk+ok,gjs+oj,gi+oi));
                      }
                    }
                  }
                } else {
                  int fk = (ox3 > 0) ? ngh : ngh + ncells - 1;
                  for (int v = 0; v < nvar; ++v) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0 = ngh + 2*(gi - (ngh + sub_x*half));
                        int fj0 = ngh + 2*(gj - (ngh + sub_y*half));
                        Real favg = 0.25*(NeighborValue(dm, remote_data, v, fk, fj0, fi0)
                                         +NeighborValue(dm, remote_data, v, fk, fj0,
                                             fi0+1)
                                         +NeighborValue(dm, remote_data, v, fk, fj0+1,
                                             fi0)
                                         +NeighborValue(dm, remote_data, v, fk, fj0+1,
                                             fi0+1));
                        u_h(m,v,gks,gj,gi) = ot*(4.0*favg - u_h(m,v,gks+ok,gj+oj,gi+oi));
                      }
                    }
                  }
                }
                modified = true;
                modified_blocks[m] = 1;

              } else if (nlev < m_lev) {
                // ==== COARSER neighbor, EDGE/CORNER: simple injection ====
                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else              { gis = ngh;            gie = ngh + ncells - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else              { gjs = ngh;            gje = ngh + ncells - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else              { gks = ngh;            gke = ngh + ncells - 1; }

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int si, sj, sk;
                        if (ox1 < 0)      si = ngh + ncells - 1;
                        else if (ox1 > 0) si = ngh;
                        else
                          si = ngh + child_x*(half) + (gi - ngh)/2;
                        if (ox2 < 0)      sj = ngh + ncells - 1;
                        else if (ox2 > 0) sj = ngh;
                        else
                          sj = ngh + child_y*(half) + (gj - ngh)/2;
                        if (ox3 < 0)      sk = ngh + ncells - 1;
                        else if (ox3 > 0) sk = ngh;
                        else
                          sk = ngh + child_z*(half) + (gk - ngh)/2;

                        u_h(m, v, gk, gj, gi) = NeighborValue(dm, remote_data, v, sk, sj,
                            si);
                      }
                    }
                  }
                }
                modified = true;
                modified_blocks[m] = 1;

              } else {
                // ==== FINER neighbor, EDGE/CORNER: simple restriction ====
                int sub_x = 0, sub_y = 0, sub_z = 0;
                if (nface == 2) {
                  if (ox1 == 0) sub_x = f1;
                  if (ox2 == 0) sub_y = f1;
                  if (ox3 == 0) sub_z = f1;
                }
                int gis, gie, gjs, gje, gks, gke;
                if (ox1 < 0)      { gis = 0;            gie = ngh - 1; }
                else if (ox1 > 0) { gis = ngh + ncells;  gie = ngh + ncells + ngh - 1; }
                else { gis = ngh + sub_x*half; gie = ngh + sub_x*half + half - 1; }
                if (ox2 < 0)      { gjs = 0;            gje = ngh - 1; }
                else if (ox2 > 0) { gjs = ngh + ncells;  gje = ngh + ncells + ngh - 1; }
                else { gjs = ngh + sub_y*half; gje = ngh + sub_y*half + half - 1; }
                if (ox3 < 0)      { gks = 0;            gke = ngh - 1; }
                else if (ox3 > 0) { gks = ngh + ncells;  gke = ngh + ncells + ngh - 1; }
                else { gks = ngh + sub_z*half; gke = ngh + sub_z*half + half - 1; }

                for (int v = 0; v < nvar; ++v) {
                  for (int gk = gks; gk <= gke; ++gk) {
                    for (int gj = gjs; gj <= gje; ++gj) {
                      for (int gi = gis; gi <= gie; ++gi) {
                        int fi0, fi1, fj0, fj1, fk0, fk1;
                        if (ox1 < 0) {
                          fi0 = ngh + ncells - 2; fi1 = ngh + ncells - 1;
                        } else if (ox1 > 0) {
                          fi0 = ngh; fi1 = ngh + 1;
                        } else {
                          fi0 = ngh + 2*(gi - (ngh + sub_x*half)); fi1 = fi0 + 1;
                        }
                        if (ox2 < 0) {
                          fj0 = ngh + ncells - 2; fj1 = ngh + ncells - 1;
                        } else if (ox2 > 0) {
                          fj0 = ngh; fj1 = ngh + 1;
                        } else {
                          fj0 = ngh + 2*(gj - (ngh + sub_y*half)); fj1 = fj0 + 1;
                        }
                        if (ox3 < 0) {
                          fk0 = ngh + ncells - 2; fk1 = ngh + ncells - 1;
                        } else if (ox3 > 0) {
                          fk0 = ngh; fk1 = ngh + 1;
                        } else {
                          fk0 = ngh + 2*(gk - (ngh + sub_z*half)); fk1 = fk0 + 1;
                        }
                        u_h(m, v, gk, gj, gi) = 0.125 * (
                          NeighborValue(dm, remote_data, v, fk0, fj0, fi0) +
                          NeighborValue(dm, remote_data, v, fk0, fj0, fi1) +
                          NeighborValue(dm, remote_data, v, fk0, fj1, fi0) +
                          NeighborValue(dm, remote_data, v, fk0, fj1, fi1) +
                          NeighborValue(dm, remote_data, v, fk1, fj0, fi0) +
                          NeighborValue(dm, remote_data, v, fk1, fj0, fi1) +
                          NeighborValue(dm, remote_data, v, fk1, fj1, fi0) +
                          NeighborValue(dm, remote_data, v, fk1, fj1, fi1));
                      }
                    }
                  }
                }
                modified = true;
                modified_blocks[m] = 1;
              }
            }
          }
        }
      }
    }
  }

  if (modified) {
    int pushed_blocks = 0;
    for (int m = 0; m < nmb; ++m) {
      if (modified_blocks[m] != 0) ++pushed_blocks;
    }
    if (pushed_blocks == nmb) {
      Kokkos::deep_copy(u, u_h);
    } else {
      for (int m = 0; m < nmb; ++m) {
        if (modified_blocks[m] == 0) continue;
        Kokkos::deep_copy(Kokkos::subview(u, m, Kokkos::ALL(), Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL()),
                          Kokkos::subview(u_h, m, Kokkos::ALL(), Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL()));
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridBoundaryValues::PackAndSend()
//! \brief Pack restricted fluxes of multigrid variables at fine/coarse boundaries
//! into boundary buffers and send to neighbors. Adapts to different block sizes per
//! level.

TaskStatus MultigridBoundaryValues::PackAndSendMG(
    const DvceArray5D<Real> &u, bool include_diagonals) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  if (global_variable::nranks == 1) return TaskStatus::complete;

  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = u.extent_int(1);

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;

  int shift_ = pmy_mg->GetLevelShift();
  int nx1_ = pmy_mg->GetSize();
  const bool use_mg_same_agg = UseMGSameLevelAgg();
  DevExeSpace pack_exec;
  int agg_send_slot = mg_same_send_pack_slot_;
#if MPI_PARALLEL_ENABLED
  if (use_mg_same_agg) {
    // Always run through EnsureMGSameLevelAggReady(): it validates mesh signature
    // in addition to dimensions, preventing stale MPI pack maps after AMR
    // topology changes that keep nmb/nnghbr unchanged.
    EnsureMGSameLevelAggReady(nvar, include_diagonals);
    if (!mg_same_has_remote_) return TaskStatus::complete;

    auto release_send_slot = [&](const int slot) -> bool {
      if (!mg_same_send_slot_inflight_[slot]) return true;
      auto &slot_reqs = mg_same_send_req_h_[slot];
      if (slot_reqs.empty()) {
        mg_same_send_slot_inflight_[slot] = false;
        mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0]
                                  || mg_same_send_slot_inflight_[1]);
        return true;
      }
      int done = 0;
      int ierr = MPI_Testall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), &done, MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in testing aggregated MG sends"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (done == 0) return false;
      if (!mg_same_persistent_ready_) {
        std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
      }
      mg_same_send_slot_inflight_[slot] = false;
      mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0]
                                || mg_same_send_slot_inflight_[1]);
      return true;
    };

    if (mg_same_send_slot_inflight_[agg_send_slot]) {
      const int alt_slot = 1 - agg_send_slot;
      if (!release_send_slot(agg_send_slot)) {
        if (!release_send_slot(alt_slot)) {
          ++mg_same_send_poll_count_;
          if (mg_same_send_poll_count_ < mg_same_poll_budget_) {
            return TaskStatus::incomplete;
          }
          auto &slot_reqs = mg_same_send_req_h_[agg_send_slot];
          int ierr = MPI_Waitall(static_cast<int>(slot_reqs.size()),
                                 slot_reqs.data(), MPI_STATUSES_IGNORE);
          if (ierr != MPI_SUCCESS) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "MPI error in waiting aggregated MG sends"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          if (!mg_same_persistent_ready_) {
            std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
          }
          mg_same_send_slot_inflight_[agg_send_slot] = false;
          mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0]
                                    || mg_same_send_slot_inflight_[1]);
        } else {
          agg_send_slot = alt_slot;
        }
      }
    }
    mg_same_send_pack_slot_ = agg_send_slot;
    mg_same_send_poll_count_ = 0;
  }
#endif

  if (use_mg_same_agg) {
    auto send_m = mg_same_send_m_d_;
    auto send_off = mg_same_send_off_d_;
    auto send_bounds = mg_same_send_bounds_d_;
    auto send_data = mg_same_send_data_d_[agg_send_slot];
    const int nsend = send_m.extent_int(0);
    if (nsend > 0) {
      const int nsv = nsend * nvar;
      Kokkos::TeamPolicy<> policy(pack_exec, nsv, Kokkos::AUTO);
      Kokkos::parallel_for("PackMGAgg", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int s = tmember.league_rank() / nvar;
        const int v = tmember.league_rank() - s * nvar;
        const int m = send_m(s);
        const int il = send_bounds(s, 0);
        const int iu = send_bounds(s, 1);
        const int jl = send_bounds(s, 2);
        const int ju = send_bounds(s, 3);
        const int kl = send_bounds(s, 4);
        const int ku = send_bounds(s, 5);

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nc = ni * nj * nk;
        const int base = send_off(s) + nc * v;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nc),
        [&](const int idx) {
          const int ii = idx % ni;
          const int tmp = idx / ni;
          const int jj = tmp % nj;
          const int kk = tmp / nj;
          const int i = il + ii;
          const int j = jl + jj;
          const int k = kl + kk;
          send_data(base + idx) = u(m, v, k, j, i);
        });
      });
    }
  } else {
    const int mg_ngh = pmy_mg->GetGhostCells();
    const int mg_depth = mg_same_halo_depth_;
    int nmnv = nmb * nnghbr * nvar;
    Kokkos::TeamPolicy<> policy(pack_exec, nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("PackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank() / (nnghbr * nvar);
      const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
      const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

      if (nghbr.d_view(m, n).gid >= 0 &&
          nghbr.d_view(m, n).lev == mblev.d_view(m)) {
        const int face_mask = MGNeighborFaceMask(n);
        const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                        + ((face_mask >> 2) & 1);
        if (!include_diagonals && nface != 1) return;
        int il = sbuf[n].isame[0].bis;
        int iu = sbuf[n].isame[0].bie;
        int jl = sbuf[n].isame[0].bjs;
        int ju = sbuf[n].isame[0].bje;
        int kl = sbuf[n].isame[0].bks;
        int ku = sbuf[n].isame[0].bke;
        AdjustMGSameSendFaceBounds(mg_ngh, shift_, nx1_,
                                   (face_mask & 1) != 0,
                                   (face_mask & 2) != 0,
                                   (face_mask & 4) != 0, mg_depth,
                                   il, iu, jl, ju, kl, ku);

        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj = nk * nj;
        if (nghbr.d_view(m, n).rank != my_rank) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
          [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
            [&](const int i) {
              int lid = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
              sbuf[n].vars(m, lid) = u(m, v, k, j, i);
            });
          });
        }
      }
      tmember.team_barrier();
    });
  }

  #if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  bool need_pack_fence = false;
  if (use_mg_same_agg) {
    need_pack_fence = (mg_same_send_m_d_.extent_int(0) > 0)
                      && !mg_same_send_rank_h_.empty();
  } else {
    for (int m = 0; m < nmb && !need_pack_fence; ++m) {
      for (int n = 0; n < nnghbr; ++n) {
        if (nghbr.h_view(m, n).gid < 0
            || nghbr.h_view(m, n).lev != pmy_pack->pmb->mb_lev.h_view(m)) {
          continue;
        }
        const int face_mask = MGNeighborFaceMask(n);
        const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                        + ((face_mask >> 2) & 1);
        if (!include_diagonals && nface != 1) continue;
        if (nghbr.h_view(m, n).rank != my_rank) {
          need_pack_fence = true;
          break;
        }
      }
    }
  }
  // Send boundary buffer to neighboring MeshBlocks using MPI. The device fence is only
  // needed when a remote send will actually consume freshly packed device data.
  if (need_pack_fence) {
    pack_exec.fence();
  }
  bool no_errors=true;
  bool posted_agg_send = false;
  if (use_mg_same_agg && !mg_same_send_rank_h_.empty()) {
    auto &slot_reqs = mg_same_send_req_h_[agg_send_slot];
    if (mg_same_persistent_ready_) {
      int ierr = MPI_Startall(static_cast<int>(slot_reqs.size()), slot_reqs.data());
      if (ierr != MPI_SUCCESS) {no_errors = false;}
      posted_agg_send = true;
    } else {
      for (int sr = 0; sr < static_cast<int>(mg_same_send_rank_h_.size()); ++sr) {
        const int rank = mg_same_send_rank_h_[sr];
        const int tag = mg_same_send_tag_h_[sr];
        const int doff = mg_same_send_data_off_h_[sr];
        const int dcount = mg_same_send_data_off_h_[sr + 1] - doff;
        if (dcount <= 0) continue;
        int ierr = MPI_Isend(mg_same_send_data_d_[agg_send_slot].data() + doff, dcount,
                             MPI_ATHENA_REAL, rank, tag, comm_vars,
                             &(slot_reqs[sr]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        posted_agg_send = true;
      }
    }
    mg_same_send_slot_inflight_[agg_send_slot] = posted_agg_send;
    mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0]
                              || mg_same_send_slot_inflight_[1]);
    if (posted_agg_send) {
      mg_same_send_pack_slot_ = 1 - agg_send_slot;
    }
  }
  if (!(use_mg_same_agg && !mg_same_send_rank_h_.empty())) {
    mg_same_send_inflight_ = (mg_same_send_slot_inflight_[0]
                              || mg_same_send_slot_inflight_[1]);
  }
  mg_same_send_poll_count_ = 0;
  if (!use_mg_same_agg) {
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.h_view(m,n).gid >= 0
            && nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m)) {
          const int face_mask = MGNeighborFaceMask(n);
          const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                          + ((face_mask >> 2) & 1);
          if (!include_diagonals && nface != 1) continue;
          int dn = nghbr.h_view(m,n).dest;
          int drank = nghbr.h_view(m,n).rank;
          if (drank != my_rank) {
            // create tag using local ID and buffer index of *receiving* MeshBlock
            int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
            int tag = CreateBvals_MPI_Tag(lid, dn);

            // get ptr to send buffer when neighbor is at coarser/same/fine level
            int il = sendbuf[n].isame[0].bis;
            int iu = sendbuf[n].isame[0].bie;
            int jl = sendbuf[n].isame[0].bjs;
            int ju = sendbuf[n].isame[0].bje;
            int kl = sendbuf[n].isame[0].bks;
            int ku = sendbuf[n].isame[0].bke;
            AdjustMGSameSendFaceBounds(pmy_mg->GetGhostCells(), shift_, nx1_,
                                       (face_mask & 1) != 0,
                                       (face_mask & 2) != 0,
                                       (face_mask & 4) != 0, mg_same_halo_depth_,
                                       il, iu, jl, ju, kl, ku);
            int data_size = nvar * MGBoundsCellCount(il, iu, jl, ju, kl, ku);

            auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

            // MG tasklists can post multiple sends in one cycle; ensure request slot is
            // reusable.
            if (sendbuf[n].vars_req[m] != MPI_REQUEST_NULL) {
              int ierr_wait = MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
              if (ierr_wait != MPI_SUCCESS) {no_errors=false;}
            }
            int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                                 comm_vars, &(sendbuf[n].vars_req[m]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
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
//! \fn TaskStatus MultigridBoundaryValuesCC::RecvAndUnpackMG()
//! \brief Receive and unpack cell-centered multigrid variables.
//! Handles ghost-cell filling at each multigrid level independently.

TaskStatus MultigridBoundaryValues::RecvAndUnpackMG(
    DvceArray5D<Real> &u, bool include_diagonals) {
  if (pmy_mg == nullptr) return TaskStatus::complete;
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
  int shift_ = pmy_mg->GetLevelShift();
  int nx1_ = pmy_mg->GetSize();
  const bool single_rank = (global_variable::nranks == 1);

  {
    const int mbgid0 = mbgid.h_view(0);
    const int shift_idx = 4 * std::max(0, shift_) + (mg_same_halo_depth_ > 1 ? 2 : 0)
                        + (include_diagonals ? 1 : 0);
    const std::uint64_t mesh_sig = pmy_mg->pmy_driver_->GetMeshSignature();
    if (static_cast<int>(mg_single_pair_valid_cache_.size()) <= shift_idx) {
      int sz = shift_idx + 1;
      mg_single_pair_m_cache_d_.resize(sz);
      mg_single_pair_sm_cache_d_.resize(sz);
      mg_single_pair_desc_cache_d_.resize(sz);
      mg_single_pair_count_cache_.resize(sz, 0);
      mg_single_cached_nmb_cache_.resize(sz, -1);
      mg_single_cached_nnghbr_cache_.resize(sz, -1);
      mg_single_cached_mbgid0_cache_.resize(sz, -1);
      mg_single_cached_nx1_cache_.resize(sz, -1);
      mg_single_cached_mesh_sig_cache_.resize(sz, 0);
      mg_single_pair_valid_cache_.resize(sz, 0);
    }

    auto &pair_m_d_cache = mg_single_pair_m_cache_d_[shift_idx];
    auto &pair_sm_d_cache = mg_single_pair_sm_cache_d_[shift_idx];
    auto &pair_desc_d_cache = mg_single_pair_desc_cache_d_[shift_idx];
    int &pair_count = mg_single_pair_count_cache_[shift_idx];
    if (!mg_single_pair_valid_cache_[shift_idx]
        || mg_single_cached_nmb_cache_[shift_idx] != nmb
        || mg_single_cached_nnghbr_cache_[shift_idx] != nnghbr
        || mg_single_cached_mbgid0_cache_[shift_idx] != mbgid0
        || mg_single_cached_nx1_cache_[shift_idx] != nx1_
        || mg_single_cached_mesh_sig_cache_[shift_idx] != mesh_sig) {
      const int ngh = pmy_mg->GetGhostCells();
      std::vector<int> pair_m_h, pair_sm_h, pair_desc_h;
      const std::size_t reserve_n = static_cast<std::size_t>(nmb) * nnghbr;
      pair_m_h.reserve(reserve_n);
      pair_sm_h.reserve(reserve_n);
      pair_desc_h.reserve(12 * reserve_n);

      for (int m = 0; m < nmb; ++m) {
        const int mlev = mblev.h_view(m);
        for (int n = 0; n < nnghbr; ++n) {
          const auto &nb = nghbr.h_view(m, n);
          if (nb.gid < 0 || nb.lev != mlev ||
              nb.rank != global_variable::my_rank) continue;
          const int recv_face_mask = MGNeighborFaceMask(n);
          const int nface = (recv_face_mask & 1) + ((recv_face_mask >> 1) & 1)
                          + ((recv_face_mask >> 2) & 1);
          if (!include_diagonals && nface != 1) continue;
          const int sm = nb.gid - mbgid0;
          const int sn = nb.dest;
          if (sm < 0 || sm >= nmb || sn < 0 || sn >= nnghbr) continue;

          int dil = rbuf[n].isame[0].bis;
          int diu = rbuf[n].isame[0].bie;
          int djl = rbuf[n].isame[0].bjs;
          int dju = rbuf[n].isame[0].bje;
          int dkl = rbuf[n].isame[0].bks;
          int dku = rbuf[n].isame[0].bke;
          AdjustMGSameRecvFaceBounds(ngh, shift_,
                                     (recv_face_mask & 1) != 0,
                                     (recv_face_mask & 2) != 0,
                                     (recv_face_mask & 4) != 0, mg_same_halo_depth_,
                                     dil, diu, djl, dju, dkl, dku);

          int sil = sbuf[sn].isame[0].bis;
          int siu = sbuf[sn].isame[0].bie;
          int sjl = sbuf[sn].isame[0].bjs;
          int sju = sbuf[sn].isame[0].bje;
          int skl = sbuf[sn].isame[0].bks;
          int sku = sbuf[sn].isame[0].bke;
          const int send_face_mask = MGNeighborFaceMask(sn);
          AdjustMGSameSendFaceBounds(ngh, shift_, nx1_,
                                     (send_face_mask & 1) != 0,
                                     (send_face_mask & 2) != 0,
                                     (send_face_mask & 4) != 0, mg_same_halo_depth_,
                                     sil, siu, sjl, sju, skl, sku);

          if (diu < dil || dju < djl || dku < dkl
              || siu < sil || sju < sjl || sku < skl) {
            continue;
          }

          pair_m_h.push_back(m);
          pair_sm_h.push_back(sm);
          pair_desc_h.push_back(dil);
          pair_desc_h.push_back(diu);
          pair_desc_h.push_back(djl);
          pair_desc_h.push_back(dju);
          pair_desc_h.push_back(dkl);
          pair_desc_h.push_back(dku);
          pair_desc_h.push_back(sil);
          pair_desc_h.push_back(siu);
          pair_desc_h.push_back(sjl);
          pair_desc_h.push_back(sju);
          pair_desc_h.push_back(skl);
          pair_desc_h.push_back(sku);
        }
      }

      pair_count = static_cast<int>(pair_m_h.size());
      Kokkos::realloc(pair_m_d_cache, pair_count);
      Kokkos::realloc(pair_sm_d_cache, pair_count);
      Kokkos::realloc(pair_desc_d_cache, pair_count, 12);

      auto pair_m_view_h = Kokkos::create_mirror_view(pair_m_d_cache);
      auto pair_sm_view_h = Kokkos::create_mirror_view(pair_sm_d_cache);
      auto pair_desc_view_h = Kokkos::create_mirror_view(pair_desc_d_cache);
      for (int p = 0; p < pair_count; ++p) {
        pair_m_view_h(p) = pair_m_h[p];
        pair_sm_view_h(p) = pair_sm_h[p];
        for (int c = 0; c < 12; ++c) {
          pair_desc_view_h(p, c) = pair_desc_h[12 * p + c];
        }
      }
      Kokkos::deep_copy(pair_m_d_cache, pair_m_view_h);
      Kokkos::deep_copy(pair_sm_d_cache, pair_sm_view_h);
      Kokkos::deep_copy(pair_desc_d_cache, pair_desc_view_h);
      mg_single_cached_nmb_cache_[shift_idx] = nmb;
      mg_single_cached_nnghbr_cache_[shift_idx] = nnghbr;
      mg_single_cached_mbgid0_cache_[shift_idx] = mbgid0;
      mg_single_cached_nx1_cache_[shift_idx] = nx1_;
      mg_single_cached_mesh_sig_cache_[shift_idx] = mesh_sig;
      mg_single_pair_valid_cache_[shift_idx] = 1;
    }

    if (pair_count == 0) {
      if (single_rank) return TaskStatus::complete;
    } else {
      const int nvar = u.extent_int(1);
      auto pair_m_d = pair_m_d_cache;
      auto pair_sm_d = pair_sm_d_cache;
      auto pair_desc_d = pair_desc_d_cache;
      const int npairs = pair_count;
      const int npv = npairs * nvar;
      Kokkos::TeamPolicy<> policy(DevExeSpace(), npv, Kokkos::AUTO);
      Kokkos::parallel_for("UnpackMGSameRankDirectPairs", policy,
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int p = tmember.league_rank() / nvar;
        const int v = tmember.league_rank() - p * nvar;
        const int m = pair_m_d(p);
        const int sm = pair_sm_d(p);
        const int dil = pair_desc_d(p, 0);
        const int diu = pair_desc_d(p, 1);
        const int djl = pair_desc_d(p, 2);
        const int dju = pair_desc_d(p, 3);
        const int dkl = pair_desc_d(p, 4);
        const int dku = pair_desc_d(p, 5);
        const int sil = pair_desc_d(p, 6);
        const int siu = pair_desc_d(p, 7);
        const int sjl = pair_desc_d(p, 8);
        const int sju = pair_desc_d(p, 9);
        const int skl = pair_desc_d(p, 10);
        const int sku = pair_desc_d(p, 11);

        const int dni = diu - dil + 1;
        const int dnj = dju - djl + 1;
        const int dnk = dku - dkl + 1;
        const int dnc = dni * dnj * dnk;
        const int sni = siu - sil + 1;
        const int snj = sju - sjl + 1;
        const int snij = sni * snj;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, dnc),
        [&](const int idx) {
          const int di = dil + (idx % dni);
          const int tmp = idx / dni;
          const int dj = djl + (tmp % dnj);
          const int dk = dkl + (tmp / dnj);
          const int si = sil + (idx % sni);
          const int sj = sjl + ((idx / sni) % snj);
          const int sk = skl + (idx / snij);
          u(m, v, dk, dj, di) = u(sm, v, sk, sj, si);
        });
      });
    }
    if (single_rank) return TaskStatus::complete;
  }

  const bool use_mg_same_agg = UseMGSameLevelAgg();
  if (use_mg_same_agg) {
    const int nvar = u.extent_int(1);
    EnsureMGSameLevelAggReady(nvar, include_diagonals);
  }
  #if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    if (use_mg_same_agg && mg_same_recv_inflight_
        && !mg_same_recv_req_h_[mg_same_recv_data_slot_].empty()) {
      auto &slot_reqs = mg_same_recv_req_h_[mg_same_recv_data_slot_];
      int done = 0;
      int ierr = MPI_Testall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), &done, MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      if (!done) {
        ++mg_same_recv_poll_count_;
        if (mg_same_recv_poll_count_ < mg_same_poll_budget_) {
          bflag = true;
        } else {
          WaitMGSameLevelRecvRequests();
          mg_same_recv_poll_count_ = 0;
        }
      } else {
        if (!mg_same_persistent_ready_) {
          std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
        }
        mg_same_recv_inflight_ = false;
        mg_same_recv_poll_count_ = 0;
      }
    } else if (use_mg_same_agg) {
      mg_same_recv_poll_count_ = 0;
    }
    if (!use_mg_same_agg) {
      for (int m=0; m<nmb; ++m) {
        for (int n=0; n<nnghbr; ++n) {
          if (nghbr.h_view(m,n).gid >= 0
              && nghbr.h_view(m,n).lev == mblev.h_view(m)) {
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
  } else {
    if (use_mg_same_agg) {
      mg_same_recv_inflight_ = false;
    }
  }

#endif

  //----- STEP 2: buffers have all completed, so unpack
  int nvar = u.extent_int(1);
  int ngh = pmy_mg->GetGhostCells();
  const int halo_depth = mg_same_halo_depth_;

  if (use_mg_same_agg) {
    auto recv_m = mg_same_recv_m_d_;
    auto recv_off = mg_same_recv_off_d_;
    auto recv_bounds = mg_same_recv_bounds_d_;
    auto recv_data = mg_same_recv_data_d_[mg_same_recv_data_slot_];
    const int nrecv = recv_m.extent_int(0);
    if (nrecv > 0) {
      const int nrv = nrecv * nvar;
      Kokkos::TeamPolicy<> policy(DevExeSpace(), nrv, Kokkos::AUTO);
      Kokkos::parallel_for("UnpackMGAgg", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int r = tmember.league_rank() / nvar;
        const int v = tmember.league_rank() - r * nvar;
        const int m = recv_m(r);
        const int il = recv_bounds(r, 0);
        const int iu = recv_bounds(r, 1);
        const int jl = recv_bounds(r, 2);
        const int ju = recv_bounds(r, 3);
        const int kl = recv_bounds(r, 4);
        const int ku = recv_bounds(r, 5);

        const int ni = iu - il + 1;
        const int nj = ju - jl + 1;
        const int nk = ku - kl + 1;
        const int nc = ni * nj * nk;
        const int base = recv_off(r) + nc * v;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nc),
        [&](const int idx) {
          const int ii = idx % ni;
          const int tmp = idx / ni;
          const int jj = tmp % nj;
          const int kk = tmp / nj;
          const int i = il + ii;
          const int j = jl + jj;
          const int k = kl + kk;
          u(m, v, k, j, i) = recv_data(base + idx);
        });
      });
    }
  } else {
    const int my_rank_l = global_variable::my_rank;
    int nmnv = nmb * nnghbr * nvar;
    Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
    Kokkos::parallel_for("UnpackMG", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank() / (nnghbr * nvar);
      const int n = (tmember.league_rank() - m * nnghbr * nvar) / nvar;
      const int v = tmember.league_rank() - m * nnghbr * nvar - n * nvar;

      if (nghbr.d_view(m, n).gid >= 0 &&
          nghbr.d_view(m, n).lev == mblev.d_view(m)) {
        const int face_mask = MGNeighborFaceMask(n);
        const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                        + ((face_mask >> 2) & 1);
        if (!include_diagonals && nface != 1) return;
        if (nghbr.d_view(m, n).rank == my_rank_l) return;
        int il = rbuf[n].isame[0].bis;
        int iu = rbuf[n].isame[0].bie;
        int jl = rbuf[n].isame[0].bjs;
        int ju = rbuf[n].isame[0].bje;
        int kl = rbuf[n].isame[0].bks;
        int ku = rbuf[n].isame[0].bke;
        AdjustMGSameRecvFaceBounds(ngh, shift_,
                                   (face_mask & 1) != 0,
                                   (face_mask & 2) != 0,
                                   (face_mask & 4) != 0, halo_depth,
                                   il, iu, jl, ju, kl, ku);

        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj = nk * nj;

        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
        [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
          [&](const int i) {
            int lid = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
            u(m, v, k, j, i) = rbuf[n].vars(m, lid);
          });
        });
      }
    });
  }

#if MPI_PARALLEL_ENABLED
  // Not MarkRecvUnpackPending(): this unpack reads the aggregated payload, not the bvals
  // receive buffers, so it carries its own per-slot event and raises the base class flag
  // in the full-fence form.
  if (global_variable::nranks > 1) MarkMGUnpackPending(mg_same_recv_data_slot_);
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBoundaryValues::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications of vars.

TaskStatus MultigridBoundaryValues::InitRecvMG(
    const int nvars, bool include_diagonals) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  int &nmb = pmy_pack->nmb_thispack;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int shift_ = pmy_mg->GetLevelShift();
  const bool use_mg_same_agg = UseMGSameLevelAgg();
  if (use_mg_same_agg) {
    EnsureMGSameLevelAggReady(nvars, include_diagonals);
    if (!mg_same_has_remote_) return TaskStatus::complete;
    if (mg_same_recv_inflight_
        && !mg_same_recv_req_h_[mg_same_recv_data_slot_].empty()) {
      auto &slot_reqs = mg_same_recv_req_h_[mg_same_recv_data_slot_];
      int done = 0;
      int ierr = MPI_Testall(static_cast<int>(slot_reqs.size()),
                             slot_reqs.data(), &done, MPI_STATUSES_IGNORE);
      if (ierr != MPI_SUCCESS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "MPI error in testing aggregated MG receives"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (!done) {
        ++mg_same_recv_poll_count_;
        if (mg_same_recv_poll_count_ < mg_same_poll_budget_) {
          return TaskStatus::incomplete;
        }
        WaitMGSameLevelRecvRequests();
      } else {
        if (!mg_same_persistent_ready_) {
          std::fill(slot_reqs.begin(), slot_reqs.end(), MPI_REQUEST_NULL);
        }
        mg_same_recv_inflight_ = false;
      }
      mg_same_recv_poll_count_ = 0;
    } else {
      mg_same_recv_poll_count_ = 0;
    }
  }

  // Initialize communications of variables
  bool no_errors=true;
  bool posted_agg_recv = false;
  const int agg_recv_slot = mg_same_recv_post_slot_;
  if (use_mg_same_agg && !mg_same_recv_rank_h_.empty()) {
    // MPI is about to write this slot; the unpack that last read it must be off it
    // first.  That unpack is two exchanges back, so the wait is normally already
    // satisfied and the smoother queued behind it keeps running either way.
    WaitMGUnpackComplete(agg_recv_slot);
    auto &slot_reqs = mg_same_recv_req_h_[agg_recv_slot];
    if (mg_same_persistent_ready_) {
      int ierr = MPI_Startall(static_cast<int>(slot_reqs.size()), slot_reqs.data());
      if (ierr != MPI_SUCCESS) {no_errors = false;}
      posted_agg_recv = true;
    } else {
      for (int rr = 0; rr < static_cast<int>(mg_same_recv_rank_h_.size()); ++rr) {
        const int rank = mg_same_recv_rank_h_[rr];
        const int tag = mg_same_recv_tag_h_[rr];
        const int doff = mg_same_recv_data_off_h_[rr];
        const int dcount = mg_same_recv_data_off_h_[rr + 1] - doff;
        if (dcount <= 0) continue;
        int ierr = MPI_Irecv(mg_same_recv_data_d_[agg_recv_slot].data() + doff, dcount,
                             MPI_ATHENA_REAL, rank, tag, comm_vars, &(slot_reqs[rr]));
        if (ierr != MPI_SUCCESS) {no_errors=false;}
        posted_agg_recv = true;
      }
    }
  }
  mg_same_recv_inflight_ = posted_agg_recv;
  if (posted_agg_recv) {
    mg_same_recv_data_slot_ = agg_recv_slot;
    mg_same_recv_post_slot_ = 1 - agg_recv_slot;
  }
  mg_same_recv_poll_count_ = 0;
  if (!use_mg_same_agg) {
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.h_view(m,n).gid >= 0
            && nghbr.h_view(m,n).lev == mblev.h_view(m)) {
          const int face_mask = MGNeighborFaceMask(n);
          const int nface = (face_mask & 1) + ((face_mask >> 1) & 1)
                          + ((face_mask >> 2) & 1);
          if (!include_diagonals && nface != 1) continue;
          // rank of destination buffer
          int drank = nghbr.h_view(m,n).rank;

          // post non-blocking receive if neighboring MeshBlock on a different rank
          if (drank != global_variable::my_rank) {
            // create tag using local ID and buffer index of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(m, n);

            // calculate amount of data to be passed, get pointer to variables
            int il = recvbuf[n].isame[0].bis;
            int iu = recvbuf[n].isame[0].bie;
            int jl = recvbuf[n].isame[0].bjs;
            int ju = recvbuf[n].isame[0].bje;
            int kl = recvbuf[n].isame[0].bks;
            int ku = recvbuf[n].isame[0].bke;
            AdjustMGSameRecvFaceBounds(pmy_mg->GetGhostCells(), shift_,
                                       (face_mask & 1) != 0,
                                       (face_mask & 2) != 0,
                                       (face_mask & 4) != 0, mg_same_halo_depth_,
                                       il, iu, jl, ju, kl, ku);
            int data_size = nvars * MGBoundsCellCount(il, iu, jl, ju, kl, ku);

            auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, Kokkos::ALL);

            // MG tasklists can post multiple receives in one cycle; ensure request slot
            // is reusable.
            if (recvbuf[n].vars_req[m] != MPI_REQUEST_NULL) {
              int ierr_wait = MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
              if (ierr_wait != MPI_SUCCESS) {no_errors=false;}
            }

            // Post non-blocking receive for this buffer on this MeshBlock
            int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                                 comm_vars, &(recvbuf[n].vars_req[m]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
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
