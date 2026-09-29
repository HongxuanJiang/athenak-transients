//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid_driver.cpp
//! \brief implementation of functions in class MultigridDriver

// C headers

// C++ headers
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>    // abs
#include <cstring>    // memcpy
#include <iomanip>    // setprecision
#include <iostream>   // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <utility>    // make_pair
#include <vector>

// Project headers
#include "../athena.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../coordinates/coordinates.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "multigrid.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
bool IsPeriodicMGBoundary(BoundaryFlag bc) {
  return (bc == BoundaryFlag::periodic);
}

KOKKOS_INLINE_FUNCTION
bool IsZeroGradMGBoundary(BoundaryFlag bc) {
  return (bc == BoundaryFlag::mg_zerograd || bc == BoundaryFlag::outflow);
}

KOKKOS_INLINE_FUNCTION
bool IsZeroFixedMGBoundary(BoundaryFlag bc) {
  return (bc == BoundaryFlag::mg_zerofixed);
}

// The cell loop is spread over (block, x3 slice) rather than block alone.  A MeshBlock
// level holds a few hundred blocks at most, which is far short of what the device wants,
// and the serial walk of all 32768 cells also forced the 25 accumulators out of
// registers:
// the write-back loop ran to a runtime bound, so mp[] had to live in local memory.  Each
// slice now writes a fixed 25 columns and a second kernel folds the slices together.
//
// Summing each slice separately and then summing the slices is a blocked summation of the
// same terms, so its error bound is the better of the two: O((n/nx3 + nx3)*eps) against
// the O(n*eps) of the flat sequential sum the single-thread version performed.
// One face of the root grid's cached multipole potential.  a runs over the slower
// tangential index of that face and b over the faster one, both across the full extent
// including ghosts, because the root sweep evaluates the expansion on its own edge and
// corner slots as well.
template <typename ExecSpace, typename PhiView>
void FillRootBoundaryPhi(const ExecSpace &exec, const PhiView &phi, const Real *mpc,
    int f, bool mp, int base, int na, int nb, int ngh, Real dx1, Real dx2, Real dx3,
    Real x1min, Real x1max, Real x2min, Real x2max, Real x3min, Real x3max,
    Real mpx, Real mpy, Real mpz, int mporder) {
  if (na <= 0 || nb <= 0) return;
  par_for("MGRootPhiCache", ExecSpace(), 0, na-1, 0, nb-1,
  KOKKOS_LAMBDA(const int a, const int b) {
    Real v = 0.0;
    if (mp) {
      if (f < 2) {
        const Real y = x2min + (b - ngh + 0.5)*dx2 - mpy;
        const Real z = x3min + (a - ngh + 0.5)*dx3 - mpz;
        v = EvalMultipolePhi(((f == 0) ? x1min : x1max) - mpx, y, z, mpc, mporder);
      } else if (f < 4) {
        const Real x = x1min + (b - ngh + 0.5)*dx1 - mpx;
        const Real z = x3min + (a - ngh + 0.5)*dx3 - mpz;
        v = EvalMultipolePhi(x, ((f == 2) ? x2min : x2max) - mpy, z, mpc, mporder);
      } else {
        const Real x = x1min + (b - ngh + 0.5)*dx1 - mpx;
        const Real y = x2min + (a - ngh + 0.5)*dx2 - mpy;
        v = EvalMultipolePhi(x, y, ((f == 4) ? x3min : x3max) - mpz, mpc, mporder);
      }
    }
    phi(base + a*nb + b) = v;
  });
}

template <typename ExecSpace, typename SrcView, typename SizeView,
          typename PartialView, typename KPartialView>
void ComputeMultipolePartials(const ExecSpace &exec, const SrcView &src,
    const SizeView &mb_size, const PartialView &partial, const KPartialView &kpartial,
    int nmb, int ngh,
    int nx1, int nx2, int nx3, Real xorigin, Real yorigin, Real zorigin,
    int order, bool skip_dipole, int ncoeff) {
  if (nmb <= 0) return;
  Kokkos::parallel_for("MGMultipoleCoeffs",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                    Kokkos::Iterate::Right>>(
          exec, {0, 0}, {nmb, nx3}),
  KOKKOS_LAMBDA(const int m, const int kk) {
    const Real dx1 = (mb_size(m).x1max - mb_size(m).x1min)
                   / static_cast<Real>(nx1);
    const Real dx2 = (mb_size(m).x2max - mb_size(m).x2min)
                   / static_cast<Real>(nx2);
    const Real dx3 = (mb_size(m).x3max - mb_size(m).x3min)
                   / static_cast<Real>(nx3);
    const Real vol = dx1*dx2*dx3;
    Real mp[25] = {};

    {
      const int k = ngh + kk;
      const Real z = mb_size(m).x3min + (k - ngh + 0.5)*dx3 - zorigin;
      const Real z2 = z*z;
      for (int j = ngh; j < ngh + nx2; ++j) {
        const Real y = mb_size(m).x2min + (j - ngh + 0.5)*dx2 - yorigin;
        const Real y2 = y*y;
        const Real yz = y*z;
        for (int i = ngh; i < ngh + nx1; ++i) {
          const Real x = mb_size(m).x1min + (i - ngh + 0.5)*dx1 - xorigin;
          const Real x2 = x*x;
          const Real xy = x*y;
          const Real zx = z*x;
          const Real r2 = x2 + y2 + z2;
          const Real s = src(m, 0, k, j, i)*vol;

          mp[0] += s;
          if (!skip_dipole) {
            mp[1] += s*y;
            mp[2] += s*z;
            mp[3] += s*x;
          }
          const Real hx2my2 = 0.5*(x2 - y2);
          mp[4] += s*xy;
          mp[5] += s*yz;
          mp[6] += s*(3.0*z2 - r2);
          mp[7] += s*zx;
          mp[8] += s*hx2my2;

          if (order == 4) {
            const Real tx2my2 = 3.0*x2 - y2;
            const Real x2mty2 = x2 - 3.0*y2;
            const Real fz2mr2 = 5.0*z2 - r2;
            mp[9]  += s*y*tx2my2;
            mp[10] += s*xy*z;
            mp[11] += s*y*fz2mr2;
            mp[12] += s*z*(z2 - 3.0*r2);
            mp[13] += s*x*fz2mr2;
            mp[14] += s*z*hx2my2;
            mp[15] += s*x*x2mty2;
            const Real sz2mr2 = 7.0*z2 - r2;
            const Real sz2mtr2 = 7.0*z2 - 3.0*r2;
            mp[16] += s*xy*hx2my2;
            mp[17] += s*yz*tx2my2;
            mp[18] += s*xy*sz2mr2;
            mp[19] += s*yz*sz2mtr2;
            mp[20] += s*(35.0*z2*z2 - 30.0*z2*r2 + 3.0*r2*r2);
            mp[21] += s*zx*sz2mtr2;
            mp[22] += s*hx2my2*sz2mr2;
            mp[23] += s*zx*x2mty2;
            mp[24] += s*0.125*(x2*x2mty2 - y2*tx2my2);
          }
        }
      }
    }
    // A compile-time bound is what keeps mp[] in registers.  Every column past ncoeff is
    // still the zero it was initialized to, and the host sum reads only ncoeff of them.
    for (int c = 0; c < 25; ++c) kpartial(m, kk, c) = mp[c];
  });

  Kokkos::parallel_for("MGMultipoleCoeffsFold",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                    Kokkos::Iterate::Right>>(
          exec, {0, 0}, {nmb, ncoeff}),
  KOKKOS_LAMBDA(const int m, const int c) {
    Real sum = 0.0;
    for (int kk = 0; kk < nx3; ++kk) sum += kpartial(m, kk, c);
    partial(m, c) = sum;
  });
}

template <typename ExecSpace, typename SrcView, typename SizeView,
          typename PartialView, typename KPartialView>
void ComputeCenterOfMassPartials(const ExecSpace &exec, const SrcView &src,
    const SizeView &mb_size, const PartialView &partial, const KPartialView &kpartial,
    int nmb, int ngh, int nx1, int nx2, int nx3) {
  if (nmb <= 0) return;
  Kokkos::parallel_for("MGCenterOfMass",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                    Kokkos::Iterate::Right>>(
          exec, {0, 0}, {nmb, nx3}),
  KOKKOS_LAMBDA(const int m, const int kk) {
    const Real dx1 = (mb_size(m).x1max - mb_size(m).x1min)
                   / static_cast<Real>(nx1);
    const Real dx2 = (mb_size(m).x2max - mb_size(m).x2min)
                   / static_cast<Real>(nx2);
    const Real dx3 = (mb_size(m).x3max - mb_size(m).x3min)
                   / static_cast<Real>(nx3);
    const Real vol = dx1*dx2*dx3;
    Real m0 = 0.0, my = 0.0, mz = 0.0, mx = 0.0, ma = 0.0;
    {
      const int k = ngh + kk;
      const Real z = mb_size(m).x3min + (k - ngh + 0.5)*dx3;
      for (int j = ngh; j < ngh + nx2; ++j) {
        const Real y = mb_size(m).x2min + (j - ngh + 0.5)*dx2;
        for (int i = ngh; i < ngh + nx1; ++i) {
          const Real x = mb_size(m).x1min + (i - ngh + 0.5)*dx1;
          const Real s = src(m, 0, k, j, i)*vol;
          m0 += s;
          my += s*y;
          mz += s*z;
          mx += s*x;
          ma += Kokkos::abs(s);
        }
      }
    }
    kpartial(m, kk, 0) = m0;
    kpartial(m, kk, 1) = my;
    kpartial(m, kk, 2) = mz;
    kpartial(m, kk, 3) = mx;
    kpartial(m, kk, 4) = ma;
  });

  Kokkos::parallel_for("MGCenterOfMassFold",
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                    Kokkos::Iterate::Right>>(
          exec, {0, 0}, {nmb, 5}),
  KOKKOS_LAMBDA(const int m, const int c) {
    Real sum = 0.0;
    for (int kk = 0; kk < nx3; ++kk) sum += kpartial(m, kk, c);
    partial(m, c) = sum;
  });
}
}  // namespace

// constructor, initializes data structures and parameters

MultigridDriver::MultigridDriver(MeshBlockPack *pmbp, int invar):
    nranks_(global_variable::nranks), nbtotal_(pmbp->pmesh->nmb_total),
    nvar_(invar),
    maxreflevel_(pmbp->pmesh->multilevel?pmbp->pmesh->max_level-pmbp->pmesh->root_level:0),
    nrbx1_(pmbp->pmesh->nmb_rootx1), nrbx2_(pmbp->pmesh->nmb_rootx2), nrbx3_(pmbp->pmesh->nmb_rootx3),
    pmy_pack_(pmbp),
    pmy_mesh_(pmbp->pmesh),
    needinit_(true), amr_mesh_changed_(false), nreflevel_(0),
    mesh_sig_(0), mesh_topology_version_(~static_cast<std::uint64_t>(0)),
    eps_(-1.0), last_defect_norm_(0.0), last_defect_valid_(false),
    unconverged_warn_cycle_(std::numeric_limits<int>::min()),
    niter_(-1), npresmooth_(1), npostsmooth_(1), coffset_(0),
    finest_exact_polish_passes_(6), same_exchange_stride_(1),
    local_sweeps_per_exchange_(1), reduced_exchange_max_edge_(4), fprolongation_(0),
    finest_bvals_fresh_(false), skip_coarser_initial_exchange_once_(false),
    active_reduce_same_exchange_(false),
    mg_tl_to_finer_valid_(false), mg_tl_to_coarser_valid_(false),
    mg_tl_fmg_prolongate_valid_(false), mg_tl_to_finer_nsmooth_(-1),
    mg_tl_to_finer_flag_(-1), mg_tl_to_finer_reduce_same_(false),
    mg_tl_to_coarser_nsmooth_(-1), mg_tl_to_coarser_skip_initial_(false),
    mg_tl_to_coarser_reduce_same_(false),
    defect_check_interval_(2), auto_target_rtol_(1.0e-6), source_norm_(0.0),
    auto_max_extra_cycles_(-1),
    coarsest_min_sweeps_(64), timing_vcycles_(0), timing_coarsest_solves_(0),
    timing_defect_norms_(0), collect_phase_timing_(false), timing_to_coarser_time_(0.0),
    timing_to_finer_time_(0.0), timing_coarsest_time_(0.0),
    timing_fmg_coarser_time_(0.0), timing_fmg_prolongate_time_(0.0),
    timing_transfer_blocks_root_time_(0.0), timing_transfer_root_blocks_time_(0.0),
    timing_octets_time_(0.0),
    finest_source_average_subtracted_(false),
    nb_rank_(0),
    octets_(nullptr), octetmap_(nullptr), octetbflag_(nullptr), octet_nb_same_(nullptr),
    octet_nb_coarse_(nullptr), octet_nb_rootloc_(nullptr), octet_parent_(nullptr),
    octet_child_i_(nullptr), octet_child_j_(nullptr), octet_child_k_(nullptr),
    octet_local_ids_(nullptr), noctets_(nullptr),
    octet_pool_(nullptr), octet_pool_stride_(nullptr),
    root_buf_nx_(0), root_buf_ny_(0), root_buf_nz_(0),
    root_flat_buf_stale_(true), root_uold_buf_valid_(false),
    root_host_authoritative_(false),
    coarse_scatter_mesh_sig_(0), coarse_scatter_block_count_(0),
    coarse_scatter_folddata_(false), distributed_coarse_solve_(false),
    coarse_owner_rank_(0) {
  if (pmy_mesh_->mb_indcs.nx2==1 || pmy_mesh_->mb_indcs.nx3==1) {
    std::cout << "### FATAL ERROR in MultigridDriver::MultigridDriver" << std::endl
        << "Currently the Multigrid solver works only in 3D." << std::endl;
    exit(EXIT_FAILURE);
    return;
  }
  for (int f = inner_x1; f <= outer_x3; ++f) {
    if (pmy_mesh_->mesh_bcs[f] == BoundaryFlag::periodic) {
      mg_mesh_bcs_[f] = BoundaryFlag::periodic;
    } else if (pmy_mesh_->mesh_bcs[f] == BoundaryFlag::outflow) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    } else if (pmy_mesh_->mesh_bcs[f] == BoundaryFlag::mg_zerofixed) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerofixed;
    } else if (pmy_mesh_->mesh_bcs[f] == BoundaryFlag::mg_multipole) {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_multipole;
    } else {
      mg_mesh_bcs_[f] = BoundaryFlag::mg_zerograd;
    }
  }
  ranklist_  = new int[nbtotal_];
  int nv = nvar_*2;
  Kokkos::realloc(rootbuf_, nbtotal_, nv);
  Kokkos::realloc(rootsrcbuf_, nbtotal_, nvar_);
  for (int n = 0; n < nbtotal_; ++n)
    ranklist_[n] = pmy_mesh_->rank_eachmb[n];
  nslist_  = new int[nranks_];
  nblist_  = new int[nranks_];
  nvlist_  = new int[nranks_];
  nvslist_ = new int[nranks_];
  nvlisti_  = new int[nranks_];
  nvslisti_ = new int[nranks_];
  // Allocate octet arrays for max possible refinement levels
  if (maxreflevel_ > 0) {
    octets_ = new std::vector<MGOctet>[maxreflevel_];
    octet_pool_ = new std::vector<Real>[maxreflevel_];
    octet_pool_stride_ = new std::size_t[maxreflevel_]();
    octetmap_ = new std::unordered_map<LogicalLocation, int, LogicalLocationHash>[maxreflevel_];
    octetbflag_ = new std::vector<bool>[maxreflevel_];
    octet_nb_same_ = new std::vector<int>[maxreflevel_];
    octet_nb_coarse_ = new std::vector<int>[maxreflevel_];
    octet_nb_rootloc_ = new std::vector<LogicalLocation>[maxreflevel_];
    octet_parent_ = new std::vector<int>[maxreflevel_];
    octet_child_i_ = new std::vector<int>[maxreflevel_];
    octet_child_j_ = new std::vector<int>[maxreflevel_];
    octet_child_k_ = new std::vector<int>[maxreflevel_];
    octet_local_ids_ = new std::vector<int>[maxreflevel_];
    noctets_ = new int[maxreflevel_]();
  }
}

//! destructor

MultigridDriver::~MultigridDriver() {
  delete [] ranklist_;
  delete [] nslist_;
  delete [] nblist_;
  delete [] nvlist_;
  delete [] nvslist_;
  delete [] nvlisti_;
  delete [] nvslisti_;
  delete [] octets_;
  delete [] octet_pool_;
  delete [] octet_pool_stride_;
  delete [] octetmap_;
  delete [] octetbflag_;
  delete [] octet_nb_same_;
  delete [] octet_nb_coarse_;
  delete [] octet_nb_rootloc_;
  delete [] octet_parent_;
  delete [] octet_child_i_;
  delete [] octet_child_j_;
  delete [] octet_child_k_;
  delete [] octet_local_ids_;
  delete [] noctets_;
}

bool MultigridDriver::UseReducedSameExchangeOnCurrentLevel(bool to_finer,
    int flag) const {
  if (same_exchange_stride_ <= 1 || global_variable::nranks <= 1) return false;
  // Use exact same-level halos on the first solve after an AMR hierarchy change.
  // This avoids setting the composite residual floor with reduced-exchange data
  // before the hierarchy has had a chance to settle.
  if (amr_mesh_changed_) return false;
  // Only the MeshBlock levels use the same-level exchange; the octet and root levels
  // are carried by MGRootBoundary and SetBoundariesOctets instead.
  if (current_level_ < nrootlevel_ + nreflevel_ - 1) return false;
  // The finest level carries the defect norm and the correction that reaches the
  // hydrodynamic potential, so it always sees a freshly exchanged halo.
  if (current_level_ == ntotallevel_ - 1) return false;
  if (to_finer && flag == 2) return false;
  // Trading extra local sweeps for fewer exchanges only pays where the message is
  // latency-bound.  A level whose blocks are e cells across sends e^2 cells per face,
  // so the trade is worth making below a threshold edge length and is a straight loss
  // above it: the fine levels are bandwidth-bound and their sweeps are the expensive
  // ones.  Measured on eight V100s on a uniform 128^3 Jeans wave, letting every level
  // relax cost 39 % more V-cycles AND a 10 % dearer V-cycle -- the exchange it removes
  // there was already hidden behind the interior sweep.
  const int shift = (ntotallevel_ - 1) - current_level_;
  const int edge = mglevels_->GetSize() >> shift;
  return (edge > 0) && (edge <= reduced_exchange_max_edge_);
}


namespace {
// The block edges MeshBlock::MeshBlock hands to block_rdx_: the two outermost edges are
// the mesh edges themselves, everything between them is interpolated.
Real MGBlockEdge(int lx, int nmbx, Real xmin, Real xmax) {
  if (lx <= 0) return xmin;
  if (lx >= nmbx) return xmax;
  return LeftEdgeX(lx, nmbx, xmin, xmax);
}

// Whether every block along one axis has bitwise the same width.  A block that
// redundantly updates a neighbour's cell multiplies the source by its OWN dx, so the
// two agree to the last bit only if the widths do.  This is a function of the mesh
// extent and the root block count alone, so every rank reaches the same answer without
// communicating -- which it must, since the schedule is a collective decision.
bool MGUniformBlockWidths(int nmbx, Real xmin, Real xmax) {
  const Real w0 = MGBlockEdge(1, nmbx, xmin, xmax) - MGBlockEdge(0, nmbx, xmin, xmax);
  for (int lx = 1; lx < nmbx; ++lx) {
    const Real w = MGBlockEdge(lx+1, nmbx, xmin, xmax)
                 - MGBlockEdge(lx, nmbx, xmin, xmax);
    if (w != w0) return false;
  }
  return true;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn bool MultigridDriver::UseExactHaloPair(int nsmooth, int level_shift)
//! \brief Whether a red/black pair on this MeshBlock level may share one halo exchange.
//!
//! With two valid ghost layers the red sweep can run over the interior grown by one
//! layer, and what it writes into that layer is bitwise what the neighbour writes into
//! its own interior, so the black sweep needs no exchange in between.  The conditions
//! below are exactly the ones under which that claim holds; each of them is the same on
//! every rank, because both sides of a message must agree on the schedule.

bool MultigridDriver::UseExactHaloPair(int nsmooth, int level_shift) const {
  if (nsmooth <= 0 || global_variable::nranks <= 1) return false;
  if (mglevels_ == nullptr) return false;
  // One ghost layer -- today's default -- leaves nothing for the grown sweep to read.
  if (mglevels_->GetGhostCells() < 2) return false;
  // The approximate reduced-exchange mode owns the schedule where it is enabled.
  if (active_reduce_same_exchange_) return false;
  // Refinement defeats the scheme, and the reason is structural rather than a missing
  // piece of plumbing.  The black sweep reads the whole post-red halo, so skipping the
  // exchange means every block has to recompute what each neighbour would have sent,
  // which it can only do if it holds everything that neighbour reads.
  //
  // Against a same-level face that holds: the halo is the neighbour's interior and the
  // depth-two exchange supplies the ring behind it.  Against a coarse-fine face it does
  // not.  MGFillFineCoarseLocalOnly writes the fine-side ghost as
  // (1/3)(2(cc +/- gy +/- gz) + u_interior): the coarse part is frozen while this level
  // smooths, so that much could be cached and reapplied between the colours, but the
  // fill covers only the interior tangential extent -- fj runs over [ngh, ngh+ncells) --
  // so the cell that is a ghost in the coarse-fine direction AND in a same-level
  // direction at once is written by nobody.  A grown sweep on the perpendicular face
  // reads exactly that cell.
  //
  // Neither obvious repair works.  Filling it means reproducing the neighbour's own
  // fill, which needs the coarse block reached only through an edge direction, that
  // block's interior clamping, and -- where the level changes along the face -- the
  // knowledge of whether the neighbour treats that direction as coarse-fine at all,
  // none of which this block holds.  Shrinking the grown box away from the coarse-fine
  // face instead leaves the black interior cells beside that face reading a stale halo,
  // so the face needs its exchange back, and the exchange is the round the scheme
  // exists to remove.  Since a refined mesh always has blocks against a level boundary
  // and the exchange is one collective round, it is all or nothing per level.
  //
  // It would also buy little: on a five-level collapse the finest level is a small part
  // of the cost, and removing fourteen sweeps and fifteen exchanges there moved the
  // V-cycle by one percent.
  if (nreflevel_ > 0) return false;
  if (!pmy_mesh_->three_d) return false;
  // The second ghost layer is the neighbour's second interior layer only while the
  // level still has two cells per block; below that it is the next block along, which
  // no same-level message carries.
  const int nc1 = mglevels_->indcs_.nx1 >> level_shift;
  const int nc2 = mglevels_->indcs_.nx2 >> level_shift;
  const int nc3 = mglevels_->indcs_.nx3 >> level_shift;
  if (nc1 < 2 || nc2 < 2 || nc3 < 2) return false;
  // Zero-gradient and zero-fixed faces copy or reflect an interior value, so the
  // tangential ghost row this scheme reads carries the same bits on both sides of the
  // face.  A multipole face instead evaluates the potential at the cell coordinate, and
  // the two blocks that name that one cell reach it by different arithmetic: the block
  // whose interior it is forms x2min + 0.5*dx, while the block reading it as a
  // tangential ghost forms its own x2min + (ncells + 0.5)*dx.  Those agree to round-off,
  // not bit for bit, so the grown sweep would not reproduce the neighbour's update.
  //
  // This one is fixable, unlike the refinement gate above: feeding CellCenterX the
  // global cell index and the mesh extent rather than the block's own origin makes both
  // blocks produce identical bits, at the price of moving every multipole ghost by a
  // rounding step.  It is not worth doing yet, because every deck here that selects the
  // multipole boundary is also refined, so the gate above would keep the scheme off
  // regardless and the only effect would be to perturb production results.
  for (int f = 0; f < 6; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) return false;
  }
  const RegionSize &ms = pmy_mesh_->mesh_size;
  if (!MGUniformBlockWidths(pmy_mesh_->nmb_rootx1, ms.x1min, ms.x1max) ||
      !MGUniformBlockWidths(pmy_mesh_->nmb_rootx2, ms.x2min, ms.x2max) ||
      !MGUniformBlockWidths(pmy_mesh_->nmb_rootx3, ms.x3min, ms.x3max)) {
    return false;
  }
  return true;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootToHost()
//! \brief Sync root grid arrays from device to host (no-op if root runs on host)

void MultigridDriver::SyncRootToHost() {
  if (mgroot_->on_host_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->u_[lev].h_view, mgroot_->u_[lev].d_view);
  Kokkos::deep_copy(exec, mgroot_->uold_[lev].h_view, mgroot_->uold_[lev].d_view);
  Kokkos::deep_copy(exec, mgroot_->src_[lev].h_view, mgroot_->src_[lev].d_view);
  exec.fence();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootDataToHost(bool include_old)
//! \brief Sync root data arrays (u and optionally uold) from device to host

void MultigridDriver::SyncRootDataToHost(bool include_old) {
  if (mgroot_->on_host_) return;
  // While a root->block transfer is in progress the host copy is the one being written,
  // so there is nothing to fetch and fetching would undo the octet restriction.
  if (root_host_authoritative_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->u_[lev].h_view, mgroot_->u_[lev].d_view);
  if (include_old) {
    Kokkos::deep_copy(exec, mgroot_->uold_[lev].h_view, mgroot_->uold_[lev].d_view);
  }
  exec.fence();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootToDevice()
//! \brief Sync root grid arrays from host to device (no-op if root runs on host)

void MultigridDriver::SyncRootToDevice() {
  if (mgroot_->on_host_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->u_[lev].d_view, mgroot_->u_[lev].h_view);
  Kokkos::deep_copy(exec, mgroot_->src_[lev].d_view, mgroot_->src_[lev].h_view);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootDataToDevice()
//! \brief Sync root data array (u) from host to device (no-op if root runs on host)

void MultigridDriver::SyncRootDataToDevice() {
  if (mgroot_->on_host_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->u_[lev].d_view, mgroot_->u_[lev].h_view);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootOldDataToDevice()
//! \brief Sync root old-data array (uold) from host to device (no-op if root runs on host)

void MultigridDriver::SyncRootOldDataToDevice() {
  if (mgroot_->on_host_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->uold_[lev].d_view, mgroot_->uold_[lev].h_view);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncRootSourceToDevice()
//! \brief Sync root source array from host to device (no-op if root runs on host)

void MultigridDriver::SyncRootSourceToDevice() {
  if (mgroot_->on_host_) return;
  int lev = mgroot_->current_level_;
  auto exec = DevExeSpace();
  Kokkos::deep_copy(exec, mgroot_->src_[lev].d_view, mgroot_->src_[lev].h_view);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SubtractAverage(MGVariable type)
//  \brief Calculate the global average and subtract it

void MultigridDriver::SubtractAverage(MGVariable type) {
  pmg->SubtractAverage(type,0,pmg->CalculateAverage(type));
  // The shift reaches the ghost cells too, so a periodic halo stays exact; a physical
  // face does not, because its ghost is a rule applied to the interior rather than a copy
  // of it.  Rather than decide which mesh this is, give up the skip.  It costs nothing:
  // the only caller that can reach here with the flag set is the end of SolveFMG, and it
  // is precisely the one that has to.
  finest_bvals_fresh_ = false;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial)
//  \brief initialize the source assuming that the source terms are already loaded

// amr_mesh_changed_ is deliberately NOT reset on entry.  PrepareForAMR() runs from three
// places -- the gravity solve, the post-AMR ghost refresh in MeshRefinement, and the
// post-solve ghost refresh -- but only the solve acts on the flag.  The bookkeeping below
// is idempotent (needinit_ is cleared on the way out, mesh_topology_version_ is latched),
// so the second call in a cycle takes the early return, and resetting here left the flag
// false before the solve ever read it.  The first solve after a hierarchy change then
// silently ran the "mesh unchanged" policy -- reduced same-level halos, and a warm-start
// cycle count taken from a defect the changed hierarchy has invalidated -- which is
// exactly backwards.  The flag is now sticky until ConsumeAMRMeshChanged() clears it at
// the end of a completed last-stage solve.
void MultigridDriver::PrepareForAMR() {
  // No claim about the finest halo survives out of the solve that made it.
  finest_bvals_fresh_ = false;
  locrootlevel_ = pmy_mesh_->root_level;

  // Detect if mesh has changed (AMR)
  int new_nbtotal = pmy_mesh_->nmb_total;
  const std::uint64_t new_topology_version = pmy_mesh_->topology_version;
  if (!needinit_ && new_nbtotal == nbtotal_ &&
      new_topology_version == mesh_topology_version_) {
    return;
  }
  if (new_nbtotal != nbtotal_) {
    nbtotal_ = new_nbtotal;
    delete[] ranklist_;
    ranklist_ = new int[nbtotal_];
    int nv = nvar_*2;
    Kokkos::realloc(rootbuf_, nbtotal_, nv);
    Kokkos::realloc(rootsrcbuf_, nbtotal_, nvar_);
    needinit_ = true;
  }

  // Calculate number of refinement levels present in mesh
  int old_nreflevel = nreflevel_;
  nreflevel_ = 0;
  std::uint64_t new_mesh_sig = 1469598103934665603ULL;
  auto hash_combine = [&new_mesh_sig](std::uint64_t v) {
    new_mesh_sig ^= v + 0x9e3779b97f4a7c15ULL + (new_mesh_sig << 6) + (new_mesh_sig >> 2);
  };
  for (int n = 0; n < nbtotal_; ++n) {
    const auto &ll = pmy_mesh_->lloc_eachmb[n];
    hash_combine(static_cast<std::uint64_t>(static_cast<std::uint32_t>(ll.lx1)));
    hash_combine(static_cast<std::uint64_t>(static_cast<std::uint32_t>(ll.lx2)));
    hash_combine(static_cast<std::uint64_t>(static_cast<std::uint32_t>(ll.lx3)));
    hash_combine(static_cast<std::uint64_t>(static_cast<std::uint32_t>(ll.level)));
    hash_combine(static_cast<std::uint64_t>(static_cast<std::uint32_t>(pmy_mesh_->rank_eachmb[n])));
    if (pmy_mesh_->multilevel) {
      int lev = ll.level - locrootlevel_;
      nreflevel_ = std::max(nreflevel_, lev);
    }
  }
  if (nreflevel_ != old_nreflevel) {
    needinit_ = true;
  }
  if (new_mesh_sig != mesh_sig_) {
    mesh_sig_ = new_mesh_sig;
    needinit_ = true;
  }
  mesh_topology_version_ = new_topology_version;
  amr_mesh_changed_ = amr_mesh_changed_ || needinit_;

  if (needinit_) {
    mglevels_->ReallocateForAMR();
    for (int n = 0; n < nbtotal_; ++n)
      ranklist_[n] = pmy_mesh_->rank_eachmb[n];
    for (int n = 0; n < nranks_; ++n) {
      nslist_[n]  = pmy_mesh_->gids_eachrank[n];
      nblist_[n]  = pmy_mesh_->nmb_eachrank[n];
      nvslist_[n] = nslist_[n]*nvar_*2;
      nvlist_[n]  = nblist_[n]*nvar_*2;
      nvslisti_[n] = nslist_[n]*nvar_;
      nvlisti_[n]  = nblist_[n]*nvar_;
    }
    if (nreflevel_ > 0) {
      InitializeOctets();
    }
    root_flat_buf_stale_ = true;
    root_uold_buf_valid_ = false;
  }
  needinit_ = false;
}

void MultigridDriver::RefreshMeshblockBoundaryValues(ParameterInput *) {
  if (pmy_pack_ == nullptr || mglevels_ == nullptr) return;
  // The object was built with the driver and outlives the regrid.  Only its cached
  // topology and its buffer capacity depend on the MeshBlock pack; the index metadata is
  // a function of the block geometry, which a regrid does not touch.  Rebuilding it
  // instead meant 112 device frees, each one a full device synchronisation, as many
  // allocations, and two MPI_Comm_dup collectives -- on a run that regrids every ten
  // cycles.
  mglevels_->pbval->RefreshForNewTopology();
}


void MultigridDriver::SetupMultigrid(Real dt, bool ftrivial) {
  (void)dt;
  (void)ftrivial;
  locrootlevel_ = pmy_mesh_->root_level;
  nrootlevel_ = mgroot_->GetNumberOfLevels();
  nmblevel_ = mglevels_->GetNumberOfLevels();

  // Include refinement levels in total (octets are V-cycle participants)
  ntotallevel_ = nrootlevel_ + nmblevel_ + nreflevel_ - 1;
  coarse_owner_rank_ = 0;
  distributed_coarse_solve_ = false;

  if (fsubtract_average_) {
    pmg = mglevels_;
    if (!finest_source_average_subtracted_) {
      SubtractAverage(MGVariable::src);
    }
  }
  finest_source_average_subtracted_ = false;
  current_level_ = ntotallevel_ - 1;
  fmglevel_ = current_level_;
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::InitializeOctets()
//! \brief Create per-cell octets for each refinement level

void MultigridDriver::InitializeOctets() {
  int ngh = mgroot_->ngh_;
  const auto &loc = pmy_mesh_->lloc_eachmb;

  // Clear and rebuild
  for (int l = 0; l < nreflevel_; ++l) {
    octetmap_[l].clear();
    noctets_[l] = 0;
  }

  // Scan all blocks: for each block at level > root, find its parent cell at each
  // octet level. An octet at level l has LogicalLocation at level (locrootlevel_ + l).
  // It represents a 2x2x2 group of cells from level (locrootlevel_ + l + 1).
  for (int n = 0; n < nbtotal_; ++n) {
    int blevel = loc[n].level;
    int blev = blevel - locrootlevel_;
    if (blev <= 0) continue;

    // For each refinement level from 0 to blev-1, this block implies an octet exists
    for (int l = blev - 1; l >= 0; --l) {
      LogicalLocation oloc;
      int shift = blevel - (locrootlevel_ + l + 1);
      oloc.lx1 = static_cast<int>(loc[n].lx1) >> (shift + 1);
      oloc.lx2 = static_cast<int>(loc[n].lx2) >> (shift + 1);
      oloc.lx3 = static_cast<int>(loc[n].lx3) >> (shift + 1);
      oloc.level = locrootlevel_ + l;

      if (octetmap_[l].count(oloc) == 0) {
        int oid = noctets_[l];
        octetmap_[l][oloc] = oid;
        noctets_[l]++;

        if (static_cast<int>(octets_[l].size()) <= oid) {
          octets_[l].resize(oid + 1);
        }
        MGOctet &oct = octets_[l][oid];
        oct.loc = oloc;
        oct.fleaf = false;
      }
    }
  }

  // Now that each level's octet count is known, give the level one contiguous pool and
  // point its octets into it.  Every active octet started zeroed before, whether it was
  // freshly allocated or reused, so zeroing the pool reproduces that exactly.
  {
    const int nc = 2 + 2*ngh;
    const std::size_t cellsz = static_cast<std::size_t>(nvar_)*nc*nc*nc;
    for (int l = 0; l < nreflevel_; ++l) {
      const std::size_t stride = static_cast<std::size_t>(noctets_[l])*cellsz;
      const std::size_t need = 4*stride;
      if (octet_pool_[l].size() < need) octet_pool_[l].resize(need);
      std::fill(octet_pool_[l].begin(), octet_pool_[l].begin() + need, 0.0);
      octet_pool_stride_[l] = stride;
      Real *base = octet_pool_[l].data();
      for (int o = 0; o < noctets_[l]; ++o)
        octets_[l][o].Attach(base, static_cast<std::size_t>(o), stride, nvar_, ngh);
    }
  }

  // Set fleaf for each octet: an octet is a leaf if none of its 8 children have octets
  for (int l = 0; l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &oct = octets_[l][o];
      oct.fleaf = true;
      if (l < nreflevel_ - 1) {
        const int child_level = oct.loc.level + 1;
        const int child_x1 = static_cast<int>(oct.loc.lx1) << 1;
        const int child_x2 = static_cast<int>(oct.loc.lx2) << 1;
        const int child_x3 = static_cast<int>(oct.loc.lx3) << 1;
        for (int ck = 0; ck <= 1 && oct.fleaf; ++ck) {
          for (int cj = 0; cj <= 1 && oct.fleaf; ++cj) {
            for (int ci = 0; ci <= 1; ++ci) {
              LogicalLocation child_loc;
              child_loc.lx1 = child_x1 + ci;
              child_loc.lx2 = child_x2 + cj;
              child_loc.lx3 = child_x3 + ck;
              child_loc.level = child_level;
              if (octetmap_[l + 1].find(child_loc) != octetmap_[l + 1].end()) {
                oct.fleaf = false;
                break;
              }
            }
          }
        }
      }
    }
  }

  // Precompute MeshBlock -> parent octet lookup (used in boundary setup hot paths)
  mb_octet_lev_.assign(nbtotal_, -1);
  mb_octet_oid_.assign(nbtotal_, -1);
  mb_octet_i_.assign(nbtotal_, -1);
  mb_octet_j_.assign(nbtotal_, -1);
  mb_octet_k_.assign(nbtotal_, -1);
  for (int n = 0; n < nbtotal_; ++n) {
    if (loc[n].level <= locrootlevel_) continue;
    LogicalLocation ploc;
    ploc.lx1 = (loc[n].lx1 >> 1);
    ploc.lx2 = (loc[n].lx2 >> 1);
    ploc.lx3 = (loc[n].lx3 >> 1);
    ploc.level = loc[n].level - 1;
    int lev = ploc.level - locrootlevel_;
    auto pit = octetmap_[lev].find(ploc);
    if (pit != octetmap_[lev].end()) {
      mb_octet_lev_[n] = lev;
      mb_octet_oid_[n] = pit->second;
      mb_octet_i_[n] = (static_cast<int>(loc[n].lx1) & 1) + ngh;
      mb_octet_j_[n] = (static_cast<int>(loc[n].lx2) & 1) + ngh;
      mb_octet_k_[n] = (static_cast<int>(loc[n].lx3) & 1) + ngh;
    }
  }

  // Precompute per-octet parent lookup and child-cell indices.
  for (int l = 0; l < nreflevel_; ++l) {
    const int no = noctets_[l];
    octet_parent_[l].assign(no, -1);
    octet_child_i_[l].assign(no, ngh);
    octet_child_j_[l].assign(no, ngh);
    octet_child_k_[l].assign(no, ngh);
  }
  if (nreflevel_ > 0) {
    octet_root_i_.assign(noctets_[0], ngh);
    octet_root_j_.assign(noctets_[0], ngh);
    octet_root_k_.assign(noctets_[0], ngh);
  } else {
    octet_root_i_.clear();
    octet_root_j_.clear();
    octet_root_k_.clear();
  }
  for (int l = 0; l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      const LogicalLocation &oloc = octets_[l][o].loc;
      octet_child_i_[l][o] = (static_cast<int>(oloc.lx1) & 1) + ngh;
      octet_child_j_[l][o] = (static_cast<int>(oloc.lx2) & 1) + ngh;
      octet_child_k_[l][o] = (static_cast<int>(oloc.lx3) & 1) + ngh;
      if (l == 0) {
        octet_root_i_[o] = static_cast<int>(oloc.lx1) + ngh;
        octet_root_j_[o] = static_cast<int>(oloc.lx2) + ngh;
        octet_root_k_[o] = static_cast<int>(oloc.lx3) + ngh;
      } else {
        LogicalLocation cloc;
        cloc.lx1 = (oloc.lx1 >> 1);
        cloc.lx2 = (oloc.lx2 >> 1);
        cloc.lx3 = (oloc.lx3 >> 1);
        cloc.level = oloc.level - 1;
        auto pit = octetmap_[l-1].find(cloc);
        if (pit != octetmap_[l-1].end()) {
          octet_parent_[l][o] = pit->second;
        }
      }
    }
  }

  // Precompute local octet IDs that require boundaries before root->block transfer.
  for (int l = 0; l < nreflevel_; ++l) {
    octet_local_ids_[l].clear();
  }
  if (nreflevel_ > 0) {
    std::vector<std::vector<unsigned char>> seen(nreflevel_);
    for (int l = 0; l < nreflevel_; ++l) {
      seen[l].assign(noctets_[l], 0);
    }
    const int padding = nslist_[global_variable::my_rank];
    for (int m = 0; m < mglevels_->nmmb_; ++m) {
      const int gid = m + padding;
      const int lev = mb_octet_lev_[gid];
      if (lev < 0) continue;
      const int oid = mb_octet_oid_[gid];
      if (oid < 0) continue;
      if (!seen[lev][oid]) {
        seen[lev][oid] = 1;
        octet_local_ids_[lev].push_back(oid);
      }
    }
  }

  // Precompute periodic same-level and coarse-level octet neighbor IDs to avoid
  // unordered_map lookups in per-cycle boundary loops.
  for (int l = 0; l < nreflevel_; ++l) {
    const int no = noctets_[l];
    octet_nb_same_[l].assign(no * 27, -1);
    octet_nb_coarse_[l].assign(no * 27, -1);
    octet_nb_rootloc_[l].assign(no * 27, LogicalLocation());
    const int nx1 = (nrbx1_ << l);
    const int nx2 = (nrbx2_ << l);
    const int nx3 = (nrbx3_ << l);

    for (int o = 0; o < no; ++o) {
      const LogicalLocation &oloc = octets_[l][o].loc;
      const int base = o * 27;

      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            const int nidx = (ox3 + 1) * 9 + (ox2 + 1) * 3 + (ox1 + 1);
            LogicalLocation nloc = oloc;
            nloc.lx1 = static_cast<int>(oloc.lx1) + ox1;
            nloc.lx2 = static_cast<int>(oloc.lx2) + ox2;
            nloc.lx3 = static_cast<int>(oloc.lx3) + ox3;
            bool valid = true;
            if (nloc.lx1 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x1] == BoundaryFlag::periodic) nloc.lx1 = nx1 - 1;
              else
                valid = false;
            } else if (nloc.lx1 >= nx1) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x1] == BoundaryFlag::periodic) nloc.lx1 = 0;
              else
                valid = false;
            }
            if (nloc.lx2 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x2] == BoundaryFlag::periodic) nloc.lx2 = nx2 - 1;
              else
                valid = false;
            } else if (nloc.lx2 >= nx2) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x2] == BoundaryFlag::periodic) nloc.lx2 = 0;
              else
                valid = false;
            }
            if (nloc.lx3 < 0) {
              if (mg_mesh_bcs_[BoundaryFace::inner_x3] == BoundaryFlag::periodic) nloc.lx3 = nx3 - 1;
              else
                valid = false;
            } else if (nloc.lx3 >= nx3) {
              if (mg_mesh_bcs_[BoundaryFace::outer_x3] == BoundaryFlag::periodic) nloc.lx3 = 0;
              else
                valid = false;
            }

            if (valid) octet_nb_rootloc_[l][base + nidx] = nloc;
            else
              octet_nb_rootloc_[l][base + nidx] = LogicalLocation{0, 0, 0, -1};
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            if (!valid) continue;

            auto sit = octetmap_[l].find(nloc);
            if (sit != octetmap_[l].end()) {
              octet_nb_same_[l][base + nidx] = sit->second;
            } else if (l > 0) {
              LogicalLocation cloc;
              cloc.lx1 = nloc.lx1 >> 1;
              cloc.lx2 = nloc.lx2 >> 1;
              cloc.lx3 = nloc.lx3 >> 1;
              cloc.level = nloc.level - 1;
              auto cit = octetmap_[l-1].find(cloc);
              if (cit != octetmap_[l-1].end()) {
                octet_nb_coarse_[l][base + nidx] = cit->second;
              }
            }
          }
        }
      }
    }
  }

  // Allocate scratch buffers for boundary exchange
  int nv = std::max(nvar_, 1);
  int cbnc = 3;  // coarse buffer is 3x3x3
  cbuf_.assign(nv * cbnc * cbnc * cbnc, 0.0);
  cbufold_.assign(nv * cbnc * cbnc * cbnc, 0.0);
  ncoarse_.fill(false);
}


void MultigridDriver::BuildRootFlatBuffers(bool include_old) {
  if (!root_flat_buf_stale_ && (!include_old || root_uold_buf_valid_)) return;
  SyncRootDataToHost(include_old);
  auto root_u_h = GetRootData_h();
  int rnx = root_u_h.extent_int(4);
  int rny = root_u_h.extent_int(3);
  int rnz = root_u_h.extent_int(2);
  int rnv = root_u_h.extent_int(1);
  int total = rnv*rnz*rny*rnx;
  if (root_flat_buf_stale_) {
    root_buf_nx_ = rnx;
    root_buf_ny_ = rny;
    root_buf_nz_ = rnz;
    root_u_buf_.resize(total);
    std::memcpy(root_u_buf_.data(), root_u_h.data(), total*sizeof(Real));
    root_flat_buf_stale_ = false;
  }
  if (include_old && !root_uold_buf_valid_) {
    auto root_uold_h = GetRootOldData_h();
    root_uold_buf_.resize(total);
    std::memcpy(root_uold_buf_.data(), root_uold_h.data(), total*sizeof(Real));
    root_uold_buf_valid_ = true;
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RefreshRootFlatBuffersFromHost()
//! \brief Rebuild the flat root-u buffer from the host copy, skipping the device fetch
//!
//! Call this only where the host and device copies of the root data are known to hold
//! the same bytes -- immediately after a push to the device.  The fetch that
//! BuildRootFlatBuffers() would otherwise do on the next octet boundary pass would
//! return exactly these values.

void MultigridDriver::RefreshRootFlatBuffersFromHost() {
  auto root_u_h = GetRootData_h();
  const int rnx = root_u_h.extent_int(4);
  const int rny = root_u_h.extent_int(3);
  const int rnz = root_u_h.extent_int(2);
  const int total = root_u_h.extent_int(1)*rnz*rny*rnx;
  root_buf_nx_ = rnx;
  root_buf_ny_ = rny;
  root_buf_nz_ = rnz;
  root_u_buf_.resize(total);
  std::memcpy(root_u_buf_.data(), root_u_h.data(), total*sizeof(Real));
  root_flat_buf_stale_ = false;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromBlocksToRoot(bool initflag)
//! \brief collect the coarsest data and transfer to the root grid

// The entry fence belongs before the timer starts, not after it.  Started first, the
// timer charges this phase with the drain of every kernel the *previous* phase left in
// flight, which is how a transfer of a few hundred numbers came to read as a third of
// the solve: the restriction kernels it follows were being billed to it.
void MultigridDriver::TransferFromBlocksToRoot(bool initflag) {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  const int nv = nvar_;
  auto rootbuf = rootbuf_;
  auto rootsrcbuf = rootsrcbuf_;
  const auto &src = mglevels_->src_[0].d_view;
  const auto &u = mglevels_->u_[0].d_view;
  const int ngh_mb = mglevels_->ngh_;
  int nmmb = mglevels_->nmmb_ - 1;
  int padding = nslist_[global_variable::my_rank];
  if (initflag) {
    par_for("Multigrid:SaveToRootSrc", DevExeSpace(), 0, nmmb,
        KOKKOS_LAMBDA(const int m) {
      for (int v = 0; v < nv; ++v) {
        rootsrcbuf.d_view(m+padding, v) = src(m, v, ngh_mb, ngh_mb, ngh_mb);
      }
    });
    rootsrcbuf.template modify<DevExeSpace>();
    rootsrcbuf.template sync<HostExeSpace>();
  } else {
    par_for("Multigrid:SaveToRootSrcU", DevExeSpace(), 0, nmmb,
        KOKKOS_LAMBDA(const int m) {
      for (int v = 0; v < nv; ++v) {
        rootbuf.d_view(m+padding, v)    = src(m, v, ngh_mb, ngh_mb, ngh_mb);
        rootbuf.d_view(m+padding, v+nv) = u(m, v, ngh_mb, ngh_mb, ngh_mb);
      }
    });
    rootbuf.template modify<DevExeSpace>();
    rootbuf.template sync<HostExeSpace>();
  }
  const int my_rank = global_variable::my_rank;
  const bool owner_only_transfer = UseDistributedCoarseSolve() && (nranks_ > 1);
  const bool is_coarse_owner = (!owner_only_transfer || (my_rank == coarse_owner_rank_));
#if MPI_PARALLEL_ENABLED
  const int send_count = initflag ? nvlisti_[my_rank] : nvlist_[my_rank];
  int *recv_counts = initflag ? nvlisti_ : nvlist_;
  int *recv_displs = initflag ? nvslisti_ : nvslist_;
  if (owner_only_transfer) {
    Real *send_ptr = initflag ? &(rootsrcbuf.h_view(nslist_[my_rank], 0))
                              : &(rootbuf.h_view(nslist_[my_rank], 0));
    Real *recv_ptr = nullptr;
    if (is_coarse_owner) {
      recv_ptr = initflag ? &(rootsrcbuf.h_view(0, 0)) : &(rootbuf.h_view(0, 0));
    }
    MPI_Gatherv(is_coarse_owner ? MPI_IN_PLACE : send_ptr,
                send_count, MPI_ATHENA_REAL,
                recv_ptr, recv_counts, recv_displs,
                MPI_ATHENA_REAL, coarse_owner_rank_, MPI_COMM_WORLD);
  } else {
    Real *recv_ptr = initflag ? &(rootsrcbuf.h_view(0,0)) : &(rootbuf.h_view(0,0));
    MPI_Allgatherv(MPI_IN_PLACE, send_count, MPI_ATHENA_REAL,
                   recv_ptr, recv_counts, recv_displs,
                   MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif

  if (is_coarse_owner) {
    const auto loc = pmy_mesh_->lloc_eachmb;
    int rootlevel = locrootlevel_;
    int ngh = mgroot_->ngh_;

    auto root_src_h = GetRootSource_h();
    auto root_u_h = GetRootData_h();

    for (int n = 0; n < nbtotal_; ++n) {
        int i = static_cast<int>(loc[n].lx1);
        int j = static_cast<int>(loc[n].lx2);
        int k = static_cast<int>(loc[n].lx3);
      if (loc[n].level == rootlevel) {
        for (int v = 0; v < nv; ++v) {
          const Real src_val = initflag ? rootsrcbuf.h_view(n, v) : rootbuf.h_view(n, v);
          root_src_h(0, v, k+ngh, j+ngh, i+ngh) = src_val;
          if (!initflag)
            root_u_h(0, v, k+ngh, j+ngh, i+ngh) = rootbuf.h_view(n, v+nv);
        }
      } else {
        // Refined block -> appropriate octet
        int olev = mb_octet_lev_[n];
        int oid = mb_octet_oid_[n];
        int oi = mb_octet_i_[n];
        int oj = mb_octet_j_[n];
        int ok = mb_octet_k_[n];
        if (olev < 0 || oid < 0) continue;
        MGOctet &oct = octets_[olev][oid];
        for (int v = 0; v < nv; ++v) {
          const Real src_val = initflag ? rootsrcbuf.h_view(n, v) : rootbuf.h_view(n, v);
          oct.Src(v, ok, oj, oi) = src_val;
          if (!initflag)
            oct.U(v, ok, oj, oi) = rootbuf.h_view(n, v+nv);
        }
      }
    }
  }
  root_flat_buf_stale_ = true;
  root_uold_buf_valid_ = false;
  mgroot_->current_level_ = nrootlevel_ - 1;
  if (is_coarse_owner) {
    SyncRootSourceToDevice();
    if (!initflag) {
      SyncRootDataToDevice();
      // The push above leaves both copies identical, so the flat buffer the octet
      // boundary pass wants next can be taken straight from the host array instead of
      // being fetched back off the device.
      if (nreflevel_ > 0) RefreshRootFlatBuffersFromHost();
    }
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_transfer_blocks_root_time_ += timer.seconds();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::TransferFromRootToBlocks(bool folddata)
//! \brief Transfer data from root/octets to block coarsest levels

void MultigridDriver::TransferFromRootToBlocks(bool folddata) {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  // The whole transfer works on the host copy of the root arrays: the octet
  // restriction writes it, the octet boundary fill and SetFromRootGrid read it.  So
  // fetch once at the top and push once at the end.  Previously the restriction pushed
  // to the device immediately, purely so that the two fetches that follow would come
  // back with the values it had just written -- three copies of the root grid and two
  // extra stream drains per V-cycle leg for no change in content.
  if (nreflevel_ > 0) {
    SyncRootDataToHost(folddata);
    root_host_authoritative_ = true;
    RestrictOctetsBeforeTransfer();
    SetOctetBoundariesBeforeTransfer(folddata);
    root_host_authoritative_ = false;
    SyncRootDataToDevice();
  } else {
    SyncRootDataToHost(folddata);
  }
  mglevels_->SetFromRootGrid(folddata);
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_transfer_root_blocks_time_ += timer.seconds();
  }
  return;
}


void MultigridDriver::TransferFromRootToBlocksDistributed(bool folddata) {
#if MPI_PARALLEL_ENABLED
  if (!UseDistributedCoarseSolve() || nranks_ <= 1) {
    TransferFromRootToBlocks(folddata);
    return;
  }
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;

  const bool is_owner = (global_variable::my_rank == coarse_owner_rank_);
  const int nc = 2 * mgroot_->ngh_ + 1;
  const int per_var_count = nc * nc * nc;
  const int per_block_count = nvar_ * per_var_count * (folddata ? 2 : 1);
  const int recv_count = nblist_[global_variable::my_rank] * per_block_count;
  if (coarse_scatter_mesh_sig_ != mesh_sig_
      || coarse_scatter_block_count_ != per_block_count
      || coarse_scatter_folddata_ != folddata
      || static_cast<int>(coarse_scatter_counts_.size()) != nranks_) {
    coarse_scatter_counts_.resize(nranks_);
    coarse_scatter_displs_.resize(nranks_);
    for (int r = 0; r < nranks_; ++r) {
      coarse_scatter_counts_[r] = nblist_[r] * per_block_count;
      coarse_scatter_displs_[r] = nslist_[r] * per_block_count;
    }
    coarse_scatter_block_count_ = per_block_count;
    coarse_scatter_folddata_ = folddata;
    coarse_scatter_mesh_sig_ = mesh_sig_;
  }
  if (static_cast<int>(coarse_scatter_recv_buf_.size()) < recv_count) {
    coarse_scatter_recv_buf_.resize(recv_count);
  }
  if (is_owner) {
    const int send_count = nbtotal_ * per_block_count;
    if (static_cast<int>(coarse_scatter_send_buf_.size()) < send_count) {
      coarse_scatter_send_buf_.resize(send_count);
    }
  }
  Real *recvbuf = (recv_count > 0) ? coarse_scatter_recv_buf_.data() : nullptr;
  Real *sendbuf = is_owner ? coarse_scatter_send_buf_.data() : nullptr;

  if (is_owner) {
    MGRootBoundary();
    // One fetch and one push around the host-side coarse work; see the comment in
    // TransferFromRootToBlocks().  The owner packs the send buffer from the host copy,
    // so the push exists only to leave the device copy consistent for the next leg.
    if (nreflevel_ > 0) {
      SyncRootDataToHost(folddata);
      root_host_authoritative_ = true;
      RestrictOctetsBeforeTransfer();
      SetOctetBoundariesBeforeTransfer(folddata);
      root_host_authoritative_ = false;
      SyncRootDataToDevice();
    } else {
      SyncRootDataToHost(folddata);
    }

    auto src_h = GetRootData_h();
    auto osrc_h = GetRootOldData_h();

    for (int n = 0; n < nbtotal_; ++n) {
      const auto &loc = pmy_mesh_->lloc_eachmb[n];
      const int lev = loc.level - locrootlevel_;
      Real *dst = sendbuf + static_cast<std::size_t>(n) * per_block_count;
      int idx = 0;
      if (lev == 0) {
        const int ci = static_cast<int>(loc.lx1);
        const int cj = static_cast<int>(loc.lx2);
        const int ck = static_cast<int>(loc.lx3);
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int i = 0; i < nc; ++i) {
                dst[idx++] = src_h(0, v, ck + k, cj + j, ci + i);
              }
            }
          }
        }
        if (folddata) {
          for (int v = 0; v < nvar_; ++v) {
            for (int k = 0; k < nc; ++k) {
              for (int j = 0; j < nc; ++j) {
                for (int i = 0; i < nc; ++i) {
                  dst[idx++] = osrc_h(0, v, ck + k, cj + j, ci + i);
                }
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
        const int olev = oloc.level - locrootlevel_;
        const int oid = octetmap_[olev][oloc];
        const int ci = (static_cast<int>(loc.lx1) & 1);
        const int cj = (static_cast<int>(loc.lx2) & 1);
        const int ck = (static_cast<int>(loc.lx3) & 1);
        const MGOctet &oct = octets_[olev][oid];
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int i = 0; i < nc; ++i) {
                dst[idx++] = oct.U(v, ck + k, cj + j, ci + i);
              }
            }
          }
        }
        if (folddata) {
          for (int v = 0; v < nvar_; ++v) {
            for (int k = 0; k < nc; ++k) {
              for (int j = 0; j < nc; ++j) {
                for (int i = 0; i < nc; ++i) {
                  dst[idx++] = oct.Uold(v, ck + k, cj + j, ci + i);
                }
              }
            }
          }
        }
      }
    }
  }

  MPI_Scatterv(sendbuf,
               coarse_scatter_counts_.data(),
               coarse_scatter_displs_.data(),
               MPI_ATHENA_REAL,
               recvbuf,
               recv_count,
               MPI_ATHENA_REAL,
               coarse_owner_rank_,
               MPI_COMM_WORLD);

  mglevels_->current_level_ = 0;
  const int nmmb = mglevels_->nmmb_;
  auto dst_h = mglevels_->u_[0].h_view;
  auto odst_h = mglevels_->uold_[0].h_view;
  for (int m = 0; m < nmmb; ++m) {
    const Real *src = recvbuf + static_cast<std::size_t>(m) * per_block_count;
    int idx = 0;
    for (int v = 0; v < nvar_; ++v) {
      for (int k = 0; k < nc; ++k) {
        for (int j = 0; j < nc; ++j) {
          for (int i = 0; i < nc; ++i) {
            dst_h(m, v, k, j, i) = src[idx++];
          }
        }
      }
    }
    if (folddata) {
      for (int v = 0; v < nvar_; ++v) {
        for (int k = 0; k < nc; ++k) {
          for (int j = 0; j < nc; ++j) {
            for (int i = 0; i < nc; ++i) {
              odst_h(m, v, k, j, i) = src[idx++];
            }
          }
        }
      }
    }
  }
  mglevels_->u_[0].template modify<HostExeSpace>();
  mglevels_->u_[0].template sync<DevExeSpace>();
  if (folddata) {
    mglevels_->uold_[0].template modify<HostExeSpace>();
    mglevels_->uold_[0].template sync<DevExeSpace>();
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_transfer_root_blocks_time_ += timer.seconds();
  }
  return;
#else
  TransferFromRootToBlocks(folddata);
#endif
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FMGProlongate(Driver *pdriver)
//! \brief FMG prolongation one level

void MultigridDriver::FMGProlongate(Driver *pdriver) {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  finest_bvals_fresh_ = false;
  int ngh = mgroot_->ngh_;
  if (current_level_ == nrootlevel_ + nreflevel_ - 1) {
    if (UseDistributedCoarseSolve()) {
      TransferFromRootToBlocksDistributed(false);
    } else {
      MGRootBoundary();
      TransferFromRootToBlocks(false);
    }
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    SetMGTaskListFMGProlongate(ngh);
    pdriver->ExecuteTaskList(pmy_mesh_, "mg_fmg_prolongate", 0);
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (IsCoarseSolveOwner()) {
      if (current_level_ == nrootlevel_ - 1)
        MGRootBoundary();
      else
        SetBoundariesOctets(true, false);
      FMGProlongateOctets();
    }
    current_level_++;
  } else { // root grid
    if (IsCoarseSolveOwner()) {
      MGRootBoundary();
      mgroot_->FMGProlongatePack();
    }
    current_level_++;
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_fmg_prolongate_time_ += timer.seconds();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::OneStepToFiner(int nsmooth)
//! \brief prolongation and smoothing one level

void MultigridDriver::OneStepToFiner(Driver *pdriver, int nsmooth) {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  int ngh = mgroot_->ngh_;
  if (current_level_ == nrootlevel_ + nreflevel_ - 1) {
    if (UseDistributedCoarseSolve()) {
      TransferFromRootToBlocksDistributed(true);
    } else {
      MGRootBoundary();
      TransferFromRootToBlocks(true);
    }
  }
  if (current_level_ >= nrootlevel_ + nreflevel_ - 1) { // MeshBlocks
    pmg = mglevels_;
    int flag = 0;
    // flag = 1: first time on meshblock levels
    if (current_level_ == nrootlevel_ + nreflevel_ - 1) flag = 1;
    // flag = 2: last step to the finest level, whose tail leaves the halo exact.  This
    // used to be conditional on eps_ >= 0, that is, on the solve being the one that
    // iterates on a defect norm; but the fixed-count solve reads the same norm when it
    // records last_defect_norm_, and it was reading it against ghosts one black sweep
    // out of date.  How many passes the tail runs is finest_exact_polish_passes_, and
    // that is the only knob it should have.
    if (current_level_ == ntotallevel_ - 2) flag = 2;
    // The two conditions above are not alternatives, they are separate facts about this
    // step, and a MeshBlock with only two multigrid levels satisfies both at once -- for
    // which the single int silently dropped the first.  Name them apart and let the
    // packed value keep its old precedence, since the halo schedule below reads it.
    const bool entering_block_levels =
        (current_level_ == nrootlevel_ + nreflevel_ - 1);
    const bool finest_level_tail = (current_level_ == ntotallevel_ - 2);
    const bool reduce_same_exchange = UseReducedSameExchangeOnCurrentLevel(true, flag);
    active_reduce_same_exchange_ = reduce_same_exchange;
    if (global_variable::nranks == 1) {
      auto update_boundaries = [&](bool include_diagonals = false) {
        DvceArray5D<Real> u = pmg->GetCurrentData();
        (void)pmg->pbval->RecvAndUnpackMG(u, include_diagonals);
        if (include_diagonals) {
          (void)FillFCBoundaryForProlongation(pdriver, 0);
          (void)PhysicalBoundaryForCorrectionProlongation(pdriver, 0);
        } else {
          (void)PhysicalBoundary(pdriver, 0);
          (void)FillFCBoundary(pdriver, 0);
        }
      };

      if (entering_block_levels) {
        pmg->ProlongateAndCorrectPack();
      } else {
        pmg->ComputeCorrection();
        update_boundaries(true);
        pmg->ProlongatePreparedCorrectionPack();
      }

      if (nsmooth > 0) {
        update_boundaries();
        for (int n = 0; n < nsmooth; ++n) {
          pmg->SmoothPack(coffset_);
          update_boundaries();
          pmg->SmoothPack(1 - coffset_);
          if (n < nsmooth - 1) {
            update_boundaries();
          }
        }
      }

      // The same exact-halo polish the multi-rank branch runs below.  What it repairs is
      // the residual that sits on coarse-fine level boundaries, and a mesh has the same
      // level boundaries however many ranks it is spread over, so running it on one rank
      // and not on two made the answer depend on the decomposition.
      if (finest_level_tail) {
        const int npass = std::max(1, finest_exact_polish_passes_);
        for (int pass = 0; pass < npass; ++pass) {
          update_boundaries();
          pmg->SmoothPack(coffset_);
          update_boundaries();
          pmg->SmoothPack(1 - coffset_);
        }
        update_boundaries();
      }
    } else {
      auto refresh_boundaries = [&](bool include_diagonals = false) {
        DvceArray5D<Real> u = pmg->GetCurrentData();
        while (pmg->pbval->InitRecvMG(pmg->nvar_, include_diagonals)
               == TaskStatus::incomplete) {}
        while (pmg->pbval->PackAndSendMG(u, include_diagonals)
               == TaskStatus::incomplete) {}
        // Face-only smoothers can overlap physical BC work with remote traffic.
        // Prolongation needs physical edges/corners derived from completed tangential
        // halos, so its full physical fill must follow both communication phases.
        if (!include_diagonals) (void)PhysicalBoundary(pdriver, 0);
        while (pmg->pbval->RecvAndUnpackMG(u, include_diagonals)
               == TaskStatus::incomplete) {}
        if (include_diagonals) {
          (void)FillFCBoundaryForProlongation(pdriver, 0);
          (void)PhysicalBoundaryForCorrectionProlongation(pdriver, 0);
        } else {
          (void)FillFCBoundary(pdriver, 0);
        }
        // Do not force-drain same-level and fine/coarse sends here. The next MG
        // boundary refresh already polls/waits before reusing the request slots
        // or repacking send storage, so skipping the explicit clear keeps more
        // communication in flight across smoother steps.
      };
      auto prolongate_with_fresh_diagonals = [&]() {
        if (entering_block_levels) {
          pmg->ProlongateAndCorrectPack();
        } else {
          pmg->ComputeCorrection();
          refresh_boundaries(true);
          pmg->ProlongatePreparedCorrectionPack();
        }
      };
      auto *gpmg = dynamic_cast<MGGravity*>(pmg);
      auto overlapped_color = [&](int color) {
        if (gpmg == nullptr) {
          refresh_boundaries();
          pmg->SmoothPack(color);
          return;
        }
        DvceArray5D<Real> u = pmg->GetCurrentData();
        while (pmg->pbval->InitRecvMG(pmg->nvar_) == TaskStatus::incomplete) {}
        while (pmg->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
        // Physical boundaries do not depend on incoming same-level halos. Update
        // them while remote face data is in flight, then smooth the strictly
        // interior cells that do not touch any ghost layer.
        (void)PhysicalBoundary(pdriver, 0);
        gpmg->SmoothPackInterior(color);
        while (pmg->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
        (void)FillFCBoundary(pdriver, 0);
        gpmg->SmoothPackBoundary(color);
      };
      auto exact_finest_polish = [&]() {
        const int npass = std::max(1, finest_exact_polish_passes_);
        if (gpmg == nullptr) {
          for (int pass = 0; pass < npass; ++pass) {
            refresh_boundaries();
            pmg->SmoothPack(coffset_);
            refresh_boundaries();
            pmg->SmoothPack(1 - coffset_);
          }
          refresh_boundaries();
          return;
        }

        for (int pass = 0; pass < npass; ++pass) {
          overlapped_color(coffset_);
          overlapped_color(1 - coffset_);
        }
        refresh_boundaries();
      };
      const bool manual_exact_fine_overlap =
          (finest_level_tail && gpmg != nullptr && !reduce_same_exchange);
      // The prolongation below advances the Multigrid one level, so the sweeps run one
      // level finer than the shift this object reports right now.
      const bool exact_halo_pair =
          (gpmg != nullptr && UseExactHaloPair(nsmooth, pmg->GetLevelShift() - 1));
      if (exact_halo_pair) {
        // See OneStepToCoarser: one exchange carries the pair, and it has to be the
        // deep one with diagonals.
        auto deep_refresh = [&]() {
          DvceArray5D<Real> u = pmg->GetCurrentData();
          pmg->pbval->SetMGSameHaloDepth(2);
          while (pmg->pbval->InitRecvMG(pmg->nvar_, true) == TaskStatus::incomplete) {}
          while (pmg->pbval->PackAndSendMG(u, true) == TaskStatus::incomplete) {}
          while (pmg->pbval->RecvAndUnpackMG(u, true) == TaskStatus::incomplete) {}
          pmg->pbval->SetMGSameHaloDepth(1);
          ApplyPhysicalBoundariesBlocks(true);
        };
        prolongate_with_fresh_diagonals();
        deep_refresh();
        for (int n = 0; n < nsmooth; ++n) {
          gpmg->SmoothPackExpanded(coffset_, 1);
          ApplyPhysicalBoundariesBlocks(true);
          pmg->SmoothPack(1 - coffset_);
          if (n < nsmooth - 1) {
            deep_refresh();
          }
        }
        if (finest_level_tail) {
          exact_finest_polish();
        }
      } else if (nsmooth > 2) {
        prolongate_with_fresh_diagonals();
        if (nsmooth > 0) {
          if (manual_exact_fine_overlap) {
            for (int n = 0; n < nsmooth; ++n) {
              overlapped_color(coffset_);
              overlapped_color(1 - coffset_);
            }
          } else {
            refresh_boundaries();
            for (int n = 0; n < nsmooth; ++n) {
              pmg->SmoothPack(coffset_);
              refresh_boundaries();
              pmg->SmoothPack(1 - coffset_);
              if (n < nsmooth - 1) {
                refresh_boundaries();
              }
            }
          }
        }
        if (finest_level_tail) {
          exact_finest_polish();
        }
      } else {
        if (manual_exact_fine_overlap) {
          prolongate_with_fresh_diagonals();
          for (int n = 0; n < nsmooth; ++n) {
            overlapped_color(coffset_);
            overlapped_color(1 - coffset_);
          }
        } else {
          SetMGTaskListToFiner(nsmooth, ngh, flag, reduce_same_exchange);
          pdriver->ExecuteTaskList(pmy_mesh_, "mg_to_finer", 0);
        }
        if (finest_level_tail) {
          exact_finest_polish();
        }
      }
    }
    // Whichever branch ran it, the finest_level_tail tail ends on a full refresh of the
    // finest level, so the halo the next V-cycle's descent is about to exchange for is
    // already the one it would get.  Nothing between the two writes u:
    // CalculateDefectNorm forms its residual into def_ and reads u only.
    if (finest_level_tail) {
      finest_bvals_fresh_ = true;
    }
    current_level_++;
  } else if (current_level_ >= nrootlevel_ - 1) { // octets
    if (IsCoarseSolveOwner()) {
      if (current_level_ == nrootlevel_ - 1)
        MGRootBoundary();
      else
        SetBoundariesOctets(true, true);
      ProlongateAndCorrectOctets();
    }
    // On the up-leg current_level_ names the level being prolongated FROM, and the
    // post-smoothing belongs on the level prolongated INTO.  The other two branches get
    // that for free, because ProlongateAndCorrectPack advances the Multigrid object's
    // own level and their smoothers read it from there.  The octets have no such object
    // -- SmoothOctets and SetBoundariesOctets derive the level from current_level_ --
    // so the driver's index has to advance here instead of at the end of the branch.
    current_level_++;
    if (IsCoarseSolveOwner()) {
      for (int n = 0; n < nsmooth; ++n) {
        SetBoundariesOctets(false, false);
        SmoothOctets(coffset_);
        SetBoundariesOctets(false, false);
        SmoothOctets(1 - coffset_);
      }
    }
  } else { // root grid
    if (IsCoarseSolveOwner()) {
      MGRootBoundary();
      mgroot_->ProlongateAndCorrectPack();
      for (int n = 0; n < nsmooth; ++n) {
        MGRootBoundary();
        mgroot_->SmoothPack(coffset_);
        MGRootBoundary();
        mgroot_->SmoothPack(1-coffset_);
      }
    }
    current_level_++;
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_to_finer_time_ += timer.seconds();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::OneStepToCoarser(int nsmooth)
//! \brief smoothing and restriction one level

void MultigridDriver::OneStepToCoarser(Driver *pdriver, int nsmooth) {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  int ngh = mgroot_->ngh_;
  if (current_level_ >= nrootlevel_ + nreflevel_) { // MeshBlocks
    pmg = mglevels_;
    const bool reduce_same_exchange = UseReducedSameExchangeOnCurrentLevel(false);
    // The finest level is the one level UseReducedSameExchangeOnCurrentLevel always
    // refuses, so requiring reduce_same_exchange here asked for two things that cannot
    // hold at once and the skip never fired.  What makes it safe is that the previous
    // V-cycle's tail left this halo exact, which is what finest_bvals_fresh_ records.
    const bool skip_initial_exchange = (current_level_ == ntotallevel_ - 1
                                        && finest_bvals_fresh_);
    skip_coarser_initial_exchange_once_ = skip_initial_exchange;
    finest_bvals_fresh_ = false;
    active_reduce_same_exchange_ = reduce_same_exchange;
    if (global_variable::nranks == 1) {
      auto update_boundaries = [&]() {
        DvceArray5D<Real> u = pmg->GetCurrentData();
        (void)pmg->pbval->RecvAndUnpackMG(u);
        (void)PhysicalBoundary(pdriver, 0);
        (void)FillFCBoundary(pdriver, 0);
      };

      if (!skip_initial_exchange) {
        update_boundaries();
      }
      if (current_level_ < fmglevel_) {
        pmg->StoreOldData();
        pmg->CalculateFASRHSPack();
      }

      for (int n = 0; n < nsmooth; ++n) {
        pmg->SmoothPack(coffset_);
        update_boundaries();
        pmg->SmoothPack(1 - coffset_);
        update_boundaries();
      }
      pmg->RestrictPack();
    } else {
      auto *gpmg = dynamic_cast<MGGravity*>(pmg);
      const bool exact_halo_pair =
          (gpmg != nullptr && UseExactHaloPair(nsmooth, pmg->GetLevelShift()));
      if (exact_halo_pair) {
        // One exchange per red/black pair instead of two.  It has to carry both ghost
        // layers and the diagonal neighbours, because the grown red sweep reads the
        // second layer straight out and reaches the edge cells through the ghost cells
        // it is itself updating.
        auto deep_refresh = [&]() {
          DvceArray5D<Real> u = pmg->GetCurrentData();
          pmg->pbval->SetMGSameHaloDepth(2);
          while (pmg->pbval->InitRecvMG(pmg->nvar_, true) == TaskStatus::incomplete) {}
          while (pmg->pbval->PackAndSendMG(u, true) == TaskStatus::incomplete) {}
          while (pmg->pbval->RecvAndUnpackMG(u, true) == TaskStatus::incomplete) {}
          pmg->pbval->SetMGSameHaloDepth(1);
          // A physical face fills its tangential ghost rows from the halo that has just
          // arrived -- that row is what a grown sweep on the perpendicular face reads --
          // so this cannot be overlapped with the exchange the way the other legs do.
          ApplyPhysicalBoundariesBlocks(true);
        };
        deep_refresh();
        if (current_level_ < fmglevel_) {
          pmg->StoreOldData();
          pmg->CalculateFASRHSPack();
        }
        for (int n = 0; n < nsmooth; ++n) {
          gpmg->SmoothPackExpanded(coffset_, 1);
          // A physical ghost is a function of the interior, so the grown sweep wrote
          // nonsense over it; re-deriving it here is what the skipped exchange's
          // boundary pass would have done, and it is local work.
          ApplyPhysicalBoundariesBlocks(true);
          pmg->SmoothPack(1 - coffset_);
          deep_refresh();
        }
        pmg->RestrictPack();
      } else if (nsmooth > 2) {
        auto refresh_boundaries = [&]() {
          DvceArray5D<Real> u = pmg->GetCurrentData();
          while (pmg->pbval->InitRecvMG(pmg->nvar_) == TaskStatus::incomplete) {}
          while (pmg->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
          // Physical boundaries are independent of same-level MPI halos, so do that
          // work while remote MG traffic is in flight instead of waiting idle on PCIe.
          (void)PhysicalBoundary(pdriver, 0);
          while (pmg->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
          (void)FillFCBoundary(pdriver, 0);
          // Request reuse is guarded inside the next refresh, so leave current
          // halo traffic in flight here instead of synchronizing immediately.
        };
        auto overlapped_color = [&](int color) {
          if (gpmg == nullptr) {
            refresh_boundaries();
            pmg->SmoothPack(color);
            return;
          }
          DvceArray5D<Real> u = pmg->GetCurrentData();
          while (pmg->pbval->InitRecvMG(pmg->nvar_) == TaskStatus::incomplete) {}
          while (pmg->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
          (void)PhysicalBoundary(pdriver, 0);
          gpmg->SmoothPackInterior(color);
          while (pmg->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
          (void)FillFCBoundary(pdriver, 0);
          gpmg->SmoothPackBoundary(color);
        };
        const bool manual_top_coarse_overlap =
            (gpmg != nullptr && !reduce_same_exchange
             && current_level_ == ntotallevel_ - 1);
        if (!skip_initial_exchange) {
          refresh_boundaries();
        }
        if (current_level_ < fmglevel_) {
          pmg->StoreOldData();
          pmg->CalculateFASRHSPack();
        }
        for (int n = 0; n < nsmooth; ++n) {
          if (manual_top_coarse_overlap) {
            overlapped_color(coffset_);
            overlapped_color(1 - coffset_);
          } else {
            pmg->SmoothPack(coffset_);
            refresh_boundaries();
            pmg->SmoothPack(1 - coffset_);
            refresh_boundaries();
          }
        }
        pmg->RestrictPack();
      } else {
        const bool manual_top_coarse_overlap =
            (gpmg != nullptr && !reduce_same_exchange
             && current_level_ == ntotallevel_ - 1);
        if (manual_top_coarse_overlap) {
          auto refresh_boundaries = [&]() {
            DvceArray5D<Real> u = pmg->GetCurrentData();
            while (pmg->pbval->InitRecvMG(pmg->nvar_) == TaskStatus::incomplete) {}
            while (pmg->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
            (void)PhysicalBoundary(pdriver, 0);
            while (pmg->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
            (void)FillFCBoundary(pdriver, 0);
          };
          auto overlapped_color = [&](int color) {
            DvceArray5D<Real> u = pmg->GetCurrentData();
            while (pmg->pbval->InitRecvMG(pmg->nvar_) == TaskStatus::incomplete) {}
            while (pmg->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
            (void)PhysicalBoundary(pdriver, 0);
            gpmg->SmoothPackInterior(color);
            while (pmg->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
            (void)FillFCBoundary(pdriver, 0);
            gpmg->SmoothPackBoundary(color);
          };
          if (!skip_initial_exchange) {
            refresh_boundaries();
          }
          if (current_level_ < fmglevel_) {
            pmg->StoreOldData();
            pmg->CalculateFASRHSPack();
          }
          for (int n = 0; n < nsmooth; ++n) {
            overlapped_color(coffset_);
            overlapped_color(1 - coffset_);
          }
          refresh_boundaries();
          pmg->RestrictPack();
        } else {
          SetMGTaskListToCoarser(nsmooth, ngh, reduce_same_exchange);
          pdriver->ExecuteTaskList(pmy_mesh_, "mg_to_coarser", 0);
        }
      }
    }
    skip_coarser_initial_exchange_once_ = false;
    if (current_level_ == nrootlevel_ + nreflevel_) {
      TransferFromBlocksToRoot(false);
    }
  } else if (current_level_ > nrootlevel_ - 1) { // octets
    if (IsCoarseSolveOwner()) {
      SetBoundariesOctets(false, false);
      if (current_level_ < fmglevel_) {
        StoreOldDataOctets();
        CalculateFASRHSOctets();
      }
      for (int n = 0; n < nsmooth; ++n) {
        SmoothOctets(coffset_);
        SetBoundariesOctets(false, false);
        SmoothOctets(1 - coffset_);
        SetBoundariesOctets(false, false);
      }
      RestrictOctets();
    }
  } else { // root grid
    if (IsCoarseSolveOwner()) {
      MGRootBoundary();
      if (current_level_ < fmglevel_) {
        mgroot_->StoreOldData();
        mgroot_->CalculateFASRHSPack();
      }
      for (int n = 0; n < nsmooth; ++n) {
        mgroot_->SmoothPack(coffset_);
        MGRootBoundary();
        mgroot_->SmoothPack(1-coffset_);
        MGRootBoundary();
      }
      mgroot_->RestrictPack();
    }
  }
  current_level_--;
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_to_coarser_time_ += timer.seconds();
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveVCycle(int npresmooth, int npostsmooth)
//! \brief Solve the V-cycle starting from the current level

void MultigridDriver::SolveVCycle(Driver *pdriver, int npresmooth, int npostsmooth) {
  ++timing_vcycles_;
  int startlevel=current_level_;
  coffset_ ^= 1;
  while (current_level_ > 0) {
    OneStepToCoarser(pdriver, npresmooth);
  }
  SolveCoarsestGrid();
  while (current_level_ < startlevel) {
    OneStepToFiner(pdriver, npostsmooth);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveMG(Driver *pdriver)
//! \brief Multigrid (MG) solve using V-cycles

void MultigridDriver::SolveMG(Driver *pdriver) {
  last_defect_valid_ = false;
  if (eps_ >= 0.0) {
    SolveIterative(pdriver);
  } else {
    SolveIterativeFixedTimes(pdriver);
  }
  return;
}

void MultigridDriver::ResetSolveCounters() {
  timing_vcycles_ = 0;
  timing_coarsest_solves_ = 0;
  timing_defect_norms_ = 0;
  timing_to_coarser_time_ = 0.0;
  timing_to_finer_time_ = 0.0;
  timing_coarsest_time_ = 0.0;
  timing_fmg_coarser_time_ = 0.0;
  timing_fmg_prolongate_time_ = 0.0;
  timing_transfer_blocks_root_time_ = 0.0;
  timing_transfer_root_blocks_time_ = 0.0;
  timing_octets_time_ = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveFMG(Driver *pdriver)
//! \brief Full multigrid (FMG) solve using FAS and V-cycles

void MultigridDriver::SolveFMG(Driver *pdriver) {
  int cycles = std::max(1, fmg_ncycle_);
  SolveFMGCoarser();
  // FMG needs an actual solve on the coarsest root grid before the first
  // prolongation; otherwise the hierarchy is prolonged from an undefined root state.
  SolveCoarsestGrid();
  while (current_level_ < ntotallevel_ - 1) {
    fmglevel_ = current_level_ + 1;
    FMGProlongate(pdriver);
    fmglevel_ = current_level_;
    for (int n = 0; n < cycles; ++n) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    }
  }
  fmglevel_ = ntotallevel_ - 1;
  if (fsubtract_average_) {
    pmg = mglevels_;
    SubtractAverage(MGVariable::u);
  }
  SolveMG(pdriver);
  return;
}


void MultigridDriver::SolveFMGCoarser() {
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  while (current_level_ >= nrootlevel_ + nreflevel_) {
    pmg = mglevels_;
    pmg->RestrictSourcePack();
    if (current_level_ == nrootlevel_ + nreflevel_) {
      TransferFromBlocksToRoot(true);
    }
    current_level_--;
  }
  if (nreflevel_ > 0) {
    if (IsCoarseSolveOwner()) {
      RestrictFMGSourceOctets();
    }
  }
  current_level_ = nrootlevel_ - 1;
  while (current_level_ > 0) {
    if (IsCoarseSolveOwner()) {
      pmg = mgroot_;
      pmg->RestrictSourcePack();
    }
    current_level_--;
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_fmg_coarser_time_ += timer.seconds();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterative(Driver *pdriver)
//! \brief Solve iteratively until defect norm drops below eps_

void MultigridDriver::SolveIterative(Driver *pdriver) {
  last_defect_valid_ = false;
  const int check_stride = std::max(1, defect_check_interval_);
  const int solve_check_stride = (fshowdef_ >= 2) ? 1 : check_stride;
  auto calc_defect_l2 = [&]() {
    Real def_l2 = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def_l2 += CalculateDefectNorm(MGNormType::l2, v);
    }
    return def_l2;
  };

  // Common production setup uses eps_=0 to request a practical convergence
  // floor rather than exact-zero residual. Run a bounded number of V-cycles
  // first (cheap, no per-iteration norm reductions), then fall back to the
  // original stagnation-based loop only if needed.
  if (eps_ == 0.0) {
    int preset_cycles = (niter_ > 0) ? niter_ : std::max(1, fmg_ncycle_ + 1);
    int iteration = 0;
    Real def = 0.0;
    if (fshowdef_ >= 2) {
      def = calc_defect_l2();
      if (global_variable::my_rank == 0) {
        std::cout << "MG initial defect = " << def << std::endl;
      }
    }
    for (int n = 0; n < preset_cycles; ++n) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
      if (fshowdef_ >= 2) {
        def = calc_defect_l2();
        if (global_variable::my_rank == 0) {
          std::cout << "MG iteration " << iteration << ": defect = " << def << std::endl;
        }
      }
      ++iteration;
    }
    if (fshowdef_ < 2) def = calc_defect_l2();
    Real best_def = def;
    // The defect has the units of the source, so the only floor that means the same
    // thing on two problems is a fraction of the source norm.  An absolute number here
    // was tuned once for one set of units and was four orders of magnitude below what a
    // collapse run reaches, so it never fired and the stagnation test below did all the
    // work; on a problem with a larger source it would have fired immediately instead.
    const Real target_defect = AutoTargetDefect();
    int max_extra_cycles = (auto_max_extra_cycles_ >= 0)
                           ? auto_max_extra_cycles_
                           : std::max(6, 3*fmg_ncycle_);
    int stagnation_checks = 0;
    constexpr Real practical_stagnation_ratio = 0.995;
    if (auto_max_extra_cycles_ < 0) {
      // AMR stages generally need more correction work than uniform-grid stages.
      if (nreflevel_ > 0) max_extra_cycles = std::max(max_extra_cycles, 12);
      if (amr_mesh_changed_) max_extra_cycles = std::max(max_extra_cycles, 32);
    }
    int extra_cycles = 0;
    while ((def > target_defect) && (extra_cycles < max_extra_cycles)) {
      int burst = std::min(solve_check_stride, max_extra_cycles - extra_cycles);
      Real olddef = def;
      for (int i = 0; i < burst; ++i) {
        SolveVCycle(pdriver, npresmooth_, npostsmooth_);
        if (fshowdef_ >= 2) {
          def = calc_defect_l2();
          if (global_variable::my_rank == 0) {
            std::cout << "MG iteration " << iteration << ": defect = " << def
                      << std::endl;
          }
        }
        ++iteration;
      }
      if (fshowdef_ < 2) def = calc_defect_l2();
      extra_cycles += burst;
      if ((def/olddef > 0.995) && (def < 1.5*target_defect)) break;
      if (def < best_def * practical_stagnation_ratio) {
        best_def = def;
        stagnation_checks = 0;
      } else {
        ++stagnation_checks;
      }
      if (stagnation_checks >= 2) break;
    }
    // The threshold = 0 policy stops on the extra-cycle cap or on two stagnant checks
    // and hands the potential to the caller either way, so an interface residual floor
    // or a cold V-cycle-only start above the cap used to leave the run using a badly
    // unconverged phi with nothing in the log.  Say so once per 1000 cycles, and only
    // when the miss is an order of magnitude, so an ordinary practical-floor stop stays
    // quiet.
    if ((def > 10.0*target_defect) && (global_variable::my_rank == 0)) {
      const int ncycle = (pmy_mesh_ != nullptr) ? pmy_mesh_->ncycle : 0;
      if (ncycle - unconverged_warn_cycle_ >= 1000 ||
          unconverged_warn_cycle_ == std::numeric_limits<int>::min()) {
        unconverged_warn_cycle_ = ncycle;
        std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
                  << "Solve ended far from the automatic target: defect = " << def
                  << ", target = " << target_defect << " after " << iteration
                  << " V-cycles (cycle " << ncycle
                  << "); the potential is used as is.  Throttled to one report per"
                  << " 1000 cycles." << std::endl;
      }
    }
    last_defect_norm_ = def;
    last_defect_valid_ = true;
    Kokkos::fence();
    return;
  }

  Real def = calc_defect_l2();
  if (fshowdef_ >= 2 && global_variable::my_rank == 0) {
    std::cout << "MG initial defect = " << def << std::endl;
  }

  int checks = 0;
  int stagnation_checks = 0;
  int divergence_checks = 0;
  // AMR topology changes can transiently increase residuals. Keep iterating
  // through short-lived spikes instead of aborting on the first bad ratio.
  const int max_checks = (niter_ > 0) ? niter_ : 160;
  constexpr Real stagnation_ratio = 0.995;
  constexpr Real divergence_ratio = 1.02;

  while (def > eps_ && checks < max_checks) {
    for (int i = 0; i < solve_check_stride; ++i) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    }
    Real olddef = def;
    def = calc_defect_l2();
    if (fshowdef_ >= 2 && global_variable::my_rank == 0) {
      std::cout << "MG iteration " << checks << ": defect = " << def << std::endl;
    }

    if (!std::isfinite(def) || !std::isfinite(olddef)) {
      if (fshowdef_) {
        std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
                  << "Non-finite defect detected; aborting iterative solve."
                  << std::endl;
      }
      break;
    }

    const Real cf = (olddef > 0.0) ? (def/olddef) : 0.0;
    if (cf > divergence_ratio) {
      ++divergence_checks;
    } else {
      divergence_checks = 0;
    }
    if (cf > stagnation_ratio) {
      ++stagnation_checks;
    } else {
      stagnation_checks = 0;
    }

    if (fshowdef_ && (cf > 0.98 || divergence_checks > 0)) {
      std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
                << "Slow convergence: defect ratio = " << cf
                << " (check " << (checks + 1) << "/" << max_checks << ")"
                << std::endl;
    }

    // If nearly converged, allow practical saturation and avoid wasting cycles.
    if (stagnation_checks >= 32 && def < 10.0*eps_) {
      if (fshowdef_) {
        std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
                  << "Stopping at practical convergence floor: defect = "
                  << def << ", target = " << eps_ << std::endl;
      }
      break;
    }

    ++checks;
  }
  if (def > eps_ && checks >= max_checks && fshowdef_) {
    std::cout << "### WARNING in MultigridDriver::SolveIterative" << std::endl
              << "Reached iteration cap before target defect: defect = " << def
              << ", target = " << eps_
              << ", checks = " << max_checks << std::endl;
  }
  last_defect_norm_ = def;
  last_defect_valid_ = true;
  Kokkos::fence();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver)
//! \brief Solve iteratively niter_ times (fixed count)

void MultigridDriver::SolveIterativeFixedTimes(Driver *pdriver) {
  last_defect_valid_ = false;
  auto calc_defect_l2 = [&]() {
    Real def_l2 = 0.0;
    for (int v = 0; v < nvar_; ++v) {
      def_l2 += CalculateDefectNorm(MGNormType::l2, v);
    }
    return def_l2;
  };
  if (fshowdef_ >= 2) {
    Real norm = calc_defect_l2();
    if (global_variable::my_rank == 0) {
      std::cout << "MG initial defect = " << norm << std::endl;
    }
  }
  for (int n = 0; n < niter_; ++n) {
    SolveVCycle(pdriver, npresmooth_, npostsmooth_);
    if (fshowdef_ >= 2) {
      Real norm = calc_defect_l2();
      if (global_variable::my_rank == 0) {
        std::cout << "MG iteration " << n << ": defect = " << norm << std::endl;
      }
    }
  }
  Real def = calc_defect_l2();
  last_defect_norm_ = def;
  last_defect_valid_ = true;
  Kokkos::fence();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SolveCoarsestGrid()
//! \brief Solve the coarsest root grid

void MultigridDriver::SolveCoarsestGrid() {
  ++timing_coarsest_solves_;
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  if (!IsCoarseSolveOwner()) return;
  pmg = mgroot_;
  int ni = (std::max(nrbx1_, std::max(nrbx2_, nrbx3_))
            >> (nrootlevel_-1));
  if (fsubtract_average_ && ni == 1) {
    MGRootBoundary();
    mgroot_->StoreOldData();
    mgroot_->ZeroClearData();
    if (collect_phase_timing_) {
      Kokkos::fence();
      timing_coarsest_time_ += timer.seconds();
    }
    return;
  }
  ni = std::max(ni, coarsest_min_sweeps_);
  if (fsubtract_average_)
    SubtractAverage(MGVariable::src);
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  MGRootBoundary();
  mgroot_->StoreOldData();
  mgroot_->CalculateFASRHSPack();
  for (int i = 0; i < ni; ++i) {
    mgroot_->SmoothPack(coffset_);
    MGRootBoundary();
    mgroot_->SmoothPack(1-coffset_);
    MGRootBoundary();
  }
  if (fsubtract_average_)
    SubtractAverage(MGVariable::u);
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_coarsest_time_ += timer.seconds();
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn Real MultigridDriver::CalculateDefectNorm(MGNormType nrm, int n)
//! \brief calculate the defect norm

Real MultigridDriver::CalculateDefectNorm(MGNormType nrm, int n) {
  ++timing_defect_norms_;
  Real norm = 0.0;
  if (mglevels_ != nullptr) {
    Real mg_norm = mglevels_->CalculateDefectNorm(nrm, n);
    if (nrm == MGNormType::max) {
      norm = std::max(norm, mg_norm);
    } else {
      norm += mg_norm;
    }
  }
  #if MPI_PARALLEL_ENABLED
  Real global_norm = 0.0;
  if (nrm == MGNormType::max) {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  norm = global_norm;
  #endif
  if (nrm != MGNormType::max) {
    Real vol = (mgroot_->size_.x1max - mgroot_->size_.x1min)
             * (mgroot_->size_.x2max - mgroot_->size_.x2min)
             * (mgroot_->size_.x3max - mgroot_->size_.x3min);
    norm /= vol;
  }
  if (nrm == MGNormType::l2) {
    norm = std::sqrt(norm);
  }
  return norm;
}

Real MultigridDriver::CalculateArrayNorm(MGVariable type, MGNormType nrm, int n) {
  Real norm = 0.0;
  if (mglevels_ != nullptr) {
    Real mg_norm = mglevels_->CalculateArrayNorm(type, nrm, n);
    if (nrm == MGNormType::max) {
      norm = std::max(norm, mg_norm);
    } else {
      norm += mg_norm;
    }
  }
#if MPI_PARALLEL_ENABLED
  Real global_norm = 0.0;
  if (nrm == MGNormType::max) {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  } else {
    MPI_Allreduce(&norm, &global_norm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  }
  norm = global_norm;
#endif
  if (nrm != MGNormType::max) {
    Real vol = (mgroot_->size_.x1max - mgroot_->size_.x1min)
             * (mgroot_->size_.x2max - mgroot_->size_.x2min)
             * (mgroot_->size_.x3max - mgroot_->size_.x3min);
    norm /= vol;
  }
  if (nrm == MGNormType::l2) {
    norm = std::sqrt(norm);
  }
  return norm;
}


//========================================================================================
// Octet operations
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SmoothOctets(int color)
//! \brief Apply smoothing on all octets at current level

void MultigridDriver::SmoothOctets(int color) {
  Kokkos::Timer timer;
  int lev = current_level_ - nrootlevel_;
  SmoothOctetLevel(octets_[lev].data(), noctets_[lev], lev + 1, color);
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::StoreOldDataOctets()
//! \brief Store u -> uold for all octets at current level

void MultigridDriver::StoreOldDataOctets() {
  Kokkos::Timer timer;
  int lev = current_level_ - nrootlevel_;
  // u and uold each occupy one contiguous run of the level pool, so the whole level is
  // a single copy rather than one vector assignment per octet.
  const std::size_t stride = octet_pool_stride_[lev];
  if (stride > 0) {
    Real *base = octet_pool_[lev].data();
    std::memcpy(base + 3*stride, base, stride*sizeof(Real));
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateFASRHSOctets()
//! \brief Calculate FAS RHS for all octets at current level

void MultigridDriver::CalculateFASRHSOctets() {
  Kokkos::Timer timer;
  int lev = current_level_ - nrootlevel_;
  CalculateFASRHSOctetLevel(octets_[lev].data(), noctets_[lev], lev + 1);
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ZeroClearOctets()
//! \brief Zero clear u data in all octets up to current level

void MultigridDriver::ZeroClearOctets() {
  int maxlev = current_level_ - 1 - nrootlevel_;
  for (int l = 0; l <= maxlev && l < nreflevel_; ++l) {
    const std::size_t stride = octet_pool_stride_[l];
    if (stride > 0) std::fill_n(octet_pool_[l].data(), stride, 0.0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictFMGSourceOctets()
//! \brief Restrict the source through the octet hierarchy for FMG initialization

void MultigridDriver::RestrictFMGSourceOctets() {
  int ngh = mgroot_->ngh_;
  // Fine octets to coarser octets
  for (int l = nreflevel_ - 1; l >= 1; --l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &foct = octets_[l][o];
      int oid = octet_parent_[l][o];
      if (oid < 0) continue;
      int oi = octet_child_i_[l][o];
      int oj = octet_child_j_[l][o];
      int ok = octet_child_k_[l][o];
      MGOctet &coct = octets_[l-1][oid];
      for (int v = 0; v < nvar_; ++v)
        coct.Src(v, ok, oj, oi) = RestrictOneSrc(foct, v, ngh, ngh, ngh);
    }
  }
  auto root_src_h = GetRootSource_h();
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    int ri = octet_root_i_[o];
    int rj = octet_root_j_[o];
    int rk = octet_root_k_[o];
    for (int v = 0; v < nvar_; ++v)
      root_src_h(0, v, rk, rj, ri) =
        RestrictOneSrc(oct, v, ngh, ngh, ngh);
  }
  root_flat_buf_stale_ = true;
  root_uold_buf_valid_ = false;
  SyncRootSourceToDevice();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictOctets()
//! \brief Compute defect and restrict octets at current level to coarser level

void MultigridDriver::RestrictOctets() {
  Kokkos::Timer timer;
  int lev = current_level_ - nrootlevel_;
  int ngh = mgroot_->ngh_;

  // The defect is now computed for the whole level in one call before the restriction
  // reads it.  An octet with no parent has its defect computed where it used to be
  // skipped, but nothing ever reads that octet's defect, so the restricted values are
  // unchanged.
  CalculateDefectOctetLevel(octets_[lev].data(), noctets_[lev], lev + 1);

  if (lev >= 1) { // fine octets to coarser octets
    for (int o = 0; o < noctets_[lev]; ++o) {
      MGOctet &foct = octets_[lev][o];
      int oid = octet_parent_[lev][o];
      if (oid < 0) continue;
      int oi = octet_child_i_[lev][o];
      int oj = octet_child_j_[lev][o];
      int ok = octet_child_k_[lev][o];
      MGOctet &coct = octets_[lev-1][oid];
      for (int v = 0; v < nvar_; ++v) {
        coct.Src(v, ok, oj, oi) = RestrictOneDef(foct, v, ngh, ngh, ngh);
        coct.U(v, ok, oj, oi) = RestrictOne(foct, v, ngh, ngh, ngh);
      }
    }
  } else { // octets to root grid
    auto root_src_h = GetRootSource_h();
    auto root_u_h = GetRootData_h();

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      int ri = octet_root_i_[o];
      int rj = octet_root_j_[o];
      int rk = octet_root_k_[o];
      for (int v = 0; v < nvar_; ++v) {
        root_src_h(0, v, rk, rj, ri) =
            RestrictOneDef(oct, v, ngh, ngh, ngh);
        root_u_h(0, v, rk, rj, ri) =
            RestrictOne(oct, v, ngh, ngh, ngh);
      }
    }
    root_flat_buf_stale_ = true;
    root_uold_buf_valid_ = false;
    SyncRootSourceToDevice();
    SyncRootDataToDevice();
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ProlongateAndCorrectOctets()
//! \brief Prolongate and correct the potential in octets, interpolating the FAS
//! correction (u - uold) from a 3x3x3 coarse neighborhood.
//!
//! Tomida & Stone (2023) interpolate the correction linearly (their eq. 18, step 8 of
//! Algorithm 3) and reserve the tricubic stencil for the FMG prolongation of the
//! solution (Algorithm 2, step 3), so which one to use is a choice the deck makes
//! through <gravity>/mg_prolongation and Multigrid::ProlongateAndCorrect already reads
//! it on the MeshBlock levels.  Reading it here as well is what makes the hierarchy
//! uniformly one or the other; hardcoding tricubic left the octets on a different
//! interpolant from the levels above and below them.  The two differ by more than a
//! constant near a refinement jump, where the coarse level is a rediscretisation rather
//! than a Galerkin coarsening and the correction is genuinely not smooth: the operator
//! norms are (38/32)^3 = 1.674 against 1, so the tricubic stencil can amplify there by
//! 67%.  FMGProlongateOctets stays tricubic, which is what the paper prescribes for it.

void MultigridDriver::ProlongateAndCorrectOctets() {
  Kokkos::Timer timer;
  int clev = current_level_ - nrootlevel_;
  int flev = clev + 1;
  int ngh = mgroot_->ngh_;
  constexpr Real wcub0[3] = {5.0, 30.0, -3.0};
  constexpr Real wcub1[3] = {-3.0, 30.0, 5.0};
  constexpr Real wlin0[3] = {1.0, 3.0, 0.0};
  constexpr Real wlin1[3] = {0.0, 3.0, 1.0};
  const bool tricubic = (fprolongation_ == 1);
  const Real *w0 = tricubic ? wcub0 : wlin0;
  const Real *w1 = tricubic ? wcub1 : wlin1;
  const Real inv = tricubic ? 1.0 / 32768.0 : 1.0 / 64.0;

  if (flev == 0) { // from root to octets
    SyncRootDataToHost(true);
    auto root_u_h = GetRootData_h();
    auto root_uold_h = GetRootOldData_h();

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      int ri = octet_root_i_[o];
      int rj = octet_root_j_[o];
      int rk = octet_root_k_[o];

      Real cbuf[3][3][3];
      for (int v = 0; v < nvar_; ++v) {
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              cbuf[kk+1][jj+1][ii+1] = root_u_h(0, v, rk+kk, rj+jj, ri+ii)
                                      - root_uold_h(0, v, rk+kk, rj+jj, ri+ii);
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                  for (int c = 0; c < 3; ++c)
                    sum += wk[a]*wj[b]*wi[c] * cbuf[a][b][c];
              oct.U(v, ngh+dk, ngh+dj, ngh+di) += sum * inv;
            }
          }
        }
      }
    }
  } else { // from coarser octets to finer octets
    for (int o = 0; o < noctets_[flev]; ++o) {
      MGOctet &foct = octets_[flev][o];
      int cid = octet_parent_[flev][o];
      if (cid < 0) continue;
      MGOctet &coct = octets_[clev][cid];
      int ci = octet_child_i_[flev][o];
      int cj = octet_child_j_[flev][o];
      int ck = octet_child_k_[flev][o];

      Real cbuf[3][3][3];
      for (int v = 0; v < nvar_; ++v) {
        for (int kk = -1; kk <= 1; ++kk)
          for (int jj = -1; jj <= 1; ++jj)
            for (int ii = -1; ii <= 1; ++ii)
              cbuf[kk+1][jj+1][ii+1] = coct.U(v, ck+kk, cj+jj, ci+ii)
                                      - coct.Uold(v, ck+kk, cj+jj, ci+ii);
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                  for (int c = 0; c < 3; ++c)
                    sum += wk[a]*wj[b]*wi[c] * cbuf[a][b][c];
              foct.U(v, ngh+dk, ngh+dj, ngh+di) += sum * inv;
            }
          }
        }
      }
    }
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::FMGProlongateOctets()
//! \brief FMG prolongation for octets (tricubic interpolation from coarser level)

void MultigridDriver::FMGProlongateOctets() {
  int clev = current_level_ - nrootlevel_;
  int flev = clev + 1;
  int ngh = mgroot_->ngh_;
  constexpr Real w0[3] = {5.0, 30.0, -3.0};
  constexpr Real w1[3] = {-3.0, 30.0, 5.0};
  constexpr Real inv = 1.0 / 32768.0;

  if (flev == 0) { // from root to octets
    SyncRootDataToHost(false);
    auto root_u_h = GetRootData_h();

    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      int ri = octet_root_i_[o];
      int rj = octet_root_j_[o];
      int rk = octet_root_k_[o];
      for (int v = 0; v < nvar_; ++v) {
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int kk = -1; kk <= 1; ++kk)
                for (int jj = -1; jj <= 1; ++jj)
                  for (int ii = -1; ii <= 1; ++ii)
                    sum += wk[kk+1]*wj[jj+1]*wi[ii+1]
                           * root_u_h(0, v, rk+kk, rj+jj, ri+ii);
              oct.U(v, ngh+dk, ngh+dj, ngh+di) = sum * inv;
            }
          }
        }
      }
    }
  } else { // from coarser octets to finer octets
    for (int o = 0; o < noctets_[flev]; ++o) {
      MGOctet &foct = octets_[flev][o];
      int cid = octet_parent_[flev][o];
      if (cid < 0) continue;
      MGOctet &coct = octets_[clev][cid];
      int ci = octet_child_i_[flev][o];
      int cj = octet_child_j_[flev][o];
      int ck = octet_child_k_[flev][o];
      for (int v = 0; v < nvar_; ++v) {
        for (int dk = 0; dk <= 1; ++dk) {
          const Real *wk = (dk == 0) ? w0 : w1;
          for (int dj = 0; dj <= 1; ++dj) {
            const Real *wj = (dj == 0) ? w0 : w1;
            for (int di = 0; di <= 1; ++di) {
              const Real *wi = (di == 0) ? w0 : w1;
              Real sum = 0.0;
              for (int kk = -1; kk <= 1; ++kk)
                for (int jj = -1; jj <= 1; ++jj)
                  for (int ii = -1; ii <= 1; ++ii)
                    sum += wk[kk+1]*wj[jj+1]*wi[ii+1]
                           * coct.U(v, ck+kk, cj+jj, ci+ii);
              foct.U(v, ngh+dk, ngh+dj, ngh+di) = sum * inv;
            }
          }
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetBoundariesOctets(bool fprolong, bool folddata)
//! \brief Apply boundary conditions for octets at current level
//! fprolong=true: skip leaf octets (used before prolongation)
//! folddata=true: also fill uold ghost cells

void MultigridDriver::SetBoundariesOctets(bool fprolong, bool folddata) {
  Kokkos::Timer timer;
  int lev = current_level_ - nrootlevel_;
  if (!fprolong && lev == 0) {
    BuildRootFlatBuffers(folddata);
  }
  ComputeOctetInteriorAverages(lev, folddata);
  auto process_octet = [&](int o, std::vector<Real> &cbuf_local,
                           std::vector<Real> &cbufold_local,
                           std::array<bool,27> &ncoarse_local) {
    MGOctet &oct = octets_[lev][o];
    if (fprolong && oct.fleaf) return;

    std::fill(ncoarse_local.begin(), ncoarse_local.end(), false);
    std::fill(cbuf_local.begin(), cbuf_local.end(), 0.0);
    if (folddata) std::fill(cbufold_local.begin(), cbufold_local.end(), 0.0);

    const LogicalLocation &loc = oct.loc;
    const int base = o * 27;
    const auto &same_lut = octet_nb_same_[lev];
    const auto &coarse_lut = octet_nb_coarse_[lev];
    const auto &rootloc_lut = octet_nb_rootloc_[lev];

    // Each octet face/edge/corner uses one of:
    // 1) same-level neighbor copy,
    // 2) prolongation from coarser level/root buffer,
    // 3) physical boundary fill for domain faces.
    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          const int nidx = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
          int nid = same_lut[base + nidx];
          const LogicalLocation &nloc = rootloc_lut[base + nidx];
          const bool valid_neighbor = (nloc.level >= 0);
          if (!valid_neighbor) continue;
          if (nid >= 0) {
            MGOctet &noct = octets_[lev][nid];
            SetOctetBoundarySameLevel(oct, noct, nid, cbuf_local, cbufold_local,
                                      nvar_, ox1, ox2, ox3, folddata);
          } else if (!fprolong) {
            if (lev > 0) {
              int cid = coarse_lut[base + nidx];
              if (cid >= 0) {
                ncoarse_local[nidx] = true;
                MGOctet &coct = octets_[lev-1][cid];
                SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_local, cbufold_local,
                                            nvar_, coct.nc, coct.nc, coct.nc, loc,
                                            ox1, ox2, ox3, folddata);
              }
            } else {
              ncoarse_local[nidx] = true;
              SetOctetBoundaryFromCoarser(root_u_buf_.data(), root_uold_buf_.data(),
                                          cbuf_local, cbufold_local, nvar_,
                                          root_buf_nx_, root_buf_ny_, root_buf_nz_,
                                          nloc, ox1, ox2, ox3, folddata);
            }
          }
        }
      }
    }

    if (!fprolong) {
      SetOctetCoarseBufferPhysicalBoundaries(oct, o, cbuf_local, cbufold_local, folddata);
      if (!folddata) {
        ProlongateOctetBoundariesFluxCons(oct, cbuf_local, nvar_, ncoarse_local);
      } else {
        ProlongateOctetBoundaries(oct, o, cbuf_local, cbufold_local, nvar_,
                                  ncoarse_local, folddata);
      }
    }
    ApplyPhysicalBoundariesOctet(oct, folddata);
  };

  for (int o = 0; o < noctets_[lev]; ++o) {
    process_octet(o, cbuf_, cbufold_, ncoarse_);
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ApplyPhysicalBoundariesOctet(MGOctet &oct, bool folddata)
//! \brief Apply physical MG boundary conditions to one octet
//!
//! A periodic mesh face is not a physical boundary here.  InitializeOctets wraps the
//! neighbour search across such a face, so the octet has a real neighbour there and
//! SetBoundariesOctets has already filled those ghosts from it -- by copy when the
//! neighbour is at the same level, by prolongation when it is coarser.  Applying a
//! periodic condition on top of that would overwrite the neighbour's column with this
//! octet's own opposite column, which is the same value only in the degenerate case of
//! a single octet spanning the domain in that direction.  This routine runs after the
//! neighbour fills, so it was the overwrite that survived.
//!
//! Each sweep covers the whole tangential extent of the octet, its ghosts included, and
//! the sweeps run x1, then x2, then x3, so that a later one composes its own rule on top
//! of what an earlier one left in the edge and corner ghosts.  That is what
//! ApplyPhysicalBoundariesBlocks already does under fill_tangential_ghosts, and the order
//! is what makes it right, so it has to stay.  Filling only the interior tangential range
//! left the ghosts that are diagonal to a domain face at zero, because the diagonal
//! neighbour they would otherwise be copied from is outside the domain: for an octet on
//! one face, five of the 27 cells of a child's tricubic window, carrying a signed weight
//! of 1475/32768, or 4.5% of the stencil.  That zero went into both the FAS correction
//! and the FMG initial guess and put a floor of a few per cent on the local contraction
//! rate of every octet touching the domain edge.  It moves the rate, not the fixed point.
//!
//! As in SetOctetCoarseBufferPhysicalBoundaries, evaluating the multipole expansion on a
//! tangential ghost is legitimate: the expansion is an exterior analytic field, so it is
//! defined a cell outside the domain, and leaving the cell at zero is the defect.

void MultigridDriver::ApplyPhysicalBoundariesOctet(MGOctet &oct, bool folddata) {
  const int ngh = mgroot_->ngh_;
  const int l = ngh;
  const int r = ngh + 1;
  const int nc = oct.nc;
  const int lev = oct.loc.level - locrootlevel_;
  const int lx1_max = (nrbx1_ << lev) - 1;
  const int lx2_max = (nrbx2_ << lev) - 1;
  const int lx3_max = (nrbx3_ << lev) - 1;
  // Every branch below is gated on the octet abutting a domain face, so an octet in
  // the interior writes nothing.  Most octets of a refined level are interior ones,
  // and this is called for each of them several times per V-cycle.
  if (oct.loc.lx1 != 0 && oct.loc.lx1 != lx1_max &&
      oct.loc.lx2 != 0 && oct.loc.lx2 != lx2_max &&
      oct.loc.lx3 != 0 && oct.loc.lx3 != lx3_max) return;
  const bool multipole_ready = (mporder_ > 0 && nmpcoeff_ > 0);
  const Real oct_dx1 = (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min)
                     / static_cast<Real>((nrbx1_ << lev)*2);
  const Real oct_dx2 = (pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min)
                     / static_cast<Real>((nrbx2_ << lev)*2);
  const Real oct_dx3 = (pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min)
                     / static_cast<Real>((nrbx3_ << lev)*2);
  const Real oct_x1min = pmy_mesh_->mesh_size.x1min
      + static_cast<Real>(oct.loc.lx1)*2.0*oct_dx1;
  const Real oct_x2min = pmy_mesh_->mesh_size.x2min
      + static_cast<Real>(oct.loc.lx2)*2.0*oct_dx2;
  const Real oct_x3min = pmy_mesh_->mesh_size.x3min
      + static_cast<Real>(oct.loc.lx3)*2.0*oct_dx3;

  // Apply to current solution first; optionally mirror onto uold for FAS paths.
  auto apply_one = [&](bool use_old) {
    // x1 boundaries
    if (oct.loc.lx1 == 0) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::inner_x1];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, k, j, l + n)
                                     : -oct.U(v, k, j, l + n);
                  if (use_old) oct.Uold(v, k, j, l - 1 - n) = val;
                  else
                    oct.U(v, k, j, l - 1 - n) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real y = oct_x2min + (j - l + 0.5)*oct_dx2 - mpo_[1];
                  const Real z = oct_x3min + (k - l + 0.5)*oct_dx3 - mpo_[2];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      pmy_mesh_->mesh_size.x1min - mpo_[0], y, z, mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, k, j, l + n)
                                     : 2.0*phi - oct.U(v, k, j, l + n);
                  if (use_old) oct.Uold(v, k, j, l - 1 - n) = val;
                  else
                    oct.U(v, k, j, l - 1 - n) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, k, j, l)
                                     : oct.U(v, k, j, l);
                  if (use_old) oct.Uold(v, k, j, l - 1 - n) = val;
                  else
                    oct.U(v, k, j, l - 1 - n) = val;
                }
              }
            }
          }
        }
      }
    }
    if (oct.loc.lx1 == lx1_max) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::outer_x1];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, k, j, r - n)
                                     : -oct.U(v, k, j, r - n);
                  if (use_old) oct.Uold(v, k, j, r + 1 + n) = val;
                  else
                    oct.U(v, k, j, r + 1 + n) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real y = oct_x2min + (j - l + 0.5)*oct_dx2 - mpo_[1];
                  const Real z = oct_x3min + (k - l + 0.5)*oct_dx3 - mpo_[2];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      pmy_mesh_->mesh_size.x1max - mpo_[0], y, z, mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, k, j, r - n)
                                     : 2.0*phi - oct.U(v, k, j, r - n);
                  if (use_old) oct.Uold(v, k, j, r + 1 + n) = val;
                  else
                    oct.U(v, k, j, r + 1 + n) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, k, j, r)
                                     : oct.U(v, k, j, r);
                  if (use_old) oct.Uold(v, k, j, r + 1 + n) = val;
                  else
                    oct.U(v, k, j, r + 1 + n) = val;
                }
              }
            }
          }
        }
      }
    }

    // x2 boundaries
    if (oct.loc.lx2 == 0) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::inner_x2];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int i = 0; i < nc; ++i) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, k, l + n, i)
                                     : -oct.U(v, k, l + n, i);
                  if (use_old) oct.Uold(v, k, l - 1 - n, i) = val;
                  else
                    oct.U(v, k, l - 1 - n, i) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real x = oct_x1min + (i - l + 0.5)*oct_dx1 - mpo_[0];
                  const Real z = oct_x3min + (k - l + 0.5)*oct_dx3 - mpo_[2];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      x, pmy_mesh_->mesh_size.x2min - mpo_[1], z,
                      mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, k, l + n, i)
                                     : 2.0*phi - oct.U(v, k, l + n, i);
                  if (use_old) oct.Uold(v, k, l - 1 - n, i) = val;
                  else
                    oct.U(v, k, l - 1 - n, i) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, k, l, i)
                                     : oct.U(v, k, l, i);
                  if (use_old) oct.Uold(v, k, l - 1 - n, i) = val;
                  else
                    oct.U(v, k, l - 1 - n, i) = val;
                }
              }
            }
          }
        }
      }
    }
    if (oct.loc.lx2 == lx2_max) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::outer_x2];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int i = 0; i < nc; ++i) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, k, r - n, i)
                                     : -oct.U(v, k, r - n, i);
                  if (use_old) oct.Uold(v, k, r + 1 + n, i) = val;
                  else
                    oct.U(v, k, r + 1 + n, i) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real x = oct_x1min + (i - l + 0.5)*oct_dx1 - mpo_[0];
                  const Real z = oct_x3min + (k - l + 0.5)*oct_dx3 - mpo_[2];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      x, pmy_mesh_->mesh_size.x2max - mpo_[1], z,
                      mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, k, r - n, i)
                                     : 2.0*phi - oct.U(v, k, r - n, i);
                  if (use_old) oct.Uold(v, k, r + 1 + n, i) = val;
                  else
                    oct.U(v, k, r + 1 + n, i) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, k, r, i)
                                     : oct.U(v, k, r, i);
                  if (use_old) oct.Uold(v, k, r + 1 + n, i) = val;
                  else
                    oct.U(v, k, r + 1 + n, i) = val;
                }
              }
            }
          }
        }
      }
    }

    // x3 boundaries
    if (oct.loc.lx3 == 0) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::inner_x3];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int j = 0; j < nc; ++j) {
            for (int i = 0; i < nc; ++i) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, l + n, j, i)
                                     : -oct.U(v, l + n, j, i);
                  if (use_old) oct.Uold(v, l - 1 - n, j, i) = val;
                  else
                    oct.U(v, l - 1 - n, j, i) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real x = oct_x1min + (i - l + 0.5)*oct_dx1 - mpo_[0];
                  const Real y = oct_x2min + (j - l + 0.5)*oct_dx2 - mpo_[1];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      x, y, pmy_mesh_->mesh_size.x3min - mpo_[2],
                      mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, l + n, j, i)
                                     : 2.0*phi - oct.U(v, l + n, j, i);
                  if (use_old) oct.Uold(v, l - 1 - n, j, i) = val;
                  else
                    oct.U(v, l - 1 - n, j, i) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, l, j, i)
                                     : oct.U(v, l, j, i);
                  if (use_old) oct.Uold(v, l - 1 - n, j, i) = val;
                  else
                    oct.U(v, l - 1 - n, j, i) = val;
                }
              }
            }
          }
        }
      }
    }
    if (oct.loc.lx3 == lx3_max) {
      BoundaryFlag bc = mg_mesh_bcs_[BoundaryFace::outer_x3];
      if (IsZeroGradMGBoundary(bc) || IsZeroFixedMGBoundary(bc) ||
          bc == BoundaryFlag::mg_multipole) {
        for (int v = 0; v < nvar_; ++v) {
          for (int j = 0; j < nc; ++j) {
            for (int i = 0; i < nc; ++i) {
              for (int n = 0; n < ngh; ++n) {
                if (IsZeroFixedMGBoundary(bc)) {
                  Real val = use_old ? -oct.Uold(v, r - n, j, i)
                                     : -oct.U(v, r - n, j, i);
                  if (use_old) oct.Uold(v, r + 1 + n, j, i) = val;
                  else
                    oct.U(v, r + 1 + n, j, i) = val;
                } else if (bc == BoundaryFlag::mg_multipole) {
                  const Real x = oct_x1min + (i - l + 0.5)*oct_dx1 - mpo_[0];
                  const Real y = oct_x2min + (j - l + 0.5)*oct_dx2 - mpo_[1];
                  const Real phi = multipole_ready ? EvalMultipolePhi(
                      x, y, pmy_mesh_->mesh_size.x3max - mpo_[2],
                      mpcoeff_, mporder_) : 0.0;
                  Real val = use_old ? 2.0*phi - oct.Uold(v, r - n, j, i)
                                     : 2.0*phi - oct.U(v, r - n, j, i);
                  if (use_old) oct.Uold(v, r + 1 + n, j, i) = val;
                  else
                    oct.U(v, r + 1 + n, j, i) = val;
                } else {
                  Real val = use_old ? oct.Uold(v, r, j, i)
                                     : oct.U(v, r, j, i);
                  if (use_old) oct.Uold(v, r + 1 + n, j, i) = val;
                  else
                    oct.U(v, r + 1 + n, j, i) = val;
                }
              }
            }
          }
        }
      }
    }
  };

  apply_one(false);
  if (folddata) apply_one(true);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetCoarseBufferPhysicalBoundaries(...)
//! \brief Apply the physical boundary conditions to an octet's 3x3x3 coarse buffer
//!
//! The buffer holds the octet's own parent-cell value at its centre and the parent-cell
//! value of each of the 26 neighbours around it.  A direction that leaves the domain has
//! no neighbour to copy from, so SetBoundariesOctets skips it and the slot keeps the
//! zero it was cleared to -- and then the transverse gradients that
//! ProlongateOctetBoundaries and ProlongateOctetBoundariesFluxCons build out of the
//! buffer read that zero as if it were a field value.  An octet that has a coarser
//! neighbour on one face and abuts a non-periodic domain face on another therefore gets
//! a first-order error in the ghost cells it prolongates, which weakens the coarse-grid
//! correction there.
//!
//! Fill those slots from the boundary condition instead, with the same rules the octet's
//! own ghost cells get from ApplyPhysicalBoundariesOctet: copy for a zero gradient,
//! reflect for a fixed zero, 2*phi - u for the multipole potential.  Sweeping x1 then x2
//! then x3, as MGRootBoundary does, composes those rules correctly on the edge and
//! corner slots, because each sweep overwrites what the earlier ones left there using a
//! partner the earlier sweep has already filled.  Unlike the cell-centred routines this
//! one does evaluate the multipole potential on edges and corners: the expansion is an
//! exterior analytic field, so it is defined a cell outside the domain, and leaving the
//! slot at zero is precisely the defect being fixed.  Periodic faces never reach here --
//! InitializeOctets wraps them, so they have a real neighbour.

void MultigridDriver::SetOctetCoarseBufferPhysicalBoundaries(const MGOctet &oct,
     int oct_id, std::vector<Real> &cbuf, std::vector<Real> &cbufold, bool folddata) {
  const int lev = oct.loc.level - locrootlevel_;
  const int nx1 = nrbx1_ << lev;
  const int nx2 = nrbx2_ << lev;
  const int nx3 = nrbx3_ << lev;
  const int lx1 = static_cast<int>(oct.loc.lx1);
  const int lx2 = static_cast<int>(oct.loc.lx2);
  const int lx3 = static_cast<int>(oct.loc.lx3);
  const BoundaryFlag bc_ix1 = mg_mesh_bcs_[BoundaryFace::inner_x1];
  const BoundaryFlag bc_ox1 = mg_mesh_bcs_[BoundaryFace::outer_x1];
  const BoundaryFlag bc_ix2 = mg_mesh_bcs_[BoundaryFace::inner_x2];
  const BoundaryFlag bc_ox2 = mg_mesh_bcs_[BoundaryFace::outer_x2];
  const BoundaryFlag bc_ix3 = mg_mesh_bcs_[BoundaryFace::inner_x3];
  const BoundaryFlag bc_ox3 = mg_mesh_bcs_[BoundaryFace::outer_x3];
  // The same test InitializeOctets used to decide that the neighbour is not there.
  const bool lo1 = (lx1 == 0) && !IsPeriodicMGBoundary(bc_ix1);
  const bool hi1 = (lx1 == nx1 - 1) && !IsPeriodicMGBoundary(bc_ox1);
  const bool lo2 = (lx2 == 0) && !IsPeriodicMGBoundary(bc_ix2);
  const bool hi2 = (lx2 == nx2 - 1) && !IsPeriodicMGBoundary(bc_ox2);
  const bool lo3 = (lx3 == 0) && !IsPeriodicMGBoundary(bc_ix3);
  const bool hi3 = (lx3 == nx3 - 1) && !IsPeriodicMGBoundary(bc_ox3);
  if (!(lo1 || hi1 || lo2 || hi2 || lo3 || hi3)) return;

  const Real x1min = pmy_mesh_->mesh_size.x1min, x1max = pmy_mesh_->mesh_size.x1max;
  const Real x2min = pmy_mesh_->mesh_size.x2min, x2max = pmy_mesh_->mesh_size.x2max;
  const Real x3min = pmy_mesh_->mesh_size.x3min, x3max = pmy_mesh_->mesh_size.x3max;
  const Real dx1 = (x1max - x1min)/static_cast<Real>(nx1);
  const Real dx2 = (x2max - x2min)/static_cast<Real>(nx2);
  const Real dx3 = (x3max - x3min)/static_cast<Real>(nx3);
  // Cell centres of the three buffer slots along each axis, at this octet's own level.
  Real cx1[3], cx2[3], cx3[3];
  for (int s = 0; s < 3; ++s) {
    cx1[s] = x1min + (static_cast<Real>(lx1 + s - 1) + 0.5)*dx1;
    cx2[s] = x2min + (static_cast<Real>(lx2 + s - 1) + 0.5)*dx2;
    cx3[s] = x3min + (static_cast<Real>(lx3 + s - 1) + 0.5)*dx3;
  }
  const bool multipole_ready = (mporder_ > 0 && nmpcoeff_ > 0);

  auto reflect = [&](BoundaryFlag bc, Real inner, Real fx, Real fy, Real fz) -> Real {
    if (IsZeroFixedMGBoundary(bc)) return -inner;
    if (bc == BoundaryFlag::mg_multipole) {
      const Real phi = multipole_ready
          ? EvalMultipolePhi(fx - mpo_[0], fy - mpo_[1], fz - mpo_[2], mpcoeff_, mporder_)
          : 0.0;
      return 2.0*phi - inner;
    }
    return inner;  // zero gradient, and the only flag left
  };

  auto fill = [&](std::vector<Real> &buf, bool use_old) {
    for (int v = 0; v < nvar_; ++v) {
      // The centre slot is this octet's own volume average.  The tricubic prolongation
      // forms it for itself a moment later, but the reflections below need it now.
      buf[((v*3 + 1)*3 + 1)*3 + 1] = use_old ? octet_avgold_[oct_id*nvar_ + v]
                                             : octet_avg_[oct_id*nvar_ + v];
      if (lo1) {
        for (int k = 0; k < 3; ++k)
          for (int j = 0; j < 3; ++j)
            BufRef(buf,3,v,k,j,0) =
                reflect(bc_ix1, BufRef(buf,3,v,k,j,1), x1min, cx2[j], cx3[k]);
      }
      if (hi1) {
        for (int k = 0; k < 3; ++k)
          for (int j = 0; j < 3; ++j)
            BufRef(buf,3,v,k,j,2) =
                reflect(bc_ox1, BufRef(buf,3,v,k,j,1), x1max, cx2[j], cx3[k]);
      }
      if (lo2) {
        for (int k = 0; k < 3; ++k)
          for (int i = 0; i < 3; ++i)
            BufRef(buf,3,v,k,0,i) =
                reflect(bc_ix2, BufRef(buf,3,v,k,1,i), cx1[i], x2min, cx3[k]);
      }
      if (hi2) {
        for (int k = 0; k < 3; ++k)
          for (int i = 0; i < 3; ++i)
            BufRef(buf,3,v,k,2,i) =
                reflect(bc_ox2, BufRef(buf,3,v,k,1,i), cx1[i], x2max, cx3[k]);
      }
      if (lo3) {
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i)
            BufRef(buf,3,v,0,j,i) =
                reflect(bc_ix3, BufRef(buf,3,v,1,j,i), cx1[i], cx2[j], x3min);
      }
      if (hi3) {
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i)
            BufRef(buf,3,v,2,j,i) =
                reflect(bc_ox3, BufRef(buf,3,v,1,j,i), cx1[i], cx2[j], x3max);
      }
    }
  };

  fill(cbuf, false);
  if (folddata) fill(cbufold, true);
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ApplyPhysicalBoundariesBlocks(bool fill_tangential_ghosts)
//! \brief Apply physical MG boundary conditions on meshblock-level MG arrays

void MultigridDriver::ApplyPhysicalBoundariesBlocks(bool fill_tangential_ghosts,
                                                   bool homogeneous) {
  if (pmg == nullptr) return;
  auto u = pmg->GetCurrentData();
  int nvar = u.extent_int(1);
  int ngh = pmg->GetGhostCells();
  int shift = pmg->GetLevelShift();
  int ncells = pmg->GetSize() >> shift;
  if (ncells < 1) return;
  const int nx1_cells = std::max(1, pmg->indcs_.nx1 >> shift);
  const int nx2_cells = std::max(1, pmg->indcs_.nx2 >> shift);
  const int nx3_cells = std::max(1, pmg->indcs_.nx3 >> shift);
  int is = ngh, ie = is + ncells - 1;
  int js = ngh, je = js + ncells - 1;
  int ks = ngh, ke = ks + ncells - 1;
  int nx = ncells + 2 * ngh;

  // A periodic mesh face has no physical boundary condition to apply: those ghosts are
  // owned by the wrapped neighbour exchange, and every thread of the three kernels
  // below would fall straight through.  A fully periodic mesh -- the Jeans, collapse and
  // tidal-disruption setups -- therefore launches nothing here, and these kernels run
  // once per halo refresh, which is tens of times per V-cycle.
  {
    bool any_physical = false;
    for (int f = 0; f < 6; ++f) {
      if (!IsPeriodicMGBoundary(mg_mesh_bcs_[f])) { any_physical = true; break; }
    }
    if (!any_physical) return;
  }

  const BoundaryFlag bc_ix1 = mg_mesh_bcs_[BoundaryFace::inner_x1];
  const BoundaryFlag bc_ox1 = mg_mesh_bcs_[BoundaryFace::outer_x1];
  const BoundaryFlag bc_ix2 = mg_mesh_bcs_[BoundaryFace::inner_x2];
  const BoundaryFlag bc_ox2 = mg_mesh_bcs_[BoundaryFace::outer_x2];
  const BoundaryFlag bc_ix3 = mg_mesh_bcs_[BoundaryFace::inner_x3];
  const BoundaryFlag bc_ox3 = mg_mesh_bcs_[BoundaryFace::outer_x3];
  // A correction carries the HOMOGENEOUS form of the boundary condition; see the
  // caller in PhysicalBoundaryForCorrectionProlongation.
  const bool mp_homogeneous = homogeneous;

  const MGBoundaryPhiCache &cache =
      EnsureBoundaryPhiCache(shift, ngh, ncells, nx1_cells, nx2_cells, nx3_cells);
  if (cache.nface == 0) return;
  auto face_m = cache.face_m;
  auto face_f = cache.face_f;
  auto phi = cache.phi;
  const bool has_phi = cache.has_phi;

  const int x1_k0 = fill_tangential_ghosts ? 0 : ks;
  const int x1_k1 = fill_tangential_ghosts ? nx : ke + 1;
  const int x1_j0 = fill_tangential_ghosts ? 0 : js;
  const int x1_j1 = fill_tangential_ghosts ? nx : je + 1;
  const int x2_k0 = fill_tangential_ghosts ? 0 : ks;
  const int x2_k1 = fill_tangential_ghosts ? nx : ke + 1;
  const bool fill_tangential = fill_tangential_ghosts;

  // Apply BCs dimension-by-dimension so corner ghosts naturally inherit from
  // already-updated lower-dimensional passes. Periodic faces are owned by wrapped
  // neighbor communication, even when another dimension makes the mesh nonperiodic.
  //
  // Each launch covers one contiguous run of the face list, so it carries no thread for a
  // block that does not touch that pair of domain faces -- on a deeply refined mesh that
  // is nearly every block.  The two faces of a direction are separate entries of the same
  // launch; they write disjoint ghost columns and read disjoint interior ones, which is
  // what let them share a thread before.  par_for flattens the index space with the
  // contiguous axis innermost, so a warp walks one row of a face rather than one cell of
  // each of 32 different rows.
  if (cache.off[2] > cache.off[0]) {
    par_for("MGBlockPhysBnd_x1", DevExeSpace(), cache.off[0], cache.off[2]-1,
            0, nvar-1, x1_k0, x1_k1-1, x1_j0, x1_j1-1,
    KOKKOS_LAMBDA(const int e, const int v, const int k, const int j) {
      const int m = face_m(e);
      const bool inner = (face_f(e) == BoundaryFace::inner_x1);
      const BoundaryFlag bc = inner ? bc_ix1 : bc_ox1;
      if (bc == BoundaryFlag::mg_multipole) {
        if (!has_phi) return;
        if (!(fill_tangential || (k >= ks && k <= ke && j >= js && j <= je))) return;
        const Real p = mp_homogeneous ? 0.0 : phi(e, k*nx + j);
        for (int n = 0; n < ngh; ++n) {
          if (inner) u(m, v, k, j, is - 1 - n) = 2.0*p - u(m, v, k, j, is + n);
          else       u(m, v, k, j, ie + 1 + n) = 2.0*p - u(m, v, k, j, ie - n);
        }
      } else {
        const bool fixed = IsZeroFixedMGBoundary(bc);
        for (int n = 0; n < ngh; ++n) {
          if (inner) {
            u(m, v, k, j, is - 1 - n) =
                fixed ? -u(m, v, k, j, is + n) : u(m, v, k, j, is);
          } else {
            u(m, v, k, j, ie + 1 + n) =
                fixed ? -u(m, v, k, j, ie - n) : u(m, v, k, j, ie);
          }
        }
      }
    });
  }

  if (cache.off[4] > cache.off[2]) {
    par_for("MGBlockPhysBnd_x2", DevExeSpace(), cache.off[2], cache.off[4]-1,
            0, nvar-1, x2_k0, x2_k1-1, 0, nx-1,
    KOKKOS_LAMBDA(const int e, const int v, const int k, const int i) {
      const int m = face_m(e);
      const bool inner = (face_f(e) == BoundaryFace::inner_x2);
      const BoundaryFlag bc = inner ? bc_ix2 : bc_ox2;
      if (bc == BoundaryFlag::mg_multipole) {
        if (!has_phi) return;
        if (!(fill_tangential || (k >= ks && k <= ke && i >= is && i <= ie))) return;
        const Real p = mp_homogeneous ? 0.0 : phi(e, k*nx + i);
        for (int n = 0; n < ngh; ++n) {
          if (inner) u(m, v, k, js - 1 - n, i) = 2.0*p - u(m, v, k, js + n, i);
          else       u(m, v, k, je + 1 + n, i) = 2.0*p - u(m, v, k, je - n, i);
        }
      } else {
        const bool fixed = IsZeroFixedMGBoundary(bc);
        for (int n = 0; n < ngh; ++n) {
          if (inner) {
            u(m, v, k, js - 1 - n, i) =
                fixed ? -u(m, v, k, js + n, i) : u(m, v, k, js, i);
          } else {
            u(m, v, k, je + 1 + n, i) =
                fixed ? -u(m, v, k, je - n, i) : u(m, v, k, je, i);
          }
        }
      }
    });
  }

  if (cache.off[6] > cache.off[4]) {
    par_for("MGBlockPhysBnd_x3", DevExeSpace(), cache.off[4], cache.off[6]-1,
            0, nvar-1, 0, nx-1, 0, nx-1,
    KOKKOS_LAMBDA(const int e, const int v, const int j, const int i) {
      const int m = face_m(e);
      const bool inner = (face_f(e) == BoundaryFace::inner_x3);
      const BoundaryFlag bc = inner ? bc_ix3 : bc_ox3;
      if (bc == BoundaryFlag::mg_multipole) {
        if (!has_phi) return;
        if (!(fill_tangential || (j >= js && j <= je && i >= is && i <= ie))) return;
        const Real p = mp_homogeneous ? 0.0 : phi(e, j*nx + i);
        for (int n = 0; n < ngh; ++n) {
          if (inner) u(m, v, ks - 1 - n, j, i) = 2.0*p - u(m, v, ks + n, j, i);
          else       u(m, v, ke + 1 + n, j, i) = 2.0*p - u(m, v, ke - n, j, i);
        }
      } else {
        const bool fixed = IsZeroFixedMGBoundary(bc);
        for (int n = 0; n < ngh; ++n) {
          if (inner) {
            u(m, v, ks - 1 - n, j, i) =
                fixed ? -u(m, v, ks + n, j, i) : u(m, v, ks, j, i);
          } else {
            u(m, v, ke + 1 + n, j, i) =
                fixed ? -u(m, v, ke - n, j, i) : u(m, v, ke, j, i);
          }
        }
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundarySameLevel(...)
//! \brief Set octet boundary from a neighbor on the same level

void MultigridDriver::SetOctetBoundarySameLevel(MGOctet &dst, const MGOctet &src_oct,
     int src_id, std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, int ox1, int ox2, int ox3, bool folddata) {
  const int ngh = mgroot_->ngh_;
  int is, ie, js, je, ks, ke, nis, njs, nks;
  if (ox1 == 0)     is = ngh,   ie = ngh+1, nis = ngh;
  else if (ox1 < 0) is = 0,     ie = ngh-1, nis = ngh+1;
  else              is = ngh+2, ie = ngh+2, nis = ngh;
  if (ox2 == 0)     js = ngh,   je = ngh+1, njs = ngh;
  else if (ox2 < 0) js = 0,     je = ngh-1, njs = ngh+1;
  else              js = ngh+2, je = ngh+2, njs = ngh;
  if (ox3 == 0)     ks = ngh,   ke = ngh+1, nks = ngh;
  else if (ox3 < 0) ks = 0,     ke = ngh-1, nks = ngh+1;
  else              ks = ngh+2, ke = ngh+2, nks = ngh;
  int ci = ox1 + 1, cj = ox2 + 1, ck = ox3 + 1;

  for (int v = 0; v < nvar; ++v) {
    for (int k = ks, nk = nks; k <= ke; ++k, ++nk) {
      for (int j = js, nj = njs; j <= je; ++j, ++nj) {
        for (int i = is, ni = nis; i <= ie; ++i, ++ni)
          dst.U(v, k, j, i) = src_oct.U(v, nk, nj, ni);
      }
    }
  }
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, ck, cj, ci) = octet_avg_[src_id*nvar + v];

  if (folddata) {
    for (int v = 0; v < nvar; ++v) {
      for (int k = ks, nk = nks; k <= ke; ++k, ++nk) {
        for (int j = js, nj = njs; j <= je; ++j, ++nj) {
          for (int i = is, ni = nis; i <= ie; ++i, ++ni)
            dst.Uold(v, k, j, i) = src_oct.Uold(v, nk, nj, ni);
        }
      }
    }
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, ck, cj, ci) = octet_avgold_[src_id*nvar + v];
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ComputeOctetInteriorAverages(int lev, bool folddata)
//! \brief Form every octet's eight-cell interior average once for a whole boundary pass.

// SetOctetBoundarySameLevel formed this for each of the 26 directions that names the
// octet as a neighbour, and SetOctetCoarseBufferPhysicalBoundaries and
// ProlongateOctetBoundaries each formed it again for the centre slot.  The summation is
// copied here unchanged, in the same order, so the values are the ones those three sites
// produced.
void MultigridDriver::ComputeOctetInteriorAverages(int lev, bool folddata) {
  const int noct = noctets_[lev];
  const int ngh = mgroot_->ngh_;
  constexpr Real fac = 0.125;
  const int l = ngh, r = ngh + 1;
  octet_avg_.resize(static_cast<std::size_t>(noct)*nvar_);
  if (folddata) octet_avgold_.resize(static_cast<std::size_t>(noct)*nvar_);
  for (int o = 0; o < noct; ++o) {
    const MGOctet &oct = octets_[lev][o];
    for (int v = 0; v < nvar_; ++v) {
      octet_avg_[o*nvar_ + v] = fac*(oct.U(v,l,l,l)+oct.U(v,l,l,r)
          +oct.U(v,l,r,l)+oct.U(v,r,l,l)
          +oct.U(v,r,r,l)+oct.U(v,r,l,r)+oct.U(v,l,r,r)+oct.U(v,r,r,r));
    }
    if (folddata) {
      for (int v = 0; v < nvar_; ++v) {
        octet_avgold_[o*nvar_ + v] = fac*(oct.Uold(v,l,l,l)+oct.Uold(v,l,l,r)
            +oct.Uold(v,l,r,l)+oct.Uold(v,r,l,l)
            +oct.Uold(v,r,r,l)+oct.Uold(v,r,l,r)+oct.Uold(v,l,r,r)+oct.Uold(v,r,r,r));
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundaryFromCoarser(...)
//! \brief Fill coarse buffer entry from a coarser neighbor (octet or root)

// un and unold are indexed with the three extents of the array they came from, because
// the root grid is one cell per root-level MeshBlock and a mesh whose root block counts
// differ between axes gives it different extents.  An octet passes its own edge three
// times over, which is what it is.
void MultigridDriver::SetOctetBoundaryFromCoarser(const Real *un, const Real *unold,
     std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, int un_nx, int un_ny, int un_nz, const LogicalLocation &loc,
     int ox1, int ox2, int ox3, bool folddata) {
  int ngh = mgroot_->ngh_;
  int ci, cj, ck;
  if (loc.level == locrootlevel_) { // from root
    ci = static_cast<int>(loc.lx1) + ngh;
    cj = static_cast<int>(loc.lx2) + ngh;
    ck = static_cast<int>(loc.lx3) + ngh;
  } else { // from a neighbor octet (given loc is MY location)
    int ix1 = (static_cast<int>(loc.lx1) & 1);
    int ix2 = (static_cast<int>(loc.lx2) & 1);
    int ix3 = (static_cast<int>(loc.lx3) & 1);
    if (ox1 == 0) ci = ix1 + ngh;
    else          ci = (ix1^1) + ngh;
    if (ox2 == 0) cj = ix2 + ngh;
    else          cj = (ix2^1) + ngh;
    if (ox3 == 0) ck = ix3 + ngh;
    else          ck = (ix3^1) + ngh;
  }
  int i = 1 + ox1, j = 1 + ox2, k = 1 + ox3;
  const int idx = ((ck*un_ny) + cj)*un_nx + ci;
  const int vstride = un_nz*un_ny*un_nx;
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, k, j, i) = un[v*vstride + idx];
  if (folddata) {
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, k, j, i) = unold[v*vstride + idx];
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ProlongateOctetBoundariesFluxCons(...)
//! \brief Flux-conservative coarse-fine ghost fill for smoothing on octet levels

void MultigridDriver::ProlongateOctetBoundariesFluxCons(MGOctet &oct,
     std::vector<Real> &cbuf, int nvar, const std::array<bool,27> &ncoarse) {
  constexpr Real ot = 1.0 / 3.0;
  // Same switch as the MeshBlock-level fill (Multigrid::FillFineCoarse..., the tg factor
  // at multigrid.cpp:3349): mg_fc_symmetric drops the tangential-gradient terms so the
  // coarse/fine interface is flux-matched and the composite operator symmetric.  The
  // octet levels are all coarser than the MeshBlock levels, so this changes the
  // coarse-grid correction operator -- the convergence rate -- and not the fixed point,
  // which the finest-level equation alone defines.  Exactly inert for the default
  // (mg_fc_symmetric = false, tg = 1.0).
  const Real tg = FCTangentialGradient() ? 1.0 : 0.0;
  const int ngh = mgroot_->ngh_;
  const int ci = ngh;
  const int cj = ngh;
  const int ck = ngh;
  const int l = ngh;
  const int r = ngh + 1;

  auto cb = [&](int v, int k, int j, int i) -> Real {
    return BufRef(cbuf, 3, v, k, j, i);
  };
  auto has_coarse = [&](int ox3, int ox2, int ox1) -> bool {
    return ncoarse[(ox3 + 1) * 9 + (ox2 + 1) * 3 + (ox1 + 1)];
  };

  // x1 faces
  for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
    if (!has_coarse(0, 0, ox1)) continue;
    int i, fi, fig;
    if (ox1 > 0) {
      i = ngh + 1; fi = ngh + 1; fig = ngh + 2;
    } else {
      i = ngh - 1; fi = ngh; fig = ngh - 1;
    }
    for (int v = 0; v < nvar; ++v) {
      Real ccval = cb(v, ck, cj, i);
      Real gx2c = tg * 0.125 * (cb(v, ck, cj + 1, i) - cb(v, ck, cj - 1, i));
      Real gx3c = tg * 0.125 * (cb(v, ck + 1, cj, i) - cb(v, ck - 1, cj, i));
      oct.U(v, l, l, fig) = ot * (2.0 * (ccval - gx2c - gx3c) + oct.U(v, l, l, fi));
      oct.U(v, l, r, fig) = ot * (2.0 * (ccval + gx2c - gx3c) + oct.U(v, l, r, fi));
      oct.U(v, r, l, fig) = ot * (2.0 * (ccval - gx2c + gx3c) + oct.U(v, r, l, fi));
      oct.U(v, r, r, fig) = ot * (2.0 * (ccval + gx2c + gx3c) + oct.U(v, r, r, fi));
    }
  }

  // x2 faces
  for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
    if (!has_coarse(0, ox2, 0)) continue;
    int j, fj, fjg;
    if (ox2 > 0) {
      j = ngh + 1; fj = ngh + 1; fjg = ngh + 2;
    } else {
      j = ngh - 1; fj = ngh; fjg = ngh - 1;
    }
    for (int v = 0; v < nvar; ++v) {
      Real ccval = cb(v, ck, j, ci);
      Real gx1c = tg * 0.125 * (cb(v, ck, j, ci + 1) - cb(v, ck, j, ci - 1));
      Real gx3c = tg * 0.125 * (cb(v, ck + 1, j, ci) - cb(v, ck - 1, j, ci));
      oct.U(v, l, fjg, l) = ot * (2.0 * (ccval - gx1c - gx3c) + oct.U(v, l, fj, l));
      oct.U(v, l, fjg, r) = ot * (2.0 * (ccval + gx1c - gx3c) + oct.U(v, l, fj, r));
      oct.U(v, r, fjg, l) = ot * (2.0 * (ccval - gx1c + gx3c) + oct.U(v, r, fj, l));
      oct.U(v, r, fjg, r) = ot * (2.0 * (ccval + gx1c + gx3c) + oct.U(v, r, fj, r));
    }
  }

  // x3 faces
  for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
    if (!has_coarse(ox3, 0, 0)) continue;
    int k, fk, fkg;
    if (ox3 > 0) {
      k = ngh + 1; fk = ngh + 1; fkg = ngh + 2;
    } else {
      k = ngh - 1; fk = ngh; fkg = ngh - 1;
    }
    for (int v = 0; v < nvar; ++v) {
      Real ccval = cb(v, k, cj, ci);
      Real gx1c = tg * 0.125 * (cb(v, k, cj, ci + 1) - cb(v, k, cj, ci - 1));
      Real gx2c = tg * 0.125 * (cb(v, k, cj + 1, ci) - cb(v, k, cj - 1, ci));
      oct.U(v, fkg, l, l) = ot * (2.0 * (ccval - gx1c - gx2c) + oct.U(v, fk, l, l));
      oct.U(v, fkg, l, r) = ot * (2.0 * (ccval + gx1c - gx2c) + oct.U(v, fk, l, r));
      oct.U(v, fkg, r, l) = ot * (2.0 * (ccval - gx1c + gx2c) + oct.U(v, fk, r, l));
      oct.U(v, fkg, r, r) = ot * (2.0 * (ccval + gx1c + gx2c) + oct.U(v, fk, r, r));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ProlongateOctetBoundaries(...)
//! \brief Tricubic coarse-to-fine ghost fill (used for prolongation/FAS-olddata)

void MultigridDriver::ProlongateOctetBoundaries(MGOctet &oct, int oct_id,
     std::vector<Real> &cbuf, std::vector<Real> &cbufold,
     int nvar, const std::array<bool,27> &ncoarse, bool folddata) {
  const int ngh = mgroot_->ngh_;
  const int nc = oct.nc;
  const int flim = nc;

  // Fill center of coarse buffer from this octet's own data (center = 1,1,1)
  for (int v = 0; v < nvar; ++v)
    BufRef(cbuf, 3, v, 1, 1, 1) = octet_avg_[oct_id*nvar + v];
  if (folddata) {
    for (int v = 0; v < nvar; ++v)
      BufRef(cbufold, 3, v, 1, 1, 1) = octet_avgold_[oct_id*nvar + v];
  }

  // Prolongate from coarse buffer to fine ghost cells where neighbor is coarser
  for (int ox3 = -1; ox3 <= 1; ++ox3) {
    for (int ox2 = -1; ox2 <= 1; ++ox2) {
      for (int ox1 = -1; ox1 <= 1; ++ox1) {
        if (ncoarse[(ox3+1)*9 + (ox2+1)*3 + (ox1+1)]) {
          int ci = ox1 + 1, cj = ox2 + 1, ck = ox3 + 1;
          int fi = ox1*2 + ngh, fj = ox2*2 + ngh, fk = ox3*2 + ngh;
          for (int v = 0; v < nvar; ++v) {
            auto cb = [&](int vv, int kk, int jj, int ii) -> Real {
              kk = std::max(0, std::min(2, kk));
              jj = std::max(0, std::min(2, jj));
              ii = std::max(0, std::min(2, ii));
              return BufRef(cbuf, 3, vv, kk, jj, ii);
            };
            if (fk >= 0 && fj >= 0 && fi >= 0)
              oct.U(v,fk,fj,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj-1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj-1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj-1,ci)+cb(v,ck-1,cj,ci-1)+cb(v,ck,cj-1,ci-1)));
            if (fk >= 0 && fj >= 0 && fi+1 < flim)
              oct.U(v,fk,fj,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj-1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj-1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj-1,ci)+cb(v,ck-1,cj,ci+1)+cb(v,ck,cj-1,ci+1)));
            if (fk >= 0 && fj+1 < flim && fi >= 0)
              oct.U(v,fk,fj+1,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj+1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj+1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj+1,ci)+cb(v,ck-1,cj,ci-1)+cb(v,ck,cj+1,ci-1)));
            if (fk+1 < flim && fj >= 0 && fi >= 0)
              oct.U(v,fk+1,fj,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj-1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj-1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj-1,ci)+cb(v,ck+1,cj,ci-1)+cb(v,ck,cj-1,ci-1)));
            if (fk+1 < flim && fj+1 < flim && fi >= 0)
              oct.U(v,fk+1,fj+1,fi) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj+1,ci-1)
                  +9.0*(cb(v,ck,cj,ci-1)+cb(v,ck,cj+1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj+1,ci)+cb(v,ck+1,cj,ci-1)+cb(v,ck,cj+1,ci-1)));
            if (fk+1 < flim && fj >= 0 && fi+1 < flim)
              oct.U(v,fk+1,fj,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj-1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj-1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj-1,ci)+cb(v,ck+1,cj,ci+1)+cb(v,ck,cj-1,ci+1)));
            if (fk >= 0 && fj+1 < flim && fi+1 < flim)
              oct.U(v,fk,fj+1,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck-1,cj+1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj+1,ci)+cb(v,ck-1,cj,ci))
                  +3.0*(cb(v,ck-1,cj+1,ci)+cb(v,ck-1,cj,ci+1)+cb(v,ck,cj+1,ci+1)));
            if (fk+1 < flim && fj+1 < flim && fi+1 < flim)
              oct.U(v,fk+1,fj+1,fi+1) =
                0.015625*(27.0*cb(v,ck,cj,ci)+cb(v,ck+1,cj+1,ci+1)
                  +9.0*(cb(v,ck,cj,ci+1)+cb(v,ck,cj+1,ci)+cb(v,ck+1,cj,ci))
                  +3.0*(cb(v,ck+1,cj+1,ci)+cb(v,ck+1,cj,ci+1)+cb(v,ck,cj+1,ci+1)));
          }
          if (folddata) {
            for (int v = 0; v < nvar; ++v) {
              auto co = [&](int vv, int kk, int jj, int ii) -> Real {
                kk = std::max(0, std::min(2, kk));
                jj = std::max(0, std::min(2, jj));
                ii = std::max(0, std::min(2, ii));
                return BufRef(cbufold, 3, vv, kk, jj, ii);
              };
              if (fk >= 0 && fj >= 0 && fi >= 0)
                oct.Uold(v,fk,fj,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj-1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj-1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj-1,ci)+co(v,ck-1,cj,ci-1)+co(v,ck,cj-1,ci-1)));
              if (fk >= 0 && fj >= 0 && fi+1 < flim)
                oct.Uold(v,fk,fj,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj-1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj-1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj-1,ci)+co(v,ck-1,cj,ci+1)+co(v,ck,cj-1,ci+1)));
              if (fk >= 0 && fj+1 < flim && fi >= 0)
                oct.Uold(v,fk,fj+1,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj+1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj+1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj+1,ci)+co(v,ck-1,cj,ci-1)+co(v,ck,cj+1,ci-1)));
              if (fk+1 < flim && fj >= 0 && fi >= 0)
                oct.Uold(v,fk+1,fj,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj-1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj-1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj-1,ci)+co(v,ck+1,cj,ci-1)+co(v,ck,cj-1,ci-1)));
              if (fk+1 < flim && fj+1 < flim && fi >= 0)
                oct.Uold(v,fk+1,fj+1,fi) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj+1,ci-1)
                    +9.0*(co(v,ck,cj,ci-1)+co(v,ck,cj+1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj+1,ci)+co(v,ck+1,cj,ci-1)+co(v,ck,cj+1,ci-1)));
              if (fk+1 < flim && fj >= 0 && fi+1 < flim)
                oct.Uold(v,fk+1,fj,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj-1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj-1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj-1,ci)+co(v,ck+1,cj,ci+1)+co(v,ck,cj-1,ci+1)));
              if (fk >= 0 && fj+1 < flim && fi+1 < flim)
                oct.Uold(v,fk,fj+1,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck-1,cj+1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj+1,ci)+co(v,ck-1,cj,ci))
                    +3.0*(co(v,ck-1,cj+1,ci)+co(v,ck-1,cj,ci+1)+co(v,ck,cj+1,ci+1)));
              if (fk+1 < flim && fj+1 < flim && fi+1 < flim)
                oct.Uold(v,fk+1,fj+1,fi+1) =
                  0.015625*(27.0*co(v,ck,cj,ci)+co(v,ck+1,cj+1,ci+1)
                    +9.0*(co(v,ck,cj,ci+1)+co(v,ck,cj+1,ci)+co(v,ck+1,cj,ci))
                    +3.0*(co(v,ck+1,cj+1,ci)+co(v,ck+1,cj,ci+1)+co(v,ck,cj+1,ci+1)));
            }
          }
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::RestrictOctetsBeforeTransfer()
//! \brief Restrict all octets to prepare for root-to-blocks transfer

// Leaves the result in the host copy of the root grid.  The caller owns the push to the
// device, which it does once for the whole transfer.
void MultigridDriver::RestrictOctetsBeforeTransfer() {
  Kokkos::Timer timer;
  const int ngh = mgroot_->ngh_;
  for (int l = nreflevel_ - 1; l >= 1; --l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &foct = octets_[l][o];
      int oid = octet_parent_[l][o];
      if (oid < 0) continue;
      MGOctet &coct = octets_[l-1][oid];
      int oi = octet_child_i_[l][o];
      int oj = octet_child_j_[l][o];
      int ok = octet_child_k_[l][o];
      for (int v = 0; v < nvar_; ++v)
        coct.U(v, ok, oj, oi) = RestrictOne(foct, v, ngh, ngh, ngh);
    }
  }
  auto root_u_h = GetRootData_h();
  for (int o = 0; o < noctets_[0]; ++o) {
    MGOctet &oct = octets_[0][o];
    int ri = octet_root_i_[o];
    int rj = octet_root_j_[o];
    int rk = octet_root_k_[o];
    for (int v = 0; v < nvar_; ++v)
      root_u_h(0, v, rk, rj, ri) =
          RestrictOne(oct, v, ngh, ngh, ngh);
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetOctetBoundariesBeforeTransfer(bool folddata)
//! \brief Set octet boundaries before transfer from root to blocks

void MultigridDriver::SetOctetBoundariesBeforeTransfer(bool folddata) {
  Kokkos::Timer timer;
  for (int lev = 0; lev < nreflevel_; ++lev) {
    if (lev == 0) {
      BuildRootFlatBuffers(folddata);
    }
    ComputeOctetInteriorAverages(lev, folddata);
    const auto &local_octets = octet_local_ids_[lev];
    auto process_local_octet = [&](int q, std::vector<Real> &cbuf_local,
                                   std::vector<Real> &cbufold_local,
                                   std::array<bool,27> &ncoarse_local) {
      int oid = local_octets[q];
      MGOctet &oct = octets_[lev][oid];
      const LogicalLocation &oloc = oct.loc;
      std::fill(ncoarse_local.begin(), ncoarse_local.end(), false);
      std::fill(cbuf_local.begin(), cbuf_local.end(), 0.0);
      std::fill(cbufold_local.begin(), cbufold_local.end(), 0.0);

      const int base = oid * 27;
      const auto &same_lut = octet_nb_same_[lev];
      const auto &coarse_lut = octet_nb_coarse_[lev];
      const auto &rootloc_lut = octet_nb_rootloc_[lev];
      for (int ox3 = -1; ox3 <= 1; ++ox3) {
        for (int ox2 = -1; ox2 <= 1; ++ox2) {
          for (int ox1 = -1; ox1 <= 1; ++ox1) {
            if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
            const int nidx = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
            int nid = same_lut[base + nidx];
            const LogicalLocation &nloc = rootloc_lut[base + nidx];
            const bool valid_neighbor = (nloc.level >= 0);
            if (!valid_neighbor) continue;
            if (nid >= 0) {
              MGOctet &noct = octets_[lev][nid];
              SetOctetBoundarySameLevel(oct, noct, nid, cbuf_local, cbufold_local,
                                        nvar_, ox1, ox2, ox3, folddata);
            } else {
              if (lev > 0) {
                int cid = coarse_lut[base + nidx];
                if (cid >= 0) {
                  ncoarse_local[nidx] = true;
                  MGOctet &coct = octets_[lev-1][cid];
                  SetOctetBoundaryFromCoarser(coct.u, coct.uold, cbuf_local,
                      cbufold_local,
                                              nvar_, coct.nc, coct.nc, coct.nc, oloc,
                                              ox1, ox2, ox3, folddata);
                }
              } else {
                ncoarse_local[nidx] = true;
                SetOctetBoundaryFromCoarser(root_u_buf_.data(), root_uold_buf_.data(),
                                            cbuf_local, cbufold_local, nvar_,
                                            root_buf_nx_, root_buf_ny_, root_buf_nz_,
                                            nloc, ox1, ox2, ox3, folddata);
              }
            }
          }
        }
      }
      SetOctetCoarseBufferPhysicalBoundaries(oct, oid, cbuf_local, cbufold_local,
                                             folddata);
      ProlongateOctetBoundaries(oct, oid, cbuf_local, cbufold_local, nvar_,
                                ncoarse_local, folddata);
      ApplyPhysicalBoundariesOctet(oct, folddata);
    };

    for (int q = 0; q < static_cast<int>(local_octets.size()); ++q) {
      process_local_octet(q, cbuf_, cbufold_, ncoarse_);
    }
  }
  if (collect_phase_timing_) timing_octets_time_ += timer.seconds();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::AllocateMultipoleCoefficients()
//! \brief Configure and initialize the isolated-boundary multipole coefficients.

void MultigridDriver::AllocateMultipoleCoefficients() {
  nmpcoeff_ = 0;
  std::fill(std::begin(mpcoeff_), std::end(mpcoeff_), 0.0);
  if (mporder_ <= 0) return;
  for (int l = 0; l <= mporder_; ++l) nmpcoeff_ += 2*l + 1;
  SyncMultipoleToDevice();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::EnsureMultipolePartials()
//! \brief Size the per-block partial-sum scratch to the current block capacity.

// The column count is the 25 of the widest expansion rather than the live nmpcoeff_, so
// that the multipole and centre-of-mass reductions share one buffer and neither the
// order nor autompo_ can resize it.  Rows follow the same grain-rounded capacity as the
// level arrays, so AMR jitter does not touch the allocation either.
void MultigridDriver::EnsureMultipolePartials() {
  const int nrows = std::max(1, mglevels_->nmmb_capacity_);
  const int nk = std::max(1, mglevels_->indcs_.nx3);
  if (mp_partial_h_.extent_int(0) != nrows) {
    if (!mglevels_->OnHost()) Kokkos::realloc(mp_partial_, nrows, 25);
    Kokkos::realloc(mp_partial_h_, nrows, 25);
  }
  // Unlike mp_partial_h_, which is where the device result lands, the per-slice scratch
  // is only ever read by the space that wrote it, so only that space needs one.
  if (mglevels_->OnHost()) {
    if (mp_kpartial_h_.extent_int(0) != nrows || mp_kpartial_h_.extent_int(1) != nk) {
      Kokkos::realloc(mp_kpartial_h_, nrows, nk, 25);
    }
  } else {
    if (mp_kpartial_.extent_int(0) != nrows || mp_kpartial_.extent_int(1) != nk) {
      Kokkos::realloc(mp_kpartial_, nrows, nk, 25);
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateMultipoleCoefficients()
//! \brief Integrate source-weighted solid harmonics over the finest MeshBlock level.

void MultigridDriver::CalculateMultipoleCoefficients() {
  if (mporder_ <= 0 || nmpcoeff_ == 0) return;
  std::fill(std::begin(mpcoeff_), std::end(mpcoeff_), 0.0);
  EnsureMultipolePartials();

  const int nmb = mglevels_->nmmb_;
  const int ngh = mglevels_->ngh_;
  const int nx1 = mglevels_->indcs_.nx1;
  const int nx2 = mglevels_->indcs_.nx2;
  const int nx3 = mglevels_->indcs_.nx3;
  const Real xorigin = mpo_[0];
  const Real yorigin = mpo_[1];
  const Real zorigin = mpo_[2];
  const int order = mporder_;
  const bool skip_dipole = nodipole_;
  const int ncoeff = nmpcoeff_;

  if (mglevels_->OnHost()) {
    ComputeMultipolePartials(HostExeSpace(),
        mglevels_->src_[mglevels_->nlevel_ - 1].h_view,
        pmy_pack_->pmb->mb_size.h_view, mp_partial_h_, mp_kpartial_h_,
        nmb, ngh, nx1, nx2, nx3,
        xorigin, yorigin, zorigin, order, skip_dipole, ncoeff);
  } else {
    ComputeMultipolePartials(DevExeSpace(),
        mglevels_->src_[mglevels_->nlevel_ - 1].d_view,
        pmy_pack_->pmb->mb_size.d_view, mp_partial_, mp_kpartial_,
        nmb, ngh, nx1, nx2, nx3,
        xorigin, yorigin, zorigin, order, skip_dipole, ncoeff);
    if (nmb > 0) {
      const auto rows = std::make_pair(0, nmb);
      Kokkos::deep_copy(Kokkos::subview(mp_partial_h_, rows, Kokkos::ALL()),
                        Kokkos::subview(mp_partial_, rows, Kokkos::ALL()));
    }
  }
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < nmpcoeff_; ++c) mpcoeff_[c] += mp_partial_h_(m, c);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, mpcoeff_, nmpcoeff_, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  ScaleMultipoleCoefficients();
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::ScaleMultipoleCoefficients()
//! \brief Apply the real solid-harmonic normalization used by EvalMultipolePhi().

void MultigridDriver::ScaleMultipoleCoefficients() {
  // The solver uses -laplacian(phi)=source with source=-4*pi*G*rho.  Positive
  // normalization factors therefore recover the negative gravitational potential.
  constexpr Real pi = 3.141592653589793238462643383279502884;
  constexpr Real c0  = 0.25/pi;
  constexpr Real c1  = 0.25/pi;
  constexpr Real c2  = 0.0625/pi;
  constexpr Real c2a = 0.75/pi;
  constexpr Real c30 = 0.0625/pi;
  constexpr Real c31 = 0.09375/pi;
  constexpr Real c32 = 3.75/pi;
  constexpr Real c33 = 0.15625/pi;
  constexpr Real c40 = 0.00390625/pi;
  constexpr Real c41 = 0.15625/pi;
  constexpr Real c42 = 0.3125/pi;
  constexpr Real c43 = 1.09375/pi;
  constexpr Real c44 = 8.75/pi;

  mpcoeff_[0] *= c0;
  mpcoeff_[1] *= c1;
  mpcoeff_[2] *= c1;
  mpcoeff_[3] *= c1;
  mpcoeff_[4] *= c2a;
  mpcoeff_[5] *= c2a;
  mpcoeff_[6] *= c2;
  mpcoeff_[7] *= c2a;
  mpcoeff_[8] *= c2a;
  if (mporder_ == 4) {
    mpcoeff_[9]  *= c33;
    mpcoeff_[10] *= c32;
    mpcoeff_[11] *= c31;
    mpcoeff_[12] *= c30;
    mpcoeff_[13] *= c31;
    mpcoeff_[14] *= c32;
    mpcoeff_[15] *= c33;
    mpcoeff_[16] *= c44;
    mpcoeff_[17] *= c43;
    mpcoeff_[18] *= c42;
    mpcoeff_[19] *= c41;
    mpcoeff_[20] *= c40;
    mpcoeff_[21] *= c41;
    mpcoeff_[22] *= c42;
    mpcoeff_[23] *= c43;
    mpcoeff_[24] *= c44;
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::CalculateCenterOfMass()
//! \brief Set the expansion origin to the source-weighted center of mass.

void MultigridDriver::CalculateCenterOfMass() {
  if (mporder_ <= 0) return;

  const int nmb = mglevels_->nmmb_;
  const int ngh = mglevels_->ngh_;
  const int nx1 = mglevels_->indcs_.nx1;
  const int nx2 = mglevels_->indcs_.nx2;
  const int nx3 = mglevels_->indcs_.nx3;
  Real totals[5] = {};
  EnsureMultipolePartials();
  if (mglevels_->OnHost()) {
    ComputeCenterOfMassPartials(HostExeSpace(),
        mglevels_->src_[mglevels_->nlevel_ - 1].h_view,
        pmy_pack_->pmb->mb_size.h_view, mp_partial_h_, mp_kpartial_h_,
        nmb, ngh, nx1, nx2, nx3);
  } else {
    ComputeCenterOfMassPartials(DevExeSpace(),
        mglevels_->src_[mglevels_->nlevel_ - 1].d_view,
        pmy_pack_->pmb->mb_size.d_view, mp_partial_, mp_kpartial_,
        nmb, ngh, nx1, nx2, nx3);
    if (nmb > 0) {
      const auto rows = std::make_pair(0, nmb);
      Kokkos::deep_copy(Kokkos::subview(mp_partial_h_, rows, Kokkos::ALL()),
                        Kokkos::subview(mp_partial_, rows, Kokkos::ALL()));
    }
  }
  for (int m = 0; m < nmb; ++m) {
    for (int c = 0; c < 5; ++c) totals[c] += mp_partial_h_(m, c);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  const Real mass_tol = 64.0*std::numeric_limits<Real>::epsilon()*totals[4];
  if (!std::isfinite(totals[0]) || totals[4] <= 0.0 ||
      std::abs(totals[0]) <= mass_tol) {
    if (mg_verbose_ > 0 && global_variable::my_rank == 0) {
      std::cout << "### WARNING in MultigridDriver::CalculateCenterOfMass" << std::endl
                << "Multipole source has zero or cancelling total mass; retaining the "
                << "previous expansion origin." << std::endl;
    }
    return;
  }
  const Real im = 1.0/totals[0];
  mpo_[0] = im*totals[3];
  mpo_[1] = im*totals[1];
  mpo_[2] = im*totals[2];
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SyncMultipoleToDevice()
//! \brief Update the persistent device copy used by root and MeshBlock boundaries.

void MultigridDriver::SyncMultipoleToDevice() {
  if (mporder_ <= 0 || nmpcoeff_ == 0) return;
  if (d_mpcoeff_.extent_int(0) != 25) {
    Kokkos::realloc(d_mpcoeff_, 25);
    Kokkos::realloc(h_mpcoeff_, 25);
  }
  for (int c = 0; c < 25; ++c) h_mpcoeff_(c) = mpcoeff_[c];
  Kokkos::deep_copy(d_mpcoeff_, h_mpcoeff_);
  ++mp_epoch_;
}


//----------------------------------------------------------------------------------------
//! \fn const MultigridDriver::MGBoundaryPhiCache &MultigridDriver::EnsureBoundaryPhiCache
//! \brief Evaluate the multipole potential on this level's faces, once per solve.

const MultigridDriver::MGBoundaryPhiCache &MultigridDriver::EnsureBoundaryPhiCache(
    int shift, int ngh, int ncells, int nx1_cells, int nx2_cells, int nx3_cells) {
  if (static_cast<int>(mgbc_phi_.size()) <= shift) mgbc_phi_.resize(shift + 1);
  MGBoundaryPhiCache &c = mgbc_phi_[shift];
  // The live block count, not the grain-rounded extent of the level arrays: mb_bcs is
  // sized to the live count, and a thread for a padding row has nothing to read.
  const int nmb = pmg->GetNumMeshBlocks();
  const int nx = ncells + 2*ngh;
  const bool multipole_ready =
      (mporder_ > 0 && nmpcoeff_ > 0 && d_mpcoeff_.extent_int(0) == 25);
  // The list follows the mesh and the potential follows the coefficients, and the two
  // move on different clocks: a regrid is rare, a new set of coefficients arrives every
  // solve.  Rebuilding the list only when the mesh moves keeps its two host-to-device
  // copies off the per-solve path.
  const bool list_stale = !c.valid || c.mesh_sig != mesh_sig_ || c.nmb != nmb ||
                          c.nx != nx;
  if (!list_stale && c.epoch == mp_epoch_) return c;

  bool any_multipole = false;
  for (int f = 0; f < 6; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) any_multipole = true;
  }

  if (list_stale) {
    // Grouped by face so that each of the three directional kernels launches over one
    // contiguous run of it, and so that the build kernel below takes the same branch
    // across a warp.
    auto mb_bcs_h = pmy_pack_->pmb->mb_bcs.h_view;
    std::vector<int> list_m, list_f;
    list_m.reserve(nmb);
    list_f.reserve(nmb);
    for (int f = 0; f < 6; ++f) {
      c.off[f] = static_cast<int>(list_m.size());
      if (IsPeriodicMGBoundary(mg_mesh_bcs_[f])) continue;
      for (int m = 0; m < nmb; ++m) {
        if (mb_bcs_h(m, f) == BoundaryFlag::block) continue;
        list_m.push_back(m);
        list_f.push_back(f);
      }
    }
    c.off[6] = static_cast<int>(list_m.size());
    c.nface = c.off[6];
    c.nx = nx;
    c.nmb = nmb;
    c.mesh_sig = mesh_sig_;
    c.valid = true;

    const int nrows = std::max(1, c.nface);
    if (c.face_m.extent_int(0) != nrows) {
      Kokkos::realloc(c.face_m, nrows);
      Kokkos::realloc(c.face_f, nrows);
      Kokkos::realloc(c.face_base, nrows, 3);
      Kokkos::realloc(c.face_dx, nrows, 3);
    }
    if (c.nface > 0) {
      auto hm = Kokkos::create_mirror_view(c.face_m);
      auto hf = Kokkos::create_mirror_view(c.face_f);
      auto hbase = Kokkos::create_mirror_view(c.face_base);
      auto hdx = Kokkos::create_mirror_view(c.face_dx);
      auto mb_gid_h = pmy_pack_->pmb->mb_gid.h_view;
      const LogicalLocation *lloc = pmy_mesh_->lloc_eachmb;
      const Real span1 = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
      const Real span2 = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
      const Real span3 = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
      const int cells[3] = {nx1_cells, nx2_cells, nx3_cells};
      const Real span[3] = {span1, span2, span3};
      const int nrb[3] = {nrbx1_, nrbx2_, nrbx3_};
      for (int e = 0; e < c.nface; ++e) {
        const int m = list_m[e];
        hm(e) = m;
        hf(e) = list_f[e];
        const LogicalLocation &loc = lloc[mb_gid_h(m)];
        const int lev = loc.level - locrootlevel_;
        const std::int64_t lx[3] = {loc.lx1, loc.lx2, loc.lx3};
        for (int d = 0; d < 3; ++d) {
          const std::int64_t nblocks = static_cast<std::int64_t>(nrb[d]) << lev;
          hbase(e, d) = static_cast<int>(lx[d]*cells[d]);
          hdx(e, d) = span[d]/static_cast<Real>(nblocks*cells[d]);
        }
      }
      Kokkos::deep_copy(c.face_m, hm);
      Kokkos::deep_copy(c.face_f, hf);
      Kokkos::deep_copy(c.face_base, hbase);
      Kokkos::deep_copy(c.face_dx, hdx);
    }
  }
  c.epoch = mp_epoch_;

  const int nrows = std::max(1, c.nface);
  c.has_phi = any_multipole && multipole_ready && c.nface > 0;
  const int phi_rows = c.has_phi ? nrows : 1;
  const int phi_cols = c.has_phi ? nx*nx : 1;
  if (c.phi.extent_int(0) != phi_rows || c.phi.extent_int(1) != phi_cols) {
    Kokkos::realloc(c.phi, phi_rows, phi_cols);
  }
  if (!c.has_phi) return c;

  auto face_f = c.face_f;
  auto face_base = c.face_base;
  auto face_dx = c.face_dx;
  auto phi = c.phi;
  auto d_mpc = d_mpcoeff_;
  const int mporder = mporder_;
  const Real mpx = mpo_[0], mpy = mpo_[1], mpz = mpo_[2];
  const int is = ngh, js = ngh, ks = ngh;
  const Real gx1min = pmy_mesh_->mesh_size.x1min;
  const Real gx1max = pmy_mesh_->mesh_size.x1max;
  const Real gx2min = pmy_mesh_->mesh_size.x2min;
  const Real gx2max = pmy_mesh_->mesh_size.x2max;
  const Real gx3min = pmy_mesh_->mesh_size.x3min;
  const Real gx3max = pmy_mesh_->mesh_size.x3max;
  const BoundaryFlag bc0 = mg_mesh_bcs_[0], bc1 = mg_mesh_bcs_[1];
  const BoundaryFlag bc2 = mg_mesh_bcs_[2], bc3 = mg_mesh_bcs_[3];
  const BoundaryFlag bc4 = mg_mesh_bcs_[4], bc5 = mg_mesh_bcs_[5];
  par_for("MGBlockPhiCache", DevExeSpace(), 0, c.nface-1, 0, nx-1, 0, nx-1,
  KOKKOS_LAMBDA(const int e, const int a, const int b) {
    const int f = face_f(e);
    const bool mp = (f == 0) ? (bc0 == BoundaryFlag::mg_multipole)
                  : (f == 1) ? (bc1 == BoundaryFlag::mg_multipole)
                  : (f == 2) ? (bc2 == BoundaryFlag::mg_multipole)
                  : (f == 3) ? (bc3 == BoundaryFlag::mg_multipole)
                  : (f == 4) ? (bc4 == BoundaryFlag::mg_multipole)
                             : (bc5 == BoundaryFlag::mg_multipole);
    if (!mp) { phi(e, a*nx + b) = 0.0; return; }
    const int b1 = face_base(e, 0), b2 = face_base(e, 1), b3 = face_base(e, 2);
    const Real dx1 = face_dx(e, 0), dx2 = face_dx(e, 1), dx3 = face_dx(e, 2);
    Real x, y, z;
    if (f < 2) {                        // a is k, b is j
      x = ((f == 0) ? gx1min : gx1max) - mpx;
      y = gx2min + (b2 + (b - js) + 0.5)*dx2 - mpy;
      z = gx3min + (b3 + (a - ks) + 0.5)*dx3 - mpz;
    } else if (f < 4) {                 // a is k, b is i
      x = gx1min + (b1 + (b - is) + 0.5)*dx1 - mpx;
      y = ((f == 2) ? gx2min : gx2max) - mpy;
      z = gx3min + (b3 + (a - ks) + 0.5)*dx3 - mpz;
    } else {                            // a is j, b is i
      x = gx1min + (b1 + (b - is) + 0.5)*dx1 - mpx;
      y = gx2min + (b2 + (a - js) + 0.5)*dx2 - mpy;
      z = ((f == 4) ? gx3min : gx3max) - mpz;
    }
    phi(e, a*nx + b) = EvalMultipolePhi(x, y, z, d_mpc.data(), mporder);
  });
  return c;
}


//----------------------------------------------------------------------------------------
//! \fn const MultigridDriver::MGRootPhiCache &MultigridDriver::EnsureRootPhiCache(...)
//! \brief The same field on the six faces of the root grid, which is a single block.

// Filled on whichever space the root's own sweep runs on, and from the copy of the
// coefficients that sweep would have read, so the cached value is the one the sweep would
// have computed down to the last bit.  The six planes together are a few thousand cells
// at the finest root level and a handful at the coarsest, so the build is cheaper than
// the single refresh it replaces, let alone the tens of them in a V-cycle.
const MultigridDriver::MGRootPhiCache &MultigridDriver::EnsureRootPhiCache(
    int ll, bool on_host, int nx, int ny, int nz, int ngh,
    Real dx1, Real dx2, Real dx3) {
  if (static_cast<int>(mgroot_phi_.size()) <= ll) mgroot_phi_.resize(ll + 1);
  MGRootPhiCache &c = mgroot_phi_[ll];
  const int plane[6] = {nz*ny, nz*ny, nz*nx, nz*nx, ny*nx, ny*nx};
  int off[6];
  int len = 0;
  for (int f = 0; f < 6; ++f) { off[f] = len; len += plane[f]; }
  if (c.valid && c.epoch == mp_epoch_ && c.len == len && c.on_host == on_host) return c;

  for (int f = 0; f < 6; ++f) c.off[f] = off[f];
  c.len = len;
  c.on_host = on_host;
  c.epoch = mp_epoch_;
  c.valid = true;
  if (on_host) {
    if (c.phi_h.extent_int(0) != len) Kokkos::realloc(c.phi_h, len);
  } else {
    if (c.phi_d.extent_int(0) != len) Kokkos::realloc(c.phi_d, len);
  }

  const Real x1min = pmy_mesh_->mesh_size.x1min, x1max = pmy_mesh_->mesh_size.x1max;
  const Real x2min = pmy_mesh_->mesh_size.x2min, x2max = pmy_mesh_->mesh_size.x2max;
  const Real x3min = pmy_mesh_->mesh_size.x3min, x3max = pmy_mesh_->mesh_size.x3max;
  const Real mpx = mpo_[0], mpy = mpo_[1], mpz = mpo_[2];
  const int mporder = mporder_;
  const bool ready =
      (mporder_ > 0 && nmpcoeff_ > 0 && d_mpcoeff_.extent_int(0) == 25);
  for (int f = 0; f < 6; ++f) {
    const bool mp = (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) && ready;
    const int na = (f < 4) ? nz : ny;
    const int nb = (f < 2) ? ny : nx;
    const int base = off[f];
    if (on_host) {
      FillRootBoundaryPhi(HostExeSpace(), c.phi_h, mpcoeff_, f, mp, base, na, nb, ngh,
          dx1, dx2, dx3, x1min, x1max, x2min, x2max, x3min, x3max,
          mpx, mpy, mpz, mporder);
    } else {
      FillRootBoundaryPhi(DevExeSpace(), c.phi_d, d_mpcoeff_.data(), f, mp, base, na, nb,
          ngh, dx1, dx2, dx3, x1min, x1max, x2min, x2max, x3min, x3max,
          mpx, mpy, mpz, mporder);
    }
  }
  return c;
}


void MultigridDriver::MGRootBoundary() {
  constexpr bool force_host_bc = false;
  int current_level = mgroot_->GetCurrentLevel();
  int nlevels = mgroot_->GetNumberOfLevels();
  int ngh = mgroot_->ngh_;
  int ll = nlevels - 1 - current_level;
  int nx = (mgroot_->indcs_.nx1 >> ll) + 2*ngh;
  int ny = (mgroot_->indcs_.nx2 >> ll) + 2*ngh;
  int nz = (mgroot_->indcs_.nx3 >> ll) + 2*ngh;
  const int ncx = nx - 2*ngh;
  const int ncy = ny - 2*ngh;
  const int ncz = nz - 2*ngh;
  BoundaryFlag bc_ix1 = mg_mesh_bcs_[BoundaryFace::inner_x1];
  BoundaryFlag bc_ox1 = mg_mesh_bcs_[BoundaryFace::outer_x1];
  BoundaryFlag bc_ix2 = mg_mesh_bcs_[BoundaryFace::inner_x2];
  BoundaryFlag bc_ox2 = mg_mesh_bcs_[BoundaryFace::outer_x2];
  BoundaryFlag bc_ix3 = mg_mesh_bcs_[BoundaryFace::inner_x3];
  BoundaryFlag bc_ox3 = mg_mesh_bcs_[BoundaryFace::outer_x3];
  const Real x1min = pmy_mesh_->mesh_size.x1min;
  const Real x1max = pmy_mesh_->mesh_size.x1max;
  const Real x2min = pmy_mesh_->mesh_size.x2min;
  const Real x2max = pmy_mesh_->mesh_size.x2max;
  const Real x3min = pmy_mesh_->mesh_size.x3min;
  const Real x3max = pmy_mesh_->mesh_size.x3max;
  const Real dx1 = (x1max - x1min)/static_cast<Real>(ncx);
  const Real dx2 = (x2max - x2min)/static_cast<Real>(ncy);
  const Real dx3 = (x3max - x3min)/static_cast<Real>(ncz);

  // The potential on the six faces is a function of the coefficients and the origin
  // alone, both of which are fixed for a whole solve, so it is evaluated once per root
  // level rather than on every one of the tens of refreshes a V-cycle makes.
  const bool bc_on_host = mgroot_->on_host_ || force_host_bc;
  const MGRootPhiCache &rc =
      EnsureRootPhiCache(ll, bc_on_host, nx, ny, nz, ngh, dx1, dx2, dx3);
  const int o_ix1 = rc.off[0], o_ox1 = rc.off[1], o_ix2 = rc.off[2];
  const int o_ox2 = rc.off[3], o_ix3 = rc.off[4], o_ox3 = rc.off[5];

  // Root-grid BCs are applied directly on device when possible, with a host
  // fallback for host-only root storage.  Both forms sweep x1, then x2, then x3 over the
  // full tangential extent, so the edge and corner ghosts come out of the composition of
  // the rules that meet there; the multipole branch takes part in that composition like
  // every other rule, because the expansion is an exterior analytic field and is defined
  // a cell outside the domain, the same reason SetOctetCoarseBufferPhysicalBoundaries
  // evaluates it on its own edge and corner slots.  Skipping the tangential ghosts there
  // left them at whatever the previous sweep or the last cycle had put in them.
  if (!bc_on_host) {
    auto u = mgroot_->GetCurrentData();
    auto rphi = rc.phi_d;
    int nvar = u.extent_int(1);
    using TeamPolicy = Kokkos::TeamPolicy<DevExeSpace>;
    TeamPolicy policy(DevExeSpace(), nvar, Kokkos::AUTO);
    Kokkos::parallel_for("MGRootBoundaryXYZ", policy,
        KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
      const int v = team.league_rank();

      for (int n = 0; n < ngh; ++n) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nz * ny), [&](const int idx) {
          const int k = idx / ny;
          const int j = idx - k * ny;
          if (IsPeriodicMGBoundary(bc_ix1)) {
            u(0, v, k, j, n) = u(0, v, k, j, nx - 2 * ngh + n);
          } else if (IsZeroFixedMGBoundary(bc_ix1)) {
            u(0, v, k, j, ngh - 1 - n) = -u(0, v, k, j, ngh + n);
          } else if (IsZeroGradMGBoundary(bc_ix1)) {
            u(0, v, k, j, ngh - 1 - n) = u(0, v, k, j, ngh);
          } else if (bc_ix1 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ix1 + k*ny + j);
            u(0, v, k, j, ngh - 1 - n) = 2.0*phi - u(0, v, k, j, ngh + n);
          }
          if (IsPeriodicMGBoundary(bc_ox1)) {
            u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, ngh + n);
          } else if (IsZeroFixedMGBoundary(bc_ox1)) {
            u(0, v, k, j, nx - ngh + n) = -u(0, v, k, j, nx - ngh - 1 - n);
          } else if (IsZeroGradMGBoundary(bc_ox1)) {
            u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, nx - ngh - 1);
          } else if (bc_ox1 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ox1 + k*ny + j);
            u(0, v, k, j, nx - ngh + n) =
                2.0*phi - u(0, v, k, j, nx - ngh - 1 - n);
          }
        });
      }
      team.team_barrier();

      for (int n = 0; n < ngh; ++n) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nz * nx), [&](const int idx) {
          const int k = idx / nx;
          const int i = idx - k * nx;
          if (IsPeriodicMGBoundary(bc_ix2)) {
            u(0, v, k, n, i) = u(0, v, k, ny - 2 * ngh + n, i);
          } else if (IsZeroFixedMGBoundary(bc_ix2)) {
            u(0, v, k, ngh - 1 - n, i) = -u(0, v, k, ngh + n, i);
          } else if (IsZeroGradMGBoundary(bc_ix2)) {
            u(0, v, k, ngh - 1 - n, i) = u(0, v, k, ngh, i);
          } else if (bc_ix2 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ix2 + k*nx + i);
            u(0, v, k, ngh - 1 - n, i) = 2.0*phi - u(0, v, k, ngh + n, i);
          }
          if (IsPeriodicMGBoundary(bc_ox2)) {
            u(0, v, k, ny - ngh + n, i) = u(0, v, k, ngh + n, i);
          } else if (IsZeroFixedMGBoundary(bc_ox2)) {
            u(0, v, k, ny - ngh + n, i) = -u(0, v, k, ny - ngh - 1 - n, i);
          } else if (IsZeroGradMGBoundary(bc_ox2)) {
            u(0, v, k, ny - ngh + n, i) = u(0, v, k, ny - ngh - 1, i);
          } else if (bc_ox2 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ox2 + k*nx + i);
            u(0, v, k, ny - ngh + n, i) =
                2.0*phi - u(0, v, k, ny - ngh - 1 - n, i);
          }
        });
      }
      team.team_barrier();

      for (int n = 0; n < ngh; ++n) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ny * nx), [&](const int idx) {
          const int j = idx / nx;
          const int i = idx - j * nx;
          if (IsPeriodicMGBoundary(bc_ix3)) {
            u(0, v, n, j, i) = u(0, v, nz - 2 * ngh + n, j, i);
          } else if (IsZeroFixedMGBoundary(bc_ix3)) {
            u(0, v, ngh - 1 - n, j, i) = -u(0, v, ngh + n, j, i);
          } else if (IsZeroGradMGBoundary(bc_ix3)) {
            u(0, v, ngh - 1 - n, j, i) = u(0, v, ngh, j, i);
          } else if (bc_ix3 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ix3 + j*nx + i);
            u(0, v, ngh - 1 - n, j, i) = 2.0*phi - u(0, v, ngh + n, j, i);
          }
          if (IsPeriodicMGBoundary(bc_ox3)) {
            u(0, v, nz - ngh + n, j, i) = u(0, v, ngh + n, j, i);
          } else if (IsZeroFixedMGBoundary(bc_ox3)) {
            u(0, v, nz - ngh + n, j, i) = -u(0, v, nz - ngh - 1 - n, j, i);
          } else if (IsZeroGradMGBoundary(bc_ox3)) {
            u(0, v, nz - ngh + n, j, i) = u(0, v, nz - ngh - 1, j, i);
          } else if (bc_ox3 == BoundaryFlag::mg_multipole) {
            const Real phi = rphi(o_ox3 + j*nx + i);
            u(0, v, nz - ngh + n, j, i) =
                2.0*phi - u(0, v, nz - ngh - 1 - n, j, i);
          }
        });
      }
    });
  } else {
    if (!mgroot_->on_host_) SyncRootDataToHost(false);
    auto u = mgroot_->GetCurrentData_h();
    auto rphi_h = rc.phi_h;
    int nvar = u.extent_int(1);

    for (int v = 0; v < nvar; ++v) {
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int n = 0; n < ngh; ++n) {
            if (IsPeriodicMGBoundary(bc_ix1)) {
              u(0, v, k, j, n) = u(0, v, k, j, nx - 2*ngh + n);
            } else if (IsZeroFixedMGBoundary(bc_ix1)) {
              u(0, v, k, j, ngh - 1 - n) = -u(0, v, k, j, ngh + n);
            } else if (IsZeroGradMGBoundary(bc_ix1)) {
              u(0, v, k, j, ngh - 1 - n) = u(0, v, k, j, ngh);
            } else if (bc_ix1 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ix1 + k*ny + j);
              u(0, v, k, j, ngh - 1 - n) = 2.0*phi - u(0, v, k, j, ngh + n);
            }
            if (IsPeriodicMGBoundary(bc_ox1)) {
              u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, ngh + n);
            } else if (IsZeroFixedMGBoundary(bc_ox1)) {
              u(0, v, k, j, nx - ngh + n) = -u(0, v, k, j, nx - ngh - 1 - n);
            } else if (IsZeroGradMGBoundary(bc_ox1)) {
              u(0, v, k, j, nx - ngh + n) = u(0, v, k, j, nx - ngh - 1);
            } else if (bc_ox1 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ox1 + k*ny + j);
              u(0, v, k, j, nx - ngh + n) =
                  2.0*phi - u(0, v, k, j, nx - ngh - 1 - n);
            }
          }
        }
      }
      for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
          for (int n = 0; n < ngh; ++n) {
            if (IsPeriodicMGBoundary(bc_ix2)) {
              u(0, v, k, n, i) = u(0, v, k, ny - 2*ngh + n, i);
            } else if (IsZeroFixedMGBoundary(bc_ix2)) {
              u(0, v, k, ngh - 1 - n, i) = -u(0, v, k, ngh + n, i);
            } else if (IsZeroGradMGBoundary(bc_ix2)) {
              u(0, v, k, ngh - 1 - n, i) = u(0, v, k, ngh, i);
            } else if (bc_ix2 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ix2 + k*nx + i);
              u(0, v, k, ngh - 1 - n, i) = 2.0*phi - u(0, v, k, ngh + n, i);
            }
            if (IsPeriodicMGBoundary(bc_ox2)) {
              u(0, v, k, ny - ngh + n, i) = u(0, v, k, ngh + n, i);
            } else if (IsZeroFixedMGBoundary(bc_ox2)) {
              u(0, v, k, ny - ngh + n, i) = -u(0, v, k, ny - ngh - 1 - n, i);
            } else if (IsZeroGradMGBoundary(bc_ox2)) {
              u(0, v, k, ny - ngh + n, i) = u(0, v, k, ny - ngh - 1, i);
            } else if (bc_ox2 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ox2 + k*nx + i);
              u(0, v, k, ny - ngh + n, i) =
                  2.0*phi - u(0, v, k, ny - ngh - 1 - n, i);
            }
          }
        }
      }
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          for (int n = 0; n < ngh; ++n) {
            if (IsPeriodicMGBoundary(bc_ix3)) {
              u(0, v, n, j, i) = u(0, v, nz - 2*ngh + n, j, i);
            } else if (IsZeroFixedMGBoundary(bc_ix3)) {
              u(0, v, ngh - 1 - n, j, i) = -u(0, v, ngh + n, j, i);
            } else if (IsZeroGradMGBoundary(bc_ix3)) {
              u(0, v, ngh - 1 - n, j, i) = u(0, v, ngh, j, i);
            } else if (bc_ix3 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ix3 + j*nx + i);
              u(0, v, ngh - 1 - n, j, i) = 2.0*phi - u(0, v, ngh + n, j, i);
            }
            if (IsPeriodicMGBoundary(bc_ox3)) {
              u(0, v, nz - ngh + n, j, i) = u(0, v, ngh + n, j, i);
            } else if (IsZeroFixedMGBoundary(bc_ox3)) {
              u(0, v, nz - ngh + n, j, i) = -u(0, v, nz - ngh - 1 - n, j, i);
            } else if (IsZeroGradMGBoundary(bc_ox3)) {
              u(0, v, nz - ngh + n, j, i) = u(0, v, nz - ngh - 1, j, i);
            } else if (bc_ox3 == BoundaryFlag::mg_multipole) {
              const Real phi = rphi_h(o_ox3 + j*nx + i);
              u(0, v, nz - ngh + n, j, i) =
                  2.0*phi - u(0, v, nz - ngh - 1 - n, j, i);
            }
          }
        }
      }
    }
    if (!mgroot_->on_host_) SyncRootDataToDevice();
  }

  root_flat_buf_stale_ = true;
  root_uold_buf_valid_ = false;
}
