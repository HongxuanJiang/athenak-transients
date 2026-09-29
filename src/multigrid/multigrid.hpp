#ifndef MULTIGRID_MULTIGRID_HPP_
#define MULTIGRID_MULTIGRID_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multigrid.hpp
//  \brief defines the Multigrid base class

// C headers

// C++ headers
#include <cstdint>  // std::int64_t
#include <cstdio> // std::size_t
#include <cstring> // memcpy
#include <algorithm>  // std::max
#include <array>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <utility>  // std::make_pair
#include <vector>

// AthenaK headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../tasklist/task_list.hpp"
#include "../bvals/bvals.hpp"

class Mesh;
class MeshBlockPack;
class ParameterInput;
class Coordinates;
class MultigridDriver;
class MultigridBoundaryValues;

enum class MGVariable {src, u};
enum class MGNormType {max, l1, l2};

//----------------------------------------------------------------------------------------
// LogicalLocation hash and equality for std::unordered_map

inline bool operator==(const LogicalLocation &l1, const LogicalLocation &l2) {
  return (l1.level == l2.level) && (l1.lx1 == l2.lx1)
      && (l1.lx2 == l2.lx2) && (l1.lx3 == l2.lx3);
}

inline std::int64_t rotl64(std::int64_t i, int s) {
  return (i << s) | (i >> (64 - s));
}

struct LogicalLocationHash {
  std::size_t operator()(const LogicalLocation &l) const {
    return static_cast<std::size_t>(l.lx1 ^ rotl64(l.lx2, 21) ^ rotl64(l.lx3, 42));
  }
};

//----------------------------------------------------------------------------------------
//! \class MGOctet
//  \brief structure containing 2x2x2 interior cells (+ ghost) for mesh refinement
//  Each octet represents a "parent cell" that has children at a finer level.
//  Arrays are flat with 4D indexing (v, k, j, i).
//
//  The four field arrays are NOT owned by the octet.  They point into the single
//  contiguous pool that InitializeOctets() lays out for the whole refinement level, so
//  a level of N octets costs one allocation instead of 4N scattered ones, and the
//  level-wide operations (clear u, copy u into uold) become one bulk call instead of
//  N.  Within a pool the fields are stored one after the other -- all of u, then all of
//  def, then src, then uold -- so each of those level-wide runs is contiguous.

class MGOctet {
 public:
  LogicalLocation loc;
  bool fleaf;
  int nc;  // nc = 2 + 2*ngh

  Real *u = nullptr, *def = nullptr, *src = nullptr, *uold = nullptr;

  // Point this octet at slot `o` of a level pool whose per-field run is `stride` Reals.
  void Attach(Real *pool, std::size_t o, std::size_t stride, int nv, int ngh) {
    nc = 2 + 2*ngh;
    const std::size_t sz = static_cast<std::size_t>(nv)*nc*nc*nc;
    u    = pool + o*sz;
    def  = pool + stride + o*sz;
    src  = pool + 2*stride + o*sz;
    uold = pool + 3*stride + o*sz;
  }

  inline Real& U(int v, int k, int j, int i) {
    return u[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Def(int v, int k, int j, int i) {
    return def[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Src(int v, int k, int j, int i) {
    return src[((v*nc + k)*nc + j)*nc + i];
  }
  inline Real& Uold(int v, int k, int j, int i) {
    return uold[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& U(int v, int k, int j, int i) const {
    return u[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Def(int v, int k, int j, int i) const {
    return def[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Src(int v, int k, int j, int i) const {
    return src[((v*nc + k)*nc + j)*nc + i];
  }
  inline const Real& Uold(int v, int k, int j, int i) const {
    return uold[((v*nc + k)*nc + j)*nc + i];
  }
};

struct MultigridTaskIDs {
      TaskID send0;
      TaskID ircv0;
      TaskID recv0;
      TaskID physb0;
      TaskID send1;
      TaskID ircv1;
      TaskID recv1;
      TaskID physb1;
      TaskID sendR;
      TaskID ircvR;
      TaskID recvR;
      TaskID physbR;
      TaskID smoothR;
      TaskID sendB;
      TaskID ircvB;
      TaskID recvB;
      TaskID physbB;
      TaskID smoothB;
      TaskID sendR2;
      TaskID ircvR2;
      TaskID recvR2;
      TaskID physbR2;
      TaskID smoothR2;
      TaskID sendB2;
      TaskID ircvB2;
      TaskID recvB2;
      TaskID physbB2;
      TaskID smoothB2;
      TaskID restrict_;
      TaskID prolongate;
      TaskID prepare_correction;
      TaskID fmg_prolongate;
      TaskID calc_rhs;
      TaskID clear_recv0;
      TaskID clear_send0;
      TaskID clear_recvB2;
      TaskID clear_sendB2;
      TaskID fc_ghosts0;
      TaskID fc_ghostsR;
      TaskID fc_ghostsB;
      TaskID fc_ghostsR2;
      TaskID fc_ghostsB2;
      TaskID fc_ghosts_prol;
};

//! \class Multigrid
//  \brief Multigrid object containing each MeshBlock and/or the root block

class Multigrid {
 public:
  Multigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
            bool on_host = false);
  virtual ~Multigrid();

  MultigridBoundaryValues *pbval = nullptr;
  void ReallocateForAMR();
  void UpdateBlockDx();
  void LoadFinestData(const DvceArray5D<Real> &src, int ns, int ngh);
  // src_min/src_floor bracket the continuous coupling weight of
  // utils/gravity_weight.hpp: the loaded source is w(s)*s, with w ramping from zero at
  // src_floor to one at src_min.  src_min at or below src_floor leaves w == 1 and the
  // loaded source untouched.
  void LoadSource(const DvceArray5D<Real> &src, int ns, int ngh, Real fac,
                  Real src_min = -1.0, Real src_floor = -1.0);
  void LoadSourceAndSubtractAverage(const DvceArray5D<Real> &src, int ns, int ngh,
                                    Real fac, Real src_min = -1.0,
                                    Real src_floor = -1.0);
  void RetrieveResult(DvceArray5D<Real> &dst, int ns, int ngh);
  void ZeroClearData();
  void RestrictPack();
  void RestrictSourcePack();
  void FMGProlongatePack();
  void ProlongateAndCorrectPack();
  void ProlongatePreparedCorrectionPack();
  virtual void SmoothPack(int color) = 0;
  virtual void CalculateDefectPack() = 0;
  virtual void CalculateFASRHSPack() = 0;
  void ComputeCorrection();
  void SetFromRootGrid(bool folddata);
  Real CalculateDefectNorm(MGNormType nrm, int n);
  Real CalculateArrayNorm(MGVariable type, MGNormType nrm, int n);
  Real CalculateAverage(MGVariable type);
  void SubtractAverage(MGVariable type, int n, Real ave);
  void StoreOldData();

  // The MeshBlock dimension of the per-block arrays is over-allocated to a whole storage
  // grain (see mesh/mb_storage.hpp) so that AMR jitter does not reallocate them; nmmb_
  // stays the LIVE block count and bounds every kernel.  Every accessor below hands out
  // the live [0, nmmb_) range only, because consumers outside this class size their loops
  // from extent(0) -- MultigridDriver::ApplyPhysicalBoundariesBlocks does exactly that --
  // and must never see the padding.
  template <typename ViewType>
  auto LiveBlocks(const ViewType &v) const {
    // Clamped, because uold_ on the finest MeshBlock level is deliberately never
    // allocated and must keep handing back the empty view it does today.
    const int nlive = (v.extent_int(0) < nmmb_) ? v.extent_int(0) : nmmb_;
    return Kokkos::subview(v, std::make_pair(0, nlive), Kokkos::ALL(), Kokkos::ALL(),
                           Kokkos::ALL(), Kokkos::ALL());
  }

  // small functions
  int GetCurrentNumberOfCells() { return 1<<current_level_; }
  int GetNumberOfLevels() { return nlevel_; }
  int GetCurrentLevel() { return current_level_; }
  void SetCurrentLevel(int lev) { current_level_ = lev; }
  int GetLevelShift() { return nlevel_ - 1 - current_level_; }
  int GetSize() { return indcs_.nx1; }
  int GetGhostCells() { return ngh_; }
  Real GetRootDx() { return rdx_; }
  Real GetRootDy() { return rdy_; }
  Real GetRootDz() { return rdz_; }
  auto GetCurrentData() { return LiveBlocks(u_[current_level_].d_view); }
  auto GetCurrentSource() { return LiveBlocks(src_[current_level_].d_view); }
  auto GetCurrentOldData() { return LiveBlocks(uold_[current_level_].d_view); }
  auto GetCurrentData_h() { return LiveBlocks(u_[current_level_].h_view); }
  auto GetCurrentSource_h() { return LiveBlocks(src_[current_level_].h_view); }
  auto GetCurrentOldData_h() { return LiveBlocks(uold_[current_level_].h_view); }
  bool OnHost() const { return on_host_; }
  auto GetDataAtLevel(int lev) { return LiveBlocks(u_[lev].d_view); }
  auto GetSourceAtLevel(int lev) { return LiveBlocks(src_[lev].d_view); }
  auto GetOldDataAtLevel(int lev) { return LiveBlocks(uold_[lev].d_view); }
  int GetNumMeshBlocks() const { return nmmb_; }
  void MarkCurrentDataDeviceModified() { u_[current_level_].template modify<DevExeSpace>(); }
  void MarkCurrentSourceDeviceModified() {
    src_[current_level_].template modify<DevExeSpace>();
  }
  void MarkCurrentOldDataDeviceModified() {
    uold_[current_level_].template modify<DevExeSpace>();
  }
  void MarkDataAtLevelDeviceModified(int lev) { u_[lev].template modify<DevExeSpace>(); }
  void MarkOldDataAtLevelDeviceModified(int lev) {
    uold_[lev].template modify<DevExeSpace>();
  }
  void SyncCurrentDataToHost() { u_[current_level_].template sync<HostExeSpace>(); }
  void SyncCurrentSourceToHost() { src_[current_level_].template sync<HostExeSpace>(); }
  void SyncCurrentOldDataToHost() { uold_[current_level_].template sync<HostExeSpace>(); }


  // actual implementations of Multigrid operations (templated on view type)
  template <typename ViewType>
  void Restrict(ViewType &dst, const ViewType &src,
                int nvar, int il, int iu, int jl, int ju, int kl, int ku, bool th);
  template <typename ViewType>
  void ProlongateAndCorrect(ViewType &dst, const ViewType &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl, bool th);
  template <typename ViewType>
  void FMGProlongate(ViewType &dst, const ViewType &src,
    int il, int iu, int jl, int ju, int kl, int ku, int fil, int fjl, int fkl);

  // Physics-dependent operations templated on view type and stencil.
  // The stencil functor must provide:
  //   Real Apply(const ViewType&, int m, int v, int k, int j, int i)
  //   Real omega_over_diag
  template <typename ViewType, typename StencilOp>
  void Smooth(ViewType &u, const ViewType &src, const StencilOp &stencil, int rlev,
              int il, int iu, int jl, int ju, int kl, int ku, int color);

  template <typename ViewType, typename StencilOp>
  void CalculateDefect(ViewType &def, const ViewType &u, const ViewType &src,
                       const StencilOp &stencil, int rlev,
                       int il, int iu, int jl, int ju, int kl, int ku) {
    using ExeSpace = typename ViewType::execution_space;
    auto brdx = [this]() {
      if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
        return block_rdx_.h_view;
      else
        return block_rdx_.d_view;
    }();
    int rlev_l = rlev;
    par_for("Multigrid::CalculateDefect", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                              : brdx(m) / static_cast<Real>(1<<rlev_l);
      Real idx2 = 1.0 / (dx * dx);
      def(m,0,k,j,i) = src(m,0,k,j,i) - stencil.Apply(u, m, 0, k, j, i) * idx2;
    });
  }

  template <typename ViewType, typename StencilOp>
  void CalculateFASRHS(ViewType &src, const ViewType &u, const StencilOp &stencil,
                       int rlev, int il, int iu, int jl, int ju, int kl, int ku) {
    using ExeSpace = typename ViewType::execution_space;
    auto brdx = [this]() {
      if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
        return block_rdx_.h_view;
      else
        return block_rdx_.d_view;
    }();
    int rlev_l = rlev;
    par_for("Multigrid::CalculateFASRHS", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                              : brdx(m) / static_cast<Real>(1<<rlev_l);
      Real idx2 = 1.0 / (dx * dx);
      src(m,0,k,j,i) += stencil.Apply(u, m, 0, k, j, i) * idx2;
    });
  }

  friend class MultigridDriver;
  friend class MultigridBoundaryValues;

 protected:
  MultigridDriver *pmy_driver_;
  MeshBlock *pmy_block_;
  MeshBlockPack *pmy_pack_;
  Mesh *pmy_mesh_;
  RegionSize size_;
  RegionIndcs indcs_;
  BoundaryFlag mg_block_bcs_[6];
  int nlevel_, ngh_, nvar_, current_level_;
  int nmmb_;           // LIVE number of MeshBlocks; the bound of every kernel
  int nmmb_capacity_;  // block extent actually allocated, >= nmmb_ (mb_storage.hpp)
  bool on_host_;

  Real rdx_, rdy_, rdz_;
  Real dvol_over_dx3_;
  Real defscale_;
  DualArray1D<Real> block_rdx_;
  DualArray1D<int> block_color_parity_;
  // Labelled at construction: Kokkos::realloc inherits the label, and a default-built
  // View reports its allocation failures anonymously.
  DvceArray1D<int> fc_childx_{"mg_fc_childx", 0};
  DvceArray1D<int> fc_childy_{"mg_fc_childy", 0};
  DvceArray1D<int> fc_childz_{"mg_fc_childz", 0};
  DualArray5D<Real> *u_ = nullptr, *def_ = nullptr, *src_ = nullptr, *uold_ = nullptr;
};


//! \class MultigridDriver
//  \brief Multigrid driver

class MultigridDriver {
 public:
  MultigridDriver(MeshBlockPack *pmbp, int invar);
  virtual ~MultigridDriver();

  auto GetRootData_h() { return mgroot_->GetCurrentData_h(); }
  auto GetRootOldData_h() { return mgroot_->GetCurrentOldData_h(); }
  auto GetRootSource_h() { return mgroot_->GetCurrentSource_h(); }
  // pure virtual function
  virtual void Solve(Driver *pdriver, int step, Real dt = 0.0) = 0;
  void PrepareForAMR();
  // Clear the sticky AMR flag; see the comment on PrepareForAMR().  Call this only
  // once a full solve has actually run under the post-refinement policy.
  void ConsumeAMRMeshChanged() { amr_mesh_changed_ = false; }
  void RefreshMeshblockBoundaryValues(ParameterInput *pin);
  int GetCoffset() const { return coffset_; }
  bool ActiveReducedSameExchange() const { return active_reduce_same_exchange_; }
  void MGRootBoundary();
  void AllocateMultipoleCoefficients();
  void EnsureMultipolePartials();
  void CalculateMultipoleCoefficients();
  void ScaleMultipoleCoefficients();
  void CalculateCenterOfMass();
  void SyncMultipoleToDevice();

  // The multipole potential on the ghost cells of a physical face, evaluated once per
  // solve per level.  The coefficients and the expansion origin are fixed for a whole
  // solve, so this field is too, while the boundary kernels that consume it run tens of
  // times per V-cycle and cost a square root and about a hundred flops per ghost cell
  // each time.  face_m / face_f is also the compact list of the (block, face) pairs that
  // actually carry a physical boundary: on a deeply refined mesh most blocks touch none,
  // and the kernels used to launch over every block and let those threads fall through.
  struct MGBoundaryPhiCache {
    DvceArray1D<int> face_m, face_f;
    // Global cell index of the block's first interior cell, and the cell size at the
    // block's refinement level, per axis.  The tangential coordinate is built from these
    // rather than from the block's own extent so that two blocks meeting on a physical
    // face agree on it bit for bit.
    DvceArray2D<int> face_base;
    DvceArray2D<Real> face_dx;
    DvceArray2D<Real> phi;   // one nx*nx tangential plane per entry
    int off[7] = {};         // first entry of each face; off[6] is the total
    int nface = 0;
    int nx = 0;
    int nmb = -1;
    bool has_phi = false;
    std::uint64_t mesh_sig = 0;
    std::uint64_t epoch = 0;
    bool valid = false;
  };
  // The root grid is one block spanning the mesh, so its faces need no list -- only the
  // six planes, whose extents differ when the root is not cubic.
  struct MGRootPhiCache {
    HostArray1D<Real> phi_h;
    DvceArray1D<Real> phi_d;
    int off[6] = {};
    int len = 0;
    bool on_host = false;
    std::uint64_t epoch = 0;
    bool valid = false;
  };
  const MGBoundaryPhiCache &EnsureBoundaryPhiCache(int shift, int ngh, int ncells,
       int nx1_cells, int nx2_cells, int nx3_cells);
  // on_host has to come from the caller rather than from mgroot_->on_host_, so that the
  // plane is filled on the space the sweep that reads it will run on.
  const MGRootPhiCache &EnsureRootPhiCache(int ll, bool on_host, int nx, int ny, int nz,
       int ngh, Real dx1, Real dx2, Real dx3);
  void TransferFromBlocksToRoot(bool initflag);
  void TransferFromRootToBlocks(bool folddata);
  void TransferFromRootToBlocksDistributed(bool folddata);
  std::uint64_t GetMeshSignature() const { return mesh_sig_; }
  // Coarse/fine face ghost fill on the MeshBlock levels: keep (true, the historical
  // scheme) or drop (false) the tangential-gradient terms of the coarse value that the
  // fine-side ghost sees.  The terms are not reciprocated by the coarse cell's equation,
  // so they make the composite operator non-symmetric; dropping them gives a
  // flux-matched, symmetric interface at the price of piecewise-constant tangential
  // interpolation.  Set from <gravity>/mg_fc_symmetric in MGGravityDriver.
  bool FCTangentialGradient() const { return fc_tangential_gradient_; }
  bool fc_tangential_gradient_ = true;

  // per-cell octet operations
  void InitializeOctets();
  void SmoothOctets(int color);
  void RestrictOctets();
  void ProlongateAndCorrectOctets();
  void FMGProlongateOctets();
  void SetBoundariesOctets(bool fprolong, bool folddata);
  void ApplyPhysicalBoundariesOctet(MGOctet &oct, bool folddata);
  void ComputeOctetInteriorAverages(int lev, bool folddata);
  void SetOctetCoarseBufferPhysicalBoundaries(const MGOctet &oct, int oct_id,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold, bool folddata);
  // homogeneous=true when the array holds a CORRECTION rather than the solution: the
  // correction must satisfy the homogeneous form of every boundary condition, so an
  // inhomogeneous Dirichlet value (the multipole potential) must not be injected.
  void ApplyPhysicalBoundariesBlocks(bool fill_tangential_ghosts = false,
                                     bool homogeneous = false);
  void ProlongateOctetBoundariesFluxCons(MGOctet &oct,
       std::vector<Real> &cbuf, int nvar, const std::array<bool,27> &ncoarse);
  void ProlongateOctetBoundaries(MGOctet &oct, int oct_id,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, const std::array<bool,27> &ncoarse, bool folddata);
  void StoreOldDataOctets();
  void CalculateFASRHSOctets();
  void ZeroClearOctets();
  void RestrictFMGSourceOctets();
  void RestrictOctetsBeforeTransfer();
  void SetOctetBoundariesBeforeTransfer(bool folddata);
  void SetOctetBoundarySameLevel(MGOctet &dst, const MGOctet &src, int src_id,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int ox1, int ox2, int ox3, bool folddata);
  void SetOctetBoundaryFromCoarser(const Real *un, const Real *unold,
       std::vector<Real> &cbuf, std::vector<Real> &cbufold,
       int nvar, int un_nx, int un_ny, int un_nz, const LogicalLocation &loc,
       int ox1, int ox2, int ox3, bool folddata);

  // Physics-dependent octet operations (virtual, overridden in derived drivers).  The
  // loop over the level's octets lives inside the override, so a level costs one
  // virtual dispatch rather than one per octet -- and the grid spacing and stencil
  // weights, which are the same for every octet of a level, are computed once.
  virtual void SmoothOctetLevel(MGOctet *octs, int noct, int rlev, int color) = 0;
  virtual void CalculateDefectOctetLevel(MGOctet *octs, int noct, int rlev) = 0;
  virtual void CalculateFASRHSOctetLevel(MGOctet *octs, int noct, int rlev) = 0;
  virtual bool SupportsDistributedCoarseSync() const { return false; }
  virtual void SyncDistributedCoarseSolution(bool folddata) {}

  DualArray2D<Real> rootbuf_;
  DualArray2D<Real> rootsrcbuf_;

  friend class Multigrid;

 protected:
  void SubtractAverage(MGVariable type);
  void SetupMultigrid(Real dt, bool ftrivial);
  void FMGProlongate(Driver *pdriver);
  void OneStepToFiner(Driver *pdriver,int nsmooth);
  void OneStepToCoarser(Driver *pdriver,int nsmooth);
  void SolveVCycle(Driver *pdriver,int npresmooth, int npostsmooth);
  void SolveFMG(Driver *pdriver);
  void SolveMG(Driver *pdriver);
  void SolveFMGCoarser();
  void SolveIterative(Driver *pdriver);
  void SolveIterativeFixedTimes(Driver *pdriver);
  void ResetSolveCounters();
  TaskStatus SendBoundary(Driver *pdrive, int stag);
  TaskStatus RecvBoundary(Driver *pdrive, int stag);
  TaskStatus StartReceive(Driver *pdrive, int stag);
  TaskStatus SendBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus RecvBoundaryForProlongation(Driver *pdrive, int stag);
  TaskStatus StartReceiveForProlongation(Driver *pdrive, int stag);
  TaskStatus SmoothRed(Driver *pdrive, int stag);
  TaskStatus SmoothBlack(Driver *pdrive, int stag);
  TaskStatus PhysicalBoundary(Driver *pdrive, int stag);
  TaskStatus PhysicalBoundaryForProlongation(Driver *pdrive, int stag);
  // Same, but for the up-leg where the array holds a CORRECTION and every boundary
  // condition must be applied in its homogeneous form.
  TaskStatus PhysicalBoundaryForCorrectionProlongation(Driver *pdrive, int stage);
  TaskStatus Restrict(Driver *pdrive, int stag);
  TaskStatus Prolongate(Driver *pdrive, int stag);
  TaskStatus PrepareCorrection(Driver *pdrive, int stag);
  TaskStatus ProlongatePreparedCorrection(Driver *pdrive, int stag);
  TaskStatus FMGProlongateTask(Driver *pdrive, int stag);
  TaskStatus FillFCBoundary(Driver *pdrive, int stag);
  TaskStatus FillFCBoundaryForProlongation(Driver *pdrive, int stag);
  bool MGBoundaryOverridesPeriodicMesh() const;
  TaskStatus CalculateFASRHS(Driver *pdrive, int stag);
  TaskStatus ClearRecv(Driver *pdrive, int stag);
  TaskStatus ClearSend(Driver *pdrive, int stag);
  void SetMGTaskListToFiner(int nsmooth, int ngh, int flag=0,
                           bool reduce_same_exchange=false);
  void SetMGTaskListFMGProlongate(int ngh);
  void SetMGTaskListToCoarser(int nsmooth, int ngh,
                              bool reduce_same_exchange=false);
  bool UseReducedSameExchangeOnCurrentLevel(bool to_finer, int flag=0) const;
  bool UseExactHaloPair(int nsmooth, int level_shift) const;
  bool UseDistributedCoarseSolve() const {
    return distributed_coarse_solve_ && nranks_ > 1;
  }
  bool IsCoarseSolveOwner() const {
    return (!UseDistributedCoarseSolve() ||
            global_variable::my_rank == coarse_owner_rank_);
  }

  virtual void SolveCoarsestGrid();
  Real CalculateDefectNorm(MGNormType nrm, int n);
  Real CalculateArrayNorm(MGVariable type, MGNormType nrm, int n);
  // container to hold names of TaskIDs
  MultigridTaskIDs id;

  // small functions
  int GetNumMultigrids() { return nblist_[global_variable::my_rank]; }

  int nranks_, nbtotal_, nvar_;
  int locrootlevel_, nrootlevel_, nmblevel_, ntotallevel_, nreflevel_, maxreflevel_;
  int current_level_, fmglevel_;
  bool finest_bvals_fresh_;
  bool skip_coarser_initial_exchange_once_;
  bool active_reduce_same_exchange_;
  bool mg_tl_to_finer_valid_;
  bool mg_tl_to_coarser_valid_;
  bool mg_tl_fmg_prolongate_valid_;
  int mg_tl_to_finer_nsmooth_;
  int mg_tl_to_finer_flag_;
  bool mg_tl_to_finer_reduce_same_;
  int mg_tl_to_coarser_nsmooth_;
  bool mg_tl_to_coarser_skip_initial_;
  bool mg_tl_to_coarser_reduce_same_;
  int *nslist_, *nblist_, *nvlist_, *nvslist_, *nvlisti_, *nvslisti_, *ranklist_;
  int nrbx1_, nrbx2_, nrbx3_;
  BoundaryFlag mg_mesh_bcs_[6];
  Mesh *pmy_mesh_;
  MeshBlockPack *pmy_pack_;
  Multigrid *mgroot_;
  Multigrid *mglevels_;
  Multigrid *pmg;
  bool fsubtract_average_, needinit_, amr_mesh_changed_;
  std::uint64_t mesh_sig_;
  std::uint64_t mesh_topology_version_;
  Real eps_;
  Real last_defect_norm_;
  bool last_defect_valid_;
  //! Mesh cycle of the last "solve ended far from target" warning, so a run that
  //! never converges says so without printing on every stage of every cycle.
  int unconverged_warn_cycle_;
  int niter_, npresmooth_, npostsmooth_;
  int coffset_;
  int finest_exact_polish_passes_;
  int same_exchange_stride_;
  int local_sweeps_per_exchange_;
  int reduced_exchange_max_edge_;
  int fprolongation_;
  int fshowdef_;
  int mg_verbose_ = 0;
  bool full_multigrid_;
  int fmg_ncycle_;
  int defect_check_interval_;
  // Dimensionless convergence floor for the threshold == 0 policy.  The defect carries
  // the units of the source, so a floor only means the same thing on two different
  // problems if it is expressed as a fraction of the source norm.
  Real auto_target_rtol_;
  Real source_norm_;
  // The stopping target of the threshold == 0 policy: a fraction of the source norm,
  // floored so that a source-free problem still has a finite target to reach.
  Real AutoTargetDefect() const {
    return std::max(auto_target_rtol_ * source_norm_, static_cast<Real>(1.0e-12));
  }
  int auto_max_extra_cycles_;
  int coarsest_min_sweeps_;
  int timing_vcycles_;
  int timing_coarsest_solves_;
  int timing_defect_norms_;
  bool collect_phase_timing_;
  double timing_to_coarser_time_;
  double timing_to_finer_time_;
  double timing_coarsest_time_;
  double timing_fmg_coarser_time_;
  double timing_fmg_prolongate_time_;
  double timing_transfer_blocks_root_time_;
  double timing_transfer_root_blocks_time_;
  // The octet regime is host-serial work nested inside to_coarser/to_finer and inside
  // the transfers, so it is invisible in the phase breakdown without an accumulator of
  // its own.  It is not fenced: the work is host-side, and the few root-grid copies it
  // enqueues on the way out are asynchronous, so a fence would only charge it with
  // whatever the meshblock levels left running.
  double timing_octets_time_;
  bool finest_source_average_subtracted_;
  bool distributed_coarse_solve_;
  int coarse_owner_rank_;

  // Isolated-boundary multipole expansion (up to hexadecapole order).
  int mporder_ = -1;
  int nmpcoeff_ = 0;
  Real mpcoeff_[25] = {};
  DvceArray1D<Real> d_mpcoeff_;
  // Scratch for the two per-block partial sums and for the staging of mpcoeff_ on its
  // way to the device.  These are members rather than locals because both reductions and
  // the push run on every solve, and a fresh Kokkos::View there means a device malloc and
  // a device free per solve; a free in particular synchronizes the whole device.
  HostArray1D<Real> h_mpcoeff_;
  DvceArray2D<Real> mp_partial_;
  HostArray2D<Real> mp_partial_h_;
  // One row of coefficients per (block, x3 slice), so the cell loop can be spread over
  // nmb*nx3 threads instead of nmb.  The per-slice sums are folded into mp_partial_ by a
  // second, tiny kernel.
  DvceArray3D<Real> mp_kpartial_;
  HostArray3D<Real> mp_kpartial_h_;
  Real mpo_[3] = {};
  bool autompo_ = false;
  bool nodipole_ = false;
  // Bumped whenever the coefficients or the expansion origin reach the device, which is
  // once per solve.  Everything downstream of them is constant in between.
  std::uint64_t mp_epoch_ = 0;

  std::vector<MGBoundaryPhiCache> mgbc_phi_;
  std::vector<MGRootPhiCache> mgroot_phi_;

  // per-cell octets
  std::vector<MGOctet> *octets_;
  std::vector<Real> *octet_pool_;  // one contiguous backing store per level
  std::size_t *octet_pool_stride_;  // Reals per field run in that level's pool
  std::unordered_map<LogicalLocation, int, LogicalLocationHash> *octetmap_;
  std::vector<bool> *octetbflag_;
  std::vector<int> *octet_nb_same_;
  std::vector<int> *octet_nb_coarse_;
  std::vector<LogicalLocation> *octet_nb_rootloc_;
  std::vector<int> *octet_parent_;
  std::vector<int> *octet_child_i_;
  std::vector<int> *octet_child_j_;
  std::vector<int> *octet_child_k_;
  std::vector<int> *octet_local_ids_;
  int *noctets_;
  std::vector<int> octet_root_i_, octet_root_j_, octet_root_k_;
  std::vector<int> mb_octet_lev_, mb_octet_oid_, mb_octet_i_, mb_octet_j_, mb_octet_k_;
  std::vector<Real> cbuf_, cbufold_;  // scratch buffers for boundary exchange
  std::array<bool,27> ncoarse_;       // 3x3x3 flags for coarser neighbors
  // Volume average of the eight interior cells of each octet of one level, indexed
  // o*nvar_ + v.  A boundary pass writes only ghosts and off-centre coarse-buffer slots,
  // so an octet's interior -- and therefore its average -- is fixed for the whole pass.
  std::vector<Real> octet_avg_, octet_avgold_;

  std::vector<Real> root_u_buf_, root_uold_buf_;
  // The flat root buffers are a byte copy of the host root view, so they carry its three
  // extents rather than one cube edge.  A root grid is one cell per root-level MeshBlock
  // and need not be cubic.
  int root_buf_nx_, root_buf_ny_, root_buf_nz_;
  bool root_flat_buf_stale_;
  bool root_uold_buf_valid_;
  // Set for the span of a root->block transfer, during which the host copy of the root
  // arrays is the only one being written and read.  SyncRootDataToHost() then has
  // nothing to fetch, and pulling anyway would overwrite the octet restriction the
  // host has just done -- which is why the old code had to push to the device first.
  bool root_host_authoritative_;
  std::vector<Real> coarse_scatter_send_buf_, coarse_scatter_recv_buf_;
  std::vector<int> coarse_scatter_counts_, coarse_scatter_displs_;
  std::uint64_t coarse_scatter_mesh_sig_;
  int coarse_scatter_block_count_;
  bool coarse_scatter_folddata_;
  void BuildRootFlatBuffers(bool include_old);
  void RefreshRootFlatBuffersFromHost();
  void SyncRootDataToHost(bool include_old);
  void SyncRootDataToDevice();
  void SyncRootOldDataToDevice();
  void SyncRootSourceToDevice();
  void SyncRootToHost();
  void SyncRootToDevice();

 private:
  int nb_rank_;
};

class MultigridBoundaryValues : public MeshBoundaryValuesCC {
 public:
  MultigridBoundaryValues(MeshBlockPack *pmbp, ParameterInput *pin, bool coarse,
      Multigrid *pmg);
  ~MultigridBoundaryValues();

  void RemapIndicesForMG();
  void RefreshForNewTopology();

  // pack/restrict fluxes at fine/coarse boundaries into boundary buffers and send
  TaskStatus PackAndSendMG(const DvceArray5D<Real> &u, bool include_diagonals = false);
  TaskStatus RecvAndUnpackMG(DvceArray5D<Real> &u, bool include_diagonals = false);
  TaskStatus InitRecvMG(const int nvars, bool include_diagonals = false);
  TaskStatus FillFineCoarseMGGhosts(DvceArray5D<Real> &u,
                                    bool include_diagonals = false);
  TaskStatus ClearRecvMG();
  TaskStatus ClearSendMG();
  // A 7-point sweep reads one ghost layer, so the same-level exchange normally carries
  // exactly one.  The exact communication-avoiding pair sweeps one layer into the halo
  // and reads two, and asks for the deeper exchange for the calls that feed it.
  void SetMGSameHaloDepth(int depth) {
    const int ngh = pmy_mg->GetGhostCells();
    mg_same_halo_depth_ = (depth < 1) ? 1 : ((depth > ngh) ? ngh : depth);
  }
#if MPI_PARALLEL_ENABLED
  void WaitMGSameLevelSendRequests();
  void WaitMGSameLevelRecvRequests();
  void FreeMGSameLevelPersistentRequests();
  void FreeFCRemotePersistentRequests();
#endif

 private:
  struct MGSameEntry {
    int rank;
    int m;
    int n;
    int tag;
    int size;
  };
  // Receive storages whose device consumer is tracked on its own: the two aggregated
  // same-level payload slots, indexed by the slot number, and the fine/coarse shell.
  enum MGRecvStore {kMGFCShell = 2, kMGNumRecvStores = 3};
#if MPI_PARALLEL_ENABLED
  // Owns one level/mode's aggregate metadata, payload storage, and persistent requests.
  // States are moved, never copied, so every MPI_Request has exactly one owner.
  struct MGSameLevelAggState {
    bool valid = false;
    int nvars = -1;
    DvceArray1D<int> send_m_d, send_off_d;
    DvceArray1D<int> recv_m_d, recv_off_d;
    DvceArray2D<int> send_bounds_d, recv_bounds_d;
    std::array<DvceArray1D<Real>, 2> send_data_d;
    std::array<DvceArray1D<Real>, 2> recv_data_d;
    std::vector<int> send_rank_h, send_data_off_h, send_tag_h;
    std::vector<int> recv_rank_h, recv_data_off_h, recv_tag_h;
    std::array<std::vector<MPI_Request>, 2> send_req_h;
    std::array<std::vector<MPI_Request>, 2> recv_req_h;
    std::array<bool, 2> send_slot_inflight{false, false};
    bool send_inflight = false;
    bool recv_inflight = false;
    bool persistent_ready = false;
    bool has_remote = false;
    int send_poll_count = 0;
    int recv_poll_count = 0;
    int send_pack_slot = 0;
    int recv_post_slot = 0;
    int recv_data_slot = 0;

    MGSameLevelAggState() = default;
    MGSameLevelAggState(const MGSameLevelAggState &) = delete;
    MGSameLevelAggState &operator=(const MGSameLevelAggState &) = delete;
    MGSameLevelAggState(MGSameLevelAggState &&) = default;
    MGSameLevelAggState &operator=(MGSameLevelAggState &&) = default;
  };
#endif
  // Compact-shell index maps for the remote fine/coarse exchange.  These are a pure
  // function of (ngh, ncells, axis_mask), i.e. of the multigrid level shift, and carry
  // no topology information, so one map is cached per shift.  Rebuilding them costs an
  // O(ni_tot^3) host loop, which must not run on every V-cycle level change.
  struct FCRemoteCompactMap {
    DvceArray1D<int> cmp2full_d;
    DvceArray1D<int> full2cmp_d;
    bool valid = false;
    int ngh = -1;
    int ncells = -1;
    int ni_tot = -1;
    int nj_tot = -1;
    int nk_tot = -1;
    int axis_mask = -1;
    int cell_count = 0;
  };
#if MPI_PARALLEL_ENABLED
  // One persistent MPI request set per multigrid level shift.  Each set records the
  // buffer base pointers, the rank grouping (rbeg/rcount) and the block size it was
  // built from.  If ANY of those change the set must be freed and rebuilt before it is
  // started again: a persistent request left bound to a stale device pointer does not
  // fault, it silently corrupts memory.  States are moved, never copied, so every
  // MPI_Request has exactly one owner.
  struct FCRemotePersistentSet {
    bool ready = false;
    const Real *recv_base = nullptr;
    std::array<const Real *, 2> send_base{{nullptr, nullptr}};
    int block_size = -1;
    std::vector<int> recv_rank_list_h, recv_rank_off_h;
    std::vector<int> send_rank_list_h, send_rank_off_h;
    std::vector<MPI_Request> recv_req_h;
    std::array<std::vector<MPI_Request>, 2> send_req_h;

    FCRemotePersistentSet() = default;
    FCRemotePersistentSet(const FCRemotePersistentSet &) = delete;
    FCRemotePersistentSet &operator=(const FCRemotePersistentSet &) = delete;
    FCRemotePersistentSet(FCRemotePersistentSet &&) = default;
    FCRemotePersistentSet &operator=(FCRemotePersistentSet &&) = default;
  };
#endif
  bool UseMGSameLevelAgg() const;
  void BuildMGSameLevelAggCache(int nvars, bool include_diagonals);
  void EnsureMGSameLevelAggReady(int nvars, bool include_diagonals);
  //! \brief note that an unpack kernel is in flight over receive storage `store`
  void MarkMGUnpackPending(int store);
  //! \brief wait for the unpacks of `store`, and only those, to retire
  void WaitMGUnpackComplete(int store);
  //! \brief wait for every pending unpack, for the paths that move the buffers themselves
  void WaitMGUnpackCompleteAll();
  //! \brief refresh the cached fine/coarse topology scan if the mesh has changed
  void EnsureFCTopologyScan(std::uint64_t mesh_sig, int nmb, int nnghbr);
  //! \brief drop everything cached about the MeshBlock topology
  void InvalidateMGTopologyCaches();
#if MPI_PARALLEL_ENABLED
  void SwapMGSameLevelAggState(MGSameLevelAggState &state);
  void FreeMGSameLevelAggState(MGSameLevelAggState &state);
  void InvalidateMGSameLevelAggCache();
  FCRemotePersistentSet *FCRemoteSlotSet(int slot);
  void FreeFCRemotePersistentSet(int shift);
  void WaitFCRemoteSendSlots();
#endif
  // data
  Multigrid *pmy_mg;
  DvceArray2D<int> fc_remote_ridx_d_;
  // Remote fine/coarse payload buffers.  The layout is FLAT and entry-major: entry e of
  // the current level occupies [e*remote_block_size, (e+1)*remote_block_size), so MPI
  // offsets/counts stay compact per level and no padding is ever transmitted.  The
  // allocation is sized from the FINEST level and only ever grows, because the
  // persistent requests above are bound to .data() and any reallocation moves it.
  DvceArray1D<Real> fc_recv_data_d_{"mg_fc_recv_data", 0};
  std::array<DvceArray1D<Real>, 2> fc_send_data_d_{
      DvceArray1D<Real>("mg_fc_send_data0", 0),
      DvceArray1D<Real>("mg_fc_send_data1", 0)};
  std::vector<FCRemoteCompactMap> fc_remote_compact_maps_;
  // Host staging for the fine/coarse edge and corner ghosts, which still travel through
  // host buffers.  It is kept across calls: allocating a mirror of the whole level array
  // on every prolongation step costs a fresh first touch of every page it holds.
  HostArray5D<Real> fc_diag_host_mirror_;
  DvceArray1D<int> fc_send_m_d_;
  DvceArray1D<int> fc_ngh_ox1_d_, fc_ngh_ox2_d_, fc_ngh_ox3_d_;
  DvceArray1D<int> fc_ngh_f1_d_, fc_ngh_f2_d_, fc_ngh_nface_d_, fc_ngh_valid_d_;
  std::vector<int> fc_ngh_ox1_h_, fc_ngh_ox2_h_, fc_ngh_ox3_h_, fc_ngh_nface_h_;
  std::vector<int> fc_recv_gid_h_, fc_recv_rank_h_;
  std::vector<int> fc_send_m_h_, fc_send_gid_h_, fc_send_rank_h_;
  std::vector<int> fc_recv_rank_list_h_, fc_recv_rank_off_h_;
  std::vector<int> fc_send_rank_list_h_, fc_send_rank_off_h_;
  // Which fine/coarse neighbours this rank has at all.  This is a property of the
  // MeshBlock tree, not of the multigrid level, so it changes only when the mesh does;
  // recomputing it meant two nmb*nnghbr host sweeps on every prolongation step.
  // Per block, bit 0 of fc_scan_block_mask_h_ says the block has an off-rank fine/coarse
  // neighbour and bit 1 says one of those is an edge or a corner -- the two questions the
  // host-staged diagonal path asks before it mirrors a block.
  std::vector<int> fc_scan_block_mask_h_;
  std::uint64_t fc_scan_mesh_sig_;
  int fc_scan_nmb_;
  int fc_scan_nnghbr_;
  bool fc_scan_valid_;
  bool fc_scan_has_local_;
  bool fc_scan_has_remote_;
  bool fc_scan_has_remote_face_;
  bool fc_scan_has_remote_diag_;
  std::uint64_t fc_mesh_sig_;
  int fc_ngh_map_nnghbr_;
  bool fc_ngh_map_valid_;
  bool fc_remote_cache_valid_;
  bool fc_remote_axis_mask_valid_;
  std::uint64_t fc_remote_axis_mask_mesh_sig_;
  int fc_remote_axis_mask_nnghbr_;
  int fc_remote_compact_axis_mask_;
  std::array<bool, 2> fc_remote_send_slot_inflight_;
  // Level shift whose persistent request set owns each in-flight send slot, -1 if none.
  std::array<int, 2> fc_remote_send_slot_shift_;
  bool fc_remote_send_inflight_;
#if MPI_PARALLEL_ENABLED
  std::vector<FCRemotePersistentSet> fc_remote_persistent_;
#endif
  int fc_remote_send_pack_slot_;
  DvceArray1D<int> mg_same_send_m_d_, mg_same_send_off_d_;
  DvceArray1D<int> mg_same_recv_m_d_, mg_same_recv_off_d_;
  DvceArray2D<int> mg_same_send_bounds_d_, mg_same_recv_bounds_d_;
  std::array<DvceArray1D<Real>, 2> mg_same_send_data_d_;
  // Two receive slots, for the same reason the send side has two: MPI must not write a
  // payload the unpack kernel of the previous exchange is still reading.  With one slot
  // that is a host wait on that kernel at the top of every InitRecvMG(); alternating
  // slots pushes the constraint back to the exchange before last, which has had a whole
  // round of device work queued behind it and is in practice already retired.  The
  // persistent requests are duplicated per slot -- MPI_Recv_init binds an address, so
  // one set per buffer -- and only ever one set is started at a time, so the two sets
  // cannot compete for the same incoming message.
  std::array<DvceArray1D<Real>, 2> mg_same_recv_data_d_;
  std::vector<int> mg_same_send_rank_h_;
  std::vector<int> mg_same_send_data_off_h_, mg_same_send_tag_h_;
  std::vector<int> mg_same_recv_rank_h_;
  std::vector<int> mg_same_recv_data_off_h_, mg_same_recv_tag_h_;
#if MPI_PARALLEL_ENABLED
  std::array<std::vector<MPI_Request>, 2> mg_same_send_req_h_;
  std::array<std::vector<MPI_Request>, 2> mg_same_recv_req_h_;
#endif
  std::uint64_t mg_same_mesh_sig_;
  int mg_same_halo_depth_ = 1;
  int mg_same_shift_;
  int mg_same_nvars_;
  bool mg_same_include_diagonals_;
  bool mg_same_cache_valid_;
  std::array<bool, 2> mg_same_send_slot_inflight_;
  bool mg_same_send_inflight_;
  bool mg_same_recv_inflight_;
  bool mg_same_persistent_ready_;
  bool mg_same_has_remote_;
  int mg_same_poll_budget_;
  int mg_same_send_poll_count_;
  int mg_same_recv_poll_count_;
  int mg_same_send_pack_slot_;
  // Receive slot the next MPI_Startall posts into, and the slot holding the payload the
  // next unpack consumes.  At most one receive is ever started, so one of each suffices.
  int mg_same_recv_post_slot_;
  int mg_same_recv_data_slot_;
  // Raised while an unpack kernel may still be reading that storage's payload.
  std::array<bool, kMGNumRecvStores> mg_unpack_pending_;
#if defined(KOKKOS_ENABLE_CUDA)
  // Recorded on the Kokkos stream immediately after the unpack that reads the storage,
  // so waiting on it releases the buffer as soon as that kernel retires and leaves the
  // smoother and boundary kernels queued behind it running.  A plain device fence would
  // drain those too, and a receive that is not posted cannot start its rendezvous, so
  // nothing on the wire could overlap them.  An event carries no identity beyond "the
  // most recent unpack of this storage", which is all the buffer needs: the stream is
  // in-order, so any earlier unpack of the same storage -- including one from a multigrid
  // level whose state has since been swapped out -- has retired by then as well.
  std::array<cudaEvent_t, kMGNumRecvStores> mg_unpack_event_;
  std::array<bool, kMGNumRecvStores> mg_unpack_event_valid_;
#endif
#if MPI_PARALLEL_ENABLED
  std::vector<MGSameLevelAggState> mg_same_level_cache_;
  std::uint64_t mg_same_level_cache_mesh_sig_ = 0;
  int mg_same_active_cache_key_ = -1;
#endif
  std::vector<DvceArray1D<int>> mg_single_pair_m_cache_d_, mg_single_pair_sm_cache_d_;
  std::vector<DvceArray2D<int>> mg_single_pair_desc_cache_d_;
  std::vector<int> mg_single_pair_count_cache_;
  std::vector<int> mg_single_cached_nmb_cache_, mg_single_cached_nnghbr_cache_;
  std::vector<int> mg_single_cached_mbgid0_cache_, mg_single_cached_nx1_cache_;
  std::vector<std::uint64_t> mg_single_cached_mesh_sig_cache_;
  std::vector<int> mg_single_pair_valid_cache_;
  // functions
};

template <typename ViewType, typename StencilOp>
void Multigrid::Smooth(ViewType &u, const ViewType &src, const StencilOp &stencil,
                       int rlev, int il, int iu, int jl, int ju, int kl, int ku,
                       int color) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [this]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
      return block_rdx_.h_view;
    else
      return block_rdx_.d_view;
  }();
  auto block_parity = [this]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>)
      return block_color_parity_.h_view;
    else
      return block_color_parity_.d_view;
  }();
  int rlev_l = rlev;
  Real odiag = stencil.omega_over_diag;
  color ^= pmy_driver_->GetCoffset();
  const int parity_origin = ngh_;
  const int level_shift = (rlev_l < 0) ? -rlev_l : 0;
  const int block_cells_odd = (indcs_.nx1 >> level_shift) & 1;
  // One thread per cell of the colour, with the x index the fastest-running one, so a
  // warp walks 2*32 consecutive cells of a row instead of 32 different rows.  The
  // neighbours a red cell reads are the black cells in between, so every cache line the
  // warp pulls in is used whole.  Cells of one colour never depend on each other, so
  // the update is the same arithmetic in a different order: bitwise identical.
  const int nhalf = (iu - il)/2 + 1;
  par_for("Multigrid::Smooth", ExeSpace(), 0, nmmb_-1, kl, ku, jl, ju, 0, nhalf-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int ih) {
    const int block_color = color ^ (block_cells_odd & block_parity(m));
    const int c = (block_color + k + j) & 1;
    const int i = il + (((il - parity_origin) & 1) ^ c) + 2*ih;
    if (i > iu) return;
    Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev_l))
                            : brdx(m) / static_cast<Real>(1<<rlev_l);
    Real dx2 = dx * dx;
    Real lap = stencil.Apply(u, m, 0, k, j, i);
    u(m,0,k,j,i) -= (lap - src(m,0,k,j,i)*dx2) * odiag;
  });
}

inline Real RestrictOne(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.U(v, fk,   fj,   fi)   + oct.U(v, fk,   fj,   fi+1)
               +oct.U(v, fk,   fj+1, fi)   + oct.U(v, fk,   fj+1, fi+1)
               +oct.U(v, fk+1, fj,   fi)   + oct.U(v, fk+1, fj,   fi+1)
               +oct.U(v, fk+1, fj+1, fi)   + oct.U(v, fk+1, fj+1, fi+1));
}

inline Real RestrictOneSrc(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.Src(v, fk,   fj,   fi)   + oct.Src(v, fk,   fj,   fi+1)
               +oct.Src(v, fk,   fj+1, fi)   + oct.Src(v, fk,   fj+1, fi+1)
               +oct.Src(v, fk+1, fj,   fi)   + oct.Src(v, fk+1, fj,   fi+1)
               +oct.Src(v, fk+1, fj+1, fi)   + oct.Src(v, fk+1, fj+1, fi+1));
}

inline Real RestrictOneDef(const MGOctet &oct, int v, int fi, int fj, int fk) {
  return 0.125*(oct.Def(v, fk,   fj,   fi)   + oct.Def(v, fk,   fj,   fi+1)
               +oct.Def(v, fk,   fj+1, fi)   + oct.Def(v, fk,   fj+1, fi+1)
               +oct.Def(v, fk+1, fj,   fi)   + oct.Def(v, fk+1, fj,   fi+1)
               +oct.Def(v, fk+1, fj+1, fi)   + oct.Def(v, fk+1, fj+1, fi+1));
}

// access flat buffer of size (nvar, nc, nc, nc)
inline Real& BufRef(std::vector<Real> &buf, int nc, int v, int k, int j, int i) {
  return buf[((v*nc + k)*nc + j)*nc + i];
}
inline const Real& BufRef(const std::vector<Real> &buf, int nc,
                          int v, int k, int j, int i) {
  return buf[((v*nc + k)*nc + j)*nc + i];
}

KOKKOS_INLINE_FUNCTION
Real EvalMultipolePhi(Real x, Real y, Real z, const Real *mpc, int order) {
  Real x2 = x*x, y2 = y*y, z2 = z*z;
  Real xy = x*y, yz = y*z, zx = z*x;
  Real r2 = x2 + y2 + z2;
  if (r2 == 0.0) return 0.0;
  Real ir2 = 1.0/r2, ir1 = Kokkos::sqrt(ir2);
  Real ir3 = ir2*ir1, ir5 = ir3*ir2;
  Real hx2my2 = 0.5*(x2-y2);
  Real phis = ir1*mpc[0]
    + ir3*(mpc[1]*y + mpc[2]*z + mpc[3]*x)
    + ir5*(mpc[4]*xy + mpc[5]*yz + (3.0*z2-r2)*mpc[6]
         + mpc[7]*zx + mpc[8]*hx2my2);
  if (order == 4) {
    Real ir7 = ir5*ir2, ir9 = ir7*ir2;
    Real x2mty2 = x2-3.0*y2;
    Real tx2my2 = 3.0*x2-y2;
    phis += ir7*(y*tx2my2*mpc[9] + x*x2mty2*mpc[15]
               + xy*z*mpc[10] + z*hx2my2*mpc[14]
               + (5.0*z2-r2)*(y*mpc[11] + x*mpc[13])
               + z*(z2-3.0*r2)*mpc[12])
         + ir9*(xy*hx2my2*mpc[16]
               + 0.125*(x2*x2mty2-y2*tx2my2)*mpc[24]
               + yz*tx2my2*mpc[17] + zx*x2mty2*mpc[23]
               + (7.0*z2-r2)*(xy*mpc[18] + hx2my2*mpc[22])
               + (7.0*z2-3.0*r2)*(yz*mpc[19] + zx*mpc[21])
               + (35.0*z2*z2-30.0*z2*r2+3.0*r2*r2)*mpc[20]);
  }
  return phis;
}


#endif // MULTIGRID_MULTIGRID_HPP_
