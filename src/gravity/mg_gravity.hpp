#ifndef GRAVITY_MG_GRAVITY_HPP_
#define GRAVITY_MG_GRAVITY_HPP_
//========================================================================================
// Athenak astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// C++ headers
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
class MultigridDriver;

// 7-point Cartesian Laplacian scaled by dx^2.  Keeping the isotropic expression
// separate preserves its arithmetic and avoids unnecessary multiplies in the common case.
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real ApplyGravityLaplacian(const ViewType &u, int m, int v, int k, int j, int i,
                           Real wy, Real wz, Real diag) {
  if (wy == 1.0 && wz == 1.0) {
    return 6.0*u(m,v,k,j,i) - u(m,v,k+1,j,i) - u(m,v,k,j+1,i)
           - u(m,v,k,j,i+1) - u(m,v,k-1,j,i) - u(m,v,k,j-1,i)
           - u(m,v,k,j,i-1);
  }
  return diag*u(m,v,k,j,i) - u(m,v,k,j,i+1) - u(m,v,k,j,i-1)
         - wy*(u(m,v,k,j+1,i) + u(m,v,k,j-1,i))
         - wz*(u(m,v,k+1,j,i) + u(m,v,k-1,j,i));
}

struct GravityStencil {
  Real omega_over_diag;
  Real wy;
  Real wz;
  Real diag;

  template <typename ViewType>
  KOKKOS_INLINE_FUNCTION
  Real Apply(const ViewType &u, int m, int v, int k, int j, int i) const {
    return ApplyGravityLaplacian(u, m, v, k, j, i, wy, wz, diag);
  }
};

//! \class MGGravity
//! \brief Multigrid gravity solver for each block

class MGGravity : public Multigrid {
 public:
  MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
            bool on_host = false);
  ~MGGravity();

  void SmoothPackBox(int color, int is, int ie, int js, int je, int ks, int ke);
  void SmoothPack(int color) final;
  void SmoothPackCoarsestEffectiveDiagonal(int color, const Real *bface);
  void SmoothPackExpanded(int color, int grow);
  void SmoothPackInterior(int color);
  void SmoothPackBoundary(int color);
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;
};


//! \class MGGravityDriver
//! \brief Multigrid gravity solver

class MGGravityDriver : public MultigridDriver {
 public:
  MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin);
  ~MGGravityDriver();

  void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;
  bool SolveDueForCycle(int ncycle, Real time) const;
  bool SolveDueAtTime(Real time) const;
  Real SolveInterval() const { return use_solve_dt_ ? solve_dt_ : 0.0; }
  int SolveEvery() const { return solve_every_; }
  // True when advancing to `time_end` would step strictly past the next scheduled solve
  // (landing exactly on it is fine: the solve is then due at the next window start).
  bool WindowCrossesSolveTime(Real time_end) const;
  // Pull the next scheduled solve forward to `time` (Solve() otherwise keeps a warm,
  // unchanged-mesh potential until next_solve_time_ and returns without solving).
  void MarkSolveDueAt(Real time) { next_solve_time_ = time; cadence_cycle_ = -1; }
  void ResetCadenceCache();
  void SetFourPiG(Real four_pi_G);
  // finest_holds_phi says the finest multigrid level already holds, cell for cell, the
  // active zone of the phi handed in -- true only for the call at the end of Solve().
  bool RefreshFinestGhosts(DvceArray5D<Real> &phi, int phi_nghost,
                           bool finest_holds_phi = false);

  // <gravity>/reciprocity_test: at the end of Driver::Initialize, solve for the initial
  // density (A) and for a deterministic modification of it (B), then form
  // S_AB = sum((rhoA-<rhoA>) phiB dV) and S_BA = sum((rhoB-<rhoB>) phiA dV) in double
  // precision.  They agree iff the composite discrete operator is symmetric.  Optionally
  // also applies the operator itself (ghost fill included) to two smooth fields u, v and
  // compares <v,Lu> with <u,Lv>.  Prints and returns; the caller ends the run.
  bool ReciprocityTestRequested() const { return reciprocity_test_; }
  void RunReciprocityTest(Driver *pdriver);

  // A root grid that coarsens to a single cell has no interior: every face is a physical
  // boundary, and the relaxation the base class would run there diverges.  See the
  // definition for the arithmetic.
  void SolveCoarsestGrid() final;
  void SolveCoarsestGridEffectiveDiagonal();

  // Octet physics (host-side).
  void SmoothOctetLevel(MGOctet *octs, int noct, int rlev, int color) final;
  void CalculateDefectOctetLevel(MGOctet *octs, int noct, int rlev) final;
  void CalculateFASRHSOctetLevel(MGOctet *octs, int noct, int rlev) final;
    bool SupportsDistributedCoarseSync() const final { return true; }
    void SyncDistributedCoarseSolution(bool folddata) final;

  friend class MGGravity;

 private:
    // Contiguous host staging buffer for the distributed-coarse-solve broadcast.
    std::vector<Real> oct_bcast_buf_;
    int ca_pack_smoother_type_ = 0;  // 0: rbgs, 1: jacobi, 2: chebyshev2
    Real ca_pack_cheb_lambda_min_ = 0.7;
    Real ca_pack_cheb_lambda_max_ = 1.95;
    Real rho_grav_min_ = 0.0;
    Real rho_grav_floor_ = 0.0;
    Real mask_radius_ = -1.0;
    Real mask_origin_[3] = {0.0, 0.0, 0.0};
    Real four_pi_G_, omega_;
    Real gravity_wy_ = 1.0, gravity_wz_ = 1.0, gravity_diag_ = 6.0;
    bool prefer_distributed_coarse_solve_ = false;
    int last_prepare_cycle_ = -1;
    std::uint64_t last_prepare_topology_ = std::numeric_limits<std::uint64_t>::max();
    int solve_every_ = 1;
    int cadence_cycle_ = -1;
    int warm_final_niter_ = 4;
    bool show_timing_ = false;
    bool use_solve_dt_ = false;
    bool cadence_due_this_cycle_ = true;
    Real solve_dt_ = 0.0;
    Real next_solve_time_ = 0.0;
    bool reciprocity_test_ = false;
    bool reciprocity_operator_test_ = true;
    Real reciprocity_mod_amp_ = 0.3;
    Real reciprocity_blob_amp_ = 0.5;
    Real reciprocity_blob_center_[3] = {3.0, 0.0, 0.0};
    Real reciprocity_blob_width_ = 2.0;
    std::string reciprocity_dump_;
};

#endif // GRAVITY_MG_GRAVITY_HPP_
