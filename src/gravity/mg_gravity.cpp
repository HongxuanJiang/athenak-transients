//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//! \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>
#include <limits>
#include <vector>

// Project headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../globals.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mhd/mhd.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../pgen/pgen.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../driver/driver.hpp"

class MeshBlockPack;

namespace {
BoundaryFlag ParseMGBoundaryFlag(const std::string &input) {
  if (input == "periodic") return BoundaryFlag::periodic;
  if (input == "zerograd" || input == "outflow") return BoundaryFlag::mg_zerograd;
  if (input == "zerofixed") return BoundaryFlag::mg_zerofixed;
  if (input == "multipole") return BoundaryFlag::mg_multipole;
  if (input == "none") return BoundaryFlag::undef;
  return BoundaryFlag::undef;
}

BoundaryFlag MapMeshToMGBoundary(BoundaryFlag input) {
  if (input == BoundaryFlag::periodic) return BoundaryFlag::periodic;
  if (input == BoundaryFlag::outflow) return BoundaryFlag::mg_zerograd;
  if (input == BoundaryFlag::mg_zerograd) return BoundaryFlag::mg_zerograd;
  if (input == BoundaryFlag::mg_zerofixed) return BoundaryFlag::mg_zerofixed;
  if (input == BoundaryFlag::mg_multipole) return BoundaryFlag::mg_multipole;
  return BoundaryFlag::mg_zerograd;
}

KOKKOS_INLINE_FUNCTION
Real ProlongationWeight(const int child_offset, const int stencil_offset) {
  if (stencil_offset == 0) return 30.0;
  if (child_offset == 0) return (stencil_offset < 0) ? 5.0 : -3.0;
  return (stencil_offset < 0) ? -3.0 : 5.0;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real ProlongationStencil27(const ViewType &u, const int m, const int k,
                           const int j, const int i, const int dk,
                           const int dj, const int di) {
  const Real wkm = ProlongationWeight(dk, -1);
  const Real wk0 = ProlongationWeight(dk, 0);
  const Real wkp = ProlongationWeight(dk, 1);
  const Real wjm = ProlongationWeight(dj, -1);
  const Real wj0 = ProlongationWeight(dj, 0);
  const Real wjp = ProlongationWeight(dj, 1);
  const Real wim = ProlongationWeight(di, -1);
  const Real wi0 = ProlongationWeight(di, 0);
  const Real wip = ProlongationWeight(di, 1);

  const Real km = wjm*(wim*u(m,0,k-1,j-1,i-1) + wi0*u(m,0,k-1,j-1,i) +
                       wip*u(m,0,k-1,j-1,i+1)) +
                  wj0*(wim*u(m,0,k-1,j,  i-1) + wi0*u(m,0,k-1,j,  i) +
                       wip*u(m,0,k-1,j,  i+1)) +
                  wjp*(wim*u(m,0,k-1,j+1,i-1) + wi0*u(m,0,k-1,j+1,i) +
                       wip*u(m,0,k-1,j+1,i+1));
  const Real k0 = wjm*(wim*u(m,0,k,  j-1,i-1) + wi0*u(m,0,k,  j-1,i) +
                       wip*u(m,0,k,  j-1,i+1)) +
                  wj0*(wim*u(m,0,k,  j,  i-1) + wi0*u(m,0,k,  j,  i) +
                       wip*u(m,0,k,  j,  i+1)) +
                  wjp*(wim*u(m,0,k,  j+1,i-1) + wi0*u(m,0,k,  j+1,i) +
                       wip*u(m,0,k,  j+1,i+1));
  const Real kp = wjm*(wim*u(m,0,k+1,j-1,i-1) + wi0*u(m,0,k+1,j-1,i) +
                       wip*u(m,0,k+1,j-1,i+1)) +
                  wj0*(wim*u(m,0,k+1,j,  i-1) + wi0*u(m,0,k+1,j,  i) +
                       wip*u(m,0,k+1,j,  i+1)) +
                  wjp*(wim*u(m,0,k+1,j+1,i-1) + wi0*u(m,0,k+1,j+1,i) +
                       wip*u(m,0,k+1,j+1,i+1));
  return wkm*km + wk0*k0 + wkp*kp;
}

template <typename ExeSpace, typename SrcView, typename SizeView>
void MaskBHSinkGravitySource(const SrcView &src, const SizeView &size,
                             const int nmb, const RegionIndcs &indcs,
                             const int ngh, const Real center_x,
                             const Real center_y, const Real center_z,
                             const Real radius) {
  const int is = ngh;
  const int js = ngh;
  const int ks = ngh;
  const int ie = is + indcs.nx1 - 1;
  const int je = js + indcs.nx2 - 1;
  const int ke = ks + indcs.nx3 - 1;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const Real radius2 = radius*radius;

  par_for("MGGravity::MaskBHSinkSource", ExeSpace(),
          0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - is, nx1, size(m).x1min, size(m).x1max);
    const Real y = CellCenterX(j - js, nx2, size(m).x2min, size(m).x2max);
    const Real z = CellCenterX(k - ks, nx3, size(m).x3min, size(m).x3max);
    const Real dx = x - center_x;
    const Real dy = y - center_y;
    const Real dz = z - center_z;
    if (dx*dx + dy*dy + dz*dz <= radius2) {
      src(m, 0, k, j, i) = 0.0;
    }
  });
}

void ApplyBHSinkGravityMask(MeshBlockPack *pmbp, Multigrid *pmg,
                            const RegionIndcs &indcs, const Real center_x,
                            const Real center_y, const Real center_z,
                            const Real radius) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || pmg == nullptr || radius <= 0.0) {
    return;
  }

  if (pmg->OnHost()) {
    MaskBHSinkGravitySource<HostExeSpace>(
        pmg->GetCurrentSource_h(), pmbp->pmb->mb_size.h_view,
        pmg->GetNumMeshBlocks(), indcs, pmg->GetGhostCells(),
        center_x, center_y, center_z, radius);
  } else {
    MaskBHSinkGravitySource<DevExeSpace>(
        pmg->GetCurrentSource(), pmbp->pmb->mb_size.d_view,
        pmg->GetNumMeshBlocks(), indcs, pmg->GetGhostCells(),
        center_x, center_y, center_z, radius);
    pmg->MarkCurrentSourceDeviceModified();
  }
}

template <typename ExeSpace, typename SrcView, typename SizeView>
void MaskGravitySourceOutsideRadius(const SrcView &src, const SizeView &size,
                                    const int nmb, const RegionIndcs &indcs,
                                    const int ngh, const Real center_x,
                                    const Real center_y, const Real center_z,
                                    const Real radius) {
  const int is = ngh;
  const int js = ngh;
  const int ks = ngh;
  const int ie = is + indcs.nx1 - 1;
  const int je = js + indcs.nx2 - 1;
  const int ke = ks + indcs.nx3 - 1;
  const Real radius2 = radius*radius;
  par_for("MGGravity::MaskSourceOutsideRadius", ExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size(m).x1min, size(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size(m).x2min, size(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size(m).x3min, size(m).x3max);
    const Real dx = x-center_x;
    const Real dy = y-center_y;
    const Real dz = z-center_z;
    if (dx*dx + dy*dy + dz*dz > radius2) src(m, 0, k, j, i) = 0.0;
  });
}

void ApplyGravitySourceRadiusMask(MeshBlockPack *pmbp, Multigrid *pmg,
                                  const RegionIndcs &indcs, const Real center_x,
                                  const Real center_y, const Real center_z,
                                  const Real radius) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || pmg == nullptr || radius <= 0.0) {
    return;
  }
  if (pmg->OnHost()) {
    MaskGravitySourceOutsideRadius<HostExeSpace>(
        pmg->GetCurrentSource_h(), pmbp->pmb->mb_size.h_view,
        pmg->GetNumMeshBlocks(), indcs, pmg->GetGhostCells(),
        center_x, center_y, center_z, radius);
  } else {
    MaskGravitySourceOutsideRadius<DevExeSpace>(
        pmg->GetCurrentSource(), pmbp->pmb->mb_size.d_view,
        pmg->GetNumMeshBlocks(), indcs, pmg->GetGhostCells(),
        center_x, center_y, center_z, radius);
    pmg->MarkCurrentSourceDeviceModified();
  }
}

// One red-black colour of a root level whose cells do not all share the same diagonal.
// b[f] is what face f adds to the diagonal of a cell that touches it; see
// MGGravity::SmoothPackCoarsestEffectiveDiagonal for where the six values come from.
template <typename ViewType>
void SmoothGravityEffectiveDiagonal(const ViewType &u, const ViewType &src, int nmmb,
                                    int is, int ie, int js, int je, int ks, int ke,
                                    int color, Real dx2, Real omega,
                                    Real wy, Real wz, Real diag,
                                    Real bx0, Real bx1, Real by0, Real by1,
                                    Real bz0, Real bz1) {
  using ExeSpace = typename ViewType::execution_space;
  const int nhalf = (ie - is)/2 + 1;
  par_for("MGGravity::SmoothCoarsestEff", ExeSpace(), 0, nmmb-1, ks, ke, js, je,
          0, nhalf-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int ih) {
    const int i = is + ((color + k + j) & 1) + 2*ih;
    if (i > ie) return;
    Real b = 0.0;
    if (i == is) b += bx0;
    if (i == ie) b += bx1;
    if (j == js) b += by0;
    if (j == je) b += by1;
    if (k == ks) b += bz0;
    if (k == ke) b += bz1;
    const Real lap = ApplyGravityLaplacian(u, m, 0, k, j, i, wy, wz, diag);
    u(m,0,k,j,i) -= (lap - src(m,0,k,j,i)*dx2) * (omega/(diag + b));
  });
}

template <typename ViewType, typename DxViewType, typename ParityViewType>
void SmoothGravityBoundaryFlat(const ViewType &u, const ViewType &src,
                               const DxViewType &brdx, const ParityViewType &block_parity,
                               int block_cells_odd, int nmmb, int is, int ie,
                               int js, int je, int ks, int ke, int color,
                               int coffset, int rlev, Real omega_over_diag,
                               Real wy, Real wz, Real diag) {
  using ExeSpace = typename ViewType::execution_space;
  const int nx = ie - is + 1;
  const int ny = je - js + 1;
  const int nz = ke - ks + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0) return;

  const int nx_int = std::max(0, nx - 2);
  const int ny_int = std::max(0, ny - 2);
  const int x_sides = (nx > 1) ? 2 : 1;
  const int y_sides = (ny > 1) ? 2 : 1;
  const int z_sides = (nz > 1) ? 2 : 1;
  const int x_count = x_sides * ny * nz;
  const int y_count = (nx_int > 0) ? y_sides * nx_int * nz : 0;
  const int z_count = (nx_int > 0 && ny_int > 0) ? z_sides * nx_int * ny_int : 0;
  const int nbnd = x_count + y_count + z_count;
  if (nbnd <= 0) return;

  const int is_l = is, ie_l = ie, js_l = js, je_l = je, ks_l = ks, ke_l = ke;
  const int nx_int_l = nx_int;
  const int x_count_l = x_count, y_count_l = y_count;
  const int y_plane = nx_int * nz;
  const int z_plane = nx_int * ny_int;
  const int active_color = color ^ coffset;
  const int block_cells_odd_l = block_cells_odd;
  const int rlev_l = rlev;
  const Real odiag = omega_over_diag;
  const Real wy_l = wy, wz_l = wz, diag_l = diag;

  par_for("MGGravity::SmoothBoundaryFlat", ExeSpace(), 0, nmmb - 1, 0, nbnd - 1,
  KOKKOS_LAMBDA(const int m, const int n) {
    int i, j, k;
    if (n < x_count_l) {
      const int plane = ny * nz;
      const int side = n / plane;
      const int rem = n - side * plane;
      k = ks_l + rem / ny;
      j = js_l + rem - (rem / ny) * ny;
      i = (side == 0) ? is_l : ie_l;
    } else if (n < x_count_l + y_count_l) {
      const int q = n - x_count_l;
      const int side = q / y_plane;
      const int rem = q - side * y_plane;
      k = ks_l + rem / nx_int_l;
      i = is_l + 1 + rem - (rem / nx_int_l) * nx_int_l;
      j = (side == 0) ? js_l : je_l;
    } else {
      const int q = n - x_count_l - y_count_l;
      const int side = q / z_plane;
      const int rem = q - side * z_plane;
      j = js_l + 1 + rem / nx_int_l;
      i = is_l + 1 + rem - (rem / nx_int_l) * nx_int_l;
      k = (side == 0) ? ks_l : ke_l;
    }

    const int block_color = active_color ^ (block_cells_odd_l & block_parity(m));
    if (((i - is_l - (block_color + k + j)) & 1) != 0) return;

    const Real dx = (rlev_l <= 0) ? brdx(m) * static_cast<Real>(1 << (-rlev_l))
                                  : brdx(m) / static_cast<Real>(1 << rlev_l);
    const Real dx2 = dx * dx;
    const Real lap = ApplyGravityLaplacian(u, m, 0, k, j, i, wy_l, wz_l, diag_l);
    u(m,0,k,j,i) -= (lap - src(m,0,k,j,i) * dx2) * odiag;
  });
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1.0);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    rho_grav_min_ = std::max(static_cast<Real>(0.0),
                             pin->GetOrAddReal("gravity", "rho_grav_min", 0.0));
    // Lower end of the continuous coupling ramp (utils/gravity_weight.hpp).  The Poisson
    // source and the gravitational momentum/work source must use the SAME weight or the
    // self-gravity pair stops conserving momentum, so this is the fluid density floor,
    // exactly what SourceTerms::Gravity reads out of the EOS.
    rho_grav_floor_ = 0.0;
    if (pmbp->phydro != nullptr) {
      rho_grav_floor_ = pmbp->phydro->peos->eos_data.dfloor;
    } else if (pmbp->pmhd != nullptr) {
      rho_grav_floor_ = pmbp->pmhd->peos->eos_data.dfloor;
    }
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    const bool amr_enabled = pmbp->pmesh->multilevel;
    const int default_np = npresmooth_;
    // AMR coarse-fine interpolation error is damped more efficiently by one
    // extra post-smoothing sweep than by symmetric 2/2 smoothing.
    const int default_npost = amr_enabled ? 2 : npostsmooth_;
    npresmooth_ = std::max(1, pin->GetOrAddInteger("gravity", "npresmooth", default_np));
    npostsmooth_ = std::max(1, pin->GetOrAddInteger("gravity", "npostsmooth",
                                                    default_npost));
    // Off by default, because whether it pays depends on the convergence threshold and
    // that is not knowable here.  Measured on eight V100s with the defaults below: a
    // refined collapse stopped at a loose defect ran 5.4 % faster, but a uniform grid
    // driven to 1e-12 ran 49 % slower, because a stale ghost layer costs nothing over
    // the five V-cycles of the former and dominates the asymptotic rate of the latter.
    // Decks that iterate to a loose target on a refined mesh should turn it on.
    const int default_same_stride = 1;
    same_exchange_stride_ = std::max(1, pin->GetOrAddInteger(
        "gravity", "same_exchange_stride", default_same_stride));
    const int default_local_sweeps = 1;
    local_sweeps_per_exchange_ = std::max(1, pin->GetOrAddInteger(
        "gravity", "local_sweeps_per_exchange", default_local_sweeps));
    reduced_exchange_max_edge_ = std::max(0, pin->GetOrAddInteger(
        "gravity", "reduced_exchange_max_edge", reduced_exchange_max_edge_));
    // Off by default, because carrying the coarse hierarchy on one rank does not remove
    // the work, it serialises it: the owner runs the octet and root levels while every
    // other rank blocks in the scatter, and the scatter itself is added on top.  Measured
    // on a five-level collapse at four ranks, turning it on cost 3.3% of the multigrid
    // time and 22% of the octet time.  It remains available for the case it was written
    // for, a rank count large enough that replicating the coarse levels stops being free.
    prefer_distributed_coarse_solve_ = pin->GetOrAddBoolean(
        "gravity", "distributed_coarse_solve", false);
    // Red-black Gauss-Seidel, not Chebyshev-Jacobi: with one local sweep per exchange
    // it does the same arithmetic as the exact schedule and only lets the black sweep
    // see a half-sweep-old ghost layer, which on eight V100s bought 5.4 % on a refined
    // collapse against 2.8 % for Chebyshev.  Chebyshev remains selectable.
    const std::string default_ca_pack_smoother = "rbgs";
    std::string ca_pack_smoother = pin->GetOrAddString("gravity", "ca_pack_smoother",
                                                       default_ca_pack_smoother);
    if (ca_pack_smoother == "jacobi") {
      ca_pack_smoother_type_ = 1;
    } else if (ca_pack_smoother == "chebyshev" || ca_pack_smoother == "chebyshev2") {
      ca_pack_smoother_type_ = 2;
    } else {
      ca_pack_smoother_type_ = 0;
    }
    ca_pack_cheb_lambda_min_ = pin->GetOrAddReal("gravity", "ca_pack_cheb_lambda_min",
                                                 ca_pack_cheb_lambda_min_);
    ca_pack_cheb_lambda_max_ = pin->GetOrAddReal("gravity", "ca_pack_cheb_lambda_max",
                                                 ca_pack_cheb_lambda_max_);
    if (ca_pack_cheb_lambda_max_ <= ca_pack_cheb_lambda_min_) {
      ca_pack_cheb_lambda_max_ = ca_pack_cheb_lambda_min_ + 1.0;
    }
    full_multigrid_ = pin->GetOrAddBoolean("gravity", "full_multigrid", true);
    fmg_ncycle_ = pin->GetOrAddInteger("gravity", "fmg_ncycle", 1);
    const std::string show_defect = pin->GetOrAddString("gravity", "show_defect", "0");
    if (show_defect == "true") {
      fshowdef_ = 1;
    } else if (show_defect == "false") {
      fshowdef_ = 0;
    } else {
      try {
        fshowdef_ = std::max(0, std::stoi(show_defect));
      } catch (const std::exception &) {
        std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                  << "gravity/show_defect must be true, false, or a non-negative integer."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    mg_verbose_ = std::max(0, pin->GetOrAddInteger("gravity", "mg_verbose", 0));
    fsubtract_average_ = pin->GetOrAddBoolean(
        "gravity", "subtract_average", pmy_mesh_->strictly_periodic);
    const std::string default_mg_prolong = amr_enabled ? "tricubic" : "trilinear";
    const std::string mg_prolongation = pin->GetOrAddString(
        "gravity", "mg_prolongation", default_mg_prolong);
    fprolongation_ = (mg_prolongation == "tricubic") ? 1 : 0;
    defect_check_interval_ = std::max(1,
        pin->GetOrAddInteger("gravity", "defect_check_interval", defect_check_interval_));
    // The floor this key used to set was an absolute defect, which only means the same
    // thing on the one problem it was tuned for; it is now a fraction of the source norm.
    // A deck that still sets it would be silently disobeyed, so refuse instead.
    if (pin->RetireDeadParameter("gravity", "auto_target_defect")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<gravity>/auto_target_defect is read by no code path. The convergence "
                << "floor for threshold = 0 is now a fixed fraction of the source norm; "
                << "set an absolute target with <gravity>/threshold instead." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto_max_extra_cycles_ = pin->GetOrAddInteger("gravity", "auto_max_extra_cycles",
                                                  auto_max_extra_cycles_);
    coarsest_min_sweeps_ = std::max(1, pin->GetOrAddInteger(
        "gravity", "coarsest_min_sweeps", coarsest_min_sweeps_));
    show_timing_ = pin->GetOrAddBoolean("gravity", "show_timing", false);
    const bool has_solve_every = pin->DoesParameterExist("gravity", "solve_every");
    const bool has_solve_dt = pin->DoesParameterExist("gravity", "solve_dt");
    if (has_solve_dt) {
      solve_dt_ = pin->GetReal("gravity", "solve_dt");
      if (solve_dt_ < 0.0) {
        std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                  << "gravity/solve_dt must be >= 0 when it is specified."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (solve_dt_ > 0.0) {
        use_solve_dt_ = true;
        next_solve_time_ = pmbp->pmesh->time;
        if (has_solve_every && global_variable::my_rank == 0) {
          std::cout << "MGGravityDriver::MGGravityDriver: gravity/solve_dt overrides "
                    << "gravity/solve_every on this run." << std::endl;
        }
      } else {
        solve_every_ = std::max(1, pin->GetOrAddInteger("gravity", "solve_every",
                                                        solve_every_));
      }
    } else if (has_solve_every) {
      solve_every_ = std::max(1, pin->GetOrAddInteger("gravity", "solve_every",
          solve_every_));
    } else {
      solve_every_ = std::max(1, pin->GetOrAddInteger("gravity", "solve_every",
          solve_every_));
    }
    if (global_variable::my_rank == 0) {
      if (use_solve_dt_) {
        std::cout << "MGGravityDriver::MGGravityDriver: using gravity/solve_dt = "
                  << solve_dt_ << std::endl;
      } else {
        std::cout << "MGGravityDriver::MGGravityDriver: using gravity/solve_every = "
                  << solve_every_ << std::endl;
      }
    }
    warm_final_niter_ = std::max(1, pin->GetOrAddInteger("gravity",
                                                          "warm_final_niter",
                                                          warm_final_niter_));
    if (eps_ < 0.0 && niter_ < 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
        exit(EXIT_FAILURE);
  }
    // niteration = 0 is only ever meaningful as "let the threshold decide": the cold
    // solve maps it to the FMG ramp (Solve() below), but every warm solve then runs
    // SolveIterativeFixedTimes with zero V-cycles, so the potential stays at its first
    // value for the rest of the run while the log still reports gravity as on.
    if (eps_ < 0.0 && niter_ == 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "gravity/niteration = 0 requires gravity/threshold >= 0." << std::endl
        << "Without a threshold every warm solve would run zero V-cycles and the "
        << "potential would be frozen at its first value." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control, or give "
        << "\"niteration\" a positive number of V-cycles." << std::endl;
        exit(EXIT_FAILURE);
  }
  if (four_pi_G_ < 0.0) {
    std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    exit(EXIT_FAILURE);
  }
  mg_mesh_bcs_[inner_x1] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[inner_x1]);
  mg_mesh_bcs_[outer_x1] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[outer_x1]);
  mg_mesh_bcs_[inner_x2] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[inner_x2]);
  mg_mesh_bcs_[outer_x2] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[outer_x2]);
  mg_mesh_bcs_[inner_x3] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[inner_x3]);
  mg_mesh_bcs_[outer_x3] = MapMeshToMGBoundary(pmy_mesh_->mesh_bcs[outer_x3]);

  // A single gravity/mg_bc value is a convenient default for every non-periodic face.
  // Explicit per-face gravity boundary conditions below take precedence.
  const std::string mg_bc_str = pin->GetOrAddString("gravity", "mg_bc", "none");
  if (mg_bc_str != "none") {
    const BoundaryFlag mg_bc = ParseMGBoundaryFlag(mg_bc_str);
    if (mg_bc == BoundaryFlag::undef || mg_bc == BoundaryFlag::periodic) {
      std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                << "gravity/mg_bc must be zerofixed, zerograd, multipole, or none."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int f = inner_x1; f <= outer_x3; ++f) {
      if (mg_mesh_bcs_[f] != BoundaryFlag::periodic) mg_mesh_bcs_[f] = mg_bc;
    }
  }

  if (pin->DoesParameterExist("gravity", "ix1_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ix1_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[inner_x1] = bc;
  }
  if (pin->DoesParameterExist("gravity", "ox1_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ox1_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[outer_x1] = bc;
  }
  if (pin->DoesParameterExist("gravity", "ix2_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ix2_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[inner_x2] = bc;
  }
  if (pin->DoesParameterExist("gravity", "ox2_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ox2_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[outer_x2] = bc;
  }
  if (pin->DoesParameterExist("gravity", "ix3_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ix3_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[inner_x3] = bc;
  }
  if (pin->DoesParameterExist("gravity", "ox3_bc")) {
    BoundaryFlag bc = ParseMGBoundaryFlag(pin->GetString("gravity", "ox3_bc"));
    if (bc != BoundaryFlag::undef) mg_mesh_bcs_[outer_x3] = bc;
  }

  for (int f = inner_x1; f <= outer_x3; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::periodic &&
        pmy_mesh_->mesh_bcs[f] != BoundaryFlag::periodic) {
      std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                << "Multigrid periodic boundary requires mesh periodic boundary on face "
                << f << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  bool has_fixed_potential_boundary = false;
  for (int f = inner_x1; f <= outer_x3; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_zerofixed ||
        mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) {
      has_fixed_potential_boundary = true;
      break;
    }
  }
  if (!has_fixed_potential_boundary && !fsubtract_average_) {
    fsubtract_average_ = true;
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in MGGravityDriver::MGGravityDriver" << std::endl
                << "All gravity boundaries are non-fixed (periodic/zerograd). "
                << "Enabling gravity/subtract_average=true for solvability."
                << std::endl;
    }
  }

  for (int f = inner_x1; f <= outer_x3; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) {
      mporder_ = 0;
      break;
    }
  }
  if (mporder_ >= 0) {
    mporder_ = pin->GetOrAddInteger("gravity", "mporder", 4);
    autompo_ = pin->GetOrAddBoolean("gravity", "auto_mporigin", true);
    nodipole_ = pin->GetOrAddBoolean("gravity", "nodipole", false);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                << "gravity/mporder must be 2 (quadrupole) or 4 (hexadecapole)."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (autompo_ && nodipole_) {
      std::cout << "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
                << "gravity/auto_mporigin and gravity/nodipole cannot both be true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!autompo_) {
      mpo_[0] = pin->GetReal("gravity", "mporigin_x1");
      mpo_[1] = pin->GetReal("gravity", "mporigin_x2");
      mpo_[2] = pin->GetReal("gravity", "mporigin_x3");
    }
    AllocateMultipoleCoefficients();
    fsubtract_average_ = false;
  }

  // Source masking parameters
  mask_radius_ = pin->GetOrAddReal("gravity", "mask_radius", -1.0);
  mask_origin_[0] = pin->GetOrAddReal("gravity", "mask_origin_x1", 0.0);
  mask_origin_[1] = pin->GetOrAddReal("gravity", "mask_origin_x2", 0.0);
  mask_origin_[2] = pin->GetOrAddReal("gravity", "mask_origin_x3", 0.0);
  // Allocate the root multigrid
  reciprocity_test_ = pin->GetOrAddBoolean("gravity", "reciprocity_test", false);
  reciprocity_operator_test_ =
      pin->GetOrAddBoolean("gravity", "reciprocity_operator_test", true);
  reciprocity_mod_amp_ = pin->GetOrAddReal("gravity", "reciprocity_mod_amp", 0.3);
  reciprocity_blob_amp_ = pin->GetOrAddReal("gravity", "reciprocity_blob_amp", 0.5);
  reciprocity_blob_center_[0] = pin->GetOrAddReal("gravity", "reciprocity_blob_x1", 3.0);
  reciprocity_blob_center_[1] = pin->GetOrAddReal("gravity", "reciprocity_blob_x2", 0.0);
  reciprocity_blob_center_[2] = pin->GetOrAddReal("gravity", "reciprocity_blob_x3", 0.0);
  reciprocity_blob_width_ = pin->GetOrAddReal("gravity", "reciprocity_blob_width", 2.0);
  reciprocity_dump_ = pin->GetOrAddString("gravity", "reciprocity_dump", "");
  // Symmetric (flux-matched) coarse/fine interface: drop the tangential-gradient terms
  // of the fine-side ghost fill.  See MultigridDriver::FCTangentialGradient.
  fc_tangential_gradient_ = !pin->GetOrAddBoolean("gravity", "mg_fc_symmetric", false);
  int nghost = pin->GetOrAddInteger("gravity", "mg_nghost", 1);
  // The coarse buffer that carries a coarser neighbour into the octet prolongation is a
  // fixed 3x3x3 stencil, and ProlongateOctetBoundariesFluxCons reaches ngh +/- 1 into it,
  // so a second ghost layer walks off both ends of cbuf_ as soon as octets exist.  One
  // layer is also all a seven-point Laplacian needs across a coarse-fine boundary
  // (Tomida & Stone 2023, section 3.3.2), so there is nothing to gain by widening it.
  // The refusal is gated on the mesh rather than made unconditional because the
  // communication-avoiding halo schedule does use two layers, and it runs on uniform
  // meshes only.  It is raised here rather than at solve time because nreflevel_, which
  // is what actually decides whether octets appear, is not known until PrepareForAMR
  // runs, by which point the run has already paid for its queue slot; multilevel is the
  // constructor-time precondition for nreflevel_ ever becoming nonzero, and it covers a
  // mesh that starts unrefined and refines later.
  if (nghost > 1 && pmbp->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<gravity>/mg_nghost = " << nghost << " is not supported on a refined "
              << "mesh: the flux-conservative coarse-fine prolongation indexes a fixed "
              << "3x3x3 coarse buffer and would read and write outside it. Use "
              << "mg_nghost = 1." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // A root grid has one cell per root-level MeshBlock, so it is tiny -- 4^3 on the
  // collapse test, 3x3x1 on the tidal-disruption decks -- and every operation on it is a
  // kernel launch whose arithmetic is a rounding error next to its own overhead, with the
  // whole sequence serialised because each level depends on the one before.  Running it
  // on the host instead took 68% off the bottom solve and 56% off the coarse levels of a
  // five-level collapse at four ranks, 4% off the multigrid time overall.  Past some size
  // the arithmetic starts to matter and the device wins again; the crossover was not
  // measured, so the default only claims the sizes that production actually uses.
  const int root_cells = pmbp->pmesh->nmb_rootx1 * pmbp->pmesh->nmb_rootx2
                       * pmbp->pmesh->nmb_rootx3;
  const bool default_root_on_host = (root_cells <= 16*16*16);
  bool root_on_host = pin->GetOrAddBoolean("gravity", "root_on_host",
                                           default_root_on_host);
  mgroot_ = new MGGravity(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGGravity(this, pmbp, nghost);
  const Real dx = mgroot_->GetRootDx();
  const Real dx_over_dy = dx / mgroot_->GetRootDy();
  const Real dx_over_dz = dx / mgroot_->GetRootDz();
  gravity_wy_ = dx_over_dy * dx_over_dy;
  gravity_wz_ = dx_over_dz * dx_over_dz;
  gravity_diag_ = 2.0 * (1.0 + gravity_wy_ + gravity_wz_);
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  // Multigrid never performs flux correction: PackAndSendFluxCC / RecvAndUnpackFluxCC
  // are queued only from the hydro, mhd and radiation task lists, and
  // nothing under src/multigrid or src/gravity touches MeshBoundaryBuffer::flux.  The
  // base class would otherwise size flux storage from nvar, allocating ~111 MiB/rank of
  // device memory that is never read or written.  Ask for zero flux components.
  mglevels_->pbval->InitializeBuffers(nvar_, 0);
  mglevels_->pbval->RemapIndicesForMG();
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGGravityDriver::SetFourPiG(Real four_pi_G) {
  four_pi_G_ = four_pi_G;
}

bool MGGravityDriver::RefreshFinestGhosts(DvceArray5D<Real> &phi, int phi_nghost,
                                          bool finest_holds_phi) {
  if (mglevels_ == nullptr || mglevels_->pbval == nullptr || pmy_pack_ == nullptr) {
    return false;
  }
  // The exchange below repeats, step for step, the refresh_boundaries() the V-cycle
  // tail ends on (OneStepToFiner: same-level halo, PhysicalBoundary, fine/coarse ghosts,
  // then the override fill inside FillFCBoundary), and finest_bvals_fresh_ records that
  // nothing has written the finest array since -- the same claim OneStepToCoarser skips
  // its initial exchange on.  The one textual difference is the pre-unpack physical
  // fill: this routine applies it whenever the mesh is not strictly periodic, the tail
  // withholds it when an MG boundary overrides a periodic mesh face, so that mesh keeps
  // the exchange.  Read before PrepareForAMR(), which clears the flag unconditionally.
  const bool halo_fresh = finest_holds_phi && finest_bvals_fresh_ &&
      !(MGBoundaryOverridesPeriodicMesh() && !pmy_pack_->pmesh->strictly_periodic);
  PrepareForAMR();
  const int saved_driver_level = current_level_;
  Multigrid *saved_pmg = pmg;
  const int saved_mg_level = mglevels_->GetCurrentLevel();
  const bool saved_reduce_same_exchange = active_reduce_same_exchange_;

  pmg = mglevels_;
  mglevels_->SetCurrentLevel(mglevels_->GetNumberOfLevels() - 1);
  active_reduce_same_exchange_ = false;
  // The caller that has just finished a solve is handing back the very array the finest
  // level already holds, so copying phi into it would only restore what is there.  Every
  // other caller (the post-AMR refresh, where phi is the authority and the multigrid
  // arrays hold whatever the last solve left) has to load it.
  if (!finest_holds_phi) mglevels_->LoadFinestData(phi, 0, phi_nghost);
  DvceArray5D<Real> u = mglevels_->GetCurrentData();

  if (!halo_fresh) {
    while (mglevels_->pbval->InitRecvMG(nvar_) == TaskStatus::incomplete) {}
    while (mglevels_->pbval->PackAndSendMG(u) == TaskStatus::incomplete) {}
    if (!pmy_pack_->pmesh->strictly_periodic) {
      ApplyPhysicalBoundariesBlocks();
    }
    while (mglevels_->pbval->RecvAndUnpackMG(u) == TaskStatus::incomplete) {}
    if (nreflevel_ > 0) {
      while (mglevels_->pbval->FillFineCoarseMGGhosts(u) == TaskStatus::incomplete) {}
    }
    if (MGBoundaryOverridesPeriodicMesh()) ApplyPhysicalBoundariesBlocks();
  }
  // Drained in both cases: the tail's refresh leaves its sends in flight on purpose.
  while (mglevels_->pbval->ClearRecvMG() == TaskStatus::incomplete) {}
  while (mglevels_->pbval->ClearSendMG() == TaskStatus::incomplete) {}
  mglevels_->RetrieveResult(phi, 0, phi_nghost);

  active_reduce_same_exchange_ = saved_reduce_same_exchange;
  mglevels_->SetCurrentLevel(saved_mg_level);
  current_level_ = saved_driver_level;
  pmg = saved_pmg;
  return true;
}

bool MGGravityDriver::SolveDueForCycle(int ncycle, Real time) const {
  if (use_solve_dt_) return SolveDueAtTime(time);
  return (solve_every_ <= 1) || ((ncycle % solve_every_) == 0);
}

bool MGGravityDriver::SolveDueAtTime(Real time) const {
  if (!use_solve_dt_) return false;
  const Real ref_time = std::max(std::abs(time), std::abs(next_solve_time_));
  const Real time_tol = static_cast<Real>(64.0) * std::numeric_limits<Real>::epsilon() *
                        std::max(static_cast<Real>(1.0), ref_time);
  return (time + time_tol) >= next_solve_time_;
}

bool MGGravityDriver::WindowCrossesSolveTime(Real time_end) const {
  if (!use_solve_dt_) return false;
  const Real ref_time = std::max(std::abs(time_end), std::abs(next_solve_time_));
  const Real time_tol = static_cast<Real>(64.0) * std::numeric_limits<Real>::epsilon() *
                        std::max(static_cast<Real>(1.0), ref_time);
  return time_end > (next_solve_time_ + time_tol);
}

void MGGravityDriver::ResetCadenceCache() {
  cadence_cycle_ = -1;
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//! \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  //delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGGravityDriver::Solve(Driver *pdriver, int stage, Real dt) {
  // Explicit energy-window mode freezes one authoritative potential throughout
  // all RK/LAT source and reflux work, then solves again at its synchronized end.
  if (pmy_pack_->pgrav != nullptr && pmy_pack_->pgrav->energy_window_open) return;
  Kokkos::Timer solve_timer;
  const bool timing_enabled = show_timing_;
  double t_entry = 0.0;
  double t_load_source = 0.0;
  double t_load_phi = 0.0;
  double t_setup = 0.0;
  double t_mg_solve = 0.0;
  double t_defect = 0.0;
  double t_retrieve = 0.0;
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;
  // A synchronized endpoint solve can precede AMR at the same cycle number.
  // Cycle identity alone does not imply that cached octets still match the mesh.
  const auto topology = pmy_pack_->pmesh->topology_version;
  if (!(stage > 1 && last_prepare_cycle_ == pmy_pack_->pmesh->ncycle &&
        last_prepare_topology_ == topology)) {
    PrepareForAMR();
    last_prepare_cycle_ = pmy_pack_->pmesh->ncycle;
    last_prepare_topology_ = topology;
  }
  const bool mesh_changed = amr_mesh_changed_;
  const bool is_last_stage = (pdriver != nullptr) && (stage >= pdriver->nexp_stages);
  auto *pgrav = pmy_pack_->pgrav;
  if (timing_enabled) {
    Kokkos::fence();
    t_entry = solve_timer.seconds();
  }
  const int ncycle = pmy_pack_->pmesh->ncycle;
  const Real cycle_time = pmy_pack_->pmesh->time;
  if (cadence_cycle_ != ncycle) {
    cadence_cycle_ = ncycle;
    if (use_solve_dt_) {
      const Real ref_time = std::max(std::abs(cycle_time), std::abs(next_solve_time_));
      const Real time_tol = static_cast<Real>(64.0) * std::numeric_limits<Real>::epsilon() *
                            std::max(static_cast<Real>(1.0), ref_time);
      cadence_due_this_cycle_ = (cycle_time + time_tol) >= next_solve_time_;
    } else {
      cadence_due_this_cycle_ =
          (solve_every_ <= 1) || ((ncycle % solve_every_) == 0);
    }
  }
  const bool cadence_due = cadence_due_this_cycle_;
  const bool use_amr_reduced_exchange =
      (nranks_ > 1) && (nreflevel_ > 0) && (same_exchange_stride_ > 1);
  {
    int nmb = pmy_pack_->nmb_thispack;
    if (pgrav != nullptr && pgrav->ResizeMeshBlockStorage(nmb)) {
      pgrav->MarkPhiInvalid();
    }
  }
  const bool has_warm_potential = (pgrav != nullptr) && pgrav->phi_valid;
  // Ordinary self-gravity updates the potential at every RK stage, as required by
  // the stage density.  Only an explicitly cadenced solve freezes a valid potential
  // between scheduled updates (the mode used by LAT gravity windows).
  const bool freeze_between_solutions = use_solve_dt_ || (solve_every_ > 1);
  const bool skip_idle_intermediate_solve =
      has_warm_potential && !is_last_stage && !mesh_changed && freeze_between_solutions;
  const bool skip_cadence_cycle =
      has_warm_potential && !mesh_changed && !cadence_due;

  if (skip_idle_intermediate_solve) {
    return;
  }
  if (skip_cadence_cycle) {
    return;
  }

  // The multigrid source is density from whichever fluid system is active.
  // Hydro and MHD both store conserved density in slot IDN.
  const DvceArray5D<Real> *fluid_cons = nullptr;
  if (pmy_pack_->phydro != nullptr) {
    fluid_cons = &(pmy_pack_->phydro->u0);
  } else if (pmy_pack_->pmhd != nullptr) {
    fluid_cons = &(pmy_pack_->pmhd->u0);
  } else {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in MGGravityDriver::Solve" << std::endl
                << "Self-gravity requires hydro or mhd state in MeshBlockPack."
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
  if (timing_enabled) solve_timer.reset();
  bool sink_gravity_mask_enabled = false;
  Real sink_gravity_mask_radius = 0.0;
  Real sink_gravity_mask_x = 0.0;
  Real sink_gravity_mask_y = 0.0;
  Real sink_gravity_mask_z = 0.0;
  problem_runtime::GetBHSinkGravityMask(pmy_pack_->pmesh->time,
                                        sink_gravity_mask_enabled,
                                        sink_gravity_mask_radius,
                                        sink_gravity_mask_x,
                                        sink_gravity_mask_y,
                                        sink_gravity_mask_z);
  const bool generic_source_mask_enabled = mask_radius_ > 0.0;
  if (fsubtract_average_ && !sink_gravity_mask_enabled &&
      !generic_source_mask_enabled) {
    mglevels_->LoadSourceAndSubtractAverage(*fluid_cons, IDN, indcs_.ng,
                                            -four_pi_G_, rho_grav_min_,
                                            rho_grav_floor_);
    finest_source_average_subtracted_ = true;
  } else {
    mglevels_->LoadSource(*fluid_cons, IDN, indcs_.ng, -four_pi_G_,
                          rho_grav_min_, rho_grav_floor_);
    if (generic_source_mask_enabled) {
      ApplyGravitySourceRadiusMask(pmy_pack_, mglevels_, indcs_,
                                   mask_origin_[0], mask_origin_[1], mask_origin_[2],
                                   mask_radius_);
    }
    if (sink_gravity_mask_enabled) {
      ApplyBHSinkGravityMask(pmy_pack_, mglevels_, indcs_,
                             sink_gravity_mask_x, sink_gravity_mask_y,
                             sink_gravity_mask_z, sink_gravity_mask_radius);
    }
    if (fsubtract_average_) {
      // Two passes, for the reason given in Multigrid::LoadSourceAndSubtractAverage.
      mglevels_->SubtractAverage(MGVariable::src, 0,
                                 mglevels_->CalculateAverage(MGVariable::src));
      mglevels_->SubtractAverage(MGVariable::src, 0,
                                 mglevels_->CalculateAverage(MGVariable::src));
      finest_source_average_subtracted_ = true;
    }
  }
  // The convergence floor for the threshold == 0 policy is a fraction of the source
  // norm, so take that norm once here, on the finest level, while the source is the
  // physical one -- the coarser levels carry the FAS right-hand side instead.
  // Only the threshold == 0 policy reads it; a deck with a threshold of its own, or a
  // fixed cycle count, would pay the global reduction for a number nothing consumes.
  mglevels_->SetCurrentLevel(mglevels_->GetNumberOfLevels() - 1);
  source_norm_ = (eps_ == 0.0) ? CalculateArrayNorm(MGVariable::src, MGNormType::l2, 0)
                               : 0.0;
  if (timing_enabled) {
    Kokkos::fence();
    t_load_source = solve_timer.seconds();
  }

  // Use a cheap warm-start update on intermediate RK stages, while honoring
  // user-configured convergence policy on completed stages.
  const bool report_stage_defect = (fshowdef_ != 0) && is_last_stage;

  const bool saved_full_multigrid = full_multigrid_;
  const Real saved_eps = eps_;
  const int saved_niter = niter_;
  const int saved_npostsmooth = npostsmooth_;
  const int saved_finest_exact_polish_passes = finest_exact_polish_passes_;
  const int saved_showdef = fshowdef_;
  fshowdef_ = is_last_stage ? saved_showdef : 0;

  // The FMG ramp is for a cold solve only.  Fresh starts and restart-remap conversions
  // carry hydro state and no potential, and from zero the ramp converges much faster than
  // plain V-cycles on a deep hierarchy; the deck's full_multigrid says whether to take
  // it.  A potential from the previous solve, including one AMR just preserved and
  // prolongated, is a better starting point than the ramp and also the only safe one:
  // SolveFMGCoarser restricts the source down the hierarchy but never touches u, so the
  // coarse levels would still hold the previous solve's potential when the bottom solve
  // forms its FAS right-hand side.  That solves L u = f + L(u_old) and returns
  // u_desired + u_old, which FMGProlongate then writes over the finer level rather than
  // adding to it.  The hierarchy is zeroed below as a second line of defence, but not
  // taking the ramp at all is both correct and cheaper.
  //
  // A cold solve converges on every stage.  The first stage's potential feeds the first
  // update of the cycle, and leaving it at the FMG ramp alone -- the earlier policy, to
  // save one solve per run -- put the ramp's residual, a few 1e-3 of the potential, into
  // that one update.  A fresh start carried the error from t = 0; a restart, whose
  // potential is not in the checkpoint, carried it from the restart cycle and then
  // differed from the uninterrupted run by a few 1e-3 in the kinetic energy for the rest
  // of the run.
  full_multigrid_ = saved_full_multigrid && !has_warm_potential;
  if (!has_warm_potential && (niter_ == 0)) niter_ = -1;
  // The defect carries the units of the source, so how far the previous solve got only
  // means the same thing on two problems when it is measured against the target that
  // SolveIterative itself will stop at.  Scaling four_pi_G by a million scales the
  // source, the potential and the defect by the same million and leaves this ratio where
  // it was, which is what a policy about convergence should do.
  const Real target_defect = (saved_eps > 0.0) ? saved_eps : AutoTargetDefect();
  const Real defect_ratio = last_defect_valid_
      ? last_defect_norm_ / target_defect
      : std::numeric_limits<Real>::infinity();
  if (has_warm_potential && is_last_stage && !mesh_changed &&
      saved_eps == 0.0 && saved_niter <= 0) {
    int warm_cycles = warm_final_niter_;
    // A V-cycle on these hierarchies takes roughly an order of magnitude off the defect,
    // so a previous solve that is already within one cycle's worth of the target needs
    // one cycle and one within three cycles' worth needs three.
    if (defect_ratio <= static_cast<Real>(16.0)) {
      warm_cycles = std::min(warm_cycles, 1);
    } else if (defect_ratio <= static_cast<Real>(1024.0)) {
      warm_cycles = std::min(warm_cycles, 3);
    }
    niter_ = warm_cycles;
  }
  if (use_amr_reduced_exchange && is_last_stage) {
    // Reduced-exchange coarser levels need one extra finest-level post sweep to
    // recover the defect floor lost to cheaper halo traffic on PCIe-era MPI runs.
    npostsmooth_ = std::max(npostsmooth_, 3);
  }
  // The tail of every V-cycle is one finest-level sweep against exact halos.  Running
  // seven instead was measured both ways.  On a five-level Bonnor-Ebert collapse on two
  // ranks (16^3 blocks, a few hundred per rank) it took 14% off the V-cycle count for 1%
  // of the V-cycle cost, because that hierarchy spends its time in the host-side octet
  // levels.  On the ten-level Star-BH deck on ten ranks (185 blocks of 32^3 per rank) the
  // finest level is the cost, and the six extra sweeps and halo exchanges added 83% to
  // every V-cycle to save 6% of them: 15.5 s of multigrid per 20 cycles against 8.5 s
  // with one pass, and 9.7 s for the parent commit.  Production looks like the second
  // case, so the count is one.
  finest_exact_polish_passes_ = 1;
  if (timing_enabled) solve_timer.reset();
  if (!full_multigrid_)
    // Reuse the previous potential as the initial guess for iterative solves.
    mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);
  if (timing_enabled) {
    Kokkos::fence();
    t_load_phi = solve_timer.seconds();
  }

  if (timing_enabled) solve_timer.reset();
  SetupMultigrid(dt, false);
  if (timing_enabled) {
    Kokkos::fence();
    t_setup = solve_timer.seconds();
  }
  if (mporder_ > 0) {
    if (autompo_) CalculateCenterOfMass();
    CalculateMultipoleCoefficients();
    SyncMultipoleToDevice();
  }
  if (full_multigrid_) {
    // FMG only ever restricts the source on the way down, so whatever u the coarse
    // levels are holding when the bottom solve runs is added to the answer it returns.
    // Zero the hierarchy so that the FAS right-hand side is the restricted source and
    // nothing else -- on a fresh or remapped run that also keeps the first
    // StoreOldData()/correction pass from seeding corrections out of stale allocation
    // contents.
    for (int lev = 0; lev < nrootlevel_; ++lev) {
      mgroot_->SetCurrentLevel(lev);
      mgroot_->ZeroClearData();
    }
    mgroot_->SetCurrentLevel(nrootlevel_ - 1);
    if (nreflevel_ > 0) {
      ZeroClearOctets();
    }
    root_flat_buf_stale_ = true;
    root_uold_buf_valid_ = false;
  }
  distributed_coarse_solve_ =
      prefer_distributed_coarse_solve_ &&
      SupportsDistributedCoarseSync() &&
      (nranks_ > 1) && (nreflevel_ > 0);
  coarse_owner_rank_ = 0;

  ResetSolveCounters();
  collect_phase_timing_ = timing_enabled;
  if (timing_enabled) solve_timer.reset();
  if (full_multigrid_)
    SolveFMG(pdriver);
  else
    SolveMG(pdriver);
  collect_phase_timing_ = false;
  if (timing_enabled) {
    Kokkos::fence();
    t_mg_solve = solve_timer.seconds();
  }

  full_multigrid_ = saved_full_multigrid;
  eps_ = saved_eps;
  niter_ = saved_niter;
  npostsmooth_ = saved_npostsmooth;
  finest_exact_polish_passes_ = saved_finest_exact_polish_passes;
  fshowdef_ = saved_showdef;

  const bool track_stage_defect =
      is_last_stage && ((fshowdef_ != 0) || (saved_eps == 0.0 && saved_niter <= 0));
  if (timing_enabled) solve_timer.reset();
  if (track_stage_defect) {
    Real defect_norm = last_defect_valid_ ? last_defect_norm_
                                          : CalculateDefectNorm(MGNormType::l2, 0);
    if (report_stage_defect) {
      if (global_variable::my_rank == 0) {
        std::cout << "MGGravityDriver::Solve: Final defect norm = " << defect_norm
                  << std::endl;
      }
    }
  }
  if (timing_enabled) {
    Kokkos::fence();
    t_defect = solve_timer.seconds();
  }

  if (timing_enabled) solve_timer.reset();
  // The refresh below ends in a RetrieveResult of its own, so pulling the solution out
  // here first would only be handing it back a copy of what the finest level already
  // holds -- and its LoadFinestData would then copy that straight back in.
  if (!RefreshFinestGhosts(pgrav->phi, indcs_.ng, true)) {
    pgrav->MarkPhiInvalid();
    std::cout << "### ERROR in MGGravityDriver::Solve" << std::endl
              << "Failed to refresh numerical self-gravity phi ghosts after MG solve."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (timing_enabled) {
    Kokkos::fence();
    t_retrieve = solve_timer.seconds();
  }
  pgrav->phi_valid = true;
  pgrav->MarkSelfPhiValid();
  if (is_last_stage) {
    // A full solve has now run under the post-refinement policy, so release the sticky
    // flag.  Both RK stages of the first cycle after a regrid see mesh_changed = true.
    ConsumeAMRMeshChanged();
  }
  if (use_solve_dt_ && is_last_stage) {
    // The due times sit on the fixed grid k*solve_dt from t = 0, not one solve_dt after
    // whichever cycle happened to solve.  Anchoring to the cycle made the schedule a
    // function of the run's history: a run restarted from a checkpoint solved at its own
    // first cycle and then every solve_dt after that, two cycles out of phase with the
    // uninterrupted run, and the frozen-potential error of the two runs differed by a
    // few 1e-3 in the kinetic energy for the rest of the run.  On the grid the schedule
    // is a function of time alone and the restart solves at the same cycles.
    const Real ref_time = std::max(std::abs(cycle_time), solve_dt_);
    const Real time_tol = static_cast<Real>(64.0) * std::numeric_limits<Real>::epsilon() *
                          std::max(static_cast<Real>(1.0), ref_time);
    const Real k = std::floor((cycle_time + time_tol) / solve_dt_);
    next_solve_time_ = (k + static_cast<Real>(1.0)) * solve_dt_;
  }
  if (timing_enabled && global_variable::my_rank == 0) {
    std::cout << "MGGravityDriver::Timing:"
              << " cycle=" << ncycle
              << " stage=" << stage
              << " entry=" << t_entry
              << " load_source=" << t_load_source
              << " load_phi=" << t_load_phi
              << " setup=" << t_setup
              << " mg=" << t_mg_solve
              << " defect=" << t_defect
              << " retrieve=" << t_retrieve
              << " vcycles=" << timing_vcycles_
              << " coarsest=" << timing_coarsest_solves_
              << " defect_norms=" << timing_defect_norms_
              << " to_coarser=" << timing_to_coarser_time_
              << " to_finer=" << timing_to_finer_time_
              << " bottom=" << timing_coarsest_time_
              << " fmg_restrict=" << timing_fmg_coarser_time_
              << " fmg_prolong=" << timing_fmg_prolongate_time_
              << " blocks_to_root=" << timing_transfer_blocks_root_time_
              << " root_to_blocks=" << timing_transfer_root_blocks_time_
              << " octets=" << timing_octets_time_
              << std::endl;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::RunReciprocityTest(Driver *pdriver)
//! \brief Double-precision reciprocity diagnostic of the converged multi-level solve.
//!
//! The energy ledger identity dU = sum(drho (phi0+phi1)/2 dV) holds exactly only if the
//! composite discrete operator L (7-point Laplacian on every leaf cell, ghosts at
//! coarse/fine faces from MultigridBoundaryValues::FillFineCoarseMGGhosts) is symmetric
//! in the dV-weighted inner product, i.e. iff sum(rhoA' phiB dV) == sum(rhoB' phiA dV)
//! for any two sources (rho' = rho - <rho>).  This computes both sums, in double
//! precision with Kokkos reductions and an MPI_Allreduce, for the initial density (A)
//! and a deterministic modification of it (B).  Every rho and phi is the array the
//! solver actually used or produced; the finest-level multigrid source is summed as
//! well, as a check that the rho copies match the source the solve saw.
//!
//! With reciprocity_operator_test the operator itself is applied to two smooth fields:
//! RefreshFinestGhosts fills every ghost the way the smoother sees it, and <v,Lu> and
//! <u,Lv> are formed with the dV/dx^2 weight that turns the code's dx^2-scaled stencil
//! into the flux form.  Per-level partial sums are printed as well.

void MGGravityDriver::RunReciprocityTest(Driver *pdriver) {
  auto *pgrav = pmy_pack_->pgrav;
  Mesh *pm = pmy_pack_->pmesh;
  if (pgrav == nullptr) return;
  const RegionIndcs &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nc1 = nx1 + 2*ng, nc2 = nx2 + 2*ng, nc3 = nx3 + 2*ng;
  const int is = ng, js = ng, ks = ng;
  const int ie = is + nx1 - 1, je = js + nx2 - 1, ke = ks + nx3 - 1;
  const int nmb = pmy_pack_->nmb_thispack;
  const int my_rank = global_variable::my_rank;
  DvceArray5D<Real> *fluid = nullptr;
  if (pmy_pack_->phydro != nullptr) {
    fluid = &(pmy_pack_->phydro->u0);
  } else if (pmy_pack_->pmhd != nullptr) {
    fluid = &(pmy_pack_->pmhd->u0);
  } else {
    if (my_rank == 0) {
      std::cout << "MGGravityDriver::RunReciprocityTest: no fluid state, skipped"
                << std::endl;
    }
    return;
  }
  auto &u0 = *fluid;
  auto &size = pmy_pack_->pmb->mb_size;
  auto &mblev = pmy_pack_->pmb->mb_lev;
  const Real x1min = pm->mesh_size.x1min, x1max = pm->mesh_size.x1max;
  const Real x2min = pm->mesh_size.x2min, x2max = pm->mesh_size.x2max;
  const Real x3min = pm->mesh_size.x3min, x3max = pm->mesh_size.x3max;
  const Real lx = x1max - x1min, ly = x2max - x2min, lz = x3max - x3min;
  const Real pi = 3.14159265358979323846;

  auto allreduce_sum = [](Real *v, int n) {
#if MPI_PARALLEL_ENABLED
    std::vector<Real> tmp(v, v + n);
    MPI_Allreduce(tmp.data(), v, n, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  };
  auto allreduce_max = [](Real *v, int n) {
#if MPI_PARALLEL_ENABLED
    std::vector<Real> tmp(v, v + n);
    MPI_Allreduce(tmp.data(), v, n, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  };

  // Cell geometry.  dV = dx dy dz; the code's stencil is scaled by dx^2 (wy, wz carry
  // the anisotropy), so dV/dx^2 = dy dz converts a stencil value to flux form.
  auto size_d = size.d_view;
  auto cell_x = [=] KOKKOS_FUNCTION (int m, int i) {
    return CellCenterX(i - is, nx1, size_d(m).x1min, size_d(m).x1max);
  };
  auto cell_y = [=] KOKKOS_FUNCTION (int m, int j) {
    return CellCenterX(j - js, nx2, size_d(m).x2min, size_d(m).x2max);
  };
  auto cell_z = [=] KOKKOS_FUNCTION (int m, int k) {
    return CellCenterX(k - ks, nx3, size_d(m).x3min, size_d(m).x3max);
  };
  auto cell_dv = [=] KOKKOS_FUNCTION (int m) {
    const Real dx = (size_d(m).x1max - size_d(m).x1min)/static_cast<Real>(nx1);
    const Real dy = (size_d(m).x2max - size_d(m).x2min)/static_cast<Real>(nx2);
    const Real dz = (size_d(m).x3max - size_d(m).x3min)/static_cast<Real>(nx3);
    return dx*dy*dz;
  };
  // The code's stencil is -dx^2 lap(u) (wy, wz carry any anisotropy), so the weight
  // that turns sum(v stencil(u)) into the flux form sum(v (-lap u) dV) is dV/dx^2.
  auto cell_wflux = [=] KOKKOS_FUNCTION (int m) {
    const Real dx = (size_d(m).x1max - size_d(m).x1min)/static_cast<Real>(nx1);
    const Real dy = (size_d(m).x2max - size_d(m).x2min)/static_cast<Real>(nx2);
    const Real dz = (size_d(m).x3max - size_d(m).x3min)/static_cast<Real>(nx3);
    return dy*dz/dx;
  };
  using Policy4 = Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>;
  Policy4 active({0, ks, js, is}, {nmb, ke+1, je+1, ie+1});

  // Volume-weighted sum and total volume of a (nmb,1,nc3,nc2,nc1) field.
  auto vol_sum = [&](const DvceArray5D<Real> &f) {
    Real stats[2] = {0.0, 0.0};
    Kokkos::parallel_reduce("recip_volsum", active,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    Real &s0, Real &s1) {
        const Real dv = cell_dv(m);
        s0 += f(m,0,k,j,i)*dv;
        s1 += dv;
      }, Kokkos::Sum<Real>(stats[0]), Kokkos::Sum<Real>(stats[1]));
    allreduce_sum(stats, 2);
    return std::make_pair(stats[0], stats[1]);
  };
  // sum((a - amean) b dV)
  auto pair_sum = [&](const DvceArray5D<Real> &a, Real amean,
                      const DvceArray5D<Real> &b) {
    Real s = 0.0;
    Kokkos::parallel_reduce("recip_pairsum", active,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &ls) {
        ls += (a(m,0,k,j,i) - amean)*b(m,0,k,j,i)*cell_dv(m);
      }, Kokkos::Sum<Real>(s));
    allreduce_sum(&s, 1);
    return s;
  };
  auto copy_active = [&](const DvceArray5D<Real> &src, int ns, int src_ng,
                         DvceArray5D<Real> &dst) {
    const int off = src_ng - ng;
    par_for("recip_copy", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      dst(m,0,k,j,i) = src(m,ns,k+off,j+off,i+off);
    });
  };

  DvceArray5D<Real> rhoA("recip_rhoA", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> rhoB("recip_rhoB", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> phiA("recip_phiA", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> phiB("recip_phiB", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> srcA("recip_srcA", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> srcB("recip_srcB", nmb, 1, nc3, nc2, nc1);
  copy_active(u0, IDN, ng, rhoA);

  auto solve_state = [&](DvceArray5D<Real> &phi_out, DvceArray5D<Real> &src_out) {
    // A cold solve each time: no warm start, no cadence skip, so both states are
    // converged to the deck's threshold by the same path.
    pgrav->MarkPhiInvalid();
    ResetCadenceCache();
    MarkSolveDueAt(pm->time);
    Solve(pdriver, pdriver->nexp_stages, pm->dt);
    if (!pgrav->phi_valid) {
      std::cout << "### FATAL ERROR in MGGravityDriver::RunReciprocityTest: the solve "
                << "did not run (cadence or window guard)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    copy_active(pgrav->phi, 0, ng, phi_out);
    mglevels_->SetCurrentLevel(mglevels_->GetNumberOfLevels() - 1);
    DvceArray5D<Real> src = mglevels_->GetCurrentSource();
    copy_active(src, 0, mglevels_->GetGhostCells(), src_out);
  };

  if (my_rank == 0) {
    std::cout << std::setprecision(17)
              << "RECIPROCITY: nmb_total=" << pm->nmb_total
              << " root_level=" << pm->root_level
              << " max_level=" << pm->max_level
              << " threshold=" << eps_ << " niteration=" << niter_
              << " subtract_average=" << fsubtract_average_
              << " mg_fc_symmetric=" << (fc_tangential_gradient_ ? 0 : 1)
              << " mod_amp=" << reciprocity_mod_amp_
              << " blob_amp=" << reciprocity_blob_amp_ << std::endl;
  }

  // ---- state A: the initial density
  solve_state(phiA, srcA);
  const int vcycles_A = timing_vcycles_;
  if (!reciprocity_dump_.empty()) {
    // Raw double-precision dump of state A, one file per rank: for every active cell
    // (x, y, z, dV, rho, phi).  Two runs of different initial conditions on the same
    // mesh can then be paired offline in double precision
    // (tst/bh_trajectory/reciprocity_pair.py).
    auto rhoA_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhoA);
    auto phiA_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phiA);
    auto size_h = size.h_view;
    std::string fname = reciprocity_dump_ + ".rank" + std::to_string(my_rank) + ".bin";
    FILE *fp = std::fopen(fname.c_str(), "wb");
    if (fp == nullptr) {
      std::cout << "### FATAL ERROR in MGGravityDriver::RunReciprocityTest: cannot open "
                << fname << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::vector<double> rec(6);
    for (int m = 0; m < nmb; ++m) {
      const double dx = (size_h(m).x1max - size_h(m).x1min)/nx1;
      const double dy = (size_h(m).x2max - size_h(m).x2min)/nx2;
      const double dz = (size_h(m).x3max - size_h(m).x3min)/nx3;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            rec[0] = CellCenterX(i - is, nx1, size_h(m).x1min, size_h(m).x1max);
            rec[1] = CellCenterX(j - js, nx2, size_h(m).x2min, size_h(m).x2max);
            rec[2] = CellCenterX(k - ks, nx3, size_h(m).x3min, size_h(m).x3max);
            rec[3] = dx*dy*dz;
            rec[4] = rhoA_h(m,0,k,j,i);
            rec[5] = phiA_h(m,0,k,j,i);
            std::fwrite(rec.data(), sizeof(double), 6, fp);
          }
        }
      }
    }
    std::fclose(fp);
    if (my_rank == 0) {
      std::cout << "RECIPROCITY: state A dumped to " << reciprocity_dump_
                << ".rank*.bin" << std::endl;
    }
  }

  // ---- state B: rhoA (1 + a cos cos cos) + b rho_max exp(-r^2/w^2)
  Real rho_max = 0.0;
  Kokkos::parallel_reduce("recip_rhomax", active,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lm) {
      lm = (lm > rhoA(m,0,k,j,i)) ? lm : rhoA(m,0,k,j,i);
    }, Kokkos::Max<Real>(rho_max));
  allreduce_max(&rho_max, 1);
  {
    const Real a = reciprocity_mod_amp_;
    const Real b = reciprocity_blob_amp_*rho_max;
    const Real xb = reciprocity_blob_center_[0], yb = reciprocity_blob_center_[1],
               zb = reciprocity_blob_center_[2];
    const Real w2 = reciprocity_blob_width_*reciprocity_blob_width_;
    par_for("recip_rhoB", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = cell_x(m,i), y = cell_y(m,j), z = cell_z(m,k);
      const Real mod = 1.0 + a*cos(2.0*pi*(x - x1min)/lx)*cos(2.0*pi*(y - x2min)/ly)
                              *cos(2.0*pi*(z - x3min)/lz);
      const Real r2 = (x-xb)*(x-xb) + (y-yb)*(y-yb) + (z-zb)*(z-zb);
      const Real val = rhoA(m,0,k,j,i)*mod + b*exp(-r2/w2);
      rhoB(m,0,k,j,i) = val;
      u0(m,IDN,k,j,i) = val;
    });
  }
  solve_state(phiB, srcB);
  const int vcycles_B = timing_vcycles_;
  // restore the fluid density
  par_for("recip_restore", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u0(m,IDN,k,j,i) = rhoA(m,0,k,j,i);
  });

  // ---- reductions
  // The constant is in the null space only when the solver subtracted the source
  // average (all-Neumann / periodic); with a fixed-potential or multipole face the
  // identity to test is the raw sum(rhoA phiB dV) = sum(rhoB phiA dV).
  auto sa = vol_sum(rhoA), sb = vol_sum(rhoB);
  const bool subtract = finest_source_average_subtracted_;
  const Real meanA = subtract ? sa.first/sa.second : 0.0;
  const Real meanB = subtract ? sb.first/sb.second : 0.0;
  const Real S_AB = pair_sum(rhoA, meanA, phiB);
  const Real S_BA = pair_sum(rhoB, meanB, phiA);
  const Real S_AA = pair_sum(rhoA, meanA, phiA);
  const Real S_BB = pair_sum(rhoB, meanB, phiB);
  // the solver's own finest source (already -4piG (rho - <rho>) if the average was
  // subtracted): src-weighted sums, scaled back to density units
  const Real T_AB = pair_sum(srcA, 0.0, phiB)/(-four_pi_G_);
  const Real T_BA = pair_sum(srcB, 0.0, phiA)/(-four_pi_G_);
  const Real scale = std::max(std::abs(S_AB), std::abs(S_BA));
  const Real tscale = std::max(std::abs(T_AB), std::abs(T_BA));
  if (my_rank == 0) {
    std::cout << std::setprecision(17)
              << "RECIPROCITY: vcycles_A=" << vcycles_A << " vcycles_B=" << vcycles_B
              << std::endl
              << "RECIPROCITY: <rhoA>=" << meanA << " <rhoB>=" << meanB
              << " volume=" << sa.second << " rho_max=" << rho_max
              << " mean_subtracted=" << subtract << std::endl
              << "RECIPROCITY: S_AA=" << S_AA << " S_BB=" << S_BB << std::endl
              << "RECIPROCITY: S_AB=" << S_AB << " S_BA=" << S_BA
              << " |S_AB-S_BA|/max=" << std::abs(S_AB - S_BA)/scale << std::endl
              << "RECIPROCITY(src): T_AB=" << T_AB << " T_BA=" << T_BA
              << " |T_AB-T_BA|/max=" << std::abs(T_AB - T_BA)/tscale << std::endl;
  }

  if (!reciprocity_operator_test_) return;

  // ---- direct operator symmetry test on two smooth fields
  DvceArray5D<Real> uf("recip_u", nmb, 1, nc3, nc2, nc1);
  DvceArray5D<Real> vf("recip_v", nmb, 1, nc3, nc2, nc1);
  par_for("recip_uv", DevExeSpace(), 0, nmb-1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = cell_x(m,i), y = cell_y(m,j), z = cell_z(m,k);
    const Real cx = cos(2.0*pi*(x - x1min)/lx), cy = cos(2.0*pi*(y - x2min)/ly);
    const Real cz = cos(2.0*pi*(z - x3min)/lz);
    const Real sx = sin(2.0*pi*(x - x1min)/lx), sy2 = sin(4.0*pi*(y - x2min)/ly);
    uf(m,0,k,j,i) = cx*cy*cz
        + 0.5*exp(-((x-2.0)*(x-2.0) + (y-1.0)*(y-1.0) + (z+1.0)*(z+1.0))/9.0);
    vf(m,0,k,j,i) = sx*sy2*cz
        + exp(-((x+3.0)*(x+3.0) + (y-2.0)*(y-2.0) + (z-1.0)*(z-1.0))/16.0);
  });
  // Fill every ghost (same level, physical, coarse/fine) exactly as the smoother and
  // the defect see them, then read the arrays back with that one ghost layer.  On a
  // uniform periodic mesh this is a plain periodic 7-point stencil and the two sums agree
  // to round-off; any difference beyond that is the coarse/fine interface treatment.
  RefreshFinestGhosts(uf, ng, false);
  RefreshFinestGhosts(vf, ng, false);
  const Real wy = gravity_wy_, wz = gravity_wz_, diag = gravity_diag_;
  const int root_level = pm->root_level;
  auto mblev_d = mblev.d_view;
  const int nlev = pm->max_level - root_level + 1;
  Real vLu_tot = 0.0, uLv_tot = 0.0;
  std::vector<Real> vLu_lev(nlev, 0.0), uLv_lev(nlev, 0.0);
  for (int l = 0; l < nlev; ++l) {
    const int lev = root_level + l;
    Real s[2] = {0.0, 0.0};
    Kokkos::parallel_reduce("recip_oper", active,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    Real &s0, Real &s1) {
        if (mblev_d(m) != lev) return;
        const Real w = cell_wflux(m);
        const Real Lu = ApplyGravityLaplacian(uf, m, 0, k, j, i, wy, wz, diag);
        const Real Lv = ApplyGravityLaplacian(vf, m, 0, k, j, i, wy, wz, diag);
        s0 += vf(m,0,k,j,i)*Lu*w;
        s1 += uf(m,0,k,j,i)*Lv*w;
      }, Kokkos::Sum<Real>(s[0]), Kokkos::Sum<Real>(s[1]));
    allreduce_sum(s, 2);
    vLu_lev[l] = s[0]; uLv_lev[l] = s[1];
    vLu_tot += s[0]; uLv_tot += s[1];
  }
  if (my_rank == 0) {
    std::cout << std::setprecision(17);
    for (int l = 0; l < nlev; ++l) {
      std::cout << "OPERATOR: level=" << (root_level + l) << " <v,Lu>=" << vLu_lev[l]
                << " <u,Lv>=" << uLv_lev[l] << " diff=" << (vLu_lev[l] - uLv_lev[l])
                << std::endl;
    }
    const Real oscale = std::max(std::abs(vLu_tot), std::abs(uLv_tot));
    std::cout << "OPERATOR: total <v,Lu>=" << vLu_tot << " <u,Lv>=" << uLv_tot
              << " |diff|/max=" << std::abs(vLu_tot - uLv_tot)/oscale << std::endl;
  }
}

void MGGravity::SmoothPackBox(int color, int is, int ie, int js, int je, int ks, int ke) {
  if (is > ie || js > je || ks > ke) return;
  auto *gdrv = static_cast<MGGravityDriver*>(pmy_driver_);
  int ll = nlevel_-1-current_level_;
  const bool ca_mode = (gdrv->nranks_ > 1 &&
                        gdrv->ActiveReducedSameExchange() &&
                        gdrv->ca_pack_smoother_type_ != 0);
  if (ca_mode) {
    const int c = color ^ pmy_driver_->GetCoffset();
    if (c != 0) return;

    const Real inv_diag = 1.0 / gdrv->gravity_diag_;
    Real w0 = gdrv->omega_ * inv_diag;
    Real w1 = w0;
    int stages = 1;
    if (gdrv->ca_pack_smoother_type_ == 2) {
      constexpr Real pi = 3.1415926535897932384626433832795;
      const Real lmin = std::max(static_cast<Real>(1.0e-6),
          gdrv->ca_pack_cheb_lambda_min_);
      const Real lmax = std::max(lmin + static_cast<Real>(1.0e-6),
          gdrv->ca_pack_cheb_lambda_max_);
      const Real d = 0.5 * (lmax + lmin);
      const Real cl = 0.5 * (lmax - lmin);
      const Real omega0 = 1.0 / (d - cl * std::cos(0.25 * pi));
      const Real omega1 = 1.0 / (d - cl * std::cos(0.75 * pi));
      w0 = omega0 * inv_diag;
      w1 = omega1 * inv_diag;
      stages = 2;
    }

    const int rlev = -ll;
    const int il_l = is, iu_l = ie;
    const int jl_l = js, ju_l = je;
    const int kl_l = ks, ku_l = ke;
    const Real w0_l = w0, w1_l = w1;
    const Real wy_l = gdrv->gravity_wy_;
    const Real wz_l = gdrv->gravity_wz_;
    const Real diag_l = gdrv->gravity_diag_;

    if (on_host_) {
      auto u = u_[current_level_].h_view;
      auto src = src_[current_level_].h_view;
      auto utmp = def_[current_level_].h_view;  // scratch for CA smoother
      auto brdx = block_rdx_.h_view;

      par_for("MGGravity::SmoothPackCAStage0", HostExeSpace(),
              0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                              : brdx(m) / static_cast<Real>(1<<rlev);
        Real dx2 = dx * dx;
        Real lap = ApplyGravityLaplacian(u, m, 0, k, j, i, wy_l, wz_l, diag_l);
        utmp(m,0,k,j,i) = u(m,0,k,j,i) + (src(m,0,k,j,i)*dx2 - lap) * w0_l;
      });
      if (stages == 1) {
        par_for("MGGravity::SmoothPackCACommit", HostExeSpace(),
                0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          u(m,0,k,j,i) = utmp(m,0,k,j,i);
        });
      } else {
        par_for("MGGravity::SmoothPackCAStage1", HostExeSpace(),
                0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                                : brdx(m) / static_cast<Real>(1<<rlev);
          Real dx2 = dx * dx;
          const Real c0 = utmp(m,0,k,j,i);
          const Real xp = (i == iu_l) ? u(m,0,k,j,i+1) : utmp(m,0,k,j,i+1);
          const Real xm = (i == il_l) ? u(m,0,k,j,i-1) : utmp(m,0,k,j,i-1);
          const Real yp = (j == ju_l) ? u(m,0,k,j+1,i) : utmp(m,0,k,j+1,i);
          const Real ym = (j == jl_l) ? u(m,0,k,j-1,i) : utmp(m,0,k,j-1,i);
          const Real zp = (k == ku_l) ? u(m,0,k+1,j,i) : utmp(m,0,k+1,j,i);
          const Real zm = (k == kl_l) ? u(m,0,k-1,j,i) : utmp(m,0,k-1,j,i);
          Real lap;
          if (wy_l == 1.0 && wz_l == 1.0) {
            lap = 6.0*c0 - xp - xm - yp - ym - zp - zm;
          } else {
            lap = diag_l*c0 - xp - xm - wy_l*(yp + ym) - wz_l*(zp + zm);
          }
          u(m,0,k,j,i) = c0 + (src(m,0,k,j,i)*dx2 - lap) * w1_l;
        });
      }
    } else {
      auto u = u_[current_level_].d_view;
      auto src = src_[current_level_].d_view;
      auto utmp = def_[current_level_].d_view;  // scratch for CA smoother
      auto brdx = block_rdx_.d_view;

      par_for("MGGravity::SmoothPackCAStage0", DevExeSpace(),
              0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                              : brdx(m) / static_cast<Real>(1<<rlev);
        Real dx2 = dx * dx;
        Real lap = ApplyGravityLaplacian(u, m, 0, k, j, i, wy_l, wz_l, diag_l);
        utmp(m,0,k,j,i) = u(m,0,k,j,i) + (src(m,0,k,j,i)*dx2 - lap) * w0_l;
      });
      if (stages == 1) {
        par_for("MGGravity::SmoothPackCACommit", DevExeSpace(),
                0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          u(m,0,k,j,i) = utmp(m,0,k,j,i);
        });
      } else {
        par_for("MGGravity::SmoothPackCAStage1", DevExeSpace(),
                0, nmmb_-1, kl_l, ku_l, jl_l, ju_l, il_l, iu_l,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real dx = (rlev <= 0) ? brdx(m) * static_cast<Real>(1<<(-rlev))
                                : brdx(m) / static_cast<Real>(1<<rlev);
          Real dx2 = dx * dx;
          const Real c0 = utmp(m,0,k,j,i);
          const Real xp = (i == iu_l) ? u(m,0,k,j,i+1) : utmp(m,0,k,j,i+1);
          const Real xm = (i == il_l) ? u(m,0,k,j,i-1) : utmp(m,0,k,j,i-1);
          const Real yp = (j == ju_l) ? u(m,0,k,j+1,i) : utmp(m,0,k,j+1,i);
          const Real ym = (j == jl_l) ? u(m,0,k,j-1,i) : utmp(m,0,k,j-1,i);
          const Real zp = (k == ku_l) ? u(m,0,k+1,j,i) : utmp(m,0,k+1,j,i);
          const Real zm = (k == kl_l) ? u(m,0,k-1,j,i) : utmp(m,0,k-1,j,i);
          Real lap;
          if (wy_l == 1.0 && wz_l == 1.0) {
            lap = 6.0*c0 - xp - xm - yp - ym - zp - zm;
          } else {
            lap = diag_l*c0 - xp - xm - wy_l*(yp + ym) - wz_l*(zp + zm);
          }
          u(m,0,k,j,i) = c0 + (src(m,0,k,j,i)*dx2 - lap) * w1_l;
        });
      }
    }
    return;
  }

  GravityStencil stencil{gdrv->omega_/gdrv->gravity_diag_, gdrv->gravity_wy_,
                         gdrv->gravity_wz_, gdrv->gravity_diag_};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color);
  }
}

void MGGravity::SmoothPack(int color) {
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  SmoothPackBox(color, is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn void MGGravity::SmoothPackCoarsestEffectiveDiagonal(int color, const Real *bface)
//! \brief Red-black sweep of the current root level against each cell's own diagonal.
//!
//! bface[f] is what face f adds to the diagonal of a cell that touches it: +w_f where the
//! ghost reflects the cell back with its own sign, -w_f where it reflects it with the
//! opposite one, and zero where the ghost is a genuine neighbour.  The arithmetic is
//! Multigrid::Smooth's, with its uniform omega/D replaced by omega/(D + b) of the cell.
//! The root grid is one block whose colour parity is zero, so the colouring reduces to
//! the parity of i + j + k.

void MGGravity::SmoothPackCoarsestEffectiveDiagonal(int color, const Real *bface) {
  auto *gdrv = static_cast<MGGravityDriver*>(pmy_driver_);
  const int ll = nlevel_ - 1 - current_level_;
  const int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  const int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  const int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  const Real dx = rdx_ * static_cast<Real>(1 << ll);
  const Real dx2 = dx*dx;
  const Real omega = gdrv->omega_;
  const Real wy = gdrv->gravity_wy_, wz = gdrv->gravity_wz_;
  const Real diag = gdrv->gravity_diag_;
  const Real bx0 = bface[0], bx1 = bface[1], by0 = bface[2];
  const Real by1 = bface[3], bz0 = bface[4], bz1 = bface[5];
  const int c0 = color ^ pmy_driver_->GetCoffset();

  if (on_host_) {
    SmoothGravityEffectiveDiagonal(u_[current_level_].h_view,
        src_[current_level_].h_view, nmmb_, is, ie, js, je, ks, ke, c0, dx2,
        omega, wy, wz, diag, bx0, bx1, by0, by1, bz0, bz1);
  } else {
    SmoothGravityEffectiveDiagonal(u_[current_level_].d_view,
        src_[current_level_].d_view, nmmb_, is, ie, js, je, ks, ke, c0, dx2,
        omega, wy, wz, diag, bx0, bx1, by0, by1, bz0, bz1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGGravity::SmoothPackExpanded(int color, int grow)
//! \brief Smooth one colour over the interior expanded by `grow` ghost layers.
//!
//! Redundantly updating the halo is what buys the exact communication-avoiding pair in
//! MultigridDriver::OneStepToCoarser its missing exchange: with two valid ghost layers
//! the expanded sweep computes, in this block's ghost layer, exactly the values the
//! neighbour computes in its own interior.  The colouring is a function of the absolute
//! index -- Smooth() derives it from parity_origin = ngh_ and the block parity -- so the
//! expanded box keeps the same checkerboard as the interior box.

void MGGravity::SmoothPackExpanded(int color, int grow) {
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  SmoothPackBox(color, is - grow, ie + grow, js - grow, je + grow,
                ks - grow, ke + grow);
}

void MGGravity::SmoothPackInterior(int color) {
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  SmoothPackBox(color, is + 1, ie - 1, js + 1, je - 1, ks + 1, ke - 1);
}

void MGGravity::SmoothPackBoundary(int color) {
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;

  auto *gdrv = static_cast<MGGravityDriver*>(pmy_driver_);
  const bool ca_mode = (gdrv->nranks_ > 1 &&
                        gdrv->ActiveReducedSameExchange() &&
                        gdrv->ca_pack_smoother_type_ != 0);
  if (!ca_mode) {
    const Real omega_over_diag = gdrv->omega_ / gdrv->gravity_diag_;
    const int rlev = -ll;
    const int block_cells_odd = (indcs_.nx1 >> ll) & 1;
    if (on_host_) {
      SmoothGravityBoundaryFlat(u_[current_level_].h_view, src_[current_level_].h_view,
                                block_rdx_.h_view, block_color_parity_.h_view,
                                block_cells_odd, nmmb_, is, ie, js, je, ks, ke,
                                color, pmy_driver_->GetCoffset(), rlev,
                                omega_over_diag, gdrv->gravity_wy_, gdrv->gravity_wz_,
                                gdrv->gravity_diag_);
    } else {
      SmoothGravityBoundaryFlat(u_[current_level_].d_view, src_[current_level_].d_view,
                                block_rdx_.d_view, block_color_parity_.d_view,
                                block_cells_odd, nmmb_, is, ie, js, je, ks, ke,
                                color, pmy_driver_->GetCoffset(), rlev,
                                omega_over_diag, gdrv->gravity_wy_, gdrv->gravity_wz_,
                                gdrv->gravity_diag_);
    }
    return;
  }

  SmoothPackBox(color, is, is, js, je, ks, ke);
  if (ie > is) {
    SmoothPackBox(color, ie, ie, js, je, ks, ke);
  }
  if (ie - is >= 2) {
    SmoothPackBox(color, is + 1, ie - 1, js, js, ks, ke);
    if (je > js) {
      SmoothPackBox(color, is + 1, ie - 1, je, je, ks, ke);
    }
  }
  if (ie - is >= 2 && je - js >= 2) {
    SmoothPackBox(color, is + 1, ie - 1, js + 1, je - 1, ks, ks);
    if (ke > ks) {
      SmoothPackBox(color, is + 1, ie - 1, js + 1, je - 1, ke, ke);
    }
  }
}

void MGGravity::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  auto *gdrv = static_cast<MGGravityDriver*>(pmy_driver_);
  GravityStencil stencil{0.0, gdrv->gravity_wy_, gdrv->gravity_wz_,
                         gdrv->gravity_diag_};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke);
  }
}

void MGGravity::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  auto *gdrv = static_cast<MGGravityDriver*>(pmy_driver_);
  GravityStencil stencil{0.0, gdrv->gravity_wy_, gdrv->gravity_wz_,
                         gdrv->gravity_diag_};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke);
  }
}


//----------------------------------------------------------------------------------------
// Host-side octet physics for MGGravityDriver

static inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i,
                                Real wy, Real wz, Real diag) {
  if (wy == 1.0 && wz == 1.0) {
    return 6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
           - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
           - o.U(v,k,j,i-1);
  }
  return diag*o.U(v,k,j,i) - o.U(v,k,j,i+1) - o.U(v,k,j,i-1)
         - wy*(o.U(v,k,j+1,i) + o.U(v,k,j-1,i))
         - wz*(o.U(v,k+1,j,i) + o.U(v,k-1,j,i));
}

//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::SolveCoarsestGrid()
//! \brief Bottom solve, with the degenerate single-cell root grid solved directly

void MGGravityDriver::SolveCoarsestGrid() {
  // When the root grid coarsens all the way to one cell, that cell has no interior
  // neighbour: all six of its neighbours are ghosts set by a physical boundary.  A
  // Dirichlet-type face (zero-fixed, or the multipole potential) sets the ghost to
  // 2*phi_b - u, so the reflection folds the cell's own value back onto the diagonal and
  // the effective diagonal becomes D + B rather than D, with
  //     B = sum over faces of w_f, counted +w_f on a Dirichlet face and -w_f otherwise.
  // With every face Dirichlet, B = D and the true diagonal is 2D, so relaxing with
  // omega/D realises a factor of 2*omega = 2.3 and each sweep amplifies the error by
  // |1 - 2*omega| = 1.3.  The base class then runs coarsest_min_sweeps of them.  That is
  // the reason the isolated-boundary decks in inputs/ carry a hand-set
  // coarsest_min_sweeps = 1 with a paragraph of explanation attached.
  //
  // Dividing by D + B instead makes the sweep exact rather than merely stable: it is one
  // equation in one unknown, and a single sweep with unit relaxation lands on its
  // solution.  Tomida & Stone (2023, sec. II.3.2) likewise solve this cell directly.
  const int ni = (std::max(nrbx1_, std::max(nrbx2_, nrbx3_)) >> (nrootlevel_ - 1));
  const bool degenerate = (ni == 1)
                          && ((nrbx1_ >> (nrootlevel_ - 1)) == 1)
                          && ((nrbx2_ >> (nrootlevel_ - 1)) == 1)
                          && ((nrbx3_ >> (nrootlevel_ - 1)) == 1);
  // A periodic root is singular for its own reason -- the constant nullspace the average
  // is subtracted against -- and the base class already zeroes the cell there, which is
  // the right representative of the solution family.  Its coarsest cells are also the
  // safe side of the same effect: a wrapped ghost that is the cell itself enters the
  // diagonal with a minus sign, so relaxing against D under-relaxes rather than over.
  if (fsubtract_average_) {
    MultigridDriver::SolveCoarsestGrid();
    return;
  }
  if (!degenerate) {
    SolveCoarsestGridEffectiveDiagonal();
    return;
  }

  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  bool counted = false;
  if (IsCoarseSolveOwner()) {
    const Real w[6] = {1.0, 1.0, gravity_wy_, gravity_wy_, gravity_wz_, gravity_wz_};
    Real b_sum = 0.0;
    for (int f = 0; f < 6; ++f) {
      const BoundaryFlag bc = mg_mesh_bcs_[f];
      const bool dirichlet = (bc == BoundaryFlag::mg_zerofixed
                              || bc == BoundaryFlag::mg_multipole);
      b_sum += dirichlet ? w[f] : -w[f];
    }
    // At least one face is Dirichlet here -- with none, the constructor forces
    // subtract_average and the base class handled the cell above -- so B > -D and the
    // effective diagonal is positive.
    const Real diag_eff = gravity_diag_ + b_sum;
    ++timing_coarsest_solves_;
    counted = true;

    pmg = mgroot_;
    MGRootBoundary();
    mgroot_->StoreOldData();
    mgroot_->CalculateFASRHSPack();
    // SmoothPack divides by gravity_diag_, so hand it the omega that turns that into a
    // division by the effective diagonal with unit relaxation.
    const Real saved_omega = omega_;
    omega_ = gravity_diag_ / diag_eff;
    mgroot_->SmoothPack(coffset_);
    MGRootBoundary();
    mgroot_->SmoothPack(1 - coffset_);
    omega_ = saved_omega;
  }
  if (!counted) ++timing_coarsest_solves_;
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_coarsest_time_ += timer.seconds();
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::SolveCoarsestGridEffectiveDiagonal()
//! \brief Bottom solve on a coarsest level that keeps an interior but still touches the
//!        domain boundary on most of its faces.
//!
//! A root grid that is not cubic does not coarsen to a single cell: an 8x4x4 root stops
//! at 2x1x1.  Each of those two cells has one interior neighbour and five faces on the
//! domain boundary, so under an isolated boundary its true diagonal is D + B = 11 while
//! SmoothPack divides by D = 6.  The relaxation factor the sweep realises is then
//! omega*(D + B)/D = 2.11, past the 2 at which Gauss-Seidel stops converging, and each
//! red-black pair multiplies the error by about 1.13.  Every extra bottom sweep makes the
//! solution worse, which is not a rate one can tune around.
//!
//! Dividing by each cell's own D + B puts the realised factor back at omega for every
//! cell, whatever mix of faces it touches.  That is the cheap half of the problem: the
//! remaining coupling between the interior cells is still relaxed, not solved, so the
//! bottom solve stays iterative -- but it is an iteration that contracts.

void MGGravityDriver::SolveCoarsestGridEffectiveDiagonal() {
  ++timing_coarsest_solves_;
  if (collect_phase_timing_) Kokkos::fence();
  Kokkos::Timer timer;
  if (!IsCoarseSolveOwner()) return;
  pmg = mgroot_;

  // Which faces reflect is a property of the level the sweep will run on, so the extents
  // come from where the root actually stands rather than from the level the descent is
  // expected to have left it at.
  const int ll = mgroot_->GetNumberOfLevels() - 1 - mgroot_->GetCurrentLevel();
  const int cnx1 = nrbx1_ >> ll;
  const int cnx2 = nrbx2_ >> ll;
  const int cnx3 = nrbx3_ >> ll;
  const Real w[6] = {1.0, 1.0, gravity_wy_, gravity_wy_, gravity_wz_, gravity_wz_};
  const int extent[6] = {cnx1, cnx1, cnx2, cnx2, cnx3, cnx3};
  Real bface[6];
  for (int f = 0; f < 6; ++f) {
    const BoundaryFlag bc = mg_mesh_bcs_[f];
    if (bc == BoundaryFlag::mg_zerofixed || bc == BoundaryFlag::mg_multipole) {
      // 2*phi_b - u folds the cell back onto the diagonal with its own sign.
      bface[f] = w[f];
    } else if (bc == BoundaryFlag::periodic) {
      // The wrapped ghost is a genuine neighbour, and so no diagonal at all, unless the
      // level is one cell wide along that axis -- then it is the cell itself.
      bface[f] = (extent[f] == 1) ? -w[f] : 0.0;
    } else {
      bface[f] = -w[f];
    }
  }

  // The same sweep count the base class would have run.
  const int ni = std::max(std::max(nrbx1_, std::max(nrbx2_, nrbx3_))
                          >> (nrootlevel_ - 1), coarsest_min_sweeps_);
  auto *root = static_cast<MGGravity*>(mgroot_);
  MGRootBoundary();
  mgroot_->StoreOldData();
  mgroot_->CalculateFASRHSPack();
  for (int i = 0; i < ni; ++i) {
    root->SmoothPackCoarsestEffectiveDiagonal(coffset_, bface);
    MGRootBoundary();
    root->SmoothPackCoarsestEffectiveDiagonal(1 - coffset_, bface);
    MGRootBoundary();
  }
  if (collect_phase_timing_) {
    Kokkos::fence();
    timing_coarsest_time_ += timer.seconds();
  }
}


// The octet loop lives here rather than in the driver so that a level costs one virtual
// call instead of one per octet, and so that the mesh spacing, the relaxation factor and
// the stencil weights -- identical for every octet of a level -- are formed once.  The
// per-cell expressions are untouched, so the result is bit for bit what the per-octet
// versions produced.

void MGGravityDriver::SmoothOctetLevel(MGOctet *octs, int noct, int rlev, int color) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx() / static_cast<Real>(1 << rlev);
  const Real dx2 = dx * dx;
  const Real omega_over_diag = omega_ / gravity_diag_;
  const Real wy = gravity_wy_, wz = gravity_wz_, diag = gravity_diag_;
  const int c = color ^ coffset_;
  for (int o = 0; o < noct; ++o) {
    MGOctet &oct = octs[o];
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
          Real lap = OctLaplacian(oct, 0, k, j, i, wy, wz, diag);
          oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*omega_over_diag;
        }
      }
    }
  }
}

void MGGravityDriver::CalculateDefectOctetLevel(MGOctet *octs, int noct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx() / static_cast<Real>(1 << rlev);
  const Real idx2 = 1.0 / (dx * dx);
  const Real wy = gravity_wy_, wz = gravity_wz_, diag = gravity_diag_;
  for (int o = 0; o < noct; ++o) {
    MGOctet &oct = octs[o];
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh; i <= ngh+1; ++i) {
          oct.Def(0,k,j,i) = oct.Src(0,k,j,i)
                           - OctLaplacian(oct, 0, k, j, i, wy, wz, diag) * idx2;
        }
      }
    }
  }
}

void MGGravityDriver::CalculateFASRHSOctetLevel(MGOctet *octs, int noct, int rlev) {
  const int ngh = mgroot_->GetGhostCells();
  const Real dx = mgroot_->GetRootDx() / static_cast<Real>(1 << rlev);
  const Real idx2 = 1.0 / (dx * dx);
  const Real wy = gravity_wy_, wz = gravity_wz_, diag = gravity_diag_;
  for (int o = 0; o < noct; ++o) {
    MGOctet &oct = octs[o];
    for (int k = ngh; k <= ngh+1; ++k) {
      for (int j = ngh; j <= ngh+1; ++j) {
        for (int i = ngh; i <= ngh+1; ++i) {
          oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i, wy, wz, diag) * idx2;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::SyncDistributedCoarseSolution(bool folddata)
//! \brief Replicate the coarse (root grid + octet) solution from the coarse-solve owner
//!        rank onto every other rank.  With gravity/distributed_coarse_solve enabled only
//!        the owner runs the bottom solve, so the result has to be broadcast before the
//!        up-leg reads the octets again.  Octets are host-resident, so a plain contiguous
//!        staging vector is all the broadcast needs.

void MGGravityDriver::SyncDistributedCoarseSolution(bool folddata) {
#if MPI_PARALLEL_ENABLED
  if (!UseDistributedCoarseSolve() || nranks_ <= 1) return;
  auto bcast_host = [&](Real *buf, int count) {
    if (count <= 0) return;
    MPI_Bcast(buf, count, MPI_ATHENA_REAL, coarse_owner_rank_, MPI_COMM_WORLD);
  };
  const bool is_owner = (global_variable::my_rank == coarse_owner_rank_);

  mgroot_->SetCurrentLevel(nrootlevel_ - 1);
  if (!mgroot_->OnHost()) {
    SyncRootDataToHost(folddata);
  }
  auto count5 = [](const auto &view) -> int {
    return view.extent_int(0) * view.extent_int(1) * view.extent_int(2) *
           view.extent_int(3) * view.extent_int(4);
  };
  auto root_u_h = mgroot_->GetCurrentData_h();
  bcast_host(root_u_h.data(), count5(root_u_h));
  if (folddata) {
    auto root_uold_h = mgroot_->GetCurrentOldData_h();
    bcast_host(root_uold_h.data(), count5(root_uold_h));
  }
  if (!mgroot_->OnHost()) {
    SyncRootDataToDevice();
    if (folddata) SyncRootOldDataToDevice();
  }

  // noctets_ is derived from the global block list, so every rank agrees on the count
  // and the broadcasts below stay collective.
  for (int lev = 0; lev < nreflevel_; ++lev) {
    const int no = noctets_[lev];
    if (no <= 0) continue;
    const int nc = octets_[lev][0].nc;
    const std::size_t count = static_cast<std::size_t>(no)*nvar_*nc*nc*nc;
    if (oct_bcast_buf_.size() < count) oct_bcast_buf_.resize(count);
    Real *buf = oct_bcast_buf_.data();
    auto pack = [&](bool old) {
      std::size_t p = 0;
      for (int o = 0; o < no; ++o) {
        MGOctet &oct = octets_[lev][o];
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int i = 0; i < nc; ++i) {
                buf[p++] = old ? oct.Uold(v,k,j,i) : oct.U(v,k,j,i);
              }
            }
          }
        }
      }
    };
    auto unpack = [&](bool old) {
      std::size_t p = 0;
      for (int o = 0; o < no; ++o) {
        MGOctet &oct = octets_[lev][o];
        for (int v = 0; v < nvar_; ++v) {
          for (int k = 0; k < nc; ++k) {
            for (int j = 0; j < nc; ++j) {
              for (int i = 0; i < nc; ++i) {
                if (old) {
                  oct.Uold(v,k,j,i) = buf[p++];
                } else {
                  oct.U(v,k,j,i) = buf[p++];
                }
              }
            }
          }
        }
      }
    };
    if (is_owner) pack(false);
    bcast_host(buf, static_cast<int>(count));
    unpack(false);
    if (folddata) {
      if (is_owner) pack(true);
      bcast_host(buf, static_cast<int>(count));
      unpack(true);
    }
  }

  root_flat_buf_stale_ = true;
  root_uold_buf_valid_ = false;
#else
  (void)folddata;
#endif
}
