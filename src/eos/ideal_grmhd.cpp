//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd.  The
//! grmhd_c2p kernel, IdealGRMHD::ConsToPrimImpl, is in ideal_grmhd_c2p_impl.hpp and is
//! compiled by the ideal_grmhd_c2p_*.cpp units.

#include <float.h>

#include <algorithm>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

namespace {

bool EventLogRequested(ParameterInput *pin) {
  for (const auto &block : pin->block) {
    if (block.block_name.compare(0, 6, "output") != 0) continue;
    for (const auto &line : block.line) {
      if (line.param_name == "file_type" && line.param_value == "log") return true;
    }
  }
  return false;
}

}  // namespace

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin),
    track_event_counters_(EventLogRequested(pin)),
    event_counters_("grmhd_event_counters", 5) {
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
  if (pin->DoesParameterExist("mhd", "sigma_ceiling")) {
    eos_data.sigma_max = pin->GetReal("mhd", "sigma_ceiling");
  } else if (pin->DoesParameterExist("mhd", "sigma_max")) {
    eos_data.sigma_max = pin->GetReal("mhd", "sigma_max");
  }
  if (pin->DoesParameterExist("mhd", "sigma_ceiling") ||
      pin->DoesParameterExist("mhd", "sigma_max")) {
    if (!isfinite(eos_data.sigma_max) || (eos_data.sigma_max <= 0.0)) {
      eos_data.sigma_max = 0.0;
    }
  }

  // Allocate the warm-start cache for the Kastaun root of the c2p inversion.  Sized like
  // one component of u0, with nmb_maxperrank so AMR/load balancing never overruns it.
  // Starts empty (0 = "no guess"), and is emptied again on every topology change that
  // does not carry it (a pure rebalance does), so the guess a cell warm starts from is
  // always that cell's own previous root.  See the
  // declaration in eos.hpp for what the cache does and does not preserve: it changes the
  // converged root at the ~1e-12 level, so it is run history.
  {
    auto &indcs = pp->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
    c2p_mu_cache = DvceArray5D<Real>("c2p_mu_cache", nmb, 1, ncells3, ncells2, ncells1);
    Kokkos::deep_copy(c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pp->pmesh->topology_version;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool IdealGRMHD::ResetC2PWarmStart()
//! \brief Empty the Kastaun warm-start cache.  Every cell then takes the cold bracket on
//! its next inversion, which is the only starting point a restart can reproduce from an
//! ordinary checkpoint: it carries u0, not the cache, so the restart recovers cold.

bool IdealGRMHD::ResetC2PWarmStart() {
  if (c2p_mu_cache.size() == 0) return false;
  Kokkos::deep_copy(c2p_mu_cache, 0.0);
  c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  const bool apply_sigma_ceiling =
      isfinite(eos_data.sigma_max) && eos_data.sigma_max > 0.0;
  if (only_testfloors) {
    if (track_event_counters_) {
      if (apply_sigma_ceiling) {
        ConsToPrimImpl<true, true, true>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      } else {
        ConsToPrimImpl<true, true, false>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      }
    } else {
      if (apply_sigma_ceiling) {
        ConsToPrimImpl<true, false, true>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      } else {
        ConsToPrimImpl<true, false, false>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      }
    }
  } else {
    if (track_event_counters_) {
      if (apply_sigma_ceiling) {
        ConsToPrimImpl<false, true, true>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      } else {
        ConsToPrimImpl<false, true, false>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      }
    } else {
      if (apply_sigma_ceiling) {
        ConsToPrimImpl<false, false, true>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      } else {
        ConsToPrimImpl<false, false, false>(
            cons, b, prim, bcc, il, iu, jl, ju, kl, ku);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma = eos_data.gamma;
  const int nwork = nmb;
  if (nwork <= 0) return;
  const int user_boundary_face = UserBoundaryFaceFilter();
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;

  const bool dual_enabled_p2c = pmy_pack->pmhd->use_dual_energy;
  const int dual_idx_p2c = pmy_pack->pmhd->dual_energy_idx;
  const Real sfloor_p2c = pmy_pack->pmhd->peos->eos_data.sfloor;

  par_for("grmhd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = a;
    if (user_boundary_face >= 0 &&
        mb_bcs.d_view(m,user_boundary_face) != BoundaryFlag::user) return;
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Same four-number Kerr-Schild null form as ConsToPrim above (see the comment
    // there); the two 4x4 metric matrices are never materialized in this kernel either.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, flat, spin, nf);

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(nf, w, gamma, u);

    // store conserved quantities in 3D array
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

    // The dual-energy auxiliary sits past the scalars and is densitized the same way:
    // the conserved variable is D*kappa, so the ratio the inversion reads back is the
    // adiabat itself, independent of sqrt(det g).
    if (dual_enabled_p2c) {
      const Real kappa = prim(m,dual_idx_p2c,k,j,i);
      cons(m,dual_idx_p2c,k,j,i) =
          u.d*((isfinite(kappa) && kappa > 0.0) ? kappa : sfloor_p2c);
    }
  });

  return;
}
