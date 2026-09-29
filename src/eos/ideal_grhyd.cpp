//========================================================================================
// Athena++ (Kokkos version) astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grhyd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic hydro
//! Uses the same algorithm as implemented for SR hydro.

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

// Occupancy knob for the 256-thread grhyd_c2p launch: blocks/SM ptxas must fit, trading
// registers per thread (2 => <=128 registers) against spilling; 0 removes the constraint.
constexpr int kC2PMinBlocksPerSM = 2;

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRHydro::IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.hydro_eos = HydroEOSModel::gamma_law;
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("hydro","gamma_max",(FLT_MAX));  // gamma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in GR hydro.
//! Operates over range of cells given in argument list.

void IdealGRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->phydro->fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  // Dual energy.  dual_eta1 decides, per cell, whether the pressure is taken from the
  // conserved energy or from the advected adiabat; see SingleC2P_IdealSRHyd_Adiabat.
  const bool dual_enabled = pmy_pack->phydro->use_dual_energy;
  const int dual_idx = pmy_pack->phydro->dual_energy_idx;
  const Real dual_eta1 = pmy_pack->phydro->dual_energy_eta1;
  auto dual_tau_ = pmy_pack->phydro->dual_etot_max;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("grhyd_c2p",
  Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, kC2PMinBlocksPerSM>>
      (DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    const int a = (idx)/nkji;
    const int m = lat_enabled ? active_indices(a) : a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);
    const Real scalar_cons_density = u.d;

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

    // Four-number Cartesian Kerr-Schild null form (f, l_1, l_2, l_3) in place of the
    // 32 doubles of glower[4][4]/gupper[4][4]; see coordinates/cartesian_ks.hpp.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, flat, spin, nf);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;
    // tau/D of the conserved state, handed to the eta2 pass, which has no metric.
    Real q_budget_tau = 0.0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u,nf,s2,u_sr);
      const HydCons1D u_sr_in = u_sr;  // the energy solve floors u_sr in place
      q_budget_tau = (u_sr_in.d > 0.0) ? u_sr_in.e/u_sr_in.d : 0.0;

      // call c2p function
      // (inline function in ideal_c2p_hyd.hpp file)
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used);

      // Dual energy: the eta1 ratio test.  u_sr_in.e is tau, the conserved energy
      // without rest mass, and w.e the internal energy the inversion recovered from it;
      // their ratio is the number of digits the cancellation left.  Below eta1 the
      // recovered pressure is not trustworthy, and neither is the Lorentz factor that
      // came out of the same root, so the cell is re-solved from the advected adiabat.
      // The test also fires on a solve that failed outright, for which the auxiliary is
      // the backstop the failure floors used to be.
      if (dual_enabled) {
        const Real dens_cons = cons(m,IDN,k,j,i);
        const Real kappa_adv = (dens_cons > 0.0)
            ? cons(m,dual_idx,k,j,i)/dens_cons : -1.0;
        const bool use_cons_e = !c2p_failure && (w.e > 0.0) &&
            ((dual_eta1 <= 0.0) ||
             (w.e > dual_eta1*fmax(u_sr_in.e, static_cast<Real>(1.0e-18))));
        if (!use_cons_e && (kappa_adv > 0.0) && isfinite(kappa_adv)) {
          HydCons1D u_aux = u_sr_in;
          HydPrim1D w_aux;
          bool aux_dfloor = false, aux_efloor = false, aux_failure = false;
          int aux_iter = 0;
          SingleC2P_IdealSRHyd_Adiabat(u_aux, eos, s2, kappa_adv, w_aux, aux_dfloor,
                                       aux_efloor, aux_failure, aux_iter);
          iter_used = (aux_iter > iter_used) ? aux_iter : iter_used;
          // A failed auxiliary solve leaves the energy-channel answer standing.  A
          // successful one replaces the energy channel's floor verdict with its own:
          // a cell the adiabat recovered cleanly is not a floored cell.
          if (!aux_failure) {
            w = w_aux;
            dfloor_used = aux_dfloor;
            efloor_used = aux_efloor;
            c2p_failure = false;
          }
        }
      }

      // apply velocity ceiling if necessary
      Real tmp = KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz);
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // The adiabat handed to the reconstruction is the transported ratio, not the
      // pressure just published: resynchronizing the two is the eta2 pass's job.  A
      // floor that moved the thermodynamic state is the exception, since it changes
      // kappa = p/rho^Gamma; a Lorentz ceiling changes no thermodynamic quantity.
      Real kappa_pub = 0.0;
      bool dual_kappa_adjusted = false;
      if (dual_enabled) {
        const Real dens_cons_pub = cons(m,IDN,k,j,i);
        kappa_pub = (dens_cons_pub > 0.0)
            ? cons(m,dual_idx,k,j,i)/dens_cons_pub : 0.0;
        if (dfloor_used || efloor_used) {
          kappa_pub = (w.d > 0.0 && w.e > 0.0)
              ? gm1*w.e/pow(w.d, eos.gamma) : 0.0;
        }
        if (!(isfinite(kappa_pub) && kappa_pub > eos.sfloor)) {
          kappa_pub = eos.sfloor;
          dual_kappa_adjusted = true;
        }
        prim(m,dual_idx,k,j,i) = kappa_pub;
        dual_tau_(m,k,j,i) = q_budget_tau;
      }

      // reset conserved variables if floor, ceiling, failure, or excision encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure || excised) {
        SingleP2C_IdealGRHyd(nf, w, eos.gamma, u);
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;
        // The auxiliary rides the mass, so a repaired D carries it.
        if (dual_enabled) {
          cons(m,dual_idx,k,j,i) = u.d*kappa_pub;
        }
      } else if (dual_enabled && dual_kappa_adjusted) {
        cons(m,dual_idx,k,j,i) = cons(m,IDN,k,j,i)*kappa_pub;
      }

      // convert scalars (if any)
      const bool preserve_scalars_across_kinematic_repair = vceiling_used &&
          !dfloor_used && !efloor_used && !c2p_failure && !excised;
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        const Real scalar_den = preserve_scalars_across_kinematic_repair ?
            scalar_cons_density : u.d;
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/scalar_den;
        if (preserve_scalars_across_kinematic_repair) {
          cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
        }
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for GR hydrodynamics.  Operates
//! over range of cells given in argument list.

void IdealGRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  const bool dual_enabled_p2c = pmy_pack->phydro->use_dual_energy;
  const int dual_idx_p2c = pmy_pack->phydro->dual_energy_idx;
  const Real sfloor_p2c = eos_data.sfloor;

  par_for("grhyd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
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

    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, flat, spin, nf);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRHyd(nf, w, gamma, u);

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

    // The dual-energy auxiliary sits past the scalars and is densitized the same way:
    // the conserved variable is D*kappa.  A primitive state that carries no adiabat yet
    // (a problem generator's first conversion) gets the one its own pressure and
    // density define; otherwise the first inversion would read an entropy-floor adiabat
    // and, wherever it took the auxiliary channel, rebuild the cell cold.
    if (dual_enabled_p2c) {
      Real kappa = prim(m,dual_idx_p2c,k,j,i);
      if (!(isfinite(kappa) && kappa > 0.0)) {
        kappa = (w.d > 0.0 && w.e > 0.0) ? (gamma - 1.0)*w.e*pow(w.d, -gamma) : 0.0;
        if (!(isfinite(kappa) && kappa > sfloor_p2c)) { kappa = sfloor_p2c; }
      }
      cons(m,dual_idx_p2c,k,j,i) = u.d*kappa;
    }
  });

  return;
}
