#ifndef EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
#define EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd_c2p_impl.hpp
//! \brief IdealGRMHD::ConsToPrimImpl, the grmhd_c2p kernel, and the helpers only it
//! reads.  Included only by the ideal_grmhd_c2p_*.cpp units, each of which explicitly
//! instantiates half of the specializations IdealGRMHD::ConsToPrim (ideal_grmhd.cpp)
//! calls, so each gets its own ptxas pass and every specialization is compiled in
//! exactly one unit.

#include <float.h>

#include <algorithm>

#include "athena.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

// Occupancy knob for the 256-thread grmhd_c2p launch: blocks/SM ptxas must fit, trading
// registers per thread (2 => <=128, i.e. 25% occupancy on a V100) against spilling; 0
// removes the constraint and restores the unbounded ~236-register, 12.5%-occupancy build.
constexpr int kC2PMinBlocksPerSM = 2;

template <bool only_testfloors, bool track_event_counters, bool apply_sigma_ceiling>
void IdealGRMHD::ConsToPrimImpl(DvceArray5D<Real> &cons,
                                const DvceFaceFld4D<Real> &b,
                                DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto eos = eos_data;
  const int nwork = nmb;
  if (nwork <= 0) return;
  const int user_boundary_face = UserBoundaryFaceFilter();
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  // Dual energy.  dual_eta1 decides, per cell, whether the pressure is taken from the
  // conserved energy or from the advected adiabat; see SingleC2P_IdealSRMHD_Adiabat.
  const bool dual_enabled = pmy_pack->pmhd->use_dual_energy;
  const int dual_idx = pmy_pack->pmhd->dual_energy_idx;
  const Real dual_eta1 = pmy_pack->pmhd->dual_energy_eta1;
  auto dual_tau_ = pmy_pack->pmhd->dual_etot_max;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  auto event_counters = event_counters_.d_view;
  if constexpr (track_event_counters) {
    Kokkos::deep_copy(DevExeSpace(), event_counters, 0);
  }

  // Warm-start cache for the Kastaun root.  It is shaped like one component of the fine
  // u0, so it is only addressable when this launch works on arrays of that shape (never
  // for the coarse arrays).  Any pass may READ it -- a good guess stays a good guess --
  // but only the authoritative u0/w0 inversion may WRITE it, so that the FOFC test pass
  // cannot publish a root for a state that is not the one being evolved.
  // A remesh (AMR or load balance) renumbers the local blocks, so an entry left over
  // from before it is a DIFFERENT cell's root -- the extent test below cannot see that.
  // Empty the cache instead; the cells then take the cold bracket for one call, which is
  // exactly what they did before the cache existed.  Costs one fill on remesh steps only.
  // A pure rebalance carries the cache with its blocks and re-stamps its version
  // (C2PHistoryCarried), so it is not emptied here.
  if (c2p_mu_cache_topology_version != pmy_pack->pmesh->topology_version) {
    Kokkos::deep_copy(DevExeSpace(), c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  }
  auto mu_cache = c2p_mu_cache;
  const bool mu_cache_addressable =
      (mu_cache.extent_int(0) >= nmb) &&
      (mu_cache.extent_int(2) == cons.extent_int(2)) &&
      (mu_cache.extent_int(3) == cons.extent_int(3)) &&
      (mu_cache.extent_int(4) == cons.extent_int(4));
  const bool read_mu_cache = mu_cache_addressable;
  const bool write_mu_cache = mu_cache_addressable &&
      !only_testfloors &&
      (cons.data() == pmy_pack->pmhd->u0.data()) &&
      (prim.data() == pmy_pack->pmhd->w0.data());

  Kokkos::parallel_for("grmhd_c2p",
  Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, kC2PMinBlocksPerSM>>
      (DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx) {
    // NVCC extended lambdas require captures to be established outside if constexpr.
    (void) mu_cache;
    (void) read_mu_cache;
    (void) write_mu_cache;
    (void) fofc_;

    int a = (idx)/nkji;
    int m = a;
    if (user_boundary_face >= 0 &&
        mb_bcs.d_view(m,user_boundary_face) != BoundaryFlag::user) return;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);
    Real scalar_cons_density = u.d;

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

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

    // Cartesian Kerr-Schild is a rank-1 update of Minkowski, g = eta + f l l with
    // l_0 = 1, so the four numbers (f, l_1, l_2, l_3) carry the whole geometry.  Holding
    // those instead of glower[4][4]/gupper[4][4] keeps 4 live doubles per thread instead
    // of 32 and removes the ~40 multiplies that materialize the two matrices.  Every
    // accessor below is an exact rank-1 identity; none of them assumes l.l = 1, which the
    // r < 1e-6 floor inside ComputeKSNullForm genuinely breaks (at the origin f = 4e6 and
    // l.l - 1 = -1), so the reduction stays valid down to the coordinate singularity.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, flat, spin, nf);

    bool coordinate_excised = use_excise && excision_floor_(m,k,j,i);
    EOS_Data cell_eos = eos;
    Real gm1 = cell_eos.gamma - 1.0;

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    // Set when the eta1 ratio test moved this cell onto the auxiliary channel.
    // tau/D for this cell.  It must come from the SR recast, NOT from
    // cons(IEN)/cons(IDN): this backend evolves T^t_t + D in that slot
    // (SingleP2C_IdealGRMHD), which is negative for ordinary states, so a ratio built
    // from the array is not the energy budget and is usually the wrong sign.  Published
    // below for the eta2 pass, which cannot form it without the metric.
    Real q_budget_tau = 0.0;
    // rho*h before the floors, kept so the velocity of whatever they inject is decided
    // once, after the magnetization ceiling too.
    Real w_prefloor = 0.0;
    bool vceiling_used=false, invalid_velocity_state=false;
    bool sigceiling_used=false, c2p_failure=false;
    // Remembered past the auxiliary block, which may clear c2p_failure: the warm-start
    // cache must not learn the root of a solve that failed.
    bool energy_solve_failed = false;
    int iter_used=0;
    // Cached Kastaun root of the previous inversion of this cell (0 means "no guess"),
    // and the root this inversion converges to.
    const Real mu_guess = read_mu_cache ? mu_cache(m,0,k,j,i) : 0.0;
    Real mu_root = 0.0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (coordinate_excised) {
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
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,nf,s2,b2,rpar,u_sr);

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      const MHDCons1D u_sr_in = u_sr;
      if (u_sr_in.d > 0.0) { q_budget_tau = u_sr_in.e/u_sr_in.d; }
      SingleC2P_IdealSRMHD(u_sr, cell_eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used,
                           mu_guess, mu_root, &w_prefloor);
      energy_solve_failed = c2p_failure;

      // Dual energy: the eta1 ratio test.  u_sr_in.e is tau, the conserved energy
      // without rest mass, and w.e is the internal energy the inversion recovered from
      // it; their ratio is the number of digits the cancellation left.  Below eta1 the
      // recovered pressure is not trustworthy -- and neither is the Lorentz factor,
      // which comes out of the same root -- so the cell is re-solved from the advected
      // adiabat instead.  The test is predictive: it fires on cells whose answer is
      // quietly wrong, and it also fires on every cell where the solve failed outright,
      // for which the auxiliary is the backstop that the failure floors used to be.
      if (dual_enabled) {
        const Real dens_cons = cons(m,IDN,k,j,i);
        const Real kappa_adv = (dens_cons > 0.0)
            ? cons(m,dual_idx,k,j,i)/dens_cons : -1.0;
        const bool use_cons_e = !c2p_failure && (w.e > 0.0) &&
            ((dual_eta1 <= 0.0) ||
             (w.e > dual_eta1*fmax(u_sr_in.e, static_cast<Real>(1.0e-18))));
        if (!use_cons_e && (kappa_adv > 0.0) && isfinite(kappa_adv)) {
          MHDCons1D u_aux = u_sr_in;
          HydPrim1D w_aux;
          bool aux_dfloor = false, aux_efloor = false, aux_failure = false;
          Real w_prefloor_aux = w_prefloor;
          SingleC2P_IdealSRMHD_Adiabat(u_aux, cell_eos, s2, b2, rpar, kappa_adv, w_aux,
                                       aux_dfloor, aux_efloor, aux_failure, iter_used,
                                       &w_prefloor_aux);
          // A failed auxiliary solve leaves the energy-channel answer standing: the
          // whole point of the ratio test is that it is allowed to be wrong about which
          // channel is better, never that it may leave the cell without one.
          if (!aux_failure) {
            w = w_aux;
            u_sr = u_aux;
            w_prefloor = w_prefloor_aux;
            // Adopt the accepted channel's verdict instead of the union of the two.
            // The flags standing here describe the energy-channel solve that has just
            // been discarded, and a cell that collapsed onto its energy floor but which
            // the adiabat then recovered cleanly is not a floored cell: reporting it as
            // one drops it to first-order fluxes, which diffuses exactly the region this
            // channel exists to resolve, and over-counts floor events.  Nothing shared
            // is lost by the assignment -- the auxiliary solve starts from the same
            // pre-floor u_sr_in and re-applies the identical conserved-density floor, so
            // a floor that belongs to both channels is already inside aux_dfloor -- and
            // w_prefloor, which the reinjection below reads together with these flags,
            // has just been swapped to the auxiliary solve's as well.
            dfloor_used = aux_dfloor;
            efloor_used = aux_efloor;
            // A cell the adiabat recovered is not a failed cell: no failure floor, no
            // FOFC flag, no event count.
            c2p_failure = false;
          }
        }
      }

      // Apply the common KORAL GAMMAMAXHD ceiling with a scaled metric norm.  The
      // traditional direct quadratic form can overflow or become NaN before its
      // comparison, which lets an exact-zero hybrid cell publish an invalid velocity.
      Real glower_lorentz[4][4];
      KSFillLower(nf, glower_lorentz);
      const auto lorentz_limit = mhd::forcefree::ApplyIsotropicLorentzLimit(
          glower_lorentz, cell_eos.gamma_max, w.vx, w.vy, w.vz);
      if (!lorentz_limit.valid) {
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        vceiling_used = true;
        invalid_velocity_state = true;
      } else if (lorentz_limit.limited) {
        vceiling_used = true;
      }
      if (apply_sigma_ceiling) {
        sigceiling_used = ApplySigmaCeiling_IdealGRMHD(cell_eos, nf, u, w);
      }

      // Everything above has raised rho or u somewhere; this is the single place the
      // velocity that comes with the added material is decided.  A c2p failure or an
      // invalid velocity is a different animal -- the state was reset, not floored -- and
      // has no pre-floor state worth re-injecting into.
      if ((dfloor_used || efloor_used || sigceiling_used) &&
          !c2p_failure && !invalid_velocity_state) {
        ReinjectFlooredMass_IdealGRMHD(cell_eos.gamma, nf, u, w_prefloor, w);
      }
    }

    // Store the converged Kastaun root so the next inversion of this cell can warm
    // start from it.  Only a successful, non-excised, authoritative inversion writes.
    if (write_mu_cache && !excised && !energy_solve_failed && (mu_root > 0.0)) {
      mu_cache(m,0,k,j,i) = mu_root;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || sigceiling_used ||
          c2p_failure) {
        fofc_(m,k,j,i) = true;
        if (track_event_counters) {
          Kokkos::atomic_increment(&event_counters(0));
        }
      }
    } else {
      if (track_event_counters) {
        if (dfloor_used) Kokkos::atomic_increment(&event_counters(0));
        if (efloor_used) Kokkos::atomic_increment(&event_counters(1));
        if (vceiling_used || sigceiling_used) {
          Kokkos::atomic_increment(&event_counters(2));
        }
        if (c2p_failure) Kokkos::atomic_increment(&event_counters(3));
        Kokkos::atomic_max(&event_counters(4), iter_used);
      }

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // The auxiliary primitive is the ADVECTED adiabat, never the one implied by the
      // pressure just published: resynchronizing the two is the eta2 pass's job, and
      // doing it here would erase the shock heating the energy channel is carrying.
      // The floor is the same entropy floor the inversion uses, so the value handed to
      // the reconstruction is one the next inversion can accept.
      Real kappa_pub = 0.0;
      // Raised wherever the published adiabat stops being the ratio the conserved array
      // holds, so that the partner further down can be brought along with it.
      bool dual_kappa_adjusted = false;
      if (dual_enabled) {
        const Real dens_cons_pub = cons(m,IDN,k,j,i);
        kappa_pub = (dens_cons_pub > 0.0)
            ? cons(m,dual_idx,k,j,i)/dens_cons_pub : 0.0;
        // A repair that moved the thermodynamic state is the exception to the rule
        // above.  The magnetization ceiling loads mass at fixed SPECIFIC internal
        // energy, and the density and energy floors add mass or energy outright, so
        // every one of them changes kappa = p/rho^Gamma.  The advected ratio then
        // describes a state the cell is no longer in, and since the eta2
        // resynchronization is deliberately infrequent that disagreement survives and
        // compounds.  Republish from the repaired primitives, which are the ones just
        // written to prim above and the ones the conserved repair below regenerates U
        // from.  A Lorentz ceiling is excluded on purpose: it changes no thermodynamic
        // quantity, so the advected adiabat crosses it intact.
        if (dfloor_used || efloor_used || sigceiling_used) {
          kappa_pub = (w.d > 0.0 && w.e > 0.0)
              ? (cell_eos.gamma - 1.0)*w.e/pow(w.d, cell_eos.gamma) : 0.0;
        }
        if (!(isfinite(kappa_pub) && kappa_pub > cell_eos.sfloor)) {
          kappa_pub = cell_eos.sfloor;
          dual_kappa_adjusted = true;
        }
        // The bound by the cell's own energy budget is applied in the eta2 pass
        // (MHD::SynchronizeDualEnergyFieldFromAdiabat), which already writes both
        // columns of every owned cell each stage and is memory bound, so the power it
        // costs is free there and not here, where this kernel is instruction bound.
        // The root solve above bounds the specific energy it actually uses by tau/D
        // itself, so nothing this inversion publishes to the gas depends on the cap.
        prim(m,dual_idx,k,j,i) = kappa_pub;
        // The eta2 pass compares eps against max_27(tau/D) and has no metric with which
        // to build tau from the conserved array, so the inversion hands it over here.
        dual_tau_(m,k,j,i) = q_budget_tau;
      }

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // A Lorentz ceiling changes only the normal-frame velocity.  It requires a new
      // conservative core (and hence a new D), but it must not consume the transported
      // total-vs-species residual by repartitioning se/si onto the unchanged internal
      // energy.  Preserve every passive primitive ratio across a purely kinematic repair.
      const bool preserve_scalars_across_kinematic_repair = vceiling_used &&
          !dfloor_used && !efloor_used && !sigceiling_used &&
          !c2p_failure && !excised;
      // dual_aux_used does NOT belong here.  A cell whose pressure came from the
      // adiabat keeps its conserved energy: the energy equation is ignored for the
      // pressure, not overwritten, the fluxes are built from the auxiliary primitives,
      // and the next inversion finds the conserved energy wanting again and takes the
      // auxiliary again -- the non-relativistic flavour's rule (ideal_hyd.cpp) and
      // Enzo's.  Regenerating U from the auxiliary state injected u_aux - u_cons into
      // the total energy at every inversion that took the channel, a one-sided ratchet
      // on the MAD funnel (the channel fires exactly where the energy channel reads
      // cold) that heated the funnel and multiplied the spikes it exists to remove.
      const bool reset_conserved = dfloor_used || efloor_used || vceiling_used ||
          sigceiling_used || c2p_failure || excised;

      // reset conserved variables if floor, ceiling, failure, or excision encountered.
      if (reset_conserved) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(nf, w_in, cell_eos.gamma, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // (needed if there are scalars below)
        // The auxiliary rides the mass, so a repaired D has to carry it: without this
        // the ratio cons(dual)/cons(IDN) -- which IS the adiabat -- would drift by
        // whatever the floors did to the density.  u_out.d is the density the rest of
        // this repair published, so the pair stays exactly the kappa handed to the
        // reconstruction, republished value included.
        if (dual_enabled) {
          cons(m,dual_idx,k,j,i) = u_out.d*kappa_pub;
        }
      } else if (dual_enabled && dual_kappa_adjusted) {
        // The clamps above are rejections of the advected ratio, and a primitive its
        // own conserved partner contradicts survives exactly one reconstruction: the
        // next inversion forms cons(dual)/cons(IDN) and recovers the value just
        // rejected, so the bound would buy nothing beyond a single stage.  No repair
        // rewrote D here, so the partner rides the density the cell already holds --
        // the same rule the dynamical inversion applies.
        cons(m,dual_idx,k,j,i) = cons(m,IDN,k,j,i)*kappa_pub;
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        Real scalar_cons = cons(m,n,k,j,i);
        if (!isfinite(scalar_cons) || scalar_cons < 0.0) {
          scalar_cons = 0.0;
          cons(m,n,k,j,i) = 0.0;
        }
        const Real scalar_den =
            preserve_scalars_across_kinematic_repair ? scalar_cons_density : u.d;
        Real scalar = (scalar_den > 0.0) ? (scalar_cons/scalar_den) : 0.0;
        prim(m,n,k,j,i) = isfinite(scalar) ? scalar : 0.0;
        if (preserve_scalars_across_kinematic_repair) {
          cons(m,n,k,j,i) = (u.d > 0.0) ? u.d*prim(m,n,k,j,i) : 0.0;
        }
      }
    }
  });

  // These counters are consumed only by event-log output. Avoid a device-to-host
  // synchronization after every C2P launch when no event log is configured.
  if constexpr (track_event_counters) {
    event_counters_.template modify<DevExeSpace>();
    event_counters_.template sync<HostMemSpace>();
    if constexpr (only_testfloors) {
      pmy_pack->pmesh->ecounter.nfofc += event_counters_.h_view(0);
    } else {
      pmy_pack->pmesh->ecounter.neos_dfloor += event_counters_.h_view(0);
      pmy_pack->pmesh->ecounter.neos_efloor += event_counters_.h_view(1);
      pmy_pack->pmesh->ecounter.neos_vceil  += event_counters_.h_view(2);
      pmy_pack->pmesh->ecounter.neos_fail   += event_counters_.h_view(3);
      pmy_pack->pmesh->ecounter.maxit_c2p = event_counters_.h_view(4);
    }
  }

  return;
}

// One specialization of ConsToPrimImpl, in the order of its template parameters:
// only_testfloors, track_event_counters, apply_sigma_ceiling.
#define INSTANTIATE_IDEAL_GRMHD_C2P(A, B, C) \
template void IdealGRMHD::ConsToPrimImpl<A, B, C>( \
    DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, \
    DvceArray5D<Real> &bcc, const int il, const int iu, const int jl, const int ju, \
    const int kl, const int ku);

#endif  // EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
