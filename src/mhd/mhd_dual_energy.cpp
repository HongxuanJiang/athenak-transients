//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_dual_energy.cpp
//! \brief Dual-energy formalism for Newtonian MHD with EOS-aware thermal support

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool DualEnergySyncEligible(const Real eint_cons, const Real local_etot_max,
                            const Real eta2) {
  if (eint_cons <= 0.0) return false;
  if (eta2 <= 0.0) return true;
  return (eint_cons > eta2*fmax(local_etot_max, 1.0e-18));
}

KOKKOS_INLINE_FUNCTION
Real BccX(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
}

KOKKOS_INLINE_FUNCTION
Real BccY(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
}

KOKKOS_INLINE_FUNCTION
Real BccZ(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
}

}  // namespace

namespace mhd {

TaskStatus MHD::DualEnergyStep(Driver *pdrive, int stage) {
  if (!use_dual_energy) return TaskStatus::complete;
  if (dual_energy_pdv) {
    // The non-relativistic auxiliary carries a compression source term.
    const Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
    ApplyDualEnergyFormalism(beta_dt);
    return TaskStatus::complete;
  }
  // The GR adiabat is advected by an exact conservation law, so it has no source term.
  // What it needs once per stage is the eta2 resynchronization from the energy channel,
  // and that pass reads nothing a neighbour has to deliver first: the density this
  // stage just updated, the adiabat it advected, and the primitives and tau/D ratios
  // the previous inversion published.  So it runs here, ahead of the conserved send.
  // That placement is what lets the auxiliary ride the ordinary machinery: neighbours
  // and coarse copies receive the resynchronized adiabat like any other conserved
  // variable, and the recoveries that follow -- the interior-first split, the ghost
  // bands, the deferred LAT publication -- need no special case for it.
  SynchronizeDualEnergyFieldFromAdiabat();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real AdiabatFromPrimitive()
//! \brief kappa = p/rho^Gamma from the published primitive state, floored at the entropy
//! floor the GR inversions already use.  `dyn` selects where the pressure lives: the
//! dynamical-GR solver publishes p in the IPR slot, the fixed-metric one publishes the
//! internal energy density in the same slot.

KOKKOS_INLINE_FUNCTION
Real AdiabatFromPrimitive(const Real rho, const Real slot, const bool dyn,
                          const Real gamma, const Real sfloor) {
  const Real gm1 = gamma - 1.0;
  const Real p = dyn ? slot : gm1*slot;
  Real kappa = ((rho > 0.0) && (p > 0.0)) ? p*pow(rho, -gamma) : 0.0;
  if (!(isfinite(kappa) && kappa > sfloor)) { kappa = sfloor; }
  return kappa;
}

void MHD::InitializeDualEnergyFieldFromAdiabat() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;
  const bool dyn = pmy_pack->pcoord->is_dynamical_relativistic;

  // Seed from the primitive state the problem generator produced.  The conserved form is
  // D*kappa, densitized exactly like D, so the ratio the inversion reads back is kappa
  // with no metric factor of its own.
  // The ghost range, not the active zone: this runs after the last conserved exchange of
  // Driver::Initialize, so an active-only seed leaves every ghost cell unseeded for the
  // first stage's flux kernel to read.  ConToPrim has already run over the same range.
  const bool multi_d_i = pmy_pack->pmesh->multi_d;
  const bool three_d_i = pmy_pack->pmesh->three_d;
  const int ng_i = indcs.ng;
  const int gis_i = is - ng_i, gie_i = ie + ng_i;
  const int gjs_i = multi_d_i ? (js - ng_i) : js, gje_i = multi_d_i ? (je + ng_i) : je;
  const int gks_i = three_d_i ? (ks - ng_i) : ks, gke_i = three_d_i ? (ke + ng_i) : ke;
  par_for("mhd_dual_adiabat_init", DevExeSpace(), 0, nmb1,
          gks_i, gke_i, gjs_i, gje_i, gis_i, gie_i,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real kappa = AdiabatFromPrimitive(w0_(m, IDN, k, j, i),
                                            w0_(m, IEN, k, j, i), dyn,
                                            eos.gamma, eos.sfloor);
    w0_(m, de_idx, k, j, i) = kappa;
    u0_(m, de_idx, k, j, i) = u0_(m, IDN, k, j, i)*kappa;
  });
}

void MHD::SynchronizeDualEnergyFieldFromAdiabat() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto qmax_ = dual_etot_max;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int ng = indcs.ng;
  const int gis = is - ng, gie = ie + ng;
  const int gjs = multi_d ? (js - ng) : js;
  const int gje = multi_d ? (je + ng) : je;
  const int gks = three_d ? (ks - ng) : ks;
  const int gke = three_d ? (ke + ng) : ke;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const Real eta2 = dual_energy_eta2;
  const Real gm1 = eos.gamma - 1.0;
  const bool dyn = pmy_pack->pcoord->is_dynamical_relativistic;

  // The eta2 test in GR is written per unit REST MASS, which is what makes it free of
  // sqrt(det g): q = tau/D is a ratio of two conserved variables carrying the same
  // densitization, and eps = u/rho is a ratio of two primitives.  In the non-relativistic
  // limit q -> (KE + eint + emag)/rho and eps -> eint/rho, so the test reduces to the
  // Newtonian `eint > eta2*max(E_tot)` exactly.  Building it any other way would need
  // the metric determinant in a file that has no business knowing the metric.
  // Where the conserved energy is still trustworthy, the advected adiabat is reset from
  // it.  Without this the auxiliary would drift and, worse, would never learn about
  // shock heating -- entropy is not conserved across a shock, and the energy channel is
  // the only one that knows by how much.
  // Owned cells only.  The pass runs ahead of the conserved send (MHD::DualEnergyStep),
  // so the exchange, the prolongation and the physical boundary conditions that follow
  // install the resynchronized adiabat in every ghost cell before anything reads it, the
  // way they do for every other conserved variable.  The stencil still reaches one
  // layer into the ghosts, where the previous ghost recovery published its tau/D.
  par_for("mhd_dual_adiabat_sync", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens_cons = u0_(m, IDN, k, j, i);
    const Real rho = w0_(m, IDN, k, j, i);
    Real kappa = (dens_cons > 0.0) ? u0_(m, de_idx, k, j, i)/dens_cons : 0.0;
    if (!(isfinite(kappa) && kappa > 0.0)) { kappa = eos.sfloor; }

    const Real slot = w0_(m, IEN, k, j, i);
    const Real p_energy = dyn ? slot : gm1*slot;
    const Real eps_energy = (rho > 0.0) ? p_energy/(gm1*rho) : 0.0;
    // max_27(tau/D).  tau is NOT u0(IEN) on the fixed-metric backend -- that slot holds
    // T^t_t + D, which is negative for ordinary states -- so the inversions publish the
    // ratio into this array and the stencil is taken over what they published.  Reading
    // it back from u0 left qmax at exactly zero and the gate unconditional, which threw
    // away the advected adiabat in every cell of every stage.
    Real qmax = 0.0;
    if (eta2 > 0.0) {
      const int kmin = three_d ? ((k - 1 > gks) ? (k - 1) : gks) : k;
      const int kmax = three_d ? ((k + 1 < gke) ? (k + 1) : gke) : k;
      const int jmin = multi_d ? ((j - 1 > gjs) ? (j - 1) : gjs) : j;
      const int jmax = multi_d ? ((j + 1 < gje) ? (j + 1) : gje) : j;
      const int imin = (i - 1 > gis) ? (i - 1) : gis;
      const int imax = (i + 1 < gie) ? (i + 1) : gie;
      for (int kk = kmin; kk <= kmax; ++kk) {
        for (int jj = jmin; jj <= jmax; ++jj) {
          for (int ii = imin; ii <= imax; ++ii) {
            const Real q = qmax_(m, kk, jj, ii);
            if (isfinite(q)) { qmax = fmax(qmax, q); }
          }
        }
      }
    }
    const bool eligible = (eps_energy > 0.0) &&
        ((eta2 <= 0.0) || (eps_energy > eta2*fmax(qmax, static_cast<Real>(1.0e-18))));
    if (eligible) {
      kappa = AdiabatFromPrimitive(rho, slot, dyn, eos.gamma, eos.sfloor);
    }
    if (kappa < eos.sfloor) { kappa = eos.sfloor; }
    // Bound the adiabat by the energy the cell owns.  Nothing in the transport bounds
    // it from above, and a cell whose density reaches its floor turns (D kappa)/D into
    // a finite numerator over a floor; an adiabat implying more internal energy than the
    // total is not a state any fluid can be in, and left unbounded it feeds the
    // pressure, the flux and the next adiabat in turn.  tau/D is the ratio the last
    // inversion published for this cell (no metric enters a ratio of two conserved
    // variables carrying the same densitization), and rho the density it published;
    // both are one stage old, which a bound can afford.  The inversions themselves cap
    // the specific energy they use by tau/D, so this cap protects the transported
    // column, not the pressure.
    const Real q_own = qmax_(m, k, j, i);
    if (isfinite(q_own) && (q_own > 0.0) && (rho > 0.0)) {
      const Real kappa_max = gm1*q_own*pow(rho, 1.0 - eos.gamma);
      if (isfinite(kappa_max) && (kappa_max > eos.sfloor) && (kappa > kappa_max)) {
        kappa = kappa_max;
      }
    }
    w0_(m, de_idx, k, j, i) = kappa;
    u0_(m, de_idx, k, j, i) = dens_cons*kappa;
  });
}

void MHD::InitializeDualEnergyFieldFromTotal() {
  if (!use_dual_energy) return;
  // The GR flavour is seeded by InitializeDualEnergyFieldFromAdiabat, after the
  // inversion; kappa is a function of the primitives and they do not exist yet here.
  // A caller that reaches this point on the GR path has not seeded anything, and must
  // not clear dual_energy_needs_init.
  if (!dual_energy_pdv) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;

  // Seed the auxiliary internal-energy field from the full conserved state, including
  // magnetic energy, so restarts and fresh starts begin synchronized.
  par_for("mhd_dual_energy_init", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real bx = BccX(b0_, m, k, j, i);
    const Real by = BccY(b0_, m, k, j, i);
    const Real bz = BccZ(b0_, m, k, j, i);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
    bool efloor_used = false, tfloor_used = false;
    Real eint = u0_(m, IEN, k, j, i) - e_k - e_m;
    eint = eos_general::ApplyMHDThermalFloors(eos, dens, eint, efloor_used,
                                              tfloor_used);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void MHD::ApplyDualEnergyFormalism(const Real dt) {
  if (!dual_energy_pdv) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  auto vf1_ = FluxBand(dual_vf.x1f);
  auto vf2_ = FluxBand(dual_vf.x2f);
  auto vf3_ = FluxBand(dual_vf.x3f);
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  // Evolve the auxiliary internal energy with interface-velocity compression. The
  // conservative-to-auxiliary synchronization is deferred until after boundary exchange
  // and prolongation, when the neighboring total-energy state is current across
  // MeshBlock boundaries.
  par_for("mhd_dual_energy_formalism", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real divv = (vf1_(m,0,k,j,i+1) - vf1_(m,0,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divv += (vf2_(m,0,k,j+1,i) - vf2_(m,0,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divv += (vf3_(m,0,k+1,j,i) - vf3_(m,0,k,j,i))/mbsize.d_view(m).dx3;
    }

    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    Real eint = u0_(m, de_idx, k, j, i);
    // The advected auxiliary internal-energy field can undershoot before this operator
    // split compression update. Reapply the thermal floor before any EOS query so the
    // tabulated Saha inversion never sees an out-of-bounds state.
    bool efloor_used = false, tfloor_used = false;
    eint = eos_general::ApplyMHDThermalFloors(eos, dens, eint, efloor_used,
                                              tfloor_used);
    if (eos.is_gamma_law) {
      eint *= exp(-(eos.gamma - 1.0)*divv*dt);
    } else {
      const auto thermo = eos.EvalThermoStateFromRhoEint(dens, eint);
      eint -= thermo.pressure*divv*dt;
    }
    efloor_used = false;
    tfloor_used = false;
    eint = eos_general::ApplyMHDThermalFloors(eos, dens, eint, efloor_used,
                                              tfloor_used);

    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void MHD::SynchronizeDualEnergyFieldFromTotal() {
  // The GR flavour is resynchronized in DualEnergyStep, ahead of the send; this pass is
  // the non-relativistic one, which reads the neighbour energies an exchange delivered.
  if (!use_dual_energy || !dual_energy_pdv) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto etot_max_ = dual_etot_max;
  auto excise_mask_ = dual_excise_mask;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto b0_ = b0;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_indcs = pmy_pack->pmesh->mb_indcs;
  const int mesh_is = mesh_indcs.is;
  const int mesh_js = mesh_indcs.js;
  const int mesh_ks = mesh_indcs.ks;
  const int mesh_nx1 = mesh_indcs.nx1;
  const int mesh_nx2 = mesh_indcs.nx2;
  const int mesh_nx3 = mesh_indcs.nx3;
  const int ng = mesh_indcs.ng;
  const int gis = is - ng;
  const int gie = ie + ng;
  const int gjs = multi_d ? (js - ng) : js;
  const int gje = multi_d ? (je + ng) : je;
  const int gks = three_d ? (ks - ng) : ks;
  const int gke = three_d ? (ke + ng) : ke;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const Real eta1 = dual_energy_eta1;
  const Real eta2 = dual_energy_eta2;
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(pmy_pack->pmesh->time, excise_enabled,
      excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
  const Real excise_r2 = excise_radius * excise_radius;

  if (excise_enabled) {
    // Cache the moving excision mask once per sync step so the eta2 stencil does not
    // repeatedly reconstruct cell centers and sink-center distances for every neighbor.
    par_for("mhd_dual_energy_excise_mask", DevExeSpace(), 0, nmb1,
            gks, gke, gjs, gje, gis, gie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - mesh_is, mesh_nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - mesh_js, mesh_nx2, mbsize.d_view(m).x2min,
                                 mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - mesh_ks, mesh_nx3, mbsize.d_view(m).x3min,
                                 mbsize.d_view(m).x3max);
      excise_mask_(m, k, j, i) =
          problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                              excise_r2) ? 1 : 0;
    });
  }

  if (eta2 > 0.0) {
    // Cache the local max(total energy) used by the eta2 test across the same ghosted
    // range that will immediately be converted to primitives. This keeps received ghost
    // cells synchronized with their neighboring interior owners before reconstruction.
    par_for("mhd_dual_energy_etot_max", DevExeSpace(), 0, nmb1,
            gks, gke, gjs, gje, gis, gie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real emax = 0.0;
      const int kmin = three_d ? ((k - 1 > gks) ? (k - 1) : gks) : k;
      const int kmax = three_d ? ((k + 1 < gke) ? (k + 1) : gke) : k;
      const int jmin = multi_d ? ((j - 1 > gjs) ? (j - 1) : gjs) : j;
      const int jmax = multi_d ? ((j + 1 < gje) ? (j + 1) : gje) : j;
      const int imin = (i - 1 > gis) ? (i - 1) : gis;
      const int imax = (i + 1 < gie) ? (i + 1) : gie;
      for (int kk = kmin; kk <= kmax; ++kk) {
        for (int jj = jmin; jj <= jmax; ++jj) {
          for (int ii = imin; ii <= imax; ++ii) {
            if (excise_enabled && excise_mask_(m, kk, jj, ii) != 0) continue;
            emax = fmax(emax, u0_(m, IEN, kk, jj, ii));
          }
        }
      }
      etot_max_(m, k, j, i) = emax;
    });
  }

  // Source terms and AMR transfers can change the conservative energy without updating
  // the auxiliary field directly. Synchronize only after the updated neighboring states
  // are available across MeshBlock boundaries.
  par_for("mhd_dual_energy_sync", DevExeSpace(), 0, nmb1,
          gks, gke, gjs, gje, gis, gie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    if (excise_enabled && excise_mask_(m, k, j, i) != 0) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyMHDThermalFloors(
          eos, dens, excise_eint, efloor_used, tfloor_used);
      u0_(m, de_idx, k, j, i) = eint_aux;
      w0_(m, de_idx, k, j, i) = eint_aux;
      return;
    }
    const Real bx = BccX(b0_, m, k, j, i);
    const Real by = BccY(b0_, m, k, j, i);
    const Real bz = BccZ(b0_, m, k, j, i);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
    Real eint = u0_(m, IEN, k, j, i) - e_k - e_m;
    const Real local_etot_max = (eta2 > 0.0) ? etot_max_(m, k, j, i) : 0.0;
    Real eint_aux = u0_(m, de_idx, k, j, i);
    if (DualEnergySyncEligible(eint, local_etot_max, eta2) &&
        eos_general::MHDEnergyChannelResolvesEint(eint, u0_(m, IEN, k, j, i), e_m,
                                                   eta1)) {
      eint_aux = eint;
    } else if ((eint > 0.0) && !(eint > eos_general::kDualEnergyMagneticFraction*e_m)) {
      // Rejected by the magnetic test alone: unless the cell is being shocked
      // (general_c2p_mhd.hpp).  A shocked cell bypasses eta2 as well, because the
      // inversion took this energy there regardless of eta2 before the magnetic test.
      const int im = (i - 1 > gis) ? (i - 1) : gis;
      const int ip = (i + 1 < gie) ? (i + 1) : gie;
      Real dv = u0_(m, IM1, k, j, im)/fmax(u0_(m, IDN, k, j, im), eos.dfloor) -
                u0_(m, IM1, k, j, ip)/fmax(u0_(m, IDN, k, j, ip), eos.dfloor);
      if (multi_d) {
        const int jm = (j - 1 > gjs) ? (j - 1) : gjs;
        const int jp = (j + 1 < gje) ? (j + 1) : gje;
        dv += u0_(m, IM2, k, jm, i)/fmax(u0_(m, IDN, k, jm, i), eos.dfloor) -
              u0_(m, IM2, k, jp, i)/fmax(u0_(m, IDN, k, jp, i), eos.dfloor);
      }
      if (three_d) {
        const int km = (k - 1 > gks) ? (k - 1) : gks;
        const int kp = (k + 1 < gke) ? (k + 1) : gke;
        dv += u0_(m, IM3, km, j, i)/fmax(u0_(m, IDN, km, j, i), eos.dfloor) -
              u0_(m, IM3, kp, j, i)/fmax(u0_(m, IDN, kp, j, i), eos.dfloor);
      }
      if (dv > 0.0) {
        bool efloor_aux = false, tfloor_aux = false;
        const Real eint_aux_floored = eos_general::ApplyMHDThermalFloors(
            eos, dens, eint_aux, efloor_aux, tfloor_aux);
        if (eos_general::MHDEnergyChannelCarriesShockHeat(
                eos, dens, eint, u0_(m, IEN, k, j, i), eint_aux_floored, dv, eta1)) {
          eint_aux = eint;
        }
      }
    }
    bool efloor_used = false, tfloor_used = false;
    eint_aux = eos_general::ApplyMHDThermalFloors(eos, dens, eint_aux, efloor_used,
                                                  tfloor_used);
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
}

void MHD::SynchronizeRestrictedDualEnergyField() {
  SynchronizeRestrictedDualEnergyField(coarse_u0, coarse_w0);
}

void MHD::SynchronizeRestrictedDualEnergyField(
    DvceArray5D<Real> &coarse_cons, DvceArray5D<Real> &coarse_prim) {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.cis;
  const int ie = indcs.cie;
  const int js = indcs.cjs;
  const int je = indcs.cje;
  const int ks = indcs.cks;
  const int ke = indcs.cke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto cu = coarse_cons;
  auto cw = coarse_prim;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  // Restriction already updated the conserved auxiliary field. Mirror it into the
  // primitive-side storage used by the coarse-boundary prolongation path.  The two
  // flavours keep different things in the two slots: the non-relativistic auxiliary is
  // an internal-energy density and the conserved and primitive forms are the same
  // number, while the GR one is kappa with D*kappa conserved, so the mirror is a
  // division by the restricted density.
  const bool pdv = dual_energy_pdv;
  par_for("dual_energy_sync_restricted_mhd", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(cu(m, IDN, k, j, i), eos.dfloor);
    if (pdv) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyMHDThermalFloors(
          eos, dens, cu(m, de_idx, k, j, i), efloor_used, tfloor_used);
      cu(m, de_idx, k, j, i) = eint_aux;
      cw(m, de_idx, k, j, i) = eint_aux;
      return;
    }
    Real kappa = cu(m, de_idx, k, j, i)/dens;
    if (!(isfinite(kappa) && kappa > eos.sfloor)) { kappa = eos.sfloor; }
    cu(m, de_idx, k, j, i) = dens*kappa;
    cw(m, de_idx, k, j, i) = kappa;
  });
}

void MHD::RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                       const int new_gids, const int new_nmb_local) {
  if (!use_dual_energy) return;
  if (new_nmb_local <= 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = new_nmb_local - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  // Step 9 of AMR regrid operates on the new rank-local MeshBlock count, not the old
  // pack size stored in pmy_pack->nmb_thispack. Restrict this kernel to the new local
  // range so it never walks into stale pack slots from the pre-regrid layout.
  const bool pdv = dual_energy_pdv;
  par_for("dual_energy_repair_refined_mhd", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag.d_view(n2o.d_view(m + new_gids)) <= 0) return;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    if (pdv) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyMHDThermalFloors(
          eos, dens, u0_(m, de_idx, k, j, i), efloor_used, tfloor_used);
      u0_(m, de_idx, k, j, i) = eint_aux;
      w0_(m, de_idx, k, j, i) = eint_aux;
      return;
    }
    Real kappa = u0_(m, de_idx, k, j, i)/dens;
    if (!(isfinite(kappa) && kappa > eos.sfloor)) { kappa = eos.sfloor; }
    u0_(m, de_idx, k, j, i) = dens*kappa;
    w0_(m, de_idx, k, j, i) = kappa;
  });
}

}  // namespace mhd
