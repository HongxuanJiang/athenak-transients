//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_dual_energy.cpp
//! \brief Dual-energy formalism for Newtonian hydro with EOS-aware thermal support

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "hydro.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool DualEnergySyncEligible(const Real eint_cons, const Real local_etot_max,
                            const Real eta2) {
  if (eint_cons <= 0.0) return false;
  if (eta2 <= 0.0) return true;
  return (eint_cons > eta2*fmax(local_etot_max, 1.0e-18));
}

}  // namespace

namespace hydro {

TaskStatus Hydro::DualEnergyStep(Driver *pdrive, int stage) {
  if (!use_dual_energy) return TaskStatus::complete;
  if (dual_energy_pdv) {
    const Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
    ApplyDualEnergyFormalism(beta_dt);
    return TaskStatus::complete;
  }
  // The GR adiabat is advected by an exact conservation law and has no source term.
  // Its eta2 resynchronization reads only what this rank holds, so it runs here, ahead
  // of the conserved send, and the exchange installs the result in every ghost cell.
  SynchronizeDualEnergyFieldFromAdiabat();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn Real AdiabatFromPrimitive()
//! \brief kappa = p/rho^Gamma from the published primitive state, floored at the entropy
//! floor the GR inversion already uses.

KOKKOS_INLINE_FUNCTION
Real AdiabatFromPrimitive(const Real rho, const Real eint, const Real gamma,
                          const Real sfloor) {
  const Real p = (gamma - 1.0)*eint;
  Real kappa = ((rho > 0.0) && (p > 0.0)) ? p*pow(rho, -gamma) : 0.0;
  if (!(isfinite(kappa) && kappa > sfloor)) { kappa = sfloor; }
  return kappa;
}

void Hydro::InitializeDualEnergyFieldFromAdiabat() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;
  // The ghost range, not the active zone: this runs after the last conserved exchange
  // of Driver::Initialize, so an active-only seed would leave every ghost cell unseeded
  // for the first stage's flux kernel.  ConToPrim has already run over the same range.
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int ng = indcs.ng;
  const int gis = is - ng, gie = ie + ng;
  const int gjs = multi_d ? (js - ng) : js, gje = multi_d ? (je + ng) : je;
  const int gks = three_d ? (ks - ng) : ks, gke = three_d ? (ke + ng) : ke;
  par_for("hydro_dual_adiabat_init", DevExeSpace(), 0, nmb1,
          gks, gke, gjs, gje, gis, gie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real kappa = AdiabatFromPrimitive(w0_(m, IDN, k, j, i), w0_(m, IEN, k, j, i),
                                            eos.gamma, eos.sfloor);
    w0_(m, de_idx, k, j, i) = kappa;
    u0_(m, de_idx, k, j, i) = u0_(m, IDN, k, j, i)*kappa;
  });
}

void Hydro::SynchronizeDualEnergyFieldFromAdiabat() {
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
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return;

  // The eta2 test is written per unit rest mass, which keeps it free of sqrt(det g):
  // q = tau/D is a ratio of two conserved variables carrying the same densitization
  // and eps = u/rho a ratio of two primitives; in the non-relativistic limit it reduces
  // to the Newtonian `eint > eta2*max(E_tot)` exactly.  Where the energy channel is
  // still trustworthy the advected adiabat is reset from it, which is how the auxiliary
  // learns about shock heating.  Owned cells only; the stencil reaches one layer into
  // the ghosts, where the previous ghost recovery published its tau/D.
  par_for("hydro_dual_adiabat_sync", DevExeSpace(), 0, nwork1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real dens_cons = u0_(m, IDN, k, j, i);
    const Real rho = w0_(m, IDN, k, j, i);
    Real kappa = (dens_cons > 0.0) ? u0_(m, de_idx, k, j, i)/dens_cons : 0.0;
    if (!(isfinite(kappa) && kappa > 0.0)) { kappa = eos.sfloor; }

    const Real eint = w0_(m, IEN, k, j, i);
    const Real eps_energy = (rho > 0.0) ? eint/rho : 0.0;
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
      kappa = AdiabatFromPrimitive(rho, eint, eos.gamma, eos.sfloor);
    }
    if (kappa < eos.sfloor) { kappa = eos.sfloor; }
    // Bound the adiabat by the energy the cell owns: a cell whose density reaches its
    // floor turns (D kappa)/D into a finite numerator over a floor, and an adiabat
    // implying more internal energy than the total is not a state any fluid can be in.
    // tau/D and rho are the values the last inversion published, one stage old, which
    // a bound can afford.
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

void Hydro::InitializeDualEnergyFieldFromTotal() {
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
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;

  // Seed the auxiliary internal-energy field from the conserved total energy so restart
  // activation and fresh starts begin from a synchronized state.
  par_for("dual_energy_init", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    bool efloor_used = false, tfloor_used = false;
    Real eint = u0_(m, IEN, k, j, i) - e_k;
    eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint, efloor_used,
                                                tfloor_used);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void Hydro::ApplyDualEnergyFormalism(const Real dt) {
  if (!use_dual_energy || !dual_energy_pdv) return;

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
  auto vf1_ = FluxBand(dual_vf.x1f);
  auto vf2_ = FluxBand(dual_vf.x2f);
  auto vf3_ = FluxBand(dual_vf.x3f);
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (dt/mesh_dt) : 0.0;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return;
  // Evolve the auxiliary internal energy with interface-velocity compression. The
  // conservative-to-auxiliary synchronization is deferred until after boundary exchange
  // and prolongation, when the neighboring total-energy state is current across
  // MeshBlock boundaries.
  par_for("dual_energy_formalism", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_dt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : dt;
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
    eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint, efloor_used,
                                                tfloor_used);
    if (eos.is_gamma_law) {
      eint *= exp(-(eos.gamma - 1.0)*divv*block_dt);
    } else {
      const Real pressure = eos.PressureFromRhoEint(dens, eint);
      eint -= pressure*divv*block_dt;
    }
    efloor_used = false;
    tfloor_used = false;
    eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint, efloor_used,
                                                tfloor_used);

    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void Hydro::SynchronizeDualEnergyFieldFromTotal(bool ghost_bands_only) {
  // The GR adiabat is resynchronized in DualEnergyStep, ahead of the send.
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
  auto excise_mask_ = dual_excise_mask;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
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
  const Real eta2 = dual_energy_eta2;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return;
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
    par_for("dual_energy_excise_mask", DevExeSpace(), 0, nwork1, gks, gke, gjs, gje, gis,
        gie,
    KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
      const int m = lat_enabled ? active_indices(a) : a;
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

  // Source terms and AMR transfers can change the conservative energy without updating
  // the auxiliary field directly. Synchronize only after the updated neighboring states
  // are available across MeshBlock boundaries.
  // ghost_bands_only (LAT ghost refreshes): only the ghost zones changed, and the eta2
  // test is a one-cell stencil, so only the ghosts plus one interior layer can see a
  // different input; every deeper cell keeps the value the last full pass wrote.
  // slab bounds: {kl, ku, jl, ju, il, iu}
  int slabs[6][6];
  int nslab = 0;
  auto add_slab = [&](int kl, int ku, int jl, int ju, int il, int iu) {
    slabs[nslab][0] = kl; slabs[nslab][1] = ku; slabs[nslab][2] = jl;
    slabs[nslab][3] = ju; slabs[nslab][4] = il; slabs[nslab][5] = iu;
    ++nslab;
  };
  if (!ghost_bands_only) {
    add_slab(gks, gke, gjs, gje, gis, gie);
  } else {
    const int kin_l = three_d ? (ks + 1) : ks, kin_u = three_d ? (ke - 1) : ke;
    const int jin_l = multi_d ? (js + 1) : js, jin_u = multi_d ? (je - 1) : je;
    if (three_d) {
      add_slab(gks, ks, gjs, gje, gis, gie);
      add_slab(ke, gke, gjs, gje, gis, gie);
    }
    if (multi_d) {
      add_slab(kin_l, kin_u, gjs, js, gis, gie);
      add_slab(kin_l, kin_u, je, gje, gis, gie);
    }
    add_slab(kin_l, kin_u, jin_l, jin_u, gis, is);
    add_slab(kin_l, kin_u, jin_l, jin_u, ie, gie);
  }
  for (int s = 0; s < nslab; ++s) {
  par_for("dual_energy_sync", DevExeSpace(), 0, nwork1, slabs[s][0], slabs[s][1],
          slabs[s][2], slabs[s][3], slabs[s][4], slabs[s][5],
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    if (excise_enabled && excise_mask_(m, k, j, i) != 0) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyHydroThermalFloors(
          eos, dens, excise_eint, efloor_used, tfloor_used);
      u0_(m, de_idx, k, j, i) = eint_aux;
      w0_(m, de_idx, k, j, i) = eint_aux;
      return;
    }
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    Real eint = u0_(m, IEN, k, j, i) - e_k;
    // Local max(total energy) for the eta2 test over the 27-cell neighborhood; the
    // kernel only writes the auxiliary field, so u0(IEN) is unchanged while it runs.
    Real local_etot_max = 0.0;
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
            if (excise_enabled && excise_mask_(m, kk, jj, ii) != 0) continue;
            local_etot_max = fmax(local_etot_max, u0_(m, IEN, kk, jj, ii));
          }
        }
      }
    }
    Real eint_aux = u0_(m, de_idx, k, j, i);
    if (DualEnergySyncEligible(eint, local_etot_max, eta2)) {
      eint_aux = eint;
    }
    bool efloor_used = false, tfloor_used = false;
    eint_aux = eos_general::ApplyHydroThermalFloors(eos, dens, eint_aux, efloor_used,
                                                    tfloor_used);
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
  }
}

void Hydro::SynchronizeRestrictedDualEnergyField(bool lat_active_only) {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.cis;
  const int ie = indcs.cie;
  const int js = indcs.cjs;
  const int je = indcs.cje;
  const int ks = indcs.cks;
  const int ke = indcs.cke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_boundary_send_indices = lat_active_only ?
      pmy_pack->lat_active_indices.d_view : pmy_pack->lat_boundary_send_indices.d_view;
  const int nwork1 = lat_enabled ?
      ((lat_active_only ? pmy_pack->lat_nactive_thispack :
                          pmy_pack->lat_nboundary_send_thispack) - 1) : nmb1;
  if (nwork1 < 0) return;
  auto cu = coarse_u0;
  auto cw = coarse_w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  // Restriction already updated the conserved auxiliary field. Mirror it into the
  // primitive-side storage used by the coarse-boundary prolongation path.  The
  // non-relativistic auxiliary is an internal-energy density, the same number in both
  // slots; the GR one is kappa with D*kappa conserved, so the mirror divides by the
  // restricted density.
  const bool pdv = dual_energy_pdv;
  par_for("dual_energy_sync_restricted_hydro", DevExeSpace(), 0, nwork1, ks, ke, js, je,
      is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? lat_boundary_send_indices(a) : a;
    const Real dens = fmax(cu(m, IDN, k, j, i), eos.dfloor);
    if (pdv) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyHydroThermalFloors(
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

void Hydro::RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
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
  par_for("dual_energy_repair_refined_hydro", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag.d_view(n2o.d_view(m + new_gids)) <= 0) return;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    if (pdv) {
      bool efloor_used = false, tfloor_used = false;
      const Real eint_aux = eos_general::ApplyHydroThermalFloors(
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

}  // namespace hydro
