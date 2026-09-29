//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file saha_table_hyd.cpp
//! \brief derived classes that implement tabulated LTE EOS variants in NR hydro

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "eos/lte_table_utils.hpp"
#include "eos/saha_table_utils.hpp"
#include "hydro/hydro.hpp"

namespace problem_runtime {
void GetExcisionState(const Real t, bool &enabled, Real &radius, Real &density,
    Real &eint,
                      Real &center_x, Real &center_y, Real &center_z);
}

namespace {

KOKKOS_INLINE_FUNCTION
bool InsideExcisionZone(const Real x, const Real y, const Real z,
                        const Real center_x, const Real center_y, const Real center_z,
                        const Real radius2) {
  const Real dx = x - center_x;
  const Real dy = y - center_y;
  const Real dz = z - center_z;
  return (dx*dx + dy*dy + dz*dz) <= radius2;
}

KOKKOS_INLINE_FUNCTION
bool TabulatedHydroC2PNeedsRepair(const EOS_Data &eos, const HydCons1D &u,
                                  const bool use_dual, const Real dual_eta1,
                                  const Real eint_aux_in) {
  if (eos_general::NeedsHydroAtmosphereReset(eos, u)) return true;

  const Real di = 1.0/u.d;
  const Real v2 = di*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  if ((eos.vceil > 0.0) && (v2 > SQR(eos.vceil))) return true;

  const Real eint_floor = eos.HydroInternalEnergyDensityFloor(u.d);
  const Real eint_ceil = eos.HydroInternalEnergyDensityCeiling(u.d);
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));

  if (!use_dual) {
    return (eint_cons < eint_floor) || (eint_cons > eint_ceil);
  }

  const bool aux_ok = (eint_aux_in >= eint_floor) && (eint_aux_in <= eint_ceil);
  if (!aux_ok) return true;

  const bool use_cons_e =
      (eint_cons > 0.0) &&
      ((dual_eta1 <= 0.0) || (eint_cons > dual_eta1*fmax(u.e, 1.0e-18)));
  if (!use_cons_e) return false;
  return (eint_cons < eint_floor) || (eint_cons > eint_ceil);
}

void TabulatedHydroConsToPrim(EquationOfState *peos, DvceArray5D<Real> &cons,
                              DvceArray5D<Real> &prim, const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nhyd = peos->pmy_pack->phydro->nhydro;
  int &nscal = peos->pmy_pack->phydro->nscalars;
  const bool use_dual = peos->pmy_pack->phydro->use_dual_energy;
  const int dual_idx = peos->pmy_pack->phydro->dual_energy_idx;
  const Real dual_eta1 = peos->pmy_pack->phydro->dual_energy_eta1;
  int &nmb = peos->pmy_pack->nmb_thispack;
  auto &eos = peos->eos_data;
  auto &fofc_ = peos->pmy_pack->phydro->fofc;
  auto &mbsize = peos->pmy_pack->pmb->mb_size;
  auto &mesh_indcs = peos->pmy_pack->pmesh->mb_indcs;
  const bool lat_enabled = peos->pmy_pack->lat_active_mask_enabled;
  auto active_indices = peos->pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? peos->pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  const int mesh_is = mesh_indcs.is;
  const int mesh_js = mesh_indcs.js;
  const int mesh_ks = mesh_indcs.ks;
  const int mesh_nx1 = mesh_indcs.nx1;
  const int mesh_nx2 = mesh_indcs.nx2;
  const int mesh_nx3 = mesh_indcs.nx3;

  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(peos->pmy_pack->pmesh->time, excise_enabled,
      excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
  const Real excise_r2 = excise_radius*excise_radius;

  const int ni = (iu - il + 1);
  const int nji = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;
  // Energy-ledger tally of floor/ceiling/excision resets: per-block atomic adds (floors
  // fire in few cells), no host-blocking reduction; active only inside a gravity
  // energy window.
  const bool floor_ledger = peos->pmy_pack->phydro->floor_energy_ledger && !only_testfloors;
  auto floor_block = peos->pmy_pack->phydro->floor_energy_block;
  const Real floor_weight = peos->pmy_pack->phydro->floor_energy_weight;
  const int floor_block_count = floor_block.extent_int(0);

  // Plain parallel_for: the former floor tallies fed only the event-log output and made
  // every C2P a host-blocking reduction.
  par_for("tabulated_hyd_c2p", DevExeSpace(), 0, nmkji-1,
  KOKKOS_LAMBDA(const int idx) {
    int a = idx/nkji;
    int m = lat_enabled ? active_indices(a) : a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;
    const bool tally = floor_ledger && (m < floor_block_count) &&
                       (i >= mesh_is) && (i < mesh_is + mesh_nx1) &&
                       (j >= mesh_js) && (j < mesh_js + mesh_nx2) &&
                       (k >= mesh_ks) && (k < mesh_ks + mesh_nx3);
    const Real energy_old = cons(m, IEN, k, j, i);
    const Real cell_volume = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;

    bool excised = false;
    if (excise_enabled) {
      const Real x = CellCenterX(i - mesh_is, mesh_nx1,
                                 mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - mesh_js, mesh_nx2,
                                 mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - mesh_ks, mesh_nx3,
                                 mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      excised = InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z, excise_r2);
    }

    if (excised) {
      HydPrim1D w;
      w.d = excise_density;
      w.vx = 0.0;
      w.vy = 0.0;
      w.vz = 0.0;
      w.e = excise_eint;
      if (only_testfloors) {
        fofc_(m, k, j, i) = true;
        return;
      }

      prim(m, IDN, k, j, i) = w.d;
      prim(m, IVX, k, j, i) = w.vx;
      prim(m, IVY, k, j, i) = w.vy;
      prim(m, IVZ, k, j, i) = w.vz;
      prim(m, IEN, k, j, i) = w.e;

      HydCons1D u;
      eos_general::SingleP2C_GeneralHyd(w, u);
      cons(m, IDN, k, j, i) = u.d;
      cons(m, IM1, k, j, i) = u.mx;
      cons(m, IM2, k, j, i) = u.my;
      cons(m, IM3, k, j, i) = u.mz;
      cons(m, IEN, k, j, i) = u.e;
      if (tally) Kokkos::atomic_add(&floor_block(m),
          floor_weight*(u.e - energy_old)*cell_volume);
      for (int n = nhyd; n < (nhyd + nscal); ++n) {
        cons(m, n, k, j, i) = 0.0;
        prim(m, n, k, j, i) = 0.0;
      }
      if (use_dual) {
        cons(m, dual_idx, k, j, i) = w.e;
        prim(m, dual_idx, k, j, i) = w.e;
      }
      return;
    }

    HydCons1D u;
    u.d = cons(m, IDN, k, j, i);
    u.mx = cons(m, IM1, k, j, i);
    u.my = cons(m, IM2, k, j, i);
    u.mz = cons(m, IM3, k, j, i);
    u.e = cons(m, IEN, k, j, i);

    if (only_testfloors &&
        !TabulatedHydroC2PNeedsRepair(eos, u, use_dual, dual_eta1,
                                      use_dual ? cons(m, dual_idx, k, j, i) : 0.0)) {
      return;
    }

    HydPrim1D w;
    bool dfloor_used = false, efloor_used = false, tfloor_used = false;
    bool vceiling_used = false;
    Real eint_aux_out = 0.0;
    if (!use_dual) {
      eos_general::SingleC2P_GeneralHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                                        vceiling_used);
    } else {
      eos_general::SingleC2P_GeneralHydDual(
          u, eos, cons(m, dual_idx, k, j, i), dual_eta1, w, eint_aux_out, dfloor_used,
          efloor_used, tfloor_used, vceiling_used);
    }

    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        fofc_(m, k, j, i) = true;
      }
    } else {
      prim(m, IDN, k, j, i) = w.d;
      prim(m, IVX, k, j, i) = w.vx;
      prim(m, IVY, k, j, i) = w.vy;
      prim(m, IVZ, k, j, i) = w.vz;
      prim(m, IEN, k, j, i) = w.e;

      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        cons(m, IDN, k, j, i) = u.d;
        cons(m, IM1, k, j, i) = u.mx;
        cons(m, IM2, k, j, i) = u.my;
        cons(m, IM3, k, j, i) = u.mz;
        cons(m, IEN, k, j, i) = u.e;
        if (tally) {
          Kokkos::atomic_add(&floor_block(m),
              floor_weight*(u.e - energy_old)*cell_volume);
        }
      }
      for (int n = nhyd; n < (nhyd + nscal); ++n) {
        if (cons(m, n, k, j, i) < 0.0) {
          cons(m, n, k, j, i) = 0.0;
        }
        prim(m, n, k, j, i) = cons(m, n, k, j, i)/u.d;
      }
      if (use_dual) {
        cons(m, dual_idx, k, j, i) = eint_aux_out;
        prim(m, dual_idx, k, j, i) = eint_aux_out;
      }
    }
  });
}

void TabulatedHydroPrimToCons(EquationOfState *peos, const DvceArray5D<Real> &prim,
                              DvceArray5D<Real> &cons, const int il, const int iu,
                              const int jl, const int ju, const int kl, const int ku) {
  int &nhyd = peos->pmy_pack->phydro->nhydro;
  int &nscal = peos->pmy_pack->phydro->nscalars;
  const bool use_dual = peos->pmy_pack->phydro->use_dual_energy;
  const int dual_idx = peos->pmy_pack->phydro->dual_energy_idx;
  int &nmb = peos->pmy_pack->nmb_thispack;
  const bool lat_enabled = peos->pmy_pack->lat_active_mask_enabled;
  auto active_indices = peos->pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? peos->pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;

  par_for("tabulated_hyd_p2c", DevExeSpace(), 0, (nwork - 1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    HydPrim1D w;
    w.d = prim(m, IDN, k, j, i);
    w.vx = prim(m, IVX, k, j, i);
    w.vy = prim(m, IVY, k, j, i);
    w.vz = prim(m, IVZ, k, j, i);
    w.e = prim(m, IEN, k, j, i);

    HydCons1D u;
    eos_general::SingleP2C_GeneralHyd(w, u);

    cons(m, IDN, k, j, i) = u.d;
    cons(m, IM1, k, j, i) = u.mx;
    cons(m, IM2, k, j, i) = u.my;
    cons(m, IM3, k, j, i) = u.mz;
    cons(m, IEN, k, j, i) = u.e;
    for (int n = nhyd; n < (nhyd + nscal); ++n) {
      cons(m, n, k, j, i) = u.d*prim(m, n, k, j, i);
    }
    if (use_dual) {
      cons(m, dual_idx, k, j, i) = prim(m, dual_idx, k, j, i);
    }
  });
}

}  // namespace

SahaTableHydro::SahaTableHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  saha_table_utils::InitializeSahaTableEOS("hydro", pp, pin, eos_data, __FILE__);
}

LTETableHydro::LTETableHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  lte_table_utils::InitializeLTETableEOS("hydro", pp, pin, eos_data, __FILE__);
}

void SahaTableHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                const bool only_testfloors,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku) {
  TabulatedHydroConsToPrim(this, cons, prim, only_testfloors, il, iu, jl, ju, kl, ku);
}

void LTETableHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                               const bool only_testfloors,
                               const int il, const int iu, const int jl, const int ju,
                               const int kl, const int ku) {
  TabulatedHydroConsToPrim(this, cons, prim, only_testfloors, il, iu, jl, ju, kl, ku);
}

void SahaTableHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku) {
  TabulatedHydroPrimToCons(this, prim, cons, il, iu, jl, ju, kl, ku);
}

void LTETableHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                               const int il, const int iu, const int jl, const int ju,
                               const int kl, const int ku) {
  TabulatedHydroPrimToCons(this, prim, cons, il, iu, jl, ju, kl, ku);
}
