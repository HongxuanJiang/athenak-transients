//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file saha_table_mhd.cpp
//! \brief derived classes that implement tabulated LTE EOS variants in NR MHD

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "eos/lte_table_utils.hpp"
#include "eos/saha_table_utils.hpp"
#include "mhd/mhd.hpp"

namespace problem_runtime {
void GetExcisionState(const Real t, bool &enabled, Real &radius, Real &density,
    Real &eint,
                      Real &center_x, Real &center_y, Real &center_z);
}

namespace {

KOKKOS_INLINE_FUNCTION
Real BccX(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x1f(m, k, j, i) + b.x1f(m, k, j, i + 1));
}

KOKKOS_INLINE_FUNCTION
Real BccY(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x2f(m, k, j, i) + b.x2f(m, k, j + 1, i));
}

KOKKOS_INLINE_FUNCTION
Real BccZ(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x3f(m, k, j, i) + b.x3f(m, k + 1, j, i));
}

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
bool TabulatedMHDC2PNeedsRepair(const EOS_Data &eos, const MHDCons1D &u,
                                const bool use_dual, const Real dual_eta1,
                                const Real eint_aux_in) {
  if (eos_general::NeedsMHDAtmosphereReset(eos, u)) return true;

  const Real di = 1.0/u.d;
  const Real v2 = di*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  if ((eos.vceil > 0.0) && (v2 > SQR(eos.vceil))) return true;

  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  const Real eint_floor = eos.HydroInternalEnergyDensityFloor(u.d);
  const Real eint_ceil = eos.HydroInternalEnergyDensityCeiling(u.d);
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz)) - emag;

  if (!use_dual) {
    return (eint_cons < eint_floor) || (eint_cons > eint_ceil);
  }

  const bool aux_ok = (eint_aux_in >= eint_floor) && (eint_aux_in <= eint_ceil);
  if (!aux_ok) return true;

  const bool use_cons_e = eos_general::MHDEnergyChannelResolvesEint(
      eint_cons, u.e, emag, dual_eta1);
  if (!use_cons_e) return false;
  return (eint_cons < eint_floor) || (eint_cons > eint_ceil);
}

void TabulatedMHDConsToPrim(EquationOfState *peos, DvceArray5D<Real> &cons,
                            const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim,
                            DvceArray5D<Real> &bcc, const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nmhd = peos->pmy_pack->pmhd->nmhd;
  int &nscal = peos->pmy_pack->pmhd->nscalars;
  const bool use_dual = peos->pmy_pack->pmhd->use_dual_energy;
  const int dual_idx = peos->pmy_pack->pmhd->dual_energy_idx;
  const Real dual_eta1 = peos->pmy_pack->pmhd->dual_energy_eta1;
  int &nmb = peos->pmy_pack->nmb_thispack;
  auto &eos = peos->eos_data;
  auto &fofc_ = peos->pmy_pack->pmhd->fofc;
  auto &mbsize = peos->pmy_pack->pmb->mb_size;
  auto &mesh_indcs = peos->pmy_pack->pmesh->mb_indcs;
  const int nwork = nmb;
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

  int nfloord_ = 0, nfloore_ = 0, nfloort_ = 0, nceilv_ = 0;
  Kokkos::parallel_reduce("tabulated_mhd_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0,
      nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt, int &sumv) {
    int a = idx/nkji;
    int m = a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

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

    MHDCons1D u;
    u.d = cons(m, IDN, k, j, i);
    u.mx = cons(m, IM1, k, j, i);
    u.my = cons(m, IM2, k, j, i);
    u.mz = cons(m, IM3, k, j, i);
    u.e = cons(m, IEN, k, j, i);
    if (only_testfloors) {
      u.bx = bcc(m, IBX, k, j, i);
      u.by = bcc(m, IBY, k, j, i);
      u.bz = bcc(m, IBZ, k, j, i);
    } else {
      u.bx = BccX(b, m, k, j, i);
      u.by = BccY(b, m, k, j, i);
      u.bz = BccZ(b, m, k, j, i);
    }

    if (excised) {
      MHDPrim1D w;
      w.d = excise_density;
      w.vx = 0.0;
      w.vy = 0.0;
      w.vz = 0.0;
      w.e = excise_eint;
      w.bx = u.bx;
      w.by = u.by;
      w.bz = u.bz;
      if (only_testfloors) {
        fofc_(m, k, j, i) = true;
        sumd++;
        return;
      }
      prim(m, IDN, k, j, i) = w.d;
      prim(m, IVX, k, j, i) = w.vx;
      prim(m, IVY, k, j, i) = w.vy;
      prim(m, IVZ, k, j, i) = w.vz;
      prim(m, IEN, k, j, i) = w.e;
      bcc(m, IBX, k, j, i) = w.bx;
      bcc(m, IBY, k, j, i) = w.by;
      bcc(m, IBZ, k, j, i) = w.bz;

      MHDCons1D u_out;
      eos_general::SingleP2C_GeneralMHD(w, u_out);
      cons(m, IDN, k, j, i) = u_out.d;
      cons(m, IM1, k, j, i) = u_out.mx;
      cons(m, IM2, k, j, i) = u_out.my;
      cons(m, IM3, k, j, i) = u_out.mz;
      cons(m, IEN, k, j, i) = u_out.e;
      for (int n = nmhd; n < (nmhd + nscal); ++n) {
        cons(m, n, k, j, i) = 0.0;
        prim(m, n, k, j, i) = 0.0;
      }
      if (use_dual) {
        cons(m, dual_idx, k, j, i) = w.e;
        prim(m, dual_idx, k, j, i) = w.e;
      }
      return;
    }

    if (only_testfloors &&
        !TabulatedMHDC2PNeedsRepair(eos, u, use_dual, dual_eta1,
                                    use_dual ? cons(m, dual_idx, k, j, i) : 0.0)) {
      return;
    }

    MHDPrim1D w;
    bool dfloor_used = false, efloor_used = false, tfloor_used = false;
    bool vceiling_used = false;
    Real eint_aux_out = 0.0;
    bool eint_from_aux = false;
    if (!use_dual) {
      eos_general::SingleC2P_GeneralMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                                        vceiling_used);
    } else {
      eos_general::SingleC2P_GeneralMHDDual(
          u, eos, cons(m, dual_idx, k, j, i), dual_eta1, w, eint_aux_out, eint_from_aux,
          dfloor_used, efloor_used, tfloor_used, vceiling_used);
    }

    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        fofc_(m, k, j, i) = true;
        sumd++;
      }
    } else {
      prim(m, IDN, k, j, i) = w.d;
      prim(m, IVX, k, j, i) = w.vx;
      prim(m, IVY, k, j, i) = w.vy;
      prim(m, IVZ, k, j, i) = w.vz;
      prim(m, IEN, k, j, i) = w.e;
      bcc(m, IBX, k, j, i) = w.bx;
      bcc(m, IBY, k, j, i) = w.by;
      bcc(m, IBZ, k, j, i) = w.bz;

      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (tfloor_used) {sumt++;}
      if (vceiling_used) {sumv++;}
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        cons(m, IDN, k, j, i) = u.d;
        cons(m, IM1, k, j, i) = u.mx;
        cons(m, IM2, k, j, i) = u.my;
        cons(m, IM3, k, j, i) = u.mz;
        cons(m, IEN, k, j, i) = u.e;
      } else if (eint_from_aux) {
        // The energy channel was rejected: see MHDEnergyChannelResolvesEint.
        cons(m, IEN, k, j, i) = u.e;
      }
      for (int n = nmhd; n < (nmhd + nscal); ++n) {
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
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_),
     Kokkos::Sum<int>(nceilv_));

  if (only_testfloors) {
    peos->pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    peos->pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    peos->pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    peos->pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
    peos->pmy_pack->pmesh->ecounter.neos_vceil += nceilv_;
  }
}

void TabulatedMHDPrimToCons(EquationOfState *peos, const DvceArray5D<Real> &prim,
                            const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nmhd = peos->pmy_pack->pmhd->nmhd;
  int &nscal = peos->pmy_pack->pmhd->nscalars;
  const bool use_dual = peos->pmy_pack->pmhd->use_dual_energy;
  const int dual_idx = peos->pmy_pack->pmhd->dual_energy_idx;
  int &nmb = peos->pmy_pack->nmb_thispack;
  const int nwork = nmb;
  if (nwork <= 0) return;

  par_for("tabulated_mhd_p2c", DevExeSpace(), 0, (nwork - 1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = a;
    MHDPrim1D w;
    w.d = prim(m, IDN, k, j, i);
    w.vx = prim(m, IVX, k, j, i);
    w.vy = prim(m, IVY, k, j, i);
    w.vz = prim(m, IVZ, k, j, i);
    w.e = prim(m, IEN, k, j, i);
    w.bx = bcc(m, IBX, k, j, i);
    w.by = bcc(m, IBY, k, j, i);
    w.bz = bcc(m, IBZ, k, j, i);

    MHDCons1D u;
    eos_general::SingleP2C_GeneralMHD(w, u);
    cons(m, IDN, k, j, i) = u.d;
    cons(m, IM1, k, j, i) = u.mx;
    cons(m, IM2, k, j, i) = u.my;
    cons(m, IM3, k, j, i) = u.mz;
    cons(m, IEN, k, j, i) = u.e;
    for (int n = nmhd; n < (nmhd + nscal); ++n) {
      cons(m, n, k, j, i) = u.d*prim(m, n, k, j, i);
    }
    if (use_dual) {
      cons(m, dual_idx, k, j, i) = prim(m, dual_idx, k, j, i);
    }
  });
}

}  // namespace

SahaTableMHD::SahaTableMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  saha_table_utils::InitializeSahaTableEOS("mhd", pp, pin, eos_data, __FILE__);
}

LTETableMHD::LTETableMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  lte_table_utils::InitializeLTETableEOS("mhd", pp, pin, eos_data, __FILE__);
}

void SahaTableMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                              DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                              const bool only_testfloors, const int il, const int iu,
                              const int jl, const int ju, const int kl, const int ku) {
  TabulatedMHDConsToPrim(this, cons, b, prim, bcc, only_testfloors, il, iu, jl, ju, kl,
      ku);
}

void LTETableMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                             DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                             const bool only_testfloors, const int il, const int iu,
                             const int jl, const int ju, const int kl, const int ku) {
  TabulatedMHDConsToPrim(this, cons, b, prim, bcc, only_testfloors, il, iu, jl, ju, kl,
      ku);
}

void SahaTableMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                              DvceArray5D<Real> &cons, const int il, const int iu,
                              const int jl, const int ju, const int kl, const int ku) {
  TabulatedMHDPrimToCons(this, prim, bcc, cons, il, iu, jl, ju, kl, ku);
}

void LTETableMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                             DvceArray5D<Real> &cons, const int il, const int iu,
                             const int jl, const int ju, const int kl, const int ku) {
  TabulatedMHDPrimToCons(this, prim, bcc, cons, il, iu, jl, ju, kl, ku);
}
