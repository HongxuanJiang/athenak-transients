//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_mhd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic mhd

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealMHD::IdealMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

void IdealMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int dual_idx = pmy_pack->pmhd->dual_energy_idx;
  bool use_dual = pmy_pack->pmhd->use_dual_energy;
  const Real dual_eta1 = pmy_pack->pmhd->dual_energy_eta1;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_indcs = pmy_pack->pmesh->mb_indcs;
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
  problem_runtime::GetExcisionState(pmy_pack->pmesh->time, excise_enabled,
      excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
  const Real excise_r2 = excise_radius * excise_radius;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0, nceilv_=0;
  Kokkos::parallel_reduce("mhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt, int &sumv) {
    int a = (idx)/nkji;
    int m = a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    bool excised = false;
    if (excise_enabled) {
      const Real x = CellCenterX(i - mesh_is, mesh_nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - mesh_js, mesh_nx2, mbsize.d_view(m).x2min,
                                 mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - mesh_ks, mesh_nx3, mbsize.d_view(m).x3min,
                                 mbsize.d_view(m).x3max);
      excised = problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                    excise_r2);
    }

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

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
        fofc_(m,k,j,i) = true;
        sumd++;
        return;
      }

      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      bcc(m,IBX,k,j,i) = w.bx;
      bcc(m,IBY,k,j,i) = w.by;
      bcc(m,IBZ,k,j,i) = w.bz;

      HydCons1D u_out;
      SingleP2C_IdealMHD(w, u_out);
      cons(m,IDN,k,j,i) = u_out.d;
      cons(m,IM1,k,j,i) = u_out.mx;
      cons(m,IM2,k,j,i) = u_out.my;
      cons(m,IM3,k,j,i) = u_out.mz;
      cons(m,IEN,k,j,i) = u_out.e;

      for (int n = nmhd; n < (nmhd + nscal); ++n) {
        cons(m,n,k,j,i) = 0.0;
        prim(m,n,k,j,i) = 0.0;
      }
      if (use_dual) {
        cons(m, dual_idx, k, j, i) = w.e;
        prim(m, dual_idx, k, j, i) = w.e;
      }
      return;
    }

    // call c2p function
    // (inline function in ideal_c2p_mhd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    bool vceiling_used=false;
    Real eint_aux_out = 0.0;
    bool eint_from_aux = false;
    if (!use_dual) {
      SingleC2P_IdealMHD(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                         vceiling_used);
    } else {
      if (NeedsIdealMHDAtmosphereReset(eos, u)) {
        ResetIdealMHDAtmosphereState(eos, u, w, dfloor_used);
        eint_aux_out = w.e;
      } else {
        w.d = u.d;
        Real di = 1.0/u.d;
        w.vx = di*u.mx;
        w.vy = di*u.my;
        w.vz = di*u.mz;
        const Real eint_floor = eos.pfloor/(eos.gamma - 1.0);
        const Real e_m = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
        const Real eint_cons =
            u.e - 0.5*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))/u.d - e_m;
        Real eint_aux = cons(m, dual_idx, k, j, i);
        if (eint_aux < eint_floor) {
          eint_aux = eint_floor;
          efloor_used = true;
        }
        if (eos.tfloor > 0.0 && ((eos.gamma - 1.0)*eint_aux*di < eos.tfloor)) {
          eint_aux = w.d*eos.tfloor/(eos.gamma - 1.0);
          tfloor_used = true;
        }
        const Real eint_aux_ceil = eos.HydroInternalEnergyDensityCeiling(w.d);
        if (eint_aux > eint_aux_ceil) {
          eint_aux = eint_aux_ceil;
          efloor_used = true;
        }
        const bool use_cons_e =
            eos_general::MHDEnergyChannelResolvesEint(eint_cons, u.e, e_m,
                                                       dual_eta1);
        eint_from_aux = !use_cons_e;
        w.e = use_cons_e ? eint_cons : eint_aux;
        if (w.e < eint_floor) {
          w.e = eint_floor;
          efloor_used = true;
        }
        if (eos.tfloor > 0.0 && ((eos.gamma - 1.0)*w.e*di < eos.tfloor)) {
          w.e = w.d*eos.tfloor/(eos.gamma - 1.0);
          tfloor_used = true;
        }
        if (w.e > eint_aux_ceil) {
          w.e = eint_aux_ceil;
          efloor_used = true;
        }
        Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
        Real vmag = sqrt(v2);
        if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
          Real fac = eos.vceil/vmag;
          w.vx *= fac;
          w.vy *= fac;
          w.vz *= fac;
          v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
          vceiling_used = true;
        }
        u.mx = w.d*w.vx;
        u.my = w.d*w.vy;
        u.mz = w.d*w.vz;
        u.e = w.e + 0.5*w.d*v2 + e_m;
        eint_aux_out = eint_aux;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // update counters and reset conserved state whenever a limiter/floor was used
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (tfloor_used) {sumt++;}
      if (vceiling_used) {sumv++;}
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        MHDPrim1D w_in;
        w_in.d = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;
        HydCons1D u_out;
        SingleP2C_IdealMHD(w_in, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // needed below for scalar conversion
      } else if (eint_from_aux) {
        // The energy channel was rejected: see MHDEnergyChannelResolvesEint.
        cons(m,IEN,k,j,i) = u.e;
      }
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
      if (use_dual) {
        cons(m, dual_idx, k, j, i) = eint_aux_out;
        prim(m, dual_idx, k, j, i) = eint_aux_out;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_),
     Kokkos::Sum<int>(nceilv_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.  Does not change cell- or face-centered magnetic fields.

void IdealMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int dual_idx = pmy_pack->pmhd->dual_energy_idx;
  bool use_dual = pmy_pack->pmhd->use_dual_energy;
  int &nmb = pmy_pack->nmb_thispack;
  const int nwork = nmb;
  if (nwork <= 0) return;

  par_for("mhd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = a;
    // load single state primitive variables
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
    SingleP2C_IdealMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
    if (use_dual) {
      cons(m,dual_idx,k,j,i) = prim(m,dual_idx,k,j,i);
    }
  });

  return;
}
