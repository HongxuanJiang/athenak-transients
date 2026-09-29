//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic hydro

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealHydro::IdealHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.hydro_eos = HydroEOSModel::gamma_law;
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list. Number of times floors used stored into event counters.

void IdealHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int dual_idx = pmy_pack->phydro->dual_energy_idx;
  bool use_dual = pmy_pack->phydro->use_dual_energy;
  const Real dual_eta1 = pmy_pack->phydro->dual_energy_eta1;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->phydro->fofc;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_indcs = pmy_pack->pmesh->mb_indcs;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
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
  Real excise_r2 = excise_radius * excise_radius;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  int nfloord_=0, nfloore_=0, nfloort_=0, nceilv_=0;
  Real floor_energy_ = 0.0, floor_mass_ = 0.0;
  const Real floor_weight = pmy_pack->phydro->floor_energy_weight;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumt, int &sumv,
                Real &de_sum, Real &dm_sum) {
    int a = (idx)/nkji;
    int m = lat_enabled ? active_indices(a) : a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;
    const bool interior = (i >= mesh_is) && (i < mesh_is + mesh_nx1) &&
                          (j >= mesh_js) && (j < mesh_js + mesh_nx2) &&
                          (k >= mesh_ks) && (k < mesh_ks + mesh_nx3);
    const Real cell_volume = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    const Real energy_old = cons(m,IEN,k,j,i);
    const Real density_old = cons(m,IDN,k,j,i);

    bool excised = false;
    if (excise_enabled) {
      Real x = CellCenterX(i - mesh_is, mesh_nx1,
                           mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      Real y = CellCenterX(j - mesh_js, mesh_nx2,
                           mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      Real z = CellCenterX(k - mesh_ks, mesh_nx3,
                           mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      Real dx = x - sink_x;
      Real dy = y - sink_y;
      Real dz = z - sink_z;
      excised = (dx*dx + dy*dy + dz*dz <= excise_r2);
    }

    if (excised) {
      HydPrim1D w;
      w.d = excise_density;
      w.vx = 0.0;
      w.vy = 0.0;
      w.vz = 0.0;
      w.e = excise_eint;
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

      HydCons1D u;
      SingleP2C_IdealHyd(w, u);
      cons(m,IDN,k,j,i) = u.d;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      cons(m,IEN,k,j,i) = u.e;
      if (interior) {
        de_sum += floor_weight*(u.e - energy_old)*cell_volume;
        dm_sum += floor_weight*(u.d - density_old)*cell_volume;
      }

      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        cons(m,n,k,j,i) = 0.0;
        prim(m,n,k,j,i) = 0.0;
      }
      if (use_dual) {
        cons(m,dual_idx,k,j,i) = w.e;
        prim(m,dual_idx,k,j,i) = w.e;
      }
      return;
    }

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // call c2p function
    // (inline function in ideal_c2p_hyd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false, tfloor_used=false;
    bool vceiling_used=false;
    Real eint_aux_out = 0.0;
    if (!use_dual) {
      SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used,
                         vceiling_used);
    } else {
      if (NeedsIdealHydroAtmosphereReset(eos, u)) {
        ResetIdealHydroAtmosphereState(eos, u, w, dfloor_used);
        eint_aux_out = w.e;
      } else {
        w.d = u.d;
        Real di = 1.0/u.d;
        w.vx = di*u.mx;
        w.vy = di*u.my;
        w.vz = di*u.mz;
        const Real eint_floor = eos.pfloor/(eos.gamma - 1.0);
        const Real eint_cons = u.e - 0.5*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))/u.d;
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
            (eint_cons > 0.0) &&
            (dual_eta1 <= 0.0 || eint_cons > dual_eta1*fmax(u.e, 1.0e-18));
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
        u.e = w.e + 0.5*w.d*v2;
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

      // update counters and reset conserved state whenever a limiter/floor was used
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (tfloor_used) {sumt++;}
      if (vceiling_used) {sumv++;}
      if (dfloor_used || efloor_used || tfloor_used || vceiling_used) {
        SingleP2C_IdealHyd(w, u);
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;
        if (interior) {
          de_sum += floor_weight*(u.e - energy_old)*cell_volume;
          dm_sum += floor_weight*(u.d - density_old)*cell_volume;
        }
      }
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        // apply scalar floor
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
      if (use_dual) {
        cons(m,dual_idx,k,j,i) = eint_aux_out;
        prim(m,dual_idx,k,j,i) = eint_aux_out;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nfloort_),
     Kokkos::Sum<int>(nceilv_), Kokkos::Sum<Real>(floor_energy_), Kokkos::Sum<Real>(floor_mass_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.eos_floor_energy += floor_energy_;
    pmy_pack->pmesh->ecounter.eos_floor_mass += floor_mass_;
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_tfloor += nfloort_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables. Operates over range of cells given
//! in argument list.  Floors never needed.

void IdealHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int dual_idx = pmy_pack->phydro->dual_energy_idx;
  bool use_dual = pmy_pack->phydro->use_dual_energy;
  int &nmb = pmy_pack->nmb_thispack;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;

  par_for("hyd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    // load single state primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealHyd(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
    if (use_dual) {
      cons(m,dual_idx,k,j,i) = prim(m,dual_idx,k,j,i);
    }
  });

  return;
}
