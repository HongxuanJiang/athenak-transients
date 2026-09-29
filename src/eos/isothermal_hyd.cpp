//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_hyd.cpp
//! \brief derived class that implements isothermal EOS for nonrelativistic hydro

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IsothermalHydro::IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.hydro_eos = HydroEOSModel::isothermal;
  eos_data.is_ideal = false;
  eos_data.is_gamma_law = false;
  eos_data.iso_cs = pin->GetReal("hydro","iso_sound_speed");
  eos_data.gamma = 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IsothermalHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! non-relativistic hydrodynamics with an isothermal EOS.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IsothermalHyd(HydCons1D &u, const Real &dfloor_,
                             HydPrim1D &w, bool &dfloor_used) {
  if ((u.d < dfloor_) ||
      ((u.d == dfloor_) && ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)))) {
    u.d = dfloor_;
    u.mx = 0.0;
    u.my = 0.0;
    u.mz = 0.0;
    dfloor_used = true;
  }
  w.d = u.d;
  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IsothermalHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic hydrodynamics with an isothermal gas EOS.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list.

void IsothermalHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                 const bool only_testfloors,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->phydro->fofc;
  Real dfloor = eos_data.dfloor;
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
  const Real excise_r2 = excise_radius * excise_radius;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  int nfloord_=0;
  Kokkos::parallel_reduce("isohyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd) {
    int a = (idx)/nkji;
    int m = lat_enabled ? active_indices(a) : a;
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
      excised = problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                    excise_r2);
    }
    if (excised) {
      HydPrim1D w;
      w.d = excise_density;
      w.vx = 0.0;
      w.vy = 0.0;
      w.vz = 0.0;
      if (only_testfloors) {
        fofc_(m,k,j,i) = true;
        sumd++;
        return;
      }
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      HydCons1D u;
      SingleP2C_IsothermalHyd(w, u);
      cons(m,IDN,k,j,i) = u.d;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        cons(m,n,k,j,i) = 0.0;
        prim(m,n,k,j,i) = 0.0;
      }
      return;
    }

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);

    // call c2p function
    HydPrim1D w;
    bool dfloor_used = false;
    SingleC2P_IsothermalHyd(u, dfloor, w, dfloor_used);
    // update counter, reset conserved if floor was hit
    if (dfloor_used) {
      SingleP2C_IsothermalHyd(w, u);
      cons(m,IDN,k,j,i) = u.d;
      cons(m,IM1,k,j,i) = u.mx;
      cons(m,IM2,k,j,i) = u.my;
      cons(m,IM3,k,j,i) = u.mz;
      sumd++;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used) {fofc_(m,k,j,i) = true;}
    } else {
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables. Operates over range of cells given
//! in argument list.  Floors never needed.

void IsothermalHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;

  par_for("isohyd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    // load single state primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IsothermalHyd(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
