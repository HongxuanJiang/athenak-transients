//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//! \brief function to compute hydro timestep across all MeshBlock(s) in a MeshBlockPack

#include <math.h>

#include <limits>
#include <iostream>
#include <algorithm> // min

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlockPack for hydrodynamic problems

TaskStatus Hydro::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }
  if (pdrive->hydro_lat && pmy_pack->lat_active_mask_enabled) {
    return TaskStatus::complete;
  }

  Mesh *pmesh = pmy_pack->pmesh;
  auto &indcs = pmesh->mb_indcs;
  const int il = indcs.is - indcs.ng;
  const int iu = indcs.ie + indcs.ng;
  const int jl = pmesh->multi_d ? (indcs.js - indcs.ng) : indcs.js;
  const int ju = pmesh->multi_d ? (indcs.je + indcs.ng) : indcs.je;
  const int kl = pmesh->three_d ? (indcs.ks - indcs.ng) : indcs.ks;
  const int ku = pmesh->three_d ? (indcs.ke + indcs.ng) : indcs.ke;
  const int nx1 = iu - il + 1;
  const int nx2 = ju - jl + 1;
  const int nx3 = ku - kl + 1;
  const int nmb = pmy_pack->nmb_thispack;
  const Real big_dt = std::numeric_limits<Real>::max();

  dtnew = big_dt;

  // capture class variables for kernel
  auto &w0_ = w0;
  auto &dtnew_eachmb_ = dtnew_eachmb;
  auto &eos = pmy_pack->phydro->peos->eos_data;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &is_special_relativistic_ = pmy_pack->pcoord->is_special_relativistic;
  auto &is_general_relativistic_ = pmy_pack->pcoord->is_general_relativistic;
  auto &is_dynamical_relativistic_ = pmy_pack->pcoord->is_dynamical_relativistic;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  const bool multi_d = pmesh->multi_d;
  const bool three_d = pmesh->three_d;

  if (pdrive->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("HydroNudt1", Kokkos::TeamPolicy<>(DevExeSpace(), nmb,
                            Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + il;
        k += kl;
        j += jl;

        Real cell_dt = mbsize.d_view(m).dx1/fabs(w0_(m,IVX,k,j,i));
        if (multi_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx2/
                                             fabs(w0_(m,IVY,k,j,i)));
        if (three_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx3/
                                             fabs(w0_(m,IVZ,k,j,i)));
        local_min = fmin(cell_dt, local_min);
      }, Kokkos::Min<Real>(mb_dt));
      dtnew_eachmb_.d_view(m) = mb_dt;
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  } else {
    // find smallest dx/(v +/- Cs) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("HydroNudt2", Kokkos::TeamPolicy<>(DevExeSpace(), nmb,
                            Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + il;
        k += kl;
        j += jl;

        Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

        if (is_general_relativistic_ || is_dynamical_relativistic_) {
          max_dv1 = 1.0;
          max_dv2 = 1.0;
          max_dv3 = 1.0;
        } else if (is_special_relativistic_) {
          Real v2 = SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i)) +
                    SQR(w0_(m,IVZ,k,j,i));
          Real lor = sqrt(1.0 + v2);
          Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

          Real lm, lp;
          eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVX,k,j,i),
                                      lor, lp, lm);
          max_dv1 = fmax(fabs(lm), lp);

          eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVY,k,j,i),
                                      lor, lp, lm);
          max_dv2 = fmax(fabs(lm), lp);

          eos.IdealSRHydroSoundSpeeds(w0_(m,IDN,k,j,i), p, w0_(m,IVZ,k,j,i),
                                      lor, lp, lm);
          max_dv3 = fmax(fabs(lm), lp);
        } else {
          Real cs;
          if (eos.use_e) {
            cs = sqrt(fmax(eos.HydroSoundSpeed2FromRhoEint(w0_(m,IDN,k,j,i),
                                                           w0_(m,IEN,k,j,i)), 0.0));
            if (eos.cs_ceil > 0.0) {
              cs = fmin(cs, eos.cs_ceil);
            }
          } else {
            cs = eos.iso_cs;
          }
          max_dv1 = fabs(w0_(m,IVX,k,j,i)) + cs;
          max_dv2 = fabs(w0_(m,IVY,k,j,i)) + cs;
          max_dv3 = fabs(w0_(m,IVZ,k,j,i)) + cs;
        }
        Real cell_dt = mbsize.d_view(m).dx1/max_dv1;
        if (multi_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx2/max_dv2);
        if (three_d) cell_dt = fmin(cell_dt, mbsize.d_view(m).dx3/max_dv3);
        local_min = fmin(cell_dt, local_min);
      }, Kokkos::Min<Real>(mb_dt));
      dtnew_eachmb_.d_view(m) = mb_dt;
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(dtnew));
  }

  dtnew_eachmb.template modify<DevExeSpace>();
  dtnew_hydro_cfl = dtnew;

  // compute timestep for diffusion
  if (pcond != nullptr) {
    pcond->NewTimeStep(w0, peos->eos_data);
  }
  if (pvisc != nullptr) {
    pvisc->NewTimeStep(w0, peos->eos_data);
  }
  // compute source terms timestep
  if (psrc != nullptr) {
    psrc->NewTimeStep(w0, peos->eos_data);
  }
  if (pmy_pack->pmesh->pgen->user_dt_func != nullptr) {
    dtnew = std::min(dtnew,
                     (pmy_pack->pmesh->pgen->user_dt_func)(pmy_pack->pmesh, pdrive));
  }

  return TaskStatus::complete;
}
} // namespace hydro
