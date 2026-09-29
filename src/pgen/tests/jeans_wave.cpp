//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file jeans_wave.cpp
//  \brief Problem generator for a Jeans wave with self-gravity

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

struct JeansWaveParams {
  Real rho0 = 1.0;
  Real amp = 1.0e-6;
  Real kx = 0.0;
  Real ky = 0.0;
  Real kz = 0.0;
  Real k_wave = 0.0;
  Real omega = 0.0;
  Real omega2 = 0.0;
  Real x0 = 0.0;
  Real y0 = 0.0;
  Real z0 = 0.0;
  Real v0 = 0.0;
  Real sound_speed = 1.0;
  Real four_pi_G = 1.0;
  Real njeans_threshold = 16.0;
  Real njeans_derefine = 2.5;
  bool initialized = false;
};

struct JeansProjection {
  Real sin_amp = 0.0;
  Real cos_amp = 0.0;
};

JeansWaveParams jw_params;

JeansProjection ProjectJeansDensity(Mesh *pm, Real time) {
  JeansProjection projection;
  if (!jw_params.initialized || pm == nullptr || pm->pmb_pack == nullptr) {
    return projection;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> w0;
  if (pmbp->phydro != nullptr) {
    w0 = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    w0 = pmbp->pmhd->w0;
  } else {
    return projection;
  }

  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;

  const Real rho0 = jw_params.rho0;
  const Real kx = jw_params.kx;
  const Real ky = jw_params.ky;
  const Real kz = jw_params.kz;
  const Real phase_shift = jw_params.k_wave*jw_params.v0*time;
  const Real x0 = jw_params.x0;
  const Real y0 = jw_params.y0;
  const Real z0 = jw_params.z0;

  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "JeansWaveProjection", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &mb_sum) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = idx - m*nkji - k*nji - j*nx1;
        k += ks;
        j += js;
        i += is;

        const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);
        const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min,
                                   size.d_view(m).x3max);
        const Real phase = kx*(x-x0) + ky*(y-y0) + kz*(z-z0) - phase_shift;
        const Real sn = std::sin(phase);
        const Real cs = std::cos(phase);
        const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        const Real drho = w0(m, IDN, k, j, i) - rho0;

        array_sum::GlobalSum vals;
        vals.the_array[0] = vol*drho*sn;
        vals.the_array[1] = vol*drho*cs;
        vals.the_array[2] = vol*sn*sn;
        vals.the_array[3] = vol*cs*cs;
        for (int n = 4; n < NHISTORY_VARIABLES; ++n) {
          vals.the_array[n] = 0.0;
        }
        mb_sum += vals;
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));

  Real global_sums[4] = {
      sums.the_array[0], sums.the_array[1], sums.the_array[2], sums.the_array[3]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, global_sums, 4, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  if (global_sums[2] > 0.0) projection.sin_amp = global_sums[0]/global_sums[2];
  if (global_sums[3] > 0.0) projection.cos_amp = global_sums[1]/global_sums[3];
  return projection;
}

}  // namespace

void JeansWaveHistory(HistoryData *pdata, Mesh *pm);
void JeansWaveRefinement(MeshBlockPack *pmbp);
void JeansWaveErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SelfGravity()
//  \brief Initialize a linear Jeans mode in Hydro or MHD.

void ProblemGenerator::SelfGravity(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool use_mhd = (pmbp->pmhd != nullptr);
  if (pmbp->phydro == nullptr && !use_mhd) {
    std::cout << "### FATAL ERROR in ProblemGenerator::SelfGravity" << std::endl
              << "The Jeans wave requires Hydro or MHD." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string fluid_block = use_mhd ? "mhd" : "hydro";

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  const Real n_jeans = pin->GetOrAddReal("problem", "n_jeans", -1.0);
  const Real v0 = pin->GetOrAddReal("problem", "v0", 0.0);
  const int n_waves = pin->GetOrAddInteger("problem", "n_waves", 1);
  const int mode_x1 = pin->GetOrAddInteger("problem", "mode_x1", n_waves);
  const int mode_x2 = pin->GetOrAddInteger("problem", "mode_x2", 0);
  const int mode_x3 = pin->GetOrAddInteger("problem", "mode_x3", 0);

  const std::string eos_type = pin->GetString(fluid_block, "eos");
  const bool is_isothermal = (eos_type == "isothermal");
  Real gamma = 0.0;
  Real gm1 = 0.0;
  Real p0 = 0.0;
  Real cs = 0.0;
  if (is_isothermal) {
    cs = pin->GetReal(fluid_block, "iso_sound_speed");
  } else {
    gamma = pin->GetOrAddReal(fluid_block, "gamma", 5.0/3.0);
    gm1 = gamma - 1.0;
    p0 = pin->GetOrAddReal("problem", "p0", 1.0);
    cs = std::sqrt(gamma*p0/rho0);
  }

  const auto &msize = pmy_mesh_->mesh_size;
  const Real domain_x1min = msize.x1min;
  const Real domain_x2min = msize.x2min;
  const Real domain_x3min = msize.x3min;
  const Real lx1 = msize.x1max - msize.x1min;
  const Real lx2 = msize.x2max - msize.x2min;
  const Real lx3 = msize.x3max - msize.x3min;
  const Real kx = 2.0*M_PI*static_cast<Real>(mode_x1)/lx1;
  const Real ky = 2.0*M_PI*static_cast<Real>(mode_x2)/lx2;
  const Real kz = 2.0*M_PI*static_cast<Real>(mode_x3)/lx3;
  const Real k_wave = std::sqrt(SQR(kx) + SQR(ky) + SQR(kz));
  if (k_wave <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::SelfGravity" << std::endl
              << "At least one Jeans-wave mode number must be nonzero." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (n_jeans > 0.0) {
    four_pi_G = SQR(n_jeans*k_wave*cs)/rho0;
    pin->SetReal("gravity", "four_pi_G", four_pi_G);
    if (pmbp->pgrav != nullptr) {
      pmbp->pgrav->four_pi_G = four_pi_G;
      if (pmbp->pgrav->pmgd != nullptr) {
        pmbp->pgrav->pmgd->SetFourPiG(four_pi_G);
      }
    }
  }

  const Real k_jeans = (four_pi_G > 0.0) ? std::sqrt(four_pi_G*rho0)/cs : 0.0;
  const Real omega2 = SQR(k_wave*cs) - four_pi_G*rho0;
  const Real omega = std::sqrt(std::abs(omega2));

  jw_params.rho0 = rho0;
  jw_params.amp = amp;
  jw_params.kx = kx;
  jw_params.ky = ky;
  jw_params.kz = kz;
  jw_params.k_wave = k_wave;
  jw_params.omega = omega;
  jw_params.omega2 = omega2;
  jw_params.x0 = domain_x1min;
  jw_params.y0 = domain_x2min;
  jw_params.z0 = domain_x3min;
  jw_params.v0 = v0;
  jw_params.sound_speed = cs;
  jw_params.four_pi_G = four_pi_G;
  jw_params.njeans_threshold =
      pin->GetOrAddReal("problem", "njeans_amr", 16.0);
  jw_params.njeans_derefine =
      pin->GetOrAddReal("problem", "njeans_derefine", 2.5);
  jw_params.initialized = true;

  user_hist = true;
  user_hist_func = JeansWaveHistory;
  user_ref_func = JeansWaveRefinement;
  pgen_final_func = JeansWaveErrors;
  pin->SetBoolean("problem", "user_hist", true);

  if (global_variable::my_rank == 0) {
    std::cout << "Jeans wave test parameters:" << std::endl;
    std::cout << "  rho0 = " << rho0 << ", cs = " << cs;
    if (!is_isothermal) std::cout << ", p0 = " << p0;
    std::cout << std::endl;
    std::cout << "  mode = (" << mode_x1 << ", " << mode_x2 << ", " << mode_x3
              << ")" << std::endl;
    std::cout << "  k = (" << kx << ", " << ky << ", " << kz << "), |k| = "
              << k_wave << std::endl;
    if (k_jeans > 0.0) {
      std::cout << "  k_Jeans = " << k_jeans << ", k/k_J = "
                << k_wave/k_jeans << std::endl;
    }
    if (n_jeans > 0.0) {
      std::cout << "  n_Jeans = " << n_jeans << " (lambda/lambda_Jeans)"
                << std::endl;
    }
    std::cout << "  four_pi_G = " << four_pi_G << std::endl;
    if (v0 != 0.0) std::cout << "  background velocity = " << v0 << std::endl;
    if (omega2 < 0.0) {
      std::cout << "  Gravitationally unstable! Growth rate = " << omega << std::endl;
    } else {
      std::cout << "  Stable oscillation. Frequency = " << omega << std::endl;
    }
  }

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const Real khat_x = kx/k_wave;
  const Real khat_y = ky/k_wave;
  const Real khat_z = kz/k_wave;
  const Real b0_val = use_mhd ? pin->GetOrAddReal("problem", "b0", 0.0) : 0.0;

  DvceArray5D<Real> u0 = use_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  par_for("jeans_wave_init", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real phase = kx*(x-domain_x1min) + ky*(y-domain_x2min) +
                       kz*(z-domain_x3min);
    const Real sinkx = std::sin(phase);
    const Real coskx = std::cos(phase);
    const Real dens = rho0*(1.0 + amp*sinkx);
    const Real mode_momentum =
        (omega2 < 0.0) ? rho0*(omega/k_wave)*amp*coskx : 0.0;

    u0(m, IDN, k, j, i) = dens;
    u0(m, IM1, k, j, i) = (mode_momentum + dens*v0)*khat_x;
    u0(m, IM2, k, j, i) = (mode_momentum + dens*v0)*khat_y;
    u0(m, IM3, k, j, i) = (mode_momentum + dens*v0)*khat_z;
    if (!is_isothermal) {
      u0(m, IEN, k, j, i) = p0/gm1*(1.0 + gamma*amp*sinkx);
      u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM1, k, j, i))/dens;
      u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM2, k, j, i))/dens;
      u0(m, IEN, k, j, i) += 0.5*SQR(u0(m, IM3, k, j, i))/dens;
      if (use_mhd) u0(m, IEN, k, j, i) += 0.5*SQR(b0_val);
    }
  });

  if (use_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("jeans_wave_bfield", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m, k, j, i) = b0_val;
      b0.x2f(m, k, j, i) = 0.0;
      b0.x3f(m, k, j, i) = 0.0;
      if (i == ie) b0.x1f(m, k, j, i+1) = b0_val;
      if (j == je) b0.x2f(m, k, j+1, i) = 0.0;
      if (k == ke) b0.x3f(m, k+1, j, i) = 0.0;
      bcc0(m, IBX, k, j, i) = b0_val;
      bcc0(m, IBY, k, j, i) = 0.0;
      bcc0(m, IBZ, k, j, i) = 0.0;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveRefinement()
//  \brief Refine when the local Jeans length is under-resolved.

void JeansWaveRefinement(MeshBlockPack *pmbp) {
  if (pmbp == nullptr || pmbp->pmesh == nullptr || pmbp->pmesh->pmr == nullptr) return;
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  DvceArray5D<Real> u0 =
      (pmbp->phydro != nullptr) ? pmbp->phydro->u0 : pmbp->pmhd->u0;
  auto &size = pmbp->pmb->mb_size;
  const Real cs = jw_params.sound_speed;
  const Real four_pi_G = jw_params.four_pi_G;
  const Real threshold = jw_params.njeans_threshold;
  const Real derefine = jw_params.njeans_derefine;

  par_for_outer("JeansWaveAMR", DevExeSpace(), 0, 0, 0, nmb-1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [&](const int idx, Real &rhomax) {
          int k = idx/nji;
          int j = (idx-k*nji)/nx1;
          int i = idx-k*nji-j*nx1;
          k += ks;
          j += js;
          i += is;
          rhomax = Kokkos::fmax(rhomax, u0(m, IDN, k, j, i));
        }, Kokkos::Max<Real>(team_rhomax));

    int flag = 0;
    if (four_pi_G > 0.0 && team_rhomax > 0.0) {
      const Real dx = Kokkos::fmax(
          size.d_view(m).dx1,
          Kokkos::fmax(size.d_view(m).dx2, size.d_view(m).dx3));
      const Real cells_per_jeans =
          2.0*M_PI*cs/(dx*Kokkos::sqrt(four_pi_G*team_rhomax));
      if (cells_per_jeans < threshold) {
        flag = 1;
      } else if (cells_per_jeans > threshold*derefine) {
        flag = -1;
      }
    }
    refine_flag.d_view(m+mbs) = flag;
  });
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveHistory()
//  \brief Track the numerical and analytical Jeans-mode amplitudes.

void JeansWaveHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 6;
  pdata->label[0] = "amp_num";
  pdata->label[1] = "amp_exact";
  pdata->label[2] = "rel_err";
  pdata->label[3] = "phase";
  pdata->label[4] = "amp_abs";
  pdata->label[5] = "cos_amp";
  for (int n = 0; n < pdata->nhist; ++n) pdata->hdata[n] = 0.0;

  const JeansProjection projection = ProjectJeansDensity(pm, pm->time);
  Real amp_exact = jw_params.rho0*jw_params.amp;
  if (jw_params.omega2 < 0.0) {
    amp_exact *= std::exp(jw_params.omega*pm->time);
  } else {
    amp_exact *= std::cos(jw_params.omega*pm->time);
  }
  const Real abs_exact = std::abs(amp_exact);
  const Real rel_err = (abs_exact > std::numeric_limits<Real>::epsilon())
      ? std::abs(projection.sin_amp-amp_exact)/abs_exact
      : std::abs(projection.sin_amp-amp_exact);

  if (global_variable::my_rank == 0) {
    pdata->hdata[0] = projection.sin_amp;
    pdata->hdata[1] = amp_exact;
    pdata->hdata[2] = rel_err;
    pdata->hdata[3] = std::atan2(projection.cos_amp, projection.sin_amp);
    pdata->hdata[4] = std::sqrt(SQR(projection.sin_amp) + SQR(projection.cos_amp));
    pdata->hdata[5] = projection.cos_amp;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void JeansWaveErrors()
//  \brief Report the measured Jeans growth rate or oscillation frequency.

void JeansWaveErrors(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  if (!jw_params.initialized || pm == nullptr ||
      pm->time <= std::numeric_limits<Real>::epsilon()) {
    return;
  }

  const JeansProjection projection = ProjectJeansDensity(pm, pm->time);
  const Real normalized_amp = projection.sin_amp/jw_params.rho0;
  Real omega_measured = 0.0;
  if (jw_params.omega2 < 0.0) {
    omega_measured =
        std::log(std::abs(normalized_amp)/jw_params.amp)/pm->time;
  } else {
    const Real ratio = std::max(static_cast<Real>(-1.0),
        std::min(static_cast<Real>(1.0), normalized_amp/jw_params.amp));
    omega_measured = std::acos(ratio)/pm->time;
  }

  if (global_variable::my_rank == 0) {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1);
    std::cout << "=====================================================" << std::endl;
    std::cout << "Jeans wave mode amplitude  : " << normalized_amp << std::endl;
    std::cout << "Jeans wave growth (A/amp)  : "
              << normalized_amp/jw_params.amp << std::endl;
    std::cout << "Jeans wave omega measured  : " << omega_measured << std::endl;
    std::cout << "Jeans wave omega analytical: " << jw_params.omega << std::endl;
    std::cout << "=====================================================" << std::endl;
  }
}
