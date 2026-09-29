//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dual_energy_cancellation.cpp
//! \brief Exact high-Mach entropy-wave test for conservative-to-primitive cancellation.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

namespace {

Real cancellation_rho0 = 1.0;
Real cancellation_pressure = 1.0;
Real cancellation_velocity = 1.0e8;
Real cancellation_amplitude = 0.1;
Real cancellation_xmin = 0.0;
Real cancellation_length = 1.0;

KOKKOS_INLINE_FUNCTION
Real CancellationDensity(const Real x, const Real time, const Real rho0,
                         const Real amplitude, const Real velocity,
                         const Real xmin, const Real length) {
  const Real phase = 2.0*M_PI*(x - xmin - velocity*time)/length;
  return rho0*(1.0 + amplitude*sin(phase));
}

void CancellationHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "dual_energy_cancellation requires hydro only" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &hydro = *pmbp->phydro;
  const auto &eos = hydro.peos->eos_data;
  if (!eos.is_gamma_law || !eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "dual_energy_cancellation requires the ideal gamma-law EOS"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  cancellation_rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  cancellation_pressure = pin->GetOrAddReal("problem", "pressure", 1.0);
  cancellation_velocity = pin->GetOrAddReal("problem", "bulk_velocity", 1.0e8);
  cancellation_amplitude = pin->GetOrAddReal("problem", "density_amplitude", 0.1);
  cancellation_xmin = pmy_mesh_->mesh_size.x1min;
  cancellation_length = pmy_mesh_->mesh_size.x1max - cancellation_xmin;

  if (!(cancellation_rho0 > 0.0) || !(cancellation_pressure > 0.0) ||
      !(cancellation_velocity > 0.0) || cancellation_amplitude < 0.0 ||
      cancellation_amplitude >= 1.0 || !(cancellation_length > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "invalid dual_energy_cancellation problem parameters" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = hydro.u0;
  auto w0 = hydro.w0;
  const int dual_idx = hydro.dual_energy_idx;
  const bool dual_enabled = hydro.use_dual_energy;
  const Real gm1 = eos.gamma - 1.0;
  const Real eint = cancellation_pressure/gm1;
  const Real rho0 = cancellation_rho0;
  const Real pressure = cancellation_pressure;
  const Real velocity = cancellation_velocity;
  const Real amplitude = cancellation_amplitude;
  const Real xmin = cancellation_xmin;
  const Real length = cancellation_length;

  par_for("dual_energy_cancellation_init", DevExeSpace(), 0,
          pmbp->nmb_thispack - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real rho = CancellationDensity(x, 0.0, rho0, amplitude, velocity,
                                         xmin, length);
    const Real momentum = rho*velocity;
    const Real total_energy = eint + 0.5*rho*velocity*velocity;

    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = momentum;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    u0(m, IEN, k, j, i) = total_energy;

    w0(m, IDN, k, j, i) = rho;
    w0(m, IVX, k, j, i) = velocity;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = eint;

    // The ordinary initialization path would recover this field by subtracting
    // kinetic energy from total energy, defeating the purpose of this test. Seed it
    // from the analytic internal energy instead.
    if (dual_enabled) {
      u0(m, dual_idx, k, j, i) = eint;
      w0(m, dual_idx, k, j, i) = eint;
    }
  });

  if (dual_enabled) {
    hydro.dual_energy_needs_init = false;
  }
  user_hist_func = CancellationHistory;

  if (global_variable::my_rank == 0) {
    const Real sound_speed = std::sqrt(eos.gamma*cancellation_pressure/cancellation_rho0);
    const Real mach = cancellation_velocity/sound_speed;
    const Real eint_to_total = eint/(eint + 0.5*cancellation_rho0*
                                      cancellation_velocity*cancellation_velocity);
    std::cout << "Dual-energy cancellation test: U=" << cancellation_velocity
              << " Mach=" << mach
              << " e_int/E=" << eint_to_total
              << " dual=" << (dual_enabled ? "on" : "off") << std::endl;
  }
}

namespace {

void CancellationHistory(HistoryData *pdata, Mesh *pm) {
  auto *hydro = pm->pmb_pack->phydro;
  const auto &eos = hydro->peos->eos_data;
  const auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = pm->pmb_pack->nmb_thispack*nkji;
  const int dual_idx = hydro->dual_energy_idx;
  const bool dual_enabled = hydro->use_dual_energy;
  const Real gm1 = eos.gamma - 1.0;
  const Real pressure_ref = cancellation_pressure;
  auto u0 = hydro->u0;
  auto w0 = hydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;

  pdata->nhist = 10;
  pdata->label[0] = "mass";
  pdata->label[1] = "momentum";
  pdata->label[2] = "total_E";
  pdata->label[3] = "kinetic_E";
  pdata->label[4] = "p_c2p";
  pdata->label[5] = "p_cons";
  pdata->label[6] = "p_aux";
  pdata->label[7] = "abs_pc2p";
  pdata->label[8] = "abs_pcons";
  pdata->label[9] = "volume";

  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "dual_energy_cancellation_history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k = (idx - m*nkji)/nji + ks;
        const int j = (idx - m*nkji - (k-ks)*nji)/nx1 + js;
        const int i = idx - m*nkji - (k-ks)*nji - (j-js)*nx1 + is;
        const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        const Real rho = u0(m, IDN, k, j, i);
        const Real invrho = 1.0/rho;
        const Real kinetic = 0.5*invrho*
            (SQR(u0(m, IM1, k, j, i)) + SQR(u0(m, IM2, k, j, i)) +
             SQR(u0(m, IM3, k, j, i)));
        const Real total_energy = u0(m, IEN, k, j, i);
        const Real pressure_cons = gm1*(total_energy - kinetic);
        const Real pressure_c2p = gm1*w0(m, IEN, k, j, i);
        Real pressure_aux = 0.0;
        if (dual_enabled) {
          pressure_aux = gm1*u0(m, dual_idx, k, j, i);
        }

        sum.the_array[0] += vol*rho;
        sum.the_array[1] += vol*u0(m, IM1, k, j, i);
        sum.the_array[2] += vol*total_energy;
        sum.the_array[3] += vol*kinetic;
        sum.the_array[4] += vol*pressure_c2p;
        sum.the_array[5] += vol*pressure_cons;
        sum.the_array[6] += vol*pressure_aux;
        sum.the_array[7] += vol*fabs(pressure_c2p - pressure_ref);
        sum.the_array[8] += vol*fabs(pressure_cons - pressure_ref);
        sum.the_array[9] += vol;
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sums.the_array[n];
  }
}

}  // namespace
