//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tabulated_eos_homologous.cpp
//! \brief Homologous-expansion regression for the tabulated-EOS dual-energy update.

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

Real expansion_rate_initial = 0.1;
Real density_initial_cgs = 1.0e-8;
Real temperature_initial_kelvin = 2.0e4;

KOKKOS_INLINE_FUNCTION
Real ExpansionRate(const Real time, const Real rate_initial) {
  return rate_initial/(1.0 + rate_initial*time);
}

void HomologousBoundary(Mesh *pm);
void HomologousHistory(HistoryData *pdata, Mesh *pm);

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_bcs_func = HomologousBoundary;
  user_hist_func = HomologousHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "tabulated_eos_homologous requires non-relativistic hydro only"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &hydro = *pmbp->phydro;
  const auto &eos = hydro.peos->eos_data;
  if (!eos.UsesTabulatedLTE() || !hydro.use_dual_energy) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "tabulated_eos_homologous requires a tabulated LTE EOS and dual energy"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "tabulated_eos_homologous requires a three-dimensional mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  expansion_rate_initial = pin->GetOrAddReal("problem", "expansion_rate", 0.1);
  density_initial_cgs = pin->GetOrAddReal("problem", "density_cgs", 1.0e-8);
  temperature_initial_kelvin =
      pin->GetOrAddReal("problem", "temperature_kelvin", 2.0e4);
  if (!(expansion_rate_initial > 0.0) || !(density_initial_cgs > 0.0) ||
      !(temperature_initial_kelvin > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "homologous-expansion parameters must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real density_initial = density_initial_cgs/eos.density_unit_cgs;
  const Real temperature_initial =
      temperature_initial_kelvin/eos.temp_unit_cgs;

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
  const int dual_index = hydro.dual_energy_idx;
  const Real rate_initial = expansion_rate_initial;
  const auto eos_device = eos;

  par_for("tabulated_eos_homologous_init", DevExeSpace(), 0,
          pmbp->nmb_thispack - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real velocity_x = rate_initial*x;
    const Real velocity_y = rate_initial*y;
    const Real velocity_z = rate_initial*z;
    const Real internal_energy_initial = density_initial*
        eos_device.SpecificEintFromRhoT(density_initial, temperature_initial);
    const Real kinetic_energy = 0.5*density_initial*
        (SQR(velocity_x) + SQR(velocity_y) + SQR(velocity_z));

    u0(m, IDN, k, j, i) = density_initial;
    u0(m, IM1, k, j, i) = density_initial*velocity_x;
    u0(m, IM2, k, j, i) = density_initial*velocity_y;
    u0(m, IM3, k, j, i) = density_initial*velocity_z;
    u0(m, IEN, k, j, i) = internal_energy_initial + kinetic_energy;
    u0(m, dual_index, k, j, i) = internal_energy_initial;

    w0(m, IDN, k, j, i) = density_initial;
    w0(m, IVX, k, j, i) = velocity_x;
    w0(m, IVY, k, j, i) = velocity_y;
    w0(m, IVZ, k, j, i) = velocity_z;
    w0(m, IEN, k, j, i) = internal_energy_initial;
    w0(m, dual_index, k, j, i) = internal_energy_initial;
  });

  hydro.dual_energy_needs_init = false;

  if (global_variable::my_rank == 0) {
    std::cout << "Tabulated-EOS homologous expansion: rho0="
              << density_initial_cgs << " g cm^-3 T0="
              << temperature_initial_kelvin << " K H0="
              << expansion_rate_initial << " code_time^-1" << std::endl;
  }
}

namespace {

void HomologousBoundary(Mesh *pm) {
  auto *hydro = pm->pmb_pack->phydro;
  const auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int gis = is-ng;
  const int gie = ie+ng;
  const int gjs = js-ng;
  const int gje = je+ng;
  const int gks = ks-ng;
  const int gke = ke+ng;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &boundary_flags = pm->pmb_pack->pmb->mb_bcs;
  auto u0 = hydro->u0;
  auto w0 = hydro->w0;
  const int dual_index = hydro->dual_energy_idx;
  const Real boundary_time = problem_runtime::HydroStageTimeOr(pm->time);
  const Real expansion_rate = ExpansionRate(boundary_time, expansion_rate_initial);

  par_for("tabulated_eos_homologous_boundary", DevExeSpace(), 0,
          pm->pmb_pack->nmb_thispack - 1, gks, gke, gjs, gje, gis, gie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const bool outside_x1 =
        (i < is && boundary_flags.d_view(m,
            BoundaryFace::inner_x1) == BoundaryFlag::user) ||
        (i > ie && boundary_flags.d_view(m,
            BoundaryFace::outer_x1) == BoundaryFlag::user);
    const bool outside_x2 =
        (j < js && boundary_flags.d_view(m,
            BoundaryFace::inner_x2) == BoundaryFlag::user) ||
        (j > je && boundary_flags.d_view(m,
            BoundaryFace::outer_x2) == BoundaryFlag::user);
    const bool outside_x3 =
        (k < ks && boundary_flags.d_view(m,
            BoundaryFace::inner_x3) == BoundaryFlag::user) ||
        (k > ke && boundary_flags.d_view(m,
            BoundaryFace::outer_x3) == BoundaryFlag::user);
    if (!(outside_x1 || outside_x2 || outside_x3)) return;

    const int source_i = (i < is) ? is : ((i > ie) ? ie : i);
    const int source_j = (j < js) ? js : ((j > je) ? je : j);
    const int source_k = (k < ks) ? ks : ((k > ke) ? ke : k);
    const Real density = u0(m, IDN, source_k, source_j, source_i);
    const Real internal_energy = u0(m, dual_index, source_k, source_j, source_i);
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real velocity_x = expansion_rate*x;
    const Real velocity_y = expansion_rate*y;
    const Real velocity_z = expansion_rate*z;
    const Real kinetic_energy = 0.5*density*
        (SQR(velocity_x) + SQR(velocity_y) + SQR(velocity_z));

    u0(m, IDN, k, j, i) = density;
    u0(m, IM1, k, j, i) = density*velocity_x;
    u0(m, IM2, k, j, i) = density*velocity_y;
    u0(m, IM3, k, j, i) = density*velocity_z;
    u0(m, IEN, k, j, i) = internal_energy + kinetic_energy;
    u0(m, dual_index, k, j, i) = internal_energy;

    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = velocity_x;
    w0(m, IVY, k, j, i) = velocity_y;
    w0(m, IVZ, k, j, i) = velocity_z;
    w0(m, IEN, k, j, i) = internal_energy;
    w0(m, dual_index, k, j, i) = internal_energy;
  });
}

void HomologousHistory(HistoryData *pdata, Mesh *pm) {
  auto *hydro = pm->pmb_pack->phydro;
  const auto &eos = hydro->peos->eos_data;
  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int cells_per_block = nx3*nx2*nx1;
  const int cells_per_plane = nx2*nx1;
  const int total_cells = pm->pmb_pack->nmb_thispack*cells_per_block;
  const int dual_index = hydro->dual_energy_idx;
  auto u0 = hydro->u0;
  auto w0 = hydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;

  pdata->nhist = 8;
  pdata->label[0] = "rho_vol";
  pdata->label[1] = "uaux_vol";
  pdata->label[2] = "temp_vol";
  pdata->label[3] = "pressure_vol";
  pdata->label[4] = "uth_vol";
  pdata->label[5] = "rho2_vol";
  pdata->label[6] = "uaux2_vol";
  pdata->label[7] = "volume";

  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "tabulated_eos_homologous_history",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int index, array_sum::GlobalSum &sum) {
        const int m = index/cells_per_block;
        const int k = (index - m*cells_per_block)/cells_per_plane + ks;
        const int j = (index - m*cells_per_block - (k-ks)*cells_per_plane)/nx1 + js;
        const int i = index - m*cells_per_block - (k-ks)*cells_per_plane -
                      (j-js)*nx1 + is;
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
        const Real density = u0(m, IDN, k, j, i);
        const Real internal_energy_aux = u0(m, dual_index, k, j, i);
        const auto thermo = eos.EvalThermoStateFromRhoEint(density, internal_energy_aux);

        sum.the_array[0] += volume*density;
        sum.the_array[1] += volume*internal_energy_aux;
        sum.the_array[2] += volume*thermo.temperature*eos.temp_unit_cgs;
        sum.the_array[3] += volume*thermo.pressure;
        sum.the_array[4] += volume*w0(m, IEN, k, j, i);
        sum.the_array[5] += volume*density*density;
        sum.the_array[6] += volume*internal_energy_aux*internal_energy_aux;
        sum.the_array[7] += volume;
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sums.the_array[n];
  }
}

}  // namespace
