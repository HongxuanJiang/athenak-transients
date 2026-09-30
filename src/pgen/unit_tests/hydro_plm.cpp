//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_plm.cpp
//! \brief Regression tests for thermodynamic PLM reconstruction at steep density jumps.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "reconstruct/specific_energy_recon.hpp"

namespace {

// Thermodynamic invariants, not a second implementation of the slope formula:
// bounded specific energy across an atmosphere, exact constant/linear specific
// energy, extrema preservation, a vanishing field, and both thermal channels. Reflected
// fixtures and m != mb exercise all directional and chunk-local index paths.
bool CheckFaceStates(const EOS_Data &eos, const int dual) {
  constexpr int nv = 8, m = 2, mb = 0, face_index = 3;
  DvceArray5D<Real> w("hydro_plm_test_w", 3, nv, 6, 6, 6);
  DvceArray5D<Real> wl("hydro_plm_test_wl", 1, nv, 6, 6, 6);
  DvceArray5D<Real> wr("hydro_plm_test_wr", 1, nv, 6, 6, 6);
  auto host = Kokkos::create_mirror_view(w);
  auto hl = Kokkos::create_mirror_view(wl);
  auto hr = Kokkos::create_mirror_view(wr);
  int checks = 0, failures = 0, old_overshoots = 0;
  Real max_error = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    for (int reverse = 0; reverse < 2; ++reverse) {
      for (int fixture = 0; fixture < 5; ++fixture) {
        Real rho[4] = {1.0e-2, 1.0e-6, 1.0e-12, 1.4e-18};
        Real eps[4] = {0.005, 0.1, 0.1, 0.1};
        if (fixture == 1) {
          for (Real &v : eps) v = 0.05;
        } else if (fixture == 2) {
          for (int a = 0; a < 4; ++a) eps[a] = 0.02*(a+1);
        } else if (fixture == 3) {
          eps[0] = 0.02; eps[1] = 0.08; eps[2] = 0.02; eps[3] = 0.04;
        } else if (fixture == 4) {
          for (Real &v : eps) v = 0.0;
        }
        if (reverse) {
          std::reverse(rho, rho+4);
          std::reverse(eps, eps+4);
        }
        Kokkos::deep_copy(host, -123.0);
        for (int k = 0; k < 6; ++k) {
          for (int j = 0; j < 6; ++j) {
            for (int i = 0; i < 6; ++i) {
              const int q = std::max(0, std::min(3, (axis==0 ? i : axis==1 ? j : k)-1));
              host(m, IDN, k, j, i) = rho[q];
              host(m, IEN, k, j, i) = rho[q]*eps[q];
              host(m, 7, k, j, i) = 2.0*rho[q]*eps[q];
            }
          }
        }
        Kokkos::deep_copy(w, host);
        Kokkos::deep_copy(hl, -321.0);
        Kokkos::deep_copy(hr, -321.0);
        Real ld, rd, unused;
        PLM(rho[0], rho[1], rho[2], ld, unused);
        PLM(rho[1], rho[2], rho[3], unused, rd);
        ld = std::max(ld, eos.dfloor);
        rd = std::max(rd, eos.dfloor);
        hl(mb, IDN, face_index, face_index, face_index) = ld;
        hr(mb, IDN, face_index, face_index, face_index) = rd;
        Kokkos::deep_copy(wl, hl);
        Kokkos::deep_copy(wr, hr);
        par_for("hydro_specific_energy_reconstruction_test", DevExeSpace(), 0, 0,
        KOKKOS_LAMBDA(int) {
          if (axis == 0) {
            hydro_reconstruction::SpecificEnergyFaceAt<ReconstructionMethod::plm, 0>(
                eos, dual, m, mb, face_index, face_index, face_index, w, wl, wr);
          }
          if (axis == 1) {
            hydro_reconstruction::SpecificEnergyFaceAt<ReconstructionMethod::plm, 1>(
                eos, dual, m, mb, face_index, face_index, face_index, w, wl, wr);
          }
          if (axis == 2) {
            hydro_reconstruction::SpecificEnergyFaceAt<ReconstructionMethod::plm, 2>(
                eos, dual, m, mb, face_index, face_index, face_index, w, wl, wr);
          }
        });
        Kokkos::deep_copy(hl, wl);
        Kokkos::deep_copy(hr, wr);
        for (int side = 0; side < 2; ++side) {
          for (int channel = 0; channel < (dual >= 0 ? 2 : 1); ++channel) {
            const auto h = side == 0 ? hl : hr;
            const Real density = side == 0 ? ld : rd;
            const Real scale = channel == 0 ? 1.0 : 2.0;
            const int n = channel == 0 ? IEN : dual;
            const Real got = h(mb, n, face_index, face_index, face_index)/density;
            Real expected = got;
            Real lower = scale*std::min({eps[side], eps[side+1], eps[side+2]});
            Real upper = scale*std::max({eps[side], eps[side+1], eps[side+2]});
            if (fixture == 1) expected = scale*0.05;
            if (fixture == 2) expected = scale*0.05;
            if (fixture == 3) expected = scale*eps[side+1];
            // A vanishing field reconstructs to exactly zero: the thermal floor is
            // applied by the face sanitiser and the dual-energy flux, not here.
            if (fixture == 4) expected = 0.0;
            const Real error =
                std::abs(got-expected)/std::max(std::abs(expected), 1.0e-30);
            max_error = std::max(max_error, error);
            const bool ok = std::isfinite(got) && (got > 0.0 || fixture == 4) &&
                error < 1.0e-12 &&
                got >= lower*(1.0-1.0e-12) && got <= upper*(1.0+1.0e-12);
            ++checks;
            if (!ok) {
              ++failures;
              std::cout << "HYDRO PLM ERROR axis=" << axis << " reverse=" << reverse
                        << " fixture=" << fixture << " side=" << side << " channel="
                        << channel << " got=" << got << " expected=" << expected
                        << " bounds=" << lower << ',' << upper << '\n';
            }
            if (fixture == 0) {
              Real ep, em;
              PLM(rho[side]*eps[side], rho[side+1]*eps[side+1],
                  rho[side+2]*eps[side+2], ep, em);
              const Real old_eps = scale*(side==0 ? ep : em)/density;
              if (old_eps > upper*(1.0+1.0e-6)) ++old_overshoots;
            }
          }
        }
        // Thermal reconstruction must not write density, velocities, or tracers.
        constexpr int untouched[] = {IDN, IVX, IVY, IVZ, 5, 6, 7};
        for (int n : untouched) {
          if (n == dual) continue;
          ++checks;
          if (hl(mb,n,3,3,3) != (n==IDN ? ld : -321.0) ||
              hr(mb,n,3,3,3) != (n==IDN ? rd : -321.0)) ++failures;
        }
      }
    }
  }
  if (old_overshoots == 0) ++failures;  // The regression must detect the old defect.
  std::cout << "HYDRO PLM TEST pass=" << (failures==0) << " checks=" << checks
            << " failures=" << failures << " old_overshoots=" << old_overshoots
            << " max_relative_error=" << max_error << '\n';
  return failures == 0;
}

// Exercise the real flux path: at this face PLM gives the same specific energy
// on both sides. With zero velocity, the HLLE energy/mass flux ratio must equal
// that specific energy. Different energies in each block expose m/mb confusion.
bool CheckFluxes(MeshBlockPack *pack) {
  auto *hyd = pack->phydro;
  const auto &ind = pack->pmesh->mb_indcs;
  auto host = Kokkos::create_mirror_view(hyd->w0);
  const int nmb = pack->nmb_thispack;
  int checks = 0, failures = 0;
  for (int lat = 0; lat < 2; ++lat) {
    pack->lat_active_mask_enabled = (lat != 0);
    pack->lat_nactive_thispack = 1;
    pack->lat_active_indices.h_view(0) = nmb - 1;
    pack->lat_active_indices.modify<HostMemSpace>();
    pack->lat_active_indices.sync<DevExeSpace>();
    for (int axis = 0; axis < 3; ++axis) {
      Kokkos::deep_copy(host, 0.0);
      const int start = axis == 0 ? ind.is : axis == 1 ? ind.js : ind.ks;
      for (int m = 0; m < nmb; ++m) {
        for (int k = 0; k < host.extent_int(2); ++k) {
          for (int j = 0; j < host.extent_int(3); ++j) {
            for (int i = 0; i < host.extent_int(4); ++i) {
              const int q = (axis == 0 ? i : axis == 1 ? j : k) - start;
              const Real rho = q <= 0 ? 1.0e-2 : q == 1 ? 1.0e-6 :
                               q == 2 ? 1.0e-12 : 1.4e-18;
              const Real eps = (q <= 0 ? 0.005 : 0.1)*(m+1);
              host(m, IDN, k, j, i) = rho;
              host(m, IEN, k, j, i) = rho*eps;
              if (hyd->use_dual_energy) {
                host(m, hyd->dual_energy_idx, k, j, i) = 2.0*rho*eps;
              }
            }
          }
        }
      }
      Kokkos::deep_copy(hyd->w0, host);
      auto flux = axis == 0 ? hyd->uflx.x1f :
                  axis == 1 ? hyd->uflx.x2f : hyd->uflx.x3f;
      Kokkos::deep_copy(flux, -321.0);
      hyd->CalculateFluxes<Hydro_RSolver::hlle>(nullptr, 1);
      auto hf = Kokkos::create_mirror_view_and_copy(HostMemSpace(), flux);
      // the register is stored on a band of the block: global index minus its origin
      const int i = ind.is + (axis == 0 ? 2 : 0) - hyd->flux_io;
      const int j = ind.js + (axis == 1 ? 2 : 0) - hyd->flux_jo;
      const int k = ind.ks + (axis == 2 ? 2 : 0) - hyd->flux_ko;
      for (int m = 0; m < nmb; ++m) {
        const int channels = hyd->use_dual_energy ? 2 : 1;
        for (int channel = 0; channel < channels; ++channel) {
          const int n = channel == 0 ? IEN : hyd->dual_energy_idx;
          ++checks;
          if (lat && m != nmb - 1) {
            if (hf(m, n, k, j, i) != -321.0) ++failures;
            continue;
          }
          const Real expected = 0.1*(m+1)*(channel+1);
          const Real got = hf(m, n, k, j, i)/hf(m, IDN, k, j, i);
          if (!std::isfinite(got) || std::abs(got/expected - 1.0) > 1.0e-12) {
            ++failures;
            std::cout << "HYDRO PLM FLUX ERROR axis=" << axis << " lat=" << lat
                      << " m=" << m << " channel=" << channel << " got=" << got
                      << " expected=" << expected << '\n';
          }
        }
      }
    }
  }
  pack->lat_active_mask_enabled = false;
  std::cout << "HYDRO PLM FLUX checks=" << checks << " failures=" << failures << '\n';
  return failures == 0;
}

}  // namespace

void ProblemGenerator::HydroPLMUnit(ParameterInput *pin, const bool restart) {
  auto *pack = pmy_mesh_->pmb_pack;
  auto *hyd = pack->phydro;
  if (restart || hyd == nullptr || !hyd->peos->eos_data.UsesTabulatedLTE() ||
      !pmy_mesh_->three_d || pack->nmb_thispack < 2 ||
      hyd->recon_method != ReconstructionMethod::plm || hyd->split_recon_chunk_nmb != 1) {
    std::cerr << "hydro_plm_unit requires a fresh 3D tabulated-EOS PLM run with "
              << "at least two local blocks and split_recon_chunk_nmb=1\n";
    std::exit(EXIT_FAILURE);
  }
  const bool gas = CheckFaceStates(hyd->peos->eos_data, -1);
  const bool dual = CheckFaceStates(hyd->peos->eos_data, 7);
  const bool flux = CheckFluxes(pack);
  std::cout << "hydro_plm_unit: " << (gas && dual && flux ? "ALL PASS" : "FAILURES")
            << std::endl;
  if (!(gas && dual && flux)) std::exit(EXIT_FAILURE);

  // Leave a valid uniform conserved state for normal startup with nlim=0.
  const auto u = hyd->u0;
  const int dual_idx = hyd->dual_energy_idx;
  Kokkos::deep_copy(u, 0.0);
  par_for("hydro_plm_unit_init", DevExeSpace(), 0, pack->nmb_thispack-1,
      0, u.extent_int(2)-1, 0, u.extent_int(3)-1, 0, u.extent_int(4)-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u(m, IDN, k, j, i) = 1.0e-2;
        u(m, IEN, k, j, i) = 5.0e-4;
        if (dual_idx >= 0) u(m, dual_idx, k, j, i) = 5.0e-4;
      });
}
