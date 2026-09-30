//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravity.cpp
//! \brief implementation of functions in class Gravity

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../coordinates/cell_locations.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../pgen/pgen.hpp"
#include "gravity.hpp"
#include "utils/gravity_weight.hpp"
#include "mg_gravity.hpp"
#include "../multigrid/multigrid.hpp"
#include "../mesh/mb_storage.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../srcterms/srcterms.hpp"
#include "../driver/driver.hpp"
#include "../globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace gravity { // NOLINT (build/namespace)
//! constructor, initializes data structures and parameters
//-------------------------------------------------------------------------------------
//! \fn Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Gravity constructor
Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin):
    pmy_pack(pmbp),
    phi("phi",1,1,1,1,1),
    coarse_phi("coarse",1,1,1,1,1),
    four_pi_G(-1.0),
    phi_valid(false),
    self_phi_time_valid(false),
    reuse_phi_after_amr(false),
    pmgd(nullptr),
    pmg(nullptr) {
    four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G",-1.0);
    reuse_phi_after_amr = pin->GetOrAddBoolean("gravity", "reuse_phi_after_amr", false);
    lat_time_centered_work = pin->GetOrAddBoolean("gravity", "lat_time_centered_work",
        false);
    lat_boundary_flux_diagnostics =
        pin->GetOrAddBoolean("gravity", "lat_boundary_flux_diagnostics", false);
    lat_ledger_debug = pin->GetOrAddBoolean("gravity", "lat_ledger_debug", false);
    if (lat_boundary_flux_diagnostics && !lat_time_centered_work) {
      throw std::runtime_error("LAT boundary ledger requires lat_time_centered_work=true");
    }
    if (lat_boundary_flux_diagnostics) {
      const char *names[] = {"boundary_mass_out", "boundary_gas_energy_out",
                            "boundary_self_mass_work_out", "boundary_bh_mass_work_out"};
      for (int n = 0; n < 4; ++n) {
        boundary_flux_total[n] = pin->GetOrAddReal("gravity", names[n], 0.0);
        if (!std::isfinite(boundary_flux_total[n])) {
          throw std::runtime_error("Nonfinite restart boundary ledger");
        }
      }
    }
    if (lat_time_centered_work) {
      centered_work_total = pin->GetOrAddReal("gravity", "lat_centered_work_total", 0.0);
      remap_energy_total = pin->GetOrAddReal("gravity", "lat_remap_energy_total", 0.0);
      floor_energy_total = pin->GetOrAddReal("gravity", "lat_floor_energy_total", 0.0);
      asym_energy_total = pin->GetOrAddReal("gravity", "lat_asym_energy_total", 0.0);
      ledger_ref_self = pin->GetOrAddReal("gravity", "lat_ledger_ref_self", 0.0);
      ledger_ref_bh = pin->GetOrAddReal("gravity", "lat_ledger_ref_bh", 0.0);
      ledger_ref_gas = pin->GetOrAddReal("gravity", "lat_ledger_ref_gas", 0.0);
      ledger_ref_valid = pin->GetOrAddBoolean("gravity", "lat_ledger_ref_valid", false);
      if (!std::isfinite(centered_work_total) || !std::isfinite(remap_energy_total) ||
          !std::isfinite(ledger_ref_self) || !std::isfinite(ledger_ref_bh)) {
        throw std::runtime_error("Nonfinite restart gravity energy ledger");
      }
    }
    // The window-end identity Delta U = sum(Delta rho * phi_avg * dV) needs the
    // gravitational work applied to every cell whose density can change, so the
    // source-term density gate must not exclude any floored cell: rho_grav_min at or
    // below the hydro density floor keeps the gate inactive (every cell sits at or
    // above dfloor after the floors are applied).
    const Real hydro_dfloor = pin->DoesParameterExist("hydro", "dfloor") ?
        pin->GetReal("hydro", "dfloor") : 0.0;
    if (lat_time_centered_work &&
        (!pin->GetOrAddBoolean("time", "lat", false) ||
         pin->GetOrAddReal("gravity", "rho_grav_min", 0.0) > hydro_dfloor ||
         pin->GetOrAddReal("gravity", "mask_radius", -1.0) > 0.0 ||
         pin->DoesBlockExist("sink_particles"))) {
      throw std::runtime_error("LAT time-centered self-gravity work requires time/lat, "
          "gravity/rho_grav_min <= hydro/dfloor, no mask_radius, and no "
          "<sink_particles>");
    }

    if (four_pi_G == 0.0) {
        std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
        exit(EXIT_FAILURE);
    }

    // create multigrid driver/solver
    // The driver allocates multigrid instances for root level and meshblock levels
    pmgd = new MGGravityDriver(pmbp, pin);
    // The window-end potential is solved at every window end, which MGGravityDriver::
    // Solve skips on a cycle that is not a multiple of solve_every (no solve_dt).
    if (lat_time_centered_work && !(pmgd->SolveInterval() > 0.0) &&
        pmgd->SolveEvery() > 1) {
      throw std::runtime_error("gravity/lat_time_centered_work needs gravity/solve_dt > "
          "0 or solve_every = 1: the window-end potential must be solved every window");
    }

    // Enroll CellCenteredBoundaryVariable object
    //gbvar.bvar_index = pmb->pbval->bvars.size();
    //pmb->pbval->bvars.push_back(&gbvar);
    //pmb->pbval->pgbvar = &gbvar;
    int nmb = pmy_pack->nmb_thispack;
    if (MeshBlockStorageReserve() > 0) nmb = MeshBlockStorageCapacity(nmb);
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(phi, nmb, 1, ncells3, ncells2, ncells1);
    if (pmy_pack->pmesh->multilevel) {
        int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
        int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
        int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
        Kokkos::realloc(coarse_phi, nmb, 1, n_ccells3, n_ccells2, n_ccells1);
    }
}

//----------------------------------------------------------------------------------------
//! \fn Gravity::~Gravity()
//! \brief Gravity destructor
Gravity::~Gravity() {
    delete pmgd;
}

bool Gravity::ResizeMeshBlockStorage(int nmb, bool exact, bool allow_shrink) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;

    auto resize5 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3,
        int n4) {
        const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                           view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                           view.extent_int(3) != n3 || view.extent_int(4) != n4);
        if (need) {
            Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0),
                           n1, n2, n3, n4);
        }
        return need;
    };

    bool resized = false;
    resized = resize5(phi, nmb, 1, ncells3, ncells2, ncells1) || resized;
    if (pmy_pack->pmesh->multilevel) {
        resized = resize5(coarse_phi, nmb, 1, n_ccells3, n_ccells2, n_ccells1) || resized;
    }
    return resized;
}

void Gravity::MarkPhiInvalid() {
    phi_valid = false;
    self_phi_time_valid = false;
}

void Gravity::MarkSelfPhiValid() {
    self_phi_time_valid = true;
    self_phi_time = problem_runtime::HydroStageTimeOr(pmy_pack->pmesh->time);
    self_phi_topology = pmy_pack->pmesh->topology_version;
}

bool Gravity::RefreshPhiGhosts(bool require_valid) {
    if ((require_valid && !phi_valid) || pmgd == nullptr) return false;
    return pmgd->RefreshFinestGhosts(phi, pmy_pack->pmesh->mb_indcs.ng);
}

bool Gravity::PotentialMatchesTime(Real time) const {
  const Real tolerance = 2048*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::abs(time));
  return phi_valid && self_phi_time_valid &&
      self_phi_topology == pmy_pack->pmesh->topology_version &&
      std::abs(self_phi_time-time) <= tolerance;
}

void Gravity::BeginLATEnergyWindow(Real dt) {
  if (!lat_time_centered_work) return;
  auto *pm = pmy_pack->pmesh;
  auto *hydro = pmy_pack->phydro;
  if (energy_window_open || energy_window_pending || hydro == nullptr ||
      !hydro->peos->eos_data.use_e ||
      hydro->psrc == nullptr || !hydro->psrc->self_gravity ||
      pmy_pack->lat_active_mask_enabled || !std::isfinite(dt) || !(dt > 0.0) ||
      !PotentialMatchesTime(pm->time)) {
    throw std::runtime_error("LAT gravity energy window requires synchronized energy-evolving hydro");
  }
  // Driver owns the synchronized solve. This method only snapshots its result.
  // History closes before AMR, so it never needs interpolation or migration.
  const auto ind = pm->mb_indcs;
  const int nb = pmy_pack->nmb_thispack;
  if (energy_window_state.extent_int(0) != nb ||
      energy_window_state.extent_int(4) != ind.nx1 ||
      energy_window_state.extent_int(3) != ind.nx2 ||
      energy_window_state.extent_int(2) != ind.nx3) {
    Kokkos::realloc(energy_window_state, nb, 2, ind.nx3, ind.nx2, ind.nx1);
  }
  const auto state = energy_window_state;
  const auto potential = phi;
  const auto u = hydro->u0;
  par_for("save_lat_gravity_work", DevExeSpace(), 0, nb-1,
          0, ind.nx3-1, 0, ind.nx2-1, 0, ind.nx1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    state(m,0,k,j,i) = u(m,IDN,k+ind.ks,j+ind.js,i+ind.is);
    state(m,1,k,j,i) = potential(m,0,k+ind.ks,j+ind.js,i+ind.is);
  });
  energy_window_start = pm->time;
  energy_window_end = pm->time+dt;
  energy_window_topology = pm->topology_version;
  for (Real &value : boundary_flux_local) value = 0.0;
  // Device-side floor tally for EOS paths without a host reduction (tabulated EOS).
  // Never zeroed here: conversions between the previous window end and this start
  // (boundary refresh after the centering correction, AMR) are still owed to the
  // ledger and are collected at this window's end.  resize keeps existing entries.
  if (hydro->floor_energy_block.extent_int(0) < nb) {
    Kokkos::resize(hydro->floor_energy_block, nb);
  }
  hydro->floor_energy_ledger = true;
  if (lat_ledger_debug) {
    const Real local = TotalGasEnergy();
    Real total = local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local, &total, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    debug_energy_begin = total;
    const Real mass_local = TotalConserved(IDN);
    Real mass_total = mass_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&mass_local, &mass_total, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    debug_mass_begin = mass_total;
    debug_dfloor_seen = pm->ecounter.neos_dfloor;
    debug_floor_mass_seen = pm->ecounter.eos_floor_mass;
    if (global_variable::my_rank == 0) {
      std::printf("LATLEDGER begin t=%.9e E_gas=%.12e mass=%.12e\n", pm->time, total,
          mass_total);
    }
  }
  // No hydro update separates the previous window's end from this start, so any
  // change of the discrete potential energies is a remap/re-solve effect.
  if (ledger_ref_valid) {
    Real u_self = 0.0, u_bh = 0.0;
    LedgerEnergies(u_self, u_bh);
    // Gas energy change since the previous window end, net of the floors applied by
    // the conversions in between (those stay in E_floor at this window's end).
    Real local[2] = {TotalGasEnergy(), GlobalFloorTallyLocal()};
    Real total[2] = {local[0], local[1]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(local, total, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    remap_energy_total += (u_self + u_bh) - (ledger_ref_self + ledger_ref_bh) +
                          (total[0] - ledger_ref_gas - total[1]);
    ledger_ref_valid = false;
    debug_ubh_begin = u_bh;
  } else if (lat_ledger_debug) {
    Real u_self = 0.0, u_bh = 0.0;
    LedgerEnergies(u_self, u_bh);
    debug_ubh_begin = u_bh;
  }
  if (lat_ledger_debug) {
    bool has_bh = false;
    Real mass = 0, softening = 0, newton_g = 0;
    problem_runtime::GetExternalBHPotentialAtExactTime(pm->time, has_bh, debug_bhx0[0],
        debug_bhx0[1], debug_bhx0[2], mass, softening, newton_g);
  }
  energy_window_open = true;
}

void Gravity::LedgerEnergies(Real &u_self, Real &u_bh) const {
  u_self = 0.0;
  u_bh = 0.0;
  auto *pm = pmy_pack->pmesh;
  auto *hydro = pmy_pack->phydro;
  if (hydro == nullptr) return;
  const bool self_potential = phi_valid && self_phi_time_valid && hydro->psrc != nullptr &&
      hydro->psrc->self_gravity;
  bool has_bh = false;
  Real bhx=0, bhy=0, bhz=0, mass=0, softening=0, newton_g=0;
  if (problem_runtime::ExternalBHGravitySourceCouplingEnabled()) {
    problem_runtime::GetExternalBHPotentialAtExactTime(pm->time, has_bh, bhx, bhy, bhz,
                                                       mass, softening, newton_g);
  }
  const Real rho_self_gate = (hydro->psrc != nullptr) ? hydro->psrc->rho_grav_min : 0.0;
  const Real rho_bh_gate = (hydro->psrc != nullptr) ? hydro->psrc->rho_external_bh_min : 0.0;
  const Real rho_gate_floor = hydro->peos->eos_data.dfloor;
  const auto ind = pm->mb_indcs;
  const int nx1 = ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int per_block = nx1*nx2*nx3;
  const int nwork = pmy_pack->nmb_thispack*per_block;
  const auto u0 = hydro->u0;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  const auto potential = phi;
  Real local[2] = {0.0, 0.0};
  Kokkos::parallel_reduce("gravity_ledger_energies",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork),
  KOKKOS_LAMBDA(const int &index, Real &self_sum, Real &bh_sum) {
    const int m = index/per_block;
    const int cell = index-m*per_block;
    const int k = cell/(nx1*nx2)+ks;
    const int j = (cell/nx1)%nx2+js;
    const int i = cell%nx1+is;
    const Real rho = u0(m,IDN,k,j,i);
    if (!(rho > 0.0)) return;
    const auto mb = size(m);
    const Real dm = rho*mb.dx1*mb.dx2*mb.dx3;
    // Each interaction energy weighs a cell as the source term couples it: 1/2 w rho phi
    // and w rho Phi_BH.  With 0<w<1 cells this is the natural energy of the coupling but
    // not a conserved one (utils/gravity_weight.hpp), so it then closes only to O(w').
    const Real w_self = self_potential ?
        gravity_weight::Weight(rho, rho_gate_floor, rho_self_gate) : 0.0;
    if (w_self > 0.0) {
      self_sum += 0.5*w_self*dm*potential(m,0,k,j,i);
    }
    const Real w_bh = has_bh ?
        gravity_weight::Weight(rho, rho_gate_floor, rho_bh_gate) : 0.0;
    if (w_bh > 0.0) {
      const Real x = CellCenterX(i-is, nx1, mb.x1min, mb.x1max);
      const Real y = CellCenterX(j-js, nx2, mb.x2min, mb.x2max);
      const Real z = CellCenterX(k-ks, nx3, mb.x3min, mb.x3max);
      bh_sum += w_bh*dm*problem_runtime::ExternalBHPotential(x,y,z,bhx,bhy,bhz,mass,
                                                             softening,newton_g);
    }
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]));
  Real total[2] = {local[0], local[1]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local, total, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  u_self = total[0];
  u_bh = total[1];
}

void Gravity::PrepareLATEnergyWindowEnd() {
  if (!lat_time_centered_work) return;
  auto *pm = pmy_pack->pmesh;
  const Real tolerance = 2048*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::abs(pm->time));
  if (!energy_window_open || std::abs(pm->time-energy_window_end)>tolerance ||
      pm->topology_version != energy_window_topology || pmy_pack->lat_active_mask_enabled) {
    throw std::runtime_error("LAT gravity work history must close on its original synchronized mesh");
  }
  // Permit the driver's endpoint solve, retaining phi as a warm initial guess.
  energy_window_open = false;
  energy_window_pending = true;
}

bool Gravity::EndLATEnergyWindow() {
  if (!lat_time_centered_work) return false;
  auto *pm = pmy_pack->pmesh;
  if (energy_window_open || !energy_window_pending || !PotentialMatchesTime(pm->time) ||
      pm->topology_version != energy_window_topology) {
    throw std::runtime_error("LAT gravity temporal work requires the fresh endpoint potential");
  }
  const auto ind = pm->mb_indcs;
  const auto state = energy_window_state;
  const auto potential = phi;
  const auto u = pmy_pack->phydro->u0;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  const int nmb = pmy_pack->nmb_thispack;
  const int per_block = ind.nx1*ind.nx2*ind.nx3;
  Real debug_energy_before = 0.0;
  if (lat_ledger_debug) {
    const Real local = TotalGasEnergy();
    debug_energy_before = local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local, &debug_energy_before, 1, MPI_ATHENA_REAL, MPI_SUM,
        MPI_COMM_WORLD);
#endif
  }
  // For a symmetric Poisson operator, Delta U_self = sum(Delta rho*phi_avg*dV).
  // The frozen-phi mass-flux work already accounts for Delta rho*phi_old.
  // This local temporal term supplies the missing half Delta rho*Delta phi;
  // it is not fitted to the measured total-energy residual.  The applied energy
  // is summed (double precision) into the ledger so a closure test can see it.
  Real work_local = 0.0, centered_local = 0.0, u0_local = 0.0, u1_local = 0.0;
  Kokkos::parallel_reduce("apply_lat_gravity_work",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*per_block),
  KOKKOS_LAMBDA(const int &index, Real &work, Real &centered, Real &u_old, Real &u_new) {
    const int m = index/per_block;
    const int cell = index-m*per_block;
    const int i = cell%ind.nx1, j = (cell/ind.nx1)%ind.nx2, k = cell/(ind.nx1*ind.nx2);
    const Real rho0 = state(m,0,k,j,i), phi0 = state(m,1,k,j,i);
    const Real rho1 = u(m,IDN,k+ind.ks,j+ind.js,i+ind.is);
    const Real phi1 = potential(m,0,k+ind.ks,j+ind.js,i+ind.is);
    const Real drho = rho1-rho0;
    const Real dphi = phi1-phi0;
    const Real de = -0.5*drho*dphi;
    u(m,IEN,k+ind.ks,j+ind.js,i+ind.is) += de;
    const auto mb = size(m);
    const Real dv = mb.dx1*mb.dx2*mb.dx3;
    work += de*dv;
    centered += drho*0.5*(phi0+phi1)*dv;
    u_old += 0.5*rho0*phi0*dv;
    u_new += 0.5*rho1*phi1*dv;
  }, Kokkos::Sum<Real>(work_local), Kokkos::Sum<Real>(centered_local),
     Kokkos::Sum<Real>(u0_local), Kokkos::Sum<Real>(u1_local));
  energy_window_pending = false;
  // Floors fold in everything since the previous window end, including the refresh
  // conversions that follow this call (they are attributed to the next window).
  const Real floor_now = pm->ecounter.eos_floor_energy;
  Real floor_delta_local = floor_now - floor_energy_seen;
  floor_energy_seen = floor_now;
  {
    auto *hydro = pmy_pack->phydro;
    auto floor_block = hydro->floor_energy_block;
    const int count = std::min(nmb, floor_block.extent_int(0));
    Real device_sum = 0.0;
    Kokkos::parallel_reduce("floor_block_sum", Kokkos::RangePolicy<>(DevExeSpace(), 0,
        count),
    KOKKOS_LAMBDA(const int &m,
        Real &s) { s += floor_block(m); }, Kokkos::Sum<Real>(device_sum));
    floor_delta_local += device_sum;
    Kokkos::deep_copy(floor_block, 0.0);
    // Conversions between this window end and the next window start (boundary refresh,
    // AMR) still tally: the array is only zeroed here and re-read at the next end.
  }
  Real ledger_local[15] = {boundary_flux_local[0], boundary_flux_local[1],
                           boundary_flux_local[2], boundary_flux_local[3], work_local,
                           centered_local, u0_local, u1_local, debug_source_energy_local,
                           floor_delta_local, debug_user_source_energy_local,
                           debug_reflux_energy_local, debug_all_face_energy_local,
                           debug_boundary_ghost_gap_local, debug_bh_work_local};
  Real increment[15];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(ledger_local, increment, 15, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int n = 0; n < 15; ++n) increment[n] = ledger_local[n];
#endif
  debug_user_source_energy_local = 0.0;
  debug_reflux_energy_local = 0.0;
  debug_all_face_energy_local = 0.0;
  debug_boundary_ghost_gap_local = 0.0;
  debug_bh_work_local = 0.0;
  for (int n = 0; n < 15; ++n) {
    if (!std::isfinite(increment[n])) {
      throw std::runtime_error("Nonfinite LAT gravity ledger increment");
    }
  }
  floor_energy_total += increment[9];
  asym_energy_total += (increment[7]-increment[6]) - increment[5];
  debug_source_energy_local = 0.0;
  if (lat_ledger_debug) {
    const Real local = TotalGasEnergy();
    Real after = local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local, &after, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    const Real mass_local = TotalConserved(IDN);
    Real mass_total = mass_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&mass_local, &mass_total, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    int dfloor_local = pm->ecounter.neos_dfloor - debug_dfloor_seen, dfloor_total = dfloor_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&dfloor_local, &dfloor_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    Real floor_mass_local = pm->ecounter.eos_floor_mass - debug_floor_mass_seen;
    Real floor_mass_total = floor_mass_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&floor_mass_local, &floor_mass_total, 1, MPI_ATHENA_REAL, MPI_SUM,
        MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      // Unexplained gas-energy change over the window: everything except sources,
      // the boundary energy flux and the floors tallied so far.
      const Real unexplained = (debug_energy_before - debug_energy_begin) - increment[8] +
                               increment[1] - increment[9] - increment[10];
      std::printf("LATLEDGER end t=%.9e E_before_corr=%.12e E_after_corr=%.12e diff=%.6e "
                  "dE_window=%.6e B_E_inc=%.6e floor_inc=%.6e user_src=%.6e unexplained=%.3e "
                  "allface_E_out=%.6e reflux_dE=%.6e div_residual=%.3e\n",
                  pm->time, debug_energy_before, after, after-debug_energy_before,
                  debug_energy_before - debug_energy_begin, increment[1], increment[9],
                  increment[10], unexplained, increment[12], increment[11],
                  (debug_energy_before - debug_energy_begin) - increment[8] - increment[9] -
                  increment[10] + increment[12] - increment[11]);
      std::printf("LATLEDGER mass t=%.9e dM_window=%.6e B_mass_inc=%.6e unexplained_mass=%.3e "
                  "dfloor_hits=%d floor_mass=%.3e unexplained_minus_floor=%.3e\n",
                  pm->time, mass_total - debug_mass_begin, increment[0],
                  mass_total - debug_mass_begin + increment[0], dfloor_total, floor_mass_total,
                  mass_total - debug_mass_begin + increment[0] - floor_mass_total);
    }
  }
  Real u_self_end = 0.0, u_bh_end = 0.0;
  Real ubh_transport = 0.0, ubh_motion = 0.0;
  if (lat_ledger_debug) {
    LedgerEnergies(u_self_end, u_bh_end);
    // Split dU_bhgas into transport sum((w1 rho1-w0 rho0) Phi(x1)) and motion
    // sum(w0 rho0 (Phi(x1)-Phi(x0))), each density carrying its own coupling weight.
    bool has_bh = false;
    Real bx1=0, by1=0, bz1=0, mass=0, softening=0, newton_g=0;
    problem_runtime::GetExternalBHPotentialAtExactTime(pm->time, has_bh, bx1, by1, bz1,
                                                       mass, softening, newton_g);
    if (has_bh) {
      const Real bx0 = debug_bhx0[0], by0 = debug_bhx0[1], bz0 = debug_bhx0[2];
      const Real gate = pmy_pack->phydro->psrc != nullptr ?
          pmy_pack->phydro->psrc->rho_external_bh_min : 0.0;
      const Real gate_floor = pmy_pack->phydro->peos->eos_data.dfloor;
      const auto ind2 = pm->mb_indcs;
      const auto st = energy_window_state;
      const auto uu = pmy_pack->phydro->u0;
      const auto sz = pmy_pack->pmb->mb_size.d_view;
      const int pb = ind2.nx1*ind2.nx2*ind2.nx3;
      Real loc[2] = {0.0, 0.0};
      Kokkos::parallel_reduce("debug_ubh_split", Kokkos::RangePolicy<>(DevExeSpace(), 0,
          nmb*pb),
      KOKKOS_LAMBDA(const int &index, Real &tr, Real &mo) {
        const int m = index/pb, cell = index-m*pb;
        const int i = cell%ind2.nx1, j = (cell/ind2.nx1)%ind2.nx2, k = cell/(ind2.nx1*ind2.nx2);
        const Real rho1 = uu(m,IDN,k+ind2.ks,j+ind2.js,i+ind2.is);
        const Real rho0 = st(m,0,k,j,i);
        const Real s1 = (rho1 > 0.0) ?
            gravity_weight::Weight(rho1, gate_floor, gate)*rho1 : 0.0;
        const Real s0 = (rho0 > 0.0) ?
            gravity_weight::Weight(rho0, gate_floor, gate)*rho0 : 0.0;
        if (!(s1 > 0.0) && !(s0 > 0.0)) return;
        const auto mb = sz(m);
        const Real dv = mb.dx1*mb.dx2*mb.dx3;
        const Real x = CellCenterX(i, ind2.nx1, mb.x1min, mb.x1max);
        const Real y = CellCenterX(j, ind2.nx2, mb.x2min, mb.x2max);
        const Real z = CellCenterX(k, ind2.nx3, mb.x3min, mb.x3max);
        const Real p1 = problem_runtime::ExternalBHPotential(x,y,z,bx1,by1,bz1,mass,
            softening,newton_g);
        const Real p0 = problem_runtime::ExternalBHPotential(x,y,z,bx0,by0,bz0,mass,
            softening,newton_g);
        tr += (s1-s0)*p1*dv;
        mo += s0*(p1-p0)*dv;
      }, Kokkos::Sum<Real>(loc[0]), Kokkos::Sum<Real>(loc[1]));
      Real tot[2] = {loc[0], loc[1]};
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(loc, tot, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      ubh_transport = tot[0]; ubh_motion = tot[1];
    }
  }
  if (lat_ledger_debug && global_variable::my_rank == 0) {
    // Expected for frozen phi0 over the window: source = -sum(drho*phi0) = -(S +
    // W_applied)
    const Real expected_source = -(increment[5]+increment[4]);
    std::printf("LATLEDGER window [%.9e,%.9e] U0=%.12e U1=%.12e dU=%.6e S=%.6e dU-S=%.3e "
                "W_applied=%.6e src_eff=%.6e src_expected=%.6e src-expected=%.3e "
                "floor_energy=%.6e B_Wself_inc=%.6e ghost_gap=%.6e\n",
                energy_window_start, pm->time, increment[6], increment[7],
                increment[7]-increment[6], increment[5], increment[7]-increment[6]-increment[5],
                increment[4], increment[8], expected_source, increment[8]-expected_source,
                increment[9], increment[2], increment[13]);
    // Self-gravity work identity alone (interface non-telescoping shows up here) and
    // the BH-gas pair: src_bh + dU_bh (+ dK_BH from the history) should vanish.
    std::printf("LATLEDGER split t=%.9e src_self=%.6e self_identity=%.3e src_bh=%.6e dU_bh=%.6e src_bh+dU_bh=%.6e "
                "ubh_transport=%.6e ubh_motion=%.6e src_bh+transport=%.3e\n",
                pm->time, increment[8]-increment[14],
                (increment[8]-increment[14]) + increment[5] + increment[4] + increment[2],
                increment[14], u_bh_end - debug_ubh_begin, increment[14] + u_bh_end - debug_ubh_begin,
                ubh_transport, ubh_motion, increment[14] + ubh_transport);
  }
  centered_work_total += increment[4];
  if (lat_boundary_flux_diagnostics) {
    for (int n = 0; n < 4; ++n) {
      boundary_flux_total[n] += increment[n];
      boundary_flux_local[n] = 0.0;
    }
  }
  LedgerEnergies(ledger_ref_self, ledger_ref_bh);
  {
    const Real local = TotalGasEnergy();
    ledger_ref_gas = local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local, &ledger_ref_gas, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  }
  ledger_ref_valid = true;
  return true;  // Caller must recover and publish updated primitives/dual energy.
}

void Gravity::StoreBoundaryFluxMetadata(ParameterInput *pin) const {
  if (!lat_time_centered_work) return;
  if (energy_window_open || energy_window_pending) {
    throw std::runtime_error("Cannot checkpoint a partially accumulated LAT gravity ledger");
  }
  pin->SetReal("gravity", "lat_centered_work_total", centered_work_total);
  pin->SetReal("gravity", "lat_remap_energy_total", remap_energy_total);
  pin->SetReal("gravity", "lat_floor_energy_total", floor_energy_total);
  pin->SetReal("gravity", "lat_asym_energy_total", asym_energy_total);
  pin->SetReal("gravity", "lat_ledger_ref_self", ledger_ref_self);
  pin->SetReal("gravity", "lat_ledger_ref_bh", ledger_ref_bh);
  pin->SetReal("gravity", "lat_ledger_ref_gas", ledger_ref_gas);
  pin->SetBoolean("gravity", "lat_ledger_ref_valid", ledger_ref_valid);
  if (!lat_boundary_flux_diagnostics) return;
  const char *names[] = {"boundary_mass_out", "boundary_gas_energy_out",
                        "boundary_self_mass_work_out", "boundary_bh_mass_work_out"};
  for (int n = 0; n < 4; ++n) pin->SetReal("gravity", names[n], boundary_flux_total[n]);
}

Real Gravity::TotalConserved(int n) const {
  auto *hydro = pmy_pack->phydro;
  if (hydro == nullptr) return 0.0;
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int nx1 = ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int per_block = nx1*nx2*nx3;
  const int nwork = pmy_pack->nmb_thispack*per_block;
  const auto u0 = hydro->u0;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  Real local = 0.0;
  Kokkos::parallel_reduce("gravity_total_conserved",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork),
  KOKKOS_LAMBDA(const int &index, Real &e) {
    const int m = index/per_block;
    const int cell = index-m*per_block;
    const int k = cell/(nx1*nx2)+ks;
    const int j = (cell/nx1)%nx2+js;
    const int i = cell%nx1+is;
    const auto mb = size(m);
    e += u0(m,n,k,j,i)*mb.dx1*mb.dx2*mb.dx3;
  }, Kokkos::Sum<Real>(local));
  return local;
}

Real Gravity::GlobalFloorTallyLocal() const {
  // Host counter delta since the last window end plus the device per-block tally
  // (zeroed at the last window end).
  Real value = pmy_pack->pmesh->ecounter.eos_floor_energy - floor_energy_seen;
  auto *hydro = pmy_pack->phydro;
  if (hydro != nullptr) {
    auto floor_block = hydro->floor_energy_block;
    const int count = std::min(pmy_pack->nmb_thispack, floor_block.extent_int(0));
    Real device_sum = 0.0;
    Kokkos::parallel_reduce("floor_block_peek", Kokkos::RangePolicy<>(DevExeSpace(), 0,
        count),
    KOKKOS_LAMBDA(const int &m,
        Real &s) { s += floor_block(m); }, Kokkos::Sum<Real>(device_sum));
    value += device_sum;
  }
  return value;
}

Real Gravity::TotalGasEnergy() const {
  auto *hydro = pmy_pack->phydro;
  if (hydro == nullptr) return 0.0;
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int nx1 = ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int per_block = nx1*nx2*nx3;
  const int nwork = pmy_pack->nmb_thispack*per_block;
  const auto u0 = hydro->u0;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  Real local = 0.0;
  Kokkos::parallel_reduce("gravity_total_gas_energy",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork),
  KOKKOS_LAMBDA(const int &index, Real &e) {
    const int m = index/per_block;
    const int cell = index-m*per_block;
    const int k = cell/(nx1*nx2)+ks;
    const int j = (cell/nx1)%nx2+js;
    const int i = cell%nx1+is;
    const auto mb = size(m);
    e += u0(m,IEN,k,j,i)*mb.dx1*mb.dx2*mb.dx3;
  }, Kokkos::Sum<Real>(local));
  return local;
}

void Gravity::LedgerPotentialMoments(Real &phi_int, Real &mass, Real &volume) const {
  phi_int = mass = volume = 0.0;
  auto *hydro = pmy_pack->phydro;
  if (hydro == nullptr) return;
  const auto ind = pmy_pack->pmesh->mb_indcs;
  const int nx1 = ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int per_block = nx1*nx2*nx3;
  const int nwork = pmy_pack->nmb_thispack*per_block;
  const auto u0 = hydro->u0;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  const auto potential = phi;
  const bool have_phi = phi_valid;
  Real local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("gravity_ledger_moments",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork),
  KOKKOS_LAMBDA(const int &index, Real &p, Real &m_sum, Real &v_sum) {
    const int m = index/per_block;
    const int cell = index-m*per_block;
    const int k = cell/(nx1*nx2)+ks;
    const int j = (cell/nx1)%nx2+js;
    const int i = cell%nx1+is;
    const auto mb = size(m);
    const Real dv = mb.dx1*mb.dx2*mb.dx3;
    if (have_phi) p += potential(m,0,k,j,i)*dv;
    m_sum += u0(m,IDN,k,j,i)*dv;
    v_sum += dv;
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]), Kokkos::Sum<Real>(local[2]));
  Real total[3] = {local[0], local[1], local[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local, total, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  phi_int = total[0];
  mass = total[1];
  volume = total[2];
}

void Gravity::AccumulateLATBoundaryFlux(Real final_dt) {
  if (!lat_boundary_flux_diagnostics) return;
  // The ledger reads Hydro's face fluxes below.  Its one caller is Hydro's source-term
  // task, and BeginLATEnergyWindow refuses to open a window without Hydro, so an MHD run
  // (self-gravity included) fails there first; the test keeps this function from
  // dereferencing a null Hydro on its own.
  if (!energy_window_open || !phi_valid || !std::isfinite(final_dt) || final_dt < 0 ||
      pmy_pack->phydro == nullptr) {
    throw std::runtime_error("Boundary transport requires a valid active LAT energy window");
  }
  const auto *pm = pmy_pack->pmesh;
  const auto ind = pm->mb_indcs;
  const auto active = pmy_pack->lat_active_indices.d_view;
  const auto step_dt = pmy_pack->lat_step_dt.d_view;
  const bool masked = pmy_pack->lat_active_mask_enabled;
  const bool per_block_dt = pmy_pack->lat_per_block_timestep;
  const int nwork = masked ? pmy_pack->lat_nactive_thispack : pmy_pack->nmb_thispack;
  if (!nwork || final_dt == 0) return;
  const Real weight = final_dt/pm->dt;
  const auto bc = pmy_pack->pmb->mb_bcs.d_view;
  const auto size = pmy_pack->pmb->mb_size.d_view;
  const auto potential = phi;
  const auto f1 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x1f);
  const auto f2 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x2f);
  const auto f3 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x3f);
  bool has_bh = false;
  Real bhx=0, bhy=0, bhz=0, mass=0, softening=0, newton_g=0;
  problem_runtime::GetExternalBHPotential(problem_runtime::HydroStageTimeOr(pm->time),
      has_bh, bhx,bhy,bhz,mass,softening,newton_g);
  // Physical boundary faces only. The owned boundary flux already includes
  // inter-rank flux exchange; internal coarse/fine faces belong to reflux, not
  // this ledger. Stage weights are final RK weights, including union block dt.
  Real local[6] = {};
  const int per_block = ind.nx1*ind.nx2*ind.nx3;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  const bool all_faces_debug = lat_ledger_debug;
  Kokkos::parallel_reduce("lat_physical_boundary_flux",
      Kokkos::RangePolicy<>(DevExeSpace(),0,nwork*per_block),
  KOKKOS_LAMBDA(const int &index, Real &dm, Real &de, Real &self_work, Real &bh_work,
                Real &de_all, Real &ghost_gap) {
    const int a = index/per_block;
    const int m = masked ? active(a) : a;
    const int cell = index-a*per_block;
    const int pos[3] = {cell%ind.nx1+ind.is, (cell/ind.nx1)%ind.nx2+ind.js,
                       cell/(ind.nx1*ind.nx2)+ind.ks};
    const int first[3] = {ind.is,ind.js,ind.ks};
    const int last[3] = {ind.ie,ind.je,ind.ke};
    const int counts[3] = {ind.nx1,ind.nx2,ind.nx3};
    const auto mb = size(m);
    const Real lower[3] = {mb.x1min,mb.x2min,mb.x3min};
    const Real upper[3] = {mb.x1max,mb.x2max,mb.x3max};
    const Real widths[3] = {mb.dx1,mb.dx2,mb.dx3};
    const Real dt = per_block_dt ? weight*step_dt(m) : final_dt;
    for (int d = 0; d < 3; ++d) {
      if ((d==1 && !multi_d) || (d==2 && !three_d)) continue;
      for (int side = 0; side < 2; ++side) {
        if (pos[d] != (side ? last[d] : first[d])) continue;
        const auto boundary = bc(m,2*d+side);
        const bool physical = !(boundary == BoundaryFlag::block ||
                                boundary == BoundaryFlag::periodic);
        if (!physical && !all_faces_debug) continue;
        int face[3] = {pos[0],pos[1],pos[2]};
        int neighbor[3] = {pos[0],pos[1],pos[2]};
        if (side) ++face[d];
        neighbor[d] += side ? 1 : -1;
        const Real mass_flux = d==0 ? f1(m,IDN,face[2],face[1],face[0]) :
            (d==1 ? f2(m,IDN,face[2],face[1],face[0]) : f3(m,IDN,face[2],face[1],
                face[0]));
        const Real energy_flux = d==0 ? f1(m,IEN,face[2],face[1],face[0]) :
            (d==1 ? f2(m,IEN,face[2],face[1],face[0]) : f3(m,IEN,face[2],face[1],
                face[0]));
        const Real signed_area_dt = (side ? dt : -dt)*mb.dx1*mb.dx2*mb.dx3/widths[d];
        de_all += signed_area_dt*energy_flux;
        if (!physical) continue;
        const Real transported_mass = signed_area_dt*mass_flux;
        dm += transported_mass;
        de += signed_area_dt*energy_flux;
        self_work += transported_mass*0.5*(potential(m,0,pos[2],pos[1],pos[0]) +
                            potential(m,0,neighbor[2],neighbor[1],neighbor[0]));
        ghost_gap += transported_mass*0.5*(potential(m,0,neighbor[2],neighbor[1],
            neighbor[0]) -
                            potential(m,0,pos[2],pos[1],pos[0]));
        if (has_bh) {
          Real x[3],xn[3];
          for (int q = 0; q < 3; ++q) {
            x[q] = CellCenterX(pos[q]-first[q],counts[q],lower[q],upper[q]);
            xn[q] = CellCenterX(neighbor[q]-first[q],counts[q],lower[q],upper[q]);
          }
          const Real phi_face = 0.5*(problem_runtime::ExternalBHPotential(
              x[0],x[1],x[2],bhx,bhy,bhz,mass,softening,newton_g) +
              problem_runtime::ExternalBHPotential(
              xn[0],xn[1],xn[2],bhx,bhy,bhz,mass,softening,newton_g));
          bh_work += transported_mass*phi_face;
        }
      }
    }
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]),
     Kokkos::Sum<Real>(local[2]), Kokkos::Sum<Real>(local[3]), Kokkos::Sum<Real>(local[4]),
     Kokkos::Sum<Real>(local[5]));
  for (int n = 0; n < 4; ++n) boundary_flux_local[n] += local[n];
  debug_all_face_energy_local += local[4];
  debug_boundary_ghost_gap_local += local[5];
}

} // namespace gravity
