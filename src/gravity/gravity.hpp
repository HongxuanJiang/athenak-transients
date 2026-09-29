#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../mesh/meshblock_pack.hpp"
#include "../parameter_input.hpp"
#include "../multigrid/multigrid.hpp"
#include "../coordinates/coordinates.hpp"
#include "mg_gravity.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
namespace gravity {
class Gravity {
 public:
  Gravity(MeshBlockPack *pmbp, ParameterInput *pin);
  ~Gravity();

  MeshBlockPack* pmy_pack;
  DvceArray5D<Real> phi, coarse_phi;  // numerical self-gravity potential
  Real four_pi_G;
  bool phi_valid;
  bool self_phi_time_valid;
  Real self_phi_time = 0.0;
  std::uint64_t self_phi_topology = std::numeric_limits<std::uint64_t>::max();
  bool reuse_phi_after_amr;
  bool lat_time_centered_work = false;
  bool energy_window_open = false;
  bool energy_window_pending = false;
  bool lat_boundary_flux_diagnostics = false;
  bool lat_ledger_debug = false;  // print per-window identity terms on rank 0
  Real debug_source_energy_local = 0.0;  // effective source energy this window (debug)
  Real debug_energy_begin = 0.0;         // global gas energy at the window start (debug)
  Real debug_user_source_energy_local = 0.0;  // effective pgen user-source energy (debug)
  Real debug_reflux_energy_local = 0.0;       // energy change by the LAT flux correction
  Real debug_all_face_energy_local = 0.0;     // weighted outward energy flux, all block faces
  Real debug_boundary_ghost_gap_local = 0.0;  // sum m_out*(phi_ghost-phi_interior)/2 at physical faces
  Real debug_bh_work_local = 0.0;             // effective external-BH part of the MHG energy source
  Real debug_stage_weight = 1.0;              // FinalFluxWeight/beta of the stage being applied
  Real debug_ubh_begin = 0.0;                 // U_bhgas at the window start
  Real debug_bhx0[3] = {0.0, 0.0, 0.0};       // BH position at the window start
  Real TotalGasEnergy() const;  // rank-local sum of u0(IEN) dV over active cells
  Real TotalConserved(int n) const;  // rank-local sum of u0(n) dV over active cells
  Real debug_mass_begin = 0.0;
  int debug_dfloor_seen = 0;
  Real debug_floor_mass_seen = 0.0;
  // Outward-positive transported mass, gas energy, self-potential mass flux,
  // and BH-potential mass flux. Field-energy boundary transport is separate.
  Real boundary_flux_local[4] = {};
  Real boundary_flux_total[4] = {};
  // Cumulative energy added to the gas by the window-end time-centering term
  // -1/2 sum(Delta rho * Delta phi * dV); a diagnostic, reduced across ranks at
  // every window end and carried through restarts.
  Real centered_work_total = 0.0;
  // Discrete potential energies change when AMR/load balancing remaps the mesh and
  // the potential is re-solved on the new topology.  The change of the gated
  // U_self + U_bhgas between one window's end and the next window's start (no hydro
  // update in between) is accumulated here so a closure test can separate the
  // remap contribution from time-integration error.
  Real remap_energy_total = 0.0;
  // Cumulative energy added by EOS floors/ceilings/excision resets (from
  // Mesh::ecounter.eos_floor_energy, reduced over ranks at every window end).
  Real floor_energy_total = 0.0;
  // Cumulative Poisson-operator asymmetry term: per window, the actual change of
  // 1/2 sum(rho phi dV) minus sum(Delta rho * phi_avg * dV).  Zero to round-off for a
  // symmetric operator (uniform mesh, or mg_fc_symmetric=true); otherwise the
  // coarse/fine interface asymmetry that the local centering term cannot see.
  Real asym_energy_total = 0.0;
  Real floor_energy_seen = 0.0;  // rank-local counter value already folded in
  Real ledger_ref_self = 0.0, ledger_ref_bh = 0.0;
  // Global gas total energy right after the window-end correction; the change to the
  // next window start, net of floors applied in between, is the gas part of E_remap
  // (prolongation/restriction of the conserved state is not exactly energy-neutral).
  Real ledger_ref_gas = 0.0;
  bool ledger_ref_valid = false;
  Real GlobalFloorTallyLocal() const;  // rank-local floor energy not yet folded in
  // Gated 1/2 sum(rho phi dV) and sum(rho Phi_BH dV) on the current synchronized
  // state, reduced over all ranks (double precision).
  void LedgerEnergies(Real &u_self, Real &u_bh) const;
  // Volume integral of phi over the mesh, and total gas mass and mesh volume, so a
  // closure test can form the Jeans-swindle energy 1/2 sum((rho-<rho>) phi dV).
  void LedgerPotentialMoments(Real &phi_int, Real &mass, Real &volume) const;
  void AccumulateLATBoundaryFlux(Real final_dt);
  void StoreBoundaryFluxMetadata(ParameterInput *pin) const;
  DvceArray5D<Real> energy_window_state;
  Real energy_window_start = 0.0, energy_window_end = 0.0;
  std::uint64_t energy_window_topology = std::numeric_limits<std::uint64_t>::max();
  bool PotentialMatchesTime(Real time) const;
  void BeginLATEnergyWindow(Real dt);
  void PrepareLATEnergyWindowEnd();
  bool EndLATEnergyWindow();
  MGGravityDriver *pmgd;
  MGGravity *pmg;
  bool ResizeMeshBlockStorage(int nmb, bool exact = false,
                              bool allow_shrink = false);
  void MarkPhiInvalid();
  void MarkSelfPhiValid();
  bool RefreshPhiGhosts(bool require_valid = true);

  friend class MGGravityDriver;
};
}  // namespace gravity
#endif // GRAVITY_GRAVITY_HPP_
