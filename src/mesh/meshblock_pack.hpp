#ifndef MESH_MESHBLOCK_PACK_HPP_
#define MESH_MESHBLOCK_PACK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_pack.hpp
//  \brief defines MeshBlockPack class, a container for MeshBlocks

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "tasklist/task_list.hpp"

// Forward declarations
class MeshBlock;
class ADM;
class Tmunu;
namespace hydro {class Hydro;}
namespace mhd {class MHD;}
namespace ion_neutral {class IonNeutral;}
namespace radiation {class Radiation;}
namespace dyngr {class DynGRMHD;}
namespace numrel {class NumericalRelativity;}
class TurbulenceDriver;
namespace radiation {class Radiation;}
namespace z4c {class Z4c;}
namespace z4c {class CCE;}
namespace adm {class ADM;}
namespace particles {class Particles;}
namespace units {class Units;}
namespace gravity {class Gravity;}
namespace sinkparticles {class SinkParticles;}

//----------------------------------------------------------------------------------------
//! \class MeshBlockPack
//! \brief data/functions associated with a single block

class MeshBlockPack {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class Mesh;
  friend class MeshBlock;
  friend class MeshBlockTree;

 public:
  MeshBlockPack(Mesh *pm, int igids, int igide);
  ~MeshBlockPack();

  // data
  Mesh *pmesh;            // ptr to Mesh containing this MeshBlockPack
  int gids, gide;         // start/end of global IDs in this MeshBlockPack
  int nmb_thispack;       // number of MBs in this pack
  bool lat_active_mask_enabled{false};
  int lat_nactive_thispack{0};
  int lat_nboundary_send_thispack{0};
  int lat_nboundary_send_edges_thispack{0};
  int lat_nflux_recv_thispack{0};
  bool lat_union_stage1_enabled{false};
  bool lat_per_block_timestep{false};
  int lat_union_pending_below_factor{0};
  // Cached boundary-need flags, computed during SetActiveMeshBlocks* to avoid
  // re-scanning neighbor lists in RefreshHydroLATBoundaries on every tick.
  bool lat_cached_needs_restrict{false};
  bool lat_cached_needs_prolongate{false};
  bool lat_cached_needs_physical_bc{false};
  bool lat_cached_needs_neighbor_recv{false};
  DualArray1D<int> lat_active_mb;  // 1 when a MeshBlock participates in this LAT substep
  DualArray1D<int> lat_active_indices;  // compact list of active MeshBlock indices
  DualArray1D<int> lat_boundary_send_mb;  // 1 when a MeshBlock may need to pack bvals
  DualArray1D<int> lat_boundary_send_indices;  // compact list of boundary senders
  DualArray1D<int> lat_boundary_send_edges;  // encoded m*nnghbr+n selected send edges
  DualArray1D<int> lat_flux_recv_indices;  // compact list of delayed reflux receivers
  DualArray2D<int> lat_send_nghbr;        // 1 when this neighbor's receiver is active
  DualArray2D<int> lat_flux_accum_nghbr;  // 1 when this MB accumulates its own slow flux
  DualArray2D<int> lat_flux_send_nghbr;   // 1 when this MB sends fine flux to coarse
  DualArray2D<int> lat_flux_recv_nghbr;   // 1 when this MB receives fine flux from active fine
  DualArray2D<int> lat_nghbr_factor;      // LAT factor for each existing neighbor
  DualArray1D<Real> lat_time_start;  // time represented by u1 for this MeshBlock
  DualArray1D<Real> lat_time_end;    // time represented by u0 for this MeshBlock
  DualArray1D<Real> lat_step_dt;     // timestep of each MeshBlock in a union LAT stage
  DualArray1D<int> lat_step_factor;  // cached power-of-two timestep factor for each MB
  DualArray1D<int> lat_step_level;   // cached log2(lat_step_factor), capped at 20

  // following Grid/Physics objects are all pointers so they can be allocated after
  // MeshBlockPack is constructed with pointer to my_pack.

  MeshBlock* pmb=nullptr;         // MeshBlocks in this MeshBlockPack
  Coordinates* pcoord=nullptr;

  // physics (controlled by AddPhysics() function in meshblock_pack.cpp)
  hydro::Hydro *phydro=nullptr;
  mhd::MHD *pmhd=nullptr;
  adm::ADM *padm=nullptr;
  Tmunu *ptmunu=nullptr;
  z4c::Z4c *pz4c=nullptr;
  dyngr::DynGRMHD *pdyngr=nullptr;
  numrel::NumericalRelativity *pnr=nullptr;
  ion_neutral::IonNeutral *pionn=nullptr;
  TurbulenceDriver *pturb=nullptr;
  radiation::Radiation *prad=nullptr;
  // Krumholz-style sink particles.  Globally replicated list, driver-invoked once per
  // cycle; its only per-RK-stage coupling is the SourceTerms sink-gravity kick.
  sinkparticles::SinkParticles *psink=nullptr;
  std::vector<z4c::CCE *> pz4c_cce;
  particles::Particles *ppart=nullptr;
  gravity::Gravity *pgrav=nullptr;


  // units (needed to convert code units to cgs for, e.g., cooling or radiation)
  units::Units *punit=nullptr;

  // map for task lists which operate over all MeshBlocks in this MeshBlockPack
  std::map<std::string, std::shared_ptr<TaskList>> tl_map;

  // functions
  void AddPhysics(ParameterInput *pin);
  //! The code temperature unit in K, i.e. the cgs value of the number every kernel in
  //! the run calls "temperature".  It belongs to the ACTIVE EOS, not to <units>:
  //! Units::temperature_cgs() is the ideal-gas unit built from <units>/mu and equals
  //! this one only for the ideal EOS, while a tabulated LTE/Saha EOS defines its own
  //! scale from m_H/k_B.  Every consumer that converts a code temperature to or from
  //! cgs must use this accessor.
  Real TemperatureUnitCGS() const;
  void AddMeshBlocks(ParameterInput *pin);
  void AddCoordinates(ParameterInput *pin);
  void ResizeActiveMask();
  void SetAllMeshBlocksActive();
  void ReleaseLATCacheMemory();
  void DetachLATActiveMask();
  void ResetLATBlockTimes(Real time);
  void SetActiveLATBlockTimes(Real start_time, Real end_time);
  void ConfigureLATUnionStage1(Real start_time, Real fine_dt, int max_factor);
  bool LATActiveMaskUsesCache() const;
  std::uint64_t LATCacheGeneration() const;
  int LATFactorForGID(int gid, int max_factor) const;
  int SetLATFluxCorrectionByCompletionPhase(int max_factor, int completed_tick);
  int SetActiveMeshBlocksByLATFactor(int max_factor, int active_factor);
  int SetActiveMeshBlocksByLATDueFactors(int max_factor, int tick_cycle,
                                         bool include_factor_one);
  int SelectLATDueUpdateFluxReceivers(int max_factor, int tick_cycle,
                                      bool include_factor_one);

 private:
  // data
  bool lat_active_mask_uses_cache{false};
  std::uint64_t lat_cache_generation{1};
  // Persistent detach-scratch set installed by DetachLATActiveMask.  Every consumer
  // reads these arrays only through the *_thispack counts written after each host
  // rebuild, so reusing them across detaches is safe, avoids ~10 device allocations
  // per detach, and keeps pointer identity stable for the rank-packed layout caches.
  DualArray1D<int> lat_detach_scratch_active_mb;
  DualArray1D<int> lat_detach_scratch_active_indices;
  DualArray1D<int> lat_detach_scratch_boundary_send_mb;
  DualArray1D<int> lat_detach_scratch_boundary_send_indices;
  DualArray1D<int> lat_detach_scratch_boundary_send_edges;
  DualArray1D<int> lat_detach_scratch_flux_recv_indices;
  DualArray2D<int> lat_detach_scratch_send_nghbr;
  DualArray2D<int> lat_detach_scratch_flux_accum_nghbr;
  DualArray2D<int> lat_detach_scratch_flux_send_nghbr;
  DualArray2D<int> lat_detach_scratch_flux_recv_nghbr;
  int lat_factor_cache_max_factor{0};
  int lat_factor_cache_nmb{0};
  int lat_factor_cache_nnghbr{0};
  std::uint64_t lat_factor_cache_metadata_version{0};
  std::vector<int> lat_factor_cache_values;
  std::vector<int> lat_factor_cache_nactive;
  std::vector<int> lat_factor_cache_nboundary_send;
  std::vector<int> lat_factor_cache_nboundary_send_edges;
  std::vector<int> lat_factor_cache_nflux_recv;
  std::vector<DualArray1D<int>> lat_factor_cache_active_mb;
  std::vector<DualArray1D<int>> lat_factor_cache_active_indices;
  std::vector<DualArray1D<int>> lat_factor_cache_boundary_send_mb;
  std::vector<DualArray1D<int>> lat_factor_cache_boundary_send_indices;
  std::vector<DualArray1D<int>> lat_factor_cache_boundary_send_edges;
  std::vector<DualArray1D<int>> lat_factor_cache_flux_recv_indices;
  std::vector<DualArray2D<int>> lat_factor_cache_send_nghbr;
  std::vector<DualArray2D<int>> lat_factor_cache_flux_accum_nghbr;
  std::vector<DualArray2D<int>> lat_factor_cache_flux_send_nghbr;
  std::vector<DualArray2D<int>> lat_factor_cache_flux_recv_nghbr;
  // Cached boundary-need flags per slot
  std::vector<bool> lat_factor_cache_needs_restrict;
  std::vector<bool> lat_factor_cache_needs_prolongate;
  std::vector<bool> lat_factor_cache_needs_physical_bc;
  std::vector<bool> lat_factor_cache_needs_neighbor_recv;
  int lat_due_cache_max_factor{0};
  int lat_due_cache_nmb{0};
  int lat_due_cache_nnghbr{0};
  int lat_due_cache_include_factor_one{0};
  int lat_due_cache_union_stage1{0};
  std::uint64_t lat_due_cache_metadata_version{0};
  // Bit n records whether lat_factor_cache_values[n] is due.  Power-of-two
  // schedules repeat these signatures within a cycle, so keying by signature
  // avoids storing duplicate masks for phases with the same due-factor set.
  std::vector<std::uint64_t> lat_due_cache_signatures;
  std::vector<int> lat_due_cache_nactive;
  std::vector<int> lat_due_cache_nboundary_send;
  std::vector<int> lat_due_cache_nboundary_send_edges;
  std::vector<int> lat_due_cache_nflux_recv;
  std::vector<int> lat_due_cache_nupdate_flux_recv;
  std::vector<DualArray1D<int>> lat_due_cache_active_mb;
  std::vector<DualArray1D<int>> lat_due_cache_active_indices;
  std::vector<DualArray1D<int>> lat_due_cache_boundary_send_mb;
  std::vector<DualArray1D<int>> lat_due_cache_boundary_send_indices;
  std::vector<DualArray1D<int>> lat_due_cache_boundary_send_edges;
  std::vector<DualArray1D<int>> lat_due_cache_flux_recv_mb;
  std::vector<DualArray1D<int>> lat_due_cache_flux_recv_indices;
  std::vector<DualArray1D<int>> lat_due_cache_update_flux_recv_indices;
  std::vector<DualArray2D<int>> lat_due_cache_send_nghbr;
  std::vector<DualArray2D<int>> lat_due_cache_flux_accum_nghbr;
  std::vector<DualArray2D<int>> lat_due_cache_flux_send_nghbr;
  std::vector<DualArray2D<int>> lat_due_cache_flux_recv_nghbr;
  std::vector<DualArray2D<int>> lat_due_cache_update_flux_recv_nghbr;
  // Cached boundary-need flags per slot
  std::vector<bool> lat_due_cache_needs_restrict;
  std::vector<bool> lat_due_cache_needs_prolongate;
  std::vector<bool> lat_due_cache_needs_physical_bc;
  std::vector<bool> lat_due_cache_needs_neighbor_recv;

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);
  void AdvanceLATCacheGeneration();
  void InvalidateLATFactorCache();
  int LATFactorCacheSlot(int factor) const;
  void ClearLATDueFactorCache();
  std::uint64_t LATDueFactorSignature(int phase, bool include_factor_one) const;
  int LATDueFactorCacheSlot(std::uint64_t signature) const;
  bool BuildLATFactorCache(int max_factor);
  bool BuildLATDueFactorCache(int max_factor, int tick_cycle, bool include_factor_one);
};

#endif // MESH_MESHBLOCK_PACK_HPP_
