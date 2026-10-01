#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//========================================================================================
// Athena++K astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.hpp
//! \brief defines Mesh class.
//! The Mesh is the overall grid structure, which is divided into local patches called
//! MeshBlocks (potentially on different levels) that tile the entire domain.  MeshBlocks
//! are grouped together into MeshBlockPacks for better performance on GPUs.

#include <cstdint>  // int32_t
#include <memory>
#include <string>

#include "athena.hpp"

// Define following structure before other "include" files to resolve declarations
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//! \brief physical size in a Mesh or a MeshBlock

struct RegionSize {
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real dx1, dx2, dx3;       // (uniform) grid spacing
};

//----------------------------------------------------------------------------------------
//! \struct RegionIndcs
//! \brief Cell indices and number of active and ghost cells in a Mesh or a MeshBlock

struct RegionIndcs {
  int ng;                       // number of ghost cells
  int nx1, nx2, nx3;            // number of active cells (not including ghost zones)
  int is,ie,js,je,ks,ke;        // indices of ACTIVE cells
  int cnx1, cnx2, cnx3;         // number of active coarse cells (not including gzs)
  int cis,cie,cjs,cje,cks,cke;  // indices of ACTIVE coarse cells
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//! \brief Information about neighboring MeshBlocks stored as 2D DualArray in MeshBlock

struct NeighborBlock {
  int gid;     // global ID
  int lev;     // logical level
  int rank;    // MPI rank
  int dest;    // index of recv buffer in target NeighborBlocks
};

//----------------------------------------------------------------------------------------
//! \struct LogicalLocation
//! \brief logical location and level of MeshBlock stored as POD
//! lx1/2/3 = logical location in x1/2/3 = index in array of nodes at current level
//! WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
//! of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
//! max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer types

struct LogicalLocation {
  std::int32_t lx1, lx2, lx3, level;
};

//----------------------------------------------------------------------------------------
//! \struct EventCounters
//! \brief stores various counters used as diagnostics throughout the code

struct EventCounters {
  int nfofc, neos_dfloor, neos_efloor, neos_tfloor, neos_vceil, neos_fail, maxit_c2p;
  // Rank-local cumulative total energy (sum over interior cells of dE*dV, weighted by
  // the RK stage's contribution to the completed step) added or removed by EOS floors,
  // ceilings and excision resets in conserved-to-primitive conversions.
  Real eos_floor_energy;
  Real eos_floor_mass;  // same weighting, for the mass added/removed by floors
  EventCounters() : nfofc(0), neos_dfloor(0), neos_efloor(0), neos_tfloor(0),
                    neos_vceil(0), neos_fail(0), maxit_c2p(0), eos_floor_energy(0.0),
                    eos_floor_mass(0.0) {}
};

// Forward declarations required due to recursive definitions amongst mesh classes
class MeshBlock;
class MeshBlockPack;
class MeshBlockTree;
class Mesh;

#include "parameter_input.hpp"
#include "meshblock.hpp"
#include "meshblock_pack.hpp"
#include "meshblock_tree.hpp"
#include "mesh_refinement.hpp"

//----------------------------------------------------------------------------------------
//! \class Mesh
//! \brief data/functions associated with the overall mesh

class Mesh {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree, MeshRefinement)
  // like to play together
  friend class MeshBlock;
  friend class MeshBlockPack;
  friend class MeshBlockTree;
  friend class MeshRefinement;
  // needs to access tree to find target MB offset by shear
  friend class ShearingBox;

 public:
  explicit Mesh(ParameterInput *pin);
  ~Mesh();

  // data
  RegionSize  mesh_size;      // (physical) size of mesh (physical root level)
  RegionIndcs mesh_indcs;     // indices of cells in mesh (physical root level)
  RegionIndcs mb_indcs;       // indices of cells in MeshBlocks (same for all MeshBlocks)
  BoundaryFlag mesh_bcs[6];   // physical boundary conditions at 6 faces of mesh
  bool strictly_periodic;     // true if all boundaries are periodic

  bool one_d, two_d, three_d; // flags to indicate 1D or 2D or 3D calculations
  bool multi_d;               // flag to indicate 2D and 3D calculations
  bool multilevel;            // true for SMR and AMR
  bool adaptive;              // true only for AMR

  int nmb_rootx1, nmb_rootx2, nmb_rootx3; // # of MeshBlocks at root level in each dir
  int nmb_total;           // total number of MeshBlocks across all levels/ranks
  int nmb_thisrank;        // number of MeshBlocks on this MPI rank (local)
  int nmb_maxperrank;      // max allowed number of MBs per device (memory limit for AMR)

  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // logical level of maximum refinement grid in Mesh

  int nprtcl_thisrank;     // number of particles this rank
  int nprtcl_total;        // total number of particles across all ranks
  std::uint64_t topology_version;  // increments whenever AMR/load-balance changes mesh topology
  int topology_last_change_cycle;
  bool hydro_lat_metadata_valid;
  bool hydro_lat_dt_limited_by_hydro;
  bool hydro_lat_same_level;
  bool hydro_lat_diagnostics;
  int hydro_lat_same_level_max_ratio;
  int hydro_lat_neighbor_limiter_mode; // 0=all, 1=hybrid, 2=face
  int hydro_lat_min_bin_count;
  Real hydro_lat_pin_density_contrast;
  Real hydro_lat_pin_density;
  int hydro_lat_metadata_nmb;
  int hydro_lat_sync_factor_current;
  // True while a problem generator has paused the LAT windows.  Driver::Execute then
  // takes every cycle as one plain step of the global time step, which is what it already
  // does whenever no block can use a factor above 1, so the LAT configuration, the
  // pgen's LAT hooks and the rest of the machinery stay in force and LAT resumes at the
  // next window as soon as the flag is cleared.  A pgen sets and clears it at a
  // synchronized point, identically on every rank (the settle steps of the remap module
  // are the user); remap::LoadAndApplyRemap accepts a mid-run call under LAT only while
  // it is set.
  bool hydro_lat_suspended;
  static constexpr std::uint64_t kInvalidLATVersion = ~std::uint64_t{0};
  // Smallest predicted relative window-time gain for which a LAT block->rank migration
  // is worth its cost (seconds of whole-block transfer against a ~20 s window).  Used
  // both to trigger an attempt (UpdateHydroLATMetadata, against the perfect-balance
  // bound) and to accept its result (MeshRefinement::RedistAndRefineMeshBlocks).
  static constexpr double kHydroLATRebalanceMinRelativeGain = 0.05;
  std::uint64_t hydro_lat_metadata_version;
  std::uint64_t hydro_lat_metadata_topology_version;
  std::uint64_t hydro_lat_lb_topology_version;
  std::uint64_t hydro_lat_lb_metadata_version;
  // Version of hydro_lat_work_eachmb, bumped only when the measured work changed AND
  // the partition it would now choose can beat the current one by the rebalance
  // threshold (see UpdateHydroLATMetadata); the lb_ copy is stamped by every partition
  // attempt.  Separate from hydro_lat_metadata_version because that one invalidates the
  // rank-packed boundary layouts, which a pure weight change must not touch.
  std::uint64_t hydro_lat_work_version;
  std::uint64_t hydro_lat_lb_work_version;
  int hydro_lat_lb_last_attempt_cycle;
  Real hydro_lat_global_hydro_dt;
  Real *hydro_lat_dt_eachmb;
  int *hydro_lat_factor_eachmb;
  // Per-block relative work of ONE step of the block at its LAT factor, 1 = a block
  // with no stiff cell; measured by the physics modules at every synchronized point and
  // gathered with the factors.  The partition cost is (steps per window) x this.
  Real *hydro_lat_work_eachmb;
  // Per-factor bin counts: hydro_lat_bin_count[i] = # of MBs with factor 2^i.
  // Valid when hydro_lat_metadata_valid.  Max 21 levels (factor up to 2^20).
  static constexpr int kMaxLATBinLevels = 21;
  int hydro_lat_bin_count[kMaxLATBinLevels];
  int HydroLATBinCountForFactor(int factor) const;
  bool hydro_lat_gid_reordered;

  // following 3x arrays allocated with length [nmb_total] in BuildTreeFromXXXX()
  float *cost_eachmb;            // cost of each MeshBlock
  int *rank_eachmb;              // rank of each MeshBlock
  LogicalLocation *lloc_eachmb;  // LogicalLocations for each MeshBlock
  // For LAT-reordered restarts, maps current GID to the restart-file GID whose
  // payload stores this block's data. Null for ordinary in-file GID order.
  int *restart_gid_eachmb;

  // following 2x arrays allocated with length [nranks] in BuildTreeFromXXXX()
  int *gids_eachrank;      // starting global ID of MeshBlocks in each rank
  int *nmb_eachrank;       // number of MeshBlocks on each rank
  // following 1x arrays allocated with length [nranks] in AddCoordinatesAndPhysics()
  int *nprtcl_eachrank;    // number of particles on each rank

  Real time, dt, dtold, dt_last_completed, cfl_no;
  int ncycle;
  EventCounters ecounter;

  int nmb_packs_thisrank;                  // number of MBPacks on this rank
  MeshBlockPack* pmb_pack;                 // container for MeshBlocks on this rank
  std::unique_ptr<ProblemGenerator> pgen;  // class containing functions to set ICs
  MeshRefinement *pmr=nullptr;             // mesh refinement data/functions (if needed)

  // functions
  void BuildTreeFromScratch(ParameterInput *pin);
  void BuildTreeFromRestart(ParameterInput *pin, IOWrapper &resfile,
                            bool single_file_per_rank=false);
  // The inputs of the LAT block->rank cost model a partition is balanced for: the factor
  // ladder depth (UpdateHydroLATMetadata).  "none" without LAT.
  static std::string HydroLATCostModel(ParameterInput *pin);
  void PrintMeshDiagnostics();
  void WriteMeshStructure();
  void NewTimeStep(const Real tlim);
  void UpdateHydroLATMetadata(int max_factor);
  void InvalidateHydroLATMetadata();
  bool HydroLATLoadBalanceCurrent() const {
    return hydro_lat_metadata_valid &&
           hydro_lat_metadata_topology_version == topology_version &&
           hydro_lat_lb_topology_version == topology_version &&
           hydro_lat_lb_metadata_version == hydro_lat_metadata_version &&
           hydro_lat_lb_work_version == hydro_lat_work_version;
  }
  void StampHydroLATLoadBalanceVersions() {
    hydro_lat_lb_topology_version = topology_version;
    hydro_lat_lb_metadata_version = hydro_lat_metadata_version;
    hydro_lat_lb_work_version = hydro_lat_work_version;
  }
  int HydroLATFactorForGID(int gid, int max_factor) const;
  void AddCoordinatesAndPhysics(ParameterInput *pinput);
  BoundaryFlag GetBoundaryFlag(const std::string& input_string);
  std::string GetBoundaryString(BoundaryFlag input_flag);

  // comparison function for sorting LogicalLocations based on level
  static bool GreaterLevel(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }

  // accessors
  int FindMeshBlockIndex(int tgid);
  int NumberOfMeshBlockCells() const {
    return (mb_indcs.nx1)*(mb_indcs.nx2)*(mb_indcs.nx3);
  }

  // Public because it launches a device lambda: CUDA rejects extended lambdas inside
  // private or protected member functions.
  void ApplyHydroLATDensityPin();

 private:
  std::unique_ptr<MeshBlockTree> ptree;  // pointer to root node in binary/quad/oct-tree
  void LoadBalance(float *clist, int *rlist, int *slist, int *nlist, int nb,
                   int max_blocks_per_rank=0, int *lat_factor_list=nullptr,
                   int lat_sync_factor=1);
  int ApplyHydroLATLoadBalanceCosts(ParameterInput *pin, LogicalLocation *lloc_list,
                                    float *cost_list, int nb, int *lat_factor_list=nullptr,
                                    int *lat_sync_factor=nullptr,
                                    bool costs_measured=false);
  bool BuildHydroLATGIDMap(ParameterInput *pin, const LogicalLocation *lloc_list, int nb,
                           const int *lat_factor_list, int lat_sync_factor,
                           int *newtoold, bool input_already_reordered=false,
                           const Real *work_list=nullptr) const;
  void ApplyHydroLATGIDMapToArrays(LogicalLocation *lloc_list, float *cost_list,
                                   int *lat_factor_list, int **restart_gid_map,
                                   int nb, const int *newtoold, bool reordered);
};
#endif  // MESH_MESH_HPP_
