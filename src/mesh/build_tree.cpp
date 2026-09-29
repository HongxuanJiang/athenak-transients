//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file build_tree.cpp
//! \brief Functions to build MeshBlock, both for new runs and restarts

#include <algorithm>
#include <iostream>
#include <cinttypes>
#include <limits> // numeric_limits<>
#include <memory> // make_unique<>
#include <sstream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

//----------------------------------------------------------------------------------------
//! \fn void WarnInertRefinementKeys(ParameterInput *pin, bool adaptive)
//! \brief Warn (rank 0) about deck keys that are read only when refinement = adaptive.
//!
//! Every one of these is read inside an `if (adaptive)`, so with refinement = static they
//! are accepted, echoed back into every checkpoint's parameter dump, and never used.  A
//! deck author reading `num_levels = 10` believes refinement is capped at 10 levels (the
//! actual cap is the <refined_regionN> list and max_level = 31), and reading
//! max_nmb_per_rank = 155 believes the LAT cost
//! balancer is free to give a rank up to that many blocks -- it is not, so the LAT
//! partition collapses to the equal-block-count split.  Saying so once at startup is the
//! difference between a knob that does nothing and a knob that silently does nothing.
//! A key a restart read from the checkpoint's parameter dump is not warned about: the run
//! that wrote it warned if its deck set it, and MeshRefinement adds ncycle_check and
//! refinement_interval to every multilevel run's dump.
void WarnInertRefinementKeys(ParameterInput *pin, bool adaptive) {
  if (adaptive || global_variable::my_rank != 0) return;
  struct InertKey { const char *block; const char *name; const char *effect; };
  const InertKey keys[] = {
    {"mesh_refinement", "num_levels",
     "refinement depth is set by the <refined_regionN> blocks instead"},
    {"mesh_refinement", "ncycle_check",
     "no AMR check runs"},
    {"mesh_refinement", "refinement_interval",
     "no AMR check runs"},
    {"mesh_refinement", "max_nmb_per_rank",
     "the per-rank block cap is not applied; the load balancer uses ceil(nmb/nranks)"},
  };
  for (const auto &key : keys) {
    if (!pin->DoesParameterExist(key.block, key.name) ||
        pin->IsFromRestartHeader(key.block, key.name)) {
      continue;
    }
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<" << key.block << ">/" << key.name << " is set but is only read with "
              << "<mesh_refinement>/refinement = adaptive; with static refinement it has "
              << "NO effect (" << key.effect << ")." << std::endl;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void Mesh::BuildTreeFromScratch():
//! Constructs MeshBlockTree, creates MeshBlockPack (containing the physics modules), and
//! divides grid into MeshBlock(s) for new runs (starting from scratch), using parameters
//! read from input file.  Also does initial load balance based on simple cost estimate.

void Mesh::BuildTreeFromScratch(ParameterInput *pin) {
  // calculate the number of MeshBlocks at root level in each dir
  nmb_rootx1 = mesh_indcs.nx1/mb_indcs.nx1;
  nmb_rootx2 = mesh_indcs.nx2/mb_indcs.nx2;
  nmb_rootx3 = mesh_indcs.nx3/mb_indcs.nx3;

  // find maximum number of MeshBlocks at root level in any dir
  int nmbmax = (nmb_rootx1 > nmb_rootx2) ? nmb_rootx1 : nmb_rootx2;
  nmbmax = (nmbmax > nmb_rootx3) ? nmbmax : nmb_rootx3;

  // Find smallest N such that 2^N > max number of MeshBlocks in any dimension (nmbmax)
  // Then N is logical level of root grid.  2^N implemented as left-shift (1<<root_level)
  for (root_level=0; ((1<<root_level) < nmbmax); root_level++) {}
  int current_level = root_level;

  // Construct tree and create root grid
  ptree = std::make_unique<MeshBlockTree>(this);
  ptree->CreateRootGrid();

  // Error check properties of input paraemters for SMR/AMR meshes.
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh_refinement", "num_levels", 1) + root_level - 1;
    if (max_level > 31) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of refinement levels must be smaller than "
                << 31 - root_level + 1 << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    max_level = 31;
  }
  WarnInertRefinementKeys(pin, adaptive);

  // Read <refined_region> blocks and construct tree accordingly
  // These regions can be used with both SMR (in which case they will remain fixed) and
  // AMR (in which case they may be defined, unless the location refinement criteria used)
  if (multilevel) {
    // error check that number of cells in MeshBlock divisible by two
    if (mb_indcs.nx1 % 2 != 0 ||
       (mb_indcs.nx2 % 2 != 0 && multi_d) ||
       (mb_indcs.nx3 % 2 != 0 && three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of cells in MeshBlock must be divisible by 2 "
                << "with SMR or AMR." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // cycle through ParameterInput list and find "refined_region" blocks, extract data
    // and expand MeshBlockTree
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 14, "refined_region") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(it->block_name, "x1min");
        ref_size.x1max = pin->GetReal(it->block_name, "x1max");
        if (multi_d) {
          ref_size.x2min = pin->GetReal(it->block_name, "x2min");
          ref_size.x2max = pin->GetReal(it->block_name, "x2max");
        } else {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (three_d) {
          ref_size.x3min = pin->GetReal(it->block_name, "x3min");
          ref_size.x3max = pin->GetReal(it->block_name, "x3max");
        } else {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
        }
        int phy_ref_lev = pin->GetInteger(it->block_name, "level");
        int log_ref_lev = phy_ref_lev + root_level;
        if (log_ref_lev > current_level) current_level = log_ref_lev;

        // error check parameters in "refinement" blocks
        if (phy_ref_lev < 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl <<"<refined_region> level must be larger than 0 (root level=0)"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (log_ref_lev > max_level) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<refined_region> level exceeds maximum allowed ("
              << max_level << ")" << std::endl << "Reduce/specify 'num_levels' in "
              << "<mesh_refinement> input block if using AMR" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min > ref_size.x1max
            || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid <refined_region> (xmax < xmin in one direction)."
              << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (   ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
            || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
            || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<refined_region> must be fully contained within root mesh"
              << std::endl;
          std::exit(EXIT_FAILURE);
        }

        // note: if following is too slow, it could be replaced with bi-section search.
        // Suppose entire root domain is tiled with MeshBlocks at the desired refinement
        // level. Find range of x1-integer indices of such MeshBlocks that cover the
        // refinement region
        std::int32_t lx1min = 0, lx1max = 0;
        std::int32_t lx2min = 0, lx2max = 0;
        std::int32_t lx3min = 0, lx3max = 0;
        std::int32_t lxmax = nmb_rootx1*(1<<phy_ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          if (LeftEdgeX(lx1min+1,lxmax,mesh_size.x1min,mesh_size.x1max) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          if (LeftEdgeX(lx1max+1,lxmax,mesh_size.x1min,mesh_size.x1max) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;

        // Find range of x2-indices of such MeshBlocks that cover the refinement region
        if (multi_d) { // 2D or 3D
          lxmax = nmb_rootx2*(1<<phy_ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            if (LeftEdgeX(lx2min+1, lxmax, mesh_size.x2min, mesh_size.x2max) >
                ref_size.x2min)
            break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            if (LeftEdgeX(lx2max+1, lxmax, mesh_size.x2min, mesh_size.x2max) >=
                ref_size.x2max)
            break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }

        // Find range of x3-indices of such MeshBlocks that cover the refinement region
        if (three_d) { // 3D
          lxmax = nmb_rootx3*(1<<phy_ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            if (LeftEdgeX(lx3min+1, lxmax, mesh_size.x3min, mesh_size.x3max) >
                ref_size.x3min)
            break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            if (LeftEdgeX(lx3max+1, lxmax, mesh_size.x3min, mesh_size.x3max) >=
                ref_size.x3max)
            break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }

        // Now add nodes to the MeshBlockTree corresponding to these MeshBlocks
        if (one_d) {  // 1D
          for (std::int32_t i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nlloc;
            nlloc.level = log_ref_lev;
            nlloc.lx1 = i;
            nlloc.lx2 = 0;
            nlloc.lx3 = 0;
            int nnew;
            ptree->AddNode(nlloc, nnew);
          }
        }
        if (two_d) {  // 2D
          for (std::int32_t j=lx2min; j<lx2max; j+=2) {
            for (std::int32_t i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nlloc;
              nlloc.level = log_ref_lev;
              nlloc.lx1 = i;
              nlloc.lx2 = j;
              nlloc.lx3 = 0;
              int nnew;
              ptree->AddNode(nlloc, nnew);
            }
          }
        }
        if (three_d) {  // 3D
          for (std::int32_t k=lx3min; k<lx3max; k+=2) {
            for (std::int32_t j=lx2min; j<lx2max; j+=2) {
              for (std::int32_t i=lx1min; i<lx1max; i+=2) {
                LogicalLocation nlloc;
                nlloc.level = log_ref_lev;
                nlloc.lx1 = i;
                nlloc.lx2 = j;
                nlloc.lx3 = k;
                int nnew;
                ptree->AddNode(nlloc, nnew);
              }
            }
          }
        }
      }
    }
  } // if (multilevel)

  if (!adaptive) max_level = current_level;

  // initial mesh hierarchy construction is completed here
  ptree->CountMeshBlocks(nmb_total);

  cost_eachmb = new float[nmb_total];
  rank_eachmb = new int[nmb_total];
  lloc_eachmb = new LogicalLocation[nmb_total];
  gids_eachrank = new int[global_variable::nranks];
  nmb_eachrank = new int[global_variable::nranks];

  // following returns LogicalLocation list sorted by Z-ordering, and total # of MBs
  ptree->CreateZOrderedLLList(lloc_eachmb, nullptr, nmb_total);

#if MPI_PARALLEL_ENABLED
  // check there is at least one MeshBlock per MPI rank
  if (nmb_total < global_variable::nranks) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Fewer MeshBlocks (nmb_total=" << nmb_total << ") than MPI ranks (nranks="
        << global_variable::nranks << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  // Initialize base costs.  A cold start deliberately uses the exact non-LAT layout
  // (unit costs, Z-ordered GIDs, block-count partition) even when LAT is enabled:
  // problem generators seed randomization from GID assignments and normalize fields
  // through layout-sensitive reductions, so a LAT-dependent initial layout gives
  // lat=true and lat=false runs different initial data.  The first synchronized-window
  // rebalance applies the LAT bin ordering afterward as a state-preserving
  // redistribution of this identical initial state.
  for (int i=0; i<nmb_total; i++) {cost_eachmb[i] = 1.0;}
  int lat_projected_nmb_max = 0;
  if (pin->IsLATEnabled() && global_variable::nranks > 1) {
    // Evaluate the LAT bin partition only to reserve rank-local capacity for that
    // later redistribution; the layout below stays LAT-independent.
    std::vector<float> lat_cost(nmb_total, 1.0f);
    std::vector<int> lat_factors(nmb_total, 0);
    std::vector<LogicalLocation> lat_lloc(lloc_eachmb, lloc_eachmb + nmb_total);
    int lat_sync = 1;
    const int lat_cap =
        ApplyHydroLATLoadBalanceCosts(pin, lat_lloc.data(), lat_cost.data(), nmb_total,
                                      lat_factors.data(), &lat_sync);
    std::vector<int> gid_map(nmb_total, 0);
    if (BuildHydroLATGIDMap(pin, lat_lloc.data(), nmb_total, lat_factors.data(),
                            lat_sync, gid_map.data())) {
      std::vector<float> reordered_cost(nmb_total, 1.0f);
      std::vector<int> reordered_factors(nmb_total, 0);
      for (int gid=0; gid<nmb_total; ++gid) {
        reordered_cost[gid] = lat_cost[gid_map[gid]];
        reordered_factors[gid] = lat_factors[gid_map[gid]];
      }
      lat_cost.swap(reordered_cost);
      lat_factors.swap(reordered_factors);
    }
    std::vector<int> lat_rank(nmb_total, 0);
    std::vector<int> lat_gids(global_variable::nranks, 0);
    std::vector<int> lat_nmb(global_variable::nranks, 0);
    LoadBalance(lat_cost.data(), lat_rank.data(), lat_gids.data(), lat_nmb.data(),
                nmb_total, lat_cap, lat_factors.data(), lat_sync);
    for (int r=0; r<global_variable::nranks; ++r) {
      lat_projected_nmb_max = std::max(lat_projected_nmb_max, lat_nmb[r]);
    }
  }
  LoadBalance(cost_eachmb, rank_eachmb, gids_eachrank, nmb_eachrank, nmb_total,
              0, nullptr, 1);

  // create MeshBlockPack for this rank
  int mbp_gids = gids_eachrank[global_variable::my_rank];
  int mbp_gide = mbp_gids + nmb_eachrank[global_variable::my_rank] - 1;
  nmb_thisrank = nmb_eachrank[global_variable::my_rank];

  pmb_pack = new MeshBlockPack(this, mbp_gids, mbp_gide);
  nmb_packs_thisrank = 1;
  pmb_pack->AddMeshBlocks(pin);
  pmb_pack->pmb->SetNeighbors(ptree, rank_eachmb);

  // Fix maximum number of MeshBlocks per rank with AMR
  nmb_maxperrank = nmb_thisrank;
  // Static SMR has no configured AMR memory cap, but LAT rebalancing may move
  // same-level blocks between ranks.  Reserve the largest initial rank count on
  // every rank so a valid pure redistribution cannot exceed a rank-local array
  // capacity (including uneven/restart-preserved partitions).
#if MPI_PARALLEL_ENABLED
  if (multilevel && !adaptive) {
    MPI_Allreduce(MPI_IN_PLACE, &nmb_maxperrank, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
  }
#endif
  if (multilevel && !adaptive) {
    // Reserve room for the LAT bin-cost partition the first synchronized-window
    // rebalance may apply; the initial layout above is deliberately LAT-independent.
    nmb_maxperrank = std::max(nmb_maxperrank, lat_projected_nmb_max);
  }
  if (adaptive) {
    if (pin->DoesParameterExist("mesh_refinement", "max_nmb_per_rank")) {
      nmb_maxperrank = pin->GetReal("mesh_refinement", "max_nmb_per_rank");
      if (nmb_maxperrank < nmb_thisrank) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "On rank=" << global_variable::my_rank << " Root grid requires "
          << "more MeshBlocks (nmb_thisrank=" << nmb_thisrank << ") than specified by "
          << "<mesh_refinement>/max_nmb_per_rank=" << nmb_maxperrank << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "With AMR maximum number of MeshBlocks per rank must be "
        << "specified in input file using <mesh_refinement>/max_nmb_per_rank"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  if (nmb_maxperrank > (1 << (NUM_BITS_LID))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Maximum number of MeshBlocks per rank cannot exceed 2^(NUM_BITS_LID) due to MPI"
      << " tag limits" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  // set initial time/cycle parameters, output diagnostics
  time = pin->GetOrAddReal("time", "start_time", 0.0);
  dt   = std::numeric_limits<float>::max();
  cfl_no = pin->GetReal("time", "cfl_number");
  ncycle = 0;
  topology_last_change_cycle = ncycle;
  if (global_variable::my_rank == 0) {PrintMeshDiagnostics();}

  return;
}

//----------------------------------------------------------------------------------------
//! \fn std::string Mesh::HydroLATCostModel(ParameterInput *pin)
//! \brief The LAT cost model a partition is balanced for, as a comparable token: the
//! factor ladder depth as Driver::Driver reads it.

std::string Mesh::HydroLATCostModel(ParameterInput *pin) {
  if (!pin->IsLATEnabled()) return "none";
  const int levels = pin->DoesParameterExist("time", "lat_levels") ?
      std::max(1, std::min(pin->GetInteger("time", "lat_levels"), 20)) : 1;
  std::string model = "levels" + std::to_string(levels);
  return model;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::BuildTreeFromRestart():
//! Constructs MeshBlockTree, creates MeshBlockPack (containing the physics modules), and
//! divides grid into MeshBlock(s) for restart runs, using parameters and data read from
//! restart file.

void Mesh::BuildTreeFromRestart(ParameterInput *pin, IOWrapper &resfile,
                                                     bool single_file_per_rank) {
  // At this point, the restartfile is already open and the ParameterInput (input file)
  // data has already been read in main(). Thus the file pointer is set to after <par_end>
  // following must be identical to calculation of headeroffset (excluding size of
  // ParameterInput data) in restart.cpp
  IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
    + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  char *headerdata = new char[headersize];

  // the master process reads the header data if single_file_per_rank is false
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    IOWrapperSizeT read_size = resfile.Read_bytes(headerdata, 1, headersize,
                                                  single_file_per_rank);
    if (read_size != headersize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Header size read from restart file is incorrect, "
                << "expected " << headersize << ", got " << read_size << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#if MPI_PARALLEL_ENABLED
  // then broadcast the header data
  if (!single_file_per_rank) {
    int mpi_err = MPI_Bcast(headerdata, headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (mpi_err != MPI_SUCCESS) {
      char error_string[1024];
      int length_of_error_string;
      MPI_Error_string(mpi_err, error_string, &length_of_error_string);
      std::cout << "MPI_Bcast failed with error: " << error_string << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#endif

  // The Mesh constructor has already filled these from <mesh>/<meshblock> in the merged
  // ParameterInput (the deck overrides the restart header for every scalar parameter),
  // and the memcpy below overwrites them with the checkpoint's values.  Keep a copy
  // so an edited grid is REPORTED rather than silently discarded: a deck author who
  // changes nx3 or a MeshBlock size, restarts, and sees the edit echoed back in the next
  // checkpoint's parameter dump has every reason to believe it took effect.
  const RegionSize deck_mesh_size = mesh_size;
  const RegionIndcs deck_mesh_indcs = mesh_indcs;
  const RegionIndcs deck_mb_indcs = mb_indcs;

  // Now copy mesh data read from restart file into Mesh variables. Order of variables
  // set by Write()'s in restart.cpp
  // Note this overwrites size and indices initialized in Mesh constructor.
  IOWrapperSizeT hdos = 0;
  std::memcpy(&nmb_total, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  std::memcpy(&mesh_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  // The root mesh has no coarse representation (Mesh::Mesh), so these carry no state.  A
  // checkpoint written before they were initialized holds heap residue there; drop it
  // instead of copying it into every checkpoint this run writes.
  mesh_indcs.cnx1 = 0;
  mesh_indcs.cnx2 = 0;
  mesh_indcs.cnx3 = 0;
  mesh_indcs.cis = 0;
  mesh_indcs.cie = 0;
  mesh_indcs.cjs = 0;
  mesh_indcs.cje = 0;
  mesh_indcs.cks = 0;
  mesh_indcs.cke = 0;
  std::memcpy(&mb_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  delete [] headerdata;

  // Report every grid key on which the deck and the checkpoint disagree.  This is a
  // WARNING and not a FATAL because a restart deck may legitimately differ (this run's
  // own deck carries keys the checkpoint predates), and because the checkpoint's grid is
  // the only self-consistent one: the payload that follows is indexed by the stored tree.
  if (global_variable::my_rank == 0) {
    auto warn_key = [](const char *key, auto deck, auto file) {
      if (deck == file) return;
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Restart ignores the input file's " << key << " = " << deck
                << "; the checkpoint's value " << file << " is used." << std::endl;
    };
    warn_key("<mesh>/nx1", deck_mesh_indcs.nx1, mesh_indcs.nx1);
    warn_key("<mesh>/nx2", deck_mesh_indcs.nx2, mesh_indcs.nx2);
    warn_key("<mesh>/nx3", deck_mesh_indcs.nx3, mesh_indcs.nx3);
    warn_key("<mesh>/nghost", deck_mesh_indcs.ng, mesh_indcs.ng);
    warn_key("<mesh>/x1min", deck_mesh_size.x1min, mesh_size.x1min);
    warn_key("<mesh>/x1max", deck_mesh_size.x1max, mesh_size.x1max);
    warn_key("<mesh>/x2min", deck_mesh_size.x2min, mesh_size.x2min);
    warn_key("<mesh>/x2max", deck_mesh_size.x2max, mesh_size.x2max);
    warn_key("<mesh>/x3min", deck_mesh_size.x3min, mesh_size.x3min);
    warn_key("<mesh>/x3max", deck_mesh_size.x3max, mesh_size.x3max);
    warn_key("<meshblock>/nx1", deck_mb_indcs.nx1, mb_indcs.nx1);
    warn_key("<meshblock>/nx2", deck_mb_indcs.nx2, mb_indcs.nx2);
    warn_key("<meshblock>/nx3", deck_mb_indcs.nx3, mb_indcs.nx3);
  }

  // calculate the number of MeshBlocks at root level in each dir
  nmb_rootx1 = mesh_indcs.nx1/mb_indcs.nx1;
  nmb_rootx2 = mesh_indcs.nx2/mb_indcs.nx2;
  nmb_rootx3 = mesh_indcs.nx3/mb_indcs.nx3;
  int current_level = root_level;

  // Error check properties of input paraemters for SMR/AMR meshes.
  if (adaptive) {
    max_level = pin->GetOrAddInteger("mesh_refinement", "num_levels", 1) + root_level - 1;
    if (max_level > 31) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Number of refinement levels must be smaller than "
                << 31 - root_level + 1 << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    max_level = 31;
  }
  WarnInertRefinementKeys(pin, adaptive);

  // allocate memory for lists read from restart
  cost_eachmb = new float[nmb_total];
  rank_eachmb = new int[nmb_total];
  lloc_eachmb = new LogicalLocation[nmb_total];
  gids_eachrank = new int[global_variable::nranks];
  nmb_eachrank = new int[global_variable::nranks];

  // allocate idlist buffer and read list of logical locations and cost
  IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(float);
  char *idlist = new char[listsize*nmb_total];
  // only the master process reads the ID list
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (resfile.Read_bytes(idlist,listsize,nmb_total,single_file_per_rank) !=
        static_cast<unsigned int>(nmb_total)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Incorrect number of MeshBlocks in restart file; "
                << "restart file is broken." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the ID list
  if (!single_file_per_rank) {
    MPI_Bcast(idlist, listsize*nmb_total, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  // everyone sets the logical location and cost lists based on bradcasted data
  int os = 0;
  for (int i=0; i<nmb_total; i++) {
    std::memcpy(&(lloc_eachmb[i]), &(idlist[os]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
  }
  for (int i=0; i<nmb_total; i++) {
    std::memcpy(&(cost_eachmb[i]), &(idlist[os]), sizeof(float));
    os += sizeof(float);
    if (lloc_eachmb[i].level > current_level) current_level = lloc_eachmb[i].level;
  }
  delete [] idlist;
  if (!adaptive) max_level = current_level;

  // <refined_regionN> is parsed on the fresh-start path only; on a restart the tree comes
  // entirely from the checkpoint's stored logical locations, so an edited (or newly
  // added) refined region does nothing at all -- while the next checkpoint faithfully
  // echoes the edited deck, which makes the omission invisible.  Check each region
  // against the restored tree and say so.  The test is "is every leaf overlapping this
  // region at least as fine as the region asks for", insensitive to a nested region
  // inside this one; it therefore catches a raised level or a newly added region, and by
  // construction not a lowered one.
  if (global_variable::my_rank == 0) {
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 14, "refined_region") != 0) continue;
      RegionSize ref;
      ref.x1min = pin->GetReal(it->block_name, "x1min");
      ref.x1max = pin->GetReal(it->block_name, "x1max");
      ref.x2min = multi_d ? pin->GetReal(it->block_name, "x2min") : mesh_size.x2min;
      ref.x2max = multi_d ? pin->GetReal(it->block_name, "x2max") : mesh_size.x2max;
      ref.x3min = three_d ? pin->GetReal(it->block_name, "x3min") : mesh_size.x3min;
      ref.x3max = three_d ? pin->GetReal(it->block_name, "x3max") : mesh_size.x3max;
      const int log_ref_lev = pin->GetInteger(it->block_name, "level") + root_level;
      int coarsest = max_level + 1;
      for (int i=0; i<nmb_total; ++i) {
        const LogicalLocation &loc = lloc_eachmb[i];
        const int dlev = loc.level - root_level;
        const std::int32_t n1 = nmb_rootx1*(1<<dlev);
        if (LeftEdgeX(loc.lx1+1, n1, mesh_size.x1min, mesh_size.x1max) <= ref.x1min ||
            LeftEdgeX(loc.lx1,   n1, mesh_size.x1min, mesh_size.x1max) >= ref.x1max) {
          continue;
        }
        if (multi_d) {
          const std::int32_t n2 = nmb_rootx2*(1<<dlev);
          if (LeftEdgeX(loc.lx2+1, n2, mesh_size.x2min, mesh_size.x2max) <= ref.x2min ||
              LeftEdgeX(loc.lx2,   n2, mesh_size.x2min, mesh_size.x2max) >= ref.x2max) {
            continue;
          }
        }
        if (three_d) {
          const std::int32_t n3 = nmb_rootx3*(1<<dlev);
          if (LeftEdgeX(loc.lx3+1, n3, mesh_size.x3min, mesh_size.x3max) <= ref.x3min ||
              LeftEdgeX(loc.lx3,   n3, mesh_size.x3min, mesh_size.x3max) >= ref.x3max) {
            continue;
          }
        }
        if (loc.level < coarsest) coarsest = loc.level;
      }
      if (coarsest <= max_level && coarsest < log_ref_lev) {
        std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Restart ignores <" << it->block_name << ">: it asks for logical "
                  << "level " << log_ref_lev << ", but the restored mesh is only refined "
                  << "to level " << coarsest << " there. <refined_regionN> blocks are "
                  << "read on a fresh start only; the restart tree comes from the "
                  << "checkpoint." << std::endl;
      }
    }
  }

  const bool hydro_lat_restart = pin->IsLATEnabled();
  if (single_file_per_rank && hydro_lat_restart) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "time/lat is not supported with per-rank restart files. "
              << "Use a single global restart file, or restart without LAT before "
              << "repartitioning." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // rebuild the MeshBlockTree
  ptree = std::make_unique<MeshBlockTree>(this);
  ptree->CreateRootGrid();
  for (int i=0; i<nmb_total; i++) {ptree->AddNodeWithoutRefinement(lloc_eachmb[i]);}

  auto restore_restart_gids = [&]() {
    for (int i=0; i<nmb_total; ++i) {
      MeshBlockTree *bt = ptree->FindMeshBlock(lloc_eachmb[i]);
      if (bt == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Tree reconstruction failed for restart gid=" << i
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      bt->gid_ = i;
    }
  };

  // Preserve the restart file's GID order. Restart payloads are stored by raw GID
  // offset, so the logical block attached to each GID must remain unchanged until
  // after the payload has been read and an explicit migration is performed.
  restore_restart_gids();

  // A restart can rebuild the live mesh in ordinary Z-order. restart_gid_eachmb then
  // keeps payload reads tied to the file GID that owns each logical block. AMR always
  // rebuilds; SMR keeps the checkpoint's existing GID order and restores its saved rank
  // boundaries below, except when that order is a LAT ordering and LAT is now off.
  delete [] restart_gid_eachmb;
  restart_gid_eachmb = nullptr;
  // Rebuild Z-ordered GIDs, permute the file-order cost list into the new order, and
  // validate the resulting (new gid) -> (file gid) map. Returns true when that map is a
  // genuine permutation the payload reader has to follow, and false when the checkpoint
  // was already Z-ordered, in which case the map is dropped.
  auto rebuild_zorder_gids = [&]() -> bool {
    delete [] restart_gid_eachmb;
    restart_gid_eachmb = new int[nmb_total];
    {
      int nnb = 0;
      ptree->CreateZOrderedLLList(lloc_eachmb, restart_gid_eachmb, nnb);
      hydro_lat_gid_reordered = false;
      if (nnb != nmb_total) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Tree reconstruction failed. Total number of blocks in "
          << "reconstructed tree=" << nnb << ", number in file=" << nmb_total << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    {
      std::vector<float> cost_file_order(cost_eachmb, cost_eachmb + nmb_total);
      for (int gid=0; gid<nmb_total; ++gid) {
        const int file_gid = restart_gid_eachmb[gid];
        if (file_gid >= 0 && file_gid < nmb_total) {
          cost_eachmb[gid] = cost_file_order[file_gid];
        }
      }
    }
    bool identity_restart_map = true;
    {
      std::vector<int> seen(nmb_total, 0);
      for (int gid=0; gid<nmb_total; ++gid) {
        const int file_gid = restart_gid_eachmb[gid];
        if (file_gid < 0 || file_gid >= nmb_total || seen[file_gid] != 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "Restart GID map is not a valid permutation at gid=" << gid
                    << ", file gid=" << file_gid << std::endl;
          std::exit(EXIT_FAILURE);
        }
        seen[file_gid] = 1;
        if (file_gid != gid) identity_restart_map = false;
      }
    }
    if (identity_restart_map) {
      delete [] restart_gid_eachmb;
      restart_gid_eachmb = nullptr;
      return false;
    }
    return true;
  };

  // Set when a static-SMR restart discards a checkpoint's LAT GID order.  The saved rank
  // boundaries below partition that discarded order, so they must not be reused.
  bool smr_zorder_restore = false;
  // Set when the checkpoint's GID order and rank boundaries are both kept, so every rank
  // owns exactly the blocks it owned in the writing run.
  bool restart_partition_verbatim = false;
  if (adaptive) {
    if (rebuild_zorder_gids() && single_file_per_rank) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Restart file uses non-Z-order GIDs, but per-rank restart files "
                << "cannot be remapped safely." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    const bool have_smr_order_metadata =
        pin->DoesParameterExist("time", "smr_restart_partition_version") &&
        pin->GetInteger("time", "smr_restart_partition_version") == 1 &&
        pin->DoesParameterExist("time", "smr_restart_gid_reordered");
    hydro_lat_gid_reordered = have_smr_order_metadata ?
        pin->GetBoolean("time", "smr_restart_gid_reordered") :
        (hydro_lat_restart &&
         pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true));
    // A LAT checkpoint's GID order interleaves the timestep bins so every rank owns a
    // slice of every bin -- which is what keeps all ranks busy while the bins execute
    // sequentially, but it deliberately trades away spatial locality: blocks adjacent in
    // space land far apart in GID and therefore on different ranks.  With LAT switched
    // off that trade has no upside left and locality is the only thing the partition can
    // still buy, so rebuild the ordinary Z-order here and repartition below.  Per-rank
    // restart files cannot be remapped, so they keep the checkpoint order.
    if (hydro_lat_gid_reordered && !hydro_lat_restart) {
      if (single_file_per_rank) {
        if (global_variable::my_rank == 0) {
          std::cout << "Mesh: SMR restart kept the checkpoint's LAT GID order; per-rank "
                    << "restart files cannot be remapped to Z-order." << std::endl;
        }
      } else {
        (void) rebuild_zorder_gids();
        smr_zorder_restore = true;
      }
    }
  }

#ifdef MPI_PARALLEL_ENABLED
  // check there is at least one MeshBlock per MPI rank
  if (!single_file_per_rank) {
    if (nmb_total < global_variable::nranks) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
        << __LINE__ << std::endl
        << "Fewer MeshBlocks (nmb_total=" << nmb_total << ") than MPI ranks (nranks="
        << global_variable::nranks << ")" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  if (!adaptive && smr_zorder_restore) {
    // The saved rank boundaries partition the LAT order just discarded, and the saved
    // costs are LAT costs (proportional to how often each block is updated).  Without
    // LAT every block is updated every cycle, so the right partition is exactly the
    // cold-start non-LAT one: unit costs over the freshly rebuilt Z-order.
    for (int i=0; i<nmb_total; i++) {cost_eachmb[i] = 1.0;}
    LoadBalance(cost_eachmb, rank_eachmb, gids_eachrank, nmb_eachrank, nmb_total,
                0, nullptr, 1);
    if (global_variable::my_rank == 0) {
      std::cout << "Mesh: SMR restart rebuilt Z-ordered GIDs and repartitioned for "
                << "non-LAT execution." << std::endl;
    }
  } else if (!adaptive) {
    // SMR topology and checkpoint GID order are fixed. Restore saved rank boundaries
    // when available; otherwise split the existing order evenly. Neither path reruns
    // LAT cost estimation or load balancing.
    std::vector<int> restart_counts;
    bool have_saved_partition = false;
    const bool have_partition_metadata =
        pin->DoesParameterExist("time", "smr_restart_partition_version") &&
        pin->GetInteger("time", "smr_restart_partition_version") == 1 &&
        pin->DoesParameterExist("time", "smr_restart_nranks") &&
        pin->DoesParameterExist("time", "smr_restart_nmb_eachrank");
    if (have_partition_metadata &&
        pin->GetInteger("time", "smr_restart_nranks") == global_variable::nranks) {
      std::istringstream counts(pin->GetString("time", "smr_restart_nmb_eachrank"));
      long long total = 0;
      bool valid = true;
      restart_counts.reserve(global_variable::nranks);
      for (int rank=0; rank<global_variable::nranks; ++rank) {
        int count = 0;
        if (!(counts >> count) || count <= 0 || count > (1 << NUM_BITS_LID)) {
          valid = false;
          break;
        }
        restart_counts.push_back(count);
        total += count;
        if (rank + 1 < global_variable::nranks) {
          char separator = '\0';
          if (!(counts >> separator) || separator != ',') {
            valid = false;
            break;
          }
        }
      }
      counts >> std::ws;
      valid = valid && counts.eof() &&
              static_cast<int>(restart_counts.size()) == global_variable::nranks &&
              total == nmb_total;
      have_saved_partition = valid;
    }
    restart_partition_verbatim = have_saved_partition;
    if (!have_saved_partition) {
      restart_counts.assign(global_variable::nranks,
                            nmb_total/global_variable::nranks);
      const int remainder = nmb_total%global_variable::nranks;
      for (int rank=global_variable::nranks - remainder;
           rank<global_variable::nranks; ++rank) {
        ++restart_counts[rank];
      }
    }
    for (int rank=0; rank<global_variable::nranks; ++rank) {
      if (restart_counts[rank] > (1 << NUM_BITS_LID)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "SMR restart rank partition exceeds MPI tag local-id capacity: rank="
                  << rank << ", MeshBlocks=" << restart_counts[rank]
                  << ", limit=" << (1 << NUM_BITS_LID) << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    int gid = 0;
    for (int rank=0; rank<global_variable::nranks; ++rank) {
      const int count = restart_counts[rank];
      gids_eachrank[rank] = gid;
      nmb_eachrank[rank] = count;
      for (int n=0; n<count; ++n) rank_eachmb[gid + n] = rank;
      gid += count;
    }
    if (global_variable::my_rank == 0) {
      if (have_saved_partition) {
        std::cout << "Mesh: SMR restart preserved checkpoint rank partition."
                  << std::endl;
      } else {
        std::cout << "Mesh: SMR restart preserved checkpoint GID order with an even "
                  << "contiguous rank partition." << std::endl;
      }
    }
  } else {
    std::vector<int> lat_factor_eachmb(nmb_total, 0);
    int lat_sync_factor = 1;
    // cost_eachmb holds the checkpoint's own window costs, already permuted into this
    // run's GID order above; say so, so the estimator keeps them.
    int max_blocks_per_rank =
        ApplyHydroLATLoadBalanceCosts(pin, lloc_eachmb, cost_eachmb, nmb_total,
                                      lat_factor_eachmb.data(), &lat_sync_factor, true);
    // One cap only, the allocation bound mesh_refinement/max_nmb_per_rank, exactly as
    // ApplyHydroLATLoadBalanceCosts returns it and as every later AMR partition uses it.
    // A separate, lower restart cap (avg + max(avg/8, 16), to keep the initial pack near
    // the average) is not a memory bound at all: nmb_maxperrank is set from
    // max_nmb_per_rank a few lines below, so the run is already budgeted for that many
    // blocks on any rank at any later cycle, and the pack grows to it the first time AMR
    // asks.  What the lower cap did do was pin the partition: with 708 MeshBlocks on 10
    // ranks it was 87, one block above 708/9, so the bin-aware partitioner had almost no
    // freedom left and returned 87/33/83/80/87/76/87/75/18/82 -- maximum normalised cost
    // 2.95 against the writing run's 1.06, and LAT per-bin utilisation as low as 34%,
    // until the next AMR pass repartitioned.
    LoadBalance(cost_eachmb, rank_eachmb, gids_eachrank, nmb_eachrank, nmb_total,
                max_blocks_per_rank, lat_factor_eachmb.data(), lat_sync_factor);
  }

  // create MeshBlockPack for this rank
  int mbp_gids = gids_eachrank[global_variable::my_rank];
  int mbp_gide = mbp_gids + nmb_eachrank[global_variable::my_rank] - 1;
  nmb_thisrank = nmb_eachrank[global_variable::my_rank];

  pmb_pack = new MeshBlockPack(this, mbp_gids, mbp_gide);
  pmb_pack->AddMeshBlocks(pin);
  pmb_pack->pmb->SetNeighbors(ptree, rank_eachmb);

  // Fix maximum number of MeshBlocks per rank with AMR
  nmb_maxperrank = nmb_thisrank;
  // See the corresponding fresh-run path above: static-SMR LAT redistribution
  // needs one capacity bound valid on every rank, including restart partitions.
#if MPI_PARALLEL_ENABLED
  if (multilevel && !adaptive) {
    MPI_Allreduce(MPI_IN_PLACE, &nmb_maxperrank, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
  }
#endif
  if (adaptive) {
    if (pin->DoesParameterExist("mesh_refinement", "max_nmb_per_rank")) {
      nmb_maxperrank = pin->GetReal("mesh_refinement", "max_nmb_per_rank");
      if (nmb_maxperrank < nmb_thisrank) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "On rank=" << global_variable::my_rank << " Root grid requires "
          << "more MeshBlocks (nmb_thisrank=" << nmb_thisrank << ") than specified by "
          << "<mesh_refinement>/max_nmb_per_rank=" << nmb_maxperrank << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "With AMR maximum number of MeshBlocks per rank must be "
        << "specified in input file using <mesh_refinement>/max_nmb_per_rank"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set remaining parameters, output diagnostics
  cfl_no = pin->GetReal("time", "cfl_number");
  // The LAT rebalance (Driver::Execute) is armed by a topology change since its last
  // attempt.  A restart that kept the writing run's partition under the same LAT cost
  // model continues that run's two anchors (restart.cpp), so the trigger is armed as it
  // was in the writing run.  Any other restart counts its partition as a topology change
  // made here, which the rebalance answers once the mesh has been stable.  A checkpoint
  // written on the cycle whose window-end pass rebalances resumes with the trigger armed
  // and rebalances one window later than the writing run (Driver::Initialize's pass does
  // not rebalance).
  topology_last_change_cycle = ncycle;
  if (restart_partition_verbatim &&
      pin->DoesParameterExist("time", "smr_restart_topology_last_change_cycle") &&
      pin->DoesParameterExist("time", "smr_restart_lat_lb_last_attempt_cycle") &&
      pin->DoesParameterExist("time", "smr_restart_lat_cost_model") &&
      pin->GetString("time", "smr_restart_lat_cost_model") == HydroLATCostModel(pin)) {
    topology_last_change_cycle =
        pin->GetInteger("time", "smr_restart_topology_last_change_cycle");
    hydro_lat_lb_last_attempt_cycle =
        pin->GetInteger("time", "smr_restart_lat_lb_last_attempt_cycle");
  }
  if (global_variable::my_rank == 0) {PrintMeshDiagnostics();}
}
