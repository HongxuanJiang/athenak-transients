//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of constructor and functions in Mesh class

#include <algorithm>
#include <array>  // std::array (sink LAT pin regions)
#include <cinttypes>
#include <cmath>  // std::abs
#include <iostream>
#include <limits>
#include <cstdio> // fclose
#include <cstdlib> // exit
#include <string> // string
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "refinement_criteria.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "radiation/radiation.hpp"
#include "sink_particles/sink_particles.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"
#include "outputs/io_wrapper.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! Mesh constructor:
//! initializes some mesh variables using parameters in input file.
//! The MeshBlockPack, MeshRefinement, and ShearingBox objects are constructed in
//! BuildTreeFromScratch() or BuildTreeFromRestart()
//! The MeshBlockTree and ProblemGenerator objects are constructed in main().
//! This is so that they can store a pointer to the Mesh which can be reliably referenced
//! only after the Mesh constructor has finished.

Mesh::Mesh(ParameterInput *pin) :
  one_d(false),
  two_d(false),
  three_d(false),
  multi_d(false),
  strictly_periodic(true),
  nmb_packs_thisrank(1),
  nprtcl_thisrank(0),
  nprtcl_total(0),
  topology_version(0),
  topology_last_change_cycle(0),
  hydro_lat_metadata_valid(false),
  hydro_lat_dt_limited_by_hydro(false),
  hydro_lat_same_level(false),
  hydro_lat_diagnostics(false),
  hydro_lat_same_level_max_ratio(1),
  hydro_lat_neighbor_limiter_mode(0),
  hydro_lat_min_bin_count(-1),
  hydro_lat_pin_density_contrast(0.0),
  hydro_lat_pin_density(0.0),
  hydro_lat_metadata_nmb(0),
  hydro_lat_sync_factor_current(1),
  hydro_lat_suspended(false),
  hydro_lat_metadata_version(0),
  hydro_lat_metadata_topology_version(0),
  hydro_lat_lb_topology_version(kInvalidLATVersion),
  hydro_lat_lb_metadata_version(kInvalidLATVersion),
  hydro_lat_work_version(0),
  hydro_lat_lb_work_version(kInvalidLATVersion),
  hydro_lat_lb_last_attempt_cycle(-1),
  hydro_lat_global_hydro_dt(std::numeric_limits<Real>::max()),
  hydro_lat_dt_eachmb(nullptr),
  hydro_lat_factor_eachmb(nullptr),
  hydro_lat_work_eachmb(nullptr),
  hydro_lat_bin_count{},
  hydro_lat_gid_reordered(false),
  restart_gid_eachmb(nullptr),
  dtold(0.),
  dt_last_completed(0.),
  ncycle(0) {
  // The separate LAT partition cap was removed: it could not be honoured once AMR filled
  // the blocks between it and the allocation bound, and the run aborted instead.  A deck
  // key the run would not read is an error, not a warning; the same key arriving in a
  // restart header is history and is dropped instead, so the production checkpoints that
  // carry it still restart (ParameterInput::RetireDeadParameter).
  if (pin->RetireDeadParameter("time", "hydro_lat_max_nmb_per_rank")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<time>/hydro_lat_max_nmb_per_rank is read by no code path; the LAT "
              << "partition cap is <mesh_refinement>/max_nmb_per_rank." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set physical size and number of cells in mesh (root level)
  mesh_size.x1min = pin->GetReal("mesh", "x1min");
  mesh_size.x1max = pin->GetReal("mesh", "x1max");
  mesh_size.x2min = pin->GetReal("mesh", "x2min");
  mesh_size.x2max = pin->GetReal("mesh", "x2max");
  mesh_size.x3min = pin->GetReal("mesh", "x3min");
  mesh_size.x3max = pin->GetReal("mesh", "x3max");

  mesh_indcs.ng  = pin->GetOrAddInteger("mesh", "nghost", 2);
  mesh_indcs.nx1 = pin->GetInteger("mesh", "nx1");
  mesh_indcs.nx2 = pin->GetInteger("mesh", "nx2");
  mesh_indcs.nx3 = pin->GetInteger("mesh", "nx3");

  // define some useful flags that indicate 1D/2D/3D calculations
  if (mesh_indcs.nx3 > 1) {
    three_d = true;
    multi_d = true;
  } else if (mesh_indcs.nx2 > 1) {
    two_d = true;
    multi_d = true;
  } else {
    one_d = true;
  }

  const bool hydro_lat_enabled = pin->IsLATEnabled();
  if (hydro_lat_enabled) {
    hydro_lat_same_level =
        pin->GetOrAddBoolean("time", "lat_same_level", false);
    hydro_lat_diagnostics =
        pin->GetOrAddBoolean("time", "lat_diagnostics", false);
    hydro_lat_same_level_max_ratio =
        pin->GetOrAddInteger("time", "lat_same_level_max_ratio", 1);
    hydro_lat_min_bin_count =
        pin->GetOrAddInteger("time", "hydro_lat_min_bin_count", -1);
    hydro_lat_pin_density_contrast =
        pin->GetOrAddReal("time", "lat_pin_density_contrast", 0.0);
    hydro_lat_pin_density = pin->GetOrAddReal("time", "lat_pin_density", 0.0);
    if (!std::isfinite(hydro_lat_pin_density_contrast) ||
        !std::isfinite(hydro_lat_pin_density)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "time/lat_pin_density_contrast and time/lat_pin_density must be finite."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (hydro_lat_min_bin_count < 0 && global_variable::my_rank == 0 &&
        global_variable::nranks > 1) {
      // The default floor scales with the rank count (see UpdateHydroLATMetadata), so it
      // is the one place where the number of ranks reaches the per-block time step.  A
      // restart of identical state on a different rank count is still conservative and
      // still valid, but it is a different trajectory; say so where an operator will see
      // it rather than leaving it to be discovered by a failed bit-for-bit comparison.
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "time/hydro_lat_min_bin_count is at its default, so the minimum LAT "
                << "bin population is 4*nranks = " << 4*global_variable::nranks
                << ". LAT bin assignment therefore depends on the rank count; set the "
                << "key explicitly to reproduce a run on a different number of ranks."
                << std::endl;
    }

    if (hydro_lat_same_level_max_ratio < 1 || hydro_lat_same_level_max_ratio > 8 ||
        (hydro_lat_same_level_max_ratio & (hydro_lat_same_level_max_ratio - 1)) != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "time/lat_same_level_max_ratio must be one of 1, 2, 4, or 8."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (hydro_lat_min_bin_count < -1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "time/hydro_lat_min_bin_count must be >= 0, or -1 for the default."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string lat_neighbor_limiter =
        pin->GetOrAddString("time", "lat_neighbor_limiter", "all");
    if (lat_neighbor_limiter == "all") {
      hydro_lat_neighbor_limiter_mode = 0;
    } else if (lat_neighbor_limiter == "hybrid") {
      hydro_lat_neighbor_limiter_mode = 1;
    } else if (lat_neighbor_limiter == "face") {
      hydro_lat_neighbor_limiter_mode = 2;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "time/lat_neighbor_limiter must be 'all', 'hybrid', or 'face'."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Set BC flags for ix1/ox1 boundaries and error check
  mesh_bcs[BoundaryFace::inner_x1] = GetBoundaryFlag(pin->GetString("mesh", "ix1_bc"));
  mesh_bcs[BoundaryFace::outer_x1] = GetBoundaryFlag(pin->GetString("mesh", "ox1_bc"));
  if ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic ||
       mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic) &&
       mesh_bcs[BoundaryFace::inner_x1] != mesh_bcs[BoundaryFace::outer_x1]) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Both inner and outer x1 bcs must be periodic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    strictly_periodic = false;
  }

  // Error checks if one of x1 boundaries set to shear_periodic.
  if (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic &&
      mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic) {
    if (one_d) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries require 2D or 3D" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!(pin->DoesBlockExist("shearing_box"))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries set but no <shearing_box>"
                << " block in input file" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else if ((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic &&
              mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::shear_periodic) ||
             (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic &&
              mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "In shearing box, both x1 bcs must be shear_periodic"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set BC flags for ix2/ox2 boundaries and error check
  if (multi_d) {
    mesh_bcs[BoundaryFace::inner_x2] = GetBoundaryFlag(pin->GetString("mesh", "ix2_bc"));
    mesh_bcs[BoundaryFace::outer_x2] = GetBoundaryFlag(pin->GetString("mesh", "ox2_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x2] != mesh_bcs[BoundaryFace::outer_x2]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x2 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
      strictly_periodic = false;
    }
    if (mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::shear_periodic ||
        mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::shear_periodic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries cannot be applied in x2"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix2/ox2 BC flags set to undef for 1D problems
    mesh_bcs[BoundaryFace::inner_x2] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x2] = BoundaryFlag::undef;
  }

  // Set BC flags for ix3/ox3 boundaries and error check
  if (three_d) {
    mesh_bcs[BoundaryFace::inner_x3] = GetBoundaryFlag(pin->GetString("mesh", "ix3_bc"));
    mesh_bcs[BoundaryFace::outer_x3] = GetBoundaryFlag(pin->GetString("mesh", "ox3_bc"));
    if ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic ||
         mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic) &&
         mesh_bcs[BoundaryFace::inner_x3] != mesh_bcs[BoundaryFace::outer_x3]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Both inner and outer x3 bcs must be periodic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
      strictly_periodic = false;
    }
    if (mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::shear_periodic ||
        mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::shear_periodic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Shear Periodic Boundaries cannot be applied in x3"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // ix3/ox3 BC flags set to undef for 1D or 2D problems
    mesh_bcs[BoundaryFace::inner_x3] = BoundaryFlag::undef;
    mesh_bcs[BoundaryFace::outer_x3] = BoundaryFlag::undef;
  }

  // set boolean flags indicating type of refinement (if any), and whether mesh is
  // periodic, depending on input strings
  adaptive = (pin->GetOrAddString("mesh_refinement","refinement","none") == "adaptive")
    ?  true : false;
  multilevel = (adaptive || pin->GetString("mesh_refinement","refinement") == "static")
    ?  true : false;

  // FIXME: The shearing box is not currently compatible with SMR/AMR
  if (multilevel && pin->DoesBlockExist("shearing_box")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Shearing box is not currently compatible with mesh refinement"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check requested number of grid cells for entire root domain
  if ( mesh_indcs.nx1 < 4 ||
      (mesh_indcs.nx2 < 4 && multi_d) ||
      (mesh_indcs.nx3 < 4 && three_d) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 < 1 || mesh_indcs.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In <mesh> block nx2 and nx3 must both be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mesh_indcs.nx2 == 1 && mesh_indcs.nx3 > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "In <mesh> block in input file: nx2=1, nx3=" << mesh_indcs.nx3
        << ", but 2D problems in x1-x3 plane not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // error check number of ghost zones
  if (mesh_indcs.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "More than 2 ghost zones required, but nghost=" <<mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((multilevel) && (mesh_indcs.ng % 2 != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of ghost zones must be divisible by two for SMR/AMR calculations, "
      << "but nghost=" << mesh_indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // passed error checks, compute grid spacing in (virtual) mesh grid
  mesh_size.dx1 = (mesh_size.x1max-mesh_size.x1min)/static_cast<Real>(mesh_indcs.nx1);
  mesh_size.dx2 = (mesh_size.x2max-mesh_size.x2min)/static_cast<Real>(mesh_indcs.nx2);
  mesh_size.dx3 = (mesh_size.x3max-mesh_size.x3min)/static_cast<Real>(mesh_indcs.nx3);

  // Read # of cells in MeshBlock from input parameters, error check
  mb_indcs.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_indcs.nx1);
  if (multi_d) {
    mb_indcs.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_indcs.nx2);
  } else {
    mb_indcs.nx2 = mesh_indcs.nx2;
  }
  if (three_d) {
    mb_indcs.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_indcs.nx3);
  } else {
    mb_indcs.nx3 = mesh_indcs.nx3;
  }

  // error check consistency of the block and mesh
  if (mesh_indcs.nx1 % mb_indcs.nx1 != 0 ||
      mesh_indcs.nx2 % mb_indcs.nx2 != 0 ||
      mesh_indcs.nx3 % mb_indcs.nx3 != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh must be evenly divisible by MeshBlocks" << std::endl
              << "Check Mesh and MeshBlock dimensions in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((mb_indcs.nx1 < 4) ||
      (mb_indcs.nx2 < 4 && multi_d) ||
      (mb_indcs.nx3 < 4 && three_d) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MeshBlock must be >= 4 cells in each active dimension" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ( (multilevel) &&
      ((mb_indcs.nx1 %2 != 0) ||
       (mb_indcs.nx2 %2 != 0 && multi_d) ||
       (mb_indcs.nx3 %2 != 0 && three_d)) ) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
      << "Number of cells in MeshBlock must be divisible by two in each dimension for "
      << "SMR/AMR calculations." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // initialize indices for Mesh cells, MeshBlock cells, and MeshBlock coarse cells The
  // root mesh has no coarse representation -- only a MeshBlock does -- so the coarse
  // members of mesh_indcs have no meaning.  They are still SET, and set here, because
  // RestartOutput::WriteOutputFile writes the whole RegionIndcs POD into every checkpoint
  // header (outputs/restart.cpp, STEP 1); left indeterminate they leak whatever the heap
  // held under the Mesh object into the file, which is neither reproducible nor
  // rank-invariant and breaks any bitwise comparison of two checkpoints of the same
  // state.
  mesh_indcs.cnx1 = 0;
  mesh_indcs.cnx2 = 0;
  mesh_indcs.cnx3 = 0;
  mesh_indcs.cis = 0;
  mesh_indcs.cie = 0;
  mesh_indcs.cjs = 0;
  mesh_indcs.cje = 0;
  mesh_indcs.cks = 0;
  mesh_indcs.cke = 0;

  mb_indcs.ng  = mesh_indcs.ng;
  mb_indcs.cnx1 = mb_indcs.nx1/2;
  mb_indcs.cnx2 = std::max(1,(mb_indcs.nx2/2));
  mb_indcs.cnx3 = std::max(1,(mb_indcs.nx3/2));

  mesh_indcs.is = mesh_indcs.ng;
  mb_indcs.is   = mb_indcs.ng;
  mb_indcs.cis  = mb_indcs.ng;

  mesh_indcs.ie = mesh_indcs.is + mesh_indcs.nx1 - 1;
  mb_indcs.ie   = mb_indcs.is + mb_indcs.nx1 - 1;
  mb_indcs.cie  = mb_indcs.cis + mb_indcs.cnx1 - 1;

  if (multi_d) {
    mesh_indcs.js = mesh_indcs.ng;
    mb_indcs.js   = mb_indcs.ng;
    mb_indcs.cjs  = mb_indcs.ng;

    mesh_indcs.je = mesh_indcs.js + mesh_indcs.nx2 - 1;
    mb_indcs.je   = mb_indcs.js + mb_indcs.nx2 - 1;
    mb_indcs.cje  = mb_indcs.cjs + mb_indcs.cnx2 - 1;
  } else {
    mesh_indcs.js = 0;
    mb_indcs.js   = 0;
    mb_indcs.cjs  = 0;

    mesh_indcs.je = 0;
    mb_indcs.je   = 0;
    mb_indcs.cje  = 0;
  }

  if (three_d) {
    mesh_indcs.ks = mesh_indcs.ng;
    mb_indcs.ks   = mb_indcs.ng;
    mb_indcs.cks  = mb_indcs.ng;

    mesh_indcs.ke = mesh_indcs.ks + mesh_indcs.nx3 - 1;
    mb_indcs.ke   = mb_indcs.ks + mb_indcs.nx3 - 1;
    mb_indcs.cke  = mb_indcs.cks + mb_indcs.cnx3 - 1;
  } else {
    mesh_indcs.ks = 0;
    mb_indcs.ks   = 0;
    mb_indcs.cks  = 0;

    mesh_indcs.ke = 0;
    mb_indcs.ke   = 0;
    mb_indcs.cke  = 0;
  }
}

int Mesh::FindMeshBlockIndex(int tgid) {
  for (int m = 0; m < pmb_pack->nmb_thispack; ++m) {
    if (pmb_pack->pmb->mb_gid.h_view(m) == tgid) return m;
  }
  return -1;
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
  if (pmb_pack->ppart != nullptr) {delete [] nprtcl_eachrank;}
  if (multilevel) {
    delete pmr;
  }
  delete pmb_pack;
  delete [] nmb_eachrank;
  delete [] gids_eachrank;
  delete [] lloc_eachmb;
  delete [] rank_eachmb;
  delete [] cost_eachmb;
  delete [] restart_gid_eachmb;
  delete [] hydro_lat_dt_eachmb;
  delete [] hydro_lat_factor_eachmb;
  delete [] hydro_lat_work_eachmb;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::PrintMeshDiagnostics()
//  \brief prints information about mesh structure, always called at start of every
//  calculation at end of BuildTree

void Mesh::PrintMeshDiagnostics() {
  std::cout << std::endl;
  std::cout <<"Root grid = "<< nmb_rootx1 <<" x "<< nmb_rootx2 <<" x "<< nmb_rootx3
            <<" MeshBlocks"<< std::endl;
  std::cout <<"Total number of MeshBlocks = " << nmb_total << std::endl;
  std::cout <<"Number of logical  levels of refinement = "<< max_level
            <<" (" << (max_level + 1) << " levels total)" << std::endl;
  std::cout <<"Number of physical levels of refinement = "<< (max_level - root_level)
            <<" (" << (max_level - root_level + 1) << " levels total)" << std::endl;

  // if more than one physical level: compute/output # of blocks and cost per level
  if ((max_level - root_level) > 1) {
    int nplevels = max_level - root_level + 1;
    std::vector<int> nb_per_plevel(nplevels, 0);
    std::vector<float> cost_per_plevel(nplevels, 0.0);
    for (int i=0; i<nmb_total; i++) {
      nb_per_plevel[(lloc_eachmb[i].level - root_level)]++;
      cost_per_plevel[(lloc_eachmb[i].level - root_level)] += cost_eachmb[i];
    }
    for (int i=root_level; i<=max_level; i++) {
      if (nb_per_plevel[i-root_level] != 0) {
        std::cout << "  Physical level = " << i-root_level << " (logical level = " << i
                  << "): " << nb_per_plevel[i-root_level] << " MeshBlocks, cost = "
                  << cost_per_plevel[i-root_level] <<  std::endl;
      }
    }
  }

  std::cout << "Number of parallel ranks = " << global_variable::nranks << std::endl;
  // if more than one rank: compute/output # of blocks and cost per rank
  if (global_variable::nranks > 1) {
    int nb_per_rank[global_variable::nranks];    // NOLINT(runtime/arrays)
    int cost_per_rank[global_variable::nranks];  // NOLINT(runtime/arrays)
    for (int i=0; i<global_variable::nranks; ++i) {
      nb_per_rank[i] = 0;
      cost_per_rank[i] = 0;
    }
    for (int i=0; i<nmb_total; i++) {
      nb_per_rank[rank_eachmb[i]]++;
      cost_per_rank[rank_eachmb[i]] += cost_eachmb[i];
    }
    int mincost = std::numeric_limits<int>::max();
    int maxcost = 0, totalcost = 0;
    for (int i=0; i<global_variable::nranks; ++i) {
      std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
                << cost_per_rank[i] << std::endl;
      mincost = std::min(mincost,cost_per_rank[i]);
      maxcost = std::max(maxcost,cost_per_rank[i]);
      totalcost += cost_per_rank[i];
    }

    // output normalized costs per rank
    std::cout << "Load Balancing:" << std::endl;
    std::cout << "  Maximum normalized cost = "
      << static_cast<float>(maxcost)/static_cast<float>(mincost) << ", Average = "
      << static_cast<float>(totalcost)/static_cast<float>(global_variable::nranks*mincost)
      << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::WriteMeshStructure(int ndim)
//  \brief writes file containing MeshBlock positions and sizes that can be used to create
//  plots using 'plot_mesh.py' script.  Only works for 2D/3D data.  Called from main if
//  '-m' option is given on command line.

void Mesh::WriteMeshStructure() {
  if (one_d) {
    std::cout << "WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Mesh only 1D, so no 'mesh_structure.dat' file produced" << std::endl;
    return;
  }

  FILE *fp = nullptr;
  if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Cannot open 'mesh_structure.dat' file for output" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int i=root_level; i<=max_level; i++) {
    for (int j=0; j<nmb_total; j++) {
      if (lloc_eachmb[j].level == i) {
        MeshBlock block(this->pmb_pack, j, 1);
        std::int32_t &lx1 = lloc_eachmb[j].lx1;
        std::int32_t &lx2 = lloc_eachmb[j].lx2;
        std::int32_t &lx3 = lloc_eachmb[j].lx3;
        std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, rank_eachmb[j],
                     cost_eachmb[j]);
        std::fprintf(
            fp,"#  Logical level %d, location = (%" PRId32 " %" PRId32 " %" PRId32")\n",
            lloc_eachmb[j].level, lx1, lx2, lx3);
        if (two_d) { // 2D
          Real &x1min = block.mb_size.h_view(0).x1min;
          Real &x1max = block.mb_size.h_view(0).x1max;
          Real &x2min = block.mb_size.h_view(0).x2min;
          Real &x2max = block.mb_size.h_view(0).x2max;
          std::fprintf(fp,"%g %g\n", x1min, x2min);
          std::fprintf(fp,"%g %g\n", x1max, x2min);
          std::fprintf(fp,"%g %g\n", x1max, x2max);
          std::fprintf(fp,"%g %g\n", x1min, x2max);
          std::fprintf(fp,"%g %g\n", x1min, x2min);
          std::fprintf(fp,"\n\n");
        }
        if (three_d) { // 3D
          Real &x1min = block.mb_size.h_view(0).x1min;
          Real &x1max = block.mb_size.h_view(0).x1max;
          Real &x2min = block.mb_size.h_view(0).x2min;
          Real &x2max = block.mb_size.h_view(0).x2max;
          Real &x3min = block.mb_size.h_view(0).x3min;
          Real &x3max = block.mb_size.h_view(0).x3max;
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1max, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3min);
          std::fprintf(fp,"%g %g %g\n", x1min, x2max, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3max);
          std::fprintf(fp,"%g %g %g\n", x1min, x2min, x3min);
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }
  std::fclose(fp);
  std::cout << "See the 'mesh_structure.dat' file for MeshBlock data" << std::endl;
  std::cout << "Use 'plot_mesh.py' script to visualize data" << std::endl << std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryFlag(std::string input_string)
//  \brief Parses input string to return scoped enumerator flag specifying boundary
//  condition. Typically called in Mesh() ctor and in pgen/*.cpp files.

BoundaryFlag Mesh::GetBoundaryFlag(const std::string& input_string) {
  if (input_string == "reflect") {
    return BoundaryFlag::reflect;
  } else if (input_string == "outflow") {
    return BoundaryFlag::outflow;
  } else if (input_string == "inflow") {
    return BoundaryFlag::inflow;
  } else if (input_string == "diode") {
    return BoundaryFlag::diode;
  } else if (input_string == "user") {
    return BoundaryFlag::user;
  } else if (input_string == "periodic") {
    return BoundaryFlag::periodic;
  } else if (input_string == "vacuum") {
    return BoundaryFlag::vacuum;
  } else if (input_string == "shear_periodic") {
    return BoundaryFlag::shear_periodic;
  } else if (input_string == "undef") {
    return BoundaryFlag::undef;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Input string = '" << input_string << "' is an invalid boundary type"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn GetBoundaryString(BoundaryFlag input_flag)
//  \brief Parses enumerated type BoundaryFlag internal integer representation to return
//  string describing the boundary condition. Typicall used to format descriptive errors
//  or diagnostics. Inverse of GetBoundaryFlag().

std::string Mesh::GetBoundaryString(BoundaryFlag input_flag) {
  switch (input_flag) {
    case BoundaryFlag::block:  // 0
      return "block";
    case BoundaryFlag::reflect:
      return "reflect";
    case BoundaryFlag::inflow:
      return "inflow";
    case BoundaryFlag::outflow:
      return "outflow";
    case BoundaryFlag::diode:
      return "diode";
    case BoundaryFlag::user:
      return "user";
    case BoundaryFlag::periodic:
      return "periodic";
    case BoundaryFlag::shear_periodic:
      return "shear_periodic";
    case BoundaryFlag::undef:
      return "undef";
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Input enum class BoundaryFlag=" << static_cast<int>(input_flag)
         << " is an invalid boundary type" << std::endl;
      std::exit(EXIT_FAILURE);
      break;
  }
}

//----------------------------------------------------------------------------------------
// \fn Mesh::NewTimeStep()

void Mesh::NewTimeStep(const Real tlim) {
  // save old timestep
  dtold = dt;
  if (dt == std::numeric_limits<float>::max()) {
    dtold = 0.;
  }

  // cycle over all MeshBlocks on this rank and find minimum dt
  // Requires at least ONE of the physics modules to be defined.
  // limit increase in timestep to 2x old value
  dt = 2.0*dt;

  // Hydro timestep
  Real hydro_dt = std::numeric_limits<Real>::max();
  if (pmb_pack->phydro != nullptr) {
    hydro_dt = (cfl_no)*(pmb_pack->phydro->dtnew_hydro_cfl);
    dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->dtnew) );
    // viscosity timestep
    if (pmb_pack->phydro->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pvisc->dtnew) );
    }
    // thermal conduction timestep
    if (pmb_pack->phydro->pcond != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->pcond->dtnew) );
    }
    // source terms timestep
    if (pmb_pack->phydro->psrc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->phydro->psrc->dtnew) );
    }
  }
  // MHD timestep
  if (pmb_pack->pmhd != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->dtnew) );
    // viscosity timestep
    if (pmb_pack->pmhd->pvisc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pvisc->dtnew) );
    }
    // resistivity timestep (includes ambipolar diffusion, handled within Resistivity)
    if (pmb_pack->pmhd->presist != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->presist->dtnew) );
    }
    // thermal conduction timestep
    if (pmb_pack->pmhd->pcond != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->pcond->dtnew) );
    }
    // source terms timestep
    if (pmb_pack->pmhd->psrc != nullptr) {
      dt = std::min(dt, (cfl_no)*(pmb_pack->pmhd->psrc->dtnew) );
    }
  }
  // z4c timestep
  if (pmb_pack->pz4c != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->pz4c->dtnew) );
  }
  // Radiation timestep
  if (pmb_pack->prad != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->prad->dtnew) );
  }
  // Sink-particle timestep
  if (pmb_pack->psink != nullptr) {
    dt = std::min(dt, (cfl_no)*(pmb_pack->psink->dtnew) );
  }
  // Particles timestep
  if (pmb_pack->ppart != nullptr) {
    dt = std::min(dt, (pmb_pack->ppart->dtnew) );
  }

#if MPI_PARALLEL_ENABLED
  Real dt_min[2] = {dt, hydro_dt};
  MPI_Allreduce(MPI_IN_PLACE, dt_min, 2, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  dt = dt_min[0];
  hydro_dt = dt_min[1];
#endif
  hydro_lat_global_hydro_dt = hydro_dt;

  // limit last time step to stop at tlim *exactly*
  if ( (time < tlim) && ((time + dt) > tlim) ) {dt = tlim - time;}
  const Real tol = static_cast<Real>(64.0)*std::numeric_limits<Real>::epsilon()*
                   std::max(static_cast<Real>(1.0), hydro_dt);
  hydro_lat_dt_limited_by_hydro =
      pmb_pack->phydro != nullptr &&
      (hydro_dt < std::numeric_limits<Real>::max()) && (dt + tol >= hydro_dt);

  return;
}

void Mesh::InvalidateHydroLATMetadata() {
  hydro_lat_metadata_valid = false;
  hydro_lat_sync_factor_current = 1;
  hydro_lat_dt_limited_by_hydro = false;
  hydro_lat_metadata_topology_version = 0;
  hydro_lat_lb_topology_version = kInvalidLATVersion;
  hydro_lat_lb_metadata_version = kInvalidLATVersion;
  hydro_lat_lb_work_version = kInvalidLATVersion;
  ++hydro_lat_metadata_version;
  for (int i = 0; i < kMaxLATBinLevels; ++i) hydro_lat_bin_count[i] = 0;
}

int Mesh::HydroLATFactorForGID(int gid, int max_factor) const {
  if (!hydro_lat_metadata_valid || hydro_lat_factor_eachmb == nullptr ||
      gid < 0 || gid >= hydro_lat_metadata_nmb) {
    return 1;
  }
  return std::max(1, std::min(std::max(1, max_factor), hydro_lat_factor_eachmb[gid]));
}

int Mesh::HydroLATBinCountForFactor(int factor) const {
  if (!hydro_lat_metadata_valid || factor < 1) return 0;
  // Compute log2(factor) for the bin index
  int bin = 0;
  int f = factor;
  while (f > 1 && bin < kMaxLATBinLevels - 1) { f >>= 1; ++bin; }
  return (bin < kMaxLATBinLevels) ? hydro_lat_bin_count[bin] : 0;
}

// Pin sharp density transitions and their complete neighbour stencil to the finest
// clock. Delayed coarse/fine reflux can otherwise temporarily fill an ambient cell
// with dense material; sources and floors acting before the mass is refunded leave
// an irreversible thermal residue. Equal clocks use synchronous flux replacement.
// Called on every rank after the problem cap; subsequent limiters only lower factors.
void Mesh::ApplyHydroLATDensityPin() {
  const Real contrast = hydro_lat_pin_density_contrast;
  const Real rho_pin = hydro_lat_pin_density;
  const int nmb = pmb_pack->nmb_thispack;
  const auto in = mb_indcs;
  const int cells = in.nx1*in.nx2*in.nx3;
  if (nmb > 0) {
    // IDN is conserved density, in code units, for both fluid modules.
    auto u = pmb_pack->phydro->u0;
    DvceArray2D<Real> extrema("lat_pin_density_extrema", nmb, 2);
    const Real big = std::numeric_limits<Real>::max();
    Kokkos::parallel_for("lat_pin_density_extrema",
        Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t member) {
      const int m = member.league_rank();
      Real rmin = big;
      Real rmax = 0.0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange<>(member, cells),
      [&](const int c, Real &lmin) {
        const int i = in.is + c%in.nx1;
        const int j = in.js + (c/in.nx1)%in.nx2;
        const int k = in.ks + c/(in.nx1*in.nx2);
        lmin = Kokkos::fmin(lmin, u(m, IDN, k, j, i));
      }, Kokkos::Min<Real>(rmin));
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange<>(member, cells),
      [&](const int c, Real &lmax) {
        const int i = in.is + c%in.nx1;
        const int j = in.js + (c/in.nx1)%in.nx2;
        const int k = in.ks + c/(in.nx1*in.nx2);
        lmax = Kokkos::fmax(lmax, u(m, IDN, k, j, i));
      }, Kokkos::Max<Real>(rmax));
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        extrema(m, 0) = rmin;
        extrema(m, 1) = rmax;
      });
    });
    auto host = Kokkos::create_mirror_view(extrema);
    Kokkos::deep_copy(host, extrema);
    auto &mb_gid = pmb_pack->pmb->mb_gid;
    auto &nghbr = pmb_pack->pmb->nghbr;
    auto pin_gid = [&](const int gid) {
      if (gid >= 0 && gid < nmb_total) hydro_lat_factor_eachmb[gid] = 1;
    };
    for (int m = 0; m < nmb; ++m) {
      const Real rmin = host(m, 0), rmax = host(m, 1);
      const bool steep = (contrast > 1.0) && (rmax > 0.0) &&
          (rmin <= 0.0 || rmax >= contrast*rmin);
      const bool dense = (rho_pin > 0.0) && (rmax >= rho_pin);
      if (!steep && !dense) continue;
      pin_gid(mb_gid.h_view(m));
      for (int n = 0; n < pmb_pack->pmb->nnghbr; ++n) {
        pin_gid(nghbr.h_view(m, n).gid);
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  // A local block may pin a neighbour owned by another rank. All ranks participate,
  // including those with no selected blocks, to keep the global factor array identical.
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, hydro_lat_factor_eachmb, nmb_total, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
  }
#endif
}

void Mesh::UpdateHydroLATMetadata(int max_factor) {
  const int capped_max_factor = std::max(1, max_factor);
  if (nmb_total <= 0 || pmb_pack == nullptr ||
      pmb_pack->phydro == nullptr) {
    InvalidateHydroLATMetadata();
    return;
  }
  std::vector<int> previous_factors;
  std::vector<Real> previous_work;
  if (hydro_lat_metadata_valid && hydro_lat_metadata_nmb == nmb_total &&
      hydro_lat_metadata_topology_version == topology_version &&
      hydro_lat_factor_eachmb != nullptr) {
    previous_factors.assign(hydro_lat_factor_eachmb,
                            hydro_lat_factor_eachmb + nmb_total);
    if (hydro_lat_work_eachmb != nullptr) {
      previous_work.assign(hydro_lat_work_eachmb, hydro_lat_work_eachmb + nmb_total);
    }
  }
  if (hydro_lat_metadata_nmb != nmb_total || hydro_lat_dt_eachmb == nullptr ||
      hydro_lat_factor_eachmb == nullptr || hydro_lat_work_eachmb == nullptr) {
    delete [] hydro_lat_dt_eachmb;
    delete [] hydro_lat_factor_eachmb;
    delete [] hydro_lat_work_eachmb;
    hydro_lat_dt_eachmb = new Real[nmb_total];
    hydro_lat_factor_eachmb = new int[nmb_total];
    hydro_lat_work_eachmb = new Real[nmb_total];
    for (int gid=0; gid<nmb_total; ++gid) hydro_lat_work_eachmb[gid] = 1.0;
    hydro_lat_metadata_nmb = nmb_total;
  }

  for (int gid=0; gid<nmb_total; ++gid) {
    hydro_lat_dt_eachmb[gid] = hydro_lat_global_hydro_dt;
    hydro_lat_factor_eachmb[gid] = 1;
  }

  hydro::Hydro *phydro = pmb_pack->phydro;
  DualArray1D<Real> *dtnew_eachmb = &(phydro->dtnew_eachmb);
  SourceTerms *psrc = phydro->psrc;
  std::vector<Real> local_hydro_dt(std::max(0, nmb_thisrank),
                                   hydro_lat_global_hydro_dt);
  dtnew_eachmb->template sync<HostMemSpace>();
  if (psrc != nullptr) {
    psrc->dtnew_eachmb.template sync<HostMemSpace>();
  }
  const int first_gid = gids_eachrank[global_variable::my_rank];
  for (int m=0; m<nmb_thisrank; ++m) {
    const int gid = first_gid + m;
    if (gid >= 0 && gid < nmb_total) {
      local_hydro_dt[m] = cfl_no*dtnew_eachmb->h_view(m);
      if (psrc != nullptr) {
        local_hydro_dt[m] = std::min(
            local_hydro_dt[m], cfl_no*psrc->dtnew_eachmb.h_view(m));
      }
      hydro_lat_dt_eachmb[gid] = local_hydro_dt[m];
    }
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allgatherv(local_hydro_dt.data(), nmb_thisrank, MPI_ATHENA_REAL,
                   hydro_lat_dt_eachmb, nmb_eachrank, gids_eachrank,
                   MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif
  int max_present_factor = 1;
  if (dt > static_cast<Real>(0.0) &&
      hydro_lat_global_hydro_dt < std::numeric_limits<Real>::max()) {
    const Real eps = static_cast<Real>(64.0)*std::numeric_limits<Real>::epsilon();
    for (int gid=0; gid<nmb_total; ++gid) {
      const Real ratio = hydro_lat_dt_eachmb[gid]/dt;
      int factor = 1;
      while (factor < capped_max_factor &&
             static_cast<Real>(2*factor) <= ratio + eps) {
        factor *= 2;
      }
      hydro_lat_factor_eachmb[gid] = factor;
      max_present_factor = std::max(max_present_factor, factor);
    }
  }

  auto print_lat_bins = [&](const char *label) {
    if (!hydro_lat_diagnostics || global_variable::my_rank != 0) return;
    std::cout << "Mesh: HD LAT " << label << " bins";
    for (int factor=1; factor<=capped_max_factor; factor*=2) {
      int count = 0;
      for (int gid=0; gid<nmb_total; ++gid) {
        if (hydro_lat_factor_eachmb[gid] == factor) ++count;
      }
      if (count > 0) std::cout << " f" << factor << "=" << count;
      if (factor > (std::numeric_limits<int>::max()/2)) break;
    }
    std::cout << std::endl;
  };
  print_lat_bins("raw local-CFL");

  auto recompute_max_present_factor = [&]() {
    int max_factor_present = 1;
    for (int gid=0; gid<nmb_total; ++gid) {
      max_factor_present = std::max(max_factor_present, hydro_lat_factor_eachmb[gid]);
    }
    return max_factor_present;
  };

  if (pgen != nullptr && pgen->user_hydro_lat_factor_cap_func != nullptr &&
      max_present_factor > 1) {
    pgen->user_hydro_lat_factor_cap_func(this, capped_max_factor,
                                         hydro_lat_factor_eachmb);
    for (int gid=0; gid<nmb_total; ++gid) {
      const int requested = std::max(1, std::min(capped_max_factor,
                                                 hydro_lat_factor_eachmb[gid]));
      int factor = 1;
      while (factor < capped_max_factor && factor <= requested/2) {
        factor *= 2;
      }
      hydro_lat_factor_eachmb[gid] = factor;
    }
    max_present_factor = recompute_max_present_factor();
    print_lat_bins("problem capped");
  }

  if (max_present_factor > 1 &&
      (hydro_lat_pin_density_contrast > 1.0 || hydro_lat_pin_density > 0.0)) {
    ApplyHydroLATDensityPin();
    max_present_factor = recompute_max_present_factor();
    print_lat_bins("density pinned");
  }

  // Sink particles: factor-1 pinning (design N13).  The sink operator runs once per LAT
  // window over the whole local grid, but its accretion/creation kernels write conserved
  // gas state in the cells around each sink, and its accretion kernel reads an ambient
  // shell around that.  Those cells must be at the window's common time whenever the
  // operator touches them, which is exactly what factor 1 guarantees: a factor-1 block
  // is due on every fine tick and reaches the window endpoint with everyone else.  Every
  // block whose bounding box, grown by its own GHOST BAND in each direction, intersects a
  // pin sphere is therefore forced to factor 1.
  //
  // Review A2/F4(a) -- measured on the A1 RS leg.  The buffer used to be one full
  // CANDIDATE-BLOCK width, which on a coarse block is 2^(lmax-l) times the finest block
  // width: on the 128^3/16^3 two-level benchmark that is 0.125 against a pin radius of
  // 0.0156, so 56 level-0 blocks nowhere near the sink were pinned, and the "one bin per
  // AMR level" collapse below then dragged all 504 level-0 blocks to factor 1.  The
  // measured cost was the ENTIRE LAT speedup of the production shape (A1: RS runs
  // 124800 MB-cycles = every block every tick, LAT delivers exactly zero; the clean
  // ladder is f1=64 f2=56 f4=504).  Bound: -11 % of the RS wall at R2's measured 52 % LAT
  // efficiency, -61 % at the 89 % measured on the hydro+sink arm.
  //
  // ng cells is what the buffer actually has to cover: the accretion/creation kernels
  // touch interior cells only, and the ambient shell they read reaches at most through
  // the ghost band.  Sink DRIFT is covered by the pin sphere itself, not here -- its
  // radius is lat_pin_safety(=2)*ngrow*dx_finest while the kernel support is only
  // (ngrow+1/2)*dx_finest, i.e. a margin of ngrow finest cells, and the sink timestep
  // bounds one cycle of motion by a single finest cell.
  //
  // Rank consistency: the sink list is replicated and LATPinRegions() is refreshed on
  // every rank at every SinkStep, so this loop produces the identical factor vector
  // everywhere -- the same property the rest of this function relies on after the
  // Allgatherv.  Placed after the pgen cap hook and before the level collapse so that a
  // problem generator's cap cannot re-coarsen a pinned block, while the collapse and the
  // neighbour limiter below can only lower factors further.
  if (pmb_pack->psink != nullptr && lloc_eachmb != nullptr && max_present_factor > 1) {
    const std::vector<std::array<Real, 4>> &pin_regions =
        pmb_pack->psink->LATPinRegions();
    if (!pin_regions.empty()) {
      const Real len1 = mesh_size.x1max - mesh_size.x1min;
      const Real len2 = mesh_size.x2max - mesh_size.x2min;
      const Real len3 = mesh_size.x3max - mesh_size.x3min;
      const bool per1 =
          (mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
      const bool per2 = multi_d &&
          (mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
      const bool per3 = three_d &&
          (mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
      // Minimum-image distance from a point to a padded interval: 0 inside, otherwise
      // the gap to the nearer edge.  The wrap uses the actual mesh bounds (the sink
      // module's own R4 fix), not an origin-at-zero shortcut.
      auto axis_gap = [](Real c, Real lo, Real hi, Real len, bool periodic) {
        const Real mid = 0.5*(lo + hi);
        const Real half = 0.5*(hi - lo);
        Real dc = c - mid;
        if (periodic && len > 0.0) {
          if (dc >  0.5*len) { dc -= len; }
          if (dc < -0.5*len) { dc += len; }
        }
        const Real gap = std::abs(dc) - half;
        return (gap > 0.0) ? gap : static_cast<Real>(0.0);
      };
      int npinned = 0;
      for (int gid=0; gid<nmb_total; ++gid) {
        if (hydro_lat_factor_eachmb[gid] <= 1) continue;
        const std::int32_t lev = lloc_eachmb[gid].level;
        const std::int32_t nmbx1 = nmb_rootx1 << (lev - root_level);
        const Real x1lo = LeftEdgeX(lloc_eachmb[gid].lx1, nmbx1,
                                    mesh_size.x1min, mesh_size.x1max);
        const Real x1hi = LeftEdgeX(lloc_eachmb[gid].lx1 + 1, nmbx1,
                                    mesh_size.x1min, mesh_size.x1max);
        Real x2lo = mesh_size.x2min, x2hi = mesh_size.x2max;
        if (multi_d) {
          const std::int32_t nmbx2 = nmb_rootx2 << (lev - root_level);
          x2lo = LeftEdgeX(lloc_eachmb[gid].lx2, nmbx2,
                           mesh_size.x2min, mesh_size.x2max);
          x2hi = LeftEdgeX(lloc_eachmb[gid].lx2 + 1, nmbx2,
                           mesh_size.x2min, mesh_size.x2max);
        }
        Real x3lo = mesh_size.x3min, x3hi = mesh_size.x3max;
        if (three_d) {
          const std::int32_t nmbx3 = nmb_rootx3 << (lev - root_level);
          x3lo = LeftEdgeX(lloc_eachmb[gid].lx3, nmbx3,
                           mesh_size.x3min, mesh_size.x3max);
          x3hi = LeftEdgeX(lloc_eachmb[gid].lx3 + 1, nmbx3,
                           mesh_size.x3min, mesh_size.x3max);
        }
        // Grow the box by this block's own ghost band (ng of ITS cells) per direction.
        const Real ngr = static_cast<Real>(mb_indcs.ng);
        const Real w1 = ngr*(x1hi - x1lo)/static_cast<Real>(mb_indcs.nx1);
        const Real w2 = multi_d ?
            ngr*(x2hi - x2lo)/static_cast<Real>(mb_indcs.nx2) : static_cast<Real>(0.0);
        const Real w3 = three_d ?
            ngr*(x3hi - x3lo)/static_cast<Real>(mb_indcs.nx3) : static_cast<Real>(0.0);
        for (const auto &region : pin_regions) {
          const Real rpin = region[3];
          if (!(rpin > 0.0)) continue;
          const Real d1 = axis_gap(region[0], x1lo - w1, x1hi + w1, len1, per1);
          if (d1 > rpin) continue;
          const Real d2 = multi_d ?
              axis_gap(region[1], x2lo - w2, x2hi + w2, len2, per2) :
              static_cast<Real>(0.0);
          if (d2 > rpin) continue;
          const Real d3 = three_d ?
              axis_gap(region[2], x3lo - w3, x3hi + w3, len3, per3) :
              static_cast<Real>(0.0);
          if (d1*d1 + d2*d2 + d3*d3 <= rpin*rpin) {
            hydro_lat_factor_eachmb[gid] = 1;
            ++npinned;
            break;
          }
        }
      }
      if (npinned > 0) {
        max_present_factor = recompute_max_present_factor();
        print_lat_bins("sink pinned");
      }
    }
  }

  // The conservative fallback only corrects AMR coarse/fine interfaces, so it keeps each
  // AMR level in one bin.  Same-level LAT flux synchronization permits raw per-block
  // bins.
  if (!hydro_lat_same_level && max_present_factor > 1 && lloc_eachmb != nullptr) {
    const int nlevels = std::max(0, max_level - root_level + 1);
    std::vector<int> level_factor(nlevels, capped_max_factor);
    std::vector<int> level_count(nlevels, 0);
    for (int gid=0; gid<nmb_total; ++gid) {
      const int ilev = lloc_eachmb[gid].level - root_level;
      if (ilev < 0 || ilev >= nlevels) continue;
      level_factor[ilev] = std::min(level_factor[ilev], hydro_lat_factor_eachmb[gid]);
      level_count[ilev] += 1;
    }
    for (int gid=0; gid<nmb_total; ++gid) {
      const int ilev = lloc_eachmb[gid].level - root_level;
      if (ilev >= 0 && ilev < nlevels && level_count[ilev] > 0) {
        hydro_lat_factor_eachmb[gid] = std::max(1, level_factor[ilev]);
      }
    }
    max_present_factor = recompute_max_present_factor();
  }

  // The floor exists so a bin is never so sparsely populated that its task list and
  // boundary exchange cost more than the sub-cycling saves, and "sparse" is a per-rank
  // property: a 32-block bin is dense on 8 ranks and leaves 68 of 100 ranks idle.  The
  // default therefore scales with nranks, which makes the FACTOR assignment -- i.e. each
  // block's time step, i.e. the trajectory -- depend on the decomposition.  No
  // rank-independent constant can replace it without being wrong at one end of the range,
  // so the rank dependence is deliberate; set time/hydro_lat_min_bin_count explicitly to
  // remove it (the Mesh constructor warns when it is left at the default).
  const int min_bin_count = (hydro_lat_min_bin_count >= 0) ?
      hydro_lat_min_bin_count : std::max(0, 4*global_variable::nranks);
  if (min_bin_count > 0) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (int factor=max_present_factor; factor>1; factor/=2) {
        int count = 0;
        for (int gid=0; gid<nmb_total; ++gid) {
          if (hydro_lat_factor_eachmb[gid] == factor) ++count;
        }
        if (count > 0 && count < min_bin_count) {
          const int clamped_factor = std::max(1, factor/2);
          for (int gid=0; gid<nmb_total; ++gid) {
            if (hydro_lat_factor_eachmb[gid] == factor) {
              hydro_lat_factor_eachmb[gid] = clamped_factor;
            }
          }
          changed = true;
        }
      }

      max_present_factor = 1;
      for (int gid=0; gid<nmb_total; ++gid) {
        max_present_factor = std::max(max_present_factor, hydro_lat_factor_eachmb[gid]);
      }
    }
  }

  // AMR neighbors need level-consistent time bins.  Delayed reflux can correct
  // the time-integrated face flux, but coarse/fine boundary fills and
  // prolongation still consume neighbor states during the step.  Keep adjacent
  // AMR levels within the standard 2:1 time-step ratio, and apply the limiter
  // to all geometric neighbors because edge/corner coarse data are used by the
  // prolongation stencil.
  // time/lat_neighbor_limiter selects how much of this safety envelope is
  // used for experiments:
  //   all    : conservative all-neighbor same-level and AMR limiting
  //   hybrid : all-neighbor AMR limiting, strict same-level face limiting, and
  //            looser edge/corner limiting once LAT has synchronized history.
  //            Edge/corner neighbors stay strict next to factor-1 blocks because
  //            those are the dynamically sensitive/source-coupled region.
  //   face   : face-only same-level limiting, with the same all-neighbor AMR
  //            limiting as hybrid.  Coarse/fine edge and corner neighbors still
  //            participate in prolongation stencils, so the old face-only AMR
  //            rule is not safe with same-level LAT.
  //
  // Same-level mixed LAT is strict until each block has a clean synchronized
  // start/end history.  After that, keep a strict 2:1 staircase around low-factor
  // same-level neighbors, but allow the configured larger ratio once both sides
  // are already in smooth high-factor bins.  The same-level part follows the
  // requested neighbor limiter: "all" limits face/edge/corner neighbors,
  // "face" limits only face neighbors, and "hybrid" keeps face neighbors strict
  // while allowing edge/corner neighbors to use the configured larger ratio.
  if (max_present_factor > 1 && pmb_pack != nullptr && pmb_pack->pmb != nullptr) {
    // The same-level staircase needs no cross-window history: mixed-factor ghosts are
    // interpolated from the current window's start/end registers, so the window after
    // a topology change is limited exactly like the first window after a restart.
    const int same_level_ratio_max = hydro_lat_same_level ?
        std::max(1, std::min(8, hydro_lat_same_level_max_ratio)) : 1;
    const bool source_coupled_hydro =
        (psrc != nullptr) && (psrc->self_gravity || psrc->external_bh_gravity);
    const int nmb = pmb_pack->nmb_thispack;
    const int nnghbr = pmb_pack->pmb->nnghbr;
    std::vector<int> limited_factor(nmb_total, 1);
    long long same_level_limit_events = 0;
    bool changed = true;
    int limiter_iter = 0;
    const int max_limiter_iters = std::max(1, nmb_total);
    while (changed) {
      if (++limiter_iter > max_limiter_iters) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "HD LAT neighbor limiter failed to converge after "
                  << max_limiter_iters << " iterations." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      for (int gid=0; gid<nmb_total; ++gid) {
        limited_factor[gid] = hydro_lat_factor_eachmb[gid];
      }
      int local_changed = 0;
      auto limit_from_neighbor = [&](const int m, const int gid, const int lev,
                                     const int factor, const int n) {
        const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
        const auto &nb = pmb_pack->pmb->nghbr.h_view(m,n);
        if (nb.gid < 0 || nb.gid >= nmb_total) {
          return;
        }
        const int nb_factor = std::max(1, hydro_lat_factor_eachmb[nb.gid]);
        if (nb.lev == lev && hydro_lat_same_level) {
          if (hydro_lat_neighbor_limiter_mode == 2 && !face_nghbr) {
            return;
          }
          const int min_factor = std::min(factor, nb_factor);
          const bool strict_source_region =
              source_coupled_hydro && min_factor <= 1;
          int same_level_ratio = 1;
          if (same_level_ratio_max > 1) {
            if (hydro_lat_neighbor_limiter_mode == 0) {
              same_level_ratio =
                  std::min(same_level_ratio_max, std::max(2, min_factor));
            } else if (hydro_lat_neighbor_limiter_mode == 1) {
              same_level_ratio = (face_nghbr || strict_source_region) ?
                  2 : std::min(same_level_ratio_max, std::max(2, min_factor));
            } else {
              same_level_ratio =
                  std::min(same_level_ratio_max, std::max(2, min_factor));
            }
          }
          const int max_same_level_factor =
              std::min(capped_max_factor, same_level_ratio*min_factor);
          if (factor > max_same_level_factor) {
            limited_factor[gid] = std::min(limited_factor[gid], max_same_level_factor);
            local_changed = 1;
            ++same_level_limit_events;
          }
          if (nb_factor > max_same_level_factor) {
            limited_factor[nb.gid] = std::min(limited_factor[nb.gid],
                                              max_same_level_factor);
            local_changed = 1;
            ++same_level_limit_events;
          }
        } else if (nb.lev > lev) {
          int max_coarse_factor = nb_factor;
          for (int d=0; d<(nb.lev - lev); ++d) {
            if (max_coarse_factor >= capped_max_factor) break;
            max_coarse_factor = std::min(capped_max_factor, 2*max_coarse_factor);
          }
          if (factor > max_coarse_factor) {
            limited_factor[gid] = std::min(limited_factor[gid], max_coarse_factor);
            local_changed = 1;
          }
          if (nb_factor > factor) {
            limited_factor[nb.gid] = std::min(limited_factor[nb.gid], factor);
            local_changed = 1;
          }
        } else {
          int max_coarse_factor = factor;
          for (int d=0; d<(lev - nb.lev); ++d) {
            if (max_coarse_factor >= capped_max_factor) break;
            max_coarse_factor = std::min(capped_max_factor, 2*max_coarse_factor);
          }
          if (nb_factor > max_coarse_factor) {
            limited_factor[nb.gid] = std::min(limited_factor[nb.gid], max_coarse_factor);
            local_changed = 1;
          }
          if (factor > nb_factor) {
            limited_factor[gid] = std::min(limited_factor[gid], nb_factor);
            local_changed = 1;
          }
        }
      };
      for (int m=0; m<nmb; ++m) {
        const int gid = pmb_pack->pmb->mb_gid.h_view(m);
        if (gid < 0 || gid >= nmb_total) continue;
        const int lev = pmb_pack->pmb->mb_lev.h_view(m);
        const int factor = std::max(1, hydro_lat_factor_eachmb[gid]);
        for (int n=0; n<nnghbr; ++n) {
          limit_from_neighbor(m, gid, lev, factor, n);
        }
      }

#if MPI_PARALLEL_ENABLED
      if (global_variable::nranks > 1) {
        int global_changed = local_changed;
        MPI_Allreduce(MPI_IN_PLACE, &global_changed, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (global_changed == 0) {
          changed = false;
          break;
        }
        MPI_Allreduce(MPI_IN_PLACE, limited_factor.data(), nmb_total, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
      } else if (local_changed == 0) {
        changed = false;
        break;
      }
#else
      if (local_changed == 0) {
        changed = false;
        break;
      }
#endif
      local_changed = 0;
      for (int gid=0; gid<nmb_total; ++gid) {
        if (limited_factor[gid] < hydro_lat_factor_eachmb[gid]) {
          local_changed = 1;
          break;
        }
      }
      changed = (local_changed != 0);
      if (changed) {
        for (int gid=0; gid<nmb_total; ++gid) {
          hydro_lat_factor_eachmb[gid] = std::max(1, limited_factor[gid]);
        }
      }
    }
    max_present_factor = recompute_max_present_factor();
    if (hydro_lat_diagnostics) {
      long long status_counts[8] = {0, 0, 0, 0, 0, same_level_limit_events, 0, 0};
      long long amr_counts[6] = {0, 0, 0, 0, 0, 0};
      int local_max_ratio = 1;
      int local_max_amr_ratio = 1;
      for (int m=0; m<nmb; ++m) {
        const int gid = pmb_pack->pmb->mb_gid.h_view(m);
        if (gid < 0 || gid >= nmb_total) continue;
        const int lev = pmb_pack->pmb->mb_lev.h_view(m);
        const int factor = std::max(1, hydro_lat_factor_eachmb[gid]);
        for (int n=0; n<nnghbr; ++n) {
          const bool face_nghbr = (n < 16) || ((n >= 24) && (n < 32));
          const auto &nb = pmb_pack->pmb->nghbr.h_view(m,n);
          if (nb.gid < 0 || nb.gid >= nmb_total) continue;
          const int nb_factor = std::max(1, hydro_lat_factor_eachmb[nb.gid]);
          if (nb.lev != lev) {
            ++amr_counts[0];
            if (face_nghbr) {
              ++amr_counts[1];
            } else {
              ++amr_counts[2];
            }
            const bool self_fine = (lev > nb.lev);
            const int fine_factor = self_fine ? factor : nb_factor;
            const int coarse_factor = self_fine ? nb_factor : factor;
            const int level_diff = std::abs(lev - nb.lev);
            int max_coarse_factor = fine_factor;
            for (int d=0; d<level_diff; ++d) {
              if (max_coarse_factor >= capped_max_factor) break;
              max_coarse_factor = std::min(capped_max_factor, 2*max_coarse_factor);
            }
            if (fine_factor != coarse_factor) ++amr_counts[3];
            if (fine_factor > coarse_factor || coarse_factor > max_coarse_factor) {
              ++amr_counts[4];
            }
            local_max_amr_ratio =
                std::max(local_max_amr_ratio, coarse_factor/fine_factor);
            continue;
          }
          ++status_counts[0];
          if (face_nghbr) {
            ++status_counts[1];
          } else {
            ++status_counts[2];
          }
          if (factor != nb_factor) ++status_counts[3];
          const bool constrained = (hydro_lat_neighbor_limiter_mode == 0) ||
                                   (hydro_lat_neighbor_limiter_mode == 1) ||
                                   face_nghbr;
          if (constrained) {
            ++status_counts[6];
            if (factor != nb_factor) ++status_counts[7];
          }
          const int min_factor = std::min(factor, nb_factor);
          const bool strict_source_region =
              source_coupled_hydro && min_factor <= 1;
          int same_level_ratio = 1;
          if (same_level_ratio_max > 1) {
            if (hydro_lat_neighbor_limiter_mode == 0) {
              same_level_ratio =
                  std::min(same_level_ratio_max, std::max(2, min_factor));
            } else if (hydro_lat_neighbor_limiter_mode == 1) {
              same_level_ratio = (face_nghbr || strict_source_region) ?
                  2 : std::min(same_level_ratio_max, std::max(2, min_factor));
            } else {
              same_level_ratio =
                  std::min(same_level_ratio_max, std::max(2, min_factor));
            }
          }
          const int max_same_level_factor =
              std::min(capped_max_factor, same_level_ratio*min_factor);
          if (constrained && std::max(factor, nb_factor) > max_same_level_factor) {
            ++status_counts[4];
          }
          local_max_ratio = std::max(local_max_ratio,
                                     std::max(factor, nb_factor)/min_factor);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (global_variable::nranks > 1) {
        MPI_Allreduce(MPI_IN_PLACE, status_counts, 8, MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, amr_counts, 6, MPI_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &local_max_ratio, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &local_max_amr_ratio, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
      }
#endif
      if (global_variable::my_rank == 0) {
        const char *limiter_mode =
            (hydro_lat_neighbor_limiter_mode == 2) ? "face" :
            ((hydro_lat_neighbor_limiter_mode == 1) ? "hybrid" : "all");
        std::cout << "Mesh: HD LAT same-level status enabled="
                  << (hydro_lat_same_level ? "true" : "false")
                  << ", limiter=" << limiter_mode
                  << ", configured_ratio=" << hydro_lat_same_level_max_ratio
                  << ", effective_ratio_max=" << same_level_ratio_max
                  << ", source_coupled_hydro="
                  << (source_coupled_hydro ? "true" : "false")
                  << ", iterations=" << limiter_iter
                  << ", directed_neighbors=" << status_counts[0]
                  << ", face=" << status_counts[1]
                  << ", edge_corner=" << status_counts[2]
                  << ", constrained_neighbors=" << status_counts[6]
                  << ", mixed_factor=" << status_counts[3]
                  << ", constrained_mixed_factor=" << status_counts[7]
                  << ", constrained_over_limit=" << status_counts[4]
                  << ", max_observed_ratio=" << local_max_ratio
                  << ", limit_events=" << status_counts[5]
                  << std::endl;
        std::cout << "Mesh: HD LAT AMR-interface status limiter=" << limiter_mode
                  << ", directed_neighbors=" << amr_counts[0]
                  << ", face=" << amr_counts[1]
                  << ", edge_corner=" << amr_counts[2]
                  << ", mixed_factor=" << amr_counts[3]
                  << ", over_limit=" << amr_counts[4]
                  << ", max_observed_coarse_to_fine_ratio="
                  << local_max_amr_ratio
                  << std::endl;
      }
    }
    const char *limiter_label =
        (hydro_lat_neighbor_limiter_mode == 2) ? "face-neighbor limited" :
        ((hydro_lat_neighbor_limiter_mode == 1) ? "hybrid neighbor limited" :
         (same_level_ratio_max > 2 ? "same-level regional/neighbor limited" :
          (same_level_ratio_max > 1 ? "same-level 2:1/neighbor limited" :
           "neighbor limited")));
    print_lat_bins(limiter_label);
  }

  hydro_lat_sync_factor_current = std::max(1, max_present_factor);

  // Populate per-factor bin count cache.
  for (int i = 0; i < kMaxLATBinLevels; ++i) hydro_lat_bin_count[i] = 0;
  for (int gid = 0; gid < nmb_total; ++gid) {
    int factor = std::max(1, hydro_lat_factor_eachmb[gid]);
    int bin = 0;
    while (factor > 1 && bin < kMaxLATBinLevels - 1) { factor >>= 1; ++bin; }
    if (bin < kMaxLATBinLevels) hydro_lat_bin_count[bin]++;
  }

  {
    std::vector<Real> local_work(std::max(0, nmb_thisrank), 1.0);
    for (int m=0; m<nmb_thisrank; ++m) {
      const int gid = first_gid + m;
      if (gid >= 0 && gid < nmb_total) hydro_lat_work_eachmb[gid] = local_work[m];
    }
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allgatherv(local_work.data(), nmb_thisrank, MPI_ATHENA_REAL,
                     hydro_lat_work_eachmb, nmb_eachrank, gids_eachrank,
                     MPI_ATHENA_REAL, MPI_COMM_WORLD);
    }
#endif
  }
  // Re-open the partition only when the measured work moved AND the current partition
  // can still be beaten by the migration threshold: the window time is the sum over
  // tick classes of the busiest rank's work in that class, and no partition can do
  // better than the perfectly balanced sum, so the gap to that bound caps what a
  // migration can gain.  Uniform work never re-opens it, and
  // once a partition is within the threshold of the bound the versions stay equal, so
  // there is no periodic re-partition and no churn.  The version deliberately does not
  // touch hydro_lat_metadata_version (rank-packed layouts).
  {
    const bool work_changed =
        previous_work.size() != static_cast<std::size_t>(nmb_total) ||
        !std::equal(previous_work.begin(), previous_work.end(), hydro_lat_work_eachmb);
    if (work_changed && global_variable::nranks > 1 && gids_eachrank != nullptr &&
        nmb_eachrank != nullptr) {
      const int nranks = global_variable::nranks;
      const int sync = hydro_lat_sync_factor_current;
      double sum_max = 0.0, sum_mean = 0.0;
      for (int cls_factor=1; cls_factor<=sync; cls_factor*=2) {
        const int multiplicity = (cls_factor == sync) ? 1 : sync/(2*cls_factor);
        double tick_max = 0.0, tick_total = 0.0;
        for (int rank=0; rank<nranks; ++rank) {
          double rank_work = 0.0;
          const int begin = gids_eachrank[rank];
          const int end = begin + nmb_eachrank[rank];
          for (int gid=begin; gid<end && gid<nmb_total; ++gid) {
            if (gid >= 0 && std::max(1, hydro_lat_factor_eachmb[gid]) <= cls_factor) {
              rank_work += static_cast<double>(hydro_lat_work_eachmb[gid]);
            }
          }
          tick_max = std::max(tick_max, rank_work);
          tick_total += rank_work;
        }
        sum_max += multiplicity*tick_max;
        sum_mean += multiplicity*tick_total/static_cast<double>(nranks);
        if (cls_factor > (std::numeric_limits<int>::max()/2)) break;
      }
      if (sum_max > 0.0 &&
          (sum_max - sum_mean)/sum_max >= kHydroLATRebalanceMinRelativeGain) {
        ++hydro_lat_work_version;
      }
    }
  }

  hydro_lat_metadata_valid = true;
  hydro_lat_metadata_topology_version = topology_version;
  print_lat_bins("final");
  const bool factors_unchanged =
      previous_factors.size() == static_cast<std::size_t>(nmb_total) &&
      std::equal(previous_factors.begin(), previous_factors.end(),
                 hydro_lat_factor_eachmb);
  if (!factors_unchanged) {
    ++hydro_lat_metadata_version;
    if (pmb_pack != nullptr) pmb_pack->InvalidateLATFactorCache();
  }
}

//----------------------------------------------------------------------------------------
// \fn Mesh::AddCoordinatesAndPhysics

void Mesh::AddCoordinatesAndPhysics(ParameterInput *pinput) {
  // cycle over MeshBlockPacks on this rank and add Coordinates and Physics
  for (int n=0; n<nmb_packs_thisrank; ++n) {
    pmb_pack->AddCoordinates(pinput);
    pmb_pack->AddPhysics(pinput);
  }

  // Determine total number of particles across all ranks
  particles::Particles *ppart = pmb_pack->ppart;
  if (ppart != nullptr) {
    nprtcl_thisrank = 0;
    for (int n=0; n<nmb_packs_thisrank; ++n) {
      nprtcl_thisrank += pmb_pack->ppart->nprtcl_thispack;
    }
    nprtcl_eachrank = new int[global_variable::nranks];
    nprtcl_eachrank[global_variable::my_rank] = nprtcl_thisrank;
#if MPI_PARALLEL_ENABLED
    // Share number of particles on each rank with all ranks
    MPI_Allgather(&nprtcl_thisrank,1,MPI_INT,nprtcl_eachrank,1,MPI_INT,MPI_COMM_WORLD);
#endif
    for (int n=0; n<global_variable::nranks; ++n) {
      nprtcl_total += nprtcl_eachrank[n];
    }
    // Assign particle IDs
    if (pmb_pack->ppart != nullptr) {
      pmb_pack->ppart->CreateParticleTags(pinput);
    }
  }
}
