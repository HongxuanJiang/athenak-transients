//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.cpp
//! \brief implementation of Radiation class constructor and assorted other functions

#include <float.h>

#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "radiation/radiation.hpp"
#include "../mesh/mb_storage.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
    nh_c("nh_c",1,1),
    nh_f("nh_f",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    na("na",1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("iflx",1,1,1,1,1),
    divfa("divfa",1,1,1,1,1),
    pmy_pack(ppack) {
  // Check for general relativity
  if (!(pmy_pack->pcoord->is_general_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation requires general relativity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydrodynamics, mhd, and units
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  are_units_enabled = pin->DoesBlockExist("units");

  // Enable radiation source term (radiation+(M)HD) by default if hydro or mhd enabled
  // Otherwise, disable radiation source term.  The former can be overriden by
  // specification in the input file.
  if (is_hydro_enabled || is_mhd_enabled) {
    rad_source = pin->GetOrAddBoolean("radiation","rad_source",true);
  } else {
    rad_source = false;
  }

  // Set radiation coupling parameters including scattering and absorption opacities,
  // radiation constant, and source term behavior.
  if (rad_source) {
    kappa_s = pin->GetReal("radiation","kappa_s");
    power_opacity = pin->GetOrAddBoolean("radiation","power_opacity",false);
    if (!(power_opacity)) {
      kappa_a = pin->GetReal("radiation","kappa_a");
      kappa_p = pin->GetReal("radiation","kappa_p");
    }
    is_compton_enabled = pin->GetOrAddBoolean("radiation","compton",false);
    if (is_compton_enabled && !(are_units_enabled)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Compton requires enabling units" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (are_units_enabled) {
      arad = (pmy_pack->punit->rad_constant_cgs*
              SQR(SQR(pmy_pack->punit->temperature_cgs()))/
              pmy_pack->punit->pressure_cgs());
    } else {
      arad = pin->GetReal("radiation","arad");
    }
    affect_fluid = pin->GetOrAddBoolean("radiation","affect_fluid",true);
  }

  // Check for fluid evolution
  fixed_fluid = pin->GetOrAddBoolean("radiation","fixed_fluid",false);

  // Source terms (if needed)
  if (pin->DoesBlockExist("rad_srcterms")) {
    psrc = new SourceTerms("rad_srcterms", ppack, pin);
  }

  // Setup angular mesh and radiation geometry data
  int nlevel = pin->GetInteger("radiation", "nlevel");
  rotate_geo = pin->GetOrAddBoolean("radiation","rotate_geo",true);
  angular_fluxes = pin->GetOrAddBoolean("radiation","angular_fluxes",true);
  n_0_floor = pin->GetOrAddReal("radiation","n_0_floor",0.1);
  prgeo = new GeodesicGrid(nlevel, rotate_geo, angular_fluxes);

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = ppack->nmb_thispack;
  if (MeshBlockStorageReserve() > 0) nmb = MeshBlockStorageCapacity(nmb);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(nh_c,prgeo->nangles,4);
  Kokkos::realloc(nh_f,prgeo->nangles,6,4);
  Kokkos::realloc(tet_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tetcov_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tet_d1_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tet_d2_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tet_d3_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  if (angular_fluxes) {Kokkos::realloc(na,nmb,prgeo->nangles,ncells3,ncells2,ncells1,6);}
  if (is_hydro_enabled || is_mhd_enabled) {
    Kokkos::realloc(norm_to_tet,nmb,4,4,ncells3,ncells2,ncells1);
  }
  }
  SetOrthonormalTetrad();

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");
  time_evolving = (evolution_t.compare("stationary") != 0);

  // allocate memory for intensities
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(i0,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_i0,nmb,prgeo->nangles,nccells3,nccells2,nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_i = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_i->InitializeBuffers(prgeo->nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (time_evolving) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("radiation","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<radiation> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    }

    // allocate second registers, fluxes, masks
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(i1,      nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    if (angular_fluxes) {
      Kokkos::realloc(divfa,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

Radiation::~Radiation() {
  delete pbval_i;
  delete prgeo;
  if (psrc != nullptr) {delete psrc;}
}

bool Radiation::ResizeMeshBlockStorage(int nmb, bool exact, bool allow_shrink) {
  // See the matching comment in Hydro::ResizeMeshBlockStorage.
  if (pbval_i != nullptr) {
    pbval_i->ResizeBuffers(nmb);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nccells1 = indcs.cnx1 + 2*(indcs.ng);
  int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;

  auto resize6 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3,
                                       int n4, int n5) {
    const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                       view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                       view.extent_int(3) != n3 || view.extent_int(4) != n4 ||
                       view.extent_int(5) != n5);
    if (need) {
      Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0), n1, n2, n3, n4, n5);
    }
    return need;
  };
  auto resize5 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3,
      int n4) {
    const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                       view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                       view.extent_int(3) != n3 || view.extent_int(4) != n4);
    if (need) {
      Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0), n1, n2, n3, n4);
    }
    return need;
  };

  bool resized = false;
  resized = resize6(tet_c, nmb, 4, 4, ncells3, ncells2, ncells1) || resized;
  resized = resize6(tetcov_c, nmb, 4, 4, ncells3, ncells2, ncells1) || resized;
  resized = resize5(tet_d1_x1f, nmb, 4, ncells3, ncells2, ncells1 + 1) || resized;
  resized = resize5(tet_d2_x2f, nmb, 4, ncells3, ncells2 + 1, ncells1) || resized;
  resized = resize5(tet_d3_x3f, nmb, 4, ncells3 + 1, ncells2, ncells1) || resized;
  if (angular_fluxes) {
    resized = resize6(na, nmb, prgeo->nangles, ncells3, ncells2, ncells1, 6) || resized;
  }
  if (is_hydro_enabled || is_mhd_enabled) {
    resized = resize6(norm_to_tet, nmb, 4, 4, ncells3, ncells2, ncells1) || resized;
  }
  resized = resize5(i0, nmb, prgeo->nangles, ncells3, ncells2, ncells1) || resized;
  if (pmy_pack->pmesh->multilevel) {
    resized = resize5(coarse_i0, nmb, prgeo->nangles, nccells3, nccells2, nccells1) ||
              resized;
  }
  if (time_evolving) {
    resized = resize5(i1, nmb, prgeo->nangles, ncells3, ncells2, ncells1) || resized;
    resized = resize5(iflx.x1f, nmb, prgeo->nangles, ncells3, ncells2,
        ncells1) || resized;
    resized = resize5(iflx.x2f, nmb, prgeo->nangles, ncells3, ncells2,
        ncells1) || resized;
    resized = resize5(iflx.x3f, nmb, prgeo->nangles, ncells3, ncells2,
        ncells1) || resized;
    if (angular_fluxes) {
      resized = resize5(divfa, nmb, prgeo->nangles, ncells3, ncells2, ncells1) || resized;
    }
  }
  return resized;
}

} // namespace radiation
