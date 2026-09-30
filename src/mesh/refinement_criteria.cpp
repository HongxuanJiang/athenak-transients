//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refinement_criteria.cpp
//! \brief Implements constructor and functions in RefinementCriteria class.

#include <cstdlib>
#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "mesh/mb_storage.hpp"

#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "radiation/radiation.hpp"
#include "refinement_criteria.hpp"
#include "utils/utils.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// RefinementCriteria constructor:

RefinementCriteria::RefinementCriteria(Mesh *pm, ParameterInput *pin) :
    ncriteria(0),
    nderived(0),
    pmy_mesh(pm),
    dvars("derived_ref_vars",1,1,1,1,1) {
  // cycle through ParameterInput list and read each <amr_criterion> block
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 13, "amr_criterion") == 0) {
      RefCritData rcrit0;
      std::string method = pin->GetString(it->block_name, "method");
      if (method.compare("min_max") == 0) {
        rcrit0.rmethod = RefCritMethod::min_max;
      } else if (method.compare("slope") == 0) {
        rcrit0.rmethod = RefCritMethod::slope;
      } else if (method.compare("second_deriv") == 0) {
        rcrit0.rmethod = RefCritMethod::second_deriv;
      } else if (method.compare("location") == 0) {
        rcrit0.rmethod = RefCritMethod::location;
      } else if (method.compare("user") == 0) {
        rcrit0.rmethod = RefCritMethod::user;
      } else {
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
        Kokkos::abort("Unknown refinement criterion");
      }
      // read refinement variable only when needed
      if ((method.compare("location")!=0) && (method.compare("user")!=0)) {
        rcrit0.rvariable = pin->GetString(it->block_name,"variable");
      }
      rcrit0.rvalue_min = pin->GetOrAddReal(it->block_name,"value_min",(-FLT_MAX));
      rcrit0.rvalue_max = pin->GetOrAddReal(it->block_name,"value_max", (FLT_MAX));
      rcrit0.rderef_value_min = pin->GetOrAddReal(it->block_name, "derefine_value_min",
                                                  rcrit0.rvalue_min);
      rcrit0.rderef_value_max = pin->GetOrAddReal(it->block_name, "derefine_value_max",
                                                  rcrit0.rvalue_max);
      rcrit0.refine_only = pin->GetOrAddBoolean(it->block_name, "refine_only", false);
      const bool has_density_min = pin->DoesParameterExist(it->block_name, "density_min");
      const bool has_rho_min = pin->DoesParameterExist(it->block_name, "rho_min");
      if (has_density_min && has_rho_min) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << it->block_name
                  << "> should set only one of density_min or rho_min." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      rcrit0.use_density_min = has_density_min || has_rho_min;
      rcrit0.density_min = -FLT_MAX;
      if (has_density_min) {
        rcrit0.density_min = pin->GetReal(it->block_name, "density_min");
      } else if (has_rho_min) {
        rcrit0.density_min = pin->GetReal(it->block_name, "rho_min");
      }
      if (rcrit0.use_density_min && rcrit0.density_min < 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << it->block_name
                  << ">/density_min must be >= 0." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      rcrit0.use_radius_gate =
          pin->DoesParameterExist(it->block_name, "radius_min") ||
          pin->DoesParameterExist(it->block_name, "radius_max");
      rcrit0.radius_min = pin->GetOrAddReal(it->block_name, "radius_min", 0.0);
      rcrit0.radius_max = pin->GetOrAddReal(it->block_name, "radius_max", FLT_MAX);
      std::string radius_center =
          pin->GetOrAddString(it->block_name, "radius_center", "location");
      rcrit0.radius_center_bh =
          (radius_center.compare("bh") == 0) ||
          (radius_center.compare("external_bh") == 0);
      if ((radius_center.compare("location") != 0) && !rcrit0.radius_center_bh) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << it->block_name
                  << ">/radius_center must be 'location', 'bh', or 'external_bh'."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (rcrit0.use_radius_gate &&
          (rcrit0.radius_min < 0.0 || rcrit0.radius_max <= rcrit0.radius_min)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << it->block_name
                  << ">/radius bounds must satisfy 0 <= radius_min < radius_max."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Input uses the same physical AMR level convention as <refined_region>/level:
      // root grid is level 0, while MeshBlock levels are stored internally as logical
      // levels offset by pm->root_level.
      const int physical_max_level =
          pin->GetOrAddInteger(it->block_name, "max_level",
                               pm->max_level - pm->root_level);
      rcrit0.max_ref_level = physical_max_level + pm->root_level;
      if (rcrit0.max_ref_level < pm->root_level || rcrit0.max_ref_level > pm->max_level) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<" << it->block_name
                  << ">/max_level must be between 0 and "
                  << pm->max_level - pm->root_level << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      rcrit0.rloc_x1  = pin->GetOrAddReal(it->block_name,"location_x1", 0.0);
      rcrit0.rloc_x2  = pin->GetOrAddReal(it->block_name,"location_x2", 0.0);
      rcrit0.rloc_x3  = pin->GetOrAddReal(it->block_name,"location_x3", 0.0);
      rcrit0.rloc_rad = pin->GetOrAddReal(it->block_name,"location_rad", 0.0);
      rcrit0.slope_floor_frac = 0.0;
      rcrit.emplace_back(rcrit0);
    }
  }
  ncriteria = rcrit.size();

  // Error if there were no <amr_criterion> blocks
  if (ncriteria==0) {
    std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
    Kokkos::abort("No <amr_criterion> blocks were found in input file");
  }

  // Error if class containing variable requested has not been initialized
  for (auto it = rcrit.begin(); it != rcrit.end(); ++it) {
    if ((it->rvariable.compare(0, 5, "hydro") == 0) &&
        (pm->pmb_pack->phydro == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("Hydro refinement variable used but <hydro> not defined");
    }
    if ((it->rvariable.compare(0, 3, "mhd") == 0) &&
        (pm->pmb_pack->pmhd == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("MHD refinement variable used but <mhd> not defined");
    }
    if ((it->rvariable.compare(0, 3, "rad") == 0) &&
        (pm->pmb_pack->prad == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("radiation refinement variable used but <radiation> not defined");
    }
  }

  // count number of derived variables used for refinement
  // This is necessary to figure out dimensions needed for dvars array
  nderived = 0;
  SetRefinementData(pm->pmb_pack, true, false);

  if (nderived > 0) {
    auto &indcs = pmy_mesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb = std::max(pm->pmb_pack->nmb_thispack, MeshBlockStorageReserve());
    Kokkos::realloc(dvars, nmb, nderived, ncells3, ncells2, ncells1);
  }

  // Set rdata array to shallow slice of target data
  SetRefinementData(pm->pmb_pack, false, false);
}

//----------------------------------------------------------------------------------------
// destructor

RefinementCriteria::~RefinementCriteria() {
}

bool RefinementCriteria::ResizeMeshBlockStorage(int nmb, bool exact) {
  if (nderived <= 0) return false;
  nmb = std::max(nmb, MeshBlockStorageReserve());
  auto &indcs = pmy_mesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  const bool need = ((exact ? (dvars.extent_int(0) != nmb) : (dvars.extent_int(0) < nmb)) ||
                     dvars.extent_int(1) != nderived ||
                     dvars.extent_int(2) != ncells3 ||
                     dvars.extent_int(3) != ncells2 ||
                     dvars.extent_int(4) != ncells1);
  if (need) {
    Kokkos::resize(dvars, nmb, nderived, ncells3, ncells2, ncells1);
  }
  return need;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::ReleaseMeshBlockStorage()
//! \brief free the derived-variable array between AMR checks (it is only read inside
//! CheckForRefinement, once every refinement_interval cycles; kept resident it cost
//! 0.37 MB per 32^3 block).  The criteria slices are re-pointed by SetRefinementData
//! before every use, so they are cleared here as well to drop their references.

void RefinementCriteria::ReleaseMeshBlockStorage() {
  if (nderived <= 0) return;
  for (auto it = rcrit.begin(); it != rcrit.end(); ++it) {
    it->rdata = DvceArray5DnSlice();
  }
  dvars = DvceArray5D<Real>();
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::SetRefinementData()
//! \brief Cycles through all criteria and load data

void RefinementCriteria::SetRefinementData(MeshBlockPack* pmbp, bool count_derived,
                                           bool load_derived) {
  int iderived = 0;  // current index of variable in dvars array
  for (auto it = rcrit.begin(); it != rcrit.end(); ++it) {
    // Only load data for methods that need it
    if ((it->rmethod != RefCritMethod::location) &&
        (it->rmethod != RefCritMethod::user)) {
      using Kokkos::ALL;
      if (it->use_density_min && !(count_derived) && !(load_derived)) {
        int n = static_cast<int>(IDN);
        if (it->rvariable.compare(0, 5, "hydro") == 0) {
          it->density_data = Kokkos::subview(pmbp->phydro->w0, ALL, n, ALL, ALL, ALL);
        } else if (it->rvariable.compare(0, 3, "mhd") == 0) {
          it->density_data = Kokkos::subview(pmbp->pmhd->w0, ALL, n, ALL, ALL, ALL);
        } else {
          std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
          Kokkos::abort("density_min in a <amr_criterion> requires a hydro or MHD variable");
        }
      }
      // hydro (lab-frame) density
      if (it->rvariable.compare("hydro_u_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->phydro->u0, ALL, n, ALL, ALL, ALL);
        }
      // hydro (rest-frame) density
      } else if (it->rvariable.compare("hydro_w_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->phydro->w0, ALL, n, ALL, ALL, ALL);
        }
      // hydro primitive internal energy selected for the active pressure field
      } else if (it->rvariable.compare("hydro_w_e") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IEN);
          it->rdata = Kokkos::subview(pmbp->phydro->w0, ALL, n, ALL, ALL, ALL);
        }
      // hydro auxiliary internal energy carried by the dual-energy formalism
      } else if (it->rvariable.compare("hydro_w_eaux") == 0) {
        if (!(pmbp->phydro->use_dual_energy)) {
          std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
          Kokkos::abort("hydro_w_eaux refinement variable requires <hydro>/dual_energy");
        }
        if (!(count_derived) && !(load_derived)) {
          int n = pmbp->phydro->dual_energy_idx;
          it->rdata = Kokkos::subview(pmbp->phydro->w0, ALL, n, ALL, ALL, ALL);
        }
      // mhd (lab-frame) density
      } else if (it->rvariable.compare("mhd_u_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->pmhd->u0, ALL, n, ALL, ALL, ALL);
        }
      // mhd (rest-frame) density
      } else if (it->rvariable.compare("mhd_w_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->pmhd->w0, ALL, n, ALL, ALL, ALL);
        }
      // mhd primitive internal energy selected for the active pressure field
      } else if (it->rvariable.compare("mhd_w_e") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IEN);
          it->rdata = Kokkos::subview(pmbp->pmhd->w0, ALL, n, ALL, ALL, ALL);
        }
      // mhd auxiliary internal energy carried by the dual-energy formalism
      } else if (it->rvariable.compare("mhd_w_eaux") == 0) {
        if (!(pmbp->pmhd->use_dual_energy)) {
          std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
          Kokkos::abort("mhd_w_eaux refinement variable requires <mhd>/dual_energy");
        }
        if (!(count_derived) && !(load_derived)) {
          int n = pmbp->pmhd->dual_energy_idx;
          it->rdata = Kokkos::subview(pmbp->pmhd->w0, ALL, n, ALL, ALL, ALL);
        }
      // derived hydro cell-centered velocity divergence
      } else if ((it->rvariable.compare("hydro_div_v") == 0) ||
                 (it->rvariable.compare("hydro_abs_div_v") == 0)) {
        if (count_derived) {
          nderived += 1;
        } else if (load_derived) {
          ComputeDerivedVariable(it->rvariable, iderived, pmbp, dvars);
          iderived += 1;
        } else {
          it->rdata = Kokkos::subview(dvars, ALL, iderived, ALL, ALL, ALL);
          iderived += 1;
        }
      // radiation coordinate frame energy density R^0^0
      } else if (it->rvariable.compare("rad_coord_e") == 0) {
        if (count_derived) {
          nderived += 1;
        } else if (load_derived) {
          ComputeDerivedVariable(it->rvariable, iderived, pmbp, dvars);
          iderived += 1;
        } else {
          it->rdata = Kokkos::subview(dvars, ALL, iderived, ALL, ALL, ALL);
          iderived += 1;
        }
      } else {
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
        Kokkos::abort("Unknown refinement variable requested in a <amr_criterion>");
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckMinMax()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! min/max of selected variable on device.  Variable is set in SetRefinementData().

void RefinementCriteria::CheckMinMax(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &refine_hold = pmbp->pmesh->pmr->refine_hold;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  auto &level = pmbp->pmb->mb_lev;
  auto &size = pmbp->pmb->mb_size;

  auto &valmax = crit.rvalue_max;
  auto &deref_valmax = crit.rderef_value_max;
  const int max_ref_level = crit.max_ref_level;
  bool refine_only = crit.refine_only;
  auto &q0 = crit.rdata;
  bool use_density_min = crit.use_density_min;
  Real density_min = crit.density_min;
  auto &rho0 = crit.density_data;
  bool use_radius_gate = crit.use_radius_gate;
  Real radius_center_x = crit.rloc_x1;
  Real radius_center_y = crit.rloc_x2;
  Real radius_center_z = crit.rloc_x3;
  if (use_radius_gate && crit.radius_center_bh) {
    bool bh_enabled = false;
    Real bh_x = 0.0, bh_y = 0.0, bh_z = 0.0;
    Real bh_mass = 0.0, bh_softening = 0.0, newton_g = 0.0;
    problem_runtime::GetExternalBHPotential(pmbp->pmesh->time, bh_enabled,
                                            bh_x, bh_y, bh_z, bh_mass, bh_softening,
                                            newton_g);
    if (bh_enabled) {
      radius_center_x = bh_x;
      radius_center_y = bh_y;
      radius_center_z = bh_z;
    }
  }
  Real radius_min2 = crit.radius_min * crit.radius_min;
  Real radius_max2 = crit.radius_max * crit.radius_max;
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_qmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& qmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        if (use_radius_gate) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
          Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
          Real r2 = SQR(x - radius_center_x) + SQR(y - radius_center_y) +
                    SQR(z - radius_center_z);
          if (r2 < radius_min2 || r2 > radius_max2) return;
        }
        if (use_density_min && rho0(m,k,j,i) <= density_min) return;
        qmax = fmax(q0(m,k,j,i), qmax);
      },Kokkos::Max<Real>(team_qmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      bool criterion_active = (team_qmax > -0.5*FLT_MAX);
      bool refine_active = criterion_active && team_qmax > valmax;
      if (refine_active && level.d_view(m) < max_ref_level) {flag = 1;}
      if (refine_active && level.d_view(m) == max_ref_level) {
        refine_hold.d_view(m+mbs) = 1;
      }
      if (criterion_active && !refine_only && level.d_view(m) <= max_ref_level &&
          (team_qmax < deref_valmax) && (flag == 0)) {flag = -1;}
    });
  }

  auto &valmin = crit.rvalue_min;
  auto &deref_valmin = crit.rderef_value_min;
  if (valmin > -(FLT_MAX)) {  // user has set a min value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_qmin= (FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& qmin) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        if (use_radius_gate) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
          Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
          Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
          Real r2 = SQR(x - radius_center_x) + SQR(y - radius_center_y) +
                    SQR(z - radius_center_z);
          if (r2 < radius_min2 || r2 > radius_max2) return;
        }
        if (use_density_min && rho0(m,k,j,i) <= density_min) return;
        qmin = fmin(q0(m,k,j,i), qmin);
      },Kokkos::Min<Real>(team_qmin));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      bool criterion_active = (team_qmin < 0.5*FLT_MAX);
      bool refine_active = criterion_active && team_qmin < valmin;
      if (refine_active && level.d_view(m) < max_ref_level) {flag = 1;}
      if (refine_active && level.d_view(m) == max_ref_level) {
        refine_hold.d_view(m+mbs) = 1;
      }
      if (criterion_active && !refine_only && level.d_view(m) <= max_ref_level &&
          (team_qmin > deref_valmin) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  refine_hold.template modify<DevExeSpace>();
  refine_hold.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckSlope()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! magnitude of normalized slope (dq/q) of selected variable on device.  Variable is set
//! in SetRefinementData().

void RefinementCriteria::CheckSlope(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &refine_hold = pmbp->pmesh->pmr->refine_hold;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  auto &level = pmbp->pmb->mb_lev;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  auto &valmax = crit.rvalue_max;
  auto &deref_valmax = crit.rderef_value_max;
  const int max_ref_level = crit.max_ref_level;
  bool refine_only = crit.refine_only;
  auto &q0 = crit.rdata;
  // 0 for every <amr_criterion> block; only <mesh_refinement>/drad_max sets it.  The
  // floor_frac == 0 path below evaluates the identical expression it always did, and
  // skips the extra reduction and its collective entirely.
  //
  // With floor_frac > 0 the denominator is softened by a fraction of the GLOBAL maximum
  // of the same variable, |grad q| dx / (q + frac*max_domain q).  The bare ratio is the
  // logarithmic slope, which for a field that decays to a vacuum floor keeps GROWING
  // out into the tail (a Gaussian's is |x-x0| dx/sigma^2) and so would pull refinement
  // away from the structure and into the emptiest part of the domain.  Adding a fixed
  // fraction of the global peak turns it into a contrast measure that vanishes where
  // the field is negligible, while leaving it the logarithmic slope wherever q is a
  // sizeable fraction of the peak.  A max-reduction is exact and order-independent, so
  // the softened measure is still bit-identical across rank counts.
  Real floor_frac = crit.slope_floor_frac;
  Real qref = 0.0;
  if (floor_frac > 0.0 && valmax < (FLT_MAX)) {
    Real qmax_all = -(FLT_MAX);
    Kokkos::parallel_reduce("RefCondQMax",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExeSpace(), {0,ks,js,is},
                                           {nmb,ks+nx3,js+nx2,is+nx1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &qmax) {
      qmax = fmax(q0(m,k,j,i), qmax);
    }, Kokkos::Max<Real>(qmax_all));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &qmax_all, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    qref = floor_frac*fmax(qmax_all, 0.0);
  }
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dqmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& dqmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        Real d2 = SQR(q0(m,k,j,i+1) - q0(m,k,j,i-1));
        if (multi_d) {d2 += SQR(q0(m,k,j+1,i) - q0(m,k,j-1,i));}
        if (three_d) {d2 += SQR(q0(m,k+1,j,i) - q0(m,k-1,j,i));}
        Real qden = (floor_frac > 0.0) ? (q0(m,k,j,i) + qref) : q0(m,k,j,i);
        dqmax = fmax((0.5*sqrt(d2)/qden), dqmax);
      },Kokkos::Max<Real>(team_dqmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_dqmax > valmax && level.d_view(m) < max_ref_level) {flag = 1;}
      if (team_dqmax > valmax && level.d_view(m) == max_ref_level) {
        refine_hold.d_view(m+mbs) = 1;
      }
      if (!refine_only && level.d_view(m) <= max_ref_level &&
          (team_dqmax < deref_valmax) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  refine_hold.template modify<DevExeSpace>();
  refine_hold.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckSecondDeriv()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! magnitude of normalized second derivative (d2q/q) of selected variable on device.
//! Variable is set in SetRefinementData().

void RefinementCriteria::CheckSecondDeriv(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &refine_hold = pmbp->pmesh->pmr->refine_hold;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  auto &level = pmbp->pmb->mb_lev;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  auto &valmax = crit.rvalue_max;
  auto &deref_valmax = crit.rderef_value_max;
  const int max_ref_level = crit.max_ref_level;
  bool refine_only = crit.refine_only;
  auto &q0 = crit.rdata;
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_d2qmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& d2qmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        Real d2q = q0(m,k,j,i+1) - 2.0*q0(m,k,j,i) + q0(m,k,j,i-1);
        if (multi_d) {d2q += (q0(m,k,j+1,i) - 2.0*q0(m,k,j,i) + q0(m,k,j-1,i));}
        if (three_d) {d2q += (q0(m,k+1,j,i) - 2.0*q0(m,k,j,i) + q0(m,k-1,j,i));}
        d2qmax = fmax((fabs(d2q)/q0(m,k,j,i)), d2qmax);
      },Kokkos::Max<Real>(team_d2qmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_d2qmax > valmax && level.d_view(m) < max_ref_level) {flag = 1;}
      if (team_d2qmax > valmax && level.d_view(m) == max_ref_level) {
        refine_hold.d_view(m+mbs) = 1;
      }
      if (!refine_only && level.d_view(m) <= max_ref_level &&
          (team_d2qmax < deref_valmax) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  refine_hold.template modify<DevExeSpace>();
  refine_hold.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckLocation()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! whether any part of MeshBlock is within given radius of a given position

void RefinementCriteria::CheckLocation(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &refine_hold = pmbp->pmesh->pmr->refine_hold;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &level = pmbp->pmb->mb_lev;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  Real &x1 = crit.rloc_x1;
  Real &x2 = crit.rloc_x2;
  Real &x3 = crit.rloc_x3;
  Real &rad = crit.rloc_rad;
  const int max_ref_level = crit.max_ref_level;
  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    if (((x1min < (x1+rad)) && (x1min > (x1-rad))) ||
        ((x1max < (x1+rad)) && (x1max > (x1-rad))) ||
        ((x1max > (x1+rad)) && (x1min < (x1-rad)))) {
      if (!(multi_d) ||
          (((x2min < (x2+rad)) && (x2min > (x2-rad))) ||
           ((x2max < (x2+rad)) && (x2max > (x2-rad))) ||
           ((x2max > (x2+rad)) && (x2min < (x2-rad)))) ) {
        if (!(three_d) ||
            (((x3min < (x3+rad)) && (x3min > (x3-rad))) ||
             ((x3max < (x3+rad)) && (x3max > (x3-rad))) ||
             ((x3max > (x3+rad)) && (x3min < (x3-rad)))) ) {
          if (level.h_view(m) < max_ref_level) {
            refine_flag.h_view(m + mbs) = 1;
          } else if (level.h_view(m) == max_ref_level) {
            refine_hold.h_view(m + mbs) = 1;
          }
        }
      }
    }
  }
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  refine_hold.template modify<HostMemSpace>();
  refine_hold.template sync<DevExeSpace>();
  return;
}
