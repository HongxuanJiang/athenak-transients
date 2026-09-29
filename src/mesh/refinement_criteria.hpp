#ifndef MESH_REFINEMENT_CRITERIA_HPP_
#define MESH_REFINEMENT_CRITERIA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refinement_criteria.hpp
//! \brief defines RefinementCriteria class containing data and functions controlling
//! how mesh is refined/derefined with AMR
//! This class implements default refinement conditions:
//!   (1) min/max of selected variable
//!   (2) gradient of selected variable
//!   (3) second derivative of selected variable
//!   (4) region with specified radius of a selected point
//! Any number of refinement criteria can be specified using multiple
//! <refinement_criteriaN> blocks in the input file.  Each block can select a different
//! method and/or hydro/MHD/radiation variables can be selected.
//! TODO(@JMS): user-defined variables can also be selected
//!
//! User-defined refinement conditions can also be enrolled by setting the *usr_ref_func
//! pointer in the problem generator.

#include <string>
#include <vector>

#include "athena.hpp"

// identifiers for refinement criteria methods
enum class RefCritMethod {min_max, slope, second_deriv, location, user};

using DvceArray5DnSlice = Kokkos::Subview<DvceArray5D<Real>,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          int,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          std::remove_const_t<decltype(Kokkos::ALL)>>;

//----------------------------------------------------------------------------------------
//! \struct RefinementCriteriaData
//! \brief physical size in a Mesh or a MeshBlock

struct RefCritData {
  RefCritMethod rmethod;           // refinement method (min_max, slope, etc.)
  std::string rvariable;           // name of variable to be tested for refinement
  Real rvalue_min, rvalue_max;     // min/max criteria for refinement
  Real rderef_value_min;           // min criterion for derefinement hysteresis
  Real rderef_value_max;           // max criterion for derefinement hysteresis
  bool refine_only;                // if true, criterion can refine but not derefine
  int max_ref_level;               // maximum logical level this criterion may control
  bool use_density_min;            // if true, apply criterion only above density_min
  Real density_min;                // minimum rest-frame density for criterion to apply
  bool use_radius_gate;            // if true, apply criterion only inside radius bounds
  bool radius_center_bh;           // if true, center radius gate on runtime BH position
  Real radius_min, radius_max;     // radial bounds for applying this criterion
  Real rloc_x1, rloc_x2, rloc_x3;  // x1-,x2-,x3-locations of point to refine around
  Real rloc_rad;                   // radius of region around point to be refined
  // Optional softening of the "slope" denominator, as a fraction of the GLOBAL maximum
  // of the same variable: |grad q| dx / (q + frac*max_domain(q)).  0 (the default, and
  // the only value any <amr_criterion> block can produce) leaves the denominator as the
  // bare q of the original criterion, bit for bit.  Set only by the
  // <mesh_refinement>/drad_max radiation criterion, whose variable spans many decades
  // down to a vacuum floor where the bare logarithmic slope keeps growing outwards and
  // carries no information about where the radiation field actually is.
  Real slope_floor_frac;
  DvceArray5DnSlice rdata;         // slice of variable "n" in 5D array(m,n,k,j,i)
  DvceArray5DnSlice density_data;  // density slice used by optional density_min gate
};

//----------------------------------------------------------------------------------------
//! \class RefinementCriteria
//! \brief data/functions associated with various refinement criteria for AMR

class RefinementCriteria {
 public:
  RefinementCriteria(Mesh *pm, ParameterInput *pin);
  ~RefinementCriteria();

  // data
  int ncriteria;
  int nderived;
  std::vector<RefCritData> rcrit;

  // functions
  void SetRefinementData(MeshBlockPack* pmbp, bool count, bool load);
  void CheckMinMax(MeshBlockPack* pmbp, RefCritData crit);
  void CheckSlope(MeshBlockPack* pmbp, RefCritData crit);
  void CheckSecondDeriv(MeshBlockPack* pmbp, RefCritData crit);
  void CheckLocation(MeshBlockPack* pmbp, RefCritData crit);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false);
  void ReleaseMeshBlockStorage();

 private:
  // data
  Mesh *pmy_mesh;
  DvceArray5D<Real> dvars;  // derived variables
};
#endif // MESH_REFINEMENT_CRITERIA_HPP_
