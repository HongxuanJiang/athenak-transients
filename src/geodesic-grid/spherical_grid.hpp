#ifndef GEODESIC_GRID_SPHERICAL_GRID_HPP_
#define GEODESIC_GRID_SPHERICAL_GRID_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_grid.hpp
//  \brief definitions for SphericalGrid class

#include <cstdint>

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class SphericalGrid

class SphericalGrid: public GeodesicGrid {
 public:
    // Creates a geodesic grid with refinement level nlev and radius rad
    SphericalGrid(MeshBlockPack *pmy_pack, int nlev, Real rad, int ninterp = -1);
    ~SphericalGrid();

    Real radius;  // const radius for SphericalGrid
    int ninterp;  // number of interpolation points along each dimension
    DualArray2D<Real> interp_coord;  // Cartesian coordinates for grid points
    DualArray2D<Real> interp_vals;   // container for data interpolated to sphere
    void InterpolateToSphere(int nvars, DvceArray5D<Real>& val);  // interpolate to sphere
    // interpolate a range of variables to a sphere
    void InterpolateToSphere(int vs, int ve, DvceArray5D<Real>& val);
    // move the sphere to Euclidean radius rad about center (a surface that follows a
    // moving black hole); the (block, cell) stencil is rebuilt on the next interpolation
    void ResetCenterAndRadius(const Real center[3], Real rad);
    // MeshBlock (index in this rank's pack) whose stencil serves angle n after the last
    // interpolation, or -1 when the angle belongs to another rank
    int OwnerMeshBlock(int n) const { return interp_indcs.h_view(n,0); }

 private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
    // Mesh::topology_version the cached interpolation indices/weights were built
    // against.  Any topology or load-balance transaction (adaptive AMR, LAT rebalancing)
    // invalidates them, as does moving the sphere (ResetCenterAndRadius).
    std::uint64_t interp_topology_version_ = ~std::uint64_t{0};
    DualArray2D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
    DualArray3D<Real> interp_wghts;  // weights for interpolation
    void SetInterpolationCoordinates();  // set indexing for interpolation
    void SetInterpolationIndices();      // set indexing for interpolation
    void SetInterpolationWeights();      // set weights for interpolation
};

#endif // GEODESIC_GRID_SPHERICAL_GRID_HPP_
