#ifndef BVALS_NARROW_BLOCKS_HPP_
#define BVALS_NARROW_BLOCKS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file narrow_blocks.hpp
//! \brief MeshBlocks narrower than 2*nghost on a multilevel mesh pass ghost cells on.

#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn bool NarrowMeshBlocks()
//! \brief A MeshBlock sends a coarser neighbour ng coarse cells normal to their shared
//! face, and a finer one ng fine cells beyond the edge of the finer block's sub-face
//! (MeshBoundaryValuesCC/FC::InitSendIndices).  When its coarse width nx/2 is below ng,
//! part of each range lies in its own ghost zones: it passes on cells its exchange
//! received (or restricted from what it received) from the neighbours on the far side.

inline bool NarrowMeshBlocks(const Mesh *pm) {
  const RegionIndcs &indcs = pm->mb_indcs;
  return pm->multilevel && ((indcs.cnx1 < indcs.ng) ||
                            (pm->multi_d && (indcs.cnx2 < indcs.ng)) ||
                            (pm->three_d && (indcs.cnx3 < indcs.ng)));
}

#endif // BVALS_NARROW_BLOCKS_HPP_
