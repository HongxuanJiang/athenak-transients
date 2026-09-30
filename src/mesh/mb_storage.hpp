#ifndef MESH_MB_STORAGE_HPP_
#define MESH_MB_STORAGE_HPP_
//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mb_storage.hpp
//! \brief sizing policy for the per-MeshBlock field arrays
//!
//! The physics modules size their 5D field arrays by the local MeshBlock count, which AMR
//! changes on (potentially) every cycle.  Two failure modes bracket the choice of policy:
//!
//!   * Resizing to the exact count reallocates several hundred MiB of device memory every
//!     time the count moves by one block.  With refinement_interval = 1 that is thousands
//!     of reallocations per run, and it fragments the CUDA heap until a modest contiguous
//!     request fails.  A production star_bh_collision run died on exactly that: "Cuda
//!     memory space failed to allocate 64 MiB (label=\"lb send stage\")" after 14k cycles,
//!     while a restart of the same state had 6.5 GiB spare.
//!
//!   * Never shrinking (the previous policy) makes device memory track the per-rank
//!     HIGH-WATER block count instead of the current one.  LAT balances by cost rather
//!     than by block count, so a rank can transiently be handed far more blocks than the
//!     average, and that peak then becomes permanent.  Measured on the same problem:
//!     memory ratcheted 9832 -> 16006 MiB in one hour while the per-rank block count was
//!     FLAT at 231 -> 214, i.e. the allocation was sized for ~418 blocks/rank to hold 214.
//!
//! The policy below grows immediately, rounds up so an AMR ramp does not reallocate on
//! every single block gained, and shrinks only once the working set has fallen a full
//! hysteresis band below the current allocation.

#include <algorithm>

//! Granularity of the block dimension.  Rounding up costs at most (kMBStorageGrain - 1)
//! blocks of slack and keeps small AMR jitter from touching the allocation at all.
constexpr int kMBStorageGrain = 16;

//! Opt-in preallocation (<mesh_refinement>/preallocate = true).  When positive, every
//! per-MeshBlock array is sized once for this many blocks and is never shrunk.  Zero
//! (the default) leaves the policy below untouched.
inline int g_mb_storage_reserve = 0;
inline void SetMeshBlockStorageReserve(int nmb) {
  g_mb_storage_reserve = std::max(nmb, 0);
}
inline int MeshBlockStorageReserve() { return g_mb_storage_reserve; }

//----------------------------------------------------------------------------------------
//! \fn int MeshBlockStorageCapacity(int nmb_needed)
//! \brief block extent to allocate in order to hold nmb_needed blocks

inline int MeshBlockStorageCapacity(int nmb_needed) {
  const int n = std::max(nmb_needed, 1);
  return std::max(((n + kMBStorageGrain - 1)/kMBStorageGrain)*kMBStorageGrain,
                  g_mb_storage_reserve);
}

//----------------------------------------------------------------------------------------
//! \fn bool MeshBlockStorageShouldResize(int have, int need, bool allow_shrink)
//! \brief whether storage currently sized for `have` blocks must be reallocated
//!
//! Growth is mandatory.  Shrinking is optional and is only ever requested after a
//! topology transaction has finished placing data, because the pre-migration resize still
//! has to keep the outgoing blocks addressable for PackAndSendAMR.

inline bool MeshBlockStorageShouldResize(int have, int need, bool allow_shrink) {
  if (g_mb_storage_reserve > 0) return have < std::max(need, g_mb_storage_reserve);
  if (need > have) return true;
  if (!allow_shrink) return false;
  // Shrink as soon as two grains of dead block storage exist (~17 MB per 32^3 hydro
  // block each).  One grain of hysteresis is kept so that a count jittering across a
  // grain boundary on every-cycle AMR cannot alternate shrink/grow reallocations, and
  // an exactly-sized allocation (have not a multiple of the grain) is never grown here.
  return have - MeshBlockStorageCapacity(need) >= 2*kMBStorageGrain;
}

#endif // MESH_MB_STORAGE_HPP_
