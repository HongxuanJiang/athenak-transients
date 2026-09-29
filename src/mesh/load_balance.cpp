//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//! \brief Contains various Mesh and MeshRefinement functions associated with
//! load balancing when MPI is used, both for uniform grids and with SMR/AMR.

#include <iostream>
#include <limits> // numeric_limits<>
#include <map>
#include <algorithm> // max
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility> // make_pair
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "gravity/gravity.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
struct LogicalLocationKey {
  std::int32_t lx1, lx2, lx3, level;

  explicit LogicalLocationKey(const LogicalLocation &loc)
      : lx1(loc.lx1), lx2(loc.lx2), lx3(loc.lx3), level(loc.level) {}

  LogicalLocationKey(std::int32_t x1, std::int32_t x2, std::int32_t x3,
                     std::int32_t lev)
      : lx1(x1), lx2(x2), lx3(x3), level(lev) {}

  bool operator==(const LogicalLocationKey &other) const {
    return lx1 == other.lx1 && lx2 == other.lx2 && lx3 == other.lx3 &&
           level == other.level;
  }
};

struct LogicalLocationKeyHash {
  std::size_t operator()(const LogicalLocationKey &key) const {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint32_t v) {
      h ^= static_cast<std::uint64_t>(v);
      h *= 1099511628211ull;
    };
    mix(static_cast<std::uint32_t>(key.lx1));
    mix(static_cast<std::uint32_t>(key.lx2));
    mix(static_cast<std::uint32_t>(key.lx3));
    mix(static_cast<std::uint32_t>(key.level));
    return static_cast<std::size_t>(h);
  }
};

LogicalLocationKey ParentKey(LogicalLocationKey key) {
  if (key.level > 0) {
    key.lx1 >>= 1;
    key.lx2 >>= 1;
    key.lx3 >>= 1;
    --key.level;
  }
  return key;
}

#if MPI_PARALLEL_ENABLED
// Return an exclusive descriptor bound whose packed extent fits in the staging target.
// The current pack/unpack kernels keep each MPI descriptor in one device chunk.
int AMRChunkEnd(const DualArray1D<AMRBuffer> &buffers, int begin, int nbuffer,
                std::size_t stage_elements) {
  const std::size_t data_offset = buffers.h_view(begin).offset;
  int end = begin + 1;
  while (end < nbuffer) {
    const std::size_t candidate =
        buffers.h_view(end).offset + buffers.h_view(end).cnt - data_offset;
    if (candidate > stage_elements) break;
    ++end;
  }
  return end;
}

std::size_t AMRChunkElements(const DualArray1D<AMRBuffer> &buffers, int begin, int end) {
  return buffers.h_view(end - 1).offset + buffers.h_view(end - 1).cnt -
         buffers.h_view(begin).offset;
}

// Round a required staging extent up to a whole number of MiB.  The staging buffer is a
// grow-only member, so the quantum keeps a marginally larger transaction from forcing a
// reallocation.  Over-allocation is harmless: every pack/unpack kernel indexes the buffer
// as (descriptor offset - data_offset), bounded by the chunk decomposition, and the only
// bulk copies are subviews cut to the live chunk extent.
std::size_t AMRStagingCapacity(std::size_t need_elements) {
  const std::size_t quantum = (std::size_t{1024}*1024)/sizeof(Real);  // 1 MiB in Reals
  const std::size_t nquanta = (need_elements + quantum - 1)/quantum;
  return std::max(nquanta, std::size_t{1})*quantum;
}

std::size_t AMRStagingExtent(const DualArray1D<AMRBuffer> &buffers, int nbuffer,
                             std::size_t target_elements) {
  const std::size_t total_elements =
      buffers.h_view(nbuffer - 1).offset + buffers.h_view(nbuffer - 1).cnt;
  std::size_t largest_message = 1;
  for (int n = 0; n < nbuffer; ++n) {
    largest_message = std::max(largest_message,
                               static_cast<std::size_t>(buffers.h_view(n).cnt));
  }
  return std::max(largest_message, std::min(target_elements, total_elements));
}

int AMRMessageCount(int ncc, int nfc, int cntcc, int cntfc) {
  const std::int64_t count = static_cast<std::int64_t>(ncc)*cntcc +
                             static_cast<std::int64_t>(nfc)*cntfc;
  if (count < 0 || count > std::numeric_limits<int>::max()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "AMR load-balance message exceeds the MPI integer count limit."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(count);
}

void ValidateAMRDescriptorCount(const char *direction, int actual, int expected) {
  if (actual == expected) return;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "AMR load-balance " << direction
            << " descriptor count mismatch: built=" << actual
            << ", expected=" << expected << std::endl;
  std::exit(EXIT_FAILURE);
}

std::vector<AMRRankMessage> BuildAMRRankMessages(
    const DualArray1D<AMRBuffer> &buffers, const std::vector<int> &peer_ranks,
    const std::vector<int> &keys,
    int nbuffer, const char *direction) {
  if (static_cast<int>(peer_ranks.size()) != nbuffer ||
      static_cast<int>(keys.size()) != nbuffer) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "AMR load-balance " << direction
              << " peer metadata count mismatch" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<AMRRankMessage> messages;
  if (nbuffer <= 0) return messages;
  std::map<int, std::vector<int>> by_rank;
  for (int n=0; n<nbuffer; ++n) by_rank[peer_ranks[n]].push_back(n);
  for (auto &peer : by_rank) {
    auto &indices = peer.second;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return keys[a] < keys[b];
    });
    for (std::size_t i=1; i<indices.size(); ++i) {
      if (keys[indices[i - 1]] == keys[indices[i]]) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "AMR rank-packed " << direction
                  << " has duplicate peer ordering key " << keys[indices[i]]
                  << " for rank " << peer.first << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    std::vector<int> counts(indices.size());
    std::vector<MPI_Aint> displacements(indices.size());
    MPI_Count payload_count = 0;
    for (std::size_t i=0; i<indices.size(); ++i) {
      const int index = indices[i];
      counts[i] = buffers.h_view(index).cnt;
      displacements[i] = static_cast<MPI_Aint>(buffers.h_view(index).offset)*
                         static_cast<MPI_Aint>(sizeof(Real));
      payload_count += counts[i];
    }
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    int ierr = MPI_Type_create_hindexed(static_cast<int>(indices.size()), counts.data(),
                                        displacements.data(), MPI_ATHENA_REAL, &datatype);
    if (ierr == MPI_SUCCESS) ierr = MPI_Type_commit(&datatype);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error building rank-packed AMR " << direction
                << " datatype" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    messages.push_back(AMRRankMessage{peer.first, payload_count, datatype,
                                      MPI_REQUEST_NULL});
  }
  return messages;
}
#endif
} // namespace

//----------------------------------------------------------------------------------------
int Mesh::ApplyHydroLATLoadBalanceCosts(ParameterInput *pin, LogicalLocation *lloc_list,
                                        float *cost_list, int nb, int *lat_factor_list,
                                        int *lat_sync_factor, bool costs_measured) {
  if (lat_sync_factor != nullptr) *lat_sync_factor = 1;
  if (pin == nullptr || cost_list == nullptr || nb <= 0) return 0;
  const bool hydro_lat = pin->IsLATEnabled();
  auto configured_amr_block_cap = [&]() {
    if (pin->DoesParameterExist("mesh_refinement", "max_nmb_per_rank")) {
      const int cap = pin->GetInteger("mesh_refinement", "max_nmb_per_rank");
      if (cap > 0) return cap;
    }
    return 0;
  };
  if (!hydro_lat) {
    // Restart files may carry stale cost data from a previous LAT partition.
    // With LAT disabled, recover the original block-count balance and leave
    // LoadBalance on the standard Z-order scalar path used by the non-LAT code.
    for (int i=0; i<nb; ++i) {
      cost_list[i] = 1.0;
      if (lat_factor_list != nullptr) lat_factor_list[i] = 1;
    }
    return 0;
  }

  int configured_levels = 1;
  if (pin->DoesParameterExist("time", "lat_levels")) {
    configured_levels = pin->GetInteger("time", "lat_levels");
  }
  configured_levels = std::max(1, std::min(configured_levels, 20));
  int configured_max_factor = 1;
  for (int n=0; n<configured_levels; ++n) configured_max_factor *= 2;

  int max_present_factor = 1;
  const bool have_exact_lat_metadata =
      hydro_lat_metadata_valid && hydro_lat_factor_eachmb != nullptr &&
      hydro_lat_metadata_nmb == nb &&
      (lloc_list == nullptr || lloc_list == lloc_eachmb);
  const bool have_amr_estimated_metadata =
      (!have_exact_lat_metadata) && hydro_lat_metadata_valid &&
      hydro_lat_factor_eachmb != nullptr && hydro_lat_metadata_nmb == nmb_total &&
      lloc_eachmb != nullptr && lloc_list != nullptr && nmb_total > 0;
  std::unordered_map<LogicalLocationKey, int, LogicalLocationKeyHash> factor_by_location;
  std::unordered_map<LogicalLocationKey, int, LogicalLocationKeyHash> min_descendant_factor;
  // The measured per-step work travels with the factor: by gid when the metadata is
  // exact, by logical location for a block that survives unchanged (a pure rebalance is
  // all such blocks), and 1 for a block that is new or re-levelled, since nothing has
  // been measured on it yet.
  std::unordered_map<LogicalLocationKey, Real, LogicalLocationKeyHash> work_by_location;
  std::vector<Real> work_list(nb, 1.0);
  if (have_amr_estimated_metadata) {
    factor_by_location.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    min_descendant_factor.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    if (hydro_lat_work_eachmb != nullptr) {
      work_by_location.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    }
    for (int old_gid=0; old_gid<hydro_lat_metadata_nmb; ++old_gid) {
      const LogicalLocation &old_lloc = lloc_eachmb[old_gid];
      const int old_factor = std::max(1, hydro_lat_factor_eachmb[old_gid]);
      factor_by_location.emplace(LogicalLocationKey(old_lloc), old_factor);
      if (hydro_lat_work_eachmb != nullptr) {
        work_by_location.emplace(LogicalLocationKey(old_lloc),
                                 hydro_lat_work_eachmb[old_gid]);
      }
      LogicalLocationKey ancestor(old_lloc);
      int depth_to_ancestor = 0;
      while (ancestor.level > 0) {
        ancestor = ParentKey(ancestor);
        ++depth_to_ancestor;
        int ancestor_factor = old_factor;
        for (int n=0; n<depth_to_ancestor && ancestor_factor<configured_max_factor; ++n) {
          ancestor_factor = std::min(configured_max_factor, 2*ancestor_factor);
        }
        auto it = min_descendant_factor.find(ancestor);
        if (it == min_descendant_factor.end()) {
          min_descendant_factor.emplace(ancestor, ancestor_factor);
        } else {
          it->second = std::min(it->second, ancestor_factor);
        }
      }
    }
  }
  auto estimate_factor = [&](const LogicalLocation &new_lloc) {
    int factor_estimate = 0;
    auto clamp_factor = [&](int factor) {
      return std::max(1, std::min(configured_max_factor, factor));
    };

    // Direct carry-over for unchanged MeshBlocks.
    const auto direct = factor_by_location.find(LogicalLocationKey(new_lloc));
    if (direct != factor_by_location.end()) {
      factor_estimate = direct->second;
    }

    // New refined children inherit a conservative half-step factor from the nearest
    // old ancestor, matching the AMR 2:1 timestep expectation without scanning all MBs.
    LogicalLocationKey ancestor(new_lloc);
    int depth_from_ancestor = 0;
    while (factor_estimate == 0 && ancestor.level > 0) {
      ancestor = ParentKey(ancestor);
      ++depth_from_ancestor;
      const auto it = factor_by_location.find(ancestor);
      if (it == factor_by_location.end()) continue;
      int factor = it->second;
      for (int n=0; n<depth_from_ancestor && factor>1; ++n) factor /= 2;
      factor_estimate = factor;
    }

    // Newly derefined parents inherit the minimum descendant factor, adjusted by
    // one factor increase per coarsening level.
    if (factor_estimate == 0) {
      const auto descendant = min_descendant_factor.find(LogicalLocationKey(new_lloc));
      if (descendant != min_descendant_factor.end()) {
        factor_estimate = descendant->second;
      }
    }

    return clamp_factor((factor_estimate > 0) ? factor_estimate : 1);
  };
  // Precompute max AMR level for level-based factor estimation fallback
  int max_level_present = root_level;
  if (!have_exact_lat_metadata && !have_amr_estimated_metadata && lloc_list != nullptr) {
    for (int i=0; i<nb; ++i) {
      max_level_present = std::max(max_level_present,
                                   static_cast<int>(lloc_list[i].level));
    }
  }
  for (int i=0; i<nb; ++i) {
    int factor = 1;
    if (have_exact_lat_metadata) {
      factor = std::max(1, hydro_lat_factor_eachmb[i]);
      if (hydro_lat_work_eachmb != nullptr) work_list[i] = hydro_lat_work_eachmb[i];
    } else if (have_amr_estimated_metadata) {
      factor = estimate_factor(lloc_list[i]);
      const auto work = work_by_location.find(LogicalLocationKey(lloc_list[i]));
      if (work != work_by_location.end()) work_list[i] = work->second;
    } else if (lloc_list != nullptr && !costs_measured) {
      // Level-based estimation fallback for a cold start:
      // No CFL metadata available yet. Estimate factors from AMR level alone.
      // Finest level gets factor=1, each coarser level doubles the factor.
      const int level_diff = max_level_present - lloc_list[i].level;
      factor = 1;
      for (int n=0; n<level_diff && factor<configured_max_factor; ++n) {
        factor = std::min(configured_max_factor, factor*2);
      }
    }
    if (lat_factor_list != nullptr) lat_factor_list[i] = factor;
    max_present_factor = std::max(max_present_factor, factor);
  }
  const bool have_level_estimated_metadata =
      (!have_exact_lat_metadata) && (!have_amr_estimated_metadata) &&
      lloc_list != nullptr && max_present_factor > 1;
  const bool have_lat_metadata = have_exact_lat_metadata ||
      have_amr_estimated_metadata || have_level_estimated_metadata;
  if (have_lat_metadata && lat_factor_list != nullptr) {
    // Same rank-dependent default as Mesh::UpdateHydroLATMetadata; see the comment there
    // for why no rank-independent constant can replace it.
    int min_bin_count = 4*global_variable::nranks;
    if (pin->DoesParameterExist("time", "hydro_lat_min_bin_count")) {
      min_bin_count = pin->GetInteger("time", "hydro_lat_min_bin_count");
      if (min_bin_count < -1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "time/hydro_lat_min_bin_count must be >= 0, or -1 for the default."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (min_bin_count == -1) min_bin_count = 4*global_variable::nranks;
    }
    min_bin_count = std::max(0, min_bin_count);
    if (min_bin_count > 0) {
      bool changed = true;
      while (changed) {
        changed = false;
        for (int factor=max_present_factor; factor>1; factor/=2) {
          int count = 0;
          for (int i=0; i<nb; ++i) {
            if (lat_factor_list[i] == factor) ++count;
          }
          if (count > 0 && count < min_bin_count) {
            const int clamped_factor = std::max(1, factor/2);
            for (int i=0; i<nb; ++i) {
              if (lat_factor_list[i] == factor) lat_factor_list[i] = clamped_factor;
            }
            changed = true;
          }
        }

        max_present_factor = 1;
        for (int i=0; i<nb; ++i) {
          max_present_factor = std::max(max_present_factor, lat_factor_list[i]);
        }
      }
    }
  }
  // A checkpoint carries the writing run's own per-block costs, and on a restart they are
  // the only true cost that exists: this run has measured nothing yet.  Keep them, and
  // do not put the level-based estimate in their place -- that estimate is pure geometry,
  // it reads 2^(max_level - level) off the tree and knows nothing about what actually
  // sets the local timestep, so on a deck whose dt does not scale with dx (the star-BH
  // deck: the softened BH CFL sets one dt for every block, and the live factors are 1
  // everywhere) it invents a ladder up to 2^time/lat_levels and a sync factor to match.
  // LoadBalance then runs its bin-aware partitioner over a tick schedule that does not
  // exist, which is worse than no bin information at all.  The per-block loop above
  // already leaves the ladder flat on this path; the first Mesh::UpdateHydroLATMetadata
  // of the restarted run installs the real one, which is how the writing run reached the
  // partition the checkpoint carries.
  const bool keep_restart_costs =
      costs_measured && !have_exact_lat_metadata && !have_amr_estimated_metadata;
  if (keep_restart_costs) {
    for (int i=0; i<nb; ++i) {
      if (!std::isfinite(cost_list[i]) || !(cost_list[i] > 0.0f)) cost_list[i] = 1.0;
    }
  } else if (have_lat_metadata) {
    // Window cost of a block = steps per window x measured cost of one of its steps
    // (Mesh::hydro_lat_work_eachmb, 1 where nothing was measured).  LoadBalance recovers
    // the per-step part as cost*factor/sync for its tick-class objective.
    for (int i=0; i<nb; ++i) {
      const int factor = (lat_factor_list != nullptr) ? std::max(1,
          lat_factor_list[i]) : 1;
      cost_list[i] = static_cast<float>(
          static_cast<Real>(max_present_factor/factor)*
          std::max(work_list[i], static_cast<Real>(1.0)));
    }
  } else {
    // Restart files can carry stale per-block costs from a non-LAT or older LAT
    // partition.  Before local-CFL metadata exists, balance by block count so the
    // memory cap is an actual per-rank allocation bound instead of packing many
    // cheap coarse blocks onto one GPU.
    for (int i=0; i<nb; ++i) {
      cost_list[i] = 1.0;
    }
  }
  if (lat_sync_factor != nullptr) *lat_sync_factor = max_present_factor;

  const int nranks = std::max(1, global_variable::nranks);
  const int min_cap = (nb + nranks - 1)/nranks;
  // One cap only: the allocation bound mesh_refinement/max_nmb_per_rank.  A separate,
  // lower LAT partition cap (the former time/hydro_lat_max_nmb_per_rank) could not be
  // honoured once AMR filled the blocks between the two caps and aborted the run.
  int cap = adaptive ? configured_amr_block_cap() : 0;
  if (cap <= 0) cap = min_cap;
  if (cap < min_cap || static_cast<long long>(cap)*static_cast<long long>(nranks) < nb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "HD LAT load-balance MeshBlocks/rank cap cannot hold current mesh: cap="
              << cap << ", minimum=" << min_cap
              << ", nranks=" << nranks << ", MeshBlocks=" << nb << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (cap > (1 << NUM_BITS_LID)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "HD LAT load-balance memory cap exceeds MPI tag local-id capacity: cap="
              << cap << ", limit=" << (1 << NUM_BITS_LID) << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return cap;
}

//----------------------------------------------------------------------------------------
bool Mesh::BuildHydroLATGIDMap(ParameterInput *pin, const LogicalLocation *lloc_list,
                               int nb, const int *lat_factor_list, int lat_sync_factor,
                               int *newtoold, bool input_already_reordered,
                               const Real *work_list) const {
  if (newtoold == nullptr || lloc_list == nullptr || nb <= 0) return false;
  for (int gid=0; gid<nb; ++gid) newtoold[gid] = gid;

  auto child_index = [&](const LogicalLocation &loc, int depth) {
    const int shift = loc.level - depth - 1;
    const int ox = (shift >= 0) ? ((loc.lx1 >> shift) & 1) : 0;
    const int oy = (multi_d && shift >= 0) ? ((loc.lx2 >> shift) & 1) : 0;
    const int oz = (three_d && shift >= 0) ? ((loc.lx3 >> shift) & 1) : 0;
    return ox + (oy << 1) + (oz << 2);
  };
  auto z_less = [&](int a, int b) {
    const LogicalLocation &la = lloc_list[a];
    const LogicalLocation &lb = lloc_list[b];
    const int min_level = std::min(la.level, lb.level);
    for (int depth=0; depth<min_level; ++depth) {
      const int ca = child_index(la, depth);
      const int cb = child_index(lb, depth);
      if (ca != cb) return ca < cb;
    }
    if (la.level != lb.level) return la.level < lb.level;
    if (la.lx1 != lb.lx1) return la.lx1 < lb.lx1;
    if (la.lx2 != lb.lx2) return la.lx2 < lb.lx2;
    return la.lx3 < lb.lx3;
  };

  const bool hydro_lat = (pin != nullptr && pin->IsLATEnabled());
  const bool reorder_enabled = hydro_lat &&
      pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true);
  if (!hydro_lat || !reorder_enabled || input_already_reordered ||
      lat_factor_list == nullptr || lat_sync_factor <= 1) {
    return false;
  }

  std::vector<int> factor_values;
  for (int factor=1; factor<=lat_sync_factor; factor*=2) {
    factor_values.push_back(factor);
    if (factor > (std::numeric_limits<int>::max()/2)) break;
  }
  std::vector<std::vector<int>> bins(factor_values.size());
  for (int gid=0; gid<nb; ++gid) {
    int factor = std::max(1, lat_factor_list[gid]);
    int idx = 0;
    while (idx + 1 < static_cast<int>(factor_values.size()) &&
           factor_values[idx] < factor) {
      ++idx;
    }
    bins[idx].push_back(gid);
  }
  // Within a bin the sequence decides which rank gets which blocks: the interleave
  // below hands the s-th rank-sized chunk of the final order the s-th slice of every
  // bin, and with a static mesh the rank cuts are pinned to equal block counts (the
  // memory cap equals the minimum cap), so the ORDER inside each bin is the only lever
  // the partition has over the measured per-step work.  Plain Z-order keeps the
  // expensive blocks (the thick disk, spatially compact) contiguous, i.e. on one rank.
  // With work weights, each bin is dealt into nranks slices of the interleave's own
  // quota: the blocks carrying real excess over a plain block go heaviest-first to the
  // slice with the least excess so far (LPT on the excess, because a heavy block
  // displaces a plain one from its slice), the plain blocks then fill the remaining
  // quota in Z-order runs, and each slice is emitted in Z-order so a rank's share of
  // the bin stays spatially compact.  Uniform work leaves every
  // bin in its Z-order, i.e. the order this function always produced.  The deal is a
  // deterministic function of the gathered weights, so every rank builds the same map.
  const int nranks = std::max(1, global_variable::nranks);
  auto deal_bin = [&](std::vector<int> &bin) {
    if (work_list == nullptr || nranks <= 1 || bin.size() <= 1) return;
    // Below one twentieth of a plain block the placement is inside the +-1 block the
    // quotas themselves round by; such blocks keep their Z-order locality.
    constexpr double kHeavyExcess = 0.05;
    std::vector<int> heavy, light;
    for (int gid : bin) {
      const double excess = static_cast<double>(work_list[gid]) - 1.0;
      if (excess > kHeavyExcess) {
        heavy.push_back(gid);
      } else {
        light.push_back(gid);
      }
    }
    if (heavy.empty()) return;
    const long long nbin = static_cast<long long>(bin.size());
    const int nslices = static_cast<int>(std::min<long long>(nranks, nbin));
    std::vector<int> quota(nslices, 0);
    for (int s=0; s<nslices; ++s) {
      quota[s] = static_cast<int>(((s + 1)*nbin)/nslices - (s*nbin)/nslices);
    }
    std::stable_sort(heavy.begin(), heavy.end(), [&](int a, int b) {
      return work_list[a] > work_list[b];
    });
    std::vector<std::vector<int>> slices(nslices);
    std::vector<double> slice_excess(nslices, 0.0);
    for (int gid : heavy) {
      int best = -1;
      for (int s=0; s<nslices; ++s) {
        if (static_cast<int>(slices[s].size()) >= quota[s]) continue;
        if (best < 0 || slice_excess[s] < slice_excess[best] - 1.0e-12) best = s;
      }
      if (best < 0) best = nslices - 1;  // unreachable: the quotas sum to the bin size
      slices[best].push_back(gid);
      slice_excess[best] += static_cast<double>(work_list[gid]) - 1.0;
    }
    int s = 0;
    for (int gid : light) {
      while (s < nslices - 1 && static_cast<int>(slices[s].size()) >= quota[s]) ++s;
      slices[s].push_back(gid);
    }
    bin.clear();
    for (auto &slice : slices) {
      std::stable_sort(slice.begin(), slice.end(), z_less);
      bin.insert(bin.end(), slice.begin(), slice.end());
    }
  };

  int nonempty_bins = 0;
  for (auto &bin : bins) {
    if (!bin.empty()) {
      ++nonempty_bins;
      std::stable_sort(bin.begin(), bin.end(), z_less);
      deal_bin(bin);
    }
  }
  if (nonempty_bins <= 1) return false;

  std::vector<int> cursor(bins.size(), 0);
  for (int pos=0; pos<nb; ++pos) {
    int best = -1;
    double best_score = -std::numeric_limits<double>::max();
    for (int b=0; b<static_cast<int>(bins.size()); ++b) {
      if (cursor[b] >= static_cast<int>(bins[b].size())) continue;
      const double ideal =
          (static_cast<double>(pos + 1)*static_cast<double>(bins[b].size()))/
          static_cast<double>(nb);
      const double score = ideal - static_cast<double>(cursor[b]);
      if (best < 0 || score > best_score + 1.0e-12 ||
          (std::abs(score - best_score) <= 1.0e-12 &&
           factor_values[b] < factor_values[best])) {
        best = b;
        best_score = score;
      }
    }
    if (best < 0) return false;
    newtoold[pos] = bins[best][cursor[best]++];
  }

  bool changed = false;
  std::vector<int> seen(nb, 0);
  for (int gid=0; gid<nb; ++gid) {
    const int old_gid = newtoold[gid];
    if (old_gid < 0 || old_gid >= nb || seen[old_gid] != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "LAT GID reorder map is not a permutation at new gid=" << gid
                << ", old gid=" << old_gid << std::endl;
      std::exit(EXIT_FAILURE);
    }
    seen[old_gid] = 1;
    changed = changed || (old_gid != gid);
  }
  return changed;
}

//----------------------------------------------------------------------------------------
void Mesh::ApplyHydroLATGIDMapToArrays(LogicalLocation *lloc_list, float *cost_list,
                                       int *lat_factor_list, int **restart_gid_map,
                                       int nb, const int *newtoold, bool reordered) {
  if (lloc_list == nullptr || cost_list == nullptr || newtoold == nullptr || nb <= 0) {
    return;
  }
  std::vector<LogicalLocation> old_lloc(lloc_list, lloc_list + nb);
  std::vector<float> old_cost(cost_list, cost_list + nb);
  std::vector<int> old_factor;
  if (lat_factor_list != nullptr) {
    old_factor.assign(lat_factor_list, lat_factor_list + nb);
  }
  std::vector<int> old_restart_map;
  if (restart_gid_map != nullptr && *restart_gid_map != nullptr) {
    old_restart_map.assign(*restart_gid_map, *restart_gid_map + nb);
  }

  for (int gid=0; gid<nb; ++gid) {
    const int old_gid = newtoold[gid];
    if (old_gid < 0 || old_gid >= nb) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "LAT GID reorder map references invalid old gid=" << old_gid
                << " at new gid=" << gid << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lloc_list[gid] = old_lloc[old_gid];
    cost_list[gid] = old_cost[old_gid];
    if (lat_factor_list != nullptr) lat_factor_list[gid] = old_factor[old_gid];
  }

  if (restart_gid_map != nullptr) {
    std::vector<int> composed(nb, 0);
    for (int gid=0; gid<nb; ++gid) {
      const int old_gid = newtoold[gid];
      composed[gid] = old_restart_map.empty() ? old_gid : old_restart_map[old_gid];
    }
    delete [] *restart_gid_map;
    *restart_gid_map = new int[nb];
    for (int gid=0; gid<nb; ++gid) (*restart_gid_map)[gid] = composed[gid];
  }

  for (int gid=0; gid<nb; ++gid) {
    ptree->SetLeafGID(lloc_list[gid], gid);
  }
  hydro_lat_gid_reordered = reordered;
  hydro_lat_lb_topology_version = kInvalidLATVersion;
  hydro_lat_lb_metadata_version = kInvalidLATVersion;
  hydro_lat_lb_work_version = kInvalidLATVersion;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb)
//! \brief Calculate distribution of MeshBlocks across ranks based on input cost list
//! input: clist = cost of each MB (array of length nmbtotal)
//!        nb = number of MeshBlocks
//! output: rlist = rank to which each MB is assigned (array of length nmbtotal)
//!         slist = starting grid ID (gid) for MB on each rank (array of length nrank)
//!         nlist = number of MBs on each rank (array of length nrank)
//! With multiple ranks in MPI, this function is needed even on a uniform mesh and not
//! just for SMR/AMR, which is why it is part of the Mesh and not MeshRefinement class.

void Mesh::LoadBalance(float *clist, int *rlist, int *slist, int *nlist, int nb,
                       int max_blocks_per_rank, int *lat_factor_list,
                       int lat_sync_factor) {
  float min_cost = std::numeric_limits<float>::max();
  float max_cost = 0.0, totalcost = 0.0;
  // find min/max and total cost in clist
  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    min_cost = std::min(min_cost,clist[i]);
    max_cost = std::max(max_cost,clist[i]);
  }

  if (max_blocks_per_rank > 0 && lat_factor_list != nullptr && lat_sync_factor > 1 &&
      global_variable::nranks > 1) {
    const int nranks = global_variable::nranks;
    const int min_cap = (nb + nranks - 1)/nranks;
    const int block_cap = max_blocks_per_rank;
    if (block_cap < min_cap ||
        static_cast<long long>(block_cap)*static_cast<long long>(nranks) < nb) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "HD LAT load-balance cap is infeasible: cap=" << block_cap
                << ", nranks=" << nranks << ", MeshBlocks=" << nb << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const int avg_ceil = (nb + nranks - 1)/nranks;
    const int balance_slack = std::max(8, (avg_ceil + 19)/20);
    const int balance_min = std::max(1, nb - block_cap*(nranks - 1));
    const int balance_cap = block_cap;
    const int search_radius = std::max(8, std::min(balance_slack, avg_ceil/10));
    std::vector<int> factor_values;
    for (int factor=1; factor<=lat_sync_factor; factor*=2) {
      factor_values.push_back(factor);
      if (factor > (std::numeric_limits<int>::max()/2)) break;
    }
    const int nfactors = static_cast<int>(factor_values.size());
    std::vector<std::vector<int>> prefix(nfactors, std::vector<int>(nb + 1, 0));
    std::vector<double> cost_prefix(nb + 1, 0.0);
    for (int i=0; i<nb; ++i) {
      for (int n=0; n<nfactors; ++n) {
        prefix[n][i + 1] = prefix[n][i] +
                           ((lat_factor_list[i] == factor_values[n]) ? 1 : 0);
      }
      cost_prefix[i + 1] = cost_prefix[i] + static_cast<double>(clist[i]);
    }
    // Power-of-two LAT factors produce only log2(sync)+1 distinct due sets.  Store one
    // prefix per phase class and its multiplicity instead of one prefix per fine tick;
    // this keeps levels near the public upper limit O(log(sync)*nb), not O(sync*nb).
    std::vector<std::vector<double>> due_prefix(
        nfactors, std::vector<double>(nb + 1, 0.0));
    std::vector<int> due_multiplicity(nfactors, 1);
    for (int phase=0; phase<nfactors - 1; ++phase) {
      due_multiplicity[phase] = lat_sync_factor/(2*factor_values[phase]);
    }
    std::vector<double> active_cost(nb, 1.0);
    for (int i=0; i<nb; ++i) {
      const int factor = std::max(1, lat_factor_list[i]);
      active_cost[i] =
          static_cast<double>(clist[i])*static_cast<double>(factor)/
          static_cast<double>(std::max(1, lat_sync_factor));
    }
    for (int phase=0; phase<nfactors; ++phase) {
      for (int i=0; i<nb; ++i) {
        const int factor = std::max(1, lat_factor_list[i]);
        due_prefix[phase][i + 1] = due_prefix[phase][i] +
            ((factor <= factor_values[phase]) ? active_cost[i] : 0.0);
      }
    }

    auto valid_cuts = [&](const std::vector<int> &cuts) {
      if (static_cast<int>(cuts.size()) != nranks + 1) return false;
      if (cuts.front() != 0 || cuts.back() != nb) return false;
      for (int r=0; r<nranks; ++r) {
        const int count = cuts[r + 1] - cuts[r];
        if (count < balance_min || count > balance_cap) return false;
      }
      return true;
    };

    struct LATPartitionObjective {
      double max_tick_work;
      double sum_tick_max_work;
      double idle_slots;
      double bin_idle_slots;
      double sum_bin_max_blocks;
      int max_blocks;
      double block_imbalance;
      double scalar_cost;
    };

    auto scalar_cost_objective = [&](const std::vector<int> &cuts) {
      double max_cost_rank = 0.0;
      for (int r=0; r<nranks; ++r) {
        max_cost_rank = std::max(max_cost_rank,
                                 cost_prefix[cuts[r + 1]] - cost_prefix[cuts[r]]);
      }
      return max_cost_rank;
    };

    auto partition_objective = [&](const std::vector<int> &cuts) {
      LATPartitionObjective objective{0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0,
                                      scalar_cost_objective(cuts)};
      std::vector<double> rank_work(nranks, 0.0);
      for (int phase=0; phase<nfactors; ++phase) {
        bool tick_has_work = false;
        double tick_max = 0.0;
        for (int r=0; r<nranks; ++r) {
          rank_work[r] =
              due_prefix[phase][cuts[r + 1]] - due_prefix[phase][cuts[r]];
          tick_has_work = tick_has_work || (rank_work[r] > 0.0);
          tick_max = std::max(tick_max, rank_work[r]);
        }
        objective.max_tick_work = std::max(objective.max_tick_work, tick_max);
        objective.sum_tick_max_work += due_multiplicity[phase]*tick_max;
        if (tick_has_work) {
          for (int r=0; r<nranks; ++r) {
            if (rank_work[r] <= 0.0) {
              objective.idle_slots += due_multiplicity[phase];
            }
          }
        }
      }
      for (int r=0; r<nranks; ++r) {
        const int count = cuts[r + 1] - cuts[r];
        objective.max_blocks = std::max(objective.max_blocks, count);
        objective.block_imbalance = std::max(
            objective.block_imbalance,
            std::abs(static_cast<double>(count) -
                     static_cast<double>(nb)/static_cast<double>(nranks)));
      }
      for (int n=0; n<nfactors; ++n) {
        const int total_in_bin = prefix[n][nb];
        if (total_in_bin <= 0) continue;
        int max_in_rank = 0;
        if (total_in_bin >= nranks) {
          for (int r=0; r<nranks; ++r) {
            const int count = prefix[n][cuts[r + 1]] - prefix[n][cuts[r]];
            max_in_rank = std::max(max_in_rank, count);
            if (count == 0) objective.bin_idle_slots += 1.0;
          }
        } else {
          for (int r=0; r<nranks; ++r) {
            max_in_rank = std::max(max_in_rank,
                                   prefix[n][cuts[r + 1]] - prefix[n][cuts[r]]);
          }
        }
        objective.sum_bin_max_blocks += static_cast<double>(max_in_rank);
      }
      return objective;
    };

    auto better_objective = [](const LATPartitionObjective &lhs,
                               const LATPartitionObjective &rhs) {
      const double eps = 1.0e-10;
      if (lhs.sum_tick_max_work < rhs.sum_tick_max_work - eps) return true;
      if (lhs.sum_tick_max_work > rhs.sum_tick_max_work + eps) return false;
      if (lhs.max_tick_work < rhs.max_tick_work - eps) return true;
      if (lhs.max_tick_work > rhs.max_tick_work + eps) return false;
      if (lhs.idle_slots < rhs.idle_slots - eps) return true;
      if (lhs.idle_slots > rhs.idle_slots + eps) return false;
      if (lhs.bin_idle_slots < rhs.bin_idle_slots - eps) return true;
      if (lhs.bin_idle_slots > rhs.bin_idle_slots + eps) return false;
      if (lhs.sum_bin_max_blocks < rhs.sum_bin_max_blocks - eps) return true;
      if (lhs.sum_bin_max_blocks > rhs.sum_bin_max_blocks + eps) return false;
      if (lhs.max_blocks < rhs.max_blocks) return true;
      if (lhs.max_blocks > rhs.max_blocks) return false;
      if (lhs.block_imbalance < rhs.block_imbalance - eps) return true;
      if (lhs.block_imbalance > rhs.block_imbalance + eps) return false;
      return lhs.scalar_cost < rhs.scalar_cost - eps;
    };

    auto build_scalar_cuts = [&](const std::vector<double> &costs) {
      std::vector<int> cuts;
      if (static_cast<int>(costs.size()) != nb) return cuts;
      double local_total = 0.0;
      double local_max = 0.0;
      for (double cost : costs) {
        local_total += cost;
        local_max = std::max(local_max, cost);
      }
      auto feasible = [&](double limit) {
        int used = 1;
        int nblocks = 0;
        double cost = 0.0;
        for (int i=0; i<nb; ++i) {
          if (costs[i] > limit) return false;
          if (nblocks > 0 &&
              (nblocks + 1 > balance_cap || cost + costs[i] > limit)) {
            used++;
            nblocks = 0;
            cost = 0.0;
            if (used > nranks) return false;
          }
          nblocks++;
          cost += costs[i];
        }
        return used <= nranks;
      };

      double lo = std::max(local_max, local_total/static_cast<double>(nranks));
      double hi = local_total;
      for (int iter=0; iter<64; ++iter) {
        const double mid = 0.5*(lo + hi);
        if (feasible(mid)) {
          hi = mid;
        } else {
          lo = mid;
        }
      }

      cuts.push_back(0);
      int nblocks = 0;
      double cost = 0.0;
      for (int i=0; i<nb; ++i) {
        if (nblocks > 0 && (nblocks + 1 > balance_cap || cost + costs[i] > hi)) {
          cuts.push_back(i);
          nblocks = 0;
          cost = 0.0;
        }
        nblocks++;
        cost += costs[i];
      }
      cuts.push_back(nb);

      while (static_cast<int>(cuts.size()) < nranks + 1) {
        int split_idx = -1;
        int split_count = 0;
        for (int r=0; r<static_cast<int>(cuts.size()) - 1; ++r) {
          const int count = cuts[r + 1] - cuts[r];
          if (count > split_count) {
            split_count = count;
            split_idx = r;
          }
        }
        if (split_idx < 0 || split_count <= 1) break;
        const int mid = cuts[split_idx] + split_count/2;
        cuts.insert(cuts.begin() + split_idx + 1, mid);
      }
      if (!valid_cuts(cuts)) cuts.clear();
      return cuts;
    };

    auto improve_cuts = [&](std::vector<int> cuts) {
      LATPartitionObjective best = partition_objective(cuts);

      // Phase 1: single-cut steepest descent.
      bool changed = true;
      for (int pass=0; pass<48 && changed; ++pass) {
        changed = false;
        int best_k = -1;
        int best_candidate = -1;
        LATPartitionObjective best_trial = best;
        for (int k=1; k<nranks; ++k) {
          const int old = cuts[k];
          const int lo = std::max({cuts[k - 1] + balance_min,
                                   cuts[k + 1] - balance_cap,
                                   old - search_radius});
          const int hi = std::min({cuts[k + 1] - balance_min,
                                   cuts[k - 1] + balance_cap,
                                   old + search_radius});
          for (int candidate=lo; candidate<=hi; ++candidate) {
            if (candidate == old) continue;
            cuts[k] = candidate;
            const LATPartitionObjective objective = partition_objective(cuts);
            if (better_objective(objective, best_trial)) {
              best_trial = objective;
              best_k = k;
              best_candidate = candidate;
            }
          }
          cuts[k] = old;
        }
        if (best_k >= 0) {
          cuts[best_k] = best_candidate;
          best = best_trial;
          changed = true;
        }
      }

      // Phase 2: move adjacent cut pairs together so neighboring ranks can
      // exchange work that cannot be fixed by one cut at a time.
      if (nranks >= 3) {
        changed = true;
        for (int pass = 0; pass < 16 && changed; ++pass) {
          changed = false;
          for (int k = 1; k < nranks - 1; ++k) {
            const int old_k = cuts[k];
            const int old_k1 = cuts[k + 1];
            const int lo_k = std::max(cuts[k - 1] + balance_min,
                                      old_k - search_radius);
            const int hi_k = std::min(old_k1 - 1,
                                      std::min(cuts[k - 1] + balance_cap,
                                               old_k + search_radius));
            int best_ck = old_k, best_ck1 = old_k1;
            LATPartitionObjective best_pair = best;
            for (int ck = lo_k; ck <= hi_k; ++ck) {
              if (ck == old_k) continue;  // Single-cut moves already tried
              cuts[k] = ck;
              const int lo_k1 = std::max({ck + balance_min,
                                           (k + 2 <= nranks) ?
                                           cuts[k + 2] - balance_cap : 1,
                                           old_k1 - search_radius});
              const int hi_k1 = std::min({
                  (k + 2 <= nranks) ? cuts[k + 2] - balance_min : nb,
                  ck + balance_cap,
                  old_k1 + search_radius});
              for (int ck1 = lo_k1; ck1 <= hi_k1; ++ck1) {
                cuts[k + 1] = ck1;
                const LATPartitionObjective obj = partition_objective(cuts);
                if (better_objective(obj, best_pair)) {
                  best_pair = obj;
                  best_ck = ck;
                  best_ck1 = ck1;
                }
              }
            }
            cuts[k] = old_k;
            cuts[k + 1] = old_k1;
            if (best_ck != old_k || best_ck1 != old_k1) {
              cuts[k] = best_ck;
              cuts[k + 1] = best_ck1;
              best = best_pair;
              changed = true;
            }
          }
        }
        // Phase 3: final single-cut polish after pair moves.
        changed = true;
        for (int pass = 0; pass < 16 && changed; ++pass) {
          changed = false;
          for (int k = 1; k < nranks; ++k) {
            const int old = cuts[k];
            const int lo = std::max({cuts[k - 1] + balance_min,
                                     cuts[k + 1] - balance_cap,
                                     old - search_radius});
            const int hi = std::min({cuts[k + 1] - balance_min,
                                     cuts[k - 1] + balance_cap,
                                     old + search_radius});
            int best_c = old;
            LATPartitionObjective best_single = best;
            for (int c = lo; c <= hi; ++c) {
              if (c == old) continue;
              cuts[k] = c;
              const LATPartitionObjective obj = partition_objective(cuts);
              if (better_objective(obj, best_single)) {
                best_single = obj;
                best_c = c;
              }
            }
            cuts[k] = old;
            if (best_c != old) {
              cuts[k] = best_c;
              best = best_single;
              changed = true;
            }
          }
        }
      }
      return cuts;
    };

    std::vector<std::vector<int>> seeds;
    std::vector<double> scalar_costs(nb, 1.0);
    for (int i=0; i<nb; ++i) scalar_costs[i] = clist[i];
    seeds.push_back(build_scalar_cuts(scalar_costs));
    seeds.push_back(build_scalar_cuts(active_cost));

    std::vector<int> equal_cuts(nranks + 1, 0);
    for (int r=0; r<=nranks; ++r) {
      equal_cuts[r] = static_cast<int>(
          (static_cast<long long>(nb)*static_cast<long long>(r) +
           static_cast<long long>(nranks)/2)/static_cast<long long>(nranks));
    }
    if (valid_cuts(equal_cuts)) seeds.push_back(equal_cuts);

    const double powers[] = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0};
    for (double power : powers) {
      for (int form=0; form<2; ++form) {
        std::vector<double> factor_weight(lat_sync_factor + 1, 1.0);
        for (int factor : factor_values) {
          const double executions = static_cast<double>(lat_sync_factor/factor);
          factor_weight[factor] = (form == 0) ? executions*std::pow(factor, power)
                                              : std::pow(executions, power);
        }
        std::vector<double> costs(nb, 1.0);
        for (int i=0; i<nb; ++i) costs[i] = factor_weight[lat_factor_list[i]];
        seeds.push_back(build_scalar_cuts(costs));
      }
    }

    // Peak-tick seeds emphasize frequently updated blocks.
    for (double alpha : {1.5, 2.0, 2.5, 3.0}) {
      std::vector<double> peak_costs(nb, 1.0);
      for (int i=0; i<nb; ++i) {
        const int factor = std::max(1, lat_factor_list[i]);
        const double executions = static_cast<double>(lat_sync_factor) /
                                  static_cast<double>(factor);
        peak_costs[i] = std::pow(executions, alpha);
      }
      seeds.push_back(build_scalar_cuts(peak_costs));
    }

    // Bin-count seeds try to spread each LAT factor bin across ranks.
    for (int target_n = 0; target_n < nfactors; ++target_n) {
      const int total_in_bin = prefix[target_n][nb];
      if (total_in_bin < nranks) continue;  // Too few to distribute
      // Build cuts that equalize the target bin's distribution.
      // For rank r, the ideal count of target-bin blocks is total_in_bin/nranks.
      // Walk through blocks in GID order, cutting when the cumulative target-bin
      // count exceeds the next rank's ideal share.
      std::vector<int> bin_balanced_cuts;
      bin_balanced_cuts.push_back(0);
      const double ideal_per_rank =
          static_cast<double>(total_in_bin) / static_cast<double>(nranks);
      int next_threshold_rank = 1;
      for (int i = 1; i < nb && next_threshold_rank < nranks; ++i) {
        const int cumulative = prefix[target_n][i];
        const double target = ideal_per_rank * next_threshold_rank;
        if (static_cast<double>(cumulative) >= target) {
          const int count = i - bin_balanced_cuts.back();
          if (count >= balance_min && count <= balance_cap) {
            bin_balanced_cuts.push_back(i);
            ++next_threshold_rank;
          }
        }
      }
      bin_balanced_cuts.push_back(nb);
      while (static_cast<int>(bin_balanced_cuts.size()) < nranks + 1) {
        int split_idx = -1, split_count = 0;
        for (int r = 0; r < static_cast<int>(bin_balanced_cuts.size()) - 1; ++r) {
          const int count = bin_balanced_cuts[r + 1] - bin_balanced_cuts[r];
          if (count > split_count) { split_count = count; split_idx = r; }
        }
        if (split_idx < 0 || split_count <= 1) break;
        const int mid = bin_balanced_cuts[split_idx] + split_count / 2;
        bin_balanced_cuts.insert(bin_balanced_cuts.begin() + split_idx + 1, mid);
      }
      while (static_cast<int>(bin_balanced_cuts.size()) > nranks + 1) {
        int merge_idx = -1, merge_count = nb + 1;
        for (int r = 0; r < static_cast<int>(bin_balanced_cuts.size()) - 2; ++r) {
          const int combined = bin_balanced_cuts[r + 2] - bin_balanced_cuts[r];
          if (combined < merge_count) { merge_count = combined; merge_idx = r; }
        }
        if (merge_idx < 0) break;
        bin_balanced_cuts.erase(bin_balanced_cuts.begin() + merge_idx + 1);
      }
      if (valid_cuts(bin_balanced_cuts)) seeds.push_back(bin_balanced_cuts);
    }

    // Tick-phase seeds target the fine phases that usually dominate LAT idle time.
    {
      std::vector<int> critical_phases;
      critical_phases.push_back(0);  // Fine phase: only factor-1
      if (lat_sync_factor >= 4) critical_phases.push_back(1);  // factor-1 + factor-2
      if (lat_sync_factor >= 8) critical_phases.push_back(2);  // factor-1..4

      for (int phase : critical_phases) {
        if (phase >= nfactors) continue;
        const double total_tick_work = due_prefix[phase][nb];
        if (total_tick_work <= 0.0) continue;
        const double target_per_rank = total_tick_work / static_cast<double>(nranks);

        std::vector<int> tick_cuts;
        tick_cuts.push_back(0);
        int next_rank = 1;
        for (int i = 1; i < nb && next_rank < nranks; ++i) {
          const double cumulative = due_prefix[phase][i];
          if (cumulative >= target_per_rank * next_rank) {
            const int count = i - tick_cuts.back();
            if (count >= balance_min && count <= balance_cap) {
              tick_cuts.push_back(i);
              ++next_rank;
            }
          }
        }
        tick_cuts.push_back(nb);
        while (static_cast<int>(tick_cuts.size()) < nranks + 1) {
          int split_idx = -1, split_count = 0;
          for (int r = 0; r < static_cast<int>(tick_cuts.size()) - 1; ++r) {
            const int c = tick_cuts[r + 1] - tick_cuts[r];
            if (c > split_count) { split_count = c; split_idx = r; }
          }
          if (split_idx < 0 || split_count <= 1) break;
          tick_cuts.insert(tick_cuts.begin() + split_idx + 1,
                           tick_cuts[split_idx] + split_count / 2);
        }
        while (static_cast<int>(tick_cuts.size()) > nranks + 1) {
          int merge_idx = -1, merge_count = nb + 1;
          for (int r = 0; r < static_cast<int>(tick_cuts.size()) - 2; ++r) {
            const int c = tick_cuts[r + 2] - tick_cuts[r];
            if (c < merge_count) { merge_count = c; merge_idx = r; }
          }
          if (merge_idx < 0) break;
          tick_cuts.erase(tick_cuts.begin() + merge_idx + 1);
        }
        if (valid_cuts(tick_cuts)) seeds.push_back(tick_cuts);
      }
    }

    std::vector<int> best_cuts;
    for (std::vector<int> cuts : seeds) {
      if (!valid_cuts(cuts)) continue;
      cuts = improve_cuts(cuts);
      if (!valid_cuts(cuts)) continue;
      const LATPartitionObjective objective = partition_objective(cuts);
      if (best_cuts.empty() ||
          better_objective(objective, partition_objective(best_cuts))) {
        best_cuts = std::move(cuts);
      }
    }

    if (valid_cuts(best_cuts)) {
      for (int rank=0; rank<nranks; ++rank) {
        slist[rank] = best_cuts[rank];
        nlist[rank] = best_cuts[rank + 1] - best_cuts[rank];
        for (int i=best_cuts[rank]; i<best_cuts[rank + 1]; ++i) rlist[i] = rank;
      }
      if (global_variable::my_rank == 0) {
        std::cout << "Mesh: HD LAT per-bin rank distribution";
        for (int n=0; n<nfactors; ++n) {
          const int total_in_bin = prefix[n][nb];
          if (total_in_bin <= 0) continue;
          std::cout << " f" << factor_values[n] << "=[";
          for (int rank=0; rank<nranks; ++rank) {
            if (rank > 0) std::cout << ",";
            std::cout << (prefix[n][best_cuts[rank + 1]] -
                          prefix[n][best_cuts[rank]]);
          }
          std::cout << "]";
        }
        // Predicted busy fraction of each rank over the window under the measured
        // work: its summed tick work against the critical path (the objective), the
        // number the measured per-rank GPU utilisation is compared with.
        const LATPartitionObjective chosen = partition_objective(best_cuts);
        if (chosen.sum_tick_max_work > 0.0) {
          std::cout << " util%=[";
          for (int rank=0; rank<nranks; ++rank) {
            double rank_total = 0.0;
            for (int phase=0; phase<nfactors; ++phase) {
              rank_total += due_multiplicity[phase]*
                  (due_prefix[phase][best_cuts[rank + 1]] -
                   due_prefix[phase][best_cuts[rank]]);
            }
            if (rank > 0) std::cout << ",";
            std::cout << static_cast<int>(
                100.0*rank_total/chosen.sum_tick_max_work + 0.5);
          }
          std::cout << "]";
        }
        std::cout << std::endl;
      }
      return;
    }
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "HD LAT load balancer could not find a partition under hard cap "
              << block_cap << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (max_blocks_per_rank > 0 && global_variable::nranks > 1) {
    const int nranks = global_variable::nranks;
    const int min_cap = (nb + nranks - 1)/nranks;
    const int block_cap = max_blocks_per_rank;
    if (block_cap < min_cap ||
        static_cast<long long>(block_cap)*static_cast<long long>(nranks) < nb) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Load-balance cap is infeasible: cap=" << block_cap
                << ", nranks=" << nranks << ", MeshBlocks=" << nb << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::vector<double> prefix(nb + 1, 0.0);
    for (int i=0; i<nb; ++i) prefix[i + 1] = prefix[i] + clist[i];

    auto feasible = [&](double limit) {
      int used = 1;
      int nblocks = 0;
      double cost = 0.0;
      for (int i=0; i<nb; ++i) {
        if (clist[i] > limit) return false;
        if (nblocks > 0 && (nblocks + 1 > block_cap || cost + clist[i] > limit)) {
          used++;
          nblocks = 0;
          cost = 0.0;
          if (used > nranks) return false;
        }
        nblocks++;
        cost += clist[i];
      }
      return used <= nranks;
    };

    double lo = std::max(static_cast<double>(max_cost),
                         static_cast<double>(totalcost)/static_cast<double>(nranks));
    double hi = static_cast<double>(totalcost);
    for (int iter=0; iter<64; ++iter) {
      const double mid = 0.5*(lo + hi);
      if (feasible(mid)) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    const double limit = hi;

    struct Partition {int start, end; double cost;};
    std::vector<Partition> parts;
    int start = 0;
    int nblocks = 0;
    double cost = 0.0;
    for (int i=0; i<nb; ++i) {
      if (nblocks > 0 && (nblocks + 1 > block_cap || cost + clist[i] > limit)) {
        parts.push_back({start, i, cost});
        start = i;
        nblocks = 0;
        cost = 0.0;
      }
      nblocks++;
      cost += clist[i];
    }
    parts.push_back({start, nb, cost});

    while (static_cast<int>(parts.size()) < nranks) {
      int split_idx = -1;
      int split_count = 0;
      for (int p=0; p<static_cast<int>(parts.size()); ++p) {
        const int count = parts[p].end - parts[p].start;
        if (count > split_count) {
          split_count = count;
          split_idx = p;
        }
      }
      if (split_idx < 0 || split_count <= 1) break;
      Partition part = parts[split_idx];
      const int mid = part.start + split_count/2;
      const double left_cost = prefix[mid] - prefix[part.start];
      const double right_cost = prefix[part.end] - prefix[mid];
      parts[split_idx] = {part.start, mid, left_cost};
      parts.insert(parts.begin() + split_idx + 1, {mid, part.end, right_cost});
    }

    if (static_cast<int>(parts.size()) == nranks) {
      for (int rank=0; rank<nranks; ++rank) {
        slist[rank] = parts[rank].start;
        nlist[rank] = parts[rank].end - parts[rank].start;
        for (int i=parts[rank].start; i<parts[rank].end; ++i) rlist[i] = rank;
      }
      return;
    }
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Load balancer could not find a partition under hard cap "
              << block_cap << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int j = (global_variable::nranks) - 1;
  float targetcost = totalcost/global_variable::nranks;
  float mycost = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i=nb-1; i>=0; i--) {
    if (targetcost == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "There is at least one process which has no MeshBlock"
                << std::endl << "Decrease the number of processes or use smaller "
                << "MeshBlocks." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j>0) {
      j--;
      totalcost -= mycost;
      mycost = 0.0;
      targetcost = totalcost/(j+1);
    }
  }
  slist[0] = 0;
  j = 0;
  for (int i=1; i<nb; i++) { // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i-1]) {
      nlist[j] = i-slist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb-slist[j];

#if MPI_PARALLEL_ENABLED
  if (nb % global_variable::nranks != 0
     && !adaptive && max_cost == min_cost && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Number of MeshBlocks cannot be divided evenly by number of MPI ranks. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::EnsureAMRStagingCapacity()
//! \brief Grow-only sizing of the persistent device pack/unpack staging buffer used by
//! both directions of the AMR load balance.  Allocating and releasing it per topology
//! transaction was pure churn: with refinement_interval=1 that is one alloc/free pair of
//! several MiB every cycle, and its contiguous request was the first observed OOM.
//!
//! The send and recv stages share one buffer because they are provably never live at the
//! same time: PackAndSendAMR finishes every device chunk with a blocking device->host
//! deep_copy into send_data before it posts any MPI send, the MPI transfers themselves
//! reference only the host snapshots send_data/recv_data, and ClearRecvAndUnpackAMR runs
//! strictly later in RedistAndRefineMeshBlocks.  Nothing device-side is left pending on
//! this buffer between the two phases, so a reallocation here is also safe.

void MeshRefinement::EnsureAMRStagingCapacity(std::size_t need_elements) {
#if MPI_PARALLEL_ENABLED
  // Never fall below the configured chunk size, so the steady state is a single
  // allocation that the largest ordinary transaction already fits into.
  const std::size_t need = std::max(need_elements, lb_chunk_elements);
  if (lb_stage.extent(0) >= need) return;
  // Any reallocation moves .data().  No persistent MPI request is bound to this buffer
  // (the AMR messages are built on send_data/recv_data in host RAM), so there is nothing
  // to tear down; release first so the larger block can reuse the old one.
  lb_stage = DvceArray1D<Real>();
  lb_stage = DvceArray1D<Real>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lb amr stage"),
      AMRStagingCapacity(need));
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitRecvAMR()
//! \brief Allocates and initializes receive buffers, and posts non-blocking receives,
//! for communicating MeshBlocks during load balancing. Equivalent to some of the work
//! done inside MPI_PARALLEL block in the Mesh::RedistributeAndRefineMeshBlocks()
//! function in amr_loadbalance.cpp in Athena++

//----------------------------------------------------------------------------------------
//! \fn int MeshRefinement::FullBlockTransferPrimitiveCount()
//! \brief Number of extra cell-centered variables a pure rebalance appends to every
//! load-balance message: the fluid primitives (and the cell-centered field), which
//! travel with the conserved state instead of being re-derived after the move, the MHD
//! inversion's warm-start history (EquationOfState::CarriedC2PHistory), which are read
//! before the next step rewrites them.

int MeshRefinement::FullBlockTransferPrimitiveCount() const {
  if (!lb_full_block_transfer) return 0;
  int n = 0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    n += pmy_mesh->pmb_pack->phydro->w0.extent_int(1);
  }
  return n;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::SetSameLevelTransferExtent(AMRBuffer &buf)
//! \brief Cell extent of a same-level load-balance transfer: the active cells, or the
//! whole MeshBlock (ghost zones included) for a pure rebalance, whose target must hold
//! the moved block verbatim.  Face-centered counts follow PackAMRBuffersFC (one extra
//! face per direction).

#if MPI_PARALLEL_ENABLED
void MeshRefinement::SetSameLevelTransferExtent(AMRBuffer &buf) const {
  auto &indcs = pmy_mesh->mb_indcs;
  const int ng = indcs.ng;
  int n1 = indcs.nx1, n2 = indcs.nx2, n3 = indcs.nx3;
  if (lb_full_block_transfer) {
    n1 += 2*ng;
    if (pmy_mesh->multi_d) n2 += 2*ng;
    if (pmy_mesh->three_d) n3 += 2*ng;
    buf.bis = 0; buf.bjs = 0; buf.bks = 0;
  } else {
    buf.bis = indcs.is; buf.bjs = indcs.js; buf.bks = indcs.ks;
  }
  buf.bie = buf.bis + n1 - 1;
  buf.bje = buf.bjs + n2 - 1;
  buf.bke = buf.bks + n3 - 1;
  buf.cntcc = n1*n2*n3;
  buf.cntfc = 3*n1*n2*n3 + n2*n3 + n1*n3 + n1*n2;
}
#endif

void MeshRefinement::InitRecvAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED
  // Step 1. (InitRecvAMR)
  // loop over new MBs on this rank, count number of MeshBlocks received by this rank
  nmb_recv = 0;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {          // old MB was de-refined
      const int root_rank = pmy_mesh->rank_eachmb[oldm];
      for (int l=0; l<nleaf; l++) {
        const int old_child = DerefOldChildGID(newm, l, nleaf);
        if (old_child < 0) continue;
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((root_rank != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[old_child] != global_variable::my_rank)) {
          nmb_recv++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        nmb_recv++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        nmb_recv++;
      }
    }
  }
  if (nmb_recv == 0) return;  // nothing to do

  // allocate array of recv buffers
  Kokkos::realloc(recvbuf, nmb_recv);

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->phydro->nvars;
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->pmhd->nvars;
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->prad != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->prad->prgeo->nangles);
  }
  if (pmy_mesh->pmb_pack->pgrav != nullptr && lb_transfer_gravity) {
    ncc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }
  ncc_tosend += FullBlockTransferPrimitiveCount();

  // Step 2. (InitRecvAMR)
  // loop over new MBs on this rank, initialize recv buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int rb_idx = 0;   // recv buffer index
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {        // old MB was de-refined
      const int root_rank = pmy_mesh->rank_eachmb[oldm];
      for (int l=0; l<nleaf; l++) {
        const int old_child = DerefOldChildGID(newm, l, nleaf);
        if (old_child < 0) continue;
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((root_rank != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[old_child] != global_variable::my_rank)) {
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[old_child];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          recvbuf.h_view(rb_idx).bis = cis + ox1*cnx1;
          recvbuf.h_view(rb_idx).bie = cie + ox1*cnx1;
          recvbuf.h_view(rb_idx).bjs = cjs + ox2*cnx2;
          recvbuf.h_view(rb_idx).bje = cje + ox2*cnx2;
          recvbuf.h_view(rb_idx).bks = cks + ox3*cnx3;
          recvbuf.h_view(rb_idx).bke = cke + ox3*cnx3;
          recvbuf.h_view(rb_idx).cntcc = cnx1*cnx2*cnx3;
          recvbuf.h_view(rb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 +
                                          cnx1*cnx3 + cnx1*cnx2;
          recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
              recvbuf.h_view(rb_idx).cntfc);
          recvbuf.h_view(rb_idx).lid   = newm - nmbs;
          recvbuf.h_view(rb_idx).use_coarse = false;
          if (rb_idx > 0) {
            recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                             recvbuf.h_view((rb_idx-1)).cnt;
          } else {
            recvbuf.h_view(rb_idx).offset = 0;
          }
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        SetSameLevelTransferExtent(recvbuf.h_view(rb_idx));
        recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
            ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
            recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = false;
        if (rb_idx > 0) {
          recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                          recvbuf.h_view((rb_idx-1)).cnt;
        } else {
          recvbuf.h_view(rb_idx).offset = 0;
        }
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        recvbuf.h_view(rb_idx).bis = il; // note il:iu etc. includes ghost zones
        recvbuf.h_view(rb_idx).bie = iu;
        recvbuf.h_view(rb_idx).bjs = jl;
        recvbuf.h_view(rb_idx).bje = ju;
        recvbuf.h_view(rb_idx).bks = kl;
        recvbuf.h_view(rb_idx).bke = ku;
        recvbuf.h_view(rb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
        recvbuf.h_view(rb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
             (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
        recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
            ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
            recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = true;
        if (rb_idx > 0) {
          recvbuf.h_view(rb_idx).offset = recvbuf.h_view((rb_idx-1)).offset +
                                          recvbuf.h_view((rb_idx-1)).cnt;
        } else {
          recvbuf.h_view(rb_idx).offset = 0;
        }
        rb_idx++;
      }
    }
  }
  ValidateAMRDescriptorCount("receive", rb_idx, nmb_recv);
  // Sync metadata and allocate the complete pageable-host receive snapshot.
  recvbuf.template modify<HostMemSpace>();
  recvbuf.template sync<DevExeSpace>();
  const std::size_t ndata =
      recvbuf.h_view(nmb_recv - 1).offset + recvbuf.h_view(nmb_recv - 1).cnt;
  recv_data = HostArray1D<Real>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lb recv data"), ndata);

  // Step 3. (InitRecvAMR)
  // loop over new MBs on this rank, post non-blocking recvs
  // Receive requests will only be accessed on host, so no need to sync after this step.
  rb_idx = 0;   // recv buffer index
  std::vector<int> recv_peer_ranks;
  std::vector<int> recv_keys;
  recv_peer_ranks.reserve(nmb_recv);
  recv_keys.reserve(nmb_recv);
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {        // old MB was de-refined
      const int root_rank = pmy_mesh->rank_eachmb[oldm];
      for (int l=0; l<nleaf; l++) {
        const int old_child = DerefOldChildGID(newm, l, nleaf);
        if (old_child < 0) continue;
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((root_rank != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[old_child] != global_variable::my_rank)) {
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[old_child];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[old_child]);
          recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, ox1, ox2, ox3));
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[oldm]);
        recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0));
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[oldm]);
        recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0));
        rb_idx++;
      }
    }
  }

  ValidateAMRDescriptorCount("posted receive", rb_idx, nmb_recv);
  // Both peers sort descriptor segments by the old destination tag.  The derived type
  // scatters one peer payload directly into the existing per-descriptor host snapshot.
  rank_recv_messages = BuildAMRRankMessages(
      recvbuf, recv_peer_ranks, recv_keys, nmb_recv, "receive");
  bool no_errors=true;
  for (auto &msg : rank_recv_messages) {
    int ierr = MPI_Irecv(recv_data.data(), 1, msg.datatype, msg.rank, 0, amr_comm,
                         &msg.request);
    if (ierr != MPI_SUCCESS) no_errors=false;
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives with AMR" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAndSendAMR()
//! \brief Allocates and initializes send buffers for communicating MeshBlocks during
//! load balancing, calls function to pack data into buffers, and posts non-blocking sends
//! Equivalent to some of the work done inside MPI_PARALLEL block in the
//! Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp in Athena++

void MeshRefinement::PackAndSendAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED
  // Step 1. (PackAndSendAMR)
  // loop over old MBs on this rank, count number of MeshBlocks to send on this rank
  nmb_send = 0;
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {          // old MB was refined
      for (int l=0; l<nleaf; l++) {
        const int new_child = RefineNewChildGID(oldm, l, nleaf);
        if (new_child < 0) continue;
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[new_child] != global_variable::my_rank)) {
          nmb_send++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB on same level
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        nmb_send++;
      }
    } else {                                        // old MB was de-refined
      // send if root MB changes rank, or if any leaf on different rank than root
      if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
          (new_rank_eachmb[newm] != global_variable::my_rank)) {
        nmb_send++;
      }
    }
  }

  if (nmb_send == 0) return;  // nothing to do

  // allocate array of send buffers
  Kokkos::realloc(sendbuf, nmb_send);

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    // Hydro AMR/load-balance buffers must include auxiliary carried fields such as
    // dual-energy internal energy, not just the standard hydro+scalar state.
    ncc_tosend += pmy_mesh->pmb_pack->phydro->nvars;
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->pmhd->nvars;
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->prad != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->prad->prgeo->nangles);
  }
  if (pmy_mesh->pmb_pack->pgrav != nullptr && lb_transfer_gravity) {
    ncc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }
  ncc_tosend += FullBlockTransferPrimitiveCount();

  // Step 2. (PackAndSendAMR)
  // loop over old MBs on this rank, initialize send buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int sb_idx = 0;   // send buffer index
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {  // old MB was refined
      for (int l=0; l<nleaf; l++) {
        const int new_child = RefineNewChildGID(oldm, l, nleaf);
        if (new_child < 0) continue;
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[new_child] != global_variable::my_rank)) {
          LogicalLocation &lloc = new_lloc_eachmb[new_child];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          sendbuf.h_view(sb_idx).bis = il + ox1*cnx1;  // il:iu etc. includes ghost zones
          sendbuf.h_view(sb_idx).bie = iu + ox1*cnx1;
          sendbuf.h_view(sb_idx).bjs = jl + ox2*cnx2;
          sendbuf.h_view(sb_idx).bje = ju + ox2*cnx2;
          sendbuf.h_view(sb_idx).bks = kl + ox3*cnx3;
          sendbuf.h_view(sb_idx).bke = ku + ox3*cnx3;
          sendbuf.h_view(sb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
          sendbuf.h_view(sb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
               (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid   = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) { // old MB on same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          SetSameLevelTransferExtent(sendbuf.h_view(sb_idx));
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      } else {                                // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          sendbuf.h_view(sb_idx).bis = cis;
          sendbuf.h_view(sb_idx).bie = cie;
          sendbuf.h_view(sb_idx).bjs = cjs;
          sendbuf.h_view(sb_idx).bje = cje;
          sendbuf.h_view(sb_idx).bks = cks;
          sendbuf.h_view(sb_idx).bke = cke;
          sendbuf.h_view(sb_idx).cntcc = cnx1*cnx2*cnx3;
          sendbuf.h_view(sb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 + cnx1*cnx3
                                          + cnx1*cnx2;
          sendbuf.h_view(sb_idx).use_coarse = true;
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          if (sb_idx > 0) {
            sendbuf.h_view(sb_idx).offset = sendbuf.h_view((sb_idx-1)).offset +
                                            sendbuf.h_view((sb_idx-1)).cnt;
          } else {
            sendbuf.h_view(sb_idx).offset = 0;
          }
          sb_idx++;
        }
      }
    }
  }
  ValidateAMRDescriptorCount("send", sb_idx, nmb_send);
  // Sync metadata and allocate the complete host snapshot plus bounded device staging.
  sendbuf.template modify<HostMemSpace>();
  sendbuf.template sync<DevExeSpace>();
  const std::size_t ndata =
      sendbuf.h_view(nmb_send - 1).offset + sendbuf.h_view(nmb_send - 1).cnt;
  const std::size_t send_stage_elements =
      AMRStagingExtent(sendbuf, nmb_send, lb_chunk_elements);
  send_data = HostArray1D<Real>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "lb send data"), ndata);
  // Grow-only staging.  The chunk decomposition below still uses send_stage_elements, not
  // the buffer capacity, so a retained larger buffer cannot change the chunk boundaries.
  EnsureAMRStagingCapacity(send_stage_elements);

  // Step 3. (PackAndSendAMR)
  // Pack every outgoing value before any local or received destination slot can overwrite
  // its source.  Device work is bounded; the full permutation snapshot lives in host RAM.
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  radiation::Radiation* prad = pmy_mesh->pmb_pack->prad;
  gravity::Gravity* pgrav = pmy_mesh->pmb_pack->pgrav;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  for (int mb_begin = 0; mb_begin < nmb_send;) {
    const int mb_end = AMRChunkEnd(sendbuf, mb_begin, nmb_send, send_stage_elements);
    const std::size_t data_offset = sendbuf.h_view(mb_begin).offset;
    const std::size_t chunk_elements = AMRChunkElements(sendbuf, mb_begin, mb_end);

    int ncc_sent = 0, nfc_sent = 0;
    if (phydro != nullptr) {
      PackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      ncc_sent += phydro->nvars;
    }
    if (pmhd != nullptr) {
      PackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      ncc_sent += pmhd->nvars;
      PackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      nfc_sent += 1;
    }
    if (prad != nullptr) {
      PackAMRBuffersCC(prad->i0, prad->coarse_i0, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      ncc_sent += prad->prgeo->nangles;
    }
    if (pgrav != nullptr && lb_transfer_gravity) {
      PackAMRBuffersCC(pgrav->phi, pgrav->coarse_phi, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      ncc_sent += 1;
    }
    if (pz4c != nullptr) {
      PackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end, data_offset);
      ncc_sent += pz4c->nz4c;
    }
    // Pure rebalance: the fluid primitives travel with the conserved state (see
    // lb_full_block_transfer).  Same-level only, so the coarse argument is never read.
    if (lb_full_block_transfer) {
      if (phydro != nullptr) {
        PackAMRBuffersCC(phydro->w0, phydro->w0, ncc_sent, nfc_sent,
                         mb_begin, mb_end, data_offset);
        ncc_sent += phydro->w0.extent_int(1);
      }
    }

    auto host_chunk = Kokkos::subview(
        send_data, std::make_pair(data_offset, data_offset + chunk_elements));
    auto device_chunk = Kokkos::subview(
        lb_stage, std::make_pair(std::size_t{0}, chunk_elements));
    Kokkos::deep_copy(host_chunk, device_chunk);
    mb_begin = mb_end;
  }
  // lb_stage is retained; the copy above is blocking, so no device work is left pending
  // on it and the recv unpack is free to reuse it.

  // Step 4. (PackAndSendAMR)
  // loop over old MBs on this rank, send data using MPI non-blocking sends
  // Send requests will only be accessed on host, so no need to sync after this step.
  sb_idx = 0;     // send buffer index
  std::vector<int> send_peer_ranks;
  std::vector<int> send_keys;
  send_peer_ranks.reserve(nmb_send);
  send_keys.reserve(nmb_send);
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {      // old MB was refined
      for (int l=0; l<nleaf; l++) {
        const int new_child = RefineNewChildGID(oldm, l, nleaf);
        if (new_child < 0) continue;
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[new_child] != global_variable::my_rank)) {
          int lid = new_child - new_gids_eachrank[new_rank_eachmb[new_child]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          send_peer_ranks.push_back(new_rank_eachmb[new_child]);
          send_keys.push_back(tag);
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) {   // old MB at same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, 0, 0, 0);
          send_peer_ranks.push_back(new_rank_eachmb[newm]);
          send_keys.push_back(tag);
          sb_idx++;
        }
      } else {                                  // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          int ox1 = ((old_lloc.lx1 & 1) == 1);
          int ox2 = ((old_lloc.lx2 & 1) == 1);
          int ox3 = ((old_lloc.lx3 & 1) == 1);
          int lid = newm - new_gids_eachrank[new_rank_eachmb[newm]];
          int tag = CreateAMR_MPI_Tag(lid, ox1, ox2, ox3);
          send_peer_ranks.push_back(new_rank_eachmb[newm]);
          send_keys.push_back(tag);
          sb_idx++;
        }
      }
    }
  }

  ValidateAMRDescriptorCount("posted send", sb_idx, nmb_send);
  rank_send_messages = BuildAMRRankMessages(
      sendbuf, send_peer_ranks, send_keys, nmb_send, "send");
  bool no_errors=true;
  for (auto &msg : rank_send_messages) {
    int ierr = MPI_Isend(send_data.data(), 1, msg.datatype, msg.rank, 0, amr_comm,
                         &msg.request);
    if (ierr != MPI_SUCCESS) no_errors=false;
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersCC()
//! \brief Packs cell-centered data into AMR communication buffers for all MBs being sent
//! Equivalent to PrepareSendSameLevel(), PrepareSendCoarseToFineAMR(), and
//! PrepareSendFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::PackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                      int ncc, int nfc, int mb_begin, int mb_end,
                                      std::size_t data_offset) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  auto &sdata = lb_stage;
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = (mb_end - mb_begin)*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/nvar;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*nvar);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int ni = sbuf.d_view(n).bie - il + 1;
    const int nj = sbuf.d_view(n).bje - jl + 1;
    const int nk = sbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = sbuf.d_view(n).lid;
    const std::size_t offset = sbuf.d_view(n).offset - data_offset +
        static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
        static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (sbuf.d_view(n).use_coarse) {
        // if de-refinement, load data from coarse_a
        sdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = ca(m,v,k,j,i);
      } else {
        // if refinement or same level, load data from a
        sdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = a(m,v,k,j,i);
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersFC()
//! \brief Packs face-centered data into AMR communication buffers for all MBs being sent

void MeshRefinement::PackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                      int ncc, int nfc, int mb_begin, int mb_end,
                                      std::size_t data_offset) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  auto &sdata = lb_stage;
  // Outer loop over (# of MeshBlocks sent)*(3 compnts of field)
  int nn = 3*(mb_end - mb_begin);
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/3;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - 3*local_n);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int m  = sbuf.d_view(n).lid;
    const int nicc = sbuf.d_view(n).bie - il + 1;
    const int njcc = sbuf.d_view(n).bje - jl + 1;
    const int nkcc = sbuf.d_view(n).bke - kl + 1;

    // pack x1 component
    if (v==0) {
      const std::size_t offset = sbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc;
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x1f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x1f(m,k,j,i);
        }
      });

    // pack x2 component
    } else if (v==1) {
      const std::size_t offset = sbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x2f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x2f(m,k,j,i);
        }
      });

    // pack x3 component
    } else {
      const std::size_t offset = sbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc +
          static_cast<std::size_t>(nicc)*(njcc + 1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = cb.x3f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata(offset + (i-il + ni*(j-jl + nj*(k-kl)))) = b.x3f(m,k,j,i);
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearRecvAndUnpackAMR()
//! \brief Checks non-blocking receives have finished, calls function to unpack buffers,
//! deletes receive buffers. Equivalent to some of the work done inside MPI_PARALLEL block
//! in the Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp

void MeshRefinement::ClearRecvAndUnpackAMR() {
#if MPI_PARALLEL_ENABLED
  // Wait for all receives to finish
  bool no_errors=true;
  for (auto &msg : rank_recv_messages) {
    MPI_Status status;
    int ierr = MPI_Wait(&msg.request, &status);
    MPI_Count count = MPI_UNDEFINED;
    if (ierr == MPI_SUCCESS) ierr = MPI_Get_elements_x(&status, MPI_ATHENA_REAL, &count);
    if (ierr != MPI_SUCCESS || count != msg.payload_count) no_errors=false;
    if (msg.datatype != MPI_DATATYPE_NULL) MPI_Type_free(&msg.datatype);
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error while waiting for AMR receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rank_recv_messages.clear();

  // Unpack the complete host snapshot through bounded device chunks.  Receives have all
  // completed, and all outgoing data was snapshotted before same-rank permutation writes.
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  radiation::Radiation* prad = pmy_mesh->pmb_pack->prad;
  gravity::Gravity* pgrav = pmy_mesh->pmb_pack->pgrav;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  const std::size_t recv_stage_elements =
      AMRStagingExtent(recvbuf, nmb_recv, lb_chunk_elements);
  // Grow-only staging, shared with the send pack that has already completed.  As on the
  // send side the chunk decomposition uses recv_stage_elements, never the capacity.
  EnsureAMRStagingCapacity(recv_stage_elements);
  for (int mb_begin = 0; mb_begin < nmb_recv;) {
    const int mb_end = AMRChunkEnd(recvbuf, mb_begin, nmb_recv, recv_stage_elements);
    const std::size_t data_offset = recvbuf.h_view(mb_begin).offset;
    const std::size_t chunk_elements = AMRChunkElements(recvbuf, mb_begin, mb_end);
    auto host_chunk = Kokkos::subview(
        recv_data, std::make_pair(data_offset, data_offset + chunk_elements));
    auto device_chunk = Kokkos::subview(
        lb_stage, std::make_pair(std::size_t{0}, chunk_elements));
    Kokkos::deep_copy(device_chunk, host_chunk);

    int ncc_recv = 0, nfc_recv = 0;
    if (phydro != nullptr) {
      UnpackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      ncc_recv += phydro->nvars;
    }
    if (pmhd != nullptr) {
      UnpackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      ncc_recv += pmhd->nvars;
      UnpackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      nfc_recv += 1;
    }
    if (prad != nullptr) {
      UnpackAMRBuffersCC(prad->i0, prad->coarse_i0, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      ncc_recv += prad->prgeo->nangles;
    }
    if (pgrav != nullptr && lb_transfer_gravity) {
      UnpackAMRBuffersCC(pgrav->phi, pgrav->coarse_phi, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      ncc_recv += 1;
    }
    if (pz4c != nullptr) {
      UnpackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end, data_offset);
      ncc_recv += pz4c->nz4c;
    }
    if (lb_full_block_transfer) {
      if (phydro != nullptr) {
        UnpackAMRBuffersCC(phydro->w0, phydro->w0, ncc_recv, nfc_recv,
                           mb_begin, mb_end, data_offset);
        ncc_recv += phydro->w0.extent_int(1);
      }
    }
    mb_begin = mb_end;
  }
  Kokkos::fence("AMR load-balance receive unpack");
  recv_data = HostArray1D<Real>();
  nmb_recv = 0;
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersCC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.
//! Equivalent to FinishRecvSameLevel(), FinishRecvCoarseToFineAMR(), and
//! FinishRecvFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::UnpackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                        int ncc, int nfc, int mb_begin, int mb_end,
                                        std::size_t data_offset) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  auto &rdata = lb_stage;
  // Outer loop over (# of MeshBlocks recv)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = (mb_end - mb_begin)*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/nvar;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*nvar);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int ni = rbuf.d_view(n).bie - il + 1;
    const int nj = rbuf.d_view(n).bje - jl + 1;
    const int nk = rbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = rbuf.d_view(n).lid;
    const std::size_t offset = rbuf.d_view(n).offset - data_offset +
        static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
        static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (rbuf.d_view(n).use_coarse) {
        // if refinement, load data into coarse_a
        ca(m,v,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
      } else {
        // if de-refinement or same level, load data into a
        a(m,v,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v))));
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersFC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.

void MeshRefinement::UnpackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                        int ncc, int nfc, int mb_begin, int mb_end,
                                        std::size_t data_offset) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  auto &rdata = lb_stage;
  // Outer loop over (# of MeshBlocks recv)*(3 compnts of field)
  int nnv = 3*(mb_end - mb_begin);
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/3;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*3);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int m  = rbuf.d_view(n).lid;
    const int nicc = rbuf.d_view(n).bie - il + 1;
    const int njcc = rbuf.d_view(n).bje - jl + 1;
    const int nkcc = rbuf.d_view(n).bke - kl + 1;

    // unpack x1 component
    if (v==0) {
      const std::size_t offset = rbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc;
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x1f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x1f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x2 component
    } else if (v==1) {
      const std::size_t offset = rbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x2f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x2f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });

    // unpack x3 component
    } else {
      const std::size_t offset = rbuf.d_view(n).offset - data_offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc +
          static_cast<std::size_t>(nicc)*(njcc + 1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x3f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        } else {
          // if de-refinement or same level, load data into a
          b.x3f(m,k,j,i) = rdata(offset + (i-il + ni*(j-jl + nj*(k-kl))));
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearSendAMR()
//! \brief Checks all non-blocking sends completed, deletes send buffers.

void MeshRefinement::ClearSendAMR() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  for (auto &msg : rank_send_messages) {
    int ierr = MPI_Wait(&msg.request, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) no_errors=false;
    if (msg.datatype != MPI_DATATYPE_NULL) MPI_Type_free(&msg.datatype);
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rank_send_messages.clear();
  // MPI no longer references the complete host snapshot after every send has completed.
  send_data = HostArray1D<Real>();
  nmb_send = 0;
#endif
  return;
}
