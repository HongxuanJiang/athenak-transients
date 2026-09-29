#ifndef UTILS_LAUNCH_CONFIG_HPP_
#define UTILS_LAUNCH_CONFIG_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file launch_config.hpp
//  \brief runtime selection of Kokkos::LaunchBounds from what the hardware and the
//         compiled kernel report about themselves.
//
// WHY THIS EXISTS
// ---------------
// Several of the heavy GRMHD kernels carry more live state than the register file can
// hold at any useful occupancy, so ptxas has to choose between keeping the working set in
// registers (few resident warps) and spilling it to local memory (more resident warps).
// `Kokkos::LaunchBounds<MaxThreads, MinBlocksPerSM>` is how we tell ptxas which side of
// that trade to take, and the right side is a property of the GPU, not of the physics.
// Hard-coding a number measured on one card makes the source machine-specific, which is
// exactly what we do not want.
//
// This header replaces the hard-coded number with a decision the binary makes for itself.
// It is deterministic: it reads static properties of the compiled kernel
// (`cudaFuncGetAttributes`) and of the device (`cudaDeviceGetAttribute`), and never
// times anything.  The same binary on the same GPU always chooses the same way, so runs
// stay reproducible and the choice is immune to machine contention.
//
// WHAT IT CANNOT DO -- read this before trusting it
// -------------------------------------------------
// The exact criterion is unreachable from attributes alone.  Kernel time is roughly
//
//     t(c) ~ max( K_lat / T(c) ,  K_bw * (G + 2*L(c)) )
//
// for candidate c, where T(c) is resident threads/SM, L(c) is the per-thread spill frame,
// G is the kernel's real (non-spill) global traffic per thread, and K_lat/K_bw are
// machine constants.  Attributes give us T(c) and L(c).  They do NOT give us G, nor any
// other traffic figure, and without it there is no way to tell whether a given increase
// in spill traffic is a large or a small perturbation on what the kernel already moves.
//
// So this facility does not pretend to decide every case.  It resolves the two ends,
// where the answer follows from the numbers, and DECLINES the middle, where it does not:
//
//   * L(hi) <= L(lo)     -- raising occupancy cost ptxas no extra spill.  Exact: take the
//                           extra warps, they are free.
//   * frames fit on chip -- the resident spill frames fit the SM's L1/shared array, so
//                           they act as an extension of the register file.  Take them.
//   * frames overflow    -- the resident frames exceed that array by 2x or more, so the
//     on chip by 2x+ AND     spill is real DRAM traffic, and that traffic grows at least
//     traffic grows at       as fast as the parallelism: E_hi/E_lo >= T_hi/T_lo.  Only
//     least as fast as       then is the higher occupancy a net loss.  The unknown G
//     the parallelism        cancels because both sides are ratios.  Take the lower one.
//   * anything else      -- NOT DECIDED.  The call site's measured default stands.
//
// The third bullet is the one judgement in the design, and 2x is chosen because it is the
// point where more than half the footprint cannot be cached -- a statement with a
// meaning, not a fitted constant.  It is consistent with the measurement that
// motivated all this:
// on V100-SXM2 the dyn-GR face kernels at <256,2> hold 504 kB of resident spill against a
// 96 kB L1 (5.25x over), Nsight attributed 75% of all warp stalls to `long_scoreboard`
// (i.e. memory), and <256,1> then measured 6.5% faster end to end on a 10-rank BBH
// production run.  A kernel whose footprint lands between 1x and 2x of L1 gets no opinion
// from this rule at all; if you need one there, pin the value (see OVERRIDES) and
// measure.
//
// WHERE THIS IS USED, AND WHERE IT DELIBERATELY IS NOT
// ----------------------------------------------------
// Applied (9 launch sites):
//   * dyn_grmhd_fluxes_impl.hpp -- the three dyn-GR Riemann kernels.
//   * dyn_grmhd_fofc_impl.hpp   -- the four dyn-GR first-order-correction kernels.
//   * mhd_fofc.cpp              -- the two fixed-metric FOFC kernels.
//
// NOT applied, on purpose:
//   * mhd_fluxes.cpp (mflux_*_rsolve).  The rule works there and would move 12 of that
//     file's 36 instantiations to MinBlocksPerSM=1, but compiling both candidates takes
//     the translation unit from 709 s to 1233 s (+74%) and, because it is the build's
//     critical path, the whole clean build from 756 s to 1259 s.  That is a permanent
//     cost on every build in exchange for a change nobody has timed.  It keeps a
//     documented build-time knob instead; see the comment on kRSolveMinBlocksPerSM.
//   * kSplitFacePreparation in mhd_fluxes.cpp.  Not a hardware parameter at all -- it
//     keys on whether a given solver's fused kernel has hoistable state -- and flipping
//     it changes which kernels run and in what order, which is a correctness risk a
//     launch bound does not carry.
//
// The general lesson: this facility is worth its instantiation cost only where the kernel
// is heavy enough that the launch bound genuinely changes ptxas's output AND the file is
// not the build's critical path.  Applying it everywhere would be a regression in build
// time for no measured gain.
//
// OVERRIDES
// ---------
//   * environment `ATHENAK_LAUNCH_MIN_BLOCKS` pins the choice without a rebuild:
//     `=2` pins every auto-tuned site, `=mflux=2` pins only the kernels whose name
//     contains "mflux", `=1,mflux=2` combines the two.  Unset, or "auto", leaves the
//     decision to the rule.  Use it to reproduce a run, to revert one kernel's automatic
//     choice, or to bisect a suspected launch-bounds problem in a bug report.  A pinned
//     value that is not one of the site's two compiled candidates is a fatal error, not a
//     silent fallback.
//   * `-DATHENAK_LAUNCH_AUTOTUNE=0` at build time removes the machinery entirely: only
//     the per-site fallback variant is instantiated, which restores the previous build
//     cost and the previous behaviour exactly.
//   * each call site passes its own `Fallback` value, used when the query is unavailable
//     (non-CUDA backend, autotune disabled, or a failed CUDA query).
//
// MPI
// ---
// The automatic decision is taken independently on every rank; there is no broadcast.  It
// is a pure function of (compiled kernel attributes, device attributes), both identical
// across ranks on a homogeneous machine, so a broadcast would add a collective
// and change nothing.  On a heterogeneous node set independent choice is the *correct*
// behaviour and a broadcast would force some ranks onto the wrong setting.  It also could
// not be broadcast safely: the decision is taken inside a lazily initialised
// function-local static that a rank with no work of that kind never reaches, so a
// collective there would deadlock.
//
// Do NOT read that as "ranks may safely disagree".  A launch bound is visible to the
// whole device compiler, not just the register allocator, so two variants of one kernel
// are not guaranteed to be bit-identical (this tree already has one case, 3b56a30c,
// where a recompilation moved FMA contraction).  Same-level interior faces are computed
// redundantly on both sides of a rank boundary -- only fine/coarse faces are
// flux-corrected -- so ranks on different variants can disagree at the last ULP there.
// That is why the *pin* is taken from rank 0 for every rank (env_switch, see
// PinnedMinBlocksFor): the override must never be the thing that splits the pool.
// Rank 0 alone prints the decision.
//
// PORTABILITY
// -----------
// The query path is CUDA-only.  On HIP, SYCL, OpenMP and Serial the dispatcher compiles
// down to a single `Kokkos::parallel_for` at the site's `Fallback` bounds -- same code as
// before this header existed, no extra template instantiations, no runtime cost.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "utils/env_switch.hpp"

//! Set to 0 to compile out the candidate variants and the runtime query.
#ifndef ATHENAK_LAUNCH_AUTOTUNE
#define ATHENAK_LAUNCH_AUTOTUNE 1
#endif

#if defined(KOKKOS_ENABLE_CUDA) && ATHENAK_LAUNCH_AUTOTUNE
#define ATHENAK_LAUNCH_AUTOTUNE_ACTIVE 1
#else
#define ATHENAK_LAUNCH_AUTOTUNE_ACTIVE 0
#endif

namespace athenak {
namespace launch {

//----------------------------------------------------------------------------------------
//! \struct SMBudget
//! \brief what the target GPU says it can give one streaming multiprocessor.

struct SMBudget {
  bool valid{false};
  int regs_per_sm{0};        //!< cudaDevAttrMaxRegistersPerMultiprocessor
  int threads_per_sm{0};     //!< cudaDevAttrMaxThreadsPerMultiProcessor
  int blocks_per_sm{0};      //!< cudaDevAttrMaxBlocksPerMultiprocessor
  int cache_bytes_per_sm{0}; //!< cudaDevAttrMaxSharedMemoryPerMultiprocessor
  int num_sms{0};            //!< cudaDevAttrMultiProcessorCount
};

//----------------------------------------------------------------------------------------
//! \struct VariantFacts
//! \brief what one compiled LaunchBounds variant of one kernel reports about itself.

struct VariantFacts {
  bool valid{false};
  int min_blocks{0};          //!< the MinBlocksPerSM this variant was compiled with
  int num_regs{0};            //!< cudaFuncAttributes::numRegs
  long long spill_bytes{0};   //!< cudaFuncAttributes::localSizeBytes, per thread
  int block_size{0};          //!< block size Kokkos will actually launch with
  int blocks_per_sm{0};       //!< cudaOccupancyMaxActiveBlocksPerMultiprocessor
  long long threads_per_sm{0};//!< resident threads/SM = blocks_per_sm * block_size
  long long footprint{0};     //!< spill_bytes * threads_per_sm, aggregate frames per SM
  long long offchip{0};       //!< per-thread spill traffic that misses the SM cache
};

//----------------------------------------------------------------------------------------
//! \fn OffChipSpillPerThread
//! \brief the part of a thread's spill frame that cannot stay in the SM's L1.
//!
//! All resident threads' frames are live at once, so the SM must hold `footprint =
//! spill * threads_per_sm` bytes of private scratch.  Modelling the L1 as holding a
//! `cache/footprint` share of it, the traffic that escapes to L2/DRAM per thread is
//!
//!     spill * max(0, 1 - cache/footprint)
//!
//! This is a residency model, not a fitted curve: the only number it takes from the
//! machine is the hardware-reported cache size.  It is deliberately crude -- a uniform
//! reuse assumption -- and its job is only to separate "the frame behaves like registers"
//! from "the frame is DRAM traffic", which are orders of magnitude apart.
//!
//! Computed in integer arithmetic so the answer cannot drift with the compiler's
//! floating-point contraction: the choice must be reproducible bit for bit.

inline long long OffChipSpillPerThread(long long spill_bytes, long long threads_per_sm,
                                       long long cache_bytes) {
  if (spill_bytes <= 0 || threads_per_sm <= 0) return 0;
  const long long footprint = spill_bytes * threads_per_sm;
  if (footprint <= cache_bytes) return 0;
  return (spill_bytes * (footprint - cache_bytes)) / footprint;
}

//----------------------------------------------------------------------------------------
//! \var kCacheThrashFactor
//! \brief How far the resident spill footprint must exceed the SM's on-chip array before
//! we are willing to call the kernel bandwidth-bound.
//!
//! This is the rule's single judgement parameter, and it is set to the one value that has
//! a meaning rather than a tuning: at 2x the cache size, more than half of the resident
//! spill footprint cannot be held on chip, so the majority of spill accesses are L2/DRAM
//! transactions.  Below that the frames are mostly cacheable and the attribute data
//! cannot tell us whether the misses matter -- see ChooseMinBlocksPerSM, which declines
//! to decide there rather than guessing.
constexpr long long kCacheThrashFactor = 2;

//----------------------------------------------------------------------------------------
//! \fn ChooseMinBlocksPerSM
//! \brief the selection rule.  Pure function of the two candidates and the device.
//!
//! Returns the chosen MinBlocksPerSM, or 0 for "the attributes do not settle this, use
//! the call site's documented fallback".
//!
//! The shape is "prefer the higher occupancy unless its spill exceeds what the extra
//! warps can hide", with the two ends resolved exactly and the middle left alone:
//!
//!   1. spill(hi) <= spill(lo).  Reaching the higher occupancy cost ptxas nothing, so
//!      there is no trade to make.  EXACT -- take the extra warps.
//!
//!   2. the resident frames at the higher occupancy fit the SM's on-chip array.  Then
//!      they behave as an extension of the register file: spill accesses hit in L1, no
//!      DRAM traffic is created, and the extra warps are close to free.  Take them.
//!
//!   3. the resident frames at the higher occupancy exceed the on-chip array by
//!      kCacheThrashFactor or more -- so the kernel is paying real DRAM traffic for its
//!      spill -- AND that traffic grows at least as fast as the parallelism does.  This
//!      is a marginal comparison, not a threshold: raising occupancy buys latency
//!      tolerance in proportion to resident threads (T_hi/T_lo) and costs off-chip spill
//!      traffic in proportion to E_hi/E_lo, so the higher occupancy is worth taking
//!      exactly while E_hi/E_lo < T_hi/T_lo.  Cross-multiplied to stay in integers, the
//!      lower occupancy wins when E_hi*T_lo >= E_lo*T_hi.  Note this needs no estimate of
//!      the kernel's real global traffic: both sides are ratios of the same quantity, so
//!      the unknown cancels.
//!
//!      This form replaced an earlier one that took the lower occupancy whenever
//!      off-chip spill rose AT ALL.  That version was falsified by measurement: the
//!      fixed-metric MHD::FOFC kernels grow their frame only 2224 -> 2472 B (+11%) while
//!      occupancy doubles, and timing them on a GRMHD torus showed MinBlocksPerSM=2 is
//!      0.54% FASTER, not slower -- every repetition of the low-occupancy arm was slower
//!      than every repetition of the high-occupancy arm.  The old test had no sense of
//!      magnitude; the marginal form does, and it costs no extra free parameters.
//!
//!   4. anything else -- in particular frames that overflow the cache by less than
//!      kCacheThrashFactor -- is NOT DECIDED HERE.  Deciding it needs the ratio of spill
//!      traffic to the kernel's real global traffic, and cudaFuncGetAttributes reports no
//!      traffic figure of any kind, so there is nothing honest to compute.  The call
//!      site's measured default stands.
//!
//! Why the deferral in case 4 matters: an earlier draft of this rule compared off-chip
//! spill bytes alone, and would then hand a kernel with a 64-byte frame back to half
//! occupancy to save 16 B/thread.  That is a regression, not a tune.  The rule must not
//! move off a measured default on evidence that thin.
//!
//! `why` is filled with a short human-readable reason for the log line.

inline int ChooseMinBlocksPerSM(const VariantFacts &lo, const VariantFacts &hi,
                                const SMBudget &budget, const char **why) {
  if (!lo.valid || !hi.valid || !budget.valid) {
    *why = "attribute query unavailable";
    return 0;
  }
  const long long cache = budget.cache_bytes_per_sm;
  if (hi.spill_bytes <= lo.spill_bytes) {
    *why = "higher occupancy costs no extra spill";
    return hi.min_blocks;
  }
  if (hi.footprint <= cache) {
    *why = "spill frames stay resident on chip";
    return hi.min_blocks;
  }
  if (hi.footprint >= kCacheThrashFactor*cache &&
      hi.offchip * lo.threads_per_sm >= lo.offchip * hi.threads_per_sm) {
    *why = "off-chip spill grows at least as fast as the added warps";
    return lo.min_blocks;
  }
  *why = "attributes do not settle it, keeping the site default";
  return 0;
}

//----------------------------------------------------------------------------------------
//! \fn PinnedMinBlocksFor
//! \brief the ATHENAK_LAUNCH_MIN_BLOCKS override, or 0 for "let the rule decide".
//!
//! Syntax, parsed once per process:
//!
//!   ATHENAK_LAUNCH_MIN_BLOCKS=2                pin every auto-tuned site to 2
//!   ATHENAK_LAUNCH_MIN_BLOCKS=auto             no pin (the default when unset)
//!   ATHENAK_LAUNCH_MIN_BLOCKS=mflux=2          pin only kernels whose name contains
//!                                              "mflux"; everything else stays automatic
//!   ATHENAK_LAUNCH_MIN_BLOCKS=1,mflux=2        pin everything to 1 except mflux* to 2
//!
//! Entries are comma separated and applied left to right, so a later entry wins.  The
//! per-kernel form matters for bug reports and for bisecting: the automatic choice can be
//! reverted for one kernel at a time, without a rebuild and without disturbing the
//! others.
//! Matching is a substring test against the kernel name passed to par_for_auto.

inline int PinnedMinBlocksFor(const std::string &name) {
  struct PinTable {
    int global{0};
    std::vector<std::pair<std::string, int>> keyed;
  };
  static const PinTable table = []() -> PinTable {
    PinTable t;
    // Through env_switch, so the pin is rank 0's for every rank.  A pin that reached only
    // some ranks would put them on different launch bounds, i.e. on different ptxas
    // output, which is the one way this facility can perturb arithmetic (see MPI above).
    bool is_set = false;
    const std::string spec = env_switch::Raw("ATHENAK_LAUNCH_MIN_BLOCKS", &is_set);
    if (!is_set || spec.empty()) return t;
    std::size_t pos = 0;
    while (pos <= spec.size()) {
      const std::size_t comma = spec.find(',', pos);
      const std::string item = spec.substr(pos, comma - pos);
      pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
      if (item.empty() || item == "auto") continue;
      const std::size_t eq = item.find('=');
      if (eq == std::string::npos) {
        const int v = std::atoi(item.c_str());
        if (v > 0) t.global = v;
      } else {
        const int v = std::atoi(item.substr(eq + 1).c_str());
        if (v > 0 && eq > 0) t.keyed.emplace_back(item.substr(0, eq), v);
      }
    }
    return t;
  }();
  int pinned = table.global;
  for (const auto &kv : table.keyed) {
    if (name.find(kv.first) != std::string::npos) pinned = kv.second;
  }
  return pinned;
}

#if ATHENAK_LAUNCH_AUTOTUNE_ACTIVE

//----------------------------------------------------------------------------------------
//! \fn QuerySMBudget
//! \brief per-process device budget, queried once.

inline const SMBudget &QuerySMBudget() {
  static const SMBudget budget = []() -> SMBudget {
    SMBudget b;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return b;
    const bool ok =
        cudaDeviceGetAttribute(&b.regs_per_sm,
                               cudaDevAttrMaxRegistersPerMultiprocessor, dev)
            == cudaSuccess &&
        cudaDeviceGetAttribute(&b.threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor, dev)
            == cudaSuccess &&
        cudaDeviceGetAttribute(&b.blocks_per_sm,
                               cudaDevAttrMaxBlocksPerMultiprocessor, dev)
            == cudaSuccess &&
        cudaDeviceGetAttribute(&b.cache_bytes_per_sm,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev)
            == cudaSuccess &&
        cudaDeviceGetAttribute(&b.num_sms, cudaDevAttrMultiProcessorCount, dev)
            == cudaSuccess;
    b.valid = ok && b.cache_bytes_per_sm > 0 && b.threads_per_sm > 0;
    return b;
  }();
  return budget;
}

//----------------------------------------------------------------------------------------
//! \fn QueryVariant
//! \brief ask CUDA what it made of one LaunchBounds variant of one kernel.
//!
//! `Functor` must be a named type (not a lambda declared at the launch site), because the
//! Kokkos driver type has to be spellable here in order to reach the `__global__` that
//! Kokkos will actually launch.  That is the whole reason `Flat4D` below is a struct.
//! Querying the real driver type -- rather than a stand-in with the same body -- means
//! these numbers describe the kernel that runs, and costs no extra instantiation.

template <int MaxThreads, int MinBlocks, class Functor>
inline VariantFacts QueryVariant(const Functor &f) {
  VariantFacts v;
  v.min_blocks = MinBlocks;
  using Space = Kokkos::Cuda;
  using LB = Kokkos::LaunchBounds<MaxThreads, MinBlocks>;
  // Same policy type as LaunchFlat4D below (async launch hint included), so the queried
  // kernel is the one that gets launched.
  using Policy = Kokkos::RangePolicy<Space, LB,
                                     Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
  using Driver = Kokkos::Impl::ParallelFor<Functor, Policy, Space>;
  using Launcher = Kokkos::Impl::CudaParallelLaunch<Driver, LB>;

  auto *inst = Space().impl_internal_space_instance();
  if (inst == nullptr) return v;
  const cudaFuncAttributes attr = Launcher::get_cuda_func_attributes(inst);

  // The block size Kokkos itself will pick for this policy -- not necessarily MaxThreads.
  // Using the real one matters: resident threads/SM, and hence the spill footprint, are
  // computed from it.
  v.block_size = Kokkos::Impl::cuda_get_opt_block_size<Functor, LB>(inst, attr, f, 1,0,0);
  if (v.block_size <= 0) return v;

  int blocks = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, Launcher::get_kernel_func(), v.block_size, 0) != cudaSuccess) {
    return v;
  }
  if (blocks <= 0) return v;

  const SMBudget &b = QuerySMBudget();
  v.num_regs = attr.numRegs;
  v.spill_bytes = static_cast<long long>(attr.localSizeBytes);
  v.blocks_per_sm = blocks;
  v.threads_per_sm = static_cast<long long>(blocks) * v.block_size;
  v.footprint = v.spill_bytes * v.threads_per_sm;
  v.offchip = OffChipSpillPerThread(v.spill_bytes, v.threads_per_sm,
                                    b.valid ? b.cache_bytes_per_sm : 0);
  v.valid = b.valid;
  return v;
}

//----------------------------------------------------------------------------------------
//! \fn ReportDecision
//! \brief one line on rank 0 recording what was read and what was chosen.

inline void ReportDecision(const std::string &name, const VariantFacts &lo,
                           const VariantFacts &hi, const SMBudget &budget,
                           int chosen, const char *why) {
  if (global_variable::my_rank != 0) return;
  std::cout << "## launch_config " << name << ": ";
  if (lo.valid && hi.valid) {
    std::cout << "<" << lo.block_size << "," << lo.min_blocks << "> reg=" << lo.num_regs
              << " spill=" << lo.spill_bytes << "B occ=" << lo.threads_per_sm
              << "thr/SM offchip=" << lo.offchip << "B | "
              << "<" << hi.block_size << "," << hi.min_blocks << "> reg=" << hi.num_regs
              << " spill=" << hi.spill_bytes << "B occ=" << hi.threads_per_sm
              << "thr/SM offchip=" << hi.offchip << "B | L1/SM="
              << budget.cache_bytes_per_sm << "B ";
  }
  std::cout << "-> MinBlocksPerSM=" << chosen << " (" << why << ")" << std::endl;
}

#endif  // ATHENAK_LAUNCH_AUTOTUNE_ACTIVE

//----------------------------------------------------------------------------------------
//! \struct Flat4D
//! \brief the (n,k,j,i) flattening `par_for` performs, as a *named* functor.
//!
//! Identical index arithmetic to the 4D `par_for` in athena.hpp -- same divisions, same
//! order, same values delivered to the body -- so a kernel launched through here does bit
//! for bit the same work as one launched through `par_for`.  The only reason it is a
//! struct rather than the lambda `par_for` uses is that a lambda's closure type cannot be
//! named outside the function that declares it, and QueryVariant needs to name it.

template <class Function>
struct Flat4D {
  Function fn;
  int nkji, nji, ni, nl, kl, jl, il;
  KOKKOS_INLINE_FUNCTION void operator()(const int &idx) const {
    int n = (idx)/nkji;
    int k = (idx - n*nkji)/nji;
    int j = (idx - n*nkji - k*nji)/ni;
    int i = (idx - n*nkji - k*nji - j*ni) + il;
    n += nl;
    k += kl;
    j += jl;
    fn(n, k, j, i);
  }
};

//----------------------------------------------------------------------------------------
//! \fn LaunchFlat4D
//! \brief launch one compiled LaunchBounds variant.

template <int MaxThreads, int MinBlocks, class ExeSpace, class Functor>
inline void LaunchFlat4D(const std::string &name, ExeSpace exec_space, int nthreads,
                         const Functor &f) {
  Kokkos::parallel_for(name,
      athenak_lw(Kokkos::RangePolicy<ExeSpace, Kokkos::LaunchBounds<MaxThreads,
          MinBlocks>>(
          exec_space, 0, nthreads)),
      f);
}

//----------------------------------------------------------------------------------------
//! \fn par_for_auto
//! \brief drop-in replacement for `par_for<MaxThreads, N>` over a 4D (n,k,j,i) range that
//!        picks MinBlocksPerSM from device and kernel attributes instead of a constant.
//!
//! `Fallback` is what the site used before, and is what it still gets on any backend or
//! build where the query is unavailable.  `Lo`/`Hi` are the compiled candidates.
//!
//! Only two candidates are compiled, and that is a deliberate limit.  MinBlocksPerSM
//! interacts with the 256-thread cap: on every NVIDIA architecture shipped so far
//! (65536 registers/SM) 2 blocks of 256 threads already forces <=128 registers/thread and
//! 4 blocks forces <=64.  For kernels with the live state these carry (~1.6 kB/thread)
//! the 3- and 4-block variants spill monotonically harder -- measured on sm_70 for
//! hlle_gr: frame 1760 B uncapped, 2128 B at <256,2>, 2416 B at <256,3>, 2496 B at
//! <256,4>, with local-memory instruction counts rising 2337 -> 5045 -> 8666 -> 10869 --
//! so they are on the far side of any threshold this rule could plausibly cross, and each
//! extra candidate is another full set of template instantiations of an already heavily
//! instantiated kernel.  {1,2} brackets the useful range; extend the template arguments
//! if a future architecture makes 4 interesting.
//!
//! The decision is resolved once per kernel, on first launch, and cached in a
//! function-local static (one per Functor type, i.e. one per call site per template
//! instantiation).  Resolving lazily rather than at startup is what lets the log report
//! the kernels the run actually uses, with the attributes they were actually compiled to.

template <int MaxThreads, int Fallback, int Lo = 1, int Hi = 2,
          class ExeSpace, class Function>
inline void par_for_auto(const std::string &name, ExeSpace exec_space,
                         const int &nl, const int &nu, const int &kl, const int &ku,
                         const int &jl, const int &ju, const int &il, const int &iu,
                         const Function &function) {
  static_assert(Fallback == Lo || Fallback == Hi,
                "par_for_auto: Fallback must be one of the compiled candidates, else a "
                "deferred decision would silently launch a variant nobody asked for.");
  static_assert(Lo < Hi, "par_for_auto: Lo must be the lower-occupancy candidate.");
  // A light-weight launch passes the functor as a kernel argument only while the functor
  // and the launch policy Kokkos stores with it stay below the 4 KB kernel-argument
  // limit (Kokkos::Impl::CudaTraits::KernelArgumentLimit; the policy is 48 B on the
  // V100 build).  Above it Kokkos selects the global-memory mechanism, whose
  // launch-bounds overload is ambiguous under nvc++ 25.7, so the CUDA build of the
  // launching unit fails (870aa382).  A lambda's captures have the same size in a CPU
  // build, so the CPU build stops here too, 16 B short of where the CUDA build would.
  static_assert(sizeof(Flat4D<Function>) + 64 < 4096,
                "par_for_auto: the kernel's captures exceed the 4 KB kernel-argument "
                "limit of a light-weight CUDA launch; capture less.");
  const int nn = nu - nl + 1;
  const int nk = ku - kl + 1;
  const int nj = ju - jl + 1;
  const int ni = iu - il + 1;
  const int nnkji = nn * nk * nj * ni;
  const int nkji  = nk * nj * ni;
  const int nji   = nj * ni;
  const Flat4D<Function> f{function, nkji, nji, ni, nl, kl, jl, il};

#if ATHENAK_LAUNCH_AUTOTUNE_ACTIVE
  // The query path describes Kokkos::Cuda kernels.  A build that enables CUDA but runs
  // these loops on another space (e.g. a Serial default execution space) must not be
  // handed a decision derived from a device it is not using.
  if constexpr (!std::is_same_v<ExeSpace, Kokkos::Cuda>) {
    LaunchFlat4D<MaxThreads, Fallback>(name, exec_space, nnkji, f);
    return;
  } else {  // NOLINT(readability/braces)
  // One decision per kernel, taken on first launch and never revisited.
  static const int chosen = [&]() -> int {
    const int pinned = PinnedMinBlocksFor(name);
    if (pinned > 0) {
      // Only Lo and Hi are compiled for this site.  Accepting anything else printed
      // "pinned by ..." and then launched Lo regardless -- an override that silently did
      // something other than what it said, which is worthless for the bisecting the
      // override exists for.  The pin is an operator-supplied environment value, never
      // set by a deck, so refusing it outright is safe.
      if (pinned != Lo && pinned != Hi) {
        std::cout << "### FATAL ERROR in par_for_auto" << std::endl
                  << "ATHENAK_LAUNCH_MIN_BLOCKS pins kernel '" << name << "' to "
                  << "MinBlocksPerSM=" << pinned << ", but only " << Lo << " and " << Hi
                  << " are compiled for this launch site." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (global_variable::my_rank == 0) {
        std::cout << "## launch_config " << name << ": -> MinBlocksPerSM=" << pinned
                  << " (pinned by ATHENAK_LAUNCH_MIN_BLOCKS)" << std::endl;
      }
      return pinned;
    }
    const SMBudget &budget = QuerySMBudget();
    const VariantFacts lo = QueryVariant<MaxThreads, Lo>(f);
    const VariantFacts hi = QueryVariant<MaxThreads, Hi>(f);
    const char *why = "";
    int c = ChooseMinBlocksPerSM(lo, hi, budget, &why);
    if (c == 0) c = Fallback;
    ReportDecision(name, lo, hi, budget, c, why);
    return c;
  }();

  if (chosen == Hi) {
    LaunchFlat4D<MaxThreads, Hi>(name, exec_space, nnkji, f);
  } else {
    LaunchFlat4D<MaxThreads, Lo>(name, exec_space, nnkji, f);
  }
  }  // end if constexpr (ExeSpace is Kokkos::Cuda)
#else
  LaunchFlat4D<MaxThreads, Fallback>(name, exec_space, nnkji, f);
#endif
}

} // namespace launch
} // namespace athenak

#endif // UTILS_LAUNCH_CONFIG_HPP_
