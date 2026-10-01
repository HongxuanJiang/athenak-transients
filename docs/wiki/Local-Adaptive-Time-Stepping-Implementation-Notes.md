# Local Adaptive Time Stepping: implementation notes

Developer-level detail for [Local Adaptive Time Stepping (LAT)](Local-Adaptive-Time-Stepping).
Start there for usage, parameters and guidance. This page records the mechanics: how the
factors are chosen, the tick loop, the partition, the interaction with other modules, edge
cases and known issues. LAT in this release covers hydrodynamics only.

## Contents

- [How the factors are chosen](#how-the-factors-are-chosen)
- [The window and tick loop](#the-window-and-tick-loop)
- [Ghost zones and conservation](#ghost-zones-and-conservation)
- [Rank distribution and rebalance](#rank-distribution-and-rebalance)
- [Log lines](#log-lines)
- [Restart](#restart)
- [Module interactions](#module-interactions)
- [Known issues and design choices](#known-issues-and-design-choices)
- [Code map](#code-map)
- [Tests and references](#tests-and-references)

## How the factors are chosen

`Mesh::UpdateHydroLATMetadata` (`src/mesh/mesh.cpp`) gathers every block's own admissible
step $\Delta t_m = \mathrm{cfl}\cdot\min(\Delta t^{\rm fluid}_m,\Delta t^{\rm src}_m)$ with
`MPI_Allgatherv`. The factor $f$ is the largest power of two with
$2f \le \Delta t_m/\Delta t_{\rm global} + \varepsilon$, where $\varepsilon = 64\,\epsilon_{\rm mach}$, capped at 2^lat_levels.

The raw ladder is then narrowed in this order:

1. **Problem cap.** The hook `user_hydro_lat_factor_cap_func`, re-rounded down to a power of two.
2. **Sink pinning.** Every block whose bounding box, grown by its own ghost band, intersects a
   `SinkParticles::LATPinRegions()` sphere is forced to factor 1.
3. **AMR-level collapse.** Unless `lat_same_level = true`, every block on an AMR level gets that
   level's minimum factor, because the conservative fallback only corrects coarse/fine
   interfaces.
4. **Minimum bin population.** A bin holding fewer than `hydro_lat_min_bin_count` blocks is
   merged into the next-faster bin, iterated to a fixed point.
5. **Neighbour limiter.** A fixed-point iteration over all 56 geometric neighbour slots that
   enforces the 2:1 coarse/fine time-step staircase and, for same-level mixed bins, a ratio of
   at most `lat_same_level_max_ratio`. Edge and corner neighbours participate because the
   prolongation stencil reads them, and `lat_neighbor_limiter` chooses how much of that
   envelope is used. Non-convergence after `nmb_total` iterations is fatal.

The surviving maximum is `hydro_lat_sync_factor_current`. Per-factor populations are cached in
`hydro_lat_bin_count`.

**Where the bins are decided.** Not in `Mesh::NewTimeStep`, which computes the global `dt` from
the per-block `dtnew_eachmb` arrays. The factors are computed right afterwards by
`Driver::RebuildHydroLATMetadata`, which calls
`Mesh::UpdateHydroLATMetadata(hydro_subcycle_factor)`, on the ratio $\Delta t_m/\Delta t$.
Under LAT the driver also calls each module's own `NewTimeStep` explicitly before that,
because the task-list versions short-circuit under an active mask and would otherwise freeze
the per-block slots.

## The window and tick loop

**Window length.** `Driver::Execute` recomputes it every cycle (`src/driver/driver.cpp`).
`lat_sync_factor` starts at 2^lat_levels, is clamped by `hydro_lat_sync_factor_current` and by
`outer_substeps`, and is then halved until the window fits below `tlim` under two tests: the
window length, and "only the last tick may be clamped". Self-gravity halves it further so a
window never exceeds `gravity/solve_dt`. A run with sinks halves it so a sink moves at most
`lat_window_motion_cells` cells per window.

**Per cycle.**

1. Compute `lat_fine_dt = pmesh->dt` and `lat_sync_factor`.
2. Apply gravity and sink window shortening, and the `user_hydro_lat_window_func` hook.
3. Window setup: `ClearHydroLAT`, `CopyCons` into the `u1` start register (skipped when the
   union predictor will write it), and `ResetLATBlockTimes`.
4. Run the tick loop, `substep = 0 ... lat_sync_factor-1`, in strides of
   `hydro_lat_tick_stride` (the smallest factor actually present, so empty phases are skipped).
5. Per tick: due mask, `RefreshHydroLATBoundaries`, union or factor-by-factor integration,
   then time advance by `completed_ticks` fine steps.
6. At `end_outer_step` only: history, outputs, restarts, AMR check, `NewTimeStep`,
   `RebuildHydroLATMetadata`, the LAT rebalance and the sink operator.

**Each tick.**

- `MeshBlockPack::SetActiveMeshBlocksByLATDueFactors(sync, tick_phase, true)` builds the due
  mask, the compact `lat_active_indices` list, the per-edge `lat_send_nghbr` flags and the
  three flux flags (`src/mesh/meshblock_pack.cpp`).
- `Driver::RefreshHydroLATBoundaries` brings every due block's ghost band to the common tick
  time. Restriction happens only from blocks that must feed a due coarser neighbour,
  prolongation only into due blocks with a coarser neighbour, followed by a ghost-band-only
  conserved-to-primitive conversion.
- The due bins are then integrated. With `lat_union_stage1`, the RK2 predictors of all due
  bins run as one union kernel with per-block `dt` (`lat_step_dt`), followed by the correctors
  slowest to fastest, so a faster bin sees a completed slower dense history. Otherwise each due
  factor runs its own full RK2 in the same slow-to-fast order.

## Ghost zones and conservation

**Ghosts across a bin boundary.** An inactive sender does not hand over its raw array but a
value interpolated to the receiver's `target_time`, using the block's own bracket
$[u^1@t_0, u^0@t_1]$. The interpolation is linear, or the SSPRK2 dense-output polynomial when
a validated stage-1 state exists (`src/bvals/bvals_cc.cpp`). The chord fallback is decided
once per cell over all components. `lat_theta` is clamped to the range 0 to 1. A value above
1 is reachable on the factor-by-factor path, and the clamp publishes the sender's end-of-span
state, so the clamp is load-bearing, not diagnostic.

**Where the stage-1 snapshot comes from.** On the union-predictor path the driver captures it
after each factor's endpoint refresh. On the factor-by-factor path the fluid saves its own as
the last task of the stage: the graph queues `Hydro::SaveLATDenseOutput` in `after_stagen`. A
graph that queues none of this still runs, but silently falls back to the linear chord at every
mixed-cadence face, because `lat_dense_stage1_valid` is never set.

**Conservation.** Every active block accumulates the time integral of its own slow-side face
flux and subtracts the integral received from finer or faster neighbours. When the bin's step
completes, the hydro update (`src/hydro/hydro_update.cpp`) applies the signed surface
correction to `u0`. This is Berger-Colella refluxing generalised to LAT bins, including
same-level factor mismatches. The receiver condition deliberately does not require the
receiver to be due this tick, because the fine neighbour may have taken several substeps
(`src/mesh/meshblock_pack.cpp`).

## Rank distribution and rebalance

**Objective.** `ApplyHydroLATLoadBalanceCosts` gives each block a window cost
$(F_{\rm sync}/f_m)\times w_m$, where $w_m$ is the measured per-step work
(`hydro_lat_work_eachmb`, a stiff-cell census, with 1 where nothing was measured).
`BuildHydroLATGIDMap` then interleaves: the s-th rank-sized chunk of the final GID order
receives the s-th slice of every bin. Inside a bin the heavy blocks are dealt heaviest-first
onto the least-loaded slice (longest-processing-time on the work excess), and the plain blocks
fill the rest in Z-order. On a static mesh the rank cuts are pinned to equal block counts (the
cap equals `min_cap`), so the order inside each bin is the only lever the partition has.

**Cut search.** It minimises, lexicographically: `sum_tick_max_work` (the critical path over the
window), `max_tick_work`, `idle_slots` (rank and tick pairs with no work), `bin_idle_slots`,
`sum_bin_max_blocks`, `max_blocks`, `block_imbalance`, then the plain scalar cost
(`better_objective`). Only log2(F_sync)+1 distinct due sets exist, so one prefix per phase
class times its multiplicity replaces one prefix per fine tick.

**Rebalance trigger.** All of the following must hold: `hydro_lat_post_amr_rebalance`,
`multilevel`, valid metadata, `!HydroLATLoadBalanceCurrent()`, `topology_stable` (at least
`kHydroLATRebalanceStableWindows * sync_factor` cycles since the last topology change), and
`topology_changed_since_rebalance`. The attempt is stamped where the repartition happens, not
only on the early returns.

**AMR.** Regrids run only at window ends (`amr_due` in the `end_outer_step` gate). After a
regrid the factor of a new block is estimated: carried over by logical location for a
surviving block, halved per refinement level from the nearest old ancestor for a new child,
and the minimum descendant factor for a derefined parent. On a cold start with no metadata the
fallback is level-based, with the finest level at factor 1 and doubling per coarser level.
`mesh_refinement/sticky_load_balance` exists for LAT and is ignored without it.

**MPI and GPU.** No collective may run from inside a rank-local bin, because one rank with no
due block would deadlock it. `lat_host::PendingLATStateSendClear()`
(`src/utils/lat_host_fast.hpp`) latches the refresh's `ClearSend` so the host does not sit in
an `MPI_Wait` while the GPU idles. `Mesh::UpdateHydroLATMetadata` refuses to run with that
latch still set, because a metadata bump frees the persistent requests the latched clear is
waiting on.

## Log lines

```
Mesh: HD LAT per-bin rank distribution f1=[a0,a1,...] f2=[b0,b1,...] util%=[u0,u1,...]
```

For each populated bin `fN`, the number of factor-N blocks landing on each rank under the
chosen cuts. `util%` is each rank's predicted busy fraction over the window, its summed tick
work divided by the chosen critical path `sum_tick_max_work`. This is the number to compare a
measured per-rank GPU utilisation against.

```
Driver: LAT union stage-1 predictor enabled/disabled (<reason>)
```

The reason names which of six capabilities refused it. When applicable it is followed by
`Driver: LAT sink cadence = window boundary, ...`. With `lat_diagnostics = true`,
`Mesh: HD LAT <label> bins f1=... f2=...` is printed after each narrowing pass, together with
`Mesh: HD LAT same-level status ...` and `Mesh: HD LAT AMR-interface status ...`.

## Restart

LAT requires a single global restart file. `single_file_per_rank` with `lat = true` is fatal
(`src/mesh/build_tree.cpp`), because arbitrary GID reordering cannot read rank-local payloads
without a global map. `BuildTreeFromRestart` therefore keeps a current-GID to file-GID map,
composed with any startup LAT interleave (`docs/lat_implementation_note.tex`, section
Restart Semantics).

Checkpoints are written only at window ends, and `restart.cpp` enforces this: a dump requested
while any LAT reflux accumulator is non-zero is fatal.

Bin assignment is not checkpointed. It is recomputed from the local CFL on the first
`NewTimeStep` after the restart. With the default `hydro_lat_min_bin_count = 4*nranks`, a
restart on a different rank count is conservative and valid but follows a different
trajectory.

## Module interactions

**Self-gravity.** The potential is frozen inside a window and `gravity/solve_dt > 0` is
mandatory. The window is shortened to at most `solve_dt`, and the solve is pulled forward
rather than the window truncated, which avoids a pathological 128/16/4/1/1-tick descent.

**Sink particles.** The operator runs once per window from `Driver::Execute`, not from the
per-bin task list, because it does MPI collectives. Every block within a pin sphere grown by
its own ghost band is forced to factor 1, and a warning fires if `SinkStep` is ever reached at
a fine-tick boundary. The window-end reflux refunds the sink potential's work on the replaced
face mass flux (`lat_apply_sink_reflux_x{1,2,3}` in `src/hydro/hydro_update.cpp`), as it does
for self-gravity and the analytic black hole. The potential is
`sinkparticles::SinkPotentialSum`, frozen over the window.

**Integrator.** Only `rk1` and rk2-equivalent tableaux (`rk2`, `imex2`) are admitted. The
delayed-reflux weights are derived for those (`driver.cpp`, weights in
`src/driver/lat_weights.hpp`). `imex3` is unanalysed under LAT.

**`<remap>`.** Mutually exclusive with LAT, refused in two places (`src/remap/remap.cpp`,
`src/main.cpp`).

**First-order flux correction (FOFC).** It runs under the LAT active list like any other
kernel (`src/hydro/hydro_fofc.cpp`). A slower bin's corrector installs the pending fine flux
estimate (`MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC`) before its first FOFC pass,
and that pass also tests the ghost ring. A same-level neighbour of the same cadence cannot add
the estimate, so along the edges where a pending coarse/fine face meets their shared face the
two blocks can test one cell on different fluxes. `Hydro::FOFC` installs the same estimate but
exchanges no flags, so there the flag both sides can form is the one on the fluxes before the
estimate. It tests those edge cells first (`Hydro::StashFOFCEdgeFlags`), and after the first
pass the cells take those flags back (`MeshBoundaryValuesCC::StashPendingEdgeFOFCFlags` and
`ReconcilePendingEdgeFOFCFlags`, `src/bvals/flux_correct_cc.cpp`). Where the test with the
estimate flagged something the test without it did not, the estimate is taken off that cell's
faces for the stage, so its update is the one it was tested on, and the window-end reflux books
the flux it used. Where the two tests agree nothing changes. A cold shear wave on three SMR
levels (`tst/inputs/hydro_lat_fofc_edges.athinput`) holds the mass drift in the slowest bin's
first window to 1e-16.

**Module guards.** All guards are in `src/driver/driver.cpp` unless noted.

| module | guard |
| --- | --- |
| hydro | `src/driver/driver.cpp` |
| `srcterms` self-gravity and external BH gravity | `src/srcterms/srcterms.cpp`, `driver.cpp` (active list and per-block `dt`) |
| turbulence driving | `src/srcterms/turb_driver.cpp` and `driver.cpp` |
| legacy `<radiation>` | `pmbp->prad != nullptr` |
| multigrid / gravity | `mg_gravity.cpp`, work reflux in `src/hydro/hydro_update.cpp` |
| `<remap>` | `src/remap/remap.cpp`, `src/main.cpp` |
| z4c | `pmbp->pz4c != nullptr` for dynamical coordinates |
| particles | `pmbp->ppart != nullptr` (there is no LAT code in the module) |
| sink particles | `driver.cpp`, `src/mesh/mesh.cpp` |

## Known issues and design choices

- **The default bin floor makes the trajectory rank-dependent.** `hydro_lat_min_bin_count`
  defaults to `4*nranks`. The comment in `src/mesh/mesh.cpp` argues that no rank-independent
  constant can be right at both ends of the range, so the dependence is deliberate. Rank 0
  warns.
- **A window may not be truncated.** Leaving a window midway abandons every pending reflux
  accumulator, so `src/driver/driver.cpp` turns that into a fatal error instead of a silent
  conservation loss. Three guarantees make it unreachable: the `nlim` clamp, the `tlim`
  pre-shrink of `lat_sync_factor`, and the once-per-window wall-clock refresh. The window-length
  test has a second condition that rejects a window whose first tick already lands on `tlim`.
- **No endpoint snap.** `pmesh->time` at a window end is `sync_factor` successive additions of
  `lat_fine_dt`, while a factor-F block's `lat_time_end` is a partial sum plus
  F times the fine step. The two disagree by $O(\texttt{sync}\cdot\epsilon\cdot|t|)$, about
  2e-7 of one fine tick. Endpoint selection is integer, so a reflux endpoint can never be
  mis-selected. Revisit only if the fine step over $|t|$ drops below 1e-9.
- **Same-level mixed bins are a physics-validation risk at large ratios.** The interpolation is
  dense RK2 between start and end conserved states (`docs/lat_implementation_note.tex`, section
  Limitations).
- **`time/hydro_lat_max_nmb_per_rank` is dead and fatal.** It is read by no code path and is
  refused via `ParameterInput::RetireDeadParameter`. A separate lower LAT cap could not be
  honoured once AMR filled the blocks between the two caps.
- **Performance.** The dominant remaining cost is communication and task-list launch overhead
  for the per-tick refresh at large sync factors (`docs/lat_implementation_note.tex`, section
  Performance Notes).
- **Defer-final-exchange.** What is dropped is the chain SendU, RecvU, Prolongate, BCs and C2P.
  Because `hydro_lat_min_bin_count` defaults to `4*nranks`, bin membership, and hence whether
  anything is ever deferred, depends on the rank count. The key's presence is tested before it
  is read.
- **Key discovery.** The key list was found with a grep over `src/` for the parameter getters
  matching `lat`, plus targeted greps for `max_nmb_per_rank`, `solve_dt` and
  `sticky_load_balance`.

## Code map

| file | role |
| --- | --- |
| `src/parameter_input.cpp` | `IsLATEnabled()`, the canonical `<time>/lat` switch with the `hydro_lat` legacy fallback, and the legacy-name map |
| `src/mesh/mesh.cpp` | `UpdateHydroLATMetadata`: per-block CFL gather, factor ladder, all narrowing passes, diagnostics lines |
| `src/mesh/load_balance.cpp` | `ApplyHydroLATLoadBalanceCosts` (window cost, min-bin-count re-clamp, per-rank block cap), `BuildHydroLATGIDMap` (bin-interleaved GID order), the cost partition and the `per-bin rank distribution` line |
| `src/mesh/meshblock_pack.cpp` | block time registers, `ConfigureLATUnionStage1`, `SetLATFluxCorrectionByCompletionPhase`, `BuildLATFactorCache`, `BuildLATDueFactorCache`, `SetActiveMeshBlocksByLATFactor`, `...ByLATDueFactors`, `SelectLATDueUpdateFluxReceivers` |
| `src/driver/driver.cpp` | admissibility checks, `ConfigureHydroLATSubstep`, `RebalanceHydroLATMesh`, `RefreshHydroLATBoundaries`, the window and tick loop, gravity and sink cadence, endpoint work, AMR and rebalance triggers |
| `src/driver/lat_weights.hpp` | `lat::FinalFluxWeight`, `FluxFaceCount`, `FluxFaceNeighborIndex` |
| `src/utils/lat_host_fast.hpp` | `PendingLATStateSendClear()` |
| `src/bvals/bvals_cc.cpp` | send gating on `lat_send_nghbr`, `LATInterpolateCC`, `LATDensePolyCC` |
| `src/bvals/flux_correct_cc.cpp` | LAT-gated flux-correction packing and delayed accumulation |
| `src/mesh/mesh_refinement.cpp` | post-AMR redistribution honouring the LAT ordering, sticky load balance |
| `src/hydro/hydro_update.cpp` | window-end reflux, gravity and sink work refunds |

## Tests and references

- Example deck: `inputs/TDE_examples/tde_05_fallback_lat.athinput`.
- Regression input: `tst/inputs/hydro_lat_fofc_edges.athinput` (FOFC at SMR edges).
- Design note: `docs/lat_implementation_note.tex` / `.pdf`.
- Berger and Colella (1989), *J. Comput. Phys.* **82**, 64.
- Gottlieb (2009), the SSPRK(2,2) tableau used for mixed-cadence ghosts (`src/driver/driver.cpp`).
