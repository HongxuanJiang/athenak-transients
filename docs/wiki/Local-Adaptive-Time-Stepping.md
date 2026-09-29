# Local Adaptive Time Stepping (LAT)

**LAT for MHD and GRMHD.** In this release, LAT is available for hydrodynamics, including self-gravity. The development version of the code also supports LAT for MHD and GRMHD, including GRMHD on dynamical spacetimes. These paths are not included in this public release; they are available on request from Hong-Xuan Jiang (masterjoe2000@outlook.com).

## Summary

LAT lets each MeshBlock advance on its own time step instead of the global minimum. Blocks
are binned by their own CFL step into powers-of-two multiples $f \in \{1,2,4,\dots\}$ of the
finest step; a **window** is $F_{\rm sync}=\max_m f_m$ fine steps long, and inside it the
driver runs $F_{\rm sync}$ **ticks**, on tick $k$ updating exactly the blocks with
$k \bmod f_m = 0$. Ghost zones of a not-due neighbour are supplied by time interpolation
from that block's own $[u^1@t_0,\,u^0@t_1]$ bracket, and the coarse/fine (and slow/fast)
flux mismatch is refluxed at each bin's endpoint, so the scheme stays conservative. It is
switched on with `<time>/lat = true`. The win is proportional to how few blocks sit in the finest bin; the cost is a
per-tick boundary refresh, a rank partition that must keep every bin busy on every tick,
and a long list of modules that are refused outright (viscosity, resistivity, shearing box,
turbulence driving, particles, Z4c, ion-neutral, `<remap>`). AMR,
outputs, restarts, self-gravity solves and the sink operator only ever run at synchronized
window boundaries.

## Physics and algorithm

**Binning.** `Mesh::UpdateHydroLATMetadata` (`src/mesh/mesh.cpp`) gathers every block's
own admissible step
$\Delta t_m = \mathrm{cfl}\cdot\min(\Delta t^{\rm fluid}_m,\Delta t^{\rm src}_m)$ (`MPI_Allgatherv`). The factor is the largest power of two with
$2f \le \Delta t_m/\Delta t_{\rm global} + \varepsilon$, $\varepsilon = 64\,\epsilon_{\rm mach}$, capped at $2^{\texttt{lat\_levels}}$.

The raw ladder is then narrowed, in this order:

1. a problem-generator cap hook `user_hydro_lat_factor_cap_func`, re-rounded down to a power
   of two;
2. **density pinning** — when enabled, a block with a large active-cell density contrast
   or a density above the configured threshold, and all its neighbours, use factor 1.
   An MPI minimum reduction includes pins contributed by neighbours on other ranks
   (`Mesh::ApplyHydroLATDensityPin`);
3. **sink pinning** — every block whose bounding box grown by its own ghost band intersects a
   `SinkParticles::LATPinRegions()` sphere is forced to factor 1;
4. **the AMR-level collapse** — unless `lat_same_level = true`, every block on an AMR level is
   given that level's minimum factor, because the conservative fallback only corrects
   coarse/fine interfaces;
5. **the minimum bin population** — a bin holding fewer than `hydro_lat_min_bin_count` blocks
   is merged into the next-faster bin, iterated to a fixed point;
6. **the neighbour limiter** — a fixed-point iteration over all 56 geometric neighbour slots
   that enforces the 2:1 coarse/fine time-step staircase and, for same-level mixed bins, a
   ratio of at most `lat_same_level_max_ratio`. Edge and corner
   neighbours participate because the prolongation stencil reads them; `lat_neighbor_limiter`
   chooses how much of that envelope is used. Non-convergence after `nmb_total` iterations
   is fatal.

The surviving maximum is `hydro_lat_sync_factor_current`, and per-factor
populations are cached in `hydro_lat_bin_count`.

**The tick loop.** `Driver::Execute` recomputes the window length each cycle
(`src/driver/driver.cpp`): `lat_sync_factor` starts at
$2^{\texttt{lat\_levels}}$, is clamped by `hydro_lat_sync_factor_current` and by
`outer_substeps`, and is then halved until the whole window fits below `tlim` under two
separate tests (window length, and "only the last tick may be clamped"). Self-gravity halves
it further so a window never exceeds `gravity/solve_dt`, and a sink
run halves it so a sink moves at most `lat_window_motion_cells` cells per window.

Each tick:

- `MeshBlockPack::SetActiveMeshBlocksByLATDueFactors(sync, tick_phase, true)` builds the due
  mask, the compact `lat_active_indices` list, the per-edge `lat_send_nghbr` flags and the
  three flux flags (`src/mesh/meshblock_pack.cpp`);
- `Driver::RefreshHydroLATBoundaries` brings every due block's ghost
  band to the common tick time — restriction only from blocks that must feed a due coarser
  neighbour, prolongation only into due blocks with a coarser neighbour, and a ghost-band-only
  C2P;
- then the due bins are integrated. With `lat_union_stage1` the RK2 **predictors of all due
  bins run as one union kernel** with per-block $dt$ (`lat_step_dt`), followed by the
  correctors slowest-to-fastest so a faster bin sees a completed slower dense history; otherwise each due factor runs its own full RK2 in the same
  slow-to-fast order.

**Ghosts across a bin boundary.** An inactive sender does not hand over its raw array but a
value interpolated to the receiver's `target_time`, using the block's own bracket
$[u^1@t_0, u^0@t_1]$: linear, or the SSPRK2 dense-output polynomial when a validated stage-1
state exists (`src/bvals/bvals_cc.cpp`). The chord fallback is decided once per cell over all
components. `lat_theta` is clamped to $[0,1]$; $\theta>1$ is reachable
on the factor-by-factor path and the clamp publishes the sender's end-of-span state.

*Where the stage-1 snapshot comes from.* On the union-predictor path the driver captures it
itself after each factor's endpoint refresh. On the
factor-by-factor path the fluid saves its own, as the last task of the stage: the
graph queues `Hydro::SaveLATDenseOutput` in
`after_stagen`. A graph that queues none of
this still runs, but silently falls back to the linear chord at every mixed-cadence face,
because `lat_dense_stage1_valid` is then never set.

**Conservation.** Every active block accumulates the time integral of its own slow-side
face flux and subtracts the integral received from finer/faster neighbours; when the bin's
step completes, the hydro update (`src/hydro/hydro_update.cpp`) applies the signed
surface correction to `u0` — Berger–Colella refluxing generalised to LAT bins, including
same-level factor mismatches. The receiver condition deliberately does **not** require the
receiver to be due this tick, because the fine neighbour may have taken several substeps
(`src/mesh/meshblock_pack.cpp`).

## Code map

| file | role |
| --- | --- |
| `src/parameter_input.cpp` | `IsLATEnabled()`: the canonical `<time>/lat` switch with the `hydro_lat` legacy fallback |
| `src/mesh/mesh.cpp` | `UpdateHydroLATMetadata`: per-block CFL gather, factor ladder, all five narrowing passes, diagnostics lines |
| `src/mesh/load_balance.cpp` | `ApplyHydroLATLoadBalanceCosts`: window cost per block, min-bin-count re-clamp, the single per-rank block cap |
| `src/mesh/load_balance.cpp` | `BuildHydroLATGIDMap`: bin-interleaved GID order, work-weighted deal inside each bin |
| `src/mesh/load_balance.cpp` | the LAT cost partition: per-tick objective, cut search, the `per-bin rank distribution` log line |
| `src/mesh/meshblock_pack.cpp` | block time registers, `ConfigureLATUnionStage1`, `SetLATFluxCorrectionByCompletionPhase` |
| `src/mesh/meshblock_pack.cpp` | `BuildLATFactorCache` / `BuildLATDueFactorCache`: cached masks + the `needs_restrict` / `needs_prolongate` flags |
| `src/mesh/meshblock_pack.cpp` | `SetActiveMeshBlocksByLATFactor`, `...ByLATDueFactors`, `SelectLATDueUpdateFluxReceivers` |
| `src/driver/driver.cpp` | LAT admissibility: which fluids, integrators, metrics, source terms and modules are allowed |
| `src/driver/driver.cpp` | `ConfigureHydroLATSubstep`: sets `pm->dt = fine_dt*factor`, arms the active mask and block times |
| `src/driver/driver.cpp` | `RebalanceHydroLATMesh`: migration-only repartition at a synchronized point |
| `src/driver/driver.cpp` | `RefreshHydroLATBoundaries`: the tick-start common-time ghost refresh |
| `src/driver/driver.cpp` | the window/tick loop, gravity and sink cadence, endpoint work, AMR and rebalance triggers |
| `src/driver/lat_weights.hpp` | `lat::FinalFluxWeight`, `FluxFaceCount`, `FluxFaceNeighborIndex` — shared helpers |
| `src/utils/lat_host_fast.hpp` | `PendingLATStateSendClear()`: latches the refresh `ClearSend` off the critical path |
| `src/bvals/bvals_cc.cpp` | send gating on `lat_send_nghbr`, `LATInterpolateCC` / `LATDensePolyCC` time interpolation |
| `src/bvals/flux_correct_cc.cpp` | LAT-gated flux-correction packing and delayed accumulation |
| `src/mesh/mesh_refinement.cpp` | post-AMR redistribution honouring the LAT ordering; sticky load balance |

## Configuration

Every key below was found with

```bash
grep -rn -E 'GetOrAddReal|GetOrAddInteger|GetOrAddBoolean|GetOrAddString|GetReal|GetInteger|GetBoolean|GetString|DoesParameterExist' \
     src/ --include=*.cpp --include=*.hpp | grep -i lat
```

(plus targeted greps for `"max_nmb_per_rank"`, `"solve_dt"`, `"sticky_load_balance"`).

| block/key | type | default | meaning | read at |
| --- | --- | --- | --- | --- |
| `time/lat` | bool | absent = off | master switch. `ParameterInput::IsLATEnabled()` returns `time/lat` if present, else `time/hydro_lat` | `src/parameter_input.cpp` |
| `time/lat_levels` | int | 1, clamped to [1,20] | number of factor doublings; the configured maximum factor is $2^{\texttt{lat\_levels}}$ (`hydro_subcycle_factor = 1 << hydro_lat_levels`) | `src/driver/driver.cpp`; also `src/mesh/load_balance.cpp` |
| `time/lat_same_level` | bool | `false` | allow different factors on blocks of the same AMR level; when false each level collapses to one bin | `src/mesh/mesh.cpp`, `src/main.cpp` |
| `time/lat_pin_density_contrast` | Real | `0` (disabled) | pin blocks with active-cell conserved-density contrast at least this value, and all their neighbours; enabled only above 1 | `Mesh::ApplyHydroLATDensityPin` |
| `time/lat_pin_density` | Real | `0` (disabled) | pin blocks with maximum active-cell conserved density at least this value in code units, and all their neighbours; enabled only above 0 | `Mesh::ApplyHydroLATDensityPin` |
| `time/lat_same_level_max_ratio` | int | 1 | maximum same-level factor ratio; must be 1, 2, 4 or 8 (else FATAL) | `src/mesh/mesh.cpp` |
| `time/lat_neighbor_limiter` | string | `"all"` | `all` \| `hybrid` \| `face` — how much of the neighbour limiter's envelope is used (see `src/mesh/mesh.cpp`); anything else is FATAL | `src/mesh/mesh.cpp`, `src/main.cpp` |
| `time/hydro_lat_min_bin_count` | int | `-1` → computed `4*nranks` | minimum blocks in a bin before it is merged into the next-faster bin. **The default makes the trajectory depend on the rank count**, and rank 0 prints a warning saying so whenever it is left at the default with `nranks > 1`. `< -1` is FATAL | `src/mesh/mesh.cpp`; mirrored in `src/mesh/load_balance.cpp` |
| `time/lat_diagnostics` | bool | `false` | print the per-pass bin tables and the same-level / AMR-interface status lines; also enables `CheckLATUnionStepDt` in release builds | `src/mesh/mesh.cpp`, `src/driver/driver.cpp` |
| `time/lat_union_stage1` | string | `"auto"` | `auto` \| `true` \| `false`. Whether all due bins share one fused RK2 predictor. `auto` derives it from capabilities; `true` selects the same capability-derived value and never overrides the integrator / user-hook conditions; any other value is FATAL | `src/driver/driver.cpp` |
| `time/hydro_lat_gid_reorder` | bool | `true` | permute GIDs so each rank owns a slice of every bin (the interleave). Read in five places, all with the same default | `src/driver/driver.cpp`; `src/mesh/load_balance.cpp`; `src/mesh/build_tree.cpp`; `src/mesh/mesh_refinement.cpp` |
| `time/hydro_lat_post_amr_rebalance` | bool | value of `hydro_lat_gid_reorder` | allow the LAT rebalance after a mesh change. `true` with `gid_reorder=false` is FATAL | `src/driver/driver.cpp` |
| `time/hydro_lat_defer_final_exchange` | bool | unset → capability-derived | drop the redundant bin-final state exchange. Presence is tested, then read. explicit `false` vetoes the derived choice. What is dropped is the chain SendU/RecvU/Prolongate/BCs/C2P. Note `time/hydro_lat_min_bin_count` defaults to `4*nranks`, so bin membership, and hence whether anything is ever deferred, depends on the rank count | `src/driver/driver.cpp` |
| `time/hydro_lat_max_nmb_per_rank` | int | — | **FATAL** on a fresh start: read by no code path, refused via `ParameterInput::RetireDeadParameter`; the LAT partition cap is `mesh_refinement/max_nmb_per_rank`. A restart-header instance is instead dropped with one rank-0 line | `src/mesh/mesh.cpp` |
| `time/hydro_subcycle`, `time/hydro_subcycle_factor` | bool, int | `false`, 1 | the non-LAT global subcycling path. Under LAT both are **overwritten**: `hydro_subcycle = true`, `hydro_subcycle_factor = 1 << lat_levels` | `src/driver/driver.cpp` |
| `mesh_refinement/max_nmb_per_rank` | int | unset → `ceil(nb/nranks)` | the one per-rank block cap the LAT partition honours (only consulted on adaptive meshes; on a static mesh the cap collapses to the minimum, which pins the rank cuts to equal block counts) | `src/mesh/load_balance.cpp`; also `src/mesh/build_tree.cpp` |
| `mesh_refinement/sticky_load_balance` | bool | `false` | only honoured when LAT is on; a message says it is ignored otherwise, and it is skipped whenever a LAT factor rebalance is active | `src/mesh/mesh_refinement.cpp` |
| `gravity/solve_dt` | Real | — | **required** (`> 0`) with LAT + self-gravity: the potential is frozen inside a window, so `solve_dt` bounds the window length. Missing or `<= 0` is FATAL in two places | `src/main.cpp` (keys on the `<gravity>` block), `src/driver/driver.cpp` (keys on the active fluid's `psrc`), `src/gravity/mg_gravity.cpp` |
| `sink_particles/lat_window_motion_cells` | Real | `1.0` | how many finest cells a sink may traverse in one LAT window; larger values buy longer windows | `src/sink_particles/sink_particles.cpp`, `src/driver/driver.cpp` |

**Legacy spellings.** `ParameterInput` maps ten legacy names onto the canonical ones
(`src/parameter_input.cpp`): `hydro_lat`→`lat`, `hydro_lat_levels`→`lat_levels`,
`hydro_lat_same_level`, `hydro_lat_same_level_max_ratio`, `hydro_lat_neighbor_limiter`,
`hydro_lat_diagnostics`, `hydro_lat_union_stage1`, and in the other direction
`lat_min_bin_count`→`hydro_lat_min_bin_count`, `lat_gid_reorder`→`hydro_lat_gid_reorder`,
`lat_post_amr_rebalance`→`hydro_lat_post_amr_rebalance`. Both spellings work.

## Density pinning at a stellar surface

The surface pin is
available in the shared mesh implementation for hydro. Enable it in `<time>`:

```ini
<time>
lat = true
lat_pin_density_contrast = 1.0e4
lat_pin_density = 0.0
```

Both options default to zero, leaving existing input decks unchanged. A contrast greater
than one selects a block when its maximum active-cell conserved density is at least that
multiple of its minimum (a nonpositive minimum with a positive maximum also qualifies).
A positive `lat_pin_density` selects blocks whose maximum conserved density reaches that
absolute threshold, in code units. Either criterion selects the block; values at or below
one disable the contrast criterion, and values at or below zero disable the absolute
criterion. Non-finite thresholds are rejected when LAT is enabled. Ghost cells are excluded.

Selected blocks and all face, edge and corner neighbours use factor 1. The pin is recomputed
at synchronized metadata rebuilds, after the problem cap and before all other limiters.
With `lat_same_level = false`, the subsequent level collapse may pin an entire refinement
level, reducing the LAT speedup. `lat_diagnostics = true` prints `density pinned bins` before
that collapse and the final bin counts afterward.

For a case migrating from its own factor-cap hook, move `problem/lat_pin_density_contrast`
and `problem/lat_pin_density` into `<time>` and remove the duplicate pin callback. Other
problem-specific caps can still use `user_hydro_lat_factor_cap_func`.

The pin prevents delayed reflux across sharp density transitions from exposing temporary
dense states to gravity, floors and radiation sources before mass is refunded. It does not
guarantee the absence of hot dilute gas from other mechanisms. The gravitational source
gate and the window-end reflux work refund are already matched independently of this pin:
`SourceTerms::Gravity` records, per stage and per coarse/fine face cell, the part of the
mass mismatch each gate refused, and the refund pays the admitted part with no density
test of its own (`src/hydro/hydro_update.cpp`).

## How it runs

**Per cycle.** `Driver::Execute` does, for each outer cycle:

1. compute `lat_fine_dt = pmesh->dt` and `lat_sync_factor` (§Physics above);
2. gravity/sink window shortening, and the `user_hydro_lat_window_func` hook;
3. window setup: `ClearHydroLAT`, `CopyCons` into the `u1` start
   register (skipped when the union predictor will write it), and `ResetLATBlockTimes`;
4. the tick loop, `substep = 0 … lat_sync_factor-1` in strides of `hydro_lat_tick_stride`
   (the smallest factor actually present, so empty phases are skipped,
   `driver.cpp`);
5. per tick: due mask → `RefreshHydroLATBoundaries` → union or factor-by-factor integration
   → time advance by `completed_ticks` fine steps;
6. at `end_outer_step` only: history, outputs, restarts, AMR check, `NewTimeStep`,
   `RebuildHydroLATMetadata`, the LAT rebalance, and the sink operator.

**Where the bins are decided.** Not in `Mesh::NewTimeStep` itself. `NewTimeStep` computes the
global `dt` from the per-block `dtnew_eachmb` arrays; the *factors* are computed immediately
afterwards by `Driver::RebuildHydroLATMetadata` → `Mesh::UpdateHydroLATMetadata(hydro_subcycle_factor)`
(`driver.cpp` after `pmesh->NewTimeStep(tlim)`), on the ratio
$\Delta t_m/\Delta t$. Under LAT the driver additionally calls each module's own
`NewTimeStep` explicitly before that, because the task-list versions
short-circuit under an active mask and would otherwise freeze the per-block slots.

**Rank distribution.** LAT changes the load-balance objective from "equal cost per rank" to
"every rank busy on every tick". `ApplyHydroLATLoadBalanceCosts` gives each block a *window*
cost $= (F_{\rm sync}/f_m)\times w_m$, where $w_m$ is the measured per-step work
(`hydro_lat_work_eachmb`, a stiff-cell census; 1 where nothing was measured). `BuildHydroLATGIDMap` then interleaves: the $s$-th rank-sized
chunk of the final GID order receives the $s$-th slice of *every* bin, and inside a bin the
heavy blocks are dealt heaviest-first onto the least-loaded slice (LPT on the work excess)
with the plain blocks filling the rest in Z-order. The comment
there records why: **on a static mesh the rank cuts are pinned to equal block counts** —
the cap equals `min_cap` — so the order inside each bin is the only lever the partition has.

The cut search itself minimises, lexicographically:
`sum_tick_max_work` (the critical path over the window), `max_tick_work`, `idle_slots`
(rank/tick pairs with no work), `bin_idle_slots`, `sum_bin_max_blocks`, `max_blocks`,
`block_imbalance`, then the plain scalar cost (`better_objective`). Only
`log2(F_sync)+1` distinct due sets exist, so one prefix per phase class times its
multiplicity replaces one prefix per fine tick.

**The two log lines.**

```
Mesh: HD LAT per-bin rank distribution f1=[a0,a1,...] f2=[b0,b1,...] util%=[u0,u1,...]
``` — for each populated bin `fN`, the number of factor-`N` blocks
landing on each rank under the chosen cuts; `util%` is each rank's predicted busy fraction
over the window, its summed tick work divided by the chosen critical path
`sum_tick_max_work`. This is the number to compare a measured per-rank GPU utilisation
against.

```
Driver: LAT union stage-1 predictor enabled/disabled (<reason>)
```
(naming which of six capabilities refused it), and, when applicable,
followed by `Driver: LAT sink cadence = window
boundary, ...`. With `lat_diagnostics = true`, `Mesh: HD LAT <label> bins f1=… f2=…` is
printed after each narrowing pass, together with
`Mesh: HD LAT same-level status …` and `Mesh: HD LAT AMR-interface status …`.

**Rebalance trigger**. All of: `hydro_lat_post_amr_rebalance`,
`multilevel`, valid metadata, `!HydroLATLoadBalanceCurrent()`, `topology_stable` (at least
`kHydroLATRebalanceStableWindows * sync_factor` cycles since the last topology change), **and**
`topology_changed_since_rebalance`. The attempt is stamped where the repartition happens, not only on
the early returns.

**Restart.** LAT requires a single global restart file; `single_file_per_rank` with `lat=true`
is FATAL (`src/mesh/build_tree.cpp`), because arbitrary GID reordering cannot read
rank-local payloads without a global map. `BuildTreeFromRestart` therefore keeps a
current-GID → file-GID map, composed with any startup LAT interleave
(`docs/lat_implementation_note.tex` §Restart Semantics). Checkpoints are only written at
window ends, and `restart.cpp` makes that an enforced invariant: a dump requested
while any LAT reflux accumulator is non-zero is FATAL.

Bin assignment itself is *not* checkpointed: it is recomputed from the local CFL on the first
`NewTimeStep` after the restart. Because the default `hydro_lat_min_bin_count = 4*nranks`, a
restart of identical state on a different rank count is still conservative and valid but is a
**different trajectory** — set the key explicitly to reproduce a run on a different number of
ranks.

**MPI/GPU notes.** No collective may be executed from inside a rank-local bin — one rank with
no due block deadlocks it. `lat_host::PendingLATStateSendClear()`
(`src/utils/lat_host_fast.hpp`) latches the refresh's `ClearSend` so the host does not sit in
an `MPI_Wait` while the GPU idles; `Mesh::UpdateHydroLATMetadata` refuses to run with that
latch still set, because a metadata bump frees the persistent requests
the latched clear is waiting on.

## Interactions

- **AMR/SMR.** LAT is the reason `mesh_refinement/sticky_load_balance` exists, and it is
  ignored without LAT. AMR regrids run only at window ends
  (`amr_due` in the `end_outer_step` gate, `driver.cpp`). After a regrid the factor of a
  new block is *estimated* — carried over by logical location for a surviving block, halved
  per refinement level from the nearest old ancestor for a new child, taken as the minimum
  descendant factor for a derefined parent. On a cold start with
  no metadata at all the fallback is level-based: finest level factor 1, doubling per coarser
  level.
- **Restart.** See above: single global file only,
  bins recomputed and rank-count dependent, accumulators asserted empty at every dump.
- **FOFC.** The first-order flux correction runs under the LAT active list like any other
  kernel (`src/hydro/hydro_fofc.cpp`).
  A slower bin's corrector installs the pending fine flux estimate
  (`MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC`) before its first FOFC pass, and that
  pass also tests the ghost ring. A same-level neighbour of the same cadence cannot add the
  estimate, so along the edges where a pending coarse/fine face meets their shared face the two
  blocks can test one cell on different fluxes.
  `Hydro::FOFC` installs the same estimate but exchanges no flags, so
  there the flag both sides can form is the one on the fluxes before the estimate: it tests
  those edge cells first (`Hydro::StashFOFCEdgeFlags`),
  and after the first pass the cells take those flags back
  (`MeshBoundaryValuesCC::StashPendingEdgeFOFCFlags` / `ReconcilePendingEdgeFOFCFlags`,
  `src/bvals/flux_correct_cc.cpp`). Where the test with the estimate flagged something the
  test without it did not, the estimate is taken off that cell's faces for the stage, so its
  update is the one it was tested on; the window-end reflux books the flux it used. Where
  the two tests agree nothing changes. A cold shear wave on three SMR levels
  (`tst/inputs/hydro_lat_fofc_edges.athinput`) holds the mass drift in the slowest bin's
  first window to $10^{-16}$.
- **Self-gravity.** Frozen potential inside a window; `gravity/solve_dt > 0` is mandatory; the
  window is shortened to at most `solve_dt`, and the solve is pulled *forward* rather than the
  window truncated, which avoids a pathological 128/16/4/1/1-tick descent.
- **Sink particles.** The operator runs once per window from `Driver::Execute`, not from the
  per-bin task list (it does MPI collectives); every block within a pin sphere grown by its own
  ghost band is forced to factor 1; a warning fires if `SinkStep` is ever reached at a fine-tick
  boundary. The window-end reflux refunds the sink potential's
  work on the replaced face mass flux (`lat_apply_sink_reflux_x{1,2,3}` in
  `src/hydro/hydro_update.cpp`), as it does for self-gravity and the analytic BH; the
  potential is `sinkparticles::SinkPotentialSum`, frozen over the window.
- **Units / integrator.** Only `rk1` and rk2-equivalent tableaux (`rk2`, `imex2`) are admitted;
  the delayed-reflux weights are derived for those (`driver.cpp`, weights in
  `src/driver/lat_weights.hpp`). `imex3` is unanalysed under LAT.
- **`<remap>`.** Mutually exclusive with LAT, refused in two places
  (`src/remap/remap.cpp`, `src/main.cpp`). The supported flow is to remap with
  LAT off and restart the remapped run with LAT on.

**Module support matrix** (guard file; "refused" = the run exits with a message).

| module | status | guard |
| --- | --- | --- |
| hydro (`<hydro>`) | supported | `src/driver/driver.cpp` |
| hydro viscosity / conduction | refused | `driver.cpp` |
| orbital advection / shearing box | refused | `driver.cpp` |
| `srcterms` — `const_accel`, `ism_cooling`, `rel_cooling`, `rad_beam` | refused | `driver.cpp` |
| `srcterms` — self-gravity, external BH gravity | supported, LAT-aware (active list + per-block `dt`) | `src/srcterms/srcterms.cpp`; `driver.cpp`; self-gravity needs `gravity/solve_dt` |
| turbulence driving (`src/srcterms/turb_driver.cpp`) | refused | `driver.cpp` |
| legacy `<radiation>` | refused | `driver.cpp` (`pmbp->prad != nullptr`) |
| multigrid / gravity (`src/multigrid`, `src/gravity`) | supported for hydro, only through the frozen-potential window; the solve is hoisted out of the bins, and the window-end reflux corrects the gravitational work | `driver.cpp`; `mg_gravity.cpp`; work reflux `src/hydro/hydro_update.cpp` |
| `<remap>` | refused, both directions | `src/remap/remap.cpp`, `src/main.cpp` |
| z4c (evolved metric) | refused | `driver.cpp` (`pmbp->pz4c != nullptr`) for dynamical coordinates |
| ion-neutral two-fluid | refused | `driver.cpp` |
| particles (`src/particles`) | refused; no LAT code in the module | `driver.cpp` (`pmbp->ppart != nullptr`) |
| sink particles | supported, window-boundary cadence + factor-1 pinning | `driver.cpp`, `mesh.cpp` |
| user `pgen` hooks (sources / `dt` / BCs) | refused unless flagged `*_lat_safe`; user BCs are refused outright with `<hydro>` | `driver.cpp` |

## Limitations and known issues

- **The default bin floor makes the trajectory rank-dependent.** `hydro_lat_min_bin_count`
  defaults to `4*nranks`; the comment in `src/mesh/mesh.cpp` argues no rank-independent
  constant can be right at both ends of the range, so the dependence is deliberate. Rank 0
  warns. Set the key explicitly for reproducibility across rank counts.
- **A window may not be truncated.** Leaving a window mid-way abandons every pending reflux
  accumulator; `src/driver/driver.cpp` turns that into a FATAL rather than a silent
  conservation loss. Three separate guarantees (the `nlim` clamp, the `tlim` pre-shrink of
  `lat_sync_factor`, and the once-per-window wall-clock refresh) make it unreachable; the
  window-length test has a second condition that rejects a window whose *first* tick already
  lands on `tlim`.
- **No endpoint snap.** `pmesh->time` at a window end is `sync_factor` successive additions of
  `lat_fine_dt`, while a factor-$F$ block's `lat_time_end` is a partial sum plus
  $F\cdot\Delta t_{\rm fine}$; the two disagree by $O(\texttt{sync}\cdot\epsilon\cdot|t|)$
  ($\sim 2\times10^{-7}$ of one fine tick). Endpoint *selection* is
  integer, so a reflux endpoint can never be mis-selected. Revisit only if
  $\Delta t_{\rm fine}/|t| < 10^{-9}$.
- **`lat_theta > 1` is reachable** on the factor-by-factor path; the clamp is load-bearing, not
  diagnostic.
- **Same-level mixed bins are a physics-validation risk at large ratios** — the interpolation
  is dense RK2 between start and end conserved states
  (`docs/lat_implementation_note.tex` §Limitations).
- **`time/hydro_lat_max_nmb_per_rank` is dead and FATAL** (see the key table above). A separate lower LAT cap could not be
  honoured once AMR filled the blocks between the two caps.
- **Cost.** The dominant remaining cost is communication and task-list launch overhead for the
  per-tick refresh at large sync factors (`docs/lat_implementation_note.tex` §Performance
  Notes); the design deliberately favours correctness and synchronized AMR/gravity behaviour
  over asynchronous per-rank progress.

## Tests

- **Decks**: `inputs/TDE_examples/tde_05_fallback_lat.athinput`.

## References

- `docs/lat_implementation_note.tex` / `.pdf` — the design note: scope, user controls, factor
  metadata, active/communication masks, boundary exchange and time interpolation, delayed flux
  correction, self-gravity, GID ordering, the load-balance objective, AMR and state migration,
  restart semantics, performance notes, invariants. (Scope and "current production
  configuration" sections are out of date; the algorithmic sections are not.)
- Berger & Colella (1989), *J. Comput. Phys.* **82**, 64 — the refluxing this generalises to
  bin boundaries.
- Gottlieb (2009) — the SSPRK(2,2) tableau whose dense output is used for mixed-cadence ghosts
  (used in `src/driver/driver.cpp`).
