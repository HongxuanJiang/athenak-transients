# Local Adaptive Time Stepping: implementation notes

Developer overview of how [LAT](Local-Adaptive-Time-Stepping.md) works inside the code. Usage, parameters and guidance are on the main page. LAT in this release covers hydrodynamics only.

## Contents

- [How the pieces fit together](#how-the-pieces-fit-together)
- [How the factors are chosen](#how-the-factors-are-chosen)
- [The window and tick loop](#the-window-and-tick-loop)
- [Ghost zones and conservation](#ghost-zones-and-conservation)
- [Parallel layout and AMR](#parallel-layout-and-amr)
- [Restart](#restart)
- [Limitations and known issues](#limitations-and-known-issues)
- [Tests](#tests)
- [Key files](#key-files)

## How the pieces fit together

```
NewTimeStep (global dt)
   -> per-block factors f_m (power of two)          Mesh::UpdateHydroLATMetadata
   -> window of F fine steps                        Driver::Execute
        tick 0 .. F-1:
          refresh ghost zones to the tick time      RefreshHydroLATBoundaries
          update the blocks that are due            active-block mask
   -> window end: reflux, AMR, outputs, gravity, sinks, new factors, rebalance
```

Everything that needs a globally consistent state (regridding, outputs, restarts, the gravity solve, the sink operator and the per-block time-step estimate) happens only at window ends. Inside a window only the hydro update runs, on the blocks that are due.

## How the factors are chosen

Every block reports the largest step it can take on its own. The factor $f$ is the largest power of two that fits in the ratio to the global minimum step, capped at `2^lat_levels`. The raw factors are then reduced in this order.

1. A problem-specific cap, if the problem generator provides one.
2. Sink pinning. Blocks near a sink are forced to factor 1.
3. AMR-level collapse. Unless `lat_same_level = true`, all blocks on one AMR level take that level's smallest factor.
4. Minimum bin population. A bin with fewer than `hydro_lat_min_bin_count` blocks is merged into the next faster bin.
5. Neighbour limiter. Neighbouring factors are kept close, with a 2:1 staircase across coarse and fine interfaces and at most `lat_same_level_max_ratio` between same-level neighbours. The pass repeats until nothing changes. Failure to converge is fatal.

The bins are decided right after the global step is computed, not inside it. Each module's own time-step estimate is refreshed first, so that no block keeps a stale value.

## The window and tick loop

The largest surviving factor sets the window length in fine steps. The driver then shortens it so that it fits before `tlim`, stays below `gravity/solve_dt` with self-gravity, and lets a sink move at most `lat_window_motion_cells` cells.

For each window the driver does the following.

1. Save the start state of every block.
2. For each fine tick, build the due mask. A block is due when its factor divides the tick number.
3. Refresh the ghost zones of the due blocks to the common tick time.
4. Integrate the due blocks, slowest bin first, so that faster bins see a completed slower history. With `lat_union_stage1` the first RK stage of all due bins runs as one fused kernel.
5. At the last tick, run the window-end work listed above.

## Ghost zones and conservation

A block that is not due still supplies ghost data to a due neighbour. The value is interpolated in time between the block's start and end states, linearly or with the second-order Runge-Kutta dense-output polynomial when a valid stage-1 state exists. If the graph never saves that state, the code silently falls back to the linear form.

Each block accumulates the time integral of its own boundary flux and subtracts what finer or faster neighbours contributed. When a slow block finishes its step, the difference is applied as a surface correction in `hydro_update.cpp`. This is Berger-Colella refluxing generalised to bins. Self-gravity and sink work are refunded in the same place.

First-order flux correction (FOFC) runs on the active list like any other kernel. At edges where a pending coarse-fine face meets a same-level face, extra bookkeeping keeps both blocks on the same flux. The test `hydro_lat_fofc_edges` covers it.

## Parallel layout and AMR

Load balance targets work on every tick rather than equal cost per rank. Each block gets a window cost of roughly (window length divided by $f$) times its measured work. Blocks are then ordered so that each rank holds a slice of every bin, and the cut search minimises the summed per-tick maximum work. On a static mesh the rank cuts are pinned to equal block counts, so only the order inside each bin matters.

After a regrid, new blocks get estimated factors from their old neighbours or ancestors until the next window end recomputes them. A rebalance is attempted only after the topology has been stable for a few windows. No MPI collective may run inside a rank-local bin, since a rank without due blocks would deadlock.

## Restart

LAT needs one global restart file. Blocks may be reordered, so the reader keeps a map from the current block ID to the file block ID. Checkpoints are written only at window ends. Bin assignment is not stored and is recomputed at the first step after a restart.

## Limitations and known issues

- The default minimum bin population is `4*nranks`, so bin membership and the trajectory depend on the rank count. This is deliberate and rank 0 warns.
- A window may not be cut short, since that would drop pending reflux accumulators. The driver turns any attempt into a fatal error.
- Same-level mixed bins are a physics-validation risk at large factor ratios.
- Only `rk1`, `rk2` and `imex2` are admitted. `imex3` is not analysed.
- The remaining cost is communication and task-list launch overhead for the refresh on every tick.
- Modules that LAT refuses are listed on the main page, with their guards in `driver.cpp` and the module sources.

## Tests

- Example deck: `inputs/TDE_examples/tde_05_fallback_lat.athinput`. Set `lat = true` and run it with and without LAT, then compare conserved totals and the solution.
- Regression tests in `tst/test_suite/nr/`: `test_nr_hydro_lat_fofc_edges_cpu.py` (deck `hydro_lat_fofc_edges.athinput`, a cold shear wave on three SMR levels), `test_nr_hydro_lat_grav_reflux_gate_cpu.py` (the gravity reflux gate) and `test_nr_amr_restart_bitwise_lat_mpicpu.py` (bitwise AMR restart under LAT).
- Run with `lat_diagnostics = true` to inspect the bin tables, and compare the predicted per-rank utilisation line with measured GPU use.
- Run the same state on two rank counts with `hydro_lat_min_bin_count` set explicitly, which should give the same trajectory.

## Key files

| file | role |
| --- | --- |
| `src/mesh/mesh.cpp` | factor ladder and all narrowing passes |
| `src/mesh/load_balance.cpp` | window cost, bin-interleaved block order, rank cuts |
| `src/mesh/meshblock_pack.cpp` | due masks, active lists, flux-correction flags |
| `src/driver/driver.cpp` | admissibility checks, window and tick loop, rebalance triggers |
| `src/bvals/bvals_cc.cpp` | time-interpolated ghost data |
| `src/bvals/flux_correct_cc.cpp` | delayed flux accumulation across bins |
| `src/hydro/hydro_update.cpp` | window-end reflux and work refunds |
| `src/mesh/build_tree.cpp` | restart reading with reordered blocks |

Design note with full derivations: `docs/lat_implementation_note.tex`. References are Berger and Colella (1989), *J. Comput. Phys.* **82**, 64, and Gottlieb (2009) for the SSPRK(2,2) tableau.
