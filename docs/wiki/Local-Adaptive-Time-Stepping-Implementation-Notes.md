# Local Adaptive Time Stepping: implementation notes

How [Local Adaptive Time Stepping (LAT)](Local-Adaptive-Time-Stepping) works inside. Start on that page for parameters and practical advice. This page explains the ideas behind them in words, for hydrodynamics, which is what this release supports.

## Overview

LAT cuts the run into *windows*. Inside a window each block takes steps of its own length, and at the end every block is at the same time again. Everything that needs one common time (regridding, outputs, the sink operator, the gravity solve) happens at window boundaries.

```
 global step --> step limit of each block --> factors (bins) --> window of F ticks
                                                                       |
 tick 0 ... F-1:  refresh ghosts of due blocks -> update them -> book flux mismatch
                                                                       |
 window end:  sink operator, outputs, regrid check, new step and factors, rebalance
```

## How the factors are chosen

The factors are recomputed at every synchronized point: at startup, at each window end and after a regrid. The work is done in `Mesh::UpdateHydroLATMetadata`.

1. Each block reports the largest step it can take alone: the CFL number times the smaller of its fluid limit and its source-term limit. All ranks share the values.
2. The raw factor is the largest power of two that does not exceed $\Delta t_m/\Delta t$, where $\Delta t_m$ is the block's step and $\Delta t$ the global step. It is capped at `2^lat_levels`.
3. The rules below then lower factors where a large one would be unsafe. They run in this order.

- **Problem cap.** A problem generator may install a cap. The TDE generator forces factor 1 near the excised black hole.
- **Density pin.** Optional (`lat_pin_density_contrast`, `lat_pin_density`). Blocks with a sharp density jump or a very dense cell go to factor 1 with their neighbours, because a delayed reflux can briefly put dense gas into an ambient cell, and sources and floors acting before the mass is refunded then leave an irreversible thermal residue.
- **Sink pinning.** A block whose bounding box, grown by its own ghost-zone width, touches a sink's pin sphere goes to factor 1. The sink operator rewrites the gas there, and that gas must be at the common time.
- **Level collapse.** Unless `lat_same_level = true`, all blocks of a refinement level take the smallest factor on that level, because the standard flux correction only handles interfaces between levels.
- **Minimum population.** A bin above factor 1 with fewer than `hydro_lat_min_bin_count` blocks is merged into the next bin down, repeatedly. A sparse bin costs more in task-list launches and boundary exchange than it saves. "Sparse" is a per-rank notion, so the default `4*nranks` scales with the rank count.
- **Neighbour limiter.** Neighbouring factors must stay close. Across a refinement interface the finer block never has a larger factor than the coarser one, and the coarser factor is at most twice the finer one per level of difference. Between same-level neighbours (with `lat_same_level`) the ratio is at most `lat_same_level_max_ratio`. Edge and corner neighbours count too, because the prolongation stencil reads their data, and `lat_neighbor_limiter` chooses how many are used. The limiter repeats until nothing changes. It stops with an error if it needs more passes than there are blocks.

The largest factor left sets the window length. The factors are computed right after the global step, from the same per-block limits.

## The window and the tick loop

The window length F, in finest steps, starts at the largest factor present. It is halved while any of these holds: it would pass `tlim` or `nlim`; with self-gravity it exceeds `gravity/solve_dt`; with sinks a sink could move farther than `lat_window_motion_cells` allows.

For each window:

1. Fix the finest step and the window length F.
2. With self-gravity, solve for the potential first if it is invalid, due, or would be passed during the window. The potential then stays frozen for the whole window.
3. Save every block's start state and reset the block clocks.
4. Run the ticks. A tick at which no block is due is skipped.
5. At the window end, with all blocks at one time, run the sink operator, make outputs and restart files, check for a regrid, compute the new global step and factors, and rebalance the ranks if due.

For each tick:

1. Find the due blocks. A block with factor f is due when the tick number is a multiple of f, and it computes its whole step at the start of its interval.
2. Refresh the ghost zones of the due blocks to the current time, including restriction and prolongation across refinement levels. Blocks that are not due but must send data to a due neighbour still take part in the exchange.
3. Update the due blocks with the two-stage (RK2) scheme. By default the first stage of all due bins runs as one kernel, each block using its own step. The second stage then runs bin by bin from the largest factor to the smallest, so a faster bin sees the finished state of the slower ones. With `lat_union_stage1 = false` each bin runs both stages in turn, again from slowest to fastest.
4. Apply the flux corrections that fall due, and advance the global time by the finest steps completed.

## Ghost zones and conservation

**Ghost data from a block in mid-step.** A block that is not due does not hand over its stored array. It hands over a value interpolated to the receiver's time between its own start and end states. The interpolation is linear, or the dense-output polynomial of the RK2 step when the stage-1 state is available.

**Conservation.** Every active block accumulates the time integral of the flux through its slow-side faces and subtracts the integral received from finer or faster neighbours. When the block's step completes, the net difference is applied as a surface correction. This is Berger-Colella refluxing generalised to LAT bins, including same-level factor mismatches. The same correction refunds the gravitational work on the mass that moved, for self-gravity, the external black hole and sinks.

**Dual energy.** The auxiliary internal-energy field gets a matching delayed correction, and it is re-synchronised from the corrected total energy where the dual-energy rules allow (see [Dual Energy](Dual-Energy)).

**First-order flux correction (FOFC).** It runs on the active blocks like any other kernel. At an interface where a slower bin's pending flux estimate meets a same-level neighbour of the same cadence, the two blocks could test one edge cell on different fluxes and flag it differently. The code reconciles the flags at those cells, so each cell is updated with the fluxes it was tested on, and the reflux books the flux that was used.

## Rank layout and rebalance

Ticks run bin by bin, and a rank that holds no block of a due bin sits idle. The partition therefore aims at work on every rank in every tick, not only at equal total cost.

1. Each bin is sorted in Z-order. The bins are then merged into one list in which every bin is spread evenly along the whole list.
2. The list is cut into one contiguous range of blocks per rank. The search prefers the cut with the shortest estimated window (the sum over ticks of the busiest rank's work). Ties go to the smaller worst tick, then to fewer idle rank-tick pairs, and further criteria follow down to an even block count.
3. On adaptive meshes a rank holds at most `mesh_refinement/max_nmb_per_rank` blocks. On a static mesh the cap is the block count divided by the rank count, rounded up.

A block's cost for a window is its number of steps in the window times its work per step. In this release the work per step is taken as 1 for every block.

At startup the layout is the ordinary Z-order one, so a LAT run and a non-LAT run begin from identical data. The LAT layout comes later. It is applied at every regrid (when at least two bins exist), and by a rebalance at a window end that needs all of these: `hydro_lat_post_amr_rebalance` on, a refined mesh, more than one rank, a largest factor above 1, a mesh unchanged for four windows, and a startup or regrid since the last attempt. The rebalance moves blocks between ranks without changing the mesh. It is carried out only if it is predicted to improve the window time or the worst tick by at least 5%, or if the current layout breaks the block cap.

After a regrid the factors of the new blocks are only estimated for the partition. A surviving block keeps its factor. A new child takes the factor of its nearest old ancestor, halved for each level of refinement. A derefined parent takes the smallest factor of its former descendants, doubled for each level of coarsening. On a cold start with no information the estimate is by level: the finest level gets factor 1 and each coarser level doubles. The real factors are recomputed at the next window end.

No MPI collective may run inside a bin. A rank with no due block would never reach it, and the other ranks would wait forever. Operations with collectives, such as the sink operator, therefore run only at window boundaries.

## Restart

1. LAT needs one global restart file. With per-rank files the data of a block cannot be found once blocks have been reordered, so `single_file_per_rank` with LAT is fatal.
2. Restart files are written at window ends, when every block is at one time.
3. The file stores blocks in the order of the run that wrote it. On adaptive meshes the restart rebuilds the ordinary Z-order. If that differs from the order in the file, it keeps a map from the new block number to the number in the file, and reads each block through it with independent reads, so ranks need not read in the same order. The map is dropped afterwards. On non-adaptive meshes the checkpoint's block order is kept, with its saved rank boundaries when the rank count is unchanged.
4. Bins are not stored. They are recomputed at the first step after the restart. With the default `hydro_lat_min_bin_count`, a restart on a different rank count is valid and conservative but follows a different trajectory.

## How LAT works with other modules

| module | what LAT does |
| --- | --- |
| self-gravity | Freezes the potential in each window. `gravity/solve_dt > 0` is mandatory. A solve that falls due is pulled forward to the window start instead of cutting the window, which avoids a long descent of ever shorter windows. See [Multigrid Self-Gravity](Multigrid-Self-Gravity). |
| sink particles | The operator runs once per window, not per bin, because it uses MPI collectives. Blocks near a sink are at factor 1. The reflux refunds the sink potential's work. |
| external black hole | The source acts only on active blocks. For a translating frame or a live black hole, the TDE generator advances the frame once per window, at the common start time. |
| integrators | Only `rk1` and the rk2-equivalent tableaux (`rk2`, `imex2`) are allowed, because the delayed-reflux weights are derived for them. |
| `<remap>` | Refused, in two places. See [Remapping](Remapping). |
| other modules | Refused ones are listed on the [main page](Local-Adaptive-Time-Stepping#what-is-supported-and-refused). |

## Known limitations

- The default bin floor makes the trajectory depend on the rank count. This is deliberate: no rank-independent constant is right at both ends of the range. Rank 0 warns.
- A window may not be cut short, because that would abandon every pending flux correction. The driver turns an attempt into a fatal error. Three guarantees keep it from happening: the `nlim` clamp, the shrinking of the window before `tlim`, and the wall clock being read once per window.
- Same-level mixed bins are a physics-validation risk at large ratios, because the interpolation is dense RK2 between start and end conserved states.
- With `lat_same_level = true` and self-gravity or an external black hole, `lat_neighbor_limiter = face` is refused.
- The time at a window end and a block's own end time can differ by round-off. Endpoints are chosen by integer tick counts, so this cannot select the wrong reflux endpoint.
- The remaining cost is communication and task-list launch overhead for the per-tick refresh at large factors.

## Tests

Run from the `tst` directory, for example `python run_test_suite.py --cpu --test test_suite/nr/test_nr_hydro_lat_fofc_edges_cpu.py`. Use `--mpicpu` for the restart test.

| test file in `tst/test_suite/nr/` | what it checks | deck |
| --- | --- | --- |
| `test_nr_hydro_lat_fofc_edges_cpu.py` | FOFC at the edges of SMR levels keeps the total mass to round-off | `tst/inputs/hydro_lat_fofc_edges.athinput` |
| `test_nr_hydro_lat_grav_reflux_gate_cpu.py` | The gravitational-work refund closes the energy budget and matches a non-LAT run | `tst/inputs/hydro_lat_grav_reflux_gate.athinput` |
| `test_nr_amr_restart_bitwise_lat_mpicpu.py` | A LAT run on an adaptive mesh, restarted from a checkpoint, equals the uninterrupted run | `tst/inputs/linear_wave_amr_restart_lat.athinput` |

The example deck is `inputs/TDE_examples/tde_05_fallback_lat.athinput`. The design note is `docs/lat_implementation_note.tex`.

## Key files

| file | role |
| --- | --- |
| `src/mesh/mesh.cpp` | per-block limits, the factor rules |
| `src/driver/driver.cpp` | refusal checks, window and tick loop, rebalance trigger |
| `src/mesh/meshblock_pack.cpp` | due and active block lists, block clocks |
| `src/mesh/load_balance.cpp` | cost model, interleaved block order, partition search |
| `src/mesh/mesh_refinement.cpp` | regrid and rebalance with the LAT order |
| `src/mesh/build_tree.cpp` | startup and restart layout, restart block map |
| `src/bvals/bvals_cc.cpp` | time-interpolated ghost data |
| `src/bvals/flux_correct_cc.cpp` | delayed flux-correction buffers |
| `src/hydro/hydro_update.cpp` | applying the corrections, gravity and sink work refunds |
| `src/parameter_input.cpp` | the LAT switch and the legacy key names |
