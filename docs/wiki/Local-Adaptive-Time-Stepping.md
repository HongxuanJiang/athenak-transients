# Local Adaptive Time Stepping (LAT)

> **LAT for MHD and GRMHD.** In this release, LAT is available for hydrodynamics, including self-gravity. The development version of the code also supports LAT for MHD and GRMHD, including GRMHD on dynamical spacetimes. These paths are not included in this public release, and a run with `time/lat = true` and an `<mhd>` block stops with an error. They are available on request from Hong-Xuan Jiang (masterjoe2000@outlook.com).

## Summary

Normally every MeshBlock (a block, for short) advances with the smallest time step found anywhere in the domain. With LAT, each block advances with its own step. A block's step is the largest step it can take on its own, rounded down to a power-of-two multiple of the finest step. That multiple is the block's *factor* (1, 2, 4, ...), and all blocks with the same factor form a *bin*. A slow region, such as a large low-density envelope, then needs far fewer updates than the small dense region that sets the global step.

Two mechanisms keep the solution consistent where neighbouring blocks advance at different rates. A block that updates takes its ghost-zone values from a neighbour that is partway through a longer step by interpolating that neighbour's state in time. The flux across such an interface is recorded on both sides, and the mismatch is added back later (refluxing), which keeps the scheme conservative.

**Use it** when a few blocks need a time step much smaller than most of the domain, for example a star or disk inside a large, cheap envelope, and the run is hydrodynamics (with or without [self-gravity](Multigrid-Self-Gravity), sink particles and an external black hole). With the default `lat_same_level = false`, all blocks of one refinement level share one factor, so a gain needs SMR or AMR.

**Do not use it** when almost all blocks need the same step (there is nothing to save, and the boundary refresh in every tick costs extra), when the run needs a module that LAT refuses (see [Practical guidance](#practical-guidance)), or when you need the same trajectory on different numbers of MPI ranks and do not set `hydro_lat_min_bin_count` explicitly.

## Quick start

Add two keys to `<time>`:

```ini
<time>
lat        = true
lat_levels = 3        # blocks may take steps of 1, 2, 4 or 8 times the finest step
```

With self-gravity, `<gravity>/solve_dt` is also required. The keys you will touch most often:

| key | what it does |
| --- | --- |
| `time/lat` | Master switch. |
| `time/lat_levels` | Number of doublings. The largest allowed step is `2^lat_levels` times the finest one. Default 1, range 1 to 20. |
| `gravity/solve_dt` | Required with self-gravity. The potential is frozen during a window, so this key caps the window length. |
| `time/hydro_lat_min_bin_count` | Smallest number of blocks a bin may hold. Set it explicitly to get the same result on different rank counts. |
| `time/lat_diagnostics` | Print the bin tables, to check what LAT decided. |

## Full parameter table

All `time/` keys are read only when LAT is on. Legacy spellings are listed below the table.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `time/lat` | bool | `false` | Master switch. If absent, the legacy `time/hydro_lat` is used. |
| `time/lat_levels` | int | 1 | Number of doublings. The largest factor is `2^lat_levels`. Values outside 1 to 20 are clamped. |
| `time/lat_same_level` | bool | `false` | Allow different factors on blocks of the same refinement level. When `false`, all blocks of a level take the smallest factor found on that level. |
| `time/lat_same_level_max_ratio` | int | 1 | Largest factor ratio between same-level neighbours; it acts only with `lat_same_level = true`. Must be 1, 2, 4 or 8, otherwise fatal. |
| `time/lat_neighbor_limiter` | string | `all` | Which same-level neighbours the factor limiter looks at: `all` (faces, edges and corners), `face` (faces only) or `hybrid` (faces held to 2:1, edges and corners to the configured ratio). It acts only with `lat_same_level = true`. With self-gravity or an external black hole, `face` together with `lat_same_level = true` is fatal. Any other value is fatal. |
| `time/hydro_lat_min_bin_count` | int | -1 | Smallest number of blocks a bin above factor 1 may hold. A smaller bin is merged into the next bin down (half the factor). `-1` means `4*nranks`, `0` switches the merge off, and values below -1 are fatal. With the default, bin membership depends on the rank count, and rank 0 warns when more than one rank is used. |
| `time/lat_diagnostics` | bool | `false` | Print the bin table at several stages of the factor selection (raw, after the problem cap and the neighbour limiter, and final), and the same-level and refinement-interface status lines. In release builds it also switches on an internal consistency check of the fused predictor step. |
| `time/lat_union_stage1` | string | `auto` | `auto`, `true` or `false` (`1` and `0` also work). Whether all due bins share one fused predictor kernel in each tick. `auto` and `true` both use the value derived from the run's capabilities, and neither overrides the integrator and user-hook conditions. Any other value is fatal. |
| `time/hydro_lat_gid_reorder` | bool | `true` | Order blocks so that each rank owns a slice of every bin (see [Parallel layout](#parallel-layout)). |
| `time/hydro_lat_post_amr_rebalance` | bool | value of `hydro_lat_gid_reorder` | Allow the LAT rebalance of the rank layout at a window end, once the mesh has stayed unchanged for four window lengths. `true` together with `hydro_lat_gid_reorder = false` is fatal. |
| `time/hydro_lat_defer_final_exchange` | bool | unset | Skip the ghost-zone exchange at the end of a bin when the next tick's refresh makes it redundant. If unset, the code decides from the run's capabilities and never skips it with the Newtonian [Dual Energy](Dual-Energy) formalism. `false` forces it off. `true` is only a request. |
| `time/hydro_lat_max_nmb_per_rank` | int | removed | Removed: a fresh start with this key is fatal. Use `mesh_refinement/max_nmb_per_rank`. |
| `time/hydro_subcycle`, `time/hydro_subcycle_factor` | bool, int | `false`, 1 | Global (non-LAT) hydro subcycling. Under LAT both are overwritten internally (`hydro_subcycle = true`, factor `2^lat_levels`). |
| `mesh_refinement/max_nmb_per_rank` | int | unset | Per-rank block cap, which is also the cap of the LAT partition. Required with AMR and read only on adaptive meshes. On a static mesh the cap is `ceil(nblocks/nranks)`, so ranks hold nearly equal block counts. |
| `mesh_refinement/sticky_load_balance` | bool | `false` | Keep each block on its previous rank across a regrid where possible. Used only when LAT is on (otherwise a message says it is ignored), and only while no block has a factor above 1. |
| `gravity/solve_dt` | Real | none | Required, and greater than 0, with LAT and self-gravity; otherwise fatal. The potential is frozen inside a window, so this bounds the window length. |
| `sink_particles/lat_window_motion_cells` | Real | 1.0 | How far a sink may move in one window, in finest cells (times `cfl_number`). Larger values allow longer windows. Must lie in [1, min(`a`, (`a` - 1/2)/`cfl_number`)] with `a = accrete_radius_cells`, otherwise fatal. Used when `sink_particles/use_sink_timestep` is `true` (the default). |

**Legacy spellings.** Both spellings of a key are accepted. `hydro_lat` maps to `lat`, `hydro_lat_levels` to `lat_levels`, and `hydro_lat_same_level`, `hydro_lat_same_level_max_ratio`, `hydro_lat_neighbor_limiter`, `hydro_lat_diagnostics` and `hydro_lat_union_stage1` to the same names without `hydro_`. In the other direction, `lat_min_bin_count`, `lat_gid_reorder` and `lat_post_amr_rebalance` map to the `hydro_lat_` names.

## How it works

### Factors and bins

Each block works out the largest step it can take on its own. The ratio of that step to the global step, rounded down to a power of two and capped at `2^lat_levels`, is the block's factor. Rules then lower factors where a large one would be unsafe, in this order:

1. A cap from the problem generator, if it installs one. The TDE generator forces factor 1 around the excised black hole.
2. Sink pinning: blocks around a sink are forced to factor 1.
3. The refinement-level collapse, unless `lat_same_level = true`.
4. The minimum bin population, `hydro_lat_min_bin_count`. A bin above factor 1 with too few blocks is merged into the next bin down.
5. A neighbour limiter that keeps the factors of neighbouring blocks close. Across a refinement interface a finer block never has a larger factor than the coarser one. Between same-level neighbours the ratio is at most `lat_same_level_max_ratio`.

The factors are recomputed at startup, at every window end and after every regrid.

### Windows and ticks

The largest factor that survives sets the length of a *window*, which is that many finest steps. Inside a window the driver runs one *tick* per finest step. On tick k, the blocks whose factor divides k take a step of `factor` finest steps.

```
tick             0     1     2     3     4 (window end)
factor 4 block   |-----------------------|
factor 2 block   |-----------|-----------|
factor 1 block   |-----|-----|-----|-----|
```

The window is shortened when needed: it must fit before `tlim` and `nlim`; with self-gravity it never exceeds `gravity/solve_dt`; and with sinks it is short enough that a sink moves at most about `cfl_number * lat_window_motion_cells` finest cells.

AMR regridding, outputs (including restart files), the sink operator and the self-gravity solve happen only when every block is at the same time, which means at a window boundary.

### Ghost zones and conservation

When a block updates, its neighbour may be partway through a longer step. The neighbour then supplies ghost values interpolated in time between its own start and end states. The interpolation is linear, or a second-order polynomial built from the Runge-Kutta stages when that is available. A fast block therefore always sees ghost data at the right time.

Each block accumulates the time integral of the flux through its own boundary and subtracts what finer or faster neighbours contributed. When a slow block finishes its step, the difference is added back as a surface correction. This is Berger-Colella refluxing, generalised to time-step bins, including same-level factor mismatches.

### Parallel layout

LAT changes the load-balance goal from equal cost per rank to work on every rank in every tick. Blocks are ordered so that each rank owns a slice of every bin. On SMR and AMR meshes this ordering is applied at every regrid. With more than one rank, a rebalance at a window end applies it as well, a few windows after startup or after a regrid, and only if the new layout is predicted to improve the window time or the worst tick by at least 5%. A uniform mesh keeps the ordinary Z-order layout. The default minimum bin population scales with the rank count, so the rank count also affects bin membership (see [Reproducibility across rank counts](#reproducibility-across-rank-counts)).

## Practical guidance

### What is supported and refused

A refused module makes the run stop with an error message.

| module | status |
| --- | --- |
| hydro (`<hydro>`) | supported |
| self-gravity ([Multigrid Self-Gravity](Multigrid-Self-Gravity)) | supported through a potential that is frozen inside each window. The solve runs outside the bins, and the reflux corrects the gravitational work. Needs `gravity/solve_dt`. |
| [Dual Energy](Dual-Energy) | supported |
| external black-hole gravity | supported |
| sink particles | supported. The sink operator runs once per window, with factor-1 pinning near each sink. |
| AMR / SMR | supported. Regrids happen at window ends. |
| hydro viscosity and conduction | refused |
| orbital advection, shearing box | refused |
| `<hydro>` source terms `const_accel`, `ism_cooling`, `rel_cooling`, `disk_cooling`, `rad_beam` | refused |
| turbulence driving | refused |
| legacy `<radiation>` | refused |
| z4c (evolved metric) and dynamical relativistic spacetimes | refused |
| ion-neutral two-fluid | refused |
| particles | refused |
| `<remap>` ([Remap usage](Remap-Usage#remap-with-lat)) | supported. The remap at the start of the run happens before the first window. The settle steps (`<remap>/settle_steps`) run without LAT, and LAT starts after the last remap pass. |
| user problem-generator hooks | sources and time-step limits are refused unless the generator flags them `*_lat_safe`. User boundary conditions are always refused with `<hydro>`. |

Only the `rk1` integrator and the rk2-equivalent tableaux (`rk2`, `imex2`) are allowed. All other integrators, including `rk3`, `rk4`, `imex2+` and `imex3`, are refused.

### Reproducibility across rank counts

With the default `hydro_lat_min_bin_count = -1` the minimum bin population is `4*nranks`, so bin membership depends on the number of ranks. A restart of the same state on a different rank count is still conservative and valid, but it follows a different trajectory. Set the key explicitly to reproduce a run on a different number of ranks. Bin assignment is not stored in checkpoints. It is recomputed from the local time-step limits at the first step after a restart.

### Mesh refinement

Regrids happen only at window ends, and only when at least `mesh_refinement/ncycle_check` cycles have passed since the previous check. A window is up to `2^lat_levels` cycles long, so a smaller `ncycle_check` simply means a check at every window end.

### Self-gravity

Set `gravity/solve_dt`. The window never exceeds this value. When a solve is due, or the window would run past the next solve time, the potential is solved at the start of the window. Shortening the window to land on the due time would instead produce a long descent of ever shorter windows. See [Multigrid Self-Gravity](Multigrid-Self-Gravity) for the solver itself.

### Sinks

Blocks near a sink are forced to factor 1, and the sink operator runs once per window. A fast sink limits the window length, because the window must be short enough that the sink moves only a little. If that keeps the windows very short, raise `lat_window_motion_cells` within the range given in the parameter table.

### Restarts and outputs

- LAT needs a single global restart file. Restarting from per-rank restart files (`single_file_per_rank = true` in the `rst` output) with `lat = true` is fatal.
- Outputs and restart files are written only at window ends. An output that falls due in the middle of a window is written at the end of that window.
- A restart does not reuse the old bins, as explained in [Reproducibility across rank counts](#reproducibility-across-rank-counts).

### Checking that it works

- Set `lat_diagnostics = true` to print the bin table at each stage of the factor selection.
- Each time the LAT partition is built on more than one rank with a largest factor above 1, rank 0 prints one line, `Mesh: HD LAT per-bin rank distribution ...`. It lists how many blocks of each bin land on each rank, and the predicted busy fraction of each rank (`util%`). Compare it with the measured GPU utilisation.
- `Driver: LAT union stage-1 predictor enabled/disabled (...)` names the reason when the fused predictor is off.

### Common problems

| symptom | cause and fix |
| --- | --- |
| Little or no speedup | Most blocks are in the finest bin, or the refinement-level collapse (`lat_same_level = false`) forces whole levels to factor 1. Check `lat_diagnostics`. |
| Fatal error about `solve_dt` | Add a positive `gravity/solve_dt`. |
| Fatal error `time/lat is not enabled for ...` | The run uses a refused module. See the table above. |
| Fatal error that `lat_neighbor_limiter` must be `all` or `hybrid` | `lat_same_level = true` with self-gravity or an external black hole does not allow `face`. |
| Results differ when the rank count changes | Set `hydro_lat_min_bin_count` explicitly. |

### Performance

The remaining cost is mainly communication and task-list launch overhead for the boundary refresh in every tick, which grows with the window length. The design favours correctness and synchronized AMR and gravity over asynchronous per-rank progress. On a static mesh the rank boundaries are fixed to nearly equal block counts, so the order of the blocks is the only thing the LAT layout can change.

### Limitations

- The default bin floor makes the trajectory depend on the rank count (see Reproducibility across rank counts).
- Same-level mixed bins (`lat_same_level = true`) are a physics-validation risk at large factor ratios, because the interpolation is a dense Runge-Kutta polynomial between the start and end conserved states.
- A window cannot be cut short. The driver shortens windows beforehand so that no limit (`tlim`, `nlim`, wall clock) falls inside one, and turns any attempt to leave a window early into a fatal error rather than a silent loss of conservation.

## Further reading

- Example deck: `inputs/TDE_examples/tde_05_fallback_lat.athinput`. It runs the remap, the settle steps and the fallback in one run: the settle steps run without LAT and LAT starts after the last remap pass.
- Berger and Colella (1989), *J. Comput. Phys.* **82**, 64: the refluxing generalised here to bin boundaries.
- Gottlieb (2009): the SSPRK(2,2) tableau whose dense output is used for mixed-cadence ghosts.
