# Local Adaptive Time Stepping (LAT)

> **LAT for MHD and GRMHD.** In this release, LAT is available for hydrodynamics, including self-gravity. The development version of the code also supports LAT for MHD and GRMHD, including GRMHD on dynamical spacetimes. These paths are not included in this public release. They are available on request from Hong-Xuan Jiang (masterjoe2000@outlook.com).

## Summary

Normally every MeshBlock advances with the smallest time step found anywhere in the
domain. LAT lets each block advance with its own step instead, rounded down to a power-of-two
multiple of the finest step. Slow regions (a large, low-density envelope, say) then take far
fewer updates than the small dense region that sets the global step. Conservation is kept by
interpolating ghost zones in time and by refluxing the flux mismatch between blocks that
advance at different rates.

**Use it** when a few blocks need a time step much smaller than most of the domain, for
example a star or disk inside a large, cheap envelope, and the run is hydrodynamics (with
or without self-gravity, sinks and an external black hole).

**Do not use it** when almost all blocks need the same step (there is nothing to save and the
per-step boundary refresh costs extra), when the run needs a module LAT refuses
(viscosity, shearing box, turbulence driving, particles, Z4c, ion-neutral, `<remap>`; see
[Practical guidance](#practical-guidance)), or when you need a trajectory that is
independent of the number of MPI ranks without setting one extra key.

## Quick start

Add one line to `<time>`:

```ini
<time>
lat = true
lat_levels = 3        # blocks may take steps of 1, 2, 4 or 8 times the finest step
```

For a run with self-gravity, `<gravity>/solve_dt` is also required (see below).

| key | what it does |
| --- | --- |
| `time/lat` | Master switch. |
| `time/lat_levels` | Number of doublings. The largest allowed step is 2^lat_levels times the finest one. Default 1, range 1 to 20. |
| `gravity/solve_dt` | Required with self-gravity. The potential is frozen during a window, so this key caps the window length. |
| `time/hydro_lat_min_bin_count` | Smallest number of blocks a bin may hold. Set it explicitly to get the same result on different rank counts. |
| `time/lat_diagnostics` | Print the bin tables, for checking what LAT decided. |

## Full parameter table

All keys are read from `<time>` unless a block is given. Legacy spellings are listed below
the table.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `time/lat` | bool | off | Master switch. If absent, the legacy `time/hydro_lat` is used. |
| `time/lat_levels` | int | 1 (clamped to 1 to 20) | Number of factor doublings. The largest factor is 2^lat_levels. |
| `time/lat_same_level` | bool | `false` | Allow different factors on blocks of the same AMR level. When false, each level collapses to one bin. |
| `time/lat_same_level_max_ratio` | int | 1 | Largest factor ratio between same-level neighbours. Must be 1, 2, 4 or 8, otherwise fatal. |
| `time/lat_neighbor_limiter` | string | `"all"` | `all`, `hybrid` or `face`. How many neighbour slots the time-step limiter uses. Anything else is fatal. |
| `time/hydro_lat_min_bin_count` | int | -1, meaning 4*nranks | Smallest bin population before the bin is merged into the next faster one. The default makes the result depend on the rank count, and rank 0 warns when it is left at the default with more than one rank. Values below -1 are fatal. |
| `time/lat_diagnostics` | bool | `false` | Print the per-pass bin tables and the same-level and AMR-interface status lines. Also enables the union-step check in release builds. |
| `time/lat_union_stage1` | string | `"auto"` | `auto`, `true` or `false`. Whether all due bins share one fused predictor kernel. `auto` and `true` both take the value derived from the run's capabilities, and neither overrides the integrator and user-hook conditions. Any other value is fatal. |
| `time/hydro_lat_gid_reorder` | bool | `true` | Reorder blocks so each rank owns a slice of every bin. |
| `time/hydro_lat_post_amr_rebalance` | bool | value of `hydro_lat_gid_reorder` | Allow the LAT rebalance after a mesh change. `true` together with `hydro_lat_gid_reorder = false` is fatal. |
| `time/hydro_lat_defer_final_exchange` | bool | unset (derived from capabilities) | Drop the redundant bin-final state exchange. An explicit `false` vetoes the derived choice. |
| `time/hydro_lat_max_nmb_per_rank` | int | none | Refused: fatal on a fresh start. Use `mesh_refinement/max_nmb_per_rank`. A copy in a restart header is dropped with one rank-0 message. |
| `time/hydro_subcycle`, `time/hydro_subcycle_factor` | bool, int | `false`, 1 | The non-LAT global subcycling path. Under LAT both are overwritten (`hydro_subcycle = true`, `hydro_subcycle_factor` = 2^lat_levels). |
| `mesh_refinement/max_nmb_per_rank` | int | unset (`ceil(nb/nranks)`) | The per-rank block cap that the LAT partition uses. Only consulted on adaptive meshes. On a static mesh the cap collapses to the minimum, which pins the rank cuts to equal block counts. |
| `mesh_refinement/sticky_load_balance` | bool | `false` | Honoured only when LAT is on (a message says it is ignored otherwise), and skipped whenever a LAT factor rebalance is active. |
| `gravity/solve_dt` | Real | none | Required (greater than 0) with LAT and self-gravity. The potential is frozen inside a window, so this bounds the window length. Missing or non-positive is fatal. |
| `sink_particles/lat_window_motion_cells` | Real | 1.0 | How many finest cells a sink may move in one window. Larger values allow longer windows. |

**Legacy spellings.** Both spellings are accepted. `hydro_lat` maps to `lat`, `hydro_lat_levels`
to `lat_levels`, and `hydro_lat_same_level`, `hydro_lat_same_level_max_ratio`,
`hydro_lat_neighbor_limiter`, `hydro_lat_diagnostics` and `hydro_lat_union_stage1` to the same
names without `hydro_`. In the other direction, `lat_min_bin_count`, `lat_gid_reorder` and
`lat_post_amr_rebalance` map to the `hydro_lat_` names.

## How it works

**Bins and windows.** Each block computes the largest time step it can take on its own.
The ratio to the global minimum is rounded down to a power of two, capped at 2^lat_levels, and
called the block's factor f (1, 2, 4, ...). Several rules then reduce factors where they are
unsafe: a problem cap, sink pins, the AMR-level collapse, a minimum bin
population and a neighbour limiter that keeps neighbouring factors close (details in the
[implementation notes](Local-Adaptive-Time-Stepping-Implementation-Notes#how-the-factors-are-chosen)).

The largest surviving factor sets the length of a **window**, which is that many finest
steps. Inside a window the driver runs one **tick** per finest step. On tick k, exactly the
blocks whose factor divides k are updated.

```
factor 4 block:  |------------ one step ------------|
factor 2 block:  |---- step ----|---- step ----|
factor 1 block:  | s | s | s | s | s | s | s | s |
tick:             0   1   2   3   4   5   6   7        (window = 8 finest steps)
```

**Ghost zones.** When a block updates, its neighbour may be partway through a longer step.
The neighbour then supplies ghost values interpolated in time between its own start and end
states (linear, or a second-order dense-output polynomial of the Runge-Kutta step when
available), so a fast block always sees ghost data at the right time.

**Conservation.** Each block accumulates the time integral of its own boundary flux and
subtracts what finer or faster neighbours contributed. When a slow block finishes its step,
the difference is added back as a surface correction. This is Berger-Colella refluxing,
generalised to time-step bins, including same-level factor mismatches.

**What runs only at window ends.** AMR regridding, outputs, restarts, history, the
self-gravity solve and the sink operator run only at synchronized window boundaries.

**Window length.** The window is shortened so it fits before `tlim`, never exceeds
`gravity/solve_dt` with self-gravity, and, with sinks, is short enough that a sink moves at
most `lat_window_motion_cells` finest cells.

**Parallel layout.** LAT changes the load-balance goal from equal cost per rank to every
rank having work on every tick. Blocks are dealt out so each rank holds a slice of every bin. The default minimum bin
population scales with the rank count, so the rank count also affects bin membership (see below).

## Practical guidance

### What is supported and refused

A refused module makes the run exit with a message.

| module | status |
| --- | --- |
| hydro (`<hydro>`) | supported |
| self-gravity (multigrid) | supported through the frozen-potential window. The solve runs outside the bins and the window-end reflux corrects the gravitational work. Needs `gravity/solve_dt`. |
| external black-hole gravity | supported |
| sink particles | supported. Run once per window, with factor-1 pinning near the sink. |
| AMR / SMR | supported. Regrids happen at window ends. |
| hydro viscosity and conduction | refused |
| orbital advection, shearing box | refused |
| source terms `const_accel`, `ism_cooling`, `rel_cooling`, `rad_beam` | refused |
| turbulence driving | refused |
| legacy `<radiation>` | refused |
| z4c (evolved metric) | refused |
| ion-neutral two-fluid | refused |
| particles | refused |
| `<remap>` | refused in both directions. Remap with LAT off, then restart the remapped run with LAT on. |
| user problem-generator hooks (sources, time step, boundary conditions) | refused unless flagged `*_lat_safe`. User boundary conditions are refused outright with `<hydro>`. |

Only the `rk1` integrator and rk2-equivalent tableaux (`rk2`, `imex2`) are allowed. `imex3` has
not been analysed under LAT.

### Reproducibility across rank counts

The default `hydro_lat_min_bin_count = 4*nranks` makes bin membership depend on the number of
ranks. A restart of the same state on a different rank count is still conservative and valid,
but follows a different trajectory. Set the key explicitly to reproduce a run on a different
number of ranks. Bin assignment is not stored in checkpoints. It is recomputed from the local
time-step limit at the first step after a restart.

### Self-gravity

Set `gravity/solve_dt`. The window is shortened to at most this value. The solve is pulled
forward instead of truncating a window, which avoids a long descent of ever shorter windows.

### Sinks

Blocks near a sink are forced to factor 1. Raise `lat_window_motion_cells` to allow longer
windows when the sink moves slowly.

### Restarts and outputs

- LAT needs a single global restart file. `single_file_per_rank` with `lat = true` is fatal.
- Checkpoints are written only at window ends. A dump requested while a reflux accumulator
  is non-zero is fatal.
- A restart does not reuse the old bins (see above).

### Checking that it works

- Set `lat_diagnostics = true` to print the bin tables after each limiting pass.
- Rank 0 prints one line, `Mesh: HD LAT per-bin rank distribution ...`, listing for each bin
  how many blocks land on each rank, plus a predicted busy fraction per rank (`util%`).
  Compare it with the measured GPU utilisation.
- `Driver: LAT union stage-1 predictor enabled/disabled (...)` names the reason when the fused
  predictor is off.

### Common problems

| symptom | cause and fix |
| --- | --- |
| Little or no speedup | Most blocks are in the finest bin, or the AMR-level collapse (`lat_same_level = false`) forces whole levels to factor 1. Check `lat_diagnostics`. |
| Fatal error about `solve_dt` | Add a positive `gravity/solve_dt`. |
| Fatal error about `hydro_lat_max_nmb_per_rank` | Remove it and use `mesh_refinement/max_nmb_per_rank`. |
| Results differ when the rank count changes | Set `hydro_lat_min_bin_count` explicitly. |

### Performance

The remaining cost is mainly communication and task-list launch overhead for the boundary
refresh on every tick, which grows with the window length. The design favours correctness
and synchronized AMR and gravity over asynchronous per-rank progress. On a static mesh the
rank cuts are pinned to equal block counts, so the order of blocks inside each bin is the only
balancing tool.

### Limitations

- Same-level mixed bins (`lat_same_level = true`) are a physics-validation risk at large
  factor ratios, because the interpolation is a dense Runge-Kutta polynomial between start
  and end conserved states.
- A window cannot be cut short. The driver is built so this cannot happen, and turns an
  attempt into a fatal error rather than a silent loss of conservation.

## Further reading

- [Implementation notes](Local-Adaptive-Time-Stepping-Implementation-Notes): factor selection
  passes, tick loop, load balance, log lines, code map, module interactions, known issues, tests.
- Example deck: `inputs/TDE_examples/tde_05_fallback_lat.athinput`.
- Design note `docs/lat_implementation_note.tex` / `.pdf`: scope, user controls, factor
  metadata, masks, boundary exchange, delayed flux correction, self-gravity, GID ordering, the
  load-balance objective, AMR and state migration, restart semantics, performance, invariants.
  (Its scope and "current production configuration" sections are out of date. The algorithmic
  sections are not.)
- Berger and Colella (1989), *J. Comput. Phys.* **82**, 64: the refluxing generalised here to bin boundaries.
- Gottlieb (2009): the SSPRK(2,2) tableau whose dense output is used for mixed-cadence ghosts.
