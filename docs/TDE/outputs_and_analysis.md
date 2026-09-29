# Outputs and analysis of TDE runs

This page describes the files that a `tde_external` run writes and the scripts in
`scripts/TDE/bin/` that turn them into slices and budget diagnostics.  The radiative
post-processing of snapshots is a separate package, described in
[`docs/athenak_rt.md`](../athenak_rt.md).

## 1. What a run writes

Every example deck defines three outputs.  The base name is `job/basename`, which is
`TDEExternalLTEPrad` in all decks.  A step runs in its own directory, and the files are
written relative to that directory.

| Output block | Setting in the decks | File | Content |
|---|---|---|---|
| `<output1>` | `file_type = hst`, `dt = 0.1` | `TDEExternalLTEPrad.hydro.hst` | Volume-integrated history, text |
| `<output2>` | `file_type = bin`, `variable = hydro_w`, `dt` from 1 to 8 | `bin/TDEExternalLTEPrad.hydro_w.NNNNN.bin` | Snapshot of the primitive variables |
| `<output3>` | `file_type = rst`, `dt` from 3 to 10 | `rst/TDEExternalLTEPrad.NNNNN.rst` | Restart file |

The output interval `dt` is in code time units, and the file number `NNNNN` counts
outputs.  After a remap the numbering continues from the source restart file, so step 2
starts at the number that follows the last file of step 1.  The `bin/` and `rst/`
directories are created by the code.  The mapping between restart files and steps is
tabulated in [`inputs/TDE_examples/README.md`](../../inputs/TDE_examples/README.md).

### History file

The `.hst` file has a header line that numbers the columns, `[1]=time [2]=dt`, followed
by one column per history quantity.  For the tabulated EOS the hydro block contains
`mass`, `1-mom`, `2-mom`, `3-mom`, `tot-E`, `1-KE`, `2-KE` and `3-KE`, which are
volume integrals over the whole box in code units (`src/outputs/history.cpp`).
These include the floor atmosphere and the excised region, so subtract the floor
contribution when you need the mass of the debris.  A passive scalar would add a
`scal-N` column.

If the deck sets `gravity/lat_time_centered_work = true` (an energy-accounting option
for LAT runs that the example decks do not use), the problem generator also writes a
second file, `TDEExternalLTEPrad.user.hst`, with the energy ledger columns
`K_BH`, `U_bhgas`, `U_self`, `W_cent`, `B_mass`, `B_E`, `B_Wself`, `B_Wbh`, `E_remap`,
`bh_x`, `bh_y`, `bh_z`, `bh_vx`, `bh_vy`, `bh_vz`, `phi_age`, `phi_int`, `mesh_vol`,
`E_floor`, `E_asym`, and `W_frame` (`src/pgen/bh_dynamics.hpp`,
`src/pgen/tde_external.cpp`).  The option requires `time/lat = true`,
`gravity/rho_grav_min <= hydro/dfloor`, and no `gravity/mask_radius`
(`src/gravity/gravity.cpp`).  Read the column names from the header of the file.

### Binary snapshots

`variable = hydro_w` writes the primitive hydro variables.  The dumps used by the
analysis and post-processing tools contain `dens`, `velx`, `vely`, `velz`, `eint`, and,
with self-gravity on, `grav_phi` (`src/outputs/basetype_output.cpp`).  All
quantities are in code units.  Each file carries a header with the time, cycle, mesh
description, and a copy of the input parameters, including the `<units>` block and the
runtime BH state, so a snapshot is self-describing.  The BH position and velocity in
the simulation frame are written to the `<problem>` block of the header as
`bh_live_x`, `bh_live_y`, `bh_live_z` and `bh_live_vx`, `bh_live_vy`, `bh_live_vz`, and
the frame origin as `frame_live_*` (`src/pgen/tde_external.cpp`).  Because the BH
is live (`problem/bh_live`), it moves: the header of every snapshot and restart file records
its state at the time of the dump.  In the BH rest frame of steps 4 and 5 the frame origin
is zero and `bh_live_x/y/z` and `bh_live_vx/vy/vz` are the BH position and velocity in the
frame.  In the translating frame of steps 1 to 3 they are relative to the moving mesh.

Snapshots are large.  A late FID snapshot on the 1024 x 1280 x 256 Rsun box is written
as one file with all MeshBlocks, and the sizes grow with the number of MeshBlocks
(`max_nmb_per_rank` times the number of ranks is an upper bound).

### BH position and velocity

| Where | Keys or columns | Content |
|---|---|---|
| Snapshot and restart header, `<problem>` block | `bh_live_x`, `bh_live_y`, `bh_live_z`, `bh_live_vx`, `bh_live_vy`, `bh_live_vz` | BH position and velocity in the simulation frame at the dump time |
| Same | `frame_live_x/y/z`, `frame_live_vx/vy/vz` | Origin and velocity of the moving frame, zero in the BH rest frame |
| Same | `bh_live_inertial_x/y/z`, `bh_live_inertial_vx/vy/vz` | Inertial BH state |
| Same | `bh_live_ax`, `bh_live_ay`, `bh_live_az` | Last gas acceleration of the BH |
| Same, with `bh_reciprocal_force` | `bh_pair_gas_impulse_x/y/z`, `bh_pair_impulse_residual` | Cumulative gas impulse on the BH and the reconciliation residual of the last LAT window |
| `TDEExternalLTEPrad.user.hst` (only with `gravity/lat_time_centered_work = true`) | `bh_x`, `bh_y`, `bh_z`, `bh_vx`, `bh_vy`, `bh_vz`, `K_BH` | BH trajectory and kinetic energy as a time series |

These keys are written when `bh_live` is on (`src/pgen/tde_external.cpp`, stored
by `StoreTDEExternalMetadata`).  A restart and every remap read them back
(`RestoreLiveBHMetadata`), so the trajectory continues through the whole chain.  The
example decks do not write the `.user.hst` file, so to follow the BH as a time series read
the header of the snapshots (the analysis scripts do this), or add the option described
above.  `athenak_rt` and `scripts/TDE/bin/analyze_tde.py` use the BH position from the header.
Do not set these keys by hand.

### Restart files

A restart file holds the full state including the parameter input, so
`athena -r rst/<file>.rst` continues a run, and command-line overrides such as
`time/tlim=...` or `time/lat=true` are applied on top of the stored input.  The remap
of the next step reads the restart file through `<remap>/source`.  Only the restart
files that are named in the chain are needed, so older ones can be removed to save
space.

## 2. Slices and derived quantities

`scripts/TDE/bin/plot_slice.py` is a slice plotter for `.bin` files with TDE-specific
options.  It runs with numpy and matplotlib only.

```bash
python3 scripts/TDE/bin/plot_slice.py <snapshot.bin> <quantity> <output.png> [options]
```

`<quantity>` is any variable stored in the file (`dens`, `velx`, `eint`, `grav_phi`) or
a derived quantity written as `derived:<name>`, for example `derived:T`,
`derived:pgas`, `derived:c_s`, `derived:vel_norm`, `derived:mu`, `derived:xion`,
`derived:xh2`, `derived:xhe1`, `derived:xhe2`, `derived:gamma1`, `derived:gamma3m1`,
and `derived:beta_rad`.  The thermodynamic quantities are recovered from the EOS table
recorded in the snapshot header.  Run `plot_slice.py <snapshot.bin> derived:? show` to
list the available names, and `plot_slice.py -h` for all options.

Two points matter for the tabulated H/He EOS.

* The table path is taken from the header (`hydro/table`).  If it is relative, the
  script tries the working directory, the directory of the snapshot, and a path
  relative to the script.  Run the script from a directory in which the header path
  resolves, or copy or link the table there.
* The chemistry fractions (`xh2`, `xion`, `xhe1`, `xhe2`) of the Chabrier-based table
  are taken from a companion table `lte_t13_prad_eos.table`, which must lie in the
  same directory as the main table (`scripts/TDE/bin/plot_slice.py`).  Generate it
  with `python3 scripts/generate_lte_table.py lte_t13_prad`.

Frequently used options.

| Option | Meaning |
|---|---|
| `-d {x,y,z}`, `-l VALUE` | Slice normal and position (default `z`, `0`) |
| `--x1_min`, `--x1_max`, `--x2_min`, `--x2_max` | Plot window in code units |
| `-c CMAP`, `-n log`, `--vmin`, `--vmax` | Colormap, norm, and limits |
| `--orthogonal_triptych` with `--ortho_{x,y,z}_{min,max}` | Three orthogonal slices in one figure |
| `--streamlines` | In-plane velocity streamlines |
| `--bound_unbound_contour` | Contour where the specific orbital energy relative to the BH vanishes |
| `--grid` | Outline the MeshBlocks |
| `--no_bh_mask`, `--bh_marker` | Toggle the excision-sphere mask and BH marker |
| `--notex`, `--dpi` | Disable LaTeX labels, set the resolution |

Example for a midplane density map of a late snapshot in code units, with a fixed
color range:

```bash
python3 scripts/TDE/bin/plot_slice.py bin/TDEExternalLTEPrad.hydro_w.00327.bin dens dens.png \
    -c Spectral_r -n log --vmin 1e-10 --vmax 1 \
    --x1_min -128 --x1_max 384 --x2_min -192 --x2_max 440 --notex
```

Two batch helpers run `plot_slice.py` over all `.bin` files in the current directory
in parallel (`xargs -P`).  They expect `plot_slice.py` in the working
directory (`PLOT_SCRIPT`, default `./plot_slice.py`) and take the number of jobs as the
first argument and the output directory as the second.

| Script | Products |
|---|---|
| `scripts/TDE/bin/plot_density_midplane_parallel.sh [jobs] [outdir]` | Midplane density, code-unit limits, secondary cgs colorbar |
| `scripts/TDE/bin/plot_density_slices_parallel.sh [jobs] [outdir]` | Density, speed, potential, sound speed, pressure, temperature, `xh2`, `xion`, `xhe1`, `xhe2`, `Gamma_1`, `Gamma_3-1`, `mu`, `beta_rad` |

Windows, color limits and products are set through environment variables (for example
`PLOT_X1_MIN`, `DENS_VMIN`, `PLOT_PRODUCTS`), which are listed at the top of each
script.  The defaults of `plot_density_slices_parallel.sh` are adapted to the step-5 box.

```bash
cd 05_fallback_lat/bin
cp /path/to/athenak/scripts/TDE/bin/plot_slice.py .
PLOT_PRODUCTS=dens,temp \
  /path/to/athenak/scripts/TDE/bin/plot_density_slices_parallel.sh 8 ../slices
```

`scripts/TDE/plot_density_profile.py` takes no arguments.  It scans the current
directory for `.bin` files, picks the first and the last, and writes solid-angle
averaged radial density profiles.

## 3. Mass, energy, and angular-momentum budgets

`scripts/TDE/bin/analyze_tde.py` reads `hydro_w` snapshots one at a time and writes
compact ASCII files with BH-centered distributions.  The full description, the file
formats, and the caveats are in [`bin/README_ANALYSIS.md`](bin/README_ANALYSIS.md).  This
section gives the commands and the practical points.

Requirements: Python 3 with numpy and numba.  `plot_analysis.py` also needs matplotlib.

```bash
python3 scripts/TDE/bin/analyze_tde.py [INPUT_DIR [OUTPUT_DIR [CONFIG.json]]]
python3 scripts/TDE/bin/plot_analysis.py [ANALYSIS_DIR [FIGURE_DIR]]
```

| Argument | Default |
|---|---|
| `INPUT_DIR` | current directory, every `*.bin` file in it is used (not recursive) |
| `OUTPUT_DIR` | `INPUT_DIR/analysis_dat` |
| `CONFIG.json` | `analysis_config.json` next to the script |
| `ANALYSIS_DIR` (plotting) | `./analysis_dat` |
| `FIGURE_DIR` (plotting) | `ANALYSIS_DIR/paper_figures` |

Typical use for the fallback stage:

```bash
python3 scripts/TDE/bin/analyze_tde.py 05_fallback_lat/bin 05_fallback_lat/analysis_dat
python3 scripts/TDE/bin/plot_analysis.py 05_fallback_lat/analysis_dat
```

What the analysis computes, per snapshot:

* **Orbital quantities** relative to the BH, using the softened BH potential of the run
  and the BH state from the header: specific energy, specific angular momentum `j`, the
  Bernoulli-like energy `Be`, and the eccentricity `e`.  Units are taken from the
  embedded `<units>` block.
* **Distributions** of mass in these quantities, in 1D and 2D histograms, in the
  channels `in_domain` (with bound/unbound and inward/outward splits), `sink_rate`,
  `outer_rate`, and shell channels at `0.5, 1.0, 1.2, 1.5` times
  `a_mb = r_t^2 / (2 R_star)` for the fallback proxy.
* **Budgets** of mass in the domain, lost through the sink, and lost through the outer
  boundary, with the rates `Mdot_sink`, `Mdot_outer`, and the fallback-shell rates.
* Only cells with `rho > density_threshold_factor * dfloor` are included (default factor
  10), so the floor atmosphere does not enter.

Outputs.

| File | Content |
|---|---|
| `<snapshot stem>.analysis.dat`, for example `TDEExternalLTEPrad.hydro_w.NNNNN.analysis.dat` | One per snapshot, with `SUMMARY`, `HIST`, and `HIST2D` records |
| `analysis_global_timeseries.dat`, `analysis_cumulative_distributions.dat` | Run-level tables rebuilt from the per-snapshot files at the end of a completed run, read by `plot_analysis.py` |
| `README_ANALYSIS.md` | Notes written next to the data |
| `paper_figures/figure_01_mass_budget.png` to `figure_07_flux_evolution.png` | Written by `plot_analysis.py`: mass budget, rates, circularization, distribution budget, selected snapshots, in-domain evolution, flux evolution |

Practical points.

* **Set the sink radius.**  `sink_radius_code` in `analysis_config.json` is `0.2`, which
  matches the `bh_excise_radius` of the example decks.  Change it if you changed the
  excision radius.  The run header also stores the excision radius.
* **Set the tidal radius if needed.**  `r_tide_code` and `stellar_radius_code` are
  `null` in the shipped configuration, in which case they are derived from the
  snapshot header where possible.  Set them explicitly for non-standard setups.
* **Resume.**  With `resume_existing_outputs = true` (the default), snapshots that
  already have a matching `.analysis.dat` file are skipped.  With `clean_output_dir =
  true`, stale run-level files are removed when the analysis starts, and the
  per-snapshot files are kept for reuse.  The plotting script needs the run-level tables,
  so run `analyze_tde.py` again on the same output directory once an interrupted
  analysis has finished.
* **Cost.**  The analysis is CPU and memory bound and uses numba.  `numba_threads` and
  `batch_blocks` in the configuration control the parallelism, and the shipped values
  (32 and 32) assume a many-core machine.  Lower them on a laptop.
* **Interpretation.**  Sink and outer losses are proxies from snapshot-local shells,
  because the dumps carry no face fluxes, and the in-domain histograms are not the full
  late-time distribution.  Read the "Scientific Caution" section of
  [`bin/README_ANALYSIS.md`](bin/README_ANALYSIS.md) before quoting numbers.  The
  energy-inferred fallback curves are not direct sink accretion rates, as stated in the
  paper.

### Configuration keys

`analysis_config.json` holds the keys below.  The values are those shipped with the
repository.

| Key | Value | Meaning |
|---|---|---|
| `snapshot_selection` | `start`, `stop` `null`, `stride` 1, `limit` `null` | Which snapshots are processed |
| `sink_radius_code`, `sink_shell_cells` | `0.2`, `1.0` | Sink radius and shell thickness in cells |
| `outer_shell_cells` | `1.0` | Shell thickness for the outer-boundary proxy |
| `control_radius_code` | `1.0` | Control sphere radius for `Mdot_control` |
| `fallback_radius_multipliers` | `[0.5, 1.0, 1.2, 1.5]` | Fallback shells in units of `a_mb` |
| `density_threshold_factor` | `10.0` | Density cut in units of `dfloor` |
| `tail_fit_min_populated_bins`, `fit_tail_alpha` | `8`, `true` | Tail fit of the energy distribution |
| `binning.energy`, `binning.j`, `binning.Be`, `binning.e` | 640, 480, 640, 240 bins | 1D histogram binning |
| `binning_2d` | 160 x 128 and 128 x 128 | 2D histogram binning |
| `numba_threads`, `batch_blocks`, `checkpoint_stride` | `32`, `32` | Parallelism and cache checkpointing |
| `resume_existing_outputs`, `clean_output_dir` | `true` | Cache behavior described above |

Further keys (compression thresholds, tail thresholds, and outer stream radius) are
documented in the script and in `bin/README_ANALYSIS.md`.

## 4. Snapshots to synthetic observables

For luminosities, spectra, and images use `athenak_rt`.  It reads the same
`hydro_w` `.bin` files (only `dens` and `eint` are needed) and the EOS table named in the
header.  See [`docs/athenak_rt.md`](../athenak_rt.md).
