# TDE Analysis Pipeline

`analyze_tde.py` reads AthenaK `hydro_w` binary snapshots, computes BH-centered inertial diagnostics one snapshot at a time, and writes compact plain-ASCII outputs into a single analysis directory such as `analysis_dat/`.

The pipeline keeps four channels separate:

- in-domain snapshot distributions
- sink-crossing loss distributions
- outer-boundary loss distributions
- fallback-shell proxy flux distributions across chosen spherical radii, parameterized by `a_mb = r_tide^2 / (2 R_star)`

This separation is deliberate. Late-time in-domain histograms are not treated as the full system because gas can be lost through the sink / excision boundary and through the outer simulation boundary.
All mass and flux diagnostics are evaluated only in cells with `rho > density_threshold_factor * dfloor` by default, with `density_threshold_factor = 10`.

## Input Data and Conventions

1. The `hydro_w` dumps contain `dens`, `velx`, `vely`, `velz`, `eint`, and `grav_phi`.
2. The BH potential used for orbital diagnostics is the softened Newtonian external potential from `src/pgen/tde_external.cpp`:
   `Phi_BH = - G_code * M_BH / sqrt(|r-r_BH|^2 + eps_soft^2)`.
3. Units are read from the embedded `[units]` block in each dump: `length_cgs`, `mass_cgs`, `time_cgs`, `mu`.
4. The `hydro_w` dumps do not store true face fluxes, so sink and outer losses are estimated from snapshot-local shell / boundary proxies.
5. When live BH metadata is present, BH-centered inertial quantities use `r_rel = x_cell - bh_live_x` and `v_rel = v_cell - bh_live_v`.

## Usage

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python analyze_tde.py
python analyze_tde.py /path/to/binfiles
python analyze_tde.py /path/to/binfiles /path/to/output
python analyze_tde.py /path/to/binfiles /path/to/output /path/to/config.json
```

The default config file is `analysis_config.json`.

By default the pipeline supports resume-style reuse of existing per-snapshot
`*.analysis.dat` files through `resume_existing_outputs: true`.
On reruns into the same output directory, already analyzed snapshots are skipped
as long as the cached per-snapshot file still matches the current binning and contains
the required instantaneous rate channels.

For speed, the defaults also:

- auto-size `batch_blocks` to use more meshblocks per analysis batch on multi-core machines
- use all detected CPU cores unless `numba_threads` is set explicitly
- rebuild later time-function plots directly from per-snapshot `*.analysis.dat` files

## Compact Output Layout

- `TDEExternal.hydro_w.XXXXX.analysis.dat`
- `README_ANALYSIS.md`
- `paper_figures/` after running `plot_analysis.py`

This reduces the file count substantially:

- one combined `.analysis.dat` per processed snapshot / bin file
- no standalone run-level time-series or cumulative `.dat` files

The per-snapshot `.analysis.dat` files are also self-contained enough to rebuild the
global time-function products later. In particular, each snapshot file stores the
scalar `SUMMARY` values needed for mass-budget, rate, tail-budget, branch, and
fallback-shell time-series reconstruction, together with the unit-conversion
scalars needed to interpret them.

## Per-Snapshot File Format

Each `*.analysis.dat` file contains:

- `SUMMARY` rows with columns `record name value unit`
- `HIST` rows with columns
  `record channel quantity bin_left bin_right bin_center dM dM_dX cumulative_M`
- `HIST2D` rows with columns
  `record channel quantity_x quantity_y bin_x_left bin_x_right bin_x_center bin_y_left bin_y_right bin_y_center dM`

Channels used inside a per-snapshot file:

- `in_domain`
- `sink_rate`
- `outer_rate`
- `sink_interval`
- `outer_interval`
- `in_domain_inward`
- `in_domain_outward`
- `in_domain_bound_inward`
- `in_domain_bound_outward`
- `in_domain_unbound`
- `in_domain_compressed_proxy`
- `in_domain_bound_inward_compressed_proxy`
- `shell_inner`
- `shell_fallback_f0p5`, `shell_fallback_f1p0`, `shell_fallback_f1p2`, `shell_fallback_f1p5`
- `shell_outer_stream`
- `fallback_rate_inward_f0p5`, `fallback_rate_outward_f0p5`, `fallback_rate_net_f0p5`
- analogous fallback-rate channels for the other configured radii

Quantities written where available:

- `energy`
- `j`
- `Be`
- `e`

The sink / outer channels in a snapshot file correspond to the interval ending at that snapshot.
The `sink_rate` and `outer_rate` channels are instantaneous proxy flux distributions in `g/s`
at that snapshot and are used internally for resume-safe reconstruction of interval fluxes.
The 2D histogram channels are embedded in the same `.analysis.dat` file, so each processed snapshot produces only one per-bin ASCII data file.
By default the fallback-shell families correspond to `0.5, 1.0, 1.2, 1.5` times `a_mb`.

## Reconstructed Time Functions

`plot_analysis.py` reconstructs global time-series and cumulative-budget figures directly
from the per-snapshot `*.analysis.dat` files. No standalone run-level `.dat` tables are
required.

The per-snapshot `SUMMARY` block carries:

- mass-budget scalars such as `M_domain`, `M_sink_cumulative`, and `M_outer_cumulative`
- rate scalars such as `Mdot_sink`, `Mdot_outer`, `Mdot_control`
- fallback-shell proxy rates, cumulative masses, and normalized `t^(-5/3)` guide values
- tail-budget and branch-diagnostic scalars
- unit-conversion scalars such as `length_cgs`, `mass_cgs`, `time_cgs`, `velocity_cgs`,
  `specific_eint_cgs`, and `angular_momentum_cgs`

## Plotting

`plot_analysis.py` reads the compact ASCII outputs and writes publication-style figures into `paper_figures/`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python plot_analysis.py /path/to/analysis_dat
python plot_analysis.py /path/to/analysis_dat /path/to/figure_dir
```

The plotting script writes PNG versions of:

- `figure_01_mass_budget`
- `figure_02_rates`
- `figure_03_circularization`
- `figure_04_distribution_budget`
- `figure_05_selected_snapshots`
- `figure_06_in_domain_evolution`
- `figure_07_flux_evolution`

## Scientific Caution

The in-domain snapshot histograms are not full late-time system distributions.
Gas can be lost through the inner sink / excision boundary and through the outer simulation boundary.

To build an approximate total budget, combine the in-domain and boundary-loss channels explicitly rather than reading the in-domain distributions alone.

Additional cautions:

- The fallback-shell inward flux is a practical proxy for return across a chosen spherical surface, not necessarily the same as sink accretion.
- The normalized `t^(-5/3)` columns are only visual guides.
- The `compressed_proxy` subsets are snapshot-local compression diagnostics based on negative velocity divergence, not true shocked tracers.
- Deep-tail fits are descriptive only and do not by themselves prove a ballistic fallback law.
- The tidal-energy-scale tail threshold is a physically motivated comparison scale, not a universal exact boundary.
- The fallback-shell radii are defined relative to `a_mb = r_tide^2 / (2 R_star)` once `r_tide` and `R_star` are available.
