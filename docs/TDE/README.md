# TDE user guide

This is the entry point of the documentation for the tidal disruption event (TDE)
workflow of this repository.  The workflow is the one described in

> Jiang, H.-X., Yang, M., Velasco-Romero, D. A., Yu, F., Xia, J.-Z., Li, X., and Mizuno, Y.,
> *An End-to-End Numerical Framework for Tidal Disruption Events with AthenaK*,
> The Astrophysical Journal Supplement Series (accepted).

**Scope.**  This repository is a fork of AthenaK (Stone et al. 2024).  We maintain and
support only the TDE workflow of the paper: the `tde_external` problem generator and the
components it uses, which are the tabulated H/He equation of state, EOS-balanced stellar
initialization, multigrid self-gravity, the dual-energy formalism, localized adaptive
time stepping, conservative remapping and frame conversion, and the `athenak_rt`
post-processing package.  For everything else in AthenaK, use the
[official AthenaK documentation](https://ias-astrophysics.github.io/athenak-docs) and the
[upstream repository](https://github.com/IAS-Astrophysics/athenak).

## Contents

1. [What the framework does](#what-the-framework-does)
2. [Physical setup and code units](#physical-setup-and-code-units)
3. [Build](#build)
4. [Equation of state table](#equation-of-state-table)
5. [Quick start](#quick-start)
6. [How the pieces work together](#how-the-pieces-work-together)
7. [Reproducing the paper](#reproducing-the-paper)
8. [Troubleshooting and resources](#troubleshooting-and-resources)
9. [How to cite](#how-to-cite)
10. [Related documents](#related-documents)

## What the framework does

The framework follows one star from hydrostatic equilibrium through disruption and
fallback in a single calculation.  It is a Newtonian, hydrodynamic model with a
softened point-mass potential for the black hole (BH), gas self-gravity, and a
tabulated hydrogen and helium equation of state (EOS) that includes radiation
pressure.  The workflow has four stages.

| Stage | What happens | Main ingredients |
|---|---|---|
| 1. Initial star and encounter | A star in hydrostatic equilibrium with the tabulated EOS approaches the BH on a parabolic orbit in a non-rotating frame that translates with the star. | EOS-balanced star, translating frame, BH potential and excision, multigrid self-gravity, dual energy |
| 2. Domain enlargement | The debris outgrows the box, so the state is remapped onto larger boxes with a new AMR hierarchy. | conservative-variable remap, refinement shells |
| 3. Frame conversion | After the disruption, the state is boosted to the inertial frame in which the BH is initially at rest, and the live BH moves in that frame. | Galilean conversion in the remap |
| 4. Fallback | The bound debris returns to the BH and is followed for several fallback times. | AMR with stream shells, localized adaptive time stepping (LAT) |

A fifth step, outside the hydrodynamic code, computes synthetic observables from the
snapshots with `athenak_rt` (see [`docs/athenak_rt.md`](../athenak_rt.md)).

## Physical setup and code units

### Units

The example decks use these code units.

| Quantity | Value |
|---|---|
| Mass `M0` | `1 Msun = 1.98841586e33 g` |
| Length `L0` | `2 Rsun = 1.3914e11 cm` |
| Time `t0` | `(L0^3 / (4 pi G M0))^(1/2) = 1270.915 s` |
| Gravity | `4 pi G = 1`, so `G = 1/(4 pi)` in code units |
| Velocity `L0/t0` | `1.0948e8 cm/s` |
| Density `M0/L0^3` | `0.7382 g/cm^3` |

The factor `4 pi` in `t0` follows from `gravity/four_pi_G = 1`.  The stellar mass is fixed
to 1 in code units, and the stellar radius is `problem/star_radius = 0.5`, which is one
solar radius.  The `<units>` block of every deck provides `mass_cgs`, `length_cgs`,
`time_cgs`, and `mu = 1.0`.  The time unit must be consistent with
`gravity/four_pi_G`, because the BH potential uses `G = four_pi_G / (4 pi)`.

### Tidal disruption parameters

For a star of mass `M_*` and radius `R_*` and a BH of mass `M_BH`,

```
r_t   = R_* (M_BH / M_*)^(1/3)          tidal radius
beta  = r_t / r_p                       penetration factor
P_mb  = 2 pi G M_BH (2 dE_mb)^(-3/2),   dE_mb = G M_BH R_* / r_t^2
```

`P_mb` is the orbital period of the most bound debris and sets the fallback time scale.
With the defaults of the example decks (`mass_ratio = 1000`, `beta = 1`,
`star_radius = 0.5`):

| Quantity | Code units | Physical |
|---|---|---|
| `M_BH` | 1000 | `1000 Msun` |
| `r_t = r_p` | 5.0 | `10 Rsun` |
| initial separation `10 r_t` | 50.0 | `100 Rsun` |
| `P_mb` | 88.04 | `1.295 d` |
| pericenter passage | about 21 | about `0.24 P_mb` |
| BH softening `bh_softening` | `5e-3` | `0.01 Rsun` |
| BH excision radius `bh_excise_radius` | 0.2 | `0.4 Rsun` |

### Changing the setup

| To change | Set | Notes |
|---|---|---|
| BH mass | `problem/mass_ratio` (`M_BH / M_*`) | Keep it the same in all five decks. `r_t` scales as `mass_ratio^(1/3)` and `P_mb` as `mass_ratio^(1/2)`, so adapt the box sizes, `tlim`, and output intervals. |
| Penetration factor | `problem/beta` | `r_p = r_t / beta`. Keep `r_p` well above `bh_excise_radius` and resolved by the mesh. |
| Initial separation | `problem/sep_initial` (units of `r_t`) | Parabolic orbits only. It sets the time to pericenter, so extend `time/tlim` of step 1 accordingly. The BH does not need to lie inside the initial box. |
| Eccentricity and inclination | `problem/ecc_bh` (`0 < e <= 1`), `problem/theta_bh` (degrees) | For `e < 1` the star starts at apoapsis. |
| Explicit initial state | `problem/provide_params = true` with `x1..vz1` | State of the star relative to the BH. |
| Star | `<units>` and `problem/star_radius` | The stellar mass is always `M0`, so set `mass_cgs` to the stellar mass, choose `length_cgs`, set `star_radius = R_* / length_cgs`, and set `time_cgs` from the formula above. The star must lie inside the density and temperature range of the EOS table, or the stellar solver stops with an error. |
| Composition | a different EOS table | Generate it with `scripts/generate_lte_table.py`. Note that `athenak_rt` assumes `X = 0.7`, `Y = 0.3`. |

The paper validates one configuration, a `1 Msun`, `1 Rsun` star, a `1000 Msun` BH,
and `beta = 1`.  The framework is not tied to this choice, but other configurations
have not been validated.  Every result uses the softened Newtonian potential.  The
full list of keys is in [`parameters.md`](parameters.md).

## Build

The TDE workflow needs only the `tde_external` problem generator.

```bash
git clone --recursive https://github.com/HongxuanJiang/athenak-transients.git athenak
cd athenak
cmake -S . -B build_tde -D PROBLEM=tde_external -D Athena_ENABLE_MPI=ON \
      -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_VOLTA70=ON \
      -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
cmake --build build_tde -j 16
```

The executable is `build_tde/src/athena`.

* `Kokkos_ARCH_*` must match your GPU.  The paper runs used NVIDIA V100 cards
  (`Kokkos_ARCH_VOLTA70`).  Other examples are `Kokkos_ARCH_AMPERE80` and
  `Kokkos_ARCH_HOPPER90`.
* `Athena_ENABLE_MPI=ON` is required for multi-GPU runs, with one MPI rank per GPU.
* If `kokkos/` is empty after cloning, run `git submodule update --init`.
* A CPU build without CUDA uses the same command with the CUDA options removed, and can
  add `-D Kokkos_ENABLE_OPENMP=ON`.  It is suitable for small tests, but not for the
  production chain.

Requirements, compiler and MPI versions, and instructions for other platforms are in the
upstream pages for [requirements](https://ias-astrophysics.github.io/athenak-docs/requirements.html),
[download](https://ias-astrophysics.github.io/athenak-docs/download.html), and
[build](https://ias-astrophysics.github.io/athenak-docs/build.html).

## Equation of state table

The example decks read `chabrier2021_t13_helm_union_prad_640.table`, the composite H/He
table of the paper (Tomida et al. 2013 chemical-equilibrium EOS at low density, the
Chabrier et al. dense-fluid EOS at high density, a fully ionized HELM-style branch at high
temperature, with radiation pressure added).  The binary `.table` files are not stored in
the git tree.  Get the table in one of two ways.

1. **Build** it with the script in the top-level directory of the repository:

   ```bash
   ./get_eos_table.sh                      # writes eos_tables/chabrier2021_t13_helm_union_prad_640.table
   ./get_eos_table.sh --output-dir DIR     # writes DIR/chabrier2021_t13_helm_union_prad_640.table
   ```

   The script downloads the Chabrier et al. dense H/He tables from the authors' web page,
   checks their SHA-256 sums, and runs `scripts/generate_lte_table.py`, which also reads
   `scripts/scvh95_data/`.  It needs Python 3 with numpy and scipy, and curl or wget, and
   takes about two minutes.  At the end it reports whether the values of the new table are
   identical to those of the table used in the paper.  The Chabrier et al. tables are not
   included in this repository because they are distributed without a license that
   permits redistribution; their source and the manual steps are described in
   [`../../scripts/chabrier2021_data/README.md`](../../scripts/chabrier2021_data/README.md).
   Running `python3 scripts/generate_lte_table.py` without a job name builds every
   maintained table.
2. **Download** `chabrier2021_t13_helm_union_prad_640.table` from the release assets of
   this repository.

The table is a `640 x 640` grid in `ln(rho)` and `ln(T)`, for `X_H = 0.7` and
`Y_He = 0.3` without metals, and it covers `-20 <= log10(rho / g cm^-3) <= 3` in density.  The
generator job covers `0 <= log10(T / K) <= 10`.  The file format, the generator, and
the checks are described in [`../eos_tables.md`](../eos_tables.md) and
[`../wiki/Tabulated-EOS.md`](../wiki/Tabulated-EOS.md).  The design of the EOS
is documented in [`../eos_implementation_note.pdf`](../eos_implementation_note.pdf).

Place the table where the decks find it.  The example decks use
`table = ../chabrier2021_t13_helm_union_prad_640.table`, which is resolved relative to the
directory in which you start the code, one level above each step directory.  A relative
path is tried in the working directory first, then under `$ATHENAK_DATA`, then under
`<source tree>/data`.  You can also override the path on the command line, for example
`hydro/table=/path/to/chabrier2021_t13_helm_union_prad_640.table`, and the other tools
accept the same file through `<tables>/eos_table` (the `athenak_rt` parameter file) or the
path in the snapshot header (plot scripts).

Two more tables are only needed by tools.  The chemistry plots of
`scripts/TDE/bin/plot_slice.py` need `lte_t13_prad_eos.table` next to the main table
(generator job `lte_t13_prad`).

## Quick start

The five input files in [`inputs/TDE_examples/`](../../inputs/TDE_examples) reproduce the
fiducial (FID) calculation of the paper, and
[`inputs/TDE_examples/README.md`](../../inputs/TDE_examples/README.md) has the commands
and file names.  Each step starts from the last restart file of the previous step.

| Step | Deck | Time (`t0`) | Box (`Rsun`) | AMR levels | Frame | Purpose |
|---|---|---|---|---|---|---|
| 1 | `tde_01_disruption.athinput` | 0 to 21 | `64^3` | 9 | translating | EOS-balanced star and first pericenter passage |
| 2 | `tde_02_remap_box256.athinput` | 21 to 24 | `256^3` | 10 | translating | remap to a larger box |
| 3 | `tde_03_remap_box512.athinput` | 24 to 40 | `512 x 512 x 256` | 9 | translating | second remap, refinement shells around the BH |
| 4 | `tde_04_bh_frame.athinput` | 40 to 70 | `512 x 512 x 256` | 8 | BH rest frame | conversion to the BH rest frame |
| 5 | `tde_05_fallback_lat.athinput` | 70 to 440 | `1024 x 1280 x 256` | 8 | BH rest frame | fallback with LAT |

Step 5 runs with LAT on, together with `problem/bh_reciprocal_force=true`, to
`time/tlim=440.2`.  The settle steps of its remap run without LAT, and LAT starts after
the last remap pass.

```bash
REPO=/path/to/athenak
ATHENA=$REPO/build_tde/src/athena
DECKS=$REPO/inputs/TDE_examples

mkdir -p tde_run && cd tde_run
ln -s /path/to/chabrier2021_t13_helm_union_prad_640.table .

for step in 01_disruption 02_remap_box256 03_remap_box512 04_bh_frame 05_fallback_lat; do
  mkdir -p $step
  (cd $step && mpirun -np 10 $ATHENA -i $DECKS/tde_$step.athinput)
done
```

Run every step from its own directory, and keep the step directories next to each other,
because the `<remap>` blocks use relative paths such as
`../04_bh_frame/rst/TDEExternalLTEPrad.00016.rst`.  The chain uses 10 GPUs with 16 GB
each.

## How the pieces work together

The problem generator `src/pgen/tde_external.cpp` connects the modules below.  Each
section is a short summary and points to the detailed page.

### EOS-balanced star and relaxation

With `problem/stellar_structure_mode = eos_balanced`, the initial star is a spherical
hydrostatic profile integrated with the same EOS table that the hydrodynamics uses:
`dm/dr = 4 pi r^2 rho`, `dP/dr = -G m rho / r^2`, and `d rho/dr = rho / (Gamma_1 P) dP/dr`.
The central pressure is bracketed so that the pressure vanishes at `star_radius`, and the
central density is adjusted so that the mass is `M0`.  The internal energy follows from the
table.  This avoids the closure mismatch of a polytropic star mapped onto a tabulated
EOS.  A short velocity damping (`relax_damp`, `relax_tau`, `relax_t_end`, `relax_radius`)
removes the residual motion of the discretized star in the first `0.2 t0`.  The mode
`legacy_polytrope` builds a Lane-Emden star instead.  See
[`../eos_implementation_note.pdf`](../eos_implementation_note.pdf) and the section
"EOS-balanced stellar initialization" of the paper.

### Translating frame and BH potential

While the star is intact and the debris is compact, the simulation runs in a
non-rotating frame that follows the dense gas (`use_translating_frame = true`).  The BH
is an external, softened Newtonian potential, `Phi = -G M / sqrt(r^2 + r_soft^2)`, with
`r_soft = bh_softening`.  The frame acceleration is the mass-weighted BH acceleration of
the gas with `rho > frame_rho_min`, so floor-density gas does not move the frame.  The
gas feels the BH with a weight that ramps from 0 at `hydro/dfloor` to 1 at
`bh_grav_rho_min`.

The BH region is treated as an excised one-way sink.  Cells within `bh_excise_radius`
are reset to the floor state with zero momentum, and only inward flux is accepted.  The
LAT factors of blocks near the BH are limited to 1.  The BH is live in
steps 2 to 5 of the example chain (`bh_live = true`); in step 1 it starts outside the
64 Rsun box, where the gas potential cannot be sampled, and follows the two-body orbit
until the first remap.  Its inertial position and velocity are advanced
with a leapfrog under the gas pull on it, sampled from the self-gravity potential, and in the
LAT stage the equal and opposite of the BH force on the gas is used instead
(`bh_reciprocal_force = true`, which needs LAT, so it is on in step 5 only).  In the translating frame the BH moves with respect to the mesh
because of both its own motion and the frame motion.  After the conversion of step 4 the
frame is the inertial frame in which the BH is initially at rest, and the live BH moves in
it.  Every remap carries the BH position, velocity, and acceleration from the source restart
file.  See the section "Moving-frame dynamics and BH treatment" of the paper,
[the live-BH keys](parameters.md#live-black-hole), and
[the frame and excision keys](parameters.md#4-moving-frame-softening-and-excision).

The paper's TDE setup absorbs gas at the BH through the excision region described above.
The sink particle module (`src/sink_particles`, with example input files in
`inputs/sink_particles/`) is also in the tree as an optional module.  Combining it with
`tde_external` is not part of the validated workflow.

### Multigrid self-gravity and its cadence

The self-gravity potential of the gas solves `nabla^2 Phi = 4 pi G rho` with a geometric
multigrid solver on the AMR hierarchy (Tomida & Stone 2023).  Solving at every
Runge-Kutta stage would dominate the cost, so the potential is refreshed at a physical
cadence `gravity/solve_dt`, starting each solve from the previous solution.  The
production value is `0.03 t0`, about 40 solves per stellar dynamical time
(`1.25 t0`).  With LAT, the length of a synchronization window is limited to `solve_dt`,
so the potential is never older than that, and `solve_dt > 0` is required.  Density
below `gravity/rho_grav_min` is excluded from the source in a smooth ramp.  Details:
[`../wiki/Multigrid-Self-Gravity.md`](../wiki/Multigrid-Self-Gravity.md) and the section
"Self-gravity solver" of the paper.

### Tabulated EOS

`hydro/eos = lte_table_chabrier2021_t13_helm_union_prad` reads the table named by
`hydro/table` and uses it for the conserved-to-primitive conversion, the dual-energy
pressure update, the stellar initialization, and the temperature in post-processing.
`hydro/lte_bounds = clamp` keeps out-of-range states at the edge of the table.  See
[`../wiki/Tabulated-EOS.md`](../wiki/Tabulated-EOS.md) and
[`../eos_tables.md`](../eos_tables.md).

### Dual energy

The cold debris stream is highly supersonic, so the internal energy obtained as
total energy minus kinetic energy becomes inaccurate.  With `hydro/dual_energy = true`
an auxiliary internal-energy density is evolved together with the gas.  It is advected
with the mass flux and receives an operator-split `-P div(v)` source.  The auxiliary
energy is used for the EOS when the energy obtained by subtraction is not positive or is
below `dual_energy_eta1` times the total energy, and the two energies are synchronized
using `dual_energy_eta2`.  The decks use `eta1 = 1e-3` and `eta2 = 1e-4` (Table 1 of the
paper).  Set both explicitly, because the code default of `eta2` is `1e-1`.  See
[`../wiki/Dual-Energy.md`](../wiki/Dual-Energy.md).

### AMR strategy

Refinement combines the standard density criterion with TDE-specific requests.

* `<amr_criterion0>` with `method = min_max`, `variable = hydro_w_d` refines where the
  density exceeds `value_max` and derefines below `derefine_value_max`.
* `<refined_regionN>` sets a minimum level in a box, for example around the BH and the
  returning stream.
* `<amr_criterion1> method = user` enables the problem-specific requests:
  * `bh_max_amr` targets blocks touching the excision sphere at
    `max_level - bh_max_amr_level_offset`.
  * `unbound_amr` targets blocks in which at least `unbound_amr_fill_frac` of the cells
    hold gas unbound from the BH, above the density `unbound_amr_rho_min`, at
    `max_level - unbound_amr_level_offset`.
  * `stream_shell_*` follows the dense stream with spherical shells around the BH.  Up
    to four radial tiers (`stream_shell_level_radius_N`, `_level_offset_N`, `_dr_N`,
    `_rho_frac_N`, `_fill_frac_N`) select a finer or coarser target level with
    distance, and `stream_shell_xsplit_radius` tracks the two sides of the BH separately
    inside a radius.
  * A block is moved toward the finest of the targets that apply to it, and a block
    finer than all of them is derefined unless a standard criterion asks to refine it.
* With LAT, `ncycle_check` and `refinement_interval` are set to 512, so that AMR checks
  happen at most about once per LAT synchronization window.
* `mesh_refinement/max_nmb_per_rank` caps the MeshBlocks per rank and therefore the device
  memory.
  When the cap rejects a refinement pass, the derefinements of that pass are kept and only
  the refinements are cancelled.

#### `<tde_amr>` scheme

A `<tde_amr>` block replaces the BH, unbound-gas and stream-shell requests above with a
single scheme that refines along the local density spine of the stream.  Regions
(nozzle, post-nozzle, self-interaction, apocentre and a spine ladder that gets coarser
with distance) set a level below `max_level`, the stream flanks sit one level below the
spine, and only blocks that the orbital midplane touches are refined beyond
`offplane_level`.  The scheme is opt-in, and a deck without the block runs the previous
logic unchanged.  With the block, `<amr_criterion>` blocks must use `method = user`, and
the `bh_max_amr`, `unbound_amr` and `stream_shell_*` keys must be removed.  Every key,
default and region is listed in the
[`<tde_amr>` table of `parameters.md`](parameters.md#tde_amr-stream-following-refinement).  A usage guide with examples and
tuning advice is in [`amr.md`](amr.md).

All keys are in [`parameters.md`](parameters.md#5-adaptive-mesh-refinement).

### Remap and frame conversion

As the debris grows, the run is restarted on a larger domain by a remap
([`../remap_usage.md`](../remap_usage.md), [`../wiki/Remapping.md`](../wiki/Remapping.md)).
A `<remap>` block on a fresh start loads the source restart file and resamples the
conserved variables onto the new mesh with `2^3` midpoint samples per target cell.
Gas energy is carried as thermal energy, and the total energy is rebuilt from it, so
kinetic energy that the target grid cannot represent is dropped instead of being turned
into heat.  Consequently the remap is not conservative, and the paper reports the
conservation audits along the FID chain.  The BH and frame state are restored from
metadata in the restart file.  Each remap first runs `settle_steps` steps of
refinement-only settling to build the AMR hierarchy, and then projects the source state
again onto the refined hierarchy.

Setting `problem/use_translating_frame = false` in a remap of a translating-frame source
converts the state with a Galilean boost by the frame velocity, skipped in
floor-density cells, and the live BH then moves in the converted frame.  A remap cannot be combined with
`time/lat = true`, and it does not run on a restart.

### Localized adaptive time stepping (LAT)

With `time/lat = true`, each MeshBlock advances with a time step that is a power of two
times the smallest one, up to `2^lat_levels` (`lat_levels = 7` gives a factor of 128), so
that the few blocks near the BH and the nozzle shock do not set the time step of the
whole box.  Neighboring blocks differ by at most a factor of two, and conservation across
mixed-step interfaces is enforced with time-integrated flux corrections, including the
gravitational work.  In the production benchmark of the paper LAT reduces the number of
MeshBlock updates by 85% and increases the throughput by a factor of 3.1.  Details:
[`../wiki/Local-Adaptive-Time-Stepping.md`](../wiki/Local-Adaptive-Time-Stepping.md) and
[`../lat_implementation_note.pdf`](../lat_implementation_note.pdf).

**LAT for MHD and GRMHD.** In this release, LAT is available for hydrodynamics, including self-gravity. The development version of the code also supports LAT for MHD and GRMHD, including GRMHD on dynamical spacetimes. These paths are not included in this public release; they are available on request from Hong-Xuan Jiang (masterjoe2000@outlook.com).

### Outputs and restart

Each deck writes a history file, `hydro_w` binary snapshots, and restart files.  Output
numbering continues across remaps.  See [`outputs_and_analysis.md`](outputs_and_analysis.md).

## Reproducing the paper

### FID chain

Run the five steps of [Quick start](#quick-start).  The FID calculation reaches
`5 P_mb = 440.2 t0`.  The setup of Table 1 of the paper is in the decks: `RK2`, `PLM`,
`HLLE`, `CFL` 0.4 in step 1 and 0.5 afterwards, `MeshBlocks` of `32^3` cells,
`dfloor = 1e-10`, `pfloor = 1e-12`, `solve_dt = 0.03`, and `lat_levels = 7`.  The finest
cell of the fallback stage is `0.0625 Rsun`, which gives `r_t / dx = 160`.
The high-resolution (HR) run of the paper continues from the step-4 restart with 10 AMR
levels and a finer vertical grid.  It is not included.

### Self-gravity cadence test

The section "Self-gravity cadence during the disruption" of the paper compares the
production cadence with a twelve times shorter one.  Repeat steps 1 to 4 in a separate
directory with the cadence changed on the command line of every step, for example

```bash
(cd 01_disruption && mpirun -np 10 $ATHENA -i $DECKS/tde_01_disruption.athinput gravity/solve_dt=0.0025)
```

and likewise for steps 2 to 4.  The paper compares the two runs from pericenter to
`t = 0.6 P_mb = 52.8 t0`, which lies inside step 4, so `time/tlim` of step 4 can be
shortened for the test.  Compare the debris energy distributions with
`scripts/TDE/bin/analyze_tde.py` (see [`outputs_and_analysis.md`](outputs_and_analysis.md)).

### LAT benchmark

The section "Production TDE benchmark" of the paper restarts one checkpoint twice, once
with LAT and once without, with all other settings identical.  With the example
decks, run step 5 with `time/lat=false problem/bh_reciprocal_force=false time/tlim=71.0`
to get a checkpoint at `t = 71`, and restart from
`05_fallback_lat/rst/TDEExternalLTEPrad.00017.rst` in two separate copies of the
directory.  Run one with `time/lat=true problem/bh_reciprocal_force=true time/tlim=...` and one with
`time/lat=false problem/bh_reciprocal_force=false time/tlim=...`, ending both at the same
time.  The paper uses five time units (`5 t0`) from a common
checkpoint.  Compare the wall time, the number of cycles, and the density fields on the
same grid.  The LAT schedule depends on `time/hydro_lat_min_bin_count`, whose default is
`4 * nranks`, so use the same number of ranks in comparisons.

## Troubleshooting and resources

| Symptom | Cause and remedy |
|---|---|
| `Failed to read lte_table` or the table is not found | The path in `hydro/table` is resolved from the working directory. Run from the step directory with the table in its parent, set `$ATHENAK_DATA`, or pass `hydro/table=/absolute/path`. |
| The EOS-balanced star fails to build | The star is outside the rho-T range of the table. Use a table that covers the stellar regime. Do not rely on clamping. |
| `A remap cannot run inside a LAT window` | A remap was called in the middle of a run while LAT windows were active. `tde_external` pauses LAT for its settle steps, so this message points at another problem generator. |
| `time/lat with gravity/self_gravity=true requires gravity/solve_dt > 0` | LAT needs the self-gravity cadence to bound the window. Set `gravity/solve_dt`. |
| `problem/remap = true is a retired key` | Migrate the deck to the `<remap>` block, see [`parameters.md`](parameters.md). |
| `use_translating_frame = false is only supported for restart/remap-based TDE conversion` | A fresh start needs the translating frame. Use `false` only in a step with a `<remap>` block or a restart. |
| `Restart compatibility check failed` | The restart or remap source was written with a different EOS or table. Set `hydro/allow_reinterpretive_restart = true` only if the change is intended. The remap decks do. |
| A remap source is not found | `remap/source` is relative to the directory of the run. Keep the step directories side by side. |
| Out of device memory | Lower `mesh_refinement/max_nmb_per_rank` and use more ranks. The AMR hierarchy grows fastest after pericenter, and step 2 of the chain needs all 10 GPUs of 16 GB. The memory of step 5 grows as the debris spreads. |
| Results of LAT runs differ with the number of ranks | The default of `time/hydro_lat_min_bin_count` is `4 * nranks`. Set it explicitly for reproducibility. |
| A step stops early | Continue it with `athena -r rst/<last file>.rst` in its directory. |

Notes on resources.

* One rank per GPU.  The paper runs used one node with ten 16 GB V100 GPUs.
* Snapshots of the late stages are large, and the restart files needed by the chain are
  listed in the [example README](../../inputs/TDE_examples/README.md).
* Analysis and radiative post-processing run on CPUs.  `athenak_rt` needs about 10 GB at
  `1024^3` ray samples, see [`docs/athenak_rt.md`](../athenak_rt.md#6-performance-and-memory).

## How to cite

If you use this framework, please cite the TDE paper and the AthenaK code paper.  If you
use the radiative post-processing `athenak_rt`, please also cite Yang et al. (2026).

```bibtex
@article{Jiang2026TDE,
  author  = {Jiang, Hong-Xuan and Yang, Mengqi and Velasco-Romero, David A. and
             Yu, Fangyuan and Xia, Jing-Ze and Li, Xinyu and Mizuno, Yosuke},
  title   = {An End-to-End Numerical Framework for Tidal Disruption Events with {AthenaK}},
  journal = {The Astrophysical Journal Supplement Series},
  year    = {2026},
  note    = {accepted},
  archivePrefix = {arXiv},
  eprint  = {2609.37859}
}

@article{Yang2026Engulfment,
  author  = {Yang, Mengqi and Lai, Dong and Wu, Fuyuan and Zhang, Jie},
  title   = {Engulfment of Eccentric Planets by Giant Stars: Hydrodynamics and Light Curves},
  journal = {The Astrophysical Journal},
  year    = {2026},
  volume  = {998},
  pages   = {118},
  doi     = {10.3847/1538-4357/ae3157},
  archivePrefix = {arXiv},
  eprint  = {2510.25547}
}

@article{Stone2026AthenaK,
  author  = {Stone, James M. and Mullen, Patrick D. and Fielding, Drummond and
             Grete, Philipp and Guo, Minghao and Kempski, Philipp and Most, Elias R. and
             White, Christopher J. and Wong, George N.},
  title   = {{AthenaK}: A Performance-portable Version of the {Athena++} Adaptive Mesh
             Refinement Framework},
  journal = {The Astrophysical Journal Supplement Series},
  year    = {2026},
  volume  = {283},
  number  = {1},
  pages   = {27},
  doi     = {10.3847/1538-4365/ae3717},
  eprint  = {2409.16053},
  archivePrefix = {arXiv}
}
```

Please update the first entry with the volume and DOI once the paper is published.
The multigrid solver builds on Tomida & Stone (2023, ApJS 266, 7), LAT follows Liska et
al. (2022, ApJS 263, 26), and the EOS uses Tomida et al. (2013), Chabrier et al. (2019,
2021), and Timmes & Swesty (2000).  Cite these where relevant.  The post-processing
package uses MESA opacity tables, see [`docs/athenak_rt.md`](../athenak_rt.md).

## Related documents

| Topic | Document |
|---|---|
| Example decks and commands | [`inputs/TDE_examples/README.md`](../../inputs/TDE_examples/README.md) |
| All `<problem>` and related keys | [`parameters.md`](parameters.md) |
| Outputs and analysis scripts | [`outputs_and_analysis.md`](outputs_and_analysis.md), [`bin/README_ANALYSIS.md`](bin/README_ANALYSIS.md) |
| Radiative post-processing | [`docs/athenak_rt.md`](../athenak_rt.md), [`athenak_rt/README.md`](../../athenak_rt/README.md) |
| Remap | [`../remap_usage.md`](../remap_usage.md), [`../wiki/Remapping.md`](../wiki/Remapping.md) |
| LAT | [`../wiki/Local-Adaptive-Time-Stepping.md`](../wiki/Local-Adaptive-Time-Stepping.md), [`../lat_implementation_note.pdf`](../lat_implementation_note.pdf) |
| Multigrid self-gravity | [`../wiki/Multigrid-Self-Gravity.md`](../wiki/Multigrid-Self-Gravity.md) |
| Tabulated EOS | [`../wiki/Tabulated-EOS.md`](../wiki/Tabulated-EOS.md), [`../eos_tables.md`](../eos_tables.md), [`../eos_implementation_note.pdf`](../eos_implementation_note.pdf) |
| Dual energy | [`../wiki/Dual-Energy.md`](../wiki/Dual-Energy.md) |
