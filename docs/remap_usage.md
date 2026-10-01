# Remap usage

This page is the user guide for running a restart remap: keys, launching, examples, refusals and troubleshooting. [Remapping](Remapping) explains how the remap works and what it guarantees.

## Summary

The remap module resamples the state of an existing restart file onto a **new mesh** (different domain, resolution, MeshBlock layout or refinement) at the start of a fresh run. It works with any problem generator (pgen) and needs no pgen code.

**Use it** to widen or shrink a domain, change the MeshBlock size for a different rank count, or add and move refinement. For a plain continuation on the same mesh, use an ordinary restart (`athena -r`).

**Do not use it** when you need exact conservation of mass, momentum and energy: the transfer is second-order accurate and [not conservative](Remapping#guarantees-and-limits). It also cannot be used for a 2D run, with an evolved spacetime (`<z4c>`), or in the same run as [LAT](Local-Adaptive-Time-Stepping) (local adaptive time stepping). [What is refused](#what-is-refused) lists every case.

The design background is in `docs/remap_module_design.md`.

## Quick start

Add a `<remap>` block to the input deck of the new (target) run and launch it with `athena -i`, never `-r` (a restart never remaps):

```ini
<remap>
enable = true                    # keep the key explicit so the command line can flip it
source = rst/OldRun.00042.rst    # restart written by the OLD mesh configuration
```

The `<mesh>`, `<meshblock>` and `<mesh_refinement>` blocks of the deck describe the new mesh. Any key can be overridden on the command line, for example `remap/source=rst/OtherRun.00010.rst` or `remap/enable=false`.

After the problem generator has built its own state on the new mesh, the module loads the source restart and overwrites:

- the **gas**: hydro or MHD conserved variables, passive scalars and the dual-energy column;
- the **face-centered B**, for MHD sources;
- the radiation intensities `i0`, in `keep` mode only (see [`radiation_i0`](#parameters));
- **time, time step and cycle number**, and by default the `<outputN>` file numbering.

[Remapping](Remapping#how-it-works) describes how each part is resampled.

Before the first run:

- `time/nlim` and `time/tlim` in the new deck are **absolute**: the run continues from the source's `ncycle` and time. To run 500 more cycles after a source that stopped at cycle 1200, set `nlim = 1700`.
- A remap can be combined with LAT: `time/lat = true` may stay in the deck. The remap at the start of the run happens before any LAT window exists. See [Remap with LAT](#remap-with-lat).
- Re-measure mass, momentum and energy after the remap. They are not carried over exactly.
- `source` must be a single global restart file. A truncated file, or a restart written as one file per MPI rank, is refused.
- The physics must match the source: see the [checklist](#checklists).

## Parameters

All keys live in `<remap>`. The block only takes effect if it exists. This is the single parameter table; [Remapping](Remapping) links here.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `enable` | bool | `true` | Master switch, overridable from the command line. |
| `source` | string | `""` | Source restart path. Required: an empty value is fatal once the remap runs. |
| `copy_output_state` | bool | `true` | Carry each `<outputN>` `file_number` and `last_time` of the source forward, matched by block name. |
| `transition_band` | bool | `true` | Smooth the transition across the old domain boundary. `false` is pure interpolation, for example in convergence tests: no fade, a hard edge at the old box. It does not change the B taper. |
| `band_mode` | string | `auto` | What a target cell outside the source box gets. `floor`: the ambient floor state (the Newtonian TDE behaviour). `keep`: the state the pgen already wrote. `auto`: `keep` for GR, `floor` for Newtonian. Case-insensitive; any other value is fatal. `floor` with a GR source is fatal. |
| `b_coarsen_ok` | bool | `false` | Accept an MHD source that has mesh refinement. B is then restricted to the source root grid and fine-level B structure is lost. The gas keeps full detail. |
| `b_taper_root_cells` | int | `4` | Width, in source root cells, of the smooth taper outside the source box. It is applied to the vector potential (hence to B) in every mode, and to the remapped gas in `keep` mode. `0` gives a hard cut-off. |
| `b_report` | bool | `true` | Print the face-centered B diagnostics. Three warnings are printed regardless: an inverse-curl residual above 1e-8 of the maximum field strength, source B that does not vanish at the old boundary, and the keep-mode seam divergence. |
| `settle_steps` | int | `0` (`tde_external`: `20`) | Plain steps to run after the startup remap before the state is remapped again onto the adapted mesh (a settle pass). `0` means no settle. Must be `>= 0`. See [Settle steps](#settle-steps). |
| `settle_passes` | int | `1` | Number of settle passes, each preceded by `settle_steps` steps. Must be `>= 0`. |
| `radiation_i0` | bool | `true` | Remap the angular intensities `i0` when both sides carry `<radiation>`. Only the `keep` pass applies them. A `floor` remap leaves the target's own `i0` untouched even though the startup banner reports the group as loaded. |

A problem generator may seed a different default by reading a key with `GetOrAdd` before the remap runs; `tde_external` does this for `settle_steps` (20).

## Worked examples

### Widen a domain: the TDE chain

Steps 2 to 5 of `inputs/TDE_examples/` each carry a `<remap>` block that loads the last restart of the previous step. Steps 2, 3 and 5 enlarge the box (the README of that directory tabulates the sizes) and step 4 keeps it and converts to the black-hole rest frame. Step 3 (`tde_03_remap_box512.athinput`) reads:

```ini
<remap>
enable        = true
source        = ../02_remap_box256/rst/TDEExternalLTEPrad.00008.rst
settle_steps  = 10
settle_passes = 1
```

Each step runs in its own directory with `athena -i`, so the relative `source` path resolves. The pgen is `tde_external`, a Newtonian run, so `band_mode = auto` resolves to `floor`: the new outer region becomes ambient floor gas, with a transition band at the old boundary. Time, cycle and output numbers continue from the source. Step 5 ships with `time/lat = false` and is continued with LAT on by a restart. With `time/lat=true` on the command line the whole step can run in one go, see [Remap with LAT](#remap-with-lat). `inputs/TDE_examples/README.md` has the full chain.

### Change the MeshBlock size or rank count

Copy the old deck and change only `<meshblock>` (and the rank count). Cell width and alignment are unchanged, so every target cell coincides with a source cell and is copied: the remap is the identity and conserves exactly. The same holds on the overlap when you also widen the domain by whole cells.

One exception applies in `floor` mode (the Newtonian default). The transition band still acts at the boundary of the old domain, so cells within about 3 source cells of it are reconstructed, not copied. That changes the boundary layer unless its gas is already close to the floor. Set `transition_band = false` to keep the layer, at the price of a hard edge when the domain is widened.

### Add refinement to an MHD run

Add `<mesh_refinement>` (or SMR regions) to the deck of an MHD run whose source is a single-level mesh. B is rebuilt as the curl of one potential, so it is divergence-free on every target block, also across fine/coarse boundaries; check `mhd_divb` after the first cycles. If the source itself is multi-level, set `b_coarsen_ok = true` and accept the loss of fine-level B.

### Remap with LAT

Leave `time/lat = true` in the deck:

```bash
athena -i new.athinput time/lat=true            # remaps, then runs with LAT
```

The remap at the start of the run happens before the first LAT window, and the driver builds the LAT bins from the remapped state at its first time step. LAT needs hydro without MHD, as always (see [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping#what-is-supported-and-refused)).

With `settle_steps > 0` (see [Settle steps](#settle-steps)) the settle steps and the passes that end them are plain steps of the global time step: LAT is paused while they run, and LAT starts at the first window after the last pass. Nothing needs to be set for this.

The older route still works: remap with `time/lat = false`, let the run write a restart, then continue it with `athena -r rst/NewRun.00017.rst time/lat=true`.

A problem generator that calls `remap::LoadAndApplyRemap` in the middle of a run has to pause LAT first, see [Writing a pgen that participates](#writing-a-pgen-that-participates).

### Settle steps

The first remap puts the source state on the target mesh as it was built at the start. The mesh then adapts to that state, so a mesh that is adaptive can be refined differently from the one the state was laid on. With `settle_steps = N` the run therefore takes `N` plain steps, with AMR checking every cycle, and then remaps the source onto the adapted mesh again. `settle_passes` repeats this.

While a settle is pending, outputs are suppressed, AMR runs every cycle (the configured cadence returns after the last pass), and with `time/lat = true` LAT is paused. Each pass runs the pgen's `loaded` and `post` hooks like the first remap. Time, `dt` and the cycle number go back to the source's at each pass, so `nlim` counts from the source's cycle. `settle_steps` works for any pgen on an adaptive mesh; the default is `0`.

### A GR run

A GR source can be remapped by the same block. With `band_mode = auto` (that is `keep`) a widening remap works: the new outer region keeps the torus or disk background the pgen just built, and the remapped state fades in over `b_taper_root_cells` source root cells outside the old box. Details are in [General relativity](#general-relativity).

### A convergence test

`pgen_name = remap_test` is a built-in 3D test problem (hydro or MHD). Run it without a `<remap>` block, with a restart output, to produce a source restart that holds a smooth analytic state. Then remap that restart with `transition_band = false`. It prints a line `remap_test_errors: divb_max= b_l1= b_max= rho_l1= eint_l1= b_far_max=` comparing the result with the analytic state.

## General relativity

The module classifies the source from its parameter dump and, independently, the target from the running code. The two classes must match, otherwise the run exits with a message naming both. The class is printed in the startup banner.

| class (banner name) | source signature | target signature | typical use |
| --- | --- | --- | --- |
| Newtonian | none of the below | no relativistic coordinates | `tde_external`, `remap_test` |
| fixed-GR | `<coord> general_rel = true`, no `<adm>` block | general-relativistic `<coord>`, no `<adm>` | `gr_torus` on a fixed Kerr-Schild metric |
| dynamical-GR | `<adm>` block (needs `<mhd>`) | `<adm>` active | dyn_grmhd runs with a prescribed metric |

- **Refused.** `<z4c>` and `<cce>` sources and `<z4c>` targets are refused permanently: a numerically evolved spacetime cannot be resampled by an interpolation that knows nothing about the constraints, and the result would be constraint-violating initial data. Special relativity is refused too.
- **Consistency checks** compare the source parameter dump with the target input. For fixed-GR, a differing `<coord> a` or `minkowski` is fatal (a key set on one side only counts as differing) and the excision keys (`excise`, `dexcise`, `pexcise`, `flux_excise_r`) only warn. For dynamical-GR everything warns, and the target's value is used. Differences in `<units>` (`bhmass_msun`, `density_cgs`, `mu`) warn in both GR classes.
- **Band mode.** `floor` assumes a Newtonian ambient state and is fatal in GR. `auto` gives `keep`. A Newtonian run can ask for `keep` explicitly, but then the gas energy is interpolated as stored (no thermal-energy carry), so it heats like a plain interpolation of `E`.
- **Carried.** The gas, B (same vector-potential route, also inside horizons) and, with identical settings, `i0`. In the dynamical-GR class the stored ADM metric section of the restart is skipped and recomputed analytically at the source time, so the metric, lapse and excision masks match the restarted state. In-tree GR pgens such as `gr_torus` need no remap-specific code.
- **Radiation.** With `<radiation>` on both sides, `nlevel`, `rotate_geo` and `angular_fluxes` must be identical (a mismatch is fatal). A group present only in the source is dropped with a warning. A group present only in the target keeps the pgen's own initialization.
- **Expect** a short transient in the first cycles of a strongly magnetized run: see [troubleshooting](#troubleshooting).

## What is refused

The run exits with a message for the following. Floors (`dfloor`, `pfloor`, `tfloor`) and `<units>` differences only warn.

### Run configuration

- A remap in the middle of a run while LAT windows are active (`time/lat = true`). The startup remap and the settle passes are not affected, because LAT is idle or paused for them. The check is in the module, so it fires for every pgen.
- A 1D or 2D mesh on either end: both must be 3D, whether or not there is a B field.
- The retired `<problem>` keys `remap`, `remap_restart_source`, `remap_interpolation`, `remap_settle_steps` and `remap_settle_passes`. On a fresh start `tde_external` refuses `problem/remap = true` and prints the `<remap>` block to write instead. Otherwise, and always on a restart, whose own dump may carry the dead keys, they only warn.

### Source and target physics

- A source with `<z4c>`, `<cce>`, `<turbulence>`, `<turb_driving>`, `<sink_particles>` or `<shearing_box>`, and a `<z4c>` target.
- Special-relativistic ends; a relativity-class mismatch; `band_mode = floor` in GR.
- A hydro source into an MHD target or the reverse, and any run carrying both `<hydro>` and `<mhd>`.
- An `isothermal` EOS on either end (there is no energy column to carry).
- A different fluid closure: the gas column count, tabulated versus analytic EOS, and for a gamma-law gas `gamma` to 1e-12. Only the kind of closure is compared, not the table file: use the same table on both ends.
- A different number of passive scalars (`nscalars`), or a source with a dual-energy column into a target with `dual_energy = false`. The reverse is allowed: the target's column is then built from the remapped state.

### Source file

- A multi-level MHD source without `b_coarsen_ok = true`.
- An MHD source whose B is not discretely divergence-free: the inverse-curl residual is a warning above 1e-8 and fatal above 1e-2 of the maximum field strength.
- A truncated file, a per-rank restart, or a payload with an unrecognized trailing section.

## Checklists

### Before the run

- Same physics modules on both ends: hydro or MHD (not both), the same `eos`, `gamma` (or the same table), `nscalars`, `dual_energy` and `<radiation>` settings, and the same relativity class (for fixed-GR also the same `<coord>` spin `a`).
- 3D mesh, absolute `nlim` and `tlim`.
- `<outputN>` blocks of the new deck use the same names as the source's if the numbering should continue.

### After the run starts

- Read the banner printed by rank 0: source path, relativity class, band mode, source time and cycle, blocks loaded, whether B and `i0` were applied. The `radiation i0 remap` line says the group was loaded; only `keep` mode applies it.
- Read the warnings. The B diagnostics show the inverse-curl residual and the maximum field strength in the interior and on the boundary shell.
- Check `mhd_divb`: it should be at round-off level. A same-domain remap has no seam.
- Compare mass, momentum and energy totals before and after.

## Troubleshooting

| symptom | cause and fix |
| --- | --- |
| Fatal error about LAT windows | A problem generator called the remap in the middle of a run while LAT windows were active. Pause LAT first, as the settle steps do (see [Writing a pgen that participates](#writing-a-pgen-that-participates)). |
| Run stops immediately after the remap | `nlim` or `tlim` is absolute. Add the wanted cycles or time to the source's final values. |
| Output numbering does not continue from the source | `copy_output_state = false`, or the new deck has no `<outputN>` block of the same name. |
| Fatal error on a multi-level MHD source | Set `b_coarsen_ok = true` and accept the loss of sub-root B structure. |
| Fatal error on the inverse-curl residual | The source B is not divergence-free (not constrained-transport evolved), so the remapped field would not be the source's. |
| Sheet-current warning | The source B does not vanish at the old boundary, so a widening remap creates taper currents outside the old box. This is physically unavoidable. |
| Divergence reported at the taper edge (`keep` mode) | Two divergence-free fields meet at a seam. The outer taper layer can carry div(B) of order the pgen B over the cell size. A same-domain remap has no seam. |
| Transient in the first cycles (GR, strong B) | Gas is remapped verbatim while B is rebuilt, so E and B are inconsistent at order h squared. C2P (primitive recovery), FOFC (first-order flux correction) and excision absorb it as a one-time transient. |
| Total energy drops | By design: kinetic energy the new grid cannot represent is dropped, not heated. See [Remapping](Remapping#guarantees-and-limits). |

## Writing a pgen that participates

A pgen may enroll hooks inside its pgen function. They run around the automatic remap:

```cpp
user_remap_skip_func   = [](Real x, Real y, Real z) { return inside_excision(...); };
user_remap_loaded_func = [](ParameterInput *src_pin) { /* restore pgen metadata */ };
user_remap_post_func   = [](const remap::RemapSummary &s) {
  /* frame switches, reseeds, banners */ };
```

`skip` marks cells that keep the ambient state (excision). `loaded` runs after the source parameters are read and before the state is applied. `post` runs last.

In `floor` mode the remap overwrites every cell, so a pgen can skip its own initialization when `remap::IsAutoRemapEnabled(pin)` is true, as `tde_external` and `remap_test` do. In `keep` mode the pgen state is the ambient that the remap blends against, so the pgen must still build it. See `src/pgen/tde_external.cpp` (excision, black-hole and frame metadata, settle passes) and `src/pgen/tests/remap_test.cpp` (error report) for the canonical patterns. The programmatic API `remap::LoadAndApplyRemap(...)` also serves mid-run re-remaps. The settle passes are the module's own use of it, so a pgen needs no code for them. Mid-run callers must call `Driver::InitBoundaryValuesAndPrimitives` afterwards. Under `time/lat = true` a mid-run caller must also pause LAT before the call: set `pm->hydro_lat_suspended = true` at a point where every rank is synchronized, and clear it when LAT may resume. While it is set, every step is a plain step of the global time step, and the remap resets the LAT bins and the cycle counters that the AMR and rebalance gates compare with `ncycle`. The settle steps do exactly this.

## Further reading

- [Remapping](Remapping): how the remap works, what is exact and what is not.
- [Implementation notes](Remapping-Implementation-Notes): the stages, known issues and how to test.
- [Dual Energy](Dual-Energy) and [Tabulated EOS](Tabulated-EOS): the closures the remap must match.
- `docs/remap_module_design.md`: the design document.
