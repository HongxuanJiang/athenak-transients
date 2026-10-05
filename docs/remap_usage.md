# Remap Usage

This page is the user guide for running a remap. It covers the parameters, the restrictions
and GR support. For the concepts and the algorithm at user level, see [Remapping](wiki/Remapping.md).

## Summary

The remap module resamples the state of an existing restart file onto a **new mesh** (different
domain, resolution, MeshBlock layout or refinement) at the start of a fresh run. It works with
any problem generator and needs no pgen code. Use it to widen or shrink a domain, change the
MeshBlock size for a different rank count, or add and move refinement. Do not use it when you
need exact conservation of mass, momentum and energy, because the transfer is second-order
accurate and not conservative. Developer notes are in
[the implementation notes](wiki/Remapping-Implementation-Notes.md) and the design document is
`docs/remap_module_design.md`.

## Quick start

Add a `<remap>` block to the input of the new (target) run and launch it with `athena -i`
(not `-r`):

```
<remap>
enable = true                    # keep the key explicit so the CLI can flip it
source = rst/OldRun.00042.rst    # restart written by the OLD mesh configuration
```

On a fresh start the module loads the source restart after the problem generator runs and
overwrites the state:

- **Hydro or MHD conserved variables**, with passive scalars and the dual-energy auxiliary.
  Cells are subsampled, with an optional transition band toward the ambient state where the new
  domain extends beyond the old one. A target cell that coincides with a source cell (same width
  and centre) is copied, so a same-grid remap is the identity. The gas energy travels as thermal
  energy, so a remap never heats the gas with kinetic energy the target grid cannot represent.
- **Face-centered B** (MHD sources), divergence-free to machine precision. The source faces
  are restricted to the source root grid, inverted to an edge vector potential, interpolated with
  a C1 Catmull-Rom scheme and curled on the target mesh, including the fine-neighbour edge
  averaging needed on SMR and AMR targets. The new magnetic energy is added back with the same
  blend weight the gas pass used.
- **Radiation** angular-grid intensities `i0`, when both source and target carry the same module.
- **Time, time step and cycle number**, and by default the `<outputN>` numbering, continue from
  the source run. `time/nlim` is therefore an absolute cycle count.

## Keys

| key | type | default | meaning |
| --- | --- | --- | --- |
| `enable` | bool | `true` | Master switch, overridable from the CLI. Has no effect unless the `<remap>` block exists. |
| `source` | string | `""` | Source restart path. Required, an empty value is fatal. |
| `copy_output_state` | bool | true | Carry `<outputN>` `file_number` and `last_time` forward. |
| `transition_band` | bool | true | Reconstruct a smooth taper toward the floor across the old domain boundary (TDE style). Set false for pure interpolation, for example in convergence tests. |
| `band_mode` | string | `auto` | `auto`, `floor` or `keep` (case-insensitive, other values are fatal). What the band does where the source does not cover the target. `floor` is the Newtonian floor-fade (the historical TDE behaviour), `keep` blends toward the state the pgen already wrote, `auto` gives `keep` in GR and `floor` in Newtonian. Ignored when `transition_band = false`. |
| `b_coarsen_ok` | bool | false | Accept multi-level MHD sources. B is restricted to the source root grid (cell-centered fields keep full detail). Off by default because fine-level B structure is lost. |
| `b_taper_root_cells` | int | `4` | Smoothstep taper width of the vector potential outside the source box, in source root cells. |
| `b_report` | bool | true | Print the face-centered B diagnostics (inverse-curl residual, boundary-field warning). The sheet-current warning and a residual of 1e-8 or above are not gated by this key. |
| `radiation_i0` | bool | true | Remap the angular-grid intensities `i0` when both sides carry `radiation`. |

Consumer-private keys may live in the same block. `tde_external` reads `settle_steps` (int, default 20)
and `settle_passes` (int, default 1) for its post-remap settle passes.

## Accuracy and conservation

**The remap is second-order accurate and is not conservative.** Each target cell is the
arithmetic mean of a few point samples of a trilinear interpolant of the source cell centres,
with no cell-volume weighting and no donor/acceptor intersection volumes. Total mass, momentum
and energy change by $O(h^2)$ of whatever the source profile resolves, and by $O(1)$ where the
source structure is not resolved on the target grid, so any coarsening remap loses what it
cannot represent. Re-measure the conserved totals after a remap rather than assuming they
carried over.

Making it conservative is a different algorithm, not a weighting fix. It needs the exact
intersection volumes of two arbitrary AMR hierarchies (a supermesh), weighted by
`sqrt(gamma)` in GR, and the B path would need a matching flux-conservative restriction. This is
out of scope: the module transfers a state, not a budget.

**Total energy is not conserved, by design.** The gas engine carries the source's thermal
energy (the dual-energy auxiliary where the source has one, otherwise
`max(E - 0.5 m^2/rho, floor_eint)` per source cell), interpolates it like `rho` and `m`, and
rebuilds `E = e_int + 0.5|m|^2/rho` (plus the new magnetic energy for MHD). It does not
interpolate `E`, because that would turn the grid-scale kinetic-energy variance the target cell
cannot represent into heat. So `E` falls by the unrepresentable kinetic energy, while the
thermal state, which sets the pressure, the temperature and any light curve, comes through
unheated. Two caveats:

- A cell that coincides with a source cell is copied verbatim, `E` included.
- `band_mode = keep` has no thermal carry. It interpolates the conserved columns as stored,
  because in GR they are densitized and C2P and FOFC own the thermodynamics. This applies to GR,
  and to Newtonian runs only if you ask for `keep` explicitly.

Two things are exact:

- **Same-grid cells are copied.** A remap that only changes the MeshBlock decomposition or widens
  the domain is the identity, and conserves exactly, on the overlap. Without this detection
  the sampler would act as a smoothing filter and erase 87.5% of the grid-scale gas power in 3D.
- **div(B) = 0 on every target block**, by construction (see the seam note under GR support).

## GR support

The module handles three relativity classes, decided from the source parameter dump and,
independently, from the target runtime. The two classes must match, otherwise the run exits with
a message naming both.

| class | source signature | target signature | examples |
| --- | --- | --- | --- |
| `kNewtonian` | no `<coord>` relativity keys | no relativistic coordinates | tde_external, remap_test |
| `kFixedGR` | `<coord> general_rel = true`, no `<adm>` | `pcoord->is_general_relativistic` | gr_torus |
| `kDynGRAnalytic` | `<adm>` block present | `pmbp->padm != nullptr` | |

The target test for dynamical GR is on `padm`, because `pcoord->is_general_relativistic` is
false for dyn-GR runs.

**Refused.** `<z4c>` or `<cce>` sources and targets are refused permanently: a numerically
evolved spacetime cannot be resampled by an interpolation that knows nothing about the
constraints, and the result would be constraint-violating initial data that looks fine and is
wrong. Special relativity is refused too.

**Consistency checks** compare the two parameter dumps and only read the target pin (never
`GetOrAdd`, which would pollute its dump). For `kFixedGR` a differing `<coord> a` or
`minkowski` is fatal and excision keys only warn. For `kDynGRAnalytic` everything warns.
`<units>` differences (`bhmass_msun`, `density_cgs`, `mu`) warn in both GR classes.

**What is remapped in GR**

- **Gas, un-densitized across the interpolation (`kDynGRAnalytic` only).** On the dyn-GR path
  every conserved column is `sqrt(gamma)` times a fluid quantity, so the module divides it out at
  each source cell centre, samples the undensitized state, and multiplies it back at the target
  cell centre. This keeps the metric curvature, largest at the punctures, out of the
  interpolation. The metric is evaluated at the source time, which needs a prescribed metric
  (`padm` still holds the pgen's `t = 0` metric while the remap runs). `kFixedGR` does not
  densitize, so nothing is divided out there. The Newtonian magnetic-energy round trip is
  Newtonian-only, because in GR the $O(h^2)$ E-versus-B inconsistency is absorbed by C2P, FOFC and
  the excision reset. No Newtonian floors and no dual-energy preparation run on the GR path.
- **Face-centered B** through the same metric-free vector-potential route, so div(B) = 0 holds
  to machine precision, including inside horizons. Horizon cells are deliberately not skipped,
  because excision never touches `b0` and skipping would break the discrete curl identity.
  In `keep` mode, faces the remap does not reach (outside the source box plus taper) keep the
  field the target's own pgen built. Two divergence-free fields meeting face to face are not
  divergence-free at the seam, so the outer taper layer can carry div(B) of order `|B_pgen|/h`.
  This is measured and reported (`max|div B|*h` against `max|B|`, never suppressed by
  `b_report`). A same-domain remap has no seam and prints nothing.
- **Radiation.** `i0` requires identical `radiation/nlevel`, `rotate_geo` and `angular_fluxes`
  on both sides (a mismatch is fatal). A group in the source but not the target is skipped with a
  warning. The reverse is reported in the summary and left to the pgen.
- **Skipped.** The ADM metric section of the payload is recomputed analytically, so the loader
  identifies it by residual byte count and steps over it.

**Band mode in GR.** Newtonian floor-fade thermodynamics is invalid in GR (there is no ambient
floor state to relax to), so `band_mode = floor` in a GR class is fatal. `auto` resolves to
`keep`: a target cell with band weight `w` gets `w * u_sampled + (1-w) * u_pgen`, with `w = 0`
leaving the pgen state untouched. This makes a widening GR remap work, because the new outer
region keeps the torus or disk background the pgen just built.

Unlike the Newtonian band, which begins inside the source boundary because the loader floors the
source's outer ghost zones there, the keep fade lives entirely outside the source box:
`w = 1` everywhere the source covers, then the clamped boundary value decays to zero across
`b_taper_root_cells` source root cells, the same geometry as the vector-potential taper. The
face-centered pass uses the same weight, evaluated at the cell centre, to decide which faces it
owns and to weight the Newtonian magnetic-energy add-back. The `(1-w)` fraction that kept the
pgen's total energy already carries the pgen's magnetic energy, so only the remapped fraction
takes the new one.

**No pgen code for `gr_torus`.** Add a `<remap>` block to a `gr_torus` input and it works. The
orchestrator refreshes the analytic metric by calling `padm->SetADMVariables(pmbp)` at the source
time before the pgen post hook runs, so dyn-GR pgens see the metric, the lapse and the excision
masks of the moment they restart into.

## Restrictions (clean fatal errors)

- The source must be hydro or MHD (not both) and free of step-3 internal state: no
  `<z4c>`/`<cce>`, `<turbulence>`/`<turb_driving>`, `<sink_particles>`, shearing box.
- Special-relativistic sources and targets are refused. General-relativistic ones are supported.
- Source and target relativity classes must match, and so must the CC module type (hydro to
  hydro, mhd to mhd).
- **The fluid EOS must match.** The source's own `<hydro|mhd>/eos` decides its conserved column
  count, which must agree with the target's, as must tabulated versus analytic and, for a
  gamma-law gas, `gamma` to 1e-12. Floors (`dfloor`, `pfloor`, `tfloor`) only warn.
- An `isothermal` EOS (no energy column) is refused on either end.
- Both ends must be 3D, whether or not there is a B field to remap.
- `band_mode = floor` with a GR class.
- A multi-level MHD source without `b_coarsen_ok = true`.
- `time/lat = true` cannot be combined with the remap. Remap first, then restart the remapped
  run with LAT. The check lives in the remap module, so it fires for every pgen.
- The `<problem>` keys `remap`, `remap_restart_source`, `remap_interpolation`,
  `remap_settle_steps` and `remap_settle_passes` are retired spellings from before the
  module existed. On a fresh start `tde_external` refuses to run when `problem/remap = true` and
  prints the `<remap>` block to write instead. Otherwise (and always on a restart, whose own dump
  may carry the dead keys) they are ignored with a warning.

## Pgen participation (optional)

A pgen may enroll hooks inside its pgen function. They run around the auto remap:

```cpp
user_remap_skip_func   = [](Real x, Real y, Real z) { return inside_excision(...); };
user_remap_loaded_func = [](ParameterInput *src_pin) { /* restore pgen metadata */ };
user_remap_post_func   = [](const remap::RemapSummary &s) {
  /* frame switches, reseeds, banners */ };
```

A pgen that fully initializes its own state should skip that work when
`remap::IsAutoRemapEnabled(pin)` is true, since the remap would overwrite it anyway. See
`src/pgen/tde_external.cpp` (excision, BH and frame metadata, settle passes) and
`src/pgen/tests/remap_test.cpp` (error report) for the canonical patterns. The programmatic API
`remap::LoadAndApplyRemap(...)` remains available for mid-run re-remaps (TDE settle passes).
Mid-run callers must call `Driver::InitBoundaryValuesAndPrimitives` afterwards.
