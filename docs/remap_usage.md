# Universal restart remap (`src/remap/`) — usage

Resample the state of an existing restart file onto a NEW mesh (different domain,
resolution, MeshBlock layout, or refinement) at the start of a fresh run.  Works with any
pgen, no pgen code needed.  Design: `docs/remap_module_design.md`.

## Quick start

Add a `<remap>` block to the input of the new (target) run:

```
<remap>
enable = true                    # keep the key explicit so the CLI can flip it
source = rst/OldRun.00042.rst    # restart written by the OLD mesh configuration
```

That is all.  On a fresh start (not a restart) the module loads the source restart after
the problem generator runs and overwrites the state:

- **hydro or MHD conserved variables** (+ passive scalars, dual-energy aux) through the
  AMR-hierarchy sampling engine originally built for the TDE remap, with cell subsampling
  and an optional boundary transition band toward the ambient floor where the target
  domain extends beyond the source domain.  A target cell that coincides with a source
  cell (same width, same center) is **copied**, so a same-grid remap is the identity; see
  "Accuracy and conservation".  The gas energy travels as THERMAL energy: the engine
  interpolates `(rho, m, e_int)` and rebuilds `E = e_int + 0.5|m|^2/rho` at the
  destination, so a remap never heats the gas with kinetic energy the target grid cannot
  represent;
- **face-centered B** (MHD sources) **divergence-free to machine precision**: the source
  faces are restricted to the source root grid, inverted exactly to an edge vector
  potential (gauge A1 = 0), interpolated with a C1 Catmull-Rom scheme, and curled on the
  target mesh — including the fine-neighbor edge averaging needed on SMR/AMR targets.
  Total energy is kept consistent (gas energy is remapped; the new magnetic energy is
  added back, weighted by the same blend weight the CC pass used);
- **Radiation** (angular-grid intensities `i0`) when both source
  and target carry the same module (see "GR support" below);
- `pm->time/dt/ncycle` and (by default) the `<outputN>` numbering continue from the
  source run.

## Keys

| key | default | meaning |
|---|---|---|
| enable | true (when the block exists) | master switch; overridable from the CLI |
| source | — | source restart path (required) |
| copy_output_state | true | carry `<outputN>` file_number/last_time forward |
| transition_band | true | reconstruct a smooth taper toward the floor across the old domain boundary (TDE-style).  Set false for pure interpolation (e.g. convergence tests) |
| band_mode | auto | what the transition band does where the source does not cover the target: `floor` = Newtonian floor-fade (the historical TDE behavior), `keep` = blend toward the state the pgen already wrote, `auto` = `keep` in GR, `floor` in Newtonian.  Ignored when `transition_band = false` |
| b_coarsen_ok | false | accept multi-level MHD sources: B is restricted to the source ROOT grid (CC fields keep full detail).  Off by default because fine-level B structure is lost |
| b_taper_root_cells | 4 | smoothstep taper width of A outside the source box (source root cells) |
| b_report | true | print the FC diagnostics (inverse-curl residual, boundary-field warning) |
| radiation_i0 | true | remap the angular-grid intensities `i0` when both sides carry `radiation` |

Consumer-private keys may live in the same block; tde_external reads `settle_steps` /
`settle_passes` (its auto-settle remap passes) from here.

## Accuracy and conservation

**The remap is second-order accurate and is NOT conservative.**  Nothing in the CC engine
integrates over cell overlaps: each target cell is the arithmetic mean of a few point
samples of a trilinear interpolant of the source cell centers, with no cell-volume
weighting and no donor/acceptor intersection volumes.  Total mass, momentum and energy
therefore change by O(h^2) of whatever the source profile resolves — and by O(1) wherever
the source structure is not resolved on the target grid at all, i.e. any coarsening remap
loses exactly what it cannot represent.  Budget for it, and re-measure the conserved
totals after a remap rather than assuming they carried over.

Making it conservative is a different algorithm, not a weighting fix: it needs the exact
intersection volumes of two arbitrary AMR hierarchies (a supermesh), in GR weighted by
`sqrt(gamma)`, and the FC/B path would need the matching flux-conservative restriction to
stay divergence-free.  That is deliberately out of scope; the module transfers a *state*,
not a *budget*.

**Total energy is not conserved, and that is the point.**  The cell-centered engine
carries the source's THERMAL energy — the dual-energy auxiliary where the source has one,
else `max(E - 0.5 m^2/rho, floor_eint)` per source cell, built once into one extra column
of the loaded source (one Real per source cell) — interpolates it with the same sampler as
`rho` and `m`, and rebuilds the target's total energy as `E = e_int + 0.5|m|^2/rho` (plus
the new magnetic energy, MHD).  It does NOT interpolate `E`.  The reason:
`E_interp - ekin(m_interp, rho_interp) = <e_int> + [<ekin> - ekin(<m>, <rho>)]`, and that
bracket is non-negative (Jensen) — it is the grid-scale kinetic-energy variance the target
cell cannot represent.  Interpolate `E` and the remap turns it into heat; carry `e_int` and
it is dropped, which is what an interpolation that cannot represent the fluctuation should
do.  So `E` falls by the unrepresentable kinetic energy;
the thermal state, which is what sets the pressure, the temperature and any light curve
computed from the remapped state, comes through unheated.
Two caveats: a cell that COINCIDES with a source cell is still copied verbatim, `E`
included, so a same-grid remap is unchanged; and `band_mode = keep` (GR's mode, and
Newtonian only if you ask for it explicitly) has no thermal carry — it interpolates the
conserved columns as stored, because in GR they are densitized and C2P/FOFC own the
thermodynamics.

Two things ARE exact:

- **Same-grid cells are copied.**  Where the source cell containing a target cell center
  has the same width and the same center, the sampler takes one sub-sample at that center
  and reproduces the source cell bit-for-bit.  A remap that only changes the MeshBlock
  decomposition or widens the domain is therefore the identity on the overlap, and it
  conserves exactly there.  (Without this detection the two sub-samples per axis would sit at
  `x_c ± h/4` of the interpolant, which on a coincident grid is exactly the filter
  `[1/8, 3/4, 1/8]` — transfer `0.75 + 0.25 cos(kh)`, i.e. 0.5 per axis at the Nyquist
  wavenumber and 0.125 in 3D, erasing 87.5% of the grid-scale gas power
  while the nodal-exact FC/B path keeps all of it.)
- **`div(B) = 0` on every target block**, by construction — see "GR support" for the one
  seam keep mode can introduce.

## GR support

The module handles three **relativity classes**, decided from the SOURCE parameter dump
and, independently, from the TARGET runtime:

| class | source signature | target signature | examples |
|---|---|---|---|
| `kNewtonian` | no `<coord>` relativity keys | no relativistic coordinates | tde_external, remap_test |
| `kFixedGR` | `<coord> general_rel = true`, no `<adm>` | `pcoord->is_general_relativistic` | gr_torus |
| `kDynGRAnalytic` | `<adm>` block present | `pmbp->padm != nullptr` |  |

**The two classes must match**; a mismatch is a clean fatal naming both sides.  Note that
`pcoord->is_general_relativistic` is FALSE for dyn-GR runs, which is why the target test
is on `padm`, not on the coordinate flag.

- **`<z4c>` (or `<cce>`) sources and targets are refused, permanently.** A numerically
  evolved spacetime cannot be resampled by an interpolation that knows nothing about the
  constraints; remapping one would produce constraint-violating initial data that looks
  fine and is wrong.  Special relativity is refused as well.
- **Consistency checks** compare the two parameter dumps and only ever *read* the target
  pin (never `GetOrAdd`, which would pollute its dump).  For `kFixedGR` a differing
  `<coord> a` or `minkowski` is fatal; excision keys only warn.  For `kDynGRAnalytic`
  everything warns.  `<units>` differences
  (`bhmass_msun, density_cgs, mu`) warn in both GR classes.

What is remapped in GR:

- **Gas (`u0`), un-densitized across the interpolation (`kDynGRAnalytic` only).** On the
  dyn-GR path every conserved column is `sqrt(gamma)` times a fluid quantity
  (PrimitiveSolver stores `cons*sdetg`), so interpolating the stored columns directly puts
  the metric inside the smoothing stencil and the recovered `rho` picks up an error
  proportional to the CURVATURE of `sqrt(gamma)` — largest exactly where the punctures
  are.  The module therefore divides `sqrt(gamma)` out at each source cell center, samples
  the undensitized state, and re-multiplies by `sqrt(gamma)` at the target cell center.
  The metric is evaluated at the **source time** (`padm` still holds the pgen's `t = 0`
  metric while the remap runs, and the punctures have since moved), which needs a
  *prescribed* metric.  `kFixedGR` is deliberately untouched: it does not densitize at all
  (`SingleP2C_IdealGRMHD` stores `rho*u^0`; the volume element lives in the flux and
  source terms), so there is nothing to divide out and a `sqrt(gamma)` round trip there
  would be a change of variables the data does not have.  The Newtonian emag round trip
  (subtract the magnetic energy from the source IEN, add the new one back after the FC
  remap) stays Newtonian-only: in GR the O(h^2) E-vs-B inconsistency is absorbed by C2P,
  FOFC and the excision reset.  No Newtonian floors and no dual-energy prep run on the
  GR path.
- **Face-centered B** through the same metric-free A-route as Newtonian, so `div(B) = 0`
  holds to machine precision — including inside horizons.  Horizon cells are deliberately
  NOT skipped: the excision never touches `b0`, and skipping would break the discrete
  curl identity that makes the guarantee exact.  In `keep` mode the FC pass respects the
  same region the CC pass does: faces the remap does not reach (outside the source box
  plus taper) keep the field the target's own pgen built, instead of being driven to zero
  under gas that still carries a magnetized conserved state.  Two divergence-free fields
  meeting face-by-face are not divergence-free at the seam, so the one cell layer at the
  outer edge of the taper can carry `div(B) ~ |B_pgen|/h`; that is measured and reported
  (`max |div(B)|*h` vs `max |B|`, never suppressed by `b_report`).  A same-domain remap
  has no seam and prints nothing.
- **Radiation.** Angular-grid intensities `i0` require identical `radiation/nlevel`,
  `rotate_geo` and `angular_fluxes` on both sides (mismatch is fatal).  A group present in the source but not in
  the target is skipped with a warning; the reverse is reported in the summary and left
  to the pgen.
- **Skipped:** the ADM metric section of the payload.  It is recomputed analytically, so
  the loader identifies it by residual byte count and steps over it. 

**Band mode.**  Newtonian floor-fade thermodynamics is invalid in GR (there is no
"ambient floor state" to relax to), so `band_mode = floor` in a GR class is a fatal
error.  `auto` resolves to `keep`: a target cell with band weight `w` gets
`w * u_sampled + (1-w) * u_pgen`, with `w = 0` leaving the pgen state untouched.  That is
what makes a *widening* GR remap work — the new outer region keeps the torus/disk
background the pgen just built instead of being flooded with floor material.
Unlike the Newtonian band (which begins *inside* the source boundary because the loader
floors the source's outer ghost zones there), the keep fade lives entirely *outside* the
source box: `w = 1` everywhere the source covers, then the clamped boundary value decays
to zero across `b_taper_root_cells` source root cells — the same geometry as the FC
vector-potential taper.  (An interior band would blend t=0 pgen material back into a
fully covered domain.)  The FC pass uses the same
weight (evaluated at the cell center) both to decide which faces it owns and to weight the
Newtonian magnetic-energy add-back: the `(1-w)` fraction of the cell that kept the pgen's
TOTAL energy already carries the pgen's magnetic energy, so only the remapped fraction
takes the new one.

**Zero pgen code for `gr_torus`.**  Add a `<remap>` block to a `gr_torus` input and it
works: the orchestrator refreshes the analytic metric by calling
`padm->SetADMVariables(pmbp)` at the *source* time before the pgen post hook runs, so
dyn-GR pgens see the metric, the lapse and the excision masks of the moment they are
restarting into. 

## Restrictions (clean fatal errors)

- Source must be hydro-XOR-mhd and free of step-3 internal state: no `<z4c>`/`<cce>`,
  `<turb_driving>`, `<sink_particles>`, shearing box.
- Special-relativistic sources and targets are refused; general-relativistic ones are
  supported (see "GR support").
- Source and target relativity classes must match, and so must the CC module type
  (hydro->hydro, mhd->mhd).
- **Fluid EOS must match**: the source's own `<hydro|mhd>/eos` decides its conserved
  column count (only `isothermal` drops the energy column), and it must agree with the
  target's on the column count, on ideal-vs-isothermal, on tabulated-vs-analytic and, for
  a gamma-law gas, on `gamma` to 1e-12.  Floors (`dfloor`, `pfloor`, `tfloor`) only warn.
- 3D on both ends, whether or not there is a B field to remap.
- `band_mode = floor` with a GR class.
- Cannot be combined with `time/lat = true` (remap first, then restart the remapped run
  with LAT).  The check lives in the remap module, so it fires for every pgen.
- `<problem>/remap` and `<problem>/remap_restart_source` are **retired** spellings from
  before the module existed.  On a fresh start `tde_external` refuses to run when
  `problem/remap = true` and prints the `<remap>` block to write instead; otherwise (and
  always on a restart, whose own dump may carry the dead keys) they are ignored with a
  warning.

## Pgen participation (optional)

A pgen may enroll hooks inside its pgen function (they run around the auto remap):

```cpp
user_remap_skip_func   = [](Real x, Real y, Real z) { return inside_excision(...); };
user_remap_loaded_func = [](ParameterInput *src_pin) { /* restore pgen metadata */ };
user_remap_post_func   = [](const remap::RemapSummary &s) {
  /* frame switches, reseeds, banners */ };
```

A pgen that fully initializes its own state should skip that work when
`remap::IsAutoRemapEnabled(pin)` is true (the remap would overwrite it anyway).  See
`src/pgen/tde_external.cpp` (excision + BH/frame metadata + settle passes),
`src/pgen/tests/remap_test.cpp`
(error report) for the canonical patterns.  The programmatic API
`remap::LoadAndApplyRemap(...)` remains available for mid-run re-remaps (TDE settle
passes); mid-run callers must call `Driver::InitBoundaryValuesAndPrimitives` afterwards.
