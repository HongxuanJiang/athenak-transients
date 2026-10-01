# Remapping: implementation notes

Developer-level detail for [Remapping](Remapping). Start there for usage and parameters. The
full design is in `docs/remap_module_design.md`.

## Contents

- [Cell-centered engine](#cell-centered-engine)
- [Face-centered B](#face-centered-b)
- [Magnetic-energy bookkeeping](#magnetic-energy-bookkeeping)
- [Step-by-step flow](#step-by-step-flow)
- [Interactions](#interactions)
- [Limitations and known issues](#limitations-and-known-issues)
- [Code map](#code-map)
- [Configuration notes](#configuration-notes)

## Cell-centered engine

For each target active cell the engine locates the containing source block, walking
`src.max_level` down through `src.block_map` (`FindContainingSourceBlock` in `remap_cc.cpp`), and
takes a small number of point samples of a trilinear interpolant of the source cell centres,
averaged into the target cell. Two apply passes are dispatched on `RemapBandMode`.

**`kFloorFade` (Newtonian).** Every target cell is overwritten. A boundary transition band
(cubic-Hermite reconstruction in log rho and log p with slope-limited endpoints,
`remap_cc.cpp`) tapers the interior profile toward the ambient hydro floor across the old
domain boundary. Cells fully outside the old box become floor state with a velocity bleed.

**`kKeepTarget` (GR).** The target pgen's own state is the ambient. The weight is `w = 1` for
every sub-sample inside the source active box, with full weight up to the last active cell
(source ghosts are never floored in keep mode). Outside the box the clamped boundary value decays
to 0 by a quintic smoothstep $t^3(t(6t-15)+10)$ over `b_taper_root_cells` source root cells
(`KeepSampleWeight`, `remap_cc.cpp`, shared with the FC pass), the same geometry as the FC
vector-potential taper. `w = 0` writes nothing, `0 < w < 1` blends conserved variables
linearly, and `w = 1` overwrites. The keep band lives entirely outside the source box, because
blending inside a fully covered domain measurably resurrects initial-condition material.

**Coincident cells are copied.** `TargetCellMatchesSourceGrid` tests width match
(`|h - dq| <= 1e-10 dq`) and centre alignment (`|t - round(t)| <= 1e-10`) per axis. When both
hold, the sampler takes one sub-sample at the cell centre, which the trilinear weights reduce
to a bit-exact copy. Without this special case every sampler would take two sub-samples at
`x_c +- h/4`, which on a coincident grid is the separable filter `[1/8, 3/4, 1/8]` (0.125 in 3D at the
Nyquist wavenumber), so a same-grid remap would remove 87.5% of the grid-scale gas power while
the nodal-exact FC/B path removes none.

**Thermal energy, not total energy** (Newtonian CC path; `BuildSourceThermalEnergy` and
`sample_full_state`, `remap_cc.cpp`). The engine interpolates `(rho, m_i, e_int)` and rebuilds
the target's total energy as `E = e_int + 0.5 |m|^2 / max(rho, dfloor)`. The reason is an
inequality: the interpolated `E` minus `ekin(m_interp, rho_interp)` equals the mean `e_int`
plus a bracket `<ekin> - ekin(<m>, <rho>)` that is non-negative by Jensen's inequality. That
bracket is the grid-scale kinetic-energy variance the target cell cannot represent, and
interpolating `E` hands exactly that variance to the thermal channel as heat. This matters when
`e_int/KE` is small, as in the TDE debris.

- The source's thermal energy is its dual-energy auxiliary where it carries one (the channel the
  source run trusted in these cold cells), otherwise `max(E - 0.5 m^2/rho, floor_eint)` per
  source cell.
- It is built once into one extra host column of the source (one `Real` per source cell, about
  1/`nvars` of the gas payload) and travels through the same trilinear sampler and sub-sample
  averaging as the conserved columns.
- `IEN` is already the gas energy at that point. The loader subtracted the source's magnetic
  energy and the FC pass adds the new one back.
- The transition band and the ambient inward shift use the same carried value rather than
  re-deriving `E - ekin`.
- There is no `E = max(E_interp, ekin + e_int)` clamp, which would inject total energy into
  every cell whose energy channel sits below the auxiliary.
- A coincident cell stays a verbatim copy. Its `E` is the source's own, not a rebuilt one
  (rebuilding reproduces it only to a rounding, and for a dual-energy source not even that).
- `kKeepTarget` is untouched (see R-9).

**Dynamical-GR un-densitization** (`RemapMetricSampler` in `remap_impl.hpp` and
`remap_load.cpp`). On the `<adm>` path every conserved column is `sqrt(gamma)` times a fluid
quantity, so interpolating the stored column would put the metric's curvature inside the
smoothing stencil (worst at the punctures). The loader evaluates `sqrt(det gamma_ij)` at every
source cell centre (`ADMMetricView::CartesianMetric`, the analytic metric at the source time),
the samplers divide by it at each interpolation node, and `SampleKeepCellAverage`
re-multiplies by `sqrt(gamma)` at the target cell centre. `kFixedGR` is deliberately left alone,
since no densitization exists there (`SingleP2C_IdealGRMHD` stores `rho*u^0` with no volume
element to divide out). A stored (non-analytic) ADM backend cannot be evaluated at a source cell
centre and falls back to interpolating as stored, with a rank-0 warning.

## Face-centered B

(`remap_fc.cpp`, `docs/remap_module_design.md` Section 5.)

1. Source B faces are exactly restricted to a uniform covering grid at the source root level
   (area-weighted mean of the 2^dl by 2^dl fine faces; block-boundary faces get 0.5 weight per
   side so mixed interior and interface coverage still area-averages correctly).
2. The discrete curl is inverted exactly in the gauge $A_1 \equiv 0$ by prefix sums. First
   $A_3(k,j{+}1,0) = A_3(k,j,0) + dy\,B_1(k,j,0)$ seeds the redundant component along the $i=0$
   plane. Then sweeping $i$: $A_2(\cdot,i{+}1) = A_2(\cdot,i) + dx\,B_3(\cdot,i)$ and
   $A_3(\cdot,i{+}1) = A_3(\cdot,i) - dx\,B_2(\cdot,i)$.
3. By construction the curl of A reproduces $B_2$ and $B_3$ exactly, and $B_1$ exactly if and
   only if the source was discretely divergence-free. The residual is measured and gated: a
   warning above 1e-8 and **fatal above 1e-2**, relative to the maximum of B.
4. Each component is interpolated off its own edge lattice by a separable Catmull-Rom cubic (C1
   continuity gives second-order-accurate B after differencing), gauge-normalized by
   subtracting each component's boundary-shell mean, and tapered to exactly zero over
   `b_taper_root_cells` root cells outside the source box by a product of quintic smoothsteps
   (`SmoothStep5`).
5. The target curl is taken per MeshBlock, with fine-neighbour edge averaging where an edge
   abuts a finer target neighbour (mirrors `gr_torus.cpp`, so shared fine/coarse fluxes stay
   restriction-consistent).

Because the target field is everywhere the discrete curl of edge samples of one global,
continuous potential, every target MeshBlock is divergence-free to machine precision,
independent of interpolation error. This includes excised horizon interiors: excision never
touches `b0`, and skipping it would break the discrete curl identity
(`docs/remap_module_design.md` Section 8.4).

**Keep mode.** The FC pass starts from the pgen's own `b0`/`bcc0` and overwrites only faces whose
curl stencil touches an edge where the taper window is still nonzero. The test is exact on the
window, not on `A`, since `A` can vanish for unrelated reasons. Elsewhere the pgen field
stands, because zeroing B everywhere outside the source and taper would leave the gas with its
pgen magnetized conserved state and the decode would read EM energy back as fluid. Two
divergence-free fields meeting face to face are not divergence-free at the seam. The one
taper-edge cell layer can carry div(B) of order `|B_pgen|/h`, which is measured (MPI-reduced
`max|div(B)|*h` against `max|B|`) and reported outside `b_report`. A same-domain remap has no seam.

## Magnetic-energy bookkeeping

Newtonian only (`!src.gr_mode`). The loader subtracts `emag = 0.5*(bx^2+by^2+bz^2)`
(face-averaged) from the source's `IEN` column at load time, so the CC engine remaps gas
energy. The FC pass adds the new magnetic energy back after building B:

```
u0(IEN) += 0.5*|bcc0_new|^2 - (1-w)*emag_old
```

with `w` the same `KeepSampleWeight` the CC pass used (factored out so the two passes cannot
drift apart). In floor-fade mode `w = 1` and `emag_old = 0`, so the add-back is `0.5*|B_new|^2`.
GR never densitizes or re-adds magnetic energy this way (`add_emag = !src.gr_mode`): the
conserved column there is a smooth analytic-metric function remapped verbatim, and the
O(h^2) E-versus-B inconsistency is absorbed by C2P, FOFC and excision.

## Step-by-step flow

1. **Fresh start only.** The fresh-start constructor of `ProblemGenerator` calls
   `CallProblemGenerator(pin, false)` and then `remap::MaybeAutoRemap(this, pin, pm)`
   (`src/pgen/pgen.cpp`). The restart constructor never calls `MaybeAutoRemap`, so a remap
   target is always launched with `athena -i <deck>`, never `athena -r`. A pgen participates by
   enrolling `user_remap_skip_func`, `_loaded_func` and `_post_func` inside its pgen function
   before the auto-remap call, and by skipping its own full initialization when
   `remap::IsAutoRemapEnabled(pin)` is true. The pgen function still runs, since keep-mode
   banding blends against exactly that state, and floor-fade overwrites it.
2. `IsAutoRemapEnabled` is a no-op unless the input has a `<remap>` block. `OptionsFromInput`
   reads the nine keys of the parameter table.
3. `LoadAndApplyRemap` refuses `time/lat = true` (`IsLATEnabled()`) and an empty `source`
   path. `LoadRemapSourceData` then parses the source restart's ASCII parameter dump and
   binary payload.
4. Band mode is resolved (`auto` to class-dependent) and `CheckRemapGRConsistency` checks the two
   parameter dumps (GR only; never mutates the target pin).
5. Newtonian-only floor prep: outer source ghost zones are floored
   (`FloorOuterSourceGhostZones`) and dual-energy `needs_init` is armed.
6. `on_loaded(&src_pin)` (the pgen's `user_remap_loaded_func`) runs before the state is applied,
   so a pgen can restore metadata (for example the TDE black hole and frame record) that its
   own `skip_cell` hook needs during the apply.
7. `ApplyRemapCC` (host-side, mirror views and host loops, one `deep_copy` back) transfers gas
   (plus `i0` as enabled). If the source is MHD, `BuildCoveringPotential` and `ApplyRemapFC`
   transfer B.
8. `pm->time`, `dt`, `dtold` and `ncycle` are installed from the source. Production decks add
   the intended post-remap cycle count to the source's final `ncycle`, with a comment.
9. If `pmbp->padm != nullptr` and it has a `SetADMVariables` callback, it is invoked now (metric,
   lapse, excision masks). This is what makes dynamical-GR pgens such as `gr_torus` work with
   no pgen code.
10. `pm->time` and `<outputN>` bookkeeping is written into `dst_pin`, and the rank-0 banner prints
    (source path, relativity class, band mode, source time and cycle, blocks loaded, gas group,
    whether FC and `i0` were applied and why not when skipped, any skipped `<adm>` residual).
11. `pgen->user_remap_post_func(summary)` runs last (frame switches, banners, the error report of `remap_test`).

**Restart behaviour.** The module never runs during a restart. A remap followed by LAT is a
two-step chain: remap with `time/lat = false`, let it write a restart, then
`athena -r <that restart> time/lat=true` (with an updated `nlim` if needed). `LoadAndApplyRemap`
is also usable programmatically for mid-run re-remaps (the TDE settle passes). Such callers must
call `Driver::InitBoundaryValuesAndPrimitives(pm)` afterwards, since the module performs no
boundary communication, prolongation, physical BCs or C2P.

**MPI and GPU.** The whole module is host-only (mirror views, host loops, one `deep_copy` back to
device per group). CC data is loaded per rank, restricted to blocks overlapping the local
pack's bounding box plus 2 root cells (`LocalPackBounds`, `ExpandRegionSize`). FC data is loaded
for all source blocks on every rank, because the covering-grid vector-potential construction is
global and independently replicated. The result is deterministic and rank-count invariant, and
bitwise identical for 1 and 4 ranks in double precision. Every diagnostic sum or print is
MPI-reduced or rank-0 gated.

## Interactions

- **LAT.** Refused for every pgen: `LoadAndApplyRemap` calls `FatalRemap` if
  `dst_pin->IsLATEnabled()`. The supported flow is remap, then restart with `time/lat = true`.
- **AMR and SMR.** Fully compatible on the target side. Both band modes sample correctly through
  the level walk of `FindContainingSourceBlock`, and the fine-neighbour edge averaging keeps
  target fine/coarse interfaces restriction-consistent (`mhd_divb` stays at the 1e-11 level). A
  multi-level MHD source is refused unless `b_coarsen_ok = true`, because B is only restricted
  to the source root grid (R-1). CC fields keep full detail regardless.
- **FOFC, C2P and excision.** Not called by the module. In GR the O(h^2) E-versus-B
  inconsistency it leaves is absorbed downstream by C2P, FOFC and the per-stage excision reset (R-5).
- **Units.** `bhmass_msun`, `density_cgs` and `mu` are compared between source and target and
  only warn on mismatch, in both GR classes.
- **Refused in the source.** `<z4c>` and `<cce>` (permanently, since a numerically evolved
  spacetime cannot be resampled by an interpolant blind to the Hamiltonian and momentum
  constraints), `<turbulence>` and `<turb_driving>` (RNG step-3 state), `<sink_particles>` and
  `<shearing_box>`. None of these carry restart length markers this module can skip past.
- **`<adm>` metric.** Never remapped. It is skipped by residual byte count and recomputed
  analytically by the target pgen (R-7).

## Limitations and known issues

- **Not conservative** (`remap_cc.cpp`). The CC accumulator is a plain mean of point samples of
  a trilinear interpolant, with no cell-volume weighting and no donor/acceptor intersection
  volumes. Totals drift by O(h^2) of whatever the source resolves, and by O(1) where the source
  structure is unresolved on the target grid. A conservative version would need an
  exact-intersection supermesh algorithm (weighted by sqrt(gamma) in GR) plus a matching
  flux-conservative FC restriction.
- **Total energy is not conserved, by construction** (Newtonian CC path). The thermal-energy carry
  drops the grid-scale kinetic-energy variance that a coarser target cannot represent, so `E`
  falls by that amount. Interpolating `E` would convert the variance into internal energy, which
  is worse, because a state that merely lacks unresolved kinetic energy is still the right
  thermal state while a heated one is not.
- **R-1.** A multi-level MHD source loses sub-root B structure (opt-in via `b_coarsen_ok`, warned).
- **R-2.** Target fine/coarse FC exactness relies on the canonical fine-edge-averaging contract
  plus init prolongation. `mhd_divb` stays at the 1e-11 level in the remap tests.
- **R-3.** A source whose B threads the old domain boundary gets taper sheet currents outside the
  old box (the 1e-2 relative sheet-current warning). This is physically unavoidable for a
  domain-widening remap.
- **R-4.** FC data is read for all blocks on every rank (global covering grid). It is a one-shot,
  page-cached I/O cost, not yet revisited.
- **R-5 (GR).** Gas is remapped verbatim while B is rebuilt from the interpolated potential, so
  remapped E and B are O(h^2)-inconsistent. A strongly magnetized, poorly resolved region pays
  for it as a one-time floor and FOFC transient in the first cycles.
- **R-6 (keep mode).** The FC pass respects the same region the CC pass does, so the cost is the
  divergence mismatch at the taper edge, which is measured and reported.
- **R-7.** The `<adm>` payload is identified purely by residual byte count. A future ADM backend
  writing a nonzero, non-cell-shaped record would trip the "unexpected payload residual" fatal
  error rather than be silently mis-parsed. The loader needs revisiting if such a backend appears.
- **R-8 (FC keep-mode seam).** An exactly divergence-free keep-mode seam would need the pgen's own
  vector potential (an inverse curl on the target hierarchy). This is not implemented.
- **R-9 (Newtonian `band_mode = keep`).** The thermal-energy carry lives in the floor-fade worker
  only. `kKeepTarget` is shared with GR, where the conserved columns are densitized and C2P and
  FOFC own the thermodynamics, so it still interpolates the stored `E`. A Newtonian keep-mode
  remap therefore shows the Jensen heating described above. `auto` resolves to `floor` for
  Newtonian, so reaching it needs an explicit `band_mode = keep`.
- **3D only** on both ends, whether or not there is a B field. A 2D source would be an extrusion,
  not a remap, and the CC sampler's unconditional `(ngh+k, ngh+j, ngh+i)` addressing would walk a
  full ghost stride into the next variable's slice on a `nout3 == 1` source.
- **Fluid EOS must match** (column count, tabulated versus analytic, and `gamma` to 1e-12 for a
  gamma-law gas). Floors (`dfloor`, `pfloor`, `tfloor`) only warn. An `isothermal` EOS is refused
  on either end.
- **`band_mode = floor` with a GR class is fatal.** Newtonian floor-fade thermodynamics (ambient
  `rho` and `p` floors, `E = eint + 0.5 rho v^2`) has no meaning for densitized GR conserved
  variables.
- **Retired `<problem>` keys.** `<problem>/remap`, `remap_restart_source`, `remap_interpolation`,
  `remap_settle_steps` and `remap_settle_passes` are pre-module `tde_external` spellings. On a
  fresh start `tde_external` refuses to run when `problem/remap = true` and prints the `<remap>`
  block to write instead. Otherwise, and always on a restart (whose dump may carry the dead keys
  from the run that wrote it), they only warn.

## Code map

| file | role |
| --- | --- |
| `src/remap/remap.hpp` | Public API: `RemapOptions`, `RemapSummary`, `LoadAndApplyRemap`, auto mode (`IsAutoRemapEnabled`, `OptionsFromInput`, `MaybeAutoRemap`) |
| `src/remap/remap.cpp` | Orchestration: LAT refusal, band-mode resolution, GR consistency check, dual-energy prep, CC and FC dispatch, time/dt/ncycle install, ADM metric refresh, `<outputN>` carry-forward, rank-0 banner |
| `src/remap/remap_impl.hpp` | Internal structs: `RemapSourceData`, `RemapSourceBlock`, `RemapCoveringField`, `RemapMetricSampler`, `RemapRelClass`, `KeepSampleWeight` declaration |
| `src/remap/remap_load.cpp` | Restart parsing: physics whitelist, relativity classification, EOS, width and dimensionality checks, payload offsets, per-block reads, FC covering-grid restriction, `CheckRemapGRConsistency` |
| `src/remap/remap_cc.cpp` | CC engine: both band-mode workers, samplers, source thermal-energy column, transition-band math, ghost flooring |
| `src/remap/remap_fc.cpp` | FC engine: inverse curl, gauge normalization, Catmull-Rom potential sampler, target edge evaluation and curl, seam and residual diagnostics |
| `src/pgen/tests/remap_test.cpp` | Built-in analytic-source test pgen, prints the `remap_test_errors:` line |
| `src/pgen/pgen.cpp` | `CallProblemGenerator(pin, false)` then `remap::MaybeAutoRemap(this, pin, pm)`, in the fresh-start constructor only |
| `src/pgen/pgen.hpp` | `user_remap_skip_func`, `user_remap_loaded_func`, `user_remap_post_func` declarations |
| `docs/remap_module_design.md` | Authoritative design |
| `docs/remap_usage.md` | User-facing usage guide |

## Configuration notes

The module's own keys are read with `GetOrAdd*` on `pin` in `remap.cpp`. The loader
(`remap_load.cpp`) reads the source pin and the consumer-private keys through read-only
accessors, since consumer-private reads must never `GetOrAdd` (to avoid polluting the source
pin's dump). `tde_external` gates its private keys on `pin->DoesBlockExist("remap")` so a
consumer never creates the block itself. `"remap"` is the final entry of the `ParameterInput`
block whitelist (`src/parameter_input.cpp`). The keys were found with
`grep -n "GetOrAdd\|GetReal\|GetInteger\|GetBoolean\|GetString\|DoesParameterExist" src/remap/remap.cpp src/remap/remap_load.cpp`.
