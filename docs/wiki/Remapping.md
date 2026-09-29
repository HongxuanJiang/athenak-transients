# Remapping

## Summary

The remap module (`src/remap/`) resamples the state of an existing AthenaK restart file
onto a *new* mesh — different domain, resolution, MeshBlock decomposition, refinement
pattern, or rank count — at the start of a fresh run, with zero pgen code required. It
carries cell-centered hydro/MHD conserved variables (+ passive scalars, dual-energy aux),
face-centered B (rebuilt **divergence-free to machine precision** via a vector-potential
route, not copied) between the two meshes, and continues `pm->time/dt
/ncycle` from the source. It underlies the TDE
external-restart remap, which is a thin wrapper around this module.

Use it whenever a run must continue on a mesh the checkpoint wasn't written for — widen or
shrink the domain, change MeshBlock size for a different rank count, add/move refinement. Cost: one host-side, single-threaded pass at
startup (no GPU, no task-list integration); face data is read for *every* source block on
*every* rank regardless of overlap (R-4 below), which is a one-shot, page-cached I/O cost,
not a per-cycle one. The transferred state is **not conservative** — see "Limitations".

## Physics and algorithm

**Cell-centered (CC) engine**: for each target active cell, locate the
containing source block (walking `src.max_level` down through `src.block_map`,
`FindContainingSourceBlock` in `remap_cc.cpp`) and take a small number of point
samples of a trilinear interpolant of the source cell centers, averaged into the target
cell. Two apply passes are dispatched on `RemapBandMode`:

- `kFloorFade` (Newtonian): every target cell is overwritten; a boundary transition band
  (cubic-Hermite reconstruction in `log rho`, `log p` with slope-limited endpoints,
  `remap_cc.cpp`) tapers the interior profile toward the ambient hydro floor
  across the old domain boundary, and cells fully outside the old box become floor state
  with a velocity bleed.
- `kKeepTarget` (GR): the target pgen's own state *is* the ambient. `w = 1` for every
  sub-sample inside the source active box (full weight, including up to the last active
  cell — source ghosts are never floored in keep mode); outside the box the clamped
  boundary value decays to 0 by a quintic smoothstep, `t^3(t(6t-15)+10)`, over
  `b_taper_root_cells` **source root cells** (`KeepSampleWeight`, `remap_cc.cpp`,
  shared with the FC pass) — the *same* geometry as the FC vector-potential taper. `w = 0`
  writes nothing, `0 < w < 1` blends conserved variables linearly, `w = 1` overwrites. The
  keep band lives **entirely outside** the source box: blending *inside* a fully covered
  domain measurably resurrects initial-condition material.

A **coincident target cell is copied**, not resampled: `TargetCellMatchesSourceGrid` tests width match (`|h - dq| <= 1e-10 dq`) and center alignment
(`|t - round(t)| <= 1e-10`) per axis; when both hold, the sampler takes one sub-sample at
the cell center, which the trilinear weights reduce to a bit-exact copy. Without this special case every sampler would take two
sub-samples at `x_c ± h/4`, which on a coincident grid is exactly the separable filter
`[1/8, 3/4, 1/8]` (0.125 in 3D at the Nyquist wavenumber), so a same-grid remap would
remove 87.5% of the grid-scale gas power while the nodal-exact FC/B path removes none.

**Thermal energy, not total energy** (Newtonian CC path; `BuildSourceThermalEnergy` and
`sample_full_state`, `remap_cc.cpp`). The gas engine interpolates `(rho, m_i, e_int)` and
rebuilds the target's total energy from what it interpolated,
`E = e_int + 0.5 |m|^2 / max(rho, dfloor)`, instead of interpolating `E` itself. The
reason is an inequality: `E_interp - ekin(m_interp, rho_interp) = <e_int> + [<ekin> -
ekin(<m>, <rho>)]`, and the bracket is non-negative by Jensen. It is the grid-scale
kinetic-energy *variance* the target cell cannot represent, and interpolating `E` hands
exactly that unrepresentable variance to the thermal channel as heat, which matters when
`e_int/KE` is small, as in the TDE debris. The source's thermal energy is its dual-energy
auxiliary where it carries one (the channel the source run itself trusted in exactly these
cold cells), else `max(E - 0.5 m^2/rho, floor_eint)` per source cell; it is built once into
one extra host column of the source (one `Real` per source cell, ~1/`nvars` of the gas
payload) and travels through the same trilinear sampler and sub-sample averaging as the
conserved columns. `IEN` is already the *gas* energy at that point — the loader subtracted
the source's magnetic energy and the FC pass adds the new one back (see "Magnetic-energy
bookkeeping"), a contract that the thermal-energy path leaves alone. The transition band and the ambient
inward shift consume the same carried value rather than re-deriving `E - ekin`. There is no
`E = max(E_interp, ekin + e_int)` clamp, which would inject total energy
into every cell whose energy channel sits below the auxiliary. A coincident cell stays a **verbatim copy**: its `E` is the source's own,
not a rebuilt one (rebuilding reproduces it only to a rounding, and for a dual-energy
source not even that). `kKeepTarget` is untouched (R-9).

**Dyn-GR un-densitization** (`RemapMetricSampler` in `remap_impl.hpp`/
`remap_load.cpp`): on the `<adm>` path every conserved column is `sqrt(gamma)` times a
fluid quantity. Interpolating the stored column directly would put the metric's curvature
inside the smoothing stencil (worst at the punctures). The loader instead evaluates
`sqrt(det gamma_ij)` at every source cell center (`ADMMetricView::CartesianMetric`, the
analytic metric **at the source time**), the samplers divide by it at each interpolation node, and
`SampleKeepCellAverage` re-multiplies by `sqrt(gamma)` at the *target* cell center. `kFixedGR` is deliberately left alone (no densitization
exists there: `SingleP2C_IdealGRMHD` stores `rho*u^0` with no volume element to divide
out); a *stored* (non-analytic) ADM backend cannot be evaluated at a source cell center
and falls back to interpolating as stored, with a rank-0 warning.

**Face-centered (FC) B, divergence-free by construction** (`remap_fc.cpp`,
`docs/remap_module_design.md` Section 5): source B faces are exactly restricted to a uniform
covering grid at the source **root** level (area-weighted mean of the 2^dl×2^dl fine
faces; block-boundary faces get 0.5× weight per side so mixed interior/interface coverage
still area-averages correctly). The discrete curl is then inverted exactly
in the gauge $A_1 \equiv 0$ by prefix sums:
$A_3(k,j{+}1,0) = A_3(k,j,0) + dy\,B_1(k,j,0)$ (seeds the redundant component along the
$i=0$ plane), then sweeping $i$: $A_2(\cdot,i{+}1) = A_2(\cdot,i) + dx\,B_3(\cdot,i)$,
$A_3(\cdot,i{+}1) = A_3(\cdot,i) - dx\,B_2(\cdot,i)$. By
construction $\mathrm{curl}(A)$ reproduces $B_2$, $B_3$ exactly, and $B_1$ exactly *iff*
the source was discretely divergence-free — the residual is measured and gated
(`remap_fc.cpp`: warning above `1e-8`, **fatal above `1e-2`** relative to
`max|B|`). Each component is then interpolated off its own edge lattice by a separable
Catmull-Rom cubic (C1 continuity ⇒ 2nd-order-accurate B after differencing,
`remap_fc.cpp`), gauge-normalized by subtracting each component's boundary-shell
mean, and tapered to exactly zero over `b_taper_root_cells` root cells outside
the source box by a product of quintic smoothsteps (`SmoothStep5`, `remap_fc.cpp`). The target curl is taken per MeshBlock, with fine-neighbor edge averaging where
an edge abuts a finer target neighbor (mirrors `gr_torus.cpp`, so shared
fine/coarse fluxes stay restriction-consistent, `remap_fc.cpp`). Because the
target field is everywhere the discrete curl of edge samples of one global, continuous
potential, **every target MeshBlock is divergence-free to machine precision**, independent
of interpolation error — including inside excised horizon interiors (excision never
touches `b0`; skipping it would break the discrete curl identity, `docs/
remap_module_design.md` Section 8.4).

In `kKeepTarget` mode the FC pass starts from the pgen's own `b0`/`bcc0` and overwrites
only faces whose curl stencil touches an edge where the taper window is still nonzero
(exact test on the *window*, not on `A`, since `A` can vanish for unrelated reasons —
`remap_fc.cpp`); elsewhere the pgen field stands (zeroing B
everywhere outside the source+taper would leave the gas with its pgen magnetized conserved state, so
the decode would read EM energy back as fluid). Two divergence-free fields meeting face-to-face
are not divergence-free at the seam — the one taper-edge cell layer can carry
`div(B) ~ |B_pgen|/h` — measured (MPI-reduced `max|div(B)|·h` vs `max|B|`) and reported
outside `b_report`; a same-domain remap has no seam.

**Magnetic-energy bookkeeping** (Newtonian only, `!src.gr_mode`): the loader subtracts
`emag = 0.5*(bx²+by²+bz²)` (face-averaged) from the source's `IEN` column at load time so the CC engine remaps *gas* energy, and the FC pass adds the
new `0.5*|bcc0_new|²` back after building B:
`u0(IEN) += 0.5*|B_new|^2 - (1-w)*emag_old`, with `w` the *same* `KeepSampleWeight` the CC
pass used (factored out so the two passes cannot drift apart). In
floor-fade mode `w ≡ 1` and `emag_old ≡ 0`, so the add-back is `0.5*|B_new|^2`. GR never densitizes/re-adds magnetic energy this way (`add_emag = !src.gr_mode`,
`remap_fc.cpp`): the conserved column there is a smooth analytic-metric function
remapped verbatim, and the O(h²) E-vs-B inconsistency is absorbed by C2P/FOFC/excision.

## Code map

| File | Role |
|---|---|
| `src/remap/remap.hpp` | Public API: `RemapOptions`, `RemapSummary`, `LoadAndApplyRemap`, `<remap>`-block auto mode (`IsAutoRemapEnabled`, `OptionsFromInput`, `MaybeAutoRemap`) |
| `src/remap/remap.cpp` | Orchestration: LAT refusal, band-mode resolution, GR consistency check, dual-energy prep, CC + FC dispatch, `pm->time/dt/ncycle` install, ADM metric refresh, `N`-seeding, `<outputN>` carry-forward, rank-0 banner |
| `src/remap/remap_impl.hpp` | Internal shared structs: `RemapSourceData`, `RemapSourceBlock`, `RemapCoveringField`, `RemapMetricSampler`, `RemapRelClass`, `KeepSampleWeight` declaration |
| `src/remap/remap_load.cpp` | Restart parsing: physics whitelist, relativity classification, EOS/width/dimensionality checks, payload offsets, per-block reads, FC covering-grid restriction, `CheckRemapGRConsistency` |
| `src/remap/remap_cc.cpp` | CC engine: both band-mode workers, samplers, source thermal-energy column, transition-band math, ghost flooring |
| `src/remap/remap_fc.cpp` | FC engine: inverse curl, gauge normalization, Catmull-Rom potential sampler, target edge evaluation + curl, seam/residual diagnostics |
| `src/pgen/tests/remap_test.cpp` | Built-in analytic-source test pgen; prints the `remap_test_errors:` line |
| `src/pgen/pgen.cpp` | `CallProblemGenerator(pin, false)` then `remap::MaybeAutoRemap(this, pin, pm)` in the **fresh-start** constructor only |
| `src/pgen/pgen.hpp` | `user_remap_skip_func` / `user_remap_loaded_func` / `user_remap_post_func` hook declarations |
| `docs/remap_module_design.md` | Authoritative design |
| `docs/remap_usage.md` | User-facing usage guide |

## Configuration

All keys are read from the `<remap>` block via read-only accessors (`remap_load.cpp`
document that consumer-private reads must never `GetOrAdd`, to avoid polluting the source
pin's dump). The **module's own** keys are all `GetOrAdd*` on `pin` in `remap.cpp`, grep
command: `grep -n "GetOrAdd\|GetReal\|GetInteger\|GetBoolean\|GetString\|DoesParameterExist" src/remap/remap.cpp src/remap/remap_load.cpp`.

| block/key | type | default | meaning | read at |
|---|---|---|---|---|
| `remap`/`enable` | bool | `true` (only evaluated if the block exists) | master switch, overridable from the CLI | `remap.cpp` |
| `remap`/`source` | string | `""` | source restart path; empty is fatal once remap runs | `remap.cpp` |
| `remap`/`transition_band` | bool | `true` | reconstruct the boundary taper; `false` = pure interpolation (e.g. convergence tests) | `remap.cpp` |
| `remap`/`band_mode` | string | `"auto"` | `auto` &#124; `floor` &#124; `keep` (case-insensitive); `auto` → `keep` for GR classes, `floor` for Newtonian; unknown value is fatal | `remap.cpp` |
| `remap`/`b_coarsen_ok` | bool | `false` | accept a multi-level MHD source (B restricted to source root grid; CC fields keep full detail) | `remap.cpp` |
| `remap`/`b_taper_root_cells` | int | `4` | smoothstep taper width of `A` outside the source box, in source root cells | `remap.cpp` |
| `remap`/`b_report` | bool | `true` | print the FC inverse-curl / boundary-field diagnostics (the sheet-current warning and the ≥1e-8 residual warning are **not** gated by this) | `remap.cpp` |
| `remap`/`radiation_i0` | bool | `true` | remap angular intensities `i0` if both sides carry `<radiation>` | `remap.cpp` |
| `remap`/`copy_output_state` | bool | `true` | carry `<outputN>` `file_number`/`last_time` forward into the target's own dump (block-name matched) | `remap.cpp`, applied in `CopyRemapOutputState`, `remap.cpp` |

**Consumer-private keys** live in the same `<remap>` block and are read only by the pgen
that owns them, never by the module: `tde_external.cpp` reads `settle_steps` (int, default
20, `tde_external.cpp`) and `settle_passes` (int, default 1, `tde_external.cpp`)
for its post-remap settle-pass scheduling, gated on `pin->DoesBlockExist("remap")` so a consumer never *creates* the block itself. `"remap"` is the
final entry of `ParameterInput`'s block whitelist (`src/parameter_input.cpp`).

## How it runs

1. **Fresh start only.** `ProblemGenerator`'s fresh-start constructor calls
   `CallProblemGenerator(pin, false)` then, immediately after,
   `remap::MaybeAutoRemap(this, pin, pm)` (`src/pgen/pgen.cpp`). The **restart**
   constructor (`ProblemGenerator(pin, pm, resfile, ...)`, `pgen.cpp`, which calls
   `CallProblemGenerator(pin, true)` at `pgen.cpp`) never calls `MaybeAutoRemap` — a
   remap target is always launched with `athena -i <deck>`, never `athena -r`. A pgen
   participates by (a) enrolling `user_remap_skip_func` / `_loaded_func` / `_post_func`
   inside its pgen function *before* the auto-remap call, and (b) skipping its own
   full initialization when `remap::IsAutoRemapEnabled(pin)` is true (its state would be
   overwritten by `kFloorFade`, or read as the ambient by `kKeepTarget` — either way the
   pgen function still runs, since keep-mode banding blends against exactly that state).
2. `IsAutoRemapEnabled` is a no-op unless the input has a `<remap>`
   block; `OptionsFromInput` reads the 9 keys above.
3. `LoadAndApplyRemap`: refuses `time/lat = true`
   (`IsLATEnabled()`, `remap.cpp`), refuses an empty `source` path, then
   `LoadRemapSourceData` parses the source restart's ASCII parameter dump and binary
   payload (see "Interactions" and "Limitations" for the accepted formats and refusals).
4. Band mode is resolved (`auto` → class-dependent) and `CheckRemapGRConsistency` checks
   the two parameter dumps (GR only; never mutates the target pin).
5. Newtonian-only floor prep: outer source ghost zones are floored
   (`FloorOuterSourceGhostZones`) and dual-energy `needs_init` is armed.
6. `on_loaded(&src_pin)` runs — the pgen's `user_remap_loaded_func` — *before* the state is
   applied, so a pgen can restore metadata (e.g. TDE's BH/frame record) that its own
   `skip_cell` hook needs during the apply that follows.
7. `ApplyRemapCC` (host-side, mirror views + host loops, one `deep_copy` back) transfers
   gas (+i0 as enabled); if the source is MHD, `BuildCoveringPotential` then
   `ApplyRemapFC` transfer B.
8. `pm->time/dt/dtold/ncycle` are installed **from the source** — `<time>/nlim` in a remap
   target's deck is therefore an **absolute** cycle count continuing the source's, not a
   post-remap budget (production decks add the intended post-remap cycle count to the
   source's final `ncycle`, with a comment).
9. If `pmbp->padm != nullptr` and it has a `SetADMVariables` callback, it is invoked now
   (metric, lapse, excision masks): this is what makes dyn-GR pgens (`gr_torus`) work with **zero**
   pgen code.
10. `pm->time`/`<outputN>` bookkeeping is written into `dst_pin`; the rank-0 banner prints
    (source path, relativity class, band mode, source time/cycle, blocks loaded, gas
    group, whether FC/i0 were applied and why not when skipped, any skipped
    `<adm>` residual).
11. `pgen->user_remap_post_func(summary)` runs last (frame switches, banners, `remap_test`'s error report).

**Restart behaviour.** The module never runs during a restart (`athena -r`); a "remap then
LAT" or "remap then continue" run is always a two-step chain: remap with `time/lat=false`,
let it write a restart, then `athena -r <that restart> time/lat=true` (or with an updated
`nlim`). `LoadAndApplyRemap` is also usable programmatically for mid-run re-remaps
(TDE's settle passes); such callers must call `Driver::InitBoundaryValuesAndPrimitives(pm)`
afterwards, since the module itself performs no boundary comms, prolongation, physical BCs
or C2P — ghost zones and derived state are left to the normal startup/mid-run pipeline.

**MPI/GPU notes.** The whole module is host-only: mirror views, host loops, one
`deep_copy` back to device per group (`remap_cc.cpp` comment: "Host-side
remap setup cannot query the device table directly"; `remap_fc.cpp`). CC data is
loaded per-rank, restricted to blocks overlapping the local pack's bounding box plus 2 root
cells (`LocalPackBounds`/`ExpandRegionSize`, `remap_cc.cpp`, called from
`remap_load.cpp`). FC data is loaded for **all** source blocks on **every** rank
(the covering-grid vector-potential construction is global and independently replicated,
so the result is deterministic and rank-count-invariant — the double-precision
result is bitwise identical for 1 and 4 ranks). Every diagnostic sum/print
is MPI-reduced/rank-0-gated (`MPI_Allreduce` in `remap_fc.cpp` for the seam
report; rank-0 `std::cout` throughout).

## Interactions

- **LAT (`<time>/lat`).** Refused outright: `LoadAndApplyRemap` calls `FatalRemap` if
  `dst_pin->IsLATEnabled()`, so the check applies to every pgen. The supported flow is
  remap → restart with `time/lat=true`.
- **AMR/SMR.** Fully compatible on the target side: `kKeepTarget`/`kFloorFade` both sample
  correctly through `FindContainingSourceBlock`'s level walk, and the FC pass's
  fine-neighbor edge averaging keeps target fine/coarse interfaces restriction-consistent
  (`mhd_divb` stays at the `1e-11` level). A
  *multi-level* MHD **source** is refused unless `b_coarsen_ok = true`, because B is only
  restricted to the source root grid (R-1 below); CC fields keep full detail regardless.
- **Restart.** See "How it runs" — mutually exclusive with being launched via `-r`; only
  runs on a genuine fresh start.
- **FOFC / C2P / excision.** Not called by the module. In GR the O(h²) E-vs-B
  inconsistency the module leaves behind (gas remapped verbatim, B rebuilt from the
  interpolated potential) is absorbed downstream by the normal C2P inversion, FOFC, and
  the excision reset every stage (R-5).
- **Units (`<units>`).** `bhmass_msun`, `density_cgs`, `mu` are compared between source and
  target and only ever **warn** on mismatch, in both GR classes.
- **Other modules refused outright in the source:** `<z4c>`/`<cce>` (permanently — a
  numerically evolved spacetime cannot be resampled by an interpolant blind to the
  Hamiltonian/momentum constraints), `<turbulence>`/`<turb_driving>` (RNG step-3 state),
  `<sink_particles>`, `<shearing_box>` — none of these carry
  restart length markers this module can skip past.
- **`<adm>` metric.** Never remapped; skipped by residual byte count and recomputed
  analytically by the target pgen; see "Limitations" R-7.

## Limitations and known issues

- **Not conservative** (`remap_cc.cpp`, `docs/remap_usage.md` "Accuracy and
  conservation"). The CC accumulator is a plain mean of point samples of a trilinear
  interpolant — no cell-volume weighting, no donor/acceptor intersection volumes ("no
  supermesh"). Total mass, momentum and energy drift by $O(h^2)$ of whatever the source
  profile resolves, and by $O(1)$ wherever the source structure is unresolved on the
  target grid (any coarsening remap loses what it cannot represent). Exact only where a
  target cell coincides with a source cell (copy) and for `div(B)` (nodal-exact FC
  construction). Making it conservative would need an exact-intersection supermesh
  algorithm (√γ-weighted in GR) plus a matching flux-conservative FC restriction — out of
  scope by design.
- **Total energy is not conserved, by construction** (Newtonian CC path). The engine
  carries the thermal energy and rebuilds `E` from the interpolated `(rho, m, e_int)`, so
  the grid-scale kinetic-energy variance a coarser target cannot represent is *dropped*:
  `E` falls by that amount. Interpolating `E` instead would convert the same variance into internal energy, which is
  worse, because a state that is merely missing unresolved kinetic energy is still the right
  thermal state, while a heated one is not.
- **R-1**: a multi-level MHD source loses sub-root B structure (opt-in via
  `b_coarsen_ok`, warned).
- **R-2**: target fine/coarse FC exactness relies on the canonical fine-edge-averaging
  contract + init prolongation; `mhd_divb` stays at the `1e-11` level in the remap tests.
- **R-3**: a source whose B threads the old domain boundary gets taper sheet currents
  outside the old box (the `1e-2` relative sheet-current warning, `remap_fc.cpp`);
  physically unavoidable for a domain-widening remap.
- **R-4**: FC data is read for all blocks on every rank (global covering grid) — one-shot,
  page-cached I/O; a documented cost, not yet revisited.
- **R-5 (GR)**: gas is remapped verbatim while B is rebuilt from the interpolated
  potential, so remapped E and B are $O(h^2)$-inconsistent; a strongly magnetized, poorly
  resolved region pays for it as a one-time floor/FOFC transient in the first cycles.
- **R-6 (keep mode)**: the FC pass respects the same region the CC pass does, so the cost is
  the divergence mismatch at the taper edge, which is measured and reported (see Physics
  section).
- **R-7**: the `<adm>` payload is identified purely by residual byte count; a future ADM
  backend writing a nonzero, non-cell-shaped record would trip the "unexpected payload
  residual" fatal rather than being silently mis-parsed (deliberate, but means the loader
  needs revisiting if such a backend appears).
- **R-8 (FC keep-mode seam)**: an exactly divergence-free keep-mode seam would need the
  pgen's *own* vector potential (an inverse curl on the target hierarchy) — not
  implemented.
- **R-9 (Newtonian `band_mode = keep`)**: the thermal-energy carry lives in the floor-fade
  worker only. `kKeepTarget` is shared with GR, where the conserved columns are densitized
  and C2P/FOFC own the thermodynamics, so it still interpolates the stored `E` — a
  Newtonian keep-mode remap therefore shows the Jensen heating described above. `auto` resolves to
  `floor` for Newtonian, so this needs an explicit `band_mode = keep` to reach.
- **3D-only**, both ends, whether or not there is a B field: a 2D source would be an
  extrusion, not a remap, and the CC sampler's unconditional `(ngh+k, ngh+j, ngh+i)`
  addressing would walk a full ghost stride into the next variable's slice on a
  `nout3 == 1` source.
- **Fluid EOS must match** (column count, ideal-vs-isothermal, tabulated-vs-analytic, and
  `gamma` to `1e-12` for a gamma-law gas); floors (`dfloor`/`pfloor`/`tfloor`) only warn.
- **`band_mode = floor` with a GR class is fatal** — Newtonian floor-fade thermodynamics
  (ambient `rho`/`p` floors, `E = eint + 0.5 rho v²`) has no meaning for densitized GR
  conserved variables.
- **Retired `<problem>` keys.** `<problem>/remap`, `remap_restart_source`,
  `remap_interpolation`, `remap_settle_steps`, `remap_settle_passes` are pre-module
  `tde_external` spellings. On a **fresh** start `tde_external` refuses to run when
  `problem/remap = true` and prints the `<remap>` block to write instead; otherwise, and always on a restart (whose dump may carry
  the dead keys from the run that wrote it), they only warn.

## References

- `docs/remap_module_design.md` — full design; source of
  the algorithmic detail above.
- `docs/remap_usage.md` — user-facing quick start, accuracy statement, GR support, and
  restriction list.
