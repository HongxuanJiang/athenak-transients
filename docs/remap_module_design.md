# Universal restart-remap module (`src/remap/`) — design

Status: authoritative design for the generalization of the tde_external remap into a
reusable module, plus new face-centered (FC) magnetic-field remap via vector-potential
interpolation.

## 0. Goal and scope

The remap machinery (restart-source loading, AMR-hierarchy sampling, old-domain
boundary transition band) is a general module that any pgen can call. New capability: MHD sources — cell-centered
(CC) conserved variables remap through the existing engine, and the face-centered B field
is remapped **divergence-free** by the route the user requested: source B -> global vector
potential A (exact discrete inverse curl) -> C1 interpolation of A -> discrete curl on the
target mesh. Because the target B is the discrete curl of edge values of a single global
continuous field, every target MeshBlock is divergence-free to machine precision by
construction, independent of interpolation error.

Consumers: `tde_external`,
test pgen `src/pgen/tests/remap_test.cpp`, and any future pgen (common-envelope etc.).

Out of scope (v1, fatal error with clear message):
- Sources containing step-3 internal state (`<z4c>`, `<turbulence>`, `<sink_particles>`
  blocks) — the restart step-3 payload has no length markers (restart.cpp), so a
  reader that does not replicate the exact module set cannot find `data_size`.
- Sources with `<z4c>`, `<cce>` or special-relativistic setups (see Section 8 for the
  general-relativistic classes that are accepted).
- Hydro source -> MHD target or vice versa (CC module type must match).
- 1D meshes for FC remap (2D and 3D supported).
- Shearing-box targets.

## 1. Module layout

```
src/remap/remap.hpp        public API: RemapOptions, RemapSourceInfo, LoadAndApplyRemap()
src/remap/remap_impl.hpp   internal shared structs (RemapSource, blocks, covering grids)
src/remap/remap_load.cpp   restart parsing (header, lloc list, payload layout, block reads)
src/remap/remap_cc.cpp     CC engine (sampling,
                           transition band, ghost flooring) with field-group generalization
src/remap/remap_fc.cpp     FC engine: B restriction to covering grid, inverse curl,
                           Catmull-Rom interpolant, target edge evaluation, curl, bcc/IEN fix
```

No task-list integration: remap runs host-side at UserProblem time or from a pgen
`after_cycle_func`. Mid-run callers must call
`pdrive->InitBoundaryValuesAndPrimitives(pm)` afterwards (as tde AutoRemapAfterCycle
does in tde_external.cpp).

## 2. Public API

```cpp
namespace remap {

struct RemapOptions {
  std::string source_path;                 // restart file
  // CC behavior (defaults preserve tde_external behavior exactly)
  bool use_transition_band = true;         // old-domain boundary reconstruction band
  std::function<bool(Real x, Real y, Real z)> skip_cell;  // e.g. BH excision; cell keeps floor
  // FC (MHD) behavior
  bool b_coarsen_ok = false;               // allow multi-level MHD source (B restricted to root)
  int  b_taper_root_cells = 4;             // smoothstep taper width outside source box
  bool b_report = true;                    // print inverse-curl consistency diagnostics
};

struct RemapSourceInfo {                   // returned to the caller
  Real time; Real dt; int ncycle;
  ParameterInput* source_pin;              // valid during the callback below only
};

// Loads the source, applies CC (+FC if MHD) remap to pmbp, sets pm->time/dt/ncycle,
// optionally copies <outputN> file_number/last_time into dst_pin, and invokes
// on_loaded(source_pin) so the pgen can read its own metadata (e.g. TDE BH/frame state).
void LoadAndApplyRemap(Mesh* pm, MeshBlockPack* pmbp, ParameterInput* dst_pin,
                       bool copy_output_state, const char* banner_label,
                       const RemapOptions& opts,
                       const std::function<void(ParameterInput*)>& on_loaded = nullptr);
}  // namespace remap
```

The TDE-specific residue stays in the pgen: excision hook body, frame/BH metadata restore
(via `on_loaded`), post-remap boost / frozen-frame switch, settle-pass scheduling, AMR
cadence controls, output suppression.

## 3. Source loading (remap_load.cpp)

Restart layout facts (verified against restart.cpp writer / pgen.cpp reader):
- ASCII parameter dump to `<par_end>`; binary mesh header (nmb_total, root_level,
  RegionSize, 2x RegionIndcs, time, dt, ncycle); `LogicalLocation[nmb]` + `float cost[nmb]`;
  then (refused) step-3 payload; then `IOWrapperSizeT data_size`; then block-major payload,
  block g at `data_offset + g*data_size`.
- Per-block section order: hydro u0 | mhd u0 | b0.x1f | b0.x2f | b0.x3f | radiation i0 | ... All arrays include ghosts (`nout = nx + 2*ng`, collapsed dims
  1); face arrays are +1 in their own direction unconditionally.
- data_size validation: the module computes the expected section sizes from the SOURCE pin
  (which blocks exist) and target var counts, accepting the legacy widths for hydro
  (with/without dual-energy aux). Any residual mismatch is fatal.

CC groups loaded per block (only blocks overlapping local pack bounds + 2 root cells):
hydro u0 OR mhd u0 (into the same `hydro` vector role).
FC data (all three face arrays) is loaded for **all** source blocks regardless of rank
overlap, because the covering-grid A construction is global and replicated on every rank
(deterministic => rank-count-invariant results).

## 4. CC engine (remap_cc.cpp)

Moved verbatim: `LogicalLocationToRegionSize`, `RegionOverlaps`, `SourcePointInsideMesh`,
`SourcePointBoundaryTransitionWeight`, `FloorOuterSourceGhostZones`,
`FindContainingSourceBlock`, `SampleSourceActiveCellContaining`, `SampleSourceCellCentered`,
`SampleRemapCellAverage`, `ApplyRemapToCurrentMesh` (renamed, parameterized).

Generalization: the engine runs once per CC field group `{target array, source payload
offset, nvars_target, nvars_source, floor_state, group kind}`:
- `hydro`: dual-energy aware, EOS transition band).
- `mhd`: same engine. **Magnetic-energy swap**: at load time the module computes, per
  source block, `emag_src = 0.5*(avg(x1f)^2 + avg(x2f)^2 + avg(x3f)^2)` on active cells and
  subtracts it from the stored IEN column, so the CC engine remaps *gas* total energy
  (internal + kinetic). After the FC remap fills target b0/bcc0, the module adds
  `0.5*|bcc0|^2` back onto the remapped IEN. The transition band's
  `HostPressureFromRhoEint` math therefore operates on gas energy as it must.
  Dual-energy aux (if present) needs no adjustment.

## 5. FC engine (remap_fc.cpp) — divergence-free B remap

### 5.1 Root covering grid
Source B faces are exactly restricted onto a uniform covering grid at the SOURCE ROOT
level: covering face = arithmetic mean of the 2^dl x 2^dl fine faces on the same plane
(area-weighted restriction on a uniform Cartesian grid; preserves the discrete div-free
property and every root-face magnetic flux exactly). Processing ascends levels so finer
data wins on shared planes. Only active-zone faces are used (ghost faces are checkpoint
payload, consistent with the CC philosophy). A multi-level MHD source is REFUSED unless
`b_coarsen_ok = true`, because fine-level B structure is lost (CC fields keep full detail).
Memory: 3 arrays of ~root-mesh size (host, replicated) — negligible for realistic roots.

### 5.2 Exact discrete inverse curl (gauge A1 = 0)
On the covering grid (nx1,nx2,nx3 root cells, spacings dx,dy,dz), with the repo edge
convention (aN cell-centered along N, at faces transversally; stencil gr_torus.cpp):

- A2(k,j,i): k in [0..nx3], j in [0..nx2-1], i in [0..nx1]   (y-edges)
- A3(k,j,i): k in [0..nx3-1], j in [0..nx2], i in [0..nx1]   (z-edges)
- A1 == 0 by gauge (never stored).

Construction:
1. i = 0 plane: A2(k,j,0) = 0. A3(k,0,0) = 0; A3(k,j+1,0) = A3(k,j,0) + dy*B1f(k,j,0).
2. Sweep i: A2(k,j,i+1) = A2(k,j,i) + dx*B3f(k,j,i)  [B3f defined for k in 0..nx3]
            A3(k,j,i+1) = A3(k,j,i) - dx*B2f(k,j,i)  [B2f defined for j in 0..nx2]
3. Then curl(A) reproduces B2f, B3f exactly by construction, and B1f(i>0) exactly iff the
   source is discretely div-free (telescoping identity). Diagnostic: max |curl(A)_1 - B1f|
   is computed and printed (`b_report`); machine-eps-scale for CT-evolved sources.
4. Gauge normalization: subtract from each component its mean over the covering-domain
   boundary shell, so A ~ 0 in the far field of a localized source (bounds the taper term).

### 5.3 Interpolant and evaluation
Each component is interpolated from its own edge lattice (node coordinates: face positions
along transverse axes, cell centers along its own axis) by separable Catmull-Rom cubic
(C1 continuity => second-order-accurate B after differencing) with index clamping at the
lattice edges. Evaluation coordinate is clamped into the source box; the value is then
multiplied by a product-of-smoothstep window that falls 1 -> 0 over `b_taper_root_cells`
root cells OUTSIDE the box on each side. Inside the box the window is exactly 1 (remapped
B in the interior is NOT modified); beyond the taper band A = 0 => B = 0 exactly. Inside
the band, curl(w*A) = w*curl(A) + grad(w) x A stays divergence-free (it is still a curl);
the grad(w) x A sheet term is bounded by the gauge normalization of 5.4. If the source B
does not vanish near its own boundary, flux must close somewhere — a warning is printed
when boundary-shell |B| exceeds 1e-10 * max|B|.

### 5.4 Target application
Host loop over target MeshBlocks (mirroring the CC engine). Per block, scratch host arrays
a2, a3 sized (ncells3+1, ncells2+1, ncells1+1); fill edges (k,j,i) in
[ks..ke+1] x [js..je+1] x [is..ie+1]:
- a2 at (x1f_i, x2v_j, x3f_k), a3 at (x1f_i, x2f_j, x3v_k), evaluated from the interpolant.
- **Fine-neighbor edge averaging** (mirrors gr_torus.cpp): where an edge abuts a
  finer target neighbor, the edge value is the average of the interpolant at the two
  half-edge midpoints, so shared fine/coarse face fluxes are restriction-consistent — the
  same contract as the canonical A-initialized MHD pgens. The init pipeline then conforms
  fine shared faces via the divergence-preserving prolongation (bvals/prolongation.cpp).
- b0.x1f(m,k,j,i) = (a3(k,j+1,i)-a3(k,j,i))/dx2 - (a2(k+1,j,i)-a2(k,j,i))/dx3, cyclic;
  a1 terms drop (gauge). 2D (nx3=1): a2 contributes only through its x-derivative to x3f;
  the k+1 terms vanish (dx3 differences suppressed as in the canonical 2D pgens).
- bcc0 = face average on active cells; remapped IEN += 0.5*|bcc0|^2 (Section 4);
  deep_copy b0, bcc0, u0 to device.
Ghost faces/cells are NOT filled — Driver::InitBoundaryValuesAndPrimitives (t=0) or the
caller's explicit call (mid-run) performs comms, prolongation, physical BCs and C2P,
identical to the canonical pgen contract.

### 5.5 Accuracy statement
Target B is a second-order-accurate resampling of the root-restricted source B, exactly
divergence-free on every target block (machine precision), with exact restriction
consistency at target fine/coarse interfaces. Same-grid remap (target == source root grid)
reproduces the source B to accumulated roundoff (the interpolant is nodal-exact and the
curl inverts the prefix sums).

## 6. Refactor + wiring

- `tde_external.cpp`: calls `remap::LoadAndApplyRemap` with
  `skip_cell` = BH excision, `on_loaded` = RestoreFrameBHStateFromMetadata (+ fatal if
  absent), then the existing boost / frozen-frame / settle-pass logic. Same banners.
- `src/pgen/tests/remap_test.cpp` (built-in test pgen; hydro or MHD chosen by input blocks):
  - Analytic 3D state: smooth rho/p/v plus B = curl(A_analytic),
    A = (0, A0y*G, A0z*G*(1+0.3*sin(2*pi*x/Lx))) with G a compact Gaussian bump, so B is
    fully 3D, div-free, and vanishes near the box edge. Pgen initializes it with the
    canonical per-level edge evaluation (SMR-safe).
  - `<problem> remap = true, remap_source = ...` runs the module instead.
  - Error norms vs the analytic B and rho are computed in-pgen at the end of the run and
    printed (`remap_test_errors` line).

## 7. Module-level `<remap>` block

The pgen-call API of section 2 remains, but the primary activation path is a
module-level input block, wired like <gravity>: any pgen can be remapped with zero pgen
code.  `ProblemGenerator`'s fresh-start constructor calls `remap::MaybeAutoRemap(this,
pin, pm)` immediately after the pgen function returns (src/pgen/pgen.cpp); it is a no-op
unless the input carries a `<remap>` block with `enable = true` (default true when the
block exists; production inputs keep an explicit `enable` key so it can be flipped from
the command line).  Keys: `enable, source, copy_output_state, transition_band,
b_coarsen_ok, b_taper_root_cells, b_report, settle_steps, settle_passes` (see remap.hpp), plus
consumer-private keys that live in the same block.
`remap` is on the ParameterInput block whitelist.

Pgens participate through three optional `ProblemGenerator` members enrolled inside the
pgen function, before the auto remap runs:
- `user_remap_skip_func(x,y,z)` — cells keep the ambient floor (TDE: BH excision);
- `user_remap_loaded_func(src_pin)` — called after the source parameter dump is read and
  BEFORE the state is applied (TDE: restore the BH/frame record, which the skip hook
  reads during the apply);
- `user_remap_post_func()` — called after the apply (TDE: frame switch/boost, settle-pass scheduling, metadata store; remap_test: error report).

A pgen that wants remap must therefore only (a) skip its own fresh initialization when
`remap::IsAutoRemapEnabled(pin)` and (b) enroll any hooks it needs — both optional.
The `<problem> remap_*` keys of tde_external are retired in favor of `<remap>`;
`remap_interpolation` is not read (trilinear is the only interpolation).
Guard note: consumers must not `GetOrAdd` any `<remap>` key unless the block already
exists, or they would CREATE the block and arm auto-remap unintentionally.

## 8. GR remap

The module accepts general-relativistic sources and targets.  Everything in this section
OVERRIDES the "Newtonian only" statements of sections 0 and 3.

**8.1 Relativity classes.**  `RemapRelClass { kNewtonian, kFixedGR, kDynGRAnalytic }`
(remap_impl.hpp), stored in `RemapSourceData` together with `bool gr_mode`.
Source classification, from the SOURCE pin in remap_load.cpp:

| test (in order) | result |
|---|---|
| `<z4c>` or `<cce>` block exists | FATAL, permanently |
| `<adm>` block exists | `kDynGRAnalytic` (source `<mhd>` required) |
| `<coord> special_rel = true` | FATAL (unchanged from v1) |
| `<coord> general_rel = true` | `kFixedGR` |
| otherwise | `kNewtonian` |

Target classification is by runtime object, not by input keys: `pz4c != nullptr` FATAL;
`padm != nullptr` -> `kDynGRAnalytic`; `is_special_relativistic` FATAL;
`is_general_relativistic` -> `kFixedGR`; else `kNewtonian`.  The `padm` test comes FIRST
because `pcoord->is_general_relativistic` is **false** for dyn-GR runs.
Source class must equal target class; the fatal names both.

Refusing z4c is deliberate: resampling a numerically evolved
spacetime with an interpolant that knows nothing about the Hamiltonian and momentum
constraints yields plausible-looking, constraint-violating initial data.

**8.2 Consistency checks** (GR classes only) compare the two parameter dumps and never
`GetOrAdd` on the target pin (that would pollute the target's own dump and, for
`<remap>`, could arm auto-remap).  `kFixedGR`: `<coord> a` (>1e-14 relative) and
`minkowski` differ -> FATAL; `excise/dexcise/pexcise/flux_excise_r` -> rank-0 WARNING.
`kDynGRAnalytic`: all `<coord>` and the `<problem>` orbit/mass key list only WARN.  `<units>`
(`bhmass_msun, density_cgs, mu`) WARN in both.

**8.3 Payload layout** mirrors src/outputs/restart.cpp exactly: hydro u0 | mhd
u0 | b0.x1f | b0.x2f | b0.x3f | rad i0 | turb (refused) | z4c
(refused) XOR adm u_adm (SKIPPED).

- `i0`: `nangles = 10*nlevel^2 + 2` from the source `<radiation> nlevel`.  Target with
  radiation requires equal `nlevel`, `rotate_geo`, `angular_fluxes` (FATAL otherwise);
  target without radiation skips the bytes with a warning.
- `adm`: derived by RESIDUAL after every known section is summed.  Residual 0 means the
  analytic backend wrote nothing (`RestartVariableCount() == 0`); a nonzero residual
  requires an `<adm>` source block and must divide `nout1*nout2*nout3*sizeof(Real)`, and
  those bytes are skipped — the target pgen recomputes its metric.  Any other residual is
  fatal ("unexpected payload residual"), which keeps the exact `data_size` validation.

**8.4 Emag round trip is Newtonian-only.**  Both halves — the loader's subtraction of
the magnetic energy from the source IEN and `ApplyRemapFC`'s add-back — are gated to
`kNewtonian`.  In GR the densitized conserved variables are smooth analytic-metric
functions and are remapped verbatim; the resulting O(h^2) E-vs-B inconsistency is
absorbed by C2P/FOFC/excision.  Horizon cells are NOT skipped anywhere: gr_torus
excision never touches `b0`, so `div(B) = 0` holds globally including inside horizons,
and the target's excision task resets the interiors every stage.  Skipping them would
break the discrete curl identity that makes the guarantee exact.

**8.5 Band mode.**  `RemapBandMode { kFloorFade, kKeepTarget }` with input key
`band_mode = auto|floor|keep`; `auto` -> `kKeepTarget` for GR classes, `kFloorFade` for
Newtonian.  GR + `floor` is FATAL (Newtonian floor-fade thermodynamics has no meaning in
GR).  `transition_band = false` bypasses band logic entirely and stays valid everywhere.
`kKeepTarget`, per target cell and per group (gas, i0), computes an
EXTERIOR fade weight `w`: `w == 1` for every sub-sample inside the source active box
(full weight up to the last active cell — the source ghosts are not floored in keep
mode), and outside the box the clamped boundary value decays by a quintic smoothstep
over `b_taper_root_cells` source root cells (the FC vector-potential taper geometry).
`w == 0` writes nothing (pgen state preserved), `0 < w < 1` blends conserved variables
linearly, `w == 1` overwrites.  This deliberately does NOT reuse the Newtonian
`SourcePointBoundaryTransitionWeight`, whose band begins *inside* the boundary: an
interior band blends t=0 pgen material back into a fully covered domain.  No
floor fade, no Newtonian pressure operations, no ambient inward-shift heuristic on this
path.  Post-blend safeguards: `i0` is clamped non-negative.

**8.6 Orchestrator.**  After `pm->time/dt/ncycle` are installed and BEFORE the pgen post
hook, the orchestrator calls `pmbp->padm->SetADMVariables(pmbp)` when that pointer and
function exist.  This one line is what makes dyn-GR pgens work with zero pgen code: the
analytic metric, lapse and excision masks are
rebuilt at the SOURCE time rather than left at t = 0.  `RemapSummary` gains
`gr_mode`, `i0_applied`; the banner reports class, band mode and which
groups were applied or skipped.

**8.7 Pgen changes.**  `gr_torus` needs NONE.  The fresh-start initialization is NOT skipped:
keep-mode banding blends against the pgen state, so that state must exist.

## 9. Risks / documented limitations

- R-1: multi-level MHD source loses sub-root B structure (explicit opt-in, warning).
- R-2: target fine/coarse interface exactness relies on the canonical fine-edge-averaging
  contract + init prolongation.
- R-3: sources whose B threads the old domain boundary get taper sheet currents in the
  band outside the old box (warned). Physically unavoidable for a domain-widening remap.
- R-4: FC data is read for all blocks on every rank (global covering grid). IO is one-shot
  and page-cached; revisit only if a production source makes it measurable.
- R-5 (Section 8): in GR the gas is remapped verbatim while B is rebuilt from the interpolated
  vector potential, so the remapped E and B are inconsistent at O(h^2).  C2P/FOFC/excision
  absorb it; a source with a strongly magnetized, poorly resolved region will pay for it
  as a one-time floor/FOFC transient in the first cycles.
- R-6 (Section 8): keep-mode blends the SOURCE conserved gas against the PGEN conserved gas in
  the band.  The FC pass respects the same keep notion (Section 10.3), so the outer band
  keeps the pgen field and only the seam layer carries `div(B) ~ |B_pgen|/h`.
- R-7 (Section 8): the ADM payload section is identified by residual byte count.  A future ADM
  backend that writes a nonzero, non-cell-shaped record would trip the "unexpected payload
  residual" fatal rather than being silently mis-parsed — deliberate, but it does mean the
  loader must be revisited when such a backend appears.

## 10. Sampler, FC and loader behavior details

Everything here
OVERRIDES the corresponding statements above.

**10.1 The CC sampler is exact on a coincident grid** (`remap_cc.cpp`,
`TargetCellMatchesSourceGrid`).  The samplers otherwise take two sub-samples per axis at
`x_c ± h/4` of the trilinear interpolant.  On a coincident grid that is exactly the
separable filter `[1/8, 3/4, 1/8]` — transfer `0.75 + 0.25 cos(kh)`, hence 0.5 per axis at
the Nyquist wavenumber and **0.125 in 3D**.  Without this special case a same-grid remap would not be the
identity: it would remove 87.5% of the grid-scale gas power while the FC/B path, nodal-exact by
construction (section 5.5), removes none — one remap, two different filters on the two
fields it transfers.  Where the source cell containing the target cell center has the same
width and the same center (both to 1e-10 of a cell), the samplers take ONE sub-sample
at that center, which the trilinear weights reduce to a bit-exact copy. 

The sampler is NON-CONSERVATIVE, as `docs/remap_usage.md` states in its own section.  The accumulator is a plain mean of point
samples: no cell-volume weight, no donor/acceptor intersection volumes.  Making it
conservative is a supermesh algorithm (exact intersection volumes of two AMR hierarchies,
`sqrt(gamma)`-weighted in GR) plus a matching flux-conservative FC restriction to keep
`div(B) = 0` — a different piece of work that this module does not attempt.

**10.2 Dyn-GR conserved variables are un-densitized across the interpolation**
(`RemapMetricSampler` in remap_impl.hpp, defined in remap_load.cpp; used by remap_cc.cpp).
On the `<adm>` path every conserved column is `sqrt(gamma)` times a fluid quantity
(`primitive_solver_hyd.hpp` stores `cons_pt[...]*sdetg`), and interpolating the stored
columns directly puts the metric profile inside the smoothing stencil: the recovered
`rho` carried an error proportional to the CURVATURE of `sqrt(gamma)`, i.e. worst at the
punctures.  The loader now evaluates `sqrt(det gamma_ij)` at every source cell center into
`RemapSourceBlock::sqrt_gamma`, the samplers divide each interpolation node by it, and
`SampleKeepCellAverage` re-multiplies by `sqrt(gamma)` at the target cell center.  The
metric is built **at the source time** (`padm` still holds the pgen's `t = 0` metric during
the remap; the punctures have moved since), from `ADMMetricView::CartesianMetric` — a
KOKKOS_INLINE_FUNCTION, so it is callable here and is bit-identical to what the target's
own kernels evaluate.  Coincident cells skip the round trip entirely — same point, same
factor — so the copy of Section 10.1 stays bit-exact. 

Two ends have no sampler, on purpose.  **kFixedGR does not densitize at all**:
`ideal_c2p_mhd.hpp SingleP2C_IdealGRMHD` stores `u.d = rho*u^0` with no `sqrt(gamma)`
anywhere (the volume element lives in the flux and source terms), so there is nothing to
divide out and imposing a `sqrt(gamma)` round trip would be a change of variables the data
does not have — it would be worse than doing nothing wherever `sqrt(gamma)` varies and the
conserved state does not.  And a **stored ADM backend** has only target-grid samples, so
`gamma_ij` cannot be evaluated at a source cell center; the loader warns on rank 0 and
keeps the plain interpolation of stored columns.

**10.3 The FC pass has a keep notion** (remap_fc.cpp).  Rebuilding
`b0`/`bcc0` from A over the whole target mesh would drive the target's B to exactly zero outside the
source box plus taper while the gas there kept
the pgen's magnetized conserved state — in GR that state then decodes against `B = 0` and
the electromagnetic energy and momentum are read back as fluid.  In `kKeepTarget` the pass
starts from the pgen's own field and writes only the faces whose curl stencil touches
an edge where the taper window is still alive (the window is EXACTLY 0 beyond the taper,
so the test is exact; it is on the window, not on A, because A can vanish for unrelated
reasons).  Two divergence-free fields meeting face-by-face are not divergence-free at the
seam, so the one cell layer at the outer edge of the taper can carry
`div(B) ~ |B_pgen|/h`.  That is a genuine cost, accepted deliberately — the alternative is
gas with no field to decode it against, over a volume rather than a layer — and it is
measured (`max |div(B)|*h` vs `max |B|`, MPI-reduced) and reported outside `b_report`.  A
same-domain remap has no seam.  New risk **R-8**: an exactly divergence-free keep-mode seam
would need the pgen's own vector potential, i.e. an inverse curl on the TARGET hierarchy.

**10.4 The magnetic-energy add-back is gated and weighted** (remap_fc.cpp).  Gating on
`!gr_mode` alone would let `band_mode = keep` with a Newtonian MHD source add `0.5|B_new|^2`
to cells the keep pass leaves holding the pgen's TOTAL energy — which already contains
the pgen's magnetic energy.  The add-back is `0.5|B_new|^2 - (1-w)*emag_pgen` with `w`
the CC pass's own blend weight (`KeepSampleWeight`, factored out of remap_cc.cpp and shared
so the two passes cannot drift apart), forced to 0 in `skip_cell` cells.  In floor-fade
mode `w == 1` and `emag_pgen == 0` in every cell, so the add-back reduces to `0.5|B_new|^2`.

**10.5 The LAT rule lives in the module** (remap.cpp `LoadAndApplyRemap`).  A remap may not
run while LAT windows are active.  The startup remap (`MaybeAutoRemap` sets
`RemapOptions::lat_idle`) and a mid-run remap made while the pgen has paused LAT
(`Mesh::hydro_lat_suspended`) are allowed; any other call under `time/lat = true` is fatal,
on the one path that both the `<remap>` auto mode and every programmatic caller take.  After
the state is replaced, the module invalidates the LAT metadata and pulls the cycle anchors
that the LAT gates compare with `ncycle` back to the rewound `ncycle`.

**10.6 The inverse-curl residual is gated** (remap_fc.cpp).  The residual is printed under
`b_report`, but it is also checked when `b_report = false`.  Relative to `max |B|`: `> 1e-8` is a warning
that `b_report` cannot suppress, `> 1e-2` is fatal.  Scale: the prefix sums accumulate
`eps*|A|` roundoffs and are divided by one cell width, so a CT-evolved source lands near
`eps*n_root*|B| ~ 1e-13` even on a 512-cell root grid.

**10.7 Loader refusals** (remap_load.cpp).

- *Fluid EOS.*  The source's gas column count is derived from the SOURCE's own `<hydro|mhd>/eos` (the same dispatch hydro.cpp/mhd.cpp use: only
  `isothermal` drops the energy column), and source/target must agree on the column count,
  on ideal-vs-isothermal, on tabulated-vs-analytic and, for a gamma-law gas, on `gamma` to
  1e-12 — all fatal.  Floors (`dfloor`, `pfloor`, `tfloor`) warn.
- *Dimensionality.*  Both ends must be 3D
  regardless of B (a 2D source into a 3D target would be an extrusion, not a remap).  The CC sampler addresses source
  blocks as `(ngh+k, ngh+j, ngh+i)` unconditionally, and on a 2D source, where
  `nout3 == 1`, that reads a full ghost stride into the next variable's slice.
- *Retired pgen keys.*  tde_external refuses a FRESH start
  on `problem/remap = true` and prints the `<remap>` block to write instead; the retired
  keys otherwise warn, and a restart only ever warns because its parameter dump can carry the
  retired keys from the run that wrote it.
