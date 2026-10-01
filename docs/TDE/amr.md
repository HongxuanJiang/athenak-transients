# Adaptive mesh refinement with `<tde_amr>`

This page explains how to use the `<tde_amr>` refinement scheme of `tde_external`.  The
key table is in [`parameters.md`](parameters.md#tde_amr-stream-following-refinement), and
the code is in `src/pgen/tde_amr.hpp` and `src/pgen/tde_external.cpp`.

## 1. When to use it

The scheme follows the dense stream of the debris and resolves it by region, instead of
combining a density threshold with BH, unbound-gas and stream-shell requests.  It is meant
for the fallback and later steps of a TDE run, with 7 or more levels, where the legacy
scheme needs too many blocks.

| | Legacy scheme | `<tde_amr>` |
|---|---|---|
| Density criterion | generic `<amr_criterion0> method = min_max` | none, the stream is found from the local spine density |
| Stream | `stream_shell_*` spherical shells | (log r, phi) spine table about the BH, with regions |
| BH and unbound gas | `bh_max_amr`, `unbound_amr` | nozzle region and the optional sink rule |
| Refinement flags | combined from all criteria | set by one hook, the problem generator |
| Off the orbital plane | refined like the plane | capped at `offplane_level` by default |

The scheme is opt-in.  A deck without a `<tde_amr>` block runs the legacy logic, bitwise
unchanged.

A deck that mixes `<tde_amr>` with generic criteria or legacy keys is rejected at
startup.  The following stop the run with an error.

* An `<amr_criterionN>` block with `method` other than `user`, such as `min_max`.
* `problem/bh_max_amr`, `problem/unbound_amr` or any `problem/stream_shell_*` key.
* No `<amr_criterionN>` block with `method = user`.
* `<mesh_refinement>/refinement` other than `adaptive`.

`<refined_regionN>` blocks are not affected, and still set a minimum level in a box.

## 2. Examples

The minimal deck needs the mesh refinement block, one user criterion and an empty
`<tde_amr>` block.  Every other key then takes its default.

```
<mesh_refinement>
refinement          = adaptive
num_levels          = 10
ncycle_check        = 20
refinement_interval = 20
max_nmb_per_rank    = 20000

<amr_criterion0>
method = user

<tde_amr>
verbose = true
```

The block of a 10-level fallback run with every key at its default follows.  Here
`r_peri = r_t / beta`, and the keys that scale with it are shown as comments, because the
generator computes them.

```
<tde_amr>
verbose            = true
nbins_r            = 128
nbins_phi          = 64
# r_bin_min        = r_peri
# r_bin_max        = -1 (farthest mesh corner)
strand_frac        = 0.01
# rho_min          = 100 * hydro/dfloor
spine_frac         = 0.5
envelope_factor    = 10.0
envelope_levels    = 1
hysteresis         = 2.0
radius_hysteresis  = 1.25
# nozzle_radius    = 2 * r_peri
nozzle_offset      = 0
# post_nozzle_radius = 5 * r_peri
post_nozzle_offset = 1
crossing_div       = 3.0
# crossing_r_max   = 20 * r_peri
crossing_offset    = 1
apocentre_frac     = 0.8
apocentre_boost    = 1
spine_offset       = 1
spine_r_per_level  = 2.0
spine_offset_max   = 1000
background_level   = 0
midplane_only      = true
midplane_hmin      = 0.0
midplane_angle_deg = 0.0
# offplane_level   = background_level
sink_offset        = -1
# core_rho is not set, so there is no core-star region
```

With `num_levels = 10` the finest logical level is 9.  Set `max_nmb_per_rank` from the
estimates in section 4, and keep `ncycle_check` and `refinement_interval` as in the
example decks.

## 3. How the target level of a block is decided

**Spine reference.**  The cells are binned by the distance `r` to the BH, in `nbins_r`
logarithmic bins between `r_bin_min` and `r_bin_max`, and by the azimuth `phi` in the
orbital plane, in `nbins_phi` bins.  The orbital plane contains the BH and is normal to the
initial `r x v` of the star.  A bin is a shell sector of every height.  Its maximum
density is the local spine density `rho_ref(r, phi)`, and the table is reduced over all
ranks at every check.  A cell is considered only if `rho > rho_min` and its bin is at
least `strand_frac` times the largest spine density at the same radius, so that bins the
stream has left carry no stream.

**Core and envelope.**  With `q = rho / rho_ref`, a cell with `q >= spine_frac` is a core
cell.  Below that a cell loses one level for every factor `envelope_factor` of `q`, up to
`envelope_levels` levels.  A cell below the envelope is background.  The flanks of the
stream are therefore resolved one level below the spine by default.

**Regions.**  A core or envelope cell takes the finest level of the regions it matches.
The levels below are `max_level - offset - drop`, with `drop` the envelope loss.

| Region | Condition | Default level |
|---|---|---|
| Nozzle | `r < nozzle_radius` | `max_level` |
| Post-nozzle | `r < post_nozzle_radius` and the radial velocity is positive | `max_level - 1` |
| Self-interaction | `r < crossing_r_max` and `-div v >= crossing_div * Omega_K` | `max_level - 1` |
| Apocentre | `r >= apocentre_frac * r_apo` | spine ladder level plus 1 |
| Spine ladder | every stream cell | see below |

The post-nozzle and self-interaction regions start at `nozzle_radius`.  Here `r_apo` is
the apocentre of the Kepler orbit that the cell would follow about the BH, computed from
its specific energy and angular momentum, and only bound cells have one.

**Spine ladder.**  The level of the spine falls by one for every doubling of the distance
(`spine_r_per_level = 2`).  The offset is `spine_offset` up to `spine_r_per_level *
nozzle_radius`, one more up to the square of it, and so on, up to `spine_offset_max`.
With the defaults the spine is at level `max_level - 1` inside `2 * nozzle_radius`, at
`max_level - 2` out to `4 * nozzle_radius`, and so on.  The ladder has no cap by default.

**Apocentre boost.**  The gas near its apocentre is the front of the stream and the first
to meet the returning debris.  It is given the ladder level of its radius plus
`apocentre_boost` levels.

**Midplane-only rule.**  The rule acts on whole blocks.  With `midplane_only = true` a
block can be refined above `offplane_level` only if it intersects the band
`|h| <= max(midplane_hmin, tan(midplane_angle_deg) * R)` around the orbital plane, where
`h` is the distance to the plane and `R` is the in-plane distance to the BH of the point
of the block nearest to it.  Other blocks are capped at `offplane_level`, which is
`background_level` by default.  A block that touches the band keeps the full target of its
cells.  Setting `midplane_angle_deg` and `midplane_hmin` widens the band, and
`midplane_only = false` removes the rule.

**Exceptions.**  The sink rule (`sink_offset >= 0`) sets level `max_level - sink_offset` for
every block within the excision radius of the BH.  The core-star region (`core_rho` set)
sets `max_level - core_offset` for cells with `rho >= core_rho`, and it is kept until
`rho < core_rho_deref`.  Both are exempt from the midplane rule.

**Flags.**  A block is flagged for refinement if its level is below the largest target of
its cells, and for derefinement if its level is above the largest relaxed target of its
cells.  Otherwise it keeps its level.

## 4. Tuning memory against resolution

The numbers below were estimated on the real stream of the HR fallback run, from a dump of
lower resolution than the target meshes.  They count the leaf blocks of the nominal
target mesh and leave out the buffer blocks that the 2:1 rule adds, so they are lower
bounds.  The defaults and `midplane_only = true` are used.

| `num_levels` | Leaf blocks (lower bound) |
|---|---|
| 7 | about 780 |
| 8 | about 1.9k |
| 10 | about 13.5k |

At 10 levels the blocks that the regions add are, in the order of their size, the nozzle
(about 5.5k), the self-interaction region (3.1k), the apocentre region (2.0k), the spine
ladder (1.5k) and the post-nozzle region (1.5k).  Blocks that two regions both select are
counted once, for the region of their finest cell.

The keys that trade memory against resolution are the following.

* `nozzle_radius`.  The nozzle holds the finest level, so its volume grows as the cube of
  the radius.  Setting it to `r_peri` instead of the default `2 * r_peri` brings the 10-level
  estimate to about 8.0k blocks.  It also sets the start of the spine ladder.
* `post_nozzle_offset`, `crossing_offset`.  Raising them from 1 to 2 lowers those two
  regions by one level.  Together with the change above, about 5.6k blocks at 10 levels.
* `spine_r_per_level`.  A larger value lowers the ladder more slowly and keeps more of the
  far stream fine.  A smaller one saves memory in the far stream.
* `spine_offset`, `apocentre_boost`.  The spine level near the nozzle and the apocentre
  level.
* `crossing_div`, `crossing_r_max`.  A larger `crossing_div` selects only the strongest
  compression, and a smaller `crossing_r_max` ends the region earlier.  The region
  is the second largest at 10 levels.
* `midplane_angle_deg`, `midplane_hmin`.  They widen the band of blocks that may be
  refined, and every step costs blocks above and below the plane.
* `strand_frac`, `rho_min`.  Raising them drops faint bins and cells from the stream, and
  lowering them includes the faint edges.
* `envelope_levels`, `envelope_factor`.  `envelope_levels = 0` removes the envelope, and
  the flanks then fall to the background.
* `background_level`, `offplane_level`.  The level of the gas that is not followed.

Estimate the effect of a change with the block counts printed by `verbose` (section 5)
on a short run, and keep `max_nmb_per_rank` above the estimate with a margin for the
buffer blocks.

## 5. Reading the `verbose` output

At start the generator prints a banner with the resolved values of every key.  With
`verbose = true`, rank 0 prints one line per AMR check.

An illustrative line follows.

```
tde_amr: cycle=775160 time=1.2e3 blocks/level 0:6 1:9 ... 9:484 | regions spine:812 nozzle:290 offplane:5000 | flags +1:12 -1:3
```

* `blocks/level` is the number of blocks at each level counted from the root, before
  the pass.
* `regions` is the number of blocks by the region that sets their nominal target.  Blocks
  capped by the midplane rule are `offplane`, and blocks that no region selects are
  `background`.  A region with no block is not printed.
* `flags +1` and `-1` are the blocks flagged for refinement and for derefinement.  Both
  should tend to zero once the mesh follows the stream.

## 6. Checks and limits

**Memory cap.**  `mesh_refinement/max_nmb_per_rank` caps the blocks per rank.  If the
closure of a refinement pass would exceed the cap, the refinements of the pass are
cancelled and its derefinements are kept, so that memory is released.  A mesh that
derefinement alone cannot bring under the cap still cancels the whole pass, and the run
prints a message.  Set the cap above the estimate of section 4.

**Cadence.**  `ncycle_check` and `refinement_interval` set the number of cycles between
checks, and with LAT they are 512 so that a check happens about once per synchronization
window.

**Hysteresis.**  A block refines at the nominal thresholds and derefines only at the
relaxed ones, which are lower in density by `hysteresis` and wider in radius and band by
`radius_hysteresis`.  Raising them stops the mesh from changing back and forth, at the
cost of keeping more blocks.  Both must be at least 1, and 1 switches the hysteresis off.

**One level per pass.**  A block changes by one level in an AMR pass.  A mesh that starts
coarse needs one pass per level to reach the target, so the settle steps of a remap
(`remap/settle_steps`, with `remap/settle_passes`) should provide at least as many passes
as the number of levels to build.
