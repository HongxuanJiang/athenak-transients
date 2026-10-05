# Parameter reference for `tde_external`

This page lists every input key that the `tde_external` problem generator reads from
the `<problem>` block, and the keys of other blocks that the TDE example decks rely on.
It was compiled from the parameter reads in the source, and each entry names the file
in which the key and its default are defined.  All paths in the "Source" column
are relative to the repository root.

The keys are grouped by role.  The "Stage" column says when a key matters in the
five-step chain of [`inputs/TDE_examples/README.md`](../../inputs/TDE_examples/README.md):

* **1** is the initial disruption, a fresh start.
* **R** is any step that begins with a `<remap>` block (steps 2 to 5).  The gas state
  comes from the source restart file and the BH and frame state is restored from
  metadata stored in that file.
* **all** means the key is used at every step.

Code units are `M0 = 1 Msun`, `L0 = 2 Rsun`, `t0 = 1270.92 s`, and `4 pi G = 1`.  The
mass of the star is fixed to 1 in code units.  See
[`README.md`](README.md#physical-setup-and-code-units) for how to change them.

## 1. Keys that select the problem

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/pgen_name` | string | `none` | Must be `tde_external`. The binary must also be built with `PROBLEM=tde_external`. | `src/pgen/pgen.cpp` |
| `problem/user_srcs` | bool | `false` in the core, forced to `true` by the problem generator | The generator always enables its user source terms, so this key has no effect. The example decks set it to `true` for clarity. | `src/pgen/pgen.cpp`, `src/pgen/tde_external.cpp` |
| `problem/external_bh_gravity_source` | bool | `true` for `tde_external` | Applies the BH pull as an external source term with the same potential stencil as the BH force. See [Live black hole](#live-black-hole). The example decks set it explicitly. | `src/srcterms/srcterms.cpp`, `src/hydro/hydro.cpp`, `src/main.cpp` |
| `problem/external_bh_dt_factor` | real | `0.5` | Safety factor of an extra time-step limit from the BH acceleration. See [Live black hole](#live-black-hole). | `src/srcterms/srcterms.cpp` |

The retired keys `problem/remap`, `problem/remap_restart_source`,
`problem/remap_interpolation`, `problem/remap_settle_steps` and
`problem/remap_settle_passes` are no longer read.  A fresh start with
`problem/remap = true` stops with an error that shows how to migrate the deck to the
`<remap>` block, and the other retired keys only print a warning (`src/pgen/tde_external.cpp`).

## 2. Star (stage 1)

These keys build the initial star.  Later steps still parse them, but the gas state
comes from the source restart file.  The example decks keep them unchanged in every
step so that all five files stay consistent.

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/star_radius` | real | `0.5` | Stellar radius in code length. With `L0 = 2 Rsun` the default is one solar radius. Must be `> 0`. The stellar mass is fixed to 1. | `src/pgen/tde_external.cpp` |
| `problem/stellar_structure_mode` | string | `legacy_polytrope` | `eos_balanced` integrates hydrostatic equilibrium with the active tabulated EOS (used by all example decks). `legacy_polytrope` builds an `n = poly_n` Lane-Emden star and maps its pressure through the EOS. The default is the legacy mode, so decks for the paper must set `eos_balanced`. | `src/pgen/tde_external.cpp` |
| `problem/poly_n` | real | `1.5` | Polytropic index, `0 < n < 5`. In `legacy_polytrope` mode it defines the star. In `eos_balanced` mode it only provides the starting guess for the central density and pressure of the solver. | `src/pgen/tde_external.cpp` |
| `problem/rho_floor` | real | `1.0e-8` | Ambient density added to the stellar density, `rho = rho_floor + rho_star`. It also defines the pressure floor of the initial state. Must be `>= 0`. The example decks use `1.0e-10`, the value of Table 1 of the paper. | `src/pgen/tde_external.cpp` |
| `problem/amp` | real | `0.0` | Amplitude of an `m = 2` density perturbation of the legacy polytrope. Must be `0` when `stellar_structure_mode = eos_balanced`. | `src/pgen/tde_external.cpp` |
| `problem/x_center`, `y_center`, `z_center` | real | `-1.0`, `0.0` | Center of the star in simulation coordinates. The initial orbit assumes that the star sits at the origin, so use `0` for all three as the example decks do. The default `x_center = -1` is not appropriate for the standard orbit setup. | `src/pgen/tde_external.cpp` |
| `problem/vx_star`, `vy_star`, `vz_star` | real | `0.0` | Bulk velocity of the gas relative to the moving frame. The orbital velocity is carried by the frame, so these are normally zero. | `src/pgen/tde_external.cpp` |
| `problem/relax_damp` | bool | `false` | Switches on velocity damping of the star during the first `relax_t_end` time units. | `src/pgen/tde_external.cpp` |
| `problem/relax_tau` | real | `0.0` | Damping time. The momentum inside `relax_radius` is multiplied by `exp(-dt / relax_tau)` every step. Must be `> 0` when `relax_damp = true`. | `src/pgen/tde_external.cpp` |
| `problem/relax_t_end` | real | `0.0` | Time after which the damping is off. Must be `> 0` when `relax_damp = true`. | `src/pgen/tde_external.cpp` |
| `problem/relax_radius` | real | `star_radius` | Radius around the star center in which the damping acts. Must be `> 0` when `relax_damp = true`. | `src/pgen/tde_external.cpp` |

The damping only acts on cells with `rho >= problem/bh_grav_rho_min` (`src/pgen/tde_external.cpp`).  Example decks use `relax_tau = 0.02` and `relax_t_end = 0.2`.

## 3. Orbit and black hole

The orbit is set up at a fresh start in star-centered coordinates.  In later steps the
BH and frame state is read back from the metadata in the source restart file, so the
orbit keys have no effect, but `mass_ratio` is still used to set the BH mass and must
stay unchanged.

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/mass_ratio` | real | `1000.0` | `M_BH / M_star`. With `M_star = 1` this is also the BH mass in code units. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `problem/beta` | real | `1.0` | Penetration factor `r_t / r_p`, with `r_t = star_radius * mass_ratio^(1/3)` and `r_p = r_t / beta`. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `problem/ecc_bh` | real | `1.0` | Orbital eccentricity, `0 < e <= 1`. `1` gives a parabolic orbit. For `e < 1` the star starts at apoapsis and `sep_initial` is ignored. | `src/pgen/tde_external.cpp` |
| `problem/theta_bh` | real | `0.0` | Inclination in degrees. The orbit is rotated about the y axis, so a positive value tilts the orbital angular momentum from `+z` toward `+x`. Both orbit branches use the same sign. | `src/pgen/tde_external.cpp` |
| `problem/sep_initial` | real | `10.0` | Initial star-BH separation in units of `r_t` for the parabolic orbit. Must be `> 0`, and large enough for the chosen `beta`. | `src/pgen/tde_external.cpp` |
| `problem/provide_params` | bool | `false` | If `true`, the star state relative to the BH is given directly by `x1..vz1` and the orbit keys above are not used. | `src/pgen/tde_external.cpp` |
| `problem/x1`, `y1`, `z1` | real | `0.0` | Star position relative to the BH when `provide_params = true`. | `src/pgen/tde_external.cpp` |
| `problem/vx1`, `vy1`, `vz1` | real | `0.0` | Star velocity relative to the BH when `provide_params = true`. | `src/pgen/tde_external.cpp` |

For the default parabolic orbit the geometry follows the Phantom TDE setup: the star
starts at distance `sep_initial * r_t` from the BH, and the pericenter lies along the
y axis.

### Live black hole

The example decks use a live BH from step 2 on; in step 1 the BH starts outside the mesh
and follows the two-body orbit (`bh_live = false`), because a live BH without the
reciprocal force samples the gas potential at its position.  The BH is a softened point mass, and
its inertial position and velocity are advanced together with the gas.  In the paper's
words, the BH is treated as a live sink particle whose position and velocity are updated
throughout the calculation.  The keys below control this.  `bh_live`,
`bh_reciprocal_force`, and `bh_pair_force` are read without being added to the parameter
dump (`src/pgen/tde_external.cpp`), so a deck that omits them is unchanged.

| Key | Type | Default | Meaning | Requirements | Source |
|---|---|---|---|---|---|
| `problem/external_bh_gravity_source` | bool | `true` for `tde_external` | Applies the BH pull on the gas as an external source term whose spatial stencil is the same one used to compute the force on the BH. The example decks set it explicitly. | Needed by `bh_live` | `src/srcterms/srcterms.cpp`, `src/hydro/hydro.cpp`, `src/main.cpp`, `src/pgen/tde_external.cpp` |
| `problem/bh_live` | bool | `false` | The BH orbit responds to the gas. Each step (each synchronization window with LAT) the inertial BH state is advanced by a kick-drift-kick leapfrog. The acceleration is the gas self-gravity acceleration interpolated at the BH position, or the reciprocal force below when that is on. The state is kept in the simulation frame for the hydro and AMR, and it works with the translating frame and with the frame fixed at the BH rest frame. | `external_bh_gravity_source = true` | `src/pgen/tde_external.cpp` |
| `problem/bh_reciprocal_force` | bool | `false` | The gas acceleration of the BH is the equal and opposite of the BH force on the gas, integrated over the mesh with the same stencil, density weight, and cell masks as the gas source term, instead of the self-gravity sample. The BH velocity is reconciled with the gas momentum impulse at the end of each LAT window. | `bh_live = true`, `time/lat = true`, self-gravity, `external_bh_gravity_source = true`. Otherwise the run stops with an error. A remap cannot run with LAT, so the example chain turns it on only in the LAT restart of step 5 (`problem/bh_reciprocal_force=true` on the command line). | `src/pgen/tde_external.cpp`, `src/pgen/bh_dynamics.hpp` |
| `problem/bh_pair_force` | string | `finite_difference` | `finite_difference` uses the same finite-difference force as the gas source term. `analytic` uses the analytic derivative of the softened potential and also selects the analytic momentum for the external BH source. Only `finite_difference` and `analytic` are accepted. | `analytic` requires `bh_reciprocal_force = true` | `src/pgen/tde_external.cpp`, `src/pgen/bh_force_pair.hpp` |
| `problem/external_bh_dt_factor` | real | `0.5` | Safety factor of the time-step limit from the BH acceleration, `dt <= factor * sqrt(dx / a_BH)`. `0` switches the limit off. Must be `>= 0`. | | `src/srcterms/srcterms.cpp`, `src/srcterms/srcterms_newdt.cpp` |

The BH state is written into every restart file and snapshot header
(`bh_live_x`, `bh_live_y`, `bh_live_z`, `bh_live_vx`, `bh_live_vy`, `bh_live_vz` for the
simulation frame, and `bh_live_inertial_*` and `bh_live_a*` for the inertial state and the
last gas acceleration).  A restart restores them (`src/pgen/tde_external.cpp`).  Every remap also
restores them from the source restart file, in the initial remap and in each settle pass, so
the position, velocity, and acceleration of the live BH carry through a remap and through the
frame conversion (`src/pgen/tde_external.cpp`).  With `bh_reciprocal_force` the cumulative gas
impulse and its reconciliation residual are stored as well (`bh_pair_gas_impulse_*`,
`bh_pair_impulse_residual`, `src/pgen/tde_external.cpp`).  Do not set these keys by hand.

Two limits of the setup.  Without the reciprocal force, the BH feels the self-gravity of
the gas, sampled from the multigrid potential, so it needs `gravity/self_gravity = true`
(the run stops if there is no valid potential sample at the BH position,
`src/pgen/tde_external.cpp`).  The reciprocal force needs LAT, so the first four steps of the chain do
not use it.

## 4. Moving frame, softening, and excision

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/use_translating_frame` | bool | `false` | `true` evolves in the non-rotating frame that translates with the dense gas (steps 1 to 3). `false` uses a frame at rest, which is the initial rest frame of the BH at the conversion, and the live BH moves in it. On a fresh start without `<remap>` it is refused, so the value `false` is only valid together with a remap or restart. When a remap source was written in a translating frame, `false` performs the Galilean conversion to the BH frame (step 4). | `src/pgen/tde_external.cpp` |
| `problem/frame_rho_min` | real | `0.0` | Only gas with `rho` above this value enters the mass-weighted BH acceleration that sets the frame acceleration. The example decks use `1.0e-9`. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/bh_softening` | real | `1.0e-2` | Softening length `r_soft` of the BH potential `Phi = -G M / sqrt(r^2 + r_soft^2)`. Must be `>= 0`. The example decks use `5.0e-3` (0.01 Rsun). | `src/pgen/tde_external.cpp` |
| `problem/bh_grav_rho_min` | real | `0.0` | Density above which the BH pull acts fully. Between `hydro/dfloor` and this value the coupling ramps smoothly from 0 to 1 in log density, so floor-density gas does not feel the BH. The value `0` or a value at or below `dfloor` gives a sharp cut. The example decks use `1.0e-9`. Must be `>= 0`. | `src/pgen/tde_external.cpp`, `src/srcterms/srcterms.cpp`, `src/utils/gravity_weight.hpp` |
| `problem/bh_inner_boundary` | bool | `true` | Enables the excised sink region around the BH. Cells inside `bh_excise_radius` are reset to the floor state with zero momentum, and only inward flux is accepted. | `src/pgen/tde_external.cpp` |
| `problem/bh_excise_radius` | real | `0.5` | Radius of the excised region. Must be `> 0` if `bh_inner_boundary = true`. The example decks use `0.2` (0.4 Rsun). | `src/pgen/tde_external.cpp` |

The excision radius is also stored in each output file header.  `athenak_rt` and the
analysis scripts read it from there (see
[`outputs_and_analysis.md`](outputs_and_analysis.md)).

## 5. Adaptive mesh refinement

The keys below add TDE-specific refinement requests on top of the standard AMR
criteria.  The generator installs its user refinement function whenever `bh_max_amr`,
`unbound_amr`, or any `stream_shell_*` key is active (`src/pgen/tde_external.cpp`).
The separate [`<tde_amr>` scheme](#tde_amr-stream-following-refinement) replaces all of
them, and [`amr.md`](amr.md) is its usage guide.
Each of the rules below that applies to a block gives a target level, and the block is
moved toward the finest of them: refined if it is coarser, derefined if it is finer and
no standard criterion asks to refine it.  With stream shells on, a block that no rule
selects is derefined.  The example decks
combine it with `<amr_criterion0>` (a density criterion) and a user criterion,
`<amr_criterion1> method = user`.

### BH neighborhood and unbound gas

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/bh_max_amr` | bool | `false` | Target level `max_level - bh_max_amr_level_offset` for every block that touches the excision sphere (or contains the BH). | `src/pgen/tde_external.cpp` |
| `problem/bh_max_amr_level_offset` | int | `0` | Offset below the finest level for the request above. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/unbound_amr` | bool | `false` | Target level `max_level - unbound_amr_level_offset` for blocks in which at least a fraction `unbound_amr_fill_frac` of the cells hold gas unbound from the BH (positive kinetic plus softened BH potential energy, and `rho > unbound_amr_rho_min`). | `src/pgen/tde_external.cpp` |
| `problem/unbound_amr_level_offset` | int | `3` | Offset below the finest level for unbound gas. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/unbound_amr_rho_min` | real | `-1.0` | Density threshold for the unbound-gas test. `-1` chooses `max(hydro/dfloor, frame_rho_min, bh_grav_rho_min)`. Other negative values are refused. | `src/pgen/tde_external.cpp` |
| `problem/unbound_amr_fill_frac` | real | `0.01` | Minimum fraction of the cells of a block that must pass the unbound-gas test. Must be in `[0, 1]`; at least one cell is always required. | `src/pgen/tde_external.cpp` |

### Stream shells

Stream-shell refinement follows the dense stream in spherical shells around the BH.  It
is switched on automatically when any `stream_shell_*` key (including the tier keys
below) is present.  For each shell the peak density is found, and blocks near that peak
get the shell's target level, `max_level - level_offset`.  Blocks whose
maximum density lies below `stream_shell_derefine_dfloor_mult * dfloor` are derefined.

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/stream_shell_dr` | real | `-1.0` | Shell thickness for the innermost zone. A negative value chooses four block widths at the target level. `0` is refused. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_level_offset` | int | `0` | Target level offset for the innermost zone. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_r_max` | real | `0.0` | Outer radius of the last zone. `0` means up to the box half-diagonal. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_xsplit_radius` | real | `0.0` | Inside this radius the left and right shell peaks (relative to the BH) are tracked separately. `0` disables the split. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_rho_frac` | real | `0.5` | Fraction of the shell peak density above which a cell counts as part of the stream. Must satisfy `0 < value <= 1`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_fill_frac` | real | `0.05` | Minimum fraction of a block's cells that must be covered to select the block away from the shell peak. Must satisfy `0 < value <= 1`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_derefine_dfloor_mult` | real | `100.0` | Blocks with `rho_max` below this multiple of `hydro/dfloor` are derefined. Must be `>= 0`. | `src/pgen/tde_external.cpp` |

### Stream-shell tiers

Up to four tiers (`N = 1..4`, `kMaxStreamLevelTiers` at `src/pgen/tde_external.cpp`) change the target level,
thickness, and thresholds beyond a given radius.  Tier radii must increase with `N`.
Each tier needs both its radius and its offset.  An omitted `rho_frac_N` or `fill_frac_N`
takes the zone-0 value, while an omitted `stream_shell_dr_N` is automatic (four block
widths at the tier's target level), not `stream_shell_dr`.

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `problem/stream_shell_level_radius_N` | real | required | Inner radius of tier `N`. Must exceed the previous tier's radius. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_level_offset_N` | int | required | Level offset of tier `N`. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_dr_N` | real | `-1.0` | Shell thickness in tier `N`. Negative means automatic. `0` is refused. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_rho_frac_N` | real | `stream_shell_rho_frac` | Density fraction for tier `N`. | `src/pgen/tde_external.cpp` |
| `problem/stream_shell_fill_frac_N` | real | `stream_shell_fill_frac` | Fill fraction for tier `N`. | `src/pgen/tde_external.cpp` |

The FID setup of step 3 and 4 uses `stream_shell_dr = 0.5`, `rho_frac = 0.5`,
`xsplit_radius = 60`, tier 1 from `r = 20` with offset 1, tier 2 from `r = 180` with
offset 2, and `stream_shell_r_max = 300`.  Step 5 uses `xsplit_radius = 40`,
`rho_frac = 0.9`, tier 1 from `r = 10`, tier 2 from `r = 50`, and
`stream_shell_r_max = 200`.

### `<tde_amr>`: stream-following refinement

The `<tde_amr>` block replaces the BH, unbound-gas and stream-shell rules above with one
scheme that follows the local density spine of the stream (`src/pgen/tde_amr.hpp`, read
in `src/pgen/tde_external.cpp`).  It is opt-in.  A deck without a `<tde_amr>` block runs
the legacy logic, bitwise unchanged.

With the block present, the problem hook is the only source of refinement flags.  The
generator stops with an error in the following cases.

* A generic `<amr_criterion>` block has a `method` other than `user`.  Only
  `method = user` blocks are accepted, and at least one must exist, normally
  `<amr_criterion0>`.
* `problem/bh_max_amr`, `problem/unbound_amr` or any `problem/stream_shell_*` key is set.
* `<mesh_refinement>/refinement` is not `adaptive`.

All levels below are logical levels counted from the root, and an offset is counted down
from the finest level, `max_level = num_levels - 1`.  Every cell gets a target level.  A
block is refined if its level is below the largest target of its cells (nominal
thresholds) and derefined if its level is above the largest target evaluated with the
relaxed thresholds (see Hysteresis).  A block in between keeps its level.

**Detection.**  The cells are binned by the distance `r` to the BH, in `nbins_r`
logarithmic bins, and by the azimuth in the orbital plane, in `nbins_phi` bins.  A bin is a
shell sector of every height.  The orbital plane is the plane through the BH normal to
the initial `r x v` of the star.  The maximum density of a bin is the local spine density
`rho_ref`.  A cell with `rho > rho_min` is a stream cell if `rho_ref >= strand_frac *`
(the largest `rho_ref` at its radius).  This gate removes the bins that the stream has
left.  A stream cell with `q = rho / rho_ref >= spine_frac` is a core cell.  A stream cell
below it loses one level per factor `envelope_factor` of `q` below `spine_frac`, up to
`envelope_levels`, and is the same as a background cell below that.  The flanks of the
stream are therefore resolved one level below the spine by default.  A core or envelope
cell takes the finest region of the table below that it matches, and an envelope cell
sits its drop below that level.

| Region | Condition | Level |
|---|---|---|
| Nozzle | `r < nozzle_radius` | `max_level - nozzle_offset` |
| Post-nozzle | `nozzle_radius <= r < post_nozzle_radius` and radial velocity `> 0` | `max_level - post_nozzle_offset` |
| Self-interaction | `nozzle_radius <= r < crossing_r_max` and `-div v >= crossing_div * Omega_K(r)` | `max_level - crossing_offset` |
| Apocentre | `r >= apocentre_frac * r_apo`, with `r_apo` the apocentre of the Kepler orbit of the cell about the BH (bound cells only) | spine ladder level plus `apocentre_boost` |
| Spine ladder | every stream cell | `max_level - spine_offset - n`, with `n` the number of factors `spine_r_per_level` that `r / nozzle_radius` exceeds, at most `spine_offset_max - spine_offset` |
| Core star | `rho >= core_rho` (only if `core_rho` is set) | `max_level - core_offset` |
| Sink | block within the excision radius of the BH (only if `sink_offset >= 0`) | `max_level - sink_offset` |
| Background | everything else | `background_level` |

At equal level the region listed lower in the table is the one reported by `verbose`.
The core star and the sink are exempt from the midplane rule below.  No level goes below
`background_level`.

**Midplane-only blocks.**  With `midplane_only = true` a block may be refined above
`offplane_level` only if it intersects the band `|h| <= max(midplane_hmin, tan(midplane_angle_deg) * R)`
around the orbital plane.  Here `h` is the distance to the plane, and `R` is the in-plane
distance to the BH of the point of the block nearest to the BH.  The rule is applied to the
whole block, so a block that the band touches keeps its full target level, and every
other block stays at or below `offplane_level`.  The whole stream, spine included,
is capped off the band, and these blocks are reported as `offplane`.

**Hysteresis.**  The refinement target uses the nominal thresholds.  The derefinement
target relaxes them: `spine_frac`, `strand_frac` and `crossing_div` are divided by
`hysteresis`, `core_rho_deref` replaces `core_rho`, and every radius bound and the
midplane band are multiplied by `radius_hysteresis` (the band is at least one finest
cell wide).  Each region can only grow under the relaxed evaluation, so a block never
derefines below its nominal target.

**Number of blocks.**  Every offset is counted from `max_level`, so adding a level to
`num_levels` moves every region one level finer.  The blocks of a region then grow by up
to a factor of 8, and by about 4 for the part restricted to the midplane band.  A larger
offset coarsens its region by the same rule.  `nozzle_radius` sets the volume of the
finest region, which grows as its cube, and it also sets the start of the spine ladder, so
a larger radius moves the whole ladder outwards.  Larger `nozzle_offset`,
`post_nozzle_offset`, `crossing_offset` and `spine_offset` lower the levels of their
regions and reduce the blocks quickly, and a smaller `spine_r_per_level` raises the
number of ladder steps.  `mesh_refinement/max_nmb_per_rank` still caps the blocks.

| Key | Type | Default | Meaning | Source |
|---|---|---|---|---|
| `tde_amr/verbose` | bool | `false` | Print one line per AMR check on rank 0, with the blocks per level, the blocks per region that sets their target, and the number of blocks flagged for refinement (`+1`) and derefinement (`-1`). | `src/pgen/tde_external.cpp` |
| `tde_amr/nbins_r` | int | `128` | Number of logarithmic distance bins of the spine table. Must be `>= 1`, and `nbins_r * nbins_phi <= 2^24`. | `src/pgen/tde_external.cpp` |
| `tde_amr/nbins_phi` | int | `64` | Number of azimuth bins in the orbital plane. Must be `>= 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/r_bin_min` | real | `r_t / beta` (the pericenter) | Inner edge of the distance bins. Every cell closer than this falls in the first bin. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/r_bin_max` | real | `-1.0` | Outer edge of the distance bins. A value `<= 0` uses the farthest mesh corner from the BH. Otherwise it must exceed `r_bin_min`. | `src/pgen/tde_external.cpp` |
| `tde_amr/strand_frac` | real | `0.01` | A bin counts as stream only if its spine density is at least this fraction of the largest spine density at its radius. Must be in `[0, 1]`. | `src/pgen/tde_external.cpp` |
| `tde_amr/rho_min` | real | `100 * hydro/dfloor` | Cells at or below this density are never stream cells. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/spine_frac` | real | `0.5` | Minimum `rho / rho_ref` of a core cell. Must satisfy `0 < value <= 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/envelope_factor` | real | `10.0` | Factor of `q` per level lost in the envelope. Must be `> 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/envelope_levels` | int | `1` | Maximum number of levels lost in the envelope. `0` removes the envelope. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/hysteresis` | real | `2.0` | Factor that relaxes the density ratios, the strand gate, `crossing_div` and the default `core_rho_deref` for derefinement. Must be `>= 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/radius_hysteresis` | real | `1.25` | Factor that widens every radius bound and the midplane band for derefinement. Must be `>= 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/nozzle_radius` | real | `2 * r_peri` | Outer radius of the nozzle region, and the base radius of the spine ladder. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/nozzle_offset` | int | `0` | Level offset below `max_level` in the nozzle. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/post_nozzle_radius` | real | `5 * r_peri` | Outer radius of the post-nozzle region. Must be `>= nozzle_radius`. | `src/pgen/tde_external.cpp` |
| `tde_amr/post_nozzle_offset` | int | `1` | Level offset in the post-nozzle region. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/crossing_div` | real | `3.0` | Compression threshold of the self-interaction region, in units of the local Kepler frequency. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/crossing_r_max` | real | `20 * r_peri` | Outer radius of the self-interaction region. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/crossing_offset` | int | `1` | Level offset in the self-interaction region. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/apocentre_frac` | real | `0.8` | Fraction of the Kepler apocentre beyond which the apocentre region starts. Must be `> 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/apocentre_boost` | int | `1` | Levels of the apocentre region finer than the spine ladder at the same radius. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/spine_offset` | int | `1` | Level offset of the spine ladder inside `spine_r_per_level * nozzle_radius`. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/spine_r_per_level` | real | `2.0` | Factor of the radius per additional ladder offset. Must be `> 1`. | `src/pgen/tde_external.cpp` |
| `tde_amr/spine_offset_max` | int | `1000` | Largest ladder offset, which in effect leaves the ladder uncapped. Must be `>= spine_offset`. | `src/pgen/tde_external.cpp` |
| `tde_amr/background_level` | int | `0` | Level of everything that is not stream, counted from the root. Must be in `[0, max_level - root_level]`. | `src/pgen/tde_external.cpp` |
| `tde_amr/midplane_only` | bool | `true` | Apply the midplane-only block rule. | `src/pgen/tde_external.cpp` |
| `tde_amr/midplane_hmin` | real | `0.0` | Minimum half-width of the midplane band. Must be `>= 0`. | `src/pgen/tde_external.cpp` |
| `tde_amr/midplane_angle_deg` | real | `0.0` | Opening angle of the band, whose half-width is `tan(angle) * R`. Must be in `[0, 90)`. | `src/pgen/tde_external.cpp` |
| `tde_amr/offplane_level` | int | `background_level` | Highest level of a block outside the band. Must be in `[background_level, max_level - root_level]`. | `src/pgen/tde_external.cpp` |
| `tde_amr/core_rho` | real | not set | Density above which a cell is refined as the core star. Without it the core-star region is off. | `src/pgen/tde_external.cpp` |
| `tde_amr/core_rho_deref` | real | `core_rho / hysteresis` | Density below which a core-star cell stops being kept. Must satisfy `0 < value <= core_rho`, and it needs `core_rho`. | `src/pgen/tde_external.cpp` |
| `tde_amr/core_offset` | int | `0` | Level offset of the core star. Must be `>= 0`, and it needs `core_rho`. | `src/pgen/tde_external.cpp` |
| `tde_amr/sink_offset` | int | `-1` | Level offset of the blocks within the excision radius of the BH. `-1` switches the sink rule off. | `src/pgen/tde_external.cpp` |

`r_peri` is the pericenter distance `r_t / beta` of the star.  The generator prints the
resolved values in a `--- TDE AMR <tde_amr> ---` banner at start.  With `verbose = true`
each check prints a line of the form `tde_amr: cycle=... time=... blocks/level 0:n0 1:n1 ...
| regions spine:n ... | flags +1:n -1:n`, where only the regions that set at least one
block target are listed.

## 6. Keys of other blocks used by the TDE decks

These keys belong to the general AthenaK modules.  Only the settings that matter for
the TDE runs are listed, and each module page describes the full set.

### `<hydro>`: equation of state, floors, dual energy

Details: [`../wiki/Tabulated-EOS.md`](../wiki/Tabulated-EOS.md),
[`../wiki/Dual-Energy.md`](../wiki/Dual-Energy.md), [`../eos_tables.md`](../eos_tables.md).

| Key | Type | Default | Meaning | Value in decks | Source |
|---|---|---|---|---|---|
| `hydro/eos` | string | required | `lte_table_chabrier2021_t13_helm_union_prad` selects the composite H/He table with radiation pressure. | as left | `src/eos/lte_table_utils.hpp` |
| `hydro/table` | string | required | Path of the EOS table file. Relative paths are resolved against the working directory, then `$ATHENAK_DATA`, then `<source tree>/data`. | `../chabrier2021_t13_helm_union_prad_640.table` | `src/eos/lte_table_utils.hpp`, `src/utils/data_path.hpp` |
| `hydro/lte_bounds` | string | `error` | `clamp` clamps out-of-range `(rho, T)` to the table edge, `error` stops the run. | `clamp` | `src/eos/lte_table_utils.hpp` |
| `hydro/lte_debug_checks` | bool | `false` | Extra table lookup checks. | `false` | `src/eos/lte_table_utils.hpp` |
| `hydro/tfloor_kelvin` | real | unset | Temperature floor in K. | `1` | `src/eos/lte_table_utils.hpp` |
| `hydro/dfloor` | real | `FLT_MIN` | Density floor. | `1.0e-10` | `src/eos/eos.cpp` |
| `hydro/pfloor` | real | `FLT_MIN` | Pressure floor. | `1.0e-12` | `src/eos/eos.cpp` |
| `hydro/sfloor` | real | `FLT_MIN` | Entropy floor. | `0.0` | `src/eos/eos.cpp` |
| `hydro/cs_ceil` | real | `0.0` | Sound-speed ceiling in code units (`0` is off). | `15.0` | `src/eos/eos.cpp` |
| `hydro/vceil` | real | `FLT_MAX` | Velocity ceiling in code units. | `15.0` | `src/eos/eos.cpp` |
| `hydro/dual_energy` | bool | `false` | Evolves an auxiliary internal energy and uses it in cold, supersonic cells. | `true` | `src/hydro/hydro.cpp` |
| `hydro/dual_energy_eta1` | real | `1.0e-3` | Threshold `eta_1` of the pressure-recovery switch. | `1.0e-3` | `src/hydro/hydro.cpp` |
| `hydro/dual_energy_eta2` | real | `1.0e-1` | Threshold `eta_2` of the auxiliary-energy synchronization. The default differs from the decks and from Table 1 of the paper, so set it explicitly. | `1.0e-4` | `src/hydro/hydro.cpp` |
| `hydro/allow_reinterpretive_restart` | bool | `false` | Overrides the fatal check that compares the EOS and table stored in a restart or remap source with the current deck. The remap decks (steps 2 to 5) set it to `true`. | `true` from step 2 | `src/eos/lte_table_utils.hpp` |
| `hydro/reconstruct`, `hydro/rsolver` | string | none | Reconstruction and Riemann solver. The paper uses PLM and HLLE. | `plm`, `hlle` | core |
| `hydro/fofc` | bool | `false` | First-order flux correction, off in the decks. | `false` | core |

### `<gravity>`: multigrid self-gravity

Details: [`../wiki/Multigrid-Self-Gravity.md`](../wiki/Multigrid-Self-Gravity.md).

| Key | Type | Default | Meaning | Value in decks | Source |
|---|---|---|---|---|---|
| `gravity/self_gravity` | bool | `true` | Enables the multigrid Poisson solver. | `true` | `src/srcterms/srcterms.cpp` |
| `gravity/four_pi_G` | real | required with self-gravity (`1.0` without a `<gravity>` block) | Value of `4 pi G` in code units. Must be `> 0`. The gravity module stores `-1` when the key is omitted, so the run stops. The BH potential uses `G = four_pi_G / (4 pi)`. | `1.0` | `src/pgen/tde_external.cpp`, `src/gravity/mg_gravity.cpp` |
| `gravity/solve_dt` | real | none (see below) | Physical interval between Poisson solves, `Delta t_sg`. Must be `>= 0`. The decks use `0.03`, and the paper compares it with `0.0025`. | `0.03` | `src/gravity/mg_gravity.cpp` |
| `gravity/solve_every` | int | `1` | Solve every N-th cycle. Used only when `solve_dt` is absent or `0`. | not set | `src/gravity/mg_gravity.cpp` |
| `gravity/threshold` | real | `-1.0` | Defect convergence threshold. `0.0` selects automatic convergence control. Either `threshold` or `niteration` must be given. | `0.0` | `src/gravity/mg_gravity.cpp` |
| `gravity/niteration` | int | `-1` | Fixed number of V-cycles. Ignored if `threshold` is given. | not set | `src/gravity/mg_gravity.cpp` |
| `gravity/fmg_ncycle` | int | `1` | Number of full-multigrid cycles at startup. | `1` | `src/gravity/mg_gravity.cpp` |
| `gravity/npresmooth`, `npostsmooth` | int | `1`, and `1` (`2` on AMR meshes) | Smoothing sweeps per level. | `1`, `2` | `src/gravity/mg_gravity.cpp` |
| `gravity/show_defect` | string | `0` | Prints the defect. `true`, `false`, or an interval. | `true` | `src/gravity/mg_gravity.cpp` |
| `gravity/rho_grav_min` | real | `0.0` | Density above which a cell sources the Poisson equation and feels the self-gravity force in full. Between `hydro/dfloor` and this value the coupling ramps from 0 to 1. | `1.0e-9` | `src/gravity/mg_gravity.cpp`, `src/srcterms/srcterms.cpp` |

The core has no default for `gravity/solve_dt`.  Without it (and without
`solve_every`) the potential is solved every cycle.  The value `0.03` in the decks is the
production cadence of the paper, and it is not a default of the code.  With
`time/lat = true` and self-gravity, `gravity/solve_dt > 0` is required, because it
bounds the length of a LAT window (`src/main.cpp`).

### `<remap>`: conservative remap and frame conversion

Details: [`../remap_usage.md`](../remap_usage.md), [`../wiki/Remapping.md`](../wiki/Remapping.md).

| Key | Type | Default | Meaning | Value in decks | Source |
|---|---|---|---|---|---|
| `remap/enable` | bool | `true` when the block exists | Master switch. Without a `<remap>` block nothing is remapped. | `true` | `src/remap/remap.cpp` |
| `remap/source` | string | none | Restart file of the previous step. Required when the remap is enabled. Relative paths are resolved from the directory the run starts in. | `../0N_.../rst/<file>.rst` | `src/remap/remap.cpp`, `src/pgen/tde_external.cpp` |
| `remap/settle_steps` | int | `20` | Number of steps of the refinement-only settling phase before the final projection. Must be `>= 0`. The remap module runs it for every pgen (default `0`); `tde_external` seeds `20`. | `10` | `src/remap/remap.cpp`, `src/pgen/tde_external.cpp` |
| `remap/settle_passes` | int | `1` | Number of settling passes. Must be `>= 0`. | `1` | `src/remap/remap.cpp` |
| `remap/transition_band` | bool | `true` | Fades the state to the floor across the edge of the source domain. | not set | `src/remap/remap.cpp` |
| `remap/band_mode` | string | `auto` | `auto`, `floor`, or `keep`. `auto` selects `floor` for Newtonian gas. | not set | `src/remap/remap.cpp` |
| `remap/copy_output_state` | bool | `true` | Continues the output file numbers of the source run. | not set | `src/remap/remap.cpp` |

A remap happens only on a fresh start (`athena -i`), never on a restart (`athena -r`).
A remap can share a run with `time/lat = true`: the settle steps run without LAT, and LAT starts after the last remap pass.

### `<time>`: integrator and localized adaptive time stepping

Details: [`../wiki/Local-Adaptive-Time-Stepping.md`](../wiki/Local-Adaptive-Time-Stepping.md).

| Key | Type | Default | Meaning | Value in decks | Source |
|---|---|---|---|---|---|
| `time/lat` | bool | off | Master switch for LAT. | `false` in the files, `true` on the command line in step 5b | `src/parameter_input.cpp` |
| `time/lat_levels` | int | `1`, clamped to `[1, 20]` | Number of factor doublings. The largest factor is `2^lat_levels`. | `7` (`f_max = 128`) | `src/driver/driver.cpp` |
| `time/lat_same_level` | bool | `false` | Allows different factors on blocks of the same AMR level. | `true` | `src/mesh/mesh.cpp` |
| `time/lat_same_level_max_ratio` | int | `1` | Largest factor ratio between same-level neighbors, one of 1, 2, 4, 8. | `2` | `src/mesh/mesh.cpp` |
| `time/lat_neighbor_limiter` | string | `all` | `all`, `hybrid`, or `face`. With the external BH source and `lat_same_level = true`, `face` is refused. | `all` | `src/mesh/mesh.cpp`, `src/main.cpp` |
| `time/lat_diagnostics` | bool | `false` | Prints the per-pass bin tables. | `false` | `src/mesh/mesh.cpp` |
| `time/hydro_lat_min_bin_count` | int | `-1`, computed as `4 * nranks` | Smallest bin population before a bin is merged into the next faster one. The default makes the LAT schedule depend on the number of MPI ranks. | not set | see the LAT page |

In addition, `tde_external` limits the LAT factor to 1 for blocks within the excision
radius plus the ghost width of the BH, and for their neighbors (`src/pgen/tde_external.cpp`).

### `<mesh_refinement>`

| Key | Type | Default | Meaning | Value in decks | Source |
|---|---|---|---|---|---|
| `mesh_refinement/refinement` | string | none | `adaptive`. | `adaptive` | core |
| `mesh_refinement/num_levels` | int | none | Number of levels including the root. | 8 to 10 | core |
| `mesh_refinement/ncycle_check` | int | `1` | Cycles between AMR checks. With LAT use a value at least as large as a LAT window. | `10` to `512` | `src/pgen/tde_external.cpp` |
| `mesh_refinement/refinement_interval` | int | `5` | Minimum cycles between refinements. | `10` to `512` | `src/pgen/tde_external.cpp` |
| `mesh_refinement/max_nmb_per_rank` | int | none | Cap on the number of MeshBlocks per rank, which bounds device memory. | `520` | `src/mesh/load_balance.cpp` |
| `mesh_refinement/preallocate` | bool | `false` | Allocate the per-MeshBlock device arrays once for `max_nmb_per_rank` blocks. AMR then never reallocates them, and a cap that does not fit in device memory fails at startup. | not set | `src/mesh/mb_storage.hpp` |
| `<amr_criterionN>` | block | none | `method = min_max` with `variable = hydro_w_d`, `value_max`, `derefine_value_max` sets the density criterion. `method = user` hands the block to the TDE refinement function. | see decks | core |
| `<refined_regionN>` | block | none | Minimum refinement level inside a box. | see decks | core |

### `<units>` and `<output>`

`<units>` needs `mass_cgs`, `length_cgs`, `time_cgs`, and `mu`, with the values of the
example decks.  Outputs are described in [`outputs_and_analysis.md`](outputs_and_analysis.md).

## 7. Keys of `tde_external` that are not documented here

The generator also reads a small number of keys for optional couplings that are not
part of the supported workflow.  They are not documented in this guide.
All other reads inside `tde_external.cpp` either belong to the keys above or restore
runtime state that the code writes into restart files itself, for example
`problem/frame_live_*`, `problem/bh_live_*`, `problem/frame_step_*`,
`problem/bh_step_*`, `problem/frame_work_total`, and `saha_runtime/*`.  Do not set
these by hand.
