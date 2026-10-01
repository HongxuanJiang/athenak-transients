# Multigrid Self-Gravity

## Summary

Adding a `<gravity>` block turns on Newtonian self-gravity. A geometric multigrid solver
computes the potential from the gas density on the whole MeshBlock hierarchy, and the force
and the gravitational work are then added to the hydro or MHD update as a source term. It
works with uniform meshes, SMR and AMR, with periodic, isolated (multipole) or fixed
boundaries, and with sink particles and local adaptive time stepping (LAT).

**Use it** for self-gravitating gas: a star that holds itself together, a collapsing cloud,
or a stream that fragments. **Do not use it** for relativistic runs, because it is
Newtonian and has no GR version, or for curvilinear coordinates (anisotropic Cartesian cells
are supported). The cost is dominated by the Poisson solve, not by the coupling. Under LAT the
potential is frozen over a whole window, so the solve runs at most once per window.

## Quick start

An isolated problem (a star in vacuum) with multipole boundaries:

```ini
<gravity>
four_pi_G  = 1.0        # 4 pi G in code units
mg_bc      = multipole  # isolated boundaries
threshold  = 1.0e-6     # solve to this L2 defect
niteration = 20         # cap on the number of defect checks
```

Everything lives in `<gravity>`. There is no `<multigrid>` block, and a deck that has one
aborts. Under LAT also set `solve_dt` (see [Practical guidance](#practical-guidance)).

| key | what it does |
| --- | --- |
| `four_pi_G` | The value of 4 pi G in code units. Required, and must be positive. |
| `mg_bc` | Boundary type for all non-periodic faces: `multipole` (isolated), `zerograd`, `zerofixed`. Periodic faces follow the mesh. |
| `threshold`, `niteration` | When to stop. At least one must be set. A negative `threshold` runs exactly `niteration` V-cycles, a positive one iterates to that defect, and 0 selects an automatic controller. |
| `solve_dt` | Solve interval in code time. Required with LAT. Without LAT it makes the solve cadence time-based. |
| `solve_every` | Solve every N cycles (ignored when `solve_dt > 0`). Reuses the potential in between. |
| `reuse_phi_after_amr` | Keep the potential across a regrid as the first guess, instead of a cold restart of the solve. |

Gas below `rho_grav_min` neither sources nor feels gravity, which is useful for a vacuum
background. `gravity/mask_radius` is the deck-level way to drop the source outside a radius.

## Full parameter table

All keys are in `<gravity>`. Defaults are read from the source.

### Switches and units

| key | type | default | meaning |
| --- | --- | --- | --- |
| `self_gravity` | bool | `true` when `<gravity>` exists | Master switch. Builds the solver and the hydro/MHD source term. |
| `four_pi_G` | Real | -1 | 4 pi G in code units. 0 is fatal. Negative at driver construction is fatal unless a problem generator calls `SetFourPiG`. |
| `rho_grav_min` | Real | 0 (clamped to at least 0) | Density below which a cell has no Poisson source and receives no gravity source term. |
| `reuse_phi_after_amr` | bool | `false` | Keep the AMR-transferred potential as the next initial guess. |

### Convergence and cadence

| key | type | default | meaning |
| --- | --- | --- | --- |
| `threshold` | Real | -1 | Target L2 defect. Below 0 means fixed `niteration` cycles, above 0 means iterate to it, 0 is the auto controller. |
| `niteration` | int | -1 | V-cycle count (`threshold < 0`) or cap on defect checks (`threshold > 0`). One of `threshold` and `niteration` must be set. |
| `solve_every` | int | 1 (at least 1) | Solve on cycles with `ncycle % N == 0`. Ignored if `solve_dt > 0`. |
| `solve_dt` | Real | absent | Solve interval in code time, must be 0 or more. Above 0 enables a time-based cadence and overrides `solve_every` (a message is printed). |
| `full_multigrid` | bool | `true` | Use FMG instead of plain V-cycles. Honoured only for a cold solve (fresh start or restart-remap conversion). A warm solve always uses plain V-cycles. |
| `fmg_ncycle` | int | 1 | V-cycles per FMG level. |
| `defect_check_interval` | int | 2 (at least 1) | V-cycles between defect evaluations. Forced to 1 when `show_defect` is 2 or more. |
| `auto_max_extra_cycles` | int | -1 (computed as max(6, 3*`fmg_ncycle`), raised for AMR or regrid) | Cycle cap of the auto controller. Read only when `threshold == 0`. |
| `warm_final_niter` | int | 4 (at least 1) | V-cycles on the last stage of a warm cycle on an unchanged mesh. Only when `threshold == 0` and `niteration <= 0`. |

### Smoother and grid hierarchy

| key | type | default | meaning |
| --- | --- | --- | --- |
| `omega` | Real | 1.15 | Over-relaxation weight of the smoother. |
| `npresmooth` | int | 1 (at least 1) | Pre-smoothing sweeps per level. |
| `npostsmooth` | int | 2 under AMR, else 1 | Post-smoothing sweeps per level. |
| `coarsest_min_sweeps` | int | 64 (at least 1) | Minimum relaxations on the coarsest root grid. |
| `mg_prolongation` | string | `tricubic` under AMR, else `trilinear` | Prolongation stencil. Any value other than `tricubic` means trilinear. |
| `mg_nghost` | int | 1 | Ghost cells on the multigrid levels. Any value above 1 is fatal on a refined mesh. |
| `mg_fc_symmetric` | bool | `false` | Make the coarse/fine operator symmetric (see the [implementation notes](Multigrid-Self-Gravity-Implementation-Notes#energy-ledger-and-coarsefine-symmetry)). |
| `subtract_average` | bool | `mesh->strictly_periodic` | Remove the mean of source and solution. Forced `true` (with a warning) if no face is `zerofixed` or `multipole`, and forced `false` if multipole is on. |
| `root_on_host` | bool | `true` if the root grid has at most 4096 blocks | Keep the root-grid arrays in host memory. |

### Parallel tuning

| key | type | default | meaning |
| --- | --- | --- | --- |
| `same_exchange_stride` | int | 1 (at least 1) | Do a same-level halo exchange only every N smoother steps. |
| `local_sweeps_per_exchange` | int | 1 (at least 1) | Local red and black sweeps between exchanges. |
| `reduced_exchange_max_edge` | int | 4 (at least 0) | Largest level edge length (cells) for which the reduced exchange applies. |
| `ca_pack_smoother` | string | `rbgs` | Communication-avoiding smoother: `rbgs`, `jacobi`, `chebyshev` or `chebyshev2`. |
| `ca_pack_cheb_lambda_min`, `ca_pack_cheb_lambda_max` | Real | 0.7, 1.95 | Chebyshev eigenvalue bounds. The upper bound is forced above the lower one. |
| `distributed_coarse_solve` | bool | `false` | Only the owner rank runs the root and octet bottom solve and broadcasts the result. Effective only on multiple ranks with refinement. |

### Boundaries and multipole

| key | type | default | meaning |
| --- | --- | --- | --- |
| `mg_bc` | string | `none` | Default for all non-periodic faces: `zerofixed`, `zerograd` (or `outflow`), `multipole`, `none`. `periodic` here is fatal. |
| `ix1_bc`, `ox1_bc`, `ix2_bc`, `ox2_bc`, `ix3_bc`, `ox3_bc` | string | mesh-derived | Per-face override, same vocabulary as `mg_bc`. |
| `mporder` | int | 4 | Multipole order, 2 or 4. Read only when some face is `multipole`. |
| `auto_mporigin` | bool | `true` | Recompute the expansion origin as the source centre of mass at each solve. |
| `nodipole` | bool | `false` | Skip the dipole terms. Fatal together with `auto_mporigin`. |
| `mporigin_x1`, `mporigin_x2`, `mporigin_x3` | Real | required if `auto_mporigin = false` | Fixed expansion origin. |
| `mask_radius` | Real | -1 (off) | Zero the Poisson source outside this radius from the mask origin. |
| `mask_origin_x1`, `mask_origin_x2`, `mask_origin_x3` | Real | 0 | Mask centre. |

### Diagnostics and LAT energy ledger

| key | type | default | meaning |
| --- | --- | --- | --- |
| `show_defect` | string or int | `"0"` | 0 or false is off, 1 or true prints the final defect, 2 or more prints each iteration. |
| `show_timing` | bool | `false` | Print a per-solve phase timing line. |
| `mg_verbose` | int | 0 (at least 0) | Gates exactly one warning: a multipole source with cancelling total mass. |
| `lat_time_centered_work` | bool | `false` | LAT only. Time-centred gravitational work for an exact energy ledger (see the implementation notes). |
| `lat_boundary_flux_diagnostics`, `lat_ledger_debug` | bool | `false` | Extra ledger columns and per-window printing. The first requires `lat_time_centered_work`. |
| `reciprocity_test` | bool | `false` | Run the reciprocity diagnostic of the coarse/fine operator at startup. Its tuning keys (`reciprocity_*`) are for developers. |

### Related keys outside `<gravity>`

| key | default | meaning |
| --- | --- | --- |
| `hydro_srcterms/self_gravity` | absent | Legacy alias. If `true` without `gravity/self_gravity = true`, the run aborts. |
| `problem/external_bh_gravity_source` | `true` for `tde_external` | Analytic softened black-hole potential, added in the same kernel but separate from the solved potential. |
| `problem/bh_grav_rho_min` | 0 | Density floor for the external black-hole branch. |
| `sink_particles/newton_g` | `four_pi_G/(4 pi)` | G for sink potentials when `<units>` is absent. |

## How it works

The solver finds the potential phi from the Poisson equation $\nabla^2\phi = 4\pi G\rho$,
with the source taken from the conserved density of whichever fluid is active.

```
fine grid   smooth --restrict--> smooth --restrict--> ... coarsest: solve
fine grid   smooth <-prolong---- smooth <-prolong---- ... (correct upwards)
```

- **Multigrid.** Smoothing (red-black Gauss-Seidel with over-relaxation) removes short
  wavelengths on each level, restriction moves the residual to a coarser grid, and the
  coarse correction is interpolated back up. A full-multigrid (FMG) start builds a good
  first guess from the coarsest level up. Later solves start from the previous potential.
- **Refinement.** Levels between the root grid and the MeshBlocks that AMR or SMR creates are
  handled on the host as small 2x2x2 groups of cells (octets).
- **Stopping.** With a negative `threshold` the solver does a fixed number of V-cycles. With a
  positive one it iterates until the L2 defect drops below it. With `threshold = 0` an
  automatic controller runs a preset burst and then extra cycles until the defect falls
  below 1e-6 of the source norm, or until `auto_max_extra_cycles` is reached.
- **Boundaries.** `zerograd` is a zero-gradient condition, `zerofixed` sets the potential to
  zero on the face, and `multipole` matches the potential to a multipole expansion of the
  source (quadrupole for `mporder = 2`, hexadecapole for 4) about the centre of mass.
- **Coupling.** The force is added as a source term in the form of Mullen, Hanawa and Gammie
  (2020): momentum from the potential gradient and energy from the potential gradient times
  the Godunov mass flux. Conservation is therefore limited by the multigrid residual and by
  how old the potential is. The dual-energy auxiliary field is deliberately not updated here.
- **When it solves.** Once per cycle, or on the `solve_every` or `solve_dt` schedule, with the
  potential reused in between. With `solve_dt` the schedule depends on time alone, so a
  restarted run solves on the same cycles as an uninterrupted one.
- **Under LAT.** The potential is frozen inside a window. No Poisson solve ever runs inside a
  window, so the source density is never loaded from blocks at mixed local times.
  A solve happens before a window, or at a window end when the run stops there.

## Practical guidance

### Choosing the solver settings

- Start with a fixed `niteration` (leave `threshold` negative) for a reproducible cost, or a
  positive `threshold` for a controlled accuracy.
- Leave `full_multigrid = true`. It matters only for the first solve and for restarts.
- Under AMR the defaults already shift to `npostsmooth = 2` and tricubic prolongation.
- `distributed_coarse_solve` stays off by default even under AMR.

### Boundary conditions

- Use `mg_bc = multipole` for an isolated system and a periodic mesh for a box.
- With all-Neumann boundaries the solved equation is the Jeans swindle, not the isolated
  problem. The driver only warns.
- `mporder` must be 2 or 4. With `auto_mporigin = false` give the three `mporigin_x*` keys.

### LAT

- LAT with self-gravity requires `gravity/solve_dt > 0`, because the potential is frozen
  inside a window and this key bounds the window length. Checked at input parse and in the
  driver, with the message "time/lat with self-gravity uses a frozen potential inside each
  LAT window and requires gravity/solve_dt > 0 to bound the window length."
- `time/lat_same_level = true` with self-gravity or the analytic black hole requires
  `lat_neighbor_limiter` to be `all` or `hybrid`.
- Ion-neutral (two-fluid) packs are refused, because there is no single gravity source
  density. LAT with self-gravity needs exactly one of hydro and MHD.
- See [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping).

### Sinks and point masses

Sink particles are deliberately absent from the Poisson source. Their potential enters through
a separate source term that uses the same softened potential as the sink-to-gas reaction, so
the pair force is equal and opposite cell by cell. `<sink_particles>/create = true` requires a
`<gravity>` block. `gravity/mask_radius` is the deck-level counterpart, which zeroes the
source outside a radius.

### Restarts

The potential is not written to restart files. The first solve after a restart is a cold FMG
solve, so a restart is not bitwise identical. It agrees with the uninterrupted run only to the
multigrid tolerance. With a `solve_dt` cadence and no LAT, the first stage after a restart
differs by the frozen-potential error of one cadence interval (a few 1e-3 in the kinetic energy
of a Jeans test), and the runs stay that far apart. Under LAT the window-start solve makes
them agree to the solver tolerance.

### Requirements and refusals

- MeshBlocks must be logically cubic and a power of two in size. Both are fatal.
- Self-gravity needs hydro or MHD state, or the solve aborts.
- One of `threshold` and `niteration` must be set, and `four_pi_G` must be positive by driver
  construction.
- `auto_target_defect` is not a key, and setting it is fatal. `auto_max_extra_cycles` and
  `warm_final_niter` have no effect unless `threshold == 0`.
- There is no flux correction in multigrid, so there is no interaction with FOFC.
- A root grid that cannot be coarsened to one cell, or has more than 100 coarsest cells, prints a
  warning and falls back to an iterative bottom solve.

### Checking that it works

- Set `show_defect = 1` for the final defect of each solve, or 2 for every iteration.
- `show_timing = true` prints the time per phase.
- The `grav_phi` output variable writes the potential (plus the analytic black-hole
  potential if enabled), and is valid only after a solve.

### Other notes

- `four_pi_G` is a bare code-unit number. The `<units>` module does not feed it, and only
  `sink_particles` consults `punit->grav_constant()`.
- Problem generators may move the black hole by sampling the gradient of the self-gravity
  field only.
- `tde_external` uses self-gravity to hold the star together while the black hole acts through
  a user source term. Its user source terms are not LAT-safe when the translating frame is on.
- Decks that switch self-gravity on: `inputs/TDE_examples/*` (5 decks) and
  `inputs/sink_particles/creation.athinput`.

## Further reading

- [Implementation notes](Multigrid-Self-Gravity-Implementation-Notes): discretisation, the
  coupling kernel, cycle details, LAT scheduling mechanics, AMR octets, energy ledger and
  coarse/fine symmetry, code map, known issues, tests.
- Mullen, Hanawa and Gammie (2020): the flux-consistent gravitational-work source form.
- `docs/lat_implementation_note.tex`, section "Self-Gravity" and the constraint list: the
  authoritative statement of LAT and gravity synchronization.
