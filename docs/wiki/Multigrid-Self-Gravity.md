# Multigrid Self-Gravity

## Summary

Adding a `<gravity>` block turns on Newtonian self-gravity. A geometric multigrid solver
computes the gravitational potential from the gas density on the whole MeshBlock hierarchy.
The force and its work are then added to the hydro or MHD update as a source term. The module
works on uniform meshes, SMR and AMR, with periodic, isolated (multipole) or fixed-potential
boundaries. For hydro runs it also works with sink particles and with local adaptive time
stepping (LAT).

**Use it** for self-gravitating gas: a star that holds itself together, a collapsing cloud,
or a stream that fragments. **Do not use it** for relativistic runs (it is Newtonian and has no
GR version), for curvilinear coordinates (anisotropic Cartesian cells are supported), or on 1D
and 2D meshes (the solver is 3D only). The cost is dominated by the Poisson solve, not by the
coupling. Under LAT the potential is frozen over a window of fine steps and is solved only
between windows.

## Quick start

An isolated problem (a star in vacuum) with multipole boundaries:

```ini
<gravity>
four_pi_G  = 1.0        # 4 pi G in code units, so G = 1/(4 pi)
mg_bc      = multipole  # isolated system
threshold  = 1.0e-6     # stop when the L2 defect is below this
niteration = 20         # but do at most 20 defect checks
```

Everything is set in `<gravity>`. There is no `<multigrid>` block, and a deck that has one
aborts. Under LAT also set `solve_dt` (see [LAT](#lat)). The keys you will use most:

| key | what it does |
| --- | --- |
| `four_pi_G` | The value of $4\pi G$ in code units. Required, and must be positive. |
| `mg_bc` | Boundary condition of the potential on all non-periodic faces: `multipole` (isolated), `zerograd` or `zerofixed`. Periodic faces follow the mesh. |
| `threshold`, `niteration` | When to stop the iteration. Set at least one. A negative `threshold` runs exactly `niteration` V-cycles, a positive one iterates until the defect is that small, and 0 selects an automatic controller. |
| `solve_dt` | Solve the potential once per `solve_dt` of code time, at the first cycle at or after each multiple of `solve_dt`. Required with LAT. Without LAT it makes the solve schedule time-based. |
| `solve_every` | Solve only every N cycles (ignored when `solve_dt > 0`). The potential is reused in between. |
| `reuse_phi_after_amr` | After a regrid, start from the transferred potential instead of a cold restart of the solve. |
| `rho_grav_min` | Density at which gas is fully coupled to gravity. Gas below it is coupled less or not at all, which is useful for a vacuum background (details in the table below). |

### What is solved, and in which units

The solver finds the potential $\phi$ from the Poisson equation $\nabla^2\phi = 4\pi G\,\rho$,
where $\rho$ is the gas density, and the gas feels the acceleration $-\nabla\phi$. The only
number the solver reads is the product $4\pi G$ in your code units, which is `four_pi_G`. There
is no separate input for $G$. For $G = 1$ set `four_pi_G = 12.566370614359172`. The value
`four_pi_G = 1.0` used in the TDE decks means $G = 1/(4\pi)$.

Two other places need $G$ itself, and both compute it as $G$ = `four_pi_G`/(4 pi). One is the
analytic black-hole potential of the `tde_external` problem generator. The other is the sink
particles, but only when the deck has no `<units>` block. With a `<units>` block the sinks take
$G$ from the unit system instead, and nothing checks that this agrees with `four_pi_G`. If you
use `<units>`, set `four_pi_G` to $4\pi$ times that $G$ yourself.

## Full parameter table

All keys are in `<gravity>` unless the table says otherwise. Defaults are the values in the
source.

### Switches and units

| key | type | default | meaning |
| --- | --- | --- | --- |
| `self_gravity` | bool | `true` when `<gravity>` exists | Master switch. If `false`, the solver and the source term are not built. |
| `four_pi_G` | Real | -1 | The value of $4\pi G$ in code units. The default means "not set": a negative value is fatal, and so is 0. |
| `rho_grav_min` | Real | 0 (negative values act as 0) | Density at which gas is fully coupled to gravity, both as a source of the potential and as a receiver of the force. Between `hydro/dfloor` and `rho_grav_min` the coupling weight rises smoothly from 0 to 1 (a cubic smoothstep in log density), and below `hydro/dfloor` it is 0. If `rho_grav_min` is at or below `hydro/dfloor`, it is a hard cut at `rho_grav_min`. The default 0 couples every cell. |
| `reuse_phi_after_amr` | bool | `false` | Keep the AMR-transferred potential as the first guess of the next solve instead of discarding it. |

### Convergence and cadence

| key | type | default | meaning |
| --- | --- | --- | --- |
| `threshold` | Real | -1 | Target L2 defect. Below 0 means a fixed number of V-cycles, above 0 means iterate to that defect, and 0 means the automatic controller (see [How it works](#how-it-works)). |
| `niteration` | int | -1 | With `threshold < 0`: the number of V-cycles, which must be at least 1. With `threshold > 0`: the cap on defect checks (160 if `niteration` is not positive). With `threshold = 0`: the length of the first burst of V-cycles (`fmg_ncycle` + 1 if not set). One of `threshold` and `niteration` must be set. |
| `solve_every` | int | 1 (at least 1) | Solve on cycles with `ncycle % N == 0`. Ignored if `solve_dt > 0`. |
| `solve_dt` | Real | absent | Solve interval in code time, 0 or more. Above 0 it enables a time-based schedule and overrides `solve_every` (a message is printed if both are set). |
| `full_multigrid` | bool | `true` | Start from a full-multigrid (FMG) ramp instead of plain V-cycles. Used only for a cold solve (a fresh start or a restart). A solve that starts from a valid potential always uses plain V-cycles. |
| `fmg_ncycle` | int | 1 | V-cycles per level of the FMG ramp. |
| `defect_check_interval` | int | 2 (at least 1) | V-cycles between defect evaluations. Forced to 1 when `show_defect` is 2 or more. |
| `auto_max_extra_cycles` | int | -1, meaning max(6, 3*`fmg_ncycle`), raised to at least 12 with refinement and to 32 after a regrid | Cap on the extra cycles of the automatic controller. Read only when `threshold = 0`. |
| `warm_final_niter` | int | 4 (at least 1) | V-cycles on the last stage of a warm cycle when the mesh has not changed. Used only when `threshold = 0` and `niteration` is not positive. |

### Smoother and grid hierarchy

| key | type | default | meaning |
| --- | --- | --- | --- |
| `omega` | Real | 1.15 | Over-relaxation weight of the smoother. |
| `npresmooth` | int | 1 (at least 1) | Pre-smoothing sweeps per level. |
| `npostsmooth` | int | 2 with SMR or AMR, else 1 (at least 1) | Post-smoothing sweeps per level. |
| `coarsest_min_sweeps` | int | 64 (at least 1) | Minimum number of relaxations on the coarsest root grid. |
| `mg_prolongation` | string | `tricubic` with SMR or AMR, else `trilinear` | Interpolation used to prolong the correction. Any value other than `tricubic` means trilinear. |
| `mg_nghost` | int | 1 | Ghost cells on the multigrid levels. A value above 1 is fatal on a refined mesh. Its only use is a multi-rank optimisation on uniform meshes. |
| `mg_fc_symmetric` | bool | `false` | Make the coarse/fine operator symmetric by dropping the tangential terms of the fine ghost value. Without it the composite operator is slightly non-symmetric, and the LAT ledger records the effect as `E_asym`. On the octet levels this changes only the convergence rate, not the solution. |
| `subtract_average` | bool | `true` on a fully periodic mesh, else `false` | Remove the mean of the source and the solution. Forced to `true` (with a warning) when no face is `zerofixed` or `multipole`, and forced to `false` when a face is `multipole`. |
| `root_on_host` | bool | `true` if the root grid has at most 4096 blocks | Keep the root-grid arrays in host memory. |

### Parallel tuning

The defaults are right for most runs. These keys trade extra local work for fewer halo
exchanges on several ranks.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `same_exchange_stride` | int | 1 (at least 1) | Switch for the reduced same-level halo exchange. Any value above 1 turns it on (the size of the value does not matter). It applies only on several ranks, only to coarser MeshBlock levels, and not to the first solve after a regrid. |
| `local_sweeps_per_exchange` | int | 1 (at least 1) | With the reduced exchange on: red and black sweeps done between two exchanges. |
| `reduced_exchange_max_edge` | int | 4 (at least 0) | With the reduced exchange on: it is used only on levels whose block edge is at most this many cells. |
| `ca_pack_smoother` | string | `rbgs` | Smoother used while the reduced exchange is active on several ranks: `rbgs` (ordinary red-black Gauss-Seidel), `jacobi`, `chebyshev` or `chebyshev2`. The last two are the same two-stage Chebyshev smoother. Any other value is silently treated as `rbgs`. It has no effect when the reduced exchange is off. |
| `ca_pack_cheb_lambda_min`, `ca_pack_cheb_lambda_max` | Real | 0.7, 1.95 | The interval of eigenvalues that the two Chebyshev relaxation weights are tuned to damp. They are dimensionless: the eigenvalues of the operator scaled by its diagonal lie between 0 and 2. Used only by the Chebyshev smoother (`omega` is not used by it). If the upper value is not above the lower one, it is reset to the lower value plus 1. |
| `distributed_coarse_solve` | bool | `false` | Only the owner rank solves the root and octet levels and broadcasts the result. Effective only on several ranks with refinement, and only worth it at rank counts where repeating the coarse levels on every rank is no longer cheap. |

### Boundaries and multipole

| key | type | default | meaning |
| --- | --- | --- | --- |
| `mg_bc` | string | `none` | Default for all non-periodic faces: `zerofixed`, `zerograd` (or `outflow`), `multipole` or `none`. `none` keeps the mesh-derived choice, which is `zerograd` on every non-periodic face. `periodic` here is fatal. |
| `ix1_bc`, `ox1_bc`, `ix2_bc`, `ox2_bc`, `ix3_bc`, `ox3_bc` | string | from `mg_bc` and the mesh | Per-face override with the same words as `mg_bc`. A value that is not recognised is ignored. |
| `mporder` | int | 4 | Multipole order, 2 or 4. Read only when some face is `multipole`. |
| `auto_mporigin` | bool | `true` | Recompute the expansion origin as the source's centre of mass at each solve. |
| `nodipole` | bool | `false` | Skip the dipole terms. Fatal together with `auto_mporigin`. |
| `mporigin_x1`, `mporigin_x2`, `mporigin_x3` | Real | required if `auto_mporigin = false` | Fixed expansion origin. |
| `mask_radius` | Real | -1 (off) | Zero the Poisson source outside this radius from the mask origin. It removes only the source: the gas inside still feels the potential. |
| `mask_origin_x1`, `mask_origin_x2`, `mask_origin_x3` | Real | 0 | Centre of the mask. |

### Diagnostics and LAT energy ledger

| key | type | default | meaning |
| --- | --- | --- | --- |
| `show_defect` | string or int | `"0"` | 0 or `false` is off, 1 or `true` prints the final defect of each solve, 2 or more prints every iteration. Any other value is fatal. |
| `show_timing` | bool | `false` | Print one timing line per solve with the time of each phase. |
| `mg_verbose` | int | 0 (at least 0) | Gates exactly one warning: a multipole source whose total mass cancels. It is not a general verbosity switch. |
| `lat_time_centered_work` | bool | `false` | LAT only. Solves again at each window end and adds $-\tfrac{1}{2}\,\Delta\rho\,\Delta\phi$ to the energy of each cell, so that the gas energy plus $\tfrac{1}{2}\sum\rho\phi\,dV$ changes only through floors, AMR remaps, boundary transport and the operator asymmetry. These terms are tallied in double precision and carried through restarts as `<gravity>` metadata. The `tde_external` problem generator writes them as history columns (`W_cent`, `E_remap`, `E_floor`, `E_asym`, and with `lat_boundary_flux_diagnostics` also `B_mass`, `B_E`, `B_Wself`, `B_Wbh`). Its requirements are in [Requirements and refusals](#requirements-and-refusals). |
| `lat_boundary_flux_diagnostics`, `lat_ledger_debug` | bool | `false` | Extra ledger columns, and a per-window printout on rank 0. The first requires `lat_time_centered_work`. |
| `reciprocity_test` | bool | `false` | Run a diagnostic of the coarse/fine operator's symmetry at startup, then stop before any evolution. Its tuning keys (`reciprocity_*`) are for developers. |

### Related keys outside `<gravity>`

| key | default | meaning |
| --- | --- | --- |
| `hydro_srcterms/self_gravity` | absent | Legacy key. It is ignored when a `<gravity>` block exists. Without a `<gravity>` block, `true` aborts the run. |
| `problem/external_bh_gravity_source` | `true` for `tde_external`, else `false` | Analytic softened black-hole potential, applied by the same source-term kernel but separate from the solved potential. |
| `problem/bh_grav_rho_min` | 0 | The `rho_grav_min` of the external black-hole branch, with the same ramp above `hydro/dfloor`. |
| `sink_particles/newton_g` | `four_pi_G`/(4 pi) | The value of $G$ for the sink particles. Read only when the deck has no `<units>` block. See [What is solved, and in which units](#what-is-solved-and-in-which-units). |

## How it works

The solver finds the potential $\phi$ that satisfies $\nabla^2\phi = 4\pi G\,\rho$. The density
is the conserved density of whichever fluid is active, multiplied by the coupling weight of
`rho_grav_min`.

```
fine grid   smooth --restrict--> smooth --restrict--> ... coarsest: solve
fine grid   smooth <-prolong---- smooth <-prolong---- ... (correct upwards)
```

- **Multigrid.** A smoothing sweep (red-black Gauss-Seidel with over-relaxation) removes
  short-wavelength error quickly but long-wavelength error slowly. So the solver restricts
  the residual to a coarser grid, where those long wavelengths are short, smooths there, and
  interpolates the correction back up. One trip down and up is a V-cycle. The coarsest grid
  is relaxed many times. A full-multigrid (FMG) start builds a good first guess by solving on
  the coarsest level and working up. Later solves start from the previous potential.
- **Refinement.** The levels between the root grid and the MeshBlocks that AMR or SMR creates
  are handled on the host as small 2x2x2 groups of cells (octets).
- **Stopping.** With a negative `threshold` the solver does exactly `niteration` V-cycles.
  With a positive one it checks the L2 defect after every `defect_check_interval` V-cycles and
  stops when it is below `threshold`, or after `niteration` checks (160 if `niteration` is not
  positive). With `threshold = 0` an automatic controller runs a first burst of V-cycles
  (`niteration` of them if set, else `fmg_ncycle` + 1) and then extra cycles, until the defect
  is below $10^{-6}$ times the L2 norm of the source (never less than $10^{-12}$), until
  `auto_max_extra_cycles` extra cycles have run, or until the defect stops improving. The
  $10^{-6}$ is fixed in the code and is not an input key. A threshold relative to the source
  carries over between problems, while an absolute `threshold` depends on your units.
- **Boundaries.** `zerograd` is a zero-gradient condition and `zerofixed` sets the potential
  to zero on the face. `multipole` sets the potential on the face from a multipole expansion
  of the source (up to the quadrupole for `mporder = 2`, up to the hexadecapole for 4) about
  its centre of mass.
- **Coupling.** The force enters as a source term in the form of Mullen, Hanawa and Gammie
  (2020). The momentum changes by the potential gradient. The energy changes by the same
  potential differences multiplied by the Godunov mass flux through each cell face. Energy
  conservation is therefore limited by the multigrid residual and by the age of the
  potential. The dual-energy auxiliary variable is not changed by this term.
- **When it solves.** By default the potential is solved at every Runge-Kutta stage, because
  the stage density changes. With `solve_every` above 1 or with `solve_dt`, it is solved only
  on the last stage of a due cycle and reused in between. The first solve of a run, and the
  solves after a regrid, always run. With `solve_dt` the due times sit on the grid
  0, `solve_dt`, 2 `solve_dt`, ... of code time, so a restarted run solves on the same cycles
  as an uninterrupted one.
- **Under LAT.** The potential is frozen inside a window, and no Poisson solve runs inside it,
  so the source density is never loaded from blocks at different local times. The driver
  solves before a window, and again at a window end when the run stops there (with
  `lat_time_centered_work` it solves at every window end). Before a window it first shortens
  the window to at most `solve_dt`. If the potential is invalid or due, or the window would run
  past the next due time, it solves first instead of cutting the window.

## Practical guidance

### Choosing the solver settings

- Start with `threshold = 0` (all the shipped TDE decks use it) for an automatic stopping
  rule. Use a fixed `niteration` (with `threshold` negative) for a reproducible cost, or a
  positive `threshold` for a controlled accuracy.
- Leave `full_multigrid = true`. It matters only for the first solve and for restarts.
- With SMR or AMR the defaults already switch to `npostsmooth = 2` and tricubic prolongation.
- Leave the parallel tuning keys at their defaults unless you are tuning performance.
- To keep a vacuum background out of the gravity, set `rho_grav_min` above the background
  density. The TDE decks use `hydro/dfloor = 1e-10` and `rho_grav_min = 1e-9`, so the coupling
  rises from 0 to 1 over that decade.

### Boundary conditions

- Use `mg_bc = multipole` for an isolated system and a periodic mesh for a box.
- With all-Neumann boundaries (the default `zerograd`) the mean of the source is subtracted,
  so the equation solved is the Jeans swindle, not the isolated problem. The driver only warns.
- `mporder` must be 2 or 4. With `auto_mporigin = false` give the three `mporigin_x*` keys.

### LAT

- Set `gravity/solve_dt` to the interval at which the potential should be refreshed. The
  potential is frozen inside a LAT window, so this key also caps the window length. A smaller
  value gives a fresher potential and shorter windows.
- LAT is available for hydro only in this release, so LAT with self-gravity is for hydro runs.
- [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) explains how a window works.
  Its list of
  [supported and refused modules](Local-Adaptive-Time-Stepping#what-is-supported-and-refused)
  applies here too. The conditions that self-gravity adds are in
  [Requirements and refusals](#requirements-and-refusals).

### Sinks and point masses

Sink particles are deliberately absent from the Poisson source. Their potential enters through
a separate source term that uses the same softened potential as the sink-to-gas reaction, so
the force pair is equal and opposite cell by cell. `<sink_particles>/create = true` requires a
`<gravity>` block, and sinks require `<hydro>`. `mask_radius` is the deck-level way to drop the
Poisson source outside a radius.

### Restarts

The potential is not written to restart files, so the first solve after a restart is a cold
FMG solve. A restarted run therefore agrees with the uninterrupted run only to the solver
tolerance, not bit for bit. With a `solve_every` or `solve_dt` cadence and no LAT, the first
stage after a restart also uses a freshly solved potential, where the uninterrupted run would
still use an older one. Under LAT the potential is solved at the first window start. LAT for MHD is not in this release, and the LAT gravity logic reads the hydro source terms only.

### Requirements and refusals

Each of these makes the run exit with an error.

Mesh and setup:

- The mesh is not 3D, or the MeshBlocks are not logically cubic or not a power of two in size.
- The deck has a `<multigrid>` block, which is not a valid block name.
- There is neither a hydro nor an MHD module.
- `<sink_particles>/create = true` without a `<gravity>` block.
- `hydro_srcterms/self_gravity = true` without a `<gravity>` block.

Gravity keys:

- `four_pi_G` is not set, is 0, or is negative.
- Neither `threshold` nor `niteration` is set, or `niteration = 0` with a negative `threshold`.
- `auto_target_defect` is set. It is no longer a key.
- `mg_bc` is `periodic` or an unknown word, or a face is periodic for gravity but not for the
  mesh.
- `mporder` is not 2 or 4, or both `auto_mporigin` and `nodipole` are `true`.
- `mg_nghost` is above 1 on a refined mesh.
- `show_defect` is not `true`, `false` or an integer, or `solve_dt` is negative.

Time stepping and LAT:

- `time/lat` with self-gravity and no positive `gravity/solve_dt`. The potential is frozen
  inside a window, and this key bounds the window length.
- `time/lat_same_level = true` with `lat_neighbor_limiter = face`, with self-gravity or with
  the analytic black hole. The limiter must be `all` or `hybrid`.
- `time/lat` with an `<mhd>` block, because LAT for MHD is not in this release. LAT also
  refuses other modules, such as an ion-neutral pack (see the LAT page linked above).
- `lat_boundary_flux_diagnostics = true` without `lat_time_centered_work = true`.
- `lat_time_centered_work = true` unless all of these hold: `time/lat = true`,
  `rho_grav_min` is at most `hydro/dfloor`, there is no positive `mask_radius`, there is no
  `<sink_particles>` block, and either `solve_dt > 0` or `solve_every = 1`. At run time it
  also needs hydro with an energy equation (not isothermal), because the correction is added
  to the gas energy.

Two root-grid conditions only warn. A root grid that cannot be coarsened to one cell prints a
warning, and its coarsest level is then solved iteratively, which is less efficient. A root
grid with more than 100 coarsest cells prints a warning about the cost of the coarsest solve.
Separately, `auto_max_extra_cycles` and `warm_final_niter` are silently ignored unless
`threshold = 0`.

### Checking that it works

- Set `show_defect = 1` for the final defect of each solve, or 2 for every iteration.
- `show_timing = true` prints the time per phase.
- The `grav_phi` output variable writes the gravitational potential (the solved potential plus
  the analytic black-hole potential if enabled). Before the first solve the solved part is zero.
  It is also added automatically to the `hydro_u`, `hydro_w`, `mhd_u` and `mhd_w` output
  bundles.
- The regression tests are in `tst/test_suite/multigrid/`. From `tst/`, run
  `python run_test_suite.py --cpu --test test_suite/multigrid`, or use `--mpicpu` or `--gpu`.
  They cover a two-sphere Poisson problem, Jeans waves and 64^3 hydro and MHD solves, with
  checks on convergence rate and on independence from MeshBlock size and rank count. Further
  gravity tests are in `tst/test_suite/nr/`.

### Other notes

- `tde_external` uses the solved potential to hold the star together, and applies the black hole
  as an analytic potential in the same source-term kernel
  (`problem/external_bh_gravity_source`).
- The decks that switch self-gravity on are the five `inputs/TDE_examples/*` decks and
  `inputs/sink_particles/creation.athinput`.

## Further reading

- Mullen, Hanawa and Gammie (2020): the flux-consistent gravitational-work source form.
- [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping): how a window works and how LAT
  and gravity stay synchronized.
