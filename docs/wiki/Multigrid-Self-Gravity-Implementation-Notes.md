# Multigrid self-gravity: implementation notes

A short tour of how the Poisson solver and its coupling to the fluid are built. Usage, keys and
guidance are in [Multigrid Self-Gravity](Multigrid-Self-Gravity.md).

## How the pieces fit

```
density (hydro or MHD)  --load source-->  multigrid solve  --> phi
                                           (FMG start, V-cycles)   |
                                                                   v
fluid update  <-- momentum and energy source from grad(phi) <------+
```

Once per cycle, or on the `solve_every` or `solve_dt` schedule, the driver loads the conserved
density as the source, solves for the potential `phi`, and the source-term kernel then applies
the force and the gravitational work at every Runge-Kutta stage. There is no gravity task
list. The driver calls the solver directly, and there is no `<multigrid>` input block.

## The solver

1. **Operator.** A 7-point Cartesian Laplacian with per-axis weights, so anisotropic cells work.
   Curvilinear coordinates and GR metrics are not supported.
2. **Smoother.** Red-black Gauss-Seidel with over-relaxation (`omega`). On several ranks an
   optional communication-avoiding Jacobi or Chebyshev variant exists, and the default is plain
   red-black.
3. **Transfers.** Restriction is an 8-cell average. Prolongation is trilinear or tricubic.
4. **Cycle.** A full-approximation-storage V-cycle. The full-multigrid (FMG) start restricts
   the source to the coarsest level, solves there and works upward, which gives the first guess
   of a cold solve.
5. **Stopping.** A negative `threshold` runs a fixed number of V-cycles. A positive one
   iterates until the L2 defect drops below it, with stagnation and divergence monitors and a
   cap of `niteration` checks, or 160 if `niteration` is not positive. With `threshold = 0` an
   automatic controller runs extra cycles until the defect is below 1e-6 of the source norm
   (a fixed value, not a key) or `auto_max_extra_cycles` is reached.
6. **Boundaries.** `zerograd` copies the interior to the ghost cell, `zerofixed` sets the
   potential to zero on the face, and `multipole` matches a multipole expansion of the source.

Every rank holds its own part of the hierarchy. Global sums use `MPI_Allreduce`, and the
finest-to-root transfer is one gather. All MeshBlock-level kernels run on the device. The
octets and, by default for small root grids, the root grid run on the host.

## Coupling into the fluid

`SourceTerms::Gravity` (`src/srcterms/srcterms.cpp`) uses the source form of Mullen, Hanawa and
Gammie (2020). Per direction the momentum change is $\tfrac12\,(\Delta t/dx)\,\rho\,
(\delta\phi_L+\delta\phi_R)$ and the energy change uses the Godunov mass flux in place of
$\rho$. Conservation is therefore limited by the solver residual and the age of the potential.
The dual-energy auxiliary is deliberately not updated. Sink particles are kept out of the
Poisson source and enter through a separate term that uses the same softened potential as the
sink-to-gas reaction.

## When a solve runs

- **Restarts.** The potential is not saved, so the first solve after a restart is a cold FMG
  solve.
- **Cadence.** With `solve_dt` the due time is the next multiple of the interval beyond the
  last solve, so the schedule depends on time alone.
- **Under LAT.** The potential is frozen in a window. The driver shortens the window until
  `lat_fine_dt * factor <= solve_dt`, then solves before the window if the potential is invalid
  or due. If the window would cross the next due time, it refreshes early instead of
  truncating the window. No solve ever runs inside a window, so the source density never mixes
  local times. A matching delayed correction re-applies the gravitational work at the
  window-end reflux, using the frozen potential.
- **Refusals under LAT.** `solve_dt > 0` is required, checked at input parse and in the driver.
  Ion-neutral packs are refused. `lat_same_level` needs `lat_neighbor_limiter` of `all` or
  `hybrid`.

## AMR and SMR

Levels between the root grid and the MeshBlocks are represented by 2x2x2 parent cells
("octets") that live on the host and take part in the V-cycle. A hash of every block's location,
level and owner detects a regrid and rebuilds them. After a regrid the potential is discarded
unless `reuse_phi_after_amr` is set. Under AMR the defaults shift to `npostsmooth = 2` and
tricubic prolongation.

## Energy ledger and coarse/fine symmetry

With `lat_time_centered_work = true` the potential is frozen over each LAT window and solved
again at the window end. The work is then centred in time, so $E_{\rm gas} + \tfrac12
\sum\rho\phi\,dV$ is conserved to round-off, apart from floors, AMR remaps and boundary
transport. Those terms are tallied separately and written as history columns. The
`lat_boundary_flux_diagnostics` and `lat_ledger_debug` keys add detail.

The composite coarse/fine operator is slightly non-symmetric. The measured reciprocity defect is
6e-12 on a uniform mesh, 2.5e-5 on two levels and 6e-5 on three. `mg_fc_symmetric = true` drops
the tangential coupling at coarse/fine faces, makes the operator symmetric to round-off and
changes the self-energy of the test star by 0.14%. The default is `false`.

## Known issues

- Conservation is only as good as the solver residual and the age of the potential.
- A restart agrees with an uninterrupted run only to the solver tolerance.
- With all-Neumann boundaries the solved equation is the Jeans swindle, and the driver only warns.
- The `tde_external` user source terms are not LAT-safe when the translating frame is on.

## Key files

| file | role |
| --- | --- |
| `src/gravity/gravity.cpp` | owns `phi`, checks the LAT ledger options, builds the driver |
| `src/gravity/mg_gravity.cpp` | all `<gravity>` keys, `Solve`, cadence, smoother and defect |
| `src/multigrid/multigrid.cpp` | per-block level arrays, source load, restriction, prolongation |
| `src/multigrid/multigrid_driver.cpp` | hierarchy, octets, V-cycle, FMG and iterative drivers, multipole |
| `src/srcterms/srcterms.cpp` | gravity and sink-gravity source terms |
| `src/driver/driver.cpp` | where the solve is called and every LAT guard |
| `src/hydro/hydro_update.cpp` | delayed gravitational-work correction under LAT |
| `src/mesh/mesh_refinement.cpp` | potential reuse or reset after a regrid |

## Tests

`tst/test_suite/multigrid/`, run from a build directory with
`python ../tst/run_test_suite.py --cpu`, `--mpicpu` or `--gpu`. Thresholds differ between the
platform files. The CPU values are below.

| test | deck | what it checks |
| --- | --- | --- |
| `test_mg_binary_gravity_*` | `binary_gravity.athinput` | Two spheres. Uniform meshes reach defect 1e-9 within 10 V-cycles at average ratio at most 0.0625. SMR needs at most 13 and 0.125. |
| `test_mg_jeans3d_*` | `jeans_wave.athinput` | Stable and unstable Jeans waves converge with resolution, within 1% and 3% of the analytic frequency. `SolveIterative` reaches 1e-8 in at most 4 checks. |
| `test_mg_poisson3d_*` | `selfgravity.athinput`, `selfgravity_mhd.athinput` | $64^3$ hydro and MHD reach 1e-8 within 10 cycles at ratio at most 0.07. The final defect is independent of MeshBlock size. |

The MPI versions add checks that the defect agrees across rank counts and an adaptive Jeans
case on four ranks.
