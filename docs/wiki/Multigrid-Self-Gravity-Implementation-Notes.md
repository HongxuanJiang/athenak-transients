# Multigrid self-gravity: implementation notes

Developer-level detail for [Multigrid Self-Gravity](Multigrid-Self-Gravity). Start there for
usage, parameters and guidance.

## Contents

- [Discretisation and solver](#discretisation-and-solver)
- [Coupling into the fluid](#coupling-into-the-fluid)
- [How a solve runs](#how-a-solve-runs)
- [LAT scheduling and refusals](#lat-scheduling-and-refusals)
- [AMR and SMR](#amr-and-smr)
- [Sinks](#sinks)
- [Energy ledger and coarse/fine symmetry](#energy-ledger-and-coarsefine-symmetry)
- [Known issues](#known-issues)
- [Code map](#code-map)
- [Tests](#tests)

## Discretisation and solver

**Equation and operator.** The finest multigrid source is loaded straight from the conserved
density of the active fluid, scaled by $-4\pi G$ (`src/gravity/mg_gravity.cpp`,
`src/multigrid/multigrid.cpp`). The discrete operator is the dx-squared-scaled 7-point
Cartesian Laplacian with per-axis anisotropy weights

$$ L u = \mathrm{diag}\,u_{ijk} - (u_{i\pm1}) - w_y (u_{j\pm1}) - w_z (u_{k\pm1}), \qquad
w_y = (dx/dy)^2,\; w_z = (dx/dz)^2,\; \mathrm{diag} = 2(1+w_y+w_z), $$

built once in the driver constructor from the root-grid spacings and applied by
`ApplyGravityLaplacian` (`src/gravity/mg_gravity.hpp`), which keeps a branch for the isotropic
case $w_y=w_z=1$. Anisotropic cells are supported, curvilinear and GR metrics are not.

**Smoother.** Red-black Gauss-Seidel with over-relaxation,
`u -= (L u - src*dx^2) * omega/diag` (`src/gravity/mg_gravity.cpp`, via `Multigrid::Smooth` for
the boundary-cell variant). The colour offset flips every V-cycle
(`src/multigrid/multigrid_driver.cpp`), and each block carries a parity from its logical
location so the checkerboard is globally consistent (`src/multigrid/multigrid.cpp`). On
multi-rank runs with `same_exchange_stride > 1` the pack smoother switches to a
communication-avoiding Jacobi or two-stage Chebyshev variant that writes through the `def_`
scratch array.

**Restriction and prolongation.** Restriction is the plain 8-cell average (factor 0.125).
Prolongation is either trilinear or a factorised 27-point tricubic with per-axis weights
(5, 30, -3)/32 and normalisation 1/32^3 (`src/multigrid/multigrid.cpp`; the same weights
appear for octet and boundary use in `src/gravity/mg_gravity.cpp`), selected by
`gravity/mg_prolongation`.

**Cycle structure.** FAS is used throughout: `StoreOldData` and `CalculateFASRHSPack` on the
down-leg, `ComputeCorrection` (u minus uold) and prolongation of the correction on the up-leg
(`src/multigrid/multigrid_tasks.cpp`). A V-cycle is `OneStepToCoarser` x N,
`SolveCoarsestGrid`, `OneStepToFiner` x N (`src/multigrid/multigrid_driver.cpp`). FMG restricts
the source to the coarsest root level, solves there, then walks up doing `fmg_ncycle` V-cycles
per level before handing over to the iterative driver. The coarsest root grid is relaxed
max(coarsest cells, `coarsest_min_sweeps`) times.

**Convergence control** (`src/multigrid/multigrid_driver.cpp`):

- `threshold < 0`: `SolveIterativeFixedTimes`, exactly `niteration` V-cycles.
- `threshold > 0`: `SolveIterative`, V-cycles in bursts of `defect_check_interval` until the L2
  defect drops below the threshold, with stagnation (ratio above 0.995) and divergence (ratio
  above 1.02) monitors and a cap of `niteration` checks, else 160.
- `threshold == 0`: the auto controller. A preset burst, then extra cycles until the defect
  drops below `auto_target_rtol_ * source_norm_` (floored at 1e-12) or `auto_max_extra_cycles`
  is reached (`MultigridDriver::AutoTargetDefect`, `src/multigrid/multigrid.hpp`).
  `auto_target_rtol_` is a hardcoded 1e-6, not an input key. Only this branch reads
  `auto_max_extra_cycles` and `warm_final_niter`. `auto_target_defect` is not a key, and
  `RetireDeadParameter` makes setting it fatal.

**Potential boundary conditions.** Mesh BCs are mapped once: `periodic` to periodic, `outflow`
to `mg_zerograd`, everything else to `mg_zerograd`. `gravity/mg_bc` then overrides every
non-periodic face, and `ix1_bc` to `ox3_bc` override individual faces. The three rules are
applied in `ApplyPhysicalBoundariesBlocks`:

- `zerograd` (Neumann): ghost = interior.
- `zerofixed` (Dirichlet, phi = 0 on the face): ghost = -interior.
- `multipole` (isolated): ghost = 2 phi_mp - interior, with phi_mp from a real-spherical-harmonic
  expansion of the source to quadrupole (`mporder = 2`, 9 coefficients) or hexadecapole
  (`mporder = 4`, 25) about the centre of mass.

A degenerate 1-cell coarsest grid is solved directly (effective-diagonal sweep), so
`mg_bc = multipole` does not need `coarsest_min_sweeps = 1`. The key is inert on that grid and
matters only when the coarsest grid keeps more than one cell.

## Coupling into the fluid

`SourceTerms::Gravity` (`src/srcterms/srcterms.cpp`) works in source form, not conservative
form. Per direction it adds

$$ \Delta(\rho v_x) = \tfrac12\,\frac{\Delta t}{dx}\,\rho\,(\delta\phi_L + \delta\phi_R),
\qquad
\Delta E = \tfrac12\,\frac{\Delta t}{dx}\,(F^{\rho}_{i}\,\delta\phi_L + F^{\rho}_{i+1}\,\delta\phi_R), $$

with $\delta\phi$ the centred differences of `pgrav->phi` and $F^\rho$ the Godunov mass fluxes.
This is the source-term form of Mullen, Hanawa and Gammie (2020). The kernel's own doc
comment states that exact conservation is limited by the multigrid residual and by any
time-lagged potential. The dual-energy auxiliary field is deliberately not updated here. The
term is invoked per RK stage from the `stagen` task list (`src/hydro/hydro_tasks.cpp`,
`src/mhd/mhd_tasks.cpp`) via `SourceTerms::ApplySrcTerms`.

## How a solve runs

1. `MeshBlockPack::AddPhysics` builds `gravity::Gravity` when `<gravity>` exists and
   `self_gravity != false`. It constructs `MGGravityDriver` (`src/gravity/gravity.cpp`),
   allocating a per-block root-level and a per-block MeshBlock-level `Multigrid`, plus the
   multigrid boundary buffers. Buffers are sized with zero flux components on purpose,
   because multigrid never does flux correction and the base class would otherwise allocate
   about 111 MiB per rank that is never touched.
2. Level counts: the root grid is coarsened while all three root block counts stay even. The
   MeshBlock levels require `nx1` a power of two and `nx1 == nx2 == nx3`.
3. Per RK stage, `Driver` calls `pmgd->Solve(this, stage, dt)`. `Solve` runs `PrepareForAMR`
   (once per cycle, reused across stages), the cadence decision, a resize and validity check,
   the source load from `phydro->u0` or `pmhd->u0` slot `IDN` scaled by `-four_pi_G` (with the
   optional radius and BH-sink masks), optionally loads `phi` as the initial guess, then
   `SetupMultigrid`, the multipole coefficients, `SolveFMG` or `SolveMG`, `RetrieveResult`
   into `pgrav->phi`, `RefreshPhiGhosts`, and finally sets `phi_valid`, calls
   `Gravity::MarkSelfPhiValid()` and advances `next_solve_time_`.
4. Stage policy inside one cycle: a cold solve takes the FMG ramp when `full_multigrid` is true
   and converges on every stage. A warm potential never takes the ramp, and on an unchanged
   mesh with a cadence configured it skips intermediate stages entirely
   (`skip_idle_intermediate_solve`). A warm last stage under `threshold == 0` runs
   `warm_final_niter` cycles, trimmed to 1 or 3 when the previous defect was already small.
   With reduced same-level exchange the last stage adds a post-smooth sweep. The tail of every
   V-cycle is one finest-level sweep against exact halos (`finest_exact_polish_passes_`).
5. **MPI.** The finest-to-root transfer is one `MPI_Allgatherv` of one cell per block (an
   owner-only `Gatherv` when `distributed_coarse_solve`). Defect and array norms and the source
   average are `MPI_Allreduce`, and so are the multipole coefficients and the centre of mass.
   With `distributed_coarse_solve` the root grid and every octet are broadcast from the owner
   after the bottom solve. Level halos go through the ordinary `MultigridBoundaryValues`
   pack, send and receive tasks.
6. **GPU.** Everything on the MeshBlock levels is a `par_for` on `DevExeSpace`: source load,
   restriction, prolongation, correction, smoother, defect, physical boundaries and multipole
   partials. The exceptions are the octets, which are host `std::vector` loops, and the root
   grid when `root_on_host = true`. Its default depends only on the root-grid block count, not
   on the build type.
7. **Restart.** `phi` is not part of the restart file, so `phi_valid` starts `false` and the
   first `Solve` takes the cold-start branch: FMG from a zeroed hierarchy (the comment in
   `src/gravity/mg_gravity.cpp` names "fresh starts and restart-remap conversions"). 
   `SourceTerms::Gravity` aborts rather than use a stale potential if a solve has not run.

**Gravity cadence.** `solve_every N` solves on cycles where `ncycle % N == 0`. `solve_dt T`
solves when `time >= next_solve_time_`. After a solve at time t the next due point is the
first grid point (k+1)*T beyond t, so the schedule is a function of time alone
(`MGGravityDriver::Solve`, the `use_solve_dt_ && is_last_stage` block, and `SolveDueAtTime`).
The per-cycle answer is cached in `cadence_due_this_cycle_` so both RK stages agree, and
`ResetCadenceCache()` and `MarkSolveDueAt()` let the driver force an early solve. Between due
points a valid potential is reused (`skip_cadence_cycle`).

## LAT scheduling and refusals

**Where a solve can happen under LAT.** There are three places, all synchronized.

- **Before a window** (`src/driver/driver.cpp`). The driver first shrinks `lat_sync_factor`
  until `lat_fine_dt * factor <= solve_dt`, because the potential is frozen inside a window.
  Then, if the potential is invalid, or a solve is due now, or the full window would step past
  the next due time, it calls `ClearHydroLAT` and solves. `WindowCrossesSolveTime` makes it
  refresh early rather than truncate the window, since truncating would produce degenerate
  128/16/4/1/1-tick descents.
- **Per stage, only outside a LAT bin.** The call is guarded by `if (!lat_bin && ...)` with
  `pgrav != nullptr`. Every invocation inside a window passes `lat_bin = true`, so no Poisson
  solve ever runs inside a LAT window, not per tick and not per bin. Only the non-window path
  (`lat_sync_factor == 1`) passes `lat_bin = false`.
- **At the window end.** A terminal solve when the run stops after this substep and the
  potential is invalid or now due.

**Is the source density the mixed-time density?** No. Because the solve is refused inside bins,
`LoadSource` never sees blocks at different local times. `docs/lat_implementation_note.tex`
gives this as the reason: "A LAT window reuses one synchronized potential; it does not run
Poisson solves inside local bins, because doing so would load density from blocks at mixed
local times." The same file's constraint list repeats that AMR, output and self-gravity source
loading happen only at synchronized states.

**Refusals and requirements.**

- LAT with self-gravity requires `gravity/solve_dt > 0`, checked twice with the same message:
  at input parse (`src/main.cpp`, which keys on the `<gravity>` block) and in the driver
  (`src/driver/driver.cpp`, through `ActiveFluidSourceTerms`).
- `time/lat_same_level = true` plus a source-coupled fluid (self-gravity or analytic BH)
  requires `lat_neighbor_limiter` to be `all` or `hybrid` (`ActiveFluidSourceTerms`).
- Two fluids are refused before any of this. An ion-neutral pack has no single gravity source
  density, because the Poisson source would need a contribution from each fluid and each would
  need its own window bookkeeping. `has_hydro == has_mhd` together with `pionn != nullptr` is
  already in the LAT `invalid` expression, so `ActiveFluidSourceTerms` may state "hydro first"
  as a total rule.
- A pure LAT GID rebalance carries `phi` with the blocks, ghosts included, like every other
  evolved array, and re-stamps a potential that was current on the old topology
  (`MeshRefinement::RedistAndRefineMeshBlocks`). The next solve is the scheduled warm one.
  Invalidating the potential instead would cost an off-cadence cold FMG solve per rebalance and
  make the potential schedule depend on the rank count.

**What the frozen potential costs, and what is corrected.** Inside a window the gravity source
term still runs per stage per bin, with the per-block time step
`block_bdt = (bdt/mesh_dt) * lat_step_dt(m)` and the active-index remap over
`lat_active_indices` (`src/srcterms/srcterms.cpp`). The gravitational work term is tied to the
Godunov mass flux, so it needs a matching delayed correction when a coarse/fine flux mismatch
is refluxed at the window end. That is `lat_apply_selfgrav_reflux_x{1,2,3}`
(`src/hydro/hydro_update.cpp`), which re-applies $-\tfrac12 F^{\rho}_{\rm acc}\,\delta\phi/dx$
with the current (frozen) potential. It is gated on `phi_valid && self_phi_time_valid`. The
setter is the argument-less `Gravity::MarkSelfPhiValid()`, and only the `self_phi_time_valid`
flag it sets is consumed anywhere.

**Restart cadence.** The checkpoint does not carry the potential, so the restart cycle solves
cold on its first RK stage where the uninterrupted run used the frozen potential of the previous
due cycle. Under LAT the window-start solve makes the two runs agree to the solver tolerance.
Without LAT the first stage after a restart differs by the frozen-potential error of one
cadence interval (a few 1e-3 in the kinetic energy of a Jeans test), and the runs stay that far
apart.

## AMR and SMR

Refinement levels between the root grid and the MeshBlock levels are represented by host-side
octets: one 2x2x2 parent cell per refined region per level, keyed by `LogicalLocation`
(`src/multigrid/multigrid_driver.cpp`, `src/multigrid/multigrid.hpp`). They take part fully in
the V-cycle (smooth, defect, FAS RHS, prolongation) but run on the host.

`PrepareForAMR` detects topology changes with an FNV-style hash over every block's logical
location, level and owning rank, rebuilds the octet map and rank lists, and sets a sticky
`amr_mesh_changed_` flag. The flag is cleared only by `ConsumeAMRMeshChanged()` at the end of a
last-stage solve. After a regrid the potential is invalidated unless `reuse_phi_after_amr` is
set (`src/mesh/mesh_refinement.cpp`), and either way the ghosts are refreshed with a fallback to
invalidation. Under AMR the defaults shift to `npostsmooth = 2` and `mg_prolongation = tricubic`,
and `distributed_coarse_solve` stays `false`.

## Sinks

Sink particles are deliberately absent from the Poisson right-hand side. Their potential is
added by a separate source term (`SourceTerms::SinkGravity`) that differences the same softened
potential the sink-to-gas reaction uses, so the pair force is equal and opposite cell by cell.
`<sink_particles>/create = true` requires a `<gravity>` block (`src/sink_particles/sink_particles.cpp`),
mirroring ORION2's gating, and G defaults to `four_pi_G/(4 pi)` when `<units>` is absent. Under
LAT the window-end reflux also refunds the sink potential's work (see the
[LAT notes](Local-Adaptive-Time-Stepping-Implementation-Notes)).

## Energy ledger and coarse/fine symmetry

`gravity/lat_time_centered_work = true` (LAT only, needs `rho_grav_min <= hydro/dfloor`, no
`mask_radius` and no sink) freezes phi over each LAT window, solves again at the window end and
adds -1/2 sum(drho dphi dV) to the gas energy. For a symmetric Poisson operator,
`E_gas + 1/2 sum(rho phi dV)` is then exact to round-off apart from floors, AMR remaps and
boundary transport. Those are tallied in double precision and exposed as user history columns
(`W_cent`, `E_remap`, `E_floor`, `E_asym`, `B_mass`, `B_E`, `B_Wself`, `B_Wbh`; the last four need
`lat_boundary_flux_diagnostics = true`) and carried through restarts as `<gravity>` metadata.
`lat_ledger_debug = true` prints the per-window identity terms.

The composite coarse/fine operator on the MeshBlock levels is not symmetric. The fine-side
ghost fill `(2(c +- gy +- gz) + f)/3` couples a fine cell to the coarse cell's tangential
neighbours with weight -+dx_c/24 without a reciprocal entry. The measured reciprocity defect
`|sum(rhoA phiB) - sum(rhoB phiA)|/|sum|` is 6e-12 uniform, 2.5e-5 on a 2-level mesh and 6e-5 on
3 levels. `mg_fc_symmetric = true` drops the tangential terms (piecewise-constant tangential
interpolation at coarse/fine faces). The operator is then symmetric to round-off, with the same
V-cycle count and a 0.14% change of the test star's self-energy. The default is `false`.
`reciprocity_test = true` runs the double-precision reciprocity diagnostic at startup.

## Known issues

- Conservation is only as good as the multigrid residual and the potential's age.
- The potential is not written to restart files, so a restart is not bitwise. It agrees with
  the uninterrupted run only to the multigrid tolerance.
- With all-Neumann boundaries the solved equation is the Jeans swindle. The driver only warns.
- `mg_verbose` gates exactly one message. It is not a general verbosity control.
- The `tde_external` user source terms are not LAT-safe when the translating frame is on.

## Code map

There is no `<multigrid>` input block (`multigrid` is not in the valid-block list in
`src/parameter_input.cpp`, so it aborts the run), and no gravity task list
(`pgrav->AssembleTasks` is commented out in `src/mesh/meshblock_pack.cpp`). The driver calls
`Solve()` directly. `src/multigrid` reads no keys.

| file | role |
| --- | --- |
| `src/gravity/gravity.hpp`, `gravity.cpp` | `Gravity`: owns `phi` and `coarse_phi`, `four_pi_G`, the `phi_valid` and `self_phi_time_valid` flags, storage resize, ghost refresh. Constructs the driver. |
| `src/gravity/mg_gravity.hpp`, `mg_gravity.cpp` | `MGGravity` (per-level physics: smoother, defect, FAS RHS) and `MGGravityDriver` (all `<gravity>` keys, `Solve`, cadence, source masking, octet physics, distributed coarse-solve broadcast) |
| `src/multigrid/multigrid.hpp` | `Multigrid`, `MultigridDriver`, `MGOctet` (2x2x2 host-side refinement cell) |
| `src/multigrid/multigrid.cpp` | Per-block level arrays, level construction, `LoadSource` and `RetrieveResult`, restriction, prolongation, correction, norms, averages |
| `src/multigrid/multigrid_driver.cpp` | Hierarchy setup, AMR detection (`PrepareForAMR`), octet build, smooth and transfer, root and blocks MPI transfer, boundary application, multipole coefficients, V-cycle, FMG and iterative drivers |
| `src/multigrid/multigrid_tasks.cpp` | The `mg_to_coarser`, `mg_to_finer` and `mg_fmg_prolongate` task lists and the boundary, smooth, restrict and prolongate tasks |
| `src/srcterms/srcterms.cpp` | `SourceTerms::Gravity` (self and external BH) and `SourceTerms::SinkGravity` |
| `src/hydro/hydro_update.cpp` | LAT delayed-reflux gravitational-work correction (`lat_apply_selfgrav_reflux_x{1,2,3}`) |
| `src/driver/driver.cpp` | Where `Solve()` is called, and every LAT and gravity scheduling guard |
| `src/mesh/mesh_refinement.cpp` | Post-AMR potential invalidation or reuse, and ghost refresh |
| `src/outputs/basetype_output.cpp` | `grav_phi` output variables read `pgrav->phi` and add the analytic external-BH potential, guarded on `phi_valid` |

The key list was found with a grep for the parameter getters over `src/gravity` and
`src/multigrid`, plus `grep -rn '"gravity"' src/`, because `gravity/self_gravity` is read
outside `src/gravity`.

## Tests

`tst/test_suite/multigrid/`, run from a build directory via
`python ../tst/run_test_suite.py --cpu`, `--mpicpu` or `--gpu` (the suffix in each filename
selects the platform). Shared helpers (stdout capture, defect parsing, convergence and
consistency assertions) are in `mg_utils.py`.

| file | deck | what it asserts |
| --- | --- | --- |
| `test_mg_binary_gravity_cpu.py` | `tst/inputs/binary_gravity.athinput` | Two compact spheres, uniform (`res` 32/64 x `rg` 4/8) and SMR: defect reaches 1e-9 within 10 V-cycles with average reduction ratio at most 0.0625. Final defect consistent between uniform and SMR. |
| `test_mg_binary_gravity_gpu.py` | same | GPU counterpart |
| `test_mg_binary_gravity_mpicpu.py` | same | Adds `test_binary_gravity_rank_consistency_mpicpu`: final defect must match across rank counts. |
| `test_mg_jeans3d_cpu.py` | `tst/inputs/jeans_wave.athinput` | Stable ($n_J=0.5$) and unstable ($n_J=2$) Jeans waves for both `fmg` and `mgi`: measured $\omega$ within 1% and 3% of $\omega^2=k^2c_s^2(1-n_J^2)$ and convergent with resolution. `test_jeans_solve_iterative_cpu` checks `SolveIterative` reaches 1e-8 in at most 4 checks. |
| `test_mg_jeans3d_gpu.py` | same | GPU counterpart |
| `test_mg_jeans3d_mpicpu.py` | same | Adds `test_jeans_amr_mpicpu`: 4 ranks, adaptive 2-level, asserts blocks are both created and destroyed and the growth rate still converges. |
| `test_mg_poisson3d_cpu.py` | `tst/inputs/selfgravity.athinput`, `tst/inputs/selfgravity_mhd.athinput` | 64^3 hydro and MHD self-gravity: defect to 1e-8 in at most 10 cycles, ratio at most 0.07. `test_selfgravity_decomposition_consistency_cpu` requires the final defect to be independent of MeshBlock size (spread below 1e-4). |
| `test_mg_poisson3d_gpu.py` | same | GPU counterpart |
| `test_mg_poisson3d_mpicpu.py` | `tst/inputs/selfgravity.athinput` | Decomposition and 1-vs-8-rank consistency (spread below 0.03). |

## References

- Mullen, Hanawa and Gammie (2020): the flux-consistent gravitational-work source form used in
  `SourceTerms::Gravity`.
- `docs/lat_implementation_note.tex`, section "Self-Gravity" and the constraint list.
