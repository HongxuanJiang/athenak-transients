# Multigrid Self-Gravity

## Summary

`<gravity>` turns on a Newtonian self-gravity solver: a cell-centred geometric multigrid
(FAS, red-black Gauss-Seidel, V-cycles or FMG) for $\nabla^2\phi = 4\pi G\rho$ on the whole
MeshBlock hierarchy, plus a source-term coupling of $-\rho\nabla\phi$ and the gravitational
work into the hydro/MHD conserved variables. It is used by TDE and sink-particle problems; it is *not* a GR module and has no relativistic
version. Cost is dominated by the solve, not the coupling. Under LAT the potential is frozen for a whole window and the
solve runs at most once per window, at a synchronized state.

## Physics and algorithm

**Equation and discretisation.** The finest multigrid source is loaded straight from the
conserved density of whichever fluid is active, scaled by $-4\pi G$
(`src/gravity/mg_gravity.cpp`; `src/multigrid/multigrid.cpp`). The discrete
operator is the dx²-scaled 7-point Cartesian Laplacian with per-axis anisotropy weights

$$ L u = \mathrm{diag}\,u_{ijk} - (u_{i\pm1}) - w_y (u_{j\pm1}) - w_z (u_{k\pm1}), \qquad
w_y = (dx/dy)^2,\; w_z = (dx/dz)^2,\; \mathrm{diag} = 2(1+w_y+w_z), $$

built once in the driver constructor from the root-grid spacings
(`src/gravity/mg_gravity.cpp`) and applied by `ApplyGravityLaplacian`
(`src/gravity/mg_gravity.hpp`), which keeps a branch for the isotropic
$w_y=w_z=1$ case. Anisotropic cells are therefore supported; curvilinear and GR metrics
are not.

**Smoother.** Red-black Gauss-Seidel with over-relaxation,
`u -= (L u - src*dx^2) * omega/diag` (`src/gravity/mg_gravity.cpp` via
`Multigrid::Smooth` for the boundary-cell variant). The colour offset flips
every V-cycle (`src/multigrid/multigrid_driver.cpp`) and each block carries a parity
from its logical location so the checkerboard is globally consistent
(`src/multigrid/multigrid.cpp`, used at `src/gravity/mg_gravity.cpp`).
On multi-rank runs with `same_exchange_stride > 1` the pack smoother switches to a
communication-avoiding Jacobi or two-stage Chebyshev variant that writes through the
`def_` scratch array (`src/gravity/mg_gravity.cpp`).

**Restriction / prolongation.** Restriction is the plain 8-cell average, factor `0.125`
(`src/multigrid/multigrid.cpp`). Prolongation is either trilinear or a factorised
27-point tricubic with per-axis weights $(5, 30, -3)/32$ and normalisation $1/32^3$
(`src/multigrid/multigrid.cpp`; the same weights appear for octet/boundary use in
`src/gravity/mg_gravity.cpp`), selected by `gravity/mg_prolongation`.

**Cycle structure.** FAS is used throughout: `StoreOldData` + `CalculateFASRHSPack` on the
down-leg (`src/multigrid/multigrid_tasks.cpp`), `ComputeCorrection` (u − uold) and
prolongation of the correction on the up-leg
(`src/multigrid/multigrid_tasks.cpp`). A V-cycle is
`OneStepToCoarser` × N → `SolveCoarsestGrid` → `OneStepToFiner` × N
(`src/multigrid/multigrid_driver.cpp`). FMG restricts the source to the coarsest
root level, solves there, then walks up doing `fmg_ncycle` V-cycles per level before handing
over to the iterative driver (`src/multigrid/multigrid_driver.cpp`). The coarsest
root grid is relaxed `max(coarsest cells, coarsest_min_sweeps)` times
(`src/multigrid/multigrid_driver.cpp`).

**Convergence control** (`src/multigrid/multigrid_driver.cpp`):
* `threshold < 0` → `SolveIterativeFixedTimes`: exactly `niteration` V-cycles.
* `threshold > 0` → `SolveIterative`: V-cycles in bursts of `defect_check_interval` until the
  L2 defect drops below the threshold, with stagnation (ratio > 0.995) and divergence
  (ratio > 1.02) monitors and a cap of `niteration` checks, else 160.
* `threshold == 0` → the "auto" controller: a preset burst, then extra cycles until the
  defect drops below `auto_target_rtol_ * source_norm_` (floored at `1.0e-12`) or
  `auto_max_extra_cycles` is reached (`MultigridDriver::AutoTargetDefect`,
  `src/multigrid/multigrid.hpp`; loop at `src/multigrid/multigrid_driver.cpp`).
  `auto_target_rtol_` is a hardcoded `1.0e-6`, not an athinput key
  (`src/multigrid/multigrid_driver.cpp`). **Only this branch reads `auto_max_extra_cycles`
  and `warm_final_niter`; `auto_target_defect` is not a key — setting it is fatal.**

**Potential boundary conditions.** Mesh BCs are mapped once
(`src/gravity/mg_gravity.cpp`): `periodic`→periodic, `outflow`→`mg_zerograd`,
everything else→`mg_zerograd`. `gravity/mg_bc` then overrides every non-periodic face and
`ix1_bc`…`ox3_bc` override individual faces. The three rules, applied in
`ApplyPhysicalBoundariesBlocks` (`src/multigrid/multigrid_driver.cpp`):
* `zerograd` (Neumann): `ghost = interior`.
* `zerofixed` (Dirichlet $\phi=0$ on the face): `ghost = -interior`.
* `multipole` (isolated): `ghost = 2\phi_{\rm mp} - interior`, with $\phi_{\rm mp}$ from a
  real-spherical-harmonic expansion of the source to quadrupole (`mporder=2`, 9 coefficients)
  or hexadecapole (`mporder=4`, 25) about the centre of mass
  (`src/multigrid/multigrid_driver.cpp`).

**Coupling into the fluid — source form, not conservative form.**
`SourceTerms::Gravity` (`src/srcterms/srcterms.cpp`) adds, per direction,

$$ \Delta(\rho v_x) = \tfrac12\,\frac{\Delta t}{dx}\,\rho\,(\delta\phi_L + \delta\phi_R),
\qquad
\Delta E = \tfrac12\,\frac{\Delta t}{dx}\,(F^{\rho}_{i}\,\delta\phi_L + F^{\rho}_{i+1}\,\delta\phi_R), $$

with $\delta\phi$ the centred differences of `pgrav->phi` and $F^\rho$ the Godunov mass
fluxes, i.e. the source-term form of Mullen, Hanawa & Gammie (2020)
(`src/srcterms/srcterms.cpp`, whose doc comment
states plainly that exact conservation is limited by the multigrid residual and by any
time-lagged potential). The dual-energy auxiliary field is deliberately *not* updated here.
The term is invoked per RK stage from the `stagen` task list
(`src/hydro/hydro_tasks.cpp`; `src/mhd/mhd_tasks.cpp`) via
`SourceTerms::ApplySrcTerms` (`src/srcterms/srcterms.cpp`).

## Code map

| File | Role |
| --- | --- |
| `src/gravity/gravity.hpp`, `gravity.cpp` | `Gravity`: owns `phi`/`coarse_phi`, `four_pi_G`, the `phi_valid` / `self_phi_time_valid` flags, storage resize, ghost refresh; constructs the driver |
| `src/gravity/mg_gravity.hpp`, `mg_gravity.cpp` | `MGGravity` (per-level physics: smoother, defect, FAS RHS) and `MGGravityDriver` (all `<gravity>` keys, `Solve`, cadence, source masking, octet physics, distributed coarse-solve broadcast) |
| `src/multigrid/multigrid.hpp` | `Multigrid`, `MultigridDriver`, `MGOctet` (2×2×2 host-side refinement cell) declarations |
| `src/multigrid/multigrid.cpp` | Per-block level arrays, level construction, `LoadSource`/`RetrieveResult`, restriction, prolongation, correction, norms, averages |
| `src/multigrid/multigrid_driver.cpp` | Hierarchy setup, AMR detection (`PrepareForAMR`), octet build/smooth/transfer, root↔blocks MPI transfer, boundary application, multipole coefficients, V-cycle / FMG / iterative drivers |
| `src/multigrid/multigrid_tasks.cpp` | The `mg_to_coarser` / `mg_to_finer` / `mg_fmg_prolongate` task lists and the boundary/smooth/restrict/prolongate tasks |
| `src/srcterms/srcterms.cpp` | `SourceTerms::Gravity` (self + external-BH) and `SourceTerms::SinkGravity` |
| `src/hydro/hydro_update.cpp` | LAT delayed-reflux gravitational-work correction (`lat_apply_selfgrav_reflux_x{1,2,3}`) |
| `src/driver/driver.cpp` | Where `Solve()` is actually called, and every LAT/gravity scheduling guard |
| `src/mesh/mesh_refinement.cpp` | Post-AMR potential invalidation / reuse and ghost refresh |

There is **no** `<multigrid>` input block — `multigrid` is not in the valid-block list
(`src/parameter_input.cpp`) and would abort the run. Every key lives in `<gravity>`.
There is also no gravity task list: `pgrav->AssembleTasks` is commented out
(`src/mesh/meshblock_pack.cpp`) and the driver calls `Solve()` directly.

## Configuration

Keys found with:

```
grep -rnE 'GetOrAddReal|GetOrAddInteger|GetOrAddBoolean|GetOrAddString|GetReal|GetInteger|GetBoolean|GetString|DoesParameterExist' src/gravity src/multigrid
grep -rn '"gravity"' src/ --include=*.cpp --include=*.hpp
```

(the second is needed because `gravity/self_gravity` is read outside `src/gravity`; `src/multigrid` reads no keys at all).

| Key | Type | Default | Meaning | Read at |
| --- | --- | --- | --- | --- |
| `self_gravity` | bool | `true` when `<gravity>` exists | Master switch: builds `Gravity` and the hydro/MHD source term | `src/mesh/meshblock_pack.cpp`, `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`, `src/srcterms/srcterms.cpp`, `src/main.cpp` |
| `four_pi_G` | Real | `-1.0` | $4\pi G$ in code units. `0.0` is fatal; `< 0` at driver construction is fatal unless a pgen calls `SetFourPiG` | `src/gravity/gravity.cpp`, `src/gravity/mg_gravity.cpp` |
| `reuse_phi_after_amr` | bool | `false` | Keep the AMR-transferred potential as the next initial guess instead of forcing a cold FMG restart | `src/gravity/gravity.cpp` |
| `omega` | Real | `1.15` | SOR relaxation weight (used as `omega/diag`) | `src/gravity/mg_gravity.cpp` |
| `rho_grav_min` | Real | `0.0` (clamped `>= 0`) | Density below which a cell contributes no Poisson source and receives no gravity source term | `src/gravity/mg_gravity.cpp`, `src/srcterms/srcterms.cpp` |
| `threshold` | Real | `-1.0` | Target L2 defect. `<0` = fixed `niteration` cycles; `>0` = iterate to it; `==0` = auto controller | `src/gravity/mg_gravity.cpp` |
| `niteration` | int | `-1` | V-cycle count (`threshold<0`) or check cap (`threshold>0`). One of `threshold`/`niteration` must be set | `src/gravity/mg_gravity.cpp` |
| `npresmooth` | int | `1` (`max(1,·)`) | Pre-smoothing sweeps per level | `src/gravity/mg_gravity.cpp`; base default `src/multigrid/multigrid_driver.cpp` |
| `npostsmooth` | int | `2` under AMR, else `1` | Post-smoothing sweeps per level | `src/gravity/mg_gravity.cpp` |
| `same_exchange_stride` | int | `2` if `nranks>1`, else `1` | Do a same-level halo exchange only every N smoother steps | `src/gravity/mg_gravity.cpp` |
| `local_sweeps_per_exchange` | int | `3` if stride>1 else `1`; further raised to `>=3` when `nranks>1 && stride>1` | Local red+black sweeps between exchanges | `src/gravity/mg_gravity.cpp` |
| `distributed_coarse_solve` | bool | `false` | Only the owner rank runs the root/octet bottom solve; result is broadcast. Even when set, effective only if `SupportsDistributedCoarseSync() && nranks_>1 && nreflevel_>0` at runtime | `src/gravity/mg_gravity.cpp` |
| `ca_pack_smoother` | string | `chebyshev` if `nranks>1`, else `rbgs` | Communication-avoiding smoother: `rbgs`, `jacobi`, `chebyshev`/`chebyshev2` | `src/gravity/mg_gravity.cpp` |
| `ca_pack_cheb_lambda_min` | Real | `0.7` | Chebyshev lower eigenvalue bound | `src/gravity/mg_gravity.cpp`; default `src/gravity/mg_gravity.hpp` |
| `ca_pack_cheb_lambda_max` | Real | `1.95`, forced `> min` | Chebyshev upper eigenvalue bound | `src/gravity/mg_gravity.cpp`; default `mg_gravity.hpp` |
| `full_multigrid` | bool | `true` | Use FMG rather than plain V-cycles. Honoured only for a cold solve (no warm potential: a fresh start or a restart-remap conversion); a warm solve (`pgrav->phi_valid`) forces plain V-cycles regardless of the key | `src/gravity/mg_gravity.cpp` |
| `fmg_ncycle` | int | `1` | V-cycles per FMG level | `src/gravity/mg_gravity.cpp` |
| `show_defect` | string/int | `"0"` | `0`/`false` off, `1`/`true` final defect only, `>=2` per-iteration defect | `src/gravity/mg_gravity.cpp` |
| `mg_verbose` | int | `0` (`max(0,·)`) | Only gates one warning: multipole source with cancelling total mass | `src/gravity/mg_gravity.cpp`; used at `src/multigrid/multigrid_driver.cpp` |
| `subtract_average` | bool | `mesh->strictly_periodic` | Remove the mean of source and solution. **Forced `true` (with a warning) if no face is `zerofixed`/`multipole`; forced `false` if multipole is on** | `src/gravity/mg_gravity.cpp` |
| `mg_prolongation` | string | `tricubic` under AMR, else `trilinear` | Prolongation stencil; any value other than `tricubic` means trilinear | `src/gravity/mg_gravity.cpp` |
| `defect_check_interval` | int | `2` (`max(1,·)`) | V-cycles between defect-norm evaluations (forced to 1 when `show_defect>=2`) | `src/gravity/mg_gravity.cpp`; default `multigrid_driver.cpp` |
| `auto_max_extra_cycles` | int | `-1` (= computed: `max(6, 3*fmg_ncycle)`, raised for AMR/regrid) | Auto-controller cycle cap. **Read only when `threshold == 0`** | `src/gravity/mg_gravity.cpp`; used at `multigrid_driver.cpp` |
| `coarsest_min_sweeps` | int | `64` (`max(1,·)`) | Minimum relaxations on the coarsest root grid | `src/gravity/mg_gravity.cpp`; default `multigrid_driver.cpp` |
| `show_timing` | bool | `false` | Print a per-solve phase timing line | `src/gravity/mg_gravity.cpp` |
| `solve_every` | int | `1` (`max(1,·)`) | Solve on cycles with `ncycle % N == 0`; ignored if `solve_dt > 0` | `src/gravity/mg_gravity.cpp`; default `mg_gravity.hpp` |
| `solve_dt` | Real | absent | Solve interval in code time; must be `>= 0`; `> 0` enables time-based cadence and overrides `solve_every` (message printed) | `src/gravity/mg_gravity.cpp`; also probed at `src/main.cpp`, `src/driver/driver.cpp` |
| `warm_final_niter` | int | `4` (`max(1,·)`) | V-cycles on the last stage of a warm, unchanged-mesh cycle. **Only when `threshold == 0` and `niteration <= 0`** | `src/gravity/mg_gravity.cpp`; default `mg_gravity.hpp` |
| `mg_bc` | string | `none` | Convenience default for all non-periodic faces: `zerofixed`, `zerograd`/`outflow`, `multipole`, `none`. `periodic` here is fatal | `src/gravity/mg_gravity.cpp` |
| `ix1_bc` | string | (mesh-derived) | Per-face override, same vocabulary as `mg_bc` | `src/gravity/mg_gravity.cpp` |
| `ox1_bc` | string | (mesh-derived) | Per-face override | `src/gravity/mg_gravity.cpp` |
| `ix2_bc` | string | (mesh-derived) | Per-face override | `src/gravity/mg_gravity.cpp` |
| `ox2_bc` | string | (mesh-derived) | Per-face override | `src/gravity/mg_gravity.cpp` |
| `ix3_bc` | string | (mesh-derived) | Per-face override | `src/gravity/mg_gravity.cpp` |
| `ox3_bc` | string | (mesh-derived) | Per-face override | `src/gravity/mg_gravity.cpp` |
| `mporder` | int | `4` | Multipole order, must be `2` or `4`. **Read only when some face is `multipole`** (`mporder_` starts at `-1`, `multigrid.hpp`) | `src/gravity/mg_gravity.cpp` |
| `auto_mporigin` | bool | `true` | Recompute the expansion origin as the source centre of mass each solve | `src/gravity/mg_gravity.cpp` |
| `nodipole` | bool | `false` | Skip the $\ell=1$ terms. Fatal together with `auto_mporigin` | `src/gravity/mg_gravity.cpp` |
| `mporigin_x1` | Real | required if `auto_mporigin=false` | Fixed expansion origin, x | `src/gravity/mg_gravity.cpp` |
| `mporigin_x2` | Real | required if `auto_mporigin=false` | Fixed expansion origin, y | `src/gravity/mg_gravity.cpp` |
| `mporigin_x3` | Real | required if `auto_mporigin=false` | Fixed expansion origin, z | `src/gravity/mg_gravity.cpp` |
| `mask_radius` | Real | `-1.0` (off) | Zero the Poisson source *outside* this radius from `mask_origin` | `src/gravity/mg_gravity.cpp` |
| `mask_origin_x1` | Real | `0.0` | Mask centre, x | `src/gravity/mg_gravity.cpp` |
| `mask_origin_x2` | Real | `0.0` | Mask centre, y | `src/gravity/mg_gravity.cpp` |
| `mask_origin_x3` | Real | `0.0` | Mask centre, z | `src/gravity/mg_gravity.cpp` |
| `mg_nghost` | int | `1` | Ghost cells on the multigrid levels. Any value `> 1` is fatal when the mesh is refined (`pmesh->multilevel`): the flux-conservative coarse-fine prolongation indexes a fixed 3x3x3 coarse buffer and would read/write outside it, so `1` is the only value the octet hierarchy supports | `src/gravity/mg_gravity.cpp` |
| `root_on_host` | bool | `true` iff the root grid has at most 4096 blocks (`root_cells = nmb_rootx1*nmb_rootx2*nmb_rootx3 <= 16*16*16`), regardless of build type | Keep the root-grid multigrid arrays in host memory | `src/gravity/mg_gravity.cpp` |

Related keys read outside `src/gravity` that share the same kernel or gate the module:

| Key | Default | Meaning | Read at |
| --- | --- | --- | --- |
| `hydro_srcterms/self_gravity` | absent | Legacy alias; if `true` without `gravity/self_gravity=true` the run aborts | `src/srcterms/srcterms.cpp` |
| `problem/external_bh_gravity_source` | `true` for `tde_external` | Analytic softened BH potential added in the *same* kernel, separate from `phi` | `src/srcterms/srcterms.cpp`, `src/hydro/hydro.cpp` |
| `problem/bh_grav_rho_min` | `0.0` | Density floor for the external-BH branch | `src/srcterms/srcterms.cpp` |
| `sink_particles/newton_g` | `four_pi_G/(4π)` | G for sink potentials when `<units>` is absent | `src/sink_particles/sink_particles.cpp` |

## How it runs

1. `MeshBlockPack::AddPhysics` builds `gravity::Gravity` when `<gravity>` exists and
   `self_gravity != false` (`src/mesh/meshblock_pack.cpp`), which constructs
   `MGGravityDriver` (`src/gravity/gravity.cpp`), allocating a per-block root-level and a
   per-block MeshBlock-level `Multigrid`, plus the multigrid boundary buffers. Buffers are
   sized with **zero flux components** on purpose — multigrid never does flux correction,
   and the base class would otherwise allocate ~111 MiB/rank that is never touched
   (`src/gravity/mg_gravity.cpp`).
2. Level counts: the root grid is coarsened while all three root block counts stay even
   (`src/multigrid/multigrid.cpp`); the MeshBlock levels require `nx1` a power of two
   and `nx1 == nx2 == nx3`.
3. Per RK stage, `Driver` calls `pmgd->Solve(this, stage, dt)`
   (`src/driver/driver.cpp`). `Solve` (`src/gravity/mg_gravity.cpp`):
   `PrepareForAMR` (once per cycle, reused across stages) → cadence decision → resize/validity
   check → load the source from `phydro->u0` or `pmhd->u0` slot `IDN` scaled by `-four_pi_G`
   with the optional radius and BH-sink masks → optionally load `phi` as the initial guess →
   `SetupMultigrid` → multipole coefficients → `SolveFMG` or `SolveMG` → `RetrieveResult` into
   `pgrav->phi` → `RefreshPhiGhosts` → set `phi_valid`, mark the potential valid
   (`Gravity::MarkSelfPhiValid()`, `mg_gravity.cpp`), advance `next_solve_time_`.
4. Stage policy inside one cycle (`src/gravity/mg_gravity.cpp`, `MGGravityDriver::Solve`): a
   cold solve takes the FMG ramp when `full_multigrid` is true and converges on every
   stage; a warm potential never takes the ramp, and on an unchanged
   mesh with a cadence configured skips intermediate stages entirely
   (`skip_idle_intermediate_solve`); a warm last stage under `threshold==0`
   runs `warm_final_niter` cycles, trimmed to 1 or 3 when the previous defect was already
   small. With reduced same-level exchange the last stage adds a post-smooth sweep. The
   tail of every V-cycle is one finest-level sweep against exact halos
   (`finest_exact_polish_passes_`).
5. **MPI.** Finest→root transfer is one `MPI_Allgatherv` of one cell per block
   (`src/multigrid/multigrid_driver.cpp`; owner-only `Gatherv` when
   `distributed_coarse_solve`). Defect and array norms and the source average are
   `MPI_Allreduce` (`src/multigrid/multigrid.cpp`); multipole
   coefficients and the centre of mass likewise. With
   `distributed_coarse_solve` the root grid and every octet are broadcast from the owner
   after the bottom solve (`src/gravity/mg_gravity.cpp`). Level halos go through the
   ordinary `MultigridBoundaryValues` pack/send/recv tasks.
6. **GPU.** Everything on the MeshBlock levels is a `par_for` on `DevExeSpace`: source load,
   restriction, prolongation, correction, smoother, defect, physical boundaries, multipole
   partials. The exceptions are the octets, which are host `std::vector` loops
   (`src/gravity/mg_gravity.cpp`, `src/multigrid/multigrid.hpp`), and the root
   grid when `root_on_host=true`, whose default depends only on root-grid block count, not
   build type (see the Configuration table).
7. **Restart.** `phi` is not part of the restart file (nothing in `src/outputs/restart.cpp`
   touches it), so `phi_valid` starts `false` (`src/gravity/gravity.cpp`) and the first
   `Solve` after a restart takes the cold-start branch: FMG from a zeroed hierarchy
   (`src/gravity/mg_gravity.cpp`; the comment there names "fresh starts
   and restart-remap conversions" explicitly). `SourceTerms::Gravity` aborts rather than
   using a stale potential if a solve has not run (`src/srcterms/srcterms.cpp`).

## Interactions

### LAT

**Where a solve can happen under LAT — three places, all synchronized:**

* **Before a window** (`src/driver/driver.cpp`). The driver first shrinks
  `lat_sync_factor` until `lat_fine_dt * factor <= solve_dt` — the potential
  is frozen inside a window, so a window can never be longer than the solve interval. Then,
  if the potential is invalid, or a solve is due now, or the full window would step past the
  next due time, it calls `ClearHydroLAT` and solves. `WindowCrossesSolveTime`
  makes it refresh *early* rather than truncating the window; truncating
  the window would produce degenerate 128/16/4/1/1-tick descents.
* **Per stage, only outside a LAT bin** (`src/driver/driver.cpp`). The call is
  guarded by `if (!lat_bin && ...)` (`pgrav != nullptr`).
  Every invocation inside a window passes `lat_bin = true`, so **no Poisson solve ever runs
  inside a LAT window** — not per tick, not per bin. Only the non-window path
  (`lat_sync_factor == 1`) passes `lat_bin = false`.
* **At the window end** (`src/driver/driver.cpp`): a terminal solve when the run
  stops after this substep and the potential is invalid or now due.

**Is the source density the mixed-time density?** No. Because the solve is refused inside
bins, `LoadSource` never sees blocks at different local times. This is stated as the reason
in `docs/lat_implementation_note.tex`: *"A LAT window reuses one synchronized
potential; it does not run Poisson solves inside local bins, because doing so would load
density from blocks at mixed local times."* The same file's constraint list repeats that AMR, output and self-gravity source loading happen only at
synchronized states.

**Refusals and requirements.**
* LAT + self-gravity **requires `gravity/solve_dt > 0`**, checked twice with
  the same message — at input parse (`src/main.cpp`, which keys on the `<gravity>`
  block) and in the driver
  (`src/driver/driver.cpp`, through `ActiveFluidSourceTerms`): *"time/lat with
  self-gravity uses a frozen potential inside each LAT window and requires gravity/solve_dt
  > 0 to bound the window length."*
* `time/lat_same_level=true` plus a source-coupled fluid (self-gravity or analytic BH)
  requires `lat_neighbor_limiter` to be `all` or `hybrid` (`src/driver/driver.cpp`,
  also through `ActiveFluidSourceTerms`).
* **Two fluids are refused before any of this.** An ion-neutral pack has no single gravity
  source density — the Poisson source would need a contribution from each fluid, and each
  would need its own window bookkeeping — and `has_hydro == has_mhd` together with
  `pionn != nullptr` is already in the LAT `invalid` expression
  (`src/driver/driver.cpp`), so
  `ActiveFluidSourceTerms` may state "hydro first" as a total rule.
* A pure LAT GID rebalance carries `phi` with the blocks, ghosts included, like every other
  evolved array, and re-stamps a potential that was current on the old topology
  (`MeshRefinement::RedistAndRefineMeshBlocks`). The next solve is the scheduled warm one;
  invalidating the potential instead would cost an off-cadence cold FMG solve per rebalance and make the
  potential schedule depend on the rank count.

**What the frozen potential costs, and what is corrected.** Inside a window the gravity
source term still runs per stage per bin, with the *per-block* timestep:
`block_bdt = (bdt/mesh_dt) * lat_step_dt(m)` and the active-index remap over
`lat_active_indices` (`src/srcterms/srcterms.cpp`). The gravitational work
term, being tied to the Godunov mass flux, needs a matching delayed correction when a
coarse/fine flux mismatch is refluxed at the window end: that is
`lat_apply_selfgrav_reflux_x{1,2,3}`, which re-applies
$-\tfrac12 F^{\rho}_{\rm acc}\,\delta\phi/dx$ with the *current* (frozen) potential
(`src/hydro/hydro_update.cpp` and
`docs/lat_implementation_note.tex`). It is gated on `phi_valid &&
self_phi_time_valid`. The setter is the argument-less `Gravity::MarkSelfPhiValid()`
(`src/gravity/gravity.cpp`), and only the `self_phi_time_valid` flag it sets is
consumed anywhere.

**What `solve_every` / `solve_dt` ("gravity cadence") do.** `solve_every N` solves on cycles
where `ncycle % N == 0`; `solve_dt T` solves when `time >= next_solve_time_`, and after a
solve at time `t` the next due point is the first grid point `(k+1)*T` beyond `t`, so the
schedule is a function of time alone and a restarted run solves on the same cycles as the
uninterrupted one (`src/gravity/mg_gravity.cpp`, `MGGravityDriver::Solve`, the
`use_solve_dt_ && is_last_stage` block; `SolveDueAtTime`). The checkpoint does not carry
the potential, so the restart cycle solves cold on its first RK stage where the
uninterrupted run used the frozen potential of the previous due cycle; under LAT the
window-start solve makes the two runs agree to the solver tolerance, without LAT the
first stage after a restart differs by the frozen-potential error of one cadence
interval (a few 1e-3 in the kinetic energy of a Jeans test) and the runs stay that far
apart. The per-cycle answer is
cached in `cadence_due_this_cycle_` so both RK stages agree, and `ResetCadenceCache()` /
`MarkSolveDueAt()` let the driver force an early solve. Between due points a valid potential
is simply reused (`skip_cadence_cycle`, `src/gravity/mg_gravity.cpp`).

### AMR / SMR

Refinement levels between the root grid and the MeshBlock levels are represented by host-side
**octets** — one 2×2×2 parent cell per refined region per level, keyed by `LogicalLocation`
(`src/multigrid/multigrid_driver.cpp`, `src/multigrid/multigrid.hpp`). They
participate fully in the V-cycle (smooth, defect, FAS RHS, prolongation) but run on the host.
`PrepareForAMR` (`src/multigrid/multigrid_driver.cpp`) detects topology changes with
an FNV-style hash over every block's logical location, level and owning rank, rebuilds the
octet map and rank lists, and sets a **sticky** `amr_mesh_changed_` flag; the flag is cleared
only by `ConsumeAMRMeshChanged()` at the end of a last-stage solve
(`src/gravity/mg_gravity.cpp`). After a regrid, the potential is invalidated unless
`reuse_phi_after_amr` is set (`src/mesh/mesh_refinement.cpp`), and either way the
ghosts are refreshed with a fallback to invalidation. Under AMR the defaults
shift: `npostsmooth=2`, `mg_prolongation=tricubic`; `distributed_coarse_solve` stays `false`
by default even under AMR (see the Configuration table).

### Sinks and point masses

Sink particles are **deliberately absent from the Poisson RHS** — their potential is added by
a separate source term that differences the same softened potential the sink→gas reaction
uses, so the pair force is equal and opposite cell by cell
(`src/srcterms/srcterms.cpp`). `<sink_particles>/create = true` requires a
`<gravity>` block (`src/sink_particles/sink_particles.cpp`), mirroring ORION2's
gating, and G defaults to `four_pi_G/(4π)` when `<units>` is absent.
`gravity/mask_radius` is the generic (deck-level) counterpart, which zeroes the source
*outside* a radius instead.

### Other

* **FOFC**: no interaction. Multigrid queues no flux-correction tasks and allocates no flux
  buffers (`src/gravity/mg_gravity.cpp`).
* **Units**: `four_pi_G` is a bare code-unit number; the `<units>` module does not feed it
  (only `sink_particles` consults `punit->grav_constant()`).
* **Outputs**: `grav_phi` output variables read `pgrav->phi` and add the analytic external-BH
  potential, guarded on `phi_valid` (`src/outputs/basetype_output.cpp`).
* **Problem generators** may move the BH by sampling $-\nabla\phi$ from the self-gravity field
  only.

## Limitations and known issues

* MeshBlocks must be logically cubic and a power of two in size — both are fatal errors
  (`src/multigrid/multigrid.cpp`).
* A root grid that cannot be coarsened to one cell, or that has > 100 coarsest DOF, prints a
  warning and falls back to an iterative bottom solve (`src/multigrid/multigrid.cpp`).
* Self-gravity requires hydro or MHD state; otherwise `Solve` aborts
  (`src/gravity/mg_gravity.cpp`). With LAT it requires *exactly one* of them: a
  two-fluid (ion-neutral) pack has no single source density and is refused
  (`src/driver/driver.cpp`).
* The potential is not written to restart files, and `phi_valid` starts `false`
  (`src/gravity/gravity.cpp`; no `pgrav` entry in `src/outputs/restart.cpp`). The first
  solve after a restart therefore begins from a different iterate than the warm solve it
  replaces, so a restart is **not** bitwise — it agrees with the uninterrupted run only to
  the multigrid tolerance.
* One of `threshold` / `niteration` must be set (`src/gravity/mg_gravity.cpp`), and
  `four_pi_G` must be positive by driver-construction time.
* `auto_max_extra_cycles` and `warm_final_niter` are **inert unless `threshold == 0`**;
  `auto_target_defect` is not a parameter — `RetireDeadParameter` makes setting
  it fatal (`src/gravity/mg_gravity.cpp`).
* A degenerate 1-cell coarsest grid is solved directly (effective-diagonal sweep,
  `src/gravity/mg_gravity.cpp`), so `mg_bc = multipole` does not need
  `coarsest_min_sweeps = 1`; the key is inert on that grid regardless of its value and only
  matters when the coarsest grid keeps more than one cell.
* Conservation is only as good as the multigrid residual and the potential's age — stated in
  the kernel's own doc comment (`src/srcterms/srcterms.cpp`).
* With all-Neumann boundaries the solved equation is the Jeans swindle, not the isolated
  problem; the driver only warns (`src/gravity/mg_gravity.cpp`).
* `mg_verbose` gates exactly one message; it is not a general verbosity control.

## Tests

`tst/test_suite/multigrid/`, run from a build directory via
`python ../tst/run_test_suite.py --cpu` / `--mpicpu` / `--gpu` (the suffix in each filename
selects the platform; `tst/run_test_suite.py`). Shared helpers — stdout capture, defect
parsing, convergence and consistency assertions — are in `mg_utils.py`.

| File | Deck | What it asserts |
| --- | --- | --- |
| `test_mg_binary_gravity_cpu.py` | `tst/inputs/binary_gravity.athinput` | Two compact spheres, uniform (`res` 32/64 × `rg` 4/8) and SMR: defect reaches 1e-9 within 10 V-cycles with average reduction ratio <= 0.0625; final defect consistent between uniform and SMR |
| `test_mg_binary_gravity_gpu.py` | same | GPU counterpart |
| `test_mg_binary_gravity_mpicpu.py` | same | Adds `test_binary_gravity_rank_consistency_mpicpu`: final defect must match across rank counts |
| `test_mg_jeans3d_cpu.py` | `tst/inputs/jeans_wave.athinput` | Stable ($n_J=0.5$) and unstable ($n_J=2$) Jeans waves for both `fmg` and `mgi`: measured $\omega$ within 1 % / 3 % of $\omega^2=k^2c_s^2(1-n_J^2)$ and convergent with resolution; `test_jeans_solve_iterative_cpu` checks `SolveIterative` reaches 1e-8 in <= 4 checks |
| `test_mg_jeans3d_gpu.py` | same | GPU counterpart |
| `test_mg_jeans3d_mpicpu.py` | same | Adds `test_jeans_amr_mpicpu`: 4 ranks, adaptive 2-level, asserts blocks are both created and destroyed and the growth rate still converges |
| `test_mg_poisson3d_cpu.py` | `tst/inputs/selfgravity.athinput`, `tst/inputs/selfgravity_mhd.athinput` | 64³ hydro and MHD self-gravity: defect to 1e-8 in <= 10 cycles, ratio <= 0.07; `test_selfgravity_decomposition_consistency_cpu` requires the final defect to be independent of MeshBlock size (spread < 1e-4) |
| `test_mg_poisson3d_gpu.py` | same | GPU counterpart |
| `test_mg_poisson3d_mpicpu.py` | `tst/inputs/selfgravity.athinput` | Decomposition and 1-vs-8-rank consistency (spread < 0.03) |

The decks live at `tst/inputs/{selfgravity,selfgravity_mhd}.athinput`. Decks that switch self-gravity on
(`grep -rl self_gravity inputs`): `inputs/TDE_examples/*` (5 decks), `inputs/sink_particles/creation.athinput`.

## Users

* **`src/pgen/tde_external.cpp`** — tidal disruption by an external BH: self-gravity holds the
  star together while the BH acts through a user source term, with
  $G = 4\pi G/(4\pi)$ derived from the same key; its user source terms are
  explicitly *not* LAT-safe when the translating frame is on.

## References

* Mullen, Hanawa & Gammie (2020) — the flux-consistent gravitational-work source form used in
  `SourceTerms::Gravity` (`src/srcterms/srcterms.cpp`).
* `docs/lat_implementation_note.tex` §"Self-Gravity" and the constraint list — the authoritative statement of LAT/gravity synchronization.


## Energy ledger and coarse/fine symmetry

`gravity/lat_time_centered_work = true` (LAT only; needs `rho_grav_min <= hydro/dfloor`,
no `mask_radius`, no sink) freezes phi over each LAT window, solves again at the
window end and adds `-1/2 sum(drho dphi dV)` to the gas energy, so that for a symmetric
Poisson operator `E_gas + 1/2 sum(rho phi dV)` is exact to round-off apart from floors,
AMR remaps and boundary transport.  Those are tallied in double precision and exposed
as user history columns (`W_cent`, `E_remap`, `E_floor`, `E_asym`, `B_mass`, `B_E`,
`B_Wself`, `B_Wbh`; the last four need `lat_boundary_flux_diagnostics = true`) and
carried through restarts as `<gravity>` metadata.  `lat_ledger_debug = true` prints the
per-window identity terms. 

The composite coarse/fine operator on the MeshBlock levels is not symmetric: the
fine-side ghost fill `(2(c +- gy +- gz) + f)/3` couples a fine cell to the coarse cell's
tangential neighbours with weight -+dx_c/24 without a reciprocal entry.  Measured
reciprocity defect `|sum(rhoA phiB) - sum(rhoB phiA)|/|sum|`: 6e-12 uniform, 2.5e-5 on a
2-level mesh, 6e-5 on 3 levels.  `mg_fc_symmetric = true` drops the tangential terms
(piecewise-constant tangential interpolation at coarse/fine faces): symmetric to
round-off, same V-cycle count, 0.14% change of the test star's self-energy.  Default
`false`.  `reciprocity_test = true` runs the double-precision reciprocity diagnostic at
startup.
