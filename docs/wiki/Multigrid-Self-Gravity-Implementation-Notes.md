# Multigrid self-gravity: implementation notes

Back to [Multigrid Self-Gravity](Multigrid-Self-Gravity), which has the parameters, defaults and
usage advice. This page explains how the solver is built, in what order things happen in a time
step, and where its limits are.

## The big picture

```
each Runge-Kutta stage:
  solve due? -- no --> keep the stored potential
     | yes
     v
  load source -> multigrid solve -> potential + ghosts -> hydro/MHD source term
```

## The solver

**The equation.** The finest-level source is the conserved density of the active fluid, times the
coupling weight of `rho_grav_min` and times `-four_pi_G`. The operator $L$ is the 7-point
Cartesian Laplacian scaled by $\Delta x^2$. Its diagonal is $d=2(1+w_y+w_z)$, and the y and z
neighbours carry the weights $w_y=(\Delta x/\Delta y)^2$ and $w_z=(\Delta x/\Delta z)^2$, taken
from the root-grid cell sizes, so anisotropic Cartesian cells work (curvilinear and GR metrics
do not). The solver solves $L\phi = -4\pi G\,\rho\,\Delta x^2$.

**Smoothing and transfers.** One sweep updates one colour of a red-black checkerboard:
$u \leftarrow u - \omega\,(Lu - \Delta x^2\,\mathrm{src})/d$, and the starting colour flips every
V-cycle. Restriction averages the eight child cells. Prolongation is trilinear or tricubic, and
the tricubic stencil is a product of one-dimensional weights (5, 30, -3)/32.

**One V-cycle** (full approximation storage):

1. Going down, smooth `npresmooth` times and restrict the residual to the next coarser level.
2. On the coarsest root grid, relax many times: the larger of the grid's edge length in cells
   and `coarsest_min_sweeps`.
3. Going up, prolong the correction and smooth `npostsmooth` times.

**Full multigrid (FMG).** A cold solve first restricts the source to the coarsest root level and
solves there. It then climbs one level at a time, doing `fmg_ncycle` V-cycles on each, and hands
over to the normal iteration. The levels, from fine to coarse, are the MeshBlock levels (each
block is coarsened to one cell, so the block size must be a power of two), the octet levels (see
[AMR and SMR](#amr-and-smr)) and the root-grid levels (coarsened while all three root block
counts stay even).

**Boundaries.** Periodic mesh faces stay periodic and every other mesh condition becomes
`zerograd`. Then `mg_bc` and the per-face keys override this. In the ghost cells, `zerograd`
copies the interior value next to it, `zerofixed` uses minus that value (so $\phi = 0$ on the
face), and `multipole` uses $2\phi_{mp}$ minus it. Here $\phi_{mp}$ comes from an expansion of
the source about its centre of mass, with 9 coefficients for `mporder = 2` and 25 for 4.

## Coupling into the fluid

The source term is in source form, not conservative form. It runs once per Runge-Kutta stage,
after the flux update, from the stage's primitive variables.

1. The momentum changes by $-\rho\nabla\phi\,\Delta t$, with the gradient taken from the two
   one-sided differences of $\phi$ across the cell.
2. The energy changes by the same two differences, each multiplied by the Godunov mass flux
   through the cell face on its side. This is the form of Mullen, Hanawa and Gammie (2020).
3. Both changes are multiplied by the coupling weight of `rho_grav_min`.

The kernel reads only the solved potential. The analytic black hole is a separate branch of the
same kernel. Exact conservation is limited by the multigrid residual and by any time-lagged
potential. The kernel stops the run if no solve has produced a potential yet.

## How a solve runs

1. **Setup.** If `<gravity>` exists and `self_gravity` is not false, the physics setup creates
   the `Gravity` object, which creates the multigrid driver. Between them they read the
   `<gravity>` keys. There is no gravity task list: the driver calls `Solve` directly at each
   stage, and `Solve` first decides from the cadence whether a solve is due.
2. **Source.** The density comes from the hydro conserved variables, or from the MHD ones if
   there is no hydro. The code applies the coupling weight, the factor `-four_pi_G`, the optional
   `mask_radius`, and the mean subtraction if it is on.
3. **Solve.** A cold solve (no valid potential) takes the FMG ramp when `full_multigrid` is on
   and converges on every stage. A warm solve loads the previous potential and runs plain
   V-cycles. With `threshold = 0`, the last stage of a warm cycle runs `warm_final_niter`
   cycles, cut to 1 or 3 when the previous defect was already close to the target.
4. **Result.** The potential is copied to `phi`, its ghost cells are refreshed, and it is
   marked valid with a time stamp. It is not written to restart files, so after a restart `phi`
   starts invalid and the first solve is cold.

## LAT scheduling

Blocks in a window of [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) (LAT) sit at
different local times, so a solve inside a window would load a density from mixed times. The
potential is frozen over each window and solved only between windows. Per window, the driver does
this.

1. It shortens the window until it is no longer than `solve_dt`.
2. If the potential is invalid or due now, or the window would run past the next due time, it
   solves before the window. Cutting the window to end on the due time instead produced runs of
   ever shorter windows.
3. It runs the window. The stage source term still runs for each active block with that
   block's own time step, but never calls the solver.
4. At the window end it solves if the run stops there and the potential is invalid or due, or
   always with `lat_time_centered_work`.

The gravitational work is tied to the Godunov mass flux. So when the window-end reflux fixes a
coarse/fine flux mismatch, the work of the moved mass must be corrected too. The reflux kernels
in `src/hydro/hydro_update.cpp` re-apply the work with the frozen potential, and the same
correction covers the black-hole and sink potentials.

## AMR and SMR

The levels between the root grid and the finest MeshBlocks are stored as octets: one 2x2x2
group of cells for each refined region on each level. They take part in the V-cycle but run on
the host, as does the root grid when `root_on_host` is on. The MeshBlock levels run as device
kernels. A hash of every block's location, level and rank detects a regrid, which rebuilds the
octets. The potential is then invalidated unless `reuse_phi_after_amr` is set.

## Energy ledger and coarse/fine symmetry

**The ledger (LAT only).** With a frozen potential, the work of the mass fluxes accounts for the
change of mass in the old potential but not for the change of the potential itself.
`lat_time_centered_work = true` supplies the missing half. The code solves again at the window
end and adds $-\tfrac{1}{2}\,\Delta\rho\,\Delta\phi$ to the energy of each cell. The gas energy
plus $\tfrac{1}{2}\sum\rho\phi\,dV$ then changes only through floors, AMR remaps, boundary
transport and the operator asymmetry below. These are tallied in double precision and carried
through restarts as `<gravity>` metadata. The `tde_external` problem generator writes them as
history columns (`W_cent`, `E_remap`, `E_floor`, `E_asym`, and with
`lat_boundary_flux_diagnostics` also `B_mass`, `B_E`, `B_Wself`, `B_Wbh`).

**The coarse/fine operator.** At a coarse/fine interface, the ghost value of a fine cell is built
from the coarse cell and its tangential neighbours, but the coarse cell has no matching term. The
composite operator is therefore slightly non-symmetric, and the ledger records the effect as
`E_asym`. `mg_fc_symmetric = true` drops the tangential terms, which makes the operator
symmetric. On the octet levels this changes only the convergence rate, not the solution.
`reciprocity_test` solves for two densities and compares cross sums of source and potential,
which are equal exactly when the operator is symmetric. It prints the result and stops the run
before any evolution.

## Known issues

- A restarted run matches the uninterrupted one only to the multigrid tolerance.
- LAT for MHD is not in this release, and the LAT gravity logic reads the hydro source terms only.

## Tests

From `tst/`, run `python run_test_suite.py --cpu --test test_suite/multigrid`, or use `--mpicpu`
or `--gpu` (the suffix in each file name selects the platform). The decks are in `tst/inputs/`.

| files | deck | what they check |
| --- | --- | --- |
| `test_mg_binary_gravity_*.py` | `binary_gravity.athinput` | Two compact spheres. On the CPU the defect reaches 1e-9 in at most 10 V-cycles on uniform meshes (average contraction at most 0.0625) and in at most 13 on a 2-level SMR mesh (at most 0.125). The two contraction rates agree within 0.5 in log10. The GPU and MPI files use other sizes and bounds, and the MPI file compares 4 and 16 ranks. |
| `test_mg_jeans3d_*.py` | `jeans_wave.athinput` | Jeans waves with $\omega^2=k^2c_s^2(1-n_J^2)$. On the CPU the measured $\omega$ is within 1% at $n_J=0.5$ and 3% at $n_J=2$, with the error shrinking with resolution, and iterative solves reach 1e-8 in at most 4 V-cycles and 1e-12 in at most 12. The GPU and MPI files add an AMR run that must create and delete blocks. |
| `test_mg_poisson3d_*.py` | `selfgravity.athinput`, `selfgravity_mhd.athinput` | 64^3 hydro and MHD reach 1e-8 in at most 10 V-cycles on the CPU (average contraction at most 0.07). The final defect does not depend on the MeshBlock size. The MPI file compares 1 and 8 ranks. |

Other gravity tests are in `tst/test_suite/nr/`: `test_nr_star_surface_eps_cpu.py` and
`test_nr_hydro_lat_grav_reflux_gate_cpu.py`.
