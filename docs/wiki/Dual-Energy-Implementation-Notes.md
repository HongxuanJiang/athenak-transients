# Dual energy: implementation notes

Back to [Dual Energy](Dual-Energy) for usage, parameters and guidance.

This page explains how the two flavours are built, how they fit into a time step, and where
their limits are, for readers who want to follow or change the code.

## One formalism, two flavours

Both flavours add one conserved variable next to the total energy. At every conversion from
conserved to primitive variables, both decide cell by cell whether the pressure comes from the
total energy or from that extra variable. They differ in what the variable is and how it evolves
(see [How it works](Dual-Energy#how-it-works)).

The code picks the flavour once, in the constructors of `Hydro` and `MHD`. Each class has a flag
`dual_energy_pdv`, which is not a deck key. The constructor sets it to true when the run is not
relativistic and to false when it is. Here "relativistic" means special or general relativity,
and for MHD also dynamical GR. True selects the Newtonian flavour, whose auxiliary is an
internal-energy density evolved with a $p\,\mathrm{d}V$ work step. False selects the GR flavour,
whose auxiliary is the adiabat. Special relativity never gets as far as either value, because
both constructors refuse `dual_energy = true` there (see
[Limitations and refusals](Dual-Energy#limitations-and-refusals)).

## The Newtonian flavour

The auxiliary `eint_aux` is the internal-energy density. In each stage it goes through three
steps.

1. The Riemann solver's mass flux carries it, with its own upwind flux. The solver also stores
   the velocity at every cell face: HLLC and HLLD give their contact velocity, and every other
   solver uses the upwind velocity, the mass flux divided by the density.
2. After the Runge-Kutta update, a compression step applies
   $\mathrm{d}e/\mathrm{d}t = -p\,\nabla\cdot v$, with the divergence taken from those face
   velocities. For a gamma-law gas the step is exact, `eint *= exp(-(gamma - 1) divv dt)`. For any
   other EOS it is an explicit Euler step with the EOS pressure.
3. Floors and ceilings are applied before and after the compression step, so a tabulated EOS never
   sees an out-of-table state.

## The GR flavour

The auxiliary is the adiabat $\kappa = p/\rho^{\Gamma}$, conserved as $D\kappa$. In smooth
adiabatic flow $\kappa$ does not change along a streamline, so $D\kappa$ obeys an exact
conservation law. It needs no source term and travels with the ordinary passive-scalar flux.
The operator-split $p\,\mathrm{d}V$ step of the Newtonian flavour is a Newtonian construction, so
GR uses this exactly conserved variable instead.

When the eta1 test rejects the energy channel in a cell, the inversion is run again from the
adiabat. It is the same root find, but the specific energy is taken as
$\varepsilon = \kappa\rho^{\Gamma-1}/(\Gamma-1)$. That is a power of the density, not a difference
of two large numbers, so it avoids the cancellation. The same second solve is the backstop when
the energy solve fails outright, and a cell it rescues is not counted as a failure.

The eta2 resync compares the thermal energy per unit rest mass with the largest $\tau/D$ among
the neighbours, where $\tau$ is the conserved energy without rest mass. This form needs no
metric. The resync code has no metric, though, and cannot build $\tau$ from the conserved
array, because on the fixed-metric MHD path that slot holds $T^t{}_t + D$, which is negative for
ordinary states. So each inversion publishes its own $\tau/D$ for every cell, and the resync
reads that. The inversion also caps the specific energy at $\tau/D$, and the resync caps the
transported column at the same budget once per stage.

- **Seeding.** The adiabat depends on the primitives, so it is seeded after the first
  conversion, ghost zones included. The Newtonian seeding does nothing on the GR path, so its
  caller must not mark the seeding as done.
- **Repairs.** After a density floor, an energy floor or (in MHD) the magnetization ceiling, the
  adiabat is recomputed from the repaired primitives. A Lorentz-factor ceiling leaves it alone,
  because it changes no thermodynamic quantity.
- **First-order flux correction.** The adiabat rides the first-order scalar flux. The Newtonian
  flavour has its own first-order flux.

## How a step runs

```
Newtonian flavour                          GR flavour
fluxes (+ face velocities)                 fluxes (adiabat rides the scalar flux)
RK update (MHD: also the CT update)        RK update (MHD: also the CT update)
compression step                           eta2 resync of the adiabat
              source terms (cooling charges the auxiliary)
              exchange, restriction, prolongation, boundary conditions
eta2 resync, then conversion               conversion with the eta1 test
with the eta1 switch                       (publishes the adiabat and tau/D)
              new time step
```

The dual-energy step comes right after the update (for MHD, after the update of the magnetic
field). In the dynamical-GR task graph it is its own task, ahead of the send. On a multilevel
mesh, restriction also copies the restricted auxiliary into the coarse primitive array.

The two resyncs sit at different places because they read different data. The Newtonian one needs
the neighbours' new total energies, so it runs after the exchange, at the start of the conversion
step. The GR one reads only data the rank already holds, so it runs before the send, and the
exchange then carries the resynced adiabat into every ghost cell like any other conserved
variable.

## Interactions with other modules

- **Cooling.** The Newtonian auxiliary is an internal-energy density, so it loses the same energy
  as the gas. The GR auxiliary is an adiabat. Cooling changes $p$ at fixed density, so $\kappa$
  scales with the internal energy, and the debit is multiplicative with a floor on the factor.
- **LAT.** After a LAT tick's flux correction, the conversion step runs the resync on the
  receiving blocks, because the flux correction rewrote their total energy. For the Newtonian
  flavour the shortcut paths that skip the final boundary exchange are switched off, because the
  resync needs current neighbour energies. The coarse/fine flux correction also carries a
  companion divergence term, for the Newtonian flavour only.
- **AMR.** The Newtonian flavour forces primitive prolongation. The GR adiabat declines it,
  because prolonging $D$ and $D\kappa$ separately and dividing is how passive scalars already
  cross a coarse/fine boundary. After a regrid, newly refined cells get the auxiliary floored again.
- **Remap.** In a Newtonian remap the auxiliary column is carried in both band modes. A keep-target
  pass leaves `dual_energy_needs_init` true, so the startup reseed seeds the target's own cells.
  A floor pass writes the column in every target cell itself and clears the flag, so nothing is
  reseeded. A GR source is not handled. See [Remapping](Remapping) and
  [Remap usage](Remap-Usage).

## Known limits

- `dual_vf` and the LAT companion buffer exist only for the Newtonian flavour, so the
  flux-correction calls must pass them only then. Passing them for GR would hand the exchange a
  placeholder array and cause an out-of-bounds write on every coarse/fine face.
- Most refusals are constructor checks that exit at startup, and the AMR criterion aborts when it
  is set up (full list on the main page). For what the GR tests show, see
  [General-relativistic flavour: what is tested](Dual-Energy#general-relativistic-flavour-what-is-tested).

## Tests

How to run a test is on the main page, under
[Checking that it works](Dual-Energy#checking-that-it-works). Tests are in `tst/test_suite/`,
decks in `tst/inputs/`.

| test | deck | what it pins |
| --- | --- | --- |
| `gr/test_gr_dual_energy_cpu.py` | `mub1_gr_dual` | GR MHD tube: no change at eta1 = 1e-4, a visible change at eta1 = 1 |
| `gr/test_grhd_dual_energy_cpu.py` | `dual_grhd_tube` | The same two-sided check for GR hydro |
| `gr/test_gr_dual_energy_aux_cpu.py` | `dual_gr_tube`, `dual_dyngr_tube` | GR MHD, fixed-metric and dynamical: the auxiliary is seeded, bounded and advected |
| `gr/test_gr_dual_energy_restart_cpu.py` | `dual_gr_tube`, `dual_dyngr_tube` | GR MHD: a restarted run reproduces the uninterrupted one |
| `gr/test_gr_dual_energy_amr_cpu.py` | `dual_gr_amr` | GR MHD on a regridding mesh: same history as with the formalism off |
| `gr/test_gr_dual_energy_cooling_cpu.py` | `dual_gr_cooling` | GR MHD: cooling debits the auxiliary together with the gas |
| `nr/test_nr_mhd_lowbeta_shock_dual_cpu.py` | `mhd_bw_lowbeta_dual` | A Newtonian low-beta shock keeps its heating |

The decks `mub1_gr_dual`, `dual_grhd_tube` and `dual_gr_amr` ship with `dual_energy = false`, and
the tests switch it on from the command line.

## Key files

| file | role |
| --- | --- |
| `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp` | Read the three keys, pick the flavour, run the refusals, allocate the buffers |
| `src/hydro/hydro_dual_energy.cpp`, `src/mhd/mhd_dual_energy.cpp` | Seeding, compression step, both resyncs, repairs after regrid |
| `src/hydro/hydro_fluxes_impl.hpp`, `src/mhd/mhd_fluxes_impl.hpp` | The Newtonian auxiliary flux and face velocities |
| `src/eos/ideal_hyd.cpp`, `src/eos/ideal_mhd.cpp`, `src/eos/general_c2p_hyd.hpp`, `src/eos/general_c2p_mhd.hpp` | The eta1 switch and the floors for the Newtonian flavour |
| `src/eos/ideal_grhyd.cpp`, `src/eos/ideal_grmhd_c2p_impl.hpp`, `src/eos/primitive_solver_hyd.hpp` | The GR eta1 test and second solve, and publication of the adiabat and tau/D (fixed-metric hydro, fixed-metric MHD, dynamical) |
| `src/eos/ideal_c2p_hyd.hpp`, `src/eos/ideal_c2p_mhd.hpp` | The adiabat versions of the fixed-metric root finds |
| `src/srcterms/srcterms.cpp`, `src/pgen/pgen.cpp`, `src/driver/driver.cpp` | Cooling debits, restart detection, seeding, LAT handling |
