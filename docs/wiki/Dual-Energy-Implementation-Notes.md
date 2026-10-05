# Dual energy: implementation notes

A short tour of how dual energy is built in the code. Usage, keys and guidance are in
[Dual Energy](Dual-Energy.md).

## The two flavours

The code picks the flavour from the coordinates. In both `Hydro` and `MHD` the member
`dual_energy_pdv` is set to `!relativistic` at construction (`src/hydro/hydro.cpp`,
`src/mhd/mhd.cpp`). It is true for the non-relativistic flavour, which needs a $p\,dV$
compression step, and false for the general-relativistic (GR) flavour, which needs none.
Special relativity has no flavour and is refused in both files.

| | non-relativistic ($p\,dV$) | general-relativistic (adiabat) |
| --- | --- | --- |
| auxiliary variable | internal-energy density | $\kappa = p/\rho^{\Gamma}$, conserved as $D\kappa$ |
| evolution | $de/dt = -p\,\nabla\cdot v$, operator split | exact conservation law, no source |
| flux | its own upwind flux | ordinary passive-scalar flux |
| extra buffers | face velocities, LAT companion, forced primitive prolongation | none |

Both flavours add one conserved column (`naux = 1`) and share the two thresholds. The extra
buffers are allocated for the $p\,dV$ flavour only. Passing them to the flux-correction exchange
for the GR flavour would make the receiver read a view of the wrong size.

## Non-relativistic flavour

1. **Transport.** The auxiliary is reconstructed like every other variable and gets an upwind
   flux $\dot m\,e_{\rm int}/\rho$ (`SetDualEnergyFluxAt`).
2. **Compression step.** After the Runge-Kutta update the auxiliary is multiplied by
   $\exp[-(\gamma-1)\,\nabla\cdot v\,\Delta t]$ for a gamma-law gas, or reduced by
   $p\,\nabla\cdot v\,\Delta t$ otherwise. The divergence uses the interface velocities that the
   Riemann solver already produced (`src/hydro/hydro_dual_energy.cpp`,
   `src/mhd/mhd_dual_energy.cpp`).
3. **Choosing the pressure.** In every conserved-to-primitive conversion the energy channel is
   used if $e_{\rm cons} > \eta_1 \max(E_{\rm tot}, 10^{-18})$, otherwise the auxiliary. The
   $10^{-18}$ is a code-unit floor on the denominator. The choice is not written back, so the
   auxiliary keeps evolving on its own equation. The tabulated-EOS path uses the same test
   (`src/eos/general_c2p_hyd.hpp`, `src/eos/general_c2p_mhd.hpp`).
4. **Resynchronisation.** The auxiliary is reset to the energy-derived value where that value
   is positive and, if $\eta_2 > 0$, exceeds $\eta_2$ times the largest total energy in the
   27-cell neighbourhood. This pass runs after the boundary exchange, at the head of the
   conversion, because it needs the neighbours' new energies.

## General-relativistic flavour

The adiabat is constant along a streamline in smooth flow, so it obeys
$\partial_\mu(\sqrt{-g}\,\rho\,\kappa\,u^\mu)=0$ and needs no source term. The
non-relativistic form was not ported, because its update is non-conservative in GR.

1. **Why it is needed.** The GR inversions recover the gas energy as a small difference of
   two large terms in a magnetically dominated cell. The error grows as
   $\varepsilon_{\rm mach}\,b^2/u$, and the temperature and Lorentz factor come out of the same
   root, so a collapsed pressure also displaces the velocity.
2. **The auxiliary channel.** The same root find is repeated with the specific energy replaced
   by $\kappa\rho^{\Gamma-1}/(\Gamma-1)$, a power instead of a difference
   (`src/eos/ideal_c2p_mhd.hpp`, `src/eos/primitive-solver/primitive_solver.hpp`). The cell keeps
   its conserved energy, so the energy equation is ignored for the pressure, not overwritten.
3. **Switch.** The test is $u_{\rm int} > \eta_1\,\tau$, with $\tau$ the conserved energy
   without rest mass. It runs after the energy solve, so a cell that fails it is inverted twice.
   If the auxiliary solve fails, the energy-channel answer stands.
4. **Resync.** Written per unit rest mass, the test is $\varepsilon > \eta_2\max_{27}(\tau/D)$.
   The inversions publish $\tau/D$ because the conserved array does not hold it. The pass
   (`SynchronizeDualEnergyFieldFromAdiabat`) runs before the conserved send, over owned cells.
   It also caps $\kappa$ at the energy the cell owns.
5. **Seeding.** The adiabat is seeded from primitives after the first inversion, and over the
   ghost cells as well, because $\kappa$ needs $p$ and $\rho$.
6. **Repairs.** A mass or energy floor, or the magnetization ceiling, changes $\kappa$, so those
   cells republish it from the repaired primitives.

Measured on exact conserved states, the energy channel fails outright from
$b^2/u \approx 7\times10^{9}$ and returns silent garbage beyond $10^{16}$. The auxiliary channel
never failed and was about 20 to 1000 times more accurate, but it is limited by the
magnetization $\sigma$ rather than by $b^2/u$.

## How the pieces fit in a step

```
fluxes -> RK update -> dual-energy step -> source terms -> exchange / prolongation
       -> resync (eta2) -> conserved-to-primitive (eta1 switch) -> new time step
```

In MHD the dual-energy step sits after the constrained-transport field update. The dynamical GR
graph has the same step as its own task (`src/dyn_grmhd/dyn_grmhd.cpp`).

## Interactions

- **Cooling.** The three cooling terms charge the auxiliary along with the gas. The
  non-relativistic auxiliary takes the same energy decrement. The GR adiabat takes a
  multiplicative one, because cooling lowers $p$ at fixed $\rho$ (`src/srcterms/srcterms.cpp`).
- **LAT.** After a tick's flux correction the resync is called explicitly on the receiver
  blocks, and the fast paths that skip the final exchange are disabled
  (`src/driver/driver.cpp`). Coarse/fine corrections carry a companion divergence term for the
  $p\,dV$ flavour only.
- **AMR.** The $p\,dV$ flavour forces primitive prolongation. The GR adiabat declines it,
  because $D$ and $D\kappa$ cross a coarse/fine boundary like passive scalars. Newly refined
  cells have the auxiliary re-floored after a regrid.
- **Restart.** The width of the stored conserved array tells `src/pgen/pgen.cpp` whether the
  column exists. A missing column is seeded once, and an unwanted one is dropped with a notice.
- **Remap.** The remap module floors the auxiliary in the outer ghost ring and reseeds it
  (see [Remapping](Remapping.md)).
- **FOFC.** The first-order fallback also re-solves the auxiliary face flux. The dynamical GR
  trial state includes the auxiliary column, otherwise the trial switch could never fire.

## Refusals

The full list with reasons is on the [main page](Dual-Energy.md#limitations-and-refusals).
Special relativity is refused in `src/hydro/hydro.cpp` and `src/mhd/mhd.cpp`. The other checks
are in the same two constructors. The `hydro_w_eaux` and `mhd_w_eaux` criteria abort in
`src/mesh/refinement_criteria.cpp`.

## Key files

| file | role |
| --- | --- |
| `src/hydro/hydro_dual_energy.cpp`, `src/mhd/mhd_dual_energy.cpp` | seeding, compression step, resyncs, repair after refinement |
| `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp` | key parsing, flavour choice, refusals, buffers |
| `src/eos/general_c2p_hyd.hpp`, `general_c2p_mhd.hpp` | the $\eta_1$ switch for tabulated EOS |
| `src/eos/ideal_hyd.cpp`, `ideal_mhd.cpp` | the same switch for ideal gas |
| `src/eos/ideal_grmhd.cpp`, `src/eos/primitive_solver_hyd.hpp` | GR switch, publication of $\kappa$ and $\tau/D$ |
| `src/eos/ideal_c2p_mhd.hpp` | GR adiabat root find |
| `src/srcterms/srcterms.cpp` | cooling charge |
| `src/pgen/tests/dual_energy_cancellation.cpp` | high-Mach stress test |

## Tests

- `tst/test_suite/gr/test_gr_dual_energy_cpu.py` (deck `tst/inputs/mub1_gr_dual.athinput`) and
  `test_grhd_dual_energy_cpu.py` (deck `tst/inputs/dual_grhd_tube.athinput`). Each checks both
  directions. At $\eta_1 = 10^{-4}$ the formalism must change nothing (relative density
  difference below $10^{-13}$). At $\eta_1 = 1$ the profile must visibly move.
- `dual_energy_cancellation` is a 1-D high-Mach entropy wave that defeats the energy
  subtraction. No deck is committed for it. Set `<problem>/pgen_name = dual_energy_cancellation`
  and run with `dual_energy` true and false. The problem keys are `rho0`, `pressure`,
  `bulk_velocity` and `density_amplitude`, and it needs ideal gamma-law hydro.
- The `inputs/TDE_examples/*.athinput` decks use the non-relativistic flavour in production.
