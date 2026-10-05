# Dual Energy

## Summary

In fast, cold, supersonic flow the total energy is dominated by kinetic energy, and the
thermal part obtained by subtraction (total minus kinetic) loses precision to round-off, which
gives wrong pressures and temperatures. The dual-energy formalism carries a second,
independently evolved variable that holds the thermal state directly, and the code uses it in
the cells where the subtraction is unreliable. The same idea, with a different variable,
protects relativistic magnetized cells against a magnetic cancellation.

**Use it** for Newtonian hydrodynamics with a tabulated or ideal-gas EOS where gas moves much
faster than the sound speed, as in the tidal disruption decks. All production decks in
`inputs/TDE_examples/` switch it on.

**Do not use it** with special relativity (refused), or with the general-relativistic flavour
for production accretion runs unless a problem is shown to need it, because its benefit there
is not established (see [Practical guidance](#practical-guidance)).

## Quick start

```ini
<hydro>
dual_energy       = true
dual_energy_eta1  = 1.0e-3
dual_energy_eta2  = 1.0e-4
```

These are the values used in `inputs/TDE_examples/tde_01_disruption.athinput`. For MHD use the
same three keys in `<mhd>`.

| key | what it does |
| --- | --- |
| `dual_energy` | Switches the formalism on. It adds one conserved variable. |
| `dual_energy_eta1` | Per-cell switch. The ordinary energy-derived thermal energy is used when it exceeds this fraction of the total energy, otherwise the tracked auxiliary value is used. |
| `dual_energy_eta2` | Resynchronisation threshold. The auxiliary variable is reset from the total energy where that is locally reliable. A value at or below 0 resets everywhere. |

The flavour is not a key. It follows the coordinate system.

## Full parameter table

Each key exists once in `<hydro>` and once in `<mhd>`.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `hydro/dual_energy` | bool | `false` | Enable the formalism for hydro in whichever flavour the coordinates select. Non-relativistic needs an EOS with `use_e = true`. Fixed-metric GR uses the adiabat (ideal gas by construction). SR is refused, and so is `<mesh_refinement>/prolong_primitives = true` under GR. |
| `hydro/dual_energy_eta1` | Real | `1.0e-3` | Use the energy-derived value when the internal energy exceeds eta1 times the total energy, else the tracked value. At or below 0 the energy-derived value is always preferred when positive. |
| `hydro/dual_energy_eta2` | Real | `1.0e-1` | Resync ratio. At or below 0 the resync is unconditional. |
| `mhd/dual_energy` | bool | `false` | Enable for MHD. Non-relativistic needs `use_e = true`. Dynamical GR needs `<mhd>/dyn_eos = ideal`. Fixed-metric GR is ideal gas by construction. SR is refused. |
| `mhd/dual_energy_eta1` | Real | `1.0e-3` | Non-relativistic: use the energy channel when the internal energy exceeds eta1 times the total energy. GR: when the internal energy density exceeds eta1 times the conserved energy without rest mass. At or below 0 the energy channel is always preferred. |
| `mhd/dual_energy_eta2` | Real | `1.0e-1` | Resync ratio. Non-relativistic: against the maximum total energy in the 27-cell neighbourhood. GR: against the maximum of tau over D in the same neighbourhood. At or below 0 the resync is unconditional. |

The GR flavour also reads `<mhd>/sfloor`, the entropy floor of the inversions, as the floor on
the adiabat.

## How it works

There are two flavours. The code picks one from the coordinates, not from a key.

| | Non-relativistic | General-relativistic |
| --- | --- | --- |
| Where | Newtonian hydro and MHD | fixed-metric GR hydro, fixed-metric GR MHD, dynamical GR MHD |
| Auxiliary variable | internal-energy density `eint_aux` | adiabat kappa = p / rho^Gamma, conserved as D times kappa |
| Evolution | de/dt = -p div(v), applied as an operator-split compression step | exact conservation law, no source term |
| Transport | its own upwind flux | the ordinary passive-scalar flux |
| Extra machinery | face-velocity buffers, a LAT companion, primitive prolongation | none |
| Cancellation it fights | total minus kinetic energy in supersonic flow | magnetic (and, in hydro, kinetic) cancellation at large Lorentz factor |

Special relativity has neither flavour and is refused.

**Choosing the pressure in each cell.** At every conserved-to-primitive conversion a ratio
test decides which channel supplies the pressure:

```
eint_cons = total energy - kinetic energy (- magnetic energy)
use the energy channel  if  eint_cons > eta1 * total energy
otherwise               use the auxiliary channel
```

In the code the total energy in this test is floored at 1e-18 (code units) to avoid a zero
denominator. The eta2 test uses the same floor.

The choice is not written back, so the auxiliary keeps evolving on its own equation. In GR
the test is `u_int > eta1 * tau`, and it runs after the energy solve, which is why a smaller
eta1 is preferred in GR than the Newtonian default.

**Resynchronisation.** A second, coarser pass resets the auxiliary variable from the energy
channel wherever that channel is still trustworthy (controlled by eta2). This is what keeps
shock heating: entropy is not conserved across a shock, and only the energy channel knows how
much heat was added.

**Where it fits in a step.**

```
fluxes -> RK update -> dual-energy step -> source terms -> exchange/prolongation
       -> resync (eta2) and conserved-to-primitive (eta1 switch) -> time step
```

The non-relativistic flavour reconstructs the auxiliary with the same schemes as every other
variable. For Newtonian runs with a tabulated EOS, face energies are rebuilt from a limited
specific energy to bound the ratio of internal energy to density across steep density gradients.

## Practical guidance

### Limitations and refusals

The run exits at startup, with one exception noted below.

| refused | reason |
| --- | --- |
| `dual_energy` under special relativity (hydro and MHD) | no implementation for SR. The check is in both `src/hydro/hydro.cpp` and `src/mhd/mhd.cpp` |
| GR hydro with `<mesh_refinement>/prolong_primitives = true` | the adiabat crosses coarse/fine boundaries as D and D times kappa, and the primitive round trip cannot carry it |
| `<mhd>/dyn_eos` other than `ideal` on the dynamical GR path | the adiabat is inverted to a pressure in closed form only for a gamma-law gas |
| an EOS without `use_e` (non-relativistic flavour) | the auxiliary is an internal-energy density |
| Riemann solvers other than `llf`, `hlle`, `hllc`, `roe` (hydro) or `llf`, `hlle`, `hlld` (MHD), non-relativistic flavour only | these solvers supply the interface velocity the compression step needs |
| `<time>/evolution` other than `dynamic` | both flavours are transport-coupled |
| `rvariable = hydro_w_eaux` or `mhd_w_eaux` in `<amr_criterion>` with the formalism off | the criterion reads a column that does not exist. This one is an abort at criterion setup. |

### General-relativistic flavour: benefit not established

The mechanism of the GR flavour is measured and its no-op property is pinned by a test, but
its benefit to a production accretion run has not been shown. On a magnetized funnel the
ratio-test switch at eta1 = 1e-4 can override energy-channel states that are still accurate
and heat the funnel, while a failure-only policy (eta1 = 1e-12) is neutral. Changing eta2 did
not alter this. Leave the GR flavour off unless a problem is shown to need it. Isolated
Lorentz-factor spikes in cold cells at the temperature floor are a floor-policy problem, not an
inversion problem. Dual energy is also not a cure for high magnetization. It moves the limit
from b^2/u to sigma, where the remedies are mass loading (`sigma_ceiling`,
`bsq_over_u_ceiling`).

### Choosing eta1 and eta2

- Keep the defaults or the TDE values (eta1 = 1e-3, eta2 = 1e-4) for Newtonian runs.
- A smaller eta1 uses the auxiliary channel less often. In GR a cell that fails the eta1 test
  runs the inversion twice, so eta1 also sets the cost.
- A positive eta2 resyncs only where the energy-derived value is itself locally reliable,
  which protects the auxiliary near strong shocks. eta2 at or below 0 resyncs everywhere.

### Interactions

- **Cooling source terms.** ISM cooling, relativistic cooling and disk cooling charge the
  auxiliary variable along with the gas. Gravitational work is deliberately not charged.
- **LAT.** Supported. The resync is invoked explicitly after a LAT tick's reflux, and the
  fast paths that skip the final exchange are disabled when dual energy is on.
- **AMR / SMR.** Works. The non-relativistic flavour forces primitive prolongation. The GR
  adiabat declines it. `hydro_w_eaux` and `mhd_w_eaux` are valid `rvariable` keys for the AMR
  criterion (on the GR path the column holds kappa, not an energy).
- **Restart.** Turning dual energy on against a checkpoint that lacks the column seeds it once
  at startup. Turning it off against a checkpoint that has the column prints a notice and drops
  the field. A checkpoint that already carries it is trusted as is.
- **Remap.** The column is carried, floored and reseeded by the remap module. See
  [Remapping](Remapping.md).

### Checking that it works

- Run a high-Mach test with dual energy on and off and compare the pressure. The test problem
  `dual_energy_cancellation` (`src/pgen/tests/`) is described in the
  [implementation notes](Dual-Energy-Implementation-Notes.md#tests). No deck is committed for
  it.
- The regression tests require that at eta1 = 1e-4 the GR formalism changes nothing, and that
  at eta1 = 1 it visibly changes the profile. A run that passes the first check and fails the
  second is a feature that has stopped working.

### Performance

- One extra conserved variable in every array.
- A per-cell mask array whenever the formalism is on, and in MHD one extra scalar array.
- Non-relativistic flavour only: face-velocity buffers, a LAT companion buffer and one extra
  payload in the flux-correction exchange.
- With eta2 above 0, a 27-cell neighbourhood scan per cell once per stage.

## Further reading

- [Implementation notes](Dual-Energy-Implementation-Notes.md): how each flavour works, step
  order, interactions, key files, tests.
- Example decks: `inputs/TDE_examples/*.athinput` (non-relativistic, tabulated EOS, `<hydro>`).
  GR test decks: `tst/inputs/mub1_gr_dual.athinput` (ships with `dual_energy = false`, switched
  on from the test's argument list) and the `tst/inputs/dual_*.athinput` family.
- Kastaun, Kalinani and Ciolfi (2021), the single-unknown GR inversion both auxiliary root
  finds are derived from.
