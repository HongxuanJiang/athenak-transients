# Dual Energy

## Summary

In fast, cold, supersonic flow almost all of the total energy is kinetic. The code finds the
thermal energy by subtracting the kinetic energy from the total, and in this regime that
subtraction loses its precision to round-off, so pressures and temperatures come out wrong.
The dual-energy formalism avoids this by evolving a second variable, the *auxiliary*, which
holds the thermal state directly. In every cell where the subtraction is unreliable, the code
takes the pressure from the auxiliary instead. General-relativistic runs use the same idea
with a different auxiliary, for cells where the GR inversion loses the thermal energy to a
cancellation (against the magnetic energy, in MHD).

**Use it** for Newtonian hydrodynamics or MHD with an ideal-gas or tabulated EOS in which the
gas moves much faster than the sound speed, as in the tidal disruption decks. All five decks in
`inputs/TDE_examples/` switch it on.

**Do not use it** with special relativity: it is refused at startup. Also leave the
general-relativistic flavour off in production runs unless a problem is shown to need it,
because this repository has no production-style GR run or test that shows a benefit (see
[General-relativistic flavour: what is tested](#general-relativistic-flavour-what-is-tested)).

## Quick start

```ini
<hydro>
dual_energy       = true
dual_energy_eta1  = 1.0e-3
dual_energy_eta2  = 1.0e-4
```

These are the values used in `inputs/TDE_examples/tde_01_disruption.athinput`, and all five TDE
decks set them. The built-in default of `dual_energy_eta1` is the same, but the built-in default
of `dual_energy_eta2` is `1.0e-1`, so a TDE-style run has to set `dual_energy_eta2 = 1.0e-4`
itself. For MHD, put the same three keys in `<mhd>`.

- `dual_energy` switches the formalism on and adds one conserved variable.
- `dual_energy_eta1` decides, cell by cell, whether the pressure comes from the total energy
  or from the auxiliary.
- `dual_energy_eta2` decides where the auxiliary is reset from the total energy.

Both thresholds are explained under [How it works](#how-it-works). The flavour is not a key.
It follows the coordinate system: Newtonian runs get the internal-energy auxiliary and
general-relativistic runs get the adiabat auxiliary.

## Full parameter table

Each key exists once in `<hydro>` and once in `<mhd>`.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `hydro/dual_energy` | bool | `false` | Turn the formalism on for hydro. Newtonian hydro needs an EOS other than `isothermal`. Fixed-metric GR hydro uses the adiabat (GR hydro is ideal gas only). Refused at startup under special relativity. Also refused: `<mesh_refinement>/prolong_primitives = true` under GR (see [Limitations and refusals](#limitations-and-refusals)). |
| `hydro/dual_energy_eta1` | Real | `1.0e-3` | Per-cell switch. The pressure comes from the total energy when the thermal energy obtained from it is larger than eta1 times the total energy (Newtonian) or times the conserved energy without rest mass (GR). Otherwise it comes from the auxiliary. At or below 0 the total energy is always used when it gives a positive thermal energy. |
| `hydro/dual_energy_eta2` | Real | `1.0e-1` | Resync threshold. The auxiliary is reset from the total energy where the thermal energy obtained from it is larger than eta2 times the largest total energy among the neighbouring cells, up to 27 (in GR the same comparison per unit rest mass). At or below 0 it is reset in every cell with a positive thermal energy. The TDE decks set `1.0e-4`. |
| `mhd/dual_energy` | bool | `false` | Turn the formalism on for MHD. Newtonian MHD needs an EOS other than `isothermal`. Dynamical GR needs `<mhd>/dyn_eos = ideal`. Fixed-metric GR MHD is ideal gas by construction. Refused at startup under special relativity. |
| `mhd/dual_energy_eta1` | Real | `1.0e-3` | As for hydro. Newtonian MHD additionally requires the thermal energy to exceed a fixed 10% of the magnetic energy (not a key) before the total energy is trusted. At or below 0 both tests are skipped. |
| `mhd/dual_energy_eta2` | Real | `1.0e-1` | As for hydro. In Newtonian MHD a cell is also reset only if its energy channel passes the `eta1` and magnetic tests. Shocked cells that fail only the magnetic test are reset anyway (see [Resynchronisation](#resynchronisation)). |

### Floor keys the auxiliary obeys

These are not dual-energy keys. They live in the same block (`<hydro>` or `<mhd>`) and they
also bound the auxiliary, in different ways for the two flavours. `FLT_MIN` is a tiny positive
number, so the default floors are effectively off.

| key | default | Newtonian flavour | GR flavour |
| --- | --- | --- | --- |
| `sfloor` | `FLT_MIN` | Entropy floor on the internal-energy density. Ideal gas only. A tabulated EOS requires `sfloor = 0` (see [Tabulated EOS](Tabulated-EOS)). | Lower bound on the adiabat. The fixed-metric inversions also apply it to the pressure the adiabat supplies. |
| `pfloor` | `FLT_MIN` | Internal-energy density floor, `pfloor` divided by gamma minus 1 (ideal gas). | Applied to the pressure the adiabat supplies, in the fixed-metric inversions. |
| `tfloor` | `FLT_MIN` | Temperature floor on the auxiliary. Ideal gas only. A tabulated EOS enforces its own table floor instead (see [Tabulated EOS](Tabulated-EOS)). | Not applied to the adiabat in the fixed-metric inversions. |
| `cs_ceil` | `0` (off) | When above 0, an upper bound on the auxiliary internal-energy density. | Not applied to the adiabat. |

In GR the adiabat also has an upper bound that you cannot set: the internal energy it implies
may not exceed the energy the cell owns. Nothing in the transport limits the adiabat from above,
and a cell whose density sits at its floor can otherwise produce a huge value.

## How it works

There are two flavours. The code picks one from the coordinates, not from a key.

| | Newtonian | General-relativistic |
| --- | --- | --- |
| Where | Newtonian hydro and MHD | fixed-metric GR hydro, fixed-metric GR MHD, dynamical GR MHD |
| Auxiliary | internal-energy density | the adiabat $\kappa = p/\rho^{\Gamma}$, conserved as $D\kappa$ ($D$ is the conserved density) |
| How it evolves | $\mathrm{d}e/\mathrm{d}t = -p\,\nabla\cdot v$, applied as a separate compression step | exact conservation law, no source term |
| Transport | its own upwind flux | the ordinary passive-scalar flux |
| Extra machinery | face-velocity buffers, a [LAT](Local-Adaptive-Time-Stepping) companion, primitive prolongation | none |
| Cancellation it fights | total minus kinetic energy in supersonic flow | loss of the thermal energy in the GR inversion, against the magnetic energy in MHD |

Special relativity has neither flavour. It is refused at startup (see
[Limitations and refusals](#limitations-and-refusals)).

### Which channel gives the pressure

At every conversion from conserved to primitive variables, a ratio test decides which channel
supplies the pressure. The *energy channel* is the thermal energy obtained by subtraction. The
*auxiliary channel* is the auxiliary itself.

```
thermal energy = total energy - kinetic energy (- magnetic energy)
use the energy channel   if  thermal energy > eta1 * total energy
otherwise                use the auxiliary channel
```

In Newtonian MHD the energy channel must also pass a second test, because at low plasma beta
the truncation error of the magnetic energy exceeds the thermal energy. It is used only if the
thermal energy is larger than 10% of the magnetic energy. In GR the test compares the thermal
energy density with eta1 times the conserved energy without rest mass.

The choice is not written back into the auxiliary, so the auxiliary keeps evolving on its own
equation. In GR the energy solve runs first, and a cell that fails the test is then solved a
second time from the adiabat. The GR test decks in `tst/inputs/` use eta1 = 1e-4, smaller than
the default, so that fewer cells need the second solve.

### Resynchronisation

A second pass resets the auxiliary from the energy channel wherever that channel is still
trustworthy. The eta2 test decides where. This is what keeps shock heating: entropy is not
conserved across a shock, and only the energy channel knows how much heat the shock added.

In Newtonian MHD a cell that fails only the magnetic test is still reset from the energy channel
if the flow converges supersonically across the cell, so a shock in low-beta gas does not lose
its heating.

### Where it fits in a step

```
fluxes -> RK update -> dual-energy step -> source terms -> exchange and prolongation
       -> resync (eta2) and conserved-to-primitive (eta1 test) -> time step
```

This is the Newtonian order. In GR the resync moves into the dual-energy step, ahead of the
exchange. The [implementation notes](Dual-Energy-Implementation-Notes#how-a-step-runs) give
both orders.

The Newtonian flavour reconstructs the auxiliary with the same schemes as every other
variable. With a tabulated EOS, how the face energies are formed is described under Face states
in [Tabulated EOS](Tabulated-EOS).

## Practical guidance

### Limitations and refusals

Each of these stops the run with a fatal error at startup.

| refused | reason |
| --- | --- |
| `dual_energy = true` under special relativity (`<coord>/special_rel = true`), in both `<hydro>` and `<mhd>` | neither flavour is implemented for SR |
| GR hydro with `<mesh_refinement>/prolong_primitives = true` | the adiabat crosses coarse/fine boundaries as $D$ and $D\kappa$, and the round trip through primitives cannot carry it |
| `<mhd>/dyn_eos` other than `ideal` on the dynamical GR path | the adiabat is turned back into a pressure in closed form only for a gamma-law gas |
| an EOS without internal energy (`isothermal`), Newtonian flavour | the auxiliary is an internal-energy density |
| Riemann solvers other than `llf`, `hlle`, `hllc`, `roe` (hydro) or `llf`, `hlle`, `hlld` (MHD), Newtonian flavour only | only these solvers are supported. The compression step uses the face velocity they store. |
| `<time>/evolution` other than `dynamic` | dual energy is implemented only for dynamic evolution |
| `rvariable = hydro_w_eaux` or `mhd_w_eaux` in `<amr_criterion>` with the formalism off | the criterion reads a column that does not exist. This one is an abort when the criterion is set up. |

### General-relativistic flavour: what is tested

No deck in `inputs/` turns the GR flavour on. It is exercised only by the tests in `tst/`
(listed under [Checking that it works](#checking-that-it-works)), which use small shock-tube,
refined-mesh, restart and cooling problems. These tests show that the formalism changes nothing
at eta1 = 1e-4 on their problems, and that it takes over at eta1 = 1. This repository has no
production-style GR run or test that shows a benefit from it, so leave it off unless your
problem is shown to need it.

A GR cell that takes its pressure from the adiabat keeps its conserved energy. The energy
equation is ignored for the pressure in that cell, not overwritten.

### Choosing eta1 and eta2

- Start from the TDE values, eta1 = 1e-3 and eta2 = 1e-4. Remember that the default of eta2 is
  1e-1, so eta2 must be set explicitly.
- A smaller eta1 means the energy channel passes the test in more cells, so the auxiliary is used
  less often. In GR a cell that fails the test is solved twice, so a smaller eta1 also makes the
  formalism cheaper.
- A larger positive eta2 resets the auxiliary only where the thermal energy from the total energy
  is a large share of the largest total energy nearby, which is where the subtraction is least
  likely to have lost precision. At or below 0 the auxiliary is reset everywhere the energy
  channel gives a positive thermal energy.

### Interactions

- **Cooling.** ISM cooling, relativistic cooling and disk cooling take their energy loss out of
  the auxiliary as well as out of the gas. Gravitational work is deliberately not charged.
- **LAT.** Supported for hydro (see [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping)). After
  a LAT tick's flux correction, the receiving blocks go through the normal conversion step, which
  runs the resync. For the Newtonian flavour the shortcut paths that skip the final boundary
  exchange are switched off.
- **AMR and SMR.** Supported. The Newtonian flavour forces AMR to prolong primitives, and the GR
  adiabat declines to. `hydro_w_eaux` and `mhd_w_eaux` are valid `rvariable` keys for the AMR
  criterion (on the GR path the column holds the adiabat, not an energy).
- **Restart.** Turning dual energy on against a checkpoint that lacks the column seeds it once
  at startup. Turning it off against a checkpoint that has the column prints a notice and drops
  the field. A checkpoint that already carries it is used as it is.
- **Remap.** In a Newtonian remap the column is carried in both band modes. With
  `band_mode = keep`, the startup reseed from the total energy seeds the target's own
  problem-generator cells. With `band_mode = floor`, the remap writes the column of every target
  cell itself (the source's auxiliary where it has one, else the thermal energy of the remapped
  state) and no reseed follows. A GR source gets no special handling of the column. See
  [Remapping](Remapping) and [Remap usage](Remap-Usage).

### Checking that it works

These are the tests and decks that exist. Run a test from `tst/` with, for example,
`python run_test_suite.py --cpu --test test_suite/gr/test_gr_dual_energy_cpu.py`.

- **GR on/off check.** `tst/test_suite/gr/test_gr_dual_energy_cpu.py` (deck
  `tst/inputs/mub1_gr_dual.athinput`, MHD) and `tst/test_suite/gr/test_grhd_dual_energy_cpu.py`
  (deck `tst/inputs/dual_grhd_tube.athinput`, hydro) require two things. At eta1 = 1e-4 the
  formalism must change nothing. At eta1 = 1 it must visibly change the profile. A run that
  passes the first check and fails the second is a feature that has stopped working. Both decks
  ship with `dual_energy = false`, and the tests switch it on from the command line.
- **Other GR MHD tests.** `test_gr_dual_energy_aux_cpu.py`, `test_gr_dual_energy_amr_cpu.py`,
  `test_gr_dual_energy_restart_cpu.py` and `test_gr_dual_energy_cooling_cpu.py`, all in
  `tst/test_suite/gr/`, use the `tst/inputs/dual_*.athinput` decks (tube, AMR and cooling).
  They check the auxiliary column itself, a refined mesh, restarts and cooling.
- **Newtonian low-beta shock.** `tst/test_suite/nr/test_nr_mhd_lowbeta_shock_dual_cpu.py` (deck
  `tst/inputs/mhd_bw_lowbeta_dual.athinput`) checks that a low-beta shock keeps its heating.
- **Newtonian star atmosphere.** `tst/test_suite/nr/test_nr_star_surface_eps_cpu.py` checks that
  the atmosphere of a star, which nothing heats, does not heat. Its decks
  (`tst/inputs/hydro_star_surface_eps*.athinput`, `mhd_star_surface_eps*.athinput` and
  `mhd_star_lowbeta_hlld*.athinput`) have `dual_energy = true`, and the low-beta ones
  exercise the magnetic test. `tst/test_suite/unit_tests/test_hydro_plm_cpu.py` also runs the
  hydro unit deck `tst/inputs/hydro_plm_unit.athinput` with dual energy on and off.
- **High-Mach cancellation problem.** `src/pgen/tests/dual_energy_cancellation.cpp` is a
  problem generator for a high-Mach entropy wave with uniform pressure. It needs ideal-gas
  hydro, and its history output compares the pressure from the inversion, from subtracting the
  kinetic energy, and from the auxiliary. It is not one of the built-in generators, and no deck
  or test uses it. To use it, configure the build with `-D PROBLEM=dual_energy_cancellation` and
  write your own deck. The `<problem>` keys are `rho0` (1.0), `pressure` (1.0),
  `bulk_velocity` (1.0e8) and `density_amplitude` (0.1).

### Performance

- One extra conserved variable in every array.
- Two extra per-cell arrays whenever the formalism is on, in both hydro and MHD: a mask and a
  cached maximum energy.
- Newtonian flavour only: face-velocity buffers, a LAT companion buffer, and one extra
  variable in the flux-correction exchange.
- With eta2 above 0, a scan of the neighbouring cells (up to 27) for every cell once per stage.

## Further reading

- [Implementation notes](Dual-Energy-Implementation-Notes): how the two flavours are built,
  the order of a step, known limits and tests.
- Example decks: `inputs/TDE_examples/*.athinput` (Newtonian, tabulated EOS, `<hydro>`). GR test
  decks: `tst/inputs/mub1_gr_dual.athinput`, `tst/inputs/dual_*.athinput`. Newtonian MHD test
  deck: `tst/inputs/mhd_bw_lowbeta_dual.athinput`.
- Kastaun, Kalinani and Ciolfi (2021), the single-unknown GR inversion that both auxiliary root
  finds are derived from.
