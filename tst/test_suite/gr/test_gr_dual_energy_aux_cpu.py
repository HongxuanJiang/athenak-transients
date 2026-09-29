"""
The GR dual-energy auxiliary, read off the auxiliary column itself.

test_gr_dual_energy_cpu.py can only see the auxiliary through its effect on the gas,
which is why an auxiliary that was never seeded, or one that ran away to 1e83, or one
that was silently recomputed every stage instead of advected, all looked the same from
outside: a tube that either moved or did not.  The auxiliary is now dumpable -- kappa in
the primitive column eint_aux, D*kappa in the conserved column reint_aux -- so the three
properties that define the GR flavour can be asserted directly.

  seeded      kappa = p/rho^Gamma in every cell of the initial dump.  The adiabat is a
              function of the primitives, so it is seeded after the first inversion; a
              seed that never runs leaves the column at its zero initialization and hands
              p = 0 to every cell the eta1 test routes to it.
  bounded     forcing the channel into every cell must stay on a physical state.  Nothing
              in the transport bounds kappa from above, and a floored density turns
              kappa = (D kappa)/D into a finite numerator over a floor; the bound is that
              a cell's specific internal energy cannot exceed its specific total energy.
  advected    the eta2 gate decides where the advected adiabat is replaced by the one the
              energy channel implies.  A gate that is really unconditional throws the
              advected adiabat away in every cell of every stage, and the only evidence is
              that eta2 stops changing anything.

Both GR backends are covered: the fixed-metric one publishes internal energy density in
the pressure slot, the dynamical one publishes pressure, and the resynchronisation pass
used to be called on only one of them.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np
import glob

# deck -> (input file, primitive pressure-slot column, is that column a pressure)
_decks = {
    "fixed": ("inputs/dual_gr_tube.athinput", "eint", False),
    "dyn": ("inputs/dual_dyngr_tube.athinput", "press", True),
}
_gamma = 2.0  # both decks


def run_case(deck, tag, extra):
    """Run one case and return its sorted list of primitive dumps."""
    testutils.run(deck, [f"job/basename={tag}"] + extra)
    dumps = sorted(glob.glob(f"tab/{tag}.mhd_w.*.tab"))
    assert dumps, f"{tag} produced no primitive output"
    return dumps


def pressure(data, column, is_pressure):
    """Gas pressure from whichever quantity this backend publishes in the slot."""
    return data[column] if is_pressure else (_gamma - 1.0)*data[column]


def internal_energy(data, column, is_pressure):
    """Internal energy density from the same slot."""
    return data[column]/(_gamma - 1.0) if is_pressure else data[column]


@pytest.mark.parametrize("flavour", list(_decks))
def test_seeded(flavour):
    """The adiabat exists before the first stage, and equals p/rho^Gamma."""
    deck, column, is_pressure = _decks[flavour]
    try:
        initial = athena_read.tab(run_case(deck, f"dual_seed_{flavour}", [])[0])
        assert initial["time"] == 0.0, "first dump is not the initial condition"

        kappa = initial["eint_aux"]
        if not np.all(kappa > 0.0):
            pytest.fail(
                "dual_energy auxiliary was not seeded: "
                f"{np.count_nonzero(kappa <= 0.0)} of {kappa.size} cells are <= 0"
            )

        # Both sides are the same closed-form expression in double precision, evaluated
        # in a different order (p*pow(rho,-Gamma) against p/rho**Gamma), so they agree to
        # a few ulp.  An unseeded auxiliary is exactly zero: relative error 1.
        expect = pressure(initial, column, is_pressure)/initial["dens"]**_gamma
        error = np.max(np.abs(kappa - expect)/expect)
        if error > 1.0e-12:
            pytest.fail(
                "seeded adiabat does not equal p/rho^Gamma: "
                f"max relative difference {error:g}"
            )
    finally:
        testutils.cleanup()


def test_adiabat_bounded():
    """Forcing the channel into every cell stays on a state a fluid can be in."""
    deck, column, is_pressure = _decks["fixed"]
    try:
        run_case(deck, "dual_bound_off", ["mhd/dual_energy=false"])
        run_case(deck, "dual_bound_on",
                 ["mhd/dual_energy=true", "mhd/dual_energy_eta1=1.0e30"])
        ref = athena_read.tab(sorted(glob.glob("tab/dual_bound_off.mhd_u.*.tab"))[-1])
        cons = athena_read.tab(sorted(glob.glob("tab/dual_bound_on.mhd_u.*.tab"))[-1])
        prim = athena_read.tab(sorted(glob.glob("tab/dual_bound_on.mhd_w.*.tab"))[-1])

        for name, column_data in prim.items():
            if isinstance(column_data, np.ndarray) and not np.all(
                    np.isfinite(column_data)):
                pytest.fail(f"forced auxiliary channel produced non-finite {name}")

        # Waves have not reached the outflow boundaries at tlim, so the tube neither
        # gains nor loses mass except through the floors.  The runaway this bounds
        # reached 2.4e44 before collapsing to 1.9e-35 of the reference.
        mass = np.sum(cons["dens"])/np.sum(ref["dens"])
        if not 0.5 < mass < 2.0:
            pytest.fail(
                "forced auxiliary channel did not conserve mass: "
                f"total D is {mass:g} times the reference"
            )

        # tau/D and u/rho are both specific energies, so no metric enters their ratio.
        # No cell's specific internal energy can exceed the largest specific total energy
        # the tube holds; the factor of ten absorbs the floors, and leaves the measured
        # unbounded case (eps/q ~ 1e80) eighty orders outside.  The budget is read from
        # the REFERENCE run: the forced run's own conserved energy is regenerated from the
        # very primitives being tested, so measured against itself the bound could never
        # fail.
        specific_internal = np.max(
            internal_energy(prim, column, is_pressure)/prim["dens"])
        specific_total = np.max(np.abs(ref["ener"])/ref["dens"])
        if specific_internal > 10.0*specific_total:
            pytest.fail(
                "adiabat implies more internal energy than the cell's budget: "
                f"max u/rho = {specific_internal:g} against max |tau|/D = "
                f"{specific_total:g}"
            )
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("flavour", list(_decks))
def test_eta2_gate(flavour):
    """eta2 decides whether the adiabat is advected or taken from the energy channel."""
    deck, _, _ = _decks[flavour]
    try:
        resync = athena_read.tab(run_case(
            deck, f"dual_eta2_resync_{flavour}",
            ["mhd/dual_energy=true", "mhd/dual_energy_eta2=0.0"])[-1])
        advect = athena_read.tab(run_case(
            deck, f"dual_eta2_advect_{flavour}",
            ["mhd/dual_energy=true", "mhd/dual_energy_eta2=1.0e30"])[-1])

        # Entropy is not conserved across a shock, so an adiabat that is recomputed from
        # the energy channel every stage and one that is purely advected differ by the
        # entropy the tube's shocks generated -- a fraction of order one tenth here.  A
        # gate that is really unconditional makes the two runs bitwise identical.
        moved = (np.max(np.abs(resync["eint_aux"] - advect["eint_aux"]))
                 / np.max(np.abs(resync["eint_aux"])))
        if moved < 1.0e-6:
            pytest.fail(
                "dual_energy_eta2 changed nothing, so the resynchronisation gate is "
                f"unconditional or never runs: max relative difference {moved:g}"
            )

        # At the production eta1 no cell takes its pressure from the auxiliary, so a pass
        # that writes only the auxiliary column cannot move the gas.
        gas = (np.max(np.abs(resync["dens"] - advect["dens"]))
               / np.max(np.abs(resync["dens"])))
        if gas > 1.0e-13:
            pytest.fail(
                "dual_energy_eta2 moved the gas where the auxiliary channel never "
                f"fires: max relative difference {gas:g}"
            )
    finally:
        testutils.cleanup()
