"""
An operator that takes energy out of the gas has to take it out of the auxiliary too.

The dual-energy auxiliary is a second, independently advected record of the same
thermodynamic state.  A sink that debits only the conserved energy is therefore undone in
every cell where the eta1 test prefers the auxiliary: the restored-hot gas radiates again
next step from the same unchanged adiabat, which is an energy source running at the local
cooling rate with positive feedback.  Four operators did exactly that -- M1's implicit
commit, and disk, ISM and relativistic cooling -- and none of them could be seen on an
adiabatic tube.

The cheapest configuration that shows it is a box in which cooling is the only thing
that happens.  A uniform, static, uniformly magnetized cell of gas at a fixed cylindrical
radius relaxes onto the Noble disk-cooling target, which here is a hundredth of its
initial internal energy; nothing moves, so the adiabat is advected unchanged and the only
thing that may reduce it is the debit.  Cooling changes p at fixed rho and fixed D, so
kappa and the internal energy scale by exactly the same factor, and the pressure the
auxiliary implies must still be the gas pressure at the end of the run.  With the debit
missing, the auxiliary is left at its seeded value and implies a hundred times the
pressure the gas has.

The resynchronisation pass is switched off for the duration (eta2 far above any specific
energy in the box) because it would otherwise repair the auxiliary from the energy channel
and hide the very thing being measured.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np
import glob

input_file = "inputs/dual_gr_cooling.athinput"
_gamma = 2.0  # deck
_initial_eint = 1.0  # p/(Gamma-1) from the deck's uniform state


def test_cooling_debits_the_auxiliary():
    """Disk cooling charges the adiabat at the same point it charges the gas."""
    try:
        testutils.run(input_file, ["job/basename=dual_cooling"])
        dumps = sorted(glob.glob("tab/dual_cooling.mhd_w.*.tab"))
        assert dumps, "dual_cooling produced no output"
        final = athena_read.tab(dumps[-1])

        # The fixed-metric GR backend publishes internal energy density in this slot.
        eint = final["eint"]
        rho = final["dens"]
        kappa = final["eint_aux"]

        # Guard against a vacuous pass: the test only discriminates if the gas really
        # did cool.  The deck's target is 0.01 against an initial 1.0, and the run is
        # several cooling times long.
        cooled = np.max(eint)/_initial_eint
        if cooled > 0.1:
            pytest.fail(
                "the gas did not cool, so the test measures nothing: internal energy "
                f"is still {cooled:g} of its initial value"
            )

        # kappa*rho^Gamma is the pressure the auxiliary carries; (Gamma-1)*eint is the
        # pressure the gas has.  They are the same number up to the difference between a
        # multiplicative debit on the auxiliary and an additive one on the gas, which is
        # second order in the fraction removed per stage -- a few percent here.  An
        # undebited auxiliary sits a factor of a hundred high.
        implied = kappa*rho**_gamma/(_gamma - 1.0)
        ratio = implied/eint
        if np.max(ratio) > 5.0:
            pytest.fail(
                "cooling did not debit the dual-energy auxiliary: the adiabat implies "
                f"{np.max(ratio):g} times the internal energy the gas has"
            )
        if np.min(ratio) < 0.2:
            pytest.fail(
                "cooling over-debited the dual-energy auxiliary: the adiabat implies "
                f"{np.min(ratio):g} times the internal energy the gas has"
            )
    finally:
        testutils.cleanup()
