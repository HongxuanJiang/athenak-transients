"""
Dual-energy formalism on the fixed-metric GR MHD path.

Two things can go wrong with an optional auxiliary channel and only one of them is
loud.  The loud one is that switching it on changes an answer it had no business
changing; the quiet one is that it silently becomes dead code and stops protecting
anything.  This test pins both ends: at the production threshold the auxiliary must
never fire on this mild magnetized tube and the solution must be unchanged, and with
the threshold forced to unity -- which selects the adiabat-derived pressure in every
cell -- the solution must visibly move.  A run that passes the first check and fails
the second is a feature that has stopped working.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np
import glob

input_file = "inputs/mub1_gr_dual.athinput"
_res = 400  # coarse enough to stay quick, fine enough to form the MUB1 structure


def arguments(tag, dual, eta1):
    """Assemble arguments for run command"""
    return [
        f"job/basename={tag}",
        "mesh/nx1=" + repr(_res),
        "meshblock/nx1=" + repr(_res),
        f"mhd/dual_energy={dual}",
        f"mhd/dual_energy_eta1={eta1}",
    ]


def density(tag, dual, eta1):
    """Run one case and return the density profile of its LAST output."""
    testutils.run(input_file, arguments(tag, dual, eta1))
    dumps = sorted(glob.glob(f"tab/{tag}.mhd_w.*.tab"))
    assert dumps, f"{tag} produced no output"
    return athena_read.tab(dumps[-1])["dens"]


def test_run():
    """Auxiliary channel is inert at the production threshold and live when forced."""
    try:
        off = density("dual_off", "false", 1.0e-4)
        on = density("dual_on", "true", 1.0e-4)
        forced = density("dual_forced", "true", 1.0)

        scale = np.max(np.abs(off))
        inert = np.max(np.abs(on - off))/scale
        live = np.max(np.abs(forced - off))/scale

        if inert > 1.0e-13:
            pytest.fail(
                "dual_energy changed the solution where it should never fire: "
                f"max relative difference {inert:g}"
            )
        if live < 1.0e-6:
            pytest.fail(
                "dual_energy with eta1=1 did not change the solution, so the "
                f"auxiliary channel is not being taken: max relative difference {live:g}"
            )
    finally:
        testutils.cleanup()
