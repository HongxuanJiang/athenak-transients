"""
Dual-energy formalism on the fixed-metric GR hydro path.

The GR hydro flavour carries the adiabat kappa = p/rho^Gamma as a passive scalar and
re-solves a cell from it wherever the eta1 ratio test rejects the energy channel.  As
for GR MHD, two failure modes are pinned: at the production threshold the auxiliary
must never fire on this tube and the solution must be unchanged, and with the
threshold forced to unity, which selects the adiabat-derived pressure in every cell,
the solution must visibly move.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np
import glob

input_file = "inputs/dual_grhd_tube.athinput"
_res = 400


def arguments(tag, dual, eta1):
    """Assemble arguments for run command"""
    return [
        f"job/basename={tag}",
        "mesh/nx1=" + repr(_res),
        "meshblock/nx1=" + repr(_res),
        f"hydro/dual_energy={dual}",
        f"hydro/dual_energy_eta1={eta1}",
    ]


def density(tag, dual, eta1):
    """Run one case and return the density profile of its LAST output."""
    testutils.run(input_file, arguments(tag, dual, eta1))
    dumps = sorted(glob.glob(f"tab/{tag}.hydro_w.*.tab"))
    assert dumps, f"{tag} produced no output"
    return athena_read.tab(dumps[-1])["dens"]


def test_run():
    """Auxiliary channel is inert at the production threshold and live when forced."""
    try:
        off = density("grhd_dual_off", "false", 1.0e-4)
        on = density("grhd_dual_on", "true", 1.0e-4)
        forced = density("grhd_dual_forced", "true", 1.0)

        scale = np.max(np.abs(off))
        inert = np.max(np.abs(on - off))/scale
        live = np.max(np.abs(forced - off))/scale

        if inert > 1.0e-13:
            pytest.fail(
                "hydro dual_energy changed the solution where it should never fire: "
                f"max relative difference {inert:g}"
            )
        if live < 1.0e-6:
            pytest.fail(
                "hydro dual_energy with eta1=1 did not change the solution, so the "
                f"auxiliary channel is not being taken: max relative difference {live:g}"
            )
        if not np.all(np.isfinite(forced)):
            pytest.fail("forced auxiliary channel produced a non-finite density")
    finally:
        testutils.cleanup()
