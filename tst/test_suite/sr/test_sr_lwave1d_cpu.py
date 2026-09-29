"""
Linear wave convergence test for special-relativistic hydro/MHD in 1D.
Runs tests in both hydro and MHD using RK3 for different
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and wave types
# The entries enumerated at the end of this comment were raised on 2026-09-19 to 1.5x
# the error this compiler produces (two significant digits).  They are not a code
# regression: the 2026-09-15 baseline binary reproduces the same digits.  At 64 cells
# ppmx/wenoz have left their convergent regime here -- the 32-cell errors still match
# the old thresholds exactly -- so both the absolute error and the 32->64 ratio carry
# the compiler's round-off, and the ratio entries listed are raised for the same reason.
# Measured with nvc++ 25.7-0 (-tp znver4), Kokkos Serial, Athena_ENABLE_MPI=ON:
#   hydro rk3 ppmx wave 4: 2.00660e-11, ratio 0.0639484
#   mhd   rk2 plm  wave 5: 6.00064e-08
#   mhd   rk3 ppmx wave 0: 5.19201e-10
#   mhd   rk3 ppmx wave 5: 8.44753e-10
#   mhd   rk3 wenoz wave 5: 8.82341e-10, ratio 0.242401
#   mhd   rk3 wenoz wave 6: 4.40788e-10
maxerrors = {
    ("hydro", "rk2", "plm", "0"): (2.1e-08, 0.28),
    ("hydro", "rk3", "ppm4", "0"): (4.6e-09, 0.23),
    ("hydro", "rk3", "ppmx", "0"): (4.3e-11, 0.097),
    ("hydro", "rk3", "wenoz", "0"): (2.5e-11, 0.13),
    ("hydro", "rk2", "plm", "4"): (1.8e-08, 0.29),
    ("hydro", "rk3", "ppm4", "4"): (6.5e-09, 0.29),
    ("hydro", "rk3", "ppmx", "4"): (3.0e-11, 0.096),
    ("hydro", "rk3", "wenoz", "4"): (1.1e-11, 0.17),
    ("hydro", "rk2", "plm", "3"): (1.8e-07, 0.33),
    ("hydro", "rk3", "ppm4", "3"): (3.8e-08, 0.26),
    ("hydro", "rk3", "ppmx", "3"): (1.2e-10, 0.063),
    ("hydro", "rk3", "wenoz", "3"): (2.7e-11, 0.036),
    ("mhd", "rk2", "plm", "0"): (5.9e-08, 0.28),
    ("mhd", "rk3", "ppm4", "0"): (1.7e-08, 0.29),
    ("mhd", "rk3", "ppmx", "0"): (7.8e-10, 0.21),
    ("mhd", "rk3", "wenoz", "0"): (5.1e-10, 0.23),
    ("mhd", "rk2", "plm", "6"): (2.3e-08, 0.28),
    ("mhd", "rk3", "ppm4", "6"): (7.9e-09, 0.32),
    ("mhd", "rk3", "ppmx", "6"): (4.5e-10, 0.24),
    ("mhd", "rk3", "wenoz", "6"): (6.6e-10, 0.25),
    ("mhd", "rk2", "plm", "5"): (9.0e-08, 0.29),
    ("mhd", "rk3", "ppm4", "5"): (2.3e-08, 0.24),
    ("mhd", "rk3", "ppmx", "5"): (1.3e-09, 0.28),
    ("mhd", "rk3", "wenoz", "5"): (1.3e-09, 0.36),
    ("mhd", "rk2", "plm", "1"): (4.4e-08, 0.28),
    ("mhd", "rk3", "ppm4", "1"): (1.2e-08, 0.24),
    ("mhd", "rk3", "ppmx", "1"): (1.2e-09, 0.25),
    ("mhd", "rk3", "wenoz", "1"): (1.2e-09, 0.25),
    ("mhd", "rk2", "plm", "4"): (4.1e-08, 0.33),
    ("mhd", "rk3", "ppm4", "4"): (1.2e-08, 0.23),
    ("mhd", "rk3", "ppmx", "4"): (1.1e-10, 0.23),
    ("mhd", "rk3", "wenoz", "4"): (1.2e-10, 0.21),
    ("mhd", "rk2", "plm", "2"): (1.6e-08, 0.29),
    ("mhd", "rk3", "ppm4", "2"): (5.2e-09, 0.25),
    ("mhd", "rk3", "ppmx", "2"): (5.2e-11, 0.17),
    ("mhd", "rk3", "wenoz", "2"): (4.8e-11, 0.26),
    ("mhd", "rk2", "plm", "3"): (3.3e-08, 0.37),
    ("mhd", "rk3", "ppm4", "3"): (4.9e-09, 0.24),
    ("mhd", "rk3", "ppmx", "3"): (1.4e-11, 0.063),
    ("mhd", "rk3", "wenoz", "3"): (5.6e-12, 0.032),
}

_recon = ["plm", "ppm4", "ppmx", "wenoz"]
_wave = {}
_wave["mhd"] = ["0", "6", "5", "1", "4", "2", "3"]
_wave["hydro"] = ["0", "4", "3"]
_flux = {}
_flux["mhd"] = ["llf", "hlle"]
_flux["hydro"] = ["llf", "hlle", "hllc"]
_res = [32, 64]  # resolutions to test


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nghost=3",
        "mesh/nx1=" + repr(res),
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=16",
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "mesh_refinement/refinement=none",
        "time/cfl_number=0.4",
        "coord/special_rel=true",
        "coord/general_rel=false",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/along_x1=true",
        "problem/amp=1.0e-6",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("soe", ["hydro", "mhd"])  # system of eqns to test
def test_run(rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    iv = "rk2" if rv == "plm" else "rk3"
    for fv in _flux[soe]:
        # Ignore return arguments
        _, _ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"sr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
        )
