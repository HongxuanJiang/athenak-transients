"""A star's atmosphere, which nothing heats, may not heat.

With a tabulated EOS the primitive carried through reconstruction is the internal energy
DENSITY, and every limiter (PLM, PPM, WENOZ) bounds it independently of the mass
density.  Both face states are then monotone in their own variable, but their ratio is
not: across the surface of a star the reconstructed density falls into ``dfloor`` while
the reconstructed energy density keeps a value set by the last dense cell, so the face
state handed to the Riemann solver has a specific internal energy of ``e_face/dfloor``.
The solver is conservative, so the energy it deposits is small; it lands on almost no
mass, and the atmosphere heats by orders of magnitude.  At a coarse/fine face the cliff
is steepest and the artefact strongest, which is where it has been seen in production.

The deck is a self-gravitating star whose surface density cliff lies on a coarse/fine
face.  The star is built in hydrostatic equilibrium with the same tabulated EOS that
evolves it, so over the few dynamical times of this test nothing does work on the gas:
there is no heating term, no shock, and the atmosphere is bound.  The MHD deck threads
the star with a uniform field too weak to move it, so the magnetic energy rides through
the face states and the HLLD fan without changing the physical answer.

The gate is therefore on the atmosphere itself, on every Newtonian module and
reconstruction that carries a tabulated EOS: no gas below 1e-8 at the start and the end
may finish hotter per unit mass than twice the hottest atmosphere gas at the start.  The
stellar centre is no reference: the atmosphere starts at a quarter of its specific
energy, so a gate there leaves room to heat eightfold unseen.  Without a heating term a
doubling would need adiabatic compression by nearly threefold in density (Gamma <= 5/3),
or mixing with gas twice as hot, and the stellar surface layers the atmosphere can reach
in a few tens of steps are colder than it is.

The low-beta arms raise the field a hundredfold, so the atmosphere is magnetically
dominated (beta ~ 1e-3) and its internal energy is a part in a thousand of the total.
There the energy channel E - KE - B^2/2 carries a truncation error of the kinetic-magnetic
exchange several times the thermal energy, and only the dual-energy auxiliary knows the
temperature.  Both arms run the decks' own dual-energy parameters (eta1 = 1e-3,
eta2 = 1e-4), so they gate three things at once: the inversion must reject that energy
channel although it is well above eta1 of the total (it heated these atmospheres
20-25-fold), the eta2 resynchronisation must not hand the rejected energy back to the
auxiliary, and the rejected channel must be reset rather than kept.  Kept, it integrates
the error of every stage until it passes any threshold and is released as heat at once;
the uniform-grid arm runs 60 steps because that took the atmosphere to 160-fold heating
by step 60 with the reset removed, while 10 steps did not show it.
"""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

TABLE = REPOSITORY_ROOT / "eos_tables/chabrier2021_t13_helm_union_prad_640.table"
# Gas below this density at the start and at the end is the atmosphere.  The densest
# stellar gas is 11; the floor is 1e-10.
ATMOSPHERE_DENSITY = 1.0e-8
# No atmosphere cell may end hotter per unit mass than twice the hottest atmosphere gas
# at the start (see the module docstring for why a factor two is heating).
MAX_EPS_OVER_INITIAL_ATMOSPHERE = 2.0

# (id, module, reconstruction, deck, overrides).  PLM runs with two ghost cells; the
# wide stencils need four.
CASES = [("hydro-plm", "hydro", "plm", "hydro_star_surface_eps", []),
         ("mhd-plm", "mhd", "plm", "mhd_star_surface_eps", []),
         ("hydro-ppm4", "hydro", "ppm4", "hydro_star_surface_eps_ho", []),
         ("mhd-ppm4", "mhd", "ppm4", "mhd_star_surface_eps_ho", []),
         ("hydro-ppmx", "hydro", "ppmx", "hydro_star_surface_eps_ho", []),
         ("mhd-ppmx", "mhd", "ppmx", "mhd_star_surface_eps_ho", []),
         ("hydro-wenoz", "hydro", "wenoz", "hydro_star_surface_eps_ho", []),
         ("mhd-wenoz", "mhd", "wenoz", "mhd_star_surface_eps_ho", []),
         ("mhd-lowbeta-plm", "mhd", "plm", "mhd_star_lowbeta_hlld", []),
         ("mhd-lowbeta-uniform-plm", "mhd", "plm", "mhd_star_lowbeta_hlld_uniform",
          ["time/nlim=60"])]


def _run(run_directory, module, reconstruction, deck, overrides):
    command = ["./athena", "-i", f"inputs/{deck}.athinput",
               f"{module}/table={TABLE}", f"{module}/reconstruct={reconstruction}",
               *overrides, "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=600)
    assert result.returncode == 0, (
        f"star surface run failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )


def _density_and_specific_energy(dump):
    data = bin_convert.read_binary(str(dump))
    density = np.asarray(data["mb_data"]["dens"])
    return density, np.asarray(data["mb_data"]["eint"])/density


@pytest.mark.parametrize("module,reconstruction,deck,overrides",
                         [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_star_atmosphere_is_not_heated(tmp_path, module, reconstruction, deck,
                                       overrides):
    """No atmosphere cell may end hotter per unit mass than the atmosphere started."""

    run = tmp_path / "star"
    _run(run, module, reconstruction, deck, overrides)

    dumps = sorted((run / "bin").glob(f"*.{module}_w.*.bin"))
    assert len(dumps) >= 2, "the run wrote no final dump to test"

    initial_density, initial_eps = _density_and_specific_energy(dumps[0])
    density, eps = _density_and_specific_energy(dumps[-1])
    atmosphere = (initial_density < ATMOSPHERE_DENSITY) & (density < ATMOSPHERE_DENSITY)
    assert atmosphere.any(), "the deck has no atmosphere to test"
    assert np.isfinite(eps[atmosphere]).all(), "the atmosphere holds a non-finite state"

    reference = float(initial_eps[initial_density < ATMOSPHERE_DENSITY].max())
    peak = float(eps[atmosphere].max())
    assert peak <= MAX_EPS_OVER_INITIAL_ATMOSPHERE*reference, (
        f"gas in the star's atmosphere reached a specific internal energy of {peak:.4e}, "
        f"{peak/reference:.1f} times the hottest atmosphere gas at the start "
        f"({reference:.4e}), with no heating term in the problem: on the {module} path "
        f"with {reconstruction} ({deck} {' '.join(overrides)}) either the face states "
        f"across the surface density cliff are not bounded in specific internal energy "
        f"or the inversion took (or kept, or copied into the auxiliary) an energy channel "
        f"that is truncation error of the magnetic energy"
    )
