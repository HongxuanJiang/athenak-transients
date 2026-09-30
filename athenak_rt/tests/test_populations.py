"""EOS-table H/He populations (``populations = eos``) versus the Saha solver."""

import math
from pathlib import Path

import numpy as np
import pytest

from athenak_rt.config import DEFAULT_POPULATIONS, RTSettings
from athenak_rt.params import parse_parameters
from athenak_rt.eos import TabulatedLteTable
from athenak_rt.opacity import (
    DEFAULT_MESA_HIGH_T,
    DEFAULT_MESA_LOW_T,
    MesaOpacityModel,
    frequency_quadrature_weights,
    mesa_alpha_scalar,
    photon_energy_grid,
    planck_nu,
    saha_state,
    table_state,
)
from athenak_rt.transfer import DIRECTION_Z, integrate_multifreq_rays

TABLE = Path.home() / "athenak-tde-release" / "chabrier2021_t13_helm_union_prad_640.table"
needs_table = pytest.mark.skipif(not TABLE.is_file(), reason="real EOS table not present")


@pytest.fixture(scope="module")
def eos():
    # The units only matter for the temperature inversion, not for the populations.
    return TabulatedLteTable(TABLE, 1.0, 1.0, 1.0)


@needs_table
def test_table_state_reproduces_nodes(eos):
    lr0, dlr, lt0, dlt, lnf, nh, nhe = eos.population_table()
    assert lnf.flags["C_CONTIGUOUS"] and lnf.dtype == np.float64
    xion, xh2 = eos.fields["xion"], eos.fields["xh2"]
    xhe1, xhe2 = eos.fields["xhe1"], eos.fields["xhe2"]
    rng = np.random.default_rng(1)
    for _ in range(40):
        i = int(rng.integers(1, eos.nrho - 1))
        j = int(rng.integers(1, eos.ntemp - 1))
        rho, temp = math.exp(eos.logrho[i]), math.exp(eos.logtemp[j])
        ne, h1, h2, he1, he2, he3 = table_state(rho, temp, *eos.population_table())
        n_h, n_he = nh * rho, nhe * rho
        for got, ref, n in (
            (h1, 1.0 - xion[i, j] - xh2[i, j], n_h),
            (h2, xion[i, j], n_h),
            (he1, 1.0 - xhe1[i, j] - xhe2[i, j], n_he),
            (he2, xhe1[i, j], n_he),
            (he3, xhe2[i, j], n_he),
        ):
            ref = min(max(ref, 1e-300), 1.0)
            assert got == pytest.approx(n * ref, rel=1e-6)  # exp/log round trip of the node
        assert ne == pytest.approx(h2 + he2 + 2.0 * he3, rel=1e-12)


@needs_table
@pytest.mark.parametrize("rho,temp", [(1e-10, 3e6), (1e-8, 1e7), (1e-12, 1e7)])
def test_fully_ionized_matches_saha(eos, rho, temp):
    tab = table_state(rho, temp, *eos.population_table())
    ref = saha_state(rho, temp)
    assert tab[0] == pytest.approx(ref[0], rel=1e-2)
    assert tab[2] == pytest.approx(ref[2], rel=1e-2)
    assert tab[5] == pytest.approx(ref[5], rel=1e-2)


def test_missing_fields_raise_clear_error(tmp_path):
    from test_pipeline import write_eos_table

    path = tmp_path / "h_only.table"
    write_eos_table(path)
    with pytest.raises(RuntimeError, match="no xh2"):
        TabulatedLteTable(path, 1.0, 1.0, 1.0).population_table()


@needs_table
@pytest.mark.parametrize("use_table", [False, True])
def test_grey_limit_identity_with_both_populations(eos, use_table):
    rng = np.random.default_rng(0)
    shape = (12, 3, 4)
    rho = 10.0 ** rng.uniform(-11, -7, size=shape)
    temp = 10.0 ** rng.uniform(3.8, 5.5, size=shape)
    rho[rng.random(shape) < 0.15] = 0.0
    args = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T).kernel_args
    _, nus = photon_energy_grid(30, 0.1, 1000.0)
    weights = frequency_quadrature_weights(nus)
    ds = 3.0e10
    pop = (True, *eos.population_table()) if use_table else (False,)
    spec, intensity = integrate_multifreq_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, nus, False, True, 1e9, *pop
    )
    nz, ny, nx = shape
    ref_int = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            tau, inten = 0.0, np.zeros(nus.size)
            for k in range(nz - 1, -1, -1):
                r, t = rho[k, j, i], temp[k, j, i]
                if not (r > 0.0 and t > 0.0):
                    continue
                dtau = mesa_alpha_scalar(r, t, *args) * ds
                b = np.array([planck_nu(nu, t) for nu in nus])
                inten += b * math.exp(-tau) * (-math.expm1(-dtau))
                tau += dtau
            ref_int[j, i] = float(np.sum(inten * weights))
    assert np.allclose(intensity, ref_int, rtol=1e-12, atol=0.0)


def test_default_populations_is_eos():
    assert DEFAULT_POPULATIONS == "eos"
    assert RTSettings().populations == "eos"
    minimal = "<input>\ndump_dir = .\ndumps = 1\n"
    assert parse_parameters(minimal).settings.populations == "eos"
    saha = minimal + "<transfer>\npopulations = saha\n"
    assert parse_parameters(saha).settings.populations == "saha"
    assert RTSettings().as_attributes()["populations"] == "eos"
