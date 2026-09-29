"""The dynamical-GR density floor only adds, and the tabulated-EOS diagnostics are live.

Floors (`dyngr_floor_drift`: a uniform periodic SR dyn-GR box, T = P/n = 100 in units of
the rest mass, v = 0.99 along a weak B, dfloor = 1e-10).  A floor that raises n must
add rest mass and heat, never remove enthalpy: the drift-frame re-injection that follows
conserves the parallel momentum w W^2 v, so less w accelerates the gas.

* pgen state at n = dfloor/2 or n = 0 with P > 0 (PrimToCons).  Its n and P are set
  independently, so T = P/n is no gas property: the floor raises n and keeps P, and the
  velocity follows from w W^2 v at the raised w (d8707ab2).  Before f24fc9c2 the reset to
  T_atm removed the heat: the n = dfloor/2 cell came out at n = 3.95e-10, not dfloor,
  with u = 19.97 at the Lorentz ceiling (against 7.02), and the n = 0 cell as the
  atmosphere (P = 1e-18); between the two, n was raised at T = P/n, which doubled P (and
  gave P = inf at n = 0, rewritten to the atmosphere).
* inverted state (a checkpoint of the n = dfloor/2 gas restarted under a floor of
  1e-10): n -> dfloor at T >= T_old; the old reset published n = 3.77e-10 and
  accelerated it to u = 10.9 (f24fc9c2).

Diagnostics (`hydro_lte_thermo_tube`: a Chabrier/HELM + radiation LTE table in the TDE
units, an ionised 5e4 K state against a half-ionised 8e3 K one).  hydro_{temperature,
mu, beta_rad, xion, gamma1} were accepted by the output constructor but never computed,
so every dump was zero (032fde99).  At t = 0 the pressure of each side is the deck's, so
the dumped T, mu and beta_rad must satisfy a T^4/3 = beta_rad P and
rho k T/(mu m_H) = (1 - beta_rad) P to the table's interpolation (measured 7e-4).
"""

# Modules
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

FLOOR = "inputs/dyngr_floor_drift.athinput"
THERMO = "inputs/hydro_lte_thermo_tube.athinput"
EOS_TABLE = REPOSITORY_ROOT / "eos_tables" / "chabrier2021_t13_helm_union_prad_640.table"

# dyngr_floor_drift: <mhd>/gamma, dfloor, and the pgen state (pl, ul) of both sides.
GAMMA = 5.0/3.0
N_ATM = 1.0e-10
P0 = 5.0e-9
V0 = 0.99
ROUNDOFF = 1.0e-12

# hydro_lte_thermo_tube: the <units> block and the pgen pressures.
MASS_CGS = 1.98841586e33
LENGTH_CGS = 1.3914e11
TIME_CGS = 1.2709152632111343e3
PRESSURES = (1.0e-11, 1.0e-14)  # x < 0, x > 0
A_RAD = 7.5657e-15
K_B = 1.380649e-16
M_H = 1.6735575e-24  # the tabulated EOS's temperature unit is v^2 m_H/k_B
MAX_IDENTITY_ERROR = 5.0e-3


def run(directory, *arguments):
    command = ["mpirun", "-np", "1", "--bind-to", "none", "./athena",
               "-d", str(directory), *arguments]
    result = subprocess.run(command, capture_output=True, check=False, text=True,
                            timeout=600)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{directory.name}: exit {result.returncode}\n" + \
        output[-4000:]


def cell_values(path):
    """The (uniform) primitives of a single-block 1D dump, checked to be uniform."""
    data = bin_convert.read_binary(str(path))
    values = {}
    for name in ("dens", "velx", "press"):
        field = np.asarray(data["mb_data"][name]).ravel()
        assert np.all(np.isfinite(field)), f"{path.name}: {name} is not finite"
        assert np.all(field == field[0]), f"{path.name}: {name} is not uniform"
        values[name] = field[0]
    return data["cycle"], values


def close(a, b):
    return abs(a/b - 1.0) < ROUNDOFF


def momentum(n, p, u):
    """Parallel momentum density w W^2 v = w W u of a gas along its (weak) field."""
    return (n + GAMMA/(GAMMA - 1.0)*p)*np.sqrt(1.0 + u*u)*u


@pytest.mark.parametrize("n0", [0.5*N_ATM, 0.0])
def test_pgen_state_below_the_floor_keeps_its_pressure(tmp_path, n0):
    directory = tmp_path / "pgen"
    run(directory, "-i", FLOOR, f"problem/dl={n0}", f"problem/dr={n0}")
    u0 = V0/np.sqrt(1.0 - V0*V0)
    dumps = sorted((directory / "bin").glob("*.mhd_w.*.bin"))
    assert len(dumps) == 2
    for dump in dumps:
        cycle, cell = cell_values(dump)
        assert close(cell["dens"], N_ATM), f"cycle {cycle}: n = {cell['dens']:.12e}"
        assert close(cell["press"], P0), \
            f"cycle {cycle}: P = {cell['press']:.12e}, not the pgen's {P0:.12e}"
        assert cell["velx"] <= u0*(1.0 + ROUNDOFF), \
            f"cycle {cycle}: the floor accelerated u {u0:.10e} -> {cell['velx']:.10e}"
        if n0 > 0.0:
            # w W u is conserved at the raised w: u sqrt(1 + u^2) = q.
            q = momentum(n0, P0, u0)/(N_ATM + GAMMA/(GAMMA - 1.0)*P0)
            u = np.sqrt(0.5*(np.sqrt(1.0 + 4.0*q*q) - 1.0))
            assert close(cell["velx"], u), \
                f"cycle {cycle}: u = {cell['velx']:.12e}, drift-frame answer {u:.12e}"


def test_inverted_state_below_the_floor_is_raised_not_accelerated(tmp_path):
    seed = tmp_path / "seed"
    run(seed, "-i", FLOOR, "mhd/dfloor=1.0e-11", "time/nlim=0")
    _, before = cell_values(sorted((seed / "bin").glob("*.mhd_w.*.bin"))[0])
    assert close(before["dens"], 0.5*N_ATM) and close(before["press"], P0)

    raised = tmp_path / "raised"
    run(raised, "-r", str(seed / "rst" / "dyngr_floor_drift.00000.rst"),
        f"mhd/dfloor={N_ATM}", "time/nlim=1")
    cycle, after = cell_values(sorted((raised / "bin").glob("*.mhd_w.*.bin"))[-1])
    assert cycle == 1
    assert close(after["dens"], N_ATM), f"n = {after['dens']:.12e}"
    assert after["press"] >= before["press"], \
        f"the floor removed heat: P {before['press']:.6e} -> {after['press']:.6e}"
    assert after["velx"] <= before["velx"], \
        f"the floor accelerated u from {before['velx']:.10e} to {after['velx']:.10e}"
    drift = momentum(after["dens"], after["press"], after["velx"]) / \
        momentum(before["dens"], before["press"], before["velx"]) - 1.0
    assert abs(drift) < ROUNDOFF, f"parallel momentum moved by {drift:.3e}"


def test_tabulated_eos_thermo_diagnostics_are_the_table_state(tmp_path):
    directory = tmp_path / "thermo"
    run(directory, "-i", THERMO, f"hydro/table={EOS_TABLE}")
    velocity = LENGTH_CGS/TIME_CGS
    density_unit = MASS_CGS/LENGTH_CGS**3
    pressure_unit = MASS_CGS/(LENGTH_CGS*TIME_CGS**2)
    temperature_unit = velocity*velocity*M_H/K_B

    fields = {}
    for name in ("w", "temperature", "mu", "beta_rad", "xion", "gamma1"):
        dumps = sorted((directory / "bin").glob(f"*.hydro_{name}.*.bin"))
        assert len(dumps) == 2, f"hydro_{name}: expected the t = 0 and the final dump"
        for index, dump in enumerate(dumps):
            data = bin_convert.read_binary(str(dump))
            for variable in data["var_names"]:
                fields[(index, variable)] = np.asarray(data["mb_data"][variable]).ravel()
            geometry, n1 = data["mb_geometry"][0], data["nx1_out_mb"]
    x = geometry[0] + (np.arange(n1) + 0.5)*(geometry[1] - geometry[0])/n1

    for index in (0, 1):
        temperature = fields[(index, "temperature")]
        assert np.all(np.isfinite(temperature) & (temperature > 0.0)), \
            f"dump {index}: temperature not positive"
        assert np.all((fields[(index, "mu")] > 0.5) & (fields[(index, "mu")] < 2.5))
        assert np.all((fields[(index, "xion")] >= 0.0) & (fields[(index, "xion")] <= 1.0))
        assert np.all((fields[(index, "gamma1")] > 1.0) &
                      (fields[(index, "gamma1")] <= GAMMA + ROUNDOFF))
        assert np.all((fields[(index, "beta_rad")] > 0.0) &
                      (fields[(index, "beta_rad")] < 1.0))

    pressure = np.where(x < 0.0, *PRESSURES)*pressure_unit
    temperature = fields[(0, "temperature")]*temperature_unit
    density = fields[(0, "dens")]*density_unit
    beta = fields[(0, "beta_rad")]
    radiation = A_RAD*temperature**4/3.0/(beta*pressure) - 1.0
    gas = density*K_B*temperature/(fields[(0, "mu")]*M_H)/((1.0 - beta)*pressure) - 1.0
    assert np.max(np.abs(radiation)) < MAX_IDENTITY_ERROR, \
        f"a T^4/3 = beta_rad P off by {np.max(np.abs(radiation)):.3e}"
    assert np.max(np.abs(gas)) < MAX_IDENTITY_ERROR, \
        f"rho k T/(mu m_H) = (1 - beta_rad) P off by {np.max(np.abs(gas)):.3e}"
    xion = fields[(0, "xion")]
    assert xion[0] > 0.99 and 0.1 < xion[-1] < 0.9, "the two sides are not the ionised " \
        "and the partly ionised state"
