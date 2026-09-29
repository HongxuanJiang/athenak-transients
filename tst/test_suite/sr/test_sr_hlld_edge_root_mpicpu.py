"""The relativistic HLLD fan takes faces whose root lies at the edge of its window.

`srmhd_hlld_face`: one special-relativistic step (donor cell, forward Euler, CFL 0.1) of
an eight-cell tube whose two halves are the left and right states of one face.  Every
other face joins two equal states, so HLLD and HLLE give them the same flux bit for bit,
and the two interface cells differ between `rsolver = hlld` and `rsolver = hlle` exactly
by the difference of the two fluxes at that face.

The four pairs are magnetised states (sigma 4e2 to 8e3, W < 1.02) one part in 1e3 apart,
drawn like the missed-root census of the fan, whose HLLD root lies just inside the edge
of the admissible pressure window.  The walk of BracketedPressure (the root solve shared
by HLLD_SR, HLLD_GR and HLLD_DYNGR) stopped a side at its first admissible probe, past
that root, with every residual of one sign, and the fan declined to HLLE: the interface
cells then matched the HLLE step to 1e-11.  The probe and the inadmissible one before it
are now bisected until the other sign appears (f4765312): the HLLD step leaves HLLE's by
1e-4 to 3e-3.
"""

# Modules
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

DECK = "inputs/srmhd_hlld_face.athinput"
# (d, p, v_x, v_y, v_z, B_y, B_z) left and right, and the shared B_x.
PAIRS = [
    ((3.915559446270657e-05, 6.738636543496513e-10, -0.03580799613297099,
      0.03366474532512007, -0.013428240727901329, 0.030849270960870732,
      -0.13093862127466024),
     (3.914711633856899e-05, 6.7380685872123e-10, -0.036595400998425995,
      0.03274048025061377, -0.013087921216738603, 0.03076542126997019,
      -0.13081385754325228), -0.02415761641355535),
    ((0.0026381308188984675, 1.1301594962010668e-06, 0.02867376698084021,
      -0.0011831217379244573, -0.009506903669959866, -3.88497291932649,
      -1.7407196117400692),
     (0.0026389295173078193, 1.1300019306803483e-06, 0.02775877929449347,
      -0.0013158646762597352, -0.010009769996611032, -3.8810298299582633,
      -1.738690446360901), 1.3342678799650616),
    ((0.015562818454399304, 2.7293176572013342e-05, 0.03307011845331272,
      0.02262071714962815, 0.019422620983125017, 1.7770208706475006,
      -2.5001604588395847),
     (0.015561953268690348, 2.726778153577253e-05, 0.032181495995948814,
      0.02172484885067205, 0.01852908223724704, 1.778380778527695,
      -2.502924117691878), 0.16537587673417722),
    ((0.25504933124090756, 0.00011567163602415471, 0.051342330566685204,
      0.039877312018708866, -0.13761666630744224, 4.848385547974168,
      8.221386260455951),
     (0.25508266535907487, 0.00011563636582159588, 0.05042930790402987,
      0.03925772984134969, -0.1382024438418445, 4.851035313863795,
      8.230690174783922), -0.9992654821468783),
]
KEYS = ("d", "p", "u", "v", "w", "by", "bz")
# Measured 1e-11 to 1.3e-11 where the fan declined, 8.8e-5 to 2.6e-3 where it solves.
MIN_HLLD_DEPARTURE = 1.0e-6


def one_step(directory, rsolver, left, right, bx):
    overrides = [f"mhd/rsolver={rsolver}"]
    for side, state in (("l", left), ("r", right)):
        overrides += [f"problem/{key}{side}={value!r}" for key, value in
                      zip(KEYS, state)] + [f"problem/bx{side}={bx!r}"]
    command = ["mpirun", "-np", "1", "--bind-to", "none", "./athena", "-i", DECK,
               "-d", str(directory), *overrides]
    result = subprocess.run(command, capture_output=True, check=False, text=True,
                            timeout=300)
    assert result.returncode == 0, \
        f"{directory.name}: exit {result.returncode}\n" + (result.stdout + result.stderr)
    dump = sorted((directory / "bin").glob("*.mhd_u.*.bin"))[-1]
    data = bin_convert.read_binary(str(dump))
    assert data["cycle"] == 1
    return np.array([np.ravel(data["mb_data"][name]) for name in data["var_names"]])


@pytest.mark.parametrize("index", range(len(PAIRS)))
def test_hlld_resolves_a_root_at_the_window_edge(tmp_path, index):
    left, right, bx = PAIRS[index]
    hlld = one_step(tmp_path / "hlld", "hlld", left, right, bx)
    hlle = one_step(tmp_path / "hlle", "hlle", left, right, bx)
    scale = np.max(np.abs(hlld), axis=1, keepdims=True)
    difference = np.abs(hlld - hlle)/scale
    # Cells 0-2 and 5-7 see only faces between equal states.
    assert np.all(np.delete(difference, [3, 4], axis=1) == 0.0)
    departure = np.max(difference[:, 3:5])
    assert departure > MIN_HLLD_DEPARTURE, (
        f"the interface step is HLLE's to {departure:.2e}: the fan declined the face")
