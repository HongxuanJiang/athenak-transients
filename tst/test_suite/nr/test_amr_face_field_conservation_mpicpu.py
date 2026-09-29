"""An adaptive MHD regrid must leave mass and energy conserved to round-off.

`mhd_amr_ot2d` is Orszag-Tang on a periodic 64^2 root grid with three adaptive levels
(slope criterion on density, ncycle_check = refinement_interval = 4), HLLD, two ranks.
The domain is closed, so the total mass and energy of the history can move only by the
rounding of the sums -- as long as every face of the mesh carries one face-centred
field, so that both blocks beside it compute one flux there.

* refinement only (derefine_value_max < 0): a child prolongs its boundary faces from
  its parent; across a face shared with a same-level block that already holds finer
  data nothing copied that block's face over the prolonged one, so the face stayed
  double-valued (a54e3baa).  Mass drifted by 1.9e-8 by t = 0.1.
* refinement and derefinement: RepairAMRFC re-interpolated the internal faces of a
  derefined parent after its neighbours had received the exact restriction as ghosts,
  so the first step after every derefining regrid broke conservation once (9017ae2b).
  With a54e3baa alone mass still drifted by 2.5e-8 and energy by 1.6e-7.
"""

# Modules
import re
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import athena_read  # noqa: E402

INPUT = "inputs/mhd_amr_ot2d.athinput"
MAX_DRIFT = 1.0e-12
ARMS = {"refine_only": ["amr_criterion0/derefine_value_max=-1.0"], "derefine": []}


@pytest.mark.parametrize("arm", list(ARMS))
def test_amr_regrid_conserves_mass_and_energy(tmp_path, arm):
    directory = tmp_path / arm
    command = ["mpirun", "-np", "2", "--bind-to", "none", "./athena", "-i", INPUT,
               "-d", str(directory), *ARMS[arm]]
    result = subprocess.run(command, capture_output=True, check=False, text=True,
                            timeout=600)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"exit {result.returncode}\n" + output[-4000:]

    created = sum(int(n) for n in re.findall(r"(\d+) MeshBlocks created", output))
    deleted = sum(int(n) for n in re.findall(r"(\d+) deleted by AMR", output))
    assert created > 0, "the deck no longer refines"
    if arm == "refine_only":
        assert deleted == 0, f"{deleted} MeshBlocks derefined in the refine-only arm"
    else:
        assert deleted > 0, "the deck no longer derefines"

    history = athena_read.hst(str(directory / "mhd_amr_ot2d.mhd.hst"))
    assert abs(history["time"][-1] - 0.1) < 1.0e-12
    for name in ("mass", "tot-E"):
        drift = np.max(np.abs(history[name]/history[name][0] - 1.0))
        assert drift < MAX_DRIFT, f"{arm}: total {name} drifted by {drift:.3e}"
