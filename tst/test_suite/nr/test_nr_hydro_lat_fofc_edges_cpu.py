"""First-order flux correction under the LAT union predictor keeps the hydro mass.

A slower bin's corrector adds the pending fine flux estimate to its coarse/fine faces
before its first FOFC test, and that test also covers the ghost ring: a face two blocks
share is first order on both sides only if both flag its cells alike.  Along the edges
where a face with the estimate meets a face shared with a same-level block of the same
cadence, the owner tests a cell with the estimate and the neighbour, which cannot add
it, without.  Hydro has no exchange of FOFC flags, so a cell flagged on one side only
made the shared face first order on that side alone, and no reflux books that face.

`hydro_lat_fofc_edges` is a cold, large-amplitude shear wave on three static levels and
three LAT bins, periodic, so the total mass is an invariant to round-off.  FOFC fires
along those edges in the first window of the slowest bin, where the mass drifted by
-5.6e-7 before the edge cells took the flags both sides can form.
"""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

INPUT = "inputs/hydro_lat_fofc_edges.athinput"
# Round-off of a sum of 4.4e4 cell masses.
MAX_MASS_DRIFT = 1.0e-13


def _total_mass(path):
    data = bin_convert.read_binary(str(path))
    n1, n2, n3 = (data[f"nx{d}_out_mb"] for d in (1, 2, 3))
    total = 0.0
    for geometry, dens in zip(data["mb_geometry"], data["mb_data"]["dens"]):
        volume = np.prod([(geometry[2*d+1] - geometry[2*d])/n
                          for d, n in enumerate((n1, n2, n3))])
        total += volume*np.sum(np.asarray(dens, dtype=np.float64))
    return total


def test_union_corrector_first_order_faces_keep_the_mass(tmp_path):
    command = ["./athena", "-i", INPUT, "-d", str(tmp_path)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False, env=environment,
                            text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "union stage-1 predictor enabled" in output, "the deck must run the union"
    events = np.loadtxt(next(tmp_path.glob("*.log")), ndmin=2)
    assert events[:, -1].sum() > 0, "FOFC never fired"

    dumps = sorted((tmp_path / "bin").glob("*.hydro_u.*.bin"))
    assert len(dumps) == 2, "expected the t = 0 and the final dump"
    mass0 = _total_mass(dumps[0])
    drift = abs(_total_mass(dumps[-1]) - mass0)/mass0
    assert drift < MAX_MASS_DRIFT, f"total mass drifted by {drift:.3e}"
