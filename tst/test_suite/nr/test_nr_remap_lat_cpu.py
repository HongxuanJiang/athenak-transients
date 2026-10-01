"""A remap target run with LAT on follows the same run with LAT off.

Remap and LAT can share a run.  The startup remap runs before any LAT window exists, so
nothing built for the old state is left behind, and the driver derives the LAT bins from
the remapped state at its first time step.  The deck remaps a smooth analytic state
(``remap_test``) from a uniform source mesh onto a periodic SMR target with two levels,
so the LAT run really uses windows of two ticks.

Two things are checked against the same deck run with ``time/lat=false``.  The total mass
is a conserved quantity of the periodic target, so a LAT window that dropped or doubled a
coarse/fine flux shows up at round-off.  The density field has to agree to the truncation
error of the scheme, since the two runs use different time steps on the coarse blocks.
"""

from pathlib import Path
import os
import re
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

SOURCE_INPUT = "inputs/remap_lat_source.athinput"
TARGET_INPUT = "inputs/remap_lat.athinput"
# Round-off of a sum of 3.3e4 cell masses.
MAX_MASS_DRIFT = 1.0e-12
# Truncation difference of LAT and synchronous stepping over about fifteen cycles.
MAX_DENSITY_DIFFERENCE = 1.0e-3


def _run(command, tmp_path):
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False, env=environment,
                            text=True, timeout=600, cwd=None)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    return output


def _final_dump(directory):
    dumps = sorted((directory / "bin").glob("*.hydro_u.*.bin"))
    assert len(dumps) == 2, "expected the dump after the remap and the final dump"
    return dumps[-1]


def _blocks(path):
    data = bin_convert.read_binary(str(path))
    n = [int(data[f"nx{d}_out_mb"]) for d in (1, 2, 3)]
    blocks = {}
    for location, geometry, dens in zip(data["mb_logical"], data["mb_geometry"],
                                        data["mb_data"]["dens"]):
        volume = np.prod([(geometry[2*d+1] - geometry[2*d])/n[d] for d in range(3)])
        blocks[tuple(int(v) for v in location)] = (
            volume, np.asarray(dens, dtype=np.float64))
    return data["time"], blocks


def test_remap_target_with_lat_follows_the_synchronous_run(tmp_path):
    source_dir = tmp_path / "source"
    _run(["./athena", "-i", SOURCE_INPUT, "-d", str(source_dir)], tmp_path)
    restarts = sorted((source_dir / "rst").glob("*.rst"))
    assert restarts, "the source run wrote no restart file"
    source = str(restarts[-1].resolve())

    outputs = {}
    results = {}
    for name, lat in (("sync", "false"), ("lat", "true")):
        directory = tmp_path / name
        outputs[name] = _run(["./athena", "-i", TARGET_INPUT, "-d", str(directory),
                              f"time/lat={lat}", f"remap/source={source}"], tmp_path)
        results[name] = _blocks(_final_dump(directory))

    assert "--- Remap (<remap> block) ---" in outputs["lat"]
    # The LAT run must really use windows: a block on level 0 steps at factor 2.
    bins = re.findall(r"HD LAT final bins f1=(\d+) f2=(\d+)", outputs["lat"])
    assert bins and all(int(f2) > 0 for _, f2 in bins), "LAT never formed a factor-2 bin"

    (time_sync, sync), (time_lat, lat) = results["sync"], results["lat"]
    assert abs(time_sync - time_lat) < 1.0e-12, "both runs must stop at tlim"
    assert set(sync) == set(lat), "the two runs must end on the same mesh"

    def mass(blocks):
        return sum(volume*dens.sum() for volume, dens in blocks.values())

    mass_sync, mass_lat = mass(sync), mass(lat)
    assert abs(mass_lat - mass_sync)/mass_sync < MAX_MASS_DRIFT, (mass_sync, mass_lat)

    difference = sum(np.abs(sync[key][1] - lat[key][1]).sum() for key in sync)
    scale = sum(np.abs(sync[key][1]).sum() for key in sync)
    assert difference/scale < MAX_DENSITY_DIFFERENCE, difference/scale
