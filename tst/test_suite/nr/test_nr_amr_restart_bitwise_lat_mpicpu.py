"""A LAT run on an adaptive mesh, restarted from a checkpoint, is the uninterrupted run.

The sibling test test_nr_amr_restart_bitwise_mpicpu.py covers the cadence that
``ncycle_check = 1`` produces, where every cycle is an AMR call and the writer's cadence
is therefore whatever ncycle says it is.  This one covers the cadence that breaks that
inference.  Under LAT, Driver::Execute calls AMR at a WINDOW END whenever
``ncycle - last_amr_call_cycle >= ncycle_check``, and a window is a sum of sync factors
that tlim, gravity, sinks and nlim all shorten, so with ``ncycle_check = 3`` the writer's
calls land on cycles that are not multiples of 3.  Reconstructing the cadence from
``ncycle % ncycle_check`` then both skips the owed pass at such a cycle and starts the
resumed run's cadence from the wrong anchor, so every later AMR event moves and the two
runs separate.  The restart writer records the anchor instead
(``mesh_refinement/last_amr_call_cycle``, written like ``<outputN>/file_number``), and
Driver::Initialize evaluates the writer's own predicate with it.

"""

# Modules
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

input_file = "inputs/linear_wave_amr_restart_lat.athinput"
basename = "LinWaveAMRRestartLAT"
ncycle_check = 3


def run(command, tag):
    """Run the binary on two ranks; fail with the log if it does not exit cleanly."""
    result = subprocess.run(["mpirun", "-np", "2", "./athena"] + command,
                            capture_output=True, check=False, text=True, timeout=900)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{tag} failed with return code {result.returncode}:\n{output[-6000:]}"
    )
    return output


def amr_cycles(output):
    """The cycles on which the run adapted the mesh, in order."""
    return [int(cycle) for cycle in re.findall(r"AMR event: cycle=(\d+)", output)]


def dumps_of(directory):
    """The run's hydro_u dumps, in file-number order."""
    found = sorted((Path(directory) / "bin").glob(f"{basename}.hydro_u.*.bin"))
    assert found, f"{directory} wrote no binary dumps"
    return found


def dump_cycles(dumps):
    """{cycle: file number} for a run's dumps.

    A LAT run's outputs are made at WINDOW ends, not on every cycle, so the file number
    is not the cycle number the way it is in the non-LAT sibling test.  The cycle is in
    the plain-text preheader of the dump, and the rst output shares the same cadence, so
    this maps a cycle onto the file number of both.
    """
    cycles = {}
    for dump in dumps:
        with dump.open("rb") as handle:
            preheader = handle.read(256).decode("utf-8", "replace")
        match = re.search(r"cycle=(\d+)", preheader)
        assert match, f"{dump} has no cycle in its preheader"
        cycles.setdefault(int(match.group(1)),
                          int(dump.name.split(".")[-2]))
    return cycles


def blocks_of(dump):
    """One dump as {logical location: {variable: block data}}."""
    data = bin_convert.read_binary(str(dump))
    return {tuple(location): {name: values[block]
                              for name, values in data["mb_data"].items()}
            for block, location in enumerate(data["mb_logical"])}


def test_lat_amr_restart_is_bitwise(tmp_path):
    """Resuming at an AMR call that is not a multiple of ncycle_check reproduces it."""
    whole = tmp_path / "whole"
    output = run(["-i", input_file, "-d", str(whole)], "uninterrupted run")
    assert "LAT" in output, "the deck did not run with LAT"
    adapted = amr_cycles(output)
    assert adapted, "the uninterrupted run never adapted the mesh"
    # The point of the deck: under LAT the writer's AMR calls are not on the multiples of
    # the check period, so the resumed run cannot infer them from its cycle number.
    off_grid = [cycle for cycle in adapted if cycle % ncycle_check != 0]
    assert off_grid, (
        f"every AMR call of the writer fell on a multiple of {ncycle_check} "
        f"({adapted}), so this deck no longer exercises the cadence anchor")
    whole_dumps = dumps_of(whole)
    whole_cycles = dump_cycles(whole_dumps)
    whole_final = whole_dumps[-1]
    whole_blocks = blocks_of(whole_final)

    for cycle in (off_grid[0], adapted[-1]):
        assert cycle in whole_cycles, (
            f"the writer adapted at cycle {cycle} but wrote no output there")
        resumed = tmp_path / f"resumed_{cycle}"
        checkpoint = whole / "rst" / f"{basename}.{whole_cycles[cycle]:05d}.rst"
        assert checkpoint.exists(), f"no checkpoint at {checkpoint}"
        output = run(["-r", str(checkpoint), "-d", str(resumed)], f"restart at {cycle}")

        resumed_adapted = amr_cycles(output)
        assert resumed_adapted[:1] == [cycle], (
            f"restart at cycle {cycle} did not repeat the AMR pass the checkpoint was "
            f"written in front of; it adapted at {resumed_adapted}")
        assert resumed_adapted == [c for c in adapted if c >= cycle], (
            f"restart at cycle {cycle} adapted at {resumed_adapted}, the writer at "
            f"{adapted}: the resumed run is not on the writer's AMR cadence")

        resumed_final = dumps_of(resumed)[-1]
        assert resumed_final.name == whole_final.name, (
            f"restart at cycle {cycle} ended on dump {resumed_final.name}, the whole "
            f"run on {whole_final.name}")
        resumed_blocks = blocks_of(resumed_final)
        assert sorted(resumed_blocks) == sorted(whole_blocks), (
            f"restart at cycle {cycle} ended on a different mesh")
        for location, variables in resumed_blocks.items():
            for name, values in variables.items():
                assert (values == whole_blocks[location][name]).all(), (
                    f"restart at cycle {cycle}: block {location} variable {name} of "
                    f"{resumed_final.name} is not bitwise the uninterrupted run's")
