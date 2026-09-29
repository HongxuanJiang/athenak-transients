"""
A run on an adaptive mesh, restarted from a checkpoint, is the uninterrupted run.

Driver::Execute makes a cycle's outputs BEFORE it calls
MeshRefinement::AdaptiveMeshRefinement, so a checkpoint always carries the mesh the
cycle that just ended was integrated on -- never the adapted mesh the writing run
carried into its next cycle.  A restart that does not repeat that pass takes its
first step on the stale tree and, because the tree is then adapted one cycle late,
every later AMR event lags the uninterrupted run for the rest of the run.

The deck writes a checkpoint every cycle, so the restart points below are the cycles
on which the uninterrupted run actually adapted the mesh -- the ones that carry the
owed pass.  Its refinement_interval equals its ncycle_check, which keeps the
per-MeshBlock refinement cooldown (run history no checkpoint carries) inert; that is
what makes bitwise the right assertion here rather than a tolerance.
"""

# Modules
import re
import subprocess
from pathlib import Path

input_file = "inputs/linear_wave_amr_restart.athinput"
basename = "LinWaveAMRRestart"


def run(command, tag):
    """Run the binary on two ranks; fail with the log if it does not exit cleanly."""
    result = subprocess.run(["mpirun", "-np", "2", "./athena"] + command,
                            capture_output=True, check=False, text=True, timeout=600)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{tag} failed with return code {result.returncode}:\n{output[-6000:]}"
    )
    return output


def amr_cycles(output):
    """The cycles on which the run adapted the mesh, in order."""
    return [int(cycle) for cycle in re.findall(r"AMR event: cycle=(\d+)", output)]


def history_rows(directory):
    """The history file's data rows, as written."""
    path = Path(directory) / f"{basename}.hydro.hst"
    assert path.exists(), f"{directory} wrote no history file"
    return [line for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def test_amr_restart_is_bitwise(tmp_path):
    """Resuming at a cycle the writer adapted on reproduces it row for row."""
    whole = tmp_path / "whole"
    output = run(["-i", input_file, "-d", str(whole)], "uninterrupted run")
    adapted = amr_cycles(output)
    assert adapted, "the uninterrupted run never adapted the mesh"
    whole_rows = history_rows(whole)

    for cycle in (adapted[0], adapted[-1]):
        resumed = tmp_path / f"resumed_{cycle}"
        # The deck checkpoints every cycle, so the file number is the cycle number.
        checkpoint = whole / "rst" / f"{basename}.{cycle:05d}.rst"
        assert checkpoint.exists(), f"no checkpoint at {checkpoint}"
        output = run(["-r", str(checkpoint), "-d", str(resumed)], f"restart at {cycle}")

        assert amr_cycles(output)[:1] == [cycle], (
            f"restart at cycle {cycle} did not repeat the AMR pass the checkpoint was "
            f"written in front of; it adapted at {amr_cycles(output)}")

        resumed_rows = history_rows(resumed)
        offset = len(whole_rows) - len(resumed_rows)
        assert offset >= 0, "the restart wrote more history rows than the whole run"
        for n, row in enumerate(resumed_rows):
            assert row == whole_rows[offset + n], (
                f"restart at cycle {cycle}, history row {n} differs:\n"
                f"  whole:   {whole_rows[offset + n]}\n"
                f"  resumed: {row}")
