"""
A restart carries the GR dual-energy auxiliary, and the resumed run is the uninterrupted one.

The auxiliary is one more conserved column, D*kappa, so a checkpoint has to write it and a
restart has to read it back into the same slot.  Reading a checkpoint of the wrong width is
handled by re-seeding the adiabat from the primitives, which is the right thing to do for an
old checkpoint and the wrong thing to do silently for a current one: after a shock the
advected adiabat and the one the pressure implies differ by the entropy the shock made, so
a re-seeded restart is a different run.  The test therefore forces the channel into every
cell (eta1 = 1), where the pressure comes from the advected adiabat and nothing else, and
demands that a run restarted at half time reproduces the uninterrupted run to round-off.
Both GR backends, since each has its own checkpoint width bookkeeping.

Round-off, not the bit: a cell that takes the channel has its conserved energy regenerated
from the published primitives, so the checkpoint holds the primitive-to-conserved image of
the state the run went on with, and inverting that image again reproduces the primitives
to a few ulp (measured 4e-14 on the dynamical backend, exact on the fixed-metric one).
With the channel off the same restart is exact on both.  A re-seeded adiabat sits at the
size of the shock's entropy jump, some parts in ten, eleven orders above the bar.
"""

# Modules
import subprocess

import pytest
import athena_read
import numpy as np

_decks = {
    "fixed": "inputs/dual_gr_tube.athinput",
    "dyn": "inputs/dual_dyngr_tube.athinput",
}
_tlim = 0.4      # both decks
_checkpoint = 0.2

_rst_block = """
<output9>
file_type = rst
dt        = {dt}
"""


def run(command, tag):
    """Run the binary; fail with the log if it does not exit cleanly."""
    result = subprocess.run(command, capture_output=True, check=False, text=True,
                            timeout=600)
    assert result.returncode == 0, (
        f"{tag} failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-6000:]}"
    )


@pytest.mark.parametrize("flavour", list(_decks))
def test_restart_reproduces_run(flavour, tmp_path):
    """The run resumed from a checkpoint ends where the uninterrupted run ends."""
    # The shared decks carry no restart output, so a copy with one appended is used;
    # everything else in it, including the dumps the comparison reads, stays the same.
    deck = tmp_path / f"dual_restart_{flavour}.athinput"
    deck.write_text(open(_decks[flavour]).read()
                    + _rst_block.format(dt=_checkpoint))
    forced = ["mhd/dual_energy=true", "mhd/dual_energy_eta1=1.0"]

    full = tmp_path / "full"
    run(["./athena", "-i", str(deck), "-d", str(full), "job/basename=full"] + forced,
        f"{flavour} uninterrupted run")
    checkpoint = full / "rst" / "full.00001.rst"
    assert checkpoint.exists(), f"no checkpoint written at t={_checkpoint}"

    resumed = tmp_path / "resumed"
    run(["./athena", "-r", str(checkpoint), "-d", str(resumed)] + forced,
        f"{flavour} resumed run")

    # The last primitive dump of each run, at tlim.  The restart keeps the basename.
    def last(directory):
        dumps = sorted((directory / "tab").glob("full.mhd_w.*.tab"))
        assert dumps, f"{directory} holds no primitive dumps"
        return athena_read.tab(str(dumps[-1]))

    reference, tested = last(full), last(resumed)
    assert reference["time"] == tested["time"] == _tlim, "the two runs ended at different times"
    for column in reference:
        if column in ("time", "cycle", "i", "x1v"):
            continue
        scale = np.max(np.abs(reference[column]))
        moved = np.max(np.abs(reference[column] - tested[column]))/(scale if scale else 1.0)
        if moved > 1.0e-12:
            pytest.fail(
                f"restart changed column '{column}': max relative difference {moved:g}"
            )
