"""
Ghost zones of MeshBlocks narrower than 2*nghost on a multilevel mesh.

Such a MeshBlock sends a coarser neighbour, besides its restricted interior, coarse ghost
cells from its far side, which its own exchange filled from the neighbour there.  Every
ghost cell with a same-level owner must hold the owner's value, and one with finer owners
their conservative average (ghost cells prolongated from a coarser owner are not
compared): exactly after start-up, and during the run to within the few stages by which
a passed-on cell lags.  A cell nothing fills holds zero and lands on the density floor.

The decks are MHD linear waves of amplitude 1e-3 on blocks of 4 cells with nghost = 4:
a 1D mesh with one refined level, and a 2D mesh with two nested ones, where a single
level-1 block lies between the root level and level 2.
"""

# Modules
import subprocess
from pathlib import Path

import numpy as np

import bin_convert

STAGE_LAG = 1.0e-4   # a tenth of the wave amplitude


def run(command, tag, nranks=2):
    """Run the binary on nranks ranks; fail with the log if it does not exit cleanly."""
    result = subprocess.run(["mpirun", "-np", str(nranks), "./athena"] + command,
                            capture_output=True, check=False, text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{tag} failed with return code {result.returncode}:\n{output[-6000:]}")
    return output


def ghost_deviation(filename):
    """Largest |ghost - owner| over every ghost cell with a same-level or finer owner."""
    data = bin_convert.read_binary(str(filename))
    nx = [data["nx1_mb"], data["nx2_mb"], data["nx3_mb"]]
    ncell = [data["Nx1"], data["Nx2"], data["Nx3"]]
    active = [n > 1 for n in ncell]
    nghost = int(next(line for line in data["header"]
                      if line.startswith("nghost")).split("=")[1])
    ng = [nghost if a else 0 for a in active]
    blocks = {(int(lev), int(l1), int(l2), int(l3)): m
              for m, (l1, l2, l3, lev) in enumerate(data["mb_logical"])}

    def owner(var, lev, cell):
        cell = [cell[a] % (ncell[a] << lev) if active[a] else 0 for a in range(3)]
        key = (lev,) + tuple(cell[a] // nx[a] for a in range(3))
        if key in blocks:
            loc = [cell[a] - key[a + 1] * nx[a] + ng[a] for a in range(3)]
            return float(data["mb_data"][var][blocks[key]][loc[2], loc[1], loc[0]])
        for up in range(1, lev + 1):
            if (lev - up,) + tuple((cell[a] >> up) // nx[a] for a in range(3)) in blocks:
                return None   # prolongated from a coarser owner
        children = []
        for c in range(8):
            off = [(c >> a) & 1 for a in range(3)]
            if any(off[a] and not active[a] for a in range(3)):
                continue
            value = owner(var, lev + 1, [2 * cell[a] + off[a] for a in range(3)])
            if value is None:
                return None
            children.append(value)
        return sum(children) / len(children)

    worst = 0.0
    for var in data["var_names"]:
        for (lev, *lloc), m in blocks.items():
            values = data["mb_data"][var][m]
            for k, j, i in np.ndindex(values.shape):
                loc = (i, j, k)
                if all(ng[a] <= loc[a] < ng[a] + nx[a] for a in range(3)):
                    continue
                cell = [lloc[a] * nx[a] + loc[a] - ng[a] for a in range(3)]
                value = owner(var, lev, cell)
                if value is not None:
                    worst = max(worst, abs(float(values[k, j, i]) - value))
    return worst


def assert_same_dumps(reference, directory):
    """Every dump in directory holds the same data as its namesake in reference."""
    dumps = sorted((directory / "bin").glob("*.bin"))
    assert dumps, f"{directory} wrote no dumps"
    for dump in dumps:
        a = bin_convert.read_binary(str(reference / "bin" / dump.name))
        b = bin_convert.read_binary(str(dump))
        for name in a["var_names"]:
            for m in range(a["n_mbs"]):
                x = np.asarray(a["mb_data"][name][m])
                y = np.asarray(b["mb_data"][name][m])
                assert np.array_equal(x, y), (
                    f"{dump.name} {name} block {m} differs by "
                    f"{float(np.max(np.abs(x - y))):.3e}")


def check_ghosts(deck, tmp_path):
    """Exact after start-up, within the stage lag at every later cycle."""
    directory = tmp_path / Path(deck).stem
    run(["-i", deck, "-d", str(directory)], deck)
    # The cell-centred field averages face values, which does not commute exactly with
    # the restriction, so only the conserved variables are compared exactly.
    for variable, exact in (("mhd_u", True), ("mhd_bcc", False)):
        dumps = sorted((directory / "bin").glob(f"*.{variable}.*.bin"))
        assert len(dumps) > 1, f"{deck} wrote no {variable} dumps"
        for n, dump in enumerate(dumps):
            limit = 1.0e-13 if (exact and n == 0) else STAGE_LAG
            deviation = ghost_deviation(dump)
            assert deviation <= limit, (
                f"{dump.name}: a ghost cell is {deviation:.3e} off its owner's value")


def test_one_dimensional_ghosts_hold_owner_values(tmp_path):
    """1D: the coarse ghost cells facing a same-level neighbour are filled."""
    check_ghosts("inputs/narrow_blocks_1d.athinput", tmp_path)


def test_three_level_ghosts_hold_owner_values(tmp_path):
    """The coarse ghost cells facing a finer neighbour are filled."""
    check_ghosts("inputs/narrow_blocks_3level.athinput", tmp_path)


def test_restart_matches_straight_run(tmp_path):
    """A verbatim restart continues the straight run bit for bit, ghost zones included.

    The deck steps with rk1, so the coarse ghost cells a narrow block sends in the
    first step after the restart are still in its neighbours' ghost zones when the
    step ends."""
    deck = "inputs/narrow_blocks_3level.athinput"
    whole = tmp_path / "whole"
    resumed = tmp_path / "resumed"
    output_whole = run(["-i", deck, "-d", str(whole)], "straight run")
    checkpoint = whole / "rst" / "narrow3level.00001.rst"
    assert checkpoint.exists(), "the straight run wrote no checkpoint at cycle 2"
    output_resumed = run(["-r", str(checkpoint), "-d", str(resumed)], "restarted run")

    def steps(output):
        return [line.split("elapsed=")[1].split(" ", 1)[1] for line in
                output.splitlines() if "elapsed=" in line and " cycle=" in line]
    tail = steps(output_resumed)
    assert tail and tail == steps(output_whole)[-len(tail):], (
        f"the restarted run took different steps:\n{tail}")
    assert_same_dumps(whole, resumed)


def test_rank_layout_does_not_change_the_run(tmp_path):
    """One and two ranks give the same run, ghost zones included.

    With more than one rank the cells for a same-rank same-level neighbour are copied
    straight into its ghost cells while the other buffers are packed, and a narrow
    block's pack for a finer neighbour reads its own ghost cells."""
    deck = "inputs/narrow_blocks_3level.athinput"
    for nranks in (1, 2):
        run(["-i", deck, "-d", str(tmp_path / f"np{nranks}")], f"{nranks} rank run",
            nranks=nranks)
    assert_same_dumps(tmp_path / "np1", tmp_path / "np2")
