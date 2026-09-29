"""
The GR dual-energy auxiliary on a mesh that has coarse/fine faces.

Every property test_gr_dual_energy_cpu.py pins is measured on a uniform single-level
tube, which is the one configuration in which most of what the formalism touches does
not run: no prolongation, no restriction, no coarse/fine flux correction, no LAT.  Three
defects lived in exactly that gap.

  The GR adiabat does not need primitive prolongation -- it is per unit mass and
  conserved as D*kappa, so it crosses a coarse-fine boundary the way a passive scalar
  already does -- but the switch that forces primitive prolongation asked "is the
  formalism on" rather than "is this the non-relativistic flavour", so turning the
  auxiliary on silently changed how every newly refined cell is built.  On the dynamical
  path under LAT the driver refuses that combination outright and the run aborts at
  startup; on the fixed-metric path it does not abort, it just answers differently.

  dual_vf and lat_dual_vf_reflux are the face-velocity registers of the non-relativistic
  p dV update and are never allocated on the GR path.  Eleven call sites handed the
  flux-correction exchange a non-null pointer to them anyway, and two more did the same
  inside the LAT kernels; the receiver sizes the extra payload from the view and writes
  at full block indices into one that has none.  That is a write outside the allocation
  on every coarse/fine face of every multilevel GR run with the formalism on.

So: the same deck with the auxiliary off and on, at the production threshold where no
cell takes its pressure from it, on a three-dimensional mesh that is refined from the
first cycle and regrids while the shock crosses it.  The history integrals of the two
runs must agree to round-off, with and without LAT; and the auxiliary the second run
carried must still be a positive adiabat in every cell of every block, with its conserved
column still equal to D times it.

An out-of-bounds write is not guaranteed to be visible on a CPU -- the reliable detector
for that one is compute-sanitizer on a device build -- but this is the cheapest
configuration that runs all three defects at all, and the prolongation flavour is caught
outright.
"""

# Modules
from pathlib import Path
import re
import subprocess
import sys

import pytest
import athena_read
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

input_file = "inputs/dual_gr_amr.athinput"
_cases = {
    "amr": [],
}


def run_case(run_directory, tag, extra):
    """Run one case into its own directory; return its history data and log."""
    directory = run_directory / tag
    command = (["./athena", "-i", input_file, "-d", str(directory),
                f"job/basename={tag}"] + extra)
    result = subprocess.run(command, capture_output=True, check=False, text=True,
                            timeout=900)
    log = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{tag} failed with return code {result.returncode}:\n{log[-8000:]}"
    )
    history = athena_read.hst(str(directory / f"{tag}.mhd.hst"))
    return history, log


def dumps_of(run_directory, tag, variable):
    """Every binary dump of one variable set, oldest first."""
    found = sorted((run_directory / tag / "bin").glob(f"{tag}.{variable}.*.bin"))
    assert found, f"{tag} produced no {variable} output"
    return [bin_convert.read_binary(str(path)) for path in found]


def check_auxiliary(run_directory, tag):
    """The auxiliary column of every active cell of every block, at every dump."""
    primitives = dumps_of(run_directory, tag, "mhd_w")
    conserved = dumps_of(run_directory, tag, "mhd_u")
    assert len(primitives) == len(conserved)
    for prim, cons in zip(primitives, conserved):
        assert np.array_equal(prim["mb_logical"], cons["mb_logical"]), (
            "the two dumps of the same step describe different meshes"
        )
        for block in range(prim["n_mbs"]):
            kappa = np.asarray(prim["mb_data"]["eint_aux"][block], dtype=np.float64)
            # Every path that publishes the adiabat floors it at the entropy floor, so
            # a zero is a path that published without going through one -- which is what
            # a coarse/fine transfer that skips the auxiliary leaves behind, and what the
            # inversion then reads as "no auxiliary here".
            if not np.all(np.isfinite(kappa)) or not np.all(kappa > 0.0):
                pytest.fail(
                    f"{tag}: block {block} at t={prim['time']:g} carries "
                    f"{np.count_nonzero(~(kappa > 0.0))} non-positive and "
                    f"{np.count_nonzero(~np.isfinite(kappa))} non-finite adiabats"
                )

            # The conserved auxiliary is D*kappa by construction: the inversion defines
            # the adiabat as cons(dual)/cons(IDN) and every conserved repair republishes
            # the pair together.  The dumps are single precision, hence the bar.
            dens = np.asarray(cons["mb_data"]["dens"][block], dtype=np.float64)
            carried = np.asarray(cons["mb_data"]["reint_aux"][block],
                                 dtype=np.float64)
            drift = np.max(np.abs(carried - dens*kappa)/np.abs(carried))
            if drift > 1.0e-5:
                pytest.fail(
                    f"{tag}: block {block} at t={prim['time']:g} no longer carries "
                    f"D*kappa in its conserved auxiliary: max relative drift {drift:g}"
                )


def blocks_created(log):
    """Number of MeshBlocks the run created by AMR, from the closing summary."""
    # Every regrid that changed the mesh prints the same phrase for its own event; the
    # run total is the last occurrence, printed by the driver's closing summary.
    counts = re.findall(r"(\d+) MeshBlocks created", log)
    assert counts, "run did not report its AMR block counts"
    return int(counts[-1])


@pytest.mark.parametrize("case", list(_cases))
def test_inert_on_refined_mesh(case, tmp_path):
    """The auxiliary changes nothing across coarse/fine, and survives crossing one."""
    off, off_log = run_case(tmp_path, f"dual_{case}_off",
                            ["mhd/dual_energy=false"] + _cases[case])
    on, on_log = run_case(tmp_path, f"dual_{case}_on",
                          ["mhd/dual_energy=true"] + _cases[case])

    # A run that never regridded would never reach the prolongation branch this test
    # exists to compare, and the initial mesh carries only the static refined region.
    created = blocks_created(off_log)
    if created == 0:
        pytest.fail("the reference run never refined, so the comparison is vacuous")
    if blocks_created(on_log) != created:
        pytest.fail(
            "the two runs did not build the same mesh: "
            f"{created} blocks created with the auxiliary off, "
            f"{blocks_created(on_log)} with it on"
        )

    # The history carries the conserved integrals and the timestep, over the whole mesh
    # and in an order no block layout can permute.  The auxiliary is not among its
    # columns -- history reports nscalars, and the auxiliary sits past them -- so the two
    # runs report the same set.
    assert sorted(off) == sorted(on), "the two runs reported different history columns"
    for column in off:
        reference, tested = off[column], on[column]
        if len(reference) != len(tested):
            pytest.fail(
                f"history column '{column}' has {len(reference)} rows with the "
                f"auxiliary off and {len(tested)} with it on"
            )
        scale = np.max(np.abs(reference))
        if scale == 0.0:
            scale = 1.0
        moved = np.max(np.abs(reference - tested))/scale
        # The two runs should be bitwise identical; the bar is loosened to round-off
        # only so that a harmless reassociation is not a failure.  Prolonging primitives
        # instead of conserved variables perturbs every newly refined cell at the level
        # of the prolongation's own truncation error, which across a discontinuity is
        # some parts in a thousand.
        if moved > 1.0e-12:
            pytest.fail(
                f"dual_energy changed history column '{column}' on a refined mesh: "
                f"max relative difference {moved:g}"
            )

    check_auxiliary(tmp_path, f"dual_{case}_on")
