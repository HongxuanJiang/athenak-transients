"""The LAT window-end gravitational-work refund must carry the source term's own gate.

The gravitational work of a Godunov mass flux is applied per stage by
SourceTerms::Gravity, gated cell by cell on the stage primitive density against
``gravity/rho_grav_min``.  At a coarse/fine face the mass flux is only made conservative
by the LAT window-end reflux, which refunds the work of the mass it moves, so that
refund is owed for exactly the mass the gate admitted.  Reading the gate off the
post-step conserved density instead pays it for stages whose work was never applied.

The deck vents dense gas across a static coarse/fine face into an ambient held at the
density floor a decade below ``rho_grav_min``, so the coarse cells beside the face cross
the threshold while a window runs, every crossing being upward: the source is skipped at
every stage and the refund is then owed for nothing.

Two laws are checked, both on conserved global quantities.

The first is the energy-closure identity.  The potential is frozen over a LAT window, so
the gravitational work of that window is the redistribution of mass in that potential,
and the total energy change must equal it: ``dE + sum(d(rho) phi dV) = 0``.  The hydro
update is conservative and the deck's floors are far from the gas, so nothing else can
move total energy.  This holds with the gate active and is checked at round-off.

The second is what the delayed reflux is for.  The reflux exists so that a LAT run uses
the same flux history a synchronous run uses, and the gravitational-work bookkeeping has
to follow it.  The total work the run applies must therefore reproduce the work of the
same deck run with ``time/lat=false``, to the truncation error of the scheme.  Paying a
refund the gate never admitted is not a truncation error of the flux history, and it
pushes the LAT run away from the synchronous one: the window-end gate leaves the run
22.4% from the synchronous work, the admitted-mass refund 10.2%.
"""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

INPUT = "inputs/hydro_lat_grav_reflux_gate.athinput"
RHO_GRAV_MIN = 1.0e-15
# The closure is an identity of the discretisation, so it may only carry the round-off of
# summing the work over the mesh.
MAX_CLOSURE_RESIDUAL = 1.0e-10
# The LAT and synchronous runs differ by the truncation error of the flux history at the
# deck's fixed Courant number.  Halving it takes the admitted-mass refund to 2.9% and the
# window-end gate to 6.4%, so the two stay a factor 2.2 apart and this sits between them.
MAX_CADENCE_DISCREPANCY = 0.15


def _run(binary_flags, run_directory):
    command = ["./athena", "-i", INPUT] + binary_flags + ["-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=300)
    assert result.returncode == 0, (
        f"gravity reflux gate run failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )
    return result.stdout


def _applied_work(run_directory):
    """Total energy the run gained, which is the gravitational work it applied."""
    history = np.loadtxt(next(run_directory.glob("*.hst")))
    return history[-1][6] - history[0][6]


def _closure(first, second):
    """dE + sum(d(rho) phi dV) over one window, and the size of the work that went in."""
    start = bin_convert.read_binary(str(first))
    end = bin_convert.read_binary(str(second))
    change = 0.0
    work = 0.0
    scale = 0.0
    for block in range(len(start["mb_logical"])):
        x1min, x1max, x2min, x2max, x3min, x3max = start["mb_geometry"][block]
        rho0 = np.asarray(start["mb_data"]["dens"][block])
        rho1 = np.asarray(end["mb_data"]["dens"][block])
        energy0 = np.asarray(start["mb_data"]["ener"][block])
        energy1 = np.asarray(end["mb_data"]["ener"][block])
        phi = np.asarray(start["mb_data"]["grav_phi"][block])
        nx3, nx2, nx1 = rho0.shape
        volume = ((x1max - x1min)/nx1)*((x2max - x2min)/nx2)*((x3max - x3min)/nx3)
        change += float((energy1 - energy0).sum())*volume
        work += float(((rho1 - rho0)*phi).sum())*volume
        scale += float(np.abs((rho1 - rho0)*phi).sum())*volume
    return change + work, scale


def _crossings(dumps):
    """Coarse cells whose density crosses rho_grav_min from below between two dumps."""
    upward = 0
    for first, second in zip(dumps[:-1], dumps[1:]):
        start = bin_convert.read_binary(str(first))
        end = bin_convert.read_binary(str(second))
        for block in range(len(start["mb_logical"])):
            if start["mb_logical"][block][3] != 0:
                continue
            rho0 = np.asarray(start["mb_data"]["dens"][block])
            rho1 = np.asarray(end["mb_data"]["dens"][block])
            upward += int(((rho0 < RHO_GRAV_MIN) & (rho1 >= RHO_GRAV_MIN)).sum())
    return upward


def test_lat_gravity_work_closes_and_follows_the_flux_history(tmp_path):
    """The refund is owed for the admitted mass, so the work closes and matches sync."""

    lat_run = tmp_path / "lat"
    sync_run = tmp_path / "sync"
    log = _run([], lat_run)
    assert "LAT union stage-1 predictor enabled" in log, "the deck did not run with LAT"
    _run(["time/lat=false"], sync_run)

    dumps = sorted((lat_run / "bin").glob("*.hydro_u.*.bin"))
    assert len(dumps) >= 4, "the run wrote too few dumps to span a window"
    assert _crossings(dumps) > 0, (
        "no coarse cell crossed rho_grav_min, so the deck never exercises the gate"
    )

    residual, scale = _closure(dumps[-2], dumps[-1])
    assert scale > 0.0, "the window carried no gravitational work to close"
    assert abs(residual) <= MAX_CLOSURE_RESIDUAL*scale, (
        f"the window's energy change misses the work of its mass redistribution by "
        f"{abs(residual)/scale:.3e} of that work, which is not round-off"
    )

    lat_work = _applied_work(lat_run)
    sync_work = _applied_work(sync_run)
    discrepancy = abs(lat_work - sync_work)/abs(sync_work)
    assert discrepancy <= MAX_CADENCE_DISCREPANCY, (
        f"the LAT run applied {lat_work:.6e} of gravitational work against the "
        f"synchronous run's {sync_work:.6e}, {discrepancy:.4f} apart: the window-end "
        f"refund is being paid for mass the source-term gate never admitted"
    )
