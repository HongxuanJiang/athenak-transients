"""A low-beta shock keeps its heating under the dual-energy formalism.

Below eint = 0.1*B^2/2 the Newtonian MHD inversion rejects the energy channel
E - KE - B^2/2 as truncation error of the magnetic energy, takes the auxiliary internal
energy and resets E from it (eos/general_c2p_mhd.hpp).  The auxiliary only compresses
adiabatically, so a shock through such gas would lose the entropy it generates, unless
the resynchronisation recognises the shock (supersonic convergence across the cell) and
keeps the energy channel there.

The deck is the Brio-Wu tube at a tenth of its gas pressure: the slow shock runs into
beta = 0.013 gas.  The reference is the conservative scheme itself (dual energy off),
which captures shock heating exactly, at four times the resolution.  The gate: the
dual-energy run may not be further from that reference, in density or in specific
internal energy, than the conservative run at the same resolution.  With the heating
dropped the post-shock plateau came out at 0.64 of its specific energy and both errors
doubled.
"""

from pathlib import Path
import os
import subprocess

import numpy as np

DECK = "inputs/mhd_bw_lowbeta_dual.athinput"
NX = 256
REFINEMENT = 4


def _run(run_directory, overrides):
    command = ["./athena", "-i", DECK, *overrides, "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=300)
    assert result.returncode == 0, (
        f"shock tube run failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )
    dumps = sorted(Path(run_directory, "tab").glob("*.mhd_w.*.tab"))
    assert len(dumps) >= 2, "the run wrote no final dump"
    data = np.loadtxt(dumps[-1])
    density = data[:, 3]
    return density, data[:, 7]/density


def test_lowbeta_shock_keeps_its_heating(tmp_path):
    """Dual energy is no further from the converged shock tube than E alone."""

    nx = NX*REFINEMENT
    ref_d, ref_eps = _run(tmp_path / "reference",
                          ["mhd/dual_energy=false", f"mesh/nx1={nx}",
                           f"meshblock/nx1={nx}"])
    ref_d = ref_d.reshape(NX, REFINEMENT).mean(axis=1)
    ref_eps = ref_eps.reshape(NX, REFINEMENT).mean(axis=1)
    cons_d, cons_eps = _run(tmp_path / "conservative", ["mhd/dual_energy=false"])
    dual_d, dual_eps = _run(tmp_path / "dual", [])

    for name, dual, cons, ref in (("density", dual_d, cons_d, ref_d),
                                  ("specific internal energy", dual_eps, cons_eps,
                                   ref_eps)):
        dual_error = float(np.mean(np.abs(dual - ref)))
        cons_error = float(np.mean(np.abs(cons - ref)))
        assert dual_error <= cons_error, (
            f"low-beta Brio-Wu: the dual-energy {name} is {dual_error:.4e} (L1) from "
            f"the {nx}-cell conservative solution, the conservative scheme at {NX} "
            f"cells only {cons_error:.4e}: the shock heating behind the slow shock "
            f"was dropped by the dual-energy selection (peak specific energy "
            f"{dual_eps.max():.4f} against {ref_eps.max():.4f})"
        )
