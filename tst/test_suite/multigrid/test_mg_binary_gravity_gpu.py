"""
Binary gravity multigrid tests for GPU.

Tests the multigrid Poisson solver with the binary gravity problem (two
uniform-density spheres) on uniform and SMR meshes.

Tests include:
  - Defect convergence per V-cycle (uniform 64^3, 2-level SMR 16^3 base)
  - Defect consistency: both configs have the same effective finest
    resolution (64^3); their per-V-cycle geometric-mean contraction ratio
    should be comparable (final defects legitimately differ, since the
    solver stops at the first defect below threshold).

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

import pytest

from test_suite.multigrid.mg_utils import (
    assert_solver_convergence,
    run_athenak,
    parse_final_defect,
    compute_geo_mean_ratio,
    assert_geo_mean_consistency,
    cleanup,
)
threshold = 4E-9

_GRAVITY_FLAGS = [
    "time/nlim=1",
    f"gravity/threshold={threshold}",
    "gravity/niteration=-1",
    "gravity/npresmooth=2",
    "gravity/npostsmooth=2",
    "gravity/full_multigrid=true",
]

_stored_defects = {}
_stored_ratios = {}


# ---------------------------------------------------------------------------
# Defect convergence per V-cycle (uniform, SMR)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("res", [64, 128])
@pytest.mark.parametrize("rg", [4, 8])
def test_binary_gravity_uniform_gpu(res, rg):
    """Binary gravity MG defect convergence on 64^3 uniform mesh."""
    try:
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
                f"meshblock/nx1={res//rg}",
                f"meshblock/nx2={res//rg}",
                f"meshblock/nx3={res//rg}",
                "mesh_refinement/refinement=none",
                "gravity/show_defect=2",
            ] + _GRAVITY_FLAGS,
        )
        assert results[0], "Binary gravity uniform GPU run failed"
        assert_solver_convergence(results[1], threshold, max_iterations=10,
                                  max_avg_ratio=0.0625,
                                  label="binary_gravity_uniform_gpu: ")
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["uniform"] = d
        r = compute_geo_mean_ratio(results[1])
        if r is not None:
            _stored_ratios["uniform"] = r
    finally:
        cleanup()


@pytest.mark.parametrize("res", [32])
@pytest.mark.parametrize("rg", [8])
def test_binary_gravity_smr_gpu(res, rg):
    """Binary gravity with 2-level SMR (32^3 base, effective finest = 128^3)."""
    try:
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
                f"meshblock/nx1={res//rg}",
                f"meshblock/nx2={res//rg}",
                f"meshblock/nx3={res//rg}",
                "refined_region3/level=2",
                "refined_region4/level=2",
                "gravity/show_defect=2",
            ] + _GRAVITY_FLAGS,
        )
        assert results[0], "Binary gravity SMR GPU run failed"
        assert_solver_convergence(results[1], threshold, max_iterations=16,
                                  max_avg_ratio=0.18,
                                  label="binary_gravity_smr_gpu: ")
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["smr"] = d
        r = compute_geo_mean_ratio(results[1])
        if r is not None:
            _stored_ratios["smr"] = r
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Defect consistency: uniform vs SMR at same effective resolution
# ---------------------------------------------------------------------------

def test_binary_gravity_defect_consistency_gpu():
    """Uniform and SMR should converge at a comparable per-cycle rate.

    Final defects differ legitimately (the solver stops at the first defect
    below threshold), so this compares the geometric-mean per-V-cycle
    contraction ratio instead; see assert_geo_mean_consistency.
    """
    if "uniform" not in _stored_defects or "smr" not in _stored_defects:
        pytest.skip("Prerequisite tests did not run or failed")
    if "uniform" not in _stored_ratios or "smr" not in _stored_ratios:
        pytest.skip("Prerequisite tests did not run or failed")
    assert_geo_mean_consistency(
        _stored_defects, _stored_ratios, threshold,
        label="binary_gravity_defect_consistency_gpu: ",
    )
