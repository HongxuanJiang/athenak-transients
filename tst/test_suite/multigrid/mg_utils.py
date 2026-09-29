"""
Shared utilities for multigrid test suite.

Provides helpers to run AthenaK with stdout capture and parse multigrid-specific
output (defect norms, binary gravity errors, Jeans wave errors).
"""

import os
import re
import math
import logging
from subprocess import Popen, PIPE
from typing import List, Dict, Optional
from test_suite.testutils import cleanup

import pytest

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "test_log.txt")
)


def run_athenak(inputfile, flags=None, mpi=False, threads=1):
    """Run the AthenaK binary and return its success flag together with stdout.

    The multigrid tests parse solver diagnostics (defect norms, Jeans growth
    rates, per-iteration output) that AthenaK prints to stdout. The shared
    ``testutils.run``/``testutils.mpi_run`` helpers only return a success bool,
    so this local runner captures and returns the output as well.

    Args:
        inputfile (str): Path to the AthenaK input file.
        flags (list): Additional command-line flags for the AthenaK binary.
        mpi (bool): Whether to launch the run under mpirun.
        threads (int): Number of MPI ranks (only used when mpi=True).

    Returns:
        list: [success (bool), stdout (str)].
    """
    if flags is None:
        flags = []
    if mpi:
        command = ["mpirun", "-np", str(threads), "./athena", "-i", inputfile] + flags
    else:
        command = ["./athena", "-i", inputfile] + flags
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
    output, _errors = process.communicate()
    if process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
    return [process.returncode == 0, output]


def parse_mg_defects(stdout: str) -> List[float]:
    """Parse MG defect values from stdout.

    Matches lines like:
      MG initial defect = 1.234e-01
      MG iteration 0: defect = 5.678e-05
      MG iteration N: defect = ...
    Returns flat list of defect values in order (initial first, then iterations).
    """
    pattern = re.compile(
        r"(?:MG\s+initial\s+defect|MG\s+iteration\s+\d+:\s+defect)\s*=\s*([0-9.eE+\-]+)")
    defects = []
    for match in pattern.finditer(stdout):
        defects.append(float(match.group(1)))
    return defects


def parse_mg_defects_per_solve(stdout: str) -> List[List[float]]:
    """Parse MG defect values grouped by solve invocation.

    Each group starts with "MG initial defect" and contains subsequent
    "MG iteration" values.  Returns a list of lists, one per solve.
    """
    pat_init = re.compile(r"MG\s+initial\s+defect\s*=\s*([0-9.eE+\-]+)")
    pat_iter = re.compile(r"MG\s+iteration\s+\d+:\s+defect\s*=\s*([0-9.eE+\-]+)")
    solves: List[List[float]] = []
    for line in stdout.splitlines():
        m = pat_init.search(line)
        if m:
            solves.append([float(m.group(1))])
            continue
        m = pat_iter.search(line)
        if m and solves:
            solves[-1].append(float(m.group(1)))
    return solves


def parse_binary_gravity_errors(stdout: str) -> Dict[str, float]:
    """Parse BinaryGravityErrors output from stdout.

    Matches lines like:
      Potential    L2       : 1.234e-02
      Acceleration L2       : 5.678e-03
      Max Potential Error    : 9.012e-02
      Max Acceleration Error : 3.456e-02
    Returns dict with keys: pot_l2, acc_l2, pot_max, acc_max.
    """
    result = {}
    patterns = {
        "pot_l2": re.compile(r"Potential\s+L2\s*:\s*([0-9.eE+\-]+)"),
        "acc_l2": re.compile(r"Acceleration\s+L2\s*:\s*([0-9.eE+\-]+)"),
        "pot_max": re.compile(r"Max\s+Potential\s+Error\s*:\s*([0-9.eE+\-]+)"),
        "acc_max": re.compile(r"Max\s+Acceleration\s+Error\s*:\s*([0-9.eE+\-]+)"),
    }
    for key, pat in patterns.items():
        m = pat.search(stdout)
        if m:
            result[key] = float(m.group(1))
    return result


def parse_jeans_omega(stdout: str) -> Optional[Dict[str, float]]:
    """Parse measured and analytical omega from JeansWaveErrors output.

    Matches lines like:
      Jeans wave omega measured  : 1.234e+01
      Jeans wave omega analytical: 1.234e+01
    Returns dict with keys 'measured' and 'analytical', or None if not found.
    """
    m_meas = re.search(
        r"Jeans\s+wave\s+omega\s+measured\s*:\s*([0-9.eE+\-]+)", stdout)
    m_anal = re.search(
        r"Jeans\s+wave\s+omega\s+analytical\s*:\s*([0-9.eE+\-]+)", stdout)
    if m_meas and m_anal:
        return {"measured": float(m_meas.group(1)),
                "analytical": float(m_anal.group(1))}
    return None


def assert_defect_convergence(stdout: str, min_orders: float = 8.0,
                              label: str = ""):
    """Assert that the MG defect drops by at least min_orders of magnitude.

    Parses all defect values from stdout and checks that
    final_defect / initial_defect < 10^{-min_orders}.
    """
    defects = parse_mg_defects(stdout)
    if len(defects) < 2:
        pytest.fail(f"{label}Expected at least 2 defect values, got {len(defects)}")

    initial = defects[0]
    final = defects[-1]
    if initial <= 0:
        pytest.fail(f"{label}Initial defect is non-positive: {initial}")

    ratio = final / initial
    threshold = 10.0 ** (-min_orders)
    if ratio > threshold:
        orders = -math.log10(ratio) if ratio > 0 else float("inf")
        pytest.fail(
            f"{label}Defect reduction insufficient: {orders:.1f} orders "
            f"(need {min_orders}). Initial={initial:.3e}, Final={final:.3e}"
        )


def assert_binary_gravity_accuracy(stdout: str,
                                   pot_l2_max: float = 1.0,
                                   acc_l2_max: float = 1.0,
                                   label: str = ""):
    """Assert that binary gravity L2 errors are below thresholds."""
    errs = parse_binary_gravity_errors(stdout)
    if not errs:
        pytest.fail(f"{label}No binary gravity errors found in output")

    if errs.get("pot_l2", float("inf")) > pot_l2_max:
        pytest.fail(
            f"{label}Potential L2 error {errs['pot_l2']:.3e} exceeds "
            f"threshold {pot_l2_max:.3e}"
        )
    if errs.get("acc_l2", float("inf")) > acc_l2_max:
        pytest.fail(
            f"{label}Acceleration L2 error {errs['acc_l2']:.3e} exceeds "
            f"threshold {acc_l2_max:.3e}"
        )


def assert_jeans_growth_rate(input_file: str, flags_fn, res_list: List[int],
                             max_rel_error: float, max_ratio: float,
                             mpi: bool = False, nranks: int = 4,
                             label: str = ""):
    """Run Jeans wave test at multiple resolutions and check growth rate accuracy.

    At each resolution, parses the measured and analytical omega from the C++
    output, computes the relative error, and then checks:
      1. Threshold: relative error at highest resolution < max_rel_error
      2. Convergence: ratio of errors (high_res / low_res) < max_ratio

    Args:
        input_file: Path to the Jeans wave input file.
        flags_fn: Callable(res) -> list of flag strings for a given resolution.
        res_list: List of resolutions to test (e.g. [32, 64]).
        max_rel_error: Maximum allowed relative omega error at highest resolution.
        max_ratio: Maximum allowed ratio of errors (high_res / low_res).
        mpi: Whether to use MPI.
        nranks: Number of MPI ranks.
        label: Label for error messages.
    """
    rel_errors = []
    for res in res_list:
        flags = flags_fn(res)
        results = run_athenak(input_file, flags, mpi=mpi, threads=nranks)
        if not results[0]:
            pytest.fail(f"{label}Run failed at resolution {res}")
        omega_data = parse_jeans_omega(results[1])
        if omega_data is None:
            pytest.fail(f"{label}No omega data found at resolution {res}")
        rel_err = (abs(omega_data["measured"] - omega_data["analytical"])
                   / omega_data["analytical"])
        rel_errors.append(rel_err)
        cleanup()

    if rel_errors[-1] > max_rel_error:
        pytest.fail(
            f"{label}Omega relative error {rel_errors[-1]:.4e} at "
            f"res={res_list[-1]} exceeds threshold {max_rel_error:.4e}"
        )

    if len(rel_errors) >= 2:
        ratio = rel_errors[-1] / rel_errors[-2]
        if ratio > max_ratio:
            pytest.fail(
                f"{label}Omega error ratio {ratio:.3f} exceeds threshold "
                f"{max_ratio:.3f}. Errors: {[f'{e:.4e}' for e in rel_errors]}"
            )


def parse_amr_block_counts(stdout: str) -> Optional[Dict[str, int]]:
    """Parse the run-total AMR block creation/deletion counts from stdout.

    AthenaK prints one cumulative line per AMR event, e.g.:
      1008 MeshBlocks created, 0 deleted by AMR
    (see src/mesh/mesh_refinement.cpp:1630-1632). Each line already reports
    the running total, so the LAST match is the run total, not the first.
    Returns dict with keys 'created' and 'deleted', or None if not found.
    """
    matches = re.findall(
        r"(\d+)\s+MeshBlocks\s+created,\s+(\d+)\s+deleted\s+by\s+AMR", stdout)
    if matches:
        created, deleted = matches[-1]
        return {"created": int(created), "deleted": int(deleted)}
    return None


def parse_final_defect(stdout: str) -> Optional[float]:
    """Parse the last 'Final defect norm' value from stdout.

    Matches: MGGravityDriver::Solve: Final defect norm = 1.234e-06
    Returns the last occurrence (there may be multiple from successive timesteps).
    """
    matches = re.findall(
        r"Final\s+defect\s+norm\s*=\s*([0-9.eE+\-]+)", stdout)
    if matches:
        return float(matches[-1])
    return None


def _solve_geo_mean_ratio(defects: List[float]) -> Optional[float]:
    """Geometric mean of the per-V-cycle contraction ratios of one solve.

    Shared by assert_solver_convergence and compute_geo_mean_ratio so the
    ratio arithmetic (and its edge cases) is defined in exactly one place.
    Returns None if there are no positive-defect consecutive pairs to form a
    ratio from (e.g. a solve with only an initial defect and no V-cycles).
    """
    ratios = [defects[i + 1] / defects[i]
              for i in range(len(defects) - 1) if defects[i] > 0]
    if not ratios:
        return None
    if any(r == 0.0 for r in ratios):
        return 0.0
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def assert_solver_convergence(stdout: str, threshold: float,
                              max_iterations: int = 40,
                              max_avg_ratio: float = 0.5,
                              label: str = ""):
    """Assert that SolveIterative reached the target threshold.

    Parses per-solve defect trajectories and checks across all solves:
      1. Every final defect <= threshold
      2. Max V-cycles in any single solve <= max_iterations
      3. Worst-case average convergence ratio <= max_avg_ratio

    Returns (max_vcycles, worst_geo_mean, solves).
    """
    solves = parse_mg_defects_per_solve(stdout)
    if not solves:
        pytest.fail(f"{label}No MG defect output found")

    max_vcycles = 0
    worst_geo_mean = 0.0
    for si, defects in enumerate(solves):
        if not defects:
            pytest.fail(
                f"{label}Solve {si}: empty MG defect trajectory")
        if any(not math.isfinite(defect) or defect < 0.0
               for defect in defects):
            pytest.fail(
                f"{label}Solve {si}: invalid MG defect trajectory {defects}")
        n_vc = len(defects) - 1
        max_vcycles = max(max_vcycles, n_vc)

        final = defects[-1]
        if final > threshold:
            pytest.fail(
                f"{label}Solve {si}: final defect {final:.3e} > "
                f"{threshold:.3e} after {n_vc} V-cycles")

        geo_mean = _solve_geo_mean_ratio(defects)
        if geo_mean is not None:
            worst_geo_mean = max(worst_geo_mean, geo_mean)

    if max_vcycles > max_iterations:
        pytest.fail(
            f"{label}Solver took too many iterations: {max_vcycles} > "
            f"{max_iterations}")

    if worst_geo_mean > max_avg_ratio:
        pytest.fail(
            f"{label}Worst average convergence ratio {worst_geo_mean:.4f} "
            f"exceeds {max_avg_ratio}")

    return max_vcycles, worst_geo_mean, solves


def compute_geo_mean_ratio(stdout: str) -> Optional[float]:
    """Return the worst-case per-V-cycle geometric-mean contraction ratio.

    Parses per-solve defect trajectories the same way assert_solver_convergence
    does (via parse_mg_defects_per_solve + _solve_geo_mean_ratio), and returns
    the largest geometric-mean ratio across solves -- the same quantity
    assert_solver_convergence checks against max_avg_ratio. Returns None if no
    MG defect output was found, or if no solve had a usable ratio.
    """
    solves = parse_mg_defects_per_solve(stdout)
    if not solves:
        return None
    worst = None
    for defects in solves:
        geo_mean = _solve_geo_mean_ratio(defects)
        if geo_mean is not None:
            worst = geo_mean if worst is None else max(worst, geo_mean)
    return worst


def assert_defect_consistency(defects: List[float], max_spread: float,
                              label: str = ""):
    """Assert that all defect values are within max_spread orders of magnitude.

    Checks that log10(max/min) < max_spread across the provided defect values.
    """
    if not defects:
        pytest.fail(f"{label}No defect values to compare")
    dmin, dmax = min(defects), max(defects)
    if dmin <= 0:
        pytest.fail(f"{label}Non-positive defect: {dmin}")
    spread = math.log10(dmax / dmin)
    if spread > max_spread:
        pytest.fail(
            f"{label}Defect spread {spread:.2f} orders exceeds threshold "
            f"{max_spread}. Values: {[f'{d:.3e}' for d in defects]}"
        )


def assert_geo_mean_consistency(final_defects: Dict[str, float],
                                geo_mean_ratios: Dict[str, float],
                                threshold: float,
                                max_log_ratio: float = 0.5,
                                label: str = ""):
    """Assert uniform and SMR runs converge at a comparable per-cycle rate.

    The solver stops at the first defect below `threshold`, so on uniform vs
    SMR meshes the final defect can legitimately differ by up to one V-cycle
    of contraction (~1.3 orders); comparing final defects directly is
    therefore the wrong test. Instead this checks that both final defects are
    below `threshold`, and that the per-V-cycle geometric-mean contraction
    ratio (from compute_geo_mean_ratio) agrees to within `max_log_ratio` in
    log10 space: abs(log10(r_smr / r_uniform)) <= max_log_ratio.
    """
    for key in ("uniform", "smr"):
        if final_defects.get(key) is None or geo_mean_ratios.get(key) is None:
            pytest.fail(f"{label}Missing stored result for '{key}'")

    for key in ("uniform", "smr"):
        fd = final_defects[key]
        if fd > threshold:
            pytest.fail(
                f"{label}Final defect for {key} {fd:.3e} exceeds threshold "
                f"{threshold:.3e}"
            )

    r_uniform = geo_mean_ratios["uniform"]
    r_smr = geo_mean_ratios["smr"]
    if r_uniform <= 0 or r_smr <= 0:
        pytest.fail(
            f"{label}Non-positive geometric-mean ratio: "
            f"r_uniform={r_uniform}, r_smr={r_smr}"
        )
    log_ratio = abs(math.log10(r_smr / r_uniform))
    if log_ratio > max_log_ratio:
        pytest.fail(
            f"{label}Contraction-ratio mismatch: "
            f"abs(log10(r_smr/r_uniform))={log_ratio:.3f} exceeds "
            f"{max_log_ratio}. r_uniform={r_uniform:.4f}, r_smr={r_smr:.4f}"
        )
