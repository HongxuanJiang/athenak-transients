"""Command-line entry point: ``python -m athenak_rt PARAMETER_FILE``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .params import SCHEMA, ParameterError, run_parameter_file
from .pipeline import result_line


def build_parser() -> argparse.ArgumentParser:
    blocks = ", ".join(f"<{name}>" for name in SCHEMA)
    parser = argparse.ArgumentParser(
        prog="python -m athenak_rt",
        description=(
            "Radiative post-processing of AthenaK TDE hydro dumps: grey tau=1 "
            "photosphere, grey formal / thermalization-depth transfer, and "
            "multifrequency H/He continuum transfer along Cartesian rays. "
            "Every setting is read from the parameter file."
        ),
        epilog=(
            f"The parameter file has the AthenaK input format with the blocks {blocks}. "
            "See docs/athenak_rt.md and the examples in athenak_rt/examples/."
        ),
    )
    parser.add_argument(
        "parameter_file",
        type=Path,
        metavar="PARAMETER_FILE",
        help="parameter file, for example athenak_rt/examples/tde_snapshot.rtin",
    )
    parser.add_argument("--version", action="version", version=f"athenak_rt {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run_parameter_file(args.parameter_file)
    except ParameterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # missing EOS table, empty selection, reuse conflict
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(
            result_line(
                result.snapshot.name,
                result.mode,
                result.direction,
                result.luminosity,
                result.hdf5_path,
            )
        )
    return 0
