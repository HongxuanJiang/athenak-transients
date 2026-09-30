"""Command-line entry point: ``python -m athenak_rt SNAPSHOT.bin [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .bands import DEFAULT_OBSERVATION_BANDS
from .config import (
    DEFAULT_MODE,
    DEFAULT_POPULATIONS,
    GRID_DTYPES,
    MODES,
    POPULATIONS,
    RTSettings,
)
from .opacity import DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T
from .snapshot import BoxSettings, snapshot_index_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m athenak_rt",
        description=(
            "Radiative post-processing of AthenaK TDE hydro snapshots: grey tau=1 "
            "photosphere, grey formal / thermalization-depth transfer, and "
            "multifrequency H/He continuum transfer along Cartesian rays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "snapshots",
        nargs="+",
        type=Path,
        metavar="SNAPSHOT.bin",
        help="AthenaK binary dump(s) with dens and eint",
    )
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument(
        "--direction",
        default="z",
        help="line of sight: x, y, z or -x, -y, -z (observer on that face)",
    )
    parser.add_argument(
        "--image-size", type=int, default=1024, help="pixels across the image plane"
    )
    parser.add_argument(
        "--los-steps", type=int, default=1024, help="samples along the line of sight"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("athenak_rt_output"), help="output directory"
    )

    tables = parser.add_argument_group("tables")
    tables.add_argument(
        "--eos-table",
        type=Path,
        default=None,
        help="AthenaK tabulated LTE EOS table (default: the <hydro>/table "
        "path recorded in the snapshot header, or its basename under "
        "$ATHENAK_EOS_TABLE_DIR / the repository's eos_tables/)",
    )
    tables.add_argument(
        "--populations",
        choices=POPULATIONS,
        default=DEFAULT_POPULATIONS,
        help="H/He ionization populations for grey-therm and multifreq: 'eos' "
        "takes them from the EOS table (falls back to Saha with a message if "
        "the table has no He/H2 fractions), 'saha' uses the built-in ideal Saha "
        "solver",
    )
    tables.add_argument(
        "--mesa-high-t",
        type=Path,
        default=DEFAULT_MESA_HIGH_T,
        help="MESA/OPAL high-T opacity table",
    )
    tables.add_argument(
        "--mesa-low-t",
        type=Path,
        default=DEFAULT_MESA_LOW_T,
        help="MESA low-T (Ferguson 2005) opacity table",
    )

    select = parser.add_argument_group("cell selection and geometry")
    select.add_argument(
        "--density-threshold-factor",
        type=float,
        default=10.0,
        help="cells with rho_code > factor * dfloor enter the ray grid",
    )
    select.add_argument(
        "--density-threshold-code",
        type=float,
        default=None,
        help="explicit threshold in code units (overrides the factor)",
    )
    select.add_argument(
        "--box",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="fixed ray-grid box in code units (default: padded "
        "bounding box of the selected cells)",
    )
    select.add_argument("--auto-box-padding-code", type=float, default=0.25)
    select.add_argument("--auto-box-padding-fraction", type=float, default=0.10)
    select.add_argument("--auto-box-min-width", type=float, default=1.0)
    select.add_argument(
        "--no-bh-mask",
        action="store_true",
        help="do not zero the black-hole excision sphere",
    )
    select.add_argument(
        "--bh-mask-radius",
        type=float,
        default=None,
        help="excision radius in code units (default: header value)",
    )

    transfer = parser.add_argument_group("transfer")
    transfer.add_argument(
        "--tau-photosphere",
        type=float,
        default=1.0,
        help="tau1 mode: optical depth of the photosphere",
    )
    transfer.add_argument(
        "--tau-stop",
        type=float,
        default=30.0,
        help="grey/multifreq modes: stop integrating past this depth",
    )
    transfer.add_argument(
        "--nfreq",
        type=int,
        default=73,
        help="multifreq: number of log-spaced photon energies",
    )
    transfer.add_argument(
        "--emin", type=float, default=0.1, help="multifreq: min energy [eV]"
    )
    transfer.add_argument(
        "--emax", type=float, default=1000.0, help="multifreq: max energy [eV]"
    )
    transfer.add_argument(
        "--no-scattering",
        action="store_true",
        help="multifreq: pure absorption instead of the "
        "thermalization-depth (scattering) treatment",
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--threads",
        type=int,
        default=None,
        help="numba threads (default: all CPUs available to the process)",
    )
    run.add_argument(
        "--grid-dtype",
        choices=GRID_DTYPES,
        default="auto",
        help="storage precision of the resampled (rho, T) cube; auto = "
        "float32 once any axis reaches 1024 samples, else float64",
    )
    run.add_argument(
        "--hdf5-compression",
        default="lzf",
        help="h5py compression filter for maps (lzf, gzip or none)",
    )
    run.add_argument(
        "--no-bands",
        action="store_true",
        help="skip the NIR/optical/UV/X-ray band products",
    )
    run.add_argument("--version", action="version", version=f"athenak_rt {__version__}")
    return parser


def settings_from_args(args: argparse.Namespace) -> RTSettings:
    return RTSettings(
        mode=args.mode,
        direction=args.direction,
        image_size=args.image_size,
        los_steps=args.los_steps,
        output_dir=args.out,
        box=BoxSettings(
            box=None if args.box is None else tuple(args.box),
            padding_code=args.auto_box_padding_code,
            padding_fraction=args.auto_box_padding_fraction,
            min_width_code=args.auto_box_min_width,
        ),
        eos_table=args.eos_table,
        populations=args.populations,
        mesa_high_t=args.mesa_high_t,
        mesa_low_t=args.mesa_low_t,
        density_threshold_factor=args.density_threshold_factor,
        density_threshold_code=args.density_threshold_code,
        bh_mask=not args.no_bh_mask,
        bh_mask_radius=args.bh_mask_radius,
        tau_photosphere=args.tau_photosphere,
        tau_stop=args.tau_stop,
        nfreq=args.nfreq,
        emin_ev=args.emin,
        emax_ev=args.emax,
        scattering=not args.no_scattering,
        grid_dtype=args.grid_dtype,
        observation_bands=() if args.no_bands else DEFAULT_OBSERVATION_BANDS,
        hdf5_compression=None
        if args.hdf5_compression in ("none", "")
        else args.hdf5_compression,
        threads=args.threads,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    settings.validate()
    snapshots = sorted(
        (Path(p).expanduser().resolve() for p in args.snapshots),
        key=snapshot_index_from_path,
    )
    for snapshot in snapshots:
        if not snapshot.is_file():
            print(f"error: snapshot not found: {snapshot}", file=sys.stderr)
            return 2

    from .pipeline import run

    results = run(settings, snapshots)
    for result in results:
        print(
            f"{result.snapshot.name} {settings.mode} {settings.direction}: "
            f"L_bol,iso = {result.luminosity:.10e} erg/s -> {result.hdf5_path}"
        )
    return 0
