#!/usr/bin/env python3
"""ASCII-only TDE snapshot analysis for AthenaK hydro_w binary dumps.

Usage:
  python analyze_tde.py
  python analyze_tde.py /path/to/binfiles
  python analyze_tde.py /path/to/binfiles /path/to/output
  python analyze_tde.py /path/to/binfiles /path/to/output /path/to/config.json

The script processes one .bin snapshot at a time, computes BH-centered inertial
diagnostics, and writes plain-text .dat outputs plus README_ANALYSIS.md.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange, set_num_threads


SAFE_POSITIVE = 1.0e-300
HYDROGEN_MASS_CGS = 1.6735575e-24
K_BOLTZMANN_CGS = 1.380649e-16
GRAV_CONSTANT_CGS = 6.67408e-8
SNAPSHOT_RE = re.compile(r"(\d+)\.bin$")
TOL = 1.0e-12
ANALYSIS_FORMAT_VERSION = 5.3
LEGACY_RUN_LEVEL_OUTPUTS = (
    "units_summary.dat",
    "analysis_global_timeseries.dat",
    "analysis_cumulative_distributions.dat",
    "analysis_fallback_shell_timeseries.dat",
    "analysis_tail_budget_timeseries.dat",
    "analysis_branch_timeseries.dat",
)

SAHA_CACHE_TEMP = 0
SAHA_CACHE_PRESS = 1
SAHA_CACHE_CS2 = 2
SAHA_CACHE_GAMMA1 = 3
SAHA_CACHE_GAMMA3M1 = 4
SAHA_CACHE_XION = 5

DEFAULT_CONFIG = {
    "snapshot_selection": {
        "start": None,
        "stop": None,
        "stride": 1,
        "limit": None,
    },
    "resume_existing_outputs": True,
    "batch_blocks": None,
    "numba_threads": None,
    "checkpoint_stride": 8,
    "clean_output_dir": True,
    "sink_radius_code": 0.2,
    "sink_shell_cells": 1.0,
    "outer_shell_cells": 1.0,
    "control_radius_code": 1.0,
    "control_shell_cells": 1.0,
    "r_tide_code": None,
    "fallback_radius_multipliers": [0.5, 1.0, 1.2, 1.5],
    "fallback_shell_cells": 1.0,
    "inner_shell_radius_code": 1.0,
    "inner_shell_cells": 1.0,
    "outer_stream_radius_code": None,
    "outer_stream_radius_factor": 2.0,
    "outer_stream_shell_cells": 1.0,
    "compression_threshold_code": None,
    "compression_percentile": 95.0,
    "stellar_radius_code": None,
    "tail_energy_multiplier": 1.0,
    "tail_j_threshold_code": None,
    "tail_j_percentile": 25.0,
    "tail_e_threshold": 0.8,
    "fit_tail_alpha": True,
    "tail_fit_min_populated_bins": 8,
    "density_threshold_factor": 10.0,
    "ecrit": 0.2,
    "Rcrit_code": 1.0,
    "binning": {
        "energy": {"nbins": 640, "min_code": None, "max_code": None, "spacing": "symlog", "linthresh_code": None},
        "j": {"nbins": 480, "min_code": 0.0, "max_code": None, "spacing": "log"},
        "Be": {"nbins": 640, "min_code": None, "max_code": None, "spacing": "symlog", "linthresh_code": None},
        "e": {"nbins": 240, "min_code": 0.0, "max_code": 2.0, "spacing": "linear"},
    },
    "binning_2d": {
        "energy_j": {"nbins_x": 160, "nbins_y": 128},
        "energy_e": {"nbins_x": 160, "nbins_y": 128},
        "j_e": {"nbins_x": 128, "nbins_y": 128},
    },
}


@dataclass(frozen=True)
class UnitSystem:
    length_cgs: float
    mass_cgs: float
    time_cgs: float
    mu: float

    @property
    def velocity_cgs(self) -> float:
        return self.length_cgs / self.time_cgs

    @property
    def density_cgs(self) -> float:
        return self.mass_cgs / self.length_cgs**3

    @property
    def pressure_cgs(self) -> float:
        return self.mass_cgs / (self.length_cgs * self.time_cgs**2)

    @property
    def specific_eint_cgs(self) -> float:
        return self.velocity_cgs**2

    @property
    def angular_momentum_cgs(self) -> float:
        return self.length_cgs * self.velocity_cgs


@dataclass(frozen=True)
class SnapshotHeader:
    path: Path
    time_code: float
    cycle: int
    location_size: int
    variable_size: int
    variable_names: Tuple[str, ...]
    input_data: Dict[str, Dict[str, str]]
    data_offset: int
    location_format: str
    variable_dtype: np.dtype
    num_variables: int
    snapshot_index: int
    meshblock_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class SnapshotRuntime:
    bh_x_code: float
    bh_y_code: float
    bh_z_code: float
    bh_vx_code: float
    bh_vy_code: float
    bh_vz_code: float
    bh_mass_code: float
    newton_g_code: float
    bh_softening_code: float
    sink_radius_code: float
    frame_x_code: float
    frame_y_code: float
    frame_z_code: float
    frame_vx_code: float
    frame_vy_code: float
    frame_vz_code: float
    use_translating_frame: bool

    @property
    def gm_code(self) -> float:
        return self.newton_g_code * self.bh_mass_code

    @property
    def bh_inertial_x_code(self) -> float:
        return self.frame_x_code + self.bh_x_code

    @property
    def bh_inertial_y_code(self) -> float:
        return self.frame_y_code + self.bh_y_code

    @property
    def bh_inertial_z_code(self) -> float:
        return self.frame_z_code + self.bh_z_code


@dataclass(frozen=True)
class CachedSnapshotAnalysis:
    path: Path
    source_bin_name: str
    snapshot_index: int
    snapshot_time_s: float
    summary: Dict[str, float]
    histograms: Dict[Tuple[str, str], np.ndarray]


@dataclass(frozen=True)
class SahaEOSCache:
    table_path: Path
    density_unit_cgs: float
    pressure_unit_cgs: float
    specific_eint_unit_cgs: float
    temp_unit_cgs: float
    logrho: np.ndarray
    logtemp: np.ndarray
    table_logeps: np.ndarray
    thermo_cache: np.ndarray
    logeps_min: float
    logeps_max: float
    inv_dlogrho: float
    inv_dlogeps: float
    nrho: int
    neps: int

    @classmethod
    def from_snapshot(cls, input_data: Dict[str, Dict[str, str]], data_file: Path) -> "SahaEOSCache":
        hydro = input_data.get("hydro", {})
        units_block = input_data.get("units", {})
        if "table" not in hydro:
            raise RuntimeError("Native table EOS requires <hydro>/table in the dump metadata.")
        length_cgs = float(units_block["length_cgs"])
        mass_cgs = float(units_block["mass_cgs"])
        time_cgs = float(units_block["time_cgs"])
        density_unit_cgs = mass_cgs / length_cgs**3
        pressure_unit_cgs = mass_cgs / (length_cgs * time_cgs**2)
        specific_eint_unit_cgs = pressure_unit_cgs / density_unit_cgs
        temp_unit_cgs = (length_cgs / time_cgs) ** 2 * HYDROGEN_MASS_CGS / K_BOLTZMANN_CGS

        table_path = cls._resolve_table_path(hydro["table"], data_file)
        _, point_info, fields = cls._read_native_table(table_path)
        if list(point_info.keys()) != ["logrho", "logtemp"]:
            raise RuntimeError("Native EOS table axes must be ordered as logrho, logtemp.")
        required = ("logpress", "logeps", "logcs2", "gamma1", "gamma3m1", "xion")
        for name in required:
            if name not in fields:
                raise RuntimeError(f'Native EOS table is missing required field "{name}".')

        logrho = np.array(point_info["logrho"], dtype=np.float64, copy=True)
        logtemp = np.array(point_info["logtemp"], dtype=np.float64, copy=True)
        nrho = logrho.size
        ntemp = logtemp.size
        inv_dlogrho = 1.0 / cls._uniform_spacing(logrho, "logrho")
        cache_eps_factor = int(hydro.get("lte_cache_eps_factor", hydro.get("saha_cache_eps_factor", "8")))
        neps = cache_eps_factor * ntemp
        if neps < 2:
            raise RuntimeError("Native EOS cache needs at least two logeps points.")

        logpress = np.array(fields["logpress"], dtype=np.float64, copy=True)
        logeps = np.array(fields["logeps"], dtype=np.float64, copy=True)
        logcs2 = np.array(fields["logcs2"], dtype=np.float64, copy=True)
        gamma1 = np.array(fields["gamma1"], dtype=np.float64, copy=True)
        gamma3m1 = np.array(fields["gamma3m1"], dtype=np.float64, copy=True)
        xion = np.array(fields["xion"], dtype=np.float64, copy=True)

        logeps_min = float(np.min(logeps[:, 0]))
        logeps_max = float(np.max(logeps[:, -1]))
        if not (logeps_max > logeps_min):
            raise RuntimeError("Native EOS inverse cache has a degenerate logeps axis.")
        dlogeps = (logeps_max - logeps_min) / (neps - 1)
        inv_dlogeps = 1.0 / dlogeps

        thermo_cache = np.empty((6, nrho, neps), dtype=np.float64)
        for ir in range(nrho):
            row_logeps = logeps[ir]
            row_logpress = logpress[ir]
            row_logcs2 = logcs2[ir]
            row_gamma1 = gamma1[ir]
            row_gamma3m1 = gamma3m1[ir]
            row_xion = xion[ir]
            row_min = float(row_logeps[0])
            row_max = float(row_logeps[-1])
            targets = logeps_min + dlogeps * np.arange(neps, dtype=np.float64)
            targets = np.clip(targets, row_min, row_max)
            log_t = np.interp(targets, row_logeps, logtemp)
            thermo_cache[SAHA_CACHE_TEMP, ir] = np.exp(log_t) / temp_unit_cgs
            thermo_cache[SAHA_CACHE_PRESS, ir] = (
                np.exp(np.interp(log_t, logtemp, row_logpress)) / pressure_unit_cgs
            )
            thermo_cache[SAHA_CACHE_CS2, ir] = (
                np.exp(np.interp(log_t, logtemp, row_logcs2)) / specific_eint_unit_cgs
            )
            thermo_cache[SAHA_CACHE_GAMMA1, ir] = np.interp(log_t, logtemp, row_gamma1)
            thermo_cache[SAHA_CACHE_GAMMA3M1, ir] = np.interp(log_t, logtemp, row_gamma3m1)
            thermo_cache[SAHA_CACHE_XION, ir] = np.interp(log_t, logtemp, row_xion)

        return cls(
            table_path=table_path,
            density_unit_cgs=density_unit_cgs,
            pressure_unit_cgs=pressure_unit_cgs,
            specific_eint_unit_cgs=specific_eint_unit_cgs,
            temp_unit_cgs=temp_unit_cgs,
            logrho=logrho,
            logtemp=logtemp,
            table_logeps=logeps,
            thermo_cache=thermo_cache,
            logeps_min=logeps_min,
            logeps_max=logeps_max,
            inv_dlogrho=inv_dlogrho,
            inv_dlogeps=inv_dlogeps,
            nrho=nrho,
            neps=neps,
        )

    @staticmethod
    def _resolve_table_path(raw_path: str, data_file: Path) -> Path:
        raw = Path(raw_path).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([
                Path.cwd() / raw,
                data_file.resolve().parent / raw,
                Path(__file__).resolve().parent.parent.parent / raw,
                Path(__file__).resolve().parent.parent / raw,
            ])
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        raise RuntimeError(f'Unable to locate native EOS table "{raw_path}".')

    @staticmethod
    def _read_header_block(fp, name: str) -> List[str]:
        begin = f"<{name}begin>"
        end = f"<{name}end>"
        line = fp.readline().decode("ascii").strip()
        if line != begin:
            raise RuntimeError(f'Table header is missing block "{name}".')
        lines: List[str] = []
        while True:
            raw = fp.readline().decode("ascii")
            if not raw:
                raise RuntimeError(f'Unexpected EOF while reading "{name}" block.')
            stripped = raw.strip()
            if stripped == end:
                return lines
            lines.append(stripped)

    @classmethod
    def _read_native_table(cls, path: Path):
        with path.open("rb") as fp:
            metadata_lines = cls._read_header_block(fp, "metadata")
            scalar_lines = cls._read_header_block(fp, "scalars")
            point_lines = cls._read_header_block(fp, "points")
            field_lines = cls._read_header_block(fp, "fields")

            metadata = cls._parse_key_value_block(metadata_lines)
            scalars = cls._parse_key_value_block(scalar_lines, cast=float)
            point_counts = cls._parse_key_value_block(point_lines, cast=int)
            field_names = [line.strip() for line in field_lines if line.strip()]

            endianness = metadata.get("endianness", "little")
            if endianness not in ("little", "big"):
                raise RuntimeError(f'Unsupported endianness "{endianness}".')
            dtype = np.dtype("<f8" if endianness == "little" else ">f8")
            raw = np.frombuffer(fp.read(), dtype=dtype)

        offset = 0
        point_arrays = {}
        for name, npts in point_counts.items():
            point_arrays[name] = np.array(raw[offset:offset + npts], dtype=np.float64, copy=True)
            offset += npts

        npoints = 1
        for npts in point_counts.values():
            npoints *= npts

        fields = {}
        shape = tuple(point_counts.values())
        for name in field_names:
            values = raw[offset:offset + npoints]
            if values.size != npoints:
                raise RuntimeError(f'Table field "{name}" is truncated.')
            fields[name] = np.array(values, dtype=np.float64, copy=True).reshape(shape)
            offset += npoints

        metadata.update({key: f"{value}" for key, value in scalars.items()})
        return metadata, point_arrays, fields

    @staticmethod
    def _parse_key_value_block(lines: Sequence[str], cast=str) -> Dict[str, object]:
        parsed = {}
        for line in lines:
            if not line:
                continue
            if "=" not in line:
                raise RuntimeError(f'Invalid table header line "{line}".')
            key, value = line.split("=", 1)
            parsed[key.strip()] = cast(value.strip())
        return parsed

    @staticmethod
    def _uniform_spacing(values: np.ndarray, name: str) -> float:
        if values.size < 2:
            raise RuntimeError(f'Native EOS axis "{name}" must contain at least two points.')
        delta = float(values[1] - values[0])
        if delta <= 0.0:
            raise RuntimeError(f'Native EOS axis "{name}" must be strictly increasing.')
        tol = 1.0e-10 * max(1.0, abs(delta))
        if not np.all(np.abs(np.diff(values) - delta) <= tol):
            raise RuntimeError(f'Native EOS axis "{name}" must be uniformly spaced.')
        return delta


@dataclass
class BatchData:
    x1min: np.ndarray
    x1max: np.ndarray
    x2min: np.ndarray
    x2max: np.ndarray
    x3min: np.ndarray
    x3max: np.ndarray
    dx1: np.ndarray
    dx2: np.ndarray
    dx3: np.ndarray
    touch_xmin: np.ndarray
    touch_xmax: np.ndarray
    touch_ymin: np.ndarray
    touch_ymax: np.ndarray
    touch_zmin: np.ndarray
    touch_zmax: np.ndarray
    fields: Dict[str, np.ndarray]
    nblocks: int


@njit(cache=True, inline="always")
def _safe_positive_numba(x: float) -> float:
    if x > SAFE_POSITIVE:
        return x
    return SAFE_POSITIVE


@njit(cache=True, inline="always")
def _find_bin(edges: np.ndarray, value: float) -> int:
    n = edges.size - 1
    if n <= 0:
        return -1
    if value < edges[0] or value > edges[n]:
        return -1
    if value == edges[n]:
        return n - 1
    lo = 0
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if value < edges[mid]:
            hi = mid
        else:
            lo = mid
    return lo


SUBSET_INWARD = 0
SUBSET_OUTWARD = 1
SUBSET_BOUND_INWARD = 2
SUBSET_BOUND_OUTWARD = 3
SUBSET_UNBOUND = 4
NSUBSET_1D = 5

COMPRESSED_PROXY = 0
COMPRESSED_BOUND_INWARD = 1
NCOMPRESSED_SUBSET = 2

SHELL_INNER = 0
SHELL_OUTER_STREAM = 1

HIST2D_FULL = 0
HIST2D_BOUND_INWARD = 1
HIST2D_COMPRESSED = 2
HIST2D_BOUND_INWARD_COMPRESSED = 3
NHIST2D_SUBSET = 4


def format_multiplier_label(value: float) -> str:
    text = f"{value:.3f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return "f" + text.replace(".", "p").replace("-", "m")


def get_problem_scalar(input_data: Dict[str, Dict[str, str]], section: str, keys: Sequence[str]) -> Optional[float]:
    block = input_data.get(section, {})
    for key in keys:
        if key in block:
            return float(block[key])
    return None


def resolve_stellar_radius_code(header: SnapshotHeader, config: dict) -> float:
    configured = config.get("stellar_radius_code")
    if configured is not None:
        value = float(configured)
        if value > 0.0:
            return value
    value = get_problem_scalar(header.input_data, "problem", ("star_radius", "stellar_radius_code", "stellar_radius"))
    if value is not None and value > 0.0:
        return value
    raise RuntimeError(
        "Could not infer stellar_radius_code from snapshot metadata. Set config['stellar_radius_code'] explicitly."
    )


def resolve_tidal_radius_code(header: SnapshotHeader, runtime: SnapshotRuntime, config: dict) -> float:
    configured = config.get("r_tide_code")
    if configured is not None:
        value = float(configured)
        if value > 0.0:
            return value
    value = get_problem_scalar(header.input_data, "problem", ("r_tidal", "r_tide", "rtide", "tidal_radius"))
    if value is not None and value > 0.0:
        return value
    star_radius_code = get_problem_scalar(header.input_data, "problem", ("star_radius", "stellar_radius_code", "stellar_radius"))
    if star_radius_code is not None and star_radius_code > 0.0 and runtime.bh_mass_code > 0.0:
        return star_radius_code * runtime.bh_mass_code ** (1.0 / 3.0)
    raise RuntimeError(
        "Could not infer r_tide_code from snapshot metadata. Set config['r_tide_code'] explicitly."
    )


def derive_edges_from_bin_spec(name: str, spec: dict, header: SnapshotHeader, runtime: SnapshotRuntime) -> np.ndarray:
    merged = deep_update(DEFAULT_CONFIG["binning"][name], spec)
    config = {"binning": {name: merged}}
    return derive_bin_edges(name, config, header, runtime)


def derive_2d_edges(config: dict, header: SnapshotHeader, runtime: SnapshotRuntime) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pairs = {}
    specs = config.get("binning_2d", {})
    quantity_pairs = {
        "energy_j": ("energy", "j"),
        "energy_e": ("energy", "e"),
        "j_e": ("j", "e"),
    }
    for name, (xname, yname) in quantity_pairs.items():
        pair_spec = specs.get(name, {})
        xedges = derive_edges_from_bin_spec(
            xname,
            {"nbins": int(pair_spec.get("nbins_x", 128))},
            header,
            runtime,
        )
        yedges = derive_edges_from_bin_spec(
            yname,
            {"nbins": int(pair_spec.get("nbins_y", 128))},
            header,
            runtime,
        )
        pairs[name] = (xedges, yedges)
    return pairs


def hist_fraction_below(edges: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    total = float(np.sum(weights))
    if total <= 0.0 or threshold <= float(edges[0]):
        return 0.0
    if threshold >= float(edges[-1]):
        return 1.0
    accum = 0.0
    for idx, weight in enumerate(weights):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if threshold >= right:
            accum += float(weight)
            continue
        if threshold <= left:
            break
        width = right - left
        if width > 0.0:
            accum += float(weight) * (threshold - left) / width
        break
    return accum / total


def hist_fraction_above(edges: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    return 1.0 - hist_fraction_below(edges, weights, threshold)


def sum_hist2d_region(
    weights: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    x_upper: Optional[float] = None,
    y_upper: Optional[float] = None,
    y_lower: Optional[float] = None,
) -> float:
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    total = 0.0
    for ix, xcenter in enumerate(xcenters):
        if x_upper is not None and xcenter >= x_upper:
            continue
        for iy, ycenter in enumerate(ycenters):
            if y_upper is not None and ycenter >= y_upper:
                continue
            if y_lower is not None and ycenter <= y_lower:
                continue
            total += float(weights[ix, iy])
    return total


def fit_tail_alpha_from_hist(
    edges: np.ndarray,
    weights: np.ndarray,
    energy_limit_code: float,
    min_bins: int,
) -> Tuple[float, int, float, float, float]:
    widths = edges[1:] - edges[:-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = (
        (centers < min(energy_limit_code, 0.0))
        & (centers < 0.0)
        & (weights > 0.0)
        & (widths > 0.0)
    )
    if int(np.count_nonzero(valid)) < int(min_bins):
        return math.nan, 0, math.nan, math.nan, math.nan
    x = np.log(-centers[valid])
    y = np.log(weights[valid] / widths[valid])
    coeff = np.polyfit(x, y, 1)
    fit = coeff[0] * x + coeff[1]
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = math.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return -float(coeff[0]), int(x.size), float(np.min(centers[valid])), float(np.max(centers[valid])), r2


def resolve_tde_scales(header: SnapshotHeader, runtime: SnapshotRuntime, units: UnitSystem, config: dict) -> Dict[str, object]:
    r_tide_code = resolve_tidal_radius_code(header, runtime, config)
    stellar_radius_code = resolve_stellar_radius_code(header, config)
    a_mb_code = (r_tide_code * r_tide_code) / max(2.0 * stellar_radius_code, SAFE_POSITIVE)
    multipliers = np.array(config.get("fallback_radius_multipliers", [0.5, 1.0, 1.2, 1.5]), dtype=np.float64)
    if multipliers.size == 0:
        raise RuntimeError("fallback_radius_multipliers must contain at least one radius multiplier.")
    if np.any(multipliers <= 0.0):
        raise RuntimeError("fallback_radius_multipliers must all be positive.")
    shell_labels = [format_multiplier_label(float(value)) for value in multipliers]
    fallback_radii_code = multipliers * a_mb_code
    outer_stream_radius_cfg = config.get("outer_stream_radius_code")
    if outer_stream_radius_cfg is None:
        outer_stream_radius_code = float(config.get("outer_stream_radius_factor", 2.0)) * a_mb_code
    else:
        outer_stream_radius_code = float(outer_stream_radius_cfg)
    inner_shell_cfg = config.get("inner_shell_radius_code")
    if inner_shell_cfg is None:
        inner_shell_radius_code = float(config.get("control_radius_code", 1.0))
    else:
        inner_shell_radius_code = float(inner_shell_cfg)
    tail_energy_threshold_code = -float(config.get("tail_energy_multiplier", 1.0)) * (
        runtime.gm_code * stellar_radius_code / max(r_tide_code * r_tide_code, SAFE_POSITIVE)
    )
    return {
        "r_tide_code": float(r_tide_code),
        "stellar_radius_code": float(stellar_radius_code),
        "a_mb_code": float(a_mb_code),
        "fallback_radius_multipliers": multipliers,
        "fallback_labels": shell_labels,
        "fallback_radii_code": fallback_radii_code,
        "inner_shell_radius_code": float(inner_shell_radius_code),
        "outer_stream_radius_code": float(outer_stream_radius_code),
        "tail_energy_threshold_code": float(tail_energy_threshold_code),
        "tail_energy_threshold_cgs": float(tail_energy_threshold_code * units.specific_eint_cgs),
    }


@njit(cache=True, inline="always")
def _saha_pressure_from_rho_eint(
    dens_code: float,
    eint_density_code: float,
    density_unit_cgs: float,
    specific_eint_unit_cgs: float,
    logrho0: float,
    logrho_min: float,
    logrho_max: float,
    inv_dlogrho: float,
    table_logeps: np.ndarray,
    cache_press: np.ndarray,
    logeps_min: float,
    logeps_max: float,
    inv_dlogeps: float,
) -> float:
    rho_code = _safe_positive_numba(dens_code)
    log_rho = math.log(rho_code * density_unit_cgs)
    if log_rho < logrho_min:
        log_rho_clamped = logrho_min
    elif log_rho > logrho_max:
        log_rho_clamped = logrho_max
    else:
        log_rho_clamped = log_rho

    nrho = table_logeps.shape[0]
    neps = cache_press.shape[1]
    ir = int((log_rho_clamped - logrho0) * inv_dlogrho)
    if ir < 0:
        ir = 0
    elif ir > nrho - 2:
        ir = nrho - 2
    wr1 = (log_rho_clamped - (logrho0 + ir / inv_dlogrho)) * inv_dlogrho
    wr0 = 1.0 - wr1

    spec_eint_cgs = _safe_positive_numba(eint_density_code / rho_code) * specific_eint_unit_cgs
    log_eps = math.log(spec_eint_cgs)

    logeps_local_min = wr0 * table_logeps[ir, 0] + wr1 * table_logeps[ir + 1, 0]
    last_it = table_logeps.shape[1] - 1
    logeps_local_max = wr0 * table_logeps[ir, last_it] + wr1 * table_logeps[ir + 1, last_it]
    if log_eps < logeps_local_min:
        log_eps = logeps_local_min
    elif log_eps > logeps_local_max:
        log_eps = logeps_local_max
    if log_eps < logeps_min:
        log_eps = logeps_min
    elif log_eps > logeps_max:
        log_eps = logeps_max

    ie = int((log_eps - logeps_min) * inv_dlogeps)
    if ie < 0:
        ie = 0
    elif ie > neps - 2:
        ie = neps - 2
    dlogeps = 1.0 / inv_dlogeps
    log_eps_lo = logeps_min + ie * dlogeps
    we1 = (log_eps - log_eps_lo) * inv_dlogeps
    we0 = 1.0 - we1

    return (
        wr0 * (we0 * cache_press[ir, ie] + we1 * cache_press[ir, ie + 1])
        + wr1 * (we0 * cache_press[ir + 1, ie] + we1 * cache_press[ir + 1, ie + 1])
    )


@njit(cache=True, inline="always")
def _diff_x(field: np.ndarray, b: int, k: int, j: int, i: int, dx: float) -> float:
    nx = field.shape[3]
    if nx <= 1 or dx <= 0.0:
        return 0.0
    if i == 0:
        return (field[b, k, j, 1] - field[b, k, j, 0]) / dx
    if i == nx - 1:
        return (field[b, k, j, nx - 1] - field[b, k, j, nx - 2]) / dx
    return 0.5 * (field[b, k, j, i + 1] - field[b, k, j, i - 1]) / dx


@njit(cache=True, inline="always")
def _diff_y(field: np.ndarray, b: int, k: int, j: int, i: int, dx: float) -> float:
    ny = field.shape[2]
    if ny <= 1 or dx <= 0.0:
        return 0.0
    if j == 0:
        return (field[b, k, 1, i] - field[b, k, 0, i]) / dx
    if j == ny - 1:
        return (field[b, k, ny - 1, i] - field[b, k, ny - 2, i]) / dx
    return 0.5 * (field[b, k, j + 1, i] - field[b, k, j - 1, i]) / dx


@njit(cache=True, inline="always")
def _diff_z(field: np.ndarray, b: int, k: int, j: int, i: int, dx: float) -> float:
    nz = field.shape[1]
    if nz <= 1 or dx <= 0.0:
        return 0.0
    if k == 0:
        return (field[b, 1, j, i] - field[b, 0, j, i]) / dx
    if k == nz - 1:
        return (field[b, nz - 1, j, i] - field[b, nz - 2, j, i]) / dx
    return 0.5 * (field[b, k + 1, j, i] - field[b, k - 1, j, i]) / dx


@njit(cache=True, inline="always")
def _divergence_same_block(
    velx: np.ndarray,
    vely: np.ndarray,
    velz: np.ndarray,
    b: int,
    k: int,
    j: int,
    i: int,
    dx1: float,
    dx2: float,
    dx3: float,
) -> float:
    return (
        _diff_x(velx, b, k, j, i, dx1)
        + _diff_y(vely, b, k, j, i, dx2)
        + _diff_z(velz, b, k, j, i, dx3)
    )


@njit(cache=True, parallel=True)
def _collect_negative_divergence_batch(
    dens: np.ndarray,
    velx: np.ndarray,
    vely: np.ndarray,
    velz: np.ndarray,
    x1min: np.ndarray,
    x2min: np.ndarray,
    x3min: np.ndarray,
    dx1: np.ndarray,
    dx2: np.ndarray,
    dx3: np.ndarray,
    bhx: float,
    bhy: float,
    bhz: float,
    sink_radius: float,
    density_threshold_code: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nblocks, nz, ny, nx = dens.shape
    values = np.zeros((nblocks, nz * ny * nx), dtype=np.float64)
    counts = np.zeros(nblocks, dtype=np.int64)
    tiny = 1.0e-300

    for b in prange(nblocks):
        count = 0
        for k in range(nz):
            z = x3min[b] + (k + 0.5) * dx3[b]
            for j in range(ny):
                y = x2min[b] + (j + 0.5) * dx2[b]
                for i in range(nx):
                    rho = dens[b, k, j, i]
                    if rho <= 0.0 or rho <= density_threshold_code:
                        continue
                    x = x1min[b] + (i + 0.5) * dx1[b]
                    rx = x - bhx
                    ry = y - bhy
                    rz = z - bhz
                    r = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if sink_radius > 0.0 and r < sink_radius:
                        continue
                    divv = _divergence_same_block(velx, vely, velz, b, k, j, i, dx1[b], dx2[b], dx3[b])
                    if divv < -tiny:
                        values[b, count] = -divv
                        count += 1
        counts[b] = count

    return values, counts


@njit(cache=True)
def _process_batch(
    dens: np.ndarray,
    velx: np.ndarray,
    vely: np.ndarray,
    velz: np.ndarray,
    eint: np.ndarray,
    have_eint: bool,
    x1min: np.ndarray,
    x2min: np.ndarray,
    x3min: np.ndarray,
    dx1: np.ndarray,
    dx2: np.ndarray,
    dx3: np.ndarray,
    touch_xmin: np.ndarray,
    touch_xmax: np.ndarray,
    touch_ymin: np.ndarray,
    touch_ymax: np.ndarray,
    touch_zmin: np.ndarray,
    touch_zmax: np.ndarray,
    bhx: float,
    bhy: float,
    bhz: float,
    bhvx: float,
    bhvy: float,
    bhvz: float,
    gm_code: float,
    soft2: float,
    sink_radius: float,
    sink_shell_cells: float,
    outer_shell_cells: float,
    control_radius: float,
    control_shell_cells: float,
    ecrit: float,
    rcirc_crit: float,
    density_threshold_code: float,
    density_unit_cgs: float,
    specific_eint_unit_cgs: float,
    saha_logrho0: float,
    saha_logrho_min: float,
    saha_logrho_max: float,
    saha_inv_dlogrho: float,
    saha_table_logeps: np.ndarray,
    saha_cache_press: np.ndarray,
    saha_logeps_min: float,
    saha_logeps_max: float,
    saha_inv_dlogeps: float,
    edges_E: np.ndarray,
    edges_bound_E: np.ndarray,
    edges_j: np.ndarray,
    edges_Be: np.ndarray,
    edges_e: np.ndarray,
    edges_2d_energy_j_x: np.ndarray,
    edges_2d_energy_j_y: np.ndarray,
    edges_2d_energy_e_x: np.ndarray,
    edges_2d_energy_e_y: np.ndarray,
    edges_2d_j_e_x: np.ndarray,
    edges_2d_j_e_y: np.ndarray,
    fallback_radii: np.ndarray,
    fallback_shell_cells: float,
    inner_shell_radius: float,
    inner_shell_cells: float,
    outer_stream_radius: float,
    outer_stream_shell_cells: float,
    compression_threshold_code: float,
    tail_energy_threshold_code: float,
    tail_j_threshold_code: float,
    tail_e_threshold: float,
    have_tail_j_threshold: bool,
    need_be: bool,
) -> Tuple[np.ndarray, ...]:
    nblocks, nz, ny, nx = dens.shape
    nbE = edges_E.size - 1
    nbBoundE = edges_bound_E.size - 1
    nbJ = edges_j.size - 1
    nbBe = edges_Be.size - 1
    nbEcc = edges_e.size - 1
    nshell = fallback_radii.size
    nshell_local = nshell + 2
    nb2d_ej_x = edges_2d_energy_j_x.size - 1
    nb2d_ej_y = edges_2d_energy_j_y.size - 1
    nb2d_ee_x = edges_2d_energy_e_x.size - 1
    nb2d_ee_y = edges_2d_energy_e_y.size - 1
    nb2d_je_x = edges_2d_j_e_x.size - 1
    nb2d_je_y = edges_2d_j_e_y.size - 1

    m_domain = 0.0
    m_bound = 0.0
    m_unbound = 0.0
    m_ecrit = 0.0
    m_rcirc = 0.0
    m_bound_high_e = 0.0
    sum_j_bound = 0.0
    sum_eps_bound = 0.0
    m_tail_domain = 0.0
    m_tail_bound_inward = 0.0
    m_tail_lowj = 0.0
    m_tail_high_e = 0.0
    m_compressed_proxy = 0.0
    m_bound_inward_compressed_proxy = 0.0

    hist_E = np.zeros(nbE, dtype=np.float64)
    hist_j = np.zeros(nbJ, dtype=np.float64)
    hist_Be = np.zeros(nbBe, dtype=np.float64)
    hist_e = np.zeros(nbEcc, dtype=np.float64)

    bound_hist_E = np.zeros(nbBoundE, dtype=np.float64)
    bound_hist_j = np.zeros(nbJ, dtype=np.float64)

    sink_rate_E = np.zeros(nbE, dtype=np.float64)
    sink_rate_j = np.zeros(nbJ, dtype=np.float64)
    sink_rate_Be = np.zeros(nbBe, dtype=np.float64)
    outer_rate_E = np.zeros(nbE, dtype=np.float64)
    outer_rate_j = np.zeros(nbJ, dtype=np.float64)
    outer_rate_Be = np.zeros(nbBe, dtype=np.float64)

    subset_hist_E = np.zeros((NSUBSET_1D, nbE), dtype=np.float64)
    subset_hist_j = np.zeros((NSUBSET_1D, nbJ), dtype=np.float64)
    subset_hist_Be = np.zeros((NSUBSET_1D, nbBe), dtype=np.float64)
    subset_hist_e = np.zeros((NSUBSET_1D, nbEcc), dtype=np.float64)

    compressed_hist_E = np.zeros((NCOMPRESSED_SUBSET, nbE), dtype=np.float64)
    compressed_hist_j = np.zeros((NCOMPRESSED_SUBSET, nbJ), dtype=np.float64)
    compressed_hist_Be = np.zeros((NCOMPRESSED_SUBSET, nbBe), dtype=np.float64)
    compressed_hist_e = np.zeros((NCOMPRESSED_SUBSET, nbEcc), dtype=np.float64)

    shell_hist_E = np.zeros((nshell_local, nbE), dtype=np.float64)
    shell_hist_j = np.zeros((nshell_local, nbJ), dtype=np.float64)
    shell_hist_Be = np.zeros((nshell_local, nbBe), dtype=np.float64)
    shell_hist_e = np.zeros((nshell_local, nbEcc), dtype=np.float64)

    fallback_rate_in_E = np.zeros((nshell, nbE), dtype=np.float64)
    fallback_rate_in_j = np.zeros((nshell, nbJ), dtype=np.float64)
    fallback_rate_in_Be = np.zeros((nshell, nbBe), dtype=np.float64)
    fallback_rate_in_e = np.zeros((nshell, nbEcc), dtype=np.float64)
    fallback_rate_out_E = np.zeros((nshell, nbE), dtype=np.float64)
    fallback_rate_out_j = np.zeros((nshell, nbJ), dtype=np.float64)
    fallback_rate_out_Be = np.zeros((nshell, nbBe), dtype=np.float64)
    fallback_rate_out_e = np.zeros((nshell, nbEcc), dtype=np.float64)
    fallback_rate_net_E = np.zeros((nshell, nbE), dtype=np.float64)
    fallback_rate_net_j = np.zeros((nshell, nbJ), dtype=np.float64)
    fallback_rate_net_Be = np.zeros((nshell, nbBe), dtype=np.float64)
    fallback_rate_net_e = np.zeros((nshell, nbEcc), dtype=np.float64)

    fallback_rate_in_total = np.zeros(nshell, dtype=np.float64)
    fallback_rate_out_total = np.zeros(nshell, dtype=np.float64)
    fallback_rate_net_total = np.zeros(nshell, dtype=np.float64)

    hist2d_energy_j = np.zeros((NHIST2D_SUBSET, nb2d_ej_x, nb2d_ej_y), dtype=np.float64)
    hist2d_energy_e = np.zeros((NHIST2D_SUBSET, nb2d_ee_x, nb2d_ee_y), dtype=np.float64)
    hist2d_j_e = np.zeros((NHIST2D_SUBSET, nb2d_je_x, nb2d_je_y), dtype=np.float64)

    sink_rate_total = 0.0
    outer_rate_total = 0.0
    control_rate_total = 0.0

    gm2 = gm_code * gm_code
    tiny = 1.0e-300

    for b in range(nblocks):
        dv = dx1[b] * dx2[b] * dx3[b]
        radial_shell = sink_shell_cells * min(dx1[b], dx2[b], dx3[b])
        outer_shell = outer_shell_cells * min(dx1[b], dx2[b], dx3[b])
        control_shell = control_shell_cells * min(dx1[b], dx2[b], dx3[b])
        fallback_shell = fallback_shell_cells * min(dx1[b], dx2[b], dx3[b])
        inner_shell = inner_shell_cells * min(dx1[b], dx2[b], dx3[b])
        outer_stream_shell = outer_stream_shell_cells * min(dx1[b], dx2[b], dx3[b])
        area_x = dx2[b] * dx3[b]
        area_y = dx1[b] * dx3[b]
        area_z = dx1[b] * dx2[b]

        for k in range(nz):
            z = x3min[b] + (k + 0.5) * dx3[b]
            for j in range(ny):
                y = x2min[b] + (j + 0.5) * dx2[b]
                for i in range(nx):
                    rho = dens[b, k, j, i]
                    if rho <= 0.0 or rho <= density_threshold_code:
                        continue

                    x = x1min[b] + (i + 0.5) * dx1[b]
                    rx = x - bhx
                    ry = y - bhy
                    rz = z - bhz
                    vx_rel = velx[b, k, j, i] - bhvx
                    vy_rel = vely[b, k, j, i] - bhvy
                    vz_rel = velz[b, k, j, i] - bhvz

                    r2 = rx * rx + ry * ry + rz * rz
                    r = math.sqrt(r2)
                    if sink_radius > 0.0 and r < sink_radius:
                        continue

                    mass = rho * dv
                    v2 = vx_rel * vx_rel + vy_rel * vy_rel + vz_rel * vz_rel
                    if r > tiny:
                        vr = (rx * vx_rel + ry * vy_rel + rz * vz_rel) / r
                    else:
                        vr = 0.0

                    jx = ry * vz_rel - rz * vy_rel
                    jy = rz * vx_rel - rx * vz_rel
                    jz = rx * vy_rel - ry * vx_rel
                    j2 = jx * jx + jy * jy + jz * jz
                    jmag = math.sqrt(j2)
                    phi_bh = -gm_code / math.sqrt(r2 + soft2)
                    eps = 0.5 * v2 + phi_bh
                    rcirc = j2 / gm_code if gm_code > tiny else math.nan

                    have_valid_e = False
                    e_est = math.nan
                    expr = 1.0
                    if gm2 > tiny:
                        expr = 1.0 + 2.0 * eps * j2 / gm2
                        if expr >= 0.0 and eps <= 0.0:
                            e_est = math.sqrt(expr)
                            have_valid_e = True

                    be_val = math.nan
                    if need_be and have_eint:
                        press = _saha_pressure_from_rho_eint(
                            rho,
                            eint[b, k, j, i],
                            density_unit_cgs,
                            specific_eint_unit_cgs,
                            saha_logrho0,
                            saha_logrho_min,
                            saha_logrho_max,
                            saha_inv_dlogrho,
                            saha_table_logeps,
                            saha_cache_press,
                            saha_logeps_min,
                            saha_logeps_max,
                            saha_inv_dlogeps,
                        )
                        hspec = eint[b, k, j, i] / rho + press / rho
                        be_val = 0.5 * v2 + hspec + phi_bh

                    divv = _divergence_same_block(velx, vely, velz, b, k, j, i, dx1[b], dx2[b], dx3[b])
                    is_compressed_proxy = divv < 0.0 and (-divv) >= compression_threshold_code

                    is_bound = eps < 0.0
                    is_inward = vr < 0.0
                    is_outward = vr > 0.0
                    is_bound_inward = is_bound and is_inward
                    is_bound_outward = is_bound and is_outward
                    is_unbound = not is_bound

                    m_domain += mass
                    if is_bound:
                        m_bound += mass
                        sum_j_bound += mass * jmag
                        sum_eps_bound += mass * eps
                        ib = _find_bin(edges_bound_E, eps)
                        if ib >= 0:
                            bound_hist_E[ib] += mass
                        jb = _find_bin(edges_j, jmag)
                        if jb >= 0:
                            bound_hist_j[jb] += mass
                    else:
                        m_unbound += mass

                    if is_bound:
                        if have_valid_e and e_est < ecrit:
                            m_ecrit += mass
                        if not math.isnan(rcirc) and rcirc < rcirc_crit:
                            m_rcirc += mass
                        if have_valid_e and e_est > tail_e_threshold:
                            m_bound_high_e += mass

                    if eps < tail_energy_threshold_code:
                        m_tail_domain += mass
                        if is_bound_inward:
                            m_tail_bound_inward += mass
                        if have_valid_e and e_est > tail_e_threshold:
                            m_tail_high_e += mass
                        if have_tail_j_threshold and jmag < tail_j_threshold_code:
                            m_tail_lowj += mass

                    if is_compressed_proxy:
                        m_compressed_proxy += mass
                        if is_bound_inward:
                            m_bound_inward_compressed_proxy += mass

                    ib = _find_bin(edges_E, eps)
                    jb = _find_bin(edges_j, jmag)
                    kb = _find_bin(edges_Be, be_val) if need_be else -1
                    eb = _find_bin(edges_e, e_est) if have_valid_e else -1
                    if ib >= 0:
                        hist_E[ib] += mass
                    if jb >= 0:
                        hist_j[jb] += mass
                    if kb >= 0:
                        hist_Be[kb] += mass
                    if eb >= 0:
                        hist_e[eb] += mass

                    if is_inward:
                        if ib >= 0:
                            subset_hist_E[SUBSET_INWARD, ib] += mass
                        if jb >= 0:
                            subset_hist_j[SUBSET_INWARD, jb] += mass
                        if kb >= 0:
                            subset_hist_Be[SUBSET_INWARD, kb] += mass
                        if eb >= 0:
                            subset_hist_e[SUBSET_INWARD, eb] += mass
                    if is_outward:
                        if ib >= 0:
                            subset_hist_E[SUBSET_OUTWARD, ib] += mass
                        if jb >= 0:
                            subset_hist_j[SUBSET_OUTWARD, jb] += mass
                        if kb >= 0:
                            subset_hist_Be[SUBSET_OUTWARD, kb] += mass
                        if eb >= 0:
                            subset_hist_e[SUBSET_OUTWARD, eb] += mass
                    if is_bound_inward:
                        if ib >= 0:
                            subset_hist_E[SUBSET_BOUND_INWARD, ib] += mass
                        if jb >= 0:
                            subset_hist_j[SUBSET_BOUND_INWARD, jb] += mass
                        if kb >= 0:
                            subset_hist_Be[SUBSET_BOUND_INWARD, kb] += mass
                        if eb >= 0:
                            subset_hist_e[SUBSET_BOUND_INWARD, eb] += mass
                    if is_bound_outward:
                        if ib >= 0:
                            subset_hist_E[SUBSET_BOUND_OUTWARD, ib] += mass
                        if jb >= 0:
                            subset_hist_j[SUBSET_BOUND_OUTWARD, jb] += mass
                        if kb >= 0:
                            subset_hist_Be[SUBSET_BOUND_OUTWARD, kb] += mass
                        if eb >= 0:
                            subset_hist_e[SUBSET_BOUND_OUTWARD, eb] += mass
                    if is_unbound:
                        if ib >= 0:
                            subset_hist_E[SUBSET_UNBOUND, ib] += mass
                        if jb >= 0:
                            subset_hist_j[SUBSET_UNBOUND, jb] += mass
                        if kb >= 0:
                            subset_hist_Be[SUBSET_UNBOUND, kb] += mass
                        if eb >= 0:
                            subset_hist_e[SUBSET_UNBOUND, eb] += mass

                    if is_compressed_proxy:
                        if ib >= 0:
                            compressed_hist_E[COMPRESSED_PROXY, ib] += mass
                        if jb >= 0:
                            compressed_hist_j[COMPRESSED_PROXY, jb] += mass
                        if kb >= 0:
                            compressed_hist_Be[COMPRESSED_PROXY, kb] += mass
                        if eb >= 0:
                            compressed_hist_e[COMPRESSED_PROXY, eb] += mass
                        if is_bound_inward:
                            if ib >= 0:
                                compressed_hist_E[COMPRESSED_BOUND_INWARD, ib] += mass
                            if jb >= 0:
                                compressed_hist_j[COMPRESSED_BOUND_INWARD, jb] += mass
                            if kb >= 0:
                                compressed_hist_Be[COMPRESSED_BOUND_INWARD, kb] += mass
                            if eb >= 0:
                                compressed_hist_e[COMPRESSED_BOUND_INWARD, eb] += mass

                    if inner_shell > tiny and r >= inner_shell_radius and r < inner_shell_radius + inner_shell:
                        if ib >= 0:
                            shell_hist_E[0, ib] += mass
                        if jb >= 0:
                            shell_hist_j[0, jb] += mass
                        if kb >= 0:
                            shell_hist_Be[0, kb] += mass
                        if eb >= 0:
                            shell_hist_e[0, eb] += mass
                    if outer_stream_shell > tiny and r >= outer_stream_radius and r < outer_stream_radius + outer_stream_shell:
                        shell_index = nshell + 1
                        if ib >= 0:
                            shell_hist_E[shell_index, ib] += mass
                        if jb >= 0:
                            shell_hist_j[shell_index, jb] += mass
                        if kb >= 0:
                            shell_hist_Be[shell_index, kb] += mass
                        if eb >= 0:
                            shell_hist_e[shell_index, eb] += mass
                    if fallback_shell > tiny:
                        for s in range(nshell):
                            r_shell = fallback_radii[s]
                            if r >= r_shell and r < r_shell + fallback_shell:
                                shell_index = s + 1
                                if ib >= 0:
                                    shell_hist_E[shell_index, ib] += mass
                                if jb >= 0:
                                    shell_hist_j[shell_index, jb] += mass
                                if kb >= 0:
                                    shell_hist_Be[shell_index, kb] += mass
                                if eb >= 0:
                                    shell_hist_e[shell_index, eb] += mass
                                mdot_shell = mass * vr / fallback_shell
                                fallback_rate_net_total[s] += mdot_shell
                                if ib >= 0:
                                    fallback_rate_net_E[s, ib] += mdot_shell
                                if jb >= 0:
                                    fallback_rate_net_j[s, jb] += mdot_shell
                                if kb >= 0:
                                    fallback_rate_net_Be[s, kb] += mdot_shell
                                if eb >= 0:
                                    fallback_rate_net_e[s, eb] += mdot_shell
                                if vr < 0.0:
                                    mdot_in = mass * (-vr) / fallback_shell
                                    fallback_rate_in_total[s] += mdot_in
                                    if ib >= 0:
                                        fallback_rate_in_E[s, ib] += mdot_in
                                    if jb >= 0:
                                        fallback_rate_in_j[s, jb] += mdot_in
                                    if kb >= 0:
                                        fallback_rate_in_Be[s, kb] += mdot_in
                                    if eb >= 0:
                                        fallback_rate_in_e[s, eb] += mdot_in
                                elif vr > 0.0:
                                    mdot_out = mass * vr / fallback_shell
                                    fallback_rate_out_total[s] += mdot_out
                                    if ib >= 0:
                                        fallback_rate_out_E[s, ib] += mdot_out
                                    if jb >= 0:
                                        fallback_rate_out_j[s, jb] += mdot_out
                                    if kb >= 0:
                                        fallback_rate_out_Be[s, kb] += mdot_out
                                    if eb >= 0:
                                        fallback_rate_out_e[s, eb] += mdot_out

                    if sink_radius > 0.0 and radial_shell > tiny:
                        if r >= sink_radius and r < sink_radius + radial_shell and vr < 0.0:
                            mdot_sink = mass * (-vr) / radial_shell
                            sink_rate_total += mdot_sink
                            if ib >= 0:
                                sink_rate_E[ib] += mdot_sink
                            if jb >= 0:
                                sink_rate_j[jb] += mdot_sink
                            if need_be:
                                if kb >= 0:
                                    sink_rate_Be[kb] += mdot_sink

                    if control_radius > 0.0 and control_shell > tiny:
                        if r >= control_radius and r < control_radius + control_shell and vr < 0.0:
                            control_rate_total += mass * (-vr) / control_shell

                    mdot_outer = 0.0
                    if outer_shell > tiny:
                        if touch_xmin[b] and i == 0 and velx[b, k, j, i] < 0.0:
                            mdot_outer += rho * (-velx[b, k, j, i]) * area_x
                        if touch_xmax[b] and i == nx - 1 and velx[b, k, j, i] > 0.0:
                            mdot_outer += rho * velx[b, k, j, i] * area_x
                        if touch_ymin[b] and j == 0 and vely[b, k, j, i] < 0.0:
                            mdot_outer += rho * (-vely[b, k, j, i]) * area_y
                        if touch_ymax[b] and j == ny - 1 and vely[b, k, j, i] > 0.0:
                            mdot_outer += rho * vely[b, k, j, i] * area_y
                        if touch_zmin[b] and k == 0 and velz[b, k, j, i] < 0.0:
                            mdot_outer += rho * (-velz[b, k, j, i]) * area_z
                        if touch_zmax[b] and k == nz - 1 and velz[b, k, j, i] > 0.0:
                            mdot_outer += rho * velz[b, k, j, i] * area_z

                    if mdot_outer > 0.0:
                        outer_rate_total += mdot_outer
                        if ib >= 0:
                            outer_rate_E[ib] += mdot_outer
                        if jb >= 0:
                            outer_rate_j[jb] += mdot_outer
                        if need_be:
                            if kb >= 0:
                                outer_rate_Be[kb] += mdot_outer

                    ejx = _find_bin(edges_2d_energy_j_x, eps)
                    ejy = _find_bin(edges_2d_energy_j_y, jmag)
                    eex = _find_bin(edges_2d_energy_e_x, eps)
                    eey = _find_bin(edges_2d_energy_e_y, e_est) if have_valid_e else -1
                    jex = _find_bin(edges_2d_j_e_x, jmag)
                    jey = _find_bin(edges_2d_j_e_y, e_est) if have_valid_e else -1

                    if ejx >= 0 and ejy >= 0:
                        hist2d_energy_j[HIST2D_FULL, ejx, ejy] += mass
                        if is_bound_inward:
                            hist2d_energy_j[HIST2D_BOUND_INWARD, ejx, ejy] += mass
                        if is_compressed_proxy:
                            hist2d_energy_j[HIST2D_COMPRESSED, ejx, ejy] += mass
                            if is_bound_inward:
                                hist2d_energy_j[HIST2D_BOUND_INWARD_COMPRESSED, ejx, ejy] += mass
                    if eex >= 0 and eey >= 0:
                        hist2d_energy_e[HIST2D_FULL, eex, eey] += mass
                        if is_bound_inward:
                            hist2d_energy_e[HIST2D_BOUND_INWARD, eex, eey] += mass
                        if is_compressed_proxy:
                            hist2d_energy_e[HIST2D_COMPRESSED, eex, eey] += mass
                            if is_bound_inward:
                                hist2d_energy_e[HIST2D_BOUND_INWARD_COMPRESSED, eex, eey] += mass
                    if jex >= 0 and jey >= 0:
                        hist2d_j_e[HIST2D_FULL, jex, jey] += mass
                        if is_bound_inward:
                            hist2d_j_e[HIST2D_BOUND_INWARD, jex, jey] += mass
                        if is_compressed_proxy:
                            hist2d_j_e[HIST2D_COMPRESSED, jex, jey] += mass
                            if is_bound_inward:
                                hist2d_j_e[HIST2D_BOUND_INWARD_COMPRESSED, jex, jey] += mass

    return (
        m_domain,
        m_bound,
        m_unbound,
        m_ecrit,
        m_rcirc,
        m_bound_high_e,
        sum_j_bound,
        sum_eps_bound,
        m_tail_domain,
        m_tail_bound_inward,
        m_tail_lowj,
        m_tail_high_e,
        m_compressed_proxy,
        m_bound_inward_compressed_proxy,
        hist_E,
        hist_j,
        hist_Be,
        hist_e,
        bound_hist_E,
        bound_hist_j,
        subset_hist_E,
        subset_hist_j,
        subset_hist_Be,
        subset_hist_e,
        compressed_hist_E,
        compressed_hist_j,
        compressed_hist_Be,
        compressed_hist_e,
        shell_hist_E,
        shell_hist_j,
        shell_hist_Be,
        shell_hist_e,
        fallback_rate_in_E,
        fallback_rate_in_j,
        fallback_rate_in_Be,
        fallback_rate_in_e,
        fallback_rate_out_E,
        fallback_rate_out_j,
        fallback_rate_out_Be,
        fallback_rate_out_e,
        fallback_rate_net_E,
        fallback_rate_net_j,
        fallback_rate_net_Be,
        fallback_rate_net_e,
        fallback_rate_in_total,
        fallback_rate_out_total,
        fallback_rate_net_total,
        hist2d_energy_j,
        hist2d_energy_e,
        hist2d_j_e,
        sink_rate_E,
        sink_rate_j,
        sink_rate_Be,
        outer_rate_E,
        outer_rate_j,
        outer_rate_Be,
        sink_rate_total,
        outer_rate_total,
        control_rate_total,
    )


def deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def snapshot_index_from_path(path: Path) -> int:
    match = SNAPSHOT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_snapshot_header(path: Path) -> SnapshotHeader:
    with path.open("rb") as fp:
        version = fp.readline().decode("ascii")
        if version != "Athena binary output version=1.1\n":
            raise RuntimeError(f"Unsupported binary format in {path}.")

        _ = fp.readline().decode("ascii")
        time_line = fp.readline().decode("ascii").strip()
        cycle_line = fp.readline().decode("ascii").strip()

        line = fp.readline().decode("ascii")
        if not line.startswith("  size of location="):
            raise RuntimeError(f"Could not read location size in {path}.")
        location_size = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  size of variable="):
            raise RuntimeError(f"Could not read variable size in {path}.")
        variable_size = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  number of variables="):
            raise RuntimeError(f"Could not read number of variables in {path}.")
        num_variables = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  variables:"):
            raise RuntimeError(f"Could not read variable list in {path}.")
        variable_names = tuple(line[12:].split())

        line = fp.readline().decode("ascii")
        if not line.startswith("  header offset="):
            raise RuntimeError(f"Could not read header offset in {path}.")
        header_offset = int(line.split("=", 1)[1].strip())

        input_data: Dict[str, Dict[str, str]] = {}
        section_name: Optional[str] = None
        start_of_data = fp.tell() + header_offset
        while fp.tell() < start_of_data:
            raw = fp.readline().decode("ascii")
            if not raw:
                break
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("<") and stripped.endswith(">"):
                section_name = stripped[1:-1]
                input_data[section_name] = {}
                continue
            if section_name is None:
                continue
            key, value = raw.split("=", 1)
            input_data[section_name][key.strip()] = value.split("#", 1)[0].strip()

    time_code = float(time_line.split("=", 1)[1].strip())
    cycle = int(cycle_line.split("=", 1)[1].strip())
    meshblock = input_data.get("meshblock", {})
    nx1 = int(meshblock.get("nx1", input_data.get("mesh", {}).get("nx1", "1")))
    nx2 = int(meshblock.get("nx2", input_data.get("mesh", {}).get("nx2", "1")))
    nx3 = int(meshblock.get("nx3", input_data.get("mesh", {}).get("nx3", "1")))
    location_format = "f" if location_size == 4 else "d"
    variable_dtype = np.float32 if variable_size == 4 else np.float64
    return SnapshotHeader(
        path=path,
        time_code=time_code,
        cycle=cycle,
        location_size=location_size,
        variable_size=variable_size,
        variable_names=variable_names,
        input_data=input_data,
        data_offset=start_of_data,
        location_format=location_format,
        variable_dtype=variable_dtype,
        num_variables=num_variables,
        snapshot_index=snapshot_index_from_path(path),
        meshblock_shape=(nx3, nx2, nx1),
    )


def build_units(input_data: Dict[str, Dict[str, str]]) -> UnitSystem:
    units_block = input_data.get("units", {})
    return UnitSystem(
        length_cgs=float(units_block["length_cgs"]),
        mass_cgs=float(units_block["mass_cgs"]),
        time_cgs=float(units_block["time_cgs"]),
        mu=float(units_block.get("mu", "1.0")),
    )


def build_runtime(header: SnapshotHeader, config: dict) -> SnapshotRuntime:
    problem = header.input_data.get("problem", {})
    gravity = header.input_data.get("gravity", {})
    bh_x = float(problem.get("bh_live_x", "0.0"))
    bh_y = float(problem.get("bh_live_y", "0.0"))
    bh_z = float(problem.get("bh_live_z", "0.0"))
    bh_vx = float(problem.get("bh_live_vx", "0.0"))
    bh_vy = float(problem.get("bh_live_vy", "0.0"))
    bh_vz = float(problem.get("bh_live_vz", "0.0"))
    frame_x = float(problem.get("frame_live_x", "0.0"))
    frame_y = float(problem.get("frame_live_y", "0.0"))
    frame_z = float(problem.get("frame_live_z", "0.0"))
    frame_vx = float(problem.get("frame_live_vx", "0.0"))
    frame_vy = float(problem.get("frame_live_vy", "0.0"))
    frame_vz = float(problem.get("frame_live_vz", "0.0"))
    bh_mass = float(problem.get("external_bh_mass", problem.get("mass_ratio", "0.0")))
    if bh_mass <= 0.0:
        raise RuntimeError(f"Could not infer BH mass from {header.path}.")
    if "external_newton_g" in problem:
        newton_g = float(problem["external_newton_g"])
    else:
        four_pi_g = float(gravity.get("four_pi_G", "1.0"))
        newton_g = four_pi_g / (4.0 * math.pi)
    bh_soft = float(problem.get("external_bh_softening", problem.get("bh_softening", "0.0")))
    sink_radius = float(config.get("sink_radius_code", problem.get("bh_excise_radius", "0.2")))
    return SnapshotRuntime(
        bh_x_code=bh_x,
        bh_y_code=bh_y,
        bh_z_code=bh_z,
        bh_vx_code=bh_vx,
        bh_vy_code=bh_vy,
        bh_vz_code=bh_vz,
        bh_mass_code=bh_mass,
        newton_g_code=newton_g,
        bh_softening_code=bh_soft,
        sink_radius_code=sink_radius,
        frame_x_code=frame_x,
        frame_y_code=frame_y,
        frame_z_code=frame_z,
        frame_vx_code=frame_vx,
        frame_vy_code=frame_vy,
        frame_vz_code=frame_vz,
        use_translating_frame=is_true(problem.get("use_translating_frame", "false")),
    )


def select_snapshot_files(input_dir: Path, config: dict) -> List[Path]:
    files = list(input_dir.glob("*.bin"))
    files = [path for path in files if SNAPSHOT_RE.search(path.name)]
    if not files:
        raise RuntimeError(f"No snapshot .bin files found in {input_dir}.")
    files.sort(key=snapshot_index_from_path)

    selection = config["snapshot_selection"]
    start = selection.get("start")
    stop = selection.get("stop")
    stride = max(int(selection.get("stride", 1)), 1)
    limit = selection.get("limit")

    filtered: List[Path] = []
    for path in files:
        idx = snapshot_index_from_path(path)
        if start is not None and idx < int(start):
            continue
        if stop is not None and idx > int(stop):
            continue
        filtered.append(path)
    filtered = filtered[::stride]
    if limit is not None:
        filtered = filtered[: int(limit)]
    if not filtered:
        raise RuntimeError("Snapshot selection removed every file.")
    return filtered


def clean_output_dir(output_dir: Path, keep_snapshot_cache: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.dat"):
        if keep_snapshot_cache and path.name.endswith(".analysis.dat"):
            continue
        if path.is_file():
            path.unlink()
    for name in ("README_ANALYSIS.md",):
        path = output_dir / name
        if path.exists():
            path.unlink()


def remove_legacy_run_level_outputs(output_dir: Path) -> None:
    for name in LEGACY_RUN_LEVEL_OUTPUTS:
        path = output_dir / name
        if path.exists():
            path.unlink()


def iter_snapshot_batches(
    header: SnapshotHeader,
    requested_variables: Sequence[str],
    batch_blocks: int,
) -> Iterator[BatchData]:
    requested = tuple(name for name in requested_variables if name in header.variable_names)
    nz, ny, nx = header.meshblock_shape
    mesh = header.input_data.get("mesh", {})
    x1mesh_min = float(mesh.get("x1min", "0.0"))
    x1mesh_max = float(mesh.get("x1max", "0.0"))
    x2mesh_min = float(mesh.get("x2min", "0.0"))
    x2mesh_max = float(mesh.get("x2max", "0.0"))
    x3mesh_min = float(mesh.get("x3min", "0.0"))
    x3mesh_max = float(mesh.get("x3max", "0.0"))
    variable_bytes = nx * ny * nz * header.variable_size

    with header.path.open("rb") as fp:
        fp.seek(0, 2)
        file_size = fp.tell()
        fp.seek(header.data_offset, 0)

        while fp.tell() < file_size:
            x1min = np.empty(batch_blocks, dtype=np.float64)
            x1max = np.empty(batch_blocks, dtype=np.float64)
            x2min = np.empty(batch_blocks, dtype=np.float64)
            x2max = np.empty(batch_blocks, dtype=np.float64)
            x3min = np.empty(batch_blocks, dtype=np.float64)
            x3max = np.empty(batch_blocks, dtype=np.float64)
            dx1 = np.empty(batch_blocks, dtype=np.float64)
            dx2 = np.empty(batch_blocks, dtype=np.float64)
            dx3 = np.empty(batch_blocks, dtype=np.float64)
            touch_xmin = np.zeros(batch_blocks, dtype=np.bool_)
            touch_xmax = np.zeros(batch_blocks, dtype=np.bool_)
            touch_ymin = np.zeros(batch_blocks, dtype=np.bool_)
            touch_ymax = np.zeros(batch_blocks, dtype=np.bool_)
            touch_zmin = np.zeros(batch_blocks, dtype=np.bool_)
            touch_zmax = np.zeros(batch_blocks, dtype=np.bool_)
            fields = {
                name: np.empty((batch_blocks, nz, ny, nx), dtype=np.float64) for name in requested
            }

            count = 0
            while count < batch_blocks and fp.tell() < file_size:
                raw = fp.read(24)
                if not raw:
                    break
                if len(raw) != 24:
                    raise RuntimeError(f"Truncated MeshBlock index record in {header.path}.")
                ois, oie, ojs, oje, oks, oke = struct.unpack("@6i", raw)
                block_nx = oie - ois + 1
                block_ny = oje - ojs + 1
                block_nz = oke - oks + 1
                if (block_nz, block_ny, block_nx) != header.meshblock_shape:
                    raise RuntimeError(
                        f"{header.path} contains a MeshBlock with shape "
                        f"{(block_nz, block_ny, block_nx)}, expected {header.meshblock_shape}."
                    )

                fp.seek(16, 1)
                lims_raw = fp.read(6 * header.location_size)
                if len(lims_raw) != 6 * header.location_size:
                    raise RuntimeError(f"Truncated MeshBlock limits in {header.path}.")
                lims = struct.unpack("=" + 6 * header.location_format, lims_raw)
                x1min[count], x1max[count], x2min[count], x2max[count], x3min[count], x3max[count] = lims
                dx1[count] = (x1max[count] - x1min[count]) / block_nx
                dx2[count] = (x2max[count] - x2min[count]) / block_ny
                dx3[count] = (x3max[count] - x3min[count]) / block_nz
                tol1 = max(1.0e-12, abs(dx1[count]) * 1.0e-12)
                tol2 = max(1.0e-12, abs(dx2[count]) * 1.0e-12)
                tol3 = max(1.0e-12, abs(dx3[count]) * 1.0e-12)
                touch_xmin[count] = abs(x1min[count] - x1mesh_min) <= tol1
                touch_xmax[count] = abs(x1max[count] - x1mesh_max) <= tol1
                touch_ymin[count] = abs(x2min[count] - x2mesh_min) <= tol2
                touch_ymax[count] = abs(x2max[count] - x2mesh_max) <= tol2
                touch_zmin[count] = abs(x3min[count] - x3mesh_min) <= tol3
                touch_zmax[count] = abs(x3max[count] - x3mesh_max) <= tol3

                for name in header.variable_names:
                    if name in requested:
                        payload = fp.read(variable_bytes)
                        if len(payload) != variable_bytes:
                            raise RuntimeError(f"Truncated variable payload in {header.path}.")
                        fields[name][count] = np.frombuffer(
                            payload, dtype=header.variable_dtype
                        ).astype(np.float64).reshape(header.meshblock_shape)
                    else:
                        fp.seek(variable_bytes, 1)
                count += 1

            if count == 0:
                break

            yield BatchData(
                x1min=x1min[:count].copy(),
                x1max=x1max[:count].copy(),
                x2min=x2min[:count].copy(),
                x2max=x2max[:count].copy(),
                x3min=x3min[:count].copy(),
                x3max=x3max[:count].copy(),
                dx1=dx1[:count].copy(),
                dx2=dx2[:count].copy(),
                dx3=dx3[:count].copy(),
                touch_xmin=touch_xmin[:count].copy(),
                touch_xmax=touch_xmax[:count].copy(),
                touch_ymin=touch_ymin[:count].copy(),
                touch_ymax=touch_ymax[:count].copy(),
                touch_zmin=touch_zmin[:count].copy(),
                touch_zmax=touch_zmax[:count].copy(),
                fields={name: array[:count].copy() for name, array in fields.items()},
                nblocks=count,
            )


def sum_histograms(arrays: Iterable[np.ndarray]) -> np.ndarray:
    total: Optional[np.ndarray] = None
    for array in arrays:
        if total is None:
            total = np.array(array, dtype=np.float64, copy=True)
        else:
            total += array
    if total is None:
        return np.zeros(0, dtype=np.float64)
    return total


def _make_symlog_edges(min_code: float, max_code: float, nbins: int, linthresh_code: float) -> np.ndarray:
    if not (max_code > min_code):
        raise RuntimeError(f"Invalid symlog bin range: {min_code} .. {max_code}")
    if linthresh_code <= 0.0:
        raise RuntimeError(f"Invalid symlog linthresh: {linthresh_code}")
    transform_min = math.asinh(min_code / linthresh_code)
    transform_max = math.asinh(max_code / linthresh_code)
    transformed = np.linspace(transform_min, transform_max, nbins + 1, dtype=np.float64)
    return linthresh_code * np.sinh(transformed)


def derive_bin_edges(name: str, config: dict, header: SnapshotHeader, runtime: SnapshotRuntime) -> np.ndarray:
    spec = config["binning"][name]
    nbins = int(spec["nbins"])
    min_code = spec.get("min_code")
    max_code = spec.get("max_code")
    if min_code is None or max_code is None:
        mesh = header.input_data.get("mesh", {})
        vmax = float(header.input_data.get("hydro", {}).get("vceil", "10.0"))
        xmax = max(abs(float(mesh.get("x1min", "0.0"))), abs(float(mesh.get("x1max", "0.0"))))
        ymax = max(abs(float(mesh.get("x2min", "0.0"))), abs(float(mesh.get("x2max", "0.0"))))
        zmax = max(abs(float(mesh.get("x3min", "0.0"))), abs(float(mesh.get("x3max", "0.0"))))
        rmax = math.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)
        gm = runtime.gm_code
        sink = max(runtime.sink_radius_code, runtime.bh_softening_code, 1.0e-3)
        if name == "energy":
            if min_code is None:
                min_code = -2.0 * gm / sink
            if max_code is None:
                max_code = 2.0 * (0.5 * vmax * vmax + gm / max(rmax, 1.0))
        elif name == "j":
            if min_code is None:
                min_code = 0.0
            if max_code is None:
                max_code = 2.0 * max(rmax, 1.0) * vmax
        elif name == "Be":
            if min_code is None:
                min_code = -2.0 * gm / sink
            if max_code is None:
                max_code = 4.0 * (0.5 * vmax * vmax + gm / max(rmax, 1.0))
        elif name == "e":
            if min_code is None:
                min_code = 0.0
            if max_code is None:
                max_code = 2.0
        else:
            raise RuntimeError(f"Unknown bin family {name}.")
    if not (max_code > min_code):
        raise RuntimeError(f"Invalid bin range for {name}: {min_code} .. {max_code}")
    spacing = str(spec.get("spacing", "linear")).strip().lower()
    if spacing == "linear":
        return np.linspace(float(min_code), float(max_code), nbins + 1, dtype=np.float64)
    if spacing == "log":
        if min_code <= 0.0:
            min_code = max(float(max_code) * 1.0e-6, 1.0e-30)
        return np.geomspace(float(min_code), float(max_code), nbins + 1, dtype=np.float64)
    if spacing == "symlog":
        linthresh_code = spec.get("linthresh_code")
        if linthresh_code is None:
            dynamic_scale = max(abs(float(min_code)), abs(float(max_code)), SAFE_POSITIVE)
            linthresh_code = max(dynamic_scale * 1.0e-3, abs(float(min_code)) / max(nbins, 1), 1.0e-30)
        return _make_symlog_edges(float(min_code), float(max_code), nbins, float(linthresh_code))
    raise RuntimeError(f"Unknown spacing='{spacing}' for {name}.")


def hist_median_from_edges(edges: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    target = 0.5 * total
    cumsum = np.cumsum(weights, dtype=np.float64)
    idx = int(np.searchsorted(cumsum, target, side="left"))
    idx = max(0, min(idx, weights.size - 1))
    left = edges[idx]
    right = edges[idx + 1]
    prev = cumsum[idx - 1] if idx > 0 else 0.0
    width = right - left
    if weights[idx] <= 0.0 or width <= 0.0:
        return 0.5 * (left + right)
    frac = (target - prev) / weights[idx]
    frac = min(max(frac, 0.0), 1.0)
    return left + frac * width


def hist_quantile_from_edges(edges: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    target = min(max(float(percentile), 0.0), 100.0) * 0.01 * total
    cumsum = np.cumsum(weights, dtype=np.float64)
    idx = int(np.searchsorted(cumsum, target, side="left"))
    idx = max(0, min(idx, weights.size - 1))
    left = edges[idx]
    right = edges[idx + 1]
    prev = cumsum[idx - 1] if idx > 0 else 0.0
    width = right - left
    if weights[idx] <= 0.0 or width <= 0.0:
        return 0.5 * (left + right)
    frac = (target - prev) / weights[idx]
    frac = min(max(frac, 0.0), 1.0)
    return left + frac * width


def convert_edges_to_physical(name: str, edges_code: np.ndarray, units: UnitSystem) -> np.ndarray:
    if name in {"energy", "Be"}:
        return edges_code * units.specific_eint_cgs
    if name == "j":
        return edges_code * units.angular_momentum_cgs
    if name == "e":
        return edges_code.copy()
    raise RuntimeError(f"Unknown edge family {name}.")


def write_histogram_dat(
    path: Path,
    edges: np.ndarray,
    weights: np.ndarray,
    weight_unit: str,
    axis_unit: str,
    quantity_name: str,
    note: str,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    dmdx = np.zeros_like(weights)
    valid = widths > 0.0
    dmdx[valid] = weights[valid] / widths[valid]
    cumulative = np.cumsum(weights, dtype=np.float64)
    with path.open("w", encoding="ascii") as fp:
        fp.write(f"# quantity = {quantity_name}\n")
        fp.write("# columns = bin_left bin_right bin_center dM dM_dX cumulative_M\n")
        fp.write(f"# units = {axis_unit} {axis_unit} {axis_unit} {weight_unit} {weight_unit}/({axis_unit}) {weight_unit}\n")
        fp.write(f"# note = {note}\n")
        for left, right, center, dm, dmdx_val, cm in zip(edges[:-1], edges[1:], centers, weights, dmdx, cumulative):
            fp.write(f"{left:.16e} {right:.16e} {center:.16e} {dm:.16e} {dmdx_val:.16e} {cm:.16e}\n")


def write_table(path: Path, columns: Sequence[str], units_line: Sequence[str], rows: Sequence[Sequence[float]], note: str) -> None:
    with path.open("w", encoding="ascii") as fp:
        fp.write("# columns = " + " ".join(columns) + "\n")
        fp.write("# units = " + " ".join(units_line) + "\n")
        fp.write(f"# note = {note}\n")
        for row in rows:
            fields = []
            for value in row:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    fields.append("nan")
                else:
                    fields.append(f"{float(value):.16e}")
            fp.write(" ".join(fields) + "\n")


def _format_scalar_value(value: object) -> str:
    if value is None:
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return f"{float(value):.16e}"
    return str(value)


def write_snapshot_analysis(
    path: Path,
    source_bin_name: str,
    snapshot_index: int,
    snapshot_time_s: float,
    summary_rows: Sequence[Tuple[str, object, str]],
    histogram_sections: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
    hist2d_sections: Optional[Sequence[Tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> None:
    with path.open("w", encoding="ascii") as fp:
        fp.write("# file_type = snapshot_analysis\n")
        fp.write(f"# source_bin = {source_bin_name}\n")
        fp.write(f"# snapshot_index = {snapshot_index}\n")
        fp.write(f"# snapshot_time_s = {snapshot_time_s:.16e}\n")
        fp.write("# note = In-domain histograms are snapshot-local. Rate channels are instantaneous at this snapshot. Interval channels end at this snapshot.\n")
        fp.write("# summary_columns = record name value unit\n")
        for name, value, unit in summary_rows:
            fp.write(f"SUMMARY {name} {_format_scalar_value(value)} {unit}\n")
        fp.write("# histogram_columns = record channel quantity bin_left bin_right bin_center dM dM_dX cumulative_M\n")
        for channel, quantity, edges, weights in histogram_sections:
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = edges[1:] - edges[:-1]
            dmdx = np.zeros_like(weights)
            valid = widths > 0.0
            dmdx[valid] = weights[valid] / widths[valid]
            cumulative = np.cumsum(weights, dtype=np.float64)
            fp.write(f"# histogram_section = {channel} {quantity}\n")
            for left, right, center, dm, dmdx_val, cm in zip(edges[:-1], edges[1:], centers, weights, dmdx, cumulative):
                fp.write(
                    f"HIST {channel} {quantity} {left:.16e} {right:.16e} {center:.16e} "
                    f"{dm:.16e} {dmdx_val:.16e} {cm:.16e}\n"
                )
        if hist2d_sections:
            fp.write(
                "# histogram2d_columns = record channel quantity_x quantity_y "
                "bin_x_left bin_x_right bin_x_center bin_y_left bin_y_right bin_y_center dM\n"
            )
            for channel, quantity_x, quantity_y, x_edges, y_edges, weights in hist2d_sections:
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                fp.write(f"# histogram2d_section = {channel} {quantity_x} {quantity_y}\n")
                for ix, x_center in enumerate(x_centers):
                    for iy, y_center in enumerate(y_centers):
                        fp.write(
                            f"HIST2D {channel} {quantity_x} {quantity_y} "
                            f"{x_edges[ix]:.16e} {x_edges[ix + 1]:.16e} {x_center:.16e} "
                            f"{y_edges[iy]:.16e} {y_edges[iy + 1]:.16e} {y_center:.16e} "
                            f"{weights[ix, iy]:.16e}\n"
                        )


def parse_snapshot_analysis(path: Path) -> CachedSnapshotAnalysis:
    source_bin_name = ""
    snapshot_index = -1
    snapshot_time_s = math.nan
    summary: Dict[str, float] = {}
    histograms: Dict[Tuple[str, str], List[List[float]]] = {}
    with path.open("r", encoding="ascii") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    key, value = body.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "source_bin":
                        source_bin_name = value
                    elif key == "snapshot_index":
                        snapshot_index = int(value)
                    elif key == "snapshot_time_s":
                        snapshot_time_s = float(value)
                continue
            parts = line.split()
            if parts[0] == "SUMMARY":
                value = math.nan if parts[2].lower() == "nan" else float(parts[2])
                summary[parts[1]] = value
            elif parts[0] == "HIST":
                key = (parts[1], parts[2])
                histograms.setdefault(key, []).append([float(x) for x in parts[3:]])
    array_hists = {
        key: np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6), dtype=np.float64)
        for key, rows in histograms.items()
    }
    return CachedSnapshotAnalysis(
        path=path,
        source_bin_name=source_bin_name,
        snapshot_index=snapshot_index,
        snapshot_time_s=snapshot_time_s,
        summary=summary,
        histograms=array_hists,
    )


def histogram_edges_match(hist: np.ndarray, expected_edges: np.ndarray) -> bool:
    if hist.shape != (expected_edges.size - 1, 6):
        return False
    left = hist[:, 0]
    right = hist[:, 1]
    return np.allclose(left, expected_edges[:-1], rtol=1.0e-12, atol=1.0e-12) and np.allclose(
        right, expected_edges[1:], rtol=1.0e-12, atol=1.0e-12
    )


def cached_snapshot_is_valid(
    cached: CachedSnapshotAnalysis,
    snapshot_path: Path,
    expected_snapshot_index: int,
    expected_edges: Dict[str, np.ndarray],
    need_be: bool,
    config: dict,
    runtime: SnapshotRuntime,
    scale_info: Dict[str, object],
    density_floor_code: float,
    density_threshold_code: float,
    output_dir: Path,
) -> bool:
    if cached.source_bin_name != snapshot_path.name:
        return False
    if cached.snapshot_index != expected_snapshot_index:
        return False
    required_summary = {
        "snapshot_time",
        "M_domain",
        "M_bound_domain",
        "M_unbound_domain",
        "mass_fraction_e_lt_ecrit",
        "mass_fraction_Rcirc_lt_Rcrit",
        "mean_j_bound",
        "median_j_bound",
        "mean_eps_bound",
        "median_eps_bound",
        "sink_rate_total_snapshot",
        "outer_rate_total_snapshot",
        "control_rate_total_snapshot",
        "analysis_format_version",
        "density_floor_code_used",
        "density_threshold_factor_used",
        "density_threshold_code_used",
        "sink_radius_code_used",
        "sink_shell_cells_used",
        "outer_shell_cells_used",
        "control_radius_code_used",
        "control_shell_cells_used",
        "r_tide_code_used",
        "stellar_radius_code_used",
        "a_mb_code_used",
        "fallback_shell_cells_used",
        "inner_shell_radius_code_used",
        "inner_shell_cells_used",
        "outer_stream_radius_code_used",
        "outer_stream_shell_cells_used",
        "compression_threshold_code_used",
        "compression_percentile_used",
        "mass_compressed_proxy",
        "mass_fraction_compressed_proxy",
        "mass_bound_inward_compressed_proxy",
        "tail_energy_multiplier_used",
        "tail_energy_threshold_code",
        "tail_energy_threshold_cgs",
        "tail_j_threshold_code_used",
        "tail_j_percentile_used",
        "tail_e_threshold_used",
        "M_tail_domain",
        "M_tail_bound_inward",
        "M_tail_lowj",
        "M_tail_high_e",
        "tail_mass_fraction_of_domain",
        "tail_mass_fraction_of_bound",
        "low_j_mass_fraction_bound",
        "high_e_mass_fraction_bound",
        "tail_alpha",
        "tail_fit_nbins",
        "tail_fit_energy_min",
        "tail_fit_energy_max",
        "tail_fit_r2",
        "ecrit_used",
        "Rcrit_code_used",
    }
    fallback_labels = scale_info["fallback_labels"]
    for label in fallback_labels:
        required_summary.update({
            f"fallback_radius_code_{label}",
            f"fallback_rate_inward_total_snapshot_{label}",
            f"fallback_rate_outward_total_snapshot_{label}",
            f"fallback_rate_net_total_snapshot_{label}",
            f"Mdot_fallback_inward_{label}",
            f"Mdot_fallback_outward_{label}",
            f"Mdot_fallback_net_{label}",
            f"cumulative_inward_mass_{label}",
            f"cumulative_outward_mass_{label}",
            f"cumulative_net_mass_{label}",
            f"t_over_tref_{label}",
            f"mdot_tminus5over3_normalized_{label}",
        })
    if not required_summary.issubset(cached.summary.keys()):
        return False
    required_hist_keys = {
        ("in_domain", "energy"),
        ("in_domain", "j"),
        ("in_domain", "e"),
        ("sink_rate", "energy"),
        ("sink_rate", "j"),
        ("outer_rate", "energy"),
        ("outer_rate", "j"),
        ("in_domain_inward", "energy"),
        ("in_domain_inward", "j"),
        ("in_domain_outward", "energy"),
        ("in_domain_outward", "j"),
        ("in_domain_bound_inward", "energy"),
        ("in_domain_bound_inward", "j"),
        ("in_domain_bound_outward", "energy"),
        ("in_domain_bound_outward", "j"),
        ("in_domain_unbound", "energy"),
        ("in_domain_unbound", "j"),
        ("in_domain_compressed_proxy", "energy"),
        ("in_domain_compressed_proxy", "j"),
        ("in_domain_bound_inward_compressed_proxy", "energy"),
        ("in_domain_bound_inward_compressed_proxy", "j"),
        ("shell_inner", "energy"),
        ("shell_inner", "j"),
        ("shell_outer_stream", "energy"),
        ("shell_outer_stream", "j"),
    }
    for label in fallback_labels:
        required_hist_keys.update({
            (f"fallback_rate_inward_{label}", "energy"),
            (f"fallback_rate_inward_{label}", "j"),
            (f"fallback_rate_inward_{label}", "e"),
            (f"fallback_rate_outward_{label}", "energy"),
            (f"fallback_rate_outward_{label}", "j"),
            (f"fallback_rate_outward_{label}", "e"),
            (f"fallback_rate_net_{label}", "energy"),
            (f"fallback_rate_net_{label}", "j"),
            (f"fallback_rate_net_{label}", "e"),
            (f"shell_fallback_{label}", "energy"),
            (f"shell_fallback_{label}", "j"),
            (f"shell_fallback_{label}", "e"),
        })
    if need_be:
        required_hist_keys.update({
            ("in_domain", "Be"),
            ("sink_rate", "Be"),
            ("outer_rate", "Be"),
            ("in_domain_inward", "Be"),
            ("in_domain_outward", "Be"),
            ("in_domain_bound_inward", "Be"),
            ("in_domain_bound_outward", "Be"),
            ("in_domain_unbound", "Be"),
            ("in_domain_compressed_proxy", "Be"),
            ("in_domain_bound_inward_compressed_proxy", "Be"),
            ("shell_inner", "Be"),
            ("shell_outer_stream", "Be"),
        })
        for label in fallback_labels:
            required_hist_keys.update({
                (f"fallback_rate_inward_{label}", "Be"),
                (f"fallback_rate_outward_{label}", "Be"),
                (f"fallback_rate_net_{label}", "Be"),
                (f"shell_fallback_{label}", "Be"),
            })
    if not required_hist_keys.issubset(cached.histograms.keys()):
        return False
    edge_lookup = {
        "energy": expected_edges["energy"],
        "j": expected_edges["j"],
        "Be": expected_edges["Be"],
        "e": expected_edges["e"],
    }
    for channel, quantity in required_hist_keys:
        if not histogram_edges_match(cached.histograms[(channel, quantity)], edge_lookup[quantity]):
            return False
    config_checks = {
        "analysis_format_version": ANALYSIS_FORMAT_VERSION,
        "density_floor_code_used": float(density_floor_code),
        "density_threshold_factor_used": float(config["density_threshold_factor"]),
        "density_threshold_code_used": float(density_threshold_code),
        "sink_radius_code_used": float(runtime.sink_radius_code),
        "sink_shell_cells_used": float(config["sink_shell_cells"]),
        "outer_shell_cells_used": float(config["outer_shell_cells"]),
        "control_radius_code_used": float(config["control_radius_code"]),
        "control_shell_cells_used": float(config["control_shell_cells"]),
        "r_tide_code_used": float(scale_info["r_tide_code"]),
        "stellar_radius_code_used": float(scale_info["stellar_radius_code"]),
        "a_mb_code_used": float(scale_info["a_mb_code"]),
        "fallback_shell_cells_used": float(config["fallback_shell_cells"]),
        "inner_shell_radius_code_used": float(scale_info["inner_shell_radius_code"]),
        "inner_shell_cells_used": float(config["inner_shell_cells"]),
        "outer_stream_radius_code_used": float(scale_info["outer_stream_radius_code"]),
        "outer_stream_shell_cells_used": float(config["outer_stream_shell_cells"]),
        "compression_percentile_used": float(config["compression_percentile"]),
        "tail_energy_multiplier_used": float(config["tail_energy_multiplier"]),
        "tail_energy_threshold_code": float(scale_info["tail_energy_threshold_code"]),
        "tail_energy_threshold_cgs": float(scale_info["tail_energy_threshold_cgs"]),
        "tail_j_percentile_used": float(config["tail_j_percentile"]),
        "tail_e_threshold_used": float(config["tail_e_threshold"]),
        "ecrit_used": float(config["ecrit"]),
        "Rcrit_code_used": float(config["Rcrit_code"]),
    }
    compression_threshold_cfg = config.get("compression_threshold_code")
    config_checks["compression_threshold_code_used"] = (
        float(compression_threshold_cfg) if compression_threshold_cfg is not None else float(cached.summary["compression_threshold_code_used"])
    )
    for key, expected in config_checks.items():
        if not math.isclose(float(cached.summary[key]), expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            return False
    tail_j_cfg = config.get("tail_j_threshold_code")
    if tail_j_cfg is not None and not math.isclose(
        float(cached.summary["tail_j_threshold_code_used"]),
        float(tail_j_cfg),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        return False
    for radius, label in zip(scale_info["fallback_radii_code"], fallback_labels):
        if not math.isclose(
            float(cached.summary[f"fallback_radius_code_{label}"]),
            float(radius),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            return False
    return True


def load_cached_snapshot_analyses(output_dir: Path) -> List[CachedSnapshotAnalysis]:
    cached_snapshots: List[CachedSnapshotAnalysis] = []
    for path in sorted(output_dir.glob("*.analysis.dat")):
        try:
            cached = parse_snapshot_analysis(path)
        except Exception:
            continue
        if cached.snapshot_index < 0:
            continue
        if "snapshot_time" not in cached.summary:
            continue
        cached_snapshots.append(cached)
    cached_snapshots.sort(key=lambda item: (item.snapshot_index, item.snapshot_time_s, item.path.name))
    return cached_snapshots


GLOBAL_TIMESERIES_COLUMNS = [
    "snapshot_index",
    "snapshot_time",
    "interval_mid_time",
    "M_domain",
    "M_sink_cumulative",
    "M_outer_cumulative",
    "M_bound_domain",
    "M_unbound_domain",
    "M_total_accounted",
    "mass_conservation_error",
    "Mdot_sink",
    "Mdot_outer",
    "dM_bound_domain_dt",
    "Mdot_control",
    "mass_fraction_e_lt_ecrit",
    "mass_fraction_Rcirc_lt_Rcrit",
    "mean_j_bound",
    "median_j_bound",
    "mean_eps_bound",
    "median_eps_bound",
    "M_tail_domain",
    "M_tail_bound_inward",
    "M_tail_lowj",
    "M_tail_high_e",
    "tail_mass_fraction_of_domain",
    "tail_mass_fraction_of_bound",
    "tail_energy_threshold_code",
    "tail_energy_threshold_cgs",
]


def write_global_timeseries(path: Path, rows: Sequence[Sequence[object]]) -> None:
    columns = GLOBAL_TIMESERIES_COLUMNS
    units_line = [
        "index",
        "s",
        "s",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g/s",
        "g/s",
        "g/s",
        "g/s",
        "dimensionless",
        "dimensionless",
        "cm^2/s",
        "cm^2/s",
        "erg/g",
        "erg/g",
        "g",
        "g",
        "g",
        "g",
        "dimensionless",
        "dimensionless",
        "code_specific_energy",
        "erg/g",
    ]
    note = (
        "One row per processed snapshot. Rate columns correspond to the interval ending at this "
        "snapshot; they are nan for the first snapshot."
    )
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="ascii") as fp:
        fp.write("# columns = " + " ".join(columns) + "\n")
        fp.write("# units = " + " ".join(units_line) + "\n")
        fp.write(f"# note = {note}\n")
        for row in rows:
            formatted = [_format_scalar_value(row[0])]
            for value in row[1:]:
                formatted.append(_format_scalar_value(value))
            fp.write(" ".join(formatted) + "\n")
    tmp_path.replace(path)


def rebuild_global_timeseries_from_snapshot_cache(output_dir: Path) -> List[List[object]]:
    columns = GLOBAL_TIMESERIES_COLUMNS
    cached_snapshots = load_cached_snapshot_analyses(output_dir)
    rows: List[List[object]] = []
    for cached in cached_snapshots:
        row: List[object] = [cached.snapshot_index]
        for key in columns[1:]:
            row.append(cached.summary.get(key, math.nan))
        rows.append(row)
    return rows


def edges_from_cached_hist(hist: np.ndarray) -> np.ndarray:
    if hist.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate((hist[:, 0], np.array([hist[-1, 1]], dtype=np.float64)))


def build_cumulative_sections_from_snapshot_cache(
    cached_snapshots: Sequence[CachedSnapshotAnalysis],
    include_be: bool,
) -> Tuple[int, List[Tuple[str, str, np.ndarray, np.ndarray]]]:
    if not cached_snapshots:
        return -1, []
    latest = cached_snapshots[-1]
    latest_snapshot_index = latest.snapshot_index
    hist_energy = latest.histograms[("in_domain", "energy")]
    hist_j = latest.histograms[("in_domain", "j")]
    hist_e = latest.histograms[("in_domain", "e")]
    latest_hist_E_g = hist_energy[:, 3].copy()
    latest_hist_j_g = hist_j[:, 3].copy()
    latest_hist_e_g = hist_e[:, 3].copy()
    latest_hist_Be_g = (
        latest.histograms[("in_domain", "Be")][:, 3].copy()
        if include_be and ("in_domain", "Be") in latest.histograms
        else np.zeros(0, dtype=np.float64)
    )
    sink_E = np.zeros_like(latest_hist_E_g)
    sink_j = np.zeros_like(latest_hist_j_g)
    sink_Be = np.zeros_like(latest_hist_Be_g)
    outer_E = np.zeros_like(latest_hist_E_g)
    outer_j = np.zeros_like(latest_hist_j_g)
    outer_Be = np.zeros_like(latest_hist_Be_g)
    for cached in cached_snapshots:
        if ("sink_interval", "energy") in cached.histograms:
            sink_E += cached.histograms[("sink_interval", "energy")][:, 3]
        if ("sink_interval", "j") in cached.histograms:
            sink_j += cached.histograms[("sink_interval", "j")][:, 3]
        if include_be and ("sink_interval", "Be") in cached.histograms:
            sink_Be += cached.histograms[("sink_interval", "Be")][:, 3]
        if ("outer_interval", "energy") in cached.histograms:
            outer_E += cached.histograms[("outer_interval", "energy")][:, 3]
        if ("outer_interval", "j") in cached.histograms:
            outer_j += cached.histograms[("outer_interval", "j")][:, 3]
        if include_be and ("outer_interval", "Be") in cached.histograms:
            outer_Be += cached.histograms[("outer_interval", "Be")][:, 3]
    edges_energy = edges_from_cached_hist(hist_energy)
    edges_j = edges_from_cached_hist(hist_j)
    edges_e = edges_from_cached_hist(hist_e)
    edges_be = (
        edges_from_cached_hist(latest.histograms[("in_domain", "Be")])
        if include_be and ("in_domain", "Be") in latest.histograms
        else np.zeros(0, dtype=np.float64)
    )
    sections: List[Tuple[str, str, np.ndarray, np.ndarray]] = [
        ("in_domain_latest", "energy", edges_energy, latest_hist_E_g),
        ("in_domain_latest", "j", edges_j, latest_hist_j_g),
        ("in_domain_latest", "e", edges_e, latest_hist_e_g),
        ("sink_cumulative", "energy", edges_energy, sink_E),
        ("sink_cumulative", "j", edges_j, sink_j),
        ("outer_cumulative", "energy", edges_energy, outer_E),
        ("outer_cumulative", "j", edges_j, outer_j),
        ("approximate_total_latest", "energy", edges_energy, latest_hist_E_g + sink_E + outer_E),
        ("approximate_total_latest", "j", edges_j, latest_hist_j_g + sink_j + outer_j),
    ]
    if include_be:
        sections.extend([
            ("in_domain_latest", "Be", edges_be, latest_hist_Be_g),
            ("sink_cumulative", "Be", edges_be, sink_Be),
            ("outer_cumulative", "Be", edges_be, outer_Be),
            ("approximate_total_latest", "Be", edges_be, latest_hist_Be_g + sink_Be + outer_Be),
        ])
    return latest_snapshot_index, sections


def rebuild_fallback_shell_timeseries_from_cache(
    cached_snapshots: Sequence[CachedSnapshotAnalysis],
    shell_labels: Sequence[str],
) -> Tuple[List[str], List[str], List[List[object]]]:
    columns = ["snapshot_index", "snapshot_time", "interval_mid_time"]
    units_line = ["index", "s", "s"]
    for label in shell_labels:
        columns.extend([
            f"Mdot_fallback_inward_{label}",
            f"Mdot_fallback_outward_{label}",
            f"Mdot_fallback_net_{label}",
            f"cumulative_inward_mass_{label}",
            f"cumulative_outward_mass_{label}",
            f"cumulative_net_mass_{label}",
            f"t_over_tref_{label}",
            f"mdot_tminus5over3_normalized_{label}",
        ])
        units_line.extend(["g/s", "g/s", "g/s", "g", "g", "g", "dimensionless", "g/s"])
    if not cached_snapshots:
        return columns, units_line, []

    have_direct_summary = True
    for cached in cached_snapshots:
        for label in shell_labels:
            for key in (
                f"Mdot_fallback_inward_{label}",
                f"Mdot_fallback_outward_{label}",
                f"Mdot_fallback_net_{label}",
                f"cumulative_inward_mass_{label}",
                f"cumulative_outward_mass_{label}",
                f"cumulative_net_mass_{label}",
                f"t_over_tref_{label}",
                f"mdot_tminus5over3_normalized_{label}",
            ):
                if key not in cached.summary:
                    have_direct_summary = False
                    break
            if not have_direct_summary:
                break
        if not have_direct_summary:
            break

    if have_direct_summary:
        rows: List[List[object]] = []
        for cached in cached_snapshots:
            row: List[object] = [
                cached.snapshot_index,
                cached.summary.get("snapshot_time", math.nan),
                cached.summary.get("interval_mid_time", math.nan),
            ]
            for label in shell_labels:
                row.extend([
                    cached.summary.get(f"Mdot_fallback_inward_{label}", math.nan),
                    cached.summary.get(f"Mdot_fallback_outward_{label}", math.nan),
                    cached.summary.get(f"Mdot_fallback_net_{label}", math.nan),
                    cached.summary.get(f"cumulative_inward_mass_{label}", math.nan),
                    cached.summary.get(f"cumulative_outward_mass_{label}", math.nan),
                    cached.summary.get(f"cumulative_net_mass_{label}", math.nan),
                    cached.summary.get(f"t_over_tref_{label}", math.nan),
                    cached.summary.get(f"mdot_tminus5over3_normalized_{label}", math.nan),
                ])
            rows.append(row)
        return columns, units_line, rows

    nshell = len(shell_labels)
    rows: List[List[object]] = []
    cumulative_in = np.zeros(nshell, dtype=np.float64)
    cumulative_out = np.zeros(nshell, dtype=np.float64)
    cumulative_net = np.zeros(nshell, dtype=np.float64)
    prev_time_s = None
    prev_in = np.full(nshell, math.nan, dtype=np.float64)
    prev_out = np.full(nshell, math.nan, dtype=np.float64)
    prev_net = np.full(nshell, math.nan, dtype=np.float64)
    raw_rows: List[List[object]] = []

    for cached in cached_snapshots:
        time_s = float(cached.summary.get("snapshot_time", math.nan))
        interval_mid_s = math.nan
        avg_in = np.full(nshell, math.nan, dtype=np.float64)
        avg_out = np.full(nshell, math.nan, dtype=np.float64)
        avg_net = np.full(nshell, math.nan, dtype=np.float64)
        inst_in = np.array([
            float(cached.summary.get(f"fallback_rate_inward_total_snapshot_{label}", math.nan))
            for label in shell_labels
        ], dtype=np.float64)
        inst_out = np.array([
            float(cached.summary.get(f"fallback_rate_outward_total_snapshot_{label}", math.nan))
            for label in shell_labels
        ], dtype=np.float64)
        inst_net = np.array([
            float(cached.summary.get(f"fallback_rate_net_total_snapshot_{label}", math.nan))
            for label in shell_labels
        ], dtype=np.float64)
        if prev_time_s is not None and time_s > prev_time_s:
            dt = time_s - prev_time_s
            interval_mid_s = 0.5 * (time_s + prev_time_s)
            avg_in = 0.5 * (prev_in + inst_in)
            avg_out = 0.5 * (prev_out + inst_out)
            avg_net = 0.5 * (prev_net + inst_net)
            cumulative_in += avg_in * dt
            cumulative_out += avg_out * dt
            cumulative_net += avg_net * dt
        row: List[object] = [cached.snapshot_index, time_s, interval_mid_s]
        for idx in range(nshell):
            row.extend([
                avg_in[idx],
                avg_out[idx],
                avg_net[idx],
                cumulative_in[idx],
                cumulative_out[idx],
                cumulative_net[idx],
                math.nan,
                math.nan,
            ])
        raw_rows.append(row)
        prev_time_s = time_s
        prev_in = inst_in
        prev_out = inst_out
        prev_net = inst_net

    for shell_index in range(nshell):
        tref = math.nan
        mdot_ref = math.nan
        for row in raw_rows:
            mdot = row[3 + shell_index * 8]
            tmid = row[2]
            if isinstance(mdot, float) and isinstance(tmid, float) and math.isfinite(mdot) and math.isfinite(tmid) and mdot > 0.0:
                tref = tmid
                mdot_ref = mdot
                break
        for row in raw_rows:
            tmid = row[2]
            if math.isfinite(tref) and math.isfinite(tmid) and tmid > 0.0 and tmid >= tref and math.isfinite(mdot_ref):
                ratio = tmid / tref
                row[3 + shell_index * 8 + 6] = ratio
                row[3 + shell_index * 8 + 7] = mdot_ref * ratio ** (-5.0 / 3.0)
    return columns, units_line, raw_rows


def write_fallback_shell_timeseries(path: Path, columns: Sequence[str], units_line: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    note = (
        "Fallback-shell inward and outward rates are practical proxy fluxes through chosen spherical shells. "
        "The normalized t^(-5/3) columns are guide curves anchored at the first interval with positive inward rate."
    )
    write_table(path, columns, units_line, rows, note)


def rebuild_tail_budget_timeseries_from_cache(cached_snapshots: Sequence[CachedSnapshotAnalysis]) -> List[List[object]]:
    rows: List[List[object]] = []
    for cached in cached_snapshots:
        rows.append([
            cached.snapshot_index,
            cached.summary.get("snapshot_time", math.nan),
            cached.summary.get("M_domain", math.nan),
            cached.summary.get("M_bound_domain", math.nan),
            cached.summary.get("tail_energy_threshold_code", math.nan),
            cached.summary.get("tail_energy_threshold_cgs", math.nan),
            cached.summary.get("M_tail_domain", math.nan),
            cached.summary.get("M_tail_bound_inward", math.nan),
            cached.summary.get("M_tail_lowj", math.nan),
            cached.summary.get("M_tail_high_e", math.nan),
            cached.summary.get("tail_mass_fraction_of_domain", math.nan),
            cached.summary.get("tail_mass_fraction_of_bound", math.nan),
        ])
    return rows


def write_tail_budget_timeseries(path: Path, rows: Sequence[Sequence[object]]) -> None:
    columns = [
        "snapshot_index",
        "snapshot_time",
        "M_domain",
        "M_bound_domain",
        "tail_energy_threshold_code",
        "tail_energy_threshold_cgs",
        "M_tail_domain",
        "M_tail_bound_inward",
        "M_tail_lowj",
        "M_tail_high_e",
        "tail_mass_fraction_of_domain",
        "tail_mass_fraction_of_bound",
    ]
    units_line = ["index", "s", "g", "g", "code_specific_energy", "erg/g", "g", "g", "g", "g", "dimensionless", "dimensionless"]
    note = (
        "Deep-tail quantities use the physically motivated tidal-energy-scale threshold "
        "tail_energy_threshold_code = - tail_energy_multiplier * G M_BH R_star / r_tide^2."
    )
    write_table(path, columns, units_line, rows, note)


def rebuild_branch_timeseries_from_cache(
    cached_snapshots: Sequence[CachedSnapshotAnalysis],
    shell_labels: Sequence[str],
) -> Tuple[List[str], List[str], List[List[object]]]:
    columns = [
        "snapshot_index",
        "snapshot_time",
        "M_domain",
        "M_bound_domain",
        "M_tail_domain",
        "M_tail_bound_inward",
        "mean_j_bound",
        "median_j_bound",
        "mean_eps_bound",
        "median_eps_bound",
        "low_j_mass_fraction_bound",
        "high_e_mass_fraction_bound",
        "compressed_proxy_mass_fraction",
        "Mdot_sink",
        "Mdot_control",
    ]
    units_line = [
        "index",
        "s",
        "g",
        "g",
        "g",
        "g",
        "cm^2/s",
        "cm^2/s",
        "erg/g",
        "erg/g",
        "dimensionless",
        "dimensionless",
        "dimensionless",
        "g/s",
        "g/s",
    ]
    for label in shell_labels:
        columns.extend([
            f"Mdot_fallback_inward_{label}",
            f"Mdot_fallback_outward_{label}",
            f"Mdot_fallback_net_{label}",
        ])
        units_line.extend(["g/s", "g/s", "g/s"])
    columns.extend([
        "tail_alpha",
        "tail_fit_nbins",
        "tail_fit_energy_min",
        "tail_fit_energy_max",
        "tail_fit_r2",
    ])
    units_line.extend(["dimensionless", "count", "code_specific_energy", "code_specific_energy", "dimensionless"])

    fallback_columns, _, fallback_rows = rebuild_fallback_shell_timeseries_from_cache(cached_snapshots, shell_labels)
    fallback_lookup = {}
    for row in fallback_rows:
        fallback_lookup[int(row[0])] = row

    rows: List[List[object]] = []
    for cached in cached_snapshots:
        row: List[object] = [
            cached.snapshot_index,
            cached.summary.get("snapshot_time", math.nan),
            cached.summary.get("M_domain", math.nan),
            cached.summary.get("M_bound_domain", math.nan),
            cached.summary.get("M_tail_domain", math.nan),
            cached.summary.get("M_tail_bound_inward", math.nan),
            cached.summary.get("mean_j_bound", math.nan),
            cached.summary.get("median_j_bound", math.nan),
            cached.summary.get("mean_eps_bound", math.nan),
            cached.summary.get("median_eps_bound", math.nan),
            cached.summary.get("low_j_mass_fraction_bound", math.nan),
            cached.summary.get("high_e_mass_fraction_bound", math.nan),
            cached.summary.get("mass_fraction_compressed_proxy", math.nan),
            cached.summary.get("Mdot_sink", math.nan),
            cached.summary.get("Mdot_control", math.nan),
        ]
        fallback_row = fallback_lookup.get(cached.snapshot_index)
        for shell_index, _label in enumerate(shell_labels):
            if fallback_row is None:
                row.extend([math.nan, math.nan, math.nan])
            else:
                offset = 3 + shell_index * 8
                row.extend([fallback_row[offset], fallback_row[offset + 1], fallback_row[offset + 2]])
        row.extend([
            cached.summary.get("tail_alpha", math.nan),
            cached.summary.get("tail_fit_nbins", math.nan),
            cached.summary.get("tail_fit_energy_min", math.nan),
            cached.summary.get("tail_fit_energy_max", math.nan),
            cached.summary.get("tail_fit_r2", math.nan),
        ])
        rows.append(row)
    return columns, units_line, rows


def write_branch_timeseries(path: Path, columns: Sequence[str], units_line: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    note = (
        "Branch diagnostics are descriptive summary measures for comparing late-time deep-bound, low-j, "
        "high-e, compressed-proxy, sink, control-radius, and fallback-shell behavior."
    )
    write_table(path, columns, units_line, rows, note)


def rebuild_and_write_run_level_outputs(
    output_dir: Path,
    shell_labels: Sequence[str],
    include_be: bool,
) -> None:
    cached_snapshots = load_cached_snapshot_analyses(output_dir)
    write_global_timeseries(output_dir / "analysis_global_timeseries.dat", [
        [cached.snapshot_index] + [cached.summary.get(key, math.nan) for key in GLOBAL_TIMESERIES_COLUMNS[1:]]
        for cached in cached_snapshots
    ])
    latest_snapshot_index, cumulative_sections = build_cumulative_sections_from_snapshot_cache(
        cached_snapshots,
        include_be=include_be,
    )
    if cumulative_sections:
        write_cumulative_distributions(
            output_dir / "analysis_cumulative_distributions.dat",
            latest_snapshot_index=latest_snapshot_index,
            histogram_sections=cumulative_sections,
        )
    fallback_columns, fallback_units, fallback_rows = rebuild_fallback_shell_timeseries_from_cache(
        cached_snapshots,
        shell_labels,
    )
    write_fallback_shell_timeseries(
        output_dir / "analysis_fallback_shell_timeseries.dat",
        fallback_columns,
        fallback_units,
        fallback_rows,
    )
    write_tail_budget_timeseries(
        output_dir / "analysis_tail_budget_timeseries.dat",
        rebuild_tail_budget_timeseries_from_cache(cached_snapshots),
    )
    branch_columns, branch_units, branch_rows = rebuild_branch_timeseries_from_cache(cached_snapshots, shell_labels)
    write_branch_timeseries(
        output_dir / "analysis_branch_timeseries.dat",
        branch_columns,
        branch_units,
        branch_rows,
    )


def write_cumulative_distributions(
    path: Path,
    latest_snapshot_index: int,
    histogram_sections: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="ascii") as fp:
        fp.write("# file_type = cumulative_distributions\n")
        fp.write(f"# latest_snapshot_index = {latest_snapshot_index}\n")
        fp.write("# note = Cumulative sink and outer distributions are integrated over all processed intervals. approximate_total_latest = in_domain_latest + sink_cumulative + outer_cumulative.\n")
        fp.write("# histogram_columns = record channel quantity bin_left bin_right bin_center dM dM_dX cumulative_M\n")
        for channel, quantity, edges, weights in histogram_sections:
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = edges[1:] - edges[:-1]
            dmdx = np.zeros_like(weights)
            valid = widths > 0.0
            dmdx[valid] = weights[valid] / widths[valid]
            cumulative = np.cumsum(weights, dtype=np.float64)
            fp.write(f"# histogram_section = {channel} {quantity}\n")
            for left, right, center, dm, dmdx_val, cm in zip(edges[:-1], edges[1:], centers, weights, dmdx, cumulative):
                fp.write(
                    f"HIST {channel} {quantity} {left:.16e} {right:.16e} {center:.16e} "
                    f"{dm:.16e} {dmdx_val:.16e} {cm:.16e}\n"
                )
    tmp_path.replace(path)


def build_cumulative_sections(
    edges_code: Dict[str, np.ndarray],
    units: UnitSystem,
    latest_hist_E_g: np.ndarray,
    latest_hist_j_g: np.ndarray,
    latest_hist_Be_g: np.ndarray,
    latest_hist_e_g: np.ndarray,
    cumulative_sink_E_code: np.ndarray,
    cumulative_sink_j_code: np.ndarray,
    cumulative_sink_Be_code: np.ndarray,
    cumulative_outer_E_code: np.ndarray,
    cumulative_outer_j_code: np.ndarray,
    cumulative_outer_Be_code: np.ndarray,
    include_be: bool,
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    sections: List[Tuple[str, str, np.ndarray, np.ndarray]] = [
        ("in_domain_latest", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), latest_hist_E_g),
        ("in_domain_latest", "j", convert_edges_to_physical("j", edges_code["j"], units), latest_hist_j_g),
        ("in_domain_latest", "e", convert_edges_to_physical("e", edges_code["e"], units), latest_hist_e_g),
        ("sink_cumulative", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), cumulative_sink_E_code * units.mass_cgs),
        ("sink_cumulative", "j", convert_edges_to_physical("j", edges_code["j"], units), cumulative_sink_j_code * units.mass_cgs),
        ("outer_cumulative", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), cumulative_outer_E_code * units.mass_cgs),
        ("outer_cumulative", "j", convert_edges_to_physical("j", edges_code["j"], units), cumulative_outer_j_code * units.mass_cgs),
        (
            "approximate_total_latest",
            "energy",
            convert_edges_to_physical("energy", edges_code["energy"], units),
            latest_hist_E_g + cumulative_sink_E_code * units.mass_cgs + cumulative_outer_E_code * units.mass_cgs,
        ),
        (
            "approximate_total_latest",
            "j",
            convert_edges_to_physical("j", edges_code["j"], units),
            latest_hist_j_g + cumulative_sink_j_code * units.mass_cgs + cumulative_outer_j_code * units.mass_cgs,
        ),
    ]
    if include_be:
        sections.extend([
            ("in_domain_latest", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), latest_hist_Be_g),
            ("sink_cumulative", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), cumulative_sink_Be_code * units.mass_cgs),
            ("outer_cumulative", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), cumulative_outer_Be_code * units.mass_cgs),
            (
                "approximate_total_latest",
                "Be",
                convert_edges_to_physical("Be", edges_code["Be"], units),
                latest_hist_Be_g + cumulative_sink_Be_code * units.mass_cgs + cumulative_outer_Be_code * units.mass_cgs,
            ),
        ])
    return sections


def write_units_summary(
    output_dir: Path,
    units: UnitSystem,
    runtime: SnapshotRuntime,
    available_variables: Sequence[str],
    have_face_fluxes: bool,
    scale_info: Dict[str, object],
) -> None:
    rows = [
        ("length_cgs", units.length_cgs),
        ("mass_cgs", units.mass_cgs),
        ("time_cgs", units.time_cgs),
        ("velocity_cgs", units.velocity_cgs),
        ("density_cgs", units.density_cgs),
        ("pressure_cgs", units.pressure_cgs),
        ("specific_eint_cgs", units.specific_eint_cgs),
        ("angular_momentum_cgs", units.angular_momentum_cgs),
        ("newton_g_code", runtime.newton_g_code),
        ("newton_g_cgs", GRAV_CONSTANT_CGS),
        ("bh_mass_code", runtime.bh_mass_code),
        ("bh_mass_cgs", runtime.bh_mass_code * units.mass_cgs),
        ("bh_softening_code", runtime.bh_softening_code),
        ("bh_softening_cgs", runtime.bh_softening_code * units.length_cgs),
        ("sink_radius_code", runtime.sink_radius_code),
        ("sink_radius_cgs", runtime.sink_radius_code * units.length_cgs),
        ("r_tide_code", float(scale_info["r_tide_code"])),
        ("r_tide_cgs", float(scale_info["r_tide_code"]) * units.length_cgs),
        ("stellar_radius_code", float(scale_info["stellar_radius_code"])),
        ("stellar_radius_cgs", float(scale_info["stellar_radius_code"]) * units.length_cgs),
        ("a_mb_code", float(scale_info["a_mb_code"])),
        ("a_mb_cgs", float(scale_info["a_mb_code"]) * units.length_cgs),
        ("tail_energy_threshold_code", float(scale_info["tail_energy_threshold_code"])),
        ("tail_energy_threshold_cgs", float(scale_info["tail_energy_threshold_cgs"])),
        ("face_fluxes_in_dump", 1.0 if have_face_fluxes else 0.0),
    ]
    path = output_dir / "units_summary.dat"
    with path.open("w", encoding="ascii") as fp:
        fp.write("# columns = name value\n")
        fp.write("# units = string mixed\n")
        fp.write("# variables = " + " ".join(available_variables) + "\n")
        fp.write(
            "# fallback_radius_labels = "
            + " ".join(str(label) for label in scale_info["fallback_labels"])
            + "\n"
        )
        for name, value in rows:
            fp.write(f"{name} {value:.16e}\n")


def write_readme(
    output_dir: Path,
    units: UnitSystem,
    runtime: SnapshotRuntime,
    available_variables: Sequence[str],
    have_eint: bool,
    have_face_fluxes: bool,
    config: dict,
    scale_info: Dict[str, object],
) -> None:
    lines = [
        "# Analysis Outputs",
        "",
        "## Inspection Summary",
        "",
        f"1. Variables available in the `hydro_w` dumps: `{', '.join(available_variables)}`.",
        "2. BH potential used for orbital diagnostics:",
        "   `Phi_BH = - G_code * M_BH / sqrt(|r-r_BH|^2 + eps_soft^2)`.",
        "   This is the external softened Newtonian potential from `src/pgen/tde_external.cpp`.",
        "   The stored `grav_phi` field in the dump is the total output potential (self-gravity plus external BH),",
        "   but `eps_orb` and `Be` in this analysis use the BH term only, as requested.",
        "3. Units are read from the embedded `[units]` block in each dump: `length_cgs`, `mass_cgs`, `time_cgs`, `mu`.",
        "   Data products are written in physical units: time in `s`, mass in `g`, rates in `g/s`, specific energy in `erg/g`,",
        "   specific angular momentum in `cm^2/s`, and circularization radius in `cm`.",
        f"4. True face fluxes are {'not ' if not have_face_fluxes else ''}available in these `.bin` dumps.",
        "   For the present `hydro_w` files there are no boundary-face flux variables written out,",
        "   so sink and outer losses are estimated from snapshot-local shell / boundary proxies.",
        "5. Assumptions used by the loss estimators:",
        f"   sink shell width = `sink_shell_cells * min(dx)` with `sink_shell_cells = {config['sink_shell_cells']}`;",
        f"   outer boundary uses the outermost Cartesian cells with `outer_shell_cells = {config['outer_shell_cells']}`;",
        f"   control radius = `{config['control_radius_code']}` code units;",
        "   BH-centered inertial diagnostics use `r_rel = x_cell - bh_live_x` and `v_rel = v_cell - bh_live_v` when live BH metadata exists.",
        "",
        "## Scientific Caution",
        "",
        "The in-domain snapshot histograms are not full late-time system distributions.",
        "Gas can be lost through the inner sink / excision boundary and through the outer simulation boundary.",
        f"All mass and flux diagnostics only include cells with `rho > {config['density_threshold_factor']} * dfloor`,",
        "so box-filling floor atmosphere added by later domain expansion is excluded from the debris budget.",
        "This directory therefore keeps three channels separate:",
        "",
        "- in-domain snapshot distributions",
        "- sink-crossing interval and cumulative distributions",
        "- outer-boundary interval and cumulative distributions",
        "- fallback-shell proxy fluxes across multiple spherical radii defined in units of `a_mb = r_tide^2 / (2 R_star)`",
        "",
        "To build an approximate total budget, combine the in-domain and boundary-loss channels explicitly.",
        "",
        "## Files",
        "",
        "- `*.analysis.dat`: one combined per-snapshot file per input bin file. Each such file contains snapshot scalar diagnostics plus 1D and 2D histogram sections for that bin file.",
        "- `paper_figures/`: optional plotting output directory created by `plot_analysis.py`.",
        "- Existing `*.analysis.dat` files can be reused on reruns when `resume_existing_outputs = true` and the cached binning still matches.",
        "- Standalone run-level time-series `.dat` files are no longer written; plotting reconstructs time functions from per-snapshot `SUMMARY` rows.",
        "",
        "## Column Definitions",
        "",
        "### `*.analysis.dat` Summary Scalars",
        "",
        "- `snapshot_index`: integer snapshot index parsed from the filename.",
        "- `snapshot_time`: snapshot time in `s`.",
        "- `interval_mid_time`: midpoint time for rate estimates in `s`; `nan` for the first snapshot.",
        f"- `M_domain`: gas mass remaining in the computational domain, excluding cells inside the sink radius and excluding cells with `rho <= {config['density_threshold_factor']} * dfloor`, in `g`.",
        "- `M_sink_cumulative`: cumulative estimated mass lost through the sink up to this snapshot, in `g`.",
        "- `M_outer_cumulative`: cumulative estimated mass lost through the outer boundary up to this snapshot, in `g`.",
        "- `M_bound_domain`: in-domain mass with `eps_orb < 0`, in `g`.",
        "- `M_unbound_domain`: in-domain mass with `eps_orb >= 0`, in `g`.",
        "- `M_total_accounted`: `M_domain + M_sink_cumulative + M_outer_cumulative`, in `g`, all defined on the density-thresholded debris region.",
        "- `mass_conservation_error`: `M_total_accounted - M_initial`, in `g`, where `M_initial` is the first processed snapshot's in-domain mass.",
        "- `Mdot_sink`: interval-averaged sink loss rate in `g/s`, assigned to the interval ending at this snapshot.",
        "- `Mdot_outer`: interval-averaged outer-boundary loss rate in `g/s`, assigned to the interval ending at this snapshot.",
        "- `dM_bound_domain_dt`: finite-difference time derivative of in-domain bound mass in `g/s`.",
        "- `Mdot_control`: interval-averaged inward mass flux through the control radius in `g/s`.",
        f"- `mass_fraction_e_lt_ecrit`: fraction of in-domain bound mass with `e_est < {config['ecrit']}`.",
        f"- `mass_fraction_Rcirc_lt_Rcrit`: fraction of in-domain bound mass with `R_circ < {config['Rcrit_code']} code units`.",
        "- `mean_j_bound`: exact mass-weighted mean `|j|` of bound in-domain gas in `cm^2/s`.",
        "- `median_j_bound`: histogram-CDF median `|j|` of bound in-domain gas in `cm^2/s`.",
        "- `mean_eps_bound`: exact mass-weighted mean `eps_orb` of bound in-domain gas in `erg/g`.",
        "- `median_eps_bound`: histogram-CDF median `eps_orb` of bound in-domain gas in `erg/g`.",
        "- `M_tail_domain`, `M_tail_bound_inward`, `M_tail_lowj`, `M_tail_high_e`: deep-tail reservoir masses in `g`.",
        "- `tail_mass_fraction_of_domain`, `tail_mass_fraction_of_bound`: deep-tail mass fractions.",
        "- `tail_energy_threshold_code`, `tail_energy_threshold_cgs`: tidal-energy-scale threshold used for the tail selection.",
        "- Unit conversion rows such as `length_cgs`, `mass_cgs`, `time_cgs`, `velocity_cgs`, `specific_eint_cgs`, and `angular_momentum_cgs` are also stored in each per-snapshot file.",
        "- Fallback-shell rates, cumulative fallback masses, and normalized `t^(-5/3)` guide values are stored in each per-snapshot `SUMMARY` block for every configured shell label; by default those labels correspond to `0.5, 1.0, 1.2, 1.5` times `a_mb`.",
        "",
        "### `*.analysis.dat` Histogram Sections",
        "",
        "- `SUMMARY` rows use columns `record name value unit`.",
        "- `HIST` rows use columns `record channel quantity bin_left bin_right bin_center dM dM_dX cumulative_M`.",
        "- `HIST2D` rows use columns `record channel quantity_x quantity_y bin_x_left bin_x_right bin_x_center bin_y_left bin_y_right bin_y_center dM`.",
        "- `channel` includes the existing `in_domain`, `sink_rate`, `outer_rate`, `sink_interval`, `outer_interval`, `in_domain_latest`, `sink_cumulative`, `outer_cumulative`, and `approximate_total_latest` channels plus new conditional channels such as `in_domain_bound_inward`, `in_domain_compressed_proxy`, `shell_inner`, `shell_fallback_f1p0`, and `fallback_rate_inward_f1p0`.",
        "- `quantity` is one of `energy`, `j`, `Be`, or `e` where available.",
        "- `HIST2D` sections are embedded in the per-snapshot `.analysis.dat` file for `energy-j`, `energy-e`, and `j-e` pairs, for `full_in_domain`, `bound_inward`, `compressed_proxy`, and `bound_inward_compressed_proxy` subsets.",
        "- `bin_left`, `bin_right`, `bin_center`: bin geometry in the file's axis units.",
        "- `dM`: mass in the bin in `g`.",
        "- `dM_dX`: `dM / (bin_right - bin_left)`.",
        "- `cumulative_M`: cumulative mass from the lowest bin upward, in `g`.",
        "",
        "For `sink_rate` and `outer_rate`, the histogram weights are instantaneous proxy rates in `g/s` rather than masses; they are stored so reruns can skip already analyzed bin files safely.",
        "The multi-radius fallback-shell channels are also instantaneous proxy rates in `g/s` at that snapshot.",
        "In per-snapshot files, `sink_interval` and `outer_interval` are the estimated mass crossing during the interval ending at that snapshot.",
        "Per-snapshot `SUMMARY` rows also include interval-averaged fallback-shell rates, cumulative fallback-shell masses, and normalized `t^(-5/3)` guide values for every configured shell label.",
        "Any later global time-series or cumulative budget plots should be reconstructed from the per-snapshot files.",
        "",
        "## Notes",
        "",
        f"- Native EOS table cache source: `{runtime.use_translating_frame}` translating-frame metadata present; table interpolation follows the AthenaK native table layout and cache logic.",
        f"- BH mass in code units: `{runtime.bh_mass_code}`.",
        f"- Newtonian `G` in code units: `{runtime.newton_g_code}`.",
        f"- Adopted tidal radius in code units: `{float(scale_info['r_tide_code']):.6e}`.",
        f"- Adopted stellar radius in code units: `{float(scale_info['stellar_radius_code']):.6e}`.",
        f"- Adopted most-bound semimajor axis in code units: `{float(scale_info['a_mb_code']):.6e}`.",
        f"- Fallback-shell labels: `{', '.join(scale_info['fallback_labels'])}`.",
        "- Fallback-shell radii are constructed as `fallback_radius_multiplier * a_mb`.",
        f"- Tail threshold in code units: `{float(scale_info['tail_energy_threshold_code']):.6e}`.",
        f"- Code length, mass, time scales: `{units.length_cgs:.6e} cm`, `{units.mass_cgs:.6e} g`, `{units.time_cgs:.6e} s`.",
        "- Use `plot_analysis.py` to convert these compact `.dat` products into publication-style PNG figures.",
        "",
        "## Additional Scientific Cautions",
        "",
        "- The fallback-shell inward flux is a practical proxy for return across a chosen spherical surface, not necessarily the same as sink accretion.",
        "- The t^(-5/3) guide columns are normalized guide curves only; they are not fitted or enforced laws.",
        "- The compression-based subset is a snapshot-local compressed_proxy diagnostic based on negative velocity divergence, not a true Lagrangian shocked tracer.",
        "- Deep-tail fits are descriptive diagnostics only; they are not proof of a ballistic fallback law.",
        "- The tidal-energy-scale tail threshold is a physically motivated threshold based on `G M_BH R_star / r_tide^2`, not a universal exact boundary.",
    ]
    (output_dir / "README_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def ensure_config(config_path: Path) -> dict:
    if config_path.exists():
        with config_path.open("r", encoding="ascii") as fp:
            user_config = json.load(fp)
        return deep_update(DEFAULT_CONFIG, user_config)
    return DEFAULT_CONFIG


def build_requested_variables(header: SnapshotHeader) -> List[str]:
    base = ["dens", "velx", "vely", "velz"]
    if "eint" in header.variable_names:
        base.append("eint")
    return base


def has_native_eos_table(input_data: Dict[str, Dict[str, str]]) -> bool:
    hydro = input_data.get("hydro", {})
    eos_name = hydro.get("eos", "").strip().lower()
    return "table" in hydro and ("table" in eos_name or eos_name == "saha")


def run_analysis(input_dir: Path, output_dir: Path, config_path: Path) -> None:
    config = ensure_config(config_path)
    resume_existing = bool(config.get("resume_existing_outputs", True))
    if config.get("clean_output_dir", True):
        clean_output_dir(output_dir, keep_snapshot_cache=resume_existing)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_run_level_outputs(output_dir)

    logger = logging.getLogger("analyze_tde")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    numba_threads = config.get("numba_threads")
    if numba_threads is None:
        resolved_numba_threads = max(os.cpu_count() or 1, 1)
    else:
        resolved_numba_threads = max(int(numba_threads), 1)
    set_num_threads(resolved_numba_threads)
    batch_blocks_cfg = config.get("batch_blocks")
    if batch_blocks_cfg is None:
        batch_blocks = min(max(resolved_numba_threads, 8), 32)
    else:
        batch_blocks = max(int(batch_blocks_cfg), 1)
    # Keep the large analysis kernel itself serial in Numba and parallelize safely
    # across independent snapshot batches at the Python level.
    # The main analysis kernel is intentionally single-threaded inside Numba.
    # Keep a modest batch-level thread pool to exploit CPU cores without
    # inflating scheduling and memory overhead.
    analysis_workers = max(1, min(4, resolved_numba_threads))

    try:
        snapshot_files = select_snapshot_files(input_dir, config)
    except RuntimeError:
        cached_snapshots = load_cached_snapshot_analyses(output_dir)
        if cached_snapshots:
            logger.info(
                f"Found {len(cached_snapshots)} cached snapshot .analysis.dat files; "
                "no input .bin snapshots were selected."
            )
            return
        raise
    first_header = read_snapshot_header(snapshot_files[0])
    units = build_units(first_header.input_data)
    runtime0 = build_runtime(first_header, config)
    scale_info0 = resolve_tde_scales(first_header, runtime0, units, config)
    have_face_fluxes = False
    have_eint = "eint" in first_header.variable_names
    have_table_eos = has_native_eos_table(first_header.input_data)
    saha_cache = SahaEOSCache.from_snapshot(first_header.input_data, first_header.path) if have_table_eos and have_eint else None

    edges_code = {
        "energy": derive_bin_edges("energy", config, first_header, runtime0),
        "j": derive_bin_edges("j", config, first_header, runtime0),
        "Be": derive_bin_edges("Be", config, first_header, runtime0),
        "e": derive_bin_edges("e", config, first_header, runtime0),
    }
    bound_energy_max = min(float(edges_code["energy"][-1]), 0.0)
    if bound_energy_max <= float(edges_code["energy"][0]):
        edges_bound_energy_code = edges_code["energy"].copy()
    else:
        edges_bound_energy_code = np.linspace(
            float(edges_code["energy"][0]),
            bound_energy_max,
            edges_code["energy"].size,
            dtype=np.float64,
        )
    edges_physical = {
        name: convert_edges_to_physical(name, edges_code[name], units) for name in edges_code
    }
    edges_2d_code = derive_2d_edges(config, first_header, runtime0)
    edges_2d_physical = {
        name: (
            convert_edges_to_physical(name.split("_")[0], pair[0], units),
            convert_edges_to_physical(name.split("_")[1], pair[1], units),
        )
        for name, pair in edges_2d_code.items()
    }
    fallback_labels = scale_info0["fallback_labels"]

    write_readme(output_dir, units, runtime0, first_header.variable_names, have_eint, have_face_fluxes, config, scale_info0)

    prev_time_code: Optional[float] = None
    prev_bound_code: Optional[float] = None
    prev_sink_rate_code: Optional[np.ndarray] = None
    prev_sink_rate_j_code: Optional[np.ndarray] = None
    prev_sink_rate_be_code: Optional[np.ndarray] = None
    prev_outer_rate_code: Optional[np.ndarray] = None
    prev_outer_rate_j_code: Optional[np.ndarray] = None
    prev_outer_rate_be_code: Optional[np.ndarray] = None
    prev_sink_rate_total_code: Optional[float] = None
    prev_outer_rate_total_code: Optional[float] = None
    prev_control_rate_total_code: Optional[float] = None
    prev_fallback_rate_in_total_code: Optional[np.ndarray] = None
    prev_fallback_rate_out_total_code: Optional[np.ndarray] = None
    prev_fallback_rate_net_total_code: Optional[np.ndarray] = None

    cumulative_sink_mass_code = 0.0
    cumulative_outer_mass_code = 0.0
    cumulative_sink_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
    cumulative_sink_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
    cumulative_sink_Be_code = np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
    cumulative_outer_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
    cumulative_outer_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
    cumulative_outer_Be_code = np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
    cumulative_fallback_in_mass_code = np.zeros(len(fallback_labels), dtype=np.float64)
    cumulative_fallback_out_mass_code = np.zeros(len(fallback_labels), dtype=np.float64)
    cumulative_fallback_net_mass_code = np.zeros(len(fallback_labels), dtype=np.float64)
    fallback_tref_s = np.full(len(fallback_labels), math.nan, dtype=np.float64)
    fallback_mdot_ref_gs = np.full(len(fallback_labels), math.nan, dtype=np.float64)

    initial_mass_code: Optional[float] = None
    reference_units = units

    total_snapshots = len(snapshot_files)
    for file_counter, snapshot_path in enumerate(snapshot_files):
        header = read_snapshot_header(snapshot_path)
        current_units = build_units(header.input_data)
        if current_units != reference_units:
            raise RuntimeError("Unit system changed across snapshots; mixed-unit output is not supported.")
        runtime = build_runtime(header, config)
        scale_info = resolve_tde_scales(header, runtime, units, config)
        if scale_info["fallback_labels"] != fallback_labels:
            raise RuntimeError("fallback_radius_multipliers changed across snapshots; mixed shell labels are not supported.")
        density_floor_code = float(header.input_data.get("hydro", {}).get("dfloor", "0.0"))
        density_threshold_code = max(float(config.get("density_threshold_factor", 10.0)) * density_floor_code, 0.0)
        if tuple(header.variable_names) != tuple(first_header.variable_names):
            logger.warning(f"Warning: variable set changed in {snapshot_path.name}; processing common fields only.")

        cached_output_path = output_dir / f"{snapshot_path.stem}.analysis.dat"
        used_cache = False
        cached = None
        if resume_existing and cached_output_path.exists():
            try:
                cached = parse_snapshot_analysis(cached_output_path)
                if cached_snapshot_is_valid(
                    cached,
                    snapshot_path,
                    header.snapshot_index,
                    edges_physical,
                    need_be=(saha_cache is not None and have_eint),
                    config=config,
                    runtime=runtime,
                    scale_info=scale_info,
                    density_floor_code=density_floor_code,
                    density_threshold_code=density_threshold_code,
                    output_dir=output_dir,
                ):
                    used_cache = True
            except Exception:
                used_cache = False

        if used_cache:
            logger.info(
                f"[{file_counter + 1}/{total_snapshots}] skipping cached {snapshot_path.name}"
            )
        elif cached_output_path.exists():
            logger.info(
                f"[{file_counter + 1}/{total_snapshots}] reprocessing {snapshot_path.name}"
            )
        else:
            logger.info(
                f"[{file_counter + 1}/{total_snapshots}] processing {snapshot_path.name}"
            )

        if used_cache and cached is not None:
            time_s = float(cached.summary["snapshot_time"])
            m_domain_g = float(cached.summary["M_domain"])
            m_bound_g = float(cached.summary["M_bound_domain"])
            m_unbound_g = float(cached.summary["M_unbound_domain"])
            hist_E_g = cached.histograms[("in_domain", "energy")][:, 3].copy()
            hist_j_g = cached.histograms[("in_domain", "j")][:, 3].copy()
            hist_e_g = cached.histograms[("in_domain", "e")][:, 3].copy()
            hist_Be_g = (
                cached.histograms[("in_domain", "Be")][:, 3].copy()
                if ("in_domain", "Be") in cached.histograms
                else np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            )
            sink_rate_E_code = cached.histograms[("sink_rate", "energy")][:, 3].copy() * units.time_cgs / units.mass_cgs
            sink_rate_j_code = cached.histograms[("sink_rate", "j")][:, 3].copy() * units.time_cgs / units.mass_cgs
            sink_rate_Be_code = (
                cached.histograms[("sink_rate", "Be")][:, 3].copy() * units.time_cgs / units.mass_cgs
                if ("sink_rate", "Be") in cached.histograms
                else np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            )
            outer_rate_E_code = cached.histograms[("outer_rate", "energy")][:, 3].copy() * units.time_cgs / units.mass_cgs
            outer_rate_j_code = cached.histograms[("outer_rate", "j")][:, 3].copy() * units.time_cgs / units.mass_cgs
            outer_rate_Be_code = (
                cached.histograms[("outer_rate", "Be")][:, 3].copy() * units.time_cgs / units.mass_cgs
                if ("outer_rate", "Be") in cached.histograms
                else np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            )
            sink_rate_total_code = float(cached.summary["sink_rate_total_snapshot"]) * units.time_cgs / units.mass_cgs
            outer_rate_total_code = float(cached.summary["outer_rate_total_snapshot"]) * units.time_cgs / units.mass_cgs
            control_rate_total_code = float(cached.summary["control_rate_total_snapshot"]) * units.time_cgs / units.mass_cgs
            fallback_rate_in_total_code = np.array([
                float(cached.summary[f"fallback_rate_inward_total_snapshot_{label}"]) * units.time_cgs / units.mass_cgs
                for label in fallback_labels
            ], dtype=np.float64)
            fallback_rate_out_total_code = np.array([
                float(cached.summary[f"fallback_rate_outward_total_snapshot_{label}"]) * units.time_cgs / units.mass_cgs
                for label in fallback_labels
            ], dtype=np.float64)
            fallback_rate_net_total_code = np.array([
                float(cached.summary[f"fallback_rate_net_total_snapshot_{label}"]) * units.time_cgs / units.mass_cgs
                for label in fallback_labels
            ], dtype=np.float64)
            m_domain_code = m_domain_g / units.mass_cgs
            m_bound_code = m_bound_g / units.mass_cgs
            m_unbound_code = m_unbound_g / units.mass_cgs
            mean_j_bound_cgs = float(cached.summary["mean_j_bound"])
            median_j_bound_cgs = float(cached.summary["median_j_bound"])
            mean_eps_bound_cgs = float(cached.summary["mean_eps_bound"])
            median_eps_bound_cgs = float(cached.summary["median_eps_bound"])
            frac_ecrit = float(cached.summary["mass_fraction_e_lt_ecrit"])
            frac_rcirc = float(cached.summary["mass_fraction_Rcirc_lt_Rcrit"])
        else:
            requested_variables = build_requested_variables(header)
            compression_threshold_cfg = config.get("compression_threshold_code")
            if compression_threshold_cfg is None:
                compression_parts: List[np.ndarray] = []
                for batch in iter_snapshot_batches(header, ("dens", "velx", "vely", "velz"), batch_blocks):
                    neg_values, neg_counts = _collect_negative_divergence_batch(
                        dens=batch.fields["dens"],
                        velx=batch.fields["velx"],
                        vely=batch.fields["vely"],
                        velz=batch.fields["velz"],
                        x1min=batch.x1min,
                        x2min=batch.x2min,
                        x3min=batch.x3min,
                        dx1=batch.dx1,
                        dx2=batch.dx2,
                        dx3=batch.dx3,
                        bhx=runtime.bh_x_code,
                        bhy=runtime.bh_y_code,
                        bhz=runtime.bh_z_code,
                        sink_radius=runtime.sink_radius_code,
                        density_threshold_code=density_threshold_code,
                    )
                    for block_index, count in enumerate(neg_counts):
                        if count > 0:
                            compression_parts.append(neg_values[block_index, :count].copy())
                if compression_parts:
                    compression_values = np.concatenate(compression_parts)
                    compression_threshold_code = float(
                        np.percentile(compression_values, float(config.get("compression_percentile", 95.0)))
                    )
                else:
                    compression_threshold_code = math.inf
            else:
                compression_threshold_code = max(float(compression_threshold_cfg), 0.0)

            m_domain_code = 0.0
            m_bound_code = 0.0
            m_unbound_code = 0.0
            m_ecrit_code = 0.0
            m_rcirc_code = 0.0
            m_bound_high_e_code = 0.0
            sum_j_bound_code = 0.0
            sum_eps_bound_code = 0.0
            m_tail_domain_code = 0.0
            m_tail_bound_inward_code = 0.0
            m_tail_lowj_code = 0.0
            m_tail_high_e_code = 0.0
            m_compressed_proxy_code = 0.0
            m_bound_inward_compressed_proxy_code = 0.0

            hist_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
            hist_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
            hist_Be_code = np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            hist_e_code = np.zeros(edges_code["e"].size - 1, dtype=np.float64)
            bound_hist_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
            bound_hist_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
            subset_hist_E_code = np.zeros((NSUBSET_1D, edges_code["energy"].size - 1), dtype=np.float64)
            subset_hist_j_code = np.zeros((NSUBSET_1D, edges_code["j"].size - 1), dtype=np.float64)
            subset_hist_Be_code = np.zeros((NSUBSET_1D, edges_code["Be"].size - 1), dtype=np.float64)
            subset_hist_e_code = np.zeros((NSUBSET_1D, edges_code["e"].size - 1), dtype=np.float64)
            compressed_hist_E_code = np.zeros((NCOMPRESSED_SUBSET, edges_code["energy"].size - 1), dtype=np.float64)
            compressed_hist_j_code = np.zeros((NCOMPRESSED_SUBSET, edges_code["j"].size - 1), dtype=np.float64)
            compressed_hist_Be_code = np.zeros((NCOMPRESSED_SUBSET, edges_code["Be"].size - 1), dtype=np.float64)
            compressed_hist_e_code = np.zeros((NCOMPRESSED_SUBSET, edges_code["e"].size - 1), dtype=np.float64)
            shell_hist_E_code = np.zeros((len(fallback_labels) + 2, edges_code["energy"].size - 1), dtype=np.float64)
            shell_hist_j_code = np.zeros((len(fallback_labels) + 2, edges_code["j"].size - 1), dtype=np.float64)
            shell_hist_Be_code = np.zeros((len(fallback_labels) + 2, edges_code["Be"].size - 1), dtype=np.float64)
            shell_hist_e_code = np.zeros((len(fallback_labels) + 2, edges_code["e"].size - 1), dtype=np.float64)
            fallback_rate_in_E_code = np.zeros((len(fallback_labels), edges_code["energy"].size - 1), dtype=np.float64)
            fallback_rate_in_j_code = np.zeros((len(fallback_labels), edges_code["j"].size - 1), dtype=np.float64)
            fallback_rate_in_Be_code = np.zeros((len(fallback_labels), edges_code["Be"].size - 1), dtype=np.float64)
            fallback_rate_in_e_code = np.zeros((len(fallback_labels), edges_code["e"].size - 1), dtype=np.float64)
            fallback_rate_out_E_code = np.zeros((len(fallback_labels), edges_code["energy"].size - 1), dtype=np.float64)
            fallback_rate_out_j_code = np.zeros((len(fallback_labels), edges_code["j"].size - 1), dtype=np.float64)
            fallback_rate_out_Be_code = np.zeros((len(fallback_labels), edges_code["Be"].size - 1), dtype=np.float64)
            fallback_rate_out_e_code = np.zeros((len(fallback_labels), edges_code["e"].size - 1), dtype=np.float64)
            fallback_rate_net_E_code = np.zeros((len(fallback_labels), edges_code["energy"].size - 1), dtype=np.float64)
            fallback_rate_net_j_code = np.zeros((len(fallback_labels), edges_code["j"].size - 1), dtype=np.float64)
            fallback_rate_net_Be_code = np.zeros((len(fallback_labels), edges_code["Be"].size - 1), dtype=np.float64)
            fallback_rate_net_e_code = np.zeros((len(fallback_labels), edges_code["e"].size - 1), dtype=np.float64)
            fallback_rate_in_total_code = np.zeros(len(fallback_labels), dtype=np.float64)
            fallback_rate_out_total_code = np.zeros(len(fallback_labels), dtype=np.float64)
            fallback_rate_net_total_code = np.zeros(len(fallback_labels), dtype=np.float64)
            hist2d_energy_j_code = np.zeros((NHIST2D_SUBSET,) + tuple(arr.size - 1 for arr in edges_2d_code["energy_j"]), dtype=np.float64)
            hist2d_energy_e_code = np.zeros((NHIST2D_SUBSET,) + tuple(arr.size - 1 for arr in edges_2d_code["energy_e"]), dtype=np.float64)
            hist2d_j_e_code = np.zeros((NHIST2D_SUBSET,) + tuple(arr.size - 1 for arr in edges_2d_code["j_e"]), dtype=np.float64)
            sink_rate_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
            sink_rate_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
            sink_rate_Be_code = np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            outer_rate_E_code = np.zeros(edges_code["energy"].size - 1, dtype=np.float64)
            outer_rate_j_code = np.zeros(edges_code["j"].size - 1, dtype=np.float64)
            outer_rate_Be_code = np.zeros(edges_code["Be"].size - 1, dtype=np.float64)
            sink_rate_total_code = 0.0
            outer_rate_total_code = 0.0
            control_rate_total_code = 0.0
            tail_j_threshold_cfg = config.get("tail_j_threshold_code")
            if tail_j_threshold_cfg is None:
                have_tail_j_threshold = False
                tail_j_threshold_code = math.nan
            else:
                have_tail_j_threshold = True
                tail_j_threshold_code = float(tail_j_threshold_cfg)

            if saha_cache is None:
                density_unit_cgs = 1.0
                specific_eint_unit_cgs = 1.0
                saha_logrho0 = 0.0
                saha_logrho_min = 0.0
                saha_logrho_max = 0.0
                saha_inv_dlogrho = 1.0
                saha_table_logeps = np.zeros((2, 2), dtype=np.float64)
                saha_cache_press = np.zeros((2, 2), dtype=np.float64)
                saha_logeps_min = 0.0
                saha_logeps_max = 0.0
                saha_inv_dlogeps = 1.0
            else:
                density_unit_cgs = saha_cache.density_unit_cgs
                specific_eint_unit_cgs = saha_cache.specific_eint_unit_cgs
                saha_logrho0 = float(saha_cache.logrho[0])
                saha_logrho_min = float(saha_cache.logrho[0])
                saha_logrho_max = float(saha_cache.logrho[-1])
                saha_inv_dlogrho = float(saha_cache.inv_dlogrho)
                saha_table_logeps = saha_cache.table_logeps
                saha_cache_press = saha_cache.thermo_cache[SAHA_CACHE_PRESS]
                saha_logeps_min = float(saha_cache.logeps_min)
                saha_logeps_max = float(saha_cache.logeps_max)
                saha_inv_dlogeps = float(saha_cache.inv_dlogeps)

            fallback_radii_code = np.asarray(scale_info["fallback_radii_code"], dtype=np.float64)

            def process_loaded_batch(batch: SnapshotBatch) -> Tuple[np.ndarray, ...]:
                dens = batch.fields["dens"]
                velx = batch.fields["velx"]
                vely = batch.fields["vely"]
                velz = batch.fields["velz"]
                eint = batch.fields.get("eint", np.zeros_like(dens))
                return _process_batch(
                    dens=dens,
                    velx=velx,
                    vely=vely,
                    velz=velz,
                    eint=eint,
                    have_eint="eint" in batch.fields,
                    x1min=batch.x1min,
                    x2min=batch.x2min,
                    x3min=batch.x3min,
                    dx1=batch.dx1,
                    dx2=batch.dx2,
                    dx3=batch.dx3,
                    touch_xmin=batch.touch_xmin,
                    touch_xmax=batch.touch_xmax,
                    touch_ymin=batch.touch_ymin,
                    touch_ymax=batch.touch_ymax,
                    touch_zmin=batch.touch_zmin,
                    touch_zmax=batch.touch_zmax,
                    bhx=runtime.bh_x_code,
                    bhy=runtime.bh_y_code,
                    bhz=runtime.bh_z_code,
                    bhvx=runtime.bh_vx_code,
                    bhvy=runtime.bh_vy_code,
                    bhvz=runtime.bh_vz_code,
                    gm_code=runtime.gm_code,
                    soft2=runtime.bh_softening_code**2,
                    sink_radius=runtime.sink_radius_code,
                    sink_shell_cells=float(config["sink_shell_cells"]),
                    outer_shell_cells=float(config["outer_shell_cells"]),
                    control_radius=float(config["control_radius_code"]),
                    control_shell_cells=float(config["control_shell_cells"]),
                    ecrit=float(config["ecrit"]),
                    rcirc_crit=float(config["Rcrit_code"]),
                    density_threshold_code=density_threshold_code,
                    density_unit_cgs=density_unit_cgs,
                    specific_eint_unit_cgs=specific_eint_unit_cgs,
                    saha_logrho0=saha_logrho0,
                    saha_logrho_min=saha_logrho_min,
                    saha_logrho_max=saha_logrho_max,
                    saha_inv_dlogrho=saha_inv_dlogrho,
                    saha_table_logeps=saha_table_logeps,
                    saha_cache_press=saha_cache_press,
                    saha_logeps_min=saha_logeps_min,
                    saha_logeps_max=saha_logeps_max,
                    saha_inv_dlogeps=saha_inv_dlogeps,
                    edges_E=edges_code["energy"],
                    edges_bound_E=edges_bound_energy_code,
                    edges_j=edges_code["j"],
                    edges_Be=edges_code["Be"],
                    edges_e=edges_code["e"],
                    edges_2d_energy_j_x=edges_2d_code["energy_j"][0],
                    edges_2d_energy_j_y=edges_2d_code["energy_j"][1],
                    edges_2d_energy_e_x=edges_2d_code["energy_e"][0],
                    edges_2d_energy_e_y=edges_2d_code["energy_e"][1],
                    edges_2d_j_e_x=edges_2d_code["j_e"][0],
                    edges_2d_j_e_y=edges_2d_code["j_e"][1],
                    fallback_radii=fallback_radii_code,
                    fallback_shell_cells=float(config["fallback_shell_cells"]),
                    inner_shell_radius=float(scale_info["inner_shell_radius_code"]),
                    inner_shell_cells=float(config["inner_shell_cells"]),
                    outer_stream_radius=float(scale_info["outer_stream_radius_code"]),
                    outer_stream_shell_cells=float(config["outer_stream_shell_cells"]),
                    compression_threshold_code=float(compression_threshold_code),
                    tail_energy_threshold_code=float(scale_info["tail_energy_threshold_code"]),
                    tail_j_threshold_code=float(tail_j_threshold_code if have_tail_j_threshold else 0.0),
                    tail_e_threshold=float(config["tail_e_threshold"]),
                    have_tail_j_threshold=have_tail_j_threshold,
                    need_be=(saha_cache is not None and "eint" in batch.fields),
                )

            def accumulate_result(result: Tuple[np.ndarray, ...]) -> None:
                nonlocal m_domain_code, m_bound_code, m_unbound_code
                nonlocal m_ecrit_code, m_rcirc_code, m_bound_high_e_code
                nonlocal sum_j_bound_code, sum_eps_bound_code
                nonlocal m_tail_domain_code, m_tail_bound_inward_code
                nonlocal m_tail_lowj_code, m_tail_high_e_code
                nonlocal m_compressed_proxy_code, m_bound_inward_compressed_proxy_code
                nonlocal hist_E_code, hist_j_code, hist_Be_code, hist_e_code
                nonlocal bound_hist_E_code, bound_hist_j_code
                nonlocal subset_hist_E_code, subset_hist_j_code, subset_hist_Be_code, subset_hist_e_code
                nonlocal compressed_hist_E_code, compressed_hist_j_code, compressed_hist_Be_code, compressed_hist_e_code
                nonlocal shell_hist_E_code, shell_hist_j_code, shell_hist_Be_code, shell_hist_e_code
                nonlocal fallback_rate_in_E_code, fallback_rate_in_j_code, fallback_rate_in_Be_code, fallback_rate_in_e_code
                nonlocal fallback_rate_out_E_code, fallback_rate_out_j_code, fallback_rate_out_Be_code, fallback_rate_out_e_code
                nonlocal fallback_rate_net_E_code, fallback_rate_net_j_code, fallback_rate_net_Be_code, fallback_rate_net_e_code
                nonlocal fallback_rate_in_total_code, fallback_rate_out_total_code, fallback_rate_net_total_code
                nonlocal hist2d_energy_j_code, hist2d_energy_e_code, hist2d_j_e_code
                nonlocal sink_rate_E_code, sink_rate_j_code, sink_rate_Be_code
                nonlocal outer_rate_E_code, outer_rate_j_code, outer_rate_Be_code
                nonlocal sink_rate_total_code, outer_rate_total_code, control_rate_total_code
                (
                    m_domain,
                    m_bound,
                    m_unbound,
                    m_ecrit,
                    m_rcirc,
                    m_bound_high_e,
                    sum_j_bound,
                    sum_eps_bound,
                    m_tail_domain,
                    m_tail_bound_inward,
                    m_tail_lowj,
                    m_tail_high_e,
                    m_compressed_proxy,
                    m_bound_inward_compressed_proxy,
                    hist_E,
                    hist_j,
                    hist_Be,
                    hist_e,
                    bound_hist_E,
                    bound_hist_j,
                    subset_hist_E,
                    subset_hist_j,
                    subset_hist_Be,
                    subset_hist_e,
                    compressed_hist_E,
                    compressed_hist_j,
                    compressed_hist_Be,
                    compressed_hist_e,
                    shell_hist_E,
                    shell_hist_j,
                    shell_hist_Be,
                    shell_hist_e,
                    fallback_in_E,
                    fallback_in_j,
                    fallback_in_Be,
                    fallback_in_e,
                    fallback_out_E,
                    fallback_out_j,
                    fallback_out_Be,
                    fallback_out_e,
                    fallback_net_E,
                    fallback_net_j,
                    fallback_net_Be,
                    fallback_net_e,
                    fallback_in_total,
                    fallback_out_total,
                    fallback_net_total,
                    hist2d_energy_j,
                    hist2d_energy_e,
                    hist2d_j_e,
                    sink_rate_E,
                    sink_rate_j,
                    sink_rate_Be,
                    outer_rate_E,
                    outer_rate_j,
                    outer_rate_Be,
                    sink_rate_total,
                    outer_rate_total,
                    control_rate_total,
                ) = result
                m_domain_code += float(m_domain)
                m_bound_code += float(m_bound)
                m_unbound_code += float(m_unbound)
                m_ecrit_code += float(m_ecrit)
                m_rcirc_code += float(m_rcirc)
                m_bound_high_e_code += float(m_bound_high_e)
                sum_j_bound_code += float(sum_j_bound)
                sum_eps_bound_code += float(sum_eps_bound)
                m_tail_domain_code += float(m_tail_domain)
                m_tail_bound_inward_code += float(m_tail_bound_inward)
                m_tail_lowj_code += float(m_tail_lowj)
                m_tail_high_e_code += float(m_tail_high_e)
                m_compressed_proxy_code += float(m_compressed_proxy)
                m_bound_inward_compressed_proxy_code += float(m_bound_inward_compressed_proxy)
                hist_E_code += hist_E
                hist_j_code += hist_j
                hist_Be_code += hist_Be
                hist_e_code += hist_e
                bound_hist_E_code += bound_hist_E
                bound_hist_j_code += bound_hist_j
                subset_hist_E_code += subset_hist_E
                subset_hist_j_code += subset_hist_j
                subset_hist_Be_code += subset_hist_Be
                subset_hist_e_code += subset_hist_e
                compressed_hist_E_code += compressed_hist_E
                compressed_hist_j_code += compressed_hist_j
                compressed_hist_Be_code += compressed_hist_Be
                compressed_hist_e_code += compressed_hist_e
                shell_hist_E_code += shell_hist_E
                shell_hist_j_code += shell_hist_j
                shell_hist_Be_code += shell_hist_Be
                shell_hist_e_code += shell_hist_e
                fallback_rate_in_E_code += fallback_in_E
                fallback_rate_in_j_code += fallback_in_j
                fallback_rate_in_Be_code += fallback_in_Be
                fallback_rate_in_e_code += fallback_in_e
                fallback_rate_out_E_code += fallback_out_E
                fallback_rate_out_j_code += fallback_out_j
                fallback_rate_out_Be_code += fallback_out_Be
                fallback_rate_out_e_code += fallback_out_e
                fallback_rate_net_E_code += fallback_net_E
                fallback_rate_net_j_code += fallback_net_j
                fallback_rate_net_Be_code += fallback_net_Be
                fallback_rate_net_e_code += fallback_net_e
                fallback_rate_in_total_code += fallback_in_total
                fallback_rate_out_total_code += fallback_out_total
                fallback_rate_net_total_code += fallback_net_total
                hist2d_energy_j_code += hist2d_energy_j
                hist2d_energy_e_code += hist2d_energy_e
                hist2d_j_e_code += hist2d_j_e
                sink_rate_E_code += sink_rate_E
                sink_rate_j_code += sink_rate_j
                sink_rate_Be_code += sink_rate_Be
                outer_rate_E_code += outer_rate_E
                outer_rate_j_code += outer_rate_j
                outer_rate_Be_code += outer_rate_Be
                sink_rate_total_code += float(sink_rate_total)
                outer_rate_total_code += float(outer_rate_total)
                control_rate_total_code += float(control_rate_total)

            batch_iter = iter_snapshot_batches(header, requested_variables, batch_blocks)
            try:
                first_batch = next(batch_iter)
            except StopIteration:
                pass
            else:
                # Compile and execute the heavy kernel once synchronously, then
                # run the remaining independent batches concurrently.
                accumulate_result(process_loaded_batch(first_batch))
                if analysis_workers <= 1:
                    for batch in batch_iter:
                        accumulate_result(process_loaded_batch(batch))
                else:
                    with ThreadPoolExecutor(max_workers=analysis_workers) as executor:
                        pending = set()

                        def submit_available() -> None:
                            while len(pending) < analysis_workers:
                                try:
                                    batch = next(batch_iter)
                                except StopIteration:
                                    break
                                pending.add(executor.submit(process_loaded_batch, batch))

                        submit_available()
                        while pending:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for future in done:
                                accumulate_result(future.result())
                            submit_available()

            mean_j_bound_code = sum_j_bound_code / m_bound_code if m_bound_code > 0.0 else math.nan
            mean_eps_bound_code = sum_eps_bound_code / m_bound_code if m_bound_code > 0.0 else math.nan
            median_j_bound_code = hist_median_from_edges(edges_code["j"], bound_hist_j_code)
            median_eps_bound_code = hist_median_from_edges(edges_bound_energy_code, bound_hist_E_code)
            frac_ecrit = m_ecrit_code / m_bound_code if m_bound_code > 0.0 else math.nan
            frac_rcirc = m_rcirc_code / m_bound_code if m_bound_code > 0.0 else math.nan
            if config.get("tail_j_threshold_code") is None:
                tail_j_threshold_code = hist_quantile_from_edges(
                    edges_code["j"],
                    bound_hist_j_code,
                    float(config.get("tail_j_percentile", 25.0)),
                )
                if math.isfinite(tail_j_threshold_code):
                    m_tail_lowj_code = sum_hist2d_region(
                        hist2d_energy_j_code[HIST2D_FULL],
                        edges_2d_code["energy_j"][0],
                        edges_2d_code["energy_j"][1],
                        x_upper=float(scale_info["tail_energy_threshold_code"]),
                        y_upper=tail_j_threshold_code,
                    )
            low_j_mass_fraction_bound = (
                hist_fraction_below(edges_code["j"], bound_hist_j_code, tail_j_threshold_code)
                if m_bound_code > 0.0 and math.isfinite(tail_j_threshold_code)
                else math.nan
            )
            high_e_mass_fraction_bound = m_bound_high_e_code / m_bound_code if m_bound_code > 0.0 else math.nan
            mass_fraction_compressed_proxy = m_compressed_proxy_code / m_domain_code if m_domain_code > 0.0 else math.nan
            tail_mass_fraction_of_domain = m_tail_domain_code / m_domain_code if m_domain_code > 0.0 else math.nan
            tail_mass_fraction_of_bound = m_tail_domain_code / m_bound_code if m_bound_code > 0.0 else math.nan
            if bool(config.get("fit_tail_alpha", True)):
                tail_alpha, tail_fit_nbins, tail_fit_energy_min, tail_fit_energy_max, tail_fit_r2 = fit_tail_alpha_from_hist(
                    edges_code["energy"],
                    hist_E_code,
                    float(scale_info["tail_energy_threshold_code"]),
                    int(config.get("tail_fit_min_populated_bins", 8)),
                )
            else:
                tail_alpha = math.nan
                tail_fit_nbins = 0
                tail_fit_energy_min = math.nan
                tail_fit_energy_max = math.nan
                tail_fit_r2 = math.nan

            time_s = header.time_code * units.time_cgs
            m_domain_g = m_domain_code * units.mass_cgs
            m_bound_g = m_bound_code * units.mass_cgs
            m_unbound_g = m_unbound_code * units.mass_cgs
            mean_j_bound_cgs = mean_j_bound_code * units.angular_momentum_cgs if not math.isnan(mean_j_bound_code) else math.nan
            median_j_bound_cgs = median_j_bound_code * units.angular_momentum_cgs if not math.isnan(median_j_bound_code) else math.nan
            mean_eps_bound_cgs = mean_eps_bound_code * units.specific_eint_cgs if not math.isnan(mean_eps_bound_code) else math.nan
            median_eps_bound_cgs = median_eps_bound_code * units.specific_eint_cgs if not math.isnan(median_eps_bound_code) else math.nan
            hist_E_g = hist_E_code * units.mass_cgs
            hist_j_g = hist_j_code * units.mass_cgs
            hist_Be_g = hist_Be_code * units.mass_cgs
            hist_e_g = hist_e_code * units.mass_cgs
            subset_hist_E_g = subset_hist_E_code * units.mass_cgs
            subset_hist_j_g = subset_hist_j_code * units.mass_cgs
            subset_hist_Be_g = subset_hist_Be_code * units.mass_cgs
            subset_hist_e_g = subset_hist_e_code * units.mass_cgs
            compressed_hist_E_g = compressed_hist_E_code * units.mass_cgs
            compressed_hist_j_g = compressed_hist_j_code * units.mass_cgs
            compressed_hist_Be_g = compressed_hist_Be_code * units.mass_cgs
            compressed_hist_e_g = compressed_hist_e_code * units.mass_cgs
            shell_hist_E_g = shell_hist_E_code * units.mass_cgs
            shell_hist_j_g = shell_hist_j_code * units.mass_cgs
            shell_hist_Be_g = shell_hist_Be_code * units.mass_cgs
            shell_hist_e_g = shell_hist_e_code * units.mass_cgs
            fallback_rate_in_E_gs = fallback_rate_in_E_code * units.mass_cgs / units.time_cgs
            fallback_rate_in_j_gs = fallback_rate_in_j_code * units.mass_cgs / units.time_cgs
            fallback_rate_in_Be_gs = fallback_rate_in_Be_code * units.mass_cgs / units.time_cgs
            fallback_rate_in_e_gs = fallback_rate_in_e_code * units.mass_cgs / units.time_cgs
            fallback_rate_out_E_gs = fallback_rate_out_E_code * units.mass_cgs / units.time_cgs
            fallback_rate_out_j_gs = fallback_rate_out_j_code * units.mass_cgs / units.time_cgs
            fallback_rate_out_Be_gs = fallback_rate_out_Be_code * units.mass_cgs / units.time_cgs
            fallback_rate_out_e_gs = fallback_rate_out_e_code * units.mass_cgs / units.time_cgs
            fallback_rate_net_E_gs = fallback_rate_net_E_code * units.mass_cgs / units.time_cgs
            fallback_rate_net_j_gs = fallback_rate_net_j_code * units.mass_cgs / units.time_cgs
            fallback_rate_net_Be_gs = fallback_rate_net_Be_code * units.mass_cgs / units.time_cgs
            fallback_rate_net_e_gs = fallback_rate_net_e_code * units.mass_cgs / units.time_cgs

            histogram_sections: List[Tuple[str, str, np.ndarray, np.ndarray]] = [
                ("in_domain", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), hist_E_g),
                ("in_domain", "j", convert_edges_to_physical("j", edges_code["j"], units), hist_j_g),
                ("in_domain", "e", convert_edges_to_physical("e", edges_code["e"], units), hist_e_g),
                ("sink_rate", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), sink_rate_E_code * units.mass_cgs / units.time_cgs),
                ("sink_rate", "j", convert_edges_to_physical("j", edges_code["j"], units), sink_rate_j_code * units.mass_cgs / units.time_cgs),
                ("outer_rate", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), outer_rate_E_code * units.mass_cgs / units.time_cgs),
                ("outer_rate", "j", convert_edges_to_physical("j", edges_code["j"], units), outer_rate_j_code * units.mass_cgs / units.time_cgs),
            ]
            subset_names = [
                "in_domain_inward",
                "in_domain_outward",
                "in_domain_bound_inward",
                "in_domain_bound_outward",
                "in_domain_unbound",
            ]
            for subset_index, channel in enumerate(subset_names):
                histogram_sections.extend([
                    (channel, "energy", convert_edges_to_physical("energy", edges_code["energy"], units), subset_hist_E_g[subset_index]),
                    (channel, "j", convert_edges_to_physical("j", edges_code["j"], units), subset_hist_j_g[subset_index]),
                    (channel, "e", convert_edges_to_physical("e", edges_code["e"], units), subset_hist_e_g[subset_index]),
                ])
                if saha_cache is not None and have_eint:
                    histogram_sections.append(
                        (channel, "Be", convert_edges_to_physical("Be", edges_code["Be"], units), subset_hist_Be_g[subset_index])
                    )
            compressed_names = [
                "in_domain_compressed_proxy",
                "in_domain_bound_inward_compressed_proxy",
            ]
            for subset_index, channel in enumerate(compressed_names):
                histogram_sections.extend([
                    (channel, "energy", convert_edges_to_physical("energy", edges_code["energy"], units), compressed_hist_E_g[subset_index]),
                    (channel, "j", convert_edges_to_physical("j", edges_code["j"], units), compressed_hist_j_g[subset_index]),
                    (channel, "e", convert_edges_to_physical("e", edges_code["e"], units), compressed_hist_e_g[subset_index]),
                ])
                if saha_cache is not None and have_eint:
                    histogram_sections.append(
                        (channel, "Be", convert_edges_to_physical("Be", edges_code["Be"], units), compressed_hist_Be_g[subset_index])
                    )
            histogram_sections.extend([
                ("shell_inner", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), shell_hist_E_g[0]),
                ("shell_inner", "j", convert_edges_to_physical("j", edges_code["j"], units), shell_hist_j_g[0]),
                ("shell_inner", "e", convert_edges_to_physical("e", edges_code["e"], units), shell_hist_e_g[0]),
                ("shell_outer_stream", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), shell_hist_E_g[len(fallback_labels) + 1]),
                ("shell_outer_stream", "j", convert_edges_to_physical("j", edges_code["j"], units), shell_hist_j_g[len(fallback_labels) + 1]),
                ("shell_outer_stream", "e", convert_edges_to_physical("e", edges_code["e"], units), shell_hist_e_g[len(fallback_labels) + 1]),
            ])
            if saha_cache is not None and have_eint:
                histogram_sections.extend([
                    ("in_domain", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), hist_Be_g),
                    ("sink_rate", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), sink_rate_Be_code * units.mass_cgs / units.time_cgs),
                    ("outer_rate", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), outer_rate_Be_code * units.mass_cgs / units.time_cgs),
                    ("shell_inner", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), shell_hist_Be_g[0]),
                    ("shell_outer_stream", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), shell_hist_Be_g[len(fallback_labels) + 1]),
                ])
            for shell_index, label in enumerate(fallback_labels):
                shell_row_index = shell_index + 1
                histogram_sections.extend([
                    (f"shell_fallback_{label}", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), shell_hist_E_g[shell_row_index]),
                    (f"shell_fallback_{label}", "j", convert_edges_to_physical("j", edges_code["j"], units), shell_hist_j_g[shell_row_index]),
                    (f"shell_fallback_{label}", "e", convert_edges_to_physical("e", edges_code["e"], units), shell_hist_e_g[shell_row_index]),
                    (f"fallback_rate_inward_{label}", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), fallback_rate_in_E_gs[shell_index]),
                    (f"fallback_rate_inward_{label}", "j", convert_edges_to_physical("j", edges_code["j"], units), fallback_rate_in_j_gs[shell_index]),
                    (f"fallback_rate_inward_{label}", "e", convert_edges_to_physical("e", edges_code["e"], units), fallback_rate_in_e_gs[shell_index]),
                    (f"fallback_rate_outward_{label}", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), fallback_rate_out_E_gs[shell_index]),
                    (f"fallback_rate_outward_{label}", "j", convert_edges_to_physical("j", edges_code["j"], units), fallback_rate_out_j_gs[shell_index]),
                    (f"fallback_rate_outward_{label}", "e", convert_edges_to_physical("e", edges_code["e"], units), fallback_rate_out_e_gs[shell_index]),
                    (f"fallback_rate_net_{label}", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), fallback_rate_net_E_gs[shell_index]),
                    (f"fallback_rate_net_{label}", "j", convert_edges_to_physical("j", edges_code["j"], units), fallback_rate_net_j_gs[shell_index]),
                    (f"fallback_rate_net_{label}", "e", convert_edges_to_physical("e", edges_code["e"], units), fallback_rate_net_e_gs[shell_index]),
                ])
                if saha_cache is not None and have_eint:
                    histogram_sections.extend([
                        (f"shell_fallback_{label}", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), shell_hist_Be_g[shell_row_index]),
                        (f"fallback_rate_inward_{label}", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), fallback_rate_in_Be_gs[shell_index]),
                        (f"fallback_rate_outward_{label}", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), fallback_rate_out_Be_gs[shell_index]),
                        (f"fallback_rate_net_{label}", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), fallback_rate_net_Be_gs[shell_index]),
                    ])

        if initial_mass_code is None:
            initial_mass_code = m_domain_code

        interval_mid_s = math.nan
        mdot_sink_gs = math.nan
        mdot_outer_gs = math.nan
        dmbound_dt_gs = math.nan
        mdot_control_gs = math.nan
        mdot_fallback_in_gs = np.full(len(fallback_labels), math.nan, dtype=np.float64)
        mdot_fallback_out_gs = np.full(len(fallback_labels), math.nan, dtype=np.float64)
        mdot_fallback_net_gs = np.full(len(fallback_labels), math.nan, dtype=np.float64)
        t_over_tref = np.full(len(fallback_labels), math.nan, dtype=np.float64)
        mdot_tminus5over3_norm_gs = np.full(len(fallback_labels), math.nan, dtype=np.float64)
        sink_interval_E_g = np.zeros_like(hist_E_g)
        sink_interval_j_g = np.zeros_like(hist_j_g)
        sink_interval_Be_g = np.zeros_like(hist_Be_g)
        outer_interval_E_g = np.zeros_like(hist_E_g)
        outer_interval_j_g = np.zeros_like(hist_j_g)
        outer_interval_Be_g = np.zeros_like(hist_Be_g)

        if prev_time_code is not None:
            dt_code = header.time_code - prev_time_code
            if dt_code <= 0.0:
                raise RuntimeError(f"Non-positive snapshot spacing around {snapshot_path.name}.")
            interval_mid_s = 0.5 * (prev_time_code + header.time_code) * units.time_cgs

            sink_interval_E_code = 0.5 * (prev_sink_rate_code + sink_rate_E_code) * dt_code
            sink_interval_j_code = 0.5 * (prev_sink_rate_j_code + sink_rate_j_code) * dt_code
            sink_interval_Be_code = 0.5 * (prev_sink_rate_be_code + sink_rate_Be_code) * dt_code
            outer_interval_E_code = 0.5 * (prev_outer_rate_code + outer_rate_E_code) * dt_code
            outer_interval_j_code = 0.5 * (prev_outer_rate_j_code + outer_rate_j_code) * dt_code
            outer_interval_Be_code = 0.5 * (prev_outer_rate_be_code + outer_rate_Be_code) * dt_code

            sink_interval_mass_code = 0.5 * (prev_sink_rate_total_code + sink_rate_total_code) * dt_code
            outer_interval_mass_code = 0.5 * (prev_outer_rate_total_code + outer_rate_total_code) * dt_code
            control_rate_avg_code = 0.5 * (prev_control_rate_total_code + control_rate_total_code)
            mdot_sink_code = sink_interval_mass_code / dt_code
            mdot_outer_code = outer_interval_mass_code / dt_code
            dmbound_dt_code = (m_bound_code - prev_bound_code) / dt_code if prev_bound_code is not None else math.nan

            cumulative_sink_mass_code += sink_interval_mass_code
            cumulative_outer_mass_code += outer_interval_mass_code
            cumulative_sink_E_code += sink_interval_E_code
            cumulative_sink_j_code += sink_interval_j_code
            cumulative_sink_Be_code += sink_interval_Be_code
            cumulative_outer_E_code += outer_interval_E_code
            cumulative_outer_j_code += outer_interval_j_code
            cumulative_outer_Be_code += outer_interval_Be_code
            fallback_in_avg_code = 0.5 * (prev_fallback_rate_in_total_code + fallback_rate_in_total_code)
            fallback_out_avg_code = 0.5 * (prev_fallback_rate_out_total_code + fallback_rate_out_total_code)
            fallback_net_avg_code = 0.5 * (prev_fallback_rate_net_total_code + fallback_rate_net_total_code)
            cumulative_fallback_in_mass_code += fallback_in_avg_code * dt_code
            cumulative_fallback_out_mass_code += fallback_out_avg_code * dt_code
            cumulative_fallback_net_mass_code += fallback_net_avg_code * dt_code
            sink_interval_E_g = sink_interval_E_code * units.mass_cgs
            sink_interval_j_g = sink_interval_j_code * units.mass_cgs
            sink_interval_Be_g = sink_interval_Be_code * units.mass_cgs
            outer_interval_E_g = outer_interval_E_code * units.mass_cgs
            outer_interval_j_g = outer_interval_j_code * units.mass_cgs
            outer_interval_Be_g = outer_interval_Be_code * units.mass_cgs
            mdot_sink_gs = mdot_sink_code * units.mass_cgs / units.time_cgs
            mdot_outer_gs = mdot_outer_code * units.mass_cgs / units.time_cgs
            dmbound_dt_gs = dmbound_dt_code * units.mass_cgs / units.time_cgs
            mdot_control_gs = control_rate_avg_code * units.mass_cgs / units.time_cgs
            mdot_fallback_in_gs = fallback_in_avg_code * units.mass_cgs / units.time_cgs
            mdot_fallback_out_gs = fallback_out_avg_code * units.mass_cgs / units.time_cgs
            mdot_fallback_net_gs = fallback_net_avg_code * units.mass_cgs / units.time_cgs
        else:
            cumulative_sink_mass_code = 0.0
            cumulative_outer_mass_code = 0.0

        if math.isfinite(interval_mid_s):
            for shell_index in range(len(fallback_labels)):
                if (
                    not math.isfinite(fallback_tref_s[shell_index])
                    and math.isfinite(mdot_fallback_in_gs[shell_index])
                    and mdot_fallback_in_gs[shell_index] > 0.0
                ):
                    fallback_tref_s[shell_index] = interval_mid_s
                    fallback_mdot_ref_gs[shell_index] = mdot_fallback_in_gs[shell_index]
                if (
                    math.isfinite(fallback_tref_s[shell_index])
                    and math.isfinite(fallback_mdot_ref_gs[shell_index])
                    and fallback_tref_s[shell_index] > 0.0
                    and interval_mid_s >= fallback_tref_s[shell_index]
                ):
                    t_over_tref[shell_index] = interval_mid_s / fallback_tref_s[shell_index]
                    mdot_tminus5over3_norm_gs[shell_index] = (
                        fallback_mdot_ref_gs[shell_index] * t_over_tref[shell_index] ** (-5.0 / 3.0)
                    )

        m_sink_cumulative_g = cumulative_sink_mass_code * units.mass_cgs
        m_outer_cumulative_g = cumulative_outer_mass_code * units.mass_cgs
        m_total_accounted_g = (m_domain_code + cumulative_sink_mass_code + cumulative_outer_mass_code) * units.mass_cgs
        mass_error_g = (m_domain_code + cumulative_sink_mass_code + cumulative_outer_mass_code - initial_mass_code) * units.mass_cgs

        if not used_cache:
            summary_rows = [
                ("analysis_format_version", ANALYSIS_FORMAT_VERSION, "version"),
                ("length_cgs", units.length_cgs, "cm"),
                ("mass_cgs", units.mass_cgs, "g"),
                ("time_cgs", units.time_cgs, "s"),
                ("velocity_cgs", units.velocity_cgs, "cm/s"),
                ("density_cgs", units.density_cgs, "g/cm^3"),
                ("pressure_cgs", units.pressure_cgs, "erg/cm^3"),
                ("specific_eint_cgs", units.specific_eint_cgs, "erg/g"),
                ("angular_momentum_cgs", units.angular_momentum_cgs, "cm^2/s"),
                ("newton_g_code", runtime.newton_g_code, "dimensionless"),
                ("newton_g_cgs", GRAV_CONSTANT_CGS, "cgs"),
                ("bh_mass_code", runtime.bh_mass_code, "code_mass"),
                ("bh_mass_cgs", runtime.bh_mass_code * units.mass_cgs, "g"),
                ("bh_softening_code", runtime.bh_softening_code, "code_length"),
                ("bh_softening_cgs", runtime.bh_softening_code * units.length_cgs, "cm"),
                ("snapshot_index", header.snapshot_index, "index"),
                ("snapshot_time", time_s, "s"),
                ("interval_mid_time", interval_mid_s, "s"),
                ("use_translating_frame", 1 if runtime.use_translating_frame else 0, "flag"),
                ("density_floor_code_used", density_floor_code, "code_density"),
                ("density_threshold_factor_used", float(config["density_threshold_factor"]), "dimensionless"),
                ("density_threshold_code_used", density_threshold_code, "code_density"),
                ("sink_radius_code_used", runtime.sink_radius_code, "code_length"),
                ("sink_shell_cells_used", float(config["sink_shell_cells"]), "dimensionless"),
                ("outer_shell_cells_used", float(config["outer_shell_cells"]), "dimensionless"),
                ("control_radius_code_used", float(config["control_radius_code"]), "code_length"),
                ("control_shell_cells_used", float(config["control_shell_cells"]), "dimensionless"),
                ("r_tide_code_used", float(scale_info["r_tide_code"]), "code_length"),
                ("stellar_radius_code_used", float(scale_info["stellar_radius_code"]), "code_length"),
                ("a_mb_code_used", float(scale_info["a_mb_code"]), "code_length"),
                ("fallback_shell_cells_used", float(config["fallback_shell_cells"]), "dimensionless"),
                ("inner_shell_radius_code_used", float(scale_info["inner_shell_radius_code"]), "code_length"),
                ("inner_shell_cells_used", float(config["inner_shell_cells"]), "dimensionless"),
                ("outer_stream_radius_code_used", float(scale_info["outer_stream_radius_code"]), "code_length"),
                ("outer_stream_shell_cells_used", float(config["outer_stream_shell_cells"]), "dimensionless"),
                ("compression_threshold_code_used", float(compression_threshold_code), "code_rate"),
                ("compression_percentile_used", float(config["compression_percentile"]), "dimensionless"),
                ("tail_energy_multiplier_used", float(config["tail_energy_multiplier"]), "dimensionless"),
                ("tail_energy_threshold_code", float(scale_info["tail_energy_threshold_code"]), "code_specific_energy"),
                ("tail_energy_threshold_cgs", float(scale_info["tail_energy_threshold_cgs"]), "erg/g"),
                ("tail_j_threshold_code_used", float(tail_j_threshold_code), "code_angular_momentum"),
                ("tail_j_percentile_used", float(config["tail_j_percentile"]), "dimensionless"),
                ("tail_e_threshold_used", float(config["tail_e_threshold"]), "dimensionless"),
                ("fit_tail_alpha_used", 1 if bool(config.get("fit_tail_alpha", True)) else 0, "flag"),
                ("tail_fit_min_populated_bins_used", int(config.get("tail_fit_min_populated_bins", 8)), "count"),
                ("ecrit_used", float(config["ecrit"]), "dimensionless"),
                ("Rcrit_code_used", float(config["Rcrit_code"]), "code_length"),
                ("bh_x_code", runtime.bh_x_code, "code_length"),
                ("bh_y_code", runtime.bh_y_code, "code_length"),
                ("bh_z_code", runtime.bh_z_code, "code_length"),
                ("bh_vx_code", runtime.bh_vx_code, "code_velocity"),
                ("bh_vy_code", runtime.bh_vy_code, "code_velocity"),
                ("bh_vz_code", runtime.bh_vz_code, "code_velocity"),
                ("M_domain", m_domain_g, "g"),
                ("M_sink_cumulative", m_sink_cumulative_g, "g"),
                ("M_outer_cumulative", m_outer_cumulative_g, "g"),
                ("M_bound_domain", m_bound_g, "g"),
                ("M_unbound_domain", m_unbound_g, "g"),
                ("M_total_accounted", m_total_accounted_g, "g"),
                ("mass_conservation_error", mass_error_g, "g"),
                ("Mdot_sink", mdot_sink_gs, "g/s"),
                ("Mdot_outer", mdot_outer_gs, "g/s"),
                ("dM_bound_domain_dt", dmbound_dt_gs, "g/s"),
                ("Mdot_control", mdot_control_gs, "g/s"),
                ("sink_rate_total_snapshot", sink_rate_total_code * units.mass_cgs / units.time_cgs, "g/s"),
                ("outer_rate_total_snapshot", outer_rate_total_code * units.mass_cgs / units.time_cgs, "g/s"),
                ("control_rate_total_snapshot", control_rate_total_code * units.mass_cgs / units.time_cgs, "g/s"),
                ("mass_fraction_e_lt_ecrit", frac_ecrit, "dimensionless"),
                ("mass_fraction_Rcirc_lt_Rcrit", frac_rcirc, "dimensionless"),
                ("mean_j_bound", mean_j_bound_cgs, "cm^2/s"),
                ("median_j_bound", median_j_bound_cgs, "cm^2/s"),
                ("mean_eps_bound", mean_eps_bound_cgs, "erg/g"),
                ("median_eps_bound", median_eps_bound_cgs, "erg/g"),
                ("M_tail_domain", m_tail_domain_code * units.mass_cgs, "g"),
                ("M_tail_bound_inward", m_tail_bound_inward_code * units.mass_cgs, "g"),
                ("M_tail_lowj", m_tail_lowj_code * units.mass_cgs, "g"),
                ("M_tail_high_e", m_tail_high_e_code * units.mass_cgs, "g"),
                ("tail_mass_fraction_of_domain", tail_mass_fraction_of_domain, "dimensionless"),
                ("tail_mass_fraction_of_bound", tail_mass_fraction_of_bound, "dimensionless"),
                ("mass_compressed_proxy", m_compressed_proxy_code * units.mass_cgs, "g"),
                ("mass_fraction_compressed_proxy", mass_fraction_compressed_proxy, "dimensionless"),
                ("mass_bound_inward_compressed_proxy", m_bound_inward_compressed_proxy_code * units.mass_cgs, "g"),
                ("low_j_mass_fraction_bound", low_j_mass_fraction_bound, "dimensionless"),
                ("high_e_mass_fraction_bound", high_e_mass_fraction_bound, "dimensionless"),
                ("tail_alpha", tail_alpha, "dimensionless"),
                ("tail_fit_nbins", tail_fit_nbins, "count"),
                ("tail_fit_energy_min", tail_fit_energy_min, "code_specific_energy"),
                ("tail_fit_energy_max", tail_fit_energy_max, "code_specific_energy"),
                ("tail_fit_r2", tail_fit_r2, "dimensionless"),
            ]
            for label, radius in zip(fallback_labels, scale_info["fallback_radii_code"]):
                shell_index = fallback_labels.index(label)
                summary_rows.extend([
                    (f"fallback_radius_code_{label}", float(radius), "code_length"),
                    (f"fallback_rate_inward_total_snapshot_{label}", fallback_rate_in_total_code[shell_index] * units.mass_cgs / units.time_cgs, "g/s"),
                    (f"fallback_rate_outward_total_snapshot_{label}", fallback_rate_out_total_code[shell_index] * units.mass_cgs / units.time_cgs, "g/s"),
                    (f"fallback_rate_net_total_snapshot_{label}", fallback_rate_net_total_code[shell_index] * units.mass_cgs / units.time_cgs, "g/s"),
                    (f"Mdot_fallback_inward_{label}", mdot_fallback_in_gs[shell_index], "g/s"),
                    (f"Mdot_fallback_outward_{label}", mdot_fallback_out_gs[shell_index], "g/s"),
                    (f"Mdot_fallback_net_{label}", mdot_fallback_net_gs[shell_index], "g/s"),
                    (f"cumulative_inward_mass_{label}", cumulative_fallback_in_mass_code[shell_index] * units.mass_cgs, "g"),
                    (f"cumulative_outward_mass_{label}", cumulative_fallback_out_mass_code[shell_index] * units.mass_cgs, "g"),
                    (f"cumulative_net_mass_{label}", cumulative_fallback_net_mass_code[shell_index] * units.mass_cgs, "g"),
                    (f"t_over_tref_{label}", t_over_tref[shell_index], "dimensionless"),
                    (f"mdot_tminus5over3_normalized_{label}", mdot_tminus5over3_norm_gs[shell_index], "g/s"),
                ])

            histogram_sections.extend([
                ("sink_interval", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), sink_interval_E_g),
                ("sink_interval", "j", convert_edges_to_physical("j", edges_code["j"], units), sink_interval_j_g),
                ("outer_interval", "energy", convert_edges_to_physical("energy", edges_code["energy"], units), outer_interval_E_g),
                ("outer_interval", "j", convert_edges_to_physical("j", edges_code["j"], units), outer_interval_j_g),
            ])
            if saha_cache is not None and have_eint:
                histogram_sections.extend([
                    ("sink_interval", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), sink_interval_Be_g),
                    ("outer_interval", "Be", convert_edges_to_physical("Be", edges_code["Be"], units), outer_interval_Be_g),
                ])

            write_snapshot_analysis(
                cached_output_path,
                source_bin_name=snapshot_path.name,
                snapshot_index=header.snapshot_index,
                snapshot_time_s=time_s,
                summary_rows=summary_rows,
                histogram_sections=histogram_sections,
                hist2d_sections=[
                    (
                        "full_in_domain",
                        "energy",
                        "j",
                        edges_2d_physical["energy_j"][0],
                        edges_2d_physical["energy_j"][1],
                        hist2d_energy_j_code[HIST2D_FULL] * units.mass_cgs,
                    ),
                    (
                        "bound_inward",
                        "energy",
                        "j",
                        edges_2d_physical["energy_j"][0],
                        edges_2d_physical["energy_j"][1],
                        hist2d_energy_j_code[HIST2D_BOUND_INWARD] * units.mass_cgs,
                    ),
                    (
                        "compressed_proxy",
                        "energy",
                        "j",
                        edges_2d_physical["energy_j"][0],
                        edges_2d_physical["energy_j"][1],
                        hist2d_energy_j_code[HIST2D_COMPRESSED] * units.mass_cgs,
                    ),
                    (
                        "bound_inward_compressed_proxy",
                        "energy",
                        "j",
                        edges_2d_physical["energy_j"][0],
                        edges_2d_physical["energy_j"][1],
                        hist2d_energy_j_code[HIST2D_BOUND_INWARD_COMPRESSED] * units.mass_cgs,
                    ),
                    (
                        "full_in_domain",
                        "energy",
                        "e",
                        edges_2d_physical["energy_e"][0],
                        edges_2d_physical["energy_e"][1],
                        hist2d_energy_e_code[HIST2D_FULL] * units.mass_cgs,
                    ),
                    (
                        "bound_inward",
                        "energy",
                        "e",
                        edges_2d_physical["energy_e"][0],
                        edges_2d_physical["energy_e"][1],
                        hist2d_energy_e_code[HIST2D_BOUND_INWARD] * units.mass_cgs,
                    ),
                    (
                        "compressed_proxy",
                        "energy",
                        "e",
                        edges_2d_physical["energy_e"][0],
                        edges_2d_physical["energy_e"][1],
                        hist2d_energy_e_code[HIST2D_COMPRESSED] * units.mass_cgs,
                    ),
                    (
                        "bound_inward_compressed_proxy",
                        "energy",
                        "e",
                        edges_2d_physical["energy_e"][0],
                        edges_2d_physical["energy_e"][1],
                        hist2d_energy_e_code[HIST2D_BOUND_INWARD_COMPRESSED] * units.mass_cgs,
                    ),
                    (
                        "full_in_domain",
                        "j",
                        "e",
                        edges_2d_physical["j_e"][0],
                        edges_2d_physical["j_e"][1],
                        hist2d_j_e_code[HIST2D_FULL] * units.mass_cgs,
                    ),
                    (
                        "bound_inward",
                        "j",
                        "e",
                        edges_2d_physical["j_e"][0],
                        edges_2d_physical["j_e"][1],
                        hist2d_j_e_code[HIST2D_BOUND_INWARD] * units.mass_cgs,
                    ),
                    (
                        "compressed_proxy",
                        "j",
                        "e",
                        edges_2d_physical["j_e"][0],
                        edges_2d_physical["j_e"][1],
                        hist2d_j_e_code[HIST2D_COMPRESSED] * units.mass_cgs,
                    ),
                    (
                        "bound_inward_compressed_proxy",
                        "j",
                        "e",
                        edges_2d_physical["j_e"][0],
                        edges_2d_physical["j_e"][1],
                        hist2d_j_e_code[HIST2D_BOUND_INWARD_COMPRESSED] * units.mass_cgs,
                    ),
                ],
            )

        prev_time_code = header.time_code
        prev_bound_code = m_bound_code
        prev_sink_rate_code = sink_rate_E_code.copy()
        prev_sink_rate_j_code = sink_rate_j_code.copy()
        prev_sink_rate_be_code = sink_rate_Be_code.copy()
        prev_outer_rate_code = outer_rate_E_code.copy()
        prev_outer_rate_j_code = outer_rate_j_code.copy()
        prev_outer_rate_be_code = outer_rate_Be_code.copy()
        prev_sink_rate_total_code = sink_rate_total_code
        prev_outer_rate_total_code = outer_rate_total_code
        prev_control_rate_total_code = control_rate_total_code
        prev_fallback_rate_in_total_code = fallback_rate_in_total_code.copy()
        prev_fallback_rate_out_total_code = fallback_rate_out_total_code.copy()
        prev_fallback_rate_net_total_code = fallback_rate_net_total_code.copy()


def main(argv: Sequence[str]) -> int:
    if len(argv) > 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent
    input_dir = Path(argv[1]).expanduser().resolve() if len(argv) >= 2 else Path.cwd()
    output_dir = Path(argv[2]).expanduser().resolve() if len(argv) >= 3 else (input_dir / "analysis_dat")
    config_path = Path(argv[3]).expanduser().resolve() if len(argv) >= 4 else (script_dir / "analysis_config.json")

    try:
        run_analysis(input_dir, output_dir, config_path)
    except Exception as exc:
        logging.getLogger("analyze_tde").exception("Analysis failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
