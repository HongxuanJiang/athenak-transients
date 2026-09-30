"""Per-dump driver (header -> box -> resample -> transfer -> products) and run loop."""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numba
from numba import get_num_threads, set_num_threads

from .bands import band_lnu_from_spectrum, band_metadata
from .config import RTSettings
from .eos import TabulatedLteTable, resolve_eos_table
from .opacity import MesaOpacityModel, frequency_quadrature_weights, photon_energy_grid
from .output import (
    attributes_mismatch,
    file_sha256,
    read_product,
    snapshot_hdf5_path,
    write_lightcurve_records,
    write_snapshot_hdf5,
)
from .snapshot import (
    DenseGasBoxInfo,
    RTBox,
    UnitSystem,
    apply_bh_excision_mask,
    axes_for_direction,
    build_runtime,
    build_units,
    choose_auto_box_density_threshold,
    choose_density_threshold,
    determine_rt_box,
    direction_code_and_sign,
    grid_storage_dtypes,
    image_geometry_for_direction,
    read_snapshot_header,
    resample_snapshot_to_grid,
)
from .transfer import (
    SAHA_POPULATION_ARGS,
    blackbody_band_maps,
    effective_temperature,
    integrate_grey_rays,
    integrate_multifreq_rays,
    integrate_tau1_rays,
)


@dataclass
class RTResult:
    snapshot: Path
    snapshot_index: int
    time_code: float
    time_s: float
    cycle: int
    mode: str
    direction: str
    luminosity: float
    maps: Dict[str, np.ndarray]
    image_u: np.ndarray
    image_v: np.ndarray
    area_cm2: float
    ds_cm: float
    valid_pixels: int
    total_pixels: int
    max_tau: float
    median_temp: float
    mean_temp: float
    units: UnitSystem
    rt_box: RTBox
    dense_box_info: Optional[DenseGasBoxInfo]
    dfloor_code: float
    density_threshold_code: float
    auto_box_threshold_code: float
    bh_xyz_code: Tuple[float, float, float]
    bh_mask_radius_code: float
    eos_table: Path
    eos_table_in_header: Optional[str]
    grid_storage_dtype: str
    threads: int
    populations: str = "n/a"  # "eos", "saha", or "n/a" (modes without ionization)
    populations_note: str = ""
    spectrum: Optional[dict] = None
    bands: Optional[dict] = None
    hdf5_path: Optional[Path] = None
    timings: Dict[str, float] = field(default_factory=dict)
    table_sha256: Dict[str, str] = field(default_factory=dict)  # eos_table, mesa_*


def configure_threads(threads: Optional[int], log=print) -> int:
    """Set the numba thread count; ``None`` means every CPU available to the process.

    The count is capped at ``numba.config.NUMBA_NUM_THREADS``, which numba fixes at
    import time; a request above it is reduced with one warning line.
    """
    if threads is None:
        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:  # not available on every platform
            threads = os.cpu_count() or 1
    threads = int(threads)
    cap = int(numba.config.NUMBA_NUM_THREADS)
    if threads > cap:
        log(
            f"WARNING: {threads} threads requested, but numba allows at most {cap} "
            f"(NUMBA_NUM_THREADS); using {cap}."
        )
        threads = cap
    set_num_threads(threads)
    return get_num_threads()


def result_line(name: str, mode: str, direction: str, luminosity: float, path, reused=False) -> str:
    """The one-line summary the command line prints for a (dump, direction) product."""
    line = f"{name} {mode} {direction}: L_bol,iso = {luminosity:.10e} erg/s -> {path}"
    return line + " (reused)" if reused else line


def table_hashes(settings: RTSettings, eos_table: Path) -> Dict[str, str]:
    """SHA-256 of the EOS table and of the MESA tables in use."""
    hashes = {"eos_table": file_sha256(eos_table)}
    for key in ("mesa_high_t", "mesa_low_t"):
        path = Path(getattr(settings, key))
        if path.is_file():
            hashes[key] = file_sha256(path)
    return hashes


def _find_eos_table(snapshot: Path, settings: RTSettings, header) -> Path:
    # A relative <hydro>/table path is relative to the run directory, which is
    # normally the parent of the dump directory (<run>/bin/).
    return resolve_eos_table(
        settings.eos_table,
        header.eos_table_in_header,
        relative_to=(snapshot.resolve().parent.parent, snapshot.resolve().parent),
    )


def _dump_context(snapshot: Path, settings: RTSettings):
    """Identity of a dump and the hashes of the tables it is processed with."""
    header = read_snapshot_header(snapshot)
    identity = {
        "snapshot_file": snapshot.name,
        "snapshot_index": int(header.snapshot_index),
        "time_code": float(header.time_code),
        "cycle": int(header.cycle),
    }
    return identity, table_hashes(settings, _find_eos_table(snapshot, settings, header))


class EosCache:
    """One TabulatedLteTable per (table path, unit system)."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, float, float, float], TabulatedLteTable] = {}

    def get(self, table_path: Path, units: UnitSystem) -> TabulatedLteTable:
        key = (str(table_path), units.length_cgs, units.mass_cgs, units.time_cgs)
        if key not in self._cache:
            self._cache[key] = TabulatedLteTable.from_units(table_path, units)
        return self._cache[key]


def _finite_stats(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.nanmedian(finite)), float(np.nanmean(finite))


def process_snapshot(
    snapshot: Path,
    settings: RTSettings,
    mesa: MesaOpacityModel,
    eos_cache: Optional[EosCache] = None,
    log=print,
    provenance: Optional[Mapping[str, str]] = None,
) -> RTResult:
    """Run one dump for ``settings.direction`` and write its HDF5 product.

    ``provenance`` (the parameter file, see ``RunConfig.provenance``) is
    stored in the product's ``/parameters`` group.
    """
    settings.validate()
    snapshot = Path(snapshot)
    eos_cache = eos_cache if eos_cache is not None else EosCache()
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    header = read_snapshot_header(snapshot)
    units = build_units(header.input_data)
    runtime = build_runtime(header)
    eos_table = _find_eos_table(snapshot, settings, header)
    eos = eos_cache.get(eos_table, units)

    # H/He populations for the ionization-dependent modes (grey-therm, multifreq).
    populations, populations_note = "n/a", ""
    pop_args = SAHA_POPULATION_ARGS  # explicit, so that every mode hits numba's cache
    if settings.mode in ("grey-therm", "multifreq"):
        populations = settings.populations
        if populations == "eos":
            try:
                pop_args = (True, *eos.population_table())
            except RuntimeError as exc:
                populations = "saha"
                populations_note = (
                    f"{exc} Falling back to the ideal Saha solver (X=0.7, Y=0.3)."
                )
                log(f"WARNING: populations = eos unavailable: {populations_note}")

    auto_threshold = choose_auto_box_density_threshold(
        header, settings.density_threshold_factor, settings.density_threshold_code
    )
    density_threshold = choose_density_threshold(
        header, settings.density_threshold_factor, settings.density_threshold_code
    )
    rt_box, dense_box_info = determine_rt_box(header, settings.box, auto_threshold)
    x_axis, y_axis, z_axis = axes_for_direction(
        rt_box, settings.direction, settings.image_size, settings.los_steps
    )
    timings["box"] = time.perf_counter() - t0

    log(
        f"Reading {snapshot.name}: time={header.time_code:g}, "
        f"BH=({runtime.bh_inertial_xyz[0]:.6g},{runtime.bh_inertial_xyz[1]:.6g},"
        f"{runtime.bh_inertial_xyz[2]:.6g}), "
        f"grid=({x_axis.size},{y_axis.size},{z_axis.size}), "
        f"mode={settings.mode}, direction={settings.direction}"
    )
    if dense_box_info is not None:
        raw = dense_box_info.raw_box
        log(
            f"  auto RT box: rho_code>{dense_box_info.threshold_code:.3e}, "
            f"selected_cells={dense_box_info.selected_cells}, "
            f"max_rho_code={dense_box_info.max_density_code:.3e}, "
            f"raw=({raw.x_min:.6g},{raw.x_max:.6g},{raw.y_min:.6g},{raw.y_max:.6g},"
            f"{raw.z_min:.6g},{raw.z_max:.6g}), "
            f"padded=({rt_box.x_min:.6g},{rt_box.x_max:.6g},{rt_box.y_min:.6g},"
            f"{rt_box.y_max:.6g},{rt_box.z_min:.6g},{rt_box.z_max:.6g})"
        )

    grid_dtype, level_dtype = grid_storage_dtypes(
        settings.image_size, settings.los_steps, settings.grid_dtype
    )
    t1 = time.perf_counter()
    rho_grid, temp_grid, level_grid = resample_snapshot_to_grid(
        header,
        eos,
        units,
        x_axis,
        y_axis,
        z_axis,
        density_threshold,
        grid_dtype=grid_dtype,
        level_dtype=level_dtype,
    )
    del level_grid
    timings["resample"] = time.perf_counter() - t1

    bh_radius_used = 0.0
    if settings.bh_mask:
        bh_radius_used = apply_bh_excision_mask(
            rho_grid, temp_grid, x_axis, y_axis, z_axis, runtime, settings.bh_mask_radius
        )

    direction_code, positive = direction_code_and_sign(settings.direction)
    image_u, image_v, ds_code, area_code = image_geometry_for_direction(
        settings.direction, x_axis, y_axis, z_axis
    )
    ds_cm = ds_code * units.length_cgs
    area_cm2 = area_code * units.length_cgs**2
    mesa_args = mesa.kernel_args

    t2 = time.perf_counter()
    spectrum = None
    bands = None
    labels, categories, band_nu, band_lambda, band_ev = band_metadata(
        settings.observation_bands
    )

    if settings.mode == "tau1":
        tau_total, ph_temp, ph_coord, flux_map, valid = integrate_tau1_rays(
            rho_grid,
            temp_grid,
            x_axis,
            y_axis,
            z_axis,
            *mesa_args,
            direction_code,
            positive,
            ds_cm,
            area_cm2,
            float(settings.tau_photosphere),
        )
        luminosity = float(np.sum(flux_map, dtype=np.float64))
        maps = {
            "tau_total": tau_total,
            "photosphere_temp_K": ph_temp,
            "photosphere_coord_code": ph_coord,
            "valid": valid,
            "bolometric_pixel_luminosity_erg_s": flux_map,
        }
        max_tau = float(np.nanmax(tau_total))
        median_temp, mean_temp = _finite_stats(ph_temp)
        if band_nu.size:
            lnu_maps = blackbody_band_maps(ph_temp, valid, band_nu, float(area_cm2))
            band_lnu = np.sum(lnu_maps, axis=(1, 2), dtype=np.float64)
            bands = {
                "source": "blackbody at the tau=tau_ph photosphere temperature",
                "lnu_pixel_erg_s_hz": lnu_maps,
                "lnu_erg_s_hz": band_lnu,
            }
    elif settings.mode in ("grey", "grey-therm"):
        intensity, tau_map = integrate_grey_rays(
            rho_grid,
            temp_grid,
            *mesa_args,
            direction_code,
            positive,
            ds_cm,
            settings.mode == "grey-therm",
            float(settings.tau_stop),
            *pop_args,
        )
        luminosity = 4.0 * math.pi * intensity.sum() * area_cm2
        valid = intensity > 0.0
        teff = effective_temperature(intensity)
        maps = {
            "intensity_erg_s_cm2_sr": intensity,
            "tau_effective": tau_map,
            "effective_temp_K": teff,
            "valid": valid,
            "bolometric_pixel_luminosity_erg_s": 4.0 * math.pi * intensity * area_cm2,
        }
        max_tau = float(np.nanmax(tau_map))
        median_temp, mean_temp = _finite_stats(teff)
    else:  # multifreq
        energies_ev, nus = photon_energy_grid(
            settings.nfreq, settings.emin_ev, settings.emax_ev
        )
        spec_rows, intensity = integrate_multifreq_rays(
            rho_grid,
            temp_grid,
            *mesa_args,
            direction_code,
            positive,
            ds_cm,
            nus,
            bool(settings.scattering),
            False,
            float(settings.tau_stop),
            *pop_args,
        )
        luminosity = 4.0 * math.pi * intensity.sum() * area_cm2
        lnu = 4.0 * math.pi * spec_rows.sum(axis=1) * area_cm2
        valid = intensity > 0.0
        teff = effective_temperature(intensity)
        maps = {
            "intensity_erg_s_cm2_sr": intensity,
            "effective_temp_K": teff,
            "valid": valid,
            "bolometric_pixel_luminosity_erg_s": 4.0 * math.pi * intensity * area_cm2,
        }
        spectrum = {
            "energy_ev": energies_ev,
            "frequency_hz": nus,
            "quadrature_weight_hz": frequency_quadrature_weights(nus),
            "lnu_erg_s_hz": lnu,
            "nu_lnu_erg_s": lnu * nus,
            "lnu_row_erg_s_hz": 4.0 * math.pi * spec_rows * area_cm2,
        }
        max_tau = math.nan
        median_temp, mean_temp = _finite_stats(teff)
        if band_nu.size:
            bands = {
                "source": "log-log interpolation of the multifrequency spectrum",
                "lnu_erg_s_hz": band_lnu_from_spectrum(nus, lnu, band_nu),
            }
    timings["transfer"] = time.perf_counter() - t2
    del rho_grid, temp_grid

    if bands is not None:
        bands.update(
            {
                "labels": labels,
                "categories": categories,
                "frequency_hz": band_nu,
                "wavelength_nm": band_lambda,
                "energy_ev": band_ev,
                "nu_lnu_erg_s": bands["lnu_erg_s_hz"] * band_nu,
            }
        )

    result = RTResult(
        snapshot=snapshot,
        snapshot_index=header.snapshot_index,
        time_code=header.time_code,
        time_s=header.time_code * units.time_cgs,
        cycle=header.cycle,
        mode=settings.mode,
        direction=settings.direction,
        luminosity=luminosity,
        maps=maps,
        image_u=image_u,
        image_v=image_v,
        area_cm2=float(area_cm2),
        ds_cm=float(ds_cm),
        valid_pixels=int(np.count_nonzero(valid)),
        total_pixels=int(valid.size),
        max_tau=max_tau,
        median_temp=median_temp,
        mean_temp=mean_temp,
        units=units,
        rt_box=rt_box,
        dense_box_info=dense_box_info,
        dfloor_code=header.dfloor_code,
        density_threshold_code=density_threshold,
        auto_box_threshold_code=auto_threshold,
        bh_xyz_code=runtime.bh_inertial_xyz,
        bh_mask_radius_code=bh_radius_used,
        eos_table=eos_table,
        eos_table_in_header=header.eos_table_in_header,
        grid_storage_dtype=np.dtype(grid_dtype).name,
        threads=get_num_threads(),
        populations=populations,
        populations_note=populations_note,
        spectrum=spectrum,
        bands=bands,
        timings=timings,
        table_sha256=table_hashes(settings, eos_table),
    )

    t3 = time.perf_counter()
    result.hdf5_path = write_snapshot_hdf5(
        settings.output_dir, result, settings, provenance
    )
    timings["output"] = time.perf_counter() - t3
    log(
        f"  timings: box={timings['box']:.2f}s resample+EOS={timings['resample']:.2f}s "
        f"transfer={timings['transfer']:.2f}s output={timings['output']:.2f}s"
    )
    log(
        f"Done {snapshot.name} [{settings.mode}, {settings.direction}]: "
        f"L={luminosity:.6e} erg/s, valid={result.valid_pixels}/{result.total_pixels}"
    )
    return result


def _lightcurve_candidates(
    output_dir: Path, snapshots: Sequence[Path], mode: str, directions: Sequence[str]
) -> Dict[str, Dict[str, Path]]:
    """Products in ``output_dir`` of the dumps sharing the prefix of ``snapshots``.

    Returns {dump stem: {direction: path}} for the stems that have a product
    for every direction.
    """
    prefixes, exact = set(), set()
    for snapshot in snapshots:
        match = re.fullmatch(r"(.*)\.\d+", snapshot.stem)
        if match:
            prefixes.add(match.group(1))
        else:
            exact.add(snapshot.stem)
    alternatives = [re.escape(p) + r"\.\d+" for p in sorted(prefixes)]
    alternatives += [re.escape(e) for e in sorted(exact)]
    pattern = re.compile(
        rf"({'|'.join(alternatives)})\.rt_{re.escape(mode)}_(-?[xyz])\.h5"
    )
    found: Dict[str, Dict[str, Path]] = {}
    if output_dir.is_dir():
        for entry in output_dir.iterdir():
            match = pattern.fullmatch(entry.name)
            if match:
                found.setdefault(match.group(1), {})[match.group(2)] = entry
    return {
        stem: paths
        for stem, paths in found.items()
        if all(d in paths for d in directions)
    }


def run(
    settings: RTSettings,
    snapshots: Sequence[Path],
    log=print,
    *,
    directions: Optional[Sequence[str]] = None,
    skip_existing: bool = False,
    provenance: Optional[Mapping[str, str]] = None,
) -> List[RTResult]:
    """Process every dump for every direction with one set of settings.

    ``directions`` defaults to ``(settings.direction,)``.  With ``skip_existing``
    a (dump, direction) whose product exists is not recomputed, provided the
    product was made with the same settings and tables from the same dump
    (time, cycle and file name); otherwise ``RuntimeError`` is raised before
    anything is computed.

    The light curve (``rt_lightcurve_<mode>.csv`` / ``.h5``) is rewritten after
    each dump once it has two rows.  Its rows are the dumps of this call and every
    other dump in the output directory (same file-name prefix, mode and all
    directions) whose product matches the settings and tables of this call.
    Returns the results computed in this call.
    """
    directions = tuple(directions) if directions else (settings.direction,)
    output_dir = Path(settings.output_dir).expanduser().resolve()
    per_direction = [
        replace(settings, direction=d, output_dir=output_dir) for d in directions
    ]
    for item in per_direction:
        item.validate()
    snapshots = [Path(s) for s in snapshots]
    contexts: Dict[Path, tuple] = {}
    for snapshot in snapshots:
        contexts[snapshot] = _dump_context(snapshot, per_direction[0])
    records: Dict[Path, dict] = {}  # product path -> read_product record

    def record_of(path: Path) -> dict:
        if path not in records:
            records[path] = read_product(path)
        return records[path]

    if skip_existing:
        # Refuse to reuse products made with other settings before computing anything.
        for snapshot in snapshots:
            identity, hashes = contexts[snapshot]
            for item in per_direction:
                target = snapshot_hdf5_path(output_dir, snapshot, item.mode, item.direction)
                if not target.is_file():
                    continue
                try:
                    mismatch = attributes_mismatch(
                        record_of(target)["attrs"], item, identity, hashes
                    )
                except (OSError, KeyError):
                    mismatch = "cannot be read"
                if mismatch is not None:
                    raise RuntimeError(
                        f"{target} exists but does not match the current settings "
                        f"or input dump ({mismatch}); remove it, change <output>/dir, "
                        "or set <run>/skip_existing = false."
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    nthreads = configure_threads(settings.threads, log)
    log(f"athenak_rt: numba ray integration with {nthreads} thread(s).")

    # Rows of the light curve: this call's dumps and the matching products on disk.
    selected = {snapshot.stem: snapshot for snapshot in snapshots}
    rows: Dict[str, Dict[str, dict]] = {}
    if snapshots:
        first_hashes = contexts[snapshots[0]][1]
        for stem, paths in _lightcurve_candidates(
            output_dir, snapshots, settings.mode, directions
        ).items():
            if stem in selected:
                identity, hashes = contexts[selected[stem]]
            else:
                identity, hashes = None, first_hashes
                dump = snapshots[0].parent / f"{stem}.bin"
                if dump.is_file():
                    try:
                        identity = _dump_context(dump, per_direction[0])[0]
                    except RuntimeError:
                        pass
            try:
                per_dir = {d: record_of(paths[d]) for d in directions}
                ok = all(
                    attributes_mismatch(per_dir[item.direction]["attrs"], item, identity, hashes)
                    is None
                    for item in per_direction
                )
            except (OSError, KeyError):
                ok = False
            if ok:
                rows[stem] = per_dir

    mesa: Optional[MesaOpacityModel] = None
    eos_cache = EosCache()
    results: List[RTResult] = []
    written = None
    for snapshot in snapshots:
        row: Dict[str, dict] = {}
        for item in per_direction:
            target = snapshot_hdf5_path(output_dir, snapshot, item.mode, item.direction)
            if skip_existing and target.is_file():
                record = record_of(target)
                log(
                    result_line(
                        snapshot.name,
                        item.mode,
                        item.direction,
                        record["luminosity_bolometric_erg_s"],
                        target,
                        reused=True,
                    )
                )
                row[item.direction] = record
                continue
            if mesa is None:
                mesa = MesaOpacityModel(Path(item.mesa_high_t), Path(item.mesa_low_t))
            result = process_snapshot(
                snapshot, item, mesa, eos_cache, log=log, provenance=provenance
            )
            results.append(result)
            records.pop(result.hdf5_path, None)
            row[item.direction] = record_of(result.hdf5_path)
        rows[snapshot.stem] = row
        if len(rows) >= 2:
            ordered = sorted(
                rows.values(),
                key=lambda r: (r[directions[0]]["dump"], r[directions[0]]["time_code"]),
            )
            written = (
                len(ordered),
                *write_lightcurve_records(
                    output_dir, settings.mode, directions, ordered, provenance
                ),
            )
    if written is not None:
        log(f"Light curve ({written[0]} dumps): {written[1]} and {written[2].name}")
    return results
