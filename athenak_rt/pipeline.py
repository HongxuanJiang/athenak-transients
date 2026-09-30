"""Per-dump driver (header -> box -> resample -> transfer -> products) and run loop."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numba import get_num_threads, set_num_threads

from .bands import band_lnu_from_spectrum, band_metadata
from .config import RTSettings
from .eos import TabulatedLteTable, resolve_eos_table
from .opacity import MesaOpacityModel, frequency_quadrature_weights, photon_energy_grid
from .output import (
    existing_output_mismatch,
    snapshot_hdf5_path,
    write_lightcurve,
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


def configure_threads(threads: Optional[int]) -> int:
    if threads is None:
        threads = len(os.sched_getaffinity(0))
    set_num_threads(int(threads))
    return get_num_threads()


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
    # A relative <hydro>/table path is relative to the run directory, which is
    # normally the parent of the dump directory (<run>/bin/).
    eos_table = resolve_eos_table(
        settings.eos_table,
        header.eos_table_in_header,
        relative_to=(snapshot.resolve().parent.parent, snapshot.resolve().parent),
    )
    eos = eos_cache.get(eos_table, units)

    # H/He populations for the ionization-dependent modes (grey-therm, multifreq).
    populations, populations_note = "n/a", ""
    pop_args = (False,)
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


def run(
    settings: RTSettings,
    snapshots: Sequence[Path],
    directions: Optional[Sequence[str]] = None,
    *,
    skip_existing: bool = False,
    provenance: Optional[Mapping[str, str]] = None,
    log=print,
) -> List[RTResult]:
    """Process every dump for every direction with one set of settings.

    ``directions`` defaults to ``(settings.direction,)``.  With more than one
    dump a light curve (``rt_lightcurve_<mode>.csv`` / ``.h5``) is rewritten
    after each dump.  With ``skip_existing`` a (dump, direction) whose product
    exists is not recomputed, provided it was made with the same settings.
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
    if skip_existing:
        # Refuse to reuse products made with other settings before computing anything.
        for snapshot in snapshots:
            for item in per_direction:
                target = snapshot_hdf5_path(output_dir, snapshot, item.mode, item.direction)
                mismatch = (
                    existing_output_mismatch(target, item) if target.is_file() else None
                )
                if mismatch is not None:
                    raise RuntimeError(
                        f"{target} exists but does not match the current settings "
                        f"({mismatch}); remove it, change <output>/dir, or set "
                        "<run>/skip_existing = false."
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    nthreads = configure_threads(settings.threads)
    log(f"athenak_rt: numba ray integration with {nthreads} thread(s).")
    mesa: Optional[MesaOpacityModel] = None
    eos_cache = EosCache()
    results: List[RTResult] = []
    rows: List[Dict[str, Path]] = []
    for snapshot in snapshots:
        row: Dict[str, Path] = {}
        for item in per_direction:
            target = snapshot_hdf5_path(output_dir, snapshot, item.mode, item.direction)
            if skip_existing and target.is_file():
                log(
                    f"Skipping {snapshot.name} [{item.mode}, {item.direction}]: "
                    f"{target.name} exists."
                )
                row[item.direction] = target
                continue
            if mesa is None:
                mesa = MesaOpacityModel(Path(item.mesa_high_t), Path(item.mesa_low_t))
            result = process_snapshot(
                snapshot, item, mesa, eos_cache, log=log, provenance=provenance
            )
            results.append(result)
            row[item.direction] = result.hdf5_path
        rows.append(row)
        if len(snapshots) > 1:
            csv_path, h5_path = write_lightcurve(
                output_dir, settings.mode, directions, rows, provenance
            )
    if len(snapshots) > 1:
        log(f"Light curve ({len(rows)} dumps): {csv_path} and {h5_path.name}")
    return results
