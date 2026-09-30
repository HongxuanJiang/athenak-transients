"""HDF5 and light-curve writers.

One HDF5 file per (dump, mode, direction).  Every run setting, the dump
metadata, the ray box, the table paths and the parameter file are stored so a
product is self-describing.  Layout::

    /                       attributes: settings + provenance + L_bol
    /parameters/...         parameter file as read and fully resolved
    /image/image_u_code     column coordinates [code length]
    /image/image_v_code     row coordinates
    /maps/...               mode-specific pixel maps (see below)
    /spectra/...            multifreq: L_nu on the photon-energy grid
    /bands/...              L_nu at the observation bands

maps (tau1):       tau_total, photosphere_temp_K, photosphere_coord_code, valid,
                   bolometric_pixel_luminosity_erg_s
maps (grey*):      intensity_erg_s_cm2_sr, tau_effective, effective_temp_K, valid,
                   bolometric_pixel_luminosity_erg_s
maps (multifreq):  intensity_erg_s_cm2_sr, effective_temp_K, valid,
                   bolometric_pixel_luminosity_erg_s

A run over several dumps also writes ``rt_lightcurve_<mode>.csv`` and
``rt_lightcurve_<mode>.h5`` (one row per dump, every direction); they are
assembled from the per-dump files, so dumps reused with ``skip_existing`` enter
them as well.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .constants import (
    LUMINOSITY_BOLOMETRIC_FORMULA,
    LUMINOSITY_NORMALIZATION,
    LUMINOSITY_REFERENCE,
    LUMINOSITY_SPECTRAL_FORMULA,
    PROJECTED_LUMINOSITY_FACTOR,
)


# Settings that may differ between a run and the outputs it reuses (skip_existing).
_REUSE_IGNORED_SETTINGS = ("output_dir", "threads", "hdf5_compression")


def product_stem(snapshot: Path, mode: str, direction: str) -> str:
    return f"{Path(snapshot).stem}.rt_{mode}_{direction}"


def snapshot_hdf5_path(output_dir: Path, snapshot: Path, mode: str, direction: str) -> Path:
    return Path(output_dir) / f"{product_stem(snapshot, mode, direction)}.h5"


def lightcurve_paths(output_dir: Path, mode: str) -> Tuple[Path, Path]:
    stem = Path(output_dir) / f"rt_lightcurve_{mode}"
    return stem.with_suffix(".csv"), stem.with_suffix(".h5")


def direction_label(direction: str) -> str:
    """Column-safe name of a direction: ``-y`` -> ``minus_y``."""
    return f"minus_{direction[1:]}" if direction.startswith("-") else direction


def _set_attrs(target, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (tuple, list)):
            value = np.asarray(value)
        target.attrs[key] = value


def _write_parameters(h5, provenance: Optional[Mapping[str, str]]) -> None:
    """/parameters: the parameter file as read and with every value resolved."""
    if not provenance:
        return
    import h5py

    string_dtype = h5py.string_dtype(encoding="utf-8")
    group = h5.create_group("parameters")
    group.attrs["path"] = provenance.get("path", "")
    for name in ("parameter_file", "resolved"):
        group.create_dataset(name, data=provenance.get(name, ""), dtype=string_dtype)


def write_snapshot_hdf5(
    output_dir: Path, result, settings, provenance: Optional[Mapping[str, str]] = None
) -> Path:
    """Write one product; it appears under its final name only once complete."""
    import h5py

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_hdf5_path(output_dir, result.snapshot, result.mode, result.direction)
    partial = path.with_name(path.name + ".part")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression = settings.hdf5_compression
    if compression in ("", "none"):
        compression = None

    with h5py.File(partial, "w") as h5:
        attrs = {
            "athenak_rt_version": __version__,
            "snapshot_file": result.snapshot.name,
            "snapshot_path": str(result.snapshot),
            "snapshot_index": int(result.snapshot_index),
            "time_code": float(result.time_code),
            "time_s": float(result.time_s),
            "cycle": int(result.cycle),
            "luminosity_bolometric_erg_s": float(result.luminosity),
            "luminosity_normalization": LUMINOSITY_NORMALIZATION,
            "luminosity_reference": LUMINOSITY_REFERENCE,
            "luminosity_bolometric_formula": LUMINOSITY_BOLOMETRIC_FORMULA,
            "luminosity_spectral_formula": LUMINOSITY_SPECTRAL_FORMULA,
            "projected_luminosity_factor": PROJECTED_LUMINOSITY_FACTOR,
            "area_pixel_cm2": float(result.area_cm2),
            "los_step_cm": float(result.ds_cm),
            "valid_pixels": int(result.valid_pixels),
            "total_pixels": int(result.total_pixels),
            "max_tau": float(result.max_tau),
            "median_temp_K": float(result.median_temp),
            "mean_temp_K": float(result.mean_temp),
            "dfloor_code": float(result.dfloor_code),
            "density_threshold_code": float(result.density_threshold_code),
            "density_selection_rule": "rho_code > density_threshold_code",
            "auto_box_threshold_code": float(result.auto_box_threshold_code),
            "rt_box_code": result.rt_box.as_array(),
            "auto_box": bool(result.dense_box_info is not None),
            "bh_xyz_code": np.asarray(result.bh_xyz_code, dtype=np.float64),
            "bh_mask_radius_code": float(result.bh_mask_radius_code),
            "length_cgs": float(result.units.length_cgs),
            "mass_cgs": float(result.units.mass_cgs),
            "time_unit_s": float(result.units.time_cgs),
            "eos_table_resolved": str(result.eos_table),
            "eos_table_in_header": result.eos_table_in_header or "",
            "grid_storage_dtype": result.grid_storage_dtype,
            "numba_threads": int(result.threads),
            "populations_used": result.populations,
            "populations_note": result.populations_note,
        }
        attrs.update(settings.as_attributes())
        if result.dense_box_info is not None:
            info = result.dense_box_info
            attrs["auto_box_selected_cells"] = int(info.selected_cells)
            attrs["auto_box_max_density_code"] = float(info.max_density_code)
            attrs["auto_box_raw_code"] = info.raw_box.as_array()
        _set_attrs(h5, attrs)
        _write_parameters(h5, provenance)

        image = h5.create_group("image")
        image.create_dataset("image_u_code", data=result.image_u)
        image.create_dataset("image_v_code", data=result.image_v)

        map_group = h5.create_group("maps")
        for name, values in result.maps.items():
            map_group.create_dataset(name, data=values, compression=compression)

        if result.spectrum is not None:
            spectra = h5.create_group("spectra")
            spectra.attrs["quantity"] = "isotropic-equivalent specific luminosity"
            spectra.attrs["quadrature"] = "trapezoid in ln(nu)"
            for name in (
                "energy_ev",
                "frequency_hz",
                "quadrature_weight_hz",
                "lnu_erg_s_hz",
                "nu_lnu_erg_s",
            ):
                spectra.create_dataset(name, data=result.spectrum[name])
            spectra.create_dataset(
                "lnu_row_erg_s_hz",
                data=result.spectrum["lnu_row_erg_s_hz"],
                compression=compression,
            )

        if result.bands is not None:
            bands = h5.create_group("bands")
            bands.attrs["source"] = result.bands["source"]
            bands.create_dataset(
                "label",
                data=np.array(result.bands["labels"], dtype=object),
                dtype=string_dtype,
            )
            bands.create_dataset(
                "category",
                data=np.array(result.bands["categories"], dtype=object),
                dtype=string_dtype,
            )
            for name in (
                "frequency_hz",
                "wavelength_nm",
                "energy_ev",
                "lnu_erg_s_hz",
                "nu_lnu_erg_s",
            ):
                bands.create_dataset(name, data=result.bands[name])
            if "lnu_pixel_erg_s_hz" in result.bands:
                bands.create_dataset(
                    "lnu_pixel_erg_s_hz",
                    data=result.bands["lnu_pixel_erg_s_hz"],
                    compression=compression,
                )
    os.replace(partial, path)
    return path




def existing_output_mismatch(path: Path, settings) -> Optional[str]:
    """None if ``path`` was written with ``settings``, else the first difference.

    Only the settings that change the result are compared (not the output
    directory, the thread count or the compression).
    """
    import h5py

    try:
        with h5py.File(path, "r") as h5:
            stored = dict(h5.attrs)
    except OSError as exc:
        return f"cannot be read ({exc})"
    for name, value in settings.as_attributes().items():
        if name in _REUSE_IGNORED_SETTINGS:
            continue
        if name not in stored:
            return f"{name} is not recorded"
        old = stored[name]
        if isinstance(old, np.generic):
            old = old.item()
        if isinstance(old, bytes):
            old = old.decode()
        if isinstance(value, float) and value != value and old != old:
            continue  # both NaN
        if not bool(np.all(np.asarray(old) == np.asarray(value))):
            return f"{name} = {old!r} there, {value!r} now"
    return None


def _read_product(path: Path) -> dict:
    """The scalar results, band and spectrum arrays of one per-dump product."""
    import h5py

    with h5py.File(path, "r") as h5:
        a = h5.attrs
        record = {
            "file": Path(path).name,
            "dump": int(a["snapshot_index"]),
            "cycle": int(a["cycle"]),
            "time_code": float(a["time_code"]),
            "time_s": float(a["time_s"]),
            "luminosity_bolometric_erg_s": float(a["luminosity_bolometric_erg_s"]),
            "valid_pixels": int(a["valid_pixels"]),
            "total_pixels": int(a["total_pixels"]),
            "max_tau": float(a.get("max_tau", np.nan)),
            "median_temp_K": float(a.get("median_temp_K", np.nan)),
            "mean_temp_K": float(a.get("mean_temp_K", np.nan)),
            "density_threshold_code": float(a["density_threshold_code"]),
            "bh_xyz_code": np.asarray(a["bh_xyz_code"], dtype=np.float64),
        }
        if "bands" in h5:
            bands = h5["bands"]
            record["bands"] = {
                "label": list(bands["label"].asstr()[...]),
                "category": list(bands["category"].asstr()[...]),
                "frequency_hz": bands["frequency_hz"][...],
                "wavelength_nm": bands["wavelength_nm"][...],
                "energy_ev": bands["energy_ev"][...],
                "lnu_erg_s_hz": bands["lnu_erg_s_hz"][...],
                "nu_lnu_erg_s": bands["nu_lnu_erg_s"][...],
            }
        if "spectra" in h5:
            spectra = h5["spectra"]
            record["spectra"] = {
                name: spectra[name][...]
                for name in ("energy_ev", "frequency_hz", "lnu_erg_s_hz", "nu_lnu_erg_s")
            }
    return record


def _number(value: float) -> str:
    return repr(float(value))


def write_lightcurve(
    output_dir: Path,
    mode: str,
    directions: Sequence[str],
    rows: Sequence[Mapping[str, Path]],
    provenance: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, Path]:
    """Light-curve table (CSV) and HDF5 file, one row per dump.

    ``rows`` holds, per dump in time order, the per-dump product of every
    direction.  CSV columns: dump, cycle, time_code, time_s, then
    ``L_bol_<dir>`` for every direction and ``nuLnu_<band>_<dir>`` for every
    band and direction (erg/s; ``-y`` is written ``minus_y``).
    """
    import h5py

    directions = list(directions)
    records: List[Dict[str, dict]] = [
        {d: _read_product(row[d]) for d in directions} for row in rows
    ]
    first = records[0][directions[0]]
    band_info = first.get("bands")
    spectrum_info = first.get("spectra")
    csv_path, h5_path = lightcurve_paths(output_dir, mode)

    header = ["dump", "cycle", "time_code", "time_s"]
    header += [f"L_bol_{direction_label(d)}" for d in directions]
    if band_info is not None:
        header += [
            f"nuLnu_{label}_{direction_label(d)}"
            for d in directions
            for label in band_info["label"]
        ]
    lines = [",".join(header)]
    for record in records:
        ref = record[directions[0]]
        cells = [str(ref["dump"]), str(ref["cycle"])]
        cells += [_number(ref["time_code"]), _number(ref["time_s"])]
        cells += [_number(record[d]["luminosity_bolometric_erg_s"]) for d in directions]
        if band_info is not None:
            for d in directions:
                cells += [_number(v) for v in record[d]["bands"]["nu_lnu_erg_s"]]
        lines.append(",".join(cells))
    partial = csv_path.with_name(csv_path.name + ".part")
    partial.write_text("\n".join(lines) + "\n")
    os.replace(partial, csv_path)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    partial = h5_path.with_name(h5_path.name + ".part")
    with h5py.File(partial, "w") as h5:
        h5.attrs["athenak_rt_version"] = __version__
        h5.attrs["mode"] = mode
        h5.attrs.create("directions", data=directions, dtype=string_dtype)
        h5.attrs["luminosity_normalization"] = LUMINOSITY_NORMALIZATION
        h5.attrs["luminosity_reference"] = LUMINOSITY_REFERENCE
        h5.attrs["luminosity_bolometric_formula"] = LUMINOSITY_BOLOMETRIC_FORMULA
        h5.attrs["luminosity_spectral_formula"] = LUMINOSITY_SPECTRAL_FORMULA
        h5.attrs["projected_luminosity_factor"] = PROJECTED_LUMINOSITY_FACTOR
        _write_parameters(h5, provenance)
        refs = [record[directions[0]] for record in records]
        h5.create_dataset("dump", data=np.array([r["dump"] for r in refs], dtype=np.int64))
        h5.create_dataset("cycle", data=np.array([r["cycle"] for r in refs], dtype=np.int64))
        h5.create_dataset("time_code", data=np.array([r["time_code"] for r in refs]))
        h5.create_dataset("time_s", data=np.array([r["time_s"] for r in refs]))
        if band_info is not None:
            bands = h5.create_group("bands")
            for name in ("label", "category"):
                bands.create_dataset(
                    name, data=np.array(band_info[name], dtype=object), dtype=string_dtype
                )
            for name in ("frequency_hz", "wavelength_nm", "energy_ev"):
                bands.create_dataset(name, data=band_info[name])
        if spectrum_info is not None:
            spectra = h5.create_group("spectra")
            for name in ("energy_ev", "frequency_hz"):
                spectra.create_dataset(name, data=spectrum_info[name])
        for d in directions:
            group = h5.create_group(d)
            per_dir = [record[d] for record in records]
            group.create_dataset(
                "hdf5_file",
                data=np.array([r["file"] for r in per_dir], dtype=object),
                dtype=string_dtype,
            )
            for name in (
                "luminosity_bolometric_erg_s",
                "valid_pixels",
                "total_pixels",
                "max_tau",
                "median_temp_K",
                "mean_temp_K",
                "density_threshold_code",
                "bh_xyz_code",
            ):
                group.create_dataset(name, data=np.array([r[name] for r in per_dir]))
            if band_info is not None:
                sub = group.create_group("bands")
                for name in ("lnu_erg_s_hz", "nu_lnu_erg_s"):
                    sub.create_dataset(
                        name, data=np.vstack([r["bands"][name] for r in per_dir])
                    )
            if spectrum_info is not None:
                sub = group.create_group("spectra")
                for name in ("lnu_erg_s_hz", "nu_lnu_erg_s"):
                    sub.create_dataset(
                        name, data=np.vstack([r["spectra"][name] for r in per_dir])
                    )
    os.replace(partial, h5_path)
    return csv_path, h5_path
