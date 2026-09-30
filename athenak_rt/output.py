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

A run also writes ``rt_lightcurve_<mode>.csv`` and ``rt_lightcurve_<mode>.h5``
(one row per dump, every direction) once at least two dumps qualify.  They are
assembled from the per-dump files: every product in the output directory for
the same dump-file prefix, mode and directions whose recorded settings, table
hashes and dump identity match the run is included, whether it was computed
now or earlier.
"""

from __future__ import annotations

import functools
import hashlib
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

# Settings that do not enter the product of a mode (checked against the kernels
# in pipeline.py): the transfer parameters the mode never reads, the bands where
# the mode writes none, and the Rosseland tables in multifreq mode.
_MODE_IGNORED_SETTINGS = {
    "tau1": ("tau_stop", "nfreq", "emin_ev", "emax_ev", "scattering", "populations"),
    "grey": (
        "tau_photosphere", "nfreq", "emin_ev", "emax_ev", "scattering",
        "populations", "observation_bands",
    ),
    "grey-therm": (
        "tau_photosphere", "nfreq", "emin_ev", "emax_ev", "scattering",
        "observation_bands",
    ),
    "multifreq": ("tau_photosphere", "mesa_high_t", "mesa_low_t"),
}

# Table files that identify a product besides the settings: (hash key, setting
# recorded by older products, modes in which the table matters).
_TABLES = (
    ("eos_table", "eos_table_setting", ("tau1", "grey", "grey-therm", "multifreq")),
    ("mesa_high_t", "mesa_high_t", ("tau1", "grey", "grey-therm")),
    ("mesa_low_t", "mesa_low_t", ("tau1", "grey", "grey-therm")),
)


@functools.lru_cache(maxsize=32)
def _sha256_of(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path) -> str:
    """SHA-256 of a file (cached per path, size and modification time)."""
    path = Path(path).resolve()
    info = path.stat()
    return _sha256_of(str(path), info.st_size, info.st_mtime_ns)


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
        for table, digest in result.table_sha256.items():
            attrs[f"{table}_sha256"] = digest
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




def stored_attributes(path: Path) -> Optional[dict]:
    """Root attributes of a product, or None if it cannot be read."""
    import h5py

    try:
        with h5py.File(path, "r") as h5:
            return dict(h5.attrs)
    except OSError:
        return None


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    return value


def _differs(old, new) -> bool:
    old = _plain(old)
    if isinstance(new, float) and new != new and old != old:
        return False  # both NaN
    return not bool(np.all(np.asarray(old) == np.asarray(new)))


def attributes_mismatch(
    stored: Mapping[str, object],
    settings,
    dump: Optional[Mapping[str, object]] = None,
    table_hashes: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """None if a product with attributes ``stored`` matches, else the first difference.

    * ``settings``: only those that enter the product of ``settings.mode`` are
      compared (not the output directory, thread count or compression, nor the
      transfer parameters the mode does not read).
    * ``dump`` (optional): ``snapshot_file``, ``snapshot_index``, ``time_code``
      and ``cycle`` of the input dump, compared with the values in the product.
    * ``table_hashes`` (optional): SHA-256 of the ``eos_table`` and ``mesa_high_t`` /
      ``mesa_low_t`` files in use.  Where the product records a hash it is compared
      instead of the path; products without hashes (older versions) and calls
      without ``table_hashes`` compare the recorded setting (the path) as before.
    """
    ignored = set(_REUSE_IGNORED_SETTINGS) | set(_MODE_IGNORED_SETTINGS[settings.mode])
    attrs = settings.as_attributes()
    hashed = set()
    for table, setting_name, modes in _TABLES:
        recorded = f"{table}_sha256"
        if table_hashes is not None and table in table_hashes and recorded in stored:
            hashed.add(setting_name)
            if settings.mode not in modes:
                continue
            if _plain(stored[recorded]) != table_hashes[table]:
                return f"{table} file differs (SHA-256 {_plain(stored[recorded])[:12]} there, {table_hashes[table][:12]} now)"
    for name, value in attrs.items():
        if name in ignored or name in hashed:
            continue
        if name not in stored:
            return f"{name} is not recorded"
        if _differs(stored[name], value):
            return f"{name} = {_plain(stored[name])!r} there, {value!r} now"
    if dump is not None:
        for name in ("snapshot_file", "snapshot_index", "cycle", "time_code"):
            if name not in dump:
                continue
            if name not in stored:
                return f"{name} is not recorded"
            if _differs(stored[name], dump[name]):
                return f"input dump differs: {name} = {_plain(stored[name])!r} there, {dump[name]!r} now"
    return None


def existing_output_mismatch(
    path: Path,
    settings,
    dump: Optional[Mapping[str, object]] = None,
    table_hashes: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """None if ``path`` was written with ``settings``, else the first difference.

    See ``attributes_mismatch`` for what is compared.
    """
    stored = stored_attributes(path)
    if stored is None:
        return "cannot be read"
    return attributes_mismatch(stored, settings, dump, table_hashes)


def _read_product(path: Path) -> dict:
    """The scalar results, band and spectrum arrays of one per-dump product."""
    import h5py

    with h5py.File(path, "r") as h5:
        a = h5.attrs
        record = {
            "attrs": dict(a),
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


def read_product(path: Path) -> dict:
    """The record of one product (see ``write_lightcurve_records``)."""
    return _read_product(path)


def write_lightcurve(
    output_dir: Path,
    mode: str,
    directions: Sequence[str],
    rows: Sequence[Mapping[str, Path]],
    provenance: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, Path]:
    """Light-curve table (CSV) and HDF5 file from per-dump product files.

    ``rows`` holds, per dump in time order, the per-dump product of every
    direction.  See ``write_lightcurve_records``.
    """
    records = [{d: _read_product(row[d]) for d in directions} for row in rows]
    return write_lightcurve_records(output_dir, mode, directions, records, provenance)


def write_lightcurve_records(
    output_dir: Path,
    mode: str,
    directions: Sequence[str],
    records: Sequence[Mapping[str, dict]],
    provenance: Optional[Mapping[str, str]] = None,
) -> Tuple[Path, Path]:
    """Light-curve table (CSV) and HDF5 file, one row per dump.

    ``records`` holds, per dump in time order, the ``read_product`` record of
    every direction.  CSV columns: dump, cycle, time_code, time_s, then
    ``L_bol_<dir>`` for every direction and ``nuLnu_<band>_<dir>`` for every
    band and direction (erg/s; ``-y`` is written ``minus_y``).
    """
    import h5py

    directions = list(directions)
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
