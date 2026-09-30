"""HDF5 and light-curve writers.

One HDF5 file per (snapshot, mode, direction).  Every run setting, the
snapshot metadata, the ray box and the table paths are stored as root
attributes so a product is self-describing.  Layout::

    /                       attributes: settings + provenance + L_bol
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
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .constants import (
    LUMINOSITY_BOLOMETRIC_FORMULA,
    LUMINOSITY_NORMALIZATION,
    LUMINOSITY_REFERENCE,
    LUMINOSITY_SPECTRAL_FORMULA,
    PROJECTED_LUMINOSITY_FACTOR,
)


def product_stem(snapshot: Path, mode: str, direction: str) -> str:
    return f"{Path(snapshot).stem}.rt_{mode}_{direction}"


def _set_attrs(target, values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (tuple, list)):
            value = np.asarray(value)
        target.attrs[key] = value


def write_snapshot_hdf5(output_dir: Path, result, settings) -> Path:
    import h5py

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (
        output_dir / f"{product_stem(result.snapshot, result.mode, result.direction)}.h5"
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression = settings.hdf5_compression
    if compression in ("", "none"):
        compression = None

    with h5py.File(path, "w") as h5:
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
    return path


def write_lightcurve(
    output_dir: Path, mode: str, direction: str, results: Sequence
) -> Path:
    path = Path(output_dir) / f"rt_lightcurve_{mode}_{direction}.dat"
    with path.open("w") as fp:
        fp.write(f"# athenak_rt {__version__} mode={mode} direction={direction}\n")
        fp.write(f"# luminosity_normalization = {LUMINOSITY_NORMALIZATION}\n")
        fp.write(f"# luminosity_reference = {LUMINOSITY_REFERENCE}\n")
        fp.write(f"# luminosity_bolometric_formula = {LUMINOSITY_BOLOMETRIC_FORMULA}\n")
        fp.write(f"# projected_luminosity_factor = {PROJECTED_LUMINOSITY_FACTOR:.1f}\n")
        fp.write(
            "# columns = snapshot_index time_code time_s luminosity_erg_s "
            "valid_pixels total_pixels max_tau median_T_K mean_T_K "
            "area_pixel_cm2 density_threshold_code bh_x_code bh_y_code bh_z_code file\n"
        )
        for r in results:
            fp.write(
                f"{r.snapshot_index:d} {r.time_code:.16e} {r.time_s:.16e} "
                f"{r.luminosity:.16e} {r.valid_pixels:d} {r.total_pixels:d} "
                f"{r.max_tau:.16e} {r.median_temp:.16e} {r.mean_temp:.16e} "
                f"{r.area_cm2:.16e} {r.density_threshold_code:.16e} "
                f"{r.bh_xyz_code[0]:.16e} {r.bh_xyz_code[1]:.16e} "
                f"{r.bh_xyz_code[2]:.16e} {r.snapshot.name}\n"
            )
    return path


def write_summary_hdf5(
    output_dir: Path, mode: str, direction: str, results: Sequence
) -> Path:
    """Stack L_bol, band L_nu and (multifreq) spectra of all processed snapshots."""
    import h5py

    path = Path(output_dir) / f"rt_summary_{mode}_{direction}.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    results = list(results)
    with h5py.File(path, "w") as h5:
        h5.attrs["athenak_rt_version"] = __version__
        h5.attrs["mode"] = mode
        h5.attrs["direction"] = direction
        h5.attrs["luminosity_normalization"] = LUMINOSITY_NORMALIZATION
        h5.attrs["luminosity_reference"] = LUMINOSITY_REFERENCE
        h5.attrs["projected_luminosity_factor"] = PROJECTED_LUMINOSITY_FACTOR
        h5.create_dataset(
            "snapshot_index",
            data=np.array([r.snapshot_index for r in results], dtype=np.int64),
        )
        h5.create_dataset("time_code", data=np.array([r.time_code for r in results]))
        h5.create_dataset("time_s", data=np.array([r.time_s for r in results]))
        h5.create_dataset(
            "luminosity_bolometric_erg_s", data=np.array([r.luminosity for r in results])
        )
        h5.create_dataset(
            "snapshot_file",
            data=np.array([r.snapshot.name for r in results], dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "hdf5_file",
            data=np.array(
                ["" if r.hdf5_path is None else r.hdf5_path.name for r in results],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        with_bands = [r for r in results if r.bands is not None]
        if with_bands:
            first = with_bands[0].bands
            bands = h5.create_group("bands")
            bands.create_dataset(
                "label", data=np.array(first["labels"], dtype=object), dtype=string_dtype
            )
            bands.create_dataset(
                "category",
                data=np.array(first["categories"], dtype=object),
                dtype=string_dtype,
            )
            for name in ("frequency_hz", "wavelength_nm", "energy_ev"):
                bands.create_dataset(name, data=first[name])
            bands.create_dataset(
                "lnu_erg_s_hz",
                data=np.vstack([r.bands["lnu_erg_s_hz"] for r in with_bands]),
            )
            bands.create_dataset(
                "nu_lnu_erg_s",
                data=np.vstack([r.bands["nu_lnu_erg_s"] for r in with_bands]),
            )
        with_spec = [r for r in results if r.spectrum is not None]
        if with_spec:
            spectra = h5.create_group("spectra")
            spectra.create_dataset("energy_ev", data=with_spec[0].spectrum["energy_ev"])
            spectra.create_dataset(
                "frequency_hz", data=with_spec[0].spectrum["frequency_hz"]
            )
            spectra.create_dataset(
                "lnu_erg_s_hz",
                data=np.vstack([r.spectrum["lnu_erg_s_hz"] for r in with_spec]),
            )
            spectra.create_dataset(
                "nu_lnu_erg_s",
                data=np.vstack([r.spectrum["nu_lnu_erg_s"] for r in with_spec]),
            )
    return path
