"""Run settings for the TDE radiative post-processing."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .bands import DEFAULT_OBSERVATION_BANDS, ObservationBand
from .opacity import DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T
from .snapshot import BoxSettings

MODES = ("tau1", "grey", "grey-therm", "multifreq")
DEFAULT_MODE = "multifreq"
GRID_DTYPES = ("auto", "float32", "float64")
POPULATIONS = ("eos", "saha")
DEFAULT_POPULATIONS = "eos"


@dataclass
class RTSettings:
    """Everything that influences a run; every field is recorded in the HDF5 output."""

    mode: str = DEFAULT_MODE
    direction: str = "z"
    image_size: int = 1024
    los_steps: int = 1024
    output_dir: Path = Path("athenak_rt_output")

    # Ray-grid box (explicit or padded dense-gas bounding box).
    box: BoxSettings = field(default_factory=BoxSettings)

    # Thermodynamics and opacity tables.
    eos_table: Optional[Path] = None  # None: use the path from the snapshot header
    mesa_high_t: Path = DEFAULT_MESA_HIGH_T
    mesa_low_t: Path = DEFAULT_MESA_LOW_T

    # Cell selection: rho_code > factor * dfloor (strict), or an explicit code value.
    density_threshold_factor: float = 10.0
    density_threshold_code: Optional[float] = None

    # Black-hole excision sphere (radius from the header unless overridden).
    bh_mask: bool = True
    bh_mask_radius: Optional[float] = None

    # Transfer.
    tau_photosphere: float = 1.0  # tau1 mode
    tau_stop: float = 30.0  # grey / grey-therm / multifreq truncation
    nfreq: int = 73  # multifreq photon-energy grid
    emin_ev: float = 0.1
    emax_ev: float = 1000.0
    scattering: bool = True  # multifreq: thermalization against sigma_T n_e
    # H/He populations for grey-therm and multifreq: "eos" (fractions of the EOS
    # table, falls back to Saha with a message if the table lacks them) or "saha".
    populations: str = DEFAULT_POPULATIONS

    # Storage precision of the resampled cube ("auto": float32 at >= 1024 samples).
    grid_dtype: str = "auto"

    observation_bands: Tuple[ObservationBand, ...] = DEFAULT_OBSERVATION_BANDS
    hdf5_compression: Optional[str] = "lzf"
    threads: Optional[int] = None

    def validate(self) -> None:
        if self.mode not in MODES:
            raise RuntimeError(f"Unknown mode {self.mode!r}; choose from {MODES}.")
        if (
            self.direction.lstrip("-") not in {"x", "y", "z"}
            or self.direction.count("-") > 1
        ):
            raise RuntimeError(f"Unsupported direction {self.direction!r}.")
        if self.image_size < 2 or self.los_steps < 2:
            raise RuntimeError("image_size and los_steps must both be >= 2.")
        if self.populations not in POPULATIONS:
            raise RuntimeError(
                f"populations must be one of {POPULATIONS}, got {self.populations!r}."
            )
        if self.grid_dtype not in GRID_DTYPES:
            raise RuntimeError(f"grid_dtype must be one of {GRID_DTYPES}.")
        if self.tau_photosphere <= 0.0 or self.tau_stop <= 0.0:
            raise RuntimeError("tau_photosphere and tau_stop must be positive.")
        if self.density_threshold_factor < 0.0:
            raise RuntimeError("density_threshold_factor must be >= 0.")
        if self.nfreq < 2:
            raise RuntimeError("nfreq must be >= 2.")
        if not 0.0 < self.emin_ev < self.emax_ev:
            raise RuntimeError("The photon energies need 0 < emin_ev < emax_ev.")
        if self.threads is not None and self.threads < 1:
            raise RuntimeError("threads must be >= 1.")

    # Settings whose names would collide with the resolved values the output records.
    _ATTRIBUTE_RENAMES = {
        "density_threshold_code": "density_threshold_code_override",
        "bh_mask_radius": "bh_mask_radius_override",
        "eos_table": "eos_table_setting",
    }

    def as_attributes(self) -> Dict[str, Any]:
        """Flat, HDF5-attribute-friendly view of every setting."""
        out: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            name = self._ATTRIBUTE_RENAMES.get(item.name, item.name)
            if item.name == "box":
                out["box_explicit"] = (
                    "" if value.box is None else " ".join(f"{v:.17g}" for v in value.box)
                )
                out["auto_box_padding_code"] = float(value.padding_code)
                out["auto_box_padding_fraction"] = float(value.padding_fraction)
                out["auto_box_min_width_code"] = float(value.min_width_code)
                out["auto_box_clip_to_mesh"] = bool(value.clip_to_mesh)
            elif item.name == "observation_bands":
                out["observation_bands"] = " ".join(band.label for band in value)
            elif value is None:
                out[name] = ""
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out
