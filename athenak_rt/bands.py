"""Observation bands (NIR/optical/UV/X-ray) at which L_nu is reported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .constants import C_CGS, EV_CGS, H_CGS


@dataclass(frozen=True)
class ObservationBand:
    label: str
    category: str
    frequency_hz: float


DEFAULT_OBSERVATION_BANDS: Tuple[ObservationBand, ...] = (
    ObservationBand("nir_J", "NIR", C_CGS / (1250.0e-7)),
    ObservationBand("nir_H", "NIR", C_CGS / (1650.0e-7)),
    ObservationBand("nir_K", "NIR", C_CGS / (2200.0e-7)),
    ObservationBand("optical_u", "optical", C_CGS / (355.0e-7)),
    ObservationBand("optical_g", "optical", C_CGS / (475.0e-7)),
    ObservationBand("optical_r", "optical", C_CGS / (622.0e-7)),
    ObservationBand("optical_i", "optical", C_CGS / (763.0e-7)),
    ObservationBand("uv_UVW1", "UV", C_CGS / (260.0e-7)),
    ObservationBand("uv_UVM2", "UV", C_CGS / (224.6e-7)),
    ObservationBand("uv_UVW2", "UV", C_CGS / (192.8e-7)),
    ObservationBand("uv_FUV150", "UV", C_CGS / (150.0e-7)),
    ObservationBand("xray_0p3keV", "X-ray", 0.3e3 * EV_CGS / H_CGS),
    ObservationBand("xray_1keV", "X-ray", 1.0e3 * EV_CGS / H_CGS),
    ObservationBand("xray_2keV", "X-ray", 2.0e3 * EV_CGS / H_CGS),
)


def band_metadata(bands: Sequence[ObservationBand]):
    """labels, categories, frequencies [Hz], wavelengths [nm], energies [eV]."""
    labels = tuple(band.label for band in bands)
    categories = tuple(band.category for band in bands)
    frequencies_hz = np.array([band.frequency_hz for band in bands], dtype=np.float64)
    wavelengths_nm = C_CGS / frequencies_hz / 1.0e-7
    energies_ev = H_CGS * frequencies_hz / EV_CGS
    return labels, categories, frequencies_hz, wavelengths_nm, energies_ev


def band_lnu_from_spectrum(
    frequencies_hz: np.ndarray,
    lnu_erg_s_hz: np.ndarray,
    band_frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Log-log linear interpolation of a spectrum onto the band frequencies.

    Bands outside the tabulated range, or where the spectrum is not positive,
    get 0.
    """
    nu = np.asarray(frequencies_hz, dtype=np.float64)
    lnu = np.asarray(lnu_erg_s_hz, dtype=np.float64)
    out = np.zeros(len(band_frequencies_hz), dtype=np.float64)
    positive = lnu > 0.0
    if np.count_nonzero(positive) < 2:
        return out
    log_nu = np.log(nu[positive])
    log_lnu = np.log(lnu[positive])
    for i, target in enumerate(band_frequencies_hz):
        if target < nu[positive][0] or target > nu[positive][-1]:
            continue
        out[i] = float(np.exp(np.interp(np.log(target), log_nu, log_lnu)))
    return out
