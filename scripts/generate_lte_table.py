#!/usr/bin/env python3

"""Generate AthenaK-native LTE EOS tables in cgs units.

The table is 2D in natural-log axes:
  - logrho = ln(rho_cgs)
  - logtemp = ln(T_kelvin)

The maintained repo-owned generation jobs are hardcoded below in base-10 bounds
for convenience. These bounds are converted exactly once to natural-log table
axes before each AthenaK table is written. The native table metadata records
this as log_axis_base = e.

This generator builds the maintained H/He LTE table products following:
  - Tomida et al. 2013, Appendix 1
  - Andalman et al. 2025, EOS section
  - Saumon, Chabrier, and Van Horn 1995
  - Chabrier et al. 2021
  - MESA/HELM-style relativistic ideal e-/e+ pairs

Maintained public models:
  - t13:
      Tomida-like partition-function LTE equilibrium with H2, H, H+, He, He+, He2+,
      and e-, plus optional radiation contributions and H2 vibrational ZPE subtraction
      in the runtime inversion energy table following Andalman et al. 2025. Above
      log10(T/K) = 7, the electron gas is replaced by a HELM-style relativistic
      ideal e-/e+ gas.
  - scvh_t13_cp_helm_union:
      Use SCvH wherever available, then the CP-like fully ionized corner where its
      own validity cuts pass, then a HELM-style fully ionized fallback only in the
      hot dense top-right corner, with low-density fallback still coming from T13.
  - chabrier2021_t13_helm_union:
      Use T13 as the low-density fallback, promote the Chabrier 2021 H/He table
      through the dense-fluid / pressure-dissociation regime, and keep a
      HELM-style fully ionized fallback only in the hot dense top-right corner
      beyond the tabulated Chabrier-family coverage.

Required runtime fields:
  - logpress
  - logeps
  - logcs2
  - gamma1
  - gamma3m1
  - xh2
  - xion
  - xhe1
  - xhe2
  - mu
  - beta_rad

The new t13 family also stores extra diagnostic fields when practical.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import roots_genlaguerre

from chabrier2021_eos import (
    CHABRIER2021_T13_BLEND_RHO_CGS_MIN,
    CHABRIER2021_T13_SWITCH_RHO_CGS,
    chabrier2021_thermo,
)
from scvh95_eos import (
    SCVH95_INTERP_LOGP_STEP,
    SCVH95_INTERP_PPT_LOGP_STEP,
    scvh95_thermo,
)

CHABRIER2021_DENSE_GAP_FILL_RHO_CGS_MIN = 1.0e-1
CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MIN = 2.0
CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MAX = 8.0
CHABRIER2021_DENSE_GAP_FILL_MAX_GAP_CELLS = 48
CHABRIER2021_DENSE_GAP_FILL_EDGE_CELLS = 8
CHABRIER2021_DENSE_ONSET_BLEND_CELLS = 16


K_B = 1.380649e-16
H_PLANCK = 6.62607015e-27
M_E = 9.1093837015e-28
M_H = 1.6735575e-24
M_H2 = 2.0 * M_H
M_HE = 4.0 * M_H
EV_TO_ERG = 1.602176634e-12
A_RAD = 7.5657e-15
AMU = 1.66053906660e-24
Q_E_ESU = 4.80320471257e-10
Q_E_SQ_CGS = Q_E_ESU**2
C_LIGHT = 2.99792458e10
HBAR = H_PLANCK / (2.0 * math.pi)
M_E_C2 = M_E * C_LIGHT**2
ELECTRON_COMPTON_LENGTH = HBAR / (M_E * C_LIGHT)

# Simplified H/He ionization-only LTE constants.
CHI_H = 13.5984 * EV_TO_ERG
CHI_HE1 = 24.5874 * EV_TO_ERG
CHI_HE2 = 54.4178 * EV_TO_ERG
U_H0 = 2.0
U_HP = 1.0
U_HE0 = 1.0
U_HE1 = 2.0
U_HE2 = 1.0
SAHA_PREFAC = (2.0 * math.pi * M_E * K_B / H_PLANCK**2) ** 1.5
PHI_H_COEFF = 2.0 * U_HP / U_H0
PHI_HE1_COEFF = 2.0 * U_HE1 / U_HE0
PHI_HE2_COEFF = 2.0 * U_HE2 / U_HE1

# Tomida et al. 2013 Appendix 1 constants.
THETA_ROT_H2 = 170.64
THETA_VIB_H2 = 5984.48
CHI_DIS_H2 = 7.17e-12
CHI_ION_H = 2.18e-11
CHI_ION_HE1 = 3.94e-11
CHI_ION_HE2 = 8.72e-11
H2_VIB_ZPE = 0.5 * K_B * THETA_VIB_H2

LN10 = math.log(10.0)
SCVH_T13_SWITCH_RHO_CGS = 5.0e-2
FI_XION_MIN = 0.99
FI_XHE2_MIN = 0.90
CP_LIQUID_GAMMA_MAX = 175.0
CP_CLASSICAL_T_OVER_TP_MIN = 3.0
HELM_HOT_FALLBACK_LOG10_TEMP_MIN = 7.0
CHABRIER2021_HELM_PREFERRED_LOG10_TEMP_MIN = 8.0
PAIR_ACTIVE_LOG10_TEMP_MIN = 7.0
PAIR_QUAD_N = 96
PAIR_BISECTION_ITERS = 72
PAIR_STATE_CHUNK_SIZE = 8192
PAIR_LOG_NUMBER_FLOOR = math.log(1.0e-300)
PAIR_DIAGNOSTIC_FIELDS = (
    "ppair",
    "epair_kin_over_rho",
    "epair_rest_over_rho",
    "logne_minus",
    "logne_plus",
    "eta_e",
    "pair_pressure_fraction",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EOS_TABLE_DIR = REPO_ROOT / "eos_tables"

# Species order used for thermodynamic bookkeeping.
IH2 = 0
IH = 1
IHP = 2
IHE = 3
IHE1 = 4
IHE2 = 5
IE = 6
NSPEC = 7
SPECIES = ("H2", "H", "Hp", "He", "He1", "He2", "e")

# The derivative matrix printed in Tomida et al. 2013 Appendix 1 is chemically
# consistent only if the unknown-vector ordering is interpreted as:
#   [H2, H, H+, e, He, He+, He2+]
# even though the adjacent column vector in the paper is typeset differently.
MH2 = 0
MH = 1
MHP = 2
ME = 3
MHE = 4
MHE1 = 5
MHE2 = 6

LTE_TABLE_JOBS = (
    {
        "name": "lte_t13_prad",
        "output": EOS_TABLE_DIR / "lte_t13_prad_eos.table",
        "model": "t13",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 2.0,
        "log10_temp_min": -0.3,
        "log10_temp_max": 9.0,
        "x_h": 0.70,
        "y_he": 0.30,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
    },
    {
        "name": "scvh_t13_cp_helm_union_prad_640",
        "output": EOS_TABLE_DIR / "scvh_t13_cp_helm_union_prad_640.table",
        "model": "scvh_t13_cp_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 2.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 9.0,
        "x_h": 0.70,
        "y_he": 0.30,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 3.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.70,
        "y_he": 0.30,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
    },
    {
        # Radiation-free companion of the production table: gas thermodynamics only, so
        # an explicit radiation module can own the
        # a_R T^4 budget without double-counting the *_prad table's built-in radiation.
        "name": "chabrier2021_t13_helm_union_640",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_640.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 3.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.70,
        "y_he": 0.30,
        "radiation_pressure": False,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
    },
    # Helium-enriched companions of the production table for the star-BH composition
    # campaign (uniform composition per run).  Z is not in the EOS, so the metals are
    # folded into Y (closest mean molecular weight): the nuclear network then carries
    # (X_H, Y_He, X_CNO) = (0.38, 0.60, 0.02) and (0.00, 0.98, 0.02) respectively.  The
    # Chabrier dense-fluid block is built from the pure H and He tables with the additive
    # volume law because the published mixture tables only span Y = 0.275-0.297.
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X038Y062",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X038Y062.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 4.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.38,
        "y_he": 0.62,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X000Y100",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X000Y100.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 5.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.0,
        "y_he": 1.0,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X745Y255",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X745Y255.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 4.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.745,
        "y_he": 0.255,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X550Y450",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X550Y450.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 4.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.55,
        "y_he": 0.45,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X200Y800",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X200Y800.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 4.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.2,
        "y_he": 0.8,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
    {
        "name": "chabrier2021_t13_helm_union_prad_640_X620Y380",
        "output": EOS_TABLE_DIR / "chabrier2021_t13_helm_union_prad_640_X620Y380.table",
        "model": "chabrier2021_t13_helm_union",
        "nrho": 640,
        "ntemp": 640,
        "log10_rho_min": -20.0,
        "log10_rho_max": 4.0,
        "log10_temp_min": 0.0,
        "log10_temp_max": 10.0,
        "x_h": 0.62,
        "y_he": 0.38,
        "radiation_pressure": True,
        "zpe_subtract": None,
        "scvh_data_dir": None,
        "scvh_logp_step": SCVH95_INTERP_LOGP_STEP,
        "scvh_ppt_logp_step": SCVH95_INTERP_PPT_LOGP_STEP,
        "chabrier_mixing": "additive_volume",
    },
)


def safe_log(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, -np.inf, dtype=np.float64)
    mask = values > 0.0
    out[mask] = np.log(values[mask])
    return out


def pairless_diagnostic_fields(
    template: np.ndarray,
    logne_minus: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Finite defaults for EOS branches that do not include thermodynamic pairs."""
    shape = template.shape
    zeros = np.zeros(shape, dtype=np.float64)
    if logne_minus is None:
        logne_minus_arr = np.full(shape, PAIR_LOG_NUMBER_FLOOR, dtype=np.float64)
    else:
        logne_minus_arr = np.asarray(logne_minus, dtype=np.float64)
        logne_minus_arr = np.where(
            np.isfinite(logne_minus_arr),
            logne_minus_arr,
            PAIR_LOG_NUMBER_FLOOR,
        )
    return {
        "ppair": zeros.copy(),
        "epair_kin_over_rho": zeros.copy(),
        "epair_rest_over_rho": zeros.copy(),
        "logne_minus": logne_minus_arr,
        "logne_plus": np.full(shape, PAIR_LOG_NUMBER_FLOOR, dtype=np.float64),
        "eta_e": zeros.copy(),
        "pair_pressure_fraction": zeros.copy(),
    }


def ensure_pair_diagnostics(
    fields: dict[str, np.ndarray],
    template_name: str = "logpress",
    logne_name: str | None = "logne",
) -> None:
    """Add finite pair diagnostic arrays to a branch field dictionary in-place."""
    template = fields[template_name]
    logne_minus = fields.get(logne_name) if logne_name is not None else None
    defaults = pairless_diagnostic_fields(template, logne_minus=logne_minus)
    for name, values in defaults.items():
        if name not in fields:
            fields[name] = values
        elif fields[name].shape == template.shape:
            fields[name] = np.where(np.isfinite(fields[name]), fields[name], values)


def finite_difference(values: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    deriv = np.empty_like(values)
    slicer_mid = [slice(None)] * values.ndim
    slicer_lo = [slice(None)] * values.ndim
    slicer_hi = [slice(None)] * values.ndim

    slicer_mid[axis] = slice(1, -1)
    slicer_lo[axis] = slice(0, -2)
    slicer_hi[axis] = slice(2, None)
    delta_mid = coords[2:] - coords[:-2]
    reshape_mid = [1] * values.ndim
    reshape_mid[axis] = delta_mid.size
    deriv[tuple(slicer_mid)] = (
        values[tuple(slicer_hi)] - values[tuple(slicer_lo)]
    ) / delta_mid.reshape(reshape_mid)

    slicer0 = [slice(None)] * values.ndim
    slicer1 = [slice(None)] * values.ndim
    slicer2 = [slice(None)] * values.ndim
    slicer0[axis] = 0
    slicer1[axis] = 1
    slicer2[axis] = 2
    deriv[tuple(slicer0)] = (
        -3.0 * values[tuple(slicer0)] + 4.0 * values[tuple(slicer1)] - values[tuple(slicer2)]
    ) / (coords[2] - coords[0])

    slicer0[axis] = -1
    slicer1[axis] = -2
    slicer2[axis] = -3
    deriv[tuple(slicer0)] = (
        3.0 * values[tuple(slicer0)] - 4.0 * values[tuple(slicer1)] + values[tuple(slicer2)]
    ) / (coords[-1] - coords[-3])
    return deriv


def ensure_parent(path: os.PathLike[str] | str) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_block(fp, begin: str, lines: Iterable[str], end: str) -> None:
    fp.write(f"<{begin}>\n".encode("ascii"))
    for line in lines:
        fp.write(f"{line}\n".encode("ascii"))
    fp.write(f"<{end}>\n".encode("ascii"))


def exp_clip(log_values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(log_values, -700.0, 700.0))

def stable_ratio_fraction(log_num: np.ndarray, log_den: np.ndarray) -> np.ndarray:
    diff = np.clip(log_den - log_num, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(diff))


def saha_rhs(temp: np.ndarray, chi: float, coeff: float) -> np.ndarray:
    return coeff * SAHA_PREFAC * np.power(temp, 1.5) * np.exp(-chi / (K_B * temp))


def he_charge_state_fractions(
    ne: np.ndarray, phi_he1: np.ndarray, phi_he2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_ne = safe_log(np.maximum(ne, 1.0e-300))
    log_phi_he1 = safe_log(phi_he1)
    log_phi_he2 = safe_log(phi_he2)

    logw0 = np.zeros_like(ne)
    logw1 = log_phi_he1 - log_ne
    logw2 = log_phi_he1 + log_phi_he2 - 2.0 * log_ne
    logw_max = np.maximum(logw0, np.maximum(logw1, logw2))

    w0 = np.exp(np.clip(logw0 - logw_max, -700.0, 0.0))
    w1 = np.exp(np.clip(logw1 - logw_max, -700.0, 0.0))
    w2 = np.exp(np.clip(logw2 - logw_max, -700.0, 0.0))
    denom = w0 + w1 + w2
    he0 = w0 / denom
    he1 = w1 / denom
    he2 = w2 / denom
    return he0, he1, he2


def charge_residual(
    ne: np.ndarray,
    n_h_tot: np.ndarray,
    n_he_tot: np.ndarray,
    phi_h: np.ndarray,
    phi_he1: np.ndarray,
    phi_he2: np.ndarray,
) -> np.ndarray:
    ne_safe = np.maximum(ne, 1.0e-300)
    log_ne = safe_log(ne_safe)
    log_phi_h = safe_log(phi_h)
    n_hp = n_h_tot * stable_ratio_fraction(log_phi_h, log_ne)
    _, he1_frac, he2_frac = he_charge_state_fractions(ne_safe, phi_he1, phi_he2)
    n_he1 = n_he_tot * he1_frac
    n_he2 = n_he_tot * he2_frac
    return n_hp + n_he1 + 2.0 * n_he2 - ne


def solve_simple_lte_composition(
    rho: np.ndarray, temp: np.ndarray, x_h: float, y_he: float
) -> dict[str, np.ndarray]:
    n_h_tot = x_h * rho / M_H
    n_he_tot = y_he * rho / (4.0 * M_H)

    phi_h = saha_rhs(temp, CHI_H, PHI_H_COEFF)
    phi_he1 = saha_rhs(temp, CHI_HE1, PHI_HE1_COEFF)
    phi_he2 = saha_rhs(temp, CHI_HE2, PHI_HE2_COEFF)

    ne_lo = np.full_like(rho, 1.0e-300)
    ne_hi = np.maximum(n_h_tot + 2.0 * n_he_tot, 1.0e-300)

    f_lo = charge_residual(ne_lo, n_h_tot, n_he_tot, phi_h, phi_he1, phi_he2)
    f_hi = charge_residual(ne_hi, n_h_tot, n_he_tot, phi_h, phi_he1, phi_he2)

    neutral_mask = f_lo <= 0.0
    ionized_mask = (~neutral_mask) & (f_hi >= 0.0)
    bracket_mask = (~neutral_mask) & (~ionized_mask)

    if np.any(f_hi[bracket_mask] > 0.0) or np.any(f_lo[bracket_mask] < 0.0):
        raise RuntimeError("Failed to bracket the simplified LTE electron-density root.")

    ne = np.zeros_like(rho)
    ne = np.where(ionized_mask, ne_hi, ne)
    ne = np.where(bracket_mask, 0.5 * (ne_lo + ne_hi), ne)

    for _ in range(96):
        if not np.any(bracket_mask):
            break
        ne_mid = 0.5 * (ne_lo + ne_hi)
        f_mid = charge_residual(ne_mid, n_h_tot, n_he_tot, phi_h, phi_he1, phi_he2)
        use_right = bracket_mask & (f_mid > 0.0)
        use_left = bracket_mask & (~use_right)
        ne_lo = np.where(use_right, ne_mid, ne_lo)
        ne_hi = np.where(use_left, ne_mid, ne_hi)

    ne = np.where(bracket_mask, 0.5 * (ne_lo + ne_hi), ne)
    ne_safe = np.maximum(ne, 1.0e-300)
    n_hp = n_h_tot * stable_ratio_fraction(safe_log(phi_h), safe_log(ne_safe))
    n_h0 = n_h_tot - n_hp
    he0_frac, he1_frac, he2_frac = he_charge_state_fractions(ne_safe, phi_he1, phi_he2)
    n_he0 = n_he_tot * he0_frac
    n_he1 = n_he_tot * he1_frac
    n_he2 = n_he_tot * he2_frac

    return {
        "n_h_tot": n_h_tot,
        "n_he_tot": n_he_tot,
        "n_e": ne,
        "n_h0": n_h0,
        "n_hp": n_hp,
        "n_he0": n_he0,
        "n_he1": n_he1,
        "n_he2": n_he2,
        "xion": np.where(n_h_tot > 0.0, n_hp / n_h_tot, 0.0),
        "xhe1": np.where(n_he_tot > 0.0, n_he1 / n_he_tot, 0.0),
        "xhe2": np.where(n_he_tot > 0.0, n_he2 / n_he_tot, 0.0),
    }


def simplified_lte_thermo(
    rho: np.ndarray, temp: np.ndarray, x_h: float, y_he: float, radiation_enabled: bool
) -> dict[str, np.ndarray]:
    comp = solve_simple_lte_composition(rho, temp, x_h, y_he)

    n_tot = (
        comp["n_h0"]
        + comp["n_hp"]
        + comp["n_he0"]
        + comp["n_he1"]
        + comp["n_he2"]
        + comp["n_e"]
    )
    p_gas = n_tot * K_B * temp
    p_rad = (A_RAD / 3.0) * np.power(temp, 4.0) if radiation_enabled else np.zeros_like(temp)
    press = p_gas + p_rad

    u_trans = 1.5 * n_tot * K_B * temp
    u_ion = (
        CHI_H * comp["n_hp"]
        + CHI_HE1 * (comp["n_he1"] + comp["n_he2"])
        + CHI_HE2 * comp["n_he2"]
    )
    u_rad = A_RAD * np.power(temp, 4.0) if radiation_enabled else np.zeros_like(temp)
    eps = (u_trans + u_ion + u_rad) / rho

    logpress = np.log(press)
    logrho = np.log(rho[:, 0])
    logtemp = np.log(temp[0, :])
    chi_rho = finite_difference(logpress, logrho, axis=0)
    chi_t = finite_difference(logpress, logtemp, axis=1)
    deps_dlogt = finite_difference(eps, logtemp, axis=1)
    cv = deps_dlogt / temp
    gamma3m1 = press * chi_t / (rho * temp * cv)
    gamma1 = chi_rho + chi_t * gamma3m1
    cs2 = gamma1 * press / rho

    mu = rho / (n_tot * M_H)
    beta_rad = np.where(press > 0.0, p_rad / press, 0.0)

    if np.any(cv <= 0.0):
        raise RuntimeError("Encountered non-positive cv while generating simplified LTE table.")
    if np.any(press <= 0.0) or np.any(eps <= 0.0) or np.any(cs2 <= 0.0):
        raise RuntimeError("Encountered non-positive thermodynamic quantity in LTE table.")
    if np.any(gamma1 <= 0.0) or np.any(gamma3m1 <= 0.0):
        raise RuntimeError("Encountered non-physical adiabatic exponents in LTE table.")

    return {
        "logpress": logpress,
        "logeps": np.log(eps),
        "logcs2": np.log(cs2),
        "gamma1": gamma1,
        "gamma3m1": gamma3m1,
        "xh2": np.zeros_like(eps),
        "xion": comp["xion"],
        "xhe1": comp["xhe1"],
        "xhe2": comp["xhe2"],
        "mu": mu,
        "beta_rad": beta_rad,
    }


def _log_ztr(temp: np.ndarray, mass: float) -> np.ndarray:
    return 1.5 * np.log(2.0 * math.pi * mass * K_B * temp) - 3.0 * math.log(H_PLANCK)


def _h2_rot_partition(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tomida et al. 2013 Appendix 1, eqs. (A2)-(A4)."""
    tmax = float(np.max(temp))
    jmax = max(16, int(math.ceil(math.sqrt(120.0 * tmax / THETA_ROT_H2) + 8.0)))
    j_even = np.arange(0, jmax + 1, 2, dtype=np.float64)
    j_odd = np.arange(1, jmax + 1, 2, dtype=np.float64)

    e_even = 0.5 * j_even * (j_even + 1.0) * THETA_ROT_H2
    # Evaluate 3 * Z_odd * exp(theta_rot / T) in the shifted form to avoid overflow.
    e_odd_shift = (0.5 * j_odd * (j_odd + 1.0) - 1.0) * THETA_ROT_H2

    temp_2d = temp[None, :]
    terms_even = (2.0 * j_even[:, None] + 1.0) * np.exp(-e_even[:, None] / temp_2d)
    terms_odd_shift = (
        3.0 * (2.0 * j_odd[:, None] + 1.0) * np.exp(-e_odd_shift[:, None] / temp_2d)
    )

    z_even = np.sum(terms_even, axis=0)
    z_odd_shift = np.sum(terms_odd_shift, axis=0)
    dln_even = np.sum(terms_even * (e_even[:, None] / temp_2d), axis=0) / z_even
    dln_odd_shift = (
        np.sum(terms_odd_shift * (e_odd_shift[:, None] / temp_2d), axis=0) / z_odd_shift
    )

    log_z_rot = 0.25 * np.log(z_even) + 0.75 * np.log(z_odd_shift)
    dln_z_rot = 0.25 * dln_even + 0.75 * dln_odd_shift
    return log_z_rot, dln_z_rot


def _h2_vib_partition(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tomida et al. 2013 Appendix 1, eq. (A5)."""
    x = 0.5 * THETA_VIB_H2 / temp
    log_z = np.empty_like(temp)
    large = x > 20.0
    log_z[large] = -x[large]
    ex2 = np.exp(-2.0 * x[~large])
    log_z[~large] = -x[~large] - np.log1p(-ex2)
    dln_z = x / np.tanh(x)
    return log_z, dln_z


def build_t13_partition_terms(logtemp: np.ndarray) -> dict[str, np.ndarray]:
    """Build Tomida 2013 Appendix 1 partition functions and derivatives on logT.

    The partition functions use:
      - fixed ortho:para H2 ratio = 3:1
      - ground electronic states only
      - ideal chemical equilibrium among H2, H, H+, He, He+, He2+, e-

    First derivatives d ln z_i / d ln T are evaluated analytically from the partition
    functions. Second derivatives d^2 ln z_i / d(ln T)^2 are then evaluated numerically
    on the logT grid, following the Andalman et al. 2025 description. These derivatives
    support the partition-derivative closure mode and remain part of the T13
    thermodynamic bookkeeping even when the final closure is taken from finite
    differences of the completed thermo surface.
    """
    temp = np.exp(logtemp)
    dlnz = np.zeros((NSPEC, temp.size), dtype=np.float64)
    logz = np.zeros((NSPEC, temp.size), dtype=np.float64)

    log_z_rot_h2, dln_z_rot_h2 = _h2_rot_partition(temp)
    log_z_vib_h2, dln_z_vib_h2 = _h2_vib_partition(temp)

    logz[IH2] = (
        _log_ztr(temp, M_H2) + log_z_rot_h2 + log_z_vib_h2 + math.log(4.0) + math.log(2.0)
    )
    dlnz[IH2] = 1.5 + dln_z_rot_h2 + dln_z_vib_h2

    logz[IH] = _log_ztr(temp, M_H) + math.log(2.0) + math.log(2.0) - CHI_DIS_H2 / (
        2.0 * K_B * temp
    )
    dlnz[IH] = 1.5 + CHI_DIS_H2 / (2.0 * K_B * temp)

    logz[IHP] = _log_ztr(temp, M_H) + math.log(2.0) + math.log(2.0) - (
        CHI_DIS_H2 + 2.0 * CHI_ION_H
    ) / (2.0 * K_B * temp)
    dlnz[IHP] = 1.5 + (CHI_DIS_H2 + 2.0 * CHI_ION_H) / (2.0 * K_B * temp)

    logz[IHE] = _log_ztr(temp, M_HE)
    dlnz[IHE] = 1.5

    logz[IHE1] = _log_ztr(temp, M_HE) - CHI_ION_HE1 / (K_B * temp)
    dlnz[IHE1] = 1.5 + CHI_ION_HE1 / (K_B * temp)

    logz[IHE2] = _log_ztr(temp, M_HE) - (CHI_ION_HE1 + CHI_ION_HE2) / (K_B * temp)
    dlnz[IHE2] = 1.5 + (CHI_ION_HE1 + CHI_ION_HE2) / (K_B * temp)

    logz[IE] = _log_ztr(temp, M_E) + math.log(2.0)
    dlnz[IE] = 1.5

    d2lnz = finite_difference(dlnz, logtemp, axis=1)

    log_k_dis = 2.0 * logz[IH] - logz[IH2]
    log_k_ion = logz[IHP] + logz[IE] - logz[IH]
    log_k_he1 = logz[IHE1] + logz[IE] - logz[IHE]
    log_k_he2 = logz[IHE2] + logz[IE] - logz[IHE1]

    return {
        "temp": temp,
        "logz": logz,
        "dlnz": dlnz,
        "d2lnz": d2lnz,
        "dlnz_rot_h2": dln_z_rot_h2,
        "dlnz_vib_h2": dln_z_vib_h2,
        "log_k_dis": log_k_dis,
        "log_k_ion": log_k_ion,
        "log_k_he1": log_k_he1,
        "log_k_he2": log_k_he2,
        "k_dis": exp_clip(log_k_dis),
        "k_ion": exp_clip(log_k_ion),
        "k_he1": exp_clip(log_k_he1),
        "k_he2": exp_clip(log_k_he2),
    }


def _t13_he_fractions(log_ne: np.ndarray, log_k_he1: np.ndarray, log_k_he2: np.ndarray):
    logw0 = np.zeros_like(log_ne)
    logw1 = log_k_he1 - log_ne
    logw2 = log_k_he1 + log_k_he2 - 2.0 * log_ne
    logw_max = np.maximum(logw0, np.maximum(logw1, logw2))
    w0 = np.exp(np.clip(logw0 - logw_max, -700.0, 0.0))
    w1 = np.exp(np.clip(logw1 - logw_max, -700.0, 0.0))
    w2 = np.exp(np.clip(logw2 - logw_max, -700.0, 0.0))
    denom = w0 + w1 + w2
    return w0 / denom, w1 / denom, w2 / denom


def _t13_hydrogen_atomic_density(
    ne: np.ndarray, n_h_tot: float, k_dis: np.ndarray, k_ion: np.ndarray
) -> np.ndarray:
    ne_safe = np.maximum(ne, 1.0e-300)
    k_dis_safe = np.maximum(k_dis, 1.0e-300)
    with np.errstate(over="ignore"):
        disc = np.sqrt((ne_safe + k_ion) ** 2 + 8.0 * n_h_tot * ne_safe**2 / k_dis_safe)
    return 2.0 * n_h_tot * ne_safe / (disc + ne_safe + k_ion)


def _t13_charge_residual(
    ne: np.ndarray,
    n_h_tot: float,
    n_he_tot: float,
    k_dis: np.ndarray,
    k_ion: np.ndarray,
    log_k_he1: np.ndarray,
    log_k_he2: np.ndarray,
) -> np.ndarray:
    ne_safe = np.maximum(ne, 1.0e-300)
    n_h = _t13_hydrogen_atomic_density(ne_safe, n_h_tot, k_dis, k_ion)
    n_hp = n_h * k_ion / ne_safe
    he0_frac, he1_frac, he2_frac = _t13_he_fractions(
        safe_log(ne_safe), log_k_he1, log_k_he2
    )
    n_he1 = n_he_tot * he1_frac
    n_he2 = n_he_tot * he2_frac
    return n_hp + n_he1 + 2.0 * n_he2 - ne_safe


def solve_t13_row(n_h_tot: float, n_he_tot: float, part: dict[str, np.ndarray]) -> np.ndarray:
    """Solve the Tomida Appendix 1 LTE composition for one density row."""
    ntemp = part["temp"].size
    ne_lo = np.full(ntemp, 1.0e-300, dtype=np.float64)
    ne_hi = np.full(ntemp, max(n_h_tot + 2.0 * n_he_tot, 1.0e-300), dtype=np.float64)

    f_lo = _t13_charge_residual(
        ne_lo, n_h_tot, n_he_tot, part["k_dis"], part["k_ion"], part["log_k_he1"], part["log_k_he2"]
    )
    f_hi = _t13_charge_residual(
        ne_hi, n_h_tot, n_he_tot, part["k_dis"], part["k_ion"], part["log_k_he1"], part["log_k_he2"]
    )

    neutral_mask = f_lo <= 0.0
    ionized_mask = (~neutral_mask) & (f_hi >= 0.0)
    bracket_mask = (~neutral_mask) & (~ionized_mask)
    if np.any(f_hi[bracket_mask] > 0.0) or np.any(f_lo[bracket_mask] < 0.0):
        raise RuntimeError("Failed to bracket the Tomida LTE electron-density root.")

    ne = np.where(ionized_mask, ne_hi, 0.0)
    ne = np.where(bracket_mask, 0.5 * (ne_lo + ne_hi), ne)
    for _ in range(128):
        if not np.any(bracket_mask):
            break
        ne_mid = 0.5 * (ne_lo + ne_hi)
        f_mid = _t13_charge_residual(
            ne_mid,
            n_h_tot,
            n_he_tot,
            part["k_dis"],
            part["k_ion"],
            part["log_k_he1"],
            part["log_k_he2"],
        )
        use_right = bracket_mask & (f_mid > 0.0)
        use_left = bracket_mask & (~use_right)
        ne_lo = np.where(use_right, ne_mid, ne_lo)
        ne_hi = np.where(use_left, ne_mid, ne_hi)
    ne = np.where(bracket_mask, 0.5 * (ne_lo + ne_hi), ne)
    ne_safe = np.maximum(ne, 1.0e-300)

    n_h = _t13_hydrogen_atomic_density(ne_safe, n_h_tot, part["k_dis"], part["k_ion"])
    n_hp = n_h * part["k_ion"] / ne_safe
    n_h2 = n_h * n_h / np.maximum(part["k_dis"], 1.0e-300)

    he0_frac, he1_frac, he2_frac = _t13_he_fractions(
        safe_log(ne_safe), part["log_k_he1"], part["log_k_he2"]
    )
    n_he = n_he_tot * he0_frac
    n_he1 = n_he_tot * he1_frac
    n_he2 = n_he_tot * he2_frac

    species = np.zeros((NSPEC, ntemp), dtype=np.float64)
    species[IH2] = n_h2
    species[IH] = n_h
    species[IHP] = n_hp
    species[IHE] = n_he
    species[IHE1] = n_he1
    species[IHE2] = n_he2
    species[IE] = ne_safe
    return species


def t13_fd_thermo_derivatives(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    pressure: np.ndarray,
    eps_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the T13 closure by finite-differencing the completed thermo surface.

    This follows Zach's released workflow more closely than the partition-derivative
    route: chi_rho, chi_T, and c_v come from finite differences of the final
    P(rho,T) and eps_physical(rho,T) tables, and Gamma_1 / Gamma_3 - 1 / c_s^2 are
    then constructed from those derivatives.
    """
    press_safe = np.maximum(pressure, 1.0e-300)
    temp = np.exp(logtemp)[None, :]
    rho = np.exp(logrho)[:, None]
    logpress = np.log(press_safe)
    chi_rho = finite_difference(logpress, logrho, axis=0)
    chi_t = finite_difference(logpress, logtemp, axis=1)
    d_eps_dlogt = finite_difference(eps_physical, logtemp, axis=1)
    cv = d_eps_dlogt / np.maximum(temp, 1.0e-300)
    gamma3m1 = pressure * chi_t / np.maximum(rho * temp * cv, 1.0e-300)
    gamma1 = chi_rho + chi_t * gamma3m1
    cs2 = gamma1 * pressure / np.maximum(rho, 1.0e-300)
    return chi_rho, chi_t, cv, gamma3m1, gamma1, cs2


def t13_lte_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
    zpe_subtract: bool,
    closure_check_max_rho_cgs: float | None = None,
) -> dict[str, np.ndarray]:
    """Build the Tomida/Andalman partition-function LTE table family.

    Pressure and internal energy come from the partition-function equilibrium solve.
    The stored closure fields then follow Zach's released approach: finite differences
    of the completed P(rho,T) and eps_physical(rho,T) surfaces.
    """
    rho_1d = np.exp(logrho)
    temp_1d = np.exp(logtemp)
    nrho = rho_1d.size
    ntemp = temp_1d.size
    part = build_t13_partition_terms(logtemp)
    eps_zpe_const = x_h * H2_VIB_ZPE / (2.0 * M_H)

    fields = {
        "logpress": np.empty((nrho, ntemp), dtype=np.float64),
        "logeps": np.empty((nrho, ntemp), dtype=np.float64),
        "logcs2": np.empty((nrho, ntemp), dtype=np.float64),
        "gamma1": np.empty((nrho, ntemp), dtype=np.float64),
        "gamma3m1": np.empty((nrho, ntemp), dtype=np.float64),
        "eps_runtime": np.empty((nrho, ntemp), dtype=np.float64),
        "xh2": np.empty((nrho, ntemp), dtype=np.float64),
        "xion": np.empty((nrho, ntemp), dtype=np.float64),
        "xhe1": np.empty((nrho, ntemp), dtype=np.float64),
        "xhe2": np.empty((nrho, ntemp), dtype=np.float64),
        "mu": np.empty((nrho, ntemp), dtype=np.float64),
        "beta_rad": np.empty((nrho, ntemp), dtype=np.float64),
        "eps_physical": np.empty((nrho, ntemp), dtype=np.float64),
        "u_trans_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_h2_rot_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_h2_vib_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_h2_zpe_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_diss_h2_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_ion_h_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_ion_he1_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_ion_he2_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "u_rad_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "pgas": np.empty((nrho, ntemp), dtype=np.float64),
        "prad": np.empty((nrho, ntemp), dtype=np.float64),
        "chi_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "chi_t": np.empty((nrho, ntemp), dtype=np.float64),
        "cv": np.empty((nrho, ntemp), dtype=np.float64),
        "cp": np.empty((nrho, ntemp), dtype=np.float64),
        "entropy": np.empty((nrho, ntemp), dtype=np.float64),
        "ppair": np.empty((nrho, ntemp), dtype=np.float64),
        "epair_kin_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "epair_rest_over_rho": np.empty((nrho, ntemp), dtype=np.float64),
        "logne_minus": np.empty((nrho, ntemp), dtype=np.float64),
        "logne_plus": np.empty((nrho, ntemp), dtype=np.float64),
        "eta_e": np.empty((nrho, ntemp), dtype=np.float64),
        "pair_pressure_fraction": np.empty((nrho, ntemp), dtype=np.float64),
    }

    for ir, rho_val in enumerate(rho_1d):
        n_h_tot = x_h * rho_val / M_H
        n_he_tot = y_he * rho_val / M_HE
        species = solve_t13_row(n_h_tot, n_he_tot, part)
        n_total = np.sum(species, axis=0)
        log_n = safe_log(np.maximum(species, 1.0e-300))
        mu_over_kt = log_n - part["logz"]
        s_i = (K_B * species / rho_val) * (1.0 + part["dlnz"] - mu_over_kt)

        n_charge = np.maximum(species[IE], 1.0e-300)
        pair_pressure = n_charge * K_B * temp_1d
        pair_u_kin = 1.5 * pair_pressure
        pair_u_rest = np.zeros_like(temp_1d)
        pair_u_total = pair_u_kin.copy()
        pair_entropy = s_i[IE].copy()
        pair_logne_minus = np.log(n_charge)
        pair_logne_plus = np.full_like(temp_1d, PAIR_LOG_NUMBER_FLOOR)
        pair_eta = np.zeros_like(temp_1d)
        pair_pressure_fraction = np.zeros_like(temp_1d)
        pair_diag_pressure = np.zeros_like(temp_1d)
        pair_diag_kin = np.zeros_like(temp_1d)
        pair_diag_rest = np.zeros_like(temp_1d)
        pair_active = temp_1d >= 10.0 ** PAIR_ACTIVE_LOG10_TEMP_MIN
        if np.any(pair_active):
            pair = _pair_lepton_state(temp_1d[pair_active], n_charge[pair_active], include_rest_mass=True)
            pair_pressure[pair_active] = pair["pressure"]
            pair_u_kin[pair_active] = pair["u_kin"]
            pair_u_rest[pair_active] = pair["u_rest"]
            pair_u_total[pair_active] = pair["u_total"]
            pair_entropy[pair_active] = pair["entropy"] / rho_val
            pair_logne_minus[pair_active] = np.log(np.maximum(pair["n_e_minus"], 1.0e-300))
            pair_logne_plus[pair_active] = np.log(np.maximum(pair["n_e_plus"], 1.0e-300))
            pair_eta[pair_active] = pair["eta_e"]
            pair_diag_pressure[pair_active] = pair["pressure_pairs"]
            pair_diag_kin[pair_active] = pair["u_kin_pairs"] / rho_val
            pair_diag_rest[pair_active] = pair["u_rest"] / rho_val

        p_gas = (n_total - n_charge) * K_B * temp_1d + pair_pressure
        p_rad = (A_RAD / 3.0) * temp_1d**4 if radiation_enabled else np.zeros_like(temp_1d)
        press = p_gas + p_rad
        pair_pressure_fraction[pair_active] = pair_diag_pressure[pair_active] / np.maximum(
            press[pair_active],
            1.0e-300,
        )

        s_gas = np.sum(s_i, axis=0) - s_i[IE] + pair_entropy
        if radiation_enabled:
            s_rad = 4.0 * A_RAD * temp_1d**3 / (3.0 * rho_val)
            entropy = s_gas + s_rad
            u_rad = A_RAD * temp_1d**4
        else:
            entropy = s_gas
            u_rad = np.zeros_like(temp_1d)

        u_trans = 1.5 * (n_total - n_charge) * K_B * temp_1d + pair_u_kin
        u_h2_rot = species[IH2] * K_B * temp_1d * part["dlnz_rot_h2"]
        u_h2_vib = species[IH2] * K_B * temp_1d * part["dlnz_vib_h2"]
        u_h2_zpe = species[IH2] * H2_VIB_ZPE
        u_diss_h2 = 0.5 * CHI_DIS_H2 * (species[IH] + species[IHP])
        u_ion_h = CHI_ION_H * species[IHP]
        u_ion_he1 = CHI_ION_HE1 * (species[IHE1] + species[IHE2])
        u_ion_he2 = CHI_ION_HE2 * species[IHE2]
        u_gas_partition = K_B * temp_1d * np.sum(species * part["dlnz"], axis=0)
        u_gas = u_gas_partition - 1.5 * n_charge * K_B * temp_1d + pair_u_total
        u_total = u_gas + u_rad
        u_total_components = (
            u_trans
            + pair_u_rest
            + u_h2_rot
            + u_h2_vib
            + u_diss_h2
            + u_ion_h
            + u_ion_he1
            + u_ion_he2
            + u_rad
        )
        energy_err = np.abs(u_total_components - u_total) / np.maximum(np.abs(u_total), 1.0)
        if np.max(energy_err) > 1.0e-10:
            raise RuntimeError(
                "Tomida LTE component energy assembly is inconsistent with the "
                "partition-function internal energy."
            )
        eps_physical = u_total / rho_val
        eps_runtime = eps_physical - (eps_zpe_const if zpe_subtract else 0.0)

        xh2 = np.where(n_h_tot > 0.0, 2.0 * species[IH2] / n_h_tot, 0.0)
        xion = np.where(n_h_tot > 0.0, species[IHP] / n_h_tot, 0.0)
        xhe1 = np.where(n_he_tot > 0.0, species[IHE1] / n_he_tot, 0.0)
        xhe2 = np.where(n_he_tot > 0.0, species[IHE2] / n_he_tot, 0.0)
        mu = rho_val / (n_total * M_H)
        beta_rad = np.where(press > 0.0, p_rad / press, 0.0)

        fields["logpress"][ir] = np.log(press)
        fields["logeps"][ir] = np.log(np.maximum(eps_runtime, 1.0e-300))
        fields["eps_runtime"][ir] = eps_runtime
        fields["xh2"][ir] = xh2
        fields["xion"][ir] = xion
        fields["xhe1"][ir] = xhe1
        fields["xhe2"][ir] = xhe2
        fields["mu"][ir] = mu
        fields["beta_rad"][ir] = beta_rad
        fields["eps_physical"][ir] = eps_physical
        fields["u_trans_over_rho"][ir] = u_trans / rho_val
        fields["u_h2_rot_over_rho"][ir] = u_h2_rot / rho_val
        fields["u_h2_vib_over_rho"][ir] = u_h2_vib / rho_val
        fields["u_h2_zpe_over_rho"][ir] = u_h2_zpe / rho_val
        fields["u_diss_h2_over_rho"][ir] = u_diss_h2 / rho_val
        fields["u_ion_h_over_rho"][ir] = u_ion_h / rho_val
        fields["u_ion_he1_over_rho"][ir] = u_ion_he1 / rho_val
        fields["u_ion_he2_over_rho"][ir] = u_ion_he2 / rho_val
        fields["u_rad_over_rho"][ir] = u_rad / rho_val
        fields["pgas"][ir] = p_gas
        fields["prad"][ir] = p_rad
        fields["entropy"][ir] = entropy
        fields["ppair"][ir] = pair_diag_pressure
        fields["epair_kin_over_rho"][ir] = pair_diag_kin
        fields["epair_rest_over_rho"][ir] = pair_diag_rest
        fields["logne_minus"][ir] = pair_logne_minus
        fields["logne_plus"][ir] = pair_logne_plus
        fields["eta_e"][ir] = pair_eta
        fields["pair_pressure_fraction"][ir] = pair_pressure_fraction

        if np.any(eps_runtime <= 0.0):
            raise RuntimeError(
                "Tomida LTE runtime inversion energy became non-positive. Increase the "
                "minimum temperature or disable ZPE subtraction for debugging."
            )

    pressure = np.exp(fields["logpress"])
    chi_rho, chi_t, cv, gamma3m1, gamma1, cs2 = t13_fd_thermo_derivatives(
        logrho,
        logtemp,
        pressure,
        fields["eps_physical"],
    )
    # In a union table the T13 block only serves rho <= rho_switch (plus the blend); above that
    # its cells are replaced by the dense-fluid / HELM blocks, so a non-physical partition-function
    # closure there (pressure ionisation, rho >~ 1e3 g/cc with hydrogen present) is irrelevant.
    # Callers building a union pass closure_check_max_rho_cgs; the standalone t13 model checks all.
    chk = np.ones_like(cv, dtype=bool)
    if closure_check_max_rho_cgs is not None:
        chk &= (np.exp(logrho) <= closure_check_max_rho_cgs)[:, None]
    if (
        np.any(cv[chk] <= 0.0)
        or np.any(cs2[chk] <= 0.0)
        or np.any(gamma1[chk] <= 0.0)
        or np.any(gamma3m1[chk] <= 0.0)
    ):
        raise RuntimeError(
            "Tomida LTE finite-difference closure became non-physical. The requested "
            "rho-T range is likely outside the valid partition-function regime."
        )
    chi_rho_cp = np.maximum(chi_rho, 1.0e-12)
    cp = cv * gamma1 / chi_rho_cp
    fields["logcs2"][:, :] = np.log(cs2)
    fields["gamma1"][:, :] = gamma1
    fields["gamma3m1"][:, :] = gamma3m1
    fields["chi_rho"][:, :] = chi_rho
    fields["chi_t"][:, :] = chi_t
    fields["cv"][:, :] = cv
    fields["cp"][:, :] = cp

    return fields


def fully_ionized_energy_offset(x_h: float, y_he: float) -> float:
    """Match the T13 energy zero in the fully ionized limit."""
    return (
        x_h * (0.5 * CHI_DIS_H2 + CHI_ION_H) / M_H
        + y_he * (CHI_ION_HE1 + CHI_ION_HE2) / M_HE
    )


def fully_ionized_validity_gate(
    rho: np.ndarray,
    temp: np.ndarray,
    x_h: float,
    y_he: float,
) -> np.ndarray:
    comp = solve_simple_lte_composition(rho, temp, x_h, y_he)
    return (comp["xion"] >= FI_XION_MIN) & (comp["xhe2"] >= FI_XHE2_MIN)


def fully_ionized_entropy(
    rho: np.ndarray,
    temp: np.ndarray,
    n_h: np.ndarray,
    n_he: np.ndarray,
    n_e: np.ndarray,
) -> np.ndarray:
    logz_hp = _log_ztr(temp, M_H)
    logz_he2 = _log_ztr(temp, M_HE)
    logz_e = _log_ztr(temp, M_E) + math.log(2.0)

    s_hp = (K_B * n_h / np.maximum(rho, 1.0e-300)) * (2.5 - (safe_log(np.maximum(n_h, 1.0e-300)) - logz_hp))
    s_he2 = (K_B * n_he / np.maximum(rho, 1.0e-300)) * (2.5 - (safe_log(np.maximum(n_he, 1.0e-300)) - logz_he2))
    s_e = (K_B * n_e / np.maximum(rho, 1.0e-300)) * (2.5 - (safe_log(np.maximum(n_e, 1.0e-300)) - logz_e))
    return s_hp + s_he2 + s_e


def fully_ionized_ion_entropy(
    rho: np.ndarray,
    temp: np.ndarray,
    n_h: np.ndarray,
    n_he: np.ndarray,
) -> np.ndarray:
    logz_hp = _log_ztr(temp, M_H)
    logz_he2 = _log_ztr(temp, M_HE)
    s_hp = (K_B * n_h / np.maximum(rho, 1.0e-300)) * (
        2.5 - (safe_log(np.maximum(n_h, 1.0e-300)) - logz_hp)
    )
    s_he2 = (K_B * n_he / np.maximum(rho, 1.0e-300)) * (
        2.5 - (safe_log(np.maximum(n_he, 1.0e-300)) - logz_he2)
    )
    return s_hp + s_he2


_PAIR_LAGUERRE_X, _PAIR_LAGUERRE_W = roots_genlaguerre(PAIR_QUAD_N, 0.5)
_PAIR_LAGUERRE_X = np.asarray(_PAIR_LAGUERRE_X, dtype=np.float64)
_PAIR_LAGUERRE_W = np.asarray(_PAIR_LAGUERRE_W, dtype=np.float64)
_PAIR_N_PREFAC = 1.0 / (math.pi**2 * ELECTRON_COMPTON_LENGTH**3)
_PAIR_P_PREFAC = M_E_C2 / (3.0 * math.pi**2 * ELECTRON_COMPTON_LENGTH**3)
_PAIR_U_PREFAC = M_E_C2 / (math.pi**2 * ELECTRON_COMPTON_LENGTH**3)


def _laguerre_fermi_weight(log_a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return exp(y)/(exp(y + log_a) + 1) without overflow."""
    a = np.asarray(log_a, dtype=np.float64)
    b = -y
    m = np.minimum(a, b)
    with np.errstate(over="ignore"):
        denom = np.exp(a - m) + np.exp(b - m)
    return np.exp(-m) / denom


def _pair_eta_bracket(theta: np.ndarray, n_charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_safe = np.maximum(theta, 1.0e-300)
    y = _PAIR_LAGUERRE_X[None, :]
    w = _PAIR_LAGUERRE_W[None, :]
    base = np.sqrt(2.0 + theta_safe[:, None] * y) * (1.0 + theta_safe[:, None] * y)
    pref = _PAIR_N_PREFAC * theta_safe**1.5

    def net_density(eta: np.ndarray) -> np.ndarray:
        h_minus = _laguerre_fermi_weight(-eta[:, None], y)
        h_plus = _laguerre_fermi_weight(eta[:, None] + 2.0 / theta_safe[:, None], y)
        return pref * np.sum(w * base * (h_minus - h_plus), axis=1)

    eta_lo = np.full_like(theta_safe, -40.0)
    eta_hi = np.full_like(theta_safe, 40.0)
    for _ in range(12):
        f_lo = net_density(eta_lo) - n_charge
        f_hi = net_density(eta_hi) - n_charge
        need_lo = f_lo > 0.0
        need_hi = f_hi < 0.0
        if not (np.any(need_lo) or np.any(need_hi)):
            break
        eta_lo = np.where(need_lo, eta_lo - 40.0, eta_lo)
        eta_hi = np.where(need_hi, eta_hi + 40.0, eta_hi)
    return eta_lo, eta_hi


def _pair_lepton_state_chunk(
    temp_1d: np.ndarray,
    n_charge: np.ndarray,
    include_rest_mass: bool,
) -> dict[str, np.ndarray]:
    """Ideal relativistic e-/e+ thermodynamics in the HELM fully ionized spirit.

    The solved degeneracy parameter eta is the electron chemical potential excluding
    rest mass divided by kT. Positrons use eta_plus = -eta - 2 m_e c^2 / kT. The
    kinetic pressure and energy integrals follow the same fully ionized, arbitrary
    relativity/degeneracy e-/e+ gas used by HELM/MESA. When requested, the internal
    energy includes only the created-pair rest mass, 2 m_e c^2 n_+, not the rest
    mass of the charge-balancing electrons already tied to the baryons.
    """
    temp_1d = np.asarray(temp_1d, dtype=np.float64)
    n_charge = np.maximum(np.asarray(n_charge, dtype=np.float64), 1.0e-300)
    theta = np.maximum(K_B * temp_1d / M_E_C2, 1.0e-300)
    y = _PAIR_LAGUERRE_X[None, :]
    w = _PAIR_LAGUERRE_W[None, :]
    theta_col = theta[:, None]
    gamma = 1.0 + theta_col * y
    root = np.sqrt(2.0 + theta_col * y)

    eta_lo, eta_hi = _pair_eta_bracket(theta, n_charge)
    for _ in range(PAIR_BISECTION_ITERS):
        eta_mid = 0.5 * (eta_lo + eta_hi)
        h_minus = _laguerre_fermi_weight(-eta_mid[:, None], y)
        h_plus = _laguerre_fermi_weight(eta_mid[:, None] + 2.0 / theta_col, y)
        n_net = (
            _PAIR_N_PREFAC
            * theta**1.5
            * np.sum(w * root * gamma * (h_minus - h_plus), axis=1)
        )
        move_hi = n_net > n_charge
        eta_hi = np.where(move_hi, eta_mid, eta_hi)
        eta_lo = np.where(move_hi, eta_lo, eta_mid)

    eta = 0.5 * (eta_lo + eta_hi)
    h_minus = _laguerre_fermi_weight(-eta[:, None], y)
    h_plus = _laguerre_fermi_weight(eta[:, None] + 2.0 / theta_col, y)
    n_minus = _PAIR_N_PREFAC * theta**1.5 * np.sum(w * root * gamma * h_minus, axis=1)
    n_plus = _PAIR_N_PREFAC * theta**1.5 * np.sum(w * root * gamma * h_plus, axis=1)
    p_minus = (
        _PAIR_P_PREFAC
        * theta**2.5
        * np.sum(w * y * root**3 * h_minus, axis=1)
    )
    p_plus = (
        _PAIR_P_PREFAC
        * theta**2.5
        * np.sum(w * y * root**3 * h_plus, axis=1)
    )
    p_lept = p_minus + p_plus
    u_kin_minus = (
        _PAIR_U_PREFAC
        * theta**2.5
        * np.sum(w * y * root * gamma * h_minus, axis=1)
    )
    u_kin_plus = (
        _PAIR_U_PREFAC
        * theta**2.5
        * np.sum(w * y * root * gamma * h_plus, axis=1)
    )
    u_kin = u_kin_minus + u_kin_plus
    u_rest = 2.0 * M_E_C2 * n_plus
    u_total = u_kin + (u_rest if include_rest_mass else 0.0)
    s_lept = np.maximum((u_total + p_lept - eta * K_B * temp_1d * n_charge) / temp_1d, 0.0)
    return {
        "eta_e": eta,
        "n_e_minus": n_minus,
        "n_e_plus": n_plus,
        "n_pair": n_plus,
        "pressure": p_lept,
        "pressure_minus": p_minus,
        "pressure_plus": p_plus,
        "pressure_pairs": 2.0 * p_plus,
        "u_kin": u_kin,
        "u_kin_minus": u_kin_minus,
        "u_kin_plus": u_kin_plus,
        "u_kin_pairs": 2.0 * u_kin_plus,
        "u_rest": u_rest,
        "u_total": u_total,
        "entropy": s_lept,
    }


def _pair_lepton_state(
    temp_1d: np.ndarray,
    n_charge: np.ndarray,
    include_rest_mass: bool,
    chunk_size: int = PAIR_STATE_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Evaluate the ideal relativistic pair gas with bounded temporary storage.

    A full 640 x 640 table contains 409600 states.  Broadcasting all of those
    states against the 96-point quadrature grid at once creates several
    multi-hundred-megabyte temporaries.  Chunking is algebraically identical,
    while keeping the generator usable on ordinary workstation nodes.
    """
    temp_flat = np.asarray(temp_1d, dtype=np.float64).reshape(-1)
    charge_flat = np.asarray(n_charge, dtype=np.float64).reshape(-1)
    if temp_flat.shape != charge_flat.shape:
        raise ValueError("Pair temperature and charge-density arrays must have matching shapes.")
    if chunk_size <= 0:
        raise ValueError("Pair state chunk size must be positive.")
    if temp_flat.size == 0:
        return _pair_lepton_state_chunk(temp_flat, charge_flat, include_rest_mass)

    output: dict[str, np.ndarray] | None = None
    for start in range(0, temp_flat.size, chunk_size):
        stop = min(start + chunk_size, temp_flat.size)
        chunk = _pair_lepton_state_chunk(
            temp_flat[start:stop],
            charge_flat[start:stop],
            include_rest_mass,
        )
        if output is None:
            output = {
                name: np.empty(temp_flat.size, dtype=np.float64)
                for name in chunk
            }
        for name, values in chunk.items():
            output[name][start:stop] = values
    assert output is not None
    return output


def overlay_helm_pair_diagnostics(
    fields: dict[str, np.ndarray],
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    pressure_name: str,
    valid_mask: np.ndarray | None = None,
) -> int:
    """Populate HELM-style e-/e+ diagnostics without modifying parent thermodynamics."""
    pressure_total = fields[pressure_name]
    shape = pressure_total.shape
    rho = np.broadcast_to(np.exp(logrho)[:, None], shape)
    temp = np.broadcast_to(np.exp(logtemp)[None, :], shape)
    n_charge = np.broadcast_to(
        x_h * np.exp(logrho)[:, None] / M_H
        + 2.0 * y_he * np.exp(logrho)[:, None] / M_HE,
        shape,
    )
    active = np.isfinite(pressure_total) & (pressure_total > 0.0)
    if valid_mask is not None:
        active &= valid_mask
    if not np.any(active):
        return 0

    pair = _pair_lepton_state(temp[active], n_charge[active], include_rest_mass=True)
    fields["ppair"][active] = pair["pressure_pairs"]
    fields["epair_kin_over_rho"][active] = pair["u_kin_pairs"] / np.maximum(rho[active], 1.0e-300)
    fields["epair_rest_over_rho"][active] = pair["u_rest"] / np.maximum(rho[active], 1.0e-300)
    fields["logne_minus"][active] = np.log(np.maximum(pair["n_e_minus"], 1.0e-300))
    fields["logne_plus"][active] = np.log(np.maximum(pair["n_e_plus"], 1.0e-300))
    fields["eta_e"][active] = pair["eta_e"]
    fields["pair_pressure_fraction"][active] = pair["pressure_pairs"] / np.maximum(
        pressure_total[active],
        1.0e-300,
    )
    return int(np.count_nonzero(active))


def finalize_dense_branch(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    pressure_total: np.ndarray,
    eps_physical: np.ndarray,
    pgas: np.ndarray,
    prad: np.ndarray,
    entropy_total: np.ndarray,
    xh2: np.ndarray,
    xion: np.ndarray,
    xhe1: np.ndarray,
    xhe2: np.ndarray,
    mu: np.ndarray,
    logne: np.ndarray,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    shape = pressure_total.shape
    pgas = np.broadcast_to(pgas, shape)
    prad = np.broadcast_to(prad, shape)
    xh2 = np.broadcast_to(xh2, shape)
    xion = np.broadcast_to(xion, shape)
    xhe1 = np.broadcast_to(xhe1, shape)
    xhe2 = np.broadcast_to(xhe2, shape)
    mu = np.broadcast_to(mu, shape)
    logne = np.broadcast_to(logne, shape)
    pressure_safe = np.maximum(pressure_total, 1.0e-300)
    eps_safe = np.maximum(eps_physical, 1.0e-300)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        chi_rho, chi_t, cv, gamma3m1, gamma1, cs2 = t13_fd_thermo_derivatives(
            logrho,
            logtemp,
            pressure_safe,
            eps_safe,
        )
        cp = cv * gamma1 / np.maximum(chi_rho, 1.0e-12)
        grad_ad = gamma3m1 / np.maximum(gamma1, 1.0e-300)
        beta_rad = np.where(pressure_total > 0.0, prad / pressure_total, 0.0)
    out = {
        "logpress": np.log(pressure_safe),
        "logeps": np.log(eps_safe),
        "logcs2": np.log(np.maximum(cs2, 1.0e-300)),
        "gamma1": gamma1,
        "gamma3m1": gamma3m1,
        "xh2": xh2,
        "xion": xion,
        "xhe1": xhe1,
        "xhe2": xhe2,
        "mu": mu,
        "beta_rad": beta_rad,
        "eps_physical": eps_physical,
        "eps_runtime": eps_physical,
        "pgas": pgas,
        "prad": prad,
        "chi_rho": chi_rho,
        "chi_t": chi_t,
        "cv": cv,
        "cp": cp,
        "entropy": entropy_total,
        "grad_ad": grad_ad,
        "logne": logne,
    }
    if extra_fields:
        out.update(extra_fields)
    return out


def helm_style_hhe_fi_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Full local HELM branch: ions, e-/e+ pairs, photons, and Coulomb free energy.

    The electron/positron component is the arbitrary-relativity,
    arbitrary-degeneracy ideal Fermi gas used by the HELM family.  The local
    implementation augments it with the repo's Potekhin-Chabrier liquid
    ion-ion free-energy correction and advertises cells only inside that fit's
    classical-liquid validity cuts.
    """
    fields, context = cp_like_hhe_fi_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
    )
    context = dict(context)
    context["corner_model"] = "helmholtz_fully_ionized_eos"
    context["coulomb_model"] = "potekhin_chabrier_liquid_ion_ion"
    context["coulomb_derivative"] = "analytic_dfree_energy_dln_gamma"
    return fields, context


def pc_liquid_reduced_free_energy(gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Potekhin-Chabrier liquid fit: return f_ii and u_ii = d f / d ln Gamma."""
    gamma_safe = np.maximum(gamma, 1.0e-12)
    a1 = -0.907347
    a2 = 0.62849
    a3 = -math.sqrt(3.0) / 2.0 - a1 / math.sqrt(a2)
    b1 = 0.00456
    b2 = 211.6
    b3 = -1.0e-4
    b4 = 0.00462

    sq = np.sqrt(gamma_safe)
    sq_a = np.sqrt(gamma_safe / a2)
    f_ii = (
        a1 * (np.sqrt(gamma_safe * (a2 + gamma_safe)) - a2 * np.log(sq_a + np.sqrt(1.0 + gamma_safe / a2)))
        + 2.0 * a3 * (sq - np.arctan(sq))
        + b1 * (gamma_safe - b2 * np.log1p(gamma_safe / b2))
        + 0.5 * b3 * np.log1p(gamma_safe * gamma_safe / b4)
    )
    # Analytic d f_ii / d ln(Gamma).  This avoids cancellation in the old
    # centered finite difference at weak coupling and is the thermodynamic
    # quantity used by the Coulomb pressure, energy, and entropy corrections.
    gamma_3_2 = gamma_safe * sq
    u_ii = (
        a1 * gamma_3_2 / np.sqrt(a2 + gamma_safe)
        + a3 * gamma_3_2 / (1.0 + gamma_safe)
        + b1 * gamma_safe**2 / (b2 + gamma_safe)
        + b3 * gamma_safe**2 / (b4 + gamma_safe**2)
    )
    return f_ii, u_ii


def cp_like_hhe_fi_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Chabrier-Potekhin backstop: fully ionized H/He + Coulomb + e-/e+ pairs."""
    rho = np.exp(logrho)[:, None]
    temp_1d = np.exp(logtemp)
    temp = temp_1d[None, :]
    valid_gate = fully_ionized_validity_gate(rho, temp, x_h, y_he)

    n_h = x_h * rho / M_H
    n_he = y_he * rho / M_HE
    n_charge = n_h + 2.0 * n_he
    n_ions = n_h + n_he
    n_total_mu = n_ions + n_charge
    ae = (3.0 / (4.0 * math.pi * np.maximum(n_charge, 1.0e-300))) ** (1.0 / 3.0)

    gamma_h = Q_E_SQ_CGS / (ae * K_B * temp)
    gamma_he = (2.0 ** (5.0 / 3.0)) * Q_E_SQ_CGS / (ae * K_B * temp)
    f_h, u_h = pc_liquid_reduced_free_energy(gamma_h)
    f_he, u_he = pc_liquid_reduced_free_energy(gamma_he)

    wp_h = np.sqrt(4.0 * math.pi * n_h * Q_E_SQ_CGS / M_H)
    wp_he = np.sqrt(4.0 * math.pi * n_he * 4.0 * Q_E_SQ_CGS / M_HE)
    tp_h = HBAR * wp_h / K_B
    tp_he = HBAR * wp_he / K_B

    shape = (rho.shape[0], temp_1d.size)
    temp_flat = np.broadcast_to(temp_1d, shape).reshape(-1)
    n_charge_flat = np.broadcast_to(n_charge, shape).reshape(-1)
    pair = _pair_lepton_state(temp_flat, n_charge_flat, include_rest_mass=True)
    pair_fields = {name: values.reshape(shape) for name, values in pair.items()}

    pion = n_ions * K_B * temp
    eps_ideal = (
        1.5 * pion
        + pair_fields["u_total"]
    ) / np.maximum(rho, 1.0e-300) + fully_ionized_energy_offset(x_h, y_he)
    entropy_ideal = fully_ionized_ion_entropy(rho, temp, n_h, n_he) + pair_fields["entropy"] / np.maximum(rho, 1.0e-300)

    u_c_vol = K_B * temp * (n_h * u_h + n_he * u_he)
    p_c = (K_B * temp / 3.0) * (n_h * u_h + n_he * u_he)
    s_c = (K_B / np.maximum(rho, 1.0e-300)) * (n_h * (u_h - f_h) + n_he * (u_he - f_he))

    pgas = pion + pair_fields["pressure"] + p_c
    eps_gas = eps_ideal + u_c_vol / np.maximum(rho, 1.0e-300)
    entropy_gas = entropy_ideal + s_c
    prad = (A_RAD / 3.0) * temp**4 if radiation_enabled else np.zeros_like(pgas)
    eps_total = eps_gas + (A_RAD * temp**4 / rho if radiation_enabled else 0.0)
    entropy_total = entropy_gas + (4.0 * A_RAD * temp**3 / (3.0 * rho) if radiation_enabled else 0.0)
    press_total = pgas + prad
    mu = rho / np.maximum(n_total_mu * M_H, 1.0e-300)
    pair_pressure_fraction = np.where(press_total > 0.0, pair_fields["pressure_pairs"] / press_total, 0.0)

    fields = finalize_dense_branch(
        logrho,
        logtemp,
        press_total,
        eps_total,
        pgas,
        prad,
        entropy_total,
        np.zeros_like(press_total),
        np.ones_like(press_total),
        np.zeros_like(press_total),
        np.ones_like(press_total),
        mu,
        np.log(np.maximum(pair_fields["n_e_minus"], 1.0e-300)),
        {
            "ppair": pair_fields["pressure_pairs"],
            "epair_kin_over_rho": pair_fields["u_kin_pairs"] / np.maximum(rho, 1.0e-300),
            "epair_rest_over_rho": pair_fields["u_rest"] / np.maximum(rho, 1.0e-300),
            "logne_minus": np.log(np.maximum(pair_fields["n_e_minus"], 1.0e-300)),
            "logne_plus": np.log(np.maximum(pair_fields["n_e_plus"], 1.0e-300)),
            "eta_e": pair_fields["eta_e"],
            "pair_pressure_fraction": pair_pressure_fraction,
        },
    )
    cp_valid = (
        np.isfinite(press_total)
        & np.isfinite(eps_total)
        & (press_total > 0.0)
        & (eps_total > 0.0)
        & (gamma_he <= CP_LIQUID_GAMMA_MAX)
        & (temp >= CP_CLASSICAL_T_OVER_TP_MIN * np.maximum(tp_h, tp_he))
    )
    fields["coulomb_valid"] = cp_valid.astype(np.float64)
    fields["fi_valid"] = (valid_gate & cp_valid).astype(np.float64)
    fields["gamma_coulomb_h"] = gamma_h
    fields["gamma_coulomb_he"] = gamma_he
    fields["f_coulomb_over_kT"] = (n_h * f_h + n_he * f_he) / np.maximum(n_h + n_he, 1.0e-300)
    context = {
        "corner_model": "cp_like_fully_ionized_liquid",
        "ionization_gate_xion_min": FI_XION_MIN,
        "ionization_gate_xhe2_min": FI_XHE2_MIN,
        "pc_liquid_gamma_max": CP_LIQUID_GAMMA_MAX,
        "pc_t_over_tp_min": CP_CLASSICAL_T_OVER_TP_MIN,
        "coulomb_enabled": True,
        "coulomb_model": "potekhin_chabrier_liquid_ion_ion",
        "coulomb_derivative": "analytic_dfree_energy_dln_gamma",
        "pair_production_enabled": True,
        "pair_model": "ideal_relativistic_fermi_dirac_electron_positron",
        "pair_rest_mass_energy_in_eps": True,
        "pair_rest_mass_energy_convention": "created_pairs_only",
    }
    return fields, context


def fill_chabrier_dense_row_gaps(
    logtemp: np.ndarray,
    rho_cgs: np.ndarray,
    linear_fields: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    rho_fill_min_cgs: float = CHABRIER2021_DENSE_GAP_FILL_RHO_CGS_MIN,
    log10_temp_min: float = CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MIN,
    log10_temp_max: float = CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MAX,
    max_gap_cells: int = CHABRIER2021_DENSE_GAP_FILL_MAX_GAP_CELLS,
    edge_fill_cells: int = CHABRIER2021_DENSE_GAP_FILL_EDGE_CELLS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Smooth bounded dense Chabrier holes row-by-row in temperature.

    The imported Chabrier validity mask contains small dense-fluid islands where the
    thermo remains smooth but the native validity cuts drop isolated cells or short
    segments. Fill those bounded holes along each rho row using monotone PCHIP in log T.
    Only tiny low-temperature edge gaps are filled one-sided by copying the nearest
    valid Chabrier value.
    """
    filled_fields = {name: np.array(values, copy=True) for name, values in linear_fields.items()}
    filled_mask = np.array(valid_mask, copy=True)
    log10_temp = logtemp / LN10
    temp_region = (
        (log10_temp >= log10_temp_min - 1.0e-12)
        & (log10_temp <= log10_temp_max + 1.0e-12)
    )
    interp_names = (
        "pressure_total",
        "eps_total",
        "cs2",
        "gamma1",
        "gamma3m1",
        "beta_rad",
        "eps_physical",
        "eps_runtime",
        "pgas",
        "prad",
        "chi_rho",
        "chi_t",
        "cv",
        "cp",
        "entropy",
        "grad_ad",
    )

    n_interp = 0
    n_edge = 0
    for ir in range(filled_mask.shape[0]):
        if rho_cgs[ir, 0] < rho_fill_min_cgs:
            continue
        row_valid = filled_mask[ir]
        row_region = temp_region
        row_gap = row_region & (~row_valid)
        if not np.any(row_gap):
            continue

        valid_idx = np.where(row_valid & row_region)[0]
        if valid_idx.size == 0:
            continue
        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1])
        row_fill_interp = np.zeros_like(row_gap, dtype=bool)
        row_fill_edge_low = np.zeros_like(row_gap, dtype=bool)
        row_fill_edge_high = np.zeros_like(row_gap, dtype=bool)

        gap_idx = np.where(row_gap)[0]
        start = prev = int(gap_idx[0])
        segments: list[tuple[int, int]] = []
        for idx in gap_idx[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
            else:
                segments.append((start, prev))
                start = prev = idx
        segments.append((start, prev))

        for g0, g1 in segments:
            glen = g1 - g0 + 1
            if g0 > first_valid and g1 < last_valid and glen <= max_gap_cells:
                row_fill_interp[g0 : g1 + 1] = True
                n_interp += glen
            elif g1 < first_valid and (first_valid - g0) <= edge_fill_cells:
                row_fill_edge_low[g0 : g1 + 1] = True
                n_edge += glen
            elif g0 > last_valid and (g1 - last_valid) <= edge_fill_cells:
                row_fill_edge_high[g0 : g1 + 1] = True
                n_edge += glen

        if not (np.any(row_fill_interp) or np.any(row_fill_edge_low) or np.any(row_fill_edge_high)):
            continue

        for name in interp_names:
            values = filled_fields[name][ir]
            valid_for_field = row_valid & np.isfinite(values)
            valid_for_interp = np.where(valid_for_field & row_region)[0]
            if valid_for_interp.size >= 2 and np.any(row_fill_interp):
                interp = PchipInterpolator(
                    logtemp[valid_for_interp],
                    values[valid_for_interp],
                    extrapolate=False,
                )
                filled_fields[name][ir, row_fill_interp] = interp(logtemp[row_fill_interp])
            if valid_for_interp.size >= 1 and np.any(row_fill_edge_low):
                filled_fields[name][ir, row_fill_edge_low] = values[valid_for_interp[0]]
            if valid_for_interp.size >= 1 and np.any(row_fill_edge_high):
                filled_fields[name][ir, row_fill_edge_high] = values[valid_for_interp[-1]]

        row_fill = row_fill_interp | row_fill_edge_low | row_fill_edge_high
        for name in ("xh2", "xion", "xhe1", "xhe2", "mu"):
            filled_fields[name][ir, row_fill] = filled_fields[name][ir, row_fill]
        filled_mask[ir, row_fill] = True

    print(
        "Chabrier dense-gap fill: "
        f"interpolated {n_interp} cells, "
        f"edge-filled {n_edge} cells"
    )
    return filled_fields, filled_mask


def build_chabrier_dense_onset_blend(
    logtemp: np.ndarray,
    rho_cgs: np.ndarray,
    t13_valid: np.ndarray,
    chabrier_valid: np.ndarray,
    rho_switch_cgs: float,
    log10_temp_min: float = CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MIN,
    log10_temp_max: float = CHABRIER2021_DENSE_GAP_FILL_LOG10_TEMP_MAX,
    blend_cells: int = CHABRIER2021_DENSE_ONSET_BLEND_CELLS,
) -> np.ndarray:
    """Blend the dense T13->Chabrier onset over a short temperature window.

    The imported Chabrier validity boundary can begin abruptly along a rho row.
    In the dense branch this creates a visible Gamma_1 seam. Smooth only the
    first few valid Chabrier cells after the low-temperature onset, leaving the
    low-rho density blend unchanged.
    """
    blend_frac = np.zeros_like(chabrier_valid, dtype=np.float64)
    log10_temp = logtemp / LN10
    temp_region = (
        (log10_temp >= log10_temp_min - 1.0e-12)
        & (log10_temp <= log10_temp_max + 1.0e-12)
    )
    n_rows = 0
    n_cells = 0
    for ir in range(chabrier_valid.shape[0]):
        if rho_cgs[ir, 0] < rho_switch_cgs:
            continue
        row_valid = chabrier_valid[ir] & temp_region
        if not np.any(row_valid):
            continue
        valid_idx = np.where(row_valid)[0]
        first_valid = int(valid_idx[0])
        if first_valid <= 0:
            continue
        if not np.any(t13_valid[ir, :first_valid]):
            continue
        run_end = first_valid
        while run_end + 1 < row_valid.size and row_valid[run_end + 1]:
            run_end += 1
        span = min(blend_cells, run_end - first_valid + 1)
        if span <= 0:
            continue
        eta = np.linspace(0.0, 1.0, span + 2, dtype=np.float64)[1:-1]
        weights, _, _ = smootherstep(eta)
        blend_frac[ir, first_valid : first_valid + span] = weights
        n_rows += 1
        n_cells += span
    print(
        "Chabrier dense-onset blend: "
        f"smoothed {n_cells} cells across {n_rows} rows"
    )
    return blend_frac


def rebuild_union_thermo_closure(
    merged: dict[str, np.ndarray],
    logrho: np.ndarray,
    logtemp: np.ndarray,
) -> None:
    """Reclose runtime energy and sound speed after union selection/repair.

    The imported branches do not share a common differentiable Helmholtz free
    energy across their seams.  Preserve their native/blended Gamma response
    fields, but make the two identities used directly by AthenaK exact:
    eps_runtime == eps_physical == exp(logeps) and
    cs2 == Gamma1 P/rho.  Recomputing Gamma from finite differences of the
    stitched surface creates large, non-physical seam spikes.
    """
    rho = np.exp(logrho)[:, None]
    del logtemp
    pressure = np.exp(merged["logpress"])
    eps_runtime = np.exp(merged["logeps"])
    gamma1 = merged["gamma1"]
    gamma3m1 = merged["gamma3m1"]
    cs2 = gamma1 * pressure / np.maximum(rho, 1.0e-300)
    grad_ad = gamma3m1 / np.maximum(gamma1, 1.0e-300)

    required = {"gamma3m1": gamma3m1, "gamma1": gamma1, "cs2": cs2, "grad_ad": grad_ad}
    for name, values in required.items():
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise RuntimeError(f"Union closure reconstruction produced invalid {name} values.")

    merged["eps_runtime"] = eps_runtime
    merged["eps_physical"] = eps_runtime.copy()
    merged["logcs2"] = np.log(cs2)
    merged["grad_ad"] = grad_ad
    merged["beta_rad"] = np.clip(
        merged["prad"] / np.maximum(pressure, 1.0e-300),
        0.0,
        1.0,
    )


def chabrier2021_t13_helm_union_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
    chabrier_data_dir: str | None = None,
    rho_blend_cgs_min: float = CHABRIER2021_T13_BLEND_RHO_CGS_MIN,
    rho_switch_cgs: float = CHABRIER2021_T13_SWITCH_RHO_CGS,
    helm_hot_fallback_log10_temp_min: float = HELM_HOT_FALLBACK_LOG10_TEMP_MIN,
    helm_preferred_log10_temp_min: float = CHABRIER2021_HELM_PREFERRED_LOG10_TEMP_MIN,
    chabrier_mixing: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not (0.0 < rho_blend_cgs_min < rho_switch_cgs):
        raise RuntimeError(
            "Chabrier/T13 union requires 0 < rho_blend_cgs_min < rho_switch_cgs."
        )
    t13_fields = t13_lte_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
        zpe_subtract=False,
        closure_check_max_rho_cgs=100.0*rho_switch_cgs,
    )
    ensure_pair_diagnostics(t13_fields, logne_name=None)
    t13_union_fields = dict(t13_fields)
    t13_union_fields["grad_ad"] = t13_fields["gamma3m1"] / np.maximum(t13_fields["gamma1"], 1.0e-300)
    simplified_fields = simplified_lte_thermo(
        np.exp(logrho)[:, None],
        np.exp(logtemp)[None, :],
        x_h,
        y_he,
        radiation_enabled,
    )
    chabrier_fields, chabrier_context = chabrier2021_thermo(
        logrho,
        logtemp,
        y_he,
        radiation_enabled,
        data_dir=chabrier_data_dir,
        mixing=chabrier_mixing,
    )
    ensure_pair_diagnostics(chabrier_fields, template_name="pressure_total", logne_name=None)
    helm_fields, helm_context = helm_style_hhe_fi_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
    )

    rho_cgs = np.exp(logrho)[:, None]
    log10_rho_cgs = logrho[:, None] / LN10
    log10_rho_blend_min = math.log10(rho_blend_cgs_min)
    log10_rho_switch = math.log10(rho_switch_cgs)
    rho_blend_eta = (log10_rho_cgs - log10_rho_blend_min) / max(
        log10_rho_switch - log10_rho_blend_min,
        1.0e-12,
    )
    rho_blend_pref, _, _ = smootherstep(rho_blend_eta)

    t13_valid = (
        np.isfinite(t13_fields["logpress"])
        & np.isfinite(t13_fields["logeps"])
        & np.isfinite(t13_fields["logcs2"])
        & np.isfinite(t13_fields["gamma1"])
        & np.isfinite(t13_fields["gamma3m1"])
    )
    chabrier_valid_raw = chabrier_fields["chabrier_valid"] > 0.5
    t13_linear = {
        "pressure_total": np.exp(t13_fields["logpress"]),
        "eps_total": np.exp(t13_fields["logeps"]),
        "cs2": np.exp(t13_fields["logcs2"]),
        "gamma1": t13_fields["gamma1"],
        "gamma3m1": t13_fields["gamma3m1"],
        "xh2": t13_fields["xh2"],
        "xion": t13_fields["xion"],
        "xhe1": t13_fields["xhe1"],
        "xhe2": t13_fields["xhe2"],
        "mu": t13_fields["mu"],
        "beta_rad": t13_fields["beta_rad"],
        "eps_physical": t13_fields["eps_physical"],
        "eps_runtime": t13_fields["eps_runtime"],
        "pgas": t13_fields["pgas"],
        "prad": t13_fields["prad"],
        "chi_rho": t13_fields["chi_rho"],
        "chi_t": t13_fields["chi_t"],
        "cv": t13_fields["cv"],
        "cp": t13_fields["cp"],
        "entropy": t13_fields["entropy"],
        "grad_ad": t13_union_fields["grad_ad"],
        "ppair": t13_fields["ppair"],
        "epair_kin_over_rho": t13_fields["epair_kin_over_rho"],
        "epair_rest_over_rho": t13_fields["epair_rest_over_rho"],
        "logne_minus": t13_fields["logne_minus"],
        "logne_plus": t13_fields["logne_plus"],
        "eta_e": t13_fields["eta_e"],
        "pair_pressure_fraction": t13_fields["pair_pressure_fraction"],
    }
    chabrier_linear = {
        "pressure_total": chabrier_fields["pressure_total"],
        "eps_total": chabrier_fields["eps_total"],
        "cs2": chabrier_fields["cs2"],
        "gamma1": chabrier_fields["gamma1"],
        "gamma3m1": chabrier_fields["gamma3m1"],
        "xh2": simplified_fields["xh2"],
        "xion": simplified_fields["xion"],
        "xhe1": simplified_fields["xhe1"],
        "xhe2": simplified_fields["xhe2"],
        "mu": simplified_fields["mu"],
        "beta_rad": chabrier_fields["beta_rad"],
        "eps_physical": chabrier_fields["eps_total"],
        "eps_runtime": chabrier_fields["eps_total"],
        "pgas": chabrier_fields["pgas"],
        "prad": chabrier_fields["prad"],
        "chi_rho": chabrier_fields["chi_rho"],
        "chi_t": chabrier_fields["chi_t"],
        "cv": chabrier_fields["cv"],
        "cp": chabrier_fields["cp"],
        "entropy": chabrier_fields["entropy_total"],
        "grad_ad": chabrier_fields["grad_ad"],
        "ppair": chabrier_fields["ppair"],
        "epair_kin_over_rho": chabrier_fields["epair_kin_over_rho"],
        "epair_rest_over_rho": chabrier_fields["epair_rest_over_rho"],
        "logne_minus": chabrier_fields["logne_minus"],
        "logne_plus": chabrier_fields["logne_plus"],
        "eta_e": chabrier_fields["eta_e"],
        "pair_pressure_fraction": chabrier_fields["pair_pressure_fraction"],
    }
    chabrier_linear, chabrier_valid = fill_chabrier_dense_row_gaps(
        logtemp,
        rho_cgs,
        chabrier_linear,
        chabrier_valid_raw,
    )
    n_pair_overlay = overlay_helm_pair_diagnostics(
        chabrier_linear,
        logrho,
        logtemp,
        x_h,
        y_he,
        pressure_name="pressure_total",
        valid_mask=chabrier_valid,
    )
    dense_onset_blend_frac = build_chabrier_dense_onset_blend(
        logtemp,
        rho_cgs,
        t13_valid,
        chabrier_valid,
        rho_switch_cgs,
    )
    helm_finite = (
        np.isfinite(helm_fields["logpress"])
        & np.isfinite(helm_fields["logeps"])
        & np.isfinite(helm_fields["logcs2"])
        & np.isfinite(helm_fields["gamma1"])
        & np.isfinite(helm_fields["gamma3m1"])
    )
    temp = np.exp(logtemp)[None, :]
    helm_gap_fallback_valid = (
        (rho_cgs >= rho_switch_cgs)
        & (~chabrier_valid)
        & (temp >= 10.0 ** helm_hot_fallback_log10_temp_min)
        & (helm_fields["fi_valid"] > 0.5)
        & helm_finite
    )
    helm_hot_preferred_valid = (
        (rho_cgs >= rho_switch_cgs)
        & (temp >= 10.0 ** helm_preferred_log10_temp_min)
        & (helm_fields["coulomb_valid"] > 0.5)
        & helm_finite
    )
    corner_valid = helm_gap_fallback_valid | helm_hot_preferred_valid
    use_blend_rho = (
        t13_valid
        & chabrier_valid
        & (rho_cgs >= rho_blend_cgs_min)
        & (rho_cgs < rho_switch_cgs)
    )
    use_blend_dense = (
        t13_valid
        & chabrier_valid
        & (rho_cgs >= rho_switch_cgs)
        & (dense_onset_blend_frac > 0.0)
    )
    use_blend = use_blend_rho | use_blend_dense
    use_corner = (~use_blend) & corner_valid
    use_chabrier = (
        chabrier_valid
        & (~use_blend_dense)
        & (~use_corner)
        & ((rho_cgs >= rho_switch_cgs) | (~t13_valid))
    )
    use_t13 = t13_valid & (~use_blend) & (~use_chabrier) & (~use_corner)
    union_valid = use_t13 | use_blend | use_chabrier | use_corner
    blend_frac_chabrier = np.where(use_blend_rho, rho_blend_pref, 0.0)
    blend_frac_chabrier = np.where(
        use_blend_dense,
        dense_onset_blend_frac,
        blend_frac_chabrier,
    )

    field_names = (
        "logpress",
        "logeps",
        "logcs2",
        "gamma1",
        "gamma3m1",
        "xh2",
        "xion",
        "xhe1",
        "xhe2",
        "mu",
        "beta_rad",
        "eps_physical",
        "eps_runtime",
        "pgas",
        "prad",
        "chi_rho",
        "chi_t",
        "cv",
        "cp",
        "entropy",
        "grad_ad",
        *PAIR_DIAGNOSTIC_FIELDS,
    )
    merged = {name: np.full_like(t13_fields["logpress"], np.nan, dtype=np.float64) for name in field_names}
    linear_field_map = {
        "gamma1": "gamma1",
        "gamma3m1": "gamma3m1",
        "xh2": "xh2",
        "xion": "xion",
        "xhe1": "xhe1",
        "xhe2": "xhe2",
        "mu": "mu",
        "beta_rad": "beta_rad",
        "eps_physical": "eps_physical",
        "eps_runtime": "eps_runtime",
        "pgas": "pgas",
        "prad": "prad",
        "chi_rho": "chi_rho",
        "chi_t": "chi_t",
        "cv": "cv",
        "cp": "cp",
        "entropy": "entropy",
        "grad_ad": "grad_ad",
        "ppair": "ppair",
        "epair_kin_over_rho": "epair_kin_over_rho",
        "epair_rest_over_rho": "epair_rest_over_rho",
        "logne_minus": "logne_minus",
        "logne_plus": "logne_plus",
        "eta_e": "eta_e",
        "pair_pressure_fraction": "pair_pressure_fraction",
    }

    pressure_total = np.full_like(t13_fields["logpress"], np.nan, dtype=np.float64)
    eps_total = np.full_like(t13_fields["logpress"], np.nan, dtype=np.float64)
    cs2 = np.full_like(t13_fields["logpress"], np.nan, dtype=np.float64)
    pressure_total[use_t13] = t13_linear["pressure_total"][use_t13]
    eps_total[use_t13] = t13_linear["eps_total"][use_t13]
    cs2[use_t13] = t13_linear["cs2"][use_t13]
    pressure_total[use_chabrier] = chabrier_linear["pressure_total"][use_chabrier]
    eps_total[use_chabrier] = chabrier_linear["eps_total"][use_chabrier]
    cs2[use_chabrier] = chabrier_linear["cs2"][use_chabrier]
    pressure_total[use_corner] = np.exp(helm_fields["logpress"][use_corner])
    eps_total[use_corner] = np.exp(helm_fields["logeps"][use_corner])
    cs2[use_corner] = np.exp(helm_fields["logcs2"][use_corner])
    if np.any(use_blend):
        w = blend_frac_chabrier[use_blend]
        pressure_total[use_blend] = (
            (1.0 - w) * t13_linear["pressure_total"][use_blend]
            + w * chabrier_linear["pressure_total"][use_blend]
        )
        eps_total[use_blend] = (
            (1.0 - w) * t13_linear["eps_total"][use_blend]
            + w * chabrier_linear["eps_total"][use_blend]
        )
        cs2[use_blend] = (
            (1.0 - w) * t13_linear["cs2"][use_blend]
            + w * chabrier_linear["cs2"][use_blend]
        )
    merged["logpress"] = np.log(np.maximum(pressure_total, 1.0e-300))
    merged["logeps"] = np.log(np.maximum(eps_total, 1.0e-300))
    merged["logcs2"] = np.log(np.maximum(cs2, 1.0e-300))
    for name, linear_name in linear_field_map.items():
        merged[name][use_t13] = t13_linear[linear_name][use_t13]
        merged[name][use_chabrier] = chabrier_linear[linear_name][use_chabrier]
        merged[name][use_corner] = helm_fields[name][use_corner]
        if np.any(use_blend):
            w = blend_frac_chabrier[use_blend]
            merged[name][use_blend] = (
                (1.0 - w) * t13_linear[linear_name][use_blend]
                + w * chabrier_linear[linear_name][use_blend]
            )
    merged["pair_pressure_fraction"] = np.where(
        pressure_total > 0.0,
        np.maximum(merged["ppair"], 0.0) / np.maximum(pressure_total, 1.0e-300),
        0.0,
    )

    merged["eps_envelope_correction"] = enforce_monotone_envelope(merged["logeps"], logtemp)
    rebuild_union_thermo_closure(merged, logrho, logtemp)

    for name in ("gamma_coulomb_h", "gamma_coulomb_he", "f_coulomb_over_kT"):
        values = np.zeros_like(pressure_total)
        values[use_corner] = helm_fields[name][use_corner]
        merged[name] = values

    merged["union_valid"] = union_valid.astype(np.float64)
    merged["source_t13"] = use_t13.astype(np.float64)
    merged["source_blend_chabrier"] = use_blend.astype(np.float64)
    merged["source_chabrier"] = use_chabrier.astype(np.float64)
    merged["source_corner"] = use_corner.astype(np.float64)
    merged["blend_frac_chabrier"] = blend_frac_chabrier
    merged["chabrier_valid"] = chabrier_valid.astype(np.float64)
    merged["corner_valid"] = corner_valid.astype(np.float64)

    context = {
        "rho_blend_cgs_min": rho_blend_cgs_min,
        "rho_switch_cgs": rho_switch_cgs,
        "helm_hot_fallback_log10_temp_min": helm_hot_fallback_log10_temp_min,
        "helm_preferred_log10_temp_min": helm_preferred_log10_temp_min,
        "pair_diagnostic_overlay_chabrier_cells": n_pair_overlay,
        "chabrier": chabrier_context,
        "corner": helm_context,
    }
    return merged, context

def fill_nan_cells(
    merged: dict[str, np.ndarray],
    t13_fields: dict[str, np.ndarray],
    helm_fields: dict[str, np.ndarray],
    t13_valid: np.ndarray,
    helm_finite: np.ndarray,
    use_t13: np.ndarray,
    use_scvh: np.ndarray,
    use_cp: np.ndarray,
    use_helm: np.ndarray,
) -> np.ndarray:
    """Fill uncovered union cells to make the table rectangular."""
    unfilled = ~(use_t13 | use_scvh | use_cp | use_helm)
    fill_with_t13 = unfilled & t13_valid
    fill_with_helm = unfilled & (~fill_with_t13) & helm_finite
    source_filler = np.zeros_like(merged["logeps"], dtype=np.int32)
    source_filler[fill_with_t13] = 1
    source_filler[fill_with_helm] = 2

    all_names = set(merged.keys()) | set(t13_fields.keys()) | set(helm_fields.keys())
    for name in all_names:
        if name not in merged:
            continue
        if name in t13_fields:
            merged[name][fill_with_t13] = t13_fields[name][fill_with_t13]
        if name in helm_fields:
            merged[name][fill_with_helm] = helm_fields[name][fill_with_helm]

    filled_t13 = int(np.count_nonzero(fill_with_t13))
    filled_helm = int(np.count_nonzero(fill_with_helm))
    remaining = int(np.count_nonzero(unfilled & (~fill_with_t13) & (~fill_with_helm)))
    print(
        "Gap filling: "
        f"T13 filler cells = {filled_t13}, "
        f"HELM filler cells = {filled_helm}, "
        f"remaining uncovered cells = {remaining}"
    )

    required = (
        "logpress",
        "logeps",
        "logcs2",
        "gamma1",
        "gamma3m1",
        "xh2",
        "xion",
        "xhe1",
        "xhe2",
        "mu",
        "beta_rad",
    )
    for name in required:
        if not np.all(np.isfinite(merged[name])):
            n_bad = int(np.count_nonzero(~np.isfinite(merged[name])))
            raise RuntimeError(f"Gap filling left {n_bad} non-finite '{name}' cells.")
    return source_filler


def enforce_monotone_envelope(
    logeps: np.ndarray,
    logtemp: np.ndarray,
) -> np.ndarray:
    """Clamp logeps in-place so each density row is strictly increasing in T."""
    del logtemp
    nrho, ntemp = logeps.shape
    correction = np.zeros_like(logeps)
    n_fixed = 0
    fixed_rows: set[int] = set()
    for ir in range(nrho):
        for it in range(1, ntemp):
            if logeps[ir, it] <= logeps[ir, it - 1]:
                min_step = max(1.0e-14, 1.0e-12 * abs(logeps[ir, it - 1]))
                old_val = logeps[ir, it]
                logeps[ir, it] = logeps[ir, it - 1] + min_step
                correction[ir, it] = logeps[ir, it] - old_val
                n_fixed += 1
                fixed_rows.add(ir)
    print(
        "Monotone envelope: "
        f"fixed {n_fixed} cells across {len(fixed_rows)} density rows"
    )
    diffs = np.diff(logeps, axis=1)
    if not np.all(diffs > 0.0):
        raise RuntimeError("Failed to enforce strict monotonicity in logeps.")
    return correction


def smootherstep(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.zeros_like(eta)
    ds = np.zeros_like(eta)
    d2s = np.zeros_like(eta)
    mid = (eta > 0.0) & (eta < 1.0)
    em = eta[mid]
    s[mid] = 6.0 * em**5 - 15.0 * em**4 + 10.0 * em**3
    ds[mid] = 30.0 * em**2 * (1.0 - em) ** 2
    d2s[mid] = 120.0 * em**3 - 180.0 * em**2 + 60.0 * em
    s[eta >= 1.0] = 1.0
    return s, ds, d2s


def scvh_t13_cp_helm_union_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
    scvh_data_dir: str | None = None,
    rho_switch_cgs: float = SCVH_T13_SWITCH_RHO_CGS,
    scvh_logp_step: float = SCVH95_INTERP_LOGP_STEP,
    scvh_ppt_logp_step: float = SCVH95_INTERP_PPT_LOGP_STEP,
    helm_hot_fallback_log10_temp_min: float = HELM_HOT_FALLBACK_LOG10_TEMP_MIN,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    t13_fields = t13_lte_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
        zpe_subtract=False,
    )
    ensure_pair_diagnostics(t13_fields, logne_name=None)
    scvh_fields, scvh_context = scvh95_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
        data_dir=scvh_data_dir,
        logp_step=scvh_logp_step,
        ppt_logp_step=scvh_ppt_logp_step,
    )
    ensure_pair_diagnostics(scvh_fields)
    cp_fields, cp_context = cp_like_hhe_fi_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
    )
    helm_fields, helm_context = helm_style_hhe_fi_thermo(
        logrho,
        logtemp,
        x_h,
        y_he,
        radiation_enabled,
    )

    rho = np.exp(logrho)[:, None]
    temp = np.exp(logtemp)[None, :]
    rho_mask_high = rho > rho_switch_cgs
    t13_valid = (
        np.isfinite(t13_fields["logpress"])
        & np.isfinite(t13_fields["logeps"])
        & np.isfinite(t13_fields["logcs2"])
        & np.isfinite(t13_fields["gamma1"])
        & np.isfinite(t13_fields["gamma3m1"])
    )
    scvh_valid = scvh_fields["scvh_valid"] > 0.5
    cp_valid = cp_fields["fi_valid"] > 0.5
    helm_finite = (
        np.isfinite(helm_fields["logpress"])
        & np.isfinite(helm_fields["logeps"])
        & np.isfinite(helm_fields["logcs2"])
        & np.isfinite(helm_fields["gamma1"])
        & np.isfinite(helm_fields["gamma3m1"])
    )
    helm_hot_valid = (
        rho_mask_high
        & (temp >= 10.0 ** helm_hot_fallback_log10_temp_min)
        & helm_finite
    )

    use_scvh = scvh_valid
    use_cp = (~use_scvh) & cp_valid
    use_helm = (~use_scvh) & (~use_cp) & helm_hot_valid
    use_t13 = (~rho_mask_high) & (~use_scvh) & (~use_cp) & (~use_helm) & t13_valid
    union_valid = use_scvh | use_cp | use_helm | use_t13

    field_names = (
        "logpress",
        "logeps",
        "logcs2",
        "gamma1",
        "gamma3m1",
        "xh2",
        "xion",
        "xhe1",
        "xhe2",
        "mu",
        "beta_rad",
        "eps_physical",
        "eps_runtime",
        "pgas",
        "prad",
        "chi_rho",
        "chi_t",
        "cv",
        "cp",
        "entropy",
        "grad_ad",
        "logne",
        *PAIR_DIAGNOSTIC_FIELDS,
    )
    merged = {name: np.full_like(t13_fields["logpress"], np.nan, dtype=np.float64) for name in field_names}
    for name in field_names:
        if name in t13_fields:
            merged[name][use_t13] = t13_fields[name][use_t13]
        if name in scvh_fields:
            merged[name][use_scvh] = scvh_fields[name][use_scvh]
        if name in cp_fields:
            merged[name][use_cp] = cp_fields[name][use_cp]
        if name in helm_fields:
            merged[name][use_helm] = helm_fields[name][use_helm]

    source_filler = fill_nan_cells(
        merged,
        t13_fields,
        helm_fields,
        t13_valid,
        helm_finite,
        use_t13,
        use_scvh,
        use_cp,
        use_helm,
    )
    eps_envelope_correction = enforce_monotone_envelope(merged["logeps"], logtemp)

    merged["union_valid"] = np.ones_like(union_valid, dtype=np.float64)
    merged["source_t13"] = use_t13.astype(np.float64)
    merged["source_scvh"] = use_scvh.astype(np.float64)
    merged["source_corner_cp"] = use_cp.astype(np.float64)
    merged["source_corner_helm"] = use_helm.astype(np.float64)
    merged["source_corner"] = (use_cp | use_helm).astype(np.float64)
    merged["source_filler"] = source_filler.astype(np.float64)
    merged["eps_envelope_correction"] = eps_envelope_correction
    merged["scvh_valid"] = scvh_valid.astype(np.float64)
    merged["corner_cp_valid"] = cp_valid.astype(np.float64)
    merged["corner_helm_hot_valid"] = helm_hot_valid.astype(np.float64)
    merged["corner_valid"] = (cp_valid | helm_hot_valid).astype(np.float64)
    if "gamma_coulomb_h" in cp_fields:
        merged["gamma_coulomb_h"] = np.where(use_cp, cp_fields["gamma_coulomb_h"], np.nan)
        merged["gamma_coulomb_he"] = np.where(use_cp, cp_fields["gamma_coulomb_he"], np.nan)
        merged["f_coulomb_over_kT"] = np.where(use_cp, cp_fields["f_coulomb_over_kT"], np.nan)

    context = {
        "rho_switch_cgs": rho_switch_cgs,
        "scvh": scvh_context,
        "corner_cp": cp_context,
        "corner_helm": helm_context,
        "helm_hot_fallback_log10_temp_min": helm_hot_fallback_log10_temp_min,
    }
    return merged, context

def validate_sparse_hhe_table(
    fields: dict[str, np.ndarray],
    model: str,
    radiation_enabled: bool,
    valid_mask_name: str,
) -> None:
    valid = fields[valid_mask_name] > 0.5
    if not np.any(valid):
        raise RuntimeError(f"{model} produced no finite thermodynamic cells.")
    if np.any(np.exp(fields["logpress"][valid]) <= 0.0):
        raise RuntimeError(f"{model} produced non-positive finite pressures.")
    if np.any(np.exp(fields["logeps"][valid]) <= 0.0):
        raise RuntimeError(f"{model} produced non-positive finite internal energies.")

    gamma_valid = valid & np.isfinite(fields["logcs2"]) & np.isfinite(fields["gamma1"]) & np.isfinite(fields["gamma3m1"])
    if np.any(fields["gamma1"][gamma_valid] <= 0.0):
        raise RuntimeError(f"{model} produced non-positive finite Gamma1 values.")
    if np.any(fields["gamma3m1"][gamma_valid] <= 0.0):
        raise RuntimeError(f"{model} produced non-positive finite Gamma3-1 values.")
    if np.any(np.exp(fields["logcs2"][gamma_valid]) <= 0.0):
        raise RuntimeError(f"{model} produced non-positive finite sound speeds.")

    for name in ("xh2", "xion", "xhe1", "xhe2", "mu", "chi_rho", "chi_t", "cv", "cp", "entropy"):
        if np.any(~np.isfinite(fields[name][valid])):
            raise RuntimeError(f"{model} produced non-finite '{name}' values inside the advertised valid mask.")

    if np.any(fields[valid_mask_name] != fields[valid_mask_name].astype(bool)):
        raise RuntimeError(f"{model} {valid_mask_name} must remain binary.")
    if np.any(fields["beta_rad"][valid] < 0.0) or np.any(fields["beta_rad"][valid] > 1.0 + 1.0e-12):
        raise RuntimeError(f"{model} beta_rad is outside [0,1] on finite cells.")
    if not radiation_enabled and np.any(np.abs(fields["beta_rad"][valid]) > 1.0e-12):
        raise RuntimeError(f"{model} beta_rad must vanish when radiation is disabled.")

    if np.any((fields["xh2"][valid] < 0.0) | (fields["xh2"][valid] > 1.0 + 1.0e-10)):
        raise RuntimeError(f"{model} xh2 is outside [0,1].")
    if np.any((fields["xion"][valid] < 0.0) | (fields["xion"][valid] > 1.0 + 1.0e-10)):
        raise RuntimeError(f"{model} xion is outside [0,1].")
    if np.any(fields["xh2"][valid] + fields["xion"][valid] > 1.0 + 1.0e-8):
        raise RuntimeError(f"{model} xh2 + xion exceeds unity.")
    if np.any((fields["xhe1"][valid] < 0.0) | (fields["xhe1"][valid] > 1.0 + 1.0e-10)):
        raise RuntimeError(f"{model} xhe1 is outside [0,1].")
    if np.any((fields["xhe2"][valid] < 0.0) | (fields["xhe2"][valid] > 1.0 + 1.0e-10)):
        raise RuntimeError(f"{model} xhe2 is outside [0,1].")
    if np.any(fields["xhe1"][valid] + fields["xhe2"][valid] > 1.0 + 1.0e-10):
        raise RuntimeError(f"{model} xhe1 + xhe2 exceeds unity.")
    if np.any(fields["mu"][valid] <= 0.0):
        raise RuntimeError(f"{model} mu must remain positive.")

    for name in PAIR_DIAGNOSTIC_FIELDS:
        if name in fields and np.any(~np.isfinite(fields[name][valid])):
            raise RuntimeError(f"{model} produced non-finite pair diagnostic '{name}' inside the advertised valid mask.")
    if "ppair" in fields and np.any(fields["ppair"][valid] < -1.0e-12):
        raise RuntimeError(f"{model} ppair must remain non-negative.")
    if "pair_pressure_fraction" in fields:
        pair_fraction = fields["pair_pressure_fraction"][valid]
        if np.any(pair_fraction < -1.0e-12):
            raise RuntimeError(f"{model} pair_pressure_fraction must remain non-negative.")


def validate_lte_table(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    fields: dict[str, np.ndarray],
    model: str,
    x_h: float,
    radiation_enabled: bool,
    zpe_subtract: bool,
    union_context: dict[str, object] | None = None,
) -> None:
    if model == "scvh_t13_cp_helm_union":
        validate_sparse_hhe_table(fields, model, radiation_enabled, valid_mask_name="union_valid")
        if np.any(fields["source_t13"] != fields["source_t13"].astype(bool)):
            raise RuntimeError(f"{model} source_t13 mask must remain binary.")
        if np.any(fields["source_scvh"] != fields["source_scvh"].astype(bool)):
            raise RuntimeError(f"{model} source_scvh mask must remain binary.")
        if np.any(fields["source_corner_cp"] != fields["source_corner_cp"].astype(bool)):
            raise RuntimeError(f"{model} source_corner_cp mask must remain binary.")
        if np.any(fields["source_corner_helm"] != fields["source_corner_helm"].astype(bool)):
            raise RuntimeError(f"{model} source_corner_helm mask must remain binary.")
        if np.any((fields["source_t13"] + fields["source_scvh"] +
                   fields["source_corner_cp"] + fields["source_corner_helm"]) > 1.0 + 1.0e-12):
            raise RuntimeError(f"{model} source masks overlap.")
        if np.any(~np.isin(fields["source_filler"], (0.0, 1.0, 2.0))):
            raise RuntimeError(f"{model} source_filler must only contain 0/1/2.")
        if np.any(fields["eps_envelope_correction"] < 0.0):
            raise RuntimeError(f"{model} eps_envelope_correction must remain non-negative.")

        logeps = fields["logeps"]
        if np.any(~np.isfinite(logeps)):
            n_bad = int(np.sum(~np.isfinite(logeps)))
            raise RuntimeError(
                f"{model}: {n_bad} non-finite logeps cells remain after gap-filling."
            )
        for name in (
            "logpress",
            "logcs2",
            "gamma1",
            "gamma3m1",
            "xh2",
            "xion",
            "xhe1",
            "xhe2",
            "mu",
            "beta_rad",
        ):
            if np.any(~np.isfinite(fields[name])):
                n_bad = int(np.sum(~np.isfinite(fields[name])))
                raise RuntimeError(
                    f"{model}: {n_bad} non-finite {name} cells remain after gap-filling."
                )
        for name in PAIR_DIAGNOSTIC_FIELDS:
            if name in fields and np.any(~np.isfinite(fields[name])):
                n_bad = int(np.sum(~np.isfinite(fields[name])))
                raise RuntimeError(
                    f"{model}: {n_bad} non-finite pair diagnostic {name} cells remain after gap-filling."
                )
        if "ppair" in fields and np.any(fields["ppair"] < -1.0e-12):
            raise RuntimeError(f"{model}: ppair must remain non-negative.")
        if "pair_pressure_fraction" in fields:
            pair_fraction = fields["pair_pressure_fraction"]
            if np.any(pair_fraction < -1.0e-12):
                raise RuntimeError(f"{model}: pair_pressure_fraction must remain non-negative.")
        diffs = np.diff(logeps, axis=1)
        if np.any(diffs <= 0.0):
            bad = np.argwhere(diffs <= 0.0)
            raise RuntimeError(
                f"{model}: {len(bad)} non-monotone logeps pairs remain after envelope. "
                f"First bad: row {bad[0][0]}, cols {bad[0][1]}->{bad[0][1] + 1}"
            )
        if np.any(np.exp(logeps) <= 0.0):
            raise RuntimeError(f"{model}: non-positive eps values.")
        if np.any(np.exp(fields["logpress"]) <= 0.0):
            raise RuntimeError(f"{model}: non-positive pressure values.")
        if np.any(np.exp(fields["logcs2"]) <= 0.0):
            raise RuntimeError(f"{model}: non-positive cs2 values.")
        if np.any(fields["gamma1"] <= 0.0):
            raise RuntimeError(f"{model}: non-positive gamma1 values.")
        if np.any(fields["gamma3m1"] <= 0.0):
            raise RuntimeError(f"{model}: non-positive gamma3m1 values.")
        if np.any(fields["mu"] <= 0.0):
            raise RuntimeError(f"{model}: non-positive mu values.")
        print(
            f"{model}: rectangular QA passed "
            "(0 non-finite required-field cells, 0 non-monotone rows)."
        )
        return
    if model == "chabrier2021_t13_helm_union":
        validate_sparse_hhe_table(fields, model, radiation_enabled, valid_mask_name="union_valid")
        for name in (
            "source_t13",
            "source_blend_chabrier",
            "source_chabrier",
            "source_corner",
            "chabrier_valid",
            "corner_valid",
        ):
            if np.any(fields[name] != fields[name].astype(bool)):
                raise RuntimeError(f"{model} {name} mask must remain binary.")
        if np.any((fields["blend_frac_chabrier"] < 0.0) | (fields["blend_frac_chabrier"] > 1.0 + 1.0e-12)):
            raise RuntimeError(f"{model} blend_frac_chabrier must lie in [0,1].")
        if np.any((fields["blend_frac_chabrier"] > 0.0) & ~(fields["source_blend_chabrier"] > 0.5)):
            raise RuntimeError(f"{model} blend_frac_chabrier must only be non-zero on source_blend_chabrier cells.")
        if np.any(
            (
                fields["source_t13"]
                + fields["source_blend_chabrier"]
                + fields["source_chabrier"]
                + fields["source_corner"]
            )
            > 1.0 + 1.0e-12
        ):
            raise RuntimeError(f"{model} source masks overlap.")
        for name in (
            "logpress",
            "logeps",
            "logcs2",
            "gamma1",
            "gamma3m1",
            "xh2",
            "xion",
            "xhe1",
            "xhe2",
            "mu",
            "beta_rad",
            "eps_physical",
            "eps_runtime",
            "pgas",
            "prad",
            "chi_rho",
            "chi_t",
            "cv",
            "cp",
            "entropy",
            "grad_ad",
        ):
            if np.any(~np.isfinite(fields[name])):
                n_bad = int(np.sum(~np.isfinite(fields[name])))
                raise RuntimeError(f"{model}: {n_bad} non-finite {name} cells remain.")
        for name in PAIR_DIAGNOSTIC_FIELDS:
            if name in fields and np.any(~np.isfinite(fields[name])):
                n_bad = int(np.sum(~np.isfinite(fields[name])))
                raise RuntimeError(f"{model}: {n_bad} non-finite pair diagnostic {name} cells remain.")
        if "ppair" in fields and np.any(fields["ppair"] < -1.0e-12):
            raise RuntimeError(f"{model}: ppair must remain non-negative.")
        if "pair_pressure_fraction" in fields:
            pair_fraction = fields["pair_pressure_fraction"]
            if np.any(pair_fraction < -1.0e-12):
                raise RuntimeError(f"{model}: pair_pressure_fraction must remain non-negative.")
        if np.any(np.diff(fields["logeps"], axis=1) <= 0.0):
            bad = np.argwhere(np.diff(fields["logeps"], axis=1) <= 0.0)
            raise RuntimeError(
                f"{model}: {len(bad)} non-monotone logeps pairs remain. "
                f"First bad: row {bad[0][0]}, cols {bad[0][1]}->{bad[0][1] + 1}"
            )
        if np.any((fields["source_corner"] > 0.5) & ~(fields["corner_valid"] > 0.5)):
            raise RuntimeError(f"{model}: HELM source cells must pass the full corner validity gate.")
        if union_context is None:
            raise RuntimeError(f"{model}: missing union context for HELM ownership validation.")
        hot_dense = (
            (np.exp(logrho)[:, None] >= union_context["rho_switch_cgs"])
            & (np.exp(logtemp)[None, :] >= 10.0 ** union_context["helm_preferred_log10_temp_min"])
            & (fields["corner_valid"] > 0.5)
        )
        if np.any(hot_dense & ~(fields["source_corner"] > 0.5)):
            raise RuntimeError(
                f"{model}: full HELM must own every valid hot dense cell above the preferred switch."
            )
        if np.any(fields["eps_envelope_correction"] < 0.0):
            raise RuntimeError(f"{model}: eps_envelope_correction must remain non-negative.")

        rho = np.exp(logrho)[:, None]
        pressure = np.exp(fields["logpress"])
        eps = np.exp(fields["logeps"])
        cs2 = np.exp(fields["logcs2"])

        def max_relative_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
            return float(np.max(np.abs(lhs - rhs) / np.maximum.reduce((np.ones_like(lhs), np.abs(lhs), np.abs(rhs)))))

        closure_errors = {
            "eps_runtime": max_relative_error(eps, fields["eps_runtime"]),
            "eps_physical": max_relative_error(eps, fields["eps_physical"]),
            "pressure_split": max_relative_error(pressure, fields["pgas"] + fields["prad"]),
            "sound_speed": max_relative_error(cs2, fields["gamma1"] * pressure / rho),
            "grad_ad": max_relative_error(
                fields["grad_ad"],
                fields["gamma3m1"] / fields["gamma1"],
            ),
        }
        worst_name = max(closure_errors, key=closure_errors.get)
        if closure_errors[worst_name] > 5.0e-12:
            raise RuntimeError(
                f"{model}: {worst_name} closure error is {closure_errors[worst_name]:.3e}."
            )
        for name in ("gamma_coulomb_h", "gamma_coulomb_he"):
            if np.any(~np.isfinite(fields[name])) or np.any(fields[name] < 0.0):
                raise RuntimeError(f"{model}: invalid full-HELM Coulomb diagnostic {name}.")
        print(
            f"{model}: rectangular QA passed "
            "(0 non-finite required-field cells, 0 non-monotone rows, "
            f"max closure error {closure_errors[worst_name]:.3e})."
        )
        return
    logeps = fields["logeps"]
    eps_runtime = fields.get("eps_runtime")
    if np.any(~np.isfinite(logeps)):
        raise RuntimeError("Encountered non-finite logeps values in LTE table.")
    if eps_runtime is not None and np.any(eps_runtime <= 0.0):
        bad = np.argwhere(eps_runtime <= 0.0)[0]
        raise RuntimeError(
            "Runtime inversion energy became non-positive at "
            f"logrho index {bad[0]}, logtemp index {bad[1]}."
        )
    if np.any(np.diff(logeps, axis=1) <= 0.0):
        bad = np.argwhere(np.diff(logeps, axis=1) <= 0.0)[0]
        raise RuntimeError(
            "eps(T) is not strictly monotonic at logrho index "
            f"{bad[0]}, logtemp interval {bad[1]}->{bad[1] + 1}."
        )
    for name in ("gamma1", "gamma3m1", "mu", "xh2", "xion", "xhe1", "xhe2", "beta_rad"):
        if np.any(~np.isfinite(fields[name])):
            raise RuntimeError(f"Encountered non-finite values in LTE field '{name}'.")

    if np.any((fields["xh2"] < 0.0) | (fields["xh2"] > 1.0 + 1.0e-10)):
        raise RuntimeError("xh2 is outside [0,1].")
    if np.any((fields["xion"] < 0.0) | (fields["xion"] > 1.0 + 1.0e-10)):
        raise RuntimeError("xion is outside [0,1].")
    if np.any(fields["xh2"] + fields["xion"] > 1.0 + 1.0e-8):
        raise RuntimeError("xh2 + xion exceeds unity for the hydrogen subsystem.")
    if np.any((fields["xhe1"] < 0.0) | (fields["xhe1"] > 1.0 + 1.0e-10)):
        raise RuntimeError("xhe1 is outside [0,1].")
    if np.any((fields["xhe2"] < 0.0) | (fields["xhe2"] > 1.0 + 1.0e-10)):
        raise RuntimeError("xhe2 is outside [0,1].")
    if np.any(fields["xhe1"] + fields["xhe2"] > 1.0 + 1.0e-10):
        raise RuntimeError("xhe1 + xhe2 exceeds unity.")
    if np.any(fields["mu"] <= 0.0):
        raise RuntimeError("mu must remain positive.")
    if np.any((fields["beta_rad"] < 0.0) | (fields["beta_rad"] > 1.0 + 1.0e-12)):
        raise RuntimeError("beta_rad is outside [0,1].")
    if np.any(np.exp(fields["logcs2"]) <= 0.0):
        raise RuntimeError("cs2 must remain positive.")

    rho_mid = len(logrho) // 2
    xh2_row = fields["xh2"][rho_mid]
    xion_row = fields["xion"][rho_mid]
    xhe1_row = fields["xhe1"][rho_mid]
    xhe2_row = fields["xhe2"][rho_mid]
    h2_mid = np.argmax(xh2_row <= 0.5)
    h_mid = np.argmax(xion_row >= 0.5)
    he1_mid = np.argmax(xhe1_row >= 0.5)
    he2_mid = np.argmax(xhe2_row >= 0.5)
    if (xh2_row[h2_mid] <= 0.5) and (xion_row[h_mid] >= 0.5) and (h2_mid > h_mid):
        raise RuntimeError("H2 dissociation midpoint occurs after hydrogen ionization.")
    if (xion_row[h_mid] >= 0.5) and (xhe1_row[he1_mid] >= 0.5) and (h_mid > he1_mid):
        raise RuntimeError("Hydrogen ionization midpoint occurs after helium first ionization.")
    if (xion_row[h_mid] >= 0.5) and (xhe2_row[he2_mid] >= 0.5) and (h_mid > he2_mid):
        raise RuntimeError("Hydrogen ionization midpoint occurs after helium double ionization.")
    if radiation_enabled and np.max(fields["beta_rad"][:, -1]) <= 1.0e-6:
        raise RuntimeError("Radiation pressure was enabled but beta_rad never becomes significant.")
    if not radiation_enabled and np.any(np.abs(fields["beta_rad"]) > 1.0e-14):
        raise RuntimeError("beta_rad must remain zero when radiation is disabled.")

    if model == "t13":
        for name in ("chi_rho", "chi_t", "cv", "cp", "entropy", *PAIR_DIAGNOSTIC_FIELDS):
            if np.any(~np.isfinite(fields[name])):
                raise RuntimeError(f"Encountered non-finite '{name}' values in the Tomida LTE table.")
        if np.any(fields["ppair"] < -1.0e-12):
            raise RuntimeError("Tomida LTE ppair must remain non-negative.")
        if np.any(fields["pair_pressure_fraction"] < -1.0e-12):
            raise RuntimeError("Tomida LTE pair_pressure_fraction must remain non-negative.")
        if np.any(fields["chi_rho"] < -1.0e-12):
            raise RuntimeError("Tomida LTE chi_rho became negative.")
        if np.any(fields["chi_t"] <= 0.0):
            raise RuntimeError("Tomida LTE chi_t must remain positive.")
        if np.any(fields["cv"] <= 0.0) or np.any(~np.isfinite(fields["cp"])) or np.any(fields["cp"] <= 0.0):
            raise RuntimeError("Tomida LTE heat capacities must remain positive and finite.")
        if zpe_subtract:
            zpe_diff = fields["eps_physical"] - fields["eps_runtime"]
            tol = 1.0e-12 * np.maximum(1.0, np.abs(fields["eps_physical"]))
            if np.any(zpe_diff < -tol):
                raise RuntimeError("ZPE subtraction made eps exceed the physical internal energy.")
            zpe_const = x_h * H2_VIB_ZPE / (2.0 * M_H)
            # The constant offset is only numerically meaningful in the molecular regime.
            # At very large eps_physical the subtraction is below floating-point resolution.
            resolved = (fields["xh2"] > 0.5) & (fields["eps_physical"] <= 1.0e12 * zpe_const)
            if np.any(resolved):
                rel = np.abs(zpe_diff[resolved] - zpe_const) / np.maximum(zpe_const, 1.0)
                if np.max(rel) > 1.0e-6:
                    raise RuntimeError("ZPE-subtracted and physical energies are inconsistent.")
            if np.any(fields["eps_runtime"] > fields["eps_physical"] + tol):
                raise RuntimeError("ZPE-subtracted and physical energies are inconsistent.")


def write_table(
    path: str,
    logrho: np.ndarray,
    logtemp: np.ndarray,
    fields: dict[str, np.ndarray],
    model: str,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
    zpe_subtract: bool,
    union_context: dict[str, object] | None = None,
) -> None:
    ensure_parent(path)

    if model == "t13":
        metadata = [
            "endianness = little",
            f"table_type = {'lte_t13_prad_lte' if radiation_enabled else 'lte_t13_lte'}",
            f"eos_name = {'lte_table_t13_prad' if radiation_enabled else 'lte_table_t13'}",
            "log_axis_base = e",
            "chemistry_model = tomida2013_h2_h_he",
            "partition_function_model = tomida2013_appendix1_partition_functions",
            "source_primary = Tomida2013_Appendix1",
            "source_application = Andalman2025_EOS",
            f"radiation_pressure = {'on' if radiation_enabled else 'off'}",
            "radiation_model = pejcha2016_style_additive_thermodynamics",
            "h2_enabled = on",
            f"zpe_subtracted = {'on' if zpe_subtract else 'off'}",
            "ortho_para_ratio_h2 = 3:1_fixed",
            "closure_method = finite_difference",
            "pair_production = on",
            "pair_backend = helm_style_ideal_relativistic_fermi_dirac_quadrature",
            "pair_reference = MESA_HELM_TimmesSwesty2000_TimmesArnett1999",
            "pair_quadrature = generalized_laguerre_alpha_1_2",
            "pair_rest_mass_energy_in_eps = on",
            "pair_rest_mass_energy_convention = created_pairs_only",
            f"pair_active_log10_temp_min = {PAIR_ACTIVE_LOG10_TEMP_MIN}",
        ]
        field_names = [
            "logpress",
            "logeps",
            "logcs2",
            "gamma1",
            "gamma3m1",
            "xh2",
            "xion",
            "xhe1",
            "xhe2",
            "mu",
            "beta_rad",
            "eps_physical",
            "u_trans_over_rho",
            "u_h2_rot_over_rho",
            "u_h2_vib_over_rho",
            "u_h2_zpe_over_rho",
            "u_diss_h2_over_rho",
            "u_ion_h_over_rho",
            "u_ion_he1_over_rho",
            "u_ion_he2_over_rho",
            "u_rad_over_rho",
            "pgas",
            "prad",
            "chi_rho",
            "chi_t",
            "cv",
            "cp",
            "entropy",
            *PAIR_DIAGNOSTIC_FIELDS,
        ]
    elif model == "scvh_t13_cp_helm_union":
        if union_context is None:
            raise RuntimeError("SCvH+T13+corner union writing requires the union build context.")
        table_type = "lte_scvh_t13_cp_helm_union_prad" if radiation_enabled else "lte_scvh_t13_cp_helm_union"
        eos_name = "lte_table_scvh_t13_cp_helm_union_prad" if radiation_enabled else "lte_table_scvh_t13_cp_helm_union"
        union_corner_backstop = (
            f"{union_context['corner_cp']['corner_model']}->"
            f"{union_context['corner_helm']['corner_model']}"
        )
        metadata = [
            "endianness = little",
            f"table_type = {table_type}",
            f"eos_name = {eos_name}",
            "log_axis_base = e",
            "chemistry_model = t13_lowrho_plus_scvh1995_plus_fully_ionized_corner",
            "union_primary_lowrho = t13",
            "union_primary_highrho = scvh1995",
            f"union_corner_backstop = {union_corner_backstop}",
            "union_policy = scvh_preferred_then_cp_then_helm_then_filled_monotone_envelope",
            f"radiation_pressure = {'on' if radiation_enabled else 'off'}",
            "h2_enabled = on",
            "closure_method = parent_native_closure_with_sparse_union_mask",
            "corner_backstop_status = prototype",
            "pair_production = on",
            "pair_backend = helm_style_ideal_relativistic_fermi_dirac_quadrature",
            "pair_reference = MESA_HELM_TimmesSwesty2000_TimmesArnett1999",
            "pair_quadrature = generalized_laguerre_alpha_1_2",
            "pair_rest_mass_energy_in_eps = on",
            "pair_rest_mass_energy_convention = created_pairs_only",
            "mu_excludes_created_pairs = true",
            f"source_url_scvh = {union_context['scvh']['source_url']}",
            f"scvh_interp_log10_press_step = {union_context['scvh']['interp_log10_press_step']}",
            f"scvh_interp_ppt_log10_press_step = {union_context['scvh']['interp_ppt_log10_press_step']}",
        ]
        field_names = [
            "logpress",
            "logeps",
            "logcs2",
            "gamma1",
            "gamma3m1",
            "xh2",
            "xion",
            "xhe1",
            "xhe2",
            "mu",
            "beta_rad",
            "union_valid",
            "source_t13",
            "source_scvh",
            "scvh_valid",
            *PAIR_DIAGNOSTIC_FIELDS,
        ]
        field_names.extend(
            [
                "source_corner_cp",
                "source_corner_helm",
                "source_corner",
                "corner_cp_valid",
                "corner_helm_hot_valid",
                "corner_valid",
                "source_filler",
                "eps_envelope_correction",
            ]
        )
    elif model == "chabrier2021_t13_helm_union":
        if union_context is None:
            raise RuntimeError("Chabrier2021+T13+corner union writing requires the union build context.")
        metadata = [
            "endianness = little",
            f"table_type = {'lte_chabrier2021_t13_helm_union_prad' if radiation_enabled else 'lte_chabrier2021_t13_helm_union'}",
            f"eos_name = {'lte_table_chabrier2021_t13_helm_union_prad' if radiation_enabled else 'lte_table_chabrier2021_t13_helm_union'}",
            "log_axis_base = e",
            "chemistry_model = t13_lowrho_chemistry_plus_simplified_hhe_atomic_proxy_on_chabrier2021_dense_thermo",
            "union_primary_lowrho = t13",
            "union_primary_highrho = chabrier2021_dense_fluid",
            f"union_corner_backstop = {union_context['corner']['corner_model']}",
            "union_policy = t13_lowrho_then_chabrier_dense_then_helm_preferred_above_1e8K",
            f"union_rho_blend_cgs_min = {union_context['rho_blend_cgs_min']}",
            f"union_rho_blend_cgs_max = {union_context['rho_switch_cgs']}",
            f"union_helm_preferred_log10_temp_min = {union_context['helm_preferred_log10_temp_min']}",
            f"radiation_pressure = {'on' if radiation_enabled else 'off'}",
            "h2_enabled = on",
            "closure_method = parent_native_response_blend_with_runtime_energy_and_sound_speed_reclosure",
            "composition_proxy_model = simplified_hhe_atomic_ion_proxy_on_pure_chabrier_cells",
            "helm_components = ideal_ions_plus_relativistic_degenerate_pairs_plus_photons_plus_coulomb",
            f"helm_coulomb_model = {union_context['corner']['coulomb_model']}",
            f"helm_coulomb_derivative = {union_context['corner']['coulomb_derivative']}",
            "helm_validity = ionized_gap_fallback_or_hot_preferred_classical_pc_liquid",
            "pair_production = on",
            "pair_backend = helm_style_ideal_relativistic_fermi_dirac_quadrature",
            "pair_diagnostic_overlay_on_chabrier_dense = helm_style_created_pair_pressure_fraction_only",
            "pair_reference = MESA_HELM_TimmesSwesty2000_TimmesArnett1999",
            "pair_quadrature = generalized_laguerre_alpha_1_2",
            "pair_rest_mass_energy_in_eps = on",
            "pair_rest_mass_energy_convention = created_pairs_only",
            "mu_excludes_created_pairs = true",
            f"source_url_chabrier_2019 = {union_context['chabrier']['source_url_2019']}",
            f"source_url_chabrier_2021 = {union_context['chabrier']['source_url_2021']}",
        ]
        field_names = [
            "logpress",
            "logeps",
            "logcs2",
            "gamma1",
            "gamma3m1",
            "xh2",
            "xion",
            "xhe1",
            "xhe2",
            "mu",
            "beta_rad",
            "eps_physical",
            "pgas",
            "prad",
            "chi_rho",
            "chi_t",
            "cv",
            "cp",
            "entropy",
            "grad_ad",
            "union_valid",
            "source_t13",
            "source_blend_chabrier",
            "source_chabrier",
            "source_corner",
            "blend_frac_chabrier",
            "chabrier_valid",
            "corner_valid",
        ]
    else:
        raise RuntimeError(f"Unsupported LTE table model '{model}'.")

    scalars = [
        f"k_b = {K_B:.17e}",
        f"h = {H_PLANCK:.17e}",
        f"m_e = {M_E:.17e}",
        f"m_h = {M_H:.17e}",
        f"m_he = {M_HE:.17e}",
        f"a_rad = {A_RAD:.17e}",
        f"x_h = {x_h:.17e}",
        f"y_he = {y_he:.17e}",
        f"chi_h = {CHI_H:.17e}",
        f"chi_he1 = {CHI_HE1:.17e}",
        f"chi_he2 = {CHI_HE2:.17e}",
        f"radiation_enabled = {1.0 if radiation_enabled else 0.0:.17e}",
    ]
    if model == "t13":
        scalars.extend(
            [
                f"chi_dis_h2 = {CHI_DIS_H2:.17e}",
                f"chi_ion_h = {CHI_ION_H:.17e}",
                f"chi_ion_he1 = {CHI_ION_HE1:.17e}",
                f"chi_ion_he2 = {CHI_ION_HE2:.17e}",
                f"theta_rot_h2 = {THETA_ROT_H2:.17e}",
                f"theta_vib_h2 = {THETA_VIB_H2:.17e}",
                f"h2_vib_zpe = {H2_VIB_ZPE:.17e}",
                f"eps_zpe_subtract_const = {x_h * H2_VIB_ZPE / (2.0 * M_H):.17e}",
                f"zpe_subtraction_enabled = {1.0 if zpe_subtract else 0.0:.17e}",
                f"pair_quad_n = {PAIR_QUAD_N:.17e}",
                f"pair_active_log10_temp_min = {PAIR_ACTIVE_LOG10_TEMP_MIN:.17e}",
                f"pair_created_rest_mass_energy = {1.0:.17e}",
            ]
        )
    elif model == "scvh_t13_cp_helm_union":
        corner_gate_xion = union_context["corner_cp"]["ionization_gate_xion_min"]
        corner_gate_xhe2 = union_context["corner_cp"]["ionization_gate_xhe2_min"]
        corner_coulomb = union_context["corner_cp"]["coulomb_enabled"]
        scalars.extend(
            [
                f"chi_dis_h2 = {CHI_DIS_H2:.17e}",
                f"chi_ion_h = {CHI_ION_H:.17e}",
                f"chi_ion_he1 = {CHI_ION_HE1:.17e}",
                f"chi_ion_he2 = {CHI_ION_HE2:.17e}",
                f"theta_rot_h2 = {THETA_ROT_H2:.17e}",
                f"theta_vib_h2 = {THETA_VIB_H2:.17e}",
                f"h2_vib_zpe = {H2_VIB_ZPE:.17e}",
                f"union_rho_switch_cgs = {union_context['rho_switch_cgs']:.17e}",
                f"scvh_log10_temp_min = {union_context['scvh']['log10_temp_min']:.17e}",
                f"scvh_log10_temp_max = {union_context['scvh']['log10_temp_max']:.17e}",
                f"scvh_log10_press_min = {union_context['scvh']['log10_press_min']:.17e}",
                f"scvh_log10_press_max = {union_context['scvh']['log10_press_max']:.17e}",
                f"fi_gate_xion_min = {corner_gate_xion:.17e}",
                f"fi_gate_xhe2_min = {corner_gate_xhe2:.17e}",
                f"fully_ionized_energy_offset = {fully_ionized_energy_offset(x_h, y_he):.17e}",
                f"corner_coulomb_enabled = {1.0 if corner_coulomb else 0.0:.17e}",
                f"pair_quad_n = {PAIR_QUAD_N:.17e}",
                f"pair_active_log10_temp_min = {PAIR_ACTIVE_LOG10_TEMP_MIN:.17e}",
                f"pair_created_rest_mass_energy = {1.0:.17e}",
            ]
        )
        scalars.extend(
            [
                f"pc_liquid_gamma_max = {union_context['corner_cp']['pc_liquid_gamma_max']:.17e}",
                f"pc_t_over_tp_min = {union_context['corner_cp']['pc_t_over_tp_min']:.17e}",
                f"helm_hot_fallback_log10_temp_min = {union_context['helm_hot_fallback_log10_temp_min']:.17e}",
            ]
        )
    elif model == "chabrier2021_t13_helm_union":
        scalars.extend(
            [
                f"chi_dis_h2 = {CHI_DIS_H2:.17e}",
                f"chi_ion_h = {CHI_ION_H:.17e}",
                f"chi_ion_he1 = {CHI_ION_HE1:.17e}",
                f"chi_ion_he2 = {CHI_ION_HE2:.17e}",
                f"theta_rot_h2 = {THETA_ROT_H2:.17e}",
                f"theta_vib_h2 = {THETA_VIB_H2:.17e}",
                f"h2_vib_zpe = {H2_VIB_ZPE:.17e}",
                f"union_rho_switch_cgs = {union_context['rho_switch_cgs']:.17e}",
                f"chabrier_log10_temp_min = {union_context['chabrier']['log10_temp_min']:.17e}",
                f"chabrier_log10_temp_max = {union_context['chabrier']['log10_temp_max']:.17e}",
                f"chabrier_log10_press_gpa_min = {union_context['chabrier']['log10_press_gpa_min']:.17e}",
                f"chabrier_log10_press_gpa_max = {union_context['chabrier']['log10_press_gpa_max']:.17e}",
                f"chabrier_interp_helium_lo = {union_context['chabrier']['interpolation_helium_lo']:.17e}",
                f"chabrier_interp_helium_hi = {union_context['chabrier']['interpolation_helium_hi']:.17e}",
                f"chabrier_interp_weight_hi = {union_context['chabrier']['interpolation_weight_hi']:.17e}",
                f"chabrier_interp_extrapolated = {1.0 if union_context['chabrier']['interpolation_extrapolated'] else 0.0:.17e}",
                f"chabrier_mixing_additive_volume = {1.0 if union_context['chabrier'].get('mixing_mode', 'mixture_interp') == 'additive_volume' else 0.0:.17e}",
                f"fi_gate_xion_min = {union_context['corner']['ionization_gate_xion_min']:.17e}",
                f"fi_gate_xhe2_min = {union_context['corner']['ionization_gate_xhe2_min']:.17e}",
                f"fully_ionized_energy_offset = {fully_ionized_energy_offset(x_h, y_he):.17e}",
                f"corner_coulomb_enabled = {1.0 if union_context['corner']['coulomb_enabled'] else 0.0:.17e}",
                f"helm_preferred_log10_temp_min = {union_context['helm_preferred_log10_temp_min']:.17e}",
                f"pc_liquid_gamma_max = {union_context['corner']['pc_liquid_gamma_max']:.17e}",
                f"pc_t_over_tp_min = {union_context['corner']['pc_t_over_tp_min']:.17e}",
                f"pair_quad_n = {PAIR_QUAD_N:.17e}",
                f"pair_state_chunk_size = {PAIR_STATE_CHUNK_SIZE:.17e}",
                f"pair_active_log10_temp_min = {PAIR_ACTIVE_LOG10_TEMP_MIN:.17e}",
                f"pair_created_rest_mass_energy = {1.0:.17e}",
            ]
        )

    points = [
        f"logrho = {len(logrho)}",
        f"logtemp = {len(logtemp)}",
    ]

    with open(path, "wb") as fp:
        write_block(fp, "metadatabegin", metadata, "metadataend")
        write_block(fp, "scalarsbegin", scalars, "scalarsend")
        write_block(fp, "pointsbegin", points, "pointsend")
        write_block(fp, "fieldsbegin", field_names, "fieldsend")

        fp.write(np.asarray(logrho, dtype=np.float64).tobytes(order="C"))
        fp.write(np.asarray(logtemp, dtype=np.float64).tobytes(order="C"))
        for name in field_names:
            fp.write(np.asarray(fields[name], dtype=np.float64).ravel(order="C").tobytes())


def validate_job(job: dict[str, object]) -> None:
    model = str(job["model"])
    supported_models = {
        "t13",
        "scvh_t13_cp_helm_union",
        "chabrier2021_t13_helm_union",
    }
    if model not in supported_models:
        raise SystemExit(
            f"Unsupported LTE model '{model}'. Maintained models: "
            + ", ".join(sorted(supported_models))
        )
    if int(job["nrho"]) < 3 or int(job["ntemp"]) < 3:
        raise SystemExit("nrho and ntemp must both be at least 3")
    if float(job["log10_rho_min"]) >= float(job["log10_rho_max"]):
        raise SystemExit("log10-rho-min must be less than log10-rho-max")
    if float(job["log10_temp_min"]) >= float(job["log10_temp_max"]):
        raise SystemExit("log10-temp-min must be less than log10-temp-max")
    if float(job["x_h"]) < 0.0 or float(job["y_he"]) < 0.0:
        raise SystemExit("x-h and y-he must both be non-negative")
    if not np.isclose(float(job["x_h"]) + float(job["y_he"]), 1.0, rtol=0.0, atol=1.0e-12):
        raise SystemExit("This generator currently assumes Z=0, so x-h + y-he must equal 1.")
    if float(job["scvh_logp_step"]) <= 0.0 or float(job["scvh_ppt_logp_step"]) <= 0.0:
        raise SystemExit("SCvH interpolation spacings must both be positive.")


def run_job(job: dict[str, object]) -> None:
    validate_job(job)

    model = str(job["model"])
    output = Path(os.fspath(job["output"]))
    nrho = int(job["nrho"])
    ntemp = int(job["ntemp"])
    log10_rho_min = float(job["log10_rho_min"])
    log10_rho_max = float(job["log10_rho_max"])
    log10_temp_min = float(job["log10_temp_min"])
    log10_temp_max = float(job["log10_temp_max"])
    x_h = float(job["x_h"])
    y_he = float(job["y_he"])
    radiation_pressure = bool(job["radiation_pressure"])
    scvh_data_dir = None if job["scvh_data_dir"] is None else os.fspath(job["scvh_data_dir"])
    scvh_logp_step = float(job["scvh_logp_step"])
    scvh_ppt_logp_step = float(job["scvh_ppt_logp_step"])

    zpe_subtract = model == "t13"
    if job["zpe_subtract"] is not None:
        zpe_subtract = bool(job["zpe_subtract"])

    logrho = np.linspace(
        log10_rho_min * math.log(10.0),
        log10_rho_max * math.log(10.0),
        nrho,
    )
    logtemp = np.linspace(
        log10_temp_min * math.log(10.0),
        log10_temp_max * math.log(10.0),
        ntemp,
    )

    union_context = None
    if model == "t13":
        if log10_rho_max > -1.0:
            print(
                "WARNING: Tomida et al. 2013 note that the ideal chemical-equilibrium "
                "partition-function EOS becomes inaccurate for rho >= 0.1 g cm^-3.",
                file=sys.stderr,
            )
        fields = t13_lte_thermo(
            logrho,
            logtemp,
            x_h,
            y_he,
            radiation_pressure,
            zpe_subtract,
        )
    elif model == "scvh_t13_cp_helm_union":
        fields, union_context = scvh_t13_cp_helm_union_thermo(
            logrho,
            logtemp,
            x_h,
            y_he,
            radiation_pressure,
            scvh_data_dir=scvh_data_dir,
            rho_switch_cgs=SCVH_T13_SWITCH_RHO_CGS,
            scvh_logp_step=scvh_logp_step,
            scvh_ppt_logp_step=scvh_ppt_logp_step,
        )
    elif model == "chabrier2021_t13_helm_union":
        fields, union_context = chabrier2021_t13_helm_union_thermo(
            logrho,
            logtemp,
            x_h,
            y_he,
            radiation_pressure,
            chabrier_data_dir=scvh_data_dir,
            rho_switch_cgs=CHABRIER2021_T13_SWITCH_RHO_CGS,
            chabrier_mixing=str(job.get("chabrier_mixing", "auto")),
        )
    else:
        raise RuntimeError(f"Unsupported LTE table model '{model}'.")

    validate_lte_table(
        logrho,
        logtemp,
        fields,
        model=model,
        x_h=x_h,
        radiation_enabled=radiation_pressure,
        zpe_subtract=zpe_subtract,
        union_context=union_context,
    )
    write_table(
        output,
        logrho,
        logtemp,
        fields,
        model=model,
        x_h=x_h,
        y_he=y_he,
        radiation_enabled=radiation_pressure,
        zpe_subtract=zpe_subtract,
        union_context=union_context,
    )

    print(
        "Wrote LTE table:",
        output,
        f"(model={model}, nrho={nrho}, ntemp={ntemp}, "
        f"log10rho=[{log10_rho_min}, {log10_rho_max}], "
        f"log10T=[{log10_temp_min}, {log10_temp_max}], "
        f"X={x_h}, Y={y_he}, prad={'on' if radiation_pressure else 'off'}, "
        f"zpe_subtract={'on' if zpe_subtract else 'off'})",
    )


def main(argv: list[str] | None = None) -> int:
    """Run every maintained job, or only the jobs named on the command line.

    Optional overrides for quick checks: --nrho N --ntemp N --output-dir DIR apply to
    the selected jobs (a coarse grid written elsewhere never touches eos_tables/).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    overrides: dict[str, object] = {}
    names: list[str] = []
    while args:
        arg = args.pop(0)
        if arg in ("--nrho", "--ntemp"):
            overrides[arg[2:]] = int(args.pop(0))
        elif arg == "--output-dir":
            overrides["output_dir"] = Path(args.pop(0))
        else:
            names.append(arg)
    known = {str(job["name"]): job for job in LTE_TABLE_JOBS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"Unknown job(s) {unknown}; known: {sorted(known)}")
    selected = [known[n] for n in names] if names else list(LTE_TABLE_JOBS)
    for job in selected:
        job = dict(job)
        for key in ("nrho", "ntemp"):
            if key in overrides:
                job[key] = overrides[key]
        if "output_dir" in overrides:
            job["output"] = Path(overrides["output_dir"]) / Path(os.fspath(job["output"])).name
        run_job(job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
