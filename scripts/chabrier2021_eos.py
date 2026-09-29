#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


LN10 = math.log(10.0)
K_B = 1.380649e-16
A_RAD = 7.5657e-15
C_LIGHT = 2.99792458e10

CHABRIER2021_FILES = {
    0.275: "TABLEEOS_2021_TP_Y0275_v1",
    0.292: "TABLEEOS_2021_TP_Y0292_v1",
    0.297: "TABLEEOS_2021_TP_Y0297_v1",
}
# Pure-component tables (Chabrier et al. 2019).  They are combined with the additive
# volume law (Chabrier & Debras 2021, Section 2) for helium fractions outside the narrow
# Y = 0.275-0.297 range spanned by the published mixture tables; see chabrier2021_thermo.
CHABRIER2021_PURE_FILES = {
    0.0: "TABLE_H_TP_v1",
    1.0: "TABLE_HE_TP_v1",
}
CHABRIER2021_MIXING_MODES = ("auto", "mixture_interp", "additive_volume")
# "auto" keeps the mixture-table interpolation (bitwise the historical behaviour) whenever
# the target Y lies within this distance of the published mixture range, and switches to
# the additive volume law otherwise.
CHABRIER2021_MIXTURE_INTERP_TOLERANCE = 0.01
CHABRIER2021_SOURCE_URL_2019 = "https://arxiv.org/abs/1902.01852"
CHABRIER2021_SOURCE_URL_2021 = "https://arxiv.org/abs/2107.04434"
CHABRIER2021_LOG10_TEMP_MIN = 2.0
CHABRIER2021_LOG10_TEMP_MAX = 8.0
CHABRIER2021_LOG10_PRESS_GPA_MIN = -9.0
CHABRIER2021_LOG10_PRESS_GPA_MAX = 13.0
CHABRIER2021_T13_BLEND_RHO_CGS_MIN = 1.0e-3
CHABRIER2021_T13_BLEND_RHO_CGS_MAX = 1.0e-2
CHABRIER2021_T13_SWITCH_RHO_CGS = CHABRIER2021_T13_BLEND_RHO_CGS_MAX


@dataclass(frozen=True)
class Chabrier2021Table:
    helium_fraction: float
    log10_temp: np.ndarray
    log10_press_gpa: np.ndarray
    fields: dict[str, np.ndarray]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "chabrier2021_data"


def ensure_chabrier2021_tables(data_dir: str | None = None, pure: bool = False) -> Path:
    root = _default_data_dir() if data_dir is None else Path(data_dir).expanduser().resolve()
    wanted = CHABRIER2021_PURE_FILES if pure else CHABRIER2021_FILES
    missing = [name for name in wanted.values() if not (root / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Chabrier 2021 dense-fluid tables are missing. Expected files in "
            f"{root}: {missing_list}"
        )
    return root


def _load_one_table(path: Path, helium_fraction: float) -> Chabrier2021Table:
    raw = np.loadtxt(path, comments="#", dtype=np.float64)
    log10_temp = np.unique(raw[:, 0])
    log10_press_gpa = np.unique(raw[:, 1])
    ntemp = log10_temp.size
    npress = log10_press_gpa.size
    expected_rows = ntemp * npress
    if raw.shape[0] != expected_rows:
        raise RuntimeError(
            f"{path} does not contain a regular Chabrier TP grid: "
            f"got {raw.shape[0]} rows, expected {expected_rows}."
        )

    fields = {
        "log10_rho": raw[:, 2].reshape(ntemp, npress),
        "log10_u_mj_per_kg": raw[:, 3].reshape(ntemp, npress),
        "log10_s_mj_per_kg_k": raw[:, 4].reshape(ntemp, npress),
        "dlnrho_dlnT_constP": raw[:, 5].reshape(ntemp, npress),
        "dlnrho_dlnP_constT": raw[:, 6].reshape(ntemp, npress),
        "dlnS_dlnT_constP": raw[:, 7].reshape(ntemp, npress),
        "dlnS_dlnP_constT": raw[:, 8].reshape(ntemp, npress),
        "grad_ad": raw[:, 9].reshape(ntemp, npress),
    }
    return Chabrier2021Table(
        helium_fraction=helium_fraction,
        log10_temp=log10_temp,
        log10_press_gpa=log10_press_gpa,
        fields=fields,
    )


@lru_cache(maxsize=4)
def _load_chabrier2021_tables(data_dir: str | None) -> dict[float, Chabrier2021Table]:
    root = ensure_chabrier2021_tables(data_dir)
    out: dict[float, Chabrier2021Table] = {}
    for helium_fraction, filename in CHABRIER2021_FILES.items():
        out[helium_fraction] = _load_one_table(root / filename, helium_fraction)
    return out


@lru_cache(maxsize=4)
def _load_chabrier2021_pure_tables(data_dir: str | None) -> dict[float, Chabrier2021Table]:
    root = ensure_chabrier2021_tables(data_dir, pure=True)
    out: dict[float, Chabrier2021Table] = {}
    for helium_fraction, filename in CHABRIER2021_PURE_FILES.items():
        out[helium_fraction] = _load_one_table(root / filename, helium_fraction)
    table_h = out[0.0]
    table_he = out[1.0]
    if (
        table_h.log10_temp.shape != table_he.log10_temp.shape
        or table_h.log10_press_gpa.shape != table_he.log10_press_gpa.shape
        or not np.allclose(table_h.log10_temp, table_he.log10_temp, rtol=0.0, atol=1.0e-9)
        or not np.allclose(table_h.log10_press_gpa, table_he.log10_press_gpa, rtol=0.0, atol=1.0e-9)
    ):
        raise RuntimeError("Chabrier pure H and pure He tables must share one (T, P) grid.")
    return out


def additive_volume_mix_rows(
    row_h: dict[str, np.ndarray],
    row_he: dict[str, np.ndarray],
    x_h: float,
    y_he: float,
) -> dict[str, np.ndarray]:
    """Mix pure-H and pure-He isotherm rows at fixed (T, P) with the additive volume law.

    At fixed temperature and pressure (Chabrier & Debras 2021, eqs. 1-3):
        1/rho = X/rho_H + Y/rho_He,      u = X u_H + Y u_He,      s = X s_H + Y s_He.
    Differentiating the specific volume gives the mixture's density derivatives,
        dlnrho/dlnT|_P = rho * [X/rho_H dlnrho_H/dlnT|_P + Y/rho_He dlnrho_He/dlnT|_P]
    (and the same form for dlnrho/dlnP|_T), and the additive entropy makes c_P additive,
    c_P = X c_P,H + Y c_P,He with c_P,i = P delta_i / (rho_i T grad_ad,i), so that
        grad_ad = P delta / (rho T c_P) = delta / (rho * sum_i w_i delta_i/(rho_i grad_ad,i)).
    The ideal entropy of mixing is omitted: it is a composition-dependent constant wherever
    the ionisation state is fixed, which is the whole fully ionised interior this block
    serves, and the entropy is a diagnostic field only (it is not a runtime table field).
    """
    weights = ((x_h, row_h), (y_he, row_he))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        inv_rho = np.zeros_like(row_h["log10_rho"])
        u = np.zeros_like(inv_rho)
        s = np.zeros_like(inv_rho)
        dlnrho_dlnT_num = np.zeros_like(inv_rho)
        dlnrho_dlnP_num = np.zeros_like(inv_rho)
        cp_over_p_t = np.zeros_like(inv_rho)
        for w, row in weights:
            if w == 0.0:
                continue
            rho_i = np.power(10.0, row["log10_rho"])
            inv_rho += w / rho_i
            u += w * np.power(10.0, row["log10_u_mj_per_kg"])
            s += w * np.power(10.0, row["log10_s_mj_per_kg_k"])
            dlnrho_dlnT_num += w / rho_i * row["dlnrho_dlnT_constP"]
            dlnrho_dlnP_num += w / rho_i * row["dlnrho_dlnP_constT"]
            # c_P,i / (P T) = delta_i / (rho_i grad_ad,i), delta_i = -dlnrho_i/dlnT|_P
            cp_over_p_t += w * (-row["dlnrho_dlnT_constP"]) / (rho_i * row["grad_ad"])
        rho = 1.0 / inv_rho
        dlnrho_dlnT = rho * dlnrho_dlnT_num
        dlnrho_dlnP = rho * dlnrho_dlnP_num
        delta = -dlnrho_dlnT
        grad_ad = delta / (rho * cp_over_p_t)
        out = {
            "log10_rho": np.log10(rho),
            "log10_press_gpa": row_h["log10_press_gpa"].copy(),
            "log10_u_mj_per_kg": np.log10(u),
            "log10_s_mj_per_kg_k": np.log10(s),
            "dlnrho_dlnT_constP": dlnrho_dlnT,
            "dlnrho_dlnP_constT": dlnrho_dlnP,
            "grad_ad": grad_ad,
        }
    # A cell is only as valid as both of its components: NaN propagates through the sums,
    # and _resample_isotherm_to_logrho drops non-finite and unphysical cells afterwards.
    return out


def _helium_fraction_bracket(target_y: float, available: np.ndarray) -> tuple[float, float, float, bool]:
    if available.size < 2:
        raise RuntimeError("Need at least two Chabrier mixture tables for Y interpolation.")

    extrapolated = False
    if target_y <= available[0]:
        i0 = 0
        i1 = 1
        extrapolated = target_y < available[0]
    elif target_y >= available[-1]:
        i0 = available.size - 2
        i1 = available.size - 1
        extrapolated = target_y > available[-1]
    else:
        i1 = int(np.searchsorted(available, target_y))
        i0 = i1 - 1
    y0 = float(available[i0])
    y1 = float(available[i1])
    weight = 0.0 if math.isclose(y0, y1) else (target_y - y0) / (y1 - y0)
    return y0, y1, float(weight), extrapolated


def _interp_tp_rows(
    table: Chabrier2021Table,
    target_log10_temp: float,
) -> dict[str, np.ndarray]:
    if target_log10_temp < table.log10_temp[0] or target_log10_temp > table.log10_temp[-1]:
        return {
            "log10_press_gpa": table.log10_press_gpa.copy(),
            **{name: np.full_like(table.log10_press_gpa, np.nan, dtype=np.float64) for name in table.fields},
        }

    if np.isclose(target_log10_temp, table.log10_temp[0]):
        i0 = i1 = 0
        w = 0.0
    elif np.isclose(target_log10_temp, table.log10_temp[-1]):
        i0 = i1 = table.log10_temp.size - 1
        w = 0.0
    else:
        i1 = int(np.searchsorted(table.log10_temp, target_log10_temp))
        i0 = i1 - 1
        t0 = table.log10_temp[i0]
        t1 = table.log10_temp[i1]
        w = 0.0 if i0 == i1 else (target_log10_temp - t0) / (t1 - t0)

    out = {"log10_press_gpa": table.log10_press_gpa.copy()}
    for name, arr in table.fields.items():
        if i0 == i1:
            out[name] = arr[i0].copy()
        else:
            out[name] = (1.0 - w) * arr[i0] + w * arr[i1]
    return out


def _strictly_increasing_mask(values: np.ndarray) -> np.ndarray:
    keep = np.zeros(values.size, dtype=bool)
    last = -np.inf
    min_step = 1.0e-10
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if value > last + min_step:
            keep[i] = True
            last = value
    return keep


def _resample_isotherm_to_logrho(
    target_log10_rho: np.ndarray,
    row: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    field_names = (
        "log10_press_gpa",
        "log10_u_mj_per_kg",
        "log10_s_mj_per_kg_k",
        "dlnrho_dlnT_constP",
        "dlnrho_dlnP_constT",
        "grad_ad",
    )
    out = {name: np.full_like(target_log10_rho, np.nan, dtype=np.float64) for name in field_names}

    finite = np.isfinite(row["log10_rho"])
    for name in field_names:
        finite &= np.isfinite(row[name])
    finite &= row["log10_u_mj_per_kg"] < 20.0
    finite &= row["log10_s_mj_per_kg_k"] < 20.0
    finite &= row["dlnrho_dlnP_constT"] > 0.0
    finite &= row["dlnrho_dlnP_constT"] < 20.0
    finite &= np.abs(row["dlnrho_dlnT_constP"]) < 100.0
    finite &= row["grad_ad"] > 0.0
    if np.count_nonzero(finite) < 2:
        return out

    x = row["log10_rho"][finite]
    keep = _strictly_increasing_mask(x)
    if np.count_nonzero(keep) < 2:
        return out
    x = x[keep]

    valid_target = (target_log10_rho >= x[0]) & (target_log10_rho <= x[-1])
    if not np.any(valid_target):
        return out

    for name in field_names:
        y = row[name][finite][keep]
        out[name][valid_target] = np.interp(target_log10_rho[valid_target], x, y)
    return out


def chabrier2021_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    y_he: float,
    radiation_enabled: bool,
    data_dir: str | None = None,
    mixing: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if mixing not in CHABRIER2021_MIXING_MODES:
        raise RuntimeError(
            f"Unknown Chabrier mixing mode '{mixing}'; expected one of "
            f"{CHABRIER2021_MIXING_MODES}."
        )
    tables = _load_chabrier2021_tables(data_dir)
    available = np.asarray(sorted(tables.keys()), dtype=np.float64)
    if mixing == "auto":
        tol = CHABRIER2021_MIXTURE_INTERP_TOLERANCE
        mixing = (
            "mixture_interp"
            if (available[0] - tol) <= y_he <= (available[-1] + tol)
            else "additive_volume"
        )
    if mixing == "mixture_interp":
        y0, y1, weight, extrapolated = _helium_fraction_bracket(y_he, available)
        table0 = tables[y0]
        table1 = tables[y1]
    else:
        pure = _load_chabrier2021_pure_tables(data_dir)
        table0 = pure[0.0]
        table1 = pure[1.0]
        y0, y1, weight, extrapolated = 0.0, 1.0, float(y_he), False
    x_h_mix = 1.0 - float(y_he)

    nrho = logrho.size
    ntemp = logtemp.size
    target_log10_rho = logrho / LN10
    target_log10_temp = logtemp / LN10
    rho = np.exp(logrho)[:, None]
    temp = np.exp(logtemp)[None, :]

    log10_press_gpa = np.full((nrho, ntemp), np.nan, dtype=np.float64)
    log10_u_mj_per_kg = np.full_like(log10_press_gpa, np.nan)
    log10_s_mj_per_kg_k = np.full_like(log10_press_gpa, np.nan)
    dlnrho_dlnT_constP = np.full_like(log10_press_gpa, np.nan)
    dlnrho_dlnP_constT = np.full_like(log10_press_gpa, np.nan)
    grad_ad_gas = np.full_like(log10_press_gpa, np.nan)

    for it, logt in enumerate(target_log10_temp):
        row0 = _interp_tp_rows(table0, float(logt))
        row1 = _interp_tp_rows(table1, float(logt))
        if mixing == "additive_volume":
            mixed_row = additive_volume_mix_rows(row0, row1, x_h_mix, float(y_he))
        else:
            mixed_row = {
                "log10_rho": (1.0 - weight) * row0["log10_rho"] + weight * row1["log10_rho"],
                "log10_press_gpa": row0["log10_press_gpa"],
                "log10_u_mj_per_kg": (1.0 - weight) * row0["log10_u_mj_per_kg"] + weight * row1["log10_u_mj_per_kg"],
                "log10_s_mj_per_kg_k": (1.0 - weight) * row0["log10_s_mj_per_kg_k"] + weight * row1["log10_s_mj_per_kg_k"],
                "dlnrho_dlnT_constP": (1.0 - weight) * row0["dlnrho_dlnT_constP"] + weight * row1["dlnrho_dlnT_constP"],
                "dlnrho_dlnP_constT": (1.0 - weight) * row0["dlnrho_dlnP_constT"] + weight * row1["dlnrho_dlnP_constT"],
                "grad_ad": (1.0 - weight) * row0["grad_ad"] + weight * row1["grad_ad"],
            }
        sampled = _resample_isotherm_to_logrho(target_log10_rho, mixed_row)
        log10_press_gpa[:, it] = sampled["log10_press_gpa"]
        log10_u_mj_per_kg[:, it] = sampled["log10_u_mj_per_kg"]
        log10_s_mj_per_kg_k[:, it] = sampled["log10_s_mj_per_kg_k"]
        dlnrho_dlnT_constP[:, it] = sampled["dlnrho_dlnT_constP"]
        dlnrho_dlnP_constT[:, it] = sampled["dlnrho_dlnP_constT"]
        grad_ad_gas[:, it] = sampled["grad_ad"]

    pressure_gas = np.power(10.0, log10_press_gpa + 10.0)
    eps_gas = np.power(10.0, log10_u_mj_per_kg) * 1.0e10
    entropy_gas = np.power(10.0, log10_s_mj_per_kg_k) * 1.0e10

    chi_rho_gas = 1.0 / dlnrho_dlnP_constT
    chi_t_gas = -dlnrho_dlnT_constP / dlnrho_dlnP_constT
    delta_gas = -dlnrho_dlnT_constP
    cp_gas = pressure_gas * delta_gas / np.maximum(rho * temp * grad_ad_gas, 1.0e-300)
    cv_gas = cp_gas - pressure_gas * chi_t_gas * chi_t_gas / np.maximum(rho * temp * chi_rho_gas, 1.0e-300)

    if radiation_enabled:
        prad = (A_RAD / 3.0) * temp**4
        eps_rad = A_RAD * temp**4 / rho
        entropy_rad = 4.0 * A_RAD * temp**3 / (3.0 * rho)
        cv_rad = 4.0 * A_RAD * temp**3 / rho
    else:
        prad = np.zeros_like(pressure_gas)
        eps_rad = np.zeros_like(pressure_gas)
        entropy_rad = np.zeros_like(pressure_gas)
        cv_rad = np.zeros_like(pressure_gas)

    pressure_total = pressure_gas + prad
    eps_total = eps_gas + eps_rad
    entropy_total = entropy_gas + entropy_rad
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        chi_rho = np.where(pressure_total > 0.0, pressure_gas * chi_rho_gas / pressure_total, np.nan)
        chi_t = np.where(
            pressure_total > 0.0,
            (pressure_gas * chi_t_gas + 4.0 * prad) / pressure_total,
            np.nan,
        )
        cv = cv_gas + cv_rad
        gamma3m1 = pressure_total * chi_t / np.maximum(rho * temp * cv, 1.0e-300)
        gamma1 = chi_rho + chi_t * gamma3m1
        cs2 = gamma1 * pressure_total / np.maximum(rho, 1.0e-300)
        cp = cv * gamma1 / np.maximum(chi_rho, 1.0e-12)
        grad_ad = gamma3m1 / np.maximum(gamma1, 1.0e-300)
        beta_rad = np.where(pressure_total > 0.0, prad / pressure_total, 0.0)

    valid = (
        np.isfinite(pressure_total)
        & np.isfinite(eps_total)
        & np.isfinite(entropy_total)
        & np.isfinite(chi_rho)
        & np.isfinite(chi_t)
        & np.isfinite(cv)
        & np.isfinite(cp)
        & np.isfinite(gamma1)
        & np.isfinite(gamma3m1)
        & np.isfinite(cs2)
        & np.isfinite(grad_ad)
        & np.isfinite(grad_ad_gas)
        & (pressure_gas > 0.0)
        & (pressure_total > 0.0)
        & (eps_gas > 0.0)
        & (eps_total > 0.0)
        & (entropy_gas > 0.0)
        & (chi_rho_gas > 0.0)
        & (chi_rho > 0.0)
        & (chi_t > 0.0)
        & (cv_gas > 0.0)
        & (cv > 0.0)
        & (cp_gas > 0.0)
        & (cp > 0.0)
        & (gamma1 > 0.0)
        & (gamma1 < 10.0)
        & (gamma3m1 > 0.0)
        & (gamma3m1 < 10.0)
        & (cs2 > 0.0)
        & (cs2 < 0.9 * C_LIGHT * C_LIGHT)
        & (grad_ad_gas > 0.0)
        & (grad_ad > 0.0)
    )

    fields = {
        "pressure_gas": np.where(valid, pressure_gas, np.nan),
        "eps_gas": np.where(valid, eps_gas, np.nan),
        "entropy_gas": np.where(valid, entropy_gas, np.nan),
        "chi_rho": np.where(valid, chi_rho, np.nan),
        "chi_t": np.where(valid, chi_t, np.nan),
        "cv": np.where(valid, cv, np.nan),
        "cp": np.where(valid, cp, np.nan),
        "gamma1": np.where(valid, gamma1, np.nan),
        "gamma3m1": np.where(valid, gamma3m1, np.nan),
        "cs2": np.where(valid, cs2, np.nan),
        "pressure_total": np.where(valid, pressure_total, np.nan),
        "eps_total": np.where(valid, eps_total, np.nan),
        "entropy_total": np.where(valid, entropy_total, np.nan),
        "pgas": np.where(valid, pressure_gas, np.nan),
        "prad": np.where(valid, prad, np.nan),
        "beta_rad": np.where(valid, beta_rad, np.nan),
        "grad_ad": np.where(valid, grad_ad, np.nan),
        "grad_ad_gas": np.where(valid, grad_ad_gas, np.nan),
        "chabrier_valid": valid.astype(np.float64),
    }
    context = {
        "source_url_2019": CHABRIER2021_SOURCE_URL_2019,
        "source_url_2021": CHABRIER2021_SOURCE_URL_2021,
        "data_dir": str(ensure_chabrier2021_tables(data_dir)),
        "available_helium_mass_fractions": tuple(float(y) for y in available),
        "target_helium_mass_fraction": float(y_he),
        "mixing_mode": mixing,
        "interpolation_helium_lo": y0,
        "interpolation_helium_hi": y1,
        "interpolation_weight_hi": weight,
        "interpolation_extrapolated": extrapolated,
        "log10_temp_min": CHABRIER2021_LOG10_TEMP_MIN,
        "log10_temp_max": CHABRIER2021_LOG10_TEMP_MAX,
        "log10_press_gpa_min": CHABRIER2021_LOG10_PRESS_GPA_MIN,
        "log10_press_gpa_max": CHABRIER2021_LOG10_PRESS_GPA_MAX,
        "rho_blend_cgs_min": CHABRIER2021_T13_BLEND_RHO_CGS_MIN,
        "rho_blend_cgs_max": CHABRIER2021_T13_BLEND_RHO_CGS_MAX,
        "rho_switch_cgs": CHABRIER2021_T13_SWITCH_RHO_CGS,
        "valid_cell_count": int(np.count_nonzero(valid)),
    }
    return fields, context
