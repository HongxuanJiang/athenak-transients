#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


K_B = 1.380649e-16
A_RAD = 7.5657e-15
M_H = 1.6735575e-24
M_HE = 6.646442e-24
LN10 = math.log(10.0)
AVO = 6.02214076e23

SCVH95_BASE_URL = "https://aas.org/sites/default/files/cdrom/volume5/volume5/apjs/v99/p713"
SCVH95_FILES = (
    "readme.doc",
    "table.doc",
    "read.f",
    "rho_crit.dat",
    "h_tab_i.dat",
    "h_tab_p1.dat",
    "h_tab_p2.dat",
    "he_tab_i.dat",
)
SCVH95_LOGT_MIN = 2.10
SCVH95_LOGT_MAX = 7.06
SCVH95_LOGP_MIN = 4.00
SCVH95_LOGP_MAX = 19.00
SCVH95_PPT_LOGT_MIN = 3.54
SCVH95_PPT_LOGT_MAX = 4.82
SCVH95_PPT_LOGP_MIN = 10.50
SCVH95_PPT_LOGP_MAX = 14.10
SCVH95_CRIT_LOGT = 4.18
SCVH95_CRIT_LOGP = 11.75
SCVH95_INTERP_LOGP_STEP = 0.05
SCVH95_INTERP_PPT_LOGP_STEP = 0.02


@dataclass(frozen=True)
class PaperTable:
    logtemp: np.ndarray
    logpress: np.ndarray
    fields: dict[str, np.ndarray]
    valid: np.ndarray


def _safe_log(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, -np.inf, dtype=np.float64)
    mask = values > 0.0
    out[mask] = np.log(values[mask])
    return out


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "scvh95_data"


def ensure_scvh95_tables(data_dir: str | None = None) -> Path:
    root = _default_data_dir() if data_dir is None else Path(data_dir).expanduser().resolve()
    missing = [name for name in SCVH95_FILES if not (root / name).exists()]
    if not missing:
        return root
    try:
        import cloudscraper  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SCvH 1995 tables are missing and cloudscraper is not installed. "
            "Install it with `python3 -m pip install --user --break-system-packages cloudscraper` "
            "or stage the original paper files in "
            f"{root}."
        ) from exc
    root.mkdir(parents=True, exist_ok=True)
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "linux", "mobile": False})
    for name in missing:
        url = f"{SCVH95_BASE_URL}/{name}"
        response = scraper.get(url, timeout=120)
        if response.status_code != 200 or "text/html" in (response.headers.get("content-type") or ""):
            raise RuntimeError(f"Failed to fetch SCvH file from {url}: HTTP {response.status_code}")
        (root / name).write_bytes(response.content)
    return root


def _parse_irregular_table(path: Path) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[np.ndarray]]]:
    logtemps: list[float] = []
    logpress_rows: list[np.ndarray] = []
    field_rows = {
        "xh2_or_he": [],
        "xh_or_he1": [],
        "logrho": [],
        "logs": [],
        "logu": [],
        "dlnrho_dlnT_constP": [],
        "dlnrho_dlnP_constT": [],
        "dlnS_dlnT_constP": [],
        "dlnS_dlnP_constT": [],
        "grad_ad": [],
    }
    with path.open("r", encoding="ascii") as fp:
        while True:
            header = fp.readline()
            if not header:
                break
            parts = header.split()
            if not parts:
                continue
            logt = float(parts[0])
            npress = int(parts[1])
            row = np.array([list(map(float, fp.readline().split())) for _ in range(npress)], dtype=np.float64)
            logtemps.append(logt)
            logpress_rows.append(row[:, 0].copy())
            field_rows["xh2_or_he"].append(row[:, 1].copy())
            field_rows["xh_or_he1"].append(row[:, 2].copy())
            field_rows["logrho"].append(row[:, 3].copy())
            field_rows["logs"].append(row[:, 4].copy())
            field_rows["logu"].append(row[:, 5].copy())
            field_rows["dlnrho_dlnT_constP"].append(row[:, 6].copy())
            field_rows["dlnrho_dlnP_constT"].append(row[:, 7].copy())
            field_rows["dlnS_dlnT_constP"].append(row[:, 8].copy())
            field_rows["dlnS_dlnP_constT"].append(row[:, 9].copy())
            field_rows["grad_ad"].append(row[:, 10].copy())
    return np.asarray(logtemps, dtype=np.float64), logpress_rows, field_rows


def _regularize_table(
    logtemps: np.ndarray,
    logpress_rows: list[np.ndarray],
    field_rows: dict[str, list[np.ndarray]],
) -> PaperTable:
    logpress = np.unique(np.concatenate(logpress_rows))
    ntemp = logtemps.size
    npress = logpress.size
    fields = {name: np.full((ntemp, npress), np.nan, dtype=np.float64) for name in field_rows}
    valid = np.zeros((ntemp, npress), dtype=bool)
    for i, row_press in enumerate(logpress_rows):
        idx = np.searchsorted(logpress, row_press)
        valid[i, idx] = True
        for name, rows in field_rows.items():
            fields[name][i, idx] = rows[i]
    return PaperTable(logtemp=logtemps, logpress=logpress, fields=fields, valid=valid)


@lru_cache(maxsize=4)
def _load_tables(data_dir: str | None) -> dict[str, object]:
    root = ensure_scvh95_tables(data_dir)
    h_i = _regularize_table(*_parse_irregular_table(root / "h_tab_i.dat"))
    h_p1 = _regularize_table(*_parse_irregular_table(root / "h_tab_p1.dat"))
    h_p2 = _regularize_table(*_parse_irregular_table(root / "h_tab_p2.dat"))
    he_i = _regularize_table(*_parse_irregular_table(root / "he_tab_i.dat"))
    rho_crit = np.loadtxt(root / "rho_crit.dat", dtype=np.float64)
    return {
        "root": str(root),
        "h_i": h_i,
        "h_p1": h_p1,
        "h_p2": h_p2,
        "he_i": he_i,
        "rho_crit": rho_crit,
    }


def _interp_row(logpress_row: np.ndarray, values_row: np.ndarray, logpress_target: np.ndarray) -> np.ndarray:
    good = np.isfinite(values_row)
    if np.count_nonzero(good) < 2:
        return np.full_like(logpress_target, np.nan, dtype=np.float64)
    interp = PchipInterpolator(logpress_row[good], values_row[good], extrapolate=False)
    return np.asarray(interp(logpress_target), dtype=np.float64)


def _interp_isotherm(table: PaperTable, target_logt: float, logpress_target: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: np.full_like(logpress_target, np.nan, dtype=np.float64) for name in table.fields}
    if target_logt < table.logtemp[0] or target_logt > table.logtemp[-1]:
        return out
    if np.isclose(target_logt, table.logtemp[0]):
        i0 = i1 = 0
        w = 0.0
    elif np.isclose(target_logt, table.logtemp[-1]):
        i0 = i1 = table.logtemp.size - 1
        w = 0.0
    else:
        i1 = int(np.searchsorted(table.logtemp, target_logt))
        i0 = max(i1 - 1, 0)
        if i1 >= table.logtemp.size:
            i1 = table.logtemp.size - 1
            i0 = i1
        t0 = table.logtemp[i0]
        t1 = table.logtemp[i1]
        w = 0.0 if i0 == i1 else (target_logt - t0) / (t1 - t0)
    for name, arr in table.fields.items():
        row0 = _interp_row(table.logpress, arr[i0], logpress_target)
        if i0 == i1:
            out[name] = row0
        else:
            row1 = _interp_row(table.logpress, arr[i1], logpress_target)
            both = np.isfinite(row0) & np.isfinite(row1)
            vals = np.full_like(logpress_target, np.nan, dtype=np.float64)
            vals[both] = (1.0 - w) * row0[both] + w * row1[both]
            out[name] = vals
    return out


def _ppt_transition_pressure(logtemp_value: float, rho_crit: np.ndarray) -> float:
    if logtemp_value >= SCVH95_CRIT_LOGT:
        return SCVH95_CRIT_LOGP
    interp = PchipInterpolator(rho_crit[:, 0], rho_crit[:, 1], extrapolate=False)
    val = float(interp(logtemp_value))
    if not np.isfinite(val):
        raise RuntimeError(f"SCvH PPT transition pressure is undefined at log10(T)={logtemp_value}")
    return val


def _hydrogen_isotherm(
    tables: dict[str, object],
    target_logt: float,
    logpress_target: np.ndarray,
) -> dict[str, np.ndarray]:
    base = _interp_isotherm(tables["h_i"], target_logt, logpress_target)  # type: ignore[arg-type]
    if SCVH95_PPT_LOGT_MIN <= target_logt < SCVH95_PPT_LOGT_MAX:
        ptrans = _ppt_transition_pressure(target_logt, tables["rho_crit"])  # type: ignore[arg-type]
        phase1 = _interp_isotherm(tables["h_p1"], target_logt, logpress_target)  # type: ignore[arg-type]
        phase2 = _interp_isotherm(tables["h_p2"], target_logt, logpress_target)  # type: ignore[arg-type]
        use_p1 = (
            (logpress_target >= SCVH95_PPT_LOGP_MIN)
            & (logpress_target < ptrans)
            & np.isfinite(phase1["logrho"])
        )
        use_p2 = (
            (logpress_target >= ptrans)
            & (logpress_target <= SCVH95_PPT_LOGP_MAX)
            & np.isfinite(phase2["logrho"])
        )
        for name in base:
            if np.any(use_p1):
                base[name][use_p1] = phase1[name][use_p1]
            if np.any(use_p2):
                base[name][use_p2] = phase2[name][use_p2]
    return base


def _scvh_entropy_of_mixing(
    x_h: float,
    y_he: float,
    temp: np.ndarray,
    press: np.ndarray,
    xh: np.ndarray,
    dxh_dlogt: np.ndarray,
    dxh_dlogp: np.ndarray,
    xh2: np.ndarray,
    dxh2_dlogt: np.ndarray,
    dxh2_dlogp: np.ndarray,
    xhe0: np.ndarray,
    dxhe0_dlogt: np.ndarray,
    dxhe0_dlogp: np.ndarray,
    xhe1: np.ndarray,
    dxhe1_dlogt: np.ndarray,
    dxhe1_dlogp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    small = 1.0e-8
    tiny = 1.0e-14
    xh = np.where(xh > small, xh, 0.0)
    xh2 = np.where(xh2 > small, xh2, 0.0)
    xhe0 = np.where(xhe0 > small, xhe0, 0.0)
    xhe1 = np.where(xhe1 > small, xhe1, 0.0)
    dxh_dP = np.where(xh > 0.0, dxh_dlogp / (press * LN10), 0.0)
    dxh_dT = np.where(xh > 0.0, dxh_dlogt / (temp * LN10), 0.0)
    dxh2_dP = np.where(xh2 > 0.0, dxh2_dlogp / (press * LN10), 0.0)
    dxh2_dT = np.where(xh2 > 0.0, dxh2_dlogt / (temp * LN10), 0.0)
    dxhe0_dP = np.where(xhe0 > 0.0, dxhe0_dlogp / (press * LN10), 0.0)
    dxhe0_dT = np.where(xhe0 > 0.0, dxhe0_dlogt / (temp * LN10), 0.0)
    dxhe1_dP = np.where(xhe1 > 0.0, dxhe1_dlogp / (press * LN10), 0.0)
    dxhe1_dT = np.where(xhe1 > 0.0, dxhe1_dlogt / (temp * LN10), 0.0)

    beta = (M_H * (y_he + tiny)) / (M_HE * (x_h + tiny))

    num = 1.5 * (1.0 + xh + 3.0 * xh2)
    dnum_dP = 1.5 * (dxh_dP + 3.0 * dxh2_dP)
    dnum_dT = 1.5 * (dxh_dT + 3.0 * dxh2_dT)
    denom = 1.0 + 2.0 * xhe0 + xhe1
    ddenom_dP = 2.0 * dxhe0_dP + dxhe1_dP
    ddenom_dT = 2.0 * dxhe0_dT + dxhe1_dT
    gamma = num / denom
    dgamma_dP = dnum_dP / denom - ddenom_dP * gamma / denom
    dgamma_dT = dnum_dT / denom - ddenom_dT * gamma / denom

    num = 1.5 * beta * gamma * (2.0 - 2.0 * xhe0 - xhe1)
    dnum_dP = 1.5 * beta * (dgamma_dP * (2.0 - 2.0 * xhe0 - xhe1) + gamma * (-2.0 * dxhe0_dP - dxhe1_dP))
    dnum_dT = 1.5 * beta * (dgamma_dT * (2.0 - 2.0 * xhe0 - xhe1) + gamma * (-2.0 * dxhe0_dT - dxhe1_dT))

    denom = 1.0 - xh2 - xh
    bad = denom <= 1.0e-5
    denom_eff = np.where(bad, 1.0e-5, denom)
    ddenom_dP = np.where(bad, 0.0, -dxh2_dP - dxh_dP)
    ddenom_dT = np.where(bad, 0.0, -dxh2_dT - dxh_dT)

    delta = num / denom_eff
    tiny_mask = delta <= tiny
    delta = np.where(tiny_mask, tiny, delta)
    ddelta_dP = np.where(tiny_mask, 0.0, dnum_dP / denom_eff - ddenom_dP * delta / denom_eff)
    ddelta_dT = np.where(tiny_mask, 0.0, dnum_dT / denom_eff - ddenom_dT * delta / denom_eff)

    a = x_h / M_H * (2.0 / (1.0 + xh + 3.0 * xh2))
    da_dP = -a * (dxh_dP + 3.0 * dxh2_dP) / (1.0 + xh + 3.0 * xh2)
    da_dT = -a * (dxh_dT + 3.0 * dxh2_dT) / (1.0 + xh + 3.0 * xh2)

    b1 = np.log(1.0 + beta * gamma)
    db1_dT = beta * dgamma_dT / (1.0 + beta * gamma)
    db1_dP = beta * dgamma_dP / (1.0 + beta * gamma)

    b21 = 0.5 * (1.0 - xh2 - xh)
    db21_dT = 0.5 * (-dxh2_dT - dxh_dT)
    db21_dP = 0.5 * (-dxh2_dP - dxh_dP)
    b22 = np.log(1.0 + delta)
    db22_dT = ddelta_dT / (1.0 + delta)
    db22_dP = ddelta_dP / (1.0 + delta)
    b2 = b21 * b22
    db2_dT = db21_dT * b22 + b21 * db22_dT
    db2_dP = db21_dP * b22 + b21 * db22_dP

    b31 = beta * gamma
    db31_dT = beta * dgamma_dT
    db31_dP = beta * dgamma_dP
    b32 = np.log(1.0 + 1.0 / (beta * gamma))
    db32_dT = -dgamma_dT / (gamma * (1.0 + beta * gamma))
    db32_dP = -dgamma_dP / (gamma * (1.0 + beta * gamma))
    b331 = np.log(1.0 + 1.0 / delta)
    db331_dT = -ddelta_dT / (delta * (1.0 + delta))
    db331_dP = -ddelta_dP / (delta * (1.0 + delta))
    b332 = (2.0 - 2.0 * xhe0 - xhe1) / 3.0
    db332_dT = (-2.0 * dxhe0_dT - dxhe1_dT) / 3.0
    db332_dP = (-2.0 * dxhe0_dP - dxhe1_dP) / 3.0
    b33 = b331 * b332
    db33_dT = db331_dT * b332 + b331 * db332_dT
    db33_dP = db331_dP * b332 + b331 * db332_dP
    b3 = b31 * (b32 - b33)
    db3_dT = db31_dT * (b32 - b33) + b31 * (db32_dT - db33_dT)
    db3_dP = db31_dP * (b32 - b33) + b31 * (db32_dP - db33_dP)

    b = b1 - b2 + b3
    db_dT = db1_dT - db2_dT + db3_dT
    db_dP = db1_dP - db2_dP + db3_dP
    smix = K_B * a * b
    d_smix_dT = K_B * (a * db_dT + da_dT * b)
    d_smix_dP = K_B * (a * db_dP + da_dP * b)
    return smix, d_smix_dT, d_smix_dP


def _mix_scvh_isotherm(
    h_fields: dict[str, np.ndarray],
    he_fields: dict[str, np.ndarray],
    logpress: np.ndarray,
    logtemp_value: float,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
) -> dict[str, np.ndarray]:
    press_gas = 10.0 ** logpress
    temp = np.full_like(logpress, 10.0 ** logtemp_value, dtype=np.float64)

    xh2 = np.clip(h_fields["xh2_or_he"], 0.0, 1.0)
    xh = np.clip(h_fields["xh_or_he1"], 0.0, 1.0)
    xion = np.clip(1.0 - xh2 - xh, 0.0, 1.0)
    xhe0 = np.clip(he_fields["xh2_or_he"], 0.0, 1.0)
    xhe1 = np.clip(he_fields["xh_or_he1"], 0.0, 1.0)
    xhe2 = np.clip(1.0 - xhe0 - xhe1, 0.0, 1.0)

    den_h = 10.0 ** h_fields["logrho"]
    den_he = 10.0 ** he_fields["logrho"]
    entr_h = 10.0 ** h_fields["logs"]
    entr_he = 10.0 ** he_fields["logs"]
    ener_h = 10.0 ** h_fields["logu"]
    ener_he = 10.0 ** he_fields["logu"]

    rho = 1.0 / (x_h / den_h + y_he / den_he)
    return {
        "press_gas": press_gas,
        "temp": temp,
        "xh2": xh2,
        "xh": xh,
        "xion": xion,
        "xhe0": xhe0,
        "xhe1": xhe1,
        "xhe2": xhe2,
        "rho_h": den_h,
        "rho_he": den_he,
        "entropy_h": entr_h,
        "entropy_he": entr_he,
        "energy_h": ener_h,
        "energy_he": ener_he,
        "rho": rho,
    }


def _finite_diff_1d(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    deriv = np.empty_like(values)
    deriv[1:-1] = (values[2:] - values[:-2]) / (coords[2:] - coords[:-2])
    deriv[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (coords[2] - coords[0])
    deriv[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (coords[-1] - coords[-3])
    return deriv


def _masked_finite_diff_1d(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    deriv = np.full_like(values, np.nan, dtype=np.float64)
    npts = values.size
    for i in range(npts):
        if not np.isfinite(values[i]):
            continue
        if i >= 1 and i + 1 < npts and np.isfinite(values[i - 1]) and np.isfinite(values[i + 1]):
            deriv[i] = (values[i + 1] - values[i - 1]) / (coords[i + 1] - coords[i - 1])
            continue
        if i == 0 and i + 2 < npts and np.isfinite(values[i + 1]) and np.isfinite(values[i + 2]):
            deriv[i] = (-3.0 * values[i] + 4.0 * values[i + 1] - values[i + 2]) / (coords[i + 2] - coords[i])
            continue
        if i == npts - 1 and i >= 2 and np.isfinite(values[i - 1]) and np.isfinite(values[i - 2]):
            deriv[i] = (3.0 * values[i] - 4.0 * values[i - 1] + values[i - 2]) / (coords[i] - coords[i - 2])
    return deriv


def _masked_finite_difference(values: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    deriv = np.full_like(values, np.nan, dtype=np.float64)
    if axis == 0:
        for j in range(values.shape[1]):
            deriv[:, j] = _masked_finite_diff_1d(values[:, j], coords)
    elif axis == 1:
        for i in range(values.shape[0]):
            deriv[i, :] = _masked_finite_diff_1d(values[i, :], coords)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported axis: {axis}")
    return deriv


def _compose_and_mix(
    tables: dict[str, object],
    target_logt: float,
    logpress_target: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
) -> dict[str, np.ndarray]:
    h = _hydrogen_isotherm(tables, target_logt, logpress_target)
    he = _interp_isotherm(tables["he_i"], target_logt, logpress_target)  # type: ignore[arg-type]
    valid = np.isfinite(h["logrho"]) & np.isfinite(he["logrho"])
    out = {name: np.full_like(logpress_target, np.nan, dtype=np.float64) for name in (
        "logrho",
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
        "logne",
    )}
    if np.count_nonzero(valid) < 4:
        return out

    logp = logpress_target[valid]
    h_valid = {name: values[valid] for name, values in h.items()}
    he_valid = {name: values[valid] for name, values in he.items()}
    mix = _mix_scvh_isotherm(h_valid, he_valid, logp, target_logt, x_h, y_he, radiation_enabled)
    xh2 = mix["xh2"]
    xh = mix["xh"]
    xion = mix["xion"]
    xhe0 = mix["xhe0"]
    xhe1 = mix["xhe1"]
    xhe2 = mix["xhe2"]
    press_gas = mix["press_gas"]
    temp = mix["temp"]
    rho = mix["rho"]
    entr_h = mix["entropy_h"]
    entr_he = mix["entropy_he"]
    ener_h = mix["energy_h"]
    ener_he = mix["energy_he"]
    den_h = mix["rho_h"]
    den_he = mix["rho_he"]

    dxh2_dlogP = _finite_diff_1d(xh2, logp)
    dxh_dlogP = _finite_diff_1d(xh, logp)
    dxhe0_dlogP = _finite_diff_1d(xhe0, logp)
    dxhe1_dlogP = _finite_diff_1d(xhe1, logp)

    dxh2_dlogT = np.zeros_like(xh2)
    dxh_dlogT = np.zeros_like(xh)
    dxhe0_dlogT = np.zeros_like(xhe0)
    dxhe1_dlogT = np.zeros_like(xhe1)
    dt = 1.0e-3
    if target_logt - dt >= SCVH95_LOGT_MIN and target_logt + dt <= SCVH95_LOGT_MAX:
        h_lo = _hydrogen_isotherm(tables, target_logt - dt, logp)
        h_hi = _hydrogen_isotherm(tables, target_logt + dt, logp)
        he_lo = _interp_isotherm(tables["he_i"], target_logt - dt, logp)  # type: ignore[arg-type]
        he_hi = _interp_isotherm(tables["he_i"], target_logt + dt, logp)  # type: ignore[arg-type]
        dxh2_dlogT = (h_hi["xh2_or_he"] - h_lo["xh2_or_he"]) / (2.0 * dt)
        dxh_dlogT = (h_hi["xh_or_he1"] - h_lo["xh_or_he1"]) / (2.0 * dt)
        dxhe0_dlogT = (he_hi["xh2_or_he"] - he_lo["xh2_or_he"]) / (2.0 * dt)
        dxhe1_dlogT = (he_hi["xh_or_he1"] - he_lo["xh_or_he1"]) / (2.0 * dt)

    smix, d_smix_dT, d_smix_dP = _scvh_entropy_of_mixing(
        x_h,
        y_he,
        temp,
        press_gas,
        xh,
        dxh_dlogT,
        dxh_dlogP,
        xh2,
        dxh2_dlogT,
        dxh2_dlogP,
        xhe0,
        dxhe0_dlogT,
        dxhe0_dlogP,
        xhe1,
        dxhe1_dlogT,
        dxhe1_dlogP,
    )

    entr = x_h * entr_h + y_he * entr_he + smix
    ener = x_h * ener_h + y_he * ener_he
    alpha = rho * (
        x_h / den_h * h_valid["dlnrho_dlnT_constP"] + y_he / den_he * he_valid["dlnrho_dlnT_constP"]
    )
    beta = rho * (
        x_h / den_h * h_valid["dlnrho_dlnP_constT"] + y_he / den_he * he_valid["dlnrho_dlnP_constT"]
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        chi_rho_gas = np.where(np.abs(beta) > 1.0e-300, 1.0 / beta, np.nan)
        chi_t_gas = np.where(np.abs(beta) > 1.0e-300, -alpha / beta, np.nan)
    dsdt_cp_hhe = (
        x_h * entr_h * h_valid["dlnS_dlnT_constP"]
        + y_he * entr_he * he_valid["dlnS_dlnT_constP"]
        + temp * d_smix_dT
    ) / np.maximum(entr, 1.0e-300)
    dsdp_ct_hhe = (
        x_h * entr_h * h_valid["dlnS_dlnP_constT"]
        + y_he * entr_he * he_valid["dlnS_dlnP_constT"]
        + press_gas * d_smix_dP
    ) / np.maximum(entr, 1.0e-300)
    grad_ad_gas = -dsdp_ct_hhe / np.maximum(dsdt_cp_hhe, 1.0e-300)
    denom_gas = 1.0 - chi_t_gas * grad_ad_gas
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gamma1_gas = np.where(np.abs(denom_gas) > 1.0e-300, chi_rho_gas / denom_gas, np.nan)
        gamma3m1_gas = np.where(np.abs(denom_gas) > 1.0e-300, chi_rho_gas * grad_ad_gas / denom_gas, np.nan)
        cv_gas = np.where(
            np.abs(gamma3m1_gas) > 1.0e-300,
            press_gas * chi_t_gas / np.maximum(rho * temp * gamma3m1_gas, 1.0e-300),
            np.nan,
        )

    if radiation_enabled:
        prad = (A_RAD / 3.0) * temp**4
        erad = A_RAD * temp**4 / rho
        srad = (prad / rho + erad) / temp
        entr_total = entr + srad
        press_total = press_gas + prad
        ener_total = ener + erad
        chi_rho = np.where(press_total > 0.0, press_gas * chi_rho_gas / press_total, np.nan)
        chi_t = np.where(press_total > 0.0, (press_gas * chi_t_gas + 4.0 * prad) / press_total, np.nan)
        cv = cv_gas + 4.0 * A_RAD * temp**3 / rho
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gamma3m1 = np.where(
                np.abs(rho * temp * cv) > 1.0e-300,
                press_total * chi_t / np.maximum(rho * temp * cv, 1.0e-300),
                np.nan,
            )
            gamma1 = chi_rho + chi_t * gamma3m1
    else:
        prad = np.zeros_like(press_gas)
        press_total = press_gas
        ener_total = ener
        entr_total = entr
        chi_rho = chi_rho_gas
        chi_t = chi_t_gas
        cv = cv_gas
        gamma1 = gamma1_gas
        gamma3m1 = gamma3m1_gas

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cp = np.where(chi_rho > 0.0, cv * gamma1 / np.maximum(chi_rho, 1.0e-300), np.nan)
        cs2 = gamma1 * press_total / rho
        grad_ad = np.where(gamma1 > 0.0, gamma3m1 / np.maximum(gamma1, 1.0e-300), np.nan)

    n_h_tot = x_h * rho / M_H
    n_he_tot = y_he * rho / M_HE
    n_h2 = 0.5 * xh2 * n_h_tot
    n_h0 = np.clip(1.0 - xh2 - xion, 0.0, 1.0) * n_h_tot
    n_hp = xion * n_h_tot
    n_he0 = xhe0 * n_he_tot
    n_he1_num = xhe1 * n_he_tot
    n_he2_num = xhe2 * n_he_tot
    n_e = n_hp + n_he1_num + 2.0 * n_he2_num
    n_total = n_h2 + n_h0 + n_hp + n_he0 + n_he1_num + n_he2_num + n_e
    mu = np.where(n_total > 0.0, rho / (n_total * M_H), np.nan)

    # Keep the internal rho-resampling coordinate in the native SCvH base-10 convention.
    out["logrho"][valid] = np.log(rho) / LN10
    out["logpress"][valid] = np.log(press_total)
    out["logeps"][valid] = np.log(ener_total)
    out["logcs2"][valid] = np.log(np.where(cs2 > 0.0, cs2, np.nan))
    out["gamma1"][valid] = gamma1
    out["gamma3m1"][valid] = gamma3m1
    out["xh2"][valid] = xh2
    out["xion"][valid] = xion
    out["xhe1"][valid] = xhe1
    out["xhe2"][valid] = xhe2
    out["mu"][valid] = mu
    out["beta_rad"][valid] = np.where(press_total > 0.0, prad / press_total, 0.0)
    out["eps_physical"][valid] = ener_total
    out["pgas"][valid] = press_gas
    out["prad"][valid] = prad
    out["chi_rho"][valid] = chi_rho
    out["chi_t"][valid] = chi_t
    out["cv"][valid] = cv
    out["cp"][valid] = cp
    out["entropy"][valid] = entr_total
    out["grad_ad"][valid] = grad_ad
    out["logne"][valid] = np.log(np.maximum(n_e, 1.0e-99))
    return out


def _resample_branch(
    sample_logrho: np.ndarray,
    sample_values: np.ndarray,
    target_logrho: np.ndarray,
) -> np.ndarray:
    good = np.isfinite(sample_logrho) & np.isfinite(sample_values)
    if np.count_nonzero(good) < 4:
        return np.full_like(target_logrho, np.nan, dtype=np.float64)
    x = sample_logrho[good]
    y = sample_values[good]
    if np.any(np.diff(x) <= 0.0):
        keep = np.concatenate(([True], np.diff(x) > 1.0e-12))
        x = x[keep]
        y = y[keep]
    if x.size < 4 or np.any(np.diff(x) <= 0.0):
        return np.full_like(target_logrho, np.nan, dtype=np.float64)
    interp = PchipInterpolator(x, y, extrapolate=False)
    vals = np.asarray(interp(target_logrho), dtype=np.float64)
    return np.where(np.isfinite(vals), vals, np.nan)


def _resample_isotherm_to_rho(
    mixed_pt: dict[str, np.ndarray],
    target_logrho: np.ndarray,
    target_logt: float,
    rho_crit: np.ndarray,
) -> dict[str, np.ndarray]:
    out = {name: np.full_like(target_logrho, np.nan, dtype=np.float64) for name in mixed_pt if name != "logrho"}
    sample_logrho = mixed_pt["logrho"]
    below_crit = SCVH95_PPT_LOGT_MIN <= target_logt < SCVH95_CRIT_LOGT
    if below_crit:
        rho1 = float(PchipInterpolator(rho_crit[:, 0], rho_crit[:, 2], extrapolate=False)(target_logt))
        rho2 = float(PchipInterpolator(rho_crit[:, 0], rho_crit[:, 3], extrapolate=False)(target_logt))
        branch_lo = sample_logrho <= rho1 + 1.0e-10
        branch_hi = sample_logrho >= rho2 - 1.0e-10
        mask_lo = target_logrho <= rho1
        mask_hi = target_logrho >= rho2
        for name, vals in mixed_pt.items():
            if name == "logrho":
                continue
            if np.any(mask_lo):
                out[name][mask_lo] = _resample_branch(sample_logrho[branch_lo], vals[branch_lo], target_logrho[mask_lo])
            if np.any(mask_hi):
                out[name][mask_hi] = _resample_branch(sample_logrho[branch_hi], vals[branch_hi], target_logrho[mask_hi])
    else:
        for name, vals in mixed_pt.items():
            if name == "logrho":
                continue
            out[name] = _resample_branch(sample_logrho, vals, target_logrho)
    return out


def _finalize_scvh_closure(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    chi_rho_fd = _masked_finite_difference(fields["logpress"], logrho, axis=0)
    chi_t_fd = _masked_finite_difference(fields["logpress"], logtemp, axis=1)
    chi_rho = fields["chi_rho"]
    chi_t = fields["chi_t"]
    gamma1 = fields["gamma1"]
    gamma3m1 = fields["gamma3m1"]
    cv = fields["cv"]
    cp = fields["cp"]
    grad_ad = fields["grad_ad"]
    logcs2 = fields["logcs2"]
    good = (
        np.isfinite(fields["logpress"])
        & np.isfinite(fields["logeps"])
        & np.isfinite(chi_rho)
        & np.isfinite(chi_t)
        & np.isfinite(cv)
        & np.isfinite(cp)
        & np.isfinite(gamma1)
        & np.isfinite(gamma3m1)
        & np.isfinite(logcs2)
        & np.isfinite(grad_ad)
        & (cv > 0.0)
        & (cp > 0.0)
        & (gamma1 > 0.0)
        & (gamma3m1 > 0.0)
        & (np.exp(logcs2) > 0.0)
        & (chi_rho > 0.0)
    )
    compare = good & np.isfinite(chi_rho_fd) & np.isfinite(chi_t_fd)
    rel_rho = np.full_like(chi_rho, np.inf, dtype=np.float64)
    rel_t = np.full_like(chi_t, np.inf, dtype=np.float64)
    rel_rho[compare] = np.abs(chi_rho_fd[compare] - chi_rho[compare]) / np.maximum(np.abs(chi_rho[compare]), 1.0e-6)
    rel_t[compare] = np.abs(chi_t_fd[compare] - chi_t[compare]) / np.maximum(np.abs(chi_t[compare]), 1.0e-6)
    good &= (~compare) | ((rel_rho <= 0.35) & (rel_t <= 0.50))
    good &= gamma1 <= 3.5

    fields["chi_rho"][:, :] = np.where(good, chi_rho, np.nan)
    fields["chi_t"][:, :] = np.where(good, chi_t, np.nan)
    fields["cv"][:, :] = np.where(good, cv, np.nan)
    fields["cp"][:, :] = np.where(good, cp, np.nan)
    fields["gamma1"][:, :] = np.where(good, gamma1, np.nan)
    fields["gamma3m1"][:, :] = np.where(good, gamma3m1, np.nan)
    fields["logcs2"][:, :] = np.where(good, logcs2, np.nan)
    fields["grad_ad"][:, :] = np.where(good, grad_ad, np.nan)
    fields["scvh_valid"][:, :] = good.astype(np.float64)


def _sanitize_scvh_species(
    logrho: np.ndarray,
    fields: dict[str, np.ndarray],
    x_h: float,
    y_he: float,
) -> None:
    rho = np.exp(logrho)[:, None]
    diag_valid = (
        np.isfinite(fields["xh2"])
        & np.isfinite(fields["xion"])
        & np.isfinite(fields["xhe1"])
        & np.isfinite(fields["xhe2"])
    )
    if not np.any(diag_valid):
        return

    xh2 = np.clip(fields["xh2"], 0.0, 1.0)
    xion = np.clip(fields["xion"], 0.0, 1.0)
    sum_h = xh2 + xion
    h_over = sum_h > 1.0
    xh2[h_over] /= sum_h[h_over]
    xion[h_over] /= sum_h[h_over]

    xhe1 = np.clip(fields["xhe1"], 0.0, 1.0)
    xhe2 = np.clip(fields["xhe2"], 0.0, 1.0)
    sum_he = xhe1 + xhe2
    he_over = sum_he > 1.0
    xhe1[he_over] /= sum_he[he_over]
    xhe2[he_over] /= sum_he[he_over]
    xhe0 = np.clip(1.0 - xhe1 - xhe2, 0.0, 1.0)

    n_h_tot = x_h * rho / M_H
    n_he_tot = y_he * rho / M_HE
    n_h2 = 0.5 * xh2 * n_h_tot
    n_h0 = np.clip(1.0 - xh2 - xion, 0.0, 1.0) * n_h_tot
    n_hp = xion * n_h_tot
    n_he0 = xhe0 * n_he_tot
    n_he1_num = xhe1 * n_he_tot
    n_he2_num = xhe2 * n_he_tot
    n_e = n_hp + n_he1_num + 2.0 * n_he2_num
    n_total = n_h2 + n_h0 + n_hp + n_he0 + n_he1_num + n_he2_num + n_e
    mu = np.where(n_total > 0.0, rho / (n_total * M_H), np.nan)
    logne = np.full_like(n_e, np.nan, dtype=np.float64)
    positive_ne = n_e > 0.0
    logne[positive_ne] = np.log(n_e[positive_ne])

    fields["xh2"][:, :] = np.where(diag_valid, xh2, np.nan)
    fields["xion"][:, :] = np.where(diag_valid, xion, np.nan)
    fields["xhe1"][:, :] = np.where(diag_valid, xhe1, np.nan)
    fields["xhe2"][:, :] = np.where(diag_valid, xhe2, np.nan)
    fields["mu"][:, :] = np.where(diag_valid, mu, np.nan)
    fields["logne"][:, :] = np.where(diag_valid, logne, np.nan)


def scvh95_thermo(
    logrho: np.ndarray,
    logtemp: np.ndarray,
    x_h: float,
    y_he: float,
    radiation_enabled: bool,
    data_dir: str | None = None,
    logp_step: float = SCVH95_INTERP_LOGP_STEP,
    ppt_logp_step: float = SCVH95_INTERP_PPT_LOGP_STEP,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if logp_step <= 0.0:
        raise ValueError("SCvH interpolation logP step must be positive.")
    if ppt_logp_step <= 0.0:
        raise ValueError("SCvH PPT interpolation logP step must be positive.")
    tables = _load_tables(data_dir)
    target_logrho = logrho / LN10
    target_logtemp = logtemp / LN10
    logp_base = np.arange(SCVH95_LOGP_MIN, SCVH95_LOGP_MAX + 1.0e-12, logp_step)
    logp_ppt = np.arange(
        SCVH95_PPT_LOGP_MIN,
        SCVH95_PPT_LOGP_MAX + 1.0e-12,
        ppt_logp_step,
    )
    logp_composite = np.unique(np.concatenate((logp_base, logp_ppt)))

    shape = (logrho.size, logtemp.size)
    fields = {
        "logpress": np.full(shape, np.nan, dtype=np.float64),
        "logeps": np.full(shape, np.nan, dtype=np.float64),
        "logcs2": np.full(shape, np.nan, dtype=np.float64),
        "gamma1": np.full(shape, np.nan, dtype=np.float64),
        "gamma3m1": np.full(shape, np.nan, dtype=np.float64),
        "xh2": np.full(shape, np.nan, dtype=np.float64),
        "xion": np.full(shape, np.nan, dtype=np.float64),
        "xhe1": np.full(shape, np.nan, dtype=np.float64),
        "xhe2": np.full(shape, np.nan, dtype=np.float64),
        "mu": np.full(shape, np.nan, dtype=np.float64),
        "beta_rad": np.full(shape, np.nan, dtype=np.float64),
        "eps_physical": np.full(shape, np.nan, dtype=np.float64),
        "eps_runtime": np.full(shape, np.nan, dtype=np.float64),
        "pgas": np.full(shape, np.nan, dtype=np.float64),
        "prad": np.full(shape, np.nan, dtype=np.float64),
        "chi_rho": np.full(shape, np.nan, dtype=np.float64),
        "chi_t": np.full(shape, np.nan, dtype=np.float64),
        "cv": np.full(shape, np.nan, dtype=np.float64),
        "cp": np.full(shape, np.nan, dtype=np.float64),
        "entropy": np.full(shape, np.nan, dtype=np.float64),
        "grad_ad": np.full(shape, np.nan, dtype=np.float64),
        "logne": np.full(shape, np.nan, dtype=np.float64),
        "scvh_valid": np.zeros(shape, dtype=np.float64),
    }

    rho_crit = tables["rho_crit"]  # type: ignore[assignment]
    for jt, logt in enumerate(target_logtemp):
        mixed_pt = _compose_and_mix(
            tables,
            float(logt),
            logp_composite,
            x_h,
            y_he,
            radiation_enabled,
        )
        if not np.any(np.isfinite(mixed_pt["logrho"])):
            continue
        row = _resample_isotherm_to_rho(mixed_pt, target_logrho, float(logt), rho_crit)
        for name, vals in row.items():
            fields[name][:, jt] = vals
        fields["eps_runtime"][:, jt] = fields["eps_physical"][:, jt]

    _sanitize_scvh_species(logrho, fields, x_h, y_he)
    _finalize_scvh_closure(logrho, logtemp, fields)

    context = {
        "source_primary": "Saumon_Chabrier_VanHorn_1995",
        "source_url": SCVH95_BASE_URL,
        "data_dir": tables["root"],
        "log10_temp_min": SCVH95_LOGT_MIN,
        "log10_temp_max": SCVH95_LOGT_MAX,
        "log10_press_min": SCVH95_LOGP_MIN,
        "log10_press_max": SCVH95_LOGP_MAX,
        "ppt_log10_temp_min": SCVH95_PPT_LOGT_MIN,
        "ppt_log10_temp_max": SCVH95_PPT_LOGT_MAX,
        "interp_log10_press_step": logp_step,
        "interp_ppt_log10_press_step": ppt_logp_step,
    }
    return fields, context
