#!/usr/bin/env python3

"""Generate an AthenaK-native H-only LTE/Saha EOS table in cgs units.

The table is 2D in natural-log axes:
  - logrho = ln(rho_cgs)
  - logtemp = ln(T_kelvin)

Fields are written as doubles in AthenaK's `TableReader` format with temperature as the
fastest-varying index:
  - logpress   = ln(p_cgs)
  - logeps     = ln(eps_cgs)          [specific internal energy, erg g^-1]
  - logcs2     = ln(cs^2_cgs)         [(cm s^-1)^2]
  - gamma1     = Gamma_1
  - gamma3m1   = Gamma_3 - 1
  - xion       = hydrogen ionization fraction
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


K_B = 1.380649e-16
H_PLANCK = 6.62607015e-27
M_E = 9.1093837015e-28
M_H = 1.6735575e-24
CHI_H = 2.179872e-11
T_ION = CHI_H / K_B
R_H = K_B / M_H
# For ground-state pure hydrogen in LTE, the usual electron spin degeneracy factor of 2
# is cancelled by the neutral-H ground-state degeneracy U_H0 ~= 2, while U_H+ ~= 1.
# With that partition-function convention:
#   x^2 / (1 - x) = (m_H / rho) * (2 pi m_e k_B T / h^2)^(3/2) * exp(-chi / k_B T)
C_SAHA = M_H * (2.0 * math.pi * M_E * K_B / H_PLANCK**2) ** 1.5

REPO_ROOT = Path(__file__).resolve().parents[1]
EOS_TABLE_DIR = REPO_ROOT / "eos_tables"

SAHA_OUTPUT = EOS_TABLE_DIR / "saha_hydrogen.table"
SAHA_NRHO = 256
SAHA_NTEMP = 512
SAHA_LOG10_RHO_MIN = -20.0
SAHA_LOG10_RHO_MAX = 2.0
SAHA_LOG10_TEMP_MIN = -8.0
SAHA_LOG10_TEMP_MAX = 12.0


def ionization_fraction(rho: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Stable H-only Saha ionization fraction."""
    logy = np.log(C_SAHA) + 1.5 * np.log(temp) - T_ION / temp - np.log(rho)
    x = np.empty_like(logy)

    low = logy < -40.0
    high = logy > 40.0
    mid = ~(low | high)

    x[low] = np.exp(0.5 * logy[low])
    x[high] = 1.0 - np.exp(-logy[high])
    if np.any(mid):
        y = np.exp(logy[mid])
        x[mid] = 2.0 * y / (np.sqrt(y * y + 4.0 * y) + y)

    return np.clip(x, 1.0e-12, 1.0 - 1.0e-12)


def saha_thermo(rho: np.ndarray, temp: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate table fields and analytic derivatives in cgs."""
    x = ionization_fraction(rho, temp)
    u_h = T_ION / temp
    g = x * (1.0 - x) / (2.0 - x)

    press = rho * R_H * temp * (1.0 + x)
    eps = 1.5 * R_H * temp * (1.0 + x) + x * CHI_H / M_H

    deps_drho_t = -(g / rho) * (1.5 * R_H * temp + CHI_H / M_H)
    deps_dtemp_rho = (
        1.5 * R_H * (1.0 + x)
        + (1.5 * R_H * temp + CHI_H / M_H) * g * (1.5 + u_h) / temp
    )
    dp_drho_t = R_H * temp * ((1.0 + x) - g)
    dp_dtemp_rho = rho * R_H * ((1.0 + x) + g * (1.5 + u_h))

    dtemp_drho_s = (press / (rho * rho) - deps_drho_t) / deps_dtemp_rho
    cs2 = dp_drho_t + dp_dtemp_rho * dtemp_drho_s
    gamma1 = rho * cs2 / press
    gamma3m1 = rho / temp * dtemp_drho_s

    if np.any(deps_dtemp_rho <= 0.0):
        raise RuntimeError("Encountered non-positive cv while generating Saha table.")
    if np.any(cs2 <= 0.0):
        raise RuntimeError("Encountered non-positive cs^2 while generating Saha table.")
    if np.any(press <= 0.0) or np.any(eps <= 0.0):
        raise RuntimeError("Encountered non-positive pressure or specific energy.")

    return {
        "logpress": np.log(press),
        "logeps": np.log(eps),
        "logcs2": np.log(cs2),
        "gamma1": gamma1,
        "gamma3m1": gamma3m1,
        "xion": x,
    }


def ensure_parent(path: os.PathLike[str] | str) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_block(fp, begin: str, lines: Iterable[str], end: str) -> None:
    fp.write(f"<{begin}>\n".encode("ascii"))
    for line in lines:
        fp.write(f"{line}\n".encode("ascii"))
    fp.write(f"<{end}>\n".encode("ascii"))


def write_table(path: os.PathLike[str] | str, logrho: np.ndarray, logtemp: np.ndarray,
                fields: dict[str, np.ndarray]) -> None:
    ensure_parent(path)

    metadata = [
        "endianness = little",
        "table_type = saha_hydrogen_lte",
        "log_axis_base = e",
    ]
    scalars = [
        f"k_b = {K_B:.17e}",
        f"h = {H_PLANCK:.17e}",
        f"m_e = {M_E:.17e}",
        f"m_h = {M_H:.17e}",
        f"chi_h = {CHI_H:.17e}",
        f"t_ion = {T_ION:.17e}",
        f"r_h = {R_H:.17e}",
        f"c_saha = {C_SAHA:.17e}",
    ]
    points = [
        f"logrho = {len(logrho)}",
        f"logtemp = {len(logtemp)}",
    ]
    field_names = ["logpress", "logeps", "logcs2", "gamma1", "gamma3m1", "xion"]

    with open(path, "wb") as fp:
        write_block(fp, "metadatabegin", metadata, "metadataend")
        write_block(fp, "scalarsbegin", scalars, "scalarsend")
        write_block(fp, "pointsbegin", points, "pointsend")
        write_block(fp, "fieldsbegin", field_names, "fieldsend")

        fp.write(np.asarray(logrho, dtype=np.float64).tobytes(order="C"))
        fp.write(np.asarray(logtemp, dtype=np.float64).tobytes(order="C"))
        for name in field_names:
            fp.write(np.asarray(fields[name], dtype=np.float64).ravel(order="C").tobytes())


def main() -> int:
    if SAHA_NRHO < 2 or SAHA_NTEMP < 2:
        raise SystemExit("nrho and ntemp must both be at least 2")
    if SAHA_LOG10_RHO_MIN >= SAHA_LOG10_RHO_MAX:
        raise SystemExit("log10-rho-min must be less than log10-rho-max")
    if SAHA_LOG10_TEMP_MIN >= SAHA_LOG10_TEMP_MAX:
        raise SystemExit("log10-temp-min must be less than log10-temp-max")

    logrho = np.linspace(
        SAHA_LOG10_RHO_MIN * math.log(10.0),
        SAHA_LOG10_RHO_MAX * math.log(10.0),
        SAHA_NRHO,
    )
    logtemp = np.linspace(
        SAHA_LOG10_TEMP_MIN * math.log(10.0),
        SAHA_LOG10_TEMP_MAX * math.log(10.0),
        SAHA_NTEMP,
    )

    rho = np.exp(logrho)[:, None]
    temp = np.exp(logtemp)[None, :]
    fields = saha_thermo(rho, temp)
    write_table(SAHA_OUTPUT, logrho, logtemp, fields)

    print(
        "Wrote Saha table:",
        SAHA_OUTPUT,
        f"(nrho={SAHA_NRHO}, ntemp={SAHA_NTEMP}, "
        f"log10rho=[{SAHA_LOG10_RHO_MIN}, {SAHA_LOG10_RHO_MAX}], "
        f"log10T=[{SAHA_LOG10_TEMP_MIN}, {SAHA_LOG10_TEMP_MAX}])",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
