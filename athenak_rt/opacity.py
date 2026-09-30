"""Opacities: MESA/OPAL grey Rosseland tables and the H/He continuum.

Grey opacity
    ``MesaOpacityModel`` holds a high-T (OPAL, GS98, X=0.7, Z=0.02) and a
    low-T (Ferguson et al. 2005) MESA table of log10 kappa_R(log T, log R),
    R = rho / T6^3.  ``mesa_alpha_scalar`` interpolates bilinearly and returns
    the absorption coefficient alpha_R = kappa_R rho [cm^-1]; the low-T table is
    used up to its maximum log T, the high-T table above it.

Multifrequency H/He continuum (no lines, no metals)
    ``table_state`` takes the populations from the tabulated EOS (the pipeline
    default, ``--populations eos``).  ``saha_state`` (``--populations saha``)
    solves Saha ionization equilibrium for an X=0.7, Y=0.3 gas (H I/II,
    He I/II/III) by bisection on n_e.  ``continuum_absorption`` returns
    the free-free (H II, He II, He III) plus bound-free (H I n<=6, He I, He II)
    absorption coefficient with the stimulated-emission factor (1 - e^{-h nu/kT}).
    Electron scattering enters as alpha_s = sigma_T n_e.

All numerics are those of the validated scripts (``light_curve_tde_cartesian``
and ``rt_continuum_experiments``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numba import njit, prange

from .constants import (
    C_CGS,
    CHI_H,
    CHI_HE1,
    CHI_HE2,
    ELECTRON_MASS_CGS,
    EV_CGS,
    H_CGS,
    K_B_CGS,
    NU_H,
    NU_HE1,
    NU_HE2,
    PROTON_MASS_CGS,
    SIGMA_THOMSON_CGS,
    X_H,
    Y_HE,
)

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MESA_HIGH_T = PACKAGE_DATA_DIR / "gs98_z0.02_x0.7.data"
DEFAULT_MESA_LOW_T = PACKAGE_DATA_DIR / "lowT_fa05_gs98_z0.02_x0.7.data"


# ---------------------------------------------------------------------------
# MESA opacity tables
# ---------------------------------------------------------------------------
class MesaOpacityTable:
    """One MESA opacity table: log10 kappa on a (log T, log R) grid."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.log_t, self.log_r, self.log_kappa = self._read(self.path)

    @staticmethod
    def _read(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = path.read_text(errors="replace").splitlines()
        log_r: Optional[np.ndarray] = None
        rows: List[List[float]] = []
        for i, line in enumerate(lines):
            if "logR =" not in line:
                continue
            for candidate in lines[i + 1 :]:
                parts = candidate.split()
                if parts:
                    log_r = np.array([float(x) for x in parts], dtype=np.float64)
                    break
            break
        if log_r is None:
            raise RuntimeError(f"Could not read logR axis from {path}.")

        for line in lines:
            parts = line.split()
            if len(parts) != log_r.size + 1:
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)
        if not rows:
            raise RuntimeError(f"Could not read opacity rows from {path}.")

        raw = np.array(rows, dtype=np.float64)
        log_t = raw[:, 0]
        log_kappa = raw[:, 1:]
        order = np.argsort(log_t)
        return (
            np.ascontiguousarray(log_t[order], dtype=np.float64),
            np.ascontiguousarray(log_r, dtype=np.float64),
            np.ascontiguousarray(log_kappa[order], dtype=np.float64),
        )

    @classmethod
    def constant(cls, log_kappa: float) -> "MesaOpacityTable":
        """A synthetic table with uniform log10 kappa (used by the tests)."""
        table = cls.__new__(cls)
        table.path = Path("<constant>")
        table.log_t = np.array([0.0, 10.0], dtype=np.float64)
        table.log_r = np.array([-20.0, 20.0], dtype=np.float64)
        table.log_kappa = np.full((2, 2), float(log_kappa), dtype=np.float64)
        return table


class MesaOpacityModel:
    """High-T + low-T MESA tables and the flat argument tuple the kernels take."""

    def __init__(self, high_t_path: Path, low_t_path: Path):
        self.high_t = MesaOpacityTable(high_t_path)
        self.low_t = MesaOpacityTable(low_t_path)

    @classmethod
    def from_tables(
        cls, high_t: MesaOpacityTable, low_t: MesaOpacityTable
    ) -> "MesaOpacityModel":
        model = cls.__new__(cls)
        model.high_t = high_t
        model.low_t = low_t
        return model

    @classmethod
    def constant_kappa(cls, kappa_cm2_g: float) -> "MesaOpacityModel":
        table = MesaOpacityTable.constant(math.log10(kappa_cm2_g))
        return cls.from_tables(table, table)

    @property
    def kernel_args(self) -> tuple:
        """Low-T (log_t, log_r, log_kappa) followed by the high-T triple."""
        return (
            self.low_t.log_t,
            self.low_t.log_r,
            self.low_t.log_kappa,
            self.high_t.log_t,
            self.high_t.log_r,
            self.high_t.log_kappa,
        )


@njit(cache=True)
def _bracket_axis(axis, value):
    n = axis.size
    if value <= axis[0]:
        return 0, 0.0
    if value >= axis[n - 1]:
        return n - 2, 1.0

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if value < axis[mid]:
            hi = mid
        else:
            lo = mid

    denom = axis[lo + 1] - axis[lo]
    if denom <= 0.0:
        return lo, 0.0
    return lo, (value - axis[lo]) / denom


@njit(cache=True)
def _interp_log_kappa_scalar(log_t, log_r, log_t_axis, log_r_axis, log_kappa_table):
    it, wt = _bracket_axis(log_t_axis, log_t)
    ir, wr = _bracket_axis(log_r_axis, log_r)
    wt0 = 1.0 - wt
    wr0 = 1.0 - wr
    v00 = log_kappa_table[it, ir]
    v01 = log_kappa_table[it, ir + 1]
    v10 = log_kappa_table[it + 1, ir]
    v11 = log_kappa_table[it + 1, ir + 1]
    return wt0 * (wr0 * v00 + wr * v01) + wt * (wr0 * v10 + wr * v11)


@njit(cache=True)
def mesa_alpha_scalar(
    rho,
    temp,
    low_log_t,
    low_log_r,
    low_log_kappa,
    high_log_t,
    high_log_r,
    high_log_kappa,
):
    """Grey absorption coefficient kappa_R(rho, T) * rho [cm^-1]; 0 for empty cells."""
    if rho <= 0.0 or temp <= 0.0 or not math.isfinite(rho) or not math.isfinite(temp):
        return 0.0

    log_t = math.log10(temp)
    log_r = math.log10(rho) - 3.0 * log_t + 18.0
    if log_t <= low_log_t[low_log_t.size - 1]:
        log_kappa = _interp_log_kappa_scalar(
            log_t, log_r, low_log_t, low_log_r, low_log_kappa
        )
    else:
        log_kappa = _interp_log_kappa_scalar(
            log_t, log_r, high_log_t, high_log_r, high_log_kappa
        )

    alpha = 10.0**log_kappa * rho
    if not math.isfinite(alpha) or alpha < 0.0:
        return 0.0
    return alpha


@njit(parallel=True, cache=True)
def mesa_alpha_grid(
    rho_grid,
    temp_grid,
    low_log_t,
    low_log_r,
    low_log_kappa,
    high_log_t,
    high_log_r,
    high_log_kappa,
):
    """Grey alpha_R on a whole (nz, ny, nx) grid."""
    nz, ny, nx = rho_grid.shape
    alpha_grid = np.zeros((nz, ny, nx), dtype=np.float64)
    plane = ny * nx
    ncell = nz * plane
    for cell in prange(ncell):
        k = cell // plane
        rem = cell - k * plane
        j = rem // nx
        i = rem - j * nx
        alpha_grid[k, j, i] = mesa_alpha_scalar(
            rho_grid[k, j, i],
            temp_grid[k, j, i],
            low_log_t,
            low_log_r,
            low_log_kappa,
            high_log_t,
            high_log_r,
            high_log_kappa,
        )
    return alpha_grid


# ---------------------------------------------------------------------------
# H/He continuum
# ---------------------------------------------------------------------------
@njit(cache=True)
def saha_state(rho, temp):
    """n_e, n_HI, n_HII, n_HeI, n_HeII, n_HeIII of an H/He gas in Saha equilibrium."""
    n_h = X_H * rho / PROTON_MASS_CGS
    n_he = Y_HE * rho / (4.0 * PROTON_MASS_CGS)
    base = (2.0 * math.pi * ELECTRON_MASS_CGS * K_B_CGS * temp / (H_CGS * H_CGS)) ** 1.5
    s_h = base * math.exp(-min(CHI_H / (K_B_CGS * temp), 700.0))
    s_1 = 4.0 * base * math.exp(-min(CHI_HE1 / (K_B_CGS * temp), 700.0))
    s_2 = base * math.exp(-min(CHI_HE2 / (K_B_CGS * temp), 700.0))
    lo = math.log(1.0e-30 * n_h + 1.0e-300)
    hi = math.log(n_h + 2.0 * n_he)
    ne = math.exp(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ne = math.exp(mid)
        x_h = s_h / (ne + s_h)
        r1 = s_1 / ne
        r2 = s_2 / ne
        f0 = 1.0 / (1.0 + r1 + r1 * r2)
        supply = n_h * x_h + n_he * (r1 * f0 + 2.0 * r1 * r2 * f0)
        if supply > ne:
            lo = mid
        else:
            hi = mid
    x_h = s_h / (ne + s_h)
    r1 = s_1 / ne
    r2 = s_2 / ne
    f0 = 1.0 / (1.0 + r1 + r1 * r2)
    return (
        ne,
        n_h * (1.0 - x_h),
        n_h * x_h,
        n_he * f0,
        n_he * r1 * f0,
        n_he * r1 * r2 * f0,
    )


@njit(cache=True)
def table_state(rho, temp, lr0, dlr, lt0, dlt, lnf, nh_per_rho, nhe_per_rho):
    """Same return values as ``saha_state``, from EOS-table ionization fractions.

    ``lnf[5, nr, nt]`` holds ln(H I, H II, He I, He II, He III fractions) on a
    uniform (ln rho, ln T) grid; it is interpolated bilinearly (clamped to the
    table range) and exponentiated, which keeps tiny neutral fractions accurate.
    """
    nr = lnf.shape[1]
    nt = lnf.shape[2]
    x = (math.log(rho) - lr0) / dlr
    y = (math.log(temp) - lt0) / dlt
    x = min(max(x, 0.0), nr - 1.000001)
    y = min(max(y, 0.0), nt - 1.000001)
    i = int(x)
    j = int(y)
    fx = x - i
    fy = y - j
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    f0 = math.exp(w00 * lnf[0, i, j] + w10 * lnf[0, i + 1, j] + w01 * lnf[0, i, j + 1] + w11 * lnf[0, i + 1, j + 1])
    f1 = math.exp(w00 * lnf[1, i, j] + w10 * lnf[1, i + 1, j] + w01 * lnf[1, i, j + 1] + w11 * lnf[1, i + 1, j + 1])
    f2 = math.exp(w00 * lnf[2, i, j] + w10 * lnf[2, i + 1, j] + w01 * lnf[2, i, j + 1] + w11 * lnf[2, i + 1, j + 1])
    f3 = math.exp(w00 * lnf[3, i, j] + w10 * lnf[3, i + 1, j] + w01 * lnf[3, i, j + 1] + w11 * lnf[3, i + 1, j + 1])
    f4 = math.exp(w00 * lnf[4, i, j] + w10 * lnf[4, i + 1, j] + w01 * lnf[4, i, j + 1] + w11 * lnf[4, i + 1, j + 1])
    n_h = nh_per_rho * rho
    n_he = nhe_per_rho * rho
    n_h1 = n_h * f0
    n_h2 = n_h * f1
    n_he1 = n_he * f2
    n_he2 = n_he * f3
    n_he3 = n_he * f4
    return n_h2 + n_he2 + 2.0 * n_he3, n_h1, n_h2, n_he1, n_he2, n_he3


@njit(cache=True)
def continuum_absorption(nu, temp, ne, n_h1, n_h2, n_he1, n_he2, n_he3):
    """H/He free-free + bound-free absorption [cm^-1] with stimulated emission."""
    x = H_CGS * nu / (K_B_CGS * temp)
    stim = 1.0 - math.exp(-x) if x < 50.0 else 1.0
    alpha = 3.692e8 / math.sqrt(temp) * ne * (n_h2 + n_he2 + 4.0 * n_he3) / nu**3
    for n in range(1, 7):
        nu_n = NU_H / (n * n)
        if nu >= nu_n:
            boltz = (CHI_H * (1.0 - 1.0 / (n * n))) / (K_B_CGS * temp)
            if boltz < 600.0:
                alpha += (
                    n_h1 * n * n * math.exp(-boltz) * 7.906e-18 * n * (nu_n / nu) ** 3
                )
    if nu >= NU_HE1:
        y = NU_HE1 / nu
        alpha += n_he1 * 7.42e-18 * (1.66 * y**2.05 - 0.66 * y**3.05)
    if nu >= NU_HE2:
        alpha += n_he2 * 1.58e-18 * (NU_HE2 / nu) ** 3
    return alpha * stim


@njit(cache=True)
def electron_scattering(ne):
    """Thomson scattering coefficient alpha_s = sigma_T n_e [cm^-1]."""
    return SIGMA_THOMSON_CGS * ne


@njit(cache=True)
def planck_nu(nu, temp):
    """B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1]; 0 where h nu / kT > 600."""
    x = H_CGS * nu / (K_B_CGS * temp)
    if x > 600.0:
        return 0.0
    return 2.0 * H_CGS * nu**3 / (C_CGS * C_CGS) / math.expm1(x)


# ---------------------------------------------------------------------------
# Photon-energy grid
# ---------------------------------------------------------------------------
def photon_energy_grid(nfreq: int = 73, emin_ev: float = 0.1, emax_ev: float = 1000.0):
    """Log-spaced photon energies [eV] and the matching frequencies [Hz]."""
    if nfreq < 2:
        raise RuntimeError("The multifrequency grid needs at least two photon energies.")
    if not (0.0 < emin_ev < emax_ev):
        raise RuntimeError("Photon energies must satisfy 0 < emin < emax.")
    energies_ev = np.logspace(math.log10(emin_ev), math.log10(emax_ev), int(nfreq))
    return energies_ev, energies_ev * EV_CGS / H_CGS


def frequency_quadrature_weights(nus: np.ndarray) -> np.ndarray:
    """Trapezoid weights in ln(nu) so that sum_f I_f w_f approximates int I_nu dnu."""
    nus = np.asarray(nus, dtype=np.float64)
    nf = nus.size
    lognu = np.log(nus)
    weight = np.empty(nf)
    for f in range(nf):
        left = lognu[f] - lognu[f - 1] if f > 0 else 0.0
        right = lognu[f + 1] - lognu[f] if f < nf - 1 else 0.0
        weight[f] = 0.5 * (left + right) * nus[f]
    return weight
