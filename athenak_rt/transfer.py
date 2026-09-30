"""Numba ray-integration kernels on the resampled (rho, T) cube.

All kernels take the cube in ``(nz, ny, nx)`` order together with a direction
code (0 = x, 1 = y, 2 = z) and a sign: for a *positive* direction the observer
sits at the +axis face and the ray is walked from the last index downward; for
a negative direction (``"-y"``) the observer sits at the -axis face and the ray
is walked from index 0 upward.  Image rows/columns are (y, x) for z rays,
(z, x) for y rays and (z, y) for x rays.

Modes (``tau_stop`` truncates every integral once its optical depth exceeds it):

tau1
    Grey Rosseland tau along the ray; the first cell where tau >= tau_ph defines
    the photosphere and the pixel emits 4 sigma T_ph^4 dA (projected
    isotropic-equivalent normalization, Yang et al. 2026 eq. 10).
grey (formal)
    I = int B(T) e^{-tau_R} dtau_R with B = sigma T^4 / pi.
grey-therm
    Grey with a scattering/absorption split kappa_s = sigma_T n_e / rho,
    kappa_a = kappa_R - kappa_s, effective depth dtau* = sqrt(a_a (a_a + a_s)) ds
    and source weight 2 sqrt(eps) / (1 + sqrt(eps)), eps = a_a / (a_a + a_s).
multifreq
    Per-frequency H/He continuum absorption a_a(nu) (Saha populations) with the
    same thermalization treatment against a_s = sigma_T n_e (``scattering=True``)
    or pure absorption (``scattering=False``).  ``grey_limit=True`` replaces
    a_a(nu) by alpha_R for every frequency, which must reproduce ``grey``.

The per-cell arithmetic is exactly that of the validated scripts.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .constants import (
    C_CGS,
    H_CGS,
    K_B_CGS,
    PROJECTED_LUMINOSITY_FACTOR,
    SAFE_POSITIVE,
    SIGMA_OVER_PI,
    SIGMA_SB_CGS,
)
from .opacity import (
    continuum_absorption,
    electron_scattering,
    mesa_alpha_scalar,
    planck_nu,
    saha_state,
    table_state,
)

_NO_LNF = np.zeros((5, 2, 2))  # placeholder when the Saha solver is used

# Population arguments (use_table, lr0, dlr, lt0, dlt, lnf, nh_per_rho, nhe_per_rho)
# for the Saha solver and for the modes without ionization.  They have the same
# types as the EOS-table arguments, so every mode shares one compiled (and cached)
# specialization.  Omitting them would make numba key the cache on object ids and
# recompile in every new process.
SAHA_POPULATION_ARGS = (False, 0.0, 1.0, 0.0, 1.0, _NO_LNF, 0.0, 0.0)

DIRECTION_X = 0
DIRECTION_Y = 1
DIRECTION_Z = 2


@njit(cache=True)
def _image_shape(nz, ny, nx, direction_code):
    if direction_code == 0:
        return nz, ny, nx
    if direction_code == 1:
        return nz, nx, ny
    return ny, nx, nz


@njit(cache=True)
def _cell_index(row, col, idx, direction_code):
    """(k, j, i) of the cell at LOS index ``idx`` of image pixel (row, col).

    The explicit int64 casts keep numba from unifying the (unsigned) parfor
    index with the signed loop counters into a float.
    """
    r = np.int64(row)
    c = np.int64(col)
    s = np.int64(idx)
    if direction_code == 0:
        return r, c, s
    if direction_code == 1:
        return r, s, c
    return s, r, c


@njit(cache=True)
def _blackbody_intensity_scalar(temp):
    """Pixel value before the projected area: 4 sigma T^4 (see constants)."""
    if temp <= 0.0 or not math.isfinite(temp):
        return 0.0
    return PROJECTED_LUMINOSITY_FACTOR * SIGMA_SB_CGS * temp**4


@njit(cache=True)
def _blackbody_lnu_pixel_scalar(temp, frequency_hz, area_cm2):
    if temp <= 0.0 or frequency_hz <= 0.0 or not math.isfinite(temp):
        return 0.0
    expo = H_CGS * frequency_hz / (K_B_CGS * temp)
    if expo >= 700.0:
        return 0.0
    b_nu = (
        2.0
        * H_CGS
        * frequency_hz
        * frequency_hz
        * frequency_hz
        / (C_CGS * C_CGS)
        / math.expm1(max(expo, SAFE_POSITIVE))
    )
    return PROJECTED_LUMINOSITY_FACTOR * math.pi * b_nu * area_cm2


@njit(parallel=True, cache=True)
def blackbody_band_maps(photosphere_temp, valid, frequencies_hz, area_cm2):
    """Per-pixel L_nu [erg/s/Hz] of a blackbody at T_ph for each band frequency."""
    nband = frequencies_hz.size
    nrow, ncol = photosphere_temp.shape
    npix = nrow * ncol
    out = np.zeros((nband, nrow, ncol), dtype=np.float64)
    for item in prange(nband * npix):
        band = item // npix
        rem = item - band * npix
        row = rem // ncol
        col = rem - row * ncol
        if valid[row, col]:
            out[band, row, col] = _blackbody_lnu_pixel_scalar(
                photosphere_temp[row, col], frequencies_hz[band], area_cm2
            )
    return out


@njit(parallel=True, cache=True)
def integrate_tau1_rays(
    rho_grid,
    temp_grid,
    x_axis,
    y_axis,
    z_axis,
    low_log_t,
    low_log_r,
    low_log_kappa,
    high_log_t,
    high_log_r,
    high_log_kappa,
    direction_code,
    positive_direction,
    ds_cm,
    area_cm2,
    tau_photosphere,
):
    """Grey tau = tau_ph photosphere.

    Returns tau_total, T_ph, photosphere LOS coordinate, pixel luminosity, valid.
    """
    nz, ny, nx = rho_grid.shape
    nrow, ncol, nlos = _image_shape(nz, ny, nx, direction_code)

    tau_total = np.zeros((nrow, ncol), dtype=np.float64)
    photosphere_temp = np.empty((nrow, ncol), dtype=np.float64)
    photosphere_coord = np.empty((nrow, ncol), dtype=np.float64)
    flux_map = np.zeros((nrow, ncol), dtype=np.float64)
    valid = np.zeros((nrow, ncol), dtype=np.bool_)

    npix = nrow * ncol
    for pix in prange(npix):
        row = pix // ncol
        col = pix - row * ncol
        tau = 0.0
        found = False
        tph = math.nan
        coord = math.nan

        for s in range(nlos):
            idx = nlos - 1 - s if positive_direction else s
            kk, jj, ii = _cell_index(row, col, idx, direction_code)
            if direction_code == 0:
                axis_coord = x_axis[ii]
            elif direction_code == 1:
                axis_coord = y_axis[jj]
            else:
                axis_coord = z_axis[kk]

            temp = temp_grid[kk, jj, ii]
            alpha = mesa_alpha_scalar(
                rho_grid[kk, jj, ii],
                temp,
                low_log_t,
                low_log_r,
                low_log_kappa,
                high_log_t,
                high_log_r,
                high_log_kappa,
            )
            tau += alpha * ds_cm
            if (not found) and tau >= tau_photosphere:
                found = True
                tph = temp
                coord = axis_coord

        tau_total[row, col] = tau
        photosphere_temp[row, col] = tph
        photosphere_coord[row, col] = coord
        if found:
            valid[row, col] = True
            flux_map[row, col] = _blackbody_intensity_scalar(tph) * area_cm2

    return tau_total, photosphere_temp, photosphere_coord, flux_map, valid


@njit(parallel=True, cache=True)
def integrate_grey_rays(
    rho_grid,
    temp_grid,
    low_log_t,
    low_log_r,
    low_log_kappa,
    high_log_t,
    high_log_r,
    high_log_kappa,
    direction_code,
    positive_direction,
    ds_cm,
    thermalization,
    tau_stop,
    use_table=False,
    lr0=0.0,
    dlr=1.0,
    lt0=0.0,
    dlt=1.0,
    lnf=_NO_LNF,
    nh_per_rho=0.0,
    nhe_per_rho=0.0,
):
    """Grey formal solution (or grey + thermalization depth).

    Returns the frequency-integrated intensity map I [erg s^-1 cm^-2 sr^-1] and
    the accumulated (effective) optical depth per pixel.
    """
    nz, ny, nx = rho_grid.shape
    nrow, ncol, nlos = _image_shape(nz, ny, nx, direction_code)
    intensity = np.zeros((nrow, ncol), dtype=np.float64)
    tau_map = np.zeros((nrow, ncol), dtype=np.float64)

    npix = nrow * ncol
    for pix in prange(npix):
        row = pix // ncol
        col = pix - row * ncol
        tau_g = 0.0
        i_g = 0.0
        for s in range(nlos):
            if tau_g >= tau_stop:
                break
            idx = nlos - 1 - s if positive_direction else s
            kk, jj, ii = _cell_index(row, col, idx, direction_code)
            r = rho_grid[kk, jj, ii]
            t = temp_grid[kk, jj, ii]
            if not (r > 0.0 and t > 0.0):
                continue
            alpha_r = mesa_alpha_scalar(
                r,
                t,
                low_log_t,
                low_log_r,
                low_log_kappa,
                high_log_t,
                high_log_r,
                high_log_kappa,
            )
            bbol = SIGMA_OVER_PI * t**4
            if thermalization:
                if use_table:
                    ne, n_h1, n_h2, n_he1, n_he2, n_he3 = table_state(
                        r, t, lr0, dlr, lt0, dlt, lnf, nh_per_rho, nhe_per_rho
                    )
                else:
                    ne, n_h1, n_h2, n_he1, n_he2, n_he3 = saha_state(r, t)
                alpha_s = electron_scattering(ne)
                a_s = min(alpha_s, alpha_r)
                a_a = max(alpha_r - a_s, 1.0e-6 * alpha_r)
                eps = a_a / (a_a + a_s)
                dtau = math.sqrt(a_a * (a_a + a_s)) * ds_cm
                i_g += (
                    (2.0 * math.sqrt(eps) / (1.0 + math.sqrt(eps)))
                    * bbol
                    * math.exp(-tau_g)
                    * (-math.expm1(-dtau))
                )
            else:
                dtau = alpha_r * ds_cm
                i_g += bbol * math.exp(-tau_g) * (-math.expm1(-dtau))
            tau_g += dtau
        intensity[row, col] = i_g
        tau_map[row, col] = tau_g
    return intensity, tau_map


@njit(parallel=True, cache=True)
def integrate_multifreq_rays(
    rho_grid,
    temp_grid,
    low_log_t,
    low_log_r,
    low_log_kappa,
    high_log_t,
    high_log_r,
    high_log_kappa,
    direction_code,
    positive_direction,
    ds_cm,
    nus,
    scattering,
    grey_limit,
    tau_stop,
    use_table=False,
    lr0=0.0,
    dlr=1.0,
    lt0=0.0,
    dlt=1.0,
    lnf=_NO_LNF,
    nh_per_rho=0.0,
    nhe_per_rho=0.0,
):
    """Multifrequency H/He continuum transfer.

    Returns
    -------
    spec : (nf, nrow) float64
        I_nu summed over the columns of each image row (sum over rows gives the
        image-integrated I_nu; multiply by 4 pi dA for L_nu).
    intensity : (nrow, ncol) float64
        Frequency-integrated intensity int I_nu dnu per pixel (trapezoid in ln nu).
    """
    nz, ny, nx = rho_grid.shape
    nrow, ncol, nlos = _image_shape(nz, ny, nx, direction_code)
    nf = nus.size
    spec = np.zeros((nf, nrow))
    intensity = np.zeros((nrow, ncol))
    lognu = np.log(nus)
    weight = np.empty(nf)  # trapezoid in ln(nu): dnu = nu dln(nu)
    for f in range(nf):
        left = lognu[f] - lognu[f - 1] if f > 0 else 0.0
        right = lognu[f + 1] - lognu[f] if f < nf - 1 else 0.0
        weight[f] = 0.5 * (left + right) * nus[f]

    for row in prange(nrow):
        tau = np.zeros(nf)
        inten = np.zeros(nf)
        for col in range(ncol):
            tau[:] = 0.0
            inten[:] = 0.0
            for s in range(nlos):
                idx = nlos - 1 - s if positive_direction else s
                kk, jj, ii = _cell_index(row, col, idx, direction_code)
                r = rho_grid[kk, jj, ii]
                t = temp_grid[kk, jj, ii]
                if not (r > 0.0 and t > 0.0):
                    continue
                alpha_r = mesa_alpha_scalar(
                    r,
                    t,
                    low_log_t,
                    low_log_r,
                    low_log_kappa,
                    high_log_t,
                    high_log_r,
                    high_log_kappa,
                )
                if use_table:
                    ne, n_h1, n_h2, n_he1, n_he2, n_he3 = table_state(
                        r, t, lr0, dlr, lt0, dlt, lnf, nh_per_rho, nhe_per_rho
                    )
                else:
                    ne, n_h1, n_h2, n_he1, n_he2, n_he3 = saha_state(r, t)
                alpha_s = electron_scattering(ne)
                active = False
                for f in range(nf):
                    if tau[f] >= tau_stop:
                        continue
                    active = True
                    nu = nus[f]
                    bnu = planck_nu(nu, t)
                    if grey_limit:
                        a_a = alpha_r
                    else:
                        a_a = continuum_absorption(
                            nu, t, ne, n_h1, n_h2, n_he1, n_he2, n_he3
                        )
                    if scattering:
                        eps = a_a / (a_a + alpha_s) if a_a + alpha_s > 0.0 else 1.0
                        dtau = math.sqrt(a_a * (a_a + alpha_s)) * ds_cm
                        inten[f] += (
                            (2.0 * math.sqrt(eps) / (1.0 + math.sqrt(eps)))
                            * bnu
                            * math.exp(-tau[f])
                            * (-math.expm1(-dtau))
                        )
                    else:
                        dtau = a_a * ds_cm
                        inten[f] += bnu * math.exp(-tau[f]) * (-math.expm1(-dtau))
                    tau[f] += dtau
                if not active:
                    break
            total = 0.0
            for f in range(nf):
                spec[f, row] += inten[f]
                total += inten[f] * weight[f]
            intensity[row, col] = total
    return spec, intensity


def effective_temperature(intensity: np.ndarray) -> np.ndarray:
    """T_eff = (pi I / sigma_SB)^(1/4) per pixel; NaN where I <= 0."""
    intensity = np.asarray(intensity, dtype=np.float64)
    out = np.full(intensity.shape, np.nan, dtype=np.float64)
    positive = intensity > 0.0
    out[positive] = (math.pi * intensity[positive] / SIGMA_SB_CGS) ** 0.25
    return out
