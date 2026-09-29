"""Kernel tests: analytic slabs, the grey-limit identity and the opacity helpers."""

import math

import numpy as np
import pytest

from athenak_rt.constants import SIGMA_OVER_PI, SIGMA_SB_CGS, SIGMA_THOMSON_CGS
from athenak_rt.opacity import (
    DEFAULT_MESA_HIGH_T,
    DEFAULT_MESA_LOW_T,
    MesaOpacityModel,
    continuum_absorption,
    frequency_quadrature_weights,
    mesa_alpha_scalar,
    photon_energy_grid,
    planck_nu,
    saha_state,
)
from athenak_rt.transfer import (
    DIRECTION_X,
    DIRECTION_Y,
    DIRECTION_Z,
    effective_temperature,
    integrate_grey_rays,
    integrate_multifreq_rays,
    integrate_tau1_rays,
)

KAPPA = 0.4  # cm^2/g, constant grey opacity used by the slab tests
RHO = 1.0e-8  # g/cm^3
TEMP = 2.0e4  # K
NLOS = 40


def constant_model():
    return MesaOpacityModel.constant_kappa(KAPPA)


def slab(nlos=NLOS, nrow=2, ncol=3, direction=DIRECTION_Z, rho=RHO, temp=TEMP):
    """Uniform slab as a (nz, ny, nx) cube with the LOS along ``direction``."""
    if direction == DIRECTION_Z:
        shape = (nlos, nrow, ncol)
    elif direction == DIRECTION_Y:
        shape = (nrow, nlos, ncol)
    else:
        shape = (nrow, ncol, nlos)
    return np.full(shape, rho), np.full(shape, temp)


def axes_for(shape):
    nz, ny, nx = shape
    return (
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
    )


# --------------------------------------------------------------------------- opacity
def test_constant_table_reproduces_kappa_rho():
    args = constant_model().kernel_args
    for rho, temp in ((1e-10, 3e3), (1e-6, 2e4), (1e-2, 5e6)):
        assert mesa_alpha_scalar(rho, temp, *args) == pytest.approx(
            KAPPA * rho, rel=1e-12
        )
    assert mesa_alpha_scalar(0.0, 1e4, *args) == 0.0
    assert mesa_alpha_scalar(1e-6, 0.0, *args) == 0.0


def test_mesa_tables_load_and_are_finite():
    model = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T)
    assert model.low_t.log_kappa.shape == (model.low_t.log_t.size, model.low_t.log_r.size)
    assert model.high_t.log_kappa.shape == (
        model.high_t.log_t.size,
        model.high_t.log_r.size,
    )
    assert np.all(np.isfinite(model.low_t.log_kappa))
    assert np.all(np.isfinite(model.high_t.log_kappa))
    # Fully ionized hot gas: close to electron scattering, 0.2 (1 + X) = 0.34.
    alpha = mesa_alpha_scalar(1e-9, 3e6, *model.kernel_args)
    assert alpha / 1e-9 == pytest.approx(0.34, rel=0.1)


def test_saha_charge_neutrality_and_limits():
    for rho, temp in ((1e-10, 8e3), (1e-8, 2e4), (1e-6, 1e5), (1e-4, 1e6)):
        ne, n_h1, n_h2, n_he1, n_he2, n_he3 = saha_state(rho, temp)
        assert ne == pytest.approx(n_h2 + n_he2 + 2.0 * n_he3, rel=1e-8)
        assert n_h1 + n_h2 == pytest.approx(0.7 * rho / 1.67262192e-24, rel=1e-12)
        assert n_he1 + n_he2 + n_he3 == pytest.approx(
            0.3 * rho / (4 * 1.67262192e-24), rel=1e-12
        )
    ne, n_h1, n_h2, n_he1, n_he2, n_he3 = saha_state(1e-10, 2e6)
    assert ne == pytest.approx(n_h2 + 2.0 * n_he3, rel=1e-6)  # fully ionized
    assert n_h1 / (n_h1 + n_h2) < 1e-8
    ne, n_h1, n_h2, *_ = saha_state(1e-8, 2.5e3)
    assert n_h2 / (n_h1 + n_h2) < 1e-10  # neutral


def test_continuum_absorption_edges_and_stimulated_emission():
    rho, temp = 1e-9, 1.5e4
    state = saha_state(rho, temp)
    nu_h = 13.598 * 1.602176634e-12 / 6.62607015e-27
    below = continuum_absorption(0.999 * nu_h, temp, *state)
    above = continuum_absorption(1.001 * nu_h, temp, *state)
    assert above > 10.0 * below  # Lyman edge
    assert below > 0.0 and math.isfinite(above)
    # Free-free only in the far Rayleigh-Jeans tail: alpha ~ nu^-3 (1 - e^-x) ~ nu^-2.
    nu1, nu2 = 1e11, 2e11
    ratio = continuum_absorption(nu1, temp, *state) / continuum_absorption(
        nu2, temp, *state
    )
    assert ratio == pytest.approx(4.0, rel=2e-3)


def test_photon_energy_grid_and_planck_quadrature():
    energies, nus = photon_energy_grid(73, 0.1, 1000.0)
    assert energies.size == 73
    assert energies[0] == pytest.approx(0.1) and energies[-1] == pytest.approx(1000.0)
    _, nus = photon_energy_grid(800, 1e-3, 1e5)
    weights = frequency_quadrature_weights(nus)
    for temp in (8e3, 3e4, 2e5):
        integral = sum(planck_nu(nu, temp) * w for nu, w in zip(nus, weights))
        assert integral == pytest.approx(SIGMA_OVER_PI * temp**4, rel=1e-4)


def test_effective_temperature_inverts_stefan_boltzmann():
    intensity = np.array(
        [[SIGMA_OVER_PI * 1.0e4**4, 0.0], [SIGMA_OVER_PI * 3.3e5**4, -1.0]]
    )
    teff = effective_temperature(intensity)
    assert teff[0, 0] == pytest.approx(1.0e4, rel=1e-12)
    assert teff[1, 0] == pytest.approx(3.3e5, rel=1e-12)
    assert np.isnan(teff[0, 1]) and np.isnan(teff[1, 1])


# --------------------------------------------------------------------------- slabs
@pytest.mark.parametrize(
    "direction,positive",
    [
        (DIRECTION_Z, True),
        (DIRECTION_Z, False),
        (DIRECTION_Y, True),
        (DIRECTION_Y, False),
        (DIRECTION_X, True),
        (DIRECTION_X, False),
    ],
)
def test_uniform_slab_grey_formal(direction, positive):
    rho, temp = slab(direction=direction)
    ds = 3.0 / (KAPPA * RHO * NLOS)  # total tau = 3
    intensity, tau = integrate_grey_rays(
        rho, temp, *constant_model().kernel_args, direction, positive, ds, False, 1e9
    )
    expected = SIGMA_OVER_PI * TEMP**4 * (1.0 - math.exp(-3.0))
    assert intensity.shape == (2, 3)
    assert np.allclose(intensity, expected, rtol=1e-12)
    assert np.allclose(tau, 3.0, rtol=1e-12)


def test_uniform_slab_tau_stop_truncation():
    rho, temp = slab()
    dtau = 0.25
    ds = dtau / (KAPPA * RHO)
    intensity, tau = integrate_grey_rays(
        rho, temp, *constant_model().kernel_args, DIRECTION_Z, True, ds, False, 1.9
    )
    # Cells are added while tau < tau_stop = 1.9, i.e. exactly 8 cells of dtau = 0.25
    # (1.9 rather than 2.0 keeps the test away from the exact roundoff boundary).
    expected = SIGMA_OVER_PI * TEMP**4 * (1.0 - math.exp(-8 * dtau))
    assert np.allclose(intensity, expected, rtol=1e-12)
    assert np.allclose(tau, 2.0, rtol=1e-12)


def test_two_layer_slab_orders_layers_from_the_observer():
    rho, temp = slab(nlos=NLOS)
    t_top, t_bottom = 3.0e4, 1.0e4
    temp[NLOS // 2 :] = t_top  # high-z half
    temp[: NLOS // 2] = t_bottom  # low-z half
    ds = 4.0 / (KAPPA * RHO * NLOS)  # tau = 2 per layer
    args = constant_model().kernel_args
    i_from_top, _ = integrate_grey_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, False, 1e9
    )
    i_from_bottom, _ = integrate_grey_rays(
        rho, temp, *args, DIRECTION_Z, False, ds, False, 1e9
    )

    def expect(near, far):
        b_near, b_far = SIGMA_OVER_PI * near**4, SIGMA_OVER_PI * far**4
        return b_near * (1 - math.exp(-2.0)) + b_far * (1 - math.exp(-2.0)) * math.exp(
            -2.0
        )

    assert np.allclose(i_from_top, expect(t_top, t_bottom), rtol=1e-12)
    assert np.allclose(i_from_bottom, expect(t_bottom, t_top), rtol=1e-12)


def test_uniform_slab_tau1_photosphere():
    rho, temp = slab()
    x, y, z = axes_for(rho.shape)
    args = constant_model().kernel_args
    ds = 0.3 / (KAPPA * RHO)  # dtau = 0.3 per cell: tau >= 1 in the 4th cell
    area = 2.5
    tau_total, tph, coord, flux, valid = integrate_tau1_rays(
        rho, temp, x, y, z, *args, DIRECTION_Z, True, ds, area, 1.0
    )
    assert valid.all()
    assert np.allclose(tph, TEMP)
    assert np.allclose(coord, z[NLOS - 4])  # rays enter from +z
    assert np.allclose(flux, 4.0 * SIGMA_SB_CGS * TEMP**4 * area, rtol=1e-12)
    assert np.allclose(tau_total, 0.3 * NLOS, rtol=1e-12)
    tau_total, tph, coord, flux, valid = integrate_tau1_rays(
        rho, temp, x, y, z, *args, DIRECTION_Z, False, ds, area, 1.0
    )
    assert np.allclose(coord, z[3])  # rays enter from -z
    # Optically thin slab: no photosphere, no emission.
    thin_ds = 0.5 / (KAPPA * RHO * NLOS)
    tau_total, tph, coord, flux, valid = integrate_tau1_rays(
        rho, temp, x, y, z, *args, DIRECTION_Z, True, thin_ds, area, 1.0
    )
    assert not valid.any()
    assert np.all(np.isnan(tph)) and np.all(flux == 0.0)
    assert np.allclose(tau_total, 0.5, rtol=1e-12)


def test_tau1_formal_agreement_for_optically_thick_isothermal_slab():
    """For a thick isothermal slab the tau=1 blackbody and the formal solution agree."""
    rho, temp = slab()
    x, y, z = axes_for(rho.shape)
    args = constant_model().kernel_args
    ds = 30.0 / (KAPPA * RHO * NLOS)
    *_, flux, valid = integrate_tau1_rays(
        rho, temp, x, y, z, *args, DIRECTION_Z, True, ds, 1.0, 1.0
    )
    intensity, _ = integrate_grey_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, False, 1e9
    )
    assert valid.all()
    assert np.allclose(4.0 * math.pi * intensity, flux, rtol=1e-12)


def test_multifreq_slab_absorption_and_thermalization():
    rho, temp = slab(nrow=1, ncol=1)
    _, nus = photon_energy_grid(40, 0.5, 200.0)
    args = constant_model().kernel_args
    length = 1.0e9
    ds = length / NLOS
    state = saha_state(RHO, TEMP)
    alpha_s = SIGMA_THOMSON_CGS * state[0]
    alpha_nu = np.array([continuum_absorption(nu, TEMP, *state) for nu in nus])
    bnu = np.array([planck_nu(nu, TEMP) for nu in nus])

    spec, intensity = integrate_multifreq_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, nus, False, False, 1e9
    )
    expected = bnu * (1.0 - np.exp(-alpha_nu * length))
    assert np.allclose(spec[:, 0], expected, rtol=1e-10)
    weights = frequency_quadrature_weights(nus)
    assert intensity[0, 0] == pytest.approx(float(np.sum(expected * weights)), rel=1e-12)

    spec, _ = integrate_multifreq_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, nus, True, False, 1e9
    )
    eps = alpha_nu / (alpha_nu + alpha_s)
    tau_star = np.sqrt(alpha_nu * (alpha_nu + alpha_s)) * length
    expected = 2.0 * np.sqrt(eps) / (1.0 + np.sqrt(eps)) * bnu * (1.0 - np.exp(-tau_star))
    assert np.allclose(spec[:, 0], expected, rtol=1e-10)
    assert alpha_s > 0.0 and np.all(eps < 1.0)


# --------------------------------------------------------------------------- grey limit
def random_cube(seed=0, shape=(12, 3, 4)):
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-11, -7, size=shape)
    temp = 10.0 ** rng.uniform(3.8, 5.5, size=shape)
    rho[rng.random(shape) < 0.15] = 0.0  # empty cells must be transparent
    return rho, temp


def test_grey_limit_identity_matches_numpy_reference():
    """multifreq kernel with alpha_nu = alpha_R equals a numpy transcription exactly."""
    rho, temp = random_cube()
    model = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T)
    args = model.kernel_args
    _, nus = photon_energy_grid(50, 0.1, 1000.0)
    weights = frequency_quadrature_weights(nus)
    ds = 3.0e10
    spec, intensity = integrate_multifreq_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, nus, False, True, 1e9
    )
    nz, ny, nx = rho.shape
    ref_spec = np.zeros((nus.size, ny))
    ref_int = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            tau = 0.0
            inten = np.zeros(nus.size)
            for k in range(nz - 1, -1, -1):
                r, t = rho[k, j, i], temp[k, j, i]
                if not (r > 0.0 and t > 0.0):
                    continue
                dtau = mesa_alpha_scalar(r, t, *args) * ds
                b = np.array([planck_nu(nu, t) for nu in nus])
                inten += b * math.exp(-tau) * (-math.expm1(-dtau))
                tau += dtau
            ref_spec[:, j] += inten
            ref_int[j, i] = float(np.sum(inten * weights))
    assert np.allclose(spec, ref_spec, rtol=1e-12, atol=0.0)
    assert np.allclose(intensity, ref_int, rtol=1e-12, atol=0.0)


def test_grey_limit_converges_to_grey_formal():
    """With a fine frequency grid the grey-limit multifreq solution is the grey one."""
    rho, temp = random_cube(seed=3)
    model = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T)
    args = model.kernel_args
    _, nus = photon_energy_grid(800, 1e-3, 1e5)
    ds = 3.0e10
    for direction, positive in (
        (DIRECTION_Z, True),
        (DIRECTION_Y, False),
        (DIRECTION_X, True),
    ):
        _, mf_intensity = integrate_multifreq_rays(
            rho, temp, *args, direction, positive, ds, nus, False, True, 1e9
        )
        grey_intensity, _ = integrate_grey_rays(
            rho, temp, *args, direction, positive, ds, False, 1e9
        )
        assert grey_intensity.shape == mf_intensity.shape
        assert np.count_nonzero(grey_intensity > 0.0) > 0.9 * grey_intensity.size
        assert np.allclose(mf_intensity, grey_intensity, rtol=1e-4, atol=0.0)


def test_grey_therm_reduces_to_formal_for_neutral_gas():
    """Below ~3000 K there are no free electrons, so the thermalization depth is tau_R."""
    rho, temp = random_cube(seed=5)
    temp = np.full_like(temp, 2.5e3)
    args = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T).kernel_args
    ds = 1.0e10
    formal, _ = integrate_grey_rays(rho, temp, *args, DIRECTION_Z, True, ds, False, 1e9)
    therm, _ = integrate_grey_rays(rho, temp, *args, DIRECTION_Z, True, ds, True, 1e9)
    assert np.allclose(therm, formal, rtol=1e-9)


def test_grey_therm_is_darker_than_formal_when_scattering_dominates():
    rho, temp = slab(nrow=1, ncol=1, rho=1e-9, temp=1e5)  # fully ionized, kappa_R ~ 0.34
    args = MesaOpacityModel(DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T).kernel_args
    ds = 3.0e11
    formal, tau_f = integrate_grey_rays(
        rho, temp, *args, DIRECTION_Z, True, ds, False, 1e9
    )
    therm, tau_t = integrate_grey_rays(rho, temp, *args, DIRECTION_Z, True, ds, True, 1e9)
    assert tau_t[0, 0] < tau_f[0, 0]
    assert therm[0, 0] < formal[0, 0]
