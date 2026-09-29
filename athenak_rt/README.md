# athenak_rt — radiative post-processing of AthenaK snapshots

`athenak_rt` turns an AthenaK `.bin` hydro dump, for example of a tidal disruption event, into
line-of-sight luminosities, spectra and image-plane maps.  It streams the
MeshBlocks of the dump (a 16 GB snapshot is never held in memory), resamples
the finest AMR level of the dense gas onto a uniform Cartesian ray grid,
recovers the temperature from the tabulated LTE equation of state the run used,
and integrates the transfer equation along parallel rays with numba.

The physics and numerics are those of the scripts validated for the paper
(Yang et al. 2026; `light_curve_tde_cartesian.py` and
`rt_continuum_experiments.py` in the reproducibility archive).  The package
only replaces their hard-coded configuration with a command line.

## Modes

| mode         | what is computed                                                                                     | product                              |
|--------------|-------------------------------------------------------------------------------------------------------|--------------------------------------|
| `tau1`       | grey Rosseland (MESA/OPAL) optical depth; the first cell with tau >= tau_ph is the photosphere and emits 4 sigma T_ph^4 dA (paper, Sec. 9) | T_ph, tau, photosphere position, band L_nu (blackbody) |
| `grey`       | grey formal solution I = int B(T) e^{-tau_R} dtau_R                                                   | I, T_eff, tau                        |
| `grey-therm` | grey with scattering/absorption split: kappa_s = sigma_T n_e / rho, kappa_a = kappa_R - kappa_s, dtau* = sqrt(a_a (a_a + a_s)) ds, source weight 2 sqrt(eps)/(1 + sqrt(eps)), eps = a_a/(a_a + a_s) | I, T_eff, tau*                       |
| `multifreq`  | (default) H/He continuum: Saha ionization, free-free + bound-free (H n <= 6, He I, He II) absorption with stimulated emission, electron scattering through the same thermalization-depth treatment per frequency, on a log-spaced photon-energy grid (default 73 points, 0.1–1000 eV) | I, T_eff, L_nu(E), band L_nu         |

`--no-scattering` turns `multifreq` into pure absorption.  All luminosities are
projected, isotropic-equivalent values for the chosen line of sight
(L = 4 pi sum I dA; for a blackbody pixel L_bol = 4 sigma T^4 dA).
`T_eff = (pi I / sigma)^{1/4}` per pixel.

Every mode uses the same cell selection (`rho_code > factor * dfloor`, strict,
default factor 10, both for the bounding box and the resampling), the same
padded bounding box, the same black-hole excision mask (radius
`<problem>/bh_excise_radius` from the header) and the same MESA Rosseland
opacity.  The integrals of the `grey*`/`multifreq` modes stop once the optical
depth exceeds `--tau-stop` (30).

## Usage

The package lives in the top-level `athenak_rt/` directory; run it from the repository root
or put the repository root on `PYTHONPATH`.  Requirements: numpy, numba, h5py (pytest for the tests).

```
cd /path/to/athenak
python -m athenak_rt SNAPSHOT.bin --direction z --mode multifreq \
    --image-size 1024 --los-steps 1024 \
    --eos-table /path/to/chabrier2021_t13_helm_union_prad_640.table \
    --density-threshold-factor 10 --threads 16 --out OUTDIR
```

* `--direction`: `x`, `y`, `z` or `-x`, `-y`, `-z` (observer on that face).
  Negative directions must be written `--direction=-y` (argparse would
  otherwise read `-y` as an option).
* `--eos-table`: defaults to the `<hydro>/table` path recorded in the snapshot
  header if it exists on this machine, else its basename under
  `$ATHENAK_EOS_TABLE_DIR`; otherwise the path must be given.
* `--mesa-high-t` / `--mesa-low-t`: default to the tables in `athenak_rt/data/`
  (OPAL GS98 X=0.7 Z=0.02 and Ferguson et al. 2005 low-T; see `data/NOTICE`).
* `--nfreq`, `--emin`, `--emax` (eV): multifrequency grid; `--tau-photosphere`
  (tau1); `--tau-stop`; `--no-bh-mask`, `--bh-mask-radius`; `--box XMIN XMAX
  YMIN YMAX ZMIN ZMAX` to fix the ray-grid box; `--density-threshold-code` to
  bypass the dfloor rule; `--grid-dtype auto|float32|float64` (storage of the
  resampled cube: `auto` = float32 once any axis reaches 1024 samples, as in the
  production runs, float64 below); `--threads` (numba); `--hdf5-compression`.
* Several snapshots may be given; a light curve
  (`rt_lightcurve_<mode>_<direction>.dat`) and a stacked summary
  (`rt_summary_<mode>_<direction>.h5`) are written across them.

The Python API mirrors the CLI: `athenak_rt.RTSettings`, `athenak_rt.run(settings,
snapshots)` and `athenak_rt.process_snapshot(...)`; the kernels in
`athenak_rt.transfer` can be used directly on any `(nz, ny, nx)` cube of
(rho [g/cm^3], T [K]).

## Output

One HDF5 file per snapshot, mode and direction,
`<snapshot stem>.rt_<mode>_<direction>.h5`.  Root attributes record every
setting (`RTSettings.as_attributes()`), the resolved table paths, the snapshot
time/cycle, the ray box (`rt_box_code`, `auto_box_raw_code`), the density
thresholds, the BH position and mask radius, the units, the pixel area, the
numba thread count and the package version.  Datasets:

```
image/image_u_code, image_v_code            pixel column / row coordinates
maps/                                       tau1: tau_total, photosphere_temp_K,
                                                  photosphere_coord_code, valid,
                                                  bolometric_pixel_luminosity_erg_s
                                            grey*: intensity_erg_s_cm2_sr, tau_effective,
                                                  effective_temp_K, valid, bolometric_pixel_...
                                            multifreq: intensity_erg_s_cm2_sr,
                                                  effective_temp_K, valid, bolometric_pixel_...
spectra/  (multifreq)                       energy_ev, frequency_hz, quadrature_weight_hz,
                                            lnu_erg_s_hz, nu_lnu_erg_s,
                                            lnu_row_erg_s_hz (L_nu per image row)
bands/                                      label, category, frequency_hz, wavelength_nm,
                                            energy_ev, lnu_erg_s_hz, nu_lnu_erg_s;
                                            tau1 also lnu_pixel_erg_s_hz (per-pixel maps)
```

Band L_nu is the blackbody value at T_ph in `tau1` and a log-log interpolation
of the multifrequency spectrum in `multifreq`.

## Layout

```
athenak_rt/
  __init__.py, __main__.py, cli.py     entry point and argument parsing
  config.py                            RTSettings (recorded in every product)
  snapshot.py                          .bin header, MeshBlock streaming, dense-gas box,
                                       ray-grid resampling, BH mask
  eos.py                               TabulatedLteTable (from plot_slice.py), table lookup
  opacity.py                           MESA tables + grey alpha_R, Saha, H/He continuum,
                                       photon-energy grid
  transfer.py                          numba kernels: tau1, grey/grey-therm, multifreq,
                                       blackbody band maps
  bands.py                             observation bands
  output.py                            HDF5 / light-curve / summary writers
  pipeline.py                          per-snapshot driver
  data/                                MESA opacity tables (+ NOTICE)
  tests/                               pytest suite
```

## Tests

```
cd /path/to/athenak
python -m pytest athenak_rt/tests
```

They cover: analytic uniform and two-layer slabs (formal solution
I = B(T)(1 - e^{-tau}), tau-stop truncation, tau=1 photosphere and its
agreement with the formal solution for a thick isothermal slab, all six
directions); the multifrequency kernel against per-frequency analytic slabs
with and without scattering; the grey-limit identity (multifrequency kernel
with alpha_nu = alpha_R equals a numpy transcription to 1e-12 and converges to
the grey formal solution with a fine frequency grid); Saha charge neutrality and
ionization limits; the continuum edges; the Planck quadrature; and an
end-to-end run on a synthetic two-level AMR dump with a synthetic EOS table
(header parsing, refinement-aware resampling, BH mask, HDF5 attributes).

## Validation against the archived scripts

At 192 x 192 x 192 on snapshot 00327 of the paper's run, the package reproduces
the luminosities of the original scripts (`light_curve_tde_cartesian.process_snapshot`
for `tau1`, `rt_continuum_experiments.py` for the others) to better than
1e-9 relative for every mode and both lines of sight (z and -y); see the
commit message.  For that comparison `--grid-dtype float32` reproduces the
storage precision `rt_continuum_experiments.py` used at any size; at the
production size (1024) the default `auto` rule already selects float32.

## Not included

* Cloudy / photoionization / line-emission post-processing.
* The native-AMR streaming grey solvers (`light_curve_tde_native_formal.py`,
  `light_curve_tde_adaptive_rays.py`): they are tied to the archived driver's
  configuration object and were not validated for the paper, so they were left
  out rather than shipped untested.
* The analytic (Mengqi-style) and constant-opacity options of the original
  script; the paper used the MESA tables only.
