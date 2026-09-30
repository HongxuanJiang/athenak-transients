# athenak_rt — radiative post-processing of AthenaK snapshots

`athenak_rt` turns AthenaK `.bin` hydro dumps, for example of a tidal disruption event, into
line-of-sight luminosities, spectra, image-plane maps and light curves.  It streams the
MeshBlocks of the dump (a 16 GB snapshot is never held in memory), resamples
the finest AMR level of the dense gas onto a uniform Cartesian ray grid,
recovers the temperature from the tabulated LTE equation of state the run used,
and integrates the transfer equation along parallel rays with numba.

The physics and numerics are those of the scripts validated for the paper
(Yang et al. 2026; `light_curve_tde_cartesian.py` and
`rt_continuum_experiments.py` in the reproducibility archive).  The package
only replaces their hard-coded configuration with a parameter file.

## Modes

| mode         | what is computed                                                                                     | product                              |
|--------------|-------------------------------------------------------------------------------------------------------|--------------------------------------|
| `tau1`       | grey Rosseland (MESA/OPAL) optical depth; the first cell with tau >= tau_ph is the photosphere and emits 4 sigma T_ph^4 dA (paper, Sec. 9) | T_ph, tau, photosphere position, band L_nu (blackbody) |
| `grey`       | grey formal solution I = int B(T) e^{-tau_R} dtau_R                                                   | I, T_eff, tau                        |
| `grey-therm` | grey with scattering/absorption split: kappa_s = sigma_T n_e / rho, kappa_a = kappa_R - kappa_s, dtau* = sqrt(a_a (a_a + a_s)) ds, source weight 2 sqrt(eps)/(1 + sqrt(eps)), eps = a_a/(a_a + a_s) | I, T_eff, tau*                       |
| `multifreq`  | (default) H/He continuum: populations from the EOS table (`populations = saha` for ideal Saha), free-free + bound-free (H n <= 6, He I, He II) absorption with stimulated emission, electron scattering through the same thermalization-depth treatment per frequency, on a log-spaced photon-energy grid (default 73 points, 0.1–1000 eV) | I, T_eff, L_nu(E), band L_nu         |

`scattering = false` turns `multifreq` into pure absorption.  All luminosities are
projected, isotropic-equivalent values for the chosen line of sight
(L = 4 pi sum I dA; for a blackbody pixel L_bol = 4 sigma T^4 dA).
`T_eff = (pi I / sigma)^{1/4}` per pixel.

Every mode uses the same cell selection (`rho_code > factor * dfloor`, strict,
default factor 10, both for the bounding box and the resampling), the same
padded bounding box, the same black-hole excision mask (radius
`<problem>/bh_excise_radius` from the header) and the same MESA Rosseland
opacity.  The integrals of the `grey*`/`multifreq` modes stop once the optical
depth exceeds `tau_stop` (30).

## Usage

The package lives in the top-level `athenak_rt/` directory; run it from the repository root
or put the repository root on `PYTHONPATH`.  Requirements: numpy, numba, h5py (pytest for the tests).

```
cd /path/to/athenak
python -m athenak_rt PARAMETER_FILE
```

The parameter file is the only argument (besides `--help` and `--version`).  It uses the
AthenaK input format: `<block>` lines, `key = value` lines, `#` comments.  Unknown blocks
or keys, keys given twice and unreadable values are errors that name the line.  Keys
left out take their defaults; only `<input>/dump_dir` and `<input>/dumps` are required.
Relative paths are relative to the parameter file's directory.  A minimal file:

```
<input>
dump_dir = /path/to/run/bin      # directory with <basename>.<variable>.NNNNN.bin
dumps    = 300:327               # 327 | 310, 327 | 300:327 | 300:327:3 | all

<transfer>
mode       = multifreq
directions = z, -y
```

`examples/tde_snapshot.rtin` (the paper's snapshot: dump 327, `multifreq`, `z` and `-y`,
1024^2 pixels, 1024 samples per ray, 73 energies from 0.1 to 1000 eV, threshold
10 x dfloor, EOS populations) and `examples/tde_lightcurve.rtin` (dumps 300-327) list every
key with a comment.

| Block | Keys (default) |
|---|---|
| `<input>` | `dump_dir` (required), `basename` (`TDEExternalLTEPrad`), `variable` (`hydro_w`), `dumps` (required), `missing_dumps` (`error`, or `skip` with a warning) |
| `<output>` | `dir` (`athenak_rt_output`), `hdf5_compression` (`lzf`, `gzip`, `none`), `bands` (`true`) |
| `<transfer>` | `mode` (`multifreq`), `directions` (`z`), `tau_photosphere` (1.0), `tau_stop` (30.0), `nfreq` (73), `emin_ev` (0.1), `emax_ev` (1000.0), `scattering` (`true`), `populations` (`eos`, or `saha`) |
| `<image>` | `image_size` (1024), `los_steps` (1024), `grid_dtype` (`auto`, `float32`, `float64`), `box` (`auto` or six numbers), `auto_box_padding_code` (0.25), `auto_box_padding_fraction` (0.1), `auto_box_min_width` (1.0), `auto_box_clip_to_mesh` (`true`) |
| `<selection>` | `density_threshold_factor` (10.0), `density_threshold_code` (`auto`), `bh_mask` (`true`), `bh_mask_radius` (`auto`: header value) |
| `<tables>` | `eos_table` (`auto`: `<hydro>/table` of the dump header, also tried relative to the run directory, the parent of `dump_dir`, and in `$ATHENAK_EOS_TABLE_DIR`), `mesa_high_t`, `mesa_low_t` (`auto`: the tables in `data/`, OPAL GS98 X=0.7 Z=0.02 and Ferguson et al. 2005 low-T; see `data/NOTICE`) |
| `<run>` | `threads` (`auto`: all CPUs), `skip_existing` (`false`) |

`grid_dtype = auto` stores the resampled cube in float32 once any axis reaches 1024
samples, as in the production runs, and in float64 below.  The full table with types and
meanings is in `docs/athenak_rt.md`.

The Python API mirrors the parameter file: `athenak_rt.run_parameter_file(path)`,
`athenak_rt.load_parameter_file(path)` (returns the settings, directions and dump list),
`athenak_rt.run(settings, dumps, directions)` and `athenak_rt.process_snapshot(...)`; the
kernels in `athenak_rt.transfer` can be used directly on any `(nz, ny, nx)` cube of
(rho [g/cm^3], T [K]).

## Output

One HDF5 file per dump, mode and direction,
`<dump stem>.rt_<mode>_<direction>.h5`.  Root attributes record every
setting (`RTSettings.as_attributes()`), the resolved table paths, the dump
time/cycle, the ray box (`rt_box_code`, `auto_box_raw_code`), the density
thresholds, the BH position and mask radius, the units, the pixel area, the
numba thread count and the package version.  Datasets:

```
parameters/parameter_file, resolved         the parameter file as read, and with every
                                            value used (itself a valid parameter file)
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

With more than one dump the run also writes, after every dump, the light curve
`rt_lightcurve_<mode>.csv`: plain CSV, one row per dump, columns `dump, cycle,
time_code, time_s, L_bol_<dir>...` and, with band products, `nuLnu_<band>_<dir>...`
(erg/s; `-y` is written `minus_y`).  `rt_lightcurve_<mode>.h5` holds the same rows
plus per-direction diagnostics (`valid_pixels`, `max_tau`, temperatures, BH position),
the band `L_nu` and the multifrequency spectra of all dumps, and the parameter file.

Products are written as `*.part` and renamed when complete.  With
`<run>/skip_existing = true` existing products are reused when their recorded settings
match (the run stops otherwise), so an interrupted light-curve run can be resubmitted
unchanged.

## Layout

```
athenak_rt/
  __init__.py, __main__.py, cli.py     entry point
  params.py                            parameter-file parser, dump selection
  config.py                            RTSettings (recorded in every product)
  snapshot.py                          .bin header, MeshBlock streaming, dense-gas box,
                                       ray-grid resampling, BH mask
  eos.py                               TabulatedLteTable (from plot_slice.py), table lookup
  opacity.py                           MESA tables + grey alpha_R, Saha, H/He continuum,
                                       photon-energy grid
  transfer.py                          numba kernels: tau1, grey/grey-therm, multifreq,
                                       blackbody band maps
  bands.py                             observation bands
  output.py                            HDF5 and light-curve writers
  pipeline.py                          per-dump driver and run loop
  data/                                MESA opacity tables (+ NOTICE)
  examples/                            annotated parameter files
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
(header parsing, refinement-aware resampling, BH mask, HDF5 attributes); and the
parameter file (parser and error messages, defaults, dump ranges and missing dumps,
the examples, single-dump and light-curve runs, restart with `skip_existing`).

## Validation against the archived scripts

At 192 x 192 x 192 on snapshot 00327 of the paper's run, the package reproduces
the luminosities of the original scripts (`light_curve_tde_cartesian.process_snapshot`
for `tau1`, `rt_continuum_experiments.py` for the others) to better than
1e-9 relative for every mode and both lines of sight (z and -y); see the
commit message.  For that comparison `grid_dtype = float32` reproduces the
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
