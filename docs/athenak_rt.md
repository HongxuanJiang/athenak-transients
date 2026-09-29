# Radiative post-processing with `athenak_rt`

`athenak_rt` (the top-level `athenak_rt/` directory) computes line-of-sight luminosities,
spectra, and image-plane maps from AthenaK `.bin` snapshots, for example of a TDE run.  It is the post-processing
described in the section "Radiative post-processing and synthetic observables" of
Jiang et al. (ApJS).  The package reads a snapshot, resamples the finest AMR level of the
dense gas onto a uniform Cartesian ray grid, recovers the gas temperature from the same
tabulated equation of state (EOS) that the simulation used, and integrates the transfer
equation along parallel rays.

This is a post-processing tool for a hydrodynamic calculation without radiation.  Read
[Physics and limits](#physics-and-limits) before quoting luminosities.

A shorter, module-level README is in
[`athenak_rt/README.md`](../athenak_rt/README.md).

## 1. Installation and requirements

The package is pure Python and is not installed into the environment.  Run it from
the repository root, or put the repository root on `PYTHONPATH`.

```bash
python3 -m pip install numpy numba h5py      # add pytest to run the tests
cd /path/to/athenak
python3 -m athenak_rt --help
```

| Requirement | Used for |
|---|---|
| numpy | arrays and the EOS table |
| numba | parallel ray-integration kernels (first call compiles them) |
| h5py | output files |
| pytest | tests only |

Data needed at run time.

| Data | Where | Note |
|---|---|---|
| Snapshot | `bin/<basename>.hydro_w.NNNNN.bin` | Must contain `dens` and `eint` and the input parameters in its header |
| EOS table | the file that the run used, for example `chabrier2021_t13_helm_union_prad_640.table` | Not in git. Download it from the release assets or regenerate it with `scripts/generate_lte_table.py` (see [`README.md`](TDE/README.md#equation-of-state-table)) |
| Opacity tables | `athenak_rt/data/gs98_z0.02_x0.7.data`, `lowT_fa05_gs98_z0.02_x0.7.data` | Included. MESA `kap` files for `X = 0.7`, `Z = 0.02` (OPAL type 1 and Ferguson et al. 2005 low-temperature), see `athenak_rt/data/NOTICE` |

The EOS table is looked up in this order (`athenak_rt/eos.py`).

1. `--eos-table PATH`, which must exist.
2. The path recorded in the snapshot header (`<hydro>/table`), if that file exists on the
   machine.  The decks use a path relative to the run directory (`../chabrier...`),
   which only resolves if you start `athenak_rt` from a directory where it points to the
   table.
3. The basename of the header path inside `$ATHENAK_EOS_TABLE_DIR`.

If none exists the run stops with a list of the places it tried.  The simplest
approach is to pass `--eos-table` explicitly.

## 2. Modes

Select a mode with `--mode`.  All modes share the same cell selection, the same ray
grid, and the same BH excision mask.  Luminosities are projected, isotropic-equivalent
values for the chosen line of sight, `L = 4 pi sum(I dA)`.  For a blackbody pixel,
`L_bol = 4 sigma T^4 dA`.  The effective temperature of a pixel is
`T_eff = (pi I / sigma)^(1/4)`.

| Mode | What is computed | Opacity | Products |
|---|---|---|---|
| `tau1` | The Rosseland optical depth is accumulated along each ray from the observer. The first cell with `tau >= tau_ph` (default 1) is the photosphere, and the pixel emits as a blackbody at the temperature of that cell. | MESA Rosseland (includes electron scattering) | photosphere temperature, coordinate, total `tau`, bolometric pixel luminosity, band `L_nu` (blackbody) |
| `grey` | Grey formal solution `I = int B(T) exp(-tau_R) dtau_R`. | MESA Rosseland | intensity, `tau`, `T_eff`, bolometric pixel luminosity |
| `grey-therm` | Grey solution with a thermalization depth. Scattering is `alpha_s = sigma_T n_e` and absorption `alpha_a = kappa_R rho - alpha_s`. The effective depth is `dtau* = sqrt(alpha_a (alpha_a + alpha_s)) ds` and the source function is weighted by `2 sqrt(eps) / (1 + sqrt(eps))` with `eps = alpha_a / (alpha_a + alpha_s)`. | MESA Rosseland | as `grey` |
| `multifreq` (default) | Hydrogen and helium continuum transfer at each photon energy on a logarithmic grid (default 73 points from 0.1 to 1000 eV). Saha ionization gives the populations of H I, H II, He I, He II, He III. The absorption includes free-free, bound-free from H (levels `n <= 6`), He I and He II, with stimulated emission. Electron scattering enters through the same thermalization-depth treatment per frequency. | H/He continuum, no lines, no metals | intensity, `T_eff`, spectrum `L_nu(E)`, band `L_nu` |

`--no-scattering` turns `multifreq` into pure absorption.  In all modes the integration
along a ray stops once the optical depth exceeds `--tau-stop` (default 30).  In
`multifreq` mode this is checked per photon energy.  The Rosseland tables are loaded
in every mode, and in `multifreq` mode they are not used for the result.

The multifrequency mode is the one adopted in the paper.  The grey `tau1` photosphere
is kept as a reference.  In the paper it overestimates the luminosity by roughly a
factor of five, because near the Planck peak the bound-free opacity is much larger than
the Rosseland mean and the emission emerges from cooler, outer layers.

### Cell selection and geometry

* **Density cut.**  Cells with `rho > factor * dfloor` enter the ray grid (strict
  inequality, default factor 10, or an explicit `--density-threshold-code`).  Other
  cells are treated as empty.
* **Ray-grid box.**  By default the box is the bounding box of the selected cells, padded
  by the larger of `--auto-box-padding-code` (0.25) and `--auto-box-padding-fraction`
  (0.10) times the box width on every side, with a minimum width of
  `--auto-box-min-width` (1.0), and clipped to the mesh.  For the bounding box the
  threshold is `factor * max(hydro/dfloor, problem/rho_floor)` from the header.  With
  the example decks both floors are equal.  `--box` fixes the box explicitly.
* **BH excision.**  The sphere of radius `problem/bh_excise_radius` from the header is
  emptied around the BH position stored in the header.  Use `--bh-mask-radius` to change
  the radius or `--no-bh-mask` to switch it off.
* **Directions.**  `--direction` is `x`, `y`, `z`, `-x`, `-y` or `-z`, and the observer
  sits on that face of the box.  The image rows and columns are `(y, x)` for `z` rays,
  `(z, x)` for `y` rays, and `(z, y)` for `x` rays.  A negative direction must be
  written with an equals sign, `--direction=-y`, otherwise `argparse` reads it as an
  option.
* **Resampling.**  The finest AMR level available at each location is sampled by
  nearest-cell lookup onto `image_size x image_size x los_steps` points.  The
  snapshot is streamed block by block and is never held in memory in full.

## 3. Command line

```
python3 -m athenak_rt SNAPSHOT.bin [SNAPSHOT.bin ...] [options]
```

Several snapshots can be given.  They are processed in order of their file number, and a
light curve and a stacked summary are written across them.

| Option | Default | Meaning |
|---|---|---|
| `--mode` | `multifreq` | `tau1`, `grey`, `grey-therm`, or `multifreq` |
| `--direction` | `z` | Line of sight, `x y z -x -y -z` |
| `--image-size` | `1024` | Pixels across the image plane, must be `>= 2` |
| `--los-steps` | `1024` | Samples along the line of sight, must be `>= 2` |
| `--out` | `athenak_rt_output` | Output directory |
| `--eos-table` | from header | EOS table, see Sec. 1 |
| `--mesa-high-t` | `data/gs98_z0.02_x0.7.data` | MESA/OPAL high-temperature opacity table |
| `--mesa-low-t` | `data/lowT_fa05_gs98_z0.02_x0.7.data` | MESA low-temperature opacity table |
| `--density-threshold-factor` | `10.0` | Cells with `rho_code > factor * dfloor` are used |
| `--density-threshold-code` | none | Explicit threshold in code units, overrides the factor |
| `--box XMIN XMAX YMIN YMAX ZMIN ZMAX` | auto | Fixed ray-grid box in code units |
| `--auto-box-padding-code` | `0.25` | Absolute padding of the automatic box |
| `--auto-box-padding-fraction` | `0.10` | Fractional padding of the automatic box |
| `--auto-box-min-width` | `1.0` | Minimum width of the automatic box |
| `--no-bh-mask` | off | Do not empty the BH excision sphere |
| `--bh-mask-radius` | header value | Excision radius in code units |
| `--tau-photosphere` | `1.0` | `tau1`: optical depth of the photosphere |
| `--tau-stop` | `30.0` | `grey`, `grey-therm`, `multifreq`: stop past this depth |
| `--nfreq` | `73` | `multifreq`: number of log-spaced photon energies |
| `--emin` | `0.1` | `multifreq`: lowest photon energy in eV |
| `--emax` | `1000.0` | `multifreq`: highest photon energy in eV |
| `--no-scattering` | off | `multifreq`: pure absorption |
| `--threads` | all CPUs of the process | Number of numba threads |
| `--grid-dtype` | `auto` | Storage of the resampled `(rho, T)` cube, `auto`, `float32`, or `float64`. `auto` uses `float32` if any axis has 1024 samples or more, else `float64`. EOS recovery and ray sums are always in double precision. |
| `--hdf5-compression` | `lzf` | `lzf`, `gzip`, or `none` |
| `--no-bands` | off | Skip the band products |
| `--version` | | Print the package version |

The program prints one line per snapshot with the bolometric luminosity and the name of
the output file.

## 4. Outputs

Each run writes into `--out`.

| File | Content |
|---|---|
| `<snapshot stem>.rt_<mode>_<direction>.h5` | One file per snapshot, mode, and direction |
| `rt_lightcurve_<mode>_<direction>.dat` | Text table with one row per snapshot, rewritten after each snapshot. Columns: `snapshot_index time_code time_s luminosity_erg_s valid_pixels total_pixels max_tau median_T_K mean_T_K area_pixel_cm2 density_threshold_code bh_x_code bh_y_code bh_z_code file` |
| `rt_summary_<mode>_<direction>.h5` | Stacked `snapshot_index`, `time_code`, `time_s`, `luminosity_bolometric_erg_s`, file names, band `L_nu`, and (multifrequency) spectra of all processed snapshots |

For example, the file for the `-y` view of one snapshot is
`TDEExternalLTEPrad.hydro_w.00327.rt_multifreq_-y.h5`.

Structure of the per-snapshot HDF5 file.

| Group and dataset | Modes | Content |
|---|---|---|
| root attributes | all | Every setting (`RTSettings`), table paths (`eos_table_resolved`, `eos_table_in_header`), `time_code`, `time_s`, `cycle`, `luminosity_bolometric_erg_s`, the normalization (`luminosity_normalization`, `luminosity_bolometric_formula`, `projected_luminosity_factor`), pixel area `area_pixel_cm2`, step `los_step_cm`, valid pixel count, `dfloor_code`, `density_threshold_code`, ray box `rt_box_code` and the raw box `auto_box_raw_code`, `bh_xyz_code`, `bh_mask_radius_code`, unit scales, `grid_storage_dtype`, `numba_threads`, package version |
| `image/image_u_code`, `image/image_v_code` | all | Pixel coordinates along the image columns and rows |
| `maps/tau_total`, `photosphere_temp_K`, `photosphere_coord_code`, `valid`, `bolometric_pixel_luminosity_erg_s` | `tau1` | Photosphere maps |
| `maps/intensity_erg_s_cm2_sr`, `tau_effective`, `effective_temp_K`, `valid`, `bolometric_pixel_luminosity_erg_s` | `grey`, `grey-therm` | Intensity and effective-temperature maps |
| `maps/intensity_erg_s_cm2_sr`, `effective_temp_K`, `valid`, `bolometric_pixel_luminosity_erg_s` | `multifreq` | Frequency-integrated intensity and effective-temperature maps |
| `spectra/energy_ev`, `frequency_hz`, `quadrature_weight_hz`, `lnu_erg_s_hz`, `nu_lnu_erg_s`, `lnu_row_erg_s_hz` | `multifreq` | Spectrum on the photon-energy grid. `lnu_row_erg_s_hz` is `L_nu` per image row. The frequency integral is a trapezoid rule in `ln(nu)`. |
| `bands/label`, `category`, `frequency_hz`, `wavelength_nm`, `energy_ev`, `lnu_erg_s_hz`, `nu_lnu_erg_s` | all unless `--no-bands` | `L_nu` in observation bands. `tau1` uses the blackbody at the photosphere temperature and adds `lnu_pixel_erg_s_hz` (per-pixel maps). `multifreq` interpolates the spectrum log-log. Bands outside the spectrum are 0. `grey` and `grey-therm` write no band data. |

The bands are `nir_J`, `nir_H`, `nir_K` (1250, 1650, 2200 nm), `optical_u`, `_g`, `_r`,
`_i` (355, 475, 622, 763 nm), `uv_UVW1`, `uv_UVM2`, `uv_UVW2`, `uv_FUV150` (260, 224.6,
192.8, 150 nm), and `xray_0p3keV`, `xray_1keV`, `xray_2keV` (`athenak_rt/bands.py`).

Reading a result.

```python
import h5py, numpy as np

with h5py.File("rt_out/TDEExternalLTEPrad.hydro_w.00327.rt_multifreq_z.h5") as f:
    print("t/t0 =", f.attrs["time_code"], " L_bol,iso =", f.attrs["luminosity_bolometric_erg_s"])
    e_ev = f["spectra/energy_ev"][:]
    nu_lnu = f["spectra/nu_lnu_erg_s"][:]
    print("peak of nu L_nu at", e_ev[np.argmax(nu_lnu)], "eV")
    teff = f["maps/effective_temp_K"][:]          # (rows, columns), NaN where I <= 0
```

## 5. Worked examples

The paper's main synthetic-observable result is the multifrequency H/He continuum
transfer of the last high-resolution (HR) snapshot, viewed along `+z` (face-on) and
along `-y` (in-plane, with the nozzle on the near side), with the grey `tau = 1`
photosphere as the reference.  The HR run is not part of the example decks, so
replace `SNAP` by a snapshot of your own run.  The example FID chain of
[`inputs/TDE_examples/README.md`](../inputs/TDE_examples/README.md) produces
snapshots that can be processed the same way.

```bash
cd /path/to/athenak
SNAP=/path/to/run/bin/TDEExternalLTEPrad.hydro_w.NNNNN.bin
EOS=/path/to/chabrier2021_t13_helm_union_prad_640.table

# multifrequency continuum transfer, face-on (from +z) and in-plane (from -y)
python3 -m athenak_rt $SNAP --mode multifreq --direction z \
    --image-size 1024 --los-steps 1024 --eos-table $EOS \
    --density-threshold-factor 10 --threads 16 --out rt_out
python3 -m athenak_rt $SNAP --mode multifreq --direction=-y \
    --image-size 1024 --los-steps 1024 --eos-table $EOS \
    --density-threshold-factor 10 --threads 16 --out rt_out

# grey tau = 1 photosphere as the reference
python3 -m athenak_rt $SNAP --mode tau1 --direction z \
    --image-size 1024 --los-steps 1024 --eos-table $EOS --threads 16 --out rt_out
python3 -m athenak_rt $SNAP --mode tau1 --direction=-y \
    --image-size 1024 --los-steps 1024 --eos-table $EOS --threads 16 --out rt_out
```

Each command prints the bolometric luminosity, and the four products are
`<stem>.rt_multifreq_z.h5`, `<stem>.rt_multifreq_-y.h5`, `<stem>.rt_tau1_z.h5`,
`<stem>.rt_tau1_-y.h5`.  For the last HR snapshot of the paper (`t / P_mb = 2.93`,
`1024^2` pixels, 1024 samples per line of sight) the published values are

| Line of sight | Multifrequency `L_bol,iso` | Grey `tau = 1` `L_bol,iso` |
|---|---|---|
| `+z` | `6.1e42 erg/s` | `3.2e43 erg/s` |
| `-y` | `3.6e42 erg/s` | `2.2e43 erg/s` |

The multifrequency spectrum peaks near 60 eV.  The numbers depend on the snapshot, on
the density threshold that separates debris from the numerical atmosphere, and on the
image sampling.  In the paper, going from `512^2` to `1024^2` pixels changes the grey
luminosities by 0.4% and 6.1%, and doubling the number of photon energies changes the
multifrequency luminosity by less than 3%.

A quick test at low resolution takes a fraction of the cost.

```bash
python3 -m athenak_rt $SNAP --mode multifreq --direction z --image-size 192 --los-steps 192 \
    --eos-table $EOS --out rt_test
```

A time-ordered sequence of snapshots gives a light curve for one line of sight.  The
light curve is written to `rt_lightcurve_multifreq_z.dat` in `--out`.

```bash
python3 -m athenak_rt /path/to/run/bin/TDEExternalLTEPrad.hydro_w.0032?.bin \
    --mode multifreq --direction z --eos-table $EOS --out rt_lc
```

The paper does not present light curves, because the simulation does not reach the
peak of the flare.

### Python interface

```python
from pathlib import Path
import athenak_rt

settings = athenak_rt.RTSettings(mode="multifreq", direction="-y", image_size=512, los_steps=512,
                             eos_table=Path("chabrier2021_t13_helm_union_prad_640.table"),
                             output_dir=Path("rt_out"))
results = athenak_rt.run(settings, [Path("snapshot.bin")])
print(results[0].luminosity, results[0].hdf5_path)
```

`athenak_rt.process_snapshot` handles one snapshot, and the kernels in `athenak_rt.transfer`
accept any `(nz, ny, nx)` cube of density in g/cm^3 and temperature in K.

## 6. Performance and memory

* The snapshot is streamed and never loaded in full.  The dense arrays are the two
  resampled cubes (density and temperature) plus a small level cube.  At `1024^3` with
  `float32` storage this is about 4.3 GB for each of the two cubes plus 1 GB for the
  level cube.  With `float64` (used by `auto` below 1024 samples) the cubes double.  The
  excision mask is applied on a small sub-cube so that it does not add a full-size
  temporary array.
* Plan for about 10 GB of host memory at `1024^3` and about 2.5 GB at `512^3` in double
  precision, plus the snapshot buffers.  `--grid-dtype float32` halves the cube memory
  below 1024 samples, at the storage precision used for the paper's production size.
* The cost grows with the number of pixels times `los_steps`, and for `multifreq` with
  the number of photon energies.  The kernels use numba threads over image rows, so
  `--threads` should match the cores available.  The first call includes compilation.
* Reading a large snapshot is limited by the file system, because the density and
  internal energy of every MeshBlock have to be read.  Copy the snapshot to fast
  storage if necessary.
* Output files are moderate in size.  A `1024^2` map takes 8 MB before compression.  For
  `tau1` the per-pixel band maps add 14 bands times that.  Use `--no-bands` to skip them.

## 7. Tests

The tests are in `athenak_rt/tests` and need pytest, numpy, numba, and h5py.  They
do not need an EOS table or a simulation snapshot.

```bash
cd /path/to/athenak
python3 -m pytest athenak_rt/tests
```

Running `pytest athenak_rt/tests` from the repository root also works, because
`conftest.py` adds the repository root to the path.  The tests cover analytic uniform and
two-layer slabs (formal solution, `tau`-stop truncation, the `tau = 1` photosphere, all
six directions), the multifrequency kernel against per-frequency analytic slabs with
and without scattering, the grey-limit identity, Saha charge neutrality and ionization
limits, the continuum edges, the Planck quadrature, and an end-to-end run on a small
synthetic two-level AMR snapshot with a synthetic EOS table.

## Physics and limits

The results are LTE post-processing estimates, and the following limits apply.

* **No radiation in the simulation.**  The hydrodynamic calculation is adiabatic and
  omits radiative cooling.  `athenak_rt` does not evolve radiation energy or momentum, does
  not enforce a radiative energy budget, and does not include photon trapping or
  advection or radiative feedback on the dynamics.  When the emitted power is comparable
  to the dissipation rate, as at the epoch of the paper's example, cooling would not be
  negligible.  A prediction of the emergent luminosity at these super-Eddington rates
  needs radiation hydrodynamics.  The luminosities that the paper quotes for its HR
  snapshot (about 25 to 40 Eddington luminosities for `M_BH = 1000 Msun`) are
  illustrative.
* **LTE, continuum only.**  The temperature comes from the LTE EOS table, and the
  source function is a (modified) blackbody.  There are no lines, no metals, and no
  non-LTE source functions.  Scattering enters only through the thermalization-depth
  approximation of a homogeneous scattering atmosphere applied locally along each ray,
  and not through a full scattering solution.
* **Composition.**  The grey modes use MESA opacities for `X = 0.7`, `Z = 0.02`, which
  is inconsistent with the metal-free H/He EOS of the hydrodynamics.  The multifrequency
  mode uses `X = 0.7`, `Y = 0.3` in its Saha solver (fixed in `athenak_rt/constants.py`),
  which matches the standard EOS table.  Do not use the multifrequency mode with a
  table of another composition without changing these constants.
* **Debris versus atmosphere.**  The snapshots contain no passive scalar that separates
  debris from the numerical atmosphere, so the density threshold does this job.  The
  luminosity depends on it, most strongly in the in-plane view.
* **Rosseland photosphere.**  The grey `tau1` photosphere is scattering inclusive, so
  it is not the thermalization surface.  It overestimates the luminosity compared with
  the frequency-dependent calculation.
* **No relativistic effects.**  Doppler boosting, aberration, and light-travel time are
  neglected.  In the paper's example snapshot the maximum gas speed is about `0.03 c`.
* **Projection.**  The luminosities are isotropic-equivalent values for one line of
  sight, and they are view dependent.
* **Not included.**  Photoionization codes and line emission, and the native-AMR
  streaming grey solvers, are not part of the package.

## References

The photospheric method follows Yang et al. (2026, ApJ, 998, 118), adapted to Cartesian
AthenaK output.  Opacities come from MESA (Paxton et al. 2011, 2013, 2015, 2018, 2019),
including OPAL tables (Iglesias & Rogers 1996) and the low-temperature tables of
Ferguson et al. (2005), for the Grevesse & Sauval (1998) mixture.  Credit these sources
when you use results derived from the bundled tables (see
`athenak_rt/data/NOTICE`).
