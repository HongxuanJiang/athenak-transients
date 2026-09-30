# Radiative post-processing with `athenak_rt`

`athenak_rt` (the top-level `athenak_rt/` directory) computes line-of-sight luminosities,
spectra, image-plane maps, and light curves from AthenaK `.bin` dumps, for example of a TDE
run.  It is the post-processing described in the section "Radiative post-processing and
synthetic observables" of [Jiang et al. (2026)](https://arxiv.org/abs/2609.37859), built on the photospheric
method of [Yang et al. (2026, ApJ, 998, 118)](https://ui.adsabs.harvard.edu/abs/2026ApJ...998..118Y) that Mengqi Yang developed for
Athena++.  If you use `athenak_rt`, please cite both papers.  The package reads a dump, resamples the
finest AMR level of the dense gas onto a uniform Cartesian ray grid, recovers the gas
temperature from the same tabulated equation of state (EOS) that the simulation used, and
integrates the transfer equation along parallel rays.  A run is described by a parameter
file in the AthenaK input format.

This is a post-processing tool for a hydrodynamic calculation without radiation.  Read
[Physics and limits](#physics-and-limits) before quoting luminosities.  A shorter README
is `athenak_rt/README.md` in the repository.

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
| Dumps | `<run>/bin/<basename>.hydro_w.NNNNN.bin` | Must contain `dens` and `eint` and the input parameters in the header |
| EOS table | the file that the run used, for example `chabrier2021_t13_helm_union_prad_640.table` | Not in git. Download it from the release assets or regenerate it with `scripts/generate_lte_table.py` (see `docs/eos_tables.md` and "Equation of state table" in `docs/TDE/README.md`) |
| Opacity tables | `athenak_rt/data/gs98_z0.02_x0.7.data`, `lowT_fa05_gs98_z0.02_x0.7.data` | Included. MESA `kap` files for `X = 0.7`, `Z = 0.02` (OPAL type 1 and Ferguson et al. 2005 low-temperature), see `athenak_rt/data/NOTICE` |

The EOS table is looked up in this order (`athenak_rt/eos.py`).

1. `<tables>/eos_table`, if it is set to a path.  The file must exist.
2. The path recorded in the dump header (`<hydro>/table`).  It is tried as written
   (relative to the current directory) and, if it is relative, relative to the run
   directory (the parent of `<input>/dump_dir`) and to `dump_dir`.  The example decks
   record `../chabrier2021_t13_helm_union_prad_640.table` relative to the directory in
   which AthenaK was started, so the table is found while the dumps are still in that
   directory's `bin/`.
3. The basename of the header path inside `$ATHENAK_EOS_TABLE_DIR`.

If none exists the run stops with a list of the places it tried.  Set
`<tables>/eos_table` when the dumps have been moved.

## 2. Modes

Select a mode with `<transfer>/mode`.  All modes share the same cell selection, the same
ray grid, and the same BH excision mask.  Luminosities are projected,
isotropic-equivalent values for the chosen line of sight, `L = 4 pi sum(I dA)`.  For a
blackbody pixel, `L_bol = 4 sigma T^4 dA`.  The effective temperature of a pixel is
`T_eff = (pi I / sigma)^(1/4)`.

| Mode | What is computed | Opacity | Products |
|---|---|---|---|
| `tau1` | The Rosseland optical depth is accumulated along each ray from the observer. The first cell with `tau >= tau_photosphere` (default 1) is the photosphere, and the pixel emits as a blackbody at the temperature of that cell. | MESA Rosseland (includes electron scattering) | photosphere temperature, coordinate, total `tau`, bolometric pixel luminosity, band `L_nu` (blackbody) |
| `grey` | Grey formal solution `I = int B(T) exp(-tau_R) dtau_R`. | MESA Rosseland | intensity, `tau`, `T_eff`, bolometric pixel luminosity |
| `grey-therm` | Grey solution with a thermalization depth. Scattering is `alpha_s = sigma_T n_e` and absorption `alpha_a = kappa_R rho - alpha_s`. The effective depth is `dtau* = sqrt(alpha_a (alpha_a + alpha_s)) ds` and the source function is weighted by `2 sqrt(eps) / (1 + sqrt(eps))` with `eps = alpha_a / (alpha_a + alpha_s)`. | MESA Rosseland | as `grey` |
| `multifreq` (default) | Hydrogen and helium continuum transfer at each photon energy on a logarithmic grid (default 73 points from 0.1 to 1000 eV). The populations of H I, H II, He I, He II, He III come from the EOS table by default (`populations = eos`), or from an ideal Saha solver with `populations = saha`. The absorption includes free-free, bound-free from H (levels `n <= 6`), He I and He II, with stimulated emission. Electron scattering enters through the same thermalization-depth treatment per frequency. | H/He continuum, no lines, no metals | intensity, `T_eff`, spectrum `L_nu(E)`, band `L_nu` |

`scattering = false` turns `multifreq` into pure absorption.  In all modes except `tau1`
the integration along a ray stops once the optical depth exceeds `tau_stop` (default 30).
In `multifreq` mode this is checked per photon energy.  The Rosseland tables are loaded
in every mode, and in `multifreq` mode they are not used for the result.  With
`populations = eos` and a table without He or H2 fractions the run falls back to Saha with
a warning, and the choice used is stored as `populations_used` in the output.

The multifrequency mode is the one adopted in the paper.  The grey `tau1` photosphere
is kept as a reference.  In the paper it overestimates the luminosity by roughly a
factor of seven, because near the Planck peak the bound-free opacity is much larger than
the Rosseland mean and the emission emerges from cooler, outer layers.

### Cell selection and geometry

* **Density cut.**  Cells with `rho > factor * dfloor` enter the ray grid (strict
  inequality, `<selection>/density_threshold_factor`, default 10), or cells above an
  explicit `density_threshold_code`.  Other cells are treated as empty.
* **Ray-grid box.**  By default the box is the bounding box of the selected cells, padded
  by the larger of `auto_box_padding_code` (0.25) and `auto_box_padding_fraction` (0.10)
  times the box width on every side, with a minimum width of `auto_box_min_width` (1.0),
  and clipped to the mesh (`auto_box_clip_to_mesh`).  For the bounding box the threshold
  is `factor * max(hydro/dfloor, problem/rho_floor)` from the header.  With the example
  decks both floors are equal.  `<image>/box` fixes the box explicitly.
* **BH excision.**  The sphere of radius `problem/bh_excise_radius` from the header is
  emptied around the BH position stored in the header.  `<selection>/bh_mask_radius`
  changes the radius and `bh_mask = false` switches the mask off.
* **Directions.**  `<transfer>/directions` lists one or more of `x`, `y`, `z`, `-x`, `-y`,
  `-z`, and the observer sits on that face of the box.  The image rows and columns are
  `(y, x)` for `z` rays, `(z, x)` for `y` rays, and `(z, y)` for `x` rays.
* **Resampling.**  The finest AMR level available at each location is sampled by
  nearest-cell lookup onto `image_size x image_size x los_steps` points.  The dump is
  streamed block by block and is never held in memory in full.

## 3. The parameter file

```bash
python3 -m athenak_rt PARAMETER_FILE
```

The program takes one argument, the parameter file.  Apart from `--help` and `--version`
there are no command-line options.  Two annotated examples that list every key are in
`athenak_rt/examples/`: `tde_snapshot.rtin` (one dump) and `tde_lightcurve.rtin` (a series
of dumps).  The smallest useful file is

```
<input>
dump_dir = /path/to/run/bin
dumps    = 327
```

### Format

The format is that of AthenaK input files.

* A block starts with its name in angle brackets on a line of its own, for example
  `<transfer>`, and holds the `key = value` lines up to the next block.  A block may
  appear more than once.
* `#` starts a comment, also after a value.  Blank lines are ignored.
* Every key that is left out takes the default listed below.  Only `<input>/dump_dir` and
  `<input>/dumps` are required.
* An unknown block, an unknown key, a key given twice, and a value that cannot be read
  stop the program with the file name, the line, and the closest valid name, so a typo
  is never ignored silently.
* `auto` selects the derived value of the keys whose default is `auto`.
* Booleans are `true` or `false` (`yes`/`no`, `on`/`off`, `1`/`0` are accepted).  Lists
  (`directions`, `box`, `dumps`) are separated by commas or blanks.
* Relative paths are taken relative to the directory of the parameter file, not the
  current directory.  `~` and `$VARIABLES` are expanded.

### Selecting the dumps

AthenaK writes its dumps as `<basename>.<variable>.<NNNNN>.bin` into the `bin/` directory
of the run.  `<input>/dump_dir` is that directory, `basename` and `variable` complete the
file name (defaults `TDEExternalLTEPrad` and `hydro_w`), and `dumps` selects the numbers.

| `dumps =` | Selects |
|---|---|
| `327` | dump 327 |
| `310, 320, 327` | a list |
| `300:327` | 300 to 327, both ends included |
| `300:327:3` | every third dump from 300 (300, 303, ..., 327) |
| `300:320:5, 327` | ranges and single numbers combined |
| `all` | every `<basename>.<variable>.NNNNN.bin` in `dump_dir` |

The selection is sorted and duplicates are removed.  The program checks that every
selected file exists before it computes anything.  With `missing_dumps = error` (the
default) a missing dump stops the run, and the message lists the missing numbers.  With
`missing_dumps = skip` the missing numbers are printed as a warning and the other dumps
are processed.  If none of the selected dumps exists the run always stops.

Every selected dump is processed for every direction.  With more than one dump the run
also writes a light curve (Sec. 4).

### Keys

| Block | Key | Type | Default | Meaning |
|---|---|---|---|---|
| `<input>` | `dump_dir` | path | required | Directory with the `.bin` dumps |
| | `basename` | string | `TDEExternalLTEPrad` | `<job>/basename` of the run |
| | `variable` | string | `hydro_w` | Output variable of the dumps, which must contain `dens` and `eint` |
| | `dumps` | selection | required | Dump numbers, see above |
| | `missing_dumps` | `error`, `skip` | `error` | What to do with selected dumps that do not exist |
| `<output>` | `dir` | path | `athenak_rt_output` | Output directory, created if needed |
| | `hdf5_compression` | `lzf`, `gzip`, `none` | `lzf` | Compression of the maps and per-row spectra |
| | `bands` | bool | `true` | Band `L_nu` products (NIR, optical, UV, X-ray) and their light-curve columns |
| `<transfer>` | `mode` | `tau1`, `grey`, `grey-therm`, `multifreq` | `multifreq` | Transfer mode, see Sec. 2 |
| | `directions` | list of `x y z -x -y -z` | `z` | Lines of sight, for example `z, -y` |
| | `tau_photosphere` | float | `1.0` | `tau1`: optical depth of the photosphere |
| | `tau_stop` | float | `30.0` | `grey`, `grey-therm`, `multifreq`: stop a ray past this depth |
| | `nfreq` | int | `73` | `multifreq`: number of log-spaced photon energies, `>= 2` |
| | `emin_ev` | float | `0.1` | `multifreq`: lowest photon energy in eV |
| | `emax_ev` | float | `1000.0` | `multifreq`: highest photon energy in eV |
| | `scattering` | bool | `true` | `multifreq`: thermalization-depth treatment of electron scattering, `false` for pure absorption |
| | `populations` | `eos`, `saha` | `eos` | H/He populations for `grey-therm` and `multifreq`, from the EOS table or the ideal Saha solver |
| `<image>` | `image_size` | int | `1024` | Pixels across the image plane, `>= 2` |
| | `los_steps` | int | `1024` | Samples along the line of sight, `>= 2` |
| | `grid_dtype` | `auto`, `float32`, `float64` | `auto` | Storage of the resampled `(rho, T)` cube. `auto` uses `float32` if any axis has 1024 samples or more, else `float64`. EOS recovery and ray sums are always in double precision. |
| | `box` | `auto` or six floats | `auto` | Fixed ray-grid box `xmin, xmax, ymin, ymax, zmin, zmax` in code units |
| | `auto_box_padding_code` | float | `0.25` | Absolute padding of the automatic box on every side |
| | `auto_box_padding_fraction` | float | `0.1` | Padding as a fraction of the box width |
| | `auto_box_min_width` | float | `1.0` | Minimum width of the automatic box |
| | `auto_box_clip_to_mesh` | bool | `true` | Clip the automatic box to the mesh |
| `<selection>` | `density_threshold_factor` | float | `10.0` | Cells with `rho_code > factor * dfloor` are used |
| | `density_threshold_code` | `auto` or float | `auto` | Explicit threshold in code units, overrides the factor |
| | `bh_mask` | bool | `true` | Empty the BH excision sphere |
| | `bh_mask_radius` | `auto` or float | `auto` | Excision radius in code units, `auto` takes `problem/bh_excise_radius` from the header |
| `<tables>` | `eos_table` | `auto` or path | `auto` | EOS table, `auto` takes the path in the dump header (Sec. 1) |
| | `mesa_high_t` | `auto` or path | `auto` | MESA/OPAL high-temperature opacity table, `auto` is `athenak_rt/data/gs98_z0.02_x0.7.data` |
| | `mesa_low_t` | `auto` or path | `auto` | MESA low-temperature opacity table, `auto` is `athenak_rt/data/lowT_fa05_gs98_z0.02_x0.7.data` |
| `<run>` | `threads` | `auto` or int | `auto` | Number of numba threads, `auto` uses all CPUs available to the process |
| | `skip_existing` | bool | `false` | Keep per-dump products that already exist (restart, see Sec. 4) |

The program prints one line per dump and direction with the bolometric luminosity and the
name of the output file.

## 4. Outputs

Each run writes into `<output>/dir`.

| File | Written | Content |
|---|---|---|
| `<dump stem>.rt_<mode>_<direction>.h5` | always, one per dump and direction | Maps, spectrum, bands, every setting, and the parameter file |
| `rt_lightcurve_<mode>.csv` | more than one dump | Light curve, one row per dump, all directions |
| `rt_lightcurve_<mode>.h5` | more than one dump | The same plus per-dump diagnostics and the spectra of all dumps |

For example, the file for the `-y` view of dump 327 is
`TDEExternalLTEPrad.hydro_w.00327.rt_multifreq_-y.h5`.

### Per-dump HDF5 file

| Group and dataset | Modes | Content |
|---|---|---|
| root attributes | all | Every setting (`RTSettings.as_attributes()`), table paths (`eos_table_resolved`, `eos_table_in_header`), `snapshot_index`, `time_code`, `time_s`, `cycle`, `luminosity_bolometric_erg_s`, the normalization (`luminosity_normalization`, `luminosity_bolometric_formula`, `projected_luminosity_factor`), pixel area `area_pixel_cm2`, step `los_step_cm`, `valid_pixels`, `max_tau`, `median_temp_K`, `mean_temp_K`, `dfloor_code`, `density_threshold_code`, ray box `rt_box_code` and the raw box `auto_box_raw_code`, `bh_xyz_code`, `bh_mask_radius_code`, unit scales, `grid_storage_dtype`, `numba_threads`, `populations_used`, package version |
| `parameters/parameter_file`, `parameters/resolved` | all | The parameter file as read, and the same file with every key and the value used (defaults filled in, paths absolute). The attribute `parameters.attrs["path"]` is the file's location. The resolved text is itself a valid parameter file. |
| `image/image_u_code`, `image/image_v_code` | all | Pixel coordinates along the image columns and rows |
| `maps/tau_total`, `photosphere_temp_K`, `photosphere_coord_code`, `valid`, `bolometric_pixel_luminosity_erg_s` | `tau1` | Photosphere maps |
| `maps/intensity_erg_s_cm2_sr`, `tau_effective`, `effective_temp_K`, `valid`, `bolometric_pixel_luminosity_erg_s` | `grey`, `grey-therm` | Intensity and effective-temperature maps |
| `maps/intensity_erg_s_cm2_sr`, `effective_temp_K`, `valid`, `bolometric_pixel_luminosity_erg_s` | `multifreq` | Frequency-integrated intensity and effective-temperature maps |
| `spectra/energy_ev`, `frequency_hz`, `quadrature_weight_hz`, `lnu_erg_s_hz`, `nu_lnu_erg_s`, `lnu_row_erg_s_hz` | `multifreq` | Spectrum on the photon-energy grid. `lnu_row_erg_s_hz` is `L_nu` per image row. The frequency integral is a trapezoid rule in `ln(nu)`. |
| `bands/label`, `category`, `frequency_hz`, `wavelength_nm`, `energy_ev`, `lnu_erg_s_hz`, `nu_lnu_erg_s` | `tau1`, `multifreq` with `bands = true` | `L_nu` in observation bands. `tau1` uses the blackbody at the photosphere temperature and adds `lnu_pixel_erg_s_hz` (per-pixel maps). `multifreq` interpolates the spectrum log-log. Bands outside the spectrum are 0. `grey` and `grey-therm` write no band data. |

The bands are `nir_J`, `nir_H`, `nir_K` (1250, 1650, 2200 nm), `optical_u`, `_g`, `_r`,
`_i` (355, 475, 622, 763 nm), `uv_UVW1`, `uv_UVM2`, `uv_UVW2`, `uv_FUV150` (260, 224.6,
192.8, 150 nm), and `xray_0p3keV`, `xray_1keV`, `xray_2keV` (`athenak_rt/bands.py`).

### Light curve

When more than one dump is selected, `rt_lightcurve_<mode>.csv` is a plain CSV file with
a header line and one row per dump in dump order.  It has no comment lines.

| Column | Unit | Content |
|---|---|---|
| `dump` | | Dump number |
| `cycle` | | Cycle of the dump |
| `time_code` | code units | Simulation time |
| `time_s` | s | `time_code` times `<units>/time_cgs` |
| `L_bol_<dir>` | erg/s | `L_bol,iso` seen from each direction |
| `nuLnu_<band>_<dir>` | erg/s | `nu L_nu` in each band for each direction, only with band products (`tau1`, `multifreq`, `bands = true`) |

In column names the direction `-y` is written `minus_y`, for example `L_bol_minus_y` and
`nuLnu_optical_g_minus_y`.  The time in units of `P_mb` is not written, because the
package does not know `P_mb`.  Divide `time_code` by the `P_mb` of the run (88.04 code
units for the example decks).

`rt_lightcurve_<mode>.h5` holds the same rows as arrays.

| Dataset | Shape | Content |
|---|---|---|
| `dump`, `cycle`, `time_code`, `time_s` | `(ndump,)` | as in the CSV |
| `bands/label`, `category`, `frequency_hz`, `wavelength_nm`, `energy_ev` | `(nband,)` | Band definitions |
| `spectra/energy_ev`, `frequency_hz` | `(nfreq,)` | `multifreq` photon-energy grid |
| `<dir>/luminosity_bolometric_erg_s` | `(ndump,)` | `L_bol,iso`, one group per direction named as in the parameter file (`z`, `-y`) |
| `<dir>/valid_pixels`, `total_pixels`, `max_tau`, `median_temp_K`, `mean_temp_K`, `density_threshold_code` | `(ndump,)` | Per-dump diagnostics |
| `<dir>/bh_xyz_code` | `(ndump, 3)` | BH position |
| `<dir>/hdf5_file` | `(ndump,)` | Name of the per-dump product |
| `<dir>/bands/lnu_erg_s_hz`, `nu_lnu_erg_s` | `(ndump, nband)` | Band luminosities |
| `<dir>/spectra/lnu_erg_s_hz`, `nu_lnu_erg_s` | `(ndump, nfreq)` | `multifreq` spectra |
| `parameters/parameter_file`, `parameters/resolved` | | As in the per-dump files |

Both files are rewritten after each dump, so a light curve can be inspected while a long
run is going.  They are assembled from the per-dump products.

### Restarting a run

Every product is first written under a temporary name ending in `.part` and renamed when it
is complete, so an interrupted run leaves no truncated product.  With
`<run>/skip_existing = true` a (dump, direction) whose product exists is not computed
again.  The settings stored in the product are compared with the current ones (every
setting except the output directory, the thread count and the compression), and the run
stops if they differ, so a product made with other settings is never reused silently.
Reused products enter the light curve as if they had been computed.

### Reading the results

```python
import h5py, numpy as np

with h5py.File("rt_out/TDEExternalLTEPrad.hydro_w.00327.rt_multifreq_z.h5") as f:
    print("t =", f.attrs["time_code"], " L_bol,iso =", f.attrs["luminosity_bolometric_erg_s"])
    e_ev = f["spectra/energy_ev"][:]
    nu_lnu = f["spectra/nu_lnu_erg_s"][:]
    print("peak of nu L_nu at", e_ev[np.argmax(nu_lnu)], "eV")
    teff = f["maps/effective_temp_K"][:]          # (rows, columns), NaN where I <= 0
    print(f["parameters/resolved"].asstr()[()])   # the settings of this product

lc = np.genfromtxt("rt_lightcurve/rt_lightcurve_multifreq.csv", delimiter=",", names=True)
t_pmb = lc["time_code"] / 88.04                   # P_mb of the example decks
print(lc["dump"], t_pmb, lc["L_bol_z"], lc["L_bol_minus_y"])
```

## 5. Worked examples

### The paper's snapshot

The paper's main synthetic-observable result is the multifrequency H/He continuum
transfer of the last high-resolution (HR) dump, 327 at `t / P_mb = 2.93`, viewed along `+z`
(face-on) and along `-y` (in-plane, with the nozzle on the near side), on `1024^2` pixels
with 1024 samples per line of sight.  The HR run is not part of the example decks, so
point `dump_dir` at a run of your own.  The example FID chain of
`inputs/TDE_examples/README.md` produces dumps that can be processed the same way.
This is `athenak_rt/examples/tde_snapshot.rtin`.

```
<input>/dump_dir.  Relative paths are relative to this file's
# directory; ~ and $VARIABLES are expanded.  A key that is left out takes the
# default given in its comment; "auto" asks for the derived value.

<input>
dump_dir      = /path/to/run/bin    # directory with the .bin dumps (required)
basename      = TDEExternalLTEPrad  # files are <basename>.<variable>.<NNNNN>.bin
variable      = hydro_w             # output variable of the dumps (needs dens and eint)
dumps         = 327                 # N | N1, N2, ... | A:B | A:B:STEP (inclusive) | all
missing_dumps = error               # error | skip (warn and process the others)

<output>
dir              = rt_out           # output directory (default athenak_rt_output)
hdf5_compression = lzf              # lzf | gzip | none
bands            = true             # NIR/optical/UV/X-ray band L_nu

<transfer>
mode            = multifreq         # tau1 | grey | grey-therm | multifreq
directions      = z, -y             # lines of sight among x y z -x -y -z (default z)
tau_photosphere = 1.0               # tau1: optical depth of the photosphere
tau_stop        = 30.0              # grey*, multifreq: stop a ray past this depth
nfreq           = 73                # multifreq: log-spaced photon energies
emin_ev         = 0.1               # multifreq: lowest photon energy [eV]
emax_ev         = 1000.0            # multifreq: highest photon energy [eV]
scattering      = true              # multifreq: thermalization depth (false: pure absorption)
populations     = eos               # eos (EOS-table fractions) | saha (ideal Saha, X=0.7, Y=0.3)

<image>
image_size                = 1024    # pixels across the image plane
los_steps                 = 1024    # samples along the line of sight
grid_dtype                = auto    # auto | float32 | float64 (auto: float32 from 1024 samples)
box                       = auto    # auto | xmin, xmax, ymin, ymax, zmin, zmax (code units)
auto_box_padding_code     = 0.25    # auto box: absolute padding per side
auto_box_padding_fraction = 0.1     # auto box: padding per side as a fraction of the width
auto_box_min_width        = 1.0     # auto box: minimum width
auto_box_clip_to_mesh     = true    # auto box: clip to the mesh

<selection>
density_threshold_factor = 10.0     # cells with rho_code > factor * dfloor enter the grid
density_threshold_code   = auto     # auto | explicit threshold in code units (overrides the factor)
bh_mask                  = true     # empty the BH excision sphere
bh_mask_radius           = auto     # auto (<problem>/bh_excise_radius of the dump) | radius in code units

<tables>
eos_table   = auto                  # auto: <hydro>/table of the dump header, also tried relative
                                    # to the run directory (parent of dump_dir) and in
                                    # $ATHENAK_EOS_TABLE_DIR; or a path, e.g.
                                    # /path/to/chabrier2021_t13_helm_union_prad_640.table
mesa_high_t = auto                  # auto: athenak_rt/data/gs98_z0.02_x0.7.data
mesa_low_t  = auto                  # auto: athenak_rt/data/lowT_fa05_gs98_z0.02_x0.7.data

<run>
threads       = auto                # numba threads (auto: all CPUs available to the process)
skip_existing = false               # true: keep per-dump outputs that already exist
```

Copy the file to the directory that should hold the results, set `dump_dir` (and
`eos_table` if the dumps have been moved away from the run directory), and run

```bash
cd /path/to/athenak
python3 -m athenak_rt /path/to/work/tde_snapshot.rtin
```

The products are `rt_out/TDEExternalLTEPrad.hydro_w.00327.rt_multifreq_z.h5` and
`..._-y.h5` next to the parameter file.  For the grey `tau = 1` photosphere as the
reference, copy the file and change only `mode = tau1`.  The products then end in
`.rt_tau1_z.h5` and `.rt_tau1_-y.h5` and can share the output directory.  For the last HR
dump of the paper the published values are

| Line of sight | Multifrequency `L_bol,iso` | Grey `tau = 1` `L_bol,iso` |
|---|---|---|
| `+z` | `6.1e42 erg/s` | `3.2e43 erg/s` |
| `-y` | `3.6e42 erg/s` | `2.2e43 erg/s` |

The multifrequency spectrum peaks near 60 eV.  The numbers depend on the dump, on the
density threshold that separates debris from the numerical atmosphere, and on the image
sampling.  In the paper, going from `512^2` to `1024^2` pixels changes the grey
luminosities by 0.4% and 6.1%, and doubling the number of photon energies changes the
multifrequency luminosity by less than 3%.  A quick test at `image_size = 192` and
`los_steps = 192` takes a small fraction of the cost.

### A light curve

`athenak_rt/examples/tde_lightcurve.rtin` has the same keys.  Its differences from the
snapshot example are

```
<input>
dumps         = 300:327             # dumps 300 to 327 inclusive; 300:327:3 takes every third
missing_dumps = skip                # warn about missing dumps and process the others

<output>
dir = rt_lightcurve

<image>
image_size = 512                    # the paper's figure used 1024; 512 is 8 times cheaper
los_steps  = 512

<run>
skip_existing = true                # an interrupted run continues where it stopped
```

The run processes every dump from both directions and writes
`rt_lightcurve/rt_lightcurve_multifreq.csv` and `.h5` besides the 56 per-dump products.
Resubmitting the same parameter file after an interruption computes only the products
that are missing.  The paper does not present light curves, because the simulation does
not reach the peak of the flare.

### Python interface

```python
import athenak_rt

results = athenak_rt.run_parameter_file("tde_snapshot.rtin")    # as the command line
print(results[0].luminosity, results[0].hdf5_path)

config = athenak_rt.load_parameter_file("tde_snapshot.rtin")    # parse and check only
print(config.settings, config.directions, config.find_dumps())
```

`athenak_rt.run(settings, dumps, directions)` runs an `athenak_rt.RTSettings` object on a
list of dump paths, and `athenak_rt.process_snapshot` handles one dump and one direction.
The kernels in `athenak_rt.transfer` accept any `(nz, ny, nx)` cube of density in g/cm^3
and temperature in K.

## 6. Performance and memory

* The dump is streamed and never loaded in full.  The dense arrays are the two
  resampled cubes (density and temperature) plus a small level cube.  At `1024^3` with
  `float32` storage this is about 4.3 GB for each of the two cubes plus 1 GB for the
  level cube.  With `float64` (used by `auto` below 1024 samples) the cubes double.  The
  excision mask is applied on a small sub-cube so that it does not add a full-size
  temporary array.
* Plan for about 10 GB of host memory at `1024^3` and about 2.5 GB at `512^3` in double
  precision, plus the dump buffers.  `grid_dtype = float32` halves the cube memory below
  1024 samples, at the storage precision used for the paper's production size.
* The cost grows with the number of pixels times `los_steps`, and for `multifreq` with
  the number of photon energies.  Each direction is a separate pass over the dump.  The
  kernels use numba threads over image rows, so `<run>/threads` should match the cores
  available.  The first call includes compilation.
* Reading a large dump is limited by the file system, because the density and internal
  energy of every MeshBlock have to be read.  Copy the dumps to fast storage if
  necessary, and point `dump_dir` there.
* Output files are moderate in size.  A `1024^2` map takes 8 MB before compression.  For
  `tau1` the per-pixel band maps add 14 bands times that.  `bands = false` skips them.

## 7. Tests

The tests are in `athenak_rt/tests` and need pytest, numpy, numba, and h5py.  They
do not need an EOS table or a simulation dump.

```bash
cd /path/to/athenak
python3 -m pytest athenak_rt/tests
```

Running `pytest athenak_rt/tests` from the repository root also works, because
`conftest.py` adds the repository root to the path.  The tests cover analytic uniform and
two-layer slabs (formal solution, `tau`-stop truncation, the `tau = 1` photosphere, all
six directions), the multifrequency kernel against per-frequency analytic slabs with
and without scattering, the grey-limit identity, Saha charge neutrality and ionization
limits, the continuum edges, the Planck quadrature, and end-to-end runs on a small
synthetic two-level AMR dump with a synthetic EOS table.  The parameter-file tests check
the parser and its error messages, that the defaults equal those of `RTSettings`, the dump
selection (lists, ranges, strides, `all`, missing dumps), the example files, a single-dump
run, a three-dump light curve with two directions, and the restart with `skip_existing`.

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
  mode takes its populations from the EOS table, which is consistent with the hydrodynamics.
  With `populations = saha` it assumes `X = 0.7`, `Y = 0.3` (fixed in
  `athenak_rt/constants.py`), so do not use that option with a table of another composition
  without changing these constants.
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

The photospheric method follows [Yang et al. (2026, ApJ, 998, 118)](https://ui.adsabs.harvard.edu/abs/2026ApJ...998..118Y)
([arXiv:2510.25547](https://arxiv.org/abs/2510.25547)), adapted to Cartesian AthenaK output, and the
multifrequency transfer is that of [Jiang et al. (2026)](https://arxiv.org/abs/2609.37859).  Please cite both
papers when you use `athenak_rt`.  Opacities come from MESA (Paxton et al. 2011, 2013, 2015, 2018, 2019),
including OPAL tables (Iglesias & Rogers 1996) and the low-temperature tables of
Ferguson et al. (2005), for the Grevesse & Sauval (1998) mixture.  Credit these sources
when you use results derived from the bundled tables (see
`athenak_rt/data/NOTICE`).
