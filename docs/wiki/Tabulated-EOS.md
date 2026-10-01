# Tabulated EOS (LTE / Saha)

## Summary

The tabulated equations of state replace the gamma-law and isothermal closures with a
two-dimensional table. The table gives the pressure, the specific internal energy, the sound
speed and the ionization and dissociation fractions as functions of density and temperature.
The tables are computed offline (see [EOS Tables](EOS-Tables)) and read when the run starts.
There are two families:

- `saha_table`: hydrogen only, with the ionization given by the Saha equation.
- `lte_table_*`: hydrogen plus helium in local thermodynamic equilibrium (LTE). They include
  H2 dissociation, ionization, degenerate electrons and, optionally, radiation pressure.

Use a tabulated EOS for non-relativistic hydrodynamics or MHD in which ionization, H2
dissociation, degenerate-electron pressure or radiation pressure matter enough that a constant
gamma is wrong, for example stellar envelopes and tidal-disruption debris.

Do not use it for special or general relativity (not implemented), with an entropy floor
(`sfloor` must be 0), with ISM, relativistic or disk cooling or with thermal conduction
(refused), or when you need metals in the EOS (the tables have X + Y = 1, so metals are folded
into helium).

A conversion costs one bilinear table lookup per cell, with no iteration. The price is memory:
about 310 MiB on the device, plus an equal copy on the host, per fluid block per rank for a
640 x 640 H+He table (see [Memory cost](#memory-cost)).

## Quick start

First build or download a table (`./get_eos_table.sh` builds the one used below; see
[EOS Tables](EOS-Tables#building-the-default-table)). Then put the keys below in the `<hydro>`
block (or `<mhd>`). They come from the first TDE example deck,
`inputs/TDE_examples/tde_01_disruption.athinput`. The `<units>` block above them in that deck
sets the unit system and is mandatory.

```ini
<hydro>
eos = lte_table_chabrier2021_t13_helm_union_prad
table = ../chabrier2021_t13_helm_union_prad_640.table
reconstruct = plm
rsolver = hlle
dfloor = 1.0e-10
pfloor = 1.0e-12
tfloor_kelvin = 1
sfloor = 0.0
cs_ceil = 15.0
vceil = 15.0
fofc = false
dual_energy = true
dual_energy_eta1 = 1.0e-3
dual_energy_eta2 = 1.0e-4
lte_bounds = clamp
lte_debug_checks = false
```

For the hydrogen-only table, use `eos = saha_table`, `table = <path>/saha_hydrogen.table`, and
`saha_bounds` and `saha_debug_checks` in place of the `lte_` names.

The keys that matter most:

- `eos` and `table` select the table family and the file. The `eos` name must match the table.
- `sfloor = 0.0` is required. The default is positive, and any positive value is a fatal error.
- `tfloor_kelvin` sets the temperature floor in Kelvin and replaces the code-unit `tfloor`.
- `lte_bounds` (or `saha_bounds`) chooses what happens when a query leaves the table: `error`
  aborts and `clamp` carries on (see [Outside the table](#outside-the-table)).
- `dual_energy` is not an EOS key, but it is usually switched on with a tabulated EOS (see
  [Dual Energy](Dual-Energy)).

## Full parameter table

All keys are read once, when the EOS is built, from the `<hydro>` or `<mhd>` block whose `eos`
is tabulated. `FLT_MIN` and `FLT_MAX` are the smallest normal and the largest single-precision
float, about 1.2e-38 and 3.4e38.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `eos` | string | required | `saha_table` or one of the `lte_table_*` names listed below. It must match the table file. |
| `table` | string | required | Path to the `.table` file. |
| `dfloor` | real | `FLT_MIN` | Density floor, in code units (the same key as for every EOS). |
| `pfloor` | real | `FLT_MIN` | Pressure floor, in code units. At each density it is converted into a temperature floor, and the higher of that and the temperature floor applies. |
| `tfloor` | real | `FLT_MIN` | Temperature floor in code units. Ignored when `tfloor_kelvin` is set. The lowest temperature of the table is a floor in any case. |
| `tfloor_kelvin` | real | unset | Temperature floor in Kelvin. When set, it replaces `tfloor`. |
| `sfloor` | real | `FLT_MIN` | Entropy floor, which is not supported. Set it to exactly `0.0`: the default is positive, and any positive value is fatal. |
| `cs_ceil` | real | `0` (off) | Sound-speed ceiling, in code units. When positive, the internal energy at which the sound speed reaches `cs_ceil` is precomputed for each density and used as a ceiling. |
| `vceil` | real | `FLT_MAX` | Ceiling on the velocity magnitude, in code units. |
| `saha_bounds` | `error` or `clamp` | `error` | Policy for queries outside the table. Read by `saha_table`, and by `lte_table_*` when `lte_bounds` is absent. |
| `lte_bounds` | `error` or `clamp` | value of `saha_bounds` | The same policy, for `lte_table_*`. It takes precedence over `saha_bounds`. |
| `saha_debug_checks` | bool | `false` | Costly consistency checks at load (see [Checking a table](#checking-a-table)). Read by `saha_table`, and by `lte_table_*` when `lte_debug_checks` is absent. |
| `lte_debug_checks` | bool | value of `saha_debug_checks` | The same checks, for `lte_table_*`. |
| `saha_cache_eps_factor` | int | `8` | Resolution of the inverse table (see [below](#from-density-and-energy-to-pressure-and-temperature)): energy points per density row = factor x number of temperature points. At least 1. Only for `saha_table`. |
| `lte_cache_eps_factor` | int | `8` | The same, for `lte_table_*`. |
| `saha_floor_cache_factor` | int | `16` | Density resolution of the precomputed floor curve: factor x (nrho - 1) + 1 points, with nrho the number of density points of the table. At least 1. Only for `saha_table`. |
| `lte_floor_cache_factor` | int | `16` | The same, for `lte_table_*`. |
| `allow_reinterpretive_restart` | bool | `false` | Allow a restart whose EOS or table differs from the one recorded in the restart file (see [Restarts](#restarts)). |
| `dual_energy` | bool | `false` | Not an EOS key. Like a tabulated EOS, it makes AMR prolong primitive variables. |

Only the bounds and the debug-check keys fall back to their `saha_` names. The two cache
factors are read under the name of the active family only: `lte_*` for `lte_table_*` and
`saha_*` for `saha_table`.

The composition (X, Y), radiation pressure, H2 and zero-point-energy settings are not deck
keys. They come from the metadata of the table file. A `<saha_runtime>` block in restart
headers carries restart-compatibility information and is not meant to be edited.

### Names accepted for `eos`

The loader accepts 19 names: `saha_table`, and nine `lte_table_*` names, each with and without
a `_prad` suffix. A name with `_prad` needs a table with radiation pressure, and a name
without it needs a table without. The loader derives the table type that goes with the name
and stops with a message naming it if the file differs.

- Maintained, because the generator can build tables for them: `saha_table`, `lte_table_t13`,
  `lte_table_scvh_t13_cp_helm_union` and `lte_table_chabrier2021_t13_helm_union`, each `lte_`
  name with or without `_prad`. The generator's job list does not cover every variant; see
  [EOS Tables](EOS-Tables#the-table-jobs).
- Unmaintained (see [Limitations](#limitations)): `lte_table_hhe`, `lte_table_hybrid_hhe_t13`,
  `lte_table_scvh1995_hhe`, `lte_table_scvh_t13_union`, `lte_table_scvh_t13_helm_union` and
  `lte_table_scvh_t13_cp_union`, each with or without `_prad`.

## How it works

### Chemistry models

Two independent backends feed the same table container and the same run-time code. HELM below
stands for the Helmholtz EOS of Timmes & Swesty (2000), whose relativistic electron-positron
gas is used at high temperature.

| family | model | content |
| --- | --- | --- |
| `saha_table` | `saha_hydrogen_lte` | Hydrogen-only Saha ionization equilibrium, X = 1, Y = 0. |
| `lte_table_*` | `t13` | Tomida et al. (2013) partition-function chemical equilibrium (H2, H, H+, He, He+, He2+, e-), with optional radiation pressure and H2 zero-point-energy subtraction. Above log10(T/K) = 7 the electron gas is replaced by a HELM-style relativistic e-/e+ pair gas. |
| `lte_table_*` | `scvh_t13_cp_helm_union` | SCvH (Saumon, Chabrier & Van Horn 1995) where it is valid, a fully ionized "CP-like" corner where its own validity cuts pass, a HELM fallback in the hot dense corner, and `t13` at low density. |
| `lte_table_*` | `chabrier2021_t13_helm_union` | `t13` at low density, the dense-fluid EOS of Chabrier et al. (2021) through the pressure-dissociation regime, and a HELM fallback in the hot dense corner beyond the Chabrier coverage. |

The default composition is X = 0.70, Y = 0.30. Other compositions are set per generator job.
The generator requires X + Y = 1, so metals are folded into Y. The published Chabrier mixture
tables cover only Y = 0.275 to 0.297. For the six composition-variant jobs the generator
therefore builds the Chabrier block from the pure-H and pure-He tables with an additive-volume
mixing law.

### The table file

A table is an ASCII header followed by raw binary data. The header has four blocks (metadata,
scalars, axes and field names). The data are float64 arrays: first each axis, then each field.

```
header:  metadata | scalars | points (axes) | fields
data:    logrho axis | logtemp axis | field 1 | field 2 | ...
```

- The axes are ln(rho in g/cm^3) and ln(T in K), each uniformly spaced.
- The byte order is recorded in the header and swapped at load time if it differs from the
  host.
- Required fields for `lte_table_*`: `logpress`, `logeps`, `logcs2`, `gamma1`, `gamma3m1`,
  `xion`, `xhe1`, `xhe2`, `mu`, `beta_rad`, plus `xh2` for tables with H2. For `saha_table`:
  `logpress`, `logeps`, `logcs2`, `gamma1`, `gamma3m1`, `xion`. The Saha loader fills in the
  rest (`mu = 1/(1+xion)`, the others zero).
- For `lte_table_*`, the mass fractions X and Y come from the scalars `x_h` and `y_he` of the
  header (default 0.70 and 0.30) and must sum to 1.

### From density and energy to pressure and temperature

The code evolves density and internal energy, but the table is indexed by density and
temperature. Inverting the energy for the temperature in every cell would be expensive, so it
is done once, on the host, at startup:

1. For each density row of the table, a uniform grid in log energy is laid out. It has
   `lte_cache_eps_factor` (default 8) times as many points as the table has temperature
   points.
2. For each energy on that grid, the temperature is found by searching the row, since the
   energy increases with temperature at fixed density.
3. The temperature, pressure, sound speed and composition fractions at that point are stored.

This grid of results is the inverse table (the `cache` in the parameter names). At run time, a
conversion from (density, energy) is a single bilinear lookup in it. A separate path
interpolates the raw table in (density, temperature) for queries that start from temperature.

### Outside the table

With `error`, a query outside the table aborts the run. With `clamp`:

- the density is clamped to the table range, and so is the cold edge (the lowest tabulated
  temperature);
- above the hottest tabulated temperature, the logarithms of pressure, energy and sound speed
  continue as the power law of the last temperature interval (a T^4 law where radiation and
  pairs dominate), while every other tabulated quantity keeps its value at the top
  temperature.

In `clamp` mode, temperature and pressure therefore keep growing with energy beyond the table
instead of saturating at its edge.

### Temperature unit

The code temperature unit belongs to the active EOS, not to `<units>`. For a tabulated EOS it
is v_code^2 m_H / k_B, with the hydrogen mass hard-coded, because the table is the authority
on the mean molecular weight. It does not use the `<units>/mu` key. The `hydro_temperature`
output is in this unit. A temperature scale computed by hand from `<units>` is wrong by
mu m_u / m_H under a tabulated EOS. For code that needs the scale, see the
[implementation notes](Tabulated-EOS-Implementation-Notes#where-other-modules-meet-the-eos).

### Memory cost

The inverse table is `lte_cache_eps_factor` times as large as the raw table (by default 8
times). Both hold 11 fields in double precision. For a 640 x 640 H+He table with the default
factor:

| item | size |
| --- | --- |
| raw table | about 34 MiB |
| inverse table | about 275 MiB |
| total on the device | about 310 MiB |
| copy in host memory | the same again, kept for the whole run |

The sizes are per `<hydro>` or `<mhd>` block per rank, so a run with both has twice as much.
Every rank reads and checks its own copy of the table file at startup, with no broadcast.

## Practical guidance

### Tables used by the examples

The repository contains no `.table` files. Build them with `get_eos_table.sh` or the generator
scripts, or download the default table from the release assets of the repository.
[EOS Tables](EOS-Tables) describes this and lists every generator job with its X, Y and
`eos` name. In the repository:

- The five TDE decks in `inputs/TDE_examples/` use `chabrier2021_t13_helm_union_prad_640.table`
  with `eos = lte_table_chabrier2021_t13_helm_union_prad`. `get_eos_table.sh` builds it.
- The tests under `tst/` read the same table from `eos_tables/`, and
  `chabrier2021_t13_helm_union_640.table` (no radiation pressure) for the PLM unit test.
- The chemistry plots of `scripts/TDE/bin/plot_slice.py` read `lte_t13_prad_eos.table`
  (job `lte_t13_prad`).
- No deck or test in the repository uses `saha_table`, the SCvH table or the composition
  variants.

### Restrictions and refusals

- Special and general relativity are refused.
- `sfloor > 0` is fatal. There is no entropy floor.
- A `<units>` block is mandatory.
- ISM, relativistic and disk cooling, and thermal conduction, are fatal in combination with a
  tabulated EOS, because those modules assume a gamma-law temperature closure.
- `rsolver = roe` is refused for hydro. Use `llf`, `hlle` or `hllc`.
- Remapping needs the source and target runs to use the same kind of closure (both ideal or
  both tabulated), otherwise it is fatal. See [Remapping](Remapping).
- The derived output variables of the table state (see below) are refused if the EOS is not
  tabulated.

### Interactions with other modules

- **AMR/SMR.** A tabulated EOS makes AMR prolong primitive variables, as dual energy does, so
  that the thermal state stays consistent after prolongation and regridding.
- **Dual energy.** The generic `tfloor` check is skipped for a tabulated EOS, because the
  tabulated floors already enforce it. The dual-energy switch applies on top; see
  [Dual Energy](Dual-Energy).
- **FOFC.** In the pass that tests for floors, a cell is flagged for first-order flux
  correction if its conversion would hit a floor, the energy ceiling or the velocity ceiling.
- **LAT.** The conversions run only over the active blocks; see
  [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping).
- **Face states.** Newtonian hydro and MHD with a tabulated EOS limit the specific energy
  (energy per unit mass) in the reconstruction and form the energy density at the cell face
  from it. Limiting the energy density independently would give artificially hot Riemann
  states at steep density gradients, such as an atmosphere. Donor-cell reconstruction and
  ideal-gas runs keep their usual reconstruction.
- **Outputs.** Nine derived output variables show the table state: `hydro_temperature`,
  `hydro_xh2`, `hydro_xion`, `hydro_xhe1`, `hydro_xhe2`, `hydro_gamma1`, `hydro_gamma3m1`,
  `hydro_mu` and `hydro_beta_rad`, plus the same with an `mhd_` prefix. `xh2` needs a table
  with H2, `xhe1` and `xhe2` need helium, and `beta_rad` needs a table with radiation
  pressure. Otherwise the run stops at startup.

### Restarts

Every restart file records which EOS, table, table type, X, Y, radiation, H2 and ZPE
settings produced it. At the next launch they are compared with the EOS being built. Any
difference is fatal unless `allow_reinterpretive_restart = true`. Then the run proceeds with a
warning that the restart is reinterpretive: the conserved state is resumed, but the closure
applied to it is not the one that produced it.

### Checking a table

- A load-time check is always on. It requires finite and positive values, ionization
  fractions that sum to at most 1, `beta_rad` in [0, 1], and energy that increases strictly
  with temperature.
- `lte_debug_checks = true` adds a check of `gamma1 = rho cs2 / p` and a round trip of the
  energy inversion to a relative error of 5e-6. It is expensive, so leave it off in
  production.
- At startup the code prints one line with the table, X, Y, H2, radiation, bounds mode and
  restart status of the EOS.

### Tests

Three tests use a tabulated EOS. They need the table in `eos_tables/` (see above) and do not
build it:

- `tst/test_suite/nr/test_nr_star_surface_eps_cpu.py`: a hydrostatic star whose atmosphere may
  not heat. It runs hydro and MHD with PLM, PPM4, PPMX and WENOZ, plus two low-beta MHD cases.
- `tst/test_suite/unit_tests/test_hydro_plm_cpu.py`: thermodynamic invariants of the PLM
  reconstruction, with and without dual energy and FOFC.
- `test_tabulated_eos_thermo_diagnostics_are_the_table_state` in
  `tst/test_suite/dyngrmhd/test_eos_floor_and_lte_diagnostics_mpicpu.py`: the derived output
  variables against the table state.

No test covers `saha_table`, restarts or remaps with a tabulated EOS,
`allow_reinterpretive_restart` or the table generators. The TDE example decks exercise restarts
and remaps with the Chabrier table, but are not part of the test suite.

### Common problems

| symptom | fix |
| --- | --- |
| Fatal error at construction about `sfloor` | Set `sfloor = 0.0`. |
| Fatal error about the table type | The `eos` name does not match the `table_type` of the table. Use a pair from the [job list](EOS-Tables#the-table-jobs). |
| Abort with an out-of-bounds message | The query left the table range. `clamp` carries on instead of aborting. |
| Restart refuses to start | The EOS or table differs from the one recorded. Use the original, or `allow_reinterpretive_restart = true` if the change is intended. |
| Temperatures wrong by a constant factor in new code | A temperature unit read directly from `<units>` is off by mu m_u / m_H under a tabulated EOS. Use the EOS unit (see [Temperature unit](#temperature-unit)). |

### Limitations

- The loader accepts 18 `lte_` table types, but the generator can build only 6 of them
  (`t13`, `scvh_t13_cp_helm_union` and `chabrier2021_t13_helm_union`, each with and without
  radiation). The loader recognizes 12 masked-union types, and eight of them have no generator
  model; the other four are the `cp_helm_union` and `chabrier2021` types. The simple
  `lte_hhe` and the hybrid `lte_hybrid_hhe_t13` types have none either. Treat the matching
  `eos` names as unmaintained unless a table for them comes from elsewhere.
- No test covers the table generators or `saha_table` (see [Tests](#tests)).

## Further reading

- [Implementation notes](Tabulated-EOS-Implementation-Notes): how the loader, the inverse
  table and the conversion kernels fit together, restart provenance and known issues.
- [EOS Tables](EOS-Tables): generator jobs, building tables and where decks look for them.
- Tomida et al. (2013), Appendix 1: the `t13` chemistry.
- Saumon, Chabrier & Van Horn (1995): the SCvH dense-fluid H/He EOS.
- Chabrier et al. (2021): the dense-fluid H/He EOS used by `chabrier2021_t13_helm_union`.
- Andalman et al. (2025): H2 vibrational zero-point-energy subtraction and radiation-pressure
  treatment referenced by `t13`.
- Timmes & Swesty (2000), Timmes & Arnett (1999): the HELM relativistic e-/e+ pair gas, used
  in the `t13` branch above log10(T/K) = 7 and in the hot dense corner of the union models.
