# Tabulated EOS (LTE / Saha)

## Summary

The tabulated equations of state replace the gamma-law and isothermal closures with a 2-D
table of pressure, specific internal energy and ionization/dissociation fractions as functions
of density and temperature. The tables are computed offline (see [EOS Tables](EOS-Tables)) and
read at startup. Two families exist: a hydrogen-only Saha table (`saha_table`) and a family of
hydrogen plus helium local-thermodynamic-equilibrium (LTE) tables (`lte_table_*`) that include
H2 dissociation, ionization, degenerate electrons and, optionally, radiation pressure.

**Use it** for non-relativistic hydrodynamics or MHD where ionization, H2 dissociation,
degenerate-electron pressure or radiation pressure matter enough that a constant gamma is
wrong, for example stellar envelopes and tidal-disruption debris.

**Do not use it** for special or general relativity (not implemented), with entropy floors
(`sfloor` must be 0), with ISM, relativistic or disk cooling source terms (refused), or when
you need metals in the EOS (the tables have X + Y = 1, so metals are folded into helium).

The run-time cost is one bilinear table lookup per cell per conversion, with no root find in
the hot path. The fixed cost is memory: an H+He 640 x 640 table with default settings needs about
310 MiB on the device plus an equal host mirror, per fluid block per rank.

## Quick start

Keys from the first TDE example deck, `inputs/TDE_examples/tde_01_disruption.athinput`
(the `<units>` block above it sets the unit system, which is mandatory):

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

The hydrogen-only version is identical except `eos = saha_table`,
`table = .../saha_hydrogen.table`, and `saha_bounds` and `saha_debug_checks` in place of the
`lte_` names.

The keys that matter most:

- **`eos`** selects the table family and chemistry model. It must match the table file.
- **`table`** is the path to the `.table` file.
- **`sfloor = 0.0`** is required. Any positive value is a fatal error.
- **`tfloor_kelvin`** sets the temperature floor in Kelvin. It supersedes the code-unit `tfloor`.
- **`lte_bounds`** (or `saha_bounds`) chooses what happens outside the table: `error` aborts,
  `clamp` extrapolates (see How it works). The example deck uses `clamp`.
- **`dual_energy`** is not an EOS key but is usually switched on with a tabulated EOS (see
  [Dual Energy](Dual-Energy)).

## Full parameter table

All keys live in the `<hydro>` or `<mhd>` block configured with a tabulated `eos`, and are read
once when the EOS is built.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `eos` | string | required | `saha_table`, or `lte_table_<model>[_prad]` for the models listed below. Selects the `table_type` the file must carry. |
| `table` | string | required | Path to the `.table` file. |
| `dfloor` | real | `FLT_MIN` | Density floor (generic to all EOS). |
| `pfloor` | real | `FLT_MIN` | Pressure floor. For a tabulated EOS this raises the effective temperature floor through a monotonic-field inversion, not a raw energy clamp. |
| `tfloor` | real | `FLT_MIN` | Code-unit temperature floor. Only relevant if `tfloor_kelvin` is absent. |
| `tfloor_kelvin` | real | unset (`tfloor` applies) | Temperature floor in Kelvin, converted with the EOS temperature unit and folded into the precomputed energy-floor curve. |
| `sfloor` | real | `FLT_MIN` | Must be exactly 0. Any value above 0 is fatal at construction. |
| `cs_ceil` | real | 0 (off) | Sound-speed ceiling. When set, a per-density energy ceiling curve is precomputed. |
| `vceil` | real | `FLT_MAX` | Velocity-magnitude ceiling (non-relativistic, generic to all EOS). |
| `saha_bounds` | `error` or `clamp` | `error` | Out-of-table policy for `saha_table`, and for `lte_table_*` when `lte_bounds` is absent. |
| `lte_bounds` | `error` or `clamp` | value of `saha_bounds` | Out-of-table policy override for `lte_table_*`. |
| `saha_debug_checks` | bool | `false` | Enable the costly consistency checks at load (see Practical guidance). Also the fallback for `lte_table_*`. |
| `lte_debug_checks` | bool | value of `saha_debug_checks` | Override for `lte_table_*`. |
| `saha_cache_eps_factor` | int | 8 | Resolution of the inverse-thermodynamics cache: energy points per row = factor times the number of temperature points. |
| `lte_cache_eps_factor` | int | 8 | Same, for `lte_table_*`. |
| `saha_floor_cache_factor` | int | 16 | Density resolution of the precomputed floor curve: factor * (nrho - 1) + 1 points. |
| `lte_floor_cache_factor` | int | 16 | Same, for `lte_table_*`. |
| `allow_reinterpretive_restart` | bool | `false` | Allow a restart with a different EOS or table than the checkpoint recorded. |
| `dual_energy` | bool | `false` | Not an EOS key, but it also forces AMR to prolong primitives, as a tabulated EOS does by itself. |

Composition (X, Y), radiation pressure, H2 and zero-point-energy settings are not deck keys.
They come from the table file's own metadata. A `<saha_runtime>` block that appears in
restart headers carries restart-compatibility information and is not meant to be edited.

**Names accepted for `eos`.** `saha_table`, and the following, each with an optional `_prad`
suffix: `lte_table_hhe`, `lte_table_t13`, `lte_table_scvh1995_hhe`, `lte_table_scvh_t13_union`,
`lte_table_scvh_t13_helm_union`, `lte_table_scvh_t13_cp_union`,
`lte_table_scvh_t13_cp_helm_union`, `lte_table_chabrier2021_t13_helm_union`,
`lte_table_hybrid_hhe_t13`. Only `lte_table_t13`, `lte_table_scvh_t13_cp_helm_union` and
`lte_table_chabrier2021_t13_helm_union` (with or without `_prad`) have a generator job and
are maintained. See the limitations below.

## How it works

### Chemistry models

Two independent backends feed the same table container and run-time code.

| family | model | content |
| --- | --- | --- |
| `saha_table` | `saha_hydrogen_lte` | Hydrogen-only Saha ionization equilibrium, X = 1, Y = 0. |
| `lte_table_*` | `t13` | Tomida et al. (2013) partition-function chemical equilibrium (H2, H, H+, He, He+, He2+, e-), optional radiation pressure and H2 zero-point-energy subtraction. Above log10(T/K) = 7 the electron gas is replaced by a HELM-style relativistic e-/e+ pair gas. |
| `lte_table_*` | `scvh_t13_cp_helm_union` | SCvH (Saumon, Chabrier & Van Horn 1995) where valid, a fully ionized "CP-like" corner where SCvH fails, HELM only in the hot dense corner, T13 as the low-density fallback. |
| `lte_table_*` | `chabrier2021_t13_helm_union` | T13 at low density, the Chabrier et al. (2021) dense-fluid EOS through the pressure-dissociation regime, HELM only above log10(T/K) = 8 and outside the Chabrier coverage. |

Composition other than the default X = 0.70, Y = 0.30 is set per generator job. Since the
generator requires X + Y = 1, metals are folded into Y. Where the Chabrier block is needed for a
Y outside the range of the published mixture tables (0.275 to 0.297), the generator uses an
additive-volume mixing law built from the pure-H and pure-He tables.

### The table file

A table is an ASCII header followed by raw binary data. The header has four blocks (metadata,
scalars, axes, field names), and the data follow as float64 arrays: each axis, then each field.

```
header:  metadata | scalars | points (axes) | fields
data:    logrho axis | logtemp axis | field 1 | field 2 | ...
```

- The axes are natural logs, ln(rho in g/cm^3) and ln(T in K), uniform on both axes.
- The byte order is recorded in the header and swapped at load time if it differs from the host.
- Required fields for `lte_table_*`: `logpress`, `logeps`, `logcs2`, `gamma1`, `gamma3m1`, `xion`,
  `xhe1`, `xhe2`, `mu`, `beta_rad`, plus `xh2` for H2-enabled tables. For `saha_table`: `logpress`,
  `logeps`, `logcs2`, `gamma1`, `gamma3m1`, `xion`. The Saha loader synthesizes the others
  (`mu = 1/(1+xion)`, the rest zero).
- The mass fractions X and Y come from the scalars block (default 0.70 and 0.30) and must sum to 1.

### From (density, energy) to (pressure, temperature)

Inverting the energy for the temperature is expensive, so it is done once at startup, on the
host. For each table density row, a uniform grid in log energy (8 times the number of
temperature points by default) is filled by bisection with temperature, pressure, sound speed
and the fractions. At run time each conversion is a single bilinear lookup in that cache. A
separate direct path interpolates the raw table in (density, temperature) for queries that start
from temperature.

### Outside the table

With `error`, an out-of-range query aborts the run. With `clamp`, density and the cold edge are
clamped, and above the hottest row the logs of pressure, energy and sound speed continue as the
power law of the last temperature interval (radiation and pairs give a T^4 law there), while
every bounded field holds its top value. Neither temperature nor pressure saturates at the
table edge.

### Temperature unit

The code temperature unit belongs to the active EOS, not to `<units>`. For a tabulated EOS the
unit is v_code^2 m_H / k_B, with the hydrogen mass hard-coded, because the table is the
authority on the mean molecular weight. It does not use the `<units>/mu` key. Code that needs
a temperature scale must use `MeshBlockPack::TemperatureUnitCGS()`.

### Memory cost

The inverse cache is eight times the raw table by default. For a 640 x 640 H+He table:

| item | size |
| --- | --- |
| raw table | about 34 MiB |
| inverse cache | about 275 MiB |
| total on the device | about 310 MiB |
| host mirror | the same again, never freed |

This is per `<hydro>` or `<mhd>` block per rank, so a run with both doubles it. The figures
assume double precision. The cache size scales with `lte_cache_eps_factor`.

## Practical guidance

### Shipped tables

Tables are built with the generator ([EOS Tables](EOS-Tables)). The ones used by the examples:

| file | table type | X | Y |
| --- | --- | --- | --- |
| `lte_t13_prad_eos.table` | `lte_t13_prad_lte` | 0.70 | 0.30 |
| `scvh_t13_cp_helm_union_prad_640.table` | `lte_scvh_t13_cp_helm_union_prad` | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_640.table` | `lte_chabrier2021_t13_helm_union` (no radiation) | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_prad_640.table` | `lte_chabrier2021_t13_helm_union_prad` | 0.70 | 0.30 |
| `..._prad_640_X745Y255.table` | same, with radiation | 0.745 | 0.255 |
| `..._prad_640_X550Y450.table` | same | 0.550 | 0.450 |
| `..._prad_640_X200Y800.table` | same | 0.200 | 0.800 |
| `..._prad_640_X038Y062.table` | same | 0.380 | 0.620 |
| `..._prad_640_X000Y100.table` | same | 0.000 | 1.000 |
| `saha_hydrogen.table` | `saha_hydrogen_lte` | 1 | 0 |

All Chabrier tables are 640 x 640. `saha_hydrogen.table` is 256 x 512.

### Restrictions and refusals

- Special and general relativity are refused.
- `sfloor > 0` is fatal. There is no entropy floor.
- A `<units>` block is mandatory.
- Pairing ISM, relativistic or disk cooling with a tabulated EOS is fatal, because those
  cooling functions assume a gamma-law temperature closure.
- Remapping requires the source and target runs to use the same kind of closure (both ideal or
  both tabulated), otherwise it is fatal.
- The `eos` names with no generator job (see limitations) are unmaintained.

### Interactions

- **AMR/SMR.** A tabulated EOS forces AMR to prolong primitives, as dual energy does, so the
  thermal closure stays consistent after prolongation and regrids.
- **Dual energy.** The generic `tfloor` check is skipped for a tabulated EOS, because the
  tabulated floors already enforce it. The dual-energy switch applies unchanged on top.
- **FOFC.** Floor and ceiling hits in the conversion set the per-cell flag that first-order
  flux correction uses. There is no separate tabulated path.
- **LAT.** The conversions run only over the active blocks.
- **Face states.** Newtonian hydro and MHD with a tabulated EOS limit the specific energy
  (energy per unit mass) in reconstruction, and form the face energy density from it, instead
  of limiting energy density independently. This avoids artificially hot Riemann states at
  steep atmosphere density gradients. The same applies to the dual-energy auxiliary. Donor
  cell, relativistic solvers and ideal-gas runs keep their existing reconstruction.
- **Outputs.** Nine derived output variables expose the table state: `hydro_temperature`,
  `hydro_xh2`, `hydro_xion`, `hydro_xhe1`, `hydro_xhe2`, `hydro_gamma1`, `hydro_gamma3m1`,
  `hydro_mu`, `hydro_beta_rad`, and the `mhd_*` equivalents. A run that requests them without
  a tabulated EOS is refused.

### Restarts

Every restart file records which EOS, table, table type, X, Y, radiation, H2 and ZPE settings
produced it. On the next launch these are compared with the EOS being built. Any mismatch is
fatal unless `allow_reinterpretive_restart = true`, which proceeds with a warning that the
restart is reinterpretive: the conserved state is resumed, but the closure evaluated on it is
not the one that produced it.

### Checking a table

- A load-time check is always on (finite, positive, ionization fractions summing to at most 1,
  `beta_rad` in [0,1], energy strictly increasing with temperature).
- `lte_debug_checks = true` adds the check Gamma1 = rho cs2 / p and a round trip of the energy
  inversion to 5e-6 relative error. It is expensive, so leave it off in production.
- The startup summary compares the restart provenance with the EOS being built.

### Common problems

| symptom | fix |
| --- | --- |
| Fatal error at construction about `sfloor` | Set `sfloor = 0.0`. |
| Fatal error about the table type | The `eos` name does not match the table's `table_type`. Use the pair from the shipped table list. |
| Abort with an out-of-bounds message | The query left the table range. `clamp` extrapolates instead of aborting. |
| Restart refuses to start | The EOS or table differs from the one recorded. Use the original, or `allow_reinterpretive_restart = true` if the change is intended. |
| Temperatures wrong by a constant factor in new code | A direct read of the `<units>` temperature is off by mu m_u / m_H under a tabulated EOS. Use the EOS unit (see above). |

### Performance and memory

Per-cell cost is one bilinear lookup. Every rank reads and checks its own copy of the table
from disk at startup, with no broadcast. Memory is described above.

### Limitations

- The C++ loader recognizes more `table_type` strings than the current generator can produce
  (the simple `lte_hhe`, the hybrid `lte_hybrid_hhe_t13`, and eight of the twelve masked-union
  types). Treat these `eos` names as unmaintained unless a table for them comes from elsewhere.
- Diagnostic columns in the t13 and union tables are loaded into host memory and then discarded.

## Further reading

- [Implementation notes](Tabulated-EOS-Implementation-Notes): code map, solver internals,
  restart mechanics, known issues, tests.
- [EOS Tables](EOS-Tables): generating tables and where they live.
- Tomida et al. (2013), Appendix 1: the `t13` chemistry.
- Saumon, Chabrier & Van Horn (1995): the SCvH dense-fluid H/He EOS.
- Chabrier et al. (2021): the dense-fluid H/He EOS used by `chabrier2021_t13_helm_union`.
- Andalman et al. (2025): H2 vibrational zero-point-energy subtraction and radiation pressure
  treatment referenced by `t13`.
- Timmes & Swesty (2000), Timmes & Arnett (1999): the HELM relativistic e-/e+ pair gas used above
  log10(T/K) of about 7 to 8 in every union model.
