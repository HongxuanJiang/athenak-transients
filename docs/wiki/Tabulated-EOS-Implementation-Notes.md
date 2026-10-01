# Tabulated EOS: implementation notes

Back to [Tabulated EOS](Tabulated-EOS). This page explains how the tabulated EOS is put
together in the code, for someone who has to change it. Usage and keys are on the main page.

## The big picture

The EOS does its expensive work once, at startup, so that each time step stays cheap.

```
table file --read--> loader --check--> build on the host --upload--> device arrays
                                                                         |
   raw table (rho, T) | inverse table (rho, eps) | floor curve | ceiling curve
                                                                         |
          conversion kernels, once per substep  <------------------------+
```

The inverse table holds temperature, pressure and the other fields on a grid in density and
specific internal energy. It exists because the code evolves energy, but the file is indexed by
temperature.

| file | role |
| --- | --- |
| `src/eos/eos.hpp` | `EOS_Data`: table storage and the interpolation kernels, on device and host. |
| `src/eos/saha_table_utils.hpp`, `lte_table_utils.hpp` | The two loaders, with the load-time checks and restart helpers. |
| `src/eos/saha_table_hyd.cpp`, `saha_table_mhd.cpp` | The per-substep conversion kernels. |
| `src/eos/general_c2p_hyd.hpp` | Single-cell conversion with floors and the atmosphere reset. |
| `src/utils/tr_table.cpp` | The table reader, shared with opacity and other tables. |

## Reading the table

The reader parses the four header blocks and then the raw float64 data: each axis array, then
each field flattened in C order, so the index is `ir*ntemp + it`. The loader requires:

- Natural-log axes. The metadata key `log_axis_base` is optional but, if present, must be `e`.
- The first two axes are `logrho` and `logtemp`, in that order, uniformly spaced to 1e-10
  relative tolerance.
- For `lte_table_*`, X + Y = 1 to 1e-10, with X and Y from the scalars `x_h` and `y_he`
  (default 0.70 and 0.30).

A byte order that differs from the host's is swapped for the whole payload. The `t13` and
union jobs also write diagnostic fields for offline plotting (17 beyond the 11 required ones in
the `chabrier2021` tables). The reader loads every field in the file, but the EOS loader uses
only the required ones, so the diagnostic columns sit in host memory until the loader returns
and are then discarded.

## Building the inverse table

The loader builds three curves on the host and uploads them once.

1. **The inverse table** (`saha_thermo_cache`). For each density row it lays out a uniform
   grid in log(eps) with `cache_eps_factor * ntemp` points. For each energy on that grid it
   searches the monotonic `logeps(logT)` row (`InvertMonotonicFieldAtRow`) and stores T, p, cs2,
   gamma1, gamma3m1, xh2, xion, xhe1, xhe2, mu and beta_rad.
2. **The energy floor** (`saha_logeps_floor`). It is the energy at the floor temperature, at
   each point of the floor grid. `tfloor_kelvin` enters here. `pfloor` raises the floor
   temperature through a search of the monotonic `logpress(logT)` row.
3. **The energy ceiling** (`saha_logeps_ceil`), one value per density row: the energy at which
   cs2 first exceeds `cs_ceil` squared. With `cs_ceil` off, it is the top of the row.

## Looking things up at run time

A conversion takes density and internal-energy density, converts them to cgs, takes the logs,
and does one lookup in the inverse table (`SahaCacheWeightsFromLogRhoEps`). Queries that start
from (rho, T) use a second path that interpolates the raw table.

- With `error`, an out-of-range query calls `Kokkos::abort` on the device or `std::exit` on the
  host. `clamp` is described on the main page under
  [Outside the table](Tabulated-EOS#outside-the-table).
- The energy axis of the inverse table ends at the largest energy of any row. In `clamp` mode,
  a (rho, eps) query above the density-interpolated top of the table therefore skips it. In the
  last temperature interval, log eps is a straight line in log T, so the query gets a weight
  beyond that interval. T, p and cs2 are their values at the top, scaled by the power laws of
  that interval. So neither T(rho, eps) nor p(rho, eps) saturates at the table edge, and
  neither jumps where the query leaves the inverse table.

## Starting up and each substep

At startup:

1. `<hydro>/eos` or `<mhd>/eos` is matched against `saha_table` and the `lte_table_*` names
   (`src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`). A match builds `SahaTableHydro`, `LTETableHydro`
   or the MHD equivalent, whose constructor calls `InitializeSahaTableEOS` or
   `InitializeLTETableEOS`.
2. `TableReader::Table::ReadTable` reads the file into one flat `double[]`. The loader
   validates the shape, the required fields and `table_type`, and runs `RunTableQAChecks`.
3. The raw table and the three curves go to device arrays. Every rank does steps 2 and 3 for
   itself. The inverse table depends only on (rho, eps), so it is not a per-block cost.

In each substep, `TabulatedHydroConsToPrim` and `TabulatedMHDConsToPrim` run one parallel loop
over the pack. For each cell:

1. If sink or black-hole excision is on and the cell is inside it, the cell is reset to the
   excision state.
2. If the density is below `dfloor`, the cell is reset to the atmosphere.
3. Otherwise `eos_general::SingleC2P_GeneralHyd` (or its dual-energy or MHD version) converts
   conserved to primitive variables. This is the inverse-table lookup, followed by the floors
   and ceilings.

When `only_testfloors` is set (the probe pass of first-order flux correction, FOFC), the kernel
only tests whether a floor or ceiling would act and sets the flag `fofc_(m,k,j,i)`. With local
adaptive time stepping ([LAT](Local-Adaptive-Time-Stepping)) on, the loops visit only the
active blocks.

## Where other modules meet the EOS

- **Temperature unit.** The base constructor sets the unit to `Units::temperature_cgs()`,
  which is `v_code^2 * mu * m_u / k_B` with `mu` from `<units>/mu`. Both table loaders then
  override it with `v_code^2 * m_H / k_B`. New code must call
  `MeshBlockPack::TemperatureUnitCGS()`, because reading `punit->temperature_cgs()` directly is
  wrong by mu m_u / m_H under a tabulated EOS.
- **AMR.** A tabulated EOS sets `pmr->prolong_prims = true`, as dual energy does, and
  `src/bvals/prolong_prims.cpp` then uses the general conversion instead of the gamma-law one.
- **Face states.** `src/reconstruct/specific_energy_recon.hpp` limits `eps = eint/rho` with the
  run's own reconstruction, forms `eint_face = rho_face * eps_face`, and applies the EOS floors
  and ceiling. In MHD the Riemann solver builds the face total energy from the corrected
  internal energy.
- **Dual energy.** `ApplyHydroThermalFloors` skips the generic `tfloor` check for a tabulated
  EOS, because the inverse table already enforces it.

## Restart provenance

Every restart file records which EOS produced it, in a `<saha_runtime>` block of its parameter
header (`store_eos_restart_metadata` in `src/outputs/restart.cpp`). On the next launch,
`main.cpp` reads these values from the restart file's own parameter block, before the new input
file and command-line overrides are applied. `PrintStartupSummary` then compares them with the
EOS being built. A difference is fatal unless `allow_reinterpretive_restart = true`.

## Known issues

- **The loader accepts families the generator cannot build.** `validate_job` accepts only the
  models `t13`, `scvh_t13_cp_helm_union` and `chabrier2021_t13_helm_union`. The loader also
  accepts the simple `lte_hhe` types, the hybrid `lte_hybrid_hhe_t13` types, and eight of the
  twelve masked-union types: `lte_scvh1995_hhe[_prad]`, `lte_scvh_t13_union[_prad]`,
  `lte_scvh_t13_helm_union[_prad]` (no `cp`) and `lte_scvh_t13_cp_union[_prad]` (no `helm`).
  Treat these `eos` names as unmaintained.
- **Redundant work.** Every rank re-reads and re-checks the file, and the diagnostic columns
  cost load-time host memory with no run-time benefit.

## Tests

The tests that use a tabulated EOS are listed on the main page under
[Tests](Tabulated-EOS#tests). They check the physics, and none tests the loader's refusals or
the restart check. Two more checks sit outside the suite:

- `scripts/generate_lte_table.py` calls `validate_lte_table(...)` before it writes each job.
  This is an offline check, not run by the AthenaK binary.
- `src/pgen/tests/tabulated_eos_homologous.cpp` is a regression generator for the dual-energy
  update, with no input deck. Build with `-D PROBLEM=tabulated_eos_homologous`, and use a 3-D
  non-relativistic `<hydro>` block with a tabulated EOS and `dual_energy = true`. It starts a
  uniform (rho, T) state in homologous expansion, `u = H(t) x` with `H(t) = H0/(1+H0 t)`. The
  `<problem>` keys are `expansion_rate`, `density_cgs` and `temperature_kelvin`.
