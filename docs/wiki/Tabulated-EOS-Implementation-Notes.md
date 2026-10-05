# Tabulated EOS: implementation notes

A short tour of how the tabulated equations of state are built in the code. Usage, keys and
guidance are in [Tabulated EOS](Tabulated-EOS.md).

## How the pieces fit

```
generate_lte_table.py  ->  .table file  ->  loader (startup, every rank)
                                               |  validate, build inverse cache and floor curves
                                               v
                  (density, energy) -> one bilinear lookup -> (pressure, temperature, fractions)
```

Tables are made offline. At startup each rank reads the file, checks it and builds a few
arrays on the host, then uploads them to the device. After that every conversion in the run is
a table lookup that runs inside a device kernel.

## The table and the loader

1. **File.** An ASCII header with four blocks (metadata, scalars, axes, field names), then raw
   float64 arrays, each axis followed by each field. The reader is `src/utils/tr_table.cpp`.
2. **Checks.** Both axes must be natural-log and uniform, X + Y must equal 1, and the byte order
   in the header is swapped if it differs from the host.
3. **Families.** The loader knows the simple `lte_hhe`, the hybrid `lte_hybrid_hhe_t13`, the
   `t13` pair and a set of 12 masked-union types (`IsMaskedUnionLTETableType`). The `eos` name
   must map to the file's `table_type` (`ExpectedTableTypeForEosName`).
4. **Diagnostic columns.** The generator writes about 20 extra columns for plotting. They are
   read and then discarded.

## From (density, energy) to the thermodynamic state

For each density row the loader inverts the monotonic energy-versus-temperature column on a
uniform log-energy grid and stores temperature, pressure, sound speed and the fractions
(11 fields). At run time one bilinear lookup in this cache replaces any root find. A second
path interpolates the raw table in (density, temperature) for queries that start from
temperature. Interpolation routines have host twins for problem generators that integrate a
stellar structure on the host.

- `pfloor` raises the effective temperature floor, `tfloor_kelvin` enters the precomputed
  energy-floor curve, and `cs_ceil` builds an energy-ceiling curve per density.
- Out of range, `error` aborts and `clamp` clamps density and the cold edge. Above the hottest
  row the logs of pressure, energy and sound speed continue as the last power law, so neither
  temperature nor pressure saturates.

## Temperature unit

The base constructor sets the temperature unit from `<units>/mu`. Both table loaders then
replace it by $v_{\rm code}^2\,m_H/k_B$, because the table owns the mean molecular weight. New
code must call `MeshBlockPack::TemperatureUnitCGS()`. Reading the `<units>` temperature
directly is off by $\mu m_u/m_H$ under a tabulated EOS.

## Run-time flow

1. `src/hydro/hydro.cpp` and `src/mhd/mhd.cpp` match the `eos` string and build the table class.
2. The loader runs the load-time checks and, with `*_debug_checks`, two more (see the main page).
3. Each substep one kernel visits every cell. It applies excision and the atmosphere reset,
   then inverts through the dual-energy or single-energy helper (`src/eos/general_c2p_hyd.hpp`).
4. In the FOFC probe pass the kernel only sets the floor flag and writes no state.
5. Under LAT the kernel loops over active blocks only.

## Restart provenance

Every restart file stores the EOS name, table path, table type, X, Y and the radiation, H2 and
ZPE settings (`src/outputs/restart.cpp`). At the next start they are compared with the EOS being
built. A mismatch is fatal unless `allow_reinterpretive_restart = true`.

## Interactions in code

- **AMR.** A tabulated EOS makes AMR prolong primitives, and `src/bvals/prolong_prims.cpp` uses
  the general conversion helpers for it.
- **Face states.** `src/reconstruct/specific_energy_recon.hpp` limits the specific energy at
  faces instead of the energy density.
- **Remap.** Source and target must both be ideal or both tabulated
  (`src/remap/remap_load.cpp`).
- **Cooling and outputs.** ISM, relativistic and disk cooling are refused
  (`src/srcterms/srcterms.cpp`). The derived outputs require a tabulated EOS
  (`src/outputs/basetype_output.cpp`).

## Memory

The inverse cache is `cache_eps_factor` times the raw table. A 640 x 640 table takes about
34 MiB raw and 275 MiB for the cache, so about 310 MiB on the device plus an equal host mirror
that is never freed. This is per fluid block and per rank, and assumes double precision.

## Known issues

- **Loader and generator disagree.** The loader accepts 12 masked-union types, but the
  generator has a model for only two of them (`scvh_t13_cp_helm_union` and
  `chabrier2021_t13_helm_union`) and a job for three of the 12 types
  (`lte_scvh_t13_cp_helm_union_prad` and both Chabrier types). The other nine, the simple
  `lte_hhe` and the hybrid family are unmaintained. They still construct a table class if a
  deck names them.
- **X + Y = 1 only.** Both the loader and the generator enforce Z = 0.
- **No entropy floor.** `sfloor` above 0 is refused and nothing replaces it.
- **Redundant work.** Every rank reads and checks the file and keeps its own cache.

## Key files

| file | role |
| --- | --- |
| `src/eos/eos.hpp` | table storage, interpolation kernels, the four table classes |
| `src/eos/lte_table_utils.hpp` | H+He loader, `eos` name to `table_type` map |
| `src/eos/saha_table_utils.hpp` | hydrogen-only loader and shared checks |
| `src/eos/saha_table_hyd.cpp`, `saha_table_mhd.cpp` | conversion kernels |
| `src/eos/general_c2p_hyd.hpp`, `general_c2p_mhd.hpp` | single-cell conversion with floors |
| `scripts/generate_lte_table.py` | offline generator and its job list |
| `scripts/chabrier2021_eos.py`, `scvh95_eos.py` | chemistry backends |

## Tests

- Decks that use a tabulated EOS exist under `tst/inputs/` (`hydro_lte_thermo_tube`,
  `hydro_plm_unit`, and the `hydro_star_*` and `mhd_star_*` decks). They are driven from
  `tst/test_suite/nr/` and `tst/test_suite/unit_tests/`. They need a generated
  `eos_tables/chabrier2021_t13_helm_union_prad_640.table`, and `hydro_plm_unit` also needs the
  non-radiation `chabrier2021_t13_helm_union_640.table`. Neither is in git. No test targets
  the loader itself.
- `src/pgen/tests/tabulated_eos_homologous.cpp` is a homologous-expansion regression for the
  dual-energy update. No deck ships for it. Set `<problem>/pgen_name = tabulated_eos_homologous`
  with `expansion_rate`, `density_cgs` and `temperature_kelvin`, on a tabulated-EOS hydro block
  with `dual_energy = true`.
- The generator validates each table before writing it.
- The five-step chain in `inputs/TDE_examples/` exercises the EOS in production.
