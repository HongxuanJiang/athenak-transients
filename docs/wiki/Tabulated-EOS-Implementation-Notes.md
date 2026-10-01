# Tabulated EOS: implementation notes

Developer-level detail for [Tabulated EOS](Tabulated-EOS). Start there for usage and
parameters.

## Contents

- [Code map](#code-map)
- [Table container and loader](#table-container-and-loader)
- [Inversion and interpolation](#inversion-and-interpolation)
- [Temperature unit](#temperature-unit)
- [Run-time sequence](#run-time-sequence)
- [Restart provenance](#restart-provenance)
- [Interactions in code](#interactions-in-code)
- [Memory cost](#memory-cost)
- [Known issues](#known-issues)
- [Tests](#tests)
- [How the keys were found](#how-the-keys-were-found)

## Code map

| file | role |
| --- | --- |
| `src/eos/eos.hpp` | `EOS_Data` (table storage, device and host interpolation kernels, `ThermoState`), `HydroEOSModel::{saha_table, lte_table}`, `SahaBoundsMode`, and the `SahaTableHydro`, `SahaTableMHD`, `LTETableHydro`, `LTETableMHD` classes. |
| `src/eos/eos.cpp` | `EquationOfState` base constructor: reads `dfloor`, `pfloor`, `tfloor`, `sfloor`, `cs_ceil`, `vceil` and sets the default (ideal-EOS) unit scales, including the temperature unit that the table loaders override. |
| `src/eos/saha_table_utils.hpp` | H-only Saha loader (`InitializeSahaTableEOS`), QA checks, restart-provenance helpers, startup summary. |
| `src/eos/lte_table_utils.hpp` | H+He loader (`InitializeLTETableEOS`), the `eos` string to `table_type` map (`ExpectedTableTypeForEosName`), the same QA, restart and summary machinery. |
| `src/eos/saha_table_hyd.cpp`, `saha_table_mhd.cpp` | Class constructors, `ConsToPrim` and `PrimToCons` for both families (excision, dual energy, LAT active-block filtering, FOFC-flag path). The MHD file adds the cell-centred B reconstruction and floor/ceiling event counters. |
| `src/eos/general_c2p_hyd.hpp`, `general_c2p_mhd.hpp` | EOS-agnostic single-cell conversions (`SingleC2P_GeneralHyd[Dual]`, floors, atmosphere reset, velocity ceiling), shared with the ideal-gas MHD path through a template. |
| `src/utils/tr_table.hpp`, `tr_table.cpp` | Generic table reader, shared with opacity and other microphysics tables. |
| `scripts/generate_lte_table.py` | Offline generator for every `lte_table_*` table, with the hard-coded job list `LTE_TABLE_JOBS`. |
| `scripts/chabrier2021_eos.py`, `scripts/scvh95_eos.py` | Chemistry backends for the dense-fluid and SCvH branches. |
| `src/pgen/tde_external.cpp`, `src/pgen/tests/polytropic_star.cpp` | Hydrostatic stellar structure integrated against the active EOS (`eos_balanced` mode), with pressure and composition floors for initial conditions. |
| `src/pgen/tests/tabulated_eos_homologous.cpp` | Regression generator for the dual-energy update under a tabulated EOS. |
| `src/reconstruct/specific_energy_recon.hpp` | Specific-energy face-state helper. |

## Table container and loader

The format is read by `src/utils/tr_table.{hpp,cpp}` (`ExtractBlock`, `ParseBlock`):

```
<metadatabegin> key = value ... <metadataend>
<scalarsbegin>  key = value ... <scalarsend>
<pointsbegin>   axis_name = npoints ... <pointsend>
<fieldsbegin>   field_name ... <fieldsend>
```

followed by raw `float64` blocks: each axis array, then each field flattened in C order
(`ir*ntemp+it`).

- The axes are natural log. The metadata key `log_axis_base = e` is optional but, if present,
  must equal `"e"`. The generator always writes it.
- `endianness = little/big` is checked against the host. A mismatch byte-swaps the whole payload.
- The loader requires `point_info[0] = ("logrho", nrho)` and `point_info[1] = ("logtemp", ntemp)`
  in that order, and uniform spacing on both axes to 1e-10 relative tolerance
  (`GetUniformSpacing`).
- Scalars `x_h` and `y_he` (default 0.70 and 0.30) set `eos_data.lte_h_mass_fraction` and
  `lte_he_mass_fraction`. The loader requires X + Y = 1 to 1e-10.
- The `saha_table` fields `mu`, `beta_rad`, `xh2`, `xhe1`, `xhe2` are synthesized on load
  (`mu = 1/(1+xion)`, the others zero).
- The t13 and union jobs write about 20 diagnostic fields (`eps_physical`, `pgas`, `prad`,
  `chi_rho`, `chi_t`, `cv`, `cp`, `entropy`, `grad_ad`, `union_valid`, `source_*`, and others)
  for offline plotting. `ReadTable` loads every listed field, but the EOS loader queries only
  the required names by string, so the diagnostic columns are read into host memory and
  discarded when the local table goes out of scope at the end of `InitializeLTETableEOS` or
  `InitializeSahaTableEOS`.
- The table metadata records `chabrier_mixing_additive_volume` when the additive-volume mixing
  law (`chabrier_mixing = "additive_volume"` in the job) was used. The generator's
  `validate_job` enforces X + Y = 1.
- `table_type` families the loader recognizes: `IsSimpleLTETableType` (`lte_hhe_lte`,
  `lte_hhe_prad_lte`), `IsHybridLTETableType` (`lte_hybrid_hhe_t13_lte`,
  `lte_hybrid_hhe_t13_prad_lte`), and a 12-member masked-union set
  (`IsMaskedUnionLTETableType`).

## Inversion and interpolation

For each table density row `ir` and a uniform grid in log(eps) of size
`saha_neps = cache_eps_factor * ntemp`, the loader binary-searches the monotonic `logeps(logT)`
row (`InvertMonotonicFieldAtRow`) and stores `T, p, cs2, gamma1, gamma3m1, xh2, xion, xhe1,
xhe2, mu, beta_rad` on that (log_rho, log_eps) grid. This is the `saha_thermo_cache` device
array, `EOS_Data::saha_cache_nvars = 11` fields.

At run time `EvalThermoStateFromRhoEint`, `PressureFromRhoEint`,
`HydroInternalEnergyDensityFloor` and related functions convert (d, eint) to cgs, take
`log_rho` and `log_eps`, and do one lookup with `SahaCacheWeightsFromLogRhoEps` and
`SahaCacheEvalFieldFromWeights`.

- A second path interpolates the raw table in (log_rho, log_T) for (rho, T) queries
  (`SahaEvalThermoStateFromLogRhoTemp`).
- `TemperatureFromRhoP` binary-searches the monotonic `logpress(logT)` row at run time
  (`SahaTemperatureFromMonotonicField`), since pressure is not tabulated on a uniform axis.
- Out-of-range queries follow `SahaBoundsMode`. `error` calls `Kokkos::abort` on the device
  (`SahaAbortIfOutOfBounds`) or `std::exit` on the host. `clamp` clamps at the cold edge and in
  density. Above the last temperature row, `log p`, `log eps` and `log cs2` continue as the power
  law of the last temperature interval, and every bounded field holds its top value.
- A (rho, eps) query above the rho-interpolated top of the table bypasses the inverse cache,
  whose eps axis ends at the largest row top. The weight of the last interval's `log eps(log T)`
  line at the query scales `T`, `p` and `cs2` from the cache values at that top by the
  interval's power laws. So neither `T(rho, eps)` nor `p(rho, eps)` saturates at the table edge,
  and neither jumps where the query leaves the cache.
- Every interpolation routine has a `Host*` twin (`HostSahaRhoWeights`, `HostSahaEvalField`,
  and others in `eos.hpp`) for pgen-time host code, such as star and TDE structure integration,
  that cannot run inside a `KOKKOS_LAMBDA`.
- `pfloor` raises the effective temperature floor through `floor_temp_from_monotonic`.
  `tfloor_kelvin` is folded into the precomputed `saha_logeps_floor` curve. `cs_ceil` builds a
  per-density `saha_logeps_ceil` curve, the energy at which `cs2` first exceeds `cs_ceil^2`.

## Temperature unit

The base constructor sets `eos_data.temp_unit_cgs = pp->punit->temperature_cgs()` for every EOS
(`src/eos/eos.cpp`), with `Units::temperature_cgs() = v_code^2 * mu * m_u / k_B`
(`src/units/units.cpp`, `mu` from `<units>/mu`, default 1). Both table loaders then override it
with `v_code^2 * m_H / k_B`, that is `length_cgs^2/time_cgs^2 * (1.6735575e-24 / 1.380649e-16)`.
New code that needs a temperature scale must call `MeshBlockPack::TemperatureUnitCGS()`. A
direct read of `punit->temperature_cgs()` is wrong by mu m_u / m_H under a tabulated EOS. The
`dyn_grmhd` and PrimitiveSolver (CompOSE) tables carry their own MeV-based convention outside
`EOS_Data`, and the accessor does not cover them.

## Run-time sequence

1. **Dispatch.** `<hydro>/eos` or `<mhd>/eos` is matched against `saha_table` or the
   `lte_table_*` names. A match constructs `SahaTableHydro`, `LTETableHydro` (or the MHD
   classes), calls the base constructor, then `InitializeSahaTableEOS` or
   `InitializeLTETableEOS` (`src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`).
2. **Load.** `TableReader::Table::ReadTable` parses the header, allocates one flat `double[]`
   for every axis and field, and reads and byte-swaps the payload. The loader validates shape,
   required fields and `table_type`, and runs `RunTableQAChecks`: finite, positive,
   `xh2+xion<=1`, `xhe1+xhe2<=1`, `beta_rad` in [0,1], and `eps` strictly increasing in `T` at
   fixed `rho`. With `*_debug_checks = true` it adds the `Gamma1 == rho*cs2/p` cross-check and a
   `logeps` inversion round trip to 5e-6 relative error.
3. **Device upload.** Axes and the raw (nrho, ntemp) table go to `DvceArray1D` and
   `DvceArray3D`, followed by the `saha_thermo_cache`, `saha_logeps_floor` and `saha_logeps_ceil`
   curves, all built on the host and uploaded once.
4. **Per-substep conversion.** `TabulatedHydroConsToPrim` and `TabulatedMHDConsToPrim` run one
   `par_for` or `parallel_reduce` over the pack. Per cell: optional sink or BH excision
   (`problem_runtime::GetExcisionState`), atmosphere reset if `d < dfloor`
   (`NeedsHydroAtmosphereReset`), then the dual-energy or single-energy inversion through
   `eos_general::SingleC2P_GeneralHyd[Dual]` or the MHD equivalent, which is the cache lookup.
   When `only_testfloors` is set (the FOFC probe pass), the kernel only checks whether a floor
   or ceiling would fire and sets `fofc_(m,k,j,i) = true` without writing state.
5. **LAT.** Both kernels honour `pmy_pack->lat_active_mask_enabled`. With the active-block mask
   on, the loops iterate only `lat_nactive_thispack` blocks through `lat_active_indices`.
6. **MPI and GPU.** Table load, QA and cache build run identically on every rank, each reading
   its own copy of the file. The `saha_thermo_cache` is allocated per `MeshBlockPack` and
   depends only on (rho, eps), so it is not a per-block cost. All interpolation kernels are
   `KOKKOS_INLINE_FUNCTION` and run on the device. Only the pgen-time structure integration
   (`Host*` variants, `tde_external`) runs on the host.

## Restart provenance

Every `.rst` write stores the active EOS's provenance in a synthetic `<saha_runtime>` block:
`<block>_table_runtime`, `<block>_table_type_runtime` (from `ExpectedTableTypeForEosName`) and
`<block>_lte_{x,y,prad,h2,zpe}_runtime` (`store_eos_restart_metadata`, `src/outputs/restart.cpp`).
On the next launch `main.cpp` reads these keys from the restart file's own parameter block,
before the new input file and command-line overrides are applied, into `..._from_restart`
locals, and republishes them under `<saha_runtime>/<block>_*_from_restart`. At EOS construction,
`PrintStartupSummary` compares them with the EOS being built (`lte_table_utils.hpp`,
`saha_table_utils.hpp`). A mismatch in EOS name, table path, table type, X, Y, radiation, H2 or
ZPE is fatal unless `allow_reinterpretive_restart = true`, which proceeds with a
`WARNING: restart is reinterpretive` line. Both loaders are fatal if `pp->punit == nullptr`.

## Interactions in code

- **AMR/SMR.** A tabulated EOS sets `pmr->prolong_prims = true` for its fluid, as dual energy
  does (`src/hydro/hydro.cpp`, and the analogous check for MHD in `src/mesh/mesh_refinement.cpp`).
  `src/bvals/prolong_prims.cpp` branches on `eos.UsesTabulatedLTE()` to call
  `eos_general::SingleC2P_GeneralHyd[Dual]`, `SingleP2C_GeneralHyd` and the MHD versions in
  place of the gamma-law fast path.
- **Face states.** `src/reconstruct/specific_energy_recon.hpp` limits `eps = eint/rho` with the
  run's own reconstruction (PLM, PPM4, PPMX or WENOZ) on its own stencil, forms
  `eint_face = rho_face * eps_face` and applies the EOS thermal floors and ceiling. In MHD the
  Riemann solver assembles the face total energy from the corrected internal energy. Density,
  velocity, field and tracer reconstruction and the conservative total-energy update are unchanged.
- **Dual energy.** `ApplyHydroThermalFloors` skips the generic `tfloor` check for a tabulated
  EOS, because the cached table closure already enforces it. The same guard is in
  `src/reconstruct/thermal_floors.hpp`.
- **Remap** (`src/remap/remap_load.cpp`). The source and target must agree on `is_ideal` and
  `UsesTabulatedLTE()`. Otherwise the conserved variables would be decoded against a different
  closure, which is fatal.
- **Cooling** (`src/srcterms/srcterms.cpp`). ISM, relativistic and disk cooling are refused.
- **Outputs** (`src/outputs/basetype_output.cpp`). Each derived variable requires
  `eos.UsesTabulatedLTE()`.

## Memory cost

`saha_table` costs `saha_nvars(11) * nrho * ntemp * 8` bytes. `saha_thermo_cache` costs
`saha_cache_nvars(11) * nrho * (cache_eps_factor*ntemp) * 8` bytes, which is `cache_eps_factor`
times the raw table by construction. For 640 x 640 at the default factor 8 (no deck overrides
it), the raw table is about 34.4 MiB and the cache about 275.0 MiB, so about 310 MiB on the
device plus an equal host mirror (`saha_table_h`, `saha_thermo_cache_h`) that is never freed. It
is per fluid block and per rank, and assumes `Real = double` (`src/athena.hpp`).

## Known issues

- **Loader accepts families the generator cannot produce.** Of the 12 masked-union types, only
  `lte_scvh_t13_cp_helm_union[_prad]` and `lte_chabrier2021_t13_helm_union[_prad]` have a
  matching generator `model`. The other eight, `lte_scvh1995_hhe[_prad]`,
  `lte_scvh_t13_union[_prad]`, `lte_scvh_t13_helm_union[_prad]` (no `cp`) and
  `lte_scvh_t13_cp_union[_prad]` (no `helm`), do not. `hydro.cpp` and `mhd.cpp` still construct
  an `LTETableHydro` or `LTETableMHD` for any of these `eos` names through
  `ExpectedTableTypeForEosName`. `validate_job` accepts only `model` in `{t13,
  scvh_t13_cp_helm_union, chabrier2021_t13_helm_union}`, and a grep finds no other `table_type`
  string under `scripts/`. The simple `lte_hhe` and hybrid families have no generator job
  either. None of them is used by a shipped table. Treat `eos = lte_table_hhe`,
  `lte_table_scvh1995_hhe*`, `lte_table_scvh_t13_union*`, `lte_table_scvh_t13_helm_union*`,
  `lte_table_scvh_t13_cp_union*` and `lte_table_hybrid_hhe_t13*` as unmaintained.
- **X + Y = 1 only.** Both the loader and `validate_job` enforce Z = 0.
- **Temperature-unit split.** See above.
- **No entropy floor.** `sfloor` is refused and nothing substitutes for it.
- **Diagnostic columns** cost load-time host memory and disk, with no run-time benefit.
- **Redundant per-rank load.** Every rank re-reads and re-checks the file.

## Tests

No input under `tst/` uses a tabulated EOS, so there is no dedicated regression harness.

- `src/pgen/tests/tabulated_eos_homologous.cpp`: a homologous-expansion regression for the
  dual-energy update. `UserProblem` is fatal unless `eos.UsesTabulatedLTE()` and
  `hydro.use_dual_energy` hold, and it supports non-relativistic 3-D hydro only. It seeds a
  uniform (rho, T) state in homologous expansion, `u = H(t) x` with `H(t) = H0/(1+H0 t)`, applies
  the same profile as a user boundary condition every step, and reports `rho_vol`, `uaux_vol`,
  `temp_vol`, `pressure_vol`, `uth_vol`, `rho2_vol`, `uaux2_vol` and `volume` as history output.
  No input deck ships for it. Set `<problem>/pgen_name = tabulated_eos_homologous` and the
  `<problem>` keys `expansion_rate`, `density_cgs` and `temperature_kelvin` (all `GetOrAddReal`)
  against any `<hydro>` block with a tabulated EOS and `dual_energy = true`.
- `RunTableQAChecks`: the load-time self-check described above.
- `scripts/generate_lte_table.py` calls `validate_lte_table(...)` before writing each job. This
  is an offline check, not run by the AthenaK binary.
- The five-step TDE chain `inputs/TDE_examples/*.athinput` exercises the EOS in production use.

## How the keys were found

A grep for `GetOrAddReal|GetOrAddInteger|GetOrAddBoolean|GetOrAddString|GetReal|GetInteger|GetBoolean|GetString|DoesParameterExist`
over `src/eos/lte_table_utils.hpp`, `saha_table_utils.hpp`, `saha_table_hyd.cpp`,
`saha_table_mhd.cpp`, `eos.cpp` and `eos.hpp`, plus a follow-up for `eos ==` and
`ExpectedTableTypeForEosName` in `src/hydro/hydro.cpp` and `src/mhd/mhd.cpp`, and for
`dual_energy` in `src/hydro/hydro.cpp`.
