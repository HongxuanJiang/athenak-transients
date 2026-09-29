# Tabulated EOS (LTE / Saha)

## Summary

AthenaK's tabulated equations of state replace the gamma-law/isothermal closures with a
2-D table of `p(rho,T)`, `eps(rho,T)` and ionization/dissociation fractions, precomputed
offline by `scripts/generate_lte_table.py` (`docs/eos_tables.md`: generating, where the
tables live, what is in git) and read at startup by
`src/eos/lte_table_utils.hpp` (H+He family) or `src/eos/saha_table_utils.hpp` (H-only
family). Two fluid classes per family exist for Hydro and MHD
(`SahaTableHydro`/`SahaTableMHD`, `LTETableHydro`/`LTETableMHD`, `src/eos/eos.hpp`), whose `ConsToPrim`/`PrimToCons` are implemented once in
`src/eos/saha_table_hyd.cpp` / `src/eos/saha_table_mhd.cpp` and shared by both table
families. Use it for non-relativistic hydro/MHD problems where composition-dependent
ionization, H2 dissociation, degenerate-electron pressure or radiation pressure matter
enough that a constant `gamma` is wrong (stellar envelopes, TDE debris). It is not implemented for SR/GR (`lte_table_utils.hpp`,
`saha_table_utils.hpp`). Runtime cost is one bilinear lookup per cell per
conversion (no root-find in the hot path — see Physics and algorithm); the fixed cost is
memory: an H+He 640×640 table with default cache settings costs roughly 310 MiB on the
device plus a host mirror of the same size (see Cost, under How it runs).

## Physics and algorithm

**Specific-energy face states.** Newtonian hydro and MHD with a tabulated LTE EOS
limit specific internal energy `eps = eint/rho` with the run's own reconstruction
(PLM, PPM4, PPMX or WENOZ) on its own stencil and form the face energy density as
`eint_face = rho_face * eps_face`, then apply the EOS thermal floors and ceiling.
This also applies to the optional dual-energy auxiliary. Independently limiting
density and energy density can produce an unbounded ratio at a steep atmosphere
density gradient, creating artificial hot Riemann states. Density, velocity,
magnetic-field and tracer reconstruction and the conservative total-energy update
are unchanged; in MHD the Riemann solver assembles the face total energy from the
corrected internal energy. The helper is `src/reconstruct/specific_energy_recon.hpp`;
donor cell, the relativistic solvers and ideal-gas runs retain their existing
reconstruction.

Two independent chemistry backends feed the same table container and the same runtime
code:

* **`saha_table`** (`src/eos/saha_table_utils.hpp`): hydrogen-only Saha ionization
  equilibrium, `table_type = saha_hydrogen_lte`. `X=1, Y=0` are hardcoded; the required fields are only `logpress, logeps,
  logcs2, gamma1, gamma3m1, xion`, and `mu`, `beta_rad`,
  `xh2`, `xhe1`, `xhe2` are synthesized on load (`mu = 1/(1+xion)`, others zero;
  `saha_table_utils.hpp`).
* **`lte_table_*`** (`src/eos/lte_table_utils.hpp`): H+He LTE with several chemistry
  models selected by the table's `table_type` metadata key. Per the
  `scripts/generate_lte_table.py` module docstring, the maintained public models are:
  * `t13` — Tomida et al. (2013) Appendix 1 partition-function chemical equilibrium
    (H2, H, H+, He, He+, He2+, e-), optional radiation pressure and H2 vibrational
    zero-point-energy subtraction (Andalman et al. 2025 style), with the electron gas
    replaced by a HELM-style relativistic ideal e-/e+ pair gas above `log10(T/K)=7`.
  * `scvh_t13_cp_helm_union` — SCvH (Saumon, Chabrier & Van Horn 1995) wherever it's
    valid, a fully-ionized "CP-like" corner where SCvH's own cuts fail, HELM only in the
    hot/dense top-right corner, T13 as the low-density fallback.
  * `chabrier2021_t13_helm_union` — T13 at low density, Chabrier et al. (2021) dense-
    fluid H/He EOS through the pressure-dissociation regime, HELM only above
    `log10(T/K)=8` and outside the tabulated Chabrier coverage.
  A fourth family, `lte_hybrid_hhe_t13_(prad_)lte`, and a simple non-H2
  `lte_hhe_(prad_)lte` family are recognized by the C++ loader
  (`IsHybridLTETableType`/`IsSimpleLTETableType`, `lte_table_utils.hpp`) but have
  no generator job in the current `scripts/generate_lte_table.py` — see Limitations.
  Composition beyond the default X=0.70/Y=0.30 is set per job (`x_h`, `y_he` in each
  `LTE_TABLE_JOBS` entry, `scripts/generate_lte_table.py`); the generator
  currently requires `X + Y = 1` (`validate_job`, `generate_lte_table.py`), so
  **Z is folded into Y**. Where the dense-fluid Chabrier
  block must be built for a Y outside the published mixture tables' range (Y=0.275-0.297),
  the generator falls back to the **additive-volume mixing law** built from the pure-H
  and pure-He Chabrier tables (`chabrier_mixing = "additive_volume"` per job,
  `scripts/generate_lte_table.py`; recorded in the table metadata as
  `chabrier_mixing_additive_volume`).

**Table container.** Both families share the ASCII-header/binary-payload format read by
`src/utils/tr_table.{hpp,cpp}` (also used elsewhere in AthenaK for opacity/microphysics
tables): four ordered blocks —

```
<metadatabegin> key = value ... <metadataend>
<scalarsbegin>  key = value ... <scalarsend>
<pointsbegin>   axis_name = npoints ... <pointsend>
<fieldsbegin>   field_name ... <fieldsend>
```
(`tr_table.cpp`, `ExtractBlock`/`ParseBlock`), followed by raw little/big-endian
`float64` blocks in the order: each axis array, then each field flattened
`ir*ntemp+it`/C-order (`tr_table.cpp`, `generate_lte_table.py`). The
axes are **natural-log**, `logrho = ln(rho/g cm^-3)`, `logtemp = ln(T/K)`, declared by
metadata `log_axis_base = e` (optional; if present it must equal `"e"`,
`lte_table_utils.hpp`, `saha_table_utils.hpp`; the generator always
writes it, `generate_lte_table.py` etc.). **Endianness** is a metadata string
(`endianness = little/big`) checked against the host at load time; a mismatch triggers a
byte-swap of the whole binary payload. The C++ loader requires
`point_info[0] = ("logrho", nrho)`, `point_info[1] = ("logtemp", ntemp)` in that order
(`lte_table_utils.hpp`, `saha_table_utils.hpp`) and a uniform grid
spacing on both axes to `1e-10` relative tolerance (`GetUniformSpacing`,
`lte_table_utils.hpp`). Required runtime fields for `lte_table_*` are `logpress,
logeps, logcs2, gamma1, gamma3m1, xion, xhe1, xhe2, mu, beta_rad`, plus `xh2` when the
table is H2-enabled; for `saha_table` only `logpress,
logeps, logcs2, gamma1, gamma3m1, xion`. Metadata keys
`x_h`/`y_he` (scalars block, defaulting to 0.70/0.30 if absent,
`lte_table_utils.hpp`) set `eos_data.lte_h_mass_fraction`/`lte_he_mass_fraction`,
and the loader currently requires `X+Y=1` to 1e-10. The
t13/union generator jobs also write ~20 extra diagnostic fields (`eps_physical, pgas,
prad, chi_rho, chi_t, cv, cp, entropy, grad_ad, union_valid, source_*`, ...,
`generate_lte_table.py`) purely for offline plotting; `tr_table.cpp` allocates
and loads all fields the header lists (`ReadTable`, `tr_table.cpp`), but the
C++ EOS loader only ever queries the 11 required-field names by string — the diagnostic
columns are read into host RAM and then discarded when the local `TableReader::Table`
goes out of scope at the end of `InitializeLTETableEOS`/`InitializeSahaTableEOS`.

**Runtime `(rho, eint) -> (p, T, cs)` inversion.** The expensive part — inverting
`eps(rho,T)` for `T(rho,eps)` — is done **once, at startup, on the host**, not per cell.
For each table density row `ir` and a **uniform grid in `log(eps)`** of size
`saha_neps = cache_eps_factor * ntemp` (default factor 8; `lte_table_utils.hpp`,
`saha_table_utils.hpp`), `InitializeLTETableEOS`/`InitializeSahaTableEOS` binary-
searches the monotonic `logeps(logT)` row (`InvertMonotonicFieldAtRow`,
`lte_table_utils.hpp`) and stores `T, p, cs2, gamma1, gamma3m1, xh2, xion, xhe1,
xhe2, mu, beta_rad` on that `(log_rho, log_eps)` grid — the `saha_thermo_cache` device
array, `EOS_Data::saha_cache_nvars = 11` fields
(`lte_table_utils.hpp`, `eos.hpp`). At runtime, `EOS_Data::
EvalThermoStateFromRhoEint`/`PressureFromRhoEint`/`HydroInternalEnergyDensityFloor`, etc.
convert `(d, eint)` to cgs, take `log_rho`/`log_eps`, and do a **single bilinear lookup**
on that cache — `SahaCacheWeightsFromLogRhoEps` + `SahaCacheEvalFieldFromWeights` — O(1), no bisection or Newton iteration in the hot conversion path.
A second, independent lookup path bilinearly interpolates the raw table directly in
`(log_rho, log_T)` for `(rho,T)->(p,cs,...)` queries
(`SahaEvalThermoStateFromLogRhoTemp`, `eos.hpp`); `TemperatureFromRhoP` instead
binary-searches the monotonic `logpress(logT)` row at runtime
(`SahaTemperatureFromMonotonicField`, `eos.hpp`) since pressure is not tabulated
on a uniform axis. Out-of-range queries are governed by `SahaBoundsMode`: `error` calls `Kokkos::abort` on device
(`SahaAbortIfOutOfBounds`, `eos.hpp`) or `std::exit` on host; `clamp` clamps at
the cold edge and in density, while above the last temperature row `log p`, `log eps` and
`log cs2` continue as the power law of the table's last temperature interval (radiation
and e-/e+ pairs give `eps, p ∝ T^4` there, so the continuation is asymptotically exact),
and every bounded field holds its top value. A `(rho, eps)` query above the
rho-interpolated top of the table bypasses the inverse cache (whose eps axis ends at the
largest row top): the weight of the last interval's `log eps(log T)` line at the query
scales `T`, `p` and `cs2` from the cache's own values at that top by the interval's power
laws (bounded fields hold the cache top values), so neither `T(rho, eps)` nor
`p(rho, eps)` saturates at the table edge, and neither jumps where the query leaves the
cache. Every interpolation routine above has a `Host*` twin
(`HostSahaRhoWeights`, `HostSahaEvalField`, ... `eos.hpp`) used by pgen-time,
host-side code (star/TDE structure integration) that cannot run inside a `KOKKOS_LAMBDA`.

**Temperature unit.** This is the one place the two EOS families' code-unit
conventions diverge and it is easy to get wrong. `EquationOfState`'s base constructor
sets `eos_data.temp_unit_cgs = pp->punit->temperature_cgs()` for *every* EOS
(`src/eos/eos.cpp`), and `Units::temperature_cgs() = v_code^2 * mu * m_u / k_B`
(`src/units/units.cpp`, with `mu` the `<units>/mu` athinput key, default 1). But
`InitializeLTETableEOS`/`InitializeSahaTableEOS` then **override** that with
`temp_unit_cgs = v_code^2 * m_H / k_B` — hardcoded hydrogen mass, not the run's
`mu` — because the table itself is the mean-molecular-weight authority
(`lte_table_utils.hpp`, `saha_table_utils.hpp`, both:
`length_cgs^2/time_cgs^2 * (1.6735575e-24 / 1.380649e-16)`). **Rule: the code
temperature unit belongs to the active EOS, not to `<units>`.**

## Code map

| File | Role |
|---|---|
| `src/eos/eos.hpp` | `EOS_Data` struct (table storage, device+host interpolation kernels, `ThermoState`), `HydroEOSModel::{saha_table, lte_table}`, `SahaBoundsMode`, and the `SahaTableHydro`/`SahaTableMHD`/`LTETableHydro`/`LTETableMHD` class declarations. |
| `src/eos/eos.cpp` | `EquationOfState` base constructor: reads `dfloor/pfloor/tfloor/sfloor/cs_ceil/vceil` and sets the *default* (ideal-EOS) unit scales, including the temperature unit later overridden by the table loaders. |
| `src/eos/saha_table_utils.hpp` | H-only Saha table loader (`InitializeSahaTableEOS`), QA checks, restart-provenance helpers, startup summary (`saha_table_utils` namespace). |
| `src/eos/lte_table_utils.hpp` | H+He LTE table loader (`InitializeLTETableEOS`), the `eos` string -> `table_type` map (`ExpectedTableTypeForEosName`), same QA/restart/summary machinery, structurally identical to the Saha file. |
| `src/eos/saha_table_hyd.cpp` | `SahaTableHydro`/`LTETableHydro` constructors and `ConsToPrim`/`PrimToCons`, shared `TabulatedHydroConsToPrim`/`...PrimToCons` kernels (excision, dual-energy, LAT active-block filtering, FOFC-flag path). |
| `src/eos/saha_table_mhd.cpp` | MHD analogue of the above, plus cell-centered `B` reconstruction and floor/ceiling event counters. |
| `src/eos/general_c2p_hyd.hpp` / `general_c2p_mhd.hpp` | EOS-agnostic single-cell C2P helpers (`SingleC2P_GeneralHyd[Dual]`, floors, atmosphere reset, velocity ceiling) called by both the tabulated and (for MHD via a template) the ideal-gas paths. |
| `src/utils/tr_table.hpp` / `tr_table.cpp` | Generic table-file reader (shared with opacity and other microphysics tables). |
| `scripts/generate_lte_table.py` | Offline generator for all `lte_table_*` tables; hardcoded job list `LTE_TABLE_JOBS`. |
| `scripts/chabrier2021_eos.py`, `scripts/scvh95_eos.py` | Chemistry backends imported by `generate_lte_table.py` for the dense-fluid and SCvH branches. |
| `eos_tables/*.table` | Shipped tables (see Configuration -> table inventory). |
| `src/pgen/tde_external.cpp`, `src/pgen/tests/polytropic_star.cpp` | Hydrostatic stellar-structure integration against the active EOS (`eos_balanced` mode), pressure/composition floors for initial conditions. |
| `src/pgen/tests/tabulated_eos_homologous.cpp` | Regression pgen for the dual-energy update under a tabulated EOS. |

## Configuration

Grep used: `GetOrAddReal|GetOrAddInteger|GetOrAddBoolean|GetOrAddString|GetReal|
GetInteger|GetBoolean|GetString|DoesParameterExist` over
`src/eos/lte_table_utils.hpp src/eos/saha_table_utils.hpp src/eos/saha_table_hyd.cpp
src/eos/saha_table_mhd.cpp src/eos/eos.cpp src/eos/eos.hpp`, plus a follow-up grep for
`eos ==`/`ExpectedTableTypeForEosName` in `src/hydro/hydro.cpp` and `src/mhd/mhd.cpp` to
find how `<hydro>/eos`/`<mhd>/eos` dispatch to these classes, and for `dual_energy` in
`src/hydro/hydro.cpp` (read there, not in the EOS files, but load-bearing for the
tabulated-EOS floor/AMR behavior below).

All keys below live in `<hydro>` or `<mhd>` (`block` = whichever fluid is configured with
`eos = saha_table` or one of the `lte_table_*` names). Read once per block at
`EquationOfState`/`SahaTableHydro`/`LTETableHydro` construction.

| block/key | type | default | meaning | read at |
|---|---|---|---|---|
| `eos` | string | none (required) | `saha_table`, or one of `lte_table_hhe[_prad]`, `lte_table_t13[_prad]`, `lte_table_scvh1995_hhe[_prad]`, `lte_table_scvh_t13_union[_prad]`, `lte_table_scvh_t13_helm_union[_prad]`, `lte_table_scvh_t13_cp_union[_prad]`, `lte_table_scvh_t13_cp_helm_union[_prad]`, `lte_table_chabrier2021_t13_helm_union[_prad]`, `lte_table_hybrid_hhe_t13[_prad]`. Dispatches the table's required `table_type` metadata. | `lte_table_utils.hpp` (map), `hydro.cpp`, `mhd.cpp` (dispatch) |
| `table` | string | none (required) | path to the `.table` file. | `lte_table_utils.hpp`, `saha_table_utils.hpp` |
| `dfloor` | real | `FLT_MIN` | density floor (generic, all EOS). | `eos.cpp` |
| `pfloor` | real | `FLT_MIN` | pressure floor; for the tabulated EOS this raises the effective *temperature* floor via a monotonic-field inversion, not a raw energy clamp (`floor_temp_from_monotonic`, `lte_table_utils.hpp`). | `eos.cpp` |
| `tfloor` | real | `FLT_MIN` | generic code-unit temperature floor; **superseded by `tfloor_kelvin`** for the tabulated EOS (see next row) — only relevant if `tfloor_kelvin` is absent. | `eos.cpp` |
| `tfloor_kelvin` | real | unset (the code-unit `tfloor` applies) | Kelvin temperature floor, converted to code units through `temp_unit_cgs` and folded into the precomputed `saha_logeps_floor` curve. | `lte_table_utils.hpp`, `saha_table_utils.hpp` |
| `sfloor` | real | `FLT_MIN` | **must be exactly 0** for the tabulated EOS; any `sfloor > 0.0` is a fatal error at construction (entropy floors are not implemented for a tabulated closure). Every shipped deck sets `sfloor = 0.0` explicitly. | `eos.cpp`, guard at `lte_table_utils.hpp`, `saha_table_utils.hpp` |
| `cs_ceil` | real | `0.0` (disabled) | optional sound-speed ceiling; when set, precomputes a per-density `saha_logeps_ceil` curve (energy at which `cs2` first exceeds `cs_ceil^2`). | `eos.cpp`, used at `lte_table_utils.hpp` |
| `vceil` | real | `FLT_MAX` | NR velocity-magnitude ceiling, generic to all EOS. | `eos.cpp` |
| `saha_bounds` | string `error`\|`clamp` | `"error"` | out-of-table-range policy for `saha_table`, and for `lte_table_*` when `lte_bounds` is absent. | `saha_table_utils.hpp` |
| `lte_bounds` | string `error`\|`clamp` | falls back to `saha_bounds` | `lte_table_*`-only override of the bounds policy. | `lte_table_utils.hpp` |
| `saha_debug_checks` | bool | `false` | for `saha_table`, and fallback for `lte_table_*`: enables the expensive `Gamma1 == rho*cs2/p` and `logeps` round-trip QA checks (see Physics -> table container / Tests). | `saha_table_utils.hpp` |
| `lte_debug_checks` | bool | falls back to `saha_debug_checks` | `lte_table_*`-only override. | `lte_table_utils.hpp` |
| `saha_cache_eps_factor` | int | `8` | `saha_table` inverse-thermo cache resolution: `saha_neps = factor * ntemp`. | `saha_table_utils.hpp` |
| `lte_cache_eps_factor` | int | `8` | same, for `lte_table_*`. | `lte_table_utils.hpp` |
| `saha_floor_cache_factor` | int | `16` | `saha_table` density resolution of the precomputed floor curve: `saha_floor_nrho = factor*(nrho-1)+1`. | `saha_table_utils.hpp` |
| `lte_floor_cache_factor` | int | `16` | same, for `lte_table_*`. | `lte_table_utils.hpp` |
| `allow_reinterpretive_restart` | bool | `false` | permits restarting with a different EOS/table than the checkpoint recorded (see How it runs -> restart). | `saha_table_utils.hpp`, `lte_table_utils.hpp` |
| `dual_energy` | bool | `false` | not an EOS key, but load-bearing: also forces AMR to prolong primitives instead of conserved energy for the same reason a tabulated EOS does (see Interactions). | `hydro/hydro.cpp` (and `mhd/mhd.cpp` analogue) |

No key above is read from the deck for `lte_h_mass_fraction`/`lte_he_mass_fraction`/
`lte_has_radiation`/`lte_has_h2`/`lte_zpe_subtracted` — those come from the **table
file's own** metadata/scalars (`table_type`, `radiation_pressure`, `h2_enabled`,
`zpe_subtracted`, scalar `x_h`/`y_he`), not from `ParameterInput`.

Internally, both loaders also read/write a synthetic `<saha_runtime>` block that is not
meant to be hand-edited; it is how restart-compatibility metadata round-trips through the
`.rst` parameter header (`GetOrAdd*` calls at `lte_table_utils.hpp` /
`saha_table_utils.hpp`; written by `src/outputs/restart.cpp` and read
back by `src/main.cpp` — see How it runs).

### Table inventory (`eos_tables/`)

`head -c 4000 <table> | grep -a x_h`:

| file | table_type | X (x_h) | Y (y_he) |
|---|---|---|---|
| `lte_t13_prad_eos.table` | `lte_t13_prad_lte` | 0.70 | 0.30 |
| `scvh_t13_cp_helm_union_prad_640.table` | `lte_scvh_t13_cp_helm_union_prad` | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_640.table` | `lte_chabrier2021_t13_helm_union` (no `_prad`) | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_prad_640.table` | `lte_chabrier2021_t13_helm_union_prad` | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_prad_640_X745Y255.table` | same, `_prad` | 0.745 | 0.255 |
| `chabrier2021_t13_helm_union_prad_640_X550Y450.table` | same | 0.550 | 0.450 |
| `chabrier2021_t13_helm_union_prad_640_X200Y800.table` | same | 0.200 | 0.800 |
| `chabrier2021_t13_helm_union_prad_640_X038Y062.table` | same | 0.380 | 0.620 |
| `chabrier2021_t13_helm_union_prad_640_X000Y100.table` | same | 0.000 | 1.000 |
| `saha_hydrogen.table` | `saha_hydrogen_lte` | 1 (hardcoded) | 0 |

All the `chabrier2021_t13_helm_union*` tables are `640 x 640`; `saha_hydrogen.table` is
`256 x 512` and carries only the 6 required Saha fields (no `xh2/xhe1/xhe2/mu/beta_rad`
columns at all — those are synthesized at load, see Physics).

## How it runs

1. **Dispatch.** `<hydro>/eos` (or `<mhd>/eos`) is matched against `"saha_table"` or the
   `lte_table_*` name list; a match constructs `SahaTableHydro`/`LTETableHydro` (or MHD)
   and calls the base `EquationOfState` constructor (floors/ceilings, default unit
   scales) followed by `InitializeSahaTableEOS`/`InitializeLTETableEOS`
   (`src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`).
2. **Load.** `TableReader::Table::ReadTable` parses the header, allocates one flat
   `double[]` covering every axis and field the header lists, and reads/byte-swaps the
   binary payload. The EOS loader validates shape, required fields,
   `table_type`, and runs `RunTableQAChecks` (finite, positive, `xh2+xion<=1`,
   `xhe1+xhe2<=1`, `beta_rad in [0,1]`, `eps` strictly increasing in `T` at fixed `rho`;
   plus, if `*_debug_checks=true`, the `Gamma1==rho*cs2/p` cross-check and a
   `logeps` inversion round-trip to `5e-6` relative error — `lte_table_utils.hpp`,
   `saha_table_utils.hpp`).
3. **Device upload.** Axes and the raw `(nrho,ntemp)` table upload to `DvceArray1D`/
   `DvceArray3D` (`Kokkos::deep_copy`), followed by construction of the `saha_thermo_cache`
   (uniform-in-`log(eps)` inverse-thermo table), `saha_logeps_floor` (per-density floor
   curve, resolution `floor_cache_factor`), and `saha_logeps_ceil` (per-density `cs_ceil`
   curve) — all built on the host, uploaded once.
4. **Per-substep C2P/P2C.** `TabulatedHydroConsToPrim`/`TabulatedMHDConsToPrim`
   (`saha_table_hyd.cpp`, `saha_table_mhd.cpp`) run one `par_for`/
   `parallel_reduce` over the pack. Per cell: optional sink/BH excision
   (`problem_runtime::GetExcisionState`), atmosphere reset if `d < dfloor`
   (`NeedsHydroAtmosphereReset`), dual-energy or single-energy inversion via
   `eos_general::SingleC2P_GeneralHyd[Dual]`/`...MHD[Dual]`
   (`general_c2p_hyd.hpp`, `general_c2p_mhd.hpp`), which is the bilinear cache
   lookup described in Physics — no bisection per cell. When `only_testfloors` is set
   (the FOFC probe pass) the kernel only checks whether a floor/ceiling *would* fire and
   sets `fofc_(m,k,j,i)=true` without writing state.
5. **LAT interplay.** Both kernels honor `pmy_pack->lat_active_mask_enabled`: when the
   Local Adaptive Timestepping active-block mask is on, the C2P/P2C loops iterate only
   `lat_nactive_thispack` blocks via `lat_active_indices` instead of the full pack
   (`saha_table_hyd.cpp`, `saha_table_mhd.cpp`).
6. **Restart.** Every `.rst` write stores the active EOS's provenance into a synthetic
   `<saha_runtime>` block: `<block>_table_runtime`, `<block>_table_type_runtime` (derived
   from `ExpectedTableTypeForEosName`), `<block>_lte_{x,y,prad,h2,zpe}_runtime`
   (`store_eos_restart_metadata`, `src/outputs/restart.cpp`). On the next launch,
   `main.cpp` reads those keys back out of the **restart file's own** parameter block
   (before the new athinput/cmdline overrides are applied) into `..._from_restart`
   locals, and re-publishes them under `<saha_runtime>/<block>_*_from_restart`. At EOS construction, `PrintStartupSummary` compares the
   restart provenance against the EOS actually being constructed
   (`lte_table_utils.hpp`, `saha_table_utils.hpp`): an EOS-name, table-
   path, table-type, X, Y, radiation, H2, or ZPE mismatch is a **fatal error** unless
   `allow_reinterpretive_restart=true`, in which case it proceeds with a `WARNING:
   restart is reinterpretive` line — conserved state is resumed, but the thermodynamic
   closure evaluated on it is not the one that produced it.
7. **MPI/GPU notes.** Table load/QA/cache-build happen identically and redundantly on
   every rank (no broadcast) — each rank reads its own copy of the `.table` file from
   disk. The `saha_thermo_cache` is allocated per-`MeshBlockPack` and is the same for
   every block in the pack (it's `(rho,eps)`-only, no block dependence), so it is not a
   per-block cost. All interpolation kernels are `KOKKOS_INLINE_FUNCTION` and run on
   device for the C2P/P2C hot path; only the pgen-time stellar-structure integration
   (`Host*` variants, tde_external) runs on host.

### Cost

Device memory is dominated by the inverse-thermo cache, not the raw table:
`saha_table` costs `saha_nvars(11) * nrho * ntemp * 8` bytes, and
`saha_thermo_cache` costs `saha_cache_nvars(11) * nrho * (cache_eps_factor*ntemp) * 8`
bytes — i.e. `cache_eps_factor`x the raw table, by construction (`eos.hpp`,
allocation sites `lte_table_utils.hpp`,
`saha_table_utils.hpp`). For a `640x640` H+He table at the default
`cache_eps_factor=8` (no deck overrides this key anywhere in `inputs/`):
raw table `~= 34.4 MiB`, cache `~= 275.0 MiB`, total `~= 310 MiB` on the device *plus* an
equal-sized host mirror (`saha_table_h`, `saha_thermo_cache_h`) that is never freed after
upload. That is per `<hydro>` or `<mhd>` block, per rank; both blocks active
simultaneously (a hydro+MHD run) doubles it. This assumes `Real = double` (the default
build; `src/athena.hpp`).

## Interactions

* **AMR/SMR.** A tabulated EOS forces `pmr->prolong_prims = true` for its fluid, the same
  switch dual-energy sets, "so the thermal closure stays consistent after prolongation
  and regrids" (`src/hydro/hydro.cpp`; MHD has the analogous check in
  `src/mesh/mesh_refinement.cpp`). Prolongation and the AMR-boundary primitive rebuild
  in `src/bvals/prolong_prims.cpp` branch on `eos.UsesTabulatedLTE()` to call
  `eos_general::SingleC2P_GeneralHyd[Dual]`/`SingleP2C_GeneralHyd`/`...MHD` instead of the
  gamma-law fast path.
* **FOFC.** The `only_testfloors` pass through the same C2P kernel is FOFC's probe: any
  floor/ceiling hit sets the per-cell `fofc_` flag (`saha_table_hyd.cpp`,
  `saha_table_mhd.cpp`), which the first-order-flux-correction machinery then
  consumes upstream — no separate tabulated-EOS FOFC path exists.
* **LAT.** See How it runs step 5 — C2P/P2C restrict to the LAT active-block index list.
* **Restart.** See How it runs step 6 — a dedicated compatibility gate distinct from the
  generic restart machinery, specific to this EOS family.
* **Remap (`src/remap/remap_load.cpp`).** Cross-run remap
  requires the source and target runs to agree on `is_ideal`/`UsesTabulatedLTE()`: "the conserved variables would be decoded against a
  different thermodynamic closure" is a fatal error, not a warning.
* **Dual energy.** `ApplyHydroThermalFloors` skips the generic `tfloor` check for a
  tabulated EOS with the comment "Tabulated LTE EOS floors already enforce tfloor
  through the cached table closure"; the same guard exists
  in `src/reconstruct/thermal_floors.hpp`. The dual-energy switch itself
  (`eint_cons` vs. the auxiliary `eint_aux` field, gated by `dual_energy_eta1`) is
  EOS-agnostic and applies identically on top.
* **Cooling source terms.** `src/srcterms/srcterms.cpp` fatally refuses to pair
  ISM/relativistic/disk cooling with a tabulated EOS: those cooling functions assume a
  gamma-law temperature closure.
* **Outputs.** Nine derived output variables expose the table's `ThermoState` fields
  directly: `hydro_temperature`, `hydro_xh2`, `hydro_xion`, `hydro_xhe1`, `hydro_xhe2`,
  `hydro_gamma1`, `hydro_gamma3m1`, `hydro_mu`, `hydro_beta_rad` (and the `mhd_*`
  equivalents), each requiring `eos.UsesTabulatedLTE()` or the run fatally refuses the
  output block (`src/outputs/basetype_output.cpp`).
* **Units.** See the Temperature unit paragraph in Physics and algorithm.
* **Units-required.** Both loaders fatal if `pp->punit == nullptr` — a `<units>` block is
  mandatory (`lte_table_utils.hpp`, `saha_table_utils.hpp`).

## Limitations and known issues

* **SR/GR unsupported.** Both loaders fatal immediately if the coordinate is special- or
  general-relativistic (`lte_table_utils.hpp`, `saha_table_utils.hpp`).
* **`sfloor` must be zero.** Not merely unused — set it to anything `>0.0` and
  construction fatals. Every shipped deck sets
  `sfloor = 0.0` explicitly for this reason.
* **`X+Y=1` only (no metals in the EOS).** Both the loader
  and the generator (`validate_job`, `generate_lte_table.py`) enforce `Z=0` by
  construction; metallicity is folded into `Y` (see Physics).
* **Loader accepts table families the current generator cannot produce.** The C++ side
  recognizes `table_type` in `{lte_hhe_lte, lte_hhe_prad_lte}` (`IsSimpleLTETableType`),
  `{lte_hybrid_hhe_t13_lte, lte_hybrid_hhe_t13_prad_lte}` (`IsHybridLTETableType`), and a
  12-member masked-union set (`IsMaskedUnionLTETableType`, `lte_table_utils.hpp`)
  of which only 4 — `lte_scvh_t13_cp_helm_union[_prad]` and
  `lte_chabrier2021_t13_helm_union[_prad]` — have a matching generator `model`; the
  other 8, `lte_scvh1995_hhe[_prad]`, `lte_scvh_t13_union[_prad]`,
  `lte_scvh_t13_helm_union[_prad]` (no `cp`), and `lte_scvh_t13_cp_union[_prad]` (no
  `helm`), do not. `hydro.cpp`/`mhd.cpp` will happily construct an
  `LTETableHydro`/`LTETableMHD` for any of these `eos =` names via
  `ExpectedTableTypeForEosName`, but
  `scripts/generate_lte_table.py`'s `validate_job` only accepts `model in {t13,
  scvh_t13_cp_helm_union, chabrier2021_t13_helm_union}` (`generate_lte_table.py`), and a grep for the other `table_type` strings finds no hits anywhere under
  `scripts/` — they are acceptance branches for table families that the current generator does not
  produce. Nothing in the
  shipped `eos_tables/` uses them (see table inventory); treat `eos = lte_table_hhe`,
  `lte_table_scvh1995_hhe*`, `lte_table_scvh_t13_union*`, `lte_table_scvh_t13_helm_union*`,
  `lte_table_scvh_t13_cp_union*`, and `lte_table_hybrid_hhe_t13*` as unmaintained/dead
  paths unless a table for them turns up from elsewhere.
* **Temperature-unit split.** See Physics and algorithm — new code that needs a
  temperature scale must call `MeshBlockPack::TemperatureUnitCGS()`; a direct read of `punit->temperature_cgs()` is wrong by $\mu m_u/m_H$
  under a tabulated EOS. The `dyn_grmhd`/PrimitiveSolver (CompOSE) tables carry their own
  MeV-based convention outside `EOS_Data` and are not covered by the accessor.
* **No entropy floor.** `sfloor` is refused (above); nothing substitutes for it.
* **Diagnostic table columns cost load-time host memory and disk for no runtime benefit**
  — see Physics and algorithm's note on the ~20 extra fields in the `t13`/union tables.
* **Redundant per-rank load.** No broadcast of the table from rank 0; every rank re-reads
  and re-QAs the same file from disk at startup (see How it runs, MPI/GPU notes).

## Tests

`grep -rl lte_table inputs tst` (also matching `saha_table`) turns up **no hits under
`tst/`** — there is no dedicated regression harness. Coverage is:

* `src/pgen/tests/tabulated_eos_homologous.cpp` — a homologous-expansion regression for
  the dual-energy update under a tabulated EOS (`ProblemGenerator::UserProblem` fatals
  unless `eos.UsesTabulatedLTE() && hydro.use_dual_energy`, non-relativistic 3-D hydro
  only, `tabulated_eos_homologous.cpp`). It seeds a uniform `(rho,T)` state in
  homologous expansion (`u = H(t) x`, `H(t)=H0/(1+H0 t)`), applies the same profile as a
  user boundary condition every step, and reports `rho_vol, uaux_vol, temp_vol,
  pressure_vol, uth_vol, rho2_vol, uaux2_vol, volume` as history output. No input deck
  ships for it under `inputs/`; run it by setting `<problem>/pgen_name =
  tabulated_eos_homologous` and the three `<problem>` keys `expansion_rate`,
  `density_cgs`, `temperature_kelvin` (all `GetOrAddReal`) against any `<hydro>` block using an `lte_table_*`/`saha_table` EOS with
  `dual_energy = true`.
* `RunTableQAChecks` (`lte_table_utils.hpp`, `saha_table_utils.hpp`) is a
  load-time self-check, always on for finiteness/positivity/monotonicity, and gated by
  `*_debug_checks=true` for the `Gamma1==rho*cs2/p` and `logeps`-round-trip checks — the
  closest thing to a correctness test that runs on every launch.
* `scripts/generate_lte_table.py` calls `validate_lte_table(...)` before writing each job
  (`generate_lte_table.py`) — an offline, generator-side check, not exercised
  by the AthenaK binary.
* Production decks that exercise the EOS in real problems: the five-step TDE chain
  `inputs/TDE_examples/*.athinput`.

## Example deck fragment

Keys only, copied from `inputs/TDE_examples/tde_01_disruption.athinput` (`<hydro>` block; the
`<units>` block above it sets `mass_cgs/length_cgs/time_cgs/mu` for `1 Msun`, `1 Rsun` at
`R=0.5` code length):

```
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

The H-only counterpart is identical except `eos = saha_table`,
`table = .../saha_hydrogen.table`, and `saha_bounds`/`saha_debug_checks` instead of the
`lte_*` key names.

## References

* Tomida et al. (2013), Appendix 1 — `t13` partition-function chemical equilibrium.
* Saumon, Chabrier & Van Horn (1995) — SCvH dense-fluid H/He EOS.
* Chabrier et al. (2021) — dense-fluid H/He EOS used by `chabrier2021_t13_helm_union`.
* Andalman et al. (2025), EOS section — H2 vibrational ZPE subtraction / radiation
  pressure treatment referenced by `t13`.
* MESA/HELM (Timmes & Swesty 2000, Timmes & Arnett 1999) — relativistic ideal e-/e+ pair
  gas used above `log10(T/K)~7-8` in every union model.
