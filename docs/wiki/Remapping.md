# Remapping

## Summary

The remap module loads the state of an existing AthenaK restart file onto a **new mesh** at
the start of a fresh run. The new mesh may have a different domain, resolution, MeshBlock
size, refinement pattern or rank count. No problem-generator code is needed. It carries
hydro or MHD conserved variables (with passive scalars and the dual-energy auxiliary), builds
the magnetic field so that it is divergence-free to machine precision, and continues the time,
time step and cycle number from the source run. The TDE external-restart remap is a thin
wrapper around it.

**Use it** to continue a run on a mesh the checkpoint was not written for: widen or shrink the
domain, change the MeshBlock size for a different rank count, or add or move refinement.

**Do not use it** when you need exact conservation (the transfer is not conservative, see
below), for a 2D run, with an evolved spacetime (`<z4c>`), or together with LAT in the same
run. For a plain continuation on the same mesh, use an ordinary restart.

Cost: one host-side, single-threaded pass at startup (no GPU). It is a one-time cost, not a per-cycle one.

## Quick start

Add a `<remap>` block to the deck of the new run and start it normally with `athena -i`
(not `-r`):

```ini
<remap>
enable = true                    # keep the key explicit so the command line can flip it
source = rst/OldRun.00042.rst    # restart written by the old mesh configuration
```

Things to know before the first run:

- `time/nlim` in the new deck is an **absolute** cycle count that continues the source's
  `ncycle`. It is not a budget after the remap. Add the cycles you want to the source's
  final `ncycle`.
- The remap cannot be combined with LAT. Remap with `time/lat = false`, let the run write a
  restart, then restart that file with `time/lat = true`.
- Re-measure the conserved totals after a remap. They are not carried over exactly.

Parameters that matter most:

| key | what it does |
| --- | --- |
| `source` | Path of the source restart file. Required. |
| `band_mode` | What happens where the new domain extends past the old one. `auto` picks `floor` for Newtonian runs and `keep` for GR runs. |
| `b_taper_root_cells` | Width, in source root cells, over which B fades to zero outside the old box. |
| `b_coarsen_ok` | Needed to accept a multi-level MHD source, at the price of losing fine-level B structure. |
| `transition_band` | Set `false` for pure interpolation, for example in convergence tests. |

See [Remap Usage](Remap-Usage) for the user guide, including GR support and the restriction list.

## Full parameter table

All keys are in `<remap>`. The block only takes effect if it exists.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `enable` | bool | `true` | Master switch, overridable from the command line. |
| `source` | string | `""` | Source restart path. An empty value is fatal once the remap runs. |
| `transition_band` | bool | `true` | Reconstruct the boundary taper. `false` gives pure interpolation. |
| `band_mode` | string | `"auto"` | `auto`, `floor` or `keep` (case-insensitive). `auto` gives `keep` for GR classes and `floor` for Newtonian. An unknown value is fatal. |
| `b_coarsen_ok` | bool | `false` | Accept a multi-level MHD source. B is restricted to the source root grid. Cell-centered fields keep full detail. |
| `b_taper_root_cells` | int | 4 | Smoothstep taper width of the vector potential outside the source box, in source root cells. |
| `b_report` | bool | `true` | Print the face-centered B diagnostics. The sheet-current warning and the residual warning at 1e-8 or above are not gated by this key. |
| `radiation_i0` | bool | `true` | Remap the angular intensities `i0` if both sides carry `<radiation>`. |
| `copy_output_state` | bool | `true` | Carry `<outputN>` `file_number` and `last_time` forward into the new run's own dumps (matched by block name). |

Consumer-private keys may sit in the same block and are read only by the problem generator that
owns them. `tde_external` reads `settle_steps` (int, default 20) and `settle_passes` (int,
default 1) for its post-remap settle passes.

## How it works

**Gas.** For each target cell the module finds the source block that contains it and averages a
few point samples of a trilinear interpolant of the source cell centres. If a target cell
coincides with a source cell (same width and centre), the value is copied exactly, so a remap that only changes the
MeshBlock layout or widens the domain is the identity on the overlap.

The gas energy is carried as **thermal** energy. The module interpolates density, momentum and
internal energy, and rebuilds the total energy at the destination. Interpolating the total
energy would turn the grid-scale kinetic-energy variance, which the target cell cannot represent, into heat.

**Outside the old domain.** The band mode decides what a new cell outside the source box gets.

| band mode | used for | behaviour |
| --- | --- | --- |
| `floor` | Newtonian | Every target cell is overwritten. A transition band tapers the profile toward the ambient floor across the old boundary, and cells fully outside become floor state. |
| `keep` | GR | The state the new problem generator already wrote is the ambient. Cells fully inside the source take the remapped state. Outside the box the remapped state fades out over `b_taper_root_cells` source root cells. |

**Magnetic field.** B is not interpolated directly. The module restricts the source faces to
the source root grid, inverts the curl to get an edge vector potential, interpolates it
smoothly, tapers it to zero outside the box, and takes the curl on the new mesh. A field that
is the curl of one continuous potential is divergence-free on every target block, whatever the interpolation error. The total energy is kept consistent: the new
magnetic energy is added back to the remapped gas energy.

**In GR** the dynamical-metric conserved variables are divided by the metric volume element
before interpolation and multiplied back afterwards, so the interpolation does not mix in the metric curvature.

```
source restart --> read source blocks --> sample gas (CC) --> rebuild B (FC) --> install time/dt/ncycle
                                                                     |
                  user_remap_loaded_func (before apply)               +--> user_remap_post_func (last)
```

## Practical guidance

### What is not conserved

The remap is second-order accurate and **not conservative**. Mass, momentum and energy drift
by an amount of order h squared for what the source resolves, and by order one for structure the target grid
cannot resolve (any coarsening loses what it cannot represent). Two things are exact:
coincident cells are copied, and div(B) is zero on every target block. Making the transfer
conservative would need an exact intersection-volume algorithm and a matching B restriction, and
is out of scope. The total energy also falls by the unresolved kinetic energy, by design.

### What is refused

The run exits with a clear message for:

- `time/lat = true` in the same run.
- `<z4c>` or `<cce>` on either end (permanently), and a source that contains `<turbulence>` or `<turb_driving>`, `<sink_particles>` or a shearing box.
- Special-relativistic runs, 2D runs (3D is required on both ends), and an `isothermal` EOS on either end.
- A mismatch of relativity class, of the CC module type (hydro to hydro, mhd to mhd), or of the fluid EOS (column count, tabulated versus analytic, and `gamma` to 1e-12 for a gamma-law gas). Floors only warn.
- `band_mode = floor` with a GR class.
- A multi-level MHD source without `b_coarsen_ok = true`.
- The retired `<problem>` keys `remap`, `remap_restart_source`, `remap_interpolation`, `remap_settle_steps` and `remap_settle_passes`. On a fresh start `tde_external` refuses `problem/remap = true` and prints the `<remap>` block to write instead. In other cases, and always on a restart, they only warn.

`<units>` differences (`bhmass_msun`, `density_cgs`, `mu`) only warn.

### Checking the result

- Read the banner printed by rank 0: source path, relativity class, band mode, source time and
  cycle, blocks loaded, whether B and `i0` were applied.
- Check `mhd_divb`. It should stay at the 1e-11 level on remap tests.
- Compare mass, momentum and energy totals before and after.
- Check the boundary-field messages (divergence at the taper seam, sheet-current warning).

### Common problems

| symptom | cause and fix |
| --- | --- |
| Fatal error about LAT | Remap with `time/lat = false`, then restart with LAT on. |
| Run stops immediately after the remap | `nlim` is absolute. Add the wanted cycles to the source's final `ncycle`. |
| Fatal error on a multi-level MHD source | Set `b_coarsen_ok = true` and accept the loss of sub-root B structure (a warning is printed). |
| Sheet-current warning | The source B threads the old domain boundary, so a widening remap creates taper currents outside the old box. This is physically unavoidable. |
| Divergence reported at the taper edge (keep mode) | Two divergence-free fields meet at a seam. The outer taper layer can carry div(B) of order the pgen B over h. A same-domain remap has no seam. |
| Transient in the first cycles (GR, strong B) | Gas is remapped verbatim while B is rebuilt, so E and B are inconsistent at order h squared. It is absorbed by C2P, FOFC and excision as a one-time transient. |

### Performance

The whole module is host-only. Cell-centered data is loaded per rank, only for blocks near the
local pack. Face-centered data is read for every source block on every rank, because the
covering-grid vector potential is global. This is a one-time, page-cached I/O cost. The result
is bitwise identical for different rank counts.

### Writing a problem generator that participates

A pgen may enroll `user_remap_skip_func`, `user_remap_loaded_func` and `user_remap_post_func`
inside its pgen function, before the auto-remap call, and should skip its own full
initialization when `remap::IsAutoRemapEnabled(pin)` is true. See [Remap Usage](Remap-Usage).
After a programmatic mid-run call to `LoadAndApplyRemap`, call `Driver::InitBoundaryValuesAndPrimitives(pm)`.

## Further reading

- [Implementation notes](Remapping-Implementation-Notes): samplers, thermal energy, GR
  un-densitization, vector-potential construction, step-by-step flow, code map, numbered limitations.
- [Remap Usage](Remap-Usage): user guide with accuracy statement, GR support and restrictions.
- `docs/remap_module_design.md`: the authoritative design.
