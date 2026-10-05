# Remapping

This page explains how the remap works: the concepts, the algorithm at user level and an
overview of the parameters. To run a remap, including the full parameter reference, the
restriction list and GR support, see [Remap Usage](../remap_usage.md).

## Summary

The remap module loads the state of an existing AthenaK restart file onto a **new mesh** at
the start of a fresh run. The new mesh may have a different domain, resolution, MeshBlock
size, refinement pattern or rank count. No problem-generator code is needed. It carries
hydro or MHD conserved variables (with passive scalars and the dual-energy auxiliary), builds
the magnetic field so that it is divergence-free to machine precision, and continues the time,
time step and cycle number from the source run. The TDE external-restart remap is a thin
wrapper around it.

**Use it** to continue a run on a mesh the checkpoint was not written for. Typical cases are to
widen or shrink the domain, change the MeshBlock size for a different rank count, or add or move
refinement. For a plain continuation on the same mesh, use an ordinary restart.

**Do not use it** when you need exact conservation (the transfer is not conservative, see
below), for a 2D run, with an evolved spacetime (`<z4c>`), or together with LAT in the same run.

The cost is one host-side, single-threaded pass at startup (no GPU), paid once and not per cycle.

## Quick start

Add a `<remap>` block to the deck of the new run and start it normally with `athena -i`
(not `-r`).

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

## Parameters

All keys live in `<remap>`, and the block only takes effect if it exists. The full reference with
types and defaults is in [Remap Usage](../remap_usage.md). The keys that matter most are

| key | default | what it does |
| --- | --- | --- |
| `source` | `""` | Path of the source restart file. Required. |
| `band_mode` | `auto` | What happens where the new domain extends past the old one. `auto` picks `floor` for Newtonian runs and `keep` for GR runs. |
| `b_taper_root_cells` | `4` | Width, in source root cells, over which B fades to zero outside the old box. |
| `b_coarsen_ok` | `false` | Needed to accept a multi-level MHD source, at the price of losing fine-level B structure. |
| `transition_band` | `true` | Set `false` for pure interpolation, for example in convergence tests. |

The remaining keys are `enable`, `b_report`, `radiation_i0` and `copy_output_state`. Consumer-private
keys may sit in the same block. `tde_external` reads `settle_steps` and `settle_passes` there.

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
by an amount of order $h^2$ for what the source resolves, and by order one for structure the target grid
cannot resolve (any coarsening loses what it cannot represent). Two things are exact:
coincident cells are copied, and div(B) is zero on every target block. Making the transfer
conservative would need an exact intersection-volume algorithm and a matching B restriction, and
is out of scope. The total energy also falls by the unresolved kinetic energy, by design.

### What is refused

The run exits with a clear message for LAT in the same run, `<z4c>` or `<cce>` on either end,
unsupported source physics, special relativity, non-3D runs, an `isothermal` EOS, and any mismatch
of relativity class, CC module type or fluid EOS. The full list is in the restrictions section of
[Remap Usage](../remap_usage.md). A multi-level MHD source also needs `b_coarsen_ok = true`.

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
| Transient in the first cycles (GR, strong B) | Gas is remapped verbatim while B is rebuilt, so E and B are inconsistent at order $h^2$. It is absorbed by C2P, FOFC and excision as a one-time transient. |

### Performance

The whole module is host-only. Cell-centered data is loaded per rank, only for blocks near the
local pack. Face-centered data is read for every source block on every rank, because the
covering-grid vector potential is global. This is a one-time, page-cached I/O cost. The result
is bitwise identical for different rank counts.

### Writing a problem generator that participates

A pgen can enroll hooks that run around the remap. See the pgen participation section of
[Remap Usage](../remap_usage.md).

## Further reading

- [Implementation notes](Remapping-Implementation-Notes.md): the stages, known limitations, tests and key files.
- [Remap Usage](../remap_usage.md): user guide with accuracy statement, GR support and restrictions.
- `docs/remap_module_design.md`: the authoritative design.
