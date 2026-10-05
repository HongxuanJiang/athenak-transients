# Remapping: implementation notes

Developer notes for [Remapping](Remapping.md). For usage see [Remap Usage](../remap_usage.md). The full design is in `docs/remap_module_design.md`.

## Overview

The remap runs once, on the host, right after the problem generator on a fresh start. It reads the source restart, rewrites the gas and magnetic field on the new mesh, and then installs the source time and cycle.

```
fresh start: pgen runs
      |
      v
 load source restart --> gas pass (cell-centered) --> B pass (face-centered)
      |                                                        |
 user_remap_loaded_func                       time, dt, ncycle, ADM refresh,
 (before apply)                               banner, user_remap_post_func
```

The restart constructor never calls the remap, so a remap target is always launched with `athena -i`.

## Stages

### Loading

The loader parses the ASCII parameter dump of the source restart and then the binary payload. It checks that the physics is supported, that the relativity class, EOS and dimensionality match the target, and that no unsupported module is present. Cell-centered data is read per rank, only for source blocks near the local pack. Face-centered data is read for all source blocks on every rank, because the vector potential is built on a global covering grid.

### Gas

For each target cell the engine finds the source block that contains it and averages a few point samples of a trilinear interpolant of the source cell centres. Three details matter.

1. A target cell that has the same width and centre as a source cell is copied exactly. Without this the sampler acts as a smoothing filter and a same-grid remap would remove most of the grid-scale gas power.
2. The gas energy travels as thermal energy. The engine interpolates density, momentum and internal energy, then rebuilds $E = e_{int} + \tfrac12 |m|^2/\rho$. Interpolating $E$ would turn the unresolved kinetic-energy variance into heat.
3. Outside the old domain the band mode decides the result. In `floor` mode (Newtonian) every cell is overwritten and a smooth band tapers the profile to the ambient floor. In `keep` mode (GR) the target pgen state is the ambient, and the remapped state fades out over `b_taper_root_cells` source root cells beyond the source box.

On the dynamical-GR path the conserved columns are divided by $\sqrt{\gamma}$ at each source cell centre before sampling and multiplied back at the target cell centre. This keeps the metric curvature out of the interpolation.

### Magnetic field

B is never interpolated directly. The steps are as follows.

1. Restrict the source faces to a uniform grid at the source root level.
2. Invert the discrete curl to an edge vector potential, using the gauge $A_1 = 0$ and prefix sums.
3. Interpolate each component with a separable Catmull-Rom cubic and taper it to zero outside the source box.
4. Take the curl on each target MeshBlock, with fine-neighbour edge averaging at refinement boundaries.

Because the target field is the discrete curl of one continuous potential, it is divergence-free to machine precision on every block. This includes excised horizon interiors. The inverse-curl residual warns above 1e-8 and is fatal above 1e-2.

In Newtonian runs the loader removes the magnetic energy from the source energy column before the gas pass, and the B pass adds the new magnetic energy back with the same band weight. GR does not do this, because C2P and FOFC absorb the small inconsistency.

In `keep` mode only faces whose curl stencil touches the taper region are overwritten. Where the pgen field meets the remapped field, divergence-free fields do not join into a divergence-free seam. The seam divergence is measured and reported.

### Orchestration

After both passes the module installs `time`, `dt`, `dtold` and `ncycle` from the source and carries the `<outputN>` numbering forward. If dynamical GR is active it refreshes the analytic ADM metric at the source time, which is why `gr_torus` needs no remap code. Rank 0 then prints a banner and the pgen post hook runs.

## Limitations and known issues

- The remap is not conservative. It is second-order accurate and loses unresolved structure on coarsening. A conservative version would need exact intersection volumes of two AMR hierarchies.
- Total energy falls by the unresolved kinetic energy, by design.
- A multi-level MHD source loses sub-root B structure, so it needs `b_coarsen_ok = true`.
- A source whose B threads the old domain boundary gets taper sheet currents when the domain widens. This cannot be avoided.
- In `keep` mode the seam between the pgen field and the remapped field carries a divergence of order $|B|/h$ in one cell layer. An exactly divergence-free seam would need the pgen vector potential.
- A Newtonian run with an explicit `band_mode = keep` has no thermal-energy carry and shows the heating described above.
- In strongly magnetized GR regions the gas and B are $O(h^2)$ inconsistent. This shows up as a one-time transient in the first cycles.
- The ADM payload is skipped by residual byte count. A future ADM backend that writes a different record would hit a fatal error rather than be misparsed.
- Only 3D is supported, only matching fluid EOS, and `band_mode = floor` is refused in GR.

## Tests

The built-in problem generator `src/pgen/tests/remap_test.cpp` has a source mode that initializes a smooth analytic 3D state, hydro or MHD, uniform, SMR or AMR. In remap mode it applies the remap and prints a `remap_test_errors:` line with the divergence and the error norms against the analytic state. A useful check sequence is as follows.

1. Run the source deck and write a restart.
2. Run the target deck with a `<remap>` block on a different mesh.
3. Confirm `mhd_divb` stays at the 1e-11 level and the error norms shrink under refinement.
4. Confirm a same-grid remap reproduces the source, and that 1 and 4 ranks give bitwise identical results.

For production use, compare the banner and the conserved totals before and after, as described in [Remapping](Remapping.md).

## Key files

| file | role |
| --- | --- |
| `src/remap/remap.cpp` | Orchestration, LAT refusal, band-mode resolution, time and output carry-forward, banner |
| `src/remap/remap_load.cpp` | Restart parsing, compatibility checks, per-block reads, covering-grid restriction |
| `src/remap/remap_cc.cpp` | Gas engine, samplers, thermal energy, transition band |
| `src/remap/remap_fc.cpp` | B engine, inverse curl, potential sampler, target curl, diagnostics |
| `src/remap/remap.hpp` | Public API and options |
| `src/pgen/pgen.cpp` | Calls the remap in the fresh-start constructor only |
| `src/pgen/tests/remap_test.cpp` | Analytic-source test problem |
