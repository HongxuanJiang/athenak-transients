# Remapping: implementation notes

Developer notes for [Remapping](Remapping). Read that page first for the ideas, and [Remap usage](Remap-Usage) for the keys. This page says how the code is organised, in what order it runs, and where it is known to be weak. The full design is in `docs/remap_module_design.md`.

## The flow in one picture

```
athena -i deck.athinput
   |
   v
ProblemGenerator constructor: run the pgen function       (builds the target state)
   |
   v
remap::MaybeAutoRemap                       (does nothing without an enabled <remap> block)
   |-- load the source restart: parameters, checks, blocks
   |-- resample the gas                      (remap_cc.cpp)
   |-- rebuild B from a vector potential     (remap_fc.cpp, MHD sources only)
   |-- install time, dt, ncycle; refresh the metric; carry output numbers
   `-- run the pgen's post hook
```

## Step by step

1. **Fresh start only.** The fresh-start constructor of `ProblemGenerator` runs the pgen function and then calls `remap::MaybeAutoRemap`. The restart constructor never calls it. A remap target is therefore always launched with `athena -i`, never `-r`.
2. **The pgen function still runs.** In `keep` mode the remap blends against the state it built. In `floor` mode the remap overwrites it, so a pgen can skip its own initialization when `remap::IsAutoRemapEnabled` is true.
3. **Options.** `OptionsFromInput` reads the `<remap>` keys. `LoadAndApplyRemap` refuses an empty `source`, and a call made while LAT windows are active (see [Interactions](#interactions)).
4. **Load.** `LoadRemapSourceData` reads the ASCII parameter dump at the head of the source file. It checks the source against the target (physics, relativity class, EOS, scalars, dimensionality) and then reads the binary blocks. The gas of a block is read only if the block overlaps the local pack's bounding box plus 2 root cells.
5. **Band mode.** `auto` becomes `keep` or `floor` from the relativity class. In GR, `CheckRemapGRConsistency` compares the two parameter sets. It only reads the target input and never adds keys to it.
6. **Newtonian preparation.** The source's outer ghost zones are set to the floor state, and one extra column of thermal energy is built for every source cell.
7. **Loaded hook.** The pgen's `user_remap_loaded_func` runs before anything is overwritten. The TDE pgen restores its black-hole and frame record here, because its excision hook needs it during the next step.
8. **Gas.** `ApplyRemapCC` resamples the gas, and the radiation `i0` where it applies. For an MHD source, `BuildCoveringPotential` and `ApplyRemapFC` then rebuild B.
9. **Bookkeeping.** For a Newtonian run with dual energy, the start-up reseed flag is set to false after a `floor` pass and true after a `keep` pass. Time, `dt`, the previous `dt` and `ncycle` are installed from the source. If the target has an ADM object with a metric callback, the module calls it now, so the metric, lapse and excision masks belong to the source time. This is why dynamical-GR pgens need no remap code.
10. **Banner and post hook.** `<time>` `start_time` and the `<outputN>` numbering are written into the target input, rank 0 prints the banner, and `user_remap_post_func` runs last.

The module does no boundary communication, prolongation, physical boundary conditions or C2P. The normal start-up after the pgen does them. A mid-run caller of `LoadAndApplyRemap`, such as the TDE settle passes, must call `Driver::InitBoundaryValuesAndPrimitives` afterwards.

## The source file

The loader reads the restart in the order the writer produced it:

```
ASCII parameter dump | mesh header (time, dt, ncycle, ...) | block locations and costs
  | refinement ages (adaptive writers) | payload size
  | per block: gas u0 | B faces x1, x2, x3 (MHD) | radiation i0 | stored <adm> metric
```

Every array includes its ghost zones, and a face array is one longer in its own direction. The loader derives the width of each section from the source's own parameter dump, never from the payload size alone. A dual-energy column and an extra passive scalar have the same width but different meanings. Any bytes left after the known sections can only be the stored `<adm>` metric. A leftover of any other size is a fatal error.

## The gas engine

The code is in `remap_cc.cpp`. A sampler finds the source block containing a point, walking down from the finest level, and takes two sub-samples per direction of a trilinear interpolant. Two apply functions then use it, one for each band mode.

**Coincident cells.** For each direction, the code tests that the new cell width equals the source width and that the centres align, both to 1e-10 of a cell. If all three directions pass, the sampler takes one sub-sample at the centre, and the trilinear weights reduce this to an exact copy. With two sub-samples per direction a same-grid remap would apply the filter $[1/8, 3/4, 1/8]$ in each direction.

**Thermal energy.** The code builds the column once per source block (`BuildSourceThermalEnergy`). It holds the source's dual-energy column where there is one, and otherwise $E - |\mathbf{m}|^2/(2\rho)$, floored. The sampler interpolates density, momentum and this column, and sets $E = e_{\rm int} + |\mathbf{m}|^2/(2\rho)$ at the end. The transition band uses the same carried value. A coincident cell keeps the source's own $E$. For an MHD source the loader has already subtracted the magnetic energy from $E$, so $E$ is the gas energy at this point.

**`floor` mode.** Every new cell is written. Near the old boundary the code reconstructs density and pressure in log space with a cubic Hermite curve with limited end slopes, and fades them to the floor values. Cells outside the old box become floor state.

**`keep` mode.** The code reads the weight $w$ from `KeepSampleWeight`, which is 1 inside the old box and falls by a quintic smoothstep over `b_taper_root_cells` source root cells outside it.

```
 w = 1 |__________
       |          \
       |           \____
 w = 0 +------------+--------+----> x
              old box edge   edge + taper
```

A cell with $w = 0$ is not written. A cell with $0 < w < 1$ gets $w\,u_{\rm new} + (1-w)\,u_{\rm pgen}$. The radiation `i0` is blended the same way and clipped at zero. The magnetic-field pass uses the same function to weight the energy add-back.

**Dynamical-GR metric.** On the `<adm>` path the loader evaluates $\sqrt{\gamma}$ at every source cell centre, at the source time (`RemapMetricSampler`). The sampler divides each node by it, and `SampleKeepCellAverage` multiplies by $\sqrt{\gamma}$ at the new cell centre. A coincident cell skips the round trip. Fixed-GR has no such factor. An ADM backend that stores its metric on the target grid has no value at a source cell centre, so the code interpolates the stored columns and warns on rank 0.

## The magnetic-field engine

The code is in `remap_fc.cpp` and the loader. The stages are the four steps described in [Remapping](Remapping#rebuilding-the-magnetic-field), plus the energy add-back.

1. **Restrict.** While reading each source block, the loader adds its face values into a covering grid at the source root level. A face on a block boundary is stored by both neighbours, so it gets half weight from each side. A fine face contributes its share of the root face area.
2. **Invert the curl.** `BuildCoveringPotential` runs prefix sums in the gauge $A_1 = 0$. It then compares the curl of the result with the stored $B_1$. The relative residual gives a warning above 1e-8 and a fatal error above 1e-2. This check runs whatever `b_report` says.
3. **Normalize and sample.** Each component has its boundary-shell mean subtracted. A separable Catmull-Rom sampler evaluates it at the target edges and multiplies it by the taper window, a product of quintic smoothsteps (`SmoothStep5`) that is exactly zero beyond the taper.
4. **Curl per block.** `ApplyRemapFC` takes the discrete curl on each target MeshBlock. Where an edge borders a finer neighbour it averages two half-edge samples, as `gr_torus.cpp` does for its own field.
5. **Energy add-back.** For Newtonian MHD, $E$ gets $+\tfrac12|\mathbf{B}_{\rm new}|^2 - (1-w)\,e_{\rm mag,old}$ in each cell, with the same $w$ as the gas pass. This is the exact inverse of the subtraction made by the loader. In GR nothing is subtracted or added.

In `keep` mode the pass starts from the pgen's own B and owns a face only if its curl stencil touches an edge where the taper window is nonzero. The test is on the window and not on $\mathbf{A}$, because $\mathbf{A}$ can vanish for unrelated reasons, for example a source with $B = 0$. The seam divergence is reduced over MPI and printed by rank 0 whatever `b_report` says.

## Interactions

- **LAT.** The startup remap runs before the first window, so it is always allowed. A mid-run call is allowed only while the problem generator has paused LAT (`Mesh::hydro_lat_suspended`), which makes every step a plain step of the global time step. After the state is replaced, the remap clears the LAT bins and pulls the cycle counters that the LAT gates compare with `ncycle` (last AMR call, last topology change, last rebalance) back to the source's cycle, because it rewinds `ncycle`. The `tde_external` settle steps are the one user: LAT is paused from the startup remap until the last settle pass has run.
- **AMR and SMR.** Fully supported on the target side. A multi-level MHD source needs `b_coarsen_ok`, because B is restricted to the source root grid.
- **MPI and GPU.** The module runs on the host with mirror views and one `deep_copy` back to the device. Gas data is loaded per rank. B data is loaded for all source blocks on every rank, because the covering potential is global. Diagnostics are reduced over MPI or printed by rank 0.
- **Refused source modules.** A source with `<z4c>`, `<turbulence>`, `<turb_driving>` or `<sink_particles>` carries internal state with no length markers in the restart, so the loader cannot skip past it. Sources with `<cce>` or `<shearing_box>` are refused as well.
- **The `<adm>` payload.** The loader does not remap it. It steps over it by its leftover byte count, which must equal exactly the size of the stored ADM section.
- **Reading keys.** The loader reads the source dump and the target keys it audits with read-only accessors, so it never adds a key to a parameter dump. A consumer that reads its own keys from `<remap>` must first check that the block exists, as `tde_external` does, because `GetOrAdd` would create the block and arm the remap.

## Known limitations

1. **Not conservative.** The gas is a plain mean of point samples. A conservative version would need exact intersection volumes of two AMR hierarchies, weighted by $\sqrt{\gamma}$ in GR, and a flux-conservative restriction of B.
2. **Total energy falls** in the Newtonian `floor` pass, because the kinetic-energy variance is dropped. This is intended.
3. **Newtonian `keep` mode has no thermal carry.** The carry lives in the `floor` worker only. `keep` is shared with GR, where the conserved columns are densitized and C2P owns the thermodynamics. A Newtonian run that asks for `keep` explicitly therefore shows the heating of an interpolated $E$.
4. **Multi-level MHD sources lose sub-root B structure.** This is opt-in through `b_coarsen_ok`.
5. **Fine and coarse faces on the target** rely on the edge-averaging contract plus the prolongation done at start-up.
6. **Source B at the old boundary** gives taper sheet currents outside the old box when the domain is widened. This cannot be avoided.
7. **B is read on every rank.** This is a one-time input cost that has not been optimized.
8. **GR: E and B are inconsistent at order $h^2$**, because the gas is remapped as stored while B is rebuilt. The first cycles show a transient.
9. **The `keep` seam.** A seam without divergence would need the pgen's own vector potential, that is, an inverse curl on the target hierarchy. This is not implemented.
10. **3D only.** A 2D source would be an extrusion, not a remap, and the gas sampler addresses source blocks as if they had a third dimension.
11. **`i0` is applied only by the `keep` pass.** The banner reports the group as loaded even when the `floor` pass has not applied it.
12. **The `<adm>` residual test is a byte count.** A future ADM backend that writes a record of another shape would stop the run with a fatal error, not be misread.

## Testing

One test in `tst/` runs a remap, and it checks only the LAT coupling (`tst/test_suite/nr/test_nr_remap_lat_cpu.py`: a hydro remap onto a periodic SMR mesh, with LAT on and off, same final mass and density to truncation error). Nothing in `tst/` checks the remapped state itself. Two things exercise that:

- The built-in pgen `remap_test` (3D, hydro or MHD). Without a `<remap>` block it initializes a smooth analytic state, with B from an analytic vector potential, and prints the errors against it, so that its output can serve as a source restart. With an enabled `<remap>` block it runs the remap and prints the same line from `user_remap_post_func`: `remap_test_errors: divb_max= b_l1= b_max= rho_l1= eint_l1= b_far_max=`. Here `b_far_max` is the field far outside the source box and should be near zero after a widening remap.
- The TDE chain in `inputs/TDE_examples/`, which remaps a Newtonian hydro run four times (steps 2 to 5).

## Key files

| file | role |
| --- | --- |
| `src/remap/remap.hpp` | Public API and the `<remap>` key list |
| `src/remap/remap.cpp` | The orchestration above, band-mode resolution, output carry-over, banner |
| `src/remap/remap_load.cpp` | Source parsing, all source and target checks, covering-grid restriction, GR audit |
| `src/remap/remap_cc.cpp` | Gas samplers, both band modes, thermal-energy column, band curves |
| `src/remap/remap_fc.cpp` | Inverse curl, potential sampler, target curl, residual and seam diagnostics |
| `src/pgen/pgen.cpp` | Calls `MaybeAutoRemap` in the fresh-start constructor |
| `src/pgen/tests/remap_test.cpp` | The analytic test pgen |
