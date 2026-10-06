# Remapping

This page explains how the restart remap works and what it guarantees. To run one (keys, launching, examples, refusals, troubleshooting) see [Remap usage](Remap-Usage).

## Summary

The remap module takes the state saved in an existing AthenaK restart file and puts it onto a **new mesh** at the start of a fresh run. The new mesh can have a different domain, resolution, MeshBlock size, refinement pattern or rank count. No problem-generator (pgen) code is needed.

The module resamples the hydro or MHD conserved variables, including passive scalars and the dual-energy column. It rebuilds the magnetic field so that it is divergence-free to machine precision. It continues the time, time step and cycle number of the source run. The TDE external-restart workflow is a thin wrapper around it.

Two facts to keep in mind:

- The resampling is **second-order accurate but not conservative**. Mass, momentum and energy change a little, and the total energy falls on purpose.
- Two things are exact. A new cell that coincides with an old cell is copied unchanged, and the new B field has zero divergence on every block.

The module runs on the host (no GPU), one thread per MPI rank, once at startup. It adds no cost per cycle.

## How it works

Everything happens once, right after the pgen has built its own initial state on the new mesh:

```
read the source restart --> resample the gas --> rebuild B --> set time, dt, ncycle
                            (cell-centered)     (face-centered)
```

### Resampling the gas

For every cell of the new mesh the module does three things.

1. It finds the source block that contains the cell, looking through the source's refinement levels from the finest down.
2. It averages point samples of a trilinear interpolant built from the source cell centres. There are two samples per direction, a quarter cell width either side of the new cell's centre: eight samples in 3D. Only active source cells are used, never the ghost zones stored in the file.
3. It writes the average into the new cell.

**A cell that coincides with a source cell is copied.** If a new cell has the same width and the same centre as a source cell, the module takes one sample at the centre and the value is copied exactly. A remap that only changes the MeshBlock layout, or only adds cells around the old domain, is therefore the identity on the overlap, except where the `floor` band described below acts. Without this rule the two-sample average would smooth a same-grid remap. It would multiply the grid-scale (Nyquist) amplitude of the gas by 0.125 in 3D, while B, which is exact on a same grid, would keep all of it.

**The energy is carried as thermal energy.** The module interpolates the density, the momentum $\mathbf{m}$ and the thermal (internal) energy $e_{\rm int}$. It then rebuilds the total energy in the new cell as $E = e_{\rm int} + |\mathbf{m}|^2/(2\rho)$. The source's thermal energy is its dual-energy column where it has one. Otherwise it is the total energy minus the kinetic energy, floored.

The module does not interpolate $E$ directly. The average kinetic energy of the samples is larger than the kinetic energy of the averaged momentum, by the grid-scale variance of the velocity. The new cell cannot represent that variance, and interpolating $E$ would hand it to the thermal energy as heat. This matters most in cold, fast gas such as TDE debris, where the internal energy is far below the kinetic energy.

In Newtonian MHD runs the source's magnetic energy is subtracted from $E$ when the file is read, so that this gas pass works with gas energy only. The field pass adds the new magnetic energy back.

### Outside the old domain

At and beyond the boundary of the old domain, the band mode decides what a new cell receives. The `band_mode` key (see the [parameter table](Remap-Usage#parameters)) selects it. The default `auto` picks `floor` for Newtonian runs and `keep` for GR runs.

| mode | the ambient state is | what happens |
| --- | --- | --- |
| `floor` | the target's density and thermal floors | Every new cell is overwritten. Across the old boundary a band fades the profile toward the floor state. Cells fully outside become floor gas. |
| `keep` | whatever the pgen already wrote | Cells inside the old box take the remapped state. Outside it, the remapped state fades out and the pgen state fades in. |

**`floor` mode.** The band reaches from about 3 source cells inside the old boundary to 4 cells outside it (for the usual two ghost zones). It rebuilds the density and pressure profile from the interior with a smooth cubic curve in log space and tapers it to the floor values. Gas that is already close to the floor is simply set to the floor state. Far outside the old box the cell is floor gas, which takes its velocity from the nearest boundary gas where that gas is dense. Before sampling, the module also sets the source's outer ghost zones to the floor state.

The band acts at the boundary of the old domain even if the new domain is the same size. Cells within about 3 source cells of that boundary are reconstructed instead of copied, which changes them unless the gas there is already close to the floor. Use `transition_band = false` or `keep` mode if the boundary layer must be preserved.

**`keep` mode.** A new cell gets $w\,u_{\rm remap} + (1-w)\,u_{\rm pgen}$, where $u$ stands for the conserved variables. The weight is $w = 1$ wherever the old box covers the cell. Outside the box it falls smoothly to 0:

$$w(t) = t^3\,(6t^2 - 15t + 10), \qquad t = 1 - d,$$

where $d$ is the distance beyond the box edge, divided by the taper width of `b_taper_root_cells` source root cells (the largest of the three directions is used). With $w = 0$ the pgen state is left untouched. The radiation intensities `i0` follow the same blend, clipped at zero, and only this mode transfers them. The fade lies entirely outside the old box, because blending inside a fully covered domain would bring back material from the pgen's own initial state. No Newtonian floors or pressure reconstruction are applied: in GR the conserved-to-primitive inversion (C2P), the first-order flux correction (FOFC) and excision enforce the physical bounds.

Setting `transition_band = false` removes the fades. The old box then has a hard edge.

### Rebuilding the magnetic field

B is not interpolated face by face, because interpolating the three face fields independently would not keep the divergence at zero. The module goes through a vector potential $\mathbf{A}$ with $\mathbf{B} = \nabla\times\mathbf{A}$.

1. **Restrict.** The source B faces are averaged onto one uniform grid at the source's root (coarsest) resolution. This is where a multi-level source loses its fine-level B structure.
2. **Invert the curl.** On that grid the module computes an edge vector potential whose discrete curl reproduces the source B, using running sums (the gauge is $A_1 = 0$). Two field components come back exactly. The third comes back exactly only if the source was discretely divergence-free, so the mismatch is measured: a warning above 1e-8 of the maximum field strength, fatal above 1e-2.
3. **Interpolate A.** Each component of $\mathbf{A}$ is interpolated with a smooth (C1) Catmull-Rom cubic scheme, which gives a second-order accurate B after differencing. The mean over the boundary shell is subtracted, so $\mathbf{A}$ is near zero far from a localized field. It is then tapered to zero over `b_taper_root_cells` source root cells outside the old box.
4. **Take the curl on the new mesh.** Each new MeshBlock takes the discrete curl of the interpolated $\mathbf{A}$. Where an edge borders a finer neighbour, the value is the average of two half-edge samples, so that fine and coarse faces agree at SMR and AMR interfaces.

The new B is the discrete curl of one continuous potential, so every new block is divergence-free to machine precision, whatever the interpolation error. This includes the interior of an excised black hole: excision never changes B, and skipping those cells would break the curl identity.

**Magnetic energy.** In Newtonian runs the new magnetic energy goes back into the total energy of each cell:

$$E \;\leftarrow\; E + \tfrac12 |\mathbf{B}_{\rm new}|^2 - (1-w)\, e_{\rm mag,old}.$$

Here $w$ is the gas blend weight of that cell and $e_{\rm mag,old}$ is the magnetic energy the pgen had already put there. In `floor` mode $w = 1$ and the last term is zero. In `keep` mode the share $(1-w)$ of the energy that came from the pgen already contains the pgen's magnetic energy, so only the remapped share takes the new one.

**`keep` mode and the seam.** In `keep` mode the B pass starts from the pgen's own field and overwrites only the faces that the remap reaches, that is, faces whose curl uses a point where the taper is still nonzero. Elsewhere the pgen field stands. Zeroing it instead would leave gas that still carries the pgen's magnetized state with no field to match.

Two divergence-free fields that meet face to face are not divergence-free at the join. The outermost cell layer of the taper can carry $\nabla\cdot\mathbf{B}$ of order $|\mathbf{B}_{\rm pgen}|/h$, with $h$ the cell size. The module measures this and always reports it, whatever `b_report` says. A remap on an unchanged domain has no seam.

### General relativity

The module sorts a run into one of three classes: Newtonian, fixed-GR (a prescribed background such as Kerr-Schild in `<coord>`) and dynamical-GR (an `<adm>` block with a prescribed metric). Source and target must be in the same class. In GR the remap differs in four ways.

- The gas is remapped as stored: no thermal-energy carry, no floors, no dual-energy preparation. The band mode is `keep`.
- In dynamical GR every conserved column is $\sqrt{\gamma}$ times a fluid quantity, with $\gamma$ the determinant of the spatial metric. Interpolating the stored column would put the curvature of the metric, largest at the punctures, into the interpolation. The module therefore divides by $\sqrt{\gamma}$ at each source cell centre, interpolates, and multiplies by $\sqrt{\gamma}$ at the new cell centre. The metric is evaluated at the source time. This works for the analytic binary-black-hole metric backend only. For other `<adm>` backends the stored columns are interpolated as they are, with a warning. Fixed-GR conserved variables are not densitized, so nothing is divided there.
- The magnetic energy is not swapped in and out. The remapped energy and the rebuilt B then disagree at order $h^2$, and C2P, FOFC and excision absorb this as a one-time transient in the first cycles.
- The stored `<adm>` metric is not remapped. The target recomputes it analytically at the source time.

A numerically evolved spacetime (`<z4c>`, `<cce>`) is refused. An interpolation that knows nothing about the Hamiltonian and momentum constraints would produce initial data that violates them while looking fine.

### Time, time step and outputs

The module sets the mesh time, the time step, the previous time step and `ncycle` to the source's values, and sets `<time>` `start_time` to the source time. Unless `copy_output_state = false`, each `<outputN>` block of the new deck that also exists in the source takes the source's `file_number` and `last_time`, so the file numbering continues. With `settle_steps` above 0 the state is remapped once more onto the adapted mesh after that many plain steps, see [Settle steps](Remap-Usage#settle-steps).

### Dual energy and the tabulated EOS

The remap needs the same kind of closure on both ends: both ideal gas (with the same `gamma`) or both tabulated, and the same number of gas columns. Otherwise the conserved variables would be decoded with a different closure, and the run stops. For a tabulated EOS only the kind is compared, not the table, so use the same table on both ends. See [Tabulated EOS](Tabulated-EOS).

The [dual-energy](Dual-Energy) column is carried as follows. If the source has it, the target must have it too. In `floor` mode the column holds the carried thermal energy described above, and the usual start-up reseed from $E$ minus kinetic energy is skipped, so the value that the source trusted in its cold cells survives. If only the target has the column, it is filled from the remapped thermal energy. In `keep` mode the column is blended like the other columns and, in a Newtonian run, reseeded at start-up.

## Guarantees and limits

### What is exact

- **Coincident cells are copied.** A remap that changes only the MeshBlock layout, or widens the domain by whole cells, reproduces the source on the overlap, and conserves exactly there. In `floor` mode this excludes the boundary layer of the old domain, where the band acts (see above).
- **div(B) = 0 on every new block**, by construction, up to the `keep`-mode seam described above.

### What is not conserved

The remap is second-order accurate and **not conservative**. Each new cell is the plain mean of a few point samples. There are no cell-volume weights and no donor/acceptor intersection volumes. Totals of mass, momentum and energy change by an amount of order $h^2$ for structure the source resolves, and by order one for structure the new grid cannot resolve: a coarsening remap loses what it cannot represent. Re-measure the totals after a remap instead of assuming they carry over.

Making the transfer conservative is a different algorithm, not a weighting fix. It needs the exact intersection volumes of two arbitrary AMR hierarchies (a supermesh), weighted by $\sqrt{\gamma}$ in GR, and a matching flux-conservative restriction for B. That is out of scope: the module transfers a state, not a budget.

**The total energy falls on purpose.** In the Newtonian `floor` pass the energy that the new cell cannot represent as kinetic energy is dropped, not turned into heat. The thermal state, which fixes the pressure, the temperature and any light curve, therefore comes through unheated. Two exceptions:

- A cell that coincides with a source cell keeps the source's own $E$.
- `keep` mode has no thermal carry. It interpolates the stored gas energy, so a Newtonian run that asks for `keep` explicitly shows the heating described above.

### Other limits

- A multi-level MHD source loses its fine-level B structure, because B is restricted to the source root grid. The gas keeps full detail.
- If the source B does not vanish at the old boundary, a widening remap creates taper currents outside the old box. The module warns when the boundary-shell field exceeds 1e-2 of the maximum.
- The remap is 3D only on both ends. It needs the same physics modules (no evolved spacetime). It can share a run with LAT, see [Remap with LAT](Remap-Usage#remap-with-lat). [What is refused](Remap-Usage#what-is-refused) gives the full list.
- The module does no boundary exchange, physical boundary conditions or primitive recovery. The normal start-up after the pgen does them. A source whose restart carries a section the module cannot size, other than the stored `<adm>` metric, is refused instead of misread.
- No test in `tst/` checks the remapped state itself. The only one that runs a remap checks the LAT coupling. The built-in `remap_test` pgen prints errors against an analytic state, see [A convergence test](Remap-Usage#a-convergence-test).
- B is read for every source block on every rank, because the vector potential is built on a global grid. Gas data is read per rank, only for the blocks near the local pack. This is a one-time input cost.

## Further reading

- [Remap usage](Remap-Usage): the parameter table, worked examples, refusals and troubleshooting.
- [Dual Energy](Dual-Energy) and [Tabulated EOS](Tabulated-EOS): the closures the remap must match.
- `docs/remap_module_design.md`: the design document.
