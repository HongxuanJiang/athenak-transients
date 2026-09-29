# Dual-Energy Formalism

## Summary

The dual-energy formalism carries a second, independently-evolved conserved variable
alongside the usual total energy `IEN`, so that pressure and temperature can be recovered
in cells where the ordinary inversion loses them to catastrophic cancellation. There are
**two flavours**, and the code chooses between them from the coordinate system rather than
from a key: `MHD::dual_energy_pdv` is set to `!relativistic` at construction
(`src/mhd/mhd.cpp`).

| | Non-relativistic (`dual_energy_pdv == true`) | General-relativistic (`dual_energy_pdv == false`) |
|---|---|---|
| Auxiliary | internal-energy density `eint_aux` | adiabat `kappa = p/rho^Gamma`, conserved as `D*kappa` |
| Equation | `de/dt = -p div(v)` — needs a source term | `d_mu(sqrt(-g) rho kappa u^mu) = 0` — exact, no source term |
| Transport | its own upwind flux, `SetDualEnergyFluxAt` | the ordinary passive-scalar flux |
| Extra machinery | `dual_vf` face-velocity registers, the p dV pass, a LAT divv companion, forced primitive prolongation | none of it |
| Where | Newtonian hydro and Newtonian MHD | fixed-metric GR hydro, fixed-metric GR MHD and dynamical GR MHD |

Hydro carries the same pair of flavours as MHD: `Hydro::dual_energy_pdv`
is `!relativistic` at construction (`src/hydro/hydro.cpp`), the fixed-metric GR hydro
inversion (`src/eos/ideal_grhyd.cpp`) re-solves an `eta1`-rejected cell from the adiabat
through `SingleC2P_IdealSRHyd_Adiabat` (`src/eos/ideal_c2p_hyd.hpp`, the Galeazzi eq. C22
root with `eps = kappa rho^(Gamma-1)/(Gamma-1)` on the always-valid bracket `z in [0, r]`),
publishes `tau/D` into `Hydro::dual_etot_max` for the eta2 pass
(`Hydro::SynchronizeDualEnergyFieldFromAdiabat`, run from `Hydro::DualEnergyStep` ahead of
the send), and seeds the adiabat after the first inversion
(`Hydro::InitializeDualEnergyFieldFromAdiabat`, called from `Driver::Initialize`). The
adiabat takes the hydro scalar flux even when the deck declares no scalars
(`aux_rides_scalar_flux` in `src/hydro/hydro_fluxes_impl.hpp`) and the first-order scalar flux
under FOFC; `dual_vf`, the LAT divv companion and the extra flux-correction payload are
allocated only for the p dV flavour. Pairing the GR hydro adiabat with
`<mesh_refinement>/prolong_primitives = true` is refused, because the primitive round trip
carries only the non-relativistic auxiliary. In GR hydro the cancellation the channel fights is
the kinetic one at large Lorentz factor rather than the magnetic one.

Special relativity has neither and is refused (`src/mhd/mhd.cpp`). The cancellation
the two flavours fight is not the same one — Newtonian `E_tot - KE` in supersonic flow
versus the magnetic cancellation in a GR inversion — but the switching logic is shared:
a per-cell `eta1` ratio test decides which channel supplies the pressure at every
`ConToPrim`, and a separate, coarser `eta2` pass resynchronises the auxiliary from the
energy channel wherever that channel is still trustworthy. Both keys are shared
(`src/mhd/mhd.cpp`), as is the single extra conserved column (`naux = 1`,
`nvars = nmhd + nscalars + naux`, `src/mhd/mhd.cpp`).

The non-relativistic flavour is what the production tabulated-EOS decks use
(`inputs/TDE_examples/*.athinput`), all under `<hydro>`. The GR flavour is used by the shock-tube and small
AMR/cooling test decks (`tst/inputs/mub1_gr_dual.athinput`, which ships with
`dual_energy = false` and is switched on from the test's argument list, and the
`tst/inputs/dual_*.athinput` family), and its benefit for production accretion runs is not
established (see **Limitations**).

## The GR flavour

**What is carried.** In GR the auxiliary is the adiabat

$$\kappa \equiv p/\rho^{\Gamma},$$

with $D\kappa$ as the conserved variable. For smooth adiabatic flow $\kappa$ is constant
along a streamline, so it obeys an exact conservation law,

$$\partial_\mu\!\left(\sqrt{-g}\,\rho\,\kappa\,u^\mu\right) = 0,$$

and needs **no source term at all**. Being a per-unit-mass quantity densitized exactly like
$D$, it is advected by the ordinary passive-scalar flux: `SetScalarFluxesAt` skips the
auxiliary column only when `dual_pdv` is true, so on the GR path the scalar loop covers it
(`src/mhd/mhd_fluxes_impl.hpp`). The same is true of the first-order fallback:
`mhd_fofc.cpp`'s scalar loop runs to `nvars` and its `dual_enabled` guard is
`dual_energy_pdv`, so the adiabat picks up a first-order upwind scalar flux there
(`src/mhd/mhd_fofc.cpp`). The non-relativistic `eint_aux` form was **not**
ported: its $\mathrm{d}e/\mathrm{d}t = -p\,\nabla\!\cdot\!v$ update is non-conservative in
GR and would need $\sqrt{-g}$ and $\nabla_\mu u^\mu$, discarding exactly the exactness the
adiabat gives for free.

**Why it is needed, and what it fixes.** Both GR inversions are Kastaun et al. (2021)
schemes with a single unknown $\mu = 1/(hW)$. They recover the gas energy as
`eoverD = qbar - mu*rbarsq + 1` (`src/eos/primitive-solver/primitive_solver.hpp`) or
`eps = w*(qbar - mu*rbar) + z2/(w+1)` (`Equation44`, `src/eos/ideal_c2p_mhd.hpp`). `qbar`
carries $-b^2/2$, so in a magnetically dominated cell the gas energy is what survives a
cancellation between two terms of order $b^2/D$, and its relative error grows like
$\varepsilon_{\rm mach}\,(b^2/u)$. Crucially **the temperature and the Lorentz factor come
out of the same root**, so a cell whose pressure collapses onto `tfloor` also returns a
displaced $W$ — the floor deletes the enthalpy that was carrying part of the conserved
momentum $r = hWv$, and the velocity has to make up the difference. That is the
isolated-bright-cell pattern seen in jet Lorentz-factor maps.

**The auxiliary channel.** `Equation44Adiabat` / `SingleC2P_IdealSRMHD_Adiabat`
(fixed metric, `src/eos/ideal_c2p_mhd.hpp`) and `AdiabatRootFunctor` /
`ConToPrim(..., kappa)` (dynamical, `src/eos/primitive-solver/primitive_solver.hpp`)
are the same root find with the one line that computes the specific energy replaced by
$\varepsilon = \kappa\rho^{\Gamma-1}/(\Gamma-1)$ — a power, not a difference. They are
separate functions rather than a runtime branch because these kernels are instruction
bound. `q` never enters that branch, which is the same statement as: **the energy equation
is ignored for the pressure in those cells**. It is ignored, not overwritten: the cell keeps
its conserved energy, the fluxes are built from the auxiliary primitives, the conserved
energy goes on integrating conservatively, and the next inversion finds it wanting again and
takes the adiabat again. This is the non-relativistic flavour's rule (`ideal_hyd.cpp`
regenerates $U$ only after a floor or ceiling) and Enzo's. The auxiliary is also the
backstop for a solve that fails outright: it is tried before any failure floor, and a cell
it recovers is not a failed cell (no failure floor, no FOFC flag, no event count).

Because the channel fires exactly where the energy channel reads cold, a one-sided injection
of $u_{\rm aux} - u_{\rm cons}$ into the total energy at each such inversion would heat the gas,
so total-energy conservation is checked directly. The closed-box check is the unit-level
record: on the MUB1 tube made periodic (400 cells, $\Gamma = 2$, $t = 0.4$, no floors active)
the total energy is conserved to $1.5\times10^{-14}$ with the channel off, at
$\eta_1 = 10^{-4}$ and $10^{-2}$, and with the channel forced into every cell
($\eta_1 = 10^{30}$, $1.6\times10^{-14}$).

**The adiabat is bounded above by the cell's own energy budget.** Nothing in the transport
bounds $\kappa$ from above, and a cell whose density reaches its floor turns
$\kappa = (D\kappa)/D$ into a finite numerator over a floor. The bound is physical rather
than numerical — an adiabat implying more internal energy than the cell owns is not a fluid
state — so the specific energy is capped at $q = \tau/D$ inside both root functions
(`epsmax`, `src/eos/ideal_c2p_mhd.hpp`), which is what bounds the pressure the gas
sees, and the transported column is capped at the same budget once per stage in the eta2
pass (`SynchronizeDualEnergyFieldFromAdiabat`, using the $\tau/D$ and density the last
inversion published — one stage old, which a bound can afford). The cap sits in the
resync pass rather than in the inversions' publication step because that kernel is memory
bound and already writes both columns of every owned cell, so the power it costs is free
there and would not be in the instruction-bound inversion. $\tau/D$ is a ratio of two conserved
variables carrying the same densitization, so no metric enters it.

**How much it actually buys, measured.** A standalone host experiment on exact conserved
states over $b^2/u \in [10^2, 10^{18}]$, twenty points per decade, with every floor
verified inactive:

- the energy channel's relative error in $p$ is $\approx \varepsilon_{\rm mach}\,(b^2/u)$
  and is independent of temperature. It declares an outright C2P failure from
  $b^2/u \approx 7\times10^{9}$, and past $\sim\!10^{16}$ it returns silent garbage with no
  flag raised — the regime `eta1` exists to catch.
- the auxiliary channel is **not** independent of $b^2/u$. Its error is set by the
  *momentum* cancellation rather than the energy one and scales as
  $\varepsilon_{\rm mach}\,b^2/(\rho h) = \varepsilon_{\rm mach}\,\sigma/h$: the fluid part
  of $S_i$ is a fraction $\rho h/b^2$ of the total, so rounding $S_i$ to double already
  costs that much. Heating the gas 100× at fixed $b^2/u$ moves the auxiliary error 100×
  and leaves the energy error unchanged, which is the discriminating test.
- the practical advantage is therefore $\approx h/\varepsilon$: measured 750–1100× at
  $T = 10^{-2}$ and $\approx 21\times$ at $T = 1$. The auxiliary channel also **never
  failed** (0 failures in ~700 solves, 7–9 iterations, against 12–25 and repeated failures
  for the energy channel).

The consequence for a magnetized funnel: the auxiliary channel removes the *energy*
cancellation, and what then limits the answer is the magnetization itself. Dual energy is
not a cure for high $\sigma$; it moves the wall from $b^2/u$ to $\sigma$, where the
remedies are mass loading (`sigma_ceiling`, `bsq_over_u_ceiling`).

**eta1 in GR.** The test is `u_int > eta1 * tau`, with `tau` the conserved energy without
rest mass — the exact analogue of the Newtonian `eint_cons > eta1 * E_tot`, since neither
includes rest mass. Both backends implement it identically
(`src/eos/ideal_grmhd.cpp`, `src/eos/primitive_solver_hyd.hpp`). It runs
*after* the energy solve (unlike the Newtonian one, where `eint_cons` is free to compute),
so a cell that fails it pays for a second inversion; that is why a smaller `eta1` is
preferred in GR than the Newtonian default. Three properties are deliberate:

- **A failed auxiliary solve leaves the energy-channel answer standing**
  (`src/eos/ideal_grmhd.cpp`, `src/eos/primitive_solver_hyd.hpp`). The
  test may be wrong about which channel is better, never about leaving the cell with one.
- **The surviving solve's floor verdict replaces the discarded one, not the union of the
  two** (`src/eos/ideal_grmhd.cpp`, `src/eos/primitive_solver_hyd.hpp`).
  A cell whose energy channel collapsed onto its floor and which the adiabat then recovered
  cleanly is not a floored cell; reporting it as one drops it to first-order fluxes and
  diffuses exactly the region this channel exists to resolve. One flag is the exception on
  the dynamical backend: its conserved-floor flag is raised in the preamble that runs ahead
  of either root functor, so it belongs to both channels and is kept as the union.
- **Taking the auxiliary forces the conserved energy to be rewritten but does not raise the
  FOFC flag** (`src/eos/primitive_solver_hyd.hpp`). The rewrite is how the energy
  equation is discarded rather than merely ignored; the flag is withheld for the same
  reason as the previous point.

**eta2 in GR.** Written per unit **rest mass**, which is what makes it free of
$\sqrt{\det g}$: $q = \tau/D$ is a ratio of two conserved variables carrying the same
densitization and $\varepsilon = u/\rho$ a ratio of two primitives. Resync where
$\varepsilon > \eta_2 \max_{27}(q)$ (`SynchronizeDualEnergyFieldFromAdiabat`). In the
non-relativistic limit $q \to (KE + e_{\rm int} + e_{\rm mag})/\rho$ and the test reduces to
the Newtonian one exactly. The resync is what keeps shock heating: entropy is not conserved
across a shock and only the energy channel knows by how much, so an auxiliary that is never
reset silently loses it.

**`tau/D` is published by the inversions, because neither pass can form it.** The eta2 pass
lives in `mhd_dual_energy.cpp`, which has no metric, and the conserved array does not hold
$\tau$ on either backend: the fixed-metric GRMHD path evolves $T^t{}_t + D$ in the `IEN`
slot, which is **negative for ordinary states**. Reading the stencil back from `u0(IEN)`
would leave `qmax` at exactly zero, so the gate would degenerate to `eps > eta2*1e-18` — true in every
cell with a positive pressure — and the advected adiabat would be thrown away and recomputed
from the previous stage's pressure every stage, which is precisely the ill-conditioned
number the channel exists to avoid. Both inversions therefore publish the per-cell ratio
into `MHD::dual_etot_max` (`src/mhd/mhd.hpp`) at the point where they still have the
conserved state in hand — `src/eos/ideal_grmhd.cpp` and
`src/eos/primitive_solver_hyd.hpp` — and the eta2 stencil is taken over what they
published (the `qmax` stencil in `SynchronizeDualEnergyFieldFromAdiabat`).

**Seeding.** The GR auxiliary is seeded from the primitives *after* the first inversion
(`src/driver/driver.cpp`), not with the non-relativistic one before it: $\kappa$ is a function of $p$ and $\rho$, and on a
restart the primitives do not exist until the inversion has run.
`InitializeDualEnergyFieldFromTotal` returns immediately on the GR path
(its `dual_energy_pdv` early return), and its caller must therefore **not** clear
`dual_energy_needs_init`: otherwise $\kappa$ would stay at its zero initialization, hand
$p = 0$ to every cell the eta1 test routed to the auxiliary, and collapse the domain onto
the density floor within tens of cycles. The seed covers the **ghost range**, not the active
zone (the ghost-inclusive loop bounds of `InitializeDualEnergyFieldFromAdiabat`): it runs
after the last conserved exchange of
`Driver::Initialize`, so an active-only seed would leave every ghost cell unseeded for the first
stage's flux kernel to read. A checkpoint that already carries the column is authoritative
and is not re-seeded.

**Repairs republish the adiabat.** The magnetization ceiling loads mass at fixed specific
internal energy, and the density and energy floors add mass or energy outright, so every one
of them changes $\kappa = p/\rho^\Gamma$. Those cells republish $\kappa$ from the repaired
primitives instead of carrying a ratio describing a state the cell has left
(`src/eos/ideal_grmhd.cpp`). A Lorentz-factor ceiling is excluded on purpose: it
changes no thermodynamic quantity. Whenever a repair rewrites the conserved density, the
auxiliary rides it — `cons(dual) = cons(IDN)*kappa_pub`
(`src/eos/ideal_grmhd.cpp`, `src/eos/primitive_solver_hyd.hpp`),
and the same on the primitive-to-conserved path (`src/eos/ideal_grmhd.cpp`).

**What it costs.** One extra conserved column and its share of every array that scales with
`nvars`; no face-velocity buffer and no compression pass, which the non-relativistic flavour
needs. A cell that fails `eta1` runs the inversion twice.

## Physics and algorithm

**Auxiliary internal-energy equation (non-relativistic only).** `eint_aux` obeys the
Lagrangian pdV-work equation `d(eint)/dt = -p * div(v)`, advanced with an operator-split
compression update after the ordinary Riemann-solver flux has already advected it as a
conserved density. Two closed forms are used depending on the EOS
(`src/hydro/hydro_dual_energy.cpp`; `ApplyDualEnergyFormalism` in
`src/mhd/mhd_dual_energy.cpp`):

```
if (eos.is_gamma_law)  eint *= exp(-(gamma - 1) * divv * dt)         // exact for gamma-law
else                    eint -= PressureFromRhoEint(dens, eint) * divv * dt   // explicit Euler
```

`divv` is built from the *interface* velocities stored in the `dual_vf` face-velocity
buffer, which are exactly the velocities the Riemann solver already produced for that face:
HLLC/HLLD supply the contact/star-region velocity, everything else uses the upwind velocity
`mass_flux/dens` (`SetDualEnergyFluxAt`, `src/hydro/hydro_fluxes_impl.hpp`;
`src/mhd/mhd_fluxes_impl.hpp`). Thermal floors and ceilings are re-applied before
and after the compression step (`eos_general::ApplyHydroThermalFloors` /
`ApplyMHDThermalFloors`) so the tabulated Saha inversion never sees an out-of-table state.
The whole pass is gated on the flavour: `MHD::DualEnergyStep` returns immediately unless
`dual_energy_pdv` (`MHD::DualEnergyStep`), and `ApplyDualEnergyFormalism`
repeats the guard (the same file, the head of `ApplyDualEnergyFormalism`).

**Advection.** The auxiliary lives at conserved-variable index `dual_energy_idx =
nhydro(+nmhd) + nscalars` and is reconstructed by the *same* PLM/PPM4/PPMX/WENOZ/DC kernels
as every other variable (`src/reconstruct/recon.hpp`, whose parallel index runs over
`[0, nvars)`). For Newtonian hydro and MHD with any reconstruction and a tabulated LTE EOS, both gas and
auxiliary face energies are then rebuilt from limited specific energy
(`src/reconstruct/specific_energy_recon.hpp`) to bound `eint/rho` across steep density gradients.
Which flux it then gets depends on the flavour: the non-relativistic
auxiliary takes its own upwind flux `mass_flux * (eint/dens)` from the reconstructed face
density/internal-energy pair, floored by `ApplyReconstructedHydroThermalFloors`
(`src/reconstruct/thermal_floors.hpp`); the GR adiabat takes the plain passive-scalar
flux `mass_flux * kappa` described above.

**Per-cell switch (`eta1`), non-relativistic form.** At every `ConToPrim` the primitive
internal energy `w.e` is chosen between the total-energy-derived value and the
independently-tracked auxiliary with the same ratio test in all EOS branches —
`src/eos/ideal_hyd.cpp`, `src/eos/ideal_mhd.cpp` (mirror), and the tabulated-EOS
helpers `src/eos/general_c2p_hyd.hpp` /
`src/eos/general_c2p_mhd.hpp`:

```
eint_cons  = u.e - KE  (- e_mag for MHD)        // from the just-updated total energy
use_cons_e = (eint_cons > 0) &&
             (eta1 <= 0 || eint_cons > eta1 * max(u.e, 1e-18))
w.e        = use_cons_e ? eint_cons : eint_aux  // eint_aux is floor/ceiling-clamped first
```

`u.e` in the denominator is the *total* conserved energy at that call, not the kinetic
energy — this is the standard Bryan-et-al-style `e_int/E_tot > eta1` switch. **The choice is
not written back into the auxiliary at this call**: regardless of which branch fed `w.e`,
`cons(dual_idx)`/`prim(dual_idx)` are set to the clamped auxiliary value
(`src/eos/ideal_hyd.cpp`), so the aux column keeps evolving on its own equation even
while the total-energy-derived value is what the rest of the code sees as pressure.

**Coarser resync (`eta2`), non-relativistic form.** A second, separate pass —
`SynchronizeDualEnergyFieldFromTotal` (`src/hydro/hydro_dual_energy.cpp`,
`MHD::SynchronizeDualEnergyFieldFromTotal`) — overwrites `eint_aux = eint_cons` wherever
`DualEnergySyncEligible` holds (`src/hydro/hydro_dual_energy.cpp`,
the file-local `DualEnergySyncEligible` in `src/mhd/mhd_dual_energy.cpp`):

```
eligible <=> eint_cons > 0  &&  (eta2 <= 0 || eint_cons > eta2 * max_{3x3x3 nbhd}(E_tot))
```

`eta2 <= 0` makes the resync unconditional; `eta2 > 0` only resyncs where the
total-energy-derived value is itself locally reliable, protecting the aux field near strong
shocks. The GR flavour has its own pass, `SynchronizeDualEnergyFieldFromAdiabat`, and the
two run at different points of the stage because they read different things. The
non-relativistic pass needs the neighbours' *new* total energies, so it runs **after the
boundary exchange**, at the head of `ConToPrim`. The GR pass reads only what this rank
already holds -- the density the stage just updated, the advected `D*kappa`, and the
primitives and `tau/D` ratios the previous inversion published -- so it runs **before the
conserved send**, from `MHD::DualEnergyStep`, over owned cells only; the exchange,
prolongation and physical boundary conditions then install the resynchronised adiabat in
every ghost cell like any other conserved variable. That placement is what lets the GR
auxiliary use the ordinary recovery machinery (interior-first split, ghost-band recovery)
with no special case, where the non-relativistic flavour has
to be excluded from both.

## Code map

| File | Role |
|---|---|
| `src/hydro/hydro.hpp` / `src/mhd/mhd.hpp` | `use_dual_energy`, `dual_energy_pdv` (MHD only), `dual_energy_idx`, `dual_energy_eta1/2`, `dual_vf`, `lat_dual_vf_reflux`, `dual_etot_max` (MHD only), `dual_excise_mask`; method declarations |
| `src/hydro/hydro.cpp` / `src/mhd/mhd.cpp` | Reads the 3 keys, picks the flavour, validates rsolver/coords/EOS, sets `naux`/`dual_energy_idx`, allocates buffers |
| `src/hydro/hydro_dual_energy.cpp` | `InitializeDualEnergyFieldFromTotal`, `ApplyDualEnergyFormalism` (p dV), `SynchronizeDualEnergyFieldFromTotal` (eta2), `SynchronizeRestrictedDualEnergyField`, `RepairRefinedDualEnergyState` |
| `src/mhd/mhd_dual_energy.cpp` | The same five, plus the GR pair `InitializeDualEnergyFieldFromAdiabat` and `SynchronizeDualEnergyFieldFromAdiabat`, and the flavour branches inside `SynchronizeRestrictedDualEnergyField` and `RepairRefinedDualEnergyState` |
| `src/eos/ideal_grmhd.cpp` | Fixed-metric GR: eta1 test, `kappa` publication and budget cap, `tau/D` publication, conserved repair, P2C |
| `src/eos/primitive_solver_hyd.hpp` | Dynamical GR: the same four; its P2C does not carry the auxiliary, and need not -- on this path it is called only by problem generators, before the adiabat is seeded |
| `src/eos/ideal_c2p_mhd.hpp` | `Equation44Adiabat`, `IllinoisRootEquation44Adiabat`, `SingleC2P_IdealSRMHD_Adiabat` |
| `src/eos/primitive-solver/primitive_solver.hpp`, `eos.hpp` | `AdiabatRootFunctor`, the root-choice line; `HasAdiabat`/`GetAdiabat`/`GetTemperatureFromAdiabat` |
| `src/hydro/hydro_fluxes_impl.hpp` / `src/mhd/mhd_fluxes_impl.hpp` | `SetDualEnergyFluxAt` (p dV flavour) and the `SetScalarFluxesAt` skip that hands the GR adiabat to the scalar path |
| `src/hydro/hydro_fofc.cpp` / `src/mhd/mhd_fofc.cpp` | `SetDualEnergyFOFCFlux`, first-order fallback for the p dV flavour; the GR adiabat rides the first-order scalar loop |
| `src/dyn_grmhd/dyn_grmhd_fofc_impl.hpp` | `ntest_` = `dual_idx + 1`, so the FOFC trial state includes the auxiliary the trial inversion reads |
| `src/hydro/hydro_tasks.cpp` / `src/mhd/mhd_tasks.cpp` | Task wiring (`id.duale`), restriction sync, `ConToPrim`/`ConToPrimGhostBands` resync calls, LAT exchange gating |
| `src/dyn_grmhd/dyn_grmhd.cpp` | The dynamical task graph's three resync call sites and the interior-first C2P refusal |
| `src/hydro/hydro_update.cpp` | LAT flux-correction companion (`lat_dual_vf_reflux`), p dV flavour only |
| `src/eos/ideal_hyd.cpp`, `src/eos/ideal_mhd.cpp` | Ideal-gas Newtonian C2P eta1 switch, inline |
| `src/eos/general_c2p_hyd.hpp`, `src/eos/general_c2p_mhd.hpp` | `SingleC2P_General{Hyd,MHD}Dual` — shared eta1 switch for the tabulated EOS path |
| `src/eos/saha_table_hyd.cpp`, `src/eos/saha_table_mhd.cpp` | Wires the tabulated Saha/LTE EOS to the generic dual C2P helpers |
| `src/bvals/prolong_prims.cpp` | Coarse-cell C2P before prolongation and P2C after; gated on `dual_energy_pdv` for MHD |
| `src/mesh/mesh_refinement.cpp` | Sets `prolong_prims` from `dual_energy_pdv`, not `use_dual_energy`; `RepairRefinedDualEnergyState` after regrid |
| `src/mesh/refinement_criteria.cpp` | `hydro_w_eaux`/`mhd_w_eaux` AMR `rvariable` keys reading the aux column |
| `src/srcterms/srcterms.cpp` | `DualEnergyTarget`, `DualEnergyCoolingDebit`, and the three cooling call sites |
| `src/remap/remap.cpp`, `remap_cc.cpp`, `remap_impl.hpp` | Aux-column carry/floor/reseed during mesh remap |
| `src/outputs/basetype_output.cpp` | `MhdScalarName` names the column `eint_aux`/`reint_aux`; MHD bundles emit it |
| `src/outputs/restart.cpp`, `src/pgen/pgen.cpp` | Restart width detection/back-compat and `dual_energy_needs_init` bookkeeping |
| `src/driver/driver.cpp` | Seeding calls for both flavours; LAT reconciliation |
| `src/pgen/tests/dual_energy_cancellation.cpp` | Dedicated high-Mach cancellation stress test (non-relativistic) |

## Configuration

Six input keys, three per fluid block. `dual_energy_idx` and `naux` are **not** input keys:
`dual_energy_idx = nhydro(+nmhd) + nscalars`, `naux = 1`, `nvars = nhydro(+nmhd) + nscalars
+ naux` (`src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`). Neither is the flavour:
`dual_energy_pdv` is derived from the coordinate system (`src/mhd/mhd.cpp`).

| Block/Key | Type | Default | Meaning | Read at |
|---|---|---|---|---|
| `hydro/dual_energy` | bool | `false` | Enables the formalism for hydro in whichever flavour the coordinates select; adds 1 auxiliary conserved variable. Non-relativistic: requires an EOS with `use_e=true`. Fixed-metric GR: the adiabat, ideal gas by construction. SR is refused; so is `<mesh_refinement>/prolong_primitives = true` under GR | `src/hydro/hydro.cpp` |
| `hydro/dual_energy_eta1` | Real | `1.0e-3` | Per-cell C2P switch: use the total-energy-derived `w.e` when `eint_cons > eta1*E_tot`, else the tracked aux value. `<= 0` always prefers `eint_cons` when positive | `src/hydro/hydro.cpp` |
| `hydro/dual_energy_eta2` | Real | `1.0e-1` | Coarser resync ratio: overwrite `eint_aux` from `eint_cons` when eligible. `<= 0` makes the resync unconditional | `src/hydro/hydro.cpp` |
| `mhd/dual_energy` | bool | `false` | Enables the formalism for MHD, in whichever flavour the coordinates select. Non-relativistic: requires `use_e=true`. Dynamical GR: requires `<mhd>/dyn_eos = ideal`. Fixed-metric GR is ideal-gas by construction. SR is refused | `src/mhd/mhd.cpp` |
| `mhd/dual_energy_eta1` | Real | `1.0e-3` | Non-relativistic: use the energy channel when `eint_cons > eta1*E_tot`. GR: when `u_int > eta1*tau`. `<= 0` always prefers the energy channel | `src/mhd/mhd.cpp` |
| `mhd/dual_energy_eta2` | Real | `1.0e-1` | Resync ratio. Non-relativistic: against `max_27(E_tot)`. GR: `eps > eta2*max_27(tau/D)`. `<= 0` makes the resync unconditional | `src/mhd/mhd.cpp` |

The GR flavour also reads `<mhd>/sfloor` — the entropy floor the Kastaun inversions already
carry (`src/eos/eos.cpp`) — as the floor on $\kappa$, so the two agree by construction
(`AdiabatFromPrimitive` in `src/mhd/mhd_dual_energy.cpp`).

## How it runs

**Construction.** `src/hydro/hydro.cpp` / `src/mhd/mhd.cpp`: reads the
flags, picks the flavour, runs the refusals listed below, sets `naux = 1` and appends to
`nvars`. Buffer allocation follows the flavour, not the feature:

| Buffer | Allocated when | Where |
|---|---|---|
| `dual_excise_mask` | `use_dual_energy` | `src/mhd/mhd.cpp`, `src/hydro/hydro.cpp` |
| `dual_etot_max` (the cached eta2 stencil; MHD only — hydro builds it inline in the kernel) | `use_dual_energy` | `src/mhd/mhd.cpp` |
| `dual_vf` (per-direction face velocities) | `dual_energy_pdv` | `src/mhd/mhd.cpp`, `src/hydro/hydro.cpp` |
| `lat_dual_vf_reflux` (LAT divv companion) | `dual_energy_pdv` | `src/hydro/hydro.cpp` |
| flux-correction payload width `nvars + 1` | `dual_energy_pdv` | `src/mhd/mhd.cpp`, `src/mhd/mhd_tasks.cpp` |

That distinction is load-bearing. `dual_vf` and `lat_dual_vf_reflux` stay at their
constructor size `(1,1,1,1,1)` on the GR path, so the flux-correction calls in
`mhd_tasks.cpp` must pass `&dual_vf` (or `&lat_dual_vf_reflux`) only for the p dV flavour.
Passing them under `use_dual_energy` would give the flux-correction exchange a non-null
pointer to nothing: the receiver derives the extra payload width from the view, gets 1, and
runs the kernel, an out-of-bounds device **write** on every coarse/fine face of every
multilevel GR run with the formalism on.

**Per-stage task order (hydro,** `src/hydro/hydro_tasks.cpp`**):**
`Fluxes → SendFlux → RecvFlux → RKUpdate → DualEnergyStep → HydroSrcTerms →
HydroStateFixup → SendU_OA/RecvU_OA → RestrictU (+SynchronizeRestrictedDualEnergyField on
multilevel) → SendU → RecvU → Prolongate → ApplyPhysicalBCs → HydroStateFixup(post) →
ConToPrim (runs SynchronizeDualEnergyFieldFromTotal, then ConsToPrim over the full ghosted
range) → NewTimeStep.` MHD differs only in where `DualEnergyStep` sits — after the CT
face-field update rather than immediately after `RKUpdate`
(`src/mhd/mhd_tasks.cpp`).

**Where the eta2 pass is called.** The non-relativistic MHD pass runs at the head of
`ConToPrimWithPolicy` (`src/mhd/mhd_tasks.cpp`, gated on `dual_energy_pdv`). The GR pass
runs from `MHD::DualEnergyStep`: in the fixed-metric task list that is the `duale` task
between CT and the source terms; in the dynamical-GR graph it is `MHD_DualE`
(`src/dyn_grmhd/dyn_grmhd.cpp`, queued beside `MHD_AddSrc`, with `MHD_RestU` depending on
it so the coarse copies and the send both carry the resynchronised column). Both sit ahead
of the conserved send. Running the GR pass at the head of the inversion would force the
full-extent recovery on every pass and disable the interior-first split and the ghost-band
recovery, so those exclusions are keyed on the non-relativistic flavour only
(`ConToPrimGhostBands`, `ConToPrimWithPolicy`, `InteriorFirstC2PUsable`). With the budget cap
in the resync pass, most of the remaining cost is per-cell work inside the inversion (two
or three divisions, the eta1 test, two column writes).

**FOFC.** The first-order flux correction re-solves every face touched by a floor/ceiling
trip, including the aux column: `SetDualEnergyFOFCFlux` for the p dV flavour
(`src/hydro/hydro_fofc.cpp`, `src/mhd/mhd_fofc.cpp`), the ordinary first-order
scalar flux for the adiabat. The dynamical module's trial conserved state is built over
`dual_idx + 1` columns rather than `nmhd + nscal`
(`src/dyn_grmhd/dyn_grmhd_fofc_impl.hpp`); leaving the auxiliary out would make the trial eta1
branch read a column nothing ever wrote, so it could never fire and the pass would flag for
first-order fluxes exactly the cells the auxiliary exists to rescue.

**Restart.** `dual_energy_needs_init` starts `true` at construction. `pgen.cpp` detects the
restart file's conserved-variable width against several candidate layouts and derives
`hydro_restart_missing_dual` / `mhd_restart_missing_dual`
(`src/pgen/pgen.cpp`). If the file already matches a "has-dual" width the
flag is cleared and the checkpoint's aux column is trusted verbatim
(`src/pgen/pgen.cpp`). If dual energy is being turned **on** against a checkpoint
that lacks it, the flag stays `true` and `Driver::Initialize`
seeds once — through `InitializeDualEnergyFieldFromTotal` for the p dV flavour, through
`InitializeDualEnergyFieldFromAdiabat` for the GR one, and only the branch that actually
seeded clears the flag. Turning it **off** against a checkpoint that carries the extra
column prints a one-line notice and drops the field (`src/pgen/pgen.cpp`).

**AMR regrid.** `RepairRefinedDualEnergyState` re-floors the aux field on newly-refined
cells after step 9 of the regrid (`src/mesh/mesh_refinement.cpp`), in
the form its own flavour branch requires: a thermal floor on an energy density, or a
division by the refined density followed by `sfloor` and a re-multiplication for the
adiabat. `SynchronizeRestrictedDualEnergyField` mirrors the coarse-restricted conserved aux
column into the primitive-side coarse array with the same split — the non-relativistic
conserved and primitive forms are the same number, the GR pair differ by a factor of the
restricted density.

**Remap.** Loading a mesh from another run's dump floors the outer source ghost ring's aux
column (`FloorOuterSourceGhostZones`, `src/remap/remap_impl.hpp`) and then sets
`dual_energy_needs_init = true` (`src/remap/remap.cpp`) so the next
`Driver::Initialize` reseeds from the freshly-remapped state rather than trusting an
interpolated aux value. See `docs/wiki/Remapping.md`.

## Interactions

- **Cooling source terms.** Three of them charge the auxiliary at the same point they charge
  the gas: ISM cooling (`src/srcterms/srcterms.cpp`), relativistic cooling and disk cooling, in both its ADM and its fixed-GR/SR/Newtonian kernel. `SourceTerms::DualEnergyTarget` resolves which fluid owns the
  conserved array being written and reports that fluid's flavour, and
  `DualEnergyCoolingDebit` applies the form that flavour requires: the
  non-relativistic auxiliary **is** an internal-energy density and takes the same increment,
  `u0(dual) -= de`; the GR auxiliary is an adiabat, cooling changes $p$ at fixed $\rho$ and
  fixed $D$, so $\kappa$ and the internal energy scale by the same factor and the debit is
  multiplicative, `u0(dual) *= max(1 - de/eint, 1e-3)`. Using either form for the other is a
  units error no test in this repository would catch. `de` must be the **comoving** internal
  energy removed: relativistic cooling therefore divides its conserved-energy increment back
  by `ut` before charging it. The floor on the factor leaves a small positive
  adiabat rather than a negative one when a step would remove more than the cell owns; the
  same step leaves the gas at its own energy floor and the eta2 pass reconciles the pair.
  Gravitational work is deliberately **not** charged — it is not a thermal source, and the
  eta2 pass folds back whatever of it is reliable.
- **LAT.** (`src/driver/driver.cpp`): after a LAT tick's fluid
  reflux, `u0(IEN)` on receiver
  blocks is rewritten by writers the aux field never sees, so
  `SynchronizeDualEnergyFieldFromTotal` is invoked explicitly over exactly the receiver-block
  set those passes installed. Dual energy also disqualifies the "skip final LAT exchange"
  fast path and the deferred-final-exchange path
  (`src/driver/driver.cpp`), because the resync needs current neighbour
  `IEN`. Coarse/fine
  LAT flux corrections carry a companion divv correction (`lat_dual_vf_reflux`) applied by a
  small explicit compression update right after the main energy reflux
  (`src/hydro/hydro_update.cpp`) — p dV flavour only,
  since that correction exponentiates the auxiliary as an energy density, which `D*kappa` is
  not.
- **AMR/SMR.** `prolong_prims` is forced by the **non-relativistic** auxiliary and by the
  tabulated EOS, not by the feature: `mesh_refinement.cpp` keys on
  `dual_energy_pdv`. The GR adiabat declines it deliberately — prolonging `D` and `D*kappa`
  separately and recovering $\kappa$ as their ratio is exactly how passive scalars already
  cross a coarse/fine boundary (`src/mhd/mhd.cpp`).
  `hydro_w_eaux`/`mhd_w_eaux` are valid `<amr_criterion>` `rvariable` keys reading the aux
  column directly (`src/mesh/refinement_criteria.cpp`); on the GR path that
  column holds $\kappa$, not an energy.
- **Restart / units.** No dimensional coupling — a pure numerical technique riding on
  whichever EOS/units are configured; the restart interaction is covered above.
- **remap.** The aux column rides along generically in Keep mode; FloorFade mode reads and
  writes it explicitly at the deep-state, band and ghost-floor stages
  (`src/remap/remap_cc.cpp`, `docs/wiki/Remapping.md`).

## Limitations and known issues

**Benefit not established for production GRMHD.** The GR flavour's *mechanism* is measured
(see "How much it actually buys" above) and its *no-op* property is pinned by a test, but its
benefit to a production accretion run has not been shown. In tests on a magnetized funnel the
ratio-test switch at $\eta_1 = 10^{-4}$ can override energy-channel states that are still
accurate (the error is $\epsilon_{\rm mach}\,b^2/u \approx 10^{-5}$ even at
$b^2/u = 3\times10^{10}$) and heat the funnel, while a failure-only policy
($\eta_1 = 10^{-12}$) is neutral. Resynchronising with a small or a large $\eta_2$ did not
change this. Leave the GR flavour off unless a problem is shown to need it, and note that
isolated Lorentz-factor spikes in cold cells at the temperature floor are a floor-policy
problem rather than an inversion problem.

**Refusals the code makes.** All but the last are construction-time `exit(EXIT_FAILURE)`.

| Refused | Why | Where |
|---|---|---|
| `<hydro>/dual_energy` under SR | Neither the adiabat form nor the p dV form is implemented for special relativity | `src/hydro/hydro.cpp` |
| `<hydro>/dual_energy` under GR together with `<mesh_refinement>/prolong_primitives = true` | The adiabat crosses coarse/fine boundaries as `D` and `D*kappa`; the primitive round trip cannot carry it | `src/hydro/hydro.cpp` |
| `<mhd>/dual_energy` under SR | The adiabat form was not ported to SR and the p dV form is not valid there | `src/mhd/mhd.cpp` |
| `<mhd>/dyn_eos` other than `ideal` on the dynamical path | The auxiliary channel inverts $\kappa$ back to a pressure, which only a gamma-law gas does in closed form. The fixed-metric path needs no test: every non-ideal `<mhd>/eos` is already refused under GR | `src/mhd/mhd.cpp` |
| An EOS without `use_e` (non-relativistic flavour only) | The auxiliary is an internal-energy density; `isothermal_hyd.cpp` never sets `use_e` | `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp` |
| Riemann solvers outside `{llf,hlle,hllc,roe}` (hydro) / `{llf,hlle,hlld}` (MHD), **non-relativistic flavour only** | Those are the solvers that produce the interface velocity the p dV step needs. The GR solvers are not gated: the adiabat needs no face velocity | `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp` |
| `<time>/evolution != dynamic` | Both flavours are transport-coupled | `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp` |
| `<amr_criterion>/rvariable = hydro_w_eaux` or `mhd_w_eaux` with the formalism off | The criterion reads a column that does not exist; this one is a `Kokkos::abort` at criterion setup, not a construction exit | `src/mesh/refinement_criteria.cpp` |

**Other standing limitations.**

- **Extra cost.** `nvars += 1` throughout (bandwidth and memory for every conserved and
  primitive array); a `dual_excise_mask` byte array whenever the formalism is on, plus a
  `dual_etot_max` scalar array on the MHD side; and, for the p dV flavour only, the `dual_vf`
  face-velocity buffer, the `lat_dual_vf_reflux` companion and the one-extra-payload flux
  exchange. With `eta2 > 0` a 27-cell neighbourhood scan runs per cell once per stage. In
  GR, a cell that fails `eta1` runs the inversion twice.

## Tests

- `tst/test_suite/gr/test_grhd_dual_energy_cpu.py` — the same two-sided gate for the GR
  hydro flavour on the MB2 tube in Minkowski GR (`tst/inputs/dual_grhd_tube.athinput`,
  `nx1 = 400`): inert at `eta1 = 1e-4`, live and finite at `eta1 = 1`.
- `tst/test_suite/gr/test_gr_dual_energy_cpu.py` — the GR gate, and it pins **both**
  directions an optional channel can fail in. On the MUB1 tube
  (`tst/inputs/mub1_gr_dual.athinput`, `nx1 = 400`) it requires that at the production
  threshold `eta1 = 1e-4` the formalism changes nothing (max relative density difference
  `< 1e-13`; measured 0.0e+00), and that with `eta1 = 1.0`, which selects the
  adiabat-derived pressure in every cell, the profile visibly moves (`> 1e-6`; measured
  1.0). A run that passes the first check and fails the second is a feature that has stopped
  working. The deck exists because `ParameterInput` refuses a command-line override for a key
  the file does not declare.
- `src/pgen/tests/dual_energy_cancellation.cpp` — a 1-D high-Mach entropy-wave test built to
  defeat `E - KE` cancellation in the ordinary non-dual path: a sinusoidal density
  perturbation advected at a large bulk velocity with uniform pressure. It seeds `eint_aux`
  from the analytic internal energy and clears `dual_energy_needs_init` itself. Keys
  `problem/rho0` (`1.0`), `problem/pressure` (`1.0`), `problem/bulk_velocity` (`1.0e8`),
  `problem/density_amplitude` (`0.1`) at `dual_energy_cancellation.cpp`. No committed
  deck; run with `problem/pgen_name=dual_energy_cancellation` and `<hydro>/dual_energy` true
  vs false. Ideal gamma-law hydro only.
- Production decks exercising the non-relativistic flavour for real physics (all tabulated
  Saha/LTE EOS, all `<hydro>`): `inputs/TDE_examples/*.athinput`.

## Example deck fragment

Non-relativistic, keys only, from `inputs/TDE_examples/tde_01_disruption.athinput`:

```
<hydro>
dual_energy       = true
dual_energy_eta1  = 1.0e-3
dual_energy_eta2  = 1.0e-4
```

GR, from `tst/inputs/mub1_gr_dual.athinput` (shipped off; the test switches it on
from the command line):

```
<mhd>
dual_energy      = false
dual_energy_eta1 = 1.0e-4
dual_energy_eta2 = 1.0e-4
```

## References

- Kastaun, Kalinani & Ciolfi (2021) for the single-unknown GR inversion both auxiliary root
  finds are derived from.
- `docs/wiki/Remapping.md` for the remap module's handling of the aux column.
