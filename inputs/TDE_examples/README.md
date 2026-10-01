# TDE example inputs

These five input files reproduce the fiducial (FID) tidal disruption calculation of
Jiang et al., *An End-to-End Numerical Framework for Tidal Disruption Events with
AthenaK* (ApJS).  A 1 Msun, 1 Rsun star built from the tabulated H/He equation of
state is disrupted by a 1000 Msun black hole on a parabolic orbit with beta = 1, and
the debris is followed to five times the fallback time of the most bound debris,
P_mb = 88.04 t0 = 1.295 d.  The physical and numerical parameters are those of
Table 1 of the paper.

| Step | File | Time (t0) | Box (Rsun) | AMR levels | Frame | What it shows |
|---|---|---|---|---|---|---|
| 1 | `tde_01_disruption.athinput` | 0 to 21 | 64^3 | 9 | translating | EOS-balanced star, first pericenter passage |
| 2 | `tde_02_remap_box256.athinput` | 21 to 24 | 256^3 | 10 | translating | remap to a larger box |
| 3 | `tde_03_remap_box512.athinput` | 24 to 40 | 512 x 512 x 256 | 9 | translating | second remap, BH-centered refinement shells |
| 4 | `tde_04_bh_frame.athinput` | 40 to 70 | 512 x 512 x 256 | 8 | BH rest frame | conversion to the BH rest frame |
| 5 | `tde_05_fallback_lat.athinput` | 70 to 440 | 1024 x 1280 x 256 | 8 | BH rest frame | fallback with localized adaptive time stepping |

Pericenter is at t = 21 t0 (0.24 P_mb).  Every step starts from the last restart file
of the previous step through the `<remap>` block, which resamples the conserved
variables onto the new mesh.  Code units are M0 = 1 Msun, L0 = 2 Rsun (so the stellar
radius is 0.5), t0 = 1271 s, and 4 pi G = 1.

The black hole is live from step 2 on (`bh_live = true`, with
`external_bh_gravity_source = true`).  In step 1 it starts outside the 64 Rsun box, where
the gas potential cannot be sampled, so it follows the two-body orbit until the first
remap (`bh_live = false`).  A live BH is a softened point mass with an excised inner
boundary whose position and velocity are integrated with the gas pull on it.  The
"BH rest frame" of steps 4 and 5 is the inertial frame in which the BH was at rest while it
followed the two-body orbit in step 1; the live BH keeps its small reflex velocity and moves
in it.  Step 4 is the conversion itself, a Galilean
boost of the state by the frame velocity done by the remap.  Every remap carries the BH
position, velocity, and acceleration from the source restart file.  The reciprocal force
(the gas pull on the BH applied with the same stencil as the BH force on the gas) needs
LAT, so it is off in steps 1 to 4 and on in step 5.

## Build

```bash
cmake -S . -B build_tde -D PROBLEM=tde_external -D Athena_ENABLE_MPI=ON \
      -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_VOLTA70=ON \
      -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
cmake --build build_tde -j 16
```

Set `Kokkos_ARCH_*` for your GPU.  The paper runs used NVIDIA V100 cards.

## Equation of state table

The runs use `eos_tables/chabrier2021_t13_helm_union_prad_640.table`, the
Chabrier et al. (2021) H/He table joined with the Tomida et al. (2013) and HELM
tables, including radiation pressure (Sec. 4 of the paper).  Build it with
`./get_eos_table.sh` in the top-level directory of the repository.  The input files look for it one directory above
the directory each step runs in.

## Running the chain

Each step runs in its own directory, and the directories sit next to each other so
that the relative restart paths in the `<remap>` blocks resolve.

```bash
REPO=/path/to/AthenaK-HXJ
ATHENA=$REPO/build_tde/src/athena
DECKS=$REPO/inputs/TDE_examples

mkdir -p tde_run && cd tde_run
ln -s $REPO/eos_tables/chabrier2021_t13_helm_union_prad_640.table .

for step in 01_disruption 02_remap_box256 03_remap_box512 04_bh_frame 05_fallback_lat; do
  mkdir -p $step
  (cd $step && mpirun -np 10 $ATHENA -i $DECKS/tde_$step.athinput)
done

```

A remap never runs on a restart (`athena -r`).  Step 5 remaps and runs with LAT on, together
with the reciprocal force of the gas on the BH, to 5 P_mb (t = 440.2).  The settle steps of
the remap run without LAT, and LAT starts after the last remap pass.  A step that stops
early is continued the usual way, with `athena -r rst/<last file>.rst` inside its directory.

The restart files used by the next step are

| Written by | File | Time (t0) |
|---|---|---|
| step 1 | `01_disruption/rst/TDEExternalLTEPrad.00007.rst` | 21 |
| step 2 | `02_remap_box256/rst/TDEExternalLTEPrad.00008.rst` | 24 |
| step 3 | `03_remap_box512/rst/TDEExternalLTEPrad.00010.rst` | 40 |
| step 4 | `04_bh_frame/rst/TDEExternalLTEPrad.00016.rst` | 70 |

Output numbers continue from the source restart file.  A different source can be
given on the command line, for example `remap/source=../01_disruption/rst/<file>.rst`.

## Resources

Steps 1 to 4 run on 10 GPUs with 16 GB each, and step 2 needs all 10 cards because the
AMR hierarchy grows quickly after pericenter.  The memory of step 5 grows as the debris
spreads.  If a step runs out of memory, lower `mesh_refinement/max_nmb_per_rank` and use
more ranks.

## Variations used in the paper

* **Self-gravity cadence (Sec. 8.1).**  Steps 1 to 4 with `gravity/solve_dt=0.0025`
  added on the command line, compared with the value `solve_dt = 0.03` set in the input files.
* **LAT benchmark (Sec. 7.4).**  Run step 5 with `time/lat=false problem/bh_reciprocal_force=false
  time/tlim=71.0` to get a checkpoint at t = 71, then restart it once with
  `time/lat=true problem/bh_reciprocal_force=true` and once without both (the reciprocal force
  needs LAT), keeping every other setting identical.
* **HR run.**  The high-resolution run of the paper continues from the step-4 restart
  with 10 AMR levels and a finer vertical grid.  It is not included here.

## Synthetic observables

`athenak_rt` computes line-of-sight luminosities, spectra, and images from the
`bin/*.bin` snapshots with the multifrequency H/He continuum transfer of Sec. 9 and
the grey reference modes.  See `athenak_rt/README.md`.
