#ifndef MHD_MHD_FLUXES_IMPL_HPP_
#define MHD_MHD_FLUXES_IMPL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes_impl.hpp
//! \brief Shared implementation of MHD::CalculateFluxes(), included by the nine
//! mhd_fluxes_<solver>.cpp translation units that each carry exactly one explicit
//! instantiation (see mhd_fluxes_advect.cpp, mhd_fluxes_llf.cpp, mhd_fluxes_hlle.cpp,
//! mhd_fluxes_hlld.cpp, mhd_fluxes_llf_sr.cpp, mhd_fluxes_hlle_sr.cpp,
//! mhd_fluxes_hlld_sr.cpp,
//! mhd_fluxes_llf_gr.cpp, mhd_fluxes_hlle_gr.cpp, mhd_fluxes_hlld_gr.cpp).  Splitting the
//! instantiations across separate translation units lets ptxas -- which register-
//! allocates one file's kernels in a single serial pass -- run once per solver in
//! parallel, instead of once for the whole file.  The three GR solvers go one step
//! further: the x2 and x3 face kernels (LaunchMHDFaceFluxKernels<solver, IVY|IVZ>) are
//! instantiated in mhd_fluxes_<solver>_gr_x2.cpp and _x3.cpp and declared extern in
//! mhd_fluxes_<solver>_gr.cpp.  Every symbol below is either a template or marked
//! inline, so ODR is satisfied across the inclusions.
//!
//! MHD::ApplyExcisionSinkBoundary() is NOT part of this header: it does not depend on
//! the rsolver template parameter, so it lives in its own mhd_fluxes_excision.cpp and is
//! compiled exactly once rather than nine times.  See that file for the sink-flux
//! helpers it uses.
//!
//! Calculate 3D fluxes of the conserved variables, and area-averaged electric
//! fields E = - (v X B) on cell faces for mhd.
//!
//! Fluxes are computed with two 1D-RangePolicy kernels per direction: (1) a per-cell
//! reconstruction kernel that materializes the L/R primitive states (w0) in the global
//! wl3d/wr3d buffers and the L/R cell-centered magnetic field (bcc0) in the
//! bl3d/br3d buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes both the interface flux and the two area-averaged EMF components.
//! All reconstruction methods (DC/PLM/PPM4/PPMX/WENOZ) and Riemann solvers
//! (Advect/LLF/HLLE/HLLD and the SR/GR variants) are supported; the reconstruction method
//! is chosen at runtime, the solver at compile time via the rsolver template parameter.
//!
//! Fluxes are stored in face-centered vector 'uflx', while electric fields are stored in
//! individual arrays: e2x1,e3x1 on x1-faces; e1x2,e3x2 on x2-faces;
//! e1x3,e2x3 on x3-faces.
//! Because constrained transport needs EMFs at every transverse cell edge, the flux/EMF
//! kernels run over a transverse range extended by one cell beyond the active domain.

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/thermal_floors.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/hlld_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"
#include "mhd/rsolvers/hlld_grmhd.hpp"
#include "reconstruct/specific_energy_recon.hpp"

namespace mhd {

KOKKOS_INLINE_FUNCTION
void SanitizeMHDFluidFaceStateAt(const EOS_Data &eos, const int m,
                                 const int k, const int j, const int i,
                                 const DvceArray5D<Real> &wl,
                                 const DvceArray5D<Real> &wr) {
  Real &dl = wl(m, IDN, k, j, i);
  Real &dr = wr(m, IDN, k, j, i);
  dl = fmax(dl, eos.dfloor);
  dr = fmax(dr, eos.dfloor);
  if (eos.use_e) {
    FloorReconstructedInternalEnergyPair(
        eos, wl(m, IEN, k, j, i), wr(m, IEN, k, j, i), dl, dr);
  }
}

KOKKOS_INLINE_FUNCTION
void SetDualEnergyFluxAt(const EOS_Data &eos, const int dual_idx,
                         const bool set_interface_velocity,
                         const int m, const int mb, const int k, const int j, const int i,
                         const DvceArray5D<Real> &wl,
                         const DvceArray5D<Real> &wr,
                         const BandView5D<Real> &flx,
                         const BandView5D<Real> &vf) {
  const Real mass_flux = flx(m, IDN, k, j, i);
  if (mass_flux > 0.0) {
    const Real dens = fmax(wl(mb, IDN, k, j, i), eos.dfloor);
    const Real eint = ApplyReconstructedHydroThermalFloors(
        eos, dens, wl(mb, dual_idx, k, j, i));
    flx(m, dual_idx, k, j, i) =
        mass_flux*(eint/dens);
    if (set_interface_velocity) vf(m, 0, k, j, i) = mass_flux/dens;
  } else if (mass_flux < 0.0) {
    const Real dens = fmax(wr(mb, IDN, k, j, i), eos.dfloor);
    const Real eint = ApplyReconstructedHydroThermalFloors(
        eos, dens, wr(mb, dual_idx, k, j, i));
    flx(m, dual_idx, k, j, i) =
        mass_flux*(eint/dens);
    if (set_interface_velocity) vf(m, 0, k, j, i) = mass_flux/dens;
  } else {
    flx(m, dual_idx, k, j, i) = 0.0;
    if (set_interface_velocity) vf(m, 0, k, j, i) = 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
void SetScalarFluxesAt(const bool dual_enabled, const int dual_idx,
                       const int first_scalar, const int nvars,
                       const int m, const int mb, const int k, const int j, const int i,
                       const DvceArray5D<Real> &wl,
                       const DvceArray5D<Real> &wr,
                       const BandView5D<Real> &flx) {
  const Real mass_flux = flx(m, IDN, k, j, i);
  for (int n = first_scalar; n < nvars; ++n) {
    if (dual_enabled && n == dual_idx) continue;
    flx(m, n, k, j, i) = mass_flux *
        ((mass_flux >= 0.0) ? wl(mb, n, k, j, i) : wr(mb, n, k, j, i));
  }
}

//----------------------------------------------------------------------------------------
//! \fn SolveFaceMHD<rsolver,ivx>()
//! \brief Dispatch the (compile-time) MHD Riemann solver for a single face, writing the
//! conserved flux and the two transverse EMF components.  Capturing the solver inputs
//! into locals before the constexpr-if is required for CUDA 11.6+.
template <MHD_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
Real SolveFaceMHD(const EOS_Data &eos, const RegionIndcs &indcs,
                  const DualArray1D<RegionSize> &size, const CoordData &coord,
                  const int m, const int mb, const int k, const int j, const int i,
                  const int is, const int js, const int ks,
                  const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
                  const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
                  const DvceArray4D<Real> &bx,
                  const BandView5D<Real> &flx,
                  const BandView4D<Real> &ey, const BandView4D<Real> &ez) {
  if constexpr (rsolver_method_ == MHD_RSolver::advect) {
    Advect<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
    LLF<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
    HLLE<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
    return HLLD<ivx>(eos, m, mb, k, j, i, is, js, ks,
                     wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
    LLF_SR<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
    HLLE_SR<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlld_sr) {
    HLLD_SR<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
    LLF_GR<ivx>(eos, indcs, size, coord, m, mb, k, j, i, is, js, ks,
                wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, mb, k, j, i, is, js, ks,
                 wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlld_gr) {
    HLLD_GR<ivx>(eos, indcs, size, coord, m, mb, k, j, i, is, js, ks,
                 wl, wr, bl, br, bx, flx, ey, ez);
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \var kIsGRMHDRiemannSolver
//! \brief True for the fixed-spacetime GRMHD solvers, which the split below specialises
//! on.  A variable template rather than a constexpr function so that device
//! code can test it without nvcc's __host__-constexpr-call diagnostic.

template <MHD_RSolver r>
inline constexpr bool kIsGRMHDRiemannSolver =
    (r == MHD_RSolver::llf_gr || r == MHD_RSolver::hlle_gr ||
     r == MHD_RSolver::hlld_gr);

//----------------------------------------------------------------------------------------
//! \var kSplitFacePreparation
//! \brief Whether the reconstructed-face closures run as their own pass rather than
//! inside the Riemann kernel.
//!
//! The separation is not free: it costs one extra read-modify-write of wl/wr over the
//! whole face range.  It pays only where the fused kernel is big enough for the code it
//! removes to matter, and that is exactly the fixed-spacetime GRMHD solvers -- they are
//! the ones carrying the optional-physics bulk and a multi-kilobyte spill frame (hlle_gr
//! 196k SASS instructions, 3488 B of stack; hlld 9.4k and 208 B).  Measured on a V100
//! with base and candidate run simultaneously on swapped GPUs: splitting gains 9.1% and
//! 12.2% end to end on the 64^3 and 128^3 GR torus, and costs about 3% on a Newtonian
//! Orszag-Tang, whose kernel had nothing worth removing.  So the Newtonian and SR solvers
//! keep the fused form, for which the emitted kernel is identical to the original
//! (hlld: 9368 instructions, REG 128, STACK 208 both before and after).
//!
//! Unlike the launch bounds in this file, this predicate is deliberately NOT resolved at
//! runtime from device attributes, for three reasons:
//!
//!   1. It is not a hardware tuning parameter.  What it keys on is a property of the
//!      SOURCE -- whether a given solver's fused kernel has multiple kilobytes of
//!      hoistable state -- not of the GPU.  Two orders of magnitude separate hlle_gr from
//!      hlld here; no plausible GPU reorders that.  A launch bound, by contrast, is a
//!      statement about a specific register file and cache, which is why those are
//!      chosen automatically and this one is not.
//!   2. Deciding it at runtime would require compiling both forms.  It gates an
//!      `if constexpr` inside SolveFaceFluxAt, which is inlined into the Riemann kernel,
//!      so each candidate is a separate instantiation of the largest kernel in the code
//!      -- on top of the launch-bounds candidates, i.e. four bodies per solver per
//!      direction.  That is not a cost worth paying for a decision taken once per run.
//!   3. Flipping it changes which kernels launch and in what order, so it carries a
//!      correctness risk that a launch bound (register allocation only) does not.
//!
//! What would falsify the premise on new hardware: the GR Riemann kernels ceasing to
//! spill.  That is directly observable with
//!   cuobjdump --dump-resource-usage \
//!       <build>/src/CMakeFiles/athena.dir/mhd/mhd_fluxes.cpp.o
//! which prints REG and STACK for every instantiation.  On sm_70 today the GR solvers
//! carry 1760-2752 B of stack and the Newtonian ones 208 B.  If a future GPU compiles
//! hlle_gr with a small frame, re-measure before assuming the split still pays.

template <MHD_RSolver r>
inline constexpr bool kSplitFacePreparation = kIsGRMHDRiemannSolver<r>;

//----------------------------------------------------------------------------------------
//! \struct MHDFaceStateContext
//! \brief Direction-independent bindings shared by the face-preparation and Riemann-solve
//! kernels.  Assembled once per CalculateFluxes() call on the host and captured by value
//! into the device functors, so the device helpers below take one reference each instead
//! of twenty-odd separate arguments.  Members are bound, never re-copied, inside the
//! kernels: EOS_Data alone is 12 Kokkos Views plus ~40 scalars, and a per-thread copy
//! lands in the local-memory frame of the GR flux kernels.

struct MHDFaceStateContext {
  EOS_Data eos;
  RegionIndcs indcs;
  DualArray1D<RegionSize> size;
  CoordData coord;
  DvceArray5D<Real> wl, wr, bl, br;
  int is{0}, js{0}, ks{0};
  int nmhd{0}, nvars{0}, dual_idx{0};
  bool dual_enabled{false};
  // True only for the non-relativistic internal-energy auxiliary.  The GR adiabat is a
  // per-unit-mass quantity, so it is advected by the ordinary scalar loop and needs no
  // interface velocity.
  bool dual_pdv{false};
  // Reconstructed-state repairs that are a pure function of the face states, and so run
  // as a separate pre-pass rather than inside the Riemann kernel.
  bool sanitize_faces{false};
};

//----------------------------------------------------------------------------------------
//! \struct MHDFaceFluxTargets
//! \brief The per-direction arrays a face kernel writes (and the face-normal field it
//! reads).  One instance is built per direction on the host.

struct MHDFaceFluxTargets {
  DvceArray4D<Real> bx;
  BandView5D<Real> flx;        // uflx face register on the flux band (mhd.hpp FluxBand)
  BandView4D<Real> eyl, ezl;   // face-centred EMF scratch, same band (EmfBand)
  BandView5D<Real> vf;         // dual_vf, same band
};

//----------------------------------------------------------------------------------------
//! \struct MHDFaceKernelRange
//! \brief Index bounds for one direction's face kernels.  The face-normal range is
//! extended by one cell when FOFC is active and the transverse ranges are extended for
//! CT, so the scalar/dual-energy stores are restricted afterwards to the sub-range they
//! owned before those extensions.

struct MHDFaceKernelRange {
  int work_start, work_end;
  int kl, ku, jl, ju, il, iu;              // faces visited by the face kernels
  int s_kl, s_ku, s_jl, s_ju, s_il, s_iu;  // sub-range receiving scalar/dual fluxes

  KOKKOS_INLINE_FUNCTION
  bool InScalarRange(const int k, const int j, const int i) const {
    return (k >= s_kl && k <= s_ku && j >= s_jl && j <= s_ju &&
            i >= s_il && i <= s_iu);
  }
};

//----------------------------------------------------------------------------------------
//! \fn PrepareFaceStatesAt
//! \brief Reconstructed-face repair that depends on nothing but the face states
//! themselves: the density/internal-energy floor.
//!
//! This used to run at the top of the Riemann kernel behind a runtime bool, which forced
//! every GR flux instantiation to carry it (and its spill frame) even when disabled.
//! Because it is a pure elementwise map on (wl,wr) at a fixed (mb,k,j,i), hoisting it
//! into a separate pass over the same face range is bitwise identical as long as the
//! pass covers exactly the faces the solver consumes and runs in the same order.

template <MHD_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void PrepareFaceStatesAt(const MHDFaceStateContext &ctx, const int m, const int mb,
                         const int k, const int j, const int i) {
  auto &eos = ctx.eos;
  auto &wl = ctx.wl;
  auto &wr = ctx.wr;
  if (ctx.sanitize_faces) {
    SanitizeMHDFluidFaceStateAt(eos, mb, k, j, i, wl, wr);
  }
}

//----------------------------------------------------------------------------------------
//! \fn FinishFaceFluxAt
//! \brief Dual-energy and passive-scalar face fluxes, shared by both solve paths.

template <MHD_RSolver rsolver_method_>
KOKKOS_INLINE_FUNCTION
void FinishFaceFluxAt(const MHDFaceStateContext &ctx, const MHDFaceFluxTargets &tgt,
                      const Real interface_velocity, const bool in_scalar_range,
                      const int m, const int mb, const int k, const int j, const int i) {
  if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
    if (ctx.dual_pdv) tgt.vf(m, 0, k, j, i) = interface_velocity;
  }
  if (in_scalar_range) {
    if (ctx.dual_pdv) {
      SetDualEnergyFluxAt(ctx.eos, ctx.dual_idx, rsolver_method_ != MHD_RSolver::hlld,
                          m, mb, k, j, i, ctx.wl, ctx.wr, tgt.flx, tgt.vf);
    }
    SetScalarFluxesAt(ctx.dual_pdv, ctx.dual_idx, ctx.nmhd, ctx.nvars,
                      m, mb, k, j, i, ctx.wl, ctx.wr, tgt.flx);
  }
}

//----------------------------------------------------------------------------------------
//! \fn SolveFaceFluxAt
//! \brief The Riemann solve for one face.  Solvers that take the split (see
//! kSplitFacePreparation) get their face states already prepared by the preceding pass;
//! the rest apply the closures inline, exactly as the original fused kernel did.

template <MHD_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void SolveFaceFluxAt(const MHDFaceStateContext &ctx, const MHDFaceFluxTargets &tgt,
                     const bool in_scalar_range, const int m, const int mb,
                     const int k, const int j, const int i) {
  if constexpr (!kSplitFacePreparation<rsolver_method_>) {
    PrepareFaceStatesAt<rsolver_method_, ivx>(ctx, m, mb, k, j, i);
  }
  const Real interface_velocity = SolveFaceMHD<rsolver_method_, ivx>(
      ctx.eos, ctx.indcs, ctx.size, ctx.coord, m, mb, k, j, i, ctx.is, ctx.js, ctx.ks,
      ctx.wl, ctx.wr, ctx.bl, ctx.br, tgt.bx, tgt.flx, tgt.eyl, tgt.ezl);
  FinishFaceFluxAt<rsolver_method_>(ctx, tgt, interface_velocity, in_scalar_range,
                                    m, mb, k, j, i);
}

//----------------------------------------------------------------------------------------
//! \brief Occupancy cap for the per-face Riemann-solve kernels (mflux_x1/x2/x3_rsolve).
//!
//! Forwarded to Kokkos::LaunchBounds: it tells ptxas the register budget to fit into.
//! Left uncapped the GR instantiations ask for 255 registers, which is 8 warps/SM (12.5%)
//! on Volta; capping at 2 blocks/SM buys 16 warps/SM (25%) for a larger spill frame.
//! Measured on sm_70 for hlle_gr (sum of the three direction kernels, static SASS):
//!
//!   bounds     REG  frame(B)  LDL+STL  warps/SM
//!   uncapped   255      1760     2337         8
//!   <256,2>    128      2128     5045        16   <-- current
//!   <256,3>     80      2416     8666        24
//!   <256,4>     64      2496    10869        32
//!
//! THIS SITE IS DELIBERATELY NOT AUTO-TUNED, unlike the dyn-GR face kernels and
//! MHD::FOFC, which use athenak::launch::par_for_auto.  The facility works here -- it was
//! tried -- but the cost is not worth it at this particular call site:
//!
//!   * Compiling both launch-bound candidates doubles this file's bounded kernels from 36
//!     to 72, and mhd_fluxes.cpp is the critical path of the whole build.  Measured with
//!     -j 32 on this machine: this translation unit went 709 s -> 1233 s (+74%), taking
//!     the entire clean build from 756 s to 1259 s (+66.5%), the object from 24.1 MB to
//!     37.6 MB (+56%), and the binary from 188 MB to 213 MB (+13%).  That is a cost every
//!     developer pays on every build, forever.
//!   * What it would buy is unmeasured.  Of the 36 instantiations the rule keeps 2 for 21
//!     of them (the Newtonian and SR solvers, where the cap is inert or the frames stay
//!     on chip), declines to decide 3, and would move 12 to MinBlocksPerSM=1 -- the GR
//!     solvers, whose frames grow 1760 B -> 2224 B and 2240 B -> 2752 B per thread as the
//!     cap tightens, i.e. 1.1-1.4 MB resident against a 96 kB L1.  The physical argument
//!     for 1 there is the same one that measured +6.5% for the dyn-GR face kernels, but
//!     it has never been timed on a fixed-metric GR problem, and this code should follow
//!     measurements rather than lead them.
//!
//! So the value stays a documented knob with its measured default.  To try the other
//! setting, build with -DATHENAK_MHD_RSOLVE_MIN_BLOCKS_PER_SM=1; to have it chosen
//! automatically from device and kernel attributes, switch the two par_for calls below to
//! athenak::launch::par_for_auto<kRSolveMaxThreads, kRSolveMinBlocksPerSM> and accept the
//! build cost above.  See src/utils/launch_config.hpp for the rule and its limits.
#ifndef ATHENAK_MHD_RSOLVE_MIN_BLOCKS_PER_SM
#define ATHENAK_MHD_RSOLVE_MIN_BLOCKS_PER_SM 2
#endif
constexpr int kRSolveMaxThreads = 256;
constexpr int kRSolveMinBlocksPerSM = ATHENAK_MHD_RSOLVE_MIN_BLOCKS_PER_SM;

//! ...with one solver excepted, and it is the one the note above says was never timed on
//! a fixed-metric GR problem.  HLLD_GR carries a five-wave fan on top of everything the
//! GR entry already needs, and its root solve re-reads the same intermediate state six to
//! ten times per face, so what the cap costs it -- a larger spill frame, re-read on every
//! iteration -- is paid far more often than for any other solver here.  Measured on the
//! SANE torus of tst/analysis/HLLD_TESTS.md, 64 cycles on two V100s.  Register allocation
//! only: the fluxes are bitwise identical either way.
template <MHD_RSolver r>
inline constexpr int kRSolveMinBlocks = kRSolveMinBlocksPerSM;
template <>
inline constexpr int kRSolveMinBlocks<MHD_RSolver::hlld_gr> = 1;

//----------------------------------------------------------------------------------------
//! \fn LaunchFacePreparationKernel
//! \brief Launch the face-preparation pass for one direction.  The pass is just the
//! density/energy floor.

template <MHD_RSolver rsolver_method_, int ivx>
void LaunchFacePreparationKernel(const char *prep_name, const MHDFaceStateContext &ctx,
                                 const MHDFaceKernelRange &rng) {
  par_for(prep_name, DevExeSpace(),
    rng.work_start, rng.work_end, rng.kl, rng.ku, rng.jl, rng.ju, rng.il, rng.iu,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = a;
      PrepareFaceStatesAt<rsolver_method_, ivx>(
          ctx, m, a - rng.work_start, k, j, i);
    });
}

//----------------------------------------------------------------------------------------
//! \fn LaunchMHDFaceFluxKernels
//! \brief Launch one direction's face kernels over one meshblock chunk.
//!
//! The fixed-spacetime GRMHD solvers get a two-kernel split: a preparation pass that
//! applies the face floor over exactly the faces the solver consumes, then a Riemann
//! kernel that is nothing but the solve.  Hoisting the floor out of the solver removes it,
//! and the spill frame it carried, from every GR instantiation.

template <MHD_RSolver rsolver_method_, int ivx>
void LaunchMHDFaceFluxKernels(const char *prep_name, const char *rsolve_name,
                              const MHDFaceStateContext &ctx,
                              const MHDFaceFluxTargets &tgt,
                              const MHDFaceKernelRange &rng,
                              const bool need_faceprep) {
  if constexpr (kSplitFacePreparation<rsolver_method_>) {
    if (need_faceprep) {
      LaunchFacePreparationKernel<rsolver_method_, ivx>(prep_name, ctx, rng);
    }
  }
  par_for<kRSolveMaxThreads, kRSolveMinBlocks<rsolver_method_>>(rsolve_name,
      DevExeSpace(),
    rng.work_start, rng.work_end, rng.kl, rng.ku, rng.jl, rng.ju, rng.il, rng.iu,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = a;
      SolveFaceFluxAt<rsolver_method_, ivx>(
          ctx, tgt, rng.InScalarRange(k, j, i), m, a - rng.work_start, k, j, i);
    });
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute MHD fluxes and
//! face-centered area-averaged EMFs.  Templated over the Riemann solver for GPU perf.

template <MHD_RSolver rsolver_method_>
void MHD::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int &nmhd_ = nmhd;
  int nvars = this->nvars;
  const bool dual_enabled_ = use_dual_energy;
  const int dual_idx_ = dual_energy_idx;
  const auto recon_method_ = recon_method;
  const int nwork1 = pmy_pack->nmb_thispack - 1;
  const bool sanitize_reconstructed_states =
      (recon_method_ != ReconstructionMethod::dc);
  // Tabulated EOS: rebuild the face internal energies from limited specific energy
  // (specific_energy_recon.hpp) in a pass between reconstruction and the face
  // preparation, so the solver assembles the face total energy from a bounded eint.
  constexpr bool newtonian_solver =
      rsolver_method_ == MHD_RSolver::advect ||
      rsolver_method_ == MHD_RSolver::llf ||
      rsolver_method_ == MHD_RSolver::hlle ||
      rsolver_method_ == MHD_RSolver::hlld ||
      rsolver_method_ == MHD_RSolver::roe;
  const bool specific_energy_faces = newtonian_solver &&
      sanitize_reconstructed_states && peos->eos_data.UsesTabulatedLTE();
  const int specific_energy_dual = dual_energy_pdv ? dual_idx_ : -1;
  if (nwork1 < 0) return;

  // Face-normal flux range. With FOFC enabled the first-order flux correction needs the
  // main fluxes/EMFs one cell beyond the active domain, so each direction's face-normal
  // range is extended by one cell on both sides (transverse ranges already cover the CT
  // edge, so they are unchanged).
  int il1 = is, iu1 = ie+1, jl2 = js, ju2 = je+1, kl3 = ks, ku3 = ke+1;
  if (use_fofc) {
    il1 = is-1; iu1 = ie+2;
    jl2 = js-1; ju2 = je+2;
    kl3 = ks-1; ku3 = ke+2;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto bl_ = bl3d;
  auto br_ = br3d;
  auto vf1_ = FluxBand(dual_vf.x1f);
  auto vf2_ = FluxBand(dual_vf.x2f);
  auto vf3_ = FluxBand(dual_vf.x3f);

  // Bind everything the face kernels read.
  MHDFaceStateContext face_ctx_;
  face_ctx_.eos = eos_;
  face_ctx_.indcs = indcs_;
  face_ctx_.size = size_;
  face_ctx_.coord = coord_;
  face_ctx_.wl = wl_;
  face_ctx_.wr = wr_;
  face_ctx_.bl = bl_;
  face_ctx_.br = br_;
  face_ctx_.is = is;
  face_ctx_.js = js;
  face_ctx_.ks = ks;
  face_ctx_.nmhd = nmhd_;
  face_ctx_.nvars = nvars;
  face_ctx_.dual_pdv = dual_energy_pdv;
  face_ctx_.dual_idx = dual_idx_;
  face_ctx_.dual_enabled = dual_enabled_;
  face_ctx_.sanitize_faces = sanitize_reconstructed_states;

  // Skip the preparation launch entirely when the face floor is off (e.g. donor-cell
  // reconstruction).
  const bool need_faceprep_ = sanitize_reconstructed_states;

  const int chunk_nmb = std::max(1, std::min(split_recon_chunk_nmb, nwork1 + 1));
  for (int work_start = 0; work_start <= nwork1; work_start += chunk_nmb) {
    const int work_end = std::min(nwork1, work_start + chunk_nmb - 1);

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto flx1 = FluxBand(uflx.x1f);
    auto &bx_ = b0.x1f;
    auto e31 = EmfBand(e3x1);
    auto e21 = EmfBand(e2x1);

    // CT-extended transverse range
    int jl = js, ju = je, kl = ks, ku = ke;
    if (pmy_pack->pmesh->multi_d) { jl = js-1; ju = je+1; }
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // Reconstruct W over cells i in [il1-1, iu1], variables n in [0, nvars-1]
    ReconDispatchChunk<IVX>(recon_method_, "mflux_x1_recon_w",
        work_start, work_end, work_start,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells i in [il1-1, iu1], components n in [0, 2]
    ReconDispatchChunk<IVX>(recon_method_, "mflux_x1_recon_b",
        work_start, work_end, work_start,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, 3, bcc0_, bl_, br_);

    const int s_il = use_fofc ? il1 : is;
    const int s_iu = use_fofc ? iu1 : ie+1;
    const int s_jl = use_fofc ? jl : js;
    const int s_ju = use_fofc ? ju : je;
    const int s_kl = use_fofc ? kl : ks;
    const int s_ku = use_fofc ? ku : ke;

    if (specific_energy_faces) {
      hydro_reconstruction::SpecificEnergyFaceDispatch<IVX>(recon_method_,
          "mflux_x1_eps", work_start, work_end, work_start,
          kl, ku, jl, ju, il1, iu1, eos_, specific_energy_dual, false,
          pmy_pack->lat_active_indices.d_view, w0_, wl_, wr_);
    }

    // Prepare the reconstructed faces, then Riemann solve over faces i in [il1, iu1]
    LaunchMHDFaceFluxKernels<rsolver_method_, IVX>(
        "mflux_x1_faceprep", "mflux_x1_rsolve", face_ctx_,
        MHDFaceFluxTargets{bx_, flx1, e31, e21, vf1_},
        MHDFaceKernelRange{work_start, work_end, kl, ku, jl, ju, il1, iu1,
                           s_kl, s_ku, s_jl, s_ju, s_il, s_iu},
        need_faceprep_);
  }

  //------------------------------------------------------------------------------------
  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto flx2 = FluxBand(uflx.x2f);
    auto &by_ = b0.x2f;
    auto e12 = EmfBand(e1x2);
    auto e32 = EmfBand(e3x2);

    int kl = ks, ku = ke;
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // Reconstruct W over cells j in [jl2-1, ju2], i in [is-1, ie+1], n in [0, nvars-1]
    ReconDispatchChunk<IVY>(recon_method_, "mflux_x2_recon_w",
        work_start, work_end, work_start,
        kl, ku, jl2-1, ju2, is-1, ie+1, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells j in [jl2-1, ju2], i in [is-1, ie+1], n in [0, 2]
    ReconDispatchChunk<IVY>(recon_method_, "mflux_x2_recon_b",
        work_start, work_end, work_start,
        kl, ku, jl2-1, ju2, is-1, ie+1, eos_, false, 3, bcc0_, bl_, br_);

    const int s_il = use_fofc ? is-1 : is;
    const int s_iu = use_fofc ? ie+1 : ie;
    const int s_jl = use_fofc ? jl2 : js;
    const int s_ju = use_fofc ? ju2 : je+1;
    const int s_kl = use_fofc ? kl : ks;
    const int s_ku = use_fofc ? ku : ke;

    if (specific_energy_faces) {
      hydro_reconstruction::SpecificEnergyFaceDispatch<IVY>(recon_method_,
          "mflux_x2_eps", work_start, work_end, work_start,
          kl, ku, jl2, ju2, is-1, ie+1, eos_, specific_energy_dual, false,
          pmy_pack->lat_active_indices.d_view, w0_, wl_, wr_);
    }

    // Prepare the reconstructed faces, then Riemann solve over faces j in [jl2, ju2],
    // i in [is-1, ie+1]
    LaunchMHDFaceFluxKernels<rsolver_method_, IVY>(
        "mflux_x2_faceprep", "mflux_x2_rsolve", face_ctx_,
        MHDFaceFluxTargets{by_, flx2, e12, e32, vf2_},
        MHDFaceKernelRange{work_start, work_end, kl, ku, jl2, ju2, is-1, ie+1,
                           s_kl, s_ku, s_jl, s_ju, s_il, s_iu},
        need_faceprep_);
  }

  //------------------------------------------------------------------------------------
  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto flx3 = FluxBand(uflx.x3f);
    auto &bz_ = b0.x3f;
    auto e23 = EmfBand(e2x3);
    auto e13 = EmfBand(e1x3);

    // Reconstruct W over cells k in [kl3-1, ku3], j in [js-1, je+1], i in [is-1, ie+1],
    // variables n in [0, nvars-1]
    ReconDispatchChunk<IVZ>(recon_method_, "mflux_x3_recon_w",
        work_start, work_end, work_start,
        kl3-1, ku3, js-1, je+1, is-1, ie+1, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over the same cells, components n in [0, 2]
    ReconDispatchChunk<IVZ>(recon_method_, "mflux_x3_recon_b",
        work_start, work_end, work_start,
        kl3-1, ku3, js-1, je+1, is-1, ie+1, eos_, false, 3, bcc0_, bl_, br_);

    const int s_il = use_fofc ? is-1 : is;
    const int s_iu = use_fofc ? ie+1 : ie;
    const int s_jl = use_fofc ? js-1 : js;
    const int s_ju = use_fofc ? je+1 : je;
    const int s_kl = use_fofc ? kl3 : ks;
    const int s_ku = use_fofc ? ku3 : ke+1;

    if (specific_energy_faces) {
      hydro_reconstruction::SpecificEnergyFaceDispatch<IVZ>(recon_method_,
          "mflux_x3_eps", work_start, work_end, work_start,
          kl3, ku3, js-1, je+1, is-1, ie+1, eos_, specific_energy_dual, false,
          pmy_pack->lat_active_indices.d_view, w0_, wl_, wr_);
    }

    // Prepare the reconstructed faces, then Riemann solve over faces k in [kl3, ku3],
    // j in [js-1, je+1], i in [is-1, ie+1]
    LaunchMHDFaceFluxKernels<rsolver_method_, IVZ>(
        "mflux_x3_faceprep", "mflux_x3_rsolve", face_ctx_,
        MHDFaceFluxTargets{bz_, flx3, e23, e13, vf3_},
        MHDFaceKernelRange{work_start, work_end, kl3, ku3, js-1, je+1, is-1, ie+1,
                           s_kl, s_ku, s_jl, s_ju, s_il, s_iu},
        need_faceprep_);
  }
  }

  return;
}

// One direction's face kernels of one solver.  The GR solvers compile their x2 and x3
// directions in units of their own (mhd_fluxes_<solver>_gr_x{2,3}.cpp), which
// instantiate them; mhd_fluxes_<solver>_gr.cpp declares them extern.
#define MHD_FACE_FLUX_KERNELS_DECL(R, IV) \
void LaunchMHDFaceFluxKernels<R, IV>( \
    const char *prep_name, const char *rsolve_name, const MHDFaceStateContext &ctx, \
    const MHDFaceFluxTargets &tgt, const MHDFaceKernelRange &rng, \
    const bool need_faceprep)

} // namespace mhd

#endif  // MHD_MHD_FLUXES_IMPL_HPP_
