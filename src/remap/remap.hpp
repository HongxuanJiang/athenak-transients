#ifndef REMAP_REMAP_HPP_
#define REMAP_REMAP_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap.hpp
//! \brief Universal restart-remap module.
//!
//! Loads an AthenaK restart file written by a DIFFERENT mesh configuration and resamples
//! its state onto the current mesh: cell-centered conserved variables (hydro or MHD,
//! optionally radiation i0) through the AMR-hierarchy
//! sampling engine originally developed in the tde_external pgen, and the face-centered
//! magnetic field through an exactly divergence-free vector-potential path (source B ->
//! exact discrete inverse curl on the source root covering grid -> C1 Catmull-Rom
//! interpolation of A -> discrete curl on the target mesh).  See
//! docs/remap_module_design.md for the full design.
//!
//! Relativity classes (remap_impl.hpp RemapRelClass) gate the behavior: Newtonian sources
//! keep the original floor-fade CC engine bit-for-bit, while fixed-GR and dyn-GR (<adm>,
//! analytic metric) sources use the keep-target CC engine, remap the GR conserved
//! variables verbatim (no magnetic-energy swap), and refresh the analytic metric at the
//! source time.  z4c sources and targets are refused.
//!
//! The module runs host-side at ProblemGenerator time or from a pgen after_cycle hook.
//! Mid-run callers must call Driver::InitBoundaryValuesAndPrimitives afterwards.

#include <functional>
#include <string>

#include "athena.hpp"

class Mesh;
class MeshBlockPack;
class ParameterInput;
class ProblemGenerator;

namespace remap {

//----------------------------------------------------------------------------------------
//! \enum RemapBandMode
//! \brief what the CC engine does inside (and outside) the old-domain boundary band.
//!
//! kFloorFade  — the historical Newtonian behavior: every target cell is overwritten, the
//!               band is reconstructed from the interior profile and tapered toward the
//!               ambient hydro floor, and cells outside the old box become floor state.
//! kKeepTarget — the target's own ProblemGenerator state is the ambient: cells with no
//!               source support are left untouched, band cells are a linear blend of the
//!               (plainly interpolated) source conserved state and the pgen state, and
//!               only fully covered cells are overwritten.  No Newtonian thermodynamics
//!               is applied anywhere, which is what makes it the only valid mode in GR.

enum class RemapBandMode { kFloorFade, kKeepTarget };

//----------------------------------------------------------------------------------------
//! \struct RemapOptions
//! \brief caller-provided behavior knobs and hooks

struct RemapOptions {
  std::string source_path;                  // path of the source restart file

  // CC behavior (defaults preserve the historical tde_external behavior exactly)
  bool use_transition_band = true;          // old-domain boundary reconstruction band
  RemapBandMode band_mode = RemapBandMode::kFloorFade;  // band semantics (see above)
  // band_mode_auto: resolve band_mode from the relativity class of the remap (GR ->
  // kKeepTarget, Newtonian -> kFloorFade) once the source has been classified.  Set
  // false by an explicit <remap>/band_mode = floor|keep, which is then honored verbatim.
  bool band_mode_auto = true;
  // cells for which skip_cell returns true keep the ambient state (e.g. excision): the
  // ambient floor under kFloorFade, the untouched pgen state under kKeepTarget
  std::function<bool(Real x, Real y, Real z)> skip_cell;

  // FC (MHD) behavior
  bool b_coarsen_ok = false;                // allow multi-level MHD source (B restricted
                                            // to the source root grid; CC keeps detail)
  int b_taper_root_cells = 4;               // smoothstep taper width (source root cells)
                                            // applied to A outside the source box
  bool b_report = true;                     // print inverse-curl consistency diagnostics

  // optional CC groups
  bool remap_radiation_i0 = true;           // remap radiation i0 intensities if both
                                            // sides carry the module (GR only)
};

//----------------------------------------------------------------------------------------
//! \struct RemapSummary
//! \brief what the remap actually did; returned by LoadAndApplyRemap and passed to the
//! ProblemGenerator user_remap_post_func hook so pgens can react.

struct RemapSummary {
  Real time = 0.0;              // source time installed into the mesh
  int ncycle = 0;
  bool fc_applied = false;      // face-centered B was remapped
  bool gr_mode = false;         // relativity class was kFixedGR or kDynGRAnalytic
  bool i0_applied = false;      // radiation (i0 intensities) was remapped
};

//----------------------------------------------------------------------------------------
//! \fn LoadAndApplyRemap
//! \brief Load the source restart and remap its state onto pmbp.
//!
//! Sets pm->time/dt/ncycle from the source, optionally copies <outputN>
//! file_number/last_time into dst_pin, and invokes on_loaded(source_pin) after the source
//! parameter dump is available (before the state is applied) so the pgen can restore its
//! own metadata (e.g. the TDE BH/frame record).  Fatal (clean exit) on any unsupported
//! source physics; see docs/remap_module_design.md section 0 for the exclusion list.

RemapSummary LoadAndApplyRemap(Mesh *pm, MeshBlockPack *pmbp, ParameterInput *dst_pin,
                               bool copy_output_state, const char *banner_label,
                               const RemapOptions &opts,
                               const std::function<void(ParameterInput *)> &on_loaded
                                   = nullptr);

//----------------------------------------------------------------------------------------
// <remap>-block auto mode: a pgen-independent activation path, wired like a module-level
// block (<gravity>-style).  If the input carries a <remap> block with enable = true
// (default true when the block exists — keep an explicit enable key in production inputs
// so it can be toggled from the command line), the ProblemGenerator fresh-start
// constructor calls MaybeAutoRemap right after the pgen function returns.  Any pgen can
// therefore be remapped with ZERO pgen code; pgens with special needs enroll the
// ProblemGenerator hook members user_remap_skip_func / user_remap_loaded_func /
// user_remap_post_func inside their pgen function.
//
// <remap> block keys (all optional except source):
//   enable            = true      master switch (block presence defaults it to true)
//   source            = <path>    source restart file (required when enabled)
//   copy_output_state = true      carry <outputN> file_number/last_time forward
//   transition_band   = true      old-domain boundary reconstruction band (CC)
//   band_mode         = auto      band semantics: auto | floor | keep.  auto selects
//                                 "keep" for GR sources and "floor" for Newtonian ones;
//                                 "floor" on a GR remap is a fatal error (its ambient
//                                 thermodynamics is Newtonian by construction)
//   b_coarsen_ok      = false     accept multi-level MHD sources (B -> root grid)
//   b_taper_root_cells= 4         A-taper width outside the source box (root cells)
//   b_report          = true      print FC diagnostics
//   radiation_i0      = true      remap radiation i0 when both sides carry it

bool IsAutoRemapEnabled(ParameterInput *pin);
RemapOptions OptionsFromInput(ParameterInput *pin);
void MaybeAutoRemap(ProblemGenerator *pgen, ParameterInput *pin, Mesh *pm);

}  // namespace remap

#endif  // REMAP_REMAP_HPP_
