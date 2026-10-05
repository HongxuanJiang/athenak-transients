//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap.cpp
//! \brief orchestration entry point of the universal restart-remap module.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "pgen/pgen.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

namespace remap {

namespace {

[[noreturn]] void FatalRemap(const std::string &msg) {
  std::cout << "### FATAL ERROR in remap module" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

// carry <outputN> file_number/last_time forward so the remapped run continues the
// source run's output sequence (matched by block name)
void CopyRemapOutputState(ParameterInput *dst, ParameterInput *src) {
  if (dst == nullptr) return;
  if (src == nullptr) return;
  for (const auto &block : src->block) {
    if (block.block_name.compare(0, 6, "output") != 0) continue;
    if (!dst->DoesBlockExist(block.block_name)) continue;
    if (src->DoesParameterExist(block.block_name, "file_number")) {
      dst->SetInteger(block.block_name, "file_number",
                      src->GetInteger(block.block_name, "file_number"));
    }
    if (src->DoesParameterExist(block.block_name, "last_time")) {
      dst->SetReal(block.block_name, "last_time",
                   src->GetReal(block.block_name, "last_time"));
    }
  }
}

}  // namespace

RemapSummary LoadAndApplyRemap(Mesh *pm, MeshBlockPack *pmbp, ParameterInput *dst_pin,
                               bool copy_output_state, const char *banner_label,
                               const RemapOptions &opts,
                               const std::function<void(ParameterInput *)> &on_loaded) {
  RemapSummary summary;
  if (pm == nullptr || pmbp == nullptr) return summary;
  // Remap and LAT (<time>/lat) cannot run together: the remap replaces the state on every
  // block at once, while LAT carries per-block time levels and bin metadata built against
  // the state that was there before, and nothing rebuilds them for the new one.  The
  // supported flow is to remap with LAT off and restart the remapped run with LAT on.
  // This check used to live in tde_external.cpp alone, so every other pgen -- BBH
  // included -- walked past it in silence into undefined behavior; it belongs to the
  // module, on the one path every caller (auto <remap> block, direct call) takes.
  if (dst_pin != nullptr && dst_pin->IsLATEnabled()) {
    FatalRemap("Remap cannot be combined with time/lat = true.  Remap first with LAT "
               "off, then restart the remapped run with time/lat = true.");
  }
  if (opts.source_path.empty()) {
    std::cout << "### FATAL ERROR in remap::LoadAndApplyRemap" << std::endl
              << "Remap requested without a valid source restart path." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ParameterInput src_pin;
  impl::RemapSourceData src;
  impl::LoadRemapSourceData(opts.source_path, pm, pmbp, opts, src, src_pin);

  // ---- relativity class: resolve the band mode, then audit the two parameter sets -----
  RemapOptions eff_opts = opts;
  if (eff_opts.band_mode_auto) {
    eff_opts.band_mode = src.gr_mode ? RemapBandMode::kKeepTarget
                                     : RemapBandMode::kFloorFade;
  } else if (src.gr_mode && eff_opts.band_mode == RemapBandMode::kFloorFade) {
    FatalRemap(std::string("<remap>/band_mode = floor is invalid for a ") +
               impl::RemapRelClassName(src.rel_class) + " remap: the floor-fade band "
               "reconstructs a Newtonian ambient state (rho/p floors, v = p/rho, "
               "E = eint + 0.5*rho*v^2), which has no meaning for densitized GR "
               "conserved variables.  Use band_mode = keep (or auto).");
  }
  impl::CheckRemapGRConsistency(src_pin, dst_pin, src);

  const bool is_mhd = (pmbp->pmhd != nullptr);
  const bool use_dual = is_mhd ? pmbp->pmhd->use_dual_energy
                               : pmbp->phydro->use_dual_energy;
  // The Newtonian floor prep (outer source ghost zones floored, dual-energy aux column
  // rebuilt from the remapped state) is meaningless in GR: there is no ambient floor
  // state, and the keep-mode pass never writes one.
  if (!src.gr_mode) {
    auto eos = is_mhd ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
    const Real floor_rho = eos.dfloor;
    const Real floor_eint = eos.HostClampHydroInternalEnergyDensity(
        floor_rho, eos.HostHydroInternalEnergyDensityFloor(floor_rho));
    const int dual_idx = use_dual ? (is_mhd ? pmbp->pmhd->dual_energy_idx
                                            : pmbp->phydro->dual_energy_idx) : -1;
    impl::FloorOuterSourceGhostZones(src, dual_idx, floor_rho, floor_eint);
    // The CC engine interpolates the source's THERMAL energy and rebuilds the total from
    // it (see sample_full_state in remap_cc.cpp), so that column is built here: after the
    // outer ghost zones are floored, whose thermal state it has to agree with.
    const int base_nvars = is_mhd ? (pmbp->pmhd->nmhd + pmbp->pmhd->nscalars)
                                  : (pmbp->phydro->nhydro + pmbp->phydro->nscalars);
    impl::BuildSourceThermalEnergy(src, base_nvars, floor_rho, floor_eint);
  }

  if (on_loaded) {
    on_loaded(&src_pin);
  }

  impl::ApplyRemapCC(pm, pmbp, eff_opts, src);
  if (src.load_fc) {
    impl::BuildCoveringPotential(src, eff_opts);
    impl::ApplyRemapFC(pm, pmbp, eff_opts, src);
  }
  // A floor-fade pass writes the dual-energy column of every target cell (the source's
  // own auxiliary where it has one, else e_int of the remapped state), so the startup
  // re-seed from E - KE, which would discard the auxiliary the source trusted in its cold
  // cells, is skipped.  A keep pass leaves the target's pgen cells to be seeded.
  if (!src.gr_mode && use_dual) {
    const bool seed = (eff_opts.band_mode == RemapBandMode::kKeepTarget);
    if (is_mhd) {
      pmbp->pmhd->dual_energy_needs_init = seed;
    } else {
      pmbp->phydro->dual_energy_needs_init = seed;
    }
  }

  pm->time = src.time;
  pm->dt = src.dt;
  pm->dtold = src.dt;
  pm->ncycle = src.ncycle;

  // Refresh the analytic ADM metric (and, for BBH, the excision + M1 radiation masks) at
  // the SOURCE time now that pm->time carries it.  Dyn-GR pgens whose metric callback is
  // registered therefore need no remap-specific code at all: the state the pgen built at
  // t = 0 has been replaced by state at t = src.time, and this puts the spacetime the
  // rest of the initialization (C2P, boundary values) sees back in sync with it.
  if (pmbp->padm != nullptr && pmbp->padm->SetADMVariables != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  summary.time = src.time;
  summary.ncycle = src.ncycle;
  summary.fc_applied = src.load_fc;
  summary.gr_mode = src.gr_mode;
  summary.i0_applied = src.load_i0;

  if (dst_pin != nullptr) {
    dst_pin->SetReal("time", "start_time", src.time);
    if (copy_output_state) {
      CopyRemapOutputState(dst_pin, &src_pin);
    }
  }

  if (global_variable::my_rank == 0) {
    auto group_state = [](const bool applied, const bool discarded,
                          const bool target_only, const bool present_source) {
      if (applied) return "yes";
      if (discarded) return "no (source group discarded: target lacks the module)";
      if (target_only) return "no (source has none; target pgen state kept)";
      if (present_source) return "no (disabled by option)";
      return "no";
    };
    std::cout << std::endl
              << ((banner_label != nullptr) ? banner_label : "--- Remap ---") << std::endl
              << "source restart         = " << eff_opts.source_path << std::endl
              << "copy output state      = "
              << (copy_output_state ? "true" : "false") << std::endl
              << "relativity class       = "
              << impl::RemapRelClassName(src.rel_class) << std::endl
              << "band mode              = "
              << ((eff_opts.band_mode == RemapBandMode::kKeepTarget)
                      ? "keep (target pgen state is the ambient)"
                      : "floor (ambient hydro floor outside the source box)")
              << (eff_opts.band_mode_auto ? " [auto]" : " [explicit]") << std::endl
              << "transition band        = "
              << (eff_opts.use_transition_band ? "on" : "off (pure interpolation)")
              << std::endl
              << "source time            = " << src.time << std::endl
              << "source cycle           = " << src.ncycle << std::endl
              << "loaded source blocks   = " << src.blocks.size() << std::endl
              << "gas CC group           = " << (src.source_has_mhd ? "mhd" : "hydro")
              << " (nvars_file = " << src.nvars << ")" << std::endl
              << "face-centered B remap  = " << (src.load_fc ? "yes" : "no") << std::endl
              << "radiation i0 remap     = "
              << group_state(src.load_i0, src.i0_discarded, src.i0_target_only,
                             src.source_has_rad) << std::endl;
    if (src.nadm_vars > 0) {
      std::cout << "skipped source <adm>   = " << src.nadm_vars << " variables"
                << std::endl;
    }
    std::cout << std::endl;
  }
  return summary;
}

bool IsAutoRemapEnabled(ParameterInput *pin) {
  if (pin == nullptr || !pin->DoesBlockExist("remap")) return false;
  return pin->GetOrAddBoolean("remap", "enable", true);
}

RemapOptions OptionsFromInput(ParameterInput *pin) {
  RemapOptions opts;
  opts.source_path = pin->GetOrAddString("remap", "source", "");
  opts.use_transition_band = pin->GetOrAddBoolean("remap", "transition_band", true);
  // band_mode: auto (default) | floor | keep.  "auto" is resolved from the relativity
  // class once the source has been classified, in LoadAndApplyRemap.
  std::string band = pin->GetOrAddString("remap", "band_mode", "auto");
  std::transform(band.begin(), band.end(), band.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (band == "auto") {
    opts.band_mode_auto = true;
    opts.band_mode = RemapBandMode::kFloorFade;
  } else if (band == "floor") {
    opts.band_mode_auto = false;
    opts.band_mode = RemapBandMode::kFloorFade;
  } else if (band == "keep") {
    opts.band_mode_auto = false;
    opts.band_mode = RemapBandMode::kKeepTarget;
  } else {
    FatalRemap("Unknown <remap>/band_mode = '" + band +
               "'; expected auto, floor or keep.");
  }
  opts.b_coarsen_ok = pin->GetOrAddBoolean("remap", "b_coarsen_ok", false);
  opts.b_taper_root_cells = pin->GetOrAddInteger("remap", "b_taper_root_cells", 4);
  opts.b_report = pin->GetOrAddBoolean("remap", "b_report", true);
  opts.remap_radiation_i0 = pin->GetOrAddBoolean("remap", "radiation_i0", true);
  return opts;
}

void MaybeAutoRemap(ProblemGenerator *pgen, ParameterInput *pin, Mesh *pm) {
  if (!IsAutoRemapEnabled(pin)) return;
  RemapOptions opts = OptionsFromInput(pin);
  opts.skip_cell = pgen->user_remap_skip_func;
  const bool copy_out = pin->GetOrAddBoolean("remap", "copy_output_state", true);
  const RemapSummary summary =
      LoadAndApplyRemap(pm, pm->pmb_pack, pin, copy_out, "--- Remap (<remap> block) ---",
                        opts, pgen->user_remap_loaded_func);
  if (pgen->user_remap_post_func) {
    pgen->user_remap_post_func(summary);
  }
}

}  // namespace remap
