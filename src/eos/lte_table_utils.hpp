#ifndef EOS_LTE_TABLE_UTILS_HPP_
#define EOS_LTE_TABLE_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lte_table_utils.hpp
//! \brief Shared setup, QA, and startup diagnostics for the nonrel H+He LTE table EOS.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "utils/tr_table.hpp"

namespace lte_table_utils {

namespace {

[[noreturn]] inline void Fatal(const std::string &file, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

inline void Require(const bool cond, const std::string &file, const std::string &msg) {
  if (!cond) {
    Fatal(file, msg);
  }
}

inline Real GetUniformSpacing(const double *values, const int npts,
    const std::string &name,
                              const std::string &file) {
  Require(npts >= 2, file, "LTE table axis '" + name + "' must have at least 2 points.");
  const Real dx = static_cast<Real>(values[1] - values[0]);
  Require(dx > 0.0, file, "LTE table axis '" + name + "' must be strictly increasing.");
  const Real tol = 1.0e-10*fmax(static_cast<Real>(1.0), fabs(dx));
  for (int i = 1; i < (npts - 1); ++i) {
    const Real ddi = static_cast<Real>(values[i + 1] - values[i]);
    Require(fabs(ddi - dx) <= tol, file,
            "LTE table axis '" + name + "' must be uniformly spaced.");
  }
  return dx;
}

inline Real Interp1D(const Real x0, const Real x1, const Real y0, const Real y1,
                     const Real x) {
  const Real denom = x1 - x0;
  if (fabs(denom) <= 0.0) {
    return y0;
  }
  return y0 + (x - x0)*(y1 - y0)/denom;
}

inline Real InvertMonotonicFieldAtRow(const double *logtemp, const double *field_row,
                                      const int ntemp, const Real target) {
  if (target <= static_cast<Real>(field_row[0])) {
    return static_cast<Real>(logtemp[0]);
  }
  if (target >= static_cast<Real>(field_row[ntemp - 1])) {
    return static_cast<Real>(logtemp[ntemp - 1]);
  }
  int ilo = 0;
  int ihi = ntemp - 1;
  while (ihi - ilo > 1) {
    const int imid = ilo + (ihi - ilo)/2;
    if (target <= static_cast<Real>(field_row[imid])) {
      ihi = imid;
    } else {
      ilo = imid;
    }
  }
  return Interp1D(static_cast<Real>(field_row[ilo]), static_cast<Real>(field_row[ihi]),
                  static_cast<Real>(logtemp[ilo]), static_cast<Real>(logtemp[ihi]), target);
}

inline Real EvalFieldAtLogTemp(const double *logtemp, const double *field_row,
                               const int ntemp, const Real log_temp) {
  if (log_temp <= static_cast<Real>(logtemp[0])) {
    return static_cast<Real>(field_row[0]);
  }
  if (log_temp >= static_cast<Real>(logtemp[ntemp - 1])) {
    return static_cast<Real>(field_row[ntemp - 1]);
  }
  int ilo = 0;
  int ihi = ntemp - 1;
  while (ihi - ilo > 1) {
    const int imid = ilo + (ihi - ilo)/2;
    if (log_temp <= static_cast<Real>(logtemp[imid])) {
      ihi = imid;
    } else {
      ilo = imid;
    }
  }
  return Interp1D(static_cast<Real>(logtemp[ilo]), static_cast<Real>(logtemp[ihi]),
                  static_cast<Real>(field_row[ilo]), static_cast<Real>(field_row[ihi]),
                  log_temp);
}

// Above the last temperature row the log fields continue as the power law of the last
// tabulated interval; radiation and e-/e+ pairs make eps, p and cs2 scale as T^4 there.
inline Real ExtrapolateLogFieldAtLogTemp(const double *logtemp, const double *field_row,
                                         const int ntemp, const Real log_temp) {
  const Real log_t_top = static_cast<Real>(logtemp[ntemp - 1]);
  if (log_temp <= log_t_top) {
    return EvalFieldAtLogTemp(logtemp, field_row, ntemp, log_temp);
  }
  const Real slope = static_cast<Real>(field_row[ntemp - 1] - field_row[ntemp - 2])/
                     static_cast<Real>(logtemp[ntemp - 1] - logtemp[ntemp - 2]);
  return static_cast<Real>(field_row[ntemp - 1]) + slope*(log_temp - log_t_top);
}

inline Real ExtrapolateLogTempAboveRow(const double *logtemp, const double *field_row,
                                       const int ntemp, const Real target) {
  const Real dv = static_cast<Real>(field_row[ntemp - 1] - field_row[ntemp - 2]);
  const Real frac = (dv > 0.0)
      ? ((target - static_cast<Real>(field_row[ntemp - 1]))/dv) : 0.0;
  return static_cast<Real>(logtemp[ntemp - 1]) +
         frac*static_cast<Real>(logtemp[ntemp - 1] - logtemp[ntemp - 2]);
}

inline const char *BoundsModeName(const SahaBoundsMode mode) {
  return (mode == SahaBoundsMode::error) ? "error" : "clamp";
}

inline const char *OnOff(const bool enabled) {
  return enabled ? "on" : "off";
}

inline bool IsSimpleLTETableType(const std::string &table_type) {
  return (table_type == "lte_hhe_lte") || (table_type == "lte_hhe_prad_lte");
}

inline bool IsT13LTETableType(const std::string &table_type) {
  return (table_type == "lte_t13_lte") || (table_type == "lte_t13_prad_lte");
}

inline bool IsHybridLTETableType(const std::string &table_type) {
  return (table_type == "lte_hybrid_hhe_t13_lte") ||
         (table_type == "lte_hybrid_hhe_t13_prad_lte");
}

inline bool IsMaskedUnionLTETableType(const std::string &table_type) {
  return (table_type == "lte_scvh1995_hhe") ||
         (table_type == "lte_scvh1995_hhe_prad") ||
         (table_type == "lte_scvh_t13_union") ||
         (table_type == "lte_scvh_t13_union_prad") ||
         (table_type == "lte_scvh_t13_helm_union") ||
         (table_type == "lte_scvh_t13_helm_union_prad") ||
         (table_type == "lte_scvh_t13_cp_union") ||
         (table_type == "lte_scvh_t13_cp_union_prad") ||
         (table_type == "lte_scvh_t13_cp_helm_union") ||
         (table_type == "lte_scvh_t13_cp_helm_union_prad") ||
         (table_type == "lte_chabrier2021_t13_helm_union") ||
         (table_type == "lte_chabrier2021_t13_helm_union_prad");
}

inline std::string ExpectedTableTypeForEosName(const std::string &eos_name) {
  if (eos_name == "lte_table_hhe") return "lte_hhe_lte";
  if (eos_name == "lte_table_hhe_prad") return "lte_hhe_prad_lte";
  if (eos_name == "lte_table_t13") return "lte_t13_lte";
  if (eos_name == "lte_table_t13_prad") return "lte_t13_prad_lte";
  if (eos_name == "lte_table_scvh1995_hhe") return "lte_scvh1995_hhe";
  if (eos_name == "lte_table_scvh1995_hhe_prad") return "lte_scvh1995_hhe_prad";
  if (eos_name == "lte_table_scvh_t13_union") return "lte_scvh_t13_union";
  if (eos_name == "lte_table_scvh_t13_union_prad") return "lte_scvh_t13_union_prad";
  if (eos_name == "lte_table_scvh_t13_helm_union") return "lte_scvh_t13_helm_union";
  if (eos_name == "lte_table_scvh_t13_helm_union_prad") return "lte_scvh_t13_helm_union_prad";
  if (eos_name == "lte_table_scvh_t13_cp_union") return "lte_scvh_t13_cp_union";
  if (eos_name == "lte_table_scvh_t13_cp_union_prad") return "lte_scvh_t13_cp_union_prad";
  if (eos_name == "lte_table_scvh_t13_cp_helm_union") return "lte_scvh_t13_cp_helm_union";
  if (eos_name == "lte_table_scvh_t13_cp_helm_union_prad") {
    return "lte_scvh_t13_cp_helm_union_prad";
  }
  if (eos_name == "lte_table_chabrier2021_t13_helm_union") {
    return "lte_chabrier2021_t13_helm_union";
  }
  if (eos_name == "lte_table_chabrier2021_t13_helm_union_prad") {
    return "lte_chabrier2021_t13_helm_union_prad";
  }
  if (eos_name == "lte_table_hybrid_hhe_t13") return "lte_hybrid_hhe_t13_lte";
  if (eos_name == "lte_table_hybrid_hhe_t13_prad") return "lte_hybrid_hhe_t13_prad_lte";
  return "";
}

inline bool RestartIsActive(ParameterInput *pin) {
  return pin->GetOrAddBoolean("saha_runtime", "restart_active", false);
}

inline std::string GetRestartEOS(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_eos_from_restart";
  return pin->GetOrAddString("saha_runtime", key, "unknown");
}

inline std::string GetRestartTable(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_table_from_restart";
  return pin->GetOrAddString("saha_runtime", key, "unknown");
}

inline std::string GetRestartTableType(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_table_type_from_restart";
  return pin->GetOrAddString("saha_runtime", key, "unknown");
}

inline Real GetRestartX(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_lte_x_from_restart";
  return pin->GetOrAddReal("saha_runtime", key, -1.0);
}

inline Real GetRestartY(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_lte_y_from_restart";
  return pin->GetOrAddReal("saha_runtime", key, -1.0);
}

inline Real GetRestartPrad(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_lte_prad_from_restart";
  return pin->GetOrAddReal("saha_runtime", key, -1.0);
}

inline Real GetRestartH2(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_lte_h2_from_restart";
  return pin->GetOrAddReal("saha_runtime", key, -1.0);
}

inline Real GetRestartZPE(ParameterInput *pin, const std::string &block) {
  const std::string key = block + "_lte_zpe_from_restart";
  return pin->GetOrAddReal("saha_runtime", key, -1.0);
}

inline bool AllowReinterpretiveRestart(ParameterInput *pin, const std::string &block) {
  return pin->GetOrAddBoolean(block, "allow_reinterpretive_restart", false);
}

inline void RunTableQAChecks(const std::string &fname, TableReader::Table &table,
                             const int nrho, const int ntemp, const bool has_h2,
                             const bool debug_checks, const std::string &file) {
  const double *logrho = table["logrho"];
  const double *logpress = table["logpress"];
  const double *logeps = table["logeps"];
  const double *logcs2 = table["logcs2"];
  const double *gamma1 = table["gamma1"];
  const double *gamma3m1 = table["gamma3m1"];
  const double *xh2 = has_h2 ? table["xh2"] : nullptr;
  const double *xion = table["xion"];
  const double *xhe1 = table["xhe1"];
  const double *xhe2 = table["xhe2"];
  const double *mu = table["mu"];
  const double *beta_rad = table["beta_rad"];
  const Real rel_tol = 5.0e-6;

  for (int ir = 0; ir < nrho; ++ir) {
    const int row = ir*ntemp;
    for (int it = 0; it < ntemp; ++it) {
      const int idx = row + it;
      const Real p = exp(static_cast<Real>(logpress[idx]));
      const Real eps = exp(static_cast<Real>(logeps[idx]));
      const Real cs2 = exp(static_cast<Real>(logcs2[idx]));
      const Real g1 = static_cast<Real>(gamma1[idx]);
      const Real g3m1 = static_cast<Real>(gamma3m1[idx]);
      const Real xh2_loc = has_h2 ? static_cast<Real>(xh2[idx]) : 0.0;
      const Real xh = static_cast<Real>(xion[idx]);
      const Real x1 = static_cast<Real>(xhe1[idx]);
      const Real x2 = static_cast<Real>(xhe2[idx]);
      const Real mu_loc = static_cast<Real>(mu[idx]);
      const Real beta = static_cast<Real>(beta_rad[idx]);
      Require(std::isfinite(logpress[idx]) && std::isfinite(logeps[idx]) &&
                  std::isfinite(logcs2[idx]) && std::isfinite(gamma1[idx]) &&
                  std::isfinite(gamma3m1[idx]) && std::isfinite(xion[idx]) &&
                  (!has_h2 || std::isfinite(xh2[idx])) &&
                  std::isfinite(xhe1[idx]) && std::isfinite(xhe2[idx]) &&
                  std::isfinite(mu[idx]) && std::isfinite(beta_rad[idx]),
              file, "Non-finite value found while validating LTE table '" + fname + "'.");
      Require((p > 0.0) && (eps > 0.0) && (cs2 > 0.0) && (g1 > 0.0) &&
                  (g3m1 > 0.0) && (mu_loc > 0.0),
              file, "Non-positive thermodynamic value found while validating LTE table '" +
                        fname + "'.");
      Require((xh2_loc >= 0.0) && (xh2_loc <= 1.0 + 1.0e-10) &&
                  (xh >= 0.0) && (xh <= 1.0 + 1.0e-10) &&
                  (x1 >= 0.0) && (x1 <= 1.0 + 1.0e-10) &&
                  (x2 >= 0.0) && (x2 <= 1.0 + 1.0e-10) &&
                  (x1 + x2 <= 1.0 + 1.0e-10),
              file, "Composition fraction outside the physical range in LTE table '" +
                        fname + "'.");
      Require(xh2_loc + xh <= 1.0 + 1.0e-8, file,
              "Hydrogen fractions xh2 + xion exceed unity in LTE table '" + fname + "'.");
      Require((beta >= 0.0) && (beta <= 1.0 + 1.0e-10), file,
              "Radiation-pressure fraction outside [0,1] in LTE table '" + fname + "'.");
      if (it > 0) {
        Require(logeps[idx] > logeps[idx - 1], file,
                "LTE table eps(rho,T) must increase monotonically with T at fixed rho.");
      }
      if (debug_checks) {
        const Real rho = exp(static_cast<Real>(logrho[ir]));
        const Real g1_cs = rho*cs2/p;
        const Real err = fabs(g1_cs - g1)/fmax(static_cast<Real>(1.0), fabs(g1));
        Require(err < rel_tol, file,
                "Gamma1 != rho*cs2/p consistency failure in LTE table '" + fname + "'.");
      }
    }
    if (debug_checks) {
      const double *logtemp = table["logtemp"];
      for (int it = 0; it < ntemp; ++it) {
        const int idx = row + it;
        const Real logt_rt = InvertMonotonicFieldAtRow(logtemp, &logeps[row], ntemp,
                                                       static_cast<Real>(logeps[idx]));
        const Real err = fabs(logt_rt - static_cast<Real>(logtemp[it]))/
                         fmax(static_cast<Real>(1.0),
                             fabs(static_cast<Real>(logtemp[it])));
        Require(err < 5.0e-6, file,
                "LTE table logeps inversion round-trip failed for '" + fname + "'.");
      }
    }
  }
}

}  // namespace

inline void InitializeLTETableEOS(const std::string &block, MeshBlockPack *pp,
                                  ParameterInput *pin, EOS_Data &eos_data,
                                  const std::string &file) {
  if (pp->pcoord->is_special_relativistic || pp->pcoord->is_general_relativistic) {
    Fatal(file, "<" + block + ">/eos = tabulated LTE is only implemented for "
                "non-relativistic " + block + ".");
  }
  if (pp->punit == nullptr) {
    Fatal(file, "<" + block + ">/eos = tabulated LTE requires a [units] block for "
                "code-to-cgs scaling.");
  }
  if (eos_data.sfloor > 0.0) {
    Fatal(file, "<" + block + ">/sfloor is not supported with <" + block +
                    ">/eos = tabulated LTE.");
  }

  eos_data.hydro_eos = HydroEOSModel::lte_table;
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = false;
  eos_data.gamma = 5.0/3.0;
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;
  eos_data.use_t = false;

  const std::string bounds_name = pin->DoesParameterExist(block, "lte_bounds") ?
      pin->GetString(block, "lte_bounds") :
      pin->GetOrAddString(block, "saha_bounds", "error");
  if (bounds_name == "error") {
    eos_data.saha_bounds_mode = SahaBoundsMode::error;
  } else if (bounds_name == "clamp") {
    eos_data.saha_bounds_mode = SahaBoundsMode::clamp;
  } else {
    Fatal(file, "<" + block + ">/lte_bounds must be either 'error' or 'clamp'.");
  }
  eos_data.saha_debug_checks = pin->DoesParameterExist(block, "lte_debug_checks") ?
      pin->GetBoolean(block, "lte_debug_checks") :
      pin->GetOrAddBoolean(block, "saha_debug_checks", false);

  TableReader::Table table;
  const std::string fname = pin->GetString(block, "table");
  const auto result = table.ReadTable(fname);
  if (result.error != TableReader::ReadResult::SUCCESS) {
    Fatal(file, "Failed to read lte_table '" + fname + "':\n" + result.message);
  }

  const auto metadata = table.GetMetadata();
  const auto scalars = table.GetScalars();
  auto find_meta = [&](const std::string &key) -> std::string {
    const auto it = metadata.find(key);
    return (it == metadata.end()) ? std::string() : it->second;
  };
  auto find_scalar = [&](const std::string &key, const Real def) -> Real {
    const auto it = scalars.find(key);
    return (it == scalars.end()) ? def : static_cast<Real>(it->second);
  };

  const std::string table_type = find_meta("table_type");
  const bool table_is_simple = IsSimpleLTETableType(table_type);
  const bool table_is_t13 = IsT13LTETableType(table_type);
  const bool table_is_hybrid = IsHybridLTETableType(table_type);
  const bool table_is_masked_union = IsMaskedUnionLTETableType(table_type);
  Require(table_is_simple || table_is_t13 || table_is_hybrid || table_is_masked_union,
      file,
          "LTE table metadata 'table_type' must be one of the supported AthenaK LTE "
          "families.");
  const std::string log_axis_base = find_meta("log_axis_base");
  Require(log_axis_base.empty() || log_axis_base == "e", file,
          "LTE table metadata 'log_axis_base' must be 'e'.");

  eos_data.lte_h_mass_fraction = find_scalar("x_h", 0.70);
  eos_data.lte_he_mass_fraction = find_scalar("y_he", 0.30);
  eos_data.lte_has_helium = eos_data.lte_he_mass_fraction > 0.0;
  eos_data.lte_has_h2 =
      (find_meta("h2_enabled") == "on") || table_is_t13 || table_is_hybrid;
  eos_data.lte_has_radiation =
      (find_meta("radiation_pressure") == "on") || (find_scalar("radiation_enabled",
          0.0) > 0.5);
  eos_data.lte_zpe_subtracted =
      (find_meta("zpe_subtracted") == "on") || (find_scalar("zpe_subtraction_enabled",
          0.0) > 0.5);

  if (table_is_t13) {
    Require(find_meta("partition_function_model") ==
                "tomida2013_appendix1_partition_functions",
            file, "T13 tables must declare "
                  "'partition_function_model = tomida2013_appendix1_partition_functions'.");
    Require(find_meta("ortho_para_ratio_h2") == "3:1_fixed", file,
            "T13 tables must declare "
            "'ortho_para_ratio_h2 = 3:1_fixed'.");
  }

  Require(fabs(eos_data.lte_h_mass_fraction + eos_data.lte_he_mass_fraction - 1.0) <= 1.0e-10,
          file, "LTE table must currently satisfy X + Y = 1 (Z = 0).");

  const std::string eos_name = pin->GetString(block, "eos");
  const std::string expected_type = ExpectedTableTypeForEosName(eos_name);
  if (!expected_type.empty()) {
    Require(table_type == expected_type, file,
            "<" + block + ">/eos = " + eos_name + " requires 'table_type = " +
                expected_type + "'.");
  }
  if (eos_name == "lte_table_hhe_prad" || eos_name == "lte_table_t13_prad" ||
      eos_name == "lte_table_scvh1995_hhe_prad" ||
      eos_name == "lte_table_scvh_t13_union_prad" ||
      eos_name == "lte_table_scvh_t13_helm_union_prad" ||
      eos_name == "lte_table_scvh_t13_cp_union_prad" ||
      eos_name == "lte_table_scvh_t13_cp_helm_union_prad" ||
      eos_name == "lte_table_chabrier2021_t13_helm_union_prad" ||
      eos_name == "lte_table_hybrid_hhe_t13_prad") {
    Require(eos_data.lte_has_radiation, file,
            "<" + block + ">/eos = " + eos_name + " requires a radiation-enabled table.");
  }
  if (eos_name == "lte_table_hhe" || eos_name == "lte_table_t13" ||
      eos_name == "lte_table_scvh1995_hhe" ||
      eos_name == "lte_table_scvh_t13_union" ||
      eos_name == "lte_table_scvh_t13_helm_union" ||
      eos_name == "lte_table_scvh_t13_cp_union" ||
      eos_name == "lte_table_scvh_t13_cp_helm_union" ||
      eos_name == "lte_table_chabrier2021_t13_helm_union" ||
      eos_name == "lte_table_hybrid_hhe_t13") {
    Require(!eos_data.lte_has_radiation, file,
            "<" + block + ">/eos = " + eos_name +
                " requires a non-radiation table.");
  }
  if (eos_name == "lte_table_t13" || eos_name == "lte_table_t13_prad" ||
      eos_name == "lte_table_scvh1995_hhe" ||
      eos_name == "lte_table_scvh1995_hhe_prad" ||
      eos_name == "lte_table_scvh_t13_union" ||
      eos_name == "lte_table_scvh_t13_union_prad" ||
      eos_name == "lte_table_scvh_t13_helm_union" ||
      eos_name == "lte_table_scvh_t13_helm_union_prad" ||
      eos_name == "lte_table_scvh_t13_cp_union" ||
      eos_name == "lte_table_scvh_t13_cp_union_prad" ||
      eos_name == "lte_table_scvh_t13_cp_helm_union" ||
      eos_name == "lte_table_scvh_t13_cp_helm_union_prad" ||
      eos_name == "lte_table_chabrier2021_t13_helm_union" ||
      eos_name == "lte_table_chabrier2021_t13_helm_union_prad" ||
      eos_name == "lte_table_hybrid_hhe_t13" ||
      eos_name == "lte_table_hybrid_hhe_t13_prad") {
    Require(eos_data.lte_has_h2, file,
            "<" + block + ">/eos = " + eos_name +
                " requires H2 support in the table.");
  }

  const auto point_info = table.GetPointInfo();
  Require(point_info.size() == 2, file,
          "LTE table must be 2D with axes (logrho, logtemp).");
  Require(point_info[0].first == "logrho" && point_info[1].first == "logtemp", file,
          "LTE table axes must be ordered as logrho, logtemp.");
  const int nrho = static_cast<int>(point_info[0].second);
  const int ntemp = static_cast<int>(point_info[1].second);
  Require(table.HasField("logpress") && table.HasField("logeps") &&
              table.HasField("logcs2") && table.HasField("gamma1") &&
              table.HasField("gamma3m1") &&
              (!eos_data.lte_has_h2 || table.HasField("xh2")) &&
              table.HasField("xion") && table.HasField("xhe1") &&
              table.HasField("xhe2") && table.HasField("mu") &&
              table.HasField("beta_rad"),
          file, "LTE table is missing one or more required fields: "
                "logpress, logeps, logcs2, gamma1, gamma3m1, xh2 (for T13 tables), "
                "xion, xhe1, xhe2, mu, beta_rad.");

  RunTableQAChecks(fname, table, nrho, ntemp, eos_data.lte_has_h2,
                   eos_data.saha_debug_checks, file);

  eos_data.saha_nrho = nrho;
  eos_data.saha_ntemp = ntemp;
  eos_data.saha_inv_dlogrho = 1.0/GetUniformSpacing(table["logrho"], nrho, "logrho",
      file);
  eos_data.saha_inv_dlogtemp =
      1.0/GetUniformSpacing(table["logtemp"], ntemp, "logtemp", file);

  eos_data.saha_logrho_min = static_cast<Real>(table["logrho"][0]);
  eos_data.saha_logrho_max = static_cast<Real>(table["logrho"][nrho - 1]);
  eos_data.saha_logtemp_min = static_cast<Real>(table["logtemp"][0]);
  eos_data.saha_logtemp_max = static_cast<Real>(table["logtemp"][ntemp - 1]);

  eos_data.saha_logrho = DvceArray1D<Real>("lte_logrho", nrho);
  eos_data.saha_logtemp = DvceArray1D<Real>("lte_logtemp", ntemp);
  eos_data.saha_table = DvceArray3D<Real>("lte_table", EOS_Data::saha_nvars, nrho, ntemp);
  eos_data.saha_logrho_h = HostArray1D<Real>("lte_logrho_h", nrho);
  eos_data.saha_logtemp_h = HostArray1D<Real>("lte_logtemp_h", ntemp);
  eos_data.saha_table_h = HostArray3D<Real>("lte_table_h", EOS_Data::saha_nvars, nrho,
      ntemp);

  auto h_logrho = eos_data.saha_logrho_h;
  auto h_logtemp = eos_data.saha_logtemp_h;
  auto h_table = eos_data.saha_table_h;

  for (int ir = 0; ir < nrho; ++ir) {
    h_logrho(ir) = static_cast<Real>(table["logrho"][ir]);
  }
  for (int it = 0; it < ntemp; ++it) {
    h_logtemp(it) = static_cast<Real>(table["logtemp"][it]);
  }

  const std::pair<const char *, int> fields[] = {
      {"logpress", EOS_Data::saha_logpress},
      {"logeps", EOS_Data::saha_logeps},
      {"logcs2", EOS_Data::saha_logcs2},
      {"gamma1", EOS_Data::saha_gamma1},
      {"gamma3m1", EOS_Data::saha_gamma3m1},
      {"xh2", EOS_Data::saha_xh2},
      {"xion", EOS_Data::saha_xion},
      {"xhe1", EOS_Data::saha_xhe1},
      {"xhe2", EOS_Data::saha_xhe2},
      {"mu", EOS_Data::saha_mu},
      {"beta_rad", EOS_Data::saha_beta_rad},
  };
  for (const auto &field : fields) {
    const double *src = table.HasField(field.first) ? table[field.first] : nullptr;
    for (int ir = 0; ir < nrho; ++ir) {
      for (int it = 0; it < ntemp; ++it) {
        h_table(field.second, ir, it) =
            (src != nullptr) ? static_cast<Real>(src[ir*ntemp + it]) : 0.0;
      }
    }
  }

  Kokkos::deep_copy(eos_data.saha_logrho, h_logrho);
  Kokkos::deep_copy(eos_data.saha_logtemp, h_logtemp);
  Kokkos::deep_copy(eos_data.saha_table, h_table);

  eos_data.density_unit_cgs = pp->punit->density_cgs();
  eos_data.pressure_unit_cgs = pp->punit->pressure_cgs();
  eos_data.specific_eint_unit_cgs =
      eos_data.pressure_unit_cgs/eos_data.density_unit_cgs;
  eos_data.temp_unit_cgs =
      pp->punit->length_cgs()*pp->punit->length_cgs()/
      (pp->punit->time_cgs()*pp->punit->time_cgs()) *
      static_cast<Real>(1.6735575e-24/1.380649e-16);
  eos_data.saha_tmin_code = exp(h_logtemp(0))/eos_data.temp_unit_cgs;
  eos_data.saha_tmax_code = exp(h_logtemp(ntemp - 1))/eos_data.temp_unit_cgs;

  const int cache_eps_factor = pin->GetOrAddInteger(block, "lte_cache_eps_factor", 8);
  Require(cache_eps_factor >= 1, file,
          "<" + block + ">/lte_cache_eps_factor must be >= 1.");
  const int floor_cache_factor = pin->GetOrAddInteger(block, "lte_floor_cache_factor",
      16);
  Require(floor_cache_factor >= 1, file,
          "<" + block + ">/lte_floor_cache_factor must be >= 1.");
  eos_data.saha_neps = cache_eps_factor*ntemp;
  eos_data.saha_floor_nrho = floor_cache_factor*(nrho - 1) + 1;
  eos_data.saha_floor_logrho_min = eos_data.saha_logrho_min;
  eos_data.saha_floor_logrho_max = eos_data.saha_logrho_max;
  eos_data.saha_floor_inv_dlogrho =
      static_cast<Real>(eos_data.saha_floor_nrho - 1)/
      (eos_data.saha_floor_logrho_max - eos_data.saha_floor_logrho_min);
  eos_data.saha_logeps_min = h_table(EOS_Data::saha_logeps, 0, 0);
  eos_data.saha_logeps_max = h_table(EOS_Data::saha_logeps, 0, ntemp - 1);
  for (int ir = 0; ir < nrho; ++ir) {
    eos_data.saha_logeps_min =
        std::min(eos_data.saha_logeps_min, h_table(EOS_Data::saha_logeps, ir, 0));
    eos_data.saha_logeps_max = std::max(eos_data.saha_logeps_max,
                                        h_table(EOS_Data::saha_logeps, ir, ntemp - 1));
  }
  Require(eos_data.saha_neps >= 2, file,
          "LTE inverse thermo cache must have at least 2 logeps points.");
  Require(eos_data.saha_logeps_max > eos_data.saha_logeps_min, file,
          "LTE inverse thermo cache has a degenerate logeps axis.");
  eos_data.saha_inv_dlogeps =
      static_cast<Real>(eos_data.saha_neps - 1)/
      (eos_data.saha_logeps_max - eos_data.saha_logeps_min);

  if (pin->DoesParameterExist(block, "tfloor_kelvin")) {
    eos_data.tfloor = pin->GetReal(block, "tfloor_kelvin")/eos_data.temp_unit_cgs;
  }

  eos_data.saha_thermo_cache =
      DvceArray3D<Real>("lte_thermo_cache", EOS_Data::saha_cache_nvars, nrho,
                        eos_data.saha_neps);
  eos_data.saha_thermo_cache_h =
      HostArray3D<Real>("lte_thermo_cache_h", EOS_Data::saha_cache_nvars, nrho,
                        eos_data.saha_neps);
  eos_data.saha_logeps_floor =
      DvceArray1D<Real>("lte_logeps_floor", eos_data.saha_floor_nrho);
  eos_data.saha_logeps_floor_h =
      HostArray1D<Real>("lte_logeps_floor_h", eos_data.saha_floor_nrho);
  eos_data.saha_logeps_ceil = DvceArray1D<Real>("lte_logeps_ceil", nrho);
  eos_data.saha_logeps_ceil_h = HostArray1D<Real>("lte_logeps_ceil_h", nrho);

  auto h_cache = eos_data.saha_thermo_cache_h;
  auto h_logeps_floor = eos_data.saha_logeps_floor_h;
  auto h_logeps_ceil = eos_data.saha_logeps_ceil_h;
  auto floor_rho_weights = [&](const Real log_rho, int &ir, Real &wr0, Real &wr1) {
    if (log_rho <= eos_data.saha_logrho_min) {
      ir = 0;
      wr0 = 1.0;
      wr1 = 0.0;
      return;
    }
    if (log_rho >= eos_data.saha_logrho_max) {
      ir = nrho - 2;
      wr0 = 0.0;
      wr1 = 1.0;
      return;
    }
    const Real x = (log_rho - eos_data.saha_logrho_min)*eos_data.saha_inv_dlogrho;
    ir = static_cast<int>(x);
    ir = std::max(0, std::min(ir, nrho - 2));
    wr1 = x - static_cast<Real>(ir);
    wr0 = 1.0 - wr1;
  };
  auto floor_field_at_temp_index = [&](const int iv, const Real log_rho, const int it) {
    int ir;
    Real wr0, wr1;
    floor_rho_weights(log_rho, ir, wr0, wr1);
    return wr0*h_table(iv, ir, it) + wr1*h_table(iv, ir + 1, it);
  };
  auto floor_eval_field = [&](const int iv, const Real log_rho, const Real log_temp) {
    int ir;
    Real wr0, wr1;
    floor_rho_weights(log_rho, ir, wr0, wr1);
    if (log_temp <= eos_data.saha_logtemp_min) {
      return wr0*h_table(iv, ir, 0) + wr1*h_table(iv, ir + 1, 0);
    }
    if (log_temp >= eos_data.saha_logtemp_max) {
      return wr0*h_table(iv, ir, ntemp - 1) + wr1*h_table(iv, ir + 1, ntemp - 1);
    }
    const Real x = (log_temp - eos_data.saha_logtemp_min)*eos_data.saha_inv_dlogtemp;
    int it = static_cast<int>(x);
    it = std::max(0, std::min(it, ntemp - 2));
    const Real wt1 = x - static_cast<Real>(it);
    const Real wt0 = 1.0 - wt1;
    return wr0*(wt0*h_table(iv, ir, it) + wt1*h_table(iv, ir, it + 1))
         + wr1*(wt0*h_table(iv, ir + 1, it) + wt1*h_table(iv, ir + 1, it + 1));
  };
  auto floor_temp_from_monotonic = [&](const int iv, const Real log_rho,
                                       const Real target) {
    const Real v0 = floor_field_at_temp_index(iv, log_rho, 0);
    const Real vn = floor_field_at_temp_index(iv, log_rho, ntemp - 1);
    if (target <= v0) return std::exp(h_logtemp(0))/eos_data.temp_unit_cgs;
    if (target >= vn) return std::exp(h_logtemp(ntemp - 1))/eos_data.temp_unit_cgs;
    int ilo = 0;
    int ihi = ntemp - 1;
    Real vlo = v0;
    Real vhi = vn;
    while (ihi - ilo > 1) {
      const int imid = ilo + (ihi - ilo)/2;
      const Real vmid = floor_field_at_temp_index(iv, log_rho, imid);
      if (target <= vmid) {
        ihi = imid;
        vhi = vmid;
      } else {
        ilo = imid;
        vlo = vmid;
      }
    }
    const Real log_t_lo = h_logtemp(ilo);
    const Real log_t_hi = h_logtemp(ihi);
    const Real denom = vhi - vlo;
    const Real frac = (std::abs(denom) > 0.0) ? ((target - vlo)/denom) : 0.0;
    return std::exp(log_t_lo + frac*(log_t_hi - log_t_lo))/eos_data.temp_unit_cgs;
  };
  const Real dlogeps =
      (eos_data.saha_logeps_max - eos_data.saha_logeps_min)/
      static_cast<Real>(eos_data.saha_neps - 1);
  const Real cs2_ceil = (eos_data.cs_ceil > 0.0) ? SQR(eos_data.cs_ceil) : -1.0;
  const Real tfloor_code = std::max(eos_data.saha_tmin_code, eos_data.tfloor);
  const Real log_p_floor = (eos_data.pfloor > 0.0)
      ? std::log(std::max(eos_data.pfloor*eos_data.pressure_unit_cgs,
                          static_cast<Real>(1.0e-99)))
      : 0.0;
  const Real floor_dlogrho =
      (eos_data.saha_floor_logrho_max - eos_data.saha_floor_logrho_min)/
      static_cast<Real>(eos_data.saha_floor_nrho - 1);
  for (int ir = 0; ir < eos_data.saha_floor_nrho; ++ir) {
    const Real log_rho = eos_data.saha_floor_logrho_min + ir*floor_dlogrho;
    Real tfloor_code_local = tfloor_code;
    if (eos_data.pfloor > 0.0) {
      const Real t_from_p =
          floor_temp_from_monotonic(EOS_Data::saha_logpress, log_rho, log_p_floor);
      tfloor_code_local = std::max(tfloor_code_local, t_from_p);
    }
    const Real log_temp =
        std::log(std::max(tfloor_code_local*eos_data.temp_unit_cgs,
                          static_cast<Real>(1.0e-99)));
    h_logeps_floor(ir) = floor_eval_field(EOS_Data::saha_logeps, log_rho, log_temp);
  }
  for (int ir = 0; ir < nrho; ++ir) {
    const Real row_logeps_min = h_table(EOS_Data::saha_logeps, ir, 0);
    const Real row_logeps_max = h_table(EOS_Data::saha_logeps, ir, ntemp - 1);
    const double *logeps_row = table["logeps"] + ir*ntemp;
    const double *logpress_row = table["logpress"] + ir*ntemp;
    const double *logcs2_row = table["logcs2"] + ir*ntemp;
    const double *gamma1_row = table["gamma1"] + ir*ntemp;
    const double *gamma3m1_row = table["gamma3m1"] + ir*ntemp;
    const double *xh2_row = table.HasField("xh2") ? table["xh2"] + ir*ntemp : nullptr;
    const double *xion_row = table["xion"] + ir*ntemp;
    const double *xhe1_row = table["xhe1"] + ir*ntemp;
    const double *xhe2_row = table["xhe2"] + ir*ntemp;
    const double *mu_row = table["mu"] + ir*ntemp;
    const double *beta_rad_row = table["beta_rad"] + ir*ntemp;
    for (int ie = 0; ie < eos_data.saha_neps; ++ie) {
      const Real log_eps_target = eos_data.saha_logeps_min + ie*dlogeps;
      Real log_temp = static_cast<Real>(table["logtemp"][0]);
      if (log_eps_target <= row_logeps_min) {
        log_temp = static_cast<Real>(table["logtemp"][0]);
      } else if (log_eps_target >= row_logeps_max) {
        log_temp = ExtrapolateLogTempAboveRow(table["logtemp"], logeps_row, ntemp,
                                              log_eps_target);
      } else {
        log_temp = InvertMonotonicFieldAtRow(table["logtemp"], logeps_row, ntemp,
                                             log_eps_target);
      }
      h_cache(EOS_Data::saha_cache_temp, ir, ie) =
          std::exp(log_temp)/eos_data.temp_unit_cgs;
      h_cache(EOS_Data::saha_cache_press, ir, ie) =
          std::exp(ExtrapolateLogFieldAtLogTemp(table["logtemp"], logpress_row, ntemp,
                                                log_temp))/
          eos_data.pressure_unit_cgs;
      h_cache(EOS_Data::saha_cache_cs2, ir, ie) =
          std::exp(ExtrapolateLogFieldAtLogTemp(table["logtemp"], logcs2_row, ntemp,
                                                log_temp))/
          eos_data.specific_eint_unit_cgs;
      h_cache(EOS_Data::saha_cache_gamma1, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], gamma1_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_gamma3m1, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], gamma3m1_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_xh2, ir, ie) =
          (xh2_row != nullptr) ? EvalFieldAtLogTemp(table["logtemp"], xh2_row, ntemp,
              log_temp)
                               : 0.0;
      h_cache(EOS_Data::saha_cache_xion, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], xion_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_xhe1, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], xhe1_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_xhe2, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], xhe2_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_mu, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], mu_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_beta_rad, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], beta_rad_row, ntemp, log_temp);
    }

    const Real row_logeps_top = (eos_data.saha_bounds_mode == SahaBoundsMode::clamp)
        ? eos_data.saha_logeps_max : row_logeps_max;
    Real row_logeps_ceil = row_logeps_top;
    if (cs2_ceil > 0.0) {
      const Real cs2_first = h_cache(EOS_Data::saha_cache_cs2, ir, 0);
      if (cs2_first > cs2_ceil) {
        row_logeps_ceil = eos_data.saha_logeps_min;
      } else {
        for (int ie = 1; ie < eos_data.saha_neps; ++ie) {
          const Real cs2_prev = h_cache(EOS_Data::saha_cache_cs2, ir, ie - 1);
          const Real cs2_curr = h_cache(EOS_Data::saha_cache_cs2, ir, ie);
          if (cs2_curr > cs2_ceil) {
            const Real logeps_prev = eos_data.saha_logeps_min + (ie - 1)*dlogeps;
            const Real logeps_curr = logeps_prev + dlogeps;
            const Real denom = cs2_curr - cs2_prev;
            const Real frac = (fabs(denom) > 0.0) ? ((cs2_ceil - cs2_prev)/denom) : 0.0;
            row_logeps_ceil = logeps_prev + frac*(logeps_curr - logeps_prev);
            break;
          }
        }
      }
    }
    h_logeps_ceil(ir) = fmin(fmax(row_logeps_ceil, row_logeps_min), row_logeps_top);
  }

  Kokkos::deep_copy(eos_data.saha_thermo_cache, h_cache);
  Kokkos::deep_copy(eos_data.saha_logeps_floor, h_logeps_floor);
  Kokkos::deep_copy(eos_data.saha_logeps_ceil, h_logeps_ceil);
}

inline void PrintStartupSummary(const std::string &block, ParameterInput *pin,
                                const EOS_Data &eos, const bool use_dual_energy) {
  if (global_variable::my_rank != 0 || eos.hydro_eos != HydroEOSModel::lte_table) {
    return;
  }
  const std::string current_eos = pin->GetString(block, "eos");
  const std::string current_table =
      pin->DoesParameterExist(block, "table") ? pin->GetString(block,
          "table") : "unknown";
  const std::string current_table_type = ExpectedTableTypeForEosName(current_eos);
  const std::string restart_eos = GetRestartEOS(pin, block);
  const std::string restart_table = GetRestartTable(pin, block);
  const std::string restart_table_type = GetRestartTableType(pin, block);
  const Real restart_x = GetRestartX(pin, block);
  const Real restart_y = GetRestartY(pin, block);
  const Real restart_prad = GetRestartPrad(pin, block);
  const Real restart_h2 = GetRestartH2(pin, block);
  const Real restart_zpe = GetRestartZPE(pin, block);
  const bool has_restart_meta = (restart_table != "unknown") || (restart_table_type != "unknown") ||
                                (restart_x >= 0.0) || (restart_y >= 0.0) ||
                                (restart_prad >= 0.0) || (restart_h2 >= 0.0) ||
                                (restart_zpe >= 0.0);
  const bool explicit_mismatch =
      (restart_eos != "unknown" && restart_eos != current_eos) ||
      (restart_table != "unknown" && restart_table != current_table) ||
      (restart_table_type != "unknown" && restart_table_type != current_table_type) ||
      (restart_x >= 0.0 && fabs(restart_x - eos.lte_h_mass_fraction) > 1.0e-12) ||
      (restart_y >= 0.0 && fabs(restart_y - eos.lte_he_mass_fraction) > 1.0e-12) ||
      (restart_prad >= 0.0 &&
       fabs(restart_prad - (eos.lte_has_radiation ? 1.0 : 0.0)) > 1.0e-12) ||
      (restart_h2 >= 0.0 &&
       fabs(restart_h2 - (eos.lte_has_h2 ? 1.0 : 0.0)) > 1.0e-12) ||
      (restart_zpe >= 0.0 &&
       fabs(restart_zpe - (eos.lte_zpe_subtracted ? 1.0 : 0.0)) > 1.0e-12);
  const bool allow_reinterpret = AllowReinterpretiveRestart(pin, block);
  const std::string restart_mode = RestartIsActive(pin) ?
      (explicit_mismatch ? (allow_reinterpret ? "reinterpretive-allowed" : "reinterpretive") :
       (has_restart_meta ? "same-eos" : "same-eos-metadata-unknown")) :
      "new-run";

  if (RestartIsActive(pin) && explicit_mismatch && !allow_reinterpret) {
    Fatal(__FILE__,
        "Restart compatibility check failed for <" + block + "> with <" + block +
                    ">/eos = " + current_eos + ". Restart metadata indicates a different "
                    "EOS/table configuration. Set <" + block +
                    ">/allow_reinterpretive_restart = true to override this safety check.");
  }

  std::cout << "LTE table EOS startup: block=" << block
            << " table=" << pin->GetString(block, "table")
            << " X=" << eos.lte_h_mass_fraction
            << " Y=" << eos.lte_he_mass_fraction
            << " h2=" << OnOff(eos.lte_has_h2)
            << " prad=" << OnOff(eos.lte_has_radiation)
            << " zpe_subtracted=" << OnOff(eos.lte_zpe_subtracted)
            << " dual_energy=" << OnOff(use_dual_energy)
            << " cache_neps=" << eos.saha_neps
            << " cs_ceil=" << eos.cs_ceil
            << " bounds=" << BoundsModeName(eos.saha_bounds_mode)
            << " debug_checks=" << OnOff(eos.saha_debug_checks)
            << " restart=" << restart_mode << std::endl;
  if (restart_mode == "reinterpretive-allowed") {
    std::cout << "WARNING: restart is reinterpretive for <" << block << ">: old EOS='"
              << restart_eos << "', old table='" << restart_table << "', new EOS='"
              << current_eos << "', new table='" << current_table
              << "'. Conserved state is resumed, but thermodynamics are not identical "
              << "to the original run." << std::endl;
  } else if (restart_mode == "same-eos-metadata-unknown") {
    std::cout << "WARNING: restart compatibility metadata is incomplete for <" << block
              << ">. EOS names match, but table provenance could not be fully checked."
              << std::endl;
  }
}

}  // namespace lte_table_utils

#endif  // EOS_LTE_TABLE_UTILS_HPP_
