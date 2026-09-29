#ifndef EOS_SAHA_TABLE_UTILS_HPP_
#define EOS_SAHA_TABLE_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file saha_table_utils.hpp
//! \brief Shared setup, QA, and startup diagnostics for the nonrel H-only LTE/Saha EOS.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "utils/tr_table.hpp"

namespace saha_table_utils {

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
  Require(npts >= 2, file, "Saha table axis '" + name + "' must have at least 2 points.");
  const Real dx = static_cast<Real>(values[1] - values[0]);
  Require(dx > 0.0, file, "Saha table axis '" + name + "' must be strictly increasing.");
  const Real tol = 1.0e-10*fmax(static_cast<Real>(1.0), fabs(dx));
  for (int i = 1; i < (npts - 1); ++i) {
    const Real ddi = static_cast<Real>(values[i + 1] - values[i]);
    Require(fabs(ddi - dx) <= tol, file,
            "Saha table axis '" + name + "' must be uniformly spaced.");
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

inline void RunTableQAChecks(const std::string &fname, TableReader::Table &table,
                             const int nrho, const int ntemp, const bool debug_checks,
                             const std::string &file) {
  const double *logrho = table["logrho"];
  const double *logtemp = table["logtemp"];
  const double *logpress = table["logpress"];
  const double *logeps = table["logeps"];
  const double *logcs2 = table["logcs2"];
  const double *gamma1 = table["gamma1"];
  const double *gamma3m1 = table["gamma3m1"];
  const double *xion = table["xion"];
  const Real rel_tol = 5.0e-6;

  for (int ir = 0; ir < nrho; ++ir) {
    const int row = ir*ntemp;
    for (int it = 0; it < ntemp; ++it) {
      const int idx = row + it;
      const Real p = exp(static_cast<Real>(logpress[idx]));
      const Real eps = exp(static_cast<Real>(logeps[idx]));
      const Real cs2 = exp(static_cast<Real>(logcs2[idx]));
      const Real g1 = static_cast<Real>(gamma1[idx]);
      const Real x = static_cast<Real>(xion[idx]);
      Require(std::isfinite(logpress[idx]) && std::isfinite(logeps[idx]) &&
                  std::isfinite(logcs2[idx]) && std::isfinite(gamma1[idx]) &&
                  std::isfinite(gamma3m1[idx]) && std::isfinite(xion[idx]),
              file, "Non-finite value found while validating Saha table '" + fname + "'.");
      Require((p > 0.0) && (eps > 0.0) && (cs2 > 0.0) && (g1 > 0.0), file,
              "Non-positive thermodynamic value found while validating Saha table '" +
                  fname + "'.");
      Require((x >= 0.0) && (x <= 1.0), file,
              "Ionization fraction outside [0,1] in Saha table '" + fname + "'.");
      if (it > 0) {
        Require(logeps[idx] > logeps[idx - 1], file,
                "Saha table eps(rho,T) must increase monotonically with T at fixed rho.");
      }
      if (debug_checks) {
        const Real rho = exp(static_cast<Real>(logrho[ir]));
        const Real g1_cs = rho*cs2/p;
        const Real err = fabs(g1_cs - g1)/fmax(static_cast<Real>(1.0), fabs(g1));
        Require(err < rel_tol, file,
                "Gamma1 != rho*cs2/p consistency failure in Saha table '" + fname + "'.");
      }
    }
    if (debug_checks) {
      for (int it = 0; it < ntemp; ++it) {
        const int idx = row + it;
        const Real logt_rt = InvertMonotonicFieldAtRow(logtemp, &logeps[row], ntemp,
                                                       static_cast<Real>(logeps[idx]));
        const Real err = fabs(logt_rt - static_cast<Real>(logtemp[it]))/
                         fmax(static_cast<Real>(1.0),
                             fabs(static_cast<Real>(logtemp[it])));
        Require(err < 5.0e-6, file,
                "Saha table logeps inversion round-trip failed for '" + fname + "'.");
      }
    }
  }
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

inline bool AllowReinterpretiveRestart(ParameterInput *pin, const std::string &block) {
  return pin->GetOrAddBoolean(block, "allow_reinterpretive_restart", false);
}

}  // namespace

inline void InitializeSahaTableEOS(const std::string &block, MeshBlockPack *pp,
                                   ParameterInput *pin, EOS_Data &eos_data,
                                   const std::string &file) {
  if (pp->pcoord->is_special_relativistic || pp->pcoord->is_general_relativistic) {
    Fatal(file, "<" + block + ">/eos = saha_table is only implemented for "
                "non-relativistic " + block + ".");
  }
  if (pp->punit == nullptr) {
    Fatal(file,
        "<" + block + ">/eos = saha_table requires a [units] block for code-to-cgs "
                "scaling.");
  }
  if (eos_data.sfloor > 0.0) {
    Fatal(file, "<" + block + ">/sfloor is not supported with <" + block +
                    ">/eos = saha_table.");
  }

  eos_data.hydro_eos = HydroEOSModel::saha_table;
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = false;
  eos_data.gamma = 5.0/3.0;
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;
  eos_data.use_t = false;
  eos_data.lte_h_mass_fraction = 1.0;
  eos_data.lte_he_mass_fraction = 0.0;
  eos_data.lte_has_helium = false;
  eos_data.lte_has_radiation = false;

  const std::string bounds_name = pin->GetOrAddString(block, "saha_bounds", "error");
  if (bounds_name == "error") {
    eos_data.saha_bounds_mode = SahaBoundsMode::error;
  } else if (bounds_name == "clamp") {
    eos_data.saha_bounds_mode = SahaBoundsMode::clamp;
  } else {
    Fatal(file, "<" + block + ">/saha_bounds must be either 'error' or 'clamp'.");
  }
  eos_data.saha_debug_checks = pin->GetOrAddBoolean(block, "saha_debug_checks", false);

  TableReader::Table table;
  const std::string fname = pin->GetString(block, "table");
  const auto result = table.ReadTable(fname);
  if (result.error != TableReader::ReadResult::SUCCESS) {
    Fatal(file, "Failed to read saha_table '" + fname + "':\n" + result.message);
  }

  const auto metadata = table.GetMetadata();
  auto find_meta = [&](const std::string &key) -> std::string {
    const auto it = metadata.find(key);
    return (it == metadata.end()) ? std::string() : it->second;
  };
  const std::string table_type = find_meta("table_type");
  const std::string log_axis_base = find_meta("log_axis_base");
  Require(log_axis_base.empty() || log_axis_base == "e", file,
          "Saha table metadata 'log_axis_base' must be 'e'.");
  if (!table_type.empty()) {
    Require(table_type == "saha_hydrogen_lte", file,
            "Saha table metadata 'table_type' must be 'saha_hydrogen_lte'.");
  } else {
    Require(!(table.HasField("xhe1") || table.HasField("xhe2") ||
              table.HasField("mu") || table.HasField("beta_rad")),
            file, "saha_table cannot read a richer H+He table without explicit "
                  "'table_type = saha_hydrogen_lte' metadata.");
  }

  const auto point_info = table.GetPointInfo();
  Require(point_info.size() == 2, file,
          "Saha table must be 2D with axes (logrho, logtemp).");
  Require(point_info[0].first == "logrho" && point_info[1].first == "logtemp", file,
          "Saha table axes must be ordered as logrho, logtemp.");
  const int nrho = static_cast<int>(point_info[0].second);
  const int ntemp = static_cast<int>(point_info[1].second);
  Require(table.HasField("logpress") && table.HasField("logeps") &&
              table.HasField("logcs2") && table.HasField("gamma1") &&
              table.HasField("gamma3m1") && table.HasField("xion"),
          file, "Saha table is missing one or more required fields: "
                "logpress, logeps, logcs2, gamma1, gamma3m1, xion.");

  RunTableQAChecks(fname, table, nrho, ntemp, eos_data.saha_debug_checks, file);

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

  eos_data.saha_logrho = DvceArray1D<Real>("saha_logrho", nrho);
  eos_data.saha_logtemp = DvceArray1D<Real>("saha_logtemp", ntemp);
  eos_data.saha_table = DvceArray3D<Real>("saha_table", EOS_Data::saha_nvars, nrho,
      ntemp);
  eos_data.saha_logrho_h = HostArray1D<Real>("saha_logrho_h", nrho);
  eos_data.saha_logtemp_h = HostArray1D<Real>("saha_logtemp_h", ntemp);
  eos_data.saha_table_h = HostArray3D<Real>("saha_table_h", EOS_Data::saha_nvars, nrho,
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
      {"xion", EOS_Data::saha_xion},
  };
  for (const auto &field : fields) {
    const double *src = table[field.first];
    for (int ir = 0; ir < nrho; ++ir) {
      for (int it = 0; it < ntemp; ++it) {
        h_table(field.second, ir, it) = static_cast<Real>(src[ir*ntemp + it]);
      }
    }
  }
  for (int ir = 0; ir < nrho; ++ir) {
    for (int it = 0; it < ntemp; ++it) {
      const Real xion = h_table(EOS_Data::saha_xion, ir, it);
      h_table(EOS_Data::saha_xh2, ir, it) = 0.0;
      h_table(EOS_Data::saha_xhe1, ir, it) = 0.0;
      h_table(EOS_Data::saha_xhe2, ir, it) = 0.0;
      h_table(EOS_Data::saha_mu, ir, it) = 1.0/fmax(static_cast<Real>(1.0) + xion,
                                                    static_cast<Real>(1.0e-30));
      h_table(EOS_Data::saha_beta_rad, ir, it) = 0.0;
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

  const int cache_eps_factor = pin->GetOrAddInteger(block, "saha_cache_eps_factor", 8);
  Require(cache_eps_factor >= 1, file,
          "<" + block + ">/saha_cache_eps_factor must be >= 1.");
  const int floor_cache_factor = pin->GetOrAddInteger(block, "saha_floor_cache_factor",
      16);
  Require(floor_cache_factor >= 1, file,
          "<" + block + ">/saha_floor_cache_factor must be >= 1.");
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
          "Saha inverse thermo cache must have at least 2 logeps points.");
  Require(eos_data.saha_logeps_max > eos_data.saha_logeps_min, file,
          "Saha inverse thermo cache has a degenerate logeps axis.");
  eos_data.saha_inv_dlogeps =
      static_cast<Real>(eos_data.saha_neps - 1)/
      (eos_data.saha_logeps_max - eos_data.saha_logeps_min);

  if (pin->DoesParameterExist(block, "tfloor_kelvin")) {
    eos_data.tfloor = pin->GetReal(block, "tfloor_kelvin")/eos_data.temp_unit_cgs;
  }

  eos_data.saha_thermo_cache =
      DvceArray3D<Real>("saha_thermo_cache", EOS_Data::saha_cache_nvars, nrho,
                        eos_data.saha_neps);
  eos_data.saha_thermo_cache_h =
      HostArray3D<Real>("saha_thermo_cache_h", EOS_Data::saha_cache_nvars, nrho,
                        eos_data.saha_neps);
  eos_data.saha_logeps_floor =
      DvceArray1D<Real>("saha_logeps_floor", eos_data.saha_floor_nrho);
  eos_data.saha_logeps_floor_h =
      HostArray1D<Real>("saha_logeps_floor_h", eos_data.saha_floor_nrho);
  eos_data.saha_logeps_ceil = DvceArray1D<Real>("saha_logeps_ceil", nrho);
  eos_data.saha_logeps_ceil_h = HostArray1D<Real>("saha_logeps_ceil_h", nrho);

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
    const double *xion_row = table["xion"] + ir*ntemp;
    Real floor_logtemp = std::log(std::max(tfloor_code*eos_data.temp_unit_cgs,
                                           static_cast<Real>(1.0e-99)));
    if (eos_data.pfloor > 0.0) {
      if (log_p_floor <= logpress_row[0]) {
        floor_logtemp = std::max(floor_logtemp, static_cast<Real>(table["logtemp"][0]));
      } else if (log_p_floor >= logpress_row[ntemp - 1]) {
        floor_logtemp = std::max(floor_logtemp,
                                 static_cast<Real>(table["logtemp"][ntemp - 1]));
      } else {
        floor_logtemp = std::max(
            floor_logtemp,
            InvertMonotonicFieldAtRow(table["logtemp"], logpress_row, ntemp,
                log_p_floor));
      }
    }
    h_logeps_floor(ir) =
        EvalFieldAtLogTemp(table["logtemp"], logeps_row, ntemp, floor_logtemp);
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
      h_cache(EOS_Data::saha_cache_xh2, ir, ie) = 0.0;
      h_cache(EOS_Data::saha_cache_xion, ir, ie) =
          EvalFieldAtLogTemp(table["logtemp"], xion_row, ntemp, log_temp);
      h_cache(EOS_Data::saha_cache_xhe1, ir, ie) = 0.0;
      h_cache(EOS_Data::saha_cache_xhe2, ir, ie) = 0.0;
      h_cache(EOS_Data::saha_cache_mu, ir, ie) =
          1.0/fmax(static_cast<Real>(1.0) + h_cache(EOS_Data::saha_cache_xion, ir, ie),
                   static_cast<Real>(1.0e-30));
      h_cache(EOS_Data::saha_cache_beta_rad, ir, ie) = 0.0;
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
  if (global_variable::my_rank != 0 || eos.hydro_eos != HydroEOSModel::saha_table) {
    return;
  }
  const std::string current_table =
      pin->DoesParameterExist(block, "table") ? pin->GetString(block,
          "table") : "unknown";
  const std::string restart_eos = GetRestartEOS(pin, block);
  const std::string restart_table = GetRestartTable(pin, block);
  const std::string restart_table_type = GetRestartTableType(pin, block);
  const Real restart_x = GetRestartX(pin, block);
  const Real restart_y = GetRestartY(pin, block);
  const Real restart_prad = GetRestartPrad(pin, block);
  const bool has_restart_meta = (restart_table != "unknown") || (restart_table_type != "unknown") ||
                                (restart_x >= 0.0) || (restart_y >= 0.0) ||
                                (restart_prad >= 0.0);
  const bool explicit_mismatch =
      (restart_eos != "unknown" && restart_eos != "saha_table") ||
      (restart_table != "unknown" && restart_table != current_table) ||
      (restart_table_type != "unknown" && restart_table_type != "saha_hydrogen_lte") ||
      (restart_x >= 0.0 && fabs(restart_x - 1.0) > 1.0e-12) ||
      (restart_y >= 0.0 && fabs(restart_y) > 1.0e-12) ||
      (restart_prad >= 0.0 && fabs(restart_prad) > 1.0e-12);
  const bool allow_reinterpret = AllowReinterpretiveRestart(pin, block);
  const std::string restart_mode = RestartIsActive(pin) ?
      (explicit_mismatch ? (allow_reinterpret ? "reinterpretive-allowed" : "reinterpretive") :
       (has_restart_meta ? "same-eos" : "same-eos-metadata-unknown")) :
      "new-run";

  if (RestartIsActive(pin) && explicit_mismatch && !allow_reinterpret) {
    Fatal(__FILE__, "Restart compatibility check failed for <" + block + "> with "
                    "<" + block + ">/eos = saha_table. Restart metadata indicates "
                    "a different EOS/table configuration. Set <" + block +
                    ">/allow_reinterpretive_restart = true to override this safety check.");
  }

  std::cout << "Saha EOS startup: block=" << block
            << " table=" << pin->GetString(block, "table")
            << " dual_energy=" << OnOff(use_dual_energy)
            << " cache_neps=" << eos.saha_neps
            << " cs_ceil=" << eos.cs_ceil
            << " bounds=" << BoundsModeName(eos.saha_bounds_mode)
            << " debug_checks=" << OnOff(eos.saha_debug_checks)
            << " restart=" << restart_mode << std::endl;
  if (restart_mode == "reinterpretive-allowed") {
    std::cout << "WARNING: restart is reinterpretive for <" << block << ">: old EOS='"
              << restart_eos << "', old table='" << restart_table
              << "', new EOS='saha_table', new table='" << current_table
              << "'. Conserved state is resumed, but thermodynamics are not identical "
              << "to the original run."
              << std::endl;
  } else if (restart_mode == "same-eos-metadata-unknown") {
    std::cout << "WARNING: restart compatibility metadata is incomplete for <" << block
              << ">. EOS names match, but table provenance could not be fully checked."
              << std::endl;
  }
}

}  // namespace saha_table_utils

#endif  // EOS_SAHA_TABLE_UTILS_HPP_
