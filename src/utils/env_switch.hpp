#ifndef UTILS_ENV_SWITCH_HPP_
#define UTILS_ENV_SWITCH_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file env_switch.hpp
//! \brief Rank-consistent environment switches.
//!
//! Every ATHENAK_* / ATHENA_* environment kill switch is taken from MPI rank 0's
//! environment and broadcast once, at startup (env_switch::Sync(), called from main()
//! right after MPI is up).  Readers then consult the broadcast table, never their own
//! process environment.  A switch that differs between ranks -- `mpirun -x` on one host
//! only, a stale shell on one node -- would otherwise put the ranks on different task
//! graphs or packing layouts, and the failure mode of that is a hang or a silently
//! corrupted exchange with no diagnostic.  Every remaining switch routes through here.
//! The A/B kill switches of the 2026-08 GPU-overlap campaign are gone: their measured
//! paths are compiled in, so what is left is the multigrid poll budget and the launch
//! auto-tuner pin -- both numeric overrides of an automatic choice, not path selectors.
//!
//! Lookups are NOT collective (plain table reads), so they are safe inside lazily
//! initialised function-local statics on any subset of ranks.  Before Sync() runs, or in
//! a non-MPI build, the lookup falls back to the local environment.
//!
//! Usage:  static const bool on = env_switch::Bool("ATHENAK_FOO", true);

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <iostream>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

extern char **environ;

namespace env_switch {

//! \brief Prefixes of the variables that are treated as run switches and synchronised.
inline bool IsSwitchName(const char *name) {
  return (std::strncmp(name, "ATHENAK_", 8) == 0) || (std::strncmp(name, "ATHENA_",
      7) == 0);
}

//! \brief The broadcast table.  Empty (and `synced == false`) until Sync() has run.
struct Table {
  std::map<std::string, std::string> values;
  bool synced = false;
};

inline Table &Global() {
  static Table table;
  return table;
}

//! \brief Serialise this process's ATHENAK_*/ATHENA_* environment as "NAME=VALUE\n..."
inline std::string LocalSwitches() {
  std::map<std::string, std::string> local;   // map: deterministic order for hashing
  for (char **e = environ; e != nullptr && *e != nullptr; ++e) {
    const char *eq = std::strchr(*e, '=');
    if (eq == nullptr) continue;
    std::string name(*e, static_cast<size_t>(eq - *e));
    if (IsSwitchName(name.c_str())) local[name] = std::string(eq + 1);
  }
  std::string out;
  for (const auto &kv : local) out += kv.first + "=" + kv.second + "\n";
  return out;
}

//! \brief Collective: broadcast rank 0's switch table and warn (once, from rank 0) if any
//! other rank's own environment disagreed with it.  Call after MPI_Init, before anything
//! reads a switch.  Safe to call in a non-MPI build (then it just snapshots the process
//! environment).
inline void Sync() {
  Table &table = Global();
  std::string blob = LocalSwitches();
  int ndiffer = 0;
#if MPI_PARALLEL_ENABLED
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::string mine = blob;
    int len = static_cast<int>(blob.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    blob.resize(static_cast<size_t>(len));
    if (len > 0) MPI_Bcast(&blob[0], len, MPI_CHAR, 0, MPI_COMM_WORLD);
    int differs = (mine != blob) ? 1 : 0;
    MPI_Reduce(&differs, &ndiffer, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank != 0) ndiffer = 0;
  }
#endif
  table.values.clear();
  size_t pos = 0;
  while (pos < blob.size()) {
    size_t nl = blob.find('\n', pos);
    if (nl == std::string::npos) nl = blob.size();
    const std::string line = blob.substr(pos, nl - pos);
    const size_t eq = line.find('=');
    if (eq != std::string::npos) table.values[line.substr(0, eq)] = line.substr(eq + 1);
    pos = nl + 1;
  }
  table.synced = true;
  if (ndiffer > 0) {
    std::cout << "### WARNING in env_switch::Sync: " << ndiffer << " rank(s) carry a different"
              << " ATHENAK_*/ATHENA_* environment than rank 0; rank 0's values are used on"
              << " every rank (set switches where rank 0 is launched, or use mpirun -x)."
              << std::endl;
  }
}

//! \brief Raw value.  Returns nullptr-equivalent (is_set=false) when unset on rank 0.
inline std::string Raw(const char *name, bool *is_set) {
  const Table &table = Global();
  if (table.synced) {
    auto it = table.values.find(name);
    if (is_set != nullptr) *is_set = (it != table.values.end());
    return (it != table.values.end()) ? it->second : std::string();
  }
  const char *s = std::getenv(name);
  if (is_set != nullptr) *is_set = (s != nullptr);
  return (s != nullptr) ? std::string(s) : std::string();
}

//! \brief Boolean switch.  Unset -> def.  "0", "false", "no", "off", "" -> false;
//! anything else -> true.  (The previous per-site idioms disagreed on the empty string;
//! empty now reads as false everywhere, i.e. VAR= disables like VAR=0.)
inline bool Bool(const char *name, bool def) {
  bool set = false;
  std::string v = Raw(name, &set);
  if (!set) return def;
  for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (v.empty() || v == "0" || v == "false" || v == "no" || v == "off") return false;
  return true;
}

//! \brief Integer switch.  Unset or unparsable -> def.
inline long Int(const char *name, long def) {
  bool set = false;
  std::string v = Raw(name, &set);
  if (!set || v.empty()) return def;
  char *end = nullptr;
  const long val = std::strtol(v.c_str(), &end, 10);
  return (end != nullptr && *end == '\0') ? val : def;
}

}  // namespace env_switch

#endif  // UTILS_ENV_SWITCH_HPP_
