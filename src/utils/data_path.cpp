//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file data_path.cpp
//! \brief ResolveDataPath (data_path.hpp)

#include "utils/data_path.hpp"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// The source tree, from the build system (src/CMakeLists.txt), for its data/ directory.
#ifndef ATHENAK_SOURCE_DIR
#define ATHENAK_SOURCE_DIR ""
#endif

namespace {

bool Readable(const std::string &path) {
  std::ifstream probe(path.c_str());
  return probe.good();
}

std::string Join(const std::string &dir, const std::string &name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

}  // namespace

std::string ResolveDataPath(const std::string &name, std::string *searched) {
  if (name.empty() || name[0] == '/') return name;
  std::vector<std::string> candidates = {name};
  const char *env = std::getenv("ATHENAK_DATA");
  if (env != nullptr && env[0] != '\0') candidates.push_back(Join(env, name));
  const std::string source_dir = ATHENAK_SOURCE_DIR;
  if (!source_dir.empty()) candidates.push_back(Join(Join(source_dir, "data"), name));
  for (const std::string &path : candidates) {
    if (Readable(path)) return path;
  }
  if (searched != nullptr) {
    searched->clear();
    for (const std::string &path : candidates) {
      *searched += (searched->empty() ? "" : ", ") + ("'" + path + "'");
    }
  }
  return name;
}
