//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parameter_input.cpp
//  \brief implementation of functions in class ParameterInput
//
// PURPOSE: Member functions of this class are used to read and parse the input file.
//   Functionality is loosely modeled after FORTRAN namelist.
//
// EXAMPLE of input file in 'Athena++' format:
//   <blockname1>      # block name; must be on a line by itself
//                     # everything after a hash symbol is a comment and is ignored
//   name1=value       # each parameter name must be on a line by itself
//   name2 = value1    # whitespace around the = is optional
//                     # blank lines are OK
//   # my comment here   comment lines are OK
//   # name3 = value3    values (and blocks) that are commented out are ignored
//
//   <blockname2>      # start new block
//   name1 = value1    # note that same parameter names can appear in different blocks
//   name2 = value2    # empty lines (like following) are OK
//
//   <blockname1>      # same blockname can re-appear, although NOT recommended
//   name3 = value3    # this would be the 3rd parameter name in blockname1
//   name1 = value4    # if parameter name is repeated, previous value is overwritten!
//
// LIMITATIONS:
//   - parameter specification (name=val #comment) must all be on a single line
//
// HISTORY:
//   - Nov 2002:  Created for Athena1.0/Cambridge release by Peter Teuben
//   - 2003-2008: Many improvements and extensions by T. Gardiner and J.M. Stone
//   - Jan 2014:  Rewritten in C++ for the Athena++ code by J.M. Stone
//========================================================================================

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

namespace {

//! Significant digits used when a Real is written back into the parameter table.  Those
//! strings are re-read verbatim as inputs on the next restart (last_time, ceilings, any
//! pgen runtime state), and the stream default of 6 was silently truncating them: a
//! checkpoint's <outputN>/last_time came back 0.016 away from the value that produced it.
//! max_digits10 is the shortest count that round-trips every Real exactly.
constexpr int kRealDumpDigits = std::numeric_limits<Real>::max_digits10;

struct TimeParameterAlias {
  const char *canonical;
  const char *legacy;
};

constexpr TimeParameterAlias time_parameter_aliases[] = {
  {"lat", "hydro_lat"},
  {"lat_levels", "hydro_lat_levels"},
  {"lat_same_level", "hydro_lat_same_level"},
  {"lat_same_level_max_ratio", "hydro_lat_same_level_max_ratio"},
  {"lat_neighbor_limiter", "hydro_lat_neighbor_limiter"},
  {"lat_diagnostics", "hydro_lat_diagnostics"},
  {"lat_union_stage1", "hydro_lat_union_stage1"},
  // The four parameters below are read under their hydro_lat_* names; map the short
  // lat_* spellings onto those read names so both forms work like every other LAT key.
  {"hydro_lat_min_bin_count", "lat_min_bin_count"},
  {"hydro_lat_gid_reorder", "lat_gid_reorder"},
  {"hydro_lat_post_amr_rebalance", "lat_post_amr_rebalance"},
};

constexpr int ntime_parameter_aliases =
    sizeof(time_parameter_aliases)/sizeof(time_parameter_aliases[0]);

std::string CanonicalTimeParameterName(const std::string &name) {
  for (const auto &alias : time_parameter_aliases) {
    if (name == alias.legacy) return alias.canonical;
  }
  return name;
}

const char *SkipTrailingSpace(const char *p) {
  while (*p != '\0' && std::isspace(static_cast<unsigned char>(*p))) ++p;
  return p;
}

//! \brief Parse an integer-valued parameter, or exit.
//!
//! This replaces a bare atoi(), which stopped at the first character it could not use and
//! returned 0 for input it could not use at all: a mistyped `nlim = 1e6` silently became
//! 1 and `nx1 = sixty-four` silently became 0, with no diagnostic anywhere.  An exactly
//! integral real IS accepted, with a one-time warning, because live decks carry values
//! like `ndiag = 1.0` and refusing those would break a running configuration.  Anything
//! else that starts with an integer is read as that integer with a warning, as upstream's
//! std::stoi does; a value with no leading integer is fatal.
//! Called with the ParameterInput lock held, which is what makes the warn-once set safe.
int ParseIntegerParameterOrExit(const std::string &value, const std::string &block,
                                const std::string &name) {
  const char *str = value.c_str();
  char *end = nullptr;
  errno = 0;
  const long ival = std::strtol(str, &end, 10);
  if (end != str && *SkipTrailingSpace(end) == '\0' && errno != ERANGE &&
      ival >= INT_MIN && ival <= INT_MAX) {
    return static_cast<int>(ival);
  }

  errno = 0;
  char *dend = nullptr;
  const double dval = std::strtod(str, &dend);
  if (dend != str && *SkipTrailingSpace(dend) == '\0' && errno != ERANGE &&
      std::floor(dval) == dval && dval >= static_cast<double>(INT_MIN) &&
      dval <= static_cast<double>(INT_MAX)) {
    static std::set<std::string> warned;
    if (warned.insert(block + "/" + name).second && global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Parameter '" << name << "' in block '" << block << "' is an integer "
                << "parameter but is written as the real '" << value << "'; using "
                << static_cast<int>(dval) << "." << std::endl;
    }
    return static_cast<int>(dval);
  }

  // Upstream AthenaK reads integers with std::stoi, which keeps the leading integer and
  // ignores the rest (an upstream deck's "1e-3" is read as 1).  Accept that prefix with a
  // warning, so upstream input files behave as they do upstream.
  errno = 0;
  end = nullptr;
  const long prefix = std::strtol(str, &end, 10);
  if (end != str && errno != ERANGE && prefix >= INT_MIN && prefix <= INT_MAX) {
    static std::set<std::string> warned_prefix;
    if (warned_prefix.insert(block + "/" + name).second && global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Parameter '" << name << "' in block '" << block << "' is an integer "
                << "parameter but is written as '" << value << "'; using its leading "
                << "integer " << prefix << ", as upstream AthenaK does." << std::endl;
    }
    return static_cast<int>(prefix);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Invalid integer value '" << value << "' for parameter '" << name
            << "' in block '" << block << "'." << std::endl;
  std::exit(EXIT_FAILURE);
}

//! \brief Parse a real-valued parameter, or exit.
//!
//! This replaces a bare atof(), which stopped at the first character it could not use and
//! returned 0 for input it could not use at all: `gamma = 5/3` silently became 5, a quoted
//! `dfloor = '1.0e-6'` silently became 0, and so did `np.float64(0.95)`, the repr of a
//! numpy scalar.  The whole value must be one number (white space around it allowed) that
//! rounds to a finite Real; anything else is fatal.  The test is on the rounded Real, not
//! on the double against numeric_limits<Real>::max(): a float build writes FLT_MAX with
//! max_digits10 as 3.40282347e+38, which strtod reads as a double just above FLT_MAX that
//! still rounds to FLT_MAX, so that value must read back.  An overflow comes back as
//! infinity and fails; an underflow is accepted at the nearest value, since a restart
//! header can hold a subnormal Real it wrote.
Real ParseRealParameterOrExit(const std::string &value, const std::string &block,
                              const std::string &name) {
  const char *str = value.c_str();
  char *end = nullptr;
  const double dval = std::strtod(str, &end);
  const Real rval = static_cast<Real>(dval);
  if (end != str && *SkipTrailingSpace(end) == '\0' && std::isfinite(dval) &&
      std::isfinite(rval)) {
    return rval;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "Invalid real value '" << value << "' for parameter '" << name
            << "' in block '" << block << "'." << std::endl;
  std::exit(EXIT_FAILURE);
}

bool ParseBooleanParameterOrExit(std::string value, const std::string &block,
                                 const std::string &name) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Invalid boolean value '" << value << "' for parameter '"
            << name << "' in block '" << block << "'. Expected true, false, 1, or 0."
            << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
// ParameterInput constructor(s)

ParameterInput::ParameterInput() : last_filename{} {
#if OPENMP_PARALLEL_ENABLED
  omp_init_lock(&lock_);
#endif
}

// this constructor automatically loads data from input_filename in argument
ParameterInput::ParameterInput(std::string input_filename) : last_filename{} {
#ifdef OPENMP_PARALLEL
  omp_init_lock(&lock_);
#endif
  IOWrapper infile;
  infile.Open(input_filename.c_str(), IOWrapper::FileMode::read);
  LoadFromFile(infile);
  infile.Close();
  CheckBlockNames();
}

//----------------------------------------------------------------------------------------
// ParameterInput destructor

ParameterInput::~ParameterInput() {
#if OPENMP_PARALLEL_ENABLED
  omp_destroy_lock(&lock_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn InputLine* InputBlock::GetPtrToLine(std::string name)
//  \brief return pointer to InputLine containing specified parameter if it exists

InputLine* InputBlock::GetPtrToLine(std::string name) {
  for (auto it = line.begin(); it != line.end(); ++it) {
    if (name.compare(it->param_name) == 0) return &*it;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::CheckBlockNames()
//  \brief Checks that all <input_block> names in the input file are valid
//! To add new names, add more strings to the valid_name list

void ParameterInput::CheckBlockNames() {
  // Following are currently recognized <input_block> names. New ones can be added to end
  std::vector<std::string> valid_name = {
    "comment", "job",
    "mesh", "meshblock", "mesh_refinement", "refined_region", "amr_criterion",
    "coord", "adm", "shearing_box",
    "time", "problem", "output", "units", "two_temperature", "eos",
    "hydro", "mhd", "forcefree", "forcefree_restart", "ion-neutral",
    "radiation", "photons", "bns_nurates", "z4c",
    "z4c_amr",
    "cce",
    "rad_srcterms", "hydro_srcterms", "mhd_srcterms", "particles", "turb_driving",
    "fastflow", "saha_runtime", "gravity",
    "orbital_advection", "planets", "turbulence",
    "sink_particles", "remap", "tde_amr"
    };

  for (auto it1 = block.begin(); it1 != block.end(); ++it1) {
    bool found = false;
    for (auto it2 = valid_name.begin(); it2 != valid_name.end(); ++it2) {
      if (it1->block_name.compare(0, it2->length(), (*it2)) == 0) {
        found = true;
        break;
      }
    }
    if (!(found)) {
      if (global_variable::my_rank == 0) {
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<< std::endl;
        std::cout<<"Allowed <input_block> names in input file are:" << std::endl;
        for (auto it2 = valid_name.begin(); it2 != valid_name.end(); ++it2) {
          std::cout<< (*it2) << std::endl;
        }
      }
      Kokkos::abort("Invalid <block_name> in input file");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::GetPtrToBlock(std::string name)
//  \brief return pointer to specified InputBlock if it exists

InputBlock* ParameterInput::GetPtrToBlock(std::string name) {
  for (auto it = block.begin(); it != block.end(); ++it) {
    if (name.compare(it->block_name) == 0) return &*it;
  }
  return nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromStream(std::istream &is)
//  \brief Load input parameters from a stream

// Block names are allocated and stored in a linked list of InputBlocks. Within each
// InputBlock the names, values, and comments of each parameter are allocated and stored
// in a linked list of InputLines.

void ParameterInput::LoadFromStream(std::istream &is) {
  std::string line, block_name, param_name, param_value, param_comment;
  std::size_t first_char, last_char;
  InputBlock *pib{};
  int blocks_found{0};
  bool source_has_canonical[ntime_parameter_aliases]{};
  bool source_has_legacy[ntime_parameter_aliases]{};

  while (is.good()) {
    std::getline(is, line);
    if (line.find('\t') != std::string::npos) {
      line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
    }
    if (line.empty()) continue;                             // skip blank line
    first_char = line.find_first_not_of(" ");               // skip white space
    if (first_char == std::string::npos) continue;          // line is all white space
    if (line.compare(first_char, 1, "#") == 0) continue;      // skip comments
    if (line.compare(first_char, 9, "<par_end>") == 0) break; // stop on <par_end>

    if (line.compare(first_char, 1, "<") == 0) {              // a new block
      first_char++;
      last_char = (line.find_first_of(">", first_char));
      block_name.assign(line, first_char, last_char-1);       // extract block name

      if (last_char == std::string::npos) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Block name '" << block_name
                  << "' not properly ended" << std::endl;
        std::exit(EXIT_FAILURE);
      }

      pib = FindOrAddBlock(block_name);  // find or add block to linked list

      if (pib == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Block name '" << block_name
                  << "' could not be found/added" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      blocks_found++;
      continue;  // skip to next line if block name was found
    } // end "a new block was found"

    // if line does not contain a block name or skippable information (comments,
    // whitespace), it must contain a parameter value
    if (blocks_found == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Input must specify a block name before first "
                  << "param = value line" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // parse line and add name/value/comment strings (if found) to current block name
    ParseLine(line, param_name, param_value, param_comment);
    AddParameter(pib, param_name, param_value, param_comment);
    if (pib->block_name == "time") {
      for (int n=0; n<ntime_parameter_aliases; ++n) {
        source_has_canonical[n] = source_has_canonical[n] ||
            (param_name == time_parameter_aliases[n].canonical);
        source_has_legacy[n] = source_has_legacy[n] ||
            (param_name == time_parameter_aliases[n].legacy);
      }
    }
  }

  // Normalize each loaded source independently so a later -i file overrides restart
  // parameters even when sources use different LAT parameter spellings.
  InputBlock *time_block = GetPtrToBlock("time");
  for (int n=0; n<ntime_parameter_aliases && time_block != nullptr; ++n) {
    if (!source_has_canonical[n] && source_has_legacy[n]) {
      const auto &alias = time_parameter_aliases[n];
      InputLine *legacy = time_block->GetPtrToLine(alias.legacy);
      AddParameter(time_block, alias.canonical, legacy->param_value,
                   std::string("# Canonical LAT parameter normalized from legacy ") +
                       alias.legacy);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void ParameterInput::LoadFromFile(IOWrapper &input)
//  \brief Read the parameters from an input or restart file.

void ParameterInput::LoadFromFile(IOWrapper &input, bool single_file_per_rank) {
  std::stringstream par;
  constexpr int kBufSize = 4096;
  char buf[kBufSize];
  IOWrapperSizeT header = 0, ret, loc;
  // <par_end> is written only by ParameterDump, so finding it is exactly what tells this
  // function it is reading a restart file's parameter dump and not an input file (the
  // search below is the code's own distinction).  The loop can also exit before `loc` is
  // ever assigned, which is why this is tracked separately.
  bool found_par_end = false;

  // search for <par_end> (reading from restart files) or EOF (reading from input file).
  do {
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      ret = input.Read_bytes(buf, sizeof(char), kBufSize, single_file_per_rank);
    }
#if MPI_PARALLEL_ENABLED
    // then broadcasts it
  if (!single_file_per_rank) {
    MPI_Bcast(&ret, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (ret == 0) {
      break;
    }
    MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
#endif
    par.write(buf, ret); // add the buffer into the stream
    header += ret;
    std::string sbuf = par.str(); // create string for search
    loc = sbuf.find("<par_end>", 0); // search from the top of the stream
    if (loc != std::string::npos) { // found <par_end>
      header = loc + 10; // store the header length
      found_par_end = true;
      break;
    }
    if (header > kBufSize*10) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<par_end> is not found in the first 40KBytes."
                << std::endl << "Probably the file is broken or the wrong file is "
                << "specified" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } while (ret == kBufSize); // till EOF (or par_end is found)

  // Now par contains the parameter inputs + some additional including <par_end>
  // Read the stream and load the parameters, marking their provenance for
  // RetireDeadParameter.
  loading_restart_header_ = found_par_end;
  LoadFromStream(par);
  loading_restart_header_ = false;
  // Seek the file to the end of the header
  input.Seek(header, single_file_per_rank);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InputBlock* ParameterInput::FindOrAddBlock(std::string name)
//  \brief find or add specified InputBlock.  Returns pointer to block.

InputBlock* ParameterInput::FindOrAddBlock(std::string name) {
  // if block contains no elements, create the first one
  if (block.empty()) {
    block.emplace_front(name);
    return &block.front();

  // else search linked list of InputBlocks to see if name exists, return if found.
  } else {
    for (auto it = block.begin(); it != block.end(); ++it) {
      if (name.compare(it->block_name) == 0) return &*it;
    }

    // Create new block at end of list if not found above, and return pointer to it
    block.emplace_back(name);
    return &block.back();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParseLine(std::string line,
//           std::string& name, std::string& value, std::string& comment)
//  \brief parse "name = value # comment" format, return name/value/comment strings.

void ParameterInput::ParseLine(std::string line, std::string& name, std::string& value,
                               std::string& comment) {
  std::size_t first_char, last_char, equal_char, hash_char, len;
  first_char = line.find_first_not_of(" ");   // find first non-white space
  equal_char = line.find_first_of("=");       // find "=" char
  hash_char  = line.find_first_of("#");       // find "#" (optional)

  // copy substring into name, remove white space at end of name
  len = equal_char - first_char;
  name.assign(line, first_char, len);

  last_char = name.find_last_not_of(" ");
  name.erase(last_char+1, std::string::npos);

  // copy substring into value, remove white space at start and end
  len = hash_char - equal_char - 1;
  value.assign(line, equal_char+1, len);

  first_char = value.find_first_not_of(" ");
  value.erase(0, first_char);

  last_char = value.find_last_not_of(" ");
  value.erase(last_char+1, std::string::npos);

  // copy substring into comment, if present
  if (hash_char != std::string::npos) {
    comment = line.substr(hash_char);
  } else {
    comment = "";
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::AddParameter(InputBlock *pb, std::string name,
//   std::string value, std::string comment)
//  \brief add name/value/comment tuple to the InputLine linked list in block *pb.
//  If a parameter with the same name already exists, the value and comment strings
//  are replaced (overwritten).

void ParameterInput::AddParameter(InputBlock *pb, std::string name, std::string value,
                                  std::string comment) {
  // if line contains no elements, create the first one
  if (pb->line.empty()) {
    pb->line.emplace_front(name,value,comment);
    pb->line.front().from_restart_header = loading_restart_header_;
    pb->max_len_parname = name.length();
    pb->max_len_parvalue = value.length();
    return;

  // else search linked list of InputBlocks to see if name exists, replace contents
  // with new values if found and return.
  } else {
    for (auto it = pb->line.begin(); it != pb->line.end(); ++it) {
      if (name.compare(it->param_name) == 0) {   // param name already exists
        it->param_value.assign(value);           // replace existing param value
        it->param_comment.assign(comment);       // replace existing param comment
        it->from_restart_header = loading_restart_header_;
        if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
        return;
      }
    }

  // Parameter not found, so create new node in linked list
    pb->line.emplace_back(name,value,comment);
    pb->line.back().from_restart_header = loading_restart_header_;
    if (name.length() > pb->max_len_parname) pb->max_len_parname = name.length();
    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! void ParameterInput::ModifyFromCmdline(int argc, char *argv[])
//  \brief parse commandline for changes to input parameters
// Note this function is very forgiving (no warnings!) if there is an error in format

void ParameterInput::ModifyFromCmdline(int argc, char *argv[]) {
  std::string input_text, block,name, value;
  InputBlock *pb;
  InputLine *pl;

  for (int i=1; i<argc; i++) {
    input_text = argv[i];
    std::size_t slash_posn = input_text.find_first_of("/");   // find "/" character
    std::size_t equal_posn = input_text.find_first_of("=");   // find "=" character

    // skip if either "/" or "=" do not exist in input
    if ((slash_posn == std::string::npos) || (equal_posn == std::string::npos)) continue;

    // extract block/name/value strings
    block = input_text.substr(0, slash_posn);
    name  = input_text.substr(slash_posn+1, (equal_posn - slash_posn - 1));
    value = input_text.substr(equal_posn+1, std::string::npos);
    if (block == "time") name = CanonicalTimeParameterName(name);

    // get pointer to node with same block name in linked list of InputBlocks
    pb = GetPtrToBlock(block);
    if (pb == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Block name '" << block << "' on command line not found"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // get pointer to node with same parameter name in linked list of InputLines
    pl = pb->GetPtrToLine(name);
    if (pl == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Parameter '" << name << "' in block '" << block
                << "' on command line not found" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pl->param_value.assign(value);   // replace existing value
    pl->from_restart_header = false;  // the user asked for this value, here, now

    if (value.length() > pb->max_len_parvalue) pb->max_len_parvalue = value.length();
  }
}


//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::RetireDeadParameter(block, name)
//! \brief settle a key the code no longer reads, and say whether the caller must refuse.
//!
//! Returns true when the key is present because a deck or the command line asked for it:
//! the run would silently disobey it, so the caller aborts and names the replacement.
//! Returns false when the key is absent, and also when it reached us in a restart file's
//! parameter dump -- there it is history, written by a binary from before the key died,
//! and refusing it would make every old checkpoint unrestartable.  That case drops the
//! line instead, with one rank-0 note, so it is gone from the next header this run writes
//! and a restart from THAT checkpoint is silent.

bool ParameterInput::RetireDeadParameter(const std::string &block,
                                         const std::string &name) {
  Lock();
  InputBlock *pb = GetPtrToBlock(block);
  InputLine *pl = (pb == nullptr) ? nullptr : pb->GetPtrToLine(name);
  if (pl == nullptr) {
    Unlock();
    return false;
  }
  if (!pl->from_restart_header) {
    Unlock();
    return true;
  }
  pb->line.remove_if([&name](const InputLine &l) {
    return l.param_name.compare(name) == 0;
  });
  Unlock();
  if (global_variable::my_rank == 0) {
    std::cout << "<" << block << ">/" << name << " came from the restart header and is "
              << "no longer read; dropped" << std::endl;
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::IsFromRestartHeader(block, name)
//! \brief true when the key exists and its value came from a restart file's parameter
//! dump with neither the deck nor the command line setting it since.

bool ParameterInput::IsFromRestartHeader(const std::string &block,
                                         const std::string &name) {
  Lock();
  InputBlock *pb = GetPtrToBlock(block);
  InputLine *pl = (pb == nullptr) ? nullptr : pb->GetPtrToLine(name);
  const bool from_header = (pl != nullptr) && pl->from_restart_header;
  Unlock();
  return from_header;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesBlockExist(std::string name)
//  \brief check whether block of given name exists

bool ParameterInput::DoesBlockExist(std::string name) {
  InputBlock *pb;
  pb = GetPtrToBlock(name);
  return (pb == nullptr ? false : true);
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::DoesParameterExist(std::string block, std::string name)
//  \brief check whether parameter of given name in given block exists

bool ParameterInput::DoesParameterExist(std::string block, std::string name) {
  InputLine *pl;
  InputBlock *pb;
  pb = GetPtrToBlock(block);
  if (pb == nullptr) return 0;
  pl = pb->GetPtrToLine(name);
  return (pl == nullptr ? false : true);
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::IsLATEnabled()
//  \brief return the canonical LAT switch, with compatibility for legacy inputs

bool ParameterInput::IsLATEnabled() {
  if (DoesParameterExist("time", "lat")) {
    return GetBoolean("time", "lat");
  }
  return DoesParameterExist("time", "hydro_lat") &&
         GetBoolean("time", "hydro_lat");
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetInteger(std::string block, std::string name)
//  \brief returns integer value of string stored in block/name

int ParameterInput::GetInteger(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int ret = ParseIntegerParameterOrExit(pl->param_value, block, name);
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetReal(std::string block, std::string name)
//  \brief returns real value of string stored in block/name

Real ParameterInput::GetReal(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real ret = ParseRealParameterOrExit(pl->param_value, block, name);
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetBoolean(std::string block, std::string name)
//  \brief returns boolean value of string stored in block/name

bool ParameterInput::GetBoolean(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  return ParseBooleanParameterOrExit(val, block, name);
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetString(std::string block, std::string name)
//  \brief returns string stored in block/name

std::string ParameterInput::GetString(std::string block, std::string name) {
  InputBlock* pb;
  InputLine* pl;

  Lock();

  // get pointer to node with same block name in linked list of InputBlocks
  pb = GetPtrToBlock(block);
  if (pb == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Block name '" << block << "' not found when trying to set value "
              << "for parameter '" << name << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get pointer to node with same parameter name in linked list of InputLines
  pl = pb->GetPtrToLine(name);
  if (pl == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Parameter name '" << name << "' not found in block '" << block
              << "'"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string val=pl->param_value;
  Unlock();

  // return value
  return val;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::GetOrAddInteger(std::string block, std::string name,
//    int default_value)
//  \brief returns integer value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

int ParameterInput::GetOrAddInteger(std::string block, std::string name, int def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  int ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = ParseIntegerParameterOrExit(pl->param_value, block, name);
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::GetOrAddReal(std::string block, std::string name,
//    Real def_value)
//  \brief returns real value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

Real ParameterInput::GetOrAddReal(std::string block, std::string name, Real def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  Real ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = ParseRealParameterOrExit(pl->param_value, block, name);
  } else {
    pb = FindOrAddBlock(block);
    ss_value << std::setprecision(kRealDumpDigits) << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::GetOrAddBoolean(std::string block, std::string name,
//    bool def_value)
//  \brief returns boolean value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

bool ParameterInput::GetOrAddBoolean(std::string block,std::string name, bool def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  bool ret = def_value;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = ParseBooleanParameterOrExit(pl->param_value, block, name);
  } else {
    pb = FindOrAddBlock(block);
    ss_value << def_value;
    AddParameter(pb, name, ss_value.str(), "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::GetOrAddString(std::string block, std::string name,
//                                                 std::string def_value)
//  \brief returns string value stored in block/name if it exists, or creates and sets
//  value to def_value if it does not exist

std::string ParameterInput::GetOrAddString(std::string block, std::string name,
                                           std::string def_value) {
  InputBlock* pb;
  InputLine *pl;
  std::stringstream ss_value;
  std::string ret;

  Lock();
  if (DoesParameterExist(block, name)) {
    pb = GetPtrToBlock(block);
    pl = pb->GetPtrToLine(name);
    ret = pl->param_value;
  } else {
    pb = FindOrAddBlock(block);
    AddParameter(pb, name, def_value, "# Default value added at run time");
    ret = def_value;
  }
  Unlock();
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn int ParameterInput::SetInteger(std::string block, std::string name, int value)
//  \brief updates an integer parameter; creates it if it does not exist

int ParameterInput::SetInteger(std::string block, std::string name, int value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn Real ParameterInput::SetReal(std::string block, std::string name, Real value)
//  \brief updates a real parameter; creates it if it does not exist

Real ParameterInput::SetReal(std::string block, std::string name, Real value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << std::setprecision(kRealDumpDigits) << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn bool ParameterInput::SetBoolean(std::string block, std::string name, bool value)
//  \brief updates a boolean parameter; creates it if it does not exist

bool ParameterInput::SetBoolean(std::string block, std::string name, bool value) {
  InputBlock* pb;
  std::stringstream ss_value;

  Lock();
  pb = FindOrAddBlock(block);
  ss_value << value;
  AddParameter(pb, name, ss_value.str(), "# Updated during run time");
  Unlock();
  return value;
}

//----------------------------------------------------------------------------------------
//! \fn std::string ParameterInput::SetString(std::string block, std::string name,
//                                            std::string  value)
//  \brief updates a string parameter; creates it if it does not exist

std::string ParameterInput::SetString(std::string block, std::string name,
                                      std::string value) {
  InputBlock* pb;

  Lock();
  pb = FindOrAddBlock(block);
  AddParameter(pb, name, value, "# Updated during run time");
  Unlock();
  return value;
}


//----------------------------------------------------------------------------------------
//! \fn void ParameterInput::ParameterDump(std::ostream& os)
//  \brief output entire InputBlock/InputLine hierarchy to specified stream

void ParameterInput::ParameterDump(std::ostream& os) {
  std::string param_name,param_value;
  std::size_t len;

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;

  for (auto itb = block.begin(); itb != block.end(); ++itb) {     // loop over InputBlocks
    os<< "<" << itb->block_name << ">" << std::endl;              // write block name
    // loop over InputLines and write each parameter name/value
    for (auto itl = itb->line.begin(); itl != itb->line.end(); ++itl) {
      param_name.assign(itl->param_name);
      param_value.assign(itl->param_value);

      len = itb->max_len_parname - param_name.length() + 1;
      param_name.append(len,' ');                         // pad name  to align vertically
      len = itb->max_len_parvalue - param_value.length() + 1;
      param_value.append(len,' ');                        // pad value to align vertically

      os<< param_name << "= " << param_value << itl->param_comment <<  std::endl;
    }
  }

  os<< "#------------------------- PAR_DUMP -------------------------" << std::endl;
  os<< "<par_end>" << std::endl;    // finish with par-end (needed for restart files)
}
