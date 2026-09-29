// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file basetype_output.cpp
//  \brief implements BaseTypeOutput constructor, and LoadOutputData functions
//

#include <cmath>     // floor
#include <iostream>
#include <sstream>
#include <string>    // std::string, to_string()
#include <cstdlib>   // std::exit
#include <cstdio>    // snprintf
#include <algorithm> // min_element
#include <utility>   // pair<>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/turb_driver.hpp"
#include "gravity/gravity.hpp"
#include "pgen/pgen.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

std::string NumberedScalarName(const char *prefix, int offset) {
  char number[3];
  std::snprintf(number, sizeof(number), "%02d", offset % 100);
  std::string vname(prefix);
  vname.append(number);
  return vname;
}

std::string MhdScalarName(mhd::MHD *pmhd, int slot, bool conserved) {
  int offset = slot - pmhd->nmhd;
  // The dual-energy auxiliary sits past the passive scalars.  Naming it separately is
  // what makes the formalism diagnosable at all: without it the only evidence of what
  // the auxiliary channel is carrying is the effect it has on everything else.
  if (pmhd->use_dual_energy && slot == pmhd->dual_energy_idx) {
    return conserved ? "reint_aux" : "eint_aux";
  }
  return NumberedScalarName(conserved ? "r_" : "s_", offset);
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void BaseTypeOutput::AdvanceOutputTime(Mesh *pm, ParameterInput *pin)
//! \brief Advance this stream's cadence after a dump, and record it for the next restart.
//!
//! The first dump of a stream (last_time still at its -1 sentinel) anchors the cadence to
//! the GLOBAL grid floor(t/dt)*dt, not to the instant the dump happened to land on.
//! Anchoring to pm->time made a stream's phase an accident of when its first write
//! occurred, and `last_time += dt` then carried that accident forever.  Concretely, in a
//! live run two <output> blocks with the identical dt = 100 came out permanently
//! 2.1333 M apart: an <output> block added to the deck mid-run has no last_time in the
//! restart header, so it first fired at the first LAT window boundary after the restart
//! point, at t = 25001.884, and 1.884 M of phase was then baked in for the rest of the
//! run.  Pairing the two dumps in analysis compares states 2.1 M apart, which is 12% of
//! an orbit at r = 2M.  With grid anchoring both streams sit on multiples of 100 no
//! matter when they were created; for a fresh start at t = 0 the two rules agree exactly.
//! floor(t/dt)*dt + dt > t always, so the anchor can never make a stream immediately due
//! again.
void BaseTypeOutput::AdvanceOutputTime(Mesh *pm, ParameterInput *pin) {
  if (out_params.last_time < 0.0) {
    out_params.last_time = (out_params.dt > 0.0) ?
        std::floor(pm->time/out_params.dt)*out_params.dt : pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}

//----------------------------------------------------------------------------------------
// BaseTypeOutput base class constructor
// Creates vector of output variable data

BaseTypeOutput::BaseTypeOutput(ParameterInput *pin, Mesh *pm, OutputParameters opar) :
    out_params(opar),
    derived_var("derived-var",1,1,1,1,1),
    outarray("cc_outvar",1,1,1,1,1),
    outfield("fc_outvar",1,1,1,1) {
  // exit for history, restart, or event log files
  if (out_params.file_type.compare("hst") == 0 ||
      out_params.file_type.compare("rst") == 0 ||
      out_params.file_type.compare("log") == 0 ||
      out_params.file_type.compare("trk") == 0) {return;}

  // initialize vector containing number of output MBs per rank
  noutmbs.assign(global_variable::nranks, 0);

  // check for valid choice of variables
  int ivar = -1;
  for (int i=0; i<(NOUTPUT_CHOICES); ++i) {
    if (out_params.variable.compare(var_choice[i]) == 0) {ivar = i;}
  }
  if (ivar < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Variable '" << out_params.variable << "' in block '" << out_params.block_name
       << "' in input file is not a valid choice" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // check that appropriate physics is defined for requested output variable
  // TODO(@user): Index limits of variable choices below may change if more choices added
  if ((ivar<16) && (pm->pmb_pack->phydro == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Hydro variable requested in <output> block '"
       << out_params.block_name << "' but no Hydro object has been constructed."
       << std::endl << "Input file is likely missing a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((out_params.variable.compare("hydro_div_v") == 0 ||
       out_params.variable.compare("hydro_abs_div_v") == 0) &&
      pm->pmb_pack->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Hydro derived variable requested in <output> block '"
       << out_params.block_name << "' but no Hydro object has been constructed."
       << std::endl << "Input file is likely missing a <hydro> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=16) && (ivar<50) && (pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of MHD variable requested in <output> block '"
       << out_params.block_name << "' but no MHD object has been constructed."
       << std::endl << "Input file is likely missing a <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=154) && (ivar<163) && (pm->pmb_pack->phydro == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Hydro tabulated-EOS diagnostic requested in <output> block '"
       << out_params.block_name << "' but no Hydro object has been constructed."
       << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=163) && (ivar<172) && (pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of MHD tabulated-EOS diagnostic requested in <output> block '"
       << out_params.block_name << "' but no MHD object has been constructed."
       << std::endl;
    exit(EXIT_FAILURE);
  }
  auto require_tabulated_lte = [&](const std::string &prefix, const EOS_Data &eos) {
    if (!eos.UsesTabulatedLTE()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Output of " << prefix << " tabulated-EOS diagnostic requested in "
                << "<output> block '" << out_params.block_name << "', but <" << prefix
                << ">/eos is not a tabulated LTE EOS." << std::endl;
      exit(EXIT_FAILURE);
    }
  };
  auto require_lte_helium = [&](const std::string &prefix, const EOS_Data &eos,
                                const std::string &varname) {
    require_tabulated_lte(prefix, eos);
    if (!eos.lte_has_helium) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Output variable '" << varname << "' requested in <output> block '"
                << out_params.block_name << "', but the active <" << prefix
                << ">/eos table does not include helium ionization fractions."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  };
  auto require_lte_radiation = [&](const std::string &prefix, const EOS_Data &eos,
                                   const std::string &varname) {
    require_tabulated_lte(prefix, eos);
    if (!eos.lte_has_radiation) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Output variable '" << varname << "' requested in <output> block '"
                << out_params.block_name << "', but the active <" << prefix
                << ">/eos table does not include radiation-pressure support."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  };
  auto require_lte_h2 = [&](const std::string &prefix, const EOS_Data &eos,
                            const std::string &varname) {
    require_tabulated_lte(prefix, eos);
    if (!eos.lte_has_h2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Output variable '" << varname << "' requested in <output> block '"
                << out_params.block_name << "', but the active <" << prefix
                << ">/eos table does not include H2 chemistry."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  };
  if ((ivar>=154) && (ivar<163) && (pm->pmb_pack->phydro != nullptr)) {
    auto &eos = pm->pmb_pack->phydro->peos->eos_data;
    if (ivar == 155) {
      require_lte_h2("hydro", eos, out_params.variable);
    } else if (ivar == 157 || ivar == 158) {
      require_lte_helium("hydro", eos, out_params.variable);
    } else if (ivar == 162) {
      require_lte_radiation("hydro", eos, out_params.variable);
    } else {
      require_tabulated_lte("hydro", eos);
    }
  }
  if ((ivar>=163) && (ivar<172) && (pm->pmb_pack->pmhd != nullptr)) {
    auto &eos = pm->pmb_pack->pmhd->peos->eos_data;
    if (ivar == 164) {
      require_lte_h2("mhd", eos, out_params.variable);
    } else if (ivar == 166 || ivar == 167) {
      require_lte_helium("mhd", eos, out_params.variable);
    } else if (ivar == 171) {
      require_lte_radiation("mhd", eos, out_params.variable);
    } else {
      require_tabulated_lte("mhd", eos);
    }
  }
  if ((ivar==38) && (pm->pmb_pack->pdyngr == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of DynMHD variable requested in <output> block '"
       << out_params.block_name << "' but no DynMHD object has been constructed."
       << std::endl << "Input file is likely missing a <adm> or <z4c>, and/or <mhd> block"
       << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==50) && (pm->pmb_pack->pturb == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Force variable requested in <output> block '"
       << out_params.block_name << "' but no Force object has been constructed."
       << std::endl << "Input file is likely missing a <forcing> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (ivar==51 && (pm->pmb_pack->prad == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation moments requested in <output> block '"
       << out_params.block_name << "' but no Radiation object has been constructed."
       << std::endl << "Input file is likely missing a <radiation> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar==52 || ivar==53) &&
      ((pm->pmb_pack->prad == nullptr) ||
       (pm->pmb_pack->phydro == nullptr && pm->pmb_pack->pmhd == nullptr))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Fluid Frame Radiation moments requested in <output> block '"
       << out_params.block_name << "' but either Radiation object has not been "
       << " constructed, or corresponding Hydro or MHD object missing" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=54) && (ivar<68) &&
      (pm->pmb_pack->prad == nullptr || pm->pmb_pack->phydro == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation Hydro variables requested in <output> block '"
       << out_params.block_name << "' but Radiation and/or Hydro object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=68) && (ivar<88) &&
      (pm->pmb_pack->prad == nullptr || pm->pmb_pack->pmhd == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Radiation MHD variables requested in <output> block '"
       << out_params.block_name << "' but Radiation and/or MHD object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=88) && (ivar<106) && (pm->pmb_pack->padm == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of ADM variable requested in <output> block '"
       << out_params.block_name << "' but ADM object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=106) && (ivar<129) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Z4c variable requested in <output> block '"
       << out_params.block_name << "' but Z4c object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=129) && (ivar<132) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of weyl variable requested in <output> block '"
       << out_params.block_name << "' but weyl object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=132) && (ivar<140) && (pm->pmb_pack->pz4c == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of constraint variables request in <output> block '"
       << out_params.block_name << "' but Z4c object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((ivar>=140) && (ivar<151) && (pm->pmb_pack->ptmunu == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of Tmunu variable requested in <output> block '"
       << out_params.block_name << "' but no Tmunu object has been constructed."
       << std::endl
       << "Tmunu is the matter source of the Einstein equations and is allocated only "
       << "when Z4c evolves the spacetime. With a prescribed metric nothing computes it "
       << "past initialization, so there is no meaningful field to output." << std::endl;
    // This branch used to fall through and then dereference the null ptmunu when
    // building the output variable list.
    exit(EXIT_FAILURE);
  }
  if ((ivar>=151) && (ivar<153) && (pm->pmb_pack->ppart == nullptr)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of particles requested in <output> block '"
       << out_params.block_name << "' but particle object not constructed."
       << std::endl << "Input file is likely missing corresponding block" << std::endl;
    exit(EXIT_FAILURE);
  }
  const bool has_total_gravity_potential =
      (pm->pmb_pack->pgrav != nullptr) || problem_runtime::HasExternalBHPotential();
  if (ivar==153 && !has_total_gravity_potential) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Output of gravity potential requested in <output> block '"
       << out_params.block_name << "' but neither self-gravity nor an external BH "
       << "potential is available." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Now load STL vector of output variables
  outvars.clear();

  // make a vector of out_params.variables
  std::vector<std::string> variables;

  variables.push_back(out_params.variable);
  if (out_params.file_type == "pdf") {
    if (out_params.nbin2 > 0) {
      variables.push_back(out_params.variable_2);
    }
  }

  for (const auto &variable : variables) {
  }

  for (const auto& variable : variables) {
    // hydro (lab-frame) density
    if (variable.compare("hydro_u_d") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_d") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->u0));
    }

    // hydro (rest-frame) density
    if (variable.compare("hydro_w_d") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_d") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->phydro->w0));
    }

    // hydro components of momentum
    if (variable.compare("hydro_u_m1") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m1") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->phydro->u0));
    }
    if (variable.compare("hydro_u_m2") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m2") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->phydro->u0));
    }
    if (variable.compare("hydro_u_m3") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_m3") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->phydro->u0));
    }

    // hydro components of velocity
    if (variable.compare("hydro_w_vx") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vx") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("velx",1,&(pm->pmb_pack->phydro->w0));
    }
    if (variable.compare("hydro_w_vy") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vy") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("vely",2,&(pm->pmb_pack->phydro->w0));
    }
    if (variable.compare("hydro_w_vz") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_vz") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      outvars.emplace_back("velz",3,&(pm->pmb_pack->phydro->w0));
    }

    // hydro total energy
    if (variable.compare("hydro_u_e") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_e") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      if (pm->pmb_pack->phydro->peos->eos_data.use_e) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->phydro->u0));
      }
    }

    // hydro internal energy or temperature
    if (variable.compare("hydro_w_e") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_e") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      if (pm->pmb_pack->phydro->peos->eos_data.use_e) {
        if (pm->pmb_pack->pdyngr != nullptr) {
          outvars.emplace_back("press",4,&(pm->pmb_pack->phydro->w0));
        } else {
          outvars.emplace_back("eint",4,&(pm->pmb_pack->phydro->w0));
        }
      }
    }
    // hydro passive scalars mass densities (s*d)
    if (variable.compare("hydro_u_s") == 0 ||
        variable.compare("hydro_u") == 0 ||
        variable.compare("rad_hydro_u_s") == 0 ||
        variable.compare("rad_hydro_u") == 0) {
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd)%100);
        std::string vname;
        vname.assign("r_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->u0));
      }
    }

    // hydro passive scalars (s)
    if (variable.compare("hydro_w_s") == 0 ||
        variable.compare("hydro_w") == 0 ||
        variable.compare("rad_hydro_w_s") == 0 ||
        variable.compare("rad_hydro_w") == 0) {
      int nhyd = pm->pmb_pack->phydro->nhydro;
      int nvars = nhyd + pm->pmb_pack->phydro->nscalars;
      for (int n=nhyd; n<nvars; ++n) {
        char number[3];
        std::snprintf(number,sizeof(number),"%02d",(n - nhyd)%100);
        std::string vname;
        vname.assign("s_");
        vname.append(number);
        outvars.emplace_back(vname,n,&(pm->pmb_pack->phydro->w0));
      }
    }

    // mhd (lab-frame) density
    if (variable.compare("mhd_u_d") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_d") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->u0));
    }

    // mhd (rest-frame) density
    if (variable.compare("mhd_w_d") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_d") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("dens",0,&(pm->pmb_pack->pmhd->w0));
    }

    // mhd components of momentum
    if (variable.compare("mhd_u_m1") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m1") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom1",1,&(pm->pmb_pack->pmhd->u0));
    }
    if (variable.compare("mhd_u_m2") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m2") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom2",2,&(pm->pmb_pack->pmhd->u0));
    }
    if (variable.compare("mhd_u_m3") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_m3") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      outvars.emplace_back("mom3",3,&(pm->pmb_pack->pmhd->u0));
    }

    // mhd components of velocity
    if (variable.compare("mhd_w_vx") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vx") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("velx",1,&(pm->pmb_pack->pmhd->w0));
    }
    if (variable.compare("mhd_w_vy") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vy") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("vely",2,&(pm->pmb_pack->pmhd->w0));
    }
    if (variable.compare("mhd_w_vz") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_vz") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("velz",3,&(pm->pmb_pack->pmhd->w0));
    }

    // mhd total energy
    if (variable.compare("mhd_u_e") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_e") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      if (pm->pmb_pack->pmhd->peos->eos_data.use_e) {
        outvars.emplace_back("ener",4,&(pm->pmb_pack->pmhd->u0));
      }
    }

    // mhd internal energy or temperature
    if (variable.compare("mhd_w_e") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_e") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      if (pm->pmb_pack->pmhd->peos->eos_data.use_e) {
        if (pm->pmb_pack->pdyngr != nullptr) {
          outvars.emplace_back("press",4,&(pm->pmb_pack->pmhd->w0));
        } else {
          outvars.emplace_back("eint",4,&(pm->pmb_pack->pmhd->w0));
        }
      }
    }

    // mhd passive scalars mass densities (s*d)
    if (variable.compare("mhd_u_s") == 0 ||
        variable.compare("mhd_u") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_u_s") == 0 ||
        variable.compare("rad_mhd_u") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0) {
      int nmhd = pm->pmb_pack->pmhd->nmhd;
      int nvars = pm->pmb_pack->pmhd->nvars;
      for (int n=nmhd; n<nvars; ++n) {
        outvars.emplace_back(MhdScalarName(pm->pmb_pack->pmhd, n, true), n,
                             &(pm->pmb_pack->pmhd->u0));
      }
    }

    // mhd passive scalars (s)
    if (variable.compare("mhd_w_s") == 0 ||
        variable.compare("mhd_w") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_w_s") == 0 ||
        variable.compare("rad_mhd_w") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      int nmhd = pm->pmb_pack->pmhd->nmhd;
      int nvars = pm->pmb_pack->pmhd->nvars;
      for (int n=nmhd; n<nvars; ++n) {
        outvars.emplace_back(MhdScalarName(pm->pmb_pack->pmhd, n, false), n,
                             &(pm->pmb_pack->pmhd->w0));
      }
    }

    // mhd cell-centered magnetic fields
    if (variable.compare("mhd_bcc1") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc1") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc1",0,&(pm->pmb_pack->pmhd->bcc0));
    }
    if (variable.compare("mhd_bcc2") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc2") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc2",1,&(pm->pmb_pack->pmhd->bcc0));
    }
    if (variable.compare("mhd_bcc3") == 0 ||
        variable.compare("mhd_bcc") == 0 ||
        variable.compare("mhd_u_bcc") == 0 ||
        variable.compare("mhd_w_bcc") == 0 ||
        variable.compare("rad_mhd_bcc3") == 0 ||
        variable.compare("rad_mhd_bcc") == 0 ||
        variable.compare("rad_mhd_u_bcc") == 0 ||
        variable.compare("rad_mhd_w_bcc") == 0) {
      outvars.emplace_back("bcc3",2,&(pm->pmb_pack->pmhd->bcc0));
    }

    // MHD temperature
    const bool requested_mhd_t = variable.compare("mhd_t") == 0;
    const bool requested_dyn_mhd_bundle =
        (variable.compare("mhd_w") == 0 || variable.compare("mhd_w_bcc") == 0) &&
        pm->pmb_pack->pdyngr != nullptr;
    if ((requested_mhd_t || requested_dyn_mhd_bundle) &&
        pm->pmb_pack->pdyngr != nullptr) {
      if (!pm->pmb_pack->pdyngr->StoreTemperature()) {
        if (requested_mhd_t) {
          outvars.emplace_back("temperature", 0, &(pm->pmb_pack->pmhd->w0),
                               false, true);
        }
      } else {
        outvars.emplace_back("temperature",0,&(pm->pmb_pack->pdyngr->temperature));
      }
    }

    // hydro/mhd z-component of vorticity (useful in 2D)
    if (variable.compare("hydro_wz") == 0 ||
        variable.compare("mhd_wz") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("vorz",i_derived,&(derived_var));
    }

    // hydro/mhd magnitude of vorticity (useful in 3D)
    if (variable.compare("hydro_w2") == 0 ||
        variable.compare("mhd_w2") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("vor2",i_derived,&(derived_var));
    }

    if (variable.compare("hydro_div_v") == 0 ||
        variable.compare("hydro_abs_div_v") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back(variable.c_str(), i_derived, &(derived_var));
    }

    if (variable.compare("hydro_temperature") == 0 ||
        variable.compare("hydro_xh2") == 0 ||
        variable.compare("hydro_xion") == 0 ||
        variable.compare("hydro_xhe1") == 0 ||
        variable.compare("hydro_xhe2") == 0 ||
        variable.compare("hydro_gamma1") == 0 ||
        variable.compare("hydro_gamma3m1") == 0 ||
        variable.compare("hydro_mu") == 0 ||
        variable.compare("hydro_beta_rad") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      if (variable.compare("hydro_temperature") == 0) {
        outvars.emplace_back("temperature", i_derived, &(derived_var));
      } else if (variable.compare("hydro_xh2") == 0) {
        outvars.emplace_back("xh2", i_derived, &(derived_var));
      } else if (variable.compare("hydro_xion") == 0) {
        outvars.emplace_back("xion", i_derived, &(derived_var));
      } else if (variable.compare("hydro_xhe1") == 0) {
        outvars.emplace_back("xhe1", i_derived, &(derived_var));
      } else if (variable.compare("hydro_xhe2") == 0) {
        outvars.emplace_back("xhe2", i_derived, &(derived_var));
      } else if (variable.compare("hydro_gamma1") == 0) {
        outvars.emplace_back("gamma1", i_derived, &(derived_var));
      } else if (variable.compare("hydro_mu") == 0) {
        outvars.emplace_back("mu", i_derived, &(derived_var));
      } else if (variable.compare("hydro_beta_rad") == 0) {
        outvars.emplace_back("beta_rad", i_derived, &(derived_var));
      } else {
        outvars.emplace_back("gamma3m1", i_derived, &(derived_var));
      }
    }

    if (variable.compare("mhd_temperature") == 0 ||
        variable.compare("mhd_xh2") == 0 ||
        variable.compare("mhd_xion") == 0 ||
        variable.compare("mhd_xhe1") == 0 ||
        variable.compare("mhd_xhe2") == 0 ||
        variable.compare("mhd_gamma1") == 0 ||
        variable.compare("mhd_gamma3m1") == 0 ||
        variable.compare("mhd_mu") == 0 ||
        variable.compare("mhd_beta_rad") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      if (variable.compare("mhd_temperature") == 0) {
        outvars.emplace_back("temperature", i_derived, &(derived_var));
      } else if (variable.compare("mhd_xh2") == 0) {
        outvars.emplace_back("xh2", i_derived, &(derived_var));
      } else if (variable.compare("mhd_xion") == 0) {
        outvars.emplace_back("xion", i_derived, &(derived_var));
      } else if (variable.compare("mhd_xhe1") == 0) {
        outvars.emplace_back("xhe1", i_derived, &(derived_var));
      } else if (variable.compare("mhd_xhe2") == 0) {
        outvars.emplace_back("xhe2", i_derived, &(derived_var));
      } else if (variable.compare("mhd_gamma1") == 0) {
        outvars.emplace_back("gamma1", i_derived, &(derived_var));
      } else if (variable.compare("mhd_mu") == 0) {
        outvars.emplace_back("mu", i_derived, &(derived_var));
      } else if (variable.compare("mhd_beta_rad") == 0) {
        outvars.emplace_back("beta_rad", i_derived, &(derived_var));
      } else {
        outvars.emplace_back("gamma3m1", i_derived, &(derived_var));
      }
    }

    // mhd z-component of current density (useful in 2D)
    if (variable.compare("mhd_jz") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("jz",i_derived,&(derived_var));
    }

    // mhd magnitude of current density (useful in 3D)
    if (variable.compare("mhd_j2") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("j2",i_derived,&(derived_var));
    }

    // Added by DBF --- check & update NOUTPUT_CHOICES
    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_curv") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("curv",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_k_jxb") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("k_jxb",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_curv_perp") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("curv_perp",i_derived,&(derived_var));
    }

    // mhd magnitude of magnetic curvature
    if (variable.compare("mhd_bmag") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("bmag",i_derived,&(derived_var));
    }

    // mhd divergence of B
    if (variable.compare("mhd_divb") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      int i_derived = out_params.n_derived - 1;
      outvars.emplace_back("divb",i_derived,&(derived_var));
    }

    // added by GNW --- contravariant components of magnetic field
    if (out_params.variable.compare("mhd_jcon") == 0) {
      pm->pmb_pack->pmhd->SetSaveWBcc();
      out_params.contains_derived = true;
      out_params.n_derived += 4;
      outvars.emplace_back("jcon0",0,&(derived_var));
      outvars.emplace_back("jcon1",1,&(derived_var));
      outvars.emplace_back("jcon2",2,&(derived_var));
      outvars.emplace_back("jcon3",3,&(derived_var));
    }

    // Hydro SGS tensor
    if (variable.compare("hydro_sgs") == 0) {
      out_params.contains_derived = true;
      // emplace all 23 components of the SGS tensor
      for (int i=0; i<23; ++i) {
          std::string variable_name;
          variable_name.assign("hydro_sgs_");
          variable_name.append(std::to_string(i+1));
          out_params.n_derived += 1;
          outvars.emplace_back(variable_name,i,&(derived_var));
      }
    }

    // Mhd SGS tensor
    if (variable.compare("mhd_sgs") == 0) {
      out_params.contains_derived = true;
      // emplace all 59 components of the SGS tensor
      for (int i=0; i<59; ++i) {
          std::string variable_name;
          variable_name.assign("mhd_sgs_");
          variable_name.append(std::to_string(i+1));
          out_params.n_derived += 1;
          outvars.emplace_back(variable_name.c_str(),i,&(derived_var));
      }
    }

    // mhd_dynamo_ks
    if (variable.compare("mhd_dynamo_ks") == 0) {
      out_params.contains_derived = true;
      // emplace all 8 components of the SGS tensor
      outvars.emplace_back("mhd_dynamo_B^2",0,&(derived_var));
      outvars.emplace_back("mhd_dynamo_B^4",1,&(derived_var));
      outvars.emplace_back("mhd_dynamo_dB^2",2,&(derived_var));
      outvars.emplace_back("mhd_dynamo_BdB^2",3,&(derived_var));
      outvars.emplace_back("mhd_dynamo_|BxJ|^2",4,&(derived_var));
      outvars.emplace_back("mhd_dynamo_|B.J|^2",5,&(derived_var));
      outvars.emplace_back("mhd_dynamo_U^2",6,&(derived_var));
      outvars.emplace_back("mhd_dynamo_dU",7,&(derived_var));
      out_params.n_derived += 8;
    }

    // turbulent forcing
    if (variable.compare("turb_force") == 0) {
      outvars.emplace_back("force1",0,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force2",1,&(pm->pmb_pack->pturb->force));
      outvars.emplace_back("force3",2,&(pm->pmb_pack->pturb->force));
    }

    // ADM variables, excluding gauge
    for (int v = 0; v < adm::ADM::nadm - 4; ++v) {
      if (variable.compare("adm") == 0 ||
          variable.compare(adm::ADM::ADM_names[v]) == 0) {
        outvars.emplace_back(adm::ADM::ADM_names[v], v, &(pm->pmb_pack->padm->u_adm));
        if (!pm->pmb_pack->padm->StoresMetricGrid()) {
          outvars.back().stream_adm_metric = true;
        }
      }
    }

    // ADM gauge variables
    if (nullptr == pm->pmb_pack->pz4c) {
      for (int v = adm::ADM::nadm - 4; v < adm::ADM::nadm; ++v) {
        if (variable.compare("adm") == 0 ||
          variable.compare(adm::ADM::ADM_names[v]) == 0) {
          outvars.emplace_back(adm::ADM::ADM_names[v], v, &(pm->pmb_pack->padm->u_adm));
          if (!pm->pmb_pack->padm->StoresMetricGrid()) {
            outvars.back().stream_adm_metric = true;
          }
        }
      }
    }

    // mat z4c variables
    for (int v = 0; v < Tmunu::N_Tmunu; ++v) {
      if (variable.compare("tmunu") == 0 ||
          variable.compare(Tmunu::Tmunu_names[v]) == 0) {
        outvars.emplace_back(Tmunu::Tmunu_names[v], v, &(pm->pmb_pack->ptmunu->u_tmunu));
      }
    }
    // con z4c variables
    for (int v = 0; v < z4c::Z4c::ncon; ++v) {
      if (variable.compare("con") == 0 ||
          variable.compare(z4c::Z4c::Constraint_names[v]) == 0) {
        outvars.emplace_back(z4c::Z4c::Constraint_names[v], v,
        &(pm->pmb_pack->pz4c->u_con));
      }
    }

    // z4c variables
    for (int v = 0; v < z4c::Z4c::nz4c; ++v) {
      if (variable.compare("z4c") == 0 ||
          variable.compare(z4c::Z4c::Z4c_names[v]) == 0) {
        outvars.emplace_back(z4c::Z4c::Z4c_names[v], v, &(pm->pmb_pack->pz4c->u0));
      }
    }

    // weyl scalars
    if (variable.compare("weyl") == 0) {
      outvars.emplace_back("weyl_rpsi4",0,&(pm->pmb_pack->pz4c->u_weyl));
      outvars.emplace_back("weyl_ipsi4",1,&(pm->pmb_pack->pz4c->u_weyl));
    }

    // radiation moments in coordinate frame
    if (variable.compare(0, 9, "rad_coord") == 0 ||
        variable.compare(0, 9, "rad_hydro") == 0 ||
        variable.compare(0, 7, "rad_mhd") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 10;
      outvars.emplace_back("r00",0,&(derived_var));
      outvars.emplace_back("r01",1,&(derived_var));
      outvars.emplace_back("r02",2,&(derived_var));
      outvars.emplace_back("r03",3,&(derived_var));
      outvars.emplace_back("r11",4,&(derived_var));
      outvars.emplace_back("r12",5,&(derived_var));
      outvars.emplace_back("r13",6,&(derived_var));
      outvars.emplace_back("r22",7,&(derived_var));
      outvars.emplace_back("r23",8,&(derived_var));
      outvars.emplace_back("r33",9,&(derived_var));
    }

    // radiation moments in fluid frame
    if (variable.compare("rad_fluid") == 0 ||
        variable.compare("rad_coord_fluid") == 0 ||
        variable.compare(0, 9, "rad_hydro") == 0 ||
        variable.compare(0, 7, "rad_mhd") == 0) {
      bool needs_fluid_only = (variable.compare("rad_fluid") == 0);
      int moments_offset = !(needs_fluid_only) ? 10 : 0;
      out_params.contains_derived = true;
      out_params.n_derived += 10;
      outvars.emplace_back("r00_ff",moments_offset+0,&(derived_var));
      outvars.emplace_back("r01_ff",moments_offset+1,&(derived_var));
      outvars.emplace_back("r02_ff",moments_offset+2,&(derived_var));
      outvars.emplace_back("r03_ff",moments_offset+3,&(derived_var));
      outvars.emplace_back("r11_ff",moments_offset+4,&(derived_var));
      outvars.emplace_back("r12_ff",moments_offset+5,&(derived_var));
      outvars.emplace_back("r13_ff",moments_offset+6,&(derived_var));
      outvars.emplace_back("r22_ff",moments_offset+7,&(derived_var));
      outvars.emplace_back("r23_ff",moments_offset+8,&(derived_var));
      outvars.emplace_back("r33_ff",moments_offset+9,&(derived_var));
    }

    const bool append_bundle_grav_phi =
        (variable.compare("hydro_u") == 0 ||
         variable.compare("hydro_w") == 0 ||
         variable.compare("rad_hydro_u") == 0 ||
         variable.compare("rad_hydro_w") == 0 ||
         variable.compare("mhd_u") == 0 ||
         variable.compare("mhd_w") == 0 ||
         variable.compare("mhd_u_bcc") == 0 ||
         variable.compare("mhd_w_bcc") == 0 ||
         variable.compare("rad_mhd_u") == 0 ||
         variable.compare("rad_mhd_w") == 0 ||
         variable.compare("rad_mhd_u_bcc") == 0 ||
         variable.compare("rad_mhd_w_bcc") == 0);
    if ((variable.compare("grav_phi") == 0 || append_bundle_grav_phi) &&
        has_total_gravity_potential) {
      const bool stream_grav_phi =
          (out_params.file_type.compare("bin") == 0 ||
           out_params.file_type.compare("vtk") == 0);
      if (stream_grav_phi) {
        outvars.emplace_back("grav_phi", 0, &(derived_var), true);
      } else {
        out_params.contains_derived = true;
        out_params.n_derived += 1;
        int i_derived = out_params.n_derived - 1;
        outvars.emplace_back("grav_phi", i_derived, &(derived_var));
      }
    }

    // particle density binned to mesh
    if (variable.compare("prtcl_d") == 0) {
      out_params.contains_derived = true;
      out_params.n_derived += 1;
      outvars.emplace_back("pdens",0,&(derived_var));
    }
  }

  const bool streams_analytic_adm = std::any_of(
      outvars.begin(), outvars.end(),
      [](const OutputVariableInfo &var) { return var.stream_adm_metric; });
  if (streams_analytic_adm &&
      (out_params.file_type == "cbin" || out_params.file_type == "pdf" ||
       out_params.file_type == "cart" || out_params.file_type == "sph")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output type '" << out_params.file_type << "' requests ADM variables, "
              << "but <adm>/metric_backend is an on-the-fly analytical metric so there "
              << "is no stored ADM grid to read." << std::endl
              << "The cbin, pdf, cart and sph writers each load output variables straight "
              << "from their backing device array; only tab, vtk and bin go through the "
              << "streaming path that can evaluate the metric per point." << std::endl
              << "Either dump ADM variables through a tab/vtk/bin output, or set "
              << "<adm>/metric_backend=stored for this run (which costs "
              << "17 x ncells x 8 bytes per MeshBlock of device memory)." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // initialize vector containing number of output MBs per rank
  noutmbs.assign(global_variable::nranks, 0);
}

//----------------------------------------------------------------------------------------
// BaseTypeOutput::LoadOutputData()
// create std::vector of HostArray3Ds containing data specified in <output> block for
// this output type

void BaseTypeOutput::LoadOutputData(Mesh *pm) {
  // out_data_ vector (indexed over # of output MBs) stores 4D array of variables
  // so start iteration over number of MeshBlocks
  // TODO(@user): get this working for multiple physics, which may be either defined/undef

  // With AMR, number and location of output MBs can change between output times.
  // So start with clean vector of output MeshBlock info, and re-compute
  outmbs.clear();

  // loop over all MeshBlocks
  // set size & starting indices of output arrays, adjusted accordingly if gz included
  auto &indcs = pm->mb_indcs;
  auto &size  = pm->pmb_pack->pmb->mb_size;
  auto &gids  = pm->pmb_pack->gids;
  for (int m=0; m<(pm->pmb_pack->nmb_thispack); ++m) {
    // skip if MeshBlock ID is specified and not equal to this ID
    if (out_params.gid >= 0 && (m+gids) != out_params.gid) { continue; }

    int ois,oie,ojs,oje,oks,oke;

    if (out_params.include_gzs) {
      int nout1 = indcs.nx1 + 2*(indcs.ng);
      int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      ois = 0; oie = nout1-1;
      ojs = 0; oje = nout2-1;
      oks = 0; oke = nout3-1;
    } else {
      ois = indcs.is; oie = indcs.ie;
      ojs = indcs.js; oje = indcs.je;
      oks = indcs.ks; oke = indcs.ke;
    }

    // check for slicing in each dimension, adjust start/end indices accordingly
    if (out_params.slice1) {
      // skip this MB if slice is out of range
      if (out_params.slice_x1 <  size.h_view(m).x1min ||
          out_params.slice_x1 >= size.h_view(m).x1max) { continue; }
      // set index of slice
      ois = CellCenterIndex(out_params.slice_x1, indcs.nx1,
                            size.h_view(m).x1min, size.h_view(m).x1max);
      ois += indcs.is;
      oie = ois;
    }

    if (out_params.slice2) {
      // skip this MB if slice is out of range
      if (out_params.slice_x2 <  size.h_view(m).x2min ||
          out_params.slice_x2 >= size.h_view(m).x2max) { continue; }
      // set index of slice
      ojs = CellCenterIndex(out_params.slice_x2, indcs.nx2,
                            size.h_view(m).x2min, size.h_view(m).x2max);
      ojs += indcs.js;
      oje = ojs;
    }

    if (out_params.slice3) {
      // skip this MB if slice is out of range
      if (out_params.slice_x3 <  size.h_view(m).x3min ||
          out_params.slice_x3 >= size.h_view(m).x3max) { continue; }
      // set index of slice
      oks = CellCenterIndex(out_params.slice_x3, indcs.nx3,
                            size.h_view(m).x3min, size.h_view(m).x3max);
      oks += indcs.ks;
      oke = oks;
    }

    // set coordinate geometry information for MB
    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min;
    Real x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min;
    Real x3max = size.h_view(m).x3max;

    int id = pm->pmb_pack->pmb->mb_gid.h_view(m);
    outmbs.emplace_back(id,ois,oie,ojs,oje,oks,oke,x1min,x1max,x2min,x2max,x3min,x3max);
  }

  std::fill(noutmbs.begin(), noutmbs.end(), 0);
  noutmbs[global_variable::my_rank] = outmbs.size();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, noutmbs.data(), global_variable::nranks,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  noutmbs_min = *std::min_element(noutmbs.begin(), noutmbs.end());
  noutmbs_max = *std::max_element(noutmbs.begin(), noutmbs.end());


  // get number of output vars and MBs, then realloc outarray (HostArray)
  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  int nout1 = 0;
  int nout2 = 0;
  int nout3 = 0;
  // note that while ois,oie,etc. can be different on each MB, the number of cells output
  // on each MeshBlock, i.e. (ois-ois+1), etc. is the same.
  if (nout_mbs > 0) {
    nout1 = (outmbs[0].oie - outmbs[0].ois + 1);
    nout2 = (outmbs[0].oje - outmbs[0].ojs + 1);
    nout3 = (outmbs[0].oke - outmbs[0].oks + 1);
    // NB: outarray stores all output data on Host
    Kokkos::realloc(outarray, nout_vars, nout_mbs, nout3, nout2, nout1);
  }

  if (out_params.contains_derived) {
    out_params.i_derived = 0;
    ComputeDerivedVariable(out_params.variable, pm);
  }

  // Reuse one staging allocation for all variables and MeshBlocks.  Creating and
  // destroying a device View for every block leaves a long chain of asynchronous
  // allocations around large multi-variable outputs and is needlessly expensive.
  DvceArray3D<Real> d_output_var;
  DvceArray3D<Real>::HostMirror h_output_var;
  if (nout_mbs > 0 && nout_vars > 0) {
    d_output_var = DvceArray3D<Real>("d_out_var", nout3, nout2, nout1);
    h_output_var = Kokkos::create_mirror(d_output_var);
  }

  // Only the ADM components this output actually asks for need staging.  Materializing
  // all 17 for a run that dumps, say, adm_alpha alone wasted 17x the scratch memory and
  // 17x the device-to-host traffic per MeshBlock.
  std::vector<int> adm_slot_component;
  for (int n = 0; n < nout_vars; ++n) {
    if (outvars[n].stream_adm_metric) adm_slot_component.push_back(outvars[n].data_index);
  }
  std::sort(adm_slot_component.begin(), adm_slot_component.end());
  adm_slot_component.erase(
      std::unique(adm_slot_component.begin(), adm_slot_component.end()),
      adm_slot_component.end());
  const int nadm_slots = static_cast<int>(adm_slot_component.size());
  if (nadm_slots > 0) {
    DualArray1D<int> slot_component("adm_slot_component", nadm_slots);
    for (int s = 0; s < nadm_slots; ++s) slot_component.h_view(s) = adm_slot_component[s];
    slot_component.template modify<HostMemSpace>();
    slot_component.template sync<DevExeSpace>();
    auto d_slot = slot_component.d_view;

    DvceArray4D<Real> d_adm("stream_adm_metric", nadm_slots, nout3, nout2, nout1);
    auto h_adm = Kokkos::create_mirror(d_adm);
    const auto metric = pm->pmb_pack->padm->GetMetricView(pm->time);
    for (int m = 0; m < nout_mbs; ++m) {
      const int mbi = pm->FindMeshBlockIndex(outmbs[m].mb_gid);
      if (mbi < 0 || mbi >= pm->pmb_pack->nmb_thispack) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Output MeshBlock GID " << outmbs[m].mb_gid
                  << " is not present in the local MeshBlockPack on rank "
                  << global_variable::my_rank << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const int ois = outmbs[m].ois;
      const int ojs = outmbs[m].ojs;
      const int oks = outmbs[m].oks;
      par_for("stream_analytic_adm_output", DevExeSpace(), 0, nout3 - 1,
              0, nout2 - 1, 0, nout1 - 1,
      KOKKOS_LAMBDA(int ko, int jo, int io) {
        adm::ADMMetricPoint point{};
        metric.CellMetricFull(mbi, oks + ko, ojs + jo, ois + io, point);
        for (int s = 0; s < nadm_slots; ++s) {
          const int v = d_slot(s);
          Real value;
          if (v >= adm::ADM::I_ADM_GXX && v <= adm::ADM::I_ADM_GZZ) {
            value = point.g_dd[v - adm::ADM::I_ADM_GXX];
          } else if (v >= adm::ADM::I_ADM_KXX && v <= adm::ADM::I_ADM_KZZ) {
            value = point.K_dd[v - adm::ADM::I_ADM_KXX];
          } else if (v == adm::ADM::I_ADM_ALPHA) {
            value = point.alpha;
          } else if (v == adm::ADM::I_ADM_PSI4) {
            // The prescribed analytical metric carries no conformal decomposition, so the
            // point value is left at zero on the hot path.  det(gamma)^(1/3) is the
            // natural stand-in and matches what the M1 metric wrapper uses; computing it
            // here costs one cbrt in a cold output kernel instead of in every solver.
            const Real detg = adm::SpatialDet(point.g_dd[S11], point.g_dd[S12],
                                              point.g_dd[S13], point.g_dd[S22],
                                              point.g_dd[S23], point.g_dd[S33]);
            value = (detg > 0.0) ? Kokkos::cbrt(detg) : 0.0;
          } else {
            value = point.beta_u[v - adm::ADM::I_ADM_BETAX];
          }
          d_adm(s, ko, jo, io) = value;
        }
      });
      Kokkos::deep_copy(h_adm, d_adm);
      for (int n = 0; n < nout_vars; ++n) {
        if (!outvars[n].stream_adm_metric) continue;
        const int slot = static_cast<int>(
            std::lower_bound(adm_slot_component.begin(), adm_slot_component.end(),
                             outvars[n].data_index) - adm_slot_component.begin());
        auto h_src = Kokkos::subview(h_adm, slot,
                                     Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto h_dst = Kokkos::subview(outarray, n, m, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(h_dst, h_src);
      }
    }
  }

  // Now copy data to host (outarray) over all variables and MeshBlocks
  for (int n=0; n<nout_vars; ++n) {
    if (outvars[n].stream_adm_metric) continue;
    for (int m=0; m<nout_mbs; ++m) {
      int mbi = pm->FindMeshBlockIndex(outmbs[m].mb_gid);
      if (mbi < 0 || mbi >= pm->pmb_pack->nmb_thispack) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Output MeshBlock GID " << outmbs[m].mb_gid
                  << " is not present in the local MeshBlockPack on rank "
                  << global_variable::my_rank << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::pair<int,int> irange = std::make_pair(outmbs[m].ois, outmbs[m].oie+1);
      std::pair<int,int> jrange = std::make_pair(outmbs[m].ojs, outmbs[m].oje+1);
      std::pair<int,int> krange = std::make_pair(outmbs[m].oks, outmbs[m].oke+1);
      if (outvars[n].stream_ideal_dyn_temperature) {
        auto w0 = *(outvars[n].data_ptr);
        const Real baryon_mass = pm->pmb_pack->pdyngr->BaryonMass();
        const int mlocal = mbi;
        const int ois = outmbs[m].ois;
        const int ojs = outmbs[m].ojs;
        const int oks = outmbs[m].oks;
        par_for("stream_ideal_dyn_temperature_output", DevExeSpace(), 0, nout3 - 1,
                0, nout2 - 1, 0, nout1 - 1,
        KOKKOS_LAMBDA(int ko, int jo, int io) {
          const int i = ois + io;
          const int j = ojs + jo;
          const int k = oks + ko;
          const Real rho = w0(mlocal, IDN, k, j, i);
          const Real press = w0(mlocal, IPR, k, j, i);
          const Real number_density =
              (baryon_mass > 0.0 && isfinite(baryon_mass)) ? (rho / baryon_mass) : rho;
          d_output_var(ko, jo, io) =
              (number_density > 0.0 && press > 0.0) ?
              (press / number_density) : 0.0;
        });
      } else if (outvars[n].stream_grav_phi) {
        const bool has_self_gravity =
            (pm->pmb_pack->pgrav != nullptr && pm->pmb_pack->pgrav->phi_valid);
        bool has_external_bh = false;
        Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
        Real bh_mass = 0.0, bh_softening = 0.0, newton_g = 0.0;
        problem_runtime::GetExternalBHPotential(pm->time, has_external_bh,
                                                bhx, bhy, bhz,
                                                bh_mass, bh_softening, newton_g);
        DvceArray5D<Real> phi;
        if (has_self_gravity) phi = pm->pmb_pack->pgrav->phi;
        auto &mb_size = pm->pmb_pack->pmb->mb_size;
        const int mlocal = mbi;
        const int ois = outmbs[m].ois;
        const int ojs = outmbs[m].ojs;
        const int oks = outmbs[m].oks;
        const Real bh_soft2 = bh_softening*bh_softening;
        par_for("stream_grav_phi_output", DevExeSpace(), 0, nout3 - 1,
                0, nout2 - 1, 0, nout1 - 1,
        KOKKOS_LAMBDA(int ko, int jo, int io) {
          const int i = ois + io;
          const int j = ojs + jo;
          const int k = oks + ko;
          Real phi_tot = has_self_gravity ? phi(mlocal, 0, k, j, i) : 0.0;
          if (has_external_bh) {
            const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                       mb_size.d_view(mlocal).x1min,
                                       mb_size.d_view(mlocal).x1max);
            const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                       mb_size.d_view(mlocal).x2min,
                                       mb_size.d_view(mlocal).x2max);
            const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                       mb_size.d_view(mlocal).x3min,
                                       mb_size.d_view(mlocal).x3max);
            const Real dx = x - bhx;
            const Real dy = y - bhy;
            const Real dz = z - bhz;
            phi_tot -= newton_g*bh_mass/sqrt(dx*dx + dy*dy + dz*dz + bh_soft2);
          }
          d_output_var(ko, jo, io) = phi_tot;
        });
      } else {
        const auto &data = *(outvars[n].data_ptr);
        const bool invalid_index = outvars[n].data_index < 0 ||
            outvars[n].data_index >= data.extent_int(1);
        const bool invalid_range = outmbs[m].ois < 0 || outmbs[m].oie >= data.extent_int(4) ||
            outmbs[m].ojs < 0 || outmbs[m].oje >= data.extent_int(3) ||
            outmbs[m].oks < 0 || outmbs[m].oke >= data.extent_int(2);
        if (mbi >= data.extent_int(0) || invalid_index || invalid_range) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Output variable '" << outvars[n].label
                    << "' references an invalid data view on rank "
                    << global_variable::my_rank << "." << std::endl;
          std::exit(EXIT_FAILURE);
        }
        auto d_slice = Kokkos::subview(*(outvars[n].data_ptr), mbi, outvars[n].data_index,
                                       krange,jrange,irange);
        Kokkos::deep_copy(d_output_var,d_slice);
      }

      // copy device staging View to its reusable host mirror
      Kokkos::deep_copy(h_output_var,d_output_var);

      // copy host mirror to 5D host View containing all output variables
      auto h_slice = Kokkos::subview(outarray,n,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(h_slice,h_output_var);
    }
  }
}
