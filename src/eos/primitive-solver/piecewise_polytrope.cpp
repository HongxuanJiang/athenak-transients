//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file piecewise_polytrope.cpp
//  \brief Implementation of PiecewisePolytrope

#include <math.h>
#include <stdio.h>        // BUFSIZ
#include <string.h>       // snprintf

#include <cstdlib>
#include <iostream>
#include <string>

#include "../../parameter_input.hpp"
#include "piecewise_polytrope.hpp"
#include "unit_system.hpp"

#define MAX_PIECES 7

bool Primitive::PiecewisePolytrope::ReadParametersFromInput(std::string block,
                                                            ParameterInput * pin) {
  UnitSystem mks = MakeMKS();

  Real poly_rmd = mks.MassDensityConversion(eos_units)*pin->GetReal(block,
                                                                    "pwp_poly_rmd");

  double densities[MAX_PIECES+1];
  double gammas[MAX_PIECES+1];
  char cstr[BUFSIZ]; // NOLINT
  int np;
  for (np = 0; np < MAX_PIECES; ++np) {
    snprintf(cstr, BUFSIZ, "pwp_density_pieces_%d", np); // NOLINT
    if (pin->DoesParameterExist(block, std::string(cstr))) {
      densities[np] = mks.MassDensityConversion(eos_units)*pin->GetReal(block,
                      std::string(cstr));

      snprintf(cstr, BUFSIZ, "pwp_gamma_pieces_%d", np); // NOLINT
      gammas[np] = pin->GetReal(block, std::string(cstr));
    } else {
      break;
    }
  }

  Real P0 = densities[1]*pow(densities[1]/poly_rmd, gammas[0] - 1.0);

  // Initialize the EOS
  InitializeFromData(densities, gammas, P0, 1.0, np);

  // A deck key the run would not read is an error: "pwp_gamma_thermal" reads like the
  // rest of the pwp_* family but is not the key this reader takes, and a deck carrying it
  // silently ran on the 5/3 default instead of the value it asked for.  From a restart
  // header it is history and is dropped instead (ParameterInput::RetireDeadParameter).
  if (pin->RetireDeadParameter(block, "pwp_gamma_thermal")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<" << block << ">/pwp_gamma_thermal is read by no code path; the "
              << "piecewise-polytrope thermal index is <" << block << ">/gamma_thermal."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Set the gamma thermal (the default is 5/3)
  if (pin->DoesParameterExist(block, "gamma_thermal")) {
    SetThermalGamma(pin->GetReal(block, "gamma_thermal"));
  }

  return true;
}
