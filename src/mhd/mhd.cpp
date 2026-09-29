//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//! \brief implementation of MHD class constructor and assorted functions

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/lte_table_utils.hpp"
#include "eos/saha_table_utils.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "../mesh/mb_storage.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    b0("B_fc",1,1,1,1),
    bcc0("B_cc",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    coarse_b0("cB_fc",1,1,1,1),
    u1("cons1",1,1,1,1,1),
    b1("B_fc1",1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    lat_correction_mask("mhd_lat_correction_mask",1,1,1,1,1),
    coarse_fofc_mask("mhd_coarse_fofc_mask",1,1,1,1,1),
    dual_vf("dual_vf",1,1,1,1,1),
    dual_etot_max("dual_etot_max",1,1,1,1),
    dual_excise_mask("dual_excise_mask",1,1,1,1),
    efld("efld",1,1,1,1),
    e3x1("e3x1",1,1,1,1),
    e2x1("e2x1",1,1,1,1),
    e1x2("e1x2",1,1,1,1),
    e3x2("e3x2",1,1,1,1),
    e2x3("e2x3",1,1,1,1),
    e1x3("e1x3",1,1,1,1),
    wl3d("wl3d",1,1,1,1,1),
    wr3d("wr3d",1,1,1,1,1),
    bl3d("bl3d",1,1,1,1,1),
    br3d("br3d",1,1,1,1,1),
    wsaved("wsaved",1,1,1,1,1),
    bccsaved("bccsaved",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    fofc_scal("fofc_scal",1,1,1,1,1),
    utest("utest",1,1,1,1,1),
    bcctest("bcctest",1,1,1,1,1),
    pmy_pack(ppack),
    e1_cc("e1_cc",1,1,1,1),
    e2_cc("e2_cc",1,1,1,1),
    e3_cc("e3_cc",1,1,1,1) {
  gr_dt = pin->GetOrAddBoolean("time", "gr_dt", false);
  if (gr_dt && (ppack == nullptr || ppack->pcoord == nullptr ||
                !ppack->pcoord->is_general_relativistic ||
                ppack->pcoord->is_dynamical_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<time>/gr_dt=true currently supports only fixed-spacetime "
              << "GRMHD; dynamical/BBH GR and non-GR cases are unsupported."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  split_recon_chunk_nmb = pin->GetOrAddInteger("mhd", "split_recon_chunk_nmb", 32);
  if (split_recon_chunk_nmb <= 0 || split_recon_chunk_nmb > nmb) {
    split_recon_chunk_nmb = nmb;
  }

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  if (gr_dt && eqn_of_state != "ideal") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<time>/gr_dt=true currently requires <mhd>/eos=ideal."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
      // DynGRMHD uses PrimitiveSolver instead, so use a no-op here.
      peos = new NoOpDynGRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRMHD(ppack, pin);
    } else {
      peos = new IdealMHD(ppack, pin);
    }
    nmhd = 5;

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
                <<"<mhd> eos = isothermal cannot be used with SR/GR"<< std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalMHD(ppack, pin);
      nmhd = 4;
    }

  // tabulated H-only LTE/Saha EOS
  } else if (eqn_of_state.compare("saha_table") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/eos = saha_table cannot be used with SR/GR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new SahaTableMHD(ppack, pin);
    nmhd = 5;

  // tabulated H+He LTE EOS
  } else if (!lte_table_utils::ExpectedTableTypeForEosName(eqn_of_state).empty()) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/eos = " << eqn_of_state
                << " cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new LTETableMHD(ppack, pin);
    nmhd = 5;

  // EOS string not recognized
  } else {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
              <<"<mhd> eos = '"<< eqn_of_state <<"' not implemented"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);
  use_dual_energy = pin->GetOrAddBoolean("mhd", "dual_energy", false);
  if (use_dual_energy) {
    // Two flavours of the same formalism.  Relativity picks which auxiliary is carried:
    // an internal-energy density updated by an operator-split p dV step is a Newtonian
    // construction, while in GR the adiabat kappa = p/rho^Gamma obeys an exact
    // conservation law, d_mu(sqrt(-g) rho kappa u^mu) = 0, and so rides the mass flux
    // with no source term at all.  Special relativity has neither implementation.
    const bool relativistic = pmy_pack->pcoord->is_special_relativistic ||
                              pmy_pack->pcoord->is_general_relativistic ||
                              pmy_pack->pcoord->is_dynamical_relativistic;
    dual_energy_pdv = !relativistic;
    if (dual_energy_pdv && !peos->eos_data.use_e) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/dual_energy requires an EOS with internal energy"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmy_pack->pcoord->is_special_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/dual_energy is implemented for non-relativistic "
                << "and general-relativistic MHD, not for special relativity"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // The auxiliary channel inverts kappa back to a pressure, which only a gamma-law
    // gas can do in closed form.  The tabulated and piecewise-polytropic policies carry
    // an entropy but not this invariant, so refuse rather than advect a quantity the
    // inversion cannot read back.
    // Only the dynamical path can choose a non-gamma-law EOS: every other <mhd>/eos is
    // already refused for SR/GR above, so the fixed-metric path is ideal-gas by
    // construction and needs no test here.
    if (pmy_pack->pcoord->is_dynamical_relativistic &&
        pin->GetOrAddString("mhd", "dyn_eos", "ideal").compare("ideal") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/dual_energy on the dynamical-GR path requires "
                << "<mhd>/dyn_eos = ideal: the auxiliary channel needs the adiabat "
                << "p/rho^Gamma, which only a gamma-law gas inverts in closed form."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    naux = 1;
    dual_energy_idx = nmhd + nscalars;
    dual_energy_needs_init = true;
    dual_energy_eta1 = pin->GetOrAddReal("mhd", "dual_energy_eta1", 1.0e-3);
    dual_energy_eta2 = pin->GetOrAddReal("mhd", "dual_energy_eta2", 1.0e-1);
  }
  nvars = nmhd + nscalars + naux;
  const bool tabulated_eos = peos->eos_data.UsesTabulatedLTE();
  // Primitive prolongation is a requirement of the NON-RELATIVISTIC auxiliary, whose
  // conserved and primitive forms are the same internal-energy density, and of the
  // tabulated EOS, whose inversion can fail on a prolonged conserved state.  The GR
  // adiabat needs neither: it is a per-unit-mass quantity conserved as D*kappa, so
  // prolonging D and D*kappa separately and recovering kappa as their ratio is exactly
  // how passive scalars already cross a coarse-fine boundary, and the ratio is bounded
  // below by sfloor and above by the cell's own energy budget in the inversion.  Asking
  // for primitive prolongation here would forbid LAT on the dynamical-GR path for no
  // reason.
  if (pmy_pack->pmesh->pmr != nullptr && (dual_energy_pdv || tabulated_eos)) {
    pmy_pack->pmesh->pmr->prolong_prims = true;
  }

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","viscosity")) {
    pvisc = new Viscosity("mhd", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Non-ideal MHD (only constructed if needed). Keep the legacy selector while
  // allowing the Ohmic and ambipolar coefficients to enable the module directly.
  if (pin->DoesParameterExist("mhd","ohmic_resistivity") ||
      pin->DoesParameterExist("mhd","eta_ohm") ||
      pin->DoesParameterExist("mhd","eta_ad")) {
    presist = new Resistivity(ppack, pin);
  } else {
    presist = nullptr;
  }

  // Thermal conduction (only constructed if needed)
  if (pin->DoesParameterExist("mhd","conductivity") ||
      pin->DoesParameterExist("mhd","tdep_conductivity")) {
    if (tabulated_eos) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/eos = " << eqn_of_state
                << " does not support thermal conduction because the conduction module "
                << "still assumes gamma-law temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pcond = new Conduction("mhd", ppack, pin);
  } else {
    pcond = nullptr;
  }

  bool gravity_self_gravity =
      (pin->DoesBlockExist("gravity") &&
       pin->GetOrAddBoolean("gravity", "self_gravity", true));
  if (pin->DoesBlockExist("mhd_srcterms") && tabulated_eos) {
    const bool ism_cooling = pin->GetOrAddBoolean("mhd_srcterms", "ism_cooling", false);
    const bool rel_cooling = pin->GetOrAddBoolean("mhd_srcterms", "rel_cooling", false);
    const bool disk_cooling = pin->GetOrAddBoolean("mhd_srcterms", "disk_cooling", false);
    if (ism_cooling || rel_cooling || disk_cooling) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/eos = " << eqn_of_state << " does not support "
                << "mhd_srcterms cooling because those source terms still assume "
                << "gamma-law temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pin->DoesBlockExist("mhd_srcterms") || gravity_self_gravity) {
    psrc = new SourceTerms("mhd_srcterms", ppack, pin);
  }

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");
  time_evolving = (evolution_t.compare("stationary") != 0);

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0,   nmb, nvars, ncells3, ncells2, ncells1);
    Kokkos::realloc(w0,   nmb, nvars, ncells3, ncells2, ncells1);
    if (use_dual_energy) {
      Kokkos::realloc(dual_etot_max, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(dual_excise_mask, nmb, ncells3, ncells2, ncells1);
    }

    // allocate memory for face-centered and cell-centered magnetic fields
    Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_b0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1+1);
    Kokkos::realloc(coarse_b0.x2f, nmb, n_ccells3, n_ccells2+1, n_ccells1);
    Kokkos::realloc(coarse_b0.x3f, nmb, n_ccells3+1, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) and face-centered variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  // dual_vf adds one face payload to flux correction, but is not an additional state
  // variable.  Keep the two capacities explicit so state exchange remains exactly nvars.
  // MHD exchanges use only the standard PackAndSend*/RecvAndUnpack* machinery, so
  // send-side per-neighbor vars payloads are stubbed on multi-rank runs, and the
  // receive slots drop the same-level footprint the rank-packed path never uses
  // (stub_recv_same_vars in bvals.hpp).
  pbval_u->stub_send_vars = true;
  pbval_u->stub_recv_same_vars = true;
  // Same-level flux sends exist only under LAT, where the InitFluxRecv of the same stage
  // (MHD::InitRecv, ResetPreviewFluxCommunication) has armed the rank-packed flux
  // buffers, so the per-neighbour send buffers carry the coarse footprint only
  // (stub_flux_same in bvals.hpp; the hydro precedent, multi-rank only).
  pbval_u->stub_flux_same = (global_variable::nranks > 1);
  pbval_u->InitializeBuffers(nvars, nvars + (dual_energy_pdv ? 1 : 0));
  pbval_b = new MeshBoundaryValuesFC(ppack, pin);
  pbval_b->stub_send_vars = true;
  pbval_b->InitializeBuffers(3);

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, nvars);
    porb_b = new OrbitalAdvectionFC(ppack, pin);
    psbox_u = new ShearingBoxCC(ppack, pin, nvars);
    psbox_b = new ShearingBoxFC(ppack, pin);
  } else {
    porb_u = nullptr;
    porb_b = nullptr;
    psbox_u = nullptr;
    psbox_b = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (time_evolving) {
    // determine if FOFC is enabled.  On the split-recon-rsolver branch the main flux
    // kernels extend their face-normal range by one cell when FOFC is on, so the
    // self-contained first-order flux correction (mhd_fofc.cpp) has the fluxes/EMFs it
    // needs over [is-1,ie+2] etc.
    use_fofc = pin->GetOrAddBoolean("mhd","fofc",false);

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ)+FOFC
      if (use_fofc && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 4 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("mhd","rsolver");
    // Special relativistic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_sr;
        } else if (rsolver.compare("hlld") == 0) {
          rsolver_method = MHD_RSolver::hlld_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_gr;
        } else if (rsolver.compare("hlld") == 0) {
          rsolver_method = MHD_RSolver::hlld_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle;
      // HLLD solver
      } else if (rsolver.compare("hlld") == 0) {
        rsolver_method = MHD_RSolver::hlld;
      // Roe solver
      // } else if (rsolver.compare("roe") == 0) {
      //   rsolver_method = MHD_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (dual_energy_pdv &&
          !(rsolver_method == MHD_RSolver::llf ||
            rsolver_method == MHD_RSolver::hlle ||
            rsolver_method == MHD_RSolver::hlld)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/dual_energy only supports non-relativistic "
                  << "llf/hlle/hlld solvers" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solver
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = MHD_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (use_dual_energy && evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/dual_energy only supports dynamic "
                << "non-relativistic MHD" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Final memory allocations
    {
      // allocate second registers
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,     nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

      // allocate fluxes, electric fields.  The face fluxes and the CornerE scratch
      // arrays live on the band the flux kernels fill (see mhd.hpp): cells
      // [ks-1,ke+1]x[js-1,je+1]x[is-1,ie+1] plus the ie+2 face along each normal.
      const bool multi_d = pmy_pack->pmesh->multi_d;
      const bool three_d = pmy_pack->pmesh->three_d;
      flux_ko = three_d ? indcs.ks-1 : 0;
      flux_jo = multi_d ? indcs.js-1 : 0;
      flux_io = indcs.is-1;
      const int nband1 = indcs.nx1 + 2;                    // cells is-1..ie+1
      const int nband2 = multi_d ? indcs.nx2 + 2 : 1;
      const int nband3 = three_d ? indcs.nx3 + 2 : 1;
      const int nbandf1 = nband1 + 1;                      // faces is-1..ie+2
      const int nbandf2 = multi_d ? nband2 + 1 : ncells2 + 1;
      const int nbandf3 = three_d ? nband3 + 1 : ncells3 + 1;
      Kokkos::realloc(uflx.x1f, nmb, nvars, nband3, nband2, nbandf1);
      Kokkos::realloc(uflx.x2f, nmb, nvars, nband3, nbandf2, nband1);
      Kokkos::realloc(uflx.x3f, nmb, nvars, nbandf3, nband2, nband1);
      if (dual_energy_pdv) {
        Kokkos::realloc(dual_vf.x1f, nmb, 1, nband3, nband2, nbandf1);
        Kokkos::realloc(dual_vf.x2f, nmb, 1, nband3, nbandf2, nband1);
        Kokkos::realloc(dual_vf.x3f, nmb, 1, nbandf3, nband2, nband1);
      }
      Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
      Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
      Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);
      const bool fofc_mask_storage = use_fofc && pmy_pack->pmesh->multilevel;
      if (fofc_mask_storage) {
        Kokkos::realloc(lat_correction_mask, nmb, 1, ncells3, ncells2, ncells1);
        Kokkos::deep_copy(lat_correction_mask, 0.0);
      }

      // allocate scratch arrays for face- and cell-centered E used in CornerE, on the
      // band of the face flux each accompanies (cell-centered: the cell band)
      Kokkos::realloc(e3x1, nmb, nband3, nband2, nbandf1);
      Kokkos::realloc(e2x1, nmb, nband3, nband2, nbandf1);
      Kokkos::realloc(e1x2, nmb, nband3, nbandf2, nband1);
      Kokkos::realloc(e3x2, nmb, nband3, nbandf2, nband1);
      Kokkos::realloc(e2x3, nmb, nbandf3, nband2, nband1);
      Kokkos::realloc(e1x3, nmb, nbandf3, nband2, nband1);
      Kokkos::realloc(e1_cc, nmb, nband3, nband2, nband1);
      Kokkos::realloc(e2_cc, nmb, nband3, nband2, nband1);
      Kokkos::realloc(e3_cc, nmb, nband3, nband2, nband1);

      // allocate global per-face L/R buffers for the split-kernel flux path.
      // Indexed by the GLOBAL cell/face index (m,n,k,j,i), so sized to the full
      // cell range (including ghost zones) in every dimension.  bl/br hold the
      // reconstructed cell-centered magnetic field (3 components).
      const int nrecon_nmb = std::min(nmb, split_recon_chunk_nmb);
      Kokkos::realloc(wl3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(wr3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(bl3d, nrecon_nmb, 3, ncells3, ncells2, ncells1);
      Kokkos::realloc(br3d, nrecon_nmb, 3, ncells3, ncells2, ncells1);

      // allocate array of flags used with FOFC
      if (use_fofc) {
        Kokkos::realloc(fofc,    nmb, ncells3, ncells2, ncells1);
        if (pmy_pack->pcoord->is_dynamical_relativistic) {
          fofc_trial_ko = flux_ko;
          fofc_trial_jo = flux_jo;
          fofc_trial_io = flux_io;
          Kokkos::realloc(utest,   nmb, nvars, nband3, nband2, nband1);
          Kokkos::realloc(bcctest, nmb, 3,    nband3, nband2, nband1);
        } else {
          Kokkos::realloc(utest,   nmb, nvars, ncells3, ncells2, ncells1);
          Kokkos::realloc(bcctest, nmb, 3,    ncells3, ncells2, ncells1);
        }
        Kokkos::deep_copy(fofc, false);
        if (nscalars > 0) {
          Kokkos::realloc(fofc_scal,    nmb, nscalars, ncells3, ncells2, ncells1);
          Kokkos::deep_copy(fofc_scal, false);
        }
      }
    }
  }

  const bool fofc_topology_sync = time_evolving && use_fofc &&
      pmy_pack->pmesh->multilevel;
  if (fofc_topology_sync) {
    pbval_fofc = new MeshBoundaryValuesCC(ppack, pin, false);
    // A mask exchange through the standard CC machinery only: the receive slots drop the
    // same-level footprint the rank-packed path never uses, and no flux routine is ever
    // called on it, so it carries no flux payload.
    pbval_fofc->stub_send_vars = true;
    pbval_fofc->stub_recv_same_vars = true;
    pbval_fofc->InitializeBuffers(1, 0);
    pbval_fofc->PrepareRankPackedVarMetadata(1);
    if (pmy_pack->pmesh->multilevel) {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      const int n_ccells1 = indcs.cnx1 + 2*indcs.ng;
      const int n_ccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
      const int n_ccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
      Kokkos::realloc(coarse_fofc_mask, nmb, 1, n_ccells3, n_ccells2, n_ccells1);
      Kokkos::deep_copy(coarse_fofc_mask, 0.0);
    }
  }

  if (peos->eos_data.hydro_eos == HydroEOSModel::saha_table) {
    saha_table_utils::PrintStartupSummary("mhd", pin, peos->eos_data, use_dual_energy);
  } else if (peos->eos_data.hydro_eos == HydroEOSModel::lte_table) {
    lte_table_utils::PrintStartupSummary("mhd", pin, peos->eos_data, use_dual_energy);
  }
}

//----------------------------------------------------------------------------------------
// destructor

MHD::~MHD() {
  delete pbval_fofc;
  pbval_fofc = nullptr;
  if (psbox_b != nullptr) {delete psbox_b;}
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_b != nullptr) {delete porb_b;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_b;
  delete pbval_u;
  if (psrc!= nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (presist!= nullptr) {delete presist;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

//----------------------------------------------------------------------------------------
// SetSaveWBcc:  set flag to save primitives and cell-centered B field, e.g., for jcon

void MHD::SetSaveWBcc() {
  wbcc_saved = true;
  ResizeMeshBlockStorage(pmy_pack->nmb_thispack);
}

bool MHD::ResizeMeshBlockStorage(int nmb, bool exact, bool allow_shrink) {
  // See the matching comment in Hydro::ResizeMeshBlockStorage.
  if (pbval_u != nullptr) {
    pbval_u->ResizeBuffers(nmb);
  }
  if (pbval_b != nullptr) {
    pbval_b->ResizeBuffers(nmb);
  }
  // The FOFC mask exchange owns its own comm buffers, seeded from the live block count
  // like pbval_u/pbval_b; a refinement that grows the rank-local block count would
  // otherwise write past them.
  if (pbval_fofc != nullptr) {
    pbval_fofc->ResizeBuffers(nmb);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  // cnx2/cnx3, not nx2/nx3: this must be the predicate the constructor sized these
  // arrays with, and they differ at nx2 == 2 or nx3 == 2 where cnx == 1, so the first AMR
  // resize reshaped coarse_u0/coarse_b0 away from the shape the prolongation/restriction
  // indices assume.  Inert at the production 36^3 blocks.
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;

  auto resize5 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3,
      int n4) {
    const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                       view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                       view.extent_int(3) != n3 || view.extent_int(4) != n4);
    if (need) {
      Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0), n1, n2, n3, n4);
    }
    return need;
  };
  auto resize4 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3) {
    const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                       view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                       view.extent_int(3) != n3);
    if (need) {
      Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0), n1, n2, n3);
    }
    return need;
  };

  bool resized = false;
  resized = resize5(u0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  resized = resize5(w0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  resized = resize5(bcc0, nmb, 3, ncells3, ncells2, ncells1) || resized;
  resized = resize4(b0.x1f, nmb, ncells3, ncells2, ncells1 + 1) || resized;
  resized = resize4(b0.x2f, nmb, ncells3, ncells2 + 1, ncells1) || resized;
  resized = resize4(b0.x3f, nmb, ncells3 + 1, ncells2, ncells1) || resized;
  if (use_dual_energy) {
    resized = resize4(dual_etot_max, nmb, ncells3, ncells2, ncells1) || resized;
    resized = resize4(dual_excise_mask, nmb, ncells3, ncells2, ncells1) || resized;
  }

  if (pmy_pack->pmesh->multilevel) {
    resized = resize5(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    resized = resize5(coarse_w0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    resized = resize4(coarse_b0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1 + 1) || resized;
    resized = resize4(coarse_b0.x2f, nmb, n_ccells3, n_ccells2 + 1, n_ccells1) || resized;
    resized = resize4(coarse_b0.x3f, nmb, n_ccells3 + 1, n_ccells2, n_ccells1) || resized;
  }

  bool resized_fofc = false;
  if (time_evolving) {
    resized = resize5(u1, nmb, nvars, ncells3, ncells2, ncells1) || resized;
    resized = resize4(b1.x1f, nmb, ncells3, ncells2, ncells1 + 1) || resized;
    resized = resize4(b1.x2f, nmb, ncells3, ncells2 + 1, ncells1) || resized;
    resized = resize4(b1.x3f, nmb, ncells3 + 1, ncells2, ncells1) || resized;
    // flux band (mhd.hpp), the same extents the constructor allocated
    const bool multi_d = pmy_pack->pmesh->multi_d;
    const bool three_d = pmy_pack->pmesh->three_d;
    const int nband1 = indcs.nx1 + 2;
    const int nband2 = multi_d ? indcs.nx2 + 2 : 1;
    const int nband3 = three_d ? indcs.nx3 + 2 : 1;
    const int nbandf1 = nband1 + 1;
    const int nbandf2 = multi_d ? nband2 + 1 : ncells2 + 1;
    const int nbandf3 = three_d ? nband3 + 1 : ncells3 + 1;
    resized = resize5(uflx.x1f, nmb, nvars, nband3, nband2, nbandf1) || resized;
    resized = resize5(uflx.x2f, nmb, nvars, nband3, nbandf2, nband1) || resized;
    resized = resize5(uflx.x3f, nmb, nvars, nbandf3, nband2, nband1) || resized;
    if (dual_energy_pdv) {
      resized = resize5(dual_vf.x1f, nmb, 1, nband3, nband2, nbandf1) || resized;
      resized = resize5(dual_vf.x2f, nmb, 1, nband3, nbandf2, nband1) || resized;
      resized = resize5(dual_vf.x3f, nmb, 1, nbandf3, nband2, nband1) || resized;
    }
    resized = resize4(efld.x1e, nmb, ncells3 + 1, ncells2 + 1, ncells1) || resized;
    resized = resize4(efld.x2e, nmb, ncells3 + 1, ncells2, ncells1 + 1) || resized;
    resized = resize4(efld.x3e, nmb, ncells3, ncells2 + 1, ncells1 + 1) || resized;
    const bool fofc_mask_storage = use_fofc && pmy_pack->pmesh->multilevel;
    if (fofc_mask_storage) {
      const bool resized_lat_correction_mask = resize5(
          lat_correction_mask, nmb, 1, ncells3, ncells2, ncells1);
      resized = resized_lat_correction_mask || resized;
      if (resized_lat_correction_mask) {
        Kokkos::deep_copy(lat_correction_mask, 0.0);
      }
    }
    if (pbval_fofc != nullptr && pmy_pack->pmesh->multilevel) {
      const bool resized_coarse_fofc = resize5(
          coarse_fofc_mask, nmb, 1, n_ccells3, n_ccells2, n_ccells1);
      resized = resized_coarse_fofc || resized;
      if (resized_coarse_fofc) Kokkos::deep_copy(coarse_fofc_mask, 0.0);
    }
    resized = resize4(e3x1, nmb, nband3, nband2, nbandf1) || resized;
    resized = resize4(e2x1, nmb, nband3, nband2, nbandf1) || resized;
    resized = resize4(e1x2, nmb, nband3, nbandf2, nband1) || resized;
    resized = resize4(e3x2, nmb, nband3, nbandf2, nband1) || resized;
    resized = resize4(e2x3, nmb, nbandf3, nband2, nband1) || resized;
    resized = resize4(e1x3, nmb, nbandf3, nband2, nband1) || resized;
    resized = resize4(e1_cc, nmb, nband3, nband2, nband1) || resized;
    resized = resize4(e2_cc, nmb, nband3, nband2, nband1) || resized;
    resized = resize4(e3_cc, nmb, nband3, nband2, nband1) || resized;
    const int nrecon = nvars;
    const int nrecon_nmb = std::min(nmb, split_recon_chunk_nmb);
    resized = resize5(wl3d, nrecon_nmb, nrecon, ncells3, ncells2, ncells1) || resized;
    resized = resize5(wr3d, nrecon_nmb, nrecon, ncells3, ncells2, ncells1) || resized;
    resized = resize5(bl3d, nrecon_nmb, 3, ncells3, ncells2, ncells1) || resized;
    resized = resize5(br3d, nrecon_nmb, 3, ncells3, ncells2, ncells1) || resized;
    if (use_fofc) {
      resized_fofc = resize4(fofc, nmb, ncells3, ncells2, ncells1) || resized_fofc;
      const bool trial_band = pmy_pack->pcoord->is_dynamical_relativistic;
      const int nt3 = trial_band ? nband3 : ncells3;
      const int nt2 = trial_band ? nband2 : ncells2;
      const int nt1 = trial_band ? nband1 : ncells1;
      resized_fofc = resize5(utest, nmb, nvars, nt3, nt2, nt1) || resized_fofc;
      resized_fofc = resize5(bcctest, nmb, 3, nt3, nt2, nt1) || resized_fofc;
      if (nscalars > 0) {
        resized_fofc = resize5(fofc_scal, nmb, nscalars, ncells3, ncells2,
                               ncells1) || resized_fofc;
      }
      resized = resized || resized_fofc;
    }
  }

  if (wbcc_saved) {
    resized = resize5(wsaved, nmb, nvars, ncells3, ncells2, ncells1) || resized;
    resized = resize5(bccsaved, nmb, 3, ncells3, ncells2, ncells1) || resized;
    const int target = std::max(1, nmb);
    if (static_cast<int>(wbcc_saved_dt.extent(0)) != target) {
      wbcc_saved_dt = DualArray1D<Real>("mhd_wbcc_saved_dt", target);
      resized = true;
    }
    // Saved output state is not part of AMR/load-balance migration.  A zero interval
    // keeps it invalid until every remapped block completes its next fluid update.
    Kokkos::deep_copy(wbcc_saved_dt.d_view, static_cast<Real>(0.0));
    wbcc_saved_dt.template modify<DevExeSpace>();
  }

  if (resized_fofc) {
    Kokkos::deep_copy(fofc, false);
    if (nscalars > 0) {
      Kokkos::deep_copy(fofc_scal, false);
    }
  }
  return resized;
}

} // namespace mhd
