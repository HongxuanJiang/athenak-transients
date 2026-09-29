//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/lte_table_utils.hpp"
#include "eos/saha_table_utils.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "utils/lat_reflux_limiter.hpp"
#include "../mesh/mb_storage.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    u1("cons1",1,1,1,1,1),
    coarse_u1("ccons1",1,1,1,1,1),
    lat_u_stage1("lat_u_stage1",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    wl3d("wl3d",1,1,1,1,1),
    wr3d("wr3d",1,1,1,1,1),
    lat_reflux("lat_reflux",1,1,1,1,1),
    lat_dual_vf_reflux("lat_dual_vf_reflux",1,1,1,1,1),
    lat_reflux_theta("lat_reflux_theta",1,1,1,1,1),
    lat_grav_reflux("lat_grav_reflux",1,1,1,1,1),
    dual_vf("dual_vf",1,1,1,1,1),
    dual_etot_max("dual_etot_max",1,1,1,1),
    dual_excise_mask("dual_excise_mask",1,1,1,1),
    sink_block_indices("hydro_sink_blocks",1),
    utest("utest",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    floor_energy_block("floor_energy_block",1) {
  lat_requested_ = pin->IsLATEnabled();
  // The delayed-reflux registers are read only inside LAT windows, and only on a mesh
  // with coarse/fine faces or with same-level LAT (ApplyLATFluxCorrection,
  // ResetLATFluxCorrection, AccumulateLATCoarseFluxes and the masked RecvFlux all
  // return otherwise), which is the condition MHD allocates its twins under.
  lat_correction_storage_ = lat_requested_ &&
      (ppack->pmesh->multilevel || ppack->pmesh->hydro_lat_same_level);

  // Allocate for the current partition. AMR grows these views through
  // ResizeMeshBlockStorage(), so reserving the hard cap here only inflates the
  // persistent CUDA pool and can make an otherwise valid restart run out of memory.
  int nmb = ppack->nmb_thispack;
  split_recon_chunk_nmb =
      pin->GetOrAddInteger("hydro", "split_recon_chunk_nmb", 32);
  if (split_recon_chunk_nmb <= 0) split_recon_chunk_nmb = std::max(1, nmb);

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRHydro(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRHydro(ppack, pin);
    } else {
      peos = new IdealHydro(ppack, pin);
    }
    nhydro = 5;
  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<hydro>/eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalHydro(ppack, pin);
      nhydro = 4;
    }
  // tabulated H-only LTE/Saha EOS
  } else if (eqn_of_state.compare("saha_table") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = saha_table cannot be used with SR/GR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new SahaTableHydro(ppack, pin);
    nhydro = 5;
  // tabulated H+He LTE EOS
  } else if (!lte_table_utils::ExpectedTableTypeForEosName(eqn_of_state).empty()) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state
                << " cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new LTETableHydro(ppack, pin);
    nhydro = 5;
  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);
  use_dual_energy = pin->GetOrAddBoolean("hydro", "dual_energy", false);
  if (use_dual_energy) {
    // Two flavours of the same formalism, selected by the coordinates as in MHD: an
    // internal-energy density updated by an operator-split p dV step is a Newtonian
    // construction, while in GR the adiabat kappa = p/rho^Gamma obeys the exact
    // conservation law d_mu(sqrt(-g) rho kappa u^mu) = 0 and rides the mass flux with
    // no source term.  Special relativity has neither implementation.
    const bool relativistic = pmy_pack->pcoord->is_special_relativistic ||
                              pmy_pack->pcoord->is_general_relativistic;
    dual_energy_pdv = !relativistic;
    if (dual_energy_pdv && !peos->eos_data.use_e) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy requires an EOS with internal energy"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (pmy_pack->pcoord->is_special_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy is implemented for non-relativistic "
                << "and general-relativistic hydro, not for special relativity"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // The GR adiabat crosses coarse/fine boundaries as D and D*kappa, the way passive
    // scalars do; the primitive round trip carries only the non-relativistic
    // auxiliary and would leave refined cells with a stale D*kappa.
    if (!dual_energy_pdv &&
        pin->DoesParameterExist("mesh_refinement", "prolong_primitives") &&
        pin->GetBoolean("mesh_refinement", "prolong_primitives")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy under GR cannot be combined with "
                << "<mesh_refinement>/prolong_primitives = true: the adiabat is "
                << "prolonged as a conserved passive scalar." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    naux = 1;
    dual_energy_idx = nhydro + nscalars;
    dual_energy_needs_init = true;
    dual_energy_eta1 = pin->GetOrAddReal("hydro", "dual_energy_eta1", 1.0e-3);
    dual_energy_eta2 = pin->GetOrAddReal("hydro", "dual_energy_eta2", 1.0e-1);
  }
  nvars = nhydro + nscalars + naux;
  const bool tabulated_eos = peos->eos_data.UsesTabulatedLTE();
  if (pmy_pack->pmesh->pmr != nullptr && (dual_energy_pdv || tabulated_eos)) {
    // Dual-energy hydro and the tabulated Saha EOS should not prolongate total energy in
    // conserved form across AMR boundaries. Rebuild fine-grid states from primitives so
    // the thermal closure stays consistent after prolongation and regrids.
    pmy_pack->pmesh->pmr->prolong_prims = true;
  }

  // Viscosity (if requested in input file)
  if (pin->DoesParameterExist("hydro","viscosity")) {
    pvisc = new Viscosity("hydro", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Thermal conduction (if requested in input file)
  if (pin->DoesParameterExist("hydro","conductivity") ||
      pin->DoesParameterExist("hydro","tdep_conductivity")) {
    if (tabulated_eos) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state
                << " does not support thermal "
                << "conduction because the conduction module still assumes gamma-law "
                << "temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pcond = new Conduction("hydro", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms are needed either for an explicit source-term block or when
  // gravity/self_gravity is enabled and should couple into hydro automatically.
  bool gravity_self_gravity =
      (pin->DoesBlockExist("gravity") &&
       pin->GetOrAddBoolean("gravity", "self_gravity", true));
  const std::string pgen_name =
      pin->DoesBlockExist("problem") ?
      pin->GetOrAddString("problem", "pgen_name", "none") : "none";
  const bool analytic_bh_source_problem =
      (pgen_name == "tde_external");
  const bool external_bh_gravity_source =
      pin->GetOrAddBoolean("problem", "external_bh_gravity_source",
                           analytic_bh_source_problem);
  if (pin->DoesBlockExist("hydro_srcterms") && tabulated_eos) {
    const bool ism_cooling = pin->GetOrAddBoolean("hydro_srcterms", "ism_cooling", false);
    const bool rel_cooling = pin->GetOrAddBoolean("hydro_srcterms", "rel_cooling", false);
    if (ism_cooling || rel_cooling) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state << " does not support "
                << "hydro_srcterms cooling because those source terms still assume "
                << "gamma-law temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pin->DoesBlockExist("hydro_srcterms") || gravity_self_gravity ||
      external_bh_gravity_source || pin->DoesBlockExist("sink_particles")) {
    psrc = new SourceTerms("hydro_srcterms", ppack, pin);
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
    Kokkos::realloc(u0, nmb, nvars, ncells3, ncells2, ncells1);
    Kokkos::realloc(w0, nmb, nvars, ncells3, ncells2, ncells1);
    if (use_dual_energy) {
      Kokkos::realloc(dual_excise_mask, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(dual_etot_max, nmb, ncells3, ncells2, ncells1);
    }
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
    if (time_evolving) {
      Kokkos::realloc(coarse_u1, nmb, nvars, n_ccells3, n_ccells2, n_ccells1);
    }
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  // Hydro exchanges use only the standard PackAndSendCC/RecvAndUnpackCC machinery,
  // so send-side per-neighbor vars payloads are stubbed on multi-rank runs.
  pbval_u->stub_send_vars = true;
  pbval_u->stub_recv_same_vars = true;
  // Same-level hydro flux corrections exist only under LAT, and LAT always routes them
  // through the rank-packed aggregate buffers, so the per-neighbor send-side flux
  // buffers only ever carry the coarse footprint.  See stub_flux_same in bvals.hpp for
  // the full invariant; PackAndSendFluxCC aborts if it is ever violated.  Only the CC
  // owner may do this -- the FC (EMF) path packs same-level data unconditionally.
  // MULTI-RANK ONLY, and this is a buffer SIZE only -- no packed value, and therefore no
  // answer, depends on it at any rank count.  At nranks==1 every neighbor is on-rank, so
  // PackAndSendFluxCC writes the destination recvbuf and never touches sendbuf.flux at
  // all; but InitFluxRecv returns before arming the rank-packed path there, so its
  // conservative guard (stub_flux_same && lat_enabled && !rank_packed_flux) would fatal
  // a single-rank hydro+multilevel+LAT run over a write that cannot happen.  Opting out
  // of the stub at one rank keeps that run alive and costs only memory nobody is short
  // of at one rank.
  pbval_u->stub_flux_same = (global_variable::nranks > 1);
  pbval_u->InitializeBuffers(nvars, nvars + (dual_energy_pdv ? 1 : 0));

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, nvars);
    psbox_u = new ShearingBoxCC(ppack, pin, nvars);
  } else {
    porb_u = nullptr;
    psbox_u = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (time_evolving) {
    const bool source_coupled_gravity =
        psrc != nullptr && (psrc->self_gravity || psrc->external_bh_gravity);
    ConfigureLATDenseOutput(pmy_pack->pmesh->hydro_lat_same_level &&
                            pmy_pack->pmesh->hydro_lat_same_level_max_ratio > 1 &&
                            !source_coupled_gravity);

    // determine if FOFC is enabled
    use_fofc = pin->GetOrAddBoolean("hydro","fofc",false);

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
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
                << std::endl << "<hydro> reconstruct = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("hydro","rsolver");
    // Special relativistic dynamic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_sr;
        } else if (rsolver.compare("hllc") == 0) {
          rsolver_method = Hydro_RSolver::hllc_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic dynamic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for GR dynamics" << std::endl;
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
        rsolver_method = Hydro_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle;
      // HLLC solver
      } else if (rsolver.compare("hllc") == 0) {
        if (peos->eos_data.use_e) {
          rsolver_method = Hydro_RSolver::hllc;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = hllc cannot be used with "
                    << "the selected EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      // Roe solver
      } else if (rsolver.compare("roe") == 0) {
        if (peos->eos_data.UsesTabulatedLTE()) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "<hydro>/eos = " << eqn_of_state
                    << " does not support <hydro>/rsolver = roe. "
                    << "Use llf, hlle, or hllc." << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (!(peos->eos_data.is_gamma_law) &&
            peos->eos_data.hydro_eos != HydroEOSModel::isothermal) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = roe requires gamma-law or "
                    << "isothermal EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        rsolver_method = Hydro_RSolver::roe;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (dual_energy_pdv &&
          !(rsolver_method == Hydro_RSolver::llf ||
            rsolver_method == Hydro_RSolver::hlle ||
            rsolver_method == Hydro_RSolver::hllc ||
            rsolver_method == Hydro_RSolver::roe)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/dual_energy only supports non-relativistic "
                  << "llf/hlle/hllc/roe solvers"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solvers
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = Hydro_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (use_dual_energy && evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy only supports dynamic "
                << "non-relativistic hydro"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Final memory allocations
    {
      // allocate second registers, fluxes
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      dtnew_eachmb = DualArray1D<Real>("hydro_dtnew_eachmb", std::max(1, nmb));
      sink_block_indices = DualArray1D<int>("hydro_sink_blocks", std::max(1, nmb));
      Kokkos::realloc(u1,       nmb, nvars, ncells3, ncells2, ncells1);
      if (lat_dense_output_enabled) {
        Kokkos::realloc(lat_u_stage1, nmb, nvars, ncells3, ncells2, ncells1);
      }
      Kokkos::realloc(uflx.x1f, nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x2f, nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, nvars, ncells3, ncells2, ncells1);
      const int nrecon_nmb = std::max(1, std::min(nmb, split_recon_chunk_nmb));
      Kokkos::realloc(wl3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(wr3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      if (lat_correction_storage_) {
        Kokkos::realloc(lat_reflux.x1f, nmb, nvars, ncells3, ncells2, 2);
        Kokkos::realloc(lat_reflux.x2f, nmb, nvars, ncells3, 2, ncells1);
        Kokkos::realloc(lat_reflux.x3f, nmb, nvars, 2, ncells2, ncells1);
        Kokkos::realloc(lat_reflux_theta.x1f, nmb, 1, ncells3, ncells2, 2);
        Kokkos::realloc(lat_reflux_theta.x2f, nmb, 1, ncells3, 2, ncells1);
        Kokkos::realloc(lat_reflux_theta.x3f, nmb, 1, 2, ncells2, ncells1);
        Kokkos::deep_copy(lat_reflux_theta.x1f, 1.0);
        Kokkos::deep_copy(lat_reflux_theta.x2f, 1.0);
        Kokkos::deep_copy(lat_reflux_theta.x3f, 1.0);
      }
      // Cleared, not merely left unallocated, without LAT storage: SourceTerms::Gravity
      // keys its gate kernels on this flag, and the 1x1x1x1x1 placeholder would pass its
      // extent test on a one-block pack.
      lat_grav_reflux_allocated = lat_correction_storage_ && (psrc != nullptr) &&
          (psrc->self_gravity || psrc->external_bh_gravity || psrc->sink_gravity);
      if (lat_grav_reflux_allocated) {
        const int ngw = lat_reflux::kNGravWork;
        Kokkos::realloc(lat_grav_reflux.x1f, nmb, ngw, ncells3, ncells2, 2);
        Kokkos::realloc(lat_grav_reflux.x2f, nmb, ngw, ncells3, 2, ncells1);
        Kokkos::realloc(lat_grav_reflux.x3f, nmb, ngw, 2, ncells2, ncells1);
        Kokkos::deep_copy(lat_grav_reflux.x1f, 0.0);
        Kokkos::deep_copy(lat_grav_reflux.x2f, 0.0);
        Kokkos::deep_copy(lat_grav_reflux.x3f, 0.0);
      }
      if (dual_energy_pdv) {
        if (lat_correction_storage_) {
          Kokkos::realloc(lat_dual_vf_reflux.x1f, nmb, 1, ncells3, ncells2, 2);
          Kokkos::realloc(lat_dual_vf_reflux.x2f, nmb, 1, ncells3, 2, ncells1);
          Kokkos::realloc(lat_dual_vf_reflux.x3f, nmb, 1, 2, ncells2, ncells1);
        }
        Kokkos::realloc(dual_vf.x1f, nmb, 1, ncells3, ncells2, ncells1+1);
        Kokkos::realloc(dual_vf.x2f, nmb, 1, ncells3, ncells2+1, ncells1);
        Kokkos::realloc(dual_vf.x3f, nmb, 1, ncells3+1, ncells2, ncells1);
      }

      // allocate array of flags used with FOFC
      if (use_fofc) {
        Kokkos::realloc(fofc,  nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest, nmb, nvars, ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
      }
    }
  }

  if (peos->eos_data.hydro_eos == HydroEOSModel::saha_table) {
    saha_table_utils::PrintStartupSummary("hydro", pin, peos->eos_data, use_dual_energy);
  } else if (peos->eos_data.hydro_eos == HydroEOSModel::lte_table) {
    lte_table_utils::PrintStartupSummary("hydro", pin, peos->eos_data, use_dual_energy);
  }
}

//----------------------------------------------------------------------------------------
// destructor

Hydro::~Hydro() {
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_u;
  if (psrc != nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

bool Hydro::ResizeMeshBlockStorage(int nmb, bool exact, bool allow_shrink) {
  // Boundary buffers follow the same grow-only high-water discipline as the field
  // arrays below.  Growing them here keeps every topology-changing path covered without
  // the callers in mesh_refinement.cpp having to know about them.  The return value is
  // deliberately not folded into `resized`: that flag means the FIELD storage moved.
  if (pbval_u != nullptr) {
    pbval_u->ResizeBuffers(nmb);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
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
  auto realloc5 = [](auto &view, int n0, int n1, int n2, int n3, int n4) {
    const bool need = (view.extent_int(0) != n0 || view.extent_int(1) != n1 ||
                       view.extent_int(2) != n2 || view.extent_int(3) != n3 ||
                       view.extent_int(4) != n4);
    if (need) {
      Kokkos::realloc(view, n0, n1, n2, n3, n4);
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
  auto realloc4 = [](auto &view, int n0, int n1, int n2, int n3) {
    const bool need = (view.extent_int(0) != n0 || view.extent_int(1) != n1 ||
                       view.extent_int(2) != n2 || view.extent_int(3) != n3);
    if (need) {
      Kokkos::realloc(view, n0, n1, n2, n3);
    }
    return need;
  };

  bool resized = false;
  if (time_evolving &&
      static_cast<int>(dtnew_eachmb.extent(0)) != std::max(1, nmb)) {
    dtnew_eachmb = DualArray1D<Real>("hydro_dtnew_eachmb", std::max(1, nmb));
    resized = true;
  }
  if (psrc != nullptr) {
    resized = psrc->ResizeMeshBlockStorage(nmb, exact) || resized;
  }
  resized = resize5(u0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  resized = resize5(w0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  const int nmb_storage = u0.extent_int(0);
  const int sink_capacity = std::max(1, nmb);
  const int old_sink_capacity = static_cast<int>(sink_block_indices.extent(0));
  if (old_sink_capacity < sink_capacity || (exact &&
      old_sink_capacity != sink_capacity)) {
    // AMR may replace this allocation after asynchronous sink kernels have used it.
    DevExeSpace().fence();
    sink_block_indices = DualArray1D<int>("hydro_sink_blocks", sink_capacity);
    resized = true;
  }
  if (use_dual_energy) {
    resized = resize4(dual_excise_mask, nmb, ncells3, ncells2, ncells1) || resized;
    resized = resize4(dual_etot_max, nmb, ncells3, ncells2, ncells1) || resized;
  }

  if (pmy_pack->pmesh->multilevel) {
    resized = resize5(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    resized = resize5(coarse_w0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    const int coarse_nmb_storage = coarse_u0.extent_int(0);
    if (time_evolving) {
      resized = realloc5(coarse_u1, coarse_nmb_storage, nvars, n_ccells3, n_ccells2,
                         n_ccells1) || resized;
    }
  }

  bool resized_fofc = false;
  if (time_evolving) {
    resized = realloc5(u1, nmb_storage, nvars, ncells3, ncells2, ncells1) || resized;
    if (lat_dense_output_enabled) {
      resized = realloc5(lat_u_stage1, nmb_storage, nvars, ncells3, ncells2, ncells1) ||
                resized;
    }
    resized = realloc5(uflx.x1f, nmb_storage, nvars, ncells3, ncells2,
        ncells1) || resized;
    resized = realloc5(uflx.x2f, nmb_storage, nvars, ncells3, ncells2,
        ncells1) || resized;
    resized = realloc5(uflx.x3f, nmb_storage, nvars, ncells3, ncells2,
        ncells1) || resized;
    const int nrecon_nmb =
        std::max(1, std::min(nmb_storage, split_recon_chunk_nmb));
    resized = realloc5(wl3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1) || resized;
    resized = realloc5(wr3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1) || resized;
    if (lat_correction_storage_) {
      resized = realloc5(lat_reflux.x1f, nmb_storage, nvars, ncells3, ncells2,
          2) || resized;
      resized = realloc5(lat_reflux.x2f, nmb_storage, nvars, ncells3, 2,
          ncells1) || resized;
      resized = realloc5(lat_reflux.x3f, nmb_storage, nvars, 2, ncells2,
          ncells1) || resized;
      resized = realloc5(lat_reflux_theta.x1f, nmb_storage, 1, ncells3, ncells2, 2) ||
          resized;
      resized = realloc5(lat_reflux_theta.x2f, nmb_storage, 1, ncells3, 2, ncells1) ||
          resized;
      resized = realloc5(lat_reflux_theta.x3f, nmb_storage, 1, 2, ncells2, ncells1) ||
          resized;
    }
    if (lat_grav_reflux_allocated) {
      const int ngw = lat_reflux::kNGravWork;
      resized = realloc5(lat_grav_reflux.x1f, nmb_storage, ngw, ncells3, ncells2, 2) ||
          resized;
      resized = realloc5(lat_grav_reflux.x2f, nmb_storage, ngw, ncells3, 2, ncells1) ||
          resized;
      resized = realloc5(lat_grav_reflux.x3f, nmb_storage, ngw, 2, ncells2, ncells1) ||
          resized;
    }
    if (dual_energy_pdv) {
      if (lat_correction_storage_) {
        resized = realloc5(lat_dual_vf_reflux.x1f, nmb_storage, 1, ncells3, ncells2,
            2) || resized;
        resized = realloc5(lat_dual_vf_reflux.x2f, nmb_storage, 1, ncells3, 2,
            ncells1) || resized;
        resized = realloc5(lat_dual_vf_reflux.x3f, nmb_storage, 1, 2, ncells2,
            ncells1) || resized;
      }
      resized = realloc5(dual_vf.x1f, nmb_storage, 1, ncells3, ncells2, ncells1 + 1) ||
                resized;
      resized = realloc5(dual_vf.x2f, nmb_storage, 1, ncells3, ncells2 + 1, ncells1) ||
                resized;
      resized = realloc5(dual_vf.x3f, nmb_storage, 1, ncells3 + 1, ncells2, ncells1) ||
                resized;
    }
    if (use_fofc) {
      resized_fofc = realloc4(fofc, nmb_storage, ncells3, ncells2, ncells1) ||
                     resized_fofc;
      resized_fofc = realloc5(utest, nmb_storage, nvars, ncells3, ncells2, ncells1) ||
                     resized_fofc;
      resized = resized || resized_fofc;
    }
  }

  if (resized_fofc) {
    Kokkos::deep_copy(fofc, false);
  }
  return resized;
}

void Hydro::ConfigureLATDenseOutput(bool enabled) {
  lat_dense_output_enabled = enabled;
}

} // namespace hydro
