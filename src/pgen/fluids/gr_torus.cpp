//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_torus.cpp
//! \brief Problem generator to initialize rotational equilibrium tori in GR, using either
//! Fishbone-Moncrief (1976) or Chakrabarti (1985) ICs, specialized for cartesian
//! Kerr-Schild coordinates.  Based on gr_torus.cpp in Athena++, with edits by CJW and SR.
//! Simplified and implemented in Kokkos by JMS.
//!
//! References:
//!    Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//!    Fishbone 1977, ApJ 215 323 (F)
//!    Chakrabarti, S. 1985, ApJ 288, 1

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>  // max(), max_element(), min(), min_element()
#include <array>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <sstream>    // stringstream
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_opacities.hpp"
#include "units/units.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "torus_ic.hpp"

#include <Kokkos_Random.hpp>

// The disk initial condition -- the Fishbone-Moncrief and Chakrabarti tori, the
// Novikov-Thorne thin disk, the vector potential and the parameters that describe them
// -- lives in torus_ic.hpp, shared with the binary generator BBH.cpp.  One problem
// generator is compiled per binary, so importing the namespace here is unambiguous.
using namespace torus_ic;  // NOLINT(build/namespaces)

namespace {
torus_pgen torus;
}  // namespace

// Prototypes for user-defined BCs and history functions
void NoInflowTorus(Mesh *pm);
void TorusFluxes(HistoryData *pdata, Mesh *pm);

// prototype for custom history function
void TorusHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for either Fishbone-Moncrief or Chakrabarti torus in GR
//! Compile with '-D PROBLEM=gr_torus' to enroll as user-specific problem generator
//!  assumes x3 is axisymmetric direction

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR torus problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // User boundary function
  user_bcs_func = NoInflowTorus;
  //user_hist_func = &TorusHistory;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  bool use_dyngr = (pmbp->pdyngr != nullptr);

  // Extract BH parameters
  torus.spin = coord.bh_spin;
  const Real r_excise = coord.rexcise;
  const bool is_radiation_enabled = (pmbp->prad != nullptr);

  // Spherical Grid for user-defined history.  Either radiation module moves the excision
  // out to the horizon (coordinates.cpp), so the innermost sphere must sit outside it.
  auto &grids = spherical_grids;
  const bool radiation_excises_horizon = is_radiation_enabled;
  const Real rflux = (radiation_excises_horizon) ? ceil(r_excise + 1.0)
                                                 : 1.0 + sqrt(1.0 - SQR(torus.spin));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, rflux));
  // NOTE(@pdmullen): Enroll additional radii for flux analysis by
  // pushing back the grids vector with additional SphericalGrid instances
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 12.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 24.0));
  user_hist_func = TorusFluxes;

  if (restart) {
    return;
  }

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Extract radiation parameters if enabled
  int nangles_;
  DualArray2D<Real> nh_c_;
  DvceArray6D<Real> norm_to_tet_, tet_c_, tetcov_c_;
  DvceArray5D<Real> i0_;
  if (is_radiation_enabled) {
    nangles_ = pmbp->prad->prgeo->nangles;
    nh_c_ = pmbp->prad->nh_c;
    norm_to_tet_ = pmbp->prad->norm_to_tet;
    tet_c_ = pmbp->prad->tet_c;
    tetcov_c_ = pmbp->prad->tetcov_c;
    i0_ = pmbp->prad->i0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    torus.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    torus.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = torus.gamma_adi - 1.0;

  // Get Radiation constant (if radiation enabled)
  if (pmbp->prad != nullptr) {
    torus.arad = pmbp->prad->arad;
  }

  const bool is_rad_split = is_radiation_enabled;

  // Read problem-specific parameters from input file
  // global parameters.  <coord>/a is the dimensionless spin, so the hole has unit mass;
  // the shared profiles carry the mass explicitly for the binary generator's sake.
  torus.M = 1.0;
  torus.rho_min = pin->GetReal("problem", "rho_min");
  torus.rho_pow = pin->GetOrAddReal("problem", "rho_pow", 0.0);
  torus.pgas_min = pin->GetReal("problem", "pgas_min");
  torus.pgas_pow = pin->GetOrAddReal("problem", "pgas_pow", 0.0);
  torus.psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * (M_PI/180.0);
  torus.sin_psi = sin(torus.psi);
  torus.cos_psi = cos(torus.psi);
  torus.rho_max = pin->GetReal("problem", "rho_max");
  const bool fm_torus = pin->GetOrAddBoolean("problem", "fm_torus", false);
  const bool chakrabarti_torus =
      pin->GetOrAddBoolean("problem", "chakrabarti_torus", false);
  // The Novikov-Thorne thin disk, selected and described by exactly the keys the binary
  // generator uses, so that one <problem> block means the same disk to both.  Its
  // pressure maximum is not a deck parameter: it is where the Page-Thorne flux function
  // peaks, and is derived below once the spin is known.
  torus.use_nt_disk = pin->GetOrAddBoolean("problem", "use_nt_disk", false);
  torus.use_chakrabarti_torus = chakrabarti_torus;
  const int nmodel = (fm_torus ? 1 : 0) + (chakrabarti_torus ? 1 : 0) +
                     (torus.use_nt_disk ? 1 : 0);
  if (nmodel != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Exactly one of problem/fm_torus, problem/chakrabarti_torus and "
              << "problem/use_nt_disk must be true." << std::endl;
    exit(EXIT_FAILURE);
  }
  torus.r_peak = torus.use_nt_disk ? 0.0 : pin->GetReal("problem", "r_peak");
  torus.n_param = pin->GetOrAddReal("problem", "n_param",0.0);
  torus.prograde = pin->GetOrAddBoolean("problem","prograde",true);
  const Real eos_tfloor = (pmbp->pmhd != nullptr)
      ? pmbp->pmhd->peos->eos_data.tfloor
      : ((pmbp->phydro != nullptr) ? pmbp->phydro->peos->eos_data.tfloor : 0.0);
  ReadNTDiskParameters(pin, torus, torus.use_nt_disk, false,
                       pin->GetOrAddReal("problem", "nt_arad", 0.0),
                       eos_tfloor);

  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

  // excision parameters
  torus.dexcise = coord.dexcise;
  torus.pexcise = coord.pexcise;

  // Compute angular momentum and prepare constants describing primitives
  if (torus.use_nt_disk) {
    // The thin disk carries no enthalpy solution: it rides on circular geodesics, so the
    // reference state is the radius at which the Page-Thorne flux function peaks.
    if (!FindLocalNTProfilePeakRadius(torus, &torus.r_peak)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Could not find a valid NT thin-disk profile peak. Check nt_r_trunc, "
                << "nt_r_out, prograde and the black hole spin." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!(torus.nt_r_trunc > 0.0) || !(torus.nt_h_over_r > 0.0) ||
        !(torus.nt_r_out > torus.r_peak) || !(torus.r_peak > torus.nt_r_trunc) ||
        !(torus.nt_inner_taper_width > 0.0) || !(torus.nt_outer_taper_width > 0.0) ||
        !(torus.rho_max > torus.rho_min)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Invalid NT thin-disk parameters. Require nt_r_trunc>0, "
                << "nt_h_over_r>0, nt_r_out>r_peak>nt_r_trunc, nt_inner_taper_width>0, "
                << "nt_outer_taper_width>0, and rho_max>rho_min." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "NT thin disk r_peak = " << torus.r_peak << std::endl;
    }
    torus.r_outer_edge = torus.nt_r_out;
  } else {
    if (fm_torus) {
      torus.l_peak = CalculateFMLFromRPeak(torus, torus.r_peak);
    } else {
      CalculateCN(torus, &torus.c_param, &torus.n_param);
      torus.l_peak = CalculateL(torus, torus.r_peak, 1.0);
    }
    // Common to both tori:
    torus.log_h_edge = LogHAux(torus, torus.r_edge, 1.0);
    torus.log_h_peak = LogHAux(torus, torus.r_peak, 1.0) - torus.log_h_edge;
    torus.ptot_over_rho_peak = gm1/torus.gamma_adi * (exp(torus.log_h_peak)-1.0);
    torus.rho_peak = pow(torus.ptot_over_rho_peak, 1.0/gm1) / torus.rho_max;
  }

  // find "outer edge" of torus (first place log_h > 0)
  if (!torus.use_nt_disk) {
    Real ra = torus.r_peak;
    Real rb = 2. * ra;
    Real log_h_trial = LogHAux(torus, rb, 1.) - torus.log_h_edge;
    for (int iter=0; iter<10000; ++iter) {
      if (log_h_trial <= 0) {
        break;
      }
      rb *= 2.;
      log_h_trial = LogHAux(torus, rb, 1.) - torus.log_h_edge;
    }
    for (int iter=0; iter<10000; ++iter) {
      if (fabs(ra - rb) < 1.e-3) {
        break;
      }
      Real r_trial = (ra + rb) / 2.;
      if (LogHAux(torus, r_trial, 1.) > torus.log_h_edge) {
        ra = r_trial;
      } else {
        rb = r_trial;
      }
    }
    torus.r_outer_edge = ra;
    std::cout << "Found torus outer edge: " << torus.r_outer_edge << std::endl;
  }

  // initialize primitive variables for new run ---------------------------------------

  const bool torus_split = is_rad_split;
  auto trs = torus;
  // The initial background must be excised on the same sphere the runtime masks use --
  // r_+ with a radiation module, r = 1 without one (coordinates.cpp) -- or the horizon
  // interior starts as ordinary atmosphere and is only floored on the first C2P.
  const Real rexcise_radius = coord.rexcise;
  auto &size = pmbp->pmb->mb_size;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  Real ptotmax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Kokkos::parallel_reduce("pgen_torus1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_ptot) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    // Extract metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);

    // Calculate Boyer-Lindquist coordinates of cell
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);
    Real sin_theta = sin(theta);
    Real cos_theta = cos(theta);
    Real sin_phi = sin(phi);
    Real cos_phi = cos(phi);

    // Account for tilt
    Real sin_vartheta;
    if (trs.psi != 0.0) {
      Real x = sin_theta * cos_phi;
      Real y = sin_theta * sin_phi;
      Real z = cos_theta;
      Real varx = trs.cos_psi * x - trs.sin_psi * z;
      Real vary = y;
      sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    } else {
      sin_vartheta = fabs(sin_theta);
    }

    // Determine if we are in the disk.  The equilibrium tori are bounded by their own
    // enthalpy; the thin disk is reconstructed from the Novikov-Thorne profile instead,
    // which also returns the state, so do that here and reuse it below.
    Real log_h;
    bool in_torus = false;
    Real nt_rho = 0.0, nt_pgas = 0.0;
    Real nt_u0_bl = 0.0, nt_u1_bl = 0.0, nt_u2_bl = 0.0, nt_u3_bl = 0.0;
    if (trs.use_nt_disk) {
      in_torus = ReconstructLocalNTDiskState(trs, x1v, x2v, x3v, &nt_rho, &nt_pgas,
                                             &nt_u0_bl, &nt_u1_bl, &nt_u2_bl, &nt_u3_bl);
    } else if (r >= trs.r_edge) {
      log_h = LogHAux(trs, r, sin_vartheta) - trs.log_h_edge;  // (FM 3.6)
      if (log_h >= 0.0) {
        in_torus = true;
      }
    }

    // Calculate background primitives -- to be consistent with the excision algorithm,
    // we have to recalculate r; we try to avoid excising cells within the horizon which
    // might have a corner sticking out of the horizon.
    Real r_excise, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(trs, x1v + copysign(0.5*dx1,x1v),
                                      x2v + copysign(0.5*dx2,x2v),
                                      x3v + copysign(0.5*dx3,x3v), &r_excise,
                                      &theta_excise, &phi_excise);
    Real rho_bg, pgas_bg;
    if (r_excise > rexcise_radius) {
      rho_bg = trs.rho_min * pow(r, trs.rho_pow);
      pgas_bg = trs.pgas_min * pow(r, trs.pgas_pow);
    } else {
      rho_bg = trs.dexcise;
      pgas_bg = trs.pexcise;
    }

    Real rho = rho_bg;
    Real pgas = pgas_bg;
    Real uu1 = 0.0;
    Real uu2 = 0.0;
    Real uu3 = 0.0;
    Real urad = 0.0;

    Real perturbation = 0.0;
    // Overwrite primitives inside torus
    if (in_torus) {
      // Calculate perturbation
      auto rand_gen = rand_pool64.get_state(); // get random number state this thread
      perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
      rand_pool64.free_state(rand_gen);        // free state for use by other threads

      // Calculate thermodynamic variables
      Real u0_bl, u1_bl, u2_bl, u3_bl;
      if (trs.use_nt_disk) {
        // The thin disk's pressure is its whole support.
        rho = nt_rho;
        pgas = nt_pgas;
        u0_bl = nt_u0_bl; u1_bl = nt_u1_bl; u2_bl = nt_u2_bl; u3_bl = nt_u3_bl;
      } else {
        Real ptot_over_rho = gm1/trs.gamma_adi * (exp(log_h) - 1.0);
        rho = pow(ptot_over_rho, 1.0/gm1) / trs.rho_peak;
        Real temp = ptot_over_rho;
        if (torus_split) temp = CalculateT(trs, rho, ptot_over_rho);
        pgas = temp * rho;

        // Calculate radiation variables (if radiation enabled)
        if (torus_split) urad = trs.arad * SQR(SQR(temp));

        // Calculate velocities in Boyer-Lindquist coordinates
        CalculateVelocityInTiltedTorus(trs, r, theta, phi,
                                       &u0_bl, &u1_bl, &u2_bl, &u3_bl);
      }

      // Transform to preferred coordinates
      Real u0, u1, u2, u3;
      TransformVector(trs, u0_bl, 0.0, u2_bl, u3_bl,
                      x1v, x2v, x3v, &u0, &u1, &u2, &u3);

      uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
      uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
      uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
    }

    // Set primitive values, including random perturbations to pressure
    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    const Real pgas_pert = fmax(pgas, pgas_bg) * (1.0 + perturbation);
    if (!use_dyngr) {
      w0_(m,IEN,k,j,i) = pgas_pert / gm1;
    } else {
      w0_(m,IPR,k,j,i) = pgas_pert;
    }
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

    // Set coordinate frame intensity (if radiation enabled)
    if (is_radiation_enabled) {
      Real q = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2 + 2.0*glower[1][3]*uu1*uu3
             + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
             + glower[3][3]*uu3*uu3;
      Real uu0 = sqrt(1.0 + q);
      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      // Go through each angle
      for (int n=0; n<nangles_; ++n) {
        // Calculate direction in fluid frame
        Real un_t = (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                     u_tet_[3]*nh_c_.d_view(n,3));
        Real n0_f = u_tet_[0]*nh_c_.d_view(n,0) - un_t;

        // Calculate intensity in tetrad frame
        Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
        i0_(m,n,k,j,i) = n0*n_0*(urad/(4.0*M_PI))/SQR(SQR(n0_f));
      }
    }

    // Compute total pressure (equal to gas pressure in non-radiating runs).
    Real ptot;
    if (!use_dyngr) {
      ptot = gm1*w0_(m,IEN,k,j,i);
    } else {
      ptot = w0_(m,IPR,k,j,i);
    }
    if (is_rad_split) ptot += urad/3.0;
    max_ptot = fmax(ptot, max_ptot);
  }, Kokkos::Max<Real>(ptotmax));

  // initialize ADM variables -----------------------------------------

  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    // parse some more parameters from input
    torus.potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
    torus.potential_cutoff   = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);

    torus.is_vertical_field = pin->GetOrAddBoolean("problem", "vertical_field", false);

    // for FM torus:
    // Eqns 33, 34 of arxiv/2202.11721
    //  SANE -> potential_r_pow = 0
    //          potential_falloff = 0  (when 0 -> falloff is disabled)
    //          potential_rho_pow = 1
    //   MAD -> potential_r_pow = 3
    //          potential_falloff = 400
    //          potential_rho_pow = 1
    torus.potential_falloff  = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
    torus.potential_r_pow    = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
    torus.potential_rho_pow  = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);

    // Which initial field to seed.  "torus" is the density-tied loop above (the SANE/MAD
    // family), and is the default so that every existing deck is unchanged.  "inclined"
    // is the inclined large-scale poloidal field of Zanni et al. (2007), in the form
    // written as eq. (1.18) of Dihingia & Fendt, "Thin Accretion Disks in GR-MHD
    // simulations" (arXiv:2404.06140).  Because it is not tied to the density it threads
    // the disk with OPEN field lines -- there is no closed loop -- which is what a
    // Blandford-Payne wind needs.  problem/potential_incl_m is then required: it is the
    // m of that equation, the inclination of the field lines to the disk surface and with
    // it the total flux threading the disk.  Small m bends the lines away from the axis
    // (Dihingia & Fendt run m = 0.1), large m approaches a uniform vertical field.
    torus.potential_inclined =
        (pin->GetOrAddString("problem", "potential_config", "torus") == "inclined");
    torus.potential_incl_m = 0.0;
    // Where the inclined potential is normalized and its switch-on ends: the inner edge
    // of an equilibrium torus, and for the thin disk the end of its inner smoothstep, as
    // BBH.cpp sets it.  The thin disk's switch-on runs in cylindrical radius from
    // nt_r_trunc = r_edge, so ending it on r_edge would make it a step in A_phi, a
    // current sheet on the cylinder R = nt_r_trunc.
    torus.potential_r_norm = (torus.potential_inclined && torus.use_nt_disk)
        ? NTDiskInnerTaperEndRadius(torus) : torus.r_edge;
    if (torus.potential_inclined) {
      torus.potential_incl_m = pin->GetReal("problem", "potential_incl_m");
      if (!(torus.potential_incl_m > 0.0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "problem/potential_incl_m must be positive; m sets the field-line "
                  << "inclination and enters as m^(5/4)." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (torus.is_vertical_field) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "problem/potential_config = inclined and problem/vertical_field = "
                  << "true are two different large-scale field prescriptions; pick one."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;
    auto trs = torus;

    par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = ErodedA1(trs, x1v, x2f, x3f, 0.5*dx2, 0.5*dx3);
      a2(m,k,j,i) = ErodedA2(trs, x1f, x2v, x3f, 0.5*dx1, 0.5*dx3);
      a3(m,k,j,i) = ErodedA3(trs, x1f, x2f, x3v, 0.5*dx1, 0.5*dx2);

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(ErodedA1(trs, xl,x2f,x3f, 0.25*dx2, 0.25*dx3) +
                           ErodedA1(trs, xr,x2f,x3f, 0.25*dx2, 0.25*dx3));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(ErodedA2(trs, x1f,xl,x3f, 0.25*dx1, 0.25*dx3) +
                           ErodedA2(trs, x1f,xr,x3f, 0.25*dx1, 0.25*dx3));
      }

      // Correct A3 at x1-faces, x2-faces, and x1x2-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
          (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
        Real xl = x3v + 0.25*dx3;
        Real xr = x3v - 0.25*dx3;
        a3(m,k,j,i) = 0.5*(ErodedA3(trs, x1f,x2f,xl, 0.25*dx1, 0.25*dx2) +
                           ErodedA3(trs, x1f,x2f,xr, 0.25*dx1, 0.25*dx2));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // find maximum bsq
    Real bsqmax = std::numeric_limits<float>::min();
    Real bsqmax_intorus = std::numeric_limits<float>::min();
    const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
    const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
    const int nji  = indcs.nx2*indcs.nx1;
    // The inclined seed on the thin disk is normalized as BBH.cpp normalizes it:
    // potential_beta_min is the MINIMUM plasma beta 2p/b^2 over the disk's midplane body
    // (NTDiskMidplaneBodyPressure), on the unperturbed support pressure.  The ratio of
    // the in-disk maxima put max(b^2) of the open R^{-5/4} field in the inner taper,
    // where the disk has been removed.  val = max b^2/p, loc = the radius it was found at.
    const bool midplane_beta_norm = torus.potential_inclined && torus.use_nt_disk;
    Kokkos::MaxLoc<Real, Real>::value_type midplane_peak;
    midplane_peak.val = std::numeric_limits<Real>::lowest();
    midplane_peak.loc = -1.0;
    Kokkos::parallel_reduce("torus_beta", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_bsq, Real &max_bsq_intorus,
                  Kokkos::ValLocScalar<Real, Real> &mp_peak) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

      // Extract metric components
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      // Calculate Boyer-Lindquist coordinates of cell
      Real r, theta, phi;
      GetBoyerLindquistCoordinates(trs, x1v, x2v, x3v, &r, &theta, &phi);
      Real sin_theta = sin(theta);
      Real cos_theta = cos(theta);
      Real sin_phi = sin(phi);
      Real cos_phi = cos(phi);

      // Account for tilt
      Real sin_vartheta, cos_vartheta;
      if (trs.psi != 0.0) {
        Real x = sin_theta * cos_phi;
        Real y = sin_theta * sin_phi;
        Real z = cos_theta;
        Real varx = trs.cos_psi * x - trs.sin_psi * z;
        Real vary = y;
        Real varz = trs.sin_psi * x + trs.cos_psi * z;
        sin_vartheta = sqrt(SQR(varx) + SQR(vary));
        cos_vartheta = varz;
      } else {
        sin_vartheta = fabs(sin_theta);
        cos_vartheta = cos_theta;
      }

      // Determine if we are in the disk, by the disk's own density profile -- the
      // Novikov-Thorne one when that is the disk, otherwise FM 3.6 / Chakrabarti, the
      // same test the vector potential uses.  This gate is what the vertical field, and
      // the inclined field on an equilibrium torus, are normalized on, and it must be
      // the disk they thread, not the FM log h evaluated with a thin disk's parameters.
      Real rho_disk = 0.0;
      const bool in_torus = (r >= trs.r_edge) &&
          TorusDensityForPotential(trs, r, sin_vartheta, cos_vartheta, &rho_disk);

      // Extract primitive velocity, magnetic field B^i, and gas pressure
      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wbx = bcc_(m,IBX,k,j,i);
      Real &wby = bcc_(m,IBY,k,j,i);
      Real &wbz = bcc_(m,IBZ,k,j,i);

      // Calculate 4-velocity (exploiting symmetry of metric)
      Real q = glower[1][1]*wvx*wvx +2.0*glower[1][2]*wvx*wvy +2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy +2.0*glower[2][3]*wvy*wvz
             + glower[3][3]*wvz*wvz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = wvx - alpha * lor * gupper[0][1];
      Real u2 = wvy - alpha * lor * gupper[0][2];
      Real u3 = wvz - alpha * lor * gupper[0][3];

      // lower vector indices
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field
      Real b0 = u_1*wbx + u_2*wby + u_3*wbz;
      Real b1 = (wbx + b0 * u1) / u0;
      Real b2 = (wby + b0 * u2) / u0;
      Real b3 = (wbz + b0 * u3) / u0;

      // lower vector indices and compute bsq
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      max_bsq = fmax(bsq, max_bsq);
      if (in_torus) {
        max_bsq_intorus = fmax(bsq, max_bsq_intorus);
      }
      Real pgas_d = 0.0;
      if (midplane_beta_norm && bsq > 0.0 &&
          NTDiskMidplaneBodyPressure(trs, x1v, x2v, x3v, r, cos_vartheta,
                                     size.d_view(m).dx3, &pgas_d) &&
          bsq/pgas_d > mp_peak.val) {
        mp_peak.val = bsq/pgas_d;
        mp_peak.loc = r;
      }
    }, Kokkos::Max<Real>(bsqmax), Kokkos::Max<Real>(bsqmax_intorus),
       Kokkos::MaxLoc<Real, Real>(midplane_peak));

#if MPI_PARALLEL_ENABLED
    // get maximum value of gas pressure and bsq over all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax_intorus, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (midplane_beta_norm) {
      // MPI_MAX returns one of its inputs unchanged, so the rank owning the winning cell
      // recognizes itself and the second reduction carries that cell's radius.
      const Real local_peak = midplane_peak.val;
      MPI_Allreduce(MPI_IN_PLACE, &midplane_peak.val, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      if (local_peak != midplane_peak.val) { midplane_peak.loc = -1.0; }
      MPI_Allreduce(MPI_IN_PLACE, &midplane_peak.loc, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
    }
#endif

    // Apply renormalization of magnetic field
    Real bnorm = sqrt((ptotmax/(0.5*bsqmax))/torus.potential_beta_min);
    if (midplane_beta_norm) {
      if (!(isfinite(midplane_peak.val) && midplane_peak.val > 0.0) ||
          !(isfinite(torus.potential_beta_min) && torus.potential_beta_min > 0.0)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "The inclined seed sampled no midplane cell of the disk body "
                  << "carrying field, so the disk's minimum plasma beta cannot be "
                  << "set. Check that the disk lies inside the mesh and that the "
                  << "mesh resolves its scale height."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      // beta = 2 p / b^2, so the smallest midplane beta is 2/max(b^2/p), and scaling b
      // by bnorm scales b^2 by bnorm^2.
      bnorm = sqrt(2.0/(torus.potential_beta_min*midplane_peak.val));
      if (global_variable::my_rank == 0) {
        std::cout << "gr_torus inclined seed: minimum midplane beta before "
                  << "normalization = " << 2.0/midplane_peak.val
                  << " at r = " << midplane_peak.loc
                  << ", bnorm = " << bnorm << std::endl;
      }
    } else if (torus.is_vertical_field || torus.potential_inclined) {
      // Since vertical field extends beyond torus, normalize based on values in torus
      // (the inclined large-scale field extends beyond it in exactly the same way)
      bnorm = sqrt((ptotmax/(0.5*bsqmax_intorus))/torus.potential_beta_min);
    }

    par_for("pgen_normb0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) *= bnorm;
      b0.x2f(m,k,j,i) *= bnorm;
      b0.x3f(m,k,j,i) *= bnorm;
      if (i==ie) { b0.x1f(m,k,j,i+1) *= bnorm; }
      if (j==je) { b0.x2f(m,k,j+1,i) *= bnorm; }
      if (k==ke) { b0.x3f(m,k+1,j,i) *= bnorm; }
    });

    // Recompute cell-centered magnetic field
    par_for("pgen_normbcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  } else {
    //pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn NoInflowTorus
//  \brief Sets boundary condition on surfaces of computational domain
// FIXME: Boundaries need to be adjusted for DynGRMHD

void NoInflowTorus(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pm->pmb_pack->phydro != nullptr) {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  }
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = u0_.extent_int(1);
  const int nwork = nmb;
  if (nwork <= 0) return;

  // Most LAT bins in a nested torus mesh contain only interior MeshBlocks.  Avoid the
  // expensive GRMHD C2P/P2C boundary sequence for those bins, and identify the exact
  // user faces so conversion kernels do not alter unrelated interior ghost zones.
  std::array<bool, 6> active_user_face{};
  for (int a=0; a<nwork; ++a) {
    const int m = a;
    for (int face=0; face<6; ++face) {
      active_user_face[face] = active_user_face[face] ||
          (mb_bcs.h_view(m,face) == BoundaryFlag::user);
    }
  }
  if (std::none_of(active_user_face.begin(), active_user_face.end(),
                   [](const bool active) { return active; })) {
    return;
  }

  // Determine if radiation is enabled
  const bool is_radiation_enabled = (pm->pmb_pack->prad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }

  // X1-Boundary
  if (active_user_face[BoundaryFace::inner_x1] ||
      active_user_face[BoundaryFace::outer_x1]) {
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nwork-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int a, int k, int j) {
      const int m = a;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
  }
  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of MeshBlocks carrying the corresponding user boundary face.
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x1]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,is-ng,is,0,(n2-1),0,(n3-1));
    }
    if (active_user_face[BoundaryFace::outer_x1]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,ie,ie+ng,0,(n2-1),0,(n3-1));
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x1]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x1);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,is-ng,is,0,(n2-1),0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x1]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x1);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,ie,ie+ng,0,(n2-1),0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x1", DevExeSpace(),0,(nwork-1),0,(nvar-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int a, int n, int k, int j) {
    const int m = a;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,is-i-1) = fmin(0.0,w0_(m,n,k,j,is));
        } else {
          w0_(m,n,k,j,is-i-1) = w0_(m,n,k,j,is);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i=0; i<ng; ++i) {
        if (n==(IVX)) {
          w0_(m,n,k,j,ie+i+1) = fmax(0.0,w0_(m,n,k,j,ie));
        } else {
          w0_(m,n,k,j,ie+i+1) = w0_(m,n,k,j,ie);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = i0_(m,n,k,j,is);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = i0_(m,n,k,j,ie);
        }
      }
    });
  }
  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x1]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    }
    if (active_user_face[BoundaryFace::outer_x1]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x1]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x1);
      peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x1]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x1);
      peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  }

  // X2-Boundary
  if (active_user_face[BoundaryFace::inner_x2] ||
      active_user_face[BoundaryFace::outer_x2]) {
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nwork-1),0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int a, int k, int i) {
      const int m = a;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of MeshBlocks carrying the corresponding user boundary face.
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x2]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    }
    if (active_user_face[BoundaryFace::outer_x2]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x2]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x2);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),js-ng,js,0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x2]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x2);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),je,je+ng,0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x2", DevExeSpace(),0,(nwork-1),0,(nvar-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int a, int n, int k, int i) {
    const int m = a;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,js-j-1,i) = fmin(0.0,w0_(m,n,k,js,i));
        } else {
          w0_(m,n,k,js-j-1,i) = w0_(m,n,k,js,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j=0; j<ng; ++j) {
        if (n==(IVY)) {
          w0_(m,n,k,je+j+1,i) = fmax(0.0,w0_(m,n,k,je,i));
        } else {
          w0_(m,n,k,je+j+1,i) = w0_(m,n,k,je,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = i0_(m,n,k,js,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = i0_(m,n,k,je,i);
        }
      }
    });
  }
  // PrimToCons on X2 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x2]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    }
    if (active_user_face[BoundaryFace::outer_x2]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x2]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x2);
      peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x2]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x2);
      peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  }

  // X3-Boundary
  if (active_user_face[BoundaryFace::inner_x3] ||
      active_user_face[BoundaryFace::outer_x3]) {
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nwork-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int a, int j, int i) {
      const int m = a;
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of MeshBlocks carrying the corresponding user boundary face.
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x3]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks);
    }
    if (active_user_face[BoundaryFace::outer_x3]) {
      pm->pmb_pack->phydro->peos->ConsToPrim(
          u0_,w0_,false,0,(n1-1),0,(n2-1),ke,ke+ng);
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &b0 = pm->pmb_pack->pmhd->b0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x3]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x3);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ks-ng,ks);
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x3]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x3);
      peos->ConsToPrim(u0_,b0,w0_,bcc,false,0,(n1-1),0,(n2-1),ke,ke+ng);
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("noinflow_hydro_x3", DevExeSpace(),0,(nwork-1),0,(nvar-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int a, int n, int j, int i) {
    const int m = a;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ks-k-1,j,i) = fmin(0.0,w0_(m,n,ks,j,i));
        } else {
          w0_(m,n,ks-k-1,j,i) = w0_(m,n,ks,j,i);
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k=0; k<ng; ++k) {
        if (n==(IVZ)) {
          w0_(m,n,ke+k+1,j,i) = fmax(0.0,w0_(m,n,ke,j,i));
        } else {
          w0_(m,n,ke+k+1,j,i) = w0_(m,n,ke,j,i);
        }
      }
    }
  });
  if (is_radiation_enabled) {
    // Set X3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = i0_(m,n,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = i0_(m,n,ke,j,i);
        }
      }
    });
  }
  // PrimToCons on X3 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    if (active_user_face[BoundaryFace::inner_x3]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    }
    if (active_user_face[BoundaryFace::outer_x3]) {
      pm->pmb_pack->phydro->peos->PrimToCons(
          w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    }
  } else if (pm->pmb_pack->pmhd != nullptr) {
    auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
    auto *peos = pm->pmb_pack->pmhd->peos;
    if (active_user_face[BoundaryFace::inner_x3]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x3);
      peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
      peos->ClearUserBoundaryFaceFilter();
    }
    if (active_user_face[BoundaryFace::outer_x3]) {
      peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x3);
      peos->PrimToCons(w0_,bcc0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
      peos->ClearUserBoundaryFaceFilter();
    }
  }
  }

  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void TorusFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nvars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // Calculate conversion for P to e if using DynGRMHD.
  Real to_ien = 1.;
  if (pmbp->pdyngr != nullptr) {
    to_ien = 1.0 / (gamma - 1.);
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = (is_mhd) ? 4 : 3;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "mdot_" + rad_str;
    pdata->label[nflux*g+1] = "edot_" + rad_str;
    pdata->label[nflux*g+2] = "ldot_" + rad_str;
    if (is_mhd) {
      pdata->label[nflux*g+3] = "phi_" + rad_str;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;
    pdata->hdata[nflux*g+2] = 0.0;
    if (is_mhd) pdata->hdata[nflux*g+3] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN)*to_ien;

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

      // Lower vector indices
      Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      // Calculate 4-magnetic field (returns zero if not MHD)
      Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
      Real b1 = (int_bx + b0 * u1) / u0;
      Real b2 = (int_by + b0 * u2) / u0;
      Real b3 = (int_bz + b0 * u3) / u0;

      // compute b_\mu in CKS and b_sq (returns zero if not MHD)
      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real a2 = SQR(spin);
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // compute mass flux
      pdata->hdata[nflux*g+0] += -1.0*int_dn*ur*sqrtmdet*domega;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      pdata->hdata[nflux*g+1] += -1.0*t1_0*sqrtmdet*domega;

      // compute angular momentum flux
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      pdata->hdata[nflux*g+2] += t1_3*sqrtmdet*domega;

      // compute magnetic flux
      if (is_mhd) {
        pdata->hdata[nflux*g+3] += 0.5*fabs(br*u0 - b0*ur)*sqrtmdet*domega;
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
