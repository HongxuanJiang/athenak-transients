//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.cpp
//  \brief implementation of functions in class Driver

#include <iostream>
#include <iomanip>    // std::setprecision()
#include <limits>
#include <algorithm>
#include <cmath>
#include <string> // string

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/narrow_blocks.hpp"
#include "outputs/outputs.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "sink_particles/sink_particles.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "radiation/radiation.hpp"
#include "driver.hpp"
#include "gravity/gravity.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/srcterms.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

constexpr int kHydroLATRebalanceStableWindows = 4;

//! The face-centred twin of SeedGhostsFromInteriorCC (mesh_refinement.hpp): each
//! component's ghost faces take the nearest face of the block's own active range of that
//! component.
void SeedGhostsFromInteriorFC(DvceFaceFld4D<Real> &b, const RegionIndcs &indcs,
                              const int nmb, const bool multi_d, const bool three_d) {
  if (nmb <= 0 || b.x1f.size() == 0) return;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n3 = b.x1f.extent_int(1), n2 = b.x1f.extent_int(2);
  const int n1 = b.x2f.extent_int(3);
  auto x1f = b.x1f, x2f = b.x2f, x3f = b.x3f;
  par_for("seed_ghosts_fc", DevExeSpace(), 0, nmb-1, 0, n3, 0, n2, 0, n1,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int ic = (i < is) ? is : ((i > ie) ? ie : i);
    const int jc = !multi_d ? j : ((j < js) ? js : ((j > je) ? je : j));
    const int kc = !three_d ? k : ((k < ks) ? ks : ((k > ke) ? ke : k));
    if (k < n3 && j < n2) {
      const int i1 = (i < is) ? is : ((i > ie+1) ? ie+1 : i);
      if (i1 != i || jc != j || kc != k) x1f(m,k,j,i) = x1f(m,kc,jc,i1);
    }
    if (k < n3 && i < n1) {
      const int j2 = !multi_d ? j : ((j < js) ? js : ((j > je+1) ? je+1 : j));
      if (ic != i || j2 != j || kc != k) x2f(m,k,j,i) = x2f(m,kc,j2,ic);
    }
    if (j < n2 && i < n1) {
      const int k3 = !three_d ? k : ((k < ks) ? ks : ((k > ke+1) ? ke+1 : k));
      if (ic != i || jc != j || k3 != k) x3f(m,k,j,i) = x3f(m,k3,jc,ic);
    }
  });
}

int HydroLATFactorWorkCount(const Mesh *pm, int active_factor, int max_factor) {
  if (pm == nullptr || !(pm->hydro_lat_metadata_valid) ||
      pm->hydro_lat_factor_eachmb == nullptr) {
    return -1;
  }
  const int requested_factor = std::max(1, active_factor);
  const int capped_max_factor = std::max(requested_factor, std::max(1, max_factor));
  if (requested_factor == capped_max_factor) {
    int count = 0;
    const int metadata_max =
        std::max(capped_max_factor, std::max(1, pm->hydro_lat_sync_factor_current));
    for (int factor=capped_max_factor; factor<=metadata_max; factor*=2) {
      count += pm->HydroLATBinCountForFactor(factor);
      if (factor > (std::numeric_limits<int>::max()/2)) break;
    }
    return count;
  }
  return pm->HydroLATBinCountForFactor(requested_factor);
}

bool HydroLATFactorPresent(const Mesh *pm, int active_factor, int max_factor) {
  const int count = HydroLATFactorWorkCount(pm, active_factor, max_factor);
  return count < 0 || count > 0;
}

bool HydroLATRefreshHasLocalWork(const MeshBlockPack *pmbp) {
  return pmbp != nullptr &&
         (pmbp->lat_nactive_thispack > 0 ||
          pmbp->lat_nboundary_send_thispack > 0);
}

bool HydroLATRefreshNeedsRestriction(MeshBlockPack *pmbp) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || !(pmbp->lat_active_mask_enabled)) {
    return true;
  }
  if (pmbp->lat_nboundary_send_thispack <= 0) return false;
  pmbp->lat_boundary_send_indices.template sync<HostMemSpace>();
  pmbp->lat_send_nghbr.template sync<HostMemSpace>();

  const int nnghbr = pmbp->pmb->nnghbr;
  for (int a=0; a<pmbp->lat_nboundary_send_thispack; ++a) {
    const int m = pmbp->lat_boundary_send_indices.h_view(a);
    const int lev = pmbp->pmb->mb_lev.h_view(m);
    for (int n=0; n<nnghbr; ++n) {
      if (pmbp->lat_send_nghbr.h_view(m,n) == 0) continue;
      const auto &nb = pmbp->pmb->nghbr.h_view(m,n);
      if (nb.gid >= 0 && nb.lev < lev) return true;
    }
  }
  return false;
}

bool HydroLATRefreshNeedsProlongation(MeshBlockPack *pmbp) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || !(pmbp->lat_active_mask_enabled)) {
    return true;
  }
  if (pmbp->lat_nactive_thispack <= 0) return false;
  pmbp->lat_active_indices.template sync<HostMemSpace>();

  const int nnghbr = pmbp->pmb->nnghbr;
  for (int a=0; a<pmbp->lat_nactive_thispack; ++a) {
    const int m = pmbp->lat_active_indices.h_view(a);
    const int lev = pmbp->pmb->mb_lev.h_view(m);
    for (int n=0; n<nnghbr; ++n) {
      const auto &nb = pmbp->pmb->nghbr.h_view(m,n);
      if (nb.gid >= 0 && nb.lev < lev) return true;
    }
  }
  return false;
}

bool HydroLATRefreshNeedsNeighborReceive(MeshBlockPack *pmbp) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || !(pmbp->lat_active_mask_enabled)) {
    return true;
  }
  if (pmbp->lat_nactive_thispack <= 0) return false;
  pmbp->lat_active_indices.template sync<HostMemSpace>();

  const int nnghbr = pmbp->pmb->nnghbr;
  for (int a=0; a<pmbp->lat_nactive_thispack; ++a) {
    const int m = pmbp->lat_active_indices.h_view(a);
    for (int n=0; n<nnghbr; ++n) {
      if (pmbp->pmb->nghbr.h_view(m,n).gid >= 0) return true;
    }
  }
  return false;
}

bool HydroLATRefreshNeedsPhysicalBC(MeshBlockPack *pmbp) {
  if (pmbp == nullptr || pmbp->pmb == nullptr || pmbp->pmesh == nullptr ||
      !(pmbp->lat_active_mask_enabled)) {
    return true;
  }
  if (pmbp->lat_nactive_thispack <= 0 || pmbp->pmesh->strictly_periodic) return false;
  pmbp->lat_active_indices.template sync<HostMemSpace>();

  for (int a=0; a<pmbp->lat_nactive_thispack; ++a) {
    const int m = pmbp->lat_active_indices.h_view(a);
    for (int f=0; f<6; ++f) {
      const BoundaryFlag bc = pmbp->pmb->mb_bcs.h_view(m,f);
      if (bc != BoundaryFlag::block && bc != BoundaryFlag::periodic &&
          bc != BoundaryFlag::shear_periodic) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters
//
// First, define each time-integrator by setting weights for each step of the algorithm
// and the CFL number stability limit when coupled to the single-stage spatial operator.
// Currently, the explicit, multistage time-integrators must be expressed as 2S-type
// algorithms as in Ketcheson (2010) Algorithm 3, which incudes 2N (Williamson) and 2R
// (van der Houwen) popular 2-register low-storage RK methods. The 2S-type integrators
// depend on a bidiagonally sparse Shu-Osher representation; at each stage l:
//
//    U^{l} = a_{l,l-2}*U^{l-2} + a_{l-1}*U^{l-1}
//          + b_{l,l-2}*dt*Div(F_{l-2}) + b_{l,l-1}*dt*Div(F_{l-1}),
//
// where U^{l-1} and U^{l-2} are previous stages and a_{l,l-2}, a_{l,l-1}=(1-a_{l,l-2}),
// and b_{l,l-2}, b_{l,l-1} are weights that are different for each stage and
// integrator. Previous timestep U^{0} = U^n is given, and the integrator solves
// for U^{l} for 1 <= l <= nstages.
//
// The 2x RHS evaluations of Div(F) and source terms per stage is avoided by adding
// another weighted average / caching of these terms each stage. The API and framework
// is extensible to three register 3S* methods, although none are currently implemented.

// Notation: exclusively using "stage", equivalent in lit. to "substage" or "substep"
// (infrequently "step"), to refer to the intermediate values of U^{l} between each
// "timestep" = "cycle" in explicit, multistage methods.

Driver::Driver(ParameterInput *pin, Mesh *pmesh, Real wtlim, Kokkos::Timer* ptimer) :
  tlim(-1.0),
  nlim(-1),
  ndiag(1),
  nmb_updated_(0),
  npart_updated_(0),
  lb_efficiency_(0),
  pwall_clock_(ptimer),
  wall_time(wtlim),
  impl_src("ru",1,1,1,1,1,1) {
  // set time-evolution option (no default)
  {
    std::string evolution_t = pin->GetString("time","evolution");
    if (evolution_t.compare("static") == 0) {
      time_evolution = TimeEvolution::tstatic;  // cannot use 'static' (keyword);
    } else if (evolution_t.compare("kinematic") == 0) {
      time_evolution = TimeEvolution::kinematic;
    } else if (evolution_t.compare("dynamic") == 0) {
      time_evolution = TimeEvolution::dynamic;
    } else {
      std::cout<<"### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
               <<"<hydro> evolution = '"<< evolution_t <<"' not implemented"<< std::endl;
      std::exit(EXIT_FAILURE);
    }
  } // extra brace to limit scope of string

  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution != TimeEvolution::tstatic) {
    integrator = pin->GetOrAddString("time", "integrator", "rk2");
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);
    hydro_subcycle = pin->GetOrAddBoolean("time", "hydro_subcycle", false);
    hydro_subcycle_factor =
        std::max(1, pin->GetOrAddInteger("time", "hydro_subcycle_factor", 1));
    hydro_lat = pin->IsLATEnabled();
    hydro_lat_levels = hydro_lat ?
        std::max(1, std::min(pin->GetOrAddInteger("time", "lat_levels", 1), 20)) :
        1;
    if (hydro_lat) {
      const bool gid_reorder =
          pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true);
      const bool post_rebalance = pin->GetOrAddBoolean(
          "time", "hydro_lat_post_amr_rebalance", gid_reorder);
      if (post_rebalance && !gid_reorder) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "time/hydro_lat_post_amr_rebalance=true requires "
                     "time/hydro_lat_gid_reorder=true"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (integrator == "rk1") {
      // RK1: first-order Runge-Kutta / the forward Euler (FE) method
      nimp_stages = 0;
      nexp_stages = 1;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
      stage_time_frac[0] = 0.0;
    } else if (integrator == "rk2") {
      // Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
      // Optimal (in error bounds) explicit two-stage, second-order SSPRK
      nimp_stages = 0;
      nexp_stages = 2;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/2 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
      stage_time_frac[0] = 0.0;

      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;
      stage_time_frac[1] = 1.0;
    } else if (integrator == "rk3") {
      // SSPRK (3,3): Gottlieb (2009) equation 3.2
      // Optimal (in error bounds) explicit three-stage, third-order SSPRK
      nimp_stages = 0;
      nexp_stages = 3;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/3 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
      stage_time_frac[0] = 0.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;
      stage_time_frac[1] = 1.0;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;
      stage_time_frac[2] = 0.5;
    } else if (integrator == "rk4") {
      // RK4()4[2S] from Table 2 of Ketcheson (2010)
      // Non-SSP, explicit four-stage, fourth-order RK
      // Stability properties are similar to classical (non-SSP) RK4
      // (but ~2x L2 principal error norm).
      // Refer to Colella (2011) for linear stability analysis of constant coeff.
      nimp_stages = 0;
      nexp_stages = 4;

      // Colella (2011) eq 101; 1st order flux is most severe constraint
      cfl_limit = 1.3925;

      // Physical RHS abscissae obtained by propagating the time coordinate
      // through the 2S register recurrence.  These are not the gam0 weights.

      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.193743905974738;
      stage_time_frac[0] = 0.0;

      gam0[1] = 0.121098479554482;
      gam1[1] = 0.721781678111411;
      beta[1] = 0.099279895495783;
      stage_time_frac[1] = 1.193743905974738;

      gam0[2] = -3.843833699660025;
      gam1[2] = 2.121209265338722;
      beta[2] = 1.131678018054042;
      stage_time_frac[2] = 0.431401321780805;

      gam0[3] = 0.546370891121863;
      gam1[3] = 0.198653035682705;
      beta[3] = 0.310665766509336;
      stage_time_frac[3] = 1.0;

      delta[0] = 1.0;
      delta[1] = 0.217683334308543;
      delta[2] = 1.065841341361089;
      delta[3] = 0.0;
    } else if (integrator == "imex2") {
      // IMEX-SSP2(3,2,2): Pareschi & Russo (2005) Table III.
      // two-stage explicit, three-stage implicit, second-order ImEx
      // Note explicit steps identical to RK2
      nimp_stages = 3;
      nexp_stages = 2;
      cfl_limit = 1.0;
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = 1.0;
      stage_time_frac[0] = 0.0;

      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;
      stage_time_frac[1] = 1.0;

      a_twid[0][0] = -1.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;

      a_twid[1][0] = 0.5;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.25;
      a_twid[2][2] = 0.25;
      a_impl = 0.5;
    } else if (integrator == "imex2+") {
      // IMEX(4,3,2): Krapp et al. (2024, arXiv:2310.04435), Eq.30.
      // three-stage explicit, four-stage implicit, second-order ImEx
      // two implicit stages added, adapting Athenak's overall architecture
      // Note explicit steps may not reduce to RK2 based on the parameters chosen
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gamma = 1.707106781186547;   //1+1/sqrt(2)
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = gamma;
      stage_time_frac[0] = 0.0;

      gam0[1] = (2.0*gamma-1.0)/(2.0*gamma*gamma);
      gam1[1] = (1.0-(2.0*gamma-1.0)/(2.0*gamma*gamma));
      beta[1] = 1.0/(2.0*gamma);
      stage_time_frac[1] = gamma;

      gam0[2] = 1.0;
      gam1[2] = 0.0;
      beta[2] = 0.0;
      stage_time_frac[2] = 1.0;

      a_twid[0][0] = 0.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = 0.0;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.0;
      a_twid[2][2] = (1.0-2.0*gamma*gamma)/2.0/gamma;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = 0.0;
      a_twid[3][1] = 0.0;
      a_twid[3][2] = 0.0;
      a_twid[3][3] = 0.0;

      a_impl = gamma;
    } else if (integrator == "imex3") {
      // IMEX-SSP3(4,3,3): Pareschi & Russo (2005) Table VI.
      // three-stage explicit, four-stage implicit, third-order ImEx
      // Note explicit steps identical to RK3
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
      stage_time_frac[0] = 0.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;
      stage_time_frac[1] = 1.0;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;
      stage_time_frac[2] = 0.5;

      Real a = 0.24169426078821;
      Real b = 0.06042356519705;
      Real e = 0.12915286960590;
      a_twid[0][0] = -2.0*a;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = a;
      a_twid[1][1] = 1.0 - 2.0*a;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = b;
      a_twid[2][1] = e - ((1.0-a)/4.0);
      a_twid[2][2] = 0.5 - b - e - 1.25*a;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = (-2.0/3.0)*b;
      a_twid[3][1] = (1.0 - 4.0*e)/6.0;
      a_twid[3][2] = (4.0*(b + e + a) - 1.0)/6.0;
      a_twid[3][3] = 2.0*(1.0 - a)/3.0;
      a_impl = a;
    // Error, unrecognized integrator name.
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "integrator=" << integrator << " not implemented. "
         << "Valid choices are [rk1,rk2,rk3,rk4,imex2,imex3]." << std::endl;
      exit(EXIT_FAILURE);
    }
    // imex2's explicit weights (gam0={1,0.5}, gam1={0,0.5}, beta={1,0.5}) reproduce the
    // rk2 update exactly once CopyCons has set u1=u0 at stage 1.
    integrator_rk2_equiv = (integrator == "rk2") || (integrator == "imex2");
    if (hydro_lat) {
      hydro_subcycle = true;
      hydro_subcycle_factor = 1 << hydro_lat_levels;
    }
    if (!hydro_subcycle || hydro_subcycle_factor <= 1) {
      hydro_subcycle = false;
      hydro_subcycle_factor = 1;
    } else {
      MeshBlockPack *pmbp = pmesh->pmb_pack;
      const bool has_hydro = pmbp != nullptr && pmbp->phydro != nullptr;
      const bool has_mhd = pmbp != nullptr && pmbp->pmhd != nullptr;
      const bool has_sink = pmbp != nullptr && pmbp->psink != nullptr;
      const bool lat_imex_supported =
          hydro_lat && (nimp_stages == 0 || integrator == "imex2");
      bool invalid = (pmbp == nullptr) || (has_hydro == has_mhd) ||
          (pmbp->prad != nullptr) ||
          (pmbp->pz4c != nullptr) ||
          (pmbp->pionn != nullptr) ||
          (pmbp->ppart != nullptr) || (nimp_stages != 0 && !lat_imex_supported);
      if (!hydro_lat) {
        invalid = invalid || !has_hydro || has_mhd;
        if (!invalid && pmbp->pcoord != nullptr) {
          invalid = pmbp->pcoord->is_special_relativistic ||
              pmbp->pcoord->is_general_relativistic ||
              pmbp->pcoord->is_dynamical_relativistic;
        }
      } else if (!invalid && pmbp->pcoord != nullptr) {
        invalid = pmbp->pcoord->is_dynamical_relativistic;
      }
      if (invalid) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << (hydro_lat ?
                      "time/lat requires exactly one HD fluid and does not "
                      "support the legacy radiation module, evolved Z4c metrics, "
                      "ion-neutral, particles, or IMEX "
                      "stages other than imex2." :
                      "time/hydro_subcycle is implemented only for non-relativistic HD "
                      "runs without MHD, radiation, Z4c/dynGR, ion-neutral, particles, "
                      "or IMEX stages.") << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (hydro_lat) {
        hydro::Hydro *phydro = pmbp->phydro;
        bool lat_unsupported = false;
        std::string lat_unsupported_reason;
        if (phydro != nullptr) {
          if (phydro->pvisc != nullptr) {
            lat_unsupported = true;
            lat_unsupported_reason = "hydro viscosity";
          } else if (phydro->pcond != nullptr) {
            lat_unsupported = true;
            lat_unsupported_reason = "hydro conduction";
          } else if (phydro->porb_u != nullptr || phydro->psbox_u != nullptr) {
            lat_unsupported = true;
            lat_unsupported_reason = "orbital advection/shearing box";
          } else if (phydro->psrc != nullptr &&
                     (phydro->psrc->const_accel || phydro->psrc->ism_cooling ||
                      phydro->psrc->rel_cooling || phydro->psrc->disk_cooling ||
                      phydro->psrc->rad_beam)) {
            lat_unsupported = true;
            lat_unsupported_reason = "non-self-gravity hydro source terms";
          }
        }
        if (!lat_unsupported && pmbp->pturb != nullptr) {
          lat_unsupported = true;
          lat_unsupported_reason = "turbulence driving";
        } else if (pmesh->pgen != nullptr && pmesh->pgen->user_srcs &&
                   !(pmesh->pgen->user_srcs_lat_safe)) {
          lat_unsupported = true;
          lat_unsupported_reason = "LAT-unsafe user source terms";
        } else if (pmesh->pgen != nullptr && pmesh->pgen->user_dt_func != nullptr &&
                   !(pmesh->pgen->user_dt_lat_safe)) {
          lat_unsupported = true;
          lat_unsupported_reason = "LAT-unsafe user timestep limits";
        } else if (pmesh->pgen != nullptr && pmesh->pgen->user_bcs &&
                   phydro != nullptr) {
          lat_unsupported = true;
          lat_unsupported_reason = "LAT-unsafe user boundary conditions";
        }
        if (lat_unsupported) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "time/lat is not enabled for "
                    << lat_unsupported_reason << ". This operator needs "
                    << "active-mask and per-MeshBlock timestep support before it can "
                    << "run with LAT." << std::endl;
          std::exit(EXIT_FAILURE);
        }
        SourceTerms *lat_grav_src = ActiveFluidSourceTerms(pmbp);
        if (lat_grav_src != nullptr && lat_grav_src->self_gravity) {
          const bool has_solve_dt = pin->DoesParameterExist("gravity", "solve_dt");
          const Real solve_dt = has_solve_dt ?
              pin->GetReal("gravity", "solve_dt") : static_cast<Real>(0.0);
          if (!(solve_dt > static_cast<Real>(0.0))) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl
                      << "time/lat with self-gravity uses a frozen potential "
                      << "inside each LAT window and requires gravity/solve_dt > 0 "
                      << "to bound the window length." << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
        const bool source_coupled_hydro = lat_grav_src != nullptr &&
            (lat_grav_src->self_gravity || lat_grav_src->external_bh_gravity);
        if (source_coupled_hydro && pmesh->hydro_lat_same_level &&
            pmesh->hydro_lat_neighbor_limiter_mode == 2) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "time/lat_neighbor_limiter must be 'all' or 'hybrid' when "
                    << "time/lat_same_level=true is used with source-coupled "
                    << "hydro such as analytic BH gravity." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      if (hydro_lat && integrator != "rk1" && !integrator_rk2_equiv) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "time/lat delayed refluxing currently supports only rk1 "
                  << "and rk2-equivalent explicit tableaux (rk2, imex2)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const bool union_user_source_safe =
          pmesh->pgen == nullptr || !(pmesh->pgen->user_srcs) ||
          pmesh->pgen->user_srcs_lat_union_safe;
      const bool union_state_fixup_safe =
          pmesh->pgen == nullptr ||
          pmesh->pgen->user_hydro_state_fixup_func == nullptr ||
          pmesh->pgen->user_hydro_state_fixup_lat_union_safe;
      const bool union_fluid_supported =
          pmbp->phydro != nullptr;
      hydro_lat_union_stage1 = hydro_lat && union_fluid_supported &&
                               integrator_rk2_equiv && union_user_source_safe &&
                               union_state_fixup_safe;
      // time/lat_union_stage1 forces the predictor cadence explicitly.  "auto" keeps the
      // capability-derived choice above.  Forcing "true" adds nothing to it: the
      // correctness conditions on the integrator and on user source/fixup hooks are
      // never overridden because the union predictor cannot express them at all.
      const std::string union_stage1_mode =
          pin->GetOrAddString("time", "lat_union_stage1", "auto");
      if (union_stage1_mode == "false" || union_stage1_mode == "0") {
        hydro_lat_union_stage1 = false;
      } else if (union_stage1_mode == "true" || union_stage1_mode == "1") {
        hydro_lat_union_stage1 = hydro_lat && union_fluid_supported &&
                                 integrator_rk2_equiv && union_user_source_safe &&
                                 union_state_fixup_safe;
      } else if (union_stage1_mode != "auto") {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "time/lat_union_stage1 must be 'auto', 'true', or 'false'"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Final corrector publication is redundant until the next common-time refresh:
      // every ghost cell a bin-final state exchange would deliver is rewritten by
      // RefreshHydroLATBoundaries before anything reads it.  Senders pack owned cells
      // only (bvals/buffs_cc.cpp), a due block receives from every neighbour
      // (MeshBlockPack::SetActiveMeshBlocksByLATDueFactors), an inactive block never
      // reads its ghosts, the tick-end reflux / CT / repair kernels write owned cells
      // only (mhd_lat.cpp), dense snapshots are taken at stage 1,
      // and outputs, restarts and AMR run at window ends, where the exchange is never
      // skipped.  The claim covers GHOST zones only.  coarse_u0 is a locally derived
      // owned quantity whose first interior layer the block's own prolongation stencil
      // reads, and the refresh's restriction is gated on a different set of blocks, so
      // MHD::RestrictU runs on the deferred path as well -- see the comment there.
      // Each module decides which of ITS bin-final exchanges qualify
      // (Hydro/MHD::SkipFinalLAT*Exchange): hydro and MHD drop the gas U
      // round trip (MHD keeps the face-field round trip under SMR/AMR, because
      // ProlongateFC rewrites the fine-side active face the endpoint bcc0 is recovered
      // against).
      // The non-relativistic dual-energy synchronization needs current neighbor IEN
      // values (the GR adiabat is resynchronized ahead of the send and rides the
      // ordinary path), an arbitrary
      // user boundary hook may modify owned cells, the hybrid force-free sidecar's
      // request bookkeeping (state_recv_started, weight staging) assumes every exchange
      // it starts completes, so those cases retain the full path.
      const bool user_boundary_may_change_owned =
          pmesh->pgen != nullptr && pmesh->pgen->user_bcs &&
          !(pmesh->strictly_periodic);
      const bool fluid_dual_energy =
          (pmbp->phydro != nullptr && pmbp->phydro->use_dual_energy &&
           pmbp->phydro->dual_energy_pdv);
      const bool defer_explicitly_set =
          pin->DoesParameterExist("time", "hydro_lat_defer_final_exchange");
      const bool defer_user_request = defer_explicitly_set &&
          pin->GetBoolean("time", "hydro_lat_defer_final_exchange");
      // An explicit "false" vetoes the capability-derived choice; an explicit "true" is
      // only a request and never overrides the correctness conditions below.
      // The deferred path leaves one more piece of state behind: the GR recovery's
      // warm-start roots (eos.hpp, c2p_mu_cache) of the ghost cells it does not invert.
      // The refresh's ghost inversion starts from them, and its converged root depends on
      // the start at the 1e-12 level; next to a union-extrapolated coarse ghost that is
      // enough to flip the exact mass-flux sign test of the CT upwind EMF
      // (mhd_corner_e.cpp), which moved the AMR shock tube by 1e-10 in mass.  An eligible
      // run therefore keeps those roots out of the full path too
      // (hydro_lat_final_ghosts_superseded_this_bin), on either cadence.
      hydro_lat_defer_final_exchange_eligible = hydro_lat && union_fluid_supported &&
          !fluid_dual_energy && !user_boundary_may_change_owned &&
          union_state_fixup_safe;
      hydro_lat_defer_final_exchange = hydro_lat_defer_final_exchange_eligible &&
          !(defer_explicitly_set && !defer_user_request);
      pmbp->lat_union_stage1_enabled = hydro_lat_union_stage1;
      // Sink particles under LAT (design N13, investigation item S2-1).  See the
      // lat_sink_driver_cadence comment in driver.hpp: under LAT the once-per-cycle sink
      // operator cannot run from the per-bin "before_timeintegrator" list, so the task
      // no-ops itself and Driver::Execute drives it at the synchronized window boundary.
      // The structural half of N13 -- factor-1 pinning of every block that intersects a
      // sink accretion sphere grown by one block width -- lives in
      // Mesh::UpdateHydroLATMetadata, which consumes psink->LATPinRegions().
      lat_sink_driver_cadence = hydro_lat && has_sink;
      if (global_variable::my_rank == 0) {
        if (!hydro_lat) {
          std::cout << "Driver: HD subcycling enabled with factor "
                    << hydro_subcycle_factor << std::endl;
        }
        if (hydro_lat) {
          // Which predictor cadence was actually chosen, and -- when it is the slower
          // one -- WHICH capability refused it.  time/lat_union_stage1=auto can fall
          // back to factor-by-factor on any of five conditions, and the fallback is a
          // large performance cliff between two decks that differ only in a physics
          // block; "disabled" on its own leaves an operator no way to tell which.
          std::cout << "Driver: LAT union stage-1 predictor "
                    << (hydro_lat_union_stage1 ? "enabled" : "disabled");
          if (!hydro_lat_union_stage1) {
            const char *why = "explicitly set by time/lat_union_stage1";
            if (!union_fluid_supported)          {why = "fluid not supported";}
            else if (!integrator_rk2_equiv)      {why = "integrator is not RK2-like";}
            else if (!union_user_source_safe)    {why = "pgen user source term";}
            else if (!union_state_fixup_safe)    {why = "pgen user state fixup";}
            std::cout << " (" << why << ")";
          }
          std::cout << std::endl;
        }
        if (hydro_lat && has_sink) {
          std::cout << "Driver: LAT sink cadence = window boundary, sink neighbourhood "
                    << "pinned to factor 1" << std::endl;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Driver::ExecuteTaskList()
//! \brief Perform tasks over all MeshBlocks for the TaskList specified by string "tl".
//! Integer argument "stage" can be used to indicate at which step in overall algorithm
//! these tasks are to be performed, e.g. which stage of a multi-stage RK integrator.

//! The "stagen" list -- and only that list -- runs with problem_runtime's hydro stage
//! time installed.  The consumers of problem_runtime::HydroStageTimeOr() are the pgen
//! BH-step interpolation, the source terms and DynGRMHD::SetADMVariables.  A task that
//! reads it must be queued into "stagen" only if it wants the stage time.
void Driver::ExecuteTaskList(Mesh *pm, std::string tl, int stage) {
  const bool hydro_stage_time_scope = (tl == "stagen") && (stage > 0);
  if (hydro_stage_time_scope) {
    const Real stage_time = (pm->dt > 0.0) ?
        (pm->time + stage_time_frac[stage-1]*pm->dt)
                                           : pm->time;
    problem_runtime::SetHydroStageTime(stage_time);
  }
  MeshBlockPack* pmbp = pm->pmb_pack;
  for (int p=0; p<(pm->nmb_packs_thisrank); ++p) {
    if (!(pmbp->tl_map[tl]->Empty())) {pmbp->tl_map[tl]->Reset();}
  }
  int npack_left = (pm->nmb_packs_thisrank);
  while (npack_left > 0) {
    if (pmbp->tl_map[tl]->Empty()) {
      npack_left--;
    } else {
      if (!pmbp->tl_map[tl]->IsComplete()) {
        auto status = pmbp->tl_map[tl]->DoAvailable(this, stage);
        if (status == TaskListStatus::complete) { npack_left--; }
      }
    }
  }
  if (hydro_stage_time_scope) {
    problem_runtime::ClearHydroStageTime();
  }
  return;
}

int Driver::ConfigureHydroLATSubstep(Mesh *pm, int substep, int active_factor,
                                     int sync_factor, Real fine_dt,
                                     bool preserve_union_times) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const int active_factor_safe = std::max(1, active_factor);
  hydro_lat_active_factor_this_bin = active_factor_safe;
  hydro_lat_final_ghosts_superseded_this_bin =
      hydro_lat_defer_final_exchange_eligible &&
      ((substep + active_factor_safe) < std::max(1, sync_factor));
  hydro_lat_skip_final_exchange_this_bin =
      hydro_lat_defer_final_exchange && hydro_lat_final_ghosts_superseded_this_bin;
  pm->dt = fine_dt*static_cast<Real>(active_factor_safe);
  if (!hydro_lat || pmbp == nullptr) {
    if (pmbp != nullptr) pmbp->SetAllMeshBlocksActive();
    return pm->nmb_total;
  }

  int nactive = pmbp->SetActiveMeshBlocksByLATFactor(
      sync_factor, active_factor);
  if (!preserve_union_times) {
    pmbp->SetActiveLATBlockTimes(pm->time, pm->time + pm->dt);
  }

  // Use cached bin counts to get the global active block count in O(1)
  // instead of scanning the full nmb_total array or paying an MPI_Allreduce.
  if (pm->hydro_lat_metadata_valid) {
    const int global_count =
        HydroLATFactorWorkCount(pm, active_factor_safe, sync_factor);
    if (global_count >= 0) return global_count;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nactive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  return nactive;
}

//----------------------------------------------------------------------------------------
//! \fn void Driver::CheckLATUnionStepDt
//! \brief Verify the invariant every stage-coupled per-block history depends on: a
//! block's UNION stage-1 step length lat_step_dt(m) = fine_dt*factor(m) must be the
//! same number as the scalar pmesh->dt of the factor corrector that later consumes it
//! (fine_dt*active_factor, ConfigureHydroLATSubstep above).  M1 normalizes each
//! implicit-stage rate in imex_source_rate by a_impl*dt when it is recorded and replays
//! it later as a tableau weight times dt (RadiationM1::TimeUpdate_), so a divergence
//! here would mis-scale the IMEX source combination silently -- no crash, no
//! conservation signature.  Both sides are the same product of the same two
//! doubles, so the comparison is exact; the tolerance only guards a future
//! reassociation of that product.
//!
//! Called from the union corrector once per factor bin.  Always active in debug builds;
//! in release builds it runs under time/lat_diagnostics so the invariant can be
//! exercised on real GPU runs (one small reduction per bin per tick).

void Driver::CheckLATUnionStepDt(Mesh *pm) {
  if (pm == nullptr || pm->pmb_pack == nullptr) return;
#ifdef NDEBUG
  if (!pm->hydro_lat_diagnostics) return;
#endif
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (!pmbp->lat_active_mask_enabled) return;
  const int nactive = pmbp->lat_nactive_thispack;
  if (nactive <= 0) return;
  auto active_indices = pmbp->lat_active_indices.d_view;
  auto step_dt = pmbp->lat_step_dt.d_view;
  const Real bin_dt = pm->dt;
  Real worst = 0.0;
  Kokkos::parallel_reduce("lat_union_step_dt_check",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nactive),
      KOKKOS_LAMBDA(const int a, Real &mismatch) {
        const Real d = fabs(step_dt(active_indices(a)) - bin_dt);
        mismatch = (d > mismatch) ? d : mismatch;
      }, Kokkos::Max<Real>(worst));
  if (worst > 64.0*std::numeric_limits<Real>::epsilon()*std::abs(bin_dt)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "LAT union stage-1 step dt does not match the corrector bin dt: "
              << "max|lat_step_dt - pmesh->dt| = " << worst << " with bin dt "
              << bin_dt << ".  The stage-coupled M1 imex source history is saved "
              << "against the former and consumed against the latter." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms *Driver::ActiveFluidSourceTerms
//! \brief Returns the SourceTerms of the fluid that carries the gravity source term.
//!
//! Hydro and MHD share one Poisson potential.  MGGravityDriver::Solve loads its source
//! density from phydro->u0 when hydro exists and from pmhd->u0 otherwise
//! (src/gravity/mg_gravity.cpp:717-730), SourceTerms::Gravity picks the Godunov mass flux
//! the same way (src/srcterms/srcterms.cpp:1040-1052), and Mesh::SetHydroLATFactors
//! already selects the LAT per-block source timestep with the same rule
//! (src/mesh/mesh.cpp:903).  Every gravity-under-LAT rule is therefore expressed once,
//! against whichever fluid is active, instead of once per fluid.
//!
//! Every caller is inside the LAT branch of Driver::Driver, which has already refused a
//! pack carrying both fluids: `has_hydro == has_mhd` and `pionn != nullptr` are both in
//! the `invalid` expression (:504-511).  So "hydro first" is a total rule here, not a
//! preference over MHD.  A two-fluid gravity mode would need a Poisson source summed
//! over both fluids and per-fluid window bookkeeping; it is out of scope, and the
//! existing one-fluid refusal already states it, so no second message is added.

SourceTerms *Driver::ActiveFluidSourceTerms(MeshBlockPack *pmbp) {
  if (pmbp == nullptr) return nullptr;
  if (pmbp->phydro != nullptr) return pmbp->phydro->psrc;
  return nullptr;
}

//----------------------------------------------------------------------------------------
void Driver::ClearHydroLAT(Mesh *pm) {
  hydro_lat_union_stage1_active = false;
  hydro_lat_active_factor_this_bin = 1;
  hydro_lat_skip_final_exchange_this_bin = false;
  hydro_lat_final_ghosts_superseded_this_bin = false;
  if (pm != nullptr && pm->pmb_pack != nullptr) {
    pm->pmb_pack->lat_union_pending_below_factor = 0;
    pm->pmb_pack->SetAllMeshBlocksActive();
  }
}

//----------------------------------------------------------------------------------------
void Driver::RebuildHydroLATMetadata(Mesh *pm) {
  pm->UpdateHydroLATMetadata(hydro_subcycle_factor);
}

bool Driver::RebalanceHydroLATMesh(Mesh *pm, ParameterInput *pin) {
  if (!hydro_lat || pm == nullptr || pin == nullptr || !pm->multilevel ||
      pm->pmr == nullptr || !pm->hydro_lat_metadata_valid ||
      pm->HydroLATLoadBalanceCurrent() ||
      !pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true)) {
    return false;
  }

  if (global_variable::nranks <= 1 || pm->hydro_lat_sync_factor_current <= 1) {
    pm->StampHydroLATLoadBalanceVersions();
    pm->hydro_lat_lb_last_attempt_cycle = pm->ncycle;
    return false;
  }

  ClearHydroLAT(pm);
  // Stamp the attempt before the repartition, on the path that performs one as well as on
  // the early returns above.  RedistAndRefineMeshBlocks stamps topology_last_change_cycle
  // itself, so a rebalance that did not record its own attempt would look to the caller
  // exactly like a fresh regrid and re-arm the trigger it just consumed.
  pm->hydro_lat_lb_last_attempt_cycle = pm->ncycle;
  const std::uint64_t old_topology_version = pm->topology_version;
  pm->pmr->RedistAndRefineMeshBlocks(pin, 0, 0);
  if (pm->topology_version == old_topology_version) return false;

  ClearHydroLAT(pm);
  // The pure rebalance moved every cell of the evolved arrays -- ghost zones included --
  // and the fluid primitives with them
  // (MeshRefinement::lb_full_block_transfer), so the repartitioned pack holds the
  // synchronized end-of-window state verbatim, and nothing is re-derived from it here.
  // Re-deriving is not a fixed point of that state: a ghost exchange followed by
  // ConsToPrim over the pack writes floored/limited conserved variables back and returns
  // primitives at solver tolerance.  That moves the run off the trajectory of the run
  // that never repartitioned (a 1-rank run never does).  Every per-block quantity the
  // next window reads before rewriting it is migrated; the coarse restriction buffers
  // are the one derived quantity rebuilt
  // before the task lists refill them, and they are an exact function of the migrated
  // interiors.
  // The rank-packed MPI boundary layouts of the new decomposition are otherwise built
  // by each module's first unmasked InitRecv, a collective every rank must enter with
  // all blocks active; the first exchange after the rebalance runs under a LAT mask, so
  // build them here for the same boundary objects that InitRecv touches.
  {
    MeshBlockPack *pmbp = pm->pmb_pack;
    if (pmbp->pz4c != nullptr) {
      pmbp->pz4c->pbval_u->PrepareRankPackedVarMetadata(pmbp->pz4c->nz4c);
      (void) pmbp->pz4c->RestrictU(this, 0);
    }
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->pbval_u->PrepareRankPackedVarMetadata(pmbp->phydro->nvars);
      (void) pmbp->phydro->RestrictU(this, 0);
    }
    if (pmbp->prad != nullptr) {
      pmbp->prad->pbval_i->PrepareRankPackedVarMetadata(pmbp->prad->prgeo->nangles);
      (void) pmbp->prad->RestrictI(this, 0);
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::PrepareRankPackedBoundaryMetadata()
//! \brief Build the rank-packed MPI boundary layouts for every module, with all blocks
//! active.
//!
//! These layouts are otherwise built by each module's first unmasked InitRecv, which is a
//! collective every rank must enter with all of its blocks active.  Two entry points into
//! the evolution reach their first exchange already under a LAT mask, where that
//! collective would deadlock and MeshBoundaryValues::EnsureRankPackedVarMetadata aborts
//! instead: the rank rebalance (which builds them inline, see RebalanceHydroLATMesh) and
//! a verbatim restart, whose skipped startup exchange is what would otherwise have built
//! them.  This helper serves the second case; the rebalance keeps its own inline copy
//! because it interleaves the coarse restrictions that a restart does not need.

void Driver::PrepareRankPackedBoundaryMetadata(Mesh *pm) {
  if (pm == nullptr || pm->pmb_pack == nullptr) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pz4c != nullptr) {
    pmbp->pz4c->pbval_u->PrepareRankPackedVarMetadata(pmbp->pz4c->nz4c);
  }
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->pbval_u->PrepareRankPackedVarMetadata(pmbp->phydro->nvars);
  }
  if (pmbp->prad != nullptr) {
    pmbp->prad->pbval_i->PrepareRankPackedVarMetadata(pmbp->prad->prgeo->nangles);
  }
}

void Driver::RefreshHydroLATBoundaries(Mesh *pm, Real target_time,
                                       bool apply_state_fixup, bool force_c2p) {
  if (pm == nullptr || pm->pmb_pack == nullptr) return;
  hydro::Hydro *phydro = pm->pmb_pack->phydro;
  if (phydro == nullptr) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (!HydroLATRefreshHasLocalWork(pmbp)) return;

  // Use cached flags from SetActiveMeshBlocks* when the LAT mask is active,
  // avoiding O(nactive*nnghbr) re-scans per refresh call.
  const bool needs_restrict = pmbp->lat_active_mask_enabled ?
      pmbp->lat_cached_needs_restrict : HydroLATRefreshNeedsRestriction(pmbp);
  const bool needs_prolongate = pm->multilevel && (pmbp->lat_active_mask_enabled ?
      pmbp->lat_cached_needs_prolongate : HydroLATRefreshNeedsProlongation(pmbp));
  const bool needs_neighbor_receive = pmbp->lat_active_mask_enabled ?
      pmbp->lat_cached_needs_neighbor_recv : HydroLATRefreshNeedsNeighborReceive(pmbp);
  const bool needs_state_fixup = apply_state_fixup &&
      pm->pgen != nullptr && pm->pgen->user_hydro_state_fixup_func != nullptr;
  const bool needs_physical_bc =
      (pmbp->lat_active_mask_enabled ?
       pmbp->lat_cached_needs_physical_bc : HydroLATRefreshNeedsPhysicalBC(pmbp)) ||
      (pm->pgen != nullptr && pm->pgen->user_bcs);
  const bool needs_state_send = pmbp->lat_nboundary_send_thispack > 0;
  const bool needs_state_recv = needs_neighbor_receive && pmbp->lat_nactive_thispack > 0;
  const bool needs_shearing_box = phydro != nullptr && phydro->psbox_u != nullptr;
  if (!(needs_restrict || needs_state_send || needs_state_recv || needs_shearing_box ||
        needs_physical_bc || needs_prolongate || needs_state_fixup || force_c2p)) {
    return;
  }
  hydro_lat_exchange_time = target_time;
  hydro_lat_exchange_time_set = true;
  problem_runtime::SetHydroStageTime(target_time);
  Real old_user_srcs_time = 0.0;
  bool old_user_srcs_time_valid = false;
  if (needs_state_fixup) {
    old_user_srcs_time = pm->pgen->user_srcs_time;
    old_user_srcs_time_valid = pm->pgen->user_srcs_time_valid;
    pm->pgen->user_srcs_time = target_time;
    pm->pgen->user_srcs_time_valid = true;
  }
  // Post receives before local fixup/restriction so MPI can progress while those kernels
  // run.
  if (needs_state_recv) {
    if (phydro != nullptr) {
      (void) phydro->InitRecv(this, -1);
    }
  }
  if (needs_state_fixup) {
    pm->pgen->user_hydro_state_fixup_func(pm->pmb_pack, target_time);
  }
  if (needs_restrict) {
    if (phydro != nullptr) {
      (void) phydro->RestrictU(this, 0);
    }
  }
  if (needs_state_send) {
    if (phydro != nullptr) {
      (void) phydro->SendU(this, 0);
    }
  }
  if (needs_state_recv) {
    if (phydro != nullptr) {
      (void) phydro->ClearRecv(this, -1);
      (void) phydro->RecvU(this, 0);
    }
  }
  if (needs_shearing_box) {
    (void) phydro->SendU_Shr(this, 0);
    (void) phydro->ClearSend(this, -4);
    (void) phydro->ClearRecv(this, -4);
    (void) phydro->RecvU_Shr(this, 0);
  }
  if (needs_prolongate) {
    if (phydro != nullptr) {
      (void) phydro->Prolongate(this, 0);
    }
  }
  if (needs_physical_bc) {
    if (phydro != nullptr) {
      (void) phydro->ApplyPhysicalBCs(this, 0);
    }
  }
  if (needs_state_fixup) {
    pm->pgen->user_hydro_state_fixup_func(pm->pmb_pack, target_time);
  }
  const bool did_c2p = pmbp->lat_nactive_thispack > 0 &&
      (needs_neighbor_receive || needs_physical_bc || needs_prolongate ||
       needs_state_fixup || force_c2p);
  if (did_c2p) {
    // Outside these cases the refresh only replaced ghost zones, so interior
    // primitives are already current from the owning block's last stage or
    // post-correction recovery and only the ghost bands need recovery.
    const bool ghost_band_c2p =
        !force_c2p && !needs_state_fixup && !needs_shearing_box;
    if (phydro != nullptr) {
      if (ghost_band_c2p) {
        (void) phydro->ConToPrimGhostBands(this, 0);
      } else {
        (void) phydro->ConToPrim(this, 0);
      }
    }
  }
  // The packed send buffers are independent of u0.  Delay their wait until local
  // receive/unpack, boundary, prolongation, and C2P kernels have been submitted.
  if (needs_state_send) {
    if (phydro != nullptr) {
      (void) phydro->ClearSend(this, -1);
    }
  }
  if (needs_state_fixup) {
    pm->pgen->user_srcs_time = old_user_srcs_time;
    pm->pgen->user_srcs_time_valid = old_user_srcs_time_valid;
  }
  problem_runtime::ClearHydroStageTime();
  hydro_lat_exchange_time_set = false;
}

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as setting ghost zones (BCs),
//  outputting ICs, and computing initial time step

void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool res_flag) {
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;

  //---- Step 1.  Set conserved variables in ghost zones for all physics.  A verbatim
  // restart keeps the checkpointed ghost zones (see InitBoundaryValuesAndPrimitives).
  const bool restart_verbatim = res_flag && pmesh->pgen != nullptr &&
                                pmesh->pgen->restart_state_verbatim;
  InitBoundaryValuesAndPrimitives(pmesh, false, restart_verbatim);

  // The startup exchange that Step 1 just skipped is also each module's first unmasked
  // InitRecv, the collective that builds the rank-packed MPI boundary layouts.  Under LAT
  // every later exchange runs masked, so the layouts could never be built and the first
  // one aborts (MeshBoundaryValues::EnsureRankPackedVarMetadata).  Build them here, with
  // all blocks active, exactly as the rank rebalance does for the same reason.
  if (restart_verbatim && hydro_lat) {
    PrepareRankPackedBoundaryMetadata(pmesh);
  }

  //---- Step 1b.  Repeat the AMR pass the writing run ran immediately after producing
  // this checkpoint.  Driver::Execute makes the outputs of a cycle BEFORE it calls
  // MeshRefinement::AdaptiveMeshRefinement, so a checkpoint always carries the mesh the
  // cycle that just ended was integrated ON, never the adapted mesh the writing run
  // carried into its next cycle.  Resuming without that pass integrates the first
  // post-restart cycle on the stale tree, and since the tree is then adapted one cycle
  // late, every AMR event of the resumed run lags the uninterrupted run for the rest of
  // the run (measured on the star-BH deck: the first post-restart history row was 0.4%
  // off in x-momentum and the mesh never caught up).
  //
  // The writing run's cadence anchor is read from the checkpoint (restart.cpp) where it
  // matters, and reconstructed otherwise: MeshRefinement's fresh-run seed
  // (last_amr_call_cycle = ncycle) is the one value that denies the owed pass and puts
  // the whole later cadence one call behind.  A checkpoint Driver::Finalize writes after
  // a wall-clock exit follows that cycle's AMR pass, and its anchor equals ncycle, so no
  // pass is owed.  The ncycle = 0 checkpoint is the other exception with no pass owed:
  // it is written from Step 3 below, before the first cycle and so before any AMR call.
  // The time/cycle/wall-clock test is Driver::Execute's loop predicate: a restart that
  // will take no further step must not adapt, so its outputs stay those of the
  // checkpoint.  The wall clock is read once, here, and Execute starts from the same
  // reading, so this pass and the window-end pass below run iff Execute takes a step.
  if (res_flag && wall_time > 0.) {
    initialize_wall_clock_ = UpdateWallClock();
  }
  if (res_flag && pmesh->adaptive && pmesh->pmr != nullptr &&
      time_evolution != TimeEvolution::tstatic &&
      (pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
      (initialize_wall_clock_ < wall_time)) {
    int amr_check_period = 1;
    if (pmesh->pmr->ncyc_check_amr > 0) {
      amr_check_period = std::max(1, pmesh->pmr->ncyc_check_amr);
    }
    bool amr_due = false;
    if (pmesh->ncycle > 0) {
      if (hydro_lat) {
        // Execute's LAT gate is (ncycle - last_amr_call_cycle) >= period evaluated at a
        // window end, and window ends are sums of variable sync factors, so the writer's
        // AMR calls land on arbitrary cycles, not on the multiples of the period.  The
        // restart writer therefore records its anchor in the checkpoint parameter dump
        // (restart.cpp); with it the writer's predicate is reproduced verbatim.
        if (pin->DoesParameterExist("mesh_refinement", "last_amr_call_cycle")) {
          pmesh->pmr->last_amr_call_cycle =
              pin->GetInteger("mesh_refinement", "last_amr_call_cycle");
          amr_due = ((pmesh->ncycle - pmesh->pmr->last_amr_call_cycle) >=
                     amr_check_period);
        } else {
          // Fallback for checkpoints written before the anchor was recorded: assume the
          // writer's calls fell on the multiples of the period.  Exact for
          // ncycle_check = 1 (every window end is a call); an approximation otherwise.
          const int remainder = pmesh->ncycle % amr_check_period;
          const int cycles_since_amr = (remainder == 0) ? amr_check_period : remainder;
          pmesh->pmr->last_amr_call_cycle = pmesh->ncycle - cycles_since_amr;
          amr_due = (cycles_since_amr >= amr_check_period);
        }
      } else if (pin->DoesParameterExist("mesh_refinement", "last_amr_call_cycle") &&
                 pin->GetInteger("mesh_refinement", "last_amr_call_cycle") >=
                 pmesh->ncycle) {
        // Written by Driver::Finalize after a wall-clock exit: that cycle's AMR pass has
        // already run on the checkpointed tree, so none is owed.
        pmesh->pmr->last_amr_call_cycle = pmesh->ncycle;
      } else {
        // Without LAT, Execute calls AdaptiveMeshRefinement every cycle (on the
        // multiples of the period when subcycling) and lets that function's own cadence
        // gate decide, so the writer's last call was the cycle before this one.
        pmesh->pmr->last_amr_call_cycle = pmesh->ncycle - 1;
        amr_due = (!hydro_subcycle) || ((pmesh->ncycle % amr_check_period) == 0);
      }
    }
    if (amr_due) {
      ClearHydroLAT(pmesh);
      if (pmesh->pmb_pack != nullptr) {
        pmesh->pmb_pack->ReleaseLATCacheMemory();
      }
      pmesh->pmr->AdaptiveMeshRefinement(this, pin);
      ClearHydroLAT(pmesh);
    }
  }

  //---- Step 2.  Compute time step (if problem involves time evolution)
  hydro::Hydro *phydro = pmesh->pmb_pack->phydro;
  radiation::Radiation *prad = pmesh->pmb_pack->prad;
  z4c::Z4c *pz4c = pmesh->pmb_pack->pz4c;
  auto compute_new_timestep = [&]() {
    phydro = pmesh->pmb_pack->phydro;
    pmhd = pmesh->pmb_pack->pmhd;
    prad = pmesh->pmb_pack->prad;
    pz4c = pmesh->pmb_pack->pz4c;
    if (phydro != nullptr) {
      (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
    }
    if (pmhd != nullptr) {
      (void) pmesh->pmb_pack->pmhd->NewTimeStep(this, nexp_stages);
    }
    if (prad != nullptr) {
      (void) pmesh->pmb_pack->prad->NewTimeStep(this, nexp_stages);
    }
    if (pz4c != nullptr) {
      (void) pmesh->pmb_pack->pz4c->NewTimeStep(this, nexp_stages);
    }

    pmesh->NewTimeStep(tlim);
    if (hydro_lat) {
      RebuildHydroLATMetadata(pmesh);
    }
  };

  // A restart that takes no step (nlim or tlim reached at the checkpoint's cycle) leaves
  // the pass to its own restart, exactly as Execute does for a run that stops: the
  // checkpoint Driver::Finalize writes would otherwise follow the pass, and
  // Mesh::NewTimeStep is not idempotent -- it limits dt to twice the stored dt and
  // commits the hybrid force-free window-end classification and transition recovery --
  // so a restart from that file would apply both a second time.  Execute's predicate,
  // with the wall clock Step 1b read.
  const bool restart_steps =
      (pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
      (initialize_wall_clock_ < wall_time);
  if (time_evolution != TimeEvolution::tstatic && (!res_flag || restart_steps)) {
    compute_new_timestep();
  }

  //---- Step 3.  Cycle through output Types and load data / write files.
  // The initial checkpoint is written here, after the first timestep pass, so that its
  // parameter dump records every other stream's cadence as this step leaves it (rst is
  // last in pout_list).  Unlike in-loop and Finalize checkpoints it therefore holds the
  // state after that pass, and a restart from 00000.rst runs the pass a second time.
  bool allow_outputs = true;
  if (pmesh->pgen != nullptr && pmesh->pgen->user_output_gate_func != nullptr) {
    allow_outputs = pmesh->pgen->user_output_gate_func(pmesh);
  }
  if (!res_flag && allow_outputs) { // only write outputs at the beginning of the run
    for (auto &out : pout->pout_list) {
      out->LoadOutputData(pmesh);
      out->WriteOutputFile(pmesh, pin);
    }
  }

  //---- Step 4.  Initialize various counters, timers, etc.
  run_time_.reset();
  nmb_updated_ = 0;

  // allocate memory for stiff source terms with ImEx integrators
  // only implemented for ion-neutral two fluid for now
  ion_neutral::IonNeutral *pionn = pmesh->pmb_pack->pionn;
  if (pionn != nullptr) {
    if (nimp_stages == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "IonNetral MHD can only be run with ImEx integrators."
          << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nmb = std::max((pmesh->pmb_pack->nmb_thispack), (pmesh->nmb_maxperrank));
    auto &indcs = pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(impl_src, nimp_stages, nmb, 8, ncells3, ncells2, ncells1);
  }

  // <gravity>/reciprocity_test: a diagnostic run.  Solve twice, print the reciprocity
  // sums, and stop before any evolution.
  if (pmesh->pmb_pack != nullptr && pmesh->pmb_pack->pgrav != nullptr &&
      pmesh->pmb_pack->pgrav->pmgd != nullptr &&
      pmesh->pmb_pack->pgrav->pmgd->ReciprocityTestRequested()) {
    pmesh->pmb_pack->pgrav->pmgd->RunReciprocityTest(this);
    Kokkos::fence();
    nlim = 0;
    tlim = pmesh->time;
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn Driver::Execute()
//! \brief Executes "main loop" by running all relevant task lists over all MeshBlockPacks
//! until a relevant stopping criteria is found (e.g. t > tlim). Calls AMR driver, and
//! performs outputs. Updates counters like (ncycle, time, etc.)

void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list(s)...\n" << std::endl;
  }

  if (time_evolution == TimeEvolution::tstatic) {
    std::cout << "\nStatic time evolution selected, solving steady-state problem...\n"
              << std::endl;
    // TODO(@user): add work for time static problems here
  } else {
    Real elapsed_time = -1.;
    std::uint64_t lb_topology_version = std::numeric_limits<std::uint64_t>::max();
    float lb_efficiency_increment = 0.0f;
    if (wall_time > 0.) {
      elapsed_time = (initialize_wall_clock_ >= 0.) ? initialize_wall_clock_
                                                    : UpdateWallClock();
    }
    while ((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
           (elapsed_time < wall_time)) {
      int outer_substeps = hydro_subcycle ? hydro_subcycle_factor : 1;
      if (nlim >= 0) {
        outer_substeps = std::min(outer_substeps, nlim - pmesh->ncycle);
      }
      const Real lat_fine_dt = pmesh->dt;
      const Real lat_time_tol = static_cast<Real>(64.0) *
          std::numeric_limits<Real>::epsilon() *
          std::max(static_cast<Real>(1.0),
                   std::max(std::abs(pmesh->time), std::abs(tlim)));
      // Rounding scale of ONE tick-by-tick time addition (driver.cpp:2111-2115).  It is
      // deliberately relative-only: lat_time_tol's max(1.0, ...) floor is a window-length
      // slack, and reusing it as a per-tick margin would make every window of a run whose
      // whole time axis is below 1 look truncated.
      const Real lat_tick_drift = static_cast<Real>(64.0) *
          std::numeric_limits<Real>::epsilon() *
          std::max(std::abs(pmesh->time), std::abs(tlim));
      int amr_check_period = 1;
      if (pmesh->adaptive && pmesh->pmr != nullptr &&
          pmesh->pmr->ncyc_check_amr > 0) {
        amr_check_period = std::max(1, pmesh->pmr->ncyc_check_amr);
      }
      int lat_sync_factor = hydro_subcycle_factor;
      if (hydro_lat) {
        int max_factor = std::max(1, hydro_subcycle_factor);
        if (pmesh->hydro_lat_metadata_valid) {
          max_factor = std::min(max_factor,
                                std::max(1, pmesh->hydro_lat_sync_factor_current));
        } else {
          max_factor = 1;
        }
        max_factor = std::min(max_factor, outer_substeps);
        lat_sync_factor = 1;
        for (int factor=2; factor<=max_factor; factor*=2) {
          lat_sync_factor = factor;
          if (factor > (std::numeric_limits<int>::max()/2)) break;
        }
        // Two conditions, and the tick loop below depends on both.  The first is the
        // window length: the whole window must fit under tlim, to within the rounding
        // slack lat_time_tol.  The second is what makes the first usable -- the fine
        // ticks clamp INDIVIDUALLY at exactly tlim (min(lat_fine_dt, tlim - pmesh->time),
        // below) and the window loop re-tests time < tlim before every substep, so a
        // window is only safe when its LAST tick is the only one the clamp can shorten.
        // The two differ by lat_time_tol, and that slack has an ABSOLUTE floor
        // (64 eps max(1,|t|,|tlim|)), so it stops being small against one fine tick as
        // soon as Mesh::NewTimeStep clamps dt to the tlim remainder for the final cycle
        // (mesh.cpp:787) on a deck whose dt is itself near that floor -- a CGS radiation
        // deck runs at dt ~ 1e-14.  With the first test alone a factor-2 window whose
        // FIRST tick already lands exactly on tlim is accepted (measured: time =
        // 4.0000000000000e-13, tlim = 4.1e-13, lat_fine_dt = tlim - time = 1.0e-14,
        // lat_time_tol = 1.42e-14, so time + 2*dt = 4.2e-13 < tlim + tol = 4.242e-13),
        // and the loop below then leaves that window after one tick with every reflux
        // accumulator still pending.  The second test costs a healthy run nothing: it can
        // only fire while the first test is within one fine tick of binding, i.e. on the
        // final window, and only when that tick is no longer than the slack itself.
        while (lat_sync_factor > 1 &&
               (((pmesh->time + lat_fine_dt*static_cast<Real>(lat_sync_factor)) >
                 (tlim + lat_time_tol)) ||
                ((pmesh->time + lat_fine_dt*static_cast<Real>(lat_sync_factor - 1) +
                  lat_tick_drift*static_cast<Real>(lat_sync_factor)) > tlim))) {
          lat_sync_factor /= 2;
        }
      }
      SourceTerms *window_grav_src = ActiveFluidSourceTerms(pmesh->pmb_pack);
      const bool hydro_lat_self_gravity =
          hydro_lat && window_grav_src != nullptr &&
          window_grav_src->self_gravity &&
          pmesh->pmb_pack->pgrav != nullptr &&
          pmesh->pmb_pack->pgrav->pmgd != nullptr;
      if (hydro_lat_self_gravity) {
        auto *pmgd = pmesh->pmb_pack->pgrav->pmgd;
        const bool invalid_gravity = !(pmesh->pmb_pack->pgrav->phi_valid);
        // The potential is frozen inside a window, so a window never exceeds solve_dt.
        const Real solve_dt = pmgd->SolveInterval();
        while (lat_sync_factor > 1 && solve_dt > 0.0 &&
               lat_fine_dt*static_cast<Real>(lat_sync_factor) > solve_dt) {
          lat_sync_factor /= 2;
        }
        // Solve now when the potential is invalid, a solve is due, or the full window
        // would step past the next due time.  Shortening the window to land exactly on
        // the due time used to produce descents of 128, 16, 4, 1, 1-tick windows in
        // which every slow block stepped at the window length; refreshing the potential
        // early instead keeps its age <= solve_dt with full-length windows.
        const bool due_now =
            invalid_gravity || pmgd->SolveDueForCycle(pmesh->ncycle, pmesh->time);
        const bool due_in_window = !due_now && pmgd->WindowCrossesSolveTime(
            pmesh->time + lat_fine_dt*static_cast<Real>(lat_sync_factor));
        const bool centered_work = pmesh->pmb_pack->pgrav->lat_time_centered_work;
        const bool centered_needs_solve = centered_work &&
            !pmesh->pmb_pack->pgrav->PotentialMatchesTime(pmesh->time);
        if (centered_needs_solve || (!centered_work && (due_now || due_in_window))) {
          // Solve() itself skips a warm potential before next_solve_time_, so an early
          // solve has to be declared due first.
          if (centered_needs_solve || due_in_window) pmgd->MarkSolveDueAt(pmesh->time);
          ClearHydroLAT(pmesh);
          pmgd->Solve(this, nexp_stages, pmesh->dt);
        }
      }
      // Design N13: the deferred sink operator advances the whole window in ONE step, so
      // it is the window length -- not the fine tick -- that its own timestep has to
      // bound.  Mesh::NewTimeStep already folds cfl_no*psink->dtnew into the global dt,
      // which is the right constraint for a per-cycle cadence but only guarantees the
      // fine tick here; without this a factor-64 ladder would push a sink up to 64
      // finest cells in one Bulirsch-Stoer call and one first-order gas->sink kick.
      // psink->dtnew is computed from the replicated sink list, so every rank halves
      // identically and the tick loop stays rank-synchronous.  Non-binding whenever the
      // sink speed is well below the finest-level fluid signal speed, which is the usual
      // case; lat_sync_factor == 1 always satisfies it, so the loop terminates.
      if (lat_sink_driver_cadence && pmesh->pmb_pack != nullptr &&
          pmesh->pmb_pack->psink != nullptr &&
          pmesh->pmb_pack->psink->use_sink_timestep) {
        // lat_window_motion_cells (default 1.0, i.e. unchanged) is the opt-in escape from
        // review A2/F4(b): with the bare sink dt this bound gives
        // lat_sync_factor <= 1 + c_s/|v_sink|, which is 1 for any supersonic sink, so one
        // fast sink silently disables LAT for the whole run.  SinkParticles validates the
        // key against the drift its pin spheres still cover.
        const Real sink_window_limit = pmesh->cfl_no*pmesh->pmb_pack->psink->dtnew*
                                       pmesh->pmb_pack->psink->lat_window_motion_cells;
        if (sink_window_limit > static_cast<Real>(0.0)) {
          while (lat_sync_factor > 1 &&
                 (lat_fine_dt*static_cast<Real>(lat_sync_factor)) >
                     (sink_window_limit + lat_time_tol)) {
            lat_sync_factor /= 2;
          }
        }
      }
      // LAT source tasks below can execute on different rank-local bin schedules, so
      // they must not contain MPI collectives. Give problem generators one common-time
      // hook after the final window length (including gravity/tlim truncation) is known.
      if (hydro_lat && pmesh->pmb_pack != nullptr && pmesh->pmb_pack->pgrav != nullptr) {
        pmesh->pmb_pack->pgrav->BeginLATEnergyWindow(
            lat_fine_dt*static_cast<Real>(std::max(1, lat_sync_factor)));
      }
      if (hydro_lat && pmesh->pgen != nullptr &&
          pmesh->pgen->user_hydro_lat_window_func != nullptr) {
        const Real lat_window_dt =
            lat_fine_dt*static_cast<Real>(std::max(1, lat_sync_factor));
        (pmesh->pgen->user_hydro_lat_window_func)(pmesh, lat_window_dt);
      }
      const bool hydro_lat_window = hydro_lat && (lat_sync_factor > 1);
      bool union_phase0_predictor = false;
      if (hydro_lat_window && hydro_lat_union_stage1) {
        int phase0_factor_count = 0;
        for (int factor=lat_sync_factor; factor>=1; factor/=2) {
          if (HydroLATFactorPresent(pmesh, factor, lat_sync_factor)) {
            ++phase0_factor_count;
          }
        }
        union_phase0_predictor = phase0_factor_count > 1;
      }
      if (hydro_lat_window) {
        outer_substeps = lat_sync_factor;
      } else if (hydro_lat) {
        outer_substeps = 1;
      }
      if (hydro_lat_window && pmesh->pmb_pack != nullptr) {
        ClearHydroLAT(pmesh);
        if (pmesh->pmb_pack->phydro != nullptr) {
          if (!union_phase0_predictor) {
            (void) pmesh->pmb_pack->phydro->CopyCons(this, 1);
          }
          pmesh->pmb_pack->phydro->ResetLATFluxCorrection();
        }
        pmesh->pmb_pack->ResetLATBlockTimes(pmesh->time);
      }
      int hydro_lat_tick_stride = 1;
      if (hydro_lat_window && pmesh->hydro_lat_metadata_valid) {
        for (int factor=1; factor<=lat_sync_factor; factor*=2) {
          if (HydroLATFactorPresent(pmesh, factor, lat_sync_factor)) {
            hydro_lat_tick_stride = factor;
            break;
          }
          if (factor > (std::numeric_limits<int>::max()/2)) break;
        }
      }
      bool hydro_lat_boundary_refresh_needed = false;
      for (int substep=0; substep<outer_substeps;
           substep += hydro_lat_tick_stride) {
        if (!((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
              (elapsed_time < wall_time))) {
          // Leaving a LAT window part-way through abandons every reflux accumulator that
          // has not reached its block's endpoint; the next ResetLATCorrections zeroes
          // them, so the coarse/fine flux mismatch of the truncated window is silently
          // lost.  Unreachable as written -- outer_substeps == lat_sync_factor was
          // already clamped against nlim, lat_sync_factor was pre-shrunk so that every
          // tick but the last leaves the mesh strictly below tlim, and elapsed_time is
          // refreshed only once per window -- but the three guarantees live in three
          // different places, so state the invariant where it would break rather than
          // losing conservation without a diagnostic.
          if (hydro_lat_window && substep > 0) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl
                      << "LAT window truncated after " << substep << " of "
                      << outer_substeps << " ticks; the pending coarse/fine reflux "
                      << "accumulators would be discarded. Terminate on a window "
                      << "boundary (tlim or wall clock), not mid-window." << std::endl;
            std::exit(EXIT_FAILURE);
          }
          break;
        }
        const bool outer_step = (!hydro_subcycle) || (substep == 0);
        if (outer_step && global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}

        int active_updates_this_tick = 0;
        auto execute_explicit_integrator = [&](bool lat_bin, bool local_work,
                                               int first_stage, int last_stage) {
          // Execute TaskLists
          // Work before time integrator indicated by "0" in stage
          if (local_work && first_stage == 1) {
            ExecuteTaskList(pmesh, "before_timeintegrator", 0);
          }
          // time-integrator tasks for each stage of integrator
          bool force_gravity_cycle = false;
          for (int stage=first_stage; stage<=last_stage; ++stage) {
            bool stage_local_work = local_work;
            if (lat_bin && hydro_lat_skip_final_exchange_this_bin &&
                stage == nexp_stages && pmesh->pmb_pack != nullptr &&
                pmesh->pmb_pack->lat_nactive_thispack <= 0 &&
                pmesh->pmb_pack->lat_nboundary_send_thispack <= 0 &&
                pmesh->pmb_pack->lat_nflux_recv_thispack <= 0) {
              stage_local_work = false;
            }
            if (stage_local_work) ExecuteTaskList(pmesh, "before_stagen", stage);
            if (!lat_bin && pmesh->pmb_pack->pgrav != nullptr) {
              auto *pmgd = pmesh->pmb_pack->pgrav->pmgd;
              const bool invalid_gravity = !(pmesh->pmb_pack->pgrav->phi_valid);
              const bool gravity_due =
                  (pmgd != nullptr) &&
                  (invalid_gravity ||
                   pmgd->SolveDueForCycle(pmesh->ncycle, pmesh->time));
              const bool skip_subcycle_gravity =
                  !force_gravity_cycle &&
                  (!gravity_due && pmesh->pmb_pack->pgrav->phi_valid);
              if (!skip_subcycle_gravity && pmgd != nullptr) {
                pmgd->Solve(this, stage, pmesh->dt);
                force_gravity_cycle = force_gravity_cycle || invalid_gravity;
              }
            }
            if (stage_local_work) {
              ExecuteTaskList(pmesh, "stagen", stage);
              ExecuteTaskList(pmesh, "after_stagen", stage);
            }
          }

          // Work after time integrator indicated by "1" in stage
          if (local_work && last_stage == nexp_stages) {
            ExecuteTaskList(pmesh, "after_timeintegrator", 1);
          }
          // A final RK state can lie at t+dt even when its last RHS abscissa
          // is interior.  Without operator-split M1 there is no later radiation
          // endpoint reconciliation, so restore prescribed metrics and gas
          // primitives here for outputs and the next cycle.
          if (local_work && last_stage == nexp_stages) {
            MeshBlockPack *pmbp = pmesh->pmb_pack;
            dyngr::DynGRMHD *pdyngr = pmbp->pdyngr;
            z4c::Z4c *pz4c = pmbp->pz4c;
            adm::ADM *padm = pmbp->padm;
            const bool prescribed_dynamic_adm =
                pdyngr != nullptr && pz4c == nullptr && padm != nullptr &&
                padm->is_dynamic;
            const Real final_rhs_time_frac = stage_time_frac[nexp_stages - 1];
            const Real endpoint_time_frac_tolerance =
                64.0*std::numeric_limits<Real>::epsilon();
            if (prescribed_dynamic_adm &&
                std::abs(final_rhs_time_frac - 1.0) >
                    endpoint_time_frac_tolerance) {
              const Real fluid_end_time = pmesh->time + pmesh->dt;
              pdyngr->SetADMVariablesAtTime(fluid_end_time);
              if (pmbp->pcoord->coord_data.bh_excise) {
                pmbp->pcoord->UpdateExcisionMasks();
              }
              problem_runtime::SetHydroStageTime(fluid_end_time);
              (void) pdyngr->ConToPrim(this, nexp_stages);
              problem_runtime::ClearHydroStageTime();
            }
          }
        };

        Real tick_dt = pmesh->dt;
        bool hydro_lat_cleared = true;
        if (hydro_lat_window) {
          const int tick_phase = substep;
          tick_dt = std::min(lat_fine_dt, tlim - pmesh->time);
          hydro_lat_cleared = false;
          // All due LAT bins in this fine tick start at the same mesh time.  A
          // union refresh supplies initial ghost zones for the first due bin.
          // This must also run on fine-only ticks: factor-1 blocks still need
          // same-level and AMR neighbor states interpolated from slower bins.
          pmesh->pmb_pack->SetActiveMeshBlocksByLATDueFactors(
              lat_sync_factor, tick_phase, true);
          RefreshHydroLATBoundaries(pmesh, pmesh->time);
          int due_factor_count = 0;
          for (int factor=lat_sync_factor; factor>=1; factor/=2) {
            if ((tick_phase % factor) == 0 &&
                HydroLATFactorPresent(pmesh, factor, lat_sync_factor)) {
              ++due_factor_count;
            }
          }
          const bool union_due_factors = hydro_lat_union_stage1 && due_factor_count > 1;
          if (union_due_factors) {
            // Every due RK2 predictor samples the same physical time.  Execute those
            // predictors as one GPU-sized union, then retain the legacy slow-to-fast
            // corrector order so faster bins see completed slower dense histories.
            pmesh->pmb_pack->ConfigureLATUnionStage1(
                pmesh->time, lat_fine_dt, lat_sync_factor);
            pmesh->pmb_pack->SelectLATDueUpdateFluxReceivers(
                lat_sync_factor, tick_phase, true);
            pmesh->dt = lat_fine_dt;
            hydro_lat_active_factor_this_bin = lat_sync_factor;
            hydro_lat_skip_final_exchange_this_bin = false;
            hydro_lat_final_ghosts_superseded_this_bin = false;
            hydro_lat_union_stage1_active = true;
            const bool local_union_work =
                (pmesh->pmb_pack->lat_nactive_thispack > 0) ||
                (pmesh->pmb_pack->lat_nboundary_send_thispack > 0) ||
                (pmesh->pmb_pack->lat_nflux_recv_thispack > 0);
            execute_explicit_integrator(
                true, local_union_work, 1, 1);
            hydro_lat_union_stage1_active = false;

            for (int active_factor=lat_sync_factor; active_factor>=1;
                 active_factor/=2) {
              if ((tick_phase % active_factor) != 0) continue;
              if (!HydroLATFactorPresent(pmesh, active_factor, lat_sync_factor)) continue;
              pmesh->pmb_pack->lat_union_pending_below_factor = active_factor;
              int nactive = ConfigureHydroLATSubstep(pmesh, substep, active_factor,
                                                     lat_sync_factor, lat_fine_dt, true);
              if (nactive <= 0) continue;
              active_updates_this_tick += nactive;
              // The predictor's per-block dt and this corrector's scalar dt must be the
              // same number for every block in this bin.
              CheckLATUnionStepDt(pmesh);
              const bool local_lat_work =
                  (pmesh->pmb_pack->lat_nactive_thispack > 0) ||
                  (pmesh->pmb_pack->lat_nboundary_send_thispack > 0) ||
                  (pmesh->pmb_pack->lat_nflux_recv_thispack > 0);
              // This replaces the predictor's state exchange.  Pending faster senders
              // expose u1, while completed slower senders use their dense RK2 history.
              RefreshHydroLATBoundaries(pmesh, pmesh->time + pmesh->dt, false, true);
              // C2P/floor synchronization in the refresh can alter the Euler endpoint.
              // Save or finish stage-local derived state only after this factor has the
              // same predictor endpoint that the factor-by-factor path would have used.
              if (pmesh->pmb_pack->phydro != nullptr) {
                (void) pmesh->pmb_pack->phydro->SaveLATDenseOutput(this, 1);
              }
              execute_explicit_integrator(
                  true, local_lat_work, 2, 2);
            }
            pmesh->pmb_pack->lat_union_pending_below_factor = 0;
            hydro_lat_final_ghosts_superseded_this_bin = false;
          } else {
            for (int active_factor=lat_sync_factor; active_factor>=1;
                 active_factor/=2) {
              if ((tick_phase % active_factor) != 0) continue;
              if (!HydroLATFactorPresent(pmesh, active_factor, lat_sync_factor)) continue;
              int nactive = ConfigureHydroLATSubstep(pmesh, substep, active_factor,
                                                     lat_sync_factor, lat_fine_dt);
              if (nactive <= 0) continue;
              active_updates_this_tick += nactive;
              const bool local_lat_work =
                  (pmesh->pmb_pack->lat_nactive_thispack > 0) ||
                  (pmesh->pmb_pack->lat_nboundary_send_thispack > 0) ||
                  (pmesh->pmb_pack->lat_nflux_recv_thispack > 0);
              execute_explicit_integrator(
                  true, local_lat_work, 1, nexp_stages);
            }
            hydro_lat_final_ghosts_superseded_this_bin = false;
          }
          pmesh->dt = tick_dt;
        } else if (hydro_lat) {
          ClearHydroLAT(pmesh);
          hydro_lat_cleared = true;
          tick_dt = std::min(pmesh->dt, tlim - pmesh->time);
          pmesh->dt = tick_dt;
          active_updates_this_tick = pmesh->nmb_total;
          execute_explicit_integrator(false, true, 1, nexp_stages);
        } else {
          active_updates_this_tick = pmesh->nmb_total;
          execute_explicit_integrator(false, true, 1, nexp_stages);
        }

        // Work outside of TaskLists.  When no factor-1 bin exists, every phase between
        // successive multiples of hydro_lat_tick_stride is empty.  Advance those phases
        // together, retaining the original sequence of floating-point time additions.
        //
        // There is deliberately no endpoint snap here.  pmesh->time at a window end is
        // lat_sync_factor successive additions of lat_fine_dt, while a factor-F block's
        // own lat_time_end is (partial sum) + F*lat_fine_dt (ConfigureHydroLATSubstep),
        // so the two disagree by O(sync*eps*|t|).  Replacing the accumulation with
        // t_window0 + n*lat_fine_dt would move every dump time and every ghost
        // interpolation parameter for a defect that is currently ~2e-7 of one fine tick
        // (dx_finest ~ 0.083 M, cfl 0.3 => fine_dt ~ 2.5e-2, t <= 5e4, sync <= 512), and
        // the one consumer that could be sensitive -- lat_theta in
        // MeshBoundaryValuesCC::PackAndSendCC -- is bounded away from {0,1} by
        // 1/sync_factor ~ 2e-3.  Endpoint SELECTION is integer (LATEndpoint), so the
        // drift can never mis-select a reflux endpoint.  Revisit only if
        // lat_fine_dt/|t| ever falls below ~1e-9.
        const int completed_substep = hydro_lat_window ?
            std::min(outer_substeps, substep + hydro_lat_tick_stride) : (substep + 1);
        const int completed_ticks = completed_substep - substep;
        Real completed_driver_interval = 0.0;
        if (hydro_lat_window) {
          for (int tick=0; tick<completed_ticks; ++tick) {
            const Real fine_tick_dt = std::min(lat_fine_dt, tlim - pmesh->time);
            pmesh->time += fine_tick_dt;
            pmesh->dt = fine_tick_dt;
            completed_driver_interval += fine_tick_dt;
          }
        } else {
          pmesh->time += tick_dt;
          completed_driver_interval = tick_dt;
        }
        pmesh->ncycle += completed_ticks;
        pmesh->dt_last_completed = completed_driver_interval;
        // Design N13: the deferred sink operator is handed exactly the physical time the
        // driver advanced since it last ran, which is not lat_fine_dt*lat_sync_factor
        // whenever tlim, nlim or the wall clock truncates a window.
        if (lat_sink_driver_cadence) {
          lat_sink_pending_dt += completed_driver_interval;
        }
        nmb_updated_ += active_updates_this_tick;
        npart_updated_ += completed_ticks*pmesh->nprtcl_total;
        if (hydro_lat_window && pmesh->pmb_pack != nullptr &&
            pmesh->pmb_pack->phydro != nullptr) {
          // Correction order is load-bearing.  (a) The fluid pass owns the correction
          // mask: it clears the mask at entry and replaces the pack's active list, so no
          // other writer may precede it.
          MeshBlockPack *lat_pmbp = pmesh->pmb_pack;
          bool corrected = false;
          const bool reflux_debug = lat_pmbp->pgrav != nullptr &&
              lat_pmbp->pgrav->lat_ledger_debug && lat_pmbp->phydro != nullptr;
          const Real reflux_energy_before =
              reflux_debug ? lat_pmbp->pgrav->TotalGasEnergy() : 0.0;
          if (lat_pmbp->phydro != nullptr) {
            corrected = lat_pmbp->phydro->ApplyLATFluxCorrection(
                pmesh->time, lat_sync_factor, completed_substep);
            if (reflux_debug) {
              lat_pmbp->pgrav->debug_reflux_energy_local +=
                  lat_pmbp->pgrav->TotalGasEnergy() - reflux_energy_before;
            }
          }
          // (d) User fixup and fluid recovery run on the fluid receiver set.
          // AMENDMENT A5.1 (dual energy x LAT): every writer above can change
          // u0(IEN) on receiver cells -- the fluid transport add in (a) -- while the auxiliary
          // internal energy the dual-energy formalism carries in u0(dual_energy_idx) is
          // an independently evolved field.  Hydro::SynchronizeDualEnergyFieldFromTotal
          // is exactly the reconciliation the amendment asks for (eint = E - KE from the
          // corrected conserved state, subject to hydro's own eta2 eligibility rule) and
          // is mask-aware, running over the receiver blocks the passes above installed.
          // It is invoked explicitly here rather than left to the copy inside
          // Hydro::ConToPrim so that the invariant does not depend on which branch of
          // that task is taken; it is idempotent, and it runs again after (f) because it
          // is part of this lambda.  Both target pgens run dual_energy=true, so without
          // it temperatures on coarse/fine faces are silently wrong.
          auto lat_fluid_recovery = [&]() {
            if (lat_pmbp->phydro != nullptr) {
              // Hydro::ConToPrim reconciles the dual-energy field itself.
              (void) lat_pmbp->phydro->ConToPrim(this, nexp_stages);
              return;
            }
          };
          if (corrected) {
            if (lat_pmbp->phydro != nullptr && pmesh->pgen != nullptr &&
                pmesh->pgen->user_hydro_state_fixup_func != nullptr) {
              pmesh->pgen->user_hydro_state_fixup_func(lat_pmbp, pmesh->time);
            }
            lat_fluid_recovery();
          }
          hydro_lat_boundary_refresh_needed = true;
          ClearHydroLAT(pmesh);
          hydro_lat_cleared = true;
        }
        // load balancing efficiency
        if (global_variable::nranks > 1) {
          if (lb_topology_version != pmesh->topology_version) {
            int minnmb = std::numeric_limits<int>::max();
            for (int i=0; i<global_variable::nranks; ++i) {
              minnmb = std::min(minnmb, pmesh->nmb_eachrank[i]);
            }
            lb_efficiency_increment =
                static_cast<float>(minnmb*global_variable::nranks)/
                static_cast<float>(pmesh->nmb_total);
            lb_topology_version = pmesh->topology_version;
          }
          lb_efficiency_ += completed_ticks*lb_efficiency_increment;
        }

        const bool stop_after_substep =
            !((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0));
        const bool end_outer_step =
            (!hydro_subcycle) || (completed_substep >= outer_substeps) ||
            stop_after_substep;
        const bool lat_synchronized_state = (!hydro_lat_window) || end_outer_step;
        // Do not pay for AMR/load balancing after the last accepted hydro update.
        // There is no following step to use the adapted mesh, and short tlim
        // benchmarks can otherwise be dominated by this terminal AMR pass.
        bool amr_due = pmesh->adaptive && !stop_after_substep;
        if (amr_due && hydro_lat) {
          amr_due = end_outer_step && pmesh->pmr != nullptr &&
              ((pmesh->ncycle - pmesh->pmr->last_amr_call_cycle) >= amr_check_period);
        } else if (amr_due && hydro_subcycle) {
          amr_due = (pmesh->pmr != nullptr) &&
              ((pmesh->ncycle % std::max(1, pmesh->pmr->ncyc_check_amr)) == 0);
        }
        if (hydro_lat && lat_synchronized_state) {
          if (!hydro_lat_cleared) {
            ClearHydroLAT(pmesh);
            hydro_lat_cleared = true;
          }
          // Design N13, investigation item S2-1: the once-per-cycle sink operator, driven
          // from here because the "before_timeintegrator" list that owns its task is
          // executed once per LAT BIN with a rank-dependent active list.  This is the
          // only point inside the LAT loop that meets all of the operator's requirements
          // at once.  (i) Every rank reaches it: the tick loop is driven by globally
          // cached factor metadata, so its trip count and this branch are rank
          // independent, and the MPI_Allreduce calls inside the kick / creation /
          // accretion stages (and inside CheckLATInvariants) are matched.  (ii) The LAT
          // mask is off and every bin has closed the window, so the direct sums that run
          // over every local cell -- the gas->sink kick and the accretion ambient shell
          // -- see one time-consistent gas state; that is why the interval is declared
          // fully synchronized.  (iii) The declared interval is the time actually
          // advanced since the previous call.  Firing once per fine tick instead would
          // be wrong even with pinning: the reads above cover blocks in slower bins that
          // sit at an earlier time mid-window (SinkParticles warns about exactly that).
          bool gravity_work_corrected = false;
          if (pmesh->pmb_pack != nullptr && pmesh->pmb_pack->pgrav != nullptr &&
              pmesh->pmb_pack->pgrav->lat_time_centered_work) {
            auto *gravity = pmesh->pmb_pack->pgrav;
            gravity->PrepareLATEnergyWindowEnd();
            gravity->pmgd->MarkSolveDueAt(pmesh->time);
            gravity->pmgd->Solve(this, nexp_stages,
                pmesh->time-gravity->energy_window_start);
            gravity_work_corrected = gravity->EndLATEnergyWindow();
          }
          bool sink_stepped = false;
          if (lat_sink_driver_cadence && lat_sink_pending_dt > 0.0 &&
              pmesh->pmb_pack != nullptr && pmesh->pmb_pack->psink != nullptr) {
            auto *psink = pmesh->pmb_pack->psink;
            psink->SetStepInterval(lat_sink_pending_dt, true);
            (void) psink->SinkStep(
                this, sinkparticles::SinkParticles::kLATSyncStage);
            lat_sink_pending_dt = 0.0;
            sink_stepped = true;
          }
          // Accretion and the Jeans surgery rewrite conserved gas state in interior
          // cells, so the neighbours' ghosts and every block's primitives (including the
          // dual-energy auxiliary) are stale until they are republished.  force_c2p
          // turns the refresh's ghost-band shortcut into a full recovery for exactly
          // that reason.  Without a sink step this keeps the previous behaviour verbatim.
          if (hydro_lat_boundary_refresh_needed || sink_stepped ||
              gravity_work_corrected) {
            RefreshHydroLATBoundaries(pmesh, pmesh->time, true,
                                      sink_stepped || gravity_work_corrected);
            hydro_lat_boundary_refresh_needed = false;
            hydro_lat_cleared = true;
          }
        }
        // A kernel launch that fails (a card too full to grow a kernel's local-memory
        // reservation, say) returns an error Kokkos reads only in debug builds, and in a
        // release build the rank goes on with the data the kernel never wrote.  The
        // runtime holds that error until it is read, so one host query per cycle, before
        // the outputs, catches every failed launch since the last one without a device
        // synchronization.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        {
#if defined(KOKKOS_ENABLE_CUDA)
          const cudaError_t device_err = cudaGetLastError();
          const char *device_msg =
              (device_err != cudaSuccess) ? cudaGetErrorString(device_err) : nullptr;
#else
          const hipError_t device_err = hipGetLastError();
          const char *device_msg =
              (device_err != hipSuccess) ? hipGetErrorString(device_err) : nullptr;
#endif
          if (device_msg != nullptr) {
            std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << std::endl << "device error on rank " << global_variable::my_rank
                      << " by cycle " << pmesh->ncycle << ": " << device_msg << std::endl;
#if MPI_PARALLEL_ENABLED
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
            std::exit(EXIT_FAILURE);
          }
        }
#endif
        bool output_due = end_outer_step;
        if (hydro_lat && lat_synchronized_state && pmesh->pgen != nullptr &&
            pmesh->pgen->user_hydro_lat_window_end_func != nullptr) {
          (pmesh->pgen->user_hydro_lat_window_end_func)(pmesh);
        }
        if (hydro_subcycle && !hydro_lat && !output_due) {
          for (auto &out : pout->pout_list) {
            float time_32 = static_cast<float>(pmesh->time);
            float next_32 = static_cast<float>(out->out_params.last_time+out->out_params.dt);
            float tlim_32 = static_cast<float>(tlim);
            int &dcycle_ = out->out_params.dcycle;
            output_due =
                ((out->out_params.dt > 0.0) && ((time_32 >= next_32) &&
                 (time_32 < tlim_32))) ||
                ((dcycle_ > 0) && ((pmesh->ncycle)%(dcycle_) == 0));
            if (output_due) break;
          }
        }
        if (output_due) {
          // Test for/make outputs
          bool allow_outputs = true;
          if (pmesh->pgen != nullptr && pmesh->pgen->user_output_gate_func != nullptr) {
            allow_outputs = pmesh->pgen->user_output_gate_func(pmesh);
          }
          if (allow_outputs) {
            for (auto &out : pout->pout_list) {
              // compare at floating point (32-bit) precision to reduce effect of round
              // off
              float time_32 = static_cast<float>(pmesh->time);
              float next_32 = static_cast<float>(out->out_params.last_time+out->out_params.dt);
              float tlim_32 = static_cast<float>(tlim);
              int &dcycle_ = out->out_params.dcycle;

              if (((out->out_params.dt > 0.0) &&
                   ((time_32 >= next_32) && (time_32<tlim_32))) ||
                  ((dcycle_ > 0) && ((pmesh->ncycle)%(dcycle_) == 0)) ) {
                out->LoadOutputData(pmesh);
                out->WriteOutputFile(pmesh, pin);
              }
            }
          }
        }
        if (amr_due) {
          if (!hydro_lat_cleared) {
            ClearHydroLAT(pmesh);
            hydro_lat_cleared = true;
          }
          if (pmesh->pmb_pack != nullptr) {
            pmesh->pmb_pack->ReleaseLATCacheMemory();
          }
          pmesh->pmr->AdaptiveMeshRefinement(this, pin);
          ClearHydroLAT(pmesh);
          hydro_lat_cleared = true;
        }
        if (lat_synchronized_state &&
            pmesh->pgen != nullptr && pmesh->pgen->after_cycle_func != nullptr) {
          const bool post_step_state_changed =
              (pmesh->pgen->after_cycle_func)(this, pin, pmesh);
          if (post_step_state_changed) {
            if (pmesh->pmb_pack != nullptr && pmesh->pmb_pack->pgrav != nullptr) {
              pmesh->pmb_pack->pgrav->MarkPhiInvalid();
            }
          }
        }
        auto *pgrav = (pmesh->pmb_pack != nullptr) ? pmesh->pmb_pack->pgrav : nullptr;
        const bool terminal_gravity_solve =
            stop_after_substep && pgrav != nullptr && pgrav->pmgd != nullptr &&
            (!(pgrav->phi_valid) ||
             (hydro_subcycle && pgrav->pmgd->SolveDueAtTime(pmesh->time)));
        if (terminal_gravity_solve) {
          pgrav->pmgd->ResetCadenceCache();
          pgrav->pmgd->Solve(this, nexp_stages, pmesh->dt);
        }
        // The wall clock is read once per outer step, here, and the while condition
        // below tests this same value, so a run that stops after this substep is known
        // before the timestep pass.
        if (end_outer_step && wall_time > 0.) {
          elapsed_time = UpdateWallClock();
        }
        const bool run_ends = stop_after_substep || !(elapsed_time < wall_time);
        // Compute a new fluid timestep only at synchronized LAT points.  If AMR ran
        // above, this uses the refined/derefined mesh before the next substep.  A run
        // that stops here takes no further step, so the pass is left to the restart:
        // Driver::Initialize runs it on the checkpointed state, and the checkpoint
        // Finalize writes must therefore precede it, exactly like one written by the
        // output pass above.  Mesh::NewTimeStep is not idempotent -- it limits dt to
        // twice the stored dt and it commits the hybrid force-free window-end
        // classification -- so a Finalize checkpoint taken after it would have the
        // restart apply both a second time.
        const bool recompute_timestep = !run_ends &&
            ((!hydro_lat_window) || end_outer_step || amr_due);
        if (hydro_lat && recompute_timestep && pmesh->pmb_pack != nullptr &&
            pmesh->pmb_pack->phydro != nullptr) {
          if (!hydro_lat_cleared) {
            ClearHydroLAT(pmesh);
            hydro_lat_cleared = true;
          }
          if (pmesh->pmb_pack->phydro != nullptr) {
            (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
          }
        }
        if (recompute_timestep) {
          pmesh->NewTimeStep(tlim);
          if (hydro_lat) {
            RebuildHydroLATMetadata(pmesh);
            const bool hydro_lat_gid_reorder =
                pin->GetOrAddBoolean("time", "hydro_lat_gid_reorder", true);
            const bool hydro_lat_post_amr_rebalance =
                pin->GetOrAddBoolean("time", "hydro_lat_post_amr_rebalance",
                                     hydro_lat_gid_reorder);
            const int hydro_lat_rebalance_delay =
                kHydroLATRebalanceStableWindows*
                std::max(1, pmesh->hydro_lat_sync_factor_current);
            const bool topology_stable =
                (pmesh->ncycle - pmesh->topology_last_change_cycle) >=
                hydro_lat_rebalance_delay;
            // This rebalance answers a change in the mesh, so it must be armed by one.
            // The elapsed-cycles form it replaces was a timer: a repartition stamps
            // topology_last_change_cycle itself, so on a static mesh the mechanism
            // rearmed its own trigger and fired every hydro_lat_rebalance_delay cycles
            // forever, migrating a block or two each time.  Measured on the production
            // SMR deck, ten ranks: blocks per rank were already 112-113 and the
            // cost-weighted imbalance 1.028, and the per-rank utilisation was 91.2 per
            // cent before and after -- the residual spread is ranks waiting on the
            // bin-final exchange behind an unevenly distributed implicit solve, which
            // moving blocks cannot address.  An adaptive mesh still gets its rebalance: a
            // regrid stamps a cycle later than the one recorded here.
            const bool topology_changed_since_rebalance =
                pmesh->hydro_lat_lb_last_attempt_cycle < 0 ||
                pmesh->topology_last_change_cycle >
                pmesh->hydro_lat_lb_last_attempt_cycle;
            if (hydro_lat_post_amr_rebalance &&
                pmesh->multilevel && pmesh->pmr != nullptr &&
                pmesh->hydro_lat_metadata_valid &&
                !pmesh->HydroLATLoadBalanceCurrent() &&
                topology_stable && topology_changed_since_rebalance &&
                RebalanceHydroLATMesh(pmesh, pin)) {
              if (pmesh->pmb_pack->phydro != nullptr) {
                (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
              }
              // The per-block timesteps above feed the factor metadata only.  The mesh dt
              // stands: the state is the one pmesh->NewTimeStep has just reduced, and
              // that pass is not idempotent -- a second one would let dt grow by a
              // second factor of two and have a hybrid force-free sidecar commit its
              // classification and transition recovery again -- so the rebalance carries
              // what the pass left (MeshRefinement::FullBlockTransferForceFreeState)
              // and a run that rebalances follows the trajectory of one that does not.
              RebuildHydroLATMetadata(pmesh);
              hydro_lat_boundary_refresh_needed = false;
              hydro_lat_cleared = true;
            }
          }
        } else if (!run_ends) {
          pmesh->dt = lat_fine_dt;
        }
      }
    }  // end while
  }    // end of (time_evolution != tstatic) clause
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::Finalize()
//! \brief Tasks to be performed after execution of Driver, such as making final output
//!  and printing diagnostic messages

void Driver::Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  // cycle through output Types and load data / write files
  //  This design allows for asynchronous outputs to implemented in the future.
  bool allow_outputs = true;
  if (pmesh->pgen != nullptr && pmesh->pgen->user_output_gate_func != nullptr) {
    allow_outputs = pmesh->pgen->user_output_gate_func(pmesh);
  }
  if (allow_outputs) {
    for (auto &out : pout->pout_list) {
      out->LoadOutputData(pmesh);
      out->WriteOutputFile(pmesh, pin);
    }
  }

  // call any problem specific functions to do work after main loop
  if (pmesh->pgen->pgen_final_func != nullptr) {
    (pmesh->pgen->pgen_final_func)(pin, pmesh);
  }

  float exe_time = run_time_.seconds();

  if (time_evolution != TimeEvolution::tstatic) {
#if MPI_PARALLEL_ENABLED
    // Collect number of MeshBlocks communicated during load balancing across all ranks
    if (pmesh->adaptive) {
      MPI_Allreduce(MPI_IN_PLACE, &(pmesh->pmr->nmb_sent_thisrank), 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
#endif
    if (global_variable::my_rank == 0) {
      // Print diagnostic messages related to the end of the simulation
      OutputCycleDiagnostics(pmesh);
      if (pmesh->ncycle == nlim) {
        std::cout << std::endl << "Terminating on cycle limit" << std::endl;
      } else if (pmesh->time >= tlim) {
        std::cout << std::endl << "Terminating on time limit" << std::endl;
      } else {
        std::cout << std::endl << "Terminating on wall clock limit" << std::endl;
      }

      std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
      std::cout << "tlim=" << tlim << " nlim=" << nlim << std::endl;

      if (pmesh->adaptive) {
        std::cout << std::endl << "Current number of MeshBlocks = " << pmesh->nmb_total
          << std::endl << pmesh->pmr->nmb_created << " MeshBlocks created, "
          << pmesh->pmr->nmb_deleted << " deleted by AMR" << std::endl;
#if MPI_PARALLEL_ENABLED
        std::cout << pmesh->pmr->nmb_sent_thisrank << " communicated for load balancing, "
          <<"load balancing efficiency = " << (lb_efficiency_/pmesh->ncycle) << std::endl;
#endif
      }

      // Calculate and print the zone-cycles/cpu-second
      // Note the need for 64-bit integers since nmb_updated can easily exceed 2^32.
      std::uint64_t zonecycles = nmb_updated_ *
                                 static_cast<uint64_t>(pmesh->NumberOfMeshBlockCells());
      float zcps = static_cast<float>(zonecycles) / exe_time;
      float pups = static_cast<float>(npart_updated_) / exe_time;

      std::cout << std::endl << "MeshBlock-cycles = " << nmb_updated_ << std::endl;
      std::cout << "cpu time used  = " << exe_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
      std::cout << "particle-updates/cpu_second = " << pups << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::OutputCycleDiagnostics()
//! \brief Simple function to print diagnostics every 'ndiag' cycles to stdout

void Driver::OutputCycleDiagnostics(Mesh *pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    Real elapsed = pwall_clock_->seconds();
    std::cout << "elapsed=" << std::scientific << std::setprecision(dtprcsn) << elapsed
              << " cycle=" << pm->ncycle
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::UpdateWallClock()
//! \brief Update and sync the wall clock across all MPI ranks. This is necessary because
//! the different MPI ranks may 1) initialize their timers at slightly different times,
//! and 2) may reach the end of a loop to update their timers at slightly different times.
//! This may result in a weird problem where one or more ranks have timers that fall
//! slightly below the wall clock time while others determine that it's time to quit.

Real Driver::UpdateWallClock() {
  Real tnow;
  if (global_variable::my_rank == 0) {
    tnow = pwall_clock_->seconds();
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&tnow, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return tnow;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::InitBoundaryValuesAndPrimitives()
//! \brief Sets boundary conditions on conserved and initializes primitives.  Used both
//! on initialization, and when new MBs created with AMR.

void Driver::InitBoundaryValuesAndPrimitives(Mesh *pm, bool repair_amr_fc,
                                             bool restart_verbatim) {
  // Note: with MPI, sends on ALL MBs must be complete before receives execute

  // restart_verbatim: the evolved arrays were restored exactly as the checkpoint holds
  // them, ghost zones included (every module checkpoints its arrays with ghosts), and
  // a step's first stage reads those ghost zones without a fresh exchange.  So the
  // stored ghosts ARE the ones the continuous run carries into its next step, while
  // re-deriving them here is not a fixed point of the end-of-step state: the coarse/
  // fine packs, the prolongation C2P and the physical BCs would now act on conserved
  // variables the final C2P has since floored/limited, and the C2P itself is not
  // idempotent, so the re-derived ghosts differ from the stored ones by up to the
  // floor adjustments and every MeshBlock face would step off the checkpointed
  // trajectory.  Keep the stored ghost zones and rebuild only what the checkpoint does
  // not carry: the coarse arrays and the primitives.
  const bool exchange = !restart_verbatim;
  // On a verbatim restart the recovery must not write floored/limited conserved
  // variables back: the checkpointed u0 already IS the write-back of the run's last
  // recovery, and the inversion is not idempotent at round-off, so a second write-back
  // would move u0 off the checkpointed trajectory.
  //
  // This recovery is COLD -- no run of this process has filled the inversion's
  // warm-start cache yet (eos.hpp, c2p_mu_cache), and an ordinary checkpoint does not
  // carry it.  A warm and a cold solve of the same u0 accept roots that differ by up to
  // the root find's 1e-12 tolerance, so without the hybrid sidecar this pass reproduces
  // the writing run's primitives only because that run emptied its own cache and
  // repeated its recovery cold before checkpointing (RestartOutput::LoadOutputData).
  // Both sides must keep doing that, or the restart resumes from primitives that differ
  // from the checkpointed trajectory's in the last few digits.  A hybrid sidecar
  // checkpoint of version 8 or later carries the writing run's w0 and tails instead, and
  // on the fixed metric its warm-start roots and signed-entropy primitive as well; they
  // are put back after this recovery (below).  A version-8 file may also hold coarse_w0,
  // which nothing reads and which is not put back.  A dynamical-GR step re-inverts u0
  // before its first flux and its inversion keeps no such cache, so there the
  // continuation is exact regardless.
  // Restart scratch copies cover only the live MeshBlocks: the storage extent may exceed
  // nmb_thispack (capacity padding or preallocation) and no kernel reads past it.
  const int nmb_live_rst = pm->pmb_pack->nmb_thispack;
  auto live_blocks = [nmb_live_rst](auto &view) {
    if constexpr (std::decay_t<decltype(view)>::rank == 5) {
      return Kokkos::subview(view, std::make_pair(0, nmb_live_rst), Kokkos::ALL,
                             Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    } else {
      return Kokkos::subview(view, std::make_pair(0, nmb_live_rst), Kokkos::ALL,
                             Kokkos::ALL, Kokkos::ALL);
    }
  };
  auto recover_primitives = [&](DvceArray5D<Real> &u, auto &&con_to_prim) {
    if (!restart_verbatim) {
      con_to_prim();
      return;
    }
    DvceArray5D<Real> u_saved("restart_u0_saved", nmb_live_rst, u.extent(1),
                              u.extent(2), u.extent(3), u.extent(4));
    Kokkos::deep_copy(u_saved, live_blocks(u));
    // Cold means cold: anything that ran since the restart read may have published roots
    // -- the multilevel coarse-register rebuild below applies the physical/user BCs, and
    // a user BC inverts the u0/w0 ghost bands.  Recovering those ghosts warm put them off
    // by the root tolerance, which the first SMR step amplified to 60% in L1 velocity.
    if (pm->pmb_pack->pmhd != nullptr) {
      (void) pm->pmb_pack->pmhd->peos->ResetC2PWarmStart();
    }
    con_to_prim();
    Kokkos::deep_copy(live_blocks(u), u_saved);
  };

  // A user boundary function writes the fine ghost cells only; each module's Prolongate
  // restricts them into the coarse ghost cells across the user face, which its
  // prolongation stencil reads.  MeshBlocks the problem generator or a regrid has just
  // created have not seen the function yet (the AMR transaction moves interiors only),
  // so run it once before their first prolongation.
  if (exchange && pm->multilevel && pm->pgen != nullptr && pm->pgen->user_bcs) {
    // That call runs before the exchange, and a user function reads the block's own
    // ghost cells along the face (every transverse row, as the built-in boundaries do).
    // After a regrid they hold whatever their storage slot held before -- another
    // block's data, or none -- so the coarse ghosts restricted from its output, the
    // fine ghosts prolongated from those, and the inversion's warm start in the cells
    // it inverted all depended on the rank layout.  Define them first, from each
    // block's own interior; the exchange below replaces all but the ghosts only the
    // boundary functions write.
    if (repair_amr_fc) {
      MeshBlockPack *pmbp = pm->pmb_pack;
      const int nmb = pmbp->nmb_thispack;
      const auto &indcs = pm->mb_indcs;
      const bool multi_d = pm->multi_d, three_d = pm->three_d;
      if (pmbp->phydro != nullptr) {
        SeedGhostsFromInteriorCC(pmbp->phydro->u0, indcs, nmb, multi_d, three_d);
      }
      if (pmbp->pmhd != nullptr) {
        SeedGhostsFromInteriorCC(pmbp->pmhd->u0, indcs, nmb, multi_d, three_d);
        SeedGhostsFromInteriorFC(pmbp->pmhd->b0, indcs, nmb, multi_d, three_d);
      }
      if (pmbp->prad != nullptr) {
        SeedGhostsFromInteriorCC(pmbp->prad->i0, indcs, nmb, multi_d, three_d);
      }
    }
    if (pm->pgen->user_bcs_func != nullptr) {
      (pm->pgen->user_bcs_func)(pm);
    }
  }

  // A MeshBlock sends a coarser neighbour ng coarse cells normal to their shared face.
  // When it is narrower than 2*ng its own restriction supplies only nx/2 of them; the
  // rest are its coarse ghost cells on the far side, which only its own exchange fills
  // (received from a coarser neighbour there, or restricted by the Prolongate step from
  // what a same-level or finer one sent).  Likewise its sends to a finer neighbour reach
  // into its own fine ghost cells along the face.  Within a step these hold the previous
  // stage's values, but at start-up nothing has filled them, and a regrid carries no
  // ghost cells with the blocks it creates or moves, so the neighbour's outer ghost cells
  // would receive zeros or another block's cells.  What such a block passes on may
  // itself have been passed on: toward a coarser neighbour it relays what a finer one
  // relayed, toward a finer neighbour what a coarser one relayed, so the blocks along a
  // chain change level monotonically and a chain from the cell's owner has at most as
  // many hops as the mesh has levels.  On such a mesh each exchange below therefore runs
  // once per level present; each pass carries every chain one hop further.
  const bool narrow_blocks = NarrowMeshBlocks(pm);
  int nlevels = 1;
  if (narrow_blocks) {
    int lmin = pm->lloc_eachmb[0].level, lmax = lmin;
    for (int gid=1; gid<pm->nmb_total; ++gid) {
      lmin = std::min(lmin, pm->lloc_eachmb[gid].level);
      lmax = std::max(lmax, pm->lloc_eachmb[gid].level);
    }
    nlevels = lmax - lmin + 1;
  }
  const int npass = exchange ? nlevels : 0;

  // A verbatim restart runs none of these exchanges, and the coarse arrays are not
  // checkpointed: the restrictions below rebuild their interiors only.  On a mesh of
  // narrow MeshBlocks the resumed run's first stage passes their coarse ghost cells on,
  // so rebuild those as the writing run's last exchange left them.  One exchange fills
  // the coarse ghost cells facing a coarser neighbour, which packs them from its fine
  // arrays, checkpointed ghost zones included.  The checkpointed fine arrays then go
  // back -- a narrow sender packed its still empty coarse ghost cells into them -- and
  // Prolongate restricts their ghost zones into the coarse ghost cells facing same-level
  // and finer neighbours (z4c receives the same-level ones in the exchange itself) and
  // applies the physical boundaries to the coarse arrays.  The fine arrays go back once
  // more, since Prolongate also rewrote the fine ghost cells facing a coarser neighbour.
  // The cells a narrow MeshBlock passes on lie within half the ghost depth (nx >= ng),
  // which one such pass fills from the neighbours' own cells.
  const bool rebuild_coarse = restart_verbatim && narrow_blocks;
  auto rebuild_coarse_ghosts = [&](DvceArray5D<Real> &u, auto &&exchange_u,
                                   auto &&prolongate_u) {
    DvceArray5D<Real> u_saved("restart_fine_saved", nmb_live_rst, u.extent(1),
                              u.extent(2), u.extent(3), u.extent(4));
    Kokkos::deep_copy(u_saved, live_blocks(u));
    exchange_u();
    Kokkos::deep_copy(live_blocks(u), u_saved);
    prolongate_u();
    Kokkos::deep_copy(live_blocks(u), u_saved);
  };

  // Initialize Z4c
  z4c::Z4c *pz4c = pm->pmb_pack->pz4c;
  if (pz4c != nullptr) {
    (void) pz4c->RestrictU(this, 0);
    for (int pass=0; pass<npass; ++pass) {
      (void) pz4c->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
      (void) pz4c->SendU(this, 0);
      (void) pz4c->ClearSend(this, -1);
      (void) pz4c->ClearRecv(this, -1);
      (void) pz4c->RecvU(this, 0);
      (void) pz4c->Z4cBoundaryRHS(this, 0);
      (void) pz4c->Prolongate(this, 0);
      (void) pz4c->ApplyPhysicalBCs(this, 0);
    }
    if (rebuild_coarse) {
      rebuild_coarse_ghosts(pz4c->u0, [&]() {
        (void) pz4c->InitRecv(this, -1);
        (void) pz4c->SendU(this, 0);
        (void) pz4c->ClearSend(this, -1);
        (void) pz4c->ClearRecv(this, -1);
        (void) pz4c->RecvU(this, 0);
      }, [&]() { (void) pz4c->Prolongate(this, 0); });
    }
  }

  // Initialize HYDRO: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  hydro::Hydro *phydro = pm->pmb_pack->phydro;
  if (phydro != nullptr) {
    // Only the non-relativistic auxiliary can be seeded here, from the conserved state
    // alone; the GR adiabat is seeded after the inversion below.
    if (phydro->use_dual_energy && phydro->dual_energy_pdv &&
        phydro->dual_energy_needs_init) {
      phydro->InitializeDualEnergyFieldFromTotal();
      phydro->dual_energy_needs_init = false;
    }
    // following functions return a TaskStatus, but it is ignored so cast to (void)
    if (pm->pgen != nullptr && pm->pgen->user_hydro_state_fixup_func != nullptr) {
      pm->pgen->user_hydro_state_fixup_func(pm->pmb_pack, pm->time);
    }
    (void) phydro->RestrictU(this, 0);
    for (int pass=0; pass<npass; ++pass) {
      (void) phydro->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
      (void) phydro->SendU(this, 0);
      (void) phydro->ClearSend(this, -1); // stage = -1 only clear SendU
      (void) phydro->ClearRecv(this, -1); // stage = -1 only clear RecvU
      (void) phydro->RecvU(this, 0);
      (void) phydro->SendU_Shr(this, 0);
      (void) phydro->ClearSend(this, -4); // stage = -4 only clear SendU_Shr
      (void) phydro->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr
      (void) phydro->RecvU_Shr(this, 0);
      (void) phydro->Prolongate(this, 0);
      (void) phydro->ApplyPhysicalBCs(this, 0);
    }
    if (rebuild_coarse) {
      rebuild_coarse_ghosts(phydro->u0, [&]() {
        (void) phydro->InitRecv(this, -1);
        (void) phydro->SendU(this, 0);
        (void) phydro->ClearSend(this, -1);
        (void) phydro->ClearRecv(this, -1);
        (void) phydro->RecvU(this, 0);
      }, [&]() { (void) phydro->Prolongate(this, 0); });
    }
    if (pm->pgen != nullptr && pm->pgen->user_hydro_state_fixup_func != nullptr) {
      pm->pgen->user_hydro_state_fixup_func(pm->pmb_pack, pm->time);
    }
    recover_primitives(phydro->u0, [&]() { (void) phydro->ConToPrim(this, 0); });
    // The GR adiabat is a function of the primitives, which on a restart exist only
    // once the inversion above has run.
    if (phydro->use_dual_energy && !phydro->dual_energy_pdv &&
        phydro->dual_energy_needs_init) {
      phydro->InitializeDualEnergyFieldFromAdiabat();
      phydro->dual_energy_needs_init = false;
    }
  }

  // Initialize MHD: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  mhd::MHD *pmhd = pm->pmb_pack->pmhd;
  dyngr::DynGRMHD *pdyngr = pm->pmb_pack->pdyngr;
  if (pmhd != nullptr) {
    // Only the non-relativistic auxiliary can be seeded here, from the conserved state
    // alone.  The flag must not be cleared on the GR path: that seeding happens after
    // the inversion below, and clearing it here left the adiabat at its zero
    // initialization, so every cell the eta1 test routed to the auxiliary channel was
    // handed p = 0.
    if (pmhd->use_dual_energy && pmhd->dual_energy_pdv &&
        pmhd->dual_energy_needs_init) {
      pmhd->InitializeDualEnergyFieldFromTotal();
      pmhd->dual_energy_needs_init = false;
    }
    (void) pmhd->RestrictU(this, 0);
    (void) pmhd->RestrictB(this, 0);
    // The restrictions above rebuild only the part of the coarse registers a block
    // derives from its own interior.  Their ghost band (coarse_u0/coarse_b0 filled from
    // coarser neighbours, the coarse boundary fill, coarse_w0 under prolong_primitives)
    // comes from the end-of-step exchange, and a prolongation that runs before the next
    // exchange of that field reads it: the imex_gas_at_v2 pre-stage publication moves
    // U only and prolongates b0 from the carried coarse_b0.  Rebuild the whole coarse
    // state with the ordinary startup exchange and then put the checkpointed u0/b0
    // ghost zones back, so the fine arrays stay verbatim.
    if (restart_verbatim && pm->multilevel) {
      DvceArray5D<Real> u_ghosts("restart_mhd_u0", nmb_live_rst,
                                 pmhd->u0.extent(1), pmhd->u0.extent(2),
                                 pmhd->u0.extent(3), pmhd->u0.extent(4));
      DvceFaceFld4D<Real> b_ghosts("restart_mhd_b0", nmb_live_rst,
                                   pmhd->b0.x1f.extent(1), pmhd->b0.x1f.extent(2),
                                   pmhd->b0.x2f.extent(3));
      Kokkos::deep_copy(u_ghosts, live_blocks(pmhd->u0));
      Kokkos::deep_copy(b_ghosts.x1f, live_blocks(pmhd->b0.x1f));
      Kokkos::deep_copy(b_ghosts.x2f, live_blocks(pmhd->b0.x2f));
      Kokkos::deep_copy(b_ghosts.x3f, live_blocks(pmhd->b0.x3f));
      auto put_back = [&]() {
        Kokkos::deep_copy(live_blocks(pmhd->u0), u_ghosts);
        Kokkos::deep_copy(live_blocks(pmhd->b0.x1f), b_ghosts.x1f);
        Kokkos::deep_copy(live_blocks(pmhd->b0.x2f), b_ghosts.x2f);
        Kokkos::deep_copy(live_blocks(pmhd->b0.x3f), b_ghosts.x3f);
      };
      (void) pmhd->InitRecv(this, -1);
      (void) pmhd->SendU(this, 0);
      (void) pmhd->SendB(this, 0);
      (void) pmhd->ClearSend(this, -1);
      (void) pmhd->ClearRecv(this, -1);
      (void) pmhd->RecvU(this, 0);
      (void) pmhd->RecvB(this, 0);
      // A narrow sender packed its coarse ghost cells, still empty here, into its
      // neighbours' fine ghost zones, which Prolongate restricts: restrict the
      // checkpointed ones instead (rebuild_coarse above).
      if (narrow_blocks) put_back();
      (void) pmhd->Prolongate(this, 0);
      (void) pmhd->ApplyPhysicalBCs(this, 0);
      put_back();
    }
    for (int pass=0; pass<npass; ++pass) {
      (void) pmhd->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
      (void) pmhd->SendU(this, 0);
      (void) pmhd->SendB(this, 0);
      (void) pmhd->ClearSend(this, -1); // stage = -1 only clear SendU, SendB
      (void) pmhd->ClearRecv(this, -1); // stage = -1 only clear RecvU, RecvB
      (void) pmhd->RecvU(this, 0);
      (void) pmhd->RecvB(this, 0);
      (void) pmhd->SendU_Shr(this, 0);
      (void) pmhd->SendB_Shr(this, 0);
      (void) pmhd->ClearSend(this, -4); // stage = -4 only clear SendU_Shr, SendB_Shr
      (void) pmhd->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr, SendB_Shr
      (void) pmhd->RecvU_Shr(this, 0);
      (void) pmhd->RecvB_Shr(this, 0);
      if (pass+1 < npass) {
        // The last pass is completed below, as a single one always was.
        (void) pmhd->Prolongate(this, 0);
        (void) pmhd->ApplyPhysicalBCs(this, 0);
      }
    }
    if (pdyngr == nullptr) {
      if (exchange) {
        (void) pmhd->Prolongate(this, 0);
        (void) pmhd->ApplyPhysicalBCs(this, 0);
      }
      if (repair_amr_fc) {
        pm->pmr->RepairAMRFC(pmhd->b0);
      }
      recover_primitives(pmhd->u0, [&]() { (void) pmhd->ConToPrim(this, 0); });
    } else {
      if (exchange) {
        (void) pmhd->Prolongate(this, 0);
        (void) pmhd->ApplyPhysicalBCs(this, 0);
      }
      if (repair_amr_fc) {
        pm->pmr->RepairAMRFC(pmhd->b0);
      }
      if (pz4c != nullptr) {
        (void) pz4c->ConvertZ4cToADM(this, 0);
      }
      recover_primitives(pmhd->u0,
                         [&]() { (void) pdyngr->ConToPrim(this, 0); });
    }
    // The GR dual-energy auxiliary is seeded HERE, not with the non-relativistic one
    // above: kappa = p/rho^Gamma is a function of the primitives, and on a restart the
    // primitives do not exist until the inversion above has run.  Seeding it earlier
    // would read whatever w0 happened to hold -- correct on a fresh start, garbage on a
    // restart that has to rebuild the auxiliary because the checkpoint predates it.
    if (pmhd->use_dual_energy && !pmhd->dual_energy_pdv &&
        pmhd->dual_energy_needs_init) {
      pmhd->InitializeDualEnergyFieldFromAdiabat();
      pmhd->dual_energy_needs_init = false;
    }
  }

  // Initialize radiation: ghost zones and intensity (everywhere)
  // DOES NOT include communications for shearing box boundaries
  radiation::Radiation *prad = pm->pmb_pack->prad;
  if (prad != nullptr) {
    (void) prad->RestrictI(this, 0);
    for (int pass=0; pass<npass; ++pass) {
      (void) prad->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
      (void) prad->SendI(this, 0);
      (void) prad->ClearSend(this, -1);
      (void) prad->ClearRecv(this, -1);
      (void) prad->RecvI(this, 0);
      (void) prad->Prolongate(this, 0);
      (void) prad->ApplyPhysicalBCs(this, 0);
    }
    if (rebuild_coarse) {
      rebuild_coarse_ghosts(prad->i0, [&]() {
        (void) prad->InitRecv(this, -1);
        (void) prad->SendI(this, 0);
        (void) prad->ClearSend(this, -1);
        (void) prad->ClearRecv(this, -1);
        (void) prad->RecvI(this, 0);
      }, [&]() { (void) prad->Prolongate(this, 0); });
    }
  }

  return;
}
