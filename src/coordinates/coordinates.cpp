//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.cpp
//! \brief
#include <float.h> // FLT_MIN

#include <iostream> // cout
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "cartesian_ks.hpp"
#include "coordinates.hpp"
#include "cell_locations.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn void ComputeKSCoordSourceTerms
//! \brief geometric source term s_i = 1/2 (d_i g_{mn}) T^{mn} for the Cartesian
//! Kerr-Schild metric, contracted without ever forming d_i g_{mn} or T^{mn}.
//!
//! The CKS metric is a rank-1 update of Minkowski, g_{mn} = eta_{mn} + f l_m l_n with
//! l_0 = 1 (see ComputeMetricAndInverse in cartesian_ks.hpp), hence
//!   d_i g_{mn} = (d_i f) l_m l_n + f [(d_i l_m) l_n + l_m (d_i l_n)].
//! T^{mn} is symmetric, so the 30-term contraction collapses to
//!   s_i = 1/2 (d_i f) L + f (d_i l_m) w^m,  w^m = T^{mn} l_n,  L = l_m w^m,
//! where the second sum runs over m = 1,2,3 only because l_0 = 1 gives d_i l_0 = 0.
//! w and L do not depend on i, so they are built once.  With
//! T^{mn} = wtot u^m u^n + ptot g^{mn} - b^m b^n and g^{mn} = eta^{mn} - f l^m l^n
//! (l^0 = -1, l^i = l_i) neither T nor the inverse metric is needed either:
//!   g^{mn} l_n    = (1 - f (l.l)) l^m
//!   l_m g^{mn} l_n = (1 - f (l.l)) (l.l),    (l.l) = l^n l_n = l_1^2+l_2^2+l_3^2 - 1.
//! (l.l) vanishes analytically -- l is null -- but it is evaluated rather than assumed:
//! inside the r < 1e-6 floor applied below the floored r breaks the identity outright
//! ((l.l) -> -1 there), so evaluating it costs four flops and keeps the contraction
//! exact everywhere.
//!
//! Only w^1..w^3, L and one direction's worth of d_i f and d_i l_j are ever live, so no
//! 4x4 metric gradient or stress-energy tensor is held in registers.  Expressions for
//! r, l_j, f, df_dx* and dl*_dx* are copied verbatim from ComputeMetricAndInverse and
//! ComputeMetricDerivatives in cartesian_ks.hpp.  Pass b0..b3 = 0 for hydrodynamics.

KOKKOS_INLINE_FUNCTION
void ComputeKSCoordSourceTerms(const Real x, const Real y, const Real z,
                               const bool minkowski, const Real a,
                               const Real wtot, const Real ptot,
                               const Real u0, const Real u1,
                               const Real u2, const Real u3,
                               const Real b0, const Real b1,
                               const Real b2, const Real b3,
                               Real *ps_1, Real *ps_2, Real *ps_3) {
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  Real eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }

  // spatial components of the null covector l (l_0 = 1 is constant)
  Real llower1 = (r*x + a * y)/( SQR(r) + SQR(a) );
  Real llower2 = (r*y - a * x)/( SQR(r) + SQR(a) );
  Real llower3 = z/r;

  Real qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real qb = SQR(r) + SQR(a);
  Real qc = 3.0*SQR(a * z)-SQR(r)*SQR(r);
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));

  Real df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r));

  if (minkowski) {
    f = 0.0;
    df_dx1 = 0.0;
    df_dx2 = 0.0;
    df_dx3 = 0.0;
  }

  // w^m = T^{mn} l_n and L = l_m w^m, shared by all three directions.  w^0 is never
  // referenced below, so it is not formed.
  Real lsq = SQR(llower1) + SQR(llower2) + SQR(llower3) - 1.0;
  Real gfac = 1.0 - f*lsq;
  Real u_dot_l = u0 + u1*llower1 + u2*llower2 + u3*llower3;
  Real b_dot_l = b0 + b1*llower1 + b2*llower2 + b3*llower3;
  Real w1 = wtot*u1*u_dot_l + ptot*gfac*llower1 - b1*b_dot_l;
  Real w2 = wtot*u2*u_dot_l + ptot*gfac*llower2 - b2*b_dot_l;
  Real w3 = wtot*u3*u_dot_l + ptot*gfac*llower3 - b3*b_dot_l;
  Real lw = wtot*SQR(u_dot_l) + ptot*gfac*lsq - SQR(b_dot_l);

  // one direction at a time, so only one d_i f and one d_i l_j triple is ever live
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb );
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) - a/( qb );
  Real dl3_dx1 = - x*z/(r*qa);
  *ps_1 = 0.5*df_dx1*lw + f*(dl1_dx1*w1 + dl2_dx1*w2 + dl3_dx1*w3);

  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + a/( qb );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) + r/( qb );
  Real dl3_dx2 = - y*z/(r*qa);
  *ps_2 = 0.5*df_dx2*lw + f*(dl1_dx2*w1 + dl2_dx2*w2 + dl3_dx2*w3);

  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (qb) * qa );
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r;
  *ps_3 = 0.5*df_dx3*lw + f*(dl1_dx3*w1 + dl2_dx3*w2 + dl3_dx3*w3);
}

}  // namespace

//----------------------------------------------------------------------------------------
// constructor, initializes coordinates data

// excision_floor/excision_flux are deliberately left default-constructed here.  They are
// only ever allocated, written or read under coord_data.bh_excise (every consumer guards
// on it, several by returning early), and the placeholder 1x1x1x1 Views this constructor
// used to create were two device allocations that no non-excising run can use.
Coordinates::Coordinates(ParameterInput *pin, MeshBlockPack *ppack) :
    pmy_pack(ppack) {
  // Check for relativistic dynamics
  // WGC: idea for handling new EOS
  is_dynamical_relativistic = (pin->DoesBlockExist("adm") || pin->DoesBlockExist("z4c"))
                         && (pin->DoesBlockExist("hydro") || pin->DoesBlockExist("mhd"));
  if(!is_dynamical_relativistic) {
    is_special_relativistic = pin->GetOrAddBoolean("coord","special_rel",false);
    is_general_relativistic = pin->GetOrAddBoolean("coord","general_rel",false);
  } else {
    is_special_relativistic = is_general_relativistic = false;
  }
  if (is_special_relativistic && is_general_relativistic) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Cannot specify both SR and GR at same time" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Read properties of metric and excision from input file for GR.
  if (is_general_relativistic || is_dynamical_relativistic) {
    coord_data.is_minkowski = pin->GetOrAddBoolean("coord","minkowski",false);
    if (!(coord_data.is_minkowski)) {
      coord_data.bh_spin = pin->GetReal("coord","a");
      coord_data.bh_excise = pin->GetOrAddBoolean("coord","excise",true);
    } else {
      coord_data.bh_spin = 0.0;
      coord_data.bh_excise = false;
    }

    if (coord_data.bh_excise) {
      // Set the density and pressure to which cells inside the excision radius will
      // be reset to.  Primitive velocities will be set to zero.
      coord_data.dexcise = pin->GetReal("coord","dexcise");
      if (is_dynamical_relativistic) {
        coord_data.texcise = pin->GetReal("coord", "texcise");
        // <coord>/pexcise is a fixed-metric-only knob.  On this path the excised state is
        // (dexcise, v = 0, T = texcise) with the pressure from the EOS, so a pexcise
        // in the deck is read by nothing at all -- and a deck that sets it (the OJ287 one
        // sets 1e-15) reads as if it were tuning the excision atmosphere.  Warn rather
        // than fail: an inert key must not stop a production restart.
        if (pin->DoesParameterExist("coord", "pexcise") &&
            global_variable::my_rank == 0) {
          std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<coord>/pexcise is not read on the dynamical-GR "
                    << "path and has no effect: excised cells are set to (dexcise, v=0, "
                    << "T=texcise), with the pressure following from the EOS.  Use "
                    << "<coord>/texcise to set the excised thermal state." << std::endl;
        }
      } else {
        coord_data.pexcise = pin->GetReal("coord", "pexcise");
      }

      const bool radiation_excises_horizon = pin->DoesBlockExist("radiation");
      coord_data.flux_excise_r = radiation_excises_horizon ?
        1.0+sqrt(1.0-SQR(coord_data.bh_spin)) :
        pin->GetOrAddReal("coord","flux_excise_r",1.0);
      coord_data.rexcise =
        radiation_excises_horizon ? 1.0+sqrt(1.0-SQR(coord_data.bh_spin)) : 1.0;

      coord_data.excision_scheme = ExcisionScheme::fixed;
      if (is_dynamical_relativistic) {
        std::string emethod = pin->GetOrAddString("coord","excision_scheme","fixed");
        if (emethod.compare("fixed") == 0) {
          coord_data.excision_scheme = ExcisionScheme::fixed;
        } else if (emethod.compare("lapse") == 0) {
          coord_data.excision_scheme = ExcisionScheme::lapse;
          coord_data.excise_lapse = pin->GetOrAddReal("coord","excise_lapse", 0.25);
        } else if (emethod.compare("horizon") == 0) {
          if (pin->DoesBlockExist("fastflow")) {
            coord_data.excision_scheme = ExcisionScheme::horizon;
            coord_data.horizon_factor = pin->GetOrAddReal("coord","horizon_factor",1.0);
          } else {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                    << __LINE__ << std::endl
                    << "Horizon excision needs <fastflow> block!" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                    << __LINE__ << std::endl
                    << "Unknown excision method: " << emethod << std::endl;
          std::exit(EXIT_FAILURE);
        }

        // Smooth excision.
        coord_data.smooth_excision = pin->GetOrAddBoolean("coord","smooth_excision",
                                                              false);
        coord_data.tdamp = pin->GetOrAddReal("coord","tdamp",1.0);
      }

      // boolean masks allocation
      int nmb = ppack->nmb_thispack;
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      // Constructed rather than Kokkos::realloc'd: realloc inherits the View's existing
      // label, and these are now default-constructed, so the allocations would have been
      // anonymous in profiling and in any out-of-memory message.
      excision_floor =
          DvceArray4D<bool>("excision_floor", nmb, ncells3, ncells2, ncells1);
      excision_flux =
          DvceArray4D<bool>("excision_flux", nmb, ncells3, ncells2, ncells1);
      if (coord_data.excision_scheme == ExcisionScheme::fixed) {
        SetExcisionMasks(excision_floor, excision_flux);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR hydrodynamics

void Coordinates::CoordSrcTerms(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = coord_data.is_minkowski;
  auto &spin = coord_data.bh_spin;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real dt_scale = (lat_per_block_dt && mesh_dt != 0.0) ? dt/mesh_dt : 0.0;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) :
                                  (pmy_pack->nmb_thispack - 1);
  if (nwork1 < 0) return;
  par_for("coord_src", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_dt = lat_per_block_dt ? dt_scale*lat_step_dt(m) : dt;
    // Extract components of metric
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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Extract primitives
    const Real &rho  = prim(m,IDN,k,j,i);
    const Real &uu1  = prim(m,IVX,k,j,i);
    const Real &uu2  = prim(m,IVY,k,j,i);
    const Real &uu3  = prim(m,IVZ,k,j,i);
    Real pgas = eos.IdealGasPressure(prim(m,IEN,k,j,i));

    // Calculate 4-velocity (exploiting symmetry of metric)
    Real uu_sq = glower[1][1]*uu1*uu1 +2.0*glower[1][2]*uu1*uu2 +2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2 +2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + uu_sq);
    Real u0 = gamma / alpha;
    Real u1 = uu1 - alpha * gamma * gupper[0][1];
    Real u2 = uu2 - alpha * gamma * gupper[0][2];
    Real u3 = uu3 - alpha * gamma * gupper[0][3];

    // Stress-energy tensor coefficients.  T^{mn} itself is never formed: the
    // contraction with the metric derivative only needs w^m = T^{mn} l_n (see
    // ComputeKSCoordSourceTerms), which is built from wtot and ptot directly.
    Real wtot = rho + gamma_prime * pgas;
    Real ptot = pgas;

    // Calculate source terms s_i = 0.5*(d_i g_{mn})*T^{mn}.  b^m = 0 for hydro.
    Real s_1, s_2, s_3;
    ComputeKSCoordSourceTerms(x1v, x2v, x3v, flat, spin, wtot, ptot,
                              u0, u1, u2, u3, 0.0, 0.0, 0.0, 0.0,
                              &s_1, &s_2, &s_3);

    // Add source terms to conserved quantities
    cons(m,IM1,k,j,i) += block_dt * s_1;
    cons(m,IM2,k,j,i) += block_dt * s_2;
    cons(m,IM3,k,j,i) += block_dt * s_3;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR MHD
//
// TODO(@user): Most of this function just copies the Hydro version.  Only difference is
// the inclusion of the magnetic field in computing the stress-energy tensor.  There must
// be a smarter way to generalize these two functions and avoid duplicated code.
// Functions distinguished only by argument list.

void Coordinates::CoordSrcTerms(const DvceArray5D<Real> &prim,
                                const DvceArray5D<Real> &bcc, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = coord_data.is_minkowski;
  auto &spin = coord_data.bh_spin;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  const int nwork1 = pmy_pack->nmb_thispack - 1;
  if (nwork1 < 0) return;
  par_for("coord_src", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Extract components of metric
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
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Extract primitives
    const Real &rho  = prim(m,IDN,k,j,i);
    const Real &uu1  = prim(m,IVX,k,j,i);
    const Real &uu2  = prim(m,IVY,k,j,i);
    const Real &uu3  = prim(m,IVZ,k,j,i);
    Real pgas = eos.IdealGasPressure(prim(m,IEN,k,j,i));

    // Calculate 4-velocity
    Real uu_sq = glower[1][1]*uu1*uu1 +2.0*glower[1][2]*uu1*uu2 +2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2 +2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + uu_sq);
    Real u0 = gamma / alpha;
    Real u1 = uu1 - alpha * gamma * gupper[0][1];
    Real u2 = uu2 - alpha * gamma * gupper[0][2];
    Real u3 = uu3 - alpha * gamma * gupper[0][3];

    // lower vector indices
    Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
    Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
    Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

    // calculate 4-magnetic field
    const Real &bb1 = bcc(m,IBX,k,j,i);
    const Real &bb2 = bcc(m,IBY,k,j,i);
    const Real &bb3 = bcc(m,IBZ,k,j,i);
    Real b0 = u_1*bb1 + u_2*bb2 + u_3*bb3;
    Real b1 = (bb1 + b0 * u1) / u0;
    Real b2 = (bb2 + b0 * u2) / u0;
    Real b3 = (bb3 + b0 * u3) / u0;

    // lower vector indices
    Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
    Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
    Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
    Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
    Real b_sq = b_0*b0 + b_1*b1 + b_2*b2 + b_3*b3;

    // Stress-energy tensor coefficients.  T^{mn} itself is never formed: the
    // contraction with the metric derivative only needs w^m = T^{mn} l_n (see
    // ComputeKSCoordSourceTerms), which is built from wtot, ptot and b^m directly.
    Real wtot = rho + gamma_prime * pgas + b_sq;
    Real ptot = pgas + 0.5*b_sq;

    // Calculate source terms s_i = 0.5*(d_i g_{mn})*T^{mn}
    Real s_1, s_2, s_3;
    ComputeKSCoordSourceTerms(x1v, x2v, x3v, flat, spin, wtot, ptot,
                              u0, u1, u2, u3, b0, b1, b2, b3,
                              &s_1, &s_2, &s_3);

    // Add source terms to conserved quantities
    cons(m,IM1,k,j,i) += dt * s_1;
    cons(m,IM2,k,j,i) += dt * s_2;
    cons(m,IM3,k,j,i) += dt * s_3;
  });

  return;
}
