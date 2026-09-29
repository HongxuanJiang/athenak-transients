#ifndef COORDINATES_CARTESIAN_KS_HPP_
#define COORDINATES_CARTESIAN_KS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cartesian_ks.hpp
//! \brief implements functions for Cartesian Kerr-Schild coordinates in GR.  This
//! includes inline functions to compute the metric and derivatives of the metric.  Based
//! on functions in 'gr_user.cpp' file in Athena++, as well as CartesianGR.cpp function
//! from CJW and SR.

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"

// #define SMALL_NUMBER 1.0e-5

KOKKOS_INLINE_FUNCTION
Real CartesianKerrSchildRadius(const Real x, const Real y, const Real z,
                               const Real spin) {
  const Real radius_cartesian = sqrt(SQR(x) + SQR(y) + SQR(z));
  const Real r2_minus_a2 = SQR(radius_cartesian) - SQR(spin);
  return sqrt(0.5*(r2_minus_a2 +
      sqrt(SQR(r2_minus_a2) + 4.0*SQR(spin)*SQR(z))));
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricAndInverse
//! \brief computes 10 covariant and contravariant components of metric in
//!  Cartesian Kerr-Schild coordinates

KOKKOS_INLINE_FUNCTION
void ComputeMetricAndInverse(Real x, Real y, Real z, bool minkowski, Real a,
                             Real glower[][4], Real gupper[][4]) {
  // NOTE(@pdmullen): The following commented out floor on z dealt with the metric
  // singularity encountered for small z near the horizon (e.g., see g_00). However, this
  // floor was operating on z even for r_ks > 1.0, where (I believe) the metric should be
  // well-behaved for all z.  We floor r_ks to 1.0, therefore, the floor on z is
  // seemingly not necessary, however, if something goes awry for z ~ 0 in future
  // applications (even after flooring r_ks = 1.0), consider revisiting this z floor...
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  Real eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  //r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  // Set covariant components
  // null vector l
  Real l_lower[4];
  l_lower[0] = 1.0;
  l_lower[1] = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  l_lower[2] = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  l_lower[3] = z/r;

  // g_nm = f*l_n*l_m + eta_nm, where eta_nm is Minkowski metric
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {f=0.0;}
  glower[0][0] = f * l_lower[0]*l_lower[0] - 1.0;
  glower[0][1] = f * l_lower[0]*l_lower[1];
  glower[0][2] = f * l_lower[0]*l_lower[2];
  glower[0][3] = f * l_lower[0]*l_lower[3];
  glower[1][0] = glower[0][1];
  glower[1][1] = f * l_lower[1]*l_lower[1] + 1.0;
  glower[1][2] = f * l_lower[1]*l_lower[2];
  glower[1][3] = f * l_lower[1]*l_lower[3];
  glower[2][0] = glower[0][2];
  glower[2][1] = glower[1][2];
  glower[2][2] = f * l_lower[2]*l_lower[2] + 1.0;
  glower[2][3] = f * l_lower[2]*l_lower[3];
  glower[3][0] = glower[0][3];
  glower[3][1] = glower[1][3];
  glower[3][2] = glower[2][3];
  glower[3][3] = f * l_lower[3]*l_lower[3] + 1.0;

  // Set contravariant components
  // null vector l
  Real l_upper[4];
  l_upper[0] = -1.0;
  l_upper[1] = l_lower[1];
  l_upper[2] = l_lower[2];
  l_upper[3] = l_lower[3];

  // g^nm = -f*l^n*l^m + eta^nm, where eta^nm is Minkowski metric
  gupper[0][0] = -f * l_upper[0]*l_upper[0] - 1.0;
  gupper[0][1] = -f * l_upper[0]*l_upper[1];
  gupper[0][2] = -f * l_upper[0]*l_upper[2];
  gupper[0][3] = -f * l_upper[0]*l_upper[3];
  gupper[1][0] = gupper[0][1];
  gupper[1][1] = -f * l_upper[1]*l_upper[1] + 1.0;
  gupper[1][2] = -f * l_upper[1]*l_upper[2];
  gupper[1][3] = -f * l_upper[1]*l_upper[3];
  gupper[2][0] = gupper[0][2];
  gupper[2][1] = gupper[1][2];
  gupper[2][2] = -f * l_upper[2]*l_upper[2] + 1.0;
  gupper[2][3] = -f * l_upper[2]*l_upper[3];
  gupper[3][0] = gupper[0][3];
  gupper[3][1] = gupper[1][3];
  gupper[3][2] = gupper[2][3];
  gupper[3][3] = -f * l_upper[3]*l_upper[3] + 1.0;

  return;
}

//----------------------------------------------------------------------------------------
//! \struct KSNullForm
//! \brief compact representation of the Cartesian Kerr-Schild metric.
//!
//! The CKS metric is a rank-1 update of Minkowski,
//!     g_{mu nu} = eta_{mu nu} + f l_mu l_nu ,   l_0 = 1
//!     g^{mu nu} = eta^{mu nu} - f l^mu l^nu ,   l^0 = -1, l^i = l_i
//! so the whole geometry at a point is carried by the four numbers (f, l_1, l_2, l_3)
//! instead of the 32 doubles of glower[4][4]/gupper[4][4].  Every metric component the
//! Riemann solvers need is recovered exactly by the accessors below.

struct KSNullForm {
  Real f, l1, l2, l3;
};

//----------------------------------------------------------------------------------------
//! \fn void ComputeKSNullForm
//! \brief computes (f, l_i) for Cartesian Kerr-Schild coordinates.
//!
//! The expressions (and their evaluation order) are character-for-character those used in
//! ComputeMetricAndInverse above, so f and l_i are bitwise identical to that routine.

KOKKOS_INLINE_FUNCTION
void ComputeKSNullForm(Real x, Real y, Real z, bool minkowski, Real a, KSNullForm &nf) {
  // NOTE(@pdmullen): See comment in ComputeMetricAndInverse
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  Real eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }

  // null vector l (spatial covariant components; l_0 = 1 identically)
  nf.l1 = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  nf.l2 = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  nf.l3 = z/r;

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {f=0.0;}
  nf.f = f;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real KSLowerNull
//! \brief returns l_i for i in {1,2,3} (index-generic; folds away for constexpr i)

KOKKOS_INLINE_FUNCTION
Real KSLowerNull(const KSNullForm &nf, const int i) {
  return (i == 1) ? nf.l1 : ((i == 2) ? nf.l2 : nf.l3);
}

//----------------------------------------------------------------------------------------
//! \fn Real KSAlpha
//! \brief lapse alpha = sqrt(-1/g^{00}).  Since g^{00} = -f (l^0)^2 - 1 = -(1+f) this is
//! sqrt(1/(1+f)), which is bitwise identical to the sqrt(-1.0/gupper[0][0]) form.

KOKKOS_INLINE_FUNCTION
Real KSAlpha(const KSNullForm &nf) {
  return sqrt(1.0/(1.0 + nf.f));
}

//----------------------------------------------------------------------------------------
//! \fn Real KSGupper00 / KSGupper0i / KSGupperij
//! \brief contravariant metric components from the null form.
//!   g^{00} = -(1+f),  g^{0i} = +f l_i,  g^{ij} = delta^{ij} - f l_i l_j

KOKKOS_INLINE_FUNCTION
Real KSGupper00(const KSNullForm &nf) {
  return -nf.f - 1.0;
}

KOKKOS_INLINE_FUNCTION
Real KSGupper0i(const KSNullForm &nf, const int i) {
  return nf.f*KSLowerNull(nf, i);
}

KOKKOS_INLINE_FUNCTION
Real KSGupperij(const KSNullForm &nf, const int i, const int j) {
  return -(nf.f*KSLowerNull(nf, i))*KSLowerNull(nf, j) + ((i == j) ? 1.0 : 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real KSGlower00 / KSGlower0i / KSGlowerij
//! \brief covariant metric components from the null form.
//!   g_{00} = f - 1,  g_{0i} = f l_i,  g_{ij} = delta_{ij} + f l_i l_j
//! Each expression evaluates in the same order as ComputeMetricAndInverse, so the
//! returned value is bitwise identical to the corresponding glower[][] entry.  Note
//! nothing here assumes the null identity l.l = 1: the r < 1e-6 floor in
//! ComputeKSNullForm breaks it, and these are exact rank-1 identities regardless.

KOKKOS_INLINE_FUNCTION
Real KSGlower00(const KSNullForm &nf) {
  return nf.f - 1.0;
}

KOKKOS_INLINE_FUNCTION
Real KSGlower0i(const KSNullForm &nf, const int i) {
  return nf.f*KSLowerNull(nf, i);
}

KOKKOS_INLINE_FUNCTION
Real KSGlowerij(const KSNullForm &nf, const int i, const int j) {
  return (nf.f*KSLowerNull(nf, i))*KSLowerNull(nf, j) + ((i == j) ? 1.0 : 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn void KSFillLower
//! \brief materializes the full 4x4 covariant metric from the null form, bitwise
//! identical to the glower[][] that ComputeMetricAndInverse would produce.  This exists
//! only as an adapter for callees whose signature is fixed as `const Real g[][4]`; it is
//! deliberately NOT used on any path that can consume the null form directly, because
//! the whole point of KSNullForm is to keep 4 live doubles instead of 16.

KOKKOS_INLINE_FUNCTION
void KSFillLower(const KSNullForm &nf, Real glower[][4]) {
  const Real f = nf.f;
  glower[0][0] = f - 1.0;
  glower[0][1] = f*nf.l1;
  glower[0][2] = f*nf.l2;
  glower[0][3] = f*nf.l3;
  glower[1][0] = glower[0][1];
  glower[1][1] = (f*nf.l1)*nf.l1 + 1.0;
  glower[1][2] = (f*nf.l1)*nf.l2;
  glower[1][3] = (f*nf.l1)*nf.l3;
  glower[2][0] = glower[0][2];
  glower[2][1] = glower[1][2];
  glower[2][2] = (f*nf.l2)*nf.l2 + 1.0;
  glower[2][3] = (f*nf.l2)*nf.l3;
  glower[3][0] = glower[0][3];
  glower[3][1] = glower[1][3];
  glower[3][2] = glower[2][3];
  glower[3][3] = (f*nf.l3)*nf.l3 + 1.0;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real KSInvLapseSq / KSGammaUpperFactor
//! \brief scalars the GR->SR conserved transform needs.
//!   -g^{00} = 1 + f  (= 1/alpha^2)
//!   gamma^{ij} = g^{ij} - g^{0i}g^{0j}/g^{00} = delta^{ij} - [f/(1+f)] l_i l_j
//! The bracket f/(1+f) lies in [0,1) for every f >= 0, so raising a spatial index this
//! way never forms the two large, nearly cancelling terms (-f l_i l_j and +f^2 l_i l_j
//! /(1+f)) that the explicit g^{ij} route subtracts when f >> 1 near the horizon.

KOKKOS_INLINE_FUNCTION
Real KSInvLapseSq(const KSNullForm &nf) {
  return 1.0 + nf.f;
}

KOKKOS_INLINE_FUNCTION
Real KSGammaUpperFactor(const KSNullForm &nf) {
  return nf.f/(1.0 + nf.f);
}

//----------------------------------------------------------------------------------------
//! \fn Real KSLDotVec
//! \brief l_mu A^mu = A^0 + l_i A^i (l_0 = 1), with (ix,iy,iz) a permutation of {1,2,3}
//! and (a0,ax,ay,az) the contravariant components in that same permuted order.

KOKKOS_INLINE_FUNCTION
Real KSLDotVec(const KSNullForm &nf, const int ix, const int iy, const int iz,
               const Real a0, const Real ax, const Real ay, const Real az) {
  return a0 + KSLowerNull(nf,ix)*ax + KSLowerNull(nf,iy)*ay + KSLowerNull(nf,iz)*az;
}

//----------------------------------------------------------------------------------------
//! \fn void KSLowerVec
//! \brief lowers a contravariant 4-vector:  A_mu = eta_{mu nu} A^nu + f l_mu (l.A).
//! Equivalent to the full 16-multiply glower[mu][nu]*A^nu contraction.

KOKKOS_INLINE_FUNCTION
void KSLowerVec(const KSNullForm &nf, const int ix, const int iy, const int iz,
                const Real a0, const Real ax, const Real ay, const Real az,
                Real &b0, Real &bx, Real &by, Real &bz) {
  const Real fld = nf.f*KSLDotVec(nf, ix, iy, iz, a0, ax, ay, az);
  b0 = fld - a0;
  bx = KSLowerNull(nf,ix)*fld + ax;
  by = KSLowerNull(nf,iy)*fld + ay;
  bz = KSLowerNull(nf,iz)*fld + az;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real KSSpatialNormSq
//! \brief q = gamma_{ij} v^i v^j = |v|^2_euclid + f (l_i v^i)^2, replacing the six-term
//! glower[i][j] contraction.

KOKKOS_INLINE_FUNCTION
Real KSSpatialNormSq(const KSNullForm &nf, const int ix, const int iy, const int iz,
                     const Real vx, const Real vy, const Real vz) {
  const Real ldv = KSLowerNull(nf,ix)*vx + KSLowerNull(nf,iy)*vy + KSLowerNull(nf,iz)*vz;
  return SQR(vx) + SQR(vy) + SQR(vz) + nf.f*SQR(ldv);
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeADMDecompositionAndDerivatives
//! \brief computes ADM quantities in Cartesian Kerr-Schild coordinates, plus the
//! analytic first derivatives that 3+1 source terms need:
//!   dalp_d[i]      = partial_i alpha
//!   dbeta_du[i][j] = partial_i beta^j   (upper shift components)
//!   dg_sp[i][n]    = partial_i gamma_(jk), n packed in S11..S33 order
//! The intermediates (H, l_k and their gradients) were always computed here for the
//! extrinsic curvature; this variant just exposes the derived derivatives so callers
//! (e.g. the M1 geometric sources) do not have to finite-difference the exact metric.

// QUESTION: doesn't this assume bh_mass to be 1?
KOKKOS_INLINE_FUNCTION
void ComputeADMDecompositionAndDerivatives(Real x, Real y, Real z, bool minkowski, Real a,
               Real * alp,
               Real * betax, Real * betay, Real * betaz,
               Real * psi4,
               Real * gxx, Real * gxy, Real * gxz, Real * gyy, Real * gyz, Real * gzz,
               Real * Kxx, Real * Kxy, Real * Kxz, Real * Kyy, Real * Kyz, Real * Kzz,
               Real dalp_d[3], Real dbeta_du[3][3], Real dg_sp[3][6]) {
  // See comments above in ComputeMetricAndInverse
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  Real eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  //r = fmax(r, 1.0);

  // l covector (spatial components only)
  Real l_d[3];
  l_d[0] = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  l_d[1] = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  l_d[2] = z/r;

  // l vector (spatial components only)
  Real l_u[3] = {l_d[0], l_d[1], l_d[2]};

  //
  // g_nm = 2*H*l_n*l_m + eta_nm, where eta is the Minkowski metric
  Real H = SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {H=0.0;}

  *alp = 1.0/sqrt(1. + 2.*H);
  *betax = 2.*H/(1. + 2.*H)*l_u[0];
  *betay = 2.*H/(1. + 2.*H)*l_u[1];
  *betaz = 2.*H/(1. + 2.*H)*l_u[2];
  Real const beta_d[3] = {2.*H*l_u[0], 2.*H*l_u[1], 2.*H*l_u[2]};

  *gxx = 2.*H*l_d[0]*l_d[0] + 1.;
  *gxy = 2.*H*l_d[0]*l_d[1];
  *gxz = 2.*H*l_d[0]*l_d[2];
  *gyy = 2.*H*l_d[1]*l_d[1] + 1.;
  *gyz = 2.*H*l_d[1]*l_d[2];
  *gzz = 2.*H*l_d[2]*l_d[2] + 1.;

  //
  // conformal factor
  Real const det = adm::SpatialDet(*gxx, *gxy, *gxz, *gyy, *gyz, *gzz);
  *psi4 = pow(det, 1./3.);

  //
  // inverse metric
  Real uxx, uxy, uxz, uyy, uyz, uzz;
  adm::SpatialInv(1./det,
             *gxx, *gxy, *gxz, *gyy, *gyz, *gzz,
             &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
  Real const g_uu[3][3] = {
    {uxx, uxy, uxz},
    {uxy, uyy, uyz},
    {uxz, uyz, uzz}
  };

  //
  // derivatives of the three metric (expressions taken from below)
  Real const qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real const qb = SQR(r) + SQR(a);
  Real const qc = 3.0*SQR(a * z) - SQR(r)*SQR(r);
  Real const dH_d[3] = {
    SQR(H)*x/(pow(r,3)) * ( ( qc ) )/ qa,
    SQR(H)*y/(pow(r,3)) * ( ( qc ) )/ qa,
    SQR(H)*z/(pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r))
  };

  // \partial_i l_k
  Real const dl_dd[3][3] = {
    // \partial_x l_k
    {x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb ),
    x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) - a/( qb ),
    - x*z/(r*qa)},
    // \partial_y l_k
    {y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + a/( qb ),
    y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) + r/( qb ),
    - y*z/(r*qa)},
    // \partial_z l_k
    {z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa ),
    z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (qb) * qa ),
    - SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r},
  };

  Real dg_ddd[3][3][3] = {};
  for (int i = 0; i < 3; i++)
  for (int a = 0; a < 3; a++)
  for (int b = 0; b < 3; b++) {
    dg_ddd[i][a][b] = 2.*dH_d[i]*l_d[a]*l_d[b] +
                      2.*H*dl_dd[i][a]*l_d[b] +
                      2.*H*l_d[a]*dl_dd[i][b];
  }

  // Exported analytic derivatives.  alpha = (1+2H)^{-1/2} and
  // beta^j = 2H l_j/(1+2H) (l^j == l_j in these coordinates), so
  //   partial_i alpha  = -alpha^3 dH_i
  //   partial_i beta^j = (2 dH_i l_j + 2H dl_ij)/(1+2H) - beta^j 2 dH_i/(1+2H)
  {
    const Real one_p_2H = 1. + 2.*H;
    for (int i = 0; i < 3; ++i) {
      dalp_d[i] = -(*alp)*(*alp)*(*alp)*dH_d[i];
      for (int j = 0; j < 3; ++j) {
        dbeta_du[i][j] =
            (2.*dH_d[i]*l_d[j] + 2.*H*dl_dd[i][j])/one_p_2H -
            (2.*H*l_d[j])*(2.*dH_d[i])/(one_p_2H*one_p_2H);
      }
      dg_sp[i][0] = dg_ddd[i][0][0];  // S11
      dg_sp[i][1] = dg_ddd[i][0][1];  // S12
      dg_sp[i][2] = dg_ddd[i][0][2];  // S13
      dg_sp[i][3] = dg_ddd[i][1][1];  // S22
      dg_sp[i][4] = dg_ddd[i][1][2];  // S23
      dg_sp[i][5] = dg_ddd[i][2][2];  // S33
    }
  }

  //
  // Compute Christoffel symbols
  Real Gamma_udd[3][3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    Gamma_udd[a][b][c] = 0.0;
    for (int d = 0; d < 3; ++d) {
      Gamma_udd[a][b][c] += 0.5*g_uu[a][d]*
                            (dg_ddd[c][b][d] + dg_ddd[b][d][c] - dg_ddd[d][b][c]);
    }
  }

  //
  // Derivatives of the shift vector
  Real const dbeta_dd[3][3] = {
    // \partial_x \beta_i
    {2.*dH_d[0]*l_d[0] + 2.*H*dl_dd[0][0],
    2.*dH_d[0]*l_d[1] + 2.*H*dl_dd[0][1],
    2.*dH_d[0]*l_d[2] + 2.*H*dl_dd[0][2]},
    // \partial_y \beta_i
    {2.*dH_d[1]*l_d[0] + 2.*H*dl_dd[1][0],
    2.*dH_d[1]*l_d[1] + 2.*H*dl_dd[1][1],
    2.*dH_d[1]*l_d[2] + 2.*H*dl_dd[1][2]},
    // \partial_z \beta_i
    {2.*dH_d[2]*l_d[0] + 2.*H*dl_dd[2][0],
    2.*dH_d[2]*l_d[1] + 2.*H*dl_dd[2][1],
    2.*dH_d[2]*l_d[2] + 2.*H*dl_dd[2][2]},
  };
  /*Real dbeta_dd[3][3];
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      dbeta_dd[a][b] = 2.*dH_d[a]*l_d[b] + 2.*H*dl_dd[a][b];
    }
  }*/

  //
  // Covariant derivative of the shift vector
  Real Dbeta_dd[3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    Dbeta_dd[a][b] = dbeta_dd[a][b];
    for (int d = 0; d < 3; ++d) {
      Dbeta_dd[a][b] -= Gamma_udd[d][a][b]*beta_d[d];
    }
  }

  //
  // Extrinsic curvature: K_ab = 1/(2 alp) * (D_a beta_b + D_b beta_a)
  /*Real delta[3][3] = {0.};
  delta[0][0] = 1.;
  delta[1][1] = 1.;
  delta[2][2] = 1.;*/
  Real K_dd[3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    K_dd[a][b] = (Dbeta_dd[a][b] + Dbeta_dd[b][a])/(2.*(*alp));
    //K_dd[a][b] = 2*(*alp)/SQR(r)*(delta[a][b] - (2. + 1./r)*l_d[a]*l_d[b]);
  }

  *Kxx = K_dd[0][0];
  *Kxy = K_dd[0][1];
  *Kxz = K_dd[0][2];
  *Kyy = K_dd[1][1];
  *Kyz = K_dd[1][2];
  *Kzz = K_dd[2][2];
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeADMDecomposition
//! \brief ADM quantities without the derivative outputs (thin wrapper)

KOKKOS_INLINE_FUNCTION
void ComputeADMDecomposition(Real x, Real y, Real z, bool minkowski, Real a,
               Real * alp,
               Real * betax, Real * betay, Real * betaz,
               Real * psi4,
               Real * gxx, Real * gxy, Real * gxz, Real * gyy, Real * gyz, Real * gzz,
               Real * Kxx, Real * Kxy, Real * Kxz, Real * Kyy, Real * Kyz, Real * Kzz) {
  Real dalp_d[3], dbeta_du[3][3], dg_sp[3][6];
  ComputeADMDecompositionAndDerivatives(x, y, z, minkowski, a, alp,
                                        betax, betay, betaz, psi4,
                                        gxx, gxy, gxz, gyy, gyz, gzz,
                                        Kxx, Kxy, Kxz, Kyy, Kyz, Kzz,
                                        dalp_d, dbeta_du, dg_sp);
}


//----------------------------------------------------------------------------------------
//! \fn void ComputeMetricDerivatives
//! \brief computes derivatives of metric in Cartesian Kerr-Schild coordinates, which are
//!  used to compute the coordinate source terms in the equations of motion.

KOKKOS_INLINE_FUNCTION
void ComputeMetricDerivatives(Real x, Real y, Real z, bool minkowski, Real a,
                              Real dg_dx1[][4], Real dg_dx2[][4], Real dg_dx3[][4]) {
  // NOTE(@pdmullen): See comment in ComputeMetricAndInverse
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  Real eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  //r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  Real llower[4];
  llower[0] = 1.0;
  llower[1] = (r*x + a * y)/( SQR(r) + SQR(a) );
  llower[2] = (r*y - a * x)/( SQR(r) + SQR(a) );
  llower[3] = z/r;

  Real qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  Real qb = SQR(r) + SQR(a);
  Real qc = 3.0*SQR(a * z)-SQR(r)*SQR(r);
  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));

  Real df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( qc ) )/ qa;
  Real df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r));
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb );
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + a/( qb );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa );
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) - a/( qb );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) + r/( qb );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (qb) * qa );
  Real dl3_dx1 = - x*z/(r*qa);
  Real dl3_dx2 = - y*z/(r*qa);
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r;

  Real dl0_dx1 = 0.0;
  Real dl0_dx2 = 0.0;
  Real dl0_dx3 = 0.0;

  if (minkowski) {
    f = 0.0;
    df_dx1 = 0.0;
    df_dx2 = 0.0;
    df_dx3 = 0.0;
  }

  // Set x-derivatives of covariant components
  dg_dx1[0][0] = df_dx1*llower[0]*llower[0] + f*dl0_dx1*llower[0] + f*llower[0]*dl0_dx1;
  dg_dx1[0][1] = df_dx1*llower[0]*llower[1] + f*dl0_dx1*llower[1] + f*llower[0]*dl1_dx1;
  dg_dx1[0][2] = df_dx1*llower[0]*llower[2] + f*dl0_dx1*llower[2] + f*llower[0]*dl2_dx1;
  dg_dx1[0][3] = df_dx1*llower[0]*llower[3] + f*dl0_dx1*llower[3] + f*llower[0]*dl3_dx1;
  dg_dx1[1][0] = dg_dx1[0][1];
  dg_dx1[1][1] = df_dx1*llower[1]*llower[1] + f*dl1_dx1*llower[1] + f*llower[1]*dl1_dx1;
  dg_dx1[1][2] = df_dx1*llower[1]*llower[2] + f*dl1_dx1*llower[2] + f*llower[1]*dl2_dx1;
  dg_dx1[1][3] = df_dx1*llower[1]*llower[3] + f*dl1_dx1*llower[3] + f*llower[1]*dl3_dx1;
  dg_dx1[2][0] = dg_dx1[0][2];
  dg_dx1[2][1] = dg_dx1[1][2];
  dg_dx1[2][2] = df_dx1*llower[2]*llower[2] + f*dl2_dx1*llower[2] + f*llower[2]*dl2_dx1;
  dg_dx1[2][3] = df_dx1*llower[2]*llower[3] + f*dl2_dx1*llower[3] + f*llower[2]*dl3_dx1;
  dg_dx1[3][0] = dg_dx1[0][3];
  dg_dx1[3][1] = dg_dx1[1][3];
  dg_dx1[3][2] = dg_dx1[2][3];
  dg_dx1[3][3] = df_dx1*llower[3]*llower[3] + f*dl3_dx1*llower[3] + f*llower[3]*dl3_dx1;

  // Set y-derivatives of covariant components
  dg_dx2[0][0] = df_dx2*llower[0]*llower[0] + f*dl0_dx2*llower[0] + f*llower[0]*dl0_dx2;
  dg_dx2[0][1] = df_dx2*llower[0]*llower[1] + f*dl0_dx2*llower[1] + f*llower[0]*dl1_dx2;
  dg_dx2[0][2] = df_dx2*llower[0]*llower[2] + f*dl0_dx2*llower[2] + f*llower[0]*dl2_dx2;
  dg_dx2[0][3] = df_dx2*llower[0]*llower[3] + f*dl0_dx2*llower[3] + f*llower[0]*dl3_dx2;
  dg_dx2[1][0] = dg_dx2[0][1];
  dg_dx2[1][1] = df_dx2*llower[1]*llower[1] + f*dl1_dx2*llower[1] + f*llower[1]*dl1_dx2;
  dg_dx2[1][2] = df_dx2*llower[1]*llower[2] + f*dl1_dx2*llower[2] + f*llower[1]*dl2_dx2;
  dg_dx2[1][3] = df_dx2*llower[1]*llower[3] + f*dl1_dx2*llower[3] + f*llower[1]*dl3_dx2;
  dg_dx2[2][0] = dg_dx2[0][2];
  dg_dx2[2][1] = dg_dx2[1][2];
  dg_dx2[2][2] = df_dx2*llower[2]*llower[2] + f*dl2_dx2*llower[2] + f*llower[2]*dl2_dx2;
  dg_dx2[2][3] = df_dx2*llower[2]*llower[3] + f*dl2_dx2*llower[3] + f*llower[2]*dl3_dx2;
  dg_dx2[3][0] = dg_dx2[0][3];
  dg_dx2[3][1] = dg_dx2[1][3];
  dg_dx2[3][2] = dg_dx2[2][3];
  dg_dx2[3][3] = df_dx2*llower[3]*llower[3] + f*dl3_dx2*llower[3] + f*llower[3]*dl3_dx2;

  // Set phi-derivatives of covariant components
  dg_dx3[0][0] = df_dx3*llower[0]*llower[0] + f*dl0_dx3*llower[0] + f*llower[0]*dl0_dx3;
  dg_dx3[0][1] = df_dx3*llower[0]*llower[1] + f*dl0_dx3*llower[1] + f*llower[0]*dl1_dx3;
  dg_dx3[0][2] = df_dx3*llower[0]*llower[2] + f*dl0_dx3*llower[2] + f*llower[0]*dl2_dx3;
  dg_dx3[0][3] = df_dx3*llower[0]*llower[3] + f*dl0_dx3*llower[3] + f*llower[0]*dl3_dx3;
  dg_dx3[1][0] = dg_dx3[0][1];
  dg_dx3[1][1] = df_dx3*llower[1]*llower[1] + f*dl1_dx3*llower[1] + f*llower[1]*dl1_dx3;
  dg_dx3[1][2] = df_dx3*llower[1]*llower[2] + f*dl1_dx3*llower[2] + f*llower[1]*dl2_dx3;
  dg_dx3[1][3] = df_dx3*llower[1]*llower[3] + f*dl1_dx3*llower[3] + f*llower[1]*dl3_dx3;
  dg_dx3[2][0] = dg_dx3[0][2];
  dg_dx3[2][1] = dg_dx3[1][2];
  dg_dx3[2][2] = df_dx3*llower[2]*llower[2] + f*dl2_dx3*llower[2] + f*llower[2]*dl2_dx3;
  dg_dx3[2][3] = df_dx3*llower[2]*llower[3] + f*dl2_dx3*llower[3] + f*llower[2]*dl3_dx3;
  dg_dx3[3][0] = dg_dx3[0][3];
  dg_dx3[3][1] = dg_dx3[1][3];
  dg_dx3[3][2] = dg_dx3[2][3];
  dg_dx3[3][3] = df_dx3*llower[3]*llower[3] + f*dl3_dx3*llower[3] + f*llower[3]*dl3_dx3;

  return;
}


#endif // COORDINATES_CARTESIAN_KS_HPP_
