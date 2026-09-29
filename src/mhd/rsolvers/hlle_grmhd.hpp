#ifndef MHD_RSOLVERS_HLLE_GRMHD_HPP_
#define MHD_RSOLVERS_HLLE_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grmhd.hpp
//! \brief HLLE Riemann solver for general relativistic MHD.  Per-face implementation for
//! the split-kernel path.
//!
//! Notes:
//!  - cf. HLLE solver in hlle_mhd_rel_no_transform.cpp in Athena++

#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/grmhd_fast_speeds.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn HLLE_GR<ivx>()
//! \brief The HLLE Riemann solver for GR MHD, single face (m,k,j,i).
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLE_GR(const EOS_Data &eos, const RegionIndcs &indcs,
             const DualArray1D<RegionSize> &size, const CoordData &coord,
             const int m, const int mb, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
             const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
             const DvceArray4D<Real> &bx,
             const BandView5D<Real> &flx,
             const BandView4D<Real> &ey, const BandView4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;

  auto &flat = coord.is_minkowski;
  auto &spin = coord.bh_spin;


  // Left/right primitives
  Real wl_idn = wl(mb,IDN,k,j,i);
  Real wl_ivx = wl(mb,ivx,k,j,i);
  Real wl_ivy = wl(mb,ivy,k,j,i);
  Real wl_ivz = wl(mb,ivz,k,j,i);
  Real wl_iby = bl(mb,iby,k,j,i);
  Real wl_ibz = bl(mb,ibz,k,j,i);

  Real wr_idn = wr(mb,IDN,k,j,i);
  Real wr_ivx = wr(mb,ivx,k,j,i);
  Real wr_ivy = wr(mb,ivy,k,j,i);
  Real wr_ivz = wr(mb,ivz,k,j,i);
  Real wr_iby = br(mb,iby,k,j,i);
  Real wr_ibz = br(mb,ibz,k,j,i);

  Real gamma_l = eos.gamma;
  Real gamma_r = eos.gamma;
  const Real gamma_prime_l = gamma_l/(gamma_l - 1.0);
  const Real gamma_prime_r = gamma_r/(gamma_r - 1.0);
  Real wl_ipr = (gamma_l - 1.0) * wl(mb,IEN,k,j,i);
  Real wr_ipr = (gamma_r - 1.0) * wr(mb,IEN,k,j,i);

  // longitudinal field
  Real bxi = bx(m,k,j,i);

  // Extract position of interface
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;
  Real x1v,x2v,x3v;
  if (ivx == IVX) {
    x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
    x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
  } else if (ivx == IVY) {
    x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
  } else {
    x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
  }
  // Compact Kerr-Schild null form: g_{mu nu} = eta_{mu nu} + f l_mu l_nu (l_0 = 1) and
  // g^{mu nu} = eta^{mu nu} - f l^mu l^nu (l^0 = -1, l^i = l_i).  Carrying (f,l1,l2,l3)
  // instead of glower[4][4]/gupper[4][4] keeps 4 live doubles in place of 32; every
  // component used below is recovered exactly by the accessors in cartesian_ks.hpp.
  KSNullForm nf;
  ComputeKSNullForm(x1v, x2v, x3v, flat, spin, nf);

  // Calculate 4-velocity in left state (contravariant compt)
  // q = gamma_ij v^i v^j = |v|^2 + f (l_i v^i)^2
  Real q = KSSpatialNormSq(nf, ivx, ivy, ivz, wl_ivx, wl_ivy, wl_ivz);

  // alpha = sqrt(-1/g^{00}) with g^{00} = -(1+f)
  Real alpha = KSAlpha(nf);
  Real gamma = sqrt(1.0 + q);
  Real uul[4];
  uul[0] = gamma / alpha;
  Real ag = alpha * gamma;
  uul[ivx] = wl_ivx - ag * KSGupper0i(nf, ivx);
  uul[ivy] = wl_ivy - ag * KSGupper0i(nf, ivy);
  uul[ivz] = wl_ivz - ag * KSGupper0i(nf, ivz);

  // lower vector indices (covariant compt): A_mu = eta_{mu nu} A^nu + f l_mu (l.A)
  Real ull[4];
  KSLowerVec(nf, ivx, ivy, ivz, uul[0], uul[ivx], uul[ivy], uul[ivz],
             ull[0], ull[ivx], ull[ivy], ull[ivz]);

  // calculate 4-magnetic field in left state (contravariant compt)
  Real bul[4];
  bul[0]   = ull[ivx]*bxi + ull[ivy]*wl_iby + ull[ivz]*wl_ibz;
  bul[ivx] = (bxi    + bul[0] * uul[ivx]) / uul[0];
  bul[ivy] = (wl_iby + bul[0] * uul[ivy]) / uul[0];
  bul[ivz] = (wl_ibz + bul[0] * uul[ivz]) / uul[0];

  // lower vector indices (covariant compt)
  Real bll[4];
  KSLowerVec(nf, ivx, ivy, ivz, bul[0], bul[ivx], bul[ivy], bul[ivz],
             bll[0], bll[ivx], bll[ivy], bll[ivz]);

  Real bsq_l = bll[0]*bul[0] + bll[ivx]*bul[ivx] + bll[ivy]*bul[ivy] +bll[ivz]*bul[ivz];

  // Calculate 4-velocity in right state (contravariant compt)
  q = KSSpatialNormSq(nf, ivx, ivy, ivz, wr_ivx, wr_ivy, wr_ivz);

  gamma = sqrt(1.0 + q);
  Real uur[4];
  uur[0] = gamma / alpha;
  ag = alpha * gamma;
  uur[ivx] = wr_ivx - ag * KSGupper0i(nf, ivx);
  uur[ivy] = wr_ivy - ag * KSGupper0i(nf, ivy);
  uur[ivz] = wr_ivz - ag * KSGupper0i(nf, ivz);

  // lower vector indices (covariant compt)
  Real ulr[4];
  KSLowerVec(nf, ivx, ivy, ivz, uur[0], uur[ivx], uur[ivy], uur[ivz],
             ulr[0], ulr[ivx], ulr[ivy], ulr[ivz]);

  // Calculate 4-magnetic field in right state
  Real bur[4];
  bur[0]   = ulr[ivx]*bxi + ulr[ivy]*wr_iby + ulr[ivz]*wr_ibz;
  bur[ivx] = (bxi    + bur[0] * uur[ivx]) / uur[0];
  bur[ivy] = (wr_iby + bur[0] * uur[ivy]) / uur[0];
  bur[ivz] = (wr_ibz + bur[0] * uur[ivz]) / uur[0];

  // lower vector indices (covariant compt)
  Real blr[4];
  KSLowerVec(nf, ivx, ivy, ivz, bur[0], bur[ivx], bur[ivy], bur[ivz],
             blr[0], blr[ivx], blr[ivy], blr[ivz]);

  Real bsq_r = blr[0]*bur[0] + blr[ivx]*bur[ivx] + blr[ivy]*bur[ivy] +blr[ivz]*bur[ivz];

  // Contravariant metric components entering the wavespeed solve
  const Real gu00 = KSGupper00(nf);
  const Real gu0x = KSGupper0i(nf, ivx);
  const Real guxx = KSGupperij(nf, ivx, ivx);

  // Calculate wavespeeds in left state.  Free-function form avoids copying all of
  // EOS_Data (12 Views + ~40 scalars) twice per face just to set the local gamma.
  Real lp_l, lm_l;
  GRMHDFastSpeedsGamma(gamma_l, wl_idn, wl_ipr, uul[0], uul[ivx], bsq_l,
                       gu00, gu0x, guxx, lp_l, lm_l);

  // Calculate wavespeeds in right state
  Real lp_r, lm_r;
  GRMHDFastSpeedsGamma(gamma_r, wr_idn, wr_ipr, uur[0], uur[ivx], bsq_r,
                       gu00, gu0x, guxx, lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);

  // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
  MHDCons1D du;
  Real wtot_r = wr_idn + gamma_prime_r * wr_ipr + bsq_r;
  Real ptot_r = wr_ipr + 0.5*bsq_r;
  Real qa = wtot_r * uur[0];
  Real wtot_l = wl_idn + gamma_prime_l * wl_ipr + bsq_l;
  Real ptot_l = wl_ipr + 0.5*bsq_l;
  Real qb = wtot_l * uul[0];
  du.d  = (wr_idn*uur[0]) - (wl_idn*uul[0]);
  du.mx = (qa*ulr[ivx] - bur[0]*blr[ivx]) - (qb*ull[ivx] - bul[0]*bll[ivx]);
  du.my = (qa*ulr[ivy] - bur[0]*blr[ivy]) - (qb*ull[ivy] - bul[0]*bll[ivy]);
  du.mz = (qa*ulr[ivz] - bur[0]*blr[ivz]) - (qb*ull[ivz] - bul[0]*bll[ivz]);
  du.e  = (qa*ulr[0] - bur[0]*blr[0] + ptot_r) - (qb*ull[0] - bul[0]*bll[0] + ptot_l);
  du.by = (bur[ivy]*uur[0] - bur[0]*uur[ivy]) - (bul[ivy]*uul[0] - bul[0]*uul[ivy]);
  du.bz = (bur[ivz]*uur[0] - bur[0]*uur[ivz]) - (bul[ivz]*uul[0] - bul[0]*uul[ivz]);

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fl;
  qa = wtot_l * uul[ivx];
  fl.d  = wl_idn * uul[ivx];
  fl.mx = qa * ull[ivx] - bul[ivx] * bll[ivx] + ptot_l;
  fl.my = qa * ull[ivy] - bul[ivx] * bll[ivy];
  fl.mz = qa * ull[ivz] - bul[ivx] * bll[ivz];
  fl.e  = qa * ull[0]   - bul[ivx] * bll[0];
  fl.by = bul[ivy] * uul[ivx] - bul[ivx] * uul[ivy];
  fl.bz = bul[ivz] * uul[ivx] - bul[ivx] * uul[ivz];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  MHDCons1D fr;
  qa = wtot_r * uur[ivx];
  fr.d  = wr_idn * uur[ivx];
  fr.mx = qa * ulr[ivx] - bur[ivx] * blr[ivx] + ptot_r;
  fr.my = qa * ulr[ivy] - bur[ivx] * blr[ivy];
  fr.mz = qa * ulr[ivz] - bur[ivx] * blr[ivz];
  fr.e  = qa * ulr[0]   - bur[ivx] * blr[0];
  fr.by = bur[ivy] * uur[ivx] - bur[ivx] * uur[ivy];
  fr.bz = bur[ivz] * uur[ivx] - bur[ivx] * uur[ivz];

  // Calculate fluxes in HLL region
  MHDCons1D flux_hll;
  qa = lambda_r*lambda_l;
  qb = 1.0/(lambda_r - lambda_l);
  flux_hll.d  = (lambda_r*fl.d  - lambda_l*fr.d  + qa*du.d ) * qb;
  flux_hll.mx = (lambda_r*fl.mx - lambda_l*fr.mx + qa*du.mx) * qb;
  flux_hll.my = (lambda_r*fl.my - lambda_l*fr.my + qa*du.my) * qb;
  flux_hll.mz = (lambda_r*fl.mz - lambda_l*fr.mz + qa*du.mz) * qb;
  flux_hll.e  = (lambda_r*fl.e  - lambda_l*fr.e  + qa*du.e ) * qb;
  flux_hll.by = (lambda_r*fl.by - lambda_l*fr.by + qa*du.by) * qb;
  flux_hll.bz = (lambda_r*fl.bz - lambda_l*fr.bz + qa*du.bz) * qb;

  // Determine region of wavefan
  MHDCons1D *flux_interface;
  if (lambda_l >= 0.0) {  // L region
    flux_interface = &fl;
  } else if (lambda_r <= 0.0) { // R region
    flux_interface = &fr;
  } else {  // HLL region
    flux_interface = &flux_hll;
  }

  // Set fluxes
  flx(m,IDN,k,j,i) = flux_interface->d;
  flx(m,ivx,k,j,i) = flux_interface->mx;
  flx(m,ivy,k,j,i) = flux_interface->my;
  flx(m,ivz,k,j,i) = flux_interface->mz;
  flx(m,IEN,k,j,i) = flux_interface->e;

  ey(m,k,j,i) = -flux_interface->by;
  ez(m,k,j,i) =  flux_interface->bz;

  // We evolve tau = T^t_t + D
  flx(m,IEN,k,j,i) += flx(m,IDN,k,j,i);
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLE_GRMHD_HPP_
