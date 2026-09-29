#ifndef DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_dyngrmhd.hpp
//! \brief HLLE Riemann solver for general relativistic magnetohydrodynamics

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

#include "eos/drift_frame_floor.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void SingleStateHLLE_DYNGR
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SingleStateHLLE_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    Real prim_l[NPRIM], Real prim_r[NPRIM], Real Bu_l[NPRIM], Real Bu_r[NPRIM],
    const int nmhd, const int nscal,
    Real g3d[NSPMETRIC], Real beta_u[3], Real alpha,
    Real flux[NCONS], Real bflux[NMAG]) {
  constexpr int ibx = ivx - IVX;
  constexpr int iby = ((ivx - IVX) + 1)%3;
  constexpr int ibz = ((ivx - IVX) + 2)%3;

  constexpr int diag[3] = {S11, S22, S33};
  constexpr int offdiag[3] = {S23, S13, S12};
  constexpr int offidx = offdiag[ivx - IVX];
  constexpr int idxy = diag[(ivx - IVX + 1) % 3];
  constexpr int idxz = diag[(ivx - IVX + 2) % 3];

  constexpr int pvx = PVX + (ivx - IVX);

  Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
  Real isdetg = 1.0/sdetg;

  // Undensitize the magnetic field before calculating the conserved variables
  Real Bu_lund[NMAG], Bu_rund[NMAG];
  for (int n = 0; n < NMAG; n++) {
    Bu_lund[n] = Bu_l[n]*isdetg;
    Bu_rund[n] = Bu_r[n]*isdetg;
  }

  // Calculate the left and right fluxes
  const auto &eos_l = eos;
  const auto &eos_r = eos;
  prim_l[PTM] = eos_l.ps.GetEOS().GetTemperatureFromP(
                prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
  prim_r[PTM] = eos_r.ps.GetEOS().GetTemperatureFromP(
                prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
  Real cons_l[NCONS], cons_r[NCONS];
  Real fl[NCONS], fr[NCONS], bfl[NMAG], bfr[NMAG];
  Real bsql, bsqr;
  SingleStateFlux<ivx>(eos_l, eos_r, prim_l, prim_r, Bu_lund, Bu_rund, nmhd,
                       nscal, g3d, beta_u, alpha, cons_l, cons_r, fl, fr,
                       bfl, bfr, bsql, bsqr);


  // Calculate the magnetosonic speeds for both states
  Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
  Real gii = (g3d[idxy]*g3d[idxz] - g3d[offidx]*g3d[offidx])*(isdetg*isdetg);
  eos_l.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                    g3d, beta_u, alpha, gii, pvx);
  eos_r.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                    g3d, beta_u, alpha, gii, pvx);

  // Get the extremal wavespeeds
  Real lambda_l = fmin(lambda_ml, lambda_mr);
  Real lambda_r = fmax(lambda_pl, lambda_pr);

  // Calculate fluxes in HLL region
  Real qa = lambda_r*lambda_l/alpha;
  Real qb = 1.0/(lambda_r - lambda_l);
  Real f_hll[NCONS], bf_hll[NMAG];
  f_hll[CDN] = (lambda_r*fl[CDN] - lambda_l*fr[CDN] +
                qa*(cons_r[CDN] - cons_l[CDN])) * qb;
  f_hll[CSX] = (lambda_r*fl[CSX] - lambda_l*fr[CSX] +
                qa*(cons_r[CSX] - cons_l[CSX])) * qb;
  f_hll[CSY] = (lambda_r*fl[CSY] - lambda_l*fr[CSY] +
                qa*(cons_r[CSY] - cons_l[CSY])) * qb;
  f_hll[CSZ] = (lambda_r*fl[CSZ] - lambda_l*fr[CSZ] +
                qa*(cons_r[CSZ] - cons_l[CSZ])) * qb;
  f_hll[CTA] = (lambda_r*fl[CTA] - lambda_l*fr[CTA] +
                qa*(cons_r[CTA] - cons_l[CTA])) * qb;
  // Bu_l/Bu_r arrive densitized.  The field jump is taken in the undensitized field, as
  // the fluid rows take it in the undensitized cons and as HLLE_DYNGR does; vol below
  // densitizes the whole flux once.
  bf_hll[ibx] = 0.0;
  bf_hll[iby] = (lambda_r*bfl[iby] - lambda_l*bfr[iby] +
                 qa*(Bu_rund[iby] - Bu_lund[iby])) * qb;
  bf_hll[ibz] = (lambda_r*bfl[ibz] - lambda_l*bfr[ibz] +
                 qa*(Bu_rund[ibz] - Bu_lund[ibz])) * qb;

  // See HLLE_DYNGR: select by value, not by pointer, so fl/fr/f_hll stay in registers.
  const bool pick_l = (lambda_l >= 0.);
  const bool pick_r = (!pick_l) && (lambda_r <= 0.);

  Real vol = sdetg*alpha;

  // Calculate the fluxes
  flux[CDN] = vol * (pick_l ? fl[CDN] : (pick_r ? fr[CDN] : f_hll[CDN]));
  flux[CSX] = vol * (pick_l ? fl[CSX] : (pick_r ? fr[CSX] : f_hll[CSX]));
  flux[CSY] = vol * (pick_l ? fl[CSY] : (pick_r ? fr[CSY] : f_hll[CSY]));
  flux[CSZ] = vol * (pick_l ? fl[CSZ] : (pick_r ? fr[CSZ] : f_hll[CSZ]));
  flux[CTA] = vol * (pick_l ? fl[CTA] : (pick_r ? fr[CTA] : f_hll[CTA]));

  bflux[IBY] = - vol * (pick_l ? bfl[iby] : (pick_r ? bfr[iby] : bf_hll[iby]));
  bflux[IBZ] = vol * (pick_l ? bfl[ibz] : (pick_r ? bfr[ibz] : bf_hll[ibz]));
}

//----------------------------------------------------------------------------------------
//! \fn void HLLE_DYNGR
//! \brief inline function for calculating GRMHD fluxes via HLLE
//----------------------------------------------------------------------------------------
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLE_DYNGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int mbuf, const int k, const int j, const int i,
     const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
     const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
     const DvceArray4D<Real> &bx,
     const int nhyd, const int nscal,
     const adm::ADMMetricView& metric,
     const BandView5D<Real> &flx, const BandView4D<Real> &ey,
     const BandView4D<Real> &ez) {
    constexpr int ibx = ivx - IVX;
    constexpr int iby = ((ivx - IVX) + 1)%3;
    constexpr int ibz = ((ivx - IVX) + 2)%3;

    constexpr int diag[3] = {S11, S22, S33};
    constexpr int offdiag[3] = {S23, S13, S12};
    constexpr int offidx = offdiag[ivx - IVX];
    constexpr int idxy = diag[(ivx - IVX + 1) % 3];
    constexpr int idxz = diag[(ivx - IVX + 2) % 3];

    constexpr int pvx = PVX + (ivx - IVX);

    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (ivx == IVX) {
      metric.template FaceMetric<1>(m, k, j, i, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      metric.template FaceMetric<2>(m, k, j, i, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      metric.template FaceMetric<3>(m, k, j, i, g3d, beta_u, alpha);
    }

    Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
    Real isdetg = 1.0/sdetg;

    // Extract left and right primitives
    Real prim_l[NPRIM]{}, prim_r[NPRIM]{};
    Real Bu_l[NMAG], Bu_r[NMAG];
    Real mb = eos.ps.GetEOS().GetBaryonMass();

    prim_l[PRH] = wl(mbuf, IDN, k, j, i)/mb;
    prim_l[PVX] = wl(mbuf, IVX, k, j, i);
    prim_l[PVY] = wl(mbuf, IVY, k, j, i);
    prim_l[PVZ] = wl(mbuf, IVZ, k, j, i);
    for (int n = 0; n < nscal; n++) {
      prim_l[PYF + n] = wl(mbuf, nhyd + n, k, j, i);
    }
    const auto &eos_l = eos;
    eos_l.ps.GetEOS().ApplyDensityLimits(prim_l[PRH]);
    eos_l.ps.GetEOS().ApplySpeciesLimits(&prim_l[PYF]);
    prim_l[PPR] = wl(mbuf, IPR, k, j, i);
    prim_l[PTM] = eos_l.ps.GetEOS().GetTemperatureFromP(
                  prim_l[PRH], prim_l[PPR], &prim_l[PYF]);
    Bu_l[ibx] = bx(m, k, j, i)*isdetg;
    Bu_l[iby] = bl(mbuf, iby, k, j, i)*isdetg;
    Bu_l[ibz] = bl(mbuf, ibz, k, j, i)*isdetg;

    prim_r[PRH] = wr(mbuf, IDN, k, j, i)/mb;
    prim_r[PVX] = wr(mbuf, IVX, k, j, i);
    prim_r[PVY] = wr(mbuf, IVY, k, j, i);
    prim_r[PVZ] = wr(mbuf, IVZ, k, j, i);
    for (int n = 0; n < nscal; n++) {
      prim_r[PYF + n] = wr(mbuf, nhyd + n, k, j, i);
    }
    const auto &eos_r = eos;
    eos_r.ps.GetEOS().ApplyDensityLimits(prim_r[PRH]);
    eos_r.ps.GetEOS().ApplySpeciesLimits(&prim_r[PYF]);
    prim_r[PPR] = wr(mbuf, IPR, k, j, i);
    prim_r[PTM] = eos_r.ps.GetEOS().GetTemperatureFromP(
                  prim_r[PRH], prim_r[PPR], &prim_r[PYF]);
    Bu_r[ibx] = bx(m, k, j, i)*isdetg;
    Bu_r[iby] = br(mbuf, iby, k, j, i)*isdetg;
    Bu_r[ibz] = br(mbuf, ibz, k, j, i)*isdetg;

    // Apply floors to make sure these values are physical.
    eos_floor::DriftFrameApplyPrimitiveFloor(eos_l.ps.GetEOS(), prim_l, Bu_l, g3d);
    eos_floor::DriftFrameApplyPrimitiveFloor(eos_r.ps.GetEOS(), prim_r, Bu_r, g3d);

    // Calculate the left and right fluxes
    Real cons_l[NCONS], cons_r[NCONS];
    Real fl[NCONS], fr[NCONS], bfl[NMAG], bfr[NMAG];
    Real bsql, bsqr;
    SingleStateFlux<ivx>(eos_l, eos_r, prim_l, prim_r, Bu_l, Bu_r, nhyd, nscal,
                         g3d, beta_u, alpha, cons_l, cons_r, fl, fr, bfl, bfr,
                         bsql, bsqr);

    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    Real gii = (g3d[idxy]*g3d[idxz] - g3d[offidx]*g3d[offidx])*(isdetg*isdetg);
    eos_l.GetGRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsql,
                                      g3d, beta_u, alpha, gii, pvx);
    eos_r.GetGRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsqr,
                                      g3d, beta_u, alpha, gii, pvx);

    // Get the extremal wavespeeds
    Real lambda_l = fmin(lambda_ml, lambda_mr);
    Real lambda_r = fmax(lambda_pl, lambda_pr);

    // Calculate fluxes in HLL region
    Real qa = lambda_r*lambda_l/alpha;
    Real qb = 1.0/(lambda_r - lambda_l);
    Real f_hll[NCONS], bf_hll[NMAG];
    f_hll[CDN] = ((lambda_r*fl[CDN] - lambda_l*fr[CDN]) +
                  qa*(cons_r[CDN] - cons_l[CDN])) * qb;
    f_hll[CSX] = ((lambda_r*fl[CSX] - lambda_l*fr[CSX]) +
                  qa*(cons_r[CSX] - cons_l[CSX])) * qb;
    f_hll[CSY] = ((lambda_r*fl[CSY] - lambda_l*fr[CSY]) +
                  qa*(cons_r[CSY] - cons_l[CSY])) * qb;
    f_hll[CSZ] = ((lambda_r*fl[CSZ] - lambda_l*fr[CSZ]) +
                  qa*(cons_r[CSZ] - cons_l[CSZ])) * qb;
    f_hll[CTA] = ((lambda_r*fl[CTA] - lambda_l*fr[CTA]) +
                  qa*(cons_r[CTA] - cons_l[CTA])) * qb;
    bf_hll[ibx] = 0.0;
    bf_hll[iby] = ((lambda_r*bfl[iby] - lambda_l*bfr[iby]) +
                   qa*(Bu_r[iby] - Bu_l[iby])) * qb;
    bf_hll[ibz] = ((lambda_r*bfl[ibz] - lambda_l*bfr[ibz]) +
                   qa*(Bu_r[ibz] - Bu_l[ibz])) * qb;

    // Select the interface state by value rather than by pointer.  Taking &fl[0],
    // &fr[0] and &f_hll[0] makes all three arrays address-taken, so ptxas must give
    // them local (spill) storage; a value select picks exactly the same number and
    // leaves every array promotable to registers.
    const bool pick_l = (lambda_l >= 0.);
    const bool pick_r = (!pick_l) && (lambda_r <= 0.);

    Real vol = sdetg*alpha;

    // Calculate the fluxes
    flx(m, IDN, k, j, i) = vol * (pick_l ? fl[CDN] : (pick_r ? fr[CDN] : f_hll[CDN]));
    flx(m, IEN, k, j, i) = vol * (pick_l ? fl[CTA] : (pick_r ? fr[CTA] : f_hll[CTA]));
    flx(m, IVX, k, j, i) = vol * (pick_l ? fl[CSX] : (pick_r ? fr[CSX] : f_hll[CSX]));
    flx(m, IVY, k, j, i) = vol * (pick_l ? fl[CSY] : (pick_r ? fr[CSY] : f_hll[CSY]));
    flx(m, IVZ, k, j, i) = vol * (pick_l ? fl[CSZ] : (pick_r ? fr[CSZ] : f_hll[CSZ]));
    // The notation here is slightly misleading, as it suggests that Ey = -Fx(By) and
    // Ez = Fx(Bz), rather than Ez = -Fx(By) and Ey = Fx(Bz). However, the appropriate
    // containers for ey and ez for each direction are passed in as arguments to this
    // function, ensuring that the result is entirely consistent.
    ey(m, k, j, i) = -vol *
        (pick_l ? bfl[iby] : (pick_r ? bfr[iby] : bf_hll[iby]));
    ez(m, k, j, i) = vol *
        (pick_l ? bfl[ibz] : (pick_r ? bfr[ibz] : bf_hll[ibz]));
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_HLLE_DYN_GRMHD_HPP_
