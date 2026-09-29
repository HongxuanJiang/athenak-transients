#ifndef HYDRO_RSOLVERS_HLLE_HYD_HPP_
#define HYDRO_RSOLVERS_HLLE_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_hyd.hpp
//! \brief Harten-Lax-vanLeer-Einfeldt (HLLE) Riemann solver for hydrodynamics.  Reads L/R
//! primitives from the global per-face buffers and writes a single flux entry.  Supports
//! gamma-law, tabulated-energy, and isothermal EOS closures.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
//! - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)

#include <algorithm>
#include <cmath>

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn HLLE<ivx>()
//! \brief Compute the HLLE flux at face (m,k,j,i) for direction ivx.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLE(const EOS_Data &eos,
          const int m, const int mb, const int k, const int j, const int i,
          const int is, const int js, const int ks,
          const DvceArray5D<Real> &wl,
          const DvceArray5D<Real> &wr,
          const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;

  const Real gm1 = eos.gamma - 1.0;
  const Real iso_cs = eos.iso_cs;


  // L/R primitives at face
  const Real wl_idn = wl(mb, IDN, k, j, i);
  const Real wl_ivx = wl(mb, ivx, k, j, i);
  const Real wl_ivy = wl(mb, ivy, k, j, i);
  const Real wl_ivz = wl(mb, ivz, k, j, i);

  const Real wr_idn = wr(mb, IDN, k, j, i);
  const Real wr_ivx = wr(mb, ivx, k, j, i);
  const Real wr_ivy = wr(mb, ivy, k, j, i);
  const Real wr_ivz = wr(mb, ivz, k, j, i);

  Real wl_ipr = 0.0, wr_ipr = 0.0;
  Real wl_cs2 = 0.0, wr_cs2 = 0.0;
  if (eos.use_e) {
    eos.EvalPressureCs2FromRhoEint(
        wl_idn, wl(mb, IEN, k, j, i), wl_ipr, wl_cs2);
    eos.EvalPressureCs2FromRhoEint(
        wr_idn, wr(mb, IEN, k, j, i), wr_ipr, wr_cs2);
  }

  Real el = 0.0, er = 0.0;
  if (eos.use_e) {
    el = wl(mb, IEN, k, j, i) +
         0.5*wl_idn*(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
    er = wr(mb, IEN, k, j, i) +
         0.5*wr_idn*(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));
  }

  //--- Sound speeds and two-sided L/R wave bounds
  Real qa, qb;
  if (eos.use_e) {
    qa = sqrt(fmax(wl_cs2, 0.0));
    qb = sqrt(fmax(wr_cs2, 0.0));
  } else {
    qa = iso_cs;
    qb = iso_cs;
  }
  Real al = fmin((wl_ivx - qa), (wr_ivx - qb));
  Real ar = fmax((wl_ivx + qa), (wr_ivx + qb));

  // Gamma-law HLLE also includes the Roe-averaged acoustic bounds.  Tabulated EOS
  // deliberately use only the physical L/R bounds and avoid this extra square-root work.
  if (eos.is_gamma_law) {
    const Real sqrtdl = sqrt(wl_idn);
    const Real sqrtdr = sqrt(wr_idn);
    const Real isdlpdr = 1.0/(sqrtdl + sqrtdr);
    const Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
    const Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
    const Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;
    const Real hroe = ((el + wl_ipr)/sqrtdl + (er + wr_ipr)/sqrtdr)*isdlpdr;
    Real a = gm1*(hroe - 0.5*(SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz)));
    a = (a < 0.0) ? 0.0 : sqrt(a);
    al = fmin((wroe_ivx - a), al);
    ar = fmax((wroe_ivx + a), ar);
  }

  Real bp = (ar > 0.0) ? ar : 1.0e-20;
  Real bm = (al < 0.0) ? al : -1.0e-20;

  //--- L/R fluxes along lines bm/bp
  qa = wl_ivx - bm;
  qb = wr_ivx - bp;

  HydCons1D fl, fr;
  fl.d  = wl_idn*qa;
  fr.d  = wr_idn*qb;
  fl.mx = wl_idn*wl_ivx*qa;
  fr.mx = wr_idn*wr_ivx*qb;
  fl.my = wl_idn*wl_ivy*qa;
  fr.my = wr_idn*wr_ivy*qb;
  fl.mz = wl_idn*wl_ivz*qa;
  fr.mz = wr_idn*wr_ivz*qb;

  if (eos.use_e) {
    fl.mx += wl_ipr;
    fr.mx += wr_ipr;
    fl.e  = el*qa + wl_ipr*wl_ivx;
    fr.e  = er*qb + wr_ipr*wr_ivx;
  } else {
    fl.mx += (iso_cs*iso_cs)*wl_idn;
    fr.mx += (iso_cs*iso_cs)*wr_idn;
  }

  //--- HLLE flux at interface
  qa = 0.0;
  if (bp != bm) { qa = 0.5*(bp + bm)/(bp - bm); }

  flx(m, IDN, k, j, i) = 0.5*(fl.d  + fr.d ) + qa*(fl.d  - fr.d );
  flx(m, ivx, k, j, i) = 0.5*(fl.mx + fr.mx) + qa*(fl.mx - fr.mx);
  flx(m, ivy, k, j, i) = 0.5*(fl.my + fr.my) + qa*(fl.my - fr.my);
  flx(m, ivz, k, j, i) = 0.5*(fl.mz + fr.mz) + qa*(fl.mz - fr.mz);
  if (eos.use_e) {
    flx(m, IEN, k, j, i) = 0.5*(fl.e + fr.e) + qa*(fl.e - fr.e);
  }
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_HYD_HPP_
