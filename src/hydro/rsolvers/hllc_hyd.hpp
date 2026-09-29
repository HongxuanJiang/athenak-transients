#ifndef HYDRO_RSOLVERS_HLLC_HYD_HPP_
#define HYDRO_RSOLVERS_HLLC_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hllc_hyd.hpp
//! \brief HLLC Riemann solver for hydrodynamics.  Reads L/R primitives from the global
//! per-face buffers and writes a single flux entry.  Supports any non-relativistic hydro
//! EOS that evolves internal energy and supplies pressure and sound speed.

#include <cmath>

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn HLLC<ivx>()
//! \brief Compute the HLLC flux at face (m,k,j,i) for direction ivx and return the
//! interface-normal velocity used by the dual-energy pressure-work update.
template <int ivx>
KOKKOS_INLINE_FUNCTION
Real HLLC(const EOS_Data &eos,
          const int m, const int mb, const int k, const int j, const int i,
          const int is, const int js, const int ks,
          const DvceArray5D<Real> &wl,
          const DvceArray5D<Real> &wr,
          const DvceArray5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx - IVX) + 1) % 3;
  constexpr int ivz = IVX + ((ivx - IVX) + 2) % 3;

  // L/R primitives at face
  const Real wl_idn = wl(mb, IDN, k, j, i);
  const Real wl_ivx = wl(mb, ivx, k, j, i);
  const Real wl_ivy = wl(mb, ivy, k, j, i);
  const Real wl_ivz = wl(mb, ivz, k, j, i);
  const Real wr_idn = wr(mb, IDN, k, j, i);
  const Real wr_ivx = wr(mb, ivx, k, j, i);
  const Real wr_ivy = wr(mb, ivy, k, j, i);
  const Real wr_ivz = wr(mb, ivz, k, j, i);

  Real wl_ipr, wl_cs2, wr_ipr, wr_cs2;
  eos.EvalPressureCs2FromRhoEint(
      wl_idn, wl(mb, IEN, k, j, i), wl_ipr, wl_cs2);
  eos.EvalPressureCs2FromRhoEint(
      wr_idn, wr(mb, IEN, k, j, i), wr_ipr, wr_cs2);
  const Real cl = sqrt(fmax(wl_cs2, 0.0));
  const Real cr = sqrt(fmax(wr_cs2, 0.0));

  const Real sl = fmin(wl_ivx - cl, wr_ivx - cr);
  const Real sr = fmax(wl_ivx + cl, wr_ivx + cr);
  // Keep the HLLC star-state denominators finite for degenerate zero-sound-speed
  // states while preserving the physical sign of each outer wave.
  const Real sl_safe = (sl < 0.0) ? sl : -1.0e-20;
  const Real sr_safe = (sr > 0.0) ? sr : 1.0e-20;

  HydCons1D ul, ur, fl, fr;
  ul.d = wl_idn;
  ul.mx = wl_idn*wl_ivx;
  ul.my = wl_idn*wl_ivy;
  ul.mz = wl_idn*wl_ivz;
  ul.e = wl(mb, IEN, k, j, i) +
         0.5*wl_idn*(SQR(wl_ivx) + SQR(wl_ivy) + SQR(wl_ivz));
  ur.d = wr_idn;
  ur.mx = wr_idn*wr_ivx;
  ur.my = wr_idn*wr_ivy;
  ur.mz = wr_idn*wr_ivz;
  ur.e = wr(mb, IEN, k, j, i) +
         0.5*wr_idn*(SQR(wr_ivx) + SQR(wr_ivy) + SQR(wr_ivz));

  fl.d = ul.mx;
  fl.mx = ul.mx*wl_ivx + wl_ipr;
  fl.my = ul.mx*wl_ivy;
  fl.mz = ul.mx*wl_ivz;
  fl.e = (ul.e + wl_ipr)*wl_ivx;
  fr.d = ur.mx;
  fr.mx = ur.mx*wr_ivx + wr_ipr;
  fr.my = ur.mx*wr_ivy;
  fr.mz = ur.mx*wr_ivz;
  fr.e = (ur.e + wr_ipr)*wr_ivx;

  if (sl >= 0.0) {
    flx(m, IDN, k, j, i) = fl.d;
    flx(m, ivx, k, j, i) = fl.mx;
    flx(m, ivy, k, j, i) = fl.my;
    flx(m, ivz, k, j, i) = fl.mz;
    flx(m, IEN, k, j, i) = fl.e;
    return wl_ivx;
  }
  if (sr <= 0.0) {
    flx(m, IDN, k, j, i) = fr.d;
    flx(m, ivx, k, j, i) = fr.mx;
    flx(m, ivy, k, j, i) = fr.my;
    flx(m, ivz, k, j, i) = fr.mz;
    flx(m, IEN, k, j, i) = fr.e;
    return wr_ivx;
  }

  const Real denom = wl_idn*(sl_safe - wl_ivx) - wr_idn*(sr_safe - wr_ivx);
  const Real sm = ((wr_ipr - wl_ipr) + wl_idn*wl_ivx*(sl_safe - wl_ivx) -
                   wr_idn*wr_ivx*(sr_safe - wr_ivx))/denom;
  const Real pstar_l = wl_ipr + wl_idn*(sl_safe - wl_ivx)*(sm - wl_ivx);
  const Real pstar_r = wr_ipr + wr_idn*(sr_safe - wr_ivx)*(sm - wr_ivx);
  const Real pstar = 0.5*(pstar_l + pstar_r);

  if (sm >= 0.0) {
    const Real rho_star = wl_idn*(sl_safe - wl_ivx)/(sl_safe - sm);
    const Real e_star = ((sl_safe - wl_ivx)*ul.e - wl_ipr*wl_ivx + pstar*sm)/(sl_safe - sm);
    flx(m, IDN, k, j, i) = fl.d + sl_safe*(rho_star - ul.d);
    flx(m, ivx, k, j, i) = fl.mx + sl_safe*(rho_star*sm - ul.mx);
    flx(m, ivy, k, j, i) = fl.my + sl_safe*(rho_star*wl_ivy - ul.my);
    flx(m, ivz, k, j, i) = fl.mz + sl_safe*(rho_star*wl_ivz - ul.mz);
    flx(m, IEN, k, j, i) = fl.e + sl_safe*(e_star - ul.e);
  } else {
    const Real rho_star = wr_idn*(sr_safe - wr_ivx)/(sr_safe - sm);
    const Real e_star = ((sr_safe - wr_ivx)*ur.e - wr_ipr*wr_ivx + pstar*sm)/(sr_safe - sm);
    flx(m, IDN, k, j, i) = fr.d + sr_safe*(rho_star - ur.d);
    flx(m, ivx, k, j, i) = fr.mx + sr_safe*(rho_star*sm - ur.mx);
    flx(m, ivy, k, j, i) = fr.my + sr_safe*(rho_star*wr_ivy - ur.my);
    flx(m, ivz, k, j, i) = fr.mz + sr_safe*(rho_star*wr_ivz - ur.mz);
    flx(m, IEN, k, j, i) = fr.e + sr_safe*(e_star - ur.e);
  }
  return sm;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLC_HYD_HPP_
