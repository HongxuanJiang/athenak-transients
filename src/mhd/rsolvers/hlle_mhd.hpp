#ifndef MHD_RSOLVERS_HLLE_MHD_HPP_
#define MHD_RSOLVERS_HLLE_MHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle_mhd.hpp
//! \brief HLLE Riemann solver for MHD.  Per-face split-kernel implementation.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn HLLE<ivx>()
//! \brief The HLLE Riemann solver for MHD (energy-evolving and isothermal), single face.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLE(const EOS_Data &eos,
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
  const Real gm1 = eos.gamma - 1.0;
  const Real iso_cs = eos.iso_cs;


  //--- Step 1.  Load L/R states

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

  Real wl_ipr = 0.0, wr_ipr = 0.0;
  Real wl_cs2 = 0.0, wr_cs2 = 0.0;
  if (eos.use_e) {
    eos.EvalPressureCs2FromRhoEint(
        wl_idn, wl(mb,IEN,k,j,i), wl_ipr, wl_cs2);
    eos.EvalPressureCs2FromRhoEint(
        wr_idn, wr(mb,IEN,k,j,i), wr_ipr, wr_cs2);
  }

  Real bxi = bx(m,k,j,i);

  //--- Step 2. Compute total energies and physical L/R fast speeds
  Real pbl = 0.5*(bxi*bxi + SQR(wl_iby) + SQR(wl_ibz));
  Real pbr = 0.5*(bxi*bxi + SQR(wr_iby) + SQR(wr_ibz));
  Real el = 0.0, er = 0.0, cl, cr;
  if (eos.use_e) {
    el = wl(mb,IEN,k,j,i) +
         0.5*wl_idn*(SQR(wl_ivx)+SQR(wl_ivy)+SQR(wl_ivz)) + pbl;
    er = wr(mb,IEN,k,j,i) +
         0.5*wr_idn*(SQR(wr_ivx)+SQR(wr_ivy)+SQR(wr_ivz)) + pbr;
    cl = eos.FastMagnetosonicSpeedFromSoundSpeed2(
        wl_idn, wl_cs2, bxi, wl_iby, wl_ibz);
    cr = eos.FastMagnetosonicSpeedFromSoundSpeed2(
        wr_idn, wr_cs2, bxi, wr_iby, wr_ibz);
  } else {
    cl = eos.IdealMHDFastSpeed(wl_idn, bxi, wl_iby, wl_ibz);
    cr = eos.IdealMHDFastSpeed(wr_idn, bxi, wr_iby, wr_ibz);
  }

  //--- Step 3. Compute two-sided wave bounds.  Gamma-law and isothermal closures also
  // include their analytic Roe bounds; a general tabulated EOS has no consistent Roe
  // linearization here, so it deliberately uses the physical L/R extrema.
  Real al = fmin(wl_ivx - cl, wr_ivx - cr);
  Real ar = fmax(wl_ivx + cl, wr_ivx + cr);
  if (eos.is_gamma_law || !eos.use_e) {
    const Real sqrtdl = sqrt(wl_idn);
    const Real sqrtdr = sqrt(wr_idn);
    const Real isdlpdr = 1.0/(sqrtdl + sqrtdr);
    const Real wroe_idn = sqrtdl*sqrtdr;
    const Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
    const Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
    const Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;
    const Real wroe_iby = (sqrtdr*wl_iby + sqrtdl*wr_iby)*isdlpdr;
    const Real wroe_ibz = (sqrtdr*wl_ibz + sqrtdl*wr_ibz)*isdlpdr;
    const Real x = 0.5*(SQR(wl_iby-wr_iby) + SQR(wl_ibz-wr_ibz)) /
                   SQR(sqrtdl+sqrtdr);
    const Real y = 0.5*(wl_idn + wr_idn)/wroe_idn;
    const Real btsq = SQR(wroe_iby) + SQR(wroe_ibz);
    const Real vaxsq = bxi*bxi/wroe_idn;
    Real bt_starsq, twid_asq;
    if (eos.is_gamma_law) {
      const Real hroe = ((el + wl_ipr + pbl)/sqrtdl +
                         (er + wr_ipr + pbr)/sqrtdr)*isdlpdr;
      bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
      const Real hp = hroe - (vaxsq + btsq/wroe_idn);
      const Real vsq = SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz);
      twid_asq = fmax(gm1*(hp-0.5*vsq) - (gm1-1.0)*x, 0.0);
    } else {
      bt_starsq = btsq*y;
      twid_asq = iso_cs*iso_cs + x;
    }
    const Real ct2 = bt_starsq/wroe_idn;
    const Real tsum = vaxsq + ct2 + twid_asq;
    const Real tdif = vaxsq + ct2 - twid_asq;
    const Real a = sqrt(0.5*(tsum + sqrt(tdif*tdif + 4.0*twid_asq*ct2)));
    al = fmin(al, wroe_ivx - a);
    ar = fmax(ar, wroe_ivx + a);
  }

  // following min/max set to TINY_NUMBER to fix bug found in converging supersonic flow
  Real bp = ar > 0.0 ? ar : 1.0e-20;
  Real bm = al < 0.0 ? al : -1.0e-20;

  //--- Step 5. Compute L/R fluxes along the lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

  Real vxl = wl_ivx - bm;
  Real vxr = wr_ivx - bp;

  MHDCons1D fl,fr;
  fl.d  = wl_idn*vxl;
  fr.d  = wr_idn*vxr;

  fl.mx = wl_idn*wl_ivx*vxl + pbl - SQR(bxi);
  fr.mx = wr_idn*wr_ivx*vxr + pbr - SQR(bxi);

  fl.my = wl_idn*wl_ivy*vxl - bxi*wl_iby;
  fr.my = wr_idn*wr_ivy*vxr - bxi*wr_iby;

  fl.mz = wl_idn*wl_ivz*vxl - bxi*wl_ibz;
  fr.mz = wr_idn*wr_ivz*vxr - bxi*wr_ibz;

  if (eos.use_e) {
    fl.mx += wl_ipr;
    fr.mx += wr_ipr;
    fl.e   = el*vxl + wl_ivx*(wl_ipr + pbl - bxi*bxi);
    fr.e   = er*vxr + wr_ivx*(wr_ipr + pbr - bxi*bxi);
    fl.e  -= bxi*(wl_iby*wl_ivy + wl_ibz*wl_ivz);
    fr.e  -= bxi*(wr_iby*wr_ivy + wr_ibz*wr_ivz);
  } else {
    fl.mx += (iso_cs*iso_cs)*wl_idn;
    fr.mx += (iso_cs*iso_cs)*wr_idn;
  }

  fl.by = wl_iby*vxl - bxi*wl_ivy;
  fr.by = wr_iby*vxr - bxi*wr_ivy;

  fl.bz = wl_ibz*vxl - bxi*wl_ivz;
  fr.bz = wr_ibz*vxr - bxi*wr_ivz;

  //--- Step 6. Compute the HLLE flux at interface.

  Real tmp=0.0;
  if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);

  flx(m,IDN,k,j,i) = 0.5*(fl.d  + fr.d ) + (fl.d  - fr.d )*tmp;
  flx(m,ivx,k,j,i) = 0.5*(fl.mx + fr.mx) + (fl.mx - fr.mx)*tmp;
  flx(m,ivy,k,j,i) = 0.5*(fl.my + fr.my) + (fl.my - fr.my)*tmp;
  flx(m,ivz,k,j,i) = 0.5*(fl.mz + fr.mz) + (fl.mz - fr.mz)*tmp;
  if (eos.use_e) flx(m,IEN,k,j,i) = 0.5*(fl.e + fr.e ) + (fl.e - fr.e)*tmp;
  ey(m,k,j,i) = -0.5*(fl.by + fr.by) - (fl.by - fr.by)*tmp;
  ez(m,k,j,i) =  0.5*(fl.bz + fr.bz) + (fl.bz - fr.bz)*tmp;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLE_MHD_HPP_
