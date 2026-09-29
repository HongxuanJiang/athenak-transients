#ifndef MHD_RSOLVERS_HLLD_SRMHD_HPP_
#define MHD_RSOLVERS_HLLD_SRMHD_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_srmhd.hpp
//! \brief Five-wave HLLD solver for special relativistic MHD, single face.
//!
//! Minkowski space is already the locally flat frame in which relativistic HLLD is
//! derived, so this is the shared fan of hlld_local.hpp with the identity frame and a
//! static interface (v = 0).  Where the fan is inadmissible the face falls back to the
//! SR HLLE flux on the same reconstructed states, as in Fields, Wong & Stone (2026,
//! arXiv:2609.06150) Sec. II.

#include <cmath>

#include "mhd/rsolvers/hlld_local.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn HLLD_SR<ivx>()
//! \brief The HLLD Riemann solver for SR MHD, single face (m,k,j,i).

template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLD_SR(const EOS_Data &eos,
             const int m, const int mb, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
             const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
             const DvceArray4D<Real> &bx,
             const BandView5D<Real> &flx,
             const BandView4D<Real> &ey, const BandView4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX) + 1)%3;
  constexpr int ivz = IVX + ((ivx-IVX) + 2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;

  MHDPrim1D wl_local{}, wr_local{};
  wl_local.d = wl(mb,IDN,k,j,i);
  wl_local.e = wl(mb,IEN,k,j,i);
  wl_local.vx = wl(mb,ivx,k,j,i);
  wl_local.vy = wl(mb,ivy,k,j,i);
  wl_local.vz = wl(mb,ivz,k,j,i);
  wl_local.by = bl(mb,iby,k,j,i);
  wl_local.bz = bl(mb,ibz,k,j,i);
  wr_local.d = wr(mb,IDN,k,j,i);
  wr_local.e = wr(mb,IEN,k,j,i);
  wr_local.vx = wr(mb,ivx,k,j,i);
  wr_local.vy = wr(mb,ivy,k,j,i);
  wr_local.vz = wr(mb,ivz,k,j,i);
  wr_local.by = br(mb,iby,k,j,i);
  wr_local.bz = br(mb,ibz,k,j,i);

  MHDCons1D u_face{}, f_face{};
  if (!hlld::SolveLocalHLLD(wl_local, wr_local, bx(m,k,j,i), eos.gamma, eos.gamma,
                            0.0, u_face, f_face)) {
    HLLE_SR<ivx>(eos, m, mb, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
    return;
  }

  flx(m,IDN,k,j,i) = f_face.d;
  flx(m,ivx,k,j,i) = f_face.mx;
  flx(m,ivy,k,j,i) = f_face.my;
  flx(m,ivz,k,j,i) = f_face.mz;
  flx(m,IEN,k,j,i) = f_face.e;

  ey(m,k,j,i) = -f_face.by;
  ez(m,k,j,i) =  f_face.bz;

  // We evolve tau = E - D
  flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
}
}  // namespace mhd
#endif  // MHD_RSOLVERS_HLLD_SRMHD_HPP_
