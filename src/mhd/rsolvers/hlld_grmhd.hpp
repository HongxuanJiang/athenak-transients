#ifndef MHD_RSOLVERS_HLLD_GRMHD_HPP_
#define MHD_RSOLVERS_HLLD_GRMHD_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and Athena++ contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_grmhd.hpp
//! \brief Five-wave HLLD solver for fixed-spacetime GRMHD.
//!
//! Everything specific to curved space is here: the face-aligned orthonormal tetrad, the
//! transformation of the reconstructed primitives into it, the speed at which the
//! interface moves in it, and the transformation of the selected state and flux back to
//! the Cartesian Kerr-Schild coordinate basis.  The fan itself is the flat-space solve of
//! hlld_local.hpp, shared with the special-relativistic entry point.
//!
//! Following Fields, Wong & Stone (2026, arXiv:2609.06150) Sec. II; the
//! frame-transforming strategy is that of Athena++ gr_user.cpp.  Any state the local
//! solve declines falls back to the coordinate-basis GR-HLLE solver on the same
//! reconstructed states and side-local adiabatic indices, as in the paper.

#include <cmath>

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/rsolvers/hlld_local.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"

namespace mhd {
namespace gr_hlld {

using hlld::Finite;
using hlld::FiniteState;

struct FaceFrame {
  Real to_global[4][4];
  Real to_local[4][4];
  int normal;
  int transverse_y;
  int transverse_z;
};

// Construct M^mu_hatnu and its inverse for a face-aligned orthonormal frame.  These
// coefficients are the generic-metric transformation used by Athena++ gr_user.cpp.
template <int normal>
KOKKOS_INLINE_FUNCTION
bool BuildFaceFrame(const Real glower[4][4], const Real gupper[4][4],
                    FaceFrame &frame) {
  frame.normal = normal;
  frame.transverse_y = 1 + normal%3;
  frame.transverse_z = 1 + (normal + 1)%3;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      frame.to_global[a][b] = 0.0;
      frame.to_local[a][b] = 0.0;
    }
  }

  constexpr int i0 = 0;
  constexpr int i1 = normal;
  constexpr int i2 = 1 + normal%3;
  constexpr int i3 = 1 + (normal + 1)%3;
  const Real g_22 = glower[i2][i2];
  const Real g_23 = glower[i2][i3];
  const Real g_33 = glower[i3][i3];
  const Real g00 = gupper[i0][i0];
  const Real g01 = gupper[i0][i1];
  const Real g02 = gupper[i0][i2];
  const Real g03 = gupper[i0][i3];
  const Real g11 = gupper[i1][i1];
  const Real g12 = gupper[i1][i2];
  const Real g13 = gupper[i1][i3];

  const Real rad_b = g00*(g00*g11 - g01*g01);
  const Real rad_d = g_33*(g_22*g_33 - g_23*g_23);
  if (!(g00 < 0.0) || !(g_33 > 0.0) || !(rad_b > 0.0) || !(rad_d > 0.0)) {
    return false;
  }
  const Real aa = -1.0/sqrt(-g00);
  const Real bb = 1.0/sqrt(rad_b);
  const Real cc = 1.0/sqrt(g_33);
  const Real dd = 1.0/sqrt(rad_d);
  const Real ee = g01*g12 - g11*g02;
  const Real ff = g01*g02 - g00*g12;
  const Real gg = g01*g13 - g11*g03;
  const Real hh = g01*g03 - g00*g13;
  const Real ii = bb*bb/cc*g00*(gg + ee*g_23/g_33);
  const Real jj = bb*bb/cc*g00*(hh + ff*g_23/g_33);

  frame.to_global[i0][0] = aa*g00;
  frame.to_global[i1][0] = aa*g01;
  frame.to_global[i2][0] = aa*g02;
  frame.to_global[i3][0] = aa*g03;
  frame.to_global[i1][1] = bb*(g01*g01 - g00*g11);
  frame.to_global[i2][1] = bb*(g01*g02 - g00*g12);
  frame.to_global[i3][1] = bb*(g01*g03 - g00*g13);
  frame.to_global[i2][2] = dd*g_33;
  frame.to_global[i3][2] = -dd*g_23;
  frame.to_global[i3][3] = cc;

  frame.to_local[0][i0] = -aa;
  frame.to_local[1][i0] = bb*g01;
  frame.to_local[1][i1] = -bb*g00;
  frame.to_local[2][i0] = bb*bb*ee/dd*g00/g_33;
  frame.to_local[2][i1] = bb*bb*ff/dd*g00/g_33;
  frame.to_local[2][i2] = 1.0/(dd*g_33);
  frame.to_local[3][i0] = ii;
  frame.to_local[3][i1] = jj;
  frame.to_local[3][i2] = g_23/(cc*g_33);
  frame.to_local[3][i3] = 1.0/cc;

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (!Finite(frame.to_global[a][b]) || !Finite(frame.to_local[a][b])) return false;
    }
  }
  return true;
}

// AthenaK stores the normal-observer projected spatial four-velocity.  Transform it
// directly, while transforming B through u^mu and b^mu so the induction constraint is
// preserved in the local frame.
KOKKOS_INLINE_FUNCTION
bool PrimitiveToLocal(const Real glower[4][4], const Real gupper[4][4],
                      const FaceFrame &frame, const Real uproj[3],
                      const Real bglobal[3], Real ulocal[3], Real blocal[3]) {
  Real q = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) q += glower[a+1][b+1]*uproj[a]*uproj[b];
  }
  if (!Finite(q) || 1.0 + q <= 0.0 || !(gupper[0][0] < 0.0)) return false;
  const Real lor = sqrt(1.0 + q);
  const Real alpha = sqrt(-1.0/gupper[0][0]);

  Real ucon[4];
  ucon[0] = lor/alpha;
  for (int a = 1; a < 4; ++a) {
    ucon[a] = uproj[a-1] - alpha*lor*gupper[0][a];
  }
  Real ucov[4] = {0.0, 0.0, 0.0, 0.0};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) ucov[a] += glower[a][b]*ucon[b];
  }
  Real bcon[4];
  bcon[0] = ucov[1]*bglobal[0] + ucov[2]*bglobal[1] + ucov[3]*bglobal[2];
  if (!(ucon[0] > 0.0)) return false;
  for (int a = 1; a < 4; ++a) {
    bcon[a] = (bglobal[a-1] + bcon[0]*ucon[a])/ucon[0];
  }

  Real bhat[4] = {0.0, 0.0, 0.0, 0.0};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) bhat[a] += frame.to_local[a][b]*bcon[b];
  }
  for (int a = 0; a < 3; ++a) {
    ulocal[a] = 0.0;
    for (int b = 1; b < 4; ++b) ulocal[a] += frame.to_local[a+1][b]*uproj[b-1];
    blocal[a] = lor*bhat[a+1] - ulocal[a]*bhat[0];
    if (!Finite(ulocal[a]) || !Finite(blocal[a])) return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool LocalFluxToGlobal(const FaceFrame &frame, const Real glower[4][4],
                       const Real bx, const MHDCons1D &u, const MHDCons1D &f,
                       MHDCons1D &global_flux) {
  const int n = frame.normal;
  const int iy = frame.transverse_y;
  const int iz = frame.transverse_z;
  const Real jn = frame.to_global[n][0]*u.d + frame.to_global[n][1]*f.d;

  const Real t0[4] = {u.e, u.mx, u.my, u.mz};
  const Real tx[4] = {f.e, f.mx, f.my, f.mz};
  Real t_con[4] = {0.0, 0.0, 0.0, 0.0};
  for (int nu = 0; nu < 4; ++nu) {
    Real row0 = 0.0, rowx = 0.0;
    for (int b = 0; b < 4; ++b) {
      row0 += frame.to_global[nu][b]*t0[b];
      rowx += frame.to_global[nu][b]*tx[b];
    }
    t_con[nu] = frame.to_global[n][0]*row0 + frame.to_global[n][1]*rowx;
  }
  Real t_mixed[4] = {0.0, 0.0, 0.0, 0.0};
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) t_mixed[mu] += glower[mu][nu]*t_con[nu];
  }

  Real dual_local[4][4] = {};
  dual_local[1][0] = bx;       dual_local[0][1] = -bx;
  dual_local[2][0] = u.by;     dual_local[0][2] = -u.by;
  dual_local[3][0] = u.bz;     dual_local[0][3] = -u.bz;
  dual_local[2][1] = f.by;     dual_local[1][2] = -f.by;
  dual_local[3][1] = f.bz;     dual_local[1][3] = -f.bz;
  Real dual_y_n = 0.0, dual_z_n = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      dual_y_n += frame.to_global[iy][a]*frame.to_global[n][b]*dual_local[a][b];
      dual_z_n += frame.to_global[iz][a]*frame.to_global[n][b]*dual_local[a][b];
    }
  }

  global_flux.d = jn;
  global_flux.mx = t_mixed[1];
  global_flux.my = t_mixed[2];
  global_flux.mz = t_mixed[3];
  global_flux.e = t_mixed[0] + jn;
  global_flux.by = -dual_y_n;
  global_flux.bz = dual_z_n;
  return FiniteState(global_flux);
}

}  // namespace gr_hlld

//----------------------------------------------------------------------------------------
//! \fn HLLD_GR<ivx>()
//! \brief Five-wave HLLD solver for fixed-spacetime GRMHD, one face.
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLD_GR(const EOS_Data &eos, const RegionIndcs &indcs,
             const DualArray1D<RegionSize> &size, const CoordData &coord,
             const int m, const int mb, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
             const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
             const DvceArray4D<Real> &bx, const BandView5D<Real> &flx,
             const BandView4D<Real> &ey, const BandView4D<Real> &ez) {
  Real x1v, x2v, x3v;
  if constexpr (ivx == IVX) {
    x1v = LeftEdgeX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
  } else if constexpr (ivx == IVY) {
    x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    x2v = LeftEdgeX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
  } else {
    x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    x3v = LeftEdgeX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
  }

  Real gamma_l = eos.gamma;
  Real gamma_r = eos.gamma;

  auto fallback_hlle = [&]() {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, mb, k, j, i, is, js, ks,
        wl, wr, bl, br, bx, flx, ey, ez);
  };

  // Every way the fan can decline ends in the same HLLE call on the same reconstructed
  // states, so the solve is one block with a single exit rather than five returns: each
  // call site inlines the whole coordinate-basis GR HLLE solver, which is the largest
  // body this file reaches.
  const bool solved = [&]() -> bool {
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);
    gr_hlld::FaceFrame frame{};
    if (!gr_hlld::BuildFaceFrame<ivx>(glower, gupper, frame)) {
      return false;
    }

    const Real uproj_l[3] = {wl(mb,IVX,k,j,i), wl(mb,IVY,k,j,i), wl(mb,IVZ,k,j,i)};
    const Real uproj_r[3] = {wr(mb,IVX,k,j,i), wr(mb,IVY,k,j,i), wr(mb,IVZ,k,j,i)};
    Real bglobal_l[3] = {bl(mb,IBX,k,j,i), bl(mb,IBY,k,j,i), bl(mb,IBZ,k,j,i)};
    Real bglobal_r[3] = {br(mb,IBX,k,j,i), br(mb,IBY,k,j,i), br(mb,IBZ,k,j,i)};
    bglobal_l[ivx-IVX] = bx(m,k,j,i);
    bglobal_r[ivx-IVX] = bx(m,k,j,i);
    Real ulocal_l[3], ulocal_r[3], blocal_l[3], blocal_r[3];
    if (!gr_hlld::PrimitiveToLocal(glower, gupper, frame, uproj_l, bglobal_l,
                                   ulocal_l, blocal_l) ||
        !gr_hlld::PrimitiveToLocal(glower, gupper, frame, uproj_r, bglobal_r,
                                   ulocal_r, blocal_r)) {
      return false;
    }

    MHDPrim1D local_l{}, local_r{};
    local_l.d = wl(mb,IDN,k,j,i);
    local_l.e = wl(mb,IEN,k,j,i);
    local_l.vx = ulocal_l[0]; local_l.vy = ulocal_l[1]; local_l.vz = ulocal_l[2];
    local_l.by = blocal_l[1]; local_l.bz = blocal_l[2];
    local_r.d = wr(mb,IDN,k,j,i);
    local_r.e = wr(mb,IEN,k,j,i);
    local_r.vx = ulocal_r[0]; local_r.vy = ulocal_r[1]; local_r.vz = ulocal_r[2];
    local_r.by = blocal_r[1]; local_r.bz = blocal_r[2];
    const Real local_bx = 0.5*(blocal_l[0] + blocal_r[0]);
    // In the tetrad frame the interface is not static: it moves with
    // v = beta^x/(alpha sqrt(gamma^xx)).  With beta^i = alpha^2 g^{0i},
    // alpha^2 = -1/g^{00} and gamma^{ij} = g^{ij} - beta^i beta^j/alpha^2 that is
    // identically g^{0x}/sqrt((g^{0x})^2 - g^{00} g^{xx}); the y and z faces follow by
    // permuting the index.  The radicand is positive wherever the face is spacelike.
    const Real face_denom = SQR(gupper[0][ivx]) - gupper[0][0]*gupper[ivx][ivx];
    if (!(face_denom > 0.0)) {
      return false;
    }
    const Real v_interface = gupper[0][ivx]/sqrt(face_denom);
    MHDCons1D local_u{}, local_f{};
    if (!hlld::SolveLocalHLLD(local_l, local_r, local_bx, gamma_l, gamma_r,
                              v_interface, local_u, local_f)) {
      return false;
    }

    MHDCons1D flux{};
    if (!gr_hlld::LocalFluxToGlobal(frame, glower, local_bx, local_u, local_f, flux)) {
      return false;
    }
    flx(m,IDN,k,j,i) = flux.d;
    flx(m,IVX,k,j,i) = flux.mx;
    flx(m,IVY,k,j,i) = flux.my;
    flx(m,IVZ,k,j,i) = flux.mz;
    flx(m,IEN,k,j,i) = flux.e;
    ey(m,k,j,i) = flux.by;
    ez(m,k,j,i) = flux.bz;

    return true;
  }();
  if (!solved) {
    fallback_hlle();
  }
}

}  // namespace mhd
#endif  // MHD_RSOLVERS_HLLD_GRMHD_HPP_
