#ifndef COORDINATES_ANALYTIC_BBH_HPP_
#define COORDINATES_ANALYTIC_BBH_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

// Device-callable analytical superposed Kerr-Schild BBH metric used by the BBH
// problem generator and by the on-the-fly ADM metric backend.

// The unqualified isfinite() guards below resolve against the global namespace, which
// nvcc/nvc++ populate implicitly but a host-only GCC build does not.  Without this the
// on-the-fly backend -- the only LAT-safe BBH metric backend -- fails to compile on CPU,
// which is where the LAT regression inputs are meant to run.  BBH.cpp relies on the same
// header for the identical guards in its own copy of these kernels.
#include <math.h>

#include "athena.hpp"
#include "coordinates/dual_number.hpp"

namespace adm {
namespace analytic_bbh {

constexpr int kMetricDim = 4;
constexpr int kTrajectorySize = 20;
constexpr Real derivative_step = 5.0e-5;

enum TrajectoryIndex {
  X1, Y1, Z1, X2, Y2, Z2,
  VX1, VY1, VZ1, VX2, VY2, VZ2,
  AX1, AY1, AZ1, AX2, AY2, AZ2,
  M1T, M2T
};

struct KeplerParameters {
  Real sep = 20.0;
  Real om = 0.0;
  Real q = 1.0;
  Real a1 = 0.0;
  Real a2 = 0.0;
  Real th_a1 = 0.0;
  Real th_a2 = 0.0;
  Real ph_a1 = 0.0;
  Real ph_a2 = 0.0;
  Real a1_buffer = 0.0;
  Real a2_buffer = 0.0;
  Real adjust_mass1 = 1.0;
  Real adjust_mass2 = 1.0;
  Real cutoff_floor = 0.0;
};

struct BoostState {
  Real J00, J01, J02, J03;
  Real J11, J12, J13;
  Real J22, J23;
  Real J33;
  Real cf;
  Real xi[3];
  Real v[3];
  Real a[3];
  Real a_param;
  Real m;
};

// Same fields as BoostState, but each carrying its time derivative in partial slot 3.
// The metric depends on t *only* through these numbers, so seeding them is what turns a
// purely spatial dual evaluation into a full four-derivative one.  Layout deliberately
// mirrors BoostState field for field so MakeDualBoostState reads as a transcription.
struct DualBoostState {
  dual::Dual4 J00, J01, J02, J03;
  dual::Dual4 J11, J12, J13;
  dual::Dual4 J22, J23;
  dual::Dual4 J33;
  dual::Dual4 cf;
  dual::Dual4 xi[3];
  dual::Dual4 v[3];
  dual::Dual4 a[3];
  dual::Dual4 a_param;
  dual::Dual4 m;
};

struct MetricState {
  BoostState bh1_minus, bh2_minus;
  BoostState bh1_now, bh2_now;
  BoostState bh1_plus, bh2_plus;
  Real time = 0.0;
};

// Value component of a scalar that may be a Real or a dual.  Guard conditions inside the
// templated kernels branch on this, so every instantiation follows the identical control
// flow and a black hole skipped by the value path has its derivative contribution skipped
// too.
KOKKOS_INLINE_FUNCTION
Real ScalarValue(const Real a) { return a; }

template <int N>
KOKKOS_INLINE_FUNCTION
Real ScalarValue(const dual::DualN<N> &a) { return a.v; }

struct Symmetric4Metric {
  Real tt, tx, ty, tz;
  Real xx, xy, xz, yy, yz, zz;
};

struct FourMetric {
  Symmetric4Metric g;
  Symmetric4Metric g_t;
  Symmetric4Metric g_x;
  Symmetric4Metric g_y;
  Symmetric4Metric g_z;
};

struct ThreeMetric {
  Real gxx, gxy, gxz, gyy, gyz, gzz;
  Real alpha;
  Real betax, betay, betaz;
  Real kxx, kxy, kxz, kyy, kyz, kzz;
};

// Derivatives of the 3+1 quantities, reconstructed analytically from the four-metric
// derivatives.  The spatial component order matches SpatialMetricIndex
// (S11,S12,S13,S22,S23,S33 == xx,xy,xz,yy,yz,zz), so a consumer can copy straight into
// an adm::ADMMetricDerivatives without remapping.
struct ThreeMetricDerivatives {
  Real dgam_ddd[3][6];   // [direction][xx,xy,xz,yy,yz,zz]
  Real dalpha_d[3];      // [direction]
  Real dbeta_du[3][3];   // [direction][upper index of beta]
  Real dalpha_dt;        // exact coordinate-time derivative for the prescribed metric
};

KOKKOS_INLINE_FUNCTION
Real SpatialDet(const Real gxx, const Real gxy, const Real gxz,
                const Real gyy, const Real gyz, const Real gzz) {
  return -SQR(gxz)*gyy + 2.0*gxy*gxz*gyz - SQR(gyz)*gxx
         - SQR(gxy)*gzz + gxx*gyy*gzz;
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void KeplerTrajectory(const Real t, const Params &params,
                      Real traj[kTrajectorySize]) {
  const Real r1 = params.q/(1.0 + params.q)*params.sep;
  const Real r2 = -params.sep/(1.0 + params.q);
  const Real phase = params.om*t;
  const Real cphase = cos(phase);
  const Real sphase = sin(phase);

  traj[X1] = r1*cphase; traj[Y1] = r1*sphase; traj[Z1] = 0.0;
  traj[X2] = r2*cphase; traj[Y2] = r2*sphase; traj[Z2] = 0.0;
  traj[VX1] = -r1*params.om*sphase;
  traj[VY1] =  r1*params.om*cphase;
  traj[VZ1] = 0.0;
  traj[VX2] = -r2*params.om*sphase;
  traj[VY2] =  r2*params.om*cphase;
  traj[VZ2] = 0.0;

  traj[AX1] = params.a1*sin(params.th_a1)*cos(params.ph_a1);
  traj[AY1] = params.a1*sin(params.th_a1)*sin(params.ph_a1);
  traj[AZ1] = params.a1*cos(params.th_a1);
  traj[AX2] = params.a2*sin(params.th_a2)*cos(params.ph_a2);
  traj[AY2] = params.a2*sin(params.th_a2)*sin(params.ph_a2);
  traj[AZ2] = params.a2*cos(params.th_a2);
  traj[M1T] = 1.0/(params.q + 1.0);
  traj[M2T] = 1.0 - traj[M1T];
}

KOKKOS_INLINE_FUNCTION
void InitializeBoostState(BoostState &bh, const Real x, const Real y, const Real z,
                          const Real vx, const Real vy, const Real vz,
                          const Real sx, const Real sy, const Real sz,
                          const Real mass_scale, const Real mass_adjust) {
  bh.xi[0] = isfinite(x) ? x : 0.0;
  bh.xi[1] = isfinite(y) ? y : 0.0;
  bh.xi[2] = isfinite(z) ? z : 0.0;
  bh.v[0] = isfinite(vx) ? vx : 0.0;
  bh.v[1] = isfinite(vy) ? vy : 0.0;
  bh.v[2] = isfinite(vz) ? vz : 0.0;
  bh.m = (isfinite(mass_scale) && isfinite(mass_adjust))
             ? mass_scale*mass_adjust : 0.0;
  if (!(bh.m > 0.0)) bh.m = 0.0;

  Real v_sq = SQR(bh.v[0]) + SQR(bh.v[1]) + SQR(bh.v[2]);
  if (!isfinite(v_sq) || v_sq < 0.0) {
    bh.v[0] = 0.0; bh.v[1] = 0.0; bh.v[2] = 0.0;
    v_sq = 0.0;
  } else if (v_sq >= 1.0) {
    const Real scale = sqrt((1.0 - 1.0e-12)/v_sq);
    bh.v[0] *= scale; bh.v[1] *= scale; bh.v[2] *= scale;
    v_sq = 1.0 - 1.0e-12;
  }

  const Real spin_mag = sqrt(SQR(sx) + SQR(sy) + SQR(sz));
  bh.a_param = (spin_mag > 0.0) ? spin_mag*bh.m : 0.0;
  const Real spin_scale = (spin_mag > 1.0e-40) ? bh.a_param/spin_mag : 0.0;
  bh.a[0] = sx*spin_scale;
  bh.a[1] = sy*spin_scale;
  bh.a[2] = sz*spin_scale;

  const Real gamma_lor = (v_sq > 0.0) ? 1.0/sqrt(1.0 - v_sq) : 1.0;
  bh.cf = (v_sq > 1.0e-30) ? (gamma_lor - 1.0)/v_sq : 0.0;
  bh.J00 = gamma_lor;
  bh.J01 = -gamma_lor*bh.v[0];
  bh.J02 = -gamma_lor*bh.v[1];
  bh.J03 = -gamma_lor*bh.v[2];
  bh.J11 = 1.0 + bh.cf*bh.v[0]*bh.v[0];
  bh.J12 = bh.cf*bh.v[0]*bh.v[1];
  bh.J13 = bh.cf*bh.v[0]*bh.v[2];
  bh.J22 = 1.0 + bh.cf*bh.v[1]*bh.v[1];
  bh.J23 = bh.cf*bh.v[1]*bh.v[2];
  bh.J33 = 1.0 + bh.cf*bh.v[2]*bh.v[2];
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void PrecomputeBoostState(const Real traj[kTrajectorySize], const Params &params,
                          BoostState &bh1, BoostState &bh2) {
  InitializeBoostState(bh1, traj[X1], traj[Y1], traj[Z1],
                       traj[VX1], traj[VY1], traj[VZ1],
                       traj[AX1], traj[AY1], traj[AZ1], traj[M1T],
                       params.adjust_mass1);
  InitializeBoostState(bh2, traj[X2], traj[Y2], traj[Z2],
                       traj[VX2], traj[VY2], traj[VZ2],
                       traj[AX2], traj[AY2], traj[AZ2], traj[M2T],
                       params.adjust_mass2);
}

// Compress three trajectory samples (t-derivative_step, t, t+derivative_step) into the
// compact device-resident metric state.  This is the ONLY place the orbit enters the
// spacetime: everything downstream of a MetricState is orbit-source agnostic, which is
// what lets a tabulated post-Newtonian orbit use the same on-the-fly backend as the
// closed-form Keplerian one without ever putting the table on the device.
template<typename Params>
KOKKOS_INLINE_FUNCTION
void BuildMetricStateFromTrajectories(const Real time,
                                      const Real traj_minus[kTrajectorySize],
                                      const Real traj_now[kTrajectorySize],
                                      const Real traj_plus[kTrajectorySize],
                                      const Params &params, MetricState &state) {
  PrecomputeBoostState(traj_minus, params, state.bh1_minus, state.bh2_minus);
  PrecomputeBoostState(traj_now, params, state.bh1_now, state.bh2_now);
  PrecomputeBoostState(traj_plus, params, state.bh1_plus, state.bh2_plus);
  state.time = time;
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void BuildKeplerMetricState(const Real time, const Params &params,
                            MetricState &state) {
  Real traj_minus[kTrajectorySize], traj_now[kTrajectorySize];
  Real traj_plus[kTrajectorySize];
  KeplerTrajectory(time-derivative_step, params, traj_minus);
  KeplerTrajectory(time, params, traj_now);
  KeplerTrajectory(time+derivative_step, params, traj_plus);
  BuildMetricStateFromTrajectories(time, traj_minus, traj_now, traj_plus, params, state);
}

//----------------------------------------------------------------------------------------
//! \fn T ShrunkExcisionRadius
//! \brief The Kerr-Schild radius m + sqrt(m^2 - a^2)/2 at which BBH.cpp excises a hole it
//! shrinks inside its horizon.  It is the smaller of the two radii BBH.cpp excises at,
//! and on the spin axis a Kerr-Schild radius is the Cartesian distance from the hole.

template<typename T>
KOKKOS_INLINE_FUNCTION
T ShrunkExcisionRadius(const T &m, const T &a) {
  const T disc = m*m - a*a;
  return (ScalarValue(disc) > 0.0) ? m + 0.5*sqrt(disc) : m;
}

//----------------------------------------------------------------------------------------
//! \fn void AddBoostedBHContributionGeneric
//! \brief Superposed boosted Kerr-Schild contribution of one black hole, templated on the
//! scalar type.
//!
//! Instantiated with T = Real it is the original value-only kernel, unchanged.
//! Instantiated with T = dual::Dual4 (and BState = DualBoostState) the very same
//! expressions carry the exact partial derivatives along, so one pass replaces the nine
//! finite-difference passes the derivatives used to cost.  Because Dual4 reproduces every
//! value operation bit for bit, the two instantiations agree exactly on the metric.
//!
//! The guards branch on ScalarValue(), i.e. on the value component only, so both
//! instantiations take the same branches: a black hole skipped here contributes neither
//! metric nor derivative, which is what the finite-difference path did as well.
//!
//! The ring singularity lies at rest-frame radius |a|.  At rest-frame radius r >= r_soft
//! the contribution is Kerr, bit for bit.  Inside it the point is moved radially to
//!   r_eff = r_soft - w (t - t^3 + t^4/2),  t = (r_soft - r)/w,  w = 2 (r_soft - r_cut),
//! and to r_eff = r_cut once t >= 1.  r_eff is C2 at r_soft, nondecreasing in r, and
//! never below r_cut = |a|(1 + buffer) + cutoff_floor > |a|.  r_soft is the shrunk
//! excision radius, capped at 2 r_cut so that the core r_eff = r_cut starts at r >= 0,
//! and raised to r_cut if the cutoff reaches it (BBH.cpp refuses a deck whose buffer
//! and cutoff_floor do that to a hole with mass at t = 0).  The excision surface lies
//! at Cartesian distance >= that radius from the hole, so the metric is Kerr on and
//! outside it and the Kerr horizon is the evolved metric's.

template<typename T, typename BState>
KOKKOS_INLINE_FUNCTION
void AddBoostedBHContributionGeneric(const T &x, const T &y, const T &z,
                                     const BState &bh, const Real buffer,
                                     const Real cutoff_floor,
                                     T gcov[kMetricDim][kMetricDim]) {
  if (!(ScalarValue(bh.m) > 0.0)) return;

  constexpr Real sqrt_two = 1.4142135623730951;
  constexpr Real inv_sqrt_two = 1.0/sqrt_two;
  T dx = x - bh.xi[0];
  T dy = y - bh.xi[1];
  T dz = z - bh.xi[2];
  const T v_dot_r = bh.v[0]*dx + bh.v[1]*dy + bh.v[2]*dz;
  T xR0 = dx + bh.cf*bh.v[0]*v_dot_r;
  T xR1 = dy + bh.cf*bh.v[1]*v_dot_r;
  T xR2 = dz + bh.cf*bh.v[2]*v_dot_r;

  const T rBH = sqrt(xR0*xR0 + xR1*xR1 + xR2*xR2);
  const T r_cut = fabs(bh.a_param)*(1.0 + buffer) + cutoff_floor;
  T r_soft = ShrunkExcisionRadius(bh.m, bh.a_param);
  if (ScalarValue(r_soft) > 2.0*ScalarValue(r_cut)) r_soft = 2.0*r_cut;
  if (!(ScalarValue(r_soft) > ScalarValue(r_cut))) r_soft = r_cut;
  if (ScalarValue(rBH) < ScalarValue(r_soft)) {
    const T width = 2.0*(r_soft - r_cut);
    T r_eff = r_cut;
    if (ScalarValue(rBH) > ScalarValue(r_soft - width)) {
      const T t = (r_soft - rBH)/width;
      r_eff = r_soft - width*t*(1.0 - t*t*(1.0 - 0.5*t));
    }
    if (ScalarValue(rBH) > 1.0e-12) {
      const T scale = r_eff/rBH;
      xR0 *= scale; xR1 *= scale; xR2 *= scale;
    } else {
      xR0 = r_eff; xR1 = 0.0; xR2 = 0.0;
    }
  }

  const T ax = bh.a[0], ay = bh.a[1], az = bh.a[2];
  const T o3 = ax*ax, o5 = ay*ay, o7 = az*az;
  const T o9 = xR0*xR0, o10 = xR1*xR1, o11 = xR2*xR2;
  const T o15 = xR0*ax + xR1*ay + xR2*az;
  const T o16 = o15*o15;
  const T o18 = o10 + o11 - o3 - o5 - o7 + o9;
  const T o20 = 4.0*o16 + o18*o18;
  const T o21 = sqrt(o20);
  const T o22 = o10 + o11 + o21 - o3 - o5 - o7 + o9;
  if (!isfinite(ScalarValue(o22)) || ScalarValue(o22) <= 0.0) return;
  const T o23 = o22*sqrt(o22);
  const T o26 = o16 + 0.25*o22*o22;
  if (!isfinite(ScalarValue(o26)) || ScalarValue(o26) <= 0.0) return;
  const T o27 = 1.0/o26;
  const T o30 = sqrt(o22);
  const T o31 = 1.0/o30;
  const T o34 = xR1*az - ay*xR2 + sqrt_two*o15*o31*ax
                   + o30*xR0*inv_sqrt_two;
  const T o43 = -az*xR0 + xR2*ax + sqrt_two*o15*o31*ay
                   + o30*xR1*inv_sqrt_two;
  const T o49 = xR0*ay - ax*xR1 + sqrt_two*o15*o31*az
                   + o30*xR2*inv_sqrt_two;
  const T o37 = 1.0/(o3 + 0.5*o22 + o5 + o7);
  const T H = inv_sqrt_two*o23*o27*bh.m;
  const T l1 = o34*o37;
  const T l2 = o43*o37;
  const T l3 = o49*o37;
  const T l0_lab = bh.J00 + bh.J01*l1 + bh.J02*l2 + bh.J03*l3;
  const T l1_lab = bh.J01 + bh.J11*l1 + bh.J12*l2 + bh.J13*l3;
  const T l2_lab = bh.J02 + bh.J12*l1 + bh.J22*l2 + bh.J23*l3;
  const T l3_lab = bh.J03 + bh.J13*l1 + bh.J23*l2 + bh.J33*l3;

  gcov[0][0] += H*l0_lab*l0_lab;
  gcov[0][1] += H*l0_lab*l1_lab;
  gcov[0][2] += H*l0_lab*l2_lab;
  gcov[0][3] += H*l0_lab*l3_lab;
  gcov[1][1] += H*l1_lab*l1_lab;
  gcov[1][2] += H*l1_lab*l2_lab;
  gcov[1][3] += H*l1_lab*l3_lab;
  gcov[2][2] += H*l2_lab*l2_lab;
  gcov[2][3] += H*l2_lab*l3_lab;
  gcov[3][3] += H*l3_lab*l3_lab;
}

KOKKOS_INLINE_FUNCTION
void AddBoostedBHContribution(const Real x, const Real y, const Real z,
                              const BoostState &bh, const Real buffer,
                              const Real cutoff_floor,
                              Real gcov[kMetricDim][kMetricDim]) {
  AddBoostedBHContributionGeneric(x, y, z, bh, buffer, cutoff_floor, gcov);
}

template<typename T, typename BState, typename Params>
KOKKOS_INLINE_FUNCTION
void ComputeMetricWithBoostGeneric(const T &x, const T &y, const T &z,
                                   const BState &bh1, const BState &bh2,
                                   T gcov[kMetricDim][kMetricDim],
                                   const Params &params) {
  gcov[0][0] = -1.0; gcov[0][1] = 0.0; gcov[0][2] = 0.0; gcov[0][3] = 0.0;
  gcov[1][0] = 0.0; gcov[1][1] = 1.0; gcov[1][2] = 0.0; gcov[1][3] = 0.0;
  gcov[2][0] = 0.0; gcov[2][1] = 0.0; gcov[2][2] = 1.0; gcov[2][3] = 0.0;
  gcov[3][0] = 0.0; gcov[3][1] = 0.0; gcov[3][2] = 0.0; gcov[3][3] = 1.0;
  AddBoostedBHContributionGeneric(x, y, z, bh1, params.a1_buffer,
                                  params.cutoff_floor, gcov);
  AddBoostedBHContributionGeneric(x, y, z, bh2, params.a2_buffer,
                                  params.cutoff_floor, gcov);
  gcov[1][0] = gcov[0][1];
  gcov[2][0] = gcov[0][2]; gcov[2][1] = gcov[1][2];
  gcov[3][0] = gcov[0][3]; gcov[3][1] = gcov[1][3];
  gcov[3][2] = gcov[2][3];
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void ComputeMetricWithBoost(const Real x, const Real y, const Real z,
                            const BoostState &bh1, const BoostState &bh2,
                            Real gcov[kMetricDim][kMetricDim],
                            const Params &params) {
  ComputeMetricWithBoostGeneric(x, y, z, bh1, bh2, gcov, params);
}

KOKKOS_INLINE_FUNCTION
void CopyMetric(const Real gcov[kMetricDim][kMetricDim],
                Symmetric4Metric &metric) {
  metric.tt = gcov[0][0]; metric.tx = gcov[0][1];
  metric.ty = gcov[0][2]; metric.tz = gcov[0][3];
  metric.xx = gcov[1][1]; metric.xy = gcov[1][2];
  metric.xz = gcov[1][3]; metric.yy = gcov[2][2];
  metric.yz = gcov[2][3]; metric.zz = gcov[3][3];
}

KOKKOS_INLINE_FUNCTION
void SanitizeDerivative(Symmetric4Metric &g) {
  Real *values[10] = {&g.tt, &g.tx, &g.ty, &g.tz, &g.xx,
                      &g.xy, &g.xz, &g.yy, &g.yz, &g.zz};
  for (int n = 0; n < 10; ++n) {
    if (!isfinite(*values[n])) *values[n] = 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CopyDualMetric
//! \brief Split one dual four-metric into its value and its four derivative blocks.
//!
//! Only the ten independent components are read, exactly as CopyMetric does, so the six
//! redundant lower-triangle duals written by ComputeMetricWithBoostGeneric are dead and
//! the compiler drops them -- which matters here, because a dual is five doubles wide.

KOKKOS_INLINE_FUNCTION
void CopyDualMetric(const dual::Dual4 gcov[kMetricDim][kMetricDim], FourMetric &four) {
  const dual::Dual4 *comp[10] = {&gcov[0][0], &gcov[0][1], &gcov[0][2], &gcov[0][3],
                                 &gcov[1][1], &gcov[1][2], &gcov[1][3], &gcov[2][2],
                                 &gcov[2][3], &gcov[3][3]};
  Real *val[10] = {&four.g.tt, &four.g.tx, &four.g.ty, &four.g.tz, &four.g.xx,
                   &four.g.xy, &four.g.xz, &four.g.yy, &four.g.yz, &four.g.zz};
  Real *dx[10] = {&four.g_x.tt, &four.g_x.tx, &four.g_x.ty, &four.g_x.tz, &four.g_x.xx,
                  &four.g_x.xy, &four.g_x.xz, &four.g_x.yy, &four.g_x.yz, &four.g_x.zz};
  Real *dy[10] = {&four.g_y.tt, &four.g_y.tx, &four.g_y.ty, &four.g_y.tz, &four.g_y.xx,
                  &four.g_y.xy, &four.g_y.xz, &four.g_y.yy, &four.g_y.yz, &four.g_y.zz};
  Real *dz[10] = {&four.g_z.tt, &four.g_z.tx, &four.g_z.ty, &four.g_z.tz, &four.g_z.xx,
                  &four.g_z.xy, &four.g_z.xz, &four.g_z.yy, &four.g_z.yz, &four.g_z.zz};
  Real *dt[10] = {&four.g_t.tt, &four.g_t.tx, &four.g_t.ty, &four.g_t.tz, &four.g_t.xx,
                  &four.g_t.xy, &four.g_t.xz, &four.g_t.yy, &four.g_t.yz, &four.g_t.zz};
  for (int n = 0; n < 10; ++n) {
    *val[n] = comp[n]->v;
    *dx[n] = comp[n]->d[0];
    *dy[n] = comp[n]->d[1];
    *dz[n] = comp[n]->d[2];
    *dt[n] = comp[n]->d[3];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MakeDualBoostState
//! \brief Attach a time derivative to every field of a BoostState.
//!
//! The values are copied verbatim from the `now` state, so the metric value produced by
//! the dual path is bitwise the value the old code produced.  Only the slot-3 partials
//! are new, and they come from three sources, in decreasing order of preference:
//!
//!  * d(xi)/dt = v, exactly.  The boosted Kerr-Schild solution is only a solution if the
//!    hole moves with the velocity v that builds its own boost matrix, so v *is* the
//!    centre's velocity by construction -- there is nothing to difference.  This matters:
//!    the metric's spatial gradient is enormous near a hole, so the position is the
//!    dominant channel of the time dependence, and differencing xi (whose magnitude is
//!    the separation, tens of M) would have been noisier than differencing the metric.
//!
//!  * cf and the boost matrix J are closed-form functions of v alone, so their time
//!    derivatives follow from dv/dt by the chain rule rather than from a second
//!    difference of numbers of order unity.
//!
//!  * dv/dt, dm/dt, d(a)/dt and d(a_param)/dt are central differences of the three
//!    sampled states, the only information a tabulated orbit supplies.  For the common
//!    case of constant masses and constant spins the plus and minus samples are bitwise
//!    equal and these are exactly zero; the surviving one, dv/dt, is differenced from
//!    numbers of order |v| ~ 1e-2 rather than from the metric, which is where the old
//!    round-off floor of eps*|g|/h ~ 2e-12 came from.
//!
//! A side sample in which the hole has no mass lies beyond the instant its mass switches
//! on (or off), and its position, velocity and spin are whatever the orbit source parks a
//! massless hole at.  Those rates are then the one-sided differences between `now` and
//! the side on which the hole has mass, and zero when neither side has any.

KOKKOS_INLINE_FUNCTION
Real CentralRate(const Real minus, const Real plus) {
  return (plus - minus)*(0.5/derivative_step);
}

KOKKOS_INLINE_FUNCTION
Real SampledRate(const Real minus, const Real now, const Real plus,
                 const bool use_minus, const bool use_plus) {
  if (use_minus && use_plus) return CentralRate(minus, plus);
  if (use_plus) return (plus - now)*(1.0/derivative_step);
  if (use_minus) return (now - minus)*(1.0/derivative_step);
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
dual::Dual4 TimeSeed(const Real value, const Real rate) {
  return dual::Dual4(value, 0.0, 0.0, 0.0, rate);
}

KOKKOS_INLINE_FUNCTION
void MakeDualBoostState(const BoostState &minus, const BoostState &now,
                        const BoostState &plus, DualBoostState &bh) {
  const bool use_minus = minus.m > 0.0;
  const bool use_plus = plus.m > 0.0;
  bh.m = TimeSeed(now.m, SampledRate(minus.m, now.m, plus.m, use_minus, use_plus));
  bh.a_param = TimeSeed(now.a_param, SampledRate(minus.a_param, now.a_param,
                                                 plus.a_param, use_minus, use_plus));
  Real sv[3];
  for (int n = 0; n < 3; ++n) {
    sv[n] = SampledRate(minus.v[n], now.v[n], plus.v[n], use_minus, use_plus);
    // d(xi)/dt := v is an ANALYTIC seed, unlike every other rate here, so it is only
    // consistent with the motion the table actually prescribes if the velocity columns
    // are dx/dt in the table's own time units.  That is not a convention the file format
    // states, so the loader now verifies it row by row and refuses the table otherwise
    // (BBH.cpp, "the velocity columns of BHn are not d(x)/dt"); with that check in place
    // the two agree to the table's own O(h^2).  The velocity column is still needed in
    // its own right and cannot be replaced by a differenced position: cf and the boost
    // matrix J below are closed-form functions of v itself, not of dxi/dt.
    bh.xi[n] = TimeSeed(now.xi[n], now.v[n]);
    bh.v[n] = TimeSeed(now.v[n], sv[n]);
    bh.a[n] = TimeSeed(now.a[n], SampledRate(minus.a[n], now.a[n], plus.a[n],
                                             use_minus, use_plus));
  }

  // gamma = (1-v^2)^(-1/2), cf = (gamma-1)/v^2, J = boost(v): closed-form derivatives.
  const Real vx = now.v[0], vy = now.v[1], vz = now.v[2];
  const Real v_sq = SQR(vx) + SQR(vy) + SQR(vz);
  const Real v_dot_s = vx*sv[0] + vy*sv[1] + vz*sv[2];
  const Real gamma_lor = now.J00;
  const Real dgamma = gamma_lor*gamma_lor*gamma_lor*v_dot_s;
  // The v_sq <= 1e-30 branch mirrors InitializeBoostState, which pins cf to zero there;
  // the boost is the identity to O(v^2) so a vanishing rate is the consistent choice.
  const Real dcf = (v_sq > 1.0e-30)
      ? (dgamma*v_sq - (gamma_lor - 1.0)*2.0*v_dot_s)/(v_sq*v_sq) : 0.0;
  const Real cf = now.cf;
  bh.cf = TimeSeed(cf, dcf);
  bh.J00 = TimeSeed(now.J00, dgamma);
  bh.J01 = TimeSeed(now.J01, -(dgamma*vx + gamma_lor*sv[0]));
  bh.J02 = TimeSeed(now.J02, -(dgamma*vy + gamma_lor*sv[1]));
  bh.J03 = TimeSeed(now.J03, -(dgamma*vz + gamma_lor*sv[2]));
  bh.J11 = TimeSeed(now.J11, dcf*vx*vx + cf*(sv[0]*vx + vx*sv[0]));
  bh.J12 = TimeSeed(now.J12, dcf*vx*vy + cf*(sv[0]*vy + vx*sv[1]));
  bh.J13 = TimeSeed(now.J13, dcf*vx*vz + cf*(sv[0]*vz + vx*sv[2]));
  bh.J22 = TimeSeed(now.J22, dcf*vy*vy + cf*(sv[1]*vy + vy*sv[1]));
  bh.J23 = TimeSeed(now.J23, dcf*vy*vz + cf*(sv[1]*vz + vy*sv[2]));
  bh.J33 = TimeSeed(now.J33, dcf*vz*vz + cf*(sv[2]*vz + vz*sv[2]));
}

//----------------------------------------------------------------------------------------
//! \fn void FourMetricFromBoostStates
//! \brief Four-metric and all four of its derivatives from ONE dual evaluation.
//!
//! Replaces the nine-point finite-difference stencil that used to cost eighteen boosted
//! black-hole evaluations per cell (nine metric evaluations, two holes each).  The value
//! is bitwise what the centre evaluation produced; the derivatives are exact for that
//! state, with no truncation error and no eps*|g|/h cancellation floor.

template<typename Params>
KOKKOS_INLINE_FUNCTION
void FourMetricFromBoostStates(const BoostState &bh1_minus, const BoostState &bh1_now,
                               const BoostState &bh1_plus, const BoostState &bh2_minus,
                               const BoostState &bh2_now, const BoostState &bh2_plus,
                               const Real x, const Real y, const Real z,
                               const Params &params, FourMetric &out) {
  DualBoostState dbh1, dbh2;
  MakeDualBoostState(bh1_minus, bh1_now, bh1_plus, dbh1);
  MakeDualBoostState(bh2_minus, bh2_now, bh2_plus, dbh2);
  const dual::Dual4 xd(x, 1.0, 0.0, 0.0, 0.0);
  const dual::Dual4 yd(y, 0.0, 1.0, 0.0, 0.0);
  const dual::Dual4 zd(z, 0.0, 0.0, 1.0, 0.0);
  dual::Dual4 gcov[kMetricDim][kMetricDim];
  ComputeMetricWithBoostGeneric(xd, yd, zd, dbh1, dbh2, gcov, params);
  CopyDualMetric(gcov, out);
  SanitizeDerivative(out.g_t);
  SanitizeDerivative(out.g_x);
  SanitizeDerivative(out.g_y);
  SanitizeDerivative(out.g_z);
}

KOKKOS_INLINE_FUNCTION
Real SafeFinite(const Real value, const Real limit, const Real fallback) {
  return (isfinite(value) && fabs(value) < limit) ? value : fallback;
}

KOKKOS_INLINE_FUNCTION
void ResetThreeMetric(ThreeMetric &gam) {
  gam.gxx = 1.0; gam.gxy = 0.0; gam.gxz = 0.0;
  gam.gyy = 1.0; gam.gyz = 0.0; gam.gzz = 1.0;
  gam.alpha = 1.0;
  gam.betax = 0.0; gam.betay = 0.0; gam.betaz = 0.0;
  gam.kxx = 0.0; gam.kxy = 0.0; gam.kxz = 0.0;
  gam.kyy = 0.0; gam.kyz = 0.0; gam.kzz = 0.0;
}

// Returns false when the point had to fall back to flat space (FourToThreeMetricChecked).
KOKKOS_INLINE_FUNCTION
bool FourToThreeMetricBase(const Symmetric4Metric &met, ThreeMetric &gam) {
  gam.gxx = met.xx; gam.gxy = met.xy; gam.gxz = met.xz;
  gam.gyy = met.yy; gam.gyz = met.yz; gam.gzz = met.zz;
  const Real det = SpatialDet(gam.gxx, gam.gxy, gam.gxz,
                              gam.gyy, gam.gyz, gam.gzz);
  if (!(det > 0.0)) {
    ResetThreeMetric(gam);
    return false;
  }
  const Real invgxx = (-gam.gyz*gam.gyz + gam.gyy*gam.gzz)/det;
  const Real invgxy = (gam.gxz*gam.gyz - gam.gxy*gam.gzz)/det;
  const Real invgxz = (-gam.gxz*gam.gyy + gam.gxy*gam.gyz)/det;
  const Real invgyy = (-gam.gxz*gam.gxz + gam.gxx*gam.gzz)/det;
  const Real invgyz = (gam.gxy*gam.gxz - gam.gxx*gam.gyz)/det;
  const Real invgzz = (-gam.gxy*gam.gxy + gam.gxx*gam.gyy)/det;
  gam.betax = met.tx*invgxx + met.ty*invgxy + met.tz*invgxz;
  gam.betay = met.tx*invgxy + met.ty*invgyy + met.tz*invgyz;
  gam.betaz = met.tx*invgxz + met.ty*invgyz + met.tz*invgzz;
  const Real alpha_sq = met.tx*gam.betax + met.ty*gam.betay
                        + met.tz*gam.betaz - met.tt;
  gam.alpha = (isfinite(alpha_sq) && alpha_sq > 0.0) ? sqrt(alpha_sq) : 0.0;
  gam.kxx = 0.0; gam.kxy = 0.0; gam.kxz = 0.0;
  gam.kyy = 0.0; gam.kyz = 0.0; gam.kzz = 0.0;
  const bool bad = !isfinite(gam.alpha) || gam.alpha <= 1.0e-12
      || !isfinite(gam.gxx) || !isfinite(gam.gyy) || !isfinite(gam.gzz)
      || gam.gxx <= 0.0 || gam.gyy <= 0.0 || gam.gzz <= 0.0
      || fabs(gam.gxx) >= 1.0e12 || fabs(gam.gxy) >= 1.0e12
      || fabs(gam.gxz) >= 1.0e12 || fabs(gam.gyy) >= 1.0e12
      || fabs(gam.gyz) >= 1.0e12 || fabs(gam.gzz) >= 1.0e12;
  if (bad) {
    ResetThreeMetric(gam);
    return false;
  }
  gam.gxy = SafeFinite(gam.gxy, 1.0e12, 0.0);
  gam.gxz = SafeFinite(gam.gxz, 1.0e12, 0.0);
  gam.gyz = SafeFinite(gam.gyz, 1.0e12, 0.0);
  gam.betax = SafeFinite(gam.betax, 1.0e6, 0.0);
  gam.betay = SafeFinite(gam.betay, 1.0e6, 0.0);
  gam.betaz = SafeFinite(gam.betaz, 1.0e6, 0.0);
  return true;
}

// Returns false when the point had to fall back to flat space, so that callers which
// also want derivatives know to zero them rather than differentiate a reset value.
KOKKOS_INLINE_FUNCTION
bool FourToThreeMetricChecked(const FourMetric &met, ThreeMetric &gam) {
  gam.gxx = met.g.xx; gam.gxy = met.g.xy; gam.gxz = met.g.xz;
  gam.gyy = met.g.yy; gam.gyz = met.g.yz; gam.gzz = met.g.zz;
  Real det = SpatialDet(gam.gxx, gam.gxy, gam.gxz, gam.gyy, gam.gyz, gam.gzz);
  if (!(det > 0.0)) {
    ResetThreeMetric(gam);
    return false;
  }

  const Real betadownx = met.g.tx, betadowny = met.g.ty, betadownz = met.g.tz;
  const Real invgxx = (-gam.gyz*gam.gyz + gam.gyy*gam.gzz)/det;
  const Real invgxy = (gam.gxz*gam.gyz - gam.gxy*gam.gzz)/det;
  const Real invgxz = (-gam.gxz*gam.gyy + gam.gxy*gam.gyz)/det;
  const Real invgyy = (-gam.gxz*gam.gxz + gam.gxx*gam.gzz)/det;
  const Real invgyz = (gam.gxy*gam.gxz - gam.gxx*gam.gyz)/det;
  const Real invgzz = (-gam.gxy*gam.gxy + gam.gxx*gam.gyy)/det;
  gam.betax = betadownx*invgxx + betadowny*invgxy + betadownz*invgxz;
  gam.betay = betadownx*invgxy + betadowny*invgyy + betadownz*invgyz;
  gam.betaz = betadownx*invgxz + betadowny*invgyz + betadownz*invgzz;
  const Real alpha_sq = betadownx*gam.betax + betadowny*gam.betay
                        + betadownz*gam.betaz - met.g.tt;
  gam.alpha = (isfinite(alpha_sq) && alpha_sq > 0.0) ? sqrt(alpha_sq) : 0.0;

  const Real dbxx = met.g_x.tx, dbyx = met.g_x.ty, dbzx = met.g_x.tz;
  const Real dbxy = met.g_y.tx, dbyy = met.g_y.ty, dbzy = met.g_y.tz;
  const Real dbxz = met.g_z.tx, dbyz = met.g_z.ty, dbzz = met.g_z.tz;
  const Real dgxxx = met.g_x.xx, dgxyx = met.g_x.xy, dgxzx = met.g_x.xz;
  const Real dgyyx = met.g_x.yy, dgyzx = met.g_x.yz, dgzzx = met.g_x.zz;
  const Real dgxxy = met.g_y.xx, dgxyy = met.g_y.xy, dgxzy = met.g_y.xz;
  const Real dgyyy = met.g_y.yy, dgyzy = met.g_y.yz, dgzzy = met.g_y.zz;
  const Real dgxxz = met.g_z.xx, dgxyz = met.g_z.xy, dgxzz = met.g_z.xz;
  const Real dgyyz = met.g_z.yy, dgyzz = met.g_z.yz, dgzzz = met.g_z.zz;

  gam.kxx = -(-2.0*dbxx - gam.betax*dgxxx - gam.betay*dgxxy
      - gam.betaz*dgxxz + 2.0*(gam.betax*dgxxx + gam.betay*dgxyx
      + gam.betaz*dgxzx) + met.g_t.xx)/(2.0*gam.alpha);
  gam.kxy = -(-dbxy - dbyx + gam.betax*dgxxy - gam.betaz*dgxyz
      + gam.betaz*dgxzy + gam.betay*dgyyx + gam.betaz*dgyzx
      + met.g_t.xy)/(2.0*gam.alpha);
  gam.kxz = -(-dbxz - dbzx + gam.betax*dgxxz + gam.betay*dgxyz
      - gam.betay*dgxzy + gam.betay*dgyzx + gam.betaz*dgzzx
      + met.g_t.xz)/(2.0*gam.alpha);
  gam.kyy = -(-2.0*dbyy - gam.betax*dgyyx - gam.betay*dgyyy
      - gam.betaz*dgyyz + 2.0*(gam.betax*dgxyy + gam.betay*dgyyy
      + gam.betaz*dgyzy) + met.g_t.yy)/(2.0*gam.alpha);
  gam.kyz = -(-dbyz - dbzy + gam.betax*dgxyz + gam.betax*dgxzy
      + gam.betay*dgyyz - gam.betax*dgyzx + gam.betaz*dgzzy
      + met.g_t.yz)/(2.0*gam.alpha);
  gam.kzz = -(-2.0*dbzz - gam.betax*dgzzx - gam.betay*dgzzy
      - gam.betaz*dgzzz + 2.0*(gam.betax*dgxzz + gam.betay*dgyzz
      + gam.betaz*dgzzz) + met.g_t.zz)/(2.0*gam.alpha);

  const bool bad = !isfinite(gam.alpha) || gam.alpha <= 1.0e-12
      || !isfinite(gam.gxx) || !isfinite(gam.gyy) || !isfinite(gam.gzz)
      || gam.gxx <= 0.0 || gam.gyy <= 0.0 || gam.gzz <= 0.0
      || fabs(gam.gxx) >= 1.0e12 || fabs(gam.gxy) >= 1.0e12
      || fabs(gam.gxz) >= 1.0e12 || fabs(gam.gyy) >= 1.0e12
      || fabs(gam.gyz) >= 1.0e12 || fabs(gam.gzz) >= 1.0e12;
  if (bad) {
    ResetThreeMetric(gam);
    return false;
  }
  gam.gxy = SafeFinite(gam.gxy, 1.0e12, 0.0);
  gam.gxz = SafeFinite(gam.gxz, 1.0e12, 0.0);
  gam.gyz = SafeFinite(gam.gyz, 1.0e12, 0.0);
  gam.betax = SafeFinite(gam.betax, 1.0e6, 0.0);
  gam.betay = SafeFinite(gam.betay, 1.0e6, 0.0);
  gam.betaz = SafeFinite(gam.betaz, 1.0e6, 0.0);
  gam.kxx = SafeFinite(gam.kxx, 1.0e6, 0.0);
  gam.kxy = SafeFinite(gam.kxy, 1.0e6, 0.0);
  gam.kxz = SafeFinite(gam.kxz, 1.0e6, 0.0);
  gam.kyy = SafeFinite(gam.kyy, 1.0e6, 0.0);
  gam.kyz = SafeFinite(gam.kyz, 1.0e6, 0.0);
  gam.kzz = SafeFinite(gam.kzz, 1.0e6, 0.0);
  return true;
}

KOKKOS_INLINE_FUNCTION
void FourToThreeMetric(const FourMetric &met, ThreeMetric &gam) {
  (void) FourToThreeMetricChecked(met, gam);
}

//----------------------------------------------------------------------------------------
//! \fn void ThreeMetricDerivativesFromFour
//! \brief Reconstruct d_k gamma_ij, d_k alpha, d_k beta^i, and d_t alpha from the
//! four-metric and its already-computed derivatives, using the exact 3+1 chain rule:
//!   gamma_ij = g_ij                  ->  d_k gamma_ij = (g_k)_ij
//!   beta_i   = g_ti                  ->  d_k beta_i   = (g_k)_ti
//!   beta^i   = gamma^ij beta_j       ->  d_k beta^i   = (d_k gamma^ij) beta_j
//!                                                     + gamma^ij (d_k beta_j)
//!   d_k gamma^ij = -gamma^ia gamma^jb (d_k gamma_ab)
//!   alpha^2  = beta^i beta_i - g_tt  ->  d_k alpha    = [ (d_k beta^i) beta_i
//!                                          + beta^i (d_k beta_i) - (g_k)_tt ] / (2 alpha)
//!
//! This replaces a grid finite-difference stencil over neighbouring cells.  For a
//! prescribed analytical spacetime it is both cheaper (the four-metric derivatives are
//! already computed for K_ij and would otherwise be discarded) and more accurate: the
//! four-metric derivatives it is fed carry no truncation error at all, so the only error
//! left is round-off, against the O(dx^{2*NGHOST-2}) truncation of the grid stencil.

KOKKOS_INLINE_FUNCTION
void ThreeMetricDerivativesFromFour(const FourMetric &met, const ThreeMetric &gam,
                                    const bool metric_ok, ThreeMetricDerivatives &d) {
  d.dalpha_dt = 0.0;
  for (int dir = 0; dir < 3; ++dir) {
    d.dalpha_d[dir] = 0.0;
    for (int a = 0; a < 3; ++a) d.dbeta_du[dir][a] = 0.0;
    for (int n = 0; n < 6; ++n) d.dgam_ddd[dir][n] = 0.0;
  }
  // A point that fell back to flat space carries no meaningful local gradient, and the
  // reset is a step function, so differentiating it would manufacture a huge source term.
  if (!metric_ok || !(gam.alpha > 1.0e-12)) return;

  const Real det = SpatialDet(gam.gxx, gam.gxy, gam.gxz, gam.gyy, gam.gyz, gam.gzz);
  if (!(det > 0.0) || !isfinite(det)) return;
  const Real idet = 1.0/det;
  // Upper spatial metric, indexed as [xx,xy,xz,yy,yz,zz] -> full 3x3 below.
  const Real iuxx = (-gam.gyz*gam.gyz + gam.gyy*gam.gzz)*idet;
  const Real iuxy = ( gam.gxz*gam.gyz - gam.gxy*gam.gzz)*idet;
  const Real iuxz = (-gam.gxz*gam.gyy + gam.gxy*gam.gyz)*idet;
  const Real iuyy = (-gam.gxz*gam.gxz + gam.gxx*gam.gzz)*idet;
  const Real iuyz = ( gam.gxy*gam.gxz - gam.gxx*gam.gyz)*idet;
  const Real iuzz = (-gam.gxy*gam.gxy + gam.gxx*gam.gyy)*idet;
  const Real gu[3][3] = {{iuxx, iuxy, iuxz}, {iuxy, iuyy, iuyz}, {iuxz, iuyz, iuzz}};

  const Real beta_d[3] = {met.g.tx, met.g.ty, met.g.tz};
  const Real beta_u[3] = {gam.betax, gam.betay, gam.betaz};
  const Symmetric4Metric *dmet[3] = {&met.g_x, &met.g_y, &met.g_z};

  // The prescribed BBH evaluator already returns partial_t g_ab.  Carry its exact lapse
  // derivative through the same 3+1 chain rule used for the spatial derivatives below.
  // Written directly in terms of covariant metric derivatives,
  //   2 alpha d alpha = -d g_tt + 2 beta^i d g_ti
  //                     - beta^i beta^j d gamma_ij.
  // This avoids reconstructing d beta^i and then cancelling its inverse-metric terms,
  // which loses precision when alpha is small.  No extra metric evaluation or finite-
  // time stencil is needed.
  {
    const Symmetric4Metric &dg = met.g_t;
    const Real dgam[3][3] = {{dg.xx, dg.xy, dg.xz},
                             {dg.xy, dg.yy, dg.yz},
                             {dg.xz, dg.yz, dg.zz}};
    const Real dbeta_d[3] = {dg.tx, dg.ty, dg.tz};
    Real dalpha_sq = -dg.tt;
    for (int a = 0; a < 3; ++a) {
      dalpha_sq += 2.0 * beta_u[a] * dbeta_d[a];
      for (int b = 0; b < 3; ++b) {
        dalpha_sq -= beta_u[a] * beta_u[b] * dgam[a][b];
      }
    }
    d.dalpha_dt = 0.5 * dalpha_sq / gam.alpha;
  }

  for (int dir = 0; dir < 3; ++dir) {
    const Symmetric4Metric &dg = *dmet[dir];
    // d_k gamma_ij, in S11..S33 order.
    d.dgam_ddd[dir][0] = dg.xx; d.dgam_ddd[dir][1] = dg.xy;
    d.dgam_ddd[dir][2] = dg.xz; d.dgam_ddd[dir][3] = dg.yy;
    d.dgam_ddd[dir][4] = dg.yz; d.dgam_ddd[dir][5] = dg.zz;
    const Real dgam[3][3] = {{dg.xx, dg.xy, dg.xz},
                             {dg.xy, dg.yy, dg.yz},
                             {dg.xz, dg.yz, dg.zz}};
    const Real dbeta_d[3] = {dg.tx, dg.ty, dg.tz};

    // d_k gamma^ij = -gamma^ia gamma^jb d_k gamma_ab
    Real dgu[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b <= a; ++b) {
        Real s = 0.0;
        for (int c = 0; c < 3; ++c) {
          for (int e = 0; e < 3; ++e) {
            s += gu[a][c]*gu[b][e]*dgam[c][e];
          }
        }
        dgu[a][b] = -s;
        dgu[b][a] = -s;
      }
    }

    // Use the direct covariant identity above for d_k alpha as well.  d beta^i is still
    // required by callers, but it need not participate in the small-lapse subtraction.
    Real dalpha_sq = -dg.tt;
    for (int a = 0; a < 3; ++a) {
      Real dbu = 0.0;
      for (int b = 0; b < 3; ++b) {
        dbu += dgu[a][b]*beta_d[b] + gu[a][b]*dbeta_d[b];
      }
      d.dbeta_du[dir][a] = dbu;
      dalpha_sq += 2.0 * beta_u[a] * dbeta_d[a];
      for (int b = 0; b < 3; ++b) {
        dalpha_sq -= beta_u[a] * beta_u[b] * dgam[a][b];
      }
    }
    d.dalpha_d[dir] = 0.5*dalpha_sq/gam.alpha;
  }

  // Match the point-wise clamping policy so a pathological cell cannot inject a huge
  // source term through the derivatives while its metric was quietly bounded.
  d.dalpha_dt = SafeFinite(d.dalpha_dt, 1.0e12, 0.0);
  for (int dir = 0; dir < 3; ++dir) {
    d.dalpha_d[dir] = SafeFinite(d.dalpha_d[dir], 1.0e12, 0.0);
    for (int a = 0; a < 3; ++a) {
      d.dbeta_du[dir][a] = SafeFinite(d.dbeta_du[dir][a], 1.0e12, 0.0);
    }
    for (int n = 0; n < 6; ++n) {
      d.dgam_ddd[dir][n] = SafeFinite(d.dgam_ddd[dir][n], 1.0e12, 0.0);
    }
  }
}

// Full four-metric plus its t/x/y/z derivatives at one point, from a single forward-mode
// dual evaluation (it used to be 9 ComputeMetricWithBoost evaluations on a 5e-5 central
// stencil).  Exposed separately from EvaluateMetricFromState so that a caller who wants
// the ADM derivatives can reuse the spatial derivatives computed here instead of
// rebuilding them with a grid stencil (see ThreeMetricDerivativesFromFour).
template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateFourMetricFromState(const MetricState &state, const Real x, const Real y,
                                 const Real z, const Params &params, FourMetric &four) {
  FourMetricFromBoostStates(state.bh1_minus, state.bh1_now, state.bh1_plus,
                            state.bh2_minus, state.bh2_now, state.bh2_plus,
                            x, y, z, params, four);
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateMetricFromState(const MetricState &state, const Real x, const Real y,
                             const Real z, const Params &params, ThreeMetric &metric) {
  FourMetric four{};
  EvaluateFourMetricFromState(state, x, y, z, params, four);
  FourToThreeMetric(four, metric);
}

// Point value AND exact spatial derivatives for the price of the point value alone.
template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateMetricAndDerivativesFromState(const MetricState &state, const Real x,
                                           const Real y, const Real z,
                                           const Params &params, ThreeMetric &metric,
                                           ThreeMetricDerivatives &derivatives) {
  FourMetric four{};
  EvaluateFourMetricFromState(state, x, y, z, params, four);
  const bool ok = FourToThreeMetricChecked(four, metric);
  ThreeMetricDerivativesFromFour(four, metric, ok, derivatives);
}

//----------------------------------------------------------------------------------------
//! \fn void EvaluateSnapshotMetricAndDerivatives
//! \brief The 3+1 metric and its exact SPATIAL first derivatives at one point of the
//! spacetime frozen at the `now` sample of a MetricState, from one Dual3 evaluation.
//!
//! For a caller that treats the metric as stationary (the direct-field geodesics,
//! radiation_m1_direct_trace.hpp).  The holes enter as the constants they are there, so
//! no time rate is formed (MakeDualBoostState), every product with a hole's parameters
//! is a Real-by-dual product, and a dual carries three partials instead of four.  What
//! needs d_t g_ab is not formed: K_ij is zero and d_t alpha is zero.  gamma_ij, alpha,
//! beta^i and their spatial derivatives are those EvaluateMetricAndDerivativesFromState
//! returns: the value arithmetic is the same operation for operation, the time partials
//! never enter a spatial one, and the 3+1 split takes the same guards
//! (FourToThreeMetricBase is FourToThreeMetricChecked without K_ij).
template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateSnapshotMetricAndDerivatives(const MetricState &state, const Real x,
                                          const Real y, const Real z,
                                          const Params &params, ThreeMetric &metric,
                                          ThreeMetricDerivatives &derivatives) {
  const dual::Dual3 xd(x, 1.0, 0.0, 0.0);
  const dual::Dual3 yd(y, 0.0, 1.0, 0.0);
  const dual::Dual3 zd(z, 0.0, 0.0, 1.0);
  dual::Dual3 gcov[kMetricDim][kMetricDim];
  ComputeMetricWithBoostGeneric(xd, yd, zd, state.bh1_now, state.bh2_now, gcov, params);
  // The ten independent components, as CopyDualMetric reads them; g_t stays zero.
  FourMetric four{};
  const dual::Dual3 *comp[10] = {&gcov[0][0], &gcov[0][1], &gcov[0][2], &gcov[0][3],
                                 &gcov[1][1], &gcov[1][2], &gcov[1][3], &gcov[2][2],
                                 &gcov[2][3], &gcov[3][3]};
  Symmetric4Metric *blk[4] = {&four.g, &four.g_x, &four.g_y, &four.g_z};
  for (int b = 0; b < 4; ++b) {
    Real *out[10] = {&blk[b]->tt, &blk[b]->tx, &blk[b]->ty, &blk[b]->tz, &blk[b]->xx,
                     &blk[b]->xy, &blk[b]->xz, &blk[b]->yy, &blk[b]->yz, &blk[b]->zz};
    for (int n = 0; n < 10; ++n) *out[n] = (b == 0) ? comp[n]->v : comp[n]->d[b - 1];
  }
  SanitizeDerivative(four.g_x);
  SanitizeDerivative(four.g_y);
  SanitizeDerivative(four.g_z);
  const bool ok = FourToThreeMetricBase(four.g, metric);
  ThreeMetricDerivativesFromFour(four, metric, ok, derivatives);
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateKeplerMetricBase(const Real time, const Real x, const Real y, const Real z,
                              const Params &params, ThreeMetric &metric) {
  Real traj[kTrajectorySize];
  KeplerTrajectory(time, params, traj);
  BoostState bh1, bh2;
  PrecomputeBoostState(traj, params, bh1, bh2);
  Real gcov[kMetricDim][kMetricDim];
  ComputeMetricWithBoost(x, y, z, bh1, bh2, gcov, params);
  Symmetric4Metric four{};
  CopyMetric(gcov, four);
  FourToThreeMetricBase(four, metric);
}

template<typename Params>
KOKKOS_INLINE_FUNCTION
void EvaluateMetricBaseFromState(const MetricState &state, const Real x, const Real y,
                                 const Real z, const Params &params,
                                 ThreeMetric &metric) {
  Real gcov[kMetricDim][kMetricDim];
  ComputeMetricWithBoost(x, y, z, state.bh1_now, state.bh2_now, gcov, params);
  Symmetric4Metric four{};
  CopyMetric(gcov, four);
  FourToThreeMetricBase(four, metric);
}

} // namespace analytic_bbh
} // namespace adm

#endif // COORDINATES_ANALYTIC_BBH_HPP_
