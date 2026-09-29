#ifndef COORDINATES_ADM_HPP_
#define COORDINATES_ADM_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.hpp
//! \brief definitions for ADM class

#include <cstdint>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/primitive-solver/ps_types.hpp"
#include "coordinates/analytic_bbh.hpp"
#include "coordinates/cell_locations.hpp"

// forward declarations
class MeshBlockPack;

namespace adm {

enum class ADMMetricBackend : unsigned char {
  stored,
  // Prescribed analytical superposed-Kerr-Schild binary, evaluated per point from a
  // compact host-built state.  The state is derived purely from three trajectory samples,
  // so this backend is agnostic to whether the orbit comes from the closed-form Keplerian
  // formula or from a tabulated post-Newtonian trajectory file.
  bbh_analytic_onthefly
};

struct ADMMetricPoint {
  Real g_dd[NSPMETRIC];
  Real K_dd[NSPMETRIC];
  Real alpha;
  Real beta_u[3];
  Real psi4;
};

struct ADMMetricDerivatives {
  Real dg_ddd[3][NSPMETRIC];
  Real dalpha_d[3];
  Real dbeta_du[3][3];
  // Exact for the prescribed analytic-BBH backend; explicitly zero for stationary fixed
  // metrics and for the stored backend, which has no time stencil in this view.
  Real dalpha_dt;
};

//! Component order of the optional base-metric cache: exactly the ten doubles the base
//! metric carries.  K_ij and psi4 are deliberately absent because the analytical base
//! evaluation defines neither -- analytic_bbh::FourToThreeMetricBase always returns
//! K_ij = 0 and CopyAnalyticMetric always reports psi4 = 0 -- so storing them would only
//! duplicate two compile-time constants.
enum ADMBaseCacheIndex {
  I_BCACHE_GXX, I_BCACHE_GXY, I_BCACHE_GXZ,
  I_BCACHE_GYY, I_BCACHE_GYZ, I_BCACHE_GZZ,
  I_BCACHE_ALPHA, I_BCACHE_BETAX, I_BCACHE_BETAY, I_BCACHE_BETAZ,
  nadmbasecache
};

struct ADMMetricView;

//! \class ADM
class ADM {
 public:
  //! WARNING: The ADM object needs to be allocated after Z4c
  ADM(MeshBlockPack *ppack, ParameterInput *pin);
  ~ADM();

  // Indices of ADM variables
  enum {
    I_ADM_GXX, I_ADM_GXY, I_ADM_GXZ, I_ADM_GYY, I_ADM_GYZ, I_ADM_GZZ,
    I_ADM_KXX, I_ADM_KXY, I_ADM_KXZ, I_ADM_KYY, I_ADM_KYZ, I_ADM_KZZ,
    I_ADM_PSI4,
    I_ADM_ALPHA, I_ADM_BETAX, I_ADM_BETAY, I_ADM_BETAZ,
    nadm
  };
  // Names of ADM variables
  static char const * const ADM_names[nadm];

  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;     // lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;    // shift vector
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi4;      // conformal factor
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;      // spatial metric
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;     // extrinsic curvature
  };
  ADM_vars adm;

  struct ADMhost_vars {
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 0> alpha;
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 1> beta_u;
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };

  DvceArray5D<Real> u_adm;                                // adm variables
  bool is_dynamic;                                        // is the metric time dependent?
  bool callback_time_override_valid;                      // callback should use override time
  Real callback_time_override;                            // explicit prescribed-metric time

  ADMMetricBackend metric_backend;

  bool StoresMetricGrid() const {
    return metric_backend == ADMMetricBackend::stored;
  }
  int RestartVariableCount() const {
    return StoresMetricGrid() ? nadm : 0;
  }
  bool IsAnalyticBBH() const {
    return metric_backend == ADMMetricBackend::bbh_analytic_onthefly;
  }

  //! Host callback supplying one trajectory sample (positions, velocities, spins, masses)
  //! at an arbitrary time.  The problem generator registers it so that ADM never has to
  //! know which orbit model is in use; without one, the closed-form Keplerian orbit built
  //! from the configured parameters is used.
  using TrajectorySampler =
      void (*)(Real time, Real traj[analytic_bbh::kTrajectorySize]);

  void ConfigureAnalyticBBH(const analytic_bbh::KeplerParameters &params,
                            TrajectorySampler sampler = nullptr);
  void SetMetricTime(Real time);
  //! The time of the installed analytic metric (SetMetricTime).
  Real MetricTime() const { return metric_time; }
  ADMMetricView GetMetricView() const;
  ADMMetricView GetMetricView(Real time) const;
  //! Fill a metric state for an arbitrary time using whichever orbit source is
  //! registered.
  void BuildMetricStateAtTime(Real time, analytic_bbh::MetricState &state) const;
  //! Refill the base-metric cache for the currently installed state.  Called by
  //! SetMetricTime; public only because CUDA forbids extended lambdas inside private
  //! member functions.  There is no reason to call it from outside ADM.
  void FillBaseMetricCache();

  void (*SetADMVariables)(MeshBlockPack *pm);

  static void SetADMVariablesToKerrSchild(MeshBlockPack *pm);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
  analytic_bbh::KeplerParameters bbh_params;
  TrajectorySampler bbh_trajectory_sampler;
  bool bbh_configured;
  analytic_bbh::MetricState bbh_metric_state{};
  Real metric_time;

  //! Optional per-install cache of the BASE metric only (gamma_ij, alpha, beta^i), for
  //! the analytical backend.  See ADM::FillBaseMetricCache for the validity protocol.
  bool base_metric_cache_enabled;
  DvceArray5D<Real> base_cache;       // (nmb, nadmbasecache, ncells3, ncells2, ncells1)
  DvceArray1D<int> base_cache_stamp;  // per-block epoch at which that block was filled
  int base_cache_epoch;               // epoch of the most recent fill
  std::uint64_t base_cache_topology_version;  // mesh topology the fill was made against
};

struct ADMMetricView {
  ADMMetricBackend backend;
  ADM::ADM_vars stored;
  DualArray1D<RegionSize> size;
  RegionIndcs indcs;
  analytic_bbh::KeplerParameters bbh;
  analytic_bbh::MetricState bbh_state;

  // Optional base-metric cache (analytical backend only).  A view may read
  // base_cache(m,...) if and only if base_cache_valid is true -- meaning the cache was
  // filled from exactly the bbh_state/bbh this view carries, against the mesh topology
  // still in force -- AND base_cache_stamp(m) == base_cache_epoch, meaning this
  // particular block was one of the blocks that fill actually covered.  Anything else
  // (LAT-inactive block, a view built for a different metric time, cache switched off)
  // falls through to on-the-fly evaluation.
  DvceArray5D<Real> base_cache;
  DvceArray1D<int> base_cache_stamp;
  int base_cache_epoch;
  bool base_cache_valid;

  KOKKOS_INLINE_FUNCTION
  void CartesianMetric(const Real x, const Real y, const Real z,
                       ADMMetricPoint &point) const {
    analytic_bbh::ThreeMetric metric{};
    analytic_bbh::EvaluateMetricBaseFromState(bbh_state, x, y, z, bbh, metric);
    CopyAnalyticMetric(metric, point);
  }

  KOKKOS_INLINE_FUNCTION
  void CartesianMetricFull(const Real x, const Real y, const Real z,
                           ADMMetricPoint &point) const {
    analytic_bbh::ThreeMetric metric{};
    analytic_bbh::EvaluateMetricFromState(bbh_state, x, y, z, bbh, metric);
    CopyAnalyticMetric(metric, point);
  }

  KOKKOS_INLINE_FUNCTION
  void CellMetric(const int m, const int k, const int j, const int i,
                  ADMMetricPoint &point) const {
    if (backend == ADMMetricBackend::stored) {
      LoadStoredBase(m, k, j, i, point);
      return;
    }
    if (BaseCacheHit(m, k, j, i)) {
      LoadCachedBase(m, k, j, i, point);
      return;
    }
    Real x, y, z;
    CellCoordinates(m, k, j, i, x, y, z);
    CartesianMetric(x, y, z, point);
  }

  KOKKOS_INLINE_FUNCTION
  void CellMetricFull(const int m, const int k, const int j, const int i,
                      ADMMetricPoint &point) const {
    if (backend == ADMMetricBackend::stored) {
      LoadStoredFull(m, k, j, i, point);
      return;
    }
    Real x, y, z;
    CellCoordinates(m, k, j, i, x, y, z);
    CartesianMetricFull(x, y, z, point);
  }

  template<int DIR>
  KOKKOS_INLINE_FUNCTION
  void FaceMetric(const int m, const int k, const int j, const int i,
                  Real gface_dd[NSPMETRIC], Real betaface_u[3],
                  Real &alphaface) const {
    static_assert(DIR >= 1 && DIR <= 3, "ADM face direction must be 1, 2, or 3");
    constexpr int di = (DIR == 1) ? 1 : 0;
    constexpr int dj = (DIR == 2) ? 1 : 0;
    constexpr int dk = (DIR == 3) ? 1 : 0;
    ADMMetricPoint left{}, right{};
    CellMetric(m, k-dk, j-dj, i-di, left);
    CellMetric(m, k, j, i, right);
    alphaface = 0.5*(left.alpha + right.alpha);
    for (int a = 0; a < 3; ++a) {
      betaface_u[a] = 0.5*(left.beta_u[a] + right.beta_u[a]);
    }
    for (int n = 0; n < NSPMETRIC; ++n) {
      gface_dd[n] = 0.5*(left.g_dd[n] + right.g_dd[n]);
    }
  }

  template<int NGHOST>
  KOKKOS_INLINE_FUNCTION
  void CellMetricAndDerivatives(const int m, const int k, const int j, const int i,
                                ADMMetricPoint &point,
                                ADMMetricDerivatives &derivatives) const {
    static_assert(NGHOST >= 2 && NGHOST <= 4,
                  "ADM derivatives require two, three, or four ghost zones");
    // Analytical backend: EvaluateMetricAndDerivativesFromState carries the four-metric
    // and all four of its derivatives out of a SINGLE forward-mode dual evaluation, and
    // the exact 3+1 chain rule turns the spatial ones straight into the ADM derivatives.
    // So the derivatives cost one metric evaluation per cell -- the same as the point
    // value alone -- against the 1 + 6*(NGHOST-1) the grid stencil below needs, i.e. one
    // instead of nineteen at NGHOST=4.  It is also the more accurate of the two for a
    // prescribed spacetime: the spatial derivatives are exact for the state, with no
    // truncation error and no eps*|g|/h cancellation floor, whereas the stencil's own
    // O(dx^(2*NGHOST-2)) truncation blows up on coarse refinement levels (dx ~ 8.5 on
    // level 2 of the production BBH mesh).  The stencil path below is still required for
    // the stored backend, which has nothing but grid samples.
    if (backend != ADMMetricBackend::stored) {
      Real x, y, z;
      CellCoordinates(m, k, j, i, x, y, z);
      analytic_bbh::ThreeMetric metric{};
      analytic_bbh::ThreeMetricDerivatives d{};
      analytic_bbh::EvaluateMetricAndDerivativesFromState(bbh_state, x, y, z, bbh,
                                                          metric, d);
      CopyAnalyticMetric(metric, point);
      derivatives.dalpha_dt = d.dalpha_dt;
      for (int dir = 0; dir < 3; ++dir) {
        // Match the stencil path: a degenerate dimension carries no gradient.
        const bool degenerate = (dir == 1 && indcs.nx2 <= 1) ||
                                (dir == 2 && indcs.nx3 <= 1);
        derivatives.dalpha_d[dir] = degenerate ? 0.0 : d.dalpha_d[dir];
        for (int a = 0; a < 3; ++a) {
          derivatives.dbeta_du[dir][a] = degenerate ? 0.0 : d.dbeta_du[dir][a];
        }
        for (int n = 0; n < NSPMETRIC; ++n) {
          derivatives.dg_ddd[dir][n] = degenerate ? 0.0 : d.dgam_ddd[dir][n];
        }
      }
      return;
    }
    CellMetricFull(m, k, j, i, point);
    derivatives.dalpha_dt = 0.0;
    for (int dir = 0; dir < 3; ++dir) {
      derivatives.dalpha_d[dir] = 0.0;
      for (int a = 0; a < 3; ++a) derivatives.dbeta_du[dir][a] = 0.0;
      for (int n = 0; n < NSPMETRIC; ++n) derivatives.dg_ddd[dir][n] = 0.0;
      if ((dir == 1 && indcs.nx2 <= 1) || (dir == 2 && indcs.nx3 <= 1)) {
        continue;
      }
      const int radius = NGHOST - 1;
      const Real idx = 1.0/((dir == 0) ? size.d_view(m).dx1
                                      : ((dir == 1) ? size.d_view(m).dx2
                                                    : size.d_view(m).dx3));
      for (int offset = 1; offset <= radius; ++offset) {
        Real coeff = 0.0;
        if constexpr (NGHOST == 2) {
          coeff = 0.5;
        } else if constexpr (NGHOST == 3) {
          coeff = (offset == 1) ? (2.0/3.0) : (-1.0/12.0);
        } else {
          coeff = (offset == 1) ? (3.0/4.0)
                                : ((offset == 2) ? (-3.0/20.0) : (1.0/60.0));
        }
        const int di = (dir == 0) ? offset : 0;
        const int dj = (dir == 1) ? offset : 0;
        const int dk = (dir == 2) ? offset : 0;
        ADMMetricPoint minus{}, plus{};
        CellMetric(m, k-dk, j-dj, i-di, minus);
        CellMetric(m, k+dk, j+dj, i+di, plus);
        derivatives.dalpha_d[dir] += coeff*(plus.alpha-minus.alpha)*idx;
        for (int a = 0; a < 3; ++a) {
          derivatives.dbeta_du[dir][a] +=
              coeff*(plus.beta_u[a]-minus.beta_u[a])*idx;
        }
        for (int n = 0; n < NSPMETRIC; ++n) {
          derivatives.dg_ddd[dir][n] +=
              coeff*(plus.g_dd[n]-minus.g_dd[n])*idx;
        }
      }
    }
  }

 private:
  KOKKOS_INLINE_FUNCTION
  void CellCoordinates(const int m, const int k, const int j, const int i,
                       Real &x, Real &y, Real &z) const {
    x = CellCenterX(i-indcs.is, indcs.nx1, size.d_view(m).x1min,
                    size.d_view(m).x1max);
    y = CellCenterX(j-indcs.js, indcs.nx2, size.d_view(m).x2min,
                    size.d_view(m).x2max);
    z = CellCenterX(k-indcs.ks, indcs.nx3, size.d_view(m).x3min,
                    size.d_view(m).x3max);
  }

  KOKKOS_INLINE_FUNCTION
  static void CopyAnalyticMetric(const analytic_bbh::ThreeMetric &metric,
                                 ADMMetricPoint &point) {
    point.g_dd[S11] = metric.gxx; point.g_dd[S12] = metric.gxy;
    point.g_dd[S13] = metric.gxz; point.g_dd[S22] = metric.gyy;
    point.g_dd[S23] = metric.gyz; point.g_dd[S33] = metric.gzz;
    point.K_dd[S11] = metric.kxx; point.K_dd[S12] = metric.kxy;
    point.K_dd[S13] = metric.kxz; point.K_dd[S22] = metric.kyy;
    point.K_dd[S23] = metric.kyz; point.K_dd[S33] = metric.kzz;
    point.alpha = metric.alpha;
    point.beta_u[0] = metric.betax;
    point.beta_u[1] = metric.betay;
    point.beta_u[2] = metric.betaz;
    // The prescribed BBH callback does not define a conformal factor and leaves
    // the stored diagnostic field at zero. Preserve that backend behavior.
    point.psi4 = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void LoadStoredBase(const int m, const int k, const int j, const int i,
                      ADMMetricPoint &point) const {
    point.g_dd[S11] = stored.g_dd(m,0,0,k,j,i);
    point.g_dd[S12] = stored.g_dd(m,0,1,k,j,i);
    point.g_dd[S13] = stored.g_dd(m,0,2,k,j,i);
    point.g_dd[S22] = stored.g_dd(m,1,1,k,j,i);
    point.g_dd[S23] = stored.g_dd(m,1,2,k,j,i);
    point.g_dd[S33] = stored.g_dd(m,2,2,k,j,i);
    point.alpha = stored.alpha(m,k,j,i);
    point.beta_u[0] = stored.beta_u(m,0,k,j,i);
    point.beta_u[1] = stored.beta_u(m,1,k,j,i);
    point.beta_u[2] = stored.beta_u(m,2,k,j,i);
    for (int n = 0; n < NSPMETRIC; ++n) point.K_dd[n] = 0.0;
    point.psi4 = 1.0;
  }

  //! Is the base metric of cell (m,k,j,i) available from the cache?  Three independent
  //! conditions, all of which must hold:
  //!   1. base_cache_valid -- the cache is allocated, switched on, was filled from the
  //!      very bbh_state this view carries, and the mesh topology has not changed since.
  //!      This is decided once on the host in ADM::GetMetricView.
  //!   2. the index is inside the allocation.  On-the-fly evaluation of an out-of-range
  //!      index is harmless (it just extrapolates a cell centre), so this guard keeps a
  //!      caller that reaches outside a block reading the same values it always did
  //!      instead of reading past the end of the array.
  //!   3. this block's stamp matches the current epoch -- i.e. the most recent fill
  //!      actually covered block m.  A LAT-inactive block fails here and falls back.
  KOKKOS_INLINE_FUNCTION
  bool BaseCacheHit(const int m, const int k, const int j, const int i) const {
    if (!base_cache_valid) return false;
    if (m < 0 || m >= base_cache.extent_int(0)) return false;
    if (k < 0 || k >= base_cache.extent_int(2)) return false;
    if (j < 0 || j >= base_cache.extent_int(3)) return false;
    if (i < 0 || i >= base_cache.extent_int(4)) return false;
    return base_cache_stamp(m) == base_cache_epoch;
  }

  //! Reload the ten cached doubles.  These are the same bit patterns the analytical
  //! evaluation produced when the cache was filled -- nothing is converted, rounded or
  //! recombined on the way in or out.  K_ij and psi4 are not cached because the base
  //! path never produces anything but zeros for them.
  KOKKOS_INLINE_FUNCTION
  void LoadCachedBase(const int m, const int k, const int j, const int i,
                      ADMMetricPoint &point) const {
    point.g_dd[S11] = base_cache(m,I_BCACHE_GXX,k,j,i);
    point.g_dd[S12] = base_cache(m,I_BCACHE_GXY,k,j,i);
    point.g_dd[S13] = base_cache(m,I_BCACHE_GXZ,k,j,i);
    point.g_dd[S22] = base_cache(m,I_BCACHE_GYY,k,j,i);
    point.g_dd[S23] = base_cache(m,I_BCACHE_GYZ,k,j,i);
    point.g_dd[S33] = base_cache(m,I_BCACHE_GZZ,k,j,i);
    point.alpha = base_cache(m,I_BCACHE_ALPHA,k,j,i);
    point.beta_u[0] = base_cache(m,I_BCACHE_BETAX,k,j,i);
    point.beta_u[1] = base_cache(m,I_BCACHE_BETAY,k,j,i);
    point.beta_u[2] = base_cache(m,I_BCACHE_BETAZ,k,j,i);
    for (int n = 0; n < NSPMETRIC; ++n) point.K_dd[n] = 0.0;
    point.psi4 = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void LoadStoredFull(const int m, const int k, const int j, const int i,
                      ADMMetricPoint &point) const {
    LoadStoredBase(m, k, j, i, point);
    point.K_dd[S11] = stored.vK_dd(m,0,0,k,j,i);
    point.K_dd[S12] = stored.vK_dd(m,0,1,k,j,i);
    point.K_dd[S13] = stored.vK_dd(m,0,2,k,j,i);
    point.K_dd[S22] = stored.vK_dd(m,1,1,k,j,i);
    point.K_dd[S23] = stored.vK_dd(m,1,2,k,j,i);
    point.K_dd[S33] = stored.vK_dd(m,2,2,k,j,i);
    point.psi4 = stored.psi4(m,k,j,i);
  }
};

KOKKOS_INLINE_FUNCTION
Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz) {
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz
         - SQR(gyz)*gxx
         - SQR(gxy)*gzz +   gxx*gyy*gzz;
}

KOKKOS_INLINE_FUNCTION
Real Trace(Real const detginv,
           Real const gxx, Real const gxy, Real const gxz,
           Real const gyy, Real const gyz, Real const gzz,
           Real const Axx, Real const Axy, Real const Axz,
           Real const Ayy, Real const Ayz, Real const Azz) {
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}

KOKKOS_INLINE_FUNCTION
// compute inverse of a 3x3 matrix
void SpatialInv(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real * uxx, Real * uxy, Real * uxz,
                Real * uyy, Real * uyz, Real * uzz) {
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

KOKKOS_INLINE_FUNCTION
void SpacetimeMetric(Real const alp,
                     Real const betax, Real const betay, Real const betaz,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real g[16]) {
  g[5]  = gxx;
  g[6]  = gxy;
  g[7]  = gxz;
  g[9]  = gxy;
  g[10] = gyy;
  g[11] = gyz;
  g[13] = gxz;
  g[14] = gyz;
  g[15] = gzz;

  Real betaup[3] = {betax, betay, betaz};
  Real betadw[3] = {
    gxx*betax + gxy*betay + gxz*betaz,
    gxy*betax + gyy*betay + gyz*betaz,
    gxz*betax + gyz*betay + gzz*betaz,
  };

  g[0] = - SQR(alp) + betadw[0]*betaup[0] + betadw[1]*betaup[1] +
          betadw[2]*betaup[2];

  g[1] = betadw[0];
  g[2] = betadw[1];
  g[3] = betadw[2];

  g[4]  = betadw[0];
  g[8]  = betadw[1];
  g[12] = betadw[2];
}

KOKKOS_INLINE_FUNCTION
void SpacetimeUpperMetric(Real const alp,
                          Real const betax, Real const betay, Real const betaz,
                          Real const gxx, Real const gxy, Real const gxz,
                          Real const gyy, Real const gyz, Real const gzz,
                          Real u[16]) {
  u[0] = - 1.0/SQR(alp);

  Real const det = SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);

  Real uxx, uxy, uxz, uyy, uyz, uzz;
  SpatialInv(1.0/det, gxx, gxy, gxz, gyy, gyz, gzz,
             &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
  u[5]  = uxx + betax*betax*u[0];
  u[6]  = uxy + betax*betay*u[0];
  u[7]  = uxz + betax*betaz*u[0];
  u[9]  = uxy + betax*betay*u[0];
  u[10] = uyy + betay*betay*u[0];
  u[11] = uyz + betay*betaz*u[0];
  u[13] = uxz + betax*betaz*u[0];
  u[14] = uyz + betay*betaz*u[0];
  u[15] = uzz + betaz*betaz*u[0];

  u[1] = betax*(-u[0]);
  u[2] = betay*(-u[0]);
  u[3] = betaz*(-u[0]);

  u[4]  = u[1];
  u[8]  = u[2];
  u[12] = u[3];
}


//----------------------------------------------------------------------------------------
//! \fn void Face1Metric
//! \brief computes components of (dynamically evolved) 3-metric, lapse and
//  shift at faces in the x direction for use in Riemann solver for a single point
//check your indices: interface i lives between cells i and i-1

KOKKOS_INLINE_FUNCTION
void Face1Metric(const int m, const int k, const int j, const int i,
     const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 0> &alpha,
     Real gface1_dd[NSPMETRIC], Real betaface1_u[3], Real &alphaface1) {
  alphaface1 = (alpha(m,k,j,i) + alpha(m,k,j,i-1))*0.5;

  for (int a = 0; a < 3; ++a) {
    betaface1_u[a] = (beta_u(m,a,k,j,i) + beta_u(m,a,k,j,i-1))*0.5;
  }

  gface1_dd[S11] = (g_dd(m,0,0,k,j,i) + g_dd(m,0,0,k,j,i-1))*0.5;
  gface1_dd[S12] = (g_dd(m,0,1,k,j,i) + g_dd(m,0,1,k,j,i-1))*0.5;
  gface1_dd[S13] = (g_dd(m,0,2,k,j,i) + g_dd(m,0,2,k,j,i-1))*0.5;
  gface1_dd[S22] = (g_dd(m,1,1,k,j,i) + g_dd(m,1,1,k,j,i-1))*0.5;
  gface1_dd[S23] = (g_dd(m,1,2,k,j,i) + g_dd(m,1,2,k,j,i-1))*0.5;
  gface1_dd[S33] = (g_dd(m,2,2,k,j,i) + g_dd(m,2,2,k,j,i-1))*0.5;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Face2Metric
//! \brief computes components of (dynamically evolved) 3-metric, lapse and
//  shift at faces in the y direction for use in Riemann solver for a single point
//check your indices: interface j lives between cells j and j-1

KOKKOS_INLINE_FUNCTION
void Face2Metric(const int m, const int k, const int j, const int i,
     const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 0> &alpha,
     Real gface2_dd[NSPMETRIC], Real betaface2_u[3], Real &alphaface2) {
  alphaface2 = (alpha(m,k,j,i) + alpha(m,k,j-1,i))*0.5;

  for (int a = 0; a < 3; ++a) {
    betaface2_u[a] = (beta_u(m,a,k,j,i) + beta_u(m,a,k,j-1,i))*0.5;
  }

  gface2_dd[S11] = (g_dd(m,0,0,k,j,i) + g_dd(m,0,0,k,j-1,i))*0.5;
  gface2_dd[S12] = (g_dd(m,0,1,k,j,i) + g_dd(m,0,1,k,j-1,i))*0.5;
  gface2_dd[S13] = (g_dd(m,0,2,k,j,i) + g_dd(m,0,2,k,j-1,i))*0.5;
  gface2_dd[S22] = (g_dd(m,1,1,k,j,i) + g_dd(m,1,1,k,j-1,i))*0.5;
  gface2_dd[S23] = (g_dd(m,1,2,k,j,i) + g_dd(m,1,2,k,j-1,i))*0.5;
  gface2_dd[S33] = (g_dd(m,2,2,k,j,i) + g_dd(m,2,2,k,j-1,i))*0.5;

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Face3Metric
//! \brief computes components of (dynamically evolved) 3-metric, lapse and
//  shift at faces in the z direction for use in Riemann solver for a single point
//check your indices: interface k lives between cells k and k-1

KOKKOS_INLINE_FUNCTION
void Face3Metric(const int m, const int k, const int j, const int i,
     const AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 1> &beta_u,
     const AthenaTensor<Real, TensorSymm::NONE, 3, 0> &alpha,
     Real gface3_dd[NSPMETRIC], Real betaface3_u[3], Real &alphaface3) {
  alphaface3 = (alpha(m,k,j,i) + alpha(m,k-1,j,i))*0.5;

  for (int a = 0; a < 3; ++a) {
    betaface3_u[a] = (beta_u(m,a,k,j,i) + beta_u(m,a,k-1,j,i))*0.5;
  }

  gface3_dd[S11] = (g_dd(m,0,0,k,j,i) + g_dd(m,0,0,k-1,j,i))*0.5;
  gface3_dd[S12] = (g_dd(m,0,1,k,j,i) + g_dd(m,0,1,k-1,j,i))*0.5;
  gface3_dd[S13] = (g_dd(m,0,2,k,j,i) + g_dd(m,0,2,k-1,j,i))*0.5;
  gface3_dd[S22] = (g_dd(m,1,1,k,j,i) + g_dd(m,1,1,k-1,j,i))*0.5;
  gface3_dd[S23] = (g_dd(m,1,2,k,j,i) + g_dd(m,1,2,k-1,j,i))*0.5;
  gface3_dd[S33] = (g_dd(m,2,2,k,j,i) + g_dd(m,2,2,k-1,j,i))*0.5;

  return;
}


} // namespace adm
#endif // COORDINATES_ADM_HPP_
