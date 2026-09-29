#ifndef REMAP_REMAP_IMPL_HPP_
#define REMAP_REMAP_IMPL_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_impl.hpp
//! \brief internal data structures shared by the remap module translation units.
//! Not part of the public API; include remap.hpp from pgens instead.

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "outputs/io_wrapper.hpp"

struct EOS_Data;

namespace remap {

struct RemapOptions;

namespace impl {

//----------------------------------------------------------------------------------------
//! \enum RemapRelClass
//! \brief relativity class of a remap end (source or target).  A remap is only performed
//! between two ends of the SAME class; z4c/CCE ends are refused outright.
//!
//! kNewtonian      — flat space, Newtonian conserved variables (the v1 module).
//! kFixedGR        — a prescribed, time-independent GR background (<coord>/general_rel;
//!                   e.g. gr_torus in Cartesian/spherical Kerr-Schild).
//! kDynGRAnalytic  — the dyn_grmhd (<adm>) path with an analytically prescribed metric
//!                   backend, i.e. the BBH pgen.  Note that Coordinates reports
//!                   is_general_relativistic == FALSE for these, which is why the class
//!                   must be derived from the presence of <adm>/padm, never from the
//!                   Coordinates flags alone.
enum class RemapRelClass { kNewtonian, kFixedGR, kDynGRAnalytic };

const char *RemapRelClassName(RemapRelClass c);

struct RemapBlockKey {
  std::int32_t lx1, lx2, lx3, level;

  bool operator==(const RemapBlockKey &other) const {
    return (lx1 == other.lx1 && lx2 == other.lx2 &&
            lx3 == other.lx3 && level == other.level);
  }
};

struct RemapBlockKeyHash {
  std::size_t operator()(const RemapBlockKey &key) const {
    std::size_t h = static_cast<std::size_t>(key.level);
    h = (h * 1315423911u) ^ static_cast<std::size_t>(key.lx1);
    h = (h * 1315423911u) ^ static_cast<std::size_t>(key.lx2);
    h = (h * 1315423911u) ^ static_cast<std::size_t>(key.lx3);
    return h;
  }
};

struct RemapSourceBlock {
  LogicalLocation lloc;
  RegionSize size;
  std::vector<Real> hydro;    // gas CC group (hydro u0 or mhd u0), (nvars,nout3,nout2,nout1)
  std::vector<Real> i0;       // optional radiation CC group, (nangles,nout3,nout2,nout1)
  // Thermal energy density of every source cell, (nout3,nout2,nout1).  Filled by
  // BuildSourceThermalEnergy on the Newtonian path: it is what the CC engine
  // interpolates instead of the total energy.
  std::vector<Real> eint;
  // sqrt(det gamma_ij) at every cell center of this block, evaluated at the SOURCE time.
  // Filled only when RemapSourceData::metric is active; it is the divisor that takes the
  // metric back out of the densitized GR conserved variables before interpolation.
  std::vector<Real> sqrt_gamma;   // (nout3,nout2,nout1)
};

//----------------------------------------------------------------------------------------
//! \struct RemapMetricSampler
//! \brief spatial metric gamma_ij at an ARBITRARY point, evaluated at the SOURCE time.
//!
//! On the dyn-GR (<adm>) path every conserved column is DENSITIZED -- PrimitiveSolver
//! stores cons*sdetg with sdetg = sqrt(det gamma_ij), for D, S_i, tau, the scalars and
//! (as volform) the M1 moments -- so interpolating the stored columns directly puts the
//! sqrt(gamma) profile inside the remap's smoothing filter: the recovered rho then
//! carries an error proportional to the CURVATURE of the metric, which peaks exactly
//! where the punctures are.  This sampler lets remap_load.cpp divide the metric out at
//! the source cell centers (the interpolation nodes) and remap_cc.cpp put it back at the
//! destination, so only the fluid profile is ever filtered.  It also supplies gamma^{ij}
//! for the M1 admissibility clamp, which the flat norm over-limits.
//!
//! Two ends deliberately have no sampler.  Fixed-GR (kFixedGR) does not densitize at all
//! -- ideal_c2p_mhd.hpp SingleP2C_IdealGRMHD stores rho*u^0, and the volume element lives
//! in the flux/source terms -- so there is nothing to divide out and a sqrt(gamma) round
//! trip there would be a change of variables the data does not have.  And a STORED ADM
//! backend has nothing but samples on the target mesh, so gamma_ij cannot be evaluated at
//! a source cell center at all; the module keeps the v1 behavior and warns on rank 0.
struct RemapMetricSampler {
  enum class Kind { kNone, kAnalyticBBH };
  Kind kind = Kind::kNone;
  adm::ADMMetricView bbh_view{};  // kAnalyticBBH: the binary metric AT THE SOURCE TIME

  bool Active() const { return kind != Kind::kNone; }
  // gamma_ij in the primitive-solver component order (S11, S12, S13, S22, S23, S33)
  void SpatialMetric(const Real x, const Real y, const Real z,
                     Real g_dd[NSPMETRIC]) const;
  Real SqrtGamma(const Real x, const Real y, const Real z) const;
};

// Uniform covering-grid face fields and edge potentials on the SOURCE ROOT grid.
// Only allocated when the source carries MHD and the target wants the FC remap.
struct RemapCoveringField {
  int n1 = 0, n2 = 0, n3 = 0;         // root cells per axis
  Real dx1 = 0.0, dx2 = 0.0, dx3 = 0.0;
  Real x1min = 0.0, x2min = 0.0, x3min = 0.0;
  Real x1max = 0.0, x2max = 0.0, x3max = 0.0;
  // faces: b1(n3, n2, n1+1), b2(n3, n2+1, n1), b3(n3+1, n2, n1)
  std::vector<Real> b1f, b2f, b3f;
  std::vector<Real> b1w, b2w, b3w;    // accumulated area weights (freed after normalize)
  // edge potentials, gauge A1 == 0:
  //   a2(k in [0..n3], j in [0..n2-1], i in [0..n1])
  //   a3(k in [0..n3-1], j in [0..n2], i in [0..n1])
  std::vector<Real> a2, a3;
  Real a2_shell_mean = 0.0, a3_shell_mean = 0.0;
  Real inv_curl_b1_err = 0.0;         // max |curl(A)_1 - b1f| consistency diagnostic
  Real boundary_bmax = 0.0;           // max |B| on the covering-domain boundary shell
  Real interior_bmax = 0.0;           // max |B| anywhere (for the warning ratio)
};

struct RemapSourceData {
  int nmb_total = 0;
  int root_level = 0;
  int max_level = 0;
  RegionSize mesh_size;
  RegionIndcs mesh_indcs;
  RegionIndcs mb_indcs;
  Real time = 0.0;
  Real dt = 0.0;
  int ncycle = 0;
  int nvars = 0;              // gas CC group width in the FILE
  int nx1 = 0;
  int nx2 = 1;
  int nx3 = 1;
  int ngh = 0;
  int nout1 = 0;
  int nout2 = 1;
  int nout3 = 1;
  int nmb_rootx1 = 1;
  int nmb_rootx2 = 1;
  int nmb_rootx3 = 1;
  IOWrapperSizeT data_size = 0;
  IOWrapperSizeT data_offset = 0;
  // per-block intra-payload byte offsets of the sections we consume.  The layout mirrors
  // src/outputs/restart.cpp exactly: hydro u0 | mhd u0 | b0.x1f | b0.x2f | b0.x3f |
  // rad i0 | turb force | z4c XOR adm | ffe.
  IOWrapperSizeT gas_cc_offset = 0;
  IOWrapperSizeT fc_offset = 0;       // start of b0.x1f (valid iff source_has_mhd)
  IOWrapperSizeT i0_offset = 0;       // valid iff source_has_rad
  IOWrapperSizeT adm_skip_bytes = 0;  // trailing <adm> u_adm section, skipped (not read)
  RemapRelClass rel_class = RemapRelClass::kNewtonian;
  bool gr_mode = false;               // rel_class != kNewtonian
  bool source_has_mhd = false;
  bool source_has_rad = false;        // source carries <radiation> (i0 intensities)
  bool load_i0 = false;               // i0 group actually loaded and applied
  bool load_fc = false;               // FC covering grid built and applied
  // banner/diagnostic bookkeeping for groups present on exactly one side
  bool i0_discarded = false;          // source has radiation, target does not
  bool i0_target_only = false;        // target has radiation, source does not
  int nvars_i0 = 0;                   // source nangles (derived from radiation/nlevel)
  int nadm_vars = 0;                  // trailing <adm> u_adm width: 0 (analytic backends,
                                      // BBH included) or exactly adm::ADM::nadm
  RemapMetricSampler metric;          // gamma_ij at the source time (GR, prescribed only)
  std::vector<RemapSourceBlock> blocks;
  std::unordered_map<RemapBlockKey, int, RemapBlockKeyHash> block_map;
  RemapCoveringField cov;
};

// ---- remap_load.cpp ----
bool LoadRemapSourceData(const std::string &path, Mesh *pm, MeshBlockPack *pmbp,
                         const RemapOptions &opts, RemapSourceData &src,
                         ParameterInput &src_pin);
// GR-only source-vs-target parameter consistency audit (fatals + rank-0 warnings).  Never
// touches dst_pin with a GetOrAdd*, so the run's own parameter dump stays clean.
void CheckRemapGRConsistency(ParameterInput &src_pin, ParameterInput *dst_pin,
                             const RemapSourceData &src);

// ---- remap_cc.cpp ----
RegionSize LogicalLocationToRegionSize(const RegionSize &mesh_size,
                                       const RegionIndcs &mesh_indcs,
                                       const RegionIndcs &mb_indcs,
                                       const int root_level,
                                       const LogicalLocation &lloc);
bool RegionOverlaps(const RegionSize &a, const RegionSize &b);
RegionSize LocalPackBounds(Mesh *pm);
RegionSize ExpandRegionSize(const RegionSize &in, const Real pad_x1,
                            const Real pad_x2, const Real pad_x3);
void FloorOuterSourceGhostZones(RemapSourceData &src, const int dual_energy_idx,
                                const Real floor_rho, const Real floor_eint);
// Fills RemapSourceBlock::eint for every source block (Newtonian path).  Must run AFTER
// FloorOuterSourceGhostZones, so the floored outer ghosts and this column agree, and
// after the loader's magnetic-energy subtraction, so IEN is the GAS total energy.
void BuildSourceThermalEnergy(RemapSourceData &src, const int base_nvars,
                              const Real floor_rho, const Real floor_eint);
void SampleRemapCellAverage(const RemapSourceData &src, const RegionSize &cell_box,
                            const std::vector<Real> &floor_state,
                            const int base_nvars,
                            const int dual_energy_idx,
                            const EOS_Data &eos,
                            const Real floor_rho, const Real floor_eint,
                            const Real floor_p,
                            const bool use_band,
                            std::vector<Real> &cell_average);
// kKeepTarget sampler: fills cell_average with the plain (thermodynamics-free) source
// average over the cell and RETURNS the cell's blend weight w in [0,1].  w == 0 means the
// cell has no source support at all and must be left holding the target's pgen state.
Real SampleKeepCellAverage(const RemapSourceData &src, const RegionSize &cell_box,
                           const std::vector<Real> RemapSourceBlock::*arr,
                           const int nvars_arr, const bool use_band,
                           const int taper_root_cells, const bool undensitize,
                           std::vector<Real> &cell_average);
// The kKeepTarget support weight at ONE point: 1 inside the source active box, a quintic
// smoothstep decaying to 0 across taper_root_cells source ROOT cells outside it, and 0
// beyond.  remap_fc.cpp evaluates the same function so that B and the gas agree, cell for
// cell, on where the target's own pgen state survives the remap.
Real KeepSampleWeight(const RemapSourceData &src, const Real x, const Real y,
                      const Real z, const bool use_band, const int taper_root_cells);
void ApplyRemapCC(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                  const RemapSourceData &src);

// ---- remap_fc.cpp ----
void BuildCoveringPotential(RemapSourceData &src, const RemapOptions &opts);
void ApplyRemapFC(Mesh *pm, MeshBlockPack *pmbp, const RemapOptions &opts,
                  const RemapSourceData &src);

}  // namespace impl
}  // namespace remap

#endif  // REMAP_REMAP_IMPL_HPP_
