#ifndef PGEN_TDE_AMR_HPP_
#define PGEN_TDE_AMR_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_amr.hpp
//! \brief Per-cell classification of the <tde_amr> refinement scheme of tde_external.
//!
//! Pure inline math, callable from device kernels and from a host-only unit test.  A cell
//! is compared with the local spine density rho_ref(r, phi), the max density of its (log
//! r, phi) bin, r the distance to the BH and phi the azimuth in the orbital plane (a bin
//! is a shell sector of every height), and is classified as core (q = rho/rho_ref >=
//! spine_frac), envelope (one level lost per factor envelope_factor below spine_frac, up
//! to envelope_levels) or background.  Core and envelope cells then take the finest
//! region they match (nozzle, post-nozzle, self-interaction, apocentre, spine ladder),
//! every region is a logical level max_level - offset, and an envelope cell sits its drop
//! below that, so the stream flanks are resolved one level below the spine.  With
//! midplane_only a block may only rise above offplane_level if it intersects the band |h|
//! <= max(midplane_hmin, tan(angle) R) around the orbital plane through the BH
//! (BlockInBand); only the core star and the sink are exempt.  The relaxed evaluation
//! (T_down) divides the q, strand and crossing thresholds by hysteresis and moves every
//! radius bound and the band outwards by radius_hysteresis (the band at least one finest
//! cell wide), so each region only grows and T_down >= T_up holds cell by cell.

#include "athena.hpp"

namespace tde_amr {

// Region ids double as the tie-break priority: at equal level the larger id is reported.
enum Region : int {
  kBackground = 0,
  kOffplane,     // a stream target capped because the block lies off the midplane band
  kSpine,
  kApocentre,
  kCrossing,
  kPostNozzle,
  kNozzle,
  kCoreStar,
  kSink,
  kNumRegions
};

struct Params {
  // Logical levels.
  int max_level;
  int background_level;
  int offplane_level;
  // Spine reference and per-cell classification.
  Real rho_min;
  Real strand_frac;
  Real spine_frac;
  Real envelope_factor;
  int envelope_levels;
  // Regions.
  Real nozzle_radius;
  Real post_nozzle_radius;
  int nozzle_offset;
  int post_nozzle_offset;
  Real crossing_div;
  Real crossing_r_max;
  int crossing_offset;
  Real apocentre_frac;
  int apocentre_boost;  // levels finer than the spine ladder at the same radius
  int spine_offset;
  Real spine_r_per_level;
  int spine_offset_max;
  bool core_enable;
  Real core_rho;
  Real core_rho_deref;
  int core_offset;
  // Midplane switch: block eligibility band |h| <= max(midplane_hmin, midplane_tan R).
  bool midplane_only;
  Real midplane_hmin;
  Real midplane_tan;
  Real band_floor;         // relaxed band half-width floor: one finest cell
  Real hysteresis;         // relaxes the density ratios, strand gate and crossing_div
  Real radius_hysteresis;  // relaxes every radius bound and the midplane band
  Real gm;  // G M_bh
  // Orbital plane: unit normal and an in-plane orthonormal pair.
  Real nx, ny, nz;
  Real e1x, e1y, e1z;
  Real e2x, e2y, e2z;
  // (log r, phi) bins of the spine table.
  int nbins_r;
  int nbins_phi;
  Real rbin_min;
  Real rbin_max;
  Real log_rbin_min;
  Real inv_dlog_r;
};

// Everything the classification needs to know about one cell.
struct Cell {
  Real rho;
  Real rho_ref;   // max density of the cell's bin, <= 0 if the bin is empty
  Real ring_max;  // max of rho_ref over phi at the bin's radius
  Real r;         // distance to the BH
  Real vr;        // radial velocity relative to the BH
  Real div_v;
  Real eps;       // specific orbital energy relative to the BH (softened potential)
  Real j2;        // squared specific angular momentum relative to the BH
};

//----------------------------------------------------------------------------------------
//! \fn int BinIndex()
//! \brief Flat index ir*nbins_phi + iphi of the (log r, phi) bin of an offset (dx,dy,dz)
//! from the BH: r is the distance to the BH and phi the azimuth in the orbital plane, so
//! a bin is a shell sector of every height.  Binning by the radius projected into the
//! plane instead would put the gas above and below the BH into narrow bins of its own,
//! where any diffuse cell is the local maximum.  Cells inside rbin_min fall in the first
//! ring; cells at r >= rbin_max are outside the table (-1).

KOKKOS_INLINE_FUNCTION
int BinIndex(const Params &p, const Real dx, const Real dy, const Real dz) {
  const Real x = dx*p.e1x + dy*p.e1y + dz*p.e1z;
  const Real y = dx*p.e2x + dy*p.e2y + dz*p.e2z;
  const Real rr = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
  if (!(rr < p.rbin_max)) return -1;
  int ir = 0;
  if (rr > p.rbin_min) {
    ir = static_cast<int>(Kokkos::floor((Kokkos::log(rr) - p.log_rbin_min)*p.inv_dlog_r));
    ir = (ir < 0) ? 0 : ((ir > p.nbins_r - 1) ? (p.nbins_r - 1) : ir);
  }
  const Real phi = Kokkos::atan2(y, x) + static_cast<Real>(M_PI);  // [0, 2 pi]
  int iphi = static_cast<int>(Kokkos::floor(phi*static_cast<Real>(p.nbins_phi)/
                                            static_cast<Real>(2.0*M_PI)));
  iphi = (iphi < 0) ? 0 : ((iphi > p.nbins_phi - 1) ? (p.nbins_phi - 1) : iphi);
  return ir*p.nbins_phi + iphi;
}

KOKKOS_INLINE_FUNCTION
Real PlaneHeight(const Params &p, const Real dx, const Real dy, const Real dz) {
  return dx*p.nx + dy*p.ny + dz*p.nz;
}

//----------------------------------------------------------------------------------------
//! \fn int EnvelopeDrop()
//! \brief 0 for a core cell (q >= frac), d = 1..nlev for an envelope cell with
//! frac/factor^d <= q < frac/factor^(d-1), and -1 below the envelope.

KOKKOS_INLINE_FUNCTION
int EnvelopeDrop(const Real q, const Real frac, const Real factor, const int nlev) {
  if (q >= frac) return 0;
  Real thr = frac;
  for (int d = 1; d <= nlev; ++d) {
    thr /= factor;
    if (q >= thr) return d;
  }
  return -1;
}

//----------------------------------------------------------------------------------------
//! \fn Real ApocentreRadius()
//! \brief Apocentre a(1+e) of the Kepler orbit with specific energy eps < 0 and squared
//! specific angular momentum j2 about a point mass GM; -1 for eps >= 0.

KOKKOS_INLINE_FUNCTION
Real ApocentreRadius(const Real eps, const Real j2, const Real gm) {
  if (!(eps < 0.0)) return -1.0;
  const Real a = -gm/(2.0*eps);
  // The softened potential can put eps below the Kepler minimum for this j, which would
  // make e^2 negative: that orbit is circular as far as the apocentre is concerned.
  const Real e2 = 1.0 + 2.0*eps*j2/(gm*gm);
  const Real e = (e2 > 0.0) ? Kokkos::sqrt(e2) : 0.0;
  return a*(1.0 + e);
}

//----------------------------------------------------------------------------------------
//! \fn int SpineOffset()
//! \brief Spine ladder: spine_offset + floor(log(r/(R_noz*relax))/log(r_per_level)),
//! with r_per_level = spine_r_per_level, never below spine_offset and capped at
//! spine_offset_max (default: no cap).  Counted by repeated multiplication so the ring edges are exact.

KOKKOS_INLINE_FUNCTION
int SpineOffset(const Params &p, const Real r, const Real relax) {
  int off = p.spine_offset;
  Real edge = p.nozzle_radius*relax*p.spine_r_per_level;
  while (off < p.spine_offset_max && r >= edge) {
    ++off;
    edge *= p.spine_r_per_level;
  }
  return off;
}

//----------------------------------------------------------------------------------------
//! \fn bool BlockInBand()
//! \brief Whether a block, given by its extent relative to the BH, intersects the band
//! |h| <= max(midplane_hmin, midplane_tan*R) around the orbital plane, the band taken at
//! the in-plane radius R of the block point nearest to the BH.  The signed plane distance
//! is linear, so its range over the block is spanned by the corners.  The relaxed band is
//! radius_hysteresis wider and at least band_floor (one finest cell), so a plane lying on
//! a block face keeps both blocks.

KOKKOS_INLINE_FUNCTION
bool BlockInBand(const Params &p, const Real x0, const Real x1, const Real y0,
                 const Real y1, const Real z0, const Real z1, const bool relaxed) {
  const Real cx = (x0 > 0.0) ? x0 : ((x1 < 0.0) ? x1 : static_cast<Real>(0.0));
  const Real cy = (y0 > 0.0) ? y0 : ((y1 < 0.0) ? y1 : static_cast<Real>(0.0));
  const Real cz = (z0 > 0.0) ? z0 : ((z1 < 0.0) ? z1 : static_cast<Real>(0.0));
  const Real hc = cx*p.nx + cy*p.ny + cz*p.nz;
  const Real rr = Kokkos::sqrt(Kokkos::fmax(cx*cx + cy*cy + cz*cz - hc*hc,
                                            static_cast<Real>(0.0)));
  Real band = Kokkos::fmax(p.midplane_hmin, p.midplane_tan*rr);
  if (relaxed) band = Kokkos::fmax(band*p.radius_hysteresis, p.band_floor);
  const Real hlo = Kokkos::fmin(p.nx*x0, p.nx*x1) + Kokkos::fmin(p.ny*y0, p.ny*y1) +
                   Kokkos::fmin(p.nz*z0, p.nz*z1);
  const Real hhi = Kokkos::fmax(p.nx*x0, p.nx*x1) + Kokkos::fmax(p.ny*y0, p.ny*y1) +
                   Kokkos::fmax(p.nz*z0, p.nz*z1);
  return hlo <= band && hhi >= -band;
}

// Keep the finest (smallest) offset; at equal offset the higher-priority region.
KOKKOS_INLINE_FUNCTION
void TakeRegion(const int o, const int g, int &off, int &reg) {
  if (off < 0 || o < off || (o == off && g > reg)) {
    off = o;
    reg = g;
  }
}

//----------------------------------------------------------------------------------------
//! \fn int CellTarget()
//! \brief Target logical level of one cell, nominal (relaxed = false, T_up) or relaxed by
//! the hysteresis factors (T_down).  in_band is BlockInBand of the cell's block for the
//! same evaluation; outside the band (midplane_only) the stream target is capped at
//! offplane_level and reported as kOffplane.  region returns the region that sets the
//! level; a level at or below background_level is otherwise reported as background.

KOKKOS_INLINE_FUNCTION
int CellTarget(const Params &p, const Cell &c, const bool relaxed, const bool in_band,
               int &region) {
  const Real hy = relaxed ? p.hysteresis : static_cast<Real>(1.0);
  const Real hr = relaxed ? p.radius_hysteresis : static_cast<Real>(1.0);
  int best_level = p.background_level;
  int best_region = kBackground;

  // Stream classification against the local spine.  A bin whose spine is far below the
  // ring maximum (a region the stream has left) carries no strand.
  int drop = -1;
  if (c.rho > p.rho_min && c.rho_ref > 0.0 &&
      c.rho_ref >= (p.strand_frac/hy)*c.ring_max) {
    drop = EnvelopeDrop(c.rho/c.rho_ref, p.spine_frac/hy, p.envelope_factor,
                        p.envelope_levels);
  }
  if (drop >= 0) {
    const Real r_noz_in = p.nozzle_radius*hr;   // nozzle is r < r_noz_in
    const Real r_noz_out = p.nozzle_radius/hr;  // regions outside it start at r_noz_out
    int off = -1;
    int reg = kBackground;
    if (c.r < r_noz_in) TakeRegion(p.nozzle_offset, kNozzle, off, reg);
    if (c.r >= r_noz_out && c.r < p.post_nozzle_radius*hr && c.vr > 0.0) {
      TakeRegion(p.post_nozzle_offset, kPostNozzle, off, reg);
    }
    if (c.r >= r_noz_out && c.r < p.crossing_r_max*hr && c.r > 0.0) {
      const Real omega_k = Kokkos::sqrt(p.gm/(c.r*c.r*c.r));
      if (-c.div_v >= (p.crossing_div/hy)*omega_k) {
        TakeRegion(p.crossing_offset, kCrossing, off, reg);
      }
    }
    const int spine_off = SpineOffset(p, c.r, hr);
    const Real r_apo = ApocentreRadius(c.eps, c.j2, p.gm);
    if (r_apo > 0.0 && c.r >= (p.apocentre_frac/hr)*r_apo) {
      const int apo_off = spine_off - p.apocentre_boost;
      TakeRegion((apo_off > 0) ? apo_off : 0, kApocentre, off, reg);
    }
    TakeRegion(spine_off, kSpine, off, reg);
    // Every stream cell matches at least the spine ladder.  Off the midplane band the
    // whole stream, spine included, stops at offplane_level.
    int level = p.max_level - off - drop;
    if (p.midplane_only && !in_band && level > p.offplane_level) {
      level = p.offplane_level;
      reg = kOffplane;
    }
    if (level > best_level || (reg == kOffplane && level == best_level)) {
      best_level = level;
      best_region = reg;
    }
  }

  // Optional pre-disruption core star, kept until rho < core_rho_deref.
  if (p.core_enable && c.rho >= (relaxed ? p.core_rho_deref : p.core_rho)) {
    const int level = p.max_level - p.core_offset;
    if (level > best_level || (level == best_level && level > p.background_level)) {
      best_level = level;
      best_region = kCoreStar;
    }
  }
  region = best_region;
  return best_level;
}

//----------------------------------------------------------------------------------------
//! \fn int BlockFlag()
//! \brief Exact target with hysteresis: refine below T_up, derefine above T_down.

KOKKOS_INLINE_FUNCTION
int BlockFlag(const int level, const int t_up, const int t_down) {
  if (level < t_up) return 1;
  if (level > t_down) return -1;
  return 0;
}

}  // namespace tde_amr

#endif  // PGEN_TDE_AMR_HPP_
