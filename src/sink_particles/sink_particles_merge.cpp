//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_merge.cpp
//! \brief friends-of-friends merging of sinks (ORION2
//! SinkParticleList::mergeSinkParticles + combineSinkParticles).  Runs purely on the
//! host, redundantly and identically on every rank from the replicated list, so it
//! contains no MPI and no device kernel.
//!
//! Differences from ORION2, all deliberate (design D-e):
//!  * the kd-tree is replaced by an O(N^2) union-find; N is at most a few hundred here,
//!    and the flat sweep makes group ordering deterministic, which is what keeps every
//!    rank's copy of the list identical;
//!  * the link length is `merge_link_cells*dx_finest`, not a hard-coded 8*dx (R19);
//!  * distances use the minimum image on periodic directions (R19b), and each group is
//!    merged in a frame unwrapped about its reference member so the mass-weighted
//!    centroid of a pair straddling a periodic face is still correct.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

namespace {

//----------------------------------------------------------------------------------------
//! \fn SepSq
//! \brief squared minimum-image separation between two positions.

Real SepSq(const Real *pa, const Real *pb, const SinkMeshGeom &g) {
  Real r2 = 0.0;
  for (int d=0; d<3; ++d) {
    r2 += SQR(SinkWrapDelta(pa[d] - pb[d], g.len[d], g.periodic[d]));
  }
  return r2;
}

//----------------------------------------------------------------------------------------
//! \fn UnwrapAbout
//! \brief copy of `p` shifted into the periodic image nearest `ref`.

SinkData UnwrapAbout(const SinkData &p, const Real *ref, const SinkMeshGeom &g) {
  SinkData q = p;
  for (int d=0; d<3; ++d) {
    q.pos[d] = ref[d] + SinkWrapDelta(p.pos[d] - ref[d], g.len[d], g.periodic[d]);
  }
  return q;
}

//----------------------------------------------------------------------------------------
//! \fn MergeInto
//! \brief `*a += b`, transcribed from SinkParticleData::operator+=
//! (SinkParticleData.cpp:124-170).  Mass, linear momentum and total angular momentum
//! about the new centre of mass are conserved exactly; the pre-merge orbital angular
//! momentum is converted into spin.  There is no energy accounting and no boundedness
//! test: the relative kinetic energy of the pair is discarded (ORION2 behaviour, R17d).
//! Both arguments must already be in a common unwrapped frame.

void MergeInto(SinkData *a, const SinkData &b) {
  const Real m1 = a->m;
  const Real m2 = b.m;
  const Real mnew = m1 + m2;
  if (!(mnew > 0.0)) { return; }

  Real x1[3], x2[3], pnew[3], p1[3], p2[3], xnew[3];
  for (int d=0; d<3; ++d) {
    x1[d] = a->pos[d];
    x2[d] = b.pos[d];
    pnew[d] = a->mom[d] + b.mom[d];
    xnew[d] = (x1[d]*m1 + x2[d]*m2)/mnew;
  }
  for (int d=0; d<3; ++d) {
    const Real vcom = pnew[d]/mnew;
    p1[d] = (m1 > 0.0) ? m1*(a->mom[d]/m1 - vcom) : 0.0;
    p2[d] = (m2 > 0.0) ? m2*(b.mom[d]/m2 - vcom) : 0.0;
  }

  Real r1[3], r2[3];
  for (int d=0; d<3; ++d) {
    r1[d] = x1[d] - xnew[d];
    r2[d] = x2[d] - xnew[d];
    a->pos[d] = xnew[d];
    a->mom[d] = pnew[d];
    a->angmom[d] += b.angmom[d];
  }
  a->m = mnew;
  a->angmom[0] += (r1[1]*p1[2] - r1[2]*p1[1]) + (r2[1]*p2[2] - r2[2]*p2[1]);
  a->angmom[1] += (r1[2]*p1[0] - r1[0]*p1[2]) + (r2[2]*p2[0] - r2[0]*p2[2]);
  a->angmom[2] += (r1[0]*p1[1] - r1[1]*p1[0]) + (r2[0]*p2[1] - r2[1]*p2[0]);
}

//----------------------------------------------------------------------------------------
//! \fn FindRoot
//! \brief union-find root with path compression.

int FindRoot(std::vector<int> *parent, int i) {
  while ((*parent)[i] != i) {
    (*parent)[i] = (*parent)[(*parent)[i]];
    i = (*parent)[i];
  }
  return i;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::MergeSinks
//! \brief link sinks closer than `merge_link_cells*dx_finest` and collapse each group.

void SinkParticles::MergeSinks(const SinkMeshGeom &g) {
  const int n = nsinks;
  if (n < 2) { return; }

  const Real link = merge_link_cells*g.dxf_max;
  const Real link2 = SQR(link);

  // friends-of-friends linking, strict `dist^2 < link^2` as in FofLib.cpp:715
  std::vector<int> parent(n);
  for (int i=0; i<n; ++i) { parent[i] = i; }
  for (int i=0; i<n; ++i) {
    for (int j=i+1; j<n; ++j) {
      if (SepSq(sinks[i].pos, sinks[j].pos, g) < link2) {
        const int ri = FindRoot(&parent, i);
        const int rj = FindRoot(&parent, j);
        if (ri != rj) { parent[std::max(ri, rj)] = std::min(ri, rj); }
      }
    }
  }

  // collect groups keyed by their lowest member index; both the group order and the
  // member order are then strictly ascending, hence identical on every rank
  std::vector<std::vector<int>> groups;
  std::vector<int> slot(n, -1);
  for (int i=0; i<n; ++i) {
    const int r = FindRoot(&parent, i);
    if (slot[r] < 0) {
      slot[r] = static_cast<int>(groups.size());
      groups.push_back(std::vector<int>());
    }
    groups[slot[r]].push_back(i);
  }

  std::vector<SinkData> merged;
  merged.reserve(sinks.size());

  for (const auto &grp : groups) {
    const int nm = static_cast<int>(grp.size());
    if (nm == 1) {
      merged.push_back(sinks[grp[0]]);
      continue;
    }

    // Members above mmergemax are "flagged" and never merge with each other.
    std::vector<int> flagged;
    if (mmergemax > 0.0) {
      for (int j=0; j<nm; ++j) {
        if (sinks[grp[j]].m > mmergemax) { flagged.push_back(j); }
      }
    }

    if (static_cast<int>(flagged.size()) > 1) {
      // Keep every flagged member, and fold each unflagged member into its nearest
      // flagged one (SinkParticleList.cpp:636-671).  The tie rule reproduces ORION2:
      // scan ascending and keep the LAST candidate within r2min/10000 of the minimum.
      const int nf = static_cast<int>(flagged.size());
      std::vector<SinkData> keep(nf);
      std::vector<SinkData> anchor(nf);  // pre-merge flagged state; the targets move
      for (int f=0; f<nf; ++f) {
        keep[f] = sinks[grp[flagged[f]]];
        anchor[f] = keep[f];
      }
      std::vector<bool> is_flagged(nm, false);
      for (int f=0; f<nf; ++f) { is_flagged[flagged[f]] = true; }

      for (int j=0; j<nm; ++j) {
        if (is_flagged[j]) { continue; }
        Real r2min = 1.0e50;
        for (int f=0; f<nf; ++f) {
          r2min = std::min(r2min, SepSq(sinks[grp[j]].pos, anchor[f].pos, g));
        }
        int target = 0;
        for (int f=0; f<nf; ++f) {
          const Real r2 = SepSq(sinks[grp[j]].pos, anchor[f].pos, g);
          // `<=`, not `<`: ORION2's tie window is RELATIVE, so at r2min == 0 -- two
          // exactly coincident sinks, which CreateSinks can produce because it has no
          // "is a sink already here" test -- the predicate read `r2 < 0` and was false
          // even for the minimizer, leaving `target` at its 0 initializer and folding the
          // sink into an arbitrary anchor rather than the nearest one.  For r2min > 0 the
          // minimizer already satisfied the strict form, so nothing else changes.
          if ((r2 - r2min) <= r2min/10000.0) { target = f; }
        }
        MergeInto(&keep[target], UnwrapAbout(sinks[grp[j]], keep[target].pos, g));
        WrapPosition(&keep[target], g);
      }
      for (int f=0; f<nf; ++f) { merged.push_back(keep[f]); }
      continue;
    }

    // Whole group collapses into one sink; the survivor inherits the id of the most
    // massive member (ties broken by the smaller id so all ranks agree).
    SinkData out = sinks[grp[0]];
    long int surviving_id = out.id;
    Real maxmass = out.m;
    for (int j=1; j<nm; ++j) {
      const SinkData &b = sinks[grp[j]];
      if (b.m > maxmass || (b.m == maxmass && b.id < surviving_id)) {
        maxmass = b.m;
        surviving_id = b.id;
      }
      MergeInto(&out, UnwrapAbout(b, out.pos, g));
    }
    out.id = surviving_id;
    WrapPosition(&out, g);
    merged.push_back(out);
  }

  if (merged.size() != sinks.size() && global_variable::my_rank == 0) {
    std::cout << "### Sink merge: " << sinks.size() << " -> " << merged.size()
              << " sinks" << std::endl;
  }
  sinks.swap(merged);
  SyncSinkCount();
}

}  // namespace sinkparticles
