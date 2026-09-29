#ifndef RECONSTRUCT_SPECIFIC_ENERGY_RECON_HPP_
#define RECONSTRUCT_SPECIFIC_ENERGY_RECON_HPP_

//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file specific_energy_recon.hpp
//! \brief Specific-energy face states for Newtonian hydro and MHD with a tabulated EOS.

#include "athena.hpp"
#include "eos/eos.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "reconstruct/thermal_floors.hpp"

namespace hydro_reconstruction {

// Limiting rho and internal-energy density independently does not bound u/rho:
// at the stellar surface the reconstructed density can approach vacuum while
// retaining finite energy, producing an artificial hot Riemann state. Limit the
// specific energy instead, with the run's own reconstruction on its own stencil,
// then restore energy density at the face. The flux update still evolves
// conservative total energy; no cell state is clipped. The magnetic face states
// are untouched: the Riemann solver assembles the face total energy from the
// corrected internal energy, so the MHD energy bookkeeping stays consistent.
// Call after the ordinary reconstruction, before the face sanitiser and the
// Riemann solve.  The thermal floor and ceiling are NOT applied here: every face this
// writes is floored with the same density fmax(rho_face, dfloor) before anything reads
// it -- IEN by the face sanitiser, which the pass requires, and the auxiliary by the
// dual-energy flux -- and the floor/ceiling map is idempotent, so applying it here too
// only cost a second table closure per face.
// The auxiliary thermal field has the same energy-density convention as IEN.
template<ReconstructionMethod recon, int axis>
KOKKOS_INLINE_FUNCTION
void SpecificEnergyFaceAt(const EOS_Data &eos, const int dual,
    const int m, const int mb, const int k, const int j, const int i,
    const DvceArray5D<Real> &w, const DvceArray5D<Real> &wl,
    const DvceArray5D<Real> &wr) {
  static_assert(axis >= 0 && axis < 3, "Invalid reconstruction direction");
  static_assert(recon != ReconstructionMethod::dc,
                "Donor cell carries the cell's own specific energy to the face");
  constexpr int di = axis == 0, dj = axis == 1, dk = axis == 2;
  constexpr int half = (recon == ReconstructionMethod::plm) ? 1 : 2;
  constexpr int kStencil = 2*half + 1;
  for (int side = 0; side < 2; ++side) {
    // side 0: the cell on the left of face i, whose right state lands in wl;
    // side 1: the cell on the right of face i, whose left state lands in wr.
    const int ci = i + (side-1)*di;
    const int cj = j + (side-1)*dj;
    const int ck = k + (side-1)*dk;
    Real rs[kStencil];
    for (int s = -half; s <= half; ++s) {
      rs[s+half] = fmax(w(m, IDN, ck+s*dk, cj+s*dj, ci+s*di), eos.dfloor);
    }
    const auto face = side == 0 ? wl : wr;
    const Real rho = fmax(face(mb, IDN, k, j, i), eos.dfloor);
    for (int channel = 0; channel < (dual >= 0 ? 2 : 1); ++channel) {
      const int n = channel == 0 ? IEN : dual;
      Real es[kStencil];
      for (int s = -half; s <= half; ++s) {
        es[s+half] = w(m, n, ck+s*dk, cj+s*dj, ci+s*di)/rs[s+half];
      }
      Real plus, minus;
      if constexpr (recon == ReconstructionMethod::plm) {
        PLM(es[0], es[1], es[2], plus, minus);
      } else if constexpr (recon == ReconstructionMethod::ppm4) {
        PPM4(es[0], es[1], es[2], es[3], es[4], plus, minus);
      } else if constexpr (recon == ReconstructionMethod::ppmx) {
        PPMX(es[0], es[1], es[2], es[3], es[4], plus, minus);
      } else {
        WENOZ(es[0], es[1], es[2], es[3], es[4], plus, minus);
      }
      face(mb, n, k, j, i) = rho*(side == 0 ? plus : minus);
    }
  }
}

// Host-side dispatch on the (grid-uniform) reconstruction method, launched over the
// faces the Riemann kernel consumes.  Work index a selects the MeshBlock through
// active_indices under LAT and lands in chunk buffer slot a - abuf0, exactly as the
// reconstruction dispatchers in recon.hpp do.
template<int ivx>
inline void SpecificEnergyFaceDispatch(ReconstructionMethod recon, const char *name,
    const int alo, const int ahi, const int abuf0,
    const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
    const EOS_Data &eos, const int dual, const bool lat_enabled,
    const DvceArray1D<int> &active_indices, const DvceArray5D<Real> &w,
    const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr) {
  constexpr int axis = ivx - IVX;
  switch (recon) {
    case ReconstructionMethod::plm:
      par_for(name, DevExeSpace(), alo, ahi, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int a, int k, int j, int i) {
          const int m = lat_enabled ? active_indices(a) : a;
          SpecificEnergyFaceAt<ReconstructionMethod::plm, axis>(
              eos, dual, m, a - abuf0, k, j, i, w, wl, wr);
        });
      break;
    case ReconstructionMethod::ppm4:
      par_for(name, DevExeSpace(), alo, ahi, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int a, int k, int j, int i) {
          const int m = lat_enabled ? active_indices(a) : a;
          SpecificEnergyFaceAt<ReconstructionMethod::ppm4, axis>(
              eos, dual, m, a - abuf0, k, j, i, w, wl, wr);
        });
      break;
    case ReconstructionMethod::ppmx:
      par_for(name, DevExeSpace(), alo, ahi, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int a, int k, int j, int i) {
          const int m = lat_enabled ? active_indices(a) : a;
          SpecificEnergyFaceAt<ReconstructionMethod::ppmx, axis>(
              eos, dual, m, a - abuf0, k, j, i, w, wl, wr);
        });
      break;
    case ReconstructionMethod::wenoz:
      par_for(name, DevExeSpace(), alo, ahi, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int a, int k, int j, int i) {
          const int m = lat_enabled ? active_indices(a) : a;
          SpecificEnergyFaceAt<ReconstructionMethod::wenoz, axis>(
              eos, dual, m, a - abuf0, k, j, i, w, wl, wr);
        });
      break;
    case ReconstructionMethod::dc:
    default:
      break;
  }
}

}  // namespace hydro_reconstruction
#endif  // RECONSTRUCT_SPECIFIC_ENERGY_RECON_HPP_
