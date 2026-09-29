#ifndef DYN_GRMHD_DYN_GRMHD_COORD_TERMS_IMPL_HPP_
#define DYN_GRMHD_DYN_GRMHD_COORD_TERMS_IMPL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_coord_terms_impl.hpp
//! \brief DynGRMHDPS::AddCoordTermsEOS<NGHOST>, the coordinate-source kernels.  Included
//! only by dyn_grmhd_coord_terms_ideal.cpp and the dyn_grmhd_ps_<eos>.cpp units of the
//! other EOS policies, each of which instantiates it (INSTANTIATE_COORD_TERMS) for its
//! policies, so each kernel is compiled in exactly one unit.

#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "z4c/tmunu.hpp"
#include "dyn_grmhd.hpp"
#include "tasklist/numerical_relativity.hpp"

#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

template<class EOSPolicy, class ErrorPolicy> template<int NGHOST>
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS(
    const DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &bcc,
    const Real dt, DvceArray5D<Real> &rhs) {
  if (fixed_evolution) {
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  const int nwork = pmy_pack->nmb_thispack;
  if (nwork <= 0) return;

  const auto metric = pmy_pack->padm->GetMetricView();
  //auto &tmunu = pmy_pack->ptmunu->tmunu;

  // fetch flag for smooth excision and
  // excision mask, and target values
  bool smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
  auto &floor = pmy_pack->pcoord->excision_floor;
  Real &dexcise = pmy_pack->pcoord->coord_data.dexcise;
  // Real &pexcise = pmy_pack->pcoord->coord_data.pexcise;
  Real &texcise = pmy_pack->pcoord->coord_data.texcise;
  Real &tdamp = pmy_pack->pcoord->coord_data.tdamp;

  int &nhyd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;

  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  auto solver_template = eos;
  const int imap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  // Launch bounds for the coordinate-source kernel.  The forward-mode AD metric
  // derivatives want ~255 registers, and capping at 128 (<256,2>, 16 warps/SM) buys that
  // occupancy with 2.3 kB/thread of spill; <256,1> keeps them in registers at 8 warps/SM
  // and 2.4x fewer local-memory instructions, which is the better trade for this kernel.
  constexpr int kCoordSrcMaxThreads = 256;
  constexpr int kCoordSrcMinBlocksPerSM = 1;
  par_for<kCoordSrcMaxThreads, kCoordSrcMinBlocksPerSM>(
      "coord_src", DevExeSpace(), 0, nwork-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = a;
    const Real block_dt = dt;
    // Extract the metric and coordinate quantities.
    adm::ADMMetricPoint metric_point{};
    adm::ADMMetricDerivatives metric_derivs{};
    metric.template CellMetricAndDerivatives<NGHOST>(
        m, k, j, i, metric_point, metric_derivs);
    Real g3d[NSPMETRIC];
    for (int n = 0; n < NSPMETRIC; ++n) g3d[n] = metric_point.g_dd[n];
    const Real alpha = metric_point.alpha;
    Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13],
                                g3d[S22], g3d[S23], g3d[S33]);
    Real vol = sqrt(detg);
    Real g3u[NSPMETRIC] = {0.};
    adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                    &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

    // Fluid quantities
    Real prim_pt[NPRIM] = {0.0};
    prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
    prim_pt[PVX] = prim(m, IVX, k, j, i);
    prim_pt[PVY] = prim(m, IVY, k, j, i);
    prim_pt[PVZ] = prim(m, IVZ, k, j, i);
    for (int s = 0; s < nscal; s++) {
      prim_pt[PYF + s] = prim(m, nhyd + s, k, j, i);
    }
    prim_pt[PPR] = prim(m, IPR, k, j, i);
    const auto &eos_cell = solver_template.ps.GetEOS();
    prim_pt[PTM] = eos_cell.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR],
                                                &prim_pt[PYF]);

    // Get the conserved variables. Note that we don't use PrimitiveSolver here --
    // that's because we would need to recalculate quantities used in E and S_d in order
    // to get S_dd.
    Real H =
      prim(m, IDN, k, j, i)*eos_cell.GetEnthalpy(prim_pt[PRH], prim_pt[PTM],
                                                 &prim_pt[PYF]);
    Real usq = Primitive::SquareVector(&prim_pt[PVX], g3d);
    Real const Wsq = 1.0 + usq;
    Real const W = sqrt(Wsq);
    Real B_u[NMAG] = {bcc(m, IBX, k, j, i)/vol,
                      bcc(m, IBY, k, j, i)/vol,
                      bcc(m, IBZ, k, j, i)/vol};
    Real Bv = 0.0;
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bv += g3d[imap[a][b]]*prim_pt[PVX + a]*B_u[b];
      }
    }
    Real Bsq = Primitive::SquareVector(B_u, g3d);
    Bv = Bv/W;
    Real bsq = Bv*Bv + Bsq/Wsq;

    Real E = (H*Wsq + Bsq) - prim_pt[PPR] - 0.5*bsq;
    //Real E = tmunu.E(m,k,j,i);

    Real S_d[3] = {0.0};
    for (int a = 0; a < 3; a++) {
      //S_d[a] = tmunu.S_d(m,a,k,j,i);
      for (int b = 0; b < 3; b++) {
        S_d[a] += ((H*Wsq + Bsq)*prim_pt[PVX + b]/W - Bv*B_u[b])*g3d[imap[a][b]];
      }
    }

    Real S_uu[3][3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b <= a; b++) {
        S_uu[a][b] = (H + Bsq/Wsq)*prim_pt[PVX + a]*prim_pt[PVX + b]
                      - B_u[a]*B_u[b]/Wsq
                      - Bv*(B_u[a]*prim_pt[PVX + b] + B_u[b]*prim_pt[PVX + a])/W
                      + (prim_pt[PPR] + 0.5*bsq)*g3u[imap[a][b]];
        /*S_uu[a][b] = 0.0;
        for (int c = 0; c < 3; c++) {
          for (int d = 0; d < 3; d++) {
            S_uu[a][b] += tmunu.S_dd(m,c,d,k,j,i)*g3u[imap[a][c]]*g3u[imap[b][d]];
          }
        }*/
        S_uu[b][a] = S_uu[a][b];
      }
    }

    // Assemble energy RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        rhs(m, IEN, k, j, i) += block_dt*vol*(
            alpha*metric_point.K_dd[imap[a][b]]*S_uu[a][b] -
            g3u[imap[a][b]]*S_d[a]*metric_derivs.dalpha_d[b]);
      }
    }

    // Assemble momentum RHS
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
          rhs(m,IM1+a, k, j, i) += 0.5*block_dt*alpha*vol*S_uu[b][c]*
              metric_derivs.dg_ddd[a][imap[b][c]];
        }
        rhs(m, IM1+a, k, j, i) +=
            block_dt*vol*S_d[b]*metric_derivs.dbeta_du[a][b];
      }
      rhs(m, IM1+a, k, j, i) -= block_dt*vol*E*metric_derivs.dalpha_d[a];
    }

    // Assemble damping source terms
    if (smoothing) {
      // D = rho*W and tau = E-D are needed for the smooth damping terms
      // inside excised regions, if existing.
      Real D = prim(m, IDN, k, j, i) * W;
      Real tau = E - D;

      // Compute the excised value for the energy.
      // Real tau_ex = (dexcise*eos_.GetEnthalpy(dexcise/mb, texcise, &prim_pt[PYF]))
      //               + Bsq - pexcise - 0.5*bsq - dexcise;
      Real tau_ex = eos_cell.GetEnergy(dexcise/mb, texcise, &prim_pt[PYF])
                    + 0.5*Bsq - dexcise;

      rhs(m, IDN, k, j, i) -= (block_dt*vol*floor(m,k,j,i)*(D-dexcise))/tdamp;
      rhs(m, IM1, k, j, i) -= (block_dt*floor(m,k,j,i)*vol*S_d[0])/tdamp;
      rhs(m, IM2, k, j, i) -= (block_dt*floor(m,k,j,i)*vol*S_d[1])/tdamp;
      rhs(m, IM3, k, j, i) -= (block_dt*floor(m,k,j,i)*vol*S_d[2])/tdamp;
      rhs(m, IEN, k, j, i) -= block_dt*vol*floor(m,k,j,i)*(tau - tau_ex)/tdamp;
      for (int s = 0; s < nscal; s++) {
        rhs(m, IYF+s, k, j, i) -=
            (block_dt*vol*floor(m,k,j,i)*(D-dexcise)*prim_pt[PYF+s])/tdamp;
      }
    }
  });
}

// Macro for defining CoordTerms templates
#define INSTANTIATE_COORD_TERMS(EOSPolicy, ErrorPolicy) \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<2>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<3>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs); \
template \
void DynGRMHDPS<EOSPolicy, ErrorPolicy>::AddCoordTermsEOS<4>( \
      const DvceArray5D<Real> &prim, \
      const DvceArray5D<Real> &bcc, const Real dt, DvceArray5D<Real> &rhs);

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_COORD_TERMS_IMPL_HPP_
