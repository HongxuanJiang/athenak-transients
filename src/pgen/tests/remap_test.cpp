//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_test.cpp
//! \brief validation pgen for the universal restart-remap module (src/remap/).
//!
//! Two modes, selected by <problem> remap:
//!  - source mode (remap = false): initializes a smooth analytic 3D state.  Hydro-only
//!    or MHD (chosen by the input's <hydro>/<mhd> block); the magnetic field is set from
//!    an analytic vector potential with the canonical edge evaluation (including the
//!    fine-neighbor half-edge averaging), so uniform, SMR, or AMR sources are valid.
//!  - remap mode (remap = true): calls remap::LoadAndApplyRemap on remap_source.
//!
//! Both modes finish by printing a parseable error-norm line against the analytic state,
//!   remap_test_errors: divb_max= b_l1= b_max= rho_l1= eint_l1= b_far_max=
//! where the B reference is the discrete curl of the analytic A on the local grid and
//! divb_max is the raw face-difference divergence (same stencil as the mhd_divb output).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "remap/remap.hpp"
#include "pgen/pgen.hpp"

namespace {

// smooth localized analytic state on the source box [-1,1]^3 (literal constants inline:
// namespace-scope constexpr Real is not device-visible without relaxed-constexpr)
KOKKOS_INLINE_FUNCTION Real RTGauss(const Real x, const Real y, const Real z) {
  return exp(-(x*x + y*y + z*z) / 0.1225);  // L = 0.35
}
KOKKOS_INLINE_FUNCTION Real RTRho(const Real x, const Real y, const Real z) {
  return 1.0 + 0.5 * RTGauss(x, y, z);
}
KOKKOS_INLINE_FUNCTION Real RTPgas(const Real x, const Real y, const Real z) {
  return 0.6 + 0.3 * RTGauss(x, y, z);
}
KOKKOS_INLINE_FUNCTION Real RTVx(const Real x, const Real y, const Real z) {
  return 0.10 * RTGauss(x, y, z);
}
KOKKOS_INLINE_FUNCTION Real RTVy(const Real x, const Real y, const Real z) {
  return -0.07 * RTGauss(x, y, z);
}
KOKKOS_INLINE_FUNCTION Real RTVz(const Real x, const Real y, const Real z) {
  return 0.04 * RTGauss(x, y, z);
}
// vector potential (A1 = 0); fully 3D divergence-free B with all components nonzero
KOKKOS_INLINE_FUNCTION Real RTA2(const Real x, const Real y, const Real z) {
  return 0.1 * RTGauss(x, y, z) * (1.0 + 0.3 * sin(1.5 * x));
}
KOKKOS_INLINE_FUNCTION Real RTA3(const Real x, const Real y, const Real z) {
  return 0.1 * RTGauss(x, y, z) * (1.0 + 0.3 * cos(1.2 * y));
}

}  // namespace

namespace {

// error norms vs the analytic state (host, active zones); called directly in source mode
// and from the user_remap_post_func hook in remap mode
void RTReportErrors(MeshBlockPack *pmbp) {
  const bool is_mhd = (pmbp->pmhd != nullptr);
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const Real gm1 = (is_mhd ? pmbp->pmhd->peos->eos_data.gamma
                           : pmbp->phydro->peos->eos_data.gamma) - 1.0;

  Real divb_max = 0.0, b_max = 0.0, b_far_max = 0.0;
  Real b_l1 = 0.0, rho_l1 = 0.0, eint_l1 = 0.0, vol = 0.0;
  {
    DvceArray5D<Real>::HostMirror u_h;
    if (is_mhd) {
      u_h = Kokkos::create_mirror_view(pmbp->pmhd->u0);
      Kokkos::deep_copy(u_h, pmbp->pmhd->u0);
    } else {
      u_h = Kokkos::create_mirror_view(pmbp->phydro->u0);
      Kokkos::deep_copy(u_h, pmbp->phydro->u0);
    }
    DvceArray4D<Real>::HostMirror x1f_h, x2f_h, x3f_h;
    if (is_mhd) {
      x1f_h = Kokkos::create_mirror_view(pmbp->pmhd->b0.x1f);
      x2f_h = Kokkos::create_mirror_view(pmbp->pmhd->b0.x2f);
      x3f_h = Kokkos::create_mirror_view(pmbp->pmhd->b0.x3f);
      Kokkos::deep_copy(x1f_h, pmbp->pmhd->b0.x1f);
      Kokkos::deep_copy(x2f_h, pmbp->pmhd->b0.x2f);
      Kokkos::deep_copy(x3f_h, pmbp->pmhd->b0.x3f);
    }

    for (int m = 0; m < nmb; ++m) {
      const Real x1min = size.h_view(m).x1min, x1max = size.h_view(m).x1max;
      const Real x2min = size.h_view(m).x2min, x2max = size.h_view(m).x2max;
      const Real x3min = size.h_view(m).x3min, x3max = size.h_view(m).x3max;
      const Real dx1 = size.h_view(m).dx1, dx2 = size.h_view(m).dx2;
      const Real dx3 = size.h_view(m).dx3;
      const Real dv = dx1 * dx2 * dx3;
      for (int k = ks; k <= ke; ++k) {
        const Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        for (int j = js; j <= je; ++j) {
          const Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
          for (int i = is; i <= ie; ++i) {
            const Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            vol += dv;
            rho_l1 += dv * std::abs(u_h(m,IDN,k,j,i) - RTRho(x1v, x2v, x3v));

            Real emag = 0.0;
            if (is_mhd) {
              const Real divb = (x1f_h(m,k,j,i+1) - x1f_h(m,k,j,i))/dx1
                              + (x2f_h(m,k,j+1,i) - x2f_h(m,k,j,i))/dx2
                              + (x3f_h(m,k+1,j,i) - x3f_h(m,k,j,i))/dx3;
              divb_max = std::max(divb_max, std::abs(divb));

              // reference faces: discrete curl of the analytic A on this grid
              auto refx1 = [&](const Real xf) {
                const Real x2f = x2min + static_cast<Real>(j-js)*dx2;
                const Real x3f = x3min + static_cast<Real>(k-ks)*dx3;
                return (RTA3(xf, x2f+dx2, x3v) - RTA3(xf, x2f, x3v))/dx2 -
                       (RTA2(xf, x2v, x3f+dx3) - RTA2(xf, x2v, x3f))/dx3;
              };
              auto refx2 = [&](const Real yf) {
                const Real x1f = x1min + static_cast<Real>(i-is)*dx1;
                return -(RTA3(x1f+dx1, yf, x3v) - RTA3(x1f, yf, x3v))/dx1;
              };
              auto refx3 = [&](const Real zf) {
                const Real x1f = x1min + static_cast<Real>(i-is)*dx1;
                return (RTA2(x1f+dx1, x2v, zf) - RTA2(x1f, x2v, zf))/dx1;
              };
              const Real x1fl = x1min + static_cast<Real>(i-is)*dx1;
              const Real x2fl = x2min + static_cast<Real>(j-js)*dx2;
              const Real x3fl = x3min + static_cast<Real>(k-ks)*dx3;
              const Real e1 = std::abs(x1f_h(m,k,j,i) - refx1(x1fl));
              const Real e2 = std::abs(x2f_h(m,k,j,i) - refx2(x2fl));
              const Real e3 = std::abs(x3f_h(m,k,j,i) - refx3(x3fl));
              b_l1 += dv * (e1 + e2 + e3) / 3.0;
              b_max = std::max(b_max, std::max(e1, std::max(e2, e3)));

              // field amplitude far outside the analytic source box [-1,1]^3 (+ taper):
              // must be ~0 after a domain-widening remap (gauge/taper regression gate)
              const Real rinf = std::max(std::abs(x1v),
                                         std::max(std::abs(x2v), std::abs(x3v)));
              if (rinf > 1.35) {
                const Real bmag = std::max(std::abs(x1f_h(m,k,j,i)),
                                           std::max(std::abs(x2f_h(m,k,j,i)),
                                                    std::abs(x3f_h(m,k,j,i))));
                b_far_max = std::max(b_far_max, bmag);
              }

              const Real bx = 0.5*(x1f_h(m,k,j,i) + x1f_h(m,k,j,i+1));
              const Real by = 0.5*(x2f_h(m,k,j,i) + x2f_h(m,k,j+1,i));
              const Real bz = 0.5*(x3f_h(m,k,j,i) + x3f_h(m,k+1,j,i));
              emag = 0.5*(bx*bx + by*by + bz*bz);
            }

            const Real rho = std::max(u_h(m,IDN,k,j,i), static_cast<Real>(1.0e-30));
            const Real ke_cell = 0.5*(SQR(u_h(m,IM1,k,j,i)) + SQR(u_h(m,IM2,k,j,i))
                                      + SQR(u_h(m,IM3,k,j,i))) / rho;
            const Real eint = u_h(m,IEN,k,j,i) - ke_cell - emag;
            eint_l1 += dv * std::abs(eint - RTPgas(x1v, x2v, x3v)/gm1);
          }
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  Real sums[3] = {b_l1, rho_l1, eint_l1};
  Real sums_g[3];
  MPI_Allreduce(sums, sums_g, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  b_l1 = sums_g[0]; rho_l1 = sums_g[1]; eint_l1 = sums_g[2];
  Real vg;
  MPI_Allreduce(&vol, &vg, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  vol = vg;
  Real maxs[3] = {divb_max, b_max, b_far_max};
  Real maxs_g[3];
  MPI_Allreduce(maxs, maxs_g, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  divb_max = maxs_g[0]; b_max = maxs_g[1]; b_far_max = maxs_g[2];
#endif

  if (global_variable::my_rank == 0 && vol > 0.0) {
    std::printf("remap_test_errors: divb_max=%.15e b_l1=%.15e b_max=%.15e "
                "rho_l1=%.15e eint_l1=%.15e b_far_max=%.15e\n",
                divb_max, b_l1/vol, b_max, rho_l1/vol, eint_l1/vol, b_far_max);
    std::cout << std::flush;
  }
  return;
}

}  // namespace

void ProblemGenerator::RemapTest(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (restart) return;

  const bool is_mhd = (pmbp->pmhd != nullptr);
  if (pmbp->phydro == nullptr && !is_mhd) {
    std::cout << "### FATAL ERROR in remap_test" << std::endl
              << "remap_test requires <hydro> or <mhd>." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR in remap_test" << std::endl
              << "remap_test requires a 3D mesh." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // remap mode: the <remap> block drives remap::MaybeAutoRemap right after this
  // function returns; enroll the error report as the post hook and skip the init.
  if (remap::IsAutoRemapEnabled(pin)) {
    MeshBlockPack *pmbp_cap = pmbp;
    user_remap_post_func = [pmbp_cap](const remap::RemapSummary &) {
      RTReportErrors(pmbp_cap);
    };
    return;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const Real gm1 = (is_mhd ? pmbp->pmhd->peos->eos_data.gamma
                           : pmbp->phydro->peos->eos_data.gamma) - 1.0;

  {
    // ---- analytic source initialization ----------------------------------------------
    auto size_d = size.d_view;
    if (is_mhd) {
      auto &b0 = pmbp->pmhd->b0;
      auto &bcc0 = pmbp->pmhd->bcc0;
      auto &u0 = pmbp->pmhd->u0;
      const int ncells1 = nx1 + 2*indcs.ng;
      const int ncells2 = nx2 + 2*indcs.ng;
      const int ncells3 = nx3 + 2*indcs.ng;
      DvceArray4D<Real> a2, a3;
      Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);
      auto &nghbr = pmbp->pmb->nghbr;
      auto &mblev = pmbp->pmb->mb_lev;

      par_for("rt_vecpot", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x1f = LeftEdgeX(i-is, nx1, size_d(m).x1min, size_d(m).x1max);
        const Real x2f = LeftEdgeX(j-js, nx2, size_d(m).x2min, size_d(m).x2max);
        const Real x3f = LeftEdgeX(k-ks, nx3, size_d(m).x3min, size_d(m).x3max);
        const Real x2v = CellCenterX(j-js, nx2, size_d(m).x2min, size_d(m).x2max);
        const Real x3v = CellCenterX(k-ks, nx3, size_d(m).x3min, size_d(m).x3max);
        const Real dx2 = size_d(m).dx2;
        const Real dx3 = size_d(m).dx3;

        a2(m,k,j,i) = RTA2(x1f, x2v, x3f);
        a3(m,k,j,i) = RTA3(x1f, x2f, x3v);

        // fine-neighbor half-edge averaging (canonical SMR/AMR contract; cf. gr_torus)
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
            (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
            (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
            (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
            (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
            (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
            (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
            (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
          a2(m,k,j,i) = 0.5*(RTA2(x1f, x2v-0.25*dx2, x3f) +
                             RTA2(x1f, x2v+0.25*dx2, x3f));
        }
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
            (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
          a3(m,k,j,i) = 0.5*(RTA3(x1f, x2f, x3v-0.25*dx3) +
                             RTA3(x1f, x2f, x3v+0.25*dx3));
        }
      });

      par_for("rt_b0", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real dx1 = size_d(m).dx1;
        const Real dx2 = size_d(m).dx2;
        const Real dx3 = size_d(m).dx3;
        b0.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                          (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3;
        b0.x2f(m,k,j,i) = -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
        b0.x3f(m,k,j,i) = (a2(m,k,j,i+1) - a2(m,k,j,i))/dx1;
        if (i==ie) {
          b0.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                              (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3;
        }
        if (j==je) {
          b0.x2f(m,k,j+1,i) = -(a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
        }
        if (k==ke) {
          b0.x3f(m,k+1,j,i) = (a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1;
        }
      });

      par_for("rt_u0_mhd", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x1v = CellCenterX(i-is, nx1, size_d(m).x1min, size_d(m).x1max);
        const Real x2v = CellCenterX(j-js, nx2, size_d(m).x2min, size_d(m).x2max);
        const Real x3v = CellCenterX(k-ks, nx3, size_d(m).x3min, size_d(m).x3max);
        const Real rho = RTRho(x1v, x2v, x3v);
        const Real pg = RTPgas(x1v, x2v, x3v);
        const Real vx = RTVx(x1v, x2v, x3v);
        const Real vy = RTVy(x1v, x2v, x3v);
        const Real vz = RTVz(x1v, x2v, x3v);
        const Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
        const Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
        const Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
        bcc0(m,IBX,k,j,i) = bx;
        bcc0(m,IBY,k,j,i) = by;
        bcc0(m,IBZ,k,j,i) = bz;
        u0(m,IDN,k,j,i) = rho;
        u0(m,IM1,k,j,i) = rho*vx;
        u0(m,IM2,k,j,i) = rho*vy;
        u0(m,IM3,k,j,i) = rho*vz;
        u0(m,IEN,k,j,i) = pg/gm1 + 0.5*rho*(vx*vx + vy*vy + vz*vz)
                        + 0.5*(bx*bx + by*by + bz*bz);
      });
      if (pmbp->pmhd->use_dual_energy) {
        const int didx = pmbp->pmhd->dual_energy_idx;
        par_for("rt_dual_mhd", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          const Real x1v = CellCenterX(i-is, nx1, size_d(m).x1min, size_d(m).x1max);
          const Real x2v = CellCenterX(j-js, nx2, size_d(m).x2min, size_d(m).x2max);
          const Real x3v = CellCenterX(k-ks, nx3, size_d(m).x3min, size_d(m).x3max);
          u0(m,didx,k,j,i) = RTPgas(x1v, x2v, x3v)/gm1;
        });
      }
    } else {
      auto &u0 = pmbp->phydro->u0;
      const bool use_dual = pmbp->phydro->use_dual_energy;
      const int didx = use_dual ? pmbp->phydro->dual_energy_idx : -1;
      par_for("rt_u0_hyd", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real x1v = CellCenterX(i-is, nx1, size_d(m).x1min, size_d(m).x1max);
        const Real x2v = CellCenterX(j-js, nx2, size_d(m).x2min, size_d(m).x2max);
        const Real x3v = CellCenterX(k-ks, nx3, size_d(m).x3min, size_d(m).x3max);
        const Real rho = RTRho(x1v, x2v, x3v);
        const Real pg = RTPgas(x1v, x2v, x3v);
        const Real vx = RTVx(x1v, x2v, x3v);
        const Real vy = RTVy(x1v, x2v, x3v);
        const Real vz = RTVz(x1v, x2v, x3v);
        u0(m,IDN,k,j,i) = rho;
        u0(m,IM1,k,j,i) = rho*vx;
        u0(m,IM2,k,j,i) = rho*vy;
        u0(m,IM3,k,j,i) = rho*vz;
        u0(m,IEN,k,j,i) = pg/gm1 + 0.5*rho*(vx*vx + vy*vy + vz*vz);
        if (didx >= 0) {
          u0(m,didx,k,j,i) = pg/gm1;
        }
      });
    }

  }


  RTReportErrors(pmbp);
  return;
}
