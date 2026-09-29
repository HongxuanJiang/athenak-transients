//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fofc.cpp
//! \brief Implements functions for first-order flux correction (FOFC) algorithm.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "hydro/rsolvers/llf_hyd_singlestate.hpp"
#include "hydro.hpp"

namespace hydro {

KOKKOS_INLINE_FUNCTION
void SetDualEnergyFOFCFlux(const EOS_Data &eos, const Real mass_flux,
                           const Real dens_l, const Real dens_r,
                           const Real eint_l, const Real eint_r,
                           Real &dual_flux, Real &face_velocity) {
  const bool use_left = (mass_flux >= 0.0);
  const Real dens_upwind = fmax(use_left ? dens_l : dens_r, eos.dfloor);
  const Real eint_upwind = use_left ? eint_l : eint_r;
  dual_flux = mass_flux*(eint_upwind/dens_upwind);
  face_velocity = mass_flux/dens_upwind;
}

//----------------------------------------------------------------------------------------
//! \brief True when a cell centre of this MeshBlock -- including the one-cell halo the
//! FOFC stencil visits -- can lie within sqrt(r2) of the excision centre.

inline bool BlockMayReachExcisionSphere(const RegionSize &size, const Real x,
                                        const Real y, const Real z, const Real r2) {
  const Real xmin = size.x1min - 0.5*size.dx1;
  const Real xmax = size.x1max + 0.5*size.dx1;
  const Real ymin = size.x2min - 0.5*size.dx2;
  const Real ymax = size.x2max + 0.5*size.dx2;
  const Real zmin = size.x3min - 0.5*size.dx3;
  const Real zmax = size.x3max + 0.5*size.dx3;

  Real dx = 0.0;
  if (x < xmin) {
    dx = xmin - x;
  } else if (x > xmax) {
    dx = x - xmax;
  }

  Real dy = 0.0;
  if (y < ymin) {
    dy = ymin - y;
  } else if (y > ymax) {
    dy = y - ymax;
  }

  Real dz = 0.0;
  if (z < zmin) {
    dz = zmin - z;
  } else if (z > zmax) {
    dz = z - zmax;
  }

  return (SQR(dx) + SQR(dy) + SQR(dz)) <= r2;
}

//----------------------------------------------------------------------------------------
//! \brief The FOFC trial state: the stage update of the conserved variables built from
//! the current face fluxes, on the cells [kl,ku]x[jl,ju]x[il,iu] of the active blocks.

void Hydro::BuildFOFCTrial(Driver *pdriver, int stage, int il, int iu, int jl, int ju,
                           int kl, int ku) {
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : pmy_pack->nmb_thispack;
  if (nwork <= 0) return;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  const Real beta_stage = pdriver->beta[stage-1];
  const Real beta_dt = beta_stage*pmy_pack->pmesh->dt;

  int &nvars_ = nvars;
  auto &u0_ = u0;
  auto &u1_ = u1;
  auto &utest_ = utest;

  par_for("FOFC-newu", DevExeSpace(), 0, nwork-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_beta_dt =
        lat_per_block_dt ? beta_stage*lat_step_dt(m) : beta_dt;
    Real dtodx1 = block_beta_dt/size.d_view(m).dx1;
    Real dtodx2 = block_beta_dt/size.d_view(m).dx2;
    Real dtodx3 = block_beta_dt/size.d_view(m).dx3;

    // Estimate conserved variables
    for (int n=0; n<nvars_; ++n) {
      Real divf = dtodx1*(flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i));
      if (multi_d) {
        divf += dtodx2*(flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i));
      }
      if (three_d) {
        divf += dtodx3*(flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i));
      }
      utest_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - divf;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \brief Test the edge cells of the active blocks on the fluxes before the pending fine
//! flux estimate is added, and stash their flags (MeshBoundaryValuesCC::
//! StashPendingEdgeFOFCFlags has the reason).  Called by Hydro::Fluxes just before
//! MeshBoundaryValuesCC::AddPendingFineFluxMismatchCC; the event log counts the stage's
//! own test only.

void Hydro::StashFOFCEdgeFlags(Driver *pdriver, int stage) {
  if (pmy_pack->lat_nactive_thispack <= 0 || !(pmy_pack->pmesh->multi_d)) return;
  const int nfofc_before = pmy_pack->pmesh->ecounter.nfofc;
  MeshBoundaryValuesCC::ForEachMeshBlockEdge(pmy_pack->pmesh->mb_indcs,
      pmy_pack->pmesh->three_d, [&](int il, int iu, int jl, int ju, int kl, int ku) {
    BuildFOFCTrial(pdriver, stage, il, iu, jl, ju, kl, ku);
    peos->ConsToPrim(utest, w0, true, il, iu, jl, ju, kl, ku);
  });
  pmy_pack->pmesh->ecounter.nfofc = nfofc_before;
  pbval_u->StashPendingEdgeFOFCFlags(fofc);
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::FOFC
//! \brief Implements first-order flux-correction (FOFC) algorithm for Hydro.  First an
//! estimate of the updated conserved variables is made. This estimate is then used to
//! flag any cell where floors will be required during the conversion to primitives. Then
//! the fluxes on the faces of flagged cells are replaced with first-order LLF fluxes.
//! Often this is enough to prevent floors from being needed. The FOFC infrastructure is
//! also exploited for sink excision. Cells in the local excision neighborhood can
//! trigger FOFC directly without requiring an updated-state estimate first.

void Hydro::FOFC(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb = pmy_pack->nmb_thispack;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &size = pmy_pack->pmb->mb_size;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  Real excise_time = pmy_pack->pmesh->time;
  if (pmy_pack->pmesh->dt > 0.0) {
    const Real stage_time = pdriver->hydro_lat ?
        pdriver->stage_time_frac[stage-1] : pdriver->gam0[stage-1];
    excise_time += stage_time * pmy_pack->pmesh->dt;
  }
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(excise_time, excise_enabled, excise_radius,
      excise_density, excise_eint, sink_x, sink_y, sink_z);
  Real excise_r = excise_radius;

  if (use_fofc) {
    // Index bounds
    int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) { jl = js-1, ju = je+1; }
    if (three_d) { kl = ks-1, ku = ke+1; }

    // Estimate updated conserved variables
    BuildFOFCTrial(pdriver, stage, il, iu, jl, ju, kl, ku);

    // Test whether conversion to primitives requires floors
    // Note b0 and w0 passed to function, but not used/changed.
    peos->ConsToPrim(utest, w0, true, il, iu, jl, ju, kl, ku);

    // The edge cells whose flags were decided without the pending fine flux estimate
    // (StashFOFCEdgeFlags) take them back.
    pbval_u->ReconcilePendingEdgeFOFCFlags(fofc, nullptr, 0, uflx,
                                           dual_energy_pdv ? &dual_vf : nullptr);
  }

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &is_sr = pmy_pack->pcoord->is_special_relativistic;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto &eos = peos->eos_data;
  auto &use_fofc_ = use_fofc;
  auto &fofc_ = fofc;
  int &nhyd_ = nhydro;
  int &nvars_ = nvars;
  // The GR adiabat rides the first-order scalar loop below; only the non-relativistic
  // auxiliary has its own flux and face velocity.
  const bool dual_enabled = dual_energy_pdv;
  int &dual_idx_ = dual_energy_idx;
  auto &vf1_ = dual_vf.x1f;
  auto &vf2_ = dual_vf.x2f;
  auto &vf3_ = dual_vf.x3f;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &w0_ = w0;

  // Index bounds
  int il = is-1, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (multi_d) { jl = js-1, ju = je+1; }
  if (three_d) { kl = ks-1, ku = ke+1; }

  // With FOFC off the kernel below is entered only for the sink excision, and each cell
  // then works only inside excise_r + half a cell diagonal of the sink.  Restrict the
  // launch to the blocks that can reach that buffer: every cell dropped this way would
  // have failed the same distance test, so the fluxes are unchanged.
  auto flx_indices = active_indices;
  bool flx_indexed = lat_enabled;
  int flx_nwork = nwork;
  if (!use_fofc && !(is_gr && use_excise)) {
    if (!excise_enabled) return;
    auto &active_dv = pmy_pack->lat_active_indices;
    if (lat_enabled) active_dv.template sync<HostMemSpace>();
    int nsink = 0;
    for (int a = 0; a < nwork; ++a) {
      const int m = lat_enabled ? active_dv.h_view(a) : a;
      const Real half_diag = 0.5*std::sqrt(SQR(size.h_view(m).dx1)
                           + (multi_d ? SQR(size.h_view(m).dx2) : 0.0)
                           + (three_d ? SQR(size.h_view(m).dx3) : 0.0));
      const Real buffer_r = excise_r + half_diag;
      if (BlockMayReachExcisionSphere(size.h_view(m), sink_x, sink_y, sink_z,
                                      buffer_r*buffer_r)) {
        sink_block_indices.h_view(nsink++) = m;
      }
    }
    if (nsink == 0) return;
    // Stream-ordered upload: DualView::sync would fence the whole device queue here.
    Kokkos::deep_copy(DevExeSpace(), sink_block_indices.d_view,
                      sink_block_indices.h_view);
    sink_block_indices.clear_sync_state();
    flx_indices = sink_block_indices.d_view;
    flx_indexed = true;
    flx_nwork = nsink;
  }

  // Now replace fluxes with first-order LLF fluxes for any cell where floors needed (if
  // using FOFC) and/or for any cell about the excision (if GR+excising)
  par_for("FOFC-flx", DevExeSpace(), 0, flx_nwork-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = flx_indexed ? flx_indices(a) : a;
    // Check for FOFC flag
    bool fofc_flag = false;
    if (use_fofc_) { fofc_flag = fofc_(m,k,j,i); }

    // Check for GR + excision
    bool fofc_excision = false;
    if (is_gr) {
      if (use_excise) { fofc_excision = excision_flux_(m,k,j,i); }
    }
    if (excise_enabled) {
      Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real half_diag = 0.5*sqrt(SQR(size.d_view(m).dx1)
                      + (multi_d ? SQR(size.d_view(m).dx2) : 0.0)
                      + (three_d ? SQR(size.d_view(m).dx3) : 0.0));
      Real buffer_r = excise_r + half_diag;
      fofc_excision = fofc_excision ||
          problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                              buffer_r*buffer_r);
    }

    // Apply FOFC
    if (fofc_flag || fofc_excision) {
      // replace x1-flux at i
      // load left state
      HydPrim1D wim1;
      wim1.d  = w0_(m,IDN,k,j,i-1);
      wim1.vx = w0_(m,IVX,k,j,i-1);
      wim1.vy = w0_(m,IVY,k,j,i-1);
      wim1.vz = w0_(m,IVZ,k,j,i-1);
      if (eos.use_e) {wim1.e  = w0_(m,IEN,k,j,i-1);}

      // load right state
      HydPrim1D wi;
      wi.d  = w0_(m,IDN,k,j,i);
      wi.vx = w0_(m,IVX,k,j,i);
      wi.vy = w0_(m,IVY,k,j,i);
      wi.vz = w0_(m,IVZ,k,j,i);
      if (eos.use_e) {wi.e = w0_(m,IEN,k,j,i);}

      // compute new 1st-order LLF flux
      HydCons1D flux;
      if (is_gr) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = LeftEdgeX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        SingleStateLLF_GRHyd(wim1, wi, x1v, x2v, x3v, IVX, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wim1, wi, eos, flux);
      } else {
        SingleStateLLF_Hyd(wim1, wi, eos, flux);
      }

      // store 1st-order fluxes
      flx1(m,IDN,k,j,i) = flux.d;
      flx1(m,IM1,k,j,i) = flux.mx;
      flx1(m,IM2,k,j,i) = flux.my;
      flx1(m,IM3,k,j,i) = flux.mz;
      if (eos.use_e) {flx1(m,IEN,k,j,i) = flux.e;}
      if (nvars_ > nhyd_) {
        for (int n=nhyd_; n<nvars_; ++n) {
          if (dual_enabled && n == dual_idx_) continue;
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*w0_(m,n,k,j,i-1);
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*w0_(m,n,k,j,i);
          }
        }
      }
      if (dual_enabled) {
        SetDualEnergyFOFCFlux(eos, flx1(m,IDN,k,j,i), wim1.d, wi.d,
                              w0_(m,dual_idx_,k,j,i-1), w0_(m,dual_idx_,k,j,i),
                              flx1(m,dual_idx_,k,j,i), vf1_(m,0,k,j,i));
      }

      // replace x1-flux at i+1
      // load right state (left state just wi from above)
      HydPrim1D wip1;
      wip1.d  = w0_(m,IDN,k,j,i+1);
      wip1.vx = w0_(m,IVX,k,j,i+1);
      wip1.vy = w0_(m,IVY,k,j,i+1);
      wip1.vz = w0_(m,IVZ,k,j,i+1);
      if (eos.use_e) {wip1.e = w0_(m,IEN,k,j,i+1);}

      // compute new 1st-order LLF flux
      if (is_gr) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = LeftEdgeX(i+1-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
        SingleStateLLF_GRHyd(wi, wip1, x1v, x2v, x3v, IVX, coord, eos, flux);
      } else if (is_sr) {
        SingleStateLLF_SRHyd(wi, wip1, eos, flux);
      } else {
        SingleStateLLF_Hyd(wi, wip1, eos, flux);
      }

      // store 1st-order fluxes
      flx1(m,IDN,k,j,i+1) = flux.d;
      flx1(m,IM1,k,j,i+1) = flux.mx;
      flx1(m,IM2,k,j,i+1) = flux.my;
      flx1(m,IM3,k,j,i+1) = flux.mz;
      if (eos.use_e) {flx1(m,IEN,k,j,i+1) = flux.e;}
      if (nvars_ > nhyd_) {
        for (int n=nhyd_; n<nvars_; ++n) {
          if (dual_enabled && n == dual_idx_) continue;
          if (flx1(m,IDN,k,j,i+1) >= 0.0) {
            flx1(m,n,k,j,i+1) = flx1(m,IDN,k,j,i+1)*w0_(m,n,k,j,i);
          } else {
            flx1(m,n,k,j,i+1) = flx1(m,IDN,k,j,i+1)*w0_(m,n,k,j,i+1);
          }
        }
      }
      if (dual_enabled) {
        SetDualEnergyFOFCFlux(eos, flx1(m,IDN,k,j,i+1), wi.d, wip1.d,
                              w0_(m,dual_idx_,k,j,i), w0_(m,dual_idx_,k,j,i+1),
                              flx1(m,dual_idx_,k,j,i+1), vf1_(m,0,k,j,i+1));
      }

      if (multi_d) {
        // replace x2-flux at j
        // load left state, permutting components of vectors
        HydPrim1D wjm1;
        wjm1.d  = w0_(m,IDN,k,j-1,i);
        wjm1.vx = w0_(m,IVY,k,j-1,i);
        wjm1.vy = w0_(m,IVZ,k,j-1,i);
        wjm1.vz = w0_(m,IVX,k,j-1,i);
        if (eos.use_e) {wjm1.e = w0_(m,IEN,k,j-1,i);}

        // load right state, permutting components of vectors
        HydPrim1D wj;
        wj.d  = w0_(m,IDN,k,j,i);
        wj.vx = w0_(m,IVY,k,j,i);
        wj.vy = w0_(m,IVZ,k,j,i);
        wj.vz = w0_(m,IVX,k,j,i);
        if (eos.use_e) {wj.e = w0_(m,IEN,k,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wjm1, wj, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wjm1, wj, eos, flux);
        } else {
          SingleStateLLF_Hyd(wjm1, wj, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx2(m,IDN,k,j,i) = flux.d;
        flx2(m,IM2,k,j,i) = flux.mx;
        flx2(m,IM3,k,j,i) = flux.my;
        flx2(m,IM1,k,j,i) = flux.mz;
        if (eos.use_e) {flx2(m,IEN,k,j,i) = flux.e;}
        if (nvars_ > nhyd_) {
          for (int n=nhyd_; n<nvars_; ++n) {
            if (dual_enabled && n == dual_idx_) continue;
            if (flx2(m,IDN,k,j,i) >= 0.0) {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*w0_(m,n,k,j-1,i);
            } else {
              flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*w0_(m,n,k,j,i);
            }
          }
        }
        if (dual_enabled) {
          SetDualEnergyFOFCFlux(eos, flx2(m,IDN,k,j,i), wjm1.d, wj.d,
                                w0_(m,dual_idx_,k,j-1,i), w0_(m,dual_idx_,k,j,i),
                                flx2(m,dual_idx_,k,j,i), vf2_(m,0,k,j,i));
        }

        // replace x2-flux at j+1
        // load left state, permutting components of vectors (just wj from above)
        // load right state, permutting components of vectors
        HydPrim1D wjp1;
        wjp1.d  = w0_(m,IDN,k,j+1,i);
        wjp1.vx = w0_(m,IVY,k,j+1,i);
        wjp1.vy = w0_(m,IVZ,k,j+1,i);
        wjp1.vz = w0_(m,IVX,k,j+1,i);
        if (eos.use_e) {wjp1.e = w0_(m,IEN,k,j+1,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = LeftEdgeX(j+1-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wj, wjp1, x1v, x2v, x3v, IVY, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wj, wjp1, eos, flux);
        } else {
          SingleStateLLF_Hyd(wj, wjp1, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx2(m,IDN,k,j+1,i) = flux.d;
        flx2(m,IM2,k,j+1,i) = flux.mx;
        flx2(m,IM3,k,j+1,i) = flux.my;
        flx2(m,IM1,k,j+1,i) = flux.mz;
        if (eos.use_e) {flx2(m,IEN,k,j+1,i) = flux.e;}
        if (nvars_ > nhyd_) {
          for (int n=nhyd_; n<nvars_; ++n) {
            if (dual_enabled && n == dual_idx_) continue;
            if (flx2(m,IDN,k,j+1,i) >= 0.0) {
              flx2(m,n,k,j+1,i) = flx2(m,IDN,k,j+1,i)*w0_(m,n,k,j,i);
            } else {
              flx2(m,n,k,j+1,i) = flx2(m,IDN,k,j+1,i)*w0_(m,n,k,j+1,i);
            }
          }
        }
        if (dual_enabled) {
          SetDualEnergyFOFCFlux(eos, flx2(m,IDN,k,j+1,i), wj.d, wjp1.d,
                                w0_(m,dual_idx_,k,j,i), w0_(m,dual_idx_,k,j+1,i),
                                flx2(m,dual_idx_,k,j+1,i), vf2_(m,0,k,j+1,i));
        }
      }

      if (three_d) {
        // replace x3-flux at k
        // load left state, permutting components of vectors
        HydPrim1D wkm1;
        wkm1.d  = w0_(m,IDN,k-1,j,i);
        wkm1.vx = w0_(m,IVZ,k-1,j,i);
        wkm1.vy = w0_(m,IVX,k-1,j,i);
        wkm1.vz = w0_(m,IVY,k-1,j,i);
        if (eos.use_e) {wkm1.e = w0_(m,IEN,k-1,j,i);}

        // load right state, permutting components of vectors
        HydPrim1D wk;
        wk.d  = w0_(m,IDN,k,j,i);
        wk.vx = w0_(m,IVZ,k,j,i);
        wk.vy = w0_(m,IVX,k,j,i);
        wk.vz = w0_(m,IVY,k,j,i);
        if (eos.use_e) {wk.e = w0_(m,IEN,k,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wkm1, wk, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wkm1, wk, eos, flux);
        } else {
          SingleStateLLF_Hyd(wkm1, wk, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx3(m,IDN,k,j,i) = flux.d;
        flx3(m,IM3,k,j,i) = flux.mx;
        flx3(m,IM1,k,j,i) = flux.my;
        flx3(m,IM2,k,j,i) = flux.mz;
        if (eos.use_e) {flx3(m,IEN,k,j,i) = flux.e;}
        if (nvars_ > nhyd_) {
          for (int n=nhyd_; n<nvars_; ++n) {
            if (dual_enabled && n == dual_idx_) continue;
            if (flx3(m,IDN,k,j,i) >= 0.0) {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*w0_(m,n,k-1,j,i);
            } else {
              flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*w0_(m,n,k,j,i);
            }
          }
        }
        if (dual_enabled) {
          SetDualEnergyFOFCFlux(eos, flx3(m,IDN,k,j,i), wkm1.d, wk.d,
                                w0_(m,dual_idx_,k-1,j,i), w0_(m,dual_idx_,k,j,i),
                                flx3(m,dual_idx_,k,j,i), vf3_(m,0,k,j,i));
        }

        // replace x3-flux at k+1
        // load left state, permutting components of vectors (just wk from above)
        // load right state, permutting components of vectors
        HydPrim1D wkp1;
        wkp1.d  = w0_(m,IDN,k+1,j,i);
        wkp1.vx = w0_(m,IVZ,k+1,j,i);
        wkp1.vy = w0_(m,IVX,k+1,j,i);
        wkp1.vz = w0_(m,IVY,k+1,j,i);
        if (eos.use_e) {wkp1.e = w0_(m,IEN,k+1,j,i);}

        // compute new first-order flux
        if (is_gr) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = LeftEdgeX(k+1-ks, nx3, x3min, x3max);
          SingleStateLLF_GRHyd(wk, wkp1, x1v, x2v, x3v, IVZ, coord, eos, flux);
        } else if (is_sr) {
          SingleStateLLF_SRHyd(wk, wkp1, eos, flux);
        } else {
          SingleStateLLF_Hyd(wk, wkp1, eos, flux);
        }

        // store 1st-order fluxes, permutting indices
        flx3(m,IDN,k+1,j,i) = flux.d;
        flx3(m,IM3,k+1,j,i) = flux.mx;
        flx3(m,IM1,k+1,j,i) = flux.my;
        flx3(m,IM2,k+1,j,i) = flux.mz;
        if (eos.use_e) {flx3(m,IEN,k+1,j,i) = flux.e;}
        if (nvars_ > nhyd_) {
          for (int n=nhyd_; n<nvars_; ++n) {
            if (dual_enabled && n == dual_idx_) continue;
            if (flx3(m,IDN,k+1,j,i) >= 0.0) {
              flx3(m,n,k+1,j,i) = flx3(m,IDN,k+1,j,i)*w0_(m,n,k,j,i);
            } else {
              flx3(m,n,k+1,j,i) = flx3(m,IDN,k+1,j,i)*w0_(m,n,k+1,j,i);
            }
          }
        }
        if (dual_enabled) {
          SetDualEnergyFOFCFlux(eos, flx3(m,IDN,k+1,j,i), wk.d, wkp1.d,
                                w0_(m,dual_idx_,k,j,i), w0_(m,dual_idx_,k+1,j,i),
                                flx3(m,dual_idx_,k+1,j,i), vf3_(m,0,k+1,j,i));
        }
      }

      // reset FOFC flag (do not reset excision flag)
      if (use_fofc_ && fofc_flag) { fofc_(m,k,j,i) = false; }
    }
  });

  return;
}

} // namespace hydro
