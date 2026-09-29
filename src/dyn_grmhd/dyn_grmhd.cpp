//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.cpp
//! \brief implementation of functions for DynGRMHD and DynGRMHDPS controlling the task
//! list.  The DynGRMHDPS members are in dyn_grmhd_ps_impl.hpp and
//! dyn_grmhd_coord_terms_impl.hpp, compiled per EOS policy by the dyn_grmhd_ps_<eos>.cpp
//! and dyn_grmhd_coord_terms_ideal.cpp units.

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

namespace {

bool OutputVariableRequested(ParameterInput *pin, const std::string &name) {
  for (auto &block : pin->block) {
    if (block.block_name.compare(0, 6, "output") != 0) {
      continue;
    }
    InputLine *line = block.GetPtrToLine("variable");
    if (line != nullptr && line->param_value == name) {
      return true;
    }
    line = block.GetPtrToLine("variable_2");
    if (line != nullptr && line->param_value == name) {
      return true;
    }
  }
  return false;
}

//! \brief An inclusive index box.  Degenerate directions carry (0,0), the convention
//! every other index range in this file uses.
struct InteriorC2PBox {
  int il, iu, jl, ju, kl, ku;
};

//! \brief The active region shrunk by `depth` layers on every evolved direction.
//! Depth 0 is the active region itself, which is what the LAT ghost refresh uses.
InteriorC2PBox MakeInteriorC2PBox(const RegionIndcs &indcs, int depth) {
  InteriorC2PBox b;
  b.il = indcs.is + depth;
  b.iu = indcs.ie - depth;
  b.jl = (indcs.nx2 > 1) ? (indcs.js + depth) : 0;
  b.ju = (indcs.nx2 > 1) ? (indcs.je - depth) : 0;
  b.kl = (indcs.nx3 > 1) ? (indcs.ks + depth) : 0;
  b.ku = (indcs.nx3 > 1) ? (indcs.ke - depth) : 0;
  return b;
}

//! \brief Recover primitives on the complement of `b` inside the full ghost-inclusive
//! extent, as six disjoint bands (two x3 slabs, two x2 bands, two x1 bands) that cover
//! the complement exactly.  Exactness matters both ways: a missed cell is stale, and a
//! cell recovered twice is not a no-op either, because ConsToPrim writes floored or
//! limited states back into u0.  Shared by the LAT ghost refresh (b = the active region)
//! and by the late half of the interior-first split (b = the deep interior) so the two
//! decompositions cannot drift apart.
void ConToPrimComplementOfBox(DynGRMHD *pdyn, const RegionIndcs &indcs,
                              const InteriorC2PBox &b) {
  const int ng = indcs.ng;
  const int n1m1 = indcs.nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  if (indcs.nx3 > 1) {
    pdyn->ConToPrimBC(0, n1m1, 0, n2m1, 0, b.kl-1);
    pdyn->ConToPrimBC(0, n1m1, 0, n2m1, b.ku+1, n3m1);
  }
  if (indcs.nx2 > 1) {
    pdyn->ConToPrimBC(0, n1m1, 0, b.jl-1, b.kl, b.ku);
    pdyn->ConToPrimBC(0, n1m1, b.ju+1, n2m1, b.kl, b.ku);
  }
  pdyn->ConToPrimBC(0, b.il-1, b.jl, b.ju, b.kl, b.ku);
  pdyn->ConToPrimBC(b.iu+1, n1m1, b.jl, b.ju, b.kl, b.ku);
}

} // namespace

// A dumb template function containing the switch statement needed to select an EOS.
template<class ErrorPolicy>
DynGRMHD* SelectDynGRMHDEOS(MeshBlockPack *ppack, ParameterInput *pin,
                            DynGRMHD_EOS eos_policy) {
  DynGRMHD* dyn_gr = nullptr;
  bool use_NQT = false;
  switch(eos_policy) {
    case DynGRMHD_EOS::eos_ideal:
      dyn_gr = new DynGRMHDPS<Primitive::IdealGas, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_piecewise_poly:
      dyn_gr = new DynGRMHDPS<Primitive::PiecewisePolytrope, ErrorPolicy>(ppack, pin);
      break;
    case DynGRMHD_EOS::eos_compose:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
    case DynGRMHD_EOS::eos_hybrid:
      use_NQT = pin->GetOrAddBoolean("mhd", "use_NQT",false);
      if (use_NQT) {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NQTLogs>,
                                ErrorPolicy>(ppack, pin);
      } else {
        dyn_gr = new DynGRMHDPS<Primitive::EOSHybrid<Primitive::NormalLogs>,
                                ErrorPolicy>(ppack, pin);
      }
      break;
  }
  return dyn_gr;
}

DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin) {
  std::string eos_string = pin->GetString("mhd", "dyn_eos");
  std::string error_string = pin->GetString("mhd", "dyn_error");
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  if (eos_string.compare("ideal") == 0) {
    eos_policy = DynGRMHD_EOS::eos_ideal;
  } else if (eos_string.compare("piecewise_poly") == 0) {
    eos_policy = DynGRMHD_EOS::eos_piecewise_poly;
  } else if (eos_string.compare("compose") == 0) {
    eos_policy = DynGRMHD_EOS::eos_compose;
  } else if (eos_string.compare("hybrid") == 0) {
    eos_policy = DynGRMHD_EOS::eos_hybrid;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_eos = '" << eos_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (error_string.compare("reset_floor") == 0) {
    error_policy = DynGRMHD_Error::reset_floor;
  } else {
    std::cout << "### FATAL ERROR in " <<__FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> dyn_error = '" << error_string
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DynGRMHD* dyn_gr = nullptr;

  switch (error_policy) {
    case DynGRMHD_Error::reset_floor:
      dyn_gr = SelectDynGRMHDEOS<Primitive::ResetFloor>(ppack, pin, eos_policy);
      break;
  }

  dyn_gr->eos_policy = eos_policy;
  dyn_gr->error_policy = error_policy;

  return dyn_gr;
}

DynGRMHD::DynGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    temperature("temperature",1,1,1,1,1),
    pmy_pack(pp),
    store_temperature(false),
    fofc_eos_min_y("fofc_eos_min_y", 1),
    fofc_eos_max_y("fofc_eos_max_y", 1) {
  std::string rsolver = pin->GetString("mhd", "rsolver");
  if (rsolver.compare("llf") == 0) {
    rsolver_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (rsolver.compare("hlle") == 0) {
    rsolver_method = DynGRMHD_RSolver::hlle_dyngr;
  } else if (rsolver.compare("hlld") == 0) {
    rsolver_method = DynGRMHD_RSolver::hlld_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> rsolver = '" << rsolver
              << "' not implemented for GR dynamics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The first-order corrector is the flux used at FOFC-flagged and excised faces only.
  // Absent the key it is the corrector every dyn-GR deck has actually run: llf under
  // rsolver = llf, hlle under everything else (hlld included).
  const char *fofc_default =
      (rsolver_method == DynGRMHD_RSolver::llf_dyngr) ? "llf" : "hlle";
  std::string fofc = pin->GetOrAddString("mhd", "fofc_method", fofc_default);
  if (fofc.compare("llf") == 0) {
    fofc_method = DynGRMHD_RSolver::llf_dyngr;
  } else if (fofc.compare("hlle") == 0) {
    fofc_method = DynGRMHD_RSolver::hlle_dyngr;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<mhd> fofc_method = '" << fofc
              << "' is not a first-order corrector; it must be 'llf' or 'hlle' "
              << "(hlld is a five-wave solver, not a first-order flux)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  scratch_level = pin->GetOrAddInteger("mhd", "dyn_scratch", 0);
  enforce_maximum = pin->GetOrAddBoolean("mhd", "enforce_maximum", true);
  dmp_M = pin->GetOrAddReal("mhd", "dmp_M", 1.2);
  scalar_pplimiter = pin->GetOrAddBoolean("mhd", "scalar_pplimiter", true);

  fixed_evolution = pin->GetOrAddBoolean("mhd", "fixed", false);

  const bool temperature_output_requested =
      OutputVariableRequested(pin, "mhd_t") ||
      OutputVariableRequested(pin, "mhd_w") ||
      OutputVariableRequested(pin, "mhd_w_bcc");
  const std::string dyn_eos = pin->GetString("mhd", "dyn_eos");
  store_temperature = temperature_output_requested && dyn_eos != "ideal";

  if (store_temperature) {
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(temperature, nmb, 1, ncells3, ncells2, ncells1);
  } else {
    Kokkos::resize(temperature, 1, 1, 1, 1, 1);
  }
}

DynGRMHD::~DynGRMHD() {
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage)
//  \brief
TaskStatus DynGRMHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // We need the first physical point on all the boundaries in order to calculate
  // the boundaries. So, we need to perform a ConToPrim at these points.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto pm = pmy_pack->pmesh;
  auto pmhd = pmy_pack->pmhd;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  // X1-boundary
  ConToPrimBC(is-ng, is+ng, 0, (n2-1), 0, (n3-1));
  ConToPrimBC(ie-ng, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    ConToPrimBC(0, (n1-1), js-ng, js+ng, 0, (n3-1));
    ConToPrimBC(0, (n1-1), je-ng, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    ConToPrimBC(0, (n1-1), 0, (n2-1), ks-ng, ks+ng);
    ConToPrimBC(0, (n1-1), 0, (n2-1), ke-ng, ke+ng);
  }

  // Physical boundaries
  pmhd->pbval_u->HydroBCs((pmy_pack), (pmhd->pbval_u->u_in), pmhd->w0);
  pmhd->pbval_b->BFieldBCs((pmy_pack), (pmhd->pbval_b->b_in), pmhd->b0);

  // User BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  // We now need to do a PrimToCon on all these boundary points.
  // X1-boundary
  PrimToConInit(is-ng, is, 0, (n2-1), 0, (n3-1));
  PrimToConInit(ie, ie+ng, 0, (n2-1), 0, (n3-1));
  // X2-boundary
  if (pm->multi_d) {
    PrimToConInit(0, (n1-1), js-ng, js, 0, (n3-1));
    PrimToConInit(0, (n1-1), je, je+ng, 0, (n3-1));
  }
  // X3-boundary
  if (pm->three_d) {
    PrimToConInit(0, (n1-1), 0, (n2-1), ks-ng, ks);
    PrimToConInit(0, (n1-1), 0, (n2-1), ke, ke+ng);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage)
//! \brief Add the perfect fluid contribution to the stress-energy tensor. This is assumed
//!  to be the first contribution, so it sets the values rather than adding.
TaskStatus DynGRMHD::SetTmunu(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  //auto &size  = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  const auto metric = pmy_pack->padm->GetMetricView();
  auto &tmunu = pmy_pack->ptmunu->tmunu;
  //auto &nhyd = pmy_pack->pmhd->nmhd;
  //int &nscal = pmy_pack->pmhd->nscalars;
  auto &prim = pmy_pack->pmhd->w0;
  // TODO(JMF): double-check that this needs to be u1, not u0!
  auto &cons = pmy_pack->pmhd->u0;
  auto &bcc = pmy_pack->pmhd->bcc0;

  par_for("dyngr_tmunu_loop",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    adm::ADMMetricPoint metric_point{};
    metric.CellMetric(m, k, j, i, metric_point);
    // Calculate the determinant/volume form
    Real detg = adm::SpatialDet(metric_point.g_dd[S11], metric_point.g_dd[S12],
                                metric_point.g_dd[S13], metric_point.g_dd[S22],
                                metric_point.g_dd[S23], metric_point.g_dd[S33]);
    Real ivol = 1.0/sqrt(detg);

    // Calculate the lower velocity components
    const int imap[3][3] = {
      {S11, S12, S13},
      {S12, S22, S23},
      {S13, S23, S33}
    };
    Real v_d[3] = {0.0};
    Real iW = 0.;
    Real B_d[3] = {0.0};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        v_d[a] += prim(m, IVX + b, k, j, i)*metric_point.g_dd[imap[a][b]];
        iW += prim(m, IVX + a, k, j, i)*prim(m, IVX + b, k, j, i) *
                metric_point.g_dd[imap[a][b]];
        B_d[a] += bcc(m, b, k, j, i)*metric_point.g_dd[imap[a][b]]*ivol;
      }
    }
    iW = 1.0/sqrt(1. + iW);
    Real Bv = 0.;
    Real Bsq = 0.;
    for (int a = 0; a < 3; ++a) {
      Bv += bcc(m, a, k, j, i) * v_d[a]*ivol;
      Bsq += bcc(m, a, k, j, i) * B_d[a]*ivol;
    }
    Real bsq = (Bsq + Bv*Bv)*(iW*iW);

    tmunu.E(m, k, j, i) = (cons(m, IEN, k, j, i) + cons(m, IDN, k, j, i))*ivol;
    for (int a = 0; a < 3; ++a) {
      tmunu.S_d(m, a, k, j, i) = cons(m, IM1 + a, k, j, i)*ivol;
      for (int b = a; b < 3; ++b) {
        tmunu.S_dd(m, a, b, k, j, i) =
              cons(m, IM1 + a, k, j, i)*ivol*v_d[b]*iW
              - (B_d[a] + Bv*v_d[a])*SQR(iW)*B_d[b]
              + (prim(m, IPR, k, j, i) + 0.5*bsq)*
                metric_point.g_dd[imap[a][b]];
      }
    }
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void DynGRMHD::SetADMVariables
//! \brief

TaskStatus DynGRMHD::SetADMVariables(Driver *pdrive, int stage) {
  const Real target_time =
      problem_runtime::HydroStageTimeOr(pmy_pack->pmesh->time);
  SetADMVariablesAtTime(target_time);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// INTERIOR-FIRST ConsToPrim
//
// The end-of-stage recovery (MHD_C2P) used to be one
// full-extent ConsToPrim launch queued after MHD_BCS, i.e. after the whole conserved-
// variable exchange had drained, leaving the GPU idle for most of that exchange.
// ConsToPrim is pointwise, so the same work is issued as two launches over disjoint
// index ranges: a DEEP INTERIOR pass queued as soon as the exchange has been posted,
// and a COMPLEMENT pass in the old DAG slot after MHD_BCS.  Both go through the same
// virtual ConToPrimBC and therefore the same kernel and template instantiation; only
// the loop bounds differ, so every cell is recovered from the same inputs by the same
// arithmetic and the result is bitwise identical.
//
// ConsToPrim is NOT read-only: besides w0/bcc0/temperature it writes u0 back whenever a
// cell is floored or limited.  So the early pass must not run before anything that
// still reads the pre-recovery u0 of an ACTIVE cell.  The readers between the last u0
// writer and MHD_BCS are:
//   * MHD_RestU  -- restricts u0 into coarse_u0; an ancestor of the early pass.
//   * MHD_SendU  -- packs active cells and fences the pack before MPI_Startall, so a
//                   dependency on MHD_SendU means "the pack has finished reading u0".
//                   This is the earliest legal point.  That fence is skipped for
//                   nranks == 1, which InteriorFirstC2PUsable() therefore refuses.
//   * MHD_RecvU / MHD_Prolong -- write ghost cells and coarse_u0 only.
//   * MHD_BCS    -- outflow/diode read one active layer, reflect reads ng layers.  The
//                   early pass therefore stops PhysicalBCInteriorReadDepth() layers
//                   short of the active-region faces, and the complement pass, which
//                   runs after MHD_BCS, covers that shell together with the ghosts.
// Face fields: the early pass reads x1f(i) and x1f(i+1) of every cell it recovers.
// Those are final after MHD_CT (expressed by the MHD_SendB dependency) EXCEPT on the
// active-region faces, which ProlongateFC may rewrite when the normal neighbour is
// coarser.  That is why the depth is clamped to at least 1: the deep interior then
// starts at is+1, whose faces are strictly interior and no prolongation touches them.
//
// Cost: each band is one launch, so a split recovery is 1 + 6 launches instead of 1.
// The exchange overlap was measured to outweigh that on this ten-GPU node; where the
// configuration falls outside the audit, InteriorFirstC2PUsable() returns false and the
// late pass covers the full extent as it always did.
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//! \fn int DynGRMHD::PhysicalBCInteriorReadDepth()
//! \brief Number of layers of ACTIVE cells the early pass must leave alone: the maximum
//! over the six mesh faces of what MHD::ApplyPhysicalBCs reads out of u0, clamped to at
//! least 1.  An upper bound over the mesh faces (a block's own flag can only be milder),
//! so it is safe for every block in the pack.  The clamp is not about u0: it keeps the
//! deep interior clear of the active-region faces that ProlongateFC may rewrite.

int DynGRMHD::PhysicalBCInteriorReadDepth() const {
  auto *pm = pmy_pack->pmesh;
  const int ng = pm->mb_indcs.ng;
  int depth = 1;
  if (!pm->strictly_periodic) {
    for (int f = 0; f < 6; ++f) {
      switch (pm->mesh_bcs[f]) {
        // hydro_bcs.cpp:76-82 -- u0(is-i-1) = +/- u0(is+i), i < ng.
        case BoundaryFlag::reflect:
          depth = std::max(depth, ng);
          break;
        // hydro_bcs.cpp:84-88, 94-101 -- reads exactly the first active layer.
        case BoundaryFlag::outflow:
        case BoundaryFlag::diode:
          depth = std::max(depth, 1);
          break;
        // inflow and vacuum write constants; periodic/shear_periodic/block/undef do
        // nothing.  `user` is refused outright in InteriorFirstC2PUsable().
        default:
          break;
      }
    }
  }
  return depth;
}

//----------------------------------------------------------------------------------------
//! \fn bool DynGRMHD::InteriorFirstC2PUsable()
//! \brief Whether this run's configuration is inside the audit above.  Evaluated per
//! call (a few loads and a six-way switch) rather than cached at queue time, because the
//! ProblemGenerator -- and therefore user_bcs -- is constructed after the task graph.
//! When it is false the early pass does nothing and the late pass covers the FULL
//! extent, i.e. exactly the old single-pass behaviour in the old DAG slot.

bool DynGRMHD::InteriorFirstC2PUsable() const {
  // Single-rank runs: MHD_SendU returns before the pack fence, so "SendU complete" does
  // not imply "the pack has finished reading u0" -- and there is no MPI traffic to
  // overlap with anyway.
  if (global_variable::nranks == 1) {
    return false;
  }
  auto *pm = pmy_pack->pmesh;
  // A user boundary callback runs after HydroBCs and may read any part of u0, so the
  // shell cannot be bounded.
  if (pm->pgen != nullptr && pm->pgen->user_bcs) {
    return false;
  }
  for (int f = 0; f < 6; ++f) {
    if (pm->mesh_bcs[f] == BoundaryFlag::user) {
      return false;
    }
  }
  // Not audited: Z4c adds its own excision task as an optional MHD_C2P dependency, and
  // prolong_prims makes MHD::Prolongate run a C2P/P2C round trip that reads w0 between
  // the two passes.
  if (pmy_pack->pz4c != nullptr) {
    return false;
  }
  if (pm->multilevel && pm->pmr != nullptr && pm->pmr->prolong_prims) {
    return false;
  }
  // The dual-energy adiabat is resynchronized in MHD_DualE, ahead of MHD_SendU's pack,
  // so both halves of the split read the column the neighbours were sent.
  // The shell must leave at least one deep-interior cell in every evolved direction.
  const auto &indcs = pm->mb_indcs;
  const int depth = PhysicalBCInteriorReadDepth();
  if (indcs.nx1 <= 2*depth) {
    return false;
  }
  if (indcs.nx2 > 1 && indcs.nx2 <= 2*depth) {
    return false;
  }
  if (indcs.nx3 > 1 && indcs.nx3 <= 2*depth) {
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynGRMHD::ConToPrimInteriorFirst
//! \brief Early half of the split: the deep interior, issued while the conserved-variable
//! exchange is in flight.

TaskStatus DynGRMHD::ConToPrimInteriorFirst(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }
  if (!InteriorFirstC2PUsable()) {
    // The late pass will cover the full extent on its own.
    return TaskStatus::complete;
  }
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int depth = PhysicalBCInteriorReadDepth();
  const InteriorC2PBox b = MakeInteriorC2PBox(indcs, depth);
  ConToPrimBC(b.il, b.iu, b.jl, b.ju, b.kl, b.ku);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynGRMHD::ConToPrimAfterExchange
//! \brief Late half of the split, in the DAG slot the single full-extent MHD_C2P used to
//! occupy: the exact complement of the deep-interior box (ghost zones plus the boundary-
//! condition read shell), or the whole extent when the split is off or not usable.

TaskStatus DynGRMHD::ConToPrimAfterExchange(Driver *pdrive, int stage) {
  if (fixed_evolution) {
    return TaskStatus::complete;
  }
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (!InteriorFirstC2PUsable()) {
    // Byte-for-byte the old DynGRMHDPS::ConToPrim call.
    const int ng = indcs.ng;
    const int n1m1 = indcs.nx1 + 2*ng - 1;
    const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
    const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
    ConToPrimBC(0, n1m1, 0, n2m1, 0, n3m1);
    return TaskStatus::complete;
  }
  ConToPrimComplementOfBox(this, indcs,
                           MakeInteriorC2PBox(indcs, PhysicalBCInteriorReadDepth()));
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
void DynGRMHD::SetADMVariablesAtTime(const Real time) {
  auto *padm = pmy_pack->padm;
  const bool old_override_valid = padm->callback_time_override_valid;
  const Real old_override = padm->callback_time_override;
  padm->callback_time_override = time;
  padm->callback_time_override_valid = true;
  padm->SetADMVariables(pmy_pack);
  padm->callback_time_override = old_override;
  padm->callback_time_override_valid = old_override_valid;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::UpdateExcisionMasks
//! \brief

TaskStatus DynGRMHD::UpdateExcisionMasks(Driver *pdrive, int stage) {
  if (pmy_pack->pcoord->coord_data.bh_excise) {
    pmy_pack->pcoord->UpdateExcisionMasks();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus DynGRMHD::SetADMVariablesAtStageEnd
//! \brief Install the analytic metric of the time level the stage has just advanced the
//! conserved state to, before anything inverts that state.
//!
//! The stage's right-hand side is evaluated on the metric of its abscissa t_s
//! (MHD_SetADM), and the update leaves U = sqrt(gamma) (D, S_i, tau) at the next
//! abscissa (the end of the step after the last stage), the time the geometric source
//! integrated it to.  Inverting that U on the metric of t_s is not the gas of either
//! time: sqrt(gamma) and gamma_ij move by O(dt), and the internal energy is what remains
//! of tau after the kinetic and magnetic parts, so where it is a small fraction of tau
//! -- cold moving gas, and the magnetized flow near a moving horizon -- the inversion
//! read it off by O(1) or found it negative.  The floors then fired on the wrong metric
//! and wrote the floored state back into U, heat the next stage's inversion (on the
//! right metric) kept, while the radiation's implicit exchange, which inverts the same U
//! on the metric of its own time level, saw a gas 10-1000 times hotter than the one
//! published.  So the metric of the new time level is installed here, after the last
//! consumer of the stage's right-hand-side metric in this graph (the EMF, the source
//! terms and, when it runs at the stage start, the radiation transport) and ahead of
//! every recovery of the updated state; the next stage's MHD_SetADM finds it installed.
//!
//! Only the metric moves.  The excision masks (and the rest of the problem generator's
//! per-install state) stay the stage's until MHD_SetADM: they are the geometry of the
//! hole the stage's fluxes were built around, and an excised cell is set to the
//! excision state, not inverted, so the recovery re-excises exactly the cells the
//! stage's operator treated as excised.  A cell the hole uncovers during the step
//! therefore starts from the excision state at the next stage, as it did.
//!
TaskStatus DynGRMHD::SetADMVariablesAtStageEnd(Driver *pdrive, int stage) {
  if (pdrive == nullptr) {
    return TaskStatus::complete;
  }
  const Mesh *pm = pmy_pack->pmesh;
  const Real stage_end_frac =
      (stage < pdrive->nexp_stages) ? pdrive->stage_time_frac[stage] : 1.0;
  const Real t_end = pm->time + stage_end_frac*pm->dt;
  if (t_end != problem_runtime::HydroStageTimeOr(pm->time)) {
    pmy_pack->padm->SetMetricTime(t_end);
  }
  return TaskStatus::complete;
}

} // namespace dyngr
