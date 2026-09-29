//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart.cpp
//! \brief writes restart files

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cstddef>     // size_t
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility> // make_pair
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/lte_table_utils.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "sink_particles/sink_particles.hpp"
#include "pgen/pgen.hpp"
#include "srcterms/turb_driver.hpp"
//#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// constructor: also calls BaseTypeOutput base class constructor

RestartOutput::RestartOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for outputs. Comments in binary.cpp constructor explain why
  mkdir("rst",0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rst/rank_%08d/", global_variable::my_rank);
    mkdir(rank_dir, 0775);
  }
}

//----------------------------------------------------------------------------------------
// RestartOutput::LoadOutputData()
// overload of standard load data function specific to restarts.  Loads dependent
// variables, including ghost zones.

void RestartOutput::LoadOutputData(Mesh *pm) {
  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nhydro=0, nmhd=0, nrad=0, nforce=3, nadm=0, nz4c=0;
  if (phydro != nullptr) {
    nhydro = phydro->nvars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nvars;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->RestartVariableCount();
  }
  // if the spacetime is evolved, we do not need to checkpoint/recover the ADM variables
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }

  // Without the hybrid sidecar a checkpoint carries the conserved state, never the
  // primitives: a restart re-derives w0 from u0, and it does so with the Kastaun
  // warm-start cache empty (eos.hpp, c2p_mu_cache), because that cache is run history
  // such a file does not hold.  This run's own primitives were recovered WARM.  Both are
  // the same root to the solve's 1e-12 tolerance but not the same double, so a run
  // restarted from this file would leave the checkpointed trajectory at that level --
  // and the stiff radiation coupling amplifies it (measured on the M1 funnel test: 7e-16
  // in eint at the restart cycle becomes a 4e-9 relative difference in J five light
  // crossings later).  Repeat the recovery here the way the restart will do it, so that
  // both sides continue from the same numbers: empty the cache, invert, and put u0 back
  // afterwards -- the inversion floors u0 in place, and the state this run carries (and
  // checkpoints) must stay the one its own step produced.  One extra C2P per checkpoint,
  // and a no-op for every policy that keeps no such cache (ResetC2PWarmStart returns
  // false).  Checkpoints are written at a synchronized point with every MeshBlock active
  // (Driver::Execute clears any LAT mask first), so this covers the same cells the
  // restart's recovery will.  A second checkpoint of the same state (the in-loop and the
  // Driver::Finalize one of a run's last cycle) repeats the recovery on this one's
  // result and writes the same u0, so the two files are the same.
  //
  // The dynamical-metric inversion keeps no cache, but it is not idempotent either: this
  // run's primitives came from the conserved state before that recovery's floors were
  // written back, the restart's from the state after.  A step that reads the owned
  // primitives before inverting again -- every LAT window, whose refresh inverts ghost
  // bands only -- then starts off the checkpointed trajectory at 1e-14.  So it repeats
  // its recovery here too, from the checkpointed state, and this run continues from the
  // replayed primitives: a restart reproduces the run that wrote this checkpoint, which
  // differs at round-off in floored cells from a run that did not.  With the hybrid
  // sidecar configured, active or dormant, on either metric, the checkpoint carries the
  // primitives themselves (the history block above) and the restart restores them, so
  // nothing is repeated here and this run's state is left as its own step produced it.
  //
  // The saved u0 is held on the HOST: a device copy is a full conserved state (about
  // 0.5 GiB per rank on the production BBH mesh) allocated on top of the running
  // footprint at every checkpoint, and the round trip over PCIe is small next to
  // writing the file.
  dyngr::DynGRMHD *pdyngr = pm->pmb_pack->pdyngr;
  if (pmhd != nullptr &&
      (pdyngr != nullptr || pmhd->peos->ResetC2PWarmStart())) {
    // create_mirror (not _view) allocates even on a host-only build, where a mirror
    // view would alias u0 and the restore below would restore nothing.
    auto u_saved =
        Kokkos::create_mirror(Kokkos::WithoutInitializing, HostMemSpace(), pmhd->u0);
    Kokkos::deep_copy(u_saved, pmhd->u0);
    if (pdyngr != nullptr) {
      (void) pdyngr->ConToPrim(nullptr, 0);
    } else {
      (void) pmhd->ConToPrim(nullptr, 0);
    }
    Kokkos::deep_copy(pmhd->u0, u_saved);
  }

  // Note for restarts, outarrays are dimensioned (m,n,k,j,i)
  if (phydro != nullptr) {
    Kokkos::realloc(outarray_hyd, nmb, nhydro, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_hyd, Kokkos::subview(phydro->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pmhd != nullptr) {
    Kokkos::realloc(outarray_mhd, nmb, nmhd, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_mhd, Kokkos::subview(pmhd->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::deep_copy(outfield.x1f, Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::deep_copy(outfield.x2f, Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::realloc(outfield.x3f, nmb, nout3+1, nout2, nout1);
    Kokkos::deep_copy(outfield.x3f, Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (prad != nullptr) {
    Kokkos::realloc(outarray_rad, nmb, nrad, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_rad, Kokkos::subview(prad->i0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pturb != nullptr) {
    Kokkos::realloc(outarray_force, nmb, nforce, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_force, Kokkos::subview(pturb->force, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }
  if (pz4c != nullptr) {
    Kokkos::realloc(outarray_z4c, nmb, nz4c, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_z4c, Kokkos::subview(pz4c->u0, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  } else if (padm != nullptr && nadm > 0) {
    Kokkos::realloc(outarray_adm, nmb, nadm, nout3, nout2, nout1);
    Kokkos::deep_copy(outarray_adm, Kokkos::subview(padm->u_adm, std::make_pair(0,nmb),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
  }

  // calculate max/min number of MeshBlocks across all ranks
  noutmbs_max = pm->nmb_eachrank[0];
  noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::string RestartOutput::FileNameForNumber(int number)
//  \brief "rst/file_basename" + "." + XXXXX + ".rst", where XXXXX is the 5-digit file
//   number, under "rst/rank_YYYYYYYY/" when each rank writes its own file.  Used both to
//   write the checkpoint and to report, at startup, which existing checkpoints a resumed
//   run is about to supersede.

std::string RestartOutput::FileNameForNumber(int number) const {
  char digits[7];
  std::snprintf(digits, sizeof(digits), ".%05d", number);
  std::string dir("rst/");
  if (out_params.single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    dir += rank_dir;
  }
  return dir + out_params.file_basename + digits + ".rst";
}

//----------------------------------------------------------------------------------------
//! \fn void RestartOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes everything to a single restart file

void RestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  sinkparticles::SinkParticles* psink = pm->pmb_pack->psink;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  adm::ADM* padm = pm->pmb_pack->padm;
  int nhydro=0, nmhd=0, nrad=0, nforce=3, nz4c=0, nadm=0, nco=0;
  if (phydro != nullptr) {
    nhydro = phydro->nvars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nvars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
    nco = pz4c->ptracker.size();
  } else if (padm != nullptr) {
    nadm = padm->RestartVariableCount();
  }
  bool single_file_per_rank = out_params.single_file_per_rank;
  const std::string fname = FileNameForNumber(out_params.file_number);
  // increment counters now so values for *next* dump are stored in restart file
  out_params.file_number++;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  AdvanceOutputTime(pm, pin);
  problem_runtime::StoreRuntimeMetadata(pin);
  if (pm != nullptr && pm->pgen != nullptr && pm->pgen->user_metadata_func != nullptr) {
    pm->pgen->user_metadata_func(pin, pm);
  }
  auto store_eos_restart_metadata = [&](const std::string &block, EquationOfState *peos) {
    if (peos == nullptr) return;
    const auto &eos = peos->eos_data;
    if (!pin->DoesBlockExist(block) || !pin->DoesParameterExist(block, "table")) return;
    pin->SetString("saha_runtime", block + "_table_runtime", pin->GetString(block,
        "table"));
    std::string table_type = "unknown";
    const std::string eos_name = pin->GetString(block, "eos");
    if (eos.hydro_eos == HydroEOSModel::saha_table) {
      table_type = "saha_hydrogen_lte";
    } else if (eos.hydro_eos == HydroEOSModel::lte_table) {
      table_type = lte_table_utils::ExpectedTableTypeForEosName(eos_name);
    }
    pin->SetString("saha_runtime", block + "_table_type_runtime", table_type);
    pin->SetReal("saha_runtime", block + "_lte_x_runtime", eos.lte_h_mass_fraction);
    pin->SetReal("saha_runtime", block + "_lte_y_runtime", eos.lte_he_mass_fraction);
    pin->SetReal("saha_runtime", block + "_lte_prad_runtime",
                 eos.lte_has_radiation ? 1.0 : 0.0);
    pin->SetReal("saha_runtime", block + "_lte_h2_runtime", eos.lte_has_h2 ? 1.0 : 0.0);
    pin->SetReal("saha_runtime", block + "_lte_zpe_runtime",
                 eos.lte_zpe_subtracted ? 1.0 : 0.0);
  };
  if (pm->pmb_pack->phydro != nullptr) {
    store_eos_restart_metadata("hydro", pm->pmb_pack->phydro->peos);
  }
  if (pm->pmb_pack->pmhd != nullptr) {
    store_eos_restart_metadata("mhd", pm->pmb_pack->pmhd->peos);
  }

  // A global SMR restart already has a fixed, balanced GID ordering. Store its exact
  // rank boundaries in the parameter header so a same-size restart can reconstruct the
  // partition without rerunning the LAT load balancer. Older restart files simply lack
  // this metadata and use the deterministic contiguous fallback in BuildTreeFromRestart.
  if (!pm->adaptive && !single_file_per_rank) {
    std::ostringstream rank_counts;
    for (int rank=0; rank<global_variable::nranks; ++rank) {
      if (rank > 0) rank_counts << ',';
      rank_counts << pm->nmb_eachrank[rank];
    }
    pin->SetInteger("time", "smr_restart_partition_version", 1);
    pin->SetInteger("time", "smr_restart_nranks", global_variable::nranks);
    pin->SetString("time", "smr_restart_nmb_eachrank", rank_counts.str());
    pin->SetBoolean("time", "smr_restart_gid_reordered",
                    pm->hydro_lat_gid_reordered);
    // The two anchors of the LAT rebalance trigger (Driver::Execute) and the cost model
    // this partition was balanced for.  A restart that keeps the partition under the same
    // model resumes them (Mesh::BuildTreeFromRestart).
    pin->SetInteger("time", "smr_restart_topology_last_change_cycle",
                    pm->topology_last_change_cycle);
    pin->SetInteger("time", "smr_restart_lat_lb_last_attempt_cycle",
                    pm->hydro_lat_lb_last_attempt_cycle);
    pin->SetString("time", "smr_restart_lat_cost_model", Mesh::HydroLATCostModel(pin));
  }

  // Record the writing run's AMR cadence anchor.  Under LAT, Driver::Execute calls AMR
  // at a window end whenever (ncycle - last_amr_call_cycle) >= ncycle_check, and window
  // ends are sums of variable sync factors (tlim, gravity solve_dt, sink and nlim
  // clamps), so the writer's AMR calls do not land on multiples of ncycle_check and
  // cannot be reconstructed from ncycle alone.  Execute makes this dump immediately
  // BEFORE this cycle's AMR pass, so the stored value is exactly the one that pass tests,
  // and the restart's owed-AMR pass (Driver::Initialize Step 1b) can evaluate the
  // writer's predicate verbatim; a Finalize dump after a wall-clock exit follows that
  // pass and stores ncycle itself.  Written by the writer like <outputN>/file_number and
  // <time>/smr_restart_nranks; it is not a user input key.
  if (pm->adaptive && pm->pmr != nullptr) {
    pin->SetInteger("mesh_refinement", "last_amr_call_cycle",
                    pm->pmr->last_amr_call_cycle);
  }
  // Each MeshBlock's cycles since its last refinement (MeshRefinement::ncyc_since_ref),
  // which gates its next refinement against refinement_interval and orders the
  // max_nmb_per_rank cancellation, is written in STEP 3 in gid order; this key holds its
  // entry count, and 0 when the writer is not adaptive.
  const bool write_ncyc_since_ref = pm->adaptive && pm->pmr != nullptr;
  if (write_ncyc_since_ref) {
    pin->SetInteger("mesh_refinement", "restart_ncyc_since_ref_count", pm->nmb_total);
  } else if (pin->DoesParameterExist("mesh_refinement",
                                     "restart_ncyc_since_ref_count")) {
    pin->SetInteger("mesh_refinement", "restart_ncyc_since_ref_count", 0);
  }

  // create string holding input parameters (copy of input file)
  std::stringstream ost;
  pin->ParameterDump(ost);
  std::string sbuf = ost.str();

  //--- STEP 1.  Root process writes header data (input file, critical variables)
  // Input file data is read by ParameterInput on restart, and the remaining header
  // variables are read in Mesh::BuildTreeFromRestart()

  // open file and  write the header; this part is serial
  IOWrapper resfile;
  resfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    // output the input parameters (input file)
    resfile.Write_any_type(sbuf.c_str(), sbuf.size(), "byte", single_file_per_rank);

    // output Mesh information
    resfile.Write_any_type(&(pm->nmb_total), (sizeof(int)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->root_level), (sizeof(int)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mesh_size), (sizeof(RegionSize)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mesh_indcs), (sizeof(RegionIndcs)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->mb_indcs), (sizeof(RegionIndcs)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->time), (sizeof(Real)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->dt), (sizeof(Real)), "byte",
                            single_file_per_rank);
    resfile.Write_any_type(&(pm->ncycle), (sizeof(int)), "byte",
                            single_file_per_rank);
  }
  //--- STEP 2.  Root process writes list of logical locations and cost of MeshBlocks
  // This data read in Mesh::BuildTreeFromRestart()

  if (global_variable::my_rank == 0 || single_file_per_rank) {
    resfile.Write_any_type(&(pm->lloc_eachmb[0]),(pm->nmb_total)*sizeof(LogicalLocation),
                           "byte", single_file_per_rank);
    resfile.Write_any_type(&(pm->cost_eachmb[0]), (pm->nmb_total)*sizeof(float),
                           "byte", single_file_per_rank);
  }

  //--- STEP 3.  Root process writes internal state of objects that require it
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    // store z4c information
    if (pz4c != nullptr) {
      resfile.Write_any_type(&(pz4c->last_output_time), sizeof(Real), "byte",
                             single_file_per_rank);
    }
    // output puncture tracker data
    if (nco > 0) {
      for (auto & pt : pz4c->ptracker) {
        resfile.Write_any_type(pt->GetPos(), 3*sizeof(Real), "byte",
                               single_file_per_rank);
      }
    }
    // turbulence driver internal RNG
    if (pturb != nullptr) {
      resfile.Write_any_type(&(pturb->rstate), sizeof(RNG_State), "byte",
                             single_file_per_rank);
    }
    // Sink-particle list.  The list is replicated identically on every rank, so one
    // writer emits it whole: a count followed by the module's packed POD image.  It
    // lives in the header (STEP 3) and not in the per-MeshBlock payload because sinks
    // are not owned by any MeshBlock and their count is independent of nmb_total; the
    // headeroffset taken below therefore already accounts for these bytes.
    if (psink != nullptr) {
      const int nsinks = psink->nsinks;
      resfile.Write_any_type(&nsinks, sizeof(int), "byte", single_file_per_rank);
      if (nsinks > 0) {
        const std::size_t sink_bytes = psink->RestartDataSize();
        std::vector<char> sink_data(sink_bytes);
        psink->PackRestartData(sink_data.data());
        resfile.Write_any_type(sink_data.data(), sink_bytes, "byte",
                               single_file_per_rank);
      }
    }
    if (write_ncyc_since_ref) {
      resfile.Write_any_type(pm->pmr->ncyc_since_ref.data(),
                             (pm->nmb_total)*sizeof(int), "byte", single_file_per_rank);
    }
  }

  //--- STEP 4.  All ranks write data over all MeshBlocks (5D arrays) in parallel
  // This data read in ProblemGenerator constructor for restarts

  // total size of all cell-centered variables and face-centered fields to be written by
  // this rank
  IOWrapperSizeT data_size = 0;
  if (phydro != nullptr) {
    data_size += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
  }
  if (prad != nullptr) {
    data_size += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
  }
  if (pturb != nullptr) {
    data_size += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr && nadm > 0) {
    data_size += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
  }
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    resfile.Write_any_type(&(data_size), sizeof(IOWrapperSizeT), "byte",
                            single_file_per_rank);
  }

  // Root rank determines the true payload offset from the actual file position after
  // writing the restart header. Using locally reconstructed ParameterDump sizes is not
  // safe because rank-local runtime metadata can change the serialized header length.
  IOWrapperSizeT headeroffset = 0;
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    headeroffset = resfile.GetPosition(single_file_per_rank);
  }
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  // write cell-centered variables in parallel
  IOWrapperSizeT offset_myrank = headeroffset;

  if (!single_file_per_rank) {
    offset_myrank += data_size*(pm->gids_eachrank[global_variable::my_rank]);
  }

  IOWrapperSizeT myoffset = offset_myrank;

  // write cell-centered variables, one MeshBlock at a time (but parallelized over all
  // ranks). MeshBlocks are written seperately to reduce number of data elements per write
  // call, to avoid exceeding 2^31 limit for very large grids per MPI rank.
  if (phydro != nullptr) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_hyd, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered hydro data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_hyd, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(), mbcnt, myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered hydro data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
    myoffset = offset_myrank;
  }
  if (pmhd != nullptr) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_mhd, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered mhd data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_mhd, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(), mbcnt, myoffset,"Real",
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered mhd data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    myoffset = offset_myrank;

    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(outfield.x1f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        size_t fldcnt = x1fptr.size();
        if (resfile.Write_any_type_at_all(x1fptr.data(),fldcnt,myoffset,"Real",
                                          single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x1f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(outfield.x2f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        fldcnt = x2fptr.size();
        if (resfile.Write_any_type_at_all(x2fptr.data(),fldcnt,myoffset,"Real",
                                          single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x2f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(outfield.x3f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        fldcnt = x3fptr.size();
        if (resfile.Write_any_type_at_all(x3fptr.data(),fldcnt,myoffset,"Real",
                                          single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x3f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(outfield.x1f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        size_t fldcnt = x1fptr.size();
        if (resfile.Write_any_type_at(x1fptr.data(),fldcnt,myoffset,"Real",
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x1f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(outfield.x2f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        fldcnt = x2fptr.size();
        if (resfile.Write_any_type_at(x2fptr.data(),fldcnt,myoffset,"Real",
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x2f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(outfield.x3f,m,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        fldcnt = x3fptr.size();
        if (resfile.Write_any_type_at(x3fptr.data(),fldcnt,myoffset,"Real",
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "b0.x3f data not written correctly to rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      }
    }
    offset_myrank += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    offset_myrank += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    offset_myrank += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    myoffset = offset_myrank;
  }

  if (prad != nullptr) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_rad, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered rad data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_rad, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(),mbcnt,myoffset,"Real",
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered rad data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
    myoffset = offset_myrank;
  }

  if (pturb != nullptr) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_force, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "cell-centered turb data not written correctly to rst file, "
          << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_force, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(), mbcnt, myoffset,"Real",
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered turb data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    myoffset = offset_myrank;
  }

  if (pz4c != nullptr) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_z4c, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered z4c data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_z4c, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(), mbcnt, myoffset,"Real",
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered z4c data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nz4c*sizeof(Real); // z4c u0
    myoffset = offset_myrank;
  } else if (padm != nullptr && nadm > 0) {
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(outarray_adm, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at_all(mbptr.data(),mbcnt,myoffset,"Real",
                                          single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered adm data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(outarray_adm, m, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL, Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Write_any_type_at(mbptr.data(), mbcnt, myoffset,"Real",
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "cell-centered adm data not written correctly"
                    << " to rst file, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    offset_myrank += nout1*nout2*nout3*nadm*sizeof(Real); // adm u_adm
    myoffset = offset_myrank;
  }

  // close file, clean up
  resfile.Close(single_file_per_rank);

  return;
}
