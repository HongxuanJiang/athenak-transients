#ifndef DYN_GRMHD_DYN_GRMHD_HPP_
#define DYN_GRMHD_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd.hpp
//  \brief definitions for DynGRMHD class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "utils/launch_config.hpp"

//----------------------------------------------------------------------------------------
//! Occupancy target for the dyn-GR face kernels (Riemann solve and FOFC).
//!
//! These kernels carry roughly 226 registers of live state, which no GPU can hold at a
//! useful occupancy.  ptxas therefore has to choose between keeping the working set in
//! registers (few resident warps) and spilling it to local memory (more resident warps),
//! and Kokkos::LaunchBounds<MaxThreads, MinBlocksPerSM> is how we tell it which.  The
//! right answer turns on the register file size, L1 capacity and memory latency of the
//! target GPU, so it is not a constant we can honestly write down here.
//!
//! The launch sites therefore call athenak::launch::par_for_auto, which compiles both
//! candidates and picks between them at runtime from what cudaFuncGetAttributes and
//! cudaDeviceGetAttribute report on the machine actually running -- deterministically,
//! with no timing.  See src/utils/launch_config.hpp for the rule and its limits.
//!
//! The value below is now only the FALLBACK, used on non-CUDA backends, with
//! -DATHENAK_LAUNCH_AUTOTUNE=0, or if the attribute query fails.  It remains overridable
//! at build time with -DATHENAK_DYNGR_FACE_MIN_BLOCKS_PER_SM=N; to pin the choice at
//! runtime instead (for a reproducible run or a bug report) set the environment
//! variable ATHENAK_LAUNCH_MIN_BLOCKS.
//!
//! Measured on NVIDIA V100-SXM2 (sm_70, 65536 registers/SM), 10-rank BBH production run:
//!   MinBlocksPerSM = 2  ->  REG 128, 1008 B stack/thread, 25.0% occupancy
//!   MinBlocksPerSM = 1  ->  REG 255,  608 B stack/thread, 12.5% occupancy, +6.5% faster
//! The spill traffic at 128 registers was 3.3x the real global-memory traffic and 75% of
//! all warp stalls, which is why the lower-occupancy choice wins there.  That is the
//! measurement the automatic rule is required to reproduce on this hardware.
#ifndef ATHENAK_DYNGR_FACE_MIN_BLOCKS_PER_SM
#define ATHENAK_DYNGR_FACE_MIN_BLOCKS_PER_SM 1
#endif
constexpr int kDynGRFaceMaxThreads     = 256;
constexpr int kDynGRFaceMinBlocksPerSM = ATHENAK_DYNGR_FACE_MIN_BLOCKS_PER_SM;

enum class DynGRMHD_RSolver {llf_dyngr, hlle_dyngr,
                             hlld_dyngr};             // Riemann solvers for dynamical GR
enum class DynGRMHD_EOS {eos_ideal, eos_piecewise_poly,
                      eos_compose, eos_hybrid};        // EOS policies for dynamical GR
enum class DynGRMHD_Error {reset_floor};               // Error policies for dynamical GR

//----------------------------------------------------------------------------------------
//! \struct DynGRMHDTaskIDs
//  \brief container to hold TaskIDs of all dyngr tasks

struct DynGRMHDTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID settmunu;
  TaskID sendf;
  TaskID recvf;
  TaskID expl;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
  TaskID zrecv;
  TaskID zcopyu;
  TaskID zmattersrc;
  TaskID zcrhsdep;
  TaskID zcrhs;
  TaskID zsombc;
  TaskID zexpl;
  TaskID zsendu;
  TaskID zrecvu;
  TaskID znewdt;
  TaskID zbcs;
  TaskID zalgc;
  TaskID z4tad;
  TaskID zadmc;
  TaskID zclear;
  TaskID zrestu;
  TaskID zadep;
  TaskID c2pdep;
  TaskID rkdep;
};

namespace dyngr {

class DynGRMHD {
 public:
  DynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);
  virtual ~DynGRMHD();

  // container to hold names of TaskIDs
  DynGRMHDTaskIDs id;

  TaskStatus SetTmunu(Driver *d, int stage);
  TaskStatus SetADMVariables(Driver *d, int stage);
  TaskStatus UpdateExcisionMasks(Driver *d, int stage);
  TaskStatus SetADMVariablesAtStageEnd(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  void SetADMVariablesAtTime(Real time);

  // functions

  virtual void QueueDynGRMHDTasks() = 0;

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage) = 0;
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke) = 0;
  // Interior-first split of the end-of-stage recovery.  ConToPrimInteriorFirst covers
  // the deep interior and is queued so that it executes while the conserved-variable
  // exchange is in flight; ConToPrimAfterExchange covers exactly the complement (ghost
  // zones plus the shell of interior cells the physical BCs read) and keeps the DAG slot
  // -- and therefore every downstream dependency -- of the old single full-extent pass.
  // Both go through the same virtual ConToPrimBC, i.e. the same kernel and the same
  // template instantiation as the full pass; only the index ranges differ, so every cell
  // is recovered from the same inputs with the same arithmetic.  See dyn_grmhd.cpp.
  TaskStatus ConToPrimInteriorFirst(Driver* pdrive, int stage);
  TaskStatus ConToPrimAfterExchange(Driver* pdrive, int stage);
  // How many layers of INTERIOR cells the physical boundary conditions read (and so must
  // not be modified before MHD_BCS runs).  0 when no physical BC touches u0.
  int PhysicalBCInteriorReadDepth() const;
  // Whether the interior-first split is legal for this run's configuration.
  bool InteriorFirstC2PUsable() const;
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke) = 0;
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke) = 0;

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost) = 0;

  // DynGRMHD policies
  DynGRMHD_RSolver rsolver_method;
  DynGRMHD_RSolver fofc_method;
  DynGRMHD_EOS eos_policy;
  DynGRMHD_Error error_policy;

  bool StoreTemperature() const { return store_temperature; }
  virtual Real BaryonMass() const = 0;

  // Storage for temperature
  DvceArray5D<Real> temperature;

 protected:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int scratch_level;        // GPU scratch level for flux and source calculations
  bool enforce_maximum;     // enforce local maximum principle during FOFC
  Real dmp_M;               // threshold multiplier for discrete maximum principle.
  bool fixed_evolution;     // Disable mhd evolution
  bool scalar_pplimiter;    // Apply positivity preserving limiter on scalar
  bool store_temperature;    // cache full-grid primitive temperature only when output needs it
  DvceArray1D<Real> fofc_eos_min_y;
  DvceArray1D<Real> fofc_eos_max_y;
};

template<class EOSPolicy, class ErrorPolicy>
class DynGRMHDPS : public DynGRMHD {
 public:
  DynGRMHDPS(MeshBlockPack *ppack, ParameterInput *pin) :
      DynGRMHD(ppack, pin), eos("mhd", ppack, pin) {
    int nscal = eos.ps.GetEOS().GetNSpecies();
    if (pmy_pack != nullptr && pmy_pack->pmhd != nullptr) {
      nscal = pmy_pack->pmhd->nscalars;
    }
    const int nbounds = (nscal > 0) ? nscal : 1;
    Kokkos::realloc(fofc_eos_min_y, nbounds);
    Kokkos::realloc(fofc_eos_max_y, nbounds);
    if (nscal > 0) {
      auto h_min_y = Kokkos::create_mirror_view(fofc_eos_min_y);
      auto h_max_y = Kokkos::create_mirror_view(fofc_eos_max_y);
      for (int n = 0; n < nscal; ++n) {
        h_min_y(n) = eos.ps.GetEOS().GetMinimumSpeciesFraction(n);
        h_max_y(n) = eos.ps.GetEOS().GetMaximumSpeciesFraction(n);
      }
      Kokkos::deep_copy(fofc_eos_min_y, h_min_y);
      Kokkos::deep_copy(fofc_eos_max_y, h_max_y);
    }
  }
  virtual ~DynGRMHDPS() {}

  // Dynamical EOS
  PrimitiveSolverHydro<EOSPolicy, ErrorPolicy> eos;

  // CalculateFluxes function templated over Riemann Solvers.
  template<DynGRMHD_RSolver T>
  TaskStatus CalcFluxes(Driver *d, int stage);

  // FOFC is templated over the FIRST-ORDER CORRECTOR (<mhd>/fofc_method), not over the
  // main Riemann solver: only llf_dyngr and hlle_dyngr are ever instantiated.
  template<DynGRMHD_RSolver fofc_method_>
  void FOFC(Driver *d, int stage);

  void BuildFOFCTrial(Driver *d, int stage, int il, int iu, int jl, int ju, int kl,
                      int ku);

  // functions
  virtual void QueueDynGRMHDTasks();

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage);
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke);
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke);
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke);
  Real BaryonMass() const override { return eos.ps.GetEOS().GetBaryonMass(); }

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost);

  template<int NGHOST>
  void AddCoordTermsEOS(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const Real dt, DvceArray5D<Real> &u0);
};

// Factory function for generating DynGRMHD based on parameter input.
// Used to make the MeshBlockPack creation a little bit cleaner.
DynGRMHD* BuildDynGRMHD(MeshBlockPack *ppack, ParameterInput *pin);

} // namespace dyngr

#endif  // DYN_GRMHD_DYN_GRMHD_HPP_
