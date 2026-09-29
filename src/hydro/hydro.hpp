#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class ShearingBoxCC;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, roe,    // non-relativistic
                          llf_sr, hlle_sr, hllc_sr,        // SR
                          llf_gr, hlle_gr};                // GR

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks

struct HydroTaskIDs {
  TaskID irecv;
  TaskID statefix_pre;
  TaskID statefix;
  TaskID statefix_post;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID duale;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
  TaskID latdense;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  ReconstructionMethod recon_method;
  Hydro_RSolver rsolver_method;
  EquationOfState *peos;  // chosen EOS

  int nhydro;             // number of hydro variables (5/4 for ideal/isothermal EOS)
  int nscalars;           // number of user passive scalars
  int naux = 0;           // number of hidden auxiliary hydro variables
  int nvars = 0;          // total hydro-carried variables = nhydro+nscalars+naux
  int dual_energy_idx = -1;
  bool use_dual_energy = false;
  // Flavour: the non-relativistic internal-energy auxiliary with its p dV step, or the
  // GR adiabat kappa = p/rho^Gamma advected as a passive scalar (false).
  bool dual_energy_pdv = false;
  bool dual_energy_needs_init = false;
  Real dual_energy_eta1 = 1.0e-3;
  // Weight applied to floor-induced energy changes recorded by ConsToPrim: the
  // contribution of the current RK stage's state to the completed step (1 outside
  // the RK stages).
  Real floor_energy_weight = 1.0;
  // Device-side per-block floor energy tally for EOS paths whose C2P is a plain
  // parallel_for (tabulated EOS): filled by atomic adds only while a gravity energy
  // window is open (floor_energy_ledger), reduced and zeroed at the window end.
  bool floor_energy_ledger = false;
  DvceArray1D<Real> floor_energy_block;
  Real dual_energy_eta2 = 1.0e-1;
  bool time_evolving = false;
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;  // primitive variables on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  ShearingBoxCC *psbox_u = nullptr;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step
  DvceArray5D<Real> coarse_u1;  // LAT start-state restriction for fine-to-coarse bvals
  DvceArray5D<Real> lat_u_stage1;  // RK2 stage-1 endpoint for LAT dense bvals
  bool lat_dense_output_enabled = false;
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceFaceFld5D<Real> lat_reflux;  // delayed LAT fine/coarse flux correction
  DvceFaceFld5D<Real> lat_dual_vf_reflux;  // delayed LAT dual-energy div(v) correction
  DvceFaceFld5D<Real> lat_reflux_theta;  // admissible fraction of the pending reflux
  // Gate record of the gravitational work carried by the pending reflux, laid out by
  // lat_reflux::GravWorkComponent.  Allocated only with a gravity source term (self,
  // external BH or sink particles).
  DvceFaceFld5D<Real> lat_grav_reflux;
  bool lat_grav_reflux_allocated = false;
  DvceFaceFld5D<Real> dual_vf;  // face velocities used by dual-energy formalism
  // tau/D published by the GR inversion for the eta2 pass, which has no metric.
  DvceArray4D<Real> dual_etot_max;
  // Cached excision membership for sync; 0/1 flag, so one byte per cell suffices.
  DvceArray4D<std::int8_t> dual_excise_mask;
  DualArray1D<int> sink_block_indices;  // reusable active blocks intersecting the sink
  DualArray1D<Real> dtnew_eachmb;  // local hydro CFL timestep for each MeshBlock
  Real dtnew_hydro_cfl;
  Real dtnew;

  // Per-chunk L/R primitive buffers used by the split-kernel flux path.
  // Shaped (split_recon_chunk_nmb, nvars, ncells3, ncells2, ncells1).
  // The buffers cover the full cell range including ghost zones and are reused
  // sequentially across chunks and coordinate directions.
  DvceArray5D<Real> wl3d;
  DvceArray5D<Real> wr3d;
  int split_recon_chunk_nmb = 32;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC
  DvceArray5D<Real> utest;  // scratch array for FOFC

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // Two-phase form used when a radiation module hangs its graph between the halves.  Part A ends at
  // id.statefix (the gate the radiation graph waits on); part B resumes at the state
  // exchange, gated on the radiation couple that has just rewritten the gas u0.
  void AssembleHydroTasksPartA(std::map<std::string, std::shared_ptr<TaskList>> tl);
  void AssembleHydroTasksPartB(std::map<std::string, std::shared_ptr<TaskList>> tl,
                               TaskID rad_gate);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus HydroStateFixup(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus DualEnergyStep(Driver *d, int stage);
  TaskStatus HydroSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus ConToPrimGhostBands(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize
  TaskStatus SaveLATDenseOutput(Driver *d, int stage);

  void ConfigureLATDenseOutput(bool enabled);
  void ResetLATFluxCorrection();
  void AccumulateLATCoarseFluxes(Driver *d, int stage);
  bool ApplyLATFluxCorrection(Real sync_time, int sync_factor, int completed_tick);

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
  void CalculateFluxes(Driver *d, int stage);
  void ApplyExcisionSinkBoundary(const Real t);

  // first-order flux correction
  void FOFC(Driver *d, int stage);
  void BuildFOFCTrial(Driver *d, int stage, int il, int iu, int jl, int ju, int kl,
                      int ku);
  void StashFOFCEdgeFlags(Driver *d, int stage);

  // dual-energy support
  void InitializeDualEnergyFieldFromTotal();
  void InitializeDualEnergyFieldFromAdiabat();
  void ApplyDualEnergyFormalism(const Real dt);
  void SynchronizeDualEnergyFieldFromTotal(bool ghost_bands_only = false);
  void SynchronizeDualEnergyFieldFromAdiabat();
  void SynchronizeRestrictedDualEnergyField(bool lat_active_only = false);
  void RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                    const int new_gids, const int new_nmb_local);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false,
                              bool allow_shrink = false);

 private:
  bool lat_requested_ = false;
  bool lat_correction_storage_ = false;  // LAT delayed-reflux registers are allocated
  bool flux_recv_complete_ = false;
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
