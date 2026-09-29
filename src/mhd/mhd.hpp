#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

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
class Resistivity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class OrbitalAdvectionFC;
class ShearingBoxCC;
class ShearingBoxFC;
class Driver;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
                        llf_sr, hlle_sr, hlld_sr,       // SR
                        llf_gr, hlle_gr, hlld_gr};       // GR

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID savest;
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID prep_fofc_flags;
  TaskID send_fofc_flags;
  TaskID recv_fofc_flags;
  TaskID replace_fofc;
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
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID sendb_oa;
  TaskID recvb_oa;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID sendb_shr;
  TaskID recvb_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of mhd variables (5/4 for ideal/isothermal EOS)
  int nscalars;            // number of passive scalars
  int naux = 0;            // number of hidden auxiliary MHD variables
  int nvars = 0;           // total MHD-carried variables = nmhd+nscalars+naux
  int dual_energy_idx = -1;
  bool use_dual_energy = false;
  // Which auxiliary the formalism carries.  The non-relativistic flavour advects an
  // internal-energy DENSITY and needs an operator-split p dV update, with the interface
  // velocities the Riemann solver produced; the GR flavour advects the adiabat
  // kappa = p/rho^Gamma, which is an exact conservation law for smooth flow and so needs
  // neither a source term nor the face-velocity buffer.  Both share eta1/eta2 and the
  // one extra conserved column.
  bool dual_energy_pdv = false;
  bool dual_energy_needs_init = false;
  Real dual_energy_eta1 = 1.0e-3;
  Real dual_energy_eta2 = 1.0e-1;
  bool time_evolving = false;
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;    // primitive variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  MeshBoundaryValuesCC *pbval_u;
  MeshBoundaryValuesFC *pbval_b;
  MeshBoundaryValuesCC *pbval_fofc = nullptr;
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  OrbitalAdvectionFC *porb_b = nullptr;
  ShearingBoxCC *psbox_u = nullptr;
  ShearingBoxFC *psbox_b = nullptr;

  // Object(s) for extra physics (viscosity, resistivity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  // Face fluxes of the conserved variables, stored on the band the flux kernels fill:
  // one face beyond the active domain on each side along the face normal (FOFC replaces
  // the flux at faces is-1 and ie+2) and one cell beyond it transversely (CT needs the
  // area-averaged EMFs at every transverse cell edge), so x1f is
  // (nmb, nvars, nx3+2, nx2+2, nx1+3) with origin (ks-1, js-1, is-1), and likewise
  // x2f/x3f.  A degenerate dimension keeps its single cell at origin 0.  No writer or
  // reader -- the Riemann and FOFC kernels, the divergence, CornerE, the LAT ledger, the
  // diffusion and gravity terms, the sink excision and the flux-correction exchange --
  // goes further, so the ghost-extended layout held 35% dead storage.  Access through
  // FluxBand(); the exchange takes FluxOrigin().  dual_vf and the CornerE scratch
  // arrays below share the same band and origin.
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  int flux_ko = 0;
  int flux_jo = 0;
  int flux_io = 0;
  BandView5D<Real> FluxBand(const DvceArray5D<Real> &a) const {
    return BandView5D<Real>{a, flux_ko, flux_jo, flux_io};
  }
  BandFaceFld5D<Real> FluxBand(const DvceFaceFld5D<Real> &f) const {
    return BandFaceFld5D<Real>{FluxBand(f.x1f), FluxBand(f.x2f), FluxBand(f.x3f)};
  }
  BandView4D<Real> EmfBand(const DvceArray4D<Real> &a) const {
    return BandView4D<Real>{a, flux_ko, flux_jo, flux_io};
  }
  FaceFldOrigin FluxOrigin() const { return FaceFldOrigin{flux_ko, flux_jo, flux_io}; }
  // Receiver-local union of cells changed by delayed LAT CC reflux or by the adjacent
  // CT face correction.  Delayed C2P/source reconciliation must use this cell mask rather
  // than reprocessing every cell in a receiver MeshBlock.
  // Shared one-cell mask used by delayed LAT correction and by the conservative
  // coarse/fine FOFC order exchange.  The latter is allocated even when LAT is disabled.
  DvceArray5D<Real> lat_correction_mask;
  DvceArray5D<Real> coarse_fofc_mask;
  DvceFaceFld5D<Real> dual_vf;  // face velocities used by dual-energy formalism
  DvceArray4D<Real> dual_etot_max;    // cached local max(total energy) for sync
  // Cached excision membership for sync; 0/1 flag, so one byte per cell suffices.
  DvceArray4D<std::int8_t> dual_excise_mask;
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS;
  // each lives on the band of the face flux it accompanies (see uflx), EmfBand() access
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  // global per-face L/R buffers for the split-kernel flux path: primitive variables
  // carried by MHD and reconstructed cell-centered B-field (3 components)
  DvceArray5D<Real> wl3d, wr3d;
  DvceArray5D<Real> bl3d, br3d;
  int split_recon_chunk_nmb = 32;
  DualArray1D<int> sink_block_indices;  // reusable active blocks intersecting the sink
  Real dtnew;
  // Optional fixed-spacetime GRMHD fast-magnetosonic CFL estimate.  The default
  // remains false so the legacy unit-coordinate-speed GR timestep is untouched.
  bool gr_dt = false;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;
  DualArray1D<Real> wbcc_saved_dt;  // elapsed time represented by each saved state

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  DvceArray5D<bool> fofc_scal;  // flag to indicate if FOFC for scalar is needed
  bool use_fofc = false;   // flag to enable FOFC

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // Two-phase form used when a radiation module hangs its graph between the halves.  Part A
  // ends at id.srctrms -- the point where the RK update of u0, the corner-E exchange, the
  // CT update of b0, the dual-energy step and the source terms have all run, so u0(IEN)
  // and b0 are stage-consistent and the couple's emag subtraction (B2) is well defined.
  // Part B resumes at the state exchange with BOTH SendU_OA and SendB_OA gated on the
  // radiation couple that has just rewritten the gas u0.  Unlike hydro, AssembleMHDTasks
  // is the COMPOSITION of the two halves, so the radiation-free graph is identical by
  // construction (TaskIDs come from AddTask call order).
  void AssembleMHDTasksPartA(std::map<std::string, std::shared_ptr<TaskList>> tl);
  void AssembleMHDTasksPartB(std::map<std::string, std::shared_ptr<TaskList>> tl,
                             TaskID rad_gate);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus PrepareFOFCFlags(Driver *d, int stage);
  TaskStatus SendFOFCFlags(Driver *d, int stage);
  TaskStatus RecvFOFCFlags(Driver *d, int stage);
  TaskStatus ReplaceCanonicalFOFC(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  // RK base weights of the stage for u0 (RKUpdate), the face fields (CT) and every
  // FOFC candidate.  The driver's (gam0, gam1).
  void TransportStageWeights(const Driver *d, int stage, Real &gam0, Real &gam1) const;
  TaskStatus DualEnergyStep(Driver *d, int stage);
  TaskStatus MHDSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus EField(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus SendB_OA(Driver *d, int stage);
  TaskStatus RecvB_OA(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendB_Shr(Driver *d, int stage);
  TaskStatus RecvB_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);
  void ApplyExcisionSinkBoundary(const Real t);

  // first-order flux correction
  void FOFC(Driver *d, int stage);
  void BuildFOFCTrial(Driver *d, int stage, int il, int iu, int jl, int ju, int kl,
                      int ku);

  // dual-energy support
  void InitializeDualEnergyFieldFromTotal();
  void InitializeDualEnergyFieldFromAdiabat();
  void SynchronizeDualEnergyFieldFromAdiabat();
  void ApplyDualEnergyFormalism(const Real dt);
  void SynchronizeDualEnergyFieldFromTotal();
  void SynchronizeRestrictedDualEnergyField();
  void SynchronizeRestrictedDualEnergyField(DvceArray5D<Real> &coarse_cons,
                                            DvceArray5D<Real> &coarse_prim);
  void RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                    const int new_gids, const int new_nmb_local);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false,
                              bool allow_shrink = false);

  // FOFC trial state.  Under dynamical GR its one user, DynGRMHDPS::FOFC, touches only
  // the cell band [ks-1,ke+1]x[js-1,je+1]x[is-1,ie+1] of the face fluxes, so there it is
  // stored on that band with the flux origin; the other backends keep the ghost-extended
  // layout at origin 0.  Access through FofcTrialBand().
  DvceArray5D<Real> utest, bcctest;
  int fofc_trial_ko = 0;
  int fofc_trial_jo = 0;
  int fofc_trial_io = 0;
  BandView5D<Real> FofcTrialBand(const DvceArray5D<Real> &a) const {
    return BandView5D<Real>{a, fofc_trial_ko, fofc_trial_jo, fofc_trial_io};
  }

 private:
  bool FOFCMaskExchangeEnabled() const;
  void ProlongateFaceFieldsFC_();
  bool fofc_replacement_only_ = false;
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // cell-centered electric fields for CornerE, on the cells [ks-1,ke+1]x[js-1,je+1]x
  // [is-1,ie+1] that the SG07 corner average reads (the flux band origin, EmfBand())
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
