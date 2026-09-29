#ifndef EOS_PRIMITIVE_SOLVER_HYD_HPP_
#define EOS_PRIMITIVE_SOLVER_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file primitive_solver_hyd.hpp
//  \brief Contains the template class for PrimitiveSolverHydro, which is independent
//  of the EquationOfState class used elsewhere in AthenaK.

// C headers
#include <float.h>
#include <math.h>

// C++ headers
#include <string>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>

// PrimitiveSolver headers
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/logs.hpp"
#include "eos/drift_frame_floor.hpp"

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "pgen/pgen.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn bool PSHydroEventLogRequested(ParameterInput*)
//! \brief whether this run configures an event-log output.
//!
//! The C2P event counters are consumed by exactly one thing, EventLogOutput in
//! src/outputs/eventlog.cpp, constructed only for an <outputN> block with file_type=log.
//! Asking the deck that question is general -- any deck that wants the counters gets them
//! -- rather than keying on a particular problem or machine.
//!
//! Deliberately a copy of the identical helper in src/eos/ideal_grmhd.cpp:32-40: that is
//! a translation unit and this is a header, and promoting it would touch files other
//! tracks are editing.  Worth unifying when the tree is quiet.
inline bool PSHydroEventLogRequested(ParameterInput *pin) {
  if (pin == nullptr) return false;
  for (const auto &block : pin->block) {
    if (block.block_name.compare(0, 6, "output") != 0) continue;
    for (const auto &line : block.line) {
      if (line.param_name == "file_type" && line.param_value == "log") return true;
    }
  }
  return false;
}

template<class EOSPolicy, class ErrorPolicy>
class PrimitiveSolverHydro {
 protected:
  Real gamma_max;
  Real sigma_max;
  Real bsq_over_u_max;

  struct GRPrimitiveLimitResult {
    bool kinematic_adjusted = false;      // the Lorentz ceiling rescaled the velocity
    bool thermodynamic_adjusted = false;  // the sigma or b^2/u ceiling raised rho or u
    bool density_adjusted = false;        // rho changed: the sigma ceiling loaded mass

    KOKKOS_INLINE_FUNCTION
    bool Any() const { return kinematic_adjusted || thermodynamic_adjusted; }
  };

  void SetPolicyParams(std::string block, ParameterInput *pin) {
    // Parameters for an ideal gas
    if constexpr(std::is_same_v<Primitive::IdealGas, EOSPolicy>) {
      ps.GetEOSMutable().SetGamma(pin->GetOrAddReal(block, "gamma", 5.0/3.0));
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 0));
    }
    // Parameters for a piecewise polytrope
    if constexpr(std::is_same_v<Primitive::PiecewisePolytrope, EOSPolicy>) {
      bool result = ps.GetEOSMutable().ReadParametersFromInput(block, pin);
      if (!result) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "There was an error while constructing the EOS."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    // Parameters for CompOSE EoS
    if constexpr (
         std::is_same_v<Primitive::EOSCompOSE<Primitive::NormalLogs>, EOSPolicy> ||
         std::is_same_v<Primitive::EOSCompOSE<Primitive::NQTLogs>, EOSPolicy>) {
      // Get and set number of scalars in table. This will currently fail if not 1.
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 1));
      std::string units = pin->GetOrAddString(block, "units", "geometric_solar");
      if (!units.compare("geometric_solar")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
      } else if (!units.compare("geometric_kilometer")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricKilometer());
      } else if (!units.compare("nuclear")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeNuclear());
      } else if (!units.compare("cgs")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeCGS());
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unknown unit system " << units << " requested."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Get table filename, then read the table,
      std::string fname = pin->GetString(block, "table");
      ps.GetEOSMutable().ReadTableFromFile(fname);

      // Ensure table was read properly
      assert(ps.GetEOSMutable().IsInitialized());
    }
        // Parameters for Hybrid EoS
    if constexpr (
         std::is_same_v<Primitive::EOSHybrid<Primitive::NormalLogs>, EOSPolicy> ||
         std::is_same_v<Primitive::EOSHybrid<Primitive::NQTLogs>, EOSPolicy>) {
      // Get and set number of scalars in table. This will currently fail if not 0.
      ps.GetEOSMutable().SetThermalGamma(pin->GetOrAddReal(block, "gamma_thermal",
                                         5.0/3.0));
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 0));
      std::string units = pin->GetOrAddString(block, "units", "geometric_solar");
      if (!units.compare("geometric_solar")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
      } else if (!units.compare("geometric_kilometer")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricKilometer());
      } else if (!units.compare("nuclear")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeNuclear());
      } else if (!units.compare("cgs")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeCGS());
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unknown unit system " << units << " requested."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Get table filename, then read the table,
      std::string fname = pin->GetString(block, "table");
      ps.GetEOSMutable().ReadTableFromFile(fname);

      // Ensure table was read properly
      assert(ps.GetEOSMutable().IsInitialized());
    }
  }

  //! \fn EnthalpyDensityDevice
  //! \brief rho*h for one cell, the currency every floor below is measured in.
  KOKKOS_INLINE_FUNCTION
  static Real EnthalpyDensityDevice(
      const Primitive::EOS<EOSPolicy, ErrorPolicy> &eos, Real prim[NPRIM]) {
    return prim[PRH]*eos.GetBaryonMass()*
           eos.GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
  }

  //! \fn FloorSetPrimitivesDevice
  //! \brief The primitive floor for a state whose n and P were set independently (pgen
  //! initial data, user-BC ghosts), not recovered by an inversion.  There T = P/n is not
  //! a property of the gas: where n undershot the floor (to zero, in the limit) carrying
  //! T to the raised density scales P by n_new/n, and n = 0 with P > 0 gives T = inf and
  //! non-finite conserved variables.  So where the floor raised n, P is kept (raised to
  //! the atmosphere pressure at the new n if below) and T is taken from it -- what
  //! eos_floor::DriftFrameApplyPrimitiveFloor does for reconstructed faces.  Where n was
  //! not raised the floor is ApplyPrimitiveFloor unchanged.
  KOKKOS_INLINE_FUNCTION
  static bool FloorSetPrimitivesDevice(
      const Primitive::EOS<EOSPolicy, ErrorPolicy> &eos, Real prim[NPRIM]) {
    const Real n_in = prim[PRH];
    const Real p_in = prim[PPR];
    const bool floored = eos.ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR],
                                                 prim[PTM], &prim[PYF]);
    if (floored && (prim[PRH] > n_in)) {
      const Real T_atm = eos.GetTemperatureFloor();
      const Real p_atm = eos.GetPressure(prim[PRH], T_atm, &prim[PYF]);
      if (isfinite(p_in) && (p_in > p_atm)) {
        prim[PPR] = p_in;
        prim[PTM] = eos.GetTemperatureFromP(prim[PRH], p_in, &prim[PYF]);
      } else {
        prim[PPR] = p_atm;
        prim[PTM] = T_atm;
      }
    }
    return floored;
  }

  //! \fn ReinjectFlooredMassDevice
  //! \brief The single drift-frame velocity re-solve that answers every floor a cell hit
  //! in this step.  Call it once, after all of them, with the enthalpy density the raw
  //! inversion recovered; see eos/drift_frame_floor.hpp for why the drift frame and not
  //! the coordinate frame.  The map is nonlinear, so re-solving after each individual
  //! floor is not the same thing and would over-correct a cell that hit two of them.
  KOKKOS_INLINE_FUNCTION
  static void ReinjectFlooredMassDevice(
      const Primitive::EOS<EOSPolicy, ErrorPolicy> &eos, const Real w_prefloor,
      Real prim[NPRIM], const Real b_u[NMAG], const Real g3d[NSPMETRIC]) {
    if (!(w_prefloor > 0.0) || !isfinite(w_prefloor)) {
      return;
    }
    eos_floor::DriftFrameReinjectVelocity(g3d, b_u, w_prefloor,
                                          EnthalpyDensityDevice(eos, prim),
                                          &prim[PVX]);
  }

  //! \fn ApplyGRPrimitiveLimitsDevice
  //! \brief Every lower bound a GRMHD cell has to respect, collected and then applied
  //! once.
  //!
  //! The bounds are of two kinds and they routinely fire together in the polar funnel,
  //! where the density sits a few per cent above the floor and the field is strong:
  //! KORAL's magnetization ceilings B2RHORATIOMAX and B2UURATIOMAX (rho >= b^2/sigma_max
  //! and u >= b^2/bsq_over_u_max), and the solver's own uniform floor, which has already
  //! run inside ConToPrim.  Taking the maximum first and raising rho and u once each is
  //! what KHARMA does (`P(RHO) = max(rho, rhoflr_max); P(UU) = max(uu, uflr_max);`), and
  //! it matters because the velocity that comes with the injected material is then
  //! decided a single time, by the caller's drift-frame re-solve, instead of once per
  //! bound.
  //!
  //! The Lorentz ceiling is deliberately not part of that: it rescales the velocity
  //! rather than injecting anything, so it runs first, both because every b^2 below
  //! depends on W and so that the re-solve is handed a sane one.
  //!
  //! cold_mass: the cell's rest mass is the conserved floor's (SolverResult::
  //! cons_mass_floor), so its T is no gas temperature and the loaded mass comes cold.
  KOKKOS_INLINE_FUNCTION
  static GRPrimitiveLimitResult ApplyGRPrimitiveLimitsDevice(
      const Primitive::EOS<EOSPolicy, ErrorPolicy> &eos,
      const Real gamma_max, const Real sigma_max, const Real bsq_over_u_max,
      Real prim[NPRIM], const Real b_u[NMAG],
      const Real g3d[NSPMETRIC], const bool cold_mass = false) {
    GRPrimitiveLimitResult result;
    const Real mb = eos.GetBaryonMass();
    const Real n_in = prim[PRH];

    Real *Wv_u = &prim[PVX];
    Real Wv_d[3];
    Primitive::LowerVector(Wv_d, Wv_u, g3d);
    Real Wvsq = Primitive::Contract(Wv_u, Wv_d);

    if (isfinite(gamma_max) && (gamma_max > 1.0) && isfinite(Wvsq)) {
      const Real Wvsq_max = SQR(gamma_max) - 1.0;
      if (Wvsq > Wvsq_max) {
        const Real factor = sqrt(Wvsq_max/Wvsq);
        prim[PVX] *= factor;
        prim[PVY] *= factor;
        prim[PVZ] *= factor;
        result.kinematic_adjusted = true;

        Wv_u = &prim[PVX];
        Primitive::LowerVector(Wv_d, Wv_u, g3d);
        Wvsq = Primitive::Contract(Wv_u, Wv_d);
      }
    }

    // Both magnetization limits are measured against the same comoving
    // b^2 = b^mu b_mu that KORAL uses for B2RHORATIOMAX and B2UURATIOMAX, so it is
    // computed once and shared.  b^2 depends only on B^i and the velocity, and nothing
    // below changes either, so one evaluation stays valid for both.
    const bool limit_sigma = isfinite(sigma_max) && (sigma_max > 0.0);
    const bool limit_bsq_over_u = isfinite(bsq_over_u_max) && (bsq_over_u_max > 0.0);
    Real bsq = 0.0;
    bool have_bsq = false;
    if ((limit_sigma || limit_bsq_over_u) && isfinite(Wvsq)) {
      const Real W = sqrt(1.0 + Wvsq);
      const Real iW = 1.0/W;

      Real B_d[3];
      Primitive::LowerVector(B_d, b_u, g3d);
      const Real Bsq = Primitive::Contract(b_u, B_d);
      const Real v_d[3] = {Wv_d[0]*iW, Wv_d[1]*iW, Wv_d[2]*iW};
      const Real Bv = Primitive::Contract(b_u, v_d);
      bsq = Bsq*SQR(iW) + SQR(Bv);
      have_bsq = isfinite(bsq);
    }

    // Mass: the magnetization ceiling is the only bound here that depends on the field,
    // and it is raised once.  T is what the cell had; the pressure that goes with the new
    // density is not.  Where the cell's rest mass is the conserved floor's (cold_mass), T
    // is the leftover tau over the floor's mass (ConToPrim), and loading b^2/sigma_max at
    // it multiplied that leftover by the loaded mass: the loaded mass comes cold there,
    // at the internal energy density the cell had.
    Real rho = prim[PRH]*mb;
    Real rho_min = 0.0;
    if (limit_sigma && have_bsq && (rho > 0.0)) {
      rho_min = bsq/sigma_max;
    }
    if ((rho > 0.0) && isfinite(rho_min) && (rho_min > rho)) {
      const Real u_kept =
          cold_mass ? eos.GetEnergy(prim[PRH], prim[PTM], &prim[PYF]) - rho : 0.0;
      prim[PRH] = rho_min/mb;
      if (cold_mass) {
        prim[PTM] = fmax(eos.GetTemperatureFromE(prim[PRH], rho_min + u_kept, &prim[PYF]),
                         eos.GetTemperatureFloor());
      }
      prim[PPR] = eos.GetPressure(prim[PRH], prim[PTM], &prim[PYF]);
      rho = prim[PRH]*mb;
      result.thermodynamic_adjusted = true;
    }

    // Energy, carried on T so that it is one raise.
    //
    // KORAL's B2UURATIOMAX (choices.h; applied in check_floors_mhd, u2p.c) is the partner
    // of the mass loading above.  Mass loading alone does not save a magnetically
    // dominated cell whose gas energy is being drained -- by radiative cooling in an M1
    // run, where the emissivity does not care how strong the field is -- because b^2 is
    // set by the field and keeps growing relative to u until the inversion has no gas
    // left to recover.  KORAL rescales u by fuu = b^2/(B2UURATIOMAX*u), which lands
    // exactly on u = b^2/B2UURATIOMAX, i.e. the same end state as flooring u here.
    //
    // Unlike KORAL, which computes both floor factors from one pre-floor state, this
    // ratio is measured *after* the mass loading.  KORAL evolves u as an independent
    // primitive, so loading mass there leaves u alone; here the thermodynamic state is
    // (n, T) and raising n at fixed T already raises u = rho*eps.  Measuring afterwards
    // therefore avoids injecting energy the loading has already supplied, and either
    // ordering ends at b^2/u <= the ceiling, so the weaker intervention is preferred.
    Real T_min = prim[PTM];
    if (limit_bsq_over_u && have_bsq && (rho > 0.0)) {
      const Real u = rho*eos.GetSpecificInternalEnergy(prim[PRH], prim[PTM], &prim[PYF]);
      const Real u_min = bsq/bsq_over_u_max;
      if (isfinite(u) && (u_min > u)) {
        // Carry the extra internal energy on the temperature.  Inverting u -> T through
        // the EOS is exact for every policy, whereas the rescale T *= u_min/u that
        // reproduces KORAL's fuu literally is exact only where u is linear in T at fixed
        // n, i.e. only for the ideal gas.  The inversion wants the rest mass included,
        // hence rho + u_min; the target is above the current energy, so it is only ever
        // asked to walk a valid state upwards.
        T_min = fmax(T_min, eos.GetTemperatureFromE(prim[PRH], rho + u_min, &prim[PYF]));
      }
    }
    if (isfinite(T_min) && (T_min > prim[PTM])) {
      prim[PTM] = T_min;
      prim[PPR] = eos.GetPressure(prim[PRH], prim[PTM], &prim[PYF]);
      result.thermodynamic_adjusted = true;
    }

    if (result.thermodynamic_adjusted) {
      eos.ApplyDensityLimits(prim[PRH]);
      eos.ApplyPressureLimits(prim[PPR], prim[PRH], &prim[PYF]);
      prim[PTM] = eos.GetTemperatureFromP(prim[PRH], prim[PPR], &prim[PYF]);
      eos.ApplyPrimitiveFloor(prim[PRH], &prim[PVX], prim[PPR],
                              prim[PTM], &prim[PYF]);
    }
    // The b^2/u ceiling alone leaves n as it found it: it raises T at fixed n.
    result.density_adjusted = (prim[PRH] != n_in);

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool ApplyGRPrimitiveLimits(Real prim[NPRIM], const Real b_u[NMAG],
                              const Real g3d[NSPMETRIC]) const {
    const Real w_prefloor = EnthalpyDensityDevice(ps.GetEOS(), prim);
    const auto limits = ApplyGRPrimitiveLimitsDevice(
        ps.GetEOS(), gamma_max, sigma_max, bsq_over_u_max, prim, b_u, g3d);
    if (limits.thermodynamic_adjusted) {
      ReinjectFlooredMassDevice(ps.GetEOS(), w_prefloor, prim, b_u, g3d);
    }
    return limits.Any();
  }

 public:
  Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> ps;
  MeshBlockPack* pmy_pack;
  unsigned int nerrs;
  unsigned int errcap;
  // Destinations for the C2P event reduction.  A Kokkos reduction whose result lands in a
  // HOST scalar is synchronous: Kokkos fences and copies back, so the host blocks on
  // every ConsToPrim -- and on the dyn-GR path that is up to 9 calls per RK stage (6 from
  // SetADMVariables' ConToPrimBC, plus PreFluxC2P, the FOFC test pass and C2P).  Each of
  // those stalls drains the GPU queue to empty.  Reducing into a device View instead is
  // asynchronous, which removes the stall without touching a single line of the kernel.
  //
  // Keeping the kernel byte-identical is the whole point: this kernel's arithmetic is
  // codegen-sensitive at the 1-ULP level (an earlier attempt that rewrote the reduction
  // as a parallel_for with atomics changed the lambda signature and moved ~32 cells by
  // one ulp at t=0), so the reduction machinery, the lambda signature and the body are
  // all left exactly as they were.  Only the reducer's destination changes.
  Kokkos::View<int, DevMemSpace> c2p_nerr_d_;
  Kokkos::View<int, DevMemSpace> c2p_limits_d_;
  bool track_event_counters_;

  // Launches a single RK stage can make through this kernel, used to slice the error
  // report budget when it cannot be accumulated across launches (see ConsToPrimImpl).
  static constexpr int kC2PLaunchesPerStage = 32;

  //! \brief GRMHD primitive ceilings applied after every production C2P.  Radiation's
  //! coupled primitive solve uses these read-only values to reject a Newton trial that
  //! the immediately following production C2P would otherwise rewrite.
  Real BsqOverUMax() const { return bsq_over_u_max; }
  Real SigmaMax() const { return sigma_max; }

  PrimitiveSolverHydro(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
//        pmy_pack(pp), ps{&eos} {
        pmy_pack(pp), nerrs(0),
        gamma_max(std::numeric_limits<Real>::max()),
        sigma_max(std::numeric_limits<Real>::max()),
        bsq_over_u_max(std::numeric_limits<Real>::max()) {
    // Assigned in the body, not the initialiser list: that list is already out of
    // declaration order (gamma_max/sigma_max are declared above nerrs) and adding to it
    // would only make that worse.
    c2p_nerr_d_ = Kokkos::View<int, DevMemSpace>("pshyd_c2p_nerrs");
    c2p_limits_d_ = Kokkos::View<int, DevMemSpace>("pshyd_c2p_limits");
    track_event_counters_ = PSHydroEventLogRequested(pin);
    SetPolicyParams(block, pin);
    int effective_nspecies = ps.GetEOS().GetNSpecies();
    if (pmy_pack != nullptr && pmy_pack->pmhd != nullptr) {
      effective_nspecies = std::max(effective_nspecies, pmy_pack->pmhd->nscalars);
    }
    if (effective_nspecies > MAX_SPECIES) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block << ">/nscalars = "
                << effective_nspecies << " exceeds PrimitiveSolver "
                << "MAX_SPECIES = " << MAX_SPECIES << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ps.GetEOSMutable().SetNSpecies(effective_nspecies);
    Real mb = ps.GetEOS().GetBaryonMass();
    ps.GetEOSMutable().SetDensityFloor(pin->GetOrAddReal(block, "dfloor", (FLT_MIN))/mb);
    ps.GetEOSMutable().SetTemperatureFloor(pin->GetOrAddReal(block, "tfloor", (FLT_MIN)));
    ps.GetEOSMutable().SetThreshold(pin->GetOrAddReal(block, "dthreshold", 1.0));
    ps.tol = pin->GetOrAddReal(block, "c2p_tol", 1e-15);
    ps.GetRootSolverMutable().iterations = pin->GetOrAddInteger(block, "c2p_iter", 50);
    errcap = pin->GetOrAddInteger(block, "c2perrs", 1000);

    // Calculate maximum allowed velocity
    gamma_max = pin->GetOrAddReal(block, "gamma_max", 50.0);
    if (!isfinite(gamma_max) || (gamma_max <= 1.0)) {
      gamma_max = std::numeric_limits<Real>::max();
    }
    Real vmax = 1.0 - 1.0e-15;
    if (isfinite(gamma_max) && (gamma_max > 1.0)) {
      vmax = sqrt(fmax(0.0, 1.0 - 1.0/(gamma_max*gamma_max)));
    }
    ps.GetEOSMutable().SetMaxVelocity(vmax);

    // `sigma_ceiling` is the physical GR magnetization ceiling enforced below using
    // sigma = b^2/rho.  `max_bsq` remains as an optional legacy limit on B^2/D inside
    // PrimitiveSolver itself and is disabled unless explicitly requested.
    if (pin->DoesParameterExist(block, "sigma_ceiling")) {
      sigma_max = pin->GetReal(block, "sigma_ceiling");
      if (!isfinite(sigma_max) || (sigma_max <= 0.0)) {
        sigma_max = std::numeric_limits<Real>::max();
      }
    }
    // `bsq_over_u_ceiling` is KORAL's B2UURATIOMAX (its default there is 100): the
    // internal-energy counterpart of `sigma_ceiling`, enforced below as b^2/u.  Absent or
    // non-positive leaves it disabled, so an input file that does not ask for it keeps
    // the behaviour it had before this limit existed.
    if (pin->DoesParameterExist(block, "bsq_over_u_ceiling")) {
      bsq_over_u_max = pin->GetReal(block, "bsq_over_u_ceiling");
      if (!isfinite(bsq_over_u_max) || (bsq_over_u_max <= 0.0)) {
        bsq_over_u_max = std::numeric_limits<Real>::max();
      }
    }
    if (pin->DoesParameterExist(block, "max_bsq")) {
      ps.GetEOSMutable().SetMaximumMagnetization(pin->GetReal(block, "max_bsq"));
    } else {
      ps.GetEOSMutable().SetMaximumMagnetization(std::numeric_limits<Real>::max());
    }

    for (int n = 0; n < ps.GetEOS().GetNSpecies(); n++) {
      std::stringstream spec_name;
      spec_name << "s" << n << "_atmosphere";
      ps.GetEOSMutable().SetSpeciesAtmosphere(
          pin->GetOrAddReal(block, spec_name.str(), 0.0), n);
    }
  }

  // The prim to con function used on the reconstructed states inside the Riemann solver.
  // It also extracts the primitives into a form usable by PrimitiveSolver.
  KOKKOS_INLINE_FUNCTION
  void PrimToConsPt(const ScrArray2D<Real> &w, const ScrArray2D<Real> &brc,
                    const DvceArray4D<Real> &bx,
                    Real prim_pt[NPRIM], Real cons_pt[NCONS], Real b[NMAG],
                    Real g3d[NSPMETRIC], Real sdetg,
                    const int m, const int k, const int j, const int i,
                    const int &nhyd, const int &nscal,
                    const int ibx, const int iby, const int ibz) const {
    auto &eos = ps.GetEOS();
    Real mb = eos.GetBaryonMass();
    // The magnetic field is densitized, but the PrimToCon call
    // needs undensitized variables.
    Real isdetg = 1.0/sdetg;
    Real bin[NMAG];
    bin[ibx] = bx(m, k, j, i)*isdetg;
    bin[iby] = brc(iby, i)*isdetg;
    bin[ibz] = brc(ibz, i)*isdetg;
    Real prim_pt_old[NPRIM];
    prim_pt[PRH] = prim_pt_old[PRH] = w(IDN, i)/mb;
    prim_pt[PVX] = prim_pt_old[PVX] = w(IVX, i);
    prim_pt[PVY] = prim_pt_old[PVY] = w(IVY, i);
    prim_pt[PVZ] = prim_pt_old[PVZ] = w(IVZ, i);
    for (int n = 0; n < nscal; n++) {
      prim_pt[PYF + n] = prim_pt_old[PYF + n] = w(nhyd + n, i);
    }
    prim_pt[PPR] = prim_pt_old[PPR] = w(IPR, i);

    const auto &ps_cell = ps;
    const auto &eos_cell = ps_cell.GetEOS();

    // Apply the floor to make sure these values are physical.
    // FIXME(JF): Is this needed if the first-order flux correction is enabled?
    prim_pt[PTM] = prim_pt_old[PTM] = eos_cell.GetTemperatureFromP(prim_pt[PRH],
                                        prim_pt[PPR], &prim_pt[PYF]);
    const Real w_prefloor = EnthalpyDensityDevice(eos_cell, prim_pt);
    const bool primitive_floor_adjusted = FloorSetPrimitivesDevice(eos_cell, prim_pt);
    const auto gr_limits = ApplyGRPrimitiveLimitsDevice(
        eos_cell, gamma_max, sigma_max, bsq_over_u_max, prim_pt, bin, g3d);
    if (primitive_floor_adjusted || gr_limits.thermodynamic_adjusted) {
      ReinjectFlooredMassDevice(eos_cell, w_prefloor, prim_pt, bin, g3d);
    }

    ps_cell.PrimToCon(prim_pt, cons_pt, bin, g3d);

    // Check for NaNs
    /*if (CheckForConservedNaNs(cons_pt)) {
      printf("Location: PrimToConsPt\n");
      DumpPrimitiveVars(prim_pt);
    }*/

    // Densitize the variables
    for (int n = 0; n < nhyd + nscal; n++) {
      cons_pt[n] *= sdetg;
    }
    b[ibx] = bx(m, k, j, i);
    b[iby] = brc(iby, i);
    b[ibz] = brc(ibz, i);

    // Previously we checked if the floor was applied and copied these variables back
    // into the original array. However, this is pointless because only the extracted
    // variables in the C-style array are used from this point forward.
  }

  void PrimToCons(DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) {
    //int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
    //auto &size = pmy_pack->pmb->mb_size;
    //auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    auto metric = pmy_pack->padm->GetMetricView();

    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;
    const int nwork = nmb;
    if (nwork <= 0) return;
    auto solver_template = *this;

    Real mb = eos_.GetBaryonMass();
    const Real gamma_max_ = gamma_max;
    const Real sigma_max_ = sigma_max;
    const Real bsq_over_u_max_ = bsq_over_u_max;

    par_for("pshyd_prim2cons", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = a;
      // Extract metric at a single point
      adm::ADMMetricPoint metric_pt{};
      metric.CellMetric(m, k, j, i, metric_pt);
      Real g3d[NSPMETRIC];
      for (int n = 0; n < NSPMETRIC; ++n) g3d[n] = metric_pt.g_dd[n];
      Real sdetg = sqrt(Primitive::GetDeterminant(g3d));

      // The magnetic field is densitized, but the PrimToCon calculation is
      // done with undensitized variables.
      Real b[NMAG] = {bcc(m, IBX, k, j, i)/sdetg,
                      bcc(m, IBY, k, j, i)/sdetg,
                      bcc(m, IBZ, k, j, i)/sdetg};

      // Extract primitive variables at a single point
      Real prim_pt[NPRIM], cons_pt[NCONS];
      prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
      prim_pt[PVX] = prim(m, IVX, k, j, i);
      prim_pt[PVY] = prim(m, IVY, k, j, i);
      prim_pt[PVZ] = prim(m, IVZ, k, j, i);
      for (int n = 0; n < nscal; n++) {
        prim_pt[PYF + n] = prim(m, nhyd + n, k, j, i);
      }
      // FIXME: Debug only! Use specific energy to validate other
      // hydro functions before breaking things.
      //Real e = prim(m, IDN, k, j, i) + prim(m, IEN, k, j, i);
      //prim_pt[PTM] = eos_.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
      //prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      prim_pt[PPR] = prim(m, IPR, k, j, i);

      const auto &ps_cell = solver_template.ps;
      const auto &eos_cell = ps_cell.GetEOS();

      // Apply the floor to make sure these values are physical.
      prim_pt[PTM] =
          eos_cell.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
      const Real w_prefloor = EnthalpyDensityDevice(eos_cell, prim_pt);
      const bool primitive_floor_adjusted = FloorSetPrimitivesDevice(eos_cell, prim_pt);
      const auto gr_limits = ApplyGRPrimitiveLimitsDevice(
          eos_cell, gamma_max_, sigma_max_, bsq_over_u_max_, prim_pt, b, g3d);
      const bool floor = primitive_floor_adjusted || gr_limits.Any();
      if (primitive_floor_adjusted || gr_limits.thermodynamic_adjusted) {
        ReinjectFlooredMassDevice(eos_cell, w_prefloor, prim_pt, b, g3d);
      }

      ps_cell.PrimToCon(prim_pt, cons_pt, b, g3d);

      // Check for NaNs
      if (CheckForConservedNaNs(cons_pt)) {
        Kokkos::printf("Error occurred in PrimToCons at (%d, %d, %d, %d)\n", m, k, j, i);
        DumpPrimitiveVars(prim_pt);
      }

      // Save the densitized conserved variables.
      cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
      cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
      cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
      cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
      cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
      for (int n = 0; n < nscal; n++) {
        cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
      }

      // If we floored the primitive variables, we need to adjust those, too.
      if (floor) {
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }
      }
    });

    return;
  }

  template<bool STORE_TEMPERATURE = true>
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &bfc,
                  DvceArray5D<Real> &bcc0, DvceArray5D<Real> &prim,
                  DvceArray5D<Real> &temperature,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku, bool floors_only=false,
                  const int cko = 0, const int cjo = 0, const int cio = 0) {
    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;
    // Dual energy: eta1 chooses, per cell, between the conserved energy and the
    // advected adiabat.  Both are read below from the conserved array, where the
    // auxiliary is D*kappa, so the ratio is kappa whatever the densitization.
    const bool dual_enabled = pmy_pack->pmhd->use_dual_energy;
    const int dual_idx = pmy_pack->pmhd->dual_energy_idx;
    const Real dual_eta1 = pmy_pack->pmhd->dual_energy_eta1;
    auto dual_tau_ = pmy_pack->pmhd->dual_etot_max;
    const int nwork = nmb;
    if (nwork <= 0) return;
    auto &fofc_ = pmy_pack->pmhd->fofc;

    // Some problem-specific parameters
    auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
    auto &smoothing = pmy_pack->pcoord->coord_data.smooth_excision;
    auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
    auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
    auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
    auto &texcise_ = pmy_pack->pcoord->coord_data.texcise;

    auto metric = pmy_pack->padm->GetMetricView();
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &is = indcs.is;
    int &js = indcs.js;
    int &ks = indcs.ks;
    auto &size = pmy_pack->pmb->mb_size;

    const int ni = (iu - il + 1);
    const int nji = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nwork*nkji;

    const int rank = global_variable::my_rank;
    const int nerrs_ = nerrs;
    // Report budget.  With an event log configured `nerrs` accumulates across the run and
    // this is the original per-RUN cap.  Without one the counter is never read back (that
    // readback is the last host synchronisation here), so `nerrs` stays 0 and the cap
    // would apply per LAUNCH -- and the dyn-GR path issues up to ~22 C2P launches per RK
    // stage (six metric-install ghost bands, PreFluxC2P, the FOFC test pass, and seven
    // each for the interior-first recovery and the post-M1 recovery), so the log volume
    // would be ~22x what it was when one ConsToPrim call meant one launch.  Hand each
    // launch a slice of the cap instead: a stage as a whole then reports about `errcap`
    // messages, and no launch -- the post-M1 recovery in particular -- is left with no
    // budget at all, which a "first launch takes everything" rule would do.
    const int errcap_ = track_event_counters_ ? static_cast<int>(errcap) :
        std::max(1, static_cast<int>(errcap)/kC2PLaunchesPerStage);
    Real mb = eos_.GetBaryonMass();
    const Real gamma_max_ = gamma_max;
    const Real sigma_max_ = sigma_max;
    const Real bsq_over_u_max_ = bsq_over_u_max;
    auto solver_template = *this;

    // FIXME: This only works for a flooring policy that has these functions!
    bool prim_failure, cons_failure;
    if (floors_only) {
      prim_failure = ps.GetEOSMutable().IsPrimitiveFlooringFailure();
      cons_failure = ps.GetEOSMutable().IsConservedFlooringFailure();
      ps.GetEOSMutable().SetPrimitiveFloorFailure(true);
      ps.GetEOSMutable().SetConservedFloorFailure(true);
    }

    // FIXME(JMF): We can short-circuit the primitive solve if FOFC is already enabled
    // due to a maximum principle violation.
    Kokkos::parallel_reduce("pshyd_c2p",athenak_lw(Kokkos::RangePolicy<>(DevExeSpace(), 0,
        nmkji)),
    KOKKOS_LAMBDA(const int &idx, int &sumerrs, int &sumlimits) {
      const int a = (idx)/nkji;
      const int m = a;
      int k = (idx - a*nkji)/nji;
      int j = (idx - a*nkji - k*nji)/ni;
      int i = (idx - a*nkji - k*nji - j*ni) + il;
      j += jl;
      k += kl;
      // cons and bcc0 may be stored on a band with origin (cko, cjo, cio) (the FOFC trial
      // state, DynGRMHDPS::FOFC); every other array here is ghost-extended.
      const int kc = k - cko, jc = j - cjo, ic = i - cio;

      if (floors_only && excise) {
        if (excision_flux_(m,k,j,i)) {
          return;
        }
      }

      // Add in a short circuit where FOFC is guaranteed.
      if (floors_only && fofc_(m, k, j, i)) {
        return;
      }

      // Extract the metric
      adm::ADMMetricPoint metric_pt{};
      metric.CellMetric(m, k, j, i, metric_pt);
      Real g3d[NSPMETRIC], g3u[NSPMETRIC], detg, sdetg;
      for (int n = 0; n < NSPMETRIC; ++n) g3d[n] = metric_pt.g_dd[n];
      detg = Primitive::GetDeterminant(g3d);
      sdetg = sqrt(detg);
      Real isdetg = 1.0/sdetg;
      adm::SpatialInv(1.0/detg,
                  g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

      // Extract the conserved variables
      Real cons_pt[NCONS], cons_pt_old[NCONS], prim_pt[NPRIM];
      cons_pt[CDN] = cons_pt_old[CDN] = cons(m, IDN, kc, jc, ic)*isdetg;
      cons_pt[CSX] = cons_pt_old[CSX] = cons(m, IM1, kc, jc, ic)*isdetg;
      cons_pt[CSY] = cons_pt_old[CSY] = cons(m, IM2, kc, jc, ic)*isdetg;
      cons_pt[CSZ] = cons_pt_old[CSZ] = cons(m, IM3, kc, jc, ic)*isdetg;
      cons_pt[CTA] = cons_pt_old[CTA] = cons(m, IEN, kc, jc, ic)*isdetg;
      for (int n = 0; n < nscal; n++) {
        cons_pt[CYD + n] = cons_pt_old[CYD + n] = cons(m, nhyd + n, kc, jc, ic)*isdetg;
      }
      // If we're only testing the floors, we can use the CC fields.
      Real b3u[NMAG];
      if (floors_only) {
        b3u[IBX] = bcc0(m, IBX, kc, jc, ic)*isdetg;
        b3u[IBY] = bcc0(m, IBY, kc, jc, ic)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, kc, jc, ic)*isdetg;
      } else {
        // Otherwise we don't have the correct CC fields yet, so use
        // the FC fields.
        bcc0(m, IBX, kc, jc, ic) = 0.5*(bfc.x1f(m,k,j,i) + bfc.x1f(m,k,j,i+1));
        bcc0(m, IBY, kc, jc, ic) = 0.5*(bfc.x2f(m,k,j,i) + bfc.x2f(m,k,j+1,i));
        bcc0(m, IBZ, kc, jc, ic) = 0.5*(bfc.x3f(m,k,j,i) + bfc.x3f(m,k+1,j,i));
        b3u[IBX] = bcc0(m, IBX, kc, jc, ic)*isdetg;
        b3u[IBY] = bcc0(m, IBY, kc, jc, ic)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, kc, jc, ic)*isdetg;
      }

      const auto &ps_cell = solver_template.ps;
      const auto &eos_cell = ps_cell.GetEOS();

      // If we're in an excised region, set the primitives to some default value.  Every
      // other cell takes the one solve below: one inlined copy of the root find, so the
      // fixed point after it adds a second copy to the variable-gamma kernels only.
      Primitive::SolverResult result;
      const bool excised_cell = excise && !smoothing && excision_floor_(m,k,j,i);
      if (excised_cell) {
        prim_pt[PRH] = dexcise_/mb;
        prim_pt[PVX] = 0.0;
        prim_pt[PVY] = 0.0;
        prim_pt[PVZ] = 0.0;
        for (int n = 0; n < nscal; n++) {
          // FIXME: Particle abundances should probably be set to a
          // default inside an excised region.
          prim_pt[PYF + n] =
              (cons_pt[CDN] > 0.0) ? cons_pt[CYD + n]/cons_pt[CDN] : 0.0;
        }
        prim_pt[PTM] = texcise_;
        prim_pt[PPR] =
            eos_cell.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
        result.error = Primitive::Error::SUCCESS;
        result.iterations = 0;
        result.cons_floor = false;
        result.prim_floor = false;
        result.cons_adjusted = true;
        ps_cell.PrimToCon(prim_pt, cons_pt, b3u, g3d);
      } else {
        result = ps_cell.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
      }

      // Dual energy: the eta1 ratio test.  The internal energy the inversion recovered
      // is measured against tau, the conserved energy the cell was handed; the ratio is
      // what the cancellation in `eoverD = qbar - mu*rbarsq + 1` left standing.  Below
      // eta1 the pressure -- and with it the Lorentz factor, which comes out of the same
      // root mu -- is re-solved from the advected adiabat instead.  The test runs before
      // the ceilings so that whichever state survives is the one they act on, and before
      // the drift-frame re-injection so the velocity is re-solved once.
      // The auxiliary is also the backstop for a solve that failed outright: the
      // adiabat is tried before any failure floor, and a cell it recovers is not a
      // failed cell.
      const bool energy_solve_failed = (result.error != Primitive::Error::SUCCESS);
      if (dual_enabled && !excised_cell) {
        const Real dens_arr = cons(m, IDN, kc, jc, ic);
        const Real kappa_adv = (dens_arr > 0.0)
            ? cons(m, dual_idx, kc, jc, ic)/dens_arr : -1.0;
        const Real rho_now = prim_pt[PRH]*mb;
        const Real eint_now = energy_solve_failed ? 0.0 :
            rho_now*eos_cell.GetSpecificInternalEnergy(
                prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
        const bool use_cons_e = !energy_solve_failed && (eint_now > 0.0) &&
            ((dual_eta1 <= 0.0) ||
             (eint_now > dual_eta1*Kokkos::fmax(cons_pt_old[CTA],
                                                static_cast<Real>(1.0e-18))));
        if (!use_cons_e && (kappa_adv > 0.0) && isfinite(kappa_adv) &&
            eos_cell.HasAdiabat()) {
          Real cons_aux[NCONS], prim_aux[NPRIM];
          for (int n = 0; n < NCONS; ++n) { cons_aux[n] = cons_pt_old[n]; }
          auto aux_result =
              ps_cell.ConToPrim(prim_aux, cons_aux, b3u, g3d, g3u, kappa_adv);
          // A failed auxiliary solve leaves the energy-channel answer standing.
          if (aux_result.error == Primitive::Error::SUCCESS) {
            for (int n = 0; n < NPRIM; ++n) { prim_pt[n] = prim_aux[n]; }
            for (int n = 0; n < NCONS; ++n) { cons_pt[n] = cons_aux[n]; }
            // The verdict is the surviving solve's, not the union of the two.
            // ApplyPrimitiveFloor runs on the state a root solve recovered, so
            // prim_floor describes that root alone: a cell whose energy channel
            // collapsed onto the atmosphere and which the adiabat then recovered
            // cleanly is not a floored cell, and reporting it as one costs it
            // first-order fluxes and a drift-frame repair -- exactly the diffusion the
            // channel exists to avoid.  cons_floor is different: the conserved floor and
            // the species limits run in the shared preamble of ConToPrim, ahead of
            // either root functor, so that flag belongs to both channels.  The auxiliary
            // solve is handed the same conserved state and re-raises it, and the union
            // is kept for it so nothing the preamble found can be lost if the two are
            // ever handed states that differ.
            result.prim_floor = aux_result.prim_floor;
            result.cons_floor = result.cons_floor || aux_result.cons_floor;
            result.cons_mass_floor = result.cons_mass_floor || aux_result.cons_mass_floor;
            result.error = Primitive::Error::SUCCESS;
            result.iterations = aux_result.iterations;
            // The floors downstream measure what they add against the state they act on,
            // and that state is now the auxiliary one.
            result.w_prefloor = aux_result.w_prefloor;
          }
        }
      }

      // ---- a failed inversion keeps the cell's primitives: KORAL's no-fixup branch ----
      // Where no channel inverts the cell, KORAL leaves the primitives unchanged
      // (koral_lite u2p.c, u2p: "leave primitives unchanged", pp = ppbak), applies no
      // fixup (DOU2PMHDFIXUPS 0, choices.h; DOFIXUPS 0, PROBLEMS/INFDISK/define.h), and
      // the step's update_entropy rebuilds U from them (p2u_mhd).  So here: the cell
      // keeps the primitives it last published and U is rebuilt from them, where those
      // are a gas state (rho and P positive, everything finite).  Only where they are not
      // does the error policy's reset stand (ResetFloor::FailureResponse: n_atm and T_atm
      // at rest), which inside a radiation field hands the exchange a near-empty 67 K
      // gas.  KORAL's entropy inversion between the two is the GR adiabat channel above,
      // which is not in use.  Excised cells never reach the solve and keep their state,
      // and the floors-only test pass (FOFC) keeps flagging the failure.  The failure is
      // still reported and counted below.
      if (!floors_only && !excised_cell && result.error != Primitive::Error::SUCCESS) {
        Real prev[NPRIM];
        prev[PRH] = prim(m, IDN, k, j, i)/mb;
        prev[PVX] = prim(m, IVX, k, j, i);
        prev[PVY] = prim(m, IVY, k, j, i);
        prev[PVZ] = prim(m, IVZ, k, j, i);
        prev[PPR] = prim(m, IPR, k, j, i);
        for (int n = 0; n < nscal; n++) prev[PYF + n] = prim(m, nhyd + n, k, j, i);
        bool prev_ok = isfinite(prev[PRH]) && prev[PRH] > 0.0 &&
                       isfinite(prev[PPR]) && prev[PPR] > 0.0 &&
                       isfinite(prev[PVX]) && isfinite(prev[PVY]) && isfinite(prev[PVZ]);
        if (prev_ok) {
          prev[PTM] = eos_cell.GetTemperatureFromP(prev[PRH], prev[PPR], &prev[PYF]);
          prev_ok = isfinite(prev[PTM]) && prev[PTM] > 0.0;
        }
        if (prev_ok) {
          prim_pt[PRH] = prev[PRH];
          prim_pt[PVX] = prev[PVX];
          prim_pt[PVY] = prev[PVY];
          prim_pt[PVZ] = prev[PVZ];
          prim_pt[PPR] = prev[PPR];
          prim_pt[PTM] = prev[PTM];
          for (int n = 0; n < nscal; n++) prim_pt[PYF + n] = prev[PYF + n];
          ps_cell.PrimToCon(prim_pt, cons_pt, b3u, g3d);
          result.cons_adjusted = true;
        }
      }

      GRPrimitiveLimitResult gr_limits;
      if (!excised_cell) {
        gr_limits = ApplyGRPrimitiveLimitsDevice(
            eos_cell, gamma_max_, sigma_max_, bsq_over_u_max_,
            prim_pt, b3u, g3d, result.cons_mass_floor);
        // One drift-frame re-solve for everything that was injected, the solver's own
        // atmosphere floor included -- which is why the reference enthalpy comes from the
        // raw inversion and not from the state the limits were handed.
        if (result.prim_floor || gr_limits.thermodynamic_adjusted) {
          ReinjectFlooredMassDevice(eos_cell, result.w_prefloor, prim_pt, b3u, g3d);
        }
      }
      const bool limit_adjusted = gr_limits.Any();
      // The floor inside ConToPrim already rewrote the conserved state, but it did so
      // before the velocity was re-solved, so it needs the rewrite again.
      const bool floor_adjusted = limit_adjusted || (result.prim_floor && !excised_cell);
      // A cell whose pressure came from the adiabat keeps its conserved energy.  The
      // energy equation is ignored for the pressure, not overwritten: the fluxes are
      // built from the auxiliary primitives, the conserved energy goes on integrating
      // conservatively, and the next inversion will find it wanting again and take the
      // auxiliary again.  This is the non-relativistic flavour's rule (ideal_hyd.cpp
      // rewrites U only after a floor or ceiling) and Enzo's.  Regenerating tau from
      // the auxiliary state instead injected the difference u_aux - u_cons into the
      // total energy at every inversion that took the channel: on the MAD funnel that
      // was a one-sided ratchet -- the channel fires exactly where the energy channel
      // reads cold -- and over 100 M it raised the funnel's median temperature from
      // 0.17 to 0.6 and multiplied the isolated Lorentz-factor spikes it exists to
      // remove.  The auxiliary must also NOT reach the FOFC flag below: dropping to
      // first-order fluxes on every cell that uses it would diffuse exactly the region
      // the channel exists to resolve.
      if (floor_adjusted) {
        result.cons_adjusted = true;
        ps_cell.PrimToCon(prim_pt, cons_pt, b3u, g3d);
      }

      if (floors_only) {
        if (result.error != Primitive::Error::SUCCESS || floor_adjusted) {
          fofc_(m,k,j,i) = true;
          sumlimits++;
        }
      } else if (!floors_only) {
        if (limit_adjusted) {
          sumlimits++;
        }
        if (result.error != Primitive::Error::SUCCESS && (nerrs_ + sumerrs < errcap_)) {
          sumerrs++;
          // Find out where the point went bad and report a bunch of information about it.
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

          adm::ADMMetricPoint diagnostic_metric{};
          metric.CellMetricFull(m, k, j, i, diagnostic_metric);
          // Split across three calls: as a single 35-argument device printf the tail was
          // silently truncated and K_dd[S22..S33] printed the same three garbage values
          // on every failure, which reads like a corrupted metric and sends anyone
          // debugging a C2P failure after the wrong bug.
          Kokkos::printf("An error occurred during the primitive solve: %s\n"
                 "  Location: (%d, %d, %d, %d)\n"
                 "            (%.17g, %.17g, %.17g)\n",
                 ErrorToString(result.error),
                 m, k, j, i,
                 x1v, x2v, x3v);
          Kokkos::printf("  Conserved vars: \n"
                 "    D   = %.17g\n"
                 "    Sx  = %.17g\n"
                 "    Sy  = %.17g\n"
                 "    Sz  = %.17g\n"
                 "    tau = %.17g\n"
                 "    Dye = %.17g\n"
                 "    Bx  = %.17g\n"
                 "    By  = %.17g\n"
                 "    Bz  = %.17g\n",
                 cons_pt_old[CDN], cons_pt_old[CSX], cons_pt_old[CSY], cons_pt_old[CSZ],
                 cons_pt_old[CTA], cons_pt_old[CYD], b3u[IBX], b3u[IBY], b3u[IBZ]);
          Kokkos::printf("  Metric vars: \n"
                 "    detg = %.17g\n"
                 "    g_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n"
                 "    alp  = %.17g\n"
                 "    beta = {%.17g, %.17g, %.17g}\n"
                 "    psi4 = %.17g\n"
                 "    K_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n",
                 detg,
                 g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 diagnostic_metric.alpha,
                 diagnostic_metric.beta_u[0], diagnostic_metric.beta_u[1],
                 diagnostic_metric.beta_u[2], diagnostic_metric.psi4,
                 diagnostic_metric.K_dd[S11], diagnostic_metric.K_dd[S12],
                 diagnostic_metric.K_dd[S13], diagnostic_metric.K_dd[S22],
                 diagnostic_metric.K_dd[S23], diagnostic_metric.K_dd[S33]);
          if (nerrs_ + sumerrs == errcap_) {
            Kokkos::printf("%d C2P errors have been detected on rank %d."
                   "All future C2P errors\n"
                   "on this rank will be suppressed. Fix your code!\n",
                   nerrs_ + sumerrs,rank);
          }
        }
        // Regardless of failure, we need to copy the primitives.
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }

        // The auxiliary primitive is the ADVECTED adiabat, never the one the pressure
        // just published implies: resynchronizing the two is the eta2 pass's job, and
        // doing it here would erase the shock heating the energy channel carries.
        Real kappa_pub = 0.0;
        // Raised wherever the published adiabat stops being the ratio the conserved array
        // holds, so that the partner further down can be brought along with it.
        bool dual_kappa_adjusted = false;
        if (dual_enabled) {
          const Real dens_pub = cons(m, IDN, kc, jc, ic);
          kappa_pub = (dens_pub > 0.0) ? cons(m, dual_idx, kc, jc, ic)/dens_pub : 0.0;
          if (!(isfinite(kappa_pub) && kappa_pub > 0.0)) {
            kappa_pub = eos_cell.GetAdiabat(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
            dual_kappa_adjusted = true;
          }
          if (!(isfinite(kappa_pub) && kappa_pub > 0.0)) {
            kappa_pub = 0.0;
            dual_kappa_adjusted = true;
          }
          // The bound by the cell's own energy budget is applied in the eta2 pass
          // (MHD::SynchronizeDualEnergyFieldFromAdiabat), which already writes both
          // columns of every owned cell each stage and is memory bound, so the power it
          // costs is free there and not here, where this kernel is instruction bound.
          // The root solve above bounds the specific energy it actually uses by tau/D
          // itself, so nothing this inversion publishes to the gas depends on the cap.
          prim(m, dual_idx, k, j, i) = kappa_pub;
          // Handed to the eta2 pass, which cannot form tau/D without the metric.  The
          // densitization cancels in the ratio, so cons_pt_old works directly.
          dual_tau_(m, k, j, i) = (cons_pt_old[CDN] > 0.0)
              ? cons_pt_old[CTA]/cons_pt_old[CDN] : 0.0;
        }

        // STORE_TEMPERATURE is a template constant; nvcc folds this branch while
        // still allowing the Kokkos device lambda to capture the view correctly.
        if (STORE_TEMPERATURE) {
          temperature(m,0,k,j,i) = prim_pt[PTM];
        }

        // If the conservative variables were floored or adjusted for consistency,
        // we need to copy the conserved variables, too.
        if (result.cons_floor || result.cons_adjusted) {
          /*if (fabs((cons_pt[CDN] - cons_pt_old[CDN])/cons_pt_old[CDN]) > 1e-12) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
            bool is_ghost = (i < is) || (i > ie) ||
                            (j < js) || (j > je) ||
                            (k < ks) || (k > ke);

            printf("Density was nontrivially adjusted on MeshBlock %d!\n"
                   "  Grid index: (i=%d, j=%d, k=%d)\n"
                   "  Physical position: (%g, %g, %g)\n"
                   "  D (old): %.17g\n"
                   "  D (new): %.17g\n"
                   "  Ghost zone? %s\n",
                   m, i, j, k,
                   x1v, x2v, x3v, cons_pt_old[CDN], cons_pt[CDN],
                   is_ghost ? "true" : "false");
          }*/
          cons(m, IDN, kc, jc, ic) = cons_pt[CDN]*sdetg;
          cons(m, IM1, kc, jc, ic) = cons_pt[CSX]*sdetg;
          cons(m, IM2, kc, jc, ic) = cons_pt[CSY]*sdetg;
          cons(m, IM3, kc, jc, ic) = cons_pt[CSZ]*sdetg;
          cons(m, IEN, kc, jc, ic) = cons_pt[CTA]*sdetg;
          for (int n = 0; n < nscal; n++) {
            cons(m, nhyd + n, kc, jc, ic) = cons_pt[CYD + n]*sdetg;
          }
          // The auxiliary rides the mass: a repaired D has to carry it, or the ratio
          // cons(dual)/cons(IDN) -- which IS the adiabat -- drifts by whatever the
          // floors did to the density.
          if (dual_enabled) {
            cons(m, dual_idx, kc, jc, ic) = cons(m, IDN, kc, jc, ic)*kappa_pub;
          }
        } else if (dual_kappa_adjusted) {
          // The bound above, the fallback and the zero that catches it are all rejections
          // of the advected ratio, and a primitive its own conserved partner contradicts
          // survives exactly one reconstruction: the next inversion forms
          // cons(dual)/cons(IDN) and recovers the value just rejected, so the bound would
          // buy nothing beyond a single stage.  No floor repaired the conserved state
          // here, so cons(m, IDN) still holds the density the ratio was read from -- the
          // same density the repair above multiplies by, which reads it back after
          // rewriting it.  Either way the partner rides the density the cell publishes.
          cons(m, dual_idx, kc, jc, ic) = cons(m, IDN, kc, jc, ic)*kappa_pub;
        }
      }
    }, Kokkos::Sum<int, DevMemSpace>(c2p_nerr_d_),
       Kokkos::Sum<int, DevMemSpace>(c2p_limits_d_));

    if (floors_only) {
      ps.GetEOSMutable().SetPrimitiveFloorFailure(prim_failure);
      ps.GetEOSMutable().SetConservedFloorFailure(cons_failure);
    }
    // The only host/device synchronisation left, and only runs that configure an event
    // log ever reach it.  Without one these counters are write-only for the whole run.
    //
    // KNOWN, BOUNDED BEHAVIOUR CHANGE when no event log is configured: `nerrs` stops
    // accumulating, so the in-kernel report budget (`nerrs_ + sumerrs < errcap_`) can
    // only be per-launch.  It is sliced above so that a stage still reports about
    // `errcap` messages in total; a healthy run reports nothing either way, and an
    // event log restores the exact previous per-run behaviour.
    if (track_event_counters_) {
      int count_errs = 0, count_limits = 0;
      Kokkos::deep_copy(count_errs, c2p_nerr_d_);
      Kokkos::deep_copy(count_limits, c2p_limits_d_);
      if (floors_only) {
        pmy_pack->pmesh->ecounter.nfofc += count_limits;
      } else {
        nerrs += count_errs;
        pmy_pack->pmesh->ecounter.neos_vceil += count_limits;
      }
    }
  }

  // Get the transformed magnetosonic speeds at a point in a given direction.
  KOKKOS_INLINE_FUNCTION
  void GetGRFastMagnetosonicSpeeds(Real& lambda_p, Real& lambda_m,
                                   Real prim[NPRIM], Real bsq, Real g3d[NSPMETRIC],
                                   Real beta_u[3], Real alpha, Real gii,
                                   int pvx) const {
    Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
    Real usq = Primitive::SquareVector(uu, g3d);
    int index = pvx - PVX;

    // Get spacetime quantities
    Real Wsq = 1.0 + usq;
    Real ialpha = 1.0/alpha;
    Real W = sqrt(Wsq);
    Real u0 = W*ialpha;
    Real u1 = uu[index] - u0*beta_u[index];
    Real g00 = -ialpha*ialpha;
    Real g01 = -g00*beta_u[index];
    Real g11 = gii - g01*beta_u[index];

    // Calculate the sound speed and the Alfven speed
    Real cs = ps.GetEOS().GetSoundSpeed(prim[PRH], prim[PTM], &prim[PYF]);
    Real csq = cs*cs;
    Real H = ps.GetEOS().GetBaryonMass()*prim[PRH]*
             ps.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
    Real vasq = bsq/(bsq + H);
    Real cmsq = csq + vasq - csq*vasq;

    // Set fast magnetosonic speed in appropriate coordinates
    Real a = u0*u0 - (g00 + u0*u0)*cmsq;
    Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) *cmsq);
    Real c = u1*u1 - (g11 + u1*u1)*cmsq;
    Real a1 = b / a;
    Real a0 = c / a;
    Real s = fmax(a1*a1 - 4.0 * a0, 0.0);
    s = sqrt(s);
    lambda_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
    lambda_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
  }

  // A function for converting PrimitiveSolver errors to strings
  KOKKOS_INLINE_FUNCTION
  static const char * ErrorToString(Primitive::Error e) {
    switch(e) {
      case Primitive::Error::SUCCESS:
        return "SUCCESS";
        break;
      case Primitive::Error::RHO_TOO_BIG:
        return "RHO_TOO_BIG";
        break;
      case Primitive::Error::RHO_TOO_SMALL:
        return "RHO_TOO_SMALL";
        break;
      case Primitive::Error::NANS_IN_CONS:
        return "NANS_IN_CONS";
        break;
      case Primitive::Error::MAG_TOO_BIG:
        return "MAG_TOO_BIG";
        break;
      case Primitive::Error::BRACKETING_FAILED:
        return "BRACKETING_FAILED";
        break;
      case Primitive::Error::NO_SOLUTION:
        return "NO_SOLUTION";
        break;
      default:
        return "OTHER";
        break;
    }
  }

  // A function for checking for NaNs in the conserved variables.
  KOKKOS_INLINE_FUNCTION
  static int CheckForConservedNaNs(const Real cons_pt[NCONS]) {
    int nans = 0;
    if (!isfinite(cons_pt[CDN])) {
      Kokkos::printf("D is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSX])) {
      Kokkos::printf("Sx is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSY])) {
      Kokkos::printf("Sy is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSZ])) {
      Kokkos::printf("Sz is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CTA])) {
      Kokkos::printf("Tau is NaN!\n"); // NOLINT
      nans = 1;
    }

    return nans;
  }

  KOKKOS_INLINE_FUNCTION
  static void DumpPrimitiveVars(const Real prim_pt[NPRIM]) {
    Kokkos::printf("Primitive vars: \n" // NOLINT
           "  rho = %.17g\n"
           "  ux  = %.17g\n"
           "  uy  = %.17g\n"
           "  uz  = %.17g\n"
           "  P   = %.17g\n"
           "  T   = %.17g\n",
           prim_pt[PRH], prim_pt[PVX], prim_pt[PVY],
           prim_pt[PVZ], prim_pt[PPR], prim_pt[PTM]);
  }
};
#endif  // EOS_PRIMITIVE_SOLVER_HYD_HPP_
