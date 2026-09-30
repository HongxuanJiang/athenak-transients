//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  Implements various (physics) source terms to be added to the Hydro or MHD eqns.
//  Source terms objects are stored in the respective fluid class, so that
//  Hydro/MHD can have different source terms

#include "srcterms.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string> // string

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ismcooling.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mb_storage.hpp"
#include "parameter_input.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "sink_particles/sink_particles.hpp"
#include "pgen/pgen.hpp"
#include "pgen/bh_force_pair.hpp"
#include "units/units.hpp"
#include "utils/gravity_weight.hpp"
#include "utils/lat_reflux_limiter.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real DiskCoolingThermalFloor(const EOS_Data &eos, const Real rho) {
  const Real gm1 = eos.gamma - 1.0;
  Real floor = eos.HydroInternalEnergyDensityFloor(rho);
  if ((eos.tfloor > 0.0) && (gm1 > 0.0)) {
    floor = fmax(floor, rho*eos.tfloor/gm1);
  }
  return floor;
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingTargetEint(const EOS_Data &eos, const Real rho, const Real cyl_r,
                           const Real omega, const Real h_over_r) {
  const Real gm1 = eos.gamma - 1.0;
  Real target = DiskCoolingThermalFloor(eos, rho);
  if ((rho > 0.0) && (cyl_r > 0.0) && (h_over_r > 0.0) &&
      (omega > 0.0) && (gm1 > 0.0)) {
    const Real cs = h_over_r*cyl_r*omega;
    target = fmax(target, rho*SQR(cs)/gm1);
  }
  return target;
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingNobleRemovedEintFromTarget(const Real rho, const Real rho_gate,
                                           const Real eint, const Real target_eint,
                                           const Real omega, const Real noble_s,
                                           const Real noble_q, const Real dt) {
  if (!(rho > rho_gate) || !(eint > target_eint) || !(target_eint > 0.0) ||
      !(omega > 0.0) || !(noble_s > 0.0) || !(noble_q > 0.0) || !(dt > 0.0)) {
    return 0.0;
  }
  const Real y = eint/target_eint;
  if (!(y > 1.0)) return 0.0;
  const Real y_minus_one = y - 1.0;
  const Real noble_switch = pow(y_minus_one + fabs(y_minus_one), noble_q);
  const Real de = dt*noble_s*omega*eint*noble_switch;
  if ((!isfinite(de)) || !(de > 0.0)) return 0.0;
  return fmin(eint - target_eint, de);
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingRemovedEint(const EOS_Data &eos, const Real rho, const Real eint,
                            const Real cyl_r, const Real mass, const Real h_over_r,
                            const Real noble_s, const Real noble_q, const Real dt) {
  if ((rho <= 10.0*eos.dfloor) || !(cyl_r > 0.0) || !(mass > 0.0) ||
      !(noble_s > 0.0) || !(noble_q > 0.0) || !(dt > 0.0)) {
    return 0.0;
  }
  const Real omega = sqrt(mass/(cyl_r*cyl_r*cyl_r));
  const Real target = DiskCoolingTargetEint(eos, rho, cyl_r, omega, h_over_r);
  return DiskCoolingNobleRemovedEintFromTarget(rho, 10.0*eos.dfloor, eint,
                                               target, omega, noble_s, noble_q, dt);
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingSRCoeffs(const EOS_Data &eos, const Real ux, const Real uy,
                         const Real uz, Real *mom_x, Real *mom_y, Real *mom_z) {
  const Real wsq = 1.0 + SQR(ux) + SQR(uy) + SQR(uz);
  if ((!isfinite(wsq)) || wsq < 1.0 || !(eos.gamma > 1.0)) {
    *mom_x = 0.0;
    *mom_y = 0.0;
    *mom_z = 0.0;
    return 1.0;
  }
  const Real wlor = sqrt(wsq);
  *mom_x = eos.gamma*wlor*ux;
  *mom_y = eos.gamma*wlor*uy;
  *mom_z = eos.gamma*wlor*uz;
  return eos.gamma*wsq - (eos.gamma - 1.0);
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingFixedGRCoeffs(const EOS_Data &eos, const Real ux, const Real uy,
                              const Real uz, Real glower[][4], Real gupper[][4],
                              Real *mom_x, Real *mom_y, Real *mom_z) {
  const Real q = glower[1][1]*SQR(ux) + 2.0*glower[1][2]*ux*uy
               + 2.0*glower[1][3]*ux*uz + glower[2][2]*SQR(uy)
               + 2.0*glower[2][3]*uy*uz + glower[3][3]*SQR(uz);
  if ((!isfinite(q)) || q < 0.0 || !(gupper[0][0] < 0.0)) {
    *mom_x = 0.0;
    *mom_y = 0.0;
    *mom_z = 0.0;
    return 1.0;
  }
  const Real alpha = sqrt(-1.0/gupper[0][0]);
  const Real lor = sqrt(1.0 + q);
  const Real ut = lor/alpha;
  const Real u1 = ux - alpha*lor*gupper[0][1];
  const Real u2 = uy - alpha*lor*gupper[0][2];
  const Real u3 = uz - alpha*lor*gupper[0][3];
  const Real u_lower_t = glower[0][0]*ut + glower[0][1]*u1
                       + glower[0][2]*u2 + glower[0][3]*u3;
  const Real u_lower_x = glower[1][0]*ut + glower[1][1]*u1
                       + glower[1][2]*u2 + glower[1][3]*u3;
  const Real u_lower_y = glower[2][0]*ut + glower[2][1]*u1
                       + glower[2][2]*u2 + glower[2][3]*u3;
  const Real u_lower_z = glower[3][0]*ut + glower[3][1]*u1
                       + glower[3][2]*u2 + glower[3][3]*u3;
  *mom_x = eos.gamma*ut*u_lower_x;
  *mom_y = eos.gamma*ut*u_lower_y;
  *mom_z = eos.gamma*ut*u_lower_z;
  return eos.gamma*ut*u_lower_t + (eos.gamma - 1.0);
}

KOKKOS_INLINE_FUNCTION
Real DiskCoolingDynGRCoeffs(const EOS_Data &eos, const Real ux, const Real uy,
                            const Real uz, const Real gxx, const Real gxy,
                            const Real gxz, const Real gyy, const Real gyz,
                            const Real gzz, Real *mom_x, Real *mom_y,
                            Real *mom_z) {
  const Real q = gxx*SQR(ux) + 2.0*gxy*ux*uy + 2.0*gxz*ux*uz
               + gyy*SQR(uy) + 2.0*gyz*uy*uz + gzz*SQR(uz);
  const Real det = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
  if ((!isfinite(q)) || q < 0.0 || (!isfinite(det)) || det <= 0.0) {
    *mom_x = 0.0;
    *mom_y = 0.0;
    *mom_z = 0.0;
    return 1.0;
  }
  const Real sdetg = (det > 0.0) ? sqrt(det) : 1.0;
  const Real wlor = sqrt(1.0 + q);
  *mom_x = sdetg*eos.gamma*wlor*(gxx*ux + gxy*uy + gxz*uz);
  *mom_y = sdetg*eos.gamma*wlor*(gxy*ux + gyy*uy + gyz*uz);
  *mom_z = sdetg*eos.gamma*wlor*(gxz*ux + gyz*uy + gzz*uz);
  return sdetg*(eos.gamma*(1.0 + q) - (eos.gamma - 1.0));
}

} // namespace

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters
// Only source terms specified in input file are initialized.
// Block name passed to constructor will be one of "hydro_srcterms", "mhd_srcterms",
// or "rad_srcterms"

SourceTerms::SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read flags for each source term implemented (default false)
  bool gravity_self_gravity = false;
  bool gravity_block_exists = pin->DoesBlockExist("gravity");
  bool legacy_self_gravity = false;
  if (gravity_block_exists) {
    gravity_self_gravity = pin->GetOrAddBoolean("gravity", "self_gravity", true);
  }
  const_accel = pin->GetOrAddBoolean(block, "const_accel", false);
  ism_cooling = pin->GetOrAddBoolean(block, "ism_cooling", false);
  rel_cooling = pin->GetOrAddBoolean(block, "rel_cooling", false);
  disk_cooling = pin->GetOrAddBoolean(block, "disk_cooling", false);
  rad_beam = pin->GetOrAddBoolean(block, "rad_beam", false);
  if (pin->DoesParameterExist(block, "self_gravity")) {
    legacy_self_gravity = pin->GetBoolean(block, "self_gravity");
  }
  if (gravity_block_exists) {
    self_gravity = gravity_self_gravity;
  } else if (legacy_self_gravity) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << block << "/self_gravity requires <gravity> self_gravity = true."
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    self_gravity = false;
  }

  const std::string pgen_name =
      pin->DoesBlockExist("problem") ?
      pin->GetOrAddString("problem", "pgen_name", "none") : "none";
  const bool analytic_bh_source_problem =
      (pgen_name == "tde_external");
  const bool external_bh_gravity_default =
      (block == "hydro_srcterms") && analytic_bh_source_problem;
  external_bh_gravity = pin->GetOrAddBoolean(
      "problem", "external_bh_gravity_source", external_bh_gravity_default);

  // Sink gravity is keyed on block existence, not on pmy_pack->psink: Hydro builds this
  // object inside its own constructor, which MeshBlockPack::AddPhysics runs long before
  // it constructs the sink module, so the pointer is still null here.  The kernel
  // re-checks psink and the live sink count at every call.  MHD sinks are not supported.
  sink_gravity = (block == "hydro_srcterms") &&
                 pin->DoesBlockExist("sink_particles");

  rho_grav_min = 0.0;
  if (self_gravity) {
    rho_grav_min = pin->GetOrAddReal("gravity", "rho_grav_min", 0.0);
  }
  rho_external_bh_min = 0.0;
  if (external_bh_gravity) {
    rho_external_bh_min = pin->GetOrAddReal("problem", "bh_grav_rho_min", 0.0);
  }
  external_bh_dt_factor = pin->GetOrAddReal("problem", "external_bh_dt_factor", 0.5);
  if (external_bh_dt_factor < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "problem/external_bh_dt_factor must be >= 0."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ResizeMeshBlockStorage(pp != nullptr ? pp->nmb_thispack : 1);


  // (1) read data for (constant) gravitational acceleration
  if (const_accel) {
    const_accel_val = pin->GetReal(block, "const_accel_val");
    const_accel_dir = pin->GetInteger(block, "const_accel_dir");
    if (const_accel_dir < 1 || const_accel_dir > 3) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "const_accle_dir must be 1,2, or 3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (2) Optically thin ISM cooling
  if (ism_cooling) {
    hrate = pin->GetReal(block, "hrate");
  }

  // (3) optically thin relativistic cooling
  if (rel_cooling) {
    crate_rel = pin->GetReal(block, "crate_rel");
    cpower_rel = pin->GetOrAddReal(block, "cpower_rel", 1.);
  }

  // (4) Noble-style disk cooling
  if (disk_cooling) {
    if (pin->DoesParameterExist(block, "disk_cooling_beta")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << block << "/disk_cooling_beta has been replaced by "
                << block << "/disk_cooling_noble_s." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    disk_cooling_h_over_r = pin->GetReal(block, "disk_cooling_h_over_r");
    disk_cooling_noble_s = pin->GetOrAddReal(block, "disk_cooling_noble_s", 1.0);
    disk_cooling_noble_q = pin->GetOrAddReal(block, "disk_cooling_noble_q", 0.5);
    if (!(disk_cooling_h_over_r > 0.0) || !(disk_cooling_noble_s > 0.0) ||
        !(disk_cooling_noble_q > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << block << "/disk_cooling_h_over_r, "
                << block << "/disk_cooling_noble_s, and "
                << block << "/disk_cooling_noble_q must be positive." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // (5) radiation beam source (radiation)
  if (rad_beam) {
    dii_dt = pin->GetReal(block, "dii_dt");
    pos1 = pin->GetReal(block, "pos_1");
    pos2 = pin->GetReal(block, "pos_2");
    pos3 = pin->GetReal(block, "pos_3");
    dir1 = pin->GetReal(block, "dir_1");
    dir2 = pin->GetReal(block, "dir_2");
    dir3 = pin->GetReal(block, "dir_3");
    width = pin->GetReal(block, "width");
    spread = pin->GetReal(block, "spread");
  }

  auto require_gamma_law_cooling = [&](const EOS_Data &eos, const std::string &prefix) {
    if ((ism_cooling || rel_cooling || disk_cooling) && eos.UsesTabulatedLTE()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<" << prefix << ">/eos uses a tabulated LTE EOS, but "
                << block << " cooling still assumes gamma-law temperature closure."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (disk_cooling && (!eos.use_e || !(eos.gamma > 1.0))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<" << prefix << ">/eos must use gamma-law internal energy for "
                << block << "/disk_cooling." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };
  if (pmy_pack->phydro != nullptr) {
    require_gamma_law_cooling(pmy_pack->phydro->peos->eos_data, "hydro");
  }
  if (pmy_pack->pmhd != nullptr) {
    require_gamma_law_cooling(pmy_pack->pmhd->peos->eos_data, "mhd");
  }
}

//----------------------------------------------------------------------------------------
// destructor

SourceTerms::~SourceTerms() {
}

bool SourceTerms::ResizeMeshBlockStorage(int nmb, bool exact) {
  const int target = std::max(std::max(1, nmb), MeshBlockStorageReserve());
  const bool need = exact ?
      (static_cast<int>(dtnew_eachmb.extent(0)) != target) :
      (static_cast<int>(dtnew_eachmb.extent(0)) < target);
  if (need) {
    dtnew_eachmb = DualArray1D<Real>("srcterms_dtnew_eachmb", target);
  }
  return need;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ApplySrcTerms
//! \brief Applies selected source terms to input arrays. Two different versions are
//! implemented for fluid and radiation fields, distinguished by their argument lists

void SourceTerms::ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  // NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars
  if (const_accel) ConstantAccel(w0, eos_data,  bdt, u0);
  if (ism_cooling) ISMCooling(w0, eos_data, bdt, u0);
  if (rel_cooling) RelCooling(w0, eos_data, bdt, u0);
  if (disk_cooling) DiskCooling(w0, eos_data, bdt, u0);
  if (self_gravity || external_bh_gravity) Gravity(w0, eos_data, bdt, u0);
  if (sink_gravity) SinkGravity(w0, eos_data, bdt, u0);
  return;
}

void SourceTerms::ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt) {
  if (rad_beam) BeamSource(i0, bdt);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::ConstantAccel
//! \brief Add constant acceleration
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                                const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  Real &g = const_accel_val;
  int &dir = const_accel_dir;

  par_for("const_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real src = bdt*g*w0(m,IDN,k,j,i);
    u0(m,dir,k,j,i) += src;
    if (eos_data.use_e) { u0(m,IEN,k,j,i) += src*w0(m,dir,k,j,i); }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::DualEnergyTarget()
//! \brief Resolve which fluid owns a conserved array, so a cooling term can charge that
//! fluid's dual-energy auxiliary.  Both fluids carry the two flavours and report which.

void SourceTerms::DualEnergyTarget(const DvceArray5D<Real> &u0, int &idx,
                                   bool &pdv) const {
  idx = -1;
  pdv = false;
  if (pmy_pack->pmhd != nullptr && pmy_pack->pmhd->use_dual_energy &&
      u0.data() == pmy_pack->pmhd->u0.data()) {
    idx = pmy_pack->pmhd->dual_energy_idx;
    pdv = pmy_pack->pmhd->dual_energy_pdv;
    return;
  }
  if (pmy_pack->phydro != nullptr && pmy_pack->phydro->use_dual_energy &&
      u0.data() == pmy_pack->phydro->u0.data()) {
    idx = pmy_pack->phydro->dual_energy_idx;
    pdv = pmy_pack->phydro->dual_energy_pdv;
    return;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DualEnergyCoolingDebit()
//! \brief Charge a gas-energy sink to the dual-energy auxiliary as well.
//!
//! A cooling term removes internal energy from the gas.  The dual-energy auxiliary is a
//! second, independently advected record of the same thermodynamic state, so a sink that
//! debits only the conserved energy is silently undone in every cell where the eta1 test
//! prefers the auxiliary -- an energy source running at the local cooling rate, with
//! positive feedback, because the restored-hot gas cools again next step from the same
//! unchanged auxiliary.
//!
//! The two flavours need different forms, and using either one for the other is a units
//! error no test in this repository would catch.  The non-relativistic auxiliary IS an
//! internal-energy density, so it takes the same increment the gas took.  The GR
//! auxiliary is the adiabat kappa = p/rho^Gamma carried as D*kappa; cooling changes p at
//! fixed rho and fixed D, so kappa and the internal energy scale by exactly the same
//! factor and the debit is multiplicative.  `de` must be the COMOVING internal energy
//! removed, not the conserved-energy increment: the two differ by the relativistic
//! coefficient, and kappa is a comoving quantity.
//!
//! The factor is floored so that a step which would remove more than the cell's internal
//! energy leaves a small positive adiabat rather than a negative one; the same step
//! leaves the gas at its own energy floor, and the eta2 pass reconciles the pair
//! afterwards.

KOKKOS_INLINE_FUNCTION
void DualEnergyCoolingDebit(const DvceArray5D<Real> &u0, const int dual_idx,
                            const bool pdv_flavour, const Real de, const Real eint,
                            const int m, const int k, const int j, const int i) {
  if (pdv_flavour) {
    u0(m, dual_idx, k, j, i) -= de;
    return;
  }
  if (!(eint > 0.0) || !isfinite(eint) || !isfinite(de)) return;
  const Real f = 1.0 - de/eint;
  u0(m, dual_idx, k, j, i) *= fmax(f, 1.0e-3);
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = hrate;
  Real temp_unit = pmy_pack->TemperatureUnitCGS();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  int dual_idx_c = -1;
  bool dual_pdv_c = false;
  DualEnergyTarget(u0, dual_idx_c, dual_pdv_c);
  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    const Real de_ism = bdt * w0(m,IDN,k,j,i) *
                        (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating);
    u0(m,IEN,k,j,i) -= de_ism;
    if (dual_idx_c >= 0) {
      DualEnergyCoolingDebit(u0, dual_idx_c, dual_pdv_c, de_ism,
                             w0(m,IEN,k,j,i), m, k, j, i);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::RelCooling()
//! \brief Add explict relativistic cooling in the energy and momentum equations.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                             const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real cooling_rate = crate_rel;
  Real cooling_power = cpower_rel;

  int dual_idx_c = -1;
  bool dual_pdv_c = false;
  DualEnergyTarget(u0, dual_idx_c, dual_pdv_c);
  par_for("cooling", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // temperature in cgs unit
    Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;

    auto &ux = w0(m,IVX,k,j,i);
    auto &uy = w0(m,IVY,k,j,i);
    auto &uz = w0(m,IVZ,k,j,i);
    Real ut = sqrt(1.0 + ux*ux + uy*uy + uz*uz);

    const Real de_rel =
        bdt*w0(m,IDN,k,j,i)*ut*pow((temp*cooling_rate), cooling_power);
    u0(m,IEN,k,j,i) -= de_rel;
    if (dual_idx_c >= 0) {
      // ut converts the comoving rate into the conserved-energy increment, so the
      // comoving internal energy removed is the increment divided back by it.
      const Real de_comoving = (ut > 0.0) ? de_rel/ut : de_rel;
      DualEnergyCoolingDebit(u0, dual_idx_c, dual_pdv_c, de_comoving,
                             w0(m,IEN,k,j,i), m, k, j, i);
    }
    u0(m,IM1,k,j,i) -= bdt*w0(m,IDN,k,j,i)*ux*pow((temp*cooling_rate), cooling_power);
    u0(m,IM2,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uy*pow((temp*cooling_rate), cooling_power);
    u0(m,IM3,k,j,i) -= bdt*w0(m,IDN,k,j,i)*uz*pow((temp*cooling_rate), cooling_power);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::DiskCooling()
//! \brief Relax gas above a target disk thickness on an orbital cooling time.
//! NOTE source terms must be computed using primitive (w0) and NOT conserved (u0) vars

void SourceTerms::DiskCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                              const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  auto &size = pmy_pack->pmb->mb_size;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;
  Real h_over_r = disk_cooling_h_over_r;
  Real noble_s = disk_cooling_noble_s;
  Real noble_q = disk_cooling_noble_q;
  bool dyn_gr = pmy_pack->pcoord->is_dynamical_relativistic &&
                (pmy_pack->padm != nullptr);
  bool fixed_gr = pmy_pack->pcoord->is_general_relativistic;
  bool sr = pmy_pack->pcoord->is_special_relativistic;
  const bool use_adm_metric = (pmy_pack->padm != nullptr) && (dyn_gr);

  if (use_adm_metric) {
    auto metric = pmy_pack->padm->GetMetricView();
    int dual_idx_c = -1;
    bool dual_pdv_c = false;
    DualEnergyTarget(u0, dual_idx_c, dual_pdv_c);
    par_for("disk_cooling_adm_gr", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
      const int m = lat_enabled ? active_indices(a) : a;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      const Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                                 size.d_view(m).x1max);
      const Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                                 size.d_view(m).x2max);
      const Real rho = w0(m, IDN, k, j, i);
      const Real eint = dyn_gr ? w0(m, IPR, k, j, i)/(eos_data.gamma - 1.0)
                               : w0(m, IEN, k, j, i);
      const Real de = DiskCoolingRemovedEint(
                                eos_data, rho, eint, sqrt(SQR(x) + SQR(y)),
                                1.0, h_over_r, noble_s, noble_q, block_bdt);
      if (!(de > 0.0)) return;

      Real mom_x = 0.0, mom_y = 0.0, mom_z = 0.0;
      adm::ADMMetricPoint metric_pt{};
      metric.CellMetric(m, k, j, i, metric_pt);
      const Real coeff = DiskCoolingDynGRCoeffs(
          eos_data, w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
          w0(m, IVZ, k, j, i), metric_pt.g_dd[S11], metric_pt.g_dd[S12],
          metric_pt.g_dd[S13], metric_pt.g_dd[S22], metric_pt.g_dd[S23],
          metric_pt.g_dd[S33], &mom_x, &mom_y, &mom_z);
      u0(m, IM1, k, j, i) -= mom_x*de;
      u0(m, IM2, k, j, i) -= mom_y*de;
      u0(m, IM3, k, j, i) -= mom_z*de;
      u0(m, IEN, k, j, i) -= coeff*de;
      if (dual_idx_c >= 0) {
        DualEnergyCoolingDebit(u0, dual_idx_c, dual_pdv_c, de, eint, m, k, j, i);
      }
    });
    return;
  }

  bool flat = false;
  Real spin = 0.0;
  if (fixed_gr) {
    flat = pmy_pack->pcoord->coord_data.is_minkowski;
    spin = pmy_pack->pcoord->coord_data.bh_spin;
  }
    int dual_idx_c = -1;
  bool dual_pdv_c = false;
  DualEnergyTarget(u0, dual_idx_c, dual_pdv_c);
par_for("disk_cooling", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
    const Real x = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real rho = w0(m, IDN, k, j, i);
    const Real eint = w0(m, IEN, k, j, i);
    const Real de = DiskCoolingRemovedEint(
                              eos_data, rho, eint, sqrt(SQR(x) + SQR(y)),
                              1.0, h_over_r, noble_s, noble_q, block_bdt);
    if (!(de > 0.0)) return;

    Real coeff = 1.0;
    Real mom_x = 0.0, mom_y = 0.0, mom_z = 0.0;
    if (fixed_gr) {
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x, y, z, flat, spin, glower, gupper);
      coeff = DiskCoolingFixedGRCoeffs(eos_data, w0(m, IVX, k, j, i),
                                       w0(m, IVY, k, j, i),
                                       w0(m, IVZ, k, j, i), glower, gupper,
                                       &mom_x, &mom_y, &mom_z);
    } else if (sr) {
      coeff = DiskCoolingSRCoeffs(eos_data, w0(m, IVX, k, j, i),
                                  w0(m, IVY, k, j, i), w0(m, IVZ, k, j, i),
                                  &mom_x, &mom_y, &mom_z);
    }
    u0(m, IM1, k, j, i) -= mom_x*de;
    u0(m, IM2, k, j, i) -= mom_y*de;
    u0(m, IM3, k, j, i) -= mom_z*de;
    u0(m, IEN, k, j, i) -= coeff*de;
    if (dual_idx_c >= 0) {
      DualEnergyCoolingDebit(u0, dual_idx_c, dual_pdv_c, de, eint, m, k, j, i);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::Gravity
//! \brief Adds self-gravity and external black-hole gravity source terms to conserved
//! variables
//! \note
//! The numerical self-gravity branch uses pgrav->phi, which contains only the
//! multigrid self-gravity potential. The external black-hole branch evaluates the
//! analytic potential directly through problem_runtime. When conserved energy is
//! present, gravitational work is coupled to the Godunov mass flux following the
//! source-term form of Mullen, Hanawa and Gammie 2020. Exact conservation is still
//! limited by the multigrid residual and by any intentionally time-lagged potential.

void SourceTerms::Gravity(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                          const Real bdt, DvceArray5D<Real> &u0) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;
  const bool have_self_phi = self_gravity && (pmy_pack->pgrav != nullptr);
  if (self_gravity && !have_self_phi) {
    std::cout << "### ERROR in SourceTerms::Gravity" << std::endl
              << "self_gravity source term enabled but pgrav is null" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (have_self_phi &&
      (!(pmy_pack->pgrav->phi_valid) || !(pmy_pack->pgrav->self_phi_time_valid))) {
    std::cout << "### ERROR in SourceTerms::Gravity" << std::endl
              << "self_gravity source term requires a valid numerical phi field. "
              << "Run the multigrid solve before applying self-gravity sources."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool use_self_gravity = have_self_phi;
  const bool analytic_bh_momentum = pmy_pack->pmesh->pgen != nullptr &&
      pmy_pack->pmesh->pgen->analytic_external_bh_momentum;

  const Real t_src = problem_runtime::HydroStageTimeOr(pmy_pack->pmesh->time);
  DvceArray5D<Real> self_phi;
  if (use_self_gravity) {
    self_phi = pmy_pack->pgrav->phi;
  }
  // gravity/rho_grav_min and gravity/rho_external_bh_min are applied as the continuous
  // coupling weights of utils/gravity_weight.hpp rather than as hard cuts: the momentum
  // source and the Mullen-Hanawa-Gammie work source are MULTIPLIED by w(rho) in [0,1],
  // which ramps in log density from the hydro density floor to the threshold.  The same
  // weight multiplies the Poisson source in Multigrid::LoadSource, so a cell that sources
  // gravity at weight w also feels it at weight w and the self-gravity pair still
  // conserves momentum.  The weight is read off the stage primitives, exactly where the
  // boolean gate used to be read, so a cell that fills during a long LAT step is coupled
  // as it fills; freezing it on the start-of-step register instead would cost a coarse
  // block a whole step of gravity and was measured to move this deck's LAT run 50% away
  // from its synchronous twin.  The refund below therefore has to record the weight.
  Real rho_gmin = rho_grav_min;
  const Real rho_gweight_floor = eos_data.dfloor;
  Real rho_bh_gmin = rho_external_bh_min;
  bool excise_enabled = false;
  Real excise_radius = 0.0;
  Real excise_density = 0.0;
  Real excise_eint = 0.0;
  Real sink_x = 0.0;
  Real sink_y = 0.0;
  Real sink_z = 0.0;
  problem_runtime::GetExcisionState(t_src, excise_enabled, excise_radius,
      excise_density, excise_eint, sink_x, sink_y, sink_z);
  Real excise_r2 = excise_radius * excise_radius;
  bool external_bh_enabled = false;
  Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
  Real bh_mass = 0.0, bh_softening = 0.0, newton_g = 0.0;
  if (external_bh_gravity &&
      problem_runtime::ExternalBHGravitySourceCouplingEnabled()) {
    problem_runtime::GetExternalBHPotential(t_src, external_bh_enabled,
                                            bhx, bhy, bhz,
                                            bh_mass, bh_softening, newton_g);
  }
  bool bh_sink_mask_enabled = false;
  Real bh_sink_mask_radius = 0.0;
  Real bh_sink_mask_x = 0.0;
  Real bh_sink_mask_y = 0.0;
  Real bh_sink_mask_z = 0.0;
  if (use_self_gravity || external_bh_enabled) {
    problem_runtime::GetBHSinkGravityMask(t_src, bh_sink_mask_enabled,
                                          bh_sink_mask_radius,
                                          bh_sink_mask_x, bh_sink_mask_y,
                                          bh_sink_mask_z);
  }
  // The LAT window-end reflux refunds the gravitational work of the mass a finer
  // neighbour moved through a coarse/fine face, but it only ever sees the summed
  // mismatch and the post-step density, while the work above is gated per stage on the
  // stage primitives.  Record here, against the very state the weights above just
  // used, how much of the mismatch taken since the previous stage each coupling refused,
  // so that the refund can be driven by the admitted part alone.  With a continuous
  // weight the refused part of a face cell's mismatch is (1 - w) of it, not all or
  // nothing.  The refused part is stored rather than the admitted one, so a face that is
  // fully coupled stays at exactly zero and the refund reads the accumulator untouched.
  if (eos_data.use_e) {
    DvceFaceFld5D<Real> *pacc = nullptr;
    DvceFaceFld5D<Real> *pgacc = nullptr;
    if (pmy_pack->phydro != nullptr && pmy_pack->phydro->lat_grav_reflux_allocated) {
      pacc = &(pmy_pack->phydro->lat_reflux);
      pgacc = &(pmy_pack->phydro->lat_grav_reflux);
    }
    // Capacity, not an exact match: Hydro/MHD allocate these registers with the
    // MeshBlock storage capacity, which ResizeMeshBlockStorage rounds up to a grain
    // (mesh/mb_storage.hpp) after every regrid, so the extent equals nmb only until
    // the first AMR pass whose block count is not a multiple of the grain.  Testing
    // for equality silently switched the whole gate off for the rest of such a run
    // and left the window-end refund crediting the FULL mismatch on faces where the
    // stage source term had applied none of it.  The kernels below iterate 0..nmb-1.
    if (pacc != nullptr && pgacc != nullptr &&
        pacc->x1f.extent_int(0) >= nmb && pgacc->x1f.extent_int(0) >= nmb) {
      const int cseen = lat_reflux::kGravWorkSeen;
      const int cself = lat_reflux::kGravWorkBlockedSelf;
      const int cbh = lat_reflux::kGravWorkBlockedBH;
      auto acc1 = pacc->x1f;
      auto acc2 = pacc->x2f;
      auto acc3 = pacc->x3f;
      auto gacc1 = pgacc->x1f;
      auto gacc2 = pgacc->x2f;
      auto gacc3 = pgacc->x3f;
      par_for("lat_grav_gate_x1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, 0, 1,
      KOKKOS_LAMBDA(int m, int k, int j, int ia) {
        const Real acc = acc1(m,IDN,k,j,ia);
        const Real delta = acc - gacc1(m,cseen,k,j,ia);
        if (delta == 0.0) return;
        gacc1(m,cseen,k,j,ia) = acc;
        const int ic = (ia == 0) ? is : ie;
        const Real rho_fl = w0(m,IDN,k,j,ic);
        Real w_self = (use_self_gravity && rho_fl > 0.0) ?
            gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
        Real w_bh = (external_bh_enabled && rho_fl > 0.0) ?
            gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
        const bool self_open = (w_self > 0.0);
        const bool bh_open = (w_bh > 0.0);
        if ((self_open || bh_open) && (excise_enabled || bh_sink_mask_enabled)) {
          const Real x = CellCenterX(ic - is, indcs.nx1,
              mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
          const Real y = CellCenterX(j - js, indcs.nx2,
              mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
          const Real z = CellCenterX(k - ks, indcs.nx3,
              mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
          if ((excise_enabled &&
               problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                   excise_r2)) ||
              (bh_sink_mask_enabled &&
               problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                        bh_sink_mask_y, bh_sink_mask_z,
                                                        bh_sink_mask_radius))) {
            w_self = 0.0;
            w_bh = 0.0;
          }
        }
        gacc1(m,cself,k,j,ia) += (1.0 - w_self)*delta;
        gacc1(m,cbh,k,j,ia) += (1.0 - w_bh)*delta;
      });
      if (multi_d) {
        par_for("lat_grav_gate_x2", DevExeSpace(), 0, nmb-1, ks, ke, 0, 1, is, ie,
        KOKKOS_LAMBDA(int m, int k, int ja, int i) {
          const Real acc = acc2(m,IDN,k,ja,i);
          const Real delta = acc - gacc2(m,cseen,k,ja,i);
          if (delta == 0.0) return;
          gacc2(m,cseen,k,ja,i) = acc;
          const int jc = (ja == 0) ? js : je;
          const Real rho_fl = w0(m,IDN,k,jc,i);
          Real w_self = (use_self_gravity && rho_fl > 0.0) ?
              gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
          Real w_bh = (external_bh_enabled && rho_fl > 0.0) ?
              gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
          const bool self_open = (w_self > 0.0);
          const bool bh_open = (w_bh > 0.0);
          if ((self_open || bh_open) && (excise_enabled || bh_sink_mask_enabled)) {
            const Real x = CellCenterX(i - is, indcs.nx1,
                mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
            const Real y = CellCenterX(jc - js, indcs.nx2,
                mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
            const Real z = CellCenterX(k - ks, indcs.nx3,
                mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
            if ((excise_enabled &&
                 problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                     excise_r2)) ||
                (bh_sink_mask_enabled &&
                 problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                          bh_sink_mask_y, bh_sink_mask_z,
                                                          bh_sink_mask_radius))) {
              w_self = 0.0;
              w_bh = 0.0;
            }
          }
          gacc2(m,cself,k,ja,i) += (1.0 - w_self)*delta;
          gacc2(m,cbh,k,ja,i) += (1.0 - w_bh)*delta;
        });
      }
      if (three_d) {
        par_for("lat_grav_gate_x3", DevExeSpace(), 0, nmb-1, 0, 1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int ka, int j, int i) {
          const Real acc = acc3(m,IDN,ka,j,i);
          const Real delta = acc - gacc3(m,cseen,ka,j,i);
          if (delta == 0.0) return;
          gacc3(m,cseen,ka,j,i) = acc;
          const int kc = (ka == 0) ? ks : ke;
          const Real rho_fl = w0(m,IDN,kc,j,i);
          Real w_self = (use_self_gravity && rho_fl > 0.0) ?
              gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
          Real w_bh = (external_bh_enabled && rho_fl > 0.0) ?
              gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
          const bool self_open = (w_self > 0.0);
          const bool bh_open = (w_bh > 0.0);
          if ((self_open || bh_open) && (excise_enabled || bh_sink_mask_enabled)) {
            const Real x = CellCenterX(i - is, indcs.nx1,
                mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
            const Real y = CellCenterX(j - js, indcs.nx2,
                mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
            const Real z = CellCenterX(kc - ks, indcs.nx3,
                mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
            if ((excise_enabled &&
                 problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                     excise_r2)) ||
                (bh_sink_mask_enabled &&
                 problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                          bh_sink_mask_y, bh_sink_mask_z,
                                                          bh_sink_mask_radius))) {
              w_self = 0.0;
              w_bh = 0.0;
            }
          }
          gacc3(m,cself,ka,j,i) += (1.0 - w_self)*delta;
          gacc3(m,cbh,ka,j,i) += (1.0 - w_bh)*delta;
        });
      }
    }
  }

  if (!use_self_gravity && !external_bh_enabled) return;

  // Get Godunov density fluxes (Hydro or MHD), used in the energy source term
  // following Mullen, Hanawa & Gammie 2020.
  // Each module's register lives on its own flux band.
  BandView5D<Real> flx1, flx2, flx3;
  if (pmy_pack->phydro != nullptr) {
    flx1 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x1f);
    flx2 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x2f);
    flx3 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x3f);
  } else if (pmy_pack->pmhd != nullptr) {
    flx1 = pmy_pack->pmhd->FluxBand(pmy_pack->pmhd->uflx.x1f);
    flx2 = pmy_pack->pmhd->FluxBand(pmy_pack->pmhd->uflx.x2f);
    flx3 = pmy_pack->pmhd->FluxBand(pmy_pack->pmhd->uflx.x3f);
  } else {
    std::cout << "### ERROR in SourceTerms::Gravity" << std::endl
              << "gravity source term requires Hydro or MHD module" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Ledger debug: the external-BH part of the energy source alone (same weights,
  // stencil and fluxes), so the BH-gas pair and the self-gravity work can be
  // reconciled separately at the window end.
  if (external_bh_enabled && eos_data.use_e && pmy_pack->pgrav != nullptr &&
      pmy_pack->pgrav->lat_ledger_debug && !excise_enabled && !bh_sink_mask_enabled) {
    const Real eff_weight = pmy_pack->pgrav->debug_stage_weight;
    Real bh_work = 0.0;
    Kokkos::parallel_reduce("gravity_bh_work_debug",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, (nwork1+1)*indcs.nx1*indcs.nx2*indcs.nx3),
    KOKKOS_LAMBDA(const int &index, Real &sum) {
      const int per_block = indcs.nx1*indcs.nx2*indcs.nx3;
      const int a = index/per_block;
      const int cell = index - a*per_block;
      const int i = cell%indcs.nx1 + is;
      const int j = (cell/indcs.nx1)%indcs.nx2 + js;
      const int k = cell/(indcs.nx1*indcs.nx2) + ks;
      const int m = lat_enabled ? active_indices(a) : a;
      const Real rho_fl = w0(m,IDN,k,j,i);
      if (rho_fl <= 0.0) return;
      const Real w_bh = gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin);
      if (!(w_bh > 0.0)) return;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      const auto mb = mbsize.d_view(m);
      const Real x = CellCenterX(i - is, indcs.nx1, mb.x1min, mb.x1max);
      const Real y = CellCenterX(j - js, indcs.nx2, mb.x2min, mb.x2max);
      const Real z = CellCenterX(k - ks, indcs.nx3, mb.x3min, mb.x3max);
      const Real vol = mb.dx1*mb.dx2*mb.dx3;
      Real e = 0.0;
      {
        const Real xl = CellCenterX(i - 1 - is, indcs.nx1, mb.x1min, mb.x1max);
        const Real xr = CellCenterX(i + 1 - is, indcs.nx1, mb.x1min, mb.x1max);
        const auto s = bh_force_pair::Stencil(0, x, y, z, xl, xr, bhx, bhy, bhz,
            bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
        e += 0.5*block_bdt/mb.dx1*(flx1(m,IDN,k,j,i)*s.dpl + flx1(m,IDN,k,j,i+1)*s.dpr);
      }
      if (multi_d) {
        const Real yl = CellCenterX(j - 1 - js, indcs.nx2, mb.x2min, mb.x2max);
        const Real yr = CellCenterX(j + 1 - js, indcs.nx2, mb.x2min, mb.x2max);
        const auto s = bh_force_pair::Stencil(1, x, y, z, yl, yr, bhx, bhy, bhz,
            bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
        e += 0.5*block_bdt/mb.dx2*(flx2(m,IDN,k,j,i)*s.dpl + flx2(m,IDN,k,j+1,i)*s.dpr);
      }
      if (three_d) {
        const Real zl = CellCenterX(k - 1 - ks, indcs.nx3, mb.x3min, mb.x3max);
        const Real zr = CellCenterX(k + 1 - ks, indcs.nx3, mb.x3min, mb.x3max);
        const auto s = bh_force_pair::Stencil(2, x, y, z, zl, zr, bhx, bhy, bhz,
            bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
        e += 0.5*block_bdt/mb.dx3*(flx3(m,IDN,k,j,i)*s.dpl + flx3(m,IDN,k+1,j,i)*s.dpr);
      }
      sum += w_bh*e*vol;
    }, Kokkos::Sum<Real>(bh_work));
    pmy_pack->pgrav->debug_bh_work_local += eff_weight*bh_work;
  }

  // x1-direction momentum and energy source terms
  par_for("gravity_x1",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
    Real rho_fl = w0(m,IDN,k,j,i);
    const Real w_self = use_self_gravity ?
        gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
    const Real w_bh = external_bh_enabled ?
        gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
    const bool use_self_cell = (w_self > 0.0);
    const bool use_bh_cell = (w_bh > 0.0);
    if (rho_fl <= 0.0 || (!use_self_cell && !use_bh_cell)) return;
    // The block size once, and the cell centre once: the excision test and the BH
    // stencil read the same three coordinates.
    const RegionSize &ms = mbsize.d_view(m);
    Real x = 0.0, y = 0.0, z = 0.0;
    if (excise_enabled || bh_sink_mask_enabled || use_bh_cell) {
      x = CellCenterX(i - is, indcs.nx1, ms.x1min, ms.x1max);
      y = CellCenterX(j - js, indcs.nx2, ms.x2min, ms.x2max);
      z = CellCenterX(k - ks, indcs.nx3, ms.x3min, ms.x3max);
    }
    if (excise_enabled || bh_sink_mask_enabled) {
      if (excise_enabled &&
          problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                              excise_r2)) {
        return;
      }
      if (bh_sink_mask_enabled &&
          problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                   bh_sink_mask_y, bh_sink_mask_z,
                                                   bh_sink_mask_radius)) {
        return;
      }
    }
    Real dx1 = ms.dx1;
    Real hdtodx1 = 0.5*block_bdt/dx1;
    Real dpl = 0.0;
    Real dpr = 0.0;
    Real bh_momentum_correction = 0.0;
    if (use_self_cell) {
      dpl = -w_self*(self_phi(m,0,k,j,i  ) - self_phi(m,0,k,j,i-1));
      dpr = -w_self*(self_phi(m,0,k,j,i+1) - self_phi(m,0,k,j,i  ));
    }
    if (use_bh_cell) {
      const Real xl = CellCenterX(i - 1 - is, indcs.nx1, ms.x1min, ms.x1max);
      const Real xr = CellCenterX(i + 1 - is, indcs.nx1, ms.x1min, ms.x1max);
      const auto bh = bh_force_pair::Stencil(0, x, y, z, xl, xr, bhx, bhy, bhz,
          bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
      if (analytic_bh_momentum) {
        bh_momentum_correction = w_bh*(bh_force_pair::Evaluate(x,y,z,bhx,bhy,bhz,bh_mass,
            bh_softening,newton_g,bh_sink_mask_radius).ax - bh.Acceleration(dx1));
      }
      dpl += w_bh*bh.dpl;
      dpr += w_bh*bh.dpr;
    }

    // Add momentum source term
    u0(m,IM1,k,j,i) += hdtodx1 * w0(m,IDN,k,j,i) * (dpl + dpr);
    if (analytic_bh_momentum) {
      u0(m,IM1,k,j,i) += block_bdt*w0(m,IDN,k,j,i)*bh_momentum_correction;
    }

    // Add gravitational work using the Godunov mass fluxes. This is not a
    // direct thermal source, so the auxiliary dual-energy field is intentionally not
    // updated here; any reliable total-energy change is folded back into eint_aux later
    // by the eta2 synchronization step after source terms and boundary exchanges.
    if (eos_data.use_e) {
      u0(m,IEN,k,j,i) += hdtodx1 * (flx1(m,IDN,k,j,i  ) * dpl
                                    + flx1(m,IDN,k,j,i+1) * dpr);
    }
  });

  if (multi_d) {
    // x2-direction momentum and energy source terms
    par_for("gravity_x2",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = lat_enabled ? active_indices(a) : a;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      Real rho_fl = w0(m,IDN,k,j,i);
      const Real w_self = use_self_gravity ?
          gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
      const Real w_bh = external_bh_enabled ?
          gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
      const bool use_self_cell = (w_self > 0.0);
      const bool use_bh_cell = (w_bh > 0.0);
      if (rho_fl <= 0.0 || (!use_self_cell && !use_bh_cell)) return;
      const RegionSize &ms = mbsize.d_view(m);
      Real x = 0.0, y = 0.0, z = 0.0;
      if (excise_enabled || bh_sink_mask_enabled || use_bh_cell) {
        x = CellCenterX(i - is, indcs.nx1, ms.x1min, ms.x1max);
        y = CellCenterX(j - js, indcs.nx2, ms.x2min, ms.x2max);
        z = CellCenterX(k - ks, indcs.nx3, ms.x3min, ms.x3max);
      }
      if (excise_enabled || bh_sink_mask_enabled) {
        if (excise_enabled &&
            problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                excise_r2)) {
          return;
        }
        if (bh_sink_mask_enabled &&
            problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                     bh_sink_mask_y, bh_sink_mask_z,
                                                     bh_sink_mask_radius)) {
          return;
        }
      }
      Real dx2 = ms.dx2;
      Real hdtodx2 = 0.5*block_bdt/dx2;
      Real dpl = 0.0;
      Real dpr = 0.0;
      Real bh_momentum_correction = 0.0;
      if (use_self_cell) {
        dpl = -w_self*(self_phi(m,0,k,j,  i) - self_phi(m,0,k,j-1,i));
        dpr = -w_self*(self_phi(m,0,k,j+1,i) - self_phi(m,0,k,j,  i));
      }
      if (use_bh_cell) {
        const Real yl = CellCenterX(j - 1 - js, indcs.nx2, ms.x2min, ms.x2max);
        const Real yr = CellCenterX(j + 1 - js, indcs.nx2, ms.x2min, ms.x2max);
        const auto bh = bh_force_pair::Stencil(1, x, y, z, yl, yr, bhx, bhy, bhz,
            bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
        if (analytic_bh_momentum) {
          bh_momentum_correction = w_bh*(bh_force_pair::Evaluate(x,y,z,bhx,bhy,bhz,
              bh_mass,
              bh_softening,newton_g,bh_sink_mask_radius).ay - bh.Acceleration(dx2));
        }
        dpl += w_bh*bh.dpl;
        dpr += w_bh*bh.dpr;
      }

      // Add momentum source term
      u0(m,IM2,k,j,i) += hdtodx2 * w0(m,IDN,k,j,i) * (dpl + dpr);
      if (analytic_bh_momentum) {
        u0(m,IM2,k,j,i) += block_bdt*w0(m,IDN,k,j,i)*bh_momentum_correction;
      }

      // Add energy source term using Godunov fluxes for any EOS with a conserved energy.
      if (eos_data.use_e) {
        u0(m,IEN,k,j,i) += hdtodx2 * (flx2(m,IDN,k,j,  i) * dpl
                                      + flx2(m,IDN,k,j+1,i) * dpr);
      }
    });
  }

  if (three_d) {
    // x3-direction momentum and energy source terms
    par_for("gravity_x3",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = lat_enabled ? active_indices(a) : a;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      Real rho_fl = w0(m,IDN,k,j,i);
      const Real w_self = use_self_gravity ?
          gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_gmin) : 0.0;
      const Real w_bh = external_bh_enabled ?
          gravity_weight::Weight(rho_fl, rho_gweight_floor, rho_bh_gmin) : 0.0;
      const bool use_self_cell = (w_self > 0.0);
      const bool use_bh_cell = (w_bh > 0.0);
      if (rho_fl <= 0.0 || (!use_self_cell && !use_bh_cell)) return;
      const RegionSize &ms = mbsize.d_view(m);
      Real x = 0.0, y = 0.0, z = 0.0;
      if (excise_enabled || bh_sink_mask_enabled || use_bh_cell) {
        x = CellCenterX(i - is, indcs.nx1, ms.x1min, ms.x1max);
        y = CellCenterX(j - js, indcs.nx2, ms.x2min, ms.x2max);
        z = CellCenterX(k - ks, indcs.nx3, ms.x3min, ms.x3max);
      }
      if (excise_enabled || bh_sink_mask_enabled) {
        if (excise_enabled &&
            problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                excise_r2)) {
          return;
        }
        if (bh_sink_mask_enabled &&
            problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                     bh_sink_mask_y, bh_sink_mask_z,
                                                     bh_sink_mask_radius)) {
          return;
        }
      }
      Real dx3 = ms.dx3;
      Real hdtodx3 = 0.5*block_bdt/dx3;
      Real dpl = 0.0;
      Real dpr = 0.0;
      Real bh_momentum_correction = 0.0;
      if (use_self_cell) {
        dpl = -w_self*(self_phi(m,0,k,  j,i) - self_phi(m,0,k-1,j,i));
        dpr = -w_self*(self_phi(m,0,k+1,j,i) - self_phi(m,0,k,  j,i));
      }
      if (use_bh_cell) {
        const Real zl = CellCenterX(k - 1 - ks, indcs.nx3, ms.x3min, ms.x3max);
        const Real zr = CellCenterX(k + 1 - ks, indcs.nx3, ms.x3min, ms.x3max);
        const auto bh = bh_force_pair::Stencil(2, x, y, z, zl, zr, bhx, bhy, bhz,
            bh_mass, bh_softening, newton_g, bh_sink_mask_radius);
        if (analytic_bh_momentum) {
          bh_momentum_correction = w_bh*(bh_force_pair::Evaluate(x,y,z,bhx,bhy,bhz,
              bh_mass,
              bh_softening,newton_g,bh_sink_mask_radius).az - bh.Acceleration(dx3));
        }
        dpl += w_bh*bh.dpl;
        dpr += w_bh*bh.dpr;
      }

      // Add momentum source term
      u0(m,IM3,k,j,i) += hdtodx3 * w0(m,IDN,k,j,i) * (dpl + dpr);
      if (analytic_bh_momentum) {
        u0(m,IM3,k,j,i) += block_bdt*w0(m,IDN,k,j,i)*bh_momentum_correction;
      }

      // Add energy source term using Godunov fluxes for any EOS with a conserved energy.
      if (eos_data.use_e) {
        u0(m,IEN,k,j,i) += hdtodx3 * (flx3(m,IDN,k,  j,i) * dpl
                                      + flx3(m,IDN,k+1,j,i) * dpr);
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::SinkGravity
//! \brief Add the sink-particle gravitational source term to the gas.
//!
//! Follows the external-BH branch of SourceTerms::Gravity exactly: a discrete gradient of
//! the softened sink potential gives the momentum kick a = G M rhat/(r^2 + s^2), and the
//! same potential differences are contracted with the Godunov mass fluxes for the work
//! term (Mullen, Hanawa & Gammie 2020 form).  Sinks are deliberately absent from the
//! Poisson RHS, so this is additive to the self-gravity term rather than folded into phi.
//! Sink positions are frozen across the whole cycle (ORION2's Lie splitting): the sink
//! push and accretion run once per cycle at a synchronized point, not per RK stage.
//! Interior cells only -- ghosts get the term from the owner through the state exchange.

void SourceTerms::SinkGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                              const Real bdt, DvceArray5D<Real> &u0) {
  sinkparticles::SinkParticles *psink = pmy_pack->psink;
  if (psink == nullptr) return;
  const int nsinks = psink->nsinks;
  if (nsinks <= 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;

  auto sink = psink->sink_gm_pos.d_view;
  using sinkparticles::SinkPotentialSum;

  // Mesh extents and per-direction periodicity for the minimum-image convention.
  const RegionSize &msize = pmy_pack->pmesh->mesh_size;
  const Real lx1 = msize.x1max - msize.x1min;
  const Real lx2 = msize.x2max - msize.x2min;
  const Real lx3 = msize.x3max - msize.x3min;
  const BoundaryFlag *mbcs = pmy_pack->pmesh->mesh_bcs;
  const bool per1 = (mbcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
  const bool per2 = multi_d &&
      (mbcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
  const bool per3 = three_d &&
      (mbcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);

  // Godunov density fluxes used in the energy source term.  Hydro-only by construction:
  // MeshBlockPack::AddPhysics refuses <sink_particles> without <hydro>.
  BandView5D<Real> flx1 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x1f);
  BandView5D<Real> flx2 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x2f);
  BandView5D<Real> flx3 = pmy_pack->phydro->FluxBand(pmy_pack->phydro->uflx.x3f);

  // x1-direction momentum and energy source terms
  par_for("sink_gravity_x1",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
    const Real rho_fl = w0(m,IDN,k,j,i);
    if (rho_fl <= 0.0) return;
    const Real x = CellCenterX(i - is, indcs.nx1,
                               mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    const Real xl = CellCenterX(i - 1 - is, indcs.nx1,
                                mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    const Real xr = CellCenterX(i + 1 - is, indcs.nx1,
                                mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    const Real y = CellCenterX(j - js, indcs.nx2,
                               mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    const Real z = CellCenterX(k - ks, indcs.nx3,
                               mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
    const Real phi_c = SinkPotentialSum(sink, nsinks, x, y, z,
                                        lx1, lx2, lx3, per1, per2, per3);
    const Real dpl = -(phi_c - SinkPotentialSum(sink, nsinks, xl, y, z,
                                                lx1, lx2, lx3, per1, per2, per3));
    const Real dpr = -(SinkPotentialSum(sink, nsinks, xr, y, z,
                                        lx1, lx2, lx3, per1, per2, per3) - phi_c);
    const Real hdtodx1 = 0.5*block_bdt/mbsize.d_view(m).dx1;

    u0(m,IM1,k,j,i) += hdtodx1 * rho_fl * (dpl + dpr);
    if (eos_data.use_e) {
      u0(m,IEN,k,j,i) += hdtodx1 * (flx1(m,IDN,k,j,i  ) * dpl
                                    + flx1(m,IDN,k,j,i+1) * dpr);
    }
  });

  if (multi_d) {
    // x2-direction momentum and energy source terms
    par_for("sink_gravity_x2",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = lat_enabled ? active_indices(a) : a;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      const Real rho_fl = w0(m,IDN,k,j,i);
      if (rho_fl <= 0.0) return;
      const Real x = CellCenterX(i - is, indcs.nx1,
                                 mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - js, indcs.nx2,
                                 mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real yl = CellCenterX(j - 1 - js, indcs.nx2,
                                  mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real yr = CellCenterX(j + 1 - js, indcs.nx2,
                                  mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, indcs.nx3,
                                 mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      const Real phi_c = SinkPotentialSum(sink, nsinks, x, y, z,
                                          lx1, lx2, lx3, per1, per2, per3);
      const Real dpl = -(phi_c - SinkPotentialSum(sink, nsinks, x, yl, z,
                                                  lx1, lx2, lx3, per1, per2, per3));
      const Real dpr = -(SinkPotentialSum(sink, nsinks, x, yr, z,
                                          lx1, lx2, lx3, per1, per2, per3) - phi_c);
      const Real hdtodx2 = 0.5*block_bdt/mbsize.d_view(m).dx2;

      u0(m,IM2,k,j,i) += hdtodx2 * rho_fl * (dpl + dpr);
      if (eos_data.use_e) {
        u0(m,IEN,k,j,i) += hdtodx2 * (flx2(m,IDN,k,j,  i) * dpl
                                      + flx2(m,IDN,k,j+1,i) * dpr);
      }
    });
  }

  if (three_d) {
    // x3-direction momentum and energy source terms
    par_for("sink_gravity_x3",DevExeSpace(),0,nwork1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int a, int k, int j, int i) {
      const int m = lat_enabled ? active_indices(a) : a;
      const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
      const Real rho_fl = w0(m,IDN,k,j,i);
      if (rho_fl <= 0.0) return;
      const Real x = CellCenterX(i - is, indcs.nx1,
                                 mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - js, indcs.nx2,
                                 mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, indcs.nx3,
                                 mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      const Real zl = CellCenterX(k - 1 - ks, indcs.nx3,
                                  mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      const Real zr = CellCenterX(k + 1 - ks, indcs.nx3,
                                  mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      const Real phi_c = SinkPotentialSum(sink, nsinks, x, y, z,
                                          lx1, lx2, lx3, per1, per2, per3);
      const Real dpl = -(phi_c - SinkPotentialSum(sink, nsinks, x, y, zl,
                                                  lx1, lx2, lx3, per1, per2, per3));
      const Real dpr = -(SinkPotentialSum(sink, nsinks, x, y, zr,
                                          lx1, lx2, lx3, per1, per2, per3) - phi_c);
      const Real hdtodx3 = 0.5*block_bdt/mbsize.d_view(m).dx3;

      u0(m,IM3,k,j,i) += hdtodx3 * rho_fl * (dpl + dpr);
      if (eos_data.use_e) {
        u0(m,IEN,k,j,i) += hdtodx3 * (flx3(m,IDN,k,  j,i) * dpl
                                      + flx3(m,IDN,k+1,j,i) * dpr);
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn SourceTerms::BeamSource()
//! \brief Add beam of radiation at position (pos1,pos2,pos3) moving in direction
//! (dir1,dir2,dir3) with physical width and angular spread (width,spread)

void SourceTerms::BeamSource(DvceArray5D<Real> &i0, const Real bdt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = (pmy_pack->nmb_thispack-1);
  int nang1 = (pmy_pack->prad->prgeo->nangles-1);

  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  Real &p1 = pos1, &p2 = pos2, &p3 = pos3;
  Real &d1 = dir1, &d2 = dir2, &d3 = dir3;
  Real &dii_dt_ = dii_dt;
  Real &width_ = width;
  Real &spread_ = spread;

  auto &tc = pmy_pack->prad->tetcov_c;
  auto &nh_c_ = pmy_pack->prad->nh_c;
  auto &tet_c_ = pmy_pack->prad->tet_c;
  par_for("rad_beam",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1 +2.0*glower[1][2]*dx1*dx2 + 2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2 +2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = glower[0][0];
    Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3);
    Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2 + 2.0*glower[1][3]*d1*d3
                + glower[2][2]*d2*d2 + 2.0*glower[2][3]*d2*d3
                + glower[3][3]*d3*d3;
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = glower[0][0]*d0 + glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3;
    Real dc1 = glower[0][1]*d0 + glower[1][1]*d1 + glower[1][2]*d2 + glower[1][3]*d3;
    Real dc2 = glower[0][2]*d0 + glower[1][2]*d1 + glower[2][2]*d2 + glower[2][3]*d3;
    Real dc3 = glower[0][3]*d0 + glower[1][3]*d1 + glower[2][3]*d2 + glower[3][3]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        Real n0 = tet_c_(m,0,0,k,j,i);
        Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1)
                 + tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
        i0(m,n,k,j,i) += n0*n_0*dii_dt_*bdt;
      }
    }
  });

  return;
}
