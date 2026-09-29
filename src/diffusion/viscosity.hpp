#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for
//  viscosity. Currently only Navier-Stokes (uniform, isotropic) shear viscosity
//  is implemented. TODO: add Braginskii viscosity

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity {
 public:
  Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real nu_iso;      // coefficient of isotropic kinematic shear viscosity
  Real nu_aniso;    // coefficient of anisotropic kinematic shear viscosity

  // function to add viscous fluxes to Hydro and/or MHD fluxes.  The flux register is
  // taken on its storage band (MHD keeps uflx on the flux band, mhd.hpp FluxBand); a
  // ghost-extended register (Hydro) is the zero-origin band, via the second overload.
  void AddViscousFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        const BandFaceFld5D<Real> &f);
  void AddViscousFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f);
  void AddViscousFluxIso(const DvceArray5D<Real> &w,const EOS_Data &eos,
                         const BandFaceFld5D<Real> &f);
  void AddViscousFluxAniso(const DvceArray5D<Real> &w,const EOS_Data &eos,
                           const BandFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_VISCOSITY_HPP_
