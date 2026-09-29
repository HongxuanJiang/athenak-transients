//========================================================================================
// AthenaK astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_task_list.cpp
//! \brief functions for MultigridTaskList class

// C headers

// Project headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "multigrid.hpp"

TaskStatus MultigridDriver::ClearRecv(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  if (pmg == nullptr || pmg->pbval == nullptr) return TaskStatus::complete;
  return pmg->pbval->ClearRecvMG();
}

TaskStatus MultigridDriver::ClearSend(Driver *pdrive, int stage) {
  (void)pdrive;
  (void)stage;
  if (pmg == nullptr || pmg->pbval == nullptr) return TaskStatus::complete;
  return pmg->pbval->ClearSendMG();
}

TaskStatus MultigridDriver::SendBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  tstat = pmg->pbval->PackAndSendMG(u);
  return tstat;
}

TaskStatus MultigridDriver::RecvBoundary(Driver *pdrive, int stage) {
  TaskStatus tstat;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  tstat = pmg->pbval->RecvAndUnpackMG(u);
  return tstat;
}

TaskStatus MultigridDriver::StartReceive(Driver *pdrive, int stage) {
  TaskStatus tstat;
  tstat = pmg->pbval->InitRecvMG(pmg->nvar_);
  return tstat;
}

TaskStatus MultigridDriver::SendBoundaryForProlongation(Driver *pdrive, int stage) {
  DvceArray5D<Real> u = pmg->GetCurrentData();
  return pmg->pbval->PackAndSendMG(u, true);
}

TaskStatus MultigridDriver::RecvBoundaryForProlongation(Driver *pdrive, int stage) {
  DvceArray5D<Real> u = pmg->GetCurrentData();
  return pmg->pbval->RecvAndUnpackMG(u, true);
}

TaskStatus MultigridDriver::StartReceiveForProlongation(Driver *pdrive, int stage) {
  return pmg->pbval->InitRecvMG(pmg->nvar_, true);
}

TaskStatus MultigridDriver::SmoothRed(Driver *pdrive, int stage) {
  if (active_reduce_same_exchange_) {
    for (int s = 0; s < local_sweeps_per_exchange_; ++s) {
      pmg->SmoothPack(coffset_);
      pmg->SmoothPack(1 - coffset_);
    }
  } else {
    pmg->SmoothPack(coffset_);
  }
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::SmoothBlack(Driver *pdrive, int stage) {
  if (active_reduce_same_exchange_) {
    return TaskStatus::complete;
  }
  pmg->SmoothPack(1-coffset_);
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Restrict(Driver *pdrive, int stage) {
  pmg->RestrictPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::Prolongate(Driver *pdrive, int stage) {
  pmg->ProlongateAndCorrectPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::PrepareCorrection(Driver *pdrive, int stage) {
  pmg->ComputeCorrection();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::ProlongatePreparedCorrection(Driver *pdrive, int stage) {
  pmg->ProlongatePreparedCorrectionPack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FMGProlongateTask(Driver *pdrive, int stage) {
  pmg->FMGProlongatePack();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::CalculateFASRHS(Driver *pdrive, int stage) {
  if (current_level_ < fmglevel_) {
    pmg->StoreOldData();
    pmg->CalculateFASRHSPack();
  }
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FillFCBoundary(Driver *pdrive, int stage) {
  if (nreflevel_ > 0) {
    DvceArray5D<Real> u = pmg->GetCurrentData();
    pmg->pbval->FillFineCoarseMGGhosts(u);
  }
  // A gravity-specific nonperiodic BC on a mesh-periodic face must be applied
  // after wrapped-neighbor communication, which otherwise overwrites it.
  if (MGBoundaryOverridesPeriodicMesh()) ApplyPhysicalBoundariesBlocks();
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::FillFCBoundaryForProlongation(Driver *pdrive, int stage) {
  if (nreflevel_ == 0) return TaskStatus::complete;
  DvceArray5D<Real> u = pmg->GetCurrentData();
  return pmg->pbval->FillFineCoarseMGGhosts(u, true);
}

TaskStatus MultigridDriver::PhysicalBoundary(Driver *pdrive, int stage) {
  if (pmy_pack_->pmesh->strictly_periodic || MGBoundaryOverridesPeriodicMesh()) {
    return TaskStatus::complete;
  }
  if (pmg == mglevels_) {
    // Meshblock-level multigrid arrays use explicit physical BC fill here.
    // Root/octet levels apply BCs in dedicated driver routines.
    ApplyPhysicalBoundariesBlocks();
  }
  return TaskStatus::complete;
}

TaskStatus MultigridDriver::PhysicalBoundaryForProlongation(Driver *pdrive, int stage) {
  if (pmy_pack_->pmesh->strictly_periodic && !MGBoundaryOverridesPeriodicMesh()) {
    return TaskStatus::complete;
  }
  if (pmg == mglevels_) {
    // Prolongation reads diagonal coarse cells. Complete physical edge/corner
    // ghosts after same-level and fine/coarse communication has filled tangential data.
    // This variant is for arrays holding the SOLUTION -- the FMG prolongation list,
    // whose FMGProlongatePack (multigrid.cpp:782) overwrites the finer level directly.
    // The correction ascent must use the homogeneous variant below.
    ApplyPhysicalBoundariesBlocks(true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MultigridDriver::PhysicalBoundaryForCorrectionProlongation
//! \brief Physical BCs for the up-leg, where the array holds a CORRECTION.
//!
//! This runs after PrepareCorrection, so the array holds (u - uold), not the solution.
//! A correction must satisfy the HOMOGENEOUS form of every boundary condition.  The
//! zerofixed (ghost = -interior) and zerograd (ghost = interior) rules already are
//! homogeneous and are idempotent here, but the multipole rule
//! ghost = 2*phi - interior is not: it injects a fixed +2*phi into the correction on
//! every V-cycle.  That term is independent of u, so it does not change the iteration
//! matrix -- it just moves the fixed point, which is why the defect settles onto a hard
//! floor proportional to the total mass within a couple of cycles and then does not
//! respond to any further cycles or to the multipole order.
//!
//! Upstream (8a6a8efa) does not hit this because its up-leg is
//! PhysicalBoundary -> ProlongateFCBoundary -> Prolongate, and its Prolongate calls
//! ComputeCorrection() AFTER the boundary fill, so the 2*phi cancels between u and uold.
//! This tree inverted that order (see the comment in SetMGTaskListToFiner) to avoid
//! stale diagonal ghosts, which is what exposed the inhomogeneous term.

TaskStatus MultigridDriver::PhysicalBoundaryForCorrectionProlongation(Driver *pdrive,
                                                                     int stage) {
  if (pmy_pack_->pmesh->strictly_periodic && !MGBoundaryOverridesPeriodicMesh()) {
    return TaskStatus::complete;
  }
  if (pmg == mglevels_) {
    ApplyPhysicalBoundariesBlocks(true, true);
  }
  return TaskStatus::complete;
}

bool MultigridDriver::MGBoundaryOverridesPeriodicMesh() const {
  int last_face = BoundaryFace::outer_x1;
  if (pmy_mesh_->multi_d) last_face = BoundaryFace::outer_x2;
  if (pmy_mesh_->three_d) last_face = BoundaryFace::outer_x3;
  for (int f = BoundaryFace::inner_x1; f <= last_face; ++f) {
    const BoundaryFlag mesh_bc = pmy_mesh_->mesh_bcs[f];
    const bool mesh_wraps = (mesh_bc == BoundaryFlag::periodic ||
                             mesh_bc == BoundaryFlag::shear_periodic);
    if (mesh_wraps && mg_mesh_bcs_[f] != BoundaryFlag::periodic) return true;
  }
  return false;
}


//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToFiner(int nsmooth, int ngh, int flag)
//! \brief Set the task list for prolongation and post smoothing

void MultigridDriver::SetMGTaskListToFiner(int nsmooth, int ngh, int flag,
                                           bool reduce_same_exchange) {
  if (mg_tl_to_finer_valid_ && mg_tl_to_finer_nsmooth_ == nsmooth &&
      mg_tl_to_finer_flag_ == flag &&
      mg_tl_to_finer_reduce_same_ == reduce_same_exchange &&
      pmy_pack_->tl_map.count("mg_to_finer") > 0) {
    return;
  }
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_finer");
  tl.emplace(std::make_pair("mg_to_finer", std::make_shared<TaskList>()));
  TaskID none(0);
  if (flag == 1) {
    // First time on meshblock levels: no boundary comm before prolongation
    id.prolongate = tl["mg_to_finer"]->AddTask(&MultigridDriver::Prolongate, this, none);
  } else {
    // Form the coarse correction first, then exchange its complete halo.  uold only
    // has face halos, so exchanging u before subtraction would leave stale diagonals.
    id.prepare_correction = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::PrepareCorrection, this, none);
    id.ircv0    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::StartReceiveForProlongation, this, none);
    id.send0    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::SendBoundaryForProlongation, this, id.prepare_correction);
    id.recv0    = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::RecvBoundaryForProlongation, this, id.ircv0 | id.send0);
    id.fc_ghosts0 = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::FillFCBoundaryForProlongation, this, id.recv0);
    id.physb0   = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::PhysicalBoundaryForCorrectionProlongation, this, id.fc_ghosts0);
    id.prolongate = tl["mg_to_finer"]->AddTask(
        &MultigridDriver::ProlongatePreparedCorrection, this, id.physb0);
  }

  TaskID clear_dep = id.prolongate;
  if (nsmooth > 0) {
    // Fine-level boundary comm after prolongation
    id.ircv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
        id.prolongate);
    id.send1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
        id.prolongate);
    id.recv1    = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
        id.ircv1 | id.send1);
    id.physb1   = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
        id.send1);
    id.fc_ghosts_prol = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                   id.physb1 | id.recv1);

    id.smoothR = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this,
        id.fc_ghosts_prol);
    if (reduce_same_exchange) {
      id.ircvR = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
                                            id.fc_ghosts_prol);
      id.smoothB = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this,
          id.smoothR);
      id.sendB = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
          id.smoothB);
      id.recvB = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
                                            id.ircvR | id.sendB);
      id.physbB = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
          id.sendB);
      id.fc_ghostsB = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                 id.physbB | id.recvB);
      clear_dep = id.fc_ghostsB;

      if (nsmooth > 1) {
        id.smoothR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this,
            id.fc_ghostsB);
        id.ircvR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
            id.fc_ghostsB);
        id.smoothB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this,
            id.smoothR2);
        id.sendB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothB2);
        id.recvB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
                                               id.ircvR2 | id.sendB2);
        id.physbB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
            id.sendB2);
        id.fc_ghostsB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary,
            this,
                                                    id.physbB2 | id.recvB2);
        clear_dep = id.fc_ghostsB2;
      }
    } else {
      id.ircvR = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
          id.fc_ghosts_prol);
      id.sendR = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
          id.smoothR);
      id.recvR = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
          id.ircvR | id.sendR);
      id.physbR = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
          id.sendR);
      id.fc_ghostsR = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                 id.physbR | id.recvR);

      id.smoothB = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this,
          id.fc_ghostsR);
      clear_dep = id.smoothB;

      if (nsmooth > 1) {
        id.ircvB = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
            id.fc_ghostsR);
        id.sendB = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothB);
        id.recvB = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
            id.ircvB | id.sendB);
        id.physbB = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
            id.recvB);
        id.fc_ghostsB = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this,
            id.physbB);

        id.smoothR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothRed, this,
            id.fc_ghostsB);

        id.ircvR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
            id.fc_ghostsB);
        id.sendR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothR2);
        id.recvR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
            id.ircvR2 | id.sendR2);
        id.physbR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
            id.sendR2);
        id.fc_ghostsR2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary,
            this,
                                                    id.physbR2 | id.recvR2);

        id.smoothB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::SmoothBlack, this,
            id.fc_ghostsR2);
        clear_dep = id.smoothB2;
      }
    }
  }

  constexpr bool do_final_finest_exchange = false;
  if (flag == 2 && do_final_finest_exchange) {
    id.clear_sendB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this,
        clear_dep);
    id.clear_recvB2 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this,
        id.clear_sendB2);
    TaskID last_clear = id.clear_recvB2;

    TaskID ircvL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::StartReceive, this,
        last_clear);
    TaskID send_dep = clear_dep | last_clear;
    TaskID sendL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::SendBoundary, this,
        send_dep);
    TaskID recvL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::RecvBoundary, this,
        ircvL | sendL);
    TaskID physL  = tl["mg_to_finer"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
        sendL);
    TaskID fcL    = tl["mg_to_finer"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                               physL | recvL);
    id.clear_send0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this, fcL);
    id.clear_recv0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this,
        id.clear_send0);
  } else {
    id.clear_send0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearSend, this,
        clear_dep);
    id.clear_recv0 = tl["mg_to_finer"]->AddTask(&MultigridDriver::ClearRecv, this,
        id.clear_send0);
  }
  mg_tl_to_finer_valid_ = true;
  mg_tl_to_finer_nsmooth_ = nsmooth;
  mg_tl_to_finer_flag_ = flag;
  mg_tl_to_finer_reduce_same_ = reduce_same_exchange;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridDriver::SetMGTaskListFMGProlongate(int ngh)
//! \brief Set the task list for FMG prolongation only (no smoothing)

void MultigridDriver::SetMGTaskListFMGProlongate(int ngh) {
  if (mg_tl_fmg_prolongate_valid_ && pmy_pack_->tl_map.count("mg_fmg_prolongate") > 0) {
    return;
  }
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_fmg_prolongate");
  tl.emplace(std::make_pair("mg_fmg_prolongate", std::make_shared<TaskList>()));
  TaskID none(0);

  // Boundary comm before prolongation
  id.ircv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::StartReceiveForProlongation, this, none);
  id.send0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::SendBoundaryForProlongation, this, none);
  id.recv0    = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::RecvBoundaryForProlongation, this,
                  id.ircv0 | id.send0);
  id.fc_ghosts0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FillFCBoundaryForProlongation, this,
                  id.recv0);
  id.physb0   = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::PhysicalBoundaryForProlongation, this,
                  id.fc_ghosts0);

  // FMG prolongation (direct overwrite)
  id.fmg_prolongate = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::FMGProlongateTask, this, id.physb0);

  id.clear_send0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearSend, this, id.fmg_prolongate);
  id.clear_recv0 = tl["mg_fmg_prolongate"]->AddTask(
                  &MultigridDriver::ClearRecv, this, id.clear_send0);
  mg_tl_fmg_prolongate_valid_ = true;
}

//----------------------------------------------------------------------------------------
//! \fn void MultigridTaskList::SetMGTaskListToCoarser(int nsmooth, int ngh)
//! \brief Set the task list for pre smoothing and restriction

void MultigridDriver::SetMGTaskListToCoarser(int nsmooth, int cycle,
                                             bool reduce_same_exchange) {
  if (mg_tl_to_coarser_valid_ && mg_tl_to_coarser_nsmooth_ == nsmooth &&
      mg_tl_to_coarser_skip_initial_ == skip_coarser_initial_exchange_once_ &&
      mg_tl_to_coarser_reduce_same_ == reduce_same_exchange &&
      pmy_pack_->tl_map.count("mg_to_coarser") > 0) {
    return;
  }
  auto &tl = pmy_pack_->tl_map;
  tl.erase("mg_to_coarser");
  tl.emplace(std::make_pair("mg_to_coarser",std::make_shared<TaskList>()));
  TaskID none(0);
  TaskID start_dep = none;
  if (!skip_coarser_initial_exchange_once_) {
    id.ircv0    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
        none);
    id.send0    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
        none);
    id.recv0    = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
                                               id.ircv0 | id.send0);
    id.physb0   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
        id.send0);
    id.fc_ghosts0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                 id.physb0 | id.recv0);
    start_dep = id.fc_ghosts0;
  } else {
    id.fc_ghosts0 = none;
  }

  id.calc_rhs   = tl["mg_to_coarser"]->AddTask(&MultigridDriver::CalculateFASRHS, this,
      start_dep);

  if (nsmooth > 0) {
    if (reduce_same_exchange) {
      id.ircvR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
          start_dep);
      id.smoothR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this,
          id.calc_rhs);
      id.smoothB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this,
          id.smoothR);
      id.sendB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
          id.smoothB);
      id.recvB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
          id.ircvR | id.sendB);
      id.physbB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
          id.sendB);
      id.fc_ghostsB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                   id.physbB | id.recvB);
      if (nsmooth > 1) {
        id.smoothR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this,
            id.fc_ghostsB);
        id.ircvR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
            id.fc_ghostsB);
        id.smoothB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this,
            id.smoothR2);
        id.sendB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothB2);
        id.recvB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
                                                 id.ircvR2 | id.sendB2);
        id.physbB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary,
            this, id.sendB2);
        id.fc_ghostsB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary,
            this,
                                                      id.physbB2 | id.recvB2);
        id.restrict_ = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this,
            id.fc_ghostsB2);
      } else {
        id.restrict_ = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this,
            id.fc_ghostsB);
      }
    } else {
      id.ircvR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
          start_dep);
      id.smoothR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this,
          id.calc_rhs);
      id.sendR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
          id.smoothR);
      id.recvR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
          id.ircvR | id.sendR);
      id.physbR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
          id.sendR);
      id.fc_ghostsR = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this,
                                                   id.physbR | id.recvR);

      id.ircvB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
          id.fc_ghostsR);
      id.smoothB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this,
          id.fc_ghostsR);
      id.sendB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
          id.smoothB);
      id.recvB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
          id.ircvB | id.sendB);
      id.physbB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary, this,
          id.recvB);
      id.fc_ghostsB = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary, this,
          id.physbB);
      if (nsmooth > 1) {
        id.smoothR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothRed, this,
            id.fc_ghostsB);

        id.ircvR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
            id.smoothR2);
        id.sendR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothR2);
        id.recvR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
            id.ircvR2 | id.sendR2);
        id.physbR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary,
            this, id.sendR2);
        id.fc_ghostsR2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary,
            this,
                                                      id.physbR2 | id.recvR2);

        id.smoothB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SmoothBlack, this,
            id.fc_ghostsR2);

        id.ircvB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::StartReceive, this,
            id.fc_ghostsR2);
        id.sendB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::SendBoundary, this,
            id.smoothB2);
        id.recvB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::RecvBoundary, this,
            id.ircvB2 | id.sendB2);
        id.physbB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::PhysicalBoundary,
            this, id.sendB2);
        id.fc_ghostsB2 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::FillFCBoundary,
            this,
                                                      id.physbB2 | id.recvB2);

        id.restrict_ = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this,
            id.fc_ghostsB2);
      } else {
        id.restrict_ = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this,
            id.fc_ghostsB);
      }
    }
  } else {
    id.restrict_  = tl["mg_to_coarser"]->AddTask(&MultigridDriver::Restrict, this,
        id.calc_rhs);
  }
  id.clear_send0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearSend, this,
      id.restrict_);
  id.clear_recv0 = tl["mg_to_coarser"]->AddTask(&MultigridDriver::ClearRecv, this,
      id.clear_send0);
  mg_tl_to_coarser_valid_ = true;
  mg_tl_to_coarser_nsmooth_ = nsmooth;
  mg_tl_to_coarser_skip_initial_ = skip_coarser_initial_exchange_once_;
  mg_tl_to_coarser_reduce_same_ = reduce_same_exchange;
}
