#ifndef UTILS_LAT_HOST_FAST_HPP_
#define UTILS_LAT_HOST_FAST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_host_fast.hpp
//! \brief Deferred completion of the LAT boundary-refresh state sends.
//!
//! Driver::RefreshHydroLATBoundaries used to end with MHD::ClearSend(-1), an MPI_Wait on
//! the U/B sends it had just started.  The GPU is idle for the whole of that wait: the
//! refresh has already queued its last kernel (the ghost-band C2P) by then.  Nothing
//! needs those send requests until the NEXT pack on the same boundary object, so the wait
//! is latched here and drained by MHD::SendU / MHD::SendB / MHD::ClearSend instead.  Only
//! WHEN the host blocks changes, never what any kernel or MPI call reads or writes, so
//! the result is bit-identical either way.
//!
//! The latch is unconditional: it was measured on the production ten-GPU node and, since
//! only WHEN the host blocks changes, there is no configuration in which the immediate
//! wait is more correct -- only slower.

namespace lat_host {

inline bool &PendingLATStateSendClear() {
  static bool pending = false;
  return pending;
}

}  // namespace lat_host

#endif  // UTILS_LAT_HOST_FAST_HPP_
