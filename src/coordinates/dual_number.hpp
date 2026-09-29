#ifndef COORDINATES_DUAL_NUMBER_HPP_
#define COORDINATES_DUAL_NUMBER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

//! \file dual_number.hpp
//! \brief Minimal, header-only forward-mode automatic-differentiation scalar.
//!
//! DualN<N> carries a value together with N partial derivatives.  Dual4 differentiates
//! with respect to (x, y, z, t): it exists so that a metric kernel can be templated on
//! the scalar type and instantiated once with Real (value only, unchanged cost) and once
//! with Dual4 (value plus all four derivatives in a single pass), instead of being
//! re-evaluated nine times on a finite-difference stencil.  Dual3 differentiates with
//! respect to (x, y, z) alone, for a metric frozen at one instant (the direct-field
//! geodesics of radiation_m1_direct_trace.hpp): there a Dual4's time slot would be zero
//! and still cost a quarter of every product.
//!
//! Two properties are load-bearing and must be preserved by anyone editing this file:
//!
//!  1. Every operator computes its value component with exactly the same floating-point
//!     operation, on exactly the same operands, in exactly the same order as the plain
//!     Real expression it replaces.  A kernel templated on the scalar type therefore
//!     returns a *bitwise* identical value whether it was instantiated with Real, Dual3
//!     or Dual4; only the partials are extra work.  In particular division always
//!     performs the true divide a.v/b.v for the value and uses a separate reciprocal only
//!     for the partials -- never value = a.v*(1/b.v), which differs in the last bit.
//!
//!  2. There is no conversion operator from a dual back to Real.  Together with living in
//!     its own namespace this keeps unqualified sqrt()/fabs() inside the metric kernels
//!     resolving to ::sqrt/::fabs for Real arguments (ordinary lookup) and to the
//!     overloads below for dual arguments (argument-dependent lookup), with no shadowing
//!     of the C library names anywhere in namespace adm.
//!
//! Everything lives in registers; nothing allocates.

#include <math.h>

#include "athena.hpp"

namespace adm {
namespace dual {

//----------------------------------------------------------------------------------------
//! \struct DualN
//! \brief Value plus N partials, in the slot order the caller seeds.

template <int N>
struct DualN {
  Real v;
  Real d[N];

  KOKKOS_INLINE_FUNCTION
  DualN() {}

  // Implicit on purpose: a plain Real appearing in a templated expression is a constant
  // with vanishing partials, exactly as it is in the Real instantiation.
  KOKKOS_INLINE_FUNCTION
  DualN(const Real value) : v(value) {   // NOLINT(runtime/explicit)
    for (int n = 0; n < N; ++n) d[n] = 0.0;
  }

  //! The value and all N partials, in slot order.
  template <class... P>
  KOKKOS_INLINE_FUNCTION
  DualN(const Real value, const P... partials) : v(value), d{partials...} {
    static_assert(sizeof...(P) == N, "a DualN is seeded with exactly N partials");
  }
};

//! Partials with respect to (x, y, z, t).
using Dual4 = DualN<4>;
//! Partials with respect to (x, y, z).
using Dual3 = DualN<3>;

//----------------------------------------------------------------------------------------
// Arithmetic.  The Real-mixed overloads are not just conveniences: they avoid N
// multiply-adds against a known-zero partial vector that a compiler would otherwise have
// to prove away.

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator+(const DualN<N> &a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a.v + b.v;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n] + b.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator+(const DualN<N> &a, const Real b) {
  DualN<N> r;
  r.v = a.v + b;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator+(const Real a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a + b.v;
  for (int n = 0; n < N; ++n) r.d[n] = b.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator-(const DualN<N> &a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a.v - b.v;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n] - b.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator-(const DualN<N> &a, const Real b) {
  DualN<N> r;
  r.v = a.v - b;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator-(const Real a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a - b.v;
  for (int n = 0; n < N; ++n) r.d[n] = -b.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator-(const DualN<N> &a) {
  DualN<N> r;
  r.v = -a.v;
  for (int n = 0; n < N; ++n) r.d[n] = -a.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator*(const DualN<N> &a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a.v*b.v;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n]*b.v + a.v*b.d[n];
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator*(const DualN<N> &a, const Real b) {
  DualN<N> r;
  r.v = a.v*b;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n]*b;
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator*(const Real a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a*b.v;
  for (int n = 0; n < N; ++n) r.d[n] = a*b.d[n];
  return r;
}

// value = a.v/b.v is a true divide so that it matches the Real path bit for bit; the
// reciprocal is only used for the partials.  Where the numerator is the literal 1.0 the
// two divisions share an operand and the compiler emits just one.
template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator/(const DualN<N> &a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a.v/b.v;
  const Real inv = 1.0/b.v;
  for (int n = 0; n < N; ++n) r.d[n] = (a.d[n] - r.v*b.d[n])*inv;
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator/(const Real a, const DualN<N> &b) {
  DualN<N> r;
  r.v = a/b.v;
  const Real inv = 1.0/b.v;
  for (int n = 0; n < N; ++n) r.d[n] = -r.v*b.d[n]*inv;
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> operator/(const DualN<N> &a, const Real b) {
  DualN<N> r;
  r.v = a.v/b;
  const Real inv = 1.0/b;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n]*inv;
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> &operator+=(DualN<N> &a, const DualN<N> &b) {
  a.v += b.v;
  for (int n = 0; n < N; ++n) a.d[n] += b.d[n];
  return a;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> &operator*=(DualN<N> &a, const DualN<N> &b) {
  a = a*b;
  return a;
}

//----------------------------------------------------------------------------------------
// Elementary functions.  Found by argument-dependent lookup from the templated kernels,
// so the unqualified call sites read identically in the Real and the dual instantiations.

//! At a zero argument the true derivative is infinite.  The Real path never sees it (its
//! stencil samples neighbours a finite step away and returns something finite), so
//! returning a zero partial instead of a NaN keeps the two paths equally well behaved on
//! the measure-zero set where the radius, or the Kerr discriminant, vanishes exactly.
template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> sqrt(const DualN<N> &a) {
  DualN<N> r;
  r.v = ::sqrt(a.v);
  const Real fac = (r.v > 0.0) ? 0.5/r.v : 0.0;
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n]*fac;
  return r;
}

template <int N>
KOKKOS_INLINE_FUNCTION
DualN<N> fabs(const DualN<N> &a) {
  DualN<N> r;
  r.v = ::fabs(a.v);
  const Real sgn = (a.v > 0.0) ? 1.0 : ((a.v < 0.0) ? -1.0 : 0.0);
  for (int n = 0; n < N; ++n) r.d[n] = a.d[n]*sgn;
  return r;
}

} // namespace dual
} // namespace adm

#endif // COORDINATES_DUAL_NUMBER_HPP_
