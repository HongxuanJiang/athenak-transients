//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sink_particles_nbody.cpp
//! \brief sink-sink N-body advance over one cycle: the Numerical Recipes Bulirsch-Stoer
//! driver with Stoermer's rule for y'' = f(y) and polynomial extrapolation, ported from
//! ORION2 ParticleInt.cpp.
//!
//! Two ORION2 defects are fixed here and must stay fixed:
//!  * R12 - every file-scope global and function-local `static` of the original is a
//!    member of the per-call BSWork below, so the integrator is reentrant.
//!  * R13 - the velocity error scale uses `distmin` (the minimum pair separation), not
//!    the separation of whichever pair happened to be examined last.
//! The list is replicated, so this runs redundantly and identically on every rank; it
//! needs and must contain no MPI.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "sink_particles/sink_particles.hpp"

namespace sinkparticles {

namespace {

constexpr int kMaxStp = 10000;
constexpr int kKmaxx = 12;
constexpr int kImaxx = kKmaxx + 1;
constexpr Real kSafe1 = 0.25;
constexpr Real kSafe2 = 0.7;
constexpr Real kRedMax = 1.0e-5;
constexpr Real kRedMin = 0.7;
constexpr Real kScalMx = 0.1;
constexpr Real kTinyNr = 1.0e-30;

//! \struct NBodyCtx
//! \brief constants of the pair force for one advance

struct NBodyCtx {
  Real gconst;
  Real soften;
  Real len[3];
  bool periodic[3];
};

//! \struct BSWork
//! \brief reentrant workspace.  All vectors are 1-based (index 0 unused) to keep the
//! transcription of the Numerical Recipes algebra literal.

struct BSWork {
  int nv;
  std::vector<Real> xtab;   // [1..KMAXX]        pzextr abscissae (ORION2 global `x`)
  std::vector<Real> dtab;   // [1..nv][1..KMAXX] pzextr tableau (ORION2 global `d`)
  std::vector<Real> err;    // [1..KMAXX]
  std::vector<Real> yerr, ysav, yseq, ctmp, ytemp;
  Real a[kImaxx + 2];
  Real alf[kKmaxx + 2][kKmaxx + 2];
  int nseq[kImaxx + 2];
  int first, kmax, kopt;
  Real epsold, xnew;

  explicit BSWork(int nvar) :
      nv(nvar),
      xtab(kKmaxx + 2, 0.0),
      dtab(static_cast<std::size_t>(nvar + 1)*(kKmaxx + 2), 0.0),
      err(kKmaxx + 2, 0.0),
      yerr(nvar + 1, 0.0), ysav(nvar + 1, 0.0), yseq(nvar + 1, 0.0),
      ctmp(nvar + 1, 0.0), ytemp(nvar + 1, 0.0),
      first(1), kmax(kKmaxx), kopt(kKmaxx), epsold(-1.0), xnew(-1.0e29) {
    for (int n=0; n<kImaxx+2; ++n) { a[n] = 0.0; }
    // ORION2's `alf` is a function-local static and is therefore zero-initialized; the
    // NR algebra reads alf[k][k], which is never written, so the zeros are load-bearing.
    for (int p=0; p<kKmaxx+2; ++p) {
      for (int q=0; q<kKmaxx+2; ++q) { alf[p][q] = 0.0; }
    }
    // nseq = {0,1,2,...,12}; nseq[13] is left 0, exactly as ORION2 ParticleInt.cpp:113.
    for (int n=0; n<kImaxx+2; ++n) { nseq[n] = (n <= kKmaxx) ? n : 0; }
  }

  Real &d(int j, int k) { return dtab[static_cast<std::size_t>(j)*(kKmaxx + 2) + k]; }
};

//----------------------------------------------------------------------------------------
//! \fn Derivs
//! \brief pair acceleration a_i = -G m_j r_hat/(r^2 + s^2), equal and opposite by
//! construction so sink-sink momentum is conserved exactly (ORION2 ParticleDerivs).
//! Separations use the minimum image on periodic directions.  ORION2 does not wrap here
//! (its FOF has the same gap, map R19b); wrapping keeps this consistent with the gas
//! kernels and with the merge, and is a no-op on non-periodic meshes.

void Derivs(const Real *y, Real *acc, const Real *mass, const NBodyCtx &ctx, int nvar) {
  const int np = nvar/6;
  for (int n=1; n<=nvar; ++n) { acc[n] = 0.0; }
  for (int i=0; i<np; ++i) {
    for (int j=i+1; j<np; ++j) {
      Real dsep[4];
      Real r2 = 0.0;
      for (int k=1; k<=3; ++k) {
        dsep[k] = SinkWrapDelta(y[3*i+k] - y[3*j+k], ctx.len[k-1], ctx.periodic[k-1]);
        r2 += SQR(dsep[k]);
      }
      Real r = std::sqrt(r2);
      r2 += SQR(ctx.soften);
      // Guard the SOFTENED DENOMINATOR, not just r.  Repairing r alone left
      // r2*r = 0*kTinyNr = 0 whenever soften == 0 and two sinks coincide (reachable from
      // two identical initN rows, or from CreateSinks filling two cells that coincide
      // after a merge): every component of acc became NaN for the WHOLE list, the
      // Bulirsch-Stoer error test then never fell below 1, and the run died at hmin with
      // a "close encounter unresolved by the softening" message that named the wrong
      // cause.  With any softening at all this is inert.
      if (r == 0.0) { r = kTinyNr; }
      const Real den = std::max(r2*r, kTinyNr);
      for (int k=1; k<=3; ++k) {
        const Real f = ctx.gconst*dsep[k]/den;
        acc[3*i+k] -= f*mass[j+1];
        acc[3*j+k] += f*mass[i+1];
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Stoerm
//! \brief Stoermer's rule for y'' = f(y) over one modified-midpoint sequence of nstep
//! substeps (ORION2 ParticleInt.cpp:261-291).

void Stoerm(const Real *y, const Real *d2y, int nv, Real htot, int nstep, Real *yout,
            const Real *mass, const NBodyCtx &ctx, BSWork *w) {
  Real *ytemp = w->ytemp.data();
  const Real h = htot/nstep;
  const Real halfh = 0.5*h;
  const int neqns = nv/2;
  for (int i=1; i<=neqns; ++i) {
    const int n = neqns + i;
    ytemp[n] = h*(y[n] + halfh*d2y[i]);
    ytemp[i] = y[i] + ytemp[n];
  }
  Derivs(ytemp, yout, mass, ctx, nv);
  const Real h2 = h*h;
  for (int nn=2; nn<=nstep; ++nn) {
    for (int i=1; i<=neqns; ++i) {
      const int n = neqns + i;
      ytemp[n] += h2*yout[i];
      ytemp[i] += ytemp[n];
    }
    Derivs(ytemp, yout, mass, ctx, nv);
  }
  for (int i=1; i<=neqns; ++i) {
    const int n = neqns + i;
    yout[n] = ytemp[n]/h + halfh*yout[i];
    yout[i] = ytemp[i];
  }
}

//----------------------------------------------------------------------------------------
//! \fn PZExtr
//! \brief Richardson extrapolation of the modified-midpoint sequence to zero step size.

void PZExtr(int iest, Real xest, const Real *yest, Real *yz, Real *dy, int nv,
            BSWork *w) {
  Real *c = w->ctmp.data();
  w->xtab[iest] = xest;
  for (int j=1; j<=nv; ++j) { dy[j] = yz[j] = yest[j]; }
  if (iest == 1) {
    for (int j=1; j<=nv; ++j) { w->d(j, 1) = yest[j]; }
  } else {
    for (int j=1; j<=nv; ++j) { c[j] = yest[j]; }
    for (int k1=1; k1<iest; ++k1) {
      Real delta = 1.0/(w->xtab[iest-k1] - xest);
      const Real f1 = xest*delta;
      const Real f2 = w->xtab[iest-k1]*delta;
      for (int j=1; j<=nv; ++j) {
        const Real q = w->d(j, k1);
        w->d(j, k1) = dy[j];
        delta = c[j] - q;
        dy[j] = f1*delta;
        c[j] = f2*delta;
        yz[j] += dy[j];
      }
    }
    for (int j=1; j<=nv; ++j) { w->d(j, iest) = dy[j]; }
  }
}

//----------------------------------------------------------------------------------------
//! \fn BSStep
//! \brief one adaptive Bulirsch-Stoer step.  Returns false on step-size underflow.

bool BSStep(Real *y, const Real *dydx, int nv, Real *xx, Real htry, Real eps,
            const Real *yscal, Real *hdid, Real *hnext, const Real *mass,
            const NBodyCtx &ctx, BSWork *w) {
  Real *err = w->err.data();
  Real *yerr = w->yerr.data();
  Real *ysav = w->ysav.data();
  Real *yseq = w->yseq.data();

  if (eps != w->epsold) {
    *hnext = w->xnew = -1.0e29;
    const Real eps1 = kSafe1*eps;
    w->a[1] = w->nseq[1] + 1;
    for (int k=1; k<=kKmaxx; ++k) { w->a[k+1] = w->a[k] + w->nseq[k+1]; }
    for (int iq=2; iq<=kKmaxx; ++iq) {
      for (int k=1; k<iq; ++k) {
        w->alf[k][iq] = std::pow(eps1, (w->a[k+1] - w->a[iq+1])/
                                       ((w->a[iq+1] - w->a[1] + 1.0)*(2*k + 1)));
      }
    }
    w->epsold = eps;
    for (w->kopt=2; w->kopt<kKmaxx; ++(w->kopt)) {
      if (w->a[w->kopt+1] > w->a[w->kopt]*w->alf[w->kopt-1][w->kopt]) { break; }
    }
    w->kmax = w->kopt;
  }

  Real h = htry;
  for (int i=1; i<=nv; ++i) { ysav[i] = y[i]; }
  if (*xx != w->xnew || h != *hnext) {
    w->first = 1;
    w->kopt = w->kmax;
  }

  int k = 1, km = 1, reduct = 0, exitflag = 0;
  Real errmax = kTinyNr, red = 1.0, scale = 1.0;
  for (;;) {
    for (k=1; k<=w->kmax; ++k) {
      w->xnew = (*xx) + h;
      if (w->xnew == (*xx)) { return false; }
      Stoerm(ysav, dydx, nv, h, w->nseq[k], yseq, mass, ctx, w);
      const Real xest = SQR(h/w->nseq[k]);
      PZExtr(k, xest, yseq, y, yerr, nv, w);
      if (k != 1) {
        errmax = kTinyNr;
        for (int i=1; i<=nv; ++i) { errmax = std::fmax(errmax, std::fabs(yerr[i]/
                                                                         yscal[i])); }
        errmax /= eps;
        km = k - 1;
        err[km] = std::pow(errmax/kSafe1, 1.0/(2*km + 1));
      }
      if (k != 1 && (k >= w->kopt-1 || w->first)) {
        if (errmax < 1.0) { exitflag = 1; break; }
        if (k == w->kmax || k == w->kopt+1) {
          red = kSafe2/err[km];
          break;
        } else if (k == w->kopt && w->alf[w->kopt-1][w->kopt] < err[km]) {
          red = 1.0/err[km];
          break;
        } else if (w->kopt == w->kmax && w->alf[km][w->kmax-1] < err[km]) {
          red = w->alf[km][w->kmax-1]*kSafe2/err[km];
          break;
        } else if (w->alf[km][w->kopt] < err[km]) {
          red = w->alf[km][w->kopt-1]/err[km];
          break;
        }
      }
    }
    if (exitflag) { break; }
    red = std::fmin(red, kRedMin);
    red = std::fmax(red, kRedMax);
    h *= red;
    reduct = 1;
  }

  *xx = w->xnew;
  *hdid = h;
  w->first = 0;
  Real wrkmin = 1.0e35;
  for (int kk=1; kk<=km; ++kk) {
    const Real fact = std::fmax(err[kk], kScalMx);
    const Real work = fact*w->a[kk+1];
    if (work < wrkmin) {
      scale = fact;
      wrkmin = work;
      w->kopt = kk + 1;
    }
  }
  *hnext = h/scale;
  if (w->kopt >= k && w->kopt != w->kmax && !reduct) {
    const Real fact = std::fmax(scale/w->alf[w->kopt-1][w->kopt], kScalMx);
    if (w->a[w->kopt+1]*fact <= wrkmin) {
      *hnext = h/fact;
      ++(w->kopt);
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn ParticleIntegrate
//! \brief NR odeint driver specialized to the second-order system.  Returns false with a
//! reason on failure; the caller turns that into a fatal error.

bool ParticleIntegrate(Real *ystart, int nvar, Real x1, Real x2, Real eps, Real h1,
                       Real hmin, const Real *mass, const NBodyCtx &ctx,
                       std::string *reason) {
  BSWork w(nvar);
  std::vector<Real> yv(nvar + 1, 0.0), dydxv(nvar + 1, 0.0), yscalv(nvar + 1, 0.0);
  Real *y = yv.data();
  Real *dydx = dydxv.data();
  Real *yscal = yscalv.data();

  const int npart = nvar/6;
  Real x = x1;
  Real h = (x2 >= x1) ? std::fabs(h1) : -std::fabs(h1);
  Real hnext = h, hdid = 0.0;
  for (int i=1; i<=nvar; ++i) { y[i] = ystart[i]; }

  for (int nstp=1; nstp<=kMaxStp; ++nstp) {
    Derivs(y, dydx, mass, ctx, nvar);

    // Error scaling on the minimum pair separation/relative speed rather than on |y|,
    // which would break for positions or velocities near zero.
    Real distmin = 1.0e50, velmin = 1.0e50;
    for (int i=0; i<npart; ++i) {
      for (int j=i+1; j<npart; ++j) {
        Real dist = 0.0, vel = 0.0;
        for (int k=1; k<=3; ++k) {
          dist += SQR(SinkWrapDelta(y[3*i+k] - y[3*j+k], ctx.len[k-1],
                                    ctx.periodic[k-1]));
          vel  += SQR(y[3*npart+3*i+k] - y[3*npart+3*j+k]);
        }
        distmin = std::fmin(distmin, std::sqrt(dist));
        velmin  = std::fmin(velmin, std::sqrt(vel));
      }
    }
    const Real velfloor = std::sqrt(eps)*distmin/(x2 - x1);
    if (velmin < velfloor) { velmin = velfloor; }
    for (int i=1; i<=nvar/2; ++i) { yscal[i] = distmin + kTinyNr; }
    for (int i=nvar/2+1; i<=nvar; ++i) { yscal[i] = velmin + kTinyNr; }

    if ((x + h - x2)*(x + h - x1) > 0.0) { h = x2 - x; }
    if (!BSStep(y, dydx, nvar, &x, h, eps, yscal, &hdid, &hnext, mass, ctx, &w)) {
      *reason = "step size underflow in the Bulirsch-Stoer step";
      return false;
    }
    if ((x - x2)*(x2 - x1) >= 0.0) {
      for (int i=1; i<=nvar; ++i) { ystart[i] = y[i]; }
      return true;
    }
    if (std::fabs(hnext) <= hmin) {
      *reason = "step size fell below hmin";
      return false;
    }
    h = hnext;
  }
  *reason = "too many steps";
  return false;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn SinkParticles::AdvanceNBody
//! \brief advance every sink over dt: pure drift for one sink, adaptive Bulirsch-Stoer
//! for two or more (ORION2 SinkParticleList::advance).

void SinkParticles::AdvanceNBody(Real dt, const SinkMeshGeom &g) {
  const int np = nsinks;
  if (np == 0 || dt <= 0.0) { return; }

  if (np == 1) {
    if (sinks[0].m > 0.0) {
      for (int d=0; d<3; ++d) { sinks[0].pos[d] += dt*sinks[0].mom[d]/sinks[0].m; }
    }
    WrapPosition(&sinks[0], g);
    return;
  }

  NBodyCtx ctx;
  ctx.gconst = newton_g;
  // The PUBLISHED softening (sink_gm_pos column 4), not soften*g.dxf_max recomputed from
  // this cycle's finest level: on an AMR level change the two differ by a factor of two,
  // and taking the fresh one here would step the sink-sink orbit with a softening the
  // gas coupling of the same cycle never saw.  Both are updated together, once, at the
  // end of SinkStep.
  ctx.soften = soften_len;
  for (int d=0; d<3; ++d) {
    ctx.len[d] = g.len[d];
    ctx.periodic[d] = g.periodic[d];
  }

  const int nvar = 6*np;
  std::vector<Real> y(nvar + 1, 0.0);
  std::vector<Real> mass(np + 1, 0.0);
  for (int i=0; i<np; ++i) {
    mass[i+1] = sinks[i].m;
    for (int k=1; k<=3; ++k) {
      y[3*i+k] = sinks[i].pos[k-1];
      y[3*np+3*i+k] = (sinks[i].m > 0.0) ? sinks[i].mom[k-1]/sinks[i].m : 0.0;
    }
  }

  std::string reason;
  if (!ParticleIntegrate(y.data(), nvar, 0.0, dt, bs_tol, dt, 1.0e-6*dt, mass.data(),
                         ctx, &reason)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "sink-particle N-body integration failed: " << reason << std::endl
              << "  nsinks = " << np << ", dt = " << dt
              << ", bs_tol = " << bs_tol << ", soften = " << ctx.soften << std::endl
              << "  a close encounter unresolved by the softening usually means the "
              << "timestep or <sink_particles>/soften is too small" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int i=0; i<np; ++i) {
    for (int k=1; k<=3; ++k) {
      sinks[i].pos[k-1] = y[3*i+k];
      sinks[i].mom[k-1] = sinks[i].m*y[3*np+3*i+k];
    }
    WrapPosition(&sinks[i], g);
  }
}

}  // namespace sinkparticles
