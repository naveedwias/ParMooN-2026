// =============================================================================
// Verification example for time-dependent Navier-Stokes discretizations:
//   u(t,x,y) = phi(t) (y^2, x^2),   p(t,x,y) = psi(t) (x + y - 1),
//   phi(t) = 1 + sin(pi t)/2,  psi(t) = cos(pi t),
// on the unit square, f = d_t u - nu Laplace(u) + (u.grad)u + grad p.
// u is divergence-free and quadratic, p is linear with zero mean, so (u, p)
// lies in every pair containing P2/P1 (Taylor-Hood, P2/P2, ...). For a
// consistent stabilization (SUPG/PSPG/grad-div, residual-based) the exact
// solution solves the semidiscrete problem, the semidiscrete error is zero
// and the fully discrete error is the error of the time integrator only.
// The Dirichlet data u = phi(t)(y^2, x^2) depend on time, which tests the
// treatment of time-dependent boundary data as well.
// Coefficient layout as in TNSE_AAJ_smooth.h (coeff[4..5] = b = u).
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

void ExampleFile()
{
  Output::print<1>("Example: TNSE_polynomial_in_space.h (u = phi(t)(y^2, x^2), "
                   "p = psi(t)(x+y-1), time-dependent Dirichlet data)");
}

static inline double phi(double t) { return 1. + 0.5*std::sin(M_PI*t); }
static inline double phi_t(double t) { return 0.5*M_PI*std::cos(M_PI*t); }
static inline double psi(double t) { return std::cos(M_PI*t); }

void ExactU1(double, double y, double *values)
{
  const double g = phi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*y*y;
  values[1] = 0.;
  values[2] = 2.*g*y;
  values[3] = 2.*g;
}

void ExactU2(double x, double, double *values)
{
  const double g = phi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*x*x;
  values[1] = 2.*g*x;
  values[2] = 0.;
  values[3] = 2.*g;
}

void ExactP(double x, double y, double *values)
{
  const double g = psi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*(x + y - 1.);
  values[1] = g;
  values[2] = g;
  values[3] = 0.;
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

static inline void boundary_point(int BdComp, double Param, double& x, double& y)
{
  switch(BdComp)
  {
    case 0: x = Param;    y = 0.;       break;
    case 1: x = 1.;       y = Param;    break;
    case 2: x = 1.-Param; y = 1.;       break;
    case 3: x = 0.;       y = 1.-Param; break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
}

void U1BoundValue(int BdComp, double Param, double &value)
{
  double x, y, v[4];
  boundary_point(BdComp, Param, x, y);
  ExactU1(x, y, v);
  value = v[0];
}

void U2BoundValue(int BdComp, double Param, double &value)
{
  double x, y, v[4];
  boundary_point(BdComp, Param, x, y);
  ExactU2(x, y, v);
  value = v[0];
}

void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  const double t = TDatabase::TimeDB->CURRENTTIME;
  const double g = phi(t), gt = phi_t(t), gp = psi(t);
  for(int i = 0; i < n_points; i++)
  {
    const double x = X[i], y = Y[i];
    coeffs[i][0] = nu;
    coeffs[i][1] = gt*y*y - 2.*nu*g + 2.*g*g*x*x*y + gp;
    coeffs[i][2] = gt*x*x - 2.*nu*g + 2.*g*g*x*y*y + gp;
    coeffs[i][3] = 0.;
    coeffs[i][4] = g*y*y;
    coeffs[i][5] = g*x*x;
    coeffs[i][6] = 0.;
  }
}

// right-hand side of the Stokes projection for the BDF2-IMEX driver (same
// convention as BDF2IMEX_smooth.h): assembled with viscosity 1,
// f_s = -Laplace(u) + grad p / nu with 1/nu = TDatabase::ParamDB->P1
void StokesProjectionCoeffs(int n_points, const double *X, const double *Y,
                            const double *const*, double **coeffs)
{
  const double inv_nu = TDatabase::ParamDB->P1;
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    coeffs[i][0] = 1.;
    coeffs[i][1] = -u1[3] + inv_nu*p[1];
    coeffs[i][2] = -u2[3] + inv_nu*p[2];
    coeffs[i][3] = 0.;
    coeffs[i][4] = 0.;
    coeffs[i][5] = 0.;
    coeffs[i][6] = 0.;
  }
}
