// =============================================================================
// Verification examples for time integrators of the Navier-Stokes equations:
//   u(t,x,y) = phi(t) (y^2, x^2),   p(t,x,y) = cos(pi t) (x + y - 1)
// on the unit square, f = d_t u - nu Laplace(u) + (u.grad)u + grad p,
//   TNSE_POLY_TIME_DEGREE 2:  phi(t) = 1 + t + t^2   (example 30)
//   TNSE_POLY_TIME_DEGREE 1:  phi(t) = 1 + t         (example 31)
// u lies in P2 and p in P1 for every t, so with a consistent stabilization the
// semidiscrete solution is exact. The trapezoidal rule is exact for solutions
// quadratic in time (example 30), BDF2 with the extrapolation 2u^n - u^{n-1}
// is exact for solutions linear in time (example 31): the fully discrete
// solutions must agree with (u, p) up to round-off and the Picard tolerance,
// for any sequence of step sizes and of meshes (averaging steps excepted).
// Time-dependent Dirichlet data. Coefficient layout as in TNSE_AAJ_smooth.h.
// Define TNSE_POLY_TIME_DEGREE before including this file.
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

void ExampleFile()
{
  Output::print<1>("Example: TNSE_polynomial_space_time.h, phi(t) = 1 + t",
                   TNSE_POLY_TIME_DEGREE == 2 ? " + t^2" : "",
                   ", u = phi(t)(y^2, x^2), p = cos(pi t)(x+y-1)");
}

static inline double phi(double t)
{ return 1. + t + (TNSE_POLY_TIME_DEGREE == 2 ? t*t : 0.); }
static inline double phi_t(double t)
{ return 1. + (TNSE_POLY_TIME_DEGREE == 2 ? 2.*t : 0.); }
static inline double psi(double t) { return std::cos(M_PI*t); }

void ExactU1(double, double y, double *values)
{
  const double g = phi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*y*y;  values[1] = 0.;  values[2] = 2.*g*y;  values[3] = 2.*g;
}

void ExactU2(double x, double, double *values)
{
  const double g = phi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*x*x;  values[1] = 2.*g*x;  values[2] = 0.;  values[3] = 2.*g;
}

void ExactP(double x, double y, double *values)
{
  const double g = psi(TDatabase::TimeDB->CURRENTTIME);
  values[0] = g*(x + y - 1.);  values[1] = g;  values[2] = g;  values[3] = 0.;
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

// Stokes projection data for the BDF2-IMEX program (see TNSE_polynomial_in_space.h)
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
    for(int k = 3; k < 7; ++k)
      coeffs[i][k] = 0.;
  }
}
