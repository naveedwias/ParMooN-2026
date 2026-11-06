// =============================================================================
// Example 1 of "Pressure-robust error estimates for a stabilized BDF2-IMEX
// Navier-Stokes scheme": smooth solution on the unit square,
//   u = curl(psi), psi = g(t)/pi sin^2(pi x) sin^2(pi y), g = 1 + sin(2 pi t)/2,
//   p = A g(t) cos(pi x) cos(pi y),   f = d_t u - nu Laplace(u) + (u.grad)u + grad p.
// The amplitude A is TDatabase::ParamDB->P0 (default 1 if P0 == 0 is not set).
// Homogeneous Dirichlet data; the time is TDatabase::TimeDB->CURRENTTIME.
// Coefficient layout: coeff[0] = nu, coeff[1..2] = f, coeff[3] = 0 (divergence),
// coeff[4..5] = u (convection field), coeff[6] = 0.
// StokesProjectionCoeffs gives f_s = -nu Laplace(u) + grad p for the Stokes
// projection of the exact velocity (same layout).
// (derivatives generated with sympy)
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

static inline double amplitude()
{
  return TDatabase::ParamDB->P0;
}

void ExampleFile()
{
  Output::print<1>("Example: BDF2IMEX_smooth.h (smooth solution, pressure amplitude A = ",
                   amplitude(), ")");
}

void ExactU1(double x, double y, double *values)
{
  const double t = TDatabase::TimeDB->CURRENTTIME;
  values[0] = 2*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*x), 2)*sin(M_PI*y)*cos(M_PI*y);
  values[1] = 4*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*x)*sin(M_PI*y)*cos(M_PI*x)*cos(M_PI*y);
  values[2] = -2*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*x), 2)*pow(sin(M_PI*y), 2) + 2*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*x), 2)*pow(cos(M_PI*y), 2);
  values[3] = -4*pow(M_PI, 2)*((1.0/2.0)*sin(2*M_PI*t) + 1)*(pow(sin(M_PI*x), 2) - pow(cos(M_PI*x), 2))*sin(M_PI*y)*cos(M_PI*y) - 8*pow(M_PI, 2)*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*x), 2)*sin(M_PI*y)*cos(M_PI*y);
}

void ExactU2(double x, double y, double *values)
{
  const double t = TDatabase::TimeDB->CURRENTTIME;
  values[0] = -2*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*x)*pow(sin(M_PI*y), 2)*cos(M_PI*x);
  values[1] = 2*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*x), 2)*pow(sin(M_PI*y), 2) - 2*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*pow(sin(M_PI*y), 2)*pow(cos(M_PI*x), 2);
  values[2] = -4*M_PI*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*x)*sin(M_PI*y)*cos(M_PI*x)*cos(M_PI*y);
  values[3] = 4*pow(M_PI, 2)*((1.0/2.0)*sin(2*M_PI*t) + 1)*(pow(sin(M_PI*y), 2) - pow(cos(M_PI*y), 2))*sin(M_PI*x)*cos(M_PI*x) + 8*pow(M_PI, 2)*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*x)*pow(sin(M_PI*y), 2)*cos(M_PI*x);
}

void ExactP(double x, double y, double *values)
{
  const double t = TDatabase::TimeDB->CURRENTTIME;
  const double A = amplitude();
  values[0] = A*((1.0/2.0)*sin(2*M_PI*t) + 1)*cos(M_PI*x)*cos(M_PI*y);
  values[1] = -M_PI*A*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*x)*cos(M_PI*y);
  values[2] = -M_PI*A*((1.0/2.0)*sin(2*M_PI*t) + 1)*sin(M_PI*y)*cos(M_PI*x);
  values[3] = -2*pow(M_PI, 2)*A*((1.0/2.0)*sin(2*M_PI*t) + 1)*cos(M_PI*x)*cos(M_PI*y);
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void U1BoundValue(int, double, double &value)
{
  value = 0.;
}

void U2BoundValue(int, double, double &value)
{
  value = 0.;
}

static inline void time_derivative(double x, double y, double t, double& dt1, double& dt2)
{
  dt1 = 2*M_PI*pow(sin(M_PI*x), 2)*sin(M_PI*y)*cos(2*M_PI*t)*cos(M_PI*y);
  dt2 = -2*M_PI*sin(M_PI*x)*pow(sin(M_PI*y), 2)*cos(2*M_PI*t)*cos(M_PI*x);
}

void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  const double t = TDatabase::TimeDB->CURRENTTIME;
  double u1[4], u2[4], p[4], dt1, dt2;
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    time_derivative(X[i], Y[i], t, dt1, dt2);
    coeffs[i][0] = nu;
    coeffs[i][1] = dt1 - nu*u1[3] + u1[0]*u1[1] + u2[0]*u1[2] + p[1];
    coeffs[i][2] = dt2 - nu*u2[3] + u1[0]*u2[1] + u2[0]*u2[2] + p[2];
    coeffs[i][3] = 0.;
    coeffs[i][4] = u1[0];
    coeffs[i][5] = u2[0];
    coeffs[i][6] = 0.;
  }
}

// right-hand side of the Stokes projection: f_s = -nu Laplace(u) + grad p
void StokesProjectionCoeffs(int n_points, const double *X, const double *Y,
                            const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    coeffs[i][0] = nu;
    coeffs[i][1] = -nu*u1[3] + p[1];
    coeffs[i][2] = -nu*u2[3] + p[2];
    coeffs[i][3] = 0.;
    coeffs[i][4] = 0.;
    coeffs[i][5] = 0.;
    coeffs[i][6] = 0.;
  }
}
