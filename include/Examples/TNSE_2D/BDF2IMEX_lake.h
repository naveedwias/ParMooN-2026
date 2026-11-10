// =============================================================================
// Lake-at-rest test for the BDF2-IMEX / CIP / Scott-Vogelius scheme: zero
// velocity and a non-trivial pressure p = A g(t) cos(pi x) cos(pi y),
// g = 1 + sin(2 pi t)/2, i.e. f = grad p. A pressure-robust method keeps the
// discrete velocity at round-off level independently of A; a method which is
// not pressure-robust produces a spurious velocity of size ~ (A/nu) inf |p - q_h|.
// A = TDatabase::ParamDB->P0 ('pressure_amplitude').
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
  Output::print<1>("Example: BDF2IMEX_lake.h (lake at rest, u = 0, pressure amplitude A = ",
                   amplitude(), ")");
}

static inline double time_factor(double t) { return 1. + 0.5*std::sin(2.*M_PI*t); }

void ExactU1(double, double, double *values)
{
  values[0] = values[1] = values[2] = values[3] = 0.;
}

void ExactU2(double, double, double *values)
{
  values[0] = values[1] = values[2] = values[3] = 0.;
}

void ExactP(double x, double y, double *values)
{
  const double g = time_factor(TDatabase::TimeDB->CURRENTTIME)*amplitude();
  values[0] = g*std::cos(M_PI*x)*std::cos(M_PI*y);
  values[1] = -g*M_PI*std::sin(M_PI*x)*std::cos(M_PI*y);
  values[2] = -g*M_PI*std::cos(M_PI*x)*std::sin(M_PI*y);
  values[3] = -2.*M_PI*M_PI*values[0];
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

void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  double p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactP(X[i], Y[i], p);
    coeffs[i][0] = nu;
    coeffs[i][1] = p[1];
    coeffs[i][2] = p[2];
    coeffs[i][3] = 0.;
    coeffs[i][4] = 1. - nu; // grad-div parameter scaling, see BDF2IMEX_smooth.h
    coeffs[i][5] = 0.;
    coeffs[i][6] = 0.;
  }
}

void StokesProjectionCoeffs(int n_points, const double *X, const double *Y,
                            const double *const*, double **coeffs)
{
  const double inv_nu = TDatabase::ParamDB->P1;
  double p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactP(X[i], Y[i], p);
    coeffs[i][0] = 1.;
    coeffs[i][1] = inv_nu*p[1];
    coeffs[i][2] = inv_nu*p[2];
    coeffs[i][3] = 0.;
    coeffs[i][4] = 0.;
    coeffs[i][5] = 0.;
    coeffs[i][6] = 0.;
  }
}
