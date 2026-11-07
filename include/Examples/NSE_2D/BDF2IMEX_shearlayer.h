// =============================================================================
// Example 2 of "Pressure-robust error estimates for a stabilized BDF2-IMEX
// Navier-Stokes scheme": doubly periodic shear layer (Bell, Colella, Glaz
// 1989; Brown, Minion 1995) on the unit square, zero forcing,
//   u_0 = ( tanh(rho (1/4 - |y - 1/2|)),  delta sin(2 pi (x + 1/4)) ),
//   rho = 80, delta = 0.05, nu = 1e-4 (from reynolds_number).
// Periodic boundary conditions in both directions (BoundCondition returns
// PERIODIC; the domain needs 'periodic_translations: (1.,0.,0.,1.)' or the
// default normal matching). No exact solution: Exact returns the initial
// velocity and zero pressure (used only for the output routines).
// Coefficient layout: coeff[0] = nu, coeff[1..2] = 0, coeff[3] = 0,
// coeff[4] = 1 - nu (grad-div scaling), coeff[5..6] = 0.
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

void ExampleFile()
{
  Output::print<1>("Example: BDF2IMEX_shearlayer.h (doubly periodic shear layer, "
                   "rho = 80, delta = 0.05)");
}

void ExactU1(double, double y, double *values)
{
  const double rho = 80.;
  double a = rho*(0.25 - std::abs(y - 0.5));
  values[0] = std::tanh(a);
  values[1] = 0.;
  values[2] = -rho*((y > 0.5) ? 1. : -1.)/(std::cosh(a)*std::cosh(a));
  values[3] = 0.;
}

void ExactU2(double x, double, double *values)
{
  const double delta = 0.05;
  values[0] = delta*std::sin(2.*M_PI*(x + 0.25));
  values[1] = delta*2.*M_PI*std::cos(2.*M_PI*(x + 0.25));
  values[2] = 0.;
  values[3] = 0.;
}

void ExactP(double, double, double *values)
{
  values[0] = values[1] = values[2] = values[3] = 0.;
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = PERIODIC;
}

void U1BoundValue(int, double, double &value)
{
  value = 0.;
}

void U2BoundValue(int, double, double &value)
{
  value = 0.;
}

void LinCoeffs(int n_points, const double *, const double *,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = nu;
    coeffs[i][1] = 0.;
    coeffs[i][2] = 0.;
    coeffs[i][3] = 0.;
    coeffs[i][4] = 1. - nu;
    coeffs[i][5] = 0.;
    coeffs[i][6] = 0.;
  }
}

void StokesProjectionCoeffs(int n_points, const double *, const double *,
                            const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
    for(int k = 0; k < 7; ++k)
      coeffs[i][k] = (k == 0) ? 1. : 0.;
}
