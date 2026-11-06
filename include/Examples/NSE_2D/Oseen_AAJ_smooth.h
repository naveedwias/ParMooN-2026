// =============================================================================
// Example 4.1 of
//   M. Afzal, N. Ahmed, V. John, A robust a posteriori error estimator for the
//   Oseen problem, Comput. Math. Appl. 223 (2026) 467-486.
//
// Oseen problem (1) on the unit square with smooth solution (35)-(36):
//   u = ( 1000 x^2 (1-x)^4 y^2 (1-y)(3-5y),  -2000 x (1-x)^3 (1-3x) y^3 (1-y)^2 )
//   p = pi^2 ( x y^3 cos(2 pi x^2 y) - x^2 y sin(2 pi x y) ) + 1/8   (mean zero)
// convection field b = u, reaction sigma = 1, Dirichlet b.c. on the whole boundary.
// For FLOW_PROBLEM_TYPE == 5 (Navier-Stokes, Section 5.1) the same solution is
// used with the convective term (u.grad)u and sigma = 0.
//
// Coefficient layout (as used by the Oseen/SPG assembling routines):
//   coeff[0] = nu, coeff[1..2] = f, coeff[3] = g (divergence rhs, = 0),
//   coeff[4..5] = b, coeff[6] = sigma
// Exact solution layout: values[0] = u, [1] = u_x, [2] = u_y, [3] = Laplace(u)
// (derivatives generated symbolically with sympy)
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

void ExampleFile()
{
  Output::print<1>("Example: Oseen_AAJ_smooth.h  (Afzal-Ahmed-John 2026, Example 4.1)");
}

// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double y, double *values)
{
  values[0] = 1000*pow(x, 2)*pow(y, 2)*pow(1 - x, 4)*(1 - y)*(3 - 5*y);
  values[1] = -4000*pow(x, 2)*pow(y, 2)*pow(1 - x, 3)*(1 - y)*(3 - 5*y) + 2000*x*pow(y, 2)*pow(1 - x, 4)*(1 - y)*(3 - 5*y);
  values[2] = -5000*pow(x, 2)*pow(y, 2)*pow(1 - x, 4)*(1 - y) - 1000*pow(x, 2)*pow(y, 2)*pow(1 - x, 4)*(3 - 5*y) + 2000*pow(x, 2)*y*pow(1 - x, 4)*(1 - y)*(3 - 5*y);
  values[3] = 10000*pow(x, 2)*pow(y, 2)*pow(1 - x, 4) + 12000*pow(x, 2)*pow(y, 2)*pow(1 - x, 2)*(1 - y)*(3 - 5*y) - 20000*pow(x, 2)*y*pow(1 - x, 4)*(1 - y) - 4000*pow(x, 2)*y*pow(1 - x, 4)*(3 - 5*y) + 2000*pow(x, 2)*pow(1 - x, 4)*(1 - y)*(3 - 5*y) - 16000*x*pow(y, 2)*pow(1 - x, 3)*(1 - y)*(3 - 5*y) + 2000*pow(y, 2)*pow(1 - x, 4)*(1 - y)*(3 - 5*y);
}

void ExactU2(double x, double y, double *values)
{
  values[0] = -2000*x*pow(y, 3)*(1 - 3*x)*pow(1 - x, 3)*pow(1 - y, 2);
  values[1] = 6000*x*pow(y, 3)*(1 - 3*x)*pow(1 - x, 2)*pow(1 - y, 2) + 6000*x*pow(y, 3)*pow(1 - x, 3)*pow(1 - y, 2) - 2000*pow(y, 3)*(1 - 3*x)*pow(1 - x, 3)*pow(1 - y, 2);
  values[2] = -2000*x*pow(y, 3)*(1 - 3*x)*pow(1 - x, 3)*(2*y - 2) - 6000*x*pow(y, 2)*(1 - 3*x)*pow(1 - x, 3)*pow(1 - y, 2);
  values[3] = -4000*x*pow(y, 3)*(1 - 3*x)*pow(1 - x, 3) + 6000*x*pow(y, 3)*(1 - 3*x)*pow(1 - y, 2)*(2*x - 2) - 36000*x*pow(y, 3)*pow(1 - x, 2)*pow(1 - y, 2) - 12000*x*pow(y, 2)*(1 - 3*x)*pow(1 - x, 3)*(2*y - 2) - 12000*x*y*(1 - 3*x)*pow(1 - x, 3)*pow(1 - y, 2) + 12000*pow(y, 3)*(1 - 3*x)*pow(1 - x, 2)*pow(1 - y, 2) + 12000*pow(y, 3)*pow(1 - x, 3)*pow(1 - y, 2);
}

void ExactP(double x, double y, double *values)
{
  values[0] = pow(M_PI, 2)*(-pow(x, 2)*y*sin(2*M_PI*x*y) + x*pow(y, 3)*cos(2*M_PI*pow(x, 2)*y)) + 1.0/8.0;
  values[1] = pow(M_PI, 2)*(-4*M_PI*pow(x, 2)*pow(y, 4)*sin(2*M_PI*pow(x, 2)*y) - 2*M_PI*pow(x, 2)*pow(y, 2)*cos(2*M_PI*x*y) - 2*x*y*sin(2*M_PI*x*y) + pow(y, 3)*cos(2*M_PI*pow(x, 2)*y));
  values[2] = pow(M_PI, 2)*(-2*M_PI*pow(x, 3)*pow(y, 3)*sin(2*M_PI*pow(x, 2)*y) - 2*M_PI*pow(x, 3)*y*cos(2*M_PI*x*y) - pow(x, 2)*sin(2*M_PI*x*y) + 3*x*pow(y, 2)*cos(2*M_PI*pow(x, 2)*y));
  values[3] = pow(M_PI, 2)*(-16*pow(M_PI, 2)*pow(x, 3)*pow(y, 5)*cos(2*M_PI*pow(x, 2)*y) + 4*pow(M_PI, 2)*pow(x, 2)*pow(y, 3)*sin(2*M_PI*x*y) - 12*M_PI*x*pow(y, 4)*sin(2*M_PI*pow(x, 2)*y) - 8*M_PI*x*pow(y, 2)*cos(2*M_PI*x*y) - 2*y*sin(2*M_PI*x*y)) + pow(M_PI, 2)*(-4*pow(M_PI, 2)*pow(x, 5)*pow(y, 3)*cos(2*M_PI*pow(x, 2)*y) + 4*pow(M_PI, 2)*pow(x, 4)*y*sin(2*M_PI*x*y) - 12*M_PI*pow(x, 3)*pow(y, 2)*sin(2*M_PI*pow(x, 2)*y) - 4*M_PI*pow(x, 3)*cos(2*M_PI*x*y) + 6*x*y*cos(2*M_PI*pow(x, 2)*y));
}

// ========================================================================
// boundary conditions
// ========================================================================
void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void U1BoundValue(int BdComp, double Param, double &value)
{
  double x = 0., y = 0.;
  switch(BdComp)
  {
    case 0: x = Param;    y = 0.;       break;
    case 1: x = 1.;       y = Param;    break;
    case 2: x = 1.-Param; y = 1.;       break;
    case 3: x = 0.;       y = 1.-Param; break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
  double v[4];
  ExactU1(x, y, v);
  value = v[0];
}

void U2BoundValue(int BdComp, double Param, double &value)
{
  double x = 0., y = 0.;
  switch(BdComp)
  {
    case 0: x = Param;    y = 0.;       break;
    case 1: x = 1.;       y = Param;    break;
    case 2: x = 1.-Param; y = 1.;       break;
    case 3: x = 0.;       y = 1.-Param; break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
  double v[4];
  ExactU2(x, y, v);
  value = v[0];
}

// ========================================================================
// coefficients for Oseen (FLOW_PROBLEM_TYPE 3 / linear) and Navier-Stokes (5)
// ========================================================================
void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  const bool navier_stokes = (TDatabase::ParamDB->FLOW_PROBLEM_TYPE == 5);
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    // convection field b = u (both for Oseen and Navier-Stokes), sigma = 1 (Oseen)
    const double b1 = u1[0], b2 = u2[0];
    const double sigma = navier_stokes ? 0. : 1.;
    coeffs[i][0] = nu;
    coeffs[i][1] = -nu*u1[3] + b1*u1[1] + b2*u1[2] + sigma*u1[0] + p[1]; // f1
    coeffs[i][2] = -nu*u2[3] + b1*u2[1] + b2*u2[2] + sigma*u2[0] + p[2]; // f2
    coeffs[i][3] = u1[1] + u2[2]; // g (divergence, = 0)
    coeffs[i][4] = b1;
    coeffs[i][5] = b2;
    coeffs[i][6] = sigma;
  }
}
