// =============================================================================
// Example 4.2 of
//   M. Afzal, N. Ahmed, V. John, A robust a posteriori error estimator for the
//   Oseen problem, Comput. Math. Appl. 223 (2026) 467-486,
// taken from Khan & Kanschat, SIAM J. Numer. Anal. 58 (2020) 492-518.
//
// Oseen problem (1) on the unit square, solution with boundary layers at x=1, y=1:
//   u = curl(phi) = (phi_y, -phi_x),
//   phi = x^2 y^2 (1 - exp(lam (x-1)))^2 (1 - exp(lam (y-1)))^2,  lam = 0.5/sqrt(nu)
//   p = exp(x+y) - p_m,   p_m = (e-1)^2  (so that p has zero mean)
// convection field b = (1,1), reaction sigma = 0, Dirichlet b.c. on the whole
// boundary. For FLOW_PROBLEM_TYPE == 5 (Navier-Stokes, Section 5.2) the same
// solution is used with the convective term (u.grad)u.
//
// Coefficient layout: coeff[0] = nu, coeff[1..2] = f, coeff[3] = g (= 0),
//   coeff[4..5] = b, coeff[6] = sigma.
// Exact solution layout: values[0] = u, [1] = u_x, [2] = u_y, [3] = Laplace(u)
// (derivatives generated symbolically with sympy)
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

void ExampleFile()
{
  Output::print<1>("Example: Oseen_AAJ_layers.h  (Afzal-Ahmed-John 2026, Example 4.2; "
                   "Khan-Kanschat 2020), lambda = 0.5/sqrt(nu)");
}

static inline double get_lambda()
{
  return 0.5*std::sqrt(TDatabase::ParamDB->RE_NR); // = 0.5/sqrt(nu)
}

// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double y, double *values)
{
  const double lam = get_lambda();
  values[0] = -2*lam*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 2*pow(x, 2)*y*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[1] = 4*pow(lam, 2)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) - 4*lam*pow(x, 2)*y*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 4*lam*x*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 4*x*y*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[2] = -2*pow(lam, 2)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 2*pow(lam, 2)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*exp(2*lam*(y - 1)) - 8*lam*pow(x, 2)*y*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 2*pow(x, 2)*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[3] = -2*pow(lam, 3)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 6*pow(lam, 3)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*exp(2*lam*(y - 1)) + 4*pow(lam, 3)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) - 4*pow(lam, 3)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(y - 1)))*exp(2*lam*(x - 1))*exp(lam*(y - 1)) - 12*pow(lam, 2)*pow(x, 2)*y*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 12*pow(lam, 2)*pow(x, 2)*y*pow(1 - exp(lam*(x - 1)), 2)*exp(2*lam*(y - 1)) - 4*pow(lam, 2)*pow(x, 2)*y*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) + 4*pow(lam, 2)*pow(x, 2)*y*pow(1 - exp(lam*(y - 1)), 2)*exp(2*lam*(x - 1)) + 16*pow(lam, 2)*x*pow(y, 2)*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) - 12*lam*pow(x, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) - 16*lam*x*y*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 4*lam*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 4*y*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
}

void ExactU2(double x, double y, double *values)
{
  const double lam = get_lambda();
  values[0] = 2*lam*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 2*x*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[1] = 2*pow(lam, 2)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 2*pow(lam, 2)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(y - 1)), 2)*exp(2*lam*(x - 1)) + 8*lam*x*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 2*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[2] = -4*pow(lam, 2)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) + 4*lam*pow(x, 2)*y*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) + 4*lam*x*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) - 4*x*y*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
  values[3] = 2*pow(lam, 3)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 4*pow(lam, 3)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) + 4*pow(lam, 3)*pow(x, 2)*pow(y, 2)*(1 - exp(lam*(x - 1)))*exp(lam*(x - 1))*exp(2*lam*(y - 1)) - 6*pow(lam, 3)*pow(x, 2)*pow(y, 2)*pow(1 - exp(lam*(y - 1)), 2)*exp(2*lam*(x - 1)) - 16*pow(lam, 2)*pow(x, 2)*y*(1 - exp(lam*(x - 1)))*(1 - exp(lam*(y - 1)))*exp(lam*(x - 1))*exp(lam*(y - 1)) + 4*pow(lam, 2)*x*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) - 4*pow(lam, 2)*x*pow(y, 2)*pow(1 - exp(lam*(x - 1)), 2)*exp(2*lam*(y - 1)) + 12*pow(lam, 2)*x*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 12*pow(lam, 2)*x*pow(y, 2)*pow(1 - exp(lam*(y - 1)), 2)*exp(2*lam*(x - 1)) + 4*lam*pow(x, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) + 16*lam*x*y*pow(1 - exp(lam*(x - 1)), 2)*(1 - exp(lam*(y - 1)))*exp(lam*(y - 1)) + 12*lam*pow(y, 2)*(1 - exp(lam*(x - 1)))*pow(1 - exp(lam*(y - 1)), 2)*exp(lam*(x - 1)) - 4*x*pow(1 - exp(lam*(x - 1)), 2)*pow(1 - exp(lam*(y - 1)), 2);
}

void ExactP(double x, double y, double *values)
{
  values[0] = exp(x + y) - M_E*(-1 + M_E) - 1 + M_E;
  values[1] = exp(x + y);
  values[2] = exp(x + y);
  values[3] = 2*exp(x + y);
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
    // Oseen: b = (1,1), sigma = 0;  Navier-Stokes: b = u
    const double b1 = navier_stokes ? u1[0] : 1.;
    const double b2 = navier_stokes ? u2[0] : 1.;
    const double sigma = 0.;
    coeffs[i][0] = nu;
    coeffs[i][1] = -nu*u1[3] + b1*u1[1] + b2*u1[2] + sigma*u1[0] + p[1]; // f1
    coeffs[i][2] = -nu*u2[3] + b1*u2[1] + b2*u2[2] + sigma*u2[0] + p[2]; // f2
    coeffs[i][3] = u1[1] + u2[2]; // g (divergence, = 0)
    coeffs[i][4] = b1;
    coeffs[i][5] = b2;
    coeffs[i][6] = sigma;
  }
}
