// =============================================================================
// Example 1 of the adaptive SUPG/PSPG/grad-div method for the time-dependent
// Navier-Stokes equations (transient extension of Afzal, Ahmed, John 2026):
// the steady solution of Oseen_AAJ_smooth.h multiplied by
//   g(t) = 1 + sin(pi t)/2 for the velocity (divergence-free) and
//   h(t) = (1. + 0.5*std::cos(M_PI*t)) for the pressure.
// f = d_t u - nu Laplace(u) + (u.grad)u + grad p, homogeneous Dirichlet data,
// u_0 = u(0). The time is TDatabase::TimeDB->CURRENTTIME.
// Coefficient layout as in Oseen_AAJ_smooth.h (coeff[4..5] = b = u, coeff[6] = 0).
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

namespace steady
{
#include "NSE_2D/Oseen_AAJ_smooth.h"
}

void ExampleFile()
{
  Output::print<1>("Example: TNSE_AAJ_smooth.h (transient Example 1, steady part from "
                   "Afzal-Ahmed-John 2026)");
}

static inline double time_factor_u(double t) { return 1. + 0.5*std::sin(M_PI*t); }
static inline double time_factor_u_t(double t) { return 0.5*M_PI*std::cos(M_PI*t); }
static inline double time_factor_p(double t) { return (1. + 0.5*std::cos(M_PI*t)); }

void ExactU1(double x, double y, double *values)
{
  steady::ExactU1(x, y, values);
  const double g = time_factor_u(TDatabase::TimeDB->CURRENTTIME);
  for(int i = 0; i < 4; ++i) values[i] *= g;
}

void ExactU2(double x, double y, double *values)
{
  steady::ExactU2(x, y, values);
  const double g = time_factor_u(TDatabase::TimeDB->CURRENTTIME);
  for(int i = 0; i < 4; ++i) values[i] *= g;
}

void ExactP(double x, double y, double *values)
{
  steady::ExactP(x, y, values);
  const double g = time_factor_p(TDatabase::TimeDB->CURRENTTIME);
  for(int i = 0; i < 4; ++i) values[i] *= g;
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
  const double t = TDatabase::TimeDB->CURRENTTIME;
  const double g = time_factor_u(t), gt = time_factor_u_t(t);
  const double gp = time_factor_p(t);
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    steady::ExactU1(X[i], Y[i], u1);
    steady::ExactU2(X[i], Y[i], u2);
    steady::ExactP(X[i], Y[i], p);
    // u = g u_s, p = gp p_s
    const double b1 = g*u1[0], b2 = g*u2[0];
    coeffs[i][0] = nu;
    coeffs[i][1] = gt*u1[0] - nu*g*u1[3] + b1*g*u1[1] + b2*g*u1[2] + gp*p[1];
    coeffs[i][2] = gt*u2[0] - nu*g*u2[3] + b1*g*u2[1] + b2*g*u2[2] + gp*p[2];
    coeffs[i][3] = g*(u1[1] + u2[2]);
    coeffs[i][4] = b1;
    coeffs[i][5] = b2;
    coeffs[i][6] = 0.;
  }
}
