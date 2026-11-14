// =============================================================================
// Two-dimensional Taylor-Green vortex (Taylor & Green 1937; the form of
// https://en.wikipedia.org/wiki/Taylor-Green_vortex for k = A = 1 on (0,2 pi)^2),
// an exact solution of the incompressible Navier-Stokes equations with f = 0:
//
//   F(t) = exp(-2 nu k^2 t),
//   u = A sin(k x) cos(k y) F,   v = -A cos(k x) sin(k y) F,
//   p = A^2/4 (cos(2 k x) + cos(2 k y)) F^2 - pbar(t),
//   stream function psi = (A/k) sin(k x) sin(k y) F   (u = psi_y, v = -psi_x),
//   vorticity omega = v_x - u_y = 2 A k sin(k x) sin(k y) F = -Laplace(psi),
//
// on the square (0, L)^2 (density 1). pbar = A^2 F^2 sin(2 k L)/(4 k L) is the
// mean of the pressure term, so that p has zero mean (pbar = 0 for k L a
// multiple of pi/2). Kinetic energy and enstrophy for k L = 2 pi (one period,
// e.g. L = 2 pi, k = 1):
//   E = 1/2 int |u|^2 = pi^2 A^2/k^2 exp(-4 nu k^2 t),
//   Z = 1/2 int omega^2 = 2 pi^2 A^2 exp(-4 nu k^2 t),
//   dissipation -dE/dt = 2 nu Z = 4 nu k^2 E.
// All formulas were checked symbolically (sympy): div u = 0, momentum residual
// 0 with f = 0, psi and omega, Laplacians, mean of p, E, Z, -dE/dt = 2 nu Z.
// Note the orientation: with u = cos x sin y, v = -sin x cos y the pressure
// has the opposite sign (ParMooN's SinCosExp.h, Example_TimeNSE2D 2).
//
// Boundary conditions (tgv_periodic):
//   true (default): periodic in x and y, the setting of the literature; needs
//     k L a multiple of 2 pi and 'periodic_translations: (L,0.,0.,L)' in the
//     input file; adaptive meshes are not supported with periodic
//     boundaries (the program stops);
//   false: Dirichlet data from the exact solution on the whole boundary (time
//     dependent; for k L a multiple of pi the sides are streamlines, u.n = 0).
// Both have the same exact solution.
//
// Parameters (input file): tgv_length L (must be the side length of the mesh,
// e.g. data/mesh/SquareTwoPi.PRM + SquareTwoPi4x4_tria.GEO for 2 pi),
// tgv_wavenumber k, tgv_amplitude A, tgv_periodic. Coefficients: see LinCoeffs.
// With the BDF2-IMEX program: bdf2_periodic true, bdf2_start euler (the
// Stokes projection is not defined with periodic boundaries).
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

static double tgv_L = 2.*M_PI, tgv_k = 1., tgv_A = 1.;
static bool tgv_periodic = true;

void ExampleFile()
{
  Output::print<1>("Example: TNSE_taylor_green.h (Taylor-Green vortex), L = ",
                   tgv_L, ", k = ", tgv_k, ", A = ", tgv_A, ", ",
                   tgv_periodic ? "periodic" : "Dirichlet", " boundary conditions");
}

void set_tgv_parameters(double L, double k, double A, bool periodic)
{
  if(L <= 0. || k <= 0.)
    ErrThrow("Taylor-Green vortex: tgv_length and tgv_wavenumber must be "
             "positive");
  tgv_L = L;
  tgv_k = k;
  tgv_A = A;
  tgv_periodic = periodic;
  // periodic boundaries need whole periods: k L = 2 pi m
  const double periods = k*L/(2.*M_PI);
  if(periodic && std::abs(periods - std::round(periods)) > 1e-10)
    ErrThrow("Taylor-Green vortex: periodic boundary conditions need "
             "tgv_wavenumber * tgv_length to be a multiple of 2 pi (it is ",
             k*L, ")");
}

static inline double tgv_F()
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  return std::exp(-2.*nu*tgv_k*tgv_k*TDatabase::TimeDB->CURRENTTIME);
}

// values: u, u_x, u_y, Laplace(u)
void ExactU1(double x, double y, double *values)
{
  const double k = tgv_k, a = tgv_A*tgv_F();
  values[0] = a*std::sin(k*x)*std::cos(k*y);
  values[1] = a*k*std::cos(k*x)*std::cos(k*y);
  values[2] = -a*k*std::sin(k*x)*std::sin(k*y);
  values[3] = -2.*k*k*values[0];
}

void ExactU2(double x, double y, double *values)
{
  const double k = tgv_k, a = tgv_A*tgv_F();
  values[0] = -a*std::cos(k*x)*std::sin(k*y);
  values[1] = a*k*std::sin(k*x)*std::sin(k*y);
  values[2] = -a*k*std::cos(k*x)*std::cos(k*y);
  values[3] = -2.*k*k*values[0];
}

// values: p, p_x, p_y, Laplace(p)
void ExactP(double x, double y, double *values)
{
  const double k = tgv_k, L = tgv_L, F = tgv_F();
  const double c = tgv_A*tgv_A*F*F/4.;
  const double pbar = c*std::sin(2.*k*L)/(k*L);
  values[0] = c*(std::cos(2.*k*x) + std::cos(2.*k*y)) - pbar;
  values[1] = -2.*k*c*std::sin(2.*k*x);
  values[2] = -2.*k*c*std::sin(2.*k*y);
  values[3] = -4.*k*k*c*(std::cos(2.*k*x) + std::cos(2.*k*y));
}

/// exact stream function (u = psi_y, v = -psi_x) and vorticity, for output
/// and post-processing
double tgv_stream_function(double x, double y)
{
  return tgv_A/tgv_k*std::sin(tgv_k*x)*std::sin(tgv_k*y)*tgv_F();
}

double tgv_vorticity(double x, double y)
{
  return 2.*tgv_A*tgv_k*std::sin(tgv_k*x)*std::sin(tgv_k*y)*tgv_F();
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = tgv_periodic ? PERIODIC : DIRICHLET;
}

// boundary parametrization of the square (0, L)^2 (as SquareTwoPi.PRM,
// Default_UnitSquare for L = 1): 0: y = 0, 1: x = L, 2: y = L, 3: x = 0
static inline void boundary_point(int BdComp, double Param, double& x, double& y)
{
  const double L = tgv_L;
  switch(BdComp)
  {
    case 0: x = L*Param;      y = 0.;            break;
    case 1: x = L;            y = L*Param;       break;
    case 2: x = L*(1.-Param); y = L;             break;
    case 3: x = 0.;           y = L*(1.-Param);  break;
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

// Coefficients for the Navier-Stokes programs: coeff[0] = nu, coeff[1..2] = f
// = 0, coeff[3] = div u = 0, coeff[4] = 1 - nu, coeff[5..6] = 0.
// coeff[4] = 1 - nu is the convention of the BDF2-IMEX program: ParMooN's
// grad-div routine uses graddiv_stab (nu + coeff[4] L_0^2), which then equals
// graddiv_stab (L_0 = 1). The SUPG/PSPG/grad-div program reads coeff[4..6]
// only for Oseen problems (problem_type 3, b = coeff[4..5]), which this
// example is not used for (b = u_h for Navier-Stokes).
void LinCoeffs(int n_points, const double *, const double *,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = nu;
    coeffs[i][1] = 0.;   // f = 0: exact Navier-Stokes solution
    coeffs[i][2] = 0.;
    coeffs[i][3] = 0.;   // div u = 0
    coeffs[i][4] = 1. - nu;
    coeffs[i][5] = 0.;
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
