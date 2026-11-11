// =============================================================================
// Translating vortex with compact support (Example 2 of the manuscript on the
// adaptive SUPG/PSPG/grad-div method), a smooth relative of the Gresho vortex
// (Gresho & Chan 1990):
//
//   xi = x - c0 - U0 t,  r = |xi|,  rho = r^2/R^2,
//   h(rho) = A (1 - rho)^m for rho < 1, 0 otherwise,
//   u = U0 + h (-xi_2, xi_1),
//   p = P - pbar,  P = -A^2 R^2/(2(2m+1)) (1 - rho)^(2m+1) for rho < 1, 0 otherwise,
//   pbar = -pi A^2 R^4 / (2 (2m+1) (2m+2))   (mean of P over Omega = (0,1)^2
//                                            while the vortex lies inside).
//
// The radial vortex is a steady solution of the Euler equations (P' = g^2/r
// with g = r h), and the uniform translation keeps it one, so
//   d_t u + (u.grad)u + grad p = 0,   f = -nu Laplace(u)
//     = -nu (4 A m/R^2) (1-rho)^(m-2) ((m+1) rho - 2) (-xi_2, xi_1).
// The right-hand side is computed below from the full momentum equation (not
// from this identity); both were checked symbolically (sympy): div u = 0, the
// Euler residual vanishes, f = -nu Laplace(u), the pressure mean.
// u in C^(m-1), f in C^(m-3) across r = R; m >= 3 is required.
// Dirichlet data u = U0 on the boundary (time independent while the vortex is
// inside, checked at construction for [time_start, time_end]).
// Vorticity omega = 2 A (1-rho)^(m-1) (1 - (m+1) rho): positive core, negative
// ring, total circulation 0; kinetic energy and enstrophy are constant in
// time (checked symbolically for m = 3,...,6):
//   int |u - U0|^2 = pi A^2 R^4 / ((2m+1)(2m+2)),  int u - U0 = 0,
//   int omega^2 = 4 m pi A^2 R^2 / ((2m-1)(2m+1)).
// The swirl amplitude A is computed from the maximal swirl speed
// v_max = A R (2m+1)^(-1/2) (2m/(2m+1))^m.
// Parameters (input file): vortex_radius R, vortex_exponent m,
// vortex_max_swirl v_max, vortex_center_x/y c0, vortex_velocity_x/y U0.
// Coefficient layout as in TNSE_AAJ_smooth.h (coeff[4..5] = b = u).
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

static double vortex_R = 0.15, vortex_A = 0., vortex_c0[2] = {0.25, 0.5},
              vortex_U0[2] = {1., 0.};
static int vortex_m = 4;

void ExampleFile()
{
  Output::print<1>("Example: TNSE_translating_vortex.h, R = ", vortex_R, ", m = ",
                   vortex_m, ", A = ", vortex_A, ", c0 = (", vortex_c0[0], ", ",
                   vortex_c0[1], "), U0 = (", vortex_U0[0], ", ", vortex_U0[1], ")");
}

void set_vortex_parameters(double R, int m, double v_max, double cx, double cy,
                           double Ux, double Uy)
{
  if(m < 3)
    ErrThrow("translating vortex: vortex_exponent must be at least 3 (f in C^0)");
  if(R <= 0.)
    ErrThrow("translating vortex: vortex_radius must be positive");
  vortex_R = R;
  vortex_m = m;
  vortex_c0[0] = cx;  vortex_c0[1] = cy;
  vortex_U0[0] = Ux;  vortex_U0[1] = Uy;
  vortex_A = v_max*std::sqrt(2.*m + 1.)/(R*std::pow(2.*m/(2.*m + 1.), m));
}

/// distance of the vortex (centre +- R) to the boundary of (0,1)^2 on [t0, t1]
double vortex_distance_to_boundary(double t0, double t1)
{
  double dmin = 1e300;
  for(double t : {t0, t1}) // the path is a straight line, its extremes suffice
    for(int k = 0; k < 2; ++k)
    {
      double c = vortex_c0[k] + vortex_U0[k]*t;
      dmin = std::min(dmin, std::min(c, 1. - c) - vortex_R);
    }
  return dmin;
}

// local quantities: xi, rho, h, h_rho, h_rhorho
static inline void vortex_local(double x, double y, double& xi1, double& xi2,
                                double& rho, double& h, double& h1, double& h2)
{
  const double t = TDatabase::TimeDB->CURRENTTIME;
  xi1 = x - vortex_c0[0] - vortex_U0[0]*t;
  xi2 = y - vortex_c0[1] - vortex_U0[1]*t;
  rho = (xi1*xi1 + xi2*xi2)/(vortex_R*vortex_R);
  const int m = vortex_m;
  const double A = vortex_A;
  if(rho < 1.)
  {
    const double s = 1. - rho;
    h  = A*std::pow(s, m);
    h1 = -m*A*std::pow(s, m - 1);
    h2 = m*(m - 1)*A*std::pow(s, m - 2);
  }
  else
    h = h1 = h2 = 0.;
}

// values: u, u_x, u_y, Laplace(u)
void ExactU1(double x, double y, double *values)
{
  double xi1, xi2, rho, h, h1, h2;
  vortex_local(x, y, xi1, xi2, rho, h, h1, h2);
  const double R2 = vortex_R*vortex_R;
  const double hx = 2.*h1*xi1/R2, hy = 2.*h1*xi2/R2;
  const double lap_h = 4.*(rho*h2 + h1)/R2;
  values[0] = vortex_U0[0] - h*xi2;
  values[1] = -hx*xi2;
  values[2] = -hy*xi2 - h;
  values[3] = -(xi2*lap_h + 2.*hy);
}

void ExactU2(double x, double y, double *values)
{
  double xi1, xi2, rho, h, h1, h2;
  vortex_local(x, y, xi1, xi2, rho, h, h1, h2);
  const double R2 = vortex_R*vortex_R;
  const double hx = 2.*h1*xi1/R2, hy = 2.*h1*xi2/R2;
  const double lap_h = 4.*(rho*h2 + h1)/R2;
  values[0] = vortex_U0[1] + h*xi1;
  values[1] = hx*xi1 + h;
  values[2] = hy*xi1;
  values[3] = xi1*lap_h + 2.*hx;
}

// values: p, p_x, p_y, Laplace(p)
void ExactP(double x, double y, double *values)
{
  double xi1, xi2, rho, h, h1, h2;
  vortex_local(x, y, xi1, xi2, rho, h, h1, h2);
  const int m = vortex_m;
  const double A = vortex_A, R = vortex_R;
  const double pbar = -M_PI*A*A*R*R*R*R/(2.*(2.*m + 1.)*(2.*m + 2.));
  if(rho < 1.)
  {
    const double s = 1. - rho;
    const double s2m = std::pow(s, 2*m);
    values[0] = -A*A*R*R/(2.*(2.*m + 1.))*s2m*s - pbar;
    values[1] = A*A*s2m*xi1;
    values[2] = A*A*s2m*xi2;
    values[3] = A*A*std::pow(s, 2*m - 1)*(2.*s - 4.*m*rho);
  }
  else
  {
    values[0] = -pbar;
    values[1] = values[2] = values[3] = 0.;
  }
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

// f = d_t u + (u.grad) u + grad p - nu Laplace(u), d_t u = -(U0.grad) u
void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    const double ut1 = -(vortex_U0[0]*u1[1] + vortex_U0[1]*u1[2]);
    const double ut2 = -(vortex_U0[0]*u2[1] + vortex_U0[1]*u2[2]);
    coeffs[i][0] = nu;
    coeffs[i][1] = ut1 + u1[0]*u1[1] + u2[0]*u1[2] + p[1] - nu*u1[3];
    coeffs[i][2] = ut2 + u1[0]*u2[1] + u2[0]*u2[2] + p[2] - nu*u2[3];
    coeffs[i][3] = 0.;
    coeffs[i][4] = u1[0];
    coeffs[i][5] = u2[0];
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
