// =============================================================================
// Time-dependent Navier-Stokes example whose boundary layers move to other walls
// (Example_TimeNSE2D 36 and 37), for the adaptive SUPG/PSPG/grad-div program
// tnse2d_adaptive_spg: in the first part of the time interval the solution has
// boundary layers at x = 1 and y = 1; this part is switched off smoothly and a
// second steady solution with layers at other walls is switched on, so that an
// adaptive method has to remove its refinement at x = 1, y = 1 and refine
// elsewhere.
//
// Steady building blocks (unit square, all of the same type as
// Oseen_AAJ_layers.h, Afzal-Ahmed-John 2026, Example 4.2 / Khan-Kanschat 2020):
//
//   u = curl phi = (phi_y, -phi_x),   phi(x, y) = A(xi) A(eta),
//   A(s) = s^2 (1 - exp(k (s - 1)))^2,
//   xi = x or 1 - x,  eta = y or 1 - y   (mirror: the layer moves to the wall
//                                         where xi = 1 or eta = 1),
//   p = exp(xi + eta) - (e - 1)^2        (zero mean on the unit square).
//
//   NSE1: xi = x,     eta = y,      k = lambda = kappa_1 / sqrt(nu), kappa_1 = 0.5
//         layers at x = 1 and y = 1 (this is exactly Oseen_AAJ_layers.h);
//   NSE2: k = mu = kappa_2 / sqrt(nu), kappa_2 = 0.75 (layers 1.5 times thinner),
//         variant a (example 36): xi = 1 - x, eta = 1 - y, layers at x = 0, y = 0
//                                 (the refinement moves completely),
//         variant b (example 37): xi = x,     eta = 1 - y, layers at x = 1, y = 0
//                                 (y = 1 -> y = 0; x = 1 stays, with width 1/mu).
//
//   A(s) and A'(s) vanish at s = 0 and s = 1, so u = 0 on the whole boundary
//   for every building block: homogeneous Dirichlet data at all times. The
//   layer width is about 1/k; the peak velocity grows like k (|u| max 19.5 for
//   NSE1 and 31.4 for NSE2 at nu = 1e-4), therefore NSE2 is scaled by
//   A_2 = lambda/mu = kappa_1/kappa_2 (layer_switch_scale_nse2: true), which
//   makes the peak velocities of both parts equal to within about 10 %.
//
// Time-dependent solution on [time_start, time_end] = [a, b] (proposed [0, 1]):
//
//   u(t) = T_1(t) u_1 + T_2(t) u_2,   p(t) = T_1(t) p_1 + T_2(t) p_2,
//   T_1(t) = 1 - S((t - c_1)/(c - c_1)),       1 on [a, c_1], 0 on [c, b],
//   T_2(t) = A_2 S((t - d)/(d_2 - d)),         0 on [a, d],  A_2 on [d_2, b],
//   S(tau) = tau^3 (10 - 15 tau + 6 tau^2) on [0, 1], 0 before, 1 after (C^2),
//
// with the times as absolute values (parameters layer_switch_c1, _c, _d, _d2;
// defaults c_1 = 0.3, c = d = 0.5, d_2 = 0.7). With c = d both factors vanish at
// t = c, so u(c) = 0 there (a continuous hand-over through zero); with c > d,
// e.g. d = 0.4, c = 0.6, both layer systems are present on (d, c).
// The solution is divergence-free, satisfies the homogeneous Dirichlet
// condition and has a pressure with zero mean at every t; u(a) = u_1 if a <= c_1.
//
// Right-hand side: f = d_t u - nu Laplace(u) + (u.grad)u + grad p with
// d_t u = T_1' u_1 + T_2' u_2; the convective term mixes both parts while both
// are active. The time is TDatabase::TimeDB->CURRENTTIME, nu = 1/RE_NR.
// Exact solution layout: values[0] = u, [1] = u_x, [2] = u_y, [3] = Laplace(u)
// (pressure: p, p_x, p_y, Laplace(p)). Coefficient layout as in
// TNSE_AAJ_smooth.h: coeff[0] = nu, [1..2] = f, [3] = div u (= 0),
// [4..5] = b = u (Navier-Stokes), [6] = 0.
//
// The variant is chosen by TNSE_LAYER_SWITCH_VARIANT (1: a, 2: b) before the
// include (see Example_TimeNSE2D.C); the parameters are set by
// set_layer_switch_parameters() from the example database.
// =============================================================================
#include "Constants.h"
#include "MooNMD_Io.h"
#include <cmath>

#ifndef TNSE_LAYER_SWITCH_VARIANT
#error "define TNSE_LAYER_SWITCH_VARIANT (1: layers to x = 0, y = 0; 2: to x = 1, y = 0)"
#endif

// ------------------------------------------------------------------ parameters
double kappa_1 = 0.5;     // lambda = kappa_1 / sqrt(nu)  (NSE1)
double kappa_2 = 0.75;    // mu     = kappa_2 / sqrt(nu)  (NSE2)
bool scale_nse2 = true;   // A_2 = kappa_1 / kappa_2, otherwise 1
double t_c1 = 0.3, t_c = 0.5, t_d = 0.5, t_d2 = 0.7;

void set_layer_switch_parameters(double k1, double k2, bool scale, double c1, double c,
                                 double d, double d2)
{
  if(!(k1 > 0. && k2 > 0.))
    ErrThrow("layer switch example: layer_switch_kappa1 and _kappa2 must be positive");
  if(!(c1 < c) || !(d < d2))
    ErrThrow("layer switch example: need layer_switch_c1 < layer_switch_c and "
             "layer_switch_d < layer_switch_d2, got c1 = ", c1, ", c = ", c,
             ", d = ", d, ", d2 = ", d2);
  kappa_1 = k1; kappa_2 = k2; scale_nse2 = scale;
  t_c1 = c1; t_c = c; t_d = d; t_d2 = d2;
}

void ExampleFile()
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  Output::print<1>("Example: TNSE_layer_switch.h, variant ",
                   TNSE_LAYER_SWITCH_VARIANT == 1 ? "a (layers move from x = 1, y = 1 "
                   "to x = 0, y = 0)" : "b (layers move from x = 1, y = 1 to x = 1, y = 0)");
  Output::print<1>("  lambda = ", kappa_1, "/sqrt(nu) = ", kappa_1/std::sqrt(nu),
                   ", mu = ", kappa_2, "/sqrt(nu) = ", kappa_2/std::sqrt(nu),
                   ", A_2 = ", scale_nse2 ? kappa_1/kappa_2 : 1.);
  Output::print<1>("  NSE1 switched off on [", t_c1, ", ", t_c, "], NSE2 switched on on [",
                   t_d, ", ", t_d2, "]");
}

// ------------------------------------------------------------------ building blocks
// A(s) = s^2 w(s)^2, w = 1 - exp(k (s - 1)); a[0..3] = A, A', A'', A'''
static void profile(double s, double k, double *a)
{
  const double E = std::exp(k*(s - 1.));
  const double w = 1. - E, w1 = -k*E, w2 = -k*k*E, w3 = -k*k*k*E;
  a[0] = s*s*w*w;
  a[1] = 2.*s*w*w + 2.*s*s*w*w1;
  a[2] = 2.*w*w + 8.*s*w*w1 + 2.*s*s*(w1*w1 + w*w2);
  a[3] = 12.*w*w1 + 12.*s*(w1*w1 + w*w2) + 2.*s*s*(3.*w1*w2 + w*w3);
}

// one steady solution: u = curl(A(xi) A(eta)), p = exp(xi + eta) - (e-1)^2,
// xi = mx ? 1 - x : x, eta = my ? 1 - y : y; arrays as in the exact solution
static void steady_solution(double x, double y, double k, bool mx, bool my,
                            double *u1, double *u2, double *p)
{
  const double sx = mx ? -1. : 1., sy = my ? -1. : 1.;
  const double xi = mx ? 1. - x : x, eta = my ? 1. - y : y;
  double a[4], b[4];
  profile(xi, k, a);
  profile(eta, k, b);
  // u1 = phi_y = sy A(xi) A'(eta)
  u1[0] = sy*a[0]*b[1];
  u1[1] = sx*sy*a[1]*b[1];
  u1[2] = a[0]*b[2];
  u1[3] = sy*(a[2]*b[1] + a[0]*b[3]);
  // u2 = -phi_x = -sx A'(xi) A(eta)
  u2[0] = -sx*a[1]*b[0];
  u2[1] = -a[2]*b[0];
  u2[2] = -sx*sy*a[1]*b[1];
  u2[3] = -sx*(a[3]*b[0] + a[1]*b[2]);
  const double ep = std::exp(xi + eta);
  p[0] = ep - (M_E - 1.)*(M_E - 1.);
  p[1] = sx*ep;
  p[2] = sy*ep;
  p[3] = 2.*ep;
}

static void nse1(double x, double y, double *u1, double *u2, double *p)
{
  const double k = kappa_1*std::sqrt(TDatabase::ParamDB->RE_NR);
  steady_solution(x, y, k, false, false, u1, u2, p);
}

static void nse2(double x, double y, double *u1, double *u2, double *p)
{
  const double k = kappa_2*std::sqrt(TDatabase::ParamDB->RE_NR);
#if TNSE_LAYER_SWITCH_VARIANT == 1
  steady_solution(x, y, k, true, true, u1, u2, p);    // layers at x = 0, y = 0
#else
  steady_solution(x, y, k, false, true, u1, u2, p);   // layers at x = 1, y = 0
#endif
}

// ------------------------------------------------------------------ time factors
// S(tau) = tau^3 (10 - 15 tau + 6 tau^2), C^2 step from 0 to 1 on [0, 1]
static double step(double tau) { return tau <= 0. ? 0. : (tau >= 1. ? 1. : tau*tau*tau*(10. - 15.*tau + 6.*tau*tau)); }
static double step_t(double tau) { return (tau <= 0. || tau >= 1.) ? 0. : 30.*tau*tau*(1. - tau)*(1. - tau); }

static double time1(double t) { return 1. - step((t - t_c1)/(t_c - t_c1)); }
static double time1_t(double t) { return -step_t((t - t_c1)/(t_c - t_c1))/(t_c - t_c1); }
static double amplitude2() { return scale_nse2 ? kappa_1/kappa_2 : 1.; }
static double time2(double t) { return amplitude2()*step((t - t_d)/(t_d2 - t_d)); }
static double time2_t(double t) { return amplitude2()*step_t((t - t_d)/(t_d2 - t_d))/(t_d2 - t_d); }

// u, p of the time-dependent solution and d_t u at (x, y, t)
static void solution(double x, double y, double t, double *u1, double *u2, double *p,
                     double *u1_t = nullptr, double *u2_t = nullptr)
{
  double a1[4], a2[4], ap[4], b1[4], b2[4], bp[4];   // NSE1 (a) and NSE2 (b)
  const double g1 = time1(t), g2 = time2(t);
  nse1(x, y, a1, a2, ap);
  nse2(x, y, b1, b2, bp);
  for(int i = 0; i < 4; ++i)
  {
    u1[i] = g1*a1[i] + g2*b1[i];
    u2[i] = g1*a2[i] + g2*b2[i];
    p[i]  = g1*ap[i] + g2*bp[i];
  }
  if(u1_t)
  {
    const double h1 = time1_t(t), h2 = time2_t(t);
    *u1_t = h1*a1[0] + h2*b1[0];
    *u2_t = h1*a2[0] + h2*b2[0];
  }
}

// ------------------------------------------------------------------ exact solution
void ExactU1(double x, double y, double *values)
{
  double u2[4], p[4];
  solution(x, y, TDatabase::TimeDB->CURRENTTIME, values, u2, p);
}

void ExactU2(double x, double y, double *values)
{
  double u1[4], p[4];
  solution(x, y, TDatabase::TimeDB->CURRENTTIME, u1, values, p);
}

void ExactP(double x, double y, double *values)
{
  double u1[4], u2[4];
  solution(x, y, TDatabase::TimeDB->CURRENTTIME, u1, u2, values);
}

// ------------------------------------------------------------------ boundary conditions
// u = 0 on the whole boundary for every t (A and A' vanish at s = 0 and s = 1)
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

// ------------------------------------------------------------------ coefficients
void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  const double t = TDatabase::TimeDB->CURRENTTIME;
  double u1[4], u2[4], p[4], u1_t, u2_t;
  for(int i = 0; i < n_points; i++)
  {
    solution(X[i], Y[i], t, u1, u2, p, &u1_t, &u2_t);
    coeffs[i][0] = nu;
    coeffs[i][1] = u1_t - nu*u1[3] + u1[0]*u1[1] + u2[0]*u1[2] + p[1];   // f1
    coeffs[i][2] = u2_t - nu*u2[3] + u1[0]*u2[1] + u2[0]*u2[2] + p[2];   // f2
    coeffs[i][3] = u1[1] + u2[2];                                        // div u (= 0)
    coeffs[i][4] = u1[0];                                                // b = u
    coeffs[i][5] = u2[0];
    coeffs[i][6] = 0.;
  }
}
