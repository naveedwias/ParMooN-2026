// =============================================================================
// Verification example for the SUPG/PSPG/grad-div method and its a posteriori
// estimator (stationary Oseen and Navier-Stokes):
//   u = (y^2, x^2),  p = x + y - 1  on the unit square,
// Oseen (FLOW_PROBLEM_TYPE 3): convection b = (1 + y, 2 - x) (divergence-free,
// unrelated to u), sigma = 1;  Navier-Stokes (FLOW_PROBLEM_TYPE 5): b = u,
// sigma = 0; f = -nu Laplace(u) + (b.grad)u + sigma u + grad p.
// u is quadratic and divergence-free, p linear with zero mean: (u, p) lies in
// every pair containing P2/P1, so a consistent method reproduces it exactly
// and every residual of the estimator vanishes (cell residual, divergence,
// facet jumps of nu grad u - p I, delta and mu terms). Any nonzero error or
// estimator contribution beyond round-off points to a bug.
// Coefficient layout as in Oseen_AAJ_smooth.h.
// =============================================================================
void ExampleFile()
{
  Output::print<1>("Example: Oseen_SPG_polynomial.h (u = (y^2, x^2), p = x+y-1)");
}

void ExactU1(double, double y, double *values)
{
  values[0] = y*y;  values[1] = 0.;  values[2] = 2.*y;  values[3] = 2.;
}

void ExactU2(double x, double, double *values)
{
  values[0] = x*x;  values[1] = 2.*x;  values[2] = 0.;  values[3] = 2.;
}

void ExactP(double x, double y, double *values)
{
  values[0] = x + y - 1.;  values[1] = 1.;  values[2] = 1.;  values[3] = 0.;
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
  const bool navier_stokes = (TDatabase::ParamDB->FLOW_PROBLEM_TYPE == 5);
  double u1[4], u2[4], p[4];
  for(int i = 0; i < n_points; i++)
  {
    ExactU1(X[i], Y[i], u1);
    ExactU2(X[i], Y[i], u2);
    ExactP(X[i], Y[i], p);
    const double b1 = navier_stokes ? u1[0] : 1. + Y[i];
    const double b2 = navier_stokes ? u2[0] : 2. - X[i];
    const double sigma = navier_stokes ? 0. : 1.;
    coeffs[i][0] = nu;
    coeffs[i][1] = -nu*u1[3] + b1*u1[1] + b2*u1[2] + sigma*u1[0] + p[1];
    coeffs[i][2] = -nu*u2[3] + b1*u2[1] + b2*u2[2] + sigma*u2[0] + p[2];
    coeffs[i][3] = u1[1] + u2[2];
    coeffs[i][4] = b1;
    coeffs[i][5] = b2;
    coeffs[i][6] = sigma;
  }
}
