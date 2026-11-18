// ======================================================================
// Profile transported along circles (Example 5 of the study of algebraic
// stabilizations on layer-adapted meshes; the steady problem of arXiv 2204,
// Sect. 7; galerkit rotating_profile.py):
//   -eps Laplace(u) + b.grad(u) = 0 in (0,1)^2, b = (-y, x), c = f = 0,
// inflow profile at y = 0 (xi = 1e-3):
//   (x - 0.375)/xi + 1            on [0.375 - xi, 0.375),
//   -0.75 (x - 0.5)/0.125 + 0.25  on [0.375, 0.5),
//   0.25 (x - 0.625)/0.125 + 0.5  on [0.5, 0.625),
//   -0.5 (x - 0.625)/xi + 0.5     on [0.625, 0.625 + xi),
//   32 (x - 0.75)(1 - x)          on [0.75, 1], 0 else;
// u = 0 at x = 1, homogeneous Neumann conditions at the outflow boundaries
// x = 0 and y = 1. Steep interior layers along r = 0.375 and r = 0.625, weak
// ones along r = 0.5 and r = 0.75, 0 <= u <= 1. No exact solution (Exact
// returns 0); the reference solution of the radial heat equation is computed
// in the post-processing (scripts/aniso_afc).
// ======================================================================
double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: aniso_rotating_profile.h, eps = ", DIFFUSION);
}

void Exact(double, double, double *values)
{
  values[0] = values[1] = values[2] = values[3] = 0.;
}

double inflow_profile(double x)
{
  const double xi = 1e-3;
  if(x >= 0.375 - xi && x < 0.375)
    return (x - 0.375)/xi + 1.;
  if(x >= 0.375 && x < 0.5)
    return -0.75*(x - 0.5)/0.125 + 0.25;
  if(x >= 0.5 && x < 0.625)
    return 0.25*(x - 0.625)/0.125 + 0.5;
  if(x >= 0.625 && x < 0.625 + xi)
    return -0.5*(x - 0.625)/xi + 0.5;
  if(x >= 0.75 && x <= 1.)
    return 32.*(x - 0.75)*(1. - x);
  return 0.;
}

void BoundCondition(int BdComp, double, BoundCond &cond)
{
  // 0: y = 0 (inflow), 1: x = 1 (u = 0), 2: y = 1, 3: x = 0 (outflow, Neumann)
  cond = (BdComp == 2 || BdComp == 3) ? NEUMANN : DIRICHLET;
}

void BoundValue(int BdComp, double Param, double &value)
{
  value = (BdComp == 0) ? inflow_profile(Param) : 0.;
}

void BilinearCoeffs(int n_points, const double *X, const double *Y,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = DIFFUSION;
    coeffs[i][1] = -Y[i];
    coeffs[i][2] = X[i];
    coeffs[i][3] = 0.;
    coeffs[i][4] = 0.;
  }
}
