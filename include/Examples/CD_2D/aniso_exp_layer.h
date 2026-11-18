// ======================================================================
// Exponential boundary layer at x = 1 (Example 1 of the study of algebraic
// stabilizations on layer-adapted meshes, galerkit problem 'exp_layer'):
//   -eps Laplace(u) + b.grad(u) = 0 in (0,1)^2,  b = (1,0),  c = 0,  f = 0,
//   u(x,y) = (1 - exp((x-1)/eps)) / (1 - exp(-1/eps))  in [0,1],
// Dirichlet data from u on the whole boundary. Since f = c = 0 the local
// discrete maximum principle holds and can be checked node by node.
// ======================================================================
double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: aniso_exp_layer.h, eps = ", DIFFUSION);
}

// values: u, u_x, u_y, Laplace(u)
void Exact(double x, double, double *values)
{
  const double eps = DIFFUSION;
  const double den = -std::expm1(-1./eps);          // 1 - exp(-1/eps), accurate
  const double e = std::exp((x - 1.)/eps);
  values[0] = -std::expm1((x - 1.)/eps)/den;        // (1 - e)/den
  values[1] = -e/(eps*den);
  values[2] = 0.;
  values[3] = -e/(eps*eps*den);
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void BoundValue(int BdComp, double Param, double &value)
{
  double x = 0., y = 0., v[4];
  switch(BdComp)
  {
    case 0: x = Param;      y = 0.;           break;
    case 1: x = 1.;         y = Param;        break;
    case 2: x = 1. - Param; y = 1.;           break;
    case 3: x = 0.;         y = 1. - Param;   break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
  Exact(x, y, v);
  value = v[0];
}

void BilinearCoeffs(int n_points, const double *, const double *,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = DIFFUSION;
    coeffs[i][1] = 1.;   // b_1
    coeffs[i][2] = 0.;   // b_2
    coeffs[i][3] = 0.;   // c
    coeffs[i][4] = 0.;   // f
  }
}
