// ======================================================================
// Straight interior layer (Example 4 of the study of algebraic
// stabilizations on layer-adapted meshes, galerkit interior_layer.py):
//   -eps Laplace(u) + u_x = 0 in (0,1)^2, b = (1,0), c = f = 0,
//   u_b = 1 on {x = 0, y > 1/2} and on {y = 1, x < 1}, u_b = 0 on the rest of
//   the boundary (also at the outflow x = 1), u_b(0, 1/2) = 1/2.
// The jump of the inflow data gives an interior parabolic layer along y = 1/2
// of width O(sqrt(eps x)) and an exponential outflow layer at x = 1 for
// y > 1/2. No exact solution; Exact returns the reference
//   u_ref(x, y) = 1/2 erfc((1/2 - y) / (2 sqrt(eps x))),
// which solves u_x = eps u_yy (the neglected eps u_xx is negligible for small
// eps) and is used away from the outflow layer (x <= 1 - tau_x).
// ======================================================================
double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: aniso_interior_layer.h, eps = ", DIFFUSION);
}

// values: u_ref and its derivatives (Laplace: u_yy only, as a reference)
void Exact(double x, double y, double *values)
{
  const double eps = DIFFUSION;
  if(x <= 0.)
  {
    values[0] = y > 0.5 ? 1. : (y < 0.5 ? 0. : 0.5);
    values[1] = values[2] = values[3] = 0.;
    return;
  }
  const double s = 2.*std::sqrt(eps*x);
  const double z = (0.5 - y)/s;
  const double g = std::exp(-z*z)/std::sqrt(M_PI);   // -d/dz of erfc(z)/2 ... times 1
  values[0] = 0.5*std::erfc(z);
  values[2] = g/s;                                     // d/dy
  values[1] = g*z/(2.*x);                              // d/dx
  values[3] = 2.*z*g/(s*s);                            // d2/dy2
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void BoundValue(int BdComp, double Param, double &value)
{
  switch(BdComp)
  {
    case 0: value = 0.; break;                         // y = 0
    case 1: value = 0.; break;                         // x = 1 (outflow)
    case 2: value = (Param > 0.) ? 1. : 0.; break;     // y = 1, x = 1 - Param < 1
    case 3:                                            // x = 0, y = 1 - Param
      value = (Param < 0.5) ? 1. : (Param > 0.5 ? 0. : 0.5); break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
}

void BilinearCoeffs(int n_points, const double *, const double *,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = DIFFUSION;
    coeffs[i][1] = 1.;
    coeffs[i][2] = 0.;
    coeffs[i][3] = 0.;
    coeffs[i][4] = 0.;
  }
}
