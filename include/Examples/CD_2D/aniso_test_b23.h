// ======================================================================
// Test problem for cross-checks of algebraic stabilizations (data that every
// code integrates exactly): b = (2,3), c = 1, f = 0, Dirichlet data
// u_b = x + y^2 on the whole boundary; layers at x = 1 and y = 1. The exact
// solution is not known (Exact returns 0).
// ======================================================================
double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: aniso_test_b23.h, eps = ", DIFFUSION);
}

void Exact(double, double, double *values)
{
  values[0] = values[1] = values[2] = values[3] = 0.;
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void BoundValue(int BdComp, double Param, double &value)
{
  double x = 0., y = 0.;
  switch(BdComp)
  {
    case 0: x = Param;      y = 0.;           break;
    case 1: x = 1.;         y = Param;        break;
    case 2: x = 1. - Param; y = 1.;           break;
    case 3: x = 0.;         y = 1. - Param;   break;
    default: ErrThrow("wrong boundary part number ", BdComp);
  }
  value = x + y*y;
}

void BilinearCoeffs(int n_points, const double *, const double *,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = DIFFUSION;
    coeffs[i][1] = 2.;   // b_1
    coeffs[i][2] = 3.;   // b_2
    coeffs[i][3] = 1.;   // c
    coeffs[i][4] = 0.;   // f
  }
}
