// =============================================================================
// Example 5 of de Frutos, Garcia-Archilla, John, Novo, Comput. Methods Appl.
// Mech. Engrg. 273 (2014) 219-237: moving internal layer and a developing
// exponential layer.
//   u_t - eps Laplace(u) + (y, -x).grad(u) = 0,  u_0 = 0, eps = 1e-5
//   u = 1 on x = 0, u = 0 on x = 1, y = 0, y = 1.
// A front is generated at x = 0 and rotates clockwise around the origin; when
// it reaches y = 0 an exponential boundary layer develops there.
// No exact solution is known; Exact returns zero.
// =============================================================================
#ifndef __FGJN14_EXAMPLE5__
#define __FGJN14_EXAMPLE5__

double diffusion_coefficient = 1e-5;
constexpr bool rhs_depends_on_time = false;
constexpr bool coefficients_depend_on_time = false;

void ExampleFile()
{
  Output::print("Example: FGJN14_Example5.h (de Frutos et al. 2014, Example 5: "
                "rotating front), eps = ", diffusion_coefficient);
}

void Exact(double, double, double *values)
{
  values[0] = 0.;
  values[1] = 0.;
  values[2] = 0.;
  values[3] = 0.;
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

// boundary components of the unit square: 0: y = 0, 1: x = 1, 2: y = 1, 3: x = 0
void BoundValue(int BdComp, double Param, double &value)
{
  value = 0.;
  if(BdComp == 3)
  {
    value = 1.;
    // avoid the discontinuity at the corners (0,0) and (0,1)
    if(Param < 1e-12 || Param > 1. - 1e-12)
      value = 0.;
  }
}

void InitialCondition(double, double, double *values)
{
  values[0] = 0.;
}

void BilinearCoeffs(int n_points, const double *X, const double *Y,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    double *coeff = coeffs[i];
    coeff[0] = diffusion_coefficient;
    coeff[1] = Y[i];
    coeff[2] = -X[i];
    coeff[3] = 0.;
    coeff[4] = 0.;
    coeff[5] = 0.;
  }
}

#endif
