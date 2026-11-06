// =============================================================================
// Example 3 of de Frutos, Garcia-Archilla, John, Novo, Comput. Methods Appl.
// Mech. Engrg. 273 (2014) 219-237: exponential boundary layer at x = 1 and
// parabolic layers at y = 0, y = 1.
//   u_t - eps Laplace(u) + (1,0).grad(u) = 1,  u = 0 on the boundary, u_0 = 0,
//   eps = 1e-4 (from 'diffusion_coefficient').
// No exact solution is known; Exact returns zero (used only for output).
// coeff[0] = eps, coeff[1..2] = b, coeff[3] = c, coeff[4] = f
// =============================================================================
#ifndef __FGJN14_EXAMPLE3__
#define __FGJN14_EXAMPLE3__

double diffusion_coefficient = 1e-4;
constexpr bool rhs_depends_on_time = false;
constexpr bool coefficients_depend_on_time = false;

void ExampleFile()
{
  Output::print("Example: FGJN14_Example3.h (de Frutos et al. 2014, Example 3: "
                "exponential and parabolic boundary layers), eps = ",
                diffusion_coefficient);
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

void BoundValue(int, double, double &value)
{
  value = 0.;
}

void InitialCondition(double, double, double *values)
{
  values[0] = 0.;
}

void BilinearCoeffs(int n_points, const double *, const double *,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    double *coeff = coeffs[i];
    coeff[0] = diffusion_coefficient;
    coeff[1] = 1.;
    coeff[2] = 0.;
    coeff[3] = 0.;
    coeff[4] = 1.;
    coeff[5] = 1.; // right-hand side at the previous time (time-independent)
  }
}

#endif
