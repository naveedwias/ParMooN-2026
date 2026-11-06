// =============================================================================
// Example 4 of de Frutos, Garcia-Archilla, John, Novo, Comput. Methods Appl.
// Mech. Engrg. 273 (2014) 219-237: rotating three body problem (LeVeque 1996),
// clockwise rotation b = (y - 0.5, 0.5 - x), f = 0, c = 0, homogeneous
// Dirichlet boundary conditions, eps = 1e-6 (from 'diffusion_coefficient'),
// final time 2 pi (one rotation). The initial condition consists of a slotted
// cylinder centred at (0.5, 0.75), a hump at (0.25, 0.5) and a cone at
// (0.5, 0.25), all with radius r_0 = 0.15.
// Exact returns the initial condition rotated to time t (eps = 0).
// =============================================================================
#ifndef __FGJN14_EXAMPLE4__
#define __FGJN14_EXAMPLE4__

double diffusion_coefficient = 1e-6;
constexpr bool rhs_depends_on_time = false;
constexpr bool coefficients_depend_on_time = false;

void ExampleFile()
{
  Output::print("Example: FGJN14_Example4.h (de Frutos et al. 2014, Example 4: "
                "rotating three bodies, clockwise), eps = ", diffusion_coefficient);
}

double three_bodies(double x, double y)
{
  const double r0 = 0.15;
  double value = 0.;
  // slotted cylinder
  double r = std::sqrt((x-0.5)*(x-0.5) + (y-0.75)*(y-0.75))/r0;
  if(r <= 1. && (std::abs(x-0.5) >= 0.0225 || y >= 0.85))
    value = 1.;
  // hump
  r = std::sqrt((x-0.25)*(x-0.25) + (y-0.5)*(y-0.5))/r0;
  if(r <= 1.)
    value = 0.25*(1. + std::cos(M_PI*std::min(r, 1.)));
  // cone
  r = std::sqrt((x-0.5)*(x-0.5) + (y-0.25)*(y-0.25))/r0;
  if(r <= 1.)
    value = 1. - r;
  return value;
}

void Exact(double x, double y, double *values)
{
  // rotate (x,y) back by the angle t around (0.5,0.5): the flow
  // b = (y-0.5, 0.5-x) rotates clockwise, i.e. the solution at time t at
  // (x,y) equals the initial condition at the point rotated counterclockwise
  double t = TDatabase::TimeDB->CURRENTTIME;
  double dx = x - 0.5, dy = y - 0.5;
  double x0 = 0.5 + std::cos(t)*dx - std::sin(t)*dy;
  double y0 = 0.5 + std::sin(t)*dx + std::cos(t)*dy;
  values[0] = three_bodies(x0, y0);
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

void InitialCondition(double x, double y, double *values)
{
  values[0] = three_bodies(x, y);
}

void BilinearCoeffs(int n_points, const double *X, const double *Y,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    double *coeff = coeffs[i];
    coeff[0] = diffusion_coefficient;
    coeff[1] = Y[i] - 0.5;
    coeff[2] = 0.5 - X[i];
    coeff[3] = 0.;
    coeff[4] = 0.;
    coeff[5] = 0.;
  }
}

#endif
