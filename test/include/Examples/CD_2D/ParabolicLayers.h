// ======================================================================
// smooth solution problem
// ======================================================================
#define __PARABOLIC_LAYERS__
#define __STORE_ANALYTIC__

double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: PrabolicLayers.h");
}

// exact solution
void Exact(double x, double /*y*/, double *values)
{
    values[0] = x;
    values[1] = 1;
    values[2] = 0;
    values[3] = 0;
}

// kind of boundary condition (for FE space needed)
void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

// value of boundary condition
void BoundValue(int, double, double &value)
{
  value = 0;
}

void BilinearCoeffs(int n_points, const double* /*X*/, const double* /*Y*/,
                    const double *const*, double **coeffs)
{
  double eps=DIFFUSION;
  double *coeff;

  for(int i=0;i<n_points;i++)
  {
    coeff = coeffs[i];
    //param = parameters[i];

    coeff[0] = eps;
    coeff[1] = 1; //x-coordiante of convection b
    coeff[2] = 0; //y-coordiante of convection b
    coeff[3] = 0; //reaction coeffecient c
    coeff[4] = 1;
  }
}


/*******************************************************************************/
// computes the data for the evaluation of the solution 
/*******************************************************************************/
void paraboliclayers_postprocessing(ConvectionDiffusion<2> & cd2d)
{
  auto & uh = cd2d.get_function();

  // Compute (an approximation of) the global minimal and maximal value of uh
  uh.PrintMinMax("solution", false);
  // Compute the global minimal and maximal value of the FE-vector that is
  // cell wise transformed to Pk elements of the same order (if not Pk elements
  // are used already)
  uh.PrintMinMax("Pk solution vector", true);

  // Compute mean oscillation using cell wise minimum and maximum of uh
  constexpr double umin = 0.;
  constexpr double umax = 1.;
  double osc_mean = uh.compute_mean_oscillation(umin, umax, false);
  Output::print("mean oscillation ", std::setprecision(12), osc_mean);
  // Compute mean oscillation using cell wise minimum and maximum of the
  // FE-vector transformed to Pk elements of the same order
  osc_mean = uh.compute_mean_oscillation(umin, umax, true);
  Output::print("mean oscillation using Pk nodal functionals ",
      std::setprecision(12), osc_mean);
  // Compute mean oscillations of the FE-vector itself
  if (!cd2d.get_space()->is_discontinuous())
  {
    auto& entries = cd2d.get_solution().get_entries_vector();
    double osc_mean_vec = 0.;
    for(double e : entries)
    {
      osc_mean_vec += std::max(0., e-umax) + std::max(0., umin-e);
    }
    osc_mean_vec /= entries.size();
    Output::print("mean vector oscillation ", std::setprecision(12),
        osc_mean_vec);
  }
}
