// ======================================================================
// A testing example designed so that it needs the correct assembly
// of all the parts of the equation and gives a solution
// the correctness of which is very easy to check
// ======================================================================

double diffCoeffMult;

void ExampleFile()
{
  Output::print<1>("Example: rectangle_x_plus_y_neumann.h");
}

// exact solution
void Exact(double x, double y, double *values)
{
  values[0] = x + y;
  values[1] = 1.0;
  values[2] = 1.0;
  values[3] = 0.0;
}

// double diffusionCoefficient(double x, double y)
double diffusionCoefficient(double , double )
{
  return diffCoeffMult;
}

// double diffusionCoefficient(double x, double y)
double convection_x(double , double y)
{
  return 2.0*y + 1.0;
}

// double diffusionCoefficient(double x, double y)
double convection_y(double x, double )
{
  return - 2.0*x - 1.0;
}

double reaction(double x, double y)
{
  return x + y;
}

// kind of boundary condition (for FE space needed)
void BoundCondition(int BdComp, double, BoundCond &cond)
{
  if(BdComp == 0 || BdComp == 1)
    cond = NEUMANN;
  else
    cond = DIRICHLET;
}

void transform(const int BdComp, const double Param, double& x, double& y, 
               double& nx, double& ny)
{
  switch(BdComp)
  {
    case 0:
      x = Param;
      y = 0.;
      nx = 0.;
      ny = -1.;
      break;
    case 1:
      x = 1.;
      y = Param;
      nx = 1.;
      ny = 0.;
      break;
    case 2:
      x = 1. - Param;
      y = 1.;
      nx = 0.;
      ny = 1.;
      break;
    case 3:
      x = 0.;
      y = 1. - Param;
      nx = -1.;
      ny = 0.;
      break;
    default:
      ErrThrow("wrong boundary part number", BdComp);
      break;
  }
}

// value of boundary condition
void BoundValue(int BdComp, double Param, double &value)
{
  // find out boundary condition at the evaluation point on the boundary
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  
  // find coordinates and normal of evaluation point on the boundary
  double x, y, nx, ny;
  transform(BdComp, Param, x, y, nx, ny);
  
  double exact[4];
  Exact(x, y, exact);
  
  if(cond == NEUMANN)
  {
    value = diffusionCoefficient(x, y) * (nx * exact[1] + ny * exact[2]);
  }
  else // Dirichlet
    value = exact[0];
}

void BilinearCoeffs(int n_points, const double *X, const double *Y,
                    const double *const*, double **coeffs)
{
  double exact[4];
  
  for(int i = 0; i < n_points; i++)
  {
    double x = X[i];
    double y = Y[i];
    
    coeffs[i][0] = diffusionCoefficient(x, y);
    coeffs[i][1] = convection_x(x, y);
    coeffs[i][2] = convection_y(x, y);
    coeffs[i][3] = reaction(x, y);
    
    Exact(x, y, exact);
    
    double diffTerm = - coeffs[i][0] * exact[3];
    double convTerm = coeffs[i][1] * exact[1] + coeffs[i][2] * exact[2];
    double reactionTerm = coeffs[i][3] * exact[0];
    
    coeffs[i][4] = diffTerm + convTerm + reactionTerm;
  }
}

