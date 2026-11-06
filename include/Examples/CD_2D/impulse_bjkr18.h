// ======================================================================
// Problem with transport of impulse, proposed in
// G. R. Barrenechea, V. John, P. Knobloch, and R. Rankin.
// A unified analysis of algebraic flux correction schemes
// for convection-diffusion equations. SeMA J., 75(4):655–685, 2018.
// Section 6.2
//
// Used with mesh impulse_bjkr18_all_levels.PRM, impulse_bjkr18_level0.mesh
// ======================================================================


double DIFFUSION;

const double tol = 1.0e-6;

void ExampleFile()
{
  Output::print<1>("Example: impulse_bjkr18.h");
}


// not exactly the solution, but it should be close
void Exact(double , double y, double *values)
{
  if( 0.375 - tol < y && y < 0.625 + tol )
    values[0] = 1.0;
  else
    values[0] = 0.0;
  
  values[1] = 0.0;
  values[2] = 0.0;
  values[3] = 0.0;
}


void BoundCondition(int BdComp, double, BoundCond &cond)
{
  if(BdComp == 1) // edge x = 10
    cond = NEUMANN;
  else
    cond = DIRICHLET;
}

void transform(const int BdComp, const double Param, double& x, double& y)
{
  switch(BdComp)
  {
    case 0:
      x = 10.0 * Param;
      y = 0.0;
      break;
    case 1:
      x = 10.0;
      y = Param;
      break;
    case 2:
      x = 10.0 * (1.0 - Param);
      y = 1.0;
      break;
    case 3:
      x = 0.0;
      y = 1.0 - Param;
      break;
    default:
      ErrThrow("wrong boundary part number", BdComp);
      break;
  }
}


void BoundValue(int BdComp, double Param, double &value)
{
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  
  if(cond == DIRICHLET)
  {
    double x, y;
    transform(BdComp, Param, x, y);
  
    if( std::abs(y - 0.0) < tol || std::abs(y - 1.0) < tol )
      value = 0.0;
    else if( std::abs(x - 0.0) < tol )
    {
      if( 0.375 - tol < y && y < 0.625 + tol )
        value = 1.0;
      else
        value = 0.0;
    }
  }
  else // neumann
    value = 0.0;
}

void BilinearCoeffs(int n_points, const double* , const double* ,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = DIFFUSION; // \epsilon
    coeffs[i][1] = 1.0; // bx
    coeffs[i][2] = 0.0; // by
    coeffs[i][3] = 0.0; // c
    coeffs[i][4] = 0.0; // g
  }
}

