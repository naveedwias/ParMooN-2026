// ======================================================================
// example 5.1 with anisotropic diffusion from
// X. Li and W. Huang. An anisotropic mesh adaptation method for the finite
// element solution of heterogeneous anisotropic diffusion problems.
// J. Comput. Phys., 229(21):8072–8094, 2010.
// 
// Used with mesh anisotropic_diffusion_li_huang2010.PRM,
// anisotropic_diffusion_li_huang2010_unstructured_2428.mesh
// ======================================================================


const double tol = 1.0e-6;
double k1, k2, angle;
double D11, D12, D22;
bool variableAngle;


void computeDs(double k1, double k2, double angle, double& D11Loc, double& D12Loc, double& D22Loc)
{
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  
  D11Loc = k1*c*c + k2*s*s;
  D12Loc = k1*s*c - k2*s*c;
  D22Loc = k1*s*s + k2*c*c;
}


double computeAngle(double x, double y)
{
  return M_PI * std::sin(x) * std::cos(y);
}


void ExampleFile()
{
  Output::print<1>("Example: anisotropic_diffusion_li_huang2010.h");
}


auto& Exact = unknown_solution_2d;


void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void transform(const int BdComp, const double Param, double& x, double& y)
{
  const double n49 = 4.0 / 9.0;
  const double n59 = 5.0 / 9.0;
  const double n19 = 1.0 / 9.0;
  
  switch(BdComp)
  {
    case 0:
      x = Param;
      y = 0.0;
      break;
    case 1:
      x = 1.0;
      y = Param;
      break;
    case 2:
      x = 1.0 - Param;
      y = 1.0;
      break;
    case 3:
      x = 0.0;
      y = 1.0 - Param;
      break;
    case 4:
      x = n59 - n19 * Param;
      y = n49;
      break;
    case 5:
      x = n49;
      y = n49 + n19 * Param;
      break;
    case 6:
      x = n49 + n19 * Param;
      y = n59;
      break;
    case 7:
      x = n59;
      y = n59 - n19 * Param;
      break;
    default:
      ErrThrow("wrong boundary part number", BdComp);
      break;
  }
}


void BoundValue(int BdComp, double, double &value)
{
  if(BdComp <= 3)
    value = 0.0;
  else // 4 <= BdComp <= 7
    value = 2.0;
}

void BilinearCoeffs(int n_points, const double* x, const double* y,
                    const double *const*, double **coeffs)
{
  const double bx = 0.0;
  const double by = 0.0;
  const double c = 0.0;
  const double f = 0.0;
  
  for(int i = 0; i < n_points; i++)
  {
    if(variableAngle)
    {
      angle = computeAngle(x[i], y[i]);
      computeDs(k1, k2, angle, D11, D12, D22);
    }
    
    coeffs[i][0] = D11;
    coeffs[i][1] = bx;
    coeffs[i][2] = by;
    coeffs[i][3] = c;
    coeffs[i][4] = f;
    coeffs[i][5] = D12;
    coeffs[i][6] = D22;
  }
}

