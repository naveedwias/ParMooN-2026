// Navier-Stokes problem with sine and cosine functions
// 
#include "Constants.h"
#include "MooNMD_Io.h"

double pressure_factor;


void ExampleFile()
{
  Output::print<1>("Example: 2d (Navier-)Stokes SinCos.h.");
}

// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double, double *values)
{
  values[0] = std::sin(M_PI*x);
  values[1] = M_PI*std::cos(M_PI*x);
  values[2] = 0;
  values[3] = -M_PI*M_PI*std::sin(M_PI*x);
}

void ExactU2(double x, double y, double *values)
{
  values[0] = -M_PI*y*std::cos(M_PI*x);
  values[1] = M_PI*M_PI*y*std::sin(M_PI*x);
  values[2] = -M_PI*std::cos(M_PI*x);
  values[3] = M_PI*M_PI*M_PI*y*std::cos(M_PI*x);
}

void ExactP(double x, double y, double *values)
{
  values[0] = pressure_factor * std::sin(M_PI*x)*std::cos(M_PI*y);
  values[1] =  pressure_factor * M_PI*std::cos(M_PI*x)*std::cos(M_PI*y);
  values[2] = -M_PI * pressure_factor * std::sin(M_PI*x)*std::sin(M_PI*y);
  values[3] = -M_PI*M_PI * pressure_factor * std::sin(M_PI*x)*std::cos(M_PI*y)-M_PI*M_PI*std::sin(M_PI*x)*std::cos(M_PI*y);
}

void InitialU1(double x, double, double *values)
{
  values[0] = std::sin(M_PI*x);
}

void InitialU2(double x, double y, double *values)
{
  values[0] = -M_PI*y*std::cos(M_PI*x);
}

void InitialP(double x, double y, double *values)
{
  values[0] = pressure_factor * std::sin(M_PI*x)*std::cos(M_PI*y);
}

// ========================================================================
// boundary conditions
// ========================================================================
void BoundCondition(int, double, BoundCond &cond)
{
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

void U1BoundValue(int BdComp, double Param, double &value)
{
  // Neumann boundary means setting T.n where n is outer normal and T is either
  // 2nu D(u) - p  (for LAPLACETYPE==1) or nu grad(u) - p   (for LAPLACETYPE==0)
  const int lt = TDatabase::ParamDB->LAPLACETYPE;
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  // find out boundary condition at the evaluation point on the boundary
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  // find coordinates and normal of evaluation point on the boundary
  double x, y, nx, ny;
  transform(BdComp, Param, x, y, nx, ny);
  // evaluate the exact solution at the given point
  double u1[4];
  double u2[4];
  double p[4];
  ExactU1(x, y, u1);
  ExactU2(x, y, u2);
  ExactP(x, y, p);
  if(cond == DIRICHLET)
    value = u1[0];
  else
  {
    // NEUMANN
    value = nu * (nx * u1[1] + ny * u1[2]) - p[0] * nx;
    if(lt == 1)
    {
      value *= 0.5;
      value += 0.5 * nu * (nx * u1[1] + ny * u2[1]);
    }
  }
  return;
}

void U2BoundValue(int BdComp, double Param, double &value)
{
  // Neumann boundary means setting T.n where n is outer normal and T is either
  // 2nu D(u) - p  (for LAPLACETYPE==1) or nu grad(u) - p   (for LAPLACETYPE==0)
  const int lt = TDatabase::ParamDB->LAPLACETYPE;
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  // find out boundary condition at the evaluation point on the boundary
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  // find coordinates and normal of evaluation point on the boundary
  double x, y, nx, ny;
  transform(BdComp, Param, x, y, nx, ny);
  // evaluate the exact solution at the given point
  double u1[4];
  double u2[4];
  double p[4];
  ExactU1(x, y, u1);
  ExactU2(x, y, u2);
  ExactP(x, y, p);
  if(cond == DIRICHLET)
    value = u2[0];
  else
  {
    // NEUMANN
    value = nu * (nx * u2[1] + ny * u2[2]) - p[0] * ny;
    if(lt == 1)
    {
      value *= 0.5;
      value += 0.5 * nu * (nx * u1[2] + ny * u2[2]);
    }
  }
  return;
}

// u times normal vector
void UNormalValue(int BdComp, double Param, double &value)
{
  // Neumann boundary means setting T.n where n is outer normal and T is either
  // 2nu D(u) - p  (for LAPLACETYPE==1) or nu grad(u) - p   (for LAPLACETYPE==0)
  const int lt = TDatabase::ParamDB->LAPLACETYPE;
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  // find out boundary condition at the evaluation point on the boundary
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  // find coordinates and normal of evaluation point on the boundary
  double x, y, nx, ny;
  transform(BdComp, Param, x, y, nx, ny);
  // evaluate the exact solution at the given point
  double u1[4];
  double u2[4];
  double p[4];
  ExactU1(x, y, u1);
  ExactU2(x, y, u2);
  ExactP(x, y, p);
  
  if(cond == DIRICHLET)
    value = nx * u1[0] + ny * u2[0];
  else
  {
    // NEUMANN
    value = nu * (nx * (nx*u1[1] + ny*u1[2]) + ny * (nx*u2[1] + ny*u2[2])) - p[0];
    if(lt == 1)
    {
      ErrThrow("LAPLACETYPE=1 not yet supported");
      //value *= 0.5;
      //value += 0.5 * nu * (nx * u1[1] + ny * u2[1]);
    }
  }
  return;
}

// u times tangential vector
void UTangValue(int BdComp, double Param, double &value)
{
  // Neumann boundary means setting T.n where n is outer normal and T is either
  // 2nu D(u) - p  (for LAPLACETYPE==1) or nu grad(u) - p   (for LAPLACETYPE==0)
  const int lt = TDatabase::ParamDB->LAPLACETYPE;
  const double nu = 1./TDatabase::ParamDB->RE_NR;
  // find out boundary condition at the evaluation point on the boundary
  BoundCond cond;
  BoundCondition(BdComp, Param, cond);
  // find coordinates and normal of evaluation point on the boundary
  double x, y, nx, ny;
  transform(BdComp, Param, x, y, nx, ny);
  // evaluate the exact solution at the given point
  double u1[4];
  double u2[4];
  double p[4];
  ExactU1(x, y, u1);
  ExactU2(x, y, u2);
  ExactP(x, y, p);
  double tx = -ny;
  double ty = nx;
  
  if(cond == DIRICHLET)
    value = tx * u1[0] + ty * u2[0];
  else
  {
    // NEUMANN
    value = nu * (tx * (nx*u1[1] + ny*u1[2]) + ty * (nx*u2[1] + ny*u2[2]));
    if(lt == 1)
    {
      ErrThrow("LAPLACETYPE=1 not yet supported");
      //value *= 0.5;
      //value += 0.5 * nu * (nx * u1[2] + ny * u2[2]);
    }
  }
  return;
}

// ========================================================================
// coefficients for Stokes form: A, B1, B2, f1, f2
// ========================================================================
void LinCoeffs(int n_points, const double *X, const double *Y,
               const double *const*, double **coeffs)
{
  const double nu=1./TDatabase::ParamDB->RE_NR;
  double val1[4];
  double val2[4];
  double val3[4];
  for(int i = 0; i < n_points; i++)
  {
    coeffs[i][0] = nu;
    
    ExactU1(X[i], Y[i], val1);
    ExactU2(X[i], Y[i], val2);
    ExactP(X[i], Y[i], val3);
    
    coeffs[i][1] = -nu*val1[3] + val3[1]; // f1
    coeffs[i][2] = -nu*val2[3] + val3[2]; // f2
    
    if(TDatabase::ParamDB->FLOW_PROBLEM_TYPE == 5) // Navier-Stokes (3 means Stokes)
    {
      coeffs[i][1] += val1[0]*val1[1] + val2[0]*val1[2]; // f1
      coeffs[i][2] += val1[0]*val2[1] + val2[0]*val2[2]; // f2
    }
    coeffs[i][3] = val1[1] + val2[2]; // g (divergence)

    // additional coefficient (used only in the Brinkman problem)
    coeffs[i][4] = 0.;
  }
  
}
