// ======================================================================
// Example 3 from
// V. John and P. Knobloch. On spurious oscillations at layers diminishing
// (SOLD) methods for convection-diffusion equations. II. Analysis for P 1 and
// Q 1 finite elements.
// Comput. Methods Appl. Mech. Engrg., 197(21-24):1997–2014, 2008.
//
// Used with mesh UnitSquareEightTriangles.GEO
// ======================================================================


double DIFFUSION;

void ExampleFile()
{
  Output::print<1>("Example: quadratic_john_knobloch2008.h");
  TDatabase::ParamDB->INTERNAL_QUAD_RULE = 99;
}

bool isInSquare(double x, double y)
{
  if(0.25 <= x && x <= 0.75 && 0.25 <= y && y <= 0.75)
    return true;
  else
    return false;
}

// not exactly the solution, but it should be close
void Exact(double x, double y, double *values)
{
  if(isInSquare(x, y))
  {
    values[0] = - 16.0 * x * x + 16.0 * x - 3.0;
    values[1] = - 32.0 * x + 16.0;
    values[2] = 0.0;
    values[3] = - 32.0;
  }
  else
  {
    values[0] = 0.0;
    values[1] = 0.0;
    values[2] = 0.0;
    values[3] = 0.0;
  }
}

void BoundCondition(int, double, BoundCond &cond)
{
  cond = DIRICHLET;
}

void transform(const int BdComp, const double Param, double& x, double& y)
{
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
    default:
      ErrThrow("wrong boundary part number", BdComp);
      break;
  }
}

void BoundValue(int, double, double &value)
{
  value = 0.0;
}

void BilinearCoeffs(int n_points, const double* x, const double* y,
                    const double *const*, double **coeffs)
{
  for(int i = 0; i < n_points; i++)
  {
    double fTerm = 0.0;
    
    if(isInSquare(x[i], y[i]))
      fTerm = 16.0 * (1.0 - 2.0 * x[i]);
    
    coeffs[i][0] = DIFFUSION; // \epsilon
    coeffs[i][1] = 1.0; // bx
    coeffs[i][2] = 0.0; // by
    coeffs[i][3] = 0.0; // c
    coeffs[i][4] = fTerm; // f
  }
}

bool isIn0406Stripe(double x)
{
  if(0.4 <= x && x <= 0.6)
    return true;
  else
    return false;
}

bool isBeyoned08(double x)
{
  if(0.8 <= x)
    return true;
  else
    return false;
}

void postprocessing(ConvectionDiffusion<2>& cd2d)
{
  // Computes values of min and diff in the above paper.
  // This algorithm is described in that paper.
  
  double min0406 = 1.0e10;
  double max08 = -1.0e10;
  double min08 = 1.0e10;
  
  const TCollection* collPtr = cd2d.get_space()->GetCollection();
  const BlockVector& sol = cd2d.get_solution();
  
  std::pair<double, double> minmax;
  std::vector<double> bf_values;
  
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    TBaseCell* cellPtr = collPtr->GetCell(cellNr);
    
    const int *globalDOF = cd2d.get_space()->GetGlobalDOF(cellNr);
    
    bool min0406Done = false;
    bool max08Done = false;
    
    for(int vertexId = 0; vertexId < cellPtr->GetN_Vertices(); vertexId++)
    {
      if(min0406Done && max08Done)
        break;
      
      const TVertex* vertexPtr = cellPtr->GetVertex(vertexId);
      
      double xCoord, yCoord;
      vertexPtr->GetCoords(xCoord, yCoord);
      
      if(!min0406Done && isIn0406Stripe(xCoord))
      {
        double solVal = sol[globalDOF[vertexId]];
        
        if(solVal < min0406)
          min0406 = solVal;
        
        min0406Done = true;
      }
      
      if(!max08Done && isBeyoned08(xCoord))
      {
        double solVal = sol[globalDOF[vertexId]];
        
        if(solVal > max08)
          max08 = solVal;
        
        if(solVal < min08)
          min08 = solVal;
        
        max08Done = true;
      }
    }
  }
  
  double min = - min0406;
  double diff = max08 - min08;
  
  Output::print("Postprocessing: min = ", std::setprecision(16), min);
  Output::print("Postprocessing: diff = ", std::setprecision(16), diff);
}

