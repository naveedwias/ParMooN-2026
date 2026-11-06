// ======================================================================
// Example for birthday paper by John, Knobloch, Partl
// ======================================================================


double DIFFUSION;
double alpha;
double beta;
double gamma;


void ExampleFile()
{
  Output::print<1>("Example: species_in_cavity.h");
}


auto& Exact = unknown_solution_2d;


void transform(const int BdComp, const double param, double& x, double& y, 
               double& nx, double& ny)
{
  switch(BdComp)
  {
    case 0:
      x = param;
      y = 0.;
      nx = 0.;
      ny = -1.;
      break;
    case 1:
      x = 1.;
      y = param;
      nx = 1.;
      ny = 0.;
      break;
    case 2:
      x = 1. - param;
      y = 1.;
      nx = 0.;
      ny = 1.;
      break;
    case 3:
      x = 0.;
      y = 1. - param;
      nx = -1.;
      ny = 0.;
      break;
    default:
      ErrThrow("wrong boundary part number", BdComp);
      break;
  }
}


void BoundCondition(int BdComp, double param, BoundCond &cond)
{
  const double tol = 1.0e-6;
  
  double x, y, nx, ny;
  
  transform(BdComp, param, x, y, nx, ny);
  
  if(std::abs(x) < tol && 0.25 - tol <= y && y <= 0.375 + tol)
    cond = DIRICHLET;
  else
    cond = NEUMANN;
}


// value of boundary condition
void BoundValue(int BdComp, double param, double &value)
{
  const double tol = 1.0e-6;
  
  BoundCond cond;
  BoundCondition(BdComp, param, cond);
  double x, y, nx, ny;
  transform(BdComp, param, x, y, nx, ny);
  
  if(std::abs(x) < tol && 0.25 - tol <= y && y <= 0.375 + tol)
    value = 1.0;
  else
    value = 0.0;
}


double get_s(double x, double y)
{
  const double b2Mult = 1.0 / std::sqrt(17.0);
  const double b1Mult = 4.0 * b2Mult;
  
  return std::abs( - b2Mult * x + b1Mult * (y - 5.0 / 16.0) );
}


double psiFunction(double alphaLoc, double s)
{
  const double s0 = 0.25 / std::sqrt(17.0);
  double ex;
  
  if(s <= s0)
    return 1.0;
  else
  {
    ex = std::exp( - alphaLoc * (s0 - s)* (s0 - s) );
    return ex;
  }
}


void BilinearCoeffs(int n_points, const double *x, const double *y,
                    const double *const*, double **coeffs)
{
  const double b2Mult = 1.0 / std::sqrt(17.0);
  const double b1Mult = 4.0 * b2Mult;
  
  for(int i = 0; i < n_points; i++)
  {
    double s = get_s(x[i], y[i]);
    double psi = psiFunction(alpha, s);
    
    coeffs[i][0] = DIFFUSION;
    coeffs[i][1] = b1Mult * psi;
    coeffs[i][2] = b2Mult * psi;
    coeffs[i][3] = beta * psi;
    coeffs[i][4] = gamma * psi;
  }
}


double integralInStripe(double yLower, double yUpper, const TFEFunction2D& fct)
{
  const TCollection* collectionPtr = fct.GetFESpace()->GetCollection();
  
  parmoon::Point point(0.0, 0.0);
  double measureOfCell, integralOverCell;
  
  double integral = 0.0;
  
  for(int cellNr = 0; cellNr < collectionPtr->GetN_Cells(); cellNr++)
  {
    const TBaseCell* cellPtr = collectionPtr->GetCell(cellNr);
    point = cellPtr->getCentroid();
    
    if(yLower <= point.y && point.y <= yUpper)
    {
      fct.compute_integral_and_measure(cellNr, integralOverCell, measureOfCell);
      integral += integralOverCell;
    }
  }
  
  return integral;
}


double integralOverLine(const TFEFunction2D& fct)
{
  const double b2Mult = 1.0 / std::sqrt(17.0);
  const double b1Mult = 4.0 * b2Mult;
  const double directionVect_x = b2Mult;
  const double directionVect_y = - b1Mult;
  // the norm of the derivative is 1.0 => multiplication by this term left out in what follows
  
  const double xStart = 0.0;
  const double yStart = 1.0;
  
  const int NPoints = 100001;
  const double tauStart = 0.0;
  const double tauEnd = 1.0 / b1Mult;
  const double width = tauEnd - tauStart;
  const double step = width / ( NPoints - 1.0 );
  
  int currentCellNr = 0;
  const TCollection* collectionPtr = fct.GetFESpace()->GetCollection();
  
  double integral = 0.0;
  double value;

  
  // i = 0 => first point
  parmoon::Point point(xStart, yStart);
  collectionPtr->find_cell(point, currentCellNr);
  TBaseCell* currentCellPtr = collectionPtr->GetCell(currentCellNr);
  
  fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
  integral += 0.5 * step * value * value;
  
  // all points between first and last
  for(int i = 1; i < NPoints - 1; i++)
  {
    double tau = tauStart + step * i;
    point.x = xStart + tau * directionVect_x;
    point.y = yStart + tau * directionVect_y;
    collectionPtr->find_cell(point, currentCellNr);
    currentCellPtr = collectionPtr->GetCell(currentCellNr);
    
    fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
    integral += step * value * value;
  }
  
  // i = NPoints - 1 => last point
  point.x = xStart + tauEnd * directionVect_x;
  point.y = yStart + tauEnd * directionVect_y;
  collectionPtr->find_cell(point, currentCellNr);
  currentCellPtr = collectionPtr->GetCell(currentCellNr);
    
  fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
  integral += 0.5 * step * value * value;
  
  return integral;
}


double errorOverLine(const TFEFunction2D& fct)
{
  std::string fileName = "reference_solution_over_line.txt";
  std::ifstream dataFile;
  dataFile.open(fileName, std::ifstream::in);
  
  std::string someString;
  std::getline(dataFile, someString);// get first line with description that is skipped
  
  
  const double b2Mult = 1.0 / std::sqrt(17.0);
  const double b1Mult = 4.0 * b2Mult;
  const double directionVect_x = b2Mult;
  const double directionVect_y = - b1Mult;
  // the norm of the derivative is 1.0 => multiplication by this term left out in what follows
  
  const double xStart = 0.0;
  const double yStart = 1.0;
  
  const int NPoints = 100001;
  const double tauStart = 0.0;
  const double tauEnd = 1.0 / b1Mult;
  const double width = tauEnd - tauStart;
  const double step = width / ( NPoints - 1.0 );
  
  int currentCellNr = 0;
  const TCollection* collectionPtr = fct.GetFESpace()->GetCollection();
  
  double integral = 0.0;
  double value;
  double valueRef;
  double tauFromFile;
  double diff;

  
  // i = 0 => first point
  parmoon::Point point(xStart, yStart);
  collectionPtr->find_cell(point, currentCellNr);
  TBaseCell* currentCellPtr = collectionPtr->GetCell(currentCellNr);
  
  fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
  dataFile >> tauFromFile >> valueRef;
  diff = value - valueRef;
  
  integral += 0.5 * step * diff * diff;
  
  // all points between first and last
  for(int i = 1; i < NPoints - 1; i++)
  {
    double tau = tauStart + step * i;
    point.x = xStart + tau * directionVect_x;
    point.y = yStart + tau * directionVect_y;
    collectionPtr->find_cell(point, currentCellNr);
    currentCellPtr = collectionPtr->GetCell(currentCellNr);
    
    fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
    dataFile >> tauFromFile >> valueRef;
    diff = value - valueRef;
  
    integral += step * diff * diff;
  }
  
  // i = NPoints - 1 => last point
  point.x = xStart + tauEnd * directionVect_x;
  point.y = yStart + tauEnd * directionVect_y;
  collectionPtr->find_cell(point, currentCellNr);
  currentCellPtr = collectionPtr->GetCell(currentCellNr);
    
  fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
  dataFile >> tauFromFile >> valueRef;
  diff = value - valueRef;
  
  integral += 0.5 * step * diff * diff;
  
  dataFile.close();
  
  return integral;
}


void plotOverLine(const TFEFunction2D& fct)
{
  std::string fileName = "reference_solution_over_line.txt";
  const int NPoints = 100001;
  
  const double b2Mult = 1.0 / std::sqrt(17.0);
  const double b1Mult = 4.0 * b2Mult;
  const double directionVect_x = b2Mult;
  const double directionVect_y = - b1Mult;
  // the norm of the derivative is 1.0 => multiplication by this term left out in what follows
  
  const double xStart = 0.0;
  const double yStart = 1.0;
  
  const double tauStart = 0.0;
  const double tauEnd = 1.0 / b1Mult;
  const double width = tauEnd - tauStart;
  const double step = width / ( NPoints - 1.0 );
 
  
  int currentCellNr = 0;
  const TCollection* collectionPtr = fct.GetFESpace()->GetCollection();
  
  double value;
  double tau;

  std::ofstream outputFile;
  outputFile.open(fileName, std::ofstream::out);
  outputFile << "tau solution" << std::setprecision(12) << std::endl;
  
  parmoon::Point point(xStart, yStart);
  collectionPtr->find_cell(point, currentCellNr);
  TBaseCell* currentCellPtr = collectionPtr->GetCell(currentCellNr);
  
  for(int i = 0; i < NPoints; i++)
  {
    tau = tauStart + step * i;
    point.x = xStart + tau * directionVect_x;
    point.y = yStart + tau * directionVect_y;
    collectionPtr->find_cell(point, currentCellNr);
    currentCellPtr = collectionPtr->GetCell(currentCellNr);
    
    fct.FindValueLocal(currentCellPtr, currentCellNr, point.x, point.y, &value);
    
    outputFile << tau << " " << value << std::endl;
  }
  
  outputFile.close();
  
  Output::print("Reference solution over line written into file ", fileName);
}


void postprocessing(ConvectionDiffusion<2>& cd2d)
{
  const TFEFunction2D& fct = cd2d.get_function();
  
//   plotOverLine(fct);
  
  const double l2Error = std::sqrt( errorOverLine(fct) );
//   const double l2Norm = std::sqrt( integralOverLine(fct) );
  
  Output::print("l2 error of solution along line: ", std::setprecision(12), l2Error);
//   
//   const double integral = integralInStripe(0.8, 1.0, fct);
//   Output::print("Integral of solution in stripe from y = 0.8 to y = 1: ",
//                 std::setprecision(12), integral);
}

