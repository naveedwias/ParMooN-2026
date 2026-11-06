#include "MizukamiHughes.h"
#include <MainUtilities.h>
#include "QuadFormula.h"
#include "FEDatabase.h"
#include "Matrix.h"
#include <cmath>
#include <algorithm>



MizukamiHughes::MizukamiHughes(const ParameterDatabase& db, const Example_CD2D& passedExample,
                               const FEMatrix& sysMatrix, const BlockVector& solution)
: cCoeffsSpace(new TFESpace2D(sysMatrix.GetTestSpace2D()->GetCollection(),
                              "cCoeffsSpace", BoundConditionNoBoundCondition, 0) ),
  bxVals(cCoeffsSpace->get_n_dof()),
  byVals(cCoeffsSpace->get_n_dof()),
  reactVals(cCoeffsSpace->get_n_dof()),
  cCoeffsVals( { BlockVector(cCoeffsSpace->get_n_dof()),
                 BlockVector(cCoeffsSpace->get_n_dof()),
                 BlockVector(cCoeffsSpace->get_n_dof()) } ),
  c1CoeffsFct(cCoeffsSpace, "c1Coeffs", cCoeffsVals[0].get_entries()),
  c2CoeffsFct(cCoeffsSpace, "c2Coeffs", cCoeffsVals[1].get_entries()),
  c3CoeffsFct(cCoeffsSpace, "c3Coeffs", cCoeffsVals[2].get_entries()),
  bxFct(cCoeffsSpace, "bxCoeffs", bxVals.get_entries()),
  byFct(cCoeffsSpace, "byCoeffs", byVals.get_entries()),
  reactFct(cCoeffsSpace, "reactCoeff", reactVals.get_entries()),
  specialTriangulation(db["Mizukami_Hughes_special_triangulation"]),
  passedSol(solution),
  sysMat(sysMatrix),
  computationStage(0),
  doAssembly(true),
  ratioEpsilonTildeEpsilon(db["Mizukami_Hughes_epsilon_ratio"]),
  example(passedExample)
{
  std::string tempString = db["space_discretization_type"];
  mizukamiHughesTypeString = tempString;
  
  if(mizukamiHughesTypeString == "Mizukami_Hughes_original")
    mizukamiHughesType = MizukamiHughesType::Original;
  else if(mizukamiHughesTypeString == "Mizukami_Hughes_improved")
    mizukamiHughesType = MizukamiHughesType::Improved;
  else
    mizukamiHughesType = MizukamiHughesType::Reaction;
  
  computeCoeffsConstantOnTriangles(passedExample.get_coeffs());
}


void MizukamiHughes::setAssemblyParameters
(std::vector<const TFEFunction2D*>& feFunctionPtr, ParameterDatabase& db)
{
  if(computationStage == 0)
  {
    computationStage = 1;
    
    Parameter disc_type{db["initial_solution_cd"]};
    
    if(disc_type.is("galerkin"))
      db["space_discretization_type"] = "galerkin";
    else if(disc_type.is("upwind"))
      db["space_discretization_type"] = "upwind";
    else // zero
    {
      doAssembly = false;
      return;
    }
  }
  else
  {
    switch(mizukamiHughesType)
    {
      case MizukamiHughesType::Original:
        computeCsOriginal(passedSol);
        break;
      case MizukamiHughesType::Improved:
        computeCsImprovedTolerances(passedSol);
        break;
      case MizukamiHughesType::Reaction:
        computeCsImprovedReactionTolerances(passedSol);
    }
    
    if(computationStage == 1)
    {
      doAssembly = true;
      computationStage = 2;
      db["space_discretization_type"] = mizukamiHughesTypeString;
    }
  }
  
  feFunctionPtr.clear();
  feFunctionPtr.resize(6);
  feFunctionPtr[0] = &c1CoeffsFct;
  feFunctionPtr[1] = &c2CoeffsFct;
  feFunctionPtr[2] = &c3CoeffsFct;;
  feFunctionPtr[3] = &bxFct;
  feFunctionPtr[4] = &byFct;
  feFunctionPtr[5] = &reactFct;
}


void MizukamiHughes::computeCoeffsConstantOnTriangles(const CoeffFct2D& bilinearCoeffs)
{
  const int nParams = 5;
  double *coeffsAtBarycenter = new double[nParams];
  
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const TCollection* collPtr = feSpace2DPtr->GetCollection();
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    const TGridCell* cellPtr = (TGridCell*)collPtr->GetCell(cellNr);
    
    const parmoon::Point barycenter = cellPtr->getCentroid();
    bilinearCoeffs(1, &barycenter.x, &barycenter.y, nullptr, &coeffsAtBarycenter);
    
    const double bx = coeffsAtBarycenter[1];
    const double by = coeffsAtBarycenter[2];
    const double react = coeffsAtBarycenter[3];
    
    bxVals[cellNr] = bx;
    byVals[cellNr] = by;
    reactVals[cellNr] = react;
  }
  
  // scheme assumes constant diffusion coefficient epsilon
  epsilonTilde = ratioEpsilonTildeEpsilon * coeffsAtBarycenter[0];
  
  delete[] coeffsAtBarycenter;
}


void MizukamiHughes::getDerivativesOfPhi(const FiniteElement& finiteElement,
                                         double* phiXi, double* phiEta,
                                         const TGridCell* cellPtr, const TCollection* collPtr,
                                         double* phiRef, double* phiOrig,
                                         double* phix, double* phiy)
{
  const BaseFunctions* baseFunctionsPtr = finiteElement.GetBaseFunct();
    
  const double xiBarycenter = 1.0 / 3.0;
  const double etaBarycenter = 1.0 / 3.0;
  
  baseFunctionsPtr->GetDerivatives(MultiIndex2D::D10, xiBarycenter, etaBarycenter, phiXi);
  baseFunctionsPtr->GetDerivatives(MultiIndex2D::D01, xiBarycenter, etaBarycenter, phiEta);
  
  ReferenceTransformation_type refTrans = finiteElement.GetRefTransID();
  FEDatabase::SetCellForRefTrans(cellPtr, refTrans);
  
  // shift to have numbering starting from 1
  double *phixData = phix+1;
  double *phiyData = phiy+1;
  
  FEDatabase::GetOrigValues(refTrans, xiBarycenter, etaBarycenter, baseFunctionsPtr,
                            collPtr, cellPtr, phiRef, phiXi, phiEta, phiOrig,
                            phixData, phiyData);
  
  // correct nearly zero values of derivatives => such values should be zero
  // the tolerance does not need to be very small because the derivatives
  // increase in their magnitude when the elements are smaller
  const double toleranceForZeroDerivative = 1.0e-10;
  
  for(int i = 1; i <= 3; i++)
  {
    if(std::abs(phix[i]) < toleranceForZeroDerivative)
      phix[i] = 0.0;
    
    if(std::abs(phiy[i]) < toleranceForZeroDerivative)
      phiy[i] = 0.0;
  }
}


void MizukamiHughes::determineZone(const double* b_phi, int& zoneNr,
                                   bool& inVZ)
{
  double b_phi_product_all = b_phi[1] * b_phi[2] * b_phi[3];
    
  if(b_phi_product_all >= 0.0)
  {
    // vertex zone
    inVZ = true;
    
    for(zoneNr = 1; zoneNr <= 3; zoneNr++)
    {
      if(b_phi[zoneNr] > 0.0)
        break;
    }
  }
  else
  {
    // edge zone
    inVZ = false;
    
    for(zoneNr = 1; zoneNr <= 3; zoneNr++)
    {
      if(b_phi[zoneNr] < 0.0)
        break;
    }
  }
}


void MizukamiHughes::determineZoneTolerances(const double* b_phi,
                                             int& zoneNr, bool& inVZ)
{
  if(b_phi[1] > tolEq && b_phi[2] < tolEq && b_phi[3] < tolEq)
  {
    zoneNr = 1;
    inVZ = true;
  }
  else if(b_phi[1] < tolEq && b_phi[2] > tolEq && b_phi[3] < tolEq)
  {
    zoneNr = 2;
    inVZ = true;
  }
  else if(b_phi[1] < tolEq && b_phi[2] < tolEq && b_phi[3] > tolEq)
  {
    zoneNr = 3;
    inVZ = true;
  }
  else
  {
    // edge zone
    inVZ = false;
    
    for(zoneNr = 1; zoneNr <= 3; zoneNr++)
    {
      if(b_phi[zoneNr] < - tolEq)
        break;
    }
  }
}


void MizukamiHughes::determineLocalNumbering(int zoneNr, int* vLoc, int* vLocDers)
{
  vLoc[1] = zoneNr - 1;
  
  if(zoneNr == 1)
  {
    vLoc[2] = 1;
    vLoc[3] = 2;
  }
  else if(zoneNr == 2)
  {
    vLoc[2] = 2;
    vLoc[3] = 0;
  }
  else // zoneNr == 3
  {
    vLoc[2] = 0;
    vLoc[3] = 1;
  }
  
  for(int i = 1; i <= 3; i++)
    vLocDers[i] = vLoc[i] + 1;
}


bool MizukamiHughes::cellIntersectsDirichletBoundary(int cellNr)
{
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  
  for(int vertexNr = 0; vertexNr < 3; vertexNr++)
  {
    int globalVertexDOF = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
    
    if( globalVertexDOF >= feSpace2DPtr->get_n_active() )
    {
      return true;
    }
  }
  
  return false;
}


bool MizukamiHughes::AllNodesConnectedToDirichletNodes(int cellNr)
{
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const int* cols = sysMat.get_vector_columns();
  const int* rows = sysMat.get_row_ptr();
  
  bool foundDirichletCoupling = false;
  
  for(int vertexNr = 0; vertexNr < 3; vertexNr++)
  {
    foundDirichletCoupling = false;
    
    int row = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
    
    for(int j = rows[row]; j <rows[row+1]; j++)
    {
      if(cols[j] >= feSpace2DPtr->get_n_active())
      {
        foundDirichletCoupling = true;
        break;
      }
    }
    
    if(!foundDirichletCoupling)
      return false;
  }
  
  return true;
}


bool MizukamiHughes::OneNodeConnectedToDirichletNodes(int cellNr)
{
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const int* cols = sysMat.get_vector_columns();
  const int* rows = sysMat.get_row_ptr();
  
  for(int vertexNr = 0; vertexNr < 3; vertexNr++)
  {
    int row = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
    
    for(int j = rows[row]; j <rows[row+1]; j++)
    {
      if(cols[j] >= feSpace2DPtr->get_n_active())
        return true;
    }
  }
  
  return false;
}


void MizukamiHughes::computeVectors_v2_v3_v_w(const TGridCell* cellPtr, const int* vLoc,
                                              double solx, double soly,
                                              double& v2x, double& v2y,
                                              double& v3x, double& v3y,
                                              double& vx, double& vy,
                                              double& wx, double& wy)
{
  const TVertex* vertex1Ptr = cellPtr->GetVertex(vLoc[1]);
  const TVertex* vertex2Ptr = cellPtr->GetVertex(vLoc[2]);
  const TVertex* vertex3Ptr = cellPtr->GetVertex(vLoc[3]);
  double a1x, a1y, a2x, a2y, a3x, a3y;
  vertex1Ptr->GetCoords(a1x, a1y);
  vertex2Ptr->GetCoords(a2x, a2y);
  vertex3Ptr->GetCoords(a3x, a3y);
  
  v2x = a2x - a1x;
  v2y = a2y - a1y;
  const double v2Length = std::sqrt(v2x * v2x + v2y * v2y);
  v2x /= v2Length;
  v2y /= v2Length;
  
  v3x = a3x - a1x;
  v3y = a3y - a1y;
  const double v3Length = std::sqrt(v3x * v3x + v3y * v3y);
  v3x /= v3Length;
  v3y /= v3Length;
  
  vx = v2x + v3x;
  vy = v2y + v3y;
  const double vLength = std::sqrt(vx * vx + vy * vy);
  vx /= vLength;
  vy /= vLength;
  
  const double solLength = std::sqrt(solx * solx + soly * soly);
  
  wx = - soly / solLength;
  wy = solx / solLength;
  
  if(wx * vx + wy * vy < 0.0)
  {
    wx = - wx;
    wy = - wy;
  }
}


void MizukamiHughes::computeVectors_s_vT_v2T_v3T(double v2x, double v2y,
                                                 double v3x, double v3y,
                                                 double vx, double vy,
                                                 double bx, double by,
                                                 double& sx, double& sy,
                                                 double& vTx, double& vTy,
                                                 double& v2Tx, double& v2Ty,
                                                 double& v3Tx, double& v3Ty)
{
  vTx = - vy;
  vTy = vx;
  
  if(vTx * v3x + vTy * v3y < 0.0)
  {
    vTx = - vTx;
    vTy = - vTy;
  }
  
  v2Tx = - v2y;
  v2Ty = v2x;
  
  v3Tx = - v3y;
  v3Ty = v3x;
  
  sx = bx;
  sy = by;
  const double sLength = std::sqrt(sx * sx + sy * sy);
  sx /= sLength;
  sy /= sLength;
}


double MizukamiHughes::sgn(double x)
{
  const double machinePrecision = 2.0e-16;
  
  if(std::abs(x) < machinePrecision)
    return 0.0;
  else if(x > 0.0)
    return 1.0;
  else
    return - 1.0;
}


void MizukamiHughes::checkCsForTwoTrianglesRefLevel2()
{
  BlockVector exactSolution(passedSol.length());
  
  TFEFunction2D exactSolutionFct(sysMat.GetAnsatzSpace2D(),
                                 "exactSolutionFct", exactSolution.get_entries()); 
  
  exactSolutionFct.Interpolate(example.get_exact(0));
  
  for(unsigned int i = 0; i < bxVals.length(); i++)
  {
    bxVals[i] = - 1.0;
    byVals[i] = - 1.0;
  }
  
  // \vec{b} = 0
  bxVals[31] = 0.0;
  byVals[31] = 0.0;
  
  // initially pointing into VZ0, numbering from 0
  bxVals[17] = 1.0;
  byVals[17] = 0.9;
  
  // initially pointing into VZ1, numbering from 0
  bxVals[27] = 1.0;
  byVals[27] = - 1.0;
  
  // initially pointing into VZ2, numbering from 0
  bxVals[22] = - 1.0;
  byVals[22] = 1.0;
  
  // intersects Dirichlet boundary
  bxVals[23] = - 1.0;
  byVals[23] = 1.0;
  
  // b \cdot \nabla u = 0, intersects boundary, but not Dirichlet boundary
  // these values have to be changed for each method
  bxVals[6] = -1.0;
  byVals[6] = 1.0;
  
  // V_2 nonempty, V_3 empty
  bxVals[2] = - 0.1;
  byVals[2] = 1.0;
  
  // V_2 empty, V_3 nonempty
  bxVals[15] = - 1.0;
  byVals[15] = 0.1;
  
  // w \cdot v^T < 0
  bxVals[28] = 1.0;
  byVals[28] = 0.1;
  
  // default option
  bxVals[9] = - 0.1;
  byVals[9] = - 1.0;
  
  
  switch(mizukamiHughesType)
  {
    case MizukamiHughesType::Original:
      computeCsOriginal(exactSolution);
      compareCsOriginal();
      break;
    case MizukamiHughesType::Improved:
      computeCsImproved(exactSolution);
      compareCsImproved();
      
      computeCsImprovedTolerances(exactSolution);
      compareCsImproved();
      break;
    case MizukamiHughesType::Reaction:
      computeCsImprovedReactionTolerances(exactSolution);
      compareCsImprovedReactionTolerances();
  }
}


void MizukamiHughes::compareCsOriginal()
{
  std::array<double,3> expectedValues({0.0, 0.0, 0.0});
  
  expectedValues = {0.66666666666666663, -0.333333333333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 2);
  
  expectedValues = {0.33333333333333, -0.33333333333333331, 0.33333333333333331};
  compareComputedAndExpectedValues(expectedValues, 6);
  
  expectedValues = {0.33333333333333, 0.33333333333333331, -0.33333333333333331};
  compareComputedAndExpectedValues(expectedValues, 9);
  
  expectedValues = {-0.33333333333333333, -0.33333333333333333, 0.66666666666666667};
  compareComputedAndExpectedValues(expectedValues, 15);
  
  expectedValues = {0.66666666666666667, -0.33333333333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 17);
  
  expectedValues = {-0.33333333333333333, -0.33333333333333333, 0.66666666666666667};
  compareComputedAndExpectedValues(expectedValues, 22);
  
  expectedValues = {0.33333333333333331, 0.33333333333333331, -0.33333333333333331};
  compareComputedAndExpectedValues(expectedValues, 23);
  
  expectedValues = {-0.33333333333333333, 0.66666666666666667, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 27);
  
  expectedValues = {0.33333333333333, 0.33333333333333331, -0.33333333333333331};
  compareComputedAndExpectedValues(expectedValues, 28);
  
  expectedValues = {0.0, 0.0, 0.0};
  compareComputedAndExpectedValues(expectedValues, 31);
}


void MizukamiHughes::compareCsImproved()
{
  std::array<double,3> expectedValues({0.0, 0.0, 0.0});
  
  expectedValues = {0.5822222222222222, -0.33333333333333331, -0.24888888888888888};
  compareComputedAndExpectedValues(expectedValues, 2);
  
  expectedValues = {0.16666666666667, -0.33333333333333, 0.16666666666667};
  compareComputedAndExpectedValues(expectedValues, 6);
  
  expectedValues = {-0.1888999545931202, 0.4919228649422574, -0.3030229103491372};
  compareComputedAndExpectedValues(expectedValues, 9);
  
  expectedValues = {-0.24888888888888888, -0.33333333333333331, 0.5822222222222222};
  compareComputedAndExpectedValues(expectedValues, 15);
  
  expectedValues = {0.62866666666667, -0.29533333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 17);
  
  expectedValues = {-0.31433333333333, -0.31433333333333, 0.62866666666667};
  compareComputedAndExpectedValues(expectedValues, 22);
  
  expectedValues = {-0.33333333333333333, -0.33333333333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 23);
  
  expectedValues = {-0.31433333333333, 0.62866666666667, -0.31433333333333};
  compareComputedAndExpectedValues(expectedValues, 27);
  
  expectedValues = {0.4919228649422574, -0.1888999545931202, -0.3030229103491372};
  compareComputedAndExpectedValues(expectedValues, 28);
  
  expectedValues = {0.0, 0.0, 0.0};
  compareComputedAndExpectedValues(expectedValues, 31);
}


void MizukamiHughes::compareCsImprovedReactionTolerances()
{
  std::array<double,3> expectedValues({0.0, 0.0, 0.0});
  
  expectedValues = {0.66666666666667, -0.33333333333333, -0.33333333333333};
  compareComputedAndExpectedValues(expectedValues, 2);
  
  expectedValues = {-0.25, -0.33333333333333, -0.25};
  compareComputedAndExpectedValues(expectedValues, 6);
  
  expectedValues = {-0.2107403144547994, 0.54407364778813272, -0.33333333333333};
  compareComputedAndExpectedValues(expectedValues, 9);
  
  expectedValues = {-0.33333333333333, -0.33333333333333, 0.66666666666667};
  compareComputedAndExpectedValues(expectedValues, 15);
  
  expectedValues = {0.66666666666667, -0.33333333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 17);
  
  expectedValues = {-0.33333333333333, -0.33333333333333, 0.66666666666667};
  compareComputedAndExpectedValues(expectedValues, 22);
  
  expectedValues = {-0.33333333333333333, -0.33333333333333333, -0.33333333333333333};
  compareComputedAndExpectedValues(expectedValues, 23);
  
  expectedValues = {-0.33333333333333, 0.66666666666667, -0.33333333333333};
  compareComputedAndExpectedValues(expectedValues, 27);
  
  expectedValues = {0.54407364778813272, -0.2107403144547994, -0.33333333333333};
  compareComputedAndExpectedValues(expectedValues, 28);
  
  expectedValues = {-0.25, -0.25, -0.25};
  compareComputedAndExpectedValues(expectedValues, 31);
}


void MizukamiHughes::compareComputedAndExpectedValues(std::array<double,3> expectedValues,
                                                      int cellNr)
{
  const double tolerance = 1.0e-14;
    
  for(int i = 0; i < 3; i++)
  {
    if(std::abs(cCoeffsVals.at(i)[cellNr] - expectedValues.at(i)) > tolerance)
      ErrThrow("Value of coefficient C_", i, " (numbered from zero) on triangle number ",
               cellNr, " is too different! It is ", setprecision(14),
               cCoeffsVals.at(i)[cellNr], " but should be ", expectedValues.at(i));
  }
}


bool MizukamiHughes::existsAlpha(double p0, double p1, double p2,
                                 double s0, double s1, double s2,
                                 double& alphaMin, double& alphaMax,
                                 double& alphaMinimizing)
{                                             
  double min = - 1.e200, max = 1.e200, s;
  bool exists = true;

  if(std::fabs(p1) < tolEq)
  {
    if(s1 > 0.0)
      exists = false;
  }
  else
  {
    s = s1 / p1;
    
    if(p1 > 0.0)
      min = s;
    else
      max = s;
  }
  
  if(std::fabs(p2) < tolEq)
  {
    if(s2 > 0.0)
      exists = false;
  }
  else
  {
    s = s2 / p2;
    if(p2 > 0.0)
      min = min > s ? min : s;
    else
      max = max < s ? max : s;
  }
  
  if(min > max)
    exists = false;
  
  
  if(exists)
  {
    if(std::fabs(p0) < tolEq)
    {
      if(s0 < tolEq)
        exists = false;
    }
    else
    {
      s = s0 / p0;
      if(p0 > 0.0)
      { 
        if(min >= s)
          exists = false;
        else
          max = max < s - tolEq ? max : s - tolEq;
      }
      else
      {
        if(max <= s)
          exists = false;
        else
          min = min > s + tolEq ? min : s + tolEq;
      }
    }
    
    if(min > max)
     exists = false;
  }
   
   
  alphaMin = - max;
  alphaMax = - min;
  
  if(p0 > 0.0)
    alphaMinimizing = alphaMin;
  else
    alphaMinimizing = alphaMax;
  
  return exists;
}


double MizukamiHughes::getPhiTimesBrace(double sx, double sy, double vx, double vy,
                                     double bx, double by, double wx, double wy,
                                     double v2x, double v2y, double v3x, double v3y,
                                     double v2Tx, double v2Ty)
{
  const double fract_s_v2T_v = std::abs(sx * v2Tx + sy * v2Ty) / std::abs(vx * v2Tx + vy * v2Ty);
  const double r2 = std::min({ 1.0, fract_s_v2T_v + 1.0 - sgn(bx * v2x + by * v2y) });
  
  const double fract_w_v2T_v_v2 = std::abs(wx * v2Tx + wy * v2Ty) / ( vx * v2x + vy * v2y );
  const double phiCoeff = std::min({ 1.0, 2.0 * fract_w_v2T_v_v2 / r2 });
  
  const double prod_v2_v3_s = (v2x - v3x) * sx + (v2y - v3y) * sy;
  const double prod_1_v2_v3 = 1.0 - v2x * v3x - v2y * v3y;
  const double brac = 1.0 + prod_v2_v3_s / prod_1_v2_v3;
  
  return phiCoeff * brac;
}


void MizukamiHughes::setCombinedCs(const double* phix, const double* phiy,
                                   const double* b_phi, const double* w_phi,
                                   const double* alphasMinimizing, const bool* VsEmpty,
                                   double& c1, double& c2, double& c3, const int* vLocDers)
{
  double cik[4][4];
  
  for(int i = 1; i <= 3; i++)
    for(int k = 2; k <= 3; k++)
    {
      if(i == k)
        continue;
      
      if(VsEmpty[k])
      {
        cik[i][k] = 0.0;
        continue;
      }
      
      const double phi_phi =  phix[vLocDers[k]] * phix[vLocDers[i]]
                            + phiy[vLocDers[k]] * phiy[vLocDers[i]];
                            
      const double denominator = b_phi[vLocDers[k]] + alphasMinimizing[k] * w_phi[vLocDers[k]];
      const double minimumTerm = - 1.0 / 3.0 - epsilonTilde * phi_phi / denominator;
      
      cik[i][k] = minimumTerm < 0.0 ? minimumTerm : 0.0;
    }
    
    
  for(int k = 2; k <= 3; k++)
  {
    cik[k][k] = 0.0;
    
    for(int i = 1; i <= 3; i++)
    {
      if(i != k)
        cik[k][k] -= cik[i][k];
    }
  }
  
  const double c2Temp = (1.0 / 3.0 + c2) * cik[2][2] + (1.0 / 3.0 + c3) * cik[2][3];
  
  c1 = (1.0 / 3.0 + c2) * cik[1][2] + (1.0 / 3.0 + c3) * cik[1][3];
  c3 = (1.0 / 3.0 + c2) * cik[3][2] + (1.0 / 3.0 + c3) * cik[3][3];
  
  c2 = c2Temp;
}


void MizukamiHughes::computeCsOriginal(const BlockVector& sol)
{
  const double machinePrecision = 1.0e-13;
  
  const int N_BaseFunct = 3;
  double phiOrig[N_BaseFunct+1];
  double phix[N_BaseFunct+1];
  double phiy[N_BaseFunct+1];
  double phiRef[N_BaseFunct];
  double phiXi[N_BaseFunct];
  double phiEta[N_BaseFunct];
  
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const TCollection* collPtr = feSpace2DPtr->GetCollection();
  
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    const TGridCell* cellPtr = (TGridCell*)collPtr->GetCell(cellNr);
    
    const double bx = bxVals[cellNr];
    const double by = byVals[cellNr];
    
    if(std::abs(bx) < machinePrecision && std::abs(by) < machinePrecision)
    {
      for(int i = 0; i < 3; i++)
        cCoeffsVals[i][cellNr] = 0.0;
      
      continue;
    }
    
    
    // derivatives have numbering starting from 1; 0. entry not used
    getDerivativesOfPhi(feSpace2DPtr->get_fe(cellNr), phiXi, phiEta, cellPtr, collPtr,
                        phiRef, phiOrig, phix, phiy);
    
    
    double b_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      b_phi[i] = bx * phix[i] + by * phiy[i];
    
    
    // determine zone into which b points
    int zoneNr; // we have zones 1, 2, 3
    bool inVZ;
    
    determineZone(b_phi, zoneNr, inVZ);
    
    
    // determine local numbering of vertices in which b points into VZ1 or EZ1
    int vLoc[4]; // 3 components, but starts from 1; 0. entry not used
    int vLocDers[4];// as vLoc, but indces are increased by one => for indexing derivatives
    determineLocalNumbering(zoneNr, vLoc, vLocDers);
    
   
     // b in VZ ?
    if(inVZ)
    {
      cCoeffsVals[vLoc[1]][cellNr] = 2.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
      
      continue;
    }

    
    // b \cdot \nabla u = 0 ?
    double solx = 0.0;
    double soly = 0.0;
    
    for(int vertexNr = 0; vertexNr < 3; vertexNr++)
    {
      int i = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
      
      solx += sol[i] * phix[vertexNr+1];
      soly += sol[i] * phiy[vertexNr+1];
    }
    
    double b_nablaSol = bx * solx + by * soly;
    
    if(std::abs(b_nablaSol) < machinePrecision)
    {
      for(int i = 1; i <= 3; i++)
        cCoeffsVals[vLoc[i]][cellNr] = b_phi[vLocDers[i]] / (3.0 * std::abs(b_phi[vLocDers[i]]));
          
      continue;
    }
    
    
    // define auxiliary vectors
    double v2x, v2y, v3x, v3y, vx, vy, wx, wy;
    computeVectors_v2_v3_v_w(cellPtr, vLoc, solx, soly,
                             v2x, v2y, v3x, v3y, vx, vy, wx, wy);
    
    
    // check V_2 and V_3
    double w_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      w_phi[i] = wx * phix[i] + wy * phiy[i];
    
    double alphaMins[4];// first two components are not used
    double alphaMaxes[4];// first two components are not used
    double alphasMinimizing[4];// first two components are not used
    bool VsEmpty[4];// first two components are not used
    
    VsEmpty[2] = !existsAlpha(w_phi[vLocDers[2]], w_phi[vLocDers[1]], w_phi[vLocDers[3]],
                              b_phi[vLocDers[2]], b_phi[vLocDers[1]], b_phi[vLocDers[3]],
                              alphaMins[2], alphaMaxes[2], alphasMinimizing[2]);
    
    VsEmpty[3] = !existsAlpha(w_phi[vLocDers[3]], w_phi[vLocDers[1]], w_phi[vLocDers[2]],
                              b_phi[vLocDers[3]], b_phi[vLocDers[1]], b_phi[vLocDers[2]],
                              alphaMins[3], alphaMaxes[3], alphasMinimizing[3]);
    
    if(!VsEmpty[2] && VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = 2.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
          
      continue;
    }
    
    if(VsEmpty[2] && !VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = 2.0 / 3.0;
          
      continue;
    }
    
    
    // both V_2 and V_3 nonempty
    for(int i = 1; i <= 3; i++)
      cCoeffsVals[vLoc[i]][cellNr] = b_phi[vLocDers[i]] / (3.0 * std::abs(b_phi[vLocDers[i]]));
    
  }// loop over cells
}


void MizukamiHughes::computeCsImproved(const BlockVector& sol)
{
  const double machinePrecision = 1.0e-13;
  
  // Reserving data for computing derivatives of basis functions \phi
  const int N_BaseFunct = 3;
  double phiOrig[N_BaseFunct+1];
  double phix[N_BaseFunct+1];
  double phiy[N_BaseFunct+1];
  double phiRef[N_BaseFunct];
  double phiXi[N_BaseFunct];
  double phiEta[N_BaseFunct];
  
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const TCollection* collPtr = feSpace2DPtr->GetCollection();
  
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    const TGridCell* cellPtr = (TGridCell*)collPtr->GetCell(cellNr);
    
    const double bx = bxVals[cellNr];
    const double by = byVals[cellNr];
    
    if(std::abs(bx) < machinePrecision && std::abs(by) < machinePrecision)
    {
      for(int i = 0; i < 3; i++)
        cCoeffsVals[i][cellNr] = 0.0;
      
      continue;
    }
    
    
    // derivatives have numbering starting from 1; 0. entry not used
    getDerivativesOfPhi(feSpace2DPtr->get_fe(cellNr), phiXi, phiEta, cellPtr, collPtr,
                        phiRef, phiOrig, phix, phiy);
    
    
    // Preparing the values of \vec{b} \cdot \nabla \phi_k
    // that are used everywhere in the code
    double b_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      b_phi[i] = bx * phix[i] + by * phiy[i];
    
    
    // determine zone into which b points
    int zoneNr; // we have zones 1, 2, 3
    bool inVZ;
    
    determineZone(b_phi, zoneNr, inVZ);
    
    
    // determine local numbering of vertices in which b points into VZ1 or EZ1
    int vLoc[4]; // 3 components, but starts from 1; 0. entry not used
    int vLocDers[4];// as vLoc, but indces are increased by one => for indexing derivatives
    
    determineLocalNumbering(zoneNr, vLoc, vLocDers);
    
    
     // b in VZ ?
    if(inVZ)
    {
      for(int i = 2; i <= 3; i++)
      {
        const double phi_dot_phi =  phix[vLocDers[1]] * phix[vLocDers[i]]
                                  + phiy[vLocDers[1]] * phiy[vLocDers[i]];
                            
        const double diff = - 1.0 / 3.0 - epsilonTilde * phi_dot_phi / b_phi[vLocDers[1]];
        
        cCoeffsVals[vLoc[i]][cellNr] = diff < 0.0 ? diff : 0.0;
      }
      
      cCoeffsVals[vLoc[1]][cellNr] = - cCoeffsVals[vLoc[2]][cellNr]
                                     - cCoeffsVals[vLoc[3]][cellNr];
      
      continue;
    }
    
    
    // element intersecting Dirichlet boundary ?
    if( cellIntersectsDirichletBoundary(cellNr) )
    {
      setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                               cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
      
      continue;
    }
    
    
    // not special triangulation + element with all vertices connected by edges to Dirichlet vertices ?
    if(!specialTriangulation)
    {
      if( AllNodesConnectedToDirichletNodes(cellNr) )
      {
        setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                                 cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
        
        continue;
      }
    }
    
    
    // b \cdot \nabla u = 0 ?
    double solx = 0.0;
    double soly = 0.0;
    
    for(int vertexNr = 0; vertexNr < 3; vertexNr++)
    {
      int i = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
      
      solx += sol[i] * phix[vertexNr+1];
      soly += sol[i] * phiy[vertexNr+1];
    }
    
    double b_nablaSol = bx * solx + by * soly;
    
    if(std::abs(b_nablaSol) < machinePrecision)
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = 1.0 / 6.0;
      cCoeffsVals[vLoc[3]][cellNr] = 1.0 / 6.0;
          
      continue;
    }
    
    
    // define auxiliary vectors
    double v2x, v2y, v3x, v3y, vx, vy, wx, wy;
    computeVectors_v2_v3_v_w(cellPtr, vLoc, solx, soly,
                             v2x, v2y, v3x, v3y, vx, vy, wx, wy);
    
    
    // check V_2 and V_3
    double w_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      w_phi[i] = wx * phix[i] + wy * phiy[i];
    
    double alphaMins[4];// first two components are not used
    double alphaMaxes[4];// first two components are not used
    double alphasMinimizing[4];// first two components are not used
    bool VsEmpty[4];// first two components are not used
    
    VsEmpty[2] = !existsAlpha(w_phi[vLocDers[2]], w_phi[vLocDers[1]], w_phi[vLocDers[3]],
                              b_phi[vLocDers[2]], b_phi[vLocDers[1]], b_phi[vLocDers[3]],
                              alphaMins[2], alphaMaxes[2], alphasMinimizing[2]);
    
    VsEmpty[3] = !existsAlpha(w_phi[vLocDers[3]], w_phi[vLocDers[1]], w_phi[vLocDers[2]],
                              b_phi[vLocDers[3]], b_phi[vLocDers[1]], b_phi[vLocDers[2]],
                              alphaMins[3], alphaMaxes[3], alphasMinimizing[3]);
    
    if(!VsEmpty[2] && VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = 2.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
    }
    else if(VsEmpty[2] && !VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = 2.0 / 3.0;
    }
    else
    {
      // define more auxiliary vectors
      double sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty;
      computeVectors_s_vT_v2T_v3T(v2x, v2y, v3x, v3y, vx, vy, bx, by,
                                  sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty);
      
      
      // w \cdot v^T < 0 ?
      if(wx * vTx + wy * vTy < 0.0)
      {
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v2x, v2y, v3x, v3y, v2Tx, v2Ty);
        
        cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[3]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[2]][cellNr];
      }
      else
      {
        // nothing from above satisfied => default option
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v3x, v3y, v2x, v2y, v3Tx, v3Ty);
        
        cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[2]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[3]][cellNr];
      }
    }
    
    
    setCombinedCs(phix, phiy, b_phi, w_phi, alphasMinimizing, VsEmpty,
                  cCoeffsVals[vLoc[1]][cellNr], cCoeffsVals[vLoc[2]][cellNr],
                  cCoeffsVals[vLoc[3]][cellNr], vLocDers);
    
  }// loop over cells
}


void MizukamiHughes::computeCsImprovedTolerances(const BlockVector& sol)
{
  const int N_BaseFunct = 3;
  double phiOrig[N_BaseFunct+1];
  double phix[N_BaseFunct+1];
  double phiy[N_BaseFunct+1];
  double phiRef[N_BaseFunct];
  double phiXi[N_BaseFunct];
  double phiEta[N_BaseFunct];
  
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const TCollection* collPtr = feSpace2DPtr->GetCollection();
  
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    const TGridCell* cellPtr = (TGridCell*)collPtr->GetCell(cellNr);
    
    const double bx = bxVals[cellNr];
    const double by = byVals[cellNr];
    
    if(std::abs(bx) < tolEq && std::abs(by) < tolEq)
    {
      for(int i = 0; i < 3; i++)
        cCoeffsVals[i][cellNr] = 0.0;
      
      continue;
    }
    
    
    // derivatives have numbering starting from 1; 0. entry not used
    getDerivativesOfPhi(feSpace2DPtr->get_fe(cellNr), phiXi, phiEta, cellPtr, collPtr,
                        phiRef, phiOrig, phix, phiy);
    
    
    double b_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      b_phi[i] = bx * phix[i] + by * phiy[i];
    
    
    // determine zone into which b points
    int zoneNr; // we have zones 1, 2, 3
    bool inVZ;
    
    determineZoneTolerances(b_phi, zoneNr, inVZ);
    
    
    // determine local numbering of vertices in which b points into VZ1 or EZ1
    int vLoc[4]; // 3 components, but starts from 1; 0. entry not used
    int vLocDers[4];// as vLoc, but indces are increased by one => for indexing derivatives

    determineLocalNumbering(zoneNr, vLoc, vLocDers);
    
    
     // b in VZ ?
    if(inVZ)
    {
      for(int i = 2; i <= 3; i++)
      {
        const double phi_dot_phi =  phix[vLocDers[1]] * phix[vLocDers[i]]
                                  + phiy[vLocDers[1]] * phiy[vLocDers[i]];
                            
        const double diff = - 1.0 / 3.0 - epsilonTilde * phi_dot_phi / b_phi[vLocDers[1]];
        
        cCoeffsVals[vLoc[i]][cellNr] = diff < 0.0 ? diff : 0.0;
      }
      
      cCoeffsVals[vLoc[1]][cellNr] = - cCoeffsVals[vLoc[2]][cellNr]
                                     - cCoeffsVals[vLoc[3]][cellNr];
      
      continue;
    }
    
    
    // element intersecting Dirichlet boundary ?
    if( cellIntersectsDirichletBoundary(cellNr) )
    {
      setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                               cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
      
      continue;
    }
    
    
    // not special triangulation + element with all vertices connected by edges to Dirichlet vertices ?
    if(!specialTriangulation)
    {
      if( AllNodesConnectedToDirichletNodes(cellNr) )
      {
        setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                                 cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
        
        continue;
      }
    }
    
    
    // b \cdot \nabla u = 0 ?
    double solx = 0.0;
    double soly = 0.0;
    
    for(int vertexNr = 0; vertexNr < 3; vertexNr++)
    {
      int i = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
      
      solx += sol[i] * phix[vertexNr+1];
      soly += sol[i] * phiy[vertexNr+1];
    }
    
    double b_nablaSol = bx * solx + by * soly;
    const double bMax = std::abs(bx) > std::abs(by) ? std::abs(bx) : std::abs(by);
    
    if(std::abs(b_nablaSol) < tolEq * bMax)
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = 1.0 / 6.0;
      cCoeffsVals[vLoc[3]][cellNr] = 1.0 / 6.0;
      
      continue;
    }
    
    
    // define auxiliary vectors
    double v2x, v2y, v3x, v3y, vx, vy, wx, wy;
    computeVectors_v2_v3_v_w(cellPtr, vLoc, solx, soly,
                             v2x, v2y, v3x, v3y, vx, vy, wx, wy);
    
    
    // check V_2 and V_3
    double w_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      w_phi[i] = wx * phix[i] + wy * phiy[i];
    
    double alphaMins[4];// first two components are not used
    double alphaMaxes[4];// first two components are not used
    double alphasMinimizing[4];// first two components are not used
    bool VsEmpty[4];// first two components are not used
    
    VsEmpty[2] = !existsAlpha(w_phi[vLocDers[2]], w_phi[vLocDers[1]], w_phi[vLocDers[3]],
                              b_phi[vLocDers[2]], b_phi[vLocDers[1]], b_phi[vLocDers[3]],
                              alphaMins[2], alphaMaxes[2], alphasMinimizing[2]);
    
    VsEmpty[3] = !existsAlpha(w_phi[vLocDers[3]], w_phi[vLocDers[1]], w_phi[vLocDers[2]],
                              b_phi[vLocDers[3]], b_phi[vLocDers[1]], b_phi[vLocDers[2]],
                              alphaMins[3], alphaMaxes[3], alphasMinimizing[3]);
    
    if(!VsEmpty[2] && VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = 2.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
    }
    else if(VsEmpty[2] && !VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = 2.0 / 3.0;
    }
    else
    {
      // define more auxiliary vectors
      double sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty;
      computeVectors_s_vT_v2T_v3T(v2x, v2y, v3x, v3y, vx, vy, bx, by,
                                  sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty);
      
      
      // w \cdot v^T < 0 ?
      if(wx * vTx + wy * vTy < 0.0)
      {
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v2x, v2y, v3x, v3y, v2Tx, v2Ty);
        
        cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[3]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[2]][cellNr];
      }
      else // nothing from above satisfied => default option
      {
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v3x, v3y, v2x, v2y, v3Tx, v3Ty);
        
        cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[2]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[3]][cellNr];
      }
    }
    
    
    setCombinedCs(phix, phiy, b_phi, w_phi, alphasMinimizing, VsEmpty,
                  cCoeffsVals[vLoc[1]][cellNr], cCoeffsVals[vLoc[2]][cellNr],
                  cCoeffsVals[vLoc[3]][cellNr], vLocDers);
  }// loop over cells
}


void MizukamiHughes::computeCsImprovedReactionTolerances(const BlockVector& sol)
{
  const int N_BaseFunct = 3;
  double phiOrig[N_BaseFunct+1];
  double phix[N_BaseFunct+1];
  double phiy[N_BaseFunct+1];
  double phiRef[N_BaseFunct];
  double phiXi[N_BaseFunct];
  double phiEta[N_BaseFunct];
  
  const TFESpace2D* feSpace2DPtr = sysMat.GetTestSpace2D().get();
  const TCollection* collPtr = feSpace2DPtr->GetCollection();
  
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    const TGridCell* cellPtr = (TGridCell*)collPtr->GetCell(cellNr);
    
    const double bx = bxVals[cellNr];
    const double by = byVals[cellNr];
    
    if(std::abs(bx) < tolEq && std::abs(by) < tolEq)
    {
      const double valToSet = std::abs(reactVals[cellNr]) < tolEq ? 0.0 : - 0.25;
      
      for(int i = 0; i < 3; i++)
        cCoeffsVals[i][cellNr] = valToSet;
        
      continue;
    }
    
    
    // derivatives have numbering starting from 1; 0. entry not used
    getDerivativesOfPhi(feSpace2DPtr->get_fe(cellNr), phiXi, phiEta, cellPtr, collPtr,
                        phiRef, phiOrig, phix, phiy);
    
    
    double b_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      b_phi[i] = bx * phix[i] + by * phiy[i];
    
    
    // determine zone into which b points
    int zoneNr; // we have zones 1, 2, 3
    bool inVZ;
    
    determineZoneTolerances(b_phi, zoneNr, inVZ);
    
    
    // determine local numbering of vertices in which b points into VZ1 or EZ1
    int vLoc[4]; // 3 components, but starts from 1; 0. entry not used
    int vLocDers[4];// as vLoc, but indces are increased by one => for indexing derivatives
    
    determineLocalNumbering(zoneNr, vLoc, vLocDers);
    
    
     // b in VZ ?
    if(inVZ)
    {
      if(std::abs(reactVals[cellNr]) < tolEq)
      {
        for(int i = 2; i <= 3; i++)
        {
          const double phi_dot_phi =  phix[vLocDers[1]] * phix[vLocDers[i]]
                                    + phiy[vLocDers[1]] * phiy[vLocDers[i]];
                              
          const double diff = - 1.0 / 3.0 - epsilonTilde * phi_dot_phi / b_phi[vLocDers[1]];
          
          cCoeffsVals[vLoc[i]][cellNr] = diff < 0.0 ? diff : 0.0;
        }
        
        cCoeffsVals[vLoc[1]][cellNr] = - cCoeffsVals[vLoc[2]][cellNr]
                                       - cCoeffsVals[vLoc[3]][cellNr];
      }
      else
      {
        const double val2 = b_phi[vLocDers[2]] + reactVals[cellNr] / 3.0;
        const double val3 = b_phi[vLocDers[3]] + reactVals[cellNr] / 3.0;

        double valMax = val2 > val3 ? 36.0 * val2 : 36.0 * val3;
        valMax = valMax > 0.0 ? valMax : 0.0;
        
        if(valMax < tolEq)
          cCoeffsVals[vLoc[1]][cellNr] = 2.0 / 3.0;
        else
          cCoeffsVals[vLoc[1]][cellNr] = std::min(  2.0 / 3.0,
                                                  - 1.0 / 3.0 + reactVals[cellNr] / valMax);
        
        cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
      }
      
      continue;
    }
    
    
    // element intersecting Dirichlet boundary ?
    if( cellIntersectsDirichletBoundary(cellNr) )
    {
      setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                               cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
      
      continue;
    }
    
    
    // not special triangulation + element with all vertices connected by edges to Dirichlet vertices ?
    if(!specialTriangulation)
    {
      if( AllNodesConnectedToDirichletNodes(cellNr) )
      {
        setCsForCaseAllOneThirds(phix, phiy, b_phi, vLocDers, cCoeffsVals[vLoc[1]][cellNr],
                                 cCoeffsVals[vLoc[2]][cellNr], cCoeffsVals[vLoc[3]][cellNr]);
        
        continue;
      }
    }
    
    
    // b \cdot \nabla u = 0 ?
    double solx = 0.0;
    double soly = 0.0;
    
    for(int vertexNr = 0; vertexNr < 3; vertexNr++)
    {
      int i = feSpace2DPtr->get_global_dof(cellNr, vertexNr);
      
      solx += sol[i] * phix[vertexNr+1];
      soly += sol[i] * phiy[vertexNr+1];
    }
    
    double b_nablaSol = bx * solx + by * soly;
    const double bMax = std::abs(bx) > std::abs(by) ? std::abs(bx) : std::abs(by);
    
    if(std::abs(b_nablaSol) < tolEq * bMax)
    {
      const double otherValueToSet = std::abs(reactVals[cellNr]) < tolEq ? 1.0 / 6.0 : - 0.25;
      
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = otherValueToSet;
      cCoeffsVals[vLoc[3]][cellNr] = otherValueToSet;
      
      continue;
    }
    
    
    // define auxiliary vectors
    double v2x, v2y, v3x, v3y, vx, vy, wx, wy;
    computeVectors_v2_v3_v_w(cellPtr, vLoc, solx, soly,
                             v2x, v2y, v3x, v3y, vx, vy, wx, wy);
    
    
    // check V_2 and V_3
    double w_phi[4]; // 3 components, but numbering starts from 1; 0. entry not used
    
    for(int i = 1; i <= 3; i++)
      w_phi[i] = wx * phix[i] + wy * phiy[i];
    
    double alphaMins[4];// first two components are not used
    double alphaMaxes[4];// first two components are not used
    double alphasMinimizing[4];// first two components are not used
    bool VsEmpty[4];// first two components are not used
    
    VsEmpty[2] = !existsAlpha(w_phi[vLocDers[2]], w_phi[vLocDers[1]], w_phi[vLocDers[3]],
                              b_phi[vLocDers[2]], b_phi[vLocDers[1]], b_phi[vLocDers[3]],
                              alphaMins[2], alphaMaxes[2], alphasMinimizing[2]);
    
    VsEmpty[3] = !existsAlpha(w_phi[vLocDers[3]], w_phi[vLocDers[1]], w_phi[vLocDers[2]],
                              b_phi[vLocDers[3]], b_phi[vLocDers[1]], b_phi[vLocDers[2]],
                              alphaMins[3], alphaMaxes[3], alphasMinimizing[3]);
    
    if(!VsEmpty[2] && VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0;
      
      const double valToSet = cForVsModifiedDueToReaction(2.0 / 3.0, 2, reactVals[cellNr],
                                                          b_phi, w_phi, vLocDers,
                                                          alphaMins[2], alphaMaxes[2]);
      
      cCoeffsVals[vLoc[2]][cellNr] = valToSet;
    }
    else if(VsEmpty[2] && !VsEmpty[3])
    {
      cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
      cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0;
      
      const double valToSet = cForVsModifiedDueToReaction(2.0 / 3.0, 3, reactVals[cellNr],
                                                          b_phi, w_phi, vLocDers,
                                                          alphaMins[3], alphaMaxes[3]);
      
      cCoeffsVals[vLoc[3]][cellNr] = valToSet;
    }
    else
    {
      // define more auxiliary vectors
      double sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty;
      computeVectors_s_vT_v2T_v3T(v2x, v2y, v3x, v3y, vx, vy, bx, by,
                                  sx, sy, vTx, vTy, v2Tx, v2Ty, v3Tx, v3Ty);
      
      
      // w \cdot v^T < 0 ?
      if(wx * vTx + wy * vTy < 0.0)
      {
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v2x, v2y, v3x, v3y, v2Tx, v2Ty);
        
        cCoeffsVals[vLoc[2]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[3]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[2]][cellNr];
      }
      else
      {
        // nothing from above satisfied => default option
        const double phiTimesBrace = getPhiTimesBrace(sx, sy, vx, vy, bx, by, wx, wy,
                                                      v3x, v3y, v2x, v2y, v3Tx, v3Ty);
        
        cCoeffsVals[vLoc[3]][cellNr] = - 1.0 / 3.0 + 0.5 * phiTimesBrace;
        cCoeffsVals[vLoc[1]][cellNr] = - 1.0 / 3.0;
        cCoeffsVals[vLoc[2]][cellNr] = 1.0 / 3.0 - cCoeffsVals[vLoc[3]][cellNr];
      }
      
      
      const double valToSet2 = cForVsModifiedDueToReaction(cCoeffsVals[vLoc[2]][cellNr],
                                                           2, reactVals[cellNr],
                                                           b_phi, w_phi, vLocDers,
                                                           alphaMins[2], alphaMaxes[2]);
      
      const double valToSet3 = cForVsModifiedDueToReaction(cCoeffsVals[vLoc[3]][cellNr],
                                                           3, reactVals[cellNr],
                                                           b_phi, w_phi, vLocDers,
                                                           alphaMins[3], alphaMaxes[3]);
      
      cCoeffsVals[vLoc[2]][cellNr] = valToSet2;
      cCoeffsVals[vLoc[3]][cellNr] = valToSet3;
    }
    
    if(std::abs(reactVals[cellNr]) < tolEq)
      setCombinedCs(phix, phiy, b_phi, w_phi, alphasMinimizing, VsEmpty,
                    cCoeffsVals[vLoc[1]][cellNr], cCoeffsVals[vLoc[2]][cellNr],
                    cCoeffsVals[vLoc[3]][cellNr], vLocDers);
  }// loop over cells
}


void MizukamiHughes::setCsForCaseAllOneThirds(const double* phix, const double* phiy,
                                                 const double* b_phi, const int* vLocDers,
                                                 double& c1, double& c2, double& c3)
{
  const double phi1_phi2 = phix[vLocDers[1]] * phix[vLocDers[2]]
                         + phiy[vLocDers[1]] * phiy[vLocDers[2]];
  const double phi2_phi3 = phix[vLocDers[2]] * phix[vLocDers[3]]
                         + phiy[vLocDers[2]] * phiy[vLocDers[3]];
  const double phi1_phi3 = phix[vLocDers[1]] * phix[vLocDers[3]]
                         + phiy[vLocDers[1]] * phiy[vLocDers[3]];
                         
  const double phi1223Min = phi1_phi2 > phi2_phi3 ? phi1_phi2 : phi2_phi3;
  const double phi1323Min = phi1_phi3 > phi2_phi3 ? phi1_phi3 : phi2_phi3;
  
  const double phi1223MinFrac = phi1223Min / b_phi[vLocDers[2]];
  const double phi1323MinFrac = phi1323Min / b_phi[vLocDers[3]];
  
  const double maxFrac = phi1223MinFrac > phi1323MinFrac ? phi1223MinFrac : phi1323MinFrac;
  double term = - 1.0 / 3.0 - epsilonTilde * maxFrac;
  term = term < 0.0 ? term : 0.0;
  
  c1 = term;
  c2 = term;
  c3 = term;
}


double MizukamiHughes::minMaxAlphaInV(double s1, double s3, double p1, double p3,
                                      double alphaMin, double alphaMax)
{
  // computes \min_{\alpha \in V_2} \max \{ 0, \xi_1 (\alpha), \xi_3 (\alpha) \}
  // for \xi_k (\alpha) = s_k + p_k \alpha
  // The resulting value can be only the value for one of the following 5 \alpha:
  // \alpha = alphaMin, alphaMax, (s3 - s1) / (p1 - p3), - s1 / p1, - s3 / p3
  // where the last 3 args may not be defined and may not be in V_2
  // => We compare the values for these (up to) 5 arguments \alpha
  
  double args[5] = {alphaMin, alphaMax, 0.0, 0.0, 0.0};
  int lastArgInd = 1;
  
  if(std::abs(p1 - p3) >= tolEq)
  {
    const double alpha1 = (s3 - s1) / (p1 - p3);
    
    if(alphaMin <= alpha1 && alpha1 <= alphaMax)
    {
      lastArgInd++;
      args[lastArgInd] = alpha1;
    }
  }
  
  if(std::abs(p1) >= tolEq)
  {
    const double alpha2 = - s1 / p1;
    
    if(alphaMin <= alpha2 && alpha2 <= alphaMax)
    {
      lastArgInd++;
      args[lastArgInd] = alpha2;
    }
  }
  
  if(std::abs(p3) >= tolEq)
  {
    const double alpha3 = - s3 / p3;
    
    if(alphaMin <= alpha3 && alpha3 <= alphaMax)
    {
      lastArgInd++;
      args[lastArgInd] = alpha3;
    }
  }
  
  double minVal = 1.0e100;
  
  for(int i = 0; i <= lastArgInd; i++)
  {
    const double innerMax = std::max({0.0, s1 + p1 * args[i], s3 + p3 * args[i]});
    
    if(minVal > innerMax)
      minVal = innerMax;
  }
  
  minVal *= 36.0;
  
  return minVal;
}


double MizukamiHughes::cForVsModifiedDueToReaction(double origValue, int zoneNr, double react,
                                                   const double* b_phi, const double* w_phi,
                                                   const int* vLocDers,
                                                   double alphaMin, double alphaMax)
{
  if(std::abs(react) < tolEq)
    return origValue;
  
  const int secondZoneNr = zoneNr == 2 ? 3 : 2;
  
  double minMax = minMaxAlphaInV(b_phi[vLocDers[1]] + react / 3.0,
                                 b_phi[vLocDers[secondZoneNr]] + react / 3.0,
                                 w_phi[vLocDers[1]], w_phi[vLocDers[secondZoneNr]],
                                 alphaMin, alphaMax);
  
  if(std::abs(minMax) < tolEq)
    return origValue;
  else
    return std::min(origValue, - 1.0 / 3.0 + react / minMax);
}
