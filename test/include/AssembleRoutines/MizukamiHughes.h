#include "GridCell.h"
#include "Example_CD2D.h"
#include "FEFunction2D.h"

#include "BlockVector.h"
#include <Structure.h>
#include <array>
#include "templateNames.h"
#include "FEMatrix.h"

enum class MizukamiHughesType
{
  Original,
  Improved,
  Reaction
};

/** @brief This class implements 3 variants the Mizukami-Hughes method for CD2D,
 * which is a nonlinear upwind. These variants are:
 * 1) the original method for a convection-diffusion equation
 *    (described in the same paper as the variant 3),
 * 2) a modification of 1) by Petr Knobloch from
 *   P. Knobloch. Numerical solution of convection-diffusion equations using a
 *   nonlinear method of upwind type. J. Sci. Comput., 43(3):454–470, 2010.
 * 3) a modification of 1) by Petr Knobloch that combines the improved method
 *    and the extension for the convection-diffusion-reaction equation from
 *   P. Knobloch. Improvements of the Mizukami-Hughes method
 *   for convection-diffusion equations. Comput. Methods Appl. Mech. Engrg.,
 *   196(1-3):579–594, 2006.
 * 
 * The implementation uses the same notation as the above papers,
 * including numbering objects in series from 1, not from 0.
 * The reason for the change in the numbering convention was purely to decrease
 * the probability of making typos during the implementation. */ 
class MizukamiHughes
{
  public:
    
    MizukamiHughes(const ParameterDatabase& db, const Example_CD2D& passedExample,
                   const FEMatrix& sysMatrix, const BlockVector& solution);
    
    void setAssemblyParameters(std::vector<const TFEFunction2D*>& feFunctionPtr,
                               ParameterDatabase& db);
    
    /** @brief Computes upwind constants
     * via the original Mizukami-Hughes method, variant 1. */
    void computeCsOriginal(const BlockVector& sol);
    
    /** @brief Computes upwind constants
     * via the improved Mizukami-Hughes method, variant 2. */
    void computeCsImproved(const BlockVector& sol);
    
    /** @brief Computes upwind constants
     * via the improved Mizukami-Hughes method, variant 2.
     * This implementation contains various modifications of the formulas in the papers
     * due to finite-precision arithmetic. These modification were taken
     * from Petr Knobloch's implementation of this method. */
    void computeCsImprovedTolerances(const BlockVector& sol);
    
    /** @brief Computes upwind constants
     * via the Mizukami-Hughes method, variant 3.
     * This implementation contains various modifications of the formulas in the papers
     * due to finite-precision arithmetic. These modification were taken
     * from Petr Knobloch's implementation of this method. */
    void computeCsImprovedReactionTolerances(const BlockVector& sol);
    
    bool doAssemble() { return doAssembly; }
    
    const TFEFunction2D& getc1Fct()
    { return c1CoeffsFct; }
    const TFEFunction2D& getc2Fct()
    { return c2CoeffsFct; }
    const TFEFunction2D& getc3Fct()
    { return c3CoeffsFct; }
    const TFEFunction2D& getbxFct()
    { return bxFct; }
    const TFEFunction2D& getbyFct()
    { return byFct; }
    const TFEFunction2D& getReactFct()
    { return reactFct; }
    
    /** @brief Routine checking the correctness of the implementations:
     * It assumes that we have just computed the solution in the example
     * "rectangle_x_plus_y_neumann" using the Mizukami-Hughes method
     * and changes the convection vector "b" so that all of the cases
     * in the Mizukami-Hughes method occur
     * and checks values of the resulting upwind constants.*/
    void checkCsForTwoTrianglesRefLevel2();
    
  private:
    
    void computeCoeffsConstantOnTriangles(const CoeffFct2D& bilinearCoeffs);
    
    double sgn(double x);
    
    std::shared_ptr<TFESpace2D> cCoeffsSpace;
    BlockVector bxVals;
    BlockVector byVals;
    BlockVector reactVals;
    
    std::array<BlockVector,3> cCoeffsVals;
    
    TFEFunction2D c1CoeffsFct;
    TFEFunction2D c2CoeffsFct;
    TFEFunction2D c3CoeffsFct;
  
    TFEFunction2D bxFct;
    TFEFunction2D byFct;
    TFEFunction2D reactFct;
    
    /** @brief Stores the imformation on whether or not we use
     * the special type of grid mentioned in the papers describing the method.
     * The user must to provide this piece of information;
     * the program does not check the grid. */
    bool specialTriangulation;
    
    const BlockVector& passedSol;
    const FEMatrix& sysMat;
    
    /** @brief 0 = computing initial guess,
     * 1 = first iteration of nonlinear loop,
     * 2 and more = further iterations of nonlinear loop */
    unsigned int computationStage;
    
    /** @brief It is false only if skipping the assembly and solving used to get initial guess
     * for the nonlinear solver.
     * We want to skip them if the requested initial guess is zero. */
    bool doAssembly;
    
    /** @brief whether exists such alpha that
     * s0 > alpha * p0, s1 <= alpha * p1, s2 <= alpha * p2
     * and computes minimum and maximum alpha */ 
    bool existsAlpha(double p0, double p1, double p2,
                     double s0, double s1, double s2,
                     double& alphaMin, double& alphaMax,
                     double& alphaMinimizing);
    
    double getPhiTimesBrace(double sx, double sy, double vx, double vy,
                         double bx, double by, double wx, double wy,
                         double v2x, double v2y, double v3x, double v3y,
                         double v2Tx, double v2Ty);
    
    void setCombinedCs(const double* phix, const double* phiy,
                       const double* b_phi, const double* w_phi,
                       const double* alphasMinimizing, const bool* VsEmpty,
                       double& c1, double& c2, double& c3, const int* vLocDers);
    
    void setCsForCaseAllOneThirds(const double* phix, const double* phiy,
                                     const double* b_phi, const int* vLocDers,
                                     double& c1, double& c2, double& c3);
    
    /** @brief Computes \min_{\alpha \in V_2} \max \{ 0, \xi_1 (\alpha), \xi_3 (\alpha) \}
     * for \xi_k (\alpha) = s_k + p_k \alpha */ 
    double minMaxAlphaInV(double s1, double s3, double p1, double p3,
                          double alphaMin, double alphaMax);
    
    double cForVsModifiedDueToReaction(double origValue, int zoneNr, double react,
                                       const double* b_phi, const double* w_phi,
                                       const int* vLocDers,
                                       double alphaMin, double alphaMax);
    
    void getDerivativesOfPhi(const FiniteElement& finiteElement,
                             double* phiXi, double* phiEta,
                             const TGridCell* cellPtr, const TCollection* collPtr,
                             double* phiRef, double* phiOrig,
                             double* phix, double* phiy);
    void determineZone(const double* b_phi, int& zoneNr, bool& inVZ);
    void determineZoneTolerances(const double* b_phi, int& zoneNr, bool& inVZ);
    void determineLocalNumbering(int zoneNr, int* vLoc, int* vLocDers);
    bool cellIntersectsDirichletBoundary(int cellNr);
    bool AllNodesConnectedToDirichletNodes(int cellNr);
    bool OneNodeConnectedToDirichletNodes(int cellNr);
    void computeVectors_v2_v3_v_w(const TGridCell* cellPtr, const int* vLoc,
                                  double solx, double soly,
                                  double& v2x, double& v2y, double& v3x, double& v3y,
                                  double& vx, double& vy, double& wx, double& wy);
    void computeVectors_s_vT_v2T_v3T(double v2x, double v2y, double v3x, double v3y,
                                     double vx, double vy, double bx, double by,
                                     double& sx, double& sy, double& vTx, double& vTy,
                                     double& v2Tx, double& v2Ty, double& v3Tx, double& v3Ty);
    
    void compareComputedAndExpectedValues(std::array<double,3> expectedValues, int cellNr);
    void compareCsOriginal();
    void compareCsImproved();
    void compareCsImprovedReactionTolerances();
    
    double epsilonTilde;
    const double ratioEpsilonTildeEpsilon;
    
    MizukamiHughesType mizukamiHughesType;
    std::string mizukamiHughesTypeString;
    
    const double tolEq = 1.0e-10;
    
    const Example_CD2D& example;
};
