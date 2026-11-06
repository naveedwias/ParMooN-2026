#include "ConvectionDiffusion.h"
#include "ConvDiff.h"
#include "Database.h"
#include "Domain.h"
#include "LocalAssembling.h"
#include "Multigrid.h"
#include "MainUtilities.h"
#include "BaseCell.h"
#include "Assemble_DG.h"
#include "Point.h"
#ifdef __2D__
 #include "Assemble2D.h"
 #include "SquareMatrix2D.h"
 #include "AuxParam2D.h"
 #include "Example_CD2D.h"
 #include "FEDatabase.h"
 #include "Upwind.h"
#else
 #include "Assemble3D.h"
 #include "SquareMatrix3D.h"
 #include "AuxParam3D.h"
 #include "Example_CD3D.h"
#endif
#ifdef _MPI
 #include "ParFECommunicator3D.h"
 #include "MumpsWrapper.h"
#endif
#include <cmath>
#include <algorithm>
#include <Utilities.h>

/* ************************************************************************* */
template<int d>
ParameterDatabase ConvectionDiffusion<d>::default_cd_database(bool complete)
{
  Output::print<5>("creating a default ConvectionDiffusion parameter database");
  // we use a ParMooN default database because this way these parameters are
  // available in the default CD2D database as well.
  ParameterDatabase db = ParameterDatabase::parmoon_default_database();
  db.set_name("ConvectionDiffusion parameter database");

  if(complete)
  {
    db.merge(TDomain::default_domain_parameters());
    if(d == 3)
      db.add_nested_database(TDomain::default_sandwich_grid_parameters());
    db.merge(Example_CD::default_example_database());
  }
  // a default output database - needed here as long as there's no class handling the output
  db.merge(ParameterDatabase::default_output_database());
  // default local assembling database
  db.merge(LocalAssembling<d>::default_local_assembling_database());
  if(complete)
  {
    db.merge(CheckpointIO::default_checkpoint_io_database());
    db.merge(Solver<>::default_solver_database());
#ifdef __2D__
    db.merge(TSlopeLimiter::default_slope_limiter_database());
#endif
  }
  // Note that the DG database cannot be in the complete case, since the DG
  // method is not a class yet. The reason for this behaviour is that the CD
  // class only saves the parameters needed by itself and forwards the other
  // parameters to other classes. The rest is deleted if not saved explicitly.
  // Hence, it cannot be in the complete case yet.
  db.merge(default_dg_database());

  #ifdef __2D__
    db.merge(defaultNonlinearSolverDatabase());
  #endif
  
  db["problem_type"] = 1; // means stationary convection-diffusion
  return db;
}

/* ************************************************************************* */
#ifdef __2D__
template<int d>
ParameterDatabase ConvectionDiffusion<d>::defaultNonlinearSolverDatabase()
{
  ParameterDatabase db("default CD nonlinear solver database");
  
  db.add( "initial_solution_cd", "zero", "Initial solution for CD nonlinear solver.",
         {"zero", "galerkin", "upwind"} );
  
  db.add( "it_max_cd", (size_t)10000, "Maximal number of iterations of CD nonlinear solver.",
          (size_t)0, (size_t)1000000 );
  
  db.add("stopping_tolerance_cd", 1e-10,
         "Stopping tolerance for residual of CD nonlinear solver.", 1e-20, 1e20);
  
  // parameters for dynamic damping from [JK08.CMAME]
  db.add("damping_factor_omega_max_cd", 1.0, 
         "Maximal value of damping_factor_omega_cd.", 0.0, 1.0);

  db.add("damping_factor_omega_min_cd", 0.01, 
         "Minimum value of damping_factor_omega_min_cd.", 0.0, 1.0);
  
  db.add("damping_parameter_c1_cd", 1.001, "See JK08.CMAME.", 1.0, 2.0);
  
  db.add("damping_parameter_c2_cd", 1.1, "See JK08.CMAME.", 1.0, 2.0);

  db.add("damping_parameter_c3_cd", 1.001, "See JK08.CMAME.", 1.0, 2.0);
  
  db.add("damping_parameter_c4_cd", 0.9, "See JK08.CMAME.", 0.0, 1.0);

  db.add("constant_damping_cd", false, 
         "Whether or not damping parameter omega in CD nonlinear solver is constant.",
         {true, false});
  
  return db;
}
#endif

/* ************************************************************************* */
template<int d>
ConvectionDiffusion<d>::SystemPerGrid::SystemPerGrid(const Example_CD& example,
    const TCollection& coll, int ansatz_order)
: fe_space(new FESpace(&coll, "space", example.get_bc(0), ansatz_order))
{
#ifdef _MPI
  fe_space->get_communicator().print_info();
#endif // _MPI
#ifdef __3D__
  matrix = BlockFEMatrix::CD3D(fe_space);
#else
  matrix = BlockFEMatrix::CD2D(fe_space);
#endif

  rhs = BlockVector(this->matrix, true);
  solution = BlockVector(this->matrix, false);
  fe_function = FEFunction(fe_space, "c", solution.get_entries());
}

/* ************************************************************************* */
template<int d>
ConvectionDiffusion<d>::SystemPerGrid::SystemPerGrid(const SystemPerGrid& other)
 : fe_space(other.fe_space), matrix(other.matrix), rhs(other.rhs),
   solution(other.solution)
{
  // the fe functions must be newly created, because copying would mean 
  // referencing the BlockVector in 'other'.
  fe_function = FEFunction(fe_space, "c", solution.get_entries());
}


/* ************************************************************************* */
template<int d>
ConvectionDiffusion<d>::ConvectionDiffusion(const TDomain& domain,
                                            const ParameterDatabase& param_db)
 : ConvectionDiffusion<d>(domain, param_db, Example_CD(param_db))
{
}

/* ************************************************************************* */
template<int d>
ConvectionDiffusion<d>::ConvectionDiffusion(const TDomain& domain,
                                            const ParameterDatabase& param_db,
                                            const Example_CD& example_cd)
 : systems(), example(example_cd), db(default_cd_database()),
   outputWriter(param_db), checkpoint_io(param_db), solver(param_db),
   limiter(param_db), errors()
{
  db.merge(param_db, false); // update this database with given values
  set_parameters();
  // The construction of the members differ, depending on whether
  // a multigrid solver will be used or not.
  bool usingMultigrid = solver.is_using_multigrid();
  int ansatz_order = TDatabase::ParamDB->ANSATZ_ORDER;
  
  auto collections = domain.get_grid_collections();
  if(!usingMultigrid)
  {
    // Get the collection on the finest grid
    TCollection& cellCollection = *collections.front();
    // create finite element space and function, a matrix, rhs, and solution
    systems.emplace_back(example, cellCollection, ansatz_order);
  }
  else
  {
    // we are using multigrid
    size_t n_levels = collections.size();
    size_t desirec_multigrid_levels = param_db["multigrid_n_levels"];
    if(desirec_multigrid_levels > n_levels)
      ErrThrow("Not enough collections (", n_levels, ") to use ",
               desirec_multigrid_levels, " multigrid levels!");
    // remove not needed coarser grid from list of collections
    for(size_t i = desirec_multigrid_levels; i < n_levels; ++i)
    {
      collections.pop_back();
    }

    auto mg = this->solver.get_multigrid();
    // Construct systems per grid and store them, finest level first
    std::list<BlockFEMatrix*> matrices;
    for(auto coll : collections)
    {
      Output::print("creating SystemPerGrid object, n_cells = ", coll->GetN_Cells());
      systems.emplace_back(example, *coll, ansatz_order);
      //prepare input argument for multigrid object
      matrices.push_front(&systems.back().matrix);
    }
    mg->initialize(matrices);
  }
  
  outputWriter.add_fe_function(&this->get_function());
  checkpoint_io.read(this->get_solution());
  output_problem_size_info();
  if (db["space_discretization_type"].is("dg") &&
      !this->get_space()->is_discontinuous())
  {
    // To me, it is not clear if this even makes sense at all. Since it is not
    // tested yet, we just exclude it in general even though it might work. But
    // you can try it out if you want to.
    ErrThrow("A DG discretization with continuous elements is not allowed.");
  }
  if (db["space_discretization_type"].is("galerkin") &&
      this->get_space()->is_discontinuous())
  {
    Output::warn("ConvectionDiffusion.C", "Your space is not continuous but ",
        "you choose the standard Galerkin discretization. If you want to use a",
        " DG discretization choose in your database:",
        "\tspace_discretization_type: dg");
  }
  
#ifdef __2D__
  Parameter disc_type{this->db["space_discretization_type"]};
  
  if(   disc_type.is("Mizukami_Hughes_improved") || disc_type.is("Mizukami_Hughes_reaction")
     || disc_type.is("Mizukami_Hughes_original") )
  {
    FEMatrix& sysMat = *get_matrix().get_blocks_uniquely().at(0).get();
    mizukamiHughesPtr = std::shared_ptr<MizukamiHughes>( new MizukamiHughes(db, example, sysMat, get_solution()) );
  }
  else
    mizukamiHughesPtr = nullptr;
#endif

}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::set_parameters()
{
  //set problem_type to CD if not yet set
  if(!db["problem_type"].is(1))
  {
    if (db["problem_type"].is(0))
    {
      db["problem_type"] = 1;
    }
    else
    {
      Output::warn<2>("The parameter problem_type doesn't correspond to CD."
          "It is now reset to the correct value for CD (=1).");
      db["problem_type"] = 1;
    }
  }
  
  if(TDatabase::ParamDB->ANSATZ_ORDER == 0)
  {
    ErrThrow("Ansatz order 0 is not used in convection diffusion "
             "reaction problems! (Vanishing convection and diffusion term).");
  }
}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::output_problem_size_info() const
{
  // print some useful information
  auto& space = *this->systems.front().fe_space;
  double hMin, hMax;
  auto coll = space.GetCollection();
  coll->GetHminHmax(&hMin, &hMax);
  bool delaunay;
  int hanging_nodes = space.get_n_hanging();
  
  delaunay = coll -> IsDelaunay();
  Output::print<1>("N_Cells    : ", setw(13), coll->GetN_Cells());
  Output::print<1>("h(min, max): ", setw(13), hMin, " ", setw(13), hMax);
  Output::print<1>("dofs all   : ", setw(13), space.get_n_dof());
  Output::print<1>("dof active : ", setw(13), space.get_n_active());
  Output::print<1>("hanging nodes: ", setw(13), hanging_nodes);

  if(hanging_nodes != 0)
      Output::print<1>("NOTE: If the initial mesh is Delaunay then so would",
                        " be the refinements");
  if(delaunay)
    Output::print<1>("The mesh is Delaunay");
  else
    Output::print<1>("The mesh is not Delaunay");
}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::assemble(
  std::function<void(LocalAssembling<d>& la)> modify_la)
{  
  LocalAssembling_type laType = LocalAssembling_type::ConvDiff;
  // this loop has more than one iteration only in case of multigrid
  for(auto & s : systems)
  {
    std::vector<const FEFunction*> feFunctionPtr = {&s.fe_function};

    #ifdef __2D__
    if(mizukamiHughesPtr)
    {
      mizukamiHughesPtr->setAssemblyParameters(feFunctionPtr, db);
      
      if(!mizukamiHughesPtr->doAssemble())
        return;
    }
    #endif
    
    // create a local assembling object which is needed to assemble the matrix
    LocalAssembling<d> laObject(this->db, laType, feFunctionPtr,
                                example.get_coeffs());

    if(modify_la)
      modify_la(laObject);
    call_assembling_routine(s, laObject, false);
  }
}

/* ************************************************************************* */
#ifdef __2D__
template<int d>
void ConvectionDiffusion<d>::assembleGeneralEAFE(int quadInner, int quadOuter)
{ 
//   // assembly of rhs
//   std::function<void(LocalAssembling<d>&)> modify_la( 
//   [&](LocalAssembling<d>& la)
//   {
//     std::vector<AssembleFctParam> laf = { CDRHS<d> };
//     la.replace_local_assembling(laf);
//   });
//   
  FEMatrix* matPtr = get_matrix().get_blocks_uniquely().at(0).get();
  const int nActiveRows = matPtr->get_n_active_rows();
//   assemble(modify_la);
  
  
  // set the requested quadrature formulas for line integrals
  QuadratureFormula_type quadInnerForm = getQuadFormType(quadInner);
  QuadratureFormula_type quadOuterForm = getQuadFormType(quadOuter);
  
  TQuadFormula quadFormulaInner(quadInnerForm);
  TQuadFormula quadFormulaOuter(quadOuterForm);
  
  int nPointsInner, nPointsOuter;
  const double* quadPointsInner;
  const double* quadPointsOuter;
  const double* quadWeightsInner;
  const double* quadWeightsOuter;
  quadFormulaInner.GetFormulaData(nPointsInner, quadWeightsInner, quadPointsInner);
  quadFormulaOuter.GetFormulaData(nPointsOuter, quadWeightsOuter, quadPointsOuter);
  
  const int maxNPoints = 64;
  
  if(nPointsOuter > maxNPoints)
    ErrThrow("Number of quadrature points of the requested outer quadrature formula"
    " is too high, namely ", nPointsOuter, ". The maximum is", maxNPoints);
  
  if(nPointsInner > maxNPoints)
    ErrThrow("Number of quadrature points of the requested inner quadrature formula"
    " is too high, namely ", nPointsInner, ". The maximum is", maxNPoints);
  
  
  // set data structures for evaluating problem coefficients
  const int nParams = 5;
  double coeffs[maxNPoints][nParams];
  double* cs[maxNPoints];
  
  for(int i = 0; i < maxNPoints; i++)
    cs[i] = coeffs[i];
  
  const int maxNPointsPlusOneForEdgeMidpoint = maxNPoints + 1;
  double xToEvaluateInner[maxNPointsPlusOneForEdgeMidpoint];
  double yToEvaluateInner[maxNPointsPlusOneForEdgeMidpoint];
  double xToEvaluateOuter[maxNPointsPlusOneForEdgeMidpoint];
  double yToEvaluateOuter[maxNPointsPlusOneForEdgeMidpoint];
  const CoeffFct2D& bilinearCoeffs = example.get_coeffs();
  
  
  // future minima and maxima of exponents in exponentials
  double psiMaxGlobal = -1.0e-10;
  double psiMinGlobal = 1.0e-10;
  
  
  // loop over triangle edges to assemble the system matrix
  const TCollection* collPtr = get_space()->GetCollection();
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    TBaseCell* cellPtr = collPtr->GetCell(cellNr);
    const TShapeDesc* shapeDescPtr = cellPtr->GetShapeDesc();
    const int* EdgeVertex;
    shapeDescPtr->GetEdgeVertex(EdgeVertex);
    
    std::vector<int> globalDOF(3);
    const int *globalDOFint = get_space()->GetGlobalDOF(cellPtr->GetCellIndex());
    
    for(unsigned int i = 0; i < globalDOF.size(); i++)
      globalDOF[i] = globalDOFint[i];
    
    TGridCell* gridCellPtr = (TGridCell*)cellPtr;
    
    double cellMeasure = gridCellPtr->GetMeasure();
    
    
    for(int jointId = 0; jointId < cellPtr->GetN_Joints(); jointId++)
    {
      const TJoint* jointPtr = cellPtr->GetJoint(jointId);
      
      // get all indices and coordinates of all vertices
      int vertexId_i = EdgeVertex[jointId*2];
      const TVertex* vertexPtr_i = gridCellPtr->GetVertex(vertexId_i);
      
      int vertexId_j = EdgeVertex[jointId*2 + 1];
      const TVertex* vertexPtr_j = gridCellPtr->GetVertex(vertexId_j);
      
      int vertexId_k = 3 - vertexId_i - vertexId_j;
      const TVertex* vertexPtr_k = gridCellPtr->GetVertex(vertexId_k);
      
      // parameters for the outer integral, \int_E \exp{\psi_E^T} / \alpha
      double x_i, x_j, x_k, y_i, y_j, y_k;
      vertexPtr_i->GetCoords(x_i, y_i);
      vertexPtr_j->GetCoords(x_j, y_j);
      vertexPtr_k->GetCoords(x_k, y_k);
      
      
      // integral of reaction term, \int_T \gamma uv
      // we do not assemble dirichlet rows
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        bilinearCoeffs(1, &x_i, &y_i, nullptr, cs);
        double valGamma = coeffs[0][3];
        
        double productGamma = cellMeasure * valGamma / (d + 1.0);
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], productGamma);
      }
      
      
      // preparing lengths of all edges
      double edgeLength_ij = std::sqrt( (x_j - x_i) * (x_j - x_i) + (y_j - y_i) * (y_j - y_i) );
      double edgeLength_ik = std::sqrt( (x_k - x_i) * (x_k - x_i) + (y_k - y_i) * (y_k - y_i) );
      double edgeLength_jk = std::sqrt( (x_k - x_j) * (x_k - x_j) + (y_k - y_j) * (y_k - y_j) );
      
      double edgeLengthOuter = edgeLength_ij;
      
      // \tau_E = directional vector of the current edge E
      double tauE_x = x_i - x_j;
      double tauE_y = y_i - y_j;
      double tauE_length = std::sqrt( tauE_x * tauE_x + tauE_y * tauE_y );
      
      
      // lengths for computing integral over edge E
      // reference interval is [-1, 1]
      const double refEdgeLength = 2.0;
      const double refIntervalStart = - 1.0;
      double transformDetOuter = edgeLengthOuter/refEdgeLength;
      
      // computation of the outer integral + of \psi_E^T (which is defined by an integral)
      double integralOuter = 0.0;
      double integralPsiE = 0.0;
      std::vector<double> integralsPsiInner(nPointsOuter, 0.0);
      
      
      // for computing minima and maxima of exponents in exponentials
      double psiEMin = 1.0e10;
      double psiEMax = - 1.0e10;
      
      
      // precalculating quadrature point for evaluating coefficients in outer integral
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      {
        double x_outer = x_j + (x_i - x_j) * ( quadPointsOuter[kOuter] - refIntervalStart )
                           / refEdgeLength;
        double y_outer = y_j + (y_i - y_j) * ( quadPointsOuter[kOuter] - refIntervalStart )
                           / refEdgeLength;
                           
        xToEvaluateOuter[kOuter] = x_outer;
        yToEvaluateOuter[kOuter] = y_outer;
      }
      
      
      // computing inner integrals, \int_0^s \frac{\beta \cdot \tau_E}{\alpha},
      // where in bounds of integral,
      // "0" corresponds to node "j" and "s" corresponds to outer quadrature point
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      {
        // setting end points and lengths
        double x_j_inner = x_j;
        double y_j_inner = y_j;
        double x_i_inner = xToEvaluateOuter[kOuter];
        double y_i_inner = yToEvaluateOuter[kOuter];
        
        double edgeLengthInner = std::sqrt(  (x_j_inner - x_i_inner) * (x_j_inner - x_i_inner)
                                           + (y_j_inner - y_i_inner) * (y_j_inner - y_i_inner) );
        double transformDetInner = edgeLengthInner/refEdgeLength;
        
        
        // precalculating quadrature points for evaluating coefficients
        for(int kInner = 0; kInner < nPointsInner; kInner++)
        {
          // x_inner, y_inner: points for evaluating \alpha and \beta
          double x_inner = x_j_inner + (x_i_inner - x_j_inner)
                           * ( quadPointsInner[kInner] - refIntervalStart ) / refEdgeLength;
          double y_inner = y_j_inner + (y_i_inner - y_j_inner)
                           * ( quadPointsInner[kInner] - refIntervalStart ) / refEdgeLength;
                           
          xToEvaluateInner[kInner] = x_inner;
          yToEvaluateInner[kInner] = y_inner;
        }
        
        // precalculating coefficient values
        bilinearCoeffs(nPointsInner, xToEvaluateInner, yToEvaluateInner, nullptr, cs);
        
        
        // finally computing inner integrals
        for(int kInner = 0; kInner < nPointsInner; kInner++)
        {
          double valAlpha = coeffs[kInner][0];
          double valBeta_x = - coeffs[kInner][1];
          double valBeta_y = - coeffs[kInner][2];
          
          double integrandPsiInner = ( valBeta_x * tauE_x + valBeta_y * tauE_y ) / valAlpha;
          
          integralsPsiInner[kOuter] += quadWeightsInner[kInner] * integrandPsiInner * transformDetInner;
        }
        integralsPsiInner[kOuter] /= tauE_length;
        
        if(integralsPsiInner[kOuter] > psiEMax)
          psiEMax = integralsPsiInner[kOuter];
        
        if(integralsPsiInner[kOuter] < psiEMin)
          psiEMin = integralsPsiInner[kOuter];
      }
      
      
      // computing \int_E \exp{\psi_E_T}
      
      // precalculating coefficient values
      bilinearCoeffs(nPointsOuter, xToEvaluateOuter, yToEvaluateOuter, nullptr, cs);
      
      // finally computing \int_E \exp{\psi_E_T}
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      { 
        double valAlpha = coeffs[kOuter][0];
        double valBeta_x = - coeffs[kOuter][1];
        double valBeta_y = - coeffs[kOuter][2];
        
        double integrandPsiE = ( valBeta_x * tauE_x + valBeta_y * tauE_y ) / valAlpha;
        integralPsiE += quadWeightsOuter[kOuter] * integrandPsiE * transformDetOuter;
      }
      integralPsiE /= tauE_length;
      
      if(integralPsiE > psiEMax)
        psiEMax = integralPsiE;
        
      if(integralPsiE < psiEMin)
        psiEMin = integralPsiE;
      
      if(psiMaxGlobal < psiEMax)
        psiMaxGlobal = psiEMax;
      
      if(psiMinGlobal > psiEMin)
        psiMinGlobal = psiEMin;
      
      integralPsiE += - psiEMax;
      
      
      // computing outer integral, \int_E \frac{ \exp{\psi_E_T} }{\alpha}
      // it holds \tilde{\alpha}^T_E (\beta) = |\tau_E| / (outer integral)
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      {
        integralsPsiInner[kOuter] += - psiEMax;
        
        double valAlpha = coeffs[kOuter][0];
        double integrandOuter = std::exp( integralsPsiInner[kOuter] ) / valAlpha;
        
        integralOuter += quadWeightsOuter[kOuter] * integrandOuter * transformDetOuter;
      }
      
      if(integralOuter == 0.0)
      {
        ErrThrow("EAFE: Integral corresponding to cell ", cellNr, " and joint ", jointId,
                 " in definition of \\alpha^tilde_ET is zero,"
                 " which produces non-finite values in results."
                 " This most probably means that you mesh is too coarse.");
      }
      
      
      // computing terms in the formula for a_ij
      double alphaTilde_ET = tauE_length / integralOuter;
      const double absKappa_ET = 1.0;
      
      // computing angle \theta^T_E via law of cosines
      double cosValue = (  edgeLength_ij * edgeLength_ij - edgeLength_ik * edgeLength_ik
                         - edgeLength_jk * edgeLength_jk ) / ( -2.0 * edgeLength_ik
                                                                    * edgeLength_jk );
      double theta_ijT = std::acos( cosValue );
      
      const double nFrac = 1.0 / (d * (d - 1.0 ) );
      double omega_ET = nFrac * absKappa_ET / std::tan(theta_ijT);
      double delta_E_exp_i_part = std::exp(integralPsiE);
      double delta_E_exp_j_part = - std::exp( - psiEMax);
      
      if(!std::isfinite(delta_E_exp_i_part) || !std::isfinite(delta_E_exp_j_part))
      {
        ErrThrow("EAFE: exp(\\psi) corresponding to cell ", cellNr, " and joint ", jointId,
                 " is not finite, which produces non-finite values in resutls."
                 "This most probably means that you mesh is too coarse.");
      }
      
      
      double delta_E_v, product;
      
      
      // we do not assamble dirichlet rows
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        // row = i, col = i
        delta_E_v = 1.0;
        product = omega_ET * alphaTilde_ET * delta_E_exp_i_part * delta_E_v;
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], product);
        
        // row = i, col = j
        delta_E_v = 1.0;
        product = omega_ET * alphaTilde_ET * delta_E_exp_j_part * delta_E_v;
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_j], product);
      }
      
      if(globalDOF[vertexId_j] < nActiveRows)
      {
        // row = j, col = i
        delta_E_v = - 1.0;
        product = omega_ET * alphaTilde_ET * delta_E_exp_i_part * delta_E_v;
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_i], product);
        
        // row = j, col = j
        delta_E_v = - 1.0;
        product = omega_ET * alphaTilde_ET * delta_E_exp_j_part * delta_E_v;
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_j], product);
      }
      
      
      // computation of the boundary integral
      // \int_{\Gamma_N} - \beta \cdot \vec{n} u
      // that is omitted while assembling contributions from Neumann
      // boundary conditions due to different original PDE in parmoon
      // integral assambled via lumping
      // - without lumping, oscillations on Neumann boundary may arise
      if( jointPtr->InnerJoint() ||
          (globalDOF[vertexId_i] > nActiveRows && globalDOF[vertexId_j] > nActiveRows) )
        continue;
      
      const TBoundEdge* boundEdgePtr = (const TBoundEdge*)jointPtr;
      double t0,t1;
      double x0, y0, x1, y1;
      boundEdgePtr->GetParameters(t0, t1);
      boundEdgePtr->get_vertices(x0, y0, x1, y1);
      
      const FiniteElement& finiteElement = matPtr->GetTestSpace()->get_fe(cellNr);
      const BaseFunctions* baseFunctionsPtr = finiteElement.GetBaseFunct();
      
      double transformDet = tauE_length / refEdgeLength;
      
      // compute unit outward normal to boundary edge
      double normal_x = - tauE_y / tauE_length;
      double normal_y = tauE_x / tauE_length;
      double tauEki_x = x_k - x_i;
      double tauEki_y = y_k - y_i;
      
      if(normal_x * tauEki_x + normal_y * tauEki_y >= 0)
      {
        normal_x = - normal_x;
        normal_y = - normal_y;
      }
      
      
      // precomputing values of basis functions at quadrature points
      double** valsQuadPointsNeu;
      valsQuadPointsNeu = FEDatabase::GetJointDerivatives2D(*baseFunctionsPtr,
                                                            quadFormulaInner, jointId, MultiIndex2D::D00);
      baseFunctionsPtr->ChangeBF(collPtr, cellPtr, nPointsInner, valsQuadPointsNeu);
      
      
      // preparing quadrature points and coefficient values
      for(int kInner = 0; kInner < nPointsInner; kInner++)
      {
        // x_inner, y_inner: points for evaluating \beta
        double x_inner = x0 + (x1 - x0)
                         * ( quadPointsInner[kInner] - refIntervalStart ) / refEdgeLength;
        double y_inner = y0 + (y1 - y0)
                         * ( quadPointsInner[kInner] - refIntervalStart ) / refEdgeLength;
                         
        xToEvaluateInner[kInner] = x_inner;
        yToEvaluateInner[kInner] = y_inner;
      }
      
      bilinearCoeffs(nPointsInner, xToEvaluateInner, yToEvaluateInner, nullptr, cs);
      
      
      // computing \int_{\Gamma_N} - \beta \cdot \vec{n} \psi_i
      double edgeIntegrals[2] = {0.0, 0.0};
      
      for(int kInner = 0; kInner < nPointsInner; kInner++)
      {
        double valBeta_x = - coeffs[kInner][1];
        double valBeta_y = - coeffs[kInner][2];
        double commonTerm = - quadWeightsInner[kInner] * transformDet
                          * (valBeta_x * normal_x + valBeta_y * normal_y);
        
        edgeIntegrals[0] += valsQuadPointsNeu[kInner][vertexId_i] * commonTerm;
        edgeIntegrals[1] += valsQuadPointsNeu[kInner][vertexId_j] * commonTerm;
      }
      
      
      // adding contributions into matrix
      // we do not assamble dirichlet rows
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], edgeIntegrals[0]);
      }
      if(globalDOF[vertexId_j] < nActiveRows)
      {
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_j], edgeIntegrals[1]);
      }
      
    }
  }
  
  Output::print("Assembly with general EAFE done; psiMax = ", psiMaxGlobal, ", psiMin = ", psiMinGlobal);
}
#endif

/* ************************************************************************* */
#ifdef __2D__
template<int d>
QuadratureFormula_type ConvectionDiffusion<d>::getQuadFormType(int formNumber)
{
  QuadratureFormula_type quadForm;
  switch(formNumber)
  {
    case 16:
      quadForm = QuadratureFormula_type::Gauss16Line;
      break;
    case 32:
      quadForm = QuadratureFormula_type::Gauss32Line;
      break;
    case 50:
      quadForm = QuadratureFormula_type::Gauss50Line;
      break;
    case 60:
      quadForm = QuadratureFormula_type::Gauss60Line;
      break;
    case 64:
      quadForm = QuadratureFormula_type::Gauss64Line;
      break;
    default:
      quadForm = (QuadratureFormula_type)(formNumber - 1);
  };
  
  return quadForm;
}

template<int d>
double ConvectionDiffusion<d>::bernoulliValueTimesAlpha(double betaTau, double alpha)
{
  const double tol = 1.0e-16;
  
  double s = betaTau / alpha;
  
  if(std::abs(s) < tol)
    return alpha;
  else
    return betaTau / (std::exp(s) - 1.0);
}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::assembleEAFEConstBetaOnEdges(int quadOuter)
{ 
//   // assembly of rhs
//   std::function<void(LocalAssembling<d>&)> modify_la( 
//   [&](LocalAssembling<d>& la)
//   {
//     std::vector<AssembleFctParam> laf = { CDRHS<d> };
//     la.replace_local_assembling(laf);
//   });
//   
  FEMatrix* matPtr = get_matrix().get_blocks_uniquely().at(0).get();
  const int nActiveRows = matPtr->get_n_active_rows();
//   assemble(modify_la);
  
  
  // set the requested quadrature formula for line integrals
  QuadratureFormula_type quadOuterForm = getQuadFormType(quadOuter);
  TQuadFormula quadFormulaOuter(quadOuterForm);
  
  int nPointsOuter;
  const int maxNPointsOuter = 64;
  const double* quadPointsOuter;
  const double* quadWeightsOuter;
  quadFormulaOuter.GetFormulaData(nPointsOuter, quadWeightsOuter, quadPointsOuter);
  
  if(nPointsOuter > maxNPointsOuter)
    ErrThrow("Number of quadrature points of the requested quadrature formula"
    " is too high, namely ", nPointsOuter, ". The maximum is", maxNPointsOuter);
  
  // set data structures for evaluating problem coefficients
  const int nParams = 5;
  double coeffs[maxNPointsOuter][nParams];
  double* cs[maxNPointsOuter];
  
  for(int i = 0; i < maxNPointsOuter; i++)
    cs[i] = coeffs[i];
  
  const int maxNPointsPlusOneForEdgeMidpoint = maxNPointsOuter + 1;
  double xToEvaluate[maxNPointsPlusOneForEdgeMidpoint];
  double yToEvaluate[maxNPointsPlusOneForEdgeMidpoint];
  const CoeffFct2D& bilinearCoeffs = example.get_coeffs();
  
  
  // loop over triangle edges to assemble the system matrix
  const TCollection* collPtr = get_space()->GetCollection();
  
  for(int cellNr = 0; cellNr < collPtr->GetN_Cells(); cellNr++)
  {
    TBaseCell* cellPtr = collPtr->GetCell(cellNr);
    const TShapeDesc* shapeDescPtr = cellPtr->GetShapeDesc();
    const int* EdgeVertex;
    shapeDescPtr->GetEdgeVertex(EdgeVertex);
    
    std::vector<int> globalDOF(3);
    const int *globalDOFint = get_space()->GetGlobalDOF(cellNr);
    
    for(unsigned int i = 0; i < globalDOF.size(); i++)
      globalDOF[i] = globalDOFint[i];
    
    
    TGridCell* gridCellPtr = (TGridCell*)cellPtr;
    
    double cellMeasure = gridCellPtr->GetMeasure();
    
    
    for(int jointId = 0; jointId < cellPtr->GetN_Joints(); jointId++)
    {
      const TJoint* jointPtr = cellPtr->GetJoint(jointId);
       
      // get all indices and coordinates of all vertices
      int vertexId_i, vertexId_j, vertexId_k;
      TVertex* vertexPtr_i;
      TVertex* vertexPtr_j;
      TVertex* vertexPtr_k;
      
      vertexId_i = EdgeVertex[jointId*2];
      vertexId_j = EdgeVertex[jointId*2 + 1];
      
      vertexPtr_i = gridCellPtr->GetVertex(vertexId_i);
      vertexPtr_j = gridCellPtr->GetVertex(vertexId_j);
      
      vertexId_k = 3 - vertexId_i - vertexId_j;
      vertexPtr_k = gridCellPtr->GetVertex(vertexId_k);
      
      // parameters for the outer integral, \int_E \exp{\psi_E^T} / \alpha
      double x_i, x_j, x_k, y_i, y_j, y_k;
      vertexPtr_i->GetCoords(x_i, y_i);
      vertexPtr_j->GetCoords(x_j, y_j);
      vertexPtr_k->GetCoords(x_k, y_k);
      
      
      // integral of reaction term, \int_T \gamma uv
      // we do not assemble dirichlet rows
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        bilinearCoeffs(1, &x_i, &y_i, nullptr, cs);
        double valGamma = coeffs[0][3];
        
        double productGamma = cellMeasure * valGamma / (d + 1.0);
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], productGamma);
      }
      
      
      // preparing lengths of all edges
      double edgeLength_ij = std::sqrt( (x_j - x_i) * (x_j - x_i) + (y_j - y_i) * (y_j - y_i) );
      double edgeLength_ik = std::sqrt( (x_k - x_i) * (x_k - x_i) + (y_k - y_i) * (y_k - y_i) );
      double edgeLength_jk = std::sqrt( (x_k - x_j) * (x_k - x_j) + (y_k - y_j) * (y_k - y_j) );
      
      double edgeLengthOuter = edgeLength_ij;
      
      
      // \tau_E = directional vector of the current edge E and its midpoint
      double tauE_x = x_i - x_j;
      double tauE_y = y_i - y_j;
      double tauE_length = std::sqrt( tauE_x * tauE_x + tauE_y * tauE_y );
      
      double eCenter_x = (x_i + x_j) / 2.0;
      double eCenter_y = (y_i + y_j) / 2.0;
      
      
      // prepare quadrature points and lengths for computing integral over edge E
      // reference interval is [-1, 1]
      const double refEdgeLength = 2.0;
      const double refIntervalStart = - 1.0;
      const double transformDetOuter = edgeLengthOuter/refEdgeLength;
      
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      {
        double x_outer = x_j + (x_i - x_j) * ( quadPointsOuter[kOuter] - refIntervalStart )
                          / refEdgeLength;
        double y_outer = y_j + (y_i - y_j) * ( quadPointsOuter[kOuter] - refIntervalStart )
                          / refEdgeLength;
        xToEvaluate[kOuter] = x_outer;
        yToEvaluate[kOuter] = y_outer;
      }
        
        
      // computing \beta \times \tau_E,
      xToEvaluate[nPointsOuter] = eCenter_x;
      yToEvaluate[nPointsOuter] = eCenter_y;
      
      bilinearCoeffs(nPointsOuter + 1, xToEvaluate, yToEvaluate, nullptr, cs);
      
      double valBeta_x = - coeffs[nPointsOuter][1];
      double valBeta_y = - coeffs[nPointsOuter][2];
      
      double tauBeta = valBeta_x * tauE_x + valBeta_y * tauE_y;
      
      
      // computation of \int_E \frac{1}{\alpha} denoted as "integralOuter"
      double integralOuter = 0.0;
      
      for(int kOuter = 0; kOuter < nPointsOuter; kOuter++)
      {
        double valAlpha = coeffs[kOuter][0];
        double integrandOuter = 1.0 / valAlpha;
        integralOuter += quadWeightsOuter[kOuter] * integrandOuter * transformDetOuter;
      }
      
      
      // terms in the formula for a_ij
      double alphaTilde_ET = tauE_length / integralOuter;
      
      double alphaBMinus = bernoulliValueTimesAlpha( - tauBeta, alphaTilde_ET );
      double alphaBPlus = bernoulliValueTimesAlpha( tauBeta, alphaTilde_ET );
      
      const double absKappa_ET = 1.0;
      
      // computing angle \theta^T_E via law of cosines
      double cosValue = (  edgeLength_ij * edgeLength_ij - edgeLength_ik * edgeLength_ik
                         - edgeLength_jk * edgeLength_jk ) / ( -2.0 * edgeLength_ik
                                                                    * edgeLength_jk );
      double theta_ijT = std::acos( cosValue );
      
      const double nFrac = 1.0 / (d * (d - 1.0 ) );
      double omega_ET = nFrac * absKappa_ET / std::tan(theta_ijT);
      
      
      double product;
      
      // adding contributions into matrix
      // we do not assamble dirichlet rows
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        // row = i, col = i
        product = omega_ET * alphaBMinus;
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], product);
        
        // row = i, col = j
        product = omega_ET * alphaBPlus;
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_j], - product);
      }
      
      if(globalDOF[vertexId_j] < nActiveRows)
      {
        // row = j, col = i
        product = omega_ET * alphaBMinus;
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_i], - product);
        
        // row = j, col = j
        product = omega_ET * alphaBPlus;
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_j], product);
      }
        
        
      // computation of the boundary integral
      // \int_{\Gamma_N} - \beta \cdot \vec{n} u
      // that is omitted while assembling contributions from Neumann
      // boundary conditions due to different original PDE in parmoon
      // integral assambled via lumping
      // - without lumping, oscillations on Neumann boundary may arise
      if( jointPtr->InnerJoint() ||
          (globalDOF[vertexId_i] >= nActiveRows && globalDOF[vertexId_j] >= nActiveRows) )
        continue;
      
      // compute unit outward normal to boundary edge
      double normal_x = - tauE_y / tauE_length;
      double normal_y = tauE_x / tauE_length;
      double tauEki_x = x_k - x_i;
      double tauEki_y = y_k - y_i;
      
      if(normal_x * tauEki_x + normal_y * tauEki_y >= 0)
      {
        normal_x = - normal_x;
        normal_y = - normal_y;
      }
      
      double lumpedBoundaryIntegral = - (valBeta_x * normal_x + valBeta_y * normal_y)
                                      * 0.5 * tauE_length;
      
      if(globalDOF[vertexId_i] < nActiveRows)
      {
        matPtr->add(globalDOF[vertexId_i], globalDOF[vertexId_i], lumpedBoundaryIntegral);
      }
      if(globalDOF[vertexId_j] < nActiveRows)
      {
        matPtr->add(globalDOF[vertexId_j], globalDOF[vertexId_j], lumpedBoundaryIntegral);
      }
      
    }// loop over joints
  }// loop over cells
  
  Output::print("Assembly with EAFE approximating beta as constant on edges done");
}
#endif

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::solve()
{
  #ifdef __2D__
  if(mizukamiHughesPtr)
  {
    solveNonlinear();
    return;
  }
  #endif
  
  double t = GetTime();
  SystemPerGrid& s = this->systems.front();
#ifndef _MPI
  this->solver.solve(s.matrix, s.rhs, s.solution);
#else
  if(this->solver.get_db()["solver_type"].is("direct"))
  {
    MumpsWrapper mumps_wrapper(s.matrix);
    mumps_wrapper.solve(s.rhs, s.solution);
  }
  else
    this->solver.solve(s.matrix, s.rhs, s.solution);
#endif
  t = GetTime() - t;
  Output::print<3>("solving time of a ConvectionDiffusion<", d, "> problem: ",
                   t, " seconds");

  // Apply limiter
#ifdef __2D__
  limiter.limit_function(&this->get_function());
#endif
}

/* ************************************************************************* */
#ifdef __2D__
template<int d>
void ConvectionDiffusion<d>::solveNonlinear()
{
  double residualOld = -1.0;
  BlockVector residualVect;
  residualVect.copy_structure(get_rhs());
  
  auto computeResidual = [&]() -> double
  {
    residualVect = get_rhs();
    this->get_matrix().apply_scaled_add(this->get_solution(), residualVect, -1.0);
    return residualVect.norm();
  };
  
  auto diffOfVectors = [](const BlockVector& vec1, const BlockVector& vec2) -> double
  {
    double diff = 0.0;
    const double* vec1Entries = vec1.get_entries();
    const double* vec2Entries = vec2.get_entries();
    
    for(unsigned int i = 0; i < vec1.length(); i++)
    {
      const double brac = vec1Entries[i] - vec2Entries[i];
      diff += brac*brac;
    }
    return std::sqrt(diff);
  };
  
  
  double residualNew;
  
  // 0. iteration to get initial solution in case that we do not start from zero vector
  if(mizukamiHughesPtr && mizukamiHughesPtr->doAssemble())
  {
    Output::print<2>("Computing initial solution.");
    solver.solve(get_matrix(), get_rhs(), get_solution());
    
    residualNew = computeResidual();
    Output::print<2>("Initial residual: ", residualNew);
  }
  
  BlockVector solutionOld = get_solution();
  
  
  // subsequent iterations
//   const double normalizedStoppingTolerance = (double)db["stopping_tolerance_cd"];
  const double normalizedStoppingTolerance = (double)db["stopping_tolerance_cd"]
                                           * std::sqrt(get_space()->get_n_dof());
  Output::print<2>("stopping_tolerance_cd normalized to ", normalizedStoppingTolerance);
  
  
  const int itMax = db["it_max_cd"];
  const bool constantDamping = db["constant_damping_cd"];
  
  // names of variables for dynamic damping used in [JK08.CMAME]
  double omegaMax = db["damping_factor_omega_max_cd"];
  double omega = omegaMax;
  const double c1 = db["damping_parameter_c1_cd"];
  const double c2 = db["damping_parameter_c2_cd"];
  const double c3 = db["damping_parameter_c3_cd"];
  const double c4 = db["damping_parameter_c4_cd"];
  const double omegaMin = db["damping_factor_omega_min_cd"];
  bool firstDamp = true;
  
//   double residualNew;
  double solutionDiff;
  int rejectedSteps = 0;
  
  assemble();
  residualNew = computeResidual();
  
  for(int itNr = 0; itNr <= itMax; itNr++)
  {
    solutionDiff = diffOfVectors(get_solution(), solutionOld);
    Output::print<2>("nonlinear step ", itNr, " residual: ", residualNew, 
                     " ratio of residuals ", residualNew / residualOld,
                     " change in solution ", solutionDiff);
    
    
    if(residualNew < normalizedStoppingTolerance && itNr >= 1)
    {
      Output::print<1>("NONLINEAR ITERATION: Nr. ", itNr, " residual ", 
                       residualNew, " rejections ", rejectedSteps);
      return;
    }

    solutionOld = get_solution();
    residualOld = residualNew;
    
    solver.solve(get_matrix(), get_rhs(), get_solution());
    
      
    if(constantDamping)
    {
      updateSolution(solutionOld, omega);
      
      assemble();
      residualNew = computeResidual();
    }
    else
    {
      
      // dynamic damping from [JK08.CMAME]
      firstDamp = true;
      
      while(1)
      {
        updateSolution(solutionOld, omega);
        assemble();
        residualNew = computeResidual();
        
        if(residualNew < residualOld || omega <= c1 * omegaMin)
        {
          // iteration accepted
          if(residualNew < residualOld && firstDamp)
          {
            omegaMax = (1.0 <= c3 * omegaMax) ? 1.0 : c3 * omegaMax;
            omega = (omegaMax <= c2 * omega) ? omegaMax : c2 * omega;
          }
          
          Output::print<2>("iteration accepted, new damping factor ", omega,
                           ", new max damping factor ", omegaMax);
          break;
        }
        else
        {
          // iteration rejected
          rejectedSteps++;
          
          omega = (omegaMin >= omega / 2.0) ? omegaMin : omega / 2.0;
          
          if(firstDamp)
          {
            omegaMax = (omegaMin >= c4 * omegaMax) ? omegaMin : c4 * omegaMax;
            firstDamp = false;
          }
          
          Output::print<2>("iteration rejected, new damping factor ", omega,
                           ", new max damping factor ", omegaMax);
        }
      }// end of while(1)
    }// end of dynamic damping
  }// end of for itNr
  
  Output::print<1>("NONLINEAR ITERATION: MAX NUMBER OF ITERATIONS, ", itMax,
                   ", REACHED: residual ", residualNew, " rejections ", rejectedSteps);
}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::updateSolution(const BlockVector& solutionOld,
                                            double omega)
{
  BlockVector& solutionNew = get_solution();
  
  for(unsigned int i = 0; i < solutionNew.active(0); i++)
    solutionNew[i] = (solutionNew[i] - solutionOld[i]) * omega + solutionOld[i];
  
  solutionNew.copy_nonactive(solutionOld);
}

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::checkMizukamiHughes()
{
  if(mizukamiHughesPtr)
    mizukamiHughesPtr->checkCsForTwoTrianglesRefLevel2();
}
#endif

/* ************************************************************************* */
template<int d>
void ConvectionDiffusion<d>::output(int i)
{
  int my_rank = 0;
#ifdef _MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // computing errors as well as writing vtk files requires a minimum 
  // consistency level of 1
  this->systems.front().fe_space->get_communicator().consistency_update(
    this->systems.front().solution.get_entries(), 1);
#endif

  const FEFunction & fe_function = this->systems.front().fe_function;
  if(db["output_compute_minmax"])
  {
    fe_function.PrintMinMax();
  }
  
  // write solution
  if(i < 0)
    outputWriter.write();
  else
    outputWriter.write(i);
  checkpoint_io.write(this->get_solution());

  // measure errors to known solution
  // If an exact solution is not known, it is usually set to be zero, so that
  // in such a case here only integrals of the solution are computed.
  if(db["output_compute_errors"])
  {
    std::array<double, ConvectionDiffusion<d>::n_errors> errors;
#ifdef __3D__
    TAuxParam3D aux;
    MultiIndex3D AllDerivatives[4] = { MultiIndex3D::D000, MultiIndex3D::D100,
                                       MultiIndex3D::D010, MultiIndex3D::D001 };
#else
    TAuxParam2D aux;
    MultiIndex2D AllDerivatives[3] = { MultiIndex2D::D00, MultiIndex2D::D10,
                                       MultiIndex2D::D01 };
#endif
    const FESpace* space = this->systems.front().fe_space.get();

#ifdef __3D__
    fe_function.GetErrors(example.get_exact(0), d+1, AllDerivatives,
                          ConvectionDiffusion<d>::n_errors,
                          conv_diff_l2_h1_linf_error<d>, example.get_coeffs(),
                          &aux, 1, &space, errors.data());
#else
    fe_function.GetErrors(example.get_exact(0), d+1, AllDerivatives,
                          ConvectionDiffusion<d>::n_errors,
                          conv_diff_l2_h1_linf_error<d>, example.get_coeffs(),
                          &aux, 1, &space, errors.data(), false,[](const TBaseCell*, int){return false;}, db);
#endif

#ifdef _MPI
    // memory for global (across all processes) error
    std::array<double, ConvectionDiffusion<d>::n_errors> errorsReduced;
    MPI_Reduce(errors.data(), errorsReduced.data(),
               ConvectionDiffusion<d>::n_errors-1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&errors[ConvectionDiffusion<d>::n_errors-1],
               &errorsReduced[ConvectionDiffusion<d>::n_errors-1], 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    for(int i = 0; i < ConvectionDiffusion<d>::n_errors-1; i++) // exclude l_inf
      errors[i] = std::sqrt(errorsReduced[i]);
    errors[ConvectionDiffusion<d>::n_errors-1] 
      = errorsReduced[ConvectionDiffusion<d>::n_errors-1];
#endif
    // copy local variable to member variable
    std::copy(errors.begin(), errors.end(), this->errors.begin());


    //print errors
    if(my_rank == 0)
    {
      Output::print<1>("L2     : ", setprecision(14), errors[0]);
      Output::print<1>("H1-semi: ", setprecision(14), errors[1]);
      Output::print<1>("SD     : ", setprecision(14), errors[2]);
      if(db["space_discretization_type"].is("dg"))
        Output::print<1>("DG     : ", setprecision(14), errors[3]);
      Output::print<1>("L_inf  : ", setprecision(14), errors[4]);
    }
  } // if(this->db["compute_errors"])

  // do postprocessing
  example.do_post_processing(*this);
}

/** ************************************************************************ */
template<int d>
double ConvectionDiffusion<d>::get_L2_error() const
{
  return this->errors[0];
}

/** ************************************************************************ */
template<int d>
double ConvectionDiffusion<d>::get_H1_semi_error() const
{
  return this->errors[1];
}

/** ************************************************************************ */
template<int d>
double ConvectionDiffusion<d>::get_SD_error() const
{
  return this->errors[2];
}

/** ************************************************************************ */
template<int d>
double ConvectionDiffusion<d>::get_DG_error() const
{
  return this->errors[3];
}

/** ************************************************************************ */
template<int d>
double ConvectionDiffusion<d>::get_L_inf_error() const
{
  return this->errors[4];
}

/** ************************************************************************ */
template<int d>
void ConvectionDiffusion<d>::call_assembling_routine(SystemPerGrid& s,
						     LocalAssembling<d>& local_assem, bool assemble_dirichlet_rows)
{
  // assemble the system matrix with given local assembling, solution and rhs
  using SquareMatrixD = typename Template_names<d>::SquareMatrixD;
  int n_fe_spaces = 1;
  const FESpace * fe_space = s.fe_space.get();
  auto * boundary_conditions = fe_space->get_boundary_condition();
  int n_square_matrices = 1;
  int n_rect_matrices = 0;
  double * rhs_entries = s.rhs.get_entries();
  int n_rhs = 1;
  auto * non_const_bound_value = example.get_bd(0);

  //fetch stiffness matrix as block
  auto blocks = s.matrix.get_blocks_uniquely();
  SquareMatrixD* block = reinterpret_cast<SquareMatrixD*>(blocks.at(0).get());

  // reset right hand side and matrix to zero
  s.rhs.reset();
  block->reset();
  // and call the assembling method
#ifdef __3D__
  Assemble3D(
#else
  Assemble2D(
#endif
    n_fe_spaces, &fe_space, n_square_matrices, &block,
    n_rect_matrices, nullptr, n_rhs, &rhs_entries, &fe_space,
    &boundary_conditions, &non_const_bound_value, local_assem,
    assemble_dirichlet_rows);
  // copy Dirichlet values from rhs to solution vector (this is not really
  // necessary in case of a direct solver)
  s.solution.copy_nonactive(s.rhs);

  // Assemble DG boundary terms
  if ( db["space_discretization_type"].is("dg") )
  {
    Assemble_DG<d>( local_assem.GetCoeffFct(), n_fe_spaces, &fe_space,
      n_square_matrices, &block, n_rect_matrices, nullptr, n_rhs,
      &rhs_entries, &boundary_conditions, &non_const_bound_value, db, "CD" );
  }
  
#ifdef __2D__
  if( db["space_discretization_type"].is("EAFE_general") )
  {
    const int quadInnerForm = this->db["EAFE_inner_quad_formula"];
    const int quadOuterForm = this->db["EAFE_outer_quad_formula"];
    
    assembleGeneralEAFE(quadInnerForm, quadOuterForm);
  }
  else if( db["space_discretization_type"].is("EAFE_beta_const_on_edges") )
  {
    const int quadOuterForm = this->db["EAFE_outer_quad_formula"];
    
    assembleEAFEConstBetaOnEdges(quadOuterForm);
  }
  else if( db["space_discretization_type"].is("upwind") )
  {
    UpwindForConvDiff(local_assem.GetCoeffFct(), block, s.rhs.get_entries(),
                      s.fe_space.get(),nullptr, nullptr, false);
  }
#endif

}


#ifdef __3D__
template class ConvectionDiffusion<3>;
#else
template class ConvectionDiffusion<2>;
#endif
