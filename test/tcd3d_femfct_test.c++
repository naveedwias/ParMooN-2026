/**
 * @brief A test program for a time dependent CD3D problem
 * algorithm "FEM-FCT", sequential and parallel versions
 *
 * @author Clemens Bartsch, Naveed Ahmed, Ondrej Partl
 */
#include "TimeConvectionDiffusion.h"
#include <Database.h>
#include <TimeDiscRout.h>
#include <TimeDiscretizations.h>
#include <MainUtilities.h>
#include <ConvDiff.h>
#include "AuxParam3D.h"
#include "ParMooN.h"
#include <cmath>

#include <ParMooN_repository_info.h>
const std::string path = parmoon::source_directory;
const std::string block_mesh_file = path + "/data/mesh/" + "block_1x2x3.mesh";

#ifdef _MPI
#include <mpi.h>
#include <MeshPartition.h>
#endif

enum TestNames
{
  LinearWithDirichletCrankNicolson,
  LinearWithDirichletNeumannCrankNicolson,
  LinearWithDirichletEuler,
  LinearWithDirichletNeumannEuler,
  NonLinearWithDirichletCrankNicolson,
  NonLinearWithDirichletNeumannCrankNicolson,
  NonLinearWithDirichletEuler,
  NonLinearWithDirichletNeumannEuler,
  ConcentrationSpecies,
  NonLinearWithDirichletNeumannEulerMesh,
  NonLinearWithDirichletNeumannCrankNicolsonMesh,
  MonolithicWithDirichlet,
  MonolithicWithDirichletNeumann,
  MolNonlinearWithDirichletCrankNicolson,
  MolNonLinearWithDirichletNeumannCrankNicolson,
  ZalExplicitWithDirichlet,
  ZalExplicitWithDirichletNeumann
};

void prepareErrorDatabase(ParameterDatabase& errorDatabase,
                          std::vector<std::string>& errorNames);
void setCommonParameters(ParameterDatabase& db);
void setTestSpecificParameters(TestNames testName,
                               ParameterDatabase& db,
                               ParameterDatabase& errorDatabase);
void timeIntegration(TimeConvectionDiffusion<3>& tcd,
                     ParameterDatabase& db,
                     ParameterDatabase errorDatabase);
bool differentErrors(double measuredError, double expectedError);
void checkErrors(TimeConvectionDiffusion<3> &tcd,
                 ParameterDatabase& errorDatabase,
                 std::vector<std::string>& errorNames);
void runTest(TestNames testName,
             ParameterDatabase& db,
             ParameterDatabase& errorDatabase,
             std::vector<std::string>& errorNames);
void check_solution_norms(TimeConvectionDiffusion<3> &tcd, int m);
void createMetisFile(const std::string& metisFileName);



int main()
{
  parmoon::parmoon_initialize();
#ifdef _MPI
  TDatabase::ParamDB->Comm = MPI_COMM_WORLD;
#endif
  
  ParameterDatabase db = TimeConvectionDiffusion<3>::default_tcd_database(true);
  
  setCommonParameters(db);
  
  std::vector<std::string>
  errorNames = { "L2(0,T;L2(c))", "L2(0,T;H1(c))", "L_inf(0,T,L_inf(c))" };
  
  ParameterDatabase errorDatabase("database of errors");
  prepareErrorDatabase(errorDatabase, errorNames);
  
  runTest(ConcentrationSpecies, db, errorDatabase, errorNames);
  runTest(LinearWithDirichletCrankNicolson, db, errorDatabase, errorNames);
  runTest(LinearWithDirichletNeumannCrankNicolson, db, errorDatabase, errorNames);
  runTest(LinearWithDirichletEuler, db, errorDatabase, errorNames);
  runTest(LinearWithDirichletNeumannEuler, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletCrankNicolson, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletEuler, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletNeumannCrankNicolson, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletNeumannCrankNicolsonMesh, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletNeumannEuler, db, errorDatabase, errorNames);
  runTest(NonLinearWithDirichletNeumannEulerMesh, db, errorDatabase, errorNames);
  runTest(MonolithicWithDirichlet, db, errorDatabase, errorNames);
  runTest(MonolithicWithDirichletNeumann, db, errorDatabase, errorNames);
  runTest(MolNonlinearWithDirichletCrankNicolson, db, errorDatabase, errorNames);
  runTest(MolNonLinearWithDirichletNeumannCrankNicolson, db, errorDatabase, errorNames);
  runTest(ZalExplicitWithDirichlet, db, errorDatabase, errorNames);
  runTest(ZalExplicitWithDirichletNeumann, db, errorDatabase, errorNames);
  
  parmoon::parmoon_finalize();
}

void setCommonParameters(ParameterDatabase& db)
{
  db["algebraic_flux_correction"].set("fem-fct-cn");
  db["space_discretization_type"] = "galerkin";
  TDatabase::ParamDB->ANSATZ_ORDER = 1;
  db["read_metis"] = true;
}

void prepareErrorDatabase(ParameterDatabase& errorDatabase,
                          std::vector<std::string>& errorNames)
{
  errorDatabase.add("checkOnlyFinalErrors", true, "", {true, false});
  
    for( unsigned int i = 0; i < errorNames.size(); i++)
      errorDatabase.add(errorNames.at(i), 0.0, "", 0.0, 1.0e3);
}

void runTest(TestNames testName,
             ParameterDatabase& db,
             ParameterDatabase& errorDatabase,
             std::vector<std::string>& errorNames)
{
  #ifdef _MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0)
  #endif 
    Output::print("\nStarting test number ", (int)testName, "\n");
  
  setTestSpecificParameters(testName, db, errorDatabase);
  
  std::string metisFileName = db["read_metis_file"];
  
  #ifdef _MPI
  if(rank == 0)
    createMetisFile(metisFileName);
  #endif 
    
  #ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  
  TDatabase::TimeDB->TIMESTEPLENGTH = db["time_step_length"];
  TDatabase::TimeDB->CURRENTTIME = db["time_start"];
  
  TDomain domain(db);
  SetTimeDiscParameters(0);
  domain.refine_and_get_hierarchy_of_collections(db);
  
  #ifdef _MPI
  if(rank == 0)
    std::remove(metisFileName.c_str());
  #endif
  
  TimeConvectionDiffusion<3> tcd(domain, db);
  
  timeIntegration(tcd, db, errorDatabase);
  
  if(errorDatabase["checkOnlyFinalErrors"])
  {
    #ifdef _MPI
    if(rank == 0)
    #endif 
      checkErrors(tcd, errorDatabase, errorNames);
  }
}

void setTestSpecificParameters(TestNames testName,
                               ParameterDatabase& db,
                               ParameterDatabase& errorDatabase)
{
  switch(testName)
  {
    case ConcentrationSpecies:
    db["read_metis_file"].set("Default_UnitCube_Hexa_2_metis.txt", false);
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 0;
    db["solver_type"] = "iterative";
    db["iterative_solver_type"] = "fgmres";
    db["preconditioner"] = "jacobi";
    db["residual_tolerance"] = 1.0e-13;
    db["residual_reduction"] =  0.0;
    db["afc_prelimiter"].set(0);
    db["boundary_file"].set("Default_UnitCube");
    db["geo_file"].set("Default_UnitCube_Hexa");
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.01;
    errorDatabase["checkOnlyFinalErrors"] = false;
    break;
    
    case LinearWithDirichletCrankNicolson:
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(1);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.035914027505992;
    errorDatabase["L2(0,T;H1(c))"] = 0.17860811359288;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.085134085563176;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case LinearWithDirichletNeumannCrankNicolson:
    db["read_metis_file"].set("Default_Block_1x2x3_Tetra_1_metis.txt", false);
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(1);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Tetra");
    db["refinement_n_initial_steps"].set(1);
    db["afc_fct_scheme"].set("linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.55398984224944;
    errorDatabase["L2(0,T;H1(c))"] = 2.379345453276;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.941369333252;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case LinearWithDirichletEuler:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(1);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["time_discretization"] = "backward_euler";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 8.8055574248847e-16;
    errorDatabase["L2(0,T;H1(c))"] = 6.5734837542128e-15;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 3.1086244689504e-15;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case LinearWithDirichletNeumannEuler:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(1);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["time_discretization"] = "backward_euler";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.10804978615762;
    errorDatabase["L2(0,T;H1(c))"] = 0.81313512105758;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.2528267239211;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletCrankNicolson:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(1);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 8.1927982446379e-16;
    errorDatabase["L2(0,T;H1(c))"] = 6.1801625955912e-15;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 2.6645352591004e-15;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletNeumannCrankNicolson:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(2);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_1_metis.txt", false);
    db["refinement_n_initial_steps"].set(1);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.296944423616;
    errorDatabase["L2(0,T;H1(c))"] = 1.4710455398687;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.7573470201824;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletNeumannCrankNicolsonMesh:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(2);
    db["boundary_file"].set("");
    db["geo_file"].set(block_mesh_file, false);
    db["read_metis_file"].set("block_mesh_file_1_metis.txt", false);
    db["refinement_n_initial_steps"].set(1);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.40278680027633;
    errorDatabase["L2(0,T;H1(c))"] = 2.7373489868535;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.8120390944888;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletEuler:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(2);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "backward_euler";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 8.7309879728458e-16;
    errorDatabase["L2(0,T;H1(c))"] = 6.51670407249e-15;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.2166812511657e-30;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletNeumannEuler:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(2);
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Tetra");
    db["read_metis_file"].set("Default_Block_1x2x3_Tetra_1_metis.txt", false);
    db["refinement_n_initial_steps"].set(1);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "backward_euler";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.63648491707003;
    errorDatabase["L2(0,T;H1(c))"] = 2.5904088669627;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.737748120519;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case NonLinearWithDirichletNeumannEulerMesh:
    db["diffusion_coefficient"] = 1e-6;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["afc_prelimiter"].set(2);
    db["boundary_file"].set("");
    db["geo_file"].set(block_mesh_file, false);
    db["read_metis_file"].set("block_mesh_file_1_metis.txt", false);
    db["refinement_n_initial_steps"].set(1);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("zalesak");
    db["solDotUsedInZalesak"] = false;
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "backward_euler";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.40929436572019;
    errorDatabase["L2(0,T;H1(c))"] = 2.6360317633841;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.9358730843547;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case MonolithicWithDirichlet:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("explicit");
    db["afc_limiter"].set("monolithic");
    db["time_discretization"] = "Runge_Kutta_Heun";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.025;
    errorDatabase["L2(0,T;L2(c))"] = 0.1120377565396;
    errorDatabase["L2(0,T;H1(c))"] = 0.61275694556238;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.18740975662506;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case MonolithicWithDirichletNeumann:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Tetra");
    db["read_metis_file"].set("Default_Block_1x2x3_Tetra_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("explicit");
    db["afc_limiter"].set("monolithic");
    db["time_discretization"] = "Runge_Kutta_Heun";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.04;
    errorDatabase["L2(0,T;L2(c))"] = 0.20893863957259;
    errorDatabase["L2(0,T;H1(c))"] = 1.5405249101034;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.82946169315684;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case MolNonlinearWithDirichletCrankNicolson:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("monolithic");
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.05;
    errorDatabase["L2(0,T;L2(c))"] = 0.11225426870716;
    errorDatabase["L2(0,T;H1(c))"] = 0.6142481896612;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.18765027365623;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case MolNonLinearWithDirichletNeumannCrankNicolson:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("non-linear");
    db["afc_limiter"].set("monolithic");
    db["afc_nonlinloop_maxit"].set(1000);
    db["time_discretization"] = "crank_nicolson";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.1;
    errorDatabase["L2(0,T;L2(c))"] = 0.15353636183791;
    errorDatabase["L2(0,T;H1(c))"] = 0.63906771308993;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.51956544729604;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case ZalExplicitWithDirichlet:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 2;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Hexa");
    db["read_metis_file"].set("Default_Block_1x2x3_Hexa_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("explicit");
    db["afc_limiter"].set("zalesak");
    db["time_discretization"] = "Runge_Kutta_Heun";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.025;
    errorDatabase["L2(0,T;L2(c))"] = 0.017220093090019;
    errorDatabase["L2(0,T;H1(c))"] = 0.083065888822555;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 0.037909515114179;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    case ZalExplicitWithDirichletNeumann:
    db["diffusion_coefficient"] = 1e-1;
    db["example"] = 3;
    db["solver_type"] = "direct";
    db["boundary_file"].set("Default_Block_1x2x3");
    db["geo_file"].set("Default_Block_1x2x3_Tetra");
    db["read_metis_file"].set("Default_Block_1x2x3_Tetra_2_metis.txt", false);
    db["refinement_n_initial_steps"].set(2);
    db["afc_fct_scheme"].set("explicit");
    db["afc_limiter"].set("zalesak");
    db["time_discretization"] = "Runge_Kutta_Heun";
    db["time_start"] = 0.0;
    db["time_end"] = 1.0;
    db["time_step_length"] = 0.04;
    errorDatabase["L2(0,T;L2(c))"] = 0.20284977632633;
    errorDatabase["L2(0,T;H1(c))"] = 1.4279145289208;
    errorDatabase["L_inf(0,T,L_inf(c))"] = 1.103462904841;
    errorDatabase["checkOnlyFinalErrors"] = true;
    break;
    
    default:
      ErrThrow("Unspecified test");
  }
}


void timeIntegration(TimeConvectionDiffusion<3>& tcd,
                     ParameterDatabase& db,
                     ParameterDatabase errorDatabase)
{
  TimeDiscretization& tss = tcd.get_time_stepping_scheme();
  tss.current_step_ = 0;
  tss.current_time_ = db["time_start"];
  
  int rank = 0;
#ifdef _MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  
  tcd.assemble_initial_time();
  
  bool checkOnlyFinalErrors = errorDatabase["checkOnlyFinalErrors"];
  
  if(checkOnlyFinalErrors)
    tcd.output();

  double end_time = tcd.get_db()["time_end"];
  while(TDatabase::TimeDB->CURRENTTIME < end_time -1e-10)
  {
//     tss.nonlin_iteration = 0; // partl, not needed, set in for loop
    tss.current_step_++;
    tss.set_time_disc_parameters();
    SetTimeDiscParameters(1);
    
    tss.current_time_ += tss.get_step_length();
    TDatabase::TimeDB->CURRENTTIME += tss.get_step_length();

    if(rank == 0)
      Output::print<1>("CURRENT TIME: ", tss.current_time_);
    
    tcd.assemble();
    tcd.solve();
    
    if(checkOnlyFinalErrors)
      tcd.output();
    else
      check_solution_norms(tcd, tss.current_step_);
  }

}

void checkErrors(TimeConvectionDiffusion<3> &tcd,
                 ParameterDatabase& errorDatabase,
                 std::vector<std::string>& errorNames)
{
  std::array<double, 3> measuredErrors(tcd.get_errors());
  
  for(unsigned int i = 0; i < measuredErrors.size(); i++)
  {
    if( differentErrors(measuredErrors.at(i), errorDatabase[errorNames.at(i)]) )
      ErrThrow( "Test failed.\nError ",
                errorNames.at(i), " is too different.\n",
                "Its value is ",std::setprecision(14), measuredErrors.at(i),
                ", but it should be ", errorDatabase[errorNames.at(i)] );
  }
}

bool differentErrors(double measuredError, double expectedError)
{
  const double tolerance = 1.0e-7;
  
  if( std::abs(measuredError - expectedError) > tolerance )
    return true;
  else
    return false;
}


void createMetisFile(const std::string& metisFileName)
{
  std::ofstream metisFile;
  metisFile.open(metisFileName, std::ios::trunc | std::ios::out);
  
  metisFile << "ParMooN Mesh Partition File" << endl;
  
  if(metisFileName == "Default_UnitCube_Hexa_2_metis.txt")
  {
    metisFile << "N_Cells: 64 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 1\nCell 1, Rank 1\nCell 2, Rank 1\n"
              << "Cell 3, Rank 1\nCell 4, Rank 1\nCell 5, Rank 1\nCell 6, Rank 1\n"
              << "Cell 7, Rank 1\nCell 8, Rank 1\nCell 9, Rank 1\nCell 10, Rank 1\n"
              << "Cell 11, Rank 1\nCell 12, Rank 1\nCell 13, Rank 1\nCell 14, Rank 1\n"
              << "Cell 15, Rank 1\nCell 16, Rank 1\nCell 17, Rank 1\nCell 18, Rank 1\n"
              << "Cell 19, Rank 1\nCell 20, Rank 0\nCell 21, Rank 0\nCell 22, Rank 1\n"
              << "Cell 23, Rank 1\nCell 24, Rank 1\nCell 25, Rank 1\nCell 26, Rank 1\n"
              << "Cell 27, Rank 1\nCell 28, Rank 0\nCell 29, Rank 0\nCell 30, Rank 0\n"
              << "Cell 31, Rank 0\nCell 32, Rank 0\nCell 33, Rank 0\nCell 34, Rank 0\n"
              << "Cell 35, Rank 0\nCell 36, Rank 1\nCell 37, Rank 0\nCell 38, Rank 0\n"
              << "Cell 39, Rank 1\nCell 40, Rank 0\nCell 41, Rank 0\nCell 42, Rank 0\n"
              << "Cell 43, Rank 0\nCell 44, Rank 1\nCell 45, Rank 1\nCell 46, Rank 0\n"
              << "Cell 47, Rank 1\nCell 48, Rank 0\nCell 49, Rank 0\nCell 50, Rank 0\n"
              << "Cell 51, Rank 0\nCell 52, Rank 0\nCell 53, Rank 0\nCell 54, Rank 0\n"
              << "Cell 55, Rank 0\nCell 56, Rank 0\nCell 57, Rank 0\nCell 58, Rank 0\n"
              << "Cell 59, Rank 0\nCell 60, Rank 0\nCell 61, Rank 0\nCell 62, Rank 0\n"
              << "Cell 63, Rank 0";
  }
  else if(metisFileName == "Default_Block_1x2x3_Hexa_2_metis.txt")
  {
    metisFile << "N_Cells: 64 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 0\nCell 1, Rank 0\nCell 2, Rank 0\n"
              << "Cell 3, Rank 0\nCell 4, Rank 0\nCell 5, Rank 0\nCell 6, Rank 0\n"
              << "Cell 7, Rank 0\nCell 8, Rank 0\nCell 9, Rank 0\nCell 10, Rank 0\n"
              << "Cell 11, Rank 0\nCell 12, Rank 1\nCell 13, Rank 1\nCell 14, Rank 1\n"
              << "Cell 15, Rank 1\nCell 16, Rank 0\nCell 17, Rank 0\nCell 18, Rank 0\n"
              << "Cell 19, Rank 0\nCell 20, Rank 1\nCell 21, Rank 0\nCell 22, Rank 1\n"
              << "Cell 23, Rank 1\nCell 24, Rank 0\nCell 25, Rank 0\nCell 26, Rank 0\n"
              << "Cell 27, Rank 0\nCell 28, Rank 0\nCell 29, Rank 0\nCell 30, Rank 0\n"
              << "Cell 31, Rank 0\nCell 32, Rank 1\nCell 33, Rank 1\nCell 34, Rank 1\n"
              << "Cell 35, Rank 1\nCell 36, Rank 0\nCell 37, Rank 0\nCell 38, Rank 1\n"
              << "Cell 39, Rank 1\nCell 40, Rank 1\nCell 41, Rank 1\nCell 42, Rank 1\n"
              << "Cell 43, Rank 1\nCell 44, Rank 1\nCell 45, Rank 1\nCell 46, Rank 1\n"
              << "Cell 47, Rank 1\nCell 48, Rank 1\nCell 49, Rank 1\nCell 50, Rank 1\n"
              << "Cell 51, Rank 1\nCell 52, Rank 1\nCell 53, Rank 1\nCell 54, Rank 1\n"
              << "Cell 55, Rank 1\nCell 56, Rank 1\nCell 57, Rank 1\nCell 58, Rank 1\n"
              << "Cell 59, Rank 1\nCell 60, Rank 0\nCell 61, Rank 1\nCell 62, Rank 0\n"
              << "Cell 63, Rank 0";
  }
  else if(metisFileName == "Default_Block_1x2x3_Tetra_1_metis.txt")
  {
    metisFile << "N_Cells: 48 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 1\nCell 1, Rank 0\nCell 2, Rank 1\n"
              << "Cell 3, Rank 1\nCell 4, Rank 0\nCell 5, Rank 0\nCell 6, Rank 1\n"
              << "Cell 7, Rank 1\nCell 8, Rank 0\nCell 9, Rank 0\nCell 10, Rank 0\n"
              << "Cell 11, Rank 0\nCell 12, Rank 0\nCell 13, Rank 0\nCell 14, Rank 0\n"
              << "Cell 15, Rank 0\nCell 16, Rank 1\nCell 17, Rank 1\nCell 18, Rank 0\n"
              << "Cell 19, Rank 1\nCell 20, Rank 1\nCell 21, Rank 0\nCell 22, Rank 0\n"
              << "Cell 23, Rank 1\nCell 24, Rank 1\nCell 25, Rank 0\nCell 26, Rank 1\n"
              << "Cell 27, Rank 0\nCell 28, Rank 0\nCell 29, Rank 0\nCell 30, Rank 1\n"
              << "Cell 31, Rank 1\nCell 32, Rank 1\nCell 33, Rank 1\nCell 34, Rank 0\n"
              << "Cell 35, Rank 0\nCell 36, Rank 1\nCell 37, Rank 0\nCell 38, Rank 0\n"
              << "Cell 39, Rank 0\nCell 40, Rank 1\nCell 41, Rank 1\nCell 42, Rank 1\n"
              << "Cell 43, Rank 0\nCell 44, Rank 1\nCell 45, Rank 1\nCell 46, Rank 1\n"
              << "Cell 47, Rank 1";
  }
  else if(metisFileName == "Default_Block_1x2x3_Hexa_1_metis.txt")
  {
    metisFile << "N_Cells: 8 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 0\nCell 1, Rank 0\nCell 2, Rank 1\n"
              << "Cell 3, Rank 1\nCell 4, Rank 0\nCell 5, Rank 0\nCell 6, Rank 1\n"
              << "Cell 7, Rank 1";
  }
  else if(metisFileName == "block_mesh_file_1_metis.txt")
  {
    metisFile << "N_Cells: 48 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 0\nCell 1, Rank 0\nCell 2, Rank 0\n"
              << "Cell 3, Rank 1\nCell 4, Rank 0\nCell 5, Rank 0\nCell 6, Rank 1\n"
              << "Cell 7, Rank 0\nCell 8, Rank 0\nCell 9, Rank 1\nCell 10, Rank 1\n"
              << "Cell 11, Rank 0\nCell 12, Rank 1\nCell 13, Rank 1\nCell 14, Rank 1\n"
              << "Cell 15, Rank 1\nCell 16, Rank 0\nCell 17, Rank 1\nCell 18, Rank 0\n"
              << "Cell 19, Rank 0\nCell 20, Rank 1\nCell 21, Rank 1\nCell 22, Rank 0\n"
              << "Cell 23, Rank 0\nCell 24, Rank 1\nCell 25, Rank 0\nCell 26, Rank 1\n"
              << "Cell 27, Rank 1\nCell 28, Rank 1\nCell 29, Rank 0\nCell 30, Rank 1\n"
              << "Cell 31, Rank 1\nCell 32, Rank 1\nCell 33, Rank 0\nCell 34, Rank 0\n"
              << "Cell 35, Rank 1\nCell 36, Rank 1\nCell 37, Rank 0\nCell 38, Rank 0\n"
              << "Cell 39, Rank 0\nCell 40, Rank 1\nCell 41, Rank 1\nCell 42, Rank 1\n"
              << "Cell 43, Rank 1\nCell 44, Rank 1\nCell 45, Rank 1\nCell 46, Rank 0\n"
              << "Cell 47, Rank 0";
  }
  else if(metisFileName == "Default_Block_1x2x3_Tetra_2_metis.txt")
  {
    metisFile << "N_Cells: 384 ,N_Processors: 2" << endl;
    
    metisFile << "Cell 0, Rank 0\nCell 1, Rank 0\nCell 2, Rank 0\n"
              << "Cell 3, Rank 0\nCell 4, Rank 0\nCell 5, Rank 0\nCell 6, Rank 0\n"
              << "Cell 7, Rank 0\nCell 8, Rank 0\nCell 9, Rank 0\nCell 10, Rank 0\n"
              << "Cell 11, Rank 0\nCell 12, Rank 0\nCell 13, Rank 0\nCell 14, Rank 0\n"
              << "Cell 15, Rank 0\nCell 16, Rank 1\nCell 17, Rank 1\nCell 18, Rank 1\n"
              << "Cell 19, Rank 1\nCell 20, Rank 1\nCell 21, Rank 1\nCell 22, Rank 1\n"
              << "Cell 23, Rank 1\nCell 24, Rank 0\nCell 25, Rank 0\nCell 26, Rank 1\n"
              << "Cell 27, Rank 0\nCell 28, Rank 0\nCell 29, Rank 1\nCell 30, Rank 1\n"
              << "Cell 31, Rank 1\nCell 32, Rank 0\nCell 33, Rank 1\nCell 34, Rank 0\n"
              << "Cell 35, Rank 0\nCell 36, Rank 0\nCell 37, Rank 1\nCell 38, Rank 0\n"
              << "Cell 39, Rank 1\nCell 40, Rank 1\nCell 41, Rank 1\nCell 42, Rank 0\n"
              << "Cell 43, Rank 1\nCell 44, Rank 1\nCell 45, Rank 0\nCell 46, Rank 0\n"
              << "Cell 47, Rank 0\nCell 48, Rank 1\nCell 49, Rank 1\nCell 50, Rank 1\n"
              << "Cell 51, Rank 1\nCell 52, Rank 1\nCell 53, Rank 1\nCell 54, Rank 1\n"
              << "Cell 55, Rank 1\nCell 56, Rank 1\nCell 57, Rank 1\nCell 58, Rank 1\n"
              << "Cell 59, Rank 1\nCell 60, Rank 1\nCell 61, Rank 1\nCell 62, Rank 1\n"
              << "Cell 63, Rank 1\nCell 64, Rank 0\nCell 65, Rank 0\nCell 66, Rank 0\n"
              << "Cell 67, Rank 0\nCell 68, Rank 0\nCell 69, Rank 0\nCell 70, Rank 0\n"
              << "Cell 71, Rank 0\nCell 72, Rank 0\nCell 73, Rank 0\nCell 74, Rank 0\n"
              << "Cell 75, Rank 0\nCell 76, Rank 0\nCell 77, Rank 0\nCell 78, Rank 0\n"
              << "Cell 79, Rank 0\nCell 80, Rank 0\nCell 81, Rank 0\nCell 82, Rank 0\n"
              << "Cell 83, Rank 0\nCell 84, Rank 0\nCell 85, Rank 0\nCell 86, Rank 0\n"
              << "Cell 87, Rank 0\nCell 88, Rank 0\nCell 89, Rank 0\nCell 90, Rank 0\n"
              << "Cell 91, Rank 0\nCell 92, Rank 0\nCell 93, Rank 0\nCell 94, Rank 0\n"
              << "Cell 95, Rank 0\nCell 96, Rank 0\nCell 97, Rank 0\nCell 98, Rank 0\n"
              << "Cell 99, Rank 0\nCell 100, Rank 0\nCell 101, Rank 0\nCell 102, Rank 0\n"
              << "Cell 103, Rank 0\nCell 104, Rank 0\nCell 105, Rank 0\nCell 106, Rank 0\n"
              << "Cell 107, Rank 0\nCell 108, Rank 0\nCell 109, Rank 0\nCell 110, Rank 0\n"
              << "Cell 111, Rank 0\nCell 112, Rank 0\nCell 113, Rank 0\nCell 114, Rank 0\n"
              << "Cell 115, Rank 0\nCell 116, Rank 0\nCell 117, Rank 0\nCell 118, Rank 0\n"
              << "Cell 119, Rank 0\nCell 120, Rank 0\nCell 121, Rank 0\nCell 122, Rank 0\n"
              << "Cell 123, Rank 0\nCell 124, Rank 0\nCell 125, Rank 0\nCell 126, Rank 0\n"
              << "Cell 127, Rank 0\nCell 128, Rank 0\nCell 129, Rank 1\nCell 130, Rank 1\n"
              << "Cell 131, Rank 0\nCell 132, Rank 0\nCell 133, Rank 1\nCell 134, Rank 1\n"
              << "Cell 135, Rank 1\nCell 136, Rank 1\nCell 137, Rank 1\nCell 138, Rank 1\n"
              << "Cell 139, Rank 1\nCell 140, Rank 1\nCell 141, Rank 1\nCell 142, Rank 1\n"
              << "Cell 143, Rank 1\nCell 144, Rank 1\nCell 145, Rank 1\nCell 146, Rank 1\n"
              << "Cell 147, Rank 1\nCell 148, Rank 1\nCell 149, Rank 1\nCell 150, Rank 1\n"
              << "Cell 151, Rank 1\nCell 152, Rank 0\nCell 153, Rank 1\nCell 154, Rank 1\n"
              << "Cell 155, Rank 0\nCell 156, Rank 1\nCell 157, Rank 1\nCell 158, Rank 1\n"
              << "Cell 159, Rank 1\nCell 160, Rank 1\nCell 161, Rank 1\nCell 162, Rank 1\n"
              << "Cell 163, Rank 1\nCell 164, Rank 1\nCell 165, Rank 1\nCell 166, Rank 1\n"
              << "Cell 167, Rank 1\nCell 168, Rank 1\nCell 169, Rank 1\nCell 170, Rank 1\n"
              << "Cell 171, Rank 1\nCell 172, Rank 1\nCell 173, Rank 1\nCell 174, Rank 1\n"
              << "Cell 175, Rank 1\nCell 176, Rank 1\nCell 177, Rank 1\nCell 178, Rank 1\n"
              << "Cell 179, Rank 1\nCell 180, Rank 1\nCell 181, Rank 1\nCell 182, Rank 1\n"
              << "Cell 183, Rank 1\nCell 184, Rank 1\nCell 185, Rank 1\nCell 186, Rank 1\n"
              << "Cell 187, Rank 1\nCell 188, Rank 1\nCell 189, Rank 1\nCell 190, Rank 1\n"
              << "Cell 191, Rank 1\nCell 192, Rank 0\nCell 193, Rank 1\nCell 194, Rank 1\n"
              << "Cell 195, Rank 0\nCell 196, Rank 1\nCell 197, Rank 1\nCell 198, Rank 1\n"
              << "Cell 199, Rank 1\nCell 200, Rank 1\nCell 201, Rank 1\nCell 202, Rank 1\n"
              << "Cell 203, Rank 1\nCell 204, Rank 1\nCell 205, Rank 1\nCell 206, Rank 1\n"
              << "Cell 207, Rank 1\nCell 208, Rank 1\nCell 209, Rank 1\nCell 210, Rank 1\n"
              << "Cell 211, Rank 1\nCell 212, Rank 1\nCell 213, Rank 1\nCell 214, Rank 1\n"
              << "Cell 215, Rank 1\nCell 216, Rank 0\nCell 217, Rank 1\nCell 218, Rank 1\n"
              << "Cell 219, Rank 0\nCell 220, Rank 1\nCell 221, Rank 1\nCell 222, Rank 1\n"
              << "Cell 223, Rank 1\nCell 224, Rank 1\nCell 225, Rank 1\nCell 226, Rank 1\n"
              << "Cell 227, Rank 1\nCell 228, Rank 1\nCell 229, Rank 1\nCell 230, Rank 1\n"
              << "Cell 231, Rank 1\nCell 232, Rank 1\nCell 233, Rank 1\nCell 234, Rank 1\n"
              << "Cell 235, Rank 1\nCell 236, Rank 1\nCell 237, Rank 1\nCell 238, Rank 1\n"
              << "Cell 239, Rank 1\nCell 240, Rank 1\nCell 241, Rank 1\nCell 242, Rank 1\n"
              << "Cell 243, Rank 1\nCell 244, Rank 1\nCell 245, Rank 1\nCell 246, Rank 1\n"
              << "Cell 247, Rank 1\nCell 248, Rank 1\nCell 249, Rank 1\nCell 250, Rank 1\n"
              << "Cell 251, Rank 1\nCell 252, Rank 1\nCell 253, Rank 1\nCell 254, Rank 1\n"
              << "Cell 255, Rank 1\nCell 256, Rank 0\nCell 257, Rank 0\nCell 258, Rank 0\n"
              << "Cell 259, Rank 0\nCell 260, Rank 0\nCell 261, Rank 0\nCell 262, Rank 0\n"
              << "Cell 263, Rank 0\nCell 264, Rank 0\nCell 265, Rank 0\nCell 266, Rank 0\n"
              << "Cell 267, Rank 0\nCell 268, Rank 0\nCell 269, Rank 0\nCell 270, Rank 0\n"
              << "Cell 271, Rank 0\nCell 272, Rank 0\nCell 273, Rank 0\nCell 274, Rank 0\n"
              << "Cell 275, Rank 0\nCell 276, Rank 0\nCell 277, Rank 0\nCell 278, Rank 0\n"
              << "Cell 279, Rank 0\nCell 280, Rank 0\nCell 281, Rank 0\nCell 282, Rank 0\n"
              << "Cell 283, Rank 0\nCell 284, Rank 0\nCell 285, Rank 0\nCell 286, Rank 0\n"
              << "Cell 287, Rank 0\nCell 288, Rank 0\nCell 289, Rank 0\nCell 290, Rank 0\n"
              << "Cell 291, Rank 0\nCell 292, Rank 0\nCell 293, Rank 0\nCell 294, Rank 0\n"
              << "Cell 295, Rank 0\nCell 296, Rank 0\nCell 297, Rank 0\nCell 298, Rank 0\n"
              << "Cell 299, Rank 0\nCell 300, Rank 0\nCell 301, Rank 0\nCell 302, Rank 0\n"
              << "Cell 303, Rank 0\nCell 304, Rank 0\nCell 305, Rank 0\nCell 306, Rank 0\n"
              << "Cell 307, Rank 0\nCell 308, Rank 0\nCell 309, Rank 0\nCell 310, Rank 0\n"
              << "Cell 311, Rank 0\nCell 312, Rank 0\nCell 313, Rank 0\nCell 314, Rank 0\n"
              << "Cell 315, Rank 0\nCell 316, Rank 0\nCell 317, Rank 0\nCell 318, Rank 0\n"
              << "Cell 319, Rank 0\nCell 320, Rank 0\nCell 321, Rank 1\nCell 322, Rank 0\n"
              << "Cell 323, Rank 0\nCell 324, Rank 1\nCell 325, Rank 1\nCell 326, Rank 0\n"
              << "Cell 327, Rank 1\nCell 328, Rank 1\nCell 329, Rank 1\nCell 330, Rank 1\n"
              << "Cell 331, Rank 1\nCell 332, Rank 1\nCell 333, Rank 1\nCell 334, Rank 1\n"
              << "Cell 335, Rank 1\nCell 336, Rank 0\nCell 337, Rank 1\nCell 338, Rank 0\n"
              << "Cell 339, Rank 0\nCell 340, Rank 0\nCell 341, Rank 0\nCell 342, Rank 0\n"
              << "Cell 343, Rank 0\nCell 344, Rank 0\nCell 345, Rank 1\nCell 346, Rank 0\n"
              << "Cell 347, Rank 0\nCell 348, Rank 0\nCell 349, Rank 0\nCell 350, Rank 0\n"
              << "Cell 351, Rank 0\nCell 352, Rank 1\nCell 353, Rank 1\nCell 354, Rank 1\n"
              << "Cell 355, Rank 1\nCell 356, Rank 1\nCell 357, Rank 1\nCell 358, Rank 1\n"
              << "Cell 359, Rank 1\nCell 360, Rank 1\nCell 361, Rank 1\nCell 362, Rank 1\n"
              << "Cell 363, Rank 1\nCell 364, Rank 1\nCell 365, Rank 1\nCell 366, Rank 1\n"
              << "Cell 367, Rank 1\nCell 368, Rank 0\nCell 369, Rank 0\nCell 370, Rank 1\n"
              << "Cell 371, Rank 0\nCell 372, Rank 0\nCell 373, Rank 1\nCell 374, Rank 0\n"
              << "Cell 375, Rank 0\nCell 376, Rank 1\nCell 377, Rank 1\nCell 378, Rank 1\n"
              << "Cell 379, Rank 0\nCell 380, Rank 1\nCell 381, Rank 1\nCell 382, Rank 0\n"
              << "Cell 383, Rank 0";
  }
  
  metisFile.close();
}


// these are the previous errors - you should get them if
// you do not change the values at node 59 by prescribing
// the Dirichlet boundary conditions - comment by Ondrej Partl
// std::vector<std::vector<double>> target_norms=
// {
//   {0.007686841218, 0.09028123769, 4.647793003e-09}, //t=0.1
//   {0.01597853147, 0.179622979, 2.226841822e-07}, //t=0.2
//   {0.02482957109, 0.2680028932, 1.593788446e-06}, //t=0.3
//   {0.03394574615, 0.3538330922, 5.448744344e-06}, //t=0.4
//   {0.04302352518, 0.4349243671, 1.281619336e-05}, //t=0.5
//   {0.05175860945, 0.5089748268, 2.462124805e-05}, //t=0.6
//   {0.05986967233, 0.5738383662, 4.054345229e-05}, //t=0.7
//   {0.06710294614, 0.6276691113, 5.919686464e-05},  //t=0.8
//   {0.07323577128, 0.6689863334, 7.873596671e-05},  //t=0.9
//   {0.07808004918, 0.6967105354, 9.708502041e-05}   //t=1
// };
std::vector<std::vector<double>> target_norms=
{
  {0.007686841351, 0.09028123756, 0.1098068789}, //t=0.1
  {0.01597853364, 0.1796229804, 0.2190507937}, //t=0.2
  {0.02482957961, 0.2680028848, 0.324693393}, //t=0.3
  {0.0339457792, 0.3538330112, 0.4234723442}, //t=0.4
  {0.04302370833, 0.4349238981, 0.5125560048}, //t=0.5
  {0.0517596078, 0.5089738488, 0.5894766554}, //t=0.6
  {0.0598735036, 0.5738399255, 0.6521594774}, //t=0.7
  {0.06711401485, 0.6276851275, 0.6989447884},  //t=0.8
  {0.07326161907, 0.6690452317, 0.7286082832},  //t=0.9
  {0.07813175239, 0.6968646354, 0.7403766544}   //t=1
};

void check_solution_norms(TimeConvectionDiffusion<3> &tcd, int m)
{

  if(m%10 == 0)
  {
    MultiIndex3D allDerivatives[4] = { MultiIndex3D::D000, MultiIndex3D::D100,
                                       MultiIndex3D::D010, MultiIndex3D::D001 };
    TAuxParam3D aux(1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    const int n_errors = 5;
    std::array<double, n_errors> locError = {};
    const TFESpace3D* space = tcd.get_function().GetFESpace3D().get();

    tcd.get_function().GetErrors(tcd.get_example().get_exact(0),
                                 4, allDerivatives,
                                 n_errors, conv_diff_l2_h1_linf_error<3>,
                                 tcd.get_example().get_coeffs(), &aux, 1, &space,
                                 locError.data());

  #ifdef _MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> errorsReduced(n_errors);
    MPI_Reduce(locError.data(), errorsReduced.data(), n_errors-1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&locError[n_errors-1], &errorsReduced[n_errors-1], 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
  if(rank == 0)
  {//this is only performed on root - just root willl have the correct values
    for(int j = 0; j < n_errors-1; j++)
      locError[j] = std::sqrt(errorsReduced[j]);
    locError[n_errors-1] = errorsReduced[n_errors-1];
  #endif // _MPI
    
    Output::print("Norms in step ", m);
    Output::dash(std::setprecision(10), locError[0], ", ", locError[1], ", ", locError[n_errors-1]);

    int control_step = m/10 - 1;
    double tol = 1e-7;

    cout<<TDatabase::TimeDB->CURRENTTIME<<endl;
    if( std::abs(locError[0] - target_norms[control_step][0]) > tol )
      ErrThrow("L2 norm at timestep ", TDatabase::TimeDB->CURRENTTIME,  " is not correct, ", "calculated norm: ",
               locError[0], ",  reference norm: ", target_norms[control_step][0]);

    if( std::abs(locError[1]  - target_norms[control_step][1]) > tol )
      ErrThrow("H1 norm at timestep ", TDatabase::TimeDB->CURRENTTIME,  " is not correct, ", "calculated norm: ",
               locError[1], ",  reference norm: ", target_norms[control_step][1]);

    if( std::abs(locError[n_errors-1]  - target_norms[control_step][2]) > tol )
      ErrThrow("L^inf norm at timestep ", TDatabase::TimeDB->CURRENTTIME,  " is not correct, ", "calculated norm: ",
               locError[n_errors-1], ",  reference norm: ", target_norms[control_step][2]);
    Output::print("Solution norms checked succesfully.");
 #ifdef _MPI
  }//end if rank == 0
#endif
  }
}
