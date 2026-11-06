/**
 * @brief A test program for the solving of CD2D problems.
 *
 * This serves as a test for the solving of CD2D problems. It is intended to
 * perform CD2D calculations with different examples in different setups to test
 * a wide variety of ParMooN core functionality.
 * So far only one such test is implemented.
 *
 * The norms of the solution are compared with reference norms.
 * If those are not approximated well enough (or something in the process goes wrong)
 * the test fails.
 *
 * Should this test fail, there are two possibilities: either you made a mistake
 * which broke the programs functionality. Then you must find the mistake.
 * Or you changed some program setup (e.g. changed the default solver). Then this tests
 * shows you how many other program parts are affected by your changes.
 * If you are not perfectly sure how to repair this, it is a good idea
 * to describe your changes in the forum and request support.
 *
 *
 * @date 2015/11/11
 * @author Clemens Bartsch
 *
 */
#include <cmath>

#include <AlgebraicFluxCorrection.h>
#include <Domain.h>
#include <Database.h>
#include "FEDatabase.h"
#include "ConvectionDiffusion.h"
#include <Multigrid.h>
#include "all_defines_external_libraries.h"
#include "ParMooN.h"

#include <Chrono.h>
#include "LocalAssembling.h"
#include <MainUtilities.h> //for error measuring

// compare the computed errors in the Darcy2D object with the given ones in 
// the array
void compareErrors(const ConvectionDiffusion<2>& cd2d,
                   std::array<double, 4> errors)
{
  const double eps = 1e-11;
  
  // check the errors
  if( std::abs(cd2d.get_L2_error() - errors[0]) > eps )
  {
    ErrThrow("Program 1: L2 error not correct. ",
             cd2d.get_L2_error() - errors[0]);
  }
  if( std::abs(cd2d.get_H1_semi_error() - errors[1]) > eps )
  {
    ErrThrow("Program 1: H1-semi error not correct. ",
             cd2d.get_H1_semi_error() - errors[1]);
  }
  if( std::abs(cd2d.get_SD_error() - errors[2]) > eps )
  {
    ErrThrow("Program 1: sd error not correct.",
             cd2d.get_SD_error() - errors[2]);
  }
  if( std::abs(cd2d.get_L_inf_error() - errors[3]) > eps )
  {
    ErrThrow("Program 1: L_inf error not correct.",
             cd2d.get_L_inf_error() - errors[3]);
  }
}

// Here the actual computations take place
void check_cd2d(TDomain & domain, const ParameterDatabase& db, int element_code,
                std::array<double, 4> errors)
{
  TDatabase::ParamDB->ANSATZ_ORDER = element_code;
  
  ConvectionDiffusion<2> cd2d(domain, db);
  cd2d.assemble();
  cd2d.solve();
  cd2d.output();
  // compare computed with given errors
  compareErrors(cd2d, errors); // throws upon a difference
}

void tests_on_quads(ParameterDatabase& db)
{
  db["geo_file"] = "UnitSquare";
  // default construct a domain object
  TDomain domain(db);
  // refine grid
  domain.refine_and_get_hierarchy_of_collections(db);

  std::array<double, 4> errors;
    
  Output::print("\nstarting with Q1");
  errors = {{ 0.030136750989447, 0.50136473756928, 0.50136473756928, 
              0.086620472046938 }};
  check_cd2d(domain, db, 1, errors);
  
  Output::print("\nstarting with Q2");
  errors = {{ 0.0019271846609828, 0.050971302154831, 0.050971302154831, 
              0.0033739148851507 }};
  check_cd2d(domain, db, 2, errors);
  
  Output::print("\nstarting with Q3");
  errors = {{ 8.8160702476897e-05, 0.0033762284959484, 0.0033762284959484,
              0.00033253078309192 }};
  check_cd2d(domain, db, 3, errors);
  
  Output::print("\nstarting with Q4");
  errors = {{ 3.3497247653192e-06, 0.00016700003852901, 0.00016700003852901,
              7.4417876531063e-06 }};
  check_cd2d(domain, db, 4, errors);
  
  if(db["solver_type"].is("iterative") && db["preconditioner"].is("multigrid"))
  {
    // the higher order elements do not/very slowly converge.
    // @todo find out why multigrid does not allow high order elements for CD2D
    return;
  }
  Output::print("\nstarting with Q5");
  errors = {{ 1.0746418844078e-07, 6.5922012588569e-06, 6.5922012588569e-06,
              4.0247144117433e-07 }};
  check_cd2d(domain, db, 5, errors);
  
  Output::print("\nstarting with Q6");
  errors = {{ 2.9751487986477e-09, 2.1654097614847e-07, 2.1654097614847e-07,
              6.0485347841421e-09 }};
  check_cd2d(domain, db, 6, errors);
  
  if(db["solver_type"].is("iterative"))
  {
    // The gmres residual is changing during restart.
    // Maybe the basis functions of these elements are not implemented with
    // high enough precision.
    return;
  }
  Output::print("\nstarting with Q7");
  errors = {{ 7.2354841489073e-11, 6.0911844559287e-09, 6.0911844559287e-09,
              2.7735080898594e-10 }};
  check_cd2d(domain, db, 7, errors);
  
  Output::print("\nstarting with Q8");
  errors = {{ 2.0882090117857e-12, 1.4998643575721e-10, 1.4998643575721e-10,
              5.4845017416483e-12 }};
  check_cd2d(domain, db, 8, errors);
}

void tests_on_triangles(ParameterDatabase& db, bool other_diagonal)
{
  db["geo_file"] = other_diagonal ? "TwoTriangles2" : "TwoTriangles";
  // default construct a domain object
  TDomain domain(db);
  // refine grid
  domain.refine_and_get_hierarchy_of_collections(db);
  
  std::array<double, 4> errors;

  Output::print("\nstarting with P1");
  errors = {{ 0.068969534539772, 0.83003777179706, 0.83003777179706,
              0.16974869712877 }};
  check_cd2d(domain, db, 1, errors);
  
  Output::print("\nstarting with P2");
  errors = {{ 0.00421775248811, 0.12664905657584, 0.12664905657584,
              0.013686550806003 }};
  check_cd2d(domain, db, 2, errors);
  
  Output::print("\nstarting with P3");
  errors = {{ 0.0003326014029943, 0.013049293861338, 0.013049293861338,
              0.0013647682269053 }};
  check_cd2d(domain, db, 3, errors);
  
  Output::print("\nstarting with P4");
  errors = {{ 2.4012013491882e-05, 0.0011161787258859, 0.0011161787258859,
              8.9920045308348e-05 }};
  check_cd2d(domain, db, 4, errors);
  
  if(db["solver_type"].is("iterative") && db["preconditioner"].is("multigrid"))
  {
    // the higher order elements do not/very slowly converge.
    // @todo find out why multigrid does not allow high order elements for CD2D
    return;
  }
  
  Output::print("\nstarting with P5");
  errors = {{ 1.431573900717e-06, 7.8967826027302e-05, 7.8967826027302e-05,
              5.8590398090697e-06 }};
  check_cd2d(domain, db, 5, errors);
  
  Output::print("\nstarting with P6");
  errors = {{ 7.4027814432244e-08, 4.7847925721341e-06, 4.7847925721341e-06,
              3.2946208161633e-07 }};
  check_cd2d(domain, db, 6, errors);
}


void check_assembly(ConvectionDiffusion<2>& cd2d,
                    const std::vector<double>& expectedSysMat,
                    const std::vector<double>& expected_rhs)
{
  const double tolerance = 1.0e-9;
  
  const FEMatrix& sysMat = *cd2d.get_matrix().get_blocks_uniquely().at(0).get();
  const double* sysMatEntries = sysMat.GetEntries();
  const BlockVector& rhs = cd2d.get_rhs();
  
  
  if(expectedSysMat.size() != (long unsigned int)sysMat.get_n_entries())
    ErrThrow("Wrong number of system matrix entries! It is ", sysMat.get_n_entries(),
             " but should be ", expectedSysMat.size(), ".");
  
  if(expected_rhs.size() != rhs.length())
    ErrThrow("Wrong number of rhs entries! It is ", rhs.length(),
             " but should be ", expected_rhs.size(), ".");

  for(int i = 0; i < sysMat.get_n_entries(); i++)
    if(std::abs(sysMatEntries[i] - expectedSysMat[i]) > tolerance)
      ErrThrow("Wrong system matrix entry, ", i, "! It is ", setprecision(10),
               sysMatEntries[i], " but should be ", expectedSysMat[i], ".");

  for(unsigned int i = 0; i < rhs.length(); i++)
    if(std::abs(rhs[i] - expected_rhs[i]) > tolerance)
      ErrThrow("Wrong rhs entry, ", setprecision(10), i, "! It is ", rhs[i],
             " but should be ", expected_rhs[i], ".");
}


void testOneEAFE(ParameterDatabase& db,
              const std::vector<double>& expectedSysMat,
              const std::vector<double>& expected_rhs,
              const std::array<double,4>& expected_errors)
{
  TDomain domain(db);
  domain.refine_and_get_hierarchy_of_collections(db);
  ConvectionDiffusion<2> cd2d(domain, db);
  
  cd2d.assemble();
  check_assembly(cd2d, expectedSysMat, expected_rhs);
  
  cd2d.solve();
  cd2d.output();
  
  compareErrors(cd2d, expected_errors);
}


void testEAFEs(ParameterDatabase& db)
{
  std::vector<double> expectedSysMat =
  {4.577906827, -0.5819767069, -0.5819767069, 1.60812265e-16, 1.60812265e-16,
   -1.581976707, -1.581976707, -1.581976707, 2.665223748, -0.3853735206,
   2.043517099e-16, -0.6353735206, -0.6353735206, 1.642452875, -0.9654126876,
   -1.581976707, 1.239455774e-16, -0.2154126876, 2.950302082, -0.9654126876,
   0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
    
  std::vector<double> expected_rhs =
  {0.28125, -0.5260416667, -0.01979166667, 0.5989583333, 0, 2, 0.5, 1.5, 1};
  
  std::array<double,4> expected_errors =
  {0.019648660001103, 0.10023320849903, 0.11296079010865, 0.061604539171163};
  
  db["space_discretization_type"] = "EAFE_general";
  db["EAFE_outer_quad_formula"] = 16;
  testOneEAFE(db, expectedSysMat, expected_rhs, expected_errors);
  
  db["space_discretization_type"] = "EAFE_beta_const_on_edges";
  db["EAFE_outer_quad_formula"] = 3;
  testOneEAFE(db, expectedSysMat, expected_rhs, expected_errors);
}


void testMizukamiHughesOriginal(ParameterDatabase& db)
{
  std::vector<double> expectedSysMat =
  {4.55468750000000, -1.00108506944444, -0.00130208333333, -1.16254340277778, -1.00065104166667, 
   0.14800347222222, -1.51367187500000, -1.20160590277778, 4.73437500000000, -1.71245659722222,
   -1.00151909722222, 0.23784722222222, -0.00086805555556, -1.00108506944444, 0.21484375000000,
   -1.00151909722222, 4.75781250000000, -0.00217013888889, -1.00195312500000, -1.69162326388889,
   -1.22070312500000, -1.19943576388889, 4.86718750000000, -1.83268229166667, -0.00130208333333,
   -1.00195312500000, -1.00151909722222, 0.24001736111111, 0.28168402777778, 4.96093750000000,
   -1.00238715277778, -1.86957465277778, -1.00282118055556, -0.00303819444444, -1.27886284722222,
   -0.00173611111111, -1.25976562500000, -1.00195312500000, -1.89040798611111, 4.93750000000000,
   -1.00238715277778, 0.30468750000000, -1.00108506944444, -1.00151909722222, 4.68750000000000, 
   -0.00173611111111, -1.16037326388889, 0.15017361111111, -1.63389756944444, -1.00195312500000,
   -1.00238715277778, 0.21701388888889, 4.89062500000000, -1.81184895833333, -0.00260416666667, 
   -1.21853298611111, -1.00151909722222, -1.00195312500000, 4.82031250000000, 0.15234375000000,
   -0.00217013888889, -1.75412326388889, -1.15820312500000, -1.53450520833333, 0.17100694444444,
   2.53081597222222, -0.50043402777778, -0.64344618055556, -1.65473090277778, 0.17317708333333,
   -0.64105902777778, 2.66232638888889, -0.50065104166667, 1.51215277777778, -0.84461805555556, 
   -0.63628472222222, -1.19726562500000, -0.50086805555556, 2.58116319444444, -0.00173611111111,
   -0.84223090277778, -1.77495659722222, -0.63867187500000, -0.50086805555556, 0.17534722222222,
   2.79383680555556, -0.00217013888889, -1.25759548611111, -0.50108506944444, 2.64756944444444, 
   -0.83984375000000, -1.31792534722222, -0.00260416666667, -0.50130208333333, 2.71397569444444,
   -0.83745659722222, 0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 1.00000000000000,
   0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
   1.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 1.00000000000000};
    
  std::vector<double> expected_rhs =
  {0.02473958333333, 0.03255208333333, 0.06380208333333, 0.03255208333333, 0.13411458333333,
   0.10286458333333, 0.07161458333333, 0.12630208333333, 0.12630208333333, -0.25737847222222,
   -0.27365451388889, -0.02141927083333, 0.26443142361111, -0.28211805555556, 0.30219184027778,
   0.34385850694444, 0.00000000000000, 2.00000000000000, 0.25000000000000, 0.50000000000000, 
   1.75000000000000, 1.50000000000000, 1.00000000000000, 0.75000000000000, 1.25000000000000};
  
  std::array<double,4> expected_errors =
  {0.00031454621874011, 0.0019952553184829, 0.0020644386953297, 0.00098752308804306};
  
  TDomain domain(db);
  domain.refine_and_get_hierarchy_of_collections(db);
  ConvectionDiffusion<2> cd2d(domain, db);
  
  cd2d.assemble();
  cd2d.solve();
  cd2d.output();
  
  compareErrors(cd2d, expected_errors);
  
  cd2d.assemble();
  check_assembly(cd2d, expectedSysMat, expected_rhs);
  
  cd2d.checkMizukamiHughes();
}


void testMizukamiHughesImproved(ParameterDatabase& db)
{
  std::vector<double> expectedSysMat =
  {0.32889889078016, 0.03196235673889, -0.00370535714286, -0.16845312500000, 0.04161480818016,
   -0.00261111111111, -0.20863020833333, -0.20725173611111, 0.65824844279570, -0.60699855497856, 
   0.00883087887606, 0.18102674346975, -0.00264930555556, 0.01848625950687, 0.14061805555556,
   0.05688147643751, 0.59388967601031, -0.00452314814815, 0.04705543245617, -0.55344618055556,
   -0.22654687500000, -0.20567534722222, 0.83224605047386, -0.78581266793681, -0.00280059523810,
   0.00304678171035, 0.01299916239572, 0.21442611788035, 0.18488888888889, 0.74028482760000,
   0.07540538936264, -0.69158576388889, 0.06508495623155, -0.00535921717172, -0.28466701388889,
   -0.00361631944444, -0.26541145833333, 0.02537422767490, -0.75087938778519, 0.82871017209269, 
   0.01519205341595, 0.22908439351592, 0.06577951388889, 0.07232291666667, 0.33570709325397,
   -0.00381423611111, -0.16692385912698, -0.00344097222222, -0.26839459325397, 0.08751388888889,
   0.09404409722222, 0.18966298292801, 0.69889878197995, -0.76076225808755, -0.00470208333333, 
   -0.22488252314815, 0.01819267183390, 0.02726070876079, 0.49102887515105, -0.00415773809524,
   -0.00399537037037, -0.32830034722222, -0.16523437500000, -0.45730271022460, 0.12995753413413,
   0.49197928904206, -0.00061805555556, -0.14409201388889, -0.62088314343956, 0.15459422033029,
   -0.14249652777778, 0.64644057581059, -0.00081622023810, 0.51245486111111, -0.34308680555556, 
   -0.13871180555556, -0.20396701388889, -0.00101909722222, 0.58817766203704, -0.00302256944444, 
   -0.34146932870370, -0.75075891902178, -0.14067485119048, -0.00101909722222, 0.16194164421558,
   0.78319726617571, -0.00381944444444, -0.26371631944444, -0.00122511574074, 0.65495115740741,
   -0.33969791666667, -0.32357118055556, -0.00454375000000, -0.00143333333333, 0.72165088383838,
   -0.33781455176768, 0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 1.00000000000000,
   0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
   1.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 1.00000000000000};
    
  std::vector<double> expected_rhs =
  {0.01636924915233, 0.02671251623665, 0.06059131299854, 0.02950864625811, 0.13007253241104,
   0.09045816301166, 0.05002693297371, 0.13820761714826, 0.07762070781454, -0.00944361051146, 
   -0.02531135496684, -0.02102962239583, 0.01652218967014, -0.03383214978417, 0.05404201388889,
   0.09548205097854, 0.00000000000000, 2.00000000000000, 0.25000000000000, 0.50000000000000,
   1.75000000000000, 1.50000000000000, 1.00000000000000, 0.75000000000000, 1.25000000000000};
  
  std::array<double,4> expected_errors =
  {0.00099515560869758, 0.0079298213078597, 0.0038332969378316, 0.0038204215591198};
  
  TDomain domain(db);
  domain.refine_and_get_hierarchy_of_collections(db);
  ConvectionDiffusion<2> cd2d(domain, db);
  
  cd2d.assemble();
  cd2d.solve();
  cd2d.output();
  
  compareErrors(cd2d, expected_errors);
  
  cd2d.assemble();
  check_assembly(cd2d, expectedSysMat, expected_rhs);
  
  cd2d.checkMizukamiHughes();
}


void testMizukamiHughesReaction(ParameterDatabase& db)
{
  std::vector<double> expectedSysMat =
  {0.42541666666667, -0.01108506944444, -0.00130208333333, -0.17254340277778, -0.01065104166667, 
   -0.00043402777778, -0.21377604166667, -0.21160590277778, 0.77437500000000, -0.72245659722222, 
   -0.01151909722222, 0.23784722222222, -0.00086805555556, -0.01108506944444, 0.15443328862710,
   -0.01151909722222, 0.73157934355501, -0.00217013888889, -0.01195312500000, -0.58153017802710,
   -0.23070312500000, -0.20943576388889, 0.90718750000000, -0.84268229166667, -0.00130208333333,
   -0.01195312500000, -0.01151909722222, 0.24001736111111, 0.04687053078492, 0.74888998487922,
   -0.01238715277778, -0.42502742465398, -0.01282118055556, -0.00303819444444, -0.28886284722222,
   -0.00173611111111, -0.26976562500000, -0.01195312500000, -0.62583586596698, 0.82026639273798, 
   -0.01238715277778, 0.16622805479911, -0.01108506944444, -0.01151909722222, 0.49312500000000, 
   -0.00173611111111, -0.17037326388889, -0.00086805555556, -0.27410590277778, -0.01195312500000,
   -0.01238715277778, 0.21701388888889, 0.93062500000000, -0.82184895833333, -0.00260416666667,
   -0.22853298611111, -0.01151909722222, -0.01195312500000, 0.56083333333333, -0.00130208333333,
   -0.00217013888889, -0.33443576388889, -0.16820312500000, -0.54450520833333, 0.17100694444444, 
   0.55081597222222, -0.00543402777778, -0.14844618055556, -0.66473090277778, 0.17317708333333,
   -0.14605902777778, 0.68232638888889, -0.00565104166667, 0.52215277777778, -0.34961805555556,
   -0.14128472222222, -0.20726562500000, -0.00586805555556, 0.60116319444444, -0.00173611111111,
   -0.34723090277778, -0.78495659722222, -0.14367187500000, -0.00586805555556, 0.17534722222222,
   0.81383680555556, -0.00217013888889, -0.26759548611111, -0.00608506944444, 0.66756944444444,
   -0.34484375000000, -0.32792534722222, -0.00260416666667, -0.00630208333333, 0.73397569444444,
   -0.34245659722222, 0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 0.00000000000000, 
   0.00000000000000, 0.00000000000000, 1.00000000000000, 0.00000000000000, 1.00000000000000, 
   0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 
   1.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000, 0.00000000000000,
   0.00000000000000, 1.00000000000000};
    
  std::vector<double> expected_rhs =
  {0.01725260416667, 0.03255208333333, 0.05734253098774, 0.03255208333333, 0.08914456728080,
   0.07573005222935, 0.04264322916667, 0.12630208333333, 0.07194010416667, -0.00987847222222,
   -0.02615451388889, -0.02141927083333, 0.01693142361111, -0.03461805555556, 0.05469184027778, 
   0.09635850694444, 0.00000000000000, 2.00000000000000, 0.25000000000000, 0.50000000000000,
   1.75000000000000, 1.50000000000000, 1.00000000000000, 0.75000000000000, 1.25000000000000};
  
  std::array<double,4> expected_errors =
  {0.001108368482153, 0.007970031699082, 0.003822707851681, 0.0040108177936128};
  
  TDomain domain(db);
  domain.refine_and_get_hierarchy_of_collections(db);
  ConvectionDiffusion<2> cd2d(domain, db);
  
  cd2d.assemble();
  cd2d.solve();
  cd2d.output();
  
  compareErrors(cd2d, expected_errors);
  
  cd2d.assemble();
  check_assembly(cd2d, expectedSysMat, expected_rhs);
  
  cd2d.checkMizukamiHughes();
}


// =======================================================================
// main program
// =======================================================================
int main(int, char**)
{
  auto db = parmoon::parmoon_initialize();
  Output::setVerbosity(2);
  
  size_t nRefinements = 2;
  
  //ParameterDatabase db = ParameterDatabase::parmoon_default_database();
  db.merge(Solver<>::default_solver_database());
  db.merge(Multigrid::default_multigrid_database());
  db.merge(Example2D::default_example_database());
  db.merge(ParameterDatabase::default_output_database());
  db["problem_type"] = 0; // problem type is not needed
  db["example"] = 0; // known sin-cos solution
  db["residual_tolerance"] = 1.0e-13;
  
  db["output_compute_errors"] = true;
  
  // from 'TDomain::default_domain_parameters()'
  db.add("boundary_file", "Default_UnitSquare", "");
  db.add("geo_file", "UnitSquare", "", {"UnitSquare", "TwoTriangles",
                                        "TwoTriangles2"});
  db.add("refinement_n_initial_steps", (size_t) nRefinements,"", (size_t)0, (size_t)10);
  
  Chrono time; // measure the time spent for each solver
  
  Output::print("\n\n ----------- direct solver -----------\n");
  db["solver_type"] = "direct";
  tests_on_quads(db);
  tests_on_triangles(db, false);
  tests_on_triangles(db, true);
  time.restart_and_print("all tests, direct solver");
  
#ifdef PARMOON_WITH_PETSC
  Output::print("\n\n ----------- petsc solver -----------\n");
  db["solver_type"] = "petsc";
  tests_on_quads(db);
  tests_on_triangles(db, false);
  tests_on_triangles(db, true);
  time.restart_and_print("all tests, petsc solver");
#endif // PARMOON_WITH_PETSC  
  
  
  Output::print("\n\n --------- fgmres+ssor solver ---------\n");
  db["solver_type"] = "iterative";
  db["iterative_solver_type"] = "fgmres";
  db["max_n_iterations"] = 1000;
  db["preconditioner"] = "ssor";
  tests_on_quads(db);
  tests_on_triangles(db, false);
  tests_on_triangles(db, true);
  time.restart_and_print("all tests, fgmres solver and ssor preconditioner");
  
  
  
  Output::print("\n\n --------- fgmres+multigrid solver ---------\n");
  db["iterative_solver_type"] = "fgmres";
  db["preconditioner"] = "multigrid";
  db["multigrid_n_levels"] = 2;
  db["multigrid_cycle_type"] = "W";
  db["multigrid_smoother"] = "jacobi";
  db["multigrid_smoother_coarse"] = "direct_solve";
  db["multigrid_correction_damp_factor"] = 1.0;
  db["multigrid_n_pre_smooth"] = 2;
  db["multigrid_n_post_smooth"] = 2;
  tests_on_quads(db);
  tests_on_triangles(db, false);
  tests_on_triangles(db, true);
  time.restart_and_print("all tests, fgmres solver and multigrid "
                         "preconditioner");
  
  
  Output::print("\n\n ----------- EAFE -----------\n");
  TDatabase::ParamDB->ANSATZ_ORDER = 1;
  db["problem_type"] = 1;
  db["solver_type"] = "direct";
  db["output_write_vtk"] = true;
  db["example"] = 25;
  db["geo_file"] = "TwoTriangles";
  db["refinement_n_initial_steps"] = 1;
  db["diffusion_coefficient"] = 1.0;
  db.add("space_discretization_type", "EAFE_general", "",
         {"EAFE_general", "EAFE_beta_const_on_edges", "Mizukami_Hughes_original",
          "Mizukami_Hughes_improved", "Mizukami_Hughes_reaction"});
  db.add("EAFE_outer_quad_formula", (size_t) 3, "", (size_t) 0, (size_t) 64);
  db.add("EAFE_inner_quad_formula", (size_t) 3, "", (size_t) 0, (size_t) 64);
  
  
  testEAFEs(db);
  
  
  // Mizukami-Hughes
  TDatabase::ParamDB->ANSATZ_ORDER = 1;
  db["problem_type"] = 1;
  db["solver_type"] = "direct";
  db["output_write_vtk"] = true;
  db["example"] = 25;
  db["geo_file"] = "TwoTriangles";
  db["refinement_n_initial_steps"] = 2;
  db["diffusion_coefficient"] = 1.0e-2;
  db.add("Mizukami_Hughes_special_triangulation", true, "");
  
  Output::print("\n\n ----------- Mizukami-Hughes original -----------\n");
  db["space_discretization_type"].set("Mizukami_Hughes_original");
  db["diffusion_coefficient"] = 1.0;
  testMizukamiHughesOriginal(db);
  
  Output::print("\n\n ----------- Mizukami-Hughes improved -----------\n");
  db["space_discretization_type"].set("Mizukami_Hughes_improved");
  db["diffusion_coefficient"] = 1.0e-2;
  testMizukamiHughesImproved(db);
  
  Output::print("\n\n ----------- Mizukami-Hughes reaction -----------\n");
  db["space_discretization_type"].set("Mizukami_Hughes_reaction");
  db["diffusion_coefficient"] = 1.0e-2;
  testMizukamiHughesReaction(db);
  
  parmoon::parmoon_finalize();
}
