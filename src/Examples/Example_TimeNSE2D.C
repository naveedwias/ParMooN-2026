#include <Example_TimeNSE2D.h>
#include <Database.h>
#include <MainUtilities.h>
#include <BoundEdge.h>
#include "FEDatabase.h"
#include "TimeNavierStokes.h"
#include "AuxParam2D.h" // used in MixingLayerSlipSmallSquares.h
#include "BaseCell.h"
#include "Example_NSE2D.h"
#include <string>

namespace bsp1              // case 0
{
 #include "TNSE_2D/Bsp1.h"
}

namespace lin_space_time   // case 1
{
#include "TNSE_2D/linear_space_time.h"
}

namespace sincosexp        // case 2
{
#include "TNSE_2D/SinCosExp.h"
}

namespace flow_around_cylinder_steady_inflow     // case 3
{
#include "flow_around_cylinder_steady_inflow.h"
}

namespace backward_facing_step_time  // case 4
{
#include "TNSE_2D/backward_facing_step.h"
}

namespace driven_cavity_time         // case 5
{
#include "TNSE_2D/DrivenCavity.h"
}

namespace mixing_layer_us       // case 6
{
#include "TNSE_2D/MixingLayerSlipSmallSquares.h"
}

namespace flow_around_cylinder_transient_inflow
{
#include "TNSE_2D/flow_around_cylinder_transient_inflow.h"
}

namespace two_outlets
{
#include "TNSE_2D/Two_Outlets.h"
}

namespace poly_sin
{
#include "TNSE_2D/PolySin.h"
}

namespace sinsincospi
{
#include "TNSE_2D/SinSinCosPi.h"
}

// ---- transient examples of the adaptive SUPG/PSPG/grad-div and the BDF2-IMEX/CIP
// ---- programs (cases 23-28; the numbers are kept from their former place in
// ---- Example_NSE2D, so that the input files remain valid)
namespace tnse_aaj_smooth            // case 23
{
#include "TNSE_2D/TNSE_AAJ_smooth.h"
}
namespace tnse_translating_vortex    // case 32 (Example 2 of the manuscript)
{
#include "TNSE_2D/TNSE_translating_vortex.h"
}
namespace bdf2imex_smooth            // case 25
{
#include "TNSE_2D/BDF2IMEX_smooth.h"
}
namespace bdf2imex_lake              // case 26
{
#include "TNSE_2D/BDF2IMEX_lake.h"
}
namespace bdf2imex_shearlayer        // case 27
{
#include "TNSE_2D/BDF2IMEX_shearlayer.h"
}
namespace tnse_polynomial_in_space   // case 28
{
#include "TNSE_2D/TNSE_polynomial_in_space.h"
}

namespace tnse_poly_quadratic_time   // case 30
{
#define TNSE_POLY_TIME_DEGREE 2
#include "TNSE_2D/TNSE_polynomial_space_time.h"
#undef TNSE_POLY_TIME_DEGREE
}
namespace tnse_poly_linear_time      // case 31
{
#define TNSE_POLY_TIME_DEGREE 1
#include "TNSE_2D/TNSE_polynomial_space_time.h"
#undef TNSE_POLY_TIME_DEGREE
}

CoeffFct2D bdf2imex_stokes_projection_coeffs(int example)
{
  switch(example)
  {
    case 25: return bdf2imex_smooth::StokesProjectionCoeffs;
    case 26: return bdf2imex_lake::StokesProjectionCoeffs;
    case 27: return bdf2imex_shearlayer::StokesProjectionCoeffs;
    case 28: return tnse_polynomial_in_space::StokesProjectionCoeffs;
    case 30: return tnse_poly_quadratic_time::StokesProjectionCoeffs;
    case 31: return tnse_poly_linear_time::StokesProjectionCoeffs;
    case 32: return tnse_translating_vortex::StokesProjectionCoeffs;
    default:
      ErrThrow("bdf2imex_stokes_projection_coeffs: no Stokes projection data "
               "for example ", example, " (available: 25-28, 30-32)");
  }
}

Example_TimeNSE2D::Example_TimeNSE2D(
  const ParameterDatabase& user_input_parameter_db)
 : Example_NonStationary2D(user_input_parameter_db)
{
  int example_code = this->example_database["example"];
  switch(example_code)
  {
    case 0:
      /** exact_solution */
      exact_solution.push_back( bsp1::ExactU1 );
      exact_solution.push_back( bsp1::ExactU2 );
      exact_solution.push_back( bsp1::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( bsp1::BoundCondition );
      boundary_conditions.push_back( bsp1::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( bsp1::U1BoundValue );
      boundary_data.push_back( bsp1::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = bsp1::LinCoeffs;
      
      /** initial condition */
      initialCondition.push_back(bsp1::InitialU1);
      initialCondition.push_back(bsp1::InitialU2);
      initialCondition.push_back(bsp1::InitialP);
      bsp1::ExampleFile();
      break;
    case 1:
      /** exact_solution */
      exact_solution.push_back( lin_space_time::ExactU1 );
      exact_solution.push_back( lin_space_time::ExactU2 );
      exact_solution.push_back( lin_space_time::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( lin_space_time::BoundCondition );
      boundary_conditions.push_back( lin_space_time::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( lin_space_time::U1BoundValue );
      boundary_data.push_back( lin_space_time::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = lin_space_time::LinCoeffs;
      
      initialCondition.push_back(lin_space_time::InitialU1);
      initialCondition.push_back(lin_space_time::InitialU2);
      initialCondition.push_back(lin_space_time::InitialP);
      
      // Set dimensionless viscosity
      lin_space_time::DIMENSIONLESS_VISCOSITY = get_nu();
      
      lin_space_time::ExampleFile();
      break;
      
    case 2: // SinCosExp
      /** exact_solution */
      exact_solution.push_back(sincosexp::ExactU1 );
      exact_solution.push_back(sincosexp::ExactU2 );
      exact_solution.push_back(sincosexp::ExactP );

      /** boundary condition */
      boundary_conditions.push_back(sincosexp::BoundCondition );
      boundary_conditions.push_back(sincosexp::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );

      /** boundary values */
      boundary_data.push_back(sincosexp::U1BoundValue );
      boundary_data.push_back(sincosexp::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );

      // Set dimensionless viscosity
      sincosexp::DIMENSIONLESS_VISCOSITY = get_nu();

      /** coefficients */
      problem_coefficients =sincosexp::LinCoeffs;

      initialCondition.push_back(sincosexp::InitialU1);
      initialCondition.push_back(sincosexp::InitialU2);
      initialCondition.push_back(sincosexp::InitialP);

      sincosexp::ExampleFile();
      break;

    case 3:
      /** exact_solution */
      exact_solution.push_back( flow_around_cylinder_steady_inflow::ExactU1 );
      exact_solution.push_back( flow_around_cylinder_steady_inflow::ExactU2 );
      exact_solution.push_back( flow_around_cylinder_steady_inflow::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( flow_around_cylinder_steady_inflow::BoundCondition );
      boundary_conditions.push_back( flow_around_cylinder_steady_inflow::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( flow_around_cylinder_steady_inflow::U1BoundValue );
      boundary_data.push_back( flow_around_cylinder_steady_inflow::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = flow_around_cylinder_steady_inflow::LinCoeffs;
      
      initialCondition.push_back(flow_around_cylinder_steady_inflow::InitialU1);
      initialCondition.push_back(flow_around_cylinder_steady_inflow::InitialU2);
      initialCondition.push_back(flow_around_cylinder_steady_inflow::InitialP);
      
      // Set dimensionless viscosity
      flow_around_cylinder_steady_inflow::DIMENSIONLESS_VISCOSITY = get_nu();

      /**post processing - drag and lift calculation and output */
      post_processing_stat = flow_around_cylinder_steady_inflow::compute_drag_lift_pdiff;

      flow_around_cylinder_steady_inflow::ExampleFile();
      break;
    case 4:
      exact_solution.push_back( backward_facing_step_time::ExactU1 );
      exact_solution.push_back( backward_facing_step_time::ExactU2 );
      exact_solution.push_back( backward_facing_step_time::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( backward_facing_step_time::BoundCondition );
      boundary_conditions.push_back( backward_facing_step_time::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( backward_facing_step_time::U1BoundValue );
      boundary_data.push_back( backward_facing_step_time::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = backward_facing_step_time::LinCoeffs;
      
      initialCondition.push_back(backward_facing_step_time::InitialU1);
      initialCondition.push_back(backward_facing_step_time::InitialU2);
      initialCondition.push_back(backward_facing_step_time::InitialP);
      
      backward_facing_step_time::ExampleFile();
      backward_facing_step_time::DIMENSIONLESS_VISCOSITY = this->get_nu();
      break;
    case 5:
      exact_solution.push_back( driven_cavity_time::ExactU1 );
      exact_solution.push_back( driven_cavity_time::ExactU2 );
      exact_solution.push_back( driven_cavity_time::ExactP );

      /** boundary condition */
      boundary_conditions.push_back( driven_cavity_time::BoundCondition );
      boundary_conditions.push_back( driven_cavity_time::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );

      /** boundary values */
      boundary_data.push_back( driven_cavity_time::U1BoundValue );
      boundary_data.push_back( driven_cavity_time::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );

      /** coefficients */
      problem_coefficients = driven_cavity_time::LinCoeffs;

      initialCondition.push_back(driven_cavity_time::InitialU1);
      initialCondition.push_back(driven_cavity_time::InitialU2);
      initialCondition.push_back(driven_cavity_time::InitialP);

      driven_cavity_time::DIMENSIONLESS_VISCOSITY = this->get_nu();
      driven_cavity_time::ExampleFile();
      break;
    case 6:
      exact_solution.push_back( mixing_layer_us::ExactU1 );
      exact_solution.push_back( mixing_layer_us::ExactU2 );
      exact_solution.push_back( mixing_layer_us::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( mixing_layer_us::BoundCondition );
      boundary_conditions.push_back( mixing_layer_us::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( mixing_layer_us::U1BoundValue );
      boundary_data.push_back( mixing_layer_us::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = mixing_layer_us::LinCoeffs;
      
      initialCondition.push_back(mixing_layer_us::InitialU1);
      initialCondition.push_back(mixing_layer_us::InitialU2);
      initialCondition.push_back(mixing_layer_us::InitialP);
      
      mixing_layer_us::ExampleFile();
      
      mixing_layer_us::DIMENSIONLESS_VISCOSITY = this->get_nu();
      
      /**post processing - drag and lift calculation and output */
      post_processing_stat = mixing_layer_us::EvaluateSolution;
      break;
    case 7:
      /** exact_solution */
      exact_solution.push_back( flow_around_cylinder_transient_inflow::ExactU1 );
      exact_solution.push_back( flow_around_cylinder_transient_inflow::ExactU2 );
      exact_solution.push_back( flow_around_cylinder_transient_inflow::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( flow_around_cylinder_transient_inflow::BoundCondition );
      boundary_conditions.push_back( flow_around_cylinder_transient_inflow::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( flow_around_cylinder_transient_inflow::U1BoundValue );
      boundary_data.push_back( flow_around_cylinder_transient_inflow::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = flow_around_cylinder_transient_inflow::LinCoeffs;
      
      initialCondition.push_back(flow_around_cylinder_transient_inflow::InitialU1);
      initialCondition.push_back(flow_around_cylinder_transient_inflow::InitialU2);
      initialCondition.push_back(flow_around_cylinder_transient_inflow::InitialP);
      
      // Set dimensionless viscosity
      flow_around_cylinder_transient_inflow::DIMENSIONLESS_VISCOSITY = get_nu();

      /**post processing - drag and lift calculation and output */
      post_processing_stat = flow_around_cylinder_transient_inflow::compute_drag_lift_pdiff;

      flow_around_cylinder_transient_inflow::ExampleFile();
      break;
    case 8:
      /** exact_solution */
      exact_solution.push_back( poly_sin::ExactU1 );
      exact_solution.push_back( poly_sin::ExactU2 );
      exact_solution.push_back( poly_sin::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( poly_sin::BoundCondition );
      boundary_conditions.push_back( poly_sin::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( poly_sin::U1BoundValue );
      boundary_data.push_back( poly_sin::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = poly_sin::LinCoeffs;
      
      initialCondition.push_back(poly_sin::InitialU1);
      initialCondition.push_back(poly_sin::InitialU2);
      initialCondition.push_back(poly_sin::InitialP);
      
      // Set dimensionless viscosity
      poly_sin::DIMENSIONLESS_VISCOSITY = get_nu();


      poly_sin::ExampleFile();
      break;
    case 9:
      /** exact_solution */
      exact_solution.push_back( sinsincospi::ExactU1 );
      exact_solution.push_back( sinsincospi::ExactU2 );
      exact_solution.push_back( sinsincospi::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( sinsincospi::BoundCondition );
      boundary_conditions.push_back( sinsincospi::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( sinsincospi::U1BoundValue );
      boundary_data.push_back( sinsincospi::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = sinsincospi::LinCoeffs;
      
      initialCondition.push_back(sinsincospi::InitialU1);
      initialCondition.push_back(sinsincospi::InitialU2);
      initialCondition.push_back(sinsincospi::InitialP);
      
      // Set dimensionless viscosity
      sinsincospi::DIMENSIONLESS_VISCOSITY = get_nu();

      sinsincospi::ExampleFile();
      break;
      /* we use example 14 to be consistent with the steady NSE case */
  case 14:
      /** exact_solution */
      exact_solution.push_back( two_outlets::ExactU1 );
      exact_solution.push_back( two_outlets::ExactU2 );
      exact_solution.push_back( two_outlets::ExactP );
      
      /** boundary condition */
      boundary_conditions.push_back( two_outlets::BoundCondition );
      boundary_conditions.push_back( two_outlets::BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      
      /** boundary values */
      boundary_data.push_back( two_outlets::U1BoundValue );
      boundary_data.push_back( two_outlets::U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      
      /** coefficients */
      problem_coefficients = two_outlets::LinCoeffs;
      
      initialCondition.push_back(two_outlets::InitialU1);
      initialCondition.push_back(two_outlets::InitialU2);
      initialCondition.push_back(two_outlets::InitialP);
      
      // Set dimensionless viscosity
      two_outlets::effective_viscosity = get_nu();

      // boundary conditions
      two_outlets::neumann_id = get_neumann_id();
      two_outlets::nitsche_id = get_nitsche_id();
      two_outlets::windkessel_id = get_windkessel_id();
     

      two_outlets::ExampleFile();
      break;
    case 23: // transient SUPG/PSPG/grad-div Navier-Stokes, transient smooth
    exact_solution.push_back( tnse_aaj_smooth::ExactU1 );
    exact_solution.push_back( tnse_aaj_smooth::ExactU2 );
    exact_solution.push_back( tnse_aaj_smooth::ExactP );
    boundary_conditions.push_back( tnse_aaj_smooth::BoundCondition );
    boundary_conditions.push_back( tnse_aaj_smooth::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( tnse_aaj_smooth::U1BoundValue );
    boundary_data.push_back( tnse_aaj_smooth::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = tnse_aaj_smooth::LinCoeffs;
    tnse_aaj_smooth::ExampleFile();
    // initial condition: the exact solution at time_start
    initialCondition.push_back( tnse_aaj_smooth::ExactU1 );
    initialCondition.push_back( tnse_aaj_smooth::ExactU2 );
    initialCondition.push_back( tnse_aaj_smooth::ExactP );
    break;

    case 28: // transient verification: u in P2, p in P1, time-dependent Dirichlet data
    exact_solution.push_back( tnse_polynomial_in_space::ExactU1 );
    exact_solution.push_back( tnse_polynomial_in_space::ExactU2 );
    exact_solution.push_back( tnse_polynomial_in_space::ExactP );
    boundary_conditions.push_back( tnse_polynomial_in_space::BoundCondition );
    boundary_conditions.push_back( tnse_polynomial_in_space::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( tnse_polynomial_in_space::U1BoundValue );
    boundary_data.push_back( tnse_polynomial_in_space::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = tnse_polynomial_in_space::LinCoeffs;
    tnse_polynomial_in_space::ExampleFile();
    // initial condition: the exact solution at time_start
    initialCondition.push_back( tnse_polynomial_in_space::ExactU1 );
    initialCondition.push_back( tnse_polynomial_in_space::ExactU2 );
    initialCondition.push_back( tnse_polynomial_in_space::ExactP );
    break;

    case 25: // BDF2-IMEX / CIP / Scott-Vogelius, smooth solution
    exact_solution.push_back( bdf2imex_smooth::ExactU1 );
    exact_solution.push_back( bdf2imex_smooth::ExactU2 );
    exact_solution.push_back( bdf2imex_smooth::ExactP );
    boundary_conditions.push_back( bdf2imex_smooth::BoundCondition );
    boundary_conditions.push_back( bdf2imex_smooth::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( bdf2imex_smooth::U1BoundValue );
    boundary_data.push_back( bdf2imex_smooth::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = bdf2imex_smooth::LinCoeffs;
    bdf2imex_smooth::ExampleFile();
    // initial condition: the exact solution at time_start
    initialCondition.push_back( bdf2imex_smooth::ExactU1 );
    initialCondition.push_back( bdf2imex_smooth::ExactU2 );
    initialCondition.push_back( bdf2imex_smooth::ExactP );
    break;

    case 26: // BDF2-IMEX / CIP / Scott-Vogelius, lake at rest (u = 0)
    exact_solution.push_back( bdf2imex_lake::ExactU1 );
    exact_solution.push_back( bdf2imex_lake::ExactU2 );
    exact_solution.push_back( bdf2imex_lake::ExactP );
    boundary_conditions.push_back( bdf2imex_lake::BoundCondition );
    boundary_conditions.push_back( bdf2imex_lake::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( bdf2imex_lake::U1BoundValue );
    boundary_data.push_back( bdf2imex_lake::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = bdf2imex_lake::LinCoeffs;
    bdf2imex_lake::ExampleFile();
    // initial condition: the exact solution at time_start
    initialCondition.push_back( bdf2imex_lake::ExactU1 );
    initialCondition.push_back( bdf2imex_lake::ExactU2 );
    initialCondition.push_back( bdf2imex_lake::ExactP );
    break;

    case 27: // BDF2-IMEX / CIP, doubly periodic shear layer
    exact_solution.push_back( bdf2imex_shearlayer::ExactU1 );
    exact_solution.push_back( bdf2imex_shearlayer::ExactU2 );
    exact_solution.push_back( bdf2imex_shearlayer::ExactP );
    boundary_conditions.push_back( bdf2imex_shearlayer::BoundCondition );
    boundary_conditions.push_back( bdf2imex_shearlayer::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( bdf2imex_shearlayer::U1BoundValue );
    boundary_data.push_back( bdf2imex_shearlayer::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = bdf2imex_shearlayer::LinCoeffs;
    bdf2imex_shearlayer::ExampleFile();
    // initial condition: the exact solution at time_start
    initialCondition.push_back( bdf2imex_shearlayer::ExactU1 );
    initialCondition.push_back( bdf2imex_shearlayer::ExactU2 );
    initialCondition.push_back( bdf2imex_shearlayer::ExactP );
    break;

    case 30: // verification: u in P2, p in P1, phi(t) quadratic (TR exact)
    exact_solution.push_back( tnse_poly_quadratic_time::ExactU1 );
    exact_solution.push_back( tnse_poly_quadratic_time::ExactU2 );
    exact_solution.push_back( tnse_poly_quadratic_time::ExactP );
    boundary_conditions.push_back( tnse_poly_quadratic_time::BoundCondition );
    boundary_conditions.push_back( tnse_poly_quadratic_time::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( tnse_poly_quadratic_time::U1BoundValue );
    boundary_data.push_back( tnse_poly_quadratic_time::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = tnse_poly_quadratic_time::LinCoeffs;
    tnse_poly_quadratic_time::ExampleFile();
    initialCondition.push_back( tnse_poly_quadratic_time::ExactU1 );
    initialCondition.push_back( tnse_poly_quadratic_time::ExactU2 );
    initialCondition.push_back( tnse_poly_quadratic_time::ExactP );
    break;

    case 31: // verification: u in P2, p in P1, phi(t) linear (BDF2-IMEX exact)
    exact_solution.push_back( tnse_poly_linear_time::ExactU1 );
    exact_solution.push_back( tnse_poly_linear_time::ExactU2 );
    exact_solution.push_back( tnse_poly_linear_time::ExactP );
    boundary_conditions.push_back( tnse_poly_linear_time::BoundCondition );
    boundary_conditions.push_back( tnse_poly_linear_time::BoundCondition );
    boundary_conditions.push_back( BoundConditionNoBoundCondition );
    boundary_data.push_back( tnse_poly_linear_time::U1BoundValue );
    boundary_data.push_back( tnse_poly_linear_time::U2BoundValue );
    boundary_data.push_back( BoundaryValueHomogenous );
    problem_coefficients = tnse_poly_linear_time::LinCoeffs;
    tnse_poly_linear_time::ExampleFile();
    initialCondition.push_back( tnse_poly_linear_time::ExactU1 );
    initialCondition.push_back( tnse_poly_linear_time::ExactU2 );
    initialCondition.push_back( tnse_poly_linear_time::ExactP );
    break;

    case 32: // translating vortex with compact support (manuscript, Example 2)
    {
      using namespace tnse_translating_vortex;
      set_vortex_parameters(example_database["vortex_radius"],
                            example_database["vortex_exponent"],
                            example_database["vortex_max_swirl"],
                            example_database["vortex_center_x"],
                            example_database["vortex_center_y"],
                            example_database["vortex_velocity_x"],
                            example_database["vortex_velocity_y"]);
      // the pressure mean and the constant boundary data need the support of
      // the vortex inside the unit square on the whole time interval
      double t0 = user_input_parameter_db.contains("time_start")
                  ? (double) user_input_parameter_db["time_start"] : 0.;
      double t1 = user_input_parameter_db.contains("time_end")
                  ? (double) user_input_parameter_db["time_end"] : t0;
      double d = vortex_distance_to_boundary(t0, t1);
      if(d < 0.)
        ErrThrow("translating vortex: the vortex leaves the unit square in [",
                 t0, ", ", t1, "] (distance ", d, ")");
      exact_solution.push_back( ExactU1 );
      exact_solution.push_back( ExactU2 );
      exact_solution.push_back( ExactP );
      boundary_conditions.push_back( BoundCondition );
      boundary_conditions.push_back( BoundCondition );
      boundary_conditions.push_back( BoundConditionNoBoundCondition );
      boundary_data.push_back( U1BoundValue );
      boundary_data.push_back( U2BoundValue );
      boundary_data.push_back( BoundaryValueHomogenous );
      problem_coefficients = LinCoeffs;
      ExampleFile();
      initialCondition.push_back( ExactU1 );
      initialCondition.push_back( ExactU2 );
      initialCondition.push_back( ExactP );
      break;
    }

  default:
      ErrThrow("Unknown time-dependent Example_TimeNSE2D example!");
  }
}

Example_TimeNSE2D::Example_TimeNSE2D(
  const std::vector<DoubleFunct2D*>& exact,
  const std::vector<BoundCondFunct2D*>& bc,
  const std::vector<BoundValueFunct2D*>& bd, const CoeffFct2D& coeffs,
  bool timedependentrhs, bool timedependentcoeffs,
  const std::vector<DoubleFunct2D*>& init_cond)
  : Example_NonStationary2D(exact, bc, bd, coeffs, timedependentrhs,
                            timedependentcoeffs, init_cond)
  {

  }

void Example_TimeNSE2D::do_post_processing(TimeNavierStokes<2>& tnse2d,
                                           double& val) const
{
  if(post_processing_stat)
  {
    post_processing_stat(tnse2d, val);
  }
  else
  {
#ifdef _MPI
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0)
#endif
      Output::info<2>("Example_TimeNSE2D","No post processing done for the current example.");
  }
}
void Example_TimeNSE2D::do_post_processing(TimeNavierStokes<2>& tnse2d) const
{
  if(post_processing_stat_old)
  {
    post_processing_stat_old(tnse2d);
  }
  else
  {
#ifdef _MPI
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0)
#endif
      Output::info<2>("Example_TimeNSE2D","No post processing done for the current example.");
  }
}

double Example_TimeNSE2D::get_nu() const
{
  int example_code = this->example_database["example"];
  double inverse_reynolds;
  double nu=-1.;

  if (example_code == 8) {
    nu = this->example_database["effective_viscosity"];
  } else {
    inverse_reynolds = this->example_database["reynolds_number"];
    nu = 1/inverse_reynolds;
  }
  return nu;
}

double Example_TimeNSE2D::get_inverse_permeability() const
{
  return this->example_database["inverse_permeability"];
}


std::vector<size_t> Example_TimeNSE2D::get_neumann_id() const
{
  std::vector<size_t> empty_vector;
  int n_neumann_bd = this->example_database["n_neumann_bd"];
  if (n_neumann_bd){
    return this->example_database["neumann_id"];
  } else {
    return empty_vector;
  }
}


std::vector<size_t> Example_TimeNSE2D::get_nitsche_id() const
{
  std::vector<size_t> empty_vector;
  int n_nitsche_bd = this->example_database["n_nitsche_bd"];
  if (n_nitsche_bd){
    return this->example_database["nitsche_id"];
  } else {
    return empty_vector;
  }
}

std::vector<size_t> Example_TimeNSE2D::get_windkessel_id() const
{
  std::vector<size_t> empty_vector;
  int n_windkessel_bd = this->example_database["n_windkessel_bd"];
  if (n_windkessel_bd){
    return this->example_database["windkessel_id"];
  } else {
    return empty_vector;
  }
}

Example_NSE2D Example_TimeNSE2D::get_stationary_example() const
{
  std::vector<BoundCondFunct2D*> bc(this->get_bc(), this->get_bc() + 3);
  std::vector<BoundValueFunct2D*> bd(this->get_bd(), this->get_bd() + 3);
  return Example_NSE2D(this->get_exact(), bc, bd, this->get_coeffs(),
                       this->get_nu());
}
