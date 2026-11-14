// =============================================================================
// Adaptive SUPG/PSPG/grad-div method for the time-dependent Navier-Stokes
// equations with the class structure: TimeNavierStokesSPG (System),
// TrapezoidalAB2 (TimeDiscRoutines), AdaptiveTimeLoop (System),
// NSE_SPG_ErrorEstimator in transient mode, SolutionTransfer (FE).
// Required: space_discretization_type supg_pspg_graddiv, NSTYPE 14,
// problem_type 5, example 23, 24 or 28 of Example_TimeNSE2D. Parameters: see
// AdaptiveTimeLoop::default_database(), tnse_write_table.
// =============================================================================
#include <Database.h>
#include <Domain.h>
#include <Example_TimeNSE2D.h>
#include "ParMooN.h"
#include "NSE_SPG_ErrorEstimator.h"
#include "AdaptiveTimeLoop.h"
#include "TimeNavierStokesSPG.h"
#include <fstream>

int main(int argc, char* argv[])
{
  ParameterDatabase defaults("TNSE2D adaptive SPG defaults");
  defaults.add("tnse_write_table", "", "file name for the table");
  defaults.add("tnse_initial_value", "l2_projection", "initial velocity: "
               "l2_projection (L2 projection of u(time_start), as in the "
               "manuscript), consistent_projection (stabilized equations at "
               "time_start with the exact time derivative; consistent with the "
               "discrete constraint) or interpolation (nodal interpolant)",
               {"l2_projection", "consistent_projection", "interpolation"});
  defaults.add("tnse_dt_alpha", 0.5, "lower bound dt >= alpha min_K delta_K of "
               "the time step in the step size control (residual-based pressure "
               "stabilization, manuscript Section 4); 0 switches it off; the "
               "constant has to be calibrated", 0., 1.e10);
  defaults.merge(NavierStokes<2>::default_nse_database(), true);
  defaults.merge(NSE_SPG_ErrorEstimator::default_spg_estimator_database(), true);
  defaults.merge(AdaptiveTimeLoop::default_database(), true);
  auto parmoon_db = parmoon::parmoon_initialize(argc, argv);
  defaults.merge(parmoon_db, true);
  parmoon_db = defaults;

  TDatabase::ParamDB->RE_NR = parmoon_db["reynolds_number"];
  TDatabase::ParamDB->INTERNAL_PROBLEM_LINEAR = 0;
  TDatabase::ParamDB->FLOW_PROBLEM_TYPE = 5;
  parmoon_db["problem_type"] = 5;
  if(!parmoon_db["space_discretization_type"].is("supg_pspg_graddiv"))
    ErrThrow("space_discretization_type must be supg_pspg_graddiv");

  TDomain domain(parmoon_db);
  check_parameters_consistency_NSE(parmoon_db);
  Example_TimeNSE2D example(parmoon_db);
  // periodic boundaries: if the example marks a boundary part PERIODIC, the
  // boundary condition function is passed to the refinement, which then pairs
  // the joints of opposite sides (periodic_translations). Periodic joints are
  // set up on the uniformly refined mesh only; the local refinement and
  // coarsening of the adaptive loop does not keep them, so adaptive meshes
  // are not allowed with periodic boundaries.
  bool periodic = false;
  for(int c = 0; c < domain.GetBdPart(0)->GetN_BdComps(); ++c)
  {
    BoundCond cond;
    example.get_bc(0)(c, 0.5, cond);
    periodic = periodic || (cond == PERIODIC);
  }
  if(periodic && (bool) parmoon_db["adaptive_mesh"])
    ErrThrow("adaptive meshes are not supported with periodic boundary "
             "conditions: set 'adaptive_mesh: false'");
  domain.refine_and_get_hierarchy_of_collections(parmoon_db,
    periodic ? example.get_bc(0) : nullptr);
  NSE_SPG_ErrorEstimator estimator(parmoon_db);

  AdaptiveTimeLoop loop(domain, parmoon_db, estimator,
    [&]() { return std::unique_ptr<AdaptiveTimeSystem>(
              new TimeNavierStokesSPG(domain, parmoon_db, example)); });
  loop.run();

  Output::print("\n", loop.get_table());
  parmoon::parmoon_finalize();
}
