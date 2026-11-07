// =============================================================================
// Adaptive SUPG/PSPG/grad-div method for the time-dependent Navier-Stokes
// equations with the class structure: TimeNavierStokesSPG (System),
// TrapezoidalAB2 (TimeDiscRoutines), AdaptiveTimeLoop (System),
// NSE_SPG_ErrorEstimator in transient mode, SolutionTransfer (FE).
// Required: space_discretization_type supg_pspg_graddiv, NSTYPE 14,
// problem_type 5, example 23 or 24. Parameters: see
// AdaptiveTimeLoop::default_database(), tnse_write_table.
// =============================================================================
#include <Database.h>
#include <Domain.h>
#include <Example_NSE2D.h>
#include "ParMooN.h"
#include "NSE_SPG_ErrorEstimator.h"
#include "AdaptiveTimeLoop.h"
#include "TimeNavierStokesSPG.h"
#include <fstream>

int main(int argc, char* argv[])
{
  ParameterDatabase defaults("TNSE2D adaptive SPG defaults");
  defaults.add("tnse_write_table", "", "file name for the table");
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
  domain.refine_and_get_hierarchy_of_collections(parmoon_db);
  Example_NSE2D example(parmoon_db);
  NSE_SPG_ErrorEstimator estimator(parmoon_db);

  AdaptiveTimeLoop loop(domain, parmoon_db, estimator,
    [&]() { return std::unique_ptr<AdaptiveTimeSystem>(
              new TimeNavierStokesSPG(domain, parmoon_db, example)); });
  loop.run();

  Output::print("\n", loop.get_table());
  std::string table_file = parmoon_db["tnse_write_table"];
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << loop.get_table();
  }
  parmoon::parmoon_finalize();
}
