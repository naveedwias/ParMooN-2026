// =============================================================================
// Adaptive SUPG method for evolutionary convection-diffusion-reaction equations
// (de Frutos, Garcia-Archilla, John, Novo, CMAME 273 (2014) 219-237) with the
// class structure: TimeConvectionDiffusionTR (System), TrapezoidalAB2
// (TimeDiscRoutines), AdaptiveTimeLoop (System), CDErrorEstimator type 5 with
// the discrete time derivative, SolutionTransfer (FE).
// Parameters: see AdaptiveTimeLoop::default_database() (adaptive_mesh,
// adaptive_tol1/2, trab2_*), tcd_estimator_mu0 (mu_0 in the estimator),
// tcd_write_table.
// =============================================================================
#include <Database.h>
#include <Domain.h>
#include <Example_TimeCD2D.h>
#include "ParMooN.h"
#include "CDErrorEstimator.h"
#include "AdaptiveTimeLoop.h"
#include "TimeConvectionDiffusionTR.h"
#include <fstream>

int main(int argc, char* argv[])
{
  ParameterDatabase defaults("TCD2D adaptive TR defaults");
  defaults.add("tcd_estimator_mu0", 1., "lower bound mu_0 of c - div(b)/2 used "
               "in the estimator; 0 drops the corresponding terms", 0., 1.e10);
  defaults.add("tcd_write_table", "", "file name for the table");
  defaults.merge(AdaptiveTimeLoop::default_database(), true);
  defaults.merge(ErrorEstimator<2>::default_error_estimator_database(), true);
  auto parmoon_db = parmoon::parmoon_initialize(argc, argv);
  defaults.merge(parmoon_db, true);
  parmoon_db = defaults;
  parmoon_db["estimator_type"] = 5; // John-Novo SUPG estimator
  TDatabase::ParamDB->INTERNAL_COERCIVITY = parmoon_db["tcd_estimator_mu0"];

  TDomain domain(parmoon_db);
  domain.refine_and_get_hierarchy_of_collections(parmoon_db);
  Example_TimeCD2D example(parmoon_db);
  CDErrorEstimator<2> estimator(parmoon_db);

  AdaptiveTimeLoop loop(domain, parmoon_db, estimator,
    [&]() { return std::unique_ptr<AdaptiveTimeSystem>(
              new TimeConvectionDiffusionTR(domain, parmoon_db, example)); });
  loop.run();

  Output::print("\n", loop.get_table());
  std::string table_file = parmoon_db["tcd_write_table"];
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << loop.get_table();
  }
  parmoon::parmoon_finalize();
}
