// =============================================================================
// SUPG/PSPG/grad-div stabilized finite element method for the Oseen problem
// (problem_type 3) or the stationary Navier-Stokes problem (problem_type 5)
// with the residual-based a posteriori error estimator and adaptive mesh
// refinement of
//   M. Afzal, N. Ahmed, V. John, A robust a posteriori error estimator for the
//   Oseen problem, Comput. Math. Appl. 223 (2026) 467-486.
//
// Required settings in the input file:
//   space_discretization_type: supg_pspg_graddiv
//   NSTYPE: 14
//   example: 21 (Example 4.1 / 5.1) or 22 (Example 4.2 / 5.2)
//   problem_type: 3 (Oseen) or 5 (Navier-Stokes)
//   conforming_closure: true
// Adaptive refinement is controlled by 'refinement_max_n_adaptive_steps'
// (> 0) and the refinement strategy parameters; with
// 'spg_uniform_refinement: true' the grid is refined uniformly instead
// (convergence study as in Section 4.1).
// A summary table (level, cells, DoFs, h, error, eta, parts, effectivity) is
// printed at the end.
// =============================================================================
#include <Domain.h>
#include <Database.h>
#include "NavierStokes.h"
#include <Example_NSE2D.h>
#include <Chrono.h>
#include <LoopInfo.h>
#include <ParameterDatabase.h>
#include "ParMooN.h"
#include <RefinementStrategy.h>
#include "NSE_SPG_ErrorEstimator.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

struct LevelData
{
  int level, n_cells, n_dofs;
  double h, error, eta, res, div, F, delta, mu, eff, div_uh;
};

int main(int argc, char* argv[])
{
  Chrono timer;
  parmoon::parmoon_initialize(argc, argv);
  ParameterDatabase parmoon_db = ParameterDatabase::parmoon_default_database();
  parmoon_db.merge(NavierStokes<2>::default_nse_database());
  parmoon_db.merge(NSE_SPG_ErrorEstimator::default_spg_estimator_database(), true);
  parmoon_db.add("spg_uniform_refinement", false,
                 "refine uniformly instead of adaptively (convergence study)");
  parmoon_db.add("spg_write_table", "", "file name for the summary table");
  parmoon_db.read(argv[1]);

  // the examples take the viscosity from TDatabase::ParamDB->RE_NR
  TDatabase::ParamDB->RE_NR = parmoon_db["reynolds_number"];
  bool linear_problem = parmoon_db["problem_type"].is(3);
  TDatabase::ParamDB->INTERNAL_PROBLEM_LINEAR = linear_problem;
  // the examples use FLOW_PROBLEM_TYPE to distinguish Oseen and Navier-Stokes
  TDatabase::ParamDB->FLOW_PROBLEM_TYPE = linear_problem ? 3 : 5;
  if(!parmoon_db["space_discretization_type"].is("supg_pspg_graddiv"))
  {
    Output::warn("Oseen2D_SPG_Adaptive", "space_discretization_type is not "
                 "'supg_pspg_graddiv'; the estimator assumes this method.");
  }

  TDomain domain(parmoon_db);
  check_parameters_consistency_NSE(parmoon_db);
  domain.refine_and_get_hierarchy_of_collections(parmoon_db);

  std::string base_name = parmoon_db["output_basename"];
  bool uniform = parmoon_db["spg_uniform_refinement"];

  NSE_SPG_ErrorEstimator estimator {parmoon_db};
  RefinementStrategy<2> refinementStrategy(parmoon_db);
  LoopInfo<double> adaptive_loop_info("adaptive", true, true, 1);
  BlockVector values;
  size_t n_adaptive_steps = domain.get_database()["refinement_max_n_adaptive_steps"];
  std::vector<LevelData> table;

  for(size_t curr_level = 0; ; ++curr_level)
  {
    Output::print("\nadaptive loop ", curr_level);
    std::ostringstream ostr;
    ostr << base_name << "_" << std::setw(2) << std::setfill('0') << curr_level;
    parmoon_db["output_basename"].set(ostr.str(), false);

    NavierStokes<2> ns(domain, parmoon_db);
    ns.assemble_linear_terms();
    ns.stop_it(0);
    LoopInfo<Residuals> nonlinear_loop_info("nonlinear");
    nonlinear_loop_info.print_time_every_step = true;
    nonlinear_loop_info.verbosity_threshold = 1;
    nonlinear_loop_info.print(0, ns.get_residuals());
    timer.restart_and_print("setting up spaces, matrices, linear assemble");

    for(unsigned int k = 1;; k++)
    {
      ns.solve();
      if(linear_problem)
        break;
      ns.assemble_nonlinear_term();
      if(ns.stop_it(k))
      {
        nonlinear_loop_info.finish(k, ns.get_residuals());
        break;
      }
      else
        nonlinear_loop_info.print(k, ns.get_residuals());
    }
    timer.restart_and_print("solving procedure");

    estimator.estimate(ns.get_example(), ns.get_velocity(), ns.get_pressure());
    estimator.info();
    timer.restart_and_print("error estimation");

    NavierStokes<2>::FEFunction estimated_error(estimator, values);
    ns.add_to_output(&estimated_error);
    ns.output();

    LevelData ld;
    ld.level = curr_level;
    ld.n_cells = domain.GetCollection(It_Finest, 0)->GetN_Cells();
    ld.n_dofs = ns.get_size();
    ld.h = estimator.get_h();
    ld.error = estimator.get_error_spg();
    ld.eta = estimator.get_estimated_global_error();
    ld.res = estimator.get_eta_res();
    ld.div = estimator.get_eta_div();
    ld.F = estimator.get_eta_F();
    ld.delta = estimator.get_eta_delta();
    ld.mu = estimator.get_eta_mu();
    ld.eff = estimator.get_effectivity_index();
    ld.div_uh = estimator.get_div_uh();
    table.push_back(ld);

    if(curr_level == n_adaptive_steps)
    {
      adaptive_loop_info.finish(curr_level, estimator.get_estimated_global_error());
      break;
    }
    adaptive_loop_info.print(curr_level, estimator.get_estimated_global_error());
    if(uniform)
    {
      // mark all cells: regular refinement of the whole grid
      std::vector<bool> all(domain.GetCollection(It_Finest, 0)->GetN_Cells(), true);
      refinementStrategy.apply_indicator(all);
    }
    else
    {
      refinementStrategy.apply_estimator(estimator);
    }
    domain.RefineByRefinementStrategy(refinementStrategy);
    if(domain.get_database()["write_adaptive_mesh"])
    {
      std::string name_after = std::string("domain_after")
                             + std::to_string(curr_level) + ".ps";
      domain.PS(name_after.c_str(), It_Finest, 0);
    }
  }

  // summary table
  std::ostringstream t;
  t << "# level  cells  dofs  h  error_spg  eta  eta_res  eta_div  eta_F  "
       "eta_delta  eta_mu  effectivity  div_uh  rate_err  rate_eta\n";
  for(size_t i = 0; i < table.size(); ++i)
  {
    const LevelData& l = table[i];
    double rate_e = 0., rate_eta = 0.;
    if(i > 0 && table[i-1].error > 0. && l.error > 0.)
      rate_e = std::log(table[i-1].error/l.error)
               / (0.5*std::log(double(l.n_dofs)/table[i-1].n_dofs));
    if(i > 0 && table[i-1].eta > 0.)
      rate_eta = std::log(table[i-1].eta/l.eta)
                 / (0.5*std::log(double(l.n_dofs)/table[i-1].n_dofs));
    t << std::setw(5) << l.level << " " << std::setw(8) << l.n_cells << " "
      << std::setw(9) << l.n_dofs << " " << std::scientific << std::setprecision(4)
      << l.h << " " << l.error << " " << l.eta << " " << l.res << " " << l.div
      << " " << l.F << " " << l.delta << " " << l.mu << " "
      << std::fixed << std::setprecision(3) << l.eff << " "
      << std::scientific << std::setprecision(3) << l.div_uh << " "
      << std::fixed << std::setprecision(2) << rate_e << " " << rate_eta << "\n";
  }
  Output::print("\nSummary (rates are w.r.t. DoFs^{-1/2}, i.e., h in 2D):\n", t.str());
  std::string table_file = parmoon_db["spg_write_table"];
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << t.str();
  }
  parmoon::parmoon_finalize();
}
