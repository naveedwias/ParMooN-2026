#ifdef __2D__
#include "AdaptiveTimeLoop.h"
#include "BaseCell.h"
#include "Collection.h"
#include "MooNMD_Io.h"
#include <algorithm>
#include <iomanip>

ParameterDatabase AdaptiveTimeLoop::default_database()
{
  ParameterDatabase db("adaptive time loop database");
  db.add("adaptive_mesh", true, "adaptive mesh refinement/coarsening in time");
  db.add("adaptive_tol1", 1.e-3, "cells with eta_K > TOL1 are refined", 0., 1.e10);
  db.add("adaptive_tol2", 5.e-5, "cells with eta_K < TOL2 are candidates for "
         "coarsening", 0., 1.e10);
  db.add("adaptive_coarsening", true, "coarsening on/off");
  db.add("adaptive_max_cell_level", 30u, "maximal refinement level of a cell",
         1u, 100u);
  db.add("adaptive_output_time", 0.1, "interval for mesh output (PostScript, "
         "needs output_write_ps) and system output (vtk, needs output_write_vtk; "
         "files <output_directory>/<output_basename>.<index>.vtk)", 0., 1.e10);
  db.merge(TrapezoidalAB2::default_database(), true);
  return db;
}

AdaptiveTimeLoop::AdaptiveTimeLoop(TDomain& dom, const ParameterDatabase& param_db,
                                   ErrorEstimator<2>& est, SystemFactory fac)
 : domain(dom), db(default_database()), estimator(est), factory(fac),
   integrator(param_db)
{
  db.merge(param_db, false);
  adaptive = db["adaptive_mesh"];
  tol1 = db["adaptive_tol1"];
  tol2 = db["adaptive_tol2"];
  coarsening = db["adaptive_coarsening"];
  max_level = db["adaptive_max_cell_level"];
  output_every = db["adaptive_output_time"];
  initial_level = param_db.contains("refinement_n_initial_steps")
                  ? (int) param_db["refinement_n_initial_steps"] : 0;
  write_ps = param_db.contains("output_write_ps")
             ? (bool) param_db["output_write_ps"] : false;
  system = factory();
  system->initialize();
  next_output = integrator.get_time() + output_every;
  output_index = 0;
  system->output(output_index++); // initial condition
  table << "# time  dt  cells  dofs  eta  error  effectivity  n_marked_refine"
           "  n_marked_coarsen\n";
  // the table is written and flushed after every accepted step, so that an
  // interrupted run still leaves its table behind
  std::string file;
  for(const char* key : {"adaptive_table_file", "tcd_write_table",
                         "tnse_write_table"})
  {
    if(param_db.contains(key))
      file = param_db[key].value_as_string();
    if(!file.empty())
      break;
  }
  if(!file.empty())
  {
    table_file.open(file);
    table_file << table.str() << std::flush;
  }
}

void AdaptiveTimeLoop::run()
{
  while(!integrator.finished())
  {
    integrator.advance(*system);
    double t = integrator.get_time();
    double k_eff = integrator.get_previous_step();

    // --- estimator -----------------------------------------------------------
    system->estimate(estimator);
    double eta = estimator.get_estimated_global_error();
    double error = system->exact_error();
    double eff = (error > 0.) ? eta/error : 0.;
    const TCollection* coll = system->get_collection();
    int n_cells = coll->GetN_Cells();
    Output::print("estimator: eta = ", eta, ", error = ", error,
                  ", effectivity = ", eff, ", cells = ", n_cells);

    // --- marking -------------------------------------------------------------
    int n_marked = 0, n_coarsen = 0;
    std::vector<bool> marks_refine(n_cells, false), marks_coarsen(n_cells, false);
    if(adaptive)
    {
      for(int c = 0; c < n_cells; ++c)
      {
        double eta_K = estimator.get_eta_K(c);
        if(eta_K > tol1 && coll->GetCell(c)->GetGeoLevel() < (int) max_level)
        { marks_refine[c] = true; n_marked++; }
        else if(coarsening && eta_K < tol2)
        { marks_coarsen[c] = true; n_coarsen++; }
      }
    }
    table << std::scientific << std::setprecision(6) << t << " " << k_eff << " "
          << n_cells << " " << system->get_n_dofs() << " " << eta << " " << error
          << " " << std::fixed << std::setprecision(4) << eff << " " << n_marked
          << " " << n_coarsen << "\n";
    if(table_file.is_open())
    {
      table_file << std::scientific << std::setprecision(6) << t << " " << k_eff
                 << " " << n_cells << " " << system->get_n_dofs() << " " << eta
                 << " " << error << " " << std::fixed << std::setprecision(4)
                 << eff << " " << n_marked << " " << n_coarsen << "\n"
                 << std::flush;
    }

    // --- output ----------------------------------------------------------------
    if(t >= next_output - 1e-12)
    {
      next_output += output_every;
      if(write_ps)
      {
        std::ostringstream name;
        name << "mesh_t" << std::fixed << std::setprecision(3) << t << ".ps";
        domain.PS(name.str().c_str(), It_Finest, 0);
      }
      system->output(output_index++);
    }

    // --- mesh adaptation -------------------------------------------------------
    if(adaptive && (n_marked > 0 || n_coarsen > 0))
    {
      auto state = system->record_state();
      int n_derefined = domain.RefineAndCoarsen(marks_refine, marks_coarsen,
                                                initial_level);
      // the system is rebuilt even if no cell changed: RefineAndCoarsen
      // recreates the collection of the finest level, so the old system may
      // refer to a deleted collection
      std::unique_ptr<AdaptiveTimeSystem> new_system = factory();
      Output::print("adaptive: ", n_marked, " marked, ", n_derefined,
                    " parents derefined, new mesh has ",
                    domain.GetCollection(It_Finest, 0)->GetN_Cells(), " cells");
      new_system->copy_time_state(*system);
      new_system->apply_state(state);
      system = std::move(new_system);
      integrator.restart_averaging();
    }
  }
  Output::print("TR-AB2: ", integrator.n_accepted_steps(), " accepted and ",
                integrator.n_rejected_steps(), " rejected steps");
}
#endif // __2D__
