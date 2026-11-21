#ifdef __2D__
#include "AdaptiveTimeLoop.h"
#include "BaseCell.h"
#include "Collection.h"
#include "MooNMD_Io.h"
#include "Vertex.h"
#include <fstream>
#include <algorithm>
#include <cmath>
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
  db.add("adaptive_coarsening_min_fraction", 0.05, "the time loop changes "
         "the mesh if a cell is marked for refinement or if the children of the "
         "parents that can be derefined are at least this fraction of the "
         "cells (0: any possible derefinement)", 0., 1.);
  db.add("adaptive_be_only_after_coarsening", false, "with "
         "trab2_be_after_mesh_change: take the backward Euler step only after "
         "mesh changes that coarsen (default false: after every mesh change; "
         "skipping it after refinements makes the step control oscillate for "
         "small trab2_tolt)");
  db.add("adaptive_remesh_min_steps", 1u, "the time loop changes the mesh "
         "at most every this many time steps (1: whenever cells are marked)",
         1u, 1000u);
  db.add("adaptive_remesh_force_factor", 0., "change the mesh before "
         "adaptive_remesh_min_steps steps have passed if a cell has eta_K > "
         "this factor * adaptive_tol1 (0: never)", 0., 1.e10);
  db.add("adaptive_initial_mesh", true, "adapt the mesh to the initial "
         "condition before the first step (with adaptive_mesh): project u_0, "
         "estimate (Navier-Stokes: after a trial backward Euler step of "
         "the first step size, which is discarded), refine and coarsen as in "
         "the time loop "
         "(adaptive_tol1, adaptive_tol2, adaptive_max_cell_level), repeat "
         "until the number of cells changes by less than 0.5 % or "
         "adaptive_initial_max_steps is reached");
  db.add("adaptive_initial_max_steps", 10u, "maximal number of refinement "
         "steps of the initial mesh adaptation", 0u, 100u);
  db.add("adaptive_write_mesh", true, "write the mesh at the start and at every "
         "output time as <output_directory>/<output_basename>_mesh_t<time>.vtk "
         "(cell data: refinement level, diameter, eta_K and, with an exact "
         "solution, the local error error_K) and, with "
         "output_write_ps, as mesh_t<time>.ps");
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
  write_mesh = db["adaptive_write_mesh"];
  {
    std::string dir = param_db.contains("output_directory")
                      ? param_db["output_directory"].value_as_string() : "";
    std::string base = param_db.contains("output_basename")
                       ? param_db["output_basename"].value_as_string() : "";
    if(!base.empty() && base.back() != '_')
      base += "_";
    mesh_prefix = (dir.empty() ? std::string() : dir + "/") + base + "mesh_t";
  }
  coarsen_fraction = db["adaptive_coarsening_min_fraction"];
  be_only_after_coarsening = db["adaptive_be_only_after_coarsening"];
  remesh_min_steps = db["adaptive_remesh_min_steps"];
  remesh_force_factor = db["adaptive_remesh_force_factor"];
  steps_since_remesh = 0;
  initial_adaptation = db["adaptive_initial_mesh"];
  initial_max_steps = db["adaptive_initial_max_steps"];
  system = factory();
  system->initialize();
  if(adaptive && initial_adaptation)
    adapt_initial_mesh();
  next_output = integrator.get_time() + output_every;
  output_index = 0;
  // initial mesh (eta_K of the last initial adaptation step, if any)
  write_mesh_files(integrator.get_time(), adaptive && initial_adaptation);
  system->output(output_index++); // initial condition
  table << "# time  dt  cells  dofs  eta  error  effectivity  n_marked_refine"
           "  n_marked_coarsen";
  for(const auto& col : system->error_columns())
    table << "  " << col.first;
  table << "\n";
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

void AdaptiveTimeLoop::adapt_initial_mesh()
{
  // the error of an unresolved initial condition is not removed by refining
  // later (it is transported and decays only on the viscous time scale), so
  // the mesh is adapted to u_0 with the criterion of the time loop
  const bool trial = system->has_be_step();
  const double t0 = integrator.get_time();
  const double k = integrator.get_step();
  bool fresh = true; // the system holds the projected u_0
  for(size_t step = 0; ; ++step)
  {
    if(trial)
    {
      // estimate the state after a backward Euler step of the first step
      // size: it damps the inconsistent (stiff) part of the projection, which
      // otherwise inflates eta_K in the whole domain and prevents coarsening
      system->save_state();
      system->predict(k, 1.);
      system->be_step(t0, k);
      fresh = false;
    }
    system->estimate(estimator);
    const TCollection* coll = system->get_collection();
    const int n_cells = coll->GetN_Cells();
    std::vector<bool> marks_refine(n_cells, false), marks_coarsen(n_cells, false);
    // the marking of the time loop (refine eta_K > TOL1, coarsen
    // eta_K < TOL2), so that the first steps do not undo the adaptation
    int n_marked = 0, n_coarsen = 0;
    for(int c = 0; c < n_cells; ++c)
    {
      const double eta_K = estimator.get_eta_K(c);
      if(eta_K > tol1 && coll->GetCell(c)->GetGeoLevel() < (int) max_level)
      { marks_refine[c] = true; n_marked++; }
      else if(coarsening && eta_K < tol2
              && coll->GetCell(c)->GetGeoLevel() > initial_level)
      { marks_coarsen[c] = true; n_coarsen++; }
    }
    Output::print("initial mesh adaptation, step ", step, ": eta = ",
                  estimator.get_estimated_global_error(), ", cells = ",
                  n_cells, ", dofs = ", system->get_n_dofs(), ", marked ",
                  n_marked, " for refinement, ", n_coarsen, " for coarsening");
    if(n_marked == 0 && n_coarsen == 0)
      break;
    if(step == initial_max_steps)
    {
      Output::warn("AdaptiveTimeLoop", "initial mesh adaptation stopped after ",
                   initial_max_steps, " steps with ", n_marked, " cells marked "
                   "for refinement and ", n_coarsen, " for coarsening "
                   "(adaptive_initial_max_steps)");
      break;
    }
    const int n_derefined = domain.RefineAndCoarsen(marks_refine, marks_coarsen,
                                                    initial_level);
    const int n_new = domain.GetCollection(It_Finest, 0)->GetN_Cells();
    Output::print("initial mesh adaptation: ", n_derefined, " parents derefined, "
                  "new mesh has ", n_new, " cells");
    // the mesh has settled when it changes by less than 0.5 % (a few cells
    // near the maximal level can be refined and coarsened in turn)
    const bool settled = std::abs(n_new - n_cells) < 0.005*n_cells;
    // the old system refers to the replaced collection: rebuild and project
    // u_0 again on the new mesh (no transfer)
    system = factory();
    system->initialize();
    fresh = true;
    if(settled)
      break;
  }
  if(!fresh)
  {
    // discard the trial step: start from the projected u_0
    system = factory();
    system->initialize();
  }
}

void AdaptiveTimeLoop::write_mesh_files(double t, bool with_eta)
{
  std::ostringstream ts;
  ts << std::fixed << std::setprecision(3) << t;
  if(write_ps)
    domain.PS(("mesh_t" + ts.str() + ".ps").c_str(), It_Finest, 0);
  if(!write_mesh)
    return;
  // every cell with its own vertices (no global numbering needed); ParaView:
  // "Surface With Edges", colour by level
  const TCollection* coll = system->get_collection();
  const int n_cells = coll->GetN_Cells();
  int n_points = 0;
  for(int c = 0; c < n_cells; ++c)
    n_points += coll->GetCell(c)->GetN_Vertices();
  const std::string name = mesh_prefix + ts.str() + ".vtk";
  std::ofstream f(name);
  if(!f)
  {
    Output::warn("AdaptiveTimeLoop", "cannot write the mesh file ", name,
                 " (does the output directory exist?)");
    return;
  }
  f << "# vtk DataFile Version 3.0\nParMooN adaptive mesh, t = "
    << std::setprecision(12) << t << ", " << n_cells << " cells\nASCII\n"
    << "DATASET UNSTRUCTURED_GRID\nPOINTS " << n_points << " double\n";
  for(int c = 0; c < n_cells; ++c)
  {
    const TBaseCell* cell = coll->GetCell(c);
    for(int v = 0; v < cell->GetN_Vertices(); ++v)
    {
      double x, y;
      cell->GetVertex(v)->GetCoords(x, y);
      f << x << " " << y << " 0\n";
    }
  }
  f << "CELLS " << n_cells << " " << n_cells + n_points << "\n";
  int offset = 0;
  for(int c = 0; c < n_cells; ++c)
  {
    const int nv = coll->GetCell(c)->GetN_Vertices();
    f << nv;
    for(int v = 0; v < nv; ++v)
      f << " " << offset + v;
    f << "\n";
    offset += nv;
  }
  f << "CELL_TYPES " << n_cells << "\n";
  for(int c = 0; c < n_cells; ++c)
    f << (coll->GetCell(c)->GetN_Vertices() == 3 ? 5 : 9) << "\n";
  f << "CELL_DATA " << n_cells << "\nSCALARS level int 1\nLOOKUP_TABLE default\n";
  for(int c = 0; c < n_cells; ++c)
    f << coll->GetCell(c)->GetGeoLevel() << "\n";
  f << "SCALARS diameter double 1\nLOOKUP_TABLE default\n";
  for(int c = 0; c < n_cells; ++c)
    f << coll->GetCell(c)->GetDiameter() << "\n";
  if(with_eta && (int) estimator.get_eta_K().size() == n_cells)
  {
    f << "SCALARS eta_K double 1\nLOOKUP_TABLE default\n";
    for(int c = 0; c < n_cells; ++c)
      f << estimator.get_eta_K(c) << "\n";
    // exact local error in the estimator's norm (examples with exact solution)
    const std::vector<double>& eK = estimator.get_error_K();
    if((int) eK.size() == n_cells)
    {
      f << "SCALARS error_K double 1\nLOOKUP_TABLE default\n";
      for(int c = 0; c < n_cells; ++c)
        f << eK[c] << "\n";
    }
  }
  Output::print<2>("mesh written to ", name);
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
    if(system->estimate_time() >= 0.)
      t = system->estimate_time();
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
          << " " << n_coarsen;
    std::ostringstream extra;
    for(const auto& col : system->error_columns())
      extra << " " << std::scientific << std::setprecision(6) << col.second;
    table << extra.str() << "\n";
    if(table_file.is_open())
    {
      table_file << std::scientific << std::setprecision(6) << t << " " << k_eff
                 << " " << n_cells << " " << system->get_n_dofs() << " " << eta
                 << " " << error << " " << std::fixed << std::setprecision(4)
                 << eff << " " << n_marked << " " << n_coarsen << extra.str()
                 << "\n" << std::flush;
    }

    // --- output ----------------------------------------------------------------
    if(t >= next_output - 1e-12)
    {
      next_output += output_every;
      write_mesh_files(t, true);
      system->output(output_index++);
    }

    // --- mesh adaptation -------------------------------------------------------
    // remesh only if a cell is marked for refinement or enough cells are
    // marked for coarsening: every remesh interpolates the state, which is then
    // inconsistent with the discrete constraint again; remeshing for a few
    // coarsened parents in every step stalls the step size control
    // (counted are the children of the parents that can really be derefined:
    // marked cells next to refined cells cannot, see RefineAndCoarsen)
    bool remesh = n_marked > 0;
    if(adaptive && !remesh && n_coarsen > 0)
    {
      const int n_par = domain.N_DerefinableParents(marks_refine, marks_coarsen,
                                                    initial_level);
      remesh = n_par > 0 && 4.*n_par >= coarsen_fraction*n_cells;
    }
    // at most every remesh_min_steps steps (each remesh is followed by a
    // backward Euler step and disturbs the step control), unless a cell
    // exceeds remesh_force_factor * TOL1
    ++steps_since_remesh;
    if(remesh && steps_since_remesh < remesh_min_steps)
    {
      const bool force = remesh_force_factor > 0.
        && estimator.get_maximal_local_error() > remesh_force_factor*tol1;
      if(!force)
        remesh = false;
    }
    if(adaptive && remesh)
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
      steps_since_remesh = 0;
      // a perturbation only if the mesh has changed (the transfer to the same
      // mesh is exact)
      if(n_marked > 0 || n_derefined > 0
         || system->get_collection()->GetN_Cells() != n_cells)
        // also a pure refinement leaves the state inconsistent with the
        // discrete constraint of the new mesh (more pressure test functions,
        // new delta_K), so by default every mesh change is followed by a
        // backward Euler step; without it the step control oscillates for
        // small TOLT (LAYER_SWITCH_NOTES.md, 3.13)
        integrator.restart_averaging(!(be_only_after_coarsening
                                       && n_derefined == 0));
    }
  }
  Output::print("TR-AB2: ", integrator.n_accepted_steps(), " accepted and ",
                integrator.n_rejected_steps(), " rejected steps (",
                integrator.n_be_steps(), " backward Euler)");
}
#endif // __2D__
