#include <Simplified/SimplifiedInfo.h>
#include <Solver.h>

ParameterDatabase SimplifiedInfo::default_solver_db()
{
  ParameterDatabase db = Solver<>::default_solver_database();

  db["solver_type"].set("iterative", false);
  db["iterative_solver_type"].set("fgmres", false);
  db["residual_tolerance"].set(1e-10, false);
  db["min_n_iterations"].set(10, false);
  db["max_n_iterations"].set(1000, false);
  db["preconditioner"].set("bi_cgstab_jacobi", false);
  db["prec_iterative_min_it"].set(10, false);
  db["prec_iterative_max_it"].set(100, false);
  db["prec_iterative_residual_reduction"].set(0.1, false);
  db["prec_optimize_residual"].set(true, false);

  return db;
}

ParameterDatabase SimplifiedInfo::default_saddle_solver_db(bool direct_u, bool direct_p)
{
  ParameterDatabase db = Solver<>::default_solver_database();

  db["solver_type"].set("iterative", false);
  db["iterative_solver_type"].set("fgmres", false);
  db["residual_tolerance"].set(1e-10, false);
  db["min_n_iterations"].set(10, false);
  db["max_n_iterations"].set(1000, false);
  db["preconditioner"].set("least_squares_commutator", false);

  ParameterDatabase v_db = direct_u ? SimplifiedInfo::default_direct_solver_db() : SimplifiedInfo::default_solver_db();
  ParameterDatabase p_db = direct_p ? SimplifiedInfo::default_direct_solver_db() : SimplifiedInfo::default_solver_db();

  v_db.set_name("Saddle Point Preconditioner Database - velocity Solver");
  p_db.set_name("Saddle Point Preconditioner Database - pressure Solver");
  db.add_nested_database(v_db);
  db.add_nested_database(p_db);

  return db;
}

ParameterDatabase SimplifiedInfo::default_direct_solver_db()
{
  ParameterDatabase db = Solver<>::default_solver_database();

  db["solver_type"].set("direct", false);

  return db;
}