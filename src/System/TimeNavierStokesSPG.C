#ifdef __2D__
#include "TimeNavierStokesSPG.h"
#include "Assemble2D.h"
#include "Database.h"
#include "FEDatabase.h"
#include "LocalAssembling.h"
#include "Matrix2D.h"
#include "MooNMD_Io.h"
#include "NSE_SPG_ErrorEstimator.h"
#include "SquareMatrix2D.h"
#include <cmath>

TimeNavierStokesSPG::TimeNavierStokesSPG(const TDomain& domain,
  const ParameterDatabase& param_db, const Example_NSE2D& example)
 : ns(domain, param_db, example),
   M(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
   M_galerkin(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
   S(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
   ud(ns.get_solution()), ud_old(ns.get_solution()), u_saved(ns.get_solution()),
   ud_saved(ns.get_solution()), u_pred(ns.get_solution()), rhs(ns.get_rhs()),
   solver(param_db), db(param_db), t(0.), last_error(-1.), n_picard(0)
{
  n_u = ns.get_velocity_space()->get_n_dof();
  n_p = ns.get_pressure_space()->get_n_dof();
  n_active_u = ns.get_velocity_space()->get_n_active();
  ud.reset();
  ud_old.reset();
  // plain velocity mass matrix (galerkin) for the L2 norm
  ParameterDatabase db_g(db);
  db_g["space_discretization_type"] = "galerkin";
  assemble_mass(M_galerkin, db_g);
  nonlin_tol = db["nonlinloop_epsilon"];
  maxit = db["nonlinloop_maxit"];
  ud_fct.reset(new TFEVectFunct2D(ns.get_velocity_space(), "dt_u", ud.get_entries(), 2));
  ns.add_to_output(ud_fct.get());
  // initial condition and (Ud^0, P^0)
  t = db["time_start"];
  TDatabase::TimeDB->CURRENTTIME = t;
  ns.get_velocity().GetComponent(0)->Interpolate(example.get_exact(0));
  ns.get_velocity().GetComponent(1)->Interpolate(example.get_exact(1));
  initial_time_derivative(t);
}

void TimeNavierStokesSPG::assemble_mass(BlockFEMatrix& Mat, const ParameterDatabase& la_db)
{
  const TFEVectFunct2D& u = ns.get_velocity();
  auto u1 = u.GetComponent(0);
  auto u2 = u.GetComponent(1);
  std::vector<const TFEFunction2D*> fcts = {u1.get(), u2.get()};
  LocalAssembling<2> la(la_db, LocalAssembling_type::TimeNavierStokesMass,
                        fcts, ns.get_example().get_coeffs());
  // block layout of NSTYPE 14 (row-major): A11 A12 B1T / A21 A22 B2T / B1 B2 C
  auto blocks = Mat.get_blocks_uniquely();
  if(blocks.size() != 9)
    ErrThrow("unexpected number of blocks: ", blocks.size());
  for(auto& b : blocks)
    b->reset();
  TSquareMatrix2D* sq[5] = {
    reinterpret_cast<TSquareMatrix2D*>(blocks[0].get()),
    reinterpret_cast<TSquareMatrix2D*>(blocks[1].get()),
    reinterpret_cast<TSquareMatrix2D*>(blocks[3].get()),
    reinterpret_cast<TSquareMatrix2D*>(blocks[4].get()),
    reinterpret_cast<TSquareMatrix2D*>(blocks[8].get())};
  TMatrix2D* re[4] = {
    reinterpret_cast<TMatrix2D*>(blocks[6].get()),
    reinterpret_cast<TMatrix2D*>(blocks[7].get()),
    reinterpret_cast<TMatrix2D*>(blocks[2].get()),
    reinterpret_cast<TMatrix2D*>(blocks[5].get())};
  const TFESpace2D* spaces[2] = {ns.get_velocity_space().get(),
                                 ns.get_pressure_space().get()};
  BoundCondFunct2D* bc[3] = {ns.get_velocity_space()->get_boundary_condition(),
                             ns.get_velocity_space()->get_boundary_condition(),
                             ns.get_pressure_space()->get_boundary_condition()};
  BoundValueFunct2D* bd[3] = {ns.get_example().get_bd(0), ns.get_example().get_bd(1),
                              ns.get_example().get_bd(2)};
  // some LocalAssembling setups declare right-hand sides which are not
  // needed here; provide scratch vectors for them
  int n_rhs = la.get_n_rhs();
  BlockVector scratch(ns.get_rhs());
  scratch.reset();
  double* rhs_array[3] = {scratch.block(0), scratch.block(1), scratch.block(2)};
  const TFESpace2D* rhs_spaces[3] = {spaces[0], spaces[0], spaces[1]};
  Assemble2D(2, spaces, 5, sq, 4, re, n_rhs, rhs_array, rhs_spaces,
             bc, bd, la, false);
}

double TimeNavierStokesSPG::l2_norm_velocity(const BlockVector& v)
{
  BlockVector Mv(v);
  Mv.reset();
  M_galerkin.apply(v, Mv);
  double s = 0.;
  for(size_t i = 0; i < 2*n_u; ++i)
    s += Mv.get_entries()[i]*v.get_entries()[i];
  return std::sqrt(std::abs(s));
}

// zero the Dirichlet entries of the velocity blocks of a vector (the
// boundary data of the examples do not depend on time)
void TimeNavierStokesSPG::zero_velocity_dirichlet(BlockVector& v)
{
  for(int c = 0; c < 2; ++c)
    for(size_t i = n_active_u; i < n_u; ++i)
      v.get_entries()[c*n_u + i] = 0.;
}

// (Ud^0, P^0) from the semidiscrete problem at t = 0 with U = U^0
void TimeNavierStokesSPG::initial_time_derivative(double time)
{
  TDatabase::TimeDB->CURRENTTIME = time;
  ns.assemble_linear_terms();        // A^{U0}, L (b = U0)
  assemble_mass(M, db);              // M^{U0}
  // S: velocity blocks and PSPG rows from M, pressure columns and the C
  // block from A
  S = M;
  auto sb = S.get_blocks_uniquely();
  auto ab = ns.get_matrix().get_blocks_uniquely();
  for(size_t i : {2, 5, 8}) // B1T, B2T, C
    *sb[i] = *ab[i];
  // right-hand side: L - A (U0, 0)
  BlockVector x(ns.get_solution());
  for(size_t i = 0; i < n_p; ++i)
    x.get_entries()[2*n_u + i] = 0.;
  rhs = ns.get_rhs();
  ns.get_matrix().apply_scaled_add(x, rhs, -1.);
  zero_velocity_dirichlet(rhs);
  BlockVector y(ns.get_solution());
  y.reset();
  solver.solve(S, rhs, y);
  for(size_t i = 0; i < 2*n_u; ++i)
    ud.get_entries()[i] = y.get_entries()[i];
  zero_velocity_dirichlet(ud);
  for(size_t i = 0; i < n_p; ++i)
    ns.get_solution().get_entries()[2*n_u + i] = y.get_entries()[2*n_u + i];
  project_pressure();
  ud_old = ud;
}

void TimeNavierStokesSPG::project_pressure()
{
  if(ns.get_matrix().pressure_projection_enabled())
    ns.get_pressure().project_into_L20();
}


void TimeNavierStokesSPG::save_state()
{
  u_saved = ns.get_solution();
  ud_saved = ud;
}

void TimeNavierStokesSPG::restore_state()
{
  ns.get_solution() = u_saved;
  ud = ud_saved;
}

void TimeNavierStokesSPG::predict(double k, double r)
{
  u_pred = ns.get_solution();
  for(size_t i = 0; i < 2*n_u; ++i)
    u_pred.get_entries()[i] += 0.5*k*((2. + r)*ud.get_entries()[i]
                                      - r*ud_old.get_entries()[i]);
}

void TimeNavierStokesSPG::tr_step(double time, double k)
{
  BlockVector& sol = ns.get_solution();
  // w = 2/k U^n + Ud^n (velocity part only)
  BlockVector w(sol);
  for(size_t i = 0; i < 2*n_u; ++i)
    w.get_entries()[i] = 2./k*u_saved.get_entries()[i] + ud_saved.get_entries()[i];
  for(size_t i = 0; i < n_p; ++i)
    w.get_entries()[2*n_u + i] = 0.;
  TDatabase::TimeDB->CURRENTTIME = time + k;
  // Picard iteration, starting from the predictor
  for(size_t i = 0; i < 2*n_u; ++i)
    sol.get_entries()[i] = u_pred.get_entries()[i];
  n_picard = 0;
  for(unsigned int it = 0; it <= maxit; ++it)
  {
    ns.assemble_linear_terms();   // A^{U}, L with b = current iterate
    assemble_mass(M, db);
    S = ns.get_matrix();
    S.add_blockfe_matrix(M, 2./k);
    rhs = ns.get_rhs();
    M.apply_scaled_add(w, rhs, 1.);
    rhs.copy_nonactive(ns.get_rhs());
    BlockVector defect(rhs);
    S.apply_scaled_add(sol, defect, -1.);
    zero_velocity_dirichlet(defect);
    if(ns.get_matrix().pressure_projection_enabled())
    {
      TFEFunction2D d_fct(ns.get_pressure_space(), "pd", &defect.get_entries()[2*n_u]);
      d_fct.project_into_L20();
    }
    double res = defect.norm();
    Output::print<3>("  Picard ", it, ": residual = ", res);
    if(res < nonlin_tol || it == maxit)
    {
      if(it == maxit && res >= nonlin_tol)
        Output::warn("TimeNavierStokesSPG", "Picard iteration did not converge, "
                     "residual = ", res);
      break;
    }
    sol.copy_nonactive(rhs);
    solver.solve(S, rhs, sol);
    project_pressure();
    n_picard++;
  }
  // Ud^{n+1} = 2/k (U^{n+1} - U^n) - Ud^n
  for(size_t i = 0; i < 2*n_u; ++i)
    ud.get_entries()[i] = 2./k*(sol.get_entries()[i] - u_saved.get_entries()[i])
                          - ud_saved.get_entries()[i];
  zero_velocity_dirichlet(ud);
  t = time + k;
}

double TimeNavierStokesSPG::predictor_difference_norm()
{
  BlockVector e(ns.get_solution());
  for(size_t i = 0; i < 2*n_u; ++i)
    e.get_entries()[i] -= u_pred.get_entries()[i];
  return l2_norm_velocity(e);
}

void TimeNavierStokesSPG::average_with_saved()
{
  BlockVector& sol = ns.get_solution();
  for(size_t i = 0; i < 2*n_u; ++i)
  {
    sol.get_entries()[i] = 0.5*(sol.get_entries()[i] + u_saved.get_entries()[i]);
    ud.get_entries()[i] = 0.5*(ud.get_entries()[i] + ud_saved.get_entries()[i]);
  }
}

void TimeNavierStokesSPG::shift_derivative()
{
  ud_old = ud_saved;
}

void TimeNavierStokesSPG::estimate(ErrorEstimator<2>& estimator)
{
  auto* est = dynamic_cast<NSE_SPG_ErrorEstimator*>(&estimator);
  if(est == nullptr)
    ErrThrow("TimeNavierStokesSPG::estimate needs an NSE_SPG_ErrorEstimator");
  TDatabase::TimeDB->CURRENTTIME = t;
  TFEVectFunct2D ud_fct(ns.get_velocity_space(), "ud", ud.get_entries(), 2);
  est->set_transient(&ud_fct);
  est->estimate(ns.get_example(), ns.get_velocity(), ns.get_pressure());
  est->set_transient(nullptr);
  last_error = est->get_error_spg();
}

const TCollection* TimeNavierStokesSPG::get_collection() const
{
  return ns.get_velocity_space()->GetCollection();
}

std::vector<SolutionTransfer::NodeValueMap> TimeNavierStokesSPG::record_state() const
{
  std::vector<SolutionTransfer::NodeValueMap> maps;
  auto add = [&](const TFEVectFunct2D& f)
  { auto m = SolutionTransfer::record(f); maps.insert(maps.end(), m.begin(), m.end()); };
  add(ns.get_velocity());
  maps.push_back(SolutionTransfer::record(ns.get_pressure()));
  TFEVectFunct2D ud_fct(ns.get_velocity_space(), "ud", const_cast<double*>(ud.get_entries()), 2);
  TFEVectFunct2D udo_fct(ns.get_velocity_space(), "udo", const_cast<double*>(ud_old.get_entries()), 2);
  add(ud_fct);
  add(udo_fct);
  return maps; // u1 u2 p d1 d2 do1 do2
}

void TimeNavierStokesSPG::apply_state(const std::vector<SolutionTransfer::NodeValueMap>& maps)
{
  if(maps.size() != 7)
    ErrThrow("TimeNavierStokesSPG::apply_state: seven maps expected");
  SolutionTransfer::apply(ns.get_velocity(), {maps[0], maps[1]});
  SolutionTransfer::apply(ns.get_pressure(), maps[2]);
  TFEVectFunct2D ud_fct(ns.get_velocity_space(), "ud", ud.get_entries(), 2);
  TFEVectFunct2D udo_fct(ns.get_velocity_space(), "udo", ud_old.get_entries(), 2);
  SolutionTransfer::apply(ud_fct, {maps[3], maps[4]});
  SolutionTransfer::apply(udo_fct, {maps[5], maps[6]});
  u_saved = ns.get_solution();
  ud_saved = ud;
  u_pred = ns.get_solution();
}

void TimeNavierStokesSPG::output(int index)
{
  TDatabase::TimeDB->CURRENTTIME = t;
  ns.output(index);
}

void TimeNavierStokesSPG::copy_time_state(const AdaptiveTimeSystem& old)
{
  const auto* o = dynamic_cast<const TimeNavierStokesSPG*>(&old);
  if(o != nullptr)
    t = o->t;
}
#endif // __2D__
