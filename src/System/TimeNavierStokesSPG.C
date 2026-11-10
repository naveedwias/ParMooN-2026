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
#include "AuxParam2D.h"
#include "MainUtilities.h"
#include "QuadratureFormulaDatabase.h"
#include "BaseFunctions.h"
#include "BaseCell.h"
#include "Vertex.h"
#include "FiniteElement.h"
#include "NSE_local_assembling_routines.h"
#include <algorithm>
#include <cmath>

TimeNavierStokesSPG::TimeNavierStokesSPG(const TDomain& domain,
  const ParameterDatabase& param_db, const Example_NSE2D& example)
 : ns(domain, param_db, example),
   M(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
   S(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
   ud(ns.get_solution()), ud_old(ns.get_solution()), u_saved(ns.get_solution()),
   ud_saved(ns.get_solution()), u_pred(ns.get_solution()), rhs(ns.get_rhs()),
   solver(param_db), db(param_db), t(0.), last_error(-1.), n_picard(0),
   last_estimate_t(-1.), dt_alpha(0.), min_delta(0.),
   est_sol(ns.get_solution()), est_ud(ns.get_solution()), est_t(0.),
   has_est(false)
{
  n_u = ns.get_velocity_space()->get_n_dof();
  n_p = ns.get_pressure_space()->get_n_dof();
  n_active_u = ns.get_velocity_space()->get_n_active();
  ud.reset();
  ud_old.reset();
  nonlin_tol = db["nonlinloop_epsilon"];
  maxit = db["nonlinloop_maxit"];
  ud_fct.reset(new TFEVectFunct2D(ns.get_velocity_space(), "dt_u", ud.get_entries(), 2));
  ns.add_to_output(ud_fct.get());
  t = db["time_start"];
  // lower bound of the step size: alpha min_K delta_K
  dt_alpha = db.contains("tnse_dt_alpha") ? (double) db["tnse_dt_alpha"] : 0.;
  {
    const TCollection* coll = ns.get_velocity_space()->GetCollection();
    const double nu = 1./TDatabase::ParamDB->RE_NR;
    const bool equal_order = (TDatabase::ParamDB->VELOCITY_SPACE
                              == TDatabase::ParamDB->PRESSURE_SPACE);
    const double delta0 = db["spg_delta0"], mu0 = db["spg_mu0"];
    min_delta = 1e300;
    for(int i = 0; i < coll->GetN_Cells(); ++i)
    {
      double dK, mK;
      compute_SPG_parameters(coll->GetCell(i)->GetDiameter(), nu, delta0, mu0,
                             equal_order, dK, mK);
      min_delta = std::min(min_delta, dK);
    }
  }
}

// L2 projection of u(time_start) onto the velocity space, Dirichlet dofs from
// the boundary data (the lifting of the manuscript): M x = (u_0, phi_i) on the
// free dofs, solved by Jacobi-preconditioned CG
void TimeNavierStokesSPG::l2_projection_of_initial_velocity()
{
  ParameterDatabase db0(db);
  db0["spg_delta0"] = 0.;                     // plain mass matrix in A11, A22
  BlockFEMatrix Mg(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(),
                                               ns.get_pressure_space()));
  assemble_mass(Mg, db0);
  auto block = Mg.get_blocks_uniquely()[0];
  const int* rowptr = block->get_row_ptr();
  const int* kcol = block->get_vector_columns();
  const double* ent = block->GetEntries();
  auto space = ns.get_velocity_space();
  const TCollection* coll = space->GetCollection();
  const Example_NSE2D& example = ns.get_example();
  int n_qp;
  const double *w, *xi, *eta;
  int degree = 2*space->get_fe(0).GetBaseFunct()->GetPolynomialDegree() + 6;
  QuadratureFormulaDatabase::qf_from_degree(degree, BFRefElements::BFUnitTriangle)
    ->GetFormulaData(n_qp, w, xi, eta);
  double wsum = 0.;
  for(int q = 0; q < n_qp; ++q)
    wsum += w[q];
  BlockVector& sol = ns.get_solution();
  for(int c = 0; c < 2; ++c)
  {
    std::vector<double> b(n_u, 0.), phi(MaxN_BaseFunctions2D);
    for(int i = 0; i < coll->GetN_Cells(); ++i)
    {
      const TBaseCell* cell = coll->GetCell(i);
      double v[6];
      for(int k = 0; k < 3; ++k)
      {
        v[2*k] = cell->GetVertex(k)->GetX();
        v[2*k+1] = cell->GetVertex(k)->GetY();
      }
      double area = 0.5*std::abs((v[2]-v[0])*(v[5]-v[1]) - (v[4]-v[0])*(v[3]-v[1]));
      const FiniteElement& fe = space->get_fe(i);
      const BaseFunctions* bf = fe.GetBaseFunct();
      int n_loc = fe.GetN_DOF();
      const int* DOF = space->GetGlobalDOF(i);
      for(int q = 0; q < n_qp; ++q)
      {
        double x = v[0] + xi[q]*(v[2]-v[0]) + eta[q]*(v[4]-v[0]);
        double y = v[1] + xi[q]*(v[3]-v[1]) + eta[q]*(v[5]-v[1]);
        double ue[4];
        example.get_exact(c)(x, y, ue);
        bf->GetDerivatives(MultiIndex2D::D00, xi[q], eta[q], phi.data());
        double wq = w[q]*area/wsum;
        for(int k = 0; k < n_loc; ++k)
          b[DOF[k]] += wq*ue[0]*phi[k];
      }
    }
    // unknowns: free dofs; Dirichlet entries (boundary data) already in x
    double* x = sol.get_entries() + c*n_u;
    const size_t na = n_active_u;
    auto matvec = [&](const std::vector<double>& p, std::vector<double>& q)
    {
      for(size_t r = 0; r < na; ++r)
      {
        double s_ = 0.;
        for(int j = rowptr[r]; j < rowptr[r+1]; ++j)
          if((size_t) kcol[j] < na)
            s_ += ent[j]*p[kcol[j]];
        q[r] = s_;
      }
    };
    std::vector<double> diag(na, 1.), rhs(na), xa(na);
    for(size_t r = 0; r < na; ++r)
    {
      double s_ = b[r];
      for(int j = rowptr[r]; j < rowptr[r+1]; ++j)
      {
        if((size_t) kcol[j] >= na)
          s_ -= ent[j]*x[kcol[j]];
        else if((size_t) kcol[j] == r)
          diag[r] = ent[j];
      }
      rhs[r] = s_;
      xa[r] = x[r];
    }
    std::vector<double> rr(na), z(na), p(na), Ap(na);
    matvec(xa, Ap);
    double rz = 0., r0 = 0.;
    for(size_t r = 0; r < na; ++r)
    {
      rr[r] = rhs[r] - Ap[r];
      z[r] = rr[r]/diag[r];
      p[r] = z[r];
      rz += rr[r]*z[r];
      r0 += rhs[r]*rhs[r];
    }
    r0 = std::sqrt(r0);
    int it = 0;
    for(; it < 5000; ++it)
    {
      double rn = 0.;
      for(size_t r = 0; r < na; ++r)
        rn += rr[r]*rr[r];
      if(std::sqrt(rn) <= 1e-14*std::max(r0, 1e-300))
        break;
      matvec(p, Ap);
      double pAp = 0.;
      for(size_t r = 0; r < na; ++r)
        pAp += p[r]*Ap[r];
      double a = rz/pAp;
      double rz_new = 0.;
      for(size_t r = 0; r < na; ++r)
      {
        xa[r] += a*p[r];
        rr[r] -= a*Ap[r];
        z[r] = rr[r]/diag[r];
        rz_new += rr[r]*z[r];
      }
      for(size_t r = 0; r < na; ++r)
        p[r] = z[r] + rz_new/rz*p[r];
      rz = rz_new;
    }
    for(size_t r = 0; r < na; ++r)
      x[r] = xa[r];
    Output::print<2>("L2 projection of u_0, component ", c, ": ", it,
                     " CG iterations");
  }
}

void TimeNavierStokesSPG::initialize()

{
  // initial condition and (Ud^0, P^0)
  t = db["time_start"];
  TDatabase::TimeDB->CURRENTTIME = t;
  const Example_NSE2D& example = ns.get_example();
  BlockVector& sol = ns.get_solution();
  ns.get_velocity().GetComponent(0)->Interpolate(example.get_exact(0));
  ns.get_velocity().GetComponent(1)->Interpolate(example.get_exact(1));
  set_dirichlet_values(t, sol);
  std::string init = db.contains("tnse_initial_value")
                     ? db["tnse_initial_value"].value_as_string() : "l2_projection";
  if(init == "l2_projection")
  {
    // the choice of the manuscript (Section 2.4): U^0 = L2 projection of u_0
    l2_projection_of_initial_velocity();
  }
  else if(init == "consistent_projection")
  {
    // The interpolant of u(t0) is not consistent with the discrete
    // constraint (divergence row with PSPG); its component in the stiff modes
    // (stiffness ~ 1/(delta_0 h^2)) is carried on by the trapezoidal rule with
    // factor ~ -1 and pollutes the pressure, which is the multiplier of the
    // constraint (L2 pressure error O(1) on fine meshes with fixed steps).
    // Projection: the stabilized equations at t0 with the exact time
    // derivative, linearized at b = I_h u(t0):
    //   A^b (U0, P0) = L(t0) - M^b I_h d_t u(t0),   U0 = g(t0) on the boundary.
    const double eps = 1e-5*std::max(1., std::abs(t));
    BlockVector up(sol), um(sol), ut(sol);
    for(int sgn : {1, -1})
    {
      TDatabase::TimeDB->CURRENTTIME = t + sgn*eps;
      BlockVector& v = (sgn > 0) ? up : um;
      TFEVectFunct2D vf(ns.get_velocity_space(), "v", v.get_entries(), 2);
      vf.GetComponent(0)->Interpolate(example.get_exact(0));
      vf.GetComponent(1)->Interpolate(example.get_exact(1));
    }
    TDatabase::TimeDB->CURRENTTIME = t;
    ut.reset();
    for(size_t i = 0; i < 2*n_u; ++i)
      ut.get_entries()[i] = (up.get_entries()[i] - um.get_entries()[i])/(2.*eps);
    ns.assemble_linear_terms();    // A^b, L(t0), b = I_h u(t0)
    assemble_mass(M, db);          // M^b
    rhs = ns.get_rhs();
    M.apply_scaled_add(ut, rhs, -1.);
    rhs.copy_nonactive(ns.get_rhs());
    sol.copy_nonactive(rhs);
    solver.solve(ns.get_matrix(), rhs, sol);
    project_pressure();
  }
  else if(init != "interpolation")
    ErrThrow("tnse_initial_value must be l2_projection, consistent_projection "
             "or interpolation");
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

// L2 norm of the velocity part by quadrature. (An assembled mass matrix is
// not usable here: its Dirichlet rows are identity rows, which weight the
// boundary values with 1 instead of O(h^2), and the Galerkin mass routine
// of ParMooN writes the second component into a different block than the
// SUPG/PSPG layout used for M.)
double TimeNavierStokesSPG::l2_norm_velocity(const BlockVector& v)
{
  auto space = ns.get_velocity_space();
  double s = 0.;
  for(int c = 0; c < 2; ++c)
  {
    TFEFunction2D f(space, "e", const_cast<double*>(v.get_entries()) + c*n_u);
    double n = f.get_L2_norm();
    s += n*n;
  }
  return std::sqrt(s);
}

// zero the Dirichlet entries of the velocity blocks of a vector
void TimeNavierStokesSPG::zero_velocity_dirichlet(BlockVector& v)
{
  for(int c = 0; c < 2; ++c)
    for(size_t i = n_active_u; i < n_u; ++i)
      v.get_entries()[c*n_u + i] = 0.;
}

void TimeNavierStokesSPG::set_dirichlet_values(double time, BlockVector& v)
{
  const double t_save = TDatabase::TimeDB->CURRENTTIME;
  TDatabase::TimeDB->CURRENTTIME = time;
  auto space = ns.get_velocity_space();
  for(int c = 0; c < 2; ++c)
  {
    TFEFunction2D comp(space, "bd", v.get_entries() + c*n_u);
    comp.SetDirichletBC(space->get_boundary_condition(),
                        ns.get_example().get_bd(c));
  }
  TDatabase::TimeDB->CURRENTTIME = t_save;
}

void TimeNavierStokesSPG::set_dirichlet_time_derivative(double time,
                                                        BlockVector& v)
{
  const double eps = 1e-5*std::max(1., std::abs(time));
  BlockVector gp(v), gm(v);
  set_dirichlet_values(time + eps, gp);
  set_dirichlet_values(time - eps, gm);
  for(int c = 0; c < 2; ++c)
    for(size_t i = n_active_u; i < n_u; ++i)
      v.get_entries()[c*n_u + i] = (gp.get_entries()[c*n_u + i]
                                    - gm.get_entries()[c*n_u + i])/(2.*eps);
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
  // Dirichlet rows: the time derivative of the boundary data (zero for
  // time-independent data); the interior rows couple to it through M
  set_dirichlet_time_derivative(time, rhs);
  BlockVector y(ns.get_solution());
  y.reset();
  y.copy_nonactive(rhs);
  solver.solve(S, rhs, y);
  for(size_t i = 0; i < 2*n_u; ++i)
    ud.get_entries()[i] = y.get_entries()[i];
  set_dirichlet_time_derivative(time, ud);
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
  // Ud^{n+1} = 2/k (U^{n+1} - U^n) - Ud^n on the free dofs. On the Dirichlet
  // dofs the derivative of the data is used: setting it to zero is only
  // correct for time-independent data, and the trapezoidal recurrence would
  // carry its O(k^2) error on with alternating sign into the interior rows of
  // the following steps (5 times larger global error on the verification
  // example 28 at k = 0.025)
  for(size_t i = 0; i < 2*n_u; ++i)
    ud.get_entries()[i] = 2./k*(sol.get_entries()[i] - u_saved.get_entries()[i])
                          - ud_saved.get_entries()[i];
  t = time + k;
  set_dirichlet_time_derivative(t, ud);
}

double TimeNavierStokesSPG::predictor_difference_norm()
{
  BlockVector e(ns.get_solution());
  for(size_t i = 0; i < 2*n_u; ++i)
    e.get_entries()[i] -= u_pred.get_entries()[i];
  return l2_norm_velocity(e);
}

void TimeNavierStokesSPG::average_with_saved(double t_mid)
{
  est_sol = ns.get_solution();
  est_ud = ud;
  est_t = t;
  has_est = true;
  // velocity, pressure (the pressure enters the estimator and the output) and
  // the time derivative are averaged; the averaged state belongs to t_mid
  BlockVector& sol = ns.get_solution();
  for(size_t i = 0; i < sol.length(); ++i)
    sol.get_entries()[i] = 0.5*(sol.get_entries()[i] + u_saved.get_entries()[i]);
  for(size_t i = 0; i < 2*n_u; ++i)
    ud.get_entries()[i] = 0.5*(ud.get_entries()[i] + ud_saved.get_entries()[i]);
  t = t_mid;
  // the mean of the boundary values at t^n and t^{n+1} is not the boundary
  // value at t_mid (O(k^2) off for time-dependent data); the next step would
  // force the difference into the trapezoidal derivative as O(k^2)/k and the
  // recurrence would carry it on with alternating sign
  set_dirichlet_values(t, sol);
  set_dirichlet_time_derivative(t, ud);
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
  // after an averaging step: estimate with the non-averaged pair at t^{n+1}
  BlockVector keep_sol(ns.get_solution()), keep_ud(ud);
  const double keep_t = t;
  if(has_est)
  {
    ns.get_solution() = est_sol;
    ud = est_ud;
    t = est_t;
  }
  TDatabase::TimeDB->CURRENTTIME = t;
  TFEVectFunct2D ud_fct(ns.get_velocity_space(), "ud", ud.get_entries(), 2);
  est->set_transient(&ud_fct);
  est->estimate(ns.get_example(), ns.get_velocity(), ns.get_pressure());
  est->set_transient(nullptr);
  last_error = est->get_error_spg();
  last_estimate_t = t;
  auto ep = est->get_error_parts();
  last_contributions = {
    {"eta_res", est->get_eta_res()}, {"eta_div", est->get_eta_div()},
    {"eta_F", est->get_eta_F()}, {"eta_delta", est->get_eta_delta()},
    {"eta_mu", est->get_eta_mu()},
    // parts of the error, square roots: nu^1/2 |e_u|_1, (sum mu_K ||div e_u||^2)^1/2,
    // (sum delta_K ||(b.grad) e_u + grad e_p||^2)^1/2
    {"err_grad", std::sqrt(std::max(ep[0], 0.))},
    {"err_div", std::sqrt(std::max(ep[2], 0.))},
    {"err_supg", std::sqrt(std::max(ep[3], 0.))}};
  cached_columns = compute_error_columns();
  if(has_est)
  {
    ns.get_solution() = keep_sol;
    ud = keep_ud;
    t = keep_t;
    TDatabase::TimeDB->CURRENTTIME = t;
    has_est = false;
  }
}

const TCollection* TimeNavierStokesSPG::get_collection() const
{
  return ns.get_velocity_space()->GetCollection();
}

std::vector<SolutionTransfer::Snapshot> TimeNavierStokesSPG::record_state() const
{
  std::vector<SolutionTransfer::Snapshot> maps;
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

void TimeNavierStokesSPG::apply_state(const std::vector<SolutionTransfer::Snapshot>& maps)
{
  if(maps.size() != 7)
    ErrThrow("TimeNavierStokesSPG::apply_state: seven maps expected");
  SolutionTransfer::apply(ns.get_velocity(), {maps[0], maps[1]});
  SolutionTransfer::apply(ns.get_pressure(), maps[2]);
  TFEVectFunct2D ud_fct(ns.get_velocity_space(), "ud", ud.get_entries(), 2);
  TFEVectFunct2D udo_fct(ns.get_velocity_space(), "udo", ud_old.get_entries(), 2);
  SolutionTransfer::apply(ud_fct, {maps[3], maps[4]});
  SolutionTransfer::apply(udo_fct, {maps[5], maps[6]});
  // the Dirichlet values of the new mesh come from the boundary data, not
  // from the interpolation of the old mesh (see TimeConvectionDiffusionTR::
  // apply_state for why this matters)
  TDatabase::TimeDB->CURRENTTIME = t;
  set_dirichlet_values(t, ns.get_solution());
  set_dirichlet_time_derivative(t, ud);
  u_saved = ns.get_solution();
  ud_saved = ud;
  u_pred = ns.get_solution();
}

void TimeNavierStokesSPG::output(int index)
{
  TDatabase::TimeDB->CURRENTTIME = t;
  ns.output(index);
}

std::vector<std::pair<std::string, double>> TimeNavierStokesSPG::error_columns()
{
  if(!cached_columns.empty())
    return cached_columns;
  return compute_error_columns();
}

std::vector<std::pair<std::string, double>> TimeNavierStokesSPG::compute_error_columns()
{
  const Example_NSE2D& example = ns.get_example();
  if(example.get_exact().size() < 3)
    return {};
  TDatabase::TimeDB->CURRENTTIME = t;
  TAuxParam2D aux;
  MultiIndex2D derivs[3] = {MultiIndex2D::D00, MultiIndex2D::D10,
                            MultiIndex2D::D01};
  const TFESpace2D* vspace = ns.get_velocity_space().get();
  const TFESpace2D* pspace = ns.get_pressure_space().get();
  double l2u = 0., h1u = 0.;
  for(int c = 0; c < 2; ++c)
  {
    double e[4] = {0., 0., 0., 0.};
    ns.get_velocity().GetComponent(c)->GetErrors(example.get_exact(c), 3, derivs,
      2, L2H1Errors, example.get_coeffs(), &aux, 1, &vspace, e);
    l2u += e[0]*e[0];
    h1u += e[1]*e[1];
  }
  double ep[4] = {0., 0., 0., 0.};
  ns.get_pressure().GetErrors(example.get_exact(2), 3, derivs, 2, L2H1Errors,
                              nullptr, &aux, 1, &pspace, ep);
  double div = ns.get_velocity().GetL2NormDivergenceError(example.get_exact(0),
                                                          example.get_exact(1));
  std::vector<std::pair<std::string, double>> cols = {
    {"L2(u)", std::sqrt(l2u)}, {"H1(u)", std::sqrt(h1u)},
    {"L2(div_u)", div}, {"L2(p)", ep[0]}};
  if(last_contributions.empty()) // before the first estimate: names only
    for(const char* n : {"eta_res", "eta_div", "eta_F", "eta_delta", "eta_mu",
                         "err_grad", "err_div", "err_supg"})
      cols.push_back({n, 0.});
  else
    cols.insert(cols.end(), last_contributions.begin(), last_contributions.end());
  return cols;
}

void TimeNavierStokesSPG::copy_time_state(const AdaptiveTimeSystem& old)
{
  const auto* o = dynamic_cast<const TimeNavierStokesSPG*>(&old);
  if(o != nullptr)
    t = o->t;
}
#endif // __2D__
