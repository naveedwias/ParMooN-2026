// =============================================================================
// Adaptive SUPG/PSPG/grad-div method for the time-dependent incompressible
// Navier-Stokes equations: trapezoidal rule with variable step (Gresho,
// Griffiths, Silvester 2008) carrying the discrete time derivative, AB2 error
// estimation with periodic averaging, Picard iteration for the nonlinear
// problem, the residual-based estimator with the discrete time derivative in
// the cell residual, and refinement/coarsening with the solution transfer.
// (Transient extension of Afzal, Ahmed, John, CMAME 223 (2026) 467-486 in the
// manner of de Frutos, Garcia-Archilla, John, Novo, CMAME 273 (2014) 219-237.)
//
// Semidiscrete problem: for all (v,q)
//   (d_t U, v) + sum_K delta_K (d_t U, (U.grad)v + grad q)_K + A^U((U,P),(v,q)) = L^U(v,q)
// TR step (eq. (TR)): with M^b the SUPG/PSPG mass matrix and A^b the steady
// SUPG/PSPG/grad-div matrix for the convection field b,
//   (2/dt M^{U^{n+1}} + A^{U^{n+1}}) (U^{n+1},P^{n+1}) = L^{U^{n+1}} + M^{U^{n+1}} (2/dt U^n + Ud^n),
//   Ud^{n+1} = 2/dt (U^{n+1} - U^n) - Ud^n.
// The initial (Ud^0, P^0) solve the semidiscrete problem at t = 0 (linear).
//
// Required: space_discretization_type: supg_pspg_graddiv, NSTYPE: 14,
// problem_type: 5, example 23 (smooth) or 24 (layers).
// Modes: tnse_time_adaptive (TR-AB2 step control, else fixed step), tnse_adaptive
// (mesh adaptation with tnse_tol1, tnse_tol2, coarsening for all Lagrange
// pairs through a coordinate-keyed node-value map).
// =============================================================================
#include <Domain.h>
#include <Database.h>
#include "NavierStokes.h"
#include <Example_NSE2D.h>
#include <Example_TimeNSE2D.h>
#include <Chrono.h>
#include <ParameterDatabase.h>
#include "ParMooN.h"
#include "NSE_SPG_ErrorEstimator.h"
#include "RefinementStrategy.h"
#include "LocalAssembling.h"
#include "Assemble2D.h"
#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "Solver.h"
#include "FEDatabase.h"
#include "NodalFunctional.h"
#include "Collection.h"
#include "BaseCell.h"
#include "Vertex.h"
#include "SquareMatrix2D.h"
#include "Matrix2D.h"
#include "FESpace2D.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

// =============================================================================
// solution transfer between meshes for Lagrange elements
// =============================================================================
typedef std::pair<long long, long long> PointKey;
inline PointKey point_key(double x, double y)
{ return {std::llround(x*1e9), std::llround(y*1e9)}; }
typedef std::map<PointKey, double> NodeValueMap;

// point in triangle by barycentric coordinates (tolerance in reference units)
bool point_in_triangle(const TBaseCell* c, double x, double y, double tol = 1e-9)
{
  double x0 = c->GetVertex(0)->GetX(), y0 = c->GetVertex(0)->GetY();
  double x1 = c->GetVertex(1)->GetX(), y1 = c->GetVertex(1)->GetY();
  double x2 = c->GetVertex(2)->GetX(), y2 = c->GetVertex(2)->GetY();
  double det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
  double l1 = ((x-x0)*(y2-y0) - (x2-x0)*(y-y0))/det;
  double l2 = ((x1-x0)*(y-y0) - (x-x0)*(y1-y0))/det;
  return l1 >= -tol && l2 >= -tol && l1 + l2 <= 1. + tol;
}

// nodal points (physical coordinates) of the finite element of the given
// space on an arbitrary triangle (a,b,c): affine map of the reference points
void nodal_points_on_triangle(const std::shared_ptr<const TFESpace2D>& space,
                              int cell_no, const double* xa, const double* xb,
                              const double* xc, std::vector<std::pair<double,double>>& pts)
{
  const FiniteElement& element = space->get_fe(cell_no);
  const NodalFunctional* nf = element.GetNodalFunctional();
  int n_points;
  const double *xi, *eta;
  nf->GetPointsForAll(n_points, xi, eta);
  for(int j = 0; j < n_points; ++j)
  {
    double x = xa[0] + xi[j]*(xb[0]-xa[0]) + eta[j]*(xc[0]-xa[0]);
    double y = xa[1] + xi[j]*(xb[1]-xa[1]) + eta[j]*(xc[1]-xa[1]);
    pts.emplace_back(x, y);
  }
}

// candidate nodal points of all possible children (regular refinement and the
// three bisections) of the triangle with vertices v0, v1, v2
void candidate_points(const std::shared_ptr<const TFESpace2D>& space,
                      int cell_no, const double v[3][2],
                      std::vector<std::pair<double,double>>& pts)
{
  double m[3][2]; // midpoints m01, m12, m20
  for(int k = 0; k < 2; ++k)
  {
    m[0][k] = 0.5*(v[0][k]+v[1][k]);
    m[1][k] = 0.5*(v[1][k]+v[2][k]);
    m[2][k] = 0.5*(v[2][k]+v[0][k]);
  }
  // the triangle itself (nodes of unchanged or closure-rebuilt cells)
  nodal_points_on_triangle(space, cell_no, v[0], v[1], v[2], pts);
  // regular refinement
  nodal_points_on_triangle(space, cell_no, v[0], m[0], m[2], pts);
  nodal_points_on_triangle(space, cell_no, m[0], v[1], m[1], pts);
  nodal_points_on_triangle(space, cell_no, m[2], m[1], v[2], pts);
  nodal_points_on_triangle(space, cell_no, m[0], m[1], m[2], pts);
  // bisections of each edge
  nodal_points_on_triangle(space, cell_no, v[0], m[0], v[2], pts);
  nodal_points_on_triangle(space, cell_no, m[0], v[1], v[2], pts);
  nodal_points_on_triangle(space, cell_no, v[1], m[1], v[0], pts);
  nodal_points_on_triangle(space, cell_no, m[1], v[2], v[0], pts);
  nodal_points_on_triangle(space, cell_no, v[2], m[2], v[1], pts);
  nodal_points_on_triangle(space, cell_no, m[2], v[0], v[1], pts);
}

// descend from c to the leaf containing (x,y)
const TBaseCell* leaf_containing(const TBaseCell* c, double x, double y)
{
  while(c != nullptr && c->ExistChildren())
  {
    const TBaseCell* next = nullptr;
    for(int i = 0; i < c->GetN_Children(); ++i)
      if(point_in_triangle(c->GetChild(i), x, y))
      { next = c->GetChild(i); break; }
    c = next;
  }
  return c;
}

// Values of the function u (on the current mesh) at all nodal points that the
// next mesh (refinement, coarsening, conforming closure) can have, keyed by
// coordinates. Must be called before the mesh is changed.
NodeValueMap node_values(const TFEFunction2D& u)
{
  NodeValueMap m;
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  int n_cells = coll->GetN_Cells();
  std::map<const TBaseCell*, int> index;
  for(int i = 0; i < n_cells; ++i)
    index[coll->GetCell(i)] = i;
  std::vector<std::pair<double,double>> pts;
  double val[3];
  for(int i = 0; i < n_cells; ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    double v[3][2];
    for(int k = 0; k < 3; ++k)
    { v[k][0] = cell->GetVertex(k)->GetX(); v[k][1] = cell->GetVertex(k)->GetY(); }
    pts.clear();
    candidate_points(space, i, v, pts);
    for(auto& p : pts)
    {
      PointKey key = point_key(p.first, p.second);
      if(m.count(key)) continue;
      u.FindValueLocal(cell, i, p.first, p.second, val);
      m[key] = val[0];
    }
    // the parent's candidate points (closure cells are rebuilt from the parent)
    const TBaseCell* parent = cell->GetParent();
    if(parent != nullptr && parent->GetChild(0) == cell)
    {
      for(int k = 0; k < 3; ++k)
      { v[k][0] = parent->GetVertex(k)->GetX(); v[k][1] = parent->GetVertex(k)->GetY(); }
      pts.clear();
      candidate_points(space, i, v, pts);
      for(auto& p : pts)
      {
        PointKey key = point_key(p.first, p.second);
        if(m.count(key)) continue;
        const TBaseCell* leaf = leaf_containing(parent, p.first, p.second);
        auto it = (leaf != nullptr) ? index.find(leaf) : index.end();
        if(it == index.end()) continue;
        u.FindValueLocal(leaf, it->second, p.first, p.second, val);
        m[key] = val[0];
      }
    }
  }
  return m;
}

// fill u on the new mesh from the map
void transfer(TFEFunction2D& u, const NodeValueMap& m)
{
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  double* values = u.GetValues();
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    double v[3][2];
    for(int k = 0; k < 3; ++k)
    { v[k][0] = cell->GetVertex(k)->GetX(); v[k][1] = cell->GetVertex(k)->GetY(); }
    std::vector<std::pair<double,double>> pts;
    nodal_points_on_triangle(space, i, v[0], v[1], v[2], pts);
    const int* DOF = space->GetGlobalDOF(i);
    for(size_t j = 0; j < pts.size(); ++j)
    {
      auto it = m.find(point_key(pts[j].first, pts[j].second));
      if(it == m.end())
        ErrThrow("transfer: no value at (", pts[j].first, ",", pts[j].second, ")");
      values[DOF[j]] = it->second;
    }
  }
}

// =============================================================================
// the transient SUPG/PSPG/grad-div system on one mesh
// =============================================================================
class TNSE_SPG
{
  public:
    NavierStokes<2> ns;              // spaces, A^b, L^b, Dirichlet data, pressure projection
    BlockFEMatrix M, M_galerkin, S;  // SUPG/PSPG mass, velocity mass, system matrix
    BlockVector ud, ud_old, u_prev, ud_prev, u_pred, rhs;
    Solver<BlockFEMatrix, BlockVector> solver;
    ParameterDatabase db;
    double t, k_old;
    size_t n_u, n_p, n_active_u;

    TNSE_SPG(const TDomain& domain, const ParameterDatabase& param_db,
             const Example_NSE2D& example)
     : ns(domain, param_db, example),
       M(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
       M_galerkin(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
       S(BlockFEMatrix::NSE2D_Type14(ns.get_velocity_space(), ns.get_pressure_space())),
       ud(ns.get_solution()), ud_old(ns.get_solution()), u_prev(ns.get_solution()),
       ud_prev(ns.get_solution()), u_pred(ns.get_solution()), rhs(ns.get_rhs()),
       solver(param_db), db(param_db), t(0.), k_old(0.)
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
    }

    void assemble_mass(BlockFEMatrix& Mat, const ParameterDatabase& la_db)
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

    double l2_norm_velocity(const BlockVector& v)
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
    void zero_velocity_dirichlet(BlockVector& v)
    {
      for(int c = 0; c < 2; ++c)
        for(size_t i = n_active_u; i < n_u; ++i)
          v.get_entries()[c*n_u + i] = 0.;
    }

    // (Ud^0, P^0) from the semidiscrete problem at t = 0 with U = U^0
    void initial_time_derivative(double time)
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

    void project_pressure()
    {
      if(ns.get_matrix().pressure_projection_enabled())
        ns.get_pressure().project_into_L20();
    }

    // one TR step of length k from t to t + k (Picard); returns the AB2 error
    // estimate. u_prev, ud_prev keep the old values for a rejection.
    double step(double k, bool first_step, double nonlin_tol, unsigned int maxit,
                int& n_picard)
    {
      BlockVector& sol = ns.get_solution();
      u_prev = sol;
      ud_prev = ud;
      double r = first_step ? 1. : k / k_old;
      // AB2 predictor (velocity)
      u_pred = sol;
      for(size_t i = 0; i < 2*n_u; ++i)
        u_pred.get_entries()[i] += 0.5*k*((2. + r)*ud.get_entries()[i]
                                          - r*ud_old.get_entries()[i]);
      // w = 2/k U^n + Ud^n (velocity part only)
      BlockVector w(sol);
      for(size_t i = 0; i < 2*n_u; ++i)
        w.get_entries()[i] = 2./k*sol.get_entries()[i] + ud.get_entries()[i];
      for(size_t i = 0; i < n_p; ++i)
        w.get_entries()[2*n_u + i] = 0.;
      TDatabase::TimeDB->CURRENTTIME = t + k;
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
        rhs = ns.get_rhs();           // L, Dirichlet values in the inactive entries
        M.apply_scaled_add(w, rhs, 1.);
        rhs.copy_nonactive(ns.get_rhs());
        // residual of the current iterate
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
            Output::warn("TNSE_SPG", "Picard iteration did not converge, "
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
        ud.get_entries()[i] = 2./k*(sol.get_entries()[i] - u_prev.get_entries()[i])
                              - ud_prev.get_entries()[i];
      zero_velocity_dirichlet(ud);
      // AB2 error estimate
      BlockVector e(sol);
      for(size_t i = 0; i < 2*n_u; ++i)
        e.get_entries()[i] -= u_pred.get_entries()[i];
      return l2_norm_velocity(e) / (3.*(1. + 1./r));
    }
};

struct TableRow
{
  double t, k, eta, error, eff;
  int cells, dofs, n_marked, n_coarsen, picard;
};

int main(int argc, char* argv[])
{
  Chrono timer;
  ParameterDatabase defaults("TNSE2D SPG adaptive defaults");
  defaults.add("tnse_time_adaptive", true, "TR-AB2 step size control");
  defaults.add("tnse_tolt", 1.e-5, "tolerance TOLT of the time integrator", 0., 1.);
  defaults.add("tnse_dt0", 1.e-4, "initial time step", 0., 1.);
  defaults.add("tnse_dt_min", 1.e-6, "minimal step (no rejection below)", 0., 1.);
  defaults.add("tnse_dt_max", 0.1, "maximal step", 0., 1.e10);
  defaults.add("tnse_averaging_steps", 10, "average every n* steps (0: off)", 0, 1000000);
  defaults.add("tnse_adaptive", false, "adaptive meshes");
  defaults.add("tnse_tol1", 1.e-2, "refine if eta_K > TOL1", 0., 1.e10);
  defaults.add("tnse_tol2", 5.e-4, "coarsen if eta_K < TOL2", 0., 1.e10);
  defaults.add("tnse_coarsening", true, "coarsening on/off");
  defaults.add("tnse_max_cell_level", 30u, "maximal cell level", 1u, 100u);
  defaults.add("tnse_output_time", 0.25, "interval for mesh output", 0., 1.e10);
  defaults.add("tnse_write_table", "", "file name for the table");
  defaults.merge(NavierStokes<2>::default_nse_database(), true);
  defaults.merge(NSE_SPG_ErrorEstimator::default_spg_estimator_database(), true);
  defaults.merge(RefinementStrategy<2>::default_refinement_strategy_database(), true);
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

  bool time_adaptive = parmoon_db["tnse_time_adaptive"];
  double tolt = parmoon_db["tnse_tolt"];
  double k = time_adaptive ? (double) parmoon_db["tnse_dt0"]
                           : (double) parmoon_db["time_step_length"];
  double k_min = parmoon_db["tnse_dt_min"], k_max = parmoon_db["tnse_dt_max"];
  int n_star = parmoon_db["tnse_averaging_steps"];
  bool adaptive = parmoon_db["tnse_adaptive"];
  double tol1 = parmoon_db["tnse_tol1"], tol2 = parmoon_db["tnse_tol2"];
  bool coarsening = parmoon_db["tnse_coarsening"];
  size_t max_level = parmoon_db["tnse_max_cell_level"];
  int initial_level = parmoon_db["refinement_n_initial_steps"];
  double t_start = parmoon_db["time_start"], t_end = parmoon_db["time_end"];
  double output_every = parmoon_db["tnse_output_time"];
  double nonlin_tol = parmoon_db["nonlinloop_epsilon"];
  unsigned int maxit = parmoon_db["nonlinloop_maxit"];
  bool write_ps = parmoon_db["output_write_ps"];
  std::string table_file = parmoon_db["tnse_write_table"];

  Example_TimeNSE2D time_example(parmoon_db);
  Example_NSE2D example = time_example.get_stationary_example();
  NSE_SPG_ErrorEstimator estimator(parmoon_db);
  std::vector<TableRow> table;

  std::unique_ptr<TNSE_SPG> sys(new TNSE_SPG(domain, parmoon_db, example));
  // initial condition: interpolation of the exact solution at t = 0
  TDatabase::TimeDB->CURRENTTIME = t_start;
  sys->ns.get_velocity().GetComponent(0)->Interpolate(example.get_exact(0));
  sys->ns.get_velocity().GetComponent(1)->Interpolate(example.get_exact(1));
  sys->t = t_start;
  sys->initial_time_derivative(t_start);
  sys->k_old = k;
  Output::print("initial time derivative computed, |Ud^0| = ", sys->ud.norm());
  timer.restart_and_print("setup");

  bool first_step = true;
  int n_accepted = 0, n_rejected = 0, n_since_avg = 0;
  double next_output = t_start + output_every;
  while(sys->t < t_end - 1e-12)
  {
    k = std::min(k, t_end - sys->t);
    int n_picard;
    double err = sys->step(k, first_step, nonlin_tol, maxit, n_picard);
    if(time_adaptive && err > tolt && k > k_min)
    {
      sys->ns.get_solution() = sys->u_prev;
      sys->ud = sys->ud_prev;
      n_rejected++;
      k = 0.85*k*std::cbrt(tolt/err);
      Output::print("TR-AB2: step rejected, |e| = ", err, ", new dt = ", k);
      continue;
    }
    n_accepted++;
    n_since_avg++;
    sys->t += k;
    double k_eff = k;
    if(time_adaptive && n_star > 0 && n_since_avg >= n_star)
    {
      BlockVector& sol = sys->ns.get_solution();
      for(size_t i = 0; i < 2*sys->n_u; ++i)
      {
        sol.get_entries()[i] = 0.5*(sol.get_entries()[i] + sys->u_prev.get_entries()[i]);
        sys->ud.get_entries()[i] = 0.5*(sys->ud.get_entries()[i] + sys->ud_prev.get_entries()[i]);
      }
      sys->t -= 0.5*k;
      k_eff = 0.5*k;
      n_since_avg = 0;
    }
    sys->ud_old = sys->ud_prev;
    sys->k_old = k_eff;
    first_step = false;
    if(time_adaptive)
      k = std::min(0.85*k*std::cbrt(tolt/std::max(err, 1e-300)), k_max);
    TDatabase::TimeDB->CURRENTTIME = sys->t;
    Output::print("t = ", sys->t, " (step ", n_accepted, ", ", n_picard,
                  " Picard it., |e| = ", err, ", next dt = ", k, ")");

    // ---- estimator -----------------------------------------------------------
    TFEVectFunct2D ud_fct(sys->ns.get_velocity_space(), "ud", sys->ud.get_entries(), 2);
    estimator.set_transient(&ud_fct);
    estimator.estimate(sys->ns.get_example(), sys->ns.get_velocity(),
                       sys->ns.get_pressure());
    estimator.set_transient(nullptr);
    double eta = estimator.get_estimated_global_error();
    double error = estimator.get_error_spg();
    double eff = estimator.get_effectivity_index();
    int n_cells = sys->ns.get_velocity_space()->GetCollection()->GetN_Cells();
    Output::print("estimator: eta = ", eta, ", error_spg = ", error,
                  ", effectivity = ", eff, ", cells = ", n_cells);
    {
      auto parts = estimator.get_error_parts();
      Output::print<2>("  error parts: nu|grad e|^2 = ", parts[0], ", mu|div e|^2 = ",
                       parts[2], ", delta|supg|^2 = ", parts[3], ", |e_p|^2 = ",
                       parts[4], ";  eta parts: res ", estimator.get_eta_res(),
                       ", div ", estimator.get_eta_div(), ", F ", estimator.get_eta_F(),
                       ", delta ", estimator.get_eta_delta(), ", mu ", estimator.get_eta_mu());
    }

    int n_marked = 0, n_coarsen = 0;
    std::vector<bool> marks_refine(n_cells, false), marks_coarsen(n_cells, false);
    if(adaptive)
    {
      const TCollection* coll = estimator.get_collection();
      for(int c = 0; c < n_cells; ++c)
      {
        double eta_K = estimator.get_eta_K(c);
        if(eta_K > tol1 && coll->GetCell(c)->GetGeoLevel() < (int) max_level)
        { marks_refine[c] = true; n_marked++; }
        else if(coarsening && eta_K < tol2)
        { marks_coarsen[c] = true; n_coarsen++; }
      }
    }
    table.push_back({sys->t, k_eff, eta, error, eff, n_cells,
                     (int) sys->ns.get_size(), n_marked, n_coarsen, n_picard});
    if(sys->t >= next_output - 1e-12)
    {
      next_output += output_every;
      if(write_ps)
      {
        std::ostringstream name;
        name << "mesh_t" << std::fixed << std::setprecision(3) << sys->t << ".ps";
        domain.PS(name.str().c_str(), It_Finest, 0);
      }
      sys->ns.output();
    }

    // ---- mesh adaptation -----------------------------------------------------
    if(adaptive && (n_marked > 0 || n_coarsen > 0))
    {
      const TFEVectFunct2D& u = sys->ns.get_velocity();
      NodeValueMap map_u1 = node_values(*u.GetComponent(0));
      NodeValueMap map_u2 = node_values(*u.GetComponent(1));
      NodeValueMap map_p = node_values(sys->ns.get_pressure());
      NodeValueMap map_d1 = node_values(*ud_fct.GetComponent(0));
      NodeValueMap map_d2 = node_values(*ud_fct.GetComponent(1));
      TFEVectFunct2D udo_fct(sys->ns.get_velocity_space(), "udo",
                             sys->ud_old.get_entries(), 2);
      NodeValueMap map_do1 = node_values(*udo_fct.GetComponent(0));
      NodeValueMap map_do2 = node_values(*udo_fct.GetComponent(1));
      double t_now = sys->t, k_old_now = sys->k_old;
      int n_derefined = domain.RefineAndCoarsen(marks_refine, marks_coarsen,
                                                initial_level);
      std::unique_ptr<TNSE_SPG> sys_new(new TNSE_SPG(domain, parmoon_db, example));
      Output::print("adaptive: ", n_marked, " marked, ", n_derefined,
                    " parents derefined, new mesh has ",
                    domain.GetCollection(It_Finest, 0)->GetN_Cells(), " cells");
      sys_new->t = t_now;
      sys_new->k_old = k_old_now;
      transfer(*sys_new->ns.get_velocity().GetComponent(0), map_u1);
      transfer(*sys_new->ns.get_velocity().GetComponent(1), map_u2);
      transfer(sys_new->ns.get_pressure(), map_p);
      TFEVectFunct2D d_new(sys_new->ns.get_velocity_space(), "ud",
                           sys_new->ud.get_entries(), 2);
      transfer(*d_new.GetComponent(0), map_d1);
      transfer(*d_new.GetComponent(1), map_d2);
      TFEVectFunct2D do_new(sys_new->ns.get_velocity_space(), "udo",
                            sys_new->ud_old.get_entries(), 2);
      transfer(*do_new.GetComponent(0), map_do1);
      transfer(*do_new.GetComponent(1), map_do2);
      sys = std::move(sys_new);
    }
  }
  Output::print("TR-AB2: ", n_accepted, " accepted and ", n_rejected,
                " rejected steps");
  timer.restart_and_print("time loop");

  std::ostringstream t;
  t << "# time  dt  cells  dofs  eta  error_spg  effectivity  n_marked  n_coarsen  picard\n";
  for(auto& r : table)
    t << std::scientific << std::setprecision(6) << r.t << " " << r.k << " "
      << r.cells << " " << r.dofs << " " << r.eta << " " << r.error << " "
      << std::fixed << std::setprecision(4) << r.eff << " " << r.n_marked << " "
      << r.n_coarsen << " " << r.picard << "\n";
  Output::print("\n", t.str());
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << t.str();
  }
  parmoon::parmoon_finalize();
}
