// =============================================================================
// Adaptive SUPG method for evolutionary convection-diffusion-reaction equations
//
//   J. de Frutos, B. Garcia-Archilla, V. John, J. Novo, An adaptive SUPG
//   method for evolutionary convection-diffusion equations, Comput. Methods
//   Appl. Mech. Engrg. 273 (2014) 219-237.
//
// The fully discrete SUPG solution U_h^{n+1} is regarded as the SUPG solution
// of the steady-state problem (6) with right-hand side f - d_t u_h, so that
// the residual-based estimator of John & Novo (2013) (estimator_type 5 of
// CDErrorEstimator, Eq. (28) of the paper with C = 1) can be applied with
// d_t u_h ~ (U_h^{n+1} - U_h^n)/tau.
//
// Two modes:
//  * 'tcd_adaptive: false': fixed (uniformly refined) mesh; at every time step
//    the estimator and, if an exact solution is known, the error in the SUPG
//    norm (5) are computed and the effectivity index is written to a table
//    (Example 2 of the paper).
//  * 'tcd_adaptive: true': the algorithm of Section 3.3: after each time step
//    cells with eta_K > TOL1 are refined (coarsening, step (3) second part, is
//    not available in ParMooN's grid hierarchy and is therefore not done), the
//    solution is interpolated to the new mesh and the time stepping continues.
//    Mesh statistics (time, cells, DoFs, eta, [error, effectivity]) are
//    written to the table.
//
// Parameters (in addition to the usual TCD2D ones):
//   tcd_adaptive, tcd_tol1 (TOL1), tcd_tol2 (TOL2), tcd_coarsening,
//   tcd_estimator_mu0 (mu_0 in (28), c - div b/2), tcd_write_table,
//   tcd_refine_every (estimate/refine every k-th time step).
// =============================================================================
#include <Database.h>
#include <Domain.h>
#include "MainUtilities.h"
#include <TimeConvectionDiffusion.h>
#include <TimeDiscRout.h>
#include "ParMooN.h"
#include "CDErrorEstimator.h"
#include "RefinementStrategy.h"
#include "ConvDiff.h"
#include "AuxParam2D.h"
#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "LocalAssembling.h"
#include "Assemble2D.h"
#include "Solver.h"
#include "SquareMatrix2D.h"
#include "FESpace2D.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <memory>
#include "FEDatabase.h"
#include "NodalFunctional.h"
#include "Collection.h"
#include "BaseCell.h"
#include "Point.h"
#include <sstream>

// SUPG norm (5) of the error at the current time
double supg_error(const TFEFunction2D& u, const Example_TimeCD2D& ex,
                  const ParameterDatabase& db)
{
  if(ex.get_exact().empty() || ex.get_exact(0) == nullptr)
    return -1.;
  TAuxParam2D aux;
  MultiIndex2D all_derivatives[3] = {MultiIndex2D::D00, MultiIndex2D::D10,
                                     MultiIndex2D::D01};
  const TFESpace2D* space = u.GetFESpace2D().get();
  std::array<double, 5> err;
  u.GetErrors(ex.get_exact(0), 3, all_derivatives, 5,
              conv_diff_l2_h1_linf_error<2>, ex.get_coeffs(), &aux, 1, &space,
              err.data(), false, [](const TBaseCell*, int){return false;}, db);
  return err[2];
}

// Interpolate the fe function 'old_u', defined on the collection 'old_coll'
// (which must still exist and whose cells are nodes of the cell tree), to the
// fe function 'new_u' on the refined mesh. For every nodal point of the new
// mesh the old leaf cell containing it is found by descending the cell tree
// from the root cells, so the cost is O(N log N) instead of O(N^2) as with
// TFEFunction2D::Interpolate(TFEFunction2D*).
// key of a point for the vertex-value map (coordinates rounded to 1e-10)
inline std::pair<long long, long long> point_key(double x, double y)
{
  return {std::llround(x*1e9), std::llround(y*1e9)};
}
typedef std::map<std::pair<long long, long long>, double> VertexValueMap;

// For P1 (ANSATZ_ORDER 1) the dofs are the vertex values: store them keyed by
// the vertex coordinates. This map survives coarsening (vertices of removed
// cells are deleted, but the kept vertices keep their coordinates).
VertexValueMap vertex_values_P1(const TFEFunction2D& u)
{
  VertexValueMap m;
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  const double* values = u.GetValues();
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    const int* DOF = space->GetGlobalDOF(i);
    for(int j = 0; j < cell->GetN_Vertices(); ++j)
      m[point_key(cell->GetVertex(j)->GetX(), cell->GetVertex(j)->GetY())]
        = values[DOF[j]];
  }
  return m;
}

// P1 transfer after refinement and coarsening: every vertex of the new mesh is
// either an old vertex (value from the map) or a new midpoint of an edge of
// the parent cell, whose end points are old vertices (linear interpolation).
// Works with the conforming closure, whose cells are rebuilt in every step.
void interpolate_P1(TFEFunction2D& new_u, const VertexValueMap& vertex_map)
{
  auto space = new_u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  double* values = new_u.GetValues();
  auto lookup = [&](double x, double y, double& value) -> bool
  {
    auto it = vertex_map.find(point_key(x, y));
    if(it == vertex_map.end())
      return false;
    value = it->second;
    return true;
  };
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    const TBaseCell* cell = coll->GetCell(i);
    const int* DOF = space->GetGlobalDOF(i);
    for(int j = 0; j < cell->GetN_Vertices(); ++j)
    {
      double x = cell->GetVertex(j)->GetX(), y = cell->GetVertex(j)->GetY();
      double value;
      if(lookup(x, y, value))
      {
        values[DOF[j]] = value;
        continue;
      }
      // new vertex: midpoint of an edge of an ancestor cell
      const TBaseCell* parent = cell->GetParent();
      bool found = false;
      while(parent != nullptr && !found)
      {
        int nv = parent->GetN_Vertices();
        for(int a = 0; a < nv && !found; ++a)
        {
          for(int b = a+1; b < nv && !found; ++b)
          {
            double xa = parent->GetVertex(a)->GetX(), ya = parent->GetVertex(a)->GetY();
            double xb = parent->GetVertex(b)->GetX(), yb = parent->GetVertex(b)->GetY();
            if(std::abs(0.5*(xa+xb) - x) < 1e-10 && std::abs(0.5*(ya+yb) - y) < 1e-10)
            {
              double va, vb;
              if(lookup(xa, ya, va) && lookup(xb, yb, vb))
              {
                values[DOF[j]] = 0.5*(va + vb);
                found = true;
              }
            }
          }
        }
        parent = parent->GetParent();
      }
      if(!found)
        ErrThrow("interpolate_P1: no value for vertex (", x, ",", y, ")");
    }
  }
}

void interpolate_from_old_mesh(TFEFunction2D& new_u, const TFEFunction2D& old_u,
                               const TCollection* old_coll, TBaseCell** roots,
                               int n_roots,
                               const VertexValueMap* vertex_map = nullptr)
{
  std::map<const TBaseCell*, int> old_index;
  for(int i = 0; i < old_coll->GetN_Cells(); ++i)
    old_index[old_coll->GetCell(i)] = i;
  auto find_old_leaf = [&](double x, double y, int& idx) -> const TBaseCell*
  {
    parmoon::Point q(x, y);
    const TBaseCell* c = nullptr;
    for(int r = 0; r < n_roots; ++r)
    {
      if(roots[r]->PointInCell(q))
      {
        c = roots[r];
        break;
      }
    }
    while(c != nullptr)
    {
      auto it = old_index.find(c);
      if(it != old_index.end())
      {
        idx = it->second;
        return c;
      }
      const TBaseCell* next = nullptr;
      for(int k = 0; k < c->GetN_Children(); ++k)
      {
        if(c->GetChild(k)->PointInCell(q))
        {
          next = c->GetChild(k);
          break;
        }
      }
      c = next;
    }
    return nullptr;
  };

  auto space = new_u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  double* values = new_u.GetValues();
  for(int i = 0; i < coll->GetN_Cells(); ++i)
  {
    TBaseCell* cell = coll->GetCell(i);
    const FiniteElement& element = space->get_fe(i);
    const NodalFunctional* nf = element.GetNodalFunctional();
    int n_points;
    const double *xi, *eta;
    nf->GetPointsForAll(n_points, xi, eta);
    double X[MaxN_PointsForNodal2D], Y[MaxN_PointsForNodal2D];
    double point_values[MaxN_PointsForNodal2D];
    double functional_values[MaxN_PointsForNodal2D];
    ReferenceTransformation_type ref_trans = element.GetRefTransID();
    FEDatabase::SetCellForRefTrans(cell, ref_trans);
    FEDatabase::GetOrigFromRef(ref_trans, n_points, xi, eta, X, Y);
    for(int j = 0; j < n_points; ++j)
    {
      if(vertex_map != nullptr)
      {
        auto it = vertex_map->find(point_key(X[j], Y[j]));
        if(it != vertex_map->end())
        {
          point_values[j] = it->second;
          continue;
        }
      }
      int idx;
      const TBaseCell* old_cell = find_old_leaf(X[j], Y[j], idx);
      if(old_cell == nullptr)
        ErrThrow("interpolate_from_old_mesh: point (", X[j], ",", Y[j],
                 ") not found in the old mesh");
      double v[3];
      old_u.FindValueLocal(old_cell, idx, X[j], Y[j], v);
      point_values[j] = v[0];
    }
    nf->GetAllFunctionals(coll, cell, point_values,
                          functional_values);
    const int* DOF = space->GetGlobalDOF(i);
    for(int j = 0; j < element.GetN_DOF(); ++j)
      values[DOF[j]] = functional_values[j];
  }
}


// =============================================================================
// Variable step trapezoidal rule with AB2 error estimation and stabilization
// (Gresho, Griffiths, Silvester, SIAM J. Sci. Comput. 30 (2008) 2018-2054;
// Section 3.3 of de Frutos et al. 2014) for the SUPG semi-discretization
//     M_s u' + A_s u = f_s,
// M_s = (u,v) + (u, delta b.grad v)_h, A_s = a_SUPG, f_s = (f,v) + (f, delta b.grad v)_h.
// Notation: u^n, d^n = u'(t_n) (consistent TR time derivative), step k_n.
//   TR:    (M_s + k_n/2 A_s) u^{n+1} = M_s (u^n + k_n/2 d^n) + k_n/2 f_s(t_{n+1})
//          d^{n+1} = 2 (u^{n+1} - u^n)/k_n - d^n
//   AB2:   u_p = u^n + k_n/2 [ (2 + r) d^n - r d^{n-1} ],  r = k_n/k_{n-1}
//   LTE:   e = (u^{n+1} - u_p) / (3 (1 + 1/r)),  measured in the L^2(Omega) norm
//   step:  accept if |e| <= TOLT, k_{n+1} = 0.85 k_n (TOLT/|e|)^{1/3} (the factor
//          0.85 is the modification of de Frutos et al.), rejected steps are
//          repeated with the reduced step
//   stabilization of the (-1)^n mode: every n* accepted steps
//          u^{n+1} <- (u^{n+1}+u^n)/2, d^{n+1} <- (d^{n+1}+d^n)/2,
//          t_{n+1} <- t_{n+1} - k_n/2, k_n <- k_n/2
// The SUPG matrices are assembled with THETA1 = 1 and tau = 1 so that the
// local routines of ParMooN use delta_K = delta_0 h_K / |b| (SDFEM_TYPE 0).
// =============================================================================
class TRAB2_SUPG
{
  public:
    std::shared_ptr<const TFESpace2D> space;
    const Example_TimeCD2D& example;
    ParameterDatabase db;
    BlockFEMatrix M, A, M_galerkin, S;
    BlockVector u, d, d_old, rhs, work;
    std::unique_ptr<TFEFunction2D> u_fct;
    Solver<BlockFEMatrix, BlockVector> solver;
    double t;        // current time
    double k, k_old; // current and previous step length

    TRAB2_SUPG(std::shared_ptr<const TFESpace2D> sp, const Example_TimeCD2D& ex,
               const ParameterDatabase& param_db)
     : space(sp), example(ex), db(param_db), M(BlockFEMatrix::CD2D(sp)),
       A(BlockFEMatrix::CD2D(sp)), M_galerkin(BlockFEMatrix::CD2D(sp)),
       S(BlockFEMatrix::CD2D(sp)), u(M, false), d(M, false), d_old(M, false),
       rhs(M, true), work(M, true), solver(param_db), t(0.), k(0.), k_old(0.)
    {
      u_fct.reset(new TFEFunction2D(space, "u", u.get_entries()));
      // Galerkin mass matrix for the L2 norm of the error estimate
      ParameterDatabase db_g(db);
      db_g["space_discretization_type"] = "galerkin";
      TFEFunction2D dummy(space, "dummy", u.get_entries());
      std::vector<const TFEFunction2D*> fcts = {&dummy};
      LocalAssembling<2> la(db_g, LocalAssembling_type::TCDMassOnly, fcts,
                            example.get_coeffs());
      // TCDMass writes into the second local matrix, so a dummy first one
      // (the system matrix S, reset later anyway) is passed
      auto block = M_galerkin.get_blocks_uniquely()[0].get();
      auto dummy_block = S.get_blocks_uniquely()[0].get();
      TSquareMatrix2D* sq[2] = {reinterpret_cast<TSquareMatrix2D*>(dummy_block),
                                reinterpret_cast<TSquareMatrix2D*>(block)};
      sq[0]->reset();
      sq[1]->reset();
      const TFESpace2D* sp_ptr = space.get();
      BoundCondFunct2D* bc = space->get_boundary_condition();
      BoundValueFunct2D* bd = example.get_bd(0);
      Assemble2D(1, &sp_ptr, 2, sq, 0, nullptr, 0, nullptr, &sp_ptr, &bc, &bd,
                 la, true);
    }

    // assemble M_s, A_s and f_s at time time
    void assemble(double time)
    {
      TDatabase::TimeDB->CURRENTTIME = time;
      TDatabase::TimeDB->THETA1 = 1.;
      TDatabase::TimeDB->CURRENTTIMESTEPLENGTH = 1.;
      std::vector<const TFEFunction2D*> fcts = {u_fct.get()};
      LocalAssembling<2> la(db, LocalAssembling_type::TCDStiffMassRhs, fcts,
                            example.get_coeffs());
      auto A_block = A.get_blocks_uniquely()[0].get();
      auto M_block = M.get_blocks_uniquely()[0].get();
      TSquareMatrix2D* sq[2] = {reinterpret_cast<TSquareMatrix2D*>(A_block),
                                reinterpret_cast<TSquareMatrix2D*>(M_block)};
      sq[0]->reset();
      sq[1]->reset();
      rhs.reset();
      double* rhs_entries = rhs.get_entries();
      const TFESpace2D* sp_ptr = space.get();
      BoundCondFunct2D* bc = space->get_boundary_condition();
      BoundValueFunct2D* bd = example.get_bd(0);
      Assemble2D(1, &sp_ptr, 2, sq, 0, nullptr, 1, &rhs_entries, &sp_ptr, &bc,
                 &bd, la, true);
    }

    // d = M_s^{-1} (f_s - A_s u) on the active dofs, 0 on Dirichlet dofs
    void compute_time_derivative(double time)
    {
      assemble(time);
      work = rhs;
      A.apply_scaled_add(u, work, -1.);
      // Dirichlet rows: d = 0 (boundary data constant in time)
      size_t n_active = space->get_n_active();
      for(size_t i = n_active; i < work.length(); ++i)
        work.get_entries()[i] = 0.;
      solver.solve(M, work, d);
    }

    double l2_norm(const BlockVector& v)
    {
      BlockVector Mv(v);
      M_galerkin.apply(v, Mv);
      return std::sqrt(std::abs(dot(Mv, v)));
    }

    // one TR step of length k from t: returns the error estimate; u, d are
    // updated, the previous values are kept in u_prev, d_prev
    double step(double step_length, BlockVector& u_prev, BlockVector& d_prev,
                BlockVector& u_pred, bool first_step)
    {
      u_prev = u;
      d_prev = d;
      double r = first_step ? 1. : step_length / k_old;
      // AB2 predictor
      u_pred = u;
      u_pred.add_scaled(d, 0.5*step_length*(2. + r));
      u_pred.add_scaled(d_old, -0.5*step_length*r);
      // TR step
      assemble(t + step_length);
      S = A;
      std::vector<std::vector<size_t>> cells = {{0, 0}};
      S.scale_blocks_actives(0.5*step_length, cells);
      S.add_blockfe_matrix(M);
      work = u;
      work.add_scaled(d, 0.5*step_length);
      BlockVector b(rhs);
      b.scaleActive(0.5*step_length);
      M.apply_scaled_add(work, b, 1.);
      b.copy_nonactive(rhs); // Dirichlet values at the new time
      u.copy_nonactive(rhs);
      solver.solve(S, b, u);
      // consistent time derivative
      d = u;
      d.add_scaled(u_prev, -1.);
      d.scale(2./step_length);
      d.add_scaled(d_prev, -1.);
      size_t n_active = space->get_n_active();
      for(size_t i = n_active; i < d.length(); ++i)
        d.get_entries()[i] = 0.;
      // local error estimate
      BlockVector e(u);
      e.add_scaled(u_pred, -1.);
      double err = l2_norm(e) / (3.*(1. + 1./r));
      return err;
    }
};


// =============================================================================
// time loop with the variable step TR-AB2 integrator and adaptive meshes
// =============================================================================
void run_trab2(TDomain& domain, const Example_TimeCD2D& example,
               const Example_CD2D& steady_example, ParameterDatabase& db,
               CDErrorEstimator<2>& estimator, std::ostringstream& table)
{
  bool adaptive = db["tcd_adaptive"];
  double tol1 = db["tcd_tol1"];
  double tol2 = db["tcd_tol2"];
  bool is_P1 = (TDatabase::ParamDB->ANSATZ_ORDER == 1);
  bool coarsening = db["tcd_coarsening"] && is_P1;
  int initial_level = db["refinement_n_initial_steps"];
  size_t max_level = db["tcd_max_cell_level"];
  double t_end = db["time_end"];
  double tolt = db["tcd_tolt"];
  double k = db["tcd_dt0"];
  double k_max = db["tcd_dt_max"];
  double k_min = db["tcd_dt_min"];
  int n_star = db["tcd_averaging_steps"];
  double output_every = db["tcd_output_time"];
  bool write_ps = db["output_write_ps"];
  BoundCondFunct2D* bc = example.get_bc(0);
  int order = TDatabase::ParamDB->ANSATZ_ORDER;

  auto make_space = [&]() {
    return std::shared_ptr<const TFESpace2D>(new TFESpace2D(
      domain.GetCollection(It_Finest, 0), "space", bc, order));
  };
  std::unique_ptr<TRAB2_SUPG> sys(new TRAB2_SUPG(make_space(), example, db));
  sys->u_fct->Interpolate(example.get_initial_cond(0));
  sys->t = db["time_start"];
  sys->compute_time_derivative(sys->t);
  sys->d_old = sys->d;
  sys->k_old = k;
  bool first_step = true;
  BlockVector u_prev(sys->u), d_prev(sys->d), u_pred(sys->u);
  int n_accepted = 0, n_rejected = 0, n_since_avg = 0;
  double next_output = sys->t + output_every;
  Output::print("TR-AB2: TOLT = ", tolt, ", dt0 = ", k, ", averaging every ",
                n_star, " steps");

  while(sys->t < t_end - 1e-12)
  {
    k = std::min(k, t_end - sys->t);
    double err = sys->step(k, u_prev, d_prev, u_pred, first_step);
    if(err > tolt && k > k_min)
    {
      // reject and repeat with a smaller step
      sys->u = u_prev;
      sys->d = d_prev;
      n_rejected++;
      k = 0.85 * k * std::cbrt(tolt / err);
      Output::print<2>("TR-AB2: step rejected, |e| = ", err, ", new dt = ", k);
      continue;
    }
    n_accepted++;
    n_since_avg++;
    sys->t += k;
    double k_eff = k;
    if(n_star > 0 && n_since_avg >= n_star)
    {
      // stabilization: average solution and derivative at t - k/2
      sys->u.add_scaled(u_prev, 1.);
      sys->u.scale(0.5);
      sys->d.add_scaled(d_prev, 1.);
      sys->d.scale(0.5);
      sys->t -= 0.5*k;
      k_eff = 0.5*k;
      n_since_avg = 0;
    }
    sys->d_old = d_prev;
    sys->k_old = k_eff;
    first_step = false;
    double k_new = 0.85 * k * std::cbrt(tolt / std::max(err, 1e-300));
    k = std::min(k_new, k_max);
    TDatabase::TimeDB->CURRENTTIME = sys->t;
    Output::print("TR-AB2: t = ", sys->t, " (step ", n_accepted, ", |e| = ",
                  err, ", next dt = ", k, ")");

    // --- error estimation and mesh adaptation -------------------------------
    const TFEFunction2D& uh = *sys->u_fct;
    TFEFunction2D dt_uh(sys->space, "dt_u", sys->d.get_entries());
    estimator.set_time_derivative(&dt_uh);
    estimator.estimate(steady_example, uh);
    estimator.set_time_derivative(nullptr);
    double eta = estimator.get_estimated_global_error();
    double error = supg_error(uh, example, db);
    double eff = (error > 0.) ? eta/error : 0.;
    int n_cells = sys->space->GetCollection()->GetN_Cells();
    int n_dofs = sys->space->get_n_dof();
    Output::print("estimator: eta = ", eta, ", error_supg = ", error,
                  ", effectivity index = ", eff, ", cells = ", n_cells);
    int n_marked = 0, n_coarsen_marked = 0;
    std::vector<bool> marks_refine(n_cells, false), marks_coarsen(n_cells, false);
    const TCollection* coll = estimator.get_collection();
    if(adaptive)
    {
      for(int c = 0; c < n_cells; ++c)
      {
        double eta_K = estimator.get_eta_K(c);
        if(eta_K > tol1 && coll->GetCell(c)->GetGeoLevel() < (int) max_level)
        {
          marks_refine[c] = true;
          n_marked++;
        }
        else if(eta_K < tol2 && coarsening)
        {
          marks_coarsen[c] = true;
          n_coarsen_marked++;
        }
      }
    }
    table << std::scientific << std::setprecision(6) << sys->t << " "
          << n_cells << " " << n_dofs << " " << eta << " " << error << " "
          << std::fixed << std::setprecision(4) << eff << " " << n_marked << " "
          << n_coarsen_marked << " " << std::scientific << k_eff << "\n";
    bool do_output = (sys->t >= next_output - 1e-12);
    if(do_output)
    {
      next_output += output_every;
      if(write_ps)
      {
        std::ostringstream name;
        name << "mesh_t" << std::fixed << std::setprecision(3) << sys->t << ".ps";
        domain.PS(name.str().c_str(), It_Finest, 0);
      }
      double umin = *std::min_element(sys->u.get_entries(), sys->u.get_entries() + sys->u.length());
      double umax = *std::max_element(sys->u.get_entries(), sys->u.get_entries() + sys->u.length());
      Output::print("solution range at t = ", sys->t, ": min = ", umin,
                    ", max = ", umax, ", cells = ", n_cells);
    }
    if(adaptive && (n_marked > 0 || n_coarsen_marked > 0))
    {
      if(!is_P1)
        ErrThrow("TR-AB2 with adaptive meshes is only implemented for P1");
      TFEFunction2D u_old(sys->space, "u", sys->u.get_entries());
      TFEFunction2D d_old_fct(sys->space, "d", sys->d.get_entries());
      TFEFunction2D dold_old_fct(sys->space, "dold", sys->d_old.get_entries());
      VertexValueMap map_u = vertex_values_P1(u_old);
      VertexValueMap map_d = vertex_values_P1(d_old_fct);
      VertexValueMap map_dold = vertex_values_P1(dold_old_fct);
      int n_derefined = domain.RefineAndCoarsen(marks_refine, marks_coarsen,
                                                initial_level);
      Output::print("adaptive: ", n_marked, " marked, ", n_derefined,
                    " parents derefined, new mesh has ",
                    domain.GetCollection(It_Finest, 0)->GetN_Cells(), " cells");
      std::unique_ptr<TRAB2_SUPG> sys_new(new TRAB2_SUPG(make_space(), example, db));
      sys_new->t = sys->t;
      sys_new->k_old = sys->k_old;
      TFEFunction2D d_new(sys_new->space, "d", sys_new->d.get_entries());
      TFEFunction2D dold_new(sys_new->space, "dold", sys_new->d_old.get_entries());
      interpolate_P1(*sys_new->u_fct, map_u);
      interpolate_P1(d_new, map_d);
      interpolate_P1(dold_new, map_dold);
      sys = std::move(sys_new);
      u_prev = sys->u;
      d_prev = sys->d;
      u_pred = sys->u;
    }
  }
  Output::print("TR-AB2: ", n_accepted, " accepted and ", n_rejected,
                " rejected steps");
}

int main(int argc, char* argv[])
{
  // the default values of the parameters of this program; parmoon_initialize
  // already reads the input file, so values given there take precedence
  ParameterDatabase defaults("TCD2D SUPG adaptive defaults");
  defaults.add("tcd_adaptive", false, "adaptive mesh refinement in time");
  defaults.add("tcd_tol1", 1.e-3, "cells with eta_K > TOL1 are refined", 0., 1.e10);
  defaults.add("tcd_tol2", 5.e-5, "cells with eta_K < TOL2 are candidates for "
               "coarsening", 0., 1.e10);
  defaults.add("tcd_coarsening", true, "coarsen the mesh (only P1)");
  defaults.add("tcd_time_adaptive", false, "use the variable step TR-AB2 "
               "integrator of Gresho/Griffiths/Silvester (Section 3.3)");
  defaults.add("tcd_tolt", 1.e-5, "tolerance TOLT of the time integrator", 0., 1.);
  defaults.add("tcd_dt0", 1.e-10, "initial time step", 0., 1.);
  defaults.add("tcd_dt_max", 0.1, "maximal time step", 0., 1.e10);
  defaults.add("tcd_dt_min", 1.e-12, "minimal time step (no rejection below)", 0., 1.);
  defaults.add("tcd_averaging_steps", 10, "stabilization: average every n* steps "
               "(0 switches it off)", 0, 1000000);
  defaults.add("tcd_output_time", 0.1, "interval for mesh output and solution "
               "range in the TR-AB2 mode", 0., 1.e10);
  defaults.add("tcd_ps_every", 1u, "write a PostScript file of the adaptive "
               "mesh every k-th step (needs output_write_ps)", 1u, 1000000u);
  defaults.add("tcd_estimator_mu0", 1., "lower bound mu_0 of c - div(b)/2 "
               "used in the estimator; 0 drops the corresponding terms",
               0., 1.e10);
  defaults.add("tcd_write_table", "", "file name for the table");
  defaults.add("tcd_refine_every", 1u, "estimate and refine every k-th step",
               1u, 1000000u);
  defaults.add("tcd_max_cell_level", 30u, "maximal refinement level of a cell",
               1u, 100u);
  defaults.merge(ErrorEstimator<2>::default_error_estimator_database(), true);
  defaults.merge(RefinementStrategy<2>::default_refinement_strategy_database(),
                 true);
  auto parmoon_db = parmoon::parmoon_initialize(argc, argv);
  Output::warn("DEPRECATED", "this monolithic program is kept for reference only "
               "and contains defects fixed in tcd2d_adaptive_tr: Dirichlet data are "
               "assumed constant in time, the step size control uses a wrong L2 "
               "norm (NSE) and the estimator after an averaging step refers to "
               "the wrong time. Use tcd2d_adaptive_tr.");
  // values from the input file take precedence over the defaults
  defaults.merge(parmoon_db, true);
  parmoon_db = defaults;
  // the estimator uses the John-Novo SUPG estimator
  parmoon_db["estimator_type"] = 5;
  TDatabase::ParamDB->INTERNAL_COERCIVITY = parmoon_db["tcd_estimator_mu0"];

  TDomain domain(parmoon_db);
  domain.refine_and_get_hierarchy_of_collections(parmoon_db);
  if(parmoon_db["output_write_ps"])
    domain.PS("Domain.ps", It_Finest, 0);

  bool adaptive = parmoon_db["tcd_adaptive"];
  double tol1 = parmoon_db["tcd_tol1"];
  double tol2 = parmoon_db["tcd_tol2"];
  bool is_P1 = (TDatabase::ParamDB->ANSATZ_ORDER == 1);
  bool coarsening = parmoon_db["tcd_coarsening"];
  if(coarsening && !is_P1)
  {
    Output::warn("TCD2D_SUPG_Adaptive", "coarsening is only implemented for "
                 "P1 (ANSATZ_ORDER 1), it is switched off");
    coarsening = false;
  }
  int initial_level = parmoon_db["refinement_n_initial_steps"];
  size_t ps_every = parmoon_db["tcd_ps_every"];
  size_t refine_every = parmoon_db["tcd_refine_every"];
  size_t max_level = parmoon_db["tcd_max_cell_level"];
  std::string table_file = parmoon_db["tcd_write_table"];
  std::ostringstream table;
  table << "# time  cells  dofs  eta  error_supg  effectivity  n_marked_refine  n_marked_coarsen\n";

  // the time-dependent example and the steady-state example (same functions)
  // needed by the estimator
  Example_TimeCD2D example(parmoon_db);
  Example_CD2D steady_example(example.get_exact(),
    std::vector<BoundCondFunct2D*>(example.get_bc(), example.get_bc() + 1),
    std::vector<BoundValueFunct2D*>(example.get_bd(), example.get_bd() + 1),
    example.get_coeffs());
  CDErrorEstimator<2> estimator(parmoon_db);
  RefinementStrategy<2> refinement_strategy(parmoon_db);

  if(parmoon_db["tcd_time_adaptive"])
  {
    table.str("");
    table << "# time  cells  dofs  eta  error_supg  effectivity  n_marked_refine"
             "  n_marked_coarsen  dt\n";
    run_trab2(domain, example, steady_example, parmoon_db, estimator, table);
    Output::print("\n", table.str());
    if(!table_file.empty())
    {
      std::ofstream out(table_file);
      out << table.str();
    }
    parmoon::parmoon_finalize();
    return 0;
  }
  std::unique_ptr<TimeConvectionDiffusion<2>> tcd(
    new TimeConvectionDiffusion<2>(domain, parmoon_db, example));
  tcd->get_time_stepping_scheme().current_step_ = 0;
  tcd->get_time_stepping_scheme().set_time_disc_parameters();
  tcd->assemble_initial_time();
  double start_time = parmoon_db["time_start"];
  TDatabase::TimeDB->CURRENTTIME = start_time;
  tcd->output();

  double current_time = start_time;
  size_t step = 0;
  while(!tcd->get_time_stepping_scheme().reached_final_time_step())
  {
    TimeDiscretization& tss = tcd->get_time_stepping_scheme();
    step++;
    tss.current_step_++;
    tss.set_time_disc_parameters();
    tss.current_time_ += tss.get_step_length();
    double tau = parmoon_db["time_step_length"];
    current_time += tau;
    TDatabase::TimeDB->CURRENTTIME = current_time;
    Output::print("\nCURRENT TIME: ", current_time);
    SetTimeDiscParameters(1);
    tcd->assemble();
    tcd->solve();
    tcd->output();

    if(step % refine_every != 0)
      continue;

    // d_t u_h ~ (U^{n+1} - U^n)/tau in the same fe space. Note that at the end
    // of TimeConvectionDiffusion::solve() solution_m1 is already set to the
    // new solution, the previous one is solution_m2.
    const TFEFunction2D& uh = tcd->get_function();
    BlockVector dt_vec(tcd->get_solution());
    dt_vec.add_scaled(tcd->get_solution_m2(), -1.);
    dt_vec.scale(1./tau);
    TFEFunction2D dt_uh(uh.GetFESpace2D(), "dt_u", dt_vec.get_entries());
    estimator.set_time_derivative(&dt_uh);
    estimator.estimate(steady_example, uh);
    estimator.set_time_derivative(nullptr);
    double eta = estimator.get_estimated_global_error();
    double error = supg_error(uh, example, parmoon_db);
    double eff = (error > 0.) ? eta/error : 0.;
    int n_cells = uh.GetFESpace2D()->GetCollection()->GetN_Cells();
    int n_dofs = uh.GetFESpace2D()->get_n_dof();
    Output::print("estimator: eta = ", eta, ", error_supg = ", error,
                  ", effectivity index = ", eff, ", cells = ", n_cells);

    int n_marked = 0, n_coarsen_marked = 0;
    std::vector<bool> marks_refine(n_cells, false), marks_coarsen(n_cells, false);
    if(adaptive)
    {
      // step (3): mark cells with eta_K > TOL1 for refinement and cells with
      // eta_K < TOL2 as candidates for coarsening
      const TCollection* coll = estimator.get_collection();
      for(int k = 0; k < n_cells; ++k)
      {
        double eta_K = estimator.get_eta_K(k);
        if(eta_K > tol1 && coll->GetCell(k)->GetGeoLevel() < (int) max_level)
        {
          marks_refine[k] = true;
          n_marked++;
        }
        else if(eta_K < tol2 && coarsening)
        {
          marks_coarsen[k] = true;
          n_coarsen_marked++;
        }
      }
      Output::print("adaptive: ", n_marked, " of ", n_cells, " cells marked for "
                    "refinement, ", n_coarsen_marked, " candidates for coarsening");
    }
    table << std::scientific << std::setprecision(6) << current_time << " "
          << n_cells << " " << n_dofs << " " << eta << " " << error << " "
          << std::fixed << std::setprecision(4) << eff << " " << n_marked << " "
          << n_coarsen_marked << "\n";

    if(adaptive && (n_marked > 0 || n_coarsen_marked > 0))
    {
      // steps (4), (5): refine/coarsen, create the new system, transfer the
      // solution and continue.
      // keep the old mesh (as a collection of the same cells) and the old
      // solution alive until it has been interpolated: the domain deletes its
      // finest collection during refinement. Cells removed by coarsening are
      // deleted, so for P1 the vertex values are stored by coordinates.
      const TCollection* coll = estimator.get_collection();
      int n_old = coll->GetN_Cells();
      TBaseCell** old_cells = new TBaseCell*[n_old];
      for(int k = 0; k < n_old; ++k)
        old_cells[k] = coll->GetCell(k);
      std::unique_ptr<TCollection> old_coll(new TCollection(n_old, old_cells));
      std::shared_ptr<TFESpace2D> old_space(new TFESpace2D(
        old_coll.get(), "old_space", example.get_bc(0),
        TDatabase::ParamDB->ANSATZ_ORDER));
      if(old_space->get_n_dof() != uh.GetFESpace2D()->get_n_dof())
        ErrThrow("old space has a different number of dofs");
      std::vector<double> old_values(uh.GetValues(),
                                     uh.GetValues() + uh.GetLength());
      TFEFunction2D old_function(old_space, "u_old", old_values.data());
      VertexValueMap vertex_map;
      if(is_P1)
        vertex_map = vertex_values_P1(old_function);
      TBaseCell** roots;
      int n_roots;
      domain.GetTreeInfo(roots, n_roots);
      int n_derefined = domain.RefineAndCoarsen(marks_refine, marks_coarsen,
                                                initial_level);
      Output::print("adaptive: ", n_derefined, " parent cells derefined, new "
                    "mesh has ", domain.GetCollection(It_Finest, 0)->GetN_Cells(),
                    " cells");
      std::unique_ptr<TimeConvectionDiffusion<2>> tcd_new(
        new TimeConvectionDiffusion<2>(domain, parmoon_db, example));
      // copy the state of the time stepping scheme
      TimeDiscretization& tss_new = tcd_new->get_time_stepping_scheme();
      tss_new.current_step_ = tcd->get_time_stepping_scheme().current_step_;
      tss_new.current_time_ = tcd->get_time_stepping_scheme().current_time_;
      tss_new.set_time_disc_parameters();
      if(is_P1)
        interpolate_P1(tcd_new->get_function(), vertex_map);
      else
        interpolate_from_old_mesh(tcd_new->get_function(), old_function,
                                  old_coll.get(), roots, n_roots);
      // (re)assemble the matrices on the new mesh; this also sets the old
      // solution to the interpolated one
      tcd_new->assemble_initial_time();
      tcd = std::move(tcd_new);
      if(parmoon_db["output_write_ps"] && step % ps_every == 0)
      {
        std::ostringstream name;
        name << "mesh_" << std::setw(6) << std::setfill('0') << step << ".ps";
        domain.PS(name.str().c_str(), It_Finest, 0);
      }
    }
  }

  Output::print("\n", table.str());
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << table.str();
  }
  parmoon::parmoon_finalize();
}
