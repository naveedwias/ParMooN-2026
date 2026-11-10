// =============================================================================
// Implementation of NSE_SPG_ErrorEstimator.h, see the header for the formulas.
// =============================================================================
#ifdef __2D__
#include "NSE_SPG_ErrorEstimator.h"
#include "NSE_local_assembling_routines.h"
#include "BaseCell.h"
#include "BoundEdge.h"
#include "Collection.h"
#include "Database.h"
#include "FEDatabase.h"
#include "FESpace2D.h"
#include "Joint.h"
#include "QuadratureFormulaDatabase.h"
#include "RefDesc.h"
#include "ShapeDesc.h"
#include "Vertex.h"
#include "Point.h"
#include "MooNMD_Io.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
  // derivatives needed in the mesh cells: velocity and pressure
  const std::vector<MultiIndex2D> derivatives_u {MultiIndex2D::D00,
                                                 MultiIndex2D::D10,
                                                 MultiIndex2D::D01,
                                                 MultiIndex2D::D20,
                                                 MultiIndex2D::D02};
  const std::vector<MultiIndex2D> derivatives_p {MultiIndex2D::D00,
                                                 MultiIndex2D::D10,
                                                 MultiIndex2D::D01};
  constexpr int max_n_coeffs = 20;


  // descend from cell to the leaf (cell without children) containing point q
  const TBaseCell* leaf_containing(const TBaseCell* c, parmoon::Point q)
  {
    while(c->ExistChildren())
    {
      const TBaseCell* next = nullptr;
      for(int i = 0; i < c->GetN_Children(); ++i)
      {
        if(c->GetChild(i)->PointInCell(q))
        {
          next = c->GetChild(i);
          break;
        }
      }
      if(next == nullptr)
        return nullptr;
      c = next;
    }
    return c;
  }

  // Find the finest-level cell on the other side of an edge of 'cell' with
  // midpoint (xm,ym) and outer unit normal (nx,ny), when the joint of the
  // edge is still attached to an ancestor of 'cell' (this happens for the
  // bisection refinements used in the conforming closure, where a child keeps
  // a whole edge of its parent). The search walks up the cell tree and tests a
  // point slightly outside the edge.
  const TBaseCell* find_leaf_neighbour(const TBaseCell* cell, double xm,
                                       double ym, double nx, double ny,
                                       double hE)
  {
    parmoon::Point q(xm + 1e-3*hE*nx, ym + 1e-3*hE*ny);
    const TBaseCell* p = cell->GetParent();
    while(p != nullptr)
    {
      for(int e = 0; e < p->GetN_Edges(); ++e)
      {
        const TBaseCell* n = p->GetJoint(e)->GetNeighbour(p);
        if(n != nullptr && n->PointInCell(q))
          return leaf_containing(n, q);
      }
      p = p->GetParent();
    }
    return nullptr;
  }

  inline double min_positive(double a, double b)
  {
    // minimum where a non-positive (or infinite) argument means "not available"
    bool a_ok = (a > 0. && std::isfinite(a));
    bool b_ok = (b > 0. && std::isfinite(b));
    if(a_ok && b_ok) return std::min(a, b);
    if(a_ok) return a;
    if(b_ok) return b;
    return 0.;
  }
}

ParameterDatabase NSE_SPG_ErrorEstimator::default_spg_estimator_database()
{
  ParameterDatabase db("default SUPG/PSPG/grad-div estimator database");
  db.add("spg_delta0", 0.5, "SUPG/PSPG parameter delta_0, see LocalAssembling",
         0., 100.);
  db.add("spg_mu0", 0.5, "grad-div parameter mu_0, see LocalAssembling",
         0., 100.);
  db.add("spg_estimator_sigma0", 0.,
         "lower bound sigma_0 of the reaction coefficient used in the minima of "
         "eta_{K,res} and eta_F; 0 means that the corresponding terms are "
         "dropped.", 0., 1.e10);
  db.add("problem_type", 3u, "3: Oseen (linear), 5: Navier-Stokes", 0u, 10u);
  db.add("spg_error_quad_degree", 19, "polynomial degree of the quadrature "
         "used for the exact error (solutions with layers need a high degree)",
         1, 19);
  return db;
}

NSE_SPG_ErrorEstimator::NSE_SPG_ErrorEstimator(const ParameterDatabase& param_db)
 : ErrorEstimator<2>(param_db), eta_res(0.), eta_div(0.), eta_F(0.),
   eta_delta(0.), eta_mu(0.), error_spg(0.), error_parts{{0.,0.,0.,0.,0.}},
   div_uh(0.), h_max(0.), delta_max(0.)
{
  auto local_db = default_spg_estimator_database();
  local_db.merge(param_db, false);
  delta0 = local_db["spg_delta0"];
  mu0 = local_db["spg_mu0"];
  sigma0 = local_db["spg_estimator_sigma0"];
  error_quad_degree = local_db["spg_error_quad_degree"];
  is_nse = local_db["problem_type"].is(5);
  equal_order = (TDatabase::ParamDB->VELOCITY_SPACE
                 == TDatabase::ParamDB->PRESSURE_SPACE);
  has_exact_solution = false;
  if(is_nse)
    sigma0 = 0.; // Section 5
}

double NSE_SPG_ErrorEstimator::get_effectivity_index() const
{
  if(!has_exact_solution || error_spg <= 0.)
    return 0.;
  return this->estimated_global_error / error_spg;
}

void NSE_SPG_ErrorEstimator::estimate(const Example_NSE2D& ex,
                                      const TFEVectFunct2D& u,
                                      const TFEFunction2D& p)
{
  const TCollection* coll = u.GetFESpace2D()->GetCollection();
  this->currentCollection = coll;
  const int n_cells = coll->GetN_Cells();
  this->eta_K.assign(n_cells, 0.);
  this->maximal_local_error = 0.;
  this->estimated_global_error = 0.;

  const TFESpace2D* fe_space_u = u.GetFESpace2D().get();
  const TFESpace2D* fe_space_p = p.GetFESpace2D().get();
  const double* values_u = u.GetValues();
  const double* values_p = p.GetValues();
  const int length_u = u.GetLength();
  auto u1 = u.GetComponent(0);
  auto u2 = u.GetComponent(1);

  const double nu = ex.get_nu();
  has_exact_solution = (ex.get_exact().size() >= 3 && ex.get_exact(0) != nullptr
                        && ex.get_exact(2) != nullptr);
  auto coeff_fct = ex.get_coeffs();
  if(!coeff_fct)
    ErrThrow("NSE_SPG_ErrorEstimator: the example has no coefficient function");
  // the index of each cell in the collection is needed to evaluate on neighbours
  coll->mark_all_cells();
  for(int i = 0; i < n_cells; ++i)
    coll->GetCell(i)->SetClipBoard(i);

  // ---------------------------------------------------------------------------
  // first pass: h = max h_K, max_K delta_K
  // ---------------------------------------------------------------------------
  h_max = 0.;
  delta_max = 0.;
  for(int i = 0; i < n_cells; ++i)
  {
    double hK = coll->GetCell(i)->GetDiameter();
    double dK, mK;
    // delta_K, mu_K with the cell size the assembly uses (CELL_MEASURE)
    compute_SPG_parameters(
      coll->GetCell(i)->Get_hK(TDatabase::ParamDB->CELL_MEASURE), nu, delta0,
      mu0, equal_order, dK, mK);
    h_max = std::max(h_max, hK);
    delta_max = std::max(delta_max, dK);
  }
  const int alpha = (nu < h_max && equal_order) ? 1 : 0;   // (23)
  // constants of the steady bound (Afzal, Ahmed, John 2026, Theorem 1) and
  // of the transient bound (adaptive SUPG/PSPG/grad-div for time-dependent
  // Navier-Stokes), respectively
  const double c_res = transient ? 320. : 40.;
  const double c_delta = transient ? 160. : 40.;
  const double c_div = transient ? 640. : 160.;
  const double c_F = transient ? 320. : 40.;
  const double c_mu = transient ? 320. : 40.;
  const double sigma_0 = transient ? 0. : sigma0;
  const double div_factor = equal_order ? c_div*delta_max
                                        : c_div*delta_max/(h_max*h_max);
  const double dim = 2.;
  const double* values_dt = (time_derivative != nullptr)
                            ? time_derivative->GetValues() : nullptr;
  if(time_derivative != nullptr
     && time_derivative->GetFESpace2D() != u.GetFESpace2D())
    ErrThrow("NSE_SPG_ErrorEstimator: the time derivative must live in the "
             "velocity space");
  std::vector<double> dt1(MaxN_QuadPoints_2D), dt2(MaxN_QuadPoints_2D);

  // ---------------------------------------------------------------------------
  // storage for values at quadrature points
  // ---------------------------------------------------------------------------
  std::vector<double> coeff_storage(MaxN_QuadPoints_2D * max_n_coeffs);
  std::vector<double*> coeffs(MaxN_QuadPoints_2D);
  for(int j = 0; j < MaxN_QuadPoints_2D; ++j)
    coeffs[j] = &coeff_storage[j * max_n_coeffs];
  // [quad point][derivative] for u1, u2, p
  std::vector<std::array<double, 5>> du1(MaxN_QuadPoints_2D), du2(MaxN_QuadPoints_2D);
  std::vector<std::array<double, 3>> dp(MaxN_QuadPoints_2D);
  std::vector<double> FEFunctValues;

  // 1D quadrature on the edges
  int degree_edges = 2 * std::max(fe_space_u->get_fe(0).GetBaseFunct()->GetPolynomialDegree(),
                                  fe_space_p->get_fe(0).GetBaseFunct()->GetPolynomialDegree()) + 2;
  const TQuadFormula* qf1d = QuadratureFormulaDatabase::qf_from_degree(
      degree_edges, BFRefElements::BFUnitLine);
  int n_points_1d;
  const double *weights_1d, *zeta_1d;
  qf1d->GetFormulaData(n_points_1d, weights_1d, zeta_1d);
  std::vector<double> X1d(n_points_1d), Y1d(n_points_1d);

  // high order quadrature on the reference triangle for the exact error
  int err_degree = error_quad_degree;
  const TQuadFormula* qf_err = QuadratureFormulaDatabase::qf_from_degree(
      err_degree, BFRefElements::BFUnitTriangle);
  int n_err_points;
  const double *w_err, *xi_err, *eta_err;
  qf_err->GetFormulaData(n_err_points, w_err, xi_err, eta_err);

  TQuadFormula qf_ref(QuadratureFormula_type::BaryCenterTria); // dummy type
  TQuadFormula qf_orig(qf_ref);
  bool needs_second_derivatives[2] = {true, false};

  double sum_res = 0., sum_div = 0., sum_F = 0., sum_delta = 0., sum_mu = 0.;
  double sum_div_uh = 0.;
  error_parts = {{0., 0., 0., 0., 0.}};

  // ---------------------------------------------------------------------------
  // second pass: local contributions
  // ---------------------------------------------------------------------------
  for(int cellIdx = 0; cellIdx < n_cells; ++cellIdx)
  {
    TBaseCell* cell = coll->GetCell(cellIdx);
    // h_K of the estimator: the diameter (theory); delta_K, mu_K: with the
    // cell size the assembly uses (TDatabase::ParamDB->CELL_MEASURE)
    const double hK = cell->GetDiameter();
    double deltaK, muK;
    compute_SPG_parameters(cell->Get_hK(TDatabase::ParamDB->CELL_MEASURE), nu,
                           delta0, mu0, equal_order, deltaK, muK);

    const FiniteElement& fe_u = fe_space_u->get_fe(cellIdx);
    const FiniteElement& fe_p = fe_space_p->get_fe(cellIdx);
    std::vector<const FiniteElement*> used_fe {&fe_u, &fe_p};
    // in case of equal order spaces the pressure element is a duplicate and
    // removed in GetOrig, the second derivatives are then taken from the first
    ReferenceTransformation_type ref_trans = FEDatabase::GetOrig(
      used_fe, coll, cell, needs_second_derivatives, qf_ref, qf_orig);
    (void) ref_trans;
    int N_Points;
    const double *weights, *X, *Y;
    qf_orig.GetFormulaData(N_Points, weights, X, Y);

    // velocity values and derivatives
    {
      const BaseFunctions* bf = fe_u.GetBaseFunct();
      int n_dof = fe_u.GetN_DOF();
      const int* DOF = fe_space_u->GetGlobalDOF(cellIdx);
      FEFunctValues.resize(2 * n_dof);
      for(int l = 0; l < n_dof; ++l)
      {
        FEFunctValues[l] = values_u[DOF[l]];
        FEFunctValues[l + n_dof] = values_u[DOF[l] + length_u];
      }
      for(size_t k = 0; k < derivatives_u.size(); ++k)
      {
        double** Orig = FEDatabase::GetOrigElementValues(*bf, derivatives_u[k]);
        for(int j = 0; j < N_Points; ++j)
        {
          double v1 = 0., v2 = 0.;
          for(int l = 0; l < n_dof; ++l)
          {
            v1 += FEFunctValues[l] * Orig[j][l];
            v2 += FEFunctValues[l + n_dof] * Orig[j][l];
          }
          du1[j][k] = v1;
          du2[j][k] = v2;
        }
      }
    }
    // time derivative of the velocity (transient Navier-Stokes)
    if(values_dt != nullptr)
    {
      const BaseFunctions* bf = fe_u.GetBaseFunct();
      int n_dof = fe_u.GetN_DOF();
      const int* DOF = fe_space_u->GetGlobalDOF(cellIdx);
      double** Orig = FEDatabase::GetOrigElementValues(*bf, MultiIndex2D::D00);
      for(int j = 0; j < N_Points; ++j)
      {
        double v1 = 0., v2 = 0.;
        for(int l = 0; l < n_dof; ++l)
        {
          v1 += values_dt[DOF[l]] * Orig[j][l];
          v2 += values_dt[DOF[l] + length_u] * Orig[j][l];
        }
        dt1[j] = v1;
        dt2[j] = v2;
      }
    }
    // pressure values and derivatives
    {
      const BaseFunctions* bf = fe_p.GetBaseFunct();
      int n_dof = fe_p.GetN_DOF();
      const int* DOF = fe_space_p->GetGlobalDOF(cellIdx);
      FEFunctValues.resize(n_dof);
      for(int l = 0; l < n_dof; ++l)
        FEFunctValues[l] = values_p[DOF[l]];
      for(size_t k = 0; k < derivatives_p.size(); ++k)
      {
        double** Orig = FEDatabase::GetOrigElementValues(*bf, derivatives_p[k]);
        for(int j = 0; j < N_Points; ++j)
        {
          double v = 0.;
          for(int l = 0; l < n_dof; ++l)
            v += FEFunctValues[l] * Orig[j][l];
          dp[j][k] = v;
        }
      }
    }
    // coefficients
    coeff_fct(N_Points, X, Y, nullptr, coeffs.data());

    // -------------------------------------------------------------------------
    // mesh cell terms
    // -------------------------------------------------------------------------
    double res_K = 0.;    // ||r_K||^2_K
    double divu_K = 0.;   // ||div u_h||^2_K
    double err_grad = 0., err_sigma = 0., err_div = 0., err_supg = 0., err_p = 0.;
    for(int j = 0; j < N_Points; ++j)
    {
      const double w = weights[j];
      const double* c = coeffs[j];
      double f1 = c[1], f2 = c[2];
      if(values_dt != nullptr)
      {
        f1 -= dt1[j];
        f2 -= dt2[j];
      }
      double b1, b2, sigma;
      if(is_nse)
      {
        b1 = du1[j][0];
        b2 = du2[j][0];
        sigma = 0.;
      }
      else
      {
        b1 = c[4];
        b2 = c[5];
        sigma = c[6];
      }
      // r_K = f + nu Lap u_h - (b.grad)u_h - sigma u_h - grad p_h, (18)/(40)
      double r1 = f1 + nu*(du1[j][3] + du1[j][4]) - (b1*du1[j][1] + b2*du1[j][2])
                  - sigma*du1[j][0] - dp[j][1];
      double r2 = f2 + nu*(du2[j][3] + du2[j][4]) - (b1*du2[j][1] + b2*du2[j][2])
                  - sigma*du2[j][0] - dp[j][2];
      res_K += w * (r1*r1 + r2*r2);
      double divu = du1[j][1] + du2[j][2];
      divu_K += w * divu*divu;

    }

    // -------------------------------------------------------------------------
    // exact error on this cell with a high order quadrature (independent of
    // the FE quadrature; layers need many points)
    // -------------------------------------------------------------------------
    if(has_exact_solution)
    {
      double x0 = cell->GetVertex(0)->GetX(), y0 = cell->GetVertex(0)->GetY();
      double x1 = cell->GetVertex(1)->GetX(), y1 = cell->GetVertex(1)->GetY();
      double x2 = cell->GetVertex(2)->GetX(), y2 = cell->GetVertex(2)->GetY();
      if(cell->GetN_Vertices() != 3)
        ErrThrow("NSE_SPG_ErrorEstimator: exact errors only on triangles");
      double detJ = std::abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0));
      for(int j = 0; j < n_err_points; ++j)
      {
        double xq = x0 + xi_err[j]*(x1-x0) + eta_err[j]*(x2-x0);
        double yq = y0 + xi_err[j]*(y1-y0) + eta_err[j]*(y2-y0);
        double w = w_err[j]*detJ;
        double v1[3], v2[3], vp[3];
        u1->FindGradientLocal(cell, cellIdx, xq, yq, v1);
        u2->FindGradientLocal(cell, cellIdx, xq, yq, v2);
        p.FindGradientLocal(cell, cellIdx, xq, yq, vp);
        double e1[10], e2[10], ep[10];
        ex.get_exact(0)(xq, yq, e1);
        ex.get_exact(1)(xq, yq, e2);
        ex.get_exact(2)(xq, yq, ep);
        double b1, b2, sigma;
        if(is_nse)
        {
          b1 = v1[0]; b2 = v2[0]; sigma = 0.;
        }
        else
        {
          double cq[max_n_coeffs];
          double* cqp[1] = {cq};
          coeff_fct(1, &xq, &yq, nullptr, cqp);
          b1 = cq[4]; b2 = cq[5]; sigma = cq[6];
        }
        double eu1 = e1[0] - v1[0], eu1x = e1[1] - v1[1], eu1y = e1[2] - v1[2];
        double eu2 = e2[0] - v2[0], eu2x = e2[1] - v2[1], eu2y = e2[2] - v2[2];
        double epv = ep[0] - vp[0], epx = ep[1] - vp[1], epy = ep[2] - vp[2];
        err_grad += w * (eu1x*eu1x + eu1y*eu1y + eu2x*eu2x + eu2y*eu2y);
        err_sigma += w * sigma * (eu1*eu1 + eu2*eu2);
        err_div += w * (eu1x + eu2y)*(eu1x + eu2y);
        err_p += w * epv*epv;
        if(is_nse && !transient)
          err_supg += w * (epx*epx + epy*epy);
        else
        {
          double s1 = b1*eu1x + b2*eu1y + epx;
          double s2 = b1*eu2x + b2*eu2y + epy;
          err_supg += w * (s1*s1 + s2*s2);
        }
      }
    }

    // -------------------------------------------------------------------------
    // facet terms
    // -------------------------------------------------------------------------
    double etaF_K = 0.; // sum of (weighted) eta_F^2 of the facets of this cell
    const int n_edges = cell->GetN_Edges();
    const int* edge_vertex;
    cell->GetRefDesc()->GetShapeDesc()->GetEdgeVertex(edge_vertex);
    for(int e = 0; e < n_edges; ++e)
    {
      const TJoint* joint = cell->GetJoint(e);
      const TVertex* v0 = cell->GetVertex(edge_vertex[2*e]);
      const TVertex* v1 = cell->GetVertex(edge_vertex[2*e+1]);
      const double x0 = v0->GetX(), y0 = v0->GetY();
      const double x1 = v1->GetX(), y1 = v1->GetY();
      // outer normal (vertices are ordered counterclockwise)
      double nx = y1 - y0, ny = x0 - x1;
      const double hE = std::sqrt(nx*nx + ny*ny);
      nx /= hE;
      ny /= hE;

      bool is_boundary = (joint->GetType() == BoundaryEdge
                          || joint->GetType() == IsoBoundEdge);
      const TBaseCell* neigh = nullptr;
      int neigh_idx = -1;
      if(is_boundary)
      {
        const TBoundEdge* bd_edge = static_cast<const TBoundEdge*>(joint);
        double t0, t1;
        bd_edge->GetParameters(t0, t1);
        int comp = bd_edge->GetBoundComp()->GetID();
        BoundCond cond;
        ex.get_bc(0)(comp, 0.5*(t0+t1), cond);
        if(cond == DIRICHLET)
          continue; // r_F = 0 on Dirichlet facets, (19)
        if(cond != NEUMANN)
          ErrThrow("NSE_SPG_ErrorEstimator: unsupported boundary condition");
        // Neumann: r_F = g - (nu grad u_h - p_h I) n, only g = 0 is supported
      }
      else
      {
        neigh = joint->GetNeighbour(cell);
        if(neigh != nullptr && neigh->ExistChildren())
        {
          // the joint is attached to an ancestor of the true neighbour
          parmoon::Point q(0.5*(x0+x1) + 1e-3*hE*nx, 0.5*(y0+y1) + 1e-3*hE*ny);
          neigh = leaf_containing(neigh, q);
        }
        if(neigh == nullptr)
        {
          // the joint still belongs to a coarser ancestor (bisection during the
          // conforming closure keeps whole edges), find the leaf neighbour
          neigh = find_leaf_neighbour(cell, 0.5*(x0+x1), 0.5*(y0+y1), nx, ny, hE);
        }
        if(neigh == nullptr)
          ErrThrow("NSE_SPG_ErrorEstimator: non-conforming mesh (hanging node) "
                   "found, use 'conforming_closure: true'");
        neigh_idx = neigh->GetClipBoard();
        if(neigh_idx < 0)
          ErrThrow("NSE_SPG_ErrorEstimator: neighbour cell not in collection");
      }

      // quadrature points on the edge
      for(int k = 0; k < n_points_1d; ++k)
      {
        double s = 0.5*(zeta_1d[k] + 1.);
        X1d[k] = x0 + s*(x1 - x0);
        Y1d[k] = y0 + s*(y1 - y0);
      }
      coeff_fct(n_points_1d, X1d.data(), Y1d.data(), nullptr, coeffs.data());

      double rF = 0.;          // ||r_F||^2_F
      double b_max_F = 0.;     // ||b||_{L^inf(F)}
      for(int k = 0; k < n_points_1d; ++k)
      {
        double vu1[3], vu2[3], vp[3];
        u1->FindGradientLocal(cell, cellIdx, X1d[k], Y1d[k], vu1);
        u2->FindGradientLocal(cell, cellIdx, X1d[k], Y1d[k], vu2);
        p.FindGradientLocal(cell, cellIdx, X1d[k], Y1d[k], vp);
        // (-nu grad u_h + p_h I) n, components 1 and 2
        double t1 = -nu*(vu1[1]*nx + vu1[2]*ny) + vp[0]*nx;
        double t2 = -nu*(vu2[1]*nx + vu2[2]*ny) + vp[0]*ny;
        double j1, j2;
        if(is_boundary)
        {
          // g - (nu grad u_h - p_h I) n with g = 0
          j1 = t1;
          j2 = t2;
        }
        else
        {
          double nu1[3], nu2[3], np[3];
          u1->FindGradientLocal(neigh, neigh_idx, X1d[k], Y1d[k], nu1);
          u2->FindGradientLocal(neigh, neigh_idx, X1d[k], Y1d[k], nu2);
          p.FindGradientLocal(neigh, neigh_idx, X1d[k], Y1d[k], np);
          double s1 = -nu*(nu1[1]*nx + nu1[2]*ny) + np[0]*nx;
          double s2 = -nu*(nu2[1]*nx + nu2[2]*ny) + np[0]*ny;
          j1 = t1 - s1;
          j2 = t2 - s2;
        }
        double w = 0.5*hE*weights_1d[k];
        rF += w * (j1*j1 + j2*j2);
        double bb1, bb2;
        if(is_nse)
        {
          bb1 = vu1[0];
          bb2 = vu2[0];
        }
        else
        {
          bb1 = coeffs[k][4];
          bb2 = coeffs[k][5];
        }
        b_max_F = std::max(b_max_F, std::sqrt(bb1*bb1 + bb2*bb2));
      }
      // weight of eta_F^2, Theorem 1 (C = 1)
      double wF;
      if(transient)
      {
        double wb = (b_max_F > 0.) ? c_F/(b_max_F*b_max_F) : 0.;
        wF = min_positive(hE/nu, wb);
      }
      else
      {
        double w1 = min_positive(hK/nu, (sigma_0 > 0.) ? 1./(hK*sigma_0) : 0.);
        double w2 = min_positive(hK/nu, (sigma_0 > 0.) ? 1./std::sqrt(nu*sigma_0) : 0.);
        double wb = (b_max_F > 0.) ? c_F/(b_max_F*b_max_F) : 0.;
        wF = min_positive(w1 + w2, wb);
      }
      double etaF2 = wF * rF;
      // interior facets are shared by two cells
      etaF_K += is_boundary ? etaF2 : 0.5*etaF2;
    }

    // -------------------------------------------------------------------------
    // local estimator contributions, Theorem 1 with C = 1
    // -------------------------------------------------------------------------
    double w_res = min_positive(min_positive((sigma_0 > 0.) ? 1./sigma_0 : 0.,
                                             hK*hK/nu), c_res*deltaK);
    double eta_res2 = w_res * res_K;
    double eta_div2 = div_factor * divu_K;
    double eta_delta2 = c_delta * deltaK * res_K;
    double w_mu = min_positive(c_mu*dim*muK*muK / ((alpha == 1) ? deltaK : 1.),
                               dim*muK*muK/nu);
    double eta_mu2 = w_mu * divu_K;

    sum_res += eta_res2;
    sum_div += eta_div2;
    sum_F += etaF_K;
    sum_delta += eta_delta2;
    sum_mu += eta_mu2;
    sum_div_uh += divu_K;

    double eta_K2 = eta_res2 + eta_div2 + etaF_K + eta_delta2 + eta_mu2;
    this->eta_K[cellIdx] = std::sqrt(eta_K2);
    this->maximal_local_error = std::max(this->maximal_local_error,
                                         this->eta_K[cellIdx]);

    error_parts[0] += nu * err_grad;
    error_parts[1] += err_sigma;
    error_parts[2] += muK * err_div;
    error_parts[3] += deltaK * err_supg;
    error_parts[4] += err_p;
  }

  eta_res = std::sqrt(sum_res);
  eta_div = std::sqrt(sum_div);
  eta_F = std::sqrt(sum_F);
  eta_delta = std::sqrt(sum_delta);
  eta_mu = std::sqrt(sum_mu);
  div_uh = std::sqrt(sum_div_uh);
  this->estimated_global_error = std::sqrt(sum_res + sum_div + sum_F + sum_delta
                                           + sum_mu);
  if(has_exact_solution)
  {
    double e2 = error_parts[0] + error_parts[1] + error_parts[2] + error_parts[3];
    if(is_nse && !transient)
      e2 += nu * error_parts[4]; // nu ||q||^2 in ||.||_{spg,nse}
    error_spg = std::sqrt(e2);
  }
  else
    error_spg = 0.;
}

void NSE_SPG_ErrorEstimator::info()
{
  Output::print("SUPG/PSPG/grad-div residual estimator (Afzal, Ahmed, John 2026)");
  Output::print("  pairs: ", equal_order ? "equal order" : "inf-sup stable",
                ", problem: ", is_nse ? (transient ? "transient Navier-Stokes"
                                                   : "Navier-Stokes") : "Oseen",
                ", h = ", h_max, ", max delta_K = ", delta_max);
  Output::print("  eta        = ", std::setprecision(10), this->estimated_global_error);
  Output::print("  eta_K,res  = ", eta_res);
  Output::print("  eta_K,div  = ", eta_div);
  Output::print("  eta_F      = ", eta_F);
  Output::print("  eta_K,delta= ", eta_delta);
  Output::print("  eta_K,mu   = ", eta_mu);
  Output::print("  ||div u_h|| = ", div_uh);
  if(has_exact_solution)
  {
    Output::print("  error_spg  = ", error_spg,
                  "  (nu|grad e|^2: ", error_parts[0], ", |sigma^1/2 e|^2: ",
                  error_parts[1], ", mu|div e|^2: ", error_parts[2],
                  ", delta|supg|^2: ", error_parts[3], ", |e_p|^2: ",
                  error_parts[4], ")");
    Output::print("  effectivity index = ", get_effectivity_index());
  }
  Output::print("  max local eta_K = ", this->maximal_local_error);
}
#endif // __2D__
