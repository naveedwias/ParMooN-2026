// Algebraic stabilizations of P1 convection-diffusion-reaction problems,
// separate from AlgebraicFluxCorrection; see AlgebraicStabilization.h.
#include "AlgebraicStabilization.h"

#include <algorithm>
#include <cmath>

#include "BaseCell.h"
#include "Collection.h"
#include "FEMatrix.h"
#include "FESpace2D.h"
#include "Joint.h"
#include "MooNMD_Io.h"

namespace AlgebraicStabilization
{

ParameterDatabase default_database()
{
  ParameterDatabase db("algebraic stabilization (AlgebraicStabilization)");
  db.add("algstab_method", "none",
         "algebraic stabilization of the separate module AlgebraicStabilization "
         "(replaces afc_limiter if not 'none'; fixed point rhs only): "
         "kuzmin_edge (AFC, Kuzmin limiter, one owner per edge), bbk "
         "(b_ij = d_ij max{xi_i, xi_j}^p), sbbk (b_ij = -C0 |E| max{xi}^p), "
         "smuas (Knobloch 2023)",
         {"none", "kuzmin_edge", "bbk", "sbbk", "smuas"});
  db.add("algstab_tie_tol", 1.e-10, "kuzmin_edge: a_ij and a_ji with relative "
         "difference at most this are a tie (owner: the lexicographically "
         "smaller node)", 0., 1.);
  db.add("algstab_bbk_p", 10.0, "exponent p of the BBK-type methods", 0., 1.e3);
  db.add("algstab_bbk_C0", 3.0, "constant C0 of the edge weight C0 |E| (sbbk)",
         0., 1.e10);
  db.add("algstab_smuas_weights", "67", "weights of SMUAS: (67) p_ij = "
         "max{a_ij, 0, a_ji}, q_ij = max{|a_ij|, a_ji}; (68) p_ij = q_ij = 1",
         {"67", "68"});
  return db;
}

bool active(const ParameterDatabase& db)
{
  return db.contains("algstab_method") && !db["algstab_method"].is("none");
}

namespace
{
// min{1, Q / P}, 1 where P = 0, clipped to [0, 1] (galerkit _ratio)
inline double ratio(double Q, double P)
{
  if(P == 0.)
    return 1.;
  return std::min(1., std::max(0., std::min(1., Q / P)));
}
inline double pos(double x) { return x > 0. ? x : 0.; }
inline double neg(double x) { return x < 0. ? x : 0.; }
}

Operator::Operator(const FEMatrix& A, const ParameterDatabase& db)
 : method(db["algstab_method"].get<std::string>())
{
  if(!A.is_square())
    ErrThrow("AlgebraicStabilization: the matrix must be square");
  N = A.get_n_rows();
  n_free = A.GetFESpace2D()->get_n_active();
  const int* row = A.get_row_ptr();
  const int* col = A.get_vector_columns();
  const double* a = A.GetEntries();
  const TStructure& st = A.GetStructure();
  // undirected edges i < j of the sparsity pattern (P1: the mesh edges)
  for(int i = 0; i < N; ++i)
    for(int k = row[i]; k < row[i+1]; ++k)
    {
      int j = col[k];
      if(j <= i)
        continue;
      int kt = st.get_index_of_entry(j, i);
      if(kt < 0)
        ErrThrow("AlgebraicStabilization: the sparsity pattern is not symmetric");
      I.push_back(i);
      J.push_back(j);
      aij.push_back(a[k]);
      aji.push_back(a[kt]);
      d.push_back(-std::max(std::max(a[k], a[kt]), 0.));
    }
  setup_geometry(A);
  p_bbk = db["algstab_bbk_p"];
  if(method == "kuzmin_edge")
  {
    // a_ij and a_ji that agree up to round-off (relative difference at most
    // algstab_tie_tol) count as a tie: an exact comparison would let the
    // round-off of the assembly choose the owner (crossed meshes have such edges)
    const double tie_tol = db["algstab_tie_tol"];
    owner_is_I.resize(I.size());
    for(size_t e = 0; e < I.size(); ++e)
    {
      const double scale = std::max(std::abs(aij[e]), std::abs(aji[e]));
      if(std::abs(aij[e] - aji[e]) > tie_tol*scale)
        owner_is_I[e] = (aji[e] < aij[e]);    // the upwind node owns the edge
      else                                     // tie: lexicographically smaller
        owner_is_I[e] = (X[I[e]] < X[J[e]]
                         || (X[I[e]] == X[J[e]] && Y[I[e]] < Y[J[e]]));
    }
  }
  else if(method == "sbbk")
  {
    const double C0 = db["algstab_bbk_C0"];
    w_edge.resize(I.size());
    for(size_t e = 0; e < I.size(); ++e)
      w_edge[e] = C0 * std::hypot(X[J[e]] - X[I[e]], Y[J[e]] - Y[I[e]]);
  }
  else if(method == "smuas")
    setup_smuas(A, db["algstab_smuas_weights"].get<std::string>());
  else if(method != "bbk")
    ErrThrow("AlgebraicStabilization: unknown method ", method);
  Output::print<2>("AlgebraicStabilization: ", method, ", ", I.size(), " edges, ",
                   n_free, " free of ", N, " nodes");
}

void Operator::setup_geometry(const FEMatrix& A)
{
  auto space = A.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  X.assign(N, 0.);
  Y.assign(N, 0.);
  on_boundary.assign(N, 0);
  std::vector<char> seen(N, 0);
  for(int c = 0; c < coll->GetN_Cells(); ++c)
  {
    const TBaseCell* cell = coll->GetCell(c);
    if(cell->GetN_Vertices() != 3 || space->get_n_local_dof(c) != 3)
      ErrThrow("AlgebraicStabilization: P1 elements on triangles only");
    const int* dof = space->GetGlobalDOF(c);
    for(int k = 0; k < 3; ++k)                 // P1: local dof k at vertex k
    {
      X[dof[k]] = cell->GetVertex(k)->GetX();
      Y[dof[k]] = cell->GetVertex(k)->GetY();
      seen[dof[k]] = 1;
    }
    const int* ev;
    cell->GetShapeDesc()->GetEdgeVertex(ev);
    for(int k = 0; k < 3; ++k)
    {
      const TJoint* jt = cell->GetJoint(k);
      if(jt->GetType() == BoundaryEdge || jt->GetType() == IsoBoundEdge)
      {
        on_boundary[dof[ev[2*k]]] = 1;
        on_boundary[dof[ev[2*k+1]]] = 1;
      }
    }
  }
  for(int i = 0; i < N; ++i)
    if(!seen[i])
      ErrThrow("AlgebraicStabilization: dof ", i, " without a vertex");
}

void Operator::setup_smuas(const FEMatrix& A, const std::string& weights)
{
  // directed edges s -> l (both directions of every edge) with the weights
  const size_t ne = I.size();
  s.resize(2*ne);
  l.resize(2*ne);
  p_w.resize(2*ne);
  q_w.resize(2*ne);
  for(size_t e = 0; e < ne; ++e)
  {
    for(int dir = 0; dir < 2; ++dir)
    {
      size_t k = e + dir*ne;
      s[k] = dir == 0 ? I[e] : J[e];
      l[k] = dir == 0 ? J[e] : I[e];
      double a_sl = dir == 0 ? aij[e] : aji[e];
      double a_ls = dir == 0 ? aji[e] : aij[e];
      bool inP = (a_sl > 0.) || (a_ls > 0.);
      if(weights == "67")
      {
        p_w[k] = inP ? std::max(std::max(a_sl, 0.), a_ls) : 0.;
        q_w[k] = std::max(std::abs(a_sl), a_ls);
      }
      else
      {
        p_w[k] = inP ? 1. : 0.;
        q_w[k] = 1.;
      }
    }
  }
  // cells of the mesh (dofs) and the cells of every vertex
  auto space = A.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  const int n_cells = coll->GetN_Cells();
  std::vector<int> cell_dof(3*n_cells);
  std::vector<std::vector<int>> cells_of(N);
  for(int c = 0; c < n_cells; ++c)
  {
    const int* dof = space->GetGlobalDOF(c);
    for(int k = 0; k < 3; ++k)
    {
      cell_dof[3*c+k] = dof[k];
      cells_of[dof[k]].push_back(c);
    }
  }
  // reflection rows (Knobloch 2023, (60)): (W u)_e = u_s + grad u_h|_T . (x_s - x_l),
  // T the cell of the patch of x_s with grad lambda_k . (x_s - x_l) >= 0 for its
  // vertices x_k != x_s (the half line enters T); u_sl = u_s if there is none
  const double tol = 1e-12;
  W_ptr.assign(1, 0);
  for(size_t e = 0; e < s.size(); ++e)
  {
    const int i = s[e];
    const double vx = X[i] - X[l[e]], vy = Y[i] - Y[l[e]];
    const double vn = std::hypot(vx, vy);
    bool found = false;
    for(int c : cells_of[i])
    {
      const int* v = &cell_dof[3*c];
      const double x0 = X[v[0]], y0 = Y[v[0]], x1 = X[v[1]], y1 = Y[v[1]],
                   x2 = X[v[2]], y2 = Y[v[2]];
      const double det = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
      const double gx[3] = {(y1 - y2)/det, (y2 - y0)/det, (y0 - y1)/det};
      const double gy[3] = {(x2 - x1)/det, (x0 - x2)/det, (x1 - x0)/det};
      double gv[3];
      bool ok = true;
      for(int k = 0; k < 3; ++k)
      {
        gv[k] = gx[k]*vx + gy[k]*vy;
        const double scale = vn*std::hypot(gx[k], gy[k]);
        if(v[k] != i && gv[k] < -tol*scale)
          ok = false;
      }
      if(ok)
      {
        // e_s + sum_k (grad lambda_k . v) e_k
        W_col.push_back(i);
        W_val.push_back(1.);
        for(int k = 0; k < 3; ++k)
        {
          W_col.push_back(v[k]);
          W_val.push_back(gv[k]);
        }
        found = true;
        break;
      }
    }
    if(!found)
    {
      W_col.push_back(i);
      W_val.push_back(1.);
    }
    W_ptr.push_back((int) W_col.size());
  }
}

std::vector<double> Operator::b_edges(const std::vector<double>& u) const
{
  if(method == "kuzmin_edge")
    return b_kuzmin(u);
  if(method == "bbk")
    return b_bbk(u);
  if(method == "sbbk")
    return b_sbbk(u);
  return b_smuas(u);
}

std::vector<double> Operator::b_kuzmin(const std::vector<double>& u) const
{
  const size_t ne = I.size();
  std::vector<double> Pp(N, 0.), Pm(N, 0.), Qp(N, 0.), Qm(N, 0.);
  std::vector<double> f(ne);
  for(size_t e = 0; e < ne; ++e)
  {
    const int i = I[e], j = J[e];
    f[e] = d[e]*(u[j] - u[i]);                 // f_ij; f_ji = -f_ij
    const double fp = pos(f[e]), fm = neg(f[e]);
    Qp[i] -= fm;    Qm[i] -= fp;               // Q_i^+- = -sum_j f_ij^-+
    Qp[j] += fp;    Qm[j] += fm;
    if(owner_is_I[e])
    {
      Pp[i] += fp;  Pm[i] += fm;
    }
    else
    {
      Pp[j] -= fm;  Pm[j] -= fp;               // f_ji^+ = -f_ij^-, f_ji^- = -f_ij^+
    }
  }
  std::vector<double> Rp(N, 1.), Rm(N, 1.);
  for(int i = 0; i < n_free; ++i)
  {
    Rp[i] = ratio(Qp[i], Pp[i]);
    Rm[i] = ratio(Qm[i], Pm[i]);
  }
  std::vector<double> b(ne);
  for(size_t e = 0; e < ne; ++e)
  {
    const int o = owner_is_I[e] ? I[e] : J[e];
    const double fo = owner_is_I[e] ? f[e] : -f[e];
    const double alpha = fo > 0. ? Rp[o] : (fo < 0. ? Rm[o] : 1.);
    b[e] = (1. - alpha)*d[e];
  }
  return b;
}

std::vector<double> Operator::b_bbk(const std::vector<double>& u) const
{
  const size_t ne = I.size();
  std::vector<double> diff(N, 0.), adiff(N, 0.);
  for(size_t e = 0; e < ne; ++e)
  {
    const double du = u[I[e]] - u[J[e]];
    diff[I[e]] += du;   diff[J[e]] -= du;
    adiff[I[e]] += std::abs(du);   adiff[J[e]] += std::abs(du);
  }
  std::vector<double> xi(N, 0.);
  for(int i = 0; i < n_free; ++i)
    xi[i] = adiff[i] > 0. ? std::abs(diff[i])/adiff[i] : 0.;
  std::vector<double> b(ne);
  for(size_t e = 0; e < ne; ++e)
    b[e] = d[e]*std::pow(std::max(xi[I[e]], xi[J[e]]), p_bbk);
  return b;
}

std::vector<double> Operator::b_sbbk(const std::vector<double>& u) const
{
  const size_t ne = I.size();
  std::vector<double> num(N, 0.), den(N, 0.);
  for(size_t e = 0; e < ne; ++e)
  {
    const double du = u[I[e]] - u[J[e]];
    num[I[e]] += du;   num[J[e]] -= du;
    den[I[e]] += std::abs(du);   den[J[e]] += std::abs(du);
  }
  std::vector<double> xi(N, 0.);
  for(int i = 0; i < n_free; ++i)               // 0 on the boundary of the domain
    if(!on_boundary[i] && den[i] > 0.)
      xi[i] = std::abs(num[i])/den[i];
  std::vector<double> b(ne);
  for(size_t e = 0; e < ne; ++e)
    b[e] = -w_edge[e]*std::pow(std::max(xi[I[e]], xi[J[e]]), p_bbk);
  return b;
}

std::vector<double> Operator::b_smuas(const std::vector<double>& u) const
{
  const size_t nd = s.size(), ne = I.size();
  std::vector<double> Pp(N, 0.), Pm(N, 0.), Qp(N, 0.), Qm(N, 0.), ds(nd);
  for(size_t k = 0; k < nd; ++k)
  {
    ds[k] = u[s[k]] - u[l[k]];                 // u_i - u_j
    double wu = 0.;                            // reflected value u_ij
    for(int m = W_ptr[k]; m < W_ptr[k+1]; ++m)
      wu += W_val[m]*u[W_col[m]];
    const double r = wu - u[s[k]];             // u_ij - u_i
    const int i = s[k];
    Pp[i] += p_w[k]*(pos(ds[k]) + pos(-r));
    Pm[i] += p_w[k]*(neg(ds[k]) + neg(-r));
    Qp[i] += q_w[k]*(pos(-ds[k]) + pos(r));
    Qm[i] += q_w[k]*(neg(-ds[k]) + neg(r));
  }
  std::vector<double> Rp(N, 1.), Rm(N, 1.);
  for(int i = 0; i < n_free; ++i)
  {
    Rp[i] = ratio(Qp[i], Pp[i]);
    Rm[i] = ratio(Qm[i], Pm[i]);
  }
  std::vector<double> beta(nd);
  for(size_t k = 0; k < nd; ++k)
    beta[k] = ds[k] > 0. ? 1. - Rp[s[k]] : (ds[k] < 0. ? 1. - Rm[s[k]] : 0.);
  std::vector<double> b(ne);
  for(size_t e = 0; e < ne; ++e)
    b[e] = -std::max(std::max(beta[e]*aij[e], 0.), beta[e + ne]*aji[e]);
  return b;
}

void Operator::add_D(FEMatrix& M) const
{
  const TStructure& st = M.GetStructure();
  double* m = M.GetEntries();
  for(size_t e = 0; e < I.size(); ++e)
  {
    const int i = I[e], j = J[e];
    m[st.get_index_of_entry(i, j)] += d[e];
    m[st.get_index_of_entry(j, i)] += d[e];
    m[st.get_index_of_entry(i, i)] -= d[e];
    m[st.get_index_of_entry(j, j)] -= d[e];
  }
}

void Operator::add_rhs_correction(const std::vector<double>& u,
                                  std::vector<double>& rhs) const
{
  const std::vector<double> b = b_edges(u);
  for(size_t e = 0; e < I.size(); ++e)
  {
    const int i = I[e], j = J[e];
    const double w = d[e] - b[e];              // (D - B)_ij
    if(is_free(i))
      rhs[i] += w*(u[j] - u[i]);
    if(is_free(j))
      rhs[j] += w*(u[i] - u[j]);
  }
}

} // namespace AlgebraicStabilization
