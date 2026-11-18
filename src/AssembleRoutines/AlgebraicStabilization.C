// Algebraic stabilizations of P1 convection-diffusion-reaction problems,
// separate from AlgebraicFluxCorrection; see AlgebraicStabilization.h.
#include "AlgebraicStabilization.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

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
         {"none", "kuzmin_edge", "bbk", "fbbk", "sbbk", "smuas"});
  db.add("algstab_write_nodal", "", "file for x, y, u, free of every dof at the "
         "end (17 digits); empty: none");
  db.add("algstab_write_edges", "", "file for the edges i, j, x_i, y_i, x_j, y_j, "
         "a_ij, a_ji of the Galerkin matrix (first AFC step); empty: none");
  db.add("algstab_tie_tol", 1.e-10, "kuzmin_edge: a_ij and a_ji with relative "
         "difference at most this are a tie (owner: the lexicographically "
         "smaller node)", 0., 1.);
  db.add("algstab_bbk_p", 10.0, "exponent p of the BBK-type methods", 0., 1.e3);
  db.add("algstab_bbk_C0", 3.0, "constant C0 of the edge weight C0 |E| (sbbk)",
         0., 1.e10);
  db.add("algstab_bbk_beta0", 0.1, "lower bound beta0 of the weights beta_ij "
         "of fbbk", 0., 1.);
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

namespace
{
void dof_coordinates(const FEMatrix& A, std::vector<double>& X, std::vector<double>& Y)
{
  auto space = A.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  const int N = A.get_n_rows();
  X.assign(N, 0.);
  Y.assign(N, 0.);
  for(int c = 0; c < coll->GetN_Cells(); ++c)
  {
    const TBaseCell* cell = coll->GetCell(c);
    if(cell->GetN_Vertices() != 3 || space->get_n_local_dof(c) != 3)
      ErrThrow("AlgebraicStabilization: P1 elements on triangles only");
    const int* dof = space->GetGlobalDOF(c);
    for(int k = 0; k < 3; ++k)
    {
      X[dof[k]] = cell->GetVertex(k)->GetX();
      Y[dof[k]] = cell->GetVertex(k)->GetY();
    }
  }
}
}

void write_nodal_values(const FEMatrix& A, const std::vector<double>& u,
                        const std::string& path)
{
  std::vector<double> X, Y;
  dof_coordinates(A, X, Y);
  const int n_free = A.GetFESpace2D()->get_n_active();
  std::ofstream f(path);
  f << "# x y u free\n";
  f.precision(17);
  for(size_t i = 0; i < X.size(); ++i)
    f << X[i] << " " << Y[i] << " " << u[i] << " " << ((int) i < n_free ? 1 : 0) << "\n";
  Output::print<1>("AlgebraicStabilization: nodal values written to ", path);
}

void write_edges(const FEMatrix& A, const std::string& path)
{
  std::vector<double> X, Y;
  dof_coordinates(A, X, Y);
  const int* row = A.get_row_ptr();
  const int* col = A.get_vector_columns();
  const double* a = A.GetEntries();
  const TStructure& st = A.GetStructure();
  std::ofstream f(path);
  f << "# i j x_i y_i x_j y_j a_ij a_ji\n";
  f.precision(17);
  for(int i = 0; i < A.get_n_rows(); ++i)
    for(int k = row[i]; k < row[i+1]; ++k)
    {
      const int j = col[k];
      if(j <= i)
        continue;
      f << i << " " << j << " " << X[i] << " " << Y[i] << " " << X[j] << " " << Y[j]
        << " " << a[k] << " " << a[st.get_index_of_entry(j, i)] << "\n";
    }
  Output::print<1>("AlgebraicStabilization: edges written to ", path);
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
  else if(method == "sbbk" || method == "fbbk")
  {
    const double C0 = db["algstab_bbk_C0"];
    w_edge.resize(I.size());
    for(size_t e = 0; e < I.size(); ++e)
      w_edge[e] = C0 * std::hypot(X[J[e]] - X[I[e]], Y[J[e]] - Y[I[e]]);
    beta_dir.assign(2*I.size(), 1.);
    if(method == "fbbk")
      setup_fbbk(db["algstab_bbk_beta0"]);
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

namespace
{
// pseudo-inverse solve of the symmetric 2x2 system M x = r (galerkit: lstsq)
void solve2_pinv(const double M[2][2], const double r[2], double x[2])
{
  const double a = M[0][0], b = M[0][1], dd = M[1][1];
  const double tr = a + dd, det = a*dd - b*b;
  const double disc = std::sqrt(std::max(0., 0.25*(a - dd)*(a - dd) + b*b));
  const double l1 = 0.5*tr + disc, l2 = 0.5*tr - disc;     // l1 >= l2
  const double cut = std::numeric_limits<double>::epsilon()*2.*std::abs(l1);
  // eigenvectors
  double v1[2], v2[2];
  if(std::abs(b) > 0.)
  {
    v1[0] = l1 - dd; v1[1] = b;
  }
  else if(a >= dd)
  {
    v1[0] = 1.; v1[1] = 0.;
  }
  else
  {
    v1[0] = 0.; v1[1] = 1.;
  }
  const double n1 = std::hypot(v1[0], v1[1]);
  v1[0] /= n1; v1[1] /= n1;
  v2[0] = -v1[1]; v2[1] = v1[0];
  x[0] = x[1] = 0.;
  if(std::abs(l1) > cut)
  {
    const double c1 = (v1[0]*r[0] + v1[1]*r[1])/l1;
    x[0] += c1*v1[0]; x[1] += c1*v1[1];
  }
  if(std::abs(l2) > cut)
  {
    const double c2 = (v2[0]*r[0] + v2[1]*r[1])/l2;
    x[0] += c2*v2[0]; x[1] += c2*v2[1];
  }
  (void) det;
}

// argmin |x - 1|^2 subject to T x = 0, x >= beta0 (T: 2 x k, rows scaled to
// unit maximum); primal active set as galerkit _ldp, dual semismooth Newton as
// fallback (galerkit _ldp_dual)
std::vector<double> ldp(std::vector<double> T0, std::vector<double> T1, double beta0)
{
  const size_t k = T0.size();
  double r0 = 0., r1 = 0.;
  for(size_t j = 0; j < k; ++j)
  {
    r0 = std::max(r0, std::abs(T0[j]));
    r1 = std::max(r1, std::abs(T1[j]));
  }
  int rows = 0;
  std::vector<double> T[2];
  if(r0 > 0.) { for(auto& t : T0) t /= r0; T[rows++] = T0; }
  if(r1 > 0.) { for(auto& t : T1) t /= r1; T[rows++] = T1; }
  std::vector<char> act(k, 0);
  std::vector<double> x(k, beta0);
  for(int it = 0; it < 500; ++it)
  {
    double M[2][2] = {{0., 0.}, {0., 0.}}, rhs[2] = {0., 0.}, lam[2] = {0., 0.};
    for(size_t j = 0; j < k; ++j)
      for(int a = 0; a < rows; ++a)
      {
        if(!act[j])
        {
          rhs[a] += T[a][j];
          for(int b2 = 0; b2 < rows; ++b2)
            M[a][b2] += T[a][j]*T[b2][j];
        }
        else
          rhs[a] += beta0*T[a][j];
      }
    if(rows == 1)
      lam[0] = M[0][0] != 0. ? rhs[0]/M[0][0] : 0.;
    else
      solve2_pinv(M, rhs, lam);
    double xmin = 1e300;
    size_t jmin = 0;
    for(size_t j = 0; j < k; ++j)
    {
      if(act[j]) { x[j] = beta0; continue; }
      double tl = 0.;
      for(int a = 0; a < rows; ++a)
        tl += T[a][j]*lam[a];
      x[j] = 1. - tl;
      if(x[j] < xmin) { xmin = x[j]; jmin = j; }
    }
    if(xmin < beta0*(1. - 1e-12))
    {
      act[jmin] = 1;
      continue;
    }
    // multipliers of the active bounds
    double lmax = 0.;
    for(int a = 0; a < rows; ++a)
      lmax = std::max(lmax, std::abs(lam[a]));
    double mumin = 1e300;
    size_t jm = 0;
    bool any = false;
    for(size_t j = 0; j < k; ++j)
      if(act[j])
      {
        double mu = beta0 - 1.;
        for(int a = 0; a < rows; ++a)
          mu += T[a][j]*lam[a];
        if(mu < mumin) { mumin = mu; jm = j; }
        any = true;
      }
    if(any && mumin < -1e-10*std::max(1., lmax))
    {
      act[jm] = 0;
      continue;
    }
    for(auto& v : x)
      v = std::max(v, beta0);
    return x;
  }
  // dual: x(lam) = max(beta0, 1 - T^T lam), semismooth Newton on T x(lam) = 0
  double lam[2] = {0., 0.};
  auto xr = [&](const double l[2], std::vector<double>& xx, double r[2])
  {
    r[0] = r[1] = 0.;
    for(size_t j = 0; j < k; ++j)
    {
      double tl = 0.;
      for(int a = 0; a < rows; ++a)
        tl += T[a][j]*l[a];
      xx[j] = std::max(beta0, 1. - tl);
      for(int a = 0; a < rows; ++a)
        r[a] += T[a][j]*xx[j];
    }
  };
  double r[2];
  xr(lam, x, r);
  for(int it = 0; it < 200; ++it)
  {
    double scale = 1e-300;
    for(size_t j = 0; j < k; ++j)
      for(int a = 0; a < rows; ++a)
        scale += std::abs(T[a][j])*std::abs(x[j]);
    const double nr = std::hypot(r[0], r[1]);
    if(nr <= 1e-12*scale)
      return x;
    double Jm[2][2] = {{0., 0.}, {0., 0.}};
    for(size_t j = 0; j < k; ++j)
    {
      double tl = 0.;
      for(int a = 0; a < rows; ++a)
        tl += T[a][j]*lam[a];
      if(1. - tl > beta0)
        for(int a = 0; a < rows; ++a)
          for(int b2 = 0; b2 < rows; ++b2)
            Jm[a][b2] -= T[a][j]*T[b2][j];
    }
    double mr[2] = {-r[0], -r[1]}, step[2] = {0., 0.};
    if(rows == 1)
      step[0] = Jm[0][0] != 0. ? mr[0]/Jm[0][0] : 0.;
    else
      solve2_pinv(Jm, mr, step);
    double t = 1.;
    std::vector<double> xn(k);
    double rn[2], ln[2];
    for(int ls = 0; ls < 60; ++ls)
    {
      ln[0] = lam[0] + t*step[0];
      ln[1] = lam[1] + t*step[1];
      xr(ln, xn, rn);
      if(std::hypot(rn[0], rn[1]) < nr)
        break;
      t *= 0.5;
    }
    lam[0] = ln[0]; lam[1] = ln[1];
    x = xn; r[0] = rn[0]; r[1] = rn[1];
  }
  ErrThrow("AlgebraicStabilization: the weights beta of fbbk could not be computed");
}
} // namespace

void Operator::setup_fbbk(double beta0)
{
  // directed edges by source node
  const size_t ne = I.size();
  std::vector<std::vector<size_t>> out(N);
  for(size_t e = 0; e < ne; ++e)
  {
    out[I[e]].push_back(e);           // I -> J
    out[J[e]].push_back(e + ne);      // J -> I
  }
  for(int i = 0; i < n_free; ++i)
  {
    if(on_boundary[i] || out[i].empty())
      continue;
    std::vector<double> T0, T1;
    double tmax = 0., s0 = 0., s1 = 0.;
    for(size_t k : out[i])
    {
      const int j = k < ne ? J[k] : I[k - ne];
      T0.push_back(X[j] - X[i]);
      T1.push_back(Y[j] - Y[i]);
      tmax = std::max(tmax, std::max(std::abs(T0.back()), std::abs(T1.back())));
      s0 += T0.back();
      s1 += T1.back();
    }
    if(std::hypot(s0, s1) <= 1e-12*tmax)
      continue;                       // symmetric patch: beta = 1
    const std::vector<double> b = ldp(T0, T1, beta0);
    for(size_t m = 0; m < out[i].size(); ++m)
      beta_dir[out[i][m]] = b[m];
  }
}

std::vector<double> Operator::b_edges(const std::vector<double>& u) const
{
  if(method == "kuzmin_edge")
    return b_kuzmin(u);
  if(method == "bbk")
    return b_bbk(u);
  if(method == "sbbk" || method == "fbbk")
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
  for(size_t e = 0; e < ne; ++e)              // beta_dir = 1 for sbbk
  {
    const double du = u[I[e]] - u[J[e]];
    const double bI = beta_dir[e], bJ = beta_dir[e + ne];
    num[I[e]] += bI*du;   num[J[e]] -= bJ*du;
    den[I[e]] += bI*std::abs(du);   den[J[e]] += bJ*std::abs(du);
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
