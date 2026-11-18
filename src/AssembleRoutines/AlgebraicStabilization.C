// Algebraic stabilizations of P1 convection-diffusion-reaction problems,
// separate from AlgebraicFluxCorrection; see AlgebraicStabilization.h.
#include "AlgebraicStabilization.h"

#include <algorithm>
#include <array>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
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
         {"none", "bjk", "mc", "muas", "kuzmin_edge", "bbk", "fbbk", "sbbk", "smuas"});
  db.add("algstab_resolved_off", false, "switch the stabilization off on the edges "
         "with mesh Peclet number |b(x_E)| |E| / (2 eps) <= algstab_pe_threshold");
  db.add("algstab_pe_threshold", 1.0, "threshold of algstab_resolved_off", 0., 1.e10);
  db.add("algstab_layer_off", false, "switch the stabilization off on the edges "
         "along a layer (long edges of cells with aspect ratio >= "
         "algstab_aspect_threshold)");
  db.add("algstab_aspect_threshold", 10.0, "aspect ratio diam(K)^2 / (2 |K|) of "
         "algstab_layer_off (10 on tensor-product, 3 on polar meshes)", 1., 1.e20);
  db.add("algstab_q_relax", 1.0, "relaxed limiters R_i = min{1, mu Q_i / P_i} "
         "(bjk, kuzmin_edge, muas, smuas); mu > 1 limits less, DMP not guaranteed",
         1., 1.e10);
  db.add("algstab_write_nodal", "", "file for x, y, u, free of every dof at the "
         "end (17 digits); empty: none");
  db.add("algstab_write_edges", "", "file for the edges i, j, x_i, y_i, x_j, y_j, "
         "a_ij, a_ji of the Galerkin matrix (first AFC step); empty: none");
  db.add("algstab_test_b", "", "test: read u (x y u per line, any node order) from "
         "this file, evaluate b_ij(u) and write it to <file>.b (i j x_i y_i x_j y_j "
         "b_ij); empty: none");
  db.add("algstab_solver", "fixed_point", "nonlinear solver of the algebraic "
         "stabilizations: fixed_point (ParMooN's fixed point rhs loop with its "
         "damping), fp_anderson (Anderson acceleration), fp_anderson_safe (with "
         "the non-monotone safeguard)", {"fixed_point", "fp_anderson",
         "fp_anderson_safe"});
  db.add("algstab_anderson_depth", 10, "depth of the Anderson acceleration", 1, 100);
  db.add("algstab_anderson_window", 10, "safeguard: an accelerated iterate is "
         "accepted if its residual is below the largest of the last window ones",
         1, 1000);
  db.add("algstab_restart", 200, "Anderson: restart from the best iterate if "
         "the best residual has not halved within this number of iterations "
         "(0: never)", 0, 1000000);
  db.add("algstab_omega_min", 1.e-4, "smallest damping of the safeguard step",
         0., 1.);
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

Operator::Operator(const FEMatrix& A, const ParameterDatabase& db,
                   const CoeffFct2D& coeffs)
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
  q_relax = db["algstab_q_relax"];
  if(method == "bjk")
  {
    // (2.4): a_ji := 0 for i free, j Dirichlet with a_ij < 0 (changes only d)
    for(size_t e = 0; e < I.size(); ++e)
    {
      double am = aij[e], at = aji[e];
      if(is_free(I[e]) && !is_free(J[e]) && aij[e] < 0.)
        at = 0.;
      if(is_free(J[e]) && !is_free(I[e]) && aji[e] < 0.)
        am = 0.;
      d[e] = -std::max(std::max(am, at), 0.);
    }
  }
  setup_switches(db, coeffs);
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
    setup_smuas(A, db["algstab_smuas_weights"].get<std::string>(), true);
  else if(method == "muas")
    setup_smuas(A, "muas", false);
  else if(method == "bjk")
    setup_gamma();
  else if(method != "bbk" && method != "mc")
    ErrThrow("AlgebraicStabilization: unknown method ", method);
  Output::print<1>("AlgebraicStabilization: ", method, ", ", I.size(), " edges (",
                   n_off, " switched off), ", n_free, " free of ", N, " nodes");
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
    for(int k = 0; k < 3; ++k)
      cell_dof.push_back(dof[k]);
    {
      const double x0 = cell->GetVertex(0)->GetX(), y0 = cell->GetVertex(0)->GetY();
      const double x1 = cell->GetVertex(1)->GetX(), y1 = cell->GetVertex(1)->GetY();
      const double x2 = cell->GetVertex(2)->GetX(), y2 = cell->GetVertex(2)->GetY();
      cell_area.push_back(0.5*std::abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)));
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

void Operator::setup_smuas(const FEMatrix& A, const std::string& weights, bool reflect)
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
      if(weights == "muas")                     // MUAS: p = a_sl^+, q = max{|a_sl|, a_ls}
      {
        p_w[k] = a_sl > 0. ? a_sl : 0.;
        q_w[k] = std::max(std::abs(a_sl), a_ls);
      }
      else if(weights == "67")
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
  if(!reflect)
    return;
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
    // Neumann nodes (free nodes on the boundary): u_ij = u_i (galerkit muas_nodes)
    if(on_boundary[i])
    {
      W_col.push_back(i);
      W_val.push_back(1.);
      W_ptr.push_back((int) W_col.size());
      continue;
    }
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
void solve2_pinv(const double M[2][2], const double r[2], double x[2],
                 double rcond = -1.)
{
  // pseudo-inverse of the symmetric 2x2 matrix by a Jacobi rotation (stable
  // eigenvectors also for tiny off-diagonal entries; the formula (l - d, b)
  // is not, on the anisotropic patches of Shishkin meshes with eps = 1e-8)
  const double a = M[0][0], b = M[0][1], dd = M[1][1];
  const double th = 0.5*std::atan2(2.*b, a - dd);
  const double c = std::cos(th), sn = std::sin(th);
  const double l1 = a*c*c + 2.*b*c*sn + dd*sn*sn;
  const double l2 = a*sn*sn - 2.*b*c*sn + dd*c*c;
  const double v1[2] = {c, sn}, v2[2] = {-sn, c};
  // cut-off: numpy lstsq default (eps * size * max |l|) or rcond * max |l|
  const double lmax = std::max(std::abs(l1), std::abs(l2));
  const double cut = (rcond < 0. ? std::numeric_limits<double>::epsilon()*2. : rcond)*lmax;
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
      solve2_pinv(Jm, mr, step, 1e-300);   // keep tiny singular values (anisotropy)
    double t = 1.;
    std::vector<double> xn(k);
    double rn[2], ln[2];
    for(int ls = 0; ls < 60; ++ls)
    {
      ln[0] = lam[0] + t*step[0];
      ln[1] = lam[1] + t*step[1];
      xr(ln, xn, rn);
      if(std::hypot(rn[0], rn[1]) < (1. - 1e-4*t)*nr)
        break;
      t *= 0.5;
      if(t <= 1e-12)
        break;
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
  std::vector<double> b;
  if(method == "kuzmin_edge")
    b = b_kuzmin(u);
  else if(method == "bjk")
    b = b_bjk(u);
  else if(method == "mc")
    b = b_mc(u);
  else if(method == "bbk")
    b = b_bbk(u);
  else if(method == "sbbk" || method == "fbbk")
    b = b_sbbk(u);
  else
    b = b_smuas(u);                            // smuas, muas
  for(size_t e = 0; e < b.size(); ++e)         // switched-off edges: Galerkin
    if(!active[e])
      b[e] = 0.;
  return b;
}

void Operator::extrema(const std::vector<double>& u, std::vector<double>& umax,
                       std::vector<double>& umin) const
{
  umax = u;
  umin = u;
  for(size_t e = 0; e < I.size(); ++e)
  {
    umax[I[e]] = std::max(umax[I[e]], u[J[e]]);
    umax[J[e]] = std::max(umax[J[e]], u[I[e]]);
    umin[I[e]] = std::min(umin[I[e]], u[J[e]]);
    umin[J[e]] = std::min(umin[J[e]], u[I[e]]);
  }
}

std::vector<double> Operator::b_bjk(const std::vector<double>& u) const
{
  const size_t ne = I.size();
  std::vector<double> Pp(N, 0.), Pm(N, 0.), f(ne), umax, umin;
  for(size_t e = 0; e < ne; ++e)
  {
    f[e] = d[e]*(u[J[e]] - u[I[e]]);
    Pp[I[e]] += pos(f[e]);   Pm[I[e]] += neg(f[e]);
    Pp[J[e]] -= neg(f[e]);   Pm[J[e]] -= pos(f[e]);
  }
  extrema(u, umax, umin);
  std::vector<double> Rp(N, 1.), Rm(N, 1.);
  for(int i = 0; i < n_free; ++i)
  {
    const double q = gamma[i]*sum_d[i];
    Rp[i] = ratio(q_relax*q*(u[i] - umax[i]), Pp[i]);
    Rm[i] = ratio(q_relax*q*(u[i] - umin[i]), Pm[i]);
  }
  std::vector<double> b(ne);
  for(size_t e = 0; e < ne; ++e)
  {
    const double ai = f[e] > 0. ? Rp[I[e]] : (f[e] < 0. ? Rm[I[e]] : 1.);
    const double aj = -f[e] > 0. ? Rp[J[e]] : (-f[e] < 0. ? Rm[J[e]] : 1.);
    const bool fi = is_free(I[e]), fj = is_free(J[e]);
    const double alpha = (fi && fj) ? std::min(ai, aj) : (fi ? ai : (fj ? aj : 1.));
    b[e] = (1. - alpha)*d[e];
  }
  return b;
}

std::vector<double> Operator::b_mc(const std::vector<double>& u) const
{
  const size_t ne = I.size();
  std::vector<double> umax, umin, b(ne);
  extrema(u, umax, umin);
  for(size_t e = 0; e < ne; ++e)
  {
    if(d[e] == 0.)
    {
      b[e] = 0.;
      continue;
    }
    const int i = I[e], j = J[e];
    const double ub_ij = 0.5*(u[i] + u[j]) + aij[e]*(u[j] - u[i])/(2.*d[e]);
    const double ub_ji = 0.5*(u[i] + u[j]) + aji[e]*(u[i] - u[j])/(2.*d[e]);
    const double f = d[e]*(u[j] - u[i]);
    double alpha = 0.;
    if(f > 0.)
      alpha = std::min(1., std::min(2.*d[e]*(ub_ij - umax[i])/f, 2.*d[e]*(umin[j] - ub_ji)/f));
    else if(f < 0.)
      alpha = std::min(1., std::min(2.*d[e]*(ub_ij - umin[i])/f, 2.*d[e]*(umax[j] - ub_ji)/f));
    alpha = std::min(1., std::max(0., alpha));
    b[e] = (1. - alpha)*d[e];
  }
  return b;
}

namespace
{
// 2D convex hull (monotone chain), counterclockwise, without collinear points
std::vector<std::array<double, 2>> convex_hull(std::vector<std::array<double, 2>> p)
{
  std::sort(p.begin(), p.end());
  p.erase(std::unique(p.begin(), p.end()), p.end());
  if(p.size() < 3)
    return p;
  auto cross = [](const std::array<double, 2>& o, const std::array<double, 2>& a,
                  const std::array<double, 2>& b)
  { return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0]); };
  std::vector<std::array<double, 2>> h(2*p.size());
  size_t k = 0;
  for(size_t i = 0; i < p.size(); ++i)
  {
    while(k >= 2 && cross(h[k-2], h[k-1], p[i]) <= 0.) --k;
    h[k++] = p[i];
  }
  for(size_t i = p.size() - 1, t = k + 1; i > 0; --i)
  {
    while(k >= t && cross(h[k-2], h[k-1], p[i-1]) <= 0.) --k;
    h[k++] = p[i-1];
  }
  h.resize(k - 1);
  return h;
}
}

void Operator::setup_gamma()
{
  // gamma_i = max_j |x_i - x_j| / dist(x_i, boundary of conv{x_j : j in S_i}) (BJK17 (6.5));
  // at Neumann nodes x_i lies on the hull: distance to the facets not through x_i
  std::vector<std::vector<int>> nb(N);
  for(size_t e = 0; e < I.size(); ++e)
  {
    nb[I[e]].push_back(J[e]);
    nb[J[e]].push_back(I[e]);
  }
  gamma.assign(N, 1.);
  sum_d.assign(N, 0.);
  for(size_t e = 0; e < I.size(); ++e)
  {
    sum_d[I[e]] += d[e];
    sum_d[J[e]] += d[e];
  }
  for(int i = 0; i < n_free; ++i)
  {
    std::vector<std::array<double, 2>> rel;
    double scale = 0.;
    for(int j : nb[i])
    {
      rel.push_back({X[j] - X[i], Y[j] - Y[i]});
      scale = std::max(scale, std::hypot(rel.back()[0], rel.back()[1]));
    }
    const bool neumann = on_boundary[i];
    if(neumann)
      rel.push_back({0., 0.});
    const auto h = convex_hull(rel);
    double dist = 1e300;
    for(size_t k = 0; k < h.size(); ++k)
    {
      const auto& p = h[k];
      const auto& q = h[(k + 1) % h.size()];
      const double len = std::hypot(q[0] - p[0], q[1] - p[1]);
      // distance of the origin (x_i) to the line through p, q (inside: >= 0)
      const double dl = std::abs(p[0]*q[1] - p[1]*q[0])/len;
      if(neumann && dl <= 1e-12*scale)
        continue;                              // facet through x_i
      dist = std::min(dist, dl);
    }
    if(!(dist > 1e-14*scale) || dist == 1e300)
      ErrThrow("AlgebraicStabilization: vertex ", i, " lies on the boundary of the "
               "convex hull of its patch");
    gamma[i] = scale/dist;
  }
}

void Operator::setup_switches(const ParameterDatabase& db, const CoeffFct2D& coeffs)
{
  const size_t ne = I.size();
  active.assign(ne, 1);
  if((bool) db["algstab_resolved_off"])
  {
    if(!coeffs)
      ErrThrow("AlgebraicStabilization: algstab_resolved_off needs the coefficients");
    const double thr = db["algstab_pe_threshold"];
    std::vector<double> cbuf(20, 0.);
    double* cp = cbuf.data();
    for(size_t e = 0; e < ne; ++e)
    {
      double xm = 0.5*(X[I[e]] + X[J[e]]), ym = 0.5*(Y[I[e]] + Y[J[e]]);
      coeffs(1, &xm, &ym, nullptr, &cp);
      const double pe = std::hypot(cbuf[1], cbuf[2])
                        *std::hypot(X[J[e]] - X[I[e]], Y[J[e]] - Y[I[e]])/(2.*cbuf[0]);
      if(!(pe > thr))
        active[e] = 0;
    }
  }
  if((bool) db["algstab_layer_off"])
  {
    // the edge is along a layer if on every cell K at the edge
    // diam(K)^2/(2|K|) >= kappa and |E| >= diam(K)/2
    const double kappa = db["algstab_aspect_threshold"];
    std::map<std::pair<int, int>, int> bad;   // cells at the edge where it is not "along"
    std::map<std::pair<int, int>, int> cnt;
    const size_t nc = cell_area.size();
    for(size_t c = 0; c < nc; ++c)
    {
      const int* v = &cell_dof[3*c];
      double L[3], diam = 0.;
      for(int k = 0; k < 3; ++k)
      {
        const int a = v[k], b2 = v[(k + 1) % 3];
        L[k] = std::hypot(X[b2] - X[a], Y[b2] - Y[a]);
        diam = std::max(diam, L[k]);
      }
      const double aspect = diam*diam/(2.*cell_area[c]);
      for(int k = 0; k < 3; ++k)
      {
        const int a = v[k], b2 = v[(k + 1) % 3];
        const auto key = std::make_pair(std::min(a, b2), std::max(a, b2));
        cnt[key]++;
        if(!(aspect >= kappa && L[k] >= 0.5*diam))
          bad[key]++;
      }
    }
    for(size_t e = 0; e < ne; ++e)
    {
      const auto key = std::make_pair(I[e], J[e]);
      if(cnt.count(key) && !bad.count(key))
        active[e] = 0;
    }
  }
  n_off = 0;
  for(size_t e = 0; e < ne; ++e)
    if(!active[e])
    {
      d[e] = 0.;
      ++n_off;
    }
}

void Operator::test_b(const std::string& path) const
{
  // u at the dofs, matched by coordinates (nearest node)
  std::ifstream in(path);
  std::vector<std::array<double, 3>> pts;
  std::string line;
  while(std::getline(in, line))
  {
    if(line.empty() || line[0] == '#')
      continue;
    std::istringstream is(line);
    std::array<double, 3> p;
    if(is >> p[0] >> p[1] >> p[2])
      pts.push_back(p);
  }
  std::vector<double> u(N, 0.);
  for(int i = 0; i < N; ++i)
  {
    double best = 1e300;
    for(const auto& p : pts)
    {
      const double dd = std::hypot(p[0] - X[i], p[1] - Y[i]);
      if(dd < best) { best = dd; u[i] = p[2]; }
    }
  }
  const std::vector<double> b = b_edges(u);
  std::ofstream f(path + ".b");
  f << "# i j x_i y_i x_j y_j b_ij\n";
  f.precision(17);
  for(size_t e = 0; e < I.size(); ++e)
    f << I[e] << " " << J[e] << " " << X[I[e]] << " " << Y[I[e]] << " " << X[J[e]]
      << " " << Y[J[e]] << " " << b[e] << "\n";
  Output::print<1>("AlgebraicStabilization: b_ij(u) written to ", path, ".b");
}

void Operator::write_edges(const std::string& path) const
{
  std::ofstream f(path);
  f << "# i j x_i y_i x_j y_j a_ij a_ji d_ij active\n";
  f.precision(17);
  for(size_t e = 0; e < I.size(); ++e)
    f << I[e] << " " << J[e] << " " << X[I[e]] << " " << Y[I[e]] << " " << X[J[e]]
      << " " << Y[J[e]] << " " << aij[e] << " " << aji[e] << " " << d[e] << " "
      << (int) active[e] << "\n";
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
    Rp[i] = ratio(q_relax*Qp[i], Pp[i]);
    Rm[i] = ratio(q_relax*Qm[i], Pm[i]);
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
    double r = 0.;                             // u_ij - u_i (smuas only)
    if(!W_ptr.empty())
    {
      double wu = 0.;                          // reflected value u_ij
      for(int m = W_ptr[k]; m < W_ptr[k+1]; ++m)
        wu += W_val[m]*u[W_col[m]];
      r = wu - u[s[k]];
    }
    const int i = s[k];
    Pp[i] += p_w[k]*(pos(ds[k]) + pos(-r));
    Pm[i] += p_w[k]*(neg(ds[k]) + neg(-r));
    Qp[i] += q_w[k]*(pos(-ds[k]) + pos(r));
    Qm[i] += q_w[k]*(neg(-ds[k]) + neg(r));
  }
  std::vector<double> Rp(N, 1.), Rm(N, 1.);
  for(int i = 0; i < n_free; ++i)
  {
    Rp[i] = ratio(q_relax*Qp[i], Pp[i]);
    Rm[i] = ratio(q_relax*Qm[i], Pm[i]);
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
