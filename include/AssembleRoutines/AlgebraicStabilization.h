#ifndef __ALGEBRAICSTABILIZATION__
#define __ALGEBRAICSTABILIZATION__

/** ****************************************************************************
 * Algebraic stabilizations of P1 convection-diffusion-reaction problems,
 * implemented separately from AlgebraicFluxCorrection (whose limiters are not
 * changed): the scheme
 *
 *   sum_j (a_ij + b_ij(u)) u_j = g_i (free i),  u_i = u_b(x_i) (Dirichlet i),
 *
 * with a symmetric matrix B(u), b_ij = b_ji <= 0, b_ii = -sum_{j != i} b_ij,
 * defined edge by edge (a_ij: Galerkin matrix with full rows also for the
 * Dirichlet nodes; d_ij = -max{a_ij, 0, a_ji}; f_ij = d_ij (u_j - u_i)).
 *
 * Methods (parameter algstab_method):
 *  bjk          AFC with the limiter of Barrenechea, John & Knobloch, M3AS 27
 *               (2017): b_ij = (1 - alpha_ij) d_ij, d_ij from the matrix with
 *               a_ji := 0 for i free, j Dirichlet, a_ij < 0 ((2.4)),
 *               P_i^+- = sum_j f_ij^+-, Q_i^+- = mu q_i (u_i - u_i^max/min),
 *               q_i = gamma_i sum_j d_ij, gamma_i = max_j |x_i - x_j| /
 *               dist(x_i, boundary of conv(patch)) ((6.5); at Neumann nodes the
 *               distance to the hull facets not through x_i), alpha_ij =
 *               min{R_i, R_j} by the signs of f_ij, f_ji (free ends only).
 *  mc           AFC with monolithic convex limiting (Kuzmin 2020) as in Ahmed &
 *               Jha (2027), eq. (8): bar states
 *               u_ij = (u_i + u_j)/2 + a_ij (u_j - u_i)/(2 d_ij), alpha_ij =
 *               min{1, 2 d_ij (u_ij - u_i^max)/f_ij, 2 d_ij (u_j^min - u_ji)/f_ij}
 *               for f_ij > 0, max and min exchanged for f_ij < 0.
 *  muas         John & Knobloch (2022) as in Knobloch (2023), (31)-(36): as smuas
 *               without reflected values, p_ij = a_ij^+, q_ij = max{|a_ij|, a_ji}
 *               (AlgebraicFluxCorrection's MUAS_MAX).
 *  kuzmin_edge  AFC with the Kuzmin limiter (Kuzmin 2007, as in Barrenechea,
 *               John & Knobloch, SINUM 2016, and Knobloch 2023 (19)-(23)):
 *               b_ij = (1 - alpha_ij) d_ij, every edge limited at exactly one
 *               node, its owner: i if a_ji < a_ij; for a_ij = a_ji (relative
 *               difference at most algstab_tie_tol, i.e. equal up to the
 *               round-off of the assembly) the node with the lexicographically
 *               smaller coordinates (x, then y).
 *               P_i^+- = sum over owned edges of f_ij^+-,
 *               Q_i^+- = -sum_j f_ij^-+ (all edges), R_i^+- = min{1, Q/P},
 *               1 where P = 0 and at Dirichlet nodes; alpha_ij = R^+ or R^-
 *               of the owner by the sign of the flux seen from the owner.
 *               (AlgebraicFluxCorrection's limiter 'kuzmin' limits an edge
 *               with a_ij = a_ji at both nodes, so that alpha is not
 *               symmetric there.)
 *  bbk          Barrenechea, Burman & Karakatsani (2017) in the form of Ahmed &
 *               Jha (2027), eq. (7): b_ij = d_ij max{xi_i, xi_j}^p,
 *               xi_i = |sum_j (u_i - u_j)| / sum_j |u_i - u_j| over the edge
 *               neighbours (0 if the denominator vanishes and at Dirichlet
 *               nodes), p = algstab_bbk_p.
 *  fbbk         edge form of Allendes, Barrenechea & Rankin, SISC 39 (2017),
 *               (33)-(37) ("F-BBK"): b_ij = -C0 |E| max{xi_i, xi_j}^p on every
 *               mesh edge, xi_i = |sum_j beta_ij (u_i - u_j)| /
 *               sum_j beta_ij |u_i - u_j| at interior free nodes, 0 on the
 *               boundary of the domain; beta_ij = argmin sum_j |beta_ij - 1|^2
 *               subject to sum_j beta_ij (x_j - x_i) = 0 and beta_ij >= beta0
 *               (a strictly convex QP per node: primal active set, dual
 *               semismooth Newton as fallback), beta = 1 on symmetric patches;
 *               C0 = algstab_bbk_C0, beta0 = algstab_bbk_beta0.
 *  sbbk         the same with beta_ij = 1 ("S-BBK").
 *  smuas        Knobloch, Numer. Algorithms 94 (2023), (60)-(68):
 *               b_ij = -max{beta_ij a_ij, 0, beta_ji a_ji},
 *               P_i^+- = sum_j p_ij {(u_i - u_j)^+- + (u_i - u_ij)^+-},
 *               Q_i^+- = sum_j q_ij {(u_j - u_i)^+- + (u_ij - u_i)^+-},
 *               u_ij = u_i + grad u_h|_T . (x_i - x_j) with T the cell of the
 *               patch of x_i entered by the half line x_i + t (x_i - x_j)
 *               (u_ij = u_i if the half line leaves the domain), weights (67)
 *               p_ij = max{a_ij, 0, a_ji}, q_ij = max{|a_ij|, a_ji}, or (68)
 *               p_ij = q_ij = 1 (p_ij = 0 unless a_ij > 0 or a_ji > 0);
 *               beta_ij = 1 - R_i^+ (u_i > u_j), 1 - R_i^- (u_i < u_j), 0.
 * Modifications (all methods):
 *  algstab_resolved_off  d_ij = b_ij = 0 on the edges with mesh Peclet number
 *               |b(x_E)| |E| / (2 eps) <= algstab_pe_threshold (x_E midpoint);
 *  algstab_layer_off     d_ij = b_ij = 0 on the edges along a layer: on every
 *               cell K at the edge diam(K)^2/(2|K|) >= algstab_aspect_threshold
 *               and |E| >= diam(K)/2 (the long edges of strongly stretched cells);
 *  algstab_q_relax mu    R_i = min{1, mu Q_i / P_i} (bjk, kuzmin_edge, muas, smuas):
 *               limits less; the DMP is no longer guaranteed for mu > 1.
 * At Neumann nodes smuas uses u_ij = u_i (no reflected value).
 * The definitions follow the galerkit implementation of the anisotropic AFC
 * study (src/galerkit/convection_diffusion/afc.py), against which this code is
 * cross-checked (scripts/aniso_afc/crosscheck_galerkit.py).
 *
 * Nonlinear solver: fixed point iteration with updated right-hand side,
 *   (A + D) u~ = g + (D - B(u^k)) u^k,
 * inside ConvectionDiffusion_AFC (afc_iteration_scheme fixed_point_rhs).
 ******************************************************************************/

#include <memory>
#include <string>
#include <vector>
#include "Constants.h"
#include "ParameterDatabase.h"

class FEMatrix;

namespace AlgebraicStabilization
{
/// algstab_method (none, bjk, mc, muas, kuzmin_edge, bbk, fbbk, sbbk, smuas) and the parameters
ParameterDatabase default_database();

/// true if algstab_method is not 'none'
bool active(const ParameterDatabase& db);

/// write x, y, u, free (1) or Dirichlet (0) of every dof of a P1 space on
/// triangles, with 17 significant digits (the vtk files store about 8)
void write_nodal_values(const FEMatrix& A, const std::vector<double>& u,
                        const std::string& path);

/// write i, j, x_i, y_i, x_j, y_j, a_ij, a_ji of every edge of the Galerkin
/// matrix with full rows (for the DMP mesh conditions)
void write_edges(const FEMatrix& A, const std::string& path);

class Operator
{
  public:
    /// A: Galerkin matrix with full rows for all nodes (before D is added),
    /// on a P1 space of triangles; the free nodes are the active dofs
    /// coeffs: coefficients of the example (eps = coeff[0], b = coeff[1..2]),
    /// needed for algstab_resolved_off
    Operator(const FEMatrix& A, const ParameterDatabase& db,
             const CoeffFct2D& coeffs = CoeffFct2D());

    /// i, j, x_i, y_i, x_j, y_j, a_ij, a_ji, d_ij, active of every edge
    void write_edges(const std::string& path) const;

    /// test: read u from path (x y u), write b_ij(u) to path.b
    void test_b(const std::string& path) const;

    /// number of edges where the stabilization is switched off
    size_t n_switched_off() const { return n_off; }

    /// M += D (all rows, as AlgebraicFluxCorrection does with D)
    void add_D(FEMatrix& M) const;

    /// rhs_i += sum_j (d_ij - b_ij(u)) (u_j - u_i) for the free nodes i
    void add_rhs_correction(const std::vector<double>& u,
                            std::vector<double>& rhs) const;

    /// b_ij(u) per undirected edge (edge e = (I[e], J[e]), I < J)
    std::vector<double> b_edges(const std::vector<double>& u) const;

    size_t n_edges() const { return I.size(); }
    const std::string& get_method() const { return method; }

  private:
    std::string method;
    int N, n_free;
    // undirected edges I < J, matrix entries, artificial diffusion
    std::vector<int> I, J;
    std::vector<double> aij, aji, d;
    // dof coordinates and the boundary of the domain
    std::vector<double> X, Y;
    std::vector<char> on_boundary;
    // kuzmin_edge: owner of the edge is I
    std::vector<char> owner_is_I;
    // bbk-type: exponent, edge weights (sbbk)
    double p_bbk;
    std::vector<double> w_edge;
    // smuas: directed edges s -> l, weights, reflection rows
    std::vector<int> s, l;
    std::vector<double> p_w, q_w;
    std::vector<int> W_ptr, W_col;
    std::vector<double> W_val;

    bool is_free(int i) const { return i < n_free; }
    // switches (active[e] = 0: d = b = 0), relaxation mu, cells (dofs, area)
    std::vector<char> active;
    size_t n_off = 0;
    double q_relax = 1.;
    std::vector<int> cell_dof;
    std::vector<double> cell_area;
    // bjk: gamma_i and sum_j d_ij
    std::vector<double> gamma, sum_d;
    std::vector<double> b_bjk(const std::vector<double>& u) const;
    std::vector<double> b_mc(const std::vector<double>& u) const;
    void extrema(const std::vector<double>& u, std::vector<double>& umax,
                 std::vector<double>& umin) const;
    void setup_switches(const ParameterDatabase& db, const CoeffFct2D& coeffs);
    void setup_gamma();
    std::vector<double> b_kuzmin(const std::vector<double>& u) const;
    std::vector<double> b_bbk(const std::vector<double>& u) const;
    std::vector<double> b_sbbk(const std::vector<double>& u) const;
    // fbbk: weights beta on the directed edges (e: I -> J, e + n_edges: J -> I)
    std::vector<double> beta_dir;
    void setup_fbbk(double beta0);
    std::vector<double> b_smuas(const std::vector<double>& u) const;
    void setup_geometry(const FEMatrix& A);
    void setup_smuas(const FEMatrix& A, const std::string& weights, bool reflect);
};

} // namespace AlgebraicStabilization

#endif // __ALGEBRAICSTABILIZATION__
