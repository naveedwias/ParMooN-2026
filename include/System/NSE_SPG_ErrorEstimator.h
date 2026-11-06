// =============================================================================
// Residual-based a posteriori error estimator for the SUPG/PSPG/grad-div
// stabilized finite element method applied to the Oseen problem (and, with the
// modifications of Section 5, to the stationary Navier-Stokes problem):
//
//   M. Afzal, N. Ahmed, V. John, A robust a posteriori error estimator for the
//   Oseen problem, Comput. Math. Appl. 223 (2026) 467-486, Theorem 1, Eq. (30).
//
// The estimator is
//   eta^2 = sum_K ( eta_{K,res}^2 + eta_{K,div}^2 + eta_{K,delta}^2 + eta_{K,mu}^2 )
//           + sum_F eta_F^2,
// with (all generic constants C = 1 as in Section 4)
//   eta_{K,res}^2 = min{ 1/sigma_0, h_K^2/nu, 40 delta_K } ||r_K||^2_K
//   eta_{K,div}^2 = 160 max_K{delta_K} ||div u_h||^2_K           (equal order)
//                 = 160 max_K{delta_K}/h^2 ||div u_h||^2_K       (inf-sup stable)
//   eta_F^2       = min{ min{h_K/nu, 1/(h_K sigma_0)} + min{h_K/nu, 1/(nu^1/2 sigma_0^1/2)},
//                        40/||b||^2_{L^inf(F)} } ||r_F||^2_F
//   eta_{K,delta}^2 = 40 delta_K ||r_K||^2_K
//   eta_{K,mu}^2  = min{ 40 d mu_K^2/delta_K^alpha, d mu_K^2/nu } ||div u_h||^2_K
// where r_K = f + nu Lap u_h - (b.grad)u_h - sigma u_h - grad p_h is the mesh
// cell residual (18) and r_F the facet residual (19); alpha = 1 if nu < h and
// equal order pairs are used, alpha = 0 else, see (23). The global term
// eta_{K,div} is distributed cell-wise so that the sum over all cells gives the
// global expression of Theorem 1; the facet terms of interior facets are
// distributed with weight 1/2 to both adjacent cells.
//
// The local indicator used for marking is
//   eta_K = ( eta_{K,res}^2 + eta_{K,div}^2 + eta_F^2 + eta_{K,delta}^2 + eta_{K,mu}^2 )^{1/2}
// as in Section 4.2.
//
// For Navier-Stokes (Section 5): b = u_h in r_K and in eta_F, sigma_0 = 0.
//
// In addition, if the example provides an exact solution, the error in the norm
//   ||(v,q)||_spg^2 = nu||grad v||^2 + ||sigma^{1/2} v||^2 + sum_K mu_K ||div v||_K^2
//                     + sum_K delta_K ||(b.grad)v + grad q||_K^2                 (9)
// is computed (for Navier-Stokes: ||.||_{spg,nse}, the norm of Section 5),
// so that the effectivity index eta / error is available.
//
// Only 2D, conforming meshes (hanging nodes are not supported) and Dirichlet or
// Neumann boundary conditions with zero Neumann data are supported.
// =============================================================================

#ifndef NSE_SPG_ERRORESTIMATOR_H
#define NSE_SPG_ERRORESTIMATOR_H

#include "ErrorEstimator.h"
#include "Example_NSE2D.h"
#include "FEFunction2D.h"
#include "FEVectFunct2D.h"
#include <array>

class NSE_SPG_ErrorEstimator : public ErrorEstimator<2>
{
  public:
    /// @brief parameters: 'spg_delta0', 'spg_mu0' (stabilization parameters,
    /// must be the same as used in the discretization), 'problem_type' (3:
    /// Oseen, 5: Navier-Stokes), 'spg_estimator_sigma0' (lower bound sigma_0
    /// of the reaction coefficient, default 0, i.e., the terms with 1/sigma_0
    /// are dropped from the minima).
    explicit NSE_SPG_ErrorEstimator(const ParameterDatabase& param_db);

    static ParameterDatabase default_spg_estimator_database();

    /// @brief compute all local and global quantities
    void estimate(const Example_NSE2D& ex, const TFEVectFunct2D& u,
                  const TFEFunction2D& p);

    /** @brief transient Navier-Stokes mode (adaptive SUPG/PSPG/grad-div
     * method for the time-dependent Navier-Stokes equations, extension of
     * Afzal, Ahmed, John 2026 in the manner of de Frutos et al. 2014):
     * the discrete time derivative dt_u (same space as the velocity) is
     * subtracted from f in the cell residual, the constants of the
     * transient bound are used (320 delta_K in eta_res, 160 delta_K in
     * eta_delta, 640 max delta in eta_div, 320/|U|^2 in eta_F with h_F,
     * 320 d mu^2/delta^alpha in eta_mu), sigma_0 = 0, and the error norm is
     * nu|grad e|^2 + sum mu_K |div e|^2 + sum delta_K |(U.grad)e_u + grad e_p|^2
     * with the discrete velocity U as convection field. Pass nullptr to
     * return to the steady setting. The pointer is only stored.
     */
    void set_transient(const TFEVectFunct2D* dt_u)
    { time_derivative = dt_u; transient = (dt_u != nullptr); }

    void info() override;

    // global contributions (square roots of the sums over all cells/facets)
    double get_eta_res() const { return eta_res; }
    double get_eta_div() const { return eta_div; }
    double get_eta_F() const { return eta_F; }
    double get_eta_delta() const { return eta_delta; }
    double get_eta_mu() const { return eta_mu; }
    /// @brief error ||(u-u_h, p-p_h)||_spg (only if an exact solution is known)
    double get_error_spg() const { return error_spg; }
    /// @brief contributions of the error: nu||grad e_u||^2, ||sigma^1/2 e_u||^2,
    /// sum mu_K||div e_u||^2, sum delta_K||(b.grad)e_u+grad e_p||^2, ||e_p||^2
    std::array<double, 5> get_error_parts() const { return error_parts; }
    /// @brief effectivity index eta/error (0 if no exact solution)
    double get_effectivity_index() const;
    /// @brief ||div u_h||_{L^2(Omega)}
    double get_div_uh() const { return div_uh; }
    /// @brief mesh width h = max_K h_K
    double get_h() const { return h_max; }
    double get_max_delta() const { return delta_max; }

  protected:
    double delta0, mu0, sigma0;
    int error_quad_degree;
    bool equal_order;
    bool is_nse;
    bool has_exact_solution;
    const TFEVectFunct2D* time_derivative = nullptr;
    bool transient = false;
    double eta_res, eta_div, eta_F, eta_delta, eta_mu;
    double error_spg;
    std::array<double, 5> error_parts;
    double div_uh;
    double h_max, delta_max;
};

#endif // NSE_SPG_ERRORESTIMATOR_H
