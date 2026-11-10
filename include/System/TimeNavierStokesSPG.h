// =============================================================================
// SUPG/PSPG/grad-div semidiscretization of the time-dependent Navier-Stokes
// equations for the TR-AB2 integrator and the adaptive time loop (adaptive
// extension of Afzal, Ahmed, John 2026):
//   (d_t U, v) + sum_K delta_K (d_t U, (U.grad)v + grad q)_K + A^U((U,P),(v,q)) = L^U(v,q).
// TR step with the SUPG/PSPG mass matrix M^b and the steady matrix A^b, b the
// current Picard iterate:
//   (2/k M^{U} + A^{U})(U^{n+1},P^{n+1}) = L^{U} + M^{U}(2/k U^n + Ud^n),
//   Ud^{n+1} = 2/k (U^{n+1} - U^n) - Ud^n;
// (Ud^0, P^0) solve the semidiscrete problem at t = 0 (linear). The error
// estimator (NSE_SPG_ErrorEstimator, transient mode) gets Ud^{n+1}.
// NavierStokes<2> provides spaces, A^b, L^b, Dirichlet data, pressure
// projection and output; requires NSTYPE 14 and space_discretization_type
// supg_pspg_graddiv.
// =============================================================================
#ifndef TIMENAVIERSTOKESSPG_H
#define TIMENAVIERSTOKESSPG_H

#include "AdaptiveTimeLoop.h"
#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "Example_NSE2D.h"
#include "NavierStokes.h"
#include "Solver.h"

class TimeNavierStokesSPG : public AdaptiveTimeSystem
{
  public:
    /// @brief system on the finest grid of the domain; initialize() sets the
    /// velocity to the interpolated exact solution at time_start and computes
    /// (Ud^0, P^0) from the semidiscrete problem
    TimeNavierStokesSPG(const TDomain& domain, const ParameterDatabase& db,
                        const Example_NSE2D& example);

    void initialize() override;
    // TrapezoidalAB2_System
    void save_state() override;
    void restore_state() override;
    void predict(double k, double r) override;
    void tr_step(double t, double k) override;
    double predictor_difference_norm() override;
    void average_with_saved(double t_mid) override;
    void shift_derivative() override;
    // AdaptiveTimeSystem
    void estimate(ErrorEstimator<2>& estimator) override;
    const TCollection* get_collection() const override;
    int get_n_dofs() const override { return (int) ns.get_size(); }
    std::vector<SolutionTransfer::Snapshot> record_state() const override;
    void apply_state(const std::vector<SolutionTransfer::Snapshot>& maps) override;
    void copy_time_state(const AdaptiveTimeSystem& old) override;
    double exact_error() const override { return last_error; }
    double estimate_time() const override { return last_estimate_t; }
    /// @brief alpha min_K delta_K (tnse_dt_alpha), the lower bound of the step
    /// size for residual-based pressure stabilization (manuscript, Section 4)
    double minimal_step() const override { return dt_alpha*min_delta; }
    /// @brief L2(u), H1(u) (seminorm), L2(div u), L2(p) at the current time,
    /// the contributions of the estimator (eta_res, eta_div, eta_F, eta_delta,
    /// eta_mu) and the parts of the error in the SPG norm of the last estimate
    std::vector<std::pair<std::string, double>> error_columns() override;
    /// @brief vtk output through NavierStokes (velocity, pressure) plus the
    /// discrete time derivative
    void output(int index) override;

    NavierStokes<2>& get_navier_stokes() { return ns; }
    const BlockVector& get_time_derivative() const { return ud; }
    double get_time() const { return t; }
    int get_n_picard() const { return n_picard; }

  protected:
    NavierStokes<2> ns;
    BlockFEMatrix M, S;
    BlockVector ud, ud_old, u_saved, ud_saved, u_pred, rhs;
    std::unique_ptr<TFEVectFunct2D> ud_fct;
    Solver<BlockFEMatrix, BlockVector> solver;
    ParameterDatabase db;
    double t, last_error, nonlin_tol;
    double last_estimate_t, dt_alpha, min_delta;
    // state before the last averaging step: the estimator is evaluated with
    // the non-averaged pair (U^{n+1}, Ud^{n+1}), which solves the discrete
    // auxiliary problem; the averaged pair does not (manuscript, Remark on
    // the mesh transfer)
    BlockVector est_sol, est_ud;
    double est_t;
    bool has_est;
    std::vector<std::pair<std::string, double>> cached_columns;
    std::vector<std::pair<std::string, double>> compute_error_columns();
    void l2_projection_of_initial_velocity();
    /// contributions of the last estimate (name, value)
    std::vector<std::pair<std::string, double>> last_contributions;
    unsigned int maxit;
    int n_picard;
    size_t n_u, n_p, n_active_u;

    void assemble_mass(BlockFEMatrix& Mat, const ParameterDatabase& la_db);
    double l2_norm_velocity(const BlockVector& v);
    void zero_velocity_dirichlet(BlockVector& v);
    /// @brief velocity Dirichlet entries of v <- boundary data at 'time'
    void set_dirichlet_values(double time, BlockVector& v);
    /// @brief velocity Dirichlet entries of v <- time derivative of the
    /// boundary data at 'time' (central difference quotient, exact zero for
    /// time-independent data)
    void set_dirichlet_time_derivative(double time, BlockVector& v);
    void initial_time_derivative(double time);
    void project_pressure();
};

#endif // TIMENAVIERSTOKESSPG_H
