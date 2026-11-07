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
    /// @brief system on the finest grid; velocity = interpolated exact
    /// solution at time_start, (Ud^0, P^0) from the semidiscrete problem
    TimeNavierStokesSPG(const TDomain& domain, const ParameterDatabase& db,
                        const Example_NSE2D& example);

    // TrapezoidalAB2_System
    void save_state() override;
    void restore_state() override;
    void predict(double k, double r) override;
    void tr_step(double t, double k) override;
    double predictor_difference_norm() override;
    void average_with_saved() override;
    void shift_derivative() override;
    // AdaptiveTimeSystem
    void estimate(ErrorEstimator<2>& estimator) override;
    const TCollection* get_collection() const override;
    int get_n_dofs() const override { return (int) ns.get_size(); }
    std::vector<SolutionTransfer::NodeValueMap> record_state() const override;
    void apply_state(const std::vector<SolutionTransfer::NodeValueMap>& maps) override;
    void copy_time_state(const AdaptiveTimeSystem& old) override;
    double exact_error() const override { return last_error; }
    /// @brief vtk output through NavierStokes (velocity, pressure) plus the
    /// discrete time derivative
    void output(int index) override;

    NavierStokes<2>& get_navier_stokes() { return ns; }
    const BlockVector& get_time_derivative() const { return ud; }
    double get_time() const { return t; }
    int get_n_picard() const { return n_picard; }

  protected:
    NavierStokes<2> ns;
    BlockFEMatrix M, M_galerkin, S;
    BlockVector ud, ud_old, u_saved, ud_saved, u_pred, rhs;
    std::unique_ptr<TFEVectFunct2D> ud_fct;
    Solver<BlockFEMatrix, BlockVector> solver;
    ParameterDatabase db;
    double t, last_error, nonlin_tol;
    unsigned int maxit;
    int n_picard;
    size_t n_u, n_p, n_active_u;

    void assemble_mass(BlockFEMatrix& Mat, const ParameterDatabase& la_db);
    double l2_norm_velocity(const BlockVector& v);
    void zero_velocity_dirichlet(BlockVector& v);
    void initial_time_derivative(double time);
    void project_pressure();
};

#endif // TIMENAVIERSTOKESSPG_H
