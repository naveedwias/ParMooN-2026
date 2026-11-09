// =============================================================================
// SUPG semidiscretization of the time-dependent convection-diffusion-reaction
// equation for the TR-AB2 integrator and the adaptive time loop:
//     M_s u' + A_s u = f_s,
//     M_s = (u,v) + (u, delta b.grad v)_h,  A_s = a_SUPG,  f_s = (f,v) + (f, delta b.grad v)_h,
// with delta_K = delta_0 h_K/|b| (SDFEM_TYPE 0, the time-dependent scaling of
// ParMooN's TCD routines is switched off by assembling with THETA1 = 1 and
// tau = 1). The discrete time derivative d = u' is carried along (consistent
// TR derivative) and used by the error estimator (CDErrorEstimator type 5,
// residual f - d_t u_h + ...), see de Frutos, Garcia-Archilla, John, Novo
// (2014).
// =============================================================================
#ifndef TIMECONVECTIONDIFFUSIONTR_H
#define TIMECONVECTIONDIFFUSIONTR_H

#include "AdaptiveTimeLoop.h"
#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "Example_CD2D.h"
#include "Example_TimeCD2D.h"
#include "FEFunction2D.h"
#include "FESpace2D.h"
#include "Solver.h"
#include "DataWriter.h"
#include <memory>

class TimeConvectionDiffusionTR : public AdaptiveTimeSystem
{
  public:
    /// @brief system on the finest grid of the domain; initialize() sets the
    /// interpolated initial condition (Dirichlet dofs from the boundary data)
    /// and computes the time derivative from the equation at time_start
    TimeConvectionDiffusionTR(const TDomain& domain, const ParameterDatabase& db,
                              const Example_TimeCD2D& example);

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
    int get_n_dofs() const override { return (int) u.length(); }
    std::vector<SolutionTransfer::Snapshot> record_state() const override;
    void apply_state(const std::vector<SolutionTransfer::Snapshot>& maps) override;
    void copy_time_state(const AdaptiveTimeSystem& old) override;
    double exact_error() const override;
    /// @brief L2(u), H1(u) (seminorm) at the current time
    std::vector<std::pair<std::string, double>> error_columns() override;
    /// @brief vtk output of the solution and the discrete time derivative
    void output(int index) override;

    const TFEFunction2D& get_function() const { return *u_fct; }
    const TFEFunction2D& get_time_derivative_function() const { return *d_fct; }
    double get_time() const { return t; }
    void set_time(double time) { t = time; }
    std::shared_ptr<const TFESpace2D> get_space() const { return space; }

  protected:
    std::shared_ptr<const TFESpace2D> space;
    const Example_TimeCD2D& example;
    Example_CD2D steady_example;
    ParameterDatabase db;
    BlockFEMatrix M, A, S;
    BlockVector u, d, d_old, u_saved, d_saved, u_pred, rhs, work;
    std::unique_ptr<TFEFunction2D> u_fct, d_fct;
    Solver<BlockFEMatrix, BlockVector> solver;
    DataWriter<2> writer;
    double t;

    void assemble(double time);
    /// @brief Dirichlet entries of v <- boundary data at 'time'
    void set_dirichlet_values(double time, BlockVector& v);
    /// @brief Dirichlet entries of v <- time derivative of the boundary data
    /// (central difference quotient, exact zero for time-independent data)
    void set_dirichlet_time_derivative(double time, BlockVector& v);
    void compute_time_derivative(double time);
    double l2_norm(const BlockVector& v);
};

#endif // TIMECONVECTIONDIFFUSIONTR_H
