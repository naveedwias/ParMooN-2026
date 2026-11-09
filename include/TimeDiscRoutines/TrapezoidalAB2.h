// =============================================================================
// Variable step trapezoidal rule with AB2 error estimation and stabilization
// (Gresho, Griffiths, Silvester, SIAM J. Sci. Comput. 30 (2008) 2018-2054),
// as used by de Frutos, Garcia-Archilla, John, Novo (CMAME 273 (2014),
// Section 3.3) and by the adaptive SUPG/PSPG/grad-div Navier-Stokes scheme.
//
// For a system M u' + A u = f carrying the discrete time derivative d = u':
//   TR:    (2/k M + A) u^{n+1} = f + M (2/k u^n + d^n),  d^{n+1} = 2/k (u^{n+1}-u^n) - d^n
//   AB2:   u_p = u^n + k/2 [(2 + r) d^n - r d^{n-1}],  r = k/k_old
//   LTE:   e = (u^{n+1} - u_p)/(3 (1 + 1/r)), accepted if |e| <= TOLT
//   step:  k_new = factor * k (TOLT/|e|)^{1/3} (factor 0.85: de Frutos et al.),
//          rejected steps are repeated with the reduced step
//   stabilization: every n* accepted steps u^{n+1} <- (u^{n+1}+u^n)/2,
//          d^{n+1} <- (d^{n+1}+d^n)/2, t <- t - k/2, k_old <- k/2
//          ('periodic', GGS 2008 and de Frutos et al. 2014), or the same only
//          if the state has been perturbed since the last averaging, i.e.
//          after the start and after mesh changes ('on_demand', see
//          restart_averaging()).
//
// Why averaging: TR is A-stable but not L-stable; a component of the initial
// value (or of a state interpolated to a new mesh) in a stiff mode, e.g. the
// part of an interpolated velocity which is not discretely divergence-free,
// is carried on with amplification factor ~ -1. The AB2 estimate then sees a
// k-independent error, the step size can only grow slowly and with small
// TOLT the control stalls. Averaging two consecutive states removes the
// alternating component. Each averaging is a local O(k^2) perturbation, so
// periodic averaging costs accuracy for smooth problems whose start is
// consistent; 'on_demand' averages once after each perturbation.
//
// The class owns the time, the step sizes and the counters; the discrete
// vectors (u, d, d_old and the saved previous values) are owned by a system
// implementing TrapezoidalAB2_System, which also knows how to assemble and
// solve one TR step and which norm to use for the error estimate.
// =============================================================================
#ifndef TRAPEZOIDALAB2_H
#define TRAPEZOIDALAB2_H

#include "ParameterDatabase.h"

class TrapezoidalAB2_System
{
  public:
    virtual ~TrapezoidalAB2_System() = default;
    /// @brief save u, d (for a possible rejection and for the AB2 predictor)
    virtual void save_state() = 0;
    /// @brief restore the saved u, d (rejected step)
    virtual void restore_state() = 0;
    /// @brief AB2 predictor into a member vector: u_p = u + k/2 [(2+r) d - r d_old]
    virtual void predict(double k, double r) = 0;
    /// @brief one TR step of length k from time t: assemble, solve, update u
    /// and d (the saved state holds u^n, d^n)
    virtual void tr_step(double t, double k) = 0;
    /// @brief L2 norm of u - u_p (the AB2 error estimate before scaling)
    virtual double predictor_difference_norm() = 0;
    /// @brief u <- (u + u_saved)/2, d <- (d + d_saved)/2; the averaged state
    /// belongs to the midpoint t_mid of the last step, which the system must
    /// take as its current time (data, boundary values, estimator, output)
    virtual void average_with_saved(double t_mid) = 0;
    /// @brief d_old <- d_saved (after an accepted step)
    virtual void shift_derivative() = 0;
};

class TrapezoidalAB2
{
  public:
    static ParameterDatabase default_database();

    explicit TrapezoidalAB2(const ParameterDatabase& db);

    /// @brief current time
    double get_time() const { return t; }
    void set_time(double time) { t = time; }
    /// @brief next step size (proposal), previous accepted (effective) step
    double get_step() const { return k; }
    double get_previous_step() const { return k_old; }
    void set_step(double step) { k = step; }
    void set_previous_step(double step) { k_old = step; }
    bool is_fixed_step() const { return fixed_step; }
    int n_accepted_steps() const { return n_accepted; }
    int n_rejected_steps() const { return n_rejected; }
    double get_end_time() const { return t_end; }
    bool finished() const { return t >= t_end - 1e-12; }

    /// @brief the state has been perturbed (e.g. transferred to a new mesh):
    /// in 'on_demand' mode the next averaging (after n* steps) is enabled
    void restart_averaging() { averaging_pending = true; }

    /** @brief advance the system by one accepted step (possibly after
     * rejections). Returns the error estimate of the accepted step; the
     * effective step length of the accepted step is get_previous_step()
     * afterwards (half the step if the step was averaged). */
    double advance(TrapezoidalAB2_System& sys);

  protected:
    double t, t_end, k, k_old;
    double tolt, k_min, k_max, factor;
    int n_star, n_accepted, n_rejected, n_since_avg;
    bool fixed_step, first_step, periodic_averaging, averaging_pending;
};

#endif // TRAPEZOIDALAB2_H
