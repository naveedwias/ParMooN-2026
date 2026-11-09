// =============================================================================
// Time loop with the TR-AB2 integrator and adaptive meshes (refinement and
// coarsening after every accepted step, as in de Frutos, Garcia-Archilla,
// John, Novo, CMAME 273 (2014), Section 3.3):
//
//   (1) one TR-AB2 step on the current mesh,
//   (2) a posteriori estimator with the discrete time derivative,
//   (3) cells with eta_K > TOL1 are refined, parents whose children all have
//       eta_K < TOL2 are coarsened (TDomain::RefineAndCoarsen),
//   (4) the system is rebuilt on the new mesh and u, d, d_old (and the
//       pressure) are transferred (SolutionTransfer),
//   (5) continue until the final time.
//
// The loop is independent of the PDE: a system implements AdaptiveTimeSystem
// (the TR-AB2 interface plus estimation, state transfer and bookkeeping),
// and a factory creates it on the current finest grid of the domain.
// =============================================================================
#ifndef ADAPTIVETIMELOOP_H
#define ADAPTIVETIMELOOP_H

#include "Domain.h"
#include "ErrorEstimator.h"
#include "ParameterDatabase.h"
#include "SolutionTransfer.h"
#include "TrapezoidalAB2.h"
#include <functional>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

class AdaptiveTimeSystem : public TrapezoidalAB2_System
{
  public:
    /// @brief estimate the error with the given estimator (the system passes
    /// its solution and discrete time derivative in the way the estimator
    /// expects)
    virtual void estimate(ErrorEstimator<2>& estimator) = 0;
    /// @brief the collection of the mesh the system lives on
    virtual const TCollection* get_collection() const = 0;
    virtual int get_n_dofs() const = 0;
    /// @brief all data needed on a new mesh (u, d, d_old, ...), recorded by
    /// SolutionTransfer before the mesh changes
    virtual std::vector<SolutionTransfer::Snapshot> record_state() const = 0;
    virtual void apply_state(const std::vector<SolutionTransfer::Snapshot>& maps) = 0;
    /// @brief copy the time integration state from the system on the old mesh
    virtual void copy_time_state(const AdaptiveTimeSystem& old) = 0;
    /// @brief error in the problem's norm against the exact solution, or a
    /// negative value if unknown
    virtual double exact_error() const = 0;
    /// @brief output (vtk etc.) at the current time with a running index
    /// (the index is kept by the loop, since the system is rebuilt on every
    /// mesh change)
    virtual void output(int index) = 0;
};

class AdaptiveTimeLoop
{
  public:
    static ParameterDatabase default_database();

    typedef std::function<std::unique_ptr<AdaptiveTimeSystem>()> SystemFactory;

    /// @param factory creates the system on the current finest grid of the
    /// domain; it is called once at the start and after every mesh change
    AdaptiveTimeLoop(TDomain& domain, const ParameterDatabase& db,
                     ErrorEstimator<2>& estimator, SystemFactory factory);

    /// @brief run until the final time; the initial system must have been
    /// set up (initial condition, initial time derivative) by the factory
    void run();

    /// @brief the table (one line per accepted step)
    const std::string get_table() const { return table.str(); }
    AdaptiveTimeSystem& get_system() { return *system; }
    TrapezoidalAB2& get_integrator() { return integrator; }

  protected:
    TDomain& domain;
    ParameterDatabase db;
    ErrorEstimator<2>& estimator;
    SystemFactory factory;
    TrapezoidalAB2 integrator;
    std::unique_ptr<AdaptiveTimeSystem> system;
    std::ostringstream table;
    std::ofstream table_file;
    bool adaptive, coarsening, write_ps;
    double tol1, tol2, output_every, next_output;
    int initial_level, output_index;
    size_t max_level;
};

#endif // ADAPTIVETIMELOOP_H
