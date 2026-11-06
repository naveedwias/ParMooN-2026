/** ****************************************************************************
*
* @name      TNSE_ROM
* @brief     solve ROM for TNSE
*
* @note      only implemented for time-independent boundary conditions
*
* Velocity POD basis functions are supposed to be (discretely) divergence-free,
* thus the pressure term in the momentum equation drops out and the computation
* of the velocity is decoupled of the pressure. The pressure is recovered with a
* stabilization-based equation (SM-ROM approach).
* First the reduced velocity is computed from the velocity POD basis functions,
* then the reduced pressure is computed from the pressure POD basis functions.
*
* For details, see:
* "Numerical and Analytical Aspects of POD-Based Reduced-Order Modeling in
* Computational Fluid Dynamics", S. Giere, 2016.
*
*******************************************************************************/

#ifndef TNSE_ROM_H
#define TNSE_ROM_H

#include "ROM.h"

#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "DataWriter.h"
#include "deque"
#include "LocalAssembling.h"
#include "Solver.h"
#include "templateNames.h"
#include "TimeDiscretizations.h"
#include "TimeNavierStokes.h"

#include "TNSE_ROM_Assembling.h"

#ifdef __2D__
#include "Example_TimeNSE2D.h"
#include "FEFunction2D.h"
#else
#include "Example_TimeNSE3D.h"
#include "FEFunction3D.h"
#endif

template<int d>
class TNSE_ROM : public TimeNavierStokes<d>
{
  public:
    using FEFunction = typename Template_names<d>::FEFunction;
    using FEVectFunct = typename Template_names<d>::FEVectFunct;
    using FESpace = typename Template_names<d>::FESpace;
    using Example_TimeNSE = typename Template_names<d>::Example_TimeNSE;
    using SquareMatrixD = typename Template_names<d>::SquareMatrixD;
    using MatrixD = typename Template_names<d>::MatrixD;
    using BdValFunct = typename Template_names<d>::BoundaryValuesFunction;
    using BdCondFunct = typename Template_names<d>::BoundaryConditionFunction;

    /** @brief constructor
     *
     * This constructor calls the other constructor creating an
     * Example_TimeNSE object
     */
    TNSE_ROM(const TDomain& domain, const ParameterDatabase& param_db);

    /** @brief The standard constructor
     *
     * @param[in] Domain   The computational domain providing the grid
     * @param[in] param_db A parameter database with parameters concerning this
     *                     class or any of its members (fe space, solver,
     *                     assemble,...)
     * @param[in] example  The example which is to be calculated.
     */
    TNSE_ROM(const TDomain&           domain,
             const ParameterDatabase& param_db,
             const Example_TimeNSE&   ex);

    /** @brief return a database with all parameters necessary for
     *         time-dependent Navier Stokes problems with ROM
     *
     * @param[in] param_db A parameter database with parameters concerning this
     *                     class or any of its members (fe space, solver,
     *                     assemble,...)
     * @param[in] is_full  flag is set to true when used with constructor
     *                     TimeNavierStokes() in order to take account of
     *                     all parameters
     */
    static ParameterDatabase set_databaseROM(const ParameterDatabase& param_db,
                                             bool is_full=false);

    /** @brief assemble all the matrices and rhs before the time iterations
     *         and call the compute_initial_condition() routine
     */
    void assemble_initial_time();

    /** @brief assemble the matrices and rhs
     *
     * This function will assemble the rhs at each first nonlinear step.
     * In addition it calls the function prepare_system_velocity() where the
     * convective nonlinear term is updated.
     *
     * @param it_counter nonlinear iteration number
     */
    void assemble_matrices_rhs(unsigned int it_counter);

    void set_fluctuations();

    /** @brief solve the system
     *
     * @param[in] is_velocity flag to select if velocity (true)
     *                        or pressure (false) has to be computed
     */
    void solve(bool is_velocity);

    /** @brief check if one of the stopping criteria is fulfilled
     *
     * either converged or maximun number of iterations reached
     *
     * @param it_counter current iterate
     */
    bool stop_it(unsigned int it_counter);

    /** @brief get the current reduced velocity residual
     */
    double get_velocity_residual() const
    { return residual_u_r; }

    /** @brief print out the residual of the corresponding fem system computed
     *         from the parent class TimeNavierStokes<d>
     */
    void print_fem_residuals(int loop_index);

  private:

    std::unique_ptr<TNSE_ROM_Assembling<d>> assembler;

    /** @brief ROM related to the velocity components */
    ROM ROM_u;
    /** @brief ROM related to the pressure */
    ROM ROM_p;

    /** @brief a local parameter database which controls this class
     *
     * The database given to the constructor will be merged into this one. Only
     * parameters which are of interest to this class are stored (and the
     * default ParMooN parameters). Note that this usually does not include
     * other parameters such as solver parameters. Those are only in the
     * Solver object.
     */
    ParameterDatabase db;

    /** @brief rank of the velocity POD basis*/
    int rank_u;

    /// @brief step matrix (convection + diffusion)
    DenseMatrix step_mat_u_r;

    /// @brief step matrix (convection + diffusion) at the previous time step
    DenseMatrix step_mat_m1_u_r;

    /** @brief Reduced system matrix */
    DenseMatrix sys_mat_u_r;

    /** @brief Reduced system rhs */
    std::vector<double> sys_rhs_u_r;

    /** @brief Linear parts of the reduced system rhs */
    DenseMatrix rhs_mat_u_r;

    /** @brief Reduced mass matrix */
    DenseMatrix mass_mat_u_r;

    /** @brief Reduced source term */
    std::vector<double> rhs_src_u_r;

    /** @brief Reduced mass term */
    std::vector<double> rhs_mass_u_r;

    /** @brief Reduced solution, i.e. ROM solution */
    std::vector<double> sol_u_r;

    /** @brief Reduced source term at the last time step for Crank Nicolson*/
    std::vector<double> rhs_src_m1_u_r;

    /** @brief Reduced mass term at the last time step */
    std::vector<double> rhs_mass_m1_u_r;

    /** @brief Reduced solution at the last time step */
    std::vector<double> sol_m1_u_r;

    /** @brief Reduced solution at the penultimate time step for BDF2*/
    std::vector<double> sol_m2_u_r;

    /** @brief rank of the pressure POD basis*/
    int rank_p;

    /** @brief Reduced system matrix */
    DenseMatrix sys_mat_p_r;

    /** @brief Reduced system rhs */
    std::vector<double> sys_rhs_p_r;

    /** @brief Reduced SM-ROM diffusion matrix*/
    DenseMatrix diff_mat_p_r;

    /** @brief SM_ROM contribution due to the mean fluctuation correction*/
    std::vector<double> rhs_mean_p_r;

    /** @brief Reduced solution, i.e. ROM solution */
    std::vector<double> sol_p_r;

    std::vector<double> sol_fluct;
    FEVectFunct u_fluct_function;
    FEFunction p_fluct_function;

    /** @brief reduced velocity residual */
    double residual_u_r;

    /** @brief coeffcients of the BDF2 scheme */
    std::vector<double> bdf2_coeffs;

    /** @brief For the time discretization in case of BDF2 scheme at the first
     *         step which need the use of an other time scheme (BDF1 or CN)
     */
    bool pre_stage_bdf = false;

    /// @brief Reynolds number
    double reynolds = 1.0;

    /** @brief flag indicating if the corresponding FEM residuals have
     *         to be computed and printed out.
     */
    bool fem_residuals = false;

    std::shared_ptr<FEMatrix> pressure_mass;
    std::vector<std::shared_ptr<FEMatrix>> divergence_matrix;
    BlockVector divergence_product;
    BlockVector divergence_product_2;

    /** @brief check and set parameters in database
     *
     * This functions checks if the parameters in the database are meaningful
     * and resets them otherwise. The hope is that after calling this function
     * this class is fully functional.
     *
     * If some parameters are set to unsupported values, an error occurs and
     * throws an exception.
     */
    void check_and_set_parameters();

    /** @brief Compute ROM initial condition
     *
     * The ROM initial condition is computed in the standard way: euclidean, L2
     * or H1 projection into the POD space. In case of L2 or H1 projection,
     * the gramian matrix (inner product matrix) is temporary stored in the
     * BlockFEMatrix mass_matrix (matrix respectively) from the parent class
     * TimeNavierStokes for the velocity
     * or a temporary BlockFEMatrix for the pressure
     */
    void compute_initial_condition();

    /** @brief Compute the defect Ax-b and store its norm.
     */
    void compute_residuals(void);

};

#endif // TNSE_ROM_H
