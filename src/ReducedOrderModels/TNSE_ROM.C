/** ****************************************************************************
*
* @name      TNSE_ROM
* @brief     solve ROM for TNSE
*
*******************************************************************************/

#include "CD_local_assembling_routines.h" // for P-ROM gramian
#include "ConvDiff.h"
#include "Database.h"
#include "LinAlg.h"
#include "MainUtilities.h"
#include "ROM_local_assembling_routines.h"
#include "TNSE_ROM.h"
#include "ReducedHelmholtz.h"
#include "Simplified/SimplifiedInfo.h"
#include "Utilities.h"

#ifdef __2D__
#include "Assemble2D.h"
#include "AuxParam2D.h"
#include "Matrix2D.h"
#include "SquareMatrix2D.h"
#else
#include "Assemble3D.h"
#include "AuxParam3D.h"
#include "Matrix3D.h"
#include "SquareMatrix3D.h"
#endif

#include <algorithm>
#include <memory>

/** ***************************************************************************/
template <int d>
ParameterDatabase TNSE_ROM<d>::set_databaseROM(
                                              const ParameterDatabase& param_db,
                                              bool is_full)
{
  ParameterDatabase db("Parameter database for TNSE_ROM");

  db.add("vp_rom_rank", std::vector<size_t>(2, 0),
         "These integers specify the dimensions of POD-basis to be used for "
         "the u-ROM and p-ROM spaces, respectively. "
         "If = 0, then all possible POD modes will be used.");

  db.add("rom_write_fluctuations", false,
         "Write fluctuations?",
         { true, false } );

  db.add("rom_compute_divergence", false,
         "Compute and write divergence?",
         { true, false } );

  db.add("rom_cache_velocity_assembly", true,
         "Accelerate online velocity assembly by caching partials?",
         { true, false } );

  db.add("rom_cache_pressure_assembly", true,
         "Accelerate online pressure assembly by caching partials?",
         { true, false } );

//   db.add("print_fem_residuals", false,
//          "This is the flag indicating if the corresponding FEM residuals have "
//          " to be computed and printed out.",
//          {true,false});

  db.merge(TimeNavierStokes<d>::default_tnse_database(), true);

  db.merge(LocalAssembling<d>::default_local_assembling_database(), true);

  db.merge(ParameterDatabase::default_output_database(), true);
  db.merge(TimeDiscretization::default_TimeDiscretization_database(), true);
  db.merge(ROM::default_rom_database(), true);

  if (is_full)
  {
    db.merge(param_db, true);
  }
  else
  {
    db.merge(param_db, false);
  }

  // change name to avoid overwritting standard FEM output or POD-basis
  // when passed to constructor TimeNavierStokes()
  // TODO: unification between naming system (outfile, output_basename,
  //       snaps_basename, pod_basename)
  std::string output_rom_basename = db["output_basename"].get<std::string>()
                                    + "_rom";
  db["output_basename"].set(output_rom_basename, false);

  return db;
}

/** ***************************************************************************/
template<int d>
TNSE_ROM<d>::TNSE_ROM(const TDomain&           domain,
                      const ParameterDatabase& param_db)
  : TNSE_ROM<d>(domain, param_db, Example_TimeNSE(param_db))
{
}

/** ***************************************************************************/
template<int d>
TNSE_ROM<d>::TNSE_ROM(const TDomain&           domain,
                      const ParameterDatabase& param_db,
                      const Example_TimeNSE&   ex)
  : TimeNavierStokes<d>(domain, TNSE_ROM<d>::set_databaseROM(param_db,true), ex),
    ROM_u(param_db,
          TNSE_ROM<d>::set_databaseROM(param_db)["vp_rom_rank"].
            Parameter::get<std::vector<size_t>>()[0], "u"),
    ROM_p(param_db,
          TNSE_ROM<d>::set_databaseROM(param_db)["vp_rom_rank"].
            Parameter::get<std::vector<size_t>>()[1], "p"),
    db(TNSE_ROM<d>::set_databaseROM(param_db, true)),
    rank_u(ROM_u.get_rank()),
    step_mat_u_r(rank_u, rank_u),
    step_mat_m1_u_r(rank_u, rank_u),
    sys_mat_u_r(rank_u, rank_u),
    sys_rhs_u_r(rank_u),
    rhs_mat_u_r(rank_u, rank_u),
    mass_mat_u_r(rank_u, rank_u),
    rhs_src_u_r(rank_u),
    rhs_mass_u_r(rank_u),
    sol_u_r(rank_u),
    rhs_src_m1_u_r(rank_u),
    rhs_mass_m1_u_r(rank_u),
    sol_m1_u_r(rank_u),
    sol_m2_u_r(rank_u),
    rank_p(ROM_p.get_rank()),
    sys_mat_p_r(rank_p, rank_p),
    sys_rhs_p_r(rank_p),
    diff_mat_p_r(rank_p, rank_p),
    rhs_mean_p_r(rank_p),
    sol_p_r(rank_p)
{
  check_and_set_parameters();

  if (db["time_discretization"].is("crank_nicolson"))
  {
    rhs_src_m1_u_r.resize(rank_u);
  }

  if (db["time_discretization"].is("bdf_two"))
  {
    sol_m2_u_r.resize(rank_u);
    bdf2_coeffs = this->get_time_stepping_scheme().get_bdf_coefficients();
  }

  auto& s = this->systems.front();

  if (db["rom_write_fluctuations"])
  {
    sol_fluct.resize(s.solution.length());

    u_fluct_function = FEVectFunct(s.velocity_space, "u_f", sol_fluct.data(), d);
    p_fluct_function = FEFunction(s.pressure_space, "p_f", sol_fluct.data() + d * s.velocity_space->get_n_dof());

    this->outputWriter.add_fe_vector_function(&this->u_fluct_function);
    this->outputWriter.add_fe_function(&this->p_fluct_function);
  }

  if (db["rom_compute_divergence"])
  {
    divergence_matrix = SimplifiedAssembling<d>::AssembleDivProduct(s.pressure_space, s.velocity_space);
    pressure_mass = SimplifiedAssembling<d>::AssembleMass(s.pressure_space);

    BlockFEMatrix MQ({ s.pressure_space });
    MQ.replace_blocks(*pressure_mass, { {0, 0} }, { false });

    divergence_product = BlockVector(MQ, true);
    divergence_product_2 = BlockVector(MQ, true);

    this->outputWriter.add_global_output_variable("div_norm", [&, MQ] () -> double
      {
        Output::root_info("TNSE_ROM", "Computing divergence.");

        using namespace utilities;

        divergence_product.reset();

        for (int k = 0; k < d; k++)
        {
          divergence_matrix[k]->multiply(s.solution.block(k), divergence_product.get_entries(), 1.0);
        }

        Solver<BlockFEMatrix, BlockVector>(SimplifiedInfo::default_direct_solver_db()).solve(MQ, divergence_product, divergence_product_2);

        return std::sqrt(dot(divergence_product, divergence_product_2));
      });
  }

  assembler.reset(new TNSE_ROM_Assembling<d>(
    s.velocity_space, s.pressure_space,
    rank_u, rank_p,
    ROM_u.get_basis()->get_columns(),
    ROM_p.get_basis()->get_columns(),
    { ROM_u.get_snaps_avr_ptr() }));

  if (db["rom_cache_velocity_assembly"])
  {
    assembler->AssembleVelocityCache();
  }

  if (db["rom_cache_pressure_assembly"])
  {
    assembler->AssemblePressureCache();
  }
}

/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::check_and_set_parameters()
{
  if (db["pod_fluctuations_only"].is(false))
  {
    ErrThrow("The parameter 'pod_fluctuations_only' has to be activated "
             "otherwise only problems with homogeneous boundary conditions "
             "can be solved.");
  }

  if (this->solver.is_using_multigrid())
  {
    ErrThrow("Multigrid for ROM is not yet enabled in TimeNavierStokes");
  }

  if (!db["space_discretization_type"].is("galerkin"))
  {
    ErrThrow("The space discretization ", db["space_discretization_type"],
             " is not yet enabled in TimeNavierStokes for ROM computation.");
  }

  if (this->get_db()["time_discretization_nonlinear_term"].is("fully_explicit"))
  {
    ErrThrow("The fully explicit time discretization for the nonlinear term, "
             "is not yet enabled in TimeNavierStokes for ROM computation.");
  }

  // if (db["nse_nonlinear_form"].is("divergence"))
  if (!db["nse_nonlinear_form"].is("convective"))
  {
    ErrThrow("The nonlinear form ", db["nse_nonlinear_form"],
             " is not yet enabled in TimeNavierStokes for ROM computation.");
  }

  if (!db["graddiv_stab"].is(0.))
  {
    Output::warn("Stabilization parameters are not yet supported for ROM-TNSE. "
                 "The parameter 'graddiv_stab' is reset to 0 ");
    db["graddiv_stab"] = 0.;
  }

  if (db["mat_time_dependent"].is(true))
  {
    Output::warn("The parameter 'mat_time_dependent' is not yet implemented "
                 "for ROM-TNSE");
  }

#ifdef _MPI
  ErrThrow("MPI is not yet enabled for TimeNavierStokes with ROM");
#endif

  // if (db["print_fem_residuals"])
  // {
  //   fem_residuals = true;
  // }

  if (db["time_discretization"].is("bdf_two"))
  {
    ErrThrow("BDF-2 is not yet supported for ROM-TNSE");

    pre_stage_bdf = true;
  }

  reynolds = db["reynolds_number"];
}

/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::compute_initial_condition()
{
  auto& s = this->systems.front();

  std::vector<double> fluct_velocity(d * this->get_velocity_space()->get_n_dof());
  std::vector<double> fluct_pressure(this->get_pressure_space()->get_n_dof());

  double* u_0 = s.solution.block(0);
  double* p_0 = s.solution.block(d);

  const double* u_mean = ROM_u.get_snaps_avr_ptr();
  const double* p_mean = ROM_p.get_snaps_avr_ptr();

  for (unsigned int i = 0; i < fluct_velocity.size(); i++)
  {
    fluct_velocity[i] = u_0[i] - u_mean[i];
  }

  for (unsigned int i = 0; i < fluct_pressure.size(); i++)
  {
    fluct_pressure[i] = p_0[i] - p_mean[i];
  }

  if (db["rom_init_inner_product"].is("euclidean"))
  {
    assembler->ProjectPressureEuclidean(fluct_pressure.data(), sol_p_r.data());
    assembler->ProjectVelocityEuclidean(fluct_velocity.data(), sol_u_r.data());
  }
  else if (db["rom_init_inner_product"].is("L2"))
  {
    assembler->ProjectPressureL2(fluct_pressure.data(), sol_p_r.data());
    assembler->ProjectVelocityL2(fluct_velocity.data(), sol_u_r.data());
  }
  else if (db["rom_init_inner_product"].is("H1"))
  {
    assembler->ProjectPressureH1(fluct_pressure.data(), sol_p_r.data());
    assembler->ProjectVelocityH1(fluct_velocity.data(), sol_u_r.data());
  }
  else
  {
    ErrThrow("Unknown inner product: ", db["rom_init_inner_product"]);
  }

  auto err = assembler->VelocityErrors(fluct_velocity.data(), sol_u_r.data());
  Output::info("TNSE_ROM_Assembling", "Initial condition Euclidean error: ", err[0]);
  Output::info("TNSE_ROM_Assembling", "Initial condition L2 error: ", err[1]);
  Output::info("TNSE_ROM_Assembling", "Initial condition H1 error: ", err[2]);

  if (db["rom_init_regularized"])
  {
    double mu = db["differential_filter_width"];
    bool scaled = db["differential_filter_local_scale"];

    Output::root_info("TNSE_ROM", "Applying differential filter to initial "
      "condition, mu = ", mu, scaled ? " * h_K" : "");

    ReducedHelmholtz<d>(std::vector<std::shared_ptr<const FESpace>>(d, this->get_velocity_space()),
      rank_u, ROM_u.get_basis()->get_columns())
      .Filter(sol_u_r.data(), sol_u_r.data(), mu, scaled);

    ReducedHelmholtz<d>({ this->get_pressure_space() },
      rank_p, ROM_p.get_basis()->get_columns())
      .Filter(sol_p_r.data(), sol_p_r.data(), mu, scaled);

    err = assembler->VelocityErrors(fluct_velocity.data(), sol_u_r.data());
    Output::info("TNSE_ROM_Assembling", "Filtered initial condition Euclidean error: ", err[0]);
    Output::info("TNSE_ROM_Assembling", "Filtered initial condition L2 error: ", err[1]);
    Output::info("TNSE_ROM_Assembling", "Filtered initial condition H1 error: ", err[2]);
  }

  ROM_u.get_full_solution(this->sol_u_r, u_0);
  ROM_p.get_full_solution(this->sol_p_r, p_0);

  sol_m1_u_r = sol_u_r;
  sol_m2_u_r = sol_u_r;

  s.solution_m1 = s.solution;
  s.solution_m2 = s.solution;

  set_fluctuations();
}

/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::assemble_initial_time()
{
  // velocity mass
  assembler->AssembleVelocityL2(1.0, mass_mat_u_r.get_columns());

  // pressure mass and mean rhs
  assembler->AssemblePressureFixedMatrixRHS_SM_ROM(
    ROM_p.get_snaps_avr_ptr(),
    1.0, diff_mat_p_r.get_columns(), rhs_mean_p_r.data());

  compute_initial_condition();

  // for FEM residual computation
  // if (fem_residuals)
  // {
  //   TimeNavierStokes<d>::assemble_initial_time();
  // }
}

/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::assemble_matrices_rhs(unsigned int it_counter)
{
  using namespace utilities;

  bool first_nonlinear_step = (it_counter == 0);

  if (first_nonlinear_step)
  {
    step_mat_m1_u_r = step_mat_u_r;
    rhs_src_m1_u_r = rhs_src_u_r;

    sol_m2_u_r = sol_m1_u_r;
    sol_m1_u_r = sol_u_r;

    if (db["extrapolation_type"].is("linear"))
    {
      // linear extrapolation to start Picard iteration
      scale_vec(sol_u_r, 2.0);
      add_vec(sol_u_r, sol_m2_u_r, -1.0);
    }
    else if (!db["extrapolation_type"].is("constant"))
    {
      ErrThrow("Unsupported extrapolation_type: ", db["extrapolation_type"]);
    }
  }

  assembler->ClearVelocityMatrixRHS(step_mat_u_r.get_columns(), rhs_src_u_r.data());

  if (db["rom_cache_velocity_assembly"])
  {
    assembler->AssembleVelocityStepMatrixRHS(
      1.0 / reynolds,
      sol_u_r.data(), { 1.0 },
      1.0, step_mat_u_r.get_columns(), rhs_src_u_r.data()
      );

    clear_vec(rhs_mass_u_r);

    assembler->AssembleVelocityMassRHS(
      { 1.0 },
      1.0, rhs_mass_u_r.data());
  }
  else
  {
    assembler->AssembleVelocityStepMatrixRHS(
      1.0 / reynolds,
      sol_u_r.data(), ROM_u.get_snaps_avr_ptr(),
      1.0, step_mat_u_r.get_columns(), rhs_src_u_r.data()
      );
  }

  pre_stage_bdf = this->get_time_stepping_scheme().pre_stage_bdf;

  double tau = this->get_time_stepping_scheme().get_step_length();
  double theta = -1.0;

  if (db["time_discretization"].is("backward_euler")
    || (pre_stage_bdf && db["bdf_two_first_step"].is("backward_euler")))
  {
    theta = 1.0;
  }
  else if (db["time_discretization"].is("crank_nicolson")
    || (pre_stage_bdf && db["bdf_two_first_step"].is("crank_nicolson")))
  {
    theta = 0.5;
  }
  else if (db["time_discretization"].is("bdf_two") && !pre_stage_bdf)
  {
    ErrThrow("BDF-2 is not currently supported!");
  }
  else
  {
    if (db["time_discretization"].is("bdf_two"))
    {
      ErrThrow("Unsupported BDF-2 first step time discretization: ", db["bdf_two_first_step"]);
    }
    else
    {
      ErrThrow("Unsupported time discretization: ", db["time_discretization"]);
    }
  }

  if (theta >= 0.0)
  {
    if (theta < 1.0 && first_nonlinear_step)
    {
      // reassemble from output of last iteration (existing values are from next-to-last)

      assembler->ClearVelocityMatrixRHS(step_mat_m1_u_r.get_columns(), rhs_src_m1_u_r.data());

      if (db["rom_cache_velocity_assembly"])
      {
        assembler->AssembleVelocityStepMatrixRHS(
          1.0 / reynolds,
          sol_m1_u_r.data(), { 1.0 },
          1.0, step_mat_m1_u_r.get_columns(), rhs_src_m1_u_r.data()
          );

        clear_vec(rhs_mass_m1_u_r);

        assembler->AssembleVelocityMassRHS(
          { 1.0 },
          1.0, rhs_mass_m1_u_r.data());
      }
      else
      {
        assembler->AssembleVelocityStepMatrixRHS(
          1.0 / reynolds,
          sol_m1_u_r.data(), ROM_u.get_snaps_avr_ptr(),
          1.0, step_mat_m1_u_r.get_columns(), rhs_src_m1_u_r.data()
          );
      }
    }

    sys_mat_u_r = mass_mat_u_r;
    sys_mat_u_r.add(step_mat_u_r, theta * tau);

    rhs_mat_u_r = mass_mat_u_r;
    rhs_mat_u_r.add(step_mat_m1_u_r, -(1.0 - theta) * tau);

    sys_rhs_u_r.resize(rhs_src_u_r.size());
    clear_vec(sys_rhs_u_r);

    rhs_mat_u_r.multiply(sol_m1_u_r, sys_rhs_u_r);

    add_vec(sys_rhs_u_r, rhs_src_u_r, theta * tau);
    add_vec(sys_rhs_u_r, rhs_src_m1_u_r, (1.0 - theta) * tau);

    add_vec(sys_rhs_u_r, rhs_mass_u_r, -1.0);
    add_vec(sys_rhs_u_r, rhs_mass_m1_u_r, 1.0);
  }

  // for FEM residual computation
  // if (fem_residuals)
  // {
  //   TimeNavierStokes<d>::assemble_matrices_rhs(it_counter);
  // }
}

/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::solve(bool is_velocity)
{
  if (is_velocity)
  {
    if (sys_mat_u_r.getNRows() != (int)sys_rhs_u_r.size())
    {
      ErrThrow("Dimension mismatch during A*x = rhs for velocity resolution: ",
              sys_mat_u_r.getNRows(), " rows in the system matrix and ",
              sys_rhs_u_r.size(), " elements in the rhs.");
    }

    sys_mat_u_r.decomposeLU();

    sys_mat_u_r.solve(sys_rhs_u_r.data(), sol_u_r.data());
  }
  else
  {
    if (!sys_mat_p_r.getEntriesLU())
    {
      sys_mat_p_r = diff_mat_p_r;
      sys_mat_p_r.decomposeLU();
    }

    sys_rhs_p_r = rhs_mean_p_r;

    if (db["rom_cache_pressure_assembly"])
    {
      assembler->AssemblePressureStepRHS_SM_ROM(
        this->get_time_stepping_scheme().get_step_length(),
        { 1.0 }, { 1.0 },
        sol_m1_u_r.data(), sol_u_r.data(),
        1.0, sys_rhs_p_r.data());
    }
    else
    {
      assembler->AssemblePressureStepRHS_SM_ROM(
        this->get_time_stepping_scheme().get_step_length(),
        this->systems.front().solution_m1.block(0),
        this->systems.front().solution.block(0),
        1.0, sys_rhs_p_r.data());
    }

    if (sys_mat_p_r.getNRows() != (int)sys_rhs_p_r.size())
    {
      ErrThrow("Dimension mismatch during A*x = rhs for pressure resolution: ",
              sys_mat_p_r.getNRows(), " rows in the system matrix and ",
              sys_rhs_p_r.size(), " elements in the rhs.");
    }

    sys_mat_p_r.solve(sys_rhs_p_r.data(), sol_p_r.data());

    ROM_p.get_full_solution(this->sol_p_r,
                            this->get_solution().block(d));
  }
}

/* ************************************************************************** */
template <int d>
bool TNSE_ROM<d>::stop_it(unsigned int it_counter)
{
  bool is_velocity = true;
  compute_residuals();

  // check if minimum number of iterations was performed already
  size_t min_it = TimeNavierStokes<d>::db["nonlinloop_minit"];
  if (it_counter < min_it)
  {
    return false;
  }

  size_t max_it = TimeNavierStokes<d>::db["nonlinloop_maxit"];
  double limit = TimeNavierStokes<d>::db["nonlinloop_epsilon"];
  bool linearized_scheme = this->linearized();
  if (TimeNavierStokes<d>::db["nonlinloop_scale_epsilon_with_size"])
  {
    limit *= std::sqrt(sol_u_r.size());
    Output::print("stopping tolerance for nonlinear iteration ", limit);
  }

  if (residual_u_r <= limit || it_counter == max_it
    || (linearized_scheme && it_counter > 0))
  {
    ROM_u.get_full_solution(this->sol_u_r, this->get_solution().block(0));

    solve(!is_velocity);

// FEM residual computation
//     if (fem_residuals)
//       TimeNavierStokes<d>::compute_residuals();

    auto& s = this->systems.front();
    s.solution_m2 = s.solution_m1;
    s.solution_m1 = s.solution;

    this->adjust_pressure();

    set_fluctuations();

    return true;
  }
  else
  {
    return false;
  }
}

template <int d>
void TNSE_ROM<d>::set_fluctuations()
{
  if (db["rom_write_fluctuations"])
  {
    auto& s = this->systems.front();

    int n_u = d * this->get_velocity_space()->get_n_dof();
    int n_p = this->get_pressure_space()->get_n_dof();

    const double* u_mean = ROM_u.get_snaps_avr_ptr();
    const double* p_mean = ROM_p.get_snaps_avr_ptr();

    for (int i = 0; i < n_u; i++)
    {
      sol_fluct[i] = s.solution[i] - u_mean[i];
    }

    for (int i = 0; i < n_p; i++)
    {
      sol_fluct[n_u + i] = s.solution[n_u + i] - p_mean[i];
    }
  }
}

/* ************************************************************************** */
template <int d>
void TNSE_ROM<d>::compute_residuals(void)
{
  // compute reduced velocity defect = sys_rhs_u_r - sys_mat_u_r * sol_u_r
  std::vector<double> defect(sys_rhs_u_r.size());
  std::transform(sys_rhs_u_r.begin(), sys_rhs_u_r.end(),
                  sys_mat_u_r.multiply(&sol_u_r).begin(),
                  defect.begin(), std::minus<double>());
  residual_u_r = std::sqrt(std::inner_product(defect.begin(),
                                              defect.end(),
                                              defect.begin(), 0.));
}


/** ***************************************************************************/
template<int d>
void TNSE_ROM<d>::print_fem_residuals(int loop_index)
{
  if (!fem_residuals)
  {
    return;
  }

  using namespace std;
  std::stringstream s;
  s << "time step " << loop_index << " corresponding fem ";
  s <<left << setprecision(10) << setw(15);
  s << this->get_residuals();
  Output::print<1>(s.str());
}


#ifdef __3D__
template class TNSE_ROM<3>;
#else
template class TNSE_ROM<2>;
#endif
