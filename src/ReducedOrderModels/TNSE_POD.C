/** **************************************************************************** 
*
* @name       TNSE_POD
* @brief      Computation of POD basis - specific routines for TNSED problems
*
*******************************************************************************/

#include "CD_local_assembling_routines.h"
#include "Database.h"
#include "MainUtilities.h"
#include "Utilities.h"
#include "NSE_local_assembling_routines.h"
#include "ROM_local_assembling_routines.h"
#include "Time_NSE_local_assembling_routines.h"
#include "TNSE_POD.h"
#include "Solver.h"
#include "SimplifiedAssembling.h"
#include "Simplified/SimplifiedInfo.h"
#include "Simplified/Helmholtz.h"

#ifdef __2D__
#include "Assemble2D.h"
#include "SquareMatrix2D.h"
#else
#include "Assemble3D.h"
#include "SquareMatrix3D.h"
#endif

#include <sys/stat.h>
#include <cmath>

/* ************************************************************************** */
template <int d>
ParameterDatabase TNSE_POD<d>::set_pod_basis_database(
                                              const ParameterDatabase& param_db)
{
  ParameterDatabase db("Parameter database for POD basis computation");

  db.merge(LocalAssembling<d>::default_local_assembling_database(), true);

  db.merge(ParameterDatabase::default_output_database(), true);
  db.merge(POD::default_pod_database(), true);
  db.merge(Solver<>::default_solver_database(), true);
  db.merge(param_db, false);

  // change name to avoid overwritting standard FEM output
  // TODO: unification between naming system (outfile, output_basename,
  //       snaps_basename, pod_basename)
  std::string output_pod_basename = db["output_basename"].get<std::string>()
                                    + "_pod";
  db["output_basename"].set(output_pod_basename, false);

  return db;
}

/* ************************************************************************** */
template <int d>
TNSE_POD<d>::TNSE_POD(TCollection& coll, const ParameterDatabase& param_db)
  : TNSE_POD<d>(coll, param_db, Example_TimeNSE(param_db))
{
}

/* ************************************************************************** */
template <int d>
TNSE_POD<d>::TNSE_POD(TCollection&             coll,
                      const ParameterDatabase& param_db,
                      const Example_TimeNSE&   ex)
  : velocity_space(new FESpace(&coll, "velocity space", ex.get_bc(0),
                               TDatabase::ParamDB->VELOCITY_SPACE)),
    pressure_space(new FESpace(&coll, "pressure space", ex.get_bc(d),
                               TDatabase::ParamDB->PRESSURE_SPACE)),
    db(TNSE_POD<d>::set_pod_basis_database(param_db)),
    solver_db(Solver<>::default_solver_database()),
    example(ex),
    outputWriter(db)
{
  solver_db.merge(param_db, false, true);
  solver_db["residual_reduction"].set<double>(0.0, false);

  // create directory db["pod_directory"]
  std::string directory_name = this->db["pod_directory"].value_as_string();
  mkdir(directory_name.c_str(), 0777);

  std::vector<std::shared_ptr<const FESpace>> spaces(d, velocity_space);
  spaces.push_back(pressure_space);
  BlockFEMatrix diag_b_matrix(spaces);

  this->pod_mode = BlockVector(diag_b_matrix, false);

  velocity_function = FEVectFunct(velocity_space, "u",
                                  pod_mode.block(0), d);
  pressure_function = FEFunction(pressure_space, "p",
                                 pod_mode.block(d));

  this->set_parameters();

  this->output_problem_size_info();

  this->outputWriter.add_fe_vector_function(&this->velocity_function);
  this->outputWriter.add_fe_function(&this->pressure_function);

  reynolds = param_db["reynolds_number"];
}

/* ************************************************************************** */
template <int d>
void TNSE_POD<d>::set_parameters()
{
  // set problem_type to Time_NSE if not yet set
  if (!db["problem_type"].is(6))
  {
    if (db["problem_type"].is(0))
    {
      db["problem_type"] = 6;
    }
    else
    {
      Output::warn<2>("The parameter problem_type doesn't correspond to "
                      "TimeNavierStokes. It is now reset to the correct value "
                      "for TimeNavierStokes (=6).");
      db["problem_type"] = 6;
    }
  }

  // For assembling of the Gramian matrix for the computation of
  // the POD basis only the "Galerkin" part of the matrices is needed.
  if (!db["space_discretization_type"].is("galerkin"))
  {
    db["space_discretization_type"] = "galerkin";
  }
}

/* ************************************************************************** */
template <int d>
void TNSE_POD<d>::assemble_gramian(BlockFEMatrix* mat_ptr, bool is_velocity, std::string inner_product)
{
  Output::root_info("TNSE_POD", "Assembling ", inner_product, " Gramian for ", is_velocity ? "velocity" : "pressure", " space");

  if (inner_product == "euclidean")
  {
    return;
  }

  SimplifiedAssembling<d> assembler(
    // we're only assembling the first block anyway
    { is_velocity ? velocity_space : pressure_space },
    indices_up_to_order<d>(1));

  if (inner_product == "L2")
  {
    assembler.AssembleMass(1.0, mat_ptr, true);
  }
  else if (inner_product == "H1")
  {
    assembler.AssembleStiffness(1.0, mat_ptr, true);
  }
  else
  {
    ErrThrow("pod_inner_product: ", inner_product, " not supported");
  }
}

/* ************************************************************************** */
template <int d> 
void TNSE_POD<d>::output_problem_size_info() const
{
  double hMin, hMax;
  auto coll   = velocity_space->GetCollection();
  int n_cells = coll->GetN_Cells();
  coll->GetHminHmax(&hMin, &hMax);

  size_t n_dof_velocity = d*velocity_space->get_n_dof();
  size_t n_dof_pressure = pressure_space->get_n_dof();
  size_t n_Total        = n_dof_velocity + n_dof_pressure;
  size_t n_Active       = d*velocity_space->get_n_active();

  Output::stat("TNSE_POD",
                "Mesh data and problem size on finest grid");
  Output::print("n cells     : ", setw(10), n_cells);
  Output::print("h(min,max)  : ", setw(12), hMin ," ", setw(12), hMax);
  Output::print("dof Velocity: ", setw(10), n_dof_velocity);
  Output::print("dof Pressure: ", setw(10), n_dof_pressure);
  Output::print("dof Total   : ", setw(10), n_Total);
  Output::print("active dof  : ", setw(10), n_Active);
}

namespace TNSE_TimeDerivative_Assembly
{
  // HACK: definition in TimeNavierStokes.C
  template<int d> void AssembleVelocityTimeDerivativeForm(double w, double,
    const SimplifiedAssemblingData& data,
    double nu, int s);
}

/* ************************************************************************** */
template <int d>
void TNSE_POD<d>::compute_pod_basis(const ParameterDatabase& param_db)
{
  unsigned int nvb = velocity_space->get_n_dof();
  unsigned int nv = d * nvb;
  unsigned int np = pressure_space->get_n_dof();

  // container to save values for output
  std::vector<std::vector<double>> snaps_avr(2);
  std::vector<std::vector<double>> dt_snaps_avr(2);
  std::vector<DenseMatrix> basis(2, DenseMatrix(0,0));

  std::vector<double> u_eigenvalues;
  std::vector<double> u_divergences;

  std::vector<double> p_eigenvalues;
  std::vector<double> p_gradients;

  bool has_dt = param_db["snaps_time_derivative"]
    && (param_db["snaps_time_derivative_tau"].get<double>() != 0.0);
  bool has_proj_dt = has_dt
    && param_db["snaps_time_derivative_projection"];
  bool has_avg_dt = has_dt
    && db["pod_dt_projection_average"]
    && db["pod_fluctuations_only"];
  bool has_reg_dt = has_dt
    && db["pod_dt_projection_regularized"]
    && db["pod_dt_projection_filter_width"].get<double>() > 0.0;
  bool has_div_dt = has_proj_dt
    && param_db["pod_dt_projection_div_fix"];

  auto vs_db = Solver<>::default_solver_database();
  auto ps_db = Solver<>::default_solver_database();

  BlockFEMatrix vs_mat;
  BlockFEMatrix ps_mat;

  if (has_proj_dt)
  {
    std::string v_db_name = "POD Database - velocity projection Solver";
    std::string p_db_name = "POD Database - pressure projection Solver";

    if (param_db.has_nested_database(v_db_name))
    {
      vs_db.merge(param_db.get_nested_database(v_db_name), false);
    }
    else
    {
      Output::warn("POD", "Missing velocity solver "
                   "database. You should provide a nested database named ",
                   v_db_name, ". Now a default database is used. Is this "
                   "intended?");
    }

    if (param_db.has_nested_database(p_db_name))
    {
      ps_db.merge(param_db.get_nested_database(p_db_name), false);
    }
    else
    {
      Output::warn("POD", "Missing pressure solver "
                   "database. You should provide a nested database named ",
                   p_db_name, ". Now a default database is used. Is this "
                   "intended?");
    }

#ifdef __3D__
    vs_mat = BlockFEMatrix::Mass_NSE3D(velocity_space);
    ps_mat = BlockFEMatrix::CD3D(pressure_space);
#else
    vs_mat = BlockFEMatrix::Mass_NSE2D(velocity_space);
    ps_mat = BlockFEMatrix::CD2D(pressure_space);
#endif

    assemble_gramian(&vs_mat, true, db["pod_dt_projection_inner_product"]);
    assemble_gramian(&ps_mat, false, db["pod_dt_projection_inner_product"]);
  }

  for (auto& compo : data_suffix)
  {
    bool is_velocity = (compo == "u") ? true : false;
    std::string name = compo + "-";
    int length = is_velocity ? d * velocity_space->get_n_dof()
                             : pressure_space->get_n_dof();

    POD pod_tmp(param_db, compo);

    double t_pod_start = GetTime();

    if (!this->db["pod_inner_product"].is("euclidean"))
    {
      if (has_proj_dt
        && (db["pod_inner_product"].get<std::string>()
        == db["pod_dt_projection_inner_product"].get<std::string>()))
      {
        pod_tmp.set_gramian_ptr((is_velocity ? vs_mat : ps_mat).get_combined_matrix());
      }
      else
      {
        #ifdef __3D__
        BlockFEMatrix gramian_matrix = is_velocity
          ? BlockFEMatrix::Mass_NSE3D(velocity_space)
          : BlockFEMatrix::CD3D(pressure_space);
        #else
        BlockFEMatrix gramian_matrix = is_velocity
          ? BlockFEMatrix::Mass_NSE2D(velocity_space)
          : BlockFEMatrix::CD2D(pressure_space);
        #endif
        assemble_gramian(&gramian_matrix, is_velocity, db["pod_inner_product"]);
        pod_tmp.set_gramian_ptr(gramian_matrix.get_combined_matrix());
      }
    }

    if (has_proj_dt)
    {
      if (is_velocity)
      {
        pod_tmp.set_projection(vs_mat, vs_db);
      }
      else
      {
        pod_tmp.set_projection(ps_mat, ps_db);
      }
    }

    if (has_reg_dt)
    {
      if (is_velocity)
      {
        double mu = db["pod_dt_projection_filter_width"];
        bool scaled = db["pod_dt_projection_filter_local_scale"];

        auto x = std::make_shared<BlockVector>(BlockFEMatrix({ velocity_space }), true);
        auto y = std::make_shared<BlockVector>(BlockFEMatrix({ velocity_space }), true);

        auto filter = std::shared_ptr<Helmholtz<d>>(new Helmholtz<d>({ velocity_space }));

        int dim = velocity_space->get_n_dof();

        pod_tmp.set_time_derivative_regularizer([=](double* u)
          {
            for (int k = 0; k < d; k++)
            {
              double* uk = u + k * dim;

              memcpy(x->get_entries(), uk, dim * sizeof(double));

              filter->Filter(*x, *y, mu, scaled);

              memcpy(uk, y->get_entries(), dim * sizeof(double));
            }
          });
      }
      else
      {
        pod_tmp.set_time_derivative_regularizer([](double*){});
      }
    }

    if (has_avg_dt)
    {
      if (is_velocity)
      {
        std::vector<std::shared_ptr<const FESpace>> spaces(d, velocity_space);
        spaces.push_back(pressure_space);

        auto assembler = std::shared_ptr<SimplifiedAssembling<d>>(
          new SimplifiedAssembling<d>(spaces, indices_up_to_order<d>(1)));

        using namespace std::placeholders;
        using namespace TNSE_TimeDerivative_Assembly;

        double nu = 1.0 / reynolds;

        BlockFEMatrix S(spaces);
        auto x = std::make_shared<BlockVector>(S, true);
        auto y = std::make_shared<BlockVector>(S, true);

        dt_snaps_avr[0].resize(nv);
        double* avr_data = dt_snaps_avr[0].data();

        POD pod_tmp_p(param_db, "p");
        auto p_mean = std::make_shared<std::vector<double>>(np);
        pod_tmp_p.compute_snaps_average(p_mean->data(), np);

        pod_tmp.set_time_derivative_getter([=](const double* u, double* dt_u)
        {
          memcpy(x->block(0), u, sizeof(double) * nv);
          memcpy(x->block(d), p_mean->data(), sizeof(double) * np);

          assembler->Reset();
          assembler->AddFEFunction(x->get_entries(), indices_up_to_order<d>(1));

          for (int s = 0; s < d; s++)
          {
            assembler->AddAssembly(std::bind(
              AssembleVelocityTimeDerivativeForm<d>,
              _1, _2, _3,
              nu, s), s, 0);
          }

          assembler->Assemble(1.0, nullptr, y.get(), true);
          y->ResetNonActive();

          memcpy(dt_u, y->get_entries(), sizeof(double) * nv);
          memcpy(avr_data, y->get_entries(), sizeof(double) * nv);
        });
      }
      else
      {
        dt_snaps_avr[1].resize(pressure_space->get_n_dof(), 0.0);

        pod_tmp.set_time_derivative_getter([&](const double*, double* dt_u)
        {
          memset(dt_u, 0, sizeof(double) * pressure_space->get_n_dof());
        });
      }
    }

    if (has_div_dt)
    {
      if (is_velocity)
      {
        std::vector<std::shared_ptr<const FESpace>> spaces(d, velocity_space);
        spaces.push_back(pressure_space);

        auto assembler = std::shared_ptr<SimplifiedAssembling<d>>(
          new SimplifiedAssembling<d>(spaces, indices_up_to_order<d>(1)));

        using namespace std::placeholders;

        auto S = std::shared_ptr<BlockFEMatrix>(new BlockFEMatrix(spaces));

        auto x = std::make_shared<BlockVector>(*S, true);
        auto y = std::make_shared<BlockVector>(*S, true);

        auto f = std::make_shared<std::vector<double>>(np);

        std::shared_ptr<FEMatrix> Su;
        if (db["pod_dt_projection_inner_product"].is("L2"))
        {
          Su = SimplifiedAssembling<d>::AssembleMass(velocity_space);
        }
        else if (db["pod_dt_projection_inner_product"].is("H1"))
        {
          Su = SimplifiedAssembling<d>::AssembleStiffness(velocity_space);
        }
        else
        {
          ErrThrow("Divergence correction not available for inner product '",
            db["pod_dt_projection_inner_product"], "'");
        }
        auto D = SimplifiedAssembling<d>::AssembleDivProduct(pressure_space, velocity_space);

        std::vector<std::vector<size_t>> blocks;
        std::vector<bool> transpose(d, false);
        for (int i = 0; i < d; i++)
        {
          blocks.push_back({ (unsigned)i, (unsigned)i });
        }

        S->replace_blocks(*Su, blocks, transpose);
        for (int k = 0; k < d; k++)
        {
          S->replace_blocks(*D[k], { { d, (unsigned)k }, { (unsigned)k, d } }, { false, true });
        }

        auto solver = std::make_shared<Solver<BlockFEMatrix, BlockVector>>(solver_db);

        x->reset();

        pod_tmp.set_time_derivative_div_fix([=](double* u)
        {
          if (!solver->has_matrix())
          {
            // if we simply call update_matrix outside, S does not get captured
            // and we segfault on solve. this is the easiest way to capture S
            // without a compiler warning
            solver->update_matrix(*S);
          }

          memset(x->block(d), 0, sizeof(double) * np);
          for (int k = 0; k < d; k++)
          {
            D[k]->multiply(u + k * nvb, x->block(d), 1.0);
          }

          solver->solve(*x, *y);

          double* y_ = y->get_entries();
          for (unsigned int i = 0; i < nv; i++)
          {
            u[i] -= y_[i];
          }
        });
      }
      else
      {
        pod_tmp.set_time_derivative_div_fix([](double*){});
      }
    }

    pod_tmp.compute_basis(length);

    Output::print<1>(name, "POD-basis computed in ",
                     GetTime()-t_pod_start, "s.");

    // save value for output
    if (length != pod_tmp.get_basis()->getNColumns())
    {
      ErrThrow("Current FE space dimension does not coincide with the number "
               "of dof of POD basis.\nDimension of FE space : ", length,
               "\nDOF of POD basis      : ",pod_tmp.get_basis()->getNColumns());
    }

    int ci = is_velocity ? 0 : 1;
    snaps_avr[ci] = pod_tmp.get_snaps_avr();
    basis[ci]     = *pod_tmp.get_basis();

    int rank = pod_tmp.get_basis()->getNRows();
    const double* eigs = pod_tmp.get_pod_eigs();

    std::string ev_out_name = outputWriter.get_output_dir() + "/"
      + outputWriter.get_base_name() + "_"
      + (is_velocity ? "u" : "p") + ".csv";

    std::ofstream ev_out_file = std::ofstream(ev_out_name, std::ios::trunc);

    if (is_velocity)
    {
      Output::info("TNSE_POD", "Writing velocity eigenvalues and divergences.");

      std::vector<double> tmp_u(nv);

      auto D = SimplifiedAssembling<d>::AssembleDivProduct(pressure_space, velocity_space);
      auto C = SimplifiedAssembling<d>::AssembleMass(pressure_space);

      BlockFEMatrix MQ({ pressure_space });
      MQ.replace_blocks(*C, { {0, 0} }, { false });

      BlockVector tmp_pstar(MQ);
      BlockVector tmp_p(MQ);

      Solver<BlockFEMatrix, BlockVector> solver(SimplifiedInfo::default_direct_solver_db());
      solver.update_matrix(MQ);

      u_eigenvalues.resize(rank);
      u_divergences.resize(rank);

      for (int i = 0; i < rank; i++)
      {
        Output::info("TNSE_POD", "Computing divergence norm for velocity mode ", i + 1, " / ", rank, "...");

        basis[ci].getRow(i, tmp_u.data());

        tmp_pstar.reset();
        for (int k = 0; k < d; k++)
        {
          D[k]->multiply(tmp_u.data() + k * nvb, tmp_pstar.get_entries(), 1.0);
        }

        solver.solve(MQ, tmp_pstar, tmp_p);

        u_divergences[i] = std::sqrt(dot(tmp_pstar, tmp_p));
      }

      ev_out_file << "Eigenvalue,Divergence\n" << std::setprecision(16) << std::scientific;

      for (int i = 0; i < rank; i++)
      {
        ev_out_file << eigs[i] << "," << u_divergences[i] << "\n";
      }
    }
    else
    {
      Output::info("TNSE_POD", "Writing pressure eigenvalues and gradients.");

      std::vector<double> tmp_p(np);

      auto D = SimplifiedAssembling<d>::AssembleGradProduct(velocity_space, pressure_space);
      auto C = SimplifiedAssembling<d>::AssembleMass(velocity_space);

      std::vector<std::shared_ptr<const FESpace>> spaces(d, velocity_space);

      std::vector<std::vector<size_t>> blocks;
      std::vector<bool> transpose(d, false);

      for (int i = 0; i < d; i++)
      {
        blocks.push_back({ (unsigned)i, (unsigned)i });
      }

      BlockFEMatrix MV(spaces);
      MV.replace_blocks(*C, blocks, transpose);

      BlockVector tmp_ustar(MV);
      BlockVector tmp_u(MV);

      Solver<BlockFEMatrix, BlockVector> solver(SimplifiedInfo::default_solver_db());
      solver.update_matrix(MV);

      p_eigenvalues.resize(rank);
      p_gradients.resize(rank);

      for (int i = 0; i < rank; i++)
      {
        Output::info("TNSE_POD", "Computing gradient norm for pressure mode ", i + 1, " / ", rank, "...");

        basis[ci].getRow(i, tmp_p.data());

        tmp_ustar.reset();
        for (int k = 0; k < d; k++)
        {
          D[k]->multiply(tmp_p.data(), tmp_ustar.get_entries() + k * nvb, 1.0);
        }

        tmp_ustar.ResetNonActive();
        solver.solve(tmp_ustar, tmp_u);

        p_gradients[i] = std::sqrt(dot(tmp_ustar, tmp_u));
      }

      ev_out_file << "Eigenvalue,Gradient\n" << std::setprecision(16) << std::scientific;

      for (int i = 0; i < rank; i++)
      {
        ev_out_file << eigs[i] << "," << p_gradients[i] << "\n";
      }
    }

    ev_out_file.close();

    memcpy(is_velocity ? u_eigenvalues.data() : p_eigenvalues.data(), eigs, rank * sizeof(double));
  }

  output(snaps_avr, dt_snaps_avr, basis);
}

/* ************************************************************************** */
template <int d>
void TNSE_POD<d>::output(const std::vector<std::vector<double>>& snaps_avr,
  const std::vector<std::vector<double>>& dt_snaps_avr,
  const std::vector<DenseMatrix>& basis)
{
  int output_step = 0;

  // write averages of snapshots into a vtk-file
  if (this->db["pod_fluctuations_only"])
  {
    int index = 0;

    for (int i = 0; i < (int)snaps_avr.size(); ++i)
    {
      for (int j = 0; j < (int)snaps_avr[i].size(); ++j)
      {
        this->pod_mode.get_entries()[index] = snaps_avr[i][j];
        index++;
      }
    }

    this->outputWriter.write(output_step++);

    if (db["pod_dt_projection_average"])
    {
      index = 0;

      for (int i = 0; i < (int)dt_snaps_avr.size(); ++i)
      {
        for (int j = 0; j < (int)dt_snaps_avr[i].size(); ++j)
        {
          this->pod_mode.get_entries()[index] = dt_snaps_avr[i][j];
          index++;
        }
      }

      this->outputWriter.write(output_step++);
    }
  }

  // calculate maximal rank
  int rank_max = 0;
  for (int k = 0; k < (int)basis.size(); ++k)
  {
    rank_max = std::max(rank_max, basis[k].getNRows());
  }

  // write the first min(rank_max, 10) POD-basis functions
  for (int i = 0; i < std::min(rank_max, 10); ++i)
  {
    int index = 0;

    for (int k = 0; k < (int)basis.size(); ++k)
    {
      for (int j = 0; j < basis[k].getNColumns(); ++j)
      {
        if (i < basis[k].getNRows())
        {
          this->pod_mode.get_entries()[index] = basis[k].getEntry(i,j);
        }
        else
        {
          this->pod_mode.get_entries()[index] = 0.;
        }

        index++;
      }
    }

    this->outputWriter.write(output_step++);
  }
}

#ifdef __3D__
template class TNSE_POD<3>;
#else
template class TNSE_POD<2>;
#endif
