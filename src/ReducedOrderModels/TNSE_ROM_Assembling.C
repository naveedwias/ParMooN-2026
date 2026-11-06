#include <TNSE_ROM_Assembling.h>
#include <Utilities.h>
#include <MooNMD_Io.h>
#include <DenseMatrix.h>

#ifdef __3D__
#include <FESpace3D.h>
#endif

#ifdef __2D__
#include <FESpace2D.h>
#endif

#include <string.h>
#include <algorithm>
#include <cmath>

namespace TNSE_ROM_Assembly
{
  using namespace ReducedBasisLookup;

  double DefaultFEReducedOnlyVelocity(const ReducedAssemblingData& data, int s, int k)
  {
    return DefaultFEValue(data.reduced_func[s], k);
  }

  double DefaultFEFullOnlyVelocity(const ReducedAssemblingData& data, int s, int k)
  {
    return DefaultFEValue(data.full_func[s], k);
  }

  double DefaultFECombinedVelocity(const ReducedAssemblingData& data, int s, int k)
  {
    return DefaultFEValue(data.reduced_func[s], k) + DefaultFEValue(data.full_func[s], k);
  }

  double DefaultFEReducedOnlyVelocityGradient(const ReducedAssemblingData& data, int s, int k, int m)
  {
    return DefaultFEGradient(data.reduced_func[s], k, m);
  }

  double DefaultFEFullOnlyVelocityGradient(const ReducedAssemblingData& data, int s, int k, int m)
  {
    return DefaultFEGradient(data.full_func[s], k, m);
  }

  template<int d>
  void AssembleVelocityMass(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      DefaultBasisValues(data.row_basis, k)
        .tensor_product(data.matrix, w);
    }
  }

  template<int d>
  void AssembleVelocityLaplacian(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      for (int m = 0; m < d; m++)
      {
        DefaultBasisGradients(data.row_basis, k, m)
          .tensor_product(data.matrix, w);
      }
    }
  }

  template<int d, int s>
  void AssembleVelocityConvection(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      // convection field: combined velocity
      const double wu_k = w * DefaultFECombinedVelocity(data, s, k);

      for (int m = 0; m < d; m++)
      {
        // convection applied to: ansatz basis function
        const auto v_m = DefaultBasisValues(data.row_basis, m);
        const auto D_k_v_m = DefaultBasisGradients(data.row_basis, m, k);

        D_k_v_m.tensor_product(data.matrix, v_m, wu_k);
      }
    }
  }

  template<int d, int s>
  void AssembleVelocityDirichletConvection(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      // convection field: dirichlet velocity
      const double wu_k = w * DefaultFEFullOnlyVelocity(data, s, k);

      for (int m = 0; m < d; m++)
      {
        // convection applied to: ansatz basis function
        const auto v_m = DefaultBasisValues(data.row_basis, m);
        const auto D_k_v_m = DefaultBasisGradients(data.row_basis, m, k);

        // ansatz (column) times test (row)
        D_k_v_m.tensor_product(data.matrix, v_m, wu_k);
      }
    }
  }

  template<int d>
  void AssembleVelocityReducedConvection(double w, double,
    const ReducedAssemblingData& data, int s, double** matrix)
  {
    for (int k = 0; k < d; k++)
    {
      // convection field: reduced velocity
      const double wu_k = w * data.row_basis[{ (unsigned)k, 0, (unsigned)s }];

      for (int m = 0; m < d; m++)
      {
        // convection applied to: ansatz basis function
        const auto v_m = DefaultBasisValues(data.row_basis, m);
        const auto D_k_v_m = DefaultBasisGradients(data.row_basis, m, k);

        // ansatz (column) times test (row)
        D_k_v_m.tensor_product(matrix, v_m, wu_k);
      }
    }
  }

  template<int d, int s>
  void AssembleVelocityErrorL2(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      double uf = DefaultFEFullOnlyVelocity(data, 0, k);
      double ur = DefaultFEReducedOnlyVelocity(data, 0, k);
      double e = uf - ur;

      data.val[s] += w * e * e;
    }
  }

  template<int d, int s>
  void AssembleVelocityErrorH1(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      for (int m = 0; m < d; m++)
      {
        double uf = DefaultFEFullOnlyVelocityGradient(data, 0, k, m);
        double ur = DefaultFEReducedOnlyVelocityGradient(data, 0, k, m);
        double e = uf - ur;

        data.val[s] += w * e * e;
      }
    }
  }

  template<int d>
  void AssembleVelocityRHS(double, double,
    const ReducedAssemblingData&)
  {
  }

  template<int d>
  void AssembleVelocityRHSDifference(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      // difference between time steps of full-order dirichlet/average part
      const double wdtu_k = w * (DefaultFEFullOnlyVelocity(data, 1, k) - DefaultFEFullOnlyVelocity(data, 0, k));
      const auto v_k = DefaultBasisValues(data.row_basis, k);

      v_k.add(data.rhs, wdtu_k);
    }
  }

  template<int d, int s>
  void AssembleVelocityRHSGradient(double w, double,
    const ReducedAssemblingData& data)
  {
    // for each test function v_i:
    // (\nabla u_f, \nabla v_i)
    for (int k = 0; k < d; k++)
    {
      for (int m = 0; m < d; m++)
      {
        // gradient of full-order dirichlet/average part
        const double wdu_k = w * DefaultFEFullOnlyVelocityGradient(data, s, m, k);
        const auto dv_k = DefaultBasisGradients(data.row_basis, m, k);

        dv_k.add(data.rhs, wdu_k);
      }
    }
  }

  template<int d, int s>
  void AssembleVelocityRHSConvection(double w, double,
    const ReducedAssemblingData& data)
  {
    // for each test function v_i:
    // ((u \cdot \nabla) u_f, v_i)
    for (int k = 0; k < d; k++)
    {
      // convection field: reduced-order fluctuations plus full-order dirichlet/average part
      const double wu_k = w * DefaultFECombinedVelocity(data, s, k);

      for (int m = 0; m < d; m++)
      {
        // convection applied to: full-order dirichlet/average part
        const double wD_k_up_m = wu_k * DefaultFEFullOnlyVelocityGradient(data, s, m, k);
        const auto v_m = DefaultBasisValues(data.row_basis, m);

        v_m.add(data.rhs, wD_k_up_m);
      }
    }
  }

  template<int d>
  void AssembleVelocityRHSDirichletConvection(double w, double,
    const ReducedAssemblingData& data)
  {
    // for each test function v_i:
    // ((u_0 \cdot \nabla) u_1, v_i)
    for (int k = 0; k < d; k++)
    {
      // convection field: full-order dirichlet/average part 0
      const double wu_k = w * DefaultFEFullOnlyVelocity(data, 0, k);

      for (int m = 0; m < d; m++)
      {
        // convection applied to: full-order dirichlet/average part 1
        const double wD_k_up_m = wu_k * DefaultFEFullOnlyVelocityGradient(data, 1, m, k);
        const auto v_m = DefaultBasisValues(data.row_basis, m);

        v_m.add(data.rhs, wD_k_up_m);
      }
    }
  }

  template<int d>
  void AssembleVelocityRHSMixedConvection(double w, double,
    const ReducedAssemblingData& data, int s, double* rhs)
  {
    // for each test function v_i:
    // ((u_r \cdot \nabla) u_f, v_i)
    for (int k = 0; k < d; k++)
    {
      // convection field: reduced velocity
      const double wu_k = w * data.row_basis[{ (unsigned)k, 0, (unsigned)s}];

      for (int m = 0; m < d; m++)
      {
        // convection applied to: full-order dirichlet/average part
        const double wD_k_up_m = wu_k * DefaultFEFullOnlyVelocityGradient(data, 0, m, k);
        const auto v_m = DefaultBasisValues(data.row_basis, m);

        v_m.add(rhs, wD_k_up_m);
      }
    }
  }

  template<int d>
  void AssembleVelocityFormL2(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      const double wu_k = w * DefaultFEFullOnlyVelocity(data, 0, k);
      const auto v_k = DefaultBasisValues(data.row_basis, k);

      v_k.add(data.rhs, wu_k);
    }
  }

  template<int d>
  void AssembleVelocityFormH1(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      for (int m = 0; m < d; m++)
      {
        const double wdu_k = w * DefaultFEFullOnlyVelocityGradient(data, 0, k, m);
        const auto dv_k = DefaultBasisGradients(data.row_basis, k, m);

        dv_k.add(data.rhs, wdu_k);
      }
    }
  }

  template<int d>
  void AssemblePressureMass(double w, double,
    const ReducedAssemblingData& data)
  {
    const auto q = DefaultBasisValues(data.row_basis);

    q.tensor_product(data.matrix, w);
  }

  template<int d>
  void AssemblePressureLaplacian(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      d_k_q.tensor_product(data.matrix, w);
    }
  }

  template<int d>
  void AssemblePressureLaplacianSM(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    for (int k = 0; k < d; k++)
    {
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      d_k_q.tensor_product(data.matrix, ww);
    }
  }

  template<int d>
  void AssemblePressureRHS(double, double,
    const ReducedAssemblingData&)
  {
  }

  template<int d>
  void AssemblePressureRHSMean(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    // for each pressure basis function q_i:
    // (\nabla p, \nabla q_i)
    for (int k = 0; k < d; k++)
    {
      const double wd_k_p = ww * DefaultFEGradient(data.full_func[0], k);
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      d_k_q.add(data.rhs, wd_k_p);
    }
  }

  template<int d>
  void AssemblePressureRHSDirichlet(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    // for each pressure basis function q_i:
    // (u, \nabla q_i)
    for (int k = 0; k < d; k++)
    {
      const double wu_k = ww * DefaultFEValue(data.aux_func[0], k);
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      d_k_q.add(data.rhs, wu_k);
    }
  }

  template<int d>
  void AssemblePressureRHSReduced(double w, double hK,
    const ReducedAssemblingData& data, int n_modes,
    double** rhs)
  {
    const double ww = w * hK;

    for (int s = 0; s < n_modes; s++)
    {
      double* c = rhs[s];

      // for each pressure basis function q_i:
      // (u, \nabla q_i)
      for (int k = 0; k < d; k++)
      {
        const double wu_k = ww * DefaultFEValue(data.aux_func[s], k);
        const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

        d_k_q.add(c, wu_k);
      }
    }
  }

  template<int d>
  void AssemblePressureRHSDifference(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    // for each pressure basis function q_i:
    // (u_1 - u_0, \nabla q_i)
    for (int k = 0; k < d; k++)
    {
      const double wdtu_k = ww * (DefaultFEValue(data.aux_func[1], k) - DefaultFEValue(data.aux_func[0], k));
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      d_k_q.add(data.rhs, wdtu_k);
    }
  }

  template<int d, int s>
  void AssemblePressureRHSConvection(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    // for each pressure basis function q_i:
    // ((u \cdot \nabla) u, \nabla q_i)
    for (int k = 0; k < d; k++)
    {
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      for (int m = 0; m < d; m++)
      {
        const double u_m = DefaultFEValue(data.aux_func[s], m);
        const double d_m_u_k = DefaultFEGradient(data.aux_func[s], k, m);

        const double wconv = ww * u_m * d_m_u_k;

        d_k_q.add(data.rhs, wconv);
      }
    }
  }

  template<int d>
  void AssemblePressureRHSDirichletConvection(double w, double hK,
    const ReducedAssemblingData& data)
  {
    const double ww = w * hK;

    // for each pressure basis function q_i:
    // ((u_0 \cdot \nabla) u_1, \nabla q_i)
    for (int k = 0; k < d; k++)
    {
      const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

      for (int m = 0; m < d; m++)
      {
        const double u_m = DefaultFEValue(data.aux_func[0], m);
        const double d_m_u_k = DefaultFEGradient(data.aux_func[1], k, m);

        const double wconv = ww * u_m * d_m_u_k;

        d_k_q.add(data.rhs, wconv);
      }
    }
  }

  template<int d>
  void AssemblePressureRHSMixedConvection(double w, double hK,
    const ReducedAssemblingData& data, int n_modes, double** rhs)
  {
    const double ww = w * hK;

    for (int s = 0; s < n_modes; s++)
    {
      double* c = rhs[s];

      // for each pressure basis function q_i:
      // ((u_0 \cdot \nabla) u_1 + (u_1 \cdot \nabla) u_0, \nabla q_i)
      for (int k = 0; k < d; k++)
      {
        const auto d_k_q = DefaultBasisGradients(data.row_basis, k);

        for (int m = 0; m < d; m++)
        {
          const double u0_m = DefaultFEValue(data.aux_func[0], m);
          const double u1_m = DefaultFEValue(data.aux_func[s + 1], m);

          const double d_m_u0_k = DefaultFEGradient(data.aux_func[0], k, m);
          const double d_m_u1_k = DefaultFEGradient(data.aux_func[s + 1], k, m);

          const double wconv = ww * (u0_m * d_m_u1_k + u1_m * d_m_u0_k);

          d_k_q.add(c, wconv);
        }
      }
    }
  }

  template<int d>
  void AssemblePressureRHSReducedConvection(double w, double hK,
    const ReducedAssemblingData& data, int n_modes, double** rhs)
  {
    const double ww = w * hK;

    for (int s = 0; s < n_modes; s++)
    {
      for (int t = 0; t < n_modes; t++)
      {
        double* c = rhs[s * n_modes + t];

        // ((v_s \cdot nabla) v_t, \nabla q_i)
        for (int k = 0; k < d; k++)
        {
          const auto wvs_k = ww * DefaultFEValue(data.aux_func[s], k);

          for (int m = 0; m < d; m++)
          {
            const auto d_m_q = DefaultBasisGradients(data.row_basis, m);
            const auto d_k_vs_m = DefaultFEGradient(data.aux_func[t], m, k);
            const auto wconv = wvs_k * d_k_vs_m;

            d_m_q.add(c, wconv);
          }
        }
      }
    }
  }

  template<int d>
  void AssemblePressureFormL2(double w, double,
    const ReducedAssemblingData& data)
  {
    const double wp = w * DefaultFEValue(data.full_func[0]);
    const auto q = DefaultBasisValues(data.row_basis);

    q.add(data.rhs, wp);
  }

  template<int d>
  void AssemblePressureFormH1(double w, double,
    const ReducedAssemblingData& data)
  {
    for (int k = 0; k < d; k++)
    {
      const double wdp_k = w * DefaultFEGradient(data.full_func[0], k);
      const auto d_q_k = DefaultBasisGradients(data.row_basis, k);

      d_q_k.add(data.rhs, wdp_k);
    }
  }
}

using namespace TNSE_ROM_Assembly;

template<int d>
TNSE_ROM_Assembling<d>::TNSE_ROM_Assembling(
  std::shared_ptr<const FESpace> velocity_space, std::shared_ptr<const FESpace> pressure_space,
  int n_velocity_modes, int n_pressure_modes,
  const double* const* velocity_basis,
  const double* const* pressure_basis,
  const std::vector<const double*> &dirichlet_parts)
: velocity_space(velocity_space), pressure_space(pressure_space),
  velocity_assembler(
      std::vector<std::shared_ptr<const FESpace>>(d, velocity_space),
      { },
      n_velocity_modes, velocity_basis,
      indices_up_to_order<d>(1)),
  pressure_assembler(
      { pressure_space },
      { std::vector<std::shared_ptr<const FESpace>>(d, velocity_space) },
      n_pressure_modes, pressure_basis,
      indices_up_to_order<d>(1)),
  n_velocity_modes(n_velocity_modes), n_pressure_modes(n_pressure_modes),
  velocity_basis(velocity_basis), pressure_basis(pressure_basis),
  dirichlet_parts(dirichlet_parts),
  n_dirichlet_parts((int)dirichlet_parts.size())
{
}

template<int d>
TNSE_ROM_Assembling<d>::TNSE_ROM_Assembling(
  std::shared_ptr<const FESpace> velocity_space, std::shared_ptr<const FESpace> pressure_space,
  int n_velocity_modes, int n_pressure_modes,
  const double* const* velocity_basis,
  const double* const* pressure_basis)
: TNSE_ROM_Assembling(velocity_space, pressure_space,
    n_velocity_modes, n_pressure_modes,
    velocity_basis, pressure_basis,
    {})
{
}

template<int d>
void TNSE_ROM_Assembling<d>::CombineVelocity(std::vector<double>& u,
  const double* u_full, const double* u_reduced)
{
  int n = d * velocity_space->get_n_dof();

  if ((int)u.size() < n)
  {
    u.resize(n);
  }

  if (u_full != nullptr)
  {
    memcpy(u.data(), u_full, n * sizeof(double));
  }
  else
  {
    memset(u.data(), 0, n * sizeof(double));
  }

  if (u_reduced != nullptr)
  {
    double* ud = u.data();

    for (int i = 0; i < n_velocity_modes; i++)
    {
      for (int j = 0; j < n; j++)
      {
        ud[j] += u_reduced[i] * velocity_basis[j][i];
      }
    }
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ClearVelocityMatrixRHS(double** matrix, double* rhs)
{
  if (matrix != nullptr)
  {
    for (int i = 0; i < n_velocity_modes; i++)
    {
      memset(matrix[i], 0, n_velocity_modes * sizeof(double));
    }
  }

  if (rhs != nullptr)
  {
    memset(rhs, 0, n_velocity_modes * sizeof(double));
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ClearPressureMatrixRHS(double** matrix, double* rhs)
{
  if (matrix != nullptr)
  {
    for (int i = 0; i < n_pressure_modes; i++)
    {
      memset(matrix[i], 0, n_pressure_modes * sizeof(double));
    }
  }

  if (rhs != nullptr)
  {
    memset(rhs, 0, n_pressure_modes * sizeof(double));
  }
}

template<int d>
void AssembleCachedMatrix(
  ReducedAssembling<d>& assembler,
  AssembleLocalReduced assembly,
  std::vector<double>& data,
  std::vector<double*>& data_matrix,
  int n,
  double multiplier, double** matrix)
{
  if (data.empty())
  {
    data.resize(n * n);
    data_matrix.resize(n);
    for (int i = 0; i < n; i++)
    {
      data_matrix[i] = data.data() + i * n;
    }

    assembler.Reset();

    assembler.AddAssembly(assembly);

    assembler.Assemble(1.0, data_matrix.data(), nullptr, true);
  }

  for (int i = 0; i < n; i++)
  {
    const double* row_data = data_matrix[i];
    double* row = matrix[i];

    for (int j = 0; j < n; j++)
    {
      row[j] += multiplier * row_data[j];
    }
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityL2(double multiplier, double** matrix)
{
  AssembleCachedMatrix<d>(
    velocity_assembler,
    AssembleVelocityMass<d>,
    velocity_L2,
    velocity_L2_matrix,
    n_velocity_modes,
    multiplier, matrix);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityH1(double multiplier, double** matrix)
{
  AssembleCachedMatrix<d>(
    velocity_assembler,
    AssembleVelocityLaplacian<d>,
    velocity_H1,
    velocity_H1_matrix,
    n_velocity_modes,
    multiplier, matrix);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureL2(double multiplier, double** matrix)
{
  AssembleCachedMatrix<d>(
    pressure_assembler,
    AssemblePressureMass<d>,
    pressure_L2,
    pressure_L2_matrix,
    n_pressure_modes,
    multiplier, matrix);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureH1(double multiplier, double** matrix)
{
  AssembleCachedMatrix<d>(
    pressure_assembler,
    AssemblePressureLaplacian<d>,
    pressure_H1,
    pressure_H1_matrix,
    n_pressure_modes,
    multiplier, matrix);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityStepMatrixRHS(
  double nu,
  const double* u_r,
  const std::vector<double> &dirichlet_factors,
  double multiplier, double** matrix, double* rhs)
{
  using namespace utilities;

  if (dirichlet_factors.size() != (unsigned)n_dirichlet_parts)
  {
    ErrThrow("Mismatched Dirichlet scaling vector length!")
  }

  // viscosity matrix: nu (\nabla v_j, \nabla v_i)
  AssembleVelocityH1(multiplier * nu, matrix);

  // convection matrix (dirichlet part): ((u_g \cdot \nabla) v_j, v_i)
  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    add_mat(matrix, dirichlet_convection_matrices[i].data(),
      n_velocity_modes, n_velocity_modes,
      multiplier * dirichlet_factors[i]);
  }

  // convection matrix (reduced part): ((u_r \cdot \nabla) v_j, v_i)
  for (int i = 0; i < n_velocity_modes; i++)
  {
    add_mat(matrix, reduced_convection_matrices[i].data(),
      n_velocity_modes, n_velocity_modes,
      multiplier * u_r[i]);
  }

  // diffusion rhs: -nu (\nabla u_g, \nabla v_i)
  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    add_vec(rhs, dirichlet_diffusion_rhs[i],
      n_velocity_modes,
      -nu * multiplier * dirichlet_factors[i]);
  }

  // convection rhs: -((u_g \cdot \nabla) u_g, v_i)
  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    for (int j = 0; j < n_dirichlet_parts; j++)
    {
      add_vec(rhs, dirichlet_convection_velocity_rhs[i][j],
        n_velocity_modes,
        -multiplier * dirichlet_factors[i] * dirichlet_factors[j]);
    }
  }

  // convection rhs: -((u_r \cdot \nabla) u_g, v_i)
  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    for (int j = 0; j < n_velocity_modes; j++)
    {
      add_vec(rhs, mixed_convection_velocity_rhs[i][j],
        n_velocity_modes,
        -multiplier * dirichlet_factors[i] * u_r[j]);
    }
  }
}


template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityStepMatrixRHS(
  double nu,
  const double* u_r, const double* u_f,
  double multiplier, double** matrix, double* rhs)
{
  // u_f: full-order velocity with the correct dirichlet bv
  // u_r: previous Picard iterate of the reduced-order velocity
  // -> u = u_f + u_r: previous Picard iterate of the combined velocity

  // viscosity matrix: nu (\nabla v_j, \nabla v_i)
  AssembleVelocityH1(multiplier * nu, matrix);

  velocity_assembler.Reset();

  // convection matrix at this time: ((u \cdot \nabla) v_j, v_i)
  velocity_assembler.AddAssembly(AssembleVelocityConvection<d, 0>);

  // RHS term at this time: (f, v_j)
  velocity_assembler.AddAssembly(AssembleVelocityRHS<d>);

  // RHS term at this time: -nu (\nabla u_f, \nabla v_j)
  velocity_assembler.AddAssembly(AssembleVelocityRHSGradient<d, 0>, -nu);

  // RHS term at this time: -( (u \cdot \nabla) u_f, v_j)
  velocity_assembler.AddAssembly(AssembleVelocityRHSConvection<d, 0>, -1.0);

  velocity_assembler.AddReducedFunction(u_r, indices_up_to_order<d>(1));
  velocity_assembler.AddFullFunction(u_f, indices_up_to_order<d>(1));

  velocity_assembler.Assemble(multiplier, matrix, rhs);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityDifferenceRHS(
  const double* u_f_0, const double* u_f_1,
  double multiplier, double* rhs)
{
  velocity_assembler.Reset();

  velocity_assembler.AddAssembly(AssembleVelocityRHSDifference<d>);

  velocity_assembler.AddFullFunction(u_f_0, indices_up_to_order<d>(0));
  velocity_assembler.AddFullFunction(u_f_1, indices_up_to_order<d>(0));

  velocity_assembler.Assemble(-multiplier, nullptr, rhs);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityMassRHS(
  const std::vector<double> &dirichlet_factors,
  double multiplier, double* rhs)
{
  using namespace utilities;

  if (dirichlet_factors.size() != (unsigned)n_dirichlet_parts)
  {
    ErrThrow("Mismatched Dirichlet scaling vector length!")
  }

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    add_vec(rhs, dirichlet_mass_rhs[i], multiplier * dirichlet_factors[i]);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureFixedMatrixRHS_SM_ROM(
  const double* p_mean,
  double multiplier, double** matrix, double* rhs)
{
  pressure_assembler.Reset();

  // SM matrix: (hK \nabla q_i, \nabla q_j)
  pressure_assembler.AddAssembly(AssemblePressureLaplacianSM<d>);

  // SM rhs: -(hK \nabla p_mean, \nabla q_j)
  pressure_assembler.AddAssembly(AssemblePressureRHSMean<d>, -1.0);

  pressure_assembler.AddFullFunction(p_mean, indices_up_to_order<d>(1));

  pressure_assembler.Assemble(multiplier, matrix, rhs);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureStepRHS_SM_ROM(
  double delta_t,
  const double* u_0, const double* u_1,
  double multiplier, double* rhs)
{
  pressure_assembler.Reset();

  // rhs term: (f, \nabla q_i)
  pressure_assembler.AddAssembly(AssemblePressureRHS<d>);

  // rhs term: -(hK (u - u_prev) / tau, \nabla q_i)
  pressure_assembler.AddAssembly(AssemblePressureRHSDifference<d>, -1.0 / delta_t);

  // rhs term: -(hK (u \cdot \nabla)u, \nabla q_i)
  pressure_assembler.AddAssembly(AssemblePressureRHSConvection<d, 1>, -1.0);

  pressure_assembler.AddAuxFunction(u_0, indices_up_to_order<d>(1), 0);
  pressure_assembler.AddAuxFunction(u_1, indices_up_to_order<d>(1), 0);

  pressure_assembler.Assemble(multiplier, nullptr, rhs);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureStepRHS_SM_ROM(
  double delta_t,
  const std::vector<double> &dirichlet_factors_0,
  const std::vector<double> &dirichlet_factors_1,
  const double* u_r_0,
  const double* u_r_1,
  double multiplier, double* rhs)
{
  using namespace utilities;

  if (dirichlet_factors_0.size() != (unsigned)n_dirichlet_parts
    || dirichlet_factors_1.size() != (unsigned)n_dirichlet_parts)
  {
    ErrThrow("Mismatched Dirichlet scaling vector length!")
  }

  double m = multiplier / delta_t;

  // -((u_g - u_g_prev) / \tau, \nabla q)
  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    add_vec(rhs, dirichlet_pressure_rhs[i],
      n_pressure_modes,
      m * dirichlet_factors_0[i]);
    add_vec(rhs, dirichlet_pressure_rhs[i],
      n_pressure_modes,
      -m * dirichlet_factors_1[i]);
  }

  // -((u_r - u_r_prev) / \tau, \nabla q)
  for (int i = 0; i < n_velocity_modes; i++)
  {
    add_vec(rhs, reduced_pressure_rhs[i],
      n_pressure_modes,
      m * u_r_0[i]);
    add_vec(rhs, reduced_pressure_rhs[i],
      n_pressure_modes,
      -m * u_r_1[i]);
  }

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    for (int j = 0; j < n_dirichlet_parts; j++)
    {
      add_vec(rhs, dirichlet_convection_pressure_rhs[i][j],
      n_pressure_modes,
      -multiplier * dirichlet_factors_1[i] * dirichlet_factors_1[j]);
    }
  }

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    for (int j = 0; j < n_velocity_modes; j++)
    {
      add_vec(rhs, mixed_convection_pressure_rhs[i][j],
      n_pressure_modes,
      -multiplier * dirichlet_factors_1[i] * u_r_1[j]);
    }
  }

  for (int i = 0; i < n_velocity_modes; i++)
  {
    for (int j = 0; j < n_velocity_modes; j++)
    {
      add_vec(rhs, reduced_convection_pressure_rhs[i][j],
      n_pressure_modes,
      -multiplier * u_r_1[i] * u_r_1[j]);
    }
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::VelocityFormL2(const double* u_full, double* u_star)
{
  memset(u_star, 0, n_velocity_modes * sizeof(double));

  velocity_assembler.Reset();

  velocity_assembler.AddAssembly(AssembleVelocityFormL2<d>);
  velocity_assembler.AddFullFunction(u_full, indices_up_to_order<d>(0));

  velocity_assembler.Assemble(1.0, nullptr, u_star);
}

template<int d>
void TNSE_ROM_Assembling<d>::VelocityFormH1(const double* u_full, double* u_star)
{
  memset(u_star, 0, n_velocity_modes * sizeof(double));

  velocity_assembler.Reset();

  velocity_assembler.AddAssembly(AssembleVelocityFormH1<d>);
  velocity_assembler.AddFullFunction(u_full, indices_up_to_order<d>(1));

  velocity_assembler.Assemble(1.0, nullptr, u_star);
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectVelocityFormL2(const double* u_star, double* u_reduced)
{
  DenseMatrix M(n_velocity_modes, n_velocity_modes);
  M.reset();

  AssembleVelocityL2(1.0, M.get_columns());

  M.decomposeLU();

  if (u_reduced != u_star)
  {
    memcpy(u_reduced, u_star, n_velocity_modes * sizeof(double));
    M.solve(u_reduced);
  }
  else
  {
    M.solve(u_reduced);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectVelocityFormH1(const double* u_star, double* u_reduced)
{
  DenseMatrix M(n_velocity_modes, n_velocity_modes);

  AssembleVelocityH1(1.0, M.get_columns());

  M.decomposeLU();

  if (u_reduced != u_star)
  {
    memcpy(u_reduced, u_star, n_velocity_modes * sizeof(double));
    M.solve(u_reduced);
  }
  else
  {
    M.solve(u_reduced);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectVelocityL2(const double* u_full, double* u_reduced)
{
  VelocityFormL2(u_full, u_reduced);
  ProjectVelocityFormL2(u_reduced, u_reduced);
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectVelocityH1(const double* u_full, double* u_reduced)
{
  VelocityFormH1(u_full, u_reduced);
  ProjectVelocityFormH1(u_reduced, u_reduced);
}

template<int d>
std::array<double, 3> TNSE_ROM_Assembling<d>::VelocityErrors(const double* u_full, double* u_reduced)
{
  int n = d * velocity_space->get_n_dof();

  double n2 = 0.0;
  for (int k = 0; k < n; k++)
  {
    double urk = 0.0;

    for (int i = 0; i < n_velocity_modes; i++)
    {
      urk += u_reduced[i] * velocity_basis[k][i];
    }

    urk -= u_full[k];
    n2 += urk * urk;
  }

  double err[2];

  err[0] = 0.0;
  err[1] = 0.0;

  velocity_assembler.Reset();
  velocity_assembler.AddAssembly(AssembleVelocityErrorL2<d, 0>);
  velocity_assembler.AddAssembly(AssembleVelocityErrorH1<d, 1>);

  velocity_assembler.AddFullFunction(u_full, indices_up_to_order<d>(1));
  velocity_assembler.AddReducedFunction(u_reduced, indices_up_to_order<d>(1));

  velocity_assembler.Assemble(1.0, nullptr, nullptr, false, err);

  return { n2, err[0], err[1] };
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectVelocityEuclidean(const double* u_full, double* u_reduced)
{
  int n = d * velocity_space->get_n_dof();

  DenseMatrix M(n_velocity_modes, n_velocity_modes);
  for (int i = 0; i < n_velocity_modes; i++)
  {
    for (int j = i; j < n_velocity_modes; j++)
    {
      double vij = 0.0;

      for (int k = 0; k < n; k++)
      {
        vij += velocity_basis[k][i] * velocity_basis[k][j];
      }

      M.setEntry(i, j, vij);
      M.setEntry(j, i, vij);
    }
  }

  M.decomposeLU();

  for (int i = 0; i < n_velocity_modes; i++)
  {
    double uvi = 0.0;

    for (int j = 0; j < n; j++)
    {
      uvi += u_full[j] * velocity_basis[j][i];
    }

    u_reduced[i] = uvi;
  }

  M.solve(u_reduced);
}

template<int d>
void TNSE_ROM_Assembling<d>::PressureFormL2(const double* p_full, double* p_star)
{
  memset(p_star, 0, n_pressure_modes * sizeof(double));

  pressure_assembler.Reset();

  pressure_assembler.AddAssembly(AssemblePressureFormL2<d>);
  pressure_assembler.AddFullFunction(p_full, indices_up_to_order<d>(0));

  pressure_assembler.Assemble(1.0, nullptr, p_star);
}

template<int d>
void TNSE_ROM_Assembling<d>::PressureFormH1(const double* p_full, double* p_star)
{
  memset(p_star, 0, n_pressure_modes * sizeof(double));

  pressure_assembler.Reset();

  pressure_assembler.AddAssembly(AssemblePressureFormH1<d>);
  pressure_assembler.AddFullFunction(p_full, indices_up_to_order<d>(1));

  pressure_assembler.Assemble(1.0, nullptr, p_star);
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectPressureFormL2(const double* p_star, double* p_reduced)
{
  DenseMatrix M(n_pressure_modes, n_pressure_modes);

  AssemblePressureL2(1.0, M.get_columns());

  M.decomposeLU();

  if (p_reduced != p_star)
  {
    memcpy(p_reduced, p_star, n_pressure_modes * sizeof(double));
    M.solve(p_reduced);
  }
  else
  {
    M.solve(p_reduced);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectPressureFormH1(const double* p_star, double* p_reduced)
{
  DenseMatrix M(n_pressure_modes, n_pressure_modes);

  AssemblePressureH1(1.0, M.get_columns());

  M.decomposeLU();

  if (p_reduced != p_star)
  {
    memcpy(p_reduced, p_star, n_pressure_modes * sizeof(double));
    M.solve(p_reduced);
  }
  else
  {
    M.solve(p_reduced);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectPressureL2(const double* p_full, double* p_reduced)
{
  PressureFormL2(p_full, p_reduced);
  ProjectPressureFormL2(p_reduced, p_reduced);
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectPressureH1(const double* p_full, double* p_reduced)
{
  PressureFormH1(p_full, p_reduced);
  ProjectPressureFormH1(p_reduced, p_reduced);
}

template<int d>
void TNSE_ROM_Assembling<d>::ProjectPressureEuclidean(const double* p_full, double* p_reduced)
{
  int n = pressure_space->get_n_dof();

  DenseMatrix M(n_pressure_modes, n_pressure_modes);
  for (int i = 0; i < n_pressure_modes; i++)
  {
    for (int j = i; j < n_pressure_modes; j++)
    {
      double qij = 0.0;

      for (int k = 0; k < n; k++)
      {
        qij += pressure_basis[k][i] * pressure_basis[k][j];
      }

      M.setEntry(i, j, qij);
      M.setEntry(j, i, qij);
    }
  }

  M.decomposeLU();

  for (int i = 0; i < n_pressure_modes; i++)
  {
    double pqi = 0.0;

    for (int j = 0; j < n; j++)
    {
      pqi += p_full[j] * pressure_basis[j][i];
    }

    p_reduced[i] = pqi;
  }

  M.solve(p_reduced);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleVelocityCache()
{
  AssembleDirichletConvection();
  AssembleReducedConvection();
  AssembleDirichletConvectionRHS();
  AssembleMixedConvectionRHS();

  AssembleDirichletMassRHS();
  AssembleDirichletDiffusionRHS();
}

template<int d>
void TNSE_ROM_Assembling<d>::AssemblePressureCache()
{
  int dim_V = d * velocity_space->get_n_dof();
  std::vector<double> velocity_basis_transpose(dim_V * n_velocity_modes);
  std::vector<double*> velocity_basis_functions(n_velocity_modes);

  for (int i = 0; i < n_velocity_modes; i++)
  {
    velocity_basis_functions[i] = velocity_basis_transpose.data() + i * dim_V;
    for (int j = 0; j < dim_V; j++)
    {
      velocity_basis_functions[i][j] = velocity_basis[j][i];
    }
  }

  AssembleDirichletPressureRHS();
  AssembleReducedPressureRHS(velocity_basis_functions);

  AssembleDirichletConvectionPressureRHS();
  AssembleReducedConvectionPressureRHS(velocity_basis_functions);
  AssembleMixedConvectionPressureRHS(velocity_basis_functions);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletConvection()
{
  // for each Dirichlet component u_g, computes convection matrix N:
  //
  // N_ij = ((u_g \cdot \nabla) v_j, v_i)

  dirichlet_convection.resize(n_dirichlet_parts * n_velocity_modes * n_velocity_modes, 0.0);
  dirichlet_convection_matrices.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling convection matrix for Dirichlet part ", i, "...");

    double* data = dirichlet_convection.data() + i * n_velocity_modes * n_velocity_modes;
    dirichlet_convection_matrices[i].resize(n_velocity_modes);

    for (int r = 0; r < n_velocity_modes; r++)
    {
      dirichlet_convection_matrices[i][r] = data + r * n_velocity_modes;
    }

    velocity_assembler.Reset();
    velocity_assembler.AddAssembly(AssembleVelocityDirichletConvection<d, 0>);
    velocity_assembler.AddFullFunction(dirichlet_parts[i], indices_up_to_order<d>(0));

    velocity_assembler.Assemble(1.0, dirichlet_convection_matrices[i].data(), nullptr);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleReducedConvection()
{
  // for each basis vector v, computes convection matrix N:
  //
  // N_ij = ((v \cdot \nabla) v_j, v_i)

  Output::info("TNSE_ROM_Assembling", "Assembling convection matrices for reduced velocity basis...");

  reduced_convection.resize(n_velocity_modes * n_velocity_modes * n_velocity_modes, 0.0);
  reduced_convection_matrices.resize(n_velocity_modes);

  velocity_assembler.Reset();

  using namespace std::placeholders;

  for (int i = 0; i < n_velocity_modes; i++)
  {
    double* data = reduced_convection.data() + i * n_velocity_modes * n_velocity_modes;
    reduced_convection_matrices[i].resize(n_velocity_modes);

    for (int r = 0; r < n_velocity_modes; r++)
    {
      reduced_convection_matrices[i][r] = data + r * n_velocity_modes;
    }

    velocity_assembler.AddAssembly(std::bind(AssembleVelocityReducedConvection<d>,
      _1, _2, _3,
      i, reduced_convection_matrices[i].data()));
  }

  velocity_assembler.Assemble(1.0, nullptr, nullptr);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletConvectionRHS()
{
  // for each pair of Dirichlet component u_g, u_h, computes convection RHS w:
  //
  // w_i = ((u_g \cdot \nabla) u_h, v_i)

  dirichlet_convection_velocity.resize(n_dirichlet_parts * n_dirichlet_parts * n_velocity_modes, 0.0);
  dirichlet_convection_velocity_rhs.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    dirichlet_convection_velocity_rhs[i].resize(n_dirichlet_parts);

    for (int j = 0; j < n_dirichlet_parts; j++)
    {
      Output::info("TNSE_ROM_Assembling", "Assembling convection RHS for Dirichlet pair ", i, ", ", j, "...");

      double* data = dirichlet_convection_velocity.data() + (i * n_dirichlet_parts + j) * n_velocity_modes;
      dirichlet_convection_velocity_rhs[i][j] = data;

      velocity_assembler.Reset();
      velocity_assembler.AddAssembly(AssembleVelocityRHSDirichletConvection<d>);
      velocity_assembler.AddFullFunction(dirichlet_parts[i], indices_up_to_order<d>(0));
      velocity_assembler.AddFullFunction(dirichlet_parts[j], indices_up_to_order<d>(1));

      velocity_assembler.Assemble(1.0, nullptr, data);
    }
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleMixedConvectionRHS()
{
  // for each pair of Dirichlet component u_g and reduced basis function v, computes convection RHS w:
  //
  // w_i = ((v \cdot \nabla) u_g, v_i)

  mixed_convection_velocity.resize(n_dirichlet_parts * n_velocity_modes * n_velocity_modes, 0.0);
  mixed_convection_velocity_rhs.resize(n_dirichlet_parts);

  using namespace std::placeholders;

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling mixed convection RHS for Dirichlet part ", i, "...");

    mixed_convection_velocity_rhs[i].resize(n_velocity_modes);

    velocity_assembler.Reset();
    velocity_assembler.AddFullFunction(dirichlet_parts[i], indices_up_to_order<d>(1));

    for (int j = 0; j < n_velocity_modes; j++)
    {
      double* data = mixed_convection_velocity.data() + (i * n_velocity_modes + j) * n_velocity_modes;

      mixed_convection_velocity_rhs[i][j] = data;

      velocity_assembler.AddAssembly(std::bind(AssembleVelocityRHSMixedConvection<d>, _1, _2, _3, j, data));
    }

    velocity_assembler.Assemble(1.0, nullptr, nullptr);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletMassRHS()
{
  // for each Dirichlet component u_g, computes mass RHS w:
  //
  // w_i = (u_g, v_i)

  dirichlet_mass.resize(n_dirichlet_parts * n_velocity_modes, 0.0);
  dirichlet_mass_rhs.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling mass RHS for Dirichlet part ", i, "...");

    double* data = dirichlet_mass.data() + i * n_velocity_modes;
    dirichlet_mass_rhs[i] = data;

    velocity_assembler.Reset();
    velocity_assembler.AddAssembly(AssembleVelocityFormL2<d>);
    velocity_assembler.AddFullFunction(dirichlet_parts[i], indices_up_to_order<d>(0));

    velocity_assembler.Assemble(1.0, nullptr, data);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletDiffusionRHS()
{
  // for each Dirichlet component u_g, computes diffusion RHS w:
  //
  // w_i = (\nabla u_g, \nabla v_i)

  dirichlet_diffusion.resize(n_dirichlet_parts * n_velocity_modes, 0.0);
  dirichlet_diffusion_rhs.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling diffusion RHS for Dirichlet part ", i, "...");

    double* data = dirichlet_diffusion.data() + i * n_velocity_modes;
    dirichlet_diffusion_rhs[i] = data;

    velocity_assembler.Reset();
    velocity_assembler.AddAssembly(AssembleVelocityRHSGradient<d, 0>);
    velocity_assembler.AddFullFunction(dirichlet_parts[i], indices_up_to_order<d>(1));

    velocity_assembler.Assemble(1.0, nullptr, data);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletPressureRHS()
{
  // for each Dirichlet component u_g, computes pressure RHS w:
  //
  // w_i = (u_g, \nabla q_i)

  dirichlet_pressure.resize(n_dirichlet_parts * n_pressure_modes, 0.0);
  dirichlet_pressure_rhs.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling pressure RHS for Dirichlet part ", i, "...");

    double* data = dirichlet_pressure.data() + i * n_pressure_modes;
    dirichlet_pressure_rhs[i] = data;

    pressure_assembler.Reset();
    pressure_assembler.AddAssembly(AssemblePressureRHSDirichlet<d>);
    pressure_assembler.AddAuxFunction(dirichlet_parts[i], indices_up_to_order<d>(0), 0);

    pressure_assembler.Assemble(1.0, nullptr, data);
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleReducedPressureRHS(const std::vector<double*> &velocity_basis_functions)
{
  // for each reduced basis function v, computes pressure RHS w:
  //
  // w_i = (v, \nabla q_i)

  Output::info("TNSE_ROM_Assembling", "Assembling pressure RHS for reduced velocity basis...");

  reduced_pressure.resize(n_velocity_modes * n_pressure_modes, 0.0);
  reduced_pressure_rhs.resize(n_velocity_modes);

  using namespace std::placeholders;

  pressure_assembler.Reset();

  for (int i = 0; i < n_velocity_modes; i++)
  {
    double* data = reduced_pressure.data() + i * n_pressure_modes;
    reduced_pressure_rhs[i] = data;

    pressure_assembler.AddAuxFunction(velocity_basis_functions[i], indices_up_to_order<d>(0), 0);
  }

  pressure_assembler.AddAssembly(std::bind(AssemblePressureRHSReduced<d>,
    _1, _2, _3,
    n_velocity_modes, reduced_pressure_rhs.data()));

  pressure_assembler.Assemble(1.0, nullptr, nullptr);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleDirichletConvectionPressureRHS()
{
  // for each pair of Dirichlet parts u_g/u_h, computes convection pressure RHS w:
  //
  // w_i = ((u_g \cdot \nabla) u_h, q_i)

  dirichlet_convection_pressure.resize(n_dirichlet_parts * n_dirichlet_parts * n_pressure_modes, 0.0);
  dirichlet_convection_pressure_rhs.resize(n_dirichlet_parts);

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    dirichlet_convection_pressure_rhs[i].resize(n_dirichlet_parts);

    for (int j = 0; j < n_dirichlet_parts; j++)
    {
      Output::info("TNSE_ROM_Assembling",
        "Assembling pressure convection RHS for Dirichlet pair ", i, ", ", j, "...");

      double* data = dirichlet_convection_pressure.data() + (i * n_dirichlet_parts + j) * n_pressure_modes;
      dirichlet_convection_pressure_rhs[i][j] = data;

      pressure_assembler.Reset();
      pressure_assembler.AddAssembly(AssemblePressureRHSDirichletConvection<d>);
      pressure_assembler.AddAuxFunction(dirichlet_parts[i], indices_up_to_order<d>(0), 0);
      pressure_assembler.AddAuxFunction(dirichlet_parts[j], indices_up_to_order<d>(0), 0);

      pressure_assembler.Assemble(1.0, nullptr, data);
    }
  }
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleReducedConvectionPressureRHS(const std::vector<double*> &velocity_basis_functions)
{
  // for each pair of reduced basis functions v/v', computes convection pressure RHS w:
  //
  // w_i = ((v \cdot \nabla) v', q_i)

  Output::info("TNSE_ROM_Assembling",
      "Assembling pressure convection RHS for reduced velocity basis...");

  reduced_convection_pressure.resize(n_velocity_modes * n_velocity_modes * n_pressure_modes, 0.0);
  reduced_convection_pressure_rhs.resize(n_velocity_modes);

  using namespace std::placeholders;

  pressure_assembler.Reset();

  std::vector<double*> rhs_temp(n_velocity_modes * n_velocity_modes);

  for (int i = 0; i < n_velocity_modes; i++)
  {
    reduced_convection_pressure_rhs[i].resize(n_velocity_modes);

    for (int j = 0; j < n_velocity_modes; j++)
    {
      double* data = reduced_convection_pressure.data()
        + (i * n_velocity_modes + j) * n_pressure_modes;

      rhs_temp[i * n_velocity_modes + j] = data;

      reduced_convection_pressure_rhs[i][j] = data;
    }

    pressure_assembler.AddAuxFunction(velocity_basis_functions[i], indices_up_to_order<d>(1), 0);
  }

  pressure_assembler.AddAssembly(std::bind(AssemblePressureRHSReducedConvection<d>,
    _1, _2, _3,
    n_velocity_modes, rhs_temp.data()));

  pressure_assembler.Assemble(1.0, nullptr, nullptr);
}

template<int d>
void TNSE_ROM_Assembling<d>::AssembleMixedConvectionPressureRHS(const std::vector<double*> &velocity_basis_functions)
{
  // for each Dirichlet part u_g and each reduced basis function v, computes convection pressure RHS w:
  //
  // w_i = ((u_g \cdot \nabla) v + ((v \cdot \nabla) u_g, q_i)

  mixed_convection_pressure.resize(n_dirichlet_parts * n_velocity_modes * n_pressure_modes, 0.0);
  mixed_convection_pressure_rhs.resize(n_dirichlet_parts);

  using namespace std::placeholders;

  for (int i = 0; i < n_dirichlet_parts; i++)
  {
    Output::info("TNSE_ROM_Assembling", "Assembling mixed pressure convection RHS for Dirichlet part ", i, "...");

    mixed_convection_pressure_rhs[i].resize(n_velocity_modes);

    pressure_assembler.Reset();
    pressure_assembler.AddAuxFunction(dirichlet_parts[i], indices_up_to_order<d>(1), 0);

    for (int j = 0; j < n_velocity_modes; j++)
    {
      mixed_convection_pressure_rhs[i][j] = mixed_convection_pressure.data()
        + (i * n_velocity_modes + j) * n_pressure_modes;

      pressure_assembler.AddAuxFunction(velocity_basis_functions[j], indices_up_to_order<d>(1), 0);
    }

    pressure_assembler.AddAssembly(std::bind(AssemblePressureRHSMixedConvection<d>,
      _1, _2, _3,
      n_velocity_modes, mixed_convection_pressure_rhs[i].data()));

    pressure_assembler.Assemble(1.0, nullptr, nullptr);
  }
}

#ifdef __3D__

template class TNSE_ROM_Assembling<3>;

#endif

#ifdef __2D__

template class TNSE_ROM_Assembling<2>;

#endif