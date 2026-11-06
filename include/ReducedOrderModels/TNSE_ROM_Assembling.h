#ifndef TNSE_ROM_ASSEMBLING_H
#define TNSE_ROM_ASSEMBLING_H

#include <ReducedAssembling.h>
#include <templateNames.h>
#include <vector>
#include <array>

template<int d>
class TNSE_ROM_Assembling
{
  using FESpace = typename Template_names<d>::FESpace;

  public:
    TNSE_ROM_Assembling(
      std::shared_ptr<const FESpace> velocity_space, std::shared_ptr<const FESpace> pressure_space,
      int n_velocity_modes, int n_pressure_modes,
      const double* const* velocity_basis,
      const double* const* pressure_basis,
      const std::vector<const double*> &dirichlet_parts);

    TNSE_ROM_Assembling(
      std::shared_ptr<const FESpace> velocity_space, std::shared_ptr<const FESpace> pressure_space,
      int n_velocity_modes, int n_pressure_modes,
      const double* const* velocity_basis,
      const double* const* pressure_basis);

    void ClearVelocityMatrixRHS(double** matrix, double* rhs);
    void ClearPressureMatrixRHS(double** matrix, double* rhs);

    void CombineVelocity(std::vector<double>& u,
      const double* u_full, const double* u_reduced);

    std::array<double, 3> VelocityErrors(const double* u_full, double* u_reduced);

    /// @brief computes u^*_i = (u_f, v_i) for each reduced velocity basis function v_i
    void VelocityFormL2(const double* u_full, double* u_star);

    /// @brief computes u^*_i = (\nabla u_f, \nabla v_i) for each reduced velocity basis function v_i
    void VelocityFormH1(const double* u_full, double* u_star);

    /// @brief computes u_r such that (u_r, v_i) = u^*_i for each reduced velocity basis function v_i
    void ProjectVelocityFormL2(const double* u_star, double* u_reduced);

    /// @brief computes u_r such that (\nabla u_r, \nabla v_i) = u^*_i for each reduced velocity basis function v_i
    void ProjectVelocityFormH1(const double* u_star, double* u_reduced);

    /// @brief computes u_r such that (u_r, v_i) = (u_f, v_i) for each reduced velocity basis function v_i
    void ProjectVelocityL2(const double* u_full, double* u_reduced);

    /// @brief computes u_r such that (\nabla u_r, \nabla v_i) = (\nabla u_f, \nabla v_i) for each reduced velocity basis function v_i
    void ProjectVelocityH1(const double* u_full, double* u_reduced);

    void ProjectVelocityEuclidean(const double* u_full, double* u_reduced);

    /// @brief computes p^*_i = (p_f, q_i) for each reduced pressure basis function q_i
    void PressureFormL2(const double* p_full, double* p_star);

    /// @brief computes p^*_i = (\nabla p_f, \nabla q_i) for each reduced pressure basis function q_i
    void PressureFormH1(const double* p_full, double* p_star);

    /// @brief computes p_r such that (p_r, q_i) = p^*_i for each reduced pressure basis function q_i
    void ProjectPressureFormL2(const double* p_star, double* p_reduced);

    /// @brief computes p_r such that (\nabla p_r, \nabla q_i) = p^*_i for each reduced pressure basis function q_i
    void ProjectPressureFormH1(const double* p_star, double* p_reduced);

    /// @brief computes p_r such that (p_r, q_i) = (p_f, q_i) for each reduced pressure basis function q_i
    void ProjectPressureL2(const double* p_full, double* p_reduced);

    /// @brief computes p_r such that (\nabla p_r, \nabla q_i) = (\nabla p_f, \nabla q_i) for each reduced pressure basis function q_i
    void ProjectPressureH1(const double* p_full, double* p_reduced);

    void ProjectPressureEuclidean(const double* p_full, double* p_reduced);

    /// @brief assembles the L2 gramian for the reduced velocity basis
    void AssembleVelocityL2(double multiplier, double** matrix);

    /// @brief assembles the H1 gramian for the reduced velocity basis
    void AssembleVelocityH1(double multiplier, double** matrix);

    /// @brief assembles the L2 gramian for the reduced pressure basis
    void AssemblePressureL2(double multiplier, double** matrix);

    /// @brief assembles the H1 gramian for the reduced pressure basis
    void AssemblePressureH1(double multiplier, double** matrix);

    void AssembleVelocityStepMatrixRHS(
      double nu,
      const double* u_r, const double* u_f,
      double multiplier, double** matrix, double* rhs);

    void AssembleVelocityStepMatrixRHS(
      double nu,
      const double* u_r,
      const std::vector<double> &dirichlet_factors,
      double multiplier, double** matrix, double* rhs);

    void AssembleVelocityDifferenceRHS(
      const double* u_f_0, const double* u_f_1,
      double multiplier, double* rhs);

    void AssembleVelocityMassRHS(
      const std::vector<double> &dirichlet_factors,
      double multiplier, double* rhs);

    void AssemblePressureFixedMatrixRHS_SM_ROM(
      const double* p_mean,
      double multiplier, double** matrix, double* rhs);

    void AssemblePressureStepRHS_SM_ROM(
      double delta_t,
      const double* u_0, const double* u_1,
      double multiplier, double* rhs);

    void AssemblePressureStepRHS_SM_ROM(
      double delta_t,
      const std::vector<double> &dirichlet_factors_0,
      const std::vector<double> &dirichlet_factors_1,
      const double* u_r_0,
      const double* u_r_1,
      double multiplier, double* rhs);

    void AssembleVelocityCache();
    void AssemblePressureCache();

  protected:
    void AssembleDirichletConvection();
    void AssembleReducedConvection();
    void AssembleDirichletConvectionRHS();
    void AssembleMixedConvectionRHS();

    void AssembleDirichletMassRHS();
    void AssembleDirichletDiffusionRHS();

    void AssembleDirichletPressureRHS();
    void AssembleReducedPressureRHS(const std::vector<double*> &velocity_basis_functions);

    void AssembleDirichletConvectionPressureRHS();
    void AssembleReducedConvectionPressureRHS(const std::vector<double*> &velocity_basis_functions);
    void AssembleMixedConvectionPressureRHS(const std::vector<double*> &velocity_basis_functions);

  protected:
    std::shared_ptr<const FESpace> velocity_space;
    std::shared_ptr<const FESpace> pressure_space;

    ReducedAssembling<d> velocity_assembler;
    ReducedAssembling<d> pressure_assembler;

    const int n_velocity_modes;
    const int n_pressure_modes;
    const double* const* velocity_basis;
    const double* const* pressure_basis;

    const std::vector<const double*> dirichlet_parts;
    const int n_dirichlet_parts;

    std::vector<double> velocity_L2;
    std::vector<double> velocity_H1;

    std::vector<double*> velocity_L2_matrix;
    std::vector<double*> velocity_H1_matrix;

    std::vector<double> pressure_L2;
    std::vector<double> pressure_H1;

    std::vector<double*> pressure_L2_matrix;
    std::vector<double*> pressure_H1_matrix;

    std::vector<double> dirichlet_convection;
    std::vector<std::vector<double*>> dirichlet_convection_matrices;

    std::vector<double> reduced_convection;
    std::vector<std::vector<double*>> reduced_convection_matrices;

    std::vector<double> dirichlet_convection_velocity;
    std::vector<std::vector<double*>> dirichlet_convection_velocity_rhs;

    std::vector<double> mixed_convection_velocity;
    std::vector<std::vector<double*>> mixed_convection_velocity_rhs;

    std::vector<double> dirichlet_mass;
    std::vector<double*> dirichlet_mass_rhs;

    std::vector<double> dirichlet_diffusion;
    std::vector<double*> dirichlet_diffusion_rhs;

    std::vector<double> dirichlet_pressure;
    std::vector<double*> dirichlet_pressure_rhs;

    std::vector<double> reduced_pressure;
    std::vector<double*> reduced_pressure_rhs;

    std::vector<double> dirichlet_convection_pressure;
    std::vector<std::vector<double*>> dirichlet_convection_pressure_rhs;

    std::vector<double> reduced_convection_pressure;
    std::vector<std::vector<double*>> reduced_convection_pressure_rhs;

    std::vector<double> mixed_convection_pressure;
    std::vector<std::vector<double*>> mixed_convection_pressure_rhs;
};

#endif