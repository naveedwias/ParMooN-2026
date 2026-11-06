#ifndef __SIMPLIFIED_STOKES_H__
#define __SIMPLIFIED_STOKES_H__

#include <Enumerations_fe.h>
#include <templateNames.h>
#include <BlockFEMatrix.h>
#include <BlockVector.h>
#include <ParameterDatabase.h>
#include <Solver.h>
#include <vector>
#include <array>
#include <functional>
#include <memory>

/// @brief Encapsulates assembly and solver for Stokes problems, i.e.
///          -\nu \Laplace u + \nabla p  = f
///          \nabla \cdot u = 0
///        with Dirichlet boundary conditions given by the input vector.
template<int d>
class Stokes
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a Stokes solver for a given set of base spaces.
    Stokes(std::shared_ptr<const FESpace> &velocity_space,
      std::shared_ptr<const FESpace> &pressure_space);

    /// @brief Construct a Stokes solver for a given set of base spaces.
    Stokes(std::shared_ptr<const FESpace> &velocity_space,
      std::shared_ptr<const FESpace> &pressure_space,
      const ParameterDatabase& solver_db);

    /// @brief Solve S y = (x*, 0), where M and S are the mass and system matrices.
    void SolveForm(const BlockVector &x_star, BlockVector &y, double nu = 1.0);

    /// @brief Solve S y = (M x, 0), where M and S are the mass and system matrices.
    void Solve(const BlockVector &x, BlockVector &y, double nu = 1.0);

    /// @brief Solve S y = (f, 0), where S is the system matrix.
    void Solve(const std::vector<std::function<double(const double*)>>& f, BlockVector& y, double nu = 1.0);

    void Assemble();

  protected:
    void CopyVelocity(const BlockVector& u, BlockVector &y, bool bd_only = false);
    void ResetPressure(BlockVector &y);
    void UpdateMatrix(double nu);

    bool CheckVectorSize(const std::string &context, const BlockVector& x);

    const std::shared_ptr<const FESpace> velocity_space;
    const std::shared_ptr<const FESpace> pressure_space;

    Solver<BlockFEMatrix, BlockVector> solver;

    bool assembled;
    double last_nu;

    std::shared_ptr<FEMatrix> Au;

    BlockFEMatrix S;
    BlockFEMatrix M;

    BlockVector z;
    BlockVector rhs;
};

#endif