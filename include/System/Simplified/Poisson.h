#ifndef __SIMPLIFIED_POISSON_H__
#define __SIMPLIFIED_POISSON_H__

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

/// @brief Encapsulates assembly and solver for Poisson problems, i.e.
///          \Delta u = f
///        with Dirichlet boundary conditions given by the input vector.
template<int d>
class Poisson
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a Poisson solver for a given set of base spaces.
    Poisson(const std::vector<std::shared_ptr<const FESpace>> &spaces);

    /// @brief Construct a Poisson solver for a given set of base spaces.
    Poisson(const std::vector<std::shared_ptr<const FESpace>> &spaces,
      const ParameterDatabase& solver_db);

    /// @brief Solve A y = x*, where M and A are the mass and stiffness matrices.
    void SolveForm(const BlockVector &x_star, BlockVector &y, bool scaled = false);

    /// @brief Solve A y = M x, where M and A are the mass and stiffness matrices.
    void Solve(const BlockVector &x, BlockVector &y, bool scaled = false);

    /// @brief Solve A y = M x, where M and A are the mass and stiffness matrices.
    void Solve(const std::vector<std::function<double(const double*)>>& f, BlockVector& y, bool scaled = false);

    void Assemble();

  protected:

    std::vector<std::shared_ptr<const FESpace>> spaces;

    Solver<BlockFEMatrix, BlockVector> solver;

    bool assembled;

    BlockFEMatrix A;
    BlockFEMatrix As;
    BlockFEMatrix M;

    BlockVector z;
};

#endif