#ifndef __SIMPLIFIED_HELMHOLTZ_H__
#define __SIMPLIFIED_HELMHOLTZ_H__

#include <Enumerations_fe.h>
#include <templateNames.h>
#include <BlockFEMatrix.h>
#include <BlockVector.h>
#include <ParameterDatabase.h>
#include <Solver.h>
#include <vector>
#include <memory>

/// @brief Encapsulates assembly and solver for Helmholtz problems of the type
///          u - \mu^2 \Delta u = f
///        with Dirichlet boundary conditions given by the input vector.
template<int d>
class Helmholtz
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a Helmholtz solver for a given set of base spaces.
    Helmholtz(const std::vector<std::shared_ptr<const FESpace>> &spaces);

    /// @brief Construct a Helmholtz solver for a given set of base spaces.
    Helmholtz(const std::vector<std::shared_ptr<const FESpace>> &spaces,
      const ParameterDatabase& solver_db);

    /// @brief Solve (M + \mu^2 A) y = x*, where M and A are the mass and stiffness matrices.
    void FilterForm(const BlockVector &x_star, BlockVector &y, double mu, bool scaled = false);

    /// @brief Solve (M + \mu^2 A) y = M x, where M and A are the mass and stiffness matrices.
    void Filter(const BlockVector &x, BlockVector &y, double mu, bool scaled = false);

    /// @brief Solve (M + \mu^2 A) y = M x, where M and A are the mass and stiffness matrices.
    void Filter(const std::vector<std::function<double(const double*)>>& f, BlockVector &y, double mu, bool scaled = false);

    void Assemble();

  protected:

    std::vector<std::shared_ptr<const FESpace>> spaces;

    Solver<BlockFEMatrix, BlockVector> solver;

    bool assembled;

    double last_mu;
    bool last_scaled;

    BlockFEMatrix A;
    BlockFEMatrix As;
    BlockFEMatrix M;
    BlockFEMatrix S;

    BlockVector z;
};

#endif