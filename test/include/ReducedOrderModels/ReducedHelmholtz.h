#ifndef __REDUCED_HELMHOLTZ_H__
#define __REDUCED_HELMHOLTZ_H__

#include <Enumerations_fe.h>
#include <templateNames.h>
#include <DenseMatrix.h>
#include <vector>

/// @brief Encapsulates assembly and solver for reduced-order Helmholtz problems.
template<int d>
class ReducedHelmholtz
{
  using FESpace = typename Template_names<d>::FESpace;
  using MultiIndex_vector = typename Template_names<d>::MultiIndex_vector;

  public:
    /// @brief Construct a Helmholtz solver for a given set of base spaces and ROM basis.
    ReducedHelmholtz(std::vector<std::shared_ptr<const FESpace>> spaces,
      int dim, const double* const* basis);

    /// @brief Solve (M + \mu^2 A) y = x*, where M and A are the ROM mass and stiffness matrices.
    void FilterForm(const double* x_star, double* y, double mu, bool scaled = false);

    /// @brief Solve (M + \mu^2 A) y = M x, where M and A are the ROM mass and stiffness matrices.
    void Filter(const double* x, double* y, double mu, bool scaled = false);

  protected:
    void Assemble();

    const std::vector<std::shared_ptr<const FESpace>> spaces;
    const int vector_dim;
    const int dim;
    const double* const* basis;

    bool assembled;

    double last_mu;
    bool last_scaled;

    DenseMatrix A;
    DenseMatrix As;
    DenseMatrix M;
    DenseMatrix S;
    std::vector<double> rhs;
};

#endif