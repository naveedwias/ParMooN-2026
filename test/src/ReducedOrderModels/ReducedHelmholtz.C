#include <ReducedHelmholtz.h>
#include <ReducedAssembling.h>
#include <MooNMD_Io.h>

#include <functional>

using namespace ReducedBasisLookup;

namespace ReducedHelmholtzAssembling
{
  void AssembleMass(double w, double,
    const ReducedAssemblingData& data, int r)
  {
    for (int k = 0; k < r; k++)
    {
      const auto v_k = DefaultBasisValues(data.row_basis, k);

      v_k.tensor_product(data.matrix, w);
    }
  }

  template<int d>
  void AssembleLaplacian(double w, double,
    const ReducedAssemblingData& data, int r)
  {
    for (int k = 0; k < r; k++)
    {
      for (int m = 0; m < d; m++)
      {
        const auto D_m_v_k = DefaultBasisGradients(data.row_basis, k, m);

        D_m_v_k.tensor_product(data.matrix, w);
      }
    }
  }

  template<int d>
  void AssembleLaplacianScaled(double w, double hK,
    const ReducedAssemblingData& data, int r)
  {
    w *= hK * hK;

    for (int k = 0; k < r; k++)
    {
      for (int m = 0; m < d; m++)
      {
        const auto D_m_v_k = DefaultBasisGradients(data.row_basis, k, m);

        D_m_v_k.tensor_product(data.matrix, w);
      }
    }
  }
}

template<int d>
ReducedHelmholtz<d>::ReducedHelmholtz(std::vector<std::shared_ptr<const FESpace>> spaces,
  int dim, const double* const* basis)
: spaces(spaces), vector_dim((int)spaces.size()), dim(dim), basis(basis),
  assembled(false), last_mu(-1.0), last_scaled(false),
  A(dim, dim), As(dim, dim), M(dim, dim),
  S(dim, dim), rhs(dim, 0.0)
{
}

template<int d>
void ReducedHelmholtz<d>::Assemble()
{
  using namespace ReducedHelmholtzAssembling;
  using namespace std::placeholders;

  M.reset();
  A.reset();
  As.reset();

  ReducedAssembling<d> assembler(spaces, {},
    dim, basis, indices_up_to_order<d>(1));

  assembler.AddAssembly(std::bind(AssembleMass, _1, _2, _3, vector_dim));
  assembler.Assemble(1.0, M.get_columns(), nullptr);

  assembler.Reset();
  assembler.AddAssembly(std::bind(AssembleLaplacian<d>, _1, _2, _3, vector_dim));
  assembler.Assemble(1.0, A.get_columns(), nullptr);

  assembler.Reset();
  assembler.AddAssembly(std::bind(AssembleLaplacianScaled<d>, _1, _2, _3, vector_dim));
  assembler.Assemble(1.0, As.get_columns(), nullptr);

  assembled = true;
}

template<int d>
void ReducedHelmholtz<d>::FilterForm(const double* x_star, double* y, double mu, bool scaled)
{
  if (!assembled)
  {
    Assemble();
  }

  if (last_mu != mu || last_scaled != scaled)
  {
    S = M;
    S.add(scaled ? As : A, mu * mu);
    S.decomposeLU();

    last_mu = mu;
    last_scaled = scaled;
  }

  S.solve(x_star, y);
}

template<int d>
void ReducedHelmholtz<d>::Filter(const double* x, double* y, double mu, bool scaled)
{
  if (!assembled)
  {
    Assemble();
  }

  std::vector<double> z(dim, 0.0);
  M.multiply(x, z.data());

  if (last_mu != mu || last_scaled != scaled)
  {
    S = M;
    S.add(scaled ? As : A, mu * mu);
    S.decomposeLU();

    last_mu = mu;
    last_scaled = scaled;
  }

  S.solve(z.data(), y);
}

#ifdef __3D__
template class ReducedHelmholtz<3>;
#endif

#ifdef __2D__
template class ReducedHelmholtz<2>;
#endif