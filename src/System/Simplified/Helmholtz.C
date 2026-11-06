#include <Simplified/SimplifiedInfo.h>
#include <Simplified/Helmholtz.h>
#include <SimplifiedAssembling.h>
#include <Solver.h>
#include <MooNMD_Io.h>

#include <functional>

template<int d>
Helmholtz<d>::Helmholtz(const std::vector<std::shared_ptr<const FESpace>> &spaces)
: Helmholtz(spaces, SimplifiedInfo::default_solver_db())
{
}

template<int d>
Helmholtz<d>::Helmholtz(const std::vector<std::shared_ptr<const FESpace>> &spaces,
  const ParameterDatabase &solver_db)
: spaces(spaces), solver(solver_db),
  assembled(false), last_mu(-1.0), last_scaled(false)
{
#ifdef _MPI
  Output::root_warn("Helmholtz", "Differential filter implementation may not "
    "function correctly in the parallelized case.");
#endif
}

template<int d>
void Helmholtz<d>::Assemble()
{
  M = BlockFEMatrix(spaces);
  A = BlockFEMatrix(spaces);
  As = BlockFEMatrix(spaces);
  S = BlockFEMatrix(spaces);

  for (unsigned int i = 0; i < spaces.size(); i++)
  {
    FEMatrix block(spaces[i], spaces[i]);

    M.replace_blocks(block, { { i, i } }, { false });
    A.replace_blocks(block, { { i, i } }, { false });
    As.replace_blocks(block, { { i, i } }, { false });
    S.replace_blocks(block, { { i, i } }, { false });
  }

  z = BlockVector(M, true);

  M.reset();
  A.reset();
  As.reset();

  SimplifiedAssembling<d> assembler(spaces, indices_up_to_order<d>(1));

  assembler.AssembleMass(1.0, &M, false);
  assembler.AssembleStiffness(1.0, &A, false);
  assembler.AssembleStiffness(1.0, &As, false, true);

  assembled = true;
}

template<int d>
void Helmholtz<d>::FilterForm(const BlockVector &x_star, BlockVector &y, double mu, bool scaled)
{
  bool needs_system = last_mu != mu || last_scaled != scaled || !assembled;

  if (!assembled)
  {
    Assemble();
  }

  if (needs_system)
  {
    S.reset();
    S.add_scaled(M, 1.0);
    S.add_scaled(scaled ? As : A, mu * mu);

    last_mu = mu;
    last_scaled = scaled;

    solver.update_matrix(S);
  }

  solver.solve(x_star, y);
}

template<int d>
void Helmholtz<d>::Filter(const BlockVector &x, BlockVector &y, double mu, bool scaled)
{
  bool needs_system = last_mu != mu || last_scaled != scaled || !assembled;

  if (!assembled)
  {
    Assemble();
  }

  M.apply(x, z);
  z.copy_nonactive(x);

  if (needs_system)
  {
    S.reset();
    S.add_scaled(M, 1.0);
    S.add_scaled(scaled ? As : A, mu * mu);

    last_mu = mu;
    last_scaled = scaled;

    solver.update_matrix(S);
  }

  solver.solve(z, y);
}

template<int d>
void Helmholtz<d>::Filter(const std::vector<std::function<double(const double*)>>& f, BlockVector &y, double mu, bool scaled)
{
  bool needs_system = last_mu != mu || last_scaled != scaled || !assembled;

  if (!assembled)
  {
    Assemble();
  }

  SimplifiedAssembling<d> assembler(spaces, indices_up_to_order<d>(0));

  for (unsigned int i = 0; i < spaces.size(); i++)
  {
    auto& func = f[i];

    assembler.AddAssembly([&](double w, double, const SimplifiedAssemblingData& data)
    {
      const auto vr = data.row_basis.slice(1, { 0, 0 });
      vr.add(data.rhs, w * func(data.point));
    }, i, 0);
  }

  assembler.Assemble(1.0, nullptr, &z, true);
  z.copy_nonactive(y);

  if (needs_system)
  {
    S.reset();
    S.add_scaled(M, 1.0);
    S.add_scaled(scaled ? As : A, mu * mu);

    last_mu = mu;
    last_scaled = scaled;

    solver.update_matrix(S);
  }

  solver.solve(z, y);
}

#ifdef __3D__
template class Helmholtz<3>;
#endif

#ifdef __2D__
template class Helmholtz<2>;
#endif