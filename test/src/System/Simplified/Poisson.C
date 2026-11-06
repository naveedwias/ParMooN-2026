#include <SimplifiedAssembling.h>
#include <Simplified/SimplifiedInfo.h>
#include <Simplified/Poisson.h>
#include <Solver.h>
#include <MooNMD_Io.h>

#include <functional>

template<int d>
Poisson<d>::Poisson(const std::vector<std::shared_ptr<const FESpace>> &spaces)
: Poisson(spaces, SimplifiedInfo::default_solver_db())
{
}

template<int d>
Poisson<d>::Poisson(const std::vector<std::shared_ptr<const FESpace>> &spaces,
  const ParameterDatabase &solver_db)
: spaces(spaces), solver(solver_db),
  assembled(false)
{
#ifdef _MPI
  Output::root_warn("Poisson", "Poisson solver implementation may not "
    "function correctly in the parallelized case.");
#endif
}

template<int d>
void Poisson<d>::Assemble()
{
  M = BlockFEMatrix(spaces);
  A = BlockFEMatrix(spaces);
  As = BlockFEMatrix(spaces);

  for (unsigned int i = 0; i < spaces.size(); i++)
  {
    FEMatrix block(spaces[i], spaces[i]);

    M.replace_blocks(block, { { i, i } }, { false });
    A.replace_blocks(block, { { i, i } }, { false });
    As.replace_blocks(block, { { i, i } }, { false });
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
void Poisson<d>::SolveForm(const BlockVector &x_star, BlockVector &y, bool scaled)
{
  if (!assembled)
  {
    Assemble();
  }

  solver.solve(scaled ? As : A, x_star, y);
}

template<int d>
void Poisson<d>::Solve(const BlockVector &x, BlockVector &y, bool scaled)
{
  if (!assembled)
  {
    Assemble();
  }

  M.apply(x, z);
  z.copy_nonactive(x);

  solver.solve(scaled ? As : A, z, y);
}

template<int d>
void Poisson<d>::Solve(const std::vector<std::function<double(const double*)>>& f, BlockVector& y, bool scaled)
{
  if (f.size() != spaces.size())
  {
    ErrThrow("Mismatched function count!");
  }

  if (!assembled)
  {
    Assemble();
  }

  SimplifiedAssembling<d> assembler(spaces, indices_up_to_order<d>(1));

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

  solver.solve(scaled ? As : A, z, y);
}

#ifdef __3D__
template class Poisson<3>;
#endif

#ifdef __2D__
template class Poisson<2>;
#endif