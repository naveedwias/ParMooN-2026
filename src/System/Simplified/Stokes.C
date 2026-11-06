#include <SimplifiedAssembling.h>
#include <Simplified/SimplifiedInfo.h>
#include <Simplified/Stokes.h>
#include <Solver.h>
#include <MooNMD_Io.h>
#include <string.h>

#include <functional>

template<int d>
Stokes<d>::Stokes(std::shared_ptr<const FESpace> &velocity_space,
  std::shared_ptr<const FESpace> &pressure_space)
: Stokes(velocity_space, pressure_space, SimplifiedInfo::default_saddle_solver_db())
{
}

template<int d>
Stokes<d>::Stokes( std::shared_ptr<const FESpace> &velocity_space,
  std::shared_ptr<const FESpace> &pressure_space,
  const ParameterDatabase &solver_db)
: velocity_space(velocity_space), pressure_space(pressure_space),
  solver(solver_db), assembled(false),
  last_nu(1.0)
{
#ifdef _MPI
  Output::root_warn("Stokes", "Stokes solver implementation may not "
    "function correctly in the parallelized case.");
#endif
}

template<int d>
void Stokes<d>::Assemble()
{
  auto Mu = SimplifiedAssembling<d>::AssembleMass(velocity_space);
  Au = SimplifiedAssembling<d>::AssembleStiffness(velocity_space);
  auto D = SimplifiedAssembling<d>::AssembleDivProduct(pressure_space, velocity_space, -1.0);

  std::vector<std::shared_ptr<const FESpace>> block_spaces(d, velocity_space);

  M = BlockFEMatrix(block_spaces);
  std::vector<std::vector<size_t>> cell_positions;
  std::vector<bool> transposed_states;
  for (int k = 0; k < d; k++)
  {
    cell_positions.push_back({ (unsigned)k, (unsigned)k });
    transposed_states.push_back(false);
  }
  M.replace_blocks(*Mu, cell_positions, transposed_states);

  block_spaces.push_back(pressure_space);
  S = BlockFEMatrix(block_spaces);

  for (int k = 0; k < d; k++)
  {
    S.replace_blocks(*D[k], { { d, (unsigned)k }, { (unsigned)k, d } }, { false, true });
  }
  S.replace_blocks(*Au, cell_positions, transposed_states);

  z = BlockVector(M, true);
  rhs = BlockVector(S, true);

  last_nu = 1.0;
  assembled = true;

  solver.update_matrix(S);
}

template<int d>
void Stokes<d>::UpdateMatrix(double nu)
{
  if (nu != last_nu)
  {
    Au = SimplifiedAssembling<d>::AssembleStiffness(velocity_space, nu);

    std::vector<std::vector<size_t>> cell_positions;
    std::vector<bool> transposed_states;
    for (int k = 0; k < d; k++)
    {
      cell_positions.push_back({ (unsigned)k, (unsigned)k });
      transposed_states.push_back(false);
    }
    S.replace_blocks(*Au, cell_positions, transposed_states);

    solver.update_matrix(S);

    last_nu = nu;
  }
}

template<int d>
bool Stokes<d>::CheckVectorSize(const std::string &context, const BlockVector& x)
{
  if (x.n_blocks() < (unsigned)d)
  {
    ErrThrow("Vector for ", context, " too small!");
  }

  for (int i = 0; i < d; i++)
  {
    if (x.length(i) != (unsigned)velocity_space->get_n_dof())
    {
      ErrThrow("Vector for ", context, " block ", i, " is not velocity-sized!");
    }
  }

  if (x.n_blocks() > (unsigned)d)
  {
    if (x.length(d) != (unsigned)pressure_space->get_n_dof())
    {
      ErrThrow("Vector for ", context, " block ", d, " is not pressure-sized!");
    }

    return true;
  }
  else
  {
    return false;
  }
}

template<int d>
void Stokes<d>::CopyVelocity(const BlockVector& u, BlockVector &y, bool bd_only)
{
  int nv = d * velocity_space->get_n_dof();

  if ((int)u.length() < nv || (int)y.length() < nv)
  {
    ErrThrow("Velocity or RHS vector too short!");
  }

  if (bd_only)
  {
    int nvb = velocity_space->get_n_dof();
    int nva = velocity_space->get_n_active();
    int nvd = nvb - nva;
    for (int i = 0; i < d; i++)
    {
      memcpy(y.block(i) + nva, u.block(i) + nva, nvd * sizeof(double));
    }
  }
  else
  {
    memcpy(y.block(0), u.block(0), nv * sizeof(double));
  }
}

template<int d>
void Stokes<d>::ResetPressure(BlockVector &y)
{
  int nv = d * velocity_space->get_n_dof();
  int np = pressure_space->get_n_dof();

  if ((int)y.length() < nv + np)
  {
    ErrThrow("RHS vector too short!");
  }

  memset(y.block(d), 0, np * sizeof(double));
}

template<int d>
void Stokes<d>::SolveForm(const BlockVector &x_star, BlockVector &y, double nu)
{
  if (!assembled)
  {
    Assemble();
  }

  CheckVectorSize("RHS", x_star);

  if (!CheckVectorSize("solution", y))
  {
    ErrThrow("Solution vector has the wrong structure!");
  }

  UpdateMatrix(nu);

  CopyVelocity(x_star, rhs);
  ResetPressure(rhs);
  ResetPressure(y);

  solver.solve(rhs, y);
}

template<int d>
void Stokes<d>::Solve(const BlockVector &x, BlockVector &y, double nu)
{
  if (!assembled)
  {
    Assemble();
  }

  UpdateMatrix(nu);

  if (CheckVectorSize("RHS", x))
  {
    BlockVector u = z;
    CopyVelocity(x, u);

    M.apply(u, z);
    z.copy_nonactive(u);
  }
  else
  {
    M.apply(x, z);
    z.copy_nonactive(x);
  }

  if (!CheckVectorSize("solution", y))
  {
    ErrThrow("Solution vector has the wrong structure!");
  }

  CopyVelocity(z, rhs);
  ResetPressure(rhs);
  ResetPressure(y);

  solver.solve(rhs, y);
}

template<int d>
void Stokes<d>::Solve(const std::vector<std::function<double(const double*)>>& f, BlockVector& y, double nu)
{
  if (f.size() != d)
  {
    ErrThrow("Mismatched function count!");
  }

  if (!assembled)
  {
    Assemble();
  }

  UpdateMatrix(nu);

  if (!CheckVectorSize("solution", y))
  {
    ErrThrow("Solution vector has the wrong structure!");
  }

  SimplifiedAssembling<d> assembler(std::vector<std::shared_ptr<const FESpace>>(d, velocity_space), indices_up_to_order<d>(1));

  for (unsigned int i = 0; i < (unsigned)d; i++)
  {
    assembler.AddAssembly([&, i](double w, double, const SimplifiedAssemblingData& data)
    {
      const auto vr = data.row_basis.slice(1, { 0, 0 });
      vr.add(data.rhs, w * f[i](data.point));
    }, i, 0);
  }

  assembler.Assemble(1.0, nullptr, &z, true);
  CopyVelocity(z, rhs);
  CopyVelocity(y, rhs, true);
  ResetPressure(rhs);
  ResetPressure(y);

  solver.solve(rhs, y);
}

#ifdef __3D__
template class Stokes<3>;
#endif

#ifdef __2D__
template class Stokes<2>;
#endif