/// @brief Tests the simplified solvers found in Systems/Simplified for
///        accuracy and boundary value compliance (2D only).

#include <ParMooN.h>
#include <MooNMD_Io.h>

#include <Domain.h>
#include <Database.h>
#include <FESpace2D.h>
#include <FEFunction2D.h>
#include <BlockFEMatrix.h>
#include <BlockVector.h>
#include <Chrono.h>
#include <Utilities.h>

#include <functional>
#include <vector>
#include <cmath>
#include <memory>
#include <cstring>

#include <SimplifiedAssembling.h>
#include <Simplified/SimplifiedInfo.h>
#include <Simplified/Helmholtz.h>
#include <Simplified/Poisson.h>
#include <Simplified/Stokes.h>

/// @brief Example function, taking x and y coordinates and returning exact
///        solution values and negative Laplacians
typedef std::function<void(const std::vector<double>&, const std::vector<double>&,
  double*, double*, double&, double&)> ExampleFunction;

/// @brief u(x, y) = x^2 + y^2, \Delta u(x, y) = 4
void Example_A(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu, double& rel_full, double& rel_interp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    u[i] = x[i] * x[i] + y[i] * y[i];
    Lu[i] = -4.0;
  }

  rel_full = 0.2;
  rel_interp = 0.4;
}

/// @brief u(x, y) = cos(2 x^2 + y^2), \Delta u(x, y) = -(16x^2 + 4y^2) cos(*) - 6 sin(*)
void Example_B(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu, double& rel_full, double& rel_interp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    double x2 = x[i] * x[i];
    double y2 = y[i] * y[i];
    double c = std::cos(2.0 * x2 + y2);
    double s = std::sin(2.0 * x2 + y2);

    u[i] = c;
    Lu[i] = (16.0 * x2 + 4.0 * y2) * c + 6 * s;
  }

  rel_full = 0.3;
  rel_interp = 0.35;
}

/// @brief u(x, y) = x^5 y^7 - x, \Delta u(x, y) = 20 x^3 y^7 + 42 x^5 y^5
void Example_C(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu, double& rel_full, double& rel_interp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    double x2 = x[i] * x[i];
    double x3 = x2 * x[i];
    double x5 = x2 * x3;

    double y2 = y[i] * y[i];
    double y3 = y2 * y[i];
    double y4 = y2 * y2;
    double y5 = y3 * y2;
    double y7 = y4 * y3;

    u[i] = x5 * y7 - x[i];
    Lu[i] = -x3 * y5 * (20.0 * y2 + 42.0 * x2);
  }

  rel_full = 0.3;
  rel_interp = 0.55;
}

/// @brief u(x, y) = x y (1 - x) (1 - y), \Delta u(x, y) = -2 (x(1-x) + y(1-y))
void Example_D(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu, double& rel_full, double& rel_interp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    double mx = x[i] * (1.0 - x[i]);
    double my = y[i] * (1.0 - y[i]);

    u[i] = mx * my;
    Lu[i] = 2.0 * (mx + my);
  }

  rel_full = 0.02;
  rel_interp = 0.05;
}

/// @brief Example function for Stokes problems, taking x and y coordinates and
///        returning exact velocity values, negative velocity Laplacians,
///        pressures, and pressure gradients
typedef std::function<void(const std::vector<double>&, const std::vector<double>&,
  double*, double*,
  double*, double*)> ExampleFunctionStokes;

/// @brief u(x, y) = (4 y * (1 - y), 0),
///        -\Delta u(x, y) = (8, 0)
///        p(x,y) = 8 - 8x
///        \nabla p(x,y) = (-8, 0)
///
///        d/dx u(1, y) = (0, 0), p(1, y) = 0
///
///        f(x, y) = (0, 0)
void Example_Stokes_A(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu,
  double* p, double* Dp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    u[i] = 4.0 * y[i] * (1.0 - y[i]);
    u[i + n] = 0.0;
    Lu[i] = 8.0;
    Lu[i + n] = 0.0;

    p[i] = 8.0 * (1.0 - x[i]);
    Dp[i] = -8.0;
    Dp[i + n] = 0.0;
  }
}

/// @brief u(x, y) = (2 xy, 1 - y^2),
///        -\Delta u(x, y) = (0, 2)
///        p(x,y) = 2y
///        \nabla p(x,y) = (0, 2)
///
///        d/dx u(1, y) = (2y, 0), p(1, y) = 2y
///
///        f(x,y) = (0, 4)
void Example_Stokes_B(const std::vector<double>& x, const std::vector<double>& y,
  double* u, double* Lu,
  double* p, double* Dp)
{
  auto n = x.size();

  for (unsigned i = 0; i < n; i++)
  {
    u[i] = 2.0 * x[i] * y[i];
    u[i + n] = 1.0 - y[i] * y[i];
    Lu[i] = 0.0;
    Lu[i + n] = 2.0;

    p[i] = 2.0 * y[i];
    Dp[i] = 0.0;
    Dp[i + n] = 2.0;
  }
}

void DirichletBoundaryCondition(int, double, BoundCond& bc)
{
  bc = DIRICHLET;
}

void NeumannBoundaryCondition(int, double, BoundCond& bc)
{
  bc = NEUMANN;
}

void StokesVelocityBoundaryCondition(int i, double, BoundCond& bc)
{
  // edge x = 1 gets neumann bc
  bc = i == 1 ? NEUMANN : DIRICHLET;
}

void AssembleErrors(double w, double, const SimplifiedAssemblingData& data,
  bool is_interpolated,
  std::function<double(const double*)> f,
  unsigned int k,
  double* err)
{
  const double fv = f(data.point);
  const double err_solver = (is_interpolated ? data.fe_func[1][{ k, 0 }] : fv) - data.fe_func[0][{ k, 0 }];
  const double err_interpolator = fv - data.fe_func[1][{ k, 0 }];

  err[0] += w * err_solver * err_solver;
  err[1] = std::max(err[1], std::abs(err_solver));
  err[2] += w * err_interpolator * err_interpolator;
}

class ErrorComparison
{
  public:
    ErrorComparison() {}

    void Update(double aL2, double rL2, int d, int r)
    {
      if (aL2 > worst_aL2)
      {
        worst_aL2 = aL2;
        worst_ad = d;
        worst_ar = r;
      }

      if (rL2 > worst_rL2)
      {
        worst_rL2 = rL2;
        worst_rd = d;
        worst_rr = r;
      }
    }

    void Print()
    {
      Output::print("Worst L2 error ratio for\n",
        label, ":\nQ_", worst_ad, ", ref ", worst_ar,
        ", error ", worst_aL2);

      Output::print("Worst domain-relative L2 error for\n",
        label, ":\nQ_", worst_rd, ", ref ", worst_rr,
        ", error ", worst_rL2,
        "\n");
    }

    std::string label;

    double worst_aL2 = 0.0;
    int worst_ad = 0;
    int worst_ar = 0;

    double worst_rL2 = 0.0;
    int worst_rd = 0;
    int worst_rr = 0;
};

std::vector<ErrorComparison> err_poisson_full;
std::vector<ErrorComparison> err_poisson_intp;

std::vector<ErrorComparison> err_helmholtz_full;
std::vector<ErrorComparison> err_helmholtz_intp;

std::vector<ErrorComparison> err_stokes_full;
std::vector<ErrorComparison> err_stokes_intp;

void compute_errors(int deg, int ref, const TDomain& domain,
  const std::vector<ExampleFunction> &functions,
  const std::vector<ExampleFunctionStokes> &functions_stokes)
{
  const double mu = 0.5;
  const double interp_epsilon = 1e-10;
  const double diri_epsilon = 1e-12;
  const double ratio_limit = 1.5;

  using namespace utilities;

  Chrono chrono;

  Output::info("Test",
    "\n\n***************************************************************************************",
    "\n\nChecking examples for Q_", deg, " elements on the unit square, refined ", ref, " times.");

  err_poisson_full.resize(functions.size());
  err_poisson_intp.resize(functions.size());

  err_helmholtz_full.resize(functions.size());
  err_helmholtz_intp.resize(functions.size());

  err_stokes_full.resize(functions_stokes.size());
  err_stokes_intp.resize(functions_stokes.size());

  const auto collections = domain.get_grid_collections();
  const TCollection *coll = collections.front();

  std::shared_ptr<const TFESpace2D> space = std::make_shared<const TFESpace2D>(coll, "Space", DirichletBoundaryCondition, deg);

  std::shared_ptr<const TFESpace2D> space_p = std::make_shared<const TFESpace2D>(coll, "Pressure space", NeumannBoundaryCondition, deg);
  std::shared_ptr<const TFESpace2D> space_u = std::make_shared<const TFESpace2D>(coll, "Velocity space", StokesVelocityBoundaryCondition, deg + 1);

  int n_cells = coll->GetN_Cells();

  int n_dof = space->get_n_dof();
  int n_active = space->get_n_active();
  int n_diri = n_dof - n_active;

  int n_dof_u = space_u->get_n_dof();
  int n_active_u = space_u->get_n_active();
  int n_diri_u = n_dof_u - n_active_u;

  Output::info("Test", "Mesh has ", n_cells, " cells, ",
    "FE space has ", n_dof, " DOF, ", n_active, " active.",
    "\nStokes pressure space has ", n_dof, " DOF.",
    "\nStokes velocity space has ", n_dof_u, " DOF, ", n_active_u, " active.");

  BlockFEMatrix mat({ space });
  BlockFEMatrix mat_stokes({ space_u, space_u, space_p });

  std::vector<double> x(n_dof);
  std::vector<double> y(n_dof);
  std::vector<int> cell(n_dof);

  std::vector<double> x_u(n_dof_u);
  std::vector<double> y_u(n_dof_u);
  std::vector<int> cell_u(n_dof_u);

  for (int c = 0; c < n_cells; c++)
  {
    const int n_local_dof = space->get_n_local_dof(c);
    const int* cell_dof = space->GetGlobalDOF(c);

    for (int i = 0; i < n_local_dof; i++)
    {
      cell[cell_dof[i]] = c;
    }
  }

  for (int c = 0; c < n_cells; c++)
  {
    const int n_local_dof = space_u->get_n_local_dof(c);
    const int* cell_dof = space_u->GetGlobalDOF(c);

    for (int i = 0; i < n_local_dof; i++)
    {
      cell_u[cell_dof[i]] = c;
    }
  }

  for (int i = 0; i < n_dof; i++)
  {
    // this is an extremely inefficient implementation, but it's not sensible
    // to redo it here
    space->GetDOFPosition(i, x[i], y[i]);
  }

  for (int i = 0; i < n_dof_u; i++)
  {
    // this is an extremely inefficient implementation, but it's not sensible
    // to redo it here
    space_u->GetDOFPosition(i, x_u[i], y_u[i]);
  }

  std::vector<double> u_ex(n_dof);
  std::vector<double> Lu_ex(n_dof);

  std::vector<double> err(n_dof);

  std::vector<double> u_ex_stokes(2 * n_dof_u);
  std::vector<double> Lu_ex_stokes(2 * n_dof_u);
  std::vector<double> p_ex_stokes(n_dof_u);
  std::vector<double> Dp_ex_stokes(2 * n_dof_u);

  std::vector<double> err_stokes(2 * n_dof_u);

  BlockVector rhs(mat, true);
  BlockVector u(mat, true);

  BlockVector rhs_stokes(mat_stokes, true);
  BlockVector sol_stokes(mat_stokes, true);

  Poisson<2> poisson({ space });
  poisson.Assemble();

  Helmholtz<2> helmholtz({ space });
  helmholtz.Assemble();

  Stokes<2> stokes(space_u, space_p, SimplifiedInfo::default_direct_solver_db());
  stokes.Assemble();

  using namespace std::placeholders;

  std::vector<double> errors(6);

  SimplifiedAssembling<2> error_assembler({ space }, indices_up_to_order<2>(0));
  SimplifiedAssembling<2> error_assembler_stokes_u({ space_u, space_u }, indices_up_to_order<2>(0));

  char ex[] = { 'A', 0 };
  int fi = 0;

  double inv_h2 = std::pow(2.0, 2.0 * ref - 1);
  double inv_hd = std::pow(2.0, (deg + 1) * (ref - 0.5));

  std::vector<double> tx(1);
  std::vector<double> ty(1);

  for (auto f: functions_stokes)
  {
    double rel_full = 1e-6;
    double rel_interp = 1e-2;

    err_stokes_full[fi].label = std::string("Function ") + ex + ": Stokes (full rhs)";
    err_stokes_intp[fi].label = std::string("Function ") + ex + ": Stokes (interpolated rhs)";

    std::function<double(const double*, int)> f_ = [&](const double* point, int k) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u[2];
      double Lu[2];
      double p;
      double Dp[2];

      f(tx, ty, u, Lu, &p, Dp);

      return Lu[k] + Dp[k];
    };

    std::function<double(const double*, int)> u_ = [&](const double* point, int k) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u[2];
      double Lu[2];
      double p;
      double Dp[2];

      f(tx, ty, u, Lu, &p, Dp);

      return u[k];
    };

    std::function<double(const double*)> p_ = [&](const double* point) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u[2];
      double Lu[2];
      double p;
      double Dp[2];

      f(tx, ty, u, Lu, &p, Dp);

      return p;
    };

    std::function<double(const double*, int)> Dp_ = [&](const double* point, int k) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u[2];
      double Lu[2];
      double p;
      double Dp[2];

      f(tx, ty, u, Lu, &p, Dp);

      return Dp[k];
    };

    std::function<double(const double*)> f0 = std::bind(f_, _1, 0);
    std::function<double(const double*)> f1 = std::bind(f_, _1, 1);

    std::function<double(const double*)> u0 = std::bind(u_, _1, 0);
    std::function<double(const double*)> u1 = std::bind(u_, _1, 1);

    auto check_case = [&](bool is_interpolated)
    {
      clear_vec(rhs_stokes.block(0), 2 * n_dof_u + n_dof);
      copy_vec(rhs_stokes.block(0),
        Dp_ex_stokes.data(), u_ex_stokes.data(),
        n_dof_u, n_active_u);
      copy_vec(rhs_stokes.block(1),
        Dp_ex_stokes.data() + n_dof_u, u_ex_stokes.data() + n_dof_u,
        n_dof_u, n_active_u);
      add_vec(rhs_stokes.block(0),
        Lu_ex_stokes.data(), n_active_u);
      add_vec(rhs_stokes.block(1),
        Lu_ex_stokes.data() + n_dof_u, n_active_u);
      clear_vec(rhs_stokes.block(2), n_dof);

      if (is_interpolated)
      {
        stokes.Solve(rhs_stokes, sol_stokes);
      }
      else
      {
        sol_stokes.copy_nonactive(rhs_stokes);
        stokes.Solve({ f0, f1 }, sol_stokes);
      }

      Output::info("Test", "Solved Stokes problem.");

      // error_assembler.Reset();
      error_assembler_stokes_u.Reset();

      error_assembler_stokes_u.AddAssembly(std::bind(AssembleErrors,
        _1, _2, _3, is_interpolated, u0, 0, errors.data()), 0, 0);
      error_assembler_stokes_u.AddAssembly(std::bind(AssembleErrors,
        _1, _2, _3, is_interpolated, u1, 1, errors.data()), 1, 0);

      error_assembler_stokes_u.AddFEFunction(sol_stokes.get_entries(), indices_up_to_order<2>(0));
      error_assembler_stokes_u.AddFEFunction(u_ex_stokes.data(), indices_up_to_order<2>(0));

      clear_vec(errors);
      // error_assembler.Assemble(1.0);
      error_assembler_stokes_u.Assemble(1.0);

      copy_vec(err_stokes.data(), sol_stokes.get_entries(), 2 * n_dof_u);
      add_vec(err_stokes, u_ex_stokes, -1.0);
      double err_diri = std::max(
        normi_vec(err_stokes.data() + n_active_u, n_diri_u),
        normi_vec(err_stokes.data() + n_dof_u + n_active_u, n_diri_u));
      double err_L2 = std::sqrt(errors[0]);
      double err_Linf = errors[1];
      // we can only expect second order errors when solving from an interpolated rhs
      double err_L2_rel = err_L2 * (is_interpolated ? inv_h2 : inv_hd);
      double err_L2_interp = std::sqrt(errors[2]);
      double err_ratio = err_L2 / std::max(interp_epsilon, err_L2_interp);

      (is_interpolated ? err_stokes_intp[fi] : err_stokes_full[fi]).Update(err_ratio, err_L2_rel, deg, ref);

      Output::info("Test", "Stokes error ",
        is_interpolated
        ? "with interpolated rhs: "
        : "with full rhs: ",
        is_interpolated
        ? "\n|u - u*|_2 = "
        : "\n|u - u_ex|_2 = ", err_L2,
        " = ", err_L2_rel, " h^", is_interpolated ? 2 : (deg + 1),
        "\n|u* - u_ex|_2 = ", err_L2_interp,
        " = ", err_L2_interp / inv_hd, " h^", deg + 1,
        "\n|u - u_ex|_inf = ", err_Linf,
        "\n|u - u_ex|_bd,inf = ", err_diri);

      if (err_diri > diri_epsilon)
      {
        ErrThrow("Mismatched boundary values!");
      }

      if (err_L2_rel > (is_interpolated ? rel_interp : rel_full))
      {
        ErrThrow("Excessive L2 error: ", err_L2_rel, " > ", is_interpolated ? rel_interp : rel_full);
      }

      if (!is_interpolated && err_ratio > ratio_limit)
      {
        ErrThrow("L2 error is too much larger than interpolator L2 error");
      }
    };

    Output::info("Test",
      "\n----------------------------------------"
      "\nChecking example ", ex);

    f(x_u, y_u,
      u_ex_stokes.data(), Lu_ex_stokes.data(),
      p_ex_stokes.data(), Dp_ex_stokes.data());
    Output::info("Test", "Retrieved exact values.");

    check_case(false);
    check_case(true);

    ex[0]++;
    fi++;
  }

  ex[0] = 'A';
  fi = 0;

  for (auto f: functions)
  {
    double rel_full = 0.0;
    double rel_interp = 0.0;

    err_poisson_full[fi].label = std::string("Function ") + ex + ": Poisson (full rhs)";
    err_poisson_intp[fi].label = std::string("Function ") + ex + ": Poisson (interpolated rhs)";

    err_helmholtz_full[fi].label = std::string("Function ") + ex + ": Helmholtz (full rhs)";
    err_helmholtz_intp[fi].label = std::string("Function ") + ex + ": Helmholtz (interpolated rhs)";

    std::function<double(const double*)> f_p = [&](const double* point) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u, Lu;

      f(tx, ty, &u, &Lu, rel_full, rel_interp);

      return Lu;
    };

    std::function<double(const double*)> u_ = [&](const double* point) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u, Lu;

      f(tx, ty, &u, &Lu, rel_full, rel_interp);

      return u;
    };

    std::function<double(const double*)> f_h = [&](const double* point) -> double
    {
      tx[0] = point[0];
      ty[0] = point[1];
      double u, Lu;

      f(tx, ty, &u, &Lu, rel_full, rel_interp);

      return u + mu * mu * Lu;
    };

    auto check_case = [&](bool is_helmholtz, bool is_interpolated)
    {
      if (is_helmholtz)
      {
        copy_vec(rhs.get_entries(), u_ex.data(), n_dof);
        add_vec(rhs.get_entries(), Lu_ex.data(), n_active, mu * mu);

        if (is_interpolated)
        {
          helmholtz.Filter(rhs, u, mu);
        }
        else
        {
          u.copy_nonactive(rhs);
          helmholtz.Filter({ f_h }, u, mu);
        }

        Output::info("Test", "Solved Helmholtz problem.");
      }
      else
      {
        copy_vec(rhs.get_entries(), Lu_ex.data(), u_ex.data(), n_dof, n_active);

        if (is_interpolated)
        {
          poisson.Solve(rhs, u);
        }
        else
        {
          u.copy_nonactive(rhs);
          poisson.Solve({ f_p }, u);
        }

        Output::info("Test", "Solved Poisson problem.");
      }

      error_assembler.Reset();

      error_assembler.AddAssembly(std::bind(AssembleErrors,
        _1, _2, _3, is_interpolated, u_, 0, errors.data()), 0, 0);
      error_assembler.AddFEFunction(u.get_entries(), indices_up_to_order<2>(0));
      error_assembler.AddFEFunction(u_ex.data(), indices_up_to_order<2>(0));

      clear_vec(errors);
      error_assembler.Assemble(1.0);

      copy_vec(err.data(), u.get_entries(), n_dof);
      add_vec(err, u_ex, -1.0);
      double err_diri = normi_vec(err.data() + n_active, n_diri);
      double err_L2 = std::sqrt(errors[0]);
      double err_Linf = errors[1];
      // we can only expect second order errors when solving from an interpolated rhs
      double err_L2_rel = err_L2 * (is_interpolated ? inv_h2 : inv_hd);
      double err_L2_interp = std::sqrt(errors[2]);
      double err_ratio = err_L2 / std::max(interp_epsilon, err_L2_interp);

      if (is_helmholtz)
      {
        (is_interpolated ? err_helmholtz_intp[fi] : err_helmholtz_full[fi]).Update(err_ratio, err_L2_rel, deg, ref);
      }
      else
      {
        (is_interpolated ? err_poisson_intp[fi] : err_poisson_full[fi]).Update(err_ratio, err_L2_rel, deg, ref);
      }

      Output::info("Test", is_helmholtz
        ? "Helmholtz error "
        : "Poisson error ",
        is_interpolated
        ? "with interpolated rhs: "
        : "with full rhs: ",
        is_interpolated
        ? "\n|u - u*|_2 = "
        : "\n|u - u_ex|_2 = ", err_L2,
        " = ", err_L2_rel, " h^", is_interpolated ? 2 : (deg + 1),
        "\n|u* - u_ex|_2 = ", err_L2_interp,
        " = ", err_L2_interp / inv_hd, " h^", deg + 1,
        "\n|u - u_ex|_inf = ", err_Linf,
        "\n|u - u_ex|_bd,inf = ", err_diri);

      if (err_diri > diri_epsilon)
      {
        ErrThrow("Mismatched boundary values!");
      }

      if (err_L2_rel > (is_interpolated ? rel_interp : rel_full))
      {
        ErrThrow("Excessive L2 error: ", err_L2_rel, " > ", is_interpolated ? rel_interp : rel_full);
      }

      if (!is_interpolated && err_ratio > ratio_limit)
      {
        ErrThrow("L2 error is too much larger than interpolator L2 error");
      }
    };

    Output::info("Test",
      "\n----------------------------------------"
      "\nChecking example ", ex);

    f(x, y, u_ex.data(), Lu_ex.data(), rel_full, rel_interp);
    Output::info("Test", "Retrieved exact values.");

    check_case(false, true);
    check_case(false, false);
    check_case(true, true);
    check_case(true, false);

    ex[0]++;
    fi++;
  }

  for (fi = 0; fi < (int)functions.size(); fi++)
  {
    err_poisson_full[fi].Print();
    err_poisson_intp[fi].Print();

    err_helmholtz_full[fi].Print();
    err_helmholtz_intp[fi].Print();
  }

  for (fi = 0; fi < (int)functions_stokes.size(); fi++)
  {
    err_stokes_full[fi].Print();
    err_stokes_intp[fi].Print();
  }

  chrono.stop();

  Output::info("Test", "Total time: ", chrono.elapsed_time());
}

int main(int, char **)
{
  ParameterDatabase db = parmoon::parmoon_initialize();

  db.add("boundary_file", "Default_UnitSquare", "");
  db.add("geo_file", "UnitSquare", "");
  db.add("refinement_n_initial_steps", (size_t)0, "", (size_t)0, (size_t)10);

  std::vector<ExampleFunction> example_functions = { Example_A, Example_B, Example_C, Example_D };
  std::vector<ExampleFunctionStokes> example_functions_stokes = { Example_Stokes_A, Example_Stokes_B };

  for (int ref = 1; ref <= 4; ref++)
  {
    db["refinement_n_initial_steps"] = (size_t)ref;

    TDomain domain(db);
    domain.refine_and_get_hierarchy_of_collections(db);

    for (int deg = ref > 1 ? 1 : 2; deg <= 3; deg++)
    {
      compute_errors(deg, ref, domain, example_functions, example_functions_stokes);
    }
  }

  parmoon::parmoon_finalize();
}