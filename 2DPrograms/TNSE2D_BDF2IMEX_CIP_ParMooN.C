// =============================================================================
// BDF2-IMEX time stepping for the incompressible Navier-Stokes equations with
// continuous interior penalty (CIP) stabilization of the convective derivative
// for exactly divergence-free (Scott-Vogelius) elements, and, for comparison,
// Taylor-Hood with grad-div stabilization:
//
//   (D_t u^{n+1}, v) + nu (grad u^{n+1}, grad v) + ((b.grad) u^{n+1}, v)
//     + S_IMEX^{n+1}(u^{n+1}, v) + gamma (div u^{n+1}, div v) - (p^{n+1}, div v) = (f^{n+1}, v),
//   (div u^{n+1}, q) = 0,
//   D_t u^{n+1} = (3 u^{n+1} - 4 u^n + u^{n-1})/(2 dt),   b = 2 u^n - u^{n-1},
//
// ("Pressure-robust error estimates for a stabilized BDF2-IMEX Navier-Stokes
// scheme", Section 3). The CIP form S_IMEX with the three terms S_1, S_2, S_3
// and the normalization max{||b||_inf, u_min} (optionally clamped by U_max)
// is assembled by assemble_CIP_NSE2D.
//
// Methods of the numerical studies:
//   SV-CIP : VELOCITY_SPACE 12, refinement_final_step_barycentric true, cip on
//   SV-G   : the same with cip_delta1 = cip_delta2 = cip_delta3 = 0
//   TH     : VELOCITY_SPACE 2 (P2/P1), graddiv_stab 0
//   TH-gd  : VELOCITY_SPACE 2, graddiv_stab gamma
// Starting values: Stokes projections of u(0) and u(t_1) ('bdf2_start:
// stokes_projection', the paper's choice) or an IMEX Euler step from the
// interpolated u(0) ('bdf2_start: euler').
// Reported: E_0 = max_j ||u(t_j) - u_h^j||_0, E_p = (dt sum ||p - p_h||_0^2)^(1/2),
// D = max_j ||div u_h^j||_0, the final time errors ||u(T)-u_h^N||_0, |u(T)-u_h^N|_1,
// the BDF2 energy F^m = ||u^m||^2 + ||2u^m - u^{m-1}||^2 and the maximum
// velocity norm (lake at rest).
// =============================================================================
#include <Domain.h>
#include <Database.h>
#include "NavierStokes.h"
#include <Example_NSE2D.h>
#include <Chrono.h>
#include <ParameterDatabase.h>
#include "ParMooN.h"
#include "LocalAssembling.h"
#include "Assemble2D.h"
#include "BlockFEMatrix.h"
#include "BlockVector.h"
#include "Solver.h"
#include "SquareMatrix2D.h"
#include "Matrix2D.h"
#include "FESpace2D.h"
#include "Structure.h"
#include "QuadratureFormulaDatabase.h"
#include "QuadFormula.h"
#include "Collection.h"
#include "BaseCell.h"
#include "Vertex.h"
#include "FEMatrix.h"
#include "CIP_NSE2D.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>


bool point_in_triangle(const TBaseCell* c, double x, double y, double tol = 1e-9)
{
  double x0 = c->GetVertex(0)->GetX(), y0 = c->GetVertex(0)->GetY();
  double x1 = c->GetVertex(1)->GetX(), y1 = c->GetVertex(1)->GetY();
  double x2 = c->GetVertex(2)->GetX(), y2 = c->GetVertex(2)->GetY();
  double det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
  double l1 = ((x-x0)*(y2-y0) - (x2-x0)*(y-y0))/det;
  double l2 = ((x1-x0)*(y-y0) - (x-x0)*(y1-y0))/det;
  return l1 >= -tol && l2 >= -tol && l1 + l2 <= 1. + tol;
}

// sample velocity, vorticity (d_1 u_2 - d_2 u_1) and divergence of the
// discrete velocity on a uniform N x N grid (cell centres of the grid) of the
// unit square and write them to a text file: x y u1 u2 omega div
void sample_on_grid(const TFEVectFunct2D& u, int N, const std::string& file,
                    double& omega_min, double& omega_max, double& tv_omega)
{
  auto space = u.GetFESpace2D();
  const TCollection* coll = space->GetCollection();
  int n_cells = coll->GetN_Cells();
  auto u1 = u.GetComponent(0);
  auto u2 = u.GetComponent(1);
  std::vector<double> om(N*N, 0.), U1(N*N), U2(N*N), DV(N*N);
  std::vector<int> found(N*N, 0);
  double v1[3], v2[3];
  for(int c = 0; c < n_cells; ++c)
  {
    const TBaseCell* cell = coll->GetCell(c);
    double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    for(int k = 0; k < 3; ++k)
    {
      xmin = std::min(xmin, cell->GetVertex(k)->GetX()); xmax = std::max(xmax, cell->GetVertex(k)->GetX());
      ymin = std::min(ymin, cell->GetVertex(k)->GetY()); ymax = std::max(ymax, cell->GetVertex(k)->GetY());
    }
    int i0 = std::max(0, (int) std::floor(xmin*N - 0.5)), i1 = std::min(N-1, (int) std::ceil(xmax*N - 0.5));
    int j0 = std::max(0, (int) std::floor(ymin*N - 0.5)), j1 = std::min(N-1, (int) std::ceil(ymax*N - 0.5));
    for(int j = j0; j <= j1; ++j)
      for(int i = i0; i <= i1; ++i)
      {
        if(found[j*N+i]) continue;
        double x = (i + 0.5)/N, y = (j + 0.5)/N;
        if(!point_in_triangle(cell, x, y)) continue;
        u1->FindGradientLocal(cell, c, x, y, v1);
        u2->FindGradientLocal(cell, c, x, y, v2);
        U1[j*N+i] = v1[0]; U2[j*N+i] = v2[0];
        om[j*N+i] = v2[1] - v1[2];
        DV[j*N+i] = v1[1] + v2[2];
        found[j*N+i] = 1;
      }
  }
  omega_min = 1e300; omega_max = -1e300; tv_omega = 0.;
  for(int j = 0; j < N; ++j)
    for(int i = 0; i < N; ++i)
    {
      omega_min = std::min(omega_min, om[j*N+i]);
      omega_max = std::max(omega_max, om[j*N+i]);
      // discrete total variation (periodic differences)
      tv_omega += std::abs(om[j*N+(i+1)%N] - om[j*N+i]) + std::abs(om[((j+1)%N)*N+i] - om[j*N+i]);
    }
  tv_omega /= N; // sum |grad omega| h^2 with differences / h
  std::ofstream out(file);
  out << "# x y u1 u2 omega div   (N = " << N << ")\n";
  for(int j = 0; j < N; ++j)
    for(int i = 0; i < N; ++i)
      out << (i+0.5)/N << " " << (j+0.5)/N << " " << U1[j*N+i] << " " << U2[j*N+i]
          << " " << om[j*N+i] << " " << DV[j*N+i] << "\n";
}

// block matrix with the layout of NSTYPE 4 (A11 A12 B1T / A21 A22 B2T /
// B1 B2 0) whose velocity blocks have the extended sparsity pattern needed
// for the edge couplings of the CIP terms
BlockFEMatrix extended_matrix(std::shared_ptr<const TFESpace2D> v,
                              std::shared_ptr<const TFESpace2D> p)
{
  BlockFEMatrix mat({v, v, p});
  std::shared_ptr<TStructure> ext(new TStructure(v, false, true));
  for(size_t r = 0; r < 2; ++r)
    for(size_t c = 0; c < 2; ++c)
      mat.replace_blocks(FEMatrix(v, ext), {{r, c}}, {false});
  mat.replace_blocks(FEMatrix(v, p), {{0, 2}}, {false});
  mat.replace_blocks(FEMatrix(v, p), {{1, 2}}, {false});
  mat.replace_blocks(FEMatrix(p, v), {{2, 0}}, {false});
  mat.replace_blocks(FEMatrix(p, v), {{2, 1}}, {false});
  return mat;
}

// assemble with the given LocalAssembling type into Mat (blocks reset) and,
// if rhs != nullptr, into rhs (Dirichlet values in the inactive entries); the
// convection field of the local routines is 'field'
void assemble_blocks(NavierStokes<2>& ns, BlockFEMatrix& Mat, BlockVector* rhs,
                     const ParameterDatabase& db, LocalAssembling_type type,
                     const BlockVector& field, bool reset = true)
{
  // the local assembling of ParMooN's Navier-Stokes routines expects the
  // velocity components and the pressure (d+1 functions)
  TFEVectFunct2D f_fct(ns.get_velocity_space(), "field",
                       const_cast<double*>(field.get_entries()), 2);
  auto u1 = f_fct.GetComponent(0);
  auto u2 = f_fct.GetComponent(1);
  size_t n_u_local = ns.get_velocity_space()->get_n_dof();
  TFEFunction2D p_fct(ns.get_pressure_space(), "field_p",
                      const_cast<double*>(field.get_entries()) + 2*n_u_local);
  std::vector<const TFEFunction2D*> fcts = {u1.get(), u2.get(), &p_fct};
  LocalAssembling<2> la(db, type, fcts, ns.get_example().get_coeffs());
  // the NavierStokes constructor sets INTERNAL_PROBLEM_LINEAR according to
  // its problem_type; with 1 the quadrature degree is 2k, which does not
  // integrate the trilinear convective term exactly (loss of skew-symmetry)
  TDatabase::ParamDB->INTERNAL_PROBLEM_LINEAR = 0;
  auto blocks = Mat.get_blocks_uniquely();
  if(reset)
    for(auto& b : blocks)
      b->reset();
  if(type == LocalAssembling_type::NavierStokesNL)
  {
    // ParMooN's 'NavierStokesNL' local assembling contains the complete
    // velocity-velocity block (viscous term, grad-div and convective term),
    // as in NavierStokes::assemble_nonlinear_term the velocity blocks are
    // reset before it is assembled
    for(size_t r = 0; r < 2; ++r)
      for(size_t c = 0; c < 2; ++c)
      {
        bool tr;
        Mat.get_block(r, c, tr)->reset();
      }
  }
  // NSTYPE 4: A11 A12 B1T / A21 A22 B2T / B1 B2 (0); NSTYPE 14 adds C
  TSquareMatrix2D* sq[5];
  int n_sq = 0;
  TMatrix2D* re[4];
  int n_re = 0;
  auto block = [&](size_t r, size_t c) -> FEMatrix* {
    bool transposed;
    std::shared_ptr<FEMatrix> ptr = Mat.get_block(r, c, transposed);
    return ptr.get();
  };
  sq[n_sq++] = reinterpret_cast<TSquareMatrix2D*>(block(0, 0));
  sq[n_sq++] = reinterpret_cast<TSquareMatrix2D*>(block(0, 1));
  sq[n_sq++] = reinterpret_cast<TSquareMatrix2D*>(block(1, 0));
  sq[n_sq++] = reinterpret_cast<TSquareMatrix2D*>(block(1, 1));
  // the local assembling of the Navier-Stokes routines always expects the
  // pressure-pressure block as fifth square matrix (nullptr for NSTYPE 4)
  sq[n_sq++] = nullptr;
  re[n_re++] = reinterpret_cast<TMatrix2D*>(block(2, 0));
  re[n_re++] = reinterpret_cast<TMatrix2D*>(block(2, 1));
  re[n_re++] = reinterpret_cast<TMatrix2D*>(block(0, 2));
  re[n_re++] = reinterpret_cast<TMatrix2D*>(block(1, 2));
  const TFESpace2D* spaces[2] = {ns.get_velocity_space().get(),
                                 ns.get_pressure_space().get()};
  BoundCondFunct2D* bc[3] = {spaces[0]->get_boundary_condition(),
                             spaces[0]->get_boundary_condition(),
                             spaces[1]->get_boundary_condition()};
  BoundValueFunct2D* bd[3] = {ns.get_example().get_bd(0), ns.get_example().get_bd(1),
                              ns.get_example().get_bd(2)};
  int n_rhs = la.get_n_rhs();
  BlockVector scratch(ns.get_rhs());
  scratch.reset();
  BlockVector& target = (rhs != nullptr) ? *rhs : scratch;
  if(reset)
    target.reset();
  double* rhs_array[3] = {target.block(0), target.block(1), target.block(2)};
  const TFESpace2D* rhs_spaces[3] = {spaces[0], spaces[0], spaces[1]};
  if(type == LocalAssembling_type::TimeNavierStokesMass)
  {
    // NSMassMatrix writes the two diagonal velocity blocks into the first two
    // local matrices
    TSquareMatrix2D* sq_m[2] = {reinterpret_cast<TSquareMatrix2D*>(block(0, 0)),
                                reinterpret_cast<TSquareMatrix2D*>(block(1, 1))};
    TMatrix2D* re_null[4] = {nullptr, nullptr, nullptr, nullptr};
    Assemble2D(2, spaces, 2, sq_m, 0, re_null, n_rhs, rhs_array, rhs_spaces, bc,
               bd, la, false);
    return;
  }
  if(type == LocalAssembling_type::NavierStokesNL)
  {
    // as in NavierStokes::assemble_nonlinear_term: only the velocity blocks
    TMatrix2D* re_null[4] = {nullptr, nullptr, nullptr, nullptr};
    double* rhs_null[3] = {nullptr, nullptr, nullptr};
    Assemble2D(2, spaces, n_sq, sq, 4, re_null, 3, rhs_null, rhs_spaces, bc,
               bd, la, false);
    return;
  }
  Assemble2D(2, spaces, n_sq, sq, n_re, re, n_rhs, rhs_array, rhs_spaces, bc,
             bd, la, false);
}

// L2 norm of the velocity part of a vector, by quadrature. (Not with the
// assembled mass matrix: its Dirichlet rows act as identity rows, so nonzero
// boundary values would be weighted with 1 instead of O(h^2).)
double l2_norm_velocity(const std::shared_ptr<const TFESpace2D>& space,
                        const BlockVector& v, size_t n_u)
{
  double s = 0.;
  for(int c = 0; c < 2; ++c)
  {
    TFEFunction2D f(space, "v", const_cast<double*>(v.get_entries()) + c*n_u);
    double n = f.get_L2_norm();
    s += n*n;
  }
  return std::sqrt(s);
}

int main(int argc, char* argv[])
{
  Chrono timer;
  ParameterDatabase defaults("BDF2-IMEX CIP defaults");
  defaults.add("cip_delta1", 1.e-2, "CIP parameter delta_1 (S_1)", 0., 1.e10);
  defaults.add("cip_delta2", 1.e-4, "CIP parameter delta_2 (S_2)", 0., 1.e10);
  defaults.add("cip_delta3", 1.e-6, "CIP parameter delta_3 (S_3)", 0., 1.e10);
  defaults.add("cip_u_min", 0.25, "lower bound of the normalization", 0., 1.e10);
  defaults.add("cip_u_max", -1., "upper bound (clamp) of the normalization, "
               "inactive if <= 0", -1., 1.e10);
  defaults.add("cip_quad_degree", 8, "degree of the edge quadrature", 1, 20);
  defaults.add("bdf2_start", "stokes_projection", "starting values u^0, u^1",
               {"stokes_projection", "euler"});
  defaults.add("bdf2_write_table", "", "file name for the per-step table");
  defaults.add("bdf2_periodic", false, "periodic boundary conditions (the "
               "example's boundary condition must return PERIODIC)");
  defaults.add("bdf2_sample_grid", 0, "if > 0: sample velocity, vorticity and "
               "divergence on an N x N grid at the final time (file "
               "<output_basename>_sample.txt) and print vorticity range and "
               "total variation", 0, 4096);
  defaults.add("bdf2_start_pressure", "interpolation", "pressure of the starting "
               "values (only for the error table): interpolant of the exact "
               "pressure, or the pressure of the Stokes projection",
               {"interpolation", "stokes"});
  defaults.add("pressure_amplitude", 1., "amplitude A of the pressure in "
               "example 25 (TDatabase::ParamDB->P0)", -1.e10, 1.e10);
  defaults.merge(NavierStokes<2>::default_nse_database(), true);
  auto parmoon_db = parmoon::parmoon_initialize(argc, argv);
  defaults.merge(parmoon_db, true);
  parmoon_db = defaults;

  TDatabase::ParamDB->RE_NR = parmoon_db["reynolds_number"];
  TDatabase::ParamDB->P0 = parmoon_db["pressure_amplitude"];
  TDatabase::ParamDB->FLOW_PROBLEM_TYPE = 5;
  parmoon_db["problem_type"] = 3; // the steps are linear (Oseen type)
  // INTERNAL_PROBLEM_LINEAR = 0 makes ParMooN choose the quadrature degree
  // 3 k instead of 2 k, which integrates the trilinear convective term
  // exactly; with the lower degree the skew-symmetry of the convective term
  // is lost and the scheme is not energy stable for small nu
  TDatabase::ParamDB->INTERNAL_PROBLEM_LINEAR = 0;

  TDomain domain(parmoon_db);
  check_parameters_consistency_NSE(parmoon_db);
  // periodic boundaries (example 27): the boundary condition function is
  // passed, so that boundary joints marked PERIODIC become periodic joints
  Example_NSE2D example_for_domain(parmoon_db);
  bool periodic = parmoon_db["bdf2_periodic"];
  domain.refine_and_get_hierarchy_of_collections(parmoon_db,
    periodic ? example_for_domain.get_bc(0) : nullptr);

  double t_start = parmoon_db["time_start"], t_end = parmoon_db["time_end"];
  double dt = parmoon_db["time_step_length"];
  int n_steps = (int) std::lround((t_end - t_start)/dt);
  dt = (t_end - t_start)/n_steps;
  CIP_parameters cip;
  cip.delta1 = parmoon_db["cip_delta1"];
  cip.delta2 = parmoon_db["cip_delta2"];
  cip.delta3 = parmoon_db["cip_delta3"];
  cip.u_min = parmoon_db["cip_u_min"];
  cip.u_max = parmoon_db["cip_u_max"];
  cip.quad_degree = parmoon_db["cip_quad_degree"];
  bool use_cip = (cip.delta1 > 0. || cip.delta2 > 0. || cip.delta3 > 0.);
  std::string table_file = parmoon_db["bdf2_write_table"];

  // the Navier-Stokes example and the Stokes projection example (same
  // solution and boundary data, right-hand side -nu Laplace(u) + grad p)
  Example_NSE2D example(parmoon_db);

  // the system object: spaces, matrix structure, solution, output, errors.
  // Its own assembling (galerkin, problem_type 3 -> Stokes) is used for the
  // Stokes projections; the time steps assemble A(b) with the extrapolated
  // field b as "current velocity".
  ParameterDatabase db_time(parmoon_db);
  db_time["space_discretization_type"] = "galerkin";
  NavierStokes<2> ns(domain, db_time, example);
  size_t n_u = ns.get_velocity_space()->get_n_dof();
  size_t n_p = ns.get_pressure_space()->get_n_dof();
  Output::print("velocity dofs: ", 2*n_u, ", pressure dofs: ", n_p, ", cells: ",
                ns.get_velocity_space()->GetCollection()->GetN_Cells(),
                ", CIP: ", use_cip ? "on" : "off", ", grad-div: ",
                (double) parmoon_db["graddiv_stab"]);
  ParameterDatabase db_mass(db_time);
  db_mass["graddiv_stab"] = 0.;
  BlockFEMatrix M = extended_matrix(ns.get_velocity_space(), ns.get_pressure_space());
  BlockFEMatrix A = extended_matrix(ns.get_velocity_space(), ns.get_pressure_space());
  BlockFEMatrix S = extended_matrix(ns.get_velocity_space(), ns.get_pressure_space());
  assemble_blocks(ns, M, nullptr, db_mass, LocalAssembling_type::TimeNavierStokesMass,
                  ns.get_solution());
  BlockVector L(ns.get_rhs());
  BlockVector u_old(ns.get_solution()), u_oldold(ns.get_solution()),
              b_vec(ns.get_solution()), rhs(ns.get_rhs()), w(ns.get_solution());
  Solver<BlockFEMatrix, BlockVector> solver(parmoon_db);

  std::string start_pressure = parmoon_db["bdf2_start_pressure"];
  // --- Stokes projection of the exact solution at the current time ----------
  // Standard Stokes projection: (grad w_h, grad v) - (r_h, div v) =
  // (-Laplace(u) + grad p, v), (div w_h, q) = 0, i.e. viscosity 1. (The
  // projection with the true small viscosity would be badly scaled for the
  // direct solver and, for non-divergence-free pairs, polluted by grad p/nu.)
  auto stokes_projection = [&](double time)
  {
    TDatabase::TimeDB->CURRENTTIME = time;
    double re_keep = TDatabase::ParamDB->RE_NR;
    TDatabase::ParamDB->P1 = 1.; // standard Stokes projection: -Laplace(u) + grad p
    TDatabase::ParamDB->RE_NR = 1.;
    Example_NSE2D stokes_example_nu1(example.get_exact(),
      std::vector<BoundCondFunct2D*>(example.get_bc(), example.get_bc() + 3),
      std::vector<BoundValueFunct2D*>(example.get_bd(), example.get_bd() + 3),
      bdf2imex_stokes_projection_coeffs(parmoon_db["example"]), 1.);
    ParameterDatabase db_s(db_time);
    db_s["reynolds_number"].set_range(1.e-10, 1.e10);
    db_s["reynolds_number"] = 1.;
    NavierStokes<2> stokes(domain, db_s, stokes_example_nu1);
    stokes.assemble_linear_terms();
    stokes.solve();
    ns.get_solution() = stokes.get_solution();
    TDatabase::ParamDB->RE_NR = re_keep;
    // the scheme does not define a pressure for the starting values; for the
    // error table either the pressure of the (viscosity 1) Stokes projection
    // or the interpolant of the exact pressure is used
    if(start_pressure == "interpolation")
    {
      ns.get_pressure().Interpolate(example.get_exact(2));
      if(S.pressure_projection_enabled())
        ns.get_pressure().project_into_L20();
    }
  };
  // --- assemble the IMEX system for the transport field b ------------------
  // S = c_M M + A(b) (+ CIP(b)), rhs = f + M w, Dirichlet data in rhs
  auto assemble_system = [&](double c_M, double time)
  {
    TDatabase::TimeDB->CURRENTTIME = time;
    // A(b) (Oseen with the convection field b, grad-div if requested) and f
    ParameterDatabase db_a(db_time);
    db_a["problem_type"] = 5;
    // Stokes part, grad-div and f
    assemble_blocks(ns, A, &L, db_a, LocalAssembling_type::NavierStokesAll, b_vec);
    // velocity-velocity block with the convective term ((b.grad)u, v): the
    // 'All' type of ParMooN's Galerkin Navier-Stokes assembling contains only
    // the linear terms, the 'NL' type the complete velocity block
    assemble_blocks(ns, A, nullptr, db_a, LocalAssembling_type::NavierStokesNL,
                    b_vec, false);
    if(use_cip)
    {
      TFEVectFunct2D b_fct(ns.get_velocity_space(), "b", b_vec.get_entries(), 2);
      assemble_CIP_NSE2D(A, b_fct, cip);
    }
    if(Output::getVerbosity() >= 3 && !use_cip)
    {
      // consistency check against the NavierStokes class (Galerkin, no CIP)
      ParameterDatabase db_chk(db_time);
      db_chk["problem_type"] = 5;
      NavierStokes<2> chk(domain, db_chk, example);
      chk.get_solution() = b_vec;
      chk.assemble_linear_terms();
      chk.assemble_nonlinear_term();
      BlockVector x(b_vec), y1(b_vec), y2(b_vec);
      for(size_t i = 0; i < x.length(); ++i)
        x.get_entries()[i] = std::sin(1. + 0.37*i);
      y1.reset(); y2.reset();
      A.apply(x, y1);
      chk.get_matrix().apply(x, y2);
      y1.add_scaled(y2, -1.);
      Output::print("DEBUG |A x - A_ns x| = ", y1.norm(), " (|A_ns x| = ", y2.norm(), ")");
    }
    if(Output::getVerbosity() >= 3)
    {
      // consistency check: B1T == (B1)^T ?
      bool tr;
      auto B1T = A.get_block(0, 2, tr);
      auto B1 = A.get_block(2, 0, tr);
      double dmax = 0., bmax = 0.;
      const int* rp = B1T->get_row_ptr();
      const int* cols = B1T->get_vector_columns();
      const double* ent = B1T->GetEntries();
      for(size_t i = 0; i < n_u; ++i)
        for(int k = rp[i]; k < rp[i+1]; ++k)
        {
          double a = ent[k];
          double b = 0.;
          try { b = B1->get(cols[k], i); } catch(...) { b = 0.; }
          dmax = std::max(dmax, std::abs(a - b));
          bmax = std::max(bmax, std::abs(a));
        }
      Output::print("DEBUG max|B1T - B1^T| = ", dmax, ", max|B1T| = ", bmax);
    }
    S = A;
    S.add_blockfe_matrix(M, c_M);
    rhs = L;
    M.apply_scaled_add(w, rhs, 1.);
    rhs.copy_nonactive(L);
  };

  // --- starting values -------------------------------------------------------
  double t = t_start;
  TDatabase::TimeDB->CURRENTTIME = t;
  std::string start = parmoon_db["bdf2_start"];
  double E0 = 0., Ep2 = 0., D = 0., max_u = 0.;
  std::ostringstream table;
  table << "# step time L2(u) H1(u) L2(p) div(u) ||b||_inf F_m\n";
  auto record = [&](int step, double F)
  {
    ns.output();
    auto err = ns.get_errors(); // 0: L2(u), 1: L2(div u), 2: H1(u), 3: L2(p)
    double l2u = err[0], divu = err[1], h1u = err[2], l2p = err[3];
    E0 = std::max(E0, l2u);
    D = std::max(D, divu);
    if(step > 0)
      Ep2 += dt*l2p*l2p;
    double un = l2_norm_velocity(ns.get_velocity_space(), ns.get_solution(), n_u);
    max_u = std::max(max_u, un);
    TFEVectFunct2D b_fct(ns.get_velocity_space(), "b", b_vec.get_entries(), 2);
    table << step << " " << std::scientific << std::setprecision(6) << t << " "
          << l2u << " " << h1u << " " << l2p << " " << divu << " "
          << max_norm(b_fct) << " " << F << "\n";
    Output::print("step ", step, ", t = ", t, ": L2(u) = ", l2u, ", H1(u) = ",
                  h1u, ", L2(p) = ", l2p, ", div = ", divu);
    if(Output::getVerbosity() >= 2)
    {
      // independent pressure error: degree-19 quadrature, point evaluation
      const TQuadFormula* qf = QuadratureFormulaDatabase::qf_from_degree(19, BFRefElements::BFUnitTriangle);
      int nq; const double *wq, *xq, *yq;
      qf->GetFormulaData(nq, wq, xq, yq);
      const TCollection* coll = ns.get_pressure_space()->GetCollection();
      double e2 = 0., mean = 0., pe[4], val[3];
      for(int i = 0; i < coll->GetN_Cells(); ++i)
      {
        const TBaseCell* c = coll->GetCell(i);
        double x0 = c->GetVertex(0)->GetX(), y0 = c->GetVertex(0)->GetY();
        double x1 = c->GetVertex(1)->GetX(), y1 = c->GetVertex(1)->GetY();
        double x2 = c->GetVertex(2)->GetX(), y2 = c->GetVertex(2)->GetY();
        double det = std::abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0));
        for(int q = 0; q < nq; ++q)
        {
          double x = x0 + xq[q]*(x1-x0) + yq[q]*(x2-x0);
          double y = y0 + xq[q]*(y1-y0) + yq[q]*(y2-y0);
          ns.get_pressure().FindValueLocal(c, i, x, y, val);
          example.get_exact(2)(x, y, pe);
          e2 += wq[q]*det*(val[0]-pe[0])*(val[0]-pe[0]);
          mean += wq[q]*det*val[0];
        }
      }
      Output::print("   independent pressure check: L2 error ", std::sqrt(e2), ", mean of p_h ", mean);
    }
  };

  if(start == "stokes_projection")
  {
    stokes_projection(t_start);
    u_oldold = ns.get_solution();
    b_vec = u_oldold;
    record(0, 0.);
    t = t_start + dt;
    stokes_projection(t);
    u_old = ns.get_solution();
  }
  else
  {
    // interpolated initial velocity and one IMEX Euler step
    ns.get_velocity().GetComponent(0)->Interpolate(example.get_exact(0));
    ns.get_velocity().GetComponent(1)->Interpolate(example.get_exact(1));
    u_oldold = ns.get_solution();
    b_vec = u_oldold;
    record(0, 0.);
    t = t_start + dt;
    w = u_oldold;
    w.scale(1./dt);
    for(size_t i = 0; i < n_p; ++i)
      w.get_entries()[2*n_u + i] = 0.;
    assemble_system(1./dt, t);
    ns.get_solution().copy_nonactive(rhs);
    solver.solve(S, rhs, ns.get_solution());
    if(S.pressure_projection_enabled())
      ns.get_pressure().project_into_L20();
    u_old = ns.get_solution();
  }
  {
    // energy F^1 = ||u^1||^2 + ||2u^1 - u^0||^2
    BlockVector e(u_old);
    e.scale(2.);
    e.add_scaled(u_oldold, -1.);
    double F1 = std::pow(l2_norm_velocity(ns.get_velocity_space(), u_old, n_u), 2)
                + std::pow(l2_norm_velocity(ns.get_velocity_space(), e, n_u), 2);
    b_vec = u_old;
    record(1, F1);
  }
  timer.restart_and_print("starting values");

  // --- BDF2-IMEX time loop --------------------------------------------------
  for(int step = 2; step <= n_steps; ++step)
  {
    t = t_start + step*dt;
    // b = 2 u^n - u^{n-1},  w = (4 u^n - u^{n-1})/(2 dt)
    b_vec = u_old;
    b_vec.scale(2.);
    b_vec.add_scaled(u_oldold, -1.);
    w = u_old;
    w.scale(4./(2.*dt));
    w.add_scaled(u_oldold, -1./(2.*dt));
    for(size_t i = 0; i < n_p; ++i)
    {
      b_vec.get_entries()[2*n_u + i] = 0.;
      w.get_entries()[2*n_u + i] = 0.;
    }
    assemble_system(3./(2.*dt), t);
    ns.get_solution().copy_nonactive(rhs);
    solver.solve(S, rhs, ns.get_solution());
    if(Output::getVerbosity() >= 2)
    {
      BlockVector res(rhs);
      S.apply_scaled_add(ns.get_solution(), res, -1.);
      for(int c = 0; c < 2; ++c)
        for(size_t i = ns.get_velocity_space()->get_n_active(); i < n_u; ++i)
          res.get_entries()[c*n_u + i] = 0.;
      Output::print("   linear solver residual: ", res.norm(), " (|rhs| = ", rhs.norm(),
                    ", |sol| = ", ns.get_solution().norm(), ")");
    }
    if(S.pressure_projection_enabled())
      ns.get_pressure().project_into_L20();
    // energy F^m = ||u^m||^2 + ||2u^m - u^{m-1}||^2
    BlockVector e(ns.get_solution());
    e.scale(2.);
    e.add_scaled(u_old, -1.);
    double F = std::pow(l2_norm_velocity(ns.get_velocity_space(), ns.get_solution(), n_u), 2)
               + std::pow(l2_norm_velocity(ns.get_velocity_space(), e, n_u), 2);
    record(step, F);
    u_oldold = u_old;
    u_old = ns.get_solution();
  }
  timer.restart_and_print("time loop");

  int n_sample = parmoon_db["bdf2_sample_grid"];
  if(n_sample > 0)
  {
    double om_min, om_max, tv;
    std::string base = parmoon_db["output_basename"];
    sample_on_grid(ns.get_velocity(), n_sample, base + "_sample.txt", om_min, om_max, tv);
    Output::print("SAMPLE: vorticity range [", om_min, ", ", om_max, "], discrete TV ",
                  tv, ", written to ", base, "_sample.txt");
  }
  auto err = ns.get_errors();
  Output::print("\nRESULTS: E_0 = ", std::scientific, std::setprecision(4), E0,
                ", E_p = ", std::sqrt(Ep2), ", D = ", D,
                ", L2(u(T)) = ", err[0], ", H1(u(T)) = ", err[2],
                ", max_m ||u_h^m||_0 = ", max_u);
  Output::print("RESULTS_LINE ", E0, " ", std::sqrt(Ep2), " ", D, " ", err[0],
                " ", err[2], " ", max_u);
  if(!table_file.empty())
  {
    std::ofstream out(table_file);
    out << table.str();
    out << "# E_0 E_p D L2(u(T)) H1(u(T)) max_u\n# RESULTS " << std::scientific
        << E0 << " " << std::sqrt(Ep2) << " " << D << " " << err[0] << " "
        << err[2] << " " << max_u << "\n";
  }
  parmoon::parmoon_finalize();
}
