#ifdef __2D__
#include "TimeConvectionDiffusionTR.h"
#include "Assemble2D.h"
#include "AuxParam2D.h"
#include "CDErrorEstimator.h"
#include "ConvDiff.h"
#include "Database.h"
#include "LocalAssembling.h"
#include "MooNMD_Io.h"
#include "SquareMatrix2D.h"
#include <algorithm>
#include <cmath>

TimeConvectionDiffusionTR::TimeConvectionDiffusionTR(
  const TDomain& domain, const ParameterDatabase& param_db,
  const Example_TimeCD2D& ex)
 : space(new TFESpace2D(domain.GetCollection(It_Finest, 0), "space",
                        ex.get_bc(0), TDatabase::ParamDB->ANSATZ_ORDER)),
   example(ex),
   steady_example(ex.get_exact(),
                  std::vector<BoundCondFunct2D*>(ex.get_bc(), ex.get_bc() + 1),
                  std::vector<BoundValueFunct2D*>(ex.get_bd(), ex.get_bd() + 1),
                  ex.get_coeffs()),
   db(param_db), M(BlockFEMatrix::CD2D(space)), A(BlockFEMatrix::CD2D(space)),
   S(BlockFEMatrix::CD2D(space)),
   u(M, false), d(M, false), d_old(M, false), u_saved(M, false),
   d_saved(M, false), u_pred(M, false), rhs(M, true), work(M, true),
   solver(param_db), writer(param_db), t(0.)
{
  u_fct.reset(new TFEFunction2D(space, "u", u.get_entries()));
  d_fct.reset(new TFEFunction2D(space, "dt_u", d.get_entries()));
  writer.add_fe_function(u_fct.get());
  writer.add_fe_function(d_fct.get());
  t = db["time_start"];
}

void TimeConvectionDiffusionTR::initialize()
{
  t = db["time_start"];
  TDatabase::TimeDB->CURRENTTIME = t;
  u_fct->Interpolate(example.get_initial_cond(0));
  // the Dirichlet degrees of freedom of the initial value are set to the
  // boundary data, not to u_0: for data which are incompatible at t = 0
  // (u_0 = 0 and g = 1 on a part of the boundary, as in Example 5 of de
  // Frutos et al. 2014) interpolating u_0 everywhere makes the boundary
  // values jump from 0 to g within the first time step, so the discrete
  // time derivative of the first step is O(1/k) and the error estimate and
  // the step size control are meaningless.
  set_dirichlet_values(t, u);
  compute_time_derivative(t);
  d_old = d;
}

void TimeConvectionDiffusionTR::assemble(double time)
{
  TDatabase::TimeDB->CURRENTTIME = time;
  TDatabase::TimeDB->THETA1 = 1.;
  TDatabase::TimeDB->CURRENTTIMESTEPLENGTH = 1.;
  std::vector<const TFEFunction2D*> fcts = {u_fct.get()};
  LocalAssembling<2> la(db, LocalAssembling_type::TCDStiffMassRhs, fcts,
                        example.get_coeffs());
  auto A_block = A.get_blocks_uniquely()[0].get();
  auto M_block = M.get_blocks_uniquely()[0].get();
  TSquareMatrix2D* sq[2] = {reinterpret_cast<TSquareMatrix2D*>(A_block),
                            reinterpret_cast<TSquareMatrix2D*>(M_block)};
  sq[0]->reset();
  sq[1]->reset();
  rhs.reset();
  double* rhs_entries = rhs.get_entries();
  const TFESpace2D* sp = space.get();
  BoundCondFunct2D* bc = space->get_boundary_condition();
  BoundValueFunct2D* bd = example.get_bd(0);
  Assemble2D(1, &sp, 2, sq, 0, nullptr, 1, &rhs_entries, &sp, &bc, &bd, la, true);
}

void TimeConvectionDiffusionTR::compute_time_derivative(double time)
{
  assemble(time);
  work = rhs;
  A.apply_scaled_add(u, work, -1.);
  // Dirichlet rows: time derivative of the boundary data
  set_dirichlet_time_derivative(time, work);
  d.copy_nonactive(work);
  solver.solve(M, work, d);
}

// L2 norm by quadrature (an assembled mass matrix has identity rows for
// the Dirichlet dofs, which would weight the boundary values with 1)
double TimeConvectionDiffusionTR::l2_norm(const BlockVector& v)
{
  TFEFunction2D f(space, "e", const_cast<double*>(v.get_entries()));
  return f.get_L2_norm();
}

void TimeConvectionDiffusionTR::save_state()
{
  u_saved = u;
  d_saved = d;
}

void TimeConvectionDiffusionTR::restore_state()
{
  u = u_saved;
  d = d_saved;
}

void TimeConvectionDiffusionTR::predict(double k, double r)
{
  u_pred = u;
  u_pred.add_scaled(d, 0.5*k*(2. + r));
  u_pred.add_scaled(d_old, -0.5*k*r);
}

void TimeConvectionDiffusionTR::tr_step(double time, double k)
{
  assemble(time + k);
  S = A;
  std::vector<std::vector<size_t>> cells = {{0, 0}};
  S.scale_blocks_actives(0.5*k, cells);
  S.add_blockfe_matrix(M);
  work = u;
  work.add_scaled(d, 0.5*k);
  BlockVector b(rhs);
  b.scaleActive(0.5*k);
  M.apply_scaled_add(work, b, 1.);
  b.copy_nonactive(rhs);
  u.copy_nonactive(rhs);
  solver.solve(S, b, u);
  // consistent time derivative
  d = u;
  d.add_scaled(u_saved, -1.);
  d.scale(2./k);
  d.add_scaled(d_saved, -1.);
  t = time + k;
  set_dirichlet_time_derivative(t, d); // see TimeNavierStokesSPG::tr_step
}

double TimeConvectionDiffusionTR::predictor_difference_norm()
{
  BlockVector e(u);
  e.add_scaled(u_pred, -1.);
  return l2_norm(e);
}

void TimeConvectionDiffusionTR::average_with_saved(double t_mid)
{
  u.add_scaled(u_saved, 1.);
  u.scale(0.5);
  d.add_scaled(d_saved, 1.);
  d.scale(0.5);
  t = t_mid;
  set_dirichlet_values(t, u);          // see TimeNavierStokesSPG
  set_dirichlet_time_derivative(t, d);
}

void TimeConvectionDiffusionTR::set_dirichlet_values(double time, BlockVector& v)
{
  const double t_save = TDatabase::TimeDB->CURRENTTIME;
  TDatabase::TimeDB->CURRENTTIME = time;
  TFEFunction2D f(space, "bd", v.get_entries());
  f.SetDirichletBC(space->get_boundary_condition(), example.get_bd(0));
  TDatabase::TimeDB->CURRENTTIME = t_save;
}

void TimeConvectionDiffusionTR::set_dirichlet_time_derivative(double time,
                                                              BlockVector& v)
{
  const double eps = 1e-5*std::max(1., std::abs(time));
  BlockVector gp(v), gm(v);
  set_dirichlet_values(time + eps, gp);
  set_dirichlet_values(time - eps, gm);
  size_t n_active = space->get_n_active();
  for(size_t i = n_active; i < v.length(); ++i)
    v.get_entries()[i] = (gp.get_entries()[i] - gm.get_entries()[i])/(2.*eps);
}

void TimeConvectionDiffusionTR::shift_derivative()
{
  d_old = d_saved;
}

void TimeConvectionDiffusionTR::estimate(ErrorEstimator<2>& estimator)
{
  auto* cd = dynamic_cast<CDErrorEstimator<2>*>(&estimator);
  if(cd == nullptr)
    ErrThrow("TimeConvectionDiffusionTR::estimate needs a CDErrorEstimator");
  TDatabase::TimeDB->CURRENTTIME = t;
  cd->set_time_derivative(d_fct.get());
  cd->estimate(steady_example, *u_fct);
  cd->set_time_derivative(nullptr);
}

const TCollection* TimeConvectionDiffusionTR::get_collection() const
{
  return space->GetCollection();
}

std::vector<SolutionTransfer::Snapshot>
TimeConvectionDiffusionTR::record_state() const
{
  TFEFunction2D dold_fct(space, "d_old", const_cast<double*>(d_old.get_entries()));
  return {SolutionTransfer::record(*u_fct), SolutionTransfer::record(*d_fct),
          SolutionTransfer::record(dold_fct)};
}

void TimeConvectionDiffusionTR::apply_state(
  const std::vector<SolutionTransfer::Snapshot>& maps)
{
  if(maps.size() != 3)
    ErrThrow("TimeConvectionDiffusionTR::apply_state: three maps expected");
  SolutionTransfer::apply(*u_fct, maps[0]);
  SolutionTransfer::apply(*d_fct, maps[1]);
  TFEFunction2D dold_fct(space, "d_old", d_old.get_entries());
  SolutionTransfer::apply(dold_fct, maps[2]);
  // The Dirichlet degrees of freedom of the new mesh must carry the boundary
  // data, not the values interpolated from the old mesh. A new boundary node
  // created by refinement inherits the mean of its parent's values, which
  // differs from g wherever g is not affine on the parent edge (in particular
  // next to a discontinuity of g). The next time step would then force this
  // node from the interpolated value to g within one step and produce a
  // discrete time derivative of size O(1/k) in the neighbouring cells.
  set_dirichlet_values(t, u);
  set_dirichlet_time_derivative(t, d);
  u_saved = u;
  d_saved = d;
  u_pred = u;
}

void TimeConvectionDiffusionTR::copy_time_state(const AdaptiveTimeSystem& old)
{
  const auto* o = dynamic_cast<const TimeConvectionDiffusionTR*>(&old);
  if(o != nullptr)
    t = o->t;
}

void TimeConvectionDiffusionTR::output(int index)
{
  TDatabase::TimeDB->CURRENTTIME = t;
  writer.write(index);
}

double TimeConvectionDiffusionTR::exact_error() const
{
  if(example.get_exact().empty() || example.get_exact(0) == nullptr)
    return -1.;
  TDatabase::TimeDB->CURRENTTIME = t;
  TAuxParam2D aux;
  MultiIndex2D all_derivatives[3] = {MultiIndex2D::D00, MultiIndex2D::D10,
                                     MultiIndex2D::D01};
  const TFESpace2D* sp = space.get();
  std::array<double, 5> err;
  u_fct->GetErrors(example.get_exact(0), 3, all_derivatives, 5,
                   conv_diff_l2_h1_linf_error<2>, example.get_coeffs(), &aux, 1,
                   &sp, err.data(), false,
                   [](const TBaseCell*, int){return false;}, db);
  return err[2]; // SUPG norm
}
#endif // __2D__
