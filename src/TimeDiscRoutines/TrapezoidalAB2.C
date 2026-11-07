#include "TrapezoidalAB2.h"
#include "MooNMD_Io.h"
#include <algorithm>
#include <cmath>

ParameterDatabase TrapezoidalAB2::default_database()
{
  ParameterDatabase db("TR-AB2 time integrator database");
  db.add("time_start", 0., "start time", -1.e10, 1.e10);
  db.add("time_end", 1., "end time", -1.e10, 1.e10);
  db.add("time_step_length", 0.01, "step length if trab2_adaptive is false",
         0., 1.e10);
  db.add("trab2_adaptive", true, "variable step size control (AB2 estimate)");
  db.add("trab2_tolt", 1.e-5, "tolerance TOLT of the error estimate", 0., 1.);
  db.add("trab2_dt0", 1.e-10, "initial step (adaptive mode)", 0., 1.e10);
  db.add("trab2_dt_min", 1.e-12, "minimal step (no rejection below)", 0., 1.e10);
  db.add("trab2_dt_max", 0.1, "maximal step", 0., 1.e10);
  db.add("trab2_factor", 0.85, "safety factor of the step size update", 0., 1.);
  db.add("trab2_averaging_steps", 10, "average every n* steps (0: off)",
         0, 1000000);
  return db;
}

TrapezoidalAB2::TrapezoidalAB2(const ParameterDatabase& param_db)
{
  ParameterDatabase db = default_database();
  db.merge(param_db, false);
  t = db["time_start"];
  t_end = db["time_end"];
  fixed_step = !db["trab2_adaptive"];
  k = fixed_step ? (double) db["time_step_length"] : (double) db["trab2_dt0"];
  k_old = k;
  tolt = db["trab2_tolt"];
  k_min = db["trab2_dt_min"];
  k_max = db["trab2_dt_max"];
  factor = db["trab2_factor"];
  n_star = db["trab2_averaging_steps"];
  n_accepted = n_rejected = n_since_avg = 0;
  first_step = true;
  Output::print<2>("TR-AB2: ", fixed_step ? "fixed step " : "adaptive, TOLT = ",
                   fixed_step ? k : tolt, ", averaging every ", n_star, " steps");
}

double TrapezoidalAB2::advance(TrapezoidalAB2_System& sys)
{
  double err = 0.;
  while(true)
  {
    k = std::min(k, t_end - t);
    double r = first_step ? 1. : k/k_old;
    sys.save_state();
    sys.predict(k, r);
    sys.tr_step(t, k);
    err = sys.predictor_difference_norm()/(3.*(1. + 1./r));
    if(!fixed_step && err > tolt && k > k_min)
    {
      sys.restore_state();
      n_rejected++;
      k = factor*k*std::cbrt(tolt/err);
      Output::print<2>("TR-AB2: step rejected, |e| = ", err, ", new dt = ", k);
      continue;
    }
    break;
  }
  n_accepted++;
  n_since_avg++;
  t += k;
  double k_eff = k;
  if(!fixed_step && n_star > 0 && n_since_avg >= n_star)
  {
    sys.average_with_saved();
    t -= 0.5*k;
    k_eff = 0.5*k;
    n_since_avg = 0;
  }
  sys.shift_derivative();
  k_old = k_eff;
  first_step = false;
  if(!fixed_step)
    k = std::min(factor*k*std::cbrt(tolt/std::max(err, 1e-300)), k_max);
  Output::print("TR-AB2: t = ", t, " (step ", n_accepted, ", |e| = ", err,
                ", next dt = ", k, ")");
  return err;
}
