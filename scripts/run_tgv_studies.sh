#!/bin/bash
# =============================================================================
# Studies for the two-dimensional Taylor-Green vortex (Example_TimeNSE2D 35,
# input data/input_files/tnse2d_adaptive_spg_tgv.dat, domain (0, 2 pi)^2,
# exact solution with f = 0) with the program tnse2d_adaptive_spg.
#
# Run from the build directory (next to ParMooN-2026/):
#   bash ../ParMooN-2026/scripts/run_tgv_studies.sh        # all studies
#   bash ../ParMooN-2026/scripts/run_tgv_studies.sh 2      # only study 2
#
# Finished runs (with a table) are skipped, so the script can be restarted.
# Figures of a study: repeat its transient_study.py command with --plot-only.
# Field plots (stream function, vorticity, errors) and the energy/dissipation
# analysis of any run directory:
#   python3 ../ParMooN-2026/scripts/tgv_plots.py tgv/<run directory>
#
# Exact quantities (L = 2 pi, k = A = 1): E = pi^2 exp(-4 nu t),
# Z = 2 pi^2 exp(-4 nu t), -dE/dt = 2 nu Z; omega in [-2F, 2F], psi in [-F, F].
# =============================================================================
set -e
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
P=../ParMooN-2026/scripts/tgv_plots.py
B=$E/tnse2d_adaptive_spg_tgv.dat
OUT=tgv
ONLY=${1:-all}
run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
plots() { for d in "$@"; do [ -f "$d/table.txt" ] && python3 $P "$d" || true; done; }

# -----------------------------------------------------------------------------
# 1. Reference run: uniform level 4, nu = 1e-2, T = 10, adaptive time steps
#    (TR-AB2, backward Euler start), vtk output every time unit.
#    Output: tgv/ref_tables.pdf, tgv/<run>/tgv_{time,fields}.pdf, _summary.txt
if run 1; then
  python3 $S --program tnse --base $B --name ref --outdir $OUT \
      --set refinement_n_initial_steps=4 time_end=10.0 output_write_ps=false \
      --vary reynolds_number=100.0 --no-meshes
  plots $OUT/ref_reynolds_number-100.0
fi

# -----------------------------------------------------------------------------
# 2. Spatial convergence on uniform meshes (time error small: TOLT 1e-6),
#    T = 1. Rates of all errors, estimator contributions, effectivity.
#    Output: tgv/conv_convergence.{txt,pdf}, _contributions.pdf, _norms.pdf
if run 2; then
  python3 $S --program tnse --base $B --name conv --outdir $OUT \
      --set time_end=1.0 trab2_tolt=1e-6 output_write_vtk=false output_write_ps=false \
      --vary refinement_n_initial_steps=2,3,4,5 --no-meshes
fi

# -----------------------------------------------------------------------------
# 3. Adaptive meshes against uniform meshes (same T, nu): tolerance sweep
#    from level 3, error versus the time-averaged DoFs; compare with study 2.
#    For this smooth, spatially periodic solution the estimator spreads the
#    refinement almost uniformly: adaptivity is not expected to help (at
#    equal DoFs the SPG-norm error is the same as on uniform meshes; the L2
#    error and the energy are not better). The study documents this.
#    Output: tgv/ada_convergence.{txt,pdf}, meshes per run, field plots.
if run 3; then
  python3 $S --program tnse --base $B --name ada --outdir $OUT \
      --set time_end=1.0 trab2_tolt=1e-6 refinement_n_initial_steps=3 adaptive_mesh=true \
            adaptive_max_cell_level=8 adaptive_output_time=0.5 \
      --vary adaptive_tol1/adaptive_tol2=2e-2/1e-3,1e-2/5e-4,5e-3/2.5e-4 --dofs mean
  plots $OUT/ada_adaptive_tol1_adaptive_tol2-5e-32.5e-4
fi

# -----------------------------------------------------------------------------
# 4. Small viscosity: Galerkin, grad-div only, SUPG/PSPG only, SUPG/PSPG +
#    grad-div on the uniform level-4 mesh, nu = 1e-3, 1e-4, 1e-5, T = 10.
#    The solution decays only slowly; look at the energy and enstrophy
#    (numerical dissipation), the vorticity extrema and the pressure error
#    (pressure robustness: grad-div). The estimator is meaningful only for
#    the full method.
#    Output: tgv/small_nu_Re<Re>_{tables,flow}.pdf, field plots per run.
if run 4; then
  for re in 1000.0 10000.0 100000.0; do
    python3 $S --program tnse --base $B --name small_nu_Re$re --outdir $OUT \
        --set reynolds_number=$re refinement_n_initial_steps=4 time_end=10.0 \
              adaptive_output_time=2.0 output_write_ps=false \
        --vary spg_delta0/spg_mu0=0.0/0.0,0.0/0.5,0.5/0.0,0.5/0.5 --no-meshes
    plots $OUT/small_nu_Re${re}_spg_delta0_spg_mu0-*
  done
fi

# -----------------------------------------------------------------------------
# 5. Time discretization: TOLT sweep on the uniform level-4 mesh (nu = 1e-2),
#    and the start without backward Euler step for comparison.
#    Output: tgv/tolt_{tables,flow}.pdf, _summary.txt
if run 5; then
  python3 $S --program tnse --base $B --name tolt --outdir $OUT \
      --set refinement_n_initial_steps=4 time_end=2.0 output_write_vtk=false output_write_ps=false \
      --vary trab2_tolt=1e-4,1e-5,1e-6 trab2_first_step=backward_euler,semidiscrete --no-meshes
fi

echo "done: results in $OUT/"
