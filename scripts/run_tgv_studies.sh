#!/bin/bash
# =============================================================================
# Studies for the two-dimensional Taylor-Green vortex (Example_TimeNSE2D 35,
# input data/input_files/tnse2d_adaptive_spg_tgv.dat, domain (0, 2 pi)^2,
# periodic in x and y, exact solution with f = 0) with the program
# tnse2d_adaptive_spg, on uniform meshes: adaptive meshes are not supported
# with periodic boundary conditions (and would not help for this smooth,
# spatially periodic solution: with Dirichlet data, tgv_periodic: false, the
# estimator spreads the refinement almost uniformly and the error at equal
# DoFs is not better than on uniform meshes).
#
# Run from the build directory (next to ParMooN-2026/):
#   bash ../ParMooN-2026/scripts/run_tgv_studies.sh        # all studies
#   bash ../ParMooN-2026/scripts/run_tgv_studies.sh 2      # only study 2 (1 ... 4)
#
# Finished runs are skipped, so the script can be restarted. The runs are
# collected in tnse_taylor_green/tnse_taylor_green_runs.jsonl (scripts/run_store.py),
# the vtk files of a run in tnse_taylor_green/tnse_taylor_green_files/<run>/.
# Figures of a study: repeat its transient_study.py command with --plot-only.
# Field plots (stream function, vorticity, errors) and the energy/dissipation
# analysis of any run directory:
#   python3 ../ParMooN-2026/scripts/tgv_plots.py tnse_taylor_green --run <run key>
#
# Exact quantities (L = 2 pi, k = A = 1): E = pi^2 exp(-4 nu t),
# Z = 2 pi^2 exp(-4 nu t), -dE/dt = 2 nu Z; omega in [-2F, 2F], psi in [-F, F].
# =============================================================================
set -e
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
P=../ParMooN-2026/scripts/tgv_plots.py
B=$E/tnse2d_adaptive_spg_tgv.dat
OUT=tnse_taylor_green     # <problem>_<example> (scripts/naming.py)
ONLY=${1:-all}
run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
# field plots of the finished runs whose keys match the patterns (quoted shell patterns)
plots() {
  for pat in "$@"; do
    for key in $(python3 -c "import sys; sys.path.insert(0, '../ParMooN-2026/scripts')
from fnmatch import fnmatch; from run_store import RunStore
print(' '.join(k for k, r in sorted(RunStore('$OUT', '$OUT').records().items())
               if fnmatch(k, '$pat') and not r.get('failed')))"); do
      python3 $P $OUT --run "$key" || true
    done
  done
}

# -----------------------------------------------------------------------------
# 1. Reference run: uniform level 4, nu = 1e-2, T = 10, adaptive time steps
#    (TR-AB2, backward Euler start), vtk output every time unit.
#    Output: tnse_taylor_green/tnse_taylor_green_ref_tables.pdf, in the run folder
#    tnse_taylor_green_{time,fields}.pdf and tnse_taylor_green_summary.txt
if run 1; then
  python3 $S --program tnse --base $B --name ref --outdir $OUT \
      --set refinement_n_initial_steps=4 time_end=10.0 output_write_ps=false \
      --vary reynolds_number=100.0 --no-meshes
  plots "ref_reynolds_number-100.0"
fi

# -----------------------------------------------------------------------------
# 2. Spatial convergence on uniform meshes (time error small: TOLT 1e-6),
#    T = 1. Rates of all errors, estimator contributions, effectivity.
#    Output: tnse_taylor_green/tnse_taylor_green_conv_convergence.{txt,pdf},
#    _contributions.pdf, _norms.pdf
if run 2; then
  python3 $S --program tnse --base $B --name conv --outdir $OUT \
      --set time_end=1.0 trab2_tolt=1e-6 output_write_vtk=false output_write_ps=false \
      --vary refinement_n_initial_steps=2,3,4,5 --no-meshes
fi

# -----------------------------------------------------------------------------
# 3. Small viscosity: Galerkin, grad-div only, SUPG/PSPG only, SUPG/PSPG +
#    grad-div on the uniform level-4 mesh, nu = 1e-3, 1e-4, 1e-5, T = 10.
#    The solution decays only slowly; look at the energy and enstrophy
#    (numerical dissipation), the vorticity extrema and the pressure error
#    (pressure robustness: grad-div). The estimator is meaningful only for
#    the full method.
#    Output: tnse_taylor_green/tnse_taylor_green_small_nu_Re<Re>_{tables,flow}.pdf,
#    field plots per run.
if run 3; then
  for re in 1000.0 10000.0 100000.0; do
    python3 $S --program tnse --base $B --name small_nu_Re$re --outdir $OUT \
        --set reynolds_number=$re refinement_n_initial_steps=4 time_end=10.0 \
              adaptive_output_time=2.0 output_write_ps=false \
        --vary spg_delta0/spg_mu0=0.0/0.0,0.0/0.5,0.5/0.0,0.5/0.5 --no-meshes
    plots "small_nu_Re${re}_spg_delta0_spg_mu0-*"
  done
fi

# -----------------------------------------------------------------------------
# 4. Time discretization: TOLT sweep on the uniform level-4 mesh (nu = 1e-2),
#    and the start without backward Euler step for comparison.
#    Output: tnse_taylor_green/tnse_taylor_green_tolt_{tables,flow}.pdf, _summary.txt
if run 4; then
  python3 $S --program tnse --base $B --name tolt --outdir $OUT \
      --set refinement_n_initial_steps=4 time_end=2.0 output_write_vtk=false output_write_ps=false \
      --vary trab2_tolt=1e-4,1e-5,1e-6 trab2_first_step=backward_euler,semidiscrete --no-meshes
fi

echo "done: results in $OUT/"
