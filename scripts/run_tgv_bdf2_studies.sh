#!/bin/bash
# =============================================================================
# BDF2-IMEX on the periodic Taylor-Green vortex (Example_TimeNSE2D 35,
# input data/input_files/tnse2d_bdf2imex_tgv.dat): the four discretizations
# of the manuscript at high Reynolds numbers.
#   TH     : Taylor-Hood P2/P1, no grad-div, no CIP
#   TH-gd  : Taylor-Hood with grad-div gamma = 0.1
#   SV-G   : Scott-Vogelius P2/P1disc on barycentric meshes, Galerkin
#   SV-CIP : Scott-Vogelius with CIP (cip_delta1..3 of the input file)
# Periodic boundaries need 'bdf2_start: euler': the Stokes projection is not
# defined without Dirichlet data. With this start u^0 is the interpolant of
# u(0), which is not divergence-free, so D (the maximum over all m >= 0)
# shows ||div u_h^0||; the divergence of later steps is in the table
# (column div(u), round-off for Scott-Vogelius).
#
# Run from the build directory (next to ParMooN-2026/):
#   bash ../ParMooN-2026/scripts/run_tgv_bdf2_studies.sh        # all
#   bash ../ParMooN-2026/scripts/run_tgv_bdf2_studies.sh 2      # only study 2
# Finished runs are skipped. Results: tgv_bdf2/<name>_results.txt (E_0, E_p,
# D, L2(u(T)), H1(u(T)), max ||u_h||) and <name>_tables.pdf (errors,
# divergence and energy F_m over time; exact F_m = 4 pi^2 exp(-4 nu t)).
# The rate columns of _results.txt are meaningful only in study 3 (mesh
# refinement); in studies 1 and 2 the runs are variants, not refinements.
# =============================================================================
set -e
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
B=$E/tnse2d_bdf2imex_tgv.dat
OUT=tgv_bdf2
ONLY=${1:-all}
run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
TH="VELOCITY_SPACE/PRESSURE_SPACE/refinement_final_step_barycentric/graddiv_stab/cip_delta1/cip_delta2/cip_delta3"
VARIANTS="2/1/false/0.0/0.0/0.0/0.0,2/1/false/0.1/0.0/0.0/0.0,12/-4711/true/0.0/0.0/0.0/0.0,12/-4711/true/0.0/1.0e-2/1.0e-4/1.0e-6"

# 1. The four variants at Re = 1e4, 1e6, 1e8, level 3, T = 5
if run 1; then
  for re in 1.0e4 1.0e6 1.0e8; do
    python3 $S --program bdf2 --base $B --name variants_Re$re --outdir $OUT \
        --set reynolds_number=$re --vary $TH=$VARIANTS --no-meshes
  done
fi

# 2. Long time at Re = 1e6 (T = 20): growth of the errors in time
if run 2; then
  python3 $S --program bdf2 --base $B --name long_Re1e6 --outdir $OUT \
      --set reynolds_number=1.0e6 time_end=20.0 --vary $TH=$VARIANTS --no-meshes
fi

# 3. Mesh refinement at Re = 1e6 (dt = h^1.5 as in the manuscript: halving h
#    divides dt by about 2.83), SV-CIP and TH-gd
if run 3; then
  python3 $S --program bdf2 --base $B --name meshes_SVCIP --outdir $OUT \
      --set reynolds_number=1.0e6 \
      --vary refinement_n_initial_steps/time_step_length=2/0.05,3/0.0177,4/0.00625 --no-meshes
  python3 $S --program bdf2 --base $B --name meshes_THgd --outdir $OUT \
      --set reynolds_number=1.0e6 VELOCITY_SPACE=2 PRESSURE_SPACE=1 \
            refinement_final_step_barycentric=false graddiv_stab=0.1 \
            cip_delta1=0.0 cip_delta2=0.0 cip_delta3=0.0 \
      --vary refinement_n_initial_steps/time_step_length=2/0.05,3/0.0177,4/0.00625 --no-meshes
fi

echo "done: results in $OUT/"
