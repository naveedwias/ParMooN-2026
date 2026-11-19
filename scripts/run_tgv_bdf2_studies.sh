#!/bin/bash
# =============================================================================
# BDF2-IMEX on the periodic Taylor-Green vortex (Example_TimeNSE2D 35,
# input data/input_files/tnse2d_bdf2imex_tgv.dat): the four discretizations
# of the manuscript at high Reynolds numbers, given by scheme labels
# (scripts/transient_study.py):
#   TH              Taylor-Hood P2/P1, no grad-div, no CIP
#   TH-gd           Taylor-Hood with grad-div, gamma = 1 (manuscript, smooth
#                   example; --set graddiv_stab=0.1 for the shear-layer value)
#   SV-G            Scott-Vogelius P2/P1disc on barycentric meshes, Galerkin
#   SV-CIP-weak / -medium / -strong
#                   Scott-Vogelius with CIP, (delta_1, delta_2, delta_3) =
#                   s (1e-2, 1e-4, 1e-6) with s = 0.1, 1 (manuscript), 10;
#                   cip_s=... gives any s (study 4)
# Periodic boundaries need 'bdf2_start: euler': the Stokes projection is not
# defined without Dirichlet data. With this start u^0 is the interpolant of
# u(0), which is not divergence-free, so D (the maximum over all m >= 0)
# shows ||div u_h^0||; the divergence of later steps is in the table
# (column div(u), round-off for Scott-Vogelius).
#
# Run from the build directory (next to ParMooN-2026/):
#   bash ../ParMooN-2026/scripts/run_tgv_bdf2_studies.sh        # all
#   bash ../ParMooN-2026/scripts/run_tgv_bdf2_studies.sh 2      # only study 2 (1 ... 4)
# Finished runs are skipped. Results in tnse_bdf2_taylor_green/ (folder and
# files named <problem>_<example>, scripts/naming.py), the runs collected in
# tnse_bdf2_taylor_green_runs.jsonl (scripts/run_store.py) under keys named by
# the labels (e.g. variants_Re1.0e6_SV-CIP-medium):
# tnse_bdf2_taylor_green_<name>_results.txt (E_0, E_p, D, L2(u(T)), H1(u(T)),
# max ||u_h||) and tnse_bdf2_taylor_green_<name>_tables.pdf (errors,
# divergence and energy F_m over time; exact F_m = 4 pi^2 exp(-4 nu t)).
# The rate columns of _results.txt are meaningful only in study 3 (mesh
# refinement); in the other studies the runs are variants, not refinements.
# =============================================================================
set -e
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
B=$E/tnse2d_bdf2imex_tgv.dat
OUT=tnse_bdf2_taylor_green     # <problem>_<example> (scripts/naming.py)
ONLY=${1:-all}
run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
SCHEMES="TH,TH-gd,SV-G,SV-CIP-medium"

# 1. The four variants at Re = 1e4, 1e6, 1e8, level 3, T = 5
if run 1; then
  for re in 1.0e4 1.0e6 1.0e8; do
    python3 $S --program bdf2 --base $B --name variants_Re$re --outdir $OUT \
        --set reynolds_number=$re --vary scheme=$SCHEMES --no-meshes
  done
fi

# 2. Long time at Re = 1e6 (T = 20): growth of the errors in time
if run 2; then
  python3 $S --program bdf2 --base $B --name long_Re1e6 --outdir $OUT \
      --set reynolds_number=1.0e6 time_end=20.0 --vary scheme=$SCHEMES --no-meshes
fi

# 3. Mesh refinement at Re = 1e6 (dt = h^1.5 as in the manuscript: halving h
#    divides dt by about 2.83), SV-CIP and TH-gd
if run 3; then
  python3 $S --program bdf2 --base $B --name meshes_SVCIP --outdir $OUT \
      --set reynolds_number=1.0e6 scheme=SV-CIP-medium \
      --vary refinement_n_initial_steps/time_step_length=2/0.05,3/0.0177,4/0.00625 --no-meshes
  python3 $S --program bdf2 --base $B --name meshes_THgd --outdir $OUT \
      --set reynolds_number=1.0e6 scheme=TH-gd \
      --vary refinement_n_initial_steps/time_step_length=2/0.05,3/0.0177,4/0.00625 --no-meshes
fi

# 4. CIP strength: Scott-Vogelius with (delta_1, delta_2, delta_3) =
#    s (1e-2, 1e-4, 1e-6), s = 0.1 (weak), 0.3, 1 (medium), 3, 10 (strong),
#    and the Galerkin method (s = 0) for comparison, Re = 1e6
if run 4; then
  python3 $S --program bdf2 --base $B --name cipscale_Re1e6 --outdir $OUT \
      --set reynolds_number=1.0e6 scheme=SV-CIP --vary cip_s=0,0.1,0.3,1,3,10 --no-meshes
fi

echo "done: results in $OUT/"
