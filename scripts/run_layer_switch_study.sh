#!/usr/bin/env bash
# Spatial behaviour of the layer-switch examples 36 / 37 (LAYER_SWITCH_NOTES.md).
#
# Run from the build folder next to ParMooN-2026 (after make tnse2d_adaptive_spg):
#   bash ../ParMooN-2026/scripts/run_layer_switch_study.sh [part ...]
# parts (default: all, in this order):
#   uniform   uniform meshes, levels $LEVELS (no mesh adaptation): spatial convergence
#   adaptive  adaptive meshes, TOL1 = 5e-2, 2.5e-2, 1.25e-2 (TOL2 = TOL1/20)
#   time      time tolerance at TOL1 = 5e-2: TOLT = 1e-3, 1e-4, 1e-5 (is the time error small?)
#   plots     figures of all finished runs: <prefix>_<study>_time.png (dt, cells, errors,
#             eta over time) and <prefix>_<study>_maxt.png/.txt (max over time vs DoFs,
#             with observed orders); transient_study.py writes its own tables as well,
#             e.g. <prefix>_uniform_spatial.pdf and _convergence.txt
# Settings (environment variables):
#   EX=a (example 36, default) or EX=b (example 37)
#   LEVELS="4,5,6,7"   uniform levels (h = 1/16 ... 1/128); level 7 takes long
#   TOLT=1e-4          time tolerance of the uniform and adaptive studies; with 1e-3 the time
#                      error dominates after the layer switch (LAYER_SWITCH_NOTES.md, 3.11)
#   EXTRA="..."        further key=value settings for all runs, e.g. EXTRA="reynolds_number=1e4"
# Runs that are already in the store are skipped (transient_study.py); --redo repeats them.
set -euo pipefail

EX=${EX:-a}
LEVELS=${LEVELS:-4,5,6,7}
TOLT=${TOLT:-1e-4}
EXTRA=${EXTRA:-}
R=${R:-../ParMooN-2026}
S=$R/scripts/transient_study.py
BASE=$R/data/input_files/tnse2d_adaptive_spg_layer_switch_$EX.dat
EXE=${EXE:-./parMooN_tnse2d_adaptive_spg_SEQUENTIAL}
if [ "$EX" = a ]; then PREFIX=tnse_layer_switch_x0y0; NAME="example 36"; else PREFIX=tnse_layer_switch_x1y0; NAME="example 37"; fi
OUT=$PREFIX                       # folder of the run store (transient_study.py default)
J=$OUT/${PREFIX}_runs.jsonl
PARTS=${*:-uniform adaptive time plots}
mkdir -p vtk

for part in $PARTS; do
  case $part in
  uniform)
    # fixed uniform meshes: the error is the spatial error of each level (plus the
    # time error of TOLT; compare with the 'time' part)
    python3 $S --program tnse --exe $EXE --base $BASE --name uniform \
        --set adaptive_mesh=false adaptive_initial_mesh=false trab2_tolt=$TOLT \
              output_write_vtk=false output_write_ps=false adaptive_write_mesh=false $EXTRA \
        --vary refinement_n_initial_steps=$LEVELS
    ;;
  adaptive)
    python3 $S --program tnse --exe $EXE --base $BASE --name adaptive \
        --set trab2_tolt=$TOLT output_write_vtk=false $EXTRA \
        --vary adaptive_tol1/adaptive_tol2=5e-2/2.5e-3,2.5e-2/1.25e-3,1.25e-2/6.25e-4
    ;;
  time)
    python3 $S --program tnse --exe $EXE --base $BASE --name time \
        --set output_write_vtk=false $EXTRA \
        --vary trab2_tolt=1e-3,1e-4,1e-5
    ;;
  plots)
    P=$R/scripts
    for study in uniform adaptive time; do
      if grep -q "\"run\": \"${study}_" $J 2>/dev/null; then
        python3 $P/layer_switch_plot.py "$J:${study}_*" --panels "dt,cells,L2(u),L2(p),error,eta" \
            --title "$NAME: $study" --out $OUT/${PREFIX}_${study}_time
        python3 $P/layer_switch_convergence.py "$J:${study}_*" --after 0.5 \
            --out $OUT/${PREFIX}_${study}_maxt | tee $OUT/${PREFIX}_${study}_maxt.txt
      fi
    done
    ;;
  *)
    echo "unknown part '$part' (uniform, adaptive, time, plots)"; exit 1 ;;
  esac
done
