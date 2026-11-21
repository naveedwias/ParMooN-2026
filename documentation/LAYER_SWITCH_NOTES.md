# Layer-switch examples 36 and 37: record of the work

This file records what was implemented for the time-dependent Navier-Stokes examples 36 and 37, what was changed in the solver to reduce the errors, and why. The commits are on the branch `adaptive-refactor`. The reference for all parameters is `ADAPTIVE_RUNS.md`. Status labels: **checked** (verified by a run or a comparison), **plausible** (supported by the runs, not proved), **open**.

## 1. How to run

Build and set the paths once (from the `build` folder next to `ParMooN-2026`):

```bash
cd build
make tnse2d_adaptive_spg
mkdir -p vtk
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
```

Recommended: through the study script, which keeps the table, the meshes, the vtk files and the figures of each run:

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_a.dat --name switch_a
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_b.dat --name switch_b
```

Overlapping variant (both layer systems present on (0.4, 0.6)):

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_a.dat --name overlap_a --set layer_switch_d=0.4 layer_switch_c=0.6
```

Short check before a long run (to t = 0.01, output every 0.005; takes a few minutes):

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_a.dat --name check_a --set time_end=0.01 adaptive_output_time=0.005
```

Directly, without the script (the outputs land in the current folder and in `vtk/`):

```bash
./parMooN_tnse2d_adaptive_spg_SEQUENTIAL $E/tnse2d_adaptive_spg_layer_switch_a.dat
./parMooN_tnse2d_adaptive_spg_SEQUENTIAL $E/tnse2d_adaptive_spg_layer_switch_b.dat
```

Switching off the new features, e.g. to reproduce the old behaviour: `--set adaptive_initial_mesh=false`, `--set trab2_first_step=semidiscrete`, `--set adaptive_write_mesh=false`, `--set adaptive_coarsening_min_fraction=0`.

Figure of the time step, the number of cells and the errors over time (from the two runs above):

```bash
python3 ../ParMooN-2026/scripts/layer_switch_plot.py \
    tnse_layer_switch_x0y0/tnse_layer_switch_x0y0_runs.jsonl:switch_a:"example 36" \
    tnse_layer_switch_x1y0/tnse_layer_switch_x1y0_runs.jsonl:switch_b:"example 37" \
    --out layer_switch_time_step
```

It writes `layer_switch_time_step.png` and `.pdf`. In the dt panel, the dips to dt/2 every 10 steps are the periodic averaging (the effective step of an averaged step is k/2), not rejections.

### Output of a script run

The script writes to `tnse_layer_switch_x0y0/` (example 36) or `tnse_layer_switch_x1y0/` (example 37). The files of a run are in `<prefix>_files/<run>/`.

| file | content |
|---|---|
| `<prefix>_runs.jsonl`, `_runs.csv` | parameters, the per-step table (time, dt, cells, DoFs, eta, error, effectivity, L2(u), H1(u), L2(div u), L2(p), ...), wall time |
| `<prefix>_files/<run>/vtk/<prefix>_mesh_t<time>.vtk` | the mesh at t = 0 and every 0.05: cell data `level`, `diameter`, `eta_K` (ParaView: Surface With Edges, colour by `level`) |
| `<prefix>_files/<run>/mesh_t<time>.ps`, `<prefix>_meshes.pdf` | the same meshes as PostScript and as one figure |
| `<prefix>_files/<run>/vtk/<prefix>_<i>.vtk` | velocity, pressure and time derivative at the same times |
| `<prefix>_<name>_tables.pdf`, `_flow.pdf`, `_summary.txt` | plots of the table and a summary |

## 2. The examples (commit `5935c75`)

File `include/Examples/TNSE_2D/TNSE_layer_switch.h`, examples 36 and 37 in `Example_TimeNSE2D.C`, input files `tnse2d_adaptive_spg_layer_switch_a.dat` / `_b.dat`.

- Two steady solutions u = curl(A(xi) A(eta)) with A(s) = s^2 (1 - exp(k (s - 1)))^2 and p = exp(xi + eta) - (e - 1)^2.
  - NSE1: layers at x = 1 and y = 1 (width 1/lambda = 0.020).
  - NSE2: layers 1.5 times thinner (width 0.013), scaled to the same peak velocity. Example 36: at x = 0 and y = 0. Example 37: at x = 1 and y = 0.
- u(t) = time1(t) NSE1 + time2(t) NSE2 with C^2 smooth switches: NSE1 is switched off on [0.3, 0.5], NSE2 is switched on on [0.5, 0.7].
- nu = 1e-3 in the input files (since 3.10; nu = 1e-4 before), domain (0, 1)^2, homogeneous Dirichlet data, f computed from the exact solution. The layer widths above are for nu = 1e-4; at nu = 1e-3 they are 0.063 and 0.042.
- Checked: the formulas against sympy at 84 points and times (difference 2e-14); div u = 0; u = 0 on the boundary.

## 3. What was changed, and why

### 3.1 Pressure error did not converge (commits `b905185`, `a9ed7c2`)

**Problem.** L2(p) did not decrease with h: about 240 for example 36 on every mesh, and 1.4 for example 23 while L2(u) dropped by a factor 60.

**Cause (checked numerically).** The initial velocity U^0 (L2 projection) is not consistent with the discrete constraint (the divergence row with PSPG). The trapezoidal rule is not L-stable: it carries the inconsistent part, which lies in stiff modes, from step to step with a factor of about -1. The pressure is the multiplier of the constraint and keeps an O(1) error. With fixed steps nothing removes it. With adaptive steps the first periodic averaging removes it, but each interpolation to a new mesh brings it back.

**Fix.**
- All `tnse2d_adaptive_spg_*.dat` files set `trab2_first_step: backward_euler`. The first step is a backward Euler step, which damps the stiff part.
- The layer-switch files also set `trab2_be_after_mesh_change: true`.
- A fixed-step run that starts the trapezoidal rule from an inconsistent U^0 prints a warning.

**Result (checked).**

| case | before | after |
|---|---|---|
| example 23, levels 4 / 5 / 6, L2(p) | 0.51 / 1.43 / 1.35 | 7.2e-2 / 1.7e-2 / 3.4e-3, O(h^2) |
| example 36, h = 1/32, 1/64, 1/128, t in [0.40, 0.42], L2(p) | about 240 on all three meshes | 0.72 / 0.20 / 0.025 |

The alternative `tnse_initial_value: consistent_projection` also works for example 23. It needs the exact solution, and for example 36 it is worse (L2(u) 2.8 / 1.0 / 0.13), so it was not chosen.

### 3.2 Mesh output for checking the adaptivity (commit `df0ef7d`)

- New parameter `adaptive_write_mesh` (default true): the mesh is written as a VTK file, with level, diameter and eta_K per cell, at the start and at every output time.
- The PostScript mesh is now also written at t = 0.
- The layer-switch files write every 0.05 (`adaptive_output_time: 0.05`), so the switch on [0.3, 0.7] is covered by 9 meshes.

### 3.3 Tolerances (commit `3c06cc9`)

With TOL1 = 5e-3 the mesh grew to 40675 cells by t = 0.01 (level 9 in the layers). This is far too many for a run to t = 1.

Short runs to t = 0.03 (TOL2 = TOL1/20), without the initial adaptation:

| TOL1 | cells | error | L2(u) | L2(p) |
|---|---|---|---|---|
| 2e-2 | 24800 | 1.85 | 1.33 | 3.70 |
| 5e-2 | 15700 | 1.89 | 1.33 | 3.71 |
| 1e-1 | 11200 | 1.94 | 1.33 | 3.71 |

The input files now use TOL1 = 5e-2 and TOL2 = 2.5e-3. The estimator was eta of about 16, so TOL1 of about eta/sqrt(cells) gives about 1.5e4 cells.

The tolerance hardly changed the error. This led to 3.4.

Note: the column `error` is the error in the SUPG/PSPG norm, not an L2 error. Example 36 has nu = 1e-4 and peak velocities of about 20, so its errors are not comparable with those of example 23 (Re = 100).

### 3.4 Initial mesh adaptation (commits `65c3f59`, `daa9fb6`)

**Why.** The runs started on 512 cells (h = 1/16), where the layers (width 0.02) are not resolved. The projected initial velocity already had L2(u) = 1.04 at the first step. Refining later did not remove this error: at nu = 1e-4 it is transported and decays only slowly (plausible), and L2(u) even grew to 1.33. The error was therefore set by the start, not by the tolerance.

**What it does.** New parameters, used only with `adaptive_mesh: true`:

| parameter | default | meaning |
|---|---|---|
| `adaptive_initial_mesh` | `true` | adapt the mesh to u_0 before the first step; `false` switches it off |
| `adaptive_initial_max_steps` | `10` | maximal number of adaptation steps |

Each adaptation step:
1. project u_0 on the current mesh;
2. take a trial backward Euler step of the first step size (Navier-Stokes only; the step is discarded);
3. estimate;
4. refine (eta_K > TOL1, up to the maximal level) and coarsen (eta_K < TOL2), with the same rule as the time loop;
5. rebuild the system.

The loop stops when the number of cells changes by less than 0.5 %, or after the maximal number of steps. The run then starts from u_0 projected on the final mesh.

**Why the trial step.** The projected u_0 is not consistent with the discrete constraint, as in 3.1. Estimated at u_0 itself, its stiff part made eta_K large in the whole domain. Almost nothing could be coarsened, the mesh grew to 21282 cells, and the first time step then derefined 1826 parents at once. Computing P^0 from the semidiscrete problem made it worse (eta 1409, 54292 cells), so the missing initial pressure was not the cause (checked). One backward Euler step damps the stiff part, and the estimate then describes the solution.

**Result, example 36, first time step (checked):**

| initial mesh | adaptation steps | cells | eta | error | L2(u) | L2(p) |
|---|---|---|---|---|---|---|
| none (512 cells) | - | 512 | 361 | 12.3 | 1.04 | 8.57 |
| adapted, estimate at u_0 (superseded) | 10 (limit) | 21282 | 4.8 | 0.045 | 4.7e-4 | 1.1e-2 |
| adapted, estimate after the trial step | 6 | 9766 | 1.8 | 0.062 | 1.3e-3 | 3.7e-2 |
| the same with the derefinement fix of 3.6 (current) | 6 | 12450 | - | 0.061 | 1.3e-3 | 3.2e-2 |

At t = 0.001 (current version): L2(u) = 1.5e-3 and L2(p) = 3.8e-3. The first time steps change the mesh by only a few dozen parents.

Convection-diffusion Example 3 (no constraint, u_0 estimated directly): eta goes from 0.28 (1250 cells) to 0.031 (9355 cells) in 6 steps.

### 3.6 Time adaptivity (commit `d09c06a`)

**Problem (checked).** The step size control did not work in examples 36 and 37. Example 36, t in [0, 0.02]: all 200 steps were backward Euler steps with dt = 1e-4 fixed, and the mesh changed in every step.

**Cause (checked).**
- The time loop remeshed whenever a single cell was marked for coarsening.
- After every mesh change the integrator took a backward Euler step (`trab2_be_after_mesh_change: true`, set in 3.1). A backward Euler step is accepted without an error estimate and keeps the step size, and it also restarts the averaging counter.
- Without that setting the control did run, but the interpolation to a new mesh in every step brought back the inconsistent stiff part. dt then collapsed to about 1.5e-5 and the pressure was polluted again (L2(p) up to 5e-2).
- Most of these "mesh changes" did not change the mesh at all. `RefineAndCoarsen` derefined parents that the closure refined again at once: a triangle with two or more regularly refined neighbours is refined regularly by the conforming closure, and with hanging nodes `Gen1RegGrid` does the same next to finer cells. The log said "44 parents derefined" while the mesh kept 9374 cells.

**Fix (option A).**
- `RefineAndCoarsen` derefines a parent only if the closure will not undo it. New function `TDomain::N_DerefinableParents` counts these parents without changing the grid.
- The time loop remeshes only if a cell is marked for refinement, or if the children of the derefinable parents make up at least `adaptive_coarsening_min_fraction` of the cells (new parameter, default 0.05; 0 gives the old behaviour).
- The backward Euler step and the averaging restart follow a remesh only if the mesh really changed.

**Result, example 36, t in [0, 0.05] (checked):**

| | before | now (input files: backward Euler after a remesh) | now, without it |
|---|---|---|---|
| steps | 200 by t = 0.02 | 85 | 99 |
| backward Euler steps | 200 | 2 | 1 |
| rejected steps | 0 | 0 | 3 |
| mesh changes | 200 | 1 | 1 |
| dt | 1e-4 fixed | 1e-4 growing to about 1.3e-3 | the same |
| L2(p) at t = 0.05 | - | 8.4e-3 | 8.4e-3 |

dt grows only to about 1e-3, not to `trab2_dt_max` = 0.01, although the exact solution is steady until t = 0.3. The step is limited by TOLT = 1e-5 (estimates 2e-6 to 6e-6 per step; plausible). L2(u) grows slowly, from 1.3e-3 to 3.5e-3 by t = 0.05 (open: probably the drift of the discrete solution towards its own steady state).

### 3.7 Time tolerance (TOLT)

Example 36 over [0, 0.3], current settings, three values of `trab2_tolt` (checked):

| TOLT | steps | median dt | wall time | L2(u)(0.3) | L2(p)(0.3) | error(0.3) |
|---|---|---|---|---|---|---|
| 1e-5 | 199 | 1.5e-3 | 32 min | 0.0357 | 0.0145 | 0.0706 |
| 1e-4 | 87 | 3.5e-3 | 21 min | 0.0352 | 0.0146 | 0.0704 |
| 1e-3 | 43 | 1.0e-2 (dt_max) | 18 min | 0.0339 | 0.0150 | 0.0699 |

All three runs: 2 backward Euler steps, no rejected step, 1 remesh. The errors do not depend on TOLT, so they are spatial errors. The input files now use TOLT = 1e-3. On the switch interval [0.3, 0.7] the estimate should reduce the step where the solution changes (plausible; to be checked in the full run).

Growth of L2(u) on [0, 0.3] (open). The exact solution is steady and the mesh is fixed, yet L2(u) grows from 1.3e-3 to 3.5e-2, about 1.5 times per 0.05 (rate about 8), slowing slightly at the end. The growth is the same for all TOLT. The error in the SPG norm grows only from 0.061 to 0.070, and nothing is refined, so the growing part is smooth and large-scale. Plausible explanation: the steady flow is unstable at Re of about 2e5 (peak velocity 20, nu = 1e-4), and discretization errors are amplified by the flow. To settle it: a run with TOL1 halved (an instability keeps the growth rate) and a run with nu = 1e-3 (the growth should weaken). If it is an instability, the long-time errors of examples 36/37 measure the instability rather than the method.

### 3.8 Instability of the steady flow at nu = 1e-4

Example 36 on [0, 0.3] (TOLT = 1e-3) (checked):

| run | cells | L2(u)(0) | L2(u)(0.3) | growth rate on [0.05, 0.15] | on [0.15, 0.3] |
|---|---|---|---|---|---|
| nu = 1e-4, TOL1 = 5e-2 | 12450 | 1.3e-3 | 0.034 | 12.2 | 7.1 |
| nu = 1e-4, TOL1 = 2.5e-2 | 18750 | 7.1e-4 | 0.020 | 13.6 | 6.7 |
| nu = 1e-3, TOL1 = 5e-2 | 3770 | 4.8e-4 | 9.1e-4 | 0.38 | 0.36 |

The finer mesh keeps the growth rate and only lowers the starting level. At nu = 1e-3 the growth is essentially gone. So the steady layer flow amplifies small perturbations at nu = 1e-4 (Re about 2e5 with the peak velocity 20); this is not a discretization effect (supported by these runs, not proved). The decreasing rate (13 -> 7) suggests transient (non-normal) amplification or saturation rather than a single unstable mode (plausible).

Consequence: at nu = 1e-4 the long-time errors of examples 36/37 mostly measure this amplification, not the method. nu = 1e-3 (`--set reynolds_number=1e3`, layer widths 0.063 and 0.042) gives errors that judge the method, and the run to t = 0.3 takes about 2 minutes instead of 18 (open: which nu to use).

### 3.9 Switch interval and coarsening

Example 36 over [0.25, 0.75] with the settings of 3.7 (checked) showed two problems:

1. **No coarsening (regression of 3.6, fixed).** The mesh stayed at 12450 cells up to t = 0.51 although eta dropped to 0.02. The rule of 3.6 counted refined neighbours that were derefined in the same pass, so the inside of a refined region could never be coarsened. Now only neighbours that stay refined count, and the check is repeated until the set of derefinable parents does not change.

   Rerun over [0.25, 0.52]: 9770 -> 2006 cells by t = 0.5, 50-430 parents per remesh, dt at dt_max = 0.01 while the layers vanish. L2(u) at t = 0.5 is now 0.091 (0.020 without the coarsening): the estimator (eta 0.17, SPG-norm error 0.067) does not see the growing smooth part of the error, which is amplified by the instability of 3.8 (plausible). The 2D adaptive tests pass.

2. **Frozen step while the new layers form (open).** From t = 0.51 on the loop refines in every step (30-170 cells). Every step is then a backward Euler step that keeps dt = 9.2e-3 without an error estimate. Refinement goes one level per step, the mesh lags behind the forming layers, and the errors grow to L2(u) = 0.31, L2(p) = 0.30 at t = 0.75. Candidate fixes:
   - (a) estimate the error of the backward Euler step against a forward Euler predictor (risk: right after a remesh the inconsistent part dominates the estimate);
   - (b) no backward Euler step after remeshes that only refine;
   - (c) remesh at most every m steps during refinement phases.

### 3.10 nu = 1e-3 and no backward Euler step after pure refinement

Following 3.8 and 3.9, the input files of examples 36 and 37 now use nu = 1e-3 (`reynolds_number: 1.0e3`): layer widths 0.063 and 0.042, peak velocities about 4.1 (NSE1) and 4.9 (A_2 NSE2). nu = 1e-4 stays available as a stress test (`--set reynolds_number=1e4`).

New parameter `adaptive_be_only_after_coarsening` (default true): with `trab2_be_after_mesh_change`, the backward Euler step follows only mesh changes that coarsen. After a pure refinement the coarse function is represented exactly on the new mesh, and the trapezoidal rule continues with its step size control (only the averaging is restarted).

Example 36, nu = 1e-3, t in [0.45, 0.75] (new layers form on [0.5, 0.7]) (checked):

| | backward Euler only after coarsening (new) | after every mesh change (old) |
|---|---|---|
| backward Euler steps | 16 of 46 | 35 of 44 |
| max L2(u) | 0.012 | 0.035 |
| max L2(p) | 0.013 | 0.056 |
| max error | 0.060 | 0.072 |

Full runs with the committed input files, through the study script (checked):

| | steps | wall time | cells (min-max) | max L2(u) | max L2(p) | max error | L2(u)(1) |
|---|---|---|---|---|---|---|---|
| example 36 | 116 | 3 min | 826-3610 | 0.040 | 0.027 | 0.070 | 0.040 |
| example 37 | 116 | 3 min | 704-3610 | 0.037 | 0.026 | 0.061 | 0.037 |

Example 36 in detail: dt = 0.01 (dt_max) on the steady interval, down to 8.5e-3 and 5e-3 when the new layers form; 3610 cells on [0, 0.3], 826 at t = 0.5, 1980 from t = 0.7 on. The meshes at t = 0, 0.05, ..., 1 are in `<prefix>_files/switch_a/vtk/`.

Open: L2(u) rises from 9e-4 (t = 0.3) to 0.032 (t = 0.5) while the old layers vanish and the mesh is coarsened, and it stays at 0.035-0.040 afterwards. The estimator does not see this part (eta 0.04 at t = 0.5). Plausible: the interpolation to the coarsened mesh leaves a smooth velocity error that decays only on the viscous time scale. To check: a smaller TOL2 (less coarsening) and its effect on L2(u) after t = 0.5.

### 3.11 Three checks of the error after the switch

Example 36, nu = 1e-3, full runs over [0, 1], each compared with the baseline (TOL1 = 5e-2, TOL2 = 2.5e-3, TOLT = 1e-3) (checked):

| run | steps | cells | L2(u)(0.3) | L2(u)(0.5) | L2(u)(1) | max L2(p) | L2(p)(1) | error(1) |
|---|---|---|---|---|---|---|---|---|
| baseline | 116 | 742-3610 | 9.2e-4 | 3.2e-2 | 4.0e-2 | 2.7e-2 | 2.0e-2 | 6.9e-2 |
| TOLT = 1e-4 | 218 | 657-3610 | 9.5e-4 | 1.1e-2 | 1.8e-2 | 2.6e-2 | 1.0e-2 | 5.9e-2 |
| TOL2 = 2.5e-4 | 115 | 1593-3772 | 9.1e-4 | 2.5e-2 | 3.9e-2 | 4.0e-2 | 1.9e-2 | 6.9e-2 |
| TOL1 = 2.5e-2 | 113 | 1023-5459 | 5.0e-4 | 3.5e-2 | 3.0e-2 | 2.0e-2 | 9.5e-3 | 4.4e-2 |
| TOL1 = 1.25e-2 | 114 | 1302-7466 | 3.2e-4 | 3.6e-2 | 4.1e-2 | 4.2e-2 | 1.9e-2 | 5.1e-2 |

Conclusions:
1. **Time error.** After the switch the time error of TOLT = 1e-3 dominates. TOLT = 1e-4 cuts L2(u) about 3 times at t = 0.5 and halves L2(p) at t = 1; dt then drops to 5e-4 during the switch. The TOLT study of 3.7 covered only the steady interval. The input files now use TOLT = 1e-4.
2. **Coarsening.** Less coarsening removes only the first jump at t of about 0.35. The rise at 0.45-0.5 and the level afterwards are unchanged, so coarsening is not the main cause (this corrects the guess in 3.10).
3. **Spatial convergence.** Before the switch the error decreases with TOL1 (L2(u)(0.3) 9.2e-4, 5.0e-4, 3.2e-4 at about 12k, 18k, 25k mean DoFs). After the switch it does not, because the time error (TOLT = 1e-3 in these runs) dominates.

Open: even with TOLT = 1e-4, L2(u) after the switch (1.1e-2 to 1.8e-2) is 10-20 times the level before. Whether TOLT = 1e-5 lowers it further, and whether the error then converges with TOL1, is what `scripts/run_layer_switch_study.sh` (parts `time` and `adaptive`) measures.

Figures: `scripts/layer_switch_plot.py` (quantities over time, several runs) and `scripts/layer_switch_convergence.py` (maximum over time against the mean DoFs, with observed orders).

### 3.12 TOLT = 1e-5 and the TOL1 series at TOLT = 1e-4

Runs of `scripts/run_layer_switch_study.sh` (parts `time` and `adaptive`), example 36, nu = 1e-3 (checked):

| run | steps | mean DoFs | L2(u)(0.3) | L2(u)(0.5) | L2(u)(1) | max L2(p) | L2(p)(1) | error(1) |
|---|---|---|---|---|---|---|---|---|
| TOLT 1e-3, TOL1 5e-2 | 116 | 12115 | 9.2e-4 | 3.2e-2 | 4.0e-2 | 2.7e-2 | 2.0e-2 | 6.9e-2 |
| TOLT 1e-4, TOL1 5e-2 | 218 | 10639 | 9.5e-4 | 1.1e-2 | 1.8e-2 | 2.6e-2 | 1.0e-2 | 5.9e-2 |
| TOLT 1e-5, TOL1 5e-2 | 892 | 9648 | 9.2e-4 | 2.2e-3 | 6.1e-3 | 7.9e-2 | 2.4e-3 | 4.7e-2 |
| TOLT 1e-4, TOL1 2.5e-2 | 190 | 15736 | 5.0e-4 | 1.6e-2 | 2.5e-2 | 1.8e-2 | 1.4e-2 | 4.3e-2 |
| TOLT 1e-4, TOL1 1.25e-2 | 173 | 22153 | 3.0e-4 | 1.3e-2 | 1.1e-2 | 9.6e-3 | 2.7e-3 | 2.4e-2 |

Conclusions:
1. **The time error still dominates L2(u) at TOLT = 1e-4.** TOLT = 1e-5 lowers L2(u) after the switch 3-5 times more (6.1e-3 at t = 1), at 4 times the steps.
2. **The error in the SUPG/PSPG norm converges with TOL1** (5.9e-2, 4.3e-2, 2.4e-2 at t = 1, at TOLT = 1e-4). L2(u) and L2(p) do not decrease monotonically, because their time part is of the same size as the spatial part.
3. **The step control oscillates at TOLT = 1e-5** during the switch: dt jumps between about 1e-4 and 4e-6, L2(p) is noisy, and it spikes to 7.9e-2 at t of about 0.617. There the mesh is refined in several consecutive tiny steps (2159 -> 3048 cells within 5e-5 time units). Plausible cause: after a pure refinement (no backward Euler step, 3.10) the trapezoidal rule carries the inconsistent part of the new mesh, and with a small TOLT the estimate reacts to it (open).

For spatial convergence studies the time error has to be well below the spatial error: TOLT = 1e-5 (or TOLT decreased together with TOL1), at the cost of more steps. The oscillation of 3 has to be understood first.

### 3.13 Cause of the oscillation of the step control (fixed)

The oscillation of 3.12 comes from the mesh changes (checked). In the TOLT = 1e-5 run, 27 of the 35 steps in which dt dropped by more than half came right after a mesh change, and most of the other 8 are the periodic averaging. Example 36, t in [0.5, 0.65], TOLT = 1e-5:

| run | steps | rejected | BE steps | remeshes | min dt | median dt | max L2(p) | L2(u)(0.65) | mean cells |
|---|---|---|---|---|---|---|---|---|---|
| A: no BE after refinements (3.10) | 340 | 100 | 30 | 97 | 1.3e-5 | 3.0e-4 | 4.1e-2 | 5.0e-3 | 1542 |
| B: BE after every mesh change | 216 | 20 | 91 | 90 | 2.5e-4 | 6.8e-4 | 8.0e-3 | 5.8e-3 | 1333 |
| C: fixed uniform mesh (level 6) | 53 | 6 | 1 | 0 | 1.2e-4 | 2.9e-3 | 4.2e-3 | 7.4e-3 | 8192 |

Without mesh changes (C) there is no oscillation, so the integrator and its error control are fine. The reasoning behind option (b) in 3.10 was wrong: after a pure refinement the coarse function is represented exactly, but the discrete constraint of the new mesh (more pressure test functions, new delta_K) is not satisfied. The trapezoidal rule carries the inconsistent part with factor -1, the AB2 estimate sees an error that does not decrease with dt, and the step is cut and rejected again and again (plausible mechanism, consistent with all runs). Adaptive meshes with about 1300-1500 cells are as accurate as the uniform mesh with 8192 cells.

Full runs over [0, 1] (checked):

| | steps | dt drops > 2x | min dt | L2(u)(1) | max L2(p) (t > 0.05) | L2(p)(1) |
|---|---|---|---|---|---|---|
| A, TOLT 1e-4 | 218 | 6 | 1e-4 | 1.8e-2 | 1.2e-2 | 1.0e-2 |
| B, TOLT 1e-4 | 184 | 2 | 1e-4 | 1.3e-2 | 1.5e-2 | 7.6e-3 |
| A, TOLT 1e-5 | 892 | 40 | 3.8e-6 | 6.1e-3 | 7.9e-2 | 2.4e-3 |
| B, TOLT 1e-5 | 621 | 4 | 1e-4 | 8.2e-3 | 1.3e-2 | 4.3e-3 |

B needs 15-30 % fewer steps and has neither the oscillation nor the pressure spike; at TOLT = 1e-5 its L2(u) at t = 1 is somewhat larger (first-order steps). `adaptive_be_only_after_coarsening` now defaults to false (B). The earlier preference for (b) came from TOLT = 1e-3, where the backward Euler steps froze a large dt.

### 3.14 Check of the error estimator

**Code.** `NSE_SPG_ErrorEstimator.C` was read against the formulas in its header (checked). The cell residual is r_K = f - U_t + nu Lap u_h - (u_h.grad)u_h - grad p_h, with U_t the discrete time derivative of the integrator, evaluated at the current time. The facet residual is the jump of (-nu grad u_h + p_h I) n; on the red-green meshes the leaf neighbour of an edge is searched correctly. eta_K collects eta_res, eta_delta, eta_mu, the cell's share of the global div term and half of its interior facet terms. The constants of the transient bound (320, 160, 640, 320/|U|^2) could not be checked against the manuscript here (no copy of the transient bound available; open).

**What the estimator measures.** U_t comes from the time integrator, so r_K is the residual of the semidiscrete equation at the current time. The estimator measures the spatial error in the SPG norm at that time; time error, and error carried over from earlier times, are not seen by construction. That is why eta did not react to the time error after the switch (3.11-3.12). It is the intended split (TOLT controls the time error), not a bug.

**Numbers** (uniform meshes, TOLT = 1e-5, nu = 1e-3). The estimator now also stores the exact local error error_K in the same norm, and the mesh files write it next to eta_K.

| run | h | eta | error | eta/error |
|---|---|---|---|---|
| steady, t = 0.02 | 1/8 | 48.2 | 1.61 | 29.9 |
| | 1/16 | 20.6 | 0.737 | 27.9 |
| | 1/32 | 5.83 | 0.227 | 25.7 |
| | 1/64 | 1.66 | 0.0655 | 25.4 |
| switch, t = 0.6 | 1/32 | 5.10 | 0.213 | 24.0 |
| | 1/64 | 1.58 | 0.0668 | 23.6 |

| run | cells | Spearman(eta_K, error_K) | overlap of the top 10 % cells | error^2 in the top 10 % eta_K cells (best possible) | eta_K/error_K, 10-90 % |
|---|---|---|---|---|---|
| steady, h = 1/32 | 2048 | 1.00 | 0.95 | 0.98 (0.98) | 18-25 |
| steady, h = 1/64 | 8192 | 1.00 | 0.96 | 0.98 (0.99) | 19-27 |
| switch, h = 1/64 | 8192 | 1.00 | 0.97 | 0.99 (0.99) | 18-24 |

The estimator is reliable and efficient here: eta and error decrease at the same rate (about a factor 3.5 per level), the effectivity is bounded and settles at about 24-25, also during the switch. The local indicators rank the cells almost exactly like the true local errors, so marking by eta_K selects practically the cells that marking by the error would select (checked). The size of the effectivity comes from the constants of the bound (C = 1 with 320, 640, ...); it is a fixed scale, which only shifts the meaning of TOL1.

### 3.15 Remeshing frequency and the pressure spikes

New parameters (time loop):
- `adaptive_remesh_min_steps` (default 1): change the mesh at most every m time steps.
- `adaptive_remesh_force_factor` (default 0, off): remesh earlier if some eta_K > F * TOL1.

Example 36, t in [0.3, 0.75], TOLT = 1e-5 (checked):

| m | steps | remeshes | BE steps | rejected | sum of abs. changes of log L2(p) | L2(u)(0.75) | L2(p)(0.75) |
|---|---|---|---|---|---|---|---|
| 1 | 556 | 151 | 152 | 61 | 129 | 6.0e-3 | 3.1e-3 |
| 5 | 704 | 93 | 94 | 111 | 179 | 6.5e-3 | 2.9e-3 |
| 10 | 585 | 48 | 49 | 95 | 128 | 5.8e-3 | 2.7e-3 |
| 5, F = 4 | 720 | 99 | 100 | 108 | 172 | 5.7e-3 | 3.0e-3 |

Less frequent remeshing does not help: each remesh then changes more cells, and more steps are rejected afterwards; the pressure noise does not decrease. The default stays m = 1 (the parameters are kept for other problems).

The pressure noise has a clear structure (checked): spikes appear in the backward Euler step right after a mesh change (e.g. L2(p) 2.8e-3 -> 1.4e-2 -> 3.4e-3) and decay within one or two steps; between remeshes the trapezoidal steps vary by about 10 %. Mechanism (plausible): the transferred velocity violates the discrete constraint of the new mesh; the backward Euler step removes that part, and the pressure, the multiplier of the constraint, takes an error of size (inconsistent part)/k in that step. The end values are not affected.

Recommended next step (not implemented): after each transfer, project the velocity onto the discretely divergence-free space of the new mesh (one linear solve with the saddle point matrix). That should remove both the pressure spike and the need for the backward Euler step after a mesh change.

### Running the spatial study

From the build folder:

```bash
bash ../ParMooN-2026/scripts/run_layer_switch_study.sh
```

It runs, for example 36 (`EX=b` for 37):
- `uniform`: uniform meshes, levels 4-7 (`LEVELS=...`), no mesh adaptation;
- `adaptive`: TOL1 = 5e-2, 2.5e-2, 1.25e-2 with TOL2 = TOL1/20;
- `time`: TOLT = 1e-3, 1e-4, 1e-5 at TOL1 = 5e-2;
- `plots`: for each study `<prefix>_<study>_time.png` (dt, cells, errors, eta over time) and `<prefix>_<study>_maxt.png/.txt` (maximum over time against the DoFs).

Parts can be run on their own, e.g. `bash ../ParMooN-2026/scripts/run_layer_switch_study.sh uniform plots`. The uniform and adaptive studies use `TOLT=1e-4` (environment variable). Finished runs are skipped. Level 7 (32768 cells) to t = 1 takes long; start with `LEVELS=4,5,6`.

### 3.5 Bug fixed on the way (commit `65c3f59`)

`TDomain::RefineAndCoarsen` read `collection->GetCell(0)` after the derefinement. When the parent of cell 0 was derefined, it read a deleted cell and crashed. This was a pre-existing bug, found with a debug build at the first large coarsening (1826 parents). Fixed: the shape is read before the derefinement. The 2D adaptive tests pass. The 3D test was not built.

## 4. Final parameter settings

| parameter | value | reason |
|---|---|---|
| `reynolds_number` | 1e3 (nu = 1e-3) | at nu = 1e-4 the steady flow amplifies perturbations (3.8) |
| `adaptive_tol1`, `adaptive_tol2` | 2.5e-2, 1.25e-3 | spatial error about 30 % below TOL1 = 5e-2 for about 50 % more cells (3.12) |
| `trab2_tolt` | 1e-5 | the time error of 1e-4 still dominates L2(u) after the switch (3.12) |
| `trab2_dt_max` | 0.01 | unchanged |
| `trab2_first_step`, `trab2_be_after_mesh_change` | backward_euler, true | pressure error (3.1) |
| `adaptive_be_only_after_coarsening` | false (default) | otherwise the step control oscillates (3.13) |
| `adaptive_remesh_min_steps`, `adaptive_remesh_force_factor` | 1, 0 (defaults) | less frequent remeshing did not help (3.15) |
| `adaptive_coarsening_min_fraction` | 0.05 (default) | no remeshing for a handful of coarsened parents (3.6) |
| `adaptive_initial_mesh` | true (default) | resolved start (3.4) |

Confirmation runs with the committed input files, through the study script, to t = 1 (checked):

| | wall time | steps | cells | dt drops > 2x | L2(u)(0.3) | L2(u)(0.5) | L2(u)(1) | max L2(u) | max L2(p) (t > 0.05) | L2(p)(1) | max error | median eta/error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| example 36 | 12 min | 498 | 535-5459 | 3 | 5.0e-4 | 5.8e-4 | 3.9e-3 | 4.5e-3 | 4.5e-3 | 1.8e-3 | 3.5e-2 | 17 |
| example 37 | 12 min | 501 | 535-5459 | 5 | 5.0e-4 | 5.8e-4 | 3.7e-3 | 3.9e-3 | 4.5e-3 | 1.9e-3 | 3.4e-2 | 17 |
| first run of 3.10 (example 36) | 3 min | 116 | 742-3610 | - | 9.2e-4 | 3.2e-2 | 4.0e-2 | 4.0e-2 | 2.7e-2 | 2.0e-2 | 7.0e-2 | - |

Compared with the first run, the maximum L2(u) is 9 times smaller and the maximum L2(p) 6 times smaller, at 4 times the run time. After the switch L2(u) rises from 6e-4 to about 4e-3 (the new layers are thinner, and the time error at TOLT = 1e-5 is not zero; 3.12).

## 5. Open points

- **Pressure spikes after a mesh change** (3.15): the first (backward Euler) step after each remesh carries a pressure error of about (inconsistent part)/k, visible as noise in L2(p) while the mesh is coarsened (t of about 0.38-0.5). Recommended: project the transferred velocity onto the discretely divergence-free space of the new mesh; this should also make the backward Euler step after a mesh change unnecessary.
- **Constants of the transient estimator** (3.14): 320, 160, 640, 320/|U|^2 are taken from the code; they were not checked against the manuscript. The effectivity is bounded (about 17-25 in these runs) but large, which comes from these constants.
- **Time error and L2 error are not estimated** (3.14): the estimator measures the spatial error in the SPG norm; the time error is controlled only through TOLT, and errors carried over from earlier times are not seen.
- **Spatial convergence study** (section "Running the spatial study"): uniform levels 4-7 and the TOL1 series with the final TOLT = 1e-5 have not been run here; `scripts/run_layer_switch_study.sh` does it (set `TOLT=1e-5`).
- **nu = 1e-4** (3.8): unstable steady flow; usable only as a stress test.
- **Manuscript.** The start of the time stepping (backward Euler first step), the backward Euler step after mesh changes and the initial mesh adaptation differ from the algorithm in the manuscript and should be stated there.
- **Changed results.** Earlier results from the `tnse2d_adaptive_spg_*.dat` files change with the new start step, mainly the pressure columns and fixed-step example 23; results of all adaptive programs change with the derefinement fix of 3.6/3.9.
