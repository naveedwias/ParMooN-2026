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

Switching off the new features, e.g. to reproduce the old behaviour: `--set adaptive_initial_mesh=false`, `--set trab2_first_step=semidiscrete`, `--set adaptive_write_mesh=false`.

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
- nu = 1e-4, domain (0, 1)^2, homogeneous Dirichlet data, f computed from the exact solution.
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
| adapted, estimate after the trial step (current) | 6 | 9766 | 1.8 | 0.062 | 1.3e-3 | 3.7e-2 |

At t = 0.001 (current version): L2(u) = 1.5e-3 and L2(p) = 3.8e-3. The first time steps change the mesh by only a few dozen parents.

Convection-diffusion Example 3 (no constraint, u_0 estimated directly): eta goes from 0.28 (1250 cells) to 0.031 (9355 cells) in 6 steps.

### 3.5 Bug fixed on the way (commit `65c3f59`)

`TDomain::RefineAndCoarsen` read `collection->GetCell(0)` after the derefinement. When the parent of cell 0 was derefined, it read a deleted cell and crashed. This was a pre-existing bug, found with a debug build at the first large coarsening (1826 parents). Fixed: the shape is read before the derefinement. The 2D adaptive tests pass. The 3D test was not built.

## 4. Open points

- **Full runs.** Examples 36 and 37 have not been run to t = 1 with the final settings. Check in the meshes that the refinement follows the layers from x = 1, y = 1 to the new walls, and how the cell count behaves on [0.3, 0.7].
- **Accuracy target.** TOL1 = 5e-2 was chosen from the cell counts. For a smaller error, lower TOL1 and compare.
- **Effectivity.** The effectivity index (eta/error) is about 30 at the start of example 36 and still about 27 on fixed meshes. It is not mesh-independent here.
- **Manuscript.** The start of the time stepping (backward Euler first step) and the initial mesh adaptation differ from the algorithm in the manuscript and should be stated there.
- **Changed results.** Earlier results from the `tnse2d_adaptive_spg_*.dat` files change with the new start step, mainly the pressure columns and fixed-step example 23.
