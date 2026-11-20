# Running the adaptive examples (branch `adaptive-refactor`)

All commands are run from the build directory, which sits next to the source tree (`build/` and `ParMooN-2026/` in the same parent directory). Everything below refers to commit `b986906` or later on `adaptive-refactor`.

After `git pull`, run `cmake .` in the build directory whenever files were added or moved (the last such commits: `bb074ff` moved the example headers, `b986906` added examples 29-31), then rebuild the programs you use.

## Where the code and the examples are

| what | where |
|---|---|
| time-dependent NSE examples 23 (Example 1, smooth), 32 (Example 2, translating vortex), 36, 37 (boundary layers that move to other walls), 28 (u in P2, p in P1), 30, 31 (exact in space and time), 34 (as 30 with an outflow boundary) and the BDF2-IMEX examples 25 (smooth), 26 (lake at rest), 27 (shear layer) | headers in `include/Examples/TNSE_2D/`, registered in `Example_TimeNSE2D` (`src/Examples/Example_TimeNSE2D.C`) |
| stationary examples 21 (AAJ27 Ex. 4.1), 22 (Ex. 4.2), 29 (u in P2, p in P1) and 33 (as 29 with a Neumann boundary) | `include/Examples/NSE_2D/`, registered in `Example_NSE2D` |
| convection-diffusion examples 11, 14, 15, 16 (FGJN14 Ex. 2-5) | `include/Examples/TCD_2D/`, `Example_TimeCD2D` |
| systems | `System/TimeNavierStokesSPG`, `System/TimeConvectionDiffusionTR`, `System/AdaptiveTimeLoop`, estimator `System/NSE_SPG_ErrorEstimator` (stationary and, with `set_transient`, transient) |
| time integrator, mesh transfer | `TimeDiscRoutines/TrapezoidalAB2`, `FE/SolutionTransfer` |
| programs | `tnse2d_adaptive_spg`, `tcd2d_adaptive_tr`, `tnse2d_bdf2imex_cip`, `oseen2d_spg_adaptive` |
| old monolithic programs | `tnse2d_spg_adaptive`, `tcd2d_supg_adaptive`: deprecated, they still contain defects fixed in the new programs and print a warning at start |

The example numbers were kept when the examples moved from `Example_NSE2D` to `Example_TimeNSE2D`. The former Example 2 (example 24, AAJ27 layers converted to Navier-Stokes) has been removed and replaced by example 32; an input with `example: 24` stops with an error. `TimeNavierStokesSPG` and the BDF2 program take an `Example_TimeNSE2D`; the initial velocity comes from its initial condition.

## Build

```bash
cd build
cmake ../ParMooN-2026          # only needed once, or after new files were added
make tcd2d_adaptive_tr tnse2d_adaptive_spg oseen2d_spg_adaptive tnse2d_bdf2imex_cip
mkdir -p vtk                   # ParMooN does not create the vtk output directory
E=../ParMooN-2026/data/input_files
S=../ParMooN-2026/scripts/transient_study.py
```

## Convection-diffusion: adaptive SUPG (de Frutos, Garcia-Archilla, John, Novo 2014)

```bash
./parMooN_tcd2d_adaptive_tr_SEQUENTIAL $E/tcd2d_adaptive_tr_ex2.dat  # Ex. 2: adaptive time step, fixed mesh
./parMooN_tcd2d_adaptive_tr_SEQUENTIAL $E/tcd2d_adaptive_tr_ex3.dat  # Ex. 3: layers, mesh + time adaptive, T = 1.2
./parMooN_tcd2d_adaptive_tr_SEQUENTIAL $E/tcd2d_adaptive_tr_ex4.dat  # Ex. 4: rotating bodies, one rotation (T = 2 pi)
./parMooN_tcd2d_adaptive_tr_SEQUENTIAL $E/tcd2d_adaptive_tr_ex5.dat  # Ex. 5: rotating front, T = 1.8
```

These inputs use `trab2_averaging: periodic`, the averaging of the TR-AB2 integrator as in the paper.

## Time-dependent Navier-Stokes: adaptive SUPG/PSPG/grad-div

```bash
./parMooN_tnse2d_adaptive_spg_SEQUENTIAL $E/tnse2d_adaptive_spg_ex1.dat  # Ex. 1: smooth, uniform mesh, dt = 1e-3 (manuscript)
./parMooN_tnse2d_adaptive_spg_SEQUENTIAL $E/tnse2d_adaptive_spg_ex2.dat  # Ex. 2: translating vortex, mesh + time adaptive
```

These inputs follow the manuscript: periodic averaging, U^0 = L2 projection of u_0, and the lower bound dt >= alpha min_K delta_K in the step-size control. All three are described below.

## Output of a run

| file | content |
|---|---|
| `<input name>_table.txt` | one line per accepted time step: time, dt, cells, DoFs, eta, error (SUPG/SPG norm), effectivity, cells marked for refinement and coarsening, L2(u), H1(u) and, for Navier-Stokes, L2(div u), L2(p); written and flushed after every step, so it can be watched during a long run and survives an interrupted one |
| `<input name>.out` | ParMooN log |
| `mesh_t<time>.ps` | mesh at every `adaptive_output_time` (CD Examples 3-5, NSE Example 2; needs `output_write_ps: true`) |
| `vtk/<basename><index>.vtk` | solution and discrete time derivative (and pressure) at the same times |

## The same runs through the script, with figures

```bash
python3 $S --program tcd  --base $E/tcd2d_adaptive_tr_ex4.dat   --name fgjn_ex4 --outdir adaptive
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_ex2.dat --name tnse_ex2 --outdir adaptive
```

This writes (with `<prefix>` = `<problem>_<example>`, `scripts/naming.py`):

- `adaptive/<prefix>_files/<run>/<prefix>_meshes.pdf`, all meshes with their cell counts;
- `adaptive/<prefix>_<name>_tables.pdf`, with effectivity, eta and error, number of cells and dt over time;
- `adaptive/<prefix>_<name>_summary.txt`.

### Where the scripts keep the runs

`transient_study.py` and `spg_study.py` collect the runs per example folder in a few files
(`scripts/run_store.py`, the layout of the anisotropic AFC study) instead of one folder per run:

| file | content |
|---|---|
| `<prefix>_runs.jsonl` | one line per run: the run key (`<name>_<key1>-<value1>_...`; steady: `<study>_P<k>P<l>_nu<nu>`), the parameters, the input file, the table the program wrote, wall time, status; a repeated run counts with its last line |
| `<prefix>_runs.csv` | the same without input and table, one row per run (spreadsheet) |
| `<prefix>_failed.log` | ParMooN log of every failed run (last 2000 lines) |
| `<prefix>_files/<run>/` | the graphical output of a run: `mesh_t*.ps`, `vtk/*.vtk`, and figures drawn from them (`<prefix>_meshes.pdf`, `tgv_plots.py`) |
| `<prefix>_<study>_tables.csv` | steady studies: the per-level tables of all runs of a study |

Every run works in a temporary folder (`.scratch`, removed afterwards); `<input name>.out`, the
stdout log of a successful run and the input file as a file are not kept (the input is in the
record). Finished runs are skipped; `--redo` (or `--force`) recomputes them, `--retry-failed`
repeats failed ones. A lock (`.lock`) stops a second run step in the same folder (a lock of a
process that no longer exists is taken over). `--dry-run` writes the input files into
`<prefix>_dry_run/`.

Read a table or an input file of a run:

```bash
python3 -c "import json; [print(r['table']) for r in map(json.loads, open('tnse_taylor_green/tnse_taylor_green_runs.jsonl')) if r['run'] == 'ref_reynolds_number-100.0']"
```

Field plots of a Taylor-Green run: `python3 ../ParMooN-2026/scripts/tgv_plots.py tnse_taylor_green
--run <run key>` (into `<prefix>_files/<run>/`).

**Run folders of earlier versions** are converted without recomputing (input file, table and
graphical output; folders without a table become failed runs):

```bash
python3 ../ParMooN-2026/scripts/convert_ns_runs.py tnse_taylor_green oseen_aaj_smooth   # keeps the old folders
python3 ../ParMooN-2026/scripts/convert_ns_runs.py --delete-old tnse_taylor_green        # then removes them
```

Checked: runs of the previous and the new scripts (steady Examples 4.1 and 5.2, Taylor-Green with
two meshes) have identical tables, graphical output and study files (summaries, convergence
tables); converted runs reproduce them with `--plot-only`, `spg_plot.py` and `tgv_plots.py`.

A short test run first:

```bash
python3 $S --program tcd --base $E/tcd2d_adaptive_tr_ex4.dat --name ex4_test --outdir adaptive --set time_end=0.3
```

Comparison of tolerances in one call (one run per pair):

```bash
python3 $S --program tcd --base $E/tcd2d_adaptive_tr_ex4.dat --name ex4_tol --outdir adaptive \
    --vary adaptive_tol1/adaptive_tol2=6e-3/3e-4,3e-3/1.5e-4,1.5e-3/7.5e-5
```

## Convergence studies versus DoFs

Vary the refinement level for the points of a curve and any other parameter for the curves:

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_ex1.dat --name tnse_ex1 --outdir conv \
    --set output_write_vtk=false time_end=0.5 \
    --vary refinement_n_initial_steps=2,3,4,5,6 reynolds_number=1e2,1e3,1e4,1e6 --no-meshes
```

For pairs as curves, use `--vary refinement_n_initial_steps=2,3,4,5 VELOCITY_SPACE/PRESSURE_SPACE=2/1,3/2`. For adaptive meshes the tolerance is the x parameter:

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_ex2.dat --name tnse_ex2_tol --outdir conv \
    --vary adaptive_tol1/adaptive_tol2=4e-2/2e-3,2e-2/1e-3,1e-2/5e-4,5e-3/2.5e-4 \
    reynolds_number=1e2,1e4 --dofs mean
```

| output | content |
|---|---|
| `<name>_convergence.txt` | per curve: cells, DoFs, and for error, eta, L2(u), H1(u), L2(div u), L2(p), every estimator contribution and every error part the value and the rate w.r.t. h ~ DoFs^-1/2 (`---` for the first point); effectivity and its min/mean/max over time |
| `<name>_convergence.pdf` | error and eta vs DoFs with reference slopes, effectivity vs DoFs with its range over time as a band, cells vs DoFs |
| `<name>_contributions.pdf` | one panel per curve: eta, error, eta_res, eta_div, eta_F, eta_delta, eta_mu and the error parts nu^1/2 \|e_u\|_1, (sum mu_K \|\|div e_u\|\|^2)^1/2, (sum delta_K \|\|(b.grad)e_u + grad e_p\|\|^2)^1/2 |
| `<name>_norms.pdf` | L2(u), H1(u), L2(div u), L2(p) vs DoFs, one curve per parameter value |
| `<name>_tables.pdf`, `<name>_summary.txt` | quantities over time per run, and the per-run summary |

| option | meaning |
|---|---|
| `--measure final` (default) / `max` | value at the final time / maximum over time (start-up excluded) |
| `--dofs final` (default) / `mean` / `max` | DoFs per run; `mean` is the natural choice for adaptive meshes |
| `--x-key KEY` | x parameter; default `refinement_n_initial_steps`, otherwise the varied adaptive tolerance |
| `--slopes 1,2` | reference slopes h^s in the plots |
| `--plot-only` | regenerate tables and figures from finished runs; runs with a table are skipped anyway |

For a spatial study, make the time error negligible: `--set trab2_adaptive=false time_step_length=0.0025 adaptive_mesh=false`, or a small `trab2_tolt`. The CD program writes L2(u) and H1(u) but no estimator contributions, so it produces no contributions figure.

## Steady Oseen and Navier-Stokes, adaptive (Afzal, Ahmed, John 2026, Examples 4.2 and 5.2)

```bash
python3 ../ParMooN-2026/scripts/spg_study.py --study ex42 ex52 --pairs 2/1 2/2 --nu 1e-3 1e-6 --max-dofs 2e5
# Example 4.2 into the asymptotic range:
python3 ../ParMooN-2026/scripts/spg_study.py --study ex42 --pairs 2/1 --nu 1e-6 --initial 3 --max-dofs 1e6
```

Runs whose table already exists are not repeated; `--plot-only` regenerates the figures.

## Boundary layers that move to other walls (examples 36, 37)

Header `include/Examples/TNSE_2D/TNSE_layer_switch.h` (documentation of the construction at its
top), registered as examples 36 and 37 in `Example_TimeNSE2D`; input files
`data/input_files/tnse2d_adaptive_spg_layer_switch_a.dat` (36) and `_b.dat` (37).

Two steady solutions of the type of example 22 (Afzal-Ahmed-John 2026, Example 4.2):
u = curl(A(xi) A(eta)), A(s) = s^2 (1 - exp(k (s - 1)))^2, p = exp(xi + eta) - (e - 1)^2,

- NSE1: xi = x, eta = y, k = lambda = 0.5/sqrt(nu): layers at x = 1 and y = 1 (equal to
  `Oseen_AAJ_layers.h` to 4e-16);
- NSE2: k = mu = 0.75/sqrt(nu) (layers 1.5 times thinner), scaled by A_2 = lambda/mu so that the
  peak velocities are about equal; example 36: xi = 1 - x, eta = 1 - y, layers at x = 0, y = 0;
  example 37: xi = x, eta = 1 - y, layers at x = 1 and y = 0 (x = 1 stays, thinner).

u(t) = time1(t) NSE1 + time2(t) NSE2 with the C^2 step S(tau) = tau^3 (10 - 15 tau + 6 tau^2):
time1 = 1 on [0, c1], decays on [c1, c], 0 after c; time2 = 0 before d, rises on [d, d2], A_2
after d2. The velocity vanishes on the whole boundary at every time (homogeneous Dirichlet data),
is divergence-free, and the pressure has zero mean; f = d_t u - nu Lap u + (u.grad)u + grad p.

| parameter | default | meaning |
|---|---|---|
| `layer_switch_kappa1` | 0.5 | lambda = kappa1/sqrt(nu) of NSE1 |
| `layer_switch_kappa2` | 0.75 | mu = kappa2/sqrt(nu) of NSE2 |
| `layer_switch_scale_nse2` | true | scale NSE2 by kappa1/kappa2 |
| `layer_switch_c1`, `layer_switch_c` | 0.3, 0.5 | NSE1 switched off on [c1, c] (absolute times) |
| `layer_switch_d`, `layer_switch_d2` | 0.5, 0.7 | NSE2 switched on on [d, d2] |

With c = d (default) both factors vanish at t = c, so u(c) = 0: all refinement can disappear
before the new layers form. With c > d (e.g. `--set layer_switch_d=0.4 layer_switch_c=0.6`) both
layer systems are present on (d, c). The input files use nu = 1e-4 (layer widths 1/lambda = 0.020,
1/mu = 0.013, peak velocities 19.5 (NSE1) and 20.9 (A_2 NSE2)), [0, 1], adaptive meshes and time steps:

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_a.dat --name switch_a
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_b.dat --name switch_b
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_layer_switch_a.dat --name overlap_a \
    --set layer_switch_d=0.4 layer_switch_c=0.6
```

Checked: the header compiled on its own and compared with sympy at 84 points and times (both
variants, before, during and after the switch): u, its derivatives and Laplacian, p and its
derivatives and f agree to 2e-14, div u = 0 exactly, u = 0 on the boundary; NSE1 equals
`Oseen_AAJ_layers.h`; invalid switch times stop with an error. ParMooN runs both variants (short
windows at the start and across t = 0.5). Uniform meshes, variant a, t in [0.40, 0.42] (NSE1 being
switched off, d_t u != 0), dt = 1e-3: L2(u) 0.204 / 0.068 / 0.015 and E_kin/E_kin_exact - 1
= -16 % / -3.0 % / -0.46 % for h = 1/32, 1/64, 1/128 (the layer, width 0.02, is resolved only on
the last). The pressure error in the table (L2(p) about 240) does not decrease with h; the same
happens for example 23 (L2(p) 1.0 / 1.4 for h = 1/16, 1/64 while L2(u) drops by a factor 60), so it
is a property of how `tnse2d_adaptive_spg` reports the pressure, not of this example (open).

## Example 2: translating vortex with compact support (example 32)

With xi = x - c0 - U0 t, r = |xi|, rho = r^2/R^2:

    u = U0 + h(rho) (-xi_2, xi_1),   h = A (1 - rho)^m for rho < 1, 0 otherwise,
    p = -A^2 R^2 / (2(2m+1)) (1 - rho)^(2m+1) for rho < 1, 0 otherwise,
    f = -nu Laplace(u) = -nu (4 A m / R^2) (1 - rho)^(m-2) ((m+1) rho - 2) (-xi_2, xi_1).

The radial vortex is a steady Euler solution, and the translation keeps it one, so the only forcing is viscous. Domain (0,1)^2. Boundary conditions: u = U0 on x = 0, y = 0, y = 1, and a do-nothing outflow (nu grad u - p I) n = 0 on x = 1. The exact solution satisfies the outflow condition exactly, which is the homogeneous Neumann case the estimator assumes, and the pressure needs no mean value condition. With a Dirichlet outflow, every error carried to x = 1 becomes an unresolvable layer of width O(nu).

Parameters in the input file: `vortex_radius` R (0.15), `vortex_exponent` m (4, at least 3), `vortex_max_swirl` (1, determines A, about 32), `vortex_center_x/y` (0.25, 0.5), `vortex_velocity_x/y` (1, 0; x must be positive). The program stops if the vortex would leave the square in [time_start, time_end].

Exact quantities (constant in time):

- int |u - U0|^2 = pi A^2 R^4 / ((2m+1)(2m+2));
- int omega^2 = 4 m pi A^2 R^2 / ((2m-1)(2m+1));
- omega = 2A (1-rho)^(m-1) (1 - (m+1) rho): positive core, negative ring, zero circulation.

Checks:

- Symbolic (sympy): div u = 0, Euler residual 0, f = -nu Laplace u, the energy and enstrophy formulas.
- Convergence at nu = 0.1, R = 0.3, P2/P1: SPG norm 1.98, L2(u) 3.34, H1(u) 2.05, L2(p) 2.49, L2(omega) 2.10; effectivity 24.5 on every mesh.
- At nu = 1e-3, H1 and L2(omega) converge with order about 1.5 (convection-dominated SUPG regime), while the interpolant converges with order 2.

Uniform level 5 (2048 cells), nu = 1e-5, T = 0.5. Exact values: omega_min = -13.84, omega_max = 64.07, enstrophy ratio 1.

| variant | L2(u) | L2(p) | enstrophy / exact | omega_min | omega_max |
|---|---|---|---|---|---|
| Galerkin | 0.181 | 0.069 | 25.8 | -430 | 344 |
| grad-div only | 0.047 | 0.019 | 1.72 | -50 | 76 |
| SUPG/PSPG only | 0.065 | 0.035 | 2.17 | -149 | 93 |
| SUPG/PSPG + grad-div | 0.028 | 0.018 | 1.10 | -38 | 63 |

Adaptive, from 2048 cells, fixed dt = 0.005, at t = 0.3: 5035 cells; SPG error 1.10e-2 (uniform 2048 cells: 4.27e-2); L2(p) 6.9e-3 (uniform 1.35e-2); enstrophy ratio 1.11; vorticity extrema -29 ... -63 and 69 ... 76. The refinement covers the moving vortex and keeps a trailing region where the dispersive wake has real errors; faster coarsening behind the vortex is a matter of `adaptive_tol2`. The effectivity is about 13 on the graded meshes and about 20 on uniform ones: still above 1, but not mesh-independent (open).

New table columns of the transient NSE program, for every example with an exact solution: `E_kin`, `E_kin_exact`, `enstrophy`, `enstrophy_exact`, `L2(omega)`, `omega_min`, `omega_max`, `omega_min_exact`, `omega_max_exact`, `centroid_x`, `centroid_y`, `centroid_err`. The script draws them over time in `<name>_flow.pdf`. The centroid is |omega|-weighted, so spurious vorticity elsewhere (a wake) biases it.

## Backward Euler first step: comparison (commit `6f853d7`)

Example 1 (example 23, Re = 100, P2/P1, U^0 = L2 projection), uniform level 6, fixed dt = 1e-3, T = 0.3. With fixed steps there is no averaging.

| | `trab2_first_step semidiscrete` | `trab2_first_step backward_euler` |
|---|---|---|
| L2(p), t = 0.001 -> 0.3 | 1.364 -> 0.757 | 3.75e-3 -> 2.77e-3 |
| SPG error, t = 0.001 -> 0.3 | 0.160 -> 0.071 | 0.0256 -> 0.0240 |
| effectivity over the run | 2.74 - 7.23 | 19.9 - 21.0 |
| L2(u) at T | 1.114e-4 | 1.115e-4 |
| wall time | 1704 s | 1398 s (18 % less) |

The same with adaptive steps and periodic averaging (TOLT 1e-5, n* = 10), level 5:

| | semidiscrete | backward_euler |
|---|---|---|
| first 10 steps | L2(p) 1.57 -> 1.03, effectivity 6.7 - 11; cleaned by the first averaging (step 10) | L2(p) 5e-2 -> 1.9e-2, effectivity 19.3 - 19.9 from step 1 |
| at T = 0.3 | error 9.400e-2, L2(p) 1.21e-2, effectivity 20.78 | identical |
| steps / rejections / wall time | 49 / 1 / 56 s | 42 / 1 / 47 s |

The averaging is unchanged. The BE step restarts its counter, so the first averaging comes 10 TR steps after it. The two mechanisms are compatible: averaging removes the inconsistent part of the start after n* steps, and the BE step avoids it from the first step on.

## Parameters to tune

| parameter | meaning |
|---|---|
| `adaptive_tol1` | refine cells with eta_K > TOL1 |
| `adaptive_tol2` | coarsen parents whose children all have eta_K < TOL2 |
| `adaptive_coarsening` | coarsening on/off |
| `adaptive_max_cell_level` | maximal refinement level of a cell, a cap that protects against singularities |
| `trab2_tolt` | tolerance of the time step control (AB2 estimate, L2 norm) |
| `trab2_dt0`, `trab2_dt_max` | initial and maximal time step |
| `trab2_averaging` | `periodic` (default): every n* steps as in Gresho et al. (2008) and the manuscript; `on_demand`: n* steps after the start and after mesh changes only (my variant, not from the literature) |
| `trab2_averaging_steps` | n*, 10 by default; 0 switches averaging off |
| `adaptive_output_time` | interval of mesh and vtk output |
| `tnse_initial_value` | `l2_projection` (default, manuscript): L2 projection of u_0; `consistent_projection`: stabilized equations at the start time with the exact time derivative; `interpolation`: nodal interpolant |
| `trab2_first_step` | `semidiscrete` (default): TR from the start, with the initial time derivative from the semidiscrete problem; `backward_euler`: the first step is a backward Euler step, which needs no initial time derivative and damps the inconsistent part of the initial value (Rannacher's idea); accepted without an error estimate, keeps the proposed step size; NSE program only |
| `trab2_be_after_mesh_change` | `false` (default); `true`: the first step after every mesh change is a backward Euler step as well |
| `tnse_dt_alpha` | lower bound dt >= alpha min_K delta_K of the adaptive step (residual-based pressure stabilization, manuscript Section 4); default 0.5, 0 switches it off; alpha has to be calibrated |

How to choose TOL1. Estimate it as TOL1 ≈ eta / sqrt(number of cells you can afford), with eta taken from a short run on the initial mesh. A TOL1 far below this marks almost every cell in every step and the mesh explodes. That is what happened with the original Example 5 tolerance, TOL1 = 2e-4 at eta ≈ 0.5-1. TOL2 is typically TOL1/20.

`on_demand` averaging is not the algorithm of the manuscript. If it is used for published results, it has to be described: averaging (Gresho et al. 2008) is applied once, n* steps after the start and after every mesh change.

Pressure error and the start of the trapezoidal rule. U^0 (L2 projection or interpolant) is not consistent with the discrete constraint (the divergence row with PSPG). The trapezoidal rule is not L-stable and carries the inconsistent part, which lies in stiff modes, on with factor about -1; the pressure is the multiplier of the constraint and keeps an O(1) error that does not decrease with h. With fixed steps nothing removes it (no averaging); with the adaptive control the first periodic averaging does, and an interpolation to a new mesh brings it back until the next averaging. All `tnse2d_adaptive_spg_*.dat` files now set `trab2_first_step: backward_euler` (the backward Euler step damps the stiff part by about 1/(1 + k lambda)); the layer-switch files also set `trab2_be_after_mesh_change: true`. A fixed-step run that starts with TR from an inconsistent U^0 prints a warning. Example 23 (Example 1 of the manuscript), dt = 1e-3, t = 0.006, L2(p):

| level | TR from L2 projection (old) | BE first step, L2 projection | TR from `consistent_projection` |
|---|---|---|---|
| 4 | 5.1e-1 | 7.2e-2 | 6.3e-2 |
| 5 | 1.43 | 1.7e-2 | 1.4e-2 |
| 6 | 1.35 | 3.4e-3 | 3.2e-3 |

Both remedies give O(h^2); `consistent_projection` needs the exact solution and Dirichlet data, the backward Euler step needs neither. L2(u) on level 6: 5.5e-5 (old), 6.4e-5 (BE first step), 7.8e-5 (`consistent_projection`). The single first-order step costs O(k^2) locally, the order of the TR error.

## Parameter reference

All parameters of the adaptive programs with their defaults and admissible values, generated from the `db.add(...)` declarations in the source (commit `6f853d7`). The descriptions are the ones in the code. A parameter not set in the input file has its default.


### Adaptive time loop: mesh adaptation and output

Source: `src/System/AdaptiveTimeLoop.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `adaptive_mesh` | `true` | - | adaptive mesh refinement/coarsening in time |
| `adaptive_tol1` | `1.e-3` | range [0., 1e10] | cells with eta_K > TOL1 are refined |
| `adaptive_tol2` | `5.e-5` | range [0., 1e10] | cells with eta_K < TOL2 are candidates for coarsening |
| `adaptive_coarsening` | `true` | - | coarsening on/off |
| `adaptive_max_cell_level` | `30` | range [1u, 100u] | maximal refinement level of a cell |
| `adaptive_output_time` | `0.1` | range [0., 1e10] | interval for mesh output (PostScript, needs output_write_ps) and system output (vtk, needs output_write_vtk; files <output_directory>/<output_basename>.<index>.vtk) |

### TR-AB2 time integrator (both adaptive programs)

Source: `src/TimeDiscRoutines/TrapezoidalAB2.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `time_start` | `0.` | any | start time |
| `time_end` | `1.` | any | end time |
| `time_step_length` | `0.01` | range [0., 1e10] | step length if trab2_adaptive is false |
| `trab2_adaptive` | `true` | - | variable step size control (AB2 estimate) |
| `trab2_tolt` | `1.e-5` | range [0., 1.] | tolerance TOLT of the error estimate |
| `trab2_dt0` | `1.e-10` | range [0., 1e10] | initial step (adaptive mode) |
| `trab2_dt_min` | `1.e-12` | range [0., 1e10] | minimal step (no rejection below) |
| `trab2_dt_max` | `0.1` | range [0., 1e10] | maximal step |
| `trab2_factor` | `0.85` | range [0., 1.] | safety factor of the step size update |
| `trab2_averaging_steps` | `10` | range [0, 1000000] | average every n* steps (0: off) |
| `trab2_averaging` | `periodic` | one of `periodic`, `on_demand` | periodic: every n* steps as in Gresho et al. 2008 and de Frutos et al. 2014; on_demand: n* steps after the start and after mesh changes only (a variant, not from the literature) |
| `trab2_first_step` | `semidiscrete` | one of `semidiscrete`, `backward_euler` | first step: semidiscrete (TR with the time derivative from the semidiscrete problem at time_start) or backward_euler (a backward Euler step, no initial time derivative needed; damps inconsistent initial data) |
| `trab2_be_after_mesh_change` | `false` | - | take a backward Euler step after every mesh change as well |

### Transient Navier-Stokes program `tnse2d_adaptive_spg`

Source: `2DPrograms/TNSE2D_AdaptiveSPG_ParMooN.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `tnse_write_table` | (empty) | - | file name for the table |
| `tnse_initial_value` | `l2_projection` | one of `l2_projection`, `consistent_projection`, `interpolation` | initial velocity: l2_projection (L2 projection of u(time_start), as in the manuscript), consistent_projection (stabilized equations at time_start with the exact time derivative; consistent with the discrete constraint) or interpolation (nodal interpolant) |
| `tnse_dt_alpha` | `0.5` | range [0., 1e10] | lower bound dt >= alpha min_K delta_K of the time step in the step size control (residual-based pressure stabilization, manuscript Section 4); 0 switches it off; the constant has to be calibrated |

### SUPG/PSPG/grad-div method and its estimator (`space_discretization_type: supg_pspg_graddiv`)

Source: `src/AssembleRoutines/LocalAssembling.C, src/System/NSE_SPG_ErrorEstimator.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `spg_delta0` | `0.5` | range [0., 100.] | SUPG/PSPG parameter delta_0 of the 'supg_pspg_graddiv' method for Oseen/Navier-Stokes (Afzal, Ahmed, John 2026): delta_K = delta_0 h_K^2 for inf-sup stable pairs; delta_0 h_K (if nu < h_K) or delta_0 h_K^2 for equal order pairs. |
| `spg_mu0` | `0.5` | range [0., 100.] | grad-div parameter mu_0 of the 'supg_pspg_graddiv' method: mu_K = mu_0 for inf-sup stable pairs and mu_K = mu_0 h_K for equal order pairs. |
| `spg_estimator_sigma0` | `0.` | range [0., 1e10] | lower bound sigma_0 of the reaction coefficient used in the minima of eta_{K,res} and eta_F; 0 means that the corresponding terms are dropped. |
| `spg_error_quad_degree` | `19` | range [1, 19] | polynomial degree of the quadrature used for the exact error (solutions with layers need a high degree) |

### Steady Oseen/Navier-Stokes program `oseen2d_spg_adaptive`

Source: `2DPrograms/Oseen2D_SPG_Adaptive_ParMooN.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `spg_uniform_refinement` | `false` | - | refine uniformly instead of adaptively (convergence study) |
| `spg_write_table` | (empty) | - | file name for the summary table |

### Translating vortex (example 32, manuscript Example 2)

Source: `src/Examples/Example2D.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `vortex_radius` | `0.15` | range [0., 1.] | translating vortex (Example_TimeNSE2D 32): radius R of the support |
| `vortex_exponent` | `4` | range [3, 20] | translating vortex: exponent m of h = A (1 - r^2/R^2)^m (m >= 3) |
| `vortex_max_swirl` | `1.0` | range [0., 1e10] | translating vortex: maximal swirl speed (determines A) |
| `vortex_center_x` | `0.25` | any | translating vortex: x of the centre at t = 0 |
| `vortex_center_y` | `0.5` | any | translating vortex: y of the centre at t = 0 |
| `vortex_velocity_x` | `1.0` | any | translating vortex: x component of the translation velocity U0 |
| `vortex_velocity_y` | `0.0` | any | translating vortex: y component of U0 |

### Convection-diffusion program `tcd2d_adaptive_tr`

Source: `2DPrograms/TCD2D_AdaptiveTR_ParMooN.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `tcd_estimator_mu0` | `1.` | range [0., 1e10] | lower bound mu_0 of c - div(b)/2 used in the estimator; 0 drops the corresponding terms |
| `tcd_write_table` | (empty) | - | file name for the table |

### BDF2-IMEX/CIP program `tnse2d_bdf2imex_cip`

Source: `2DPrograms/TNSE2D_BDF2IMEX_CIP_ParMooN.C`

| parameter | default | admissible | meaning |
|---|---|---|---|
| `cip_delta1` | `1.e-2` | range [0., 1e10] | CIP parameter delta_1 (S_1) |
| `cip_delta2` | `1.e-4` | range [0., 1e10] | CIP parameter delta_2 (S_2) |
| `cip_delta3` | `1.e-6` | range [0., 1e10] | CIP parameter delta_3 (S_3) |
| `cip_u_min` | `0.25` | range [0., 1e10] | lower bound of the normalization |
| `cip_u_max` | `-1.` | range [-1., 1e10] | upper bound (clamp) of the normalization, inactive if <= 0 |
| `cip_quad_degree` | `8` | range [1, 20] | degree of the edge quadrature |
| `bdf2_start` | `stokes_projection` | one of `stokes_projection`, `euler` | starting values u^0, u^1 |
| `bdf2_write_table` | (empty) | - | file name for the per-step table |
| `bdf2_periodic` | `false` | - | periodic boundary conditions (the example's boundary condition must return PERIODIC) |
| `bdf2_sample_grid` | `0` | range [0, 4096] | if > 0: sample velocity, vorticity and divergence on an N x N grid at the final time (file <output_basename>_sample.txt) and print vorticity range and total variation |
| `bdf2_start_pressure` | `interpolation` | one of `interpolation`, `stokes` | pressure of the starting values (only for the error table): interpolant of the exact pressure, or the pressure of the Stokes projection |
| `pressure_amplitude` | `1.` | any | amplitude A of the pressure in example 25 (TDatabase::ParamDB->P0) |

### General ParMooN parameters used by the inputs

These are declared by ParMooN itself. Values are those of `tnse2d_adaptive_spg_ex2.dat`.

| parameter | value in Example 2 | meaning |
|---|---|---|
| `example` | 32 | example number (Example_TimeNSE2D for the transient programs) |
| `problem_type` | 5 | 5 = Navier-Stokes, 3 = Oseen |
| `reynolds_number` | 1.0e5 | 1/nu |
| `VELOCITY_SPACE`, `PRESSURE_SPACE` | 2, 1 | finite element pair (here Taylor-Hood P2/P1) |
| `space_discretization_type` | `supg_pspg_graddiv` | the stabilized method; with `spg_delta0 = spg_mu0 = 0` it is the Galerkin method |
| `NSTYPE`, `LAPLACETYPE`, `NSE_NONLINEAR_FORM` | 14, 0, 0 | matrix structure, gradient form of the viscous term (do-nothing outflow (nu grad u - p I) n = 0), convective form |
| `boundary_file`, `geo_file`, `refinement_n_initial_steps` | `Default_UnitSquare`, `TwoTriangles`, 4 | domain and initial uniform refinements |
| `nonlinloop_maxit`, `nonlinloop_epsilon` | 30, 1e-10 | Picard iteration per time step |
| `solver_type`, `direct_solver_type` | `direct`, `umfpack` | linear solver |
| `output_write_vtk`, `output_write_ps`, `output_directory`, `output_basename`, `outfile` | | output files |

### Typical settings

```
# start with a backward Euler step (no initial time derivative needed)
trab2_first_step: backward_euler
trab2_be_after_mesh_change: true

# time step control as in the manuscript
trab2_adaptive: true
trab2_tolt: 1.0e-5
trab2_averaging: periodic
trab2_averaging_steps: 10
tnse_dt_alpha: 0.5

# fixed steps (verification runs)
trab2_adaptive: false
time_step_length: 1.0e-3

# mesh adaptation (TOL1 about eta / sqrt(number of cells you can afford), TOL2 = TOL1/20)
adaptive_mesh: true
adaptive_tol1: 5.0e-3
adaptive_tol2: 2.5e-4
adaptive_max_cell_level: 9

# Galerkin method instead of SUPG/PSPG/grad-div (same program)
spg_delta0: 0.0
spg_mu0: 0.0
```

## Verification runs (fixed mesh or fixed time step)

Temporal order of the TR step (Example 28, exact solution in P2/P1, phi(t) = 1 + sin(pi t)/2; rates should be 2.00 in `verify_tr_temporal.txt`). With `--set example=30` the solution is quadratic in time and the errors must be at round-off level for every dt:

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_verify_time.dat --name verify_tr \
    --vary time_step_length=0.1,0.05,0.025,0.0125 --no-meshes
```

Temporal order of BDF2-IMEX/CIP with Scott-Vogelius (rates in `verify_bdf2_results.txt`). With `--set example=31` the solution is linear in time and the errors must be at round-off level:

```bash
python3 $S --program bdf2 --base $E/tnse2d_bdf2imex_verify_time.dat --name verify_bdf2 \
    --vary time_step_length=0.1,0.05,0.025 --no-meshes
```

Spatial convergence. The errors at the final time in all norms, with rates, are in the second table of `<name>_summary.txt`.

```bash
python3 $S --program tnse --base $E/tnse2d_adaptive_spg_ex1.dat --name spatial_P2P1 \
    --set trab2_adaptive=false time_step_length=0.0025 time_end=0.05 adaptive_mesh=false output_write_vtk=false \
    --vary refinement_n_initial_steps=2,3,4,5,6 --no-meshes
python3 $S --program tcd --base $E/tcd2d_adaptive_tr_ex2.dat --name spatial_cd \
    --set trab2_adaptive=false time_step_length=0.001 time_end=0.1 adaptive_mesh=false output_write_vtk=false \
    --vary refinement_n_initial_steps=3,4,5,6,7 --no-meshes
```

Reference values from the sandbox runs:

- P2/P1: SPG norm rate 2.00, L2(p) rate 2.15, effectivity 19.9-20.0.
- CD, P1: SUPG norm rate 1.48-1.51, effectivity 6.76-6.9.

## Regression suite and exactness checks

```bash
python3 ../ParMooN-2026/test/regression/regression.py --build .                   # all checks, about 6 minutes
python3 ../ParMooN-2026/test/regression/regression.py --build . --cases tnse_ex1  # selected cases
python3 ../ParMooN-2026/test/regression/regression.py --build . --update          # new references
```

Two kinds of checks:

- 13 reference cases. Short runs of the shipped inputs: transient NSE examples 23, 24, 28 with fixed and adaptive steps and meshes; BDF2 examples 25-28 with Scott-Vogelius and Taylor-Hood; steady Oseen examples 21, 22; CD examples 11, 15; the old TNSE program. Every number of the table and the result lines of the log are compared with the references in `test/regression/reference/` (rtol 1e-10). The references come from the sandbox. Another compiler or BLAS changes the last digits, and in adaptive runs this can flip a marking decision. Create your own references once with `--update` from a known good commit, before the next change.
- 14 exactness checks. These need no references and are independent of the machine: examples whose solution the scheme must reproduce up to round-off.

| check | example | must be exact because | sandbox result |
|---|---|---|---|
| `exact_oseen_P2P1`, `exact_oseen_P2P2`, `exact_nse_P3P2` | 29, stationary, u in P2, p in P1 | consistent stabilization; all estimator residuals vanish | error <= 4e-14, eta <= 3e-13 |
| `exact_tr_fixed`, `exact_tr_adaptive_dt` | 30, phi = 1 + t + t^2 | trapezoidal rule exact for quadratic solutions (no averaging) | <= 8e-12 |
| `exact_tr_mesh_alternation` | 30, refinement and coarsening of all cells in alternation (32 <-> 2048 cells) | P2 transfer exact | <= 2e-13 |
| `exact_tr_galerkin` | 30 with spg_delta0 = spg_mu0 = 0 | initial derivative without PSPG (differentiated constraint) | <= 3e-13 |
| `exact_be_first_fixed`, `exact_be_first_adaptive_dt` | 31 with `trab2_first_step backward_euler` (fixed and adaptive steps) | BE and TR exact for solutions linear in time; the log must show step 1 as BE and all later steps as TR | <= 7e-12 |
| `exact_be_mesh_alternation` | 31, refinement/coarsening every step, `trab2_be_after_mesh_change true` | every step after a mesh change must be BE | <= 1e-13 |
| `exact_oseen_neumann`, `exact_tr_neumann` | 33, 34: outflow (Neumann) boundary with the exact traction | Neumann data, estimator facets, initial derivative with outflow | <= 5e-13 |
| `exact_bdf2_sv_cip`, `exact_bdf2_th` | 31, phi = 1 + t | BDF2 and the extrapolation 2u^n - u^{n-1} exact for linear solutions | <= 1e-14 |

A failing exactness check points to a defect in the assembly, the estimator, the time stepping or the mesh transfer. Expected non-exact cases: the IMEX Euler start of the BDF2 program (convection explicit in the first step) and steps with averaging (O(k^2) perturbation).

## Cost and known limitations

- Examples 3 and 5 and the full rotation of Example 4 take hours and need several GB of memory at the finest meshes. Example 5 stayed at about 31-37k cells with the shipped tolerances in short tests.
- Equal-order pairs (e.g. P2/P2) in the time-dependent Navier-Stokes solver become unstable for time steps well below the stabilization parameter (delta = delta_0 h). Observed on Example 23, levels 2-3, for dt = 0.01, but not for dt = 0.04. P2/P1 is not affected. Use inf-sup stable pairs for transient runs until this is resolved.
- Time-dependent Dirichlet data are supported by both time-dependent solvers (verified with Examples 28, 30, 31). The estimator of the Navier-Stokes solver assumes Dirichlet or homogeneous Neumann data; outflow problems such as the flow around a cylinder are not covered yet.
- CIP (BDF2 program): S_3 (`cip_delta3 > 0`) is implemented for velocity degree 2 only; higher degrees now stop with an error.
- CIP normalization: the code divides by max{||u~||_inf, `cip_u_min`} with `cip_u_min` = 0.25. The BDF2 manuscript divides by ||u~||_inf, with S := 0 for u~ = 0, which the code also implements. Set `cip_u_min: 0` to follow the manuscript, or state the lower bound in the paper.
- `CELL_MEASURE`: the assembly, the estimator (delta_K, mu_K) and the step lower bound use the same cell measure (fixed in `b986906`); the estimator's own h_K weights are the cell diameter, as in the theory.
- Initial time derivative (fixed in `1cd520e` and `545ba8c`). The linear semidiscrete problem for (Ud^0, P^0) prescribed by the manuscript is singular in two cases. Without PSPG (spg_delta0 = 0) its pressure rows vanish. With PSPG and an outflow boundary, its PSPG pressure rows determine only Ud + grad P. In both cases the code now uses the differentiated constraint B Ud = 0. Equal-order pairs with PSPG and an outflow boundary stop with an error, because the initial derivative is not determined there. The manuscript should state this.
- The pressure of runs started from the L2 projection with the trapezoidal rule was polluted by the inconsistent part of U^0 (L2(p) = O(1), not decreasing with h). Fixed: the input files start with a backward Euler step (see the section on the pressure error above). The manuscript should state the start step.
