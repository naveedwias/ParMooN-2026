# Algebraic stabilizations on layer-adapted meshes with ParMooN (branch `adaptive-refactor`)

The ParMooN version of the study in the paper *Algebraically stabilized finite element methods on
layer-adapted meshes: accuracy, discrete maximum principle and nonlinear solvers*
(galerkit-latex, `convection_diffusion/anisotropic_afc`). Everything below refers to commit
`db77b0c` or later. All commands are run from the build directory, which sits next to the source
tree (`build/` and `ParMooN-2026/` in the same parent directory). The tools themselves are
described in `scripts/aniso_afc/README.md`.

## Where the code is

| what | where |
|---|---|
| the eight methods (BJK, Kuzmin, MC, MUAS, SMUAS, BBK, F-BBK, S-BBK), the switches, the relaxed and the regularized limiters | `include/AssembleRoutines/AlgebraicStabilization.h`, `src/AssembleRoutines/AlgebraicStabilization.C` (separate from `AlgebraicFluxCorrection`) |
| fixed point iteration, Anderson acceleration with safeguard, Newton with continuation | `ConvectionDiffusion_AFC::solve`, `solve_algstab_anderson`, `solve_algstab_newton` (`src/System/ConvectionDiffusion_AFC.C`) |
| program | `cd2d_afc` (`2DPrograms/CD2D_AFC_ParMooN.C`) |
| Examples 1-5 of the paper | `Example_CD2D` 27, 7, 18, 29, 30 (28: cross-check problem b = (2,3), c = 1) |
| meshes, measures, study, storage of the runs, tables, regression | `scripts/aniso_afc/` (`meshes.py`, `measures.py`, `run_study.py`, `store.py`, `report_extra.py`, `regression.py`, `convert_runs.py`) |

## Build and check

```bash
cd ParMooN-2026 && git pull && cd ../build
cmake . && make cd2d_afc
python3 ../ParMooN-2026/scripts/aniso_afc/regression.py --exe ./parMooN_cd2d_afc_SEQUENTIAL
```

`cmake .` is needed once after the pull that added `AlgebraicStabilization.C`. The regression
must print `18 of 18 cases passed`. Python 3.6 or later with numpy and scipy is required
(matplotlib for the figure); galerkit is not needed.

## Reproducing the paper's results

Finished runs are skipped, so every command can be stopped and repeated, and the steps can be
spread over several sessions. Run one step at a time in a run folder: a second step in the same
`cd_aniso/` stops with a message (`cd_aniso/.lock`). `J` is the number of parallel runs.

```bash
S=../ParMooN-2026/scripts/aniso_afc/run_study.py
J=$(nproc)
C123="1:1e-4 1:1e-8 2:1e-8 3:1e-8"

# Examples 1-3: accuracy, DMP, iterations, meshes
python3 $S --jobs $J --cases $C123 --n 16 32 64 128 --meshes uniform shishkin_r shishkin_l shishkin_x
# switch-off by the Peclet number, Example 2
python3 $S --jobs $J --cases 2:1e-8 --n 64 128 --meshes shishkin_r shishkin_x --variants roff
# layer-adapted meshes, with and without the layer switch
python3 $S --jobs $J --cases $C123 --n 16 32 64 128 \
        --meshes shishkin_r bakhvalov_shishkin bakhvalov duran --variants plain loff
# Anderson acceleration without and with the safeguard
for sv in fp_anderson fp_anderson_safe; do
  python3 $S --jobs $J --cases $C123 --n 128 --meshes uniform shishkin_r shishkin_x --solver $sv
done
# Newton with regularized limiters and continuation
python3 $S --jobs $J --cases $C123 --n 16 32 64 128 \
        --meshes uniform shishkin_r bakhvalov_shishkin bakhvalov --variants plain loff --solver newton_cont
# Example 3 at n = 32 with the three solvers
for sv in fp_anderson_safe newton_cont; do
  python3 $S --jobs $J --cases 3:1e-8 --n 32 --meshes uniform shishkin_r shishkin_x --solver $sv
done
# Example 4 (straight interior layer)
for sv in fixed_point fp_anderson_safe newton_cont; do
  python3 $S --jobs $J --cases 4:1e-8 --n 16 32 64 \
          --meshes uniform shishkin_r bakhvalov_shishkin bakhvalov --variants plain loff --solver $sv
done
# Example 5 (profile along circles), with the relaxed limiter mu = 2
for sv in fixed_point fp_anderson_safe newton_cont; do
  python3 $S --jobs $J --cases 5:1e-5 --n 32 64 \
          --meshes uniform polar polar_layer tensor_layer --variants plain loff mu2 --solver $sv
done
# the tables and the figure of the paper from all runs (reads the records only, runs nothing)
python3 $S --paper paper_results
```

`--paper` can be called at any time; it writes what the finished runs allow and names the
missing tables with the step that gives their runs.

About 3900 runs. Most take seconds; the expensive ones are the fixed point runs at n = 128 that
reach the limit of 3000 iterations. Several hours to a day on 8 cores is an estimate, not a
measurement: each step was tested at n <= 32 only.

## Output

Two places, both in `build/`:

**`paper_results/`: the results of the paper**, written by `--paper`, under the paper's file
names, ready to replace the files in the paper's `tables/` folder:

| file | contents |
|---|---|
| `aniso_accuracy.tex`, `aniso_dmp.tex`, `aniso_efficiency.tex`, `aniso_mesh.tex` | Examples 1-3: accuracy, DMP, iterations, meshes |
| `aniso_roff.tex` | switch-off by the Peclet number (Example 2) |
| `aniso_layermesh_it.tex`, `aniso_layermesh_acc.tex` | layer-adapted meshes |
| `aniso_anderson.tex`, `aniso_newton_all.tex`, `aniso_newton_reg.tex` | nonlinear solvers |
| `aniso_interior.tex`, `aniso_rotating.tex` | Examples 4 and 5 |
| `aniso_errors.pdf` | L2 errors against the number of nodes (PDF instead of the paper's PNG) |
| `rotating_solutions_n64.png` | Example 5, n = 64: the four meshes and the solutions of Galerkin, SUPG, Kuzmin, MUAS, BBK (min, max in brackets); `rotating_figure.py`, see below |

**`cd_aniso/`: the runs** (raw data, not needed for the paper): one folder per example with
four files, e.g. for Example 1:

```
cd_aniso/cd_exp_layer/
  cd_exp_layer_runs.jsonl      one line per run: parameters, all measures, iterations, status
  cd_exp_layer_runs.csv        the same as a table (spreadsheet)
  cd_exp_layer_failed.log      ParMooN logs of runs that failed or did not converge
  cd_exp_layer_solutions.npz   nodal values (Examples 4, 5 by default; --solutions all / none)
```

The other folders are `cd_boundary_layers`, `cd_parabolic_layers`, `cd_interior_layer`,
`cd_rotating_profile`. A line of the `.jsonl` file (Example 1, eps = 1e-4, Shishkin "/", n = 16,
BJK):

```
{"run": "eps1e-04_shishkin_r_n16_bjk", "example": 1, "eps": 0.0001, "mesh": "shishkin_r", "n": 16,
 "method": "bjk", "variant": "plain", "solver": "fixed_point", "wall_time": 0.4, ...,
 "L2": 0.000677, "Linf_nodes": 0.01628, "Linf_away": 0.001475, "iterations": 192,
 "residual": 9.76e-11, "converged": true, "certificate": {...}, "local_dmp": {...}, ...}
```

The run key (`run`) is `eps<eps>_<mesh>_n<n>_<method>[_<variant>][_<solver>]`, with the solver
suffixes `_aa` (Anderson), `_aas` (Anderson with safeguard), `_nw` (Newton), `_nc` (Newton with
continuation). A solution: `np.load("cd_rotating_profile_solutions.npz")["eps1e-05_polar_n64_bjk_aas"]`,
its mesh points: key `xy_eps1e-05_polar_n64`. Every run is computed in a temporary folder
(`cd_aniso/.scratch`) that is removed afterwards; only the main process writes the four files,
so parallel runs cannot interfere. A run that was repeated counts with its last line.

Failed runs (e.g. a mesh that cannot be built, such as the layer-adapted polar mesh at n = 16)
have `"failed": true` and a reason in the `.jsonl` (column `failed` of the `.csv`) and are not
repeated, unless `--retry-failed`. `--report-only` still writes the tables of given cases into
`cd_aniso/`, if needed.

**Runs of an earlier version** (one folder per run, up to `be902fa`) are converted without
recomputing:

```bash
python3 ../ParMooN-2026/scripts/aniso_afc/convert_runs.py                # converts, keeps the old folders
python3 ../ParMooN-2026/scripts/aniso_afc/convert_runs.py --delete-old   # then deletes them
rm -f cd_aniso/cd_aniso_*                                                # interim tables of earlier versions
```

The paper tables of converted runs are byte-identical to those of the old layout.

## Example 5: meshes and solutions (`rotating_solutions_n64`)

```bash
python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py --run --jobs $J   # the runs it needs, then the figure
python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py                   # only the figure, from existing runs
```

Writes `paper_results/rotating_solutions_n64.png` (layout of galerkit's figure: one row per
mesh, the mesh and the solutions of Galerkin, SUPG, Kuzmin, MUAS, BBK, colour scale [-0.1, 1.1],
min and max of u_h in the titles) and, in `rotating_vtk/`, one VTK file per mesh
(`rotating_n64_<mesh>.vtk`: the mesh and the point data `u_Galerkin`, `u_SUPG`, `u_Kuzmin`,
`u_MUAS`, `u_BBK`) and `rotating_solutions_n64.md` (the run, solver, iterations, residual, min and
max of every panel). The solver of an algebraic method is chosen as in galerkit: the first of
`fp_anderson_safe`, `newton_cont`, `fixed_point` that converged (`--solver` fixes one). `--run`
computes only the missing runs, in that order, each solver only where the earlier ones did not
converge; the runs of step 10 are used if they exist. `--paper` also writes the PNG (not the
VTK files) when the runs exist. Options: `--n`, `--variant loff|mu2`, `--figdir`, `--vtkdir`,
`--outdir`.

## Not yet produced

- Six figures of the paper: `aniso_meshes`, `aniso_iterations`, `interior_cut_shishkin_n64`,
  `interior_errors`, `rotating_outlet_polar_layer_n64`, `rotating_errors` (the data are in the
  run folders).
- In the tables: the wall times of `aniso_newton_reg` (recorded as `wall_time`), the column
  max u_h of Kuzmin for Example 3 in `aniso_layermesh_acc`, and the rows with mu = 2 in
  `aniso_rotating` (the runs are done).

## What to expect against the galerkit results

- Errors, DMP certificates and local DMP agree with galerkit wherever both codes converge
  (checked for Example 1). The L2 errors of Examples 1, 2 are ParMooN's (its quadrature; within
  2e-4 relative of galerkit's); those of Examples 4, 5 are the discrete lumped norms of the paper.
- The fixed point iteration uses ParMooN's dynamic damping. Its iteration counts, and therefore
  which runs reach the limit of 3000 iterations, differ from the paper's tables.
- Anderson with safeguard: the same convergence, counts within a few percent, the same numbers of
  safeguard rejections. For S-BBK the count depends on round-off (318 or 226 iterations for the
  same solution on two machines).
- Newton with continuation: the continuation stages equal galerkit's in most cases; the final
  Newton steps on sigma = 0 can take a few more iterations (finite-difference instead of exact
  Jacobian).
- The stopping criterion is the Euclidean residual 1e-10 of the paper: ParMooN multiplies
  `afc_nonlinloop_epsilon` by sqrt(N), and `run_study.py` passes 1e-10 / sqrt(N).

## Troubleshooting

| symptom | cause and remedy |
|---|---|
| `ValueError: Wrong number of columns` in `read_edges`, during step 3 | fixed in `1f16349` (eps was not in the run folder names; Example 1 with two eps shared folders). Pull, `rm -rf cd_aniso`, restart. |
| `git pull` refuses: local changes to `run_study.py` / `regression.py` | a patch was applied by hand: `git checkout -- scripts/aniso_afc/run_study.py scripts/aniso_afc/regression.py`, then pull |
| `TypeError ... capture_output` in `regression.py` | Python 3.6; fixed in `1f16349` |
| regression FAIL with equal error, other iteration count | round-off in a sensitive case; fixed in `956eeb4` for Anderson with S-BBK. Other such cases: `--it-tol 0.5`; report a different error or a change of convergence |
| `the weights beta of fbbk could not be computed` | fixed in `35c2dbf` (Jacobi rotation in the weight QP) |
| `another run step (process N) is running in .../cd_aniso` | a second run step in the same folder: wait for the first, or stop it (`kill N`). A lock of a process that no longer exists is taken over automatically. |
| thousands of folders `cd_aniso/cd_<example>_aniso_eps...` | runs of a version up to `be902fa`: `convert_runs.py --delete-old` (see Output) |
| `--report-only` stops with `max() arg is an empty sequence` (e.g. `--name ex5`), before `cb6f6b1` | the case has no runs yet (check `ls -d cd_aniso/cd_rotating_profile_* \| wc -l`): run its step first. Since `cb6f6b1` the report says so instead of crashing, and writes only the tables of the requested cases. |

## Parameter reference

Parameters of `AlgebraicStabilization` (default in brackets):

| parameter | meaning |
|---|---|
| `algstab_method` (`none`) | `bjk`, `kuzmin_edge`, `mc`, `muas`, `smuas`, `bbk`, `fbbk`, `sbbk`; needs `algebraic_flux_correction: afc`, `afc_iteration_scheme: fixed_point_rhs`, `afc_initial_iterate: afc_zero` |
| `algstab_solver` (`fixed_point`) | `fixed_point`, `fp_anderson`, `fp_anderson_safe`, `newton`, `newton_cont` |
| `algstab_resolved_off` (false), `algstab_pe_threshold` (1) | stabilization off where \|b(x_E)\| \|E\| / (2 eps) <= threshold |
| `algstab_layer_off` (false), `algstab_aspect_threshold` (10) | off along the layers (long edges of cells with aspect ratio >= threshold; 3 on polar meshes) |
| `algstab_q_relax` (1) | relaxed limiters, mu |
| `algstab_anderson_depth` (10), `algstab_anderson_window` (10), `algstab_restart` (200), `algstab_omega_min` (1e-4) | Anderson acceleration and its safeguard |
| `algstab_reg` (1e-6), `algstab_newton_maxit` (40), `algstab_fd_step` (1e-10) | Newton continuation: target sigma, iterations per stage, finite-difference step |
| `algstab_bbk_p` (10), `algstab_bbk_C0` (3), `algstab_bbk_beta0` (0.1) | BBK-type methods: exponent, edge weight C0 \|E\|, lower bound of the F-BBK weights |
| `algstab_smuas_weights` (`67`) | SMUAS weights (67) or (68) |
| `algstab_tie_tol` (1e-10) | Kuzmin: a_ij = a_ji up to this relative tolerance counts as a tie |
| `algstab_write_nodal`, `algstab_write_edges`, `algstab_test_b` (empty) | output: nodal values (17 digits), edges with d_ij and switches, b_ij(u) for a given u |

SUPG in the study uses `space_discretization_type: supg`, `SDFEM_TYPE: 2`, `CELL_MEASURE: 4`
(the coth formula with the cell size in the direction of b; ParMooN's default uses the cell
diameter and gives errors of order 1 on the Shishkin meshes).

## Findings for the paper

- **MUAS in ParMooN.** `afc_limiter: MUAS_MAX` is the MUAS of the paper; `afc_limiter: MUAS` is
  a different variant. The old `afc_limiter: BBK` is an edge form of S-BBK type, not BBK.
- **Kuzmin on crossed meshes.** Many edges have a_ij = a_ji, exactly or up to round-off; with an
  exact comparison the round-off of the assembly decides the owner of an edge.
- **S-BBK on the Shishkin "/" mesh.** At least two solutions, 3.9e-6 apart, each with residual
  about 1e-13; which one is reached depends on the start.
- **Fixed point and Newton.** BJK on a Shishkin mesh with reaction: the fixed point iteration
  stalls at residual 4.9e-3 after 1e5 iterations; Newton converges in 12 iterations.
- **The regularization is not symmetric.** The smoothed maximum b + pos_sigma(a - b) gives
  max_sigma(a, 0) = a^2/(a + sigma) but max_sigma(0, a) = a, so the regularized b_ij depends on
  the orientation of the edge, i.e. on the node numbering, at O(sigma). It disappears after the
  final Newton steps on sigma = 0; a symmetric form (a + b + |a - b|_sigma)/2 would avoid it.
- **Finite-difference Jacobian.** galerkit's step 1e-7 (`reg_jacobian="fd"`) is too coarse for
  sigma = 1e-6 and fails in the final Newton steps on sigma = 0; 1e-10 reproduces the exact
  Jacobian's continuation.
- **Wording in the paper.** The orders "about 1.55" and "0.95" refer to n; with respect to
  N^{-1} ln N (the reference slope of Figure 2) they are 2.01 and 1.22. The "L2 errors" of
  Examples 4, 5 are lumped nodal norms. For runs that were not solved, "best iterate" includes
  the Newton fallback.

## Verification against galerkit (branch `feature/anisotropic-afc`)

- Galerkin: equal to round-off on uniform, Shishkin, crossed and polar meshes (Examples 1, 3, 4, 5, 28).
- b_ij(u) of all eight methods: equal to round-off for random u and the solution, with the
  switches, mu = 2, Neumann nodes; the regularized b_ij(u) of seven methods (not MC): equal to
  round-off for sigma = 1e-2 and 1e-5.
- Meshes: 72 meshes identical (points and triangles); the statistics equal the mesh table.
- Measures: equal to `results.json` (Example 1); Laplace counts equal to the mesh table.
- Solvers: Anderson with safeguard and Newton with continuation as described above.
