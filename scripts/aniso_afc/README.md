# Algebraic stabilizations on layer-adapted meshes (ParMooN)

Tools for the study of the paper *Algebraically stabilized finite element methods on
layer-adapted meshes: accuracy, discrete maximum principle and nonlinear solvers*
(galerkit-latex, `convection_diffusion/anisotropic_afc`), with the ParMooN program
`cd2d_afc` and the module `AlgebraicStabilization`
(`src/AssembleRoutines/AlgebraicStabilization.C`).

## Methods (`algstab_method`)

| paper | ParMooN | notes |
|---|---|---|
| AFC, BJK | `bjk` | equal to `afc_limiter: BJK17` of AlgebraicFluxCorrection |
| AFC, Kuzmin | `kuzmin_edge` | one owner per edge; a_ij = a_ji up to round-off (`algstab_tie_tol`) goes to the lexicographically smaller node. `afc_limiter: kuzmin` limits such edges at both nodes (B(u) not symmetric) |
| AFC, MC | `mc` | equal to `afc_limiter: monolithic` |
| MUAS | `muas` | equal to `afc_limiter: MUAS_MAX`; `afc_limiter: MUAS` is a different variant (Q_i with \|a_ij\| instead of max{\|a_ij\|, a_ji}) |
| SMUAS | `smuas` | weights (67) or (68) (`algstab_smuas_weights`); new |
| BBK | `bbk` | b_ij = d_ij max{xi_i, xi_j}^p; new (`afc_limiter: BBK` is an S-BBK-type edge form) |
| F-BBK, S-BBK | `fbbk`, `sbbk` | edge weights C0 \|E\|, weights beta_ij of ABR17 (QP per node); new |

Options for all methods: `algstab_resolved_off` (Pe_E <= `algstab_pe_threshold`),
`algstab_layer_off` (long edges of cells with aspect ratio >= `algstab_aspect_threshold`),
`algstab_q_relax` (relaxed limiters, mu). Solver: `afc_iteration_scheme: fixed_point_rhs`.
Output: `algstab_write_nodal` (x y u free, 17 digits), `algstab_write_edges` (a_ij, a_ji,
d_ij, switch), `algstab_test_b` (b_ij(u) for a given u).

Note: ParMooN stops the AFC iteration at |r|_2 <= `afc_nonlinloop_epsilon` * sqrt(N); for
the criterion |r|_2 <= tol of the paper use tol / sqrt(N) (`run_study.py` does).

## Examples (`Example_CD2D`)

| paper | example | | paper | example |
|---|---|---|---|---|
| 1 exponential layer | 27 | | 4 straight interior layer | 29 |
| 2 boundary layers | 7 | | 5 profile along circles | 30 |
| 3 parabolic layers | 18 | | cross-check problem (b = (2,3), c = 1) | 28 |

## Scripts

- `meshes.py`: Shishkin (right, left, both ends), Bakhvalov-Shishkin, Bakhvalov,
  Duran-Lombardi, splittings `/`, `\`, crossed; the meshes of Example 5 (polar, polar_layer,
  tensor_layer). `python3 meshes.py` compares 72 meshes with galerkit (identical).
- `geo_io.py`: points and triangles -> ParMooN `.GEO` (Default_UnitSquare).
- `measures.py`: errors, reference solutions (erfc; radial heat equation), DMP certificate,
  local DMP, Laplace > 0 edges.
- `run_study.py`: the study (runs, records, LaTeX tables, figure); see its docstring.
- `crosscheck_galerkit.py`, `test_b_galerkit.py`: comparisons with galerkit (solutions;
  b_ij(u) edge by edge). Require galerkit (`pip install -e <galerkit>`).

## Verification (with galerkit, branch feature/anisotropic-afc)

- Galerkin: equal to round-off on uniform, Shishkin, crossed and polar meshes (Examples 1, 3,
  4, 5, 28); SUPG with `SDFEM_TYPE 2`, `CELL_MEASURE 4`: 1e-12.
- b_ij(u) of all eight methods: equal to round-off (fbbk: 1e-11 ... 2e-9 relative, the
  accuracy of the weight QP) for random u and the solution, also with the switches, mu = 2,
  and at Neumann nodes (Example 5).
- Measures: maximal errors, errors away from the layers, DMP certificate and local DMP equal
  to galerkit's results.json (Example 1); Laplace > 0 counts and mesh data equal to the mesh
  table of the paper.

## Nonlinear solvers (`algstab_solver`)

- `fixed_point`: ParMooN's fixed point rhs loop with its dynamic damping.
- `fp_anderson`, `fp_anderson_safe`: Anderson acceleration of depth 10, with the non-monotone
  safeguard of the paper (galerkit `_fp_anderson`). Against galerkit: the same convergence,
  iterations within a few percent, the same numbers of safeguard rejections.
- `newton`, `newton_cont`: Newton's method with a coloured finite-difference Jacobian
  (`algstab_fd_step` 1e-10; galerkit uses exact Jacobians), backtracking and a damped Picard
  fallback; `newton_cont`: regularized limiters (`algstab_reg` = 1e-6, sigma = 0.1, ..., 1e-6,
  `algstab_newton_maxit` = 40 per stage), then Newton on sigma = 0. The regularized b_ij(u)
  of all seven methods equal galerkit's to round-off; the continuation stages equal galerkit's
  (polish at sigma = 0: a few iterations more). MC has no regularization.
  The smoothed max/min of the regularization are not symmetric, so the regularized scheme
  depends on the orientation of the edges at O(sigma); the module orients every edge from the
  lexicographically smaller node (as galerkit's numbering of its tensor-product meshes).

## Tables

`python3 run_study.py --paper paper_results` writes all tables and the figure of the paper,
under the paper's file names, from the finished runs (see
`documentation/ANISO_AFC_RUNS.md`). `--report-only` writes the tables of given cases: the
accuracy, iteration, DMP and mesh tables (`run_study.py`) and the tables for the switches, the
layer-adapted meshes, all solvers, the solved runs and Examples 4 and 5 (`report_extra.py`),
in the formats of the paper. The runs of a full study, e.g.:

    S=../ParMooN-2026/scripts/aniso_afc/run_study.py
    python3 $S --jobs 4                                     # Examples 1-3, fixed point
    python3 $S --jobs 4 --variants plain roff loff          # switches
    python3 $S --jobs 4 --solver fp_anderson_safe
    python3 $S --jobs 4 --solver newton_cont --variants plain loff
    python3 $S --jobs 4 --meshes uniform shishkin_r bakhvalov_shishkin bakhvalov duran \
                --variants plain loff                       # layer-adapted meshes
    python3 $S --jobs 4 --cases 4:1e-8 --n 16 32 64 --meshes uniform shishkin_r \
                bakhvalov_shishkin bakhvalov --variants plain loff --solver newton_cont
    python3 $S --jobs 4 --cases 5:1e-5 --n 32 64 --meshes uniform polar polar_layer \
                tensor_layer --variants plain loff --solver newton_cont

    python3 $S --paper paper_results                       # the tables of the paper

Finished runs are skipped, so the commands can be repeated and combined.

## Regression

`python3 regression.py --exe <build>/parMooN_cd2d_afc_SEQUENTIAL`: 18 small cases (all
methods with the fixed point iteration, Anderson with safeguard, Newton continuation, the
layer switch on a crossed mesh, Newton on Example 2) against `regression_reference.json`
(values verified against galerkit).
