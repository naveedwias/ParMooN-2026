#!/usr/bin/env python3
"""
Study of algebraically stabilized P1 methods on layer-adapted meshes with
ParMooN (program cd2d_afc, module AlgebraicStabilization): accuracy, discrete
maximum principle and nonlinear iterations, as in the paper "Algebraically
stabilized finite element methods on layer-adapted meshes" (galerkit-latex,
convection_diffusion/anisotropic_afc).

Cases (example number of the paper, eps): 1 (1e-4, 1e-8), 2 (1e-8), 3 (1e-8);
Examples 4, 5: --cases 4:1e-8 5:1e-5 (with the fixed point iteration only, see
below). Meshes: uniform ("/"), Shishkin "/", "\\", crossed (shishkin_r,
shishkin_l, shishkin_x), and the other layer-adapted meshes with "/"
(bakhvalov_shishkin, bakhvalov, duran); Example 5: uniform, tensor_layer,
polar, polar_layer. Methods: Galerkin, SUPG (ParMooN's parameter), BJK,
Kuzmin (kuzmin_edge: one owner per edge, ties up to round-off to the
lexicographically smaller node), MC, MUAS, SMUAS (67), BBK, F-BBK, S-BBK.
SUPG: SDFEM_TYPE 2, CELL_MEASURE 4 (coth formula with the cell size in the
direction of b; agrees with galerkit's SUPG to 1e-12).
Variants: plain, roff (stabilization off where Pe_E <= 1), loff (off along the
layers, kappa = 10, Example 5: 3), mu2 (relaxed limiters, mu = 2).

Nonlinear solver: the fixed point iteration with updated right-hand side
(A + D) u~ = g + (D - B(u)) u with ParMooN's dynamic damping, started from the
low-order solution (afc_initial_iterate afc_zero), or (--solver) Anderson
acceleration of depth 10 without (fp_anderson) or with the non-monotone safeguard
(fp_anderson_safe, run key suffix _aas) as in the paper; at most --maxit (3000)
iterations, Euclidean residual --tol (1e-10; ParMooN's afc_nonlinloop_epsilon
is set to tol / sqrt(N), because ParMooN multiplies it by sqrt(N)), or Newton's
method (newton; newton_cont: regularized limiters with continuation sigma = 0.1,
..., 1e-6, at most 40 iterations per stage, then Newton on sigma = 0; run key
suffix _nc; not for MC), with a coloured finite-difference Jacobian.

The runs are collected per example in the folder <outdir>/<prefix>, prefix =
cd_<example name> (scripts/naming.py), in a few files (store.py):
<prefix>_runs.jsonl (one line per run: parameters, measures, iterations, status),
<prefix>_runs.csv (the same as a table), <prefix>_failed.log (logs of the runs
that failed or did not converge), <prefix>_solutions.npz (nodal values; --solutions
auto: Examples 4, 5; all; none). Every run is computed in a temporary folder
(<outdir>/.scratch, removed afterwards); only the main process writes the files.
Finished runs (a line in the .jsonl) are skipped; failed ones too, unless
--retry-failed. Different studies: different --outdir.

Output: the runs only. --paper DIR writes all tables and the figure of the
paper under the paper's file names into DIR (and nothing else); --report-only
writes the tables of the given cases into <outdir>/: <prefix>_<study>_accuracy.tex
(maximal nodal error, its order in n between the two largest n, L2 error),
_iterations.tex, _dmp.tex, _mesh.tex, _summary.md, _errors.pdf, and the tables
of report_extra.py.

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/run_study.py [--n 16 32 64 128]
      [--cases 1:1e-4 1:1e-8 2:1e-8 3:1e-8] [--meshes ...] [--methods ...]
      [--variants plain roff loff] [--jobs 4] [--name aniso] [--report-only]
  python3 .../run_study.py --paper paper_results     (all tables of the paper, paper names)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import measures as Q  # noqa: E402
import meshes as M  # noqa: E402
import naming  # noqa: E402
from geo_io import write_unit_square_geo  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402

EXAMPLE = {1: 27, 2: 7, 3: 18, 4: 29, 5: 30}        # paper example -> Example_CD2D
METHODS = ("galerkin", "supg", "bjk", "kuzmin_edge", "mc", "muas", "smuas", "bbk", "fbbk", "sbbk")
LABEL = {"galerkin": "Galerkin", "supg": "SUPG", "bjk": "BJK", "kuzmin_edge": "Kuzmin", "mc": "MC",
         "muas": "MUAS", "smuas": "SMUAS", "bbk": "BBK", "fbbk": "F-BBK", "sbbk": "S-BBK"}
LINEAR = ("galerkin", "supg")
MESHES = ("uniform", "shishkin_r", "shishkin_l", "shishkin_x")
MESH_LABEL = {"uniform": "uniform", "shishkin_r": "Shishkin /", "shishkin_l": "Shishkin \\textbackslash",
              "shishkin_x": "Shishkin $\\times$", "bakhvalov_shishkin": "Bakhvalov--Shishkin",
              "bakhvalov": "Bakhvalov", "duran": "Dur\\'an--Lombardi", "tensor_layer": "tensor layer",
              "polar": "polar", "polar_layer": "polar layer"}

DAT = """[ anisotropic AFC study: Example {ex}, eps = {eps:g}, {mesh}, n = {n}, {method} {variant} ]
outfile: {prefix}.out
boundary_file: Default_UnitSquare
geo_file: {geo}
refinement_n_initial_steps: 0
problem_type: 1
verbosity: 1
example: {example}
diffusion_coefficient: {eps:.17g}
output_write_vtk: false
output_write_ps: false
output_compute_errors: true
output_directory: .
output_basename: {prefix}_
solver_type: direct
direct_solver_type: umfpack
ANSATZ_ORDER: 1
space_discretization_type: {disc}
algebraic_flux_correction: {afc}
afc_limiter: kuzmin
afc_iteration_scheme: fixed_point_rhs
afc_initial_iterate: afc_zero
afc_nonlinloop_maxit: {maxit}
afc_nonlinloop_epsilon: {tol}
afc_nonlinloop_anderson_acc: no
afc_nonlinloop_damping_factor_min: 1e-4
algstab_method: {algstab}
algstab_layer_off: {loff}
algstab_resolved_off: {roff}
algstab_aspect_threshold: {kappa}
algstab_q_relax: {mu}
algstab_solver: {solver}
algstab_reg: {reg}
algstab_newton_maxit: {nc_maxit}
algstab_write_nodal: {prefix}_nodal.txt
algstab_write_edges: {prefix}_edges.txt
"""


def make_mesh(ex, eps, kind, n):
    """points, triangles of the mesh of a case (layer widths as in the paper)"""
    if ex == 5:
        return M.rotating_mesh(kind, n, eps=eps)
    tx, ty, both = Q.layer_widths(ex, eps, n)
    side_y = "interior" if ex == 4 else ("both" if both else "right")
    if kind == "uniform":
        g = np.linspace(0.0, 1.0, n + 1)
        return M.tensor_mesh(g, g, "right")
    if kind.startswith("shishkin"):
        diag = {"shishkin_r": "right", "shishkin_l": "left", "shishkin_x": "crossed"}[kind]
        if ex == 4:                                   # interior layer: layer_points("interior")
            return M.layer_mesh(n, "shishkin", eps, np.sqrt(eps), side_y="interior", diagonal=diag)
        return M.shishkin_mesh(n, tx, ty, side_y=side_y, diagonal=diag)
    # Bakhvalov-Shishkin, Bakhvalov, Duran-Lombardi: layer scales eps/beta or sqrt(eps)
    eb_x = {1: eps, 2: eps / 2, 3: eps, 4: eps}[ex]
    eb_y = {1: None, 2: eps / 3, 3: np.sqrt(eps), 4: np.sqrt(eps)}[ex]
    return M.layer_mesh(n, kind, eb_x, eb_y, side_y=side_y)


SOLVER_SUFFIX = {"fixed_point": "", "fp_anderson": "_aa", "fp_anderson_safe": "_aas", "newton": "_nw",
                 "newton_cont": "_nc"}


def mesh_fits(ex, mesh):
    """polar and tensor_layer meshes only for Example 5, which has no Shishkin-type meshes"""
    if ex == 5:
        return mesh in ("uniform", "polar", "polar_layer", "tensor_layer")
    return mesh not in ("polar", "polar_layer", "tensor_layer")


def acquire_lock(outdir):
    """<outdir>/.lock with the process id; stop if another run step is running in outdir (a lock of a
    process that no longer exists is taken over)"""
    os.makedirs(outdir, exist_ok=True)
    lock = os.path.join(outdir, ".lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().split()[0])
            os.kill(pid, 0)                           # raises if the process does not exist
            sys.exit(f"another run step (process {pid}) is running in {outdir}; wait for it, or stop it "
                     f"(kill {pid}) and start again")
        except (ValueError, IndexError, ProcessLookupError):
            pass                                      # stale lock
        except PermissionError:
            sys.exit(f"{lock} belongs to a running process of another user")
    with open(lock, "w") as f:
        f.write(f"{os.getpid()}\n")
    return lock


def run_key_of(eps, mesh, n, method, variant, solver):
    return run_key(eps, mesh, n, method, variant, solver, "" if method in LINEAR else SOLVER_SUFFIX[solver])


def _tail(text, nlines):
    return "\n".join(text.splitlines()[-nlines:])


def run_job(job):
    """one run in a temporary folder (removed afterwards); returns (prefix, record, solution, log)
    for the main process, which alone writes the collected files (store.py)"""
    exe, outdir, ex, eps, mesh, n, method, variant, maxit, tol, solver, keep_solution = job
    pre = naming.prefix("cd", EXAMPLE[ex])
    key = run_key_of(eps, mesh, n, method, variant, solver)
    base = dict(run=key, example=ex, eps=eps, mesh=mesh, n=n, method=method, variant=variant,
                solver="linear" if method in LINEAR else solver)
    scratch = os.path.join(outdir, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    rdir = tempfile.mkdtemp(prefix=f"{pre}_{key}_", dir=scratch)
    try:
        return _run_job(job, pre, key, base, rdir)
    except Exception as exc:                          # one damaged run must not stop the study
        return pre, dict(base, failed=True, reason=f"{type(exc).__name__}: {exc}"), None, None
    finally:
        shutil.rmtree(rdir, ignore_errors=True)


def _run_job(job, pre, key, base, rdir):
    exe, outdir, ex, eps, mesh, n, method, variant, maxit, tol, solver, keep_solution = job
    try:
        P, T = make_mesh(ex, eps, mesh, n)
    except ValueError as exc:                          # e.g. too few cells for a graded mesh
        return pre, dict(base, failed=True, reason=str(exc)), None, None
    geo = os.path.join(rdir, f"{pre}_mesh.GEO")
    write_unit_square_geo(P, T, geo, f"Example {ex}, {mesh}, n = {n}")
    kappa = 3.0 if ex == 5 else 10.0                 # Example 5: 3 on every mesh (as galerkit)
    txt = DAT.format(ex=ex, eps=eps, mesh=mesh, n=n, method=method, variant=variant, prefix=pre, geo=geo,
                     example=EXAMPLE[ex], disc="supg" if method == "supg" else "galerkin",
                     afc="none" if method in LINEAR else "afc", maxit=maxit,
                     # ParMooN stops at |r|_2 <= afc_nonlinloop_epsilon * sqrt(N): the criterion
                     # |r|_2 <= tol of the paper needs tol / sqrt(N)
                     tol=tol / np.sqrt(len(P)),
                     algstab="none" if method in LINEAR else method,
                     loff=str(variant == "loff").lower(), roff=str(variant == "roff").lower(), kappa=kappa,
                     mu=2.0 if variant == "mu2" else 1.0, solver=solver, reg=1e-6, nc_maxit=40)
    if method == "supg":                              # coth formula with the cell size in the
        # direction of b, one delta per cell with b at the barycentre (= galerkit's SUPG; for
        # constant b the same as per quadrature point, for Example 5 not)
        txt += "SDFEM_TYPE: 2\nCELL_MEASURE: 4\nSUPG_DELTA_CELLWISE: 1\n"
    dat = os.path.join(rdir, f"{pre}_{key}.dat")
    open(dat, "w").write(txt)
    t0 = time.perf_counter()
    with open(os.path.join(rdir, f"{pre}_stdout.log"), "w") as log:
        subprocess.run([exe, os.path.basename(dat)], cwd=rdir, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    logtxt = open(os.path.join(rdir, f"{pre}_stdout.log"), errors="replace").read()
    rec = dict(base, wall_time=wall, **M.mesh_stats(P, T))
    nodal = os.path.join(rdir, f"{pre}_nodal.txt")
    if not os.path.exists(nodal):
        rec.update(failed=True, reason="ParMooN wrote no solution (see the log)")
        return pre, rec, None, _tail(logtxt, 2000)
    Pn, un, fr = Q.read_nodal(nodal)
    u, _ = Q.to_mesh_order(P, Pn, un)
    free = Q.to_mesh_order(P, Pn, fr)[0].astype(bool)
    rec.update(Q.errors(ex, eps, n, P, T, u))
    m = re.findall(r"L2\s*:\s*([0-9.eE+-]+)", logtxt)
    if m and ex in (1, 2):
        rec.update(L2=float(m[-1]))
    # res can be -nan / inf when the iteration blows up (the run then counts as not converged)
    it = re.findall(r"(MAX_)?NONLINEAR ITERATION: ite (\d+) res ([0-9.eE+-]+|-?nan|-?inf) rejections (\d+)", logtxt)
    log = None
    if method not in LINEAR:
        if it:
            rec.update(iterations=int(it[-1][1]), residual=float(it[-1][2]), converged=it[-1][0] == "",
                       rejections=int(it[-1][3]))
        else:
            rec.update(iterations=0, residual=float("nan"), converged=False)
        r0 = re.findall(r"res0 ([0-9.eE+-]+|nan|inf)\)", logtxt)
        if r0:                                        # Newton: residual of the scheme with sigma = 0
            rec.update(residual0=float(r0[-1]))
        if solver == "newton_cont" and it:            # continuation ended with Newton on sigma = 0
            rec.update(polished=", polished" in logtxt)
        so = re.findall(r"\((\d+) switched off\)", logtxt)
        rec.update(n_switched_off=int(so[-1]) if so else 0)
        if not rec["converged"]:
            log = _tail(logtxt, 200)
    edges = os.path.join(rdir, f"{pre}_edges.txt")
    if os.path.exists(edges) and method not in LINEAR:
        E = Q.read_edges(edges)
        rec.update(certificate=Q.dmp_certificate(method, P, free, E))
        rec.update(local_dmp=Q.local_dmp(P, free, E, u))
    rec.update(laplace=Q.laplace_positive(P, T))
    solution = (mesh_key(eps, mesh, n), P, u) if keep_solution else None
    return pre, rec, solution, log


# ----------------------------------------------------------------------------- report
def load(outdir):
    """all runs (store.py: <outdir>/<prefix>/<prefix>_runs.jsonl)"""
    return RunStore(outdir).records()


def get(R, **kw):
    for r in R:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def solved(r):
    """the nonlinear problem of a run is solved: linear methods always; Newton with continuation only if it
    ended with Newton on the original scheme (sigma = 0, 'polished', galerkit's definition); the other solvers
    if they reached the tolerance. Runs of versions without the 'polished' record: their converged flag."""
    if r is None:
        return False
    if r.get("method") in LINEAR or r.get("solver") == "linear":
        return True
    if not r.get("converged"):
        return False
    return not (r.get("solver") == "newton_cont" and r.get("polished") is False)


# caption text for the values of unsolved runs (the solvers return their best iterate, run_study / C++)
UNSOLVED_NOTE = ("values of unsolved runs are those of the iterate the solver returned (fixed point and Anderson: "
                 "the one with the smallest residual; Newton with continuation: the solution of the last "
                 "regularized stage), not of a solution")


def tex_num(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.1f}\\mathrm{{e}}{{{e}}}$"


def mism(s):
    return f"\\mism{{{s}}}"


def report(outdir, study, cases, meshes, methods, ns, solver="fixed_point", write_dir=None):
    """tables from the run records in outdir, written to write_dir (default: outdir)"""
    wdir = write_dir or outdir
    R = [r for r in load(outdir) if r.get("variant", "plain") in ("plain", "roff", "loff", "mu2")
         and r.get("solver", "fixed_point") in (solver, "linear")]
    missing = [f"{ex}:{eps:g}" for ex, eps in cases if not any(r.get("example") == ex and r.get("eps") == eps
                                                              for r in R)]
    if missing:
        print("report: no runs (solver " + solver + ") for the cases " + ", ".join(missing)
              + " - their tables stay empty")
    study = study + SOLVER_SUFFIX[solver]
    n1, n0 = max(ns), (sorted(ns)[-2] if len(ns) > 1 else None)
    acc_meshes = [m for m in ("uniform", "shishkin_r", "shishkin_x") if m in meshes]
    lines_acc, lines_it, lines_dmp, lines_mesh, md = [], [], [], [], []
    md.append(f"# {study}: accuracy, iterations and DMP at n = {n1}\n")
    for ex, eps in cases:
        cap = f"Example~{ex}, $\\varepsilon={eps:g}$".replace("e-0", "e-").replace("e-", "\\cdot10^{-") \
            .replace("$\\varepsilon=1\\cdot10^{-", "$\\varepsilon=10^{-")
        cap = re.sub(r"10\^\{-(\d+)", r"10^{-\1}", cap)
        # accuracy
        key = "Linf_nodes" if ex in (1, 2) else ("Linf_mid" if ex == 4 else ("Linf_r1" if ex == 5 else None))
        l2key = "L2" if ex in (1, 2) else ("L2_away_lumped" if ex == 4 else ("L2_r1_lumped" if ex == 5 else None))
        if key is not None:
            lines_acc += ["\\begin{table}[!htbp]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
                          f"\\caption{{{cap}, $n={n1}$: maximal nodal error, its order in $n$ (between $n={n0}$ "
                          f"and {n1}), and the $L^2$ error (ParMooN); $^*$: nonlinear problem not solved, "
                          f"{UNSOLVED_NOTE}.}}",
                          "\\begin{tabular}{l " + " ".join(["rrr"] * len(acc_meshes)) + "}", "\\toprule",
                          " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MESH_LABEL[m]}}}" for m in acc_meshes)
                          + "\\\\", "method & " + " & ".join(["max & ord. & $L^2$"] * len(acc_meshes)) + "\\\\",
                          "\\midrule"]
            for meth in methods:
                cells = []
                for m in acc_meshes:
                    r1 = get(R, example=ex, eps=eps, mesh=m, n=n1, method=meth, variant="plain")
                    r0 = get(R, example=ex, eps=eps, mesh=m, n=n0, method=meth, variant="plain") if n0 else None
                    if r1 is None or key not in r1:
                        cells += ["--"] * 3
                        continue
                    ordv = (np.log(r0[key] / r1[key]) / np.log(n1 / n0)
                            if r0 and key in r0 and r0[key] > 0 and r1[key] > 0 else None)
                    mark = "" if solved(r1) else "$^*$"
                    cells += [tex_num(r1[key]) + mark, f"{ordv:.2f}" if ordv is not None else "--",
                              tex_num(r1.get(l2key))]
                lines_acc.append(f"{LABEL[meth]} & " + " & ".join(cells) + " \\\\")
            lines_acc += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        md.append(f"\n## Example {ex}, eps = {eps:g}\n")
        md.append("| method | " + " | ".join(f"{m}: max / L2 / it. / cert / local DMP" for m in meshes) + " |")
        md.append("|---" * (len(meshes) + 1) + "|")
        for meth in methods:
            cells = []
            for m in meshes:
                r = get(R, example=ex, eps=eps, mesh=m, n=n1, method=meth, variant="plain")
                if r is None:
                    cells.append("--")
                    continue
                e = r.get(key) if key else None
                it = r.get("iterations")
                c = r.get("certificate", {})
                ld = r.get("local_dmp", {})
                cells.append(f"{e:.1e} / {r.get(l2key, float('nan')):.1e} / "
                             f"{it if it is not None else '-'}{'' if solved(r) else '*'} / "
                             f"{'ok' if c.get('n_bad', 0) == 0 else c.get('n_bad')} / {ld.get('n_viol', '-')}"
                             if e is not None else f"it. {it}")
            md.append(f"| {LABEL[meth]} | " + " | ".join(cells) + " |")
    # iterations and DMP: one table over all cases (uniform, Shishkin /, crossed)
    cols = [(ex, eps, m) for ex, eps in cases for m in acc_meshes]
    head = (" & " + " & ".join(f"\\multicolumn{{{len(acc_meshes)}}}{{c}}{{Example~{ex}, $\\varepsilon={eps:g}$}}"
                               for ex, eps in cases) + "\\\\",
            "method & " + " & ".join(["u & / & $\\times$"[: 3 * len(acc_meshes) + 6]] * len(cases)) + "\\\\")
    for kind, lines in (("it", lines_it), ("dmp", lines_dmp)):
        cap = (f"Iterations of the fixed point iteration at $n={n1}$ (u: uniform, /: Shishkin /, $\\times$: "
               f"Shishkin $\\times$); \\mism{{colour}}: not converged in 3000 iterations." if kind == "it" else
               f"DMP at $n={n1}$: ``ok'' if the sufficient mesh condition of the method holds on every edge, "
               "otherwise the number of edges where it fails (\\mism{colour}).")
        lines += ["\\begin{table}[!htbp]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
                  f"\\caption{{{cap}}}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l " +
                  " ".join(["r" * len(acc_meshes)] * len(cases)) + "}", "\\toprule", head[0], head[1], "\\midrule"]
        for meth in [m for m in methods if m not in LINEAR]:
            cells = []
            for ex, eps, m in cols:
                r = get(R, example=ex, eps=eps, mesh=m, n=n1, method=meth, variant="plain")
                if r is None:
                    cells.append("--")
                elif kind == "it":
                    cells.append(str(r.get("iterations")) if solved(r) else mism(r.get("iterations")))
                else:
                    nb = r.get("certificate", {}).get("n_bad", 0)
                    cells.append("ok" if nb == 0 else mism(nb))
            lines.append(f"{LABEL[meth]} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    # meshes
    lines_mesh += ["\\begin{table}[!htbp]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
                   f"\\caption{{The meshes at $n={n1}$: nodes, largest aspect ratio, shortest edge, largest angle, "
                   "and the edges with a positive off-diagonal entry of the $P_1$ Laplace matrix.}",
                   "\\begin{tabular}{llrrrrr}", "\\toprule",
                   "case & mesh & nodes & aspect & $h_{\\min}$ & max angle & Laplace $>0$\\\\", "\\midrule"]
    for ex, eps in cases:
        for m in meshes:
            r = next((r for r in R if r["example"] == ex and r["eps"] == eps and r["mesh"] == m and r["n"] == n1),
                     None)
            if r:
                lines_mesh.append(f"Example~{ex}, $\\varepsilon={eps:g}$ & {MESH_LABEL[m]} & {r['nodes']} & "
                                  f"{tex_num(r['max_aspect'])} & {tex_num(r['h_min'])} & {r['max_angle']:.1f} & "
                                  f"{r['laplace']['n_positive']} \\\\")
    lines_mesh += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    pre = "cd_aniso"
    for suffix, lines in (("accuracy", lines_acc), ("iterations", lines_it), ("dmp", lines_dmp),
                          ("mesh", lines_mesh)):
        f = os.path.join(wdir, f"{pre}_{study}_{suffix}.tex")
        open(f, "w").write("% generated by scripts/aniso_afc/run_study.py (ParMooN)\n" + "\n".join(lines) + "\n")
        print(f"wrote {f}")
    f = os.path.join(wdir, f"{pre}_{study}_summary.md")
    open(f, "w").write("\n".join(md) + "\n")
    print(f"wrote {f}")
    # figure: L2 error against N (Examples 1 and 2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ex12 = [(ex, eps) for ex, eps in cases if ex in (1, 2)]
        if ex12:
            fig, axs = plt.subplots(len(ex12), len(acc_meshes), figsize=(4 * len(acc_meshes), 3.2 * len(ex12)),
                                    squeeze=False)
            for a, (ex, eps) in enumerate(ex12):
                for b, m in enumerate(acc_meshes):
                    ax = axs[a][b]
                    for meth in methods:
                        pts = sorted((r["nodes"], r.get("L2")) for r in R if r["example"] == ex and r["eps"] == eps
                                     and r["mesh"] == m and r["method"] == meth and r.get("variant") == "plain"
                                     and r.get("L2") is not None)
                        if pts:
                            x, y = zip(*pts)
                            ax.loglog(x, y, "--" if meth == "supg" else "o-", ms=3, lw=1, label=LABEL[meth])
                    ax.set_title(f"Example {ex}, eps = {eps:g}, {m}", fontsize=8)
                    ax.set_xlabel("N (nodes)")
                    ax.grid(True, alpha=0.3)
            axs[0][0].set_ylabel("L2 error")
            axs[0][-1].legend(fontsize=6)
            fig.tight_layout()
            f = os.path.join(wdir, f"{pre}_{study}_errors.pdf")
            fig.savefig(f)
            print(f"wrote {f}")
    except Exception as exc:                          # figures are optional
        print("no figure:", exc)


# the paper: the four reports and the files of the paper they give (paper name <- report file)
PAPER_REPORTS = [
    ("ex123", ["1:1e-4", "1:1e-8", "2:1e-8", "3:1e-8"], [16, 32, 64, 128],
     ["uniform", "shishkin_r", "shishkin_l", "shishkin_x"]),
    ("ex3_n32", ["3:1e-8"], [32], ["uniform", "shishkin_r", "shishkin_x"]),
    ("ex4", ["4:1e-8"], [16, 32, 64], ["uniform", "shishkin_r", "bakhvalov_shishkin", "bakhvalov"]),
    ("ex5", ["5:1e-5"], [32, 64], ["uniform", "polar", "polar_layer", "tensor_layer"]),
]
PAPER_FILES = [
    ("aniso_accuracy.tex", "ex123_accuracy.tex", "step 3"),
    ("aniso_dmp.tex", "ex123_dmp.tex", "step 3"),
    ("aniso_efficiency.tex", "ex123_iterations.tex", "step 3"),
    ("aniso_mesh.tex", "ex123_mesh.tex", "step 3"),
    ("aniso_roff.tex", "ex123_switch_roff.tex", "step 4"),
    ("aniso_layermesh_it.tex", "ex123_layermesh_it.tex", "step 5"),
    ("aniso_anderson.tex", "ex123_solvers.tex", "steps 6, 7"),
    ("aniso_newton_all.tex", "ex123_solved.tex", "step 7"),
    ("aniso_interior.tex", "ex4_ex4.tex", "step 9"),
]


def paper_report(outdir, paper_dir, methods):
    """all tables and figures of the paper from the runs in outdir, under the paper's file names in
    paper_dir (and nothing else)"""
    import shutil
    import tempfile
    from report_extra import report_extra
    tmp = tempfile.mkdtemp(prefix="aniso_paper_")
    for name, cases, ns, meshes in PAPER_REPORTS:
        cs = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in cases]
        report(outdir, name, cs, meshes, methods, ns, "fixed_point", write_dir=tmp)
        report_extra(outdir, name, cs, methods, ns, write_dir=tmp)
    os.makedirs(paper_dir, exist_ok=True)
    print(f"\nresults of the paper in {paper_dir}/:")
    missing = []
    for paper_name, report_name, step in PAPER_FILES:
        src = os.path.join(tmp, f"cd_aniso_{report_name}")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(paper_dir, paper_name))
            print(f"  {paper_name}")
        else:
            missing.append((paper_name, step))
    # three tables in galerkit's layout (paper_tables.py)
    try:
        from paper_tables import make_all as make_tables
        written = {os.path.basename(p) for p in make_tables(outdir, paper_dir, quiet=True)}
    except Exception as exc:
        print(f"  tables of paper_tables.py: {exc}")
        written = set()
    for name, step in (("aniso_layermesh_acc.tex", "step 5"), ("aniso_newton_reg.tex", "step 8"),
                       ("aniso_rotating.tex", "step 10")):
        if name in written:
            print(f"  {name}")
        else:
            missing.append((name, step))
    # figures of Examples 1-4 in galerkit's layout (paper_figures.py)
    try:
        from paper_figures import make_all
        written = {os.path.basename(p) for p in make_all(outdir, paper_dir, quiet=True)}
    except Exception as exc:                          # figures are optional
        print(f"  figures of Examples 1-4: {exc}")
        written = set()
    for name, step in (("aniso_meshes.pdf", ""), ("aniso_meshes.png", ""),
                       ("aniso_errors.pdf", "step 3"), ("aniso_errors.png", "step 3"),
                       ("aniso_iterations.pdf", "step 3"), ("aniso_iterations.png", "step 3"),
                       ("interior_cut_shishkin_n64.png", "step 9"), ("interior_cut_uniform_n64.png", "step 9"),
                       ("interior_solutions_shishkin_n64.png", "step 9"), ("interior_errors.png", "step 9")):
        if name in written:
            print(f"  {name}")
        else:
            missing.append((name, (step or "matplotlib") + ", or paper_figures.py --run"))
    # Example 5 at n = 64: meshes and solutions, outlet profiles (rotating_figure.py; the VTK files
    # and the summaries with its own call)
    try:
        from rotating_figure import MESHES as ROT_MESHES, make_errors_figure, make_figure, make_outlet_figures
        pngs = [make_figure(outdir, paper_dir, vtkdir=None, quiet=True)]
        pngs += make_outlet_figures(outdir, paper_dir, mddir=None, quiet=True)
        pngs += [make_errors_figure(outdir, paper_dir, mddir=None, quiet=True)]
        pngs = {os.path.basename(p) for p in pngs if p}
        expected = (["rotating_solutions_n64.png"] + [f"rotating_outlet_{m}_n64.png" for m in ROT_MESHES]
                    + ["rotating_errors.png"])
    except Exception as exc:                          # figures are optional
        print(f"  Example 5 figures: {exc}")
        pngs, expected = set(), []
    for name in expected:
        if name in pngs:
            print(f"  {name}")
        else:
            missing.append((name, "step 10, or rotating_figure.py --run"))
    for paper_name, step in missing:
        print(f"  missing: {paper_name} (no runs yet: {step} of documentation/ANISO_AFC_RUNS.md)")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default="./parMooN_cd2d_afc_SEQUENTIAL")
    ap.add_argument("--outdir", default="cd_aniso")
    ap.add_argument("--name", default="aniso", help="study name (part of the folder and file names)")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--cases", nargs="+", default=["1:1e-4", "1:1e-8", "2:1e-8", "3:1e-8"])
    ap.add_argument("--meshes", nargs="+", default=list(MESHES))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--variants", nargs="+", default=["plain"])
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--solver", default="fixed_point", choices=list(SOLVER_SUFFIX),
                    help="fixed_point (ParMooN's loop with damping), fp_anderson, fp_anderson_safe, newton, "
                         "newton_cont (regularized limiters, sigma = 0.1 ... 1e-6, 40 iterations per stage, "
                         "then sigma = 0; not for MC)")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--solutions", default="auto", choices=["auto", "all", "none"],
                    help="keep the nodal values in <prefix>_solutions.npz: auto (Examples 4, 5: their figures), "
                         "all, none")
    ap.add_argument("--retry-failed", action="store_true", help="repeat the runs that failed (default: skipped)")
    ap.add_argument("--redo", action="store_true",
                    help="recompute the selected runs even if finished (the new record and solution count; "
                         "e.g. the SUPG runs of Example 5 after SUPG_DELTA_CELLWISE)")
    ap.add_argument("--paper", metavar="DIR", help="no runs: all tables and the figure of the paper from the "
                    "existing runs, under the paper's file names in DIR (e.g. --paper paper_results)")
    args = ap.parse_args()
    cases = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cases]
    if args.paper:
        RunStore(args.outdir).write_csv(prefixes=[os.path.basename(os.path.dirname(p)) for p in
                                                  glob.glob(os.path.join(args.outdir, "*", "*_runs.jsonl"))])
        paper_report(args.outdir, args.paper, args.methods)
        return
    os.makedirs(args.outdir, exist_ok=True)
    if not args.report_only:
        exe = os.path.abspath(args.exe)
        outdir = os.path.abspath(args.outdir)
        store = RunStore(outdir)
        lock = acquire_lock(outdir)                   # one run step per outdir at a time
        # leftovers of an interrupted run step
        shutil.rmtree(os.path.join(outdir, ".scratch"), ignore_errors=True)
        done = store.done(retry_failed=args.retry_failed)
        keep = {"all": lambda ex: True, "none": lambda ex: False, "auto": lambda ex: ex in (4, 5)}[args.solutions]
        jobs, skipped = [], 0
        for ex, eps in cases:
            for m in args.meshes:
                if not mesh_fits(ex, m):
                    continue
                for n in args.n:
                    for meth in args.methods:
                        if meth == "mc" and args.solver == "newton_cont":   # MC: no regularization
                            continue
                        for var in args.variants:
                            if meth in LINEAR and var != "plain":
                                continue
                            key = run_key_of(eps, m, n, meth, var, args.solver)
                            if not args.redo and (naming.prefix("cd", EXAMPLE[ex]), key) in done:
                                skipped += 1
                                continue
                            jobs.append((exe, outdir, ex, eps, m, n, meth, var, args.maxit, args.tol, args.solver,
                                         keep(ex)))
        print(f"{len(jobs)} runs ({skipped} finished runs skipped)")
        t0 = time.perf_counter()
        pool = Pool(args.jobs) if args.jobs > 1 else None
        results = pool.imap_unordered(run_job, jobs) if pool else map(run_job, jobs)
        for k, (pre, rec, solution, log) in enumerate(results, 1):
            store.add(pre, rec, solution, log)                # only the main process writes
            if rec.get("failed"):
                print(f"  run failed ({pre} {rec['run']}): {rec.get('reason')}")
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)")
        if pool:
            pool.close()
            pool.join()
        store.write_csv()
        try:
            os.rmdir(os.path.join(outdir, ".scratch"))
        except OSError:
            pass
        os.remove(lock)
    if not args.report_only:                          # runs only: no interim tables in outdir
        print(f"runs in {args.outdir}/; the tables and the figure of the paper: "
              f"python3 {os.path.relpath(__file__)} --paper paper_results")
        return
    report(args.outdir, args.name, cases, [m for m in args.meshes if any(mesh_fits(ex, m) for ex, _ in cases)],
           args.methods, args.n, args.solver)
    from report_extra import report_extra
    report_extra(args.outdir, args.name, cases, args.methods, args.n)


if __name__ == "__main__":
    main()
