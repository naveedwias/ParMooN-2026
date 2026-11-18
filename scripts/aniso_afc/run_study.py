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
layers, kappa = 10, polar meshes 3), mu2 (relaxed limiters, mu = 2).

Nonlinear solver: the fixed point iteration with updated right-hand side
(A + D) u~ = g + (D - B(u)) u with ParMooN's dynamic damping, started from the
low-order solution (afc_initial_iterate afc_zero), or (--solver) Anderson
acceleration of depth 10 without (fp_anderson) or with the non-monotone safeguard
(fp_anderson_safe, folder suffix _aas) as in the paper; at most --maxit (3000)
iterations, Euclidean residual --tol (1e-10; ParMooN's afc_nonlinloop_epsilon
is set to tol / sqrt(N), because ParMooN multiplies it by sqrt(N)), or Newton's
method (newton; newton_cont: regularized limiters with continuation sigma = 0.1,
..., 1e-6, at most 40 iterations per stage, then Newton on sigma = 0; folder
suffix _nc; not for MC), with a coloured finite-difference Jacobian.

Every run is a folder <prefix>_<study>_<mesh>_n<n>_<method>[_<variant>] in the
folder <prefix> = cd_<example name> (scripts/naming.py), with the ParMooN input,
log, nodal values (17 digits), edges, and <prefix>_record.json (measures:
measures.py). Finished runs (with a record) are skipped.

Output (per case, in <outdir>/): <prefix>_<study>_accuracy.tex (maximal nodal
error, its order in n between the two largest n, L2 error), _iterations.tex,
_dmp.tex, _mesh.tex, _summary.md, _errors.pdf (L2 error against N).

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/run_study.py [--n 16 32 64 128]
      [--cases 1:1e-4 1:1e-8 2:1e-8 3:1e-8] [--meshes ...] [--methods ...]
      [--variants plain roff loff] [--jobs 4] [--name aniso] [--report-only]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
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


def run_name(study, eps, mesh, n, method, variant, solver="fixed_point"):
    # eps is part of the name: Example 1 is run with two values of eps
    return (f"{study}_eps{eps:.0e}_{mesh}_n{n}_{method}" + ("" if variant == "plain" else f"_{variant}")
            + ("" if method in LINEAR else SOLVER_SUFFIX[solver]))


def run_job(job):
    try:
        return _run_job(job)
    except Exception as exc:                          # one damaged run must not stop the study
        exe, outdir, study, ex, eps, mesh, n, method, variant, maxit, tol, solver = job
        pre = naming.prefix("cd", EXAMPLE[ex])
        rdir = os.path.join(outdir, f"{pre}_{run_name(study, eps, mesh, n, method, variant, solver)}")
        os.makedirs(rdir, exist_ok=True)
        rec_file = os.path.join(rdir, f"{pre}_record.json")
        json.dump(dict(example=ex, eps=eps, mesh=mesh, n=n, method=method, variant=variant,
                       solver="linear" if method in LINEAR else solver, failed=True,
                       reason=f"{type(exc).__name__}: {exc}"), open(rec_file, "w"), indent=1)
        print(f"  run failed ({os.path.basename(rdir)}): {type(exc).__name__}: {exc}")
        return rec_file


def _run_job(job):
    exe, outdir, study, ex, eps, mesh, n, method, variant, maxit, tol, solver = job
    pre = naming.prefix("cd", EXAMPLE[ex])
    rdir = os.path.join(outdir, f"{pre}_{run_name(study, eps, mesh, n, method, variant, solver)}")
    rec_file = os.path.join(rdir, f"{pre}_record.json")
    if os.path.exists(rec_file):
        return rec_file
    os.makedirs(rdir, exist_ok=True)
    try:
        P, T = make_mesh(ex, eps, mesh, n)
    except ValueError as exc:                          # e.g. too few cells for a graded mesh
        json.dump(dict(example=ex, eps=eps, mesh=mesh, n=n, method=method, variant=variant,
                       solver="linear" if method in LINEAR else solver, failed=True, reason=str(exc)),
                  open(rec_file, "w"), indent=1)
        return rec_file
    geo = os.path.join(rdir, f"{pre}_mesh.GEO")
    write_unit_square_geo(P, T, geo, f"Example {ex}, {mesh}, n = {n}")
    kappa = 3.0 if mesh.startswith("polar") else 10.0
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
        txt += "SDFEM_TYPE: 2\nCELL_MEASURE: 4\n"       # direction of b (= galerkit's SUPG)
    dat = os.path.join(rdir, f"{pre}_{run_name(study, eps, mesh, n, method, variant, solver)}.dat")
    open(dat, "w").write(txt)
    t0 = time.perf_counter()
    with open(os.path.join(rdir, f"{pre}_stdout.log"), "w") as log:
        subprocess.run([exe, os.path.basename(dat)], cwd=rdir, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    logtxt = open(os.path.join(rdir, f"{pre}_stdout.log"), errors="replace").read()
    rec = dict(example=ex, eps=eps, mesh=mesh, n=n, method=method, variant=variant,
               solver="linear" if method in LINEAR else solver, wall_time=wall,
               **M.mesh_stats(P, T))
    nodal = os.path.join(rdir, f"{pre}_nodal.txt")
    if not os.path.exists(nodal):
        rec.update(failed=True)
        json.dump(rec, open(rec_file, "w"), indent=1)
        return rec_file
    Pn, un, fr = Q.read_nodal(nodal)
    u, _ = Q.to_mesh_order(P, Pn, un)
    free = Q.to_mesh_order(P, Pn, fr)[0].astype(bool)
    rec.update(Q.errors(ex, eps, n, P, T, u))
    m = re.findall(r"L2\s*:\s*([0-9.eE+-]+)", logtxt)
    if m and ex in (1, 2):
        rec.update(L2=float(m[-1]))
    it = re.findall(r"(MAX_)?NONLINEAR ITERATION: ite (\d+) res ([0-9.eE+-]+) rejections (\d+)", logtxt)
    if method not in LINEAR:
        if it:
            rec.update(iterations=int(it[-1][1]), residual=float(it[-1][2]), converged=it[-1][0] == "",
                       rejections=int(it[-1][3]))
        else:
            rec.update(iterations=0, residual=float("nan"), converged=False)
        so = re.findall(r"\((\d+) switched off\)", logtxt)
        rec.update(n_switched_off=int(so[-1]) if so else 0)
    edges = os.path.join(rdir, f"{pre}_edges.txt")
    if os.path.exists(edges) and method not in LINEAR:
        E = Q.read_edges(edges)
        rec.update(certificate=Q.dmp_certificate(method, P, free, E))
        rec.update(local_dmp=Q.local_dmp(P, free, E, u))
    rec.update(laplace=Q.laplace_positive(P, T))
    json.dump(rec, open(rec_file, "w"), indent=1)
    return rec_file


# ----------------------------------------------------------------------------- report
def load(outdir):
    out = []
    for f in glob.glob(os.path.join(outdir, "*", "*_record.json")):
        out.append(json.load(open(f)))
    return out


def get(R, **kw):
    for r in R:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def tex_num(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10 ** e:.1f}\\mathrm{{e}}{{{e}}}$"


def mism(s):
    return f"\\mism{{{s}}}"


def report(outdir, study, cases, meshes, methods, ns, solver="fixed_point"):
    R = [r for r in load(outdir) if r.get("variant", "plain") in ("plain", "roff", "loff", "mu2")
         and r.get("solver", "fixed_point") in (solver, "linear")]
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
                          f"and {n1}), and the $L^2$ error (ParMooN).}}",
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
                    mark = "" if r1.get("converged", True) else "$^*$"
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
                             f"{it if it is not None else '-'}{'' if r.get('converged', True) else '*'} / "
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
               f"Shishkin $\\times$); \\mism{{colour}}: not converged." if kind == "it" else
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
                    cells.append(str(r.get("iterations")) if r.get("converged") else mism(r.get("iterations")))
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
        f = os.path.join(outdir, f"{pre}_{study}_{suffix}.tex")
        open(f, "w").write("% generated by scripts/aniso_afc/run_study.py (ParMooN)\n" + "\n".join(lines) + "\n")
        print(f"wrote {f}")
    f = os.path.join(outdir, f"{pre}_{study}_summary.md")
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
            f = os.path.join(outdir, f"{pre}_{study}_errors.pdf")
            fig.savefig(f)
            print(f"wrote {f}")
    except Exception as exc:                          # figures are optional
        print("no figure:", exc)


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
    args = ap.parse_args()
    cases = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in args.cases]
    os.makedirs(args.outdir, exist_ok=True)
    if not args.report_only:
        exe = os.path.abspath(args.exe)
        jobs = [(exe, os.path.abspath(args.outdir), args.name, ex, eps, m, n, meth, var, args.maxit, args.tol,
                 args.solver)
                for ex, eps in cases for m in args.meshes for n in args.n for meth in args.methods
                for var in args.variants
                if not (meth in LINEAR and var != "plain")
                and mesh_fits(ex, m)
                and not (meth == "mc" and args.solver == "newton_cont")]    # MC: no regularization
        print(f"{len(jobs)} runs")
        t0 = time.perf_counter()
        if args.jobs > 1:
            with Pool(args.jobs) as pool:
                for k, _ in enumerate(pool.imap_unordered(run_job, jobs), 1):
                    if k % 20 == 0:
                        print(f"  {k}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)")
        else:
            for k, job in enumerate(jobs, 1):
                run_job(job)
                if k % 20 == 0:
                    print(f"  {k}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)")
    report(args.outdir, args.name, cases, [m for m in args.meshes if any(mesh_fits(ex, m) for ex, _ in cases)],
           args.methods, args.n, args.solver)
    from report_extra import report_extra
    report_extra(args.outdir, args.name, cases, args.methods, args.n)


if __name__ == "__main__":
    main()
