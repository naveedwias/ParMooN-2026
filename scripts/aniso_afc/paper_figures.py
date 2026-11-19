#!/usr/bin/env python3
"""
The figures of the paper for Examples 1-4, made from ParMooN's runs (run_study.py) in the layout of
galerkit's make_report.py and interior_layer.py (galerkit branch feature/anisotropic-afc).
(Example 5: rotating_figure.py.)

  meshes     aniso_meshes.{pdf,png}: the four mesh types (uniform, Shishkin "/", "\\", crossed) of
             Example 3 at n = 16, eps = 1e-2 (only to make the layer cells visible); no runs needed.
  errors     aniso_errors.{pdf,png}: L2 error against n of the eight algebraic methods and SUPG,
             Example 1 (eps = 1e-4, 1e-8) and Example 2 (eps = 1e-8) on the uniform, Shishkin "/" and
             crossed meshes; dotted: (N^-1 ln N)^2. Fixed point runs (step 3 of the run guide); where the
             fixed point iteration did not converge, the run of another solver (fp_anderson_safe,
             newton_cont; steps 6, 7) with a smaller residual of the scheme, if there is one (galerkit:
             a Newton solve as fallback).
  iterations aniso_iterations.{pdf,png}: fixed point iterations against n, Examples 1-3 with eps = 1e-8,
             same meshes; large open markers: not converged (3000 iterations).
  interior   Example 4 (eps = 1e-8, n = 16, 32, 64; step 9 of the run guide), solver per method as galerkit:
             the first of fp_anderson_safe, newton_cont (only if it ends at sigma = 0), fixed_point that
             converged with a residual of the scheme <= 1e-8:
             interior_cut_shishkin_n64.png, interior_cut_uniform_n64.png: u along the node column closest
               to x = 1/2 across the interior layer, the ten methods (+ limiter off along the layers)
               against the reference 1/2 erfc((1/2 - y) / (2 sqrt(eps x)));
             interior_solutions_shishkin_n64.png: the solutions in the layer region on the Shishkin mesh;
             interior_errors.png: maximal nodal error for 0.1 <= x <= 1 - tau_x against n, per mesh.

--run computes the missing runs first (run_study.py; finished runs are skipped): for errors and
iterations the fixed point runs of Examples 1-3 (n = 16 ... 128: hours), for interior the runs of
Example 4 solver by solver. Without --run nothing is computed.

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/paper_figures.py                       # all, from existing runs
  python3 ../ParMooN-2026/scripts/aniso_afc/paper_figures.py --figures interior --run --jobs 4
Output: <figdir> (default paper_results), and <mddir>/interior_n64.md (default mddir paper_figures_runs: the
runs used in the interior figures, solver, residual, solved).
"""
import argparse
import os
import subprocess
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import measures as Q  # noqa: E402
import naming  # noqa: E402
from run_study import EXAMPLE, make_mesh  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402

FIGURES = ("meshes", "errors", "iterations", "interior")
AFC = ("bjk", "kuzmin_edge", "mc", "muas", "smuas", "bbk", "fbbk", "sbbk")
LINEAR = ("galerkin", "supg")
ALL = ("galerkin", "supg") + AFC
NAME = {"galerkin": "Galerkin", "supg": "SUPG", "bjk": "BJK", "kuzmin_edge": "Kuzmin", "mc": "MC", "muas": "MUAS",
        "smuas": "SMUAS", "bbk": "BBK", "fbbk": "F-BBK", "sbbk": "S-BBK"}
MESHP = {"uniform": "uniform", "shishkin_r": "Shishkin /", "shishkin_l": "Shishkin \\", "shishkin_x": "Shishkin ×"}
PROBLEM = {ex: naming.STEADY_CD[EXAMPLE[ex]] for ex in EXAMPLE}      # exp_layer, boundary_layers, ...
SUFFIX = {"fixed_point": "", "fp_anderson": "_aa", "fp_anderson_safe": "_aas", "newton": "_nw", "newton_cont": "_nc"}
NS = (16, 32, 64, 128)
MESHES3 = ("uniform", "shishkin_r", "shishkin_x")
# galerkit make_report.py style
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MRK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
# Example 4 (galerkit interior_layer.py)
EPS4 = 1e-8
NS4 = (16, 32, 64)
MESHES4 = ("uniform", "shishkin_r", "bakhvalov_shishkin", "bakhvalov")
MNAME4 = {"uniform": "uniform", "shishkin_r": "Shishkin", "bakhvalov_shishkin": "Bakhvalov-Shishkin",
          "bakhvalov": "Bakhvalov"}
FILE4 = {"uniform": "uniform", "shishkin_r": "shishkin"}           # galerkit's file names
SOLVERS4 = ("fp_anderson_safe", "newton_cont", "fixed_point")
SOLVED_TOL = 1e-8


def _plt(style="report"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    if style == "report":
        plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                             "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                             "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                             "lines.markersize": 4.5, "legend.frameon": False, "legend.fontsize": 7.5,
                             "savefig.bbox": "tight", "pdf.fonttype": 42})
    else:
        plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                             "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})
    return plt


class Runs:
    def __init__(self, outdir):
        self.outdir = outdir
        self.by = {}
        for r in RunStore(outdir).records():
            self.by[r["prefix"], r["run"]] = r
        self._npz = {}

    def get(self, ex, eps, mesh, n, method, variant="plain", solver="fixed_point"):
        key = run_key(eps, mesh, n, method, variant, solver, "" if method in LINEAR else SUFFIX[solver])
        r = self.by.get((naming.prefix("cd", EXAMPLE[ex]), key))
        return None if r is None or r.get("failed") else r

    def solution(self, ex, rec):
        pre = naming.prefix("cd", EXAMPLE[ex])
        if pre not in self._npz:
            path = os.path.join(self.outdir, pre, f"{pre}_solutions.npz")
            self._npz[pre] = np.load(path) if os.path.exists(path) else None
        Z = self._npz[pre]
        if Z is None or rec is None or rec["run"] not in Z.files:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return Z[rec["run"]]

    def points(self, ex, eps, mesh, n):
        P, T = make_mesh(ex, eps, mesh, n)
        pre = naming.prefix("cd", EXAMPLE[ex])
        self.solution(ex, {"run": "-"})                          # load the npz
        Z = self._npz.get(pre)
        mk = mesh_key(eps, mesh, n)
        if Z is not None and mk in Z.files and (Z[mk].shape != P.shape or
                                                not np.allclose(Z[mk], P, rtol=0, atol=1e-12)):
            raise RuntimeError(f"the stored points {mk} differ from run_study.make_mesh")
        return P, T


def residual0(r):
    if r.get("solver", "").startswith("newton"):
        return r.get("residual0")
    return r.get("residual")


def solved4(r):
    """galerkit interior_layer.py: the solver converged (Newton: ended at sigma = 0, 'polished') and the
    residual of the scheme is <= 1e-8; runs of earlier versions without residual0 / polished: converged flag"""
    if r is None:
        return False
    if r["method"] in LINEAR:
        return True
    if not r.get("converged"):
        return False
    if r.get("solver", "").startswith("newton") and r.get("polished") is False:
        return False
    x = residual0(r)
    return True if x is None else x <= SOLVED_TOL


def choose4(R, mesh, n, method, variant):
    if method in LINEAR:
        return R.get(4, EPS4, mesh, n, method)
    cands = [R.get(4, EPS4, mesh, n, method, variant, s) for s in SOLVERS4
             if not (method == "mc" and s == "newton_cont")]
    cands = [r for r in cands if r is not None]
    for r in cands:
        if solved4(r):
            return r
    if not cands:
        return None

    def res(r):
        x = residual0(r)
        return x if x is not None and np.isfinite(x) else np.inf
    return min(cands, key=res)


def _save(fig, figdir, name, pdf=True):
    os.makedirs(figdir, exist_ok=True)
    out = []
    if pdf:
        out.append(os.path.join(figdir, name + ".pdf"))
        fig.savefig(out[-1])
    out.append(os.path.join(figdir, name + ".png"))
    fig.savefig(out[-1], dpi=170 if pdf else None)
    return out


# ----------------------------------------------------------------------------- Examples 1-3
def fig_meshes(R, figdir):
    """the four mesh types of Example 3 (parabolic layers), n = 16, eps = 1e-2"""
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.6))
    for a, kind in zip(ax, ("uniform", "shishkin_r", "shishkin_l", "shishkin_x")):
        P, T = make_mesh(3, 1e-2, kind, 16)
        a.triplot(P[:, 0], P[:, 1], T, color=INK, lw=0.4)
        a.set_aspect("equal")
        a.axis("off")
        a.set_title(MESHP[kind], fontsize=9)
    out = _save(fig, figdir, "aniso_meshes")
    plt.close(fig)
    return out


def accuracy_run(R, ex, eps, mesh, n, method):
    """the fixed point run; if it did not converge, the run of another solver with a smaller residual"""
    r = R.get(ex, eps, mesh, n, method)
    if r is None or method in LINEAR or r.get("converged"):
        return r
    best, best_res = r, r.get("residual") if r.get("residual") is not None else np.inf
    for s in ("fp_anderson_safe", "newton_cont"):
        q = R.get(ex, eps, mesh, n, method, "plain", s)
        x = None if q is None else residual0(q)
        if x is not None and np.isfinite(x) and x < best_res:
            best, best_res = q, x
    return best


def fig_errors(R, figdir, ns=NS):
    cases = [(1, 1e-4), (1, 1e-8), (2, 1e-8)]
    if not any(R.get(ex, eps, m, n, s) for ex, eps in cases for m in MESHES3 for n in ns for s in AFC):
        return []
    plt = _plt()
    fig, ax = plt.subplots(len(cases), 3, figsize=(10, 2.7 * len(cases)), sharex=True)
    for i, (ex, e) in enumerate(cases):
        for j, m in enumerate(MESHES3):
            a = ax[i, j]
            for k, s in enumerate(AFC):
                y = [accuracy_run(R, ex, e, m, n, s) for n in ns]
                y = [r["L2"] if r and r.get("L2") is not None else np.nan for r in y]
                a.loglog(ns, np.maximum(y, 1e-16), "-", marker=MRK[k], color=COL[k], label=NAME[s])
            r = [R.get(ex, e, m, n, "supg") for n in ns]
            a.loglog(ns, [max(x["L2"], 1e-16) if x and x.get("L2") is not None else np.nan for x in r], "--",
                     color=MUTED, lw=1.0, label="SUPG")
            nn = np.array(ns, float)
            a.loglog(nn, 0.5 * (np.log(nn) / nn) ** 2, ":", color=MUTED, lw=0.8)
            a.set_title(f"{PROBLEM[ex]}, ε = {e:g}, {MESHP[m]}", fontsize=8)
            if j == 0:
                a.set_ylabel("$L^2$ error")
            if i == len(cases) - 1:
                a.set_xlabel("n (cells per direction)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=9, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.005, "dotted: (N⁻¹ ln N)²", ha="right", color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = _save(fig, figdir, "aniso_errors")
    plt.close(fig)
    return out


def fig_iterations(R, figdir, ns=NS):
    cases = [(1, 1e-8), (2, 1e-8), (3, 1e-8)]
    if not any(R.get(ex, eps, m, n, s) for ex, eps in cases for m in MESHES3 for n in ns for s in AFC):
        return []
    plt = _plt()
    fig, ax = plt.subplots(len(cases), 3, figsize=(10, 2.7 * len(cases)), sharex=True, sharey=True)
    for i, (ex, e) in enumerate(cases):
        for j, m in enumerate(MESHES3):
            a = ax[i, j]
            for k, s in enumerate(AFC):
                rr = [R.get(ex, e, m, n, s) for n in ns]
                y = [r["iterations"] if r and r.get("iterations") is not None else np.nan for r in rr]
                a.semilogx(ns, np.maximum(y, 1), "-", marker=MRK[k], color=COL[k], label=NAME[s])
                for n, r in zip(ns, rr):
                    if r and not r.get("converged"):
                        a.plot([n], [r["iterations"]], marker=MRK[k], ms=8, mfc="none", mec=COL[k])
            a.set_yscale("log")
            a.set_title(f"{PROBLEM[ex]}, ε = {e:g}, {MESHP[m]}", fontsize=8)
            if j == 0:
                a.set_ylabel("fixed point iterations")
            if i == len(cases) - 1:
                a.set_xlabel("n")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = _save(fig, figdir, "aniso_iterations")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- Example 4
def ty4(n):
    return 2.0 * np.sqrt(EPS4) * np.log(n)


def fig_interior(R, figdir, n_plot=64, ns=NS4, mddir=None):
    if not any(choose4(R, m, n, s, "plain") for m in MESHES4 for n in ns for s in ALL):
        return []
    out, lines = [], [f"# Example 4, eps = {EPS4:g}: runs of the interior figures", "",
                      "solver: the first of " + ", ".join(SOLVERS4) + " that solved the scheme (galerkit's rule)", "",
                      "| mesh | n | method | run | solver | iterations | residual (sigma = 0) | solved | Linf_mid |",
                      "|---|---|---|---|---|---|---|---|---|"]
    for m in MESHES4:
        for n in ns:
            for s in ALL:
                for var in (("plain", "loff") if s in AFC else ("plain",)):
                    r = choose4(R, m, n, s, var)
                    if r is None:
                        continue
                    lin = s in LINEAR
                    x = None if lin else residual0(r)
                    lines.append(f"| {m} | {n} | {NAME[s]}{' +off' if var == 'loff' else ''} | {r['run']} | "
                                 f"{r.get('solver')} | {'-' if lin else r.get('iterations')} | "
                                 f"{'-' if lin else ('?' if x is None else '%.2e' % x)} | "
                                 f"{'yes' if solved4(r) else 'no'} | {r.get('Linf_mid', float('nan')):.3e} |")
    plt = _plt("interior")
    # cross sections at x ~ 1/2
    for kind, zoom in (("shishkin_r", True), ("uniform", False)):
        P, T = R.points(4, EPS4, kind, n_plot)
        xs = np.unique(np.round(P[:, 0], 14))
        xc = xs[np.argmin(np.abs(xs - 0.5))]
        col = np.flatnonzero(np.abs(P[:, 0] - xc) < 1e-13)
        col = col[np.argsort(P[col, 1])]
        ty = ty4(n_plot)
        yy = np.linspace(0.5 - 1.5 * ty, 0.5 + 1.5 * ty, 800) if zoom else np.linspace(0, 1, 800)
        ref = Q.exact(4, EPS4, np.full_like(yy, xc), yy)
        sc = 1e4 if zoom else 1
        fig, axs = plt.subplots(2, 5, figsize=(13, 5.2), sharex=True, sharey=True)
        for ax, s in zip(axs.ravel(), ALL):
            ax.plot((yy - 0.5) * sc, ref, color="#6b6a64", lw=1.0, label="reference")
            r = choose4(R, kind, n_plot, s, "plain")
            u = R.solution(4, r)
            if u is not None:
                ax.plot((P[col, 1] - 0.5) * sc, u[col], "o-", ms=2.5, lw=1.0, color="#2a78d6", label=NAME[s])
                if s in AFC:
                    u2 = R.solution(4, choose4(R, kind, n_plot, s, "loff"))
                    if u2 is not None:
                        ax.plot((P[col, 1] - 0.5) * sc, u2[col], "s--", ms=2, lw=0.8, color="#eb6834",
                                label="+ limiter off along layers")
                ax.set_title(NAME[s] + ("" if solved4(r) else " (not solved)"))
            if zoom:
                ax.set_xlim(-1.5e4 * ty, 1.5e4 * ty)
            ax.set_ylim(-0.15, 1.15)
        for ax in axs[1]:
            ax.set_xlabel("(y - 1/2) x 1e4" if zoom else "y - 1/2")
        axs[0, 0].set_ylabel("u")
        axs[1, 0].set_ylabel("u")
        axs[0, 2].legend(loc="upper left", fontsize=7, frameon=False)
        fig.suptitle(f"interior_layer, eps = {EPS4:g}: u along x = {xc:.3f} across the interior layer, "
                     f"{MNAME4[kind]} mesh, n = {n_plot}")
        out += _save(fig, figdir, f"interior_cut_{FILE4[kind]}_n{n_plot}", pdf=False)
        plt.close(fig)
    # solutions in the layer region, Shishkin mesh
    import matplotlib.tri as mtri
    P, T = R.points(4, EPS4, "shishkin_r", n_plot)
    ty = ty4(n_plot)
    tri = mtri.Triangulation(P[:, 0], (P[:, 1] - 0.5) * 1e4, T)
    fig, axs = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
    panels = [("ref", None)] + [(s, "plain") for s in ALL] + [("kuzmin_edge", "loff")]
    tc = None
    for ax, (s, var) in zip(axs.ravel(), panels):
        if s == "ref":
            u = Q.exact(4, EPS4, P[:, 0], P[:, 1])
            ttl = "reference (asymptotic)"
        else:
            u = R.solution(4, choose4(R, "shishkin_r", n_plot, s, var))
            if u is None:
                ax.set_visible(False)
                continue
            ttl = NAME[s] + (" +off" if var == "loff" else "") + f": max u {u.max():.3f}, min {u.min():.1e}"
        tc = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis", vmin=-0.05, vmax=1.05)
        ax.set_ylim(-1.5e4 * ty, 1.5e4 * ty)
        ax.set_title(ttl, fontsize=8)
        ax.grid(False)
    for ax in axs[-1]:
        ax.set_xlabel("x")
    for ax in axs[:, 0]:
        ax.set_ylabel("(y - 1/2) x 1e4")
    if tc is not None:
        fig.colorbar(tc, ax=axs, shrink=0.6, label="u")
    fig.suptitle(f"interior_layer, eps = {EPS4:g}: solutions in the interior layer region, Shishkin mesh, n = {n_plot}")
    out += _save(fig, figdir, f"interior_solutions_shishkin_n{n_plot}", pdf=False)
    plt.close(fig)
    # maximal error away from the outflow layer against n
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    cols = ["#1f1f1e", "#6b6a64", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
            "#e34948"]
    mk = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for ax, k in zip(axs, MESHES4):
        for c, m, s in zip(cols, mk, ALL):
            rr = [choose4(R, k, n, s, "plain") for n in ns]
            v = [r["Linf_mid"] if r and r.get("Linf_mid") is not None else np.nan for r in rr]
            ax.loglog(ns, v, marker=m, color=c, lw=1.2, ms=4, label=NAME[s])
            bad = [(n, r["Linf_mid"]) for n, r in zip(ns, rr) if r and not solved4(r)]
            if bad:
                ax.plot(*zip(*bad), "x", color="k", ms=7)
        ax.set_title(MNAME4[k])
        ax.set_xlabel("n")
    axs[0].set_ylabel("max nodal error, 0.1 <= x <= 1 - tau_x")
    axs[-1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(f"interior_layer, eps = {EPS4:g}: error against the asymptotic reference (x: not solved)")
    out += _save(fig, figdir, "interior_errors", pdf=False)
    plt.close(fig)
    if mddir:
        os.makedirs(mddir, exist_ok=True)
        with open(os.path.join(mddir, f"interior_n{n_plot}.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return out


def make_all(outdir, figdir, figures=FIGURES, quiet=False, mddir=None):
    """the requested figures from the runs in outdir (and the summary of the interior runs in mddir);
    returns the files written"""
    R = Runs(outdir)
    out = []
    for name, fn in (("meshes", fig_meshes), ("errors", fig_errors), ("iterations", fig_iterations),
                     ("interior", fig_interior)):
        if name in figures:
            w = fn(R, figdir, mddir=mddir) if name == "interior" else fn(R, figdir)
            out += w
            if not quiet:
                print(f"{name}: " + (", ".join(os.path.basename(p) for p in w) if w else "no runs (see --run)"))
    return out


# ----------------------------------------------------------------------------- runs
def _study(args, *opts):
    cmd = [sys.executable, os.path.join(HERE, "run_study.py"), "--exe", args.exe, "--outdir", args.outdir,
           "--jobs", str(args.jobs), *opts]
    print("\n" + " ".join(opts), flush=True)
    subprocess.run(cmd, check=True)


def compute_runs(args):
    if "errors" in args.figures or "iterations" in args.figures:      # step 3 (fixed point), the 3 meshes used
        _study(args, "--cases", "1:1e-4", "1:1e-8", "2:1e-8", "3:1e-8", "--n", *map(str, NS),
               "--meshes", *MESHES3, "--methods", "supg", *AFC)
    if "interior" in args.figures:                                     # Example 4, solver by solver
        for k, solver in enumerate(SOLVERS4):
            R = Runs(args.outdir)
            for var in ("plain", "loff"):
                groups = {}
                for m in MESHES4:
                    for n in NS4:
                        meths = tuple(s for s in (ALL if var == "plain" else AFC)
                                      if (k == 0 or (s not in LINEAR and not solved4(choose4(R, m, n, s, var))))
                                      and not (s == "mc" and solver == "newton_cont")
                                      and not (var == "loff" and m == "uniform"))
                        if meths:
                            groups.setdefault((meths, n), []).append(m)
                for (meths, n), meshes in groups.items():
                    _study(args, "--cases", f"4:{EPS4:g}", "--n", str(n), "--meshes", *meshes, "--methods", *meths,
                           "--variants", var, "--solver", solver)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=list(FIGURES), choices=FIGURES)
    ap.add_argument("--outdir", default="cd_aniso", help="the runs of run_study.py (default cd_aniso)")
    ap.add_argument("--figdir", default="paper_results", help="folder of the figures (default paper_results)")
    ap.add_argument("--mddir", default="paper_figures_runs", help="summary of the runs of the interior figures")
    ap.add_argument("--run", action="store_true", help="compute the missing runs first (run_study.py)")
    ap.add_argument("--exe", default="./parMooN_cd2d_afc_SEQUENTIAL")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.run:
        compute_runs(args)
    make_all(args.outdir, args.figdir, args.figures, mddir=args.mddir)


if __name__ == "__main__":
    main()
