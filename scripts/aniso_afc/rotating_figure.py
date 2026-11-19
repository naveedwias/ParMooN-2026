#!/usr/bin/env python3
"""
Example 5 (profile transported along circles): the figures of the paper made from ParMooN's runs,
in the layout of galerkit's rotating_profile.py (galerkit branch feature/anisotropic-afc).

  solutions  rotating_solutions_n64.png: the four meshes (uniform, polar, polar layer, tensor
             layer) and the solutions of Galerkin, SUPG, Kuzmin, MUAS and BBK on each of them,
             min u_h and max u_h in brackets; one row per mesh, colour scale [-0.1, 1.1] (viridis).
  outlet     rotating_outlet_<mesh>_n64.png (one per mesh): the outlet profile u_h(0, y) of the ten
             methods (Galerkin, SUPG, BJK, Kuzmin, MC, MUAS, SMUAS, BBK, F-BBK, S-BBK) against the
             reference (the radial heat equation, measures.radial_reference, identical to galerkit's),
             each with the variants "limiter off along the layers" (loff, aspect threshold 3; on the
             three refined meshes) and "relaxed, mu = 2" (mu2; BJK, Kuzmin, MUAS, SMUAS).

In addition, in <vtkdir>: one VTK file per mesh (legacy ASCII, unstructured grid of triangles)
with the mesh and every available solution as point data (u_Galerkin, u_SUPG, u_Kuzmin, ...,
u_<method>_loff, u_<method>_mu2), for ParaView / VisIt, and the summaries
rotating_solutions_n64.md (the runs of the solution figure: solver, iterations, residual, min,
max) and rotating_outlet_n64.md (the outlet errors: maximal and L1 error on x = 0).

The solutions are read from the runs of run_study.py (<outdir>/cd_rotating_profile/
cd_rotating_profile_solutions.npz; Example 5 keeps its solutions by default). The solver of an
algebraic method is chosen as in galerkit: the first of Anderson with safeguard
(fp_anderson_safe), Newton with continuation (newton_cont) and the fixed point iteration
(fixed_point) that solved the scheme (Euclidean residual of the scheme without regularization
<= 1e-8 at the solution, as galerkit; for Newton with continuation the residual0 of the run),
otherwise the run with the smallest such residual (--solver auto, the default); --solver <name>
takes the runs of one solver only. Newton runs of versions before residual0 was recorded fall
back to their converged flag: recompute them with run_study.py --redo --solver newton_cont.

--run computes the missing runs of the requested figures first (with run_study.py; finished runs
are skipped): in the order of the solvers above, every solver only for the runs that no earlier
solver has solved. Without --run nothing is computed.

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py --run --jobs 4
  python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py                 # from existing runs
  python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py --figures outlet
Output: <figdir>/rotating_solutions_n64.png, <figdir>/rotating_outlet_<mesh>_n64.png (default
figdir paper_results), and in <vtkdir> (default rotating_vtk): rotating_n64_<mesh>.vtk,
rotating_solutions_n64.md, rotating_outlet_n64.md.
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
import meshes as M  # noqa: E402
import naming  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402

EXAMPLE_CD2D = 30                                    # Example 5 of the paper
PREFIX = naming.prefix("cd", EXAMPLE_CD2D)           # cd_rotating_profile
MESHES = ("uniform", "polar", "polar_layer", "tensor_layer")
OFF_MESHES = MESHES[1:]                              # loff on the refined meshes (as galerkit)
MNAME = {"uniform": "uniform", "polar": "polar (aligned, uniform in r)", "polar_layer": "polar, layer-adapted",
         "tensor_layer": "tensor, layer-refined in x and y"}
ALL = ("galerkin", "supg", "bjk", "kuzmin_edge", "mc", "muas", "smuas", "bbk", "fbbk", "sbbk")
SHOW = ("galerkin", "supg", "kuzmin_edge", "muas", "bbk")    # the solution figure
RELAXED = ("bjk", "kuzmin_edge", "muas", "smuas")           # mu2
NAME = {"galerkin": "Galerkin", "supg": "SUPG", "bjk": "BJK", "kuzmin_edge": "Kuzmin", "mc": "MC", "muas": "MUAS",
        "smuas": "SMUAS", "bbk": "BBK", "fbbk": "F-BBK", "sbbk": "S-BBK"}
VNAME = {"plain": "", "loff": " +off", "mu2": " mu=2"}
LINEAR = ("galerkin", "supg")
SOLVERS = ("fp_anderson_safe", "newton_cont", "fixed_point")   # galerkit's order
SUFFIX = {"fixed_point": "", "fp_anderson": "_aa", "fp_anderson_safe": "_aas", "newton": "_nw",
          "newton_cont": "_nc"}
FIGURES = ("solutions", "outlet")
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 / older


def key_of(eps, mesh, n, method, variant, solver):
    return run_key(eps, mesh, n, method, variant, "linear", "" if method in LINEAR else SUFFIX[solver])


def records(outdir):
    path = os.path.join(outdir, PREFIX, f"{PREFIX}_runs.jsonl")
    return RunStore.read_jsonl(path) if os.path.exists(path) else {}


def choose(recs, eps, mesh, n, method, variant, solvers):
    """the run used for (mesh, method, variant): the first converged run in the order of solvers,
    otherwise the run with the smallest residual; None if there is no run with a solution"""
    if method in LINEAR:
        r = recs.get(key_of(eps, mesh, n, method, "plain", solvers[0]))
        return r if r is not None and not r.get("failed") else None
    cands = [recs.get(key_of(eps, mesh, n, method, variant, s)) for s in solvers
             if not (method == "mc" and s == "newton_cont")]
    cands = [r for r in cands if r is not None and not r.get("failed")]
    for r in cands:
        if solved(r):
            return r
    if not cands:
        return None

    def res(r):
        x = residual0(r)
        return x if x is not None and np.isfinite(x) else np.inf
    return min(cands, key=res)


SOLVED_TOL = 1e-8                                    # galerkit rotating_profile.py: |r(u_h)|_2 <= 1e-8, sigma = 0


def residual0(r):
    """the residual of the scheme without regularization at the solution of the run; None for a Newton run
    of an earlier version (no res0 in its log: recompute it with run_study.py --redo)"""
    if r.get("solver", "").startswith("newton"):
        return r.get("residual0")
    return r.get("residual")


def solved(r):
    """as galerkit: linear methods always; algebraic ones if the residual with sigma = 0 is <= 1e-8 (Newton
    runs of earlier versions: their converged flag)"""
    if r is None:
        return False
    if r["method"] in LINEAR:
        return True
    x = residual0(r)
    return bool(r.get("converged")) if x is None else x <= SOLVED_TOL


def needed(figures):
    """(mesh, method, variant) of the runs the figures use"""
    out = []
    if "solutions" in figures:
        out += [(m, meth, "plain") for m in MESHES for meth in SHOW]
    if "outlet" in figures:
        out += [(m, meth, "plain") for m in MESHES for meth in ALL]
        out += [(m, meth, "loff") for m in OFF_MESHES for meth in ALL if meth not in LINEAR]
        out += [(m, meth, "mu2") for m in MESHES for meth in RELAXED]
    return list(dict.fromkeys(out))


# ----------------------------------------------------------------------------- runs
def compute_runs(args, figures, solvers):
    """missing runs with run_study.py, solver by solver, only where no earlier solver converged"""
    study = os.path.join(HERE, "run_study.py")
    todo_all = needed(figures)
    for k, solver in enumerate(solvers):
        recs = records(args.outdir)
        groups = {}                                   # (variant, methods) -> meshes: one call per group
        for mesh in MESHES:
            for var in ("plain", "loff", "mu2"):
                meths = tuple(meth for (m, meth, v) in todo_all if m == mesh and v == var
                              and (k == 0 or (meth not in LINEAR and not solved(
                                  choose(recs, args.eps, mesh, args.n, meth, var, solvers[:k]))))
                              and not (meth == "mc" and solver == "newton_cont"))
                if meths:
                    groups.setdefault((var, meths), []).append(mesh)
        for (var, meths), meshes in groups.items():
            cmd = [sys.executable, study, "--exe", args.exe, "--outdir", args.outdir, "--jobs", str(args.jobs),
                   "--cases", f"5:{args.eps:g}", "--n", str(args.n), "--meshes", *meshes, "--methods", *meths,
                   "--variants", var, "--solver", solver, "--solutions", "auto"]
            print(f"\n[{solver}, {var}] meshes {' '.join(meshes)}: {' '.join(meths)}", flush=True)
            subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------- data
class Runs:
    """the solutions of the runs of Example 5 at one n (chosen solver per method and variant)"""

    def __init__(self, outdir, n, eps, solver="auto"):
        self.outdir, self.n, self.eps = outdir, n, eps
        self.solvers = SOLVERS if solver == "auto" else (solver,)
        self.recs = records(outdir)
        path = os.path.join(outdir, PREFIX, f"{PREFIX}_solutions.npz")
        self.Z = np.load(path) if os.path.exists(path) else None
        self.mesh_cache = {}

    def mesh(self, kind):
        if kind not in self.mesh_cache:
            P, T = M.rotating_mesh(kind, self.n, eps=self.eps)
            mk = mesh_key(self.eps, kind, self.n)
            if self.Z is not None and mk in self.Z.files:
                Ps = self.Z[mk]
                if Ps.shape != P.shape or not np.allclose(Ps, P, rtol=0, atol=1e-12):
                    raise RuntimeError(f"the stored points {mk} differ from meshes.rotating_mesh")
            self.mesh_cache[kind] = (P, T)
        return self.mesh_cache[kind]

    def get(self, kind, method, variant="plain"):
        """(u, record) or (None, reason)"""
        r = choose(self.recs, self.eps, kind, self.n, method, variant, self.solvers)
        if r is None:
            return None, "no run"
        if self.Z is None or r["run"] not in self.Z.files:
            return None, "run without stored solution"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # duplicate names: the last entry counts
            return self.Z[r["run"]], r

    @property
    def empty(self):
        return self.Z is None


def outlet(P, u):
    """nodes on x = 0, sorted in y"""
    o = np.flatnonzero(np.abs(P[:, 0]) < 1e-12)
    o = o[np.argsort(P[o, 1])]
    return P[o, 1], u[o]


def reference_outlet(outdir, eps):
    """r and the reference at theta = pi/2 (cached in <outdir>/<prefix>/)"""
    f = os.path.join(outdir, PREFIX, f"reference_outlet_eps{eps:.0e}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["r"], z["u"]
    r, u = Q.radial_reference(eps, last_only=True)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, r=r, u=u)
    return r, u


def write_vtk(path, P, T, fields, title):
    """legacy ASCII VTK, unstructured grid of triangles (cell type 5), point data per field"""
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title[:255] + "\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(P)} double\n")
        f.write("\n".join(f"{x:.17g} {y:.17g} 0" for x, y in P) + "\n")
        f.write(f"CELLS {len(T)} {4 * len(T)}\n")
        f.write("\n".join(f"3 {a} {b} {c}" for a, b, c in T) + "\n")
        f.write(f"CELL_TYPES {len(T)}\n" + "\n".join("5" for _ in T) + "\n")
        if fields:
            f.write(f"POINT_DATA {len(P)}\n")
            for name, u in fields:
                f.write(f"SCALARS {name} double\nLOOKUP_TABLE default\n")
                f.write("\n".join(f"{v:.17g}" for v in u) + "\n")


def _plt(dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.color": "#e4e3de", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": dpi})
    return plt


def _status(r):
    lin = r["method"] in LINEAR
    x = None if lin else residual0(r)
    return (r.get("solver"), "-" if lin else r.get("iterations"),
            "-" if lin else ("?" if x is None else "%.2e" % x), "-" if lin else ("yes" if solved(r) else "no"))


# ----------------------------------------------------------------------------- figures
def make_figure(outdir, figdir, vtkdir=None, n=64, eps=M.ROT_EPS, variant="plain", solver="auto", dpi=150,
                quiet=False, runs=None):
    """rotating_solutions_n<n>.png (and the VTK files, if vtkdir); returns the path of the PNG, or None
    without solutions"""
    R = runs or Runs(outdir, n, eps, solver)
    if R.empty or all(R.get(m, meth, variant)[0] is None for m in MESHES for meth in SHOW):
        if not quiet:
            print(f"no solutions of Example 5 at n = {n} in {outdir} (compute them with --run)")
        return None
    plt = _plt(dpi)
    import matplotlib.tri as mtri
    fig, axs = plt.subplots(len(MESHES), len(SHOW) + 1, figsize=(16, 12.5))
    tc, unsolved, missing = None, False, []
    lines = [f"# Example 5, eps = {eps:g}, n = {n}: runs of the figure rotating_solutions_n{n}", "",
             f"variant {variant}, solver {solver}" + (" (first converged of " + ", ".join(SOLVERS) + ")"
                                                      if solver == "auto" else ""), "",
             "| mesh | method | run | solver | iterations | residual (sigma = 0) | solved (<= 1e-8) | min u_h | max u_h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, mesh in enumerate(MESHES):
        P, T = R.mesh(mesh)
        tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
        ax = axs[i, 0]
        ax.triplot(tri, lw=0.15, color="#6b6a64")
        ax.set_title(f"{MNAME[mesh]} mesh", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(False)
        for j, meth in enumerate(SHOW):
            ax = axs[i, j + 1]
            u, r = R.get(mesh, meth, variant)
            if u is None:
                ax.set_visible(False)
                missing.append((mesh, meth, r))
                continue
            star = "" if solved(r) else "*"
            unsolved |= bool(star)
            tc = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis", vmin=-0.1, vmax=1.1)
            ax.set_title(f"{NAME[meth]}{star}: [{u.min():.2f}, {u.max():.2f}]", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(False)
            lines.append(f"| {mesh} | {NAME[meth]} | {r['run']} | " + " | ".join(map(str, _status(r)))
                         + f" | {u.min():.4f} | {u.max():.4f} |")
    if tc is not None:
        fig.colorbar(tc, ax=axs, shrink=0.5, label="u")
    fig.suptitle(f"rotating_profile, eps = {eps:g}, n = {n}: meshes and solutions (min, max in brackets)"
                 + ("; * nonlinear problem not solved" if unsolved else ""))
    os.makedirs(figdir, exist_ok=True)
    png = os.path.join(figdir, f"rotating_solutions_n{n}.png")
    fig.savefig(png)
    plt.close(fig)
    for mesh, meth, why in missing:
        lines.append(f"| {mesh} | {NAME[meth]} | missing ({why}) | | | | | | |")
    if vtkdir:
        os.makedirs(vtkdir, exist_ok=True)
        for mesh in MESHES:
            P, T = R.mesh(mesh)
            fields = []
            for var in ("plain", "loff", "mu2"):
                for meth in ALL:
                    if var != "plain" and meth in LINEAR:
                        continue
                    u, _ = R.get(mesh, meth, var)
                    if u is not None:
                        fields.append(("u_" + NAME[meth].replace("-", "") + ("" if var == "plain" else "_" + var), u))
            f = os.path.join(vtkdir, f"rotating_n{n}_{mesh}.vtk")
            write_vtk(f, P, T, fields, f"Example 5 (rotating profile), eps = {eps:g}, {mesh} mesh, n = {n}")
            if not quiet:
                print(f"wrote {f} ({len(fields)} solutions)")
        open(os.path.join(vtkdir, f"rotating_solutions_n{n}.md"), "w").write("\n".join(lines) + "\n")
    if not quiet:
        print(f"wrote {png}")
        print("\n".join(lines[4:]))
        if missing:
            print(f"\n{len(missing)} panels missing (empty in the figure); compute them with --run")
    return png


def make_outlet_figures(outdir, figdir, mddir=None, n=64, eps=M.ROT_EPS, solver="auto", dpi=150, quiet=False,
                        runs=None):
    """rotating_outlet_<mesh>_n<n>.png for the four meshes (and rotating_outlet_n<n>.md in mddir);
    returns the paths of the PNGs written"""
    R = runs or Runs(outdir, n, eps, solver)
    if R.empty:
        if not quiet:
            print(f"no solutions of Example 5 in {outdir} (compute them with --run)")
        return []
    r_ref, u_ref = reference_outlet(outdir, eps)
    plt = _plt(dpi)
    os.makedirs(figdir, exist_ok=True)
    pngs, missing = [], 0
    lines = [f"# Example 5, eps = {eps:g}, n = {n}: outlet profile u_h(0, y) against the reference", "",
             "max and L1 (trapezoidal rule on the outlet nodes) error of u_h(0, y); solver " + solver
             + (" (first converged of " + ", ".join(SOLVERS) + ")" if solver == "auto" else ""), "",
             "| mesh | method | run | solver | iterations | residual (sigma = 0) | solved (<= 1e-8) | outlet max error | outlet L1 error |",
             "|---|---|---|---|---|---|---|---|---|"]
    styles = (("plain", "o-", "#2a78d6", "as is"), ("loff", "--", "#eb6834", "limiter off along layers"),
              ("mu2", ":", "#1baf7a", "relaxed, mu = 2"))
    for mesh in MESHES:
        P, T = R.mesh(mesh)
        fig, axs = plt.subplots(2, 5, figsize=(13, 5.2), sharex=True, sharey=True)
        drawn = False
        for ax, meth in zip(axs.ravel(), ALL):
            ax.plot(r_ref, u_ref, color="#6b6a64", lw=1.0, label="reference")
            title = NAME[meth]
            for var, ls, col, lab in styles:
                if var != "plain" and meth in LINEAR:
                    continue
                u, rec = R.get(mesh, meth, var)
                if u is None:
                    if var == "plain":
                        title += " (no run)"
                        missing += 1
                    continue
                yo, uo = outlet(P, u)
                e = np.abs(uo - np.interp(yo, r_ref, u_ref))
                ok = solved(rec)
                if var == "plain":
                    ax.plot(yo, uo, ls, ms=2, lw=0.9, color=col, label=lab)
                    if not ok:
                        title += " (not solved)"
                else:
                    ax.plot(yo, uo, ls, lw=1.1, color=col, label=lab + ("" if ok else " (not solved)"))
                drawn = True
                lines.append(f"| {mesh} | {NAME[meth]}{VNAME[var]} | {rec['run']} | " + " | ".join(map(str, _status(rec)))
                             + f" | {e.max():.3e} | {_trapz(e, yo):.3e} |")
            ax.set_title(title)
            ax.set_xlim(0.3, 1.0)
            ax.set_ylim(-0.15, 1.15)
        for ax in axs[1]:
            ax.set_xlabel("y (outlet x = 0)")
        axs[0, 0].set_ylabel("u(0, y)")
        axs[1, 0].set_ylabel("u(0, y)")
        axs[0, 0].legend(fontsize=7, frameon=False)
        axs[0, 2].legend(fontsize=7, frameon=False, loc="upper right")
        fig.suptitle(f"rotating_profile, eps = {eps:g}: outlet profile, {MNAME[mesh]} mesh, n = {n}")
        if drawn:
            f = os.path.join(figdir, f"rotating_outlet_{mesh}_n{n}.png")
            fig.savefig(f)
            pngs.append(f)
            if not quiet:
                print(f"wrote {f}")
        plt.close(fig)
    if mddir:
        os.makedirs(mddir, exist_ok=True)
        open(os.path.join(mddir, f"rotating_outlet_n{n}.md"), "w").write("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines[4:]))
        if missing:
            print(f"\n{missing} panels without a run; compute them with --run --figures outlet")
    return pngs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=list(FIGURES), choices=FIGURES)
    ap.add_argument("--outdir", default="cd_aniso", help="the runs of run_study.py (default cd_aniso)")
    ap.add_argument("--figdir", default="paper_results", help="folder of the PNGs (default paper_results)")
    ap.add_argument("--vtkdir", default="rotating_vtk", help="folder of the VTK files and the summaries")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--eps", type=float, default=M.ROT_EPS)
    ap.add_argument("--variant", default="plain", choices=["plain", "loff", "mu2"],
                    help="of the algebraic methods in the solution figure (the paper: plain)")
    ap.add_argument("--solver", default="auto", choices=["auto", *SOLVERS],
                    help="auto: the first converged of fp_anderson_safe, newton_cont, fixed_point (as galerkit)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--run", action="store_true", help="compute the missing runs first (run_study.py)")
    ap.add_argument("--exe", default="./parMooN_cd2d_afc_SEQUENTIAL")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.run:
        compute_runs(args, args.figures, SOLVERS if args.solver == "auto" else (args.solver,))
    R = Runs(args.outdir, args.n, args.eps, args.solver)
    ok = True
    if "solutions" in args.figures:
        ok &= make_figure(args.outdir, args.figdir, args.vtkdir, args.n, args.eps, args.variant, args.solver,
                          args.dpi, runs=R) is not None
    if "outlet" in args.figures:
        ok &= bool(make_outlet_figures(args.outdir, args.figdir, args.vtkdir, args.n, args.eps, args.solver,
                                       args.dpi, runs=R))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
