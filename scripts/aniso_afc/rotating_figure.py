#!/usr/bin/env python3
"""
Example 5 (profile transported along circles), the figure rotating_solutions_n64 of the paper:
the four meshes (uniform, polar, polar layer, tensor layer) at n = 64 and the solutions of
Galerkin, SUPG, Kuzmin, MUAS and BBK on each of them, with min u_h and max u_h in brackets.
Layout as galerkit's rotating_profile.py (galerkit branch feature/anisotropic-afc):
one row per mesh, the mesh in the first column, the five solutions to its right,
colour scale [-0.1, 1.1] (viridis), PNG.

In addition, one VTK file per mesh (legacy ASCII, unstructured grid of triangles) with the
mesh and the five solutions as point data u_Galerkin, u_SUPG, u_Kuzmin, u_MUAS, u_BBK, for
ParaView / VisIt, and a summary of the runs used (solver, iterations, residual, min, max).

The solutions are read from the runs of run_study.py (<outdir>/cd_rotating_profile/
cd_rotating_profile_solutions.npz; Example 5 keeps its solutions by default). The solver of an
algebraic method is chosen as in galerkit: the first of Anderson with safeguard
(fp_anderson_safe), Newton with continuation (newton_cont) and the fixed point iteration
(fixed_point) whose run converged, otherwise the run with the smallest residual
(--solver auto, the default); --solver <name> takes the runs of one solver only.

--run computes the missing runs first (with run_study.py; finished runs are skipped): in the
order of the solvers above, every solver only for the methods and meshes that no earlier
solver has solved. Without --run nothing is computed.

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py --run --jobs 4
  python3 ../ParMooN-2026/scripts/aniso_afc/rotating_figure.py          # from existing runs
Output: <figdir>/rotating_solutions_n64.png (default figdir paper_results), and in <vtkdir>
(default rotating_vtk): rotating_n64_<mesh>.vtk (four files), rotating_solutions_n64.md.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import meshes as M  # noqa: E402
import naming  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402

EXAMPLE_CD2D = 30                                    # Example 5 of the paper
PREFIX = naming.prefix("cd", EXAMPLE_CD2D)           # cd_rotating_profile
MESHES = ("uniform", "polar", "polar_layer", "tensor_layer")
MNAME = {"uniform": "uniform", "polar": "polar (aligned, uniform in r)", "polar_layer": "polar, layer-adapted",
         "tensor_layer": "tensor, layer-refined in x and y"}
METHODS = ("galerkin", "supg", "kuzmin_edge", "muas", "bbk")
NAME = {"galerkin": "Galerkin", "supg": "SUPG", "kuzmin_edge": "Kuzmin", "muas": "MUAS", "bbk": "BBK"}
LINEAR = ("galerkin", "supg")
SOLVERS = ("fp_anderson_safe", "newton_cont", "fixed_point")   # galerkit's order
SUFFIX = {"fixed_point": "", "fp_anderson": "_aa", "fp_anderson_safe": "_aas", "newton": "_nw",
          "newton_cont": "_nc"}


def key_of(eps, mesh, n, method, variant, solver):
    return run_key(eps, mesh, n, method, variant, "linear", "" if method in LINEAR else SUFFIX[solver])


def records(outdir):
    path = os.path.join(outdir, PREFIX, f"{PREFIX}_runs.jsonl")
    return RunStore.read_jsonl(path) if os.path.exists(path) else {}


def choose(recs, eps, mesh, n, method, variant, solvers):
    """the run used for (mesh, method): the first converged run in the order of solvers, otherwise the
    run with the smallest residual; None if there is no run with a solution"""
    if method in LINEAR:
        r = recs.get(key_of(eps, mesh, n, method, "plain", solvers[0]))
        return r if r is not None and not r.get("failed") else None
    cands = [recs.get(key_of(eps, mesh, n, method, variant, s)) for s in solvers]
    cands = [r for r in cands if r is not None and not r.get("failed")]
    for r in cands:
        if r.get("converged"):
            return r
    if not cands:
        return None

    def res(r):
        x = r.get("residual")
        return x if x is not None and np.isfinite(x) else np.inf
    return min(cands, key=res)


def solved(r):
    return r is not None and (r["method"] in LINEAR or bool(r.get("converged")))


# ----------------------------------------------------------------------------- runs
def compute_runs(args, solvers):
    """missing runs with run_study.py, solver by solver, only where no earlier solver converged"""
    study = os.path.join(HERE, "run_study.py")
    for k, solver in enumerate(solvers):
        recs = records(args.outdir)
        groups = {}                                   # methods to run -> meshes (one call per group)
        for mesh in MESHES:
            if k == 0:
                todo = tuple(METHODS)
            else:                                     # only what the earlier solvers have not solved
                todo = tuple(m for m in METHODS if m not in LINEAR and not solved(
                    choose(recs, args.eps, mesh, args.n, m, args.variant, solvers[:k])))
            if todo:
                groups.setdefault(todo, []).append(mesh)
        for todo, meshes in groups.items():
            cmd = [sys.executable, study, "--exe", args.exe, "--outdir", args.outdir, "--jobs", str(args.jobs),
                   "--cases", f"5:{args.eps:g}", "--n", str(args.n), "--meshes", *meshes, "--methods", *todo,
                   "--variants", args.variant, "--solver", solver, "--solutions", "auto"]
            print(f"\n[{solver}] meshes {' '.join(meshes)}: {' '.join(todo)}", flush=True)
            subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------------- output
def load_solutions(outdir):
    path = os.path.join(outdir, PREFIX, f"{PREFIX}_solutions.npz")
    return np.load(path) if os.path.exists(path) else None


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


def make_figure(outdir, figdir, vtkdir=None, n=64, eps=M.ROT_EPS, variant="plain", solver="auto", dpi=150,
                quiet=False):
    """the figure (and the VTK files, if vtkdir); returns the path of the PNG, or None without solutions"""
    solvers = SOLVERS if solver == "auto" else (solver,)
    recs = records(outdir)
    Z = load_solutions(outdir)
    if Z is None:
        if not quiet:
            print(f"no solutions of Example 5 in {outdir}/{PREFIX}/ (compute them with --run)")
        return None
    sol, missing = {}, []
    for mesh in MESHES:
        P, T = M.rotating_mesh(mesh, n, eps=eps)
        mk = mesh_key(eps, mesh, n)
        if mk in Z.files and (Z[mk].shape != P.shape or not np.allclose(Z[mk], P, rtol=0, atol=1e-12)):
            raise RuntimeError(f"the stored points {mk} differ from meshes.rotating_mesh: other version of the mesh")
        for meth in METHODS:
            r = choose(recs, eps, mesh, n, meth, variant, solvers)
            if r is None or r["run"] not in Z.files:
                missing.append((mesh, meth, "no run" if r is None else "run without stored solution"))
                continue
            sol[mesh, meth] = (Z[r["run"]], r)
    if not sol:
        if not quiet:
            print(f"no solutions of Example 5 at n = {n} in {outdir} (compute them with --run)")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.color": "#e4e3de", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": dpi})
    fig, axs = plt.subplots(len(MESHES), len(METHODS) + 1, figsize=(16, 12.5))
    tc, lines, unsolved = None, [], False
    lines += [f"# Example 5, eps = {eps:g}, n = {n}: runs of the figure rotating_solutions_n{n}", "",
              f"variant {variant}, solver {solver}" + (" (first converged of " + ", ".join(SOLVERS) + ")"
                                                       if solver == "auto" else ""), "",
              "| mesh | method | run | solver | iterations | residual | converged | min u_h | max u_h |",
              "|---|---|---|---|---|---|---|---|---|"]
    if vtkdir:
        os.makedirs(vtkdir, exist_ok=True)
    for i, mesh in enumerate(MESHES):
        P, T = M.rotating_mesh(mesh, n, eps=eps)
        tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
        ax = axs[i, 0]
        ax.triplot(tri, lw=0.15, color="#6b6a64")
        ax.set_title(f"{MNAME[mesh]} mesh", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(False)
        fields = []
        for j, meth in enumerate(METHODS):
            ax = axs[i, j + 1]
            if (mesh, meth) not in sol:
                ax.set_visible(False)
                continue
            u, r = sol[mesh, meth]
            star = "" if solved(r) else "*"
            unsolved |= bool(star)
            tc = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis", vmin=-0.1, vmax=1.1)
            ax.set_title(f"{NAME[meth]}{star}: [{u.min():.2f}, {u.max():.2f}]", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(False)
            fields.append((f"u_{NAME[meth]}", u))
            lin = meth in LINEAR
            lines.append(f"| {mesh} | {NAME[meth]} | {r['run']} | {r.get('solver')} | "
                         f"{'-' if lin else r.get('iterations')} | {'-' if lin else '%.2e' % r.get('residual')} | "
                         f"{'-' if lin else ('yes' if r.get('converged') else 'no')} | {u.min():.4f} | {u.max():.4f} |")
        if vtkdir:
            f = os.path.join(vtkdir, f"rotating_n{n}_{mesh}.vtk")
            write_vtk(f, P, T, fields, f"Example 5 (rotating profile), eps = {eps:g}, {mesh} mesh, n = {n}")
            if not quiet:
                print(f"wrote {f}")
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
        open(os.path.join(vtkdir, f"rotating_solutions_n{n}.md"), "w").write("\n".join(lines) + "\n")
    if not quiet:
        print(f"wrote {png}")
        print("\n".join(lines[4:]))
        if missing:
            print(f"\n{len(missing)} panels missing (empty in the figure); compute them with --run")
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="cd_aniso", help="the runs of run_study.py (default cd_aniso)")
    ap.add_argument("--figdir", default="paper_results", help="folder of the PNG (default paper_results)")
    ap.add_argument("--vtkdir", default="rotating_vtk", help="folder of the VTK files and the summary")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--eps", type=float, default=M.ROT_EPS)
    ap.add_argument("--variant", default="plain", choices=["plain", "loff", "mu2"],
                    help="of the algebraic methods (the paper: plain)")
    ap.add_argument("--solver", default="auto", choices=["auto", *SOLVERS],
                    help="auto: the first converged of fp_anderson_safe, newton_cont, fixed_point (as galerkit)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--run", action="store_true", help="compute the missing runs first (run_study.py)")
    ap.add_argument("--exe", default="./parMooN_cd2d_afc_SEQUENTIAL")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.run:
        compute_runs(args, SOLVERS if args.solver == "auto" else (args.solver,))
    if make_figure(args.outdir, args.figdir, args.vtkdir, args.n, args.eps, args.variant, args.solver,
                   args.dpi) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
