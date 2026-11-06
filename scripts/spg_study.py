#!/usr/bin/env python3
"""
Run the numerical studies of

  M. Afzal, N. Ahmed, V. John, A robust a posteriori error estimator for the
  Oseen problem, Comput. Math. Appl. 223 (2026) 467-486

with the ParMooN program `parMooN_oseen2d_spg_adaptive_SEQUENTIAL`, store the
per-level tables as CSV files and produce the figures of Sections 4 and 5
(errors, estimator, effectivity indices and the individual contributions versus
the number of degrees of freedom).

Studies (option --study, several can be given):
  ex41  Example 4.1, Oseen, smooth solution, uniform refinement   (Figs. 1-4)
  ex42  Example 4.2, Oseen, boundary layers, adaptive refinement  (Figs. 7-10)
  ex51  Example 5.1, Navier-Stokes, smooth solution, uniform      (Figs. 11-13)
  ex52  Example 5.2, Navier-Stokes, boundary layers, adaptive     (Fig. 15)

Typical use (from the build directory):

  python3 ../ParMooN-2026/scripts/spg_study.py --study ex41 ex42 \\
      --pairs 2/1 2/2 --nu 1e-3 1e-4 1e-5 1e-6 --max-dofs 2e5

Runs whose table file already exists in the output directory are not repeated,
so the script can be restarted and the plots regenerated with --plot-only.
"""
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# configuration of the four studies
# ---------------------------------------------------------------------------
STUDIES = {
    # example, problem_type, sigma0, uniform, initial refinements, n levels,
    # rates O(h^k): equal order k+1/2, inf-sup stable k
    "ex41": dict(example=21, problem_type=3, sigma0=1.0, uniform=True,
                 initial=1, levels=6, title="Example 4.1 (Oseen, smooth)"),
    "ex42": dict(example=22, problem_type=3, sigma0=0.0, uniform=False,
                 initial=1, levels=16, title="Example 4.2 (Oseen, layers)"),
    "ex51": dict(example=21, problem_type=5, sigma0=0.0, uniform=True,
                 initial=2, levels=4, title="Example 5.1 (Navier-Stokes, smooth)"),
    "ex52": dict(example=22, problem_type=5, sigma0=0.0, uniform=False,
                 initial=1, levels=12, title="Example 5.2 (Navier-Stokes, layers)"),
}
DEFAULT_NU = {"ex41": [1e-3, 1e-4, 1e-5, 1e-6], "ex42": [1e-3, 1e-4, 1e-5, 1e-6],
              "ex51": [1e-2, 2.5e-3], "ex52": [1e-2]}
DEFAULT_PAIRS = ["1/1", "2/1", "2/2", "3/2", "3/3"]

TABLE_COLUMNS = ["level", "cells", "dofs", "h", "error_spg", "eta", "eta_res",
                 "eta_div", "eta_F", "eta_delta", "eta_mu", "effectivity",
                 "div_uh", "rate_err", "rate_eta"]

DAT_TEMPLATE = """[ ParMooN database for the SUPG/PSPG/grad-div Oseen estimator study ]
outfile: {name}.out
boundary_file: Default_UnitSquare
geo_file: TwoTriangles
refinement_n_initial_steps: {initial}
refinement_max_n_adaptive_steps: {levels}
conforming_closure: true
spg_uniform_refinement: {uniform}
problem_type: {problem_type}
example: {example}
verbosity: 1
output_write_vtk: {vtk}
output_directory: {vtk_dir}
output_compute_errors: true
output_basename: {name}
space_discretization_type: supg_pspg_graddiv
spg_delta0: {delta0}
spg_mu0: {mu0}
spg_estimator_sigma0: {sigma0}
spg_write_table: {table}
nonlinloop_maxit: 50
nonlinloop_epsilon: 1e-10
solver_type: direct
direct_solver_type: umfpack
VELOCITY_SPACE: {vel}
PRESSURE_SPACE: {pres}
reynolds_number: {reynolds}
NSTYPE: 14
LAPLACETYPE: 0
NSE_NONLINEAR_FORM: 0
refinement_strategy: 0
refinement_tolerance: 0.5
min_fraction_to_change: 0.1
decrease_reftol_factor: 0.8
write_adaptive_mesh: {ps}
"""


def pair_label(pair):
    v, p = pair.split("/")
    return f"P{v}/P{p}"


def run_name(study, pair, nu):
    v, p = pair.split("/")
    return f"{study}_P{v}P{p}_nu{nu:.0e}".replace("e-0", "e-")


def expected_order(pair):
    v, p = (int(x) for x in pair.split("/"))
    return v + 0.5 if v == p else float(v)


def read_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            vals = line.split()
            rows.append({k: float(x) for k, x in zip(TABLE_COLUMNS, vals)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run_one(args, study, pair, nu):
    cfg = STUDIES[study]
    name = run_name(study, pair, nu)
    workdir = os.path.join(args.outdir, name)
    table = os.path.join(workdir, "table.txt")
    if os.path.exists(table) and not args.force:
        print(f"  {name}: table exists, skipping")
        return read_table(table)
    os.makedirs(workdir, exist_ok=True)
    v, p = pair.split("/")
    levels = args.levels if args.levels is not None else cfg["levels"]
    dat = DAT_TEMPLATE.format(
        name=name, initial=cfg["initial"], levels=levels,
        uniform="true" if cfg["uniform"] else "false",
        problem_type=cfg["problem_type"], example=cfg["example"],
        vtk="true" if args.vtk else "false", vtk_dir="vtk",
        delta0=args.delta0, mu0=args.mu0, sigma0=cfg["sigma0"],
        table="table.txt", vel=v, pres=p, reynolds=f"{1.0/nu:.6e}",
        ps="true" if (not cfg["uniform"] and args.ps) else "false")
    dat_file = os.path.join(workdir, name + ".dat")
    with open(dat_file, "w") as f:
        f.write(dat)
    exe = os.path.abspath(args.exe)
    print(f"  {name}: running ({pair_label(pair)}, nu = {nu:g}, {levels} levels) ...",
          end="", flush=True)
    t0 = time.time()
    with open(os.path.join(workdir, "stdout.log"), "w") as log:
        try:
            subprocess.run([exe, name + ".dat"], cwd=workdir, stdout=log,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=args.timeout)
        except subprocess.CalledProcessError as e:
            print(f" FAILED (exit code {e.returncode}), see {log.name}")
            return None
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT after {args.timeout} s")
            return None
    print(f" done in {time.time()-t0:.0f} s")
    if not os.path.exists(table):
        print(f"    no table written, see {workdir}/stdout.log")
        return None
    rows = read_table(table)
    write_csv(os.path.join(workdir, "table.csv"), rows)
    return rows


def truncate_by_dofs(rows, max_dofs):
    return [r for r in rows if r["dofs"] <= max_dofs]


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_study(args, study, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = STUDIES[study]
    pairs = [p for p in args.pairs if any(k[0] == p for k in results)]
    nus = sorted({k[1] for k in results}, reverse=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))

    def grid():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.4*nrows),
                                 squeeze=False)
        return fig, axes.ravel()

    def finish(fig, axes, name):
        for ax in axes[len(pairs):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", fontsize=8,
                       ncol=min(len(labels), 6), frameon=False)
        fig.suptitle(cfg["title"])
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        out = os.path.join(args.outdir, f"{study}_{name}.{args.format}")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")

    # ---- quantity vs DoFs with reference slope: error, estimator
    for key, ylabel, fname in [("error_spg", r"$\|(u-u_h,p-p_h)\|_{spg}$", "error"),
                               ("eta", r"$\eta$", "estimator")]:
        fig, axes = grid()
        for ax, pair in zip(axes, pairs):
            for i, nu in enumerate(nus):
                rows = results.get((pair, nu))
                if not rows:
                    continue
                d = np.array([r["dofs"] for r in rows])
                y = np.array([r[key] for r in rows])
                ax.loglog(d, y, "o-", ms=3, color=colors[i % 10],
                          label=rf"$\nu = {nu:g}$")
            if cfg["uniform"] and rows:
                k = expected_order(pair)
                d = np.array([r["dofs"] for r in rows])
                # reference line through the last point: O(h^k) = O(DoFs^{-k/2})
                y_ref = y[-1] * (d / d[-1]) ** (-k/2)
                ax.loglog(d, y_ref, "k--", lw=1, label=rf"$O(h^{{{k:g}}})$")
            ax.set_title(pair_label(pair))
            ax.set_xlabel("DoFs")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        finish(fig, axes, fname)

    # ---- effectivity index
    fig, axes = grid()
    for ax, pair in zip(axes, pairs):
        for i, nu in enumerate(nus):
            rows = results.get((pair, nu))
            if not rows:
                continue
            d = [r["dofs"] for r in rows]
            e = [r["effectivity"] for r in rows]
            ax.semilogx(d, e, "o-", ms=3, color=colors[i % 10],
                        label=rf"$\nu = {nu:g}$")
        ax.set_title(pair_label(pair))
        ax.set_xlabel("DoFs")
        ax.set_ylabel("effectivity index")
        ax.grid(True, which="both", alpha=0.3)
    finish(fig, axes, "effectivity")

    # ---- contributions of the estimator, for the viscosity closest to 1e-5
    nu_c = min(nus, key=lambda x: abs(np.log10(x) + 5)) if nus else None
    if nu_c is not None:
        fig, axes = grid()
        parts = [("eta_res", r"$\eta_{K,res}$"), ("eta_div", r"$\eta_{K,div}$"),
                 ("eta_F", r"$\eta_F$"), ("eta_delta", r"$\eta_{K,\delta}$"),
                 ("eta_mu", r"$\eta_{K,\mu}$")]
        for ax, pair in zip(axes, pairs):
            rows = results.get((pair, nu_c))
            if not rows:
                continue
            d = [r["dofs"] for r in rows]
            for i, (key, lab) in enumerate(parts):
                ax.loglog(d, [r[key] for r in rows], "o-", ms=3,
                          color=colors[i], label=lab)
            ax.set_title(rf"{pair_label(pair)}, $\nu = {nu_c:g}$")
            ax.set_xlabel("DoFs")
            ax.set_ylabel("estimated value")
            ax.grid(True, which="both", alpha=0.3)
        finish(fig, axes, "contributions")

    # ---- summary text table
    out = os.path.join(args.outdir, f"{study}_summary.txt")
    with open(out, "w") as f:
        f.write(f"{cfg['title']}\n")
        for pair in pairs:
            for nu in nus:
                rows = results.get((pair, nu))
                if not rows:
                    continue
                f.write(f"\n{pair_label(pair)}, nu = {nu:g}\n")
                f.write("level   cells    dofs       error_spg      eta   "
                        "effectivity  rate_err\n")
                for r in rows:
                    f.write(f"{int(r['level']):5d} {int(r['cells']):7d} "
                            f"{int(r['dofs']):8d} {r['error_spg']:12.4e} "
                            f"{r['eta']:10.4e} {r['effectivity']:8.3f} "
                            f"{r['rate_err']:7.2f}\n")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default="./parMooN_oseen2d_spg_adaptive_SEQUENTIAL",
                    help="path to the ParMooN executable")
    ap.add_argument("--outdir", default="spg_results",
                    help="directory for input files, logs, tables and figures")
    ap.add_argument("--study", nargs="+", default=["ex41"],
                    choices=sorted(STUDIES), help="which studies to run")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                    help="velocity/pressure orders, e.g. 2/1 2/2 (default: all five)")
    ap.add_argument("--nu", nargs="+", type=float, default=None,
                    help="viscosities (default depends on the study)")
    ap.add_argument("--levels", type=int, default=None,
                    help="number of refinement levels (default depends on the study)")
    ap.add_argument("--max-dofs", type=float, default=None,
                    help="drop levels with more DoFs from the plots")
    ap.add_argument("--delta0", type=float, default=0.5)
    ap.add_argument("--mu0", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=3600,
                    help="time limit per run in seconds")
    ap.add_argument("--vtk", action="store_true", help="write VTK output")
    ap.add_argument("--ps", action="store_true",
                    help="write PostScript files of the adaptive meshes")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--force", action="store_true", help="rerun existing runs")
    ap.add_argument("--plot-only", action="store_true",
                    help="do not run anything, only plot existing tables")
    args = ap.parse_args()

    if not args.plot_only and not os.access(args.exe, os.X_OK):
        sys.exit(f"executable {args.exe} not found; use --exe")
    os.makedirs(args.outdir, exist_ok=True)

    for study in args.study:
        nus = args.nu if args.nu else DEFAULT_NU[study]
        print(f"== {STUDIES[study]['title']}")
        results = {}
        for pair, nu in itertools.product(args.pairs, nus):
            name = run_name(study, pair, nu)
            table = os.path.join(args.outdir, name, "table.txt")
            if args.plot_only:
                rows = read_table(table) if os.path.exists(table) else None
            else:
                rows = run_one(args, study, pair, nu)
            if rows:
                if args.max_dofs:
                    rows = truncate_by_dofs(rows, args.max_dofs)
                results[(pair, nu)] = rows
        if results:
            plot_study(args, study, results)
        else:
            print("  no results")


if __name__ == "__main__":
    main()
