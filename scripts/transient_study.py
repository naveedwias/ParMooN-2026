#!/usr/bin/env python3
"""
Run several simulations with the time-dependent adaptive programs

  parMooN_tcd2d_supg_adaptive_SEQUENTIAL   (de Frutos, Garcia-Archilla, John, Novo 2014)
  parMooN_tnse2d_spg_adaptive_SEQUENTIAL   (adaptive SUPG/PSPG/grad-div for time-dependent NSE)
  parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL   (BDF2-IMEX with CIP for Scott-Vogelius / Taylor-Hood)

from a base input file, varying one or more parameters, store the outputs and
plot the per-time tables (effectivity index, estimator and error, number of
cells, time step, Picard iterations) and the PostScript meshes written by the
programs.

A study is defined by a base .dat file and a list of parameter variations
given as key=value1,value2,... Each combination of values becomes one run in
<outdir>/<name>_<key1>-<value1>_<key2>-<value2>/. Only keys that appear in the
base file (or are valid parameters of the program) should be given.

Examples
  # Example 2 of FGJN14 on three meshes and two diffusion coefficients
  python3 transient_study.py --program tcd --base ../ParMooN-2026/data/input_files/tcd2d_fgjn14_ex2.dat \\
      --name fgjn_ex2 --vary refinement_n_initial_steps=5,6,7 diffusion_coefficient=1e-3,1e-6

  # transient Navier-Stokes Example 1 for several pairs and viscosities
  python3 transient_study.py --program tnse --base ../ParMooN-2026/data/input_files/tnse2d_spg_ex1.dat \\
      --name tnse_ex1 --vary VELOCITY_SPACE/PRESSURE_SPACE=2/1,2/2,1/1 reynolds_number=100,1000

  # BDF2-IMEX, Table 1: SV-CIP on n = 4, 8, 16 with dt = h^(3/2)
  python3 transient_study.py --program bdf2 --base ../ParMooN-2026/data/input_files/tnse2d_bdf2imex_cip.dat \
      --name sv_cip --vary refinement_n_initial_steps/time_step_length=3/0.125,4/0.0441942,5/0.015625
  # (the special key a/b=v1/w1,v2/w2 sets two parameters at once); the results
  # E_0, E_p, D of all runs are collected in <name>_results.txt with rates

  # only plot existing results
  python3 transient_study.py --program tnse --name tnse_ex1 --plot-only

The special key VELOCITY_SPACE/PRESSURE_SPACE sets both orders at once. Runs
whose table file exists are skipped unless --force is given; --dry-run only
writes the input files.
"""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time

import numpy as np

PROGRAMS = {
    "tcd": dict(exe="parMooN_tcd2d_supg_adaptive_SEQUENTIAL", table_key="tcd_write_table"),
    "tnse": dict(exe="parMooN_tnse2d_spg_adaptive_SEQUENTIAL", table_key="tnse_write_table"),
    "bdf2": dict(exe="parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", table_key="bdf2_write_table"),
}


# ---------------------------------------------------------------------------
# input files
# ---------------------------------------------------------------------------
def read_dat(path):
    lines = open(path).read().splitlines()
    return lines


def set_param(lines, key, value):
    """replace 'key: ...' or append it; returns the new lines"""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:")
    out, done = [], False
    for ln in lines:
        if pat.match(ln):
            if not done:
                out.append(f"{key}: {value}")
                done = True
            # drop duplicates
        else:
            out.append(ln)
    if not done:
        out.append(f"{key}: {value}")
    return out


def parse_vary(items):
    """['a=1,2', 'b=x,y'] -> [('a', ['1','2']), ('b', ['x','y'])]"""
    vary = []
    for it in items:
        if "=" not in it:
            sys.exit(f"--vary entries must be key=v1,v2,...: {it}")
        key, vals = it.split("=", 1)
        vary.append((key.strip(), [v.strip() for v in vals.split(",") if v.strip()]))
    return vary


def run_name(name, combo):
    parts = [name]
    for key, val in combo:
        k = key.replace("/", "_").replace("VELOCITY_SPACE_PRESSURE_SPACE", "pair")
        parts.append(f"{k}-{val.replace('/', '')}")
    return "_".join(parts)


def apply_combo(lines, combo):
    for key, val in combo:
        if "/" in key:
            # several parameters set at once: a/b=v/w
            keys = key.split("/")
            vals = val.split("/")
            if len(keys) != len(vals):
                sys.exit(f"--vary {key}={val}: number of values does not match")
            for k, v in zip(keys, vals):
                lines = set_param(lines, k, v)
        else:
            lines = set_param(lines, key, val)
    return lines


# ---------------------------------------------------------------------------
# tables (both programs write a header line starting with '#')
# ---------------------------------------------------------------------------
def read_table(path):
    header = None
    rows = []
    results = None
    for line in open(path):
        if line.startswith("# RESULTS"):
            results = [float(x) for x in line.split()[2:]]
            continue
        if line.startswith("#"):
            if header is None:
                header = line[1:].split()
            continue
        if line.strip():
            rows.append([float(x) for x in line.split()])
    if header is None or not rows:
        return None
    data = np.array(rows)
    tab = {h: data[:, i] for i, h in enumerate(header) if i < data.shape[1]}
    if results is not None:
        tab["_results"] = results  # E_0 E_p D L2(u(T)) H1(u(T)) max_u
    return tab


def column(tab, *names):
    for n in names:
        if n in tab:
            return tab[n]
    return None


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_tables(args, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    if any(column(t, "L2(u)") is not None for _, t in runs):
        # BDF2-IMEX tables
        panels = [("L2(u)", r"$\|u-u_h\|_0$", True), ("L2(p)", r"$\|p-p_h\|_0$", True),
                  ("div(u)", r"$\|\nabla\cdot u_h\|_0$", True), ("F_m", r"energy $F^m$", False)]
    else:
        panels = [("effectivity", "effectivity index", False),
                  ("error", r"$\eta$ (solid), error (dashed)", True),
                  ("cells", "number of cells", False),
                  ("dt", "time step", True)]
        if any(column(t, "picard") is not None for _, t in runs):
            panels.append(("picard", "Picard iterations", False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8*len(panels), 3.3), squeeze=False)
    axes = axes.ravel()
    for i, (label, tab) in enumerate(runs):
        t = column(tab, "time")
        c = colors[i % 10]
        for ax, (key, ylabel, log) in zip(axes, panels):
            if key == "error":
                eta = column(tab, "eta")
                err = column(tab, "error_supg", "error_spg")
                ax.plot(t, eta, "-", color=c, label=label)
                if err is not None and np.all(err > 0):
                    ax.plot(t, err, "--", color=c)
            else:
                y = column(tab, key)
                if y is None:
                    continue
                if key == "effectivity" and not np.all(y > 0):
                    continue
                if log and not np.all(y > 0):
                    y = np.where(y > 0, y, np.nan)
                ax.plot(t, y, "-", color=c, label=label)
            ax.set_xlabel("t")
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    if not h:
        h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), frameon=False, fontsize=8)
    fig.suptitle(args.name)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    out = os.path.join(args.outdir, f"{args.name}_tables.{args.format}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_meshes(args, run_dir, label):
    """draw all mesh_*.ps files of a run into one figure"""
    files = sorted(f for f in os.listdir(run_dir) if f.startswith("mesh_") and f.endswith(".ps"))
    if not files:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    n = len(files)
    ncols = min(4, n)
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.5*nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), files):
        txt = open(os.path.join(run_dir, f)).read()
        cells = []
        for block in re.findall(r"newpath\n((?:[-\d.]+ [-\d.]+ [ML]\n)+)closepath", txt):
            pts = [(float(a), float(b)) for a, b in re.findall(r"([-\d.]+) ([-\d.]+) [ML]", block)]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            cells.append(pts)
        allp = np.array([p for c in cells for p in c])
        x0, y0 = allp.min(axis=0)
        scale = max((allp.max(axis=0) - allp.min(axis=0)))
        segs = []
        for c in cells:
            c = [((x-x0)/scale, (y-y0)/scale) for x, y in c]
            for i in range(len(c)):
                segs.append([c[i], c[(i+1) % len(c)]])
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.2))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"{f[5:-3]} ({len(cells)} cells)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(label)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(run_dir, f"meshes.{args.format}")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")



def spatial_summary(args, runs):
    """One line per run (mesh): max/final DoFs and cells, max_t error, max_t eta,
    min/mean/max effectivity index, number of steps; spatial convergence rates
    between consecutive runs computed from the DoFs (rate w.r.t. h ~ DoFs^-1/2)
    and a log-log figure of max_t error and max_t eta versus DoFs. Meant for
    series over refinement_n_initial_steps (tcd and tnse programs)."""
    rows = []
    for label, tab in runs:
        err = column(tab, "error_supg", "error_spg")
        eta = column(tab, "eta")
        eff = column(tab, "effectivity")
        dofs = column(tab, "dofs")
        cells = column(tab, "cells")
        if err is None or eta is None or dofs is None:
            return
        ok = err > 0
        rows.append(dict(label=label, dofs=dofs[-1], cells=cells[-1] if cells is not None else 0,
                         err=err[ok].max() if ok.any() else 0., eta=eta.max(),
                         eff_min=eff[ok].min() if ok.any() else 0.,
                         eff_mean=eff[ok].mean() if ok.any() else 0.,
                         eff_max=eff[ok].max() if ok.any() else 0., steps=len(err)))
    if not rows:
        return
    out = os.path.join(args.outdir, f"{args.name}_summary.txt")
    with open(out, "w") as f:
        f.write("# run  cells  dofs  max_t_error  max_t_eta  eff_min  eff_mean  eff_max  steps"
                "  rate_error  rate_eta   (rates w.r.t. h ~ DoFs^-1/2 between consecutive runs)\n")
        for i, r in enumerate(rows):
            rate_e = rate_eta = float("nan")
            if i > 0 and rows[i-1]["dofs"] != r["dofs"]:
                q = 0.5*np.log(r["dofs"]/rows[i-1]["dofs"])
                if rows[i-1]["err"] > 0 and r["err"] > 0:
                    rate_e = np.log(rows[i-1]["err"]/r["err"])/q
                if rows[i-1]["eta"] > 0 and r["eta"] > 0:
                    rate_eta = np.log(rows[i-1]["eta"]/r["eta"])/q
            f.write(f"{r['label']}  {int(r['cells'])}  {int(r['dofs'])}  {r['err']:.4e}  {r['eta']:.4e}"
                    f"  {r['eff_min']:.3f}  {r['eff_mean']:.3f}  {r['eff_max']:.3f}  {r['steps']}"
                    f"  {rate_e:.2f}  {rate_eta:.2f}\n")
    print(f"wrote {out}")
    if len(rows) > 1 and all(r["err"] > 0 for r in rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.array([r["dofs"] for r in rows])
        e = np.array([r["err"] for r in rows])
        et = np.array([r["eta"] for r in rows])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
        ax1.loglog(d, e, "o-", label=r"$\max_t$ error")
        ax1.loglog(d, et, "s--", label=r"$\max_t\, \eta$")
        for k in (1, 2, 3):
            ax1.loglog(d, e[-1]*(d/d[-1])**(-k/2), ":", color="gray", lw=0.8)
        ax1.set_xlabel("DoFs"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.semilogx(d, [r["eff_mean"] for r in rows], "o-", label="mean")
        ax2.semilogx(d, [r["eff_min"] for r in rows], "v:", label="min")
        ax2.semilogx(d, [r["eff_max"] for r in rows], "^:", label="max")
        ax2.set_xlabel("DoFs"); ax2.set_ylabel("effectivity index"); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.suptitle(args.name)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(args.outdir, f"{args.name}_spatial.{args.format}")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", choices=sorted(PROGRAMS), required=True)
    ap.add_argument("--exe", default=None, help="path to the executable (default: ./<program exe>)")
    ap.add_argument("--base", default=None, help="base input file")
    ap.add_argument("--name", required=True, help="name of the study")
    ap.add_argument("--outdir", default="transient_results")
    ap.add_argument("--vary", nargs="*", default=[], help="key=v1,v2,... (several allowed)")
    ap.add_argument("--set", nargs="*", default=[], help="key=value fixed for all runs")
    ap.add_argument("--timeout", type=float, default=36000)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-meshes", action="store_true", help="do not draw the PostScript meshes")
    args = ap.parse_args()

    prog = PROGRAMS[args.program]
    exe = os.path.abspath(args.exe or os.path.join(".", prog["exe"]))
    os.makedirs(args.outdir, exist_ok=True)

    vary = parse_vary(args.vary)
    combos = [list(zip([k for k, _ in vary], vals))
              for vals in itertools.product(*[v for _, v in vary])] if vary else [[]]
    fixed = parse_vary(args.set)

    runs = []
    for combo in combos:
        name = run_name(args.name, combo)
        run_dir = os.path.join(args.outdir, name)
        table = os.path.join(run_dir, "table.txt")
        label = ", ".join(f"{k.replace('VELOCITY_SPACE/PRESSURE_SPACE', 'pair')}={v}"
                          for k, v in combo) or args.name
        if not args.plot_only:
            if args.base is None:
                sys.exit("--base is required unless --plot-only is given")
            if os.path.exists(table) and not args.force:
                print(f"{name}: table exists, skipping")
            else:
                os.makedirs(run_dir, exist_ok=True)
                lines = read_dat(args.base)
                lines = apply_combo(lines, [(k, v[0]) for k, v in fixed])
                lines = apply_combo(lines, combo)
                lines = set_param(lines, prog["table_key"], "table.txt")
                lines = set_param(lines, "outfile", name + ".out")
                lines = set_param(lines, "output_basename", name)
                # make relative geo_file paths work from the run directory
                for i, ln in enumerate(lines):
                    m = re.match(r"^\s*geo_file\s*:\s*(.+)$", ln)
                    if m and os.path.exists(m.group(1)):
                        lines[i] = f"geo_file: {os.path.abspath(m.group(1))}"
                dat = os.path.join(run_dir, name + ".dat")
                open(dat, "w").write("\n".join(lines) + "\n")
                if args.dry_run:
                    print(f"{name}: wrote {dat}")
                    continue
                print(f"{name}: running ...", end="", flush=True)
                t0 = time.time()
                with open(os.path.join(run_dir, "stdout.log"), "w") as log:
                    try:
                        subprocess.run([exe, name + ".dat"], cwd=run_dir, stdout=log,
                                       stderr=subprocess.STDOUT, check=True,
                                       timeout=args.timeout)
                        print(f" done in {time.time()-t0:.0f} s")
                    except subprocess.CalledProcessError as e:
                        print(f" FAILED (exit code {e.returncode}), see {log.name}")
                    except subprocess.TimeoutExpired:
                        print(f" TIMEOUT after {args.timeout} s")
        if os.path.exists(table):
            tab = read_table(table)
            if tab is not None:
                runs.append((label, tab))
                if not args.no_meshes:
                    plot_meshes(args, run_dir, label)
    if runs:
        plot_tables(args, runs)
        spatial_summary(args, runs)
        res = [(label, tab["_results"]) for label, tab in runs if "_results" in tab]
        if res:
            out = os.path.join(args.outdir, f"{args.name}_results.txt")
            with open(out, "w") as f:
                f.write("# run  E_0  E_p  D  L2(u(T))  H1(u(T))  max_u  [rates of E_0, E_p w.r.t. the previous run, factor 2 assumed]\n")
                prev = None
                for label, r in res:
                    rate = ""
                    if prev is not None and prev[0] > 0 and r[0] > 0:
                        rate = f"  {np.log2(prev[0]/r[0]):.2f} {np.log2(prev[1]/r[1]) if prev[1] > 0 and r[1] > 0 else float('nan'):.2f}"
                    f.write(f"{label}  " + "  ".join(f"{x:.4e}" for x in r) + rate + "\n")
                    prev = r
            print(f"wrote {out}")
    else:
        print("no tables to plot")


if __name__ == "__main__":
    main()
