#!/usr/bin/env python3
"""
Run several simulations with the time-dependent adaptive programs

  parMooN_tcd2d_adaptive_tr_SEQUENTIAL     (--program tcd:  de Frutos, Garcia-Archilla, John, Novo 2014)
  parMooN_tnse2d_adaptive_spg_SEQUENTIAL   (--program tnse: adaptive SUPG/PSPG/grad-div, time-dependent NSE)
  parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL   (--program bdf2: BDF2-IMEX with CIP, Scott-Vogelius / Taylor-Hood)
  (--program tcd_old / tnse_old: the monolithic drivers with the tcd_*/tnse_* parameters)

The steady Oseen / Navier-Stokes studies (parMooN_oseen2d_spg_adaptive) are run and
plotted by spg_study.py and spg_plot.py.

from a base input file, varying one or more parameters, store the outputs and
plot the per-time tables (effectivity index, estimator and error, number of
cells, time step, Picard iterations) and the PostScript meshes written by the
programs.

A study is defined by a base .dat file and a list of parameter variations
given as key=value1,value2,... Each combination of values becomes one run in
<outdir>/<name>_<key1>-<value1>_<key2>-<value2>/. Only keys that appear in the
base file (or are valid parameters of the program) should be given.

Examples
  # Example 2 of FGJN14 (effectivity in time) on three meshes and two diffusion coefficients
  python3 transient_study.py --program tcd --base ../ParMooN-2026/data/input_files/tcd2d_adaptive_tr_ex2.dat \\
      --name fgjn_ex2 --vary refinement_n_initial_steps=5,6,7 diffusion_coefficient=1e-3,1e-6

  # FGJN14 Examples 3, 4, 5 (adaptive meshes): one run each, meshes at the output times
  python3 transient_study.py --program tcd --base ../ParMooN-2026/data/input_files/tcd2d_adaptive_tr_ex4.dat --name fgjn_ex4

  # transient Navier-Stokes Example 1: spatial convergence and effectivity for several pairs
  python3 transient_study.py --program tnse --base ../ParMooN-2026/data/input_files/tnse2d_adaptive_spg_ex1.dat \\
      --name tnse_ex1_P2P1 --vary refinement_n_initial_steps=3,4,5,6
  # spatial convergence (errors at the final time in all norms, rates w.r.t.
  # h ~ DoFs^-1/2 in <name>_summary.txt); time step small and fixed
  python3 transient_study.py --program tnse --base ../ParMooN-2026/data/input_files/tnse2d_adaptive_spg_ex1.dat \\
      --name spatial_P2P1 --set trab2_adaptive=false time_step_length=0.0025 time_end=0.05 \\
      adaptive_mesh=false output_write_vtk=false --vary refinement_n_initial_steps=2,3,4,5,6 --no-meshes
  python3 transient_study.py --program tcd --base ../ParMooN-2026/data/input_files/tcd2d_adaptive_tr_ex2.dat \\
      --name spatial_cd --set trab2_adaptive=false time_step_length=0.001 time_end=0.1 \\
      adaptive_mesh=false output_write_vtk=false --vary refinement_n_initial_steps=3,4,5,6,7 --no-meshes

  # temporal convergence of the TR step on the verification example 28 (exact
  # P2/P1 solution, so the error is the time discretization error only)
  python3 transient_study.py --program tnse --base ../ParMooN-2026/data/input_files/tnse2d_adaptive_spg_verify_time.dat \\
      --name verify_time --vary time_step_length=0.1,0.05,0.025,0.0125,0.00625 --no-meshes
  python3 transient_study.py --program tnse --base ../ParMooN-2026/data/input_files/tnse2d_adaptive_spg_ex1.dat \\
      --name tnse_ex1_pairs --set refinement_n_initial_steps=5 --vary VELOCITY_SPACE/PRESSURE_SPACE=2/1,2/2,1/1,3/2

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
    # adaptive SUPG for evolutionary convection-diffusion (de Frutos et al. 2014),
    # class-based driver (TimeConvectionDiffusionTR + AdaptiveTimeLoop)
    "tcd": dict(exe="parMooN_tcd2d_adaptive_tr_SEQUENTIAL", table_key="tcd_write_table"),
    # adaptive SUPG/PSPG/grad-div for time-dependent Navier-Stokes, class-based driver
    "tnse": dict(exe="parMooN_tnse2d_adaptive_spg_SEQUENTIAL", table_key="tnse_write_table"),
    # BDF2-IMEX with CIP (Scott-Vogelius) / Taylor-Hood
    "bdf2": dict(exe="parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", table_key="bdf2_write_table"),
    # the monolithic drivers kept for reference (parameters tcd_*/tnse_*)
    "tcd_old": dict(exe="parMooN_tcd2d_supg_adaptive_SEQUENTIAL", table_key="tcd_write_table"),
    "tnse_old": dict(exe="parMooN_tnse2d_spg_adaptive_SEQUENTIAL", table_key="tnse_write_table"),
}


# ---------------------------------------------------------------------------
# input files
# ---------------------------------------------------------------------------
def read_dat(path):
    lines = open(path).read().splitlines()
    return lines


def set_param(lines, key, value):
    """replace 'key: ...' or append it; returns the new lines"""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*)$")
    out, done = [], False
    for ln in lines:
        m = pat.match(ln)
        if m:
            if not done:
                # ParMooN checks the type: keep a floating point parameter a
                # floating point literal ("100" would be read as an integer)
                old = m.group(1).strip()
                v = str(value).strip()
                if re.fullmatch(r"[-+]?\d+", v) and re.search(r"[.eE]", old) \
                        and re.fullmatch(r"[-+]?[\d.]+([eE][-+]?\d+)?", old):
                    v += ".0"
                out.append(f"{key}: {v}")
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


def resolve_mesh_path(path, base):
    """absolute path of a relative geo_file entry: relative to the current
    directory, to the base input file, or to the repository (data/mesh/...);
    None if it is a built-in mesh name or cannot be found"""
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path, os.path.join(os.path.dirname(os.path.abspath(base)), path)]
    if "data/mesh/" in path:
        candidates.append(os.path.join(repo, path[path.index("data/mesh/"):]))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def fmt_rate(x):
    """rate with two digits, '---' if there is no previous run or the rate is
    not defined (same mesh, zero error, ...)"""
    if x is None or not np.isfinite(x):
        return "---"
    return f"{x:.2f}"


def rate_between(e_prev, e, q):
    """log(e_prev/e)/q, None if not defined"""
    if e_prev is None or e is None or e_prev <= 0 or e <= 0 or q == 0:
        return None
    return np.log(e_prev/e)/q


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
                err = column(tab, "error", "error_supg", "error_spg")
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
        err = column(tab, "error", "error_supg", "error_spg")
        eta = column(tab, "eta")
        eff = column(tab, "effectivity")
        dofs = column(tab, "dofs")
        cells = column(tab, "cells")
        if err is None or eta is None or dofs is None:
            return
        tt = column(tab, "time")
        # the initial state and the start-up phase of the time integrator
        # (interpolated data, error ~ 0, meaningless effectivity): exclude
        # the first 2% of the time interval from the effectivity statistics
        if tt is not None:
            ok = (err > 0) & (tt > tt[0] + 0.02*(tt[-1] - tt[0]))
        else:
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
            rate_e = rate_eta = None
            if i > 0:
                q = 0.5*np.log(r["dofs"]/rows[i-1]["dofs"])
                rate_e = rate_between(rows[i-1]["err"], r["err"], q)
                rate_eta = rate_between(rows[i-1]["eta"], r["eta"], q)
            f.write(f"{r['label']}  {int(r['cells'])}  {int(r['dofs'])}  {r['err']:.4e}  {r['eta']:.4e}"
                    f"  {r['eff_min']:.3f}  {r['eff_mean']:.3f}  {r['eff_max']:.3f}  {r['steps']}"
                    f"  {fmt_rate(rate_e)}  {fmt_rate(rate_eta)}\n")
        # errors at the final time in all norms the program writes
        norms = [n for n in ("error", "eta", "L2(u)", "H1(u)", "L2(div_u)", "L2(p)")
                 if all(n in tab for _, tab in runs)]
        if norms:
            f.write("\n# errors at the final time and rates w.r.t. h ~ DoFs^-1/2"
                    " ('error' is the norm of the method: SUPG / SPG norm)\n")
            f.write("# run  dofs  " + "  ".join(f"{n}  rate" for n in norms) + "\n")
            prev = None
            for label, tab in runs:
                dofs = column(tab, "dofs")[-1]
                vals = [tab[n][-1] for n in norms]
                line = f"{label}  {int(dofs)}"
                for j, v in enumerate(vals):
                    rate = None
                    if prev is not None:
                        rate = rate_between(prev[1][j], v, 0.5*np.log(dofs/prev[0]))
                    line += f"  {v:.4e}  {fmt_rate(rate)}"
                f.write(line + "\n")
                prev = (dofs, vals)
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


def temporal_summary(args, runs):
    """Temporal convergence for series over time_step_length (fixed steps,
    trab2_adaptive false, same mesh): error and eta at the final time and the
    maximum over time, with rates w.r.t. the step length between consecutive
    runs, and a log-log figure. Written only if every run label contains
    time_step_length=<value>."""
    import re
    rows = []
    for label, tab in runs:
        m = re.search(r"time_step_length=([0-9.eE+-]+)", label)
        if m is None:
            return
        err = column(tab, "error", "error_supg", "error_spg")
        eta = column(tab, "eta")
        if err is None or eta is None:
            return
        rows.append(dict(label=label, dt=float(m.group(1)), err_T=err[-1],
                         err_max=err.max(), eta_T=eta[-1], steps=len(err)))
    if len(rows) < 2:
        return
    rows.sort(key=lambda r: -r["dt"])
    out = os.path.join(args.outdir, f"{args.name}_temporal.txt")
    with open(out, "w") as f:
        f.write("# dt  error(T)  max_t error  eta(T)  steps  rate_error(T)  rate_max_error"
                "  rate_eta(T)   (rates w.r.t. dt between consecutive runs)\n")
        for i, r in enumerate(rows):
            rates = [None]*3
            if i > 0:
                q = np.log(rows[i-1]["dt"]/r["dt"])
                for j, key in enumerate(("err_T", "err_max", "eta_T")):
                    rates[j] = rate_between(rows[i-1][key], r[key], q)
            f.write(f"{r['dt']:.6g}  {r['err_T']:.4e}  {r['err_max']:.4e}  {r['eta_T']:.4e}"
                    f"  {r['steps']}  " + "  ".join(fmt_rate(x) for x in rates) + "\n")
    print(f"wrote {out}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = np.array([r["dt"] for r in rows])
    e = np.array([r["err_T"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(dt, e, "o-", label="error at $T$")
    ax.loglog(dt, [r["err_max"] for r in rows], "v:", label=r"$\max_t$ error")
    ax.loglog(dt, [r["eta_T"] for r in rows], "s--", label=r"$\eta$ at $T$")
    ax.loglog(dt, e[-1]*(dt/dt[-1])**2, ":", color="gray", lw=0.8, label="slope 2")
    ax.set_xlabel(r"$\Delta t$"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(args.name)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"{args.name}_temporal.{args.format}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


CONTRIB = ["eta_res", "eta_div", "eta_F", "eta_delta", "eta_mu"]
CONTRIB_LABEL = {"eta_res": r"$\eta_{res}$", "eta_div": r"$\eta_{div}$",
                 "eta_F": r"$\eta_F$", "eta_delta": r"$\eta_\delta$",
                 "eta_mu": r"$\eta_\mu$"}
ERRPART = ["err_grad", "err_div", "err_supg"]
ERRPART_LABEL = {"err_grad": r"$\nu^{1/2}|e_u|_1$",
                 "err_div": r"$(\sum\mu_K\|\nabla\cdot e_u\|^2)^{1/2}$",
                 "err_supg": r"$(\sum\delta_K\|(b\cdot\nabla)e_u+\nabla e_p\|^2)^{1/2}$"}
NORMS = ["L2(u)", "H1(u)", "L2(div_u)", "L2(p)"]


def convergence_study(args, runs, combos):
    """Convergence versus the number of degrees of freedom.

    Runs which differ only in the x parameter (refinement level, or the
    adaptive tolerance) form one curve; every value of the other varied
    parameters (viscosity, pair, ...) gives one curve. Per run one value is
    taken: at the final time, or the maximum over time (--measure). Writes
    <name>_convergence.txt (values and rates w.r.t. h ~ DoFs^-1/2) and the
    figures <name>_convergence (error, eta, effectivity with its range over
    time, cells), <name>_contributions (estimator contributions and error
    parts per curve) and <name>_norms (L2(u), H1(u), L2(div u), L2(p))."""
    keys = [k for k, _ in combos[0]] if combos and combos[0] else []
    if not keys:
        return
    xkey = args.x_key
    if xkey is None:
        for cand in ("refinement_n_initial_steps", "adaptive_tol1/adaptive_tol2",
                     "adaptive_tol1"):
            if cand in keys:
                xkey = cand
                break
    if xkey is None or xkey not in keys:
        return
    groups = {}
    for (label, tab), combo in zip(runs, combos):
        rest = tuple((k, v) for k, v in combo if k != xkey)
        groups.setdefault(rest, []).append(tab)
    if max(len(g) for g in groups.values()) < 2:
        return

    def value(tab, key):
        v = tab.get(key)
        if v is None:
            return None
        tt = tab["time"]
        if args.measure == "final":
            return v[-1]
        ok = tt > tt[0] + 0.02*(tt[-1] - tt[0])
        return v[ok].max() if ok.any() else v[-1]

    def dofs_of(tab):
        d = tab["dofs"]
        return {"final": d[-1], "mean": d.mean(), "max": d.max()}[args.dofs]

    def eff_range(tab):
        tt, eff = tab["time"], tab.get("effectivity")
        err = tab.get("error")
        if eff is None or err is None:
            return (np.nan,)*3
        ok = (err > 0) & (tt > tt[0] + 0.02*(tt[-1] - tt[0]))
        if not ok.any():
            return (np.nan,)*3
        return eff[ok].min(), eff[ok].mean(), eff[ok].max()

    def group_label(rest):
        return ", ".join(f"{k.replace('VELOCITY_SPACE/PRESSURE_SPACE', 'pair')}={v}"
                         for k, v in rest) or args.name

    curves = []
    for rest, tabs in groups.items():
        tabs = sorted(tabs, key=dofs_of)
        curves.append((group_label(rest), tabs))

    # ---- table
    out = os.path.join(args.outdir, f"{args.name}_convergence.txt")
    cols = ["error", "eta"] + [c for c in NORMS + CONTRIB + ERRPART
                               if all(c in t for _, ts in curves for t in ts)]
    with open(out, "w") as f:
        f.write(f"# convergence study, value per run: {args.measure} time, DoFs: "
                f"{args.dofs}; rates w.r.t. h ~ DoFs^-1/2; effectivity at the "
                f"{args.measure} time and its min/mean/max over time\n")
        for glabel, tabs in curves:
            f.write(f"\n# {glabel}\n# cells  dofs  " +
                    "  ".join(f"{c}  rate" for c in cols) +
                    "  effectivity  eff_min  eff_mean  eff_max\n")
            prev = None
            for tab in tabs:
                d = dofs_of(tab)
                vals = [value(tab, c) for c in cols]
                line = f"{int(tab['cells'][-1])}  {int(round(d))}"
                for j, v in enumerate(vals):
                    r = None
                    if prev is not None:
                        r = rate_between(prev[1][j], v, 0.5*np.log(d/prev[0]))
                    line += f"  {v:.4e}  {fmt_rate(r)}"
                e = value(tab, "eta"), value(tab, "error")
                eff = e[0]/e[1] if e[1] and e[1] > 0 else np.nan
                lo, me, hi = eff_range(tab)
                line += f"  {eff:.3f}  {lo:.3f}  {me:.3f}  {hi:.3f}"
                f.write(line + "\n")
                prev = (d, vals)
    print(f"wrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    slopes = [float(x) for x in args.slopes.split(",") if x.strip()]

    def ref_lines(ax, d, y):
        d = np.asarray(d, dtype=float)
        for s_ in slopes:
            yy = y[-1]*(d/d[-1])**(-s_/2.)
            ax.loglog(d, yy, ":", color="gray", lw=0.8)
            ax.annotate(f"$h^{{{s_:g}}}$", (d[0], yy[0]), fontsize=7, color="gray")

    # ---- convergence figure
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for i, (glabel, tabs) in enumerate(curves):
        c = colors[i % len(colors)]
        d = [dofs_of(t) for t in tabs]
        err = np.array([value(t, "error") for t in tabs], dtype=float)
        eta = np.array([value(t, "eta") for t in tabs], dtype=float)
        axs[0].loglog(d, err, "o-", color=c, label=f"error, {glabel}")
        axs[0].loglog(d, eta, "s--", color=c, label=r"$\eta$" + f", {glabel}")
        if i == 0:
            ref_lines(axs[0], d, err)
        eff = eta/err
        rng = np.array([eff_range(t) for t in tabs])
        axs[1].semilogx(d, eff, "o-", color=c, label=glabel)
        axs[1].fill_between(d, rng[:, 0], rng[:, 2], color=c, alpha=0.15)
        axs[2].loglog(d, [t["cells"][-1] if args.dofs == "final" else
                          {"mean": t["cells"].mean(), "max": t["cells"].max()}[args.dofs]
                          for t in tabs], "o-", color=c, label=glabel)
    axs[0].set_title(f"error and estimator ({args.measure} time)")
    axs[1].set_title("effectivity index (band: range over time)")
    axs[2].set_title("number of cells")
    for ax in axs:
        ax.set_xlabel("degrees of freedom")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=7)
    fig.suptitle(args.name)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"{args.name}_convergence.{args.format}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- contributions of the estimator and parts of the error
    have_c = all(c in t for _, ts in curves for t in ts for c in CONTRIB)
    if have_c:
        n = len(curves)
        fig, axs = plt.subplots(1, n, figsize=(4.4*n, 3.9), squeeze=False)
        for i, (glabel, tabs) in enumerate(curves):
            ax = axs[0][i]
            d = [dofs_of(t) for t in tabs]
            ax.loglog(d, [value(t, "eta") for t in tabs], "k-", lw=2, label=r"$\eta$")
            ax.loglog(d, [value(t, "error") for t in tabs], "k--", lw=2, label="error")
            for j, cname in enumerate(CONTRIB):
                y = np.array([value(t, cname) for t in tabs], dtype=float)
                if (y > 0).any():
                    ax.loglog(d, np.where(y > 0, y, np.nan), "o-",
                              color=colors[j % len(colors)], ms=3,
                              label=CONTRIB_LABEL[cname])
            for j, cname in enumerate(ERRPART):
                if all(cname in t for t in tabs):
                    y = np.array([value(t, cname) for t in tabs], dtype=float)
                    if (y > 0).any():
                        ax.loglog(d, np.where(y > 0, y, np.nan), "v:",
                                  color=colors[(j + 5) % len(colors)], ms=3,
                                  label=ERRPART_LABEL[cname])
            ax.set_title(glabel, fontsize=9)
            ax.set_xlabel("degrees of freedom")
            ax.grid(True, which="both", alpha=0.3)
        axs[0][0].legend(fontsize=6)
        fig.suptitle(f"{args.name}: contributions of the estimator (solid) and of "
                     f"the error (dotted), {args.measure} time")
        fig.tight_layout()
        out = os.path.join(args.outdir, f"{args.name}_contributions.{args.format}")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    # ---- standard norms
    norms = [c for c in NORMS if all(c in t for _, ts in curves for t in ts)]
    if norms:
        fig, axs = plt.subplots(1, len(norms), figsize=(3.6*len(norms), 3.6),
                                squeeze=False)
        for k, nname in enumerate(norms):
            ax = axs[0][k]
            for i, (glabel, tabs) in enumerate(curves):
                d = [dofs_of(t) for t in tabs]
                y = np.array([value(t, nname) for t in tabs], dtype=float)
                ax.loglog(d, y, "o-", color=colors[i % len(colors)], label=glabel)
                if i == 0:
                    ref_lines(ax, d, y)
            ax.set_title(nname)
            ax.set_xlabel("degrees of freedom")
            ax.grid(True, which="both", alpha=0.3)
        axs[0][0].legend(fontsize=7)
        fig.suptitle(f"{args.name}: errors in standard norms, {args.measure} time")
        fig.tight_layout()
        out = os.path.join(args.outdir, f"{args.name}_norms.{args.format}")
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
    ap.add_argument("--x-key", default=None,
                    help="varied parameter along a convergence curve (default: "
                         "refinement_n_initial_steps if varied, else the varied adaptive "
                         "tolerance); all other varied parameters give one curve each")
    ap.add_argument("--measure", default="final", choices=["final", "max"],
                    help="value per run in the convergence study: at the final time or "
                         "the maximum over time (start-up excluded)")
    ap.add_argument("--dofs", default="final", choices=["final", "mean", "max"],
                    help="DoFs per run in the convergence study (adaptive meshes: "
                         "final, time-averaged or maximal number)")
    ap.add_argument("--slopes", default="1,2",
                    help="reference slopes in h (DoFs^-s/2) for the convergence plots")
    args = ap.parse_args()

    prog = PROGRAMS[args.program]
    exe = os.path.abspath(args.exe or os.path.join(".", prog["exe"]))
    os.makedirs(args.outdir, exist_ok=True)

    vary = parse_vary(args.vary)
    combos = [list(zip([k for k, _ in vary], vals))
              for vals in itertools.product(*[v for _, v in vary])] if vary else [[]]
    fixed = parse_vary(args.set)

    runs = []
    run_combos = []
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
                    if m:
                        geo = resolve_mesh_path(m.group(1).strip(), args.base)
                        if geo is not None:
                            lines[i] = f"geo_file: {geo}"
                    # ParMooN does not create the vtk output directory
                    m = re.match(r"^\s*output_directory\s*:\s*(.+)$", ln)
                    if m:
                        os.makedirs(os.path.join(run_dir, m.group(1).strip()),
                                    exist_ok=True)
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
                run_combos.append(combo)
                if not args.no_meshes:
                    plot_meshes(args, run_dir, label)
    if runs:
        plot_tables(args, runs)
        spatial_summary(args, runs)
        temporal_summary(args, runs)
        convergence_study(args, runs, run_combos)
        res = [(label, tab["_results"]) for label, tab in runs if "_results" in tab]
        if res:
            out = os.path.join(args.outdir, f"{args.name}_results.txt")
            with open(out, "w") as f:
                f.write("# run  E_0  E_p  D  L2(u(T))  H1(u(T))  max_u  rate_E_0  rate_E_p"
                        "   (rates w.r.t. the previous run: w.r.t. time_step_length if it is"
                        " varied, otherwise a factor 2 per run is assumed, e.g. one"
                        " refinement level)\n")
                prev = None
                for label, r in res:
                    r_e0 = r_ep = None
                    if prev is not None:
                        m0 = re.search(r"time_step_length=([0-9.eE+-]+)", prev[0])
                        m1 = re.search(r"time_step_length=([0-9.eE+-]+)", label)
                        q = (np.log(float(m0.group(1))/float(m1.group(1)))
                             if m0 and m1 else np.log(2.))
                        r_e0 = rate_between(prev[1][0], r[0], q)
                        r_ep = rate_between(prev[1][1], r[1], q)
                    f.write(f"{label}  " + "  ".join(f"{x:.4e}" for x in r)
                            + f"  {fmt_rate(r_e0)}  {fmt_rate(r_ep)}\n")
                    prev = (label, r)
            print(f"wrote {out}")
    else:
        print("no tables to plot")


if __name__ == "__main__":
    main()
