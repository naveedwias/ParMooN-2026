#!/usr/bin/env python3
"""
Post-processing of a Taylor-Green vortex run (Example_TimeNSE2D 35) of the
program tnse2d_adaptive_spg.

Exact solution (see include/Examples/TNSE_2D/TNSE_taylor_green.h), F = exp(-2 nu k^2 t):
  u = A sin(kx) cos(ky) F,  v = -A cos(kx) sin(ky) F,  p = A^2/4 (cos 2kx + cos 2ky) F^2,
  psi = (A/k) sin(kx) sin(ky) F  (u = psi_y, v = -psi_x),  omega = 2 A k sin(kx) sin(ky) F,
  for k L = 2 pi:  E = pi^2 A^2/k^2 F^2,  Z = 2 pi^2 A^2 F^2,  -dE/dt = 2 nu Z = 4 nu k^2 E.

Writes, for a run directory (the input file *.dat, the table, the vtk files),
with <out> = RUN_DIR/<problem>_<example> (e.g. tnse_taylor_green, from the
input file; scripts/naming.py):
  <out>_time.pdf    energy and enstrophy relative to the exact values;
                    dissipation -dE_h/dt against 2 nu Z_h and the exact rate,
                    and the numerical dissipation -dE_h/dt - 2 nu Z_h;
                    errors and estimator over time; effectivity; vorticity
                    extrema; time step and cells
  <out>_fields.pdf  stream function, vorticity, velocity error and pressure
                    error at selected output times (discrete: colours,
                    exact: dashed contour lines)
  <out>_summary.txt measured energy decay rate against 4 nu k^2, mean
                    numerical dissipation, errors at the final time

The stream function psi_h is the P1 least-squares fit of grad psi = (-v_h, u_h)
on the output mesh with psi_h = psi on the boundary (0 for k L = 2 pi); the
vorticity is the P1 gradient of the output velocity (cellwise, averaged to
the vertices). The vtk files contain the velocity at the mesh vertices, so
psi_h and omega_h are first-order reconstructions for plotting, not the
finite element solution itself (the errors in the table are computed by
quadrature in the program).

Usage:
  python3 tgv_plots.py FOLDER --run KEY [--out NAME] [--snapshots 0,mid,last] [--format pdf]
  python3 tgv_plots.py RUN_DIR [--out NAME] ...
FOLDER is the example folder of transient_study.py (e.g. tnse_taylor_green) and KEY
a run key of its <prefix>_runs.jsonl (scripts/run_store.py; the input file and the
table are taken from the record, the vtk files from <prefix>_files/KEY/; default
<out> = <prefix>_files/KEY/<prefix>). RUN_DIR is a directory with the input file
*.dat, the table and vtk/ (a run folder of an earlier version, or the directory in
which the program was run).
"""
import argparse
import glob
import math
import os
import re

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import naming  # noqa: E402


def parse_params(text):
    prm = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*([A-Za-z_0-9]+)\s*:\s*(.*?)\s*$", ln)
        if m:
            prm[m.group(1)] = m.group(2)
    return prm


def parse_table_text(text):
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("# RESULTS")]
    hdr = lines[0].lstrip("#").split()
    data = np.array([[float(x) for x in l.split()] for l in lines[1:] if not l.startswith("#")])
    return {h: data[:, i] for i, h in enumerate(hdr) if i < data.shape[1]}


def load_stored_run(folder, key):
    """parameters, table and vtk folder of a run of transient_study.py (run_store.py)"""
    from run_store import RunStore
    prefix = os.path.basename(os.path.normpath(folder))
    store = RunStore(folder, prefix)
    rec = store.get(key)
    if rec is None:
        keys = sorted(store.records())
        raise SystemExit(f"no finished run {key} in {folder}/{prefix}_runs.jsonl"
                         + (f"; runs: {', '.join(keys)}" if keys else ""))
    prm = parse_params(rec["input"])
    prm["_prefix"] = prefix
    return prm, parse_table_text(rec["table"]), store.files_dir(key)


def read_dat(run_dir):
    dats = sorted(glob.glob(os.path.join(run_dir, "*.dat")))
    if not dats:
        raise SystemExit(f"no input file in {run_dir}")
    prm = {}
    for ln in open(dats[0]):
        m = re.match(r"^\s*([A-Za-z_0-9]+)\s*:\s*(.*?)\s*$", ln)
        if m:
            prm[m.group(1)] = m.group(2)
    prm["_prefix"] = naming.prefix_from_input(dats[0], "tnse")
    return prm


def read_table(run_dir, prm):
    cand = [os.path.join(run_dir, f"{prm['_prefix']}_table.txt"),
            os.path.join(run_dir, "table.txt")]
    if "tnse_write_table" in prm:
        cand.append(os.path.join(run_dir, prm["tnse_write_table"]))
    for f in cand:
        if os.path.exists(f):
            lines = [l for l in open(f).read().splitlines() if l.strip()]
            hdr = lines[0].lstrip("#").split()
            data = np.array([[float(x) for x in l.split()] for l in lines[1:]])
            return {h: data[:, i] for i, h in enumerate(hdr) if i < data.shape[1]}
    raise SystemExit(f"no table in {run_dir}")


def read_vtk(fname):
    txt = open(fname).read()
    n = int(re.search(r"POINTS (\d+)", txt).group(1))
    pts = np.array(re.search(r"POINTS \d+ \w+\n(.*?)\nCELLS", txt, re.S).group(1).split(),
                   float).reshape(-1, 3)[:, :2]
    m = re.search(r"CELLS (\d+) (\d+)\n(.*?)\nCELL_TYPES", txt, re.S)
    raw = np.array(m.group(3).split(), int)
    tris, i = [], 0
    while i < len(raw):
        k = raw[i]
        if k == 3:
            tris.append(raw[i + 1:i + 4])
        i += k + 1
    tris = np.array(tris)

    def scalar(name):
        mm = re.search(r"SCALARS " + re.escape(name) + r" \w+\nLOOKUP_TABLE default\n(.*?)\n(?:SCALARS|VECTORS|CELL_DATA|$)",
                       txt + "\nSCALARS", re.S)
        return np.array(mm.group(1).split(), float)[:n] if mm else None
    return pts, tris, scalar("u0"), scalar("u1"), scalar("p")


def p1_gradients(pts, tris):
    """gradients of the three P1 basis functions per triangle, and areas"""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    area = 0.5 * np.abs(det)
    g = np.zeros((len(tris), 3, 2))
    g[:, 0, 0] = (b[:, 1] - c[:, 1]) / det
    g[:, 0, 1] = (c[:, 0] - b[:, 0]) / det
    g[:, 1, 0] = (c[:, 1] - a[:, 1]) / det
    g[:, 1, 1] = (a[:, 0] - c[:, 0]) / det
    g[:, 2, 0] = (a[:, 1] - b[:, 1]) / det
    g[:, 2, 1] = (b[:, 0] - a[:, 0]) / det
    return g, area


def vorticity_p1(pts, tris, u, v):
    g, area = p1_gradients(pts, tris)
    vx = np.einsum("tk,tk->t", v[tris], g[:, :, 0])
    uy = np.einsum("tk,tk->t", u[tris], g[:, :, 1])
    om_cell = vx - uy
    num = np.zeros(len(pts))
    den = np.zeros(len(pts))
    for j in range(3):
        np.add.at(num, tris[:, j], om_cell * area)
        np.add.at(den, tris[:, j], area)
    return num / den


def stream_function_p1(pts, tris, u, v, psi_boundary, is_bnd):
    """P1 least squares: (grad psi, grad phi) = ((-v, u), grad phi), psi = psi_boundary on the boundary"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    g, area = p1_gradients(pts, tris)
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(pts))
    ubar = u[tris].mean(axis=1)
    vbar = v[tris].mean(axis=1)
    for i in range(3):
        rhs_i = area * (-vbar * g[:, i, 0] + ubar * g[:, i, 1])
        np.add.at(rhs, tris[:, i], rhs_i)
        for j in range(3):
            rows.append(tris[:, i])
            cols.append(tris[:, j])
            vals.append(area * (g[:, i, 0] * g[:, j, 0] + g[:, i, 1] * g[:, j, 1]))
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(pts), len(pts))).tocsr()
    free = ~is_bnd
    psi = psi_boundary.copy()
    b = rhs[free] - K[free][:, is_bnd] @ psi[is_bnd]
    psi[free] = spsolve(K[free][:, free], b)
    return psi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="example folder (with --run) or run directory")
    ap.add_argument("--run", default=None, help="run key in the example folder (transient_study.py)")
    ap.add_argument("--out", default=None,
                    help="prefix of the output files (default: RUN_DIR/<problem>_<example>)")
    ap.add_argument("--snapshots", default="0,mid,last",
                    help="vtk output indices to plot: numbers, 'mid', 'last'")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if args.run:
        prm, tab, files_dir = load_stored_run(args.run_dir, args.run)
    else:
        prm = read_dat(args.run_dir)
        tab = read_table(args.run_dir, prm)
        files_dir = args.run_dir
    nu = 1.0 / float(prm.get("reynolds_number", "100"))
    L = float(prm.get("tgv_length", 2 * math.pi))
    k = float(prm.get("tgv_wavenumber", 1.0))
    A = float(prm.get("tgv_amplitude", 1.0))
    t0 = float(prm.get("time_start", 0.0))
    dt_out = float(prm.get("adaptive_output_time", 0.1))
    out = args.out or os.path.join(files_dir, prm["_prefix"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    t = tab["time"]
    F2 = np.exp(-4 * nu * k * k * t)

    # ---------------------------------------------------------------- time series
    E, Ee = tab["E_kin"], tab["E_kin_exact"]
    Z, Ze = tab["enstrophy"], tab["enstrophy_exact"]
    # dissipation rate -dE_h/dt by central differences on the (possibly
    # non-uniform) output times
    dEdt = np.gradient(E, t) if len(t) > 2 else np.full_like(E, np.nan)
    eps_h = -dEdt
    eps_visc = 2 * nu * Z
    eps_exact = 2 * nu * Ze
    eps_num = eps_h - eps_visc
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = axs[0][0]
    ax.plot(t, E / Ee - 1, label=r"$E_h/E - 1$")
    ax.plot(t, Z / Ze - 1, label=r"$Z_h/Z - 1$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("kinetic energy and enstrophy (relative to exact)")
    ax.legend()
    ax = axs[0][1]
    ax.plot(t, eps_h, label=r"$-dE_h/dt$")
    ax.plot(t, eps_visc, "--", label=r"$2\nu Z_h$")
    ax.plot(t, eps_exact, "k:", label=r"exact $2\nu Z = 4\nu k^2 E$")
    ax.plot(t, eps_num, label=r"numerical $-dE_h/dt - 2\nu Z_h$")
    ax.set_title("dissipation rate")
    ax.legend(fontsize=8)
    ax = axs[0][2]
    for key, lab in (("L2(u)", r"$\|u-u_h\|_0$"), ("H1(u)", r"$|u-u_h|_1$"), ("L2(p)", r"$\|p-p_h\|_0$"),
                     ("L2(omega)", r"$\|\omega-\omega_h\|_0$"), ("error", "SPG norm"), ("eta", r"$\eta$")):
        if key in tab:
            ax.semilogy(t, np.maximum(tab[key], 1e-16), label=lab)
    ax.set_title("errors and estimator")
    ax.legend(fontsize=8)
    ax = axs[1][0]
    ax.plot(t, tab["effectivity"])
    ax.set_title("effectivity index")
    ax = axs[1][1]
    ax.plot(t, tab["omega_max"], label=r"$\max\omega_h$")
    ax.plot(t, tab["omega_min"], label=r"$\min\omega_h$")
    ax.plot(t, 2 * A * k * np.sqrt(F2), "k--", lw=0.8, label=r"exact $\pm 2Ak F$")
    ax.plot(t, -2 * A * k * np.sqrt(F2), "k--", lw=0.8)
    ax.set_title("vorticity extrema")
    ax.legend(fontsize=8)
    ax = axs[1][2]
    ax.semilogy(t, tab["dt"], label=r"$\Delta t$")
    ax2 = ax.twinx()
    ax2.plot(t, tab["cells"], "C1", label="cells")
    ax.set_title(r"$\Delta t$ (left), cells (right)")
    for a in axs.flat:
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
    fig.suptitle(f"Taylor-Green vortex, nu = {nu:g}, L = {L:g}, k = {k:g}, A = {A:g}")
    fig.tight_layout()
    f_time = f"{out}_time.{args.format}"
    fig.savefig(f_time, dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- summary
    ok = E > 0
    rate = -np.polyfit(t[ok], np.log(E[ok]), 1)[0] if ok.sum() > 2 else float("nan")
    rate_exact = 4 * nu * k * k
    with open(f"{out}_summary.txt", "w") as f:
        f.write(f"# Taylor-Green vortex, nu = {nu:g}, L = {L:g}, k = {k:g}, A = {A:g}\n")
        f.write(f"energy decay rate: measured {rate:.6e}, exact 4 nu k^2 = {rate_exact:.6e}, "
                f"relative deviation {rate / rate_exact - 1:+.3e}\n")
        sel = np.isfinite(eps_num)
        if sel.any():
            f.write(f"numerical dissipation -dE_h/dt - 2 nu Z_h: mean {eps_num[sel].mean():.4e} "
                    f"(exact dissipation mean {eps_exact[sel].mean():.4e}, ratio "
                    f"{eps_num[sel].mean() / eps_exact[sel].mean():+.3e})\n")
        f.write(f"final time {t[-1]:g}: E_h/E - 1 = {E[-1]/Ee[-1]-1:+.3e}, Z_h/Z - 1 = {Z[-1]/Ze[-1]-1:+.3e}\n")
        for key in ("L2(u)", "H1(u)", "L2(p)", "L2(omega)", "error", "eta", "effectivity"):
            if key in tab:
                f.write(f"  {key:12s} final {tab[key][-1]:.4e}   max over time {np.nanmax(tab[key]):.4e}\n")
        f.write(f"steps {len(t)}, cells {int(tab['cells'][0])} ... {int(tab['cells'][-1])}\n")

    # ---------------------------------------------------------------- fields
    vdir = os.path.join(files_dir, prm.get("output_directory", "vtk"))
    base = prm.get("output_basename", "tgv")
    files = sorted(glob.glob(os.path.join(vdir, base + "*.vtk")),
                   key=lambda f: int(re.search(r"(\d+)\.vtk$", f).group(1)))
    if files:
        idx = []
        for s in args.snapshots.split(","):
            s = s.strip()
            idx.append(0 if s == "0" else len(files) - 1 if s == "last" else
                       len(files) // 2 if s == "mid" else int(s))
        idx = sorted(set(i for i in idx if 0 <= i < len(files)))

        def time_of(i):
            # output 0 at time_start; output i at the first step with t >= t0 + i dt_out
            if i == 0:
                return t0
            later = t[t >= t0 + i * dt_out - 1e-12]
            return later[0] if len(later) else t[-1]
        fig, axs = plt.subplots(len(idx), 4, figsize=(16, 3.9 * len(idx)), squeeze=False)
        for r, i in enumerate(idx):
            pts, tris, u, v, p = read_vtk(files[i])
            ti = time_of(i)
            F = math.exp(-2 * nu * k * k * ti)
            x, y = pts[:, 0], pts[:, 1]
            psi_ex = A / k * np.sin(k * x) * np.sin(k * y) * F
            om_ex = 2 * A * k * np.sin(k * x) * np.sin(k * y) * F
            ue = A * np.sin(k * x) * np.cos(k * y) * F
            ve = -A * np.cos(k * x) * np.sin(k * y) * F
            pe = A * A / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * F * F
            pe -= A * A / 4 * F * F * math.sin(2 * k * L) / (k * L)
            tol = 1e-8 * L
            is_bnd = (x < tol) | (x > L - tol) | (y < tol) | (y > L - tol)
            psi_h = stream_function_p1(pts, tris, u, v, psi_ex, is_bnd)
            om_h = vorticity_p1(pts, tris, u, v)
            tri = mtri.Triangulation(x, y, tris)
            levels = np.linspace(-abs(A / k) * F, abs(A / k) * F, 13)
            ax = axs[r][0]
            cf = ax.tricontourf(tri, psi_h, levels=21, cmap="RdBu_r")
            ax.tricontour(tri, psi_ex, levels=levels, colors="k", linewidths=0.6, linestyles="--")
            fig.colorbar(cf, ax=ax)
            ax.set_title(rf"$\psi_h$ (dashed: exact), t = {ti:.3g}", fontsize=9)
            ax = axs[r][1]
            cf = ax.tricontourf(tri, om_h, levels=21, cmap="RdBu_r")
            ax.tricontour(tri, om_ex, levels=np.linspace(-2 * abs(A) * k * F, 2 * abs(A) * k * F, 9),
                          colors="k", linewidths=0.6, linestyles="--")
            fig.colorbar(cf, ax=ax)
            ax.set_title(rf"$\omega_h$ in [{om_h.min():.3g}, {om_h.max():.3g}], exact $\pm${2*abs(A)*k*F:.3g}", fontsize=9)
            ax = axs[r][2]
            err = np.hypot(u - ue, v - ve)
            cf = ax.tripcolor(tri, err, shading="gouraud", cmap="viridis")
            fig.colorbar(cf, ax=ax)
            ax.set_title(r"$|u - u_h|$ at the vertices", fontsize=9)
            ax = axs[r][3]
            if p is not None and np.abs(p).max() > 0:
                cf = ax.tripcolor(tri, p - pe, shading="gouraud", cmap="RdBu_r")
                fig.colorbar(cf, ax=ax)
                ax.set_title(r"$p_h - p$", fontsize=9)
            else:
                ax.set_title("no pressure at this output\n(backward Euler start: no P^0)", fontsize=9)
            for a in axs[r]:
                a.set_aspect("equal")
                a.triplot(tri, lw=0.15, color="0.6", alpha=0.5) if len(tris) < 20000 else None
        fig.tight_layout()
        f_fields = f"{out}_fields.{args.format}"
        fig.savefig(f_fields, dpi=130)
        plt.close(fig)
        print(f"wrote {f_fields}")
    print(f"wrote {f_time}")
    print(f"wrote {out}_summary.txt")


if __name__ == "__main__":
    main()
