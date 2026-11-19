#!/usr/bin/env python3
"""
The Hemker problem (Hemker 1996) with ParMooN's cd2d_afc on a layer-adapted polar mesh around the
circle: runs, quantities of interest, figures and VTK files.

Problem (Example_CD2D 2, include/Examples/CD_2D/Hemker1996.h, boundary data/mesh/Hemker.PRM):
  -eps Lap u + u_x = 0 in (-3, 9) x (-3, 3) minus the unit disk; u = 1 on the circle, u = 0 at the
  inflow x = -3, homogeneous Neumann on y = -3, y = 3 and x = 9; eps = 1e-4 (default). The solution
  has a boundary layer at the front of the circle (width ~ eps), parabolic layers along its sides
  and two interior layers along y = +-1 downstream (width ~ sqrt(eps x)); 0 <= u <= 1.

Mesh (hemker_mesh): the square [-a, a]^2 (a = 1.1, --square) minus the disk is filled with a polar block
(rays from the circle to the nodes on the boundary of the square), graded towards the circle in
three zones along every ray: n_r/4 rings in [0, tau_1], tau_1 = sigma eps ln n (front layer),
n_r/4 rings in [tau_1, tau_2], tau_2 = sigma sqrt(eps) ln n (parabolic layers at the sides), the
rest uniform (sigma = 2, n_r = n/2). The rest of the rectangle is a tensor grid whose y-nodes are
refined around y = +-1 (half-width tau_y = sigma sqrt(eps) ln n, a quarter of the n nodes in [-a, a]
each), so the interior layers are refined from x = a to the outflow. Between the circle and x = a
they are resolved only by the polar block. n = cells per side of the square; the mesh is
conforming, the circle nodes lie on the circle. A small square keeps the stretch of the interior
layers inside the polar block short: at about 21 000 nodes the layer width at x = 4 (Galerkin /
Kuzmin) is 0.165 / 0.363 with a = 2, 0.130 / 0.180 with a = 1.25 and 0.084 / 0.123 with a = 1.1.

Methods: Galerkin, SUPG (as run_study.py), and the algebraic stabilizations (default BJK, Kuzmin,
MC, MUAS, SMUAS, BBK). Nonlinear solver of an algebraic method: the first of Anderson with
safeguard, Newton with continuation (not for MC) and the fixed point iteration that solves the
scheme (run_study.solved), otherwise the attempt with the smallest residual of the scheme.

Quantities of interest (as galerkit qoi.hemker_measures):
  u_min, u_max over the nodes, osc = u_max - u_min (1 for a solution in [0, 1]);
  smear_x4 = y_01 - y_09 on x = 4, scanning y from 3 down to 0: y_01 (y_09) the first y with
  u_h >= 0.1 (0.9) (reference 0.0723 for eps = 1e-4, Augustin et al., CMAME 200, 2011, as quoted in
  galerkit); cut lines u_h(4, y), u_h(x, 1), u_h(x, 2) (u_h is P1: exact linear interpolation).

Output (default outdir cd_hemker, figdir hemker_results):
  <outdir>/cd_hemker/cd_hemker_runs.jsonl, _runs.csv, _solutions.npz  (aniso_afc/store.py)
  <figdir>/hemker_mesh_n<n>.png        the mesh: whole domain, around the circle, the front layer
  <figdir>/hemker_solutions_n<n>.png   u_h of every method, [min, max] in the titles
  <figdir>/hemker_cut_x4_n<n>.png      u_h(4, y) across the upper interior layer
  <figdir>/hemker_cut_y_n<n>.png       u_h(x, 1) and u_h(x, 2)
  <figdir>/hemker_smear.png            smear_x4 and the undershoot against n (several n)
  <figdir>/hemker_qoi.md, hemker_qoi.csv   the quantities of interest of every run
  <figdir>/hemker_n<n>.vtk             mesh and all solutions (ParaView)

Usage (from the build directory):
  python3 ../ParMooN-2026/scripts/aniso_afc/hemker_study.py --n 32 64 128 --jobs 4
  python3 ../ParMooN-2026/scripts/aniso_afc/hemker_study.py --n 64 --plot-only
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import measures as Q  # noqa: E402
from meshes import graded_points, grid_triangles  # noqa: E402
from run_study import DAT, LABEL, SOLVER_SUFFIX, solved  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402

PREFIX = "cd_hemker"
EXAMPLE_CD2D = 2
PRM = os.path.join(REPO, "data", "mesh", "Hemker.PRM")
XMIN, XMAX, YMIN, YMAX = -3.0, 9.0, -3.0, 3.0
METHODS = ("galerkin", "supg", "bjk", "kuzmin_edge", "mc", "muas", "smuas", "bbk")
LINEAR = ("galerkin", "supg")
SOLVERS = ("fp_anderson_safe", "newton_cont", "fixed_point")
SMEAR_REF = 0.0723
SIGMA = 2.0
SQUARE = 1.1                                            # half-width a of the square of the polar block


def mesh_name(a):
    """mesh name in the run keys: hemker_polar_a<a>"""
    return f"hemker_polar_a{a:g}"


# ----------------------------------------------------------------------------- mesh
def radial_distances(L, nr, eps, n, sigma=SIGMA):
    """nr + 1 distances in [0, L] from the circle: n_r/4 cells in [0, tau1], n_r/4 in [tau1, tau2], the rest
    uniform; tau1 = sigma eps ln n (front layer), tau2 = sigma sqrt(eps) ln n (parabolic layers)"""
    t1 = min(sigma * eps * np.log(n), 0.05 * L)
    t2 = min(sigma * np.sqrt(eps) * np.log(n), 0.4 * L)
    n1 = n2 = max(1, nr // 4)
    n3 = nr - n1 - n2
    if t2 <= t1 or n3 < 1:
        return np.linspace(0.0, L, nr + 1)
    return np.concatenate([np.linspace(0, t1, n1 + 1), np.linspace(t1, t2, n2 + 1)[1:],
                           np.linspace(t2, L, n3 + 1)[1:]])


def hemker_mesh(n, eps=1e-4, a=SQUARE, sigma=SIGMA):
    """points (N, 2), triangles (M, 3) of the layer-adapted polar mesh (see the module docstring)"""
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    h = 2 * a / n
    ty = min(0.4, sigma * np.sqrt(eps) * np.log(n))
    xs = np.linspace(-a, a, n + 1)                                   # top and bottom of the square
    ys = graded_points(n, [-1.0, 1.0], [ty, ty], [0.25, 0.25], -a, a)  # left/right: layers at y = +-1
    X = np.concatenate([np.linspace(XMIN, -a, max(2, round((-a - XMIN) / h)) + 1)[:-1], xs,
                        np.linspace(a, XMAX, max(2, round((XMAX - a) / h)) + 1)[1:]])
    Y = np.concatenate([np.linspace(YMIN, -a, max(2, round((-a - YMIN) / h)) + 1)[:-1], ys,
                        np.linspace(a, YMAX, max(2, round((YMAX - a) / h)) + 1)[1:]])
    ia, ib = int(np.argmin(np.abs(X + a))), int(np.argmin(np.abs(X - a)))
    ja, jb = int(np.argmin(np.abs(Y + a))), int(np.argmin(np.abs(Y - a)))
    nx, ny = len(X), len(Y)
    inside = np.zeros((nx, ny), bool)
    inside[ia + 1:ib, ja + 1:jb] = True                              # strictly inside the square
    gid = -np.ones((nx, ny), int)
    gid[~inside] = np.arange((~inside).sum())
    GX, GY = np.meshgrid(X, Y, indexing="ij")
    P = [np.stack([GX[~inside], GY[~inside]], 1)]
    # tensor cells outside the square
    I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    keep = ~((I >= ia) & (I < ib) & (J >= ja) & (J < jb))
    I, J = I[keep], J[keep]
    a_, b_, c_, d_ = gid[I, J], gid[I + 1, J], gid[I + 1, J + 1], gid[I, J + 1]
    T = [np.stack([a_, b_, c_], 1), np.stack([a_, c_, d_], 1)]
    # boundary of the square, counterclockwise from the corner (a, -a)
    loop = ([gid[ib, j] for j in range(ja, jb + 1)] + [gid[i, jb] for i in range(ib - 1, ia - 1, -1)]
            + [gid[ia, j] for j in range(jb - 1, ja - 1, -1)] + [gid[i, ja] for i in range(ia + 1, ib)])
    Pall = np.vstack(P)
    q = Pall[loop]
    c = q / np.linalg.norm(q, axis=1)[:, None]                        # circle nodes on the rays
    L = np.linalg.norm(q - c, axis=1)
    nr = n // 2
    nt = len(loop)
    ring = np.empty((nt, nr + 1), int)
    ring[:, nr] = loop
    new = []
    base = Pall.shape[0]
    for j in range(nt):
        d = radial_distances(L[j], nr, eps, n, sigma)
        for k in range(nr):
            new.append(c[j] + d[k] / L[j] * (q[j] - c[j]))
            ring[j, k] = base + len(new) - 1
    Pall = np.vstack([Pall, np.array(new)])
    jj, kk = np.meshgrid(np.arange(nt), np.arange(nr), indexing="ij")
    jj, kk = jj.ravel(), kk.ravel()
    j1 = (jj + 1) % nt
    a_, b_, c_, d_ = ring[jj, kk], ring[j1, kk], ring[j1, kk + 1], ring[jj, kk + 1]
    T += [np.stack([a_, b_, c_], 1), np.stack([a_, c_, d_], 1)]
    T = np.vstack(T)
    on_circle = np.zeros(len(Pall), bool)
    on_circle[ring[:, 0]] = True
    return Pall, T, on_circle


def check_mesh(P, T, on_circle):
    """conformity and validity: positive areas, every interior edge shared by two triangles, the total
    area equals the area of the domain, circle nodes on the circle"""
    p = P[T]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    exact_area = (XMAX - XMIN) * (YMAX - YMIN) - np.pi
    return dict(nodes=len(P), cells=len(T), min_area=float(np.abs(area).min()),
                same_orientation=bool(np.all(area > 0) or np.all(area < 0)),
                edges_more_than_two=int((cnt > 2).sum()), boundary_edges=int((cnt == 1).sum()),
                area_error=float(np.abs(area).sum() - exact_area),       # polygon: O(h_theta^2) below pi
                circle_radius_error=float(np.abs(np.linalg.norm(P[on_circle], axis=1) - 1).max()))


def prm_circle_angles(path=PRM):
    """start and end angle of the circle in Hemker.PRM, as ParMooN reads them (7 digits: 3 pi and pi are
    stored as 9.424778 and 3.141593); the circle parameter of a node must use these values, otherwise
    ParMooN places the node up to 4e-7 away from it"""
    vals = open(path).read().split("PARAMETERS")[1].split()
    return float(vals[-2]), float(vals[-1])


def write_hemker_geo(P, T, on_circle, path, title="Hemker problem, layer-adapted polar mesh"):
    """ParMooN .GEO for data/mesh/Hemker.PRM: boundary part 1 = the rectangle (components 0: y = -3,
    1: x = 9, 2: y = 3, 3: x = -3), part 2 = the circle (parameter t: angle th0 + t (th1 - th0) with the
    angles of the .PRM file, about 3 pi - 2 pi t)"""
    T = np.array(T, int)
    p = P[T]
    area2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    T[area2 < 0] = T[area2 < 0][:, [0, 2, 1]]
    tol = 1e-12
    th0, th1 = prm_circle_angles()
    coord, knpr = [], []
    for k, (x, y) in enumerate(P):
        if on_circle[k]:
            # angle(t) = th0 + t (th1 - th0) with th0 ~ 3 pi, th1 ~ pi: take the angle in (th1, th0] (wrapping t
            # by 1 instead would rotate the node by (th1 - th0) + 2 pi ~ 3e-7, the rounding of the .PRM angles)
            th = np.arctan2(y, x)
            th += 2 * np.pi * np.ceil((th1 - th) / (2 * np.pi))   # the angle in (th1, th1 + 2 pi] = (th1, th0]
            t = (th0 - th) / (th0 - th1)
            if not -1e-12 <= t < 1.0:
                raise ValueError(f"circle parameter {t} of ({x}, {y}) outside [0, 1)")
            t = max(t, 0.0)
            coord.append(f"{t:.17g} 0"); knpr.append(2)
        elif abs(y - YMIN) < tol:
            coord.append(f"{(x - XMIN) / (XMAX - XMIN):.17g} 0"); knpr.append(1)
        elif abs(x - XMAX) < tol:
            coord.append(f"{1 + (y - YMIN) / (YMAX - YMIN):.17g} 0"); knpr.append(1)
        elif abs(y - YMAX) < tol:
            coord.append(f"{2 + (XMAX - x) / (XMAX - XMIN):.17g} 0"); knpr.append(1)
        elif abs(x - XMIN) < tol:
            coord.append(f"{3 + (YMAX - y) / (YMAX - YMIN):.17g} 0"); knpr.append(1)
        else:
            coord.append(f"{x:.17g} {y:.17g}"); knpr.append(0)
    knpr = np.array(knpr)
    with open(path, "w") as f:
        f.write(title + "\nn_elem n_vert ignored max_n_vert_per_elem n_bound_parts\n")
        f.write(f"  {len(T)}   {len(P)}   0   3   2\n")
        f.write("DCORVG\n" + "\n".join(coord) + "\n")
        f.write("KVERT\n" + "\n".join(f"{a + 1} {b + 1} {c + 1}" for a, b, c in T) + "\n")
        f.write("KNPR\n" + " ".join(map(str, knpr)) + "\n")
        kmm = []
        for part in (1, 2):
            ids = np.flatnonzero(knpr == part)
            par = np.array([float(coord[i].split()[0]) for i in ids])
            kmm += [ids[par.argmin()] + 1, ids[par.argmax()] + 1]
        f.write("KMM\n" + " ".join(map(str, kmm)) + "\n")
    return path


# ----------------------------------------------------------------------------- quantities of interest
def qoi(P, T, u):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
    f = mtri.LinearTriInterpolator(tri, u)
    y = np.linspace(3.0, 0.0, 100001)
    v = np.asarray(f(np.full_like(y, 4.0), y).filled(np.nan))

    def first(level):
        k = np.flatnonzero(v >= level)
        return float(y[k[0]]) if k.size else float("nan")
    y01, y09 = first(0.1), first(0.9)
    umin, umax = float(u.min()), float(u.max())
    return dict(u_min=umin, u_max=umax, osc=umax - umin, undershoot=max(0.0, -umin), overshoot=max(0.0, umax - 1),
                smear_x4=y01 - y09, y_01=y01, y_09=y09)


def profiles(P, T, u, m=2000):
    import matplotlib.tri as mtri
    f = mtri.LinearTriInterpolator(mtri.Triangulation(P[:, 0], P[:, 1], T), u)
    y = np.linspace(YMIN, YMAX, m + 1)
    x = np.linspace(XMIN, XMAX, m + 1)
    return {"x=4": (y, np.asarray(f(np.full_like(y, 4.0), y).filled(np.nan))),
            "y=1": (x, np.asarray(f(x, np.full_like(x, 1.0)).filled(np.nan))),
            "y=2": (x, np.asarray(f(x, np.full_like(x, 2.0)).filled(np.nan)))}


# ----------------------------------------------------------------------------- runs
def one_attempt(exe, work, geo, eps, n, method, solver, maxit, tol, P):
    pre = "hemker"
    txt = DAT.format(ex=0, eps=eps, mesh="hemker_polar", n=n, method=method, variant="plain", prefix=pre, geo=geo,
                     example=EXAMPLE_CD2D, disc="supg" if method == "supg" else "galerkin",
                     afc="none" if method in LINEAR else "afc", maxit=maxit, tol=tol / np.sqrt(len(P)),
                     algstab="none" if method in LINEAR else method, loff="false", roff="false", kappa=10.0, mu=1.0,
                     solver=solver, reg=1e-6, nc_maxit=40)
    txt = re.sub(r"^boundary_file:.*$", f"boundary_file: {PRM}", txt, flags=re.M)
    txt = re.sub(r"^output_compute_errors:.*$", "output_compute_errors: false", txt, flags=re.M)
    if method == "supg":
        txt += "SDFEM_TYPE: 2\nCELL_MEASURE: 4\nSUPG_DELTA_CELLWISE: 1\n"
    dat = os.path.join(work, f"{pre}_{method}_{solver}.dat")
    open(dat, "w").write(txt)
    for f in ("hemker_nodal.txt", "hemker_edges.txt"):
        if os.path.exists(os.path.join(work, f)):
            os.remove(os.path.join(work, f))
    t0 = time.perf_counter()
    out = subprocess.run([exe, os.path.basename(dat)], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True).stdout
    rec = dict(method=method, solver="linear" if method in LINEAR else solver, wall_time=time.perf_counter() - t0)
    nodal = os.path.join(work, "hemker_nodal.txt")
    if not os.path.exists(nodal):
        rec.update(failed=True, reason="no solution written", log="\n".join(out.splitlines()[-200:]))
        return rec, None
    Pn, un, _ = Q.read_nodal(nodal)
    u, dist = Q.to_mesh_order(P, Pn, un)
    if dist > 1e-8:
        rec.update(failed=True, reason=f"ParMooN's nodes differ from the mesh ({dist:.1e})")
        return rec, None
    if method not in LINEAR:
        it = re.findall(r"(MAX_)?NONLINEAR ITERATION: ite (\d+) res ([0-9.eE+-]+|-?nan|-?inf) rejections (\d+)", out)
        if it:
            rec.update(iterations=int(it[-1][1]), residual=float(it[-1][2]), converged=it[-1][0] == "")
        else:
            rec.update(iterations=0, residual=float("nan"), converged=False)
        r0 = re.findall(r"res0 ([0-9.eE+-]+|nan|inf)\)", out)
        if r0:
            rec.update(residual0=float(r0[-1]))
        if solver == "newton_cont":
            rec.update(polished=", polished" in out)
    return rec, u


def run_job(job):
    """one method at one n: the solver attempts in a temporary folder; returns (record, solution)"""
    exe, outdir, eps, n, method, maxit, tol, a = job
    P, T, on_circle = hemker_mesh(n, eps, a)
    scratch = os.path.join(outdir, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"n{n}_{method}_", dir=scratch)
    try:
        geo = write_hemker_geo(P, T, on_circle, os.path.join(work, "hemker_mesh.GEO"))
        attempts, best = [], None
        for solver in (("linear",) if method in LINEAR else SOLVERS):
            if method == "mc" and solver == "newton_cont":
                continue
            rec, u = one_attempt(exe, work, geo, eps, n, method, "fixed_point" if solver == "linear" else solver,
                                 maxit, tol, P)
            attempts.append({k: v for k, v in rec.items() if k != "log"})
            if u is None:
                continue
            res = rec.get("residual0", rec.get("residual", 0.0)) if method not in LINEAR else 0.0
            if best is None or (res is not None and np.isfinite(res) and res < best[2]):
                best = (rec, u, res if res is not None and np.isfinite(res) else np.inf)
            if solved(dict(rec, method=method)):
                best = (rec, u, res)
                break
        key = run_key(eps, mesh_name(a), n, method, "plain", "linear", "")
        base = dict(run=key, example="hemker", eps=eps, mesh=mesh_name(a), square=a, n=n, method=method,
                    variant="plain", attempts=attempts)
        if best is None:
            return dict(base, failed=True, reason="no attempt wrote a solution"), None, P
        rec, u, _ = best
        out = dict(base, **{k: v for k, v in rec.items() if k != "method"}, solved=solved(dict(rec, method=method)),
                   **qoi(P, T, u))
        return out, u, P
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- output
def load_runs(outdir, eps, a=SQUARE):
    st = RunStore(outdir)
    recs = [r for r in st.records() if r.get("example") == "hemker" and r.get("eps") == eps and not r.get("failed")
            and r.get("mesh") == mesh_name(a)]
    path = os.path.join(outdir, PREFIX, f"{PREFIX}_solutions.npz")
    Z = np.load(path) if os.path.exists(path) else None
    return recs, Z


def solution(Z, rec):
    if Z is None or rec["run"] not in Z.files:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Z[rec["run"]]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.color": "#e4e3de", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})
    return plt


COLORS = {"galerkin": "#1f1f1e", "supg": "#6b6a64", "bjk": "#2a78d6", "kuzmin_edge": "#eb6834", "mc": "#1baf7a",
          "muas": "#eda100", "smuas": "#e87ba4", "bbk": "#008300", "fbbk": "#6250d6", "sbbk": "#e34948"}


def make_outputs(outdir, figdir, eps, ns, methods, a=SQUARE):
    from rotating_figure import write_vtk
    recs, Z = load_runs(outdir, eps, a)
    if not recs:
        print(f"no Hemker runs for eps = {eps:g} in {outdir}")
        return
    plt = _plt()
    import matplotlib.tri as mtri
    os.makedirs(figdir, exist_ok=True)
    by = {(r["n"], r["method"]): r for r in recs}
    ns = [n for n in ns if any((n, m) in by for m in methods)]
    # table of the quantities of interest
    cols = ["n", "method", "solver", "iterations", "solved", "u_min", "u_max", "osc", "undershoot", "overshoot",
            "smear_x4", "y_01", "y_09", "wall_time"]
    with open(os.path.join(figdir, "hemker_qoi.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n in ns:
            for m in methods:
                r = by.get((n, m))
                if r:
                    w.writerow([n, LABEL[m]] + [r.get(c) for c in cols[2:]])
    lines = [f"# Hemker problem, eps = {eps:g}, layer-adapted polar mesh", "",
             f"smear_x4 = y(u = 0.1) - y(u = 0.9) on x = 4 (reference {SMEAR_REF} for eps = 1e-4); "
             "osc = max u_h - min u_h (1 for a solution in [0, 1]); * = nonlinear problem not solved", ""]
    for n in ns:
        P, T, _ = hemker_mesh(n, eps, a)
        lines += [f"## n = {n} ({len(P)} nodes, {len(T)} triangles)", "",
                  "| method | solver | it. | u_min | u_max | osc | smear_x4 | wall time |", "|---|---|---|---|---|---|---|---|"]
        for m in methods:
            r = by.get((n, m))
            if not r:
                continue
            star = "" if r.get("solved", True) else "*"
            lines.append(f"| {LABEL[m]}{star} | {r.get('solver')} | {r.get('iterations', '-')} | {r['u_min']:.3e} | "
                         f"{r['u_max']:.6f} | {r['osc']:.4f} | {r['smear_x4']:.4f} | {r.get('wall_time', 0):.1f} s |")
        lines.append("")
    open(os.path.join(figdir, "hemker_qoi.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for n in ns:
        P, T, on_circle = hemker_mesh(n, eps, a)
        tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
        # mesh
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[2, 1, 1]))
        for ax, (xl, yl, ttl) in zip(axs, (((XMIN, XMAX), (YMIN, YMAX), "whole domain"),
                                           ((-2.2, 2.2), (-2.2, 2.2), "around the circle"),
                                           ((-1.03, -0.97), (-0.03, 0.03), "front of the circle (layer ~ eps)"))):
            ax.triplot(tri, lw=0.15 if ttl == "whole domain" else 0.3, color="#3b3f45")
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(False); ax.set_title(ttl)
        fig.suptitle(f"Hemker problem, eps = {eps:g}: layer-adapted polar mesh, n = {n} ({len(P)} nodes)")
        fig.savefig(os.path.join(figdir, f"hemker_mesh_n{n}.png")); plt.close(fig)
        # solutions
        ms = [m for m in methods if (n, m) in by and solution(Z, by[(n, m)]) is not None]
        if not ms:
            continue
        ncol = 2
        fig, axs = plt.subplots((len(ms) + 1) // 2, ncol, figsize=(13, 2.3 * ((len(ms) + 1) // 2)), squeeze=False)
        fields = []
        tc = None
        for ax, m in zip(axs.ravel(), ms):
            u = solution(Z, by[(n, m)])
            fields.append(("u_" + LABEL[m].replace("-", ""), u))
            tc = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis", vmin=-0.1, vmax=1.1)
            star = "" if by[(n, m)].get("solved", True) else " (not solved)"
            ax.set_title(f"{LABEL[m]}{star}: [{u.min():.3g}, {u.max():.4g}]", fontsize=8)
            ax.set_aspect("equal"); ax.grid(False)
        for ax in axs.ravel()[len(ms):]:
            ax.set_visible(False)
        fig.colorbar(tc, ax=axs, shrink=0.6, label="u")
        fig.suptitle(f"Hemker problem, eps = {eps:g}, n = {n}: solutions (min, max in brackets)")
        fig.savefig(os.path.join(figdir, f"hemker_solutions_n{n}.png")); plt.close(fig)
        write_vtk(os.path.join(figdir, f"hemker_n{n}.vtk"), P, T, fields, f"Hemker problem, eps = {eps:g}, n = {n}")
        # cut lines
        prof = {m: profiles(P, T, solution(Z, by[(n, m)])) for m in ms}
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
        for m in ms:
            y, v = prof[m]["x=4"]
            for ax in axs:
                ax.plot(y, v, lw=1.1, color=COLORS[m], label=LABEL[m])
        axs[0].set_xlim(-3, 3); axs[0].set_title("u_h(4, y)")
        r0 = by.get((n, "galerkin"))
        axs[1].set_xlim(0.8, 1.2); axs[1].set_ylim(-0.1, 1.1)
        axs[1].set_title(f"zoom on the upper layer (reference width {SMEAR_REF})")
        for ax in axs:
            ax.set_xlabel("y")
        axs[0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(f"Hemker problem, eps = {eps:g}, n = {n}: cut line x = 4")
        fig.savefig(os.path.join(figdir, f"hemker_cut_x4_n{n}.png")); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        for m in ms:
            for ax, key in zip(axs, ("y=1", "y=2")):
                x, v = prof[m][key]
                ax.plot(x, v, lw=1.1, color=COLORS[m], label=LABEL[m])
        for ax, key in zip(axs, ("y=1", "y=2")):
            ax.set_title(f"u_h(x, {key[2:]})"); ax.set_xlabel("x"); ax.set_xlim(XMIN, XMAX)
        axs[0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(f"Hemker problem, eps = {eps:g}, n = {n}: cut lines y = 1 and y = 2")
        fig.savefig(os.path.join(figdir, f"hemker_cut_y_n{n}.png")); plt.close(fig)
    if len(ns) > 1:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        for m in methods:
            pts = [(by[(n, m)]["n"], by[(n, m)]["smear_x4"], by[(n, m)]["undershoot"], by[(n, m)].get("solved", True))
                   for n in ns if (n, m) in by]
            if not pts:
                continue
            nn, sm, un, ok = zip(*pts)
            axs[0].semilogx(nn, sm, "o-", color=COLORS[m], label=LABEL[m], ms=4)
            axs[1].loglog(nn, np.maximum(un, 1e-16), "o-", color=COLORS[m], label=LABEL[m], ms=4)
            for x, s, uu, k in pts:
                if not k:
                    axs[0].plot(x, s, "x", color="k", ms=7)
                    axs[1].plot(x, max(uu, 1e-16), "x", color="k", ms=7)
        axs[0].axhline(SMEAR_REF, color="#6b6a64", ls="--", lw=1, label="reference 0.0723")
        axs[0].set_title("smear_x4 (width of the interior layer at x = 4)"); axs[0].set_xlabel("n")
        axs[1].set_title("undershoot -min u_h (1e-16: none)"); axs[1].set_xlabel("n")
        axs[0].legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle(f"Hemker problem, eps = {eps:g}: quantities of interest against n (x: not solved)")
        fig.savefig(os.path.join(figdir, "hemker_smear.png")); plt.close(fig)
    print(f"figures, VTK files and tables in {figdir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default="./parMooN_cd2d_afc_SEQUENTIAL")
    ap.add_argument("--outdir", default="cd_hemker")
    ap.add_argument("--figdir", default="hemker_results")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(LABEL))
    ap.add_argument("--square", type=float, default=SQUARE,
                    help="half-width a of the square [-a, a]^2 of the polar block (1 < a < 3; default 1.1)")
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--redo", action="store_true", help="recompute finished runs")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--check-mesh", action="store_true", help="only build and check the meshes")
    args = ap.parse_args()
    if args.check_mesh:
        for n in args.n:
            print(n, check_mesh(*hemker_mesh(n, args.eps, args.square)))
        return
    if not args.plot_only:
        exe = os.path.abspath(args.exe)
        outdir = os.path.abspath(args.outdir)
        store = RunStore(outdir)
        os.makedirs(outdir, exist_ok=True)
        done = {r["run"] for r in store.records() if not r.get("failed")}
        jobs = [(exe, outdir, args.eps, n, m, args.maxit, args.tol, args.square) for n in args.n for m in args.methods
                if args.redo or run_key(args.eps, mesh_name(args.square), n, m, "plain", "linear", "") not in done]
        print(f"{len(jobs)} runs")
        pool = Pool(args.jobs) if args.jobs > 1 else None
        for rec, u, P in (pool.imap_unordered(run_job, jobs) if pool else map(run_job, jobs)):
            store.add(PREFIX, rec, None if u is None else (mesh_key(args.eps, mesh_name(args.square), rec["n"]), P, u))
            print(f"  n = {rec['n']:4d} {LABEL[rec['method']]:8s} "
                  + (f"failed: {rec.get('reason')}" if rec.get("failed") else
                     f"{rec.get('solver')}, smear_x4 {rec['smear_x4']:.4f}, [{rec['u_min']:.3g}, {rec['u_max']:.4g}]"
                     + ("" if rec.get("solved", True) else " (not solved)")), flush=True)
        if pool:
            pool.close(); pool.join()
        store.write_csv()
        shutil.rmtree(os.path.join(outdir, ".scratch"), ignore_errors=True)
    make_outputs(args.outdir, args.figdir, args.eps, args.n, args.methods, args.square)


if __name__ == "__main__":
    main()
