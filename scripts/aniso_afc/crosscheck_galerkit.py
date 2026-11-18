#!/usr/bin/env python3
"""
Cross-check of ParMooN's algebraic stabilizations against galerkit (the code of
the anisotropic AFC study): the same problem on the same triangulation in both
codes, nonlinear problems solved to a small residual, nodal solutions compared.

The mesh is built with galerkit (galerkit.core.mesh) and written as a ParMooN
.GEO file (geo_io.py); ParMooN runs cd2d_afc on it and writes the P1 solution
at the vertices (vtk), which is matched to galerkit's nodes by coordinates.
Galerkin first: it must agree to round-off (same mesh, data and assembly);
then every pair (galerkit method, ParMooN limiter).

Requires galerkit (pip install -e <galerkit>) and the ParMooN program cd2d_afc.

Usage:
  python3 crosscheck_galerkit.py --exe <build>/parMooN_cd2d_afc_SEQUENTIAL
         [--problem boundary_layers] [--eps 1e-2] [--n 8] [--mesh uniform|shishkin_r]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_io import read_vtk_point_scalar, write_unit_square_geo  # noqa: E402

# galerkit problem -> ParMooN example (Example_CD2D)
PARMOON_EXAMPLE = {"boundary_layers": 7, "exp_layer": 27}
# (galerkit stab, ParMooN afc_limiter, extra ParMooN parameters)
PAIRS = [
    ("afc", "BJK17", {}),
    ("afc_kuzmin", "kuzmin", {}),
    ("afc_mc", "monolithic", {}),
    ("muas", "MUAS", {}),
    ("muas", "MUAS_MAX", {}),
    ("bbk", "BBK", {"power_p": "10.0"}),
]

DAT = """[ cross-check with galerkit ]
outfile: {name}.out
boundary_file: Default_UnitSquare
geo_file: {geo}
refinement_n_initial_steps: 0
problem_type: 1
verbosity: 1
example: {example}
diffusion_coefficient: {eps:.17g}
output_write_vtk: true
output_write_ps: false
output_compute_errors: true
output_directory: .
output_basename: {name}
solver_type: direct
direct_solver_type: umfpack
ANSATZ_ORDER: 1
DISCTYPE: 1
SDFEM_TYPE: 0
algebraic_flux_correction: {afc}
afc_limiter: {limiter}
afc_iteration_scheme: {scheme}
afc_nonlinloop_maxit: 20000
afc_nonlinloop_epsilon: 1e-12
afc_initial_iterate: galerkin
afc_nonlinloop_anderson_acc: no
"""


def run_parmoon(exe, workdir, name, geo, example, eps, afc="none", limiter="kuzmin",
                scheme="fixed_point_rhs", extra=None):
    os.makedirs(workdir, exist_ok=True)
    txt = DAT.format(name=name, geo=geo, example=example, eps=eps, afc=afc, limiter=limiter, scheme=scheme)
    for k, v in (extra or {}).items():
        txt += f"{k}: {v}\n"
    dat = os.path.join(workdir, name + ".dat")
    open(dat, "w").write(txt)
    with open(os.path.join(workdir, name + ".log"), "w") as log:
        subprocess.run([exe, name + ".dat"], cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    vtks = sorted(f for f in os.listdir(workdir) if f.startswith(name) and f.endswith(".vtk"))
    if not vtks:
        return None, None
    pts, u, _ = read_vtk_point_scalar(os.path.join(workdir, vtks[-1]))
    return pts, u


def match(pts_ref, pts, vals):
    """values at pts_ref: nearest vtk vertex (the vtk files store about 8 significant
    digits); the matching distance must be far below the smallest mesh spacing"""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(pts).query(pts_ref)
    if len(set(idx)) != len(idx):
        raise ValueError("ambiguous node matching")
    match.max_dist = max(getattr(match, "max_dist", 0.0), float(dist.max()))
    return np.asarray(vals)[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--problem", default="boundary_layers")
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mesh", default="uniform")
    ap.add_argument("--workdir", default="crosscheck")
    ap.add_argument("--scheme", default="fixed_point_rhs", help="ParMooN afc_iteration_scheme")
    ap.add_argument("--pairs", default=None, help="comma-separated ParMooN limiters to test (default: all)")
    args = ap.parse_args()

    from galerkit.convection_diffusion import CDRSolver, get_problem
    from galerkit.core.mesh import rectangle_mesh, shishkin_mesh

    pb = get_problem(args.problem, eps=args.eps)
    if args.mesh == "uniform":
        mesh = rectangle_mesh(args.n, diagonal="right")
    else:
        diag = {"shishkin_r": "right", "shishkin_l": "left", "shishkin_x": "crossed"}[args.mesh]
        ln = np.log(args.n)
        if args.problem == "exp_layer":
            mesh = shishkin_mesh(args.n, tau_x=2 * args.eps * ln, tau_y=None, diagonal=diag)
        else:
            mesh = shishkin_mesh(args.n, tau_x=2 * args.eps / 2 * ln, tau_y=2 * args.eps / 3 * ln,
                                 side_y="right", diagonal=diag)
    os.makedirs(args.workdir, exist_ok=True)
    geo = os.path.abspath(os.path.join(args.workdir, f"{args.mesh}{args.n}.GEO"))
    write_unit_square_geo(mesh.points, mesh.cells, geo, f"galerkit {args.mesh} n={args.n}")
    exe = os.path.abspath(args.exe)
    example = PARMOON_EXAMPLE[args.problem]
    P = mesh.points
    print(f"problem {args.problem}, eps {args.eps:g}, mesh {args.mesh}, n = {args.n}: "
          f"{len(P)} nodes, {len(mesh.cells)} triangles")

    # Galerkin: same mesh, data and assembly
    ug, _ = CDRSolver(pb, mesh, k=1, stab="none").solve()
    pts, up = run_parmoon(exe, args.workdir, "galerkin", geo, example, args.eps, afc="none")
    if up is None:
        sys.exit("ParMooN Galerkin run failed (see the log in the work directory)")
    d = np.abs(match(P, pts, up) - np.asarray(ug)).max()
    e = mesh.points[mesh.cells]
    hmin = min(np.linalg.norm(e[:, (k + 1) % 3] - e[:, k], axis=1).min() for k in range(3))
    print(f"  Galerkin                  max |u_ParMooN - u_galerkit| = {d:.2e}   (max |u| {np.abs(ug).max():.3g}; "
          f"node matching distance {match.max_dist:.1e}, h_min {hmin:.1e})")

    pairs = PAIRS if args.pairs is None else [p for p in PAIRS if p[1] in args.pairs.split(",")]
    for gk, pm, extra in pairs:
        S = CDRSolver(pb, mesh, k=1, stab=gk, afc_solver="auto", afc_maxiter=20000, afc_tol=1e-12) \
            if "afc_tol" in CDRSolver.__init__.__code__.co_varnames else \
            CDRSolver(pb, mesh, k=1, stab=gk, afc_solver="auto", afc_maxiter=20000)
        u_g, info = S.solve()
        a = info.get("afc", {})
        name = f"afc_{pm}"
        pts, up = run_parmoon(exe, args.workdir, name, geo, example, args.eps, afc="afc", limiter=pm,
                              scheme=args.scheme, extra=extra)
        if up is None:
            print(f"  {gk:10s} vs {pm:11s}  ParMooN run failed")
            continue
        d = np.abs(match(P, pts, up) - np.asarray(u_g)).max()
        log = open(os.path.join(args.workdir, name + ".log"), errors="replace").read()
        print(f"  {gk:10s} vs {pm:11s}  max diff {d:.2e}   galerkit: converged {a.get('converged')}, "
              f"residual {a.get('residual', float('nan')):.1e}, {a.get('iterations')} it.")


if __name__ == "__main__":
    main()
