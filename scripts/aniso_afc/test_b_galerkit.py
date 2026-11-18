#!/usr/bin/env python3
"""
Pointwise test of the algebraic stabilizations of AlgebraicStabilization
against galerkit: for the same mesh, problem and u (random and galerkit's
solution), b_ij(u) of every edge in both codes. Independent of the nonlinear
solvers, so it also decides cases where a fixed point iteration does not
converge.

Usage:
  python3 test_b_galerkit.py --exe <build>/parMooN_cd2d_afc_SEQUENTIAL
          [--problem test_b23] [--eps 1e-3] [--n 8] [--mesh shishkin_r]
          [--methods bjk,mc,muas,smuas,kuzmin_edge,bbk,fbbk,sbbk] [--loff] [--roff] [--mu 1]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosscheck_galerkit import PARMOON_EXAMPLE, run_parmoon  # noqa: E402
from geo_io import write_unit_square_geo  # noqa: E402

GK = {"bjk": "afc", "mc": "afc_mc", "muas": "muas", "smuas": "smuas", "kuzmin_edge": "afc_kuzmin",
      "bbk": "bbk", "fbbk": "fbbk", "sbbk": "sbbk"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--problem", default="test_b23")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mesh", default="shishkin_r")
    ap.add_argument("--methods", default="bjk,mc,muas,smuas,kuzmin_edge,bbk,fbbk,sbbk")
    ap.add_argument("--loff", action="store_true")
    ap.add_argument("--roff", action="store_true")
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--workdir", default="test_b")
    args = ap.parse_args()

    from galerkit.convection_diffusion import CDRSolver, get_problem
    from galerkit.convection_diffusion.problems import CDRProblem
    from galerkit.core.mesh import SimplexMesh
    import meshes as M

    if args.problem == "test_b23":
        pb = CDRProblem(name="test_b23", dim=2, eps=args.eps, b=np.array([2.0, 3.0]), c=1.0, f=0.0,
                        g=lambda x, t=0.0: x[0] + x[1] ** 2)
    else:
        pb = get_problem(args.problem, eps=args.eps)
    ln = np.log(args.n)
    if args.mesh == "uniform":
        P, T = M.tensor_mesh(np.linspace(0, 1, args.n + 1), np.linspace(0, 1, args.n + 1), "right")
    elif args.mesh in ("polar", "polar_layer", "tensor_layer"):
        P, T = M.rotating_mesh(args.mesh, args.n, eps=args.eps)
    else:
        diag = {"shishkin_r": "right", "shishkin_l": "left", "shishkin_x": "crossed"}[args.mesh]
        if args.problem == "exp_layer":
            P, T = M.shishkin_mesh(args.n, 2 * args.eps * ln, None, diagonal=diag)
        else:
            P, T = M.shishkin_mesh(args.n, 2 * args.eps / 2 * ln, 2 * args.eps / 3 * ln, diagonal=diag)
    order = np.lexsort((P[:, 1], P[:, 0]))         # lexicographic numbering (Kuzmin ties)
    new = np.empty_like(order)
    new[order] = np.arange(order.size)
    P, T = P[order], new[T]
    mesh = SimplexMesh(P, T)
    os.makedirs(args.workdir, exist_ok=True)
    geo = os.path.abspath(os.path.join(args.workdir, "mesh.GEO"))
    write_unit_square_geo(P, T, geo)
    run_parmoon.tol = 1e-12
    rng = np.random.default_rng(1)
    opts = dict(afc_layer_off=args.loff, afc_resolved_off=args.roff, afc_aspect_threshold=args.kappa,
                afc_q_relax=args.mu)
    print(f"{args.problem}, eps {args.eps:g}, {args.mesh}, n = {args.n}, loff {args.loff}, roff {args.roff}, "
          f"mu {args.mu}: max |b_ParMooN - b_galerkit| / max |b|")
    for meth in args.methods.split(","):
        S = CDRSolver(pb, mesh, k=1, stab=GK[meth], **opts)
        op = S.afc_operator(0.0)
        if meth == "kuzmin_edge":                   # the same tie rule (see crosscheck_galerkit.py)
            sc = np.maximum(np.abs(op.aij), np.abs(op.aji))
            op.aji = np.where(np.abs(op.aij - op.aji) <= 1e-10 * sc, op.aij, op.aji)
        ug, _ = S.solve()
        res = []
        for label, u in (("random", rng.standard_normal(len(P))), ("solution", np.asarray(ug))):
            ufile = os.path.abspath(os.path.join(args.workdir, f"u_{meth}_{label}.txt"))
            np.savetxt(ufile, np.column_stack([P, u]), fmt="%.17g")
            extra = {"algstab_test_b": ufile, "afc_nonlinloop_maxit": "1",
                     "algstab_layer_off": str(args.loff).lower(), "algstab_resolved_off": str(args.roff).lower(),
                     "algstab_aspect_threshold": str(args.kappa), "algstab_q_relax": str(args.mu)}
            run_parmoon(os.path.abspath(args.exe), args.workdir, f"t_{meth}_{label}", geo,
                        PARMOON_EXAMPLE[args.problem], args.eps, afc="afc", limiter="algstab:" + meth, extra=extra)
            bfile = ufile + ".b"
            if not os.path.exists(bfile):
                res.append(f"{label}: ParMooN run failed")
                continue
            d = np.loadtxt(bfile)
            from scipy.spatial import cKDTree
            tree = cKDTree(P)                       # ParMooN nodes -> galerkit nodes (nearest)
            _, gi = tree.query(d[:, 2:4])
            _, gj = tree.query(d[:, 4:6])
            bp = {(min(a, b), max(a, b)): v for a, b, v in zip(gi, gj, d[:, 6])}
            bg = op.b_edges(u)
            diff, scale, miss = 0.0, max(np.abs(bg).max(), 1e-300), 0
            for e, (i, j) in enumerate(zip(op.I, op.J)):
                key = (min(i, j), max(i, j))
                if key not in bp:
                    miss += 1
                    continue
                diff = max(diff, abs(bp[key] - bg[e]))
            res.append(f"{label}: {diff / scale:.1e}" + (f" ({miss} edges not matched)" if miss else ""))
        print(f"  {meth:12s} " + ",  ".join(res))


if __name__ == "__main__":
    main()
