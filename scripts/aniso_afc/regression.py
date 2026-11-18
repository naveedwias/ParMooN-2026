#!/usr/bin/env python3
"""
Regression cases of the algebraic stabilizations (module AlgebraicStabilization)
and their solvers, with ParMooN only: small runs of cd2d_afc whose solutions
were verified against galerkit (scripts/aniso_afc/README.md), compared with the
stored reference values in regression_reference.json.

  python3 regression.py --exe <build>/parMooN_cd2d_afc_SEQUENTIAL [--update]

Per case: maximal nodal error (Example 1) or the solution at three nodes
(Example 28), convergence, and the iteration count within --it-tol (relative,
default 0.2: round-off can change the counts of the nonlinear iterations a
little). --update writes new reference values (only after a check against
galerkit).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import meshes as M  # noqa: E402
import measures as Q  # noqa: E402
from geo_io import write_unit_square_geo  # noqa: E402
from run_study import DAT, EXAMPLE  # noqa: E402

CASES = []
for meth in ("bjk", "kuzmin_edge", "mc", "muas", "smuas", "bbk", "fbbk", "sbbk"):
    CASES.append(("ex1_shishkin_fp", 1, 1e-4, "shishkin_r", 8, meth, "fixed_point", "plain"))
for meth in ("bjk", "smuas", "sbbk"):
    CASES.append(("ex1_shishkin_aas", 1, 1e-4, "shishkin_r", 8, meth, "fp_anderson_safe", "plain"))
for meth in ("bjk", "muas", "bbk"):
    CASES.append(("ex1_shishkin_nc", 1, 1e-4, "shishkin_r", 8, meth, "newton_cont", "plain"))
for meth in ("bjk", "kuzmin_edge", "muas"):
    CASES.append(("ex1_crossed_loff", 1, 1e-4, "shishkin_x", 8, meth, "fixed_point", "loff"))
CASES.append(("ex2_shishkin_fbbk_nw", 2, 1e-8, "shishkin_r", 16, "fbbk", "newton", "plain"))


def run_case(exe, case, work):
    name, ex, eps, mesh, n, meth, solver, var = case
    from run_study import make_mesh
    P, T = make_mesh(ex, eps, mesh, n)
    pre = f"{name}_{meth}"
    geo = os.path.join(work, pre + ".GEO")
    write_unit_square_geo(P, T, geo)
    txt = DAT.format(ex=ex, eps=eps, mesh=mesh, n=n, method=meth, variant=var, prefix=pre, geo=geo,
                     example=EXAMPLE[ex], disc="galerkin", afc="afc", maxit=3000, tol=1e-10 / np.sqrt(len(P)),
                     algstab=meth, loff=str(var == "loff").lower(), roff="false", kappa=10.0, mu=1.0,
                     solver=solver, reg=1e-6, nc_maxit=40)
    open(os.path.join(work, pre + ".dat"), "w").write(txt)
    out = subprocess.run([exe, pre + ".dat"], cwd=work, capture_output=True, text=True).stdout
    m = re.findall(r"(MAX_)?NONLINEAR ITERATION: ite (\d+) res (\S+)", out)
    nodal = os.path.join(work, pre + "_nodal.txt")
    if not m or not os.path.exists(nodal):
        return dict(failed=True)
    Pn, un, _ = Q.read_nodal(nodal)
    u, _ = Q.to_mesh_order(P, Pn, un)
    rec = dict(iterations=int(m[-1][1]), converged=m[-1][0] == "")
    rec.update(Linf=float(Q.errors(ex, eps, n, P, T, u)["Linf_nodes"]))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--it-tol", type=float, default=0.2)
    ap.add_argument("--err-tol", type=float, default=1e-6, help="relative tolerance of the errors")
    args = ap.parse_args()
    ref_file = os.path.join(HERE, "regression_reference.json")
    ref = json.load(open(ref_file)) if os.path.exists(ref_file) else {}
    work = tempfile.mkdtemp(prefix="aniso_regression_")
    new, fails = {}, 0
    for case in CASES:
        key = f"{case[0]}_{case[5]}"
        r = run_case(os.path.abspath(args.exe), case, work)
        new[key] = r
        if args.update:
            print(f"  {key:32s} {r}")
            continue
        g = ref.get(key)
        ok = g is not None and not r.get("failed") and r["converged"] == g["converged"] \
            and abs(r["Linf"] - g["Linf"]) <= args.err_tol * max(abs(g["Linf"]), 1e-12) \
            and abs(r["iterations"] - g["iterations"]) <= args.it_tol * max(g["iterations"], 5)
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {key:32s} it {r.get('iterations')} (ref {g and g['iterations']}), "
              f"max error {r.get('Linf', float('nan')):.6e} (ref {g and g['Linf']:.6e})" if g else
              f"  FAIL {key}: no reference")
    shutil.rmtree(work, ignore_errors=True)
    if args.update:
        json.dump(new, open(ref_file, "w"), indent=1)
        print(f"wrote {ref_file}")
    else:
        print(f"{len(CASES) - fails} of {len(CASES)} cases passed")
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
