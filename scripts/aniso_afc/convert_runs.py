#!/usr/bin/env python3
"""
Convert the runs of the anisotropic AFC study from the old layout (one folder per
run, <outdir>/<prefix>_<study>_eps<eps>_<mesh>_n<n>_<method>[...] with
<prefix>_record.json, nodal values, edges and log; commits up to be902fa) to the
layout of store.py (one folder per example with <prefix>_runs.jsonl, .csv,
_failed.log, _solutions.npz). Nothing is recomputed.

  python3 convert_runs.py [--outdir cd_aniso] [--solutions auto|all|none] [--delete-old]

The old folders are kept unless --delete-old (after a successful conversion; also
the folders of interrupted runs, which have no record); old records of runs that are
already in the new files are skipped. Old folders without
eps in the name (before 1f16349) are not converted: their records may be mixed.
"""
import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import measures as Q  # noqa: E402
from run_study import EXAMPLE, LINEAR, SOLVER_SUFFIX, make_mesh  # noqa: E402
from store import RunStore, mesh_key, run_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="cd_aniso")
    ap.add_argument("--solutions", default="auto", choices=["auto", "all", "none"])
    ap.add_argument("--delete-old", action="store_true")
    args = ap.parse_args()
    store = RunStore(args.outdir)
    done = store.done()
    keep = {"all": lambda ex: True, "none": lambda ex: False, "auto": lambda ex: ex in (4, 5)}[args.solutions]
    old = sorted(d for d in glob.glob(os.path.join(args.outdir, "cd_*")) if os.path.isdir(d)
                 and not glob.glob(os.path.join(d, "*_runs.jsonl")))       # not a folder of the new layout
    folders = [d for d in old if glob.glob(os.path.join(d, "*_record.json"))]
    incomplete = [d for d in old if d not in folders]                       # interrupted runs: no record
    converted, skipped, unsafe = [], 0, []
    for d in folders:
        if "_eps" not in os.path.basename(d):
            unsafe.append(d)
            continue
        rf = glob.glob(os.path.join(d, "*_record.json"))[0]
        pre = os.path.basename(rf)[:-len("_record.json")]
        rec = json.load(open(rf))
        solver = rec.get("solver", "fixed_point")
        meth = rec["method"]
        key = run_key(rec["eps"], rec["mesh"], rec["n"], meth, rec.get("variant", "plain"), solver,
                      "" if meth in LINEAR else SOLVER_SUFFIX.get(solver, ""))
        if (pre, key) in done:
            skipped += 1
            converted.append(d)
            continue
        rec["run"] = key
        solution, log = None, None
        nodal = os.path.join(d, f"{pre}_nodal.txt")
        if keep(rec["example"]) and os.path.exists(nodal) and not rec.get("failed"):
            P, T = make_mesh(rec["example"], rec["eps"], rec["mesh"], rec["n"])
            Pn, un, _ = Q.read_nodal(nodal)
            u, _ = Q.to_mesh_order(P, Pn, un)
            solution = (mesh_key(rec["eps"], rec["mesh"], rec["n"]), P, u)
        lf = os.path.join(d, f"{pre}_stdout.log")
        if os.path.exists(lf) and (rec.get("failed") or (meth not in LINEAR and not rec.get("converged", True))):
            lines = open(lf, errors="replace").read().splitlines()
            log = "\n".join(lines[-(2000 if rec.get("failed") else 200):])
        store.add(pre, rec, solution, log)
        done.add((pre, key))
        converted.append(d)
    store.write_csv(prefixes={os.path.basename(p) for p in glob.glob(os.path.join(args.outdir, "cd_*"))
                              if os.path.isdir(p) and glob.glob(os.path.join(p, "*_runs.jsonl"))})
    print(f"{len(converted) - skipped} runs converted, {skipped} already in the new files; "
          f"folders per example: {', '.join(sorted(store.touched)) or '-'}")
    if unsafe:
        print(f"{len(unsafe)} old folders without eps in the name (possibly mixed records) not converted; "
              f"delete them: rm -rf " + " ".join(os.path.basename(u) for u in unsafe[:3]) + (" ..." if len(unsafe) > 3 else ""))
    if incomplete:
        print(f"{len(incomplete)} old folders of interrupted runs (no record; the next run step repeats them)"
              + (" deleted" if args.delete_old else "; --delete-old deletes them"))
    if args.delete_old:
        for d in converted + incomplete:
            shutil.rmtree(d, ignore_errors=True)
        print(f"{len(converted)} old run folders deleted")
    elif converted:
        print("the old run folders are kept; delete them with --delete-old (or rm -rf), "
              "or keep them as a backup")


if __name__ == "__main__":
    main()
