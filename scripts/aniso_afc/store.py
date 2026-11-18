"""
Storage of the runs of the anisotropic AFC study: one folder per example
(<outdir>/<prefix>, prefix = cd_<example name>, scripts/naming.py) with

  <prefix>_runs.jsonl      one line per run: parameters, measures, iterations, status
                           (JSON Lines; a run is only appended, an interrupted write loses
                           at most that line; a run repeated later counts with its last line)
  <prefix>_runs.csv        the same as a table (written from the .jsonl)
  <prefix>_failed.log      ParMooN logs of the runs that failed (complete, at most 2000
                           lines) or did not converge (the last 200 lines)
  <prefix>_solutions.npz   nodal values: one entry per run (key = run key) and the
                           coordinates once per mesh (key xy_eps<eps>_<mesh>_n<n>);
                           np.load(file)[key]

Only the main process writes (the runs themselves work in temporary folders), so
parallel runs cannot interfere.
"""
import csv
import glob
import io
import json
import os
import zipfile

import numpy as np

CSV_COLUMNS = ["run", "example", "eps", "mesh", "n", "method", "variant", "solver", "failed", "converged",
               "iterations", "residual", "rejections", "L2", "Linf_nodes", "Linf_away", "Linf_mid",
               "L2_away_lumped", "Linf_r1", "L2_r1_lumped", "u_min", "u_max", "undershoot", "overshoot",
               "cert_condition", "cert_bad", "cert_switched_off", "local_viol", "local_excess", "laplace_positive",
               "max_angle", "nodes", "cells", "max_aspect", "h_min", "n_switched_off", "wall_time", "reason"]


def run_key(eps, mesh, n, method, variant, solver, solver_suffix):
    """the key of a run (unique within an example)"""
    return (f"eps{eps:.0e}_{mesh}_n{n}_{method}" + ("" if variant == "plain" else f"_{variant}")
            + solver_suffix)


def mesh_key(eps, mesh, n):
    return f"xy_eps{eps:.0e}_{mesh}_n{n}"


class RunStore:
    def __init__(self, outdir):
        self.outdir = outdir
        self.touched = set()

    def folder(self, prefix):
        d = os.path.join(self.outdir, prefix)
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------ reading
    @staticmethod
    def read_jsonl(path):
        recs = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:                    # an interrupted last line
                    continue
                recs[r["run"]] = r                    # the last line of a run counts
        return recs

    def records(self):
        """all runs of all examples (the last line of every run)"""
        out = []
        for path in sorted(glob.glob(os.path.join(self.outdir, "*", "*_runs.jsonl"))):
            prefix = os.path.basename(os.path.dirname(path))
            for r in self.read_jsonl(path).values():
                r.setdefault("prefix", prefix)
                out.append(r)
        return out

    def done(self, retry_failed=False):
        """(prefix, run key) of the finished runs; with retry_failed without the failed ones"""
        out = set()
        for r in self.records():
            if retry_failed and r.get("failed"):
                continue
            out.add((r["prefix"], r["run"]))
        return out

    # ------------------------------------------------------------------ writing
    def add(self, prefix, rec, solution=None, log=None):
        """rec: the record (with 'run'); solution: (mesh key, points, u) or None; log: text or None"""
        d = self.folder(prefix)
        rec = dict(rec, prefix=prefix)
        with open(os.path.join(d, f"{prefix}_runs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if log:
            with open(os.path.join(d, f"{prefix}_failed.log"), "a") as f:
                status = "failed" if rec.get("failed") else "not converged"
                f.write(f"==================== {rec['run']} ({status}) ====================\n")
                f.write(log.rstrip("\n") + "\n")
        if solution is not None:
            mkey, P, u = solution
            path = os.path.join(d, f"{prefix}_solutions.npz")
            with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as z:
                names = set(z.namelist())
                for key, arr in ((mkey, P), (rec["run"], u)):
                    if key + ".npy" in names:
                        if key == mkey:
                            continue
                        # a repeated run: the zip keeps both entries; numpy reads the last one
                    buf = io.BytesIO()
                    np.save(buf, np.asarray(arr))
                    z.writestr(key + ".npy", buf.getvalue())
        self.touched.add(prefix)

    def write_csv(self, prefixes=None):
        for prefix in sorted(prefixes if prefixes is not None else self.touched):
            path = os.path.join(self.outdir, prefix, f"{prefix}_runs.jsonl")
            if not os.path.exists(path):
                continue
            recs = sorted(self.read_jsonl(path).values(), key=lambda r: r["run"])
            with open(os.path.join(self.outdir, prefix, f"{prefix}_runs.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CSV_COLUMNS)
                for r in recs:
                    c = r.get("certificate") or {}
                    ld = r.get("local_dmp") or {}
                    lp = r.get("laplace") or {}
                    flat = dict(r, cert_condition=c.get("condition"), cert_bad=c.get("n_bad"),
                                cert_switched_off=c.get("n_switched_off"), local_viol=ld.get("n_viol"),
                                local_excess=ld.get("excess"), laplace_positive=lp.get("n_positive"),
                                max_angle=r.get("max_angle", lp.get("max_angle")))
                    row = []
                    for k in CSV_COLUMNS:
                        v = flat.get(k)
                        if isinstance(v, bool):
                            v = int(v)
                        elif isinstance(v, float):
                            v = f"{v:.6e}"
                        row.append("" if v is None else v)
                    w.writerow(row)
