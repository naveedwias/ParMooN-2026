#!/usr/bin/env python3
"""
Convert the run folders of earlier versions of spg_study.py and transient_study.py (one folder
per run, <prefix>/<prefix>_<run>/ with the input file, <prefix>_table.txt, <prefix>_stdout.log,
PostScript meshes and vtk files) into the layout of scripts/run_store.py: the runs in
<prefix>_runs.jsonl / .csv, the graphical output in <prefix>_files/<run>/. Nothing is recomputed.

A run folder without a table is recorded as failed (with its log in <prefix>_failed.log).
Runs that are already in <prefix>_runs.jsonl are not converted again. The old folders are kept
unless --delete-old is given (then they are removed after their run has been converted).

Usage (from the folder that contains the example folders, e.g. the build directory):
  python3 ../ParMooN-2026/scripts/convert_ns_runs.py oseen_aaj_smooth tnse_taylor_green
  python3 ../ParMooN-2026/scripts/convert_ns_runs.py --delete-old tnse_taylor_green
"""
import argparse
import fnmatch
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_store import KEEP, RunStore  # noqa: E402

KEEP_CONVERTED = KEEP + ("*_meshes.pdf", "*_meshes.png", "*_time.pdf", "*_fields.pdf", "*_summary.txt")


def convert(folder, delete_old=False):
    folder = os.path.normpath(folder)
    prefix = os.path.basename(folder)
    store = RunStore(folder, prefix)
    store.lock()
    try:
        have = set(store.records())
        n_new = n_failed = n_skipped = 0
        for d in sorted(glob.glob(os.path.join(folder, f"{prefix}_*"))):
            base = os.path.basename(d)
            if not os.path.isdir(d) or base in (f"{prefix}_files", f"{prefix}_dry_run"):
                continue
            key = base[len(prefix) + 1:]
            if key in have:
                n_skipped += 1
                continue
            dats = sorted(glob.glob(os.path.join(d, "*.dat")))
            table = os.path.join(d, f"{prefix}_table.txt")
            rec = dict(run=key, study=key.split("_P")[0] if "_P" in key else "", params={}, converted=True,
                       input=open(dats[0]).read() if dats else "")
            log = None
            if os.path.exists(table):
                rec["table"] = open(table).read()
                n_new += 1
            else:
                rec.update(failed=True, reason="no table (converted run folder)")
                logs = glob.glob(os.path.join(d, "*stdout.log"))
                log = open(logs[0], errors="replace").read() if logs else "(no log)"
                n_failed += 1
            files = []
            for root, _, names in os.walk(d):
                for nm in names:
                    if any(fnmatch.fnmatch(nm, p) for p in KEEP_CONVERTED):
                        files.append(os.path.relpath(os.path.join(root, nm), d))
            dest = store.files_dir(key)
            for rel in files:
                os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
                (shutil.move if delete_old else shutil.copy2)(os.path.join(d, rel), os.path.join(dest, rel))
            rec["files"] = sorted(files)
            store.add(rec, log)
            if delete_old:
                shutil.rmtree(d)
        print(f"{folder}: {n_new} runs converted, {n_failed} without table (recorded as failed), "
              f"{n_skipped} already in {prefix}_runs.jsonl" + ("; old folders removed" if delete_old else
                                                               "; old folders kept (--delete-old removes them)"))
    finally:
        store.unlock()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="example folders (<problem>_<example>)")
    ap.add_argument("--delete-old", action="store_true", help="remove the old run folders after converting")
    args = ap.parse_args()
    for f in args.folders:
        if not os.path.isdir(f):
            sys.exit(f"{f}: no such folder")
        convert(f, args.delete_old)


if __name__ == "__main__":
    main()
