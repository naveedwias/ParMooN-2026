"""
Storage of the runs of the Navier-Stokes studies (spg_study.py, transient_study.py), one
folder per example (<problem>_<example>, scripts/naming.py), with a few files instead of one
folder per run (the layout of the anisotropic AFC study, scripts/aniso_afc/store.py):

  <prefix>_runs.jsonl   one line per run: "run" (the run key), "study", the parameters of the
                        run, "input" (the ParMooN input file), "table" (the text of the table the
                        program wrote), "wall_time", "failed" / "reason", "files"; a run is only
                        appended (an interrupted write loses at most that line), a run repeated
                        later (--redo) counts with its last line
  <prefix>_runs.csv     one row per run: key, study, status, wall time, parameters
  <prefix>_failed.log   ParMooN logs of the runs that failed (the last 2000 lines)
  <prefix>_files/<run>/ the graphical output of a run, if it wrote any: the PostScript meshes
                        (*.ps) and the VTK files (*.vtk, in their subfolders), and the figures
                        drawn from them

Every run is computed in a temporary folder (<folder>/.scratch, removed afterwards); a lock
(<folder>/.lock) prevents two run steps in the same folder (a lock of a process that no longer
exists is taken over). Run folders of earlier versions: scripts/convert_ns_runs.py.
"""
import csv
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

KEEP = ("*.ps", "*.vtk")                     # graphical output kept per run


class RunStore:
    def __init__(self, folder, prefix):
        self.folder = folder
        self.prefix = prefix
        self.lockfile = None

    def path(self, suffix):
        return os.path.join(self.folder, f"{self.prefix}_{suffix}")

    # ------------------------------------------------------------------ reading
    def records(self):
        """{run key: record}, the last line of every run"""
        recs = {}
        p = self.path("runs.jsonl")
        if not os.path.exists(p):
            return recs
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:                    # an interrupted last line
                    continue
                recs[r["run"]] = r
        return recs

    def get(self, key):
        r = self.records().get(key)
        return None if r is None or r.get("failed") else r

    def files_dir(self, key):
        return os.path.join(self.folder, f"{self.prefix}_files", key)

    # ------------------------------------------------------------------ writing
    def add(self, rec, log=None):
        os.makedirs(self.folder, exist_ok=True)
        with open(self.path("runs.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec.get("failed") or log:
            with open(self.path("failed.log"), "a") as f:
                f.write(f"==================== {rec['run']} ({rec.get('reason', 'failed')}) "
                        "====================\n")
                f.write("\n".join((log or "(the program wrote no output)").splitlines()[-2000:]) + "\n")

    def write_csv(self):
        recs = sorted(self.records().values(), key=lambda r: r["run"])
        if not recs:
            return
        params = sorted({k for r in recs for k in r.get("params", {})})
        with open(self.path("runs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "study", "failed", "reason", "wall_time", "files"] + params)
            for r in recs:
                w.writerow([r["run"], r.get("study", ""), int(bool(r.get("failed"))), r.get("reason", ""),
                            f"{r.get('wall_time', 0):.1f}", len(r.get("files", []))]
                           + [r.get("params", {}).get(k, "") for k in params])

    # ------------------------------------------------------------------ lock
    def lock(self):
        os.makedirs(self.folder, exist_ok=True)
        lk = os.path.join(self.folder, ".lock")
        if os.path.exists(lk):
            try:
                pid = int(open(lk).read().split()[0])
                os.kill(pid, 0)
                sys.exit(f"another run step (process {pid}) is running in {self.folder}; wait for it, "
                         f"or stop it (kill {pid}) and start again")
            except (ValueError, IndexError, ProcessLookupError):
                pass                                  # stale lock
            except PermissionError:
                sys.exit(f"{lk} belongs to a running process of another user")
        with open(lk, "w") as f:
            f.write(f"{os.getpid()}\n")
        self.lockfile = lk
        shutil.rmtree(os.path.join(self.folder, ".scratch"), ignore_errors=True)   # leftovers

    def unlock(self):
        self.write_csv()
        try:
            os.rmdir(os.path.join(self.folder, ".scratch"))
        except OSError:
            pass
        if self.lockfile and os.path.exists(self.lockfile):
            os.remove(self.lockfile)
        self.lockfile = None

    # ------------------------------------------------------------------ one run
    def run(self, key, exe, dat_name, dat_text, table_name, timeout, study="", params=None, subdirs=()):
        """run ParMooN in a temporary folder on the input dat_text (written as dat_name), keep the
        table and the graphical output, append the record; returns the record"""
        scratch = os.path.join(self.folder, ".scratch")
        os.makedirs(scratch, exist_ok=True)
        work = tempfile.mkdtemp(prefix=f"{key}_", dir=scratch)
        rec = dict(run=key, study=study, params=params or {}, input=dat_text)
        log = None
        try:
            for d in subdirs:                          # e.g. the vtk folder (ParMooN does not create it)
                os.makedirs(os.path.join(work, d), exist_ok=True)
            with open(os.path.join(work, dat_name), "w") as f:
                f.write(dat_text)
            t0 = time.time()
            logfile = os.path.join(work, "stdout.log")
            with open(logfile, "w") as lf:
                try:
                    p = subprocess.run([exe, dat_name], cwd=work, stdout=lf, stderr=subprocess.STDOUT,
                                       timeout=timeout)
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            rec["wall_time"] = time.time() - t0
            logtxt = open(logfile, errors="replace").read()
            table = os.path.join(work, table_name)
            if os.path.exists(table):
                rec["table"] = open(table).read()
            if code != 0 or "table" not in rec:
                rec.update(failed=True, reason=("timeout" if code == "timeout" else
                                                f"exit code {code}" if code != 0 else "no table written"))
                log = logtxt
            # graphical output
            files = []
            for root, _, names in os.walk(work):
                for nm in names:
                    if any(fnmatch.fnmatch(nm, pat) for pat in KEEP):
                        files.append(os.path.relpath(os.path.join(root, nm), work))
            dest = self.files_dir(key)
            shutil.rmtree(dest, ignore_errors=True)    # output of an earlier run of this key
            if files:
                for rel in files:
                    os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
                    shutil.move(os.path.join(work, rel), os.path.join(dest, rel))
            rec["files"] = sorted(files)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.add(rec, log)
        return rec
