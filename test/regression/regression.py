#!/usr/bin/env python3
"""
Regression tests for the adaptive / time-dependent programs.

Each case runs a short version of one of the shipped inputs and compares all
numbers of the program's table file and of selected result lines of the log
(RESULTS_LINE, 'estimator:', 'TR-AB2: t =') with frozen references in
test/regression/reference/<case>/. Used to check that refactorings (moving
code and examples) do not change any result.

  python3 regression.py --build ../../../build            # run and compare
  python3 regression.py --build ../../../build --update   # (re)create references
  python3 regression.py --build ... --cases tnse_ex1 bdf2_sv_cip_ex25

In addition, VERIFY cases check exactness: examples whose solution lies in the
discrete space must be reproduced up to round-off (bounds on the table
columns, no references needed).

Exit code 0 if every case agrees within --rtol (default 1e-10) / --atol and
every exactness check holds.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
INP = os.path.join(REPO, "data", "input_files")
REF = os.path.join(HERE, "reference")

# name: (executable, base input, overrides, table key)
CASES = {
    # transient Navier-Stokes, SUPG/PSPG/grad-div, TR-AB2 (examples 23, 28, 32)
    "tnse_verify_tr": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                       {"time_end": "0.2", "time_step_length": "0.05"}, "tnse_write_table"),
    "tnse_ex1": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_ex1.dat",
                 {"refinement_n_initial_steps": "3", "time_end": "0.01", "output_write_vtk": "false"},
                 "tnse_write_table"),
    "tnse_ex1_adaptive_dt": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_ex1.dat",
                             {"refinement_n_initial_steps": "3", "time_end": "0.15", "trab2_adaptive": "true",
                              "tnse_initial_value": "consistent_projection", "output_write_vtk": "false"},
                             "tnse_write_table"),
    "tnse_ex2_adaptive": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_ex2.dat",
                          {"refinement_n_initial_steps": "3", "time_end": "0.05", "output_write_vtk": "false",
                           "output_write_ps": "false", "adaptive_max_cell_level": "6"}, "tnse_write_table"),
    # BDF2-IMEX / CIP (examples 25, 26, 27, 28)
    "bdf2_sv_cip_ex25": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_cip.dat",
                         {"time_end": "0.25", "time_step_length": "0.0625"}, "bdf2_write_table"),
    "bdf2_lake_ex26": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_cip.dat",
                       {"example": "26", "pressure_amplitude": "100.0", "time_end": "0.25",
                        "time_step_length": "0.0625"}, "bdf2_write_table"),
    "bdf2_shear_ex27": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_shearlayer.dat",
                        {"refinement_n_initial_steps": "3", "time_end": "0.05"}, "bdf2_write_table"),
    "bdf2_th_ex28": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_verify_time.dat",
                     {"VELOCITY_SPACE": "2", "PRESSURE_SPACE": "1", "refinement_final_step_barycentric": "false",
                      "refinement_n_initial_steps": "2", "time_end": "0.2", "time_step_length": "0.05"},
                     "bdf2_write_table"),
    # steady Oseen, SUPG/PSPG/grad-div estimator (examples 21, 22)
    "oseen_ex21": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                   {"refinement_max_n_adaptive_steps": "3"}, "spg_write_table"),
    "oseen_ex22_adaptive": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                            {"example": "22", "reynolds_number": "1.0e6", "spg_uniform_refinement": "false",
                             "refinement_n_initial_steps": "2", "refinement_max_n_adaptive_steps": "4",
                             "spg_estimator_sigma0": "0.0"}, "spg_write_table"),
    # convection-diffusion (FGJN14 examples 11 and 15)
    "cd_ex2": ("parMooN_tcd2d_adaptive_tr_SEQUENTIAL", "tcd2d_adaptive_tr_ex2.dat",
               {"refinement_n_initial_steps": "4", "time_end": "0.1", "output_write_vtk": "false"},
               "tcd_write_table"),
    "cd_ex4_adaptive": ("parMooN_tcd2d_adaptive_tr_SEQUENTIAL", "tcd2d_adaptive_tr_ex4.dat",
                        {"time_end": "0.01", "output_write_vtk": "false", "output_write_ps": "false"},
                        "tcd_write_table"),
    # the monolithic drivers kept for reference
    "old_tnse_ex1": ("parMooN_tnse2d_spg_adaptive_SEQUENTIAL", "tnse2d_spg_ex1.dat",
                     {"refinement_n_initial_steps": "3", "time_end": "0.01", "output_write_vtk": "false"},
                     "tnse_write_table"),
}

# Exactness checks: examples whose solution lies in the discrete space (and,
# for the time integrators, is polynomial of a degree the scheme integrates
# exactly). The maximum over the table rows of the listed columns must stay
# below the bound; these checks do not need references and are independent of
# the machine. name: (executable, base input, overrides, table key, {column: bound})
VERIFY = {
    # stationary SUPG/PSPG/grad-div method and estimator, example 29 (u in P2, p in P1)
    "exact_oseen_P2P1": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                         {"example": "29", "reynolds_number": "1.0e3", "refinement_max_n_adaptive_steps": "2"},
                         "spg_write_table", {"error_spg": 1e-12, "eta": 1e-11}),
    "exact_oseen_P2P2": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                         {"example": "29", "reynolds_number": "1.0e3", "PRESSURE_SPACE": "2",
                          "refinement_max_n_adaptive_steps": "2"},
                         "spg_write_table", {"error_spg": 1e-12, "eta": 1e-11}),
    "exact_nse_P3P2": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                       {"example": "29", "problem_type": "5", "reynolds_number": "1.0e3",
                        "VELOCITY_SPACE": "3", "PRESSURE_SPACE": "2", "nonlinloop_epsilon": "1e-14",
                        "refinement_max_n_adaptive_steps": "2"},
                       "spg_write_table", {"error_spg": 1e-11, "eta": 1e-10}),
    # trapezoidal rule, example 30 (phi quadratic in time): exact for any steps and meshes
    "exact_tr_fixed": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                       {"example": "30", "time_end": "0.5", "time_step_length": "0.05",
                        "nonlinloop_epsilon": "1e-13"},
                       "tnse_write_table", {"error": 1e-11, "eta": 1e-9, "L2(u)": 1e-12, "L2(p)": 1e-11}),
    "exact_tr_adaptive_dt": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                             {"example": "30", "time_end": "0.5", "trab2_adaptive": "true",
                              "trab2_tolt": "1e-4", "trab2_averaging_steps": "0", "tnse_dt_alpha": "0.0",
                              "nonlinloop_epsilon": "1e-13"},
                             "tnse_write_table", {"error": 1e-10, "eta": 1e-9, "L2(u)": 1e-12,
                                                  "L2(p)": 1e-10}),
    "exact_tr_galerkin": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                          {"example": "30", "time_end": "0.3", "time_step_length": "0.05",
                           "nonlinloop_epsilon": "1e-13", "spg_delta0": "0.0", "spg_mu0": "0.0"},
                          "tnse_write_table", {"L2(u)": 1e-12, "L2(p)": 1e-11}),
    # outflow (do-nothing/Neumann) boundary with PSPG: the initial derivative
    # needs the differentiated constraint (the PSPG system is singular there)
    "exact_tr_neumann": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                         {"example": "34", "time_end": "0.3", "time_step_length": "0.05",
                          "nonlinloop_epsilon": "1e-13"},
                         "tnse_write_table", {"error": 1e-11, "L2(u)": 1e-12, "L2(p)": 1e-11}),
    "exact_oseen_neumann": ("parMooN_oseen2d_spg_adaptive_SEQUENTIAL", "oseen2d_spg_aaj2026.dat",
                            {"example": "33", "reynolds_number": "1.0e3", "refinement_max_n_adaptive_steps": "2"},
                            "spg_write_table", {"error_spg": 1e-12}),
    "exact_tr_mesh_alternation": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL",
                                  "tnse2d_adaptive_spg_verify_time.dat",
                                  {"example": "30", "time_end": "0.5", "time_step_length": "0.05",
                                   "nonlinloop_epsilon": "1e-13", "adaptive_mesh": "true",
                                   "adaptive_tol1": "1e-30", "adaptive_tol2": "1.0",
                                   "adaptive_coarsening": "true", "adaptive_max_cell_level": "5"},
                                  "tnse_write_table", {"error": 1e-11, "eta": 1e-9, "L2(u)": 1e-12,
                                                       "L2(p)": 1e-11}),
    # backward Euler first step (trab2_first_step backward_euler), example 31
    # (phi linear in time: BE and TR are both exact); the last entry checks
    # the sequence of BE and TR steps in the log
    "exact_be_first_fixed": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                             {"example": "31", "time_end": "0.5", "nonlinloop_epsilon": "1e-13",
                              "time_step_length": "0.05", "trab2_first_step": "backward_euler"},
                             "tnse_write_table", {"error": 1e-11, "eta": 1e-9, "L2(u)": 1e-12, "L2(p)": 1e-11},
                             "be_first"),
    "exact_be_first_adaptive_dt": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                                   {"example": "31", "time_end": "0.5", "nonlinloop_epsilon": "1e-13",
                                    "trab2_adaptive": "true", "trab2_tolt": "1e-4",
                                    "trab2_averaging_steps": "0", "tnse_dt_alpha": "0.0",
                                    "trab2_first_step": "backward_euler"},
                                   "tnse_write_table", {"error": 1e-10, "eta": 1e-9, "L2(u)": 1e-12,
                                                        "L2(p)": 1e-10},
                                   "be_first"),
    "exact_be_mesh_alternation": ("parMooN_tnse2d_adaptive_spg_SEQUENTIAL", "tnse2d_adaptive_spg_verify_time.dat",
                                  {"example": "31", "time_end": "0.5", "nonlinloop_epsilon": "1e-13",
                                   "time_step_length": "0.05", "adaptive_mesh": "true",
                                   "adaptive_tol1": "1e-30", "adaptive_tol2": "1.0",
                                   "adaptive_coarsening": "true", "adaptive_max_cell_level": "5",
                                   "trab2_first_step": "backward_euler",
                                   "trab2_be_after_mesh_change": "true"},
                                  "tnse_write_table", {"error": 1e-11, "eta": 1e-9, "L2(u)": 1e-12,
                                                       "L2(p)": 1e-11},
                                  "be_after_mesh_change"),
    # BDF2-IMEX/CIP, example 31 (phi linear in time), Stokes projection start
    "exact_bdf2_sv_cip": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_verify_time.dat",
                          {"example": "31", "time_end": "0.5", "time_step_length": "0.05"},
                          "bdf2_write_table", {"L2(u)": 1e-13, "H1(u)": 1e-12, "L2(p)": 1e-12,
                                               "div(u)": 1e-12}),
    "exact_bdf2_th": ("parMooN_tnse2d_bdf2imex_cip_SEQUENTIAL", "tnse2d_bdf2imex_verify_time.dat",
                      {"example": "31", "time_end": "0.5", "time_step_length": "0.05",
                       "VELOCITY_SPACE": "2", "PRESSURE_SPACE": "1",
                       "refinement_final_step_barycentric": "false", "refinement_n_initial_steps": "2"},
                      "bdf2_write_table", {"L2(u)": 1e-13, "H1(u)": 1e-12, "L2(p)": 1e-12}),
}

LOG_PATTERNS = ("RESULTS_LINE", "estimator:", "TR-AB2: t =")
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)")


def set_param(lines, key, value):
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*)$")
    out, done = [], False
    for ln in lines:
        m = pat.match(ln)
        if m:
            if not done:
                old, v = m.group(1).strip(), str(value)
                if re.fullmatch(r"[-+]?\d+", v) and re.search(r"[.eE]", old) \
                        and re.fullmatch(r"[-+]?[\d.]+([eE][-+]?\d+)?", old):
                    v += ".0"
                out.append(f"{key}: {v}")
                done = True
        else:
            out.append(ln)
    if not done:
        out.append(f"{key}: {value}")
    return out


def prepare_input(base, overrides, run_dir, table_key):
    lines = open(os.path.join(INP, base)).read().splitlines()
    for k, v in overrides.items():
        lines = set_param(lines, k, v)
    lines = set_param(lines, "outfile", "run.out")
    lines = set_param(lines, table_key, "table.txt")
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*geo_file\s*:\s*(.+)$", ln)
        if m and "data/mesh/" in m.group(1):
            p = m.group(1).strip()
            lines[i] = "geo_file: " + os.path.join(REPO, p[p.index("data/mesh/"):])
        m = re.match(r"^\s*output_directory\s*:\s*(.+)$", ln)
        if m:
            os.makedirs(os.path.join(run_dir, m.group(1).strip()), exist_ok=True)
    path = os.path.join(run_dir, "input.dat")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def extract(run_dir):
    """numbers of the table and of the selected log lines"""
    res = {}
    t = os.path.join(run_dir, "table.txt")
    res["table"] = [ln for ln in open(t).read().splitlines() if ln.strip()] if os.path.exists(t) else []
    o = os.path.join(run_dir, "run.out")
    log = open(o, errors="replace").read().splitlines() if os.path.exists(o) else []
    res["log"] = [ln.strip() for ln in log if ln.strip().startswith(LOG_PATTERNS)]
    return res


def compare_lines(ref, new, rtol, atol, what):
    if len(ref) != len(new):
        return [f"{what}: {len(new)} lines instead of {len(ref)}"]
    msgs = []
    for i, (a, b) in enumerate(zip(ref, new)):
        if NUM.sub("#", a) != NUM.sub("#", b):
            msgs.append(f"{what} line {i}: text differs\n    ref: {a}\n    new: {b}")
            continue
        for x, y in zip(NUM.findall(a), NUM.findall(b)):
            fx, fy = float(x), float(y)
            if fx != fx and fy != fy:
                continue
            if abs(fx - fy) > atol + rtol*max(abs(fx), abs(fy)):
                msgs.append(f"{what} line {i}: {x} -> {y}\n    ref: {a}\n    new: {b}")
                break
        if len(msgs) > 5:
            break
    return msgs


def check_step_sequence(out_file, mode):
    """Sequence of backward Euler and TR steps in the log of the adaptive time
    loop. 'be_first': step 1 is backward Euler, every later step is TR.
    'be_after_mesh_change': step 1 and every step that follows a mesh change
    are backward Euler, all other steps TR. Returns a list of messages."""
    if not os.path.exists(out_file):
        return ["no log"]
    steps = []       # (step number, is_be, follows_mesh_change)
    after_change = False
    for ln in open(out_file, errors="replace"):
        m = re.match(r"\s*TR-AB2: t = \S+ \(step (\d+), (backward Euler|\|e\|)", ln)
        if m:
            steps.append((int(m.group(1)), m.group(2) == "backward Euler", after_change))
            after_change = False
        elif ln.lstrip().startswith("adaptive:"):
            after_change = True
    if not steps:
        return ["no steps in the log"]
    msgs = []
    if steps[0][0] != 1 or not steps[0][1]:
        msgs.append("step 1 is not a backward Euler step")
    for n, is_be, follows in steps[1:]:
        want = (mode == "be_after_mesh_change" and follows)
        if is_be != want:
            msgs.append(f"step {n} is {'BE' if is_be else 'TR'}, expected {'BE' if want else 'TR'}")
            break
    n_be = sum(1 for _, b, _ in steps if b)
    if not msgs and mode == "be_first" and n_be != 1:
        msgs.append(f"{n_be} backward Euler steps instead of 1")
    if not msgs and mode == "be_after_mesh_change" and n_be < 2:
        msgs.append("no backward Euler step after a mesh change")
    return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="build directory with the executables")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of the cases")
    ap.add_argument("--update", action="store_true", help="write the references")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-14)
    ap.add_argument("--keep", action="store_true", help="keep the run directories")
    args = ap.parse_args()
    names = args.cases or (list(CASES) + list(VERIFY))
    failed = []
    for name in [n for n in names if n in VERIFY]:
        exe, base, over, tkey, bounds = VERIFY[name][:5]
        step_check = VERIFY[name][5] if len(VERIFY[name]) > 5 else None
        exe_path = os.path.join(os.path.abspath(args.build), exe)
        if not os.path.exists(exe_path):
            print(f"{name:26s} SKIPPED (no {exe})")
            continue
        run_dir = tempfile.mkdtemp(prefix=f"reg_{name}_")
        inp = prepare_input(base, over, run_dir, tkey)
        p = subprocess.run([exe_path, inp], cwd=run_dir, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=1800)
        tab = os.path.join(run_dir, "table.txt")
        if p.returncode != 0 or not os.path.exists(tab):
            print(f"{name:26s} FAILED (exit code {p.returncode}), see {run_dir}")
            failed.append(name)
            continue
        lines = [l for l in open(tab).read().splitlines() if l.strip()]
        header = lines[0].lstrip("#").split()
        rows = [l.split() for l in lines[1:]
                if not l.lstrip().startswith("#") and NUM.fullmatch(l.split()[0])]
        msgs = []
        for col, bound in bounds.items():
            if col not in header:
                msgs.append(f"column {col} missing")
                continue
            j = header.index(col)
            vals = [abs(float(r[j])) for r in rows if len(r) > j and NUM.fullmatch(r[j])]
            vmax = max(vals) if vals else float("nan")
            if not vmax <= bound:
                msgs.append(f"max {col} = {vmax:.3e} > {bound:.0e}")
        if step_check is not None:
            msgs += check_step_sequence(os.path.join(run_dir, "run.out"), step_check)
        if msgs:
            print(f"{name:26s} NOT EXACT: " + "; ".join(msgs))
            failed.append(name)
        else:
            worst = ", ".join(f"{c} {max(abs(float(r[header.index(c)])) for r in rows if len(r) > header.index(c)):.1e}"
                              for c in bounds)
            seq = ""
            if step_check is not None:
                be = sum(1 for l in open(os.path.join(run_dir, "run.out"), errors="replace")
                         if "backward Euler, next dt" in l)
                seq = f"; steps: {be} BE, {len(rows) - be} TR, sequence ok"
            print(f"{name:26s} exact ({len(rows)} rows: {worst}{seq})")
            if not args.keep:
                shutil.rmtree(run_dir, ignore_errors=True)
    for name in [n for n in names if n in CASES]:
        exe, base, over, tkey = CASES[name]
        exe_path = os.path.join(os.path.abspath(args.build), exe)
        if not os.path.exists(exe_path):
            print(f"{name:24s} SKIPPED (no {exe})")
            continue
        run_dir = tempfile.mkdtemp(prefix=f"reg_{name}_")
        inp = prepare_input(base, over, run_dir, tkey)
        p = subprocess.run([exe_path, inp], cwd=run_dir, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=1800)
        res = extract(run_dir)
        ref_dir = os.path.join(REF, name)
        if p.returncode != 0:
            print(f"{name:24s} FAILED (exit code {p.returncode}), see {run_dir}")
            failed.append(name)
            continue
        if args.update:
            os.makedirs(ref_dir, exist_ok=True)
            open(os.path.join(ref_dir, "table.txt"), "w").write("\n".join(res["table"]) + "\n")
            open(os.path.join(ref_dir, "log.txt"), "w").write("\n".join(res["log"]) + "\n")
            print(f"{name:24s} reference written ({len(res['table'])} table lines, {len(res['log'])} log lines)")
        else:
            if not os.path.isdir(ref_dir):
                print(f"{name:24s} NO REFERENCE")
                failed.append(name)
                continue
            ref_t = [l for l in open(os.path.join(ref_dir, "table.txt")).read().splitlines() if l.strip()]
            ref_l = [l for l in open(os.path.join(ref_dir, "log.txt")).read().splitlines() if l.strip()]
            msgs = compare_lines(ref_t, res["table"], args.rtol, args.atol, "table") \
                + compare_lines(ref_l, res["log"], args.rtol, args.atol, "log")
            if msgs:
                print(f"{name:24s} DIFFERS")
                for m in msgs[:6]:
                    print("   " + m)
                failed.append(name)
            else:
                print(f"{name:24s} ok ({len(ref_t)} table lines, {len(ref_l)} log lines)")
        if not args.keep and name not in failed:
            shutil.rmtree(run_dir, ignore_errors=True)
    if failed:
        print("FAILED:", " ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
