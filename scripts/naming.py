"""
Names of the output folders and files of the study and plotting scripts.

Every script that writes results uses the prefix <problem>_<example>:
the output folder is named <problem>_<example>, and every file and run
folder inside it is named <problem>_<example>_<previous name>. The
PostScript meshes of the adaptive programs (mesh_t*.ps) keep their names.

  problem  (from the program)                example (from the example number)
  tnse      tnse2d_adaptive_spg              Example_TimeNSE2D / Example_NSE2D /
  tnse_bdf2 tnse2d_bdf2imex_cip              Example_TimeCD2D, see EXAMPLES
  tcd       tcd2d_adaptive_tr
  oseen/nse oseen2d_spg_adaptive (problem_type 3 / 5)
  tnse_old, tcd_old  the deprecated monolithic programs
"""
import re

# problem name per program key of transient_study.py / spg_study.py
PROBLEMS = {
    "tnse": "tnse",
    "bdf2": "tnse_bdf2",
    "tcd": "tcd",
    "tnse_old": "tnse_old",
    "tcd_old": "tcd_old",
}

# example names per example class
TIME_NSE = {23: "aaj_smooth", 25: "bdf2_smooth", 26: "lake_at_rest", 27: "shear_layer",
            28: "poly_space", 30: "poly_space_time2", 31: "poly_space_time1",
            32: "translating_vortex", 34: "poly_neumann", 35: "taylor_green"}
STEADY_NSE = {21: "aaj_smooth", 22: "aaj_layers", 29: "spg_polynomial",
              33: "spg_polynomial_neumann"}
TIME_CD = {11: "fgjn_ex2", 14: "fgjn_ex3", 15: "fgjn_ex4", 16: "fgjn_ex5"}
EXAMPLES = {"tnse": TIME_NSE, "tnse_bdf2": TIME_NSE, "tnse_old": TIME_NSE,
            "tcd": TIME_CD, "tcd_old": TIME_CD, "oseen": STEADY_NSE, "nse": STEADY_NSE}


def example_name(problem, example):
    """name of an example number for a problem; 'example<N>' if it has none"""
    try:
        n = int(float(example))
    except (TypeError, ValueError):
        return f"example{example}"
    return EXAMPLES.get(problem, {}).get(n, f"example{n}")


def prefix(problem, example):
    """<problem>_<example>, e.g. tnse_taylor_green"""
    return f"{problem}_{example_name(problem, example)}"


def steady_problem(problem_type):
    """problem name of the steady SPG program: oseen (3) or nse (5)"""
    return "nse" if int(float(problem_type)) == 5 else "oseen"


def read_param(lines, key):
    """value of 'key: value' in the lines of a ParMooN input file (None if absent)"""
    for ln in lines:
        m = re.match(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$", ln)
        if m:
            return m.group(1)
    return None


def prefix_from_input(dat_file, problem=None):
    """prefix of the run whose input file is dat_file; the problem is taken
    from the argument, else from the program-specific keys in the file"""
    lines = open(dat_file).read().splitlines()
    example = read_param(lines, "example")
    if problem is None:
        if read_param(lines, "bdf2_write_table") is not None:
            problem = "tnse_bdf2"
        elif read_param(lines, "tnse_write_table") is not None:
            problem = "tnse"
        elif read_param(lines, "tcd_write_table") is not None:
            problem = "tcd"
        elif read_param(lines, "spg_write_table") is not None:
            problem = steady_problem(read_param(lines, "problem_type") or 3)
        else:
            problem = "run"
    return prefix(problem, example)
