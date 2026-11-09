#!/usr/bin/env python3
"""
Plot the results produced by spg_study.py (steady Oseen / Navier-Stokes
studies of Afzal, Ahmed, John 2026) without rerunning anything.

spg_study.py stores one directory per run, <outdir>/<study>_P<k>P<l>_nu<nu>/,
with the per-level table 'table.txt'. This script collects those tables and
draws, per study,

  <study>_error.pdf          error ||(u-u_h,p-p_h)||_spg vs DoFs, one panel per pair,
                             with the O(h^k) reference line (uniform studies)
  <study>_estimator.pdf      the estimator eta vs DoFs
  <study>_effectivity.pdf    effectivity index vs DoFs
  <study>_contributions.pdf  eta_res, eta_div, eta_F, eta_delta, eta_mu vs DoFs
                             for one viscosity (--nu-contrib, default: closest to 1e-5)
  <study>_pairs.pdf          all pairs in one panel (error and effectivity), one
                             viscosity (--nu-pairs)
  <study>_summary.txt        the numbers as a text table

Examples:
  python3 spg_plot.py --outdir spg_results --study ex41
  python3 spg_plot.py --outdir spg_results --study ex41 ex42 --format png --max-dofs 1e6
"""
import math
import argparse
import glob
import os
import re

import numpy as np

COLUMNS = ["level", "cells", "dofs", "h", "error_spg", "eta", "eta_res",
           "eta_div", "eta_F", "eta_delta", "eta_mu", "effectivity",
           "div_uh", "rate_err", "rate_eta"]
TITLES = {"ex41": "Example 4.1 (Oseen, smooth)", "ex42": "Example 4.2 (Oseen, layers)",
          "ex51": "Example 5.1 (Navier-Stokes, smooth)",
          "ex52": "Example 5.2 (Navier-Stokes, layers)"}
UNIFORM = {"ex41": True, "ex42": False, "ex51": True, "ex52": False}



def fmt_rate7(x, first):
    """rate in a 7 character field, '---' for the first level (no previous
    level) or if not defined"""
    if first or x is None or not math.isfinite(x):
        return f"{'---':>7s}"
    return f"{x:7.2f}"

def read_table(path):
    rows = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        rows.append({k: float(x) for k, x in zip(COLUMNS, line.split())})
    return rows


def collect(outdir, study):
    """return {(pair, nu): rows}"""
    results = {}
    pat = re.compile(rf"^{study}_P(\d)P(\d)_nu([0-9.e+-]+)$")
    for d in sorted(glob.glob(os.path.join(outdir, f"{study}_P*P*_nu*"))):
        m = pat.match(os.path.basename(d))
        table = os.path.join(d, "table.txt")
        if not m or not os.path.exists(table):
            continue
        pair = f"{m.group(1)}/{m.group(2)}"
        nu = float(m.group(3))
        rows = read_table(table)
        if rows:
            results[(pair, nu)] = rows
    return results


def expected_order(pair):
    k, l = (int(x) for x in pair.split("/"))
    return k + 0.5 if k == l else float(k)


def pair_label(pair):
    k, l = pair.split("/")
    return f"$P_{k}/P_{l}$"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="spg_results")
    ap.add_argument("--study", nargs="+", default=["ex41"])
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--max-dofs", type=float, default=None)
    ap.add_argument("--nu-contrib", type=float, default=None,
                    help="viscosity for the contributions plot")
    ap.add_argument("--nu-pairs", type=float, default=None,
                    help="viscosity for the all-pairs plot")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for study in args.study:
        results = collect(args.outdir, study)
        if not results:
            print(f"{study}: no results in {args.outdir}")
            continue
        if args.max_dofs:
            results = {k: [r for r in v if r["dofs"] <= args.max_dofs]
                       for k, v in results.items()}
        pairs = sorted({k[0] for k in results}, key=lambda p: (int(p[0]), int(p[2])))
        nus = sorted({k[1] for k in results}, reverse=True)
        title = TITLES.get(study, study)
        uniform = UNIFORM.get(study, False)
        ncols = min(3, len(pairs))
        nrows = int(np.ceil(len(pairs)/ncols))

        def panels():
            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows),
                                     squeeze=False)
            return fig, axes.ravel()

        def finish(fig, axes, name):
            for ax in axes[len(pairs):]:
                ax.axis("off")
            h, l = axes[0].get_legend_handles_labels()
            if h:
                fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), frameon=False)
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))
            out = os.path.join(args.outdir, f"{study}_{name}.{args.format}")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")

        # error and estimator vs DoFs
        for key, ylabel, name in [("error_spg", r"$\|(u-u_h,p-p_h)\|_{spg}$", "error"),
                                  ("eta", r"$\eta$", "estimator")]:
            fig, axes = panels()
            for ax, pair in zip(axes, pairs):
                last = None
                for i, nu in enumerate(nus):
                    rows = results.get((pair, nu))
                    if not rows:
                        continue
                    d = np.array([r["dofs"] for r in rows])
                    y = np.array([r[key] for r in rows])
                    ax.loglog(d, y, "o-", ms=3, color=colors[i % 10], label=rf"$\nu={nu:g}$")
                    last = (d, y)
                if uniform and last is not None:
                    k = expected_order(pair)
                    d, y = last
                    ax.loglog(d, y[-1]*(d/d[-1])**(-k/2), "k--", lw=1, label=rf"$O(h^{{{k:g}}})$")
                ax.set_title(pair_label(pair))
                ax.set_xlabel("DoFs")
                ax.set_ylabel(ylabel)
            finish(fig, axes, name)

        # effectivity
        fig, axes = panels()
        for ax, pair in zip(axes, pairs):
            for i, nu in enumerate(nus):
                rows = results.get((pair, nu))
                if rows:
                    ax.semilogx([r["dofs"] for r in rows], [r["effectivity"] for r in rows],
                                "o-", ms=3, color=colors[i % 10], label=rf"$\nu={nu:g}$")
            ax.set_title(pair_label(pair))
            ax.set_xlabel("DoFs")
            ax.set_ylabel("effectivity index")
        finish(fig, axes, "effectivity")

        # contributions
        nu_c = args.nu_contrib or min(nus, key=lambda x: abs(np.log10(x) + 5))
        fig, axes = panels()
        parts = [("eta_res", r"$\eta_{K,res}$"), ("eta_div", r"$\eta_{K,div}$"),
                 ("eta_F", r"$\eta_F$"), ("eta_delta", r"$\eta_{K,\delta}$"),
                 ("eta_mu", r"$\eta_{K,\mu}$")]
        for ax, pair in zip(axes, pairs):
            rows = results.get((pair, nu_c))
            if rows:
                d = [r["dofs"] for r in rows]
                for i, (key, lab) in enumerate(parts):
                    ax.loglog(d, [r[key] for r in rows], "o-", ms=3, color=colors[i], label=lab)
            ax.set_title(rf"{pair_label(pair)}, $\nu={nu_c:g}$")
            ax.set_xlabel("DoFs")
            ax.set_ylabel("estimated value")
        finish(fig, axes, "contributions")

        # all pairs in one panel
        nu_p = args.nu_pairs or nus[-1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
        for i, pair in enumerate(pairs):
            rows = results.get((pair, nu_p))
            if not rows:
                continue
            d = [r["dofs"] for r in rows]
            ax1.loglog(d, [r["error_spg"] for r in rows], "o-", ms=3, color=colors[i], label=pair_label(pair))
            ax2.semilogx(d, [r["effectivity"] for r in rows], "o-", ms=3, color=colors[i], label=pair_label(pair))
        ax1.set_xlabel("DoFs"); ax1.set_ylabel(r"$\|(u-u_h,p-p_h)\|_{spg}$")
        ax2.set_xlabel("DoFs"); ax2.set_ylabel("effectivity index")
        ax1.legend(fontsize=8)
        fig.suptitle(rf"{title}, $\nu={nu_p:g}$")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(args.outdir, f"{study}_pairs.{args.format}")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")

        # summary
        out = os.path.join(args.outdir, f"{study}_summary.txt")
        with open(out, "w") as f:
            f.write(title + "\n")
            for pair in pairs:
                for nu in nus:
                    rows = results.get((pair, nu))
                    if not rows:
                        continue
                    f.write(f"\n{pair_label(pair)}  nu = {nu:g}\n")
                    f.write("level  cells   dofs    error_spg     eta        eff   rate_err rate_eta\n")
                    for r in rows:
                        f.write(f"{int(r['level']):5d} {int(r['cells']):6d} {int(r['dofs']):7d} "
                                f"{r['error_spg']:11.4e} {r['eta']:10.4e} {r['effectivity']:7.3f} "
                                f"{fmt_rate7(r['rate_err'], r is rows[0])} {fmt_rate7(r['rate_eta'], r is rows[0])}\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
