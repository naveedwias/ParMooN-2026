#!/usr/bin/env python3
"""Convergence with the adaptive tolerance for the layer-switch examples 36/37.

For a series of runs (e.g. TOL1 = 5e-2, 2.5e-2, 1.25e-2) plots, against the
mean number of degrees of freedom over time, the maximum over time of
L2(u), L2(p), the error in the SUPG/PSPG norm and the estimator eta; prints
the same numbers with the observed orders with respect to the DoFs.

Usage:
  python3 layer_switch_convergence.py <runs.jsonl>[:<run>[:<label>]] ... --out name
"""
import argparse

import numpy as np

from layer_switch_plot import load_many, SERIES, INK, INK2, MUTED, GRID, SURFACE

QUANTITIES = [("L2(u)", "max_t L2(u)"), ("L2(p)", "max_t L2(p)"),
              ("error", "max_t error (SPG norm)"), ("eta", "max_t eta")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="layer_switch_convergence")
    ap.add_argument("--after", type=float, default=None,
                    help="also report the maxima over t >= AFTER")
    args = ap.parse_args()
    runs = [lt for s in args.runs for lt in load_many(s)]
    rows = []
    for label, tab in runs:
        r = {"label": label, "dofs": tab["dofs"].mean(), "cells": tab["cells"].mean(),
             "steps": len(tab["time"])}
        for key, _ in QUANTITIES:
            r[key] = tab[key].max()
            if args.after is not None:
                sel = tab[key][tab["time"] >= args.after]
                r[key + "_after"] = sel.max() if sel.size else float("nan")
        rows.append(r)
    rows.sort(key=lambda r: r["dofs"])
    head = "run  mean_dofs  mean_cells  steps  " + "  ".join(f"max {k}  order" for k, _ in QUANTITIES)
    print(head)
    for i, r in enumerate(rows):
        line = f"{r['label']}  {r['dofs']:.0f}  {r['cells']:.0f}  {r['steps']}"
        for k, _ in QUANTITIES:
            if i == 0:
                order = "-"
            else:
                q = rows[i - 1]
                order = f"{np.log(q[k]/r[k])/np.log(r['dofs']/q['dofs']):.2f}"
            line += f"  {r[k]:.3e}  {order}"
        print(line)
    if args.after is not None:
        print(f"maxima over t >= {args.after}: " + "; ".join(
            f"{r['label']}: " + ", ".join(f"{k} {r[k + '_after']:.3e}" for k, _ in QUANTITIES)
            for r in rows))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(10, 3.0))
    x = np.array([r["dofs"] for r in rows])
    for ax, (k, title) in zip(axes, QUANTITIES):
        y = np.array([r[k] for r in rows])
        ax.loglog(x, y, color=SERIES[0], lw=1.6, marker="o", ms=5)
        for xi, yi, r in zip(x, y, rows):
            ax.annotate(r["label"], (xi, yi), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=INK2)
        # reference slope -1/2 (O(h) for uniform refinement in 2D)
        ref = y[0]*(x/x[0])**-0.5
        ax.loglog(x, ref, color=MUTED, lw=0.8, ls=(0, (3, 3)), label="slope -1/2")
        ax.set_title(title, loc="left", fontsize=9)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(x, minor=True)
        ax.set_xticklabels([f"{v/1e3:.0f}k" for v in x], minor=True, fontsize=7.5)
        ax.set_xlabel("mean DoFs")
        ax.set_xlim(x.min()/1.15, x.max()*1.25)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
