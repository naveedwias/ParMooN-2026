#!/usr/bin/env python3
"""Time step, mesh size and errors over time for the layer-switch examples 36/37.

Reads the runs written by transient_study.py (<prefix>_runs.jsonl) and draws
four panels sharing the time axis: the time step dt (log scale), the number of
cells, L2(u) and L2(p) (log scale). The switch intervals are shaded: NSE1 is
switched off on [c1, c], NSE2 switched on on [d, d2] (defaults 0.3, 0.5, 0.5,
0.7; change with --switch).

Usage (from the folder in which transient_study.py was run):
  python3 ../ParMooN-2026/scripts/layer_switch_plot.py \
      tnse_layer_switch_x0y0/tnse_layer_switch_x0y0_runs.jsonl:switch_a:"example 36" \
      tnse_layer_switch_x1y0/tnse_layer_switch_x1y0_runs.jsonl:switch_b:"example 37" \
      --out layer_switch_time_step

Each argument is <runs.jsonl>[:<run name>[:<label>]]; without a run name the
last run of the file is used. Writes <out>.png (200 dpi) and <out>.pdf.
"""
import argparse
import io
import json

import numpy as np

# reference palette (dataviz skill, light mode): categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SHADE = "#eeede8"


def load(spec):
    parts = spec.split(":")
    path = parts[0]
    run = parts[1] if len(parts) > 1 and parts[1] else None
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if run is not None:
        sel = [r for r in recs if r.get("run") == run or r.get("key", "").startswith(run)
               or r.get("run", "").startswith(run)]
        if not sel:
            raise SystemExit(f"{path}: no run '{run}' (runs: {[r.get('run') for r in recs]})")
        rec = sel[-1]
    else:
        rec = recs[-1]
    label = parts[2] if len(parts) > 2 else (run or path)
    lines = rec["table"].strip().split("\n")
    head = lines[0].lstrip("#").split()
    data = np.loadtxt(io.StringIO("\n".join(lines[1:])), ndmin=2)
    return label, {k: data[:, i] for i, k in enumerate(head)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<runs.jsonl>[:<run name>[:<label>]]")
    ap.add_argument("--out", default="layer_switch_time_step")
    ap.add_argument("--switch", type=float, nargs=4, default=[0.3, 0.5, 0.5, 0.7],
                    metavar=("C1", "C", "D", "D2"))
    args = ap.parse_args()
    runs = [load(s) for s in args.runs]

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
    panels = [("dt", "time step dt", True), ("cells", "cells", False),
              ("L2(u)", "L2(u) error", True), ("L2(p)", "L2(p) error", True)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 8.4), sharex=True)
    c1, c, d, d2 = args.switch
    for ax, (key, title, logy) in zip(axes, panels):
        ax.axvspan(c1, c, color=SHADE, zorder=0, lw=0)
        if d2 > d:
            ax.axvspan(d, d2, color=SHADE, zorder=0, lw=0)
        for k, (label, tab) in enumerate(runs):
            if key not in tab:
                continue
            ax.plot(tab["time"], tab[key], color=SERIES[k % len(SERIES)], lw=1.6,
                    marker="o" if key == "dt" else None, ms=2.5, label=label,
                    drawstyle="steps-post" if key == "cells" else "default")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9.5, pad=4)
        if d2 > d and abs(c - d) < 1e-12:
            ax.axvline(c, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    top = axes[0]
    tr = matplotlib.transforms.blended_transform_factory(top.transData, top.transAxes)
    top.text((c1 + c)/2, 0.12, "NSE1\nswitched off", ha="center", va="bottom",
             color=INK2, fontsize=8, transform=tr)
    top.text((d + d2)/2, 0.12, "NSE2\nswitched on", ha="center", va="bottom",
             color=INK2, fontsize=8, transform=tr)
    axes[-1].set_xlabel("time t")
    axes[-1].set_xlim(min(t["time"].min() for _, t in runs) - 0.01,
                      max(t["time"].max() for _, t in runs) + 0.01)
    if len(runs) > 1:
        top.legend(frameon=False, loc="lower right", fontsize=8.5, bbox_to_anchor=(1.0, 0.05))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
