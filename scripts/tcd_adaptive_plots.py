#!/usr/bin/env python3
"""
Post-processing for the adaptive SUPG runs (de Frutos, Garcia-Archilla, John,
Novo 2014) made with parMooN_tcd2d_supg_adaptive:

  * draw the meshes written by the program as PostScript files (mesh_*.ps or
    any ParMooN 'domain.PS' file) as PNG/PDF images, optionally a zoom,
  * plot the table (number of cells, estimator and, if available, the
    effectivity index versus time).

Examples:
  python3 tcd_adaptive_plots.py --mesh mesh_000120.ps mesh_000240.ps
  python3 tcd_adaptive_plots.py --mesh mesh_000240.ps --zoom 0.9 1 0 0.1
  python3 tcd_adaptive_plots.py --table ex3_table.txt
"""
import argparse
import os
import re

import numpy as np


def read_ps_mesh(path):
    """Return the list of cells (each a list of (x,y) vertices) of a ParMooN
    PostScript mesh file, with coordinates normalised to the bounding box."""
    txt = open(path).read()
    cells = []
    for block in re.findall(r"newpath\n((?:[-\d.]+ [-\d.]+ [ML]\n)+)closepath", txt):
        pts = [(float(a), float(b))
               for a, b in re.findall(r"([-\d.]+) ([-\d.]+) [ML]", block)]
        # the last point repeats the first one
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        cells.append(pts)
    allp = np.array([p for c in cells for p in c])
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    scale = max(x1 - x0, y1 - y0)
    return [[((x - x0)/scale, (y - y0)/scale) for x, y in c] for c in cells]


def plot_mesh(path, out, zoom=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    cells = read_ps_mesh(path)
    segs = []
    for c in cells:
        n = len(c)
        for i in range(n):
            segs.append([c[i], c[(i+1) % n]])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(LineCollection(segs, colors="k", linewidths=0.25))
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title if title else f"{os.path.basename(path)} ({len(cells)} cells)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(cells)} cells)")


def plot_table(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    t, cells, dofs, eta, err, eff = (data[:, i] for i in range(6))
    has_error = np.all(err > 0)
    fig, axes = plt.subplots(1, 3 if has_error else 2,
                             figsize=(12 if has_error else 8, 3.5))
    axes[0].plot(t, cells, "k-")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("number of cells")
    axes[1].semilogy(t, eta, "b-", label=r"$\eta$")
    if has_error:
        axes[1].semilogy(t, err, "r--", label=r"$\|u-u_h\|_{SUPG}$")
        axes[2].plot(t, eff, "k-")
        axes[2].set_xlabel("t")
        axes[2].set_ylabel("effectivity index")
    axes[1].set_xlabel("t")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", nargs="*", default=[], help="PostScript mesh files")
    ap.add_argument("--zoom", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"),
                    help="plot only this window")
    ap.add_argument("--table", nargs="*", default=[], help="table files")
    ap.add_argument("--format", default="png", choices=["png", "pdf"])
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for m in args.mesh:
        base = os.path.splitext(os.path.basename(m))[0]
        suffix = "_zoom" if args.zoom else ""
        plot_mesh(m, os.path.join(args.outdir, f"{base}{suffix}.{args.format}"),
                  args.zoom)
    for tb in args.table:
        base = os.path.splitext(os.path.basename(tb))[0]
        plot_table(tb, os.path.join(args.outdir, f"{base}.{args.format}"))


if __name__ == "__main__":
    main()
