#!/usr/bin/env python3
"""
Write a ParMooN GEO file for a uniform N x N triangulation of the unit square
(each square split into two triangles with the same orientation as
data/mesh/TwoTriangles.GEO, i.e. the diagonal from (x+h, y) to (x, y+h)).
Boundary vertices are given by the global boundary parameter (as required by
the Default_UnitSquare / UnitSquare.PRM description: 0: y=0, 1: x=1, 2: y=1,
3: x=0), interior vertices by their coordinates.

Usage: make_unitsquare_geo.py N [output file]
"""
import sys

def main():
    n = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else f"UnitSquare{n}x{n}_tria.GEO"
    h = 1.0 / n
    # vertices: boundary vertices first (counterclockwise from (0,0)), then interior
    index = {}
    lines_coord = []
    knpr = []
    def add(i, j, bnd, param):
        index[(i, j)] = len(lines_coord) + 1
        if bnd:
            lines_coord.append(f"{param:.16g} 0")
            knpr.append(1)
        else:
            lines_coord.append(f"{i*h:.16g} {j*h:.16g}")
            knpr.append(0)
    for i in range(n):          # bottom: y = 0, parameter x
        add(i, 0, True, i*h)
    for j in range(n):          # right: x = 1, parameter 1 + y
        add(n, j, True, 1 + j*h)
    for i in range(n, 0, -1):   # top: y = 1, parameter 2 + (1 - x)
        add(i, n, True, 2 + (1 - i*h))
    for j in range(n, 0, -1):   # left: x = 0, parameter 3 + (1 - y)
        add(0, j, True, 3 + (1 - j*h))
    for j in range(1, n):
        for i in range(1, n):
            add(i, j, False, 0.)
    cells = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = index[(i, j)], index[(i+1, j)], index[(i+1, j+1)], index[(i, j+1)]
            cells.append((a, b, d))
            cells.append((b, c, d))
    with open(out, "w") as f:
        f.write(f"uniform {n}x{n} triangulation of the unit square [0,1]^2\n")
        f.write("n_elem   n_vert ignored  max_n_vert_per_elem  ignored\n")
        f.write(f"  {len(cells)}   {len(lines_coord)}   0   3   1\n")
        f.write("DCORVG\n")
        f.write("\n".join(lines_coord) + "\n")
        f.write("KVERT\n")
        f.write("\n".join(f"{a} {b} {c}" for a, b, c in cells) + "\n")
        f.write("KNPR\n")
        f.write(" ".join(str(k) for k in knpr) + "\n")
        f.write("KMM\n    1    4\n")
    print(f"wrote {out}: {len(cells)} cells, {len(lines_coord)} vertices")

if __name__ == "__main__":
    main()
