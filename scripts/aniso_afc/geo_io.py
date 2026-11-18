"""
Write a triangulation of the unit square (0,1)^2, given by vertex coordinates
and triangles, as a ParMooN .GEO file for boundary_file Default_UnitSquare.

Boundary vertices are stored by their global boundary parameter (component k
plus the local parameter t in [0,1]) as in scripts/make_unitsquare_geo.py:
  0: y = 0, t = x;  1: x = 1, t = y;  2: y = 1, t = 1 - x;  3: x = 0, t = 1 - y.
Triangles are oriented counterclockwise.
"""
import numpy as np


def boundary_parameter(x, y, tol=1e-14):
    """global boundary parameter of a point on the boundary of (0,1)^2, None inside"""
    if abs(y) <= tol:
        return x
    if abs(x - 1.0) <= tol:
        return 1.0 + y
    if abs(y - 1.0) <= tol:
        return 2.0 + (1.0 - x)
    if abs(x) <= tol:
        return 3.0 + (1.0 - y)
    return None


def write_unit_square_geo(points, cells, path, title="triangulation of the unit square"):
    points = np.asarray(points, float)
    cells = np.array(cells, int)
    # counterclockwise orientation
    p = points[cells]
    area2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    if np.any(area2 == 0):
        raise ValueError("degenerate triangle")
    cw = area2 < 0
    cells[cw] = cells[cw][:, [0, 2, 1]]
    coord, knpr = [], []
    for x, y in points:
        t = boundary_parameter(x, y)
        if t is None:
            coord.append(f"{x:.17g} {y:.17g}")
            knpr.append(0)
        else:
            coord.append(f"{t:.17g} 0")
            knpr.append(1)
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write("n_elem   n_vert ignored  max_n_vert_per_elem  ignored\n")
        f.write(f"  {len(cells)}   {len(points)}   0   3   1\n")
        f.write("DCORVG\n" + "\n".join(coord) + "\n")
        f.write("KVERT\n" + "\n".join(f"{a + 1} {b + 1} {c + 1}" for a, b, c in cells) + "\n")
        f.write("KNPR\n" + " ".join(str(k) for k in knpr) + "\n")
        f.write("KMM\n    1    4\n")
    return path


def read_vtk_point_scalar(path, name=None):
    """vertex coordinates (N,2) and one point scalar (the first, or the one called name)"""
    import re
    txt = open(path).read()
    n = int(re.search(r"POINTS (\d+)", txt).group(1))
    pts = np.array(re.search(r"POINTS \d+ \w+\n(.*?)\nCELLS", txt, re.S).group(1).split(),
                   float).reshape(-1, 3)[:, :2]
    blocks = re.findall(r"SCALARS (\S+) \w+\nLOOKUP_TABLE default\n(.*?)(?=\n(?:SCALARS|VECTORS|CELL_DATA)|\Z)",
                        txt, re.S)
    for nm, data in blocks:
        if name is None or nm == name:
            return pts, np.array(data.split(), float)[:n], nm
    raise KeyError(f"no point scalar {name} in {path}")
