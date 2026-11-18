"""
Layer-adapted meshes of the unit square for the study of algebraic
stabilizations on anisotropic meshes (P1, written as ParMooN .GEO files with
geo_io.write_unit_square_geo).

1D node distributions (n cells), for a layer of width ~ eb (eb = eps / beta for
an exponential layer, sqrt(eps) for a parabolic one), sigma = 2, q = 1/2:
  shishkin            q n uniform cells in [0, tau], tau = min{q, sigma eb ln n}, the rest uniform
  bakhvalov_shishkin  x_i = -sigma eb ln(1 - (1 - 1/n) i / (q n)) for i <= q n (Linss), then uniform
  bakhvalov           x = phi(i/n), phi(t) = -sigma eb ln(1 - t/q) up to the point alpha where the
                      tangent of phi passes through (1, 1), then that tangent (C^1)
  duran               Duran & Lombardi: x_i = i h eb up to eb, then x_{i+1} = (1 + h) x_i up to 1, h such
                      that there are n cells (graded, no transition point)
and the Shishkin nodes with an explicit transition point tau (shishkin_points), at the right end,
the left end, both ends (n/4 cells in each layer region) or uniform. Tensor-product meshes are
split along the diagonal "/" (right), "\\" (left) or along both diagonals (crossed, with a node at
the centre of every rectangle).

The definitions are those of galerkit (galerkit.core.mesh, branch feature/anisotropic-afc), and
check_against_galerkit() verifies that the points and the triangles agree.
"""
import numpy as np


def shishkin_points(n, tau, side="right", a=0.0, b=1.0):
    L = b - a
    if side in ("right", "left"):
        if n % 2:
            raise ValueError("n must be even")
        t = min(tau, L / 2)
        coarse = np.linspace(0.0, L - t, n // 2 + 1)
        fine = np.linspace(L - t, L, n // 2 + 1)[1:]
        s = np.concatenate([coarse, fine])
        return a + (s if side == "right" else (L - s)[::-1])
    if side == "both":
        if n % 4:
            raise ValueError("n must be divisible by 4")
        t = min(tau, L / 4)
        left = np.linspace(0.0, t, n // 4 + 1)
        mid = np.linspace(t, L - t, n // 2 + 1)[1:]
        right = np.linspace(L - t, L, n // 4 + 1)[1:]
        return a + np.concatenate([left, mid, right])
    if side in (None, "none", "uniform"):
        return np.linspace(a, b, n + 1)
    raise ValueError("side must be right, left, both or none")


def layer_at_zero(n, kind, eb, sigma=2.0, q=0.5):
    t = np.arange(n + 1) / n
    if kind == "uniform" or sigma * eb * np.log(n) >= q:
        return t
    if kind == "shishkin":
        tau = sigma * eb * np.log(n)
        m = int(round(q * n))
        return np.concatenate([np.linspace(0, tau, m + 1), np.linspace(tau, 1, n - m + 1)[1:]])
    if kind == "bakhvalov_shishkin":
        m = int(round(q * n))
        i = np.arange(m + 1)
        xl = -sigma * eb * np.log(1 - (1 - 1.0 / n) * i / m)
        return np.concatenate([xl, np.linspace(xl[-1], 1, n - m + 1)[1:]])
    if kind == "bakhvalov":
        def phi(s):
            return -sigma * eb * np.log(1 - s / q)

        def dphi(s):
            return sigma * eb / (q - s)
        lo, hi = 0.0, q * (1 - 1e-15)            # tangent through (1, 1): g(alpha) = 0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            g = dphi(mid) * (1 - mid) - (1 - phi(mid))
            lo, hi = (mid, hi) if g < 0 else (lo, mid)
        a = 0.5 * (lo + hi)
        x = np.where(t <= a, phi(np.minimum(t, a)), phi(a) + dphi(a) * (t - a))
        x[-1] = 1.0
        return x
    if kind == "duran":
        def nodes(h):
            x = [0.0]
            while x[-1] + 1e-15 < eb:
                x.append(x[-1] + h * eb)
            while x[-1] < 1.0:
                x.append(x[-1] * (1 + h))
            x[-1] = 1.0
            if len(x) > 2 and (x[-1] - x[-2]) < 0.5 * (x[-2] - x[-3]):
                del x[-2]                        # no tiny last cell
            return np.array(x)
        lo, hi = 1e-4, 10.0                      # the number of cells decreases with h
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            lo, hi = (lo, mid) if len(nodes(mid)) - 1 <= n else (mid, hi)
        x = nodes(hi)
        while len(x) - 1 < n:                    # pad: split the largest cells
            k = int(np.argmax(np.diff(x)))
            x = np.insert(x, k + 1, 0.5 * (x[k] + x[k + 1]))
        return x
    raise ValueError(f"unknown kind {kind}")


def layer_points(n, kind, eb, side="right", sigma=2.0):
    if side == "left":
        return layer_at_zero(n, kind, eb, sigma)
    if side == "right":
        return (1.0 - layer_at_zero(n, kind, eb, sigma))[::-1]
    if side in ("both", "interior"):
        if n % 2:
            raise ValueError("n must be even")
        sig = sigma * np.log(n) / np.log(n // 2)  # transition point sigma eb ln n as on [0, 1]
        half = 0.5 * layer_at_zero(n // 2, kind, 2 * eb, sig, q=0.5)
        if side == "both":
            return np.concatenate([half, (1.0 - half)[::-1][1:]])
        return np.concatenate([0.5 - half[::-1], 0.5 + half[1:]])
    raise ValueError("side must be right, left, both or interior")


def grid_triangles(nx, ny, idx, diagonal):
    """triangles of a structured (nx+1) x (ny+1) vertex grid with index array idx[i, j]"""
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    I, J = I.ravel(), J.ravel()
    a, b, c, d = idx[I, J], idx[I + 1, J], idx[I + 1, J + 1], idx[I, J + 1]
    if diagonal == "right":
        flip = np.zeros(I.size, bool)
    elif diagonal == "left":
        flip = np.ones(I.size, bool)
    else:
        raise ValueError("diagonal must be right or left")
    t1 = np.where(flip[:, None], np.stack([a, b, d], 1), np.stack([a, b, c], 1))
    t2 = np.where(flip[:, None], np.stack([b, c, d], 1), np.stack([a, c, d], 1))
    return np.vstack([t1, t2])


def graded_points(n, centers, widths, weights, a=0.0, b=1.0):
    """n + 1 nodes on [a, b], piecewise uniform: round(weight_k n) cells in each layer region
    [c_k - w_k, c_k + w_k], the remaining cells over the gaps in proportion to their lengths"""
    centers, widths, weights = (np.atleast_1d(np.asarray(v, float)) for v in (centers, widths, weights))
    order = np.argsort(centers)
    centers, widths, weights = centers[order], widths[order], weights[order]
    lay = [(c - w, c + w) for c, w in zip(centers, widths)]
    bp = [a] + [x for ab in lay for x in ab] + [b]
    if np.any(np.diff(bp) <= 0):
        raise ValueError("graded_points: layer regions overlap or leave [a, b]")
    nl = [max(1, int(round(wt * n))) for wt in weights]
    gaps = [(bp[2 * k], bp[2 * k + 1]) for k in range(len(lay) + 1)]
    rest = n - sum(nl)
    if rest < len(gaps):
        raise ValueError("graded_points: too few cells left for the gaps")
    glen = np.array([g1 - g0 for g0, g1 in gaps])
    ng = np.maximum(1, np.floor(rest * glen / glen.sum()).astype(int))
    while ng.sum() < rest:
        ng[np.argmax(glen / ng)] += 1
    while ng.sum() > rest:
        k = np.argmax(np.where(ng > 1, ng / glen, -1))
        ng[k] -= 1
    pts = [np.array([a])]
    for k, (g0, g1) in enumerate(gaps):
        pts.append(np.linspace(g0, g1, ng[k] + 1)[1:])
        if k < len(lay):
            pts.append(np.linspace(lay[k][0], lay[k][1], nl[k] + 1)[1:])
    return np.concatenate(pts)


def polar_square_mesh(radii, n_theta, n_outer=None, r_star=None, diagonal="right"):
    """mesh of the unit square aligned with the circles r = const around (0, 0): rings at the radii
    (0 = radii[0] < ... < radii[-1] = r_star < 1), n_outer rings blending the circle r_star into the
    square boundary (r = r_star + s (R(theta) - r_star), R = 1 / max(cos, sin)), n_theta (even)
    uniform angular cells in [0, pi/2]; a fan around the origin, the quadrilaterals split by diagonal"""
    radii = np.asarray(radii, float)
    if radii[0] != 0.0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must start at 0 and increase")
    if n_theta % 2:
        raise ValueError("n_theta must be even (the corner (1, 1) is a node)")
    r_star = radii[-1] if r_star is None else float(r_star)
    n_outer = max(2, int(round(len(radii) * (1.2 - r_star)))) if n_outer is None else int(n_outer)
    th = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    R = 1.0 / np.maximum(np.cos(th), np.sin(th))
    rings = [r * np.ones_like(th) for r in radii[1:]]
    for k in range(1, n_outer + 1):
        rings.append(r_star + (k / n_outer) * (R - r_star))
    rings = np.array(rings)
    P = np.vstack([np.zeros((1, 2)),
                   np.stack([(rings * np.cos(th)).ravel(), (rings * np.sin(th)).ravel()], 1)])
    nr = rings.shape[0]
    idx = 1 + np.arange(nr * (n_theta + 1)).reshape(nr, n_theta + 1)
    fan = np.stack([np.zeros(n_theta, int), idx[0, :-1], idx[0, 1:]], 1)
    T = np.vstack([fan, grid_triangles(nr - 1, n_theta, idx, diagonal)])
    P[np.abs(P[:, 0] - 1.0) < 1e-12, 0] = 1.0       # boundary nodes exactly on the square
    P[np.abs(P[:, 1] - 1.0) < 1e-12, 1] = 1.0
    P[np.abs(P[:, 0]) < 1e-14, 0] = 0.0
    P[np.abs(P[:, 1]) < 1e-14, 1] = 0.0
    return P, T


# Example 5 (profile transported along circles): layers along r = 0.375, 0.5, 0.625, 0.75
ROT_EPS = 1e-5
ROT_CENTERS = np.array([0.375, 0.5, 0.625, 0.75])
ROT_WEIGHTS = np.array([0.25, 0.1, 0.25, 0.1])
ROT_R_STAR = 0.9


def rotating_mesh(kind, n, eps=ROT_EPS):
    """meshes of Example 5: uniform, tensor_layer (tensor product refined around the layer radii),
    polar (uniform radii), polar_layer (radii refined around the layer radii, Shishkin type)"""
    tau = 2.0 * np.sqrt(eps) * np.log(n)
    if kind == "uniform":
        g = np.linspace(0.0, 1.0, n + 1)
        return tensor_mesh(g, g, "right")
    if kind == "tensor_layer":
        g = graded_points(n, ROT_CENTERS, tau * np.ones(4), ROT_WEIGHTS, 0.0, 1.0)
        return tensor_mesh(g, g, "right")
    n_outer = max(2, n // 8)
    k = n - n_outer
    if kind == "polar":
        radii = np.linspace(0.0, ROT_R_STAR, k + 1)
    elif kind == "polar_layer":
        radii = graded_points(k, ROT_CENTERS, tau * np.ones(4), ROT_WEIGHTS, 0.0, ROT_R_STAR)
    else:
        raise ValueError(kind)
    return polar_square_mesh(radii, n, n_outer=n_outer)


def tensor_mesh(xs, ys, diagonal="right"):
    """points (N, 2) and triangles (M, 3) of the tensor-product mesh xs x ys"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    nx, ny = xs.size - 1, ys.size - 1
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(P.shape[0]).reshape(nx + 1, ny + 1)
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    I, J = I.ravel(), J.ravel()
    a, b, c, d = idx[I, J], idx[I + 1, J], idx[I + 1, J + 1], idx[I, J + 1]
    if diagonal == "crossed":
        cen = P.shape[0] + np.arange(I.size)
        Pc = np.stack([0.5 * (xs[I] + xs[I + 1]), 0.5 * (ys[J] + ys[J + 1])], 1)
        T = np.vstack([np.stack([a, b, cen], 1), np.stack([b, c, cen], 1),
                       np.stack([c, d, cen], 1), np.stack([d, a, cen], 1)])
        return np.vstack([P, Pc]), T
    if diagonal == "right":                        # "/": diagonal (i,j)-(i+1,j+1)
        T = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    elif diagonal == "left":                       # "\": diagonal (i+1,j)-(i,j+1)
        T = np.vstack([np.stack([a, b, d], 1), np.stack([b, c, d], 1)])
    else:
        raise ValueError("diagonal must be right, left or crossed")
    return P, T


def shishkin_mesh(n, tau_x=None, tau_y=None, side_x="right", side_y="right", diagonal="right"):
    xs = shishkin_points(n, tau_x, side_x) if tau_x is not None else np.linspace(0.0, 1.0, n + 1)
    ys = shishkin_points(n, tau_y, side_y) if tau_y is not None else np.linspace(0.0, 1.0, n + 1)
    return tensor_mesh(xs, ys, diagonal)


def layer_mesh(n, kind, eb_x=None, eb_y=None, side_x="right", side_y="right", diagonal="right", sigma=2.0):
    xs = layer_points(n, kind, eb_x, side_x, sigma) if eb_x is not None else np.linspace(0.0, 1.0, n + 1)
    ys = layer_points(n, kind, eb_y, side_y, sigma) if eb_y is not None else np.linspace(0.0, 1.0, n + 1)
    return tensor_mesh(xs, ys, diagonal)


def mesh_stats(P, T):
    """nodes, cells, largest aspect ratio (longest edge^2 / (2 area)), shortest edge, largest angle (deg)"""
    Q = P[T]
    e = np.stack([np.linalg.norm(Q[:, (k + 1) % 3] - Q[:, k], axis=1) for k in range(3)], 1)
    area = 0.5 * np.abs((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
                        - (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))
    ang = []
    for k in range(3):
        u = Q[:, (k + 1) % 3] - Q[:, k]
        v = Q[:, (k + 2) % 3] - Q[:, k]
        c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return dict(nodes=len(P), cells=len(T), max_aspect=float((e.max(1) ** 2 / (2 * area)).max()),
                h_min=float(e.min()), max_angle=float(np.max(ang)))


def check_against_galerkit():
    """compare points and triangles with galerkit.core.mesh (requires galerkit)"""
    from galerkit.core import mesh as gm

    def same(P, T, m):
        if P.shape != m.points.shape or np.abs(P - m.points).max() > 0:
            return False
        a = {tuple(sorted(t)) for t in T.tolist()}
        b = {tuple(sorted(t)) for t in m.cells.tolist()}
        return a == b
    eps = 1e-8
    ln = np.log
    cases = []
    for n in (16, 32, 64, 128):
        for diag in ("right", "left", "crossed"):
            cases.append((f"shishkin exp n={n} {diag}", shishkin_mesh(n, 2 * eps * ln(n), None, diagonal=diag),
                          gm.shishkin_mesh(n, 2 * eps * ln(n), None, diagonal=diag)))
            cases.append((f"shishkin parab n={n} {diag}",
                          shishkin_mesh(n, 2 * eps * ln(n), 2 * np.sqrt(eps) * ln(n), side_y="both", diagonal=diag),
                          gm.shishkin_mesh(n, 2 * eps * ln(n), 2 * np.sqrt(eps) * ln(n), side_y="both",
                                           diagonal=diag)))
        for kind in ("bakhvalov_shishkin", "bakhvalov", "duran"):
            cases.append((f"{kind} n={n}", layer_mesh(n, kind, eps / 2, eps / 3),
                          gm.layer_mesh(n, kind, eb_x=eps / 2, eb_y=eps / 3)))
            cases.append((f"{kind} parab n={n}", layer_mesh(n, kind, eps, np.sqrt(eps), side_y="both"),
                          gm.layer_mesh(n, kind, eb_x=eps, eb_y=np.sqrt(eps), side_y="both")))
            cases.append((f"{kind} interior n={n}", layer_mesh(n, kind, eps, np.sqrt(eps), side_y="interior"),
                          gm.layer_mesh(n, kind, eb_x=eps, eb_y=np.sqrt(eps), side_y="interior")))
    import importlib.util
    import os
    import galerkit
    ex = os.path.join(os.path.dirname(galerkit.__file__), "..", "..", "examples", "convection_diffusion", "2d",
                      "anisotropic_afc", "rotating_profile.py")
    spec = importlib.util.spec_from_file_location("rotating_profile", ex)
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    for n in (32, 64, 128):
        for kind in ("uniform", "tensor_layer", "polar", "polar_layer"):
            cases.append((f"rotating {kind} n={n}", rotating_mesh(kind, n), rp.make_mesh(kind, n)))
    bad = [name for name, (P, T), m in cases if not same(P, T, m)]
    return len(cases), bad


if __name__ == "__main__":
    n, bad = check_against_galerkit()
    print(f"{n} meshes compared with galerkit: {n - len(bad)} identical" + (f"; different: {bad}" if bad else ""))
