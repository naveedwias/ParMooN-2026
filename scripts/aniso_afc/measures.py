"""
Measures of the study of algebraic stabilizations on layer-adapted meshes,
computed from ParMooN's output of cd2d_afc:
  <name>_nodal.txt  (algstab_write_nodal: x y u free, 17 digits)
  <name>_edges.txt  (algstab_write_edges: i j x_i y_i x_j y_j a_ij a_ji [d_ij active])
and the triangulation (points, triangles) of the run (meshes.py).

Error measures (as in the paper and in galerkit run_aniso.py / interior_layer.py /
rotating_profile.py):
  Examples 1-2  maximal nodal error (all nodes) and away from the layer regions
                (x <= 1 - tau_x, in Example 2 also y <= 1 - tau_y); the L2 error is
                ParMooN's (output_compute_errors, read from the log)
  Example 4     maximal nodal error against the reference 1/2 erfc((1/2-y)/(2 sqrt(eps x)))
                for 0.1 <= x <= 1 - tau_x, and the discrete L2 error with lumped P1 mass
                (sum_i |patch_i|/3 e_i^2) for x <= 1 - tau_x
  Example 5     maximal nodal error and lumped L2 error for r <= 1 against the reference of the
                radial heat equation u_theta = eps (u_rr + u_r / r) (radial_reference)
  bounds        undershoot max(0, lo - min u), overshoot max(0, max u - hi)
DMP checks (galerkit dmp.py):
  certificate   the sufficient mesh condition of the method on the edges with a free node:
                Kuzmin, MC: min{a_ij, a_ji} <= 0; F-BBK, S-BBK: max{a_ij, a_ji} <= tau_E = C0 |E|
                (tau_E = 0 if neither end is an interior free node); BJK, MUAS, SMUAS, BBK: none;
                edges where the stabilization is switched off need max{a_ij, a_ji} <= 0
                (tolerance 1e-12 max |d_ij|)
  local DMP     free nodes with u_i outside [min, max] of the neighbours by more than
                1e-8 (max u - min u)
  Laplace > 0   edges with a free vertex and a positive off-diagonal entry of the P1 Laplace
                matrix (not of Delaunay type), and the largest angle
"""
import numpy as np


# ------------------------------------------------------------------------------- reading
def read_nodal(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2], d[:, 3].astype(bool)


def read_edges(path):
    d = np.loadtxt(path)
    out = dict(xi=d[:, 2:4], xj=d[:, 4:6], aij=d[:, 6], aji=d[:, 7])
    if d.shape[1] >= 10:
        out.update(d=d[:, 8], active=d[:, 9].astype(bool))
    return out


def to_mesh_order(P_mesh, P_out, values):
    """values given at the points P_out, at the points P_mesh (nearest point)"""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(P_out).query(P_mesh)
    return np.asarray(values)[idx], float(dist.max())


def edge_indices(P_mesh, edges):
    from scipy.spatial import cKDTree
    t = cKDTree(P_mesh)
    _, i = t.query(edges["xi"])
    _, j = t.query(edges["xj"])
    return i, j


# ------------------------------------------------------------------------------- exact and reference solutions
def exact(example, eps, x, y):
    if example == 1:                                     # exponential layer
        return -np.expm1((x - 1) / eps) / (-np.expm1(-1 / eps))
    if example == 2:                                     # boundary layers, b = (2, 3)
        return (x * y ** 2 - y ** 2 * np.exp(2 * (x - 1) / eps) - x * np.exp(3 * (y - 1) / eps)
                + np.exp((2 * (x - 1) + 3 * (y - 1)) / eps))
    if example == 4:                                     # straight interior layer: reference
        from scipy.special import erfc
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 0.5 * erfc((0.5 - y) / (2 * np.sqrt(eps * x)))
        return np.where(x > 0, v, np.where(y > 0.5, 1.0, np.where(y < 0.5, 0.0, 0.5)))
    raise ValueError(f"no exact solution for example {example}")


def rotating_inflow(x, xi=1e-3):
    x = np.asarray(x, float)
    u = np.zeros_like(x)
    m = (x >= 0.375 - xi) & (x < 0.375)
    u[m] = (x[m] - 0.375) / xi + 1
    m = (x >= 0.375) & (x < 0.5)
    u[m] = -0.75 * (x[m] - 0.5) / 0.125 + 0.25
    m = (x >= 0.5) & (x < 0.625)
    u[m] = 0.25 * (x[m] - 0.625) / 0.125 + 0.5
    m = (x >= 0.625) & (x < 0.625 + xi)
    u[m] = -0.5 * (x[m] - 0.625) / xi + 0.5
    m = (x >= 0.75) & (x <= 1)
    u[m] = 32 * (x[m] - 0.75) * (1 - x[m])
    return u


def radial_reference(eps=1e-5, xi=1e-3, nr=40000, ntheta=4000, r_max=1.5, last_only=False):
    """Example 5: u_theta = eps (u_rr + u_r / r), theta in [0, pi/2], u(r, 0) = inflow profile;
    finite volumes in r (u_r = 0 at r = 0 and r_max), two implicit Euler steps, then Crank-Nicolson
    (as galerkit problems.rotating_profile_reference). Returns r, theta, U[ntheta+1, nr+1];
    last_only: r and the profile at theta = pi/2 (the outlet x = 0), without storing U (1.3 GB)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    r = np.linspace(0.0, r_max, nr + 1)
    h = r[1] - r[0]
    th = np.linspace(0.0, 0.5 * np.pi, ntheta + 1)
    k = th[1] - th[0]
    rp = np.minimum(r + 0.5 * h, r_max)
    rm = np.maximum(r - 0.5 * h, 0.0)
    vol = np.where((r == 0) | (r == r_max), 0.5 * h * np.maximum(r, 0.25 * h), h * r)
    vol[0] = h * h / 8.0
    up = rp / h / vol
    lo = rm / h / vol
    up[-1] = 0.0
    lo[0] = 0.0
    L = sp.diags([lo[1:], -(up + lo), up[:-1]], [-1, 0, 1], format="csc")
    Id = sp.identity(nr + 1, format="csc")
    be = splu((Id - k * eps * L).tocsc())
    cn_l = splu((Id - 0.5 * k * eps * L).tocsc())
    cn_r = (Id + 0.5 * k * eps * L).tocsr()
    if last_only:
        u = rotating_inflow(r, xi)
        for j in range(ntheta):
            u = be.solve(u) if j < 2 else cn_l.solve(cn_r @ u)
        return r, u
    U = np.empty((ntheta + 1, nr + 1))
    U[0] = rotating_inflow(r, xi)
    for j in range(ntheta):
        U[j + 1] = be.solve(U[j]) if j < 2 else cn_l.solve(cn_r @ U[j])
    return r, th, U


_REF = {}


def rotating_reference_at(P, eps=1e-5):
    from scipy.interpolate import RegularGridInterpolator
    if eps not in _REF:
        _REF[eps] = radial_reference(eps)
    r, th, U = _REF[eps]
    f = RegularGridInterpolator((th, r), U, bounds_error=False, fill_value=0.0)
    R = np.hypot(P[:, 0], P[:, 1])
    T = np.arctan2(P[:, 1], P[:, 0])
    return f(np.stack([T, R], -1))


# ------------------------------------------------------------------------------- layer widths
SIGMA = 2.0


def layer_widths(example, eps, n):
    """tau_x, tau_y, layers in y at both ends (as galerkit run_aniso.layer_widths)"""
    ln = np.log(n)
    if example == 1:
        return SIGMA * eps * ln, None, False
    if example == 2:
        return SIGMA * eps / 2 * ln, SIGMA * eps / 3 * ln, False
    if example == 3:
        return SIGMA * eps * ln, SIGMA * np.sqrt(eps) * ln, True
    if example == 4:
        return SIGMA * eps * ln, SIGMA * np.sqrt(eps) * ln, False
    raise ValueError(example)


def lumped_weights(P, T):
    Q = P[T]
    area = 0.5 * np.abs((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
                        - (Q[:, 2, 0] - Q[:, 0, 0]) * (Q[:, 1, 1] - Q[:, 0, 1]))
    return np.bincount(T.ravel(), np.repeat(area / 3.0, 3), P.shape[0])


# ------------------------------------------------------------------------------- measures
def errors(example, eps, n, P, T, u):
    """error measures of one run (u at the points P of the mesh (P, T))"""
    out = {}
    if example in (1, 2):
        e = np.abs(u - exact(example, eps, P[:, 0], P[:, 1]))
        tx, ty, both = layer_widths(example, eps, n)
        away = P[:, 0] <= 1 - tx + 1e-14
        if ty is not None:
            away &= P[:, 1] <= 1 - ty + 1e-14
        out.update(Linf_nodes=float(e.max()), Linf_away=float(e[away].max()))
    elif example == 4:
        e = np.abs(u - exact(4, eps, P[:, 0], P[:, 1]))
        tx, ty, _ = layer_widths(4, eps, n)
        away = P[:, 0] <= 1 - tx + 1e-14
        mid = away & (P[:, 0] >= 0.1)
        w = lumped_weights(P, T)
        out.update(Linf_mid=float(e[mid].max()), L2_away_lumped=float(np.sqrt(np.sum(w[away] * e[away] ** 2))))
    elif example == 5:
        e = np.abs(u - rotating_reference_at(P, eps))
        inside = np.hypot(P[:, 0], P[:, 1]) <= 1 + 1e-14
        w = lumped_weights(P, T)
        out.update(Linf_r1=float(e[inside].max()), L2_r1_lumped=float(np.sqrt(np.sum(w[inside] * e[inside] ** 2))))
    lo, hi = {1: (0.0, 1.0), 3: (0.0, 1.0), 4: (0.0, 1.0), 5: (0.0, 1.0)}.get(example, (None, None))
    out.update(u_min=float(u.min()), u_max=float(u.max()))
    if lo is not None:
        out.update(undershoot=float(max(0.0, lo - u.min())), overshoot=float(max(0.0, u.max() - hi)))
    return out


ANY_MESH = ("bjk", "muas", "smuas", "bbk")


def dmp_certificate(method, P, free, edges, C0=3.0):
    """the sufficient DMP mesh condition of the method, edge by edge (galerkit dmp_certificate)"""
    i, j = edge_indices(P, edges)
    a1, a2 = edges["aij"], edges["aji"]
    touch = free[i] | free[j]
    d = edges.get("d", -np.maximum(np.maximum(a1, a2), 0.0))
    tol = 1e-12 * max(float(np.abs(d).max(initial=0.0)), 1e-300)
    if method in ANY_MESH:
        res = dict(condition="any mesh", n_bad=0, guaranteed=True)
    elif method in ("kuzmin", "kuzmin_edge", "mc"):
        bad = touch & (np.minimum(a1, a2) > tol)
        res = dict(condition="min(a_ij, a_ji) <= 0", n_bad=int(bad.sum()),
                   guaranteed=method != "mc" and not bad.any())
    elif method in ("fbbk", "sbbk"):
        onb = (np.abs(P[:, 0]) < 1e-14) | (np.abs(P[:, 0] - 1) < 1e-14) | (np.abs(P[:, 1]) < 1e-14) \
            | (np.abs(P[:, 1] - 1) < 1e-14)
        xi_node = free & ~onb
        w = np.where(xi_node[i] | xi_node[j], C0 * np.linalg.norm(P[j] - P[i], axis=1), 0.0)
        bad = touch & (np.maximum(a1, a2) > w + tol)
        res = dict(condition="max(a_ij, a_ji) <= tau_E", n_bad=int(bad.sum()), guaranteed=not bad.any())
    else:
        res = dict(condition="unknown", n_bad=-1, guaranteed=False)
    if "active" in edges:
        off = ~edges["active"]
        bad_off = touch & off & (np.maximum(a1, a2) > tol)
        res.update(n_switched_off=int((touch & off).sum()), n_bad_off=int(bad_off.sum()))
        res["n_bad"] += int(bad_off.sum())
        res["guaranteed"] = res["guaranteed"] and not bad_off.any()
    return res


def local_dmp(P, free, edges, u, rtol=1e-8):
    i, j = edge_indices(P, edges)
    umax = np.full(u.size, -np.inf)
    umin = np.full(u.size, np.inf)
    np.maximum.at(umax, i, u[j])
    np.maximum.at(umax, j, u[i])
    np.minimum.at(umin, i, u[j])
    np.minimum.at(umin, j, u[i])
    rng = max(float(np.ptp(u)), 1e-300)
    exc = np.maximum(u - umax, umin - u)[free] / rng
    return dict(n_viol=int((exc > rtol).sum()), excess=float(max(exc.max(initial=0.0), 0.0)))


def laplace_positive(P, T, free=None):
    """edges with a free vertex and a positive off-diagonal P1 Laplace entry; largest angle"""
    Q = P[T]
    det = (Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1]) - (Q[:, 2, 0] - Q[:, 0, 0]) * (Q[:, 1, 1] - Q[:, 0, 1])
    G = np.empty((len(T), 3, 2))
    G[:, 0] = np.stack([Q[:, 1, 1] - Q[:, 2, 1], Q[:, 2, 0] - Q[:, 1, 0]], 1) / det[:, None]
    G[:, 1] = np.stack([Q[:, 2, 1] - Q[:, 0, 1], Q[:, 0, 0] - Q[:, 2, 0]], 1) / det[:, None]
    G[:, 2] = np.stack([Q[:, 0, 1] - Q[:, 1, 1], Q[:, 1, 0] - Q[:, 0, 0]], 1) / det[:, None]
    vol = 0.5 * np.abs(det)
    if free is None:
        onb = (np.abs(P[:, 0]) < 1e-14) | (np.abs(P[:, 0] - 1) < 1e-14) | (np.abs(P[:, 1]) < 1e-14) \
            | (np.abs(P[:, 1] - 1) < 1e-14)
        free = ~onb
    N = len(P)
    pr = [(0, 1), (0, 2), (1, 2)]
    I = np.concatenate([T[:, a] for a, _ in pr])
    J = np.concatenate([T[:, b] for _, b in pr])
    V = np.concatenate([np.einsum("cd,cd->c", G[:, a], G[:, b]) * vol for a, b in pr])
    key = np.minimum(I, J).astype(np.int64) * (N + 1) + np.maximum(I, J)
    uk, inv = np.unique(key, return_inverse=True)
    a = np.bincount(inv, V)
    ei, ej = uk // (N + 1), uk % (N + 1)
    sel = free[ei] | free[ej]
    pos = sel & (a > 1e-12 * np.abs(a).max())
    gn = G / np.linalg.norm(G, axis=2, keepdims=True)
    ang = max(float(np.degrees(np.arccos(np.clip(-np.einsum("cd,cd->c", gn[:, a_], gn[:, b_]), -1, 1))).max())
              for a_, b_ in pr)
    return dict(n_positive=int(pos.sum()), max_angle=ang)
