"""Reference (non-shooting) committor solvers for overdamped dynamics
(Task 9, controller ruling R4; reused by later network-acceptance tasks).

Both solvers target the committor of overdamped Langevin dynamics with an
isotropic, position-independent mobility mu,

    dx = -mu grad V dt + sqrt(2 mu kT) dW,

whose backward-Kolmogorov (generator) equation is

    L q = mu (kT Laplacian q - grad V . grad q) = 0   outside A u B,
    q = 0 on A,   q = 1 on B.

`mu` (= 1/(m gamma) for `AnalyticBackend(integrator="overdamped")`) only sets
the time scale; it cancels in L q = 0, so neither solver takes it. Only
V/kT matters.

committor_1d(potential, kT, a, b, xs) -> np.ndarray
    Exact 1D solution  q(x) = int_a^x e^{V/kT} dy / int_a^b e^{V/kT} dy,
    evaluated with adaptive Gauss-Kronrod quadrature (scipy.integrate.quad)
    on the segments between consecutive sorted evaluation points, then
    cumulatively summed (so q is monotone by construction and q(a)=0,
    q(b)=1 exactly). `potential` is anything with `energy_grad(x)` taking a
    shape-(1,) array (e.g. `DoubleWell1D`). Points x <= a give 0, x >= b
    give 1 (A = {x <= a}, B = {x >= b}).

committor_2d(potential, kT, A_mask_fn, B_mask_fn, bounds, n, *,
             energy_cutoff=50.0) -> Committor2D
    Finite-volume solution of L q = 0 on a vertex-centred rectangular grid
    covering `bounds = ((xmin, xmax), (ymin, ymax))` with `n` intervals per
    axis (an int, or an `(nx, ny)` pair); nodes are at
    `xmin + i*hx`, i = 0..nx (so `bounds` are themselves nodes).

    Discretisation: the conservative form  div(e^{-V/kT} grad q) = 0  on the
    node control volumes, with the flux across each dual face between
    neighbouring nodes i, j weighted by the geometric mean of the Boltzmann
    factors, sqrt(pi_i pi_j) = exp(-(V_i + V_j) / (2 kT)) (the "square-root
    approximation" / SQRA scheme; symmetric, detailed-balanced w.r.t.
    e^{-V/kT}, reproduces linear q for flat V exactly, second order in h for
    smooth V -- verified in tests/test_committor_ref.py). Face lengths are
    halved on the domain edge (half control volumes), and there is no flux
    through the outer boundary: **the outer boundary is reflecting**.

    `A_mask_fn(p)` / `B_mask_fn(p)` take an array of points `p` of shape
    (..., 2) (p[..., 0] = x, p[..., 1] = y) and return a boolean array of
    shape `p.shape[:-1]`; they are evaluated on the grid nodes. A node is a
    Dirichlet node (q = 0 / q = 1) iff its mask is true; A and B must not
    overlap and must both be non-empty (ValueError otherwise). Region
    boundaries are resolved to the node staircase, i.e. with O(h) accuracy
    unless they are grid-aligned.

    `potential.energy_grad` takes a shape-(2,) array (DoubleWell2D,
    MullerBrown, ChannelDoubleWell2D, ...); it is evaluated once per node.
    Non-Dirichlet nodes with (V - V_min)/kT > energy_cutoff, or non-finite
    V, are *excluded* (treated as an impenetrable wall; their q is NaN).
    Non-Dirichlet nodes whose connected component (through non-excluded
    nodes) touches neither A nor B have an undefined committor: q is NaN and
    they are flagged in `disconnected`. Nothing is silently filled.

    The returned `Committor2D` holds the grid (`x`, `y` node coordinates,
    `q` of shape (nx+1, ny+1) indexed [ix, iy], i.e. `indexing="ij"`), the
    boolean node masks `A`, `B`, `excluded`, `disconnected`, the node
    energies `V`, and is callable: `res(points)` bilinearly interpolates q
    at points of shape (..., 2) (points must lie inside `bounds`; NaN
    propagates from excluded/disconnected neighbours).

    Grid convergence: call twice with n and 2n and compare `res(points)` at
    the points of interest (`grid_convergence` does exactly that).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------------
# 1D
# ----------------------------------------------------------------------------


def committor_1d(potential, kT: float, a: float, b: float, xs) -> np.ndarray:
    """Exact 1D overdamped committor q(x) for A = {x <= a}, B = {x >= b}.

    See the module docstring. Returns an array with the shape of `xs`.
    """
    if not (math.isfinite(kT) and kT > 0):
        raise ValueError(f"kT must be finite and > 0, got {kT!r}")
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError(f"need finite a < b, got a={a!r}, b={b!r}")
    xs = np.asarray(xs, dtype=float)
    beta = 1.0 / kT

    def V(y: float) -> float:
        return float(potential.energy_grad(np.array([y], dtype=float))[0])

    # Shift by the maximum of V on [a, b] (dense scan) so e^{beta(V - Vmax)}
    # <= ~1 and never overflows; the shift cancels in the ratio.
    scan = np.linspace(a, b, 4001)
    v_max = max(V(y) for y in scan)

    def f(y: float) -> float:
        return math.exp(beta * (V(y) - v_max))

    inner = np.clip(xs.ravel(), a, b)
    breaks = np.unique(np.concatenate([[a, b], inner]))
    seg = np.array(
        [
            quad(f, lo, hi, epsabs=0.0, epsrel=1e-12, limit=500)[0]
            for lo, hi in zip(breaks[:-1], breaks[1:])
        ]
    )
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    q_breaks = cum / cum[-1]
    q_breaks[0], q_breaks[-1] = 0.0, 1.0
    q = q_breaks[np.searchsorted(breaks, inner)]
    return q.reshape(xs.shape)


# ----------------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class Committor2D:
    """Grid solution of the 2D committor problem (see module docstring)."""

    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    A: np.ndarray
    B: np.ndarray
    excluded: np.ndarray
    disconnected: np.ndarray
    V: np.ndarray
    kT: float

    def __post_init__(self) -> None:
        self._interp = RegularGridInterpolator(
            (self.x, self.y), self.q, method="linear", bounds_error=True
        )

    def __call__(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return self._interp(pts.reshape(-1, 2)).reshape(pts.shape[:-1])


def _grid_sizes(n) -> tuple[int, int]:
    if isinstance(n, (int, np.integer)):
        nx = ny = int(n)
    else:
        nx, ny = (int(v) for v in n)
    if nx < 2 or ny < 2:
        raise ValueError(f"need at least 2 intervals per axis, got n={n!r}")
    return nx, ny


def committor_2d(
    potential,
    kT: float,
    A_mask_fn: Callable[[np.ndarray], np.ndarray],
    B_mask_fn: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Sequence[float]],
    n,
    *,
    energy_cutoff: float = 50.0,
) -> Committor2D:
    """Finite-volume backward-Kolmogorov committor on a 2D grid.

    See the module docstring for the scheme, the mask convention, the
    reflecting outer boundary, and the exclusion rules.
    """
    if not (math.isfinite(kT) and kT > 0):
        raise ValueError(f"kT must be finite and > 0, got {kT!r}")
    (xmin, xmax), (ymin, ymax) = bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"degenerate bounds {bounds!r}")
    nx, ny = _grid_sizes(n)
    beta = 1.0 / kT
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    hx = (xmax - xmin) / nx
    hy = (ymax - ymin) / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X, Y], axis=-1)
    shape = X.shape

    A = np.asarray(A_mask_fn(P), dtype=bool)
    B = np.asarray(B_mask_fn(P), dtype=bool)
    if A.shape != shape or B.shape != shape:
        raise ValueError(
            f"mask functions must return shape {shape}, got {A.shape} / {B.shape}"
        )
    if np.any(A & B):
        raise ValueError("A and B masks overlap")
    if not A.any() or not B.any():
        raise ValueError("A and B must each contain at least one grid node")

    V = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            V[i, j] = potential.energy_grad(P[i, j])[0]
    finite = np.isfinite(V)
    dirichlet = A | B
    if not finite[~dirichlet].any():
        raise ValueError("no finite-energy interior nodes")
    v_min = float(np.min(V[finite & ~dirichlet])) if (finite & ~dirichlet).any() else 0.0
    excluded = ~dirichlet & (~finite | (beta * (V - v_min) > energy_cutoff))

    # Boltzmann factor sqrt-weights, shifted by v_min (cancels in L q = 0).
    with np.errstate(over="ignore", invalid="ignore"):
        half = np.where(finite, np.exp(-0.5 * beta * (V - v_min)), 0.0)

    idx = np.arange(X.size).reshape(shape)
    rows, cols, wts = [], [], []

    # x-links (i,j)-(i+1,j): face length hy (halved on y edges), distance hx.
    face_y = np.full(ny + 1, hy)
    face_y[0] = face_y[-1] = 0.5 * hy
    w_x = half[:-1, :] * half[1:, :] * (face_y[None, :] / hx)
    ok_x = ~excluded[:-1, :] & ~excluded[1:, :] & (w_x > 0)
    rows.append(idx[:-1, :][ok_x])
    cols.append(idx[1:, :][ok_x])
    wts.append(w_x[ok_x])
    # y-links (i,j)-(i,j+1): face length hx (halved on x edges), distance hy.
    face_x = np.full(nx + 1, hx)
    face_x[0] = face_x[-1] = 0.5 * hx
    w_y = half[:, :-1] * half[:, 1:] * (face_x[:, None] / hy)
    ok_y = ~excluded[:, :-1] & ~excluded[:, 1:] & (w_y > 0)
    rows.append(idx[:, :-1][ok_y])
    cols.append(idx[:, 1:][ok_y])
    wts.append(w_y[ok_y])

    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.concatenate(wts)
    N = X.size
    W = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))),
                      shape=(N, N)).tocsr()

    # Connected components through non-excluded nodes; a free node whose
    # component contains no Dirichlet node has an undefined committor.
    _, labels = connected_components(W, directed=False)
    flat_dir = dirichlet.ravel()
    good_labels = np.unique(labels[flat_dir])
    reachable = np.isin(labels, good_labels)
    free = (~dirichlet & ~excluded).ravel()
    disconnected = free & ~reachable
    unknown = np.flatnonzero(free & reachable)

    q = np.full(N, np.nan)
    q[A.ravel()] = 0.0
    q[B.ravel()] = 1.0

    if unknown.size:
        W_uu = W[unknown][:, unknown]
        deg = np.asarray(W[unknown].sum(axis=1)).ravel()
        rhs = np.asarray(W[unknown][:, np.flatnonzero(B.ravel())].sum(axis=1)).ravel()
        M = sp.diags(deg) - W_uu
        # Symmetric diagonal scaling (weights span many orders of magnitude).
        s = 1.0 / np.sqrt(deg)
        S = sp.diags(s)
        z = spsolve((S @ M @ S).tocsc(), s * rhs)
        q[unknown] = np.clip(s * z, 0.0, 1.0)

    return Committor2D(
        x=xs,
        y=ys,
        q=q.reshape(shape),
        A=A,
        B=B,
        excluded=excluded,
        disconnected=disconnected.reshape(shape),
        V=V,
        kT=float(kT),
    )


def grid_convergence(
    solve: Callable[[int], Committor2D], n: int, points
) -> tuple[Committor2D, Committor2D, float]:
    """Solve at `n` and `2n` intervals (via `solve(n)`) and return both
    solutions plus max |q_n - q_2n| over `points` (shape (..., 2))."""
    coarse = solve(n)
    fine = solve(2 * n)
    diff = float(np.max(np.abs(coarse(points) - fine(points))))
    return coarse, fine, diff
