"""Exact references for 1D overdamped Langevin dynamics (test oracles).

For dx = -D*beta*V'(x) dt + sqrt(2D) dW (i.e. D = kT/(m*gamma), beta = 1/kT):

- Mean first-passage time from x0 to b > x0, with reflection at the lower
  end (x_lo -> -inf in the limit; a finite x_lo is an explicit reflecting
  wall)::

      tau(x0 -> b) = (1/D) * int_{x0}^{b} dy e^{beta V(y)} int_{x_lo}^{y} dz e^{-beta V(z)}

  With point injection at x0 and absorption at b, the steady-state
  (recycled) flux into b is exactly 1/tau (Hill relation).

- Committor to b against a (a < x < b), both absorbing::

      q(x) = int_a^x e^{beta V} / int_a^b e^{beta V}

Both are evaluated by (cumulative) trapezoid quadrature on a uniform grid;
`n` points give O(h^2) error, far below any test tolerance at the default.
`V` must be a vectorized callable on a 1D numpy array.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _cumtrapz(f: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(f)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * h * (f[1:] + f[:-1]))
    return out


def mfpt_1d(
    V: Callable[[np.ndarray], np.ndarray],
    x0: float,
    b: float,
    D: float,
    beta: float,
    x_lo: float,
    n: int = 400_001,
) -> float:
    """tau(x0 -> b) with a reflecting boundary at `x_lo` (use x_lo far enough
    below the potential's confining wall to stand in for -inf).
    """
    if not (x_lo < x0 < b):
        raise ValueError("need x_lo < x0 < b")
    y = np.linspace(x_lo, b, n)
    vy = beta * V(y)
    # Shift exponents to avoid overflow; the product e^{+V} * e^{-V} is
    # shift-invariant, so the same constant cancels exactly.
    c = vy.min()
    inner = _cumtrapz(np.exp(-(vy - c)), y[1] - y[0])
    y2 = np.linspace(x0, b, n)
    outer = np.exp(beta * V(y2) - c) * np.interp(y2, y, inner)
    return float(_cumtrapz(outer, y2[1] - y2[0])[-1]) / D


def committor_1d(
    V: Callable[[np.ndarray], np.ndarray],
    x: float,
    a: float,
    b: float,
    beta: float,
    n: int = 400_001,
) -> float:
    """q(x) = P(hit b before a | start at x), a < x < b."""
    if not (a <= x <= b):
        raise ValueError("need a <= x <= b")
    y = np.linspace(a, b, n)
    h = y[1] - y[0]
    vy = beta * V(y)
    f = np.exp(vy - vy.max())
    cum = _cumtrapz(f, h)
    return float(np.interp(x, y, cum) / cum[-1])
