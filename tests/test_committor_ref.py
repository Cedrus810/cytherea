"""Unit tests for the committor reference solvers in tests/reference/committor_ref.py
(Task 9, controller ruling R4).

The reference solvers are what the A0 acceptance shots are compared against,
so they are checked here against closed forms first:

1D (`committor_1d`, exact quadrature of q = int_a^x e^{bV} / int_a^b e^{bV}):
- flat potential -> linear q;
- linear potential V = F x -> q = (e^{bF(x-a)} - 1) / (e^{bF(b-a)} - 1);
- symmetric double well -> q(0) = 1/2 and q(-x) = 1 - q(x);
- clipping outside [a, b].

2D (`committor_2d`, finite-volume backward-Kolmogorov solve):
- flat potential, A/B = slabs aligned with grid nodes -> q exactly linear
  (the scheme preserves linear functions; reflecting walls parallel to x);
- separable DoubleWell2D, A/B = x-slabs -> q(x, y) = q_1d(x), with second-
  order grid convergence (error ratio ~ 4 when h halves);
- flat potential, annulus A: r <= r1, B: r >= r2 (not grid aligned) ->
  q = ln(r/r1)/ln(r2/r1) within the O(h) staircase error, and converging;
- maximum principle 0 <= q <= 1, A/B values exactly 0/1;
- inaccessible (energy-cutoff) cells and components that touch neither A nor B
  are reported as NaN, never silently filled.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from reference.committor_ref import Committor2D, committor_1d, committor_2d
from cytherea.backends.analytic import DoubleWell1D, DoubleWell2D, FreeParticle


@dataclasses.dataclass(frozen=True)
class _Linear1D:
    F: float

    def energy_grad(self, x):
        x = np.asarray(x, dtype=float)
        return float(self.F * x[0]), np.array([self.F])


@dataclasses.dataclass(frozen=True)
class _Box2D:
    """V = 0 in the open box |x|,|y| < 1 and in a separate pocket
    |x| < 0.3, 1.2 < y < 1.45; `height` everywhere else (an energy wall
    that also separates the pocket from the box).
    """

    height: float

    def energy_grad(self, x):
        x = np.asarray(x, dtype=float)
        box = abs(x[0]) < 1.0 and abs(x[1]) < 1.0
        pocket = abs(x[0]) < 0.3 and 1.2 < x[1] < 1.45
        return (0.0 if (box or pocket) else self.height), np.zeros(2)


# ----------------------------------------------------------------------------
# 1D
# ----------------------------------------------------------------------------


def test_1d_flat_potential_is_linear():
    xs = np.linspace(-0.3, 0.7, 11)
    q = committor_1d(FreeParticle(1), kT=1.0, a=-0.3, b=0.7, xs=xs)
    np.testing.assert_allclose(q, (xs + 0.3) / 1.0, atol=1e-12)


@pytest.mark.parametrize("F,kT", [(3.0, 1.0), (-2.0, 0.5), (10.0, 2.0)])
def test_1d_linear_potential_closed_form(F, kT):
    a, b = -1.0, 2.0
    xs = np.linspace(a, b, 17)
    q = committor_1d(_Linear1D(F), kT=kT, a=a, b=b, xs=xs)
    beta = 1.0 / kT
    exact = np.expm1(beta * F * (xs - a)) / np.expm1(beta * F * (b - a))
    np.testing.assert_allclose(q, exact, rtol=1e-9, atol=1e-12)


def test_1d_symmetric_double_well():
    pot = DoubleWell1D(barrier=5.0, x0=1.0)
    xs = np.linspace(-0.7, 0.7, 15)
    q = committor_1d(pot, kT=1.0, a=-0.8, b=0.8, xs=xs)
    assert q[7] == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(q, 1.0 - q[::-1], atol=1e-12)
    assert np.all(np.diff(q) > 0)


def test_1d_clips_outside_interval_and_hits_endpoints():
    pot = DoubleWell1D(barrier=5.0)
    q = committor_1d(pot, kT=1.0, a=-0.8, b=0.8, xs=[-2.0, -0.8, 0.8, 3.0])
    np.testing.assert_allclose(q, [0.0, 0.0, 1.0, 1.0], atol=1e-14)


def test_1d_rejects_bad_interval():
    with pytest.raises(ValueError):
        committor_1d(FreeParticle(1), kT=1.0, a=1.0, b=1.0, xs=[1.0])
    with pytest.raises(ValueError):
        committor_1d(FreeParticle(1), kT=0.0, a=0.0, b=1.0, xs=[0.5])


# ----------------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------------

_TOL = 1e-12


def test_2d_flat_potential_slabs_is_exactly_linear():
    res = committor_2d(
        FreeParticle(2),
        kT=1.0,
        A_mask_fn=lambda p: p[..., 0] <= 0.2 + _TOL,
        B_mask_fn=lambda p: p[..., 0] >= 0.8 - _TOL,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        n=20,
    )
    assert isinstance(res, Committor2D)
    X, _Y = np.meshgrid(res.x, res.y, indexing="ij")
    exact = np.clip((X - 0.2) / 0.6, 0.0, 1.0)
    np.testing.assert_allclose(res.q, exact, atol=1e-10)
    # interpolator reproduces linear functions exactly
    pts = np.array([[0.33, 0.1], [0.5, 0.97], [0.71, 0.5]])
    np.testing.assert_allclose(res(pts), (pts[:, 0] - 0.2) / 0.6, atol=1e-10)


def _sep_error(n):
    pot2 = DoubleWell2D(barrier=2.0, ky=4.0)
    res = committor_2d(
        pot2,
        kT=1.0,
        A_mask_fn=lambda p: p[..., 0] <= -0.8 + _TOL,
        B_mask_fn=lambda p: p[..., 0] >= 0.8 - _TOL,
        bounds=((-1.2, 1.2), (-1.5, 1.5)),
        n=n,
    )
    q1 = committor_1d(DoubleWell1D(barrier=2.0), kT=1.0, a=-0.8, b=0.8, xs=res.x)
    return float(np.max(np.abs(res.q - q1[:, None]))), res


def test_2d_separable_matches_1d_with_second_order_convergence():
    e1, res1 = _sep_error(60)
    e2, res2 = _sep_error(120)
    assert e2 < 2e-4, e2
    assert e1 / e2 > 3.0, (e1, e2)  # O(h^2): ratio ~4
    # q independent of y (separable, reflecting in y)
    assert np.max(np.ptp(res2.q, axis=1)) < 1e-10


def _annulus_error(n):
    r1, r2 = 0.3, 1.0
    res = committor_2d(
        FreeParticle(2),
        kT=1.0,
        A_mask_fn=lambda p: np.hypot(p[..., 0], p[..., 1]) <= r1,
        B_mask_fn=lambda p: np.hypot(p[..., 0], p[..., 1]) >= r2,
        bounds=((-1.2, 1.2), (-1.2, 1.2)),
        n=n,
    )
    rr = np.linspace(0.4, 0.9, 6)
    ang = np.linspace(0.1, 2 * np.pi, 7)[:-1]
    pts = np.array([[r * np.cos(t), r * np.sin(t)] for r in rr for t in ang])
    exact = np.log(np.hypot(pts[:, 0], pts[:, 1]) / r1) / math.log(r2 / r1)
    return float(np.max(np.abs(res(pts) - exact)))


def test_2d_annulus_log_profile_converges():
    e1 = _annulus_error(96)
    e2 = _annulus_error(192)
    # Region boundaries are resolved to the node staircase: first order in h
    # (measured: 0.033 -> 0.015 -> 0.0067 at n = 96 -> 192 -> 384).
    assert e2 < 0.02, e2
    assert e2 < 0.6 * e1, (e1, e2)


def test_2d_maximum_principle_and_dirichlet_values():
    res = committor_2d(
        DoubleWell2D(barrier=3.0, ky=2.0),
        kT=0.7,
        A_mask_fn=lambda p: np.hypot(p[..., 0] + 1, p[..., 1]) <= 0.31,
        B_mask_fn=lambda p: np.hypot(p[..., 0] - 1, p[..., 1]) <= 0.31,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        n=80,
    )
    assert np.all(res.q[res.A] == 0.0)
    assert np.all(res.q[res.B] == 1.0)
    assert np.nanmin(res.q) >= 0.0 and np.nanmax(res.q) <= 1.0
    assert not np.any(np.isnan(res.q))
    # symmetry x -> -x maps q -> 1 - q
    np.testing.assert_allclose(res.q, 1.0 - res.q[::-1, :], atol=1e-9)


def test_2d_inaccessible_and_disconnected_cells_are_nan():
    # Energy wall of 1000 kT: wall cells exceed the energy cutoff and are
    # excluded (NaN); the pocket is accessible but touches neither A nor B,
    # so its committor is undefined -> NaN and flagged `disconnected`.
    res = committor_2d(
        _Box2D(height=1000.0),
        kT=1.0,
        A_mask_fn=lambda p: (p[..., 0] <= -0.8 + _TOL) & (p[..., 0] > -1.0) & (np.abs(p[..., 1]) < 1.0),
        B_mask_fn=lambda p: (p[..., 0] >= 0.8 - _TOL) & (p[..., 0] < 1.0) & (np.abs(p[..., 1]) < 1.0),
        bounds=((-1.5, 1.5), (-1.5, 1.5)),
        n=60,
    )
    X, Y = np.meshgrid(res.x, res.y, indexing="ij")
    box = (np.abs(X) < 1.0) & (np.abs(Y) < 1.0)
    pocket = (np.abs(X) < 0.3) & (Y > 1.2) & (Y < 1.45)
    wall = ~box & ~pocket
    assert pocket.sum() > 0 and wall.sum() > 0
    assert np.all(res.excluded[wall]) and not np.any(res.excluded[~wall])
    assert np.all(np.isnan(res.q[wall]))
    assert np.all(res.disconnected[pocket]) and np.all(np.isnan(res.q[pocket]))
    assert not np.any(np.isnan(res.q[box]))
    # flat inside the box with slab A/B: linear in x
    inner = box & ~res.A & ~res.B
    np.testing.assert_allclose(res.q[inner], (X[inner] + 0.8) / 1.6, atol=1e-9)


def test_2d_requires_nonempty_A_and_B():
    with pytest.raises(ValueError):
        committor_2d(
            FreeParticle(2),
            kT=1.0,
            A_mask_fn=lambda p: p[..., 0] < -10,
            B_mask_fn=lambda p: p[..., 0] > 0.5,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            n=10,
        )
