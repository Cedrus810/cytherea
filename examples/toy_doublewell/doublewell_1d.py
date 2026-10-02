"""1D double-well committor toy (A0 acceptance 9.2).

V(x) = barrier * (x^2 - 1)^2 with barrier = 5 kT (`DoubleWell1D`),
overdamped dynamics with D = kT/(m gamma) = 1, reduced units.

A = {x <= -0.8}, B = {x >= 0.8} (V(+-0.8) = 0.648 kT: the edge of each
basin), both with a short persistence `tau_persist` (10 observations).
Shots start from fixed points in the transition region and stop at the first
persistent entry into A or B (`AbsorbingAB`), or time out at `t_max`.

Reference: `tests/reference/committor_ref.committor_1d` with a = -0.8,
b = 0.8, i.e. q(x) = int_{-0.8}^x e^{V/kT} / int_{-0.8}^{0.8} e^{V/kT}.
The persistence requirement moves the effective absorbing boundary into the
basin by ~sqrt(2 D tau_persist) = 0.1; since dq/dx at x = +-0.8 is only
e^{V(0.8)/kT} / Z = 0.023 per unit length (Z = int e^{V/kT} = 83), the
induced committor shift is ~0.002 (quantified in the acceptance test/report).

Time step: the stiffest curvature is V''(+-1) = 8*barrier = 40, so
dt = 5e-4 gives mu V'' dt = 0.02; dt_obs = dt.
"""

from __future__ import annotations

from collections.abc import Sequence

from cytherea.backends.analytic import AnalyticBackend, DoubleWell1D
from cytherea.engine.shot import ObsSpec
from cytherea.observe.events import AbsorbingAB, Region

from .committor_shots import CoordAtLeast, CoordAtMost, make_shot_fn as _make_shot_fn, obs_x

CONFIG: dict = {
    "units": "reduced",
    "barrier": 5.0,
    "x0": 1.0,
    "kT": 1.0,
    "gamma": 1.0,
    "mass": 1.0,
    "dt": 5.0e-4,
    "dt_obs": 5.0e-4,
    "A_max": -0.8,
    "B_min": 0.8,
    "tau_persist": 5.0e-3,
    "t_max": 50.0,
    "n_points": 20,
    "n_shots": 1600,  # A0 9.2/9.3: 400 gave noise RMSE 0.0204, too close to the 0.03 gate (report)
}


def make_potential(cfg: dict = CONFIG) -> DoubleWell1D:
    return DoubleWell1D(barrier=cfg["barrier"], x0=cfg["x0"])


def make_backend(cfg: dict = CONFIG) -> AnalyticBackend:
    return AnalyticBackend(
        make_potential(cfg),
        integrator="overdamped",
        dt=cfg["dt"],
        kT=cfg["kT"],
        gamma=cfg["gamma"],
        mass=cfg["mass"],
    )


def make_stop(cfg: dict = CONFIG) -> AbsorbingAB:
    """A fresh stop rule (also used by the offline replay)."""
    return AbsorbingAB(
        A=Region("A", CoordAtMost("x", cfg["A_max"])),
        B=Region("B", CoordAtLeast("x", cfg["B_min"])),
        tau_persist=cfg["tau_persist"],
        t_max=cfg["t_max"],
    )


def make_obs(cfg: dict = CONFIG) -> ObsSpec:
    return ObsSpec(fns={"x": obs_x}, dt_obs=cfg["dt_obs"], store_stride=1)


def make_shot_fn(points: Sequence[float], cfg: dict = CONFIG):
    return _make_shot_fn([[p] for p in points], make_backend(cfg), make_stop(cfg), make_obs(cfg),
                         cfg["kT"], cfg["mass"], "doublewell_1d")
