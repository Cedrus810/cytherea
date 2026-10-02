"""Mueller-Brown committor toy (A0 acceptance 9.3).

V = 0.1 * V_MB (`MullerBrown(scale=0.1)`) with kT = 1, i.e. the standard
Mueller-Brown surface at kT = 10 in its native energy units. Overdamped
dynamics, D = kT/(m gamma) = 1, reduced units.

Stationary points (native units -> kT units after scaling by 0.1):

    minimum A (-0.558, 1.442)  V_MB = -146.70  (deepest)
    minimum B ( 0.623, 0.028)  V_MB = -108.17
    minimum C (-0.050, 0.467)  V_MB =  -80.77  (intermediate)
    saddle S1 (-0.822, 0.624)  V_MB =  -40.66  (A <-> C, 10.6 kT above A)
    saddle S2 ( 0.212, 0.293)  V_MB =  -72.25  (C <-> B, 0.85 kT above C)

A = disk of radius 0.15 around minimum A, B = disk of radius 0.15 around
minimum B, with persistence `tau_persist` (10 observations). Since C drains
to B over only 0.85 kT, the committor's transition region lies around S1.

Time step: the largest Hessian eigenvalue near the basins is ~407 (minimum A,
scaled units), so dt = 5e-5 gives mu lambda dt = 0.02; observations every
10 steps (dt_obs = 5e-4).

Reference: `tests/reference/committor_ref.committor_2d` on
`REFERENCE_BOUNDS` with the same `Disk` predicates (`Disk.mask`).
"""

from __future__ import annotations

from collections.abc import Sequence

from cytherea.backends.analytic import AnalyticBackend, MullerBrown
from cytherea.engine.shot import ObsSpec
from cytherea.observe.events import AbsorbingAB, Region

from .committor_shots import Disk, make_shot_fn as _make_shot_fn, obs_x, obs_y

CONFIG: dict = {
    "units": "reduced",
    "scale": 0.1,
    "kT": 1.0,
    "gamma": 1.0,
    "mass": 1.0,
    "dt": 5.0e-5,
    "dt_obs": 5.0e-4,
    "A_center": (-0.5582, 1.4417),
    "B_center": (0.6235, 0.0280),
    "radius": 0.15,
    "tau_persist": 5.0e-3,
    "t_max": 50.0,
    "n_points": 30,
    "n_shots": 1600,  # A0 9.2/9.3: 400 gave noise RMSE 0.0204, too close to the 0.03 gate (report)
}

REFERENCE_BOUNDS = ((-1.7, 1.3), (-0.5, 2.2))


def make_potential(cfg: dict = CONFIG) -> MullerBrown:
    return MullerBrown(scale=cfg["scale"])


def region_A(cfg: dict = CONFIG) -> Disk:
    return Disk(*cfg["A_center"], cfg["radius"])


def region_B(cfg: dict = CONFIG) -> Disk:
    return Disk(*cfg["B_center"], cfg["radius"])


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
        A=Region("A", region_A(cfg)),
        B=Region("B", region_B(cfg)),
        tau_persist=cfg["tau_persist"],
        t_max=cfg["t_max"],
    )


def make_obs(cfg: dict = CONFIG) -> ObsSpec:
    return ObsSpec(fns={"x": obs_x, "y": obs_y}, dt_obs=cfg["dt_obs"], store_stride=1)


def make_shot_fn(points: Sequence[Sequence[float]], cfg: dict = CONFIG):
    return _make_shot_fn(points, make_backend(cfg), make_stop(cfg), make_obs(cfg),
                         cfg["kT"], cfg["mass"], "mueller_brown")
