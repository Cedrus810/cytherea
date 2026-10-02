"""Shared pieces for committor shooting from a set of fixed starting points.

All starting points are the frames of *one* `EnsembleFramePool` (frame_id =
point index), so a whole committor campaign (all points x all shots) is a
single `run_batch` call over keys `ShotKey(seed, frame_id=point, shot_id=k,
stage)`: with ``frame_id >= 0`` the sampler uses exactly that frame
(contract K3), and every shot of the campaign shares one protocol hash
(ruling R39), which `run_batch` checks on resume. `make_shot_fn` returns
``functools.partial(run_shot, ...)`` (no store: `run_batch` is the only
writer, ruling R33), picklable for `n_workers > 1`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence

import numpy as np

from cytherea.backends.analytic import AnalyticBackend
from cytherea.backends.base import MDState
from cytherea.engine.shot import ObsSpec, run_shot
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import EnsembleFrameSampler
from cytherea.keys import ShotKey
from cytherea.observe.events import StopRule


def obs_x(state: MDState) -> float:
    return float(state.x[0])


def obs_y(state: MDState) -> float:
    return float(state.x[1])


@dataclasses.dataclass(frozen=True)
class CoordAtMost:
    """Picklable Region predicate: obs[name] <= cutoff."""

    name: str
    cutoff: float

    def __call__(self, obs: dict) -> bool:
        return obs[self.name] <= self.cutoff

    @property
    def spec(self) -> str:  # enters the protocol hash (ruling R39)
        return f"{self.name} <= {self.cutoff!r}"


@dataclasses.dataclass(frozen=True)
class CoordAtLeast:
    """Picklable Region predicate: obs[name] >= cutoff."""

    name: str
    cutoff: float

    def __call__(self, obs: dict) -> bool:
        return obs[self.name] >= self.cutoff

    @property
    def spec(self) -> str:
        return f"{self.name} >= {self.cutoff!r}"


@dataclasses.dataclass(frozen=True)
class Disk:
    """Picklable Region predicate: (obs["x"]-cx)^2 + (obs["y"]-cy)^2 <= r^2.

    `mask(p)` evaluates the same predicate on an array of points (..., 2),
    for the grid reference solver -- one definition for both.
    """

    cx: float
    cy: float
    r: float

    def __call__(self, obs: dict) -> bool:
        return (obs["x"] - self.cx) ** 2 + (obs["y"] - self.cy) ** 2 <= self.r**2

    def mask(self, p: np.ndarray) -> np.ndarray:
        return (p[..., 0] - self.cx) ** 2 + (p[..., 1] - self.cy) ** 2 <= self.r**2

    @property
    def spec(self) -> str:
        return f"(x - {self.cx!r})^2 + (y - {self.cy!r})^2 <= {self.r!r}^2"


def point_sampler(
    points: Sequence[Sequence[float]],
    backend: AnalyticBackend,
    kT: float,
    mass: float,
    topology_ref: str,
) -> EnsembleFrameSampler:
    """One sampler whose pool holds every starting point (frame_id = point
    index, equal weights).

    No energy window / pair-distance checks apply to a single reduced-unit
    particle; the IC gate still rejects non-finite coordinates/velocities.
    """
    frames = [
        EnsembleFrame(
            coordinates=np.asarray(p, dtype=float).reshape(-1),
            box=None,
            topology_ref=topology_ref,
            temperature=kT,
            weight=1.0,
            source_id=f"point{i}",
            frame_id=i,
            time=0.0,
        )
        for i, p in enumerate(points)
    ]
    return EnsembleFrameSampler(
        pool=EnsembleFramePool(frames),
        masses=np.array([mass]),
        kT=kT,
        backend=backend,
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=False,
    )


def make_shot_fn(
    points: Sequence[Sequence[float]],
    backend: AnalyticBackend,
    stop: StopRule,
    obs: ObsSpec,
    kT: float,
    mass: float,
    topology_ref: str,
):
    """``partial(run_shot, sampler=point_sampler(...), ...)`` for `run_batch`."""
    return functools.partial(
        run_shot,
        sampler=point_sampler(points, backend, kT, mass, topology_ref),
        backend=backend,
        stop=stop,
        obs=obs,
        physics_cfg=None,
    )


def committor_keys(
    point_ids: Sequence[int], n_shots: int, global_seed: int, stage: str
) -> list[ShotKey]:
    return [
        ShotKey(global_seed=global_seed, frame_id=int(i), shot_id=k, stage=stage)
        for i in point_ids
        for k in range(n_shots)
    ]
