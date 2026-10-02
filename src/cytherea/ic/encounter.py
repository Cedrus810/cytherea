"""Encounter initial conditions on the NAM b sphere (Task 16, design 4.3).

`EncounterSampler.sample(key)` builds an A + B starting configuration:

1. a frame of A from ``pool_A`` in state ``label[0]`` and a frame of B from
   ``pool_B`` in state ``label[1]``, each drawn by weight
   (``derive_rng(key, "ic/encounter/frame_A")`` / ``.../frame_B``);
2. A is translated so that its centre of mass is at the origin; B is rotated
   about its centre of mass by a Haar-uniform rotation (a normalised 4D
   Gaussian quaternion, ``.../rotation``) and its centre of mass is put at
   ``b * n``, with ``n`` uniform on the unit sphere (a normalised 3D Gaussian,
   ``.../direction``). The relative pose is therefore uniform on SO(3) x S^2
   at COM distance exactly ``b`` (to rounding);
3. the A-B contact check: the smallest distance between an atom of A and
   one of B must be at least ``min_pair_dist``, else the IC is rejected with
   the coordinate-level reason ``"encounter_clash"`` (`ICRejectedError`).
   Never redrawn: a redraw would condition the pose distribution on "no
   clash" and so bias it. Choose b beyond contact distance
   (`clash_fraction` estimates how often a b clashes);
4. velocities, constraints and the IC gate exactly as `EnsembleFrameSampler`
   (Maxwell-Boltzmann at ``kT``, keyed velocity redraws, energy/force
   checks, optional ``energy_window``) on the combined configuration.

Atoms are ordered A then B; ``masses`` covers both. Both pools must be
non-periodic (implicit solvent: an encounter in a periodic box is Phase C
work). Keys must have ``frame_id == -1`` (both frames are weighted draws, so
every shot carries weight 1; contract K3/K11). Same key, same IC, bit for bit.

``InitialState.meta`` has the K2 keys (``frame_id`` = -1, ``frame_time`` =
None, ``frame_weight`` = 1.0, ``source_id`` = "A:<id>|B:<id>", ``topology_ref``,
``state`` = the label, ``n_redraws``) plus ``frame_id_A``, ``frame_id_B``,
``b``, ``quaternion`` (w, x, y, z with w >= 0), ``direction`` and
``com_distance``; ``InitialState.origin_label`` = ``label``, which
`run_shot` writes to ``ShotRecord.origin_label``.

Every frame of both pools must have the same temperature (the combined
frame has one); with ``boltzmann_constant`` it must also satisfy
kB * T = kT, as in `EnsembleFrameSampler`.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

import numpy as np
from scipy.spatial import cKDTree

from cytherea.backends.base import PotentialBackend
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import (
    CONSTRAINT_TOLERANCE,
    Constraints,
    EnsembleFrameSampler,
    ICRejectedError,
    InitialState,
    ValidityReport,
)
from cytherea.keys import ShotKey, derive_rng

_TEMPERATURE_RTOL = 1e-4  # as EnsembleFrameSampler's frame-temperature check


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform unit quaternion (w, x, y, z), sign fixed to w >= 0."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def random_direction(rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(size=3)
    return n / np.linalg.norm(n)


def _com(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (m[:, None] * x).sum(axis=0) / m.sum()


class EncounterSampler:
    def __init__(
        self,
        pool_A: EnsembleFramePool,
        pool_B: EnsembleFramePool,
        b: float,
        masses: np.ndarray,
        kT: float,
        backend: PotentialBackend,
        min_pair_dist: float,
        label: tuple[int, int],
        *,
        energy_window: tuple[float, float] | None = None,
        constraints: Constraints | None = None,
        max_redraws: int = 20,
        remove_com_momentum: bool = True,
        constraint_tolerance: float = CONSTRAINT_TOLERANCE,
        boltzmann_constant: float | None = None,
    ) -> None:
        self.pool_A, self.pool_B = pool_A, pool_B
        self.b = float(b)
        self.masses = np.asarray(masses, dtype=float)
        self.kT = float(kT)
        self.backend = backend
        self.min_pair_dist = float(min_pair_dist)
        self.label = tuple(int(v) for v in label)
        self.energy_window = energy_window
        self.constraints = constraints
        self.max_redraws = int(max_redraws)
        self.remove_com_momentum = bool(remove_com_momentum)
        self.constraint_tolerance = float(constraint_tolerance)
        self.boltzmann_constant = boltzmann_constant
        if not (math.isfinite(self.b) and self.b > 0):
            raise ValueError(f"b must be finite and > 0, got {b!r}")
        if not (math.isfinite(self.min_pair_dist) and self.min_pair_dist >= 0):
            raise ValueError(f"min_pair_dist must be finite and >= 0, got {min_pair_dist!r}")
        if len(self.label) != 2:
            raise ValueError(f"label must be (state of A, state of B), got {label!r}")
        fa, fb = pool_A.frames[0], pool_B.frames[0]
        self.n_A = int(np.asarray(fa.coordinates).shape[0])
        self.n_B = int(np.asarray(fb.coordinates).shape[0])
        for name, pool, n in (("pool_A", pool_A, self.n_A), ("pool_B", pool_B, self.n_B)):
            for f in pool.frames:
                c = np.asarray(f.coordinates)
                if c.ndim != 2 or c.shape != (n, 3):
                    raise ValueError(f"{name}: every frame needs coordinates of shape ({n}, 3), got {c.shape}")
                if f.box is not None:
                    raise ValueError(f"{name}: periodic frames are not supported (implicit-solvent encounters)")
        if self.masses.shape != (self.n_A + self.n_B,):
            raise ValueError(f"masses must have shape ({self.n_A + self.n_B},), got {self.masses.shape}")
        if not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise ValueError("masses must be finite and positive")
        if len({f.topology_ref for f in pool_A.frames}) != 1 or len({f.topology_ref for f in pool_B.frames}) != 1:
            raise ValueError("each pool must hold frames of one topology")
        # the combined frame carries one temperature: every frame of both pools must share it
        temps = [float(f.temperature) for f in (*pool_A.frames, *pool_B.frames)]
        t0 = temps[0]
        if not all(abs(t - t0) <= _TEMPERATURE_RTOL * abs(t0) for t in temps):
            raise ValueError(
                f"pool_A / pool_B frames come from different temperatures ({min(temps)} .. {max(temps)}): "
                "an encounter combines frames of one ensemble temperature"
            )
        if boltzmann_constant is not None and not (
                abs(float(boltzmann_constant) * t0 - self.kT) <= _TEMPERATURE_RTOL * self.kT):
            raise ValueError(f"boltzmann_constant * temperature = {float(boltzmann_constant) * t0} "
                             f"does not match kT = {self.kT}")
        self.temperature = t0
        self.topology_ref = f"{fa.topology_ref}+{fb.topology_ref}"

    # ------------------------------------------------------------- protocol

    def protocol_description(self) -> dict:
        def pool(p):
            state_of = p.state_of
            return {"n_frames": len(p), "frames_sha256": p.content_sha256(),
                    "states": None if state_of is None else [repr(state_of(f)) for f in p.frames]}

        import hashlib

        return {
            "pool_A": pool(self.pool_A), "pool_B": pool(self.pool_B), "b": self.b,
            "masses_sha256": hashlib.sha256(np.ascontiguousarray(self.masses).tobytes()).hexdigest(),
            "kT": self.kT, "min_pair_dist": self.min_pair_dist, "label": list(self.label),
            "energy_window": None if self.energy_window is None else [float(v) for v in self.energy_window],
            "constraints": None if self.constraints is None else type(self.constraints).__qualname__,
            "n_constraints": None if self.constraints is None else int(self.constraints.n_constraints),
            "max_redraws": self.max_redraws, "remove_com_momentum": self.remove_com_momentum,
            "constraint_tolerance": self.constraint_tolerance,
        }

    # ------------------------------------------------------------ placement

    def _draw(self, pool: EnsembleFramePool, state: Hashable, key: ShotKey, name: str) -> EnsembleFrame:
        st = state if pool.state_of is not None else None
        return pool.choose(derive_rng(key, f"ic/encounter/{name}"), state=st)

    def place(self, key: ShotKey) -> tuple[np.ndarray, dict]:
        """Combined coordinates (A then B) and the placement meta for ``key``."""
        if key.frame_id != -1:
            raise ValueError(f"encounter keys draw both frames by weight: frame_id must be -1, got {key.frame_id}")
        fa = self._draw(self.pool_A, self.label[0], key, "frame_A")
        fb = self._draw(self.pool_B, self.label[1], key, "frame_B")
        mA, mB = self.masses[: self.n_A], self.masses[self.n_A:]
        xa = np.asarray(fa.coordinates, dtype=float)
        xa = xa - _com(xa, mA)
        xb = np.asarray(fb.coordinates, dtype=float)
        xb = xb - _com(xb, mB)
        q = random_quaternion(derive_rng(key, "ic/encounter/rotation"))
        n = random_direction(derive_rng(key, "ic/encounter/direction"))
        xb = xb @ quaternion_matrix(q).T + self.b * n
        x = np.concatenate([xa, xb])
        meta = {
            "frame_id_A": int(fa.frame_id), "frame_id_B": int(fb.frame_id),
            "source_id": f"A:{fa.source_id}|B:{fb.source_id}", "b": self.b,
            "quaternion": q.tolist(), "direction": n.tolist(),
            "com_distance": float(np.linalg.norm(_com(xb, mB) - _com(xa, mA))),
        }
        return x, meta

    def min_ab_distance(self, x: np.ndarray) -> float:
        xa, xb = x[: self.n_A], x[self.n_A:]
        d, _ = cKDTree(xa).query(xb, k=1)
        return float(np.min(d))

    def clash_fraction(self, keys) -> float:
        """Fraction of ``keys`` whose placement fails the contact check."""
        keys = list(keys)
        return sum(self.min_ab_distance(self.place(k)[0]) < self.min_pair_dist for k in keys) / max(1, len(keys))

    # --------------------------------------------------------------- sample

    def sample(self, key: ShotKey) -> tuple[InitialState, ValidityReport]:
        x, meta = self.place(key)
        dmin = self.min_ab_distance(x)
        if not (dmin >= self.min_pair_dist):
            raise ICRejectedError(
                ["encounter_clash"], [{"attempt": 0, "frame_id": -1, "reasons": ["encounter_clash"]}],
                frame_id=-1, level="coordinate",
            )
        frame = EnsembleFrame(coordinates=x, box=None, topology_ref=self.topology_ref,
                              temperature=self.temperature, weight=1.0, source_id=meta["source_id"],
                              frame_id=0, time=0.0)
        inner = EnsembleFrameSampler(
            pool=EnsembleFramePool([frame]), masses=self.masses, kT=self.kT, backend=self.backend,
            energy_window=self.energy_window, min_pair_dist=None, max_redraws=self.max_redraws,
            constraints=self.constraints, remove_com_momentum=self.remove_com_momentum,
            constraint_tolerance=self.constraint_tolerance, topology_ref=self.topology_ref,
            boltzmann_constant=self.boltzmann_constant,
        )
        try:
            ist, rep = inner.sample(key)
        except ICRejectedError as exc:
            raise ICRejectedError(exc.reasons, [dict(a, frame_id=-1) for a in exc.attempts], frame_id=-1,
                                  level=exc.level) from None
        rep.checks["min_ab_distance"] = dmin
        rep.checks["com_distance"] = meta["com_distance"]
        out_meta = {
            "frame_id": -1, "frame_time": None, "frame_weight": 1.0, "source_id": meta["source_id"],
            "topology_ref": self.topology_ref, "state": list(self.label), "n_redraws": rep.n_redraws,
            **{k: v for k, v in meta.items() if k != "source_id"},
        }
        rep.rejected_attempts = [dict(a, frame_id=-1) for a in rep.rejected_attempts]
        return InitialState(state=ist.state, frame_id=-1, meta=out_meta, origin_label=self.label), rep
