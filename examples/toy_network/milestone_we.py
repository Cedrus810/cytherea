"""WE milestoning toys for the A0 acceptance part 2 (Task 11.2-11.4).

Overdamped Langevin (kT = 1, gamma = 1, D = 1, reduced units), dt = 1e-3,
observed every step (dt_obs = dt). A = {x <= -1}, B = {x >= 1} (the well
minima), absorbing without persistence. Twelve milestones: slabs of
half-width 0.02 (about half an rms step, sqrt(2 D dt) = 0.045) around
x = linspace(-0.85, 0.85, 12), across all y.

* ``dw2d``: `DoubleWell2D(barrier=2, ky=1)`; separable, so the committor is
  the 1D one, q(x_m), and x-slabs are an exact Markov milestoning up to
  the slab width.
* ``channel``: `ChannelDoubleWell2D(barrier_plus=1, barrier_minus=4,
  wall=10)`; origin label (1, 1) starts in the y > 0 channel (y0 = +1),
  (2, 2) in the y < 0 one (y0 = -1). The wall keeps every trajectory in its
  channel, so the label -- not the x-milestone -- decides the barrier: the
  pooled x-milestone network is not Markov, the per-label (augmented) one is.

Each WE run starts 2 walkers at every milestone centre (per label), with
equal weights summing to 1, bins on x (20 equal bins on [-1, 1] plus the two
outer ones), 4 walkers per (bin, label), tau_seg = 0.05, `N_ITER`
iterations. `run_replica` is a module-level function so spawn workers can
run it; every run writes its own Store.
"""

from __future__ import annotations

import math

import numpy as np

from cytherea.backends.analytic import AnalyticBackend, ChannelDoubleWell2D, DoubleWell2D
from cytherea.backends.base import MDState
from cytherea.keys import SegmentKey
from cytherea.network import StageNetwork
from cytherea.observe.events import AbsorbingAB, SpecPredicate, spec_region
from cytherea.resample.we import BinnedWE, Walker, run_we
from cytherea.store import Store

KT, GAMMA, DT = 1.0, 1.0, 1.0e-3
TAU_SEG, N_ITER, TARGET_PER_BIN = 0.05, 50, 4
X_A, X_B = -1.0, 1.0
MILESTONES = np.linspace(-0.85, 0.85, 12)
HALF_WIDTH = 0.02
BIN_EDGES = np.linspace(-1.0, 1.0, 21)
LABELS = {"dw2d": {(0, 0): 0.0}, "channel": {(1, 1): 1.0, (2, 2): -1.0}}  # label -> y0


def make_potential(name: str):
    if name == "dw2d":
        return DoubleWell2D(barrier=2.0, ky=1.0)
    if name == "channel":
        return ChannelDoubleWell2D(barrier_plus=1.0, barrier_minus=4.0, wall=10.0)
    raise ValueError(name)


class Milestones:
    """``milestone_of`` for the toys: "A" / "B" / "m<k>" / None from z0 = x."""

    def __call__(self, obs) -> str | None:
        x = float(obs["z0"])
        if x <= X_A:
            return "A"
        if x >= X_B:
            return "B"
        k = int(np.argmin(np.abs(MILESTONES - x)))
        return f"m{k}" if abs(x - MILESTONES[k]) <= HALF_WIDTH else None


def network(augmented: bool) -> StageNetwork:
    return StageNetwork([f"m{k}" for k in range(len(MILESTONES))], ["A", "B"], augmented)


def _z(state: MDState) -> np.ndarray:
    return np.asarray(state.x, dtype=float)


def _bin(z) -> int:
    return int(np.searchsorted(BIN_EDGES, z[0], side="right"))


def run_replica(name: str, seed: int, run_id: str, store_path: str) -> dict:
    """One WE run of toy ``name``; returns a small summary (the records are in the Store)."""
    import time

    t0 = time.perf_counter()
    backend = AnalyticBackend(make_potential(name), "overdamped", DT, KT, gamma=GAMMA, mass=1.0)
    stop = AbsorbingAB(
        spec_region("A", lambda o: o["z0"] <= X_A, {"z0_le": X_A}),
        spec_region("B", lambda o: o["z0"] >= X_B, {"z0_ge": X_B}),
        tau_persist=0.0, t_max=2.0 * TAU_SEG,
    )
    starts = [(float(x), y0, lab) for lab, y0 in LABELS[name].items() for x in MILESTONES for _ in range(2)]
    w0 = 1.0 / len(starts)
    init = []
    for i, (x, y, lab) in enumerate(starts):
        pos = np.array([x, y])
        init.append(Walker(SegmentKey(seed, run_id, 0, i), None, lab, w0, pos.copy(),
                           MDState(x=pos.copy(), v=np.zeros(2), t=0.0)))
    store = Store(store_path)
    res = run_we(init, backend, BinnedWE(SpecPredicate(_bin, {"edges": BIN_EDGES.tolist()}), TARGET_PER_BIN),
                 stop, _z, N_ITER, TAU_SEG, store, seed, run_id, None, dt_obs=DT)
    return {"run_id": run_id, "seed": seed, "store": store_path, "final_weight": res.final_weight,
            "n_segments": int(res.n_walkers.sum()), "valid": res.valid,
            "absorbed": {k: float(v.sum()) for k, v in res.absorbed.items()},
            "wall_s": time.perf_counter() - t0}


def reference_committor(name: str, n: int = 400):
    """q_B at every milestone (per label): the 1D quadrature for ``dw2d`` (separable),
    the finite-volume 2D solve at (x_m, y0) for ``channel``. Needs tests/ on sys.path."""
    from reference.committor_ref import committor_1d, committor_2d
    from cytherea.backends.analytic import DoubleWell1D

    if name == "dw2d":
        q = committor_1d(DoubleWell1D(2.0), KT, X_A, X_B, MILESTONES)
        return {(0, 0): np.asarray(q)}
    ref = committor_2d(make_potential(name), KT, lambda p: p[..., 0] <= X_A, lambda p: p[..., 0] >= X_B,
                       ((X_A - 0.4, X_B + 0.4), (-2.8, 2.8)), n)
    return {lab: ref(np.column_stack([MILESTONES, np.full(MILESTONES.size, y0)]))
            for lab, y0 in LABELS[name].items()}


__all__ = ["Milestones", "network", "run_replica", "reference_committor", "MILESTONES", "LABELS"]
