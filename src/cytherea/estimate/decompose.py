"""Exact population / dynamical decomposition and frame bootstrap (design 4.4).

    Delta<A> = sum dW * Abar  +  sum Wbar * dA,   xbar = (x + x')/2, dx = x' - x

is an identity (no cross term). Sums use ``math.fsum`` so the only round-off
is in the element-wise products; the identity is checked in the function.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Hashable, Mapping, Sequence

import numpy as np


@dataclasses.dataclass
class Decomposition:
    delta: float
    population: float
    dynamical: float


def decompose(W: np.ndarray, Wp: np.ndarray, A: np.ndarray, Ap: np.ndarray) -> Decomposition:
    """Midpoint decomposition of <A>' - <A> with <A> = sum W A (design 4.4).

    ``W``/``Wp`` are the (e.g. P_i P_j) weights of the reference / variant,
    ``A``/``Ap`` the per-cell observable (e.g. beta_inf(i, j)); all four must
    have the same shape and be finite. Raises ``RuntimeError`` if
    ``|population + dynamical - delta| > 1e-12 * sum (|W|+|W'|)(|A|+|A'|)``.
    """
    arrs = [np.asarray(x, dtype=float) for x in (W, Wp, A, Ap)]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError(f"W, Wp, A, Ap must have equal shapes, got {[a.shape for a in arrs]}")
    if not all(np.all(np.isfinite(a)) for a in arrs):
        raise ValueError("W, Wp, A, Ap must be finite")
    W, Wp, A, Ap = (a.ravel() for a in arrs)
    delta = math.fsum(Wp * Ap) - math.fsum(W * A)
    population = math.fsum((Wp - W) * (A + Ap) / 2)
    dynamical = math.fsum((W + Wp) / 2 * (Ap - A))
    scale = math.fsum((np.abs(W) + np.abs(Wp)) * (np.abs(A) + np.abs(Ap)))
    if abs(population + dynamical - delta) > 1e-12 * scale:
        raise RuntimeError(f"decomposition identity violated: {population} + {dynamical} != {delta}")
    return Decomposition(delta=delta, population=population, dynamical=dynamical)


def hierarchical_bootstrap(
    groups: Mapping[Hashable, Mapping[Hashable, Sequence[float]]],
    stat: Callable,
    n_boot: int,
    rng: np.random.Generator,
    resample_states: bool = False,
    resample_shots: bool = False,
) -> np.ndarray:
    """Cluster bootstrap of frames within fixed states (design 4.4, spec
    change S1). Returns ``stat`` per replicate.

    ``groups = {state: {frame: [shot values]}}``. Default: every state is
    kept exactly once, in the key order of ``groups`` -- states are fixed
    strata (e.g. the (i, j) cells with their own weights W_ij in the
    decomposition), not random draws. Within each state as many frames as it
    has are drawn with replacement, and every drawn frame brings *all of its
    shots unchanged* (a single-stage cluster bootstrap; Davison & Hinkley
    1997, sec. 3.8).

    ``resample_shots=True`` restores the two-stage scheme (the design's
    original "frames, then shots within frames"): within each drawn frame
    its shots are drawn again with replacement. That counts the shot noise
    twice -- a frame mean already carries it -- and inflates the variance:
    for the grand mean the ratio of the expected bootstrap variance to the
    true one is

        (F-1)/F + (K-1)/K * se^2 / (K sb^2 + se^2)

    (F frames, K shots per frame, sb^2 / se^2 the between-frame /
    within-frame variances), about 1.7 for binary p_beta-like data
    (F=20, K=10, p_f ~ Beta(12, 28)). The default's ratio is (F-1)/F
    (0.95 at F = 20): consistent as F grows, and slightly *anti*-conservative
    with few frames per state (2/3 at F = 3); with fewer than ~10 frames per
    state widen the intervals or use more frames.

    Every state needs >= 2 frames (``ValueError`` otherwise). Build
    ``groups`` from records with `cytherea.estimate.records_to_groups`.

    ``resample_states=True`` adds a top level that first draws the states
    themselves with replacement (for a grand mean over exchangeable states).

    ``stat`` receives one replicate as a list of ``(state_key, frames)``
    pairs, where ``frames`` is a list of ``(frame_key, shots)`` pairs and
    ``shots`` a read-only 1-D float array. Lists, not dicts, because draws
    with replacement repeat keys; the keys identify the stratum / frame each
    draw came from (frame weights, if any, are the business of ``stat``: the
    bootstrap draws frames uniformly within a state). ``stat`` may return a
    scalar or an array; the result has shape ``(n_boot,) + shape(stat)``.

    Draws follow the iteration order of ``groups`` and of each state's
    frames: for results that are a function of the record set alone (not of
    insertion order), build ``groups`` in a canonical order (e.g. sorted
    keys). ``stat`` is called once per replicate from Python, so cost is
    O(n_boot * number of frames) interpreter work.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    data = []
    for s, frames in groups.items():
        if len(frames) < 2:
            # fixreview-p6 minor 7 (as R30 in estimate_T): one frame gives
            # zero between-frame variance, i.e. a silently too narrow interval
            raise ValueError(
                f"state {s!r} has {len(frames)} frame(s); the frame bootstrap needs >= 2 frames "
                "per state"
            )
        fr = []
        for f, shots in frames.items():
            v = np.array(shots, dtype=float)
            if v.ndim != 1 or v.size == 0:
                raise ValueError(f"frame {f!r} of state {s!r} must hold a non-empty 1-D list of shots")
            v.setflags(write=False)
            fr.append((f, v))
        data.append((s, fr))
    if not data:
        raise ValueError("groups is empty")

    S = len(data)
    out = []
    for _ in range(n_boot):
        states = rng.integers(0, S, size=S) if resample_states else range(S)
        rep = []
        for s in states:
            key, frames = data[s]
            picks = rng.integers(0, len(frames), size=len(frames))
            if resample_shots:
                drawn = []
                for f in picks:
                    fkey, v = frames[f]
                    drawn.append((fkey, v[rng.integers(0, v.size, size=v.size)]))
            else:
                drawn = [frames[f] for f in picks]
            rep.append((key, drawn))
        out.append(stat(rep))
    return np.asarray(out, dtype=float)
