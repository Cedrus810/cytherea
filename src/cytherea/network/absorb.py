"""Absorbing milestone network and its Markov test (design 4.3 / 4.5, Task 11).

Milestones
----------
``milestone_of(obs)`` maps one observation row (``{"t": ..., "z0": ...,
...}``) to the name of a transient milestone, an absorbing state, or
``None`` (in between). A trajectory's milestone label is the LAST milestone
it visited (core-set milestoning / transition-based assignment): rows that
map to ``None`` keep the label, re-entering the current milestone is not a
transition. Each change of label to another transient milestone is a
*hit* of that milestone and a transition from the previous one; the first
row that maps to an absorbing state ends the trajectory (a transition
from the current label into it). A record whose own rows never reach an
absorbing state but whose ``final_state_label`` is an absorbing name (an
event confirmed by the stop rule, e.g. with persistence) is absorbed there
at its end. Rows before the first milestone are ignored.

Milestones are regions, so their width is part of the model: the network
lumps every entry point of a milestone into one state, and on a 1D double
well (12 slabs between the wells, 2 kT barrier, dt = 1e-3) a half-width
of 0.02 biases the network committor by < 0.01 while 0.05 biases it by
0.04. Use slabs no wider than about one observation step.

Records
-------
Segment records (Task 10) are read through `segment_own_rows` -- never the
raw ``observables``, which repeat ancestor rows (fixreview-p7 I-2) -- and
linked through ``parent_digest``: a segment continues its parent's label,
the children of an absorbed segment are ignored, and a record without a
parent (an initial walker, or a recycled one: ``ic_meta["recycled_from"]``)
starts a new trajectory. Every parent must be in the given set
(``ValueError`` otherwise). Shot records are independent trajectories.
Weights: ``record.weight`` for segments, `shot_weights` for shots (K11). A
record with stop reason ``"nonfinite"`` invalidates the data (K4,
``ValueError``).

Counts and the network
----------------------
``N[m, n]`` = total weight of the transitions m -> n (n transient or
absorbing). WE resampling keeps every walker's expected future weight, so
these weighted counts are unbiased fluxes; ``Q = N[:, transient] / row
sum``, ``R = N[:, absorbing] / row sum`` and ``B = (I - Q)^-1 R`` is the
probability of ending in each absorbing state from each milestone.

Direct absorption probabilities
-------------------------------
The outcome of a hit in record s is the absorption vector of s's future:
``e_a`` when s is absorbed in a after the hit, else ``G(s) = sum over
children c of (w_c / w_s) G(c)`` (0 for a trajectory that ends unabsorbed).
Splits and merges keep ``E[G]`` equal to the absorption probability (a
merged-away walker has no children; the survivor's larger weight carries
it), so the direct estimate at milestone m for origin label l is
``sum_hits w_s G(s) / sum_hits w_s sum_a G_a(s)`` -- conditional on
absorption; the unabsorbed weight fraction is reported.

Markov test
-----------
`markov_test` builds the network from ``train`` and compares its B with the
direct estimates from ``heldout``, stratified by (origin label, milestone)
-- origin labels are the extra state the design warns about (4.5, point 1):
a network that pooled the labels gives one prediction for all of them. The
statistic is ``D = max |B_net[m, a] - B_dir[l, m, a]|`` over the strata
with at least ``min_hits`` hits. Its null distribution comes from a
bootstrap over independent units -- WE runs ``(global_seed, run_id)`` (walkers
of one run are coupled by resampling) or single shots -- of train and
heldout, both recomputed, centred on the observed differences (as the CK
test, ruling R23): ``D*_b = max |(B*_net - B*_dir) - (B_net - B_dir)|``.
The test passes when ``D <= ci[1]``, the ``1 - alpha`` quantile of ``D*``;
``ci = (0, ci[1])`` is the acceptance interval of ``max_dev``. Use at least
~20 units on each side.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from cytherea.estimate.records import shot_weights
from cytherea.observe.events import Observables
from cytherea.resample.we import NONFINITE_REASON, segment_own_rows
from cytherea.store import ShotRecord

_ROW_SUM_TOL = 1e-9


@dataclasses.dataclass(frozen=True)
class StageNetwork:
    """Milestone names (``transient``), absorbing state names and whether
    the network is defined on augmented states (milestone, origin label):
    ``augmented=True`` networks are built one label at a time (``label``
    required), ``augmented=False`` ones pool every label (``label=None``)."""

    transient: list[str]
    absorbing: list[str]
    augmented: bool

    def __post_init__(self) -> None:
        tr, ab = list(self.transient), list(self.absorbing)
        if not tr or not ab:
            raise ValueError("a network needs at least one transient and one absorbing state")
        names = tr + ab
        if len(set(names)) != len(names):
            raise ValueError(f"state names must be distinct, got {names}")
        if not all(isinstance(n, str) for n in names):
            raise TypeError("state names must be str")
        object.__setattr__(self, "transient", tr)
        object.__setattr__(self, "absorbing", ab)


@dataclasses.dataclass
class MarkovReport:
    """Result of `markov_test` (module docstring). ``strata``: one entry
    per compared (origin label, milestone) with its hit count, network and
    direct absorption vectors; ``excluded_strata``: strata with fewer than
    ``min_hits`` hits or no absorbed weight; ``n_boot_failed``: replicates
    whose train network had an empty or non-absorbing row (left out)."""

    passed: bool
    max_dev: float
    ci: tuple[float, float]
    worst: tuple | None = None
    strata: list[dict] = dataclasses.field(default_factory=list)
    excluded_strata: list[dict] = dataclasses.field(default_factory=list)
    alpha: float = 0.05
    n_boot: int = 0
    n_boot_failed: int = 0
    n_units_train: int = 0
    n_units_heldout: int = 0
    unabsorbed_fraction: float = 0.0


# ---------------------------------------------------------------------------
# One pass over the records: per-unit counts and per-stratum hit outcomes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _UnitStats:
    N: np.ndarray            # (M, M + A) transition weights
    hit_w: dict              # (label, m) -> sum of hit weights
    hit_out: dict            # (label, m) -> (A,) sum of w * G
    hit_n: dict              # (label, m) -> number of hits


def _check_label(net: StageNetwork, label) -> tuple[int, int] | None:
    if net.augmented and label is None:
        raise ValueError("an augmented network is built per origin label: pass label=(i, j)")
    if not net.augmented and label is not None:
        raise ValueError(
            "a non-augmented network pools every origin label: pass label=None "
            "(use augmented=True to build a network per label)"
        )
    return None if label is None else tuple(int(v) for v in label)


def _rows(series: Mapping[str, list]) -> list[Observables]:
    names = list(series)
    n = len(series[names[0]]) if names else 0
    return [{k: series[k][i] for k in names} for i in range(n)]


def _unit_of(r: ShotRecord) -> tuple:
    if r.kind == "segment":
        return ("we", r.key.get("global_seed"), r.key.get("run_id"))
    return ("shot", r.key_digest)


def _scan(
    records: Iterable[ShotRecord],
    net: StageNetwork,
    milestone_of: Callable[[Observables], str | None],
    label,
) -> dict[tuple, _UnitStats]:
    label = _check_label(net, label)
    recs = [r for r in records if label is None or r.origin_label == label]
    for r in recs:
        if r.stop_reason == NONFINITE_REASON:
            raise ValueError(
                f"record {r.key_digest} stopped as non-finite (contract K4): the data are invalid"
            )
    M, A = len(net.transient), len(net.absorbing)
    t_idx = {n: i for i, n in enumerate(net.transient)}
    a_idx = {n: i for i, n in enumerate(net.absorbing)}

    shots = [r for r in recs if r.kind == "shot"]
    weight = {r.key_digest: float(r.weight) for r in recs if r.kind == "segment"}
    weight.update(zip((r.key_digest for r in shots), (float(w) for w in shot_weights(shots))))

    by_digest = {r.key_digest: r for r in recs}
    children: dict[str, list[ShotRecord]] = defaultdict(list)
    for r in recs:
        if r.parent_digest is not None:
            if r.parent_digest not in by_digest:
                raise ValueError(
                    f"record {r.key_digest} continues segment {r.parent_digest}, which is not in "
                    "the given records: pass whole WE runs (every ancestor of every segment)"
                )
            children[r.parent_digest].append(r)

    def it_of(r: ShotRecord) -> int:
        return int(r.key.get("iteration", 0)) if r.kind == "segment" else 0

    order = sorted(recs, key=it_of)
    end: dict[str, tuple] = {}       # digest -> (last label index or None, absorbed index or None, skipped)
    hits: dict[str, list[int]] = {}
    units: dict[tuple, _UnitStats] = {}

    def stats(r) -> _UnitStats:
        u = _unit_of(r)
        if u not in units:
            units[u] = _UnitStats(np.zeros((M, M + A)), defaultdict(float),
                                  defaultdict(lambda: np.zeros(A)), defaultdict(int))
        return units[u]

    for r in order:
        if r.parent_digest is not None:
            p_last, p_abs, p_skip = end[r.parent_digest]
            if p_abs is not None or p_skip:
                end[r.key_digest] = (None, None, True)
                continue
            last = p_last
        else:
            last = None
        w = weight[r.key_digest]
        st = stats(r)
        absorbed = None
        my_hits: list[int] = []
        for obs in _rows(segment_own_rows(r)):
            name = milestone_of(obs)
            if name is None:
                continue
            if name in a_idx:
                absorbed = a_idx[name]
                break
            if name not in t_idx:
                raise ValueError(f"milestone_of returned {name!r}, which is not a state of the network")
            m = t_idx[name]
            if m != last:
                if last is not None:
                    st.N[last, m] += w
                my_hits.append(m)
                last = m
        if absorbed is None and r.final_state_label in a_idx:
            absorbed = a_idx[r.final_state_label]
        if absorbed is not None and last is not None:
            st.N[last, M + absorbed] += w
        end[r.key_digest] = (last, absorbed, False)
        hits[r.key_digest] = my_hits

    # outcome vectors, children before parents
    G: dict[str, np.ndarray] = {}
    for r in reversed(order):
        last, absorbed, skipped = end[r.key_digest]
        if skipped:
            continue
        if absorbed is not None:
            g = np.zeros(A)
            g[absorbed] = 1.0
        else:
            g = np.zeros(A)
            w = weight[r.key_digest]
            for c in children.get(r.key_digest, ()):
                if c.key_digest in G:
                    g += (weight[c.key_digest] / w) * G[c.key_digest]
        G[r.key_digest] = g
        if hits[r.key_digest]:
            st = stats(r)
            w = weight[r.key_digest]
            lab = None if r.origin_label is None else tuple(r.origin_label)
            for m in hits[r.key_digest]:
                st.hit_w[(lab, m)] += w
                st.hit_out[(lab, m)] += w * g
                st.hit_n[(lab, m)] += 1
    return units


def _counts(units: Iterable[_UnitStats], M: int, A: int) -> np.ndarray:
    N = np.zeros((M, M + A))
    for u in units:
        N += u.N
    return N


def _normalise(N: np.ndarray, M: int) -> tuple[np.ndarray, np.ndarray]:
    tot = N.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(tot > 0, N / np.where(tot > 0, tot, 1.0), np.nan)
    return P[:, :M], P[:, M:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transitions(
    records: Iterable[ShotRecord],
    net: StageNetwork,
    milestone_of: Callable[[Observables], str | None],
    label: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(Q, R)`` from the weighted transition counts (module docstring).

    ``Q[m, n]`` (transient -> transient, zero diagonal) and ``R[m, a]``
    (transient -> absorbing) in the order of ``net.transient`` /
    ``net.absorbing``; each row sums to 1. A milestone without outgoing
    transitions gets a NaN row (`solve_absorption` refuses it). ``label``:
    the origin label of an augmented network, ``None`` for a pooled one.
    """
    M, A = len(net.transient), len(net.absorbing)
    return _normalise(_counts(_scan(records, net, milestone_of, label).values(), M, A), M)


def _non_absorbing_rows(Q: np.ndarray, R: np.ndarray) -> list[int]:
    """Transient states from which no absorbing state can be reached."""
    M = Q.shape[0]
    reach = R.sum(axis=1) > 0
    changed = True
    while changed:
        nxt = reach | ((Q > 0) & reach[None, :]).any(axis=1)
        changed = bool((nxt != reach).any())
        reach = nxt
    return [m for m in range(M) if not reach[m]]


def solve_absorption(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``B = (I - Q)^-1 R``: ``B[m, a]`` = probability of being absorbed in
    ``a`` starting from milestone ``m``. ``[Q R]`` must be a finite,
    non-negative, row-stochastic matrix (rows sum to 1 within 1e-9) from
    whose every transient state an absorbing state can be reached;
    ``ValueError`` otherwise (naming the offending rows)."""
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or R.ndim != 2 or R.shape[0] != Q.shape[0]:
        raise ValueError(f"need Q (M, M) and R (M, A), got {Q.shape} and {R.shape}")
    bad = [m for m in range(Q.shape[0]) if not (np.all(np.isfinite(Q[m])) and np.all(np.isfinite(R[m])))]
    if bad:
        raise ValueError(f"rows {bad} are not finite (milestones without outgoing transitions?)")
    if (Q < 0).any() or (R < 0).any():
        raise ValueError("Q and R must be non-negative")
    sums = Q.sum(axis=1) + R.sum(axis=1)
    off = [m for m in range(Q.shape[0]) if abs(sums[m] - 1.0) > _ROW_SUM_TOL]
    if off:
        raise ValueError(f"rows {off} of [Q R] do not sum to 1: {sums[off].tolist()}")
    closed = _non_absorbing_rows(Q, R)
    if closed:
        raise ValueError(f"no absorbing state is reachable from transient rows {closed}")
    return np.linalg.solve(np.eye(Q.shape[0]) - Q, R)


def _direct(units: Iterable[_UnitStats]) -> tuple[dict, dict, dict]:
    w, out, n = defaultdict(float), {}, defaultdict(int)
    for u in units:
        for k, v in u.hit_w.items():
            w[k] += v
            n[k] += u.hit_n[k]
            out[k] = out.get(k, 0.0) + u.hit_out[k]
    return w, out, n


def markov_test(
    train: Iterable[ShotRecord],
    heldout: Iterable[ShotRecord],
    net: StageNetwork,
    milestone_of: Callable[[Observables], str | None],
    label: tuple[int, int] | None,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    min_hits: int = 10,
    seed: int = 0,
) -> MarkovReport:
    """Network (from ``train``) vs direct absorption probabilities (from
    ``heldout``) per (origin label, milestone); module docstring. ``seed``
    seeds the bootstrap (numpy PCG64)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot!r}")
    M, A = len(net.transient), len(net.absorbing)
    tr = list(_scan(train, net, milestone_of, label).values())
    ho = list(_scan(heldout, net, milestone_of, label).values())
    if len(tr) < 2 or len(ho) < 2:
        raise ValueError(
            f"the bootstrap needs >= 2 independent units on each side (WE runs or shots); got "
            f"{len(tr)} train and {len(ho)} heldout"
        )
    B_net = solve_absorption(*_normalise(_counts(tr, M, A), M))

    w_all, out_all, n_all = _direct(ho)
    strata, excluded = [], []
    for key in sorted(w_all, key=lambda k: (str(k[0]), k[1])):
        lab, m = key
        absorbed = float(out_all[key].sum())
        entry = {"label": lab, "milestone": net.transient[m], "n_hits": int(n_all[key]),
                 "weight": float(w_all[key]), "absorbed_weight": absorbed}
        if n_all[key] < min_hits or absorbed <= 0.0:
            excluded.append(entry)
            continue
        entry["direct"] = (out_all[key] / absorbed).tolist()
        entry["network"] = B_net[m].tolist()
        strata.append((key, entry))
    if not strata:
        raise ValueError(f"no (label, milestone) stratum of the heldout data has >= {min_hits} hits")
    keys = [k for k, _ in strata]
    d_obs = np.array([np.asarray(e["network"]) - np.asarray(e["direct"]) for _, e in strata])
    max_dev = float(np.max(np.abs(d_obs)))
    worst_i = int(np.argmax(np.max(np.abs(d_obs), axis=1)))
    worst = (strata[worst_i][1]["label"], strata[worst_i][1]["milestone"])
    tot_w = sum(w_all.values())
    absorbed_w = sum(float(v.sum()) for v in out_all.values())

    # bootstrap: unit statistics as arrays, replicates as resampling counts
    N_tr = np.stack([u.N.ravel() for u in tr])
    S = len(keys)
    hout = np.array([[u.hit_out.get(k, np.zeros(A)) for k in keys] for u in ho]).reshape(len(ho), S * A)
    rng = np.random.Generator(np.random.PCG64(seed))
    c_tr = np.stack([np.bincount(rng.integers(0, len(tr), len(tr)), minlength=len(tr)) for _ in range(n_boot)])
    c_ho = np.stack([np.bincount(rng.integers(0, len(ho), len(ho)), minlength=len(ho)) for _ in range(n_boot)])
    N_b = (c_tr @ N_tr).reshape(n_boot, M, M + A)
    out_b = (c_ho @ hout).reshape(n_boot, S, A)
    d_star, n_failed = [], 0
    m_of = np.array([k[1] for k in keys])
    for b in range(n_boot):
        Qb, Rb = _normalise(N_b[b], M)
        try:
            Bb = solve_absorption(Qb, Rb)
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1
            continue
        den = out_b[b].sum(axis=1)
        ok = den > 0
        if not ok.any():
            n_failed += 1
            continue
        dirb = out_b[b][ok] / den[ok, None]
        dev = (Bb[m_of[ok]] - dirb) - d_obs[ok]
        d_star.append(float(np.max(np.abs(dev))))
    if not d_star:
        raise ValueError("every bootstrap replicate failed (too few units?)")
    crit = float(np.quantile(np.asarray(d_star), 1.0 - alpha))
    return MarkovReport(
        passed=bool(max_dev <= crit),
        max_dev=max_dev,
        ci=(0.0, crit),
        worst=worst,
        strata=[e for _, e in strata],
        excluded_strata=excluded,
        alpha=float(alpha),
        n_boot=int(n_boot),
        n_boot_failed=int(n_failed),
        n_units_train=len(tr),
        n_units_heldout=len(ho),
        unabsorbed_fraction=float(1.0 - absorbed_w / tot_w) if tot_w > 0 else 0.0,
    )
