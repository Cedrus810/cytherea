"""Label-constrained weighted ensemble (design doc section 4.5, Task 10).

WE segments are unbiased dynamics; resampling only redistributes *which*
trajectories get compute. The contracts of this module:

Weight bookkeeping (fullreview G-I1)
-------------------------------------
Splits halve a weight (exact in binary floating point for normal numbers);
merges add two weights. `BinnedWE.resample` keeps an exact *ledger*: every
output walker carries the list of ``(input walker, k)`` shares it is made
of (``2**-k`` of that input's weight). After every call it checks, and
raises `RuntimeError` otherwise, that

- every input walker is accounted for exactly once: for each input the
  shares ``2**-k`` over all outputs sum to exactly 1 (integer arithmetic,
  no tolerance -- a dropped walker of any weight, however small, fails);
- every output weight equals ``fsum(w_in * 2**-k)`` over its shares to
  1e-12 relative (so floats and ledger cannot drift apart);
- ``sum(w)`` is conserved per ``(bin, label)`` group to 1e-12 relative
  (``math.fsum`` on both sides);
- ``n_out == n_in + n_splits - n_merges``;
- with the label constraint on, every share of an output has the output's
  ``origin_label`` (structural label isolation).

``BinnedWE.last_ledger`` exposes the ledger of the most recent call.

Merges only within one origin label
------------------------------------
Every walker carries the ``origin_label`` of the initial-state class it
descends from, and `BinnedWE` groups walkers by ``(bin, origin_label)`` and
never merges across groups, so each label's total weight is conserved
exactly and a rare label can never be starved of walkers by a common one.
``allow_cross_label_merge=True`` groups by bin only (the merged walker keeps
the survivor's label). A cross-label merge with the survivor chosen with
probability proportional to weight is still unbiased for every per-label
observable (it is ordinary pairwise resampling in (state, label) space); its
costs are that each label's total weight becomes random, so the denominator
of p(outcome | label) is no longer a known constant and the variance rises,
and that a rare label can be starved. Use it only when that is acceptable.

Keys and noise (fullreview G-C1, contract K7)
---------------------------------------------
Segment noise is seeded by the walker's own ``SegmentKey(global_seed,
run_id, iteration, walker_id)`` (a fresh propagator per segment), so it is
independent of execution order *and* two runs that differ only in
``global_seed`` share no noise: replicas made by changing the seed are
independent. Resampling draws come from ``derive_rng(IterKey, "resample:
<group>")``, one substream per group; recycling draws from
``derive_rng(IterKey, "recycle")``. Resample output is independent of the
order of its input list (walkers are sorted by ``walker_id`` first).

Segment clock and stop-rule persistence (fullreview G-I2, G-M3)
----------------------------------------------------------------
Each segment builds a fresh propagator from the walker's state at ``t=0``
(contract K1) and observes it after build (``k = 0``, the segment's start)
and after every ``dt_obs`` chunk up to ``tau_seg``, at the engine clock
``t = step_index * propagator.dt``. The stop rule sees this
*segment-local* clock.

The stop rule's persistence state is carried across segment boundaries,
not reset: each walker carries ``stop_tail``, its most recent
observations covering the rule's persistence horizon ``tau_persist``, with
their step offsets. Before a segment runs, the rule
is ``reset()`` and the tail is replayed at negative local times, so an
episode inside a persistent region that started in an earlier segment
keeps its entry time. Replaying the last ``tau_persist`` of observations
reconstructs a persistence tracker's state exactly: an entry that has not
fired yet is less than ``tau_persist`` old, and older rows cannot matter
(truncating history can only delay an entry, so the replay cannot fire
where the original run did not). Children of a split inherit the tail; a
merge keeps the survivor's; a recycled walker starts with an empty tail.
The event definition is therefore the one ``run_shot`` uses, whatever
``tau_persist`` is relative to ``tau_seg``; an event whose entry lies in an
earlier segment is reported with a negative (segment-local)
``event_time``. The horizon comes from the rule's protocol description
(`cytherea.engine.shot._describe_stop_rule`): the cytherea rules are
understood; a custom rule must include ``tau_persist`` in its
``protocol_description()`` or WE refuses it. Rules with a deadline of their
own (``FixedLag``'s ``tau``, ``t_max``) must not fire before ``tau_seg``
(checked before anything runs); at or after ``tau_seg`` they are the same
as reaching the segment end.

Segment records
---------------
kind ``"segment"``; ``observables`` holds ``t`` and ``z0, z1, ...`` (plus
any extra ``observables``) exactly as the stop rule saw them: the replayed
tail rows (``t < 0``, copied from ancestor segments) followed by this
segment's own rows ``k = 0..`` (``t >= 0``). ``observables_thinned=False``,
and ``offline_replay`` of the record's series with the same rule reproduces
its event. The first own row is the post-build state, i.e. the parent's
last state. **Consumers that count rows or transitions (Task 11) must not
read ``observables`` as "this segment's trajectory"** (fixreview-p7 I-2):
the tail rows are copies of ancestor rows, carried by every split child,
and row k=0 of a continuing segment repeats its parent's last row. Use
`segment_own_rows(record)` (the rows the segment itself contributes:
``t >= 0``, minus k=0 when ``parent_digest`` is set) or
`lineage_series(records, digest)` (one walker's history, root to digest, on
the WE clock). ``ic_meta`` is the *segment meta* (see `run_segment`);
``code_version`` / ``physics_config_hash`` / ``backend_provenance`` come
from the engine helpers exactly as in ``run_shot`` (contracts K8, K9: the
physics hash is ``resolve_physics_config``'s, i.e. of the effective config), and
``protocol_hash`` hashes the segment protocol (ruling R39). The WE clock of
a row is ``ic_meta["t_start"] + t``.

Recycling and lineage (fullreview G-I3)
----------------------------------------
Recycle targets are drawn with probability proportional to their
``weight`` (fixreview-p7 I-1), and those weights are part of the WE
protocol hash; a target whose own state already satisfies the sink is
refused before anything runs (p7 M-3).

In steady-state mode a walker whose segment ends in the sink is replaced
by one at a recycle target. The new walker's segment record has
``parent_digest=None`` (its trajectory starts at the recycle target, not at
the end of the sink segment) and ``ic_meta["recycled_from"]`` = the digest
of the sink segment its weight came from, ``ic_meta["recycle_target"]`` =
the target index. Following ``parent_digest`` therefore never crosses a
sink -> source jump; a milestoning analysis (Task 11) treats a record with
``recycled_from`` set as a trajectory start.

Deterministic dynamics (fullreview G-I5)
-----------------------------------------
Split children start from the same x and v and differ only through their
propagation noise. With a deterministic integrator (OpenMM ``verlet`` or
``nose_hoover``, Langevin at zero friction or zero temperature, analytic
BAOAB at ``gamma=0``, anything at ``kT=0``) they stay bitwise clones
forever: WE is unbiased but wastes its compute on duplicates, and ``n_eff``
overstates the number of independent samples. `run_we` therefore rejects
such dynamics with ``ValueError`` unless ``allow_deterministic=True``. The
check uses a backend/config ``stochastic`` flag when present, else the
effective physics config, else an empirical probe (two one-chunk
propagations from the same state with different keys must differ). Low
friction Langevin children do diverge, but only through the gamma-weighted
noise plus chaos -- slowly at gamma = 0.1/ps.

Cost (fullreview G-I6)
-----------------------
Every segment calls ``backend.build``; for OpenMM that creates a new
``Context`` (on CUDA roughly 0.5-3 s for a solvated system, i.e. 10-50 % of
a 10-20 ps segment), and every ``dt_obs`` chunk does a ``get_state`` round
trip. Context reuse is deferred until it can be measured on the GPU (it
needs per-segment reseeding, which OpenMM only honours at Context creation
or ``reinitialize``). `run_we` reports per-iteration wall times
(``WERun.build_seconds``, ``WERun.propagate_seconds``) so the overhead is
visible.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import numbers
import time
from collections.abc import Mapping
from typing import Callable, Literal, Protocol

import numpy as np

from cytherea.backends.base import MDState, NumericalInstabilityError, PhysicsConfig
from cytherea.engine.shot import (
    _all_finite,
    _check_measurement_purpose,
    _describe,
    _describe_stop_rule,
    _observable_value,
    _steps_per_chunk,
    code_identity,
    code_version,
    resolve_physics_config,
)
from cytherea.keys import IterKey, SegmentKey, derive_rng, key_digest
from cytherea.observe.events import StopDecision, StopRule
from cytherea.store import ShotRecord, Store, _plain_mapping, config_hash

logger = logging.getLogger(__name__)

# Stop reasons that mean "no event happened; the walker keeps going".
NON_EVENT_REASONS = frozenset({"fixed_lag", "timeout"})
# Stop reason that marks the steady-state sink (recycled + counted as flux).
SINK_REASON = "B"
# Contract K4: a non-finite time, observable or state ends the segment.
NONFINITE_REASON = "nonfinite"
_CONSERVATION_RTOL = 1e-12
_TIME_RTOL = 1e-9
_SEGMENT_PROTOCOL_VERSION = "we-segment/2"


@dataclasses.dataclass
class Walker:
    """One WE walker. ``segment_key`` names the segment this walker runs (or
    just ran) in iteration ``segment_key.iteration``; ``parent`` is the
    segment it descends from (``None`` for initial walkers and for walkers
    whose trajectory restarts at a recycle target). ``z`` is the progress
    coordinate the resampler bins on; ``state`` is what the next segment
    starts from.

    ``recycled_from``: the sink segment whose weight this walker carries,
    when its trajectory was restarted at recycle target ``recycle_target``
    (steady-state WE); ``None`` otherwise. ``stop_tail``: the carried
    stop-rule history, ``((step_offset, obs), ...)`` with ``step_offset <=
    0`` integration steps relative to ``state`` (see the module docstring);
    ``()`` for a fresh trajectory.

    Walkers are never mutated by this module; split children get copies of
    ``z`` and ``state``.
    """

    segment_key: SegmentKey
    parent: SegmentKey | None
    origin_label: tuple[int, int]
    weight: float
    z: np.ndarray
    state: MDState
    recycled_from: SegmentKey | None = None
    recycle_target: int | None = None
    stop_tail: tuple = ()


class Resampler(Protocol):
    """Splits and merges walkers between iterations.

    Carry contract (fixreview-p7 M-6): every walker a resampler returns
    descends from one input walker and must carry that walker's
    ``stop_tail``, ``recycled_from`` and ``recycle_target`` and its
    ``origin_label`` (use ``dataclasses.replace(parent, segment_key=...,
    parent=..., weight=..., z=copy, state=copy)``, as `BinnedWE` does); a
    merge keeps the survivor's. Building ``Walker(...)`` from scratch would
    silently drop the carried persistence state and the recycling marks.
    """

    kind: Literal["none", "uniform", "adaptive", "we", "revo"]

    def resample(self, walkers: list[Walker], it: int, key: IterKey) -> list[Walker]: ...


@dataclasses.dataclass
class WERun:
    """Per-iteration diagnostics of one WE run (arrays of length ``n_iter``).

    Entry ``j`` describes the ensemble that *ran* iteration
    ``start_iteration + j``: ``n_walkers``, ``weights_sum`` and ``n_eff``
    (= 1/sum(p^2), p = w/sum(w)) of those walkers, and ``flux_to_sink`` =
    total weight of those segments that ended in the sink (stop reason
    ``"B"``). ``flux_to_sink`` is a weight per iteration, not a rate: with
    total weight 1 the sink rate is ``flux_to_sink / tau_seg`` (``tau_seg``
    is stored on the result). Time base (p7 M-4): this is the rate of
    *confirmed* persistent events, and weight is re-injected only at the
    next iteration boundary, so by the Hill relation the steady-state flux
    is about 1 / (MFPT_entry + tau_persist + tau_seg / 2); recycled copies
    of one target also share its velocities. Both are negligible when the
    MFPT is much longer than tau_seg + tau_persist.

    ``absorbed[reason][j]``: total weight of the segments that ended with
    event ``reason`` (``"A"``, ``"B"``, ``"reaction"``, ``"escape"``,
    ``"nonfinite"``, ...; ``absorbed["B"]`` equals ``flux_to_sink``). The
    weight leaving the ensemble at ``j`` is the sum of ``absorbed`` over
    every reason except a recycled sink, so ``weights_sum[j+1] ==
    weights_sum[j] - removed[j]``. ``final_walkers`` is the ensemble after
    the last segment (not resampled), ``final_weight`` its total weight --
    the WE analogue of unfinished (timeout) weight. ``n_underweight_alone
    [j]`` counts groups left as one walker below the weight floor by the
    resample after iteration ``j``. ``build_seconds`` / ``propagate_seconds``
    are wall times (G-I6). ``n_nonfinite`` counts segments stopped by
    contract K4; ``valid`` is False when any occurred. Iterations after the
    ensemble has emptied are all zero.
    """

    run_id: str
    flux_to_sink: np.ndarray
    n_eff: np.ndarray
    n_walkers: np.ndarray
    weights_sum: np.ndarray
    absorbed: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    final_walkers: list[Walker] = dataclasses.field(default_factory=list)
    final_weight: float = 0.0
    n_underweight_alone: np.ndarray | None = None
    build_seconds: np.ndarray | None = None
    propagate_seconds: np.ndarray | None = None
    n_nonfinite: int = 0
    global_seed: int | None = None
    start_iteration: int = 0
    tau_seg: float | None = None

    @property
    def valid(self) -> bool:
        return self.n_nonfinite == 0


def n_eff(weights: np.ndarray) -> float:
    """Effective sample size 1/sum(p_k^2) with p = w/sum(w) (0 if empty)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(1.0 / np.sum(p * p))


def _check_weight(w: float) -> None:
    if not (isinstance(w, numbers.Real) and math.isfinite(w) and w > 0.0):
        raise ValueError(f"walker weight must be finite and > 0, got {w!r}")


def _copy_state(s: MDState) -> MDState:
    return MDState(
        x=np.array(s.x, dtype=float, copy=True),
        v=np.array(s.v, dtype=float, copy=True),
        t=float(s.t),
        box=None if s.box is None else np.array(s.box, dtype=float, copy=True),
    )


# One resampler entry: (weight, source walker, ledger shares). A share
# (walker_id, k) is 2**-k of that input walker's weight.
_Entry = tuple[float, Walker, tuple[tuple[int, int], ...]]


class BinnedWE(Resampler):
    """Huber-Kim split/merge on a binned progress coordinate.

    Groups: ``(bin_of(z), origin_label)`` -- the target count
    ``target_per_bin`` applies **per (bin, label) group**, so one label can
    never starve another of walkers in a shared bin. With
    ``allow_cross_label_merge=True`` groups (and the target) are per bin.

    Within each group, in this order (Huber-Kim with the usual 2x / 0.5x
    thresholds, followed by a count adjustment, as WESTPA does it):

    1. Forced merges: while the lightest walker is below ``min_weight`` and
       the group has >= 2 walkers, merge the two lightest. If the group ends
       up as a single walker below ``min_weight`` (it was alone, or the
       whole group's weight is below the floor) it is kept unchanged --
       weight is never dropped -- and counted in ``n_underweight_alone``.
    2. Weight balancing with ``ideal = W_group / target``: halve the
       heaviest walker while it exceeds ``2 * ideal``; then merge the two
       lightest while both are below ``ideal / 2``. (Count-only adjustment
       is not enough: it leaves one heavy walker roaming among tiny ones,
       which in the 10.4 benchmark collapsed the barrier-region weight by
       ten orders of magnitude between rare bursts.)
    3. Count adjustment: while count > target merge the two lightest; while
       count < target halve the heaviest.

    No split ever produces a half below ``min_weight`` (it is skipped).
    Splits are halvings, so they conserve weight exactly. Every call is
    checked against the exact ledger described in the module docstring.

    A merge keeps one of the two walkers' state, chosen with probability
    proportional to weight; the survivor carries the summed weight (and its
    own stop-rule tail and recycle marker). Ties in weight are broken by
    ``walker_id``. Outputs never contain weights below ``min_weight`` (hence
    no 0, NaN or subnormal) except in the ``n_underweight_alone`` case.

    Children get ``SegmentKey(global_seed, run_id, it + 1, i)`` with
    ``i = 0..N-1`` in output order (groups in sorted key order, then by
    parent ``walker_id``) and ``parent`` = the segment key of the walker
    they descend from (for a merge: the survivor) -- or ``None``, with
    ``recycled_from`` kept, when that walker was just recycled.
    """

    kind: Literal["we"] = "we"

    def __init__(
        self,
        bin_of: Callable[[np.ndarray], int],
        target_per_bin: int,
        allow_cross_label_merge: bool = False,
        min_weight: float = 1e-250,
    ) -> None:
        if int(target_per_bin) < 1:
            raise ValueError("target_per_bin must be >= 1")
        if not (math.isfinite(min_weight) and min_weight >= np.finfo(float).tiny):
            raise ValueError("min_weight must be a finite normal float > 0")
        self.bin_of = bin_of
        self.target_per_bin = int(target_per_bin)
        self.allow_cross_label_merge = bool(allow_cross_label_merge)
        self.min_weight = float(min_weight)
        self.n_underweight_alone = 0
        self.last_ledger: dict[SegmentKey, tuple[tuple[SegmentKey, int], ...]] = {}

    def protocol_description(self) -> dict:
        """For the WE run protocol hash (ruling R39); ``bin_of`` must carry a
        ``spec`` (e.g. wrap it in `cytherea.engine.shot.SpecPredicate`)."""
        return {
            "kind": self.kind,
            "bin_of": _describe(self.bin_of, "resampler.bin_of"),
            "target_per_bin": self.target_per_bin,
            "allow_cross_label_merge": self.allow_cross_label_merge,
            "min_weight": self.min_weight,
        }

    def _group_key(self, w: Walker) -> tuple:
        b = int(self.bin_of(w.z))
        if self.allow_cross_label_merge:
            return (b,)
        return (b, tuple(int(v) for v in w.origin_label))

    def resample(self, walkers: list[Walker], it: int, key: IterKey) -> list[Walker]:
        if key.iteration != it:
            raise ValueError(f"IterKey.iteration={key.iteration} != it={it}")
        seen: set[SegmentKey] = set()
        for w in walkers:
            _check_weight(w.weight)
            sk = w.segment_key
            if (sk.global_seed, sk.run_id, sk.iteration) != (key.global_seed, key.run_id, it):
                raise ValueError(
                    f"walker {sk} does not belong to run {key.run_id!r} "
                    f"(global_seed {key.global_seed}) iteration {it}"
                )
            if sk in seen:
                raise ValueError(f"duplicate walker segment key {sk}")
            seen.add(sk)

        ordered = sorted(walkers, key=lambda w: w.segment_key.walker_id)
        groups: dict[tuple, list[Walker]] = {}
        for w in ordered:
            groups.setdefault(self._group_key(w), []).append(w)

        entries: list[_Entry] = []
        n_splits = n_merges = 0
        for g in sorted(groups):
            members = groups[g]
            rng = derive_rng(key, f"resample:{g!r}")
            out, s, m = self._resample_group(members, rng)
            before = math.fsum(w.weight for w in members)
            after = math.fsum(e[0] for e in out)
            if not abs(after - before) <= _CONSERVATION_RTOL * before:
                raise RuntimeError(
                    f"weight not conserved in group {g!r}: before={before!r} after={after!r}"
                )
            entries.extend(out)
            n_splits += s
            n_merges += m

        self._check_ledger(ordered, entries, n_splits, n_merges)

        children: list[Walker] = []
        ledger: dict[SegmentKey, tuple[tuple[SegmentKey, int], ...]] = {}
        by_id = {w.segment_key.walker_id: w for w in ordered}
        for i, (wt, src, shares) in enumerate(entries):
            child_key = SegmentKey(key.global_seed, key.run_id, it + 1, i)
            recycled = src.recycled_from is not None
            children.append(
                Walker(
                    segment_key=child_key,
                    parent=None if recycled else src.segment_key,
                    origin_label=src.origin_label,
                    weight=wt,
                    z=np.array(src.z, dtype=float, copy=True),
                    state=_copy_state(src.state),
                    recycled_from=src.recycled_from,
                    recycle_target=src.recycle_target,
                    stop_tail=src.stop_tail,
                )
            )
            ledger[child_key] = tuple((by_id[wid].segment_key, k) for wid, k in shares)
        self.last_ledger = ledger
        return children

    def _check_ledger(
        self, inputs: list[Walker], entries: list[_Entry], n_splits: int, n_merges: int
    ) -> None:
        by_id = {w.segment_key.walker_id: w for w in inputs}
        if len(entries) != len(inputs) + n_splits - n_merges:
            raise RuntimeError(
                f"walker count bookkeeping broken: {len(inputs)} in + {n_splits} splits "
                f"- {n_merges} merges != {len(entries)} out"
            )
        exps: dict[int, list[int]] = {}
        for wt, src, shares in entries:
            parts = []
            for wid, k in shares:
                exps.setdefault(wid, []).append(k)
                w_in = by_id[wid]
                if not self.allow_cross_label_merge and tuple(w_in.origin_label) != tuple(
                    src.origin_label
                ):
                    raise RuntimeError(
                        f"label constraint broken: weight of {w_in.segment_key} "
                        f"(label {w_in.origin_label}) merged into label {src.origin_label}"
                    )
                parts.append(math.ldexp(w_in.weight, -k))
            exact = math.fsum(parts)
            if not abs(wt - exact) <= _CONSERVATION_RTOL * exact:
                raise RuntimeError(
                    f"output weight {wt!r} != ledger weight {exact!r} (shares {shares})"
                )
        missing = sorted(set(by_id) - set(exps))
        if missing:
            raise RuntimeError(f"walkers lost in resample (walker ids {missing})")
        for wid, ks in exps.items():
            top = max(ks)
            if sum(1 << (top - k) for k in ks) != 1 << top:
                raise RuntimeError(
                    f"walker {by_id[wid].segment_key} is not accounted for exactly once "
                    f"(share exponents {sorted(ks)})"
                )

    def _resample_group(
        self, members: list[Walker], rng: np.random.Generator
    ) -> tuple[list[_Entry], int, int]:
        counts = {"split": 0, "merge": 0}

        def order(e: _Entry) -> tuple[float, int]:
            return (e[0], e[1].segment_key.walker_id)

        def merge_two_lightest(es: list[_Entry]) -> list[_Entry]:
            es.sort(key=order)
            (wa, a, sa), (wb, b, sb) = es[0], es[1]
            total = wa + wb
            survivor = a if rng.random() < wa / total else b
            counts["merge"] += 1
            return [(total, survivor, sa + sb)] + es[2:]

        def split_heaviest(es: list[_Entry], above: float) -> bool:
            es.sort(key=order)
            wt, src, shares = es[-1]
            half = 0.5 * wt  # exact for normal floats: conservation is exact
            if wt <= above or half < self.min_weight:
                return False
            halved = tuple((wid, k + 1) for wid, k in shares)
            es[-1:] = [(half, src, halved), (half, src, halved)]
            counts["split"] += 1
            return True

        def two_lightest_below(es: list[_Entry], limit: float) -> bool:
            if len(es) < 2:
                return False
            es.sort(key=order)
            return es[1][0] < limit

        es: list[_Entry] = [(w.weight, w, ((w.segment_key.walker_id, 0),)) for w in members]
        # 1. forced merges below the weight floor
        while len(es) > 1 and min(es, key=order)[0] < self.min_weight:
            es = merge_two_lightest(es)
        if len(es) == 1 and es[0][0] < self.min_weight:
            self.n_underweight_alone += 1
        # 2. weight balancing toward the group's ideal weight
        ideal = math.fsum(e[0] for e in es) / self.target_per_bin
        while split_heaviest(es, above=2.0 * ideal):
            pass
        while two_lightest_below(es, 0.5 * ideal):
            es = merge_two_lightest(es)
        # 3. count adjustment
        while len(es) > self.target_per_bin:
            es = merge_two_lightest(es)
        while len(es) < self.target_per_bin and split_heaviest(es, above=0.0):
            pass
        es.sort(key=lambda e: e[1].segment_key.walker_id)
        return es, counts["split"], counts["merge"]


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def _rule_times(stop: StopRule) -> tuple[float, float | None, dict]:
    """``(tau_persist, deadline, description)`` of `stop` from its protocol
    description: the persistence horizon and the time at which the rule
    returns a non-event decision by itself (``FixedLag``'s ``tau`` or
    ``t_max``; None if it has none)."""
    desc = _describe_stop_rule(stop)
    params = desc.get("params")
    kind = desc.get("kind")
    if not isinstance(params, Mapping):
        params = {}
    if "tau_persist" in params:
        tau_persist = float(params["tau_persist"])
    elif kind == "fixed_lag":
        tau_persist = 0.0
    else:
        raise ValueError(
            f"stop rule {desc.get('class')!r} does not state its persistence horizon: "
            "WE carries a stop rule's state across segments by replaying the last "
            "tau_persist of observations, so its protocol_description() must "
            "include 'tau_persist' (0 for a memoryless rule)"
        )
    if not (math.isfinite(tau_persist) and tau_persist >= 0.0):
        raise ValueError(f"stop rule tau_persist must be finite and >= 0, got {tau_persist!r}")
    deadline = None
    for name in ("t_max", "tau"):
        if params.get(name) is not None:
            deadline = float(params[name])
            break
    return tau_persist, deadline, desc


@dataclasses.dataclass(frozen=True)
class _SegmentSetup:
    """Everything about a segment that does not depend on the walker,
    resolved and validated once, before anything runs."""

    physics_cfg: PhysicsConfig | None
    effective_cfg: object
    physics_hash: str
    tau_seg: float
    dt_obs: float
    n_obs: int
    tau_persist: float
    extra_obs: Mapping[str, Callable[[MDState], float]]
    protocol_hash: str


def _describe_z_fn(z_fn) -> object:
    """The progress coordinate in the segment protocol (int2-m4): its
    ``spec`` / ``protocol_description()`` when it has one, else its
    qualified name -- like observable functions, which enter by name
    (ruling R39)."""
    desc = getattr(z_fn, "protocol_description", None)
    if callable(desc):
        return {"description": _describe(desc(), "z_fn.protocol_description()")}
    spec = getattr(z_fn, "spec", None)
    if spec is not None:
        return {"spec": _describe(spec, "z_fn.spec")}
    return {"name": f"{getattr(z_fn, '__module__', '?')}.{getattr(z_fn, '__qualname__', type(z_fn).__qualname__)}"}


def _segment_setup(
    backend,
    stop: StopRule,
    physics_cfg: PhysicsConfig | None,
    tau_seg: float,
    dt_obs: float,
    observables: Mapping[str, Callable[[MDState], float]] | None,
    z_fn=None,
) -> _SegmentSetup:
    for name, value in (("tau_seg", tau_seg), ("dt_obs", dt_obs)):
        if not isinstance(value, numbers.Real) or not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    tau_seg, dt_obs = float(tau_seg), float(dt_obs)
    try:
        n_obs = _steps_per_chunk(tau_seg, dt_obs)
    except ValueError:
        raise ValueError(
            f"tau_seg={tau_seg!r} must be an integer multiple of dt_obs={dt_obs!r}"
        ) from None
    extra = dict(observables or {})
    for name in extra:
        if not isinstance(name, str) or name == "t" or (name[:1] == "z" and name[1:].isdigit()):
            raise ValueError(
                f"extra observable name {name!r} is reserved (t and z0, z1, ... are the "
                "segment clock and the progress coordinate)"
            )
    tau_persist, deadline, rule_desc = _rule_times(stop)
    if deadline is not None and deadline < tau_seg * (1.0 - _TIME_RTOL):
        raise ValueError(
            f"stop rule {stop.kind!r} returns a non-event decision at t={deadline!r} "
            f"< tau_seg={tau_seg!r}; WE segments are synchronous and must run to "
            "tau_seg (use FixedLag(tau_seg) or t_max >= tau_seg)"
        )
    effective, physics_hash = resolve_physics_config(backend, physics_cfg)
    _check_measurement_purpose(physics_cfg, effective)
    protocol = {
        "scheme": _SEGMENT_PROTOCOL_VERSION,
        "stop_rule": rule_desc,
        "obs": {
            "progress_coordinate": "z" if z_fn is None else _describe_z_fn(z_fn),
            "observables": sorted(extra),
            "dt_obs": dt_obs,
            "store_stride": 1,
        },
        "tau_seg": tau_seg,
        "carry_stop_state": True,
    }
    return _SegmentSetup(
        physics_cfg=physics_cfg,
        effective_cfg=effective,
        physics_hash=physics_hash,
        tau_seg=tau_seg,
        dt_obs=dt_obs,
        n_obs=n_obs,
        tau_persist=tau_persist,
        extra_obs=extra,
        protocol_hash=config_hash(protocol),
    )


def _z_of(z_fn: Callable[[MDState], np.ndarray], state: MDState) -> np.ndarray:
    z = np.atleast_1d(np.asarray(z_fn(state), dtype=float))
    if z.ndim != 1:
        raise ValueError(f"z_fn must return a scalar or a 1-D array, got shape {z.shape}")
    return z


def _observe(setup: _SegmentSetup, z_fn, state: MDState) -> tuple[np.ndarray, dict[str, float]]:
    z = _z_of(z_fn, state)
    obs = {f"z{i}": float(v) for i, v in enumerate(z)}
    for name, fn in setup.extra_obs.items():
        obs[name] = _observable_value(name, fn(state))
    return z, obs


def run_segment(
    w: Walker,
    backend,
    tau_seg: float,
    stop: StopRule,
    z_fn: Callable[[MDState], np.ndarray],
    store: Store,
    rng_key: SegmentKey,
    dt_obs: float,
    *,
    physics_cfg: PhysicsConfig | None = None,
    observables: Mapping[str, Callable[[MDState], float]] | None = None,
) -> tuple[Walker, StopDecision | None]:
    """Run one WE segment of length ``tau_seg`` and append its record.

    - ``rng_key`` must equal ``w.segment_key`` (checked), so the noise is a
      pure function of (global_seed, run_id, iteration, walker_id).
    - The propagator is built first, ``backend.build(w.state with t=0,
      physics_cfg, rng_key)``; ``dt_obs`` must then be an integer multiple of
      ``propagator.dt`` (`cytherea.engine.shot._steps_per_chunk`, ruling
      R34) and ``tau_seg`` of ``dt_obs``. Both checks happen before any
      store write.
    - The stop rule is ``reset()``, the walker's ``stop_tail`` is replayed
      (module docstring), then the rule is fed ``obs = {"z0": z[0], ...,
      **extra observables}`` at ``t = step_index * dt`` for the post-build
      state and after every ``dt_obs`` chunk up to ``tau_seg``. The segment
      stops early on an *event*; a non-finite ``t``, observable or state
      stops it as ``"nonfinite"`` (contract K4, same check as ``run_shot``;
      the reason string is also accepted from the rule), and so does a
      `NumericalInstabilityError` from the propagator (contract K10: a NaN
      row is recorded at the observation the chunk was heading for, and the
      message goes to the record's ``warnings``). A non-event
      decision (``fixed_lag`` / ``timeout``) before ``tau_seg`` is a
      configuration error (``ValueError``).
    - Record: kind ``"segment"``; key/weight/origin_label of ``w``;
      ``parent_digest`` of ``w.parent``; ``stop_reason`` = the event reason,
      or ``"fixed_lag"`` when the segment ran its full ``tau_seg`` without an
      event; ``event_time`` and ``observables["t"]`` on the segment-local
      clock (tail rows negative); ``observables_thinned=False``;
      ``physics_config_hash`` / ``backend_provenance`` of the cfg passed to
      ``build`` (K8; hash of its effective config, K9); ``code_version`` from
      the engine; ``protocol_hash`` of
      the segment protocol (stop rule, observable names, dt_obs, tau_seg,
      carry scheme; R39); ``ic_meta`` = the segment meta::

          {"kind": "we_segment", "iteration", "walker_id", "t_start" (WE
           clock of t=0: iteration * tau_seg), "tau_seg", "dt_obs",
           "n_carried" (tail rows, t < 0), "recycled_from" (digest or None),
           "recycle_target" (int or None), "stop_state_carried" (bool)}

      (`run_we` adds ``"we_protocol_hash"``).

    Returns the walker advanced to the segment end (same key, weight and
    label; new ``z``/``state``/``stop_tail``, recycle marker cleared) and
    the decision (``None`` if the rule never fired; a synthesized
    ``"nonfinite"`` decision for K4).
    """
    setup = _segment_setup(backend, stop, physics_cfg, tau_seg, dt_obs, observables, z_fn)
    return _run_segment(setup, w, backend, stop, z_fn, store, rng_key)


def _run_segment(
    setup: _SegmentSetup,
    w: Walker,
    backend,
    stop: StopRule,
    z_fn,
    store: Store,
    rng_key: SegmentKey,
    *,
    extra_meta: Mapping | None = None,
    timings: dict | None = None,
) -> tuple[Walker, StopDecision | None]:
    if rng_key != w.segment_key:
        raise ValueError(f"rng_key {rng_key} != walker segment_key {w.segment_key}")
    _check_weight(w.weight)

    t0 = time.perf_counter()
    prop = backend.build(dataclasses.replace(w.state, t=0.0), setup.physics_cfg, rng_key)
    t_build = time.perf_counter() - t0
    dt = float(prop.dt)
    steps_per_obs = _steps_per_chunk(setup.dt_obs, dt)
    steps_per_seg = setup.n_obs * steps_per_obs

    stop.reset()
    steps: list[int] = []
    rows: list[dict[str, float]] = []
    decision: StopDecision | None = None
    # Replay the carried history (all but its last row, which is the state
    # this segment starts from and is observed afresh below).
    for rel, o in w.stop_tail[:-1]:
        steps.append(int(rel))
        rows.append(dict(o))
        if stop.update(dict(o), int(rel) * dt) is not None:
            raise RuntimeError(
                f"replaying the carried stop-rule history of {w.segment_key} produced a "
                "decision; that history already ran without one (is the stop rule a "
                "pure function of its observation series?)"
            )
    n_carried = len(rows)

    t1 = time.perf_counter()
    step = 0
    instability: str | None = None
    state = prop.get_state()
    while True:
        t = step * dt  # contract K1: multiplied, never accumulated
        z, obs_now = _observe(setup, z_fn, state)
        steps.append(step)
        rows.append(obs_now)
        if not _all_finite(t, obs_now, state):
            decision = StopDecision(
                reason=NONFINITE_REASON, event_time=t if math.isfinite(t) else None
            )
            break
        decision = stop.update(obs_now, t)
        if decision is not None:
            if decision.reason in NON_EVENT_REASONS and step < steps_per_seg:
                raise ValueError(
                    f"stop rule returned non-event {decision.reason!r} at t={t!r} "
                    f"< tau_seg={setup.tau_seg!r}; WE segments must run to tau_seg"
                )
            break
        if step >= steps_per_seg:
            break
        try:
            prop.run(steps_per_obs)
            state = prop.get_state()
        except NumericalInstabilityError as exc:
            # contract K10, exactly as in run_shot: the observation this chunk
            # was heading for is recorded as non-finite.
            step += steps_per_obs
            steps.append(step)
            rows.append({name: math.nan for name in rows[-1]})
            instability = f"NumericalInstabilityError at t={step * dt!r}: {exc}"
            decision = StopDecision(reason=NONFINITE_REASON, event_time=step * dt)
            break
        step += steps_per_obs
    t_prop = time.perf_counter() - t1
    if timings is not None:
        timings["build"] = timings.get("build", 0.0) + t_build
        timings["propagate"] = timings.get("propagate", 0.0) + t_prop

    event = decision is not None and decision.reason not in NON_EVENT_REASONS
    reason = decision.reason if event else "fixed_lag"
    observables = {"t": [s * dt for s in steps]}
    for name in rows[-1]:
        observables[name] = [r[name] for r in rows]

    # An unfired persistence episode entered less than tau_persist ago, so
    # the rows of the last tau_persist determine the rule's state.
    horizon_steps = math.ceil(setup.tau_persist / dt * (1.0 - _TIME_RTOL))
    new_tail = tuple(
        (s - step, r) for s, r in zip(steps, rows) if s - step >= -horizon_steps
    )

    it = w.segment_key.iteration
    meta = {
        "kind": "we_segment",
        "iteration": it,
        "walker_id": w.segment_key.walker_id,
        "t_start": it * steps_per_seg * dt,
        "tau_seg": setup.tau_seg,
        "dt_obs": setup.dt_obs,
        "n_carried": n_carried,
        "recycled_from": key_digest(w.recycled_from) if w.recycled_from is not None else None,
        "recycle_target": w.recycle_target,
        "stop_state_carried": bool(w.stop_tail),
    }
    if extra_meta:
        meta.update(extra_meta)
    store.append(
        ShotRecord(
            key_digest=key_digest(w.segment_key),
            key=dataclasses.asdict(w.segment_key),
            kind="segment",
            frame_id=None,
            origin_label=tuple(int(v) for v in w.origin_label),
            ic_validity={},
            stop_rule_kind=stop.kind,
            stop_reason=reason,
            event_time=decision.event_time if decision is not None else None,
            physics_config_hash=setup.physics_hash,
            backend_provenance=backend.provenance(setup.physics_cfg),
            code_version=code_version(),
            observables=observables,
            final_state_label=reason if event and reason != NONFINITE_REASON else None,
            weight=float(w.weight),
            parent_digest=key_digest(w.parent) if w.parent is not None else None,
            ic_meta=_plain_mapping(meta, "segment meta"),
            observables_thinned=False,
            protocol_hash=setup.protocol_hash,
            warnings=[instability] if instability is not None else [],
        )
    )
    out = dataclasses.replace(
        w, z=z, state=state, recycled_from=None, recycle_target=None, stop_tail=new_tail
    )
    return out, decision


# ---------------------------------------------------------------------------
# Deterministic-dynamics guard (G-I5)
# ---------------------------------------------------------------------------


def _cfg_get(cfg: object, name: str):
    if isinstance(cfg, Mapping):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _declared_stochastic(backend, effective_cfg: object) -> bool | None:
    """True/False when the backend or its effective config says whether the
    dynamics are stochastic; None when it cannot be told."""
    for src in (backend, effective_cfg):
        flag = _cfg_get(src, "stochastic") if src is not None else None
        if isinstance(flag, (bool, np.bool_)):
            return bool(flag)
    cfg = effective_cfg
    if cfg is None:
        return None
    integ = _cfg_get(cfg, "integrator")
    if integ in ("overdamped", "baoab"):  # analytic backend
        kT, gamma = _cfg_get(cfg, "kT"), _cfg_get(cfg, "gamma")
        if kT is None or gamma is None:
            return None
        if float(kT) <= 0.0:
            return False
        return integ == "overdamped" or float(gamma) > 0.0
    if integ in ("verlet", "nose_hoover", "langevin_middle"):  # OpenMM
        if _cfg_get(cfg, "system_stochastic_forces"):
            return True  # e.g. an Andersen thermostat in the System
        if integ != "langevin_middle":
            return False
        T, fr = _cfg_get(cfg, "temperature_K"), _cfg_get(cfg, "friction_per_ps")
        if T is None or fr is None:
            return None
        return float(T) > 0.0 and float(fr) > 0.0
    return None


def _probe_stochastic(
    backend, setup: _SegmentSetup, state: MDState, global_seed: int, run_id: str
) -> bool:
    ends = []
    for i in (0, 1):
        key = SegmentKey(global_seed, f"{run_id}/stochastic-probe", 0, i)
        prop = backend.build(dataclasses.replace(state, t=0.0), setup.physics_cfg, key)
        prop.run(_steps_per_chunk(setup.dt_obs, float(prop.dt)))
        s = prop.get_state()
        ends.append((np.asarray(s.x).tobytes(), np.asarray(s.v).tobytes()))
    return ends[0] != ends[1]


# ---------------------------------------------------------------------------
# The WE loop
# ---------------------------------------------------------------------------


def _describe_recycle_targets(targets: list[Walker]) -> list:
    def arr(a, where):
        return None if a is None else _describe(np.asarray(a, dtype=float), where)

    return [
        {
            "origin_label": [int(v) for v in t.origin_label],
            "weight": float(t.weight),
            "x": arr(t.state.x, "recycle_to.x"),
            "v": arr(t.state.v, "recycle_to.v"),
            "box": arr(t.state.box, "recycle_to.box"),
        }
        for t in targets
    ]


def _check_target_outside_sink(setup: _SegmentSetup, stop: StopRule, z_fn, tgt: Walker, idx: int) -> None:
    """p7 M-3: a target that already satisfies the sink would make every
    recycled walker fire again at once. Feed a fresh rule the target's own
    observation for longer than the persistence horizon."""
    _z, obs = _observe(setup, z_fn, tgt.state)
    n = int(math.ceil(setup.tau_persist / setup.dt_obs)) + 1
    stop.reset()
    try:
        for k in range(n + 1):
            dec = stop.update(obs, k * setup.dt_obs)
            if dec is not None and dec.reason == SINK_REASON:
                raise ValueError(
                    f"recycle target {idx} lies in the sink: the stop rule reports "
                    f"{SINK_REASON!r} for its own state, so every recycled walker would "
                    "re-enter the sink at once"
                )
            if dec is not None:
                break
    finally:
        stop.reset()


def _check_continuation(store: Store, global_seed: int, run_id: str, setup: _SegmentSetup,
                        we_hash: str, allow: bool) -> None:
    """int2-m5: refuse to extend a stored WE run with other code / physics /
    protocol (ruling R39 and contract K9, as in run_batch)."""
    seen = store.distinct_values(
        ("code_version", "physics_config_hash", "protocol_hash", "ic_meta.we_protocol_hash"),
        kind="segment", key={"global_seed": global_seed, "run_id": run_id},
    )
    if not seen:
        return
    from cytherea.exec.batch import ResumeConfigMismatchError  # (exec imports engine; no cycle)

    current = code_identity(code_version())
    problems = []
    for code, phys, proto, we_h in sorted(seen, key=repr):
        if code_identity(code) != current:
            problems.append(f"code_version {code!r} != current {code_version()!r}")
        if phys != setup.physics_hash:
            problems.append(f"physics_config_hash {str(phys)[:12]}... != current {setup.physics_hash[:12]}...")
        if proto != setup.protocol_hash:
            problems.append(f"protocol_hash {str(proto)[:12]}... != current {setup.protocol_hash[:12]}...")
        if we_h != we_hash:
            problems.append(f"we_protocol_hash {str(we_h)[:12]}... != current {we_hash[:12]}...")
    if not problems:
        return
    msg = (f"run_we: the store already holds segments of run {run_id!r} (seed {global_seed}) "
           f"made differently: {'; '.join(sorted(set(problems))[:5])}")
    if allow:
        logger.warning("%s (allow_config_change=True, continuing)", msg)
        return
    raise ResumeConfigMismatchError(msg + ". Use a new run_id or store, or pass allow_config_change=True.")


def run_we(
    init: list[Walker],
    backend,
    resampler: Resampler,
    stop: StopRule,
    z_fn: Callable[[MDState], np.ndarray],
    n_iter: int,
    tau_seg: float,
    store: Store,
    global_seed: int,
    run_id: str,
    recycle_to: list[Walker] | None,
    *,
    dt_obs: float,
    physics_cfg: PhysicsConfig | None = None,
    observables: Mapping[str, Callable[[MDState], float]] | None = None,
    allow_deterministic: bool = False,
    start_iteration: int = 0,
    allow_config_change: bool = False,
) -> WERun:
    """Run ``n_iter`` WE iterations ``start_iteration, ...``: propagate every
    walker (in ``walker_id`` order; see `run_segment`), then resample with
    ``IterKey(global_seed, run_id, it)`` (skipped after the last iteration;
    ``WERun.final_walkers`` is the unresampled ensemble).

    Validated before anything runs (G-M4, G-M5): ``init`` carries keys
    ``SegmentKey(global_seed, run_id, start_iteration, i)`` with distinct
    ``i`` and finite positive weights summing to 1 (to 1e-12; at most 1 for
    a continuation, ``start_iteration > 0``); ``tau_seg`` is an integer
    multiple of ``dt_obs``; the stop rule cannot fire a non-event before
    ``tau_seg`` and states its persistence horizon; the physics config is a
    measurement config; the dynamics are stochastic unless
    ``allow_deterministic=True`` (module docstring, G-I5); the protocol
    (segment protocol + ``resampler.protocol_description()`` when it has
    one + recycle targets) can be hashed (R39; stored as
    ``ic_meta["we_protocol_hash"]``). ``dt_obs`` vs the propagator's own
    ``dt`` is checked on the first build, before any record is written.

    A segment ending in an event (any reason other than ``fixed_lag`` /
    ``timeout``) terminates its walker; its weight is added to
    ``absorbed[reason]`` (``flux_to_sink`` for the sink ``"B"``).

    - ``recycle_to`` empty/None: terminated walkers leave the ensemble.
    - ``recycle_to`` non-empty (steady state): only ``"B"`` and
      ``"nonfinite"`` may terminate; any other event raises ``ValueError``
      (after that segment's record was written -- it cannot be known in
      advance). A walker ending in ``"B"`` is replaced by one at a recycle
      target drawn with probability proportional to the targets' weights
      via ``derive_rng(IterKey(global_seed, run_id, it), "recycle")``, with
      the **same weight** (its own, not the target's), a copy of the target's
      state, ``z = z_fn(target.state)``, the **target's** ``origin_label``,
      an empty stop-rule tail, ``parent=None`` and ``recycled_from`` = the
      sink segment (module docstring, G-I3). Recycled copies of one target
      share its velocities; they decorrelate through the noise.
    - A ``"nonfinite"`` walker (K4) is removed in both modes, never
      recycled; its weight is reported in ``absorbed["nonfinite"]`` and makes
      ``WERun.valid`` False.

    To continue a run: ``nxt = resampler.resample(res.final_walkers, last,
    IterKey(global_seed, run_id, last))`` with ``last = start_iteration +
    n_iter - 1``, then ``run_we(nxt, ..., start_iteration=last + 1)``; this
    reproduces the uninterrupted run exactly. Before anything runs, the
    records of this ``(global_seed, run_id)`` already in ``store`` are
    compared with the current run (int2-m5, as `run_batch` does for shots):
    a different ``code_version`` (source hash), ``physics_config_hash``,
    segment ``protocol_hash`` or WE protocol hash raises
    `cytherea.exec.batch.ResumeConfigMismatchError` unless
    ``allow_config_change=True`` (then it is logged). A crashed WE run
    cannot be resumed from the store (records hold no x/v; p7 M-7).
    """
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    start = int(start_iteration)
    if start < 0:
        raise ValueError(f"start_iteration must be >= 0, got {start}")
    if not init:
        raise ValueError("init must contain at least one walker")
    ids = set()
    for w in init:
        _check_weight(w.weight)
        sk = w.segment_key
        if (sk.global_seed, sk.run_id, sk.iteration) != (global_seed, run_id, start) or (
            sk.walker_id in ids
        ):
            raise ValueError(
                f"init walkers need distinct SegmentKey({global_seed}, {run_id!r}, {start}, i); "
                f"got {sk}"
            )
        ids.add(sk.walker_id)
    total0 = math.fsum(w.weight for w in init)
    if start == 0 and abs(total0 - 1.0) > _CONSERVATION_RTOL:
        raise ValueError(
            f"init weights must sum to 1 (got {total0!r}) so that flux_to_sink is a "
            "probability flux"
        )
    if total0 > 1.0 + _CONSERVATION_RTOL:
        raise ValueError(f"walker weights sum to {total0!r} > 1")

    setup = _segment_setup(backend, stop, physics_cfg, tau_seg, dt_obs, observables, z_fn)
    recycle = list(recycle_to or [])
    for tgt in recycle:
        arrays = [tgt.state.x, tgt.state.v] + ([] if tgt.state.box is None else [tgt.state.box])
        if not all(bool(np.all(np.isfinite(a))) for a in arrays):
            raise ValueError("recycle target state is not finite")
        _check_weight(tgt.weight)
    recycle_z = [_z_of(z_fn, tgt.state) for tgt in recycle]
    recycle_p = None
    if recycle:
        tw = np.array([float(t.weight) for t in recycle])
        recycle_p = tw / tw.sum()
        for idx, tgt in enumerate(recycle):
            _check_target_outside_sink(setup, stop, z_fn, tgt, idx)
    res_desc = getattr(resampler, "protocol_description", None)
    we_protocol = {
        "segment": setup.protocol_hash,
        "resampler": (
            _describe(res_desc(), "resampler.protocol_description()")
            if callable(res_desc)
            else {"kind": getattr(resampler, "kind", None), "class": type(resampler).__qualname__}
        ),
        "recycle_to": _describe_recycle_targets(recycle),
    }
    extra_meta = {"we_protocol_hash": config_hash(we_protocol)}
    _check_continuation(store, global_seed, run_id, setup, extra_meta["we_protocol_hash"],
                        allow_config_change)

    if not allow_deterministic:
        stochastic = _declared_stochastic(backend, setup.effective_cfg)
        if stochastic is None:
            stochastic = _probe_stochastic(backend, setup, init[0].state, global_seed, run_id)
        if not stochastic:
            raise ValueError(
                "the dynamics are deterministic (e.g. OpenMM verlet / nose_hoover, zero "
                "friction or temperature, analytic BAOAB at gamma=0): split WE children "
                "would stay bitwise clones forever. Pass allow_deterministic=True to run "
                "anyway."
            )

    flux = np.zeros(n_iter)
    neff = np.zeros(n_iter)
    nwalk = np.zeros(n_iter, dtype=int)
    wsum = np.zeros(n_iter)
    n_alone = np.zeros(n_iter, dtype=int)
    t_build = np.zeros(n_iter)
    t_prop = np.zeros(n_iter)
    absorbed_lists: dict[str, list[list[float]]] = {}
    n_nonfinite = 0

    walkers = sorted(init, key=lambda w: w.segment_key.walker_id)
    for j in range(n_iter):
        it = start + j
        if not walkers:
            break
        ws = np.array([w.weight for w in walkers])
        nwalk[j] = len(walkers)
        wsum[j] = math.fsum(ws)
        neff[j] = n_eff(ws)

        survivors: list[Walker] = []
        recycle_rng = None
        timings: dict = {}
        for w in walkers:
            out, dec = _run_segment(
                setup, w, backend, stop, z_fn, store, w.segment_key,
                extra_meta=extra_meta, timings=timings,
            )
            if dec is None or dec.reason in NON_EVENT_REASONS:
                survivors.append(out)
                continue
            reason = dec.reason
            absorbed_lists.setdefault(reason, [[] for _ in range(n_iter)])[j].append(out.weight)
            if reason == NONFINITE_REASON:
                n_nonfinite += 1
                continue
            if not recycle:
                continue
            if reason != SINK_REASON:
                raise ValueError(
                    f"steady-state WE: segment {w.segment_key} ended with event "
                    f"{reason!r}; only {SINK_REASON!r} is a sink"
                )
            if recycle_rng is None:
                recycle_rng = derive_rng(IterKey(global_seed, run_id, it), "recycle")
            idx = int(recycle_rng.choice(len(recycle), p=recycle_p))
            tgt = recycle[idx]
            survivors.append(
                Walker(
                    segment_key=out.segment_key,
                    parent=None,
                    origin_label=tgt.origin_label,
                    weight=out.weight,
                    z=recycle_z[idx].copy(),
                    state=dataclasses.replace(_copy_state(tgt.state), t=0.0),
                    recycled_from=out.segment_key,
                    recycle_target=idx,
                    stop_tail=(),
                )
            )
        t_build[j] = timings.get("build", 0.0)
        t_prop[j] = timings.get("propagate", 0.0)
        if SINK_REASON in absorbed_lists:
            flux[j] = math.fsum(absorbed_lists[SINK_REASON][j])
        if j < n_iter - 1 and survivors:
            before = getattr(resampler, "n_underweight_alone", 0)
            walkers = resampler.resample(survivors, it, IterKey(global_seed, run_id, it))
            n_alone[j] = getattr(resampler, "n_underweight_alone", 0) - before
        else:
            walkers = survivors

    absorbed = {
        reason: np.array([math.fsum(v) for v in lists]) for reason, lists in absorbed_lists.items()
    }
    return WERun(
        run_id=run_id,
        flux_to_sink=flux,
        n_eff=neff,
        n_walkers=nwalk,
        weights_sum=wsum,
        absorbed=absorbed,
        final_walkers=walkers,
        final_weight=math.fsum(w.weight for w in walkers),
        n_underweight_alone=n_alone,
        build_seconds=t_build,
        propagate_seconds=t_prop,
        n_nonfinite=n_nonfinite,
        global_seed=global_seed,
        start_iteration=start,
        tau_seg=setup.tau_seg,
    )


# ---------------------------------------------------------------------------
# Reading segment records (fixreview-p7 I-2)
# ---------------------------------------------------------------------------


def segment_own_rows(record: ShotRecord) -> dict[str, list]:
    """The rows segment `record` contributes itself (module docstring,
    "Segment records"): ``t >= 0`` (the replayed ancestor tail removed), and
    without row k=0 when the segment continues a parent (``parent_digest``
    set: that row re-observes the parent's last state). Times stay on the
    segment-local clock; add ``record.ic_meta["t_start"]`` for the WE clock.
    Use this -- never the raw ``observables`` -- to count rows or
    transitions over segment records (Task 11)."""
    obs = record.observables
    t = obs["t"]
    keep = [i for i, ti in enumerate(t) if ti >= 0.0]
    if record.parent_digest is not None and keep:
        keep = keep[1:]
    return {name: [series[i] for i in keep] for name, series in obs.items()}


def lineage_series(records: Mapping[str, ShotRecord], digest: str) -> dict[str, list]:
    """One walker's history from its root (an initial or recycled walker)
    to segment `digest`, following ``parent_digest``, as one series on the
    WE clock (``t`` = ``ic_meta["t_start"]`` + segment-local t). `records`
    maps key digests to segment records (e.g. ``{r.key_digest: r for r in
    store.iter(kind="segment")}``). Every row appears once."""
    chain = []
    cur = records[digest]
    while True:
        chain.append(cur)
        if cur.parent_digest is None:
            break
        cur = records[cur.parent_digest]
    out: dict[str, list] = {}
    for rec in reversed(chain):
        own = segment_own_rows(rec)
        t0 = float(rec.ic_meta["t_start"])
        for name, series in own.items():
            vals = [t0 + float(v) for v in series] if name == "t" else list(series)
            out.setdefault(name, []).extend(vals)
    return out
