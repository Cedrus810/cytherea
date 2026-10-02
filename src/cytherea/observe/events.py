"""Observables, regions, and persistent-event stop rules (Task 6).

Every trajectory's outcome (A / B / reaction / escape / timeout / nonfinite)
is decided by one of these rules, and committor and association-probability
estimates are just counts of those outcomes -- so "online" (called once per
observation as the trajectory runs) and "offline" (replayed afterwards from a
saved observable series, via `offline_replay`) must produce *exactly* the
same `StopDecision` for the same series. That equivalence is what makes it
safe to, e.g., re-derive a stopping decision from a stored trajectory without
re-running dynamics -- provided the stored series is the one the rule saw
(see `offline_replay` on thinned records).

Locked semantics (task-6-brief.md, controller instructions, fixplan K4-K6):

- `update(obs, t)` requires `t` to be non-decreasing across calls; a
  decrease raises `ValueError`. Equal `t` twice is *not* an error.
- **Non-finite input (contract K4).** If `t` or *any* value in `obs` is NaN
  or +-inf, the rule immediately returns
  `StopDecision("nonfinite", event_time=t)` -- or `event_time=None` when `t`
  itself is non-finite. This takes precedence over every other outcome
  (a completed event, escape, timeout, fixed lag, even the A/B overlap
  check): a blown-up trajectory must never be classified by region
  predicates, which silently evaluate False on NaN and would otherwise turn
  it into a "timeout" or a normal "fixed_lag" record (h2oleps lesson: an
  all-NaN trajectory must not look like a normal one). `FixedLag` ignores
  `obs` for its own decision but still applies this check to every value.
  The only checks that come before it are the caller-bug guards: an
  `update()` after a decision raises `RuntimeError`, and a *finite* `t`
  below the last finite `t` raises `ValueError`. A non-finite `t` is never
  stored as the guard's "last t", so it cannot disable the monotonic-time
  check (with NaN, `t < last_t` would be False forever after).
- Persistence is measured in observation time, not wall/sim-step count: an
  event fires at the first observation where `(t - t_entry) >= tau_persist`
  while the observed point has stayed continuously inside the region since
  `t_entry`. `event_time` is `t_entry` -- the time of the first observation
  found inside, not the time persistence was confirmed. Any observation
  found outside the region resets the entry clock (`_PersistenceTracker`
  below implements exactly this and is shared by `AbsorbingAB` and
  `BSurface`).
- `AbsorbingAB`: an observation inside both `A` and `B` simultaneously means
  the regions overlap, which is an ill-posed configuration -> `ValueError`
  (not "pick one arbitrarily"). Timeout ("timeout", `event_time=None`) is
  reported once `t >= t_max`, but only if no event (A or B) also completed
  at that exact observation -- an event and a coincident timeout are
  resolved in the event's favor. An event that has *entered* but not yet
  completed persistence at `t_max` is reported as "timeout": choose
  `t_max >= (target horizon) + tau_persist` if events entering just before
  the horizon must be counted (constructors warn when
  `tau_persist >= t_max`, where no event could ever complete).
- `BSurface`: escape (`obs[r_name] >= q`) requires no persistence and fires
  on the observation it is first true. A *pending* (not yet persistent)
  reaction never blocks escape. A reaction whose persistence completes at
  the very same observation where escape also triggers wins over escape --
  i.e. precedence is: completed reaction > escape > pending
  reaction/timeout. This is implemented by checking the reaction's
  persistence completion first and only falling through to escape when it
  did not complete on this observation (see `BSurface.update`). The escape
  test is a plain `>=` on a continuous observable, with *no* tolerance: the
  shared tolerance below is a clock tolerance and applies only to time
  comparisons.
- `FixedLag(tau)` fires at the first observation with `t >= tau` up to the
  shared time tolerance, so that e.g. a `tau` reached only up to
  double-precision rounding (`sum([0.1] * 10) == 0.9999999999999999 !=
  1.0`) still counts as "there".
- **Every time comparison against a caller-supplied cutoff uses the same
  tolerance (contract K5)**, via the one helper `_at_least(value,
  threshold, scale=...)`: `value >= threshold - 1e-9 * max(1, |x|)`, where
  `|x|` is the largest clock magnitude taking part in the comparison
  (`|value|`, `|threshold|`, and for persistence also the absolute clock
  `|t|`, `|t_entry|` the difference was formed from). This applies
  identically to (1) `FixedLag`'s `t` vs `tau`, (2) `_PersistenceTracker`'s
  `(t - t_entry)` vs `tau_persist`, and (3) every `t` vs `t_max` timeout
  check (`AbsorbingAB` and `BSurface`). Why this scale (review D-I3, which
  superseded ruling R16's `1e-12 * max(1, |threshold|)`): the rounding
  error of a simulation clock -- accumulated `t += dt` as OpenMM does, or
  even `step * dt` -- grows with `|t|`, not with the (often small)
  threshold; at `t ~ 1 ns` with `dt = 2 fs` the accumulated error of
  `t - t_entry` reaches ~2e-11 ps and of `t` itself ~1e-8 ps, well past the
  old 1e-12 tolerance, so persistence was confirmed one observation late
  in more than half of the entry phases and could slip past `t_max` into a
  false "timeout". A tolerance of `1e-9 * |t|` is still far below one
  observation interval (dt_obs / |t| is ~1e-5 even at 1 ns with a 10 fs
  cadence), so it can never make a decision fire one observation early.
- Once any rule returns a `StopDecision`, it is done: further `update()`
  calls raise `RuntimeError` until `reset()` is called.
- A `Region`'s predicate (and `BSurface`'s `r_name` lookup) index `obs`
  directly (`obs[name]`), so a missing observable raises `KeyError` naming
  it -- it is never silently treated as "outside the region". Nothing in
  this module catches or reinterprets that `KeyError`.
- Constructors validate their parameters: `tau`, `tau_persist`, `t_max` and
  `q` must be finite (a NaN cutoff never fires and would hang the shot),
  `tau >= 0`, `tau_persist >= 0`, `t_max > 0`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

Observables = dict[str, float]

# Contract K5: the one relative time tolerance used by every time comparison.
_TIME_RTOL = 1e-9


def _at_least(value: float, threshold: float, *, scale: float = 0.0) -> bool:
    """`value >= threshold`, tolerant of floating-point clock rounding.

    Equivalent to `value >= threshold - 1e-9 * max(1, |value|, |threshold|,
    |scale|)` (contract K5). `scale` is the magnitude of the absolute clock
    a difference was formed from (persistence passes `max(|t|, |t_entry|)`),
    since that, not the threshold, sets the rounding error. Every time
    comparison in this module (`FixedLag`'s `tau`, `_PersistenceTracker`'s
    `tau_persist`, and every `t_max` timeout check) goes through this one
    function so the tolerance cannot drift apart between them.
    """
    tol = _TIME_RTOL * max(1.0, abs(value), abs(threshold), abs(scale))
    return value >= threshold - tol


def _is_finite(value: Any) -> bool:
    """True iff `value` (a scalar or array-like observable) is all-finite."""
    return bool(np.all(np.isfinite(value)))


def _nonfinite_decision(obs: Observables, t: float) -> StopDecision | None:
    """Contract K4: a `"nonfinite"` decision if `t` or any observable is
    NaN/inf, else `None`."""
    t_ok = _is_finite(t)
    if t_ok and all(_is_finite(v) for v in obs.values()):
        return None
    return StopDecision(reason="nonfinite", event_time=float(t) if t_ok else None)


def _finite_param(name: str, value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _check_persist_and_t_max(tau_persist: float, t_max: float) -> tuple[float, float]:
    tau_persist = _finite_param("tau_persist", tau_persist)
    t_max = _finite_param("t_max", t_max)
    if tau_persist < 0.0:
        raise ValueError(f"tau_persist must be >= 0, got {tau_persist!r}")
    if t_max <= 0.0:
        raise ValueError(f"t_max must be > 0, got {t_max!r}")
    if tau_persist >= t_max:
        warnings.warn(
            f"tau_persist={tau_persist!r} >= t_max={t_max!r}: in run_shot a "
            "persistent event can complete before the timeout only if it was "
            "entered at t <= t_max - tau_persist <= 0, so (with the shot clock "
            "starting at 0) essentially every persistent event will be "
            "reported as 'timeout'. (Under WE this is fine: run_segment carries "
            "the persistence history across segments; int2-m7.)",
            UserWarning,
            stacklevel=3,
        )
    return tau_persist, t_max


@dataclass(frozen=True)
class Region:
    """A named subset of observable space, given as a boolean predicate.

    For a Region to enter a shot's protocol hash (ruling R39) its predicate
    must carry a canonical description: build it with `spec_region`, wrap
    the predicate in `SpecPredicate`, or give the predicate a ``spec``
    attribute. An opaque lambda cannot be described (never hashed by id).
    """

    name: str
    predicate: Callable[[Observables], bool]


# ---------------------------------------------------------------------------
# Protocol descriptions (ruling R39)
# ---------------------------------------------------------------------------


class ProtocolDescriptionError(TypeError):
    """Part of the shot protocol (stop rule, region, sampler, constraints,
    ...) cannot be described canonically, so no protocol_hash can be
    computed. Give the object a ``protocol_description()`` method, or a
    Region predicate a ``spec`` (see `spec_region`)."""


@dataclass(frozen=True)
class SpecPredicate:
    """A Region predicate that carries a canonical description (`spec`, a
    str or a plain JSON dict) of what it tests, so it can enter the
    protocol hash. Calling it calls `fn`."""

    fn: Callable[[Observables], bool]
    spec: object

    def __call__(self, obs: Observables) -> bool:
        return self.fn(obs)


@dataclass(frozen=True)
class SpecLabeler:
    """A final-state labeler (`run_shot(labeler=...)`) that carries a
    canonical description (`spec`, a str or a plain JSON dict) of the state
    definition it applies, so it enters the protocol hash (R39, int2-m2).
    Calling it calls `fn`. Change the spec whenever `fn` changes."""

    fn: Callable[[Observables], str]
    spec: object

    def __call__(self, obs: Observables) -> str:
        return self.fn(obs)


def spec_region(name: str, predicate: Callable[[Observables], bool], spec: object) -> Region:
    """``Region(name, SpecPredicate(predicate, spec))``: a region whose
    predicate is described by `spec` (e.g. ``"position < -1.2"`` or
    ``{"obs": "position", "lt": -1.2}``). The spec is the caller's promise
    of what the predicate computes; change it whenever the predicate
    changes."""
    return Region(name, SpecPredicate(predicate, spec))


def region_description(region: Region, where: str = "region") -> dict:
    """``{"region": name, "spec": spec}`` for a Region whose predicate is
    described (a ``spec`` on the Region, else on its predicate). Raises
    `ProtocolDescriptionError` for an opaque predicate."""
    spec = getattr(region, "spec", None)
    if spec is None:
        spec = getattr(region.predicate, "spec", None)
    if spec is None:
        raise ProtocolDescriptionError(
            f"{where}: Region {region.name!r} has an opaque predicate and no spec; "
            "build it with cytherea.observe.spec_region(name, predicate, spec) "
            "so the stop rule can be hashed (ruling R39)"
        )
    return {"region": region.name, "spec": spec}


@dataclass
class StopDecision:
    reason: Literal[
        "fixed_lag",
        "A",
        "B",
        "reaction",
        "escape",
        "timeout",
        "nonfinite",
        "pes_uncertain",
    ]
    event_time: float | None


@runtime_checkable
class StopRule(Protocol):
    """A stateful, resettable online stop-decision engine.

    `kind` is a plain class attribute (not a method) so callers can branch
    on rule type without invoking anything.
    """

    kind: Literal["fixed_lag", "absorbing_AB", "b_surface"]

    def update(self, obs: Observables, t: float) -> StopDecision | None: ...

    def reset(self) -> None: ...


class _PersistenceTracker:
    """Shared bookkeeping for "has the observed point been continuously
    inside a region for at least `tau_persist`" (see module docstring).
    """

    def __init__(self, tau_persist: float) -> None:
        self._tau_persist = tau_persist
        self._entry_time: float | None = None

    def reset(self) -> None:
        self._entry_time = None

    def observe(self, inside: bool, t: float) -> float | None:
        """Record this observation. Returns the entry time if persistence
        is satisfied as of `t`, else `None`.
        """
        if not inside:
            self._entry_time = None
            return None
        if self._entry_time is None:
            self._entry_time = t
        if _at_least(
            t - self._entry_time,
            self._tau_persist,
            scale=max(abs(t), abs(self._entry_time)),
        ):
            return self._entry_time
        return None


class _MonotonicTimeGuard:
    """Shared bookkeeping: reject decreasing `t`, and reject any `update`
    call once a decision has already been returned, until `reset()`.

    A non-finite `t` passes `check` (the caller then returns a "nonfinite"
    decision) but is never stored as `_last_t`: NaN would make every later
    `t < _last_t` comparison False and silently switch the guard off.
    """

    def __init__(self) -> None:
        self._last_t: float | None = None
        self._done = False

    def reset(self) -> None:
        self._last_t = None
        self._done = False

    def check(self, t: float) -> None:
        if self._done:
            raise RuntimeError(
                "update() called again after a StopDecision was already "
                "returned; call reset() first"
            )
        if not _is_finite(t):
            return
        if self._last_t is not None and t < self._last_t:
            raise ValueError(
                f"observation time must be monotonically increasing: "
                f"got t={t!r} after t={self._last_t!r}"
            )
        self._last_t = t

    def mark_done(self) -> None:
        self._done = True


class FixedLag:
    """Fires unconditionally once `t` reaches `tau` (observables play no part
    in the decision -- `obs` is accepted to match the `StopRule` signature,
    and is only checked for non-finite values, contract K4).
    """

    kind: Literal["fixed_lag"] = "fixed_lag"

    def __init__(self, tau: float) -> None:
        tau = _finite_param("tau", tau)
        if tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {tau!r}")
        self._tau = tau
        self._guard = _MonotonicTimeGuard()

    def protocol_description(self) -> dict:
        """Canonical constructor parameters, for the engine's protocol_hash
        (ruling R39): ``{"tau": tau}``."""
        return {"tau": self._tau}

    def reset(self) -> None:
        self._guard.reset()

    def update(self, obs: Observables, t: float) -> StopDecision | None:
        self._guard.check(t)
        decision = _nonfinite_decision(obs, t)
        if decision is None and _at_least(t, self._tau):
            decision = StopDecision(reason="fixed_lag", event_time=t)
        if decision is not None:
            self._guard.mark_done()
        return decision


class AbsorbingAB:
    """Two mutually-exclusive persistent absorbing regions, A and B, plus a
    timeout at `t_max`. See module docstring for exact precedence rules.
    """

    kind: Literal["absorbing_AB"] = "absorbing_AB"

    def __init__(self, A: Region, B: Region, tau_persist: float, t_max: float) -> None:
        tau_persist, t_max = _check_persist_and_t_max(tau_persist, t_max)
        self._A = A
        self._B = B
        self._t_max = t_max
        self._guard = _MonotonicTimeGuard()
        self._tracker_A = _PersistenceTracker(tau_persist)
        self._tracker_B = _PersistenceTracker(tau_persist)

    def protocol_description(self) -> dict:
        """Canonical constructor parameters, for the engine's protocol_hash
        (ruling R39): both regions by name and spec (`region_description`;
        raises `ProtocolDescriptionError` for an opaque predicate),
        tau_persist and t_max."""
        return {
            "A": region_description(self._A, "stop.A"),
            "B": region_description(self._B, "stop.B"),
            "tau_persist": self._tracker_A._tau_persist,
            "t_max": self._t_max,
        }

    def reset(self) -> None:
        self._guard.reset()
        self._tracker_A.reset()
        self._tracker_B.reset()

    def update(self, obs: Observables, t: float) -> StopDecision | None:
        self._guard.check(t)
        decision = _nonfinite_decision(obs, t)
        if decision is not None:
            self._guard.mark_done()
            return decision

        in_A = self._A.predicate(obs)
        in_B = self._B.predicate(obs)
        if in_A and in_B:
            raise ValueError(
                f"observation at t={t!r} is inside both region "
                f"{self._A.name!r} and region {self._B.name!r}: "
                "A and B must not overlap"
            )

        entry_A = self._tracker_A.observe(in_A, t)
        entry_B = self._tracker_B.observe(in_B, t)

        if entry_A is not None:
            decision = StopDecision(reason="A", event_time=entry_A)
        elif entry_B is not None:
            decision = StopDecision(reason="B", event_time=entry_B)
        elif _at_least(t, self._t_max):
            decision = StopDecision(reason="timeout", event_time=None)

        if decision is not None:
            self._guard.mark_done()
        return decision


class BSurface:
    """A persistent "reaction" region plus a non-persistent escape surface
    `obs[r_name] >= q`, and a timeout at `t_max`. See module docstring for
    the exact escape/reaction precedence rule.
    """

    kind: Literal["b_surface"] = "b_surface"

    def __init__(
        self,
        reaction: Region,
        r_name: str,
        q: float,
        tau_persist: float,
        t_max: float,
    ) -> None:
        tau_persist, t_max = _check_persist_and_t_max(tau_persist, t_max)
        self._reaction = reaction
        self._r_name = r_name
        self._q = _finite_param("q", q)
        self._t_max = t_max
        self._guard = _MonotonicTimeGuard()
        self._tracker = _PersistenceTracker(tau_persist)

    def protocol_description(self) -> dict:
        """Canonical constructor parameters, for the engine's protocol_hash
        (ruling R39): the reaction region by name and spec, r_name, q,
        tau_persist and t_max."""
        return {
            "reaction": region_description(self._reaction, "stop.reaction"),
            "r_name": self._r_name,
            "q": self._q,
            "tau_persist": self._tracker._tau_persist,
            "t_max": self._t_max,
        }

    def reset(self) -> None:
        self._guard.reset()
        self._tracker.reset()

    def update(self, obs: Observables, t: float) -> StopDecision | None:
        self._guard.check(t)
        decision = _nonfinite_decision(obs, t)
        if decision is not None:
            self._guard.mark_done()
            return decision

        in_reaction = self._reaction.predicate(obs)
        entry_reaction = self._tracker.observe(in_reaction, t)
        r = obs[self._r_name]
        escaped = r >= self._q  # plain >=: no clock tolerance (see docstring)

        if entry_reaction is not None:
            # Reaction persistence completed on this exact observation:
            # it wins even if escape is also true here (see module
            # docstring precedence rule).
            decision = StopDecision(reason="reaction", event_time=entry_reaction)
        elif escaped:
            decision = StopDecision(reason="escape", event_time=t)
        elif _at_least(t, self._t_max):
            decision = StopDecision(reason="timeout", event_time=None)

        if decision is not None:
            self._guard.mark_done()
        return decision


_MISSING = object()


def _series_of(source: Any) -> Mapping[str, Any]:
    """The observable series to replay, from either a plain series mapping
    or a stored record (contract K6); see `offline_replay`."""
    if isinstance(source, Mapping):
        if "t" in source or "observables" not in source:
            return source  # plain series dict, as before
        series = source["observables"]
        thinned = source.get("observables_thinned", _MISSING)
    elif hasattr(source, "observables"):
        series = source.observables
        thinned = getattr(source, "observables_thinned", _MISSING)
    else:
        raise TypeError(
            "offline_replay expects a series mapping {'t': ..., name: ...} or a "
            f"record with an 'observables' series, got {type(source).__name__}"
        )
    if thinned is _MISSING or thinned is None:
        raise ValueError(
            "record has no observables_thinned flag, so it cannot be shown that "
            "its stored series is the one the stop rule saw online; refusing "
            "to replay (pass record.observables explicitly if you know it is "
            "unthinned)"
        )
    if thinned:
        raise ValueError(
            "record observables were thinned (observables_thinned=True, "
            "store_stride > 1): the stored series is not the series the stop "
            "rule saw online, so an offline replay would not reproduce the "
            "online decision; re-run with store_stride=1 to replay"
        )
    return series


def offline_replay(rule: StopRule, series: Any) -> StopDecision | None:
    """Reset `rule` and feed it `series` row by row, in `t` order.

    `series` is either a plain series mapping -- `series["t"]` gives the
    observation times, every other key becomes an `Observables` entry with
    the value at the matching row index -- or a stored record (an object or
    a dict with an `observables` series and an `observables_thinned` flag,
    e.g. a `ShotRecord`). A record flagged `observables_thinned=True`
    (stored with `store_stride > 1`) raises `ValueError`, since its series
    is not what the rule saw online; a record without the flag is refused
    too (contract K6). All series columns must have the same length as
    `t`, else `ValueError`.

    Returns the first `StopDecision` produced, or `None` if the series ends
    without one. This must agree, row for row, with calling `rule.update`
    online as each observation arrives (task 6.3) -- it is nothing more
    than that loop, driven from arrays instead of a live trajectory.

    One addition for stored records (int2-m1): the engine also stops a
    trajectory as ``"nonfinite"`` when only its *state* (x, v, box) went
    non-finite while every observable stayed finite -- a check the series
    cannot show. A record with ``stop_reason == "nonfinite"`` whose series
    ends without a decision therefore replays to
    ``StopDecision("nonfinite", record.event_time)``, the online decision.
    """
    data = _series_of(series)
    times = data["t"]
    names = [name for name in data if name != "t"]
    n = len(times)
    ragged = {name: len(data[name]) for name in names if len(data[name]) != n}
    if ragged:
        raise ValueError(
            f"series columns must all have the length of 't' ({n}); got {ragged}"
        )
    rule.reset()
    for i in range(n):
        obs = {name: data[name][i] for name in names}
        decision = rule.update(obs, float(times[i]))
        if decision is not None:
            return decision
    if _record_field(series, "stop_reason") == "nonfinite":
        return StopDecision("nonfinite", _record_field(series, "event_time"))
    return None


def _record_field(source: Any, name: str) -> Any:
    """`name` of a stored record (object or dict); None for a plain series."""
    if isinstance(source, Mapping):
        return source.get(name) if "observables" in source and "t" not in source else None
    return getattr(source, name, None)
