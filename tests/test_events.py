"""Tests for cytherea.observe.events: persistent-event stop rules for
trajectory shooting (task-6-brief.md, "必测用例" 6.1-6.6, plus the locked
semantics called out in the task instructions: overlap detection, timeout
precedence, monotonic time, the post-decision RuntimeError, missing
observables raising KeyError, and BSurface's escape/reaction precedence).
"""

from __future__ import annotations

import numpy as np
import pytest

from cytherea.observe.events import (
    AbsorbingAB,
    BSurface,
    FixedLag,
    Observables,
    ProtocolDescriptionError,
    Region,
    SpecPredicate,
    StopDecision,
    offline_replay,
    region_description,
    spec_region,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

REGION_A = spec_region("A", lambda o: o["x"] <= -5.0, "x <= -5")
REGION_B = spec_region("B", lambda o: o["x"] >= 5.0, "x >= 5")


def _run(rule, steps: list[tuple[float, Observables]]) -> list[StopDecision | None]:
    """Feed (t, obs) pairs to `rule.update` in order, collecting every result."""
    return [rule.update(obs, t) for t, obs in steps]


# ---------------------------------------------------------------------------
# 6.1: enter B, stay 0.8*tau_p, leave, re-enter and stay 1.2*tau_p -> fires
#      at the second entry's time.
# ---------------------------------------------------------------------------


def test_6_1_persistent_event_fires_at_second_entry_time():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    steps = [
        (0.0, {"x": 6.0}),  # enter B, entry_time = 0.0
        (0.8, {"x": 6.0}),  # still inside, duration 0.8 < tau_p: not fired
        (0.9, {"x": 0.0}),  # leaves B: resets
        (2.0, {"x": 6.0}),  # re-enter B, entry_time = 2.0
        (3.2, {"x": 6.0}),  # duration 1.2 >= tau_p: fires
    ]
    results = _run(rule, steps)
    assert results[:4] == [None, None, None, None]
    decision = results[4]
    assert decision is not None
    assert decision.reason == "B"
    assert decision.event_time == 2.0


# ---------------------------------------------------------------------------
# 6.2: jitter between A and B, every dwell < tau_p -> no event before t_max.
# ---------------------------------------------------------------------------


def test_6_2_jitter_below_persistence_never_fires_before_timeout():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=5.0)
    steps = [
        (0.0, {"x": -6.0}),  # enter A
        (0.5, {"x": -6.0}),  # duration 0.5 < 1.0
        (0.9, {"x": 6.0}),  # switch straight to B, entry_time = 0.9
        (1.3, {"x": 6.0}),  # duration 0.4 < 1.0
        (1.7, {"x": -6.0}),  # switch to A, entry_time = 1.7
        (2.5, {"x": -6.0}),  # duration 0.8 < 1.0
        (3.0, {"x": 0.0}),  # neutral, resets A
        (3.4, {"x": 6.0}),  # enter B, entry_time = 3.4
        (4.0, {"x": 6.0}),  # duration 0.6 < 1.0
        (4.9, {"x": 0.0}),  # neutral
    ]
    results = _run(rule, steps)
    assert all(r is None for r in results)
    final = rule.update({"x": 0.0}, 5.0)
    assert final is not None
    assert final.reason == "timeout"
    assert final.event_time is None


# ---------------------------------------------------------------------------
# 6.3: online and offline replay of the same series give identical decisions.
# ---------------------------------------------------------------------------


def test_6_3_online_and_offline_replay_agree():
    series_t = np.array([0.0, 0.8, 0.9, 2.0, 3.2])
    series_x = np.array([6.0, 6.0, 0.0, 6.0, 6.0])

    online_rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    online_result = None
    for t, x in zip(series_t, series_x):
        online_result = online_rule.update({"x": float(x)}, float(t))
        if online_result is not None:
            break

    offline_rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    offline_result = offline_replay(offline_rule, {"t": series_t, "x": series_x})

    assert online_result is not None
    assert offline_result is not None
    assert online_result.reason == offline_result.reason
    assert online_result.event_time == offline_result.event_time


def test_6_3_offline_replay_returns_none_when_no_event_in_series():
    series_t = np.array([0.0, 1.0, 2.0])
    series_x = np.array([0.0, 0.0, 0.0])
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    result = offline_replay(rule, {"t": series_t, "x": series_x})
    assert result is None


# ---------------------------------------------------------------------------
# 6.4: BSurface escape fires immediately, no persistence required.
# ---------------------------------------------------------------------------

REACTION = spec_region("reaction", lambda o: o["committed"] >= 1.0, "committed >= 1")


@pytest.mark.filterwarnings("ignore:tau_persist")
def test_6_4_bsurface_escape_fires_without_persistence():
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1000.0, t_max=100.0)
    result = rule.update({"committed": 0.0, "r": 5.0}, 0.0)
    assert result is not None
    assert result.reason == "escape"
    assert result.event_time == 0.0


@pytest.mark.filterwarnings("ignore:tau_persist")
def test_6_4_bsurface_escape_fires_at_the_observation_that_crosses_q():
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1000.0, t_max=100.0)
    steps = [
        (0.0, {"committed": 0.0, "r": 1.0}),
        (0.5, {"committed": 0.0, "r": 4.9}),
        (1.0, {"committed": 0.0, "r": 5.0}),
    ]
    results = _run(rule, steps)
    assert results[:2] == [None, None]
    assert results[2] is not None
    assert results[2].reason == "escape"
    assert results[2].event_time == 1.0


# ---------------------------------------------------------------------------
# 6.5: never enters any region -> timeout at t_max, event_time None.
# ---------------------------------------------------------------------------


def test_6_5_never_enters_any_region_gives_timeout_at_t_max():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=3.0)
    steps = [
        (0.0, {"x": 0.0}),
        (1.0, {"x": 1.0}),
        (2.0, {"x": -1.0}),
    ]
    results = _run(rule, steps)
    assert all(r is None for r in results)
    final = rule.update({"x": 0.0}, 3.0)
    assert final == StopDecision(reason="timeout", event_time=None)


def test_6_5_bsurface_never_reacts_or_escapes_gives_timeout():
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=2.0)
    assert rule.update({"committed": 0.0, "r": 0.0}, 0.0) is None
    assert rule.update({"committed": 0.0, "r": 0.0}, 1.0) is None
    final = rule.update({"committed": 0.0, "r": 0.0}, 2.0)
    assert final == StopDecision(reason="timeout", event_time=None)


# ---------------------------------------------------------------------------
# 6.6: FixedLag(tau) fires exactly at t = tau.
# ---------------------------------------------------------------------------


def test_6_6_fixed_lag_fires_exactly_at_tau():
    rule = FixedLag(tau=2.0)
    assert rule.update({}, 0.0) is None
    assert rule.update({}, 1.0) is None
    result = rule.update({}, 2.0)
    assert result == StopDecision(reason="fixed_lag", event_time=2.0)


def test_6_6_fixed_lag_tolerates_floating_point_undershoot():
    # Repeatedly accumulating 0.1 ten times gives 0.9999999999999999 in IEEE
    # double, slightly below 1.0 due to binary rounding, not because the
    # trajectory is "not there yet". FixedLag's epsilon must swallow this.
    tau = 1.0
    almost_tau = 0.0
    for _ in range(10):
        almost_tau += 0.1
    assert almost_tau < tau  # sanity check on the premise of this test

    rule = FixedLag(tau=tau)
    result = rule.update({}, almost_tau)
    assert result is not None
    assert result.reason == "fixed_lag"
    assert result.event_time == almost_tau


# ---------------------------------------------------------------------------
# AbsorbingAB: overlapping A/B at the same observation is ill-posed.
# ---------------------------------------------------------------------------


def test_absorbing_ab_overlap_raises_value_error():
    overlapping_A = spec_region("A", lambda o: o["x"] >= 0.0, "x >= 0")
    overlapping_B = spec_region("B", lambda o: o["x"] >= 0.0, "x >= 0")
    rule = AbsorbingAB(overlapping_A, overlapping_B, tau_persist=1.0, t_max=10.0)
    with pytest.raises(ValueError):
        rule.update({"x": 1.0}, 0.0)


@pytest.mark.filterwarnings("ignore:tau_persist")
def test_absorbing_ab_event_wins_over_coincident_timeout():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=1.0)
    assert rule.update({"x": 6.0}, 0.0) is None  # enter B, entry_time=0.0
    # duration 1.0 >= tau_persist(1.0) AND t == t_max(1.0) simultaneously:
    # the event must win over "timeout".
    decision = rule.update({"x": 6.0}, 1.0)
    assert decision == StopDecision(reason="B", event_time=0.0)


# ---------------------------------------------------------------------------
# Missing observables raise KeyError naming the missing key, not silently
# "outside".
# ---------------------------------------------------------------------------


def test_absorbing_ab_missing_observable_raises_key_error():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=10.0)
    with pytest.raises(KeyError):
        rule.update({}, 0.0)


def test_bsurface_missing_r_name_raises_key_error():
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=10.0)
    with pytest.raises(KeyError):
        rule.update({"committed": 0.0}, 0.0)


def test_bsurface_missing_reaction_observable_raises_key_error():
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=10.0)
    with pytest.raises(KeyError):
        rule.update({"r": 0.0}, 0.0)


# ---------------------------------------------------------------------------
# Monotonic time and the post-decision RuntimeError.
# ---------------------------------------------------------------------------


def test_decreasing_time_raises_value_error():
    rule = FixedLag(tau=10.0)
    rule.update({}, 1.0)
    with pytest.raises(ValueError):
        rule.update({}, 0.5)


def test_repeated_time_is_allowed_not_an_error():
    rule = FixedLag(tau=10.0)
    assert rule.update({}, 1.0) is None
    assert rule.update({}, 1.0) is None  # same t twice: not a decrease


def test_update_after_decision_raises_runtime_error_until_reset():
    rule = FixedLag(tau=1.0)
    decision = rule.update({}, 1.0)
    assert decision is not None
    with pytest.raises(RuntimeError):
        rule.update({}, 2.0)
    rule.reset()
    # after reset, behaves like a fresh rule again
    assert rule.update({}, 0.0) is None
    assert rule.update({}, 1.0) == StopDecision(reason="fixed_lag", event_time=1.0)


def test_absorbing_ab_reset_clears_persistence_state():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    assert rule.update({"x": 6.0}, 0.0) is None  # enter B, entry_time=0.0
    rule.reset()
    # After reset, entering B again at t=0.5 must count as a *new* entry
    # (entry_time=0.5), not resume the pre-reset dwell.
    assert rule.update({"x": 6.0}, 0.5) is None
    decision = rule.update({"x": 6.0}, 1.4)  # duration 0.9 < 1.0: not yet
    assert decision is None
    decision = rule.update({"x": 6.0}, 1.6)  # duration 1.1 >= 1.0: fires
    assert decision == StopDecision(reason="B", event_time=0.5)


# ---------------------------------------------------------------------------
# BSurface precedence between escape and the persistent reaction region.
# ---------------------------------------------------------------------------


def test_bsurface_escape_precedes_pending_reaction():
    # tau_persist is large enough that the reaction is still only "pending"
    # (not yet persistent) when escape triggers at the same observation.
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=5.0, t_max=100.0)
    assert rule.update({"committed": 1.0, "r": 0.0}, 0.0) is None  # enter reaction
    decision = rule.update({"committed": 1.0, "r": 5.0}, 2.0)  # pending + escape
    assert decision == StopDecision(reason="escape", event_time=2.0)


def test_bsurface_completed_reaction_precedes_simultaneous_escape():
    # tau_persist is short enough that the reaction's persistence completes
    # at exactly the same observation where escape also triggers: the
    # completed reaction must win.
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=100.0)
    assert rule.update({"committed": 1.0, "r": 0.0}, 0.0) is None  # enter reaction
    decision = rule.update({"committed": 1.0, "r": 5.0}, 1.0)  # persist done + escape
    assert decision == StopDecision(reason="reaction", event_time=0.0)


# ---------------------------------------------------------------------------
# kind attribute
# ---------------------------------------------------------------------------


def test_kind_attributes():
    assert FixedLag(tau=1.0).kind == "fixed_lag"
    assert AbsorbingAB(REGION_A, REGION_B, 1.0, 10.0).kind == "absorbing_AB"
    assert BSurface(REACTION, "r", 5.0, 1.0, 10.0).kind == "b_surface"


# ---------------------------------------------------------------------------
# Fix round 1 (ruling R16): the persistence check `(t - t_entry) >=
# tau_persist` and the timeout check `t >= t_max` need the same relative
# floating-point tolerance FixedLag already has, because a real integration
# clock accumulates `t` via repeated `t += dt`, not `step * dt` -- the
# review's reproducer found 7/500 entry phases (dt=0.05, tau_persist=0.5)
# where accumulated rounding made a persistence event silently fall through
# to "timeout" instead of firing.
# ---------------------------------------------------------------------------


def test_r16_persistence_tolerates_accumulated_fp_rounding_across_entry_phases():
    """Reviewer's sweep: dt=0.05, tau_persist=0.5 (10 steps). For every one
    of 500 entry phases k, with t accumulated step-by-step (not k*dt), the
    persistent-B event must fire at exactly the 10th observation after
    entry -- never earlier, never later, never falling through to timeout.
    """
    dt = 0.05
    tau_persist = 0.5
    n_persist_steps = 10  # tau_persist / dt, exact only mathematically

    failures = []
    for k in range(500):
        t = 0.0
        for _ in range(k):
            t += dt
        entry_t = t

        obs_times = [entry_t]
        for _ in range(n_persist_steps):
            t += dt
            obs_times.append(t)

        rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=tau_persist, t_max=1.0e9)
        fired_at = None
        fired_decision = None
        for i, ti in enumerate(obs_times):
            decision = rule.update({"x": 6.0}, ti)
            if decision is not None:
                fired_at = i
                fired_decision = decision
                break

        if fired_at != n_persist_steps or fired_decision != StopDecision(
            reason="B", event_time=entry_t
        ):
            failures.append((k, fired_at, fired_decision))

    assert failures == [], f"{len(failures)}/500 entry phases misfired: {failures[:10]}"


def test_r16_persistence_completing_at_accumulated_t_max_wins_over_timeout():
    """An event whose persistence completes at the observation where `t` is
    (an accumulated float) equal to `t_max` must be reported as the event,
    not "timeout" -- mirrors the reviewer's finding that accumulated
    rounding can make a mathematically-on-time event look late enough to
    collide with (or slip past) t_max.
    """
    dt = 0.05
    tau_persist = 0.5
    n_persist_steps = 10
    k = 137  # arbitrary phase offset, exercises accumulated rounding

    t = 0.0
    for _ in range(k):
        t += dt
    entry_t = t
    for _ in range(n_persist_steps):
        t += dt
    completion_t = t  # accumulated, not necessarily entry_t + 0.5 bit-for-bit

    rule = AbsorbingAB(
        REGION_A, REGION_B, tau_persist=tau_persist, t_max=completion_t
    )
    assert rule.update({"x": 6.0}, entry_t) is None
    decision = rule.update({"x": 6.0}, completion_t)
    assert decision == StopDecision(reason="B", event_time=entry_t)


@pytest.mark.filterwarnings("ignore:tau_persist")
def test_r16_timeout_tolerates_floating_point_undershoot():
    """Direct analogue of FixedLag's own undershoot test, for the `t_max`
    cutoff: accumulating 0.1 ten times gives 0.9999999999999999, slightly
    below the mathematically-intended 1.0 cutoff. A trajectory whose clock
    lands there, having entered no region, must still report "timeout".
    """
    t_max = 1.0
    almost_t_max = 0.0
    for _ in range(10):
        almost_t_max += 0.1
    assert almost_t_max < t_max  # sanity check on the premise of this test

    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=1000.0, t_max=t_max)
    decision = rule.update({"x": 0.0}, almost_t_max)
    assert decision == StopDecision(reason="timeout", event_time=None)


# ---------------------------------------------------------------------------
# Full-review fix P3 (D-I3 / contract K5): one shared time tolerance
# 1e-9*max(1, |x|) for FixedLag, persistence and timeout, where x is the
# largest clock magnitude entering the comparison. The previous 1e-12 *
# max(1, |threshold|) scale was too tight at realistic shot times (review
# probe p2_r16_scale.py): the rounding error of an accumulated clock grows
# with |t|, not with |threshold|.
# ---------------------------------------------------------------------------


def _accumulated_clock(dt: float, n_steps: int) -> np.ndarray:
    """OpenMM-style `time += stepSize` clock (np.cumsum is sequential)."""
    return np.concatenate([[0.0], np.cumsum(np.full(n_steps, dt))])


def test_k5_persistence_at_realistic_t_fires_on_time_for_every_entry_phase():
    """dt=2 fs, dt_obs=10 fs, tau_p=1 ps over a 1 ns shot: persistence must
    be confirmed exactly k = tau_p/dt_obs observations after entry, for
    every entry phase -- never one observation late (review: 55% late with
    the old tolerance), never one early.
    """
    dt, stride, tau_p, T = 0.002, 5, 1.0, 1000.0
    t = _accumulated_clock(dt, int(round(T / dt)))[::stride]
    k = int(round(tau_p / (dt * stride)))
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=tau_p, t_max=1.0e9)
    late, early = [], []
    for i in range(0, len(t) - k, 499):  # ~200 entry phases over 0..1 ns
        rule.reset()
        assert rule.update({"x": 6.0}, float(t[i])) is None  # entry
        for j in range(1, k):
            if rule.update({"x": 6.0}, float(t[i + j])) is not None:
                early.append(i)
                break
        else:
            decision = rule.update({"x": 6.0}, float(t[i + k]))
            if decision != StopDecision(reason="B", event_time=float(t[i])):
                late.append(i)
    assert early == [] and late == [], (len(early), len(late))


@pytest.mark.parametrize(
    "dt, stride, tau",
    [(0.002, 5, 1000.0), (0.001, 10, 1000.0), (0.004, 5, 5000.0)],
)
def test_k5_fixed_lag_fires_at_the_observation_equal_to_tau(dt, stride, tau):
    """FixedLag(1000 ps) on an accumulated clock lands at 999.99999999018
    (error -9.8e-9); it must still fire there, and not one observation
    earlier (6.6: "恰好在 t = tau")."""
    t = _accumulated_clock(dt, int(round(tau / dt)))[::stride]
    j = int(round(tau / (dt * stride)))
    rule = FixedLag(tau=tau)
    assert rule.update({}, float(t[j - 1])) is None
    decision = rule.update({}, float(t[j]))
    assert decision == StopDecision(reason="fixed_lag", event_time=float(t[j]))


def test_k5_timeout_at_large_t_max_on_accumulated_clock():
    """t_max = 5000 ps at dt = 4 fs misses by ~9e-8 on an accumulated clock
    (old tolerance 5e-9); the timeout must still fire at that observation."""
    dt, stride, t_max = 0.004, 5, 5000.0
    t = _accumulated_clock(dt, int(round(t_max / dt)))[::stride]
    j = int(round(t_max / (dt * stride)))
    for rule in (
        AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=t_max),
        BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=t_max),
    ):
        obs = {"x": 0.0, "committed": 0.0, "r": 0.0}
        assert rule.update(obs, float(t[j - 1])) is None
        assert rule.update(obs, float(t[j])) == StopDecision(
            reason="timeout", event_time=None
        )


def test_k5_persistence_tolerance_scales_with_the_absolute_clock():
    """The error of `t - t_entry` comes from rounding at the magnitude of
    `t`, not of `tau_persist`: at t ~ 1e5 ps on an accumulated 2 fs clock,
    500 additions put `t - t_entry` ~3.4e-9 below tau_p = 1 ps -- beyond a
    threshold-scaled 1e-9 -- at every one of these entry phases."""
    dt, k, tau_p = 0.002, 500, 1.0
    for j in range(0, 200, 7):
        t_entry = 1.0e5 + j * 0.01
        t = t_entry
        times = [t]
        for _ in range(k):
            t += dt
            times.append(t)
        rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=tau_p, t_max=1.0e9)
        results = [rule.update({"x": 6.0}, ti) for ti in times]
        assert results[:-1] == [None] * k, j
        assert results[-1] == StopDecision(reason="B", event_time=t_entry), j


def test_k5_single_shared_tolerance_helper():
    """All three comparisons share one helper with the K5 scale."""
    from cytherea.observe import events

    assert events._TIME_RTOL == 1e-9
    # below 1: absolute 1e-9
    assert events._at_least(1.0 - 0.9e-9, 1.0)
    assert not events._at_least(1.0 - 1.1e-9, 1.0)
    # scale follows the largest clock magnitude, including `scale=`
    assert events._at_least(1.0 - 0.9e-6, 1.0, scale=1000.0)
    assert not events._at_least(1.0 - 1.1e-6, 1.0, scale=1000.0)
    assert events._at_least(1000.0 - 0.9e-6, 1000.0)
    assert not events._at_least(1000.0 - 1.1e-6, 1000.0)


# ---------------------------------------------------------------------------
# Full-review fix P3 (D-I7 / contract K4): NaN/inf observables or t give an
# immediate StopDecision("nonfinite"); a NaN t must not poison the
# monotonic-time guard.
# ---------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")


def _all_rules():
    return [
        FixedLag(tau=1.0),
        AbsorbingAB(REGION_A, REGION_B, tau_persist=0.5, t_max=2.0),
        BSurface(REACTION, r_name="r", q=5.0, tau_persist=0.5, t_max=2.0),
    ]


def _good_obs():
    return {"x": 0.0, "committed": 0.0, "r": 0.0}


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
@pytest.mark.parametrize("key", ["x", "committed", "r", "unused_extra"])
def test_k4_nonfinite_observable_stops_immediately(bad, key):
    for rule in _all_rules():
        assert rule.update(_good_obs(), 0.0) is None
        obs = _good_obs()
        obs[key] = bad
        decision = rule.update(obs, 0.25)
        assert decision == StopDecision(reason="nonfinite", event_time=0.25), (
            rule.kind,
            decision,
        )
        # a decision was returned: the rule is done until reset
        with pytest.raises(RuntimeError):
            rule.update(_good_obs(), 0.5)


def test_k4_nonfinite_blown_up_trajectory_is_not_a_timeout():
    """Review probe p3 (e): an all-NaN trajectory used to run to "timeout"
    (AbsorbingAB/BSurface) or a normal "fixed_lag" record."""
    for rule in _all_rules():
        decision = rule.update({"x": NAN, "committed": NAN, "r": NAN}, 0.0)
        assert decision == StopDecision(reason="nonfinite", event_time=0.0)


@pytest.mark.parametrize("bad_t", [NAN, INF, -INF])
def test_k4_nonfinite_t_stops_with_event_time_none(bad_t):
    for rule in _all_rules():
        assert rule.update(_good_obs(), 0.0) is None
        decision = rule.update(_good_obs(), bad_t)
        assert decision == StopDecision(reason="nonfinite", event_time=None)


def test_k4_nonfinite_t_on_first_observation():
    for rule in _all_rules():
        assert rule.update(_good_obs(), NAN) == StopDecision(
            reason="nonfinite", event_time=None
        )


def test_k4_nan_t_does_not_disable_the_monotonic_guard():
    """Review probe p3 (e): t=NaN used to be stored as `_last_t`, after
    which `t < NaN` is always False and any decreasing t was accepted."""
    from cytherea.observe.events import _MonotonicTimeGuard

    guard = _MonotonicTimeGuard()
    guard.check(1.0)
    guard.check(NAN)
    with pytest.raises(ValueError):
        guard.check(0.5)

    # the same through a rule: NaN t first, reset, then a decreasing t
    # sequence must still be rejected
    rule = FixedLag(tau=10.0)
    assert rule.update({}, NAN) == StopDecision(reason="nonfinite", event_time=None)
    rule.reset()
    assert rule.update({}, 1.0) is None
    with pytest.raises(ValueError):
        rule.update({}, 0.5)


def test_k4_decreasing_t_still_raises_before_nonfinite_obs():
    """Time monotonicity is a caller bug and stays loud even when the same
    observation is also non-finite."""
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=0.5, t_max=2.0)
    assert rule.update({"x": 0.0}, 1.0) is None
    with pytest.raises(ValueError):
        rule.update({"x": NAN}, 0.5)


def test_k4_nonfinite_beats_overlap_and_completed_event():
    # a NaN observation must not be classified by the region predicates at
    # all -- even if a predicate on some *other* finite key would complete
    # an event at this observation
    A = spec_region("A", lambda o: o["x"] <= -5.0, "x <= -5")
    B = spec_region("B", lambda o: o["x"] >= 5.0, "x >= 5")
    rule = AbsorbingAB(A, B, tau_persist=0.5, t_max=100.0)
    assert rule.update({"x": 6.0, "y": 0.0}, 0.0) is None
    decision = rule.update({"x": 6.0, "y": NAN}, 1.0)
    assert decision == StopDecision(reason="nonfinite", event_time=1.0)


def test_k4_nonfinite_array_valued_observable():
    rule = FixedLag(tau=1.0)
    decision = rule.update({"v": np.array([0.0, NAN])}, 0.1)
    assert decision == StopDecision(reason="nonfinite", event_time=0.1)


def test_k4_offline_replay_of_nan_series_reports_nonfinite():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=0.5, t_max=2.0)
    series = {"t": np.array([0.0, 0.1, 0.2]), "x": np.array([0.0, NAN, NAN])}
    assert offline_replay(rule, series) == StopDecision(
        reason="nonfinite", event_time=0.1
    )


# ---------------------------------------------------------------------------
# D-I7: constructor validation of the rule parameters (a NaN tau/t_max
# never fires and hangs the shot).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [NAN, INF, -INF, -1.0])
def test_fixed_lag_rejects_bad_tau(tau):
    with pytest.raises(ValueError):
        FixedLag(tau=tau)


def test_fixed_lag_tau_zero_allowed():
    assert FixedLag(tau=0.0).update({}, 0.0) == StopDecision(
        reason="fixed_lag", event_time=0.0
    )


@pytest.mark.parametrize(
    "tau_persist, t_max",
    [(NAN, 10.0), (INF, 10.0), (-0.1, 10.0), (1.0, NAN), (1.0, INF), (1.0, 0.0), (1.0, -1.0)],
)
def test_absorbing_rules_reject_bad_times(tau_persist, t_max):
    with pytest.raises(ValueError):
        AbsorbingAB(REGION_A, REGION_B, tau_persist=tau_persist, t_max=t_max)
    with pytest.raises(ValueError):
        BSurface(REACTION, r_name="r", q=5.0, tau_persist=tau_persist, t_max=t_max)


@pytest.mark.parametrize("q", [NAN, INF, -INF])
def test_bsurface_rejects_nonfinite_q(q):
    with pytest.raises(ValueError):
        BSurface(REACTION, r_name="r", q=q, tau_persist=1.0, t_max=10.0)


def test_tau_persist_not_below_t_max_warns():
    with pytest.warns(UserWarning, match="tau_persist"):
        AbsorbingAB(REGION_A, REGION_B, tau_persist=10.0, t_max=10.0)
    with pytest.warns(UserWarning, match="tau_persist"):
        BSurface(REACTION, r_name="r", q=5.0, tau_persist=20.0, t_max=10.0)


# ---------------------------------------------------------------------------
# D-I9 surviving mutations M14 / M15.
# ---------------------------------------------------------------------------


def test_bsurface_reset_clears_persistence_state():
    """M14: a stale pending reaction from the previous shot must not
    complete in the next shot with a foreign event_time."""
    rule = BSurface(REACTION, r_name="r", q=5.0, tau_persist=1.0, t_max=100.0)
    assert rule.update({"committed": 1.0, "r": 0.0}, 0.0) is None  # pending
    rule.reset()
    assert rule.update({"committed": 1.0, "r": 0.0}, 0.5) is None  # new entry
    assert rule.update({"committed": 1.0, "r": 0.0}, 1.4) is None  # 0.9 < 1.0
    decision = rule.update({"committed": 1.0, "r": 0.0}, 1.6)
    assert decision == StopDecision(reason="reaction", event_time=0.5)


@pytest.mark.parametrize("pre_used", ["pending", "decided"])
def test_offline_replay_of_a_used_rule_equals_fresh_rule(pre_used):
    """M15: offline_replay must reset the rule first, whatever state the
    caller left it in."""
    series = {
        "t": np.array([0.0, 0.8, 0.9, 2.0, 3.2]),
        "x": np.array([6.0, 6.0, 0.0, 6.0, 6.0]),
    }
    fresh = offline_replay(
        AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0), series
    )
    used = AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)
    if pre_used == "pending":
        assert used.update({"x": 6.0}, -5.0) is None  # stale entry at t=-5
    else:
        assert used.update({"x": 6.0}, 0.0) is None
        assert used.update({"x": 6.0}, 50.0) is not None  # done
    assert offline_replay(used, series) == fresh
    assert fresh == StopDecision(reason="B", event_time=2.0)


# ---------------------------------------------------------------------------
# Contract K6: offline_replay refuses records whose stored observables were
# thinned (store_stride > 1); a plain series dict works as before.
# ---------------------------------------------------------------------------


def _record_like(thinned, **extra):
    from types import SimpleNamespace

    return SimpleNamespace(
        observables={"t": [0.0, 0.8, 0.9, 2.0, 3.2], "x": [6.0, 6.0, 0.0, 6.0, 6.0]},
        observables_thinned=thinned,
        **extra,
    )


def _ab():
    return AbsorbingAB(REGION_A, REGION_B, tau_persist=1.0, t_max=100.0)


def test_k6_offline_replay_rejects_thinned_record():
    with pytest.raises(ValueError, match="thinned"):
        offline_replay(_ab(), _record_like(True))


def test_k6_offline_replay_rejects_thinned_record_dict():
    rec = vars(_record_like(True))
    with pytest.raises(ValueError, match="thinned"):
        offline_replay(_ab(), rec)


def test_k6_offline_replay_accepts_unthinned_record():
    assert offline_replay(_ab(), _record_like(False)) == StopDecision(
        reason="B", event_time=2.0
    )
    assert offline_replay(_ab(), vars(_record_like(False))) == StopDecision(
        reason="B", event_time=2.0
    )


def test_k6_offline_replay_rejects_record_without_thinning_flag():
    """A record-like object that cannot say whether its series was thinned
    is refused rather than silently replayed."""
    from types import SimpleNamespace

    rec = SimpleNamespace(observables={"t": [0.0], "x": [0.0]})
    with pytest.raises(ValueError, match="no observables_thinned flag"):
        offline_replay(_ab(), rec)
    with pytest.raises(ValueError, match="no observables_thinned flag"):
        offline_replay(_ab(), {"observables": {"t": [0.0], "x": [0.0]}})
    with pytest.raises(ValueError, match="no observables_thinned flag"):
        offline_replay(_ab(), _record_like(None))


def test_k6_offline_replay_plain_series_with_lists():
    assert offline_replay(
        _ab(), {"t": [0.0, 0.8, 0.9, 2.0, 3.2], "x": [6.0, 6.0, 0.0, 6.0, 6.0]}
    ) == StopDecision(reason="B", event_time=2.0)


def test_offline_replay_rejects_ragged_series():
    with pytest.raises(ValueError, match="length"):
        offline_replay(_ab(), {"t": np.array([0.0, 1.0]), "x": np.array([0.0])})


# ---------------------------------------------------------------------------
# Public API exported from cytherea.observe.
# ---------------------------------------------------------------------------


def test_public_api_exported_from_observe_package():
    import cytherea.observe as observe
    from cytherea.observe import events

    for name in (
        "AbsorbingAB",
        "BSurface",
        "FixedLag",
        "Observables",
        "ProtocolDescriptionError",
        "Region",
        "SpecPredicate",
        "StopDecision",
        "StopRule",
        "offline_replay",
        "region_description",
        "spec_region",
    ):
        assert getattr(observe, name) is getattr(events, name)
        assert name in observe.__all__


# ---------------------------------------------------------------------------
# R39: public protocol_description() on the stop rules (INT2 seam 7)
# ---------------------------------------------------------------------------


def test_fixed_lag_protocol_description_is_its_tau():
    assert FixedLag(0.2).protocol_description() == {"tau": 0.2}
    d = FixedLag(1).protocol_description()
    assert d == {"tau": 1.0} and type(d["tau"]) is float


def test_absorbing_ab_protocol_description_names_regions_by_spec():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=0.5, t_max=100.0)
    assert rule.protocol_description() == {
        "A": {"region": "A", "spec": "x <= -5"},
        "B": {"region": "B", "spec": "x >= 5"},
        "tau_persist": 0.5,
        "t_max": 100.0,
    }


def test_bsurface_protocol_description():
    rule = BSurface(REACTION, "r", q=2.5, tau_persist=0.5, t_max=10.0)
    assert rule.protocol_description() == {
        "reaction": {"region": "reaction", "spec": "committed >= 1"},
        "r_name": "r",
        "q": 2.5,
        "tau_persist": 0.5,
        "t_max": 10.0,
    }


def test_protocol_description_does_not_depend_on_rule_state():
    rule = AbsorbingAB(REGION_A, REGION_B, tau_persist=0.5, t_max=100.0)
    before = rule.protocol_description()
    rule.update({"x": -6.0}, 0.0)
    rule.update({"x": -6.0}, 1.0)
    assert rule.protocol_description() == before


def test_region_description_spec_sources_and_opaque_predicate():
    assert region_description(spec_region("A", lambda o: True, {"obs": "x", "lt": 1.0})) == {
        "region": "A", "spec": {"obs": "x", "lt": 1.0}
    }
    assert region_description(Region("A", SpecPredicate(lambda o: True, "s"))) == {"region": "A", "spec": "s"}

    def pred(o):
        return True

    pred.spec = "t"
    assert region_description(Region("A", pred)) == {"region": "A", "spec": "t"}
    opaque = AbsorbingAB(Region("A", lambda o: False), REGION_B, tau_persist=0.5, t_max=100.0)
    with pytest.raises(ProtocolDescriptionError, match="Region 'A'"):
        opaque.protocol_description()
    # an opaque region still works as a stop rule; only hashing needs a spec
    assert opaque.update({"x": 0.0}, 0.0) is None


def test_spec_predicate_calls_through():
    region = spec_region("A", lambda o: o["x"] < 0.0, "x < 0")
    assert region.predicate({"x": -1.0}) is True
    assert region.predicate({"x": 1.0}) is False
