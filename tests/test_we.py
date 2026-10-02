"""Task 10: label-constrained weighted ensemble (BinnedWE, run_segment, run_we).

Tests 10.1-10.7 from the task brief plus the controller's decision tests
(cross-label isolation under pressure, execution-order independence of
segment noise, weight conservation under steady-state recycling) and the
fullreview G-we regression tests (C1, I1-I6, K4, K8, R39, Minors).
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np
import openmm
import pytest

from reference.mfpt_ref import committor_1d, mfpt_1d
from cytherea.backends.analytic import AnalyticBackend, DoubleWell1D
from cytherea.backends.base import MDState, NumericalInstabilityError, PhysicsConfig
from cytherea.backends.openmm_backend import OpenMMBackend
from cytherea.engine.shot import ProtocolDescriptionError, SpecPredicate, code_version, spec_region
from cytherea.keys import IterKey, SegmentKey, derive_rng, key_digest
from cytherea.observe.events import AbsorbingAB, FixedLag, Region, StopDecision, offline_replay
from cytherea.resample.we import (
    BinnedWE,
    Walker,
    WERun,
    lineage_series,
    n_eff,
    run_segment,
    run_we,
    segment_own_rows,
)
from cytherea.store import Store, config_hash

LA = (0, 0)
LB = (1, 1)


# Aliases kept for the tests below (the K8 shims are gone: fixreview-p7 M-8).
_Analytic = AnalyticBackend
_OpenMM = OpenMMBackend


def _walker(run_id, it, wid, weight, x, label, parent=None, seed=0):
    x = np.array([float(x)])
    return Walker(
        segment_key=SegmentKey(seed, run_id, it, wid),
        parent=parent,
        origin_label=label,
        weight=weight,
        z=x.copy(),
        state=MDState(x=x.copy(), v=np.zeros(1), t=0.0),
    )


def _walkers(specs, run_id="r", it=0, seed=0):
    """specs: list of (weight, x, label)."""
    return [_walker(run_id, it, i, w, x, lab, seed=seed) for i, (w, x, lab) in enumerate(specs)]


def _floor_bin_fn(z):
    return int(math.floor(z[0]))


_floor_bin = SpecPredicate(_floor_bin_fn, "floor(z0)")


def _label_sums(walkers):
    out = {}
    for w in walkers:
        out.setdefault(w.origin_label, []).append(w.weight)
    return {k: math.fsum(v) for k, v in out.items()}


def _rel_err(a, b):
    return abs(a - b) / abs(b)


def _relabel(walkers, rng, n_bins):
    """Move every walker to a random z (simulates propagation between rounds)."""
    return [
        dataclasses.replace(w, z=np.array([rng.uniform(0, n_bins)]))
        for w in walkers
    ]


def _check_ledger_exact(we, inputs, out, labels_isolated=True):
    """Every input walker is accounted for exactly once in `we.last_ledger`
    (sum of 2**-k over its shares == 1, exactly), each output's weight is
    the ledger weight, and (label constraint) every share of an output
    carries the output's label."""
    by_key = {w.segment_key: w for w in inputs}
    frac = {k: Fraction(0) for k in by_key}
    assert set(we.last_ledger) == {w.segment_key for w in out}
    for w in out:
        shares = we.last_ledger[w.segment_key]
        exact = math.fsum(math.ldexp(by_key[k].weight, -e) for k, e in shares)
        assert w.weight == pytest.approx(exact, rel=1e-12, abs=0.0)
        for k, e in shares:
            frac[k] += Fraction(1, 2**e)
            if labels_isolated:
                assert by_key[k].origin_label == w.origin_label
    assert all(f == 1 for f in frac.values()), {k: f for k, f in frac.items() if f != 1}


# ---------------------------------------------------------------- 10.1


def test_10_1_weight_conserved_every_round():
    rng = derive_rng(IterKey(0, "t101", 0), "test-setup")
    labels = [LA, LB]
    specs = [
        (10.0 ** rng.uniform(-20, 0), rng.uniform(0, 5), labels[i % 2])
        for i in range(60)
    ]
    walkers = _walkers(specs, run_id="t101", seed=7)
    total0 = math.fsum(w.weight for w in walkers)
    per_label0 = _label_sums(walkers)
    we = BinnedWE(_floor_bin, target_per_bin=3)
    worst = 0.0
    for it in range(50):
        out = we.resample(walkers, it, IterKey(7, "t101", it))
        total = math.fsum(w.weight for w in out)
        worst = max(worst, _rel_err(total, total0))
        assert _rel_err(total, total0) < 1e-12
        # Label constraint => each label's total weight is conserved too.
        for lab, s in _label_sums(out).items():
            assert _rel_err(s, per_label0[lab]) < 1e-12
        assert set(_label_sums(out)) == set(per_label0)
        # G-I1: every input walker (however light) is accounted for exactly.
        _check_ledger_exact(we, walkers, out)
        walkers = _relabel(out, rng, 5)
    assert worst < 1e-12


def test_10_1_every_input_accounted_for_with_tiny_weights():
    # G-I1 variant of 10.1: weights spanning 1e-200..1 share bins, so a
    # walker lighter than 1e-12 of its group is common; each must be found
    # in the outputs' ledger with total share exactly 1.
    rng = derive_rng(IterKey(0, "t101b", 0), "test-setup")
    specs = [(10.0 ** rng.uniform(-200, 0), rng.uniform(0, 3), [LA, LB][i % 2]) for i in range(40)]
    walkers = _walkers(specs, run_id="t101b")
    we = BinnedWE(_floor_bin, target_per_bin=4)
    for it in range(20):
        out = we.resample(walkers, it, IterKey(0, "t101b", it))
        _check_ledger_exact(we, walkers, out)
        walkers = _relabel(out, rng, 3)


def test_tiny_walker_is_never_dropped():
    # G-I1: a walker 1e-100 of its group's weight must survive unchanged
    # (target 2, so nothing forces a merge): the output multiset is exact.
    walkers = _walkers([(1.0, 0.5, LA), (1e-100, 0.5, LA)])
    out = BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 0, IterKey(0, "r", 0))
    assert sorted(w.weight for w in out) == [1e-100, 1.0]
    walkers = _walkers([(0.6, 0.5, LA), (0.4, 0.5, LA), (1e-150, 0.5, LA)])
    out = BinnedWE(_floor_bin, target_per_bin=3).resample(walkers, 0, IterKey(0, "r", 0))
    assert sorted(w.weight for w in out) == [1e-150, 0.4, 0.6]


def test_runtime_ledger_catches_silently_dropped_walkers(monkeypatch):
    # G-I1: the reviewer's mutation -- _resample_group silently drops every
    # walker lighter than 1e-13 of its group. The per-group relative sum
    # check cannot see it; the exact ledger must.
    orig = BinnedWE._resample_group

    def dropping(self, members, rng):
        es, s, m = orig(self, members, rng)
        tot = math.fsum(e[0] for e in es)
        return [e for e in es if e[0] > 1e-13 * tot] or es, s, m

    monkeypatch.setattr(BinnedWE, "_resample_group", dropping)
    walkers = _walkers([(1.0, 0.5, LA), (1e-100, 0.5, LA)])
    with pytest.raises(RuntimeError, match="count bookkeeping|lost"):
        BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 0, IterKey(0, "r", 0))


def test_runtime_ledger_catches_share_loss(monkeypatch):
    # A mutation that keeps the walker count but loses a merged walker's
    # share (merge without carrying its ledger) must also be caught.
    orig = BinnedWE._resample_group

    def lossy(self, members, rng):
        es, s, m = orig(self, members, rng)
        return [(wt, src, sh[:1]) for wt, src, sh in es], s, m

    monkeypatch.setattr(BinnedWE, "_resample_group", lossy)
    walkers = _walkers([(0.25, 0.5, LA), (0.25, 0.5, LA), (0.5, 0.5, LA)])
    with pytest.raises(RuntimeError):
        BinnedWE(_floor_bin, target_per_bin=1).resample(walkers, 0, IterKey(0, "r", 0))


def test_resample_hits_target_per_bin_label_group():
    specs = [(0.1, 0.5, LA)] * 7 + [(0.05, 0.5, LB)] + [(0.3, 1.5, LA)]
    out = BinnedWE(_floor_bin, target_per_bin=3).resample(
        _walkers(specs), 0, IterKey(0, "r", 0)
    )
    groups = {}
    for w in out:
        groups.setdefault((_floor_bin(w.z), w.origin_label), []).append(w)
    assert {k: len(v) for k, v in groups.items()} == {
        (0, LA): 3,
        (0, LB): 3,
        (1, LA): 3,
    }


def test_resample_rejects_bad_input_weights():
    we = BinnedWE(_floor_bin, target_per_bin=2)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            we.resample(_walkers([(bad, 0.5, LA), (0.5, 0.5, LA)]), 0, IterKey(0, "r", 0))


def test_resample_rejects_walkers_of_another_seed():
    walkers = _walkers([(0.5, 0.5, LA), (0.5, 0.5, LA)], seed=1)
    with pytest.raises(ValueError, match="global_seed"):
        BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 0, IterKey(2, "r", 0))


# ---------------------------------------------------------------- 10.2


def _mixed_bin(n=10):
    # Interleaved weights so smallest-pair merging would pair across labels
    # if labels were not respected.
    specs = []
    for i in range(n):
        specs.append((0.01 * (2 * i + 1), 0.5, LA))
        specs.append((0.01 * (2 * i + 2), 0.5, LB))
    return _walkers(specs)


def test_10_2_no_cross_label_merge_by_default():
    walkers = _mixed_bin()
    we = BinnedWE(_floor_bin, target_per_bin=2)
    out = we.resample(walkers, 0, IterKey(0, "r", 0))
    assert sorted(w.origin_label for w in out) == [LA, LA, LB, LB]
    # G-M2: structural -- every input merged into an output has its label.
    _check_ledger_exact(we, walkers, out, labels_isolated=True)
    before, after = _label_sums(walkers), _label_sums(out)
    for lab in (LA, LB):
        assert _rel_err(after[lab], before[lab]) < 1e-12


def test_10_2_cross_label_merge_only_when_allowed():
    walkers = _mixed_bin()
    we = BinnedWE(_floor_bin, target_per_bin=2, allow_cross_label_merge=True)
    out = we.resample(walkers, 0, IterKey(0, "r", 0))
    assert len(out) == 2  # target applies per bin when labels are pooled
    total_before = math.fsum(w.weight for w in walkers)
    assert _rel_err(math.fsum(w.weight for w in out), total_before) < 1e-12
    _check_ledger_exact(we, walkers, out, labels_isolated=False)
    by_key = {w.segment_key: w for w in walkers}
    mixed = any(
        len({by_key[k].origin_label for k, _ in we.last_ledger[w.segment_key]}) > 1 for w in out
    )
    assert mixed, "pooled merging should have merged walkers of different labels"


def test_no_cross_label_merge_under_min_weight_pressure():
    # One under-weight LB walker alone among many LA walkers in the same bin:
    # it must be kept (not merged into LA), counted as a warning, and its
    # weight must survive exactly.
    specs = [(0.2, 0.5, LA)] * 5 + [(1e-300, 0.5, LB)]
    walkers = _walkers(specs)
    we = BinnedWE(_floor_bin, target_per_bin=1, min_weight=1e-250)
    out = we.resample(walkers, 0, IterKey(0, "r", 0))
    lb = [w for w in out if w.origin_label == LB]
    assert len(lb) == 1 and lb[0].weight == 1e-300
    assert len([w for w in out if w.origin_label == LA]) == 1
    assert we.n_underweight_alone == 1


def test_label_isolation_over_many_rounds_under_pressure():
    rng = derive_rng(IterKey(0, "iso", 0), "test-setup")
    specs = [(1.0 / 40, rng.uniform(0, 2), [LA, LB][i % 2]) for i in range(40)]
    walkers = _walkers(specs, run_id="iso", seed=1)
    per0 = _label_sums(walkers)
    we = BinnedWE(_floor_bin, target_per_bin=1)  # heavy merge pressure
    for it in range(30):
        out = we.resample(walkers, it, IterKey(1, "iso", it))
        _check_ledger_exact(we, walkers, out, labels_isolated=True)
        for lab, s in _label_sums(out).items():
            assert _rel_err(s, per0[lab]) < 1e-12
        walkers = _relabel(out, rng, 2)


def test_underweight_walker_force_merged_within_group():
    specs = [(0.5, 0.5, LA), (0.5 - 1e-300, 0.5, LA), (1e-300, 0.5, LA), (5e-324, 0.5, LA)]
    we = BinnedWE(_floor_bin, target_per_bin=4, min_weight=1e-250)
    walkers = _walkers(specs)
    out = we.resample(walkers, 0, IterKey(0, "r", 0))
    assert all(w.weight >= 1e-250 for w in out)
    assert we.n_underweight_alone == 0
    assert _rel_err(math.fsum(w.weight for w in out), 1.0) < 1e-12
    _check_ledger_exact(we, walkers, out)


def test_merge_survivor_chosen_proportional_to_weight():
    we = BinnedWE(_floor_bin, target_per_bin=1)
    n, hits = 4000, 0
    for k in range(n):
        walkers = _walkers([(0.25, 0.5, LA), (0.75, 0.6, LA)], seed=k)
        (out,) = we.resample(walkers, 0, IterKey(k, "r", 0))
        assert out.weight == 1.0
        hits += out.parent == walkers[0].segment_key
    p = hits / n
    assert abs(p - 0.25) < 4 * math.sqrt(0.25 * 0.75 / n)


def test_huber_kim_weight_balancing_within_group():
    # One heavy walker among many tiny ones: a count-only scheme would keep
    # the heavy walker whole and merge the tiny ones; Huber-Kim must leave
    # every walker within [ideal/2, 2*ideal] except at most one light one.
    specs = [(0.9, 0.5, LA)] + [(0.1 / 9, 0.5, LA)] * 9
    target = 4
    out = BinnedWE(_floor_bin, target_per_bin=target).resample(_walkers(specs), 0, IterKey(0, "r", 0))
    ideal = 1.0 / target
    ws = sorted(w.weight for w in out)
    assert len(ws) == target
    assert max(ws) <= 2 * ideal
    assert sum(w < ideal / 2 for w in ws) <= 1


def test_child_keys_and_parents():
    walkers = _walkers([(0.25, 0.5, LA), (0.75, 1.5, LB)], run_id="kid", it=3, seed=4)
    out = BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 3, IterKey(4, "kid", 3))
    assert [w.segment_key for w in out] == [SegmentKey(4, "kid", 4, i) for i in range(4)]
    parents = {w.segment_key for w in walkers}
    assert all(w.parent in parents for w in out)


def test_split_children_do_not_alias_parent_arrays():
    # G-M8: children get their own z / state arrays.
    walkers = _walkers([(1.0, 0.5, LA)])
    out = BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 0, IterKey(0, "r", 0))
    assert len(out) == 2
    a, b = out
    assert a.state.x is not b.state.x and a.z is not b.z
    assert a.state.x is not walkers[0].state.x
    a.state.x[0] = 99.0
    assert b.state.x[0] == 0.5 and walkers[0].state.x[0] == 0.5


def test_children_inherit_stop_tail_and_recycle_marker():
    # G-I2 / G-I3: split children clone the carried stop-rule tail; a merge
    # keeps the survivor's; a just-recycled walker's children have no parent
    # but keep `recycled_from`.
    tail_a = ((-1, {"z0": 0.1}), (0, {"z0": 0.2}))
    tail_b = ((0, {"z0": 0.9}),)
    ws = _walkers([(0.25, 0.5, LA), (0.75, 0.5, LA)])
    ws[0] = dataclasses.replace(ws[0], stop_tail=tail_a)
    sink = SegmentKey(0, "r", 0, 7)
    ws[1] = dataclasses.replace(ws[1], stop_tail=tail_b, recycled_from=sink, recycle_target=0)
    we = BinnedWE(_floor_bin, target_per_bin=1)
    (merged,) = we.resample(ws, 0, IterKey(0, "r", 0))
    assert merged.stop_tail in (tail_a, tail_b)
    if merged.stop_tail == tail_b:
        assert merged.parent is None and merged.recycled_from == sink and merged.recycle_target == 0
    else:
        assert merged.parent == ws[0].segment_key and merged.recycled_from is None
    split = BinnedWE(_floor_bin, target_per_bin=4).resample([ws[1]], 0, IterKey(0, "r", 0))
    assert len(split) == 4
    assert all(c.stop_tail == tail_b and c.parent is None and c.recycled_from == sink for c in split)


def test_resample_rejects_iteration_mismatch():
    walkers = _walkers([(1.0, 0.5, LA)], it=2)
    with pytest.raises(ValueError):
        BinnedWE(_floor_bin, 2).resample(walkers, 2, IterKey(0, "r", 3))
    with pytest.raises(ValueError):
        BinnedWE(_floor_bin, 2).resample(walkers, 1, IterKey(0, "r", 1))


# ---------------------------------------------------------------- 10.3


def _snapshot(ws):
    return [
        (w.segment_key, w.parent, w.origin_label, w.weight.hex(), w.z.tobytes(), w.state.x.tobytes())
        for w in ws
    ]


def test_10_3_resample_bitwise_reproducible_and_keyed():
    rng = derive_rng(IterKey(0, "t103", 0), "test-setup")
    specs = [(rng.uniform(0.01, 1.0), rng.uniform(0, 3), [LA, LB][i % 2]) for i in range(80)]
    we = BinnedWE(_floor_bin, target_per_bin=2)
    walkers = _walkers(specs, run_id="t103", it=5, seed=11)
    key = IterKey(11, "t103", 5)
    a = _snapshot(we.resample(walkers, 5, key))
    b = _snapshot(BinnedWE(_floor_bin, target_per_bin=2).resample(walkers, 5, key))
    c = _snapshot(we.resample(list(reversed(walkers)), 5, key))
    assert a == b == c
    walkers12 = _walkers(specs, run_id="t103", it=5, seed=12)
    d = _snapshot(we.resample(walkers12, 5, IterKey(12, "t103", 5)))
    # survivor choice really comes from the keyed RNG (compared without keys)
    assert [s[2:] for s in a] != [s[2:] for s in d]


# ---------------------------------------------------------------- 10.6


def test_10_6_n_eff_formula():
    assert n_eff(np.array([0.5, 0.25, 0.25])) == pytest.approx(1.0 / 0.375, rel=1e-15)
    assert n_eff(np.array([2.0, 1.0, 1.0])) == pytest.approx(1.0 / 0.375, rel=1e-15)
    assert n_eff(np.full(7, 3.0)) == pytest.approx(7.0, rel=1e-15)
    assert n_eff(np.array([1.0, 1e-200])) == pytest.approx(1.0, rel=1e-15)


# ---------------------------------------------------------------- 10.7


def test_10_7_two_hundred_rounds_of_forced_split():
    # Round r: the lightest walker is put alone in bin 1 (so it is split
    # into 32 pieces, a factor 32 per round: 2^-5 per round reaches the
    # 1e-250 floor well inside 200 rounds); everyone else is packed into
    # bin 0 and merged back down. Initial set also holds a subnormal and a
    # sub-floor weight that must be force-merged away.
    min_w = 1e-250
    specs = [(1.0 - 2e-300, 0.5, LA), (1e-300, 0.5, LA), (5e-324, 0.5, LA), (1e-300, 0.5, LA)]
    walkers = _walkers(specs, run_id="t107", seed=3)
    total0 = math.fsum(w.weight for w in walkers)
    we = BinnedWE(_floor_bin, target_per_bin=32, min_weight=min_w)
    smallest_seen = 1.0
    for it in range(200):
        out = we.resample(walkers, it, IterKey(3, "t107", it))
        ws = np.array([w.weight for w in out])
        assert np.all(np.isfinite(ws))
        assert np.all(ws >= min_w)
        assert _rel_err(math.fsum(ws), total0) < 1e-12
        _check_ledger_exact(we, walkers, out)
        smallest_seen = min(smallest_seen, float(ws.min()))
        lightest = min(range(len(out)), key=lambda i: out[i].weight)
        walkers = [
            dataclasses.replace(w, z=np.array([1.5 if i == lightest else 0.5]))
            for i, w in enumerate(out)
        ]
    assert smallest_seen < 1e-240  # the floor was actually reached
    assert we.n_underweight_alone == 0


# ------------------------------------------------------------ segments

KT = 1.0


def _dw_backend(barrier=8.0, dt=1e-3):
    return _Analytic(DoubleWell1D(barrier), "overdamped", dt, KT, gamma=1.0, mass=1.0)


def _z(state):
    return np.asarray(state.x, dtype=float)


def _never():
    return spec_region("never", lambda o: False, "never")


def _ge(name, x):
    return spec_region(name, lambda o: o["z0"] >= x, {"z0_ge": x})


def _le(name, x):
    return spec_region(name, lambda o: o["z0"] <= x, {"z0_le": x})


def test_run_segment_fixed_lag_record(tmp_path):
    store = Store(tmp_path / "s.db")
    w = _walker("seg", 0, 0, 0.5, -1.0, LA)
    w.parent = SegmentKey(0, "seg", -1, 9)  # arbitrary parent, just to check the digest link
    out, dec = run_segment(w, _dw_backend(), 0.1, FixedLag(0.1), _z, store, w.segment_key, dt_obs=0.01)
    assert dec is not None and dec.reason == "fixed_lag"
    rec = store.get(key_digest(w.segment_key))
    assert rec.kind == "segment"
    assert rec.key == {"global_seed": 0, "run_id": "seg", "iteration": 0, "walker_id": 0}
    assert rec.origin_label == LA and rec.weight == 0.5
    assert rec.parent_digest == key_digest(w.parent)
    assert rec.stop_reason == "fixed_lag"
    # R35 #4: the series starts with the post-build state at t = 0; the
    # clock is step_index * dt (K1).
    assert len(rec.observables["t"]) == 11 and len(rec.observables["z0"]) == 11
    assert rec.observables["t"] == [k * 10 * 1e-3 for k in range(11)]
    assert rec.observables["z0"][0] == -1.0
    assert rec.observables["z0"][-1] == out.z[0] == out.state.x[0]
    assert out.segment_key == w.segment_key and out.weight == 0.5
    # K6 / K8 / R39 / provenance fields
    assert rec.observables_thinned is False
    assert rec.code_version == code_version()
    assert rec.protocol_hash is not None
    assert rec.ic_meta["kind"] == "we_segment" and rec.ic_meta["t_start"] == 0.0
    assert rec.ic_meta["recycled_from"] is None and rec.ic_meta["n_carried"] == 0


def test_segment_protocol_hash_tracks_the_stop_rule(tmp_path):
    w = _walker("ph", 0, 0, 1.0, -1.0, LA)
    hashes = []
    for i, tp in enumerate((0.0, 0.0, 0.05)):
        store = Store(tmp_path / f"{i}.db")
        run_segment(w, _dw_backend(), 0.1, _sink_rule(1.0, 0.1, tp), _z, store, w.segment_key, 0.01)
        hashes.append(store.get(key_digest(w.segment_key)).protocol_hash)
    assert hashes[0] == hashes[1] != hashes[2]


def test_run_segment_no_decision_is_recorded_as_fixed_lag(tmp_path):
    store = Store(tmp_path / "s.db")
    w = _walker("seg", 0, 0, 1.0, -1.0, LA)
    stop = AbsorbingAB(_never(), _never(), tau_persist=0.0, t_max=10.0)
    _, dec = run_segment(w, _dw_backend(), 0.05, stop, _z, store, w.segment_key, dt_obs=0.01)
    assert dec is None
    assert store.get(key_digest(w.segment_key)).stop_reason == "fixed_lag"


def test_run_segment_stops_early_on_event(tmp_path):
    store = Store(tmp_path / "s.db")
    w = _walker("seg", 0, 0, 1.0, 0.0, LA)  # barrier top: leaves |x|<0.02 fast
    stop = AbsorbingAB(_le("A", -0.02), _ge("B", 0.02), 0.0, 10.0)
    out, dec = run_segment(w, _dw_backend(), 5.0, stop, _z, store, w.segment_key, dt_obs=0.001)
    assert dec.reason in ("A", "B")
    rec = store.get(key_digest(w.segment_key))
    assert rec.stop_reason == dec.reason == rec.final_state_label
    assert rec.event_time == dec.event_time < 5.0
    assert len(rec.observables["t"]) < 5000


def test_run_segment_validates_dt_obs_and_key(tmp_path):
    store = Store(tmp_path / "s.db")
    w = _walker("seg", 0, 0, 1.0, -1.0, LA)
    with pytest.raises(ValueError):
        run_segment(w, _dw_backend(), 0.1, FixedLag(0.1), _z, store, w.segment_key, dt_obs=0.0015)
    with pytest.raises(ValueError):
        run_segment(w, _dw_backend(), 0.1, FixedLag(0.1), _z, store, w.segment_key, dt_obs=0.03)
    with pytest.raises(ValueError):
        run_segment(w, _dw_backend(), 0.1, FixedLag(0.1), _z, store, SegmentKey(0, "seg", 0, 1), dt_obs=0.01)
    assert list(store.iter()) == []


def test_segment_noise_independent_of_execution_order(tmp_path):
    ws = [_walker("ord", 2, i, 0.25, -1.0, LA) for i in range(4)]
    backend, stop = _dw_backend(), FixedLag(0.05)

    def run(order, name):
        store = Store(tmp_path / f"{name}.db")
        res = {}
        for i in order:
            out, _ = run_segment(ws[i], backend, 0.05, stop, _z, store, ws[i].segment_key, 0.01)
            res[i] = out.state.x.tobytes()
        return res

    fwd, rev = run([0, 1, 2, 3], "fwd"), run([3, 1, 0, 2], "rev")
    assert fwd == rev
    assert len(set(fwd.values())) == 4  # distinct walkers get distinct noise


# ------------------------------------------------ G-I4: dt, cfg, K8 hash


class _NoDtBackend:
    """Wraps a backend but exposes no `dt`: the step must come from the
    propagator (ruling R34)."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, inner):
        self._inner = inner
        self.built_with = []

    def build(self, s, cfg, key):
        self.built_with.append(cfg)
        return self._inner.build(s, cfg, key)

    def energy_forces(self, x, box=None):
        return self._inner.energy_forces(x, box)

    def effective_config(self, cfg=None):
        return self._inner.effective_config(cfg)

    def provenance(self, cfg=None):
        return {"wrapped": self._inner.provenance(cfg), "cfg_seen": cfg is not None}


def _analytic_cfg(purpose="measurement"):
    return PhysicsConfig(
        integrator="langevin_middle", dt_ps=1e-3, temperature_K=300.0, friction_per_ps=0.1,
        constraints="none", rigid_water=False, platform="Reference", precision="double",
        deterministic_forces=False, purpose=purpose,
    )


def test_run_segment_reads_dt_from_propagator_not_backend(tmp_path):
    backend = _NoDtBackend(_dw_backend(dt=1e-3))
    assert not hasattr(backend, "dt")
    w = _walker("nodt", 0, 0, 1.0, -1.0, LA)
    store = Store(tmp_path / "s.db")
    out, dec = run_segment(w, backend, 0.05, FixedLag(0.05), _z, store, w.segment_key, dt_obs=0.01)
    assert dec.reason == "fixed_lag"
    assert store.get(key_digest(w.segment_key)).observables["t"][-1] == pytest.approx(0.05)
    with pytest.raises(ValueError):  # 0.0015 is not a multiple of the propagator's 1e-3
        run_segment(w, backend, 0.003, FixedLag(0.003), _z, Store(tmp_path / "t.db"), w.segment_key,
                    dt_obs=0.0015)


def test_run_segment_hashes_the_cfg_it_builds_with(tmp_path):
    # K8: physics_cfg is passed to build; K9: the hash is of
    # effective_config(cfg), for None and an explicit cfg alike.
    backend = _NoDtBackend(_dw_backend())
    cfg = _analytic_cfg()
    w = _walker("cfg", 0, 0, 1.0, -1.0, LA)
    s1, s2 = Store(tmp_path / "a.db"), Store(tmp_path / "b.db")
    run_segment(w, backend, 0.02, FixedLag(0.02), _z, s1, w.segment_key, 0.01, physics_cfg=cfg)
    assert backend.built_with[-1] is cfg
    rec = s1.get(key_digest(w.segment_key))
    assert rec.physics_config_hash == config_hash(backend.effective_config(cfg))
    assert rec.backend_provenance["cfg_seen"] is True
    run_segment(w, backend, 0.02, FixedLag(0.02), _z, s2, w.segment_key, 0.01)
    rec = s2.get(key_digest(w.segment_key))
    assert backend.built_with[-1] is None
    assert rec.physics_config_hash == config_hash(backend.effective_config(None))


def test_run_we_rejects_equilibration_cfg_before_running(tmp_path):
    store = Store(tmp_path / "s.db")
    with pytest.raises(ValueError, match="measurement"):
        run_we(
            _init("eq", 2, -1.0, LA), _dw_backend(), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z,
            2, 0.02, store, 0, "eq", None, dt_obs=0.01, physics_cfg=_analytic_cfg("equilibration"),
        )
    assert list(store.iter()) == []


# ---------------------------------------------------------------- run_we


def _sink_rule(b, tau_seg, tau_persist=0.0):
    return AbsorbingAB(_never(), _ge("sink", b), tau_persist=tau_persist, t_max=2.0 * tau_seg + tau_persist)


def _edges_bin(edges):
    edges = np.asarray(edges, dtype=float)
    return SpecPredicate(
        lambda z: int(np.searchsorted(edges, z[0], side="right")), {"edges": edges.tolist()}
    )


def _init(run_id, n, x, label, seed=0):
    return [_walker(run_id, 0, i, 1.0 / n, x, label, seed=seed) for i in range(n)]


def _src(x=-1.0, label=LA):
    return _walker("src", 0, 0, 1.0, x, label)


def test_recycling_conserves_weight_and_relabels(tmp_path):
    # Low barrier so the sink is hit within a few iterations.
    store = Store(tmp_path / "s.db")
    backend = _dw_backend(barrier=1.0, dt=2e-3)
    tau = 0.1
    res = run_we(
        _init("rec", 4, -1.0, LA, seed=5), backend,
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=2),
        _sink_rule(1.0, tau), _z, 25, tau, store, 5, "rec", [_src(label=(9, 9))], dt_obs=0.01,
    )
    assert isinstance(res, WERun) and res.run_id == "rec"
    assert np.all(np.abs(res.weights_sum - 1.0) < 1e-12)
    assert res.flux_to_sink.sum() > 0
    assert np.array_equal(res.absorbed["B"], res.flux_to_sink)
    assert res.final_weight == pytest.approx(1.0, rel=1e-12)
    recs = list(store.iter(kind="segment"))
    # flux_to_sink[it] is exactly the weight of iteration-it segments ending in B.
    for it in range(25):
        w_b = math.fsum(r.weight for r in recs if r.key["iteration"] == it and r.stop_reason == "B")
        assert res.flux_to_sink[it] == pytest.approx(w_b, rel=1e-12, abs=0.0)
    # Recycled walkers carry the recycle target's label from then on.
    assert any(r.origin_label == (9, 9) for r in recs)
    first_b = min(r.key["iteration"] for r in recs if r.stop_reason == "B")
    assert all(r.origin_label == LA for r in recs if r.key["iteration"] <= first_b)
    # G-I6: timing instrumentation
    assert res.build_seconds.shape == res.propagate_seconds.shape == (25,)
    assert np.all(res.build_seconds > 0) and np.all(res.propagate_seconds > 0)


def test_recycled_walkers_are_marked_and_not_linked_to_the_sink(tmp_path):
    # G-I3: a recycled walker's record has parent_digest None and
    # ic_meta["recycled_from"] = the sink segment; following parent_digest
    # never crosses a sink -> source jump.
    store = Store(tmp_path / "s.db")
    tau = 0.1
    run_we(
        _init("rl", 4, -1.0, LA, seed=5), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=2),
        _sink_rule(1.0, tau), _z, 25, tau, store, 5, "rl", [_src(-1.0)], dt_obs=0.01,
    )
    recs = {r.key_digest: r for r in store.iter(kind="segment")}
    sinks = {d for d, r in recs.items() if r.stop_reason == "B"}
    recycled = [r for r in recs.values() if r.ic_meta["recycled_from"] is not None]
    assert sinks and recycled
    assert not any(r.parent_digest in sinks for r in recs.values())
    for r in recycled:
        assert r.parent_digest is None
        assert r.ic_meta["recycled_from"] in sinks
        assert r.ic_meta["recycle_target"] == 0
        assert r.ic_meta["n_carried"] == 0
        assert r.observables["z0"][0] == -1.0  # starts at the source, not at the sink
    # (a recycled walker may be merged into another walker of its bin by the
    # following resample: its weight lives on in the survivor, its marker
    # with its trajectory -- so not every sink segment has a marked child)


def test_recycled_z_is_computed_from_target_state(tmp_path):
    # G-M6: a target whose `z` disagrees with its state is binned (and
    # recorded) by z_fn(state), not by its stale `z`.
    store = Store(tmp_path / "s.db")
    tgt = dataclasses.replace(_src(-1.0), z=np.array([5.0]))
    res = run_we(
        _init("mz", 2, 0.95, LA), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=1),
        _sink_rule(1.0, 0.1), _z, 1, 0.1, store, 0, "mz", [tgt], dt_obs=0.01,
    )
    recycled = [w for w in res.final_walkers if w.recycled_from is not None]
    assert recycled, "no walker reached the sink from x=0.95 in one segment"
    assert all(w.z.tolist() == [-1.0] for w in recycled)


def test_run_we_records_stats_and_lineage(tmp_path):
    store = Store(tmp_path / "s.db")
    tau = 0.05
    res = run_we(
        _init("lin", 4, -1.0, LA, seed=1), _dw_backend(),
        BinnedWE(_edges_bin([-1.05, -1.0, -0.95]), target_per_bin=2),
        FixedLag(tau), _z, 6, tau, store, 1, "lin", None, dt_obs=0.01,
    )
    recs = list(store.iter(kind="segment"))
    digests = {key_digest(SegmentKey(**r.key)): r for r in recs}
    proto = {r.protocol_hash for r in recs}
    we_proto = {r.ic_meta["we_protocol_hash"] for r in recs}
    assert len(proto) == 1 and len(we_proto) == 1 and None not in proto
    for it in range(6):
        rs = [r for r in recs if r.key["iteration"] == it]
        ws = np.array([r.weight for r in rs])
        assert res.n_walkers[it] == len(rs)
        assert res.weights_sum[it] == pytest.approx(math.fsum(ws), rel=1e-15)
        assert res.n_eff[it] == pytest.approx(1.0 / np.sum((ws / ws.sum()) ** 2), rel=1e-12)
        assert sorted(r.key["walker_id"] for r in rs) == list(range(len(rs)))
        for r in rs:
            assert r.ic_meta["t_start"] == pytest.approx(it * tau)
            if it == 0:
                assert r.parent_digest is None
            else:
                parent = digests[r.parent_digest]
                assert parent.key["iteration"] == it - 1
                # the child's first own row is the parent's last state
                own = [i for i, t in enumerate(r.observables["t"]) if t >= 0]
                assert r.observables["z0"][own[0]] == parent.observables["z0"][-1]
    assert np.all(res.flux_to_sink == 0.0)
    assert len(res.final_walkers) == res.n_walkers[-1]


def test_run_we_bitwise_reproducible(tmp_path):
    def go(name, run_id, seed=2):
        store = Store(tmp_path / f"{name}.db")
        tau = 0.05
        res = run_we(
            _init(run_id, 4, -0.5, LA, seed=seed), _dw_backend(barrier=1.0),
            BinnedWE(_edges_bin(np.linspace(-1, 1, 9)), target_per_bin=2),
            _sink_rule(0.5, tau), _z, 8, tau, store, seed, run_id,
            [_src(-0.5)], dt_obs=0.01,
        )
        xs = [tuple(r.observables["z0"]) for r in store.iter(kind="segment")]
        return res, xs

    (a, xa), (b, xb), (c, xc) = go("a", "rep"), go("b", "rep"), go("c", "other")
    for f in ("flux_to_sink", "n_eff", "n_walkers", "weights_sum"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert xa == xb and xa != xc


def test_replicas_differing_only_in_global_seed_share_no_noise(tmp_path):
    # G-C1: same run_id, different global_seed -> independent segment noise
    # (the iteration-0 segments start from identical states, so any shared
    # noise would make them bitwise identical).
    def go(seed, name):
        store = Store(tmp_path / f"{name}.db")
        run_we(
            _init("rep", 4, -1.0, LA, seed=seed), _dw_backend(barrier=1.0),
            BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), 2), FixedLag(0.1), _z, 3, 0.1,
            store, seed, "rep", None, dt_obs=0.01,
        )
        return {
            (r.key["iteration"], r.key["walker_id"]): r.observables["z0"]
            for r in store.iter(kind="segment")
        }

    a, b, a2 = go(1, "a"), go(2, "b"), go(1, "a2")
    assert a == a2
    it0 = [k for k in a if k[0] == 0]
    assert len(it0) == 4
    assert all(a[k][1:] != b[k][1:] for k in it0)
    common = set(a) & set(b)
    assert sum(a[k] == b[k] for k in common) == 0


def test_run_we_absorption_removes_weight_without_recycling(tmp_path):
    store = Store(tmp_path / "s.db")
    tau = 0.1
    res = run_we(
        _init("abs", 4, 0.0, LA, seed=3), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=2),
        _sink_rule(0.5, tau), _z, 10, tau, store, 3, "abs", None, dt_obs=0.01,
    )
    for it in range(9):
        assert res.weights_sum[it + 1] == pytest.approx(
            res.weights_sum[it] - res.flux_to_sink[it], rel=1e-12, abs=1e-15
        )
    assert res.flux_to_sink.sum() > 0
    assert res.final_weight == pytest.approx(1.0 - res.flux_to_sink.sum(), rel=1e-12, abs=1e-15)


def test_run_we_reports_non_sink_absorbed_weight(tmp_path):
    # G-M1: weight absorbed in A is reported, and sum(w) bookkeeping closes.
    store = Store(tmp_path / "s.db")
    res = run_we(
        _init("ab", 4, 0.0, LA, seed=3), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), 2),
        AbsorbingAB(_le("A", -0.5), _ge("B", 0.5), 0.0, 1.0), _z, 10, 0.1, store, 3, "ab", None,
        dt_obs=0.01,
    )
    recs = list(store.iter(kind="segment"))
    for reason in ("A", "B"):
        w = math.fsum(r.weight for r in recs if r.stop_reason == reason)
        assert math.fsum(res.absorbed[reason]) == pytest.approx(w, rel=1e-12)
    assert res.absorbed["A"].sum() > 0
    removed = res.absorbed["A"] + res.absorbed["B"]
    for it in range(9):
        assert res.weights_sum[it + 1] == pytest.approx(res.weights_sum[it] - removed[it], abs=1e-15)
    assert res.final_weight + removed.sum() == pytest.approx(1.0, rel=1e-12)


def test_run_we_steady_state_rejects_non_sink_terminal(tmp_path):
    store = Store(tmp_path / "s.db")
    with pytest.raises(ValueError):
        run_we(
            _init("ss", 2, 0.0, LA), _dw_backend(),
            BinnedWE(_floor_bin, 2), AbsorbingAB(_le("A", -0.02), _ge("B", 0.02), 0.0, 1.0), _z, 3,
            0.1, store, 0, "ss", [_src(0.0)], dt_obs=0.001,
        )


def test_run_we_validates_configuration_before_running(tmp_path):
    # G-M4 / G-M5: errors that can be known up front raise before any record.
    store = Store(tmp_path / "s.db")
    be, res = _dw_backend(), BinnedWE(_floor_bin, 2)
    bad = [
        dict(stop=FixedLag(0.05), tau=0.1),  # fires a non-event before tau_seg
        dict(stop=_sink_rule(1.0, 0.02), tau=0.1),  # t_max = 0.04 < tau_seg
        dict(stop=FixedLag(0.1), tau=0.105),  # tau_seg not a multiple of dt_obs
        dict(stop=FixedLag(0.1), tau=0.1, init=_init("v", 2, -1.0, LA)[:1]),  # weights sum 0.5
        dict(stop=FixedLag(0.1), tau=0.1, init=[_walker("v", 1, 0, 1.0, -1.0, LA)]),  # iteration 1
        dict(stop=FixedLag(0.1), tau=0.1, init=[_walker("v", 0, 0, 1.0, -1.0, LA, seed=1)]),  # seed
    ]
    for case in bad:
        with pytest.raises(ValueError):
            run_we(
                case.get("init", _init("v", 2, -1.0, LA)), be, res, case["stop"], _z, 2, case["tau"],
                store, 0, "v", None, dt_obs=0.01,
            )
    # an undescribable binning function or region fails loudly too (R39;
    # ProtocolDescriptionError is a TypeError -- after the P3 merge the
    # region check raises observe.events' class of that name)
    with pytest.raises(ProtocolDescriptionError):
        run_we(_init("v", 2, -1.0, LA), be, BinnedWE(_floor_bin_fn, 2), FixedLag(0.1), _z, 2, 0.1,
               store, 0, "v", None, dt_obs=0.01)
    with pytest.raises(TypeError, match="spec"):
        run_we(_init("v", 2, -1.0, LA), be, res,
               AbsorbingAB(Region("A", lambda o: False), _never(), 0.0, 1.0), _z, 2, 0.1,
               store, 0, "v", None, dt_obs=0.01)
    assert list(store.iter()) == []


def test_run_we_continuation_reproduces_uninterrupted_run(tmp_path):
    # G-M5: final_walkers + start_iteration extend a run exactly (including
    # the carried stop-rule state and recycling).
    def args(store):
        return dict(
            backend=_dw_backend(barrier=1.0, dt=2e-3),
            stop=_sink_rule(0.5, 0.1, tau_persist=0.15), z_fn=_z, tau_seg=0.1, store=store,
            global_seed=4, run_id="cont", recycle_to=[_src(-0.5)], dt_obs=0.01,
        )

    bins = np.linspace(-1, 1, 9)
    s_full, s_part = Store(tmp_path / "full.db"), Store(tmp_path / "part.db")
    full = run_we(_init("cont", 4, -0.5, LA, seed=4), resampler=BinnedWE(_edges_bin(bins), 2),
                  n_iter=10, **args(s_full))
    rs = BinnedWE(_edges_bin(bins), 2)
    p1 = run_we(_init("cont", 4, -0.5, LA, seed=4), resampler=rs, n_iter=4, **args(s_part))
    nxt = rs.resample(p1.final_walkers, 3, IterKey(4, "cont", 3))
    p2 = run_we(nxt, resampler=rs, n_iter=6, start_iteration=4, **args(s_part))

    def view(store):
        return {r.key_digest: (r.observables, r.stop_reason, r.event_time, r.weight,
                               r.parent_digest, r.ic_meta) for r in store.iter()}

    assert view(s_full) == view(s_part)
    assert np.array_equal(full.flux_to_sink, np.r_[p1.flux_to_sink, p2.flux_to_sink])
    assert full.flux_to_sink.sum() > 0


# ------------------------------------------- G-I2: persistence across segments


def test_persistence_longer_than_a_segment_still_fires(tmp_path):
    # Reviewer probe P3: tau_persist = 0.2 > tau_seg = 0.1, walker sitting
    # in B (deep right well). With the stop-rule state carried, B fires in
    # the second segment with its entry at the first segment's start.
    store = Store(tmp_path / "s.db")
    stop = AbsorbingAB(_never(), _ge("B", 0.0), tau_persist=0.2, t_max=1.0)
    res = run_we(
        [_walker("per", 0, 0, 1.0, 1.0, LA)], _dw_backend(barrier=8.0), BinnedWE(_floor_bin, 1),
        stop, _z, 4, 0.1, store, 0, "per", None, dt_obs=0.01,
    )
    recs = sorted(store.iter(kind="segment"), key=lambda r: r.key["iteration"])
    assert [r.stop_reason for r in recs] == ["fixed_lag", "B"]
    b = recs[1]
    assert b.event_time == pytest.approx(-0.1)  # segment-local: entered in segment 0
    assert b.ic_meta["t_start"] + b.event_time == pytest.approx(0.0)  # WE clock
    assert b.ic_meta["n_carried"] == 10 and b.ic_meta["stop_state_carried"] is True
    assert res.flux_to_sink[1] == 1.0


@pytest.mark.parametrize("tau_persist", [0.25, 0.255])
def test_we_persistence_matches_one_uninterrupted_trajectory(tmp_path, tau_persist):
    # One walker, one bin, target 1: the WE run is a single trajectory cut
    # into segments. Its event (reason and WE-clock time) must equal
    # offline_replay of the concatenated series with a fresh rule -- the
    # run_shot event definition -- for a tau_persist spanning 2.5 segments,
    # also when it is not a multiple of dt_obs (p7 M-1: the oldest kept tail
    # row then matters).
    store = Store(tmp_path / "s.db")
    tau, dt_obs = 0.1, 0.01

    def rule():
        return AbsorbingAB(_le("A", -1.2), _ge("B", 0.4), tau_persist=tau_persist, t_max=1e6)

    res = run_we(
        [_walker("one", 0, 0, 1.0, -1.0, LA)], _dw_backend(barrier=1.0, dt=1e-3),
        BinnedWE(_floor_bin, 1), rule(), _z, 400, tau, store, 0, "one", None, dt_obs=dt_obs,
    )
    recs = sorted(store.iter(kind="segment"), key=lambda r: r.key["iteration"])
    assert recs[-1].stop_reason in ("A", "B"), "no event in 400 segments; lengthen the run"
    ts, zs = [], []
    for r in recs:
        own = [i for i, t in enumerate(r.observables["t"]) if t >= 0]
        if r.key["iteration"] > 0:
            own = own[1:]  # row k=0 re-observes the parent's last state
        ts += [r.ic_meta["t_start"] + r.observables["t"][i] for i in own]
        zs += [r.observables["z0"][i] for i in own]
    # p7 I-2: the public lineage helper gives exactly this series
    lin = lineage_series({r.key_digest: r for r in recs}, recs[-1].key_digest)
    assert lin["t"] == pytest.approx(ts, abs=1e-12) and lin["z0"] == zs
    dec = offline_replay(rule(), {"t": np.array(ts), "z0": np.array(zs)})
    last = recs[-1]
    assert dec is not None and dec.reason == last.stop_reason
    assert dec.event_time == pytest.approx(last.ic_meta["t_start"] + last.event_time, abs=1e-9)
    assert last.event_time < 0  # tau_persist > tau_seg: the entry lies in an earlier segment
    # and every single record replays to its own decision
    for r in recs:
        d = offline_replay(rule(), {k: np.array(v) for k, v in r.observables.items()})
        assert (d.reason if d else "fixed_lag") == r.stop_reason
    assert res.flux_to_sink.sum() == (1.0 if last.stop_reason == "B" else 0.0)


# ------------------------------------------- G-I5: deterministic dynamics


def test_run_we_rejects_deterministic_dynamics_unless_opted_in(tmp_path):
    kw = dict(z_fn=_z, n_iter=2, tau_seg=0.02, recycle_to=None, dt_obs=0.01)
    det = [
        _Analytic(DoubleWell1D(1.0), "baoab", 1e-3, KT, gamma=0.0),
        _Analytic(DoubleWell1D(1.0), "overdamped", 1e-3, 0.0, gamma=1.0),
    ]
    for i, be in enumerate(det):
        with pytest.raises(ValueError, match="deterministic"):
            run_we(_init("d", 2, -1.0, LA), be, BinnedWE(_floor_bin, 2), FixedLag(0.02),
                   store=Store(tmp_path / f"d{i}.db"), global_seed=0, run_id="d", **kw)
        res = run_we(_init("d", 2, -1.0, LA), be, BinnedWE(_floor_bin, 2), FixedLag(0.02),
                     store=Store(tmp_path / f"o{i}.db"), global_seed=0, run_id="d",
                     allow_deterministic=True, **kw)
        assert res.n_walkers[0] == 2
    # stochastic BAOAB passes
    run_we(_init("d", 2, -1.0, LA), _Analytic(DoubleWell1D(1.0), "baoab", 1e-3, KT, gamma=0.1),
           BinnedWE(_floor_bin, 2), FixedLag(0.02), store=Store(tmp_path / "s.db"),
           global_seed=0, run_id="d", **kw)


class _OpaqueBackend(_NoDtBackend):
    """Says nothing about its dynamics: run_we must probe them."""

    def effective_config(self, cfg=None):
        return {"opaque": True}


def test_deterministic_probe_for_undeclared_backends(tmp_path):
    kw = dict(z_fn=_z, n_iter=1, tau_seg=0.02, recycle_to=None, dt_obs=0.01)
    det = _OpaqueBackend(_Analytic(DoubleWell1D(1.0), "baoab", 1e-3, KT, gamma=0.0))
    with pytest.raises(ValueError, match="deterministic"):
        run_we(_init("p", 2, -1.0, LA), det, BinnedWE(_floor_bin, 2), FixedLag(0.02),
               store=Store(tmp_path / "a.db"), global_seed=0, run_id="p", **kw)
    sto = _OpaqueBackend(_dw_backend())
    run_we(_init("p", 2, -1.0, LA), sto, BinnedWE(_floor_bin, 2), FixedLag(0.02),
           store=Store(tmp_path / "b.db"), global_seed=0, run_id="p", **kw)


# ------------------------------------------------------ K4: nonfinite


class _NaNProp:
    def __init__(self, inner, nan_after):
        self._inner, self._left = inner, nan_after

    @property
    def dt(self):
        return self._inner.dt

    def run(self, n):
        self._inner.run(n)
        self._left -= n

    def get_state(self):
        s = self._inner.get_state()
        if self._left <= 0:
            s = dataclasses.replace(s, x=np.full_like(s.x, np.nan))
        return s

    def set_state(self, s):
        self._inner.set_state(s)


class _RaisingNaNProp(_NaNProp):
    """Blows up like OpenMM CPU/CUDA: run() raises instead of returning NaN (K10)."""

    def run(self, n):
        super().run(n)
        if self._left <= 0:
            raise NumericalInstabilityError("Particle coordinate is NaN.")


class _NaNBackend(_NoDtBackend):
    """Walker 1's propagator blows up after 30 steps."""

    prop_cls = _NaNProp

    def build(self, s, cfg, key):
        prop = self._inner.build(s, cfg, key)
        return self.prop_cls(prop, 30) if key.walker_id == 1 else prop


class _RaisingNaNBackend(_NaNBackend):
    prop_cls = _RaisingNaNProp


@pytest.mark.parametrize("backend_cls", [_NaNBackend, _RaisingNaNBackend])
def test_nonfinite_segment_is_recorded_and_removed(backend_cls, tmp_path):
    for i, recycle in enumerate((None, [_src(-1.0)])):
        store = Store(tmp_path / f"n{i}.db")
        res = run_we(
            _init("nan", 2, -1.0, LA), backend_cls(_dw_backend()), BinnedWE(_floor_bin, 1),
            _sink_rule(1.0, 0.1), _z, 2, 0.1, store, 0, "nan", recycle, dt_obs=0.01,
        )
        bad = [r for r in store.iter(kind="segment") if r.stop_reason == "nonfinite"]
        assert len(bad) == 1 and bad[0].key["iteration"] == 0
        assert bad[0].final_state_label is None and bad[0].event_time == pytest.approx(0.03)
        raised = backend_cls is _RaisingNaNBackend
        assert any("NumericalInstabilityError" in w for w in bad[0].warnings) is raised
        assert math.isnan(bad[0].observables["z0"][-1])
        assert res.n_nonfinite == 1 and not res.valid
        assert res.absorbed["nonfinite"][0] == 0.5
        assert res.weights_sum[1] == pytest.approx(0.5)


class _RuleSaysNonfinite(FixedLag):
    """A (P3-style) rule that itself returns the K4 reason string."""

    def update(self, obs, t):
        if obs["z0"] > 0.5:
            return StopDecision(reason="nonfinite", event_time=t)
        return super().update(obs, t)


def test_nonfinite_reason_from_the_rule_is_handled(tmp_path):
    store = Store(tmp_path / "s.db")
    init = [_walker("rn", 0, 0, 0.5, -1.0, LA), _walker("rn", 0, 1, 0.5, 1.0, LA)]
    res = run_we(init, _dw_backend(), BinnedWE(_floor_bin, 1), _RuleSaysNonfinite(0.1), _z, 2, 0.1,
                 store, 0, "rn", None, dt_obs=0.01)
    assert res.n_nonfinite == 1 and res.absorbed["nonfinite"][0] == 0.5


# ---------------------------------------------- G-I4: OpenMM Reference smoke


def _omm_backend(integrator="langevin_middle"):
    system = openmm.System()
    system.addParticle(1.0)
    f = openmm.CustomExternalForce("8*(x^2-1)^2 + 50*(y^2+z^2)")
    f.addParticle(0, [])
    system.addForce(f)
    cfg = PhysicsConfig(
        integrator=integrator, dt_ps=0.001, temperature_K=120.0, friction_per_ps=0.1,
        constraints="none", rigid_water=False, platform="Reference", precision="double",
        deterministic_forces=False, purpose="measurement",
    )
    return _OpenMM(system, None, cfg), cfg


def _omm_init(run_id, n, seed=0, v=(0.0, 0.0, 0.0)):
    st = MDState(x=np.array([[-1.0, 0.0, 0.0]]), v=np.array([v]), t=0.0)
    return [
        Walker(SegmentKey(seed, run_id, 0, i), None, LA, 1.0 / n, np.array([-1.0]), st)
        for i in range(n)
    ]


def _omm_z(state):
    return np.array([state.x[0, 0]])


def test_openmm_reference_run_we_smoke(tmp_path):
    backend, cfg = _omm_backend()
    store = Store(tmp_path / "s.db")
    res = run_we(
        _omm_init("omm", 2), backend, BinnedWE(_edges_bin([-1.1, -1.0, -0.9]), 2),
        FixedLag(0.01), _omm_z, 3, 0.01, store, 0, "omm", None, dt_obs=0.002, physics_cfg=cfg,
    )
    recs = list(store.iter(kind="segment"))
    assert len(recs) == res.n_walkers.sum() and np.all(np.abs(res.weights_sum - 1) < 1e-12)
    assert all(r.physics_config_hash == config_hash(backend.effective_config(cfg)) for r in recs)
    assert all(len(r.observables["t"]) == 6 for r in recs)
    assert recs[0].observables["t"][-1] == pytest.approx(0.01)
    assert len({tuple(r.observables["z0"]) for r in recs}) == len(recs)  # stochastic
    # with cfg=None the effective (constructor) cfg is hashed -- the same hash (K9)
    s2 = Store(tmp_path / "t.db")
    w = _omm_init("omm2", 1)[0]
    run_segment(w, backend, 0.01, FixedLag(0.01), _omm_z, s2, w.segment_key, dt_obs=0.002)
    assert s2.get(key_digest(w.segment_key)).physics_config_hash == config_hash(
        backend.effective_config(None)
    ) == recs[0].physics_config_hash


def test_openmm_verlet_we_is_rejected_and_clones_when_forced(tmp_path):
    backend, _ = _omm_backend("verlet")
    with pytest.raises(ValueError, match="deterministic"):
        run_we(_omm_init("nve", 1), backend, BinnedWE(_edges_bin([0.0]), 2), FixedLag(0.004),
               _omm_z, 2, 0.004, Store(tmp_path / "a.db"), 0, "nve", None, dt_obs=0.002)
    store = Store(tmp_path / "b.db")
    run_we(_omm_init("nve", 1, v=(0.5, 0.1, 0.0)), backend, BinnedWE(_edges_bin([0.0]), 2),
           FixedLag(0.004), _omm_z, 2, 0.004, store, 0, "nve", None, dt_obs=0.002,
           allow_deterministic=True)
    it1 = [r.observables["z0"] for r in store.iter(kind="segment") if r.key["iteration"] == 1]
    assert len(it1) == 2 and it1[0] == it1[1]  # the documented degeneracy


# ------------------------------------------------------------ references


def test_reference_mfpt_free_diffusion_closed_form():
    # V = 0 with reflecting wall at x_lo: tau = ((b-x_lo)^2 - (x0-x_lo)^2) / (2D)
    tau = mfpt_1d(lambda x: 0.0 * x, 0.3, 2.0, D=0.5, beta=1.0, x_lo=-1.0)
    assert tau == pytest.approx((3.0**2 - 1.3**2) / (2 * 0.5), rel=1e-9)


def test_reference_committor_free_diffusion_closed_form():
    assert committor_1d(lambda x: 0.0 * x, 0.25, -1.0, 1.0, beta=1.0) == pytest.approx(0.625, rel=1e-9)


# ---------------------------------------------------------------- 10.4

T_975 = {4: 2.7764451051977987}  # Student-t 97.5% quantile, 4 dof (5 replicas)


@pytest.mark.slow
def test_10_4_steady_state_flux_matches_exact_mfpt(tmp_path):
    # Source = point x0 = -1 (left minimum), sink = {x >= 1} (right minimum;
    # at a minimum, discrete-observation overshoot barely moves the MFPT).
    # Bins span the whole source well (a single heavy source bin makes the
    # flux estimate burst-dominated), 0.05 wide up to just past the barrier;
    # tau_seg = 0.01 ~ the intra-well relaxation time 1/V''(-1) = 1/64.
    # Euler-Maruyama at dt*V''(-1) = 0.032 shifts the rate by ~ +2-3%.
    barrier, dt, tau, n_iter, burn = 8.0, 5e-4, 0.01, 400, 50
    D = KT / 1.0  # m = 1, gamma = 1
    V = lambda x: barrier * (x**2 - 1.0) ** 2
    tau_exact = mfpt_1d(V, -1.0, 1.0, D=D, beta=1.0 / KT, x_lo=-3.0)
    backend = _dw_backend(barrier=barrier, dt=dt)
    bin_of = _edges_bin(np.r_[-1.1, -1.0, np.arange(-0.9, 0.1001, 0.05), 0.3])
    rates = []
    for rep in range(5):
        run_id = f"mfpt-{rep}"
        store = Store(tmp_path / f"{run_id}.db")
        res = run_we(
            _init(run_id, 4, -1.0, LA, seed=100 + rep), backend, BinnedWE(bin_of, target_per_bin=8),
            _sink_rule(1.0, tau), _z, n_iter, tau, store, 100 + rep, run_id,
            [_src(-1.0)], dt_obs=5 * dt,
        )
        assert np.all(np.abs(res.weights_sum - 1.0) < 1e-12)
        rates.append(res.flux_to_sink[burn:].mean() / tau)  # weight/iteration -> rate
    rates = np.array(rates)
    mean, se = rates.mean(), rates.std(ddof=1) / np.sqrt(len(rates))
    half = T_975[len(rates) - 1] * se
    exact = 1.0 / tau_exact
    print(
        f"\n[10.4] exact flux 1/MFPT = {exact:.6e} (MFPT = {tau_exact:.4f}); "
        f"WE flux = {mean:.6e} +- {half:.3e} (95% t-CI, n=5); "
        f"replicas/exact = {np.array2string(rates / exact, precision=3)}; "
        f"rel.dev = {(mean - exact) / exact:+.2%}, CI half-width = {half / exact:.1%}"
    )
    assert abs(mean - exact) <= half
    assert half / exact < 0.25  # the CI is informative, not trivially wide


# ---------------------------------------------------------------- 10.5


@dataclasses.dataclass(frozen=True)
class _TiltedDW:
    """V(x) = barrier*(x^2-1)^2 + tilt*x (test-only asymmetric double well)."""

    barrier: float
    tilt: float

    def energy_grad(self, x):
        x = np.asarray(x, dtype=float)
        u = x**2 - 1.0
        return float(self.barrier * u[0] ** 2 + self.tilt * x[0]), 4.0 * self.barrier * x * u + self.tilt


@pytest.mark.slow
def test_10_5_per_label_absorption_probabilities(tmp_path):
    # Label LA starts in the left well, LB in the right well; outcomes are
    # absorption in A = {x <= -1} or B = {x >= 1}. Starting points are
    # chosen so that each label's cross-barrier outcome is moderately rare
    # (~8%): WE splitting helps there, and a WE that lost it (p = 0) would
    # fail the RMSE bound. Estimate = mean over 10 independent WE replicas:
    # the outcome is decided within ~6 iterations, so each replica's spread
    # is brute-force-like (sd ~0.03 per label), and 5 replicas leave too
    # little margin under 0.03 for a right-skewed estimator.
    pot = _TiltedDW(barrier=2.0, tilt=0.5)
    V = lambda x: pot.barrier * (x**2 - 1.0) ** 2 + pot.tilt * x
    a, b = -1.0, 1.0
    starts = {LA: -0.5, LB: 0.7}
    dt, tau, n_iter, n_init = 1e-3, 0.05, 30, 32
    backend = _Analytic(pot, "overdamped", dt, KT, gamma=1.0, mass=1.0)
    stop = AbsorbingAB(_le("A", a), _ge("B", b), tau_persist=0.0, t_max=2.0 * tau)
    bin_of = _edges_bin(np.linspace(-0.9, 0.9, 19))
    ref = {lab: committor_1d(V, x0, a, b, beta=1.0 / KT) for lab, x0 in starts.items()}
    per_rep = {LA: [], LB: []}
    worst_left = 0.0
    for rep in range(10):
        run_id = f"lab-{rep}"
        labels = [LA] * n_init + [LB] * n_init
        init = [
            _walker(run_id, 0, i, 0.5 / n_init, starts[lab], lab, seed=9 + rep)
            for i, lab in enumerate(labels)
        ]
        store = Store(tmp_path / f"{run_id}.db")
        run_we(
            init, backend, BinnedWE(bin_of, target_per_bin=4), stop, _z, n_iter, tau,
            store, 9 + rep, run_id, None, dt_obs=dt,
        )
        recs = list(store.iter(kind="segment"))
        for lab in starts:
            wA = math.fsum(r.weight for r in recs if r.origin_label == lab and r.stop_reason == "A")
            wB = math.fsum(r.weight for r in recs if r.origin_label == lab and r.stop_reason == "B")
            per_rep[lab].append(wB / (wA + wB))
            worst_left = max(worst_left, 0.5 - (wA + wB))  # each label starts with weight 0.5
    est = {lab: float(np.mean(v)) for lab, v in per_rep.items()}
    rmse = math.sqrt(sum((est[k] - ref[k]) ** 2 for k in starts) / len(starts))
    print(
        f"\n[10.5] p_B WE (10-replica mean) = {{LA: {est[LA]:.4f}, LB: {est[LB]:.4f}}}, "
        f"committor = {{LA: {ref[LA]:.4f}, LB: {ref[LB]:.4f}}}, RMSE = {rmse:.4f}; "
        f"per-replica LA = {np.round(per_rep[LA], 4).tolist()}, LB = {np.round(per_rep[LB], 4).tolist()}; "
        f"max unabsorbed weight = {worst_left:.1e}"
    )
    assert worst_left < 1e-3
    assert rmse < 0.03


# ---------------------------------------------------------------------------
# Fix wave 2, package L6 (fixreview-p7 I-1, I-2, M-3, M-4; int2-m4, m5, m7)
# ---------------------------------------------------------------------------

from cytherea.exec.batch import ResumeConfigMismatchError  # noqa: E402


def _two_target_run(tmp_path, weights, name="tw", n_iter=40):
    store = Store(tmp_path / f"{name}.db")
    targets = [dataclasses.replace(_src(-1.0, (8, 8)), weight=weights[0]),
               dataclasses.replace(_src(-1.0, (9, 9)), weight=weights[1])]
    res = run_we(
        _init(name, 4, -1.0, LA, seed=5), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=2),
        _sink_rule(1.0, 0.1), _z, n_iter, 0.1, store, 5, name, targets, dt_obs=0.01,
    )
    return res, list(store.iter(kind="segment"))


def test_i1_recycle_targets_are_drawn_by_their_weights(tmp_path):
    """p7 I-1: targets weighted 0.9 / 0.1 got 24 / 22 recycled children."""
    _res, recs = _two_target_run(tmp_path, (0.9, 0.1))
    # one entry per recycling event (split children share the event's marks)
    events = {r.ic_meta["recycled_from"]: r.ic_meta["recycle_target"]
              for r in recs if r.ic_meta["recycled_from"] is not None}
    picks = list(events.values())
    n = len(picks)
    assert n >= 30, "too few recycling events; lengthen the run"
    k0 = picks.count(0)
    assert abs(k0 - 0.9 * n) <= 4 * math.sqrt(n * 0.09), (k0, n)


def test_i1_recycle_target_weights_enter_the_we_protocol_hash(tmp_path):
    _, a = _two_target_run(tmp_path, (0.5, 0.5), "ha", n_iter=1)
    _, b = _two_target_run(tmp_path, (0.6, 0.4), "hb", n_iter=1)
    assert a[0].ic_meta["we_protocol_hash"] != b[0].ic_meta["we_protocol_hash"]


def test_m3_recycle_target_inside_the_sink_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sink"):
        run_we(_init("ins", 2, -1.0, LA), _dw_backend(barrier=1.0, dt=2e-3), BinnedWE(_floor_bin, 2),
               _sink_rule(1.0, 0.1, tau_persist=0.05), _z, 2, 0.1, Store(tmp_path / "s.db"), 0, "ins",
               [_src(1.5)], dt_obs=0.01)


def test_i2_segment_own_rows_drop_the_carried_tail_and_the_repeated_start(tmp_path):
    """p7 I-2: a record's series = replayed ancestor rows (t < 0) + own rows,
    whose k=0 repeats the parent's last row. segment_own_rows keeps only what
    the segment itself contributes."""
    store = Store(tmp_path / "s.db")
    run_we(
        _init("own", 4, -1.0, LA, seed=5), _dw_backend(barrier=1.0, dt=2e-3),
        BinnedWE(_edges_bin(np.linspace(-1, 1, 11)), target_per_bin=2),
        _sink_rule(1.0, 0.1, tau_persist=0.03), _z, 15, 0.1, store, 5, "own", [_src(-1.0)], dt_obs=0.01,
    )
    recs = {r.key_digest: r for r in store.iter(kind="segment")}
    assert any(min(r.observables["t"]) < 0 for r in recs.values())  # carried tails exist
    for r in recs.values():
        own = segment_own_rows(r)
        assert set(own) == set(r.observables)
        assert all(t > 0 for t in own["t"]) if r.parent_digest is not None else own["t"][0] == 0.0
        if r.parent_digest is not None and r.stop_reason == "fixed_lag":
            assert len(own["t"]) == 10  # tau_seg / dt_obs new observations
        # a lineage never repeats a time
        lin = lineage_series(recs, r.key_digest)
        assert np.all(np.diff(lin["t"]) > 0)


def test_int2_m4_progress_coordinate_enters_the_segment_protocol(tmp_path):
    def z_other(state):
        return np.atleast_1d(np.asarray(state.x, dtype=float)[:1] * 2.0)

    calls = []

    def proto(z_fn):
        calls.append(1)
        store = Store(tmp_path / f"z{len(calls)}.db")
        run_we(_init("zz", 2, -1.0, LA), _dw_backend(), BinnedWE(_floor_bin, 2), FixedLag(0.02), z_fn,
               1, 0.02, store, 0, "zz", None, dt_obs=0.01)
        return next(store.iter()).protocol_hash

    assert proto(_z) != proto(z_other)
    spec_z = SpecPredicate(_z, "z0 = x[0]")
    assert proto(spec_z) != proto(_z)


def test_int2_m5_we_continuation_refuses_another_physics_config(tmp_path):
    store = Store(tmp_path / "s.db")
    kw = dict(dt_obs=0.01)
    res = run_we(_init("ct", 2, -1.0, LA), _dw_backend(), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z,
                 1, 0.02, store, 0, "ct", None, **kw)
    nxt = BinnedWE(_floor_bin, 2).resample(res.final_walkers, 0, IterKey(0, "ct", 0))
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_we(nxt, _dw_backend(barrier=7.0), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z, 1, 0.02,
               store, 0, "ct", None, start_iteration=1, **kw)
    run_we(nxt, _dw_backend(barrier=7.0), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z, 1, 0.02,
           store, 0, "ct", None, start_iteration=1, allow_config_change=True, **kw)
    # the matching continuation is accepted
    store2 = Store(tmp_path / "s2.db")
    res = run_we(_init("ct", 2, -1.0, LA), _dw_backend(), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z,
                 1, 0.02, store2, 0, "ct", None, **kw)
    nxt = BinnedWE(_floor_bin, 2).resample(res.final_walkers, 0, IterKey(0, "ct", 0))
    out = run_we(nxt, _dw_backend(), BinnedWE(_floor_bin, 2), FixedLag(0.02), _z, 1, 0.02,
                 store2, 0, "ct", None, start_iteration=1, **kw)
    assert out.tau_seg == 0.02


def test_int2_m7_tau_persist_warning_mentions_we_carry():
    with pytest.warns(UserWarning, match="WE"):
        AbsorbingAB(_le("A", -1.2), _ge("B", 0.4), tau_persist=0.05, t_max=0.03)
