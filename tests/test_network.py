"""Task 11: absorbing milestone network and Markov test (cytherea.network).

Fast tests: 11.1 (solve_absorption against closed forms), the counting
rules of build_transitions on hand-built shot and WE segment records (last
milestone visited, carried labels, own rows only, absorbed lineages, splits
and merges in the direct estimate), argument checks, and the Markov test on
synthetic discrete chains (a Markov chain passes at about its nominal size;
a label-dependent chain pooled into one network fails, per-label networks
pass). The WE acceptance cases 11.2-11.4 are in test_acceptance_a0_part2.py.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cytherea.network import MarkovReport, StageNetwork, build_transitions, markov_test, solve_absorption
from cytherea.store import ShotRecord

A_CODE, B_CODE = 100, 101
_ids = itertools.count()


def _milestone_of(obs):
    m = int(obs["m"])
    if m < 0:
        return None
    if m == A_CODE:
        return "A"
    if m == B_CODE:
        return "B"
    return f"m{m}"


def _net(M, augmented=False):
    return StageNetwork([f"m{k}" for k in range(M)], ["A", "B"], augmented)


def _rec(ms, *, kind="shot", weight=1.0, label=(0, 0), parent=None, run="r", it=0, seed=0,
         t=None, final=None, stop="fixed_lag"):
    i = next(_ids)
    if kind == "segment":
        key = {"global_seed": seed, "run_id": run, "iteration": it, "walker_id": i}
    else:
        key = {"shot_id": i}
    return ShotRecord(
        key_digest=f"{i:064x}", key=key, kind=kind, frame_id=None, origin_label=label,
        ic_validity={}, stop_rule_kind="absorbing_AB", stop_reason=stop, event_time=None,
        physics_config_hash="0" * 64, backend_provenance={}, code_version="test",
        observables={"t": list(t) if t is not None else [float(k) for k in range(len(ms))],
                     "m": [float(m) for m in ms]},
        final_state_label=final, weight=weight, parent_digest=parent,
        observables_thinned=False,
    )


# --------------------------------------------------------------- 11.1


def _ruin(M, p):
    """Gambler's ruin on interior states 1..M (A = 0, B = M+1), step +1 w.p. p."""
    Q = np.zeros((M, M))
    R = np.zeros((M, 2))
    for i in range(M):
        if i + 1 < M:
            Q[i, i + 1] = p
        else:
            R[i, 1] = p
        if i > 0:
            Q[i, i - 1] = 1 - p
        else:
            R[i, 0] = 1 - p
    k = np.arange(1, M + 1)
    if p == 0.5:
        pb = k / (M + 1)
    else:
        r = (1 - p) / p
        pb = (1 - r**k) / (1 - r ** (M + 1))
    return Q, R, np.column_stack([1 - pb, pb])


@pytest.mark.parametrize("p", [0.3, 0.5, 0.62])
def test_11_1_solve_absorption_matches_gamblers_ruin(p):
    Q, R, exact = _ruin(12, p)
    B = solve_absorption(Q, R)
    assert np.max(np.abs(B - exact)) < 1e-12
    assert np.max(np.abs(B.sum(axis=1) - 1.0)) < 1e-12


def test_11_1_random_network_solves_the_first_step_equations():
    rng = np.random.Generator(np.random.PCG64(4))
    M, A = 15, 3
    P = rng.dirichlet(np.ones(M + A) * 0.5, size=M)
    P[:, :M][np.eye(M, dtype=bool)] = 0.0
    P /= P.sum(axis=1, keepdims=True)
    Q, R = P[:, :M], P[:, M:]
    B = solve_absorption(Q, R)
    assert np.max(np.abs(B - (Q @ B + R))) < 1e-12
    assert np.max(np.abs(B.sum(axis=1) - 1.0)) < 1e-12
    assert np.all(B >= 0)


def test_solve_absorption_refuses_invalid_networks():
    Q, R, _ = _ruin(4, 0.4)
    bad = Q.copy()
    bad[1] = np.nan
    with pytest.raises(ValueError, match=r"rows \[1\] are not finite"):
        solve_absorption(bad, R)
    with pytest.raises(ValueError, match="do not sum to 1"):
        solve_absorption(0.9 * Q, R)
    neg = Q.copy()
    neg[0, 1], neg[0, 0] = 1.2, -0.2
    with pytest.raises(ValueError, match="non-negative"):
        solve_absorption(neg, R)
    # states 2 and 3 form a closed class: never absorbed
    Qc = np.array([[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    Rc = np.array([[0.5, 0], [0, 0.5], [0, 0], [0, 0]])
    with pytest.raises(ValueError, match=r"transient rows \[2, 3\]"):
        solve_absorption(Qc, Rc)
    with pytest.raises(ValueError, match="need Q"):
        solve_absorption(Q[:, :3], R)


# --------------------------------------------------------- counting rules


def test_last_milestone_transitions_in_one_shot():
    # rows: gap, m0, gap, m0 again (no transition), m1, gap, m1, m2, m1, A
    r = _rec([-1, 0, -1, 0, 1, -1, 1, 2, 1, A_CODE])
    Q, R = build_transitions([r], _net(3), _milestone_of, None)
    assert Q.tolist() == [[0, 1, 0], [0, 0, 0.5], [0, 1, 0]]
    assert R.tolist() == [[0, 0], [0.5, 0], [0, 0]]
    # rows after the absorption are ignored; a milestone never left is a NaN row
    r2 = _rec([0, B_CODE, 1, 2])
    Q2, R2 = build_transitions([r2], _net(3), _milestone_of, None)
    assert R2[0].tolist() == [0, 1] and np.isnan(Q2[1]).all() and np.isnan(R2[2]).all()


def test_stop_rule_event_absorbs_at_the_end():
    # persistence-confirmed event: the rows never map to B, the record ends in B
    r = _rec([0, 1, -1, -1], final="B", stop="B")
    Q, R = build_transitions([r], _net(2), _milestone_of, None)
    assert Q[0].tolist() == [0, 1] and R[1].tolist() == [0, 1]


def test_weights_and_shot_frame_weights():
    r1 = _rec([0, 1, A_CODE], weight=3.0)
    r2 = _rec([0, 1, B_CODE], weight=1.0)
    _, R = build_transitions([r1, r2], _net(2), _milestone_of, None)
    assert R[1].tolist() == [0.75, 0.25]


def test_segments_carry_the_label_and_count_only_their_own_rows():
    # root (it 0): m0 .. m1; child (it 1): row k=0 repeats the parent's last row,
    # and a replayed ancestor tail (t < 0) that must not be counted; the child's own
    # rows go back to m0.  Grandchildren (it 2, split w/2 each): m0 -> m1 -> B and m0 -> A.
    root = _rec([0, -1, 1], kind="segment", it=0, weight=1.0)
    child = _rec([2, 2, 1, -1, 0], kind="segment", it=1, weight=1.0, parent=root.key_digest,
                 t=[-2.0, -1.0, 0.0, 1.0, 2.0])
    g1 = _rec([0, 1, B_CODE], kind="segment", it=2, weight=0.5, parent=child.key_digest)
    g2 = _rec([0, A_CODE], kind="segment", it=2, weight=0.5, parent=child.key_digest)
    Q, R = build_transitions([g2, child, g1, root], _net(3), _milestone_of, None)
    # weights: m0->m1 (root, 1.0), m1->m0 (child, 1.0), m0->m1 (g1, .5), m1->B (g1, .5), m0->A (g2, .5)
    assert Q[0].tolist() == [0, 1.5 / 2.0, 0] and R[0].tolist() == [0.5 / 2.0, 0]
    assert Q[1].tolist() == [1.0 / 1.5, 0, 0] and R[1].tolist() == [0, 0.5 / 1.5]
    assert np.isnan(Q[2]).all()  # m2 appears only in the replayed tail


def test_children_of_an_absorbed_segment_are_ignored():
    root = _rec([0, 1, A_CODE, 1], kind="segment", it=0)
    late = _rec([1, 0, 1, B_CODE], kind="segment", it=1, parent=root.key_digest)
    Q, R = build_transitions([root, late], _net(2), _milestone_of, None)
    assert Q[0].tolist() == [0, 1] and R[1].tolist() == [1, 0]


def test_recycled_walker_starts_a_new_trajectory():
    sink = _rec([0, 1, B_CODE], kind="segment", it=0, final="B", stop="B")
    recycled = _rec([0, -1, 1, A_CODE], kind="segment", it=1)  # parent None
    recycled.ic_meta = {"recycled_from": sink.key_digest}
    Q, R = build_transitions([sink, recycled], _net(2), _milestone_of, None)
    assert Q[0].tolist() == [0, 1] and R[1].tolist() == [0.5, 0.5]


def test_input_checks():
    root = _rec([0, 1], kind="segment", it=0)
    orphan = _rec([1, 0], kind="segment", it=1, parent="f" * 64)
    with pytest.raises(ValueError, match="not in the given records"):
        build_transitions([root, orphan], _net(2), _milestone_of, None)
    with pytest.raises(ValueError, match="non-finite"):
        build_transitions([_rec([0, 1], stop="nonfinite")], _net(2), _milestone_of, None)
    with pytest.raises(ValueError, match="not a state of the network"):
        build_transitions([_rec([0, 5])], _net(2), _milestone_of, None)
    with pytest.raises(ValueError, match="augmented network is built per origin label"):
        build_transitions([root], _net(2, augmented=True), _milestone_of, None)
    with pytest.raises(ValueError, match="pools every origin label"):
        build_transitions([root], _net(2), _milestone_of, (0, 0))
    with pytest.raises(ValueError, match="distinct"):
        StageNetwork(["m0", "A"], ["A"], False)


def test_label_filter_of_an_augmented_network():
    a = _rec([0, 1, A_CODE], label=(1, 1))
    b = _rec([0, 1, B_CODE], label=(2, 2))
    _, R1 = build_transitions([a, b], _net(2, True), _milestone_of, (1, 1))
    _, R2 = build_transitions([a, b], _net(2, True), _milestone_of, (2, 2))
    assert R1[1].tolist() == [1, 0] and R2[1].tolist() == [0, 1]


def test_direct_outcome_through_splits_and_merges():
    # run 1: root s0 (w 1) at m0 splits into s1 (w .5: m0 -> m1 -> B) and s2 (w .5: m0 -> A)
    s0 = _rec([0], kind="segment", run="r1", it=0, weight=1.0)
    s1 = _rec([0, 1, B_CODE], kind="segment", run="r1", it=1, weight=0.5, parent=s0.key_digest)
    s2 = _rec([0, A_CODE], kind="segment", run="r1", it=1, weight=0.5, parent=s0.key_digest)
    # run 2: two roots (w .5 each) at m1; a merge keeps u0 (child w 1.0 -> A), u1 has no child
    u0 = _rec([1], kind="segment", run="r2", it=0, weight=0.5)
    u1 = _rec([1], kind="segment", run="r2", it=0, weight=0.5)
    u2 = _rec([1, A_CODE], kind="segment", run="r2", it=1, weight=1.0, parent=u0.key_digest)
    recs = [s0, s1, s2, u0, u1, u2]
    rep = markov_test(recs, recs, _net(2), _milestone_of, None, n_boot=20, min_hits=1)
    got = {e["milestone"]: e for e in rep.strata}
    assert got["m0"]["direct"] == [0.5, 0.5] and got["m0"]["n_hits"] == 1
    # m1: hits in s1 (w .5 -> B), u0 (w .5, G = (1.0/0.5) e_A), u1 (w .5, G = 0)
    assert got["m1"]["n_hits"] == 3 and got["m1"]["weight"] == 1.5
    assert got["m1"]["absorbed_weight"] == pytest.approx(1.5)
    assert got["m1"]["direct"] == pytest.approx([1.0 / 1.5, 0.5 / 1.5])
    assert rep.n_units_train == rep.n_units_heldout == 2
    assert isinstance(rep, MarkovReport)


# ------------------------------------------------------- Markov test


def _chain_shots(rng, n, M, p_right, label, start=None):
    """Nearest-neighbour walk on milestones 0..M-1 (A below 0, B above M-1), with
    a random number of non-milestone rows between visits."""
    out = []
    for _ in range(n):
        m = int(rng.integers(0, M)) if start is None else start
        rows = [m]
        while True:
            rows.extend([-1] * int(rng.integers(0, 3)))
            m += 1 if rng.random() < p_right else -1
            if m < 0:
                rows.append(A_CODE)
                break
            if m >= M:
                rows.append(B_CODE)
                break
            rows.append(m)
        out.append(_rec(rows, label=label))
    return out


def test_markov_chain_passes_at_about_the_nominal_size():
    M, n_pass, n_seeds = 8, 0, 30
    for seed in range(n_seeds):
        rng = np.random.Generator(np.random.PCG64(1000 + seed))
        train = _chain_shots(rng, 300, M, 0.45, (0, 0))
        held = _chain_shots(rng, 300, M, 0.45, (0, 0))
        rep = markov_test(train, held, _net(M), _milestone_of, None, n_boot=300, seed=seed)
        n_pass += rep.passed
        assert rep.n_boot_failed == 0 and rep.unabsorbed_fraction == 0.0
    assert n_pass >= 0.85 * n_seeds, n_pass


def test_pooled_labels_fail_and_augmented_networks_pass():
    """The 11.3 / 11.4 logic on a discrete chain: the label sets the drift, so the
    milestone alone is not a Markov state."""
    M = 8
    rng = np.random.Generator(np.random.PCG64(7))
    labs = {(1, 1): 0.35, (2, 2): 0.65}
    train = [r for lab, p in labs.items() for r in _chain_shots(rng, 300, M, p, lab)]
    held = [r for lab, p in labs.items() for r in _chain_shots(rng, 300, M, p, lab)]
    pooled = markov_test(train, held, _net(M), _milestone_of, None, n_boot=500)
    assert pooled.passed is False and pooled.max_dev > 0.2
    assert {s["label"] for s in pooled.strata} == set(labs)
    for lab, p in labs.items():
        rep = markov_test(train, held, _net(M, True), _milestone_of, lab, n_boot=500)
        assert rep.passed is True, (lab, rep.max_dev, rep.ci)
        assert {s["label"] for s in rep.strata} == {lab}
        Q, R = build_transitions(train, _net(M, True), _milestone_of, lab)
        _, _, exact = _ruin(M, p)
        # sanity only (~130 visits at the far end); the accuracy gate is 11.2
        assert np.max(np.abs(solve_absorption(Q, R) - exact)) < 0.1


def test_markov_test_argument_checks():
    rng = np.random.Generator(np.random.PCG64(3))
    one = _chain_shots(rng, 1, 3, 0.5, (0, 0))
    many = _chain_shots(rng, 50, 3, 0.5, (0, 0))
    with pytest.raises(ValueError, match=">= 2 independent units"):
        markov_test(one, many, _net(3), _milestone_of, None)
    with pytest.raises(ValueError, match="alpha"):
        markov_test(many, many, _net(3), _milestone_of, None, alpha=1.5)
    with pytest.raises(ValueError, match="no \\(label, milestone\\) stratum"):
        markov_test(many, many, _net(3), _milestone_of, None, min_hits=10**6)
    # one WE run is one unit, however many segments it has
    segs = [_rec([0, 1, B_CODE], kind="segment", run="only", it=0) for _ in range(5)]
    with pytest.raises(ValueError, match="got 1 train"):
        markov_test(segs, many, _net(3), _milestone_of, None)
