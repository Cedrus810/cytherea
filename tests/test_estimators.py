"""Tests for cytherea.estimate: T(tau), CK, committor, NAM k_on, decomposition,
frame bootstrap (Task 8; design doc sections 4.1-4.4).

The statistical tests are written so that a wrong implementation fails them:
CI tests check *coverage of the known truth* over many independent synthetic
datasets (a CI that is too narrow or too wide both fail), the CK tests use a
Markov chain (must pass) and a lumped non-Markov chain (must fail), and the
hierarchical bootstrap is compared with its exact analytic variance.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from cytherea.estimate import (
    CommittorEstimate,
    Decomposition,
    KonEstimate,
    TEstimate,
    ck_test,
    decompose,
    estimate_committor,
    estimate_kon,
    estimate_T,
    hierarchical_bootstrap,
    nam_beta_inf,
)
from cytherea.store import ShotRecord

# ----------------------------------------------------------------- helpers

P3 = np.array(
    [
        [0.90, 0.07, 0.03],
        [0.05, 0.85, 0.10],
        [0.02, 0.08, 0.90],
    ]
)


def _pairs_from_chain(P, n_per_state, rng):
    """Shooting-style data: n_per_state shots from each start state, one lag each."""
    n = P.shape[0]
    start = np.repeat(np.arange(n), n_per_state)
    cum = np.cumsum(P, axis=1)
    u = rng.random(start.size)
    end = (u[:, None] > cum[start]).sum(axis=1)
    return start, end


def _simulate_dtrajs(P, n_traj, n_steps, rng):
    """n_traj discrete trajectories of the Markov chain P, starts ~ stationary."""
    n = P.shape[0]
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    pi = pi / pi.sum()
    cum = np.cumsum(P, axis=1)
    x = np.empty((n_traj, n_steps), dtype=np.int64)
    x[:, 0] = rng.choice(n, size=n_traj, p=pi)
    u = rng.random((n_traj, n_steps))
    for t in range(1, n_steps):
        x[:, t] = (u[:, t, None] > cum[x[:, t - 1]]).sum(axis=1)
    return [row for row in x]


# 4 hidden states observed through the lumping {0,1} -> 0, {2,3} -> 1.
# Hidden 0 and 2 exit their lump fast (0.30/step), hidden 1 and 3 slowly
# (0.02/step), and the fast/slow states inside a lump interconvert slowly
# (0.01/step). The dwell time in an observed state is therefore a mixture of
# a fast and a slow exponential with memory of which one applies: the
# observed process is not Markov. (Symmetric -> uniform stationary law.)
P_HIDDEN = np.array(
    [
        [0.69, 0.01, 0.30, 0.00],
        [0.01, 0.97, 0.00, 0.02],
        [0.30, 0.00, 0.69, 0.01],
        [0.00, 0.02, 0.01, 0.97],
    ]
)
LUMP = np.array([0, 0, 1, 1])


def _rec(stop_reason, weight=1.0, i=0):
    return ShotRecord(
        key_digest=f"{i:064x}",
        key={"shot_id": i},
        kind="shot",
        frame_id=0,
        origin_label=None,
        ic_validity={"ok": True, "reasons": [], "n_redraws": 0},
        stop_rule_kind="committor",
        stop_reason=stop_reason,
        event_time=1.0,
        physics_config_hash="0" * 64,
        backend_provenance={},
        code_version="test",
        observables={},
        final_state_label=None,
        weight=weight,
    )


def _records(counts: dict, weight=1.0):
    out = []
    for reason, n in counts.items():
        out += [_rec(reason, weight, len(out) + k) for k in range(n)]
    return out


def _jeffreys(x, n):
    lo = 0.0 if x == 0 else stats.beta.ppf(0.025, x + 0.5, n - x + 0.5)
    hi = 1.0 if x == n else stats.beta.ppf(0.975, x + 0.5, n - x + 0.5)
    return lo, hi


def _jeffreys_cp(x, n):
    """Jeffreys with Clopper-Pearson bounds where <= 1 count is in a tail
    (the T(tau) interval): x = 0 upper, x = 1 lower, mirrored."""
    lo, hi = _jeffreys(x, n)
    if x == 0:
        hi = max(hi, stats.beta.ppf(0.975, 1, n))
    if x == 1:
        lo = min(lo, stats.beta.ppf(0.025, 1, n))
    if x == n:
        lo = min(lo, stats.beta.ppf(0.025, n, 1))
    if x == n - 1:
        hi = max(hi, stats.beta.ppf(0.975, n, 1))
    return lo, hi


# ----------------------------------------------------------------- 4.1 T(tau)


def test_8_1_estimate_T_consistent_with_truth():
    """8.1: T from 3-state Markov data agrees with the truth.

    Single dataset (2000 shots per state): every element within 5 binomial
    SEs of the truth, rows normalised, the CI brackets the estimate. That the
    95% CI contains the truth at the nominal rate is the calibrated statement
    and is tested over 200 datasets in test_8_1_estimate_T_ci_coverage (a
    single-dataset "truth in all 9 per-element CIs" has joint probability
    well below 95% and would only test the seed).
    """
    rng = np.random.default_rng(81)
    n = 2000
    start, end = _pairs_from_chain(P3, n, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                     reversible=False, n_boot=500, rng=np.random.default_rng(1))
    assert isinstance(est, TEstimate)
    assert est.T.shape == est.ci_low.shape == est.ci_high.shape == (3, 3)
    np.testing.assert_allclose(est.T.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(np.abs(est.T - P3) < 5 * np.sqrt(P3 * (1 - P3) / n))
    assert np.all(est.ci_low <= est.T) and np.all(est.T <= est.ci_high)
    # CI half-width matches the binomial SE (1.96 SE) to within 25%
    half = (est.ci_high - est.ci_low) / 2
    np.testing.assert_allclose(half, 1.96 * np.sqrt(est.T * (1 - est.T) / n), rtol=0.25)


def test_8_1_estimate_T_ci_coverage():
    """Per-element CI coverage of the true T over 200 independent datasets.

    Nominal 95% (the exact coverage of the interval at these T_ij and
    n = 300 is 0.937-0.964); with 400 datasets the binomial SD of a coverage
    is 1.1%, so a correct CI lands in [0.90, 0.99]. A CI of width 0, one
    that is too narrow, or a CI of [0, 1] all fail. Data and bootstrap use
    separate random streams, so the datasets do not depend on how many
    random numbers the estimator consumes.
    """
    rng, boot = np.random.default_rng(812), np.random.default_rng(813)
    n_sets = 400
    hits = np.zeros((3, 3))
    for _ in range(n_sets):
        start, end = _pairs_from_chain(P3, 300, rng)
        est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                         reversible=False, n_boot=20, rng=boot)
        hits += (est.ci_low <= P3) & (P3 <= est.ci_high)
    cov = hits / n_sets
    assert cov.min() >= 0.90, cov
    assert cov.max() <= 0.99, cov


def _clustered_shots(rng, F=30, K=10, m=0.3, kappa=10.0):
    """2 start states, F frames each, K shots per frame; frame f of either
    state ends in state 1 with its own probability p_f ~ Beta(m kappa,
    (1 - m) kappa), so shots of a frame are correlated (reviewer's design;
    between-frame variance m(1-m)/(kappa+1) = 0.019)."""
    start, end, fid = [], [], []
    for s in (0, 1):
        p = rng.beta(m * kappa, (1 - m) * kappa, F)
        for f in range(F):
            start += [s] * K
            end += list((rng.random(K) < p[f]).astype(int))
            fid += [s * F + f] * K
    return np.array(start), np.array(end), np.array(fid)


def test_estimate_T_frame_cluster_coverage():
    """Frame-clustered shots: coverage of the true T_01 = m over 300 datasets.
    The shot-level interval ignores the clustering and under-covers
    (reviewer: 0.86); with frame_ids (frames as the clustering unit of n_eff
    and of the bootstrap) it must reach >= 0.92 (binomial SD at 0.95 over
    300 sets: 1.3%). The naive coverage is asserted < 0.90, which shows this
    design discriminates the two."""
    rng, boot = np.random.default_rng(824), np.random.default_rng(825)
    n_sets, m = 300, 0.3
    hit_frames = hit_naive = 0
    for _ in range(n_sets):
        start, end, fid = _clustered_shots(rng, m=m)
        w = np.ones(start.size)
        a = estimate_T(start, end, w, 2, 1.0, False, 20, boot, frame_ids=fid)
        b = estimate_T(start, end, w, 2, 1.0, False, 20, boot)
        np.testing.assert_array_equal(a.T, b.T)
        hit_frames += a.ci_low[0, 1] <= m <= a.ci_high[0, 1]
        hit_naive += b.ci_low[0, 1] <= m <= b.ci_high[0, 1]
    msg = f"coverage frames {hit_frames / n_sets}, naive {hit_naive / n_sets}"
    assert hit_frames / n_sets >= 0.92, msg
    assert hit_naive / n_sets < 0.90, msg


def test_estimate_T_frame_ids_all_distinct_equals_shot_interval():
    """One shot per frame is the unclustered case: same intervals."""
    rng = np.random.default_rng(7)
    start, end = _pairs_from_chain(P3, 50, rng)
    w = np.ones(start.size)
    a = estimate_T(start, end, w, 3, 1.0, False, 50, np.random.default_rng(1))
    b = estimate_T(start, end, w, 3, 1.0, False, 50, np.random.default_rng(1),
                   frame_ids=np.arange(start.size))
    np.testing.assert_allclose(a.ci_low, b.ci_low, rtol=1e-12)
    np.testing.assert_allclose(a.ci_high, b.ci_high, rtol=1e-12)


def test_estimate_T_frame_weights_reproduce_design_formula():
    """Per-shot weight w_k / K_k gives design 4.1's
    T_ij = sum_k w_k n_(k->j) / sum_k w_k (n = fraction of frame k's shots)."""
    # state 0: frame 0 (w=2, 4 shots, 1 ends in 1), frame 1 (w=1, 2 shots, both end in 1);
    # state 1: frames 2 and 3 (w=1, 1 shot each)
    start = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    end = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    fid = np.array([0, 0, 0, 0, 1, 1, 2, 3])
    w = np.array([2 / 4] * 4 + [1 / 2] * 2 + [1.0, 1.0])
    est = estimate_T(start, end, w, 2, 1.0, False, 10, np.random.default_rng(0), frame_ids=fid)
    expected_01 = (2 * 0.25 + 1 * 1.0) / 3
    assert est.T[0, 1] == pytest.approx(expected_01, rel=1e-14)


def test_estimate_T_frame_in_two_states_raises():
    with pytest.raises(ValueError, match="frame"):
        estimate_T(np.array([0, 1]), np.array([0, 1]), np.ones(2), 2, 1.0, False, 10,
                   np.random.default_rng(0), frame_ids=np.array([5, 5]))
    with pytest.raises(ValueError, match="frame_ids"):
        estimate_T(np.array([0, 1]), np.array([0, 1]), np.ones(2), 2, 1.0, False, 10,
                   np.random.default_rng(0), frame_ids=np.array([5]))


def test_estimate_T_single_frame_state_raises():
    """R30: a start state with one frame would give a zero-width CI silently."""
    start = np.array([0] * 10 + [1] * 10)
    end = np.array([0, 1] * 10)
    fid = np.array([0] * 10 + list(range(1, 11)))
    with pytest.raises(ValueError, match="start state 0 has 1 frame"):
        estimate_T(start, end, np.ones(20), 2, 1.0, False, 50, np.random.default_rng(0), frame_ids=fid)


def test_estimate_T_weighted_counts_exact():
    """C_ij = sum of weights of shots i->j, row-normalized (design 4.1)."""
    start = np.array([0, 0, 0, 1, 1])
    end = np.array([0, 1, 1, 0, 1])
    w = np.array([1.0, 2.0, 1.0, 0.5, 1.5])
    est = estimate_T(start, end, w, 2, lag=1.0, reversible=False,
                     n_boot=10, rng=np.random.default_rng(0))
    np.testing.assert_allclose(est.T, [[0.25, 0.75], [0.25, 0.75]], rtol=0, atol=1e-15)


def test_estimate_T_implied_timescale():
    """2-state T = [[1-a, a], [b, 1-b]]: its = -lag / ln(1 - a - b)."""
    a, b, lag = 0.1, 0.3, 2.5
    start = np.array([0, 0, 1, 1])
    end = np.array([0, 1, 0, 1])
    w = np.array([1 - a, a, b, 1 - b])
    est = estimate_T(start, end, w, 2, lag=lag, reversible=False,
                     n_boot=10, rng=np.random.default_rng(0))
    assert est.its.shape == (1,)
    assert est.its[0] == pytest.approx(-lag / math.log(1 - a - b), rel=1e-12)


def test_estimate_T_its_unit_eigenvalue_is_plus_inf():
    """A second unit eigenvalue (disconnected T) gives its = +inf, never -inf."""
    est = estimate_T(np.array([0, 1]), np.array([0, 1]), np.ones(2), 2, lag=1.0,
                     reversible=False, n_boot=10, rng=np.random.default_rng(0))
    assert est.its.tolist() == [math.inf]


def test_estimate_T_its_sorted_descending():
    rng = np.random.default_rng(3)
    start, end = _pairs_from_chain(P3, 500, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                     reversible=False, n_boot=10, rng=rng)
    assert est.its.shape == (2,)
    assert est.its[0] >= est.its[1] > 0
    lam = np.sort(np.abs(np.linalg.eigvals(est.T)))[::-1]
    np.testing.assert_allclose(est.its, -1.0 / np.log(lam[1:]), rtol=1e-12)


def test_estimate_T_reversible_detailed_balance_and_truth():
    """reversible=True: deeptime reversible MLE; detailed balance holds and
    the CI brackets the estimate. (Coverage of the truth is tested over many
    datasets in test_estimate_T_reversible_ci_coverage; "truth in all 9 CIs"
    on one dataset would only test the seed -- reviewer M9.)"""
    # reversible truth: symmetric flux matrix X, T = X / rowsum
    X = np.array([[50.0, 3.0, 1.0], [3.0, 30.0, 4.0], [1.0, 4.0, 40.0]])
    P = X / X.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(84)
    start, end = _pairs_from_chain(P, 2000, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                     reversible=True, n_boot=200, rng=np.random.default_rng(2))
    T = est.T
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-12)
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    pi /= pi.sum()
    flux = pi[:, None] * T
    np.testing.assert_allclose(flux, flux.T, atol=1e-8)
    assert np.all(est.ci_low <= T) and np.all(T <= est.ci_high)
    assert np.all(np.abs(T - P) < 5 * np.sqrt(P * (1 - P) / 2000))
    # it really is a different estimator from row normalisation
    nonrev = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                        reversible=False, n_boot=10, rng=np.random.default_rng(2))
    assert not np.allclose(T, nonrev.T, atol=1e-6)


def test_estimate_T_reversible_ci_coverage():
    """Reversible path: per-element coverage of the reversible truth over 200
    datasets x 100 bootstrap replicates (binomial SD 1.9%). The percentile
    bootstrap of the reversible MLE covers 0.91-0.945 here for both the
    pre-fix and the multiplicity implementation (measured on 400 identical
    datasets), so the bound is 0.88 (~2 SD below that)."""
    X = np.array([[50.0, 3.0, 1.0], [3.0, 30.0, 4.0], [1.0, 4.0, 40.0]])
    P = X / X.sum(axis=1, keepdims=True)
    rng, boot = np.random.default_rng(841), np.random.default_rng(842)
    n_sets = 200
    hits = np.zeros((3, 3))
    for _ in range(n_sets):
        start, end = _pairs_from_chain(P, 300, rng)
        est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                         reversible=True, n_boot=100, rng=boot)
        hits += (est.ci_low <= P) & (P <= est.ci_high)
    cov = hits / n_sets
    assert cov.min() >= 0.88, f"reversible per-element coverage:\n{cov}"


def test_estimate_T_reversible_disconnected_raises():
    start = np.array([0, 0, 1, 1])
    end = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="connected"):
        estimate_T(start, end, np.ones(4), 2, lag=1.0, reversible=True,
                   n_boot=10, rng=np.random.default_rng(0))


def test_estimate_T_empty_row_raises_naming_state():
    start = np.array([0, 0, 1, 1])
    end = np.array([0, 2, 1, 2])
    with pytest.raises(ValueError, match="state 2"):
        estimate_T(start, end, np.ones(4), 3, lag=1.0, reversible=False,
                   n_boot=10, rng=np.random.default_rng(0))


def test_estimate_T_zero_weight_row_raises():
    start = np.array([0, 1])
    end = np.array([1, 0])
    with pytest.raises(ValueError, match="state 1"):
        estimate_T(start, end, np.array([1.0, 0.0]), 2, lag=1.0, reversible=False,
                   n_boot=10, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    "start,end,w",
    [
        ([0, 3], [0, 1], [1.0, 1.0]),       # state out of range
        ([0, -1], [0, 1], [1.0, 1.0]),
        ([0, 1], [0, 1], [1.0, -1.0]),      # negative weight
        ([0, 1], [0, 1], [1.0, np.nan]),    # non-finite weight
        ([0, 1, 1], [0, 1], [1.0, 1.0]),    # length mismatch
    ],
)
def test_estimate_T_rejects_bad_input(start, end, w):
    with pytest.raises(ValueError):
        estimate_T(np.array(start), np.array(end), np.array(w), 2, lag=1.0,
                   reversible=False, n_boot=10, rng=np.random.default_rng(0))


def test_estimate_T_does_not_claim_ck():
    """estimate_T sees single-lag pairs only, so it cannot run CK: it must not
    report a pass (ck_passed=None meaning "not tested", ck_max_dev=nan)."""
    rng = np.random.default_rng(5)
    start, end = _pairs_from_chain(P3, 100, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, lag=1.0,
                     reversible=False, n_boot=10, rng=rng)
    assert est.ck_passed is None
    assert math.isnan(est.ck_max_dev)


def test_estimate_T_reproducible_with_rng():
    rng = np.random.default_rng(6)
    start, end = _pairs_from_chain(P3, 100, rng)
    a = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 50, np.random.default_rng(9))
    b = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 50, np.random.default_rng(9))
    np.testing.assert_array_equal(a.ci_low, b.ci_low)
    np.testing.assert_array_equal(a.ci_high, b.ci_high)


# ----------------------------------------------------------------- CK test


def test_8_1_ck_markov_chain_passes():
    """8.1: CK on a genuine 3-state Markov chain passes."""
    rng = np.random.default_rng(811)
    dtrajs = _simulate_dtrajs(P3, 20, 2000, rng)
    passed, max_dev = ck_test(dtrajs, lag_steps=1, ks=(2, 3, 5), n_states=3,
                              n_boot=500, rng=np.random.default_rng(10))
    assert passed is True
    assert 0.0 < max_dev < 0.02


def test_8_2_ck_lumped_hidden_chain_fails():
    """8.2: observed = lumping of two hidden states with different exit
    kinetics -> not Markov -> CK must fail."""
    rng = np.random.default_rng(82)
    hidden = _simulate_dtrajs(P_HIDDEN, 20, 2000, rng)
    dtrajs = [LUMP[h] for h in hidden]
    passed, max_dev = ck_test(dtrajs, lag_steps=1, ks=(2, 3, 5), n_states=2,
                              n_boot=500, rng=np.random.default_rng(11))
    assert passed is False
    assert max_dev > 0.05


def test_ck_needs_several_trajectories():
    with pytest.raises(ValueError, match="trajector"):
        ck_test([np.zeros(100, dtype=int)], 1, (2,), 1, 10, np.random.default_rng(0))


def test_ck_rejects_bad_ks():
    dtrajs = [np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])]
    with pytest.raises(ValueError):
        ck_test(dtrajs, 1, (), 2, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        ck_test(dtrajs, 1, (0,), 2, 10, np.random.default_rng(0))


def _nn_chain(n=6, p=0.03):
    """Sparse nearest-neighbour Markov chain (reviewer's false-fail case)."""
    P = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            P[i, i - 1] = p
        if i < n - 1:
            P[i, i + 1] = p
        P[i, i] = 1 - P[i].sum()
    return P


def test_ck_size_and_power_over_seeds():
    """CK is one global 5%-level test: over 40 independent datasets
    (20 x 2000 steps, ks = 2, 3, 5, n_boot = 300) the sparse 6-state
    nearest-neighbour Markov chain passes in >= 85% (the old per-element
    rule passed 0/40), the 3-state chain in >= 85%, and the lumped non-Markov
    chain fails in >= 90%."""
    n = 40
    rates = {"nn6": 0, "p3": 0, "lumped_fail": 0}
    P6 = _nn_chain()
    for seed in range(n):
        rng = np.random.default_rng(5000 + seed)
        rates["nn6"] += ck_test(_simulate_dtrajs(P6, 20, 2000, rng), 1, (2, 3, 5), 6, 300, rng)[0]
        rates["p3"] += ck_test(_simulate_dtrajs(P3, 20, 2000, rng), 1, (2, 3, 5), 3, 300, rng)[0]
        h = _simulate_dtrajs(P_HIDDEN, 20, 2000, rng)
        rates["lumped_fail"] += not ck_test([LUMP[x] for x in h], 1, (2, 3, 5), 2, 300, rng)[0]
    msg = f"out of {n}: {rates}"
    assert rates["nn6"] >= 0.85 * n, msg
    assert rates["p3"] >= 0.85 * n, msg
    assert rates["lumped_fail"] >= 0.9 * n, msg


def test_ck_sparse_markov_chain_passes():
    """Single-seed regression for the reviewer's case (6-state NN, p = 0.03)."""
    rng = np.random.default_rng(5000)
    passed, D = ck_test(_simulate_dtrajs(_nn_chain(), 20, 2000, rng), 1, (2, 3, 5), 6, 300, rng)
    assert passed is True, D


# ----------------------------------------------------------------- 4.2 committor


def test_8_3_committor_timeout_invalidates():
    """8.3: 6 of 100 records time out -> timeout_frac 0.06, valid=False."""
    recs = _records({"A": 34, "B": 60, "timeout": 6})
    est = estimate_committor(recs)
    assert isinstance(est, CommittorEstimate)
    assert (est.n_A, est.n_B, est.n_timeout) == (34, 60, 6)
    assert est.timeout_frac == pytest.approx(0.06, rel=1e-15)
    assert est.valid is False
    assert est.q == pytest.approx(60 / 94, rel=1e-15)


def test_committor_five_percent_timeout_is_still_valid():
    est = estimate_committor(_records({"A": 50, "B": 45, "timeout": 5}))
    assert est.timeout_frac == pytest.approx(0.05)
    assert est.valid is True


def test_committor_jeffreys_interval():
    for nA, nB in [(34, 60), (10, 0), (0, 7), (1, 1)]:
        est = estimate_committor(_records({"A": nA, "B": nB}))
        lo, hi = _jeffreys(nB, nA + nB)
        assert est.ci == pytest.approx((lo, hi), rel=1e-10, abs=1e-15)
        assert est.ci[0] <= est.q <= est.ci[1]


def test_committor_ci_coverage():
    """Coverage of the true q over 2000 simulated experiments (n=40, q=0.3):
    binomial SD of the coverage is 0.5%, so a 95% interval lands in
    [0.93, 0.97]; Wald-with-bug / too-wide intervals fail."""
    rng = np.random.default_rng(83)
    q_true, n, n_sets = 0.3, 40, 2000
    hits = 0
    for _ in range(n_sets):
        nB = int(rng.binomial(n, q_true))
        est = estimate_committor(_records({"A": n - nB, "B": nB}))
        hits += est.ci[0] <= q_true <= est.ci[1]
    assert 0.93 <= hits / n_sets <= 0.97, hits / n_sets


def test_committor_weighted():
    """Weighted records: q = W_B/(W_A+W_B); n_eff = min(Korn-Graubard
    q(1-q)/V with V = sum w^2 (y-q)^2 / (sum w)^2, Kish); timeout_frac weighted."""
    recs = [_rec("A", 2.0, 0), _rec("A", 1.0, 1), _rec("B", 3.0, 2), _rec("B", 0.5, 3),
            _rec("timeout", 0.5, 4)]
    est = estimate_committor(recs)
    w = np.array([2.0, 1.0, 3.0, 0.5])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    q = 3.5 / 6.5
    V = (w**2 * (y - q) ** 2).sum() / w.sum() ** 2
    n_eff = min(q * (1 - q) / V, w.sum() ** 2 / (w**2).sum())
    assert est.q == pytest.approx(q, rel=1e-14)
    assert est.ci[0] == pytest.approx(stats.beta.ppf(0.025, q * n_eff + 0.5, (1 - q) * n_eff + 0.5))
    assert est.ci[1] == pytest.approx(stats.beta.ppf(0.975, q * n_eff + 0.5, (1 - q) * n_eff + 0.5))
    assert (est.n_A, est.n_B, est.n_timeout) == (2, 2, 1)
    assert est.timeout_frac == pytest.approx(0.5 / 7.0)
    assert est.valid is False


def _is_records(rng, p_true, p0, n):
    """Importance-sampled outcomes: drawn with P(B) = p0, reweighted to p_true,
    so the weights correlate with the outcome."""
    y = rng.random(n) < p0
    w = np.where(y, p_true / p0, (1 - p_true) / (1 - p0))
    return [_rec("B" if yi else "A", float(wi), i) for i, (yi, wi) in enumerate(zip(y, w))]


@pytest.mark.parametrize("p_true,p0,upper", [(0.3, 0.05, 0.98), (0.01, 0.3, 1.0)])
def test_committor_outcome_correlated_weights_coverage(p_true, p0, upper):
    """Coverage of the true q with outcome-correlated (importance) weights,
    n = 400 records, 1000 experiments (binomial SD 0.7%). Successes
    up-weighted (0.3, 0.05): Kish alone gave 0.848; required [0.92, 0.98].
    Successes down-weighted (0.01, 0.3): n_KG > n_Kish, so the Kish interval
    is used and the case over-covers (~1.0, the documented price of
    n_eff = min(n_KG, n_Kish)); only >= 0.92 is required."""
    rng = np.random.default_rng(835)
    n_sets = 1000
    hits = 0
    for _ in range(n_sets):
        est = estimate_committor(_is_records(rng, p_true, p0, 400))
        hits += est.ci[0] <= p_true <= est.ci[1]
    assert 0.92 <= hits / n_sets <= upper, hits / n_sets


@pytest.mark.parametrize("sigma,n", [(2.0, 100), (2.0, 400), (3.0, 100), (3.0, 400)])
def test_committor_heavy_tailed_independent_weights_coverage(sigma, n):
    """y ~ Bern(0.3), weights lognormal(0, sigma) independent of y: the
    linearised n_eff alone covered 0.65-0.86 here (reviewer); required >= 0.92.
    600 experiments per case: binomial SD at 0.95 is 0.9%, so 0.92 is ~3.4 SD
    below nominal while the old 0.86 would be ~10 SD below the threshold."""
    rng = np.random.default_rng(836)
    n_sets = 600
    hits = 0
    for _ in range(n_sets):
        y = rng.random(n) < 0.3
        w = rng.lognormal(0.0, sigma, n)
        est = estimate_committor([_rec("B" if a else "A", float(b), i)
                                  for i, (a, b) in enumerate(zip(y, w))])
        hits += est.ci[0] <= 0.3 <= est.ci[1]
    assert hits / n_sets >= 0.92, hits / n_sets


def _kish_jeffreys(recs):
    w = np.array([r.weight for r in recs])
    y = np.array([r.stop_reason == "B" for r in recs], dtype=float)
    q = (w * y).sum() / w.sum()
    ne = w.sum() ** 2 / (w**2).sum()
    lo = 0.0 if q == 0 else stats.beta.ppf(0.025, q * ne + 0.5, (1 - q) * ne + 0.5)
    hi = 1.0 if q == 1 else stats.beta.ppf(0.975, q * ne + 0.5, (1 - q) * ne + 0.5)
    return lo, hi


def test_committor_dominant_and_negligible_weights_not_narrower_than_kish():
    """One dominant weight, and a single negligible-weight minority record:
    the interval is at least as wide as the Kish-only interval (the
    linearised n_eff alone collapsed the second case to width ~5e-8)."""
    rng = np.random.default_rng(0)
    dominant = [_rec("B", 1.0, 0)] + [_rec("B" if rng.random() < 0.3 else "A", 1e-3, i)
                                      for i in range(1, 100)]
    minority = [_rec("B", 1.0, i) for i in range(99)] + [_rec("A", 1e-6, 99)]
    for recs in (dominant, minority):
        ci = estimate_committor(recs).ci
        k = _kish_jeffreys(recs)
        assert ci[1] - ci[0] >= (k[1] - k[0]) - 1e-12, (ci, k)
    assert estimate_committor(minority).ci[0] < 0.99  # was 0.99999995


def test_committor_weighted_edges_use_kish():
    """q = 0 or 1 with unequal weights: V = 0, n_eff falls back to Kish."""
    w = np.array([1.0, 3.0, 0.5])
    est = estimate_committor([_rec("A", x, i) for i, x in enumerate(w)])
    n_kish = w.sum() ** 2 / (w**2).sum()
    assert est.q == 0.0
    assert est.ci == pytest.approx((0.0, stats.beta.ppf(0.975, 0.5, n_kish + 0.5)))
    est = estimate_committor([_rec("B", x, i) for i, x in enumerate(w)])
    assert est.q == 1.0
    assert est.ci == pytest.approx((stats.beta.ppf(0.025, n_kish + 0.5, 0.5), 1.0))


def test_committor_uniform_weights_equal_unweighted():
    """Scaling all weights by a constant changes nothing (n_eff = n)."""
    counts = {"A": 30, "B": 12, "timeout": 1}
    a = estimate_committor(_records(counts))
    b = estimate_committor(_records(counts, weight=0.25))
    assert b.q == pytest.approx(a.q, rel=1e-14)
    assert b.ci == pytest.approx(a.ci, rel=1e-12)
    assert b.timeout_frac == pytest.approx(a.timeout_frac, rel=1e-14)


def test_committor_all_timeouts():
    est = estimate_committor(_records({"timeout": 5}))
    assert math.isnan(est.q)
    assert est.timeout_frac == 1.0
    assert est.valid is False


@pytest.mark.parametrize("bad", ["reaction", "escape", "fixed_lag", "pes_uncertain"])
def test_committor_rejects_foreign_stop_reason(bad):
    with pytest.raises(ValueError, match=bad):
        estimate_committor(_records({"A": 3, "B": 3, bad: 1}))


def test_committor_rejects_empty_and_bad_weight():
    with pytest.raises(ValueError):
        estimate_committor([])
    with pytest.raises(ValueError):
        estimate_committor([_rec("A", -1.0)])
    with pytest.raises(ValueError):
        estimate_committor([_rec("A", math.nan)])


# ----------------------------------------------------------------- 4.3 NAM


@pytest.mark.parametrize("b,q", [(1.0, 2.0), (0.5, 100.0), (3.0, 3.0001)])
def test_8_4_nam_beta_one(b, q):
    assert nam_beta_inf(1.0, b, q) == 1.0


@pytest.mark.parametrize("beta", [0.0, 0.01, 0.3, 0.9])
def test_8_5_nam_large_q_limit(beta):
    vals = [nam_beta_inf(beta, 1.0, q) for q in (10.0, 1e3, 1e6, 1e12)]
    errs = [abs(v - beta) for v in vals]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-11
    assert nam_beta_inf(beta, 1.0, math.inf) == beta


def test_nam_hand_value_and_monotone():
    # beta=0.2, Omega=b/q=0.5: 0.2/(1-0.8*0.5) = 1/3
    assert nam_beta_inf(0.2, 1.0, 2.0) == pytest.approx(1 / 3, rel=1e-15)
    betas = np.linspace(0, 1, 101)
    vals = [nam_beta_inf(x, 2.0, 3.0) for x in betas]
    assert np.all(np.diff(vals) > 0)
    assert all(x <= v <= 1 for x, v in zip(betas, vals))


@pytest.mark.parametrize(
    "beta,b,q",
    [(-0.1, 1, 2), (1.1, 1, 2), (math.nan, 1, 2), (0.5, 2, 2), (0.5, 3, 2),
     (0.5, 0, 2), (0.5, -1, 2), (0.5, math.inf, math.inf)],
)
def test_nam_rejects_bad_input(beta, b, q):
    with pytest.raises(ValueError):
        nam_beta_inf(beta, b, q)


def test_estimate_kon_values_and_ci():
    b, q, D = 2.0, 4.0, 0.1  # nm, nm, nm^2/ps -> k_on in nm^3/ps
    est = estimate_kon(_records({"reaction": 30, "escape": 70}), b, q, D)
    assert isinstance(est, KonEstimate)
    assert est.beta == pytest.approx(0.3, rel=1e-15)
    bi = 0.3 / (1 - 0.7 * 0.5)
    assert est.beta_inf == pytest.approx(bi, rel=1e-14)
    kD = 4 * math.pi * D * b
    assert est.kon == pytest.approx(kD * bi, rel=1e-14)
    lo, hi = _jeffreys(30, 100)
    assert est.beta_ci == pytest.approx((lo, hi), rel=1e-10)
    assert est.beta_inf_ci == pytest.approx((nam_beta_inf(lo, b, q), nam_beta_inf(hi, b, q)), rel=1e-12)
    assert est.ci == pytest.approx((kD * est.beta_inf_ci[0], kD * est.beta_inf_ci[1]), rel=1e-12)
    assert est.ci[0] < est.kon < est.ci[1]
    assert (est.n_reaction, est.n_escape, est.n_timeout) == (30, 70, 0)
    assert est.valid is True


def test_estimate_kon_counts_timeouts():
    est = estimate_kon(_records({"reaction": 30, "escape": 64, "timeout": 6}), 2.0, 4.0, 0.1)
    assert est.n_timeout == 6
    assert est.timeout_frac == pytest.approx(0.06)
    assert est.valid is False
    assert est.beta == pytest.approx(30 / 94)


def test_estimate_kon_rejects_bad_input():
    with pytest.raises(ValueError, match="'A'"):
        estimate_kon(_records({"reaction": 1, "A": 1}), 2.0, 4.0, 0.1)
    with pytest.raises(ValueError):
        estimate_kon(_records({"reaction": 1}), 2.0, 4.0, 0.0)
    with pytest.raises(ValueError):
        estimate_kon(_records({"reaction": 1}), 4.0, 2.0, 0.1)


# ----------------------------------------------------------------- 4.4 decomposition


def test_8_6_decomposition_identity_random():
    """8.6: population + dynamical == delta to < 1e-12 on random inputs, and
    delta is the directly computed change of <A>."""
    rng = np.random.default_rng(86)
    for _ in range(200):
        K = int(rng.integers(1, 41))
        W = rng.random((K, K)); W /= W.sum()
        Wp = rng.random((K, K)); Wp /= Wp.sum()
        A = rng.random((K, K))
        Ap = rng.random((K, K))
        d = decompose(W, Wp, A, Ap)
        assert isinstance(d, Decomposition)
        assert abs(d.population + d.dynamical - d.delta) < 1e-12
        assert d.delta == pytest.approx(float((Wp * Ap).sum() - (W * A).sum()), abs=1e-13)


def test_decomposition_terms_by_hand():
    """Terms are computed independently (not dynamical = delta - population)."""
    d = decompose(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([1.0, 2.0]), np.array([3.0, 5.0]))
    # dW=[-1,1], Abar=[2,3.5] -> 1.5 ; Wbar=[.5,.5], dA=[2,3] -> 2.5 ; delta=5-1
    assert (d.delta, d.population, d.dynamical) == (4.0, 1.5, 2.5)
    # same A -> purely population; same W -> purely dynamical
    W, Wp, A = np.array([0.5, 0.5]), np.array([0.7, 0.3]), np.array([1.0, 0.0])
    d = decompose(W, Wp, A, A)
    assert d.dynamical == 0.0 and d.population == pytest.approx(0.2) == d.delta
    d = decompose(W, W, A, np.array([0.0, 1.0]))
    assert d.population == 0.0 and d.dynamical == pytest.approx(0.0) == d.delta


def test_decomposition_rejects_bad_input():
    a = np.ones((2, 2))
    with pytest.raises(ValueError):
        decompose(a, a, a, np.ones((2, 3)))
    with pytest.raises(ValueError):
        decompose(a, a, a, np.array([[1.0, np.nan], [1.0, 1.0]]))


# ----------------------------------------------------------------- hierarchical bootstrap


def _gaussian_design(S, F, K, sa, sb, se, rng):
    """y_sfk = a_s + b_sf + e_sfk, a~N(0,sa^2), b~N(0,sb^2), e~N(0,se^2)."""
    a = rng.normal(0, sa, S)
    b = rng.normal(0, sb, (S, F))
    e = rng.normal(0, se, (S, F, K))
    y = a[:, None, None] + b[:, :, None] + e
    groups = {10 + s: {100 * s + f: list(y[s, f]) for f in range(F)} for s in range(S)}
    return y, groups


def _all_shots(rep):
    return np.concatenate([shots for _, frames in rep for _, shots in frames])


def _grand_mean(rep):
    return float(np.mean(_all_shots(rep)))


def _exact_boot_var(y, resample_states, resample_shots=False):
    """Exact (conditional on the data) variance of the resampled grand mean.

    Balanced design, S states x F frames x K shots, plug-in variances
    (divisor = number of units):
        V_A     = var_s(ybar_s)             between-state
        V_B,s   = var_f(ybar_sf)            between-frame within state s
        V_E,sf  = var_k(y_sfk)              between-shot within frame (s,f)
    One resampled frame contributes g = the mean of a uniformly chosen
    frame's shots, as they are (default):  Var(g | s) = V_B,s, or of K
    shots redrawn from it (resample_shots):  Var(g | s) = V_B,s + mean_f(V_E,sf)/K.
    A resampled state mean m = mean of F iid g:  Var(m | s) = Var(g | s)/F.
    States fixed:      Var* = (1/S^2) sum_s Var(m | s)
    States resampled:  Var* = (1/S) [V_A + mean_s Var(m | s)]
    """
    S, F, K = y.shape
    VE = y.var(axis=2)                       # (S, F)
    VB = y.mean(axis=2).var(axis=1)          # (S,)
    VA = y.mean(axis=(1, 2)).var()
    var_m = (VB + (VE.mean(axis=1) / K if resample_shots else 0.0)) / F   # (S,)
    if resample_states:
        return (VA + var_m.mean()) / S
    return var_m.sum() / S**2


@pytest.mark.parametrize("resample_shots", [False, True])
@pytest.mark.parametrize("resample_states", [True, False])
def test_8_7_hierarchical_bootstrap_variance_exact(resample_states, resample_shots):
    """8.7: bootstrap variance of the grand mean vs its exact analytic value.

    Design S=12 states, F=2 frames, K=2 shots, sigma_a=0.5, sigma_b=1,
    sigma_e=3 (seeded). n_boot=4000: the Monte-Carlo relative SE of a
    variance estimate of a near-Gaussian statistic is sqrt(2/n_boot) = 2.2%,
    so +-10% is a 4.5-sigma band. The check discriminates: the analytic
    variance of the other shot mode (frames kept whole vs shots redrawn)
    and, with shots redrawn, of skipping the frame level, differ from the
    correct one by more than 20% for this data (asserted).
    """
    y, groups = _gaussian_design(12, 2, 2, 0.5, 1.0, 3.0, np.random.default_rng(87))
    exact = _exact_boot_var(y, resample_states, resample_shots)
    # sensitivity: the analytic variance of a wrong scheme
    S, F, K = y.shape
    VE, VA = y.var(axis=2), y.mean(axis=(1, 2)).var()
    no_frame = VE.mean(axis=1) / (F * K)
    wrong = [_exact_boot_var(y, resample_states, not resample_shots),
             (VA + no_frame.mean()) / S if resample_states else no_frame.sum() / S**2]
    for w in wrong:
        assert abs(w / exact - 1) > 0.2, (w, exact)

    boot = hierarchical_bootstrap(groups, _grand_mean, 4000, np.random.default_rng(870),
                                  resample_states=resample_states, resample_shots=resample_shots)
    assert boot.shape == (4000,)
    ratio = boot.var(ddof=1) / exact
    assert abs(ratio - 1) < 0.10, f"boot var {boot.var(ddof=1):.5f}, exact {exact:.5f}"
    assert abs(boot.mean() - y.mean()) < 4 * math.sqrt(exact / 4000)


@pytest.mark.slow
def test_8_7_hierarchical_bootstrap_variance_population():
    """Production mode (states fixed): mean bootstrap variance over datasets vs
    the analytic formulas in the sigmas.

    With states fixed the target is the sampling variance of the grand mean
    given the states (a_s fixed):
        Var(ybar | states) = (sb^2 + se^2/K) / (S F).
    Taking expectations of the plug-in variances in _exact_boot_var:
        E V_E = (K-1)/K se^2,   E V_B = (F-1)/F (sb^2 + se^2/K)
        default (frames whole):    E Var* = E V_B / (S F)
                                   -> ratio (F-1)/F exactly
        resample_shots=True:       E Var* = (E V_B + E V_E / K) / (S F)
                                   -> ratio (F-1)/F + (K-1)/K se^2/(K sb^2 + se^2).
    S=8, F=10, K=4, sb=1.5, se=3: ratio 0.900 (default) and 1.263 (two-stage).
    M=300 datasets x n_boot=200: one dataset's Var* has relative SD
    ~ sqrt(2/(S(F-1))) = 0.17 plus 0.1 Monte Carlo, so the mean over M has
    ~1.1% -> +-5% is a ~4.5-sigma band.
    """
    S, F, K, sa, sb, se = 8, 10, 4, 1.0, 1.5, 3.0
    true_cond = (sb**2 + se**2 / K) / (S * F)
    EVE = (K - 1) / K * se**2
    EVB = (F - 1) / F * (sb**2 + se**2 / K)
    expected = {False: EVB / (S * F), True: (EVB + EVE / K) / (S * F)}
    assert expected[False] / true_cond == pytest.approx((F - 1) / F, rel=1e-12)
    rng = np.random.default_rng(8700)
    v = {False: [], True: []}
    for _ in range(300):
        groups = _gaussian_design(S, F, K, sa, sb, se, rng)[1]
        for mode in (False, True):
            v[mode].append(hierarchical_bootstrap(groups, _grand_mean, 200, rng,
                                                  resample_shots=mode).var(ddof=1))
    for mode in (False, True):
        msg = (f"resample_shots={mode}: mean boot var {np.mean(v[mode]):.5f}, "
               f"expected {expected[mode]:.5f}, Var(ybar|states) {true_cond:.5f}")
        assert abs(np.mean(v[mode]) / expected[mode] - 1) < 0.05, msg
    assert abs(np.mean(v[False]) / true_cond - 0.9) < 0.05


@pytest.mark.slow
def test_s1_frame_bootstrap_binary_pbeta_variance_ratio():
    """S1: binary p_beta-like data (reviewer probe p7): 16 fixed cells x
    20 frames x 10 binary shots, frame success probability p_f ~ Beta(12, 28).
    Statistic: mean over cells of the per-cell success fraction. The ratio of
    the mean bootstrap variance to the true sampling variance (over 300
    datasets) is (F-1)/F = 0.95 for the default and ~1.67 analytically for
    the two-stage frame -> shot scheme (reviewer: 1.78 measured). Required:
    default in [0.85, 1.07]; two-stage > 1.4.

    Monte-Carlo precision: the true variance is estimated from 300 datasets
    (relative SD sqrt(2/299) = 8%), so the ratio band is ~ +-1.5 SD around
    0.95 on the upper side and the two-stage threshold is > 3 SD away."""
    S, F, K, m, kappa = 16, 20, 10, 0.3, 40.0

    def stat(rep):
        return np.mean([np.mean(np.concatenate([s for _, s in fr])) for _, fr in rep])

    rng = np.random.default_rng(5)
    est, bv = [], {False: [], True: []}
    for _ in range(300):
        p = rng.beta(m * kappa, (1 - m) * kappa, (S, F))
        y = (rng.random((S, F, K)) < p[..., None]).astype(float)
        groups = {s: {f: y[s, f] for f in range(F)} for s in range(S)}
        est.append(y.mean(axis=(1, 2)).mean())
        for mode in (False, True):
            bv[mode].append(hierarchical_bootstrap(groups, stat, 100, rng, resample_shots=mode).var(ddof=1))
    true_var = np.var(est, ddof=1)
    r = {mode: np.mean(bv[mode]) / true_var for mode in (False, True)}
    sb2 = m * (1 - m) / (kappa + 1)
    se2 = m * (1 - m) - sb2
    analytic_two_stage = (F - 1) / F + (K - 1) / K * se2 / (K * sb2 + se2)
    msg = f"ratios {r}, analytic two-stage {analytic_two_stage:.3f}, default {(F - 1) / F:.3f}"
    assert 0.85 <= r[False] <= 1.07, msg
    assert r[True] > 1.4, msg


def test_hierarchical_bootstrap_structure():
    groups = {7: {0: [1.0, 2.0], 1: [3.0]}, 9: {5: [10.0, 20.0, 30.0], 6: [40.0]}}
    seen = []

    def stat(rep):
        seen.append(rep)
        return 0.0

    hierarchical_bootstrap(groups, stat, 50, np.random.default_rng(0))   # default: states fixed
    for rep in seen:
        assert [key for key, _ in rep] == [7, 9]                        # strata by key, in order
        (_, fr7), (_, fr9) = rep
        assert len(fr7) == 2 and len(fr9) == 2                          # frames per state kept
        for key, frs in ((7, fr7), (9, fr9)):
            for fkey, shots in frs:
                assert fkey in groups[key]
                assert set(shots) <= set(groups[key][fkey]) and len(shots) == len(groups[key][fkey])
    assert {fkey for rep in seen for fkey, _ in rep[0][1]} == {0, 1}    # frames really resampled
    seen.clear()
    hierarchical_bootstrap(groups, stat, 200, np.random.default_rng(1), resample_states=True)
    assert {tuple(key for key, _ in rep) for rep in seen} == {(7, 7), (7, 9), (9, 7), (9, 9)}
    for rep in seen:
        for key, frames in rep:
            for fkey, shots in frames:
                assert shots.tolist() == list(groups[key][fkey])   # default: frames kept whole


def test_hierarchical_bootstrap_vector_stat_and_reproducible():
    _, groups = _gaussian_design(4, 2, 3, 1, 1, 1, np.random.default_rng(3))

    def stat(rep):
        v = _all_shots(rep)
        return v.mean(), v.max()

    a = hierarchical_bootstrap(groups, stat, 20, np.random.default_rng(4))
    b = hierarchical_bootstrap(groups, stat, 20, np.random.default_rng(4))
    assert a.shape == (20, 2)
    np.testing.assert_array_equal(a, b)


def test_hierarchical_bootstrap_rejects_empty_levels():
    with pytest.raises(ValueError):
        hierarchical_bootstrap({}, np.mean, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        hierarchical_bootstrap({0: {}}, np.mean, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        hierarchical_bootstrap({0: {1: []}}, np.mean, 10, np.random.default_rng(0))


def test_dataclasses_are_plain():
    assert dataclasses.is_dataclass(TEstimate)
    assert {f.name for f in dataclasses.fields(KonEstimate)} >= {
        "beta", "beta_inf", "kon", "ci", "n_reaction", "n_escape", "n_timeout", "valid"}


# ================================================================= fix wave P6
# Regression tests for fullreview-E-estimate (I1-I3, M1-M9), spec change S1
# and contract K4. Each was run against phase-a@5e84e88 first (RED).


def _two_state_rare(rng, p, n, q=0.3):
    """n shots from each of 2 states; 0 -> 1 with the rare probability p,
    1 -> 0 with probability q."""
    start = np.repeat([0, 1], n)
    end = np.concatenate([(rng.random(n) < p).astype(int), (rng.random(n) >= q).astype(int)])
    return start, end


def test_estimate_T_unobserved_transition_ci_not_zero_width():
    """I1: C_01 = 0 used to give the zero-width CI [0, 0]. Now the upper
    bound is the Clopper-Pearson one, 1 - 0.025^(1/n) (~3.7/n)."""
    n = 100
    start = np.repeat([0, 1], n)
    end = np.concatenate([np.zeros(n, int), np.r_[np.zeros(30, int), np.ones(n - 30, int)]])
    for reversible in (False, True):
        if reversible:  # needs 0 <-> 1 connectivity: one 0 -> 1 shot
            end[0] = 1
        est = estimate_T(start, end, np.ones(2 * n), 2, 1.0, reversible, 200, np.random.default_rng(0))
        x = int(end[:n].sum())
        assert est.ci_low[0, 1] <= est.T[0, 1] <= est.ci_high[0, 1]
        assert est.ci_high[0, 1] >= _jeffreys_cp(x, n)[1] - 1e-12, (reversible, est.ci_high)
    # non-reversible, zero count: exactly the Jeffreys upper bound
    end[0] = 0
    est = estimate_T(start, end, np.ones(2 * n), 2, 1.0, False, 10, np.random.default_rng(0))
    assert est.ci_low[0, 1] == 0.0
    assert est.ci_high[0, 1] == pytest.approx(1 - 0.025 ** (1 / n), rel=1e-10)


def test_estimate_T_nonreversible_ci_is_per_element_jeffreys():
    """I1: with unit weights and no frames the CI of every element is the
    binomial Jeffreys interval of (C_ij, n_i) -- the marginal posterior of
    T_ij under a Beta(1/2, 1/2) prior for 'end in j' vs 'not' -- with the
    Clopper-Pearson bound in a tail holding <= 1 count."""
    rng = np.random.default_rng(11)
    start, end = _pairs_from_chain(P3, 57, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 10, rng)
    for i in range(3):
        for j in range(3):
            x = int(((start == i) & (end == j)).sum())
            lo, hi = _jeffreys_cp(x, 57)
            assert (est.ci_low[i, j], est.ci_high[i, j]) == pytest.approx((lo, hi), rel=1e-9, abs=1e-15)


def test_estimate_T_rare_transition_exact_coverage():
    """I1, exact: for a 2-state row with n = 200 shots, every count x in
    0..n is fed through estimate_T and the exact coverage
    sum_x Binom(x; n, p) 1[p in CI(x)] is computed. Required >= 0.92 for all
    p on a fine grid of expected counts 0.05-50 and in [0.93, 0.99] at
    expected counts 1, 2, 5 (0.981 / 0.984 / 0.931). The old percentile
    bootstrap gave [0, 0] at x = 0, i.e. coverage <= P(x > 0)."""
    n = 200
    lo, hi = np.empty(n + 1), np.empty(n + 1)
    for x in range(n + 1):
        start = np.repeat([0, 1], n)
        end = np.r_[np.ones(x, int), np.zeros(n - x, int), np.zeros(n // 2, int), np.ones(n - n // 2, int)]
        est = estimate_T(start, end, np.ones(2 * n), 2, 1.0, False, 1, np.random.default_rng(0))
        lo[x], hi[x] = est.ci_low[0, 1], est.ci_high[0, 1]
    xs = np.arange(n + 1)

    def cov(p):
        return stats.binom.pmf(xs, n, p)[(lo <= p) & (p <= hi)].sum()

    grid = np.geomspace(0.05, 50, 300) / n
    c = np.array([cov(p) for p in grid])
    assert c.min() >= 0.92, (c.min(), grid[c.argmin()] * n)
    for lam in (1, 2, 5):
        assert 0.93 <= cov(lam / n) <= 0.99, (lam, cov(lam / n))


@pytest.mark.parametrize("reversible", [False, True])
@pytest.mark.parametrize("expected_count", [1, 2, 5])
def test_estimate_T_rare_transition_coverage(expected_count, reversible):
    """I1, Monte Carlo through the full estimator: coverage of a rare T_01 at
    expected counts 1, 2, 5 (n = 200 shots per state; exact coverage of the
    non-reversible interval 0.981 / 0.984 / 0.931). The old percentile
    bootstrap covered 0.63 / 0.83 / 0.92 on these datasets (RED run).
    Required: >= 0.90 and <= 0.995 (not trivially wide); 400 datasets
    (binomial SD ~1.2%). The reversible path needs 0 <-> 1 connectivity, so
    its datasets are conditioned on C_01 > 0 (this removes the x = 0 case,
    whose conditional coverage is then that of x >= 1)."""
    n, n_sets = 200, 400
    p = expected_count / n
    rng, boot = np.random.default_rng(900 + expected_count), np.random.default_rng(1)
    hits = done = 0
    while done < n_sets:
        start, end = _two_state_rare(rng, p, n)
        if reversible and end[:n].sum() == 0:
            continue
        est = estimate_T(start, end, np.ones(2 * n), 2, 1.0, reversible, 40 if reversible else 1, boot)
        hits += est.ci_low[0, 1] <= p <= est.ci_high[0, 1]
        done += 1
    cov = hits / n_sets
    assert 0.90 <= cov <= 0.995, cov


def test_estimate_T_rare_transition_coverage_reversible_3state():
    """I1, reversible path with 3 states: a rare 0 <-> 2 pair (expected
    count ~1.1 / 1.3) inside a reversible chain, 200 datasets x 100
    replicates (binomial SD ~1.5%). The rare elements must reach >= 0.92
    (pre-fix percentile bootstrap on 400 identical datasets: 0.88 / 0.90;
    now 0.995). The other elements use the plain percentile bootstrap of the
    reversible MLE, which covers only ~0.90-0.95 here (T_12: 0.90 at
    n_boot = 200, same before and after the fix), so they get >= 0.85."""
    X = np.array([[60.0, 8.0, 0.25], [8.0, 40.0, 6.0], [0.25, 6.0, 50.0]])
    P = X / X.sum(axis=1, keepdims=True)
    rng, boot = np.random.default_rng(907), np.random.default_rng(908)
    hits = np.zeros((3, 3))
    n_sets = 200
    for _ in range(n_sets):
        start, end = _pairs_from_chain(P, 300, rng)
        est = estimate_T(start, end, np.ones(start.size), 3, 1.0, True, 100, boot)
        hits += (est.ci_low <= P) & (P <= est.ci_high)
    cov = hits / n_sets
    assert min(cov[0, 2], cov[2, 0]) >= 0.92, cov
    assert cov.min() >= 0.85, cov


def test_estimate_T_bootstrap_memory_is_independent_of_units():
    """I2: the stratified bootstrap used to materialise (n_boot, n_units, n)
    per state (here 400 x 2000 x 10 doubles = 64 MB). With multiplicity
    vectors the peak is O(n_boot n^2) plus one bounded block."""
    import tracemalloc

    rng = np.random.default_rng(12)
    n = 10
    P = np.full((n, n), 0.5 / (n - 1)) + np.eye(n) * (0.5 - 0.5 / (n - 1))
    start, end = _pairs_from_chain(P, 2000, rng)
    w = np.ones(start.size)
    tracemalloc.start()
    try:
        estimate_T(start, end, w, n, 1.0, False, 400, rng)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 24e6, f"peak {peak / 1e6:.1f} MB"


def test_ck_bootstrap_memory_is_independent_of_trajectories():
    """I2: ck_test used to materialise (n_boot, n_traj, n, n) per lag
    (300 x 60 x 20 x 20 doubles = 58 MB)."""
    import tracemalloc

    rng = np.random.default_rng(13)
    P = _nn_chain(20, 0.2)
    d = _simulate_dtrajs(P, 60, 400, rng)
    tracemalloc.start()
    try:
        ck_test(d, 1, (2, 3), 20, 300, rng)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 24e6, f"peak {peak / 1e6:.1f} MB"


def _rare_state_dtrajs(seed, n_traj=20, n_steps=1000):
    """Reviewer probe p3: state 2 is entered rarely (0.001/step from 1); all
    trajectories start in state 0."""
    P = np.array([[0.98, 0.02, 0.0], [0.02, 0.979, 0.001], [0.0, 0.01, 0.99]])
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, 1)
    x = np.zeros((n_traj, n_steps), dtype=np.int64)
    u = rng.random(x.shape)
    for t in range(1, n_steps):
        x[:, t] = (u[:, t, None] > cum[x[:, t - 1]]).sum(1)
    return list(x)


def test_ck_rare_state_never_aborts_and_counts_degenerate_replicates():
    """I3: on the reviewer's rare-state chain ck_test raised in 7 of 30
    datasets (a replicate lost state 2's row). It must now run on every
    dataset, report the replicates that lost a row, and (the chain is
    Markov) pass in most of them."""
    results, errors = [], []
    for seed in range(30):
        d = _rare_state_dtrajs(seed)
        if not any((row == 2).any() for row in d):
            continue
        try:
            results.append(ck_test(d, 1, (2, 3), 3, 200, np.random.default_rng(99)))
        except ValueError as exc:
            errors.append(f"seed {seed}: {exc}")
    assert not errors, errors
    passed = n_degen = 0
    n_visited = len(results)
    for res in results:
        assert 0 <= res.n_boot_degenerate <= res.n_boot == 200
        passed += res.passed
        n_degen += res.n_boot_degenerate > 0
    assert n_visited >= 20
    assert n_degen > 0          # the case the old code aborted on really occurs
    assert passed >= 0.8 * n_visited, (passed, n_visited)


def test_ck_unvisited_state_of_common_state_space_is_excluded():
    """I3: a state of the common state space that this variant never visits
    is excluded from the test (and reported), not an error."""
    rng = np.random.default_rng(14)
    d = _simulate_dtrajs(P3, 20, 1000, rng)            # states 0, 1, 2 of a 5-state space
    res = ck_test(d, 1, (2, 3), 5, 200, np.random.default_rng(15))
    assert res.active_set.tolist() == [0, 1, 2]
    assert res.excluded_states.tolist() == [3, 4]
    # identical to running on the 3-state space
    ref = ck_test(d, 1, (2, 3), 3, 200, np.random.default_rng(15))
    assert (res.passed, res.max_dev) == (ref.passed, ref.max_dev)
    passed, D = res                                      # legacy 2-tuple protocol
    assert (passed, D) == (res[0], res[1]) == (res.passed, res.max_dev)


def test_ck_active_set_is_largest_strongly_connected_set():
    """I3: a state that is only entered (never left within the data) or only
    left is not in the strongly connected active set."""
    rng = np.random.default_rng(16)
    d = _simulate_dtrajs(P3, 20, 500, rng)
    d = [np.r_[3, x] for x in d[:10]] + d[10:]           # state 3: left once, never entered
    res = ck_test(d, 1, (2,), 4, 50, np.random.default_rng(0))
    assert res.active_set.tolist() == [0, 1, 2]
    assert res.excluded_states.tolist() == [3]


def test_ck_row_missing_at_long_lag_is_reported():
    """I3: a state visited only in the last few frames is strongly connected
    at lag 1 but has no lag-5 counts; its row is left out of D at that lag
    and reported, instead of an error."""
    rng = np.random.default_rng(23)
    d = [np.r_[x, 1, 2, 1, 2] for x in _simulate_dtrajs(np.array([[0.9, 0.1], [0.1, 0.9]]), 12, 200, rng)]
    res = ck_test(d, 1, (2, 5), 3, 50, np.random.default_rng(0))
    assert res.active_set.tolist() == [0, 1, 2]
    assert res.rows_missing == {5: (2,)}
    assert math.isfinite(res.max_dev)


@pytest.mark.filterwarnings("ignore:ck_test with 2 < 10 trajectories")
def test_ck_single_state_active_set_raises():
    d = [np.zeros(50, dtype=int), np.zeros(40, dtype=int)]
    with pytest.raises(ValueError, match="active set"):
        ck_test(d, 1, (2,), 3, 10, np.random.default_rng(0))


def test_ck_warns_with_few_trajectories():
    """M1: CK size is inflated below ~10 trajectories (22% at 3 x 4000)."""
    rng = np.random.default_rng(17)
    d = _simulate_dtrajs(P3, 3, 2000, rng)
    with pytest.warns(UserWarning, match="10 trajectories"):
        ck_test(d, 1, (2,), 3, 50, rng)


def test_ck_does_not_depend_on_trajectory_order():
    """M3: the result is a function of the set of trajectories."""
    rng = np.random.default_rng(18)
    d = _simulate_dtrajs(P3, 12, 300, rng)
    a = ck_test(d, 1, (2, 3), 3, 100, np.random.default_rng(5))
    b = ck_test(d[::-1], 1, (2, 3), 3, 100, np.random.default_rng(5))
    assert (a.passed, a.max_dev) == (b.passed, b.max_dev)


def test_estimate_T_does_not_depend_on_shot_order():
    """M3: permuting the shots (with or without frame ids) changes nothing."""
    rng = np.random.default_rng(19)
    start, end, fid = _clustered_shots(rng, F=8, K=4)
    w = rng.random(start.size) + 0.5
    perm = rng.permutation(start.size)
    for frames in (None, fid):
        for rev in (False, True):
            a = estimate_T(start, end, w, 2, 1.0, rev, 50, np.random.default_rng(3), frame_ids=frames)
            b = estimate_T(start[perm], end[perm], w[perm], 2, 1.0, rev, 50, np.random.default_rng(3),
                           frame_ids=None if frames is None else frames[perm])
            np.testing.assert_allclose(a.T, b.T, rtol=1e-13)
            np.testing.assert_allclose(a.ci_low, b.ci_low, rtol=1e-12)
            np.testing.assert_allclose(a.ci_high, b.ci_high, rtol=1e-12)
            np.testing.assert_allclose(a.its_ci_low, b.its_ci_low, rtol=1e-12)


def test_estimate_T_its_ci():
    """M2: implied-timescale CI from the bootstrap replicates; coverage of the
    true timescales over 150 datasets (300 shots per state) >= 0.88."""
    lam = np.sort(np.abs(np.linalg.eigvals(P3)))[::-1][1:]
    its_true = -1.0 / np.log(lam)
    rng = np.random.default_rng(20)
    hits = np.zeros(2)
    for _ in range(150):
        start, end = _pairs_from_chain(P3, 300, rng)
        est = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 200, rng)
        assert np.all(est.its_ci_low <= est.its) and np.all(est.its <= est.its_ci_high)
        hits += (est.its_ci_low <= its_true) & (its_true <= est.its_ci_high)
    assert hits.min() / 150 >= 0.88, hits / 150


def test_estimate_T_reversible_sparse_replicates_do_not_abort():
    """Deferred minor #3: a rare connecting pair (one count each way) is lost
    in ~60% of the replicates; those are estimated per connected block and
    counted instead of aborting the whole estimate."""
    start = np.repeat([0, 1], 100)
    end = np.r_[np.zeros(99, int), 1, 0, np.ones(99, int)]
    est = estimate_T(start, end, np.ones(200), 2, 1.0, True, 300, np.random.default_rng(1))
    assert 0 < est.n_boot_degenerate < 300
    assert est.ci_low[0, 1] <= est.T[0, 1] <= est.ci_high[0, 1]


def test_estimate_T_ck_fields_mean_not_tested():
    """Deferred minor #6: ck_passed is None ('not tested'), not False."""
    rng = np.random.default_rng(21)
    start, end = _pairs_from_chain(P3, 50, rng)
    est = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 5, rng)
    assert est.ck_passed is None and math.isnan(est.ck_max_dev)


# ----------------------------------------------------------------- K4 nonfinite


def test_committor_counts_nonfinite_separately():
    """K4: 'nonfinite' is neither an outcome nor a timeout; any occurrence
    makes the estimate invalid and is reported as a count."""
    est = estimate_committor(_records({"A": 50, "B": 49, "nonfinite": 1}))
    assert (est.n_A, est.n_B, est.n_timeout, est.n_nonfinite) == (50, 49, 0, 1)
    assert est.q == pytest.approx(49 / 99, rel=1e-15)
    assert est.ci == pytest.approx(_jeffreys(49, 99), rel=1e-10)
    assert est.timeout_frac == 0.0
    assert est.valid is False
    ok = estimate_committor(_records({"A": 50, "B": 49, "timeout": 1}))
    assert ok.n_nonfinite == 0 and ok.valid is True


def test_kon_counts_nonfinite_separately():
    est = estimate_kon(_records({"reaction": 30, "escape": 70, "nonfinite": 2}), 2.0, 4.0, 0.1)
    assert (est.n_reaction, est.n_escape, est.n_timeout, est.n_nonfinite) == (30, 70, 0, 2)
    assert est.beta == pytest.approx(0.3, rel=1e-15)
    assert est.valid is False


def test_committor_only_nonfinite():
    est = estimate_committor(_records({"nonfinite": 3}))
    assert math.isnan(est.q) and est.n_nonfinite == 3 and est.valid is False


def test_estimate_T_stop_reasons_nonfinite():
    """K4 for T(tau): with stop_reasons given, 'nonfinite' shots are dropped
    from the counts (their end state is meaningless, e.g. -1), counted, and
    valid is False. Any reason other than 'fixed_lag' / 'nonfinite' raises."""
    rng = np.random.default_rng(22)
    start, end = _pairs_from_chain(P3, 100, rng)
    ref = estimate_T(start, end, np.ones(start.size), 3, 1.0, False, 20, np.random.default_rng(1))
    assert ref.valid is True and ref.n_nonfinite == 0
    s2 = np.r_[start, 0, 2]
    e2 = np.r_[end, -1, -1]
    reasons = np.array(["fixed_lag"] * start.size + ["nonfinite"] * 2)
    est = estimate_T(s2, e2, np.ones(s2.size), 3, 1.0, False, 20, np.random.default_rng(1),
                     stop_reasons=reasons)
    np.testing.assert_array_equal(est.T, ref.T)
    assert est.n_nonfinite == 2 and est.valid is False
    with pytest.raises(ValueError, match="timeout"):
        estimate_T(s2, e2, np.ones(s2.size), 3, 1.0, False, 20, np.random.default_rng(1),
                   stop_reasons=np.array(["fixed_lag"] * start.size + ["timeout"] * 2))


@pytest.mark.parametrize("bad", ["weird", "fixed_lag", "pes_uncertain"])
def test_committor_unknown_reason_still_raises(bad):
    with pytest.raises(ValueError, match=bad):
        estimate_committor(_records({"A": 3, "nonfinite": 1, bad: 1}))


# ----------------------------------------------------------------- M4 pooling guards


def _rec_f(stop_reason, frame_id, i, kind="shot"):
    return dataclasses.replace(_rec(stop_reason, 1.0, i), frame_id=frame_id, kind=kind)


def test_committor_rejects_segments_and_pooled_frames():
    """M4: q_B(X) is per configuration. Records from several frames are not
    pooled unless explicitly allowed; WE segments are always rejected."""
    recs = [_rec_f("A", 0, 0), _rec_f("B", 0, 1), _rec_f("B", 1, 2)]
    with pytest.raises(ValueError, match="frame"):
        estimate_committor(recs)
    est = estimate_committor(recs, allow_multiple_frames=True)
    assert est.q == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="segment"):
        estimate_committor([_rec_f("A", 0, 0, kind="segment")])


def test_kon_rejects_segments_and_clustered_frames():
    """M4: k_on pools b-surface configurations (one shot each is fine), but
    several shots per frame from several frames are clustered."""
    single = [_rec_f("reaction" if i % 3 == 0 else "escape", i, i) for i in range(30)]
    est = estimate_kon(single, 2.0, 4.0, 0.1)
    assert est.n_reaction == 10
    clustered = [_rec_f("reaction" if i % 3 == 0 else "escape", i // 2, i) for i in range(30)]
    with pytest.raises(ValueError, match="frame"):
        estimate_kon(clustered, 2.0, 4.0, 0.1)
    estimate_kon(clustered, 2.0, 4.0, 0.1, allow_clustered_frames=True)
    with pytest.raises(ValueError, match="segment"):
        estimate_kon([_rec_f("reaction", 0, 0, kind="segment")], 2.0, 4.0, 0.1)


# ----------------------------------------------------------------- S1 bootstrap


def test_hierarchical_bootstrap_default_keeps_frames_whole():
    """S1: default = states fixed, frames resampled within each state, every
    drawn frame brings its shots unchanged. resample_shots=True restores the
    two-stage frame -> shot scheme."""
    groups = {7: {0: [1.0, 2.0], 1: [3.0]}, 9: {5: [10.0, 20.0, 30.0], 6: [40.0]}}
    seen = []

    def stat(rep):
        seen.append(rep)
        return 0.0

    hierarchical_bootstrap(groups, stat, 100, np.random.default_rng(0))
    for rep in seen:
        for key, frames in rep:
            for fkey, shots in frames:
                assert shots.tolist() == list(groups[key][fkey])
    seen.clear()
    hierarchical_bootstrap(groups, stat, 100, np.random.default_rng(0), resample_shots=True)
    assert any(shots.tolist() != list(groups[9][f]) for rep in seen for f, shots in rep[1][1])


# ---------------------------------------------------------------------------
# Fix wave 2, package L5 (fixreview-p6 N1 + minors, int2-I3 / contract K11)
# ---------------------------------------------------------------------------

import warnings  # noqa: E402

from cytherea.estimate import records_to_groups, records_to_transitions, shot_weights  # noqa: E402
from cytherea.estimate.msm import _row_n_eff  # noqa: E402


def _short_dtrajs(n_traj=12, length=10, seed=0):
    rng = np.random.default_rng(seed)
    cum = np.array([[0.9, 0.1], [0.2, 0.8]]).cumsum(1)
    out = []
    for _ in range(n_traj):
        x = [0]
        for _ in range(length - 1):
            x.append(int((rng.random() > cum[x[-1]]).sum()))
        out.append(np.array(x))
    return out


def test_n1_ck_with_no_data_at_a_requested_lag_raises_instead_of_passing():
    """p6-N1: 10-step trajectories, lag 2, ks (5, 10): no trajectory is
    longer than k*lag, and ck_test returned passed=True, max_dev=0."""
    with pytest.raises(ValueError, match="no data at lag"):
        ck_test(_short_dtrajs(), 2, (5, 10), 2, 100, np.random.default_rng(1))
    with pytest.raises(ValueError, match="lag 20"):  # one empty lag is enough
        ck_test(_short_dtrajs(length=15), 2, (2, 10), 2, 100, np.random.default_rng(1))


def test_n1_ck_warns_when_most_rows_are_missing_at_a_lag():
    d = _short_dtrajs(n_traj=40, length=12, seed=3)
    d[0] = np.concatenate([np.zeros(30, int), np.ones(30, int)])  # one long trajectory
    with pytest.warns(UserWarning, match="rows"):
        ck_test(d, 2, (2, 20), 2, 50, np.random.default_rng(1))


def test_p6_m6_row_n_eff_boundary_branches_hand_computed():
    """p6 minor 6: the clustered boundary elements borrow the row's largest
    design effect; without an interior element the frame-Kish size is used."""
    C_u = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0]])  # 2 frames, 5 unit-weight shots
    got = _row_n_eff(C_u, C_u.sum(axis=1), 5.0, clustered=True)
    assert got == pytest.approx([4.6875, 4.6875, 4.6875])
    C_u = np.array([[3.0, 0.0], [2.0, 0.0]])
    got = _row_n_eff(C_u, C_u.sum(axis=1), 5.0, clustered=True)
    assert got == pytest.approx([25 / 13, 25 / 13])
    assert _row_n_eff(C_u, C_u.sum(axis=1), 5.0, clustered=False) == pytest.approx([5.0, 5.0])


def test_p6_m1_estimate_T_reports_n_eff():
    start = np.repeat([0, 1], 40)
    end = np.r_[np.zeros(30, int), np.ones(10, int), np.ones(40, int)]
    est = estimate_T(start, end, np.ones(80), 2, 1.0, False, 10, np.random.default_rng(0))
    assert est.n_eff.shape == (2, 2) and est.n_eff[0, 0] == pytest.approx(40.0)


def test_p6_m7_hierarchical_bootstrap_refuses_single_frame_states():
    with pytest.raises(ValueError, match="2 frames"):
        hierarchical_bootstrap({"s": {"f0": [1.0, 0.0]}}, lambda rep: 0.0, 5, np.random.default_rng(0))


def _krec(stop_reason, i, frame_id, key_frame_id, frame_weight=None, state=None, weight=1.0,
          final=None):
    meta = {} if frame_weight is None else {"frame_weight": frame_weight, "state": state}
    return dataclasses.replace(
        _rec(stop_reason, weight, i), frame_id=frame_id, key={"frame_id": key_frame_id, "shot_id": i},
        ic_meta=meta, final_state_label=final,
    )


def test_k11_shot_weights_enumerated_design_uses_frame_weight_over_k():
    recs = [_krec("reaction", 0, 0, 0, 1.0), _krec("escape", 1, 1, 1, 9.0), _krec("escape", 2, 1, 1, 9.0)]
    assert shot_weights(recs).tolist() == [1.0, 4.5, 4.5]
    drawn = [_krec("reaction", 0, 0, -1, 1.0), _krec("escape", 1, 1, -1, 9.0)]
    assert shot_weights(drawn).tolist() == [1.0, 1.0]  # already drawn proportional to weight
    with pytest.raises(ValueError, match="frame_id"):
        shot_weights([recs[0], drawn[1]])
    legacy = [_rec("A", 2.0, 0), _rec("B", 1.0, 1)]  # no ic_meta frame weights: record.weight
    assert shot_weights(legacy).tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="frame_weight"):
        shot_weights([recs[0], _krec("escape", 5, 2, 2)])


def test_k11_kon_and_pooled_committor_are_frame_weighted():
    """int2-I3: two frames with weights 1 and 9, one shot each: beta must be
    the weighted 0.1, not the unweighted 0.5."""
    recs = [_krec("reaction", 0, 0, 0, 1.0), _krec("escape", 1, 1, 1, 9.0)]
    est = estimate_kon(recs, b=1.0, q=2.0, D_AB=1.0)
    assert est.beta == pytest.approx(0.1)
    crec = [_krec("B", 0, 0, 0, 1.0), _krec("A", 1, 1, 1, 9.0)]
    assert estimate_committor(crec, allow_multiple_frames=True).q == pytest.approx(0.1)
    assert estimate_kon(recs, 1.0, 2.0, 1.0, weights=[1.0, 1.0]).beta == pytest.approx(0.5)


def test_s1_records_to_groups_and_transitions():
    recs = [
        _krec("fixed_lag", 0, 3, 3, 1.0, state="A", final="A"),
        _krec("fixed_lag", 1, 3, 3, 1.0, state="A", final="B"),
        _krec("fixed_lag", 2, 1, 1, 2.0, state="A", final="A"),
        _krec("fixed_lag", 3, 7, 7, 1.0, state="B", final="B"),
        _krec("nonfinite", 4, 7, 7, 1.0, state="B", final=None),
        _krec("fixed_lag", 5, 8, 8, 1.0, state="B", final="A"),
    ]
    good = recs[:4] + recs[5:]
    groups = records_to_groups(good, lambda r: float(r.final_state_label == "B"))
    assert groups == {"A": {1: [0.0], 3: [0.0, 1.0]}, "B": {7: [1.0], 8: [0.0]}}
    assert list(groups) == ["A", "B"] and list(groups["A"]) == [1, 3]  # canonical order
    with pytest.raises(ValueError, match="nonfinite"):
        records_to_groups(recs, lambda r: 0.0)
    tr = records_to_transitions(recs, {"A": 0, "B": 1})
    assert tr["start_states"].tolist() == [0, 0, 0, 1, 1, 1]
    assert tr["end_states"].tolist() == [0, 1, 0, 1, 1, 0]  # nonfinite: placeholder = start, dropped (K4)
    assert tr["stop_reasons"] == ["fixed_lag"] * 4 + ["nonfinite", "fixed_lag"]
    assert tr["frame_ids"].tolist() == [3, 3, 1, 7, 7, 8]
    assert tr["weights"].tolist() == [0.5, 0.5, 2.0, 0.5, 0.5, 1.0]
    est = estimate_T(**tr, n_states=2, lag=1.0, reversible=False, n_boot=10, rng=np.random.default_rng(0))
    assert est.n_nonfinite == 1 and not est.valid


def test_records_to_transitions_keeps_caller_weights_with_their_records():
    """Review P1: records are put in key-digest order; caller weights must be
    permuted with them, so the input order cannot change T."""
    recs = [
        _krec("fixed_lag", 0, 0, 0, 1.0, state="A", final="A"),
        _krec("fixed_lag", 1, 1, 1, 1.0, state="A", final="B"),
    ]
    w = {recs[0].key_digest: 9.0, recs[1].key_digest: 1.0}
    for order in (recs, recs[::-1]):
        tr = records_to_transitions(order, {"A": 0, "B": 1}, weights=[w[r.key_digest] for r in order])
        by_frame = dict(zip(tr["frame_ids"].tolist(), zip(tr["weights"].tolist(), tr["end_states"].tolist())))
        assert by_frame == {0: (9.0, 0), 1: (1.0, 1)}
        stay = tr["weights"][tr["end_states"] == 0].sum() / tr["weights"].sum()
        assert stay == pytest.approx(0.9)
    with pytest.raises(ValueError, match="one entry per record"):
        records_to_transitions(recs, {"A": 0, "B": 1}, weights=[1.0])


# ------------------------------------------------- CK from multi-lag fixed-lag shots

def _multilag_shots(P, n_frames, shots_per_frame, ks, rng, lump=None, start_pick=None):
    """Shots started at t = 0 from every state of the observed chain, the state
    recorded at each k of `ks`. With `lump`, P is a hidden chain observed
    through `lump` and each frame is a hidden state drawn uniformly from the
    lump (its equilibrium law here); all shots of a frame share it."""
    n_hidden = P.shape[0]
    obs_of = np.arange(n_hidden) if lump is None else lump
    n_obs = int(obs_of.max()) + 1
    cum = np.cumsum(P, axis=1)
    start, ends, frames = [], [], []
    fid = 0
    for s in range(n_obs):
        members = np.flatnonzero(obs_of == s)
        for _ in range(n_frames):
            h0 = rng.choice(members)
            for _ in range(shots_per_frame):
                x, row, t = h0, [], 0
                for k in sorted(ks):
                    while t < k:
                        x = int((rng.random() > cum[x]).sum())
                        t += 1
                    row.append(obs_of[x])
                start.append(s)
                ends.append([row[sorted(ks).index(k)] for k in ks])
                frames.append(fid)
            fid += 1
    return np.array(start), np.array(ends, dtype=np.int64), np.array(frames)


def test_ck_shots_markov_chain_passes_and_lumped_hidden_chain_fails():
    from cytherea.estimate import ck_test_shots

    ks = (1, 2, 5, 10, 20)
    rng = np.random.Generator(np.random.PCG64(41))
    s, e, f = _multilag_shots(P3, 50, 10, ks, rng)
    res = ck_test_shots(s, e, ks, 3, 300, rng, frame_ids=f)
    assert res.passed and res.active_set.tolist() == [0, 1, 2] and res.n_boot_degenerate == 0
    s, e, f = _multilag_shots(P_HIDDEN, 50, 10, ks, rng, lump=LUMP)
    assert not ck_test_shots(s, e, ks, 2, 300, rng, frame_ids=f).passed
    assert not ck_test_shots(s, e, ks, 2, 300, rng).passed


def test_ck_shots_size_over_seeds():
    """A Markov chain passes in >= 85 % of 30 independent campaigns (nominal 95 %)."""
    from cytherea.estimate import ck_test_shots

    ks = (1, 2, 5, 10)
    passed = 0
    for seed in range(30):
        rng = np.random.Generator(np.random.PCG64(1000 + seed))
        s, e, f = _multilag_shots(P3, 20, 5, ks, rng)
        passed += ck_test_shots(s, e, ks, 3, 200, rng, frame_ids=f).passed
    assert passed >= 0.85 * 30, passed


def test_ck_shots_depends_only_on_the_set_of_shots():
    from cytherea.estimate import ck_test_shots

    ks = (1, 3, 6)
    s, e, f = _multilag_shots(P3, 10, 4, ks, np.random.Generator(np.random.PCG64(5)))
    perm = np.random.Generator(np.random.PCG64(6)).permutation(s.size)
    for fids in (None, f):
        a = ck_test_shots(s, e, ks, 3, 100, np.random.Generator(np.random.PCG64(7)), frame_ids=fids)
        b = ck_test_shots(s[perm], e[perm], ks, 3, 100, np.random.Generator(np.random.PCG64(7)),
                          frame_ids=None if fids is None else fids[perm])
        assert (a.passed, a.max_dev) == (b.passed, b.max_dev)


def test_ck_shots_validates_its_input():
    from cytherea.estimate import ck_test_shots

    rng = np.random.Generator(np.random.PCG64(0))
    s = np.array([0, 0, 1, 1])
    e = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    ck_test_shots(s, e, (1, 2), 2, 10, rng)
    with pytest.raises(ValueError, match="must contain 1"):
        ck_test_shots(s, e, (2, 3), 2, 10, rng)
    with pytest.raises(ValueError, match="shape"):
        ck_test_shots(s, e[:, :1], (1, 2), 2, 10, rng)
    with pytest.raises(ValueError, match="state 2"):
        ck_test_shots(s, e, (1, 2), 3, 10, rng)
    with pytest.raises(ValueError, match="outside"):
        ck_test_shots(s, e + 1, (1, 2), 2, 10, rng)
    with pytest.raises(ValueError, match="frame"):
        ck_test_shots(s, e, (1, 2), 2, 10, rng, frame_ids=np.array([0, 1, 1, 2]))
    with pytest.raises(ValueError, match="distinct"):
        ck_test_shots(s, e, (1, 1), 2, 10, rng)


def test_ck_shots_with_tau_only_shots():
    """T(tau) from the long shots plus tau-only shots (A1 14.3, 2026-10-04): a
    Markov chain still passes, the lumped chain still fails, and the extra
    shots tighten T(tau) (the k = 1 deviation stays 0 by construction)."""
    from cytherea.estimate import ck_test_shots

    ks = (1, 2, 5, 10, 20)
    rng = np.random.Generator(np.random.PCG64(43))
    s, e, f = _multilag_shots(P3, 50, 4, ks, rng)
    ts, te, tf = _multilag_shots(P3, 50, 6, (1,), rng)
    res = ck_test_shots(s, e, ks, 3, 300, rng, frame_ids=f,
                        tau_only={"start": ts, "end": te[:, 0], "frame_ids": tf})
    assert res.passed
    s, e, f = _multilag_shots(P_HIDDEN, 50, 4, ks, rng, lump=LUMP)
    ts, te, tf = _multilag_shots(P_HIDDEN, 50, 6, (1,), rng, lump=LUMP)
    assert not ck_test_shots(s, e, ks, 2, 300, rng, frame_ids=f,
                             tau_only={"start": ts, "end": te[:, 0], "frame_ids": tf}).passed
    with pytest.raises(ValueError, match="frame_ids"):
        ck_test_shots(s, e, ks, 2, 10, rng, frame_ids=f, tau_only={"start": ts, "end": te[:, 0]})
    with pytest.raises(ValueError, match="two start states"):
        ck_test_shots(s, e, ks, 2, 10, rng, frame_ids=f,
                      tau_only={"start": 1 - ts, "end": te[:, 0], "frame_ids": tf})


def test_ck_shots_tau_only_size_over_seeds():
    """With tau-only shots the test keeps its size: a Markov chain passes in >= 85 %."""
    from cytherea.estimate import ck_test_shots

    ks = (1, 2, 5, 10)
    passed = 0
    for seed in range(30):
        rng = np.random.Generator(np.random.PCG64(3000 + seed))
        s, e, f = _multilag_shots(P3, 20, 2, ks, rng)
        ts, te, tf = _multilag_shots(P3, 20, 3, (1,), rng)
        passed += ck_test_shots(s, e, ks, 3, 200, rng, frame_ids=f,
                                tau_only={"start": ts, "end": te[:, 0], "frame_ids": tf}).passed
    assert passed >= 0.85 * 30, passed
