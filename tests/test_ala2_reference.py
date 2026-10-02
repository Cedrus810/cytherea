"""Tests for the A1 alanine-dipeptide reference scripts (Task 14a).

The scripts under examples/alanine_dipeptide/ are plain OpenMM + deeptime and
deliberately independent of cytherea.  Two layers:

* fast: analyze_ref's state assignment and MSM on a synthetic 3-state Markov
  chain with a known transition matrix (implied timescales are analytic).
* slow (``-m slow``): build_system + ref_long on the CPU platform for a few ps,
  including a simulated hard crash between checkpoints and ``--resume``;
  checks file formats, frame contiguity and that analyze_ref runs end to end.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "alanine_dipeptide"
sys.path.insert(0, str(EX))

import ala2_common as C  # noqa: E402
import analyze_ref as A  # noqa: E402

# One point well inside each core box (phi, psi) and one non-core point.
CORE_CENTRES = {0: (-120.0, 150.0), 1: (-70.0, -40.0), 2: (60.0, 45.0)}
NONCORE = (-80.0, 40.0)


def test_core_boxes_do_not_overlap_and_centres_are_inside():
    grid = np.linspace(-180, 180, 721)
    phi, psi = np.meshgrid(grid, grid)
    A.core_labels(phi.ravel(), psi.ravel())  # raises on overlap
    for s, (f, p) in CORE_CENTRES.items():
        assert A.core_labels(np.array([f]), np.array([p]))[0] == s
    assert A.core_labels(np.array([NONCORE[0]]), np.array([NONCORE[1]]))[0] == -1


def test_transition_based_assignment_keeps_last_core_and_drops_lead():
    core = np.array([-1, -1, 0, 0, -1, -1, 1, -1, 0, -1, 2])
    dtraj, lead = A.transition_based_assignment(core)
    assert lead == 2
    assert dtraj.tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 2]


def test_transition_based_assignment_with_initial_label_for_shots():
    # 14b: a shot started in core 1 that leaves the core immediately keeps label 1
    core = np.array([-1, -1, 0, -1, -1, 2, -1])
    lab, lead = C.transition_based_assignment(core, initial_label=1)
    assert lead == 0
    assert lab.tolist() == [1, 1, 0, 0, 0, 2, 2]
    lab, _ = C.transition_based_assignment(np.array([-1, -1]), initial_label=2)
    assert lab.tolist() == [2, 2]
    with pytest.raises(ValueError):
        C.transition_based_assignment(core, initial_label=-1)
    # the analysis uses the very same shared definition
    assert A.core_labels is C.core_labels and A.CORE_BOXES_DEG is C.CORE_BOXES_DEG


def test_frame_index_roundtrip(tmp_path):
    fi = C.FrameIndex(500, 5000)
    assert fi.dcd_frame_to_step(0) == 5000
    assert fi.dcd_frame_to_record(0) == 10
    assert fi.record_to_dcd_frame(20) == 1
    k = np.arange(100)
    assert np.array_equal(fi.record_to_dcd_frame(fi.dcd_frame_to_record(k)), k)
    assert fi.step_to_time_ps(5000) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        fi.record_to_dcd_frame(5)       # record 5 (5 ps) has no DCD frame
    with pytest.raises(ValueError):
        fi.step_to_dcd_frame(0)         # no DCD frame at step 0
    (tmp_path / "run.json").write_text(json.dumps({"phipsi_interval_steps": 500, "dcd_interval_steps": 5000}))
    assert C.FrameIndex.from_run(tmp_path).dcd_frame_to_record(3) == 40


def _write_synthetic_run(tmp_path: Path, T: np.ndarray, n: int, noncore_every: int = 0) -> Path:
    rng = np.random.Generator(np.random.PCG64(12345))
    cum = np.cumsum(T, axis=1)
    u = rng.random(n)
    s = np.empty(n, dtype=np.int64)
    s[0] = 0
    for t in range(1, n):
        s[t] = int(np.searchsorted(cum[s[t - 1]], u[t], side="right"))
    phi = np.array([CORE_CENTRES[k][0] for k in s]) + rng.normal(0, 3, n)
    psi = np.array([CORE_CENTRES[k][1] for k in s]) + rng.normal(0, 3, n)
    if noncore_every:
        # Replace isolated frames by a non-core point *without* a state change:
        # transition-based assignment must make them invisible.
        idx = np.arange(1, n - 1, noncore_every)
        idx = idx[(s[idx - 1] == s[idx]) & (s[idx + 1] == s[idx])]
        phi[idx], psi[idx] = NONCORE
    rec = np.zeros(n, dtype=C.PHIPSI_DTYPE)
    rec["step"] = np.arange(n) * 500
    rec["phi"], rec["psi"] = phi, psi
    run = tmp_path / "synthetic"
    run.mkdir()
    rec.tofile(run / "phipsi.bin")
    C.write_phipsi_meta(run / "phipsi.json", [0, 1, 2, 3], [1, 2, 3, 4], 500)
    return run


def test_analysis_recovers_analytic_timescales_on_markov_chain(tmp_path):
    # Reversible 3-state chain at 1 ps resolution (detailed balance w.r.t. pi).
    pi = np.array([0.6, 0.3, 0.1])
    K = np.array([[0.0, 0.02, 0.002], [0.04, 0.0, 0.003], [0.012, 0.009, 0.0]])
    assert np.allclose(pi[:, None] * K, (pi[:, None] * K).T)
    T = K + np.diag(1 - K.sum(axis=1))
    lam = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    t_exact = [-1.0 / math.log(l) for l in lam[1:]]

    run = _write_synthetic_run(tmp_path, T, 400_000, noncore_every=7)
    rc = A.main(["--run", str(run), "--lags-ps", "1,2,5,10,20", "--n-boot", "40", "--n-blocks", "10"])
    assert rc == 0
    res = json.loads((run / "analysis.json").read_text())
    assert res["n_leading_frames_dropped"] == 0
    assert res["obs_interval_ps"] == C.OBS_INTERVAL_PS and res["obs_interval_matches_state_contract"] is True
    assert res["tba_converged_lag_ps"] == 1.0
    assert res["msm"]["lag_ps"] == 1.0
    for row in res["implied_timescales"]:
        for i in range(2):
            assert row["timescales_ps"][i] == pytest.approx(t_exact[i], rel=0.08), row
            assert row["ci95_lo_ps"][i] <= row["timescales_ps"][i] <= row["ci95_hi_ps"][i]
        assert row["ci_valid"] == [True, True]
    pops = res["msm"]["stationary_population"]
    for s, name in enumerate(A.STATE_NAMES):
        assert pops[name]["value"] == pytest.approx(pi[s], abs=0.02)
    ck = res["ck_test"]
    assert ck["pass"] is True
    # the CK horizon reaches the slowest timescale (not just k = 1..5)
    assert ck["horizon_reaches_t2"] is True and max(ck["k"]) * ck["tau_ps"] >= ck["t2_ps"]
    assert (run / "its.png").stat().st_size > 10_000
    # all bootstrap replicates see all 3 states here -> none excluded
    assert all(r["n_boot_excluded_active_set_mismatch"] == 0 for r in res["implied_timescales"])
    # core-start counting drops start frames outside cores (here: isolated mid-stay
    # frames, an unphysical construction -- see the bridge test for the real sign).
    cs = {r["lag_ps"]: r for r in res["core_start"]["implied_timescales"]}
    assert all(r["n_counts"] < res["implied_timescales"][0]["n_counts"] for r in cs.values())
    for lag in (5.0, 10.0, 20.0):
        for i in range(2):
            assert cs[lag]["timescales_ps"][i] == pytest.approx(t_exact[i], rel=0.08), cs[lag]
    # core-start ITS are already flat below 10 ps here, but the 14b shooting lag has an
    # explicit lower bound (10 ps) and the core-start summary is evaluated AT that lag
    assert res["core_start"]["converged_lag_ps"] < 10.0
    assert res["shoot_lag_ps"] == 10.0 and res["shoot_lag"]["min_lag_ps"] == 10.0
    assert res["core_start"]["msm"]["lag_ps"] == res["shoot_lag_ps"]
    assert res["core_start"]["ck_test"]["tau_ps"] == res["shoot_lag_ps"]
    assert res["contract_14b"]["shoot_lag_ps"] == 10.0
    assert res["contract_14b"]["obs_interval_ps"] == 1.0
    # R40: this run's short lag list is not the 14b grid -> its shoot lag is diagnostic
    assert res["contract_14b"]["lag_grid_matches_contract"] is False
    assert res["contract_14b"]["lag_grid_ps"] == list(C.SHOOT_LAG_GRID_PS)
    assert res["contract_14b"]["requested_lags_ps"] == [1.0, 2.0, 5.0, 10.0, 20.0]


def test_bootstrap_excludes_replicates_with_a_different_active_set(tmp_path):
    # states 0 <-> 1 fast; one single excursion into state 2 in the middle of the run:
    # most resamples of 10 blocks miss it and would report the 0<->1 time as "t2"
    T = np.array([[0.98, 0.02, 0.0], [0.04, 0.96, 0.0], [0.0, 0.0, 1.0]])
    run = _write_synthetic_run(tmp_path, T, 100_000)
    rec = C.load_phipsi(run / "phipsi.bin")
    mid = slice(55_000, 55_400)  # inside one block (edges every 10 000)
    rec["phi"][mid], rec["psi"][mid] = CORE_CENTRES[2]
    rec.tofile(run / "phipsi.bin")
    rc = A.main(["--run", str(run), "--lags-ps", "1,2,5", "--n-boot", "50", "--n-blocks", "10"])
    assert rc == 0
    res = json.loads((run / "analysis.json").read_text())
    for row in res["implied_timescales"]:
        assert row["active_states"] == A.STATE_NAMES
        n_ex = row["n_boot_excluded_active_set_mismatch"]
        assert 0 < n_ex < 50 and row["n_boot_used"] + n_ex == 50
        # t2 is the slow 2 <-> rest process; its CI must not be dragged down to the fast one
        t_fast = -row["lag_ps"] / math.log(np.sort(np.abs(np.linalg.eigvals(T[:2, :2])))[0] ** row["lag_frames"])
        assert row["ci95_lo_ps"][0] > 5 * t_fast
        # > 5 % of the replicates excluded -> the (conditional) CI is flagged invalid
        assert row["ci_valid"][0] is (n_ex / 50 <= 0.05)
    assert any(r["ci_valid"][0] is False for r in res["implied_timescales"])


def test_count_matrix_core_start_exact():
    # frame:        0  1  2   3   4  5  6   7  8
    core = np.array([0, 0, -1, -1, 1, 1, -1, 0, 0])
    tba, lead = C.transition_based_assignment(core)
    assert lead == 0 and tba.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]
    # TBA counts at lag 2: pairs (t, t+2) for t = 0..6
    c = A.count_matrix(tba, tba, 2)
    assert c.tolist() == [[2, 2, 0], [2, 1, 0], [0, 0, 0]]
    # core-start counts at lag 2: only t with core[t] >= 0, i.e. t = 0, 1, 4, 5
    #   t=0: 0 -> tba[2]=0 ; t=1: 0 -> tba[3]=0 ; t=4: 1 -> tba[6]=1 ; t=5: 1 -> tba[7]=0
    c = A.count_matrix(core, tba, 2)
    assert c.tolist() == [[2, 0, 0], [1, 1, 0], [0, 0, 0]]
    # lag 1: core-start windows can never jump core -> core through a bridge frame
    c = A.count_matrix(core, tba, 1)
    assert c.tolist() == [[3, 0, 0], [0, 2, 0], [0, 0, 0]]


def _bridge_run(tmp_path: Path, n_coarse: int = 300_000, bridge: int = 4, p_recross: float = 0.01,
                seed: int = 2024) -> Path:
    """Coarse 3-state chain; EVERY state change passes through ``bridge`` non-core
    frames (as real transitions pass through the psi ~ 0..100 bridge), plus
    occasional non-core excursions that return to the same core (recrossings)."""
    pi = np.array([0.6, 0.3, 0.1])
    K = np.array([[0.0, 0.004, 0.0004], [0.008, 0.0, 0.0006], [0.0024, 0.0018, 0.0]])
    assert np.allclose(pi[:, None] * K, (pi[:, None] * K).T)
    T = K + np.diag(1 - K.sum(axis=1))
    rng = np.random.Generator(np.random.PCG64(seed))
    cum = np.cumsum(T, axis=1)
    u = rng.random(n_coarse)
    rc = rng.random(n_coarse) < p_recross
    s = np.empty(n_coarse, dtype=np.int64)
    s[0] = 0
    for t in range(1, n_coarse):
        s[t] = int(np.searchsorted(cum[s[t - 1]], u[t], side="right"))
    frames = []
    for t in range(n_coarse):
        if t and s[t] != s[t - 1]:
            frames.extend([-1] * bridge)          # committed crossing through the bridge
        elif rc[t]:
            frames.extend([-1] * bridge)          # excursion that returns (recrossing)
        frames.append(int(s[t]))
    lab = np.array(frames)
    n = lab.size
    phi = np.where(lab >= 0, np.array([CORE_CENTRES.get(int(k), NONCORE)[0] for k in lab]), NONCORE[0])
    psi = np.where(lab >= 0, np.array([CORE_CENTRES.get(int(k), NONCORE)[1] for k in lab]), NONCORE[1])
    phi = phi + rng.normal(0, 2, n)
    psi = psi + rng.normal(0, 2, n)
    rec = np.zeros(n, dtype=C.PHIPSI_DTYPE)
    rec["step"] = np.arange(n) * 500
    rec["phi"], rec["psi"] = phi, psi
    run = tmp_path / "bridge"
    run.mkdir()
    rec.tofile(run / "phipsi.bin")
    C.write_phipsi_meta(run / "phipsi.json", [0, 1, 2, 3], [1, 2, 3, 4], 500)
    return run


def test_core_start_bias_is_high_and_summary_is_at_its_own_lag(tmp_path):
    run = _bridge_run(tmp_path)
    lags = "1,2,5,10,20,50,100,200"
    rc = A.main(["--run", str(run), "--lags-ps", lags, "--n-boot", "40", "--n-blocks", "10"])
    assert rc == 0
    res = json.loads((run / "analysis.json").read_text())
    tba = {r["lag_ps"]: r for r in res["implied_timescales"]}
    cs = {r["lag_ps"]: r for r in res["core_start"]["implied_timescales"]}
    # at 1 ps no core-start window can reach another core: degenerate active set
    assert len(cs[1.0]["active_states"]) < 3 and cs[1.0]["active_complete"] is False
    # physical sign of the short-tau bias: core-start t2 is biased HIGH (the excluded
    # windows start on the bridge, where committed crossings are over-represented) ...
    assert cs[5.0]["timescales_ps"][0] > 1.2 * tba[5.0]["timescales_ps"][0]
    assert cs[10.0]["timescales_ps"][0] > tba[10.0]["timescales_ps"][0]
    # ... and converges to the TBA value at long tau
    assert cs[100.0]["timescales_ps"][0] == pytest.approx(tba[100.0]["timescales_ps"][0], rel=0.1)
    # I1: the core-start summary is evaluated at the core-start (shooting) lag, never at
    # the TBA lag where it is degenerate
    assert res["tba_converged_lag_ps"] is not None and res["tba_converged_lag_ps"] <= 2.0
    sl = res["shoot_lag_ps"]
    assert sl is not None and sl >= 10.0 and sl >= res["core_start"]["converged_lag_ps"]
    summ = res["core_start"]["msm"]
    assert summ["lag_ps"] == sl
    assert summ["active_states"] == A.STATE_NAMES
    assert summ["transition_matrix"] is not None and summ["timescales_ps"][0] is not None
    # N2: the design-independent row-normalised T (what 14b compares) with CIs
    Cm = np.array(summ["count_matrix"])
    P = np.array(summ["transition_matrix_row_normalised"])
    assert np.allclose(P, Cm / Cm.sum(axis=1, keepdims=True))
    lo = np.array(summ["transition_matrix_row_normalised_ci95_lo"])
    hi = np.array(summ["transition_matrix_row_normalised_ci95_hi"])
    assert np.all(lo <= P + 1e-12) and np.all(P <= hi + 1e-12) and np.all(hi - lo > 0)
    assert summ["row_normalised_ci_valid_per_row"] == [True, True, True]
    assert "row_normalised" in res["contract_14b"]["compare_against"]
    assert res["core_start"]["ck_test"]["tau_ps"] == sl
    assert res["core_start"]["ck_test"]["horizon_reaches_t2"] is True

    # forcing the degenerate lag gives an explicit null + reason, not a silent T = [[1]]
    out2 = tmp_path / "forced"
    rc = A.main(["--run", str(run), "--out", str(out2), "--lags-ps", lags, "--n-boot", "10",
                 "--n-blocks", "10", "--shoot-lag-ps", "1"])
    assert rc == 0
    res2 = json.loads((out2 / "analysis.json").read_text())
    assert res2["shoot_lag_ps"] == 1.0 and res2["shoot_lag"]["below_min_lag"] is True
    assert res2["core_start"]["msm"] is None
    assert "degenerate" in res2["core_start"]["msm_null_reason"]


def _rows(lags, t2, rel_common=0.25, rel_indep=0.01, n_boot=400, seed=0):
    """ITS rows with paired bootstrap replicates: a large noise component SHARED by
    all lags (same resampled blocks) plus a small independent one."""
    rng = np.random.Generator(np.random.PCG64(seed))
    common = rng.normal(0, rel_common, n_boot)
    rows = []
    for l, t in zip(lags, t2):
        b = t * np.exp(common + rng.normal(0, rel_indep, n_boot))
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({"lag_ps": float(l), "timescales_ps": [float(t), np.nan], "ci95_lo_ps": [lo, np.nan],
                     "ci95_hi_ps": [hi, np.nan], "active_complete": True, "ci_valid": [True, False],
                     "_boot_t2": b})
    return rows


def test_converged_lag_keeps_lags_beyond_t2_and_detects_rising_its():
    # rising ITS (the F-ala2 M1 synthetic): 114.7 -> 123.8 -> 142.5 -> 185.2 ps; the
    # old rule dropped tau > t2 and declared convergence at 20 ps
    rows = _rows([20, 100, 200, 500], [114.7, 123.8, 142.5, 185.2])
    lag, info = A.converged_lag(rows, tol=0.10)
    assert lag is None
    assert info["lag_exceeds_t2_ps"] == [200.0, 500.0]
    # flat ITS where the only larger lag has tau > t2 (real core-start 100/200 ps case):
    # the old rule excluded 200 ps and returned None
    rows = _rows([50, 100, 200], [214.5, 184.6, 187.7])
    lag, _ = A.converged_lag(rows, tol=0.10)
    assert lag == 100.0
    # an explicit lower bound is honoured
    rows = _rows([1, 2, 5, 10, 20], [100, 101, 99, 100, 102])
    assert A.converged_lag(rows, tol=0.10)[0] == 1.0
    assert A.converged_lag(rows, tol=0.10, min_lag_ps=10.0)[0] == 10.0
    # candidates with a degenerate active set or an invalid CI are not eligible
    rows[3]["active_complete"] = False
    assert A.converged_lag(rows, tol=0.10, min_lag_ps=10.0)[0] is None


def test_converged_lag_uses_the_paired_ratio_not_the_marginal_ci():
    # t2 drops 150 -> 120 -> 118: 150 lies inside the (wide, shared-noise) MARGINAL CIs
    # of the larger lags, but the paired ratio 150/120 = 1.25 is tight and excludes 1
    rows = _rows([10, 20, 50], [150, 120, 118], rel_common=0.25, rel_indep=0.02)
    assert rows[1]["ci95_lo_ps"][0] <= 150 <= rows[1]["ci95_hi_ps"][0]   # marginal CI would accept
    assert A.converged_lag(rows, tol=0.10)[0] == 20.0
    # a difference beyond --its-tol that is NOT significant in the paired ratio is accepted
    rows = _rows([10, 20, 50], [150, 120, 118], rel_common=0.0, rel_indep=0.3)
    assert A.converged_lag(rows, tol=0.10)[0] == 10.0
    assert A.converged_lag(rows, tol=0.10, use_ci=False)[0] == 20.0
    # a larger lag with an invalid CI does not take part in the comparison
    rows = _rows([10, 20, 50], [150, 150, 118], rel_common=0.25, rel_indep=0.02)
    assert A.converged_lag(rows, tol=0.10)[0] is None
    rows[2]["ci_valid"] = [False, False]
    lag, info = A.converged_lag(rows, tol=0.10)
    assert lag == 10.0 and info["left_out_invalid_ci_or_incomplete_ps"] == [50.0]


def test_converged_lag_accept_adds_a_condition_per_candidate():
    rows = _rows([1, 10, 20, 50], [100.0, 100.0, 100.0, 100.0], rel_common=0.0)
    seen = []

    def accept(lag_ps):
        seen.append(lag_ps)
        return lag_ps >= 20.0

    lag, info = A.converged_lag(rows, tol=0.10, min_lag_ps=10.0, accept=accept)
    assert lag == 20.0 and seen == [10.0, 20.0]
    assert info["rejected_by_accept_ps"] == [10.0]
    assert A.converged_lag(rows, tol=0.10, accept=lambda _l: False)[0] is None


def _two_bridge_run(tmp_path: Path, n_coarse: int = 300_000, seed: int = 5) -> Path:
    """The full-A1 pattern: a SLOW 0 <-> 2 process (alphaL, few events, 1-frame bridge)
    sets t2, a FAST 0 <-> 1 process (C7eq <-> alphaR) sets t3 and every one of its
    crossings -- and its frequent recrossing excursions -- spends 6 frames on the
    non-core bridge.  Core-start t2 is flat from 10 ps on, but core-start t3 is still
    biased high there (~2x at 10 ps), so a t2-only lag rule picks a lag whose
    core-start CK fails on the C7eq/alphaR diagonal."""
    pi = np.array([0.7, 0.26, 0.04])
    K = np.zeros((3, 3))
    K[0, 1], K[0, 2] = 0.004, 0.00005
    K[1, 0], K[2, 0] = K[0, 1] * pi[0] / pi[1], K[0, 2] * pi[0] / pi[2]
    T = K + np.diag(1 - K.sum(axis=1))
    rng = np.random.Generator(np.random.PCG64(seed))
    cum = np.cumsum(T, axis=1)
    u = rng.random(n_coarse)
    exc = rng.random(n_coarse) < 0.01
    s = np.empty(n_coarse, dtype=np.int64)
    s[0] = 0
    for t in range(1, n_coarse):
        s[t] = int(np.searchsorted(cum[s[t - 1]], u[t], side="right"))
    frames = []
    for t in range(n_coarse):
        if t and s[t] != s[t - 1]:
            frames.extend([-1] * (1 if 2 in (s[t], s[t - 1]) else 6))
        elif exc[t] and s[t] < 2:
            frames.extend([-1] * 6)
        frames.append(int(s[t]))
    lab = np.array(frames)
    n = lab.size
    phi = np.where(lab >= 0, np.array([CORE_CENTRES.get(int(k), NONCORE)[0] for k in lab]), NONCORE[0])
    psi = np.where(lab >= 0, np.array([CORE_CENTRES.get(int(k), NONCORE)[1] for k in lab]), NONCORE[1])
    rec = np.zeros(n, dtype=C.PHIPSI_DTYPE)
    rec["step"] = np.arange(n) * 500
    rec["phi"], rec["psi"] = phi + rng.normal(0, 2, n), psi + rng.normal(0, 2, n)
    run = tmp_path / "two_bridge"
    run.mkdir()
    rec.tofile(run / "phipsi.bin")
    C.write_phipsi_meta(run / "phipsi.json", [0, 1, 2, 3], [1, 2, 3, 4], 500)
    return run


def test_shooting_lag_requires_the_core_start_ck_to_pass(tmp_path):
    """Full-A1 finding (947 ns): with alphaL visited, t2 is the alphaL process and its
    core-start ITS is flat from 10 ps, but the C7eq <-> alphaR t3 is still biased high
    there and the core-start CK -- which 14b repeats on the shots (14.3) -- fails.
    The shooting lag must also pass the core-start CK."""
    run = _two_bridge_run(tmp_path)
    rc = A.main(["--run", str(run), "--lags-ps", "1,2,5,10,20,50,100,200,500", "--n-boot", "40",
                 "--n-blocks", "10"])
    assert rc == 0
    res = json.loads((run / "analysis.json").read_text())
    sb = res["shoot_lag"]
    t2_only = sb["t2_only_lag_ps"]
    assert t2_only is not None and t2_only <= 20.0
    scan = {r["lag_ps"]: r for r in sb["ck_scan"]}
    assert scan[t2_only]["pass"] is False
    sl = res["shoot_lag_ps"]
    assert sl is not None and sl > t2_only and scan[sl]["pass"] is True
    assert all(r["pass"] is not True for l, r in scan.items() if l < sl)
    ck = res["core_start"]["ck_test"]
    assert ck["tau_ps"] == sl and ck["pass"] is True and ck["horizon_reaches_t2"] is True
    # the bias that the CK catches: core-start t3 at the t2-only lag vs at the chosen lag
    cs = {r["lag_ps"]: r for r in res["core_start"]["implied_timescales"]}
    tba = {r["lag_ps"]: r for r in res["implied_timescales"]}
    assert cs[t2_only]["timescales_ps"][1] > 1.2 * tba[t2_only]["timescales_ps"][1]
    assert cs[sl]["timescales_ps"][1] == pytest.approx(tba[sl]["timescales_ps"][1], rel=0.15)


def test_shooting_lag_is_invariant_to_lag_list_density(tmp_path):
    """Short data (wide marginal CIs, as in the first 19.5 ns of A1).  The round-1 rule
    compared t2(tau_i) with the MARGINAL CI of each larger lag and chose 50 ps on the
    sparse list but 30 ps on the dense one; the paired-ratio rule chooses the same lag
    on both.  (Beyond this, a choice can only move by grid granularity: it is always a
    listed lag, and extra lags only ADD constraints to the lags both lists share.)
    The CK condition of the shooting lag does not depend on the list either (on data
    this short it rejects every candidate, so both lists give null)."""
    run = _bridge_run(tmp_path, n_coarse=30_000, seed=3)
    got = {}
    for name, lags in (("sparse", "1,2,5,10,20,50,100,200"),
                       ("dense", "1,2,5,10,20,30,50,70,100,150,200")):
        out = tmp_path / name
        assert A.main(["--run", str(run), "--out", str(out), "--lags-ps", lags, "--n-boot", "100",
                       "--n-blocks", "10"]) == 0
        res = json.loads((out / "analysis.json").read_text())
        got[name] = (res["shoot_lag"]["t2_only_lag_ps"], res["shoot_lag_ps"])
    assert got["sparse"][0] is not None and got["sparse"] == got["dense"], got


def test_ck_horizon_reaches_t2_and_detects_hidden_memory(tmp_path):
    """F-ala2 I2 reproducer: hidden 4-state chain lumped into 3 labels (C7eq has memory).
    With k = 1..5 at tau = 1 ps the CK test passed vacuously; with a horizon >= t2 it fails."""
    K = np.zeros((4, 4))
    K[0, 2] = 0.05
    K[2, 0] = 0.05
    K[0, 1] = 0.002
    K[1, 0] = 0.002
    K[2, 3] = 0.004
    K[3, 2] = 0.008
    T = K + np.diag(1 - K.sum(1))
    lab = np.array([0, 0, 1, 2])
    rng = np.random.Generator(np.random.PCG64(7))
    n = 400_000
    cum = np.cumsum(T, 1)
    u = rng.random(n)
    h = np.empty(n, np.int64)
    h[0] = 1
    for t in range(1, n):
        h[t] = np.searchsorted(cum[h[t - 1]], u[t], side="right")
    s = lab[h]
    rec = np.zeros(n, dtype=C.PHIPSI_DTYPE)
    rec["step"] = np.arange(n) * 500
    rec["phi"] = [CORE_CENTRES[k][0] for k in s]
    rec["psi"] = [CORE_CENTRES[k][1] for k in s]
    run = tmp_path / "memory"
    run.mkdir()
    rec.tofile(run / "phipsi.bin")
    C.write_phipsi_meta(run / "phipsi.json", [0, 1, 2, 3], [1, 2, 3, 4], 500)
    rc = A.main(["--run", str(run), "--lags-ps", "1,2,5,10,20,50,100,200,500", "--n-boot", "40",
                 "--n-blocks", "10", "--msm-lag-ps", "1"])
    assert rc == 0
    res = json.loads((run / "analysis.json").read_text())
    ck = res["ck_test"]
    assert ck["tau_ps"] == 1.0
    assert ck["horizon_reaches_t2"] is True and max(ck["k"]) * ck["tau_ps"] >= ck["t2_ps"]
    assert ck["pass"] is False
    # the rising ITS must not be reported as converged
    assert res["tba_converged_lag_ps"] is None or res["tba_converged_lag_ps"] >= 200.0


def test_user_lag_beyond_block_length_is_refused(tmp_path):
    T = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    run = _write_synthetic_run(tmp_path, T, 2000)
    with pytest.raises(SystemExit, match="block"):
        A.main(["--run", str(run), "--n-blocks", "10", "--lags-ps", "1,2,5", "--msm-lag-ps", "150"])
    with pytest.raises(SystemExit, match="block"):
        A.main(["--run", str(run), "--n-blocks", "10", "--lags-ps", "1,2,5", "--shoot-lag-ps", "150"])


def test_14b_lag_grid_is_a_pinned_constant_and_the_default(tmp_path):
    # ruling R40: 14b fixes tau on the DEFAULT grid, pinned in ala2_common
    assert C.SHOOT_LAG_GRID_PS == (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
    args = A.parse_args(["--run", str(tmp_path)])
    assert args.lags_ps == "1,2,5,10,20,50,100,200,500,1000,2000"
    assert C.lag_grid_is_contract(float(x) for x in args.lags_ps.split(","))
    assert C.lag_grid_is_contract(reversed(C.SHOOT_LAG_GRID_PS))
    dense = (1, 2, 5, 10, 20, 30, 50, 70, 100, 150, 200, 300)
    assert not C.lag_grid_is_contract(dense)
    assert not C.lag_grid_is_contract(C.SHOOT_LAG_GRID_PS[:-1])


def test_obs_interval_contract_and_shot_labelling():
    assert C.OBS_INTERVAL_PS == 1.0
    assert C.OBS_INTERVAL_STEPS * C.TIMESTEP_PS == pytest.approx(C.OBS_INTERVAL_PS)
    C.check_obs_interval(1.0)
    with pytest.raises(ValueError, match="1 ps"):
        C.check_obs_interval(5.0)
    phi = np.array([CORE_CENTRES[1][0], NONCORE[0], CORE_CENTRES[0][0], NONCORE[0]])
    psi = np.array([CORE_CENTRES[1][1], NONCORE[1], CORE_CENTRES[0][1], NONCORE[1]])
    assert C.label_shot(phi, psi, start_label=1, interval_ps=1.0).tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError):
        C.label_shot(phi, psi, start_label=1, interval_ps=10.0)


def test_initial_label_must_agree_with_a_core_start_frame():
    with pytest.raises(ValueError, match="disagrees"):
        C.transition_based_assignment(np.array([0, -1, 1]), initial_label=1)
    lab, _ = C.transition_based_assignment(np.array([1, -1, 0]), initial_label=1)
    assert lab.tolist() == [1, 1, 0]


def test_analysis_rejects_noncontiguous_steps(tmp_path):
    T = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    run = _write_synthetic_run(tmp_path, T, 1000)
    rec = C.load_phipsi(run / "phipsi.bin")
    rec = np.concatenate([rec[:500], rec[499:]])  # one duplicated frame
    rec.tofile(run / "phipsi.bin")
    with pytest.raises(SystemExit, match="not contiguous"):
        A.main(["--run", str(run)])


def test_dcd_truncate_roundtrip(tmp_path):
    import mdtraj
    from openmm import app, unit, Vec3

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("X", ch)
    for k in range(3):
        top.addAtom(f"C{k}", app.element.carbon, res)
    path = tmp_path / "t.dcd"
    with open(path, "wb") as fh:
        dcd = app.DCDFile(fh, top, 0.002, 10, 10)
        for f in range(5):
            pos = [Vec3(0.1 * f, 0.2 * k, 0.3) for k in range(3)] * unit.nanometer
            box = [Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)] * unit.nanometer
            dcd.writeModel(pos, periodicBoxVectors=box)
    assert C.dcd_n_frames(path) == 5
    C.truncate_dcd(path, 3)
    assert C.dcd_n_frames(path) == 3
    with open(path, "r+b") as fh:
        dcd = app.DCDFile(fh, top, 0.002, 10, 10, append=True)
        pos = [Vec3(9.0, 0.2 * k, 0.3) for k in range(3)] * unit.nanometer
        dcd.writeModel(pos, periodicBoxVectors=[Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2)] * unit.nanometer)
    pdb = tmp_path / "t.pdb"
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(top, [Vec3(0, 0, 0)] * 3 * unit.nanometer, fh)
    t = mdtraj.load(str(path), top=str(pdb))
    assert t.n_frames == 4
    assert np.allclose(t.xyz[:3, 0, 0], [0.0, 0.1, 0.2], atol=1e-6)
    assert t.xyz[3, 0, 0] == pytest.approx(9.0)


def _run(script: str, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EX / script), *args], capture_output=True, text=True, env=env)


@pytest.mark.slow
def test_cpu_build_ref_crash_resume_and_analyze(tmp_path):
    env = dict(os.environ, OPENMM_CPU_THREADS="8")
    sysdir = tmp_path / "sys"
    p = _run("build_system.py", "--out", str(sysdir), "--platform", "CPU", "--nvt-ps", "1", "--npt-ps", "3",
             "--npt-discard-ps", "1", "--relax-ps", "1", env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    for f in ("system.xml", "state.xml", "topology.pdb", "build.json", "equil_volume.csv"):
        assert (sysdir / f).stat().st_size > 0
    info = json.loads((sysdir / "build.json").read_text())
    assert info["net_charge_e"] == 0 and info["n_water"] > 500
    assert "MonteCarloBarostat" not in (sysdir / "system.xml").read_text()

    run = tmp_path / "ref"
    common = ["--out", str(run), "--total-ns", "0.012", "--seed", "11", "--platform", "CPU",
              "--system-dir", str(sysdir), "--dcd-ps", "2", "--checkpoint-ns", "0.004", "--flush-ps", "1",
              "--report-ps", "2"]
    p = _run("ref_long.py", *common, "--crash-at-ps", "7", env=env)
    assert p.returncode == -9, p.stdout + p.stderr  # SIGKILL
    assert len(C.load_phipsi(run / "phipsi.bin")) == 8  # steps 0..7 ps, past the 4 ps checkpoint
    assert C.dcd_n_frames(run / "traj.dcd") == 3
    assert json.loads((run / "checkpoint.json").read_text())["step"] == 2000

    # new runs record input hashes + scripts' git commit ...
    run_meta = json.loads((run / "run.json").read_text())
    assert set(run_meta["input_sha256"]) == {"system.xml", "state.xml", "topology.pdb"}
    assert run_meta["input_sha256"]["state.xml"] == C.sha256_file(sysdir / "state.xml")
    assert "commit" in run_meta["scripts_git"]
    # ... but a run.json WITHOUT them (like the live 1 us reference, created before the
    # fields existed) must stay resumable
    del run_meta["input_sha256"], run_meta["scripts_git"]
    (run / "run.json").write_text(json.dumps(run_meta, indent=2))

    p = _run("ref_long.py", *common, "--resume", env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    log_text = (run / "progress.log").read_text()
    assert "dropped 3 phi/psi records and 1 DCD frames" in log_text
    assert "no input_sha256 (older run)" in log_text

    rec = C.load_phipsi(run / "phipsi.bin")
    assert (run / "phipsi.bin").stat().st_size == 13 * C.PHIPSI_DTYPE.itemsize
    assert rec["step"].tolist() == [500 * k for k in range(13)]
    assert np.all(np.isfinite(rec["phi"])) and np.all(np.abs(rec["phi"]) <= 180)
    meta = json.loads((run / "phipsi.json").read_text())
    assert meta["record_bytes"] == 16 and meta["interval_ps"] == 1.0
    assert C.dcd_n_frames(run / "traj.dcd") == 6
    run_meta = json.loads((run / "run.json").read_text())
    assert run_meta["gamma_per_ps"] == 0.1 and run_meta["seed"] == 11 and len(run_meta["segments"]) == 2
    assert json.loads((run / "checkpoint.json").read_text())["step"] == 6000

    import mdtraj

    t = mdtraj.load(str(run / "traj.dcd"), top=str(sysdir / "topology.pdb"))
    _, phi = mdtraj.compute_phi(t)
    ref = rec[rec["step"] % 1000 == 0][1:]
    dphi = (np.degrees(phi[:, 0]) - ref["phi"] + 180) % 360 - 180
    assert np.abs(dphi).max() < 1e-3

    # resuming a finished run is a no-op; a mismatched seed is refused
    p = _run("ref_long.py", *common, "--resume", env=env)
    assert p.returncode == 0 and len(C.load_phipsi(run / "phipsi.bin")) == 13
    bad = list(common)
    bad[bad.index("--seed") + 1] = "12"
    p = _run("ref_long.py", *bad, "--resume", env=env)
    assert p.returncode != 0 and "seed" in p.stderr

    # checkpoint.chk missing -> automatic fallback to checkpoint.prev.chk (the 8 ps one)
    assert json.loads((run / "checkpoint.json").read_text())["step"] == 6000
    (run / "checkpoint.chk").unlink()
    ext = list(common)
    ext[ext.index("--total-ns") + 1] = "0.014"
    p = _run("ref_long.py", *ext, "--resume", env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    log_text = (run / "progress.log").read_text()
    assert "using FALLBACK checkpoint.prev.chk" in log_text
    assert "RESUME from checkpoint step 4000" in log_text
    rec = C.load_phipsi(run / "phipsi.bin")
    assert rec["step"].tolist() == [500 * k for k in range(15)]
    assert C.dcd_n_frames(run / "traj.dcd") == 7
    assert (run / "checkpoint.chk").exists() and (run / "checkpoint.prev.chk").exists()
    assert not list(run.glob("*.tmp"))

    # checkpoint.chk UNREADABLE (not missing) -> fallback to .prev (the 12 ps one) and the
    # bad file is moved aside, so the next rotation cannot copy it over the good .prev
    assert json.loads((run / "checkpoint.json").read_text())["step"] == 7000
    good_prev = (run / "checkpoint.prev.chk").read_bytes()
    (run / "checkpoint.chk").write_bytes(b"garbage, not an OpenMM checkpoint")
    ext[ext.index("--total-ns") + 1] = "0.016"
    p = _run("ref_long.py", *ext, "--resume", env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    log_text = (run / "progress.log").read_text()
    assert "could not load checkpoint.chk" in log_text
    assert "moved the unreadable checkpoint.chk to checkpoint.bad-" in log_text
    assert "RESUME from checkpoint step 6000" in log_text
    bad = list(run.glob("checkpoint.bad-*.chk"))
    assert len(bad) == 1 and bad[0].read_bytes().startswith(b"garbage")
    # the good .prev we resumed from survives the next rotation unchanged
    assert (run / "checkpoint.prev.chk").read_bytes() == good_prev
    rec = C.load_phipsi(run / "phipsi.bin")
    assert rec["step"].tolist() == [500 * k for k in range(17)]
    assert C.dcd_n_frames(run / "traj.dcd") == 8

    # a changed input file is refused on resume of a run that recorded hashes
    run2 = tmp_path / "ref2"
    c2 = list(common)
    c2[c2.index("--out") + 1] = str(run2)
    c2[c2.index("--total-ns") + 1] = "0.004"
    p = _run("ref_long.py", *c2, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    with open(sysdir / "state.xml", "a") as fh:
        fh.write("\n")
    p = _run("ref_long.py", *c2, "--resume", env=env)
    assert p.returncode != 0 and "sha256 mismatch" in p.stderr

    p = _run("analyze_ref.py", "--run", str(run), "--n-boot", "5", env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((run / "analysis.json").read_text())["n_records"] == 17
    assert (run / "its.png").exists()


def test_several_runs_and_pieces_are_independent_trajectories(tmp_path):
    """Multi-trajectory reference (user, 2026-10-01: run A1 as parallel
    independent trajectories): DIR:K pieces and several --run dirs are
    separate trajectories -- no transition, count or bootstrap block crosses
    their edges -- and they recover the same chain as one long run."""
    pi = np.array([0.6, 0.3, 0.1])
    K = np.array([[0.0, 0.02, 0.002], [0.04, 0.0, 0.003], [0.012, 0.009, 0.0]])
    T = K + np.diag(1 - K.sum(axis=1))
    lam = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    t_exact = [-1.0 / math.log(l) for l in lam[1:]]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _write_synthetic_run(tmp_path / "a", T, 200_000)
    b = _write_synthetic_run(tmp_path / "b", T, 200_000)
    out = tmp_path / "combined"
    rc = A.main(["--run", f"{a}:2", "--run", str(b), "--out", str(out), "--lags-ps", "1,2,5,10,20",
                 "--n-boot", "40", "--n-blocks", "12", "--skip-ps", "100"])
    assert rc == 0
    res = json.loads((out / "analysis.json").read_text())
    assert res["n_trajectories"] == 3
    assert res["trajectory_ns"] == pytest.approx([99.95, 99.95, 199.9], abs=0.01)
    assert res["bootstrap"]["n_blocks"] == 12
    for row in res["implied_timescales"]:
        for i in range(2):
            assert row["timescales_ps"][i] == pytest.approx(t_exact[i], rel=0.1), row
    with pytest.raises(SystemExit, match="--out"):
        A.main(["--run", str(a), "--run", str(b)])
    assert A._run_spec("/x/y:3") == (Path("/x/y"), 3) and A._run_spec("/x/y") == (Path("/x/y"), 1)
