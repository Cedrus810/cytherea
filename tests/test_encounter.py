"""Task 16: EncounterSampler (b-sphere placement, SO(3) x S^2 orientation).

16.1  1e4 samples: COM distance == b (< 1e-9); the orientation is Haar:
      KS tests of the quaternion's scalar part (density (4/pi) sqrt(1 - w^2)
      on [0, 1]), of the rotation axis and of the image of a fixed body vector
      (z uniform on [-1, 1]), and of the placement direction; p > 0.01 each.
16.2  the same key twice gives bitwise identical ICs.
Plus: clashes are coordinate-level rejections (never redrawn), state labels
restrict the frame draws, input checks, and a run_shot end to end (the
sampler's protocol description enters the protocol hash).

Toy molecules: A = 3 LJ atoms, B = 4 LJ atoms with unequal masses (so the
COM is mass-weighted), analytic LJCluster backend, reduced units.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from cytherea.backends.analytic import AnalyticBackend, LJCluster
from cytherea.engine.shot import ObsSpec, run_shot
from cytherea.ic.encounter import EncounterSampler, quaternion_matrix
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import ICRejectedError
from cytherea.keys import ShotKey
from cytherea.observe.events import BSurface, spec_region

MASSES_A = np.array([1.0, 2.0, 3.0])
MASSES_B = np.array([1.0, 1.5, 4.0, 2.5])
GEOM_A = np.array([[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [0.56, 0.97, 0.0]])
GEOM_B = np.array([[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [0.56, 0.97, 0.0], [0.56, 0.32, 0.91]])


def _frames(geom, n, state=None, prefix="f"):
    rng = np.random.Generator(np.random.PCG64(5))
    return [EnsembleFrame(coordinates=geom + rng.normal(0, 0.02, geom.shape) + 3.0 * k, box=None,
                          topology_ref=prefix, temperature=1.0, weight=1.0 + k, source_id=f"{prefix}{k}",
                          frame_id=k, time=0.0) for k in range(n)]


def _sampler(b=6.0, min_pair_dist=0.8, states=False, **kw):
    fa, fb = _frames(GEOM_A, 3, prefix="A"), _frames(GEOM_B, 4, prefix="B")
    state_of = (lambda f: f.frame_id % 2) if states else None
    backend = AnalyticBackend(LJCluster(7), "baoab", 1e-3, 1.0, gamma=0.1)
    return EncounterSampler(EnsembleFramePool(fa, state_of), EnsembleFramePool(fb, state_of), b,
                            np.concatenate([MASSES_A, MASSES_B]), 1.0, backend, min_pair_dist,
                            kw.pop("label", (0, 1)), **kw)


def _key(i, stage="enc"):
    return ShotKey(global_seed=3, frame_id=-1, shot_id=i, stage=stage)


def _com(x, m):
    return (m[:, None] * x).sum(0) / m.sum()


def _w_cdf(w):
    w = np.clip(w, 0.0, 1.0)
    return (2.0 / math.pi) * (w * np.sqrt(1 - w * w) + np.arcsin(w))


def test_16_1_com_distance_and_haar_orientation():
    s = _sampler()
    n = 10_000
    q = np.empty((n, 4))
    dirs = np.empty((n, 3))
    body = np.empty((n, 3))
    worst = 0.0
    for i in range(n):
        ist, rep = s.sample(_key(i))
        x = ist.state.x
        d = np.linalg.norm(_com(x[3:], MASSES_B) - _com(x[:3], MASSES_A))
        worst = max(worst, abs(d - 6.0))
        q[i] = ist.meta["quaternion"]
        dirs[i] = ist.meta["direction"]
        R = quaternion_matrix(q[i])
        body[i] = R @ np.array([0.0, 0.0, 1.0])
        if i < 50:  # the applied rotation is the recorded one
            fb = s.pool_B.get(ist.meta["frame_id_B"])
            xb0 = fb.coordinates - _com(fb.coordinates, MASSES_B)
            assert np.allclose(x[3:] - _com(x[3:], MASSES_B), xb0 @ R.T, atol=1e-12)
    assert worst < 1e-9
    assert np.all(q[:, 0] >= 0) and np.allclose(np.linalg.norm(q, axis=1), 1.0)
    axis = q[:, 1:] / np.linalg.norm(q[:, 1:], axis=1, keepdims=True)
    p = {
        "w": stats.kstest(q[:, 0], _w_cdf).pvalue,
        "axis_z": stats.kstest(axis[:, 2], stats.uniform(-1, 2).cdf).pvalue,
        "body_z": stats.kstest(body[:, 2], stats.uniform(-1, 2).cdf).pvalue,
        "body_phi": stats.kstest(np.arctan2(body[:, 1], body[:, 0]), stats.uniform(-math.pi, 2 * math.pi).cdf).pvalue,
        "dir_z": stats.kstest(dirs[:, 2], stats.uniform(-1, 2).cdf).pvalue,
        "dir_phi": stats.kstest(np.arctan2(dirs[:, 1], dirs[:, 0]), stats.uniform(-math.pi, 2 * math.pi).cdf).pvalue,
    }
    assert all(v > 0.01 for v in p.values()), p


def test_16_1_ks_detects_a_non_haar_rotation():
    """Power check: uniform Euler angles are NOT Haar; the w test must reject them."""
    rng = np.random.Generator(np.random.PCG64(1))
    a, b_, c = rng.uniform(-math.pi, math.pi, (3, 10_000))
    w = np.abs(np.cos(b_ / 2) * np.cos((a + c) / 2))
    assert stats.kstest(w, _w_cdf).pvalue < 1e-6


def test_16_2_same_key_bitwise_identical():
    s1, s2 = _sampler(), _sampler()
    for i in range(20):
        a, ra = s1.sample(_key(i))
        b, rb = s2.sample(_key(i))
        assert a.state.x.tobytes() == b.state.x.tobytes()
        assert a.state.v.tobytes() == b.state.v.tobytes()
        assert a.meta == b.meta and ra.to_dict() == rb.to_dict()
    x0 = s1.sample(_key(0))[0].state.x
    assert not np.array_equal(x0, s1.sample(_key(1))[0].state.x)
    assert not np.array_equal(x0, s1.sample(_key(0, stage="other"))[0].state.x)


def test_meta_and_frame_weights():
    s = _sampler()
    ist, rep = s.sample(_key(0))
    m = ist.meta
    assert ist.frame_id == -1 and m["frame_id"] == -1 and m["frame_weight"] == 1.0 and m["state"] == [0, 1]
    assert m["b"] == 6.0 and abs(m["com_distance"] - 6.0) < 1e-9 and m["source_id"].startswith("A:")
    assert rep.ok and rep.checks["min_ab_distance"] >= 0.8 and ist.state.t == 0.0
    counts = np.bincount([s.place(_key(i))[1]["frame_id_B"] for i in range(4000)], minlength=4)
    assert stats.chisquare(counts, 4000 * np.array([1, 2, 3, 4]) / 10).pvalue > 0.001


def test_state_labels_restrict_the_frame_draws():
    s = _sampler(states=True, label=(1, 0))
    for i in range(200):
        _, meta = s.place(_key(i))
        assert meta["frame_id_A"] % 2 == 1 and meta["frame_id_B"] % 2 == 0


def test_clash_is_a_coordinate_level_rejection_and_never_redrawn():
    s = _sampler(b=1.0, min_pair_dist=0.8)
    frac = s.clash_fraction([_key(i) for i in range(300)])
    assert frac > 0.5
    k = next(_key(i) for i in range(300) if s.min_ab_distance(s.place(_key(i))[0]) < 0.8)
    for _ in range(2):
        with pytest.raises(ICRejectedError) as ei:
            s.sample(k)
        assert ei.value.reasons == ["encounter_clash"] and ei.value.level == "coordinate"
        assert ei.value.frame_id == -1
    assert _sampler(b=8.0).clash_fraction([_key(i) for i in range(300)]) == 0.0


def test_input_checks():
    with pytest.raises(ValueError, match="frame_id must be -1"):
        _sampler().sample(ShotKey(3, 0, 0, "enc"))
    with pytest.raises(ValueError, match="b must be"):
        _sampler(b=0.0)
    fa = _frames(GEOM_A, 2, prefix="A")
    fb = [EnsembleFrame(coordinates=GEOM_B, box=np.eye(3) * 10, topology_ref="B", temperature=1.0, weight=1.0,
                        source_id="b", frame_id=0, time=0.0)]
    backend = AnalyticBackend(LJCluster(7), "baoab", 1e-3, 1.0, gamma=0.1)
    with pytest.raises(ValueError, match="periodic"):
        EncounterSampler(EnsembleFramePool(fa), EnsembleFramePool(fb), 5.0, np.ones(7), 1.0, backend, 0.5, (0, 0))
    with pytest.raises(ValueError, match="masses must have shape"):
        EncounterSampler(EnsembleFramePool(fa), EnsembleFramePool(_frames(GEOM_B, 1, prefix="B")), 5.0,
                         np.ones(6), 1.0, backend, 0.5, (0, 0))


def test_run_shot_end_to_end_from_the_b_sphere():
    s = _sampler(b=6.0)

    def com_dist(state):
        x = state.x
        return float(np.linalg.norm(_com(x[3:], MASSES_B) - _com(x[:3], MASSES_A)))

    com_dist.spec = "COM distance A-B"
    obs = ObsSpec(fns={"r": com_dist}, dt_obs=0.01, store_stride=1)
    stop = BSurface(spec_region("bound", lambda o: o["r"] <= 2.0, {"r_le": 2.0}), "r", 12.0,
                    tau_persist=0.05, t_max=0.5)
    rec = run_shot(_key(0), s, s.backend, stop, obs, None)
    assert rec.frame_id == -1 and rec.ic_meta["b"] == 6.0
    assert abs(rec.observables["r"][0] - 6.0) < 1e-9
    assert rec.protocol_hash is not None
    rec2 = run_shot(_key(0), _sampler(b=7.0), s.backend, stop, obs, None)
    assert rec2.protocol_hash != rec.protocol_hash  # b enters the protocol


# ------------------------------------------------- review fixes (P2b, P2c, P1-A3)


def test_origin_label_reaches_the_record():
    s = _sampler(label=(0, 1))
    obs = ObsSpec(fns={"r": lambda st: 0.0}, dt_obs=0.01, store_stride=1)
    obs.fns["r"].spec = "zero"
    rec = run_shot(_key(0), s, s.backend, _fixed_stop(), obs, None)
    assert rec.origin_label == (0, 1)


def _fixed_stop():
    from cytherea.observe.events import FixedLag

    return FixedLag(0.02)


def test_both_pools_must_share_one_temperature():
    fa = _frames(GEOM_A, 2, prefix="A")
    fb = [EnsembleFrame(coordinates=GEOM_B, box=None, topology_ref="B", temperature=2.0, weight=1.0,
                        source_id="b", frame_id=0, time=0.0)]
    backend = AnalyticBackend(LJCluster(7), "baoab", 1e-3, 1.0, gamma=0.1)
    m = np.concatenate([MASSES_A, MASSES_B])
    with pytest.raises(ValueError, match="different temperatures"):
        EncounterSampler(EnsembleFramePool(fa), EnsembleFramePool(fb), 6.0, m, 1.0, backend, 0.5, (0, 0))
    fb1 = [dataclasses.replace(fb[0], temperature=1.0)]
    EncounterSampler(EnsembleFramePool(fa), EnsembleFramePool(fb1), 6.0, m, 1.0, backend, 0.5, (0, 0))
    with pytest.raises(ValueError, match="does not match kT"):
        EncounterSampler(EnsembleFramePool(fa), EnsembleFramePool(fb1), 6.0, m, 1.0, backend, 0.5, (0, 0),
                         boltzmann_constant=0.5)


def _a3_record(i, r, Q, reason, event_time):
    from cytherea.store import ShotRecord

    t = [float(k) for k in range(len(r))]
    return ShotRecord(
        key_digest=f"{i:064x}", key={"global_seed": 0, "frame_id": -1, "shot_id": i, "stage": "a3"},
        kind="shot", frame_id=-1, origin_label=(0, 0), ic_validity={}, stop_rule_kind="b_surface",
        stop_reason=reason, event_time=event_time, physics_config_hash="0" * 64, backend_provenance={},
        code_version="test", observables={"t": t, "r": list(r), "Q": list(Q)}, final_state_label=None,
        observables_thinned=False)


def test_a3_analysis_flags_nonfinite_and_degenerate_intervals():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.encounter_pair import shoot as S

    n = 40
    react = _a3_record(0, [5.0] * n, [0.0] * 5 + [0.5] * (n - 5), "reaction", 5.0)
    bad = _a3_record(1, [5.0] * 9 + [math.nan], [0.0] * 10, "nonfinite", 9.0)
    res = S.analyze_records([react, bad], n_draw=2000)
    assert res["valid"] is False and res["q2"]["n_nonfinite"] == 1 and res["q1"]["n_nonfinite"] == 1
    assert res["q2"]["n_reaction"] == 1 and res["diff_ci_note"] is not None
    lo, hi = res["q2"]["beta_ci"]
    assert lo < 0.5 < hi  # Jeffreys, not a degenerate [1, 1]
    dlo, dhi = res["diff_ci95"]
    assert dlo < 0.0 < dhi and dhi - dlo > 0.2  # the difference interval does not collapse either
    # an escape that crosses q1 = 10 nm on the way to q2 = 15 nm escapes at q1 in the replay
    esc = _a3_record(2, list(np.linspace(5.0, 15.0, n)), [0.0] * n, "escape", float(n - 1))
    res = S.analyze_records([react, esc], n_draw=2000)
    assert res["valid"] is True
    assert (res["q1"]["n_reaction"], res["q1"]["n_escape"]) == (1, 1) == (res["q2"]["n_reaction"], res["q2"]["n_escape"])


def test_a3_difference_interval_covers_zero_when_q_does_not_matter():
    """Same outcome at q1 and q2 for every shot: the beta_inf difference comes only
    from the NAM correction and its interval has finite width on any n."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.encounter_pair import shoot as S

    o = ["reaction"] * 3 + ["escape"] * 27
    lo, hi = S.diff_interval(o, o, 10.0, 15.0, n_draw=4000)
    assert lo < hi
    lo1, hi1 = S.diff_interval(["reaction"], ["reaction"], 10.0, 15.0, n_draw=4000)
    assert hi1 - lo1 > 0.05


def _shoot_module():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.encounter_pair import shoot as S

    return S


@pytest.mark.parametrize("o1, o2", [("reaction", "escape"), ("reaction", "timeout"), ("reaction", "nonfinite"),
                                    ("timeout", "reaction"), ("timeout", "escape"), ("nonfinite", "escape")])
def test_a3_paired_outcomes_refuse_impossible_pairs(o1, o2):
    """q1 < q2 replay one trajectory with one reaction rule: a reaction at q1 is a
    reaction at q2, and a shot that never left q1 (timeout / non-finite) cannot
    react or escape at q2 (handoff BUG-03)."""
    S = _shoot_module()
    with pytest.raises(ValueError, match="impossible"):
        S.diff_interval(["escape", o1], ["escape", o2], 10.0, 15.0, n_draw=100)


def test_a3_paired_posterior_q1_marginal_is_jeffreys_and_q2_follows_the_continuation():
    """The q1 outcome ~ Dirichlet(n + 1/2), so beta(q1) ~ Beta(n_R1 + 1/2, n_E1 + 1/2) as
    in the one-sided estimate; beta(q2) = (r1 + e1 rho) / (r1 + e1 (rho + eps)) with the
    continuation from q1 (rho, eps) ~ Dirichlet(m + 1/2) over the q1 escapes only."""
    from scipy import stats

    S = _shoot_module()
    o1 = ["reaction"] * 2 + ["escape"] * 25 + ["timeout"] * 3
    o2 = ["reaction"] * 2 + ["reaction"] * 3 + ["escape"] * 20 + ["timeout"] * 2 + ["timeout"] * 3
    b1, b2 = S.paired_posterior(o1, o2, np.random.Generator(np.random.PCG64(7)), 200_000)
    qs = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    assert np.allclose(np.quantile(b1, qs), stats.beta(2.5, 25.5).ppf(qs), atol=3e-3)
    assert np.all(b2 >= b1)  # every q1 reaction is a q2 reaction
    rng = np.random.Generator(np.random.PCG64(8))
    p1 = rng.dirichlet([2.5, 25.5, 3.5], 200_000)
    c = rng.dirichlet([3.5, 20.5, 2.5], 200_000)
    ref = (p1[:, 0] + p1[:, 1] * c[:, 0]) / (p1[:, 0] + p1[:, 1] * (c[:, 0] + c[:, 1]))
    assert np.allclose(np.quantile(b2, qs), np.quantile(ref, qs), atol=3e-3)


def test_a3_difference_interval_coverage_with_known_truth():
    """Shots simulated from the q1 -> q2 outcome tree with known probabilities: the
    95 % credible interval of beta_inf(q2) - beta_inf(q1) covers the true difference
    in >= 90 % of 400 replicates of 30 shots, for a rare and a common reaction."""
    S = _shoot_module()
    b, q1, q2, n, reps = 5.0, 10.0, 15.0, 30, 400
    rng = np.random.Generator(np.random.PCG64(2026))
    names = np.array(["reaction", "escape", "timeout"])
    for p1, cont in (((0.03, 0.95, 0.02), (0.02, 0.97, 0.01)), ((0.3, 0.68, 0.02), (0.2, 0.79, 0.01))):
        r1, e1 = p1[0], p1[1]
        beta1 = r1 / (r1 + e1)
        beta2 = (r1 + e1 * cont[0]) / (r1 + e1 * (cont[0] + cont[1]))
        truth = S.nam_beta_inf_array(beta2, b, q2) - S.nam_beta_inf_array(beta1, b, q1)
        hits = 0
        for k in range(reps):
            a1 = rng.choice(3, size=n, p=p1)
            a2 = np.where(a1 == 1, rng.choice(3, size=n, p=cont), a1)
            lo, hi = S.diff_interval(list(names[a1]), list(names[a2]), q1, q2, seed=k, n_draw=2000)
            hits += lo <= truth <= hi
        assert hits / reps >= 0.90, (p1, cont, hits / reps)


def test_a3_analysis_reports_the_prior_sensitivity_of_the_difference():
    S = _shoot_module()
    n = 40
    recs = [_a3_record(i, list(np.linspace(5.0, 15.0, n)), [0.0] * n, "escape", float(n - 1)) for i in range(5)]
    recs.append(_a3_record(5, [5.0] * n, [0.0] * 5 + [0.5] * (n - 5), "reaction", 5.0))
    res = S.analyze_records(recs, n_draw=4000)
    assert "credible" in res["diff_ci_method"]
    lo, hi = res["diff_ci95"]
    ulo, uhi = res["diff_ci95_uniform_prior"]
    assert lo < hi and ulo < uhi and (lo, hi) != (ulo, uhi)
    from cytherea.estimate.association import nam_beta_inf

    betas = [0.0, 0.2, 1.0]
    assert S.nam_beta_inf_array(betas, 5.0, 15.0).tolist() == pytest.approx([nam_beta_inf(x, 5.0, 15.0) for x in betas])
