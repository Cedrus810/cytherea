"""A0 analytic acceptance, part 1 (Task 9): free diffusion + double-well committors.

Every number here comes from the real engine end to end: IC sampler (with
its gate) -> `run_shot` via `run_batch(n_workers>1)` (sole Store writer) ->
online stop rules -> `estimate_kon` / `estimate_committor`, and is compared
with an exact answer:

9.1  3D free diffusion, NAM b-surface: reaction |r| <= a, start |r| = b = 2a,
     escape |r| = q = 8a, N = 4000. beta_inf = nam_beta_inf(beta) must equal
     a/b within 3 sigma (sigma from estimate_kon's Jeffreys interval mapped
     through the monotone NAM map); the same at dt/2, and the dt -> dt/2
     change must be within 3 sigma of the difference.
9.2  1D double well (barrier 5 kT), 20 points x 1600 shots (plan: 400; raised
     because at 400 the noise-only RMSE is 0.0204, so the 0.03 gate fails by
     chance -- it did once, chi2 48/20 with the fixed seed, while four other
     seeds gave 13-26; docs/reports/A0_part1.md): RMSE of q_B vs the
     exact quadrature committor < 0.03; every point valid (timeouts <= 5%);
     dt-halving spot check at 3 points (2000 shots each at dt and dt/2).
9.3  Mueller-Brown (kT = 10 native), 30 points x 1600 shots (as 9.2): RMSE vs the
     finite-volume backward-Kolmogorov reference < 0.03 (reference grid
     converged: n = 300 vs 600); dt-halving spot check at 3 points.
9.4  offline_replay of every stored record of every campaign above
     reproduces the online stop decision (reason and event_time) exactly.

Marked slow; run explicitly with
    PYTHONPATH=src pytest -m slow -k acceptance_a0 -v
Environment knobs (all optional): CYTHEREA_A0_A0_WORKERS (default 16),
CYTHEREA_A0_A0_WORKDIR (where the sqlite stores go; default a pytest tmp dir;
~5 GB peak, deleted at module teardown unless CYTHEREA_A0_A0_KEEP=1),
CYTHEREA_A0_A0_RESULTS (directory to write a0_part1_results.json and the
Mueller-Brown reference grid for the report plots).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toy_diffusion import nam_sphere  # noqa: E402
from examples.toy_doublewell import doublewell_1d, mueller_brown  # noqa: E402
from examples.toy_doublewell.committor_shots import committor_keys  # noqa: E402
from reference.committor_ref import committor_1d, committor_2d  # noqa: E402
from cytherea.estimate.association import estimate_kon, nam_beta_inf  # noqa: E402
from cytherea.estimate.committor import estimate_committor  # noqa: E402
from cytherea.exec.batch import ShotFailure, run_batch  # noqa: E402
from cytherea.keys import ShotKey, key_digest  # noqa: E402
from cytherea.observe.events import offline_replay  # noqa: E402
from cytherea.store import Store  # noqa: E402

pytestmark = pytest.mark.slow

N_WORKERS = int(os.environ.get("CYTHEREA_A0_A0_WORKERS", "16"))
SEED = 20260930
Z95 = 1.959963984540054
N_SPOT = 2000  # shots per point and per dt in the dt-halving spot checks

RESULTS: dict = {}
CAMPAIGNS: dict = {}  # name -> Campaign, for the 9.4 replay


# ----------------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class Campaign:
    name: str
    store_path: str
    keys: list
    records: list  # records with observables stripped (estimators only need outcome/weight)
    make_stop: object  # picklable zero-arg factory of the campaign's stop rule
    reasons: tuple
    wall_s: float


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    base = os.environ.get("CYTHEREA_A0_A0_WORKDIR")
    if base:
        path = Path(base) / f"a0_part1_{os.getpid()}"
        path.mkdir(parents=True, exist_ok=False)
    else:
        path = tmp_path_factory.mktemp("a0_part1")
    yield path
    if os.environ.get("CYTHEREA_A0_A0_KEEP") != "1":
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def results_sink():
    yield
    out = os.environ.get("CYTHEREA_A0_A0_RESULTS")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "a0_part1_results.json", "w") as fh:
            json.dump(RESULTS, fh, indent=1, default=float)


def _run_campaign(name, keys, shot_fn, make_stop, reasons, workdir, chunk):
    """run_batch over `keys` in chunks (bounded parent memory: run_batch
    returns full records), all into one Store. Returns a Campaign."""
    store_path = str(Path(workdir) / f"{name}.sqlite")
    store = Store(store_path)
    records = []
    t0 = time.perf_counter()
    for i in range(0, len(keys), chunk):
        res = run_batch(keys[i : i + chunk], shot_fn, store, n_workers=N_WORKERS)
        for r in res:
            assert not isinstance(r, ShotFailure), f"IC rejected: {r}"
            records.append(dataclasses.replace(r, observables={}))
    camp = Campaign(name, store_path, list(keys), records, make_stop, reasons,
                    time.perf_counter() - t0)
    CAMPAIGNS[name] = camp
    return camp


def _per_point_committor(records, n_points):
    by_point: dict[int, list] = {i: [] for i in range(n_points)}
    for r in records:
        by_point[r.key["frame_id"]].append(r)
    return {i: estimate_committor(rs) for i, rs in by_point.items() if rs}


def _committor_summary(ests, q_ref, n_shots):
    ids = sorted(ests)
    q_hat = np.array([ests[i].q for i in ids])
    ref = np.array([q_ref[i] for i in ids])
    resid = q_hat - ref
    var = np.clip(ref * (1 - ref), 1e-12, None) / n_shots
    return {
        "q_hat": q_hat.tolist(),
        "q_ref": ref.tolist(),
        "ci": [list(ests[i].ci) for i in ids],
        "n_A": [ests[i].n_A for i in ids],
        "n_B": [ests[i].n_B for i in ids],
        "n_timeout": [ests[i].n_timeout for i in ids],
        "valid": [ests[i].valid for i in ids],
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "mean_resid": float(np.mean(resid)),
        "chi2": float(np.sum(resid**2 / var)),
        "dof": len(ids),
        "expected_rmse_noise": float(np.sqrt(np.mean(var))),
    }


def _spot_check(module, cfg, points, q_ref, name, workdir, chunk):
    """dt-halving at the 3 points whose reference q is closest to 0.2/0.5/0.8:
    N_SPOT shots at dt and at dt/2 (same dt_obs and persistence)."""
    q_ref = np.asarray(q_ref)
    idxs = sorted({int(np.argmin(np.abs(q_ref - p))) for p in (0.2, 0.5, 0.8)})
    out = {"points": idxs, "q_ref": q_ref[idxs].tolist()}
    for tag, dt in (("dt", cfg["dt"]), ("dt2", cfg["dt"] / 2)):
        c = dict(cfg, dt=dt)
        camp = _run_campaign(
            f"{name}_spot_{tag}",
            committor_keys(idxs, N_SPOT, SEED, f"a0_{name}_spot_{tag}"),
            module.make_shot_fn(points, c),
            functools.partial(module.make_stop, c),
            ("A", "B", "timeout"),
            workdir,
            chunk,
        )
        ests = _per_point_committor(camp.records, len(points))
        out[tag] = {"dt": dt, "q": [ests[i].q for i in idxs],
                    "valid": [ests[i].valid for i in idxs], "wall_s": camp.wall_s}
    q1, q2 = np.array(out["dt"]["q"]), np.array(out["dt2"]["q"])
    qbar = 0.5 * (q1 + q2)
    sd = np.sqrt(2 * qbar * (1 - qbar) / N_SPOT)
    out["diff"] = (q2 - q1).tolist()
    out["sigma_diff"] = sd.tolist()
    out["dev_dt2_vs_ref"] = (q2 - np.asarray(out["q_ref"])).tolist()
    out["sigma_single"] = np.sqrt(qbar * (1 - qbar) / N_SPOT).tolist()
    return out


# ----------------------------------------------------------------------------
# 9.1 free diffusion / NAM
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nam_results(workdir):
    base = nam_sphere.CONFIG
    a, b, q, n = base["a"], base["b"], base["q"], base["n_shots"]
    D = nam_sphere.diffusion_coefficient(base)
    out = {"a": a, "b": b, "q": q, "N": n, "D": D,
           "beta_exact_finite_q": nam_sphere.beta_exact(a, b, q),
           "beta_inf_exact": a / b, "runs": {}}
    for tag, dt in (("dt", base["dt"]), ("dt2", base["dt"] / 2)):
        cfg = dict(base, dt=dt)
        # frame_id = -1: a weighted (here uniform) draw of the start direction (K3)
        keys = [ShotKey(SEED, -1, k, f"a0_nam_{tag}") for k in range(n)]
        camp = _run_campaign(
            f"nam_{tag}", keys, nam_sphere.make_shot_fn(cfg),
            functools.partial(nam_sphere.make_stop, cfg),
            ("reaction", "escape", "timeout"), workdir, chunk=1000,
        )
        # several shots share a direction; harmless, the outcome is isotropic
        est = estimate_kon(camp.records, b=b, q=q, D_AB=D, allow_clustered_frames=True)
        sigma_inf = (est.beta_inf_ci[1] - est.beta_inf_ci[0]) / (2 * Z95)
        # plain binomial sigma propagated through d beta_inf / d beta
        omega = b / q
        slope = (1 - omega) / (1 - (1 - est.beta) * omega) ** 2
        sigma_binom = math.sqrt(est.beta * (1 - est.beta) / n) * slope
        out["runs"][tag] = {
            "dt": dt,
            "n_reaction": est.n_reaction, "n_escape": est.n_escape,
            "n_timeout": est.n_timeout, "valid": est.valid,
            "beta": est.beta, "beta_ci": list(est.beta_ci),
            "beta_inf": est.beta_inf, "beta_inf_ci": list(est.beta_inf_ci),
            "sigma_beta_inf": sigma_inf, "sigma_beta_inf_binomial": sigma_binom,
            "z_vs_a_over_b": (est.beta_inf - a / b) / sigma_inf,
            "beta_discrete_prediction": nam_sphere.beta_discrete_prediction(cfg),
            "beta_inf_discrete_prediction": nam_beta_inf(
                nam_sphere.beta_discrete_prediction(cfg), b, q),
            "wall_s": camp.wall_s,
        }
    r1, r2 = out["runs"]["dt"], out["runs"]["dt2"]
    out["dt_change"] = r2["beta_inf"] - r1["beta_inf"]
    out["sigma_dt_change"] = math.hypot(r1["sigma_beta_inf"], r2["sigma_beta_inf"])
    RESULTS["9.1"] = out
    return out


def test_acceptance_a0_9_1_nam_free_diffusion(nam_results):
    r = nam_results
    for tag in ("dt", "dt2"):
        run = r["runs"][tag]
        print(f"\n9.1 [{tag}={run['dt']:g}] beta={run['beta']:.4f} "
              f"(exact finite-q {r['beta_exact_finite_q']:.4f}, discrete-monitoring "
              f"prediction {run['beta_discrete_prediction']:.4f}); beta_inf="
              f"{run['beta_inf']:.4f} +- {run['sigma_beta_inf']:.4f} vs a/b = "
              f"{r['beta_inf_exact']:.4f} (z = {run['z_vs_a_over_b']:+.2f}); "
              f"reaction/escape/timeout = {run['n_reaction']}/{run['n_escape']}/"
              f"{run['n_timeout']}; wall {run['wall_s']:.0f} s")
        assert run["valid"] and run["n_timeout"] == 0
        assert run["n_reaction"] + run["n_escape"] == r["N"]
        assert abs(run["beta_inf"] - r["beta_inf_exact"]) <= 3 * run["sigma_beta_inf"]
    print(f"9.1 dt-halving: d(beta_inf) = {r['dt_change']:+.4f} +- "
          f"{r['sigma_dt_change']:.4f}")
    assert abs(r["dt_change"]) <= 3 * r["sigma_dt_change"]


# ----------------------------------------------------------------------------
# 9.2 1D double well
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dw_results(workdir):
    cfg = doublewell_1d.CONFIG
    pot = doublewell_1d.make_potential(cfg)
    a, b = cfg["A_max"], cfg["B_min"]

    def qref(x):
        return float(committor_1d(pot, cfg["kT"], a, b, [x])[0])

    # 20 points spanning the transition region: q_ref = 0.025 ... 0.975
    targets = np.linspace(0.025, 0.975, cfg["n_points"])
    points = [brentq(lambda x, p=p: qref(x) - p, a, b, xtol=1e-14) for p in targets]
    q_ref = committor_1d(pot, cfg["kT"], a, b, points)

    camp = _run_campaign(
        "dw1d",
        committor_keys(range(len(points)), cfg["n_shots"], SEED, "a0_dw1d"),
        doublewell_1d.make_shot_fn(points, cfg),
        functools.partial(doublewell_1d.make_stop, cfg),
        ("A", "B", "timeout"), workdir, chunk=8000,
    )
    ests = _per_point_committor(camp.records, len(points))
    out = {"points": points, "wall_s": camp.wall_s, "config": cfg,
           **_committor_summary(ests, q_ref, cfg["n_shots"])}
    # persistence sensitivity of the reference: boundaries moved into the
    # basins by sqrt(2 D tau_persist)
    delta = math.sqrt(2 * cfg["kT"] / (cfg["mass"] * cfg["gamma"]) * cfg["tau_persist"])
    q_shift = committor_1d(pot, cfg["kT"], a - delta, b + delta, points)
    out["persistence_shift"] = delta
    out["persistence_sensitivity"] = float(np.max(np.abs(q_shift - q_ref)))
    out["spot"] = _spot_check(doublewell_1d, cfg, points, q_ref, "dw1d", workdir, chunk=12000)
    RESULTS["9.2"] = out
    return out


def _print_committor(tag, r):
    print(f"\n{tag}: RMSE = {r['rmse']:.4f} (noise-only expectation "
          f"{r['expected_rmse_noise']:.4f}), max|resid| = {r['max_abs_resid']:.4f}, "
          f"mean resid = {r['mean_resid']:+.4f}, chi2/dof = {r['chi2']:.1f}/{r['dof']}, "
          f"timeouts = {sum(r['n_timeout'])}, wall {r['wall_s']:.0f} s")
    s = r["spot"]
    for k, i in enumerate(s["points"]):
        print(f"  spot point {i}: q_ref={s['q_ref'][k]:.4f} q(dt)={s['dt']['q'][k]:.4f} "
              f"q(dt/2)={s['dt2']['q'][k]:.4f} diff={s['diff'][k]:+.4f} "
              f"+- {s['sigma_diff'][k]:.4f}")


def _assert_committor(r):
    assert all(r["valid"]), "a point has timeout fraction > 5% (or no resolved shots)"
    assert r["rmse"] < 0.03
    s = r["spot"]
    assert all(s["dt"]["valid"]) and all(s["dt2"]["valid"])
    for k in range(len(s["points"])):
        assert abs(s["diff"][k]) <= 3 * s["sigma_diff"][k]
        assert abs(s["dev_dt2_vs_ref"][k]) <= 3 * s["sigma_single"][k]


def test_acceptance_a0_9_2_doublewell_1d_committor(dw_results):
    _print_committor("9.2 1D double well", dw_results)
    print(f"  persistence sensitivity of reference: {dw_results['persistence_sensitivity']:.4f}")
    _assert_committor(dw_results)


# ----------------------------------------------------------------------------
# 9.3 Mueller-Brown
# ----------------------------------------------------------------------------


def _mb_reference(cfg, n, radius=None):
    c = dict(cfg) if radius is None else dict(cfg, radius=radius)
    return committor_2d(
        mueller_brown.make_potential(c), c["kT"],
        mueller_brown.region_A(c).mask, mueller_brown.region_B(c).mask,
        mueller_brown.REFERENCE_BOUNDS, n,
    )


def _mb_points(ref, cfg, n_points):
    """Transition-region points: on a 0.02 lattice, keep accessible points
    (V <= V(S1) + 2 kT) with 0.02 < q_ref < 0.98; stratify q_ref into
    n_points equal bins and take the lowest-energy point of each bin."""
    pot = mueller_brown.make_potential(cfg)
    gx = np.arange(-1.5, 1.1 + 1e-9, 0.02)
    gy = np.arange(-0.3, 2.0 + 1e-9, 0.02)
    P = np.array([[x, y] for x in gx for y in gy])
    V = np.array([pot.energy_grad(p)[0] for p in P])
    v_s1 = pot.energy_grad(np.array([-0.822, 0.6243]))[0]
    q = ref(P)
    edges = np.linspace(0.02, 0.98, n_points + 1)
    points = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (q > lo) & (q <= hi) & (V <= v_s1 + 2 * cfg["kT"])
        assert m.any(), f"no candidate point with q_ref in ({lo:.3f}, {hi:.3f}]"
        j = np.flatnonzero(m)[np.argmin(V[m])]
        points.append(P[j].tolist())
    return points


@pytest.fixture(scope="module")
def mb_results(workdir):
    cfg = mueller_brown.CONFIG
    t0 = time.perf_counter()
    coarse = _mb_reference(cfg, 300)
    fine = _mb_reference(cfg, 600)
    t_ref = time.perf_counter() - t0
    points = _mb_points(fine, cfg, cfg["n_points"])
    q_ref = fine(np.array(points))
    grid_diff = float(np.max(np.abs(coarse(np.array(points)) - q_ref)))

    camp = _run_campaign(
        "mb",
        committor_keys(range(len(points)), cfg["n_shots"], SEED, "a0_mb"),
        mueller_brown.make_shot_fn(points, cfg),
        functools.partial(mueller_brown.make_stop, cfg),
        ("A", "B", "timeout"), workdir, chunk=12000,
    )
    ests = _per_point_committor(camp.records, len(points))
    out = {"points": points, "wall_s": camp.wall_s, "reference_wall_s": t_ref,
           "grid_n": [300, 600], "grid_diff_at_points": grid_diff, "config": cfg,
           **_committor_summary(ests, q_ref, cfg["n_shots"])}
    delta = math.sqrt(2 * cfg["kT"] / (cfg["mass"] * cfg["gamma"]) * cfg["tau_persist"])
    shrunk = _mb_reference(cfg, 300, radius=cfg["radius"] - delta)
    out["persistence_shift"] = delta
    out["persistence_sensitivity"] = float(
        np.max(np.abs(shrunk(np.array(points)) - coarse(np.array(points)))))
    out["spot"] = _spot_check(mueller_brown, cfg, points, q_ref, "mb", workdir, chunk=12000)
    RESULTS["9.3"] = out
    res_dir = os.environ.get("CYTHEREA_A0_A0_RESULTS")
    if res_dir:
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(res_dir) / "a0_part1_mb_reference.npz",
                            x=fine.x, y=fine.y, q=fine.q, V=fine.V)
    return out


def test_acceptance_a0_9_3_mueller_brown_committor(mb_results):
    _print_committor("9.3 Mueller-Brown", mb_results)
    print(f"  reference grid n=300 vs n=600: max |dq| at points = "
          f"{mb_results['grid_diff_at_points']:.2e}; persistence sensitivity "
          f"{mb_results['persistence_sensitivity']:.4f}")
    assert mb_results["grid_diff_at_points"] < 0.005
    _assert_committor(mb_results)


# ----------------------------------------------------------------------------
# 9.4 offline replay
# ----------------------------------------------------------------------------


def _replay_one(make_stop, rec):
    series = {k: np.asarray(v, dtype=float) for k, v in rec.observables.items()}
    d = offline_replay(make_stop(), series)
    ok = d is not None and d.reason == rec.stop_reason and d.event_time == rec.event_time
    detail = "" if ok else f"offline={d!r} online=({rec.stop_reason!r}, {rec.event_time!r})"
    return rec.key_digest, ok, detail, len(series["t"])


def _replay_campaign(camp):
    digests, mismatches, n_obs = set(), [], 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        inflight: set = set()

        def drain(done):
            nonlocal n_obs
            for f in done:
                dg, ok, detail, n = f.result()
                assert dg not in digests
                digests.add(dg)
                n_obs += n
                if not ok:
                    mismatches.append((dg, detail))

        for reason in camp.reasons:
            for rec in Store(camp.store_path).iter(stop_reason=reason):
                inflight.add(ex.submit(_replay_one, camp.make_stop, rec))
                if len(inflight) >= 4 * N_WORKERS:
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    drain(done)
        drain(wait(inflight).done)
    return digests, mismatches, n_obs


def test_acceptance_a0_9_4_offline_replay_all_records(nam_results, dw_results, mb_results):
    summary = {}
    total = 0
    for name, camp in CAMPAIGNS.items():
        t0 = time.perf_counter()
        digests, mismatches, n_obs = _replay_campaign(camp)
        wall = time.perf_counter() - t0
        expected = {key_digest(k) for k in camp.keys}
        summary[name] = {"records": len(digests), "observations": n_obs,
                         "mismatches": len(mismatches), "wall_s": wall}
        print(f"\n9.4 {name}: replayed {len(digests)} records / {n_obs} observations, "
              f"{len(mismatches)} mismatches, {wall:.0f} s")
        assert digests == expected, "store record set != campaign key set"
        assert not mismatches, mismatches[:5]
        total += len(digests)
    RESULTS["9.4"] = summary
    assert set(CAMPAIGNS) == {"nam_dt", "nam_dt2", "dw1d", "dw1d_spot_dt", "dw1d_spot_dt2",
                              "mb", "mb_spot_dt", "mb_spot_dt2"}
    print(f"9.4 total: {total} records, 100% agreement")
