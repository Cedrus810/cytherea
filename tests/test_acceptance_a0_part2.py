"""A0 analytic acceptance, part 2 (Task 11): the absorbing network on real WE data.

Every number comes from the engine end to end: `run_we` (BinnedWE with the
label constraint, analytic overdamped backend, AbsorbingAB at the well
minima) writes segment records into one Store per run; `cytherea.network`
reads them back. Toys and parameters: examples/toy_network/milestone_we.py.
Each campaign is 2 x N_RUNS independent WE runs: the first N_RUNS are the
training set, the others the held-out set of the Markov test (one WE run is
one bootstrap unit). N_RUNS = 48: the error of a network committor is
dominated by run-to-run noise that shifts the whole curve (per-run P_B sd
0.053 on dw2d); with 24 + 24 runs four independent dw2d campaigns gave RMSE
0.026, 0.017, 0.018 and 0.006 (docs/reports/A0_part2.md), too close to 0.03.
``expected_rmse_noise`` (bootstrap over runs) is reported next to each RMSE.

11.2  DoubleWell2D (barrier 2 kT, separable), 12 x-milestones: network
      committor B[:, B] (all runs) vs the exact 1D committor at the
      milestone centres, RMSE < 0.03. The Markov test is reported.
11.3  ChannelDoubleWell2D (channel barriers 1 and 4 kT, wall 10 kT), label
      = sign(y0), every label pooled into one x-milestone network:
      markov_test.passed is False.
11.4  The same data, augmented network (one per label): markov_test.passed
      is True for each label, and each label's network committor vs the 2D
      finite-volume reference at (x_m, y0), RMSE < 0.03 (reference grid
      n = 400 checked against n = 200).

Marked slow; run explicitly with
    PYTHONPATH=src pytest -m slow tests/test_acceptance_a0_part2.py -v -s
Environment: CYTHEREA_A0_A0_WORKERS (default 16), CYTHEREA_A0_A0_RESULTS (directory
for a0_part2_results.json, used by docs/reports/A0_part2.md).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toy_network import milestone_we as T  # noqa: E402
from cytherea.network import build_transitions, markov_test, solve_absorption  # noqa: E402
from cytherea.network.absorb import _counts, _normalise, _scan  # noqa: E402
from cytherea.store import Store  # noqa: E402

pytestmark = pytest.mark.slow

N_WORKERS = int(os.environ.get("CYTHEREA_A0_A0_WORKERS", "16"))
N_RUNS = 48
SEED = 20261002
N_BOOT = 2000
RESULTS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def results_sink():
    yield
    out = os.environ.get("CYTHEREA_A0_A0_RESULTS")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "a0_part2_results.json", "w") as fh:
            json.dump(RESULTS, fh, indent=1, default=float)


@pytest.fixture(scope="module")
def campaigns(tmp_path_factory):
    """{toy: (train records, heldout records, run summaries)}."""
    work = tmp_path_factory.mktemp("a0_part2")
    jobs = [(name, SEED + 1000 * k + i, f"{name}-{i:02d}", str(work / f"{name}-{i:02d}.sqlite"))
            for k, name in enumerate(("dw2d", "channel")) for i in range(2 * N_RUNS)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        summaries = list(ex.map(T.run_replica, *zip(*jobs)))
    out = {}
    for name in ("dw2d", "channel"):
        runs = [s for s in summaries if s["run_id"].startswith(name)]
        assert all(s["valid"] for s in runs)
        recs = [list(Store(s["store"]).iter(kind="segment")) for s in runs]
        out[name] = ([r for rs in recs[:N_RUNS] for r in rs], [r for rs in recs[N_RUNS:] for r in rs], runs)
        RESULTS[f"{name}_runs"] = {
            "n_runs": len(runs), "n_segments": sum(s["n_segments"] for s in runs),
            "max_final_weight": max(s["final_weight"] for s in runs),
            "wall_s_per_run": float(np.mean([s["wall_s"] for s in runs])),
        }
    return out


def _noise_rmse(train, held, net, label, n_boot=300, seed=0):
    """sqrt(mean over milestones of the bootstrap variance of q_net), WE runs resampled."""
    units = list(_scan(train + held, net, T.Milestones(), label).values())
    M = len(net.transient)
    rng = np.random.Generator(np.random.PCG64(seed))
    qs = [solve_absorption(*_normalise(_counts([units[i] for i in rng.integers(0, len(units), len(units))],
                                                M, 2), M))[:, 1] for _ in range(n_boot)]
    return float(np.sqrt(np.mean(np.var(qs, axis=0))))


def _summary(rep):
    return {"passed": rep.passed, "max_dev": rep.max_dev, "ci": list(rep.ci), "worst": rep.worst,
            "n_strata": len(rep.strata), "n_excluded": len(rep.excluded_strata),
            "n_boot_failed": rep.n_boot_failed, "unabsorbed_fraction": rep.unabsorbed_fraction,
            "strata": rep.strata}


def test_11_2_double_well_network_committor(campaigns):
    train, held, runs = campaigns["dw2d"]
    assert max(s["final_weight"] for s in runs) < 1e-3
    net = T.network(False)
    q = solve_absorption(*build_transitions(train + held, net, T.Milestones(), None))[:, 1]
    ref = T.reference_committor("dw2d")[(0, 0)]
    rmse = float(np.sqrt(np.mean((q - ref) ** 2)))
    rep = markov_test(train, held, net, T.Milestones(), None, n_boot=N_BOOT, seed=1)
    noise = _noise_rmse(train, held, net, None)
    RESULTS["11.2"] = {"q_net": q.tolist(), "q_ref": ref.tolist(), "rmse": rmse,
                       "expected_rmse_noise": noise, "markov": _summary(rep)}
    print(f"\n[11.2] RMSE {rmse:.4f} (noise {noise:.4f}); markov passed={rep.passed} max_dev={rep.max_dev:.4f} ci={rep.ci}")
    assert rmse < 0.03


def test_11_3_pooled_labels_fail_the_markov_test(campaigns):
    train, held, _ = campaigns["channel"]
    rep = markov_test(train, held, T.network(False), T.Milestones(), None, n_boot=N_BOOT, seed=2)
    RESULTS["11.3"] = {"markov": _summary(rep)}
    print(f"\n[11.3] pooled: passed={rep.passed} max_dev={rep.max_dev:.4f} ci={rep.ci} worst={rep.worst}")
    assert {s["label"] for s in rep.strata} == set(T.LABELS["channel"])
    assert rep.passed is False


def test_11_4_augmented_network_passes_and_matches_the_reference(campaigns):
    train, held, _ = campaigns["channel"]
    ref = T.reference_committor("channel", n=400)
    ref_coarse = T.reference_committor("channel", n=200)
    net = T.network(True)
    out = {}
    for i, lab in enumerate(T.LABELS["channel"]):
        grid_change = float(np.max(np.abs(ref[lab] - ref_coarse[lab])))
        assert grid_change < 0.005, grid_change
        rep = markov_test(train, held, net, T.Milestones(), lab, n_boot=N_BOOT, seed=3 + i)
        q = solve_absorption(*build_transitions(train + held, net, T.Milestones(), lab))[:, 1]
        rmse = float(np.sqrt(np.mean((q - ref[lab]) ** 2)))
        out[str(lab)] = {"q_net": q.tolist(), "q_ref": ref[lab].tolist(), "rmse": rmse,
                         "expected_rmse_noise": _noise_rmse(train, held, net, lab),
                         "reference_grid_change": grid_change, "markov": _summary(rep)}
        print(f"\n[11.4] label {lab}: RMSE {rmse:.4f}; markov passed={rep.passed} "
              f"max_dev={rep.max_dev:.4f} ci={rep.ci}")
    RESULTS["11.4"] = out
    for lab in T.LABELS["channel"]:
        assert out[str(lab)]["markov"]["passed"] is True
        assert out[str(lab)]["rmse"] < 0.03
