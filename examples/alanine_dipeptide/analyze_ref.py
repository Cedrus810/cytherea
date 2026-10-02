#!/usr/bin/env python
"""MSM analysis of the A1 reference phi/psi series (deeptime, reversible MLE).

State definition: ala2_common.CORE_BOXES_DEG / core_labels /
transition_based_assignment / OBS_INTERVAL_PS (the single definition shared with
the 14b shooting labels).  Core sets (degrees, phi/psi in (-180, 180]; closed boxes):
  0  C7eq/C5 (beta + PII basin): phi in [-180, -30] and (psi in [100, 180] or psi in [-180, -160])
  1  alphaR                    : phi in [-180, -30] and  psi in [-80, -10]
  2  alphaL                    : phi in [  30, 100] and  psi in [0, 90]
Everything else (the psi ~ 0..100 bridge at phi < 0, C7ax, the phi ~ 0 and
phi ~ +/-180 seams) is a non-core region.  Transition-based assignment
(core-set / milestoning): each frame carries the label of the LAST core
visited; leading frames before the first core entry are dropped.  Labels are
defined at the 1 ps observation interval (OBS_INTERVAL_PS).

For each lag tau (ps) in --lags-ps:
  * sliding-window counts, largest strongly connected set, reversible MLE MSM
  * implied timescales t_i = -tau / ln|lambda_i|, i = 2, 3
  * 95 % CIs from a bootstrap over contiguous trajectory BLOCKS
    (--n-blocks blocks, resampled with replacement --n-boot times; counts are
    summed over the resampled blocks; transitions across block edges are dropped).
    Replicates whose largest connected set differs from the full estimate's
    (e.g. a resample without any alphaL visit, whose "t2" would be a different
    process) are EXCLUDED from the CI and counted per lag; replicates with an
    infinite timescale are dropped too.  The CI is conditional on the kept
    replicates, so it is flagged ci_valid = false when more than
    --ci-max-excluded-frac of them are lost.
Two counting modes are reported:
  * "tba": C_ij = #{t: TBA[t]=i, TBA[t+tau]=j}  (implied_timescales, msm, ck_test)
  * "core_start": C_ij = #{t: raw core label[t]=i, TBA[t+tau]=j}, i.e. only
    windows that START INSIDE a core.  This is the reference for 14b FixedLag
    shots, which are launched from core frames and labelled by the last core
    visited along the shot (seeded with the start core).  At short tau it is
    biased HIGH relative to TBA (transitions pass through non-core bridge frames,
    and the windows that start there -- over-represented in committed crossings --
    are excluded); at tau <= the bridge crossing time its count matrix is not even
    connected.

Lag selection -- two lags are reported:
  * tba_converged_lag_ps (MSM quality of the reference): smallest lag whose slowest
    ITS t2 agrees with t2 at EVERY larger lag in the list;
  * shoot_lag_ps (the 14b FixedLag tau): the same rule applied to the CORE-START
    ITS, restricted to lags >= --min-shoot-lag-ps (default 10 ps = 10 observation
    intervals: shorter shots from core frames almost never reach another core and
    their T is dominated by the within-basin/bridge bias), AND the core-start CK at
    that lag must pass.  t2 alone is not enough: with alphaL visited, t2 is the
    alphaL process (flat from 10 ps) while the C7eq <-> alphaR t3 is still in the
    bridge bias, and the CK fails on its diagonal.  --shoot-lag-ps overrides.
  "Agrees" means: the 95 % percentile CI of the PAIRED per-replicate ratio
  t2(tau_i)[b] / t2(tau_j)[b] contains 1 (the bootstrap resamples the same blocks
  at every lag, so the ratio is far tighter than the marginal CIs; this makes the
  choice insensitive to how dense the lag list is), OR the point estimates differ
  by at most --its-tol (relative).  Larger lags with tau_j > t2 are kept in the
  comparison (listed in lag_exceeds_t2_ps), so a rising ITS is not hidden; larger
  lags whose t2 CI is invalid are left out.  A candidate lag needs at least one
  comparable larger lag, a finite t2, a valid CI and the full visited state set.
The TBA summary (msm, ck_test) is evaluated at the TBA lag (or --msm-lag-ps); the
core-start summary (core_start.msm, core_start.ck_test) at shoot_lag_ps.  Each
summary reports the reversible-MLE T (for ITS / populations) AND the row-normalised
T_ij = C_ij / sum_j C_ij with block-bootstrap CIs; 14b compares the ROW-NORMALISED
one, which does not depend on how many shots start in each state.  A summary
whose active set misses a state that the trajectory visited is DEGENERATE and is
reported as null with a reason (msm_null_reason).
Chapman-Kolmogorov test (both modes): P_pred(k tau) = T(tau)^k vs T_est(k tau)
(MSM estimated at lag k tau) for a geometric list of k up to
k_max = max(--ck-k, ceil(--ck-horizon * t2 / tau)), capped by the block length;
it passes when every diagonal element of the prediction lies inside the estimate's
95 % bootstrap interval widened by --ck-atol.  If the cap keeps k_max tau below t2
the result is inconclusive (pass = null, horizon_reaches_t2 = false).

Outputs (in --out, default = the run dir): analysis.json, its.png.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ala2_common as C  # noqa: E402

# The state definition lives in ala2_common (shared with 14b shooting labels).
STATE_NAMES = C.STATE_NAMES
CORE_BOXES_DEG = C.CORE_BOXES_DEG
N_STATES = C.N_STATES
core_labels = C.core_labels
transition_based_assignment = C.transition_based_assignment


def count_matrix(start: np.ndarray, end: np.ndarray, lag: int) -> np.ndarray:
    """Sliding-window counts C_ij = #{t : start[t] = i, end[t + lag] = j}.

    Frames with start[t] < 0 are skipped.  ``start = end = TBA labels`` gives the
    usual TBA/milestoning counts; ``start = raw core labels, end = TBA labels``
    gives the core-start counts that FixedLag shots launched from core frames
    estimate (14b).
    """
    c = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    if start.size > lag:
        i, j = start[:-lag], end[lag:]
        ok = i >= 0
        np.add.at(c, (i[ok], j[ok]), 1.0)
    return c


def estimate_msm(counts: np.ndarray):
    """Reversible MLE on the largest connected set.

    Returns dict(T full-size with NaN outside the set, pi full-size, active,
    eigvals sorted desc by modulus) or None if fewer than 1 connected state.
    """
    from deeptime.markov import TransitionCountModel
    from deeptime.markov.msm import MaximumLikelihoodMSM

    if counts.sum() <= 0:
        return None
    cm = TransitionCountModel(counts).submodel_largest(directed=True)
    active = np.asarray(cm.state_symbols, dtype=int)
    if active.size == 0:
        return None
    if active.size == 1:
        T_act = np.ones((1, 1))
        pi_act = np.ones(1)
    else:
        msm = MaximumLikelihoodMSM(reversible=True).fit_from_counts(cm).fetch_model()
        T_act = np.asarray(msm.transition_matrix)
        pi_act = np.asarray(msm.stationary_distribution)
    T = np.full((N_STATES, N_STATES), np.nan)
    T[np.ix_(active, active)] = T_act
    pi = np.zeros(N_STATES)
    pi[active] = pi_act
    ev = np.linalg.eigvals(T_act)
    ev = ev[np.argsort(-np.abs(ev))]
    return {"T": T, "T_active": T_act, "pi": pi, "active": active, "eigvals": ev}


def row_normalise(counts: np.ndarray) -> np.ndarray:
    """Non-reversible T_ij = C_ij / sum_j C_ij; rows without counts are NaN."""
    c = np.asarray(counts, dtype=np.float64)
    tot = c.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, c / np.where(tot > 0, tot, 1.0), np.nan)


def timescales_ps(model, lag_ps: float) -> np.ndarray:
    out = np.full(N_STATES - 1, np.nan)
    if model is None:
        return out
    ev = np.abs(model["eigvals"][1:]).real
    for i, lam in enumerate(ev[: N_STATES - 1]):
        if 0 < lam < 1:
            out[i] = -lag_ps / math.log(lam)
        elif lam >= 1:
            out[i] = np.inf
    return out


def ci(samples: np.ndarray, axis=0):
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(samples, 2.5, axis=axis)
        hi = np.nanpercentile(samples, 97.5, axis=axis)
    return lo, hi


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


ESTIMATOR = "deeptime MaximumLikelihoodMSM(reversible=True) on the largest strongly connected count set"
PROPAGATOR = "LangevinMiddle, gamma = 0.1 / ps, 300 K, dt = 2 fs, fixed box (as ref_long.py)"


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def converged_lag(rows, tol: float, min_lag_ps: float = 0.0, use_ci: bool = True, accept=None):
    """Smallest lag (ps) whose slowest ITS agrees with t2 at every larger lag.

    ``rows`` are ITS rows (``lag_ps``, ``timescales_ps``; optional
    ``active_complete``, ``ci_valid`` and ``_boot_t2``, the per-replicate t2 of the
    block bootstrap, aligned across lags -- the SAME resampled blocks for every
    lag, NaN where a replicate was excluded).  Lag j agrees with candidate i when

      * the 95 % percentile CI of the PAIRED per-replicate ratio
        t2_i[b] / t2_j[b] contains 1 (replicates are shared across lags, so the
        marginal CIs are far too wide for this comparison), or
      * |t2_j / t2_i - 1| <= tol.

    Larger lags with tau > t2 stay in the comparison (listed in the info); larger
    lags whose t2 CI is invalid (ci_valid false) or whose active set is incomplete
    are left out of it (listed).  Candidates need lag >= min_lag_ps, finite t2, a
    valid t2 CI, a complete active set and at least one comparable larger lag.
    ``use_ci=False`` drops the paired-ratio clause (relative tolerance only).
    ``accept(lag_ps) -> bool`` is a further per-candidate condition, called only for
    candidates that pass the t2 rule, in increasing lag order (the shooting lag uses
    it to require a passing core-start CK); rejected candidates are listed in
    ``info["rejected_by_accept_ps"]``.
    Returns (lag_ps or None, info).
    """
    lags = np.array([_f(r["lag_ps"]) for r in rows])
    t2 = np.array([_f(r["timescales_ps"][0]) for r in rows])
    complete = [bool(r.get("active_complete", True)) for r in rows]
    civ = [bool(r.get("ci_valid", [True])[0]) for r in rows]
    boot = [None if r.get("_boot_t2") is None else np.asarray(r["_boot_t2"], dtype=float) for r in rows]
    comparable = [bool(np.isfinite(t2[j]) and complete[j] and civ[j]) for j in range(len(rows))]
    info = {
        "rule": ("95% percentile CI of the paired bootstrap ratio t2(tau_i)/t2(tau_j) contains 1, or "
                 if use_ci else "") + f"t2(tau_i) within {tol:.0%} of t2(tau_j), for every larger lag tau_j "
                f"(tau > t2 kept; invalid-CI lags left out); tau_i >= {min_lag_ps:g} ps",
        "min_lag_ps": float(min_lag_ps),
        "lag_exceeds_t2_ps": [float(lags[j]) for j in range(len(rows)) if np.isfinite(t2[j]) and lags[j] > t2[j]],
        "left_out_invalid_ci_or_incomplete_ps": [float(lags[j]) for j in range(len(rows))
                                                 if np.isfinite(t2[j]) and not comparable[j]],
        "rejected_by_accept_ps": [],
    }

    def paired_ok(i, j):
        if not use_ci or boot[i] is None or boot[j] is None:
            return False
        ok = np.isfinite(boot[i]) & np.isfinite(boot[j]) & (boot[j] > 0)
        if ok.sum() < 2:
            return False
        lo, hi = np.percentile(boot[i][ok] / boot[j][ok], [2.5, 97.5])
        return bool(lo <= 1.0 <= hi)

    for i in range(len(rows)):
        if lags[i] < min_lag_ps * (1 - 1e-9) or not comparable[i]:
            continue
        later = [j for j in range(i + 1, len(rows)) if comparable[j]]
        if not later:
            continue
        if all(abs(t2[j] / t2[i] - 1.0) <= tol or paired_ok(i, j) for j in later):
            if accept is not None and not accept(float(lags[i])):
                info["rejected_by_accept_ps"].append(float(lags[i]))
                continue
            return float(lags[i]), info
    return None, info


def ck_k_list(k_max: int, n_points: int) -> list[int]:
    """Geometric k list 1 .. k_max (both included, at most n_points values)."""
    if k_max < 1:
        return []
    ks = np.unique(np.round(np.geomspace(1, k_max, num=max(2, min(k_max, n_points)))).astype(int))
    return [int(k) for k in ks]


def _run_spec(text: str) -> tuple[Path, int]:
    path, _, pieces = text.rpartition(":") if ":" in text and text.rsplit(":", 1)[1].isdigit() else (text, "", "1")
    k = int(pieces)
    if k < 1:
        raise argparse.ArgumentTypeError(f"{text}: PIECES must be >= 1")
    return Path(path), k


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, action="append", type=_run_spec, metavar="DIR[:PIECES]",
                   help="ref_long.py output dir (reads phipsi.bin); repeat for several independent "
                        "trajectories. DIR:K cuts that run into K contiguous pieces, each treated as its "
                        "own trajectory (pairs across piece edges are dropped)")
    p.add_argument("--out", type=Path, default=None,
                   help="where to write analysis.json / its.png (default: the --run dir; required "
                        "with several --run)")
    p.add_argument("--lags-ps", type=str, default=",".join(f"{x:g}" for x in C.SHOOT_LAG_GRID_PS),
                   help="lag grid (ps). The default is the 14b contract grid ala2_common.SHOOT_LAG_GRID_PS "
                        "(ruling R40); any other grid gives a diagnostic shoot_lag_ps only")
    p.add_argument("--n-blocks", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed (explicit PCG64, no global state)")
    p.add_argument("--its-tol", type=float, default=0.10)
    p.add_argument("--msm-lag-ps", type=float, default=None, help="override the TBA MSM lag")
    p.add_argument("--shoot-lag-ps", type=float, default=None,
                   help="override the 14b shooting lag (core-start summary lag)")
    p.add_argument("--min-shoot-lag-ps", type=float, default=10.0,
                   help="lower bound for the automatically chosen shooting lag")
    p.add_argument("--ck-k", type=int, default=5, help="minimum k_max of the CK test")
    p.add_argument("--ck-horizon", type=float, default=2.0,
                   help="CK horizon in units of t2: k_max = ceil(ck_horizon * t2 / tau); must be >= 1")
    p.add_argument("--ck-points", type=int, default=12, help="number of (geometric) k values in the CK test")
    p.add_argument("--ck-atol", type=float, default=0.01)
    p.add_argument("--ci-max-excluded-frac", type=float, default=0.05,
                   help="a bootstrap CI is flagged invalid when more replicates than this are lost")
    p.add_argument("--skip-ps", type=float, default=0.0, help="discard this much at the start")
    args = p.parse_args(argv)
    if args.ck_horizon < 1.0:
        p.error("--ck-horizon must be >= 1 (the CK horizon has to reach t2)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t_wall = time.time()
    runs = args.run
    if len(runs) > 1 and args.out is None:
        raise SystemExit("--out is required with several --run")
    out = args.out or runs[0][0]
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((r / "phipsi.json").read_text()) for r, _ in runs]
    meta = metas[0]
    if any(m["interval_steps"] != meta["interval_steps"] or m["interval_ps"] != meta["interval_ps"]
           for m in metas):
        raise SystemExit("the --run trajectories have different phi/psi intervals")
    dt_ps = float(meta["interval_ps"])
    try:
        C.check_obs_interval(dt_ps)
        obs_ok = True
    except ValueError as exc:
        obs_ok = False
        print(f"WARNING: {exc}. This analysis is NOT the 14b reference.", file=sys.stderr)
    requested_lags_ps = [float(x) for x in args.lags_ps.split(",") if x.strip()]
    grid_ok = C.lag_grid_is_contract(requested_lags_ps)
    if not grid_ok:
        print(f"WARNING: --lags-ps {args.lags_ps} is not the 14b contract grid "
              f"(ala2_common.SHOOT_LAG_GRID_PS); shoot_lag_ps is diagnostic only, not the 14b tau.",
              file=sys.stderr)
    # Segments: every run (minus --skip-ps of burn-in at its start), optionally cut
    # into PIECES contiguous pieces. Each segment is its own trajectory: TBA labels,
    # counts and bootstrap blocks never cross a segment edge.
    skip = int(round(args.skip_ps / dt_ps))
    seg_phi, seg_psi, seg_core_all, seg_dtraj, seg_core, n_records = [], [], [], [], [], 0
    n_leading = 0
    for (run_dir, pieces), m in zip(runs, metas):
        rec = C.load_phipsi(run_dir / "phipsi.bin")
        steps = rec["step"]
        n_records += int(steps.size)
        if steps.size < 2:
            raise SystemExit(f"{run_dir}: fewer than 2 phi/psi records")
        if not np.all(np.diff(steps) == m["interval_steps"]):
            raise SystemExit(f"{run_dir}: phipsi.bin steps are not contiguous (duplicated/missing frames)")
        phi_r = rec["phi"][skip:].astype(np.float64)
        psi_r = rec["psi"][skip:].astype(np.float64)
        for idx in np.array_split(np.arange(phi_r.size), pieces):
            if idx.size < 2:
                raise SystemExit(f"{run_dir}: a piece has < 2 frames (too many PIECES)")
            ca = core_labels(phi_r[idx], psi_r[idx])
            dt_seg, n_lead = transition_based_assignment(ca)
            seg_phi.append(phi_r[idx])
            seg_psi.append(psi_r[idx])
            seg_core_all.append(ca)
            seg_dtraj.append(dt_seg)
            seg_core.append(ca[n_lead:])  # raw core labels aligned with dtraj
            n_leading += n_lead
    phi, psi = np.concatenate(seg_phi), np.concatenate(seg_psi)
    core_all = np.concatenate(seg_core_all)
    dtraj, core = np.concatenate(seg_dtraj), np.concatenate(seg_core)
    seg_len = np.array([d.size for d in seg_dtraj])
    seg_edges = np.concatenate([[0], np.cumsum(seg_len)])
    n_frames = phi.size
    n = dtraj.size
    sim_ns = sum((a.size - 1) for a in seg_phi) * dt_ps / 1000.0
    visited = sorted(int(x) for x in np.unique(core_all[core_all >= 0]))
    never_visited = [STATE_NAMES[s] for s in range(N_STATES) if s not in visited]

    trans_counts = np.zeros((N_STATES, N_STATES), dtype=int)
    for d in seg_dtraj:
        for i in np.flatnonzero(np.diff(d) != 0):
            trans_counts[d[i], d[i + 1]] += 1

    lag_steps_all = sorted({max(1, int(round(float(x) / dt_ps))) for x in args.lags_ps.split(",") if x.strip()})
    n_blocks = max(len(seg_len), min(args.n_blocks, n // max(1, 2 * min(lag_steps_all)) if n else 1))
    # blocks per segment ~ its length (>= 1 each); no block straddles a segment edge
    per_seg = np.maximum(1, np.floor(n_blocks * seg_len / max(1, n)).astype(int))
    while per_seg.sum() < n_blocks:
        per_seg[np.argmax(seg_len / per_seg)] += 1
    n_blocks = int(per_seg.sum())
    block_edges = np.concatenate([
        np.linspace(seg_edges[i], seg_edges[i + 1], per_seg[i] + 1).astype(int)[(1 if i else 0):]
        for i in range(len(seg_len))
    ])
    block_len = int(np.min(np.diff(block_edges))) if n_blocks else 0
    # a lag is usable if every block still has >= 2 * lag frames
    lag_steps = [l for l in lag_steps_all if l < n and 2 * l <= block_len]
    dropped_lags = [l * dt_ps for l in lag_steps_all if l not in lag_steps]

    def user_lag(value_ps, flag):
        L = max(1, int(round(value_ps / dt_ps)))
        if 2 * L > block_len or L >= n:
            raise SystemExit(f"{flag} {value_ps:g} ps exceeds half the bootstrap block length "
                             f"({block_len * dt_ps / 2:g} ps with {n_blocks} blocks): no MSM/CK possible; "
                             "use a shorter lag or fewer --n-blocks")
        return L

    rng = np.random.Generator(np.random.PCG64(args.seed))
    boot_idx = rng.integers(0, n_blocks, size=(args.n_boot, n_blocks))

    # counting modes: (start labels, end labels)
    modes = {"tba": (dtraj, dtraj), "core_start": (core, dtraj)}
    _cache: dict = {}

    def counts(mode, lag):
        """(full-trajectory counts, per-block counts (n_blocks, S, S))."""
        key = (mode, lag)
        if key not in _cache:
            st, en = modes[mode]
            blk = np.stack([count_matrix(st[block_edges[b]: block_edges[b + 1]],
                                         en[block_edges[b]: block_edges[b + 1]], lag) for b in range(n_blocks)])
            full = sum(count_matrix(st[seg_edges[i]: seg_edges[i + 1]], en[seg_edges[i]: seg_edges[i + 1]], lag)
                       for i in range(len(seg_len)))
            _cache[key] = (full, blk)
        return _cache[key]

    def boot_models(mode, lag, full):
        """Bootstrap models whose active set equals the full estimate's; the rest are excluded.

        A replicate that misses a state (e.g. no alphaL visit in the resampled blocks)
        would otherwise report its slowest *remaining* process as t2 and mix a
        different process into the CI.
        """
        kept, n_excl, _ = boot_models_aligned(mode, lag, full)
        return kept, n_excl

    def boot_models_aligned(mode, lag, full):
        """As boot_models, plus the replicate index of each kept model (same blocks for every lag)."""
        _, blk = counts(mode, lag)
        ref_active = tuple(full["active"]) if full is not None else None
        kept, n_excl, idx = [], 0, []
        for b, bi in enumerate(boot_idx):
            m = estimate_msm(blk[bi].sum(axis=0))
            if m is None or tuple(m["active"]) != ref_active:
                n_excl += 1
                continue
            kept.append(m)
            idx.append(b)
        return kept, n_excl, idx

    def ci_ok(n_good):
        return bool(args.n_boot > 0 and (args.n_boot - n_good) / args.n_boot <= args.ci_max_excluded_frac)

    def is_complete(full):
        return full is not None and sorted(int(a) for a in full["active"]) == visited

    def its_table(mode):
        rows = []
        for l in lag_steps:
            full = estimate_msm(counts(mode, l)[0])
            ts = timescales_ps(full, l * dt_ps)
            kept, n_excl, kidx = boot_models_aligned(mode, l, full)
            bts = np.array([timescales_ps(m, l * dt_ps) for m in kept]).reshape(-1, N_STATES - 1)
            boot_t2 = np.full(args.n_boot, np.nan)
            boot_t2[kidx] = bts[:, 0]
            lo, hi = ci(np.where(np.isfinite(bts), bts, np.nan)) if len(kept) else (
                np.full(N_STATES - 1, np.nan), np.full(N_STATES - 1, np.nan))
            n_valid = [int(np.isfinite(bts[:, i]).sum()) for i in range(N_STATES - 1)]
            rows.append({
                "lag_ps": l * dt_ps, "lag_frames": l,
                "timescales_ps": ts, "ci95_lo_ps": lo, "ci95_hi_ps": hi,
                "n_boot_used": len(kept),
                "n_boot_excluded_active_set_mismatch": n_excl,
                "n_boot_valid": n_valid,
                "ci_valid": [bool(np.isfinite(ts[i]) and ci_ok(n_valid[i])) for i in range(N_STATES - 1)],
                "n_counts": float(counts(mode, l)[0].sum()),
                "active_states": [STATE_NAMES[s] for s in (full["active"] if full else [])],
                "active_complete": is_complete(full),
                "lag_exceeds_t2": bool(np.isfinite(ts[0]) and l * dt_ps > ts[0]),
                "stationary": full["pi"] if full else np.full(N_STATES, np.nan),
                "_boot_t2": np.where(np.isfinite(boot_t2), boot_t2, np.nan),   # internal, not written
            })
        return rows

    its_rows = its_table("tba")
    its_core = its_table("core_start")
    conv, conv_info = converged_lag(its_rows, args.its_tol)
    conv_core, conv_core_info = converged_lag(its_core, args.its_tol)
    shoot_t2_only, _ = converged_lag(its_core, args.its_tol, min_lag_ps=args.min_shoot_lag_ps)

    # --- TBA MSM lag (reference MSM quality)
    if args.msm_lag_ps is not None:
        msm_lag_frames = user_lag(args.msm_lag_ps, "--msm-lag-ps")
        lag_source = "user (--msm-lag-ps)"
    elif conv is not None:
        msm_lag_frames = int(round(conv / dt_ps))
        lag_source = "converged TBA ITS (" + conv_info["rule"] + ")"
    elif lag_steps:
        msm_lag_frames = lag_steps[len(lag_steps) // 2]
        lag_source = "FALLBACK: TBA slowest ITS not converged over the lag list; middle lag used"
    else:
        msm_lag_frames = None
        lag_source = "no usable lag (trajectory too short for the lag list / blocks)"

    def msm_summary(mode, L):
        """(full model or None, summary dict or None, null reason or None)."""
        full = estimate_msm(counts(mode, L)[0])
        if full is None:
            return None, None, f"no {mode} counts at lag {L * dt_ps:g} ps"
        if not is_complete(full):
            return full, None, (
                f"degenerate active set {[STATE_NAMES[s] for s in full['active']]} at lag {L * dt_ps:g} ps: "
                f"the largest strongly connected {mode} count set misses visited state(s) "
                f"{[STATE_NAMES[s] for s in visited if s not in full['active']]}")
        kept, n_excl = boot_models(mode, L, full)
        bpi = np.array([m["pi"] for m in kept]).reshape(-1, N_STATES)
        bT = np.array([m["T"] for m in kept]).reshape(-1, N_STATES, N_STATES)
        nanS = np.full(N_STATES, np.nan)
        plo, phi_ = ci(bpi) if len(kept) else (nanS, nanS)
        Tlo, Thi = ci(bT) if len(kept) else (np.full((N_STATES,) * 2, np.nan),) * 2
        # row-normalised (non-reversible) T: independent of how many windows/shots start
        # in each state, so it is what a shooting design with its own per-state shot
        # numbers must be compared with.  Bootstrap over ALL replicates (no active-set
        # condition needed); rows without counts are NaN.
        C_full = counts(mode, L)[0]
        _, blk = counts(mode, L)
        P = row_normalise(C_full)
        bP = np.stack([row_normalise(blk[bi].sum(axis=0)) for bi in boot_idx]) if args.n_boot else \
            np.full((0, N_STATES, N_STATES), np.nan)
        Plo, Phi = ci(bP) if len(bP) else (np.full((N_STATES,) * 2, np.nan),) * 2
        n_row_ok = [int(np.isfinite(bP[:, s_, 0]).sum()) for s_ in range(N_STATES)] if len(bP) else [0] * N_STATES
        return full, {
            "lag_ps": L * dt_ps,
            "counting": mode,
            "estimator": ESTIMATOR,
            "count_matrix": counts(mode, L)[0],
            "active_states": [STATE_NAMES[s] for s in full["active"]],
            "complete_state_set": bool(full["active"].size == N_STATES),
            "never_visited_states": never_visited,
            "transition_matrix": full["T"],
            "transition_matrix_ci95_lo": Tlo, "transition_matrix_ci95_hi": Thi,
            "transition_matrix_row_normalised": P,
            "transition_matrix_row_normalised_ci95_lo": Plo,
            "transition_matrix_row_normalised_ci95_hi": Phi,
            "row_normalised_n_boot_valid_per_row": n_row_ok,
            "row_normalised_ci_valid_per_row": [bool(np.isfinite(P[s_, 0]) and ci_ok(n_row_ok[s_]))
                                                for s_ in range(N_STATES)],
            "note": "transition_matrix = reversible MLE (use for ITS / populations); "
                    "transition_matrix_row_normalised = C_ij / sum_j C_ij (design-independent; 14b compares THIS)",
            "stationary_population": {STATE_NAMES[s]: {"value": full["pi"][s], "ci95": [plo[s], phi_[s]]}
                                      for s in range(N_STATES)},
            "timescales_ps": timescales_ps(full, L * dt_ps),
            "n_boot_used": len(kept), "n_boot_excluded_active_set_mismatch": n_excl,
            "ci_valid": ci_ok(len(kept)),
        }, None

    def ck_test(mode, L, full):
        """(ck dict or None, null reason or None).  Horizon k_max tau >= ck_horizon * t2."""
        if full is None or full["active"].size < 2:
            return None, "CK needs an MSM with >= 2 connected states"
        tau = L * dt_ps
        t2 = timescales_ps(full, tau)[0]
        k_target = args.ck_k
        if np.isfinite(t2):
            k_target = max(k_target, int(math.ceil(args.ck_horizon * t2 / tau)))
        k_cap = max(0, min(block_len // (2 * L), (n - 1) // L))
        ks = ck_k_list(min(k_target, k_cap), args.ck_points)
        reaches = bool(np.isfinite(t2) and ks and ks[-1] * tau >= t2)
        rows, ck_pass = [], (True if reaches and len(ks) >= 2 else None)
        for k in ks:
            kl = k * L
            est = estimate_msm(counts(mode, kl)[0])
            pred_full = np.full(N_STATES, np.nan)
            est_full = np.full(N_STATES, np.nan)
            Tk = np.linalg.matrix_power(full["T_active"], k)
            pred_full[full["active"]] = np.diag(Tk)
            if est is not None:
                est_full = np.diag(est["T"]).copy()
            kept, n_excl = boot_models(mode, kl, est)
            b_est = np.array([np.diag(m["T"]) for m in kept]).reshape(-1, N_STATES)
            elo, ehi = ci(b_est) if len(kept) else (np.full(N_STATES, np.nan),) * 2
            ok = []
            for s in range(N_STATES):
                if not (np.isfinite(pred_full[s]) and np.isfinite(elo[s])):
                    ok.append(None)
                    continue
                good = bool(elo[s] - args.ck_atol <= pred_full[s] <= ehi[s] + args.ck_atol)
                ok.append(good)
                if not good and ck_pass is not None:
                    ck_pass = False
            rows.append({"k": k, "lag_ps": kl * dt_ps, "predicted_stay": pred_full, "estimated_stay": est_full,
                         "estimated_ci95_lo": elo, "estimated_ci95_hi": ehi, "within_ci": ok,
                         "n_boot_used": len(kept), "n_boot_excluded_active_set_mismatch": n_excl,
                         "ci_valid": ci_ok(len(kept))})
        return {"counting": mode, "tau_ps": tau, "t2_ps": t2, "k": ks,
                "horizon_ps": ks[-1] * tau if ks else None,
                "horizon_target_ps": args.ck_horizon * t2 if np.isfinite(t2) else None,
                "horizon_reaches_t2": reaches,
                "horizon_note": None if reaches else
                "INCONCLUSIVE: k_max tau < t2 (capped by the block length / trajectory); pass = null",
                "rows": rows, "pass": ck_pass,
                "criterion": f"diag(T(tau)^k) inside [2.5,97.5]% bootstrap interval of diag(T_est(k tau)) "
                             f"+/- {args.ck_atol}, k geometric up to max({args.ck_k}, "
                             f"ceil({args.ck_horizon:g} t2 / tau))"}, None

    # --- 14b shooting lag: core-start t2 converged (paired rule) AND core-start CK passes.
    # t2 alone is not enough: once alphaL is visited, t2 is the alphaL process, whose
    # core-start ITS is flat from ~10 ps on, while the faster C7eq <-> alphaR process
    # (t3) is still in the short-tau bridge bias there -- and the CK, which 14b repeats
    # on the shots (14.3), fails on its diagonal (full A1 data: t3 core-start 316 vs
    # TBA 168 ps at 10 ps).  The CK checks every process at once.
    _core_eval: dict = {}

    def core_eval(L):
        """(summary or None, null reason, ck or None, ck null reason) of core-start at L frames."""
        if L not in _core_eval:
            full_c, summ, reason = msm_summary("core_start", L)
            if summ is None:
                ck_c, ck_r = None, "no core-start summary: " + reason
            else:
                ck_c, ck_r = ck_test("core_start", L, full_c)
            _core_eval[L] = (summ, reason, ck_c, ck_r)
        return _core_eval[L]

    def core_ck_passes(lag_ps):
        ck_c = core_eval(int(round(lag_ps / dt_ps)))[2]
        return ck_c is not None and ck_c["pass"] is True

    shoot_conv, shoot_info = converged_lag(its_core, args.its_tol, min_lag_ps=args.min_shoot_lag_ps,
                                           accept=core_ck_passes)
    shoot_strict, _ = converged_lag(its_core, args.its_tol, min_lag_ps=args.min_shoot_lag_ps, use_ci=False,
                                    accept=core_ck_passes)
    for r in its_rows + its_core:
        r.pop("_boot_t2", None)
    shoot_rule = shoot_info["rule"] + "; and the core-start CK at that lag passes (pass = true)"
    if args.shoot_lag_ps is not None:
        shoot_frames = user_lag(args.shoot_lag_ps, "--shoot-lag-ps")
        shoot_source = "user (--shoot-lag-ps)"
    elif shoot_conv is not None:
        shoot_frames = int(round(shoot_conv / dt_ps))
        shoot_source = "converged core-start t2 with a passing core-start CK (" + shoot_rule + ")"
    else:
        shoot_frames = None
        why = ("the core-start slowest ITS is not converged at any lag" if shoot_t2_only is None else
               f"the core-start CK does not pass at any lag where core-start t2 is converged "
               f"(rejected: {shoot_info['rejected_by_accept_ps']} ps)")
        shoot_source = (f"NOT DETERMINED: {why} >= {args.min_shoot_lag_ps:g} ps of the lag list "
                        "(longer trajectory or more lags needed)")
    shoot_lag_ps = shoot_frames * dt_ps if shoot_frames is not None else None

    # TBA summary + CK at the TBA lag
    summary_msm = ck = None
    msm_reason = ck_reason = "no usable MSM lag"
    if msm_lag_frames is not None:
        full, summary_msm, msm_reason = msm_summary("tba", msm_lag_frames)
        if summary_msm is not None:
            summary_msm["lag_source"] = lag_source
            ck, ck_reason = ck_test("tba", msm_lag_frames, full)
        else:
            ck_reason = "no MSM summary: " + msm_reason
    # core-start summary + CK at the SHOOTING lag (never at the TBA lag)
    summary_core = ck_core = None
    core_reason = core_ck_reason = "shooting lag not determined: " + shoot_source
    if shoot_frames is not None:
        summary_core, core_reason, ck_core, core_ck_reason = core_eval(shoot_frames)
        if summary_core is not None:
            summary_core["lag_source"] = shoot_source

    frac = np.array([(dtraj == s).mean() if n else np.nan for s in range(N_STATES)])
    core_frac = np.array([(core_all == s).mean() for s in range(N_STATES)])
    shoot_block = {
        "lag_ps": shoot_lag_ps, "source": shoot_source, "min_lag_ps": args.min_shoot_lag_ps,
        "below_min_lag": bool(shoot_lag_ps is not None and shoot_lag_ps < args.min_shoot_lag_ps * (1 - 1e-9)),
        "core_start_converged_lag_ps": conv_core, "rule": shoot_rule,
        "t2_only_lag_ps": shoot_t2_only,
        "t2_only_note": "the t2 rule alone, without the CK condition (the pre-2026-10-02 shooting lag)",
        "ck_scan": [{"lag_ps": L * dt_ps, "pass": v[2]["pass"] if v[2] is not None else None,
                     "null_reason": v[3]} for L, v in sorted(_core_eval.items())],
        "tol_only_lag_ps": shoot_strict,
        "tol_only_note": "same rule without the paired-ratio clause (relative tolerance only), CK condition kept",
        "lag_exceeds_t2_ps": shoot_info["lag_exceeds_t2_ps"],
        "why_min_lag": "shots shorter than ~10 observation intervals from core frames almost never reach "
                       "another core (transitions pass through the non-core bridge), so core-start T(tau) "
                       "is dominated by the short-tau bias rather than the barrier-crossing kinetics",
    }
    result = {
        "input": [f"{r / 'phipsi.bin'}" + (f" ({k} pieces)" if k > 1 else "") for r, k in runs],
        "n_trajectories": int(len(seg_len)),
        "trajectory_ns": [float((d.size - 1) * dt_ps / 1000.0) for d in seg_phi],
        "n_records": n_records, "skip_ps": args.skip_ps, "skip_applies_to": "the start of every --run",
        "frame_interval_ps": dt_ps,
        "obs_interval_ps": dt_ps, "obs_interval_matches_state_contract": obs_ok,
        "simulated_ns": sim_ns,
        "states": STATE_NAMES,
        "core_boxes_deg": CORE_BOXES_DEG,
        "state_definition": "ala2_common.CORE_BOXES_DEG / core_labels / transition_based_assignment "
                            f"at OBS_INTERVAL_PS = {C.OBS_INTERVAL_PS:g} ps",
        "assignment": "transition-based (last core visited); leading non-core frames dropped",
        "visited_states": [STATE_NAMES[s] for s in visited],
        "never_visited_states": never_visited,
        "estimator": ESTIMATOR,
        "n_leading_frames_dropped": n_leading,
        "fraction_frames_in_core": {STATE_NAMES[s]: core_frac[s] for s in range(N_STATES)},
        "fraction_frames_noncore": float((core_all == C.NONCORE).mean()),
        "fraction_tba": {STATE_NAMES[s]: frac[s] for s in range(N_STATES)},
        "core_to_core_transitions": {f"{STATE_NAMES[i]}->{STATE_NAMES[j]}": int(trans_counts[i, j])
                                     for i in range(N_STATES) for j in range(N_STATES) if i != j},
        "bootstrap": {"n_blocks": n_blocks, "block_ns": block_len * dt_ps / 1000.0, "n_boot": args.n_boot,
                      "rng": f"numpy PCG64(seed={args.seed})",
                      "replicate_policy": "replicates whose largest connected set differs from the full "
                                          "estimate's are EXCLUDED (counted in n_boot_excluded_active_set_mismatch); "
                                          "replicates with an infinite timescale are dropped (n_boot_valid)",
                      "ci_valid_rule": f"at most {args.ci_max_excluded_frac:.0%} of the replicates lost"},
        "lags_dropped_ps": dropped_lags,
        "implied_timescales": its_rows,
        "tba_converged_lag_ps": conv,
        "tba_convergence": conv_info,
        "shoot_lag_ps": shoot_lag_ps,
        "shoot_lag": shoot_block,
        "msm": summary_msm, "msm_null_reason": None if summary_msm is not None else msm_reason,
        "ck_test": ck, "ck_null_reason": None if ck is not None else ck_reason,
        "core_start": {
            "definition": "C_ij(tau) = #{t : raw core label at t = i (frame INSIDE core i), TBA label at t+tau = j}; "
                          "the quantity estimated by 14b FixedLag shots launched from core frames and labelled "
                          "by the last core visited (seeded with the start core)",
            "implied_timescales": its_core,
            "converged_lag_ps": conv_core,
            "convergence": conv_core_info,
            "msm": summary_core, "msm_null_reason": None if summary_core is not None else core_reason,
            "ck_test": ck_core, "ck_null_reason": None if ck_core is not None else core_ck_reason,
        },
        "contract_14b": {
            "compare_against": "core_start.msm.transition_matrix_row_normalised (with its CIs) at shoot_lag_ps; "
                               "the reversible transition_matrix is for ITS/populations only",
            "shoot_lag_ps": shoot_lag_ps,
            "lag_grid_ps": list(C.SHOOT_LAG_GRID_PS),
            "requested_lags_ps": requested_lags_ps,
            "lag_grid_matches_contract": grid_ok,
            "lag_grid_rule": "14b fixes tau on the contract grid ala2_common.SHOOT_LAG_GRID_PS (ruling R40); "
                             "a shoot_lag_ps chosen on any other grid is diagnostic only",
            "obs_interval_ps": C.OBS_INTERVAL_PS,
            "labels": "ala2_common.label_shot(phi, psi, start_label, interval_ps=OBS_INTERVAL_PS)",
            "start_frames": "raw core label >= 0 (C.core_labels)",
            "estimator": "compare ROW-NORMALISED T_ij = C_ij / sum_j C_ij (design-independent). The reversible "
                         "MLE (" + ESTIMATOR + ") couples rows through detailed balance, so with 3 states it "
                         "depends on the per-state number of shots (3.5-6 % on alphaL elements); use it for ITS only",
            "propagator": PROPAGATOR,
        },
        "deeptime_version": __import__("deeptime").__version__,
        "wall_s": time.time() - t_wall,
    }
    (out / "analysis.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    plot(out / "its.png", phi, psi, its_rows, conv, summary_msm, ck, sim_ns, its_core, shoot_lag_ps, ck_core)
    # console summary
    print(f"{sim_ns:.3f} ns, {n_frames} frames; TBA fractions "
          + ", ".join(f"{STATE_NAMES[s]} {frac[s]:.3f}" for s in range(N_STATES)))
    if never_visited:
        print(f"WARNING: never visited: {never_visited} -- the reference state set is incomplete")
    print("transitions:", result["core_to_core_transitions"])
    for r in its_rows:
        print(f"  lag {r['lag_ps']:8.1f} ps  t2 {r['timescales_ps'][0]:10.1f} "
              f"[{r['ci95_lo_ps'][0]:.1f}, {r['ci95_hi_ps'][0]:.1f}]  t3 {r['timescales_ps'][1]:9.1f} "
              f"[{r['ci95_lo_ps'][1]:.1f}, {r['ci95_hi_ps'][1]:.1f}] ps  "
              f"(boot excluded {r['n_boot_excluded_active_set_mismatch']}/{args.n_boot})")
    for r in its_core:
        print(f"  core-start lag {r['lag_ps']:8.1f} ps  t2 {r['timescales_ps'][0]:10.1f} "
              f"[{r['ci95_lo_ps'][0]:.1f}, {r['ci95_hi_ps'][0]:.1f}] ps  active {r['active_states']}"
              f"{'' if r['ci_valid'][0] else '  (CI invalid)'}")

    def ck_str(c, reason):
        if c is None:
            return f"n/a ({reason})"
        return f"pass={c['pass']} (tau {c['tau_ps']:g} ps, k up to {max(c['k'])}, horizon {c['horizon_ps']:g} ps " \
               f"vs t2 {c['t2_ps']:.1f} ps)"

    print(f"TBA converged lag: {conv} ps; TBA MSM lag used: {summary_msm['lag_ps'] if summary_msm else None} ps "
          f"({lag_source}); CK {ck_str(ck, result['ck_null_reason'])}")
    print(f"core-start converged lag: {conv_core} ps; SHOOTING LAG (14b): {shoot_lag_ps} ps ({shoot_source}); "
          f"core-start CK {ck_str(ck_core, result['core_start']['ck_null_reason'])}")
    if summary_core is None:
        print(f"core_start.msm: null ({core_reason})")
    print(f"wrote {out / 'analysis.json'} and {out / 'its.png'}")
    return 0


def plot(path, phi, psi, its_rows, conv, msm, ck, sim_ns, its_core=(), shoot_lag=None, ck_core=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    ax = axes[0]
    h, xe, ye = np.histogram2d(phi, psi, bins=90, range=[[-180, 180], [-180, 180]])
    with np.errstate(divide="ignore"):
        g = -np.log(h.T / h.max())
    ax.imshow(g, origin="lower", extent=[-180, 180, -180, 180], cmap="Blues_r", vmax=8, aspect="equal")
    for name in STATE_NAMES:
        box = CORE_BOXES_DEG[name]
        for (plo, phh) in box["phi"]:
            for (slo, shi) in box["psi"]:
                ax.add_patch(Rectangle((plo, slo), phh - plo, shi - slo, fill=False, lw=1.2, ec=ink))
        (plo, phh), (slo, shi) = box["phi"][0], box["psi"][0]
        ax.text(plo + 3, shi - 14, name, color=ink, fontsize=8)
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("psi (deg)")
    ax.set_title(f"-ln p(phi, psi), {sim_ns:.2f} ns, core sets", color=ink, fontsize=10)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))

    ax = axes[1]
    lags = np.array([r["lag_ps"] for r in its_rows])
    for i, lab in enumerate(["t2 (slowest)", "t3"]):
        ts = np.array([r["timescales_ps"][i] for r in its_rows], dtype=float)
        lo = np.array([r["ci95_lo_ps"][i] for r in its_rows], dtype=float)
        hi = np.array([r["ci95_hi_ps"][i] for r in its_rows], dtype=float)
        if lags.size == 0:
            continue
        ax.fill_between(lags, lo, hi, color=series[i], alpha=0.18, lw=0)
        ax.plot(lags, ts, "-o", color=series[i], lw=2, ms=4, label=lab)
    for r_i, lab, col, fmt in ((0, "t2 core-start", "#1baf7a", "s"), (1, "t3 core-start", series[1], "^")):
        if not its_core:
            break
        ts = np.array([r["timescales_ps"][r_i] for r in its_core], dtype=float)
        lo = np.array([r["ci95_lo_ps"][r_i] for r in its_core], dtype=float)
        hi = np.array([r["ci95_hi_ps"][r_i] for r in its_core], dtype=float)
        cl = np.array([r["lag_ps"] for r in its_core])
        ax.errorbar(cl * 1.06, ts, yerr=_yerr(ts, lo, hi), fmt=fmt, mfc="none" if r_i else col,
                    color=col, ms=4, lw=1, capsize=2, label=lab)
    if lags.size:
        vals = [v for r in its_rows for v in r["timescales_ps"] if v is not None and np.isfinite(v) and v > 0]
        ylo = 0.5 * min([lags.min()] + vals)
        yhi = 3.0 * max([lags.max()] + [v for r in its_rows for v in r["ci95_hi_ps"]
                                          if v is not None and np.isfinite(v)] + vals)
        xx = np.array([lags.min(), lags.max()])
        ax.fill_between(xx, np.full(2, ylo), xx, color="#9a9994", alpha=0.25, lw=0)
        ax.plot(xx, xx, color=ink2, lw=1)
        ax.text(xx[1], xx[1], "t = tau ", ha="right", va="bottom", fontsize=8, color=ink2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
    if conv is not None:
        ax.axvline(conv, color=ink2, lw=1, ls="--")
        ax.text(conv, ax.get_ylim()[1], f" TBA lag {conv:g} ps", va="top", fontsize=8, color=ink2)
    if shoot_lag is not None:
        ax.axvline(shoot_lag, color="#1baf7a", lw=1, ls="--")
        ax.text(shoot_lag, ax.get_ylim()[1] / 2.2, f" shooting lag {shoot_lag:g} ps", va="top", fontsize=8,
                color="#1baf7a")
    ax.grid(True, which="major", color=grid, lw=0.6)
    ax.set_xlabel("lag time tau (ps)")
    ax.set_ylabel("implied timescale (ps)")
    ax.set_title("Implied timescales (95% block-bootstrap CI)", color=ink, fontsize=10)
    ax.legend(frameon=False, loc="lower right")

    for ax, c, what in ((axes[2], ck, "TBA"), (axes[3], ck_core, "core-start")):
        _plot_ck(ax, c, what, ink, ink2, grid)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def _yerr(v, lo, hi):
    """Error-bar half widths; a point estimate outside its percentile CI gives 0, not < 0."""
    return [np.maximum(np.nan_to_num(v - lo), 0.0), np.maximum(np.nan_to_num(hi - v), 0.0)]


def _plot_ck(ax, ck, what, ink, ink2, grid):
    if ck and ck["rows"]:
        colors = ["#2a78d6", "#eb6834", "#1baf7a"]
        klag = np.array([r["lag_ps"] for r in ck["rows"]])
        for s in range(N_STATES):
            est = np.array([r["estimated_stay"][s] for r in ck["rows"]], dtype=float)
            pred = np.array([r["predicted_stay"][s] for r in ck["rows"]], dtype=float)
            lo = np.array([r["estimated_ci95_lo"][s] for r in ck["rows"]], dtype=float)
            hi = np.array([r["estimated_ci95_hi"][s] for r in ck["rows"]], dtype=float)
            if not np.any(np.isfinite(est)):
                continue
            ax.errorbar(klag, est, yerr=_yerr(est, lo, hi), fmt="o",
                        color=colors[s], ms=5, capsize=2, lw=1, label=f"{STATE_NAMES[s]} estimated")
            ax.plot(klag, pred, "--", color=colors[s], lw=2, label=f"{STATE_NAMES[s]} MSM(tau)^k")
        if ck.get("t2_ps") is not None and np.isfinite(ck["t2_ps"]):
            ax.axvline(ck["t2_ps"], color=ink2, lw=1, ls=":")
            ax.text(ck["t2_ps"], 1.0, " t2", va="top", fontsize=8, color=ink2)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        verdict = "PASS" if ck["pass"] else "FAIL" if ck["pass"] is False else "n/a"
        ax.set_title(f"CK ({what}), tau = {ck['tau_ps']:g} ps: {verdict}", color=ink, fontsize=10)
    else:
        ax.text(0.5, 0.5, "CK test not possible", ha="center", va="center", transform=ax.transAxes, color=ink2)
        ax.set_title(f"CK ({what})", color=ink, fontsize=10)
    ax.grid(True, color=grid, lw=0.6)
    ax.set_xlabel("k tau (ps)")
    ax.set_ylabel("P(stay in state after k tau)")


if __name__ == "__main__":
    sys.exit(main())
