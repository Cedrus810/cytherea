"""Committor q_B from first-hitting shots, absorbing at A and B (design 4.2).

Interval: Jeffreys 95%, i.e. the 2.5 / 97.5 percentiles of
Beta(x + 1/2, n - x + 1/2), with the usual boundary modification (lower = 0
when x = 0, upper = 1 when x = n; Brown, Cai & DasGupta 2001).

Weights (contract K11): per record, `cytherea.estimate.shot_weights` --
``record.weight`` times, for an enumerated frame design, the frame weight
over the frame's number of shots -- or the caller's ``weights``. If all
weights are 1 this is the exact binomial Jeffreys
interval on the raw counts. Otherwise ``q`` is the weighted (ratio) fraction
q = sum w y / sum w over the resolved (A or B) records, y = 1 for B, and the
Jeffreys interval is evaluated at (x = q n_eff, n_eff) with

    n_eff = min(n_KG, n_Kish),
    n_KG   = q (1 - q) / V,  V = sum w^2 (y - q)^2 / (sum w)^2
             (Korn-Graubard 1998: linearised variance of the ratio estimator),
    n_Kish = (sum w)^2 / sum w^2.

Why the minimum: each of the two alone is anti-conservative somewhere.
Kish ignores the outcome, so it is too narrow when weights correlate with
the outcome (importance weights that up-weight successes: coverage 0.85).
The linearised n_KG is too narrow when weights are heavy-tailed but
unrelated to the outcome (lognormal sigma = 3: coverage 0.65-0.72), and it
collapses when one outcome is carried only by negligible weights. The
minimum is never narrower than either. The price is over-coverage when a
rare outcome is down-weighted (importance weights that down-weight
successes: n_KG > n_Kish and the Kish interval is used, coverage ~1.0).
The minimum can still under-cover (0.5-0.85) when the weights are *both*
outcome-correlated and heavy-tailed, or when very few records carry the
successes; there, bootstrap over independent replicas instead.
At q = 0 or q = 1, V = 0 carries no information and n_eff = n_Kish (the
interval then has its boundary endpoint 0 or 1). With unit weights
n_eff = n exactly.

This assumes records are independent. Walkers of one weighted-ensemble run
are correlated (shared ancestry); for WE data the error bar must come from a
bootstrap over independent WE runs (design 4.5), not from this interval.

One configuration per call: q_B(X) is defined per configuration X, so
records from more than one ``frame_id`` raise ``ValueError`` unless
``allow_multiple_frames=True`` (pooling is only an independent-records
estimate when every frame contributes one shot; shots grouped by frame need
the frame bootstrap, :func:`cytherea.estimate.hierarchical_bootstrap`).
Records of ``kind == "segment"`` (WE walkers) are always rejected.

Stop reasons: "A", "B" (outcomes), "timeout" (unresolved; counted and
weighed against the 5% rule) and "nonfinite" (contract K4: the observable
or t became NaN/inf). "nonfinite" is neither an outcome nor a timeout: it
is counted in ``n_nonfinite``, excluded from q and its CI, and any
occurrence sets ``valid = False`` (a blown-up trajectory is a defect of the
run, not a statistical outcome). Any other reason raises ``ValueError``.

``timeout_frac`` is the weighted timeout fraction W_timeout / W_total, with
W_total the weight of *all* records (A, B, timeout, nonfinite): the
probability mass that never resolved; the count fraction for unit weights.
``n_A``/``n_B``/``n_timeout``/``n_nonfinite`` are always raw record counts.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import beta as _beta

from cytherea.estimate.records import shot_weights
from cytherea.store import ShotRecord

TIMEOUT_LIMIT = 0.05  # design 4.2: timeout fraction above 5% invalidates the estimate


@dataclasses.dataclass
class CommittorEstimate:
    q: float
    ci: tuple[float, float]
    n_A: int
    n_B: int
    n_timeout: int
    timeout_frac: float
    valid: bool
    n_nonfinite: int = 0


@dataclasses.dataclass
class _TwoOutcome:
    p: float
    ci: tuple[float, float]
    n_success: int
    n_failure: int
    n_timeout: int
    timeout_frac: float
    valid: bool
    n_nonfinite: int


def jeffreys_interval(p, n_eff, cp_tails: bool = False):
    """Equal-tailed 95% Jeffreys interval at (x = p n_eff, n_eff), elementwise.

    The 2.5 / 97.5 percentiles of Beta(x + 1/2, n_eff - x + 1/2), with the
    boundary modification lower = 0 at p = 0 and upper = 1 at p = 1 (Brown,
    Cai & DasGupta 2001). ``n_eff`` need not be an integer. Arrays broadcast.

    ``cp_tails=True`` (used for T(tau)) additionally takes the Clopper-Pearson
    bound where at most one count sits in a tail: upper >= 1 - 0.025^(1/n_eff)
    at x = 0, lower <= 1 - 0.975^(1/n_eff) at 0 < x <= 1 (mirrored at the top).
    Plain Jeffreys dips to ~0.88 coverage at expected counts ~2.5 (the x = 0
    upper bound ~2.5/n is too tight) and to ~0.90 at ~0.1; with the CP tails
    the exact coverage is >= 0.92 for every p at n = 30-1000.
    """
    p = np.asarray(p, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    x = p * n_eff
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(p <= 0, 0.0, _beta.ppf(0.025, x + 0.5, n_eff - x + 0.5))
        hi = np.where(p >= 1, 1.0, _beta.ppf(0.975, x + 0.5, n_eff - x + 0.5))
        if cp_tails:
            hi = np.where(p <= 0, np.maximum(hi, 1 - 0.025 ** (1 / n_eff)), hi)
            lo = np.where(p >= 1, np.minimum(lo, 0.025 ** (1 / n_eff)), lo)
            lo = np.where((p > 0) & (x <= 1), np.minimum(lo, 1 - 0.975 ** (1 / n_eff)), lo)
            hi = np.where((p < 1) & (n_eff - x <= 1), np.maximum(hi, 0.975 ** (1 / n_eff)), hi)
    return lo, hi


def _reject_segments(records: Sequence[ShotRecord]) -> None:
    for r in records:
        if r.kind == "segment":
            raise ValueError("records of kind 'segment' (WE walkers, correlated by ancestry) cannot be "
                             "pooled into a Jeffreys interval; bootstrap over independent WE runs")


def _check_single_frame(records: Sequence[ShotRecord], allow_multiple: bool) -> None:
    """q_B(X) is per configuration: one frame_id per call unless allowed."""
    _reject_segments(records)
    frames = {r.frame_id for r in records}
    if len(frames) > 1 and not allow_multiple:
        raise ValueError(f"records come from {len(frames)} different frame_ids; q_B(X) is defined per "
                         "configuration -- pass allow_multiple_frames=True to pool them anyway")


def _check_unclustered(records: Sequence[ShotRecord], allow_clustered: bool) -> None:
    """Pooling configurations is fine with one shot each; several shots per
    frame from several frames are clustered (independence fails)."""
    _reject_segments(records)
    per_frame = Counter(r.frame_id for r in records)
    if allow_clustered or len(per_frame) <= 1:
        return
    repeated = sum(c > 1 for c in per_frame.values())
    if repeated:
        raise ValueError(f"{repeated} of {len(per_frame)} frame_ids carry several shots each: the records "
                         "are clustered by frame and the independent-records interval would be too narrow; "
                         "use a frame bootstrap or pass allow_clustered_frames=True")


def _resolve_weights(records: Sequence[ShotRecord], weights) -> list[float]:
    """`weights` if given (one per record), else `shot_weights(records)`
    (contract K11: frame weights of an enumerated design)."""
    if weights is None:
        return [float(w) for w in shot_weights(records)]
    w = [float(x) for x in weights]
    if len(w) != len(records):
        raise ValueError(f"weights has {len(w)} entries for {len(records)} records")
    return w


def _two_outcome(records: Sequence[ShotRecord], success: str, failure: str,
                 weights: Sequence[float]) -> _TwoOutcome:
    """Weighted success fraction among {success, failure}; timeouts and
    nonfinite stops counted separately (module docstring)."""
    allowed = (success, failure, "timeout", "nonfinite")
    n = dict.fromkeys(allowed, 0)
    w_timeout = w_nonfinite = 0.0
    w_res, y_res = [], []
    for r, wr in zip(records, weights):
        if r.stop_reason not in n:
            raise ValueError(f"stop_reason {r.stop_reason!r} is not one of {allowed}")
        if not (math.isfinite(wr) and wr >= 0):
            raise ValueError(f"record weight must be finite and >= 0, got {wr!r}")
        n[r.stop_reason] += 1
        if r.stop_reason == "timeout":
            w_timeout += wr
        elif r.stop_reason == "nonfinite":
            w_nonfinite += wr
        else:
            w_res.append(wr)
            y_res.append(r.stop_reason == success)
    w = np.asarray(w_res, dtype=float)
    y = np.asarray(y_res, dtype=float)
    W = math.fsum(w)
    w_total = W + w_timeout + w_nonfinite
    if w_total <= 0:
        raise ValueError("no records with positive weight")
    timeout_frac = w_timeout / w_total
    if W > 0:
        p = math.fsum(w * y) / W
        V = math.fsum(w**2 * (y - p) ** 2) / W**2
        n_eff = W**2 / math.fsum(w**2)  # Kish
        if V > 0:  # p in {0, 1} gives V = 0: no linearised information, keep Kish
            n_eff = min(n_eff, p * (1 - p) / V)
        lo, hi = (float(v) for v in jeffreys_interval(p, n_eff))
    else:
        p, lo, hi = math.nan, math.nan, math.nan
    valid = timeout_frac <= TIMEOUT_LIMIT and W > 0 and n["nonfinite"] == 0
    return _TwoOutcome(p=p, ci=(lo, hi), n_success=n[success], n_failure=n[failure],
                       n_timeout=n["timeout"], timeout_frac=timeout_frac, valid=bool(valid),
                       n_nonfinite=n["nonfinite"])


def estimate_committor(records: Iterable[ShotRecord], allow_multiple_frames: bool = False,
                       weights: Sequence[float] | None = None) -> CommittorEstimate:
    """q_B = N_B / (N_A + N_B) with (weighted) Jeffreys 95% CI (module docstring).

    Only stop reasons "A", "B", "timeout" and "nonfinite" are accepted
    (anything else raises ``ValueError``). ``valid`` is False when
    ``timeout_frac > 0.05``, when any record stopped "nonfinite", or when no
    shot resolved (then ``q`` and the CI are nan). Records must come from a
    single ``frame_id`` unless ``allow_multiple_frames=True``; WE segments
    are rejected. Weights: `cytherea.estimate.shot_weights` (contract K11:
    pooled frames of an enumerated design count with their frame weight),
    unless ``weights`` (one per record) is given.
    """
    records = list(records)
    _check_single_frame(records, allow_multiple_frames)
    r = _two_outcome(records, "B", "A", _resolve_weights(records, weights))
    return CommittorEstimate(q=r.p, ci=r.ci, n_A=r.n_failure, n_B=r.n_success,
                             n_timeout=r.n_timeout, timeout_frac=r.timeout_frac, valid=r.valid,
                             n_nonfinite=r.n_nonfinite)
