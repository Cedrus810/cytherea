"""NAM b-surface association probability and k_on (design 4.3).

    beta_inf = beta / (1 - (1 - beta) * Omega),  Omega = k_D(b) / k_D(q) = b / q,
    k_D(r) = 4 pi D_AB r,                        k_on = k_D(b) * beta_inf.

Units are whatever the caller uses: with D_AB in nm^2/ps and b in nm, k_on is
in nm^3/ps (per molecule pair). No unit conversion is done here.

beta_inf is strictly increasing in beta (d beta_inf / d beta =
(1 - Omega) / denominator^2 > 0), so the Jeffreys interval on beta maps
exactly onto intervals for beta_inf and k_on by transforming its endpoints.
The k_on interval therefore reflects the sampling uncertainty of beta only:
the uncertainty of D_AB (and of b, q) is *not* included.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

from cytherea.estimate.committor import _check_unclustered, _resolve_weights, _two_outcome
from cytherea.store import ShotRecord


@dataclasses.dataclass
class KonEstimate:
    """``ci`` is the 95% interval of ``kon``; ``beta_ci`` and ``beta_inf_ci``
    are the corresponding intervals of ``beta`` and ``beta_inf``. ``ci``
    covers the sampling uncertainty of beta only, not that of D_AB."""

    beta: float
    beta_inf: float
    kon: float
    ci: tuple[float, float]
    beta_ci: tuple[float, float]
    beta_inf_ci: tuple[float, float]
    n_reaction: int
    n_escape: int
    n_timeout: int
    timeout_frac: float
    valid: bool
    n_nonfinite: int = 0


def _check_surfaces(b: float, q: float) -> None:
    if not (math.isfinite(b) and b > 0 and b < q) or math.isnan(q):
        raise ValueError(f"need 0 < b < q (q may be inf), got b={b!r}, q={q!r}")


def nam_beta_inf(beta: float, b: float, q: float) -> float:
    """beta_inf = beta / (1 - (1 - beta) b / q), design 4.3. ``q = inf`` gives beta."""
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    _check_surfaces(b, q)
    omega = b / q
    return beta / (1.0 - (1.0 - beta) * omega)


def estimate_kon(records: Iterable[ShotRecord], b: float, q: float, D_AB: float,
                 allow_clustered_frames: bool = False,
                 weights: Sequence[float] | None = None) -> KonEstimate:
    """beta = N_reaction / (N_reaction + N_escape) from shots started on the
    b-surface; beta_inf and k_on per design 4.3.

    Only stop reasons "reaction", "escape", "timeout" and "nonfinite" are
    accepted. Timeouts are counted, never dropped: ``valid`` is False when
    ``timeout_frac > 0.05``. "nonfinite" stops (contract K4) are neither an
    outcome nor a timeout: counted in ``n_nonfinite``, excluded from beta,
    and any occurrence sets ``valid = False``. Interval and weight handling
    as in :mod:`cytherea.estimate.committor` (weighted Jeffreys 95%,
    n_eff = min(Korn-Graubard, Kish); independent records assumed). The CI
    excludes D_AB uncertainty.

    beta is an average over b-surface configurations, so records from many
    frames are pooled -- but only as independent records: if several frames
    each carry several shots the records are clustered and ``ValueError`` is
    raised unless ``allow_clustered_frames=True``. WE segments are rejected.
    The b-surface frames enter with their frame weights
    (`cytherea.estimate.shot_weights`, contract K11) unless ``weights`` (one
    per record) is given.
    """
    _check_surfaces(b, q)
    if not (math.isfinite(D_AB) and D_AB > 0):
        raise ValueError(f"D_AB must be finite and > 0, got {D_AB!r}")
    records = list(records)
    _check_unclustered(records, allow_clustered_frames)
    r = _two_outcome(records, "reaction", "escape", _resolve_weights(records, weights))
    k_D = 4.0 * math.pi * D_AB * b
    if math.isnan(r.p):
        nan2 = (math.nan, math.nan)
        return KonEstimate(math.nan, math.nan, math.nan, nan2, nan2, nan2,
                           r.n_success, r.n_failure, r.n_timeout, r.timeout_frac, r.valid,
                           n_nonfinite=r.n_nonfinite)
    bi = nam_beta_inf(r.p, b, q)
    bi_ci = (nam_beta_inf(r.ci[0], b, q), nam_beta_inf(r.ci[1], b, q))
    return KonEstimate(beta=r.p, beta_inf=bi, kon=k_D * bi, ci=(k_D * bi_ci[0], k_D * bi_ci[1]),
                       beta_ci=r.ci, beta_inf_ci=bi_ci, n_reaction=r.n_success,
                       n_escape=r.n_failure, n_timeout=r.n_timeout,
                       timeout_frac=r.timeout_frac, valid=r.valid, n_nonfinite=r.n_nonfinite)
