"""Fixed-lag transition matrix T(tau) and the Chapman-Kolmogorov test (design 4.1).

``estimate_T`` works on shooting data: one (start state, end state, weight)
triple per fixed-lag shot. ``ck_test`` works on continuous discrete
trajectories, because CK needs the same process observed at several lags.

Intervals of T (95%, per element; marginal, not joint)
------------------------------------------------------
* ``reversible=False``: a per-element *confidence* interval (Jeffreys
  with Clopper-Pearson tails; fixreview-p6 minor 2: not a posterior -- no
  joint prior on a row has Beta(1/2, 1/2) marginals, their means would sum
  to m/2). Row i of T is a multinomial; the marginal of "end in j" versus
  "end elsewhere" is binomial with probability T_ij, and the interval is
  the equal-tailed Jeffreys interval for that binomial: the 2.5 / 97.5
  percentiles of
  Beta(x + 1/2, n_eff - x + 1/2) with x = T_ij n_eff (lower = 0 when
  T_ij = 0, upper = 1 when T_ij = 1), with Clopper-Pearson bounds in the
  tails where at most one count sits: upper = 1 - 0.025^(1/n_eff) at x = 0
  (the "rule of three" bound), lower = 1 - 0.975^(1/n_eff) at x <= 1, and
  mirrored at the top (``jeffreys_interval(..., cp_tails=True)``). Exact
  coverage is about >= 0.91 for every T_ij at n >= 20 (minimum 0.914 at
  n = 50, 0.918 at n = 20, 0.899 at n = 10; plain Jeffreys dips to 0.88 at
  an expected count of ~2.5); at expected counts 1 / 2 / 5 it is
  0.98 / 0.98 / 0.93. These intervals are marginal: joint or derived
  quantities (row comparisons, pi, fluxes, ITS) must come from the
  bootstrap, never from combining per-element intervals.
  With few frames per state and strong frame-to-frame variation the
  effective number of clusters is small and every interval under-covers
  (inherent, not specific to this estimator; fixreview-p6 minor 1): at 20
  frames x 50 shots and design effects 5-25, coverage is 0.78-0.88; at 100
  frames it recovers to 0.91. Design shooting campaigns with >= 50 frames
  per state and <= 10 shots per frame, and report ``TEstimate.n_eff``. With unit weights and no frame clustering the
  interval is a function of (C_ij, n_i) only.
  Why not the percentile bootstrap (used before): for an unobserved
  transition every replicate is 0 and the interval collapsed to [0, 0],
  and at expected counts 1 / 2 it covered 0.66 / 0.85. Why the marginal
  Beta(1/2, 1/2) and not a joint Dirichlet(1/2, ..., 1/2) row prior: the
  joint prior adds m/2 pseudo-counts to every row of an m-state matrix,
  which at m ~ 100 and ~100 shots per state biases the large elements by
  tens of percent; the marginal prior has the same per-element calibration
  at every m.
  The effective size n_eff follows :mod:`cytherea.estimate.committor`
  (R29), with the resampling unit (frame if ``frame_ids`` given, else shot)
  as the cluster: n_eff_ij = min(n_KG_ij, n_Kish_i), where
  n_Kish_i = (sum w)^2 / sum w^2 over the shots of row i and
  n_KG_ij = T_ij (1 - T_ij) / V_ij with the linearised (ratio-estimator)
  variance V_ij = sum_u (C_uj - T_ij W_u)^2 / W_i^2 over the units u of
  row i (C_uj: weight of unit u's shots ending in j; W_u: its total
  weight). At T_ij in {0, 1} V carries no information: without frames
  n_Kish_i is used (as in the committor); with frames the row's largest
  design effect n_Kish_i / n_eff_ij over its interior elements is borrowed,
  and if the row has no interior element the unit-level Kish size (the
  weighted number of frames) is used, i.e. the shots of a frame are taken
  as perfectly correlated (conservative).
* ``reversible=True``: percentile interval (2.5 / 97.5) of the reversible
  MLE over the bootstrap replicates, widened to contain the non-reversible
  Jeffreys interval above for every element whose effective count
  min(x, n_eff - x) is below ``SPARSE_COUNT`` = 5 (there the percentile
  bootstrap under-covers; from ~5 counts on it reaches 0.95, reviewer probe
  p4). Reversibility adds no information about a rare T_ij of a 2-state
  chain and only indirect information (via the jointly estimated pi) in
  larger ones, so the widening is conservative.

Bootstrap (reversible CIs and the implied-timescale CI ``its_ci_*``)
-------------------------------------------------------------------
Resampling is *stratified by start state* (the number of shots/frames
launched from each state is fixed by the shooting design). Without
``frame_ids`` the unit is the shot, which assumes shots are independent;
with ``frame_ids`` the unit is the frame (cluster bootstrap: frames drawn
with replacement within each start state, each bringing all of its shots).
A replicate is represented by the multiplicity vector of its units (how
often each unit was drawn, per state), so its counts are ``M @ C_unit``: the same
statistics as drawing units with replacement, with memory O(n_boot n^2)
plus one bounded block, independent of the number of units. Units are put
into a canonical order first, so results depend only on the *set* of
shots, not on the order in which records came out of the store.

Degenerate replicates are handled and counted (``n_boot_degenerate``),
never aborting: a reversible replicate whose counts are not strongly
connected (a rare connecting pair was lost) is estimated block by block on
its strongly connected components (0 between blocks; a state whose counts
all leave its block keeps its row-normalised row); a replicate with a
zero-weight row (every resampled shot of a state has weight 0) is dropped
from the implied-timescale interval.

Reversible MLE weighting (reviewer M8): the deeptime reversible MLE sees
the weighted counts, so a row's influence on the joint pi is its weight
mass, not its number of shots. With frame weights normalised per state a
heavily sampled rare state then barely constrains pi (consistent, but
inefficient). Rescale rows to their effective size before calling if that
matters.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence

import numpy as np
from deeptime.markov.tools.estimation import connected_sets, is_connected, largest_connected_set
from deeptime.markov.tools.estimation import transition_matrix as _dt_transition_matrix

from cytherea.estimate.committor import jeffreys_interval

SPARSE_COUNT = 5.0  # effective count below which the reversible percentile CI is widened
CK_MIN_TRAJ = 10  # below this many trajectories the CK bootstrap test is anti-conservative
_BLOCK_ELEMS = 1 << 20  # max elements of one bootstrap block (multiplicities / replicate counts)


@dataclasses.dataclass
class TEstimate:
    """Result of :func:`estimate_T`.

    ``ci_low`` / ``ci_high``: per-element 95% intervals (module docstring).
    ``its_ci_low`` / ``its_ci_high``: 95% percentile intervals of the implied
    timescales over the bootstrap replicates (nearest rank, since replicates
    may have ``+inf`` timescales). ``n_boot_degenerate``: replicates that
    needed the degenerate-replicate handling (module docstring).
    ``n_nonfinite``: shots with stop reason "nonfinite" (contract K4; only
    known when ``stop_reasons`` is passed); ``valid`` is False iff it is > 0.
    ``n_eff``: the per-element effective sizes behind the intervals (module
    docstring); their ratio to the shot count is the design effect to report.

    ``ck_passed`` / ``ck_max_dev`` are *not* evaluated by ``estimate_T``
    (single-lag pairs cannot be CK-tested): they are ``None`` / ``nan``
    meaning "not tested". Run :func:`ck_test` and fill them with
    ``dataclasses.replace(est, ck_passed=res.passed, ck_max_dev=res.max_dev)``.
    """

    T: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    its: np.ndarray
    ck_passed: bool | None
    ck_max_dev: float
    its_ci_low: np.ndarray
    its_ci_high: np.ndarray
    n_boot_degenerate: int = 0
    n_nonfinite: int = 0
    valid: bool = True
    n_eff: np.ndarray | None = None


@dataclasses.dataclass(eq=False)
class CKResult:
    """Result of :func:`ck_test`.

    ``passed`` / ``max_dev``: the test decision and the observed statistic D.
    ``active_set``: the states tested (largest strongly connected set of the
    pooled lag-tau counts); ``excluded_states``: the other states of
    ``[0, n_states)`` (unvisited, or not strongly connected to the rest),
    which play no part in the test. ``n_boot_degenerate``: replicates in
    which at least one active state had no outgoing counts at some lag (its
    rows were left out of that replicate's max, see :func:`ck_test`);
    ``degenerate_frac = n_boot_degenerate / n_boot``. ``rows_missing``:
    lag (in steps) -> active states with no outgoing counts in the data at
    that lag (left out of D and of every replicate at that lag).

    For backward compatibility the object also behaves as the 2-tuple
    ``(passed, max_dev)`` (unpacking and indexing).
    """

    passed: bool
    max_dev: float
    active_set: np.ndarray
    excluded_states: np.ndarray
    n_boot: int
    n_boot_degenerate: int
    rows_missing: dict[int, tuple[int, ...]]

    @property
    def degenerate_frac(self) -> float:
        return self.n_boot_degenerate / self.n_boot

    def __iter__(self):
        return iter((self.passed, self.max_dev))

    def __getitem__(self, i):
        return (self.passed, self.max_dev)[i]

    def __len__(self) -> int:
        return 2


def _check_states(x: np.ndarray, n_states: int, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"{name} must be a 1-D integer array")
    if x.size and (x.min() < 0 or x.max() >= n_states):
        raise ValueError(f"{name} contains states outside [0, {n_states})")
    return x.astype(np.int64)


def _row_normalize(C: np.ndarray) -> np.ndarray:
    rows = C.sum(axis=-1, keepdims=True)
    return C / rows


def _row_normalize_fill(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalise a stack (..., m, m); empty rows become identity rows.
    Returns ``(T, empty)`` with ``empty`` of shape (..., m)."""
    s = C.sum(axis=-1, keepdims=True)
    T = np.divide(C, s, out=np.zeros_like(C), where=s > 0)
    empty = s[..., 0] <= 0
    m = C.shape[-1]
    T[..., np.arange(m), np.arange(m)] += empty
    return T, empty


def _require_rows(C: np.ndarray, what: str) -> None:
    empty = np.flatnonzero(C.sum(axis=1) <= 0)
    if empty.size:
        raise ValueError(f"no counts from state {int(empty[0])} {what}; cannot estimate its row of T")


def _reversible_T(C: np.ndarray) -> np.ndarray:
    if not is_connected(C, directed=True):
        raise ValueError("count matrix is not strongly connected; reversible MLE undefined")
    return np.asarray(_dt_transition_matrix(C, reversible=True, maxerr=1e-12))


def _reversible_T_replicate(C: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reversible MLE of one bootstrap replicate; returns ``(T, degenerate)``.

    Not strongly connected: MLE per strongly connected component (0 between
    components); a state whose counts all leave its component keeps its
    row-normalised row; a zero row stays nan (dropped from the ITS CI)."""
    if is_connected(C, directed=True):
        return np.asarray(_dt_transition_matrix(C, reversible=True, maxerr=1e-12)), False
    n = C.shape[0]
    T = np.zeros((n, n))
    for comp in connected_sets(C, directed=True):
        comp = np.asarray(comp)
        sub = C[np.ix_(comp, comp)]
        if comp.size > 1:  # a strongly connected block of size > 1 has all its rows > 0
            T[np.ix_(comp, comp)] = _dt_transition_matrix(sub, reversible=True, maxerr=1e-12)
        elif sub[0, 0] > 0:
            T[comp[0], comp[0]] = 1.0
        else:
            row = C[comp[0]]
            T[comp[0]] = row / row.sum() if row.sum() > 0 else np.nan
    return T, True


def _implied_timescales(T: np.ndarray, lag: float) -> np.ndarray:
    """-lag / ln|lambda| for the non-unit eigenvalues; |lambda| is clipped to
    <= 1 and |lambda| = 1 (a further unit eigenvalue, i.e. a disconnected T)
    gives +inf, never a negative or -inf timescale. |lambda| = 0 gives 0.
    ``T`` may be a stack (..., n, n)."""
    lam = np.sort(np.abs(np.linalg.eigvals(T)), axis=-1)[..., ::-1][..., 1:]
    lam = np.minimum(lam, 1.0)
    its = np.full(lam.shape, np.inf)
    below = lam < 1.0
    with np.errstate(divide="ignore"):
        its[below] = -lag / np.log(lam[below])
    return its


def _multiplicities(rng: np.random.Generator, n_units: int, size: int) -> np.ndarray:
    """(size, n_units): how often each unit occurs in a draw of n_units units
    with replacement (one row per replicate; each row is Multinomial(n_units,
    uniform)). Drawn as picks + bincount, ~6x faster than rng.multinomial."""
    pick = rng.integers(0, n_units, size=(size, n_units))
    pick += np.arange(size)[:, None] * n_units
    return np.bincount(pick.ravel(), minlength=size * n_units).reshape(size, n_units)


def _row_n_eff(C_u: np.ndarray, W_u: np.ndarray, w2: float, clustered: bool) -> np.ndarray:
    """Per-element effective size of one row of T (module docstring).

    ``C_u`` (U, n): weighted end counts per unit; ``W_u`` (U,): weight mass
    per unit; ``w2``: sum of the squared *shot* weights of the row."""
    W = W_u.sum()
    T = C_u.sum(axis=0) / W
    n_kish = W**2 / w2
    V = ((C_u - T[None, :] * W_u[:, None]) ** 2).sum(axis=0) / W**2
    interior = (T > 0) & (T < 1)
    n_kg = np.full(T.shape, np.inf)
    np.divide(T * (1 - T), V, out=n_kg, where=interior & (V > 0))
    n_eff = np.minimum(n_kg, n_kish)
    if not clustered:
        deff = 1.0
    elif interior.any():
        deff = n_kish / n_eff[interior].min()
    else:
        deff = n_kish / (W**2 / np.sum(W_u**2))
    n_eff[~interior] = n_kish / deff
    return n_eff


def estimate_T(
    start_states: np.ndarray,
    end_states: np.ndarray,
    weights: np.ndarray,
    n_states: int,
    lag: float,
    reversible: bool,
    n_boot: int,
    rng: np.random.Generator,
    frame_ids: np.ndarray | None = None,
    stop_reasons: Sequence[str] | None = None,
) -> TEstimate:
    """Estimate T(lag) from fixed-lag shots (design 4.1).

    Count matrix ``C_ij = sum of weights of shots with start i, end j``.
    Design 4.1 writes ``T_ij = sum_k w_k n_(k->j) / sum_k w_k`` with frame
    weight ``w_k`` and ``n_(k->j)`` the *fraction* of frame k's K_k shots that
    end in j; to reproduce it, give every shot of frame k the weight
    ``w_k / K_k``.
    ``reversible=False``: ``T = C`` row-normalized. ``reversible=True``:
    deeptime reversible maximum-likelihood estimate on ``C`` (weighted counts
    are used as-is; the data must be strongly connected). A state with zero
    total start weight raises ``ValueError`` naming the state (rows are
    never invented).

    ``frame_ids`` (optional, one per shot): the frame each shot was launched
    from. When given, frames are the clustering and resampling unit (module
    docstring); every frame must belong to a single start state, and every
    start state needs >= 2 frames (with one frame the between-frame
    variability is unobservable and the interval would be too narrow). With
    only 2-3 frames per state the intervals still under-cover.

    ``stop_reasons`` (optional, one per shot; contract K4): allowed values
    are "fixed_lag" and "nonfinite". "nonfinite" shots are dropped before
    counting (their end state is meaningless and is not checked), counted
    in ``n_nonfinite``, and make ``valid`` False. Any other reason raises
    ``ValueError``. (Without ``stop_reasons`` the caller must have filtered
    them; ``n_nonfinite`` is then 0.)

    ``its`` are ``-lag / ln|lambda_k|`` for the non-unit eigenvalues, largest
    first; ``lag`` is in whatever time unit the caller uses. CIs: see module
    docstring. CK fields are not evaluated here (see :class:`TEstimate`).
    """
    start = _check_states(start_states, n_states, "start_states")
    end_raw = np.asarray(end_states)
    w = np.asarray(weights, dtype=float)
    if not (start.shape == end_raw.shape == w.shape):
        raise ValueError("start_states, end_states and weights must have equal length")
    fid = None if frame_ids is None else np.asarray(frame_ids)
    if fid is not None and fid.shape != start.shape:
        raise ValueError("frame_ids must have one entry per shot")
    n_nonfinite = 0
    if stop_reasons is not None:
        reasons = np.asarray(stop_reasons, dtype=object)
        if reasons.shape != start.shape:
            raise ValueError("stop_reasons must have one entry per shot")
        bad = sorted({str(r) for r in reasons} - {"fixed_lag", "nonfinite"})
        if bad:
            raise ValueError(f"stop_reason {bad[0]!r} is not one of ('fixed_lag', 'nonfinite')")
        keep = reasons != "nonfinite"
        n_nonfinite = int((~keep).sum())
        start, end_raw, w = start[keep], end_raw[keep], w[keep]
        fid = None if fid is None else fid[keep]
    end = _check_states(end_raw, n_states, "end_states")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and >= 0")
    if not (np.isfinite(lag) and lag > 0):
        raise ValueError("lag must be finite and > 0")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    n = n_states
    # canonical order (M3): shots sorted by content (within their frame), frames
    # by id, so everything below depends on the set of shots only
    if fid is None:
        perm = np.lexsort((w, end, start))
        start, end, w = start[perm], end[perm], w[perm]
        unit = np.arange(start.size)
    else:
        _, unit = np.unique(fid, return_inverse=True)
        unit = unit.ravel()
        perm = np.lexsort((w, end, unit, start))
        start, end, w, unit = start[perm], end[perm], w[perm], unit[perm]

    C = np.bincount(start * n + end, weights=w, minlength=n * n).reshape(n, n)
    _require_rows(C, "(total start weight is zero)")
    T = _reversible_T(C) if reversible else _row_normalize(C)

    if fid is not None:
        unit_state = np.full(int(unit.max()) + 1, -1)
        unit_state[unit] = start
        if np.any(unit_state[unit] != start):
            raise ValueError("a frame id occurs with more than one start state")
        frames_per_state = np.bincount(unit_state, minlength=n)
        few = np.flatnonzero(frames_per_state < 2)
        if few.size:
            raise ValueError(f"start state {int(few[0])} has {int(frames_per_state[few[0]])} frame(s); "
                             "the frame bootstrap needs >= 2 frames per state")

    # One start state at a time (shots are sorted by start state): the per-unit
    # counts C_u (units x n) of that state give the effective sizes of its row
    # (I1) and, times multiplicity vectors, its bootstrap rows (I2). Memory:
    # one state's C_u, one block of multiplicities, and C_boot (n_boot n^2).
    T_nr = _row_normalize(C)
    n_eff = np.empty((n, n))
    C_boot = np.zeros((n_boot, n, n))
    bounds = np.searchsorted(start, np.arange(n + 1))
    for i in range(n):
        sl = slice(bounds[i], bounds[i + 1])
        _, local = np.unique(unit[sl], return_inverse=True)
        n_u = int(local.max()) + 1
        C_u = np.bincount(local * n + end[sl], weights=w[sl], minlength=n_u * n).reshape(n_u, n)
        n_eff[i] = _row_n_eff(C_u, C_u.sum(axis=1), float(np.sum(w[sl] ** 2)), clustered=fid is not None)
        block = max(1, min(n_boot, _BLOCK_ELEMS // n_u))
        for b0 in range(0, n_boot, block):
            b1 = min(n_boot, b0 + block)
            C_boot[b0:b1, i, :] = _multiplicities(rng, n_u, b1 - b0) @ C_u
    j_lo, j_hi = jeffreys_interval(T_nr, n_eff, cp_tails=True)
    zero_row = np.any(C_boot.sum(axis=2) <= 0, axis=1)
    if reversible:
        T_boot = np.empty_like(C_boot)
        degenerate = zero_row.copy()
        for b in range(n_boot):
            T_boot[b], disconnected = _reversible_T_replicate(C_boot[b])
            degenerate[b] |= disconnected
        lo, hi = np.nanpercentile(T_boot, [2.5, 97.5], axis=0)
        x = T_nr * n_eff
        sparse = np.minimum(x, n_eff - x) < SPARSE_COUNT
        lo = np.where(sparse, np.minimum(lo, j_lo), lo)
        hi = np.where(sparse, np.maximum(hi, j_hi), hi)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            T_boot = _row_normalize(C_boot)
        degenerate = zero_row
        lo, hi = j_lo, j_hi

    its = _implied_timescales(T, lag)
    ok = ~np.isnan(T_boot).any(axis=(1, 2))
    if ok.any():
        its_b = _implied_timescales(T_boot[ok], lag)
        its_lo = np.percentile(its_b, 2.5, axis=0, method="lower")
        its_hi = np.percentile(its_b, 97.5, axis=0, method="higher")
    else:
        its_lo = its_hi = np.full(its.shape, np.nan)

    return TEstimate(T=T, ci_low=lo, ci_high=hi, its=its, ck_passed=None, ck_max_dev=float("nan"),
                     its_ci_low=its_lo, its_ci_high=its_hi, n_boot_degenerate=int(degenerate.sum()),
                     n_nonfinite=n_nonfinite, valid=n_nonfinite == 0, n_eff=n_eff)


def ck_test(
    dtrajs: list[np.ndarray],
    lag_steps: int,
    ks: Sequence[int],
    n_states: int,
    n_boot: int,
    rng: np.random.Generator,
) -> CKResult:
    """Chapman-Kolmogorov test T(k tau) ~ T(tau)^k (design 4.1), one global test.

    Active set: the test runs on the largest strongly connected set of the
    pooled lag-tau counts (``CKResult.active_set``). The other states of the
    common state space ``[0, n_states)`` -- e.g. states this variant never
    visits -- are excluded and reported, never an error. Counts are
    restricted to the active set (transitions to or from excluded states
    are dropped); fewer than 2 active states raise ``ValueError``.

    T(tau) and each T(k tau) are non-reversible (row-normalized), unweighted
    estimates from sliding-window counts over all ``dtrajs``: this tests the
    Markovianity of the discretisation, not a reversible or reweighted
    estimator. Test statistic

        D = max over k in ks and all elements ij of |T(tau)^k - T(k tau)|.

    Its null distribution comes from a bootstrap over whole trajectories
    (``n_boot`` replicates, drawn as multiplicity vectors after putting the
    trajectories into a canonical order) in which *both* terms are
    re-estimated from the same resampled trajectories, centred at the
    observed deviation:

        D*_b = max_k,ij |(T*_b(tau)^k - T*_b(k tau)) - (T(tau)^k - T(k tau))|.

    The test passes iff ``D <= 95th percentile of {D*_b}``: a single test at
    level 5% for all elements and lags together (no per-element multiple
    comparisons). The max is unstudentised, so it is dominated by the
    high-variance elements; non-Markovian deviations in low-variance, rarely
    populated rows have little power (a studentised max |dev| / sd* is the
    usual alternative).

    Degenerate replicates: if an active state has no outgoing counts in a
    replicate at some lag (a rarely visited state was not drawn), its row is
    set to the identity for the matrix power and its rows are left out of
    that replicate's max; such replicates are counted in
    ``CKResult.n_boot_degenerate``. Rows with no counts in the data itself at
    a lag k tau are left out of D and of every D*_b at that lag
    (``CKResult.rows_missing``).

    With fewer than 10 trajectories the bootstrap test is anti-conservative
    (false-fail rate of a Markov chain 22% with 3 trajectories, 10% with 5,
    6% with 10): a ``UserWarning`` is issued. Cut a single long trajectory
    into >= 10 blocks. Returns a :class:`CKResult` (also unpackable as
    ``(passed, max_dev)``).
    """
    if len(dtrajs) < 2:
        raise ValueError("ck_test bootstraps over trajectories and needs >= 2 trajectories")
    if lag_steps < 1:
        raise ValueError("lag_steps must be >= 1")
    ks = [int(k) for k in ks]
    if not ks or min(ks) < 1:
        raise ValueError("ks must be a non-empty sequence of integers >= 1")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    trajs = [_check_states(d, n_states, "dtraj") for d in dtrajs]
    trajs.sort(key=lambda d: (d.size, d.tobytes()))  # canonical order (M3)
    if len(trajs) < CK_MIN_TRAJ:
        warnings.warn(f"ck_test with {len(trajs)} < {CK_MIN_TRAJ} trajectories: the bootstrap test is "
                      "anti-conservative (it false-fails Markov chains well above 5%); cut long "
                      "trajectories into >= 10 blocks", UserWarning, stacklevel=2)
    n = n_states

    C_tau = np.zeros(n * n)
    for d in trajs:
        if d.size > lag_steps:
            C_tau += np.bincount(d[:-lag_steps] * n + d[lag_steps:], minlength=n * n)
    C_tau = C_tau.reshape(n, n)
    active = np.sort(np.asarray(largest_connected_set(C_tau, directed=True), dtype=np.int64))
    if active.size < 2:
        raise ValueError(f"the active set (largest strongly connected set of the lag-{lag_steps} "
                         f"counts) has {active.size} state(s); CK needs >= 2")
    excluded = np.setdiff1d(np.arange(n), active)
    m = active.size
    to_active = np.full(n, -1)
    to_active[active] = np.arange(m)

    def per_traj_counts(lag: int) -> np.ndarray:
        out = np.zeros((len(trajs), m * m))
        for t, d in enumerate(trajs):
            if d.size > lag:
                a, b = to_active[d[:-lag]], to_active[d[lag:]]
                keep = (a >= 0) & (b >= 0)
                out[t] = np.bincount(a[keep] * m + b[keep], minlength=m * m)
        return out

    counts = {lag: per_traj_counts(lag) for lag in {lag_steps, *(k * lag_steps for k in ks)}}

    T1, _ = _row_normalize_fill(counts[lag_steps].sum(axis=0).reshape(m, m))  # no empty row (SCC)
    dev, missing = {}, {}
    D = 0.0
    for k in ks:
        Tk, empty_k = _row_normalize_fill(counts[k * lag_steps].sum(axis=0).reshape(m, m))
        dev[k] = np.linalg.matrix_power(T1, k) - Tk
        missing[k] = empty_k
        if empty_k.all():
            # fixreview-p6 N1: never "pass" a lag that has no data at all
            raise ValueError(
                f"ck_test: no data at lag {k * lag_steps} steps (k={k}): no trajectory "
                f"has a pair of active-set frames {k * lag_steps} steps apart; use longer "
                "trajectories or smaller ks"
            )
        if empty_k.sum() >= m / 2:
            warnings.warn(
                f"ck_test: {int(empty_k.sum())} of {m} active-set rows have no data at lag "
                f"{k * lag_steps} steps; those rows are untested there (rows_missing)",
                stacklevel=2,
            )
        D = max(D, float(np.abs(dev[k][~empty_k]).max()))
    rows_missing = {k * lag_steps: tuple(int(s) for s in active[missing[k]])
                    for k in ks if missing[k].any()}

    n_traj = len(trajs)
    D_boot = np.zeros(n_boot)
    degenerate = np.zeros(n_boot, dtype=bool)
    block = max(1, min(n_boot, _BLOCK_ELEMS // max(m * m, n_traj)))
    for b0 in range(0, n_boot, block):
        b1 = min(n_boot, b0 + block)
        M = _multiplicities(rng, n_traj, b1 - b0).astype(float)
        T1b, empty1 = _row_normalize_fill((M @ counts[lag_steps]).reshape(-1, m, m))
        degenerate[b0:b1] |= empty1.any(axis=1)
        for k in ks:
            Tkb, emptyk = _row_normalize_fill((M @ counts[k * lag_steps]).reshape(-1, m, m))
            degenerate[b0:b1] |= emptyk.any(axis=1)
            devb = np.abs(np.linalg.matrix_power(T1b, k) - Tkb - dev[k])
            use = ~(empty1 | emptyk | missing[k][None, :])  # (block, m): rows kept in the max
            devb = np.where(use[:, :, None], devb, 0.0)
            D_boot[b0:b1] = np.maximum(D_boot[b0:b1], devb.max(axis=(1, 2)))
    if degenerate.mean() > 0.05:
        warnings.warn(
            f"ck_test: {int(degenerate.sum())} of {n_boot} bootstrap replicates lost a row "
            "(degenerate); the test is conservative there -- add trajectories that visit the "
            "rare states",
            stacklevel=2,
        )
    return CKResult(passed=bool(D <= np.percentile(D_boot, 95)), max_dev=D, active_set=active,
                    excluded_states=excluded, n_boot=n_boot,
                    n_boot_degenerate=int(degenerate.sum()), rows_missing=rows_missing)


def ck_test_shots(
    start_states: np.ndarray,
    end_states_by_k: np.ndarray,
    ks: Sequence[int],
    n_states: int,
    n_boot: int,
    rng: np.random.Generator,
    frame_ids: np.ndarray | None = None,
    tau_only: dict | None = None,
) -> CKResult:
    """Chapman-Kolmogorov test T(k tau) ~ T(tau)^k from fixed-lag shots that
    record their state at several multiples of tau (design 4.1, Task 14.3).

    ``end_states_by_k[i, m]`` is the state of shot i at t = ``ks[m]`` tau;
    ``ks`` are distinct integers >= 1 and must contain 1. Every T is the
    row-normalised, unweighted estimate from the shots' t = 0 windows only
    (for shots started inside cores: the core-start T(k tau)), so this tests
    the shots' own Markovianity, not a comparison with a reference. Every
    state of ``[0, n_states)`` needs shots (a state reached at k tau must
    have its own row of T(tau)); a missing one raises ``ValueError`` naming
    it. Shots that stopped non-finite carry no end state and must be left
    out by the caller (and reported, contract K4).

    Statistic and null distribution as in :func:`ck_test`:
    D = max over k in ks and all ij of |T(tau)^k - T(k tau)|, and
    D*_b = max_k,ij |(T*_b(tau)^k - T*_b(k tau)) - (T(tau)^k - T(k tau))|
    with both terms re-estimated on the same bootstrap replicate. The
    bootstrap is stratified by start state (the shooting design fixes the
    number of shots per state); its unit is the frame when ``frame_ids`` is
    given (every frame in one start state; all its shots drawn together),
    else the shot. Passes iff D <= the 95th percentile of {D*_b}. Units are
    put in a canonical order first, so the result depends only on the set
    of shots (and their frame ids). Returns a :class:`CKResult` with all
    states active and ``rows_missing`` empty (stratification keeps every row).

    ``tau_only`` (optional): shots that ran only tau, as a dict with
    ``start`` and ``end`` (state at tau) and, when ``frame_ids`` is given,
    ``frame_ids`` (required then). T(tau) is then estimated from the long
    shots' first tau together with these shots; T(k tau) for k >= 2 from the
    long shots only (the k = 1 deviation is 0 by construction). A frame
    carries its shots of both kinds through the bootstrap, so a frame id must
    keep one start state across both. Without ``frame_ids`` every tau-only
    shot is its own unit.
    """
    start = _check_states(start_states, n_states, "start_states")
    ends = np.asarray(end_states_by_k)
    ks = [int(k) for k in ks]
    if not ks or min(ks) < 1 or len(set(ks)) != len(ks):
        raise ValueError("ks must be distinct integers >= 1")
    if 1 not in ks:
        raise ValueError("ks must contain 1 (T(tau) is estimated from the k = 1 column)")
    if ends.shape != (start.size, len(ks)):
        raise ValueError(f"end_states_by_k must have shape (n_shots, len(ks)) = ({start.size}, {len(ks)}), "
                         f"got {ends.shape}")
    ends = _check_states(ends.reshape(-1), n_states, "end_states_by_k").reshape(ends.shape)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    n = n_states
    if tau_only is not None:
        t_start = _check_states(tau_only["start"], n_states, "tau_only['start']")
        t_end = _check_states(tau_only["end"], n_states, "tau_only['end']")
        if t_end.shape != t_start.shape:
            raise ValueError("tau_only['end'] must have one entry per tau-only shot")
        if frame_ids is not None and "frame_ids" not in tau_only:
            raise ValueError("tau_only needs 'frame_ids' when frame_ids is given (frames are the bootstrap unit)")
    else:
        t_start = t_end = np.zeros(0, dtype=np.int64)
    counts_per_state = np.bincount(np.concatenate([start, t_start]), minlength=n)
    if (counts_per_state == 0).any():
        raise ValueError(f"no shots start in state {int(np.flatnonzero(counts_per_state == 0)[0])}; "
                         "every state needs its own row of T(tau)")

    if frame_ids is None:
        unit = np.lexsort((*ends.T[::-1], start))  # canonical shot order
        unit_of_shot = np.empty(start.size, dtype=np.int64)
        unit_of_shot[unit] = np.arange(start.size)
        t_unit = np.lexsort((t_end, t_start))
        unit_of_tau = np.empty(t_start.size, dtype=np.int64)
        unit_of_tau[t_unit] = start.size + np.arange(t_start.size)
        unit_state = np.concatenate([start[unit], t_start[t_unit]])
    else:
        fids = np.asarray(frame_ids)
        if fids.shape != start.shape:
            raise ValueError("frame_ids must have one entry per shot")
        t_fids = np.asarray(tau_only["frame_ids"]) if tau_only is not None else fids[:0]
        if t_fids.shape != t_start.shape:
            raise ValueError("tau_only['frame_ids'] must have one entry per tau-only shot")
        uniq, inv = np.unique(np.concatenate([fids, t_fids]), return_inverse=True)
        unit_state = np.full(uniq.size, -1)
        for u, s in zip(inv, np.concatenate([start, t_start])):
            if unit_state[u] not in (-1, s):
                raise ValueError(f"frame {uniq[u]!r} has shots from two start states")
            unit_state[u] = s
        order = np.lexsort((np.arange(uniq.size), unit_state))
        rank = np.empty(uniq.size, dtype=np.int64)
        rank[order] = np.arange(uniq.size)
        inv, unit_state = rank[inv], unit_state[order]
        unit_of_shot, unit_of_tau = inv[: start.size], inv[start.size:]
    n_units = unit_state.size

    # per-unit counts at every k: (len(ks), n_units, n*n); T(tau) also gets the tau-only shots
    C_u = np.zeros((len(ks), n_units, n * n))
    for m in range(len(ks)):
        np.add.at(C_u[m], (unit_of_shot, start * n + ends[:, m]), 1.0)
    i1 = ks.index(1)
    np.add.at(C_u[i1], (unit_of_tau, t_start * n + t_end), 1.0)

    def stat(C: np.ndarray) -> np.ndarray:
        """(..., len(ks), n, n) counts -> T(tau)^k - T(k tau) per k."""
        T = _row_normalize(C)
        T1 = T[..., i1, :, :]
        return np.stack([np.linalg.matrix_power(T1, k) - T[..., m, :, :] for m, k in enumerate(ks)], axis=-3)

    dev = stat(C_u.sum(axis=1).reshape(len(ks), n, n))
    D = float(np.abs(dev).max())

    by_state = [np.flatnonzero(unit_state == s) for s in range(n)]
    D_boot = np.zeros(n_boot)
    block = max(1, min(n_boot, _BLOCK_ELEMS // max(n_units, len(ks) * n * n)))
    for b0 in range(0, n_boot, block):
        b1 = min(n_boot, b0 + block)
        M = np.zeros((b1 - b0, n_units))
        for idx in by_state:
            M[:, idx] = _multiplicities(rng, idx.size, b1 - b0)
        Cb = np.einsum("bu,kuc->bkc", M, C_u).reshape(b1 - b0, len(ks), n, n)
        D_boot[b0:b1] = np.abs(stat(Cb) - dev).max(axis=(1, 2, 3))
    return CKResult(passed=bool(D <= np.percentile(D_boot, 95)), max_dev=D, active_set=np.arange(n),
                    excluded_states=np.zeros(0, dtype=np.int64), n_boot=n_boot, n_boot_degenerate=0,
                    rows_missing={})
