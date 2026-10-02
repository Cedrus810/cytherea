"""`run_batch`: an order-independent, crash-resumable batch shot executor
(Task 7).

For each key in `keys`, in order:

1. Skip it (fetch the already-recorded `ShotRecord` from `store` instead of
   re-running anything) if `store` already has a record for
   `key_digest(key)` -- this is what makes resuming after a crash cheap and
   correct: only the keys missing from `store` are (re-)run (Review Focus
   3, tests 7.4/7.4a/7.4b). A key with a logged failure and no record is
   skipped too, unless ``retry_failures`` covers its failure kind (see
   "Failures" below). Existing records are checked from their SQL summaries
   (`Store.summaries_many`, no payload decoding; fixreview-p1 N-I4) and,
   with the default ``return_records=True``, then read in one pass
   (`Store.get_many`).
2. Otherwise, run it -- via `shot_fn(key)` directly (in this process) when
   `n_workers == 1`, or dispatched to a `ProcessPoolExecutor` (spawn
   context) when `n_workers > 1`.
3. If `shot_fn` raises `ICRejectedError`, append a `FailureRecord` to the
   store's failure log and put a `ShotFailure` in the results list (batch
   continues). Any other exception propagates and aborts the batch (see
   "Prompt cancellation" below for the `n_workers > 1` case) -- unless
   ``on_error="record"``, in which case it is logged as a failure too and
   the batch continues (see "Failures" below).
4. Otherwise, `run_batch` checks the returned `ShotRecord` (right key, same
   config and code as the rest of the store -- see "Resume refuses a
   config or code change" below), appends it to `store`, then records it in
   the results list.

The results list always has `len(keys)` entries, one per key, **in `keys`
order** -- regardless of completion order, `n_workers`, or which keys were
skipped as already-present. With ``return_records=False`` a record's entry
is its `cytherea.store.RecordSummary` (key digest, kind, stop reason and the
three hashes) instead of the full `ShotRecord`, for stored and fresh shots
alike, so a batch of 10^5 shots never holds their series in memory (read
them with `Store.iter` when needed).

`run_batch` validates its own inputs eagerly, before doing any work:
`n_workers` must be `>= 1`, and `keys` must not contain two keys with the
same `key_digest` (both raise `ValueError`; controller ruling R35 #3).

`shot_fn` must not write to `store` itself (controller ruling R33)
---------------------------------------------------------------------
`run_batch` is the *sole* writer of the `store` it is given. `shot_fn` --
typically `functools.partial(run_shot, ..., store=None)` (see
`cytherea.engine.shot`'s module docstring) -- must only ever *compute and
return* a `ShotRecord`, never persist one anywhere itself.

An earlier version of this contract had `shot_fn` write to a private
"scratch" store instead of taking `store=None`. That had a real correctness
gap: a crash between `shot_fn` finishing (so the scratch store already has
the record) and `run_batch`'s own append into the *real* store would, on
resume, recompute that same key and try to write it into the *same* scratch
store again, raising `DuplicateKeyError` -- a `run_batch` call that used to
succeed would now raise partway through resuming, for a scenario resuming is
specifically supposed to handle. Removing all persistence from `shot_fn`
(via `store=None`) removes the scratch store -- and this failure mode --
entirely: nothing exists for `run_batch` to collide with on resume, because
nothing but `run_batch` itself ever writes anything, anywhere. It also
removes the parallel-mode variant of the same problem, where multiple
spawned workers each held their own `Store(scratch_path)` (which pickles as
just `{"_path": scratch_path}`) and could genuinely write to that same file
concurrently.

Parent-only, ordered writes
----------------------------
Whether `n_workers` is 1 or greater than 1, the append into `store` happens
exactly once per newly-run key, performed by this function itself, in *this*
process (the parent -- workers, if any, only ever compute and return a
`ShotRecord`/raise, they never touch `store` at all), and strictly in the
order the corresponding keys appear in `keys`. This holds even under
`n_workers > 1`, where the underlying shot computations run concurrently and
may *finish* out of order: `run_batch` submits every pending shot's future
up front, then resolves (`future.result()`, blocking as needed) and appends
them one at a time, in ascending `keys`-index order. This is what makes the
on-disk record set identical between a serial and a parallel run over the
same keys (test 7.1), and what keeps crash-resume deterministic (tests 7.4,
7.4a).

`on_before_append`
-------------------
An optional `Callable[[ShotRecord], None]`, called synchronously with each
freshly-computed record (never for a skipped, already-present one, and
never for an `ICRejectedError` -> `ShotFailure`), immediately before that
record's `store.append(record)` call. This is primarily a test seam: it
lets a test deterministically observe (or act at) the exact boundary
between "the shot finished" and "the parent persisted it" -- e.g. a test
verifying crash-resume can have this callback kill the current process the
instant a chosen key's shot finishes, reliably reproducing a crash in that
exact window without relying on timing (test 7.4b). It runs unconditionally
before every real append, so it can also serve as a lightweight progress
callback if a caller wants one.

Prompt cancellation on error (controller ruling R35 #2)
-----------------------------------------------------------
If a future raises anything other than `ICRejectedError`, `run_batch`
cancels every not-yet-started pending future
(`executor.shutdown(wait=False, cancel_futures=True)`) before re-raising,
so the batch stops promptly instead of waiting for every already-submitted
shot to finish first. Futures already running when the exception is
detected still run to completion (they cannot be interrupted mid-flight),
but nothing still queued gets started.

Picklability
-------------
Under `n_workers > 1`, `shot_fn` and every element of `keys` must be
picklable (the `ProcessPoolExecutor` uses the "spawn" start method): a
module-level function, or a `functools.partial` built from one, works; a
lambda or a locally-defined closure does not. `shot_fn` is shipped **once
per worker** (pool initializer), not once per key, so a sampler with a
large frame pool is not re-pickled for every shot (fullreview A-m5). Only
keys travel per task. Because appends happen in `keys` order, a crash
discards every result that was computed but is queued behind a slower,
earlier key; resume recomputes those.

Failures (fullreview A-I2)
---------------------------
IC-gate rejections are deterministic (the draws are keyed), so re-running
them on every resume is pointless and used to leave no trace in the store.
Each one is now appended to the store's append-only failure log
(`Store.append_failure`, with the reasons of every attempt). On a later
`run_batch` over the same store, a key with a logged failure and no record
is **skipped** and reported as a `ShotFailure` with ``from_store=True``,
unless ``retry_failures=True`` or a set naming its failure kind, e.g.
``retry_failures={"error"}`` (then it is attempted again and a new failure
row is appended if it fails again). With ``on_error="record"`` any
other ``Exception`` raised by `shot_fn` (e.g. OpenMM "Particle coordinate is
NaN") is logged the same way, as ``failure_kind="error"`` with the
traceback, and the batch continues; the default ``on_error="raise"`` keeps
the fail-fast behaviour and logs nothing. A broken process pool is never
recorded as a per-key failure. Estimators never see failures: they live in
their own table, outside `Store.iter`.

Resume refuses a config or code change (contract K8, fullreview A-I4/A-I7)
---------------------------------------------------------------------------
Keys only encode *which* shot, not *how* it was run, so resuming against a
store written with another physics config, shot protocol or other code used
to mix old and new records silently. `run_batch` now compares every relevant record with
the current batch and raises `ResumeConfigMismatchError` -- before anything
is run or appended -- on any difference in

- ``physics_config_hash`` (contract K9: ``config_hash(backend.
  effective_config(physics_cfg))``, which covers the analytic potential and
  dynamics parameters and the OpenMM System/Topology digests, platform and
  precision): the hash the current `shot_fn` records. It is
  taken from the ``physics_config_hash=`` argument if given, else from a
  ``physics_config_hash`` attribute of `shot_fn`, else computed exactly as
  `run_shot` would (`cytherea.engine.shot.resolve_physics_config`) when
  `shot_fn` is a ``functools.partial(run_shot, ..., backend=...,
  physics_cfg=...)``. For any other `shot_fn` it is only known once the
  first shot returns; the stored records are then checked at that point,
  still before the first append.
- ``protocol_hash`` (ruling R39): the stop rule and its parameters, the
  ObsSpec, the IC sampler/frame pool and the labeler's spec, if any
  (`cytherea.engine.shot.protocol_description`). Known up front from the
  ``protocol_hash=`` argument, a ``protocol_hash`` attribute of `shot_fn`,
  or a ``functools.partial(run_shot, ..., stop=..., obs=..., sampler=...
  [, labeler=...])``;
  otherwise from the first fresh record, as for the physics hash. A stored
  record without a protocol_hash (written before R39) counts as a mismatch.
- ``code_version``, compared by `cytherea.engine.shot.code_identity` (the
  ``"+dirty"`` suffix and the informational git commit are ignored; the
  source hash is not). The reference is this process's `code_version()`,
  resolved once per batch; freshly computed records must match it too, so a
  worker running other code than the parent (e.g. a checkout changed
  mid-batch) is caught rather than stamped over.

Logged failures that would be skipped are checked the same way. Pass
``allow_code_change=True`` to accept a different ``code_version`` only
(fixreview-p1 m-2: routine after any commit under R38, since the package is
installed editable), or ``allow_config_change=True`` to accept every
mismatch (physics, protocol and code); accepted mismatches are logged as
warnings (logger ``cytherea.exec.batch``). A record returned for the
wrong key (``record.key_digest != key_digest(key)``) is always an error
(A-m6).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import multiprocessing
import traceback as _traceback
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Literal

from cytherea.engine.shot import (
    code_identity,
    code_version,
    protocol_hash as _protocol_hash,
    resolve_physics_config,
    run_shot,
)
from cytherea.ic.sampler import ICRejectedError
from cytherea.keys import Key, key_digest
from cytherea.store import FailureRecord, RecordSummary, ShotRecord, Store

logger = logging.getLogger(__name__)

_ON_ERROR = ("raise", "record")
_FAILURE_KINDS = frozenset({"ic_rejected", "error"})
_MAX_MESSAGE = 4000


@dataclasses.dataclass
class ShotFailure:
    """Reported in `run_batch`'s results list in place of a `ShotRecord` for
    a key that failed: ``failure_kind="ic_rejected"`` (`shot_fn` raised
    `ICRejectedError`) or ``"error"`` (another exception, with
    ``on_error="record"``). The same failure is in the store's failure log;
    ``from_store=True`` means it was read from there (the key was skipped on
    this call) rather than produced now.
    """

    key_digest: str
    reasons: list[str]
    failure_kind: str = "ic_rejected"
    error_type: str | None = None
    message: str = ""
    from_store: bool = False
    attempts: list[dict] = dataclasses.field(default_factory=list)
    frame_id: int | None = None
    level: str | None = None


class ResumeConfigMismatchError(ValueError):
    """Stored and current records disagree on physics config or code
    version (contract K8); see the module docstring."""


def _validate_inputs(
    keys: list[Key], digests: list[str], n_workers: int, on_error: str
) -> None:
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers!r}")
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for digest in digests:
        if digest in seen:
            duplicates.add(digest)
        seen.add(digest)
    if duplicates:
        raise ValueError(
            "run_batch: keys contains duplicate key(s) (same key_digest): "
            f"{sorted(duplicates)}"
        )


_HASH_FIELDS = ("physics_config_hash", "protocol_hash")


def _known_hashes(shot_fn: Callable) -> dict[str, str | None]:
    """The physics_config_hash / protocol_hash `shot_fn` will record, where
    knowable up front (None otherwise)."""
    out: dict[str, str | None] = {}
    for name in _HASH_FIELDS:
        declared = getattr(shot_fn, name, None)
        out[name] = declared if isinstance(declared, str) else None
    if isinstance(shot_fn, functools.partial) and shot_fn.func is run_shot and not shot_fn.args:
        kw = shot_fn.keywords
        if out["physics_config_hash"] is None and "backend" in kw and "physics_cfg" in kw:
            out["physics_config_hash"] = resolve_physics_config(kw["backend"], kw["physics_cfg"])[1]
        if out["protocol_hash"] is None and all(k in kw for k in ("stop", "obs", "sampler")):
            out["protocol_hash"] = _protocol_hash(
                kw["stop"], kw["obs"], kw["sampler"], kw.get("labeler")
            )
    return out


class _ResumeGuard:
    """Implements "Resume refuses a config or code change". Items are
    `ShotRecord`s, `RecordSummary`s or `FailureRecord`s."""

    def __init__(
        self, refs: dict[str, str | None], current_code: str, allow: bool, allow_code: bool = False
    ) -> None:
        self.refs = dict(refs)
        self.current_code = current_code
        self._code_id = code_identity(current_code)
        self._allow = allow
        self._allow_code = allow_code or allow
        self._deferred: list = []

    @property
    def ref_hash(self) -> str | None:  # physics hash, stamped on failures
        return self.refs["physics_config_hash"]

    def _mismatches(
        self, items: Iterable, fields: Iterable[str] = _HASH_FIELDS, check_code: bool = True,
    ) -> list[tuple[str, str]]:
        out = []
        for item in items:
            if check_code and code_identity(item.code_version) != self._code_id:
                out.append((
                    "code_version",
                    f"{item.key_digest[:12]}: code_version {item.code_version!r} "
                    f"!= current {self.current_code!r}",
                ))
            for name in fields:
                ref = self.refs[name]
                have = getattr(item, name)
                if ref is None:
                    continue
                if have is None and isinstance(item, FailureRecord):
                    continue  # failures may predate knowing the hash
                if have != ref:
                    shown = "None (not recorded)" if have is None else f"{have[:12]}..."
                    out.append(
                        (name, f"{item.key_digest[:12]}: {name} {shown} != current {ref[:12]}...")
                    )
        return out

    def _report(self, problems: list[tuple[str, str]], what: str) -> None:
        if not problems:
            return
        refused = [
            msg for field, msg in problems
            if not (self._allow or (field == "code_version" and self._allow_code))
        ]
        head = f"run_batch: {len(problems)} mismatch(es) between {what} and the current batch"
        if not refused:
            shown = "; ".join(m for _, m in problems[:5]) + ("; ..." if len(problems) > 5 else "")
            flag = "allow_config_change" if self._allow else "allow_code_change"
            logger.warning("%s (%s=True, continuing): %s", head, flag, shown)
            return
        shown = "; ".join(refused[:5]) + ("; ..." if len(refused) > 5 else "")
        raise ResumeConfigMismatchError(
            f"{head}: {shown}. Refusing to mix records made with a different "
            "physics config, shot protocol or code; use a new store, or pass "
            "allow_code_change=True (code only) / allow_config_change=True (all) "
            "to resume anyway."
        )

    def check_stored(self, items: list) -> None:
        if any(ref is None for ref in self.refs.values()):
            # Some hash not known yet: check the rest now, those later.
            self._deferred = list(items)
        self._report(self._mismatches(items), "records already in the store")

    def check_fresh(self, record: ShotRecord) -> None:
        newly_known = [n for n, ref in self.refs.items() if ref is None]
        for name in newly_known:
            self.refs[name] = getattr(record, name)
        if newly_known:
            deferred, self._deferred = self._deferred, []
            self._report(
                self._mismatches(deferred, fields=newly_known, check_code=False),
                "records already in the store",
            )
        self._report(self._mismatches([record]), "a freshly computed record")

    @property
    def unverified_stored(self) -> int:
        return len(self._deferred)


def _key_dict(key: Key) -> dict:
    if dataclasses.is_dataclass(key) and not isinstance(key, type):
        return dataclasses.asdict(key)
    return {"repr": repr(key)}


def _failure_result(f: FailureRecord, from_store: bool) -> ShotFailure:
    return ShotFailure(
        key_digest=f.key_digest,
        reasons=list(f.reasons),
        failure_kind=f.failure_kind,
        error_type=f.error_type,
        message=f.message,
        from_store=from_store,
        attempts=[dict(a) for a in f.attempts],
        frame_id=f.frame_id,
        level=f.level,
    )


class _WorkerICRejected(Exception):
    """Carries an `ICRejectedError` out of a worker process intact. (Older
    `ICRejectedError`s did not survive pickling: their ``args`` held the
    message, so unpickling rebuilt ``reasons`` from its characters. Harmless
    once `ICRejectedError` pickles itself correctly.)"""

    def __init__(self, reasons, message, attempts=None, frame_id=None, level=None) -> None:
        super().__init__(reasons, message, attempts, frame_id, level)
        self.reasons = list(reasons)
        self.message = message
        self.attempts = list(attempts or [])
        self.frame_id = frame_id
        self.level = level

    def __str__(self) -> str:
        return self.message


_WORKER_SHOT_FN: Callable | None = None


def _init_worker(shot_fn: Callable) -> None:
    global _WORKER_SHOT_FN
    _WORKER_SHOT_FN = shot_fn


def _call_in_worker(key: Key) -> ShotRecord:
    assert _WORKER_SHOT_FN is not None, "worker not initialized"
    try:
        return _WORKER_SHOT_FN(key)
    except ICRejectedError as exc:
        raise _WorkerICRejected(
            list(exc.reasons), str(exc), list(getattr(exc, "attempts", None) or []),
            getattr(exc, "frame_id", None), getattr(exc, "level", None),
        ) from None


class _Batch:
    """State shared by the serial and parallel paths of one `run_batch`."""

    def __init__(
        self, keys, digests, store, guard, on_error, on_before_append, results, return_records=True
    ):
        self.keys = keys
        self.digests = digests
        self.store = store
        self.guard = guard
        self.on_error = on_error
        self.on_before_append = on_before_append
        self.results = results
        self.return_records = return_records

    def fail(self, i: int, kind: str, exc: BaseException, reasons: list[str]) -> None:
        failure = FailureRecord(
            key_digest=self.digests[i],
            key=_key_dict(self.keys[i]),
            failure_kind=kind,  # type: ignore[arg-type]
            reasons=reasons,
            error_type="ICRejectedError" if kind == "ic_rejected" else type(exc).__name__,
            message=str(exc)[:_MAX_MESSAGE],
            code_version=self.guard.current_code,
            physics_config_hash=self.guard.ref_hash,
            protocol_hash=self.guard.refs["protocol_hash"],
            traceback=(
                None if kind == "ic_rejected" else "".join(_traceback.format_exception(exc))
            ),
            attempts=list(getattr(exc, "attempts", None) or []) if kind == "ic_rejected" else [],
            frame_id=getattr(exc, "frame_id", None) if kind == "ic_rejected" else None,
            level=getattr(exc, "level", None) if kind == "ic_rejected" else None,
        )
        self.store.append_failure(failure)
        self.results[i] = _failure_result(failure, from_store=False)

    def handle_exception(self, i: int, exc: Exception) -> None:
        """Record `exc` for key `i`, or re-raise it."""
        if isinstance(exc, (ICRejectedError, _WorkerICRejected)):
            self.fail(i, "ic_rejected", exc, list(exc.reasons))
        elif self.on_error == "record" and not isinstance(exc, BrokenProcessPool):
            self.fail(i, "error", exc, [f"error:{type(exc).__name__}"])
        else:
            raise exc

    def accept(self, i: int, record: object) -> None:
        if not isinstance(record, ShotRecord):
            raise TypeError(
                f"shot_fn returned {type(record).__name__}, expected ShotRecord"
            )
        if record.key_digest != self.digests[i]:
            raise RuntimeError(
                f"shot_fn returned a record with key_digest {record.key_digest!r} for "
                f"key {self.keys[i]!r} (key_digest {self.digests[i]!r})"
            )
        self.guard.check_fresh(record)
        if self.on_before_append is not None:
            self.on_before_append(record)
        self.store.append(record)
        self.results[i] = record if self.return_records else _summary_of(record)


def _summary_of(record: ShotRecord) -> RecordSummary:
    return RecordSummary(
        record.key_digest, record.kind, record.stop_reason, record.code_version,
        record.physics_config_hash, record.protocol_hash,
    )


def _retry_kinds(retry_failures: object) -> frozenset[str]:
    if retry_failures is True:
        return _FAILURE_KINDS
    if retry_failures is False or retry_failures is None:
        return frozenset()
    if isinstance(retry_failures, str) or not isinstance(retry_failures, Iterable):
        raise ValueError(
            f"retry_failures must be a bool or a set of failure kinds, got {retry_failures!r}"
        )
    kinds = frozenset(retry_failures)
    if not kinds <= _FAILURE_KINDS:
        raise ValueError(
            f"retry_failures: unknown failure kind(s) {sorted(kinds - _FAILURE_KINDS)}; "
            f"expected a subset of {sorted(_FAILURE_KINDS)}"
        )
    return kinds


def run_batch(
    keys: Sequence[Key],
    shot_fn: Callable[[Key], ShotRecord],
    store: Store,
    n_workers: int = 1,
    *,
    on_before_append: Callable[[ShotRecord], None] | None = None,
    allow_config_change: bool = False,
    allow_code_change: bool = False,
    retry_failures: bool | Iterable[str] = False,
    on_error: Literal["raise", "record"] = "raise",
    physics_config_hash: str | None = None,
    protocol_hash: str | None = None,
    return_records: bool = True,
) -> list[ShotRecord | RecordSummary | ShotFailure]:
    """Run `shot_fn` over every key in `keys` not already present in `store`.

    See the module docstring for the parent-only/ordered-write contract, the
    skip-if-already-recorded resume behavior (and its config/code check,
    `allow_config_change`), the failure log (`retry_failures`, `on_error`),
    the `shot_fn`-must-not-persist requirement, prompt cancellation on
    error, and the picklability requirement for `n_workers > 1`.

    `retry_failures`: True re-runs every key whose only entry is a logged
    failure; a set such as ``{"error"}`` re-runs only those failure kinds
    (p1 m-3: retry transient errors without re-running deterministic IC
    rejections). `return_records=False` returns `RecordSummary`s instead of
    `ShotRecord`s (N-I4).
    """
    keys = list(keys)
    digests = [key_digest(k) for k in keys]
    _validate_inputs(keys, digests, n_workers, on_error)
    retry = _retry_kinds(retry_failures)
    if (
        isinstance(shot_fn, functools.partial)
        and shot_fn.func is run_shot
        and shot_fn.keywords.get("store") is not None
    ):
        raise ValueError(
            "run_batch: shot_fn is partial(run_shot, ..., store=<a Store>); run_batch is the "
            "only writer of the store (ruling R33), so pass store=None to run_shot"
        )

    refs = _known_hashes(shot_fn)
    if physics_config_hash is not None:
        refs["physics_config_hash"] = physics_config_hash
    if protocol_hash is not None:
        refs["protocol_hash"] = protocol_hash
    guard = _ResumeGuard(refs, code_version(), allow_config_change, allow_code_change)

    existing = store.summaries_many(digests)
    missing = [d for d in digests if d not in existing]
    logged = {
        d: fs for d, fs in store.failures_many(missing).items() if fs[-1].failure_kind not in retry
    }
    guard.check_stored([*existing.values(), *(fs[-1] for fs in logged.values())])
    if return_records and existing:
        existing = store.get_many(list(existing))

    results: list[ShotRecord | RecordSummary | ShotFailure | None] = [None] * len(keys)
    pending: list[int] = []
    for i, digest in enumerate(digests):
        if digest in existing:
            results[i] = existing[digest]
        elif digest in logged:
            results[i] = _failure_result(logged[digest][-1], from_store=True)
        else:
            pending.append(i)

    batch = _Batch(keys, digests, store, guard, on_error, on_before_append, results, return_records)
    if pending:
        if n_workers == 1:
            for i in pending:
                _run_one(batch, shot_fn, i)
        else:
            _run_parallel(batch, shot_fn, pending, n_workers)

    if guard.unverified_stored:
        logger.warning(
            "run_batch: the physics config / protocol of %d stored record(s) "
            "could not be verified (no shot was run and shot_fn does not "
            "declare its physics_config_hash / protocol_hash)",
            guard.unverified_stored,
        )

    # Every slot was filled above: from the store (skip branch) or by the
    # pending loop.
    return results  # type: ignore[return-value]


def _run_one(batch: _Batch, shot_fn: Callable[[Key], ShotRecord], i: int) -> None:
    try:
        record = shot_fn(batch.keys[i])
    except Exception as exc:
        batch.handle_exception(i, exc)
        return
    batch.accept(i, record)


def _run_parallel(
    batch: _Batch,
    shot_fn: Callable[[Key], ShotRecord],
    pending: list[int],
    n_workers: int,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx, initializer=_init_worker, initargs=(shot_fn,)
    )
    try:
        futures = {i: executor.submit(_call_in_worker, batch.keys[i]) for i in pending}
        # Resolve in ascending (== keys) order, not completion order, so the
        # parent's appends into `store` land in keys order regardless of
        # which worker finishes first.
        for i in pending:
            try:
                record = futures[i].result()
            except Exception as exc:
                batch.handle_exception(i, exc)
                continue
            batch.accept(i, record)
    except BaseException:
        # Controller ruling R35 #2: stop promptly on an exception that is
        # not recorded, rather than waiting for every already-submitted (but
        # not yet started) shot to run first.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
