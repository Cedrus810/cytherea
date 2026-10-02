"""Append-only record store and configuration provenance hashing.

Every trajectory ("shot") and WE segment produced anywhere in cytherea is
written through :class:`Store`. It is the project's record of truth: the
basis for crash-safe resume (only missing keys get re-run, see design doc's
batch-resume section / task 7) and for reproducibility audits (design doc
section 7). Because of that, the API deliberately exposes no way to update
or delete a row -- the only write entry points are :meth:`Store.append`
(records) and :meth:`Store.append_failure` (the failure log), and appending
an already-present record ``key_digest`` raises :class:`DuplicateKeyError`
rather than overwriting anything.

Storage is plain stdlib ``sqlite3``: one row per record, with the full
record serialized as JSON in a ``payload`` column plus a few denormalized
columns (`kind`, `stop_reason`, `origin_label`) to make :meth:`Store.iter`
filtering cheap without deserializing every row. Each :class:`Store` method
opens its own short-lived connection rather than holding one open for the
object's lifetime; ``append`` is one ``BEGIN IMMEDIATE ... COMMIT``
transaction, so a record is either fully present or absent, including
under ``SIGKILL``.

Journal and durability (spec S3, fullreview A-C2)
--------------------------------------------------
The database uses the rollback journal (``PRAGMA journal_mode=DELETE``) with
``PRAGMA synchronous=FULL`` on every connection -- **not WAL**. WAL keeps its
index in a memory-mapped ``-shm`` file, which the SQLite documentation says
does not work on a network filesystem; the repository and ``runs/`` live on
NFS here. Opening a store that was left in WAL mode converts it. Connections
wait up to ``_BUSY_TIMEOUT_S`` seconds for a lock instead of sqlite's 5 s
default, so a slow concurrent reader never aborts a batch after a shot has
already been computed (A-m13).

One owning host per store file (spec S3)
-----------------------------------------
The NFS mount here uses ``local_lock=all`` (POSIX locks are per host) and a
long attribute cache (``nocto``), so SQLite's locking cannot protect a file
that two hosts touch, even one after the other. Therefore the host that
creates a store writes its hostname into the ``meta`` table, and opening the
store from any other host raises :class:`StoreOwnershipError` -- unless
``Store(path, takeover=True)`` is passed, which transfers ownership, logs a
warning (logger ``cytherea.store``) and appends a row to the store's own
``host_log`` table. A pickled ``Store`` re-runs this check when it is
unpickled, so a store object cannot be smuggled to another host either.

**Resuming on a different host:** stop every writer on the old host, then
*on the old (owning) host* run ``Store(path).backup_to(copy_path)``, which
makes a consistent copy with the SQLite backup API; move that copy to the
new host's disk and open it there with ``takeover=True``. Never copy the
database file by hand (fixreview-p1 N-I2): if a writer was killed in the
middle of a commit, the file is only consistent together with its hot
``<db>-journal``, and a bare copy is corrupt (broken indexes, uncommitted
rows). Opening the store on the owning host rolls a hot journal back, and
``backup_to`` does so before copying. Do not open the same file from two
hosts over NFS. Single-host use on NFS is fine: locks and the page cache are
coherent within one host.

Every ``append`` / ``append_failure`` re-checks, inside its transaction, that
this host still owns the file, so a ``Store`` object left open on the old
host fails (StoreOwnershipError) after another host took over -- reliably
on one host, best effort across NFS hosts (attribute caching).

Payload types and JSON
-----------------------
:class:`ShotRecord` normalizes every field to plain JSON types when it is
constructed (numpy scalars -> ``int``/``float``/``bool``, 0-d arrays -> their
scalar, tuples -> lists, ``origin_label`` -> a tuple of ints) and rejects
anything else, and non-``str`` mapping keys, with ``TypeError`` -- so a record
read back from the store compares equal to the one that was written
(fullreview A-I6), and bad values fail at construction rather than at
``append``. Non-finite floats are stored as the JSON extension tokens
``NaN``/``Infinity``/``-Infinity`` (Python's ``json`` default): they round-trip
exactly through this module, but strict JSON parsers reject them (A-m10).

Schema compatibility
---------------------
Fields added in the 2026-10-01 fix wave (``ic_meta``, ``observables_thinned``,
``warnings``, ``protocol_hash``) have defaults, so a payload written before them still decodes;
its ``observables_thinned`` is ``None`` ("unknown"), which offline replay
refuses. Stores created before the fix wave have no ``meta`` table: the
first open adopts the file for the opening host. Phase A has no production
stores, so nothing older than that is supported.

``config_hash`` computes a canonical JSON encoding (object keys sorted,
floats encoded via ``repr`` so that e.g. 0.002 and 0.0020000001 -- which
would print identically under a rounding encoder -- hash differently) and
returns its sha256 hex digest. It is used to fingerprint physics
configuration dicts (or dataclasses) into ``ShotRecord.physics_config_hash``
and similar provenance fields.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import numbers
import operator
import os
import socket
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Seconds a connection waits for a lock before raising (sqlite default: 5).
_BUSY_TIMEOUT_S = 60.0
# Rows fetched per page by Store.iter (keyset pagination, one short-lived
# connection per page, so no read lock is held while the caller works).
_ITER_PAGE = 256
# Max bound parameters per IN (...) query in get_many / failures_many.
_IN_CHUNK = 500

_RECORD_KINDS = ("shot", "segment")


def _current_hostname() -> str:
    """The host identity written into / checked against a store's meta
    table. A module-level function so tests can simulate another host."""
    return socket.gethostname()


# ---------------------------------------------------------------------------
# Plain-JSON normalization (fullreview A-I6)
# ---------------------------------------------------------------------------


def _plain(value: object, where: str) -> object:
    """Recursively convert `value` to plain JSON types, or raise TypeError."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), where)
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"{where}: mapping keys must be str, got {type(k).__name__} {k!r}"
                )
            out[str.__str__(k)] = _plain(v, f"{where}[{k!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{where}[{i}]") for i, v in enumerate(value)]
    raise TypeError(
        f"{where}: {type(value).__name__} is not JSON-representable (value: {value!r})"
    )


def _plain_mapping(value: object, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(value).__name__}")
    return _plain(value, where)  # type: ignore[return-value]


def _req_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a str, got {type(value).__name__} {value!r}")
    return str.__str__(value)


def _opt_str(value: object, where: str) -> str | None:
    return None if value is None else _req_str(value, where)


def _as_int(value: object, where: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{where} must be an integer, got bool {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"{where} must be an integer, got {type(value).__name__} {value!r}"
        ) from None


def _as_float(value: object, where: str) -> float:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{where} must be a real number, got {type(value).__name__} {value!r}")
    return float(value)


@dataclasses.dataclass
class ShotRecord:
    """One completed shot or WE segment, as it will be written to the Store.

    ``key`` and ``key_digest`` come from ``cytherea.keys``: the caller
    computes ``key_digest = key_digest(key)`` and passes both the digest and
    ``dataclasses.asdict(key)`` (or an equivalent plain dict) in here --
    ``Store`` itself does not depend on the ``keys`` module's types, only on
    plain JSON-able values, which keeps this module free of that coupling.

    Every field is normalized to plain JSON types on construction (see the
    module docstring); a value that cannot be represented raises
    ``TypeError`` here, and an unknown ``kind`` raises ``ValueError``.

    Fields added in the fix wave:

    - ``ic_meta`` (contract K2): the sampler's ``InitialState.meta``, stored
      as given (``frame_id``, ``frame_time``, ``frame_weight``,
      ``source_id``, ``topology_ref``, ``state``, ``n_redraws``, ...).
    - ``observables_thinned`` (contract K6): ``True`` when the stored
      ``observables`` series was thinned (``store_stride > 1``), so it is not
      the series the stop rule saw online; ``False`` when it is that series.
      Every record produced by the engine carries a bool. ``None`` means
      "unknown" (a record built by hand, or decoded from a payload written
      before the field existed) and offline replay refuses it.
    - ``warnings``: provenance caveats the engine attached (e.g. an
      initial state rebased to t=0, contract K1).
    - ``protocol_hash`` (ruling R39): hash of the shot protocol -- stop
      rule and its parameters, ObsSpec, IC sampler and frame pool (see
      ``cytherea.engine.shot.protocol_description``). ``None`` for records
      not produced by the engine or written before the field existed.
    """

    key_digest: str
    key: dict
    kind: Literal["shot", "segment"]
    frame_id: int | None
    origin_label: tuple[int, int] | None
    ic_validity: dict
    stop_rule_kind: str
    stop_reason: str
    event_time: float | None
    physics_config_hash: str
    backend_provenance: dict
    code_version: str
    observables: dict[str, list[float]]
    final_state_label: str | None
    weight: float = 1.0
    parent_digest: str | None = None
    ic_meta: dict = dataclasses.field(default_factory=dict)
    observables_thinned: bool | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)
    protocol_hash: str | None = None

    def __post_init__(self) -> None:
        self.key_digest = _req_str(self.key_digest, "ShotRecord.key_digest")
        self.protocol_hash = _opt_str(self.protocol_hash, "ShotRecord.protocol_hash")
        self.key = _plain_mapping(self.key, "ShotRecord.key")
        if self.kind not in _RECORD_KINDS:
            raise ValueError(
                f"ShotRecord.kind must be one of {_RECORD_KINDS}, got {self.kind!r}"
            )
        self.kind = str.__str__(self.kind)
        if self.frame_id is not None:
            self.frame_id = _as_int(self.frame_id, "ShotRecord.frame_id")
        if self.origin_label is not None:
            if isinstance(self.origin_label, (str, bytes)) or not isinstance(
                self.origin_label, Iterable
            ):
                raise TypeError(
                    "ShotRecord.origin_label must be a sequence of ints or None, "
                    f"got {self.origin_label!r}"
                )
            self.origin_label = tuple(
                _as_int(v, "ShotRecord.origin_label[...]") for v in self.origin_label
            )
        self.ic_validity = _plain_mapping(self.ic_validity, "ShotRecord.ic_validity")
        self.stop_rule_kind = _req_str(self.stop_rule_kind, "ShotRecord.stop_rule_kind")
        self.stop_reason = _req_str(self.stop_reason, "ShotRecord.stop_reason")
        if self.event_time is not None:
            self.event_time = _as_float(self.event_time, "ShotRecord.event_time")
        self.physics_config_hash = _req_str(
            self.physics_config_hash, "ShotRecord.physics_config_hash"
        )
        self.backend_provenance = _plain_mapping(
            self.backend_provenance, "ShotRecord.backend_provenance"
        )
        self.code_version = _req_str(self.code_version, "ShotRecord.code_version")
        if not isinstance(self.observables, Mapping):
            raise TypeError("ShotRecord.observables must be a mapping name -> series")
        observables: dict[str, list[float]] = {}
        for name, seq in self.observables.items():
            name = _req_str(name, "ShotRecord.observables key")
            if isinstance(seq, np.ndarray):
                seq = seq.tolist()
            if isinstance(seq, (str, bytes, Mapping)) or not isinstance(seq, Iterable):
                raise TypeError(f"ShotRecord.observables[{name!r}] must be a sequence")
            observables[name] = [
                _as_int(v, f"ShotRecord.observables[{name!r}]")
                if isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_))
                else _as_float(v, f"ShotRecord.observables[{name!r}]")
                for v in seq
            ]
        self.observables = observables
        self.final_state_label = _opt_str(self.final_state_label, "ShotRecord.final_state_label")
        self.weight = _as_float(self.weight, "ShotRecord.weight")
        self.parent_digest = _opt_str(self.parent_digest, "ShotRecord.parent_digest")
        self.ic_meta = _plain_mapping(self.ic_meta, "ShotRecord.ic_meta")
        if self.observables_thinned is not None:
            if not isinstance(self.observables_thinned, (bool, np.bool_)):
                raise TypeError(
                    "ShotRecord.observables_thinned must be a bool or None, "
                    f"got {self.observables_thinned!r}"
                )
            self.observables_thinned = bool(self.observables_thinned)
        if isinstance(self.warnings, (str, bytes)) or not isinstance(self.warnings, Iterable):
            raise TypeError("ShotRecord.warnings must be a list of str")
        self.warnings = [_req_str(w, "ShotRecord.warnings[...]") for w in self.warnings]


@dataclasses.dataclass
class FailureRecord:
    """One failed attempt at a key, as written to the store's failure log
    (fullreview A-I2). The log is append-only and separate from the records
    table: a key can have several failed attempts and later still a record,
    and ``Store.has`` / ``Store.iter`` never see failures.

    ``failure_kind`` is ``"ic_rejected"`` (every IC draw failed the gate;
    ``reasons`` lists the reasons of all attempts) or ``"error"`` (the shot
    raised; ``reasons == ["error:<ExceptionType>"]`` and ``traceback`` holds
    the formatted traceback). For IC rejections, ``attempts``, ``frame_id``
    and ``level`` ("coordinate" / "velocity") come from the
    ``ICRejectedError``.
    """

    key_digest: str
    key: dict
    failure_kind: Literal["ic_rejected", "error"]
    reasons: list[str]
    error_type: str
    message: str
    code_version: str
    physics_config_hash: str | None = None
    traceback: str | None = None
    protocol_hash: str | None = None
    attempts: list[dict] = dataclasses.field(default_factory=list)
    frame_id: int | None = None
    level: str | None = None

    def __post_init__(self) -> None:
        self.key_digest = _req_str(self.key_digest, "FailureRecord.key_digest")
        self.protocol_hash = _opt_str(self.protocol_hash, "FailureRecord.protocol_hash")
        self.attempts = _plain(list(self.attempts), "FailureRecord.attempts")  # type: ignore[assignment]
        if self.frame_id is not None:
            self.frame_id = _as_int(self.frame_id, "FailureRecord.frame_id")
        self.level = _opt_str(self.level, "FailureRecord.level")
        self.key = _plain_mapping(self.key, "FailureRecord.key")
        if self.failure_kind not in ("ic_rejected", "error"):
            raise ValueError(f"FailureRecord.failure_kind invalid: {self.failure_kind!r}")
        self.reasons = [_req_str(r, "FailureRecord.reasons[...]") for r in self.reasons]
        self.error_type = _req_str(self.error_type, "FailureRecord.error_type")
        self.message = _req_str(self.message, "FailureRecord.message")
        self.code_version = _req_str(self.code_version, "FailureRecord.code_version")
        self.physics_config_hash = _opt_str(
            self.physics_config_hash, "FailureRecord.physics_config_hash"
        )
        self.traceback = _opt_str(self.traceback, "FailureRecord.traceback")


class DuplicateKeyError(Exception):
    """Raised by :meth:`Store.append` when ``rec.key_digest`` already exists.

    The store is left unchanged: the failed insert is rolled back as part of
    the same transaction (see module docstring).
    """


class StoreOwnershipError(RuntimeError):
    """Raised when a store owned by another host is opened without
    ``takeover=True`` (spec S3; see the module docstring)."""


class StoreError(RuntimeError):
    """The database could not be put into the required configuration."""


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS records (
        key_digest   TEXT PRIMARY KEY,
        kind         TEXT NOT NULL,
        stop_reason  TEXT NOT NULL,
        origin_label TEXT,
        payload      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS failures (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        key_digest TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS failures_by_digest ON failures (key_digest, id)",
    """
    CREATE TABLE IF NOT EXISTS meta (
        name  TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS host_log (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        event     TEXT NOT NULL,
        host      TEXT NOT NULL,
        prev_host TEXT,
        at_utc    TEXT NOT NULL
    )
    """,
)

_SCHEMA_VERSION = "2"

# Filter names Store.iter() understands. Kept as an explicit allow-list so
# that a typo'd filter name fails loudly (ValueError) instead of silently
# matching everything.
_ITER_FILTERS = frozenset(
    {"kind", "stop_reason", "origin_label", "physics_config_hash", "protocol_hash"}
)
# Filters on payload fields, evaluated in SQL with json_extract (p1 m-5).
_JSON_FILTERS = ("physics_config_hash", "protocol_hash")


@dataclasses.dataclass(frozen=True)
class RecordSummary:
    """The few fields of a stored record that resume bookkeeping needs, read
    in SQL without decoding the payload (fixreview-p1 N-I4): `run_batch`
    checks them against the current batch, and returns them instead of
    full records with ``return_records=False``."""

    key_digest: str
    kind: str
    stop_reason: str
    code_version: str
    physics_config_hash: str
    protocol_hash: str | None


def _record_to_json(rec: ShotRecord) -> str:
    # dataclasses.asdict deep-copies nested containers; every field is
    # already plain JSON (ShotRecord.__post_init__). json.dumps encodes
    # floats via repr-precision double formatting, which is round-trip exact
    # in CPython, and (allow_nan=True, the default) encodes NaN/Infinity as
    # the non-standard tokens documented in the module docstring.
    return json.dumps(dataclasses.asdict(rec))


_RECORD_FIELDS = None  # filled lazily: dataclasses.fields(ShotRecord)


def _record_from_json(text: str) -> ShotRecord:
    """Decode a payload this module wrote (fixreview-p1 N-I4: trusted path).

    `append` normalises every field before it serialises (ShotRecord
    construction plus ``dataclasses.replace``), and JSON decoding yields
    exactly those plain types back -- except ``origin_label``, a list again,
    which becomes the tuple here. So the per-value re-validation of
    ``ShotRecord.__post_init__`` (about 3x the cost of ``json.loads``) is
    skipped. Fields missing from an older payload take their defaults; an
    unknown field falls back to the validating constructor, which raises."""
    global _RECORD_FIELDS
    if _RECORD_FIELDS is None:
        _RECORD_FIELDS = dataclasses.fields(ShotRecord)
    data = json.loads(text)
    if not isinstance(data, dict) or not set(data) <= {f.name for f in _RECORD_FIELDS}:
        return ShotRecord(**data)
    rec = object.__new__(ShotRecord)
    for f in _RECORD_FIELDS:
        if f.name in data:
            value = data[f.name]
        elif f.default is not dataclasses.MISSING:
            value = f.default
        elif f.default_factory is not dataclasses.MISSING:
            value = f.default_factory()
        else:
            return ShotRecord(**data)  # raises the usual TypeError
        object.__setattr__(rec, f.name, value)
    if rec.origin_label is not None:
        rec.origin_label = tuple(rec.origin_label)
    return rec


def _failure_from_json(text: str) -> FailureRecord:
    return FailureRecord(**json.loads(text))


def _origin_label_column(origin_label: tuple[int, int] | None) -> str | None:
    if origin_label is None:
        return None
    return json.dumps([_as_int(v, "origin_label") for v in origin_label])


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _chunks(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


class Store:
    """Append-only sqlite-backed record store. See module docstring.

    ``takeover=True`` lets this host take ownership of a store created on
    another host (spec S3); it is logged, and recorded in ``host_log()``.
    """

    def __init__(self, path: str | os.PathLike, *, takeover: bool = False) -> None:
        self._path = os.fspath(path)
        host = _current_hostname()
        conn = self._raw_connect()
        try:
            # Must run outside a transaction. Returns the mode now in effect;
            # a silent fallback (e.g. a read-only file) is an error (A-m2).
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            if str(mode).lower() != "delete":
                raise StoreError(
                    f"{self._path}: could not set journal_mode=DELETE (got {mode!r})"
                )
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in _SCHEMA:
                    conn.execute(stmt)
                row = conn.execute(
                    "SELECT value FROM meta WHERE name = 'owner_host'"
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO meta (name, value) VALUES ('owner_host', ?), "
                        "('created_utc', ?), ('schema_version', ?)",
                        (host, _utc_now(), _SCHEMA_VERSION),
                    )
                    conn.execute(
                        "INSERT INTO host_log (event, host, prev_host, at_utc) "
                        "VALUES ('create', ?, NULL, ?)",
                        (host, _utc_now()),
                    )
                elif row[0] != host:
                    if not takeover:
                        raise StoreOwnershipError(
                            f"store {self._path} is owned by host {row[0]!r}, not "
                            f"this host {host!r}. Opening one SQLite file from two "
                            "hosts over NFS is unsafe. To resume here, stop the "
                            f"writers on {row[0]!r}, run Store(path).backup_to(copy) "
                            "there (never copy the file by hand), move the copy to "
                            "this host, and open it with Store(copy, takeover=True)."
                        )
                    conn.execute(
                        "UPDATE meta SET value = ? WHERE name = 'owner_host'", (host,)
                    )
                    conn.execute(
                        "INSERT INTO host_log (event, host, prev_host, at_utc) "
                        "VALUES ('takeover', ?, ?, ?)",
                        (host, row[0], _utc_now()),
                    )
                    logger.warning(
                        "store %s: ownership takeover by host %r from owner host %r",
                        self._path, host, row[0],
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def __reduce__(self):
        # Unpickling re-opens the store, re-running the ownership check on
        # whatever host it is unpickled on (never with takeover).
        return (Store, (self._path,))

    def _raw_connect(self) -> sqlite3.Connection:
        # isolation_level=None => autocommit mode; we issue BEGIN/COMMIT/
        # ROLLBACK ourselves explicitly so that append() is one transaction
        # under our control (needed for the crash-safety contract: an
        # exception raised anywhere between BEGIN and COMMIT must leave no
        # trace of the row).
        conn = sqlite3.connect(self._path, isolation_level=None, timeout=_BUSY_TIMEOUT_S)
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    # -- ownership / provenance of the file itself -----------------------

    def owner_host(self) -> str:
        conn = self._raw_connect()
        try:
            row = conn.execute("SELECT value FROM meta WHERE name = 'owner_host'").fetchone()
        finally:
            conn.close()
        return row[0]

    def host_log(self) -> list[dict]:
        conn = self._raw_connect()
        try:
            rows = conn.execute(
                "SELECT event, host, prev_host, at_utc FROM host_log ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [dict(event=e, host=h, prev_host=p, at_utc=a) for e, h, p, a in rows]

    def _require_owner(self, conn: sqlite3.Connection) -> None:
        """Inside a write transaction: this host must still own the file
        (p1 m-4; see the module docstring)."""
        row = conn.execute("SELECT value FROM meta WHERE name = 'owner_host'").fetchone()
        host = _current_hostname()
        if row is not None and row[0] != host:
            raise StoreOwnershipError(
                f"store {self._path} is now owned by host {row[0]!r} (taken over "
                f"after this Store was opened on {host!r}); refusing to write"
            )

    def backup_to(self, dst: str | os.PathLike) -> None:
        """Write a consistent copy of the store to `dst` with the SQLite
        backup API -- the only supported way to move a store to another host
        (spec S3, fixreview-p1 N-I2; module docstring). Run it on the owning
        host after its writers have stopped; a hot journal left by a crashed
        writer is rolled back first, so the copy holds exactly the committed
        records. `dst` must not exist. The copy keeps this host as owner:
        open it on the new host with ``Store(dst, takeover=True)``."""
        dst = os.fspath(dst)
        if os.path.exists(dst):
            raise FileExistsError(f"backup destination {dst} already exists")
        src = self._raw_connect()
        try:
            src.execute("BEGIN")  # a read transaction: rolls back a hot journal
            try:
                row = src.execute(
                    "SELECT value FROM meta WHERE name = 'owner_host'"
                ).fetchone()
            finally:
                src.execute("COMMIT")
            host = _current_hostname()
            if row is not None and row[0] != host:
                raise StoreOwnershipError(
                    f"store {self._path} is owned by host {row[0]!r}; back it up "
                    f"there, not on {host!r}"
                )
            out = sqlite3.connect(dst, isolation_level=None)
            try:
                src.backup(out)
                mode = out.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
                check = out.execute("PRAGMA integrity_check").fetchall()
            finally:
                out.close()
        finally:
            src.close()
        if str(mode).lower() != "delete" or check != [("ok",)]:
            raise StoreError(f"backup {dst} failed verification: journal_mode={mode!r}, {check}")

    # -- records -----------------------------------------------------------

    def append(self, rec: ShotRecord) -> None:
        # Re-run the field normalization, in case the record was mutated
        # after construction.
        rec = dataclasses.replace(rec)
        payload = _record_to_json(rec)
        origin_label_column = _origin_label_column(rec.origin_label)
        conn = self._raw_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._require_owner(conn)
                conn.execute(
                    "INSERT INTO records "
                    "(key_digest, kind, stop_reason, origin_label, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rec.key_digest, rec.kind, rec.stop_reason,
                     origin_label_column, payload),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                # Only the primary-key collision is a duplicate (A-m1).
                if "UNIQUE" in str(exc) and "key_digest" in str(exc):
                    raise DuplicateKeyError(rec.key_digest) from exc
                raise
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def get(self, digest: str) -> ShotRecord:
        conn = self._raw_connect()
        try:
            row = conn.execute(
                "SELECT payload FROM records WHERE key_digest = ?", (digest,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(digest)
        return _record_from_json(row[0])

    def get_many(self, digests: Iterable[str]) -> dict[str, ShotRecord]:
        """``{digest: record}`` for those of `digests` that are present,
        read over one connection (A-m4: resume no longer opens two
        connections per key)."""
        wanted = list(dict.fromkeys(digests))
        out: dict[str, ShotRecord] = {}
        conn = self._raw_connect()
        try:
            for chunk in _chunks(wanted, _IN_CHUNK):
                marks = ",".join("?" * len(chunk))
                for digest, payload in conn.execute(
                    f"SELECT key_digest, payload FROM records WHERE key_digest IN ({marks})",
                    chunk,
                ):
                    out[digest] = _record_from_json(payload)
        finally:
            conn.close()
        return out

    def summaries_many(self, digests: Iterable[str]) -> dict[str, RecordSummary]:
        """``{digest: RecordSummary}`` for those of `digests` that are
        present, extracted in SQL (``json_extract``) without decoding any
        payload (fixreview-p1 N-I4: ~50 us per record instead of ~10 ms)."""
        wanted = list(dict.fromkeys(digests))
        out: dict[str, RecordSummary] = {}
        conn = self._raw_connect()
        try:
            for chunk in _chunks(wanted, _IN_CHUNK):
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(
                    "SELECT key_digest, kind, stop_reason, "
                    "json_extract(payload, '$.code_version'), "
                    "json_extract(payload, '$.physics_config_hash'), "
                    "json_extract(payload, '$.protocol_hash') "
                    f"FROM records WHERE key_digest IN ({marks})",
                    chunk,
                ):
                    out[row[0]] = RecordSummary(*row)
        finally:
            conn.close()
        return out

    def distinct_values(
        self,
        paths: Iterable[str],
        *,
        kind: str | None = None,
        key: Mapping[str, object] | None = None,
    ) -> set[tuple]:
        """Distinct tuples of the payload fields at `paths` (dotted JSON
        paths, e.g. ``"physics_config_hash"`` or ``"ic_meta.we_protocol_hash"``)
        over the records of `kind` whose ``key`` fields equal `key`, computed
        in SQL without decoding payloads (used by `run_we`'s continuation
        guard)."""
        paths = list(paths)
        for p in paths + list(key or {}):
            if not all(part.isidentifier() for part in p.split(".")):
                raise ValueError(f"invalid field path {p!r}")
        cols = ", ".join(f"json_extract(payload, '$.{p}')" for p in paths)
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        for name, value in (key or {}).items():
            clauses.append(f"json_extract(payload, '$.key.{name}') = ?")
            params.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._raw_connect()
        try:
            rows = conn.execute(f"SELECT DISTINCT {cols} FROM records{where}", params).fetchall()
        finally:
            conn.close()
        return {tuple(r) for r in rows}

    def has(self, digest: str) -> bool:
        conn = self._raw_connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM records WHERE key_digest = ?", (digest,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def iter(self, **filters: object) -> Iterator[ShotRecord]:
        """Yield matching records in ``key_digest`` order (A-m3: independent
        of execution and resume order), streamed page by page (A-m4)."""
        unknown = set(filters) - _ITER_FILTERS
        if unknown:
            raise ValueError(
                f"unknown filter(s) for Store.iter: {sorted(unknown)}; "
                f"supported: {sorted(_ITER_FILTERS)}"
            )

        clauses: list[str] = []
        params: list[object] = []
        if "kind" in filters:
            clauses.append("kind = ?")
            params.append(filters["kind"])
        if "stop_reason" in filters:
            clauses.append("stop_reason = ?")
            params.append(filters["stop_reason"])
        if "origin_label" in filters:
            column = _origin_label_column(filters["origin_label"])  # type: ignore[arg-type]
            if column is None:
                clauses.append("origin_label IS NULL")
            else:
                clauses.append("origin_label = ?")
                params.append(column)
        for name in _JSON_FILTERS:
            if name in filters:
                if filters[name] is None:
                    clauses.append(f"json_extract(payload, '$.{name}') IS NULL")
                else:
                    clauses.append(f"json_extract(payload, '$.{name}') = ?")
                    params.append(_req_str(filters[name], f"Store.iter({name}=...)"))
        clauses.append("key_digest > ?")
        sql = (
            "SELECT key_digest, payload FROM records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY key_digest LIMIT ?"
        )
        return self._iter_pages(sql, params)

    def _iter_pages(self, sql: str, params: list[object]) -> Iterator[ShotRecord]:
        last = ""
        page = _ITER_PAGE
        while True:
            conn = self._raw_connect()
            try:
                rows = conn.execute(sql, [*params, last, page]).fetchall()
            finally:
                conn.close()
            for _digest, payload in rows:
                yield _record_from_json(payload)
            if len(rows) < page:
                return
            last = rows[-1][0]

    # -- failure log (A-I2) ---------------------------------------------

    def append_failure(self, failure: FailureRecord) -> None:
        failure = dataclasses.replace(failure)
        payload = json.dumps(dataclasses.asdict(failure))
        conn = self._raw_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._require_owner(conn)
                conn.execute(
                    "INSERT INTO failures (key_digest, payload) VALUES (?, ?)",
                    (failure.key_digest, payload),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def failures(self, digest: str | None = None) -> list[FailureRecord]:
        """Every logged failure (for `digest`, or all), oldest first."""
        conn = self._raw_connect()
        try:
            if digest is None:
                rows = conn.execute("SELECT payload FROM failures ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM failures WHERE key_digest = ? ORDER BY id",
                    (digest,),
                ).fetchall()
        finally:
            conn.close()
        return [_failure_from_json(p) for (p,) in rows]

    def failures_many(self, digests: Iterable[str]) -> dict[str, list[FailureRecord]]:
        """``{digest: [failures, oldest first]}`` for those of `digests`
        with at least one logged failure, over one connection."""
        wanted = list(dict.fromkeys(digests))
        out: dict[str, list[FailureRecord]] = {}
        conn = self._raw_connect()
        try:
            for chunk in _chunks(wanted, _IN_CHUNK):
                marks = ",".join("?" * len(chunk))
                for digest, payload in conn.execute(
                    f"SELECT key_digest, payload FROM failures WHERE key_digest IN ({marks}) "
                    "ORDER BY id",
                    chunk,
                ):
                    out.setdefault(digest, []).append(_failure_from_json(payload))
        finally:
            conn.close()
        return out


def _canonical_json(obj: object) -> str:
    """Recursively render ``obj`` as canonical JSON text for hashing.

    - Dict keys are sorted (recursively) so that key order never affects the
      hash. Keys must be ``str``: ``{1: "a"}`` and ``{"1": "a"}`` used to
      collide, so a non-str key now raises ``TypeError`` (fullreview A-I5).
    - Numbers are canonicalized by *value category*, not Python type
      (A-I5): any integral (``int``, numpy integer; not ``bool``) renders as
      ``str(int(x))``; any other real (``float``, ``np.float64``,
      ``np.float32``, ...) as ``repr(float(x))`` -- the shortest decimal
      string that round-trips to the exact same double, which is what makes
      0.002 and 0.0020000001 hash differently. ``bool``/``np.bool_`` render
      as ``true``/``false``. So ``{"dt": np.float64(0.002)}`` and
      ``{"dt": 0.002}`` hash equal, while ``3`` and ``3.0`` do not.
    - numpy arrays render as their ``tolist()``; dataclass instances are
      expanded via ``dataclasses.asdict``.
    - Anything else (sets, arbitrary objects, ...) raises ``TypeError``
      rather than being silently stringified.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)

    if obj is None:
        return "null"
    if isinstance(obj, (bool, np.bool_)):  # must precede the integral check
        return "true" if obj else "false"
    if isinstance(obj, numbers.Integral):
        return str(int(obj))
    if isinstance(obj, numbers.Real):
        return repr(float(obj))
    if isinstance(obj, str):
        return json.dumps(str.__str__(obj))
    if isinstance(obj, np.ndarray):
        return _canonical_json(obj.tolist())
    if isinstance(obj, Mapping):
        for k in obj:
            if not isinstance(k, str):
                raise TypeError(
                    f"config_hash: mapping keys must be str, got {type(k).__name__} {k!r}"
                )
        items = sorted((str.__str__(k), _canonical_json(v)) for k, v in obj.items())
        body = ",".join(f"{json.dumps(k)}:{v}" for k, v in items)
        return "{" + body + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonical_json(v) for v in obj) + "]"

    raise TypeError(
        f"config_hash: unsupported type {type(obj).__name__!r} "
        f"(value: {obj!r})"
    )


def config_hash(obj: Mapping) -> str:
    """Return the sha256 hex digest of ``obj``'s canonical JSON encoding.

    See :func:`_canonical_json` for exactly what "canonical" means here.
    """
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
