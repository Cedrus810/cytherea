"""Tests for cytherea.store: append-only shot record store (Task 2).

Every trajectory shot and WE segment is written through `Store`. It is the
project's record of truth and the basis of crash-safe resume (design doc
section on batch resume, task 7). These tests pin down the exact
reproducibility / append-only contract from the brief's test table
(task-2-brief.md, section "必测用例").
"""

from __future__ import annotations

import dataclasses
import math
import sqlite3

import pytest

from cytherea.store import DuplicateKeyError, ShotRecord, Store, config_hash


def _make_record(**overrides) -> ShotRecord:
    defaults = dict(
        key_digest="a" * 64,
        key={"global_seed": 1, "frame_id": 2, "shot_id": 3, "stage": "ic"},
        kind="shot",
        frame_id=2,
        origin_label=(2, 5),
        ic_validity={"ok": True, "reasons": [], "n_redraws": 0},
        stop_rule_kind="boundary",
        stop_reason="product",
        event_time=1.0 / 3.0,
        physics_config_hash="deadbeef",
        backend_provenance={"platform": "CUDA", "precision": "mixed"},
        code_version="0.1.0",
        observables={"t": [0.0, 0.5, 1.0], "d_com": [0.1 + 0.2, 1e308, -1e-308]},
        final_state_label="A",
        weight=1.0,
        parent_digest=None,
    )
    defaults.update(overrides)
    return ShotRecord(**defaults)


# --- 2.1: write one record, read it back, all fields equal, floats bitwise

def test_append_then_get_roundtrips_all_fields_bitwise(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record()
    store.append(rec)

    got = store.get(rec.key_digest)

    assert got == rec
    # explicit bitwise checks for the floats most likely to be mangled by a
    # lossy encoding (repeating fractions, denormal-adjacent magnitudes).
    assert got.event_time == rec.event_time
    assert got.observables["d_com"] == rec.observables["d_com"]
    assert got.observables["d_com"][0] == 0.1 + 0.2  # not "rounded" to 0.3
    # origin_label must round-trip as a tuple, not a list
    assert got.origin_label == (2, 5)
    assert isinstance(got.origin_label, tuple)


def test_append_then_get_roundtrips_nan_and_inf_faithfully(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record(
        key_digest="b" * 64,
        event_time=float("nan"),
        observables={"t": [0.0, 1.0], "x": [float("inf"), float("-inf")]},
    )
    store.append(rec)

    got = store.get(rec.key_digest)

    assert math.isnan(got.event_time)
    assert got.observables["x"] == [float("inf"), float("-inf")]


# --- 2.2: same digest written twice raises, store still has exactly one row

def test_duplicate_key_digest_raises_and_store_unchanged(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record()
    store.append(rec)

    with pytest.raises(DuplicateKeyError):
        store.append(_make_record(stop_reason="different_but_same_digest"))

    all_records = list(store.iter())
    assert len(all_records) == 1
    assert all_records[0].stop_reason == "product"  # original, not overwritten


# --- 2.3: Store exposes no update/delete API at all

def test_store_has_no_update_or_delete_methods():
    for forbidden in ("update", "delete", "remove", "overwrite", "set"):
        assert not hasattr(Store, forbidden), (
            f"Store must not expose a {forbidden!r} method (append-only contract)"
        )


def test_store_append_of_existing_digest_cannot_overwrite_via_public_api(tmp_path):
    # Re-stating 2.3 concretely: the only write entry point is `append`, and
    # `append` on an existing digest is required (2.2) to raise rather than
    # silently overwrite. There is no other method to reach for.
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record()
    store.append(rec)
    with pytest.raises(DuplicateKeyError):
        store.append(_make_record(final_state_label="B"))
    assert store.get(rec.key_digest).final_state_label == "A"


# --- 2.4: config_hash ignores dict key order

def test_config_hash_ignores_key_order():
    a = {"dt": 0.002, "gamma": 0.1, "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "gamma": 0.1, "dt": 0.002}
    assert config_hash(a) == config_hash(b)


# --- 2.5: config_hash distinguishes near-equal floats

def test_config_hash_distinguishes_near_equal_floats():
    a = {"dt": 0.002}
    b = {"dt": 0.0020000001}
    assert config_hash(a) != config_hash(b)


def test_config_hash_supports_dataclasses_and_nested_containers():
    @dataclasses.dataclass
    class Cfg:
        dt: float
        seeds: tuple[int, ...]

    c1 = Cfg(dt=0.002, seeds=(1, 2, 3))
    c2 = {"dt": 0.002, "seeds": [1, 2, 3]}
    assert config_hash(c1) == config_hash(c2)


def test_config_hash_rejects_unsupported_types():
    class Weird:
        pass

    with pytest.raises(TypeError):
        config_hash({"bad": Weird()})

    with pytest.raises(TypeError):
        config_hash({"bad": {1, 2, 3}})  # a set is not JSON-representable


# --- 2.6: exception mid-transaction leaves no partial row

class _FlakyConnection:
    """Wraps a real sqlite3.Connection, raising once on the COMMIT call.

    sqlite3.Connection is a C-level immutable type: its methods cannot be
    monkeypatched directly (attempting ``monkeypatch.setattr(sqlite3.
    Connection, "execute", ...)`` raises TypeError). Instead we swap out
    Store's connection factory for one that hands back this thin wrapper,
    which forwards everything to a real connection except that the *first*
    "COMMIT" is turned into a simulated crash -- standing in for a process
    being killed after the row logically exists inside the open transaction
    but before the commit that would make it durable.
    """

    def __init__(self, real_conn, state):
        self._real = real_conn
        self._state = state

    def execute(self, sql, parameters=()):
        if self._state["armed"] and sql.strip() == "COMMIT":
            self._state["armed"] = False
            raise RuntimeError("simulated crash right before commit lands")
        return self._real.execute(sql, parameters)

    def close(self):
        self._real.close()


def test_crash_mid_transaction_leaves_no_partial_row(tmp_path, monkeypatch):
    db_path = tmp_path / "store.sqlite"
    store = Store(db_path)
    rec = _make_record()

    state = {"armed": True}

    def flaky_raw_connect(self):
        real_conn = sqlite3.connect(str(db_path), isolation_level=None)
        return _FlakyConnection(real_conn, state)

    monkeypatch.setattr(Store, "_raw_connect", flaky_raw_connect)

    with pytest.raises(RuntimeError):
        store.append(rec)

    # verify now, while still patched, that the row genuinely never landed
    # (not even half-written) on this same store/connection factory.
    assert store.has(rec.key_digest) is False
    assert list(store.iter()) == []

    monkeypatch.undo()

    # re-open a completely fresh Store/connection against the same file to
    # rule out an in-memory illusion (e.g. an uncommitted-but-visible row on
    # the same connection): a real crash would leave the file on disk clean.
    fresh = Store(db_path)
    assert fresh.has(rec.key_digest) is False
    assert list(fresh.iter()) == []

    # and a subsequent, unpatched append for the same key succeeds normally
    fresh.append(rec)
    assert fresh.has(rec.key_digest) is True


# --- iter() filtering: kind, stop_reason, origin_label; unknown -> ValueError

def test_iter_filters_by_kind_stop_reason_and_origin_label(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    r1 = _make_record(key_digest="1" * 64, kind="shot", stop_reason="product",
                       origin_label=(0, 0))
    r2 = _make_record(key_digest="2" * 64, kind="segment", stop_reason="timeout",
                       origin_label=(1, 2))
    r3 = _make_record(key_digest="3" * 64, kind="shot", stop_reason="timeout",
                       origin_label=None)
    for r in (r1, r2, r3):
        store.append(r)

    assert {r.key_digest for r in store.iter(kind="shot")} == {r1.key_digest, r3.key_digest}
    assert {r.key_digest for r in store.iter(stop_reason="timeout")} == {r2.key_digest, r3.key_digest}
    assert [r.key_digest for r in store.iter(origin_label=(1, 2))] == [r2.key_digest]
    assert [r.key_digest for r in store.iter(origin_label=None)] == [r3.key_digest]
    assert {r.key_digest for r in store.iter(kind="shot", stop_reason="timeout")} == {r3.key_digest}
    assert {r.key_digest for r in store.iter()} == {r1.key_digest, r2.key_digest, r3.key_digest}


def test_iter_rejects_unknown_filter_names(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    with pytest.raises(ValueError):
        list(store.iter(bogus="x"))


# --- misc: has() and get() on a missing digest

def test_has_is_false_for_missing_digest(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    assert store.has("f" * 64) is False


def test_get_missing_digest_raises(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    with pytest.raises(KeyError):
        store.get("f" * 64)


# ===========================================================================
# Fix wave P1 (fullreview A-C2 via spec S3, A-I5, A-I6, A-m1/m3/m4/m12,
# contracts K2/K6, and the persisted failure log for A-I2).
# ===========================================================================

import json
import logging
import multiprocessing
import os
import pickle
import signal
import time

import numpy as np

import cytherea.store as store_mod
from cytherea.store import FailureRecord, StoreOwnershipError


def _pragma(path, name):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


# --- S3: rollback journal (not WAL), synchronous=FULL ---------------------

def test_store_uses_delete_journal_not_wal(tmp_path):
    path = tmp_path / "store.sqlite"
    store = Store(path)
    store.append(_make_record())
    assert _pragma(path, "journal_mode") == "delete"
    assert not os.path.exists(str(path) + "-wal")
    assert not os.path.exists(str(path) + "-shm")


def test_store_connections_use_synchronous_full(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    conn = store._raw_connect()
    try:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    finally:
        conn.close()


def test_opening_a_wal_store_converts_it_to_delete_journal(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    assert _pragma(path, "journal_mode") == "wal"
    Store(path)
    assert _pragma(path, "journal_mode") == "delete"


# --- S3: one owning host per store file -----------------------------------

def test_store_records_its_creating_host(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    store = Store(tmp_path / "store.sqlite")
    assert store.owner_host() == "hostA"
    # reopening on the same host is fine and does not change the owner
    assert Store(tmp_path / "store.sqlite").owner_host() == "hostA"


def test_store_refuses_to_open_from_another_host(tmp_path, monkeypatch):
    path = tmp_path / "store.sqlite"
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    Store(path).append(_make_record())
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    with pytest.raises(StoreOwnershipError, match="hostA"):
        Store(path)


def test_store_takeover_from_another_host_is_explicit_and_logged(tmp_path, monkeypatch, caplog):
    path = tmp_path / "store.sqlite"
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    rec = _make_record()
    Store(path).append(rec)
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    with caplog.at_level(logging.WARNING, logger="cytherea.store"):
        taken = Store(path, takeover=True)
    assert taken.owner_host() == "hostB"
    assert any("takeover" in r.getMessage() and "hostA" in r.getMessage() for r in caplog.records)
    events = taken.host_log()
    assert [e["event"] for e in events] == ["create", "takeover"]
    assert events[1]["host"] == "hostB" and events[1]["prev_host"] == "hostA"
    assert taken.get(rec.key_digest) == rec  # data untouched
    # the old host is now the foreign one
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    with pytest.raises(StoreOwnershipError):
        Store(path)


def test_unpickled_store_rechecks_ownership_on_the_new_host(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    blob = pickle.dumps(Store(tmp_path / "store.sqlite"))
    assert pickle.loads(blob).owner_host() == "hostA"
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    with pytest.raises(StoreOwnershipError):
        pickle.loads(blob)


# --- A-m1: only a real duplicate is reported as DuplicateKeyError ----------

def test_invalid_record_fields_fail_at_construction_not_as_duplicate():
    with pytest.raises(TypeError):
        _make_record(stop_reason=None)
    with pytest.raises(ValueError):
        _make_record(kind="bogus")


# --- A-I6: payload normalization, so a stored record == the fresh one -----

def test_numpy_and_tuple_payload_values_normalize_and_roundtrip(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record(
        frame_id=np.int64(3),
        origin_label=(np.int64(1), 2),
        event_time=np.float32(0.5),
        weight=np.float64(0.25),
        observables={"t": [0.0, np.float64(0.1)], "x": [np.float32(1.5), np.array(2.5)]},
        backend_provenance={"box": (1.0, 2.0), "n": np.int64(7), "flag": np.bool_(True)},
        ic_validity={"ok": True, "reasons": [], "n_redraws": np.int64(0),
                     "checks": {"energy_window": np.float64(-3.0)}},
        ic_meta={"frame_weight": np.float64(0.5), "state": None, "frame_id": np.int64(3)},
    )
    assert type(rec.frame_id) is int
    assert rec.origin_label == (1, 2) and all(type(v) is int for v in rec.origin_label)
    assert type(rec.event_time) is float and type(rec.weight) is float
    assert rec.observables["x"] == [1.5, 2.5]
    assert all(type(v) is float for v in rec.observables["x"])
    assert rec.backend_provenance == {"box": [1.0, 2.0], "n": 7, "flag": True}
    store.append(rec)
    got = store.get(rec.key_digest)
    assert got == rec


def test_non_str_mapping_keys_are_rejected_at_construction():
    with pytest.raises(TypeError):
        _make_record(ic_validity={"ok": True, "reasons": [], "n_redraws": 0, "checks": {0: 1.0}})


def test_unserializable_payload_values_are_rejected_at_construction():
    with pytest.raises(TypeError):
        _make_record(backend_provenance={"obj": object()})


# --- A-I5: config_hash is type-stable --------------------------------------

def test_config_hash_is_stable_across_numpy_scalar_types():
    assert config_hash({"dt": 0.002}) == config_hash({"dt": np.float64(0.002)})
    assert config_hash({"n": 3}) == config_hash({"n": np.int64(3)})
    assert config_hash({"on": True}) == config_hash({"on": np.bool_(True)})
    assert config_hash({"n": 3}) != config_hash({"n": 3.0})
    assert config_hash({"on": True}) != config_hash({"on": 1})


def test_config_hash_rejects_non_str_mapping_keys():
    # {1: "a"} and {"1": "a"} used to collide silently.
    with pytest.raises(TypeError):
        config_hash({1: "a"})


# --- K2 / K6 / K8 record fields; records without them stay readable -------

def test_new_record_fields_roundtrip(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    rec = _make_record(
        ic_meta={"frame_id": 2, "frame_time": 50.0, "frame_weight": 0.5,
                 "source_id": "s0", "topology_ref": "top", "state": 1, "n_redraws": 0},
        observables_thinned=True,
        warnings=["something"],
    )
    store.append(rec)
    got = store.get(rec.key_digest)
    assert got == rec
    assert got.ic_meta["frame_time"] == 50.0
    assert got.observables_thinned is True


def test_record_written_without_new_fields_reads_back_with_unknown_defaults():
    # A payload in the pre-fix schema (no ic_meta / observables_thinned /
    # warnings) still decodes; the thinning flag is "unknown" (None), never
    # a made-up False.
    payload = json.loads(store_mod._record_to_json(_make_record()))
    for name in ("ic_meta", "observables_thinned", "warnings"):
        payload.pop(name)
    got = store_mod._record_from_json(json.dumps(payload))
    assert got.ic_meta == {}
    assert got.observables_thinned is None
    assert got.warnings == []


# --- A-m3/m4: iteration is order-independent and streamed -----------------

def test_iter_orders_by_key_digest_not_append_order(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    digests = ["c" * 64, "a" * 64, "b" * 64]
    for d in digests:
        store.append(_make_record(key_digest=d))
    assert [r.key_digest for r in store.iter()] == sorted(digests)


def test_iter_streams_in_pages_and_get_many(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_ITER_PAGE", 7)
    store = Store(tmp_path / "store.sqlite")
    digests = [f"{i:064x}" for i in range(30)]
    for d in reversed(digests):
        store.append(_make_record(key_digest=d, stop_reason="timeout" if int(d, 16) % 3 else "A"))
    assert [r.key_digest for r in store.iter()] == digests
    assert [r.key_digest for r in store.iter(stop_reason="A")] == [d for d in digests if int(d, 16) % 3 == 0]
    got = store.get_many(digests[:5] + ["f" * 64])
    assert set(got) == set(digests[:5])
    assert all(got[d].key_digest == d for d in got)


# --- A-I2: persisted, append-only failure log ------------------------------

def _make_failure(**overrides) -> FailureRecord:
    defaults = dict(
        key_digest="a" * 64,
        key={"global_seed": 1, "frame_id": 2, "shot_id": 3, "stage": "ic"},
        failure_kind="ic_rejected",
        reasons=["energy_window", "energy_window"],
        error_type="ICRejectedError",
        message="IC rejected after 2 failed attempt(s)",
        code_version="0.1.0+src.abc",
        physics_config_hash=None,
        traceback=None,
    )
    defaults.update(overrides)
    return FailureRecord(**defaults)


def test_failures_are_appended_and_read_back_in_attempt_order(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    f1 = _make_failure()
    f2 = _make_failure(failure_kind="error", reasons=["error:RuntimeError"], error_type="RuntimeError",
                       message="boom", traceback="Traceback ...")
    store.append_failure(f1)
    store.append_failure(f2)
    store.append_failure(_make_failure(key_digest="b" * 64))
    assert store.failures("a" * 64) == [f1, f2]
    assert len(store.failures()) == 3
    assert set(store.failures_many(["a" * 64, "c" * 64])) == {"a" * 64}
    # failures never count as records
    assert store.has("a" * 64) is False
    assert list(store.iter()) == []


# --- A-m12: a hard kill mid-append never leaves a torn database -----------

def _append_forever(path: str) -> None:
    store = Store(path)
    i = 0
    while True:
        store.append(_make_record(key_digest=f"{i:064x}",
                                  observables={"t": [0.0] * 200, "x": [float(i)] * 200}))
        i += 1


def test_sigkill_during_appends_leaves_a_consistent_store(tmp_path):
    path = str(tmp_path / "store.sqlite")
    Store(path)
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_append_forever, args=(path,))
    proc.start()
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and sum(1 for _ in Store(path).iter()) < 20:
        time.sleep(0.01)
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(timeout=10.0)
    assert proc.exitcode == -signal.SIGKILL
    assert _pragma(path, "integrity_check") == "ok"
    records = list(Store(path).iter())
    assert len(records) >= 20
    for r in records:
        assert r.observables["x"] == [float(int(r.key_digest, 16))] * 200


# ---------------------------------------------------------------------------
# Fix wave 2, package L3 (fixreview-p1 N-I2, N-I4, m-4, m-5)
# ---------------------------------------------------------------------------

import shutil

from cytherea.store import RecordSummary


def _filled_store(path, n=300):
    store = Store(path)
    for i in range(n):
        store.append(_make_record(key_digest=f"{i:064x}", observables={"t": [0.0] * 50, "x": [float(i)] * 50}))
    return store


def _kill_mid_transaction(path: str) -> None:
    # A writer killed after its transaction spilled rewritten pages into the
    # database file: only the hot journal can restore the committed state.
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA cache_size=1")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE records SET payload = payload || ' '")
    for i in range(3):
        conn.execute(
            "INSERT INTO records (key_digest, kind, stop_reason, origin_label, payload) "
            "VALUES (?, 'shot', 'B', NULL, ?)", (f"b{i:063x}", "{}" * 5000),
        )
    os.kill(os.getpid(), signal.SIGKILL)


def _integrity(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA integrity_check").fetchall(), conn.execute(
            "SELECT count(*) FROM records").fetchone()[0]
    finally:
        conn.close()


def test_n_i2_backup_after_a_writer_was_killed_mid_commit_is_consistent(tmp_path, monkeypatch):
    """p1-N-I2: copying only the database file after a crash mid-commit gives
    a corrupt store (the hot journal is left behind); Store.backup_to (the
    SQLite backup API, on the owning host) must give a consistent copy with
    exactly the committed rows, which the new host opens with takeover."""
    path = str(tmp_path / "s.sqlite")
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    _filled_store(path)
    p = multiprocessing.get_context("spawn").Process(target=_kill_mid_transaction, args=(path,))
    p.start()
    p.join(60)
    assert p.exitcode == -signal.SIGKILL
    assert os.path.exists(path + "-journal")  # a hot journal
    raw = str(tmp_path / "raw_copy.sqlite")
    shutil.copyfile(path, raw)  # the old, documented procedure
    assert _integrity(raw)[0] != [("ok",)]  # (the failure the fix is about)

    dst = tmp_path / "moved.sqlite"
    Store(path).backup_to(dst)
    assert _integrity(dst) == ([("ok",)], 300)
    assert not os.path.exists(str(dst) + "-journal")
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    moved = Store(dst, takeover=True)
    assert moved.owner_host() == "hostB" and len(list(moved.iter())) == 300


def test_n_i2_backup_refuses_an_existing_destination_and_a_foreign_host(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    store = _filled_store(tmp_path / "s.sqlite", n=3)
    (tmp_path / "exists.sqlite").write_bytes(b"")
    with pytest.raises(FileExistsError):
        store.backup_to(tmp_path / "exists.sqlite")
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    with pytest.raises(StoreOwnershipError, match="hostA"):
        store.backup_to(tmp_path / "other.sqlite")


def test_n_i4_summaries_come_from_sql_without_decoding_payloads(tmp_path, monkeypatch):
    store = Store(tmp_path / "s.sqlite")
    recs = [_make_record(key_digest=f"{i:064x}", code_version=f"v{i}", physics_config_hash=f"p{i}",
                         protocol_hash=None if i == 0 else f"q{i}", stop_reason="A" if i % 2 else "B")
            for i in range(4)]
    for r in recs:
        store.append(r)
    monkeypatch.setattr(store_mod, "_record_from_json", lambda text: pytest.fail("decoded a payload"))
    got = store.summaries_many([r.key_digest for r in recs] + ["f" * 64])
    assert got == {
        r.key_digest: RecordSummary(r.key_digest, r.kind, r.stop_reason, r.code_version,
                                    r.physics_config_hash, r.protocol_hash)
        for r in recs
    }


@pytest.mark.parametrize("overrides", [
    {},
    {"origin_label": None, "frame_id": None, "event_time": None, "final_state_label": None},
    {"observables": {"t": [0.0, 1.0], "x": [float("nan"), float("inf")], "n": [1, 2]}},
    {"ic_meta": {"frame_weight": 0.25, "nested": {"a": [1, 2.5]}}, "observables_thinned": True,
     "warnings": ["w"], "protocol_hash": "q", "weight": 0.5, "parent_digest": "b" * 64},
])
def test_n_i4_fast_decode_equals_validated_construction(tmp_path, overrides):
    """The trusted decode path (no re-normalisation of what the store wrote)
    must give exactly the record a validating ShotRecord(**payload) gives."""
    store = Store(tmp_path / "s.sqlite")
    rec = _make_record(**overrides)
    store.append(rec)
    conn = sqlite3.connect(tmp_path / "s.sqlite")
    payload = conn.execute("SELECT payload FROM records").fetchone()[0]
    conn.close()
    fast, slow = store.get(rec.key_digest), ShotRecord(**json.loads(payload))
    assert type(fast) is ShotRecord
    assert json.dumps(dataclasses.asdict(fast)) == json.dumps(dataclasses.asdict(slow))
    assert type(fast.origin_label) is type(slow.origin_label)


def test_n_i4_fast_decode_of_an_old_payload_uses_defaults(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    rec = _make_record()
    store.append(rec)
    old = {k: v for k, v in json.loads(store_mod._record_to_json(rec)).items()
           if k not in ("ic_meta", "observables_thinned", "warnings", "protocol_hash")}
    got = store_mod._record_from_json(json.dumps(old))
    assert got.ic_meta == {} and got.observables_thinned is None and got.warnings == []
    assert got.protocol_hash is None
    with pytest.raises(TypeError):
        store_mod._record_from_json(json.dumps({**old, "bogus": 1}))


def test_m4_append_after_another_host_took_over_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    old = Store(tmp_path / "s.sqlite")
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostB")
    Store(tmp_path / "s.sqlite", takeover=True)
    monkeypatch.setattr(store_mod, "_current_hostname", lambda: "hostA")
    with pytest.raises(StoreOwnershipError, match="hostB"):
        old.append(_make_record())
    with pytest.raises(StoreOwnershipError, match="hostB"):
        old.append_failure(_make_failure())
    assert _integrity(tmp_path / "s.sqlite")[1] == 0


def test_m5_iter_filters_by_physics_and_protocol_hash(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    for i, (p, q) in enumerate([("p1", "q1"), ("p1", "q2"), ("p2", "q1"), ("p1", None)]):
        store.append(_make_record(key_digest=f"{i:064x}", physics_config_hash=p, protocol_hash=q))
    assert [r.key_digest[-1] for r in store.iter(physics_config_hash="p1")] == ["0", "1", "3"]
    assert [r.key_digest[-1] for r in store.iter(protocol_hash="q1")] == ["0", "2"]
    assert [r.key_digest[-1] for r in store.iter(physics_config_hash="p1", protocol_hash="q1")] == ["0"]
    assert [r.key_digest[-1] for r in store.iter(protocol_hash=None)] == ["3"]
