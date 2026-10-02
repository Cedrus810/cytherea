"""Tests for cytherea.engine.shot (run_shot, ObsSpec) and cytherea.exec.batch
(run_batch, ShotFailure) -- Task 7.

Covers the brief's test table (task-7-brief.md, 7.1-7.5) plus the controller
decisions layered on top of it, including fix-round-1 rulings R33-R35:

- dt_obs must be an integer multiple of the propagator's own dt (within
  1e-9 relative, resolved *after* `backend.build(...)` -- ruling R34, since
  a backend's effective dt can depend on the `PhysicsConfig` passed to
  `build`), else ValueError, and nothing is recorded.
- ICRejectedError from run_shot records nothing and re-raises; from inside
  run_batch it becomes a ShotFailure entry and the batch continues.
- observables thinning by store_stride always keeps the first and last
  observation.
- the first observation comes from `propagator.get_state()` after `build()`,
  not the raw IC (ruling R35 #4).
- `run_batch` rejects `n_workers < 1` and duplicate keys up front (ruling
  R35 #3), and cancels not-yet-started futures promptly on a non-IC
  exception (ruling R35 #2).

Design note on run_shot + run_batch composition (ruling R33; see shot.py/
batch.py module docstrings for the full rationale): `run_shot`'s `store`
parameter is optional (`store: Store | None = None`) and is written to only
when given. Every `shot_fn` built in this file for use with `run_batch`
binds `run_shot` with `store=None` (see `_make_shot_fn`) -- `run_batch` is
the sole writer of the real target store, appending each `shot_fn`-returned
record itself, exactly once, in the parent process, in `keys` order,
regardless of `n_workers`. (An earlier version of this file had `shot_fn`
write to a private "scratch" store instead; that left a real crash-resume
gap -- a kill between `shot_fn` returning and `run_batch`'s own append could
leave a scratch-store entry with no matching real-store entry, and
recomputing that key on resume would then collide with the scratch store's
own uniqueness constraint. `store=None` removes the scratch store, and this
failure mode, entirely -- see tests 7.4a/7.4b below, which pin down exactly
the scenario that used to break.)

Everything here uses the Task 3 analytic backend, per the brief, except one
small test-double-based unit test isolating ruling R35 #4 (which the
analytic backend cannot exercise, since its `build()` never adjusts the
state it was given).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import multiprocessing
import os
import signal
import time

import numpy as np
import pytest

from cytherea.backends.analytic import AnalyticBackend, DoubleWell1D, LJCluster
from cytherea.backends.base import MDState, NumericalInstabilityError, PhysicsConfig
from cytherea.engine.shot import ObsSpec, run_shot
from cytherea.exec.batch import ShotFailure, run_batch
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import (
    EnsembleFrameSampler,
    ICRejectedError,
    InitialState,
    ValidityReport,
)
from cytherea.keys import ShotKey, key_digest
from cytherea.observe.events import FixedLag, SpecLabeler
from cytherea.store import ShotRecord, Store, config_hash

# ---------------------------------------------------------------------------
# module-level fixtures (must be module-level: some tests pickle these into
# worker processes via ProcessPoolExecutor(mp_context="spawn") or a spawned
# multiprocessing.Process, and Python's default pickler references functions
# and classes by (module, qualname), not by value).
# ---------------------------------------------------------------------------


def _shot_key(shot_id: int, global_seed: int = 1, frame_id: int = 0, stage: str = "shot") -> ShotKey:
    return ShotKey(global_seed=global_seed, frame_id=frame_id, shot_id=shot_id, stage=stage)


def _obs_position(state: MDState) -> float:
    return float(state.x[0])


def _obs_lj_dist(state: MDState) -> float:
    return float(np.linalg.norm(state.x[0] - state.x[1]))


def _label_sign(obs: dict) -> str:
    return "pos" if obs["position"] >= 0.0 else "neg"


# Described for the protocol hash (R39 + int2-m2: an opaque labeler is refused).
_LABEL_SIGN = SpecLabeler(_label_sign, {"obs": "position", "ge": 0.0, "then": "pos", "else": "neg"})


def _make_fixture_1d(gamma: float = 0.05):
    """A trivial 1D double-well shot: one fixed frame, no IC-gate checks (so
    the gate always passes on the first draw), baoab dynamics with low
    friction. Deterministic and cheap -- used by most of this module's tests.
    """
    potential = DoubleWell1D(barrier=2.0, x0=1.0)
    backend = AnalyticBackend(potential, integrator="baoab", dt=0.01, kT=1.0, gamma=gamma, mass=1.0)
    frame = EnsembleFrame(
        coordinates=np.array([0.3]),
        box=None,
        topology_ref="toy1d",
        temperature=1.0,
        weight=1.0,
        source_id="s0",
        frame_id=0,  # K3: ShotKey.frame_id (default 0 here) selects the frame
        time=0.0,
    )
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.array([1.0]),
        kT=1.0,
        backend=backend,
        energy_window=None,
        min_pair_dist=None,
        max_redraws=5,
        remove_com_momentum=False,
    )
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=2)
    return backend, sampler, obs


def _make_stop_1d() -> FixedLag:
    return FixedLag(tau=0.2)


def _make_shot_fn(physics_cfg: PhysicsConfig | None = None, gamma: float = 0.05, labeler=_LABEL_SIGN):
    """Build a `shot_fn` for use with `run_batch` (ruling R33): `run_shot` is
    bound with `store=None`, so calling `shot_fn(key)` only ever computes and
    returns a `ShotRecord` -- it never writes anywhere. `run_batch` is the
    sole writer of whatever real target store it is given.
    """
    backend, sampler, obs = _make_fixture_1d(gamma=gamma)
    stop = _make_stop_1d()
    return functools.partial(
        run_shot,
        sampler=sampler,
        backend=backend,
        stop=stop,
        obs=obs,
        physics_cfg=physics_cfg,
        store=None,
        labeler=labeler,
    )


def _lj_frame(frame_id, coords, weight=1.0, source_id="s", temperature=1.0, time=0.0):
    return EnsembleFrame(
        coordinates=np.asarray(coords, dtype=float),
        box=None,
        topology_ref="toy",
        temperature=temperature,
        weight=weight,
        source_id=source_id,
        frame_id=frame_id,
        time=time,
    )


def _lj_backend(n_atoms, kT=1.0):
    return AnalyticBackend(LJCluster(n_atoms), integrator="baoab", dt=0.001, kT=kT, gamma=0.1)


def _record_count(store: Store) -> int:
    return sum(1 for _ in store.iter())


def _crash_worker(store_path: str, keys: list[ShotKey], per_shot_delay: float) -> None:
    """multiprocessing.Process target for 7.4: runs `run_batch` serially, with
    an artificial per-shot delay so the parent has time to observe partial
    progress and SIGKILL this process mid-batch (arbitrary-timing crash, as
    opposed to 7.4a/7.4b's deterministic on_before_append-hook crash).
    """
    shot_fn = _make_shot_fn()

    def _slow_shot_fn(key: ShotKey) -> ShotRecord:
        time.sleep(per_shot_delay)
        return shot_fn(key)

    store = Store(store_path)
    run_batch(keys, _slow_shot_fn, store, n_workers=1)


def _crash_at_key_worker(store_path: str, keys: list[ShotKey], target_shot_id: int, n_workers: int) -> None:
    """multiprocessing.Process target for 7.4a/7.4b: deterministically
    SIGKILLs this process (and, for `n_workers > 1`, its whole
    `ProcessPoolExecutor` pool) the instant `target_shot_id`'s shot has been
    computed by `shot_fn` but *before* `run_batch`'s own `store.append` for
    it runs -- i.e. exactly the window Important-1 (fix round 1) identified
    as broken under the old scratch-store contract -- with no dependence on
    timing at all, via `run_batch`'s `on_before_append` hook.

    Because `run_batch` resolves and appends pending keys strictly in `keys`
    order (regardless of `n_workers` -- see `exec/batch.py`'s module
    docstring), this deterministically leaves exactly the keys before
    `target_shot_id` committed, `target_shot_id` itself computed-but-not-
    committed, and every key after it never even attempted.

    `os.setsid()` makes this process its own session/process-group leader
    before doing anything else, so that any `ProcessPoolExecutor` workers it
    later spawns (`n_workers > 1`) share *its* new process group. The kill
    hook then signals that whole group (`os.killpg(0, SIGKILL)`) instead of
    just this one PID: a plain single-PID `SIGKILL` here would leave the
    pool's worker processes as orphaned grandchildren with nothing left to
    ever signal them to exit (they are not this test's direct children, so
    nothing outside this process can clean them up afterwards) -- confirmed
    empirically to otherwise leak indefinitely-running `spawn_main`
    processes. Signaling the whole group is also the more faithful crash
    simulation: it is what actually happens when a process supervisor or
    shell kills a job's entire process tree.
    """
    os.setsid()
    shot_fn = _make_shot_fn()

    def _kill_at_target(record: ShotRecord) -> None:
        if record.key["shot_id"] == target_shot_id:
            os.killpg(0, signal.SIGKILL)

    store = Store(store_path)
    run_batch(keys, shot_fn, store, n_workers=n_workers, on_before_append=_kill_at_target)


# ---------------------------------------------------------------------------
# 7.1: n_workers=1 vs n_workers=4 over 100 keys -> field-identical records
# ---------------------------------------------------------------------------


def test_7_1_serial_and_parallel_batches_produce_identical_records(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(100)]

    store_serial = Store(tmp_path / "serial.sqlite")
    results_serial = run_batch(keys, _make_shot_fn(), store_serial, n_workers=1)

    store_parallel = Store(tmp_path / "parallel.sqlite")
    results_parallel = run_batch(keys, _make_shot_fn(), store_parallel, n_workers=4)

    assert len(results_serial) == len(results_parallel) == 100
    for rec_serial, rec_parallel in zip(results_serial, results_parallel):
        assert isinstance(rec_serial, ShotRecord)
        assert isinstance(rec_parallel, ShotRecord)
        assert rec_serial == rec_parallel

    for key in keys:
        digest = key_digest(key)
        assert store_serial.get(digest) == store_parallel.get(digest)


# ---------------------------------------------------------------------------
# 7.2: shuffled key order -> each key's own record is unchanged
# ---------------------------------------------------------------------------


def test_7_2_shuffled_key_order_same_records_per_key(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(30)]
    # A deterministic, non-identity permutation -- no RNG involved (project
    # rule: all randomness goes through derive_rng; this is just re-ordering
    # a list, not sampling anything).
    shuffled = list(reversed(keys))
    assert shuffled != keys

    store_ordered = Store(tmp_path / "ordered.sqlite")
    run_batch(keys, _make_shot_fn(), store_ordered, n_workers=1)

    store_shuffled = Store(tmp_path / "shuffled.sqlite")
    run_batch(shuffled, _make_shot_fn(), store_shuffled, n_workers=1)

    for key in keys:
        digest = key_digest(key)
        assert store_ordered.get(digest) == store_shuffled.get(digest)


# ---------------------------------------------------------------------------
# 7.3: an IC draw is rejected, redrawn, and succeeds -> n_redraws >= 1
# (contract K3: only velocity-level failures are redrawn, for the same frame;
# a coordinate-level failure such as the energy window raises instead)
# ---------------------------------------------------------------------------


class _FirstVelocityDrawNaN(EnsembleFrameSampler):
    """The first velocity draw of every `sample()` call is NaN (a
    velocity-level failure), so the accepted IC needs exactly one redraw."""

    def sample(self, key):
        self._n_draws = 0
        return super().sample(key)

    def _draw_velocities(self, rng, shape):
        v = super()._draw_velocities(rng, shape)
        self._n_draws += 1
        return np.full(shape, np.nan) if self._n_draws == 1 else v


def test_7_3_ic_redraw_then_success_is_recorded(tmp_path):
    backend = _lj_backend(2)
    good = _lj_frame(0, [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], weight=1.0)
    sampler = _FirstVelocityDrawNaN(
        pool=EnsembleFramePool([good]),
        masses=np.ones(2),
        kT=1.0,
        backend=backend,
        energy_window=(-10.0, 10.0),
        min_pair_dist=None,
        max_redraws=50,
    )
    key = _shot_key(shot_id=0)

    obs = ObsSpec(fns={"dist": _obs_lj_dist}, dt_obs=0.005, store_stride=1)
    stop = FixedLag(tau=0.005)
    store = Store(tmp_path / "store.sqlite")

    rec = run_shot(key, sampler, backend, stop, obs, None, store)

    assert rec.ic_validity["ok"] is True
    assert rec.ic_validity["n_redraws"] == 1
    assert store.get(key_digest(key)) == rec


# ---------------------------------------------------------------------------
# 7.4: crash-resume (Review Focus 3) -- kill mid-batch (arbitrary timing),
# resume, compare to a clean serial run record-for-record.
# ---------------------------------------------------------------------------


def test_7_4_crash_resume_matches_clean_run(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(20)]

    # Clean reference run: no interruption.
    clean_store = Store(tmp_path / "clean.sqlite")
    run_batch(keys, _make_shot_fn(), clean_store, n_workers=1)
    assert _record_count(clean_store) == len(keys)

    # Interrupted run: a subprocess is killed partway through, then resumed.
    resume_store_path = str(tmp_path / "resume.sqlite")
    store_for_polling = Store(resume_store_path)  # creates the schema up front

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_crash_worker, args=(resume_store_path, keys, 0.05))
    proc.start()

    deadline = time.monotonic() + 15.0
    killed = False
    while time.monotonic() < deadline:
        if _record_count(store_for_polling) >= 8:
            os.kill(proc.pid, signal.SIGKILL)
            killed = True
            break
        if not proc.is_alive():
            break
        time.sleep(0.01)
    proc.join(timeout=5.0)

    assert killed, "the worker finished before it could be killed -- test does not exercise resume"
    assert not proc.is_alive()

    partial_count = _record_count(store_for_polling)
    assert 0 < partial_count < len(keys), (
        "expected a genuinely partial store (some, not all, records present) "
        f"after the kill; got {partial_count} of {len(keys)}"
    )

    # Resume: same keys, same store file.
    resume_store = Store(resume_store_path)
    run_batch(keys, _make_shot_fn(), resume_store, n_workers=1)

    assert _record_count(resume_store) == len(keys)
    for key in keys:
        digest = key_digest(key)
        assert resume_store.get(digest) == clean_store.get(digest)


# ---------------------------------------------------------------------------
# 7.4a (fix round 1, ruling R33): the same crash-resume contract under
# n_workers > 1, with a deterministic kill point.
# ---------------------------------------------------------------------------


def test_7_4a_parallel_crash_resume_matches_clean_run(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(20)]
    target_shot_id = 10

    clean_store = Store(tmp_path / "clean.sqlite")
    run_batch(keys, _make_shot_fn(), clean_store, n_workers=1)
    assert _record_count(clean_store) == len(keys)

    resume_store_path = str(tmp_path / "resume.sqlite")
    store_for_check = Store(resume_store_path)

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_crash_at_key_worker, args=(resume_store_path, keys, target_shot_id, 4)
    )
    proc.start()
    proc.join(timeout=30.0)

    assert not proc.is_alive(), "worker did not terminate in time"
    assert proc.exitcode == -signal.SIGKILL, f"expected SIGKILL exit, got {proc.exitcode!r}"

    # run_batch appends strictly in keys order even with n_workers=4 (it
    # resolves futures in ascending index order): exactly the keys before
    # target_shot_id must have committed.
    assert _record_count(store_for_check) == target_shot_id
    for key in keys[:target_shot_id]:
        assert store_for_check.has(key_digest(key))
    assert not store_for_check.has(key_digest(keys[target_shot_id]))
    for key in keys[target_shot_id:]:
        assert not store_for_check.has(key_digest(key))

    resume_store = Store(resume_store_path)
    run_batch(keys, _make_shot_fn(), resume_store, n_workers=4)

    assert _record_count(resume_store) == len(keys)
    for key in keys:
        digest = key_digest(key)
        assert resume_store.get(digest) == clean_store.get(digest)


# ---------------------------------------------------------------------------
# 7.4b (fix round 1, ruling R33, Important-1): a crash landing exactly
# between shot_fn returning and the parent's own append -- the precise
# scenario the review identified as broken under the old scratch-store
# contract.
# ---------------------------------------------------------------------------


def test_7_4b_crash_between_shot_return_and_append_matches_clean_run(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(20)]
    target_shot_id = 10

    clean_store = Store(tmp_path / "clean.sqlite")
    run_batch(keys, _make_shot_fn(), clean_store, n_workers=1)
    assert _record_count(clean_store) == len(keys)

    resume_store_path = str(tmp_path / "resume.sqlite")
    store_for_check = Store(resume_store_path)

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_crash_at_key_worker, args=(resume_store_path, keys, target_shot_id, 1)
    )
    proc.start()
    proc.join(timeout=15.0)

    assert not proc.is_alive(), "worker did not terminate in time"
    assert proc.exitcode == -signal.SIGKILL, f"expected SIGKILL exit, got {proc.exitcode!r}"

    # Deterministic: exactly the keys strictly before target_shot_id
    # committed; target_shot_id's own record was computed (that's what
    # triggered the kill) but never appended; nothing after it was attempted.
    assert _record_count(store_for_check) == target_shot_id
    for key in keys[:target_shot_id]:
        assert store_for_check.has(key_digest(key))
    assert not store_for_check.has(key_digest(keys[target_shot_id]))
    for key in keys[target_shot_id:]:
        assert not store_for_check.has(key_digest(key))

    # Resuming must not raise (this is exactly the DuplicateKeyError
    # scenario Important-1 flagged under the old scratch-store contract) and
    # must reach full, clean-run-identical parity.
    resume_store = Store(resume_store_path)
    run_batch(keys, _make_shot_fn(), resume_store, n_workers=1)

    assert _record_count(resume_store) == len(keys)
    for key in keys:
        digest = key_digest(key)
        assert resume_store.get(digest) == clean_store.get(digest)


# ---------------------------------------------------------------------------
# 7.5: physics_config_hash matches config_hash(effective_config(physics_cfg)) (K9)
# ---------------------------------------------------------------------------


def test_7_5_physics_config_hash_matches_config_hash(tmp_path):
    backend, sampler, obs = _make_fixture_1d()
    stop = _make_stop_1d()
    physics_cfg = PhysicsConfig(
        integrator="verlet",
        dt_ps=0.002,
        temperature_K=300.0,
        friction_per_ps=0.0,
        constraints="none",
        rigid_water=False,
        platform="CPU",
        precision="mixed",
        deterministic_forces=False,
        purpose="measurement",
    )
    store = Store(tmp_path / "store.sqlite")
    key = _shot_key(shot_id=0)

    rec = run_shot(key, sampler, backend, stop, obs, physics_cfg, store)

    assert rec.physics_config_hash == config_hash(backend.effective_config(physics_cfg))

    # physics_cfg=None hashes the backend's effective config too (contract K9).
    store2 = Store(tmp_path / "store2.sqlite")
    rec_none_cfg = run_shot(_shot_key(shot_id=1), sampler, backend, _make_stop_1d(), obs, None, store2)
    assert rec_none_cfg.physics_config_hash == config_hash(backend.effective_config(None))
    assert rec_none_cfg.physics_config_hash != config_hash(None)
    # the analytic backend ignores cfg: the same dynamics, the same hash (K9)
    assert rec_none_cfg.physics_config_hash == rec.physics_config_hash


# ---------------------------------------------------------------------------
# Controller decisions not in the brief's table but locked by the task
# instructions (original round) and fix-round-1 rulings R33-R35.
# ---------------------------------------------------------------------------


def test_dt_obs_not_integer_multiple_of_propagator_dt_raises(tmp_path):
    backend, sampler, _obs = _make_fixture_1d()  # propagator.dt == backend.dt == 0.01
    bad_obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.017, store_stride=1)
    stop = _make_stop_1d()
    store = Store(tmp_path / "store.sqlite")
    key = _shot_key(shot_id=0)

    with pytest.raises(ValueError):
        run_shot(key, sampler, backend, stop, bad_obs, None, store)

    assert not store.has(key_digest(key))


def test_ic_rejected_records_nothing_and_reraises(tmp_path):
    # A single, permanently-NaN frame: the IC gate can never pass, so
    # sample() exhausts max_redraws and raises.
    frame = EnsembleFrame(
        coordinates=np.array([np.nan]),
        box=None,
        topology_ref="toy1d",
        temperature=1.0,
        weight=1.0,
        source_id="s",
        frame_id=0,
        time=0.0,
    )
    pool = EnsembleFramePool([frame])
    backend = AnalyticBackend(DoubleWell1D(barrier=2.0), integrator="baoab", dt=0.01, kT=1.0, gamma=0.05)
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.array([1.0]),
        kT=1.0,
        backend=backend,
        energy_window=None,
        min_pair_dist=None,
        max_redraws=2,
        remove_com_momentum=False,
    )
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=1)
    stop = FixedLag(tau=0.2)
    store = Store(tmp_path / "store.sqlite")
    key = _shot_key(shot_id=0)

    with pytest.raises(ICRejectedError):
        run_shot(key, sampler, backend, stop, obs, None, store)

    assert not store.has(key_digest(key))


def test_ic_rejected_in_batch_becomes_failure_without_stopping_batch(tmp_path):
    # Frame that always fails the IC gate (permanently NaN).
    nan_frame = EnsembleFrame(
        coordinates=np.array([np.nan]),
        box=None,
        topology_ref="toy1d",
        temperature=1.0,
        weight=1.0,
        source_id="s",
        frame_id=0,
        time=0.0,
    )
    bad_pool = EnsembleFramePool([nan_frame])
    bad_backend = AnalyticBackend(DoubleWell1D(barrier=2.0), integrator="baoab", dt=0.01, kT=1.0, gamma=0.05)
    bad_sampler = EnsembleFrameSampler(
        pool=bad_pool,
        masses=np.array([1.0]),
        kT=1.0,
        backend=bad_backend,
        energy_window=None,
        min_pair_dist=None,
        max_redraws=1,
        remove_com_momentum=False,
    )

    good_backend, good_sampler, good_obs = _make_fixture_1d()

    def shot_fn(key: ShotKey) -> ShotRecord:
        # store=None throughout, per ruling R33: this shot_fn (used with
        # run_batch below) never persists anything itself.
        if key.shot_id == 1:
            return run_shot(key, bad_sampler, bad_backend, FixedLag(tau=0.2), good_obs, None)
        return run_shot(key, good_sampler, good_backend, FixedLag(tau=0.2), good_obs, None)

    keys = [_shot_key(shot_id=i) for i in range(3)]
    store = Store(tmp_path / "real.sqlite")

    results = run_batch(keys, shot_fn, store, n_workers=1)

    assert len(results) == 3
    assert isinstance(results[0], ShotRecord)
    assert isinstance(results[1], ShotFailure)
    assert isinstance(results[2], ShotRecord)
    assert results[1].key_digest == key_digest(keys[1])
    assert len(results[1].reasons) >= 1

    assert store.has(key_digest(keys[0]))
    assert not store.has(key_digest(keys[1]))
    assert store.has(key_digest(keys[2]))


def test_observables_thinning_keeps_first_and_last(tmp_path):
    backend, sampler, _default_obs = _make_fixture_1d()
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=3)
    stop = FixedLag(tau=0.2)  # expected raw observations at t=0,.05,.10,.15,.20
    store = Store(tmp_path / "store.sqlite")
    key = _shot_key(shot_id=0)

    rec = run_shot(key, sampler, backend, stop, obs, None, store)

    times = rec.observables["t"]
    positions = rec.observables["position"]
    assert len(times) == len(positions)
    assert times[0] == pytest.approx(0.0, abs=1e-9)
    assert times[-1] == pytest.approx(0.2, abs=1e-9)
    # store_stride=3 must actually thin something (not keep every point) for
    # this to be a meaningful test.
    assert len(times) < 5


# ---------------------------------------------------------------------------
# Ruling R35 #4: the first observation must come from propagator.get_state()
# after build(), not the raw IC. The analytic backend's build() never
# adjusts the state it is given, so this needs a minimal test double to
# actually distinguish the two -- a real backend (e.g. one that projects
# constraints or minimizes energy at build time) could otherwise report a
# build-time-adjusted state that differs from the raw IC, and run_shot must
# observe *that*, not the pre-build IC.
# ---------------------------------------------------------------------------


class _FakeSampler:
    def __init__(self, istate: InitialState, validity: ValidityReport) -> None:
        self._istate = istate
        self._validity = validity

    def sample(self, key):
        return self._istate, self._validity

    def protocol_description(self):  # ruling R39: a test double describes itself
        return {"fake_sampler": True}


class _FakePropagator:
    def __init__(self, state_after_build: MDState, dt: float) -> None:
        self._state = state_after_build
        self.dt = dt

    def run(self, n_steps: int) -> None:
        pass

    def get_state(self) -> MDState:
        return self._state

    def set_state(self, s: MDState) -> None:
        self._state = s


class _FakeBackend:
    kind = "fake"
    gpu_resident = False

    def __init__(self, propagator: _FakePropagator) -> None:
        self._propagator = propagator

    def build(self, s, cfg, rng_key):
        return self._propagator

    def energy_forces(self, x, box=None):
        return 0.0, np.zeros_like(x)

    def effective_config(self, cfg=None) -> dict:
        return {"backend": "fake"}

    def provenance(self, cfg=None) -> dict:
        return {"kind": "fake"}


def test_first_observation_comes_from_propagator_get_state_not_raw_ic(tmp_path):
    raw_ic_state = MDState(x=np.array([999.0]), v=np.array([0.0]), t=0.0)
    # What the propagator reports right after build() -- deliberately
    # different from the raw IC, standing in for e.g. a constraint
    # projection or energy minimization a real backend might perform.
    post_build_state = MDState(x=np.array([0.0]), v=np.array([0.0]), t=0.0)

    istate = InitialState(state=raw_ic_state, frame_id=0, meta={})
    validity = ValidityReport(ok=True, reasons=[], checks={}, n_redraws=0)
    sampler = _FakeSampler(istate, validity)
    propagator = _FakePropagator(post_build_state, dt=0.01)
    backend = _FakeBackend(propagator)

    obs = ObsSpec(fns={"x0": lambda s: float(s.x[0])}, dt_obs=0.01, store_stride=1)
    stop = FixedLag(tau=0.0)  # fires on the very first observation
    store = Store(tmp_path / "store.sqlite")
    key = _shot_key(shot_id=0)

    rec = run_shot(key, sampler, backend, stop, obs, None, store)

    # Had run_shot used the raw IC for its first observation, x0 would be
    # 999.0; it must instead be 0.0, from propagator.get_state().
    assert rec.observables["x0"][0] == 0.0


# ---------------------------------------------------------------------------
# Ruling R35 #3: reject n_workers < 1 and duplicate keys up front.
# ---------------------------------------------------------------------------


def test_run_batch_rejects_n_workers_below_one(tmp_path):
    keys = [_shot_key(shot_id=0)]
    store = Store(tmp_path / "store.sqlite")

    with pytest.raises(ValueError):
        run_batch(keys, _make_shot_fn(), store, n_workers=0)

    assert _record_count(store) == 0


def test_run_batch_rejects_duplicate_keys(tmp_path):
    key = _shot_key(shot_id=0)
    store = Store(tmp_path / "store.sqlite")

    with pytest.raises(ValueError):
        run_batch([key, key], _make_shot_fn(), store, n_workers=1)

    assert _record_count(store) == 0


# ---------------------------------------------------------------------------
# Ruling R35 #2: on a non-IC exception from a future, cancel pending futures
# and stop promptly rather than waiting for every submitted shot to finish.
#
# This is checked functionally (which shots ever actually started), not by a
# wall-clock threshold: this machine sometimes runs other CPU-bound jobs
# concurrently (see project convention "no background contention" -- tests
# should not assume they have the machine to themselves), and a
# elapsed-time assertion here proved flaky under that contention. Instead,
# each shot that gets to run writes a marker file; with n_workers=2 and 8
# keys where shot_id=2 always raises immediately, at most 2 "waves" of
# shots (4 keys total, 3 of them markers -- shot_id=2 raises before writing
# its own marker) can possibly have already been dispatched to a worker by
# the time the parent observes the exception and cancels everything still
# queued -- reaching a 3rd wave (shot_id 5, 6 or 7) requires a worker to
# free up *again*, which cannot happen before cancellation takes effect.
# Without prompt cancellation (the pre-fix behavior: the executor's default
# `shutdown(wait=True)` inside a `with` block blocks until every submitted
# task finishes before the exception can propagate), every one of the 7
# non-raising keys would run to completion and write its marker.
# ---------------------------------------------------------------------------


def _boom_shot_fn_impl(key: ShotKey, marker_dir: str) -> ShotRecord:
    if key.shot_id == 2:
        raise RuntimeError("boom")
    marker_path = os.path.join(marker_dir, f"started_{key.shot_id}")
    open(marker_path, "w").close()
    time.sleep(0.3)
    return _make_shot_fn()(key)


def test_run_batch_parallel_cancels_pending_futures_on_error(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(8)]
    store = Store(tmp_path / "store.sqlite")
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    boom_shot_fn = functools.partial(_boom_shot_fn_impl, marker_dir=str(marker_dir))

    with pytest.raises(RuntimeError, match="boom"):
        run_batch(keys, boom_shot_fn, store, n_workers=2)

    started = {p.name for p in marker_dir.iterdir()}
    # 7 = every key except shot_id=2 (which raises before writing a marker).
    # Prompt cancellation must have kept at least one of them from ever
    # starting.
    assert len(started) < 7, (
        f"{len(started)} of 7 possible shots started -- pending futures "
        "were not cancelled promptly"
    )


# ===========================================================================
# Fix wave P1: engine side of contracts K1, K2, K4, K6, K8 and fullreview
# A-C1 (engine side), A-I1, A-I3, A-I6, A-I7, A-m7, A-m8, D-I5/D-I6 (record
# side).
# ===========================================================================

import cytherea.engine.shot as shot_mod
from cytherea.engine.shot import ProtocolDescriptionError, SpecPredicate, protocol_hash, spec_region
from cytherea.observe.events import AbsorbingAB, Region, StopDecision, offline_replay


def _fixture_with_frame_time(frame_time: float):
    potential = DoubleWell1D(barrier=2.0, x0=1.0)
    backend = AnalyticBackend(potential, integrator="baoab", dt=0.01, kT=1.0, gamma=0.05, mass=1.0)
    frame = EnsembleFrame(
        coordinates=np.array([0.3]), box=None, topology_ref="toy1d", temperature=1.0,
        weight=1.0, source_id="s0", frame_id=0, time=frame_time,
    )
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool([frame]), masses=np.array([1.0]), kT=1.0, backend=backend,
        energy_window=None, min_pair_dist=None, max_redraws=5, remove_com_momentum=False,
    )
    return backend, sampler


def _regions():
    return (
        spec_region("A", lambda o: o["position"] < -1.2, "position < -1.2"),
        spec_region("B", lambda o: o["position"] > 1.2, "position > 1.2"),
    )


# --- A-C1 / K1: the shot clock starts at 0 whatever the frame's time ------

@pytest.mark.parametrize("frame_time", [0.0, 50.0])
def test_fixed_lag_clock_starts_at_zero_regardless_of_frame_time(frame_time):
    backend, sampler = _fixture_with_frame_time(frame_time)
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=1)
    rec = run_shot(_shot_key(0), sampler, backend, FixedLag(tau=1.0), obs, None)
    assert rec.stop_reason == "fixed_lag"
    assert rec.observables["t"][0] == 0.0
    assert len(rec.observables["t"]) == 21  # 20 chunks of 0.05 -- not zero steps
    assert rec.event_time == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("frame_time", [0.0, 50.0])
def test_absorbing_t_max_is_relative_to_shot_start_regardless_of_frame_time(frame_time):
    backend, sampler = _fixture_with_frame_time(frame_time)
    A, B = _regions()
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=1)
    rec = run_shot(_shot_key(0), sampler, backend, AbsorbingAB(A, B, 0.1, 5.0), obs, None)
    assert rec.observables["t"][0] == 0.0
    assert len(rec.observables["t"]) > 1
    assert max(rec.observables["t"]) <= 5.0 + 1e-9


def test_engine_clock_is_step_index_times_dt_not_accumulated():
    # K1: t = step_index * dt (multiplication), bitwise.
    backend, sampler, _ = _make_fixture_1d()
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.07, store_stride=1)
    rec = run_shot(_shot_key(0), sampler, backend, FixedLag(tau=3.0), obs, None)
    n = len(rec.observables["t"])
    assert rec.observables["t"] == [(i * 7) * 0.01 for i in range(n)]


class _ClockFakePropagator:
    """A propagator whose own state.t is garbage: the engine must not use it."""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.n_run_calls = 0
        self.x = 0.3

    def run(self, n_steps):
        self.n_run_calls += 1

    def get_state(self):
        return MDState(x=np.array([self.x]), v=np.array([0.0]), t=123.456)

    def set_state(self, s):
        pass


class _RecordingBackend:
    """A backend double that records what build/provenance received."""

    kind = "fake"
    gpu_resident = False

    def __init__(self, propagator, effective=None, legacy=None):
        self._propagator = propagator
        self.built_with = []
        self.provenance_cfgs = []
        self.effective_cfgs = []
        self._effective = effective if effective is not None else {"backend": "fake"}
        if legacy == "no_effective_config":
            self.effective_config = None
        elif legacy == "provenance_without_cfg":
            self.provenance = lambda: {"kind": "fake"}

    def effective_config(self, cfg=None):
        self.effective_cfgs.append(cfg)
        return self._effective

    def build(self, s, cfg, rng_key):
        self.built_with.append((s, cfg))
        return self._propagator

    def energy_forces(self, x, box=None):
        return 0.0, np.zeros_like(x)

    def provenance(self, cfg=None):
        self.provenance_cfgs.append(cfg)
        return {"kind": "fake", "dt_ps": getattr(cfg, "dt_ps", None)}


def _fake_sampler(meta=None, t=0.0):
    istate = InitialState(
        state=MDState(x=np.array([0.3]), v=np.array([0.0]), t=t),
        frame_id=4,
        meta=meta if meta is not None else {},
    )
    return _FakeSampler(istate, ValidityReport(ok=True, reasons=[], checks={}, n_redraws=0))


def test_engine_ignores_propagator_state_time():
    prop = _ClockFakePropagator(dt=0.1)
    backend = _RecordingBackend(prop)
    obs = ObsSpec(fns={"x": lambda s: float(s.x[0])}, dt_obs=0.1, store_stride=1)
    rec = run_shot(_shot_key(0), _fake_sampler(), backend, FixedLag(tau=0.5), obs, None)
    assert rec.observables["t"] == [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5]
    assert rec.event_time == 0.5


def test_initial_state_is_built_at_t_zero_with_a_warning_if_rebased():
    prop = _ClockFakePropagator(dt=0.1)
    backend = _RecordingBackend(prop)
    obs = ObsSpec(fns={"x": lambda s: float(s.x[0])}, dt_obs=0.1, store_stride=1)
    rec = run_shot(_shot_key(0), _fake_sampler(t=50.0), backend, FixedLag(tau=0.0), obs, None)
    built_state, _cfg = backend.built_with[0]
    assert built_state.t == 0.0
    assert any("rebased" in w for w in rec.warnings)


# --- K2: InitialState.meta is recorded as ic_meta --------------------------

def test_record_carries_initial_state_meta_as_ic_meta():
    meta = {
        "frame_id": 4, "frame_time": 50.0, "frame_weight": np.float64(0.25),
        "source_id": "traj0", "topology_ref": "sha256:abc", "state": None, "n_redraws": 2,
    }
    prop = _ClockFakePropagator()
    rec = run_shot(_shot_key(0), _fake_sampler(meta=meta), _RecordingBackend(prop), FixedLag(tau=0.0),
                   ObsSpec(fns={"x": lambda s: float(s.x[0])}, dt_obs=0.1, store_stride=1), None)
    assert rec.ic_meta == {**meta, "frame_weight": 0.25}
    assert rec.frame_id == 4


def test_real_sampler_meta_is_recorded():
    backend, sampler, obs = _make_fixture_1d()
    rec = run_shot(_shot_key(0), sampler, backend, FixedLag(tau=0.1), obs, None)
    istate, _ = sampler.sample(_shot_key(0))
    assert rec.ic_meta == istate.meta
    assert rec.ic_meta["frame_weight"] == 1.0 and rec.ic_meta["source_id"] == "s0"


# --- A-I3 / K4: non-finite trajectories are recorded as "nonfinite" --------

class _ExplodingPropagator(_ClockFakePropagator):
    def __init__(self, what):
        super().__init__(dt=0.1)
        self.what = what
        self.v = 0.0

    def run(self, n_steps):
        super().run(n_steps)
        if self.what == "x":
            self.x = float("nan")
        else:
            self.v = float("inf")

    def get_state(self):
        # t advances here only so that pre-fix code (which used state.t)
        # terminates at t_max instead of looping forever.
        return MDState(x=np.array([self.x]), v=np.array([self.v]), t=self.n_run_calls * self.dt)


@pytest.mark.parametrize("what", ["x", "v"])
def test_nonfinite_state_or_observable_stops_with_nonfinite_not_timeout(what, tmp_path):
    A = spec_region("A", lambda o: o["x"] < -0.8, "x < -0.8")
    B = spec_region("B", lambda o: o["x"] > 0.8, "x > 0.8")
    prop = _ExplodingPropagator(what)
    store = Store(tmp_path / "s.sqlite")
    # "v" case: the observable itself stays finite -- only the state blew up.
    rec = run_shot(_shot_key(0), _fake_sampler(), _RecordingBackend(prop), AbsorbingAB(A, B, 0.0, 10.0),
                   ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None, store,
                   labeler=SpecLabeler(lambda o: "A", "always A"))
    assert rec.stop_reason == "nonfinite"
    assert rec.event_time == pytest.approx(0.1)
    assert len(rec.observables["t"]) == 2  # stopped at the first non-finite observation
    assert prop.n_run_calls == 1
    assert rec.final_state_label is None  # never label a blown-up trajectory
    assert store.get(rec.key_digest).stop_reason == "nonfinite"
    # int2-m1: the stored record replays to the same decision, also when only
    # the state (not the observable) went non-finite.
    replayed = offline_replay(AbsorbingAB(A, B, 0.0, 10.0), store.get(rec.key_digest))
    assert replayed == StopDecision("nonfinite", rec.event_time)


# --- K10: a backend's NumericalInstabilityError becomes a "nonfinite" record --

class _UnstablePropagator(_ClockFakePropagator):
    """Raises NumericalInstabilityError (as the OpenMM CPU/CUDA platforms do
    instead of returning a NaN state) from run() or get_state()."""

    def __init__(self, where, exc=None):
        super().__init__(dt=0.1)
        self.where = where
        self.exc = exc

    def _boom(self):
        raise self.exc if self.exc is not None else NumericalInstabilityError(
            "Particle coordinate is NaN."
        )

    def run(self, n_steps):
        super().run(n_steps)
        if self.where == "run":
            self._boom()

    def get_state(self):
        if self.where == "get_state" and self.n_run_calls >= 1:
            self._boom()
        return super().get_state()


@pytest.mark.parametrize("where", ["run", "get_state"])
def test_numerical_instability_is_recorded_as_nonfinite(where, tmp_path):
    A = spec_region("A", lambda o: o["x"] < -0.8, "x < -0.8")
    B = spec_region("B", lambda o: o["x"] > 0.8, "x > 0.8")
    prop = _UnstablePropagator(where)
    store = Store(tmp_path / "s.sqlite")
    rule = AbsorbingAB(A, B, 0.0, 10.0)
    rec = run_shot(_shot_key(0), _fake_sampler(), _RecordingBackend(prop), rule,
                   ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None, store,
                   labeler=SpecLabeler(lambda o: "A", "always A"))
    assert rec.stop_reason == "nonfinite"
    assert rec.event_time == pytest.approx(0.1)  # the observation the chunk was heading for
    assert rec.observables["t"] == [0.0, pytest.approx(0.1)]
    assert rec.observables["x"][0] == 0.3 and math.isnan(rec.observables["x"][1])
    assert rec.final_state_label is None
    assert any("NumericalInstabilityError" in w and "coordinate is NaN" in w for w in rec.warnings)
    stored = store.get(rec.key_digest)
    assert stored.stop_reason == "nonfinite"
    assert offline_replay(AbsorbingAB(A, B, 0.0, 10.0), stored) == StopDecision("nonfinite", rec.event_time)


def test_numerical_instability_in_batch_is_a_record_not_a_failure(tmp_path):
    """p1-N-I3: with the default on_error='raise' the batch must not abort,
    and the shot must be visible to estimators (a stored nonfinite record)."""
    prop = _UnstablePropagator("run")
    fn = functools.partial(
        run_shot, sampler=_fake_sampler(), backend=_RecordingBackend(prop),
        stop=FixedLag(tau=1.0), obs=ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), physics_cfg=None,
    )
    store = Store(tmp_path / "s.sqlite")
    out = run_batch([_shot_key(0)], fn, store)
    assert [r.stop_reason for r in out] == ["nonfinite"]
    assert [r.stop_reason for r in store.iter(stop_reason="nonfinite")] == ["nonfinite"]


def test_other_backend_errors_still_propagate():
    prop = _UnstablePropagator("run", exc=RuntimeError("device lost"))
    with pytest.raises(RuntimeError, match="device lost"):
        run_shot(_shot_key(0), _fake_sampler(), _RecordingBackend(prop), FixedLag(tau=1.0),
                 ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None)


# --- K6 / D-I6: thinning flag, and a real stored record replays exactly ----

@pytest.mark.parametrize("stride,flag", [(1, False), (2, True), (3, True)])
def test_observables_thinned_flag_follows_store_stride(stride, flag, tmp_path):
    backend, sampler, _ = _make_fixture_1d()
    store = Store(tmp_path / "s.sqlite")
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=stride)
    rec = run_shot(_shot_key(0), sampler, backend, FixedLag(tau=0.5), obs, None, store)
    assert rec.observables_thinned is flag
    assert store.get(rec.key_digest).observables_thinned is flag


def test_stride_one_stored_records_replay_offline_to_the_online_decision(tmp_path):
    backend, sampler, _ = _make_fixture_1d()
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=1)
    A, B = _regions()
    rules = {
        "absorbing": lambda: AbsorbingAB(A, B, 0.15, 4.0),
        "fixed_lag": lambda: FixedLag(tau=0.35),
    }
    reasons = set()
    for name, make_rule in rules.items():
        store = Store(tmp_path / f"{name}.sqlite")
        for i in range(12):
            online = run_shot(_shot_key(i), sampler, backend, make_rule(), obs, None, store)
            stored = store.get(online.key_digest)
            assert stored.observables_thinned is False
            decision = offline_replay(make_rule(), stored.observables)
            assert (decision.reason, decision.event_time) == (online.stop_reason, online.event_time)
            via_record = offline_replay(make_rule(), stored)
            assert (via_record.reason, via_record.event_time) == (online.stop_reason, online.event_time)
            reasons.add(online.stop_reason)
    assert "fixed_lag" in reasons and reasons & {"A", "B", "timeout"}


def test_thinned_stored_record_is_refused_by_record_mode_replay(tmp_path):
    backend, sampler, _ = _make_fixture_1d()
    store = Store(tmp_path / "s.sqlite")
    obs = ObsSpec(fns={"position": _obs_position}, dt_obs=0.05, store_stride=3)
    rec = run_shot(_shot_key(0), sampler, backend, FixedLag(tau=0.5), obs, None, store)
    stored = store.get(rec.key_digest)
    assert stored.observables_thinned is True
    with pytest.raises(ValueError, match="thinned"):
        offline_replay(FixedLag(tau=0.5), stored)


# --- K8 / A-I1: the hash and provenance describe the cfg actually used ----

def _pc(**overrides):
    base = dict(integrator="verlet", dt_ps=0.002, temperature_K=300.0, friction_per_ps=0.0,
                constraints="none", rigid_water=False, platform="Reference", precision="double",
                deterministic_forces=True, purpose="measurement")
    base.update(overrides)
    return PhysicsConfig(**base)


def test_explicit_cfg_is_hashed_and_passed_to_build_and_provenance():
    cfg = _pc(dt_ps=0.001)
    backend = _RecordingBackend(_ClockFakePropagator(), effective=_pc(dt_ps=0.004))
    rec = run_shot(_shot_key(0), _fake_sampler(), backend, FixedLag(tau=0.0),
                   ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), cfg)
    assert rec.physics_config_hash == config_hash(_pc(dt_ps=0.004))  # K9: the effective config
    assert backend.built_with[0][1] is cfg
    assert backend.provenance_cfgs == [cfg]
    assert backend.effective_cfgs == [cfg]  # effective_config(cfg) with the cfg given to build
    assert rec.backend_provenance["dt_ps"] == 0.001
    assert rec.warnings == []


def test_none_cfg_hashes_backend_effective_config():
    eff = _pc(dt_ps=0.004)
    backend = _RecordingBackend(_ClockFakePropagator(), effective=eff)
    rec = run_shot(_shot_key(0), _fake_sampler(), backend, FixedLag(tau=0.0),
                   ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None)
    assert rec.physics_config_hash == config_hash(eff)
    assert rec.physics_config_hash != config_hash(None)
    assert backend.built_with[0][1] is None  # build still gets what the caller passed
    assert rec.warnings == []


@pytest.mark.parametrize("legacy", ["no_effective_config", "provenance_without_cfg"])
def test_backend_without_the_k8_protocol_is_rejected_before_sampling(legacy):
    # PotentialBackend requires effective_config(cfg=None) and
    # provenance(cfg=None); there is no config_hash(None) fallback any more.
    sampler = _CountingSampler()
    backend = _RecordingBackend(_ClockFakePropagator(), legacy=legacy)
    with pytest.raises(TypeError, match="contract K8"):
        run_shot(_shot_key(0), sampler, backend, FixedLag(tau=0.0),
                 ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None)
    assert sampler.calls == 0 and backend.built_with == []


@pytest.mark.parametrize("via", ["explicit", "effective"])
def test_equilibration_purpose_is_rejected_before_build(via):
    eq = _pc(integrator="langevin_middle", friction_per_ps=1.0, purpose="equilibration")
    backend = _RecordingBackend(_ClockFakePropagator(), effective=eq if via == "effective" else None)
    with pytest.raises(ValueError, match="measurement"):
        run_shot(_shot_key(0), _fake_sampler(), backend, FixedLag(tau=0.0),
                 ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), eq if via == "explicit" else None)
    assert backend.built_with == []


# --- A-m7 / A-m8 / A-I6: ObsSpec and observable values validated up front --

class _CountingSampler(_FakeSampler):
    def __init__(self):
        super().__init__(_fake_sampler()._istate, ValidityReport(True, [], {}, 0))
        self.calls = 0

    def sample(self, key):
        self.calls += 1
        return super().sample(key)


@pytest.mark.parametrize(
    "obs",
    [
        ObsSpec(fns={"t": lambda s: 0.0}, dt_obs=0.1, store_stride=1),       # would overwrite the clock
        ObsSpec(fns={"x": lambda s: 0.0}, dt_obs=0.1, store_stride=0),       # stride < 1
        ObsSpec(fns={"x": lambda s: 0.0}, dt_obs=0.1, store_stride=1.5),     # non-integer stride
        ObsSpec(fns={"x": lambda s: 0.0}, dt_obs=float("nan"), store_stride=1),
    ],
)
def test_bad_obs_spec_is_rejected_before_sampling(obs):
    sampler = _CountingSampler()
    with pytest.raises((ValueError, TypeError)):
        run_shot(_shot_key(0), sampler, _RecordingBackend(_ClockFakePropagator()), FixedLag(tau=0.0), obs, None)
    assert sampler.calls == 0


def test_numpy_observable_values_are_stored_as_python_floats(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    obs = ObsSpec(fns={"f32": lambda s: np.float32(1.5), "arr0": lambda s: np.array(2.5),
                       "i": lambda s: np.int64(3)}, dt_obs=0.1, store_stride=1)
    rec = run_shot(_shot_key(0), _fake_sampler(), _RecordingBackend(_ClockFakePropagator()),
                   FixedLag(tau=0.1), obs, None, store)
    assert rec.observables["f32"] == [1.5, 1.5] and rec.observables["arr0"] == [2.5, 2.5]
    assert all(type(v) is float for name in ("f32", "arr0", "i") for v in rec.observables[name])
    assert store.get(rec.key_digest) == rec


@pytest.mark.parametrize("bad", [lambda s: [1.0, 2.0], lambda s: "x", lambda s: None])
def test_non_scalar_observable_fails_before_any_integration(bad):
    prop = _ClockFakePropagator()
    with pytest.raises(TypeError, match="observable 'bad'"):
        run_shot(_shot_key(0), _fake_sampler(), _RecordingBackend(prop), FixedLag(tau=1.0),
                 ObsSpec(fns={"bad": bad}, dt_obs=0.1, store_stride=1), None)
    assert prop.n_run_calls == 0


# --- A-I7: code_version is host-independent and resolved once --------------

def test_code_version_is_a_source_hash_independent_of_git(monkeypatch):
    v = shot_mod.code_version()
    assert "+src." in v and v == shot_mod.code_version()
    ident = shot_mod.code_identity(v)
    assert ident.count("+src.") == 1 and ".g" not in ident and "+dirty" not in ident
    # a host without git (node 180) reports the same identity
    shot_mod.code_version.cache_clear()
    monkeypatch.setattr(shot_mod, "_git_state", lambda: (None, False))
    try:
        assert shot_mod.code_identity(shot_mod.code_version()) == ident
    finally:
        shot_mod.code_version.cache_clear()


def test_code_identity_ignores_dirty_suffix_and_git_component():
    assert shot_mod.code_identity("0.1.0+src.0123456789ab.g5e84e88+dirty") == "0.1.0+src.0123456789ab"
    assert shot_mod.code_identity("0.1.0+src.0123456789ab") == "0.1.0+src.0123456789ab"
    assert shot_mod.code_identity("0.1.0+5e84e88+dirty") == "0.1.0+5e84e88"


# ===========================================================================
# Fix wave P1: run_batch -- persisted failures (A-I2), resume refuses a
# config/code mismatch (A-I4, contract K8), record/key check (A-m6),
# shot_fn shipped once per worker (A-m5), one code version per batch (A-I7).
# ===========================================================================

import dataclasses as _dc
import logging as _logging

from cytherea.exec.batch import ResumeConfigMismatchError


def _nan_sampler_and_backend():
    frame = EnsembleFrame(coordinates=np.array([np.nan]), box=None, topology_ref="toy1d",
                          temperature=1.0, weight=1.0, source_id="s", frame_id=0, time=0.0)
    backend = AnalyticBackend(DoubleWell1D(barrier=2.0), integrator="baoab", dt=0.01, kT=1.0, gamma=0.05)
    sampler = EnsembleFrameSampler(pool=EnsembleFramePool([frame]), masses=np.array([1.0]), kT=1.0,
                                   backend=backend, energy_window=None, min_pair_dist=None,
                                   max_redraws=1, remove_com_momentum=False)
    return sampler, backend


class _MixedShotFn:
    """shot_id 1 always fails the IC gate; shot_id 2 raises RuntimeError when
    `boom` is set; everything else is a normal shot. Counts calls per key."""

    def __init__(self, boom=False):
        self.good = _make_shot_fn()
        self.bad_sampler, self.bad_backend = _nan_sampler_and_backend()
        self.boom = boom
        self.calls = []

    def __call__(self, key):
        self.calls.append(key.shot_id)
        if key.shot_id == 1:
            return run_shot(key, self.bad_sampler, self.bad_backend, FixedLag(tau=0.2),
                            ObsSpec({"position": _obs_position}, 0.05, 1), None)
        if key.shot_id == 2 and self.boom:
            raise RuntimeError("deterministic dynamics failure")
        return self.good(key)


def test_ic_rejection_is_persisted_and_skipped_on_resume_unless_retried(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(3)]
    store = Store(tmp_path / "s.sqlite")
    fn = _MixedShotFn()
    first = run_batch(keys, fn, store)
    assert isinstance(first[1], ShotFailure) and first[1].failure_kind == "ic_rejected"
    logged = store.failures(key_digest(keys[1]))
    assert len(logged) == 1
    assert logged[0].failure_kind == "ic_rejected" and logged[0].error_type == "ICRejectedError"
    assert logged[0].reasons == first[1].reasons and len(logged[0].reasons) >= 1
    assert logged[0].attempts == first[1].attempts and logged[0].level == first[1].level
    assert logged[0].code_version == shot_mod.code_version()
    assert not store.has(key_digest(keys[1]))

    fn2 = _MixedShotFn()
    again = run_batch(keys, fn2, store)
    assert fn2.calls == []  # nothing re-run: records and the failure are both on file
    assert isinstance(again[1], ShotFailure) and again[1].from_store
    assert again[1].reasons == first[1].reasons

    fn3 = _MixedShotFn()
    retried = run_batch(keys, fn3, store, retry_failures=True)
    assert fn3.calls == [1]
    assert isinstance(retried[1], ShotFailure) and not retried[1].from_store
    assert len(store.failures(key_digest(keys[1]))) == 2


@pytest.mark.parametrize("n_workers", [1, 2])
def test_on_error_record_persists_a_failure_and_continues(tmp_path, n_workers):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    results = run_batch(keys, _MixedShotFn(boom=True), store, n_workers=n_workers, on_error="record")
    assert [type(r).__name__ for r in results] == ["ShotRecord", "ShotFailure", "ShotFailure", "ShotRecord"]
    assert results[2].failure_kind == "error" and results[2].error_type == "RuntimeError"
    (logged,) = store.failures(key_digest(keys[2]))
    assert logged.reasons == ["error:RuntimeError"]
    assert "deterministic dynamics failure" in logged.message
    assert "Traceback" in logged.traceback
    assert store.has(key_digest(keys[3]))
    # resume does not hit the failing key again
    fn = _MixedShotFn(boom=True)
    run_batch(keys, fn, store, on_error="record")
    assert fn.calls == []


def test_on_error_raise_is_the_default_and_persists_nothing(tmp_path):
    keys = [_shot_key(shot_id=i) for i in (0, 2, 3)]
    store = Store(tmp_path / "s.sqlite")
    with pytest.raises(RuntimeError, match="deterministic"):
        run_batch(keys, _MixedShotFn(boom=True), store)
    assert store.failures() == []
    assert store.has(key_digest(keys[0])) and not store.has(key_digest(keys[2]))


def test_on_error_value_is_validated(tmp_path):
    with pytest.raises(ValueError):
        run_batch([_shot_key(0)], _make_shot_fn(), Store(tmp_path / "s.sqlite"), on_error="ignore")


# --- K8 / A-I4: resume refuses records made with another config/code ------

def test_resume_with_a_different_physics_config_is_refused(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    # K9: the analytic backend ignores cfg, so change its own dynamics
    run_batch(keys[:2], _make_shot_fn(_pc(), gamma=0.05), store)
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_batch(keys, _make_shot_fn(_pc(), gamma=0.06), store)
    assert sum(1 for _ in store.iter()) == 2  # nothing new was written


def test_resume_mismatch_is_detected_for_an_opaque_shot_fn_before_any_append(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _make_shot_fn(_pc(), gamma=0.05), store)
    inner = _make_shot_fn(_pc(), gamma=0.06)
    with pytest.raises(ResumeConfigMismatchError):
        run_batch(keys, lambda k: inner(k), store)  # a plain callable: hash not knowable up front
    assert sum(1 for _ in store.iter()) == 2


def test_allow_config_change_resumes_anyway_and_logs(tmp_path, caplog):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    old_fn, new_fn = _make_shot_fn(_pc(), gamma=0.05), _make_shot_fn(_pc(), gamma=0.06)
    run_batch(keys[:2], old_fn, store)
    with caplog.at_level(_logging.WARNING, logger="cytherea.exec.batch"):
        results = run_batch(keys, new_fn, store, allow_config_change=True)
    assert all(isinstance(r, ShotRecord) for r in results)
    old_hash = config_hash(old_fn.keywords["backend"].effective_config(None))
    new_hash = config_hash(new_fn.keywords["backend"].effective_config(None))
    assert old_hash != new_hash
    assert results[0].physics_config_hash == old_hash  # old record kept
    assert results[3].physics_config_hash == new_hash
    assert any("allow_config_change" in r.getMessage() for r in caplog.records)


def _store_with_code_version(tmp_path, version):
    store = Store(tmp_path / "s.sqlite")
    rec = _make_shot_fn()(_shot_key(0))
    store.append(_dc.replace(rec, code_version=version))
    return store


def test_resume_with_different_code_is_refused(tmp_path):
    store = _store_with_code_version(tmp_path, "0.1.0+src.000000000000.g1234567+dirty")
    with pytest.raises(ResumeConfigMismatchError, match="code_version"):
        run_batch([_shot_key(0), _shot_key(1)], _make_shot_fn(), store)


def test_resume_ignores_dirty_suffix_and_git_component_of_code_version(tmp_path):
    ident = shot_mod.code_identity(shot_mod.code_version())
    store = _store_with_code_version(tmp_path, ident + ".gdeadbeef0000+dirty")
    results = run_batch([_shot_key(0), _shot_key(1)], _make_shot_fn(), store)
    assert all(isinstance(r, ShotRecord) for r in results)


def test_fresh_record_from_other_code_is_refused(tmp_path):
    # A-I7: a shot computed by different code than this batch's (e.g. a
    # worker that imported a newer checkout) must not be appended silently.
    inner = _make_shot_fn()
    store = Store(tmp_path / "s.sqlite")
    with pytest.raises(ResumeConfigMismatchError, match="code_version"):
        run_batch([_shot_key(0)], lambda k: _dc.replace(inner(k), code_version="9.9+src.ffffffffffff"), store)
    assert not store.has(key_digest(_shot_key(0)))


# --- A-m6: the returned record must be the one for the key ----------------

def test_record_for_the_wrong_key_is_refused(tmp_path):
    inner = _make_shot_fn()
    store = Store(tmp_path / "s.sqlite")
    with pytest.raises(RuntimeError, match="key_digest"):
        run_batch([_shot_key(0)], lambda k: inner(_shot_key(k.shot_id + 1)), store)
    assert sum(1 for _ in store.iter()) == 0


# --- A-m5: under n_workers > 1, shot_fn is pickled once per worker --------

_PICKLE_COUNT = {"n": 0}


class _CountingPickleShotFn:
    def __init__(self):
        self.fn = _make_shot_fn()

    def __call__(self, key):
        return self.fn(key)

    def __reduce__(self):
        _PICKLE_COUNT["n"] += 1
        return (_CountingPickleShotFn, ())


def test_parallel_batch_ships_shot_fn_once_per_worker(tmp_path):
    _PICKLE_COUNT["n"] = 0
    keys = [_shot_key(shot_id=i) for i in range(12)]
    results = run_batch(keys, _CountingPickleShotFn(), Store(tmp_path / "s.sqlite"), n_workers=2)
    assert all(isinstance(r, ShotRecord) for r in results)
    assert _PICKLE_COUNT["n"] <= 2


# ===========================================================================
# Ruling R39: records carry protocol_hash (stop rule + params, ObsSpec,
# sampler/IC config incl. frame pool content); resume refuses a change.
# ===========================================================================


def _proto_shot_fn(tau=0.2, dt_obs=0.05, stride=1, kT=1.0, weight=1.0):
    backend = AnalyticBackend(DoubleWell1D(barrier=2.0, x0=1.0), integrator="baoab", dt=0.01,
                              kT=1.0, gamma=0.05, mass=1.0)
    frame = EnsembleFrame(coordinates=np.array([0.3]), box=None, topology_ref="toy1d", temperature=1.0,
                          weight=weight, source_id="s0", frame_id=0, time=0.0)
    sampler = EnsembleFrameSampler(pool=EnsembleFramePool([frame]), masses=np.array([1.0]), kT=kT,
                                   backend=backend, energy_window=None, min_pair_dist=None,
                                   max_redraws=5, remove_com_momentum=False)
    return functools.partial(
        run_shot, sampler=sampler, backend=backend, stop=FixedLag(tau=tau),
        obs=ObsSpec(fns={"position": _obs_position}, dt_obs=dt_obs, store_stride=stride),
        physics_cfg=None, store=None,
    )


def test_records_carry_the_protocol_hash():
    fn = _proto_shot_fn()
    rec = fn(_shot_key(0))
    kw = fn.keywords
    assert rec.protocol_hash == protocol_hash(kw["stop"], kw["obs"], kw["sampler"])
    # fresh, equal objects describe the same protocol
    assert _proto_shot_fn()(_shot_key(0)).protocol_hash == rec.protocol_hash
    desc = shot_mod.protocol_description(kw["stop"], kw["obs"], kw["sampler"])
    assert desc["stop_rule"] == {"kind": "fixed_lag", "class": "FixedLag", "params": {"tau": 0.2}}
    assert desc["obs"] == {"observables": ["position"], "dt_obs": 0.05, "store_stride": 1}
    assert desc["sampler"]["kT"] == 1.0 and "backend" not in desc["sampler"]
    assert desc["sampler"]["pool"]["n_frames"] == 1


@pytest.mark.parametrize(
    "change",
    [dict(tau=0.3), dict(dt_obs=0.1), dict(stride=2), dict(kT=1.5), dict(weight=2.0)],
    ids=["FixedLag_tau", "dt_obs", "store_stride", "kT", "frame_weight"],
)
def test_resume_with_a_changed_protocol_is_refused(tmp_path, change):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _proto_shot_fn(), store)
    with pytest.raises(ResumeConfigMismatchError, match="protocol_hash"):
        run_batch(keys, _proto_shot_fn(**change), store)
    assert sum(1 for _ in store.iter()) == 2
    # the same change is accepted when explicitly allowed
    run_batch(keys, _proto_shot_fn(**change), store, allow_config_change=True)
    assert sum(1 for _ in store.iter()) == 4


def test_resume_with_an_unchanged_protocol_proceeds(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _proto_shot_fn(), store)
    results = run_batch(keys, _proto_shot_fn(), store)
    assert all(isinstance(r, ShotRecord) for r in results)
    assert len({r.protocol_hash for r in results}) == 1


def test_protocol_change_is_caught_for_an_opaque_shot_fn_before_any_append(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _proto_shot_fn(), store)
    inner = _proto_shot_fn(tau=0.3)
    with pytest.raises(ResumeConfigMismatchError, match="protocol_hash"):
        run_batch(keys, lambda k: inner(k), store)
    assert sum(1 for _ in store.iter()) == 2


def test_stored_record_without_protocol_hash_is_refused_on_resume(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    rec = _proto_shot_fn()(_shot_key(0))
    store.append(_dc.replace(rec, protocol_hash=None))  # e.g. written before R39
    with pytest.raises(ResumeConfigMismatchError, match="protocol_hash None"):
        run_batch([_shot_key(0), _shot_key(1)], _proto_shot_fn(), store)


def test_region_with_opaque_predicate_and_no_spec_is_rejected_before_sampling():
    sampler = _CountingSampler()
    rule = AbsorbingAB(Region("A", lambda o: o["x"] < -1.0), Region("B", lambda o: o["x"] > 1.0), 0.1, 5.0)
    with pytest.raises(ProtocolDescriptionError, match="Region 'A'"):
        run_shot(_shot_key(0), sampler, _RecordingBackend(_ClockFakePropagator()), rule,
                 ObsSpec({"x": lambda s: float(s.x[0])}, 0.1, 1), None)
    assert sampler.calls == 0


def test_region_spec_enters_the_protocol_hash():
    obs = ObsSpec({"x": lambda s: 0.0}, 0.1, 1)
    sampler = _proto_shot_fn().keywords["sampler"]

    def rule(spec_a, t_max=5.0):
        a = Region("A", SpecPredicate(lambda o: o["x"] < -1.0, spec_a))
        return AbsorbingAB(a, spec_region("B", lambda o: o["x"] > 1.0, "x > 1"), 0.1, t_max)

    h = protocol_hash(rule("x < -1"), obs, sampler)
    assert h == protocol_hash(rule("x < -1"), obs, sampler)
    assert h != protocol_hash(rule("x < -2"), obs, sampler)
    assert h != protocol_hash(rule("x < -1", t_max=6.0), obs, sampler)
    # a predicate that is a plain function with a `spec` attribute also works
    def pred(o):
        return o["x"] < -1.0
    pred.spec = "x < -1"
    a2 = AbsorbingAB(Region("A", pred), spec_region("B", lambda o: o["x"] > 1.0, "x > 1"), 0.1, 5.0)
    assert protocol_hash(a2, obs, sampler) == h


class _CustomRule:
    kind = "fixed_lag"

    def __init__(self, tau, describe=True):
        self.tau = tau
        if describe:
            self.protocol_description = lambda: {"tau": self.tau}

    def reset(self):
        pass

    def update(self, obs, t):
        return None


def test_unknown_stop_rule_needs_protocol_description():
    obs = ObsSpec({"x": lambda s: 0.0}, 0.1, 1)
    sampler = _proto_shot_fn().keywords["sampler"]
    with pytest.raises(ProtocolDescriptionError, match="protocol_description"):
        protocol_hash(_CustomRule(1.0, describe=False), obs, sampler)
    assert protocol_hash(_CustomRule(1.0), obs, sampler) != protocol_hash(_CustomRule(2.0), obs, sampler)


class _LooseSampler:
    """A sampler-like object for protocol description only."""

    def __init__(self, constraints, weight=1.0):
        frame = EnsembleFrame(coordinates=np.zeros((2, 3)), box=None, topology_ref="t", temperature=1.0,
                              weight=weight, source_id="s", frame_id=0, time=0.0)
        self.pool = EnsembleFramePool([frame])
        self.kT = 1.0
        self.masses = np.ones(2)
        self.constraints = constraints
        self.backend = object()  # excluded: covered by physics_config_hash


class _PairConstraints:
    def __init__(self, d):
        self.pairs = np.array([[0, 1]])
        self.distances = np.array([d])
        self.projection_tolerance = 1e-10


class _GoldenSampler:
    def protocol_description(self):
        return {"sampler": "golden"}


def _golden_rules():
    from cytherea.observe.events import BSurface

    return {
        "fixed_lag_0.2": FixedLag(0.2),
        "fixed_lag_1000": FixedLag(1000.0),
        "fixed_lag_int1": FixedLag(1),
        "absorbing": AbsorbingAB(
            spec_region("A", lambda o: o["x"] < -1.0, "x < -1"),
            Region("B", SpecPredicate(lambda o: o["x"] > 1.0, {"obs": "x", "gt": 1.0})),
            0.1, 5.0,
        ),
        "bsurface": BSurface(spec_region("reaction", lambda o: o["x"] > 2.0, "x > 2"), "r", 2.5, 0.05, 10.0),
    }


# Frozen with P1's private-attribute introspection (phase-a 0aca052, before
# the stop rules had a public protocol_description()): moving the parameters
# behind the public method must not change any stored protocol_hash.
_GOLDEN_PROTOCOL_HASHES = {
    "absorbing": "ced8d375cc2c055cb3189744f3478b1bb9a03aa2f702e730c64eaddb0ba37c49",
    "bsurface": "ebcedb33a3d93b21cf2adfcf33a9ee921449b48067f2668b9ed2d5e1a0b241f2",
    "fixed_lag_0.2": "5c0041ad73c041271bab316f3858aa8ad7e65cc311368433d9f3fdbc02b8b402",
    "fixed_lag_1000": "5f9c34d22067cdc857978753326a6ab97e750bffc6b927fe62a9d67369174fb0",
    "fixed_lag_int1": "7f139da2b2e5eecd90fa1513de80cbf657e0bda2e46c3bfae52538a2f8b09ce5",
}


def test_stop_rule_public_protocol_description_keeps_the_frozen_protocol_hashes():
    obs = ObsSpec(fns={"x": lambda s: 0.0, "r": lambda s: 0.0}, dt_obs=0.05, store_stride=1)
    got = {k: protocol_hash(r, obs, _GoldenSampler()) for k, r in _golden_rules().items()}
    assert got == _GOLDEN_PROTOCOL_HASHES
    desc = shot_mod.protocol_description(_golden_rules()["absorbing"], obs, _GoldenSampler())["stop_rule"]
    assert desc == {
        "kind": "absorbing_AB", "class": "AbsorbingAB",
        "params": {"A": {"region": "A", "spec": "x < -1"},
                   "B": {"region": "B", "spec": {"gt": 1.0, "obs": "x"}},
                   "tau_persist": 0.1, "t_max": 5.0},
    }


def test_engine_describes_stop_rules_through_their_public_method():
    obs = ObsSpec({"x": lambda s: 0.0}, 0.1, 1)

    class _Relabelled(FixedLag):
        def protocol_description(self):
            return {"tau": self._tau, "variant": "relabelled"}

    d = shot_mod.protocol_description(_Relabelled(0.2), obs, _GoldenSampler())["stop_rule"]
    assert d["params"] == {"tau": 0.2, "variant": "relabelled"}
    # the engine re-exports the observe-side names unchanged
    from cytherea import observe

    assert ProtocolDescriptionError is observe.ProtocolDescriptionError
    assert SpecPredicate is observe.SpecPredicate and spec_region is observe.spec_region


def test_sampler_constraints_must_be_describable():
    obs = ObsSpec({"x": lambda s: 0.0}, 0.1, 1)
    stop = FixedLag(tau=1.0)
    with pytest.raises(ProtocolDescriptionError, match="sampler.constraints"):
        protocol_hash(stop, obs, _LooseSampler(lambda x, v: (x, v, 0.0)))
    h1 = protocol_hash(stop, obs, _LooseSampler(_PairConstraints(0.1)))
    assert h1 == protocol_hash(stop, obs, _LooseSampler(_PairConstraints(0.1)))
    assert h1 != protocol_hash(stop, obs, _LooseSampler(_PairConstraints(0.11)))
    assert h1 != protocol_hash(stop, obs, _LooseSampler(_PairConstraints(0.1), weight=2.0))
    assert protocol_hash(stop, obs, _LooseSampler(None)) != h1


# ===========================================================================
# Fix wave 2, package L1: contract K9 (physics hash = the effective config,
# for cfg=None and an explicit cfg alike) and R39 + labeler (int2-m2).
# ===========================================================================

from cytherea.engine.shot import resolve_physics_config  # noqa: E402


def _label_at(threshold):
    """A described labeler (R39 + int2-m2): `spec` says what it computes."""

    def lab(obs):
        return "pos" if obs["position"] >= threshold else "neg"

    lab.spec = {"obs": "position", "ge": threshold, "then": "pos", "else": "neg"}
    return lab


def test_k9_analytic_explicit_cfg_gamma_change_refuses_resume(tmp_path):
    """int2-I1 / p1-N-I1 / p5-m3: the analytic backend ignores cfg, so with
    an explicit cfg the hash must still describe its real dynamics."""
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _make_shot_fn(_pc(), gamma=0.05), store)
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_batch(keys, _make_shot_fn(_pc(), gamma=0.08), store)
    assert sum(1 for _ in store.iter()) == 2


def test_k9_none_and_equivalent_explicit_cfg_give_the_same_hash(tmp_path):
    """K9 removes the K8 known limitation: the same dynamics hash one way."""
    backend, _sampler, _obs = _make_fixture_1d()
    h_none = resolve_physics_config(backend, None)[1]
    h_cfg = resolve_physics_config(backend, _pc())[1]
    assert h_none == h_cfg == config_hash(backend.effective_config(None))
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _make_shot_fn(None), store)
    results = run_batch(keys, _make_shot_fn(_pc()), store)  # accepted: same dynamics
    assert len({r.physics_config_hash for r in results}) == 1


def test_k9_labeler_spec_enters_protocol_hash_and_refuses_resume(tmp_path):
    """int2-m2: for FixedLag shots the label *is* the outcome, so a changed
    labeler must not be resumed silently."""
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _make_shot_fn(labeler=_label_at(0.0)), store)
    run_batch(keys[:3], _make_shot_fn(labeler=_label_at(0.0)), store)  # same spec: fine
    with pytest.raises(ResumeConfigMismatchError, match="protocol_hash"):
        run_batch(keys, _make_shot_fn(labeler=_label_at(0.5)), store)
    assert sum(1 for _ in store.iter()) == 3


def test_k9_opaque_labeler_is_refused_before_sampling():
    backend, sampler, obs = _make_fixture_1d()
    with pytest.raises(ProtocolDescriptionError, match="labeler"):
        run_shot(_shot_key(0), sampler, backend, _make_stop_1d(), obs, None,
                 labeler=lambda o: "A")


def test_k9_spec_labeler_description_and_batch_up_front_refusal(tmp_path):
    backend, sampler, obs = _make_fixture_1d()
    stop = _make_stop_1d()
    base = protocol_hash(stop, obs, sampler)
    assert protocol_hash(stop, obs, sampler, None) == base  # no labeler: hash unchanged
    lab = SpecLabeler(_label_sign, "sign")
    assert shot_mod.protocol_description(stop, obs, sampler, lab)["labeler"] == {"spec": "sign"}
    assert protocol_hash(stop, obs, sampler, lab) != base
    assert protocol_hash(stop, obs, sampler, lab) == protocol_hash(
        stop, obs, sampler, SpecLabeler(lambda o: "x", "sign")  # the spec is the promise
    )
    rec = run_shot(_shot_key(0), sampler, backend, stop, obs, None, labeler=lab)
    assert rec.protocol_hash == protocol_hash(stop, obs, sampler, lab)
    assert rec.final_state_label in ("pos", "neg")
    # an opaque labeler in a run_shot partial is refused before anything runs
    store = Store(tmp_path / "s.sqlite")
    with pytest.raises(ProtocolDescriptionError, match="labeler"):
        run_batch([_shot_key(0)], _make_shot_fn(labeler=_label_sign), store)
    assert sum(1 for _ in store.iter()) == 0


# --- Fix wave 2, package L3 (fixreview-p1 N-I4, m-2, m-3, m-6, m-8; int2-m3) --

import cytherea.store as store_mod  # noqa: E402
from cytherea.store import RecordSummary  # noqa: E402


def test_n_i4_resume_guard_and_skip_use_sql_summaries_not_payloads(tmp_path, monkeypatch):
    keys = [_shot_key(shot_id=i) for i in range(4)]
    store = Store(tmp_path / "s.sqlite")
    fn = _make_shot_fn()
    first = run_batch(keys, fn, store)
    monkeypatch.setattr(store_mod, "_record_from_json", lambda text: pytest.fail("decoded a payload"))
    again = run_batch(keys, fn, store, return_records=False)
    assert again == [
        RecordSummary(r.key_digest, r.kind, r.stop_reason, r.code_version, r.physics_config_hash,
                      r.protocol_hash)
        for r in first
    ]
    # the guard still refuses a changed physics config from summaries alone
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_batch(keys, _make_shot_fn(gamma=0.06), store, return_records=False)


def test_n_i4_return_records_false_returns_summaries_for_fresh_shots(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(3)]
    store = Store(tmp_path / "s.sqlite")
    out = run_batch(keys, _MixedShotFn(), store, return_records=False)
    assert [type(r).__name__ for r in out] == ["RecordSummary", "ShotFailure", "RecordSummary"]
    assert out[0] == store.summaries_many([out[0].key_digest])[out[0].key_digest]


def test_m2_allow_code_change_is_separate_from_allow_config_change(tmp_path):
    store = _store_with_code_version(tmp_path, "0.1.0+src.000000000000")
    keys = [_shot_key(0), _shot_key(1)]
    out = run_batch(keys, _make_shot_fn(), store, allow_code_change=True)
    assert out[0].code_version == "0.1.0+src.000000000000"
    # allow_code_change does not cover a physics change
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_batch(keys, _make_shot_fn(gamma=0.06), store, allow_code_change=True)


def test_m3_retry_failures_can_target_failure_kinds(tmp_path):
    keys = [_shot_key(shot_id=i) for i in range(3)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys, _MixedShotFn(boom=True), store, on_error="record")  # 1: ic_rejected, 2: error
    fn = _MixedShotFn(boom=False)
    out = run_batch(keys, fn, store, retry_failures={"error"})
    assert fn.calls == [2]  # the transient error is retried, the IC rejection is not
    assert isinstance(out[1], ShotFailure) and out[1].from_store
    assert isinstance(out[2], ShotRecord)
    with pytest.raises(ValueError, match="retry_failures"):
        run_batch(keys, fn, store, retry_failures={"bogus"})


def test_m6_partial_run_shot_with_a_store_is_rejected_up_front(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    fn = functools.partial(_make_shot_fn(), store=store)
    with pytest.raises(ValueError, match="store"):
        run_batch([_shot_key(0)], fn, store)
    assert not list(store.iter())


class _HasCallableAttr:
    def __init__(self):
        self.tol = 1e-5
        self.project = lambda x: x


@pytest.mark.parametrize("value,match", [
    (_HasCallableAttr(), "callable"),
    ({1: "a"}, "key"),
])
def test_m8_nested_descriptions_never_drop_or_collide(value, match):
    with pytest.raises(ProtocolDescriptionError, match=match):
        shot_mod._describe(value, "sampler.constraints")


def test_int2_m3_pool_content_hash_is_computed_once_and_frames_are_frozen(monkeypatch):
    import cytherea.ic.frames as frames_mod

    _backend, sampler, _obs = _make_fixture_1d()
    calls = []
    real = frames_mod.frames_sha256
    monkeypatch.setattr(frames_mod, "frames_sha256", lambda fr: calls.append(1) or real(fr))
    first = shot_mod._describe_pool(sampler.pool)
    assert shot_mod._describe_pool(sampler.pool) == first
    assert len(calls) <= 1
    frame = sampler.pool.frames[0]
    with pytest.raises(ValueError):
        frame.coordinates[0] = 1.0  # read-only: the cached hash cannot go stale
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.weight = 2.0
