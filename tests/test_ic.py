"""Tests for cytherea.ic: ensemble frames, the weighted pool, and the IC
validity gate (Task 5, plus the full-review fix package P2).

Covers the brief's test table (task-5-brief.md, 5.1-5.7) plus the
controller decisions layered on top of it:
  - low-dof skip for the instantaneous-temperature check (must not bias
    the MB distribution for small systems -- and must never *reject* on
    account of being skipped);
  - `remove_com_momentum` interacting with non-(n_atoms, 3) coordinate
    shapes (must raise, not silently misbehave) and with a single-particle
    system (COM removal legitimately zeroes v -- not an error, but worth
    pinning down so nobody "fixes" it later).

and the locked fix-wave contracts (fixplan.md):
  - K1: the shot clock starts at 0; the frame's own time is provenance
    (`meta["frame_time"]`);
  - K2: `InitialState.meta` carries exactly the documented keys;
  - K3: `ShotKey.frame_id >= 0` selects that frame, `-1` draws by weight;
    coordinate-level failures raise `ICRejectedError` (never a silent frame
    switch), only velocity-level failures redraw velocities;
and the review-D findings (C1, I1, I2, I4, I5, I8, I9 surviving mutations,
Minors). Each regression test names the finding it covers.

Test 5.3 in particular is the regression test named in the task: the 1996
VENUS/h2oleps bug where `NaN >= tol` is False, so a NaN initial condition
sailed through a numeric gate unnoticed. The gate here must never return
`ok=True` for it.

New-API names are accessed through the module (`S.<name>`) rather than
imported at the top, so each regression test fails on its own (and for its
own reason) when run against the pre-fix code.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

import cytherea.ic.sampler as S
from cytherea.backends.analytic import AnalyticBackend, DoubleWell1D, LJCluster
from cytherea.backends.base import MDState
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import (
    EnsembleFrameSampler,
    ICRejectedError,
    InitialState,
)
from cytherea.keys import ShotKey


def _lj_frame(frame_id, coords, weight=1.0, source_id="s", temperature=1.0, time=0.0,
              box=None, topology_ref="toy"):
    return EnsembleFrame(
        coordinates=np.asarray(coords, dtype=float),
        box=None if box is None else np.asarray(box, dtype=float),
        topology_ref=topology_ref,
        temperature=temperature,
        weight=weight,
        source_id=source_id,
        frame_id=frame_id,
        time=time,
    )


def _lj_backend(n_atoms, kT=1.0):
    return AnalyticBackend(
        LJCluster(n_atoms), integrator="baoab", dt=0.001, kT=kT, gamma=0.1
    )


class _FakeBackend:
    """Energy/forces stand-in: returns a fixed energy and forces from `F`
    (default zeros) at any x. Models a broken ML/QM backend or coincident
    atoms (inf/NaN energy) without needing one."""

    kind = "analytic"

    def __init__(self, E=0.0, F=None):
        self.E = E
        self.F = F

    def energy_forces(self, x, box=None):
        F = np.zeros_like(np.asarray(x, dtype=float)) if self.F is None else self.F(x)
        return self.E, F

    def effective_config(self, cfg=None):
        return {"backend": "fake", "E": self.E}

    def provenance(self, cfg=None):
        return {}


def _well_separated_atoms(n_atoms, spacing=5.0):
    """Coordinates with LJ energy near 0 (atoms far apart) -- safe defaults
    for tests that aren't specifically exercising the energy/min-dist gate.
    """
    return np.array([[i * spacing, 0.0, 0.0] for i in range(n_atoms)])


def _key(shot_id=0, global_seed=1, frame_id=0, stage="ic"):
    return ShotKey(global_seed=global_seed, frame_id=frame_id, shot_id=shot_id, stage=stage)


def _istate(x, v, box=None, frame_id=0):
    return InitialState(
        state=MDState(x=np.asarray(x, float), v=np.asarray(v, float), t=0.0, box=box),
        frame_id=frame_id,
        meta={},
    )


# --- 5.1: weighted frame choice ----------------------------------------


def test_5_1_frame_choice_proportional_to_weight():
    weights = np.array([1.0, 2.0, 7.0])
    # weights come from the frames themselves (EnsembleFrame.weight)
    frames = [
        _lj_frame(0, _well_separated_atoms(2), weight=1.0),
        _lj_frame(1, _well_separated_atoms(2), weight=2.0),
        _lj_frame(2, _well_separated_atoms(2), weight=7.0),
    ]
    pool = EnsembleFramePool(frames)

    n = 100_000
    counts = np.zeros(3)
    rng = np.random.default_rng(12345)  # test-local RNG is fine here: we are
    # testing EnsembleFramePool.choose's *statistics*, not IC reproducibility
    # (that is 5.2, and it uses derive_rng explicitly).
    for _ in range(n):
        f = pool.choose(rng)
        counts[f.frame_id] += 1

    probs = weights / weights.sum()
    expected = probs * n
    # multinomial std dev per cell
    sigma = np.sqrt(n * probs * (1 - probs))
    assert np.all(np.abs(counts - expected) < 3 * sigma)


def test_5_1_choose_state_of_none_and_state_given_raises():
    frames = [_lj_frame(0, _well_separated_atoms(2))]
    pool = EnsembleFramePool(frames)  # no state_of
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pool.choose(rng, state=0)


# --- 5.2: same ShotKey -> bit-identical InitialState --------------------


def test_5_2_same_key_gives_identical_initial_state():
    frames = [_lj_frame(0, _well_separated_atoms(4), weight=1.0)]
    pool = EnsembleFramePool(frames)
    masses = np.ones(4)
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=1.0,
        backend=_lj_backend(4),
        energy_window=(-100.0, 100.0),
        min_pair_dist=0.1,
    )
    key = _key(shot_id=7)
    s1, r1 = sampler.sample(key)
    s2, r2 = sampler.sample(key)

    assert np.array_equal(s1.state.x, s2.state.x)
    assert np.array_equal(s1.state.v, s2.state.v)
    assert s1.state.t == s2.state.t
    assert s1.frame_id == s2.frame_id
    assert r1.ok and r2.ok


def test_mk_golden_ic_bits():
    """M-k: pin the IC bits for one key, so a reordering of the draws (or a
    change of RNG substream labels) cannot go unnoticed. Update deliberately
    (and record it) if the IC stream layout is ever changed on purpose."""
    frames = [_lj_frame(i, _well_separated_atoms(3), weight=w) for i, w in enumerate([1.0, 2.0, 3.0])]
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames),
        masses=np.array([1.0, 2.0, 4.0]),
        kT=1.0,
        backend=_lj_backend(3),
        energy_window=None,
        min_pair_dist=None,
    )
    istate, _ = sampler.sample(_key(shot_id=3, frame_id=-1))
    assert istate.frame_id == GOLDEN_FRAME_ID
    np.testing.assert_array_equal(istate.state.v[0], GOLDEN_V0)
    chosen = [sampler.sample(_key(shot_id=sid, frame_id=-1))[0].frame_id for sid in range(16)]
    assert chosen == GOLDEN_FRAME_IDS


# Pinned 2026-10-01 (fix package P2, numpy PCG64 stream; substreams
# "ic/frame" and "ic/velocities/{k}"). NEP 19 does not promise Generator
# distribution streams across numpy versions, so a numpy upgrade may also
# legitimately change these; record it when it does.
GOLDEN_FRAME_ID = 2
GOLDEN_V0 = np.array([0.40437581514285686, 1.0031585996183472, 0.6119711732623071])
GOLDEN_FRAME_IDS = [1, 2, 2, 2, 1, 0, 2, 2, 2, 1, 1, 1, 1, 0, 1, 2]


# --- 5.3: NaN coordinates -> never ok=True (h2oleps regression) ---------


def test_5_3_nan_coordinates_never_pass_the_gate():
    coords = _well_separated_atoms(3)
    coords[1, 0] = np.nan
    frame = _lj_frame(0, coords)
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(3),
        kT=1.0,
        backend=_lj_backend(3),
        energy_window=(-100.0, 100.0),
        min_pair_dist=0.1,
    )
    istate = InitialState(
        state=MDState(x=coords, v=np.zeros((3, 3)), t=0.0),
        frame_id=0,
        meta={},
    )
    report = sampler.validate(istate)
    assert report.ok is False
    assert "nonfinite_x" in report.reasons

    # Also: NaN must not sneak past even if some other numeric comparison
    # would otherwise be satisfied (NaN >= tol is False, not an error) --
    # exercise the full sample() path with only-NaN frames available.
    with pytest.raises(ICRejectedError) as excinfo:
        sampler.sample(_key(shot_id=1))
    assert "nonfinite_x" in excinfo.value.reasons
    # never, ever ok=True for this frame
    assert all(
        not sampler.validate(
            InitialState(
                state=MDState(x=coords, v=v, t=0.0),
                frame_id=0,
                meta={},
            )
        ).ok
        for v in [np.zeros((3, 3)), np.ones((3, 3)), np.full((3, 3), np.nan)]
    )


# --- 5.4 (K3): energy outside window -> reject with reason, no frame switch


def _bad_good_pool(w_bad=1.0, w_good=1.0):
    # "bad" frame: atoms overlapping heavily -> huge repulsive LJ energy,
    # far outside any sane window, independent of velocity.
    bad = _lj_frame(0, [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], weight=w_bad)
    # "good" frame: well-separated atoms -> near-zero energy, inside window.
    good = _lj_frame(1, _well_separated_atoms(2), weight=w_good)
    return bad, good, EnsembleFramePool([bad, good])


def test_5_4_energy_outside_window_rejects_explicit_frame_without_switching():
    backend = _lj_backend(2)
    bad, good, pool = _bad_good_pool()
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(2),
        kT=1.0,
        backend=backend,
        energy_window=(-10.0, 10.0),
        min_pair_dist=None,
        max_redraws=50,
    )
    e_bad, _ = backend.energy_forces(np.asarray(bad.coordinates))
    e_good, _ = backend.energy_forces(np.asarray(good.coordinates))
    assert not (-10.0 <= e_bad <= 10.0)
    assert -10.0 <= e_good <= 10.0

    # key.frame_id selects the frame (K3): the bad frame is rejected with
    # its reason after ONE attempt -- a coordinate-level failure is
    # deterministic per frame, so redrawing (velocities or frame) is never
    # attempted.
    with pytest.raises(ICRejectedError) as excinfo:
        sampler.sample(_key(shot_id=0, frame_id=0))
    err = excinfo.value
    assert err.reasons == ["energy_window"]
    assert err.frame_id == 0
    assert err.level == "coordinate"
    assert len(err.attempts) == 1
    assert err.attempts[0]["frame_id"] == 0
    assert err.attempts[0]["reasons"] == ["energy_window"]

    istate, report = sampler.sample(_key(shot_id=0, frame_id=1))
    assert report.ok and report.n_redraws == 0
    assert istate.frame_id == 1


def test_k3_weighted_draw_never_switches_frame_on_coordinate_failure():
    """D-I4/I5, A-I8 (p4_provenance.py): a clashing frame carrying 70% of
    the weight used to be silently renormalised away (200/200 shots from the
    other frame, n_redraws>0 with reasons=[]). Now every shot that draws the
    clashing frame raises with its reason, so the excluded weight is
    visible, and accepted shots never involve a frame switch."""
    backend = _lj_backend(2)
    _bad, _good, pool = _bad_good_pool(w_bad=0.7, w_good=0.3)
    sampler = EnsembleFrameSampler(
        pool=pool, masses=np.ones(2), kT=1.0, backend=backend,
        energy_window=(-10.0, 10.0), min_pair_dist=None,
        remove_com_momentum=False, max_redraws=20,
    )
    n = 400
    n_rejected = 0
    for sid in range(n):
        try:
            istate, report = sampler.sample(_key(shot_id=sid, frame_id=-1))
        except ICRejectedError as exc:
            n_rejected += 1
            assert exc.reasons == ["energy_window"]
            assert exc.frame_id == 0
            continue
        assert istate.frame_id == 1
        assert report.n_redraws == 0
        assert report.rejected_attempts == []
    sigma = math.sqrt(n * 0.7 * 0.3)
    assert abs(n_rejected - 0.7 * n) < 4 * sigma


def test_k3_explicit_frame_id_selects_that_frame():
    """D-I5 / A-I8 (p4): `key.frame_id` used to be ignored (0,1,2 -> 1,0,1)."""
    frames = [_lj_frame(10 + i, _well_separated_atoms(2)) for i in range(5)]
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0,
        backend=_lj_backend(2), energy_window=None, min_pair_dist=None,
        remove_com_momentum=False,
    )
    for fid in range(10, 15):
        for sid in range(4):
            istate, _ = sampler.sample(_key(shot_id=sid, frame_id=fid))
            assert istate.frame_id == fid
            assert istate.meta["frame_id"] == fid
            np.testing.assert_array_equal(istate.state.x, frames[fid - 10].coordinates)


def test_k3_unknown_or_invalid_frame_id_raises():
    frames = [_lj_frame(3, _well_separated_atoms(2))]
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0,
        backend=_lj_backend(2), energy_window=None, min_pair_dist=None,
    )
    with pytest.raises(KeyError):
        sampler.sample(_key(frame_id=0))
    with pytest.raises(ValueError):
        sampler.sample(_key(frame_id=-2))


def test_k3_explicit_frame_outside_sampler_state_raises():
    frames = [_lj_frame(0, _well_separated_atoms(2)), _lj_frame(1, _well_separated_atoms(2))]
    pool = EnsembleFramePool(frames, state_of=lambda f: f.frame_id)
    sampler = EnsembleFrameSampler(
        pool=pool, masses=np.ones(2), kT=1.0, backend=_lj_backend(2),
        energy_window=None, min_pair_dist=None, state=1,
    )
    with pytest.raises(ValueError):
        sampler.sample(_key(frame_id=0))
    istate, _ = sampler.sample(_key(frame_id=1))
    assert istate.meta["state"] == 1


def test_k3_weighted_draw_is_proportional_to_weight():
    frames = [_lj_frame(i, _well_separated_atoms(2), weight=w) for i, w in enumerate([1.0, 2.0, 7.0])]
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0,
        backend=_FakeBackend(), energy_window=None, min_pair_dist=None,
    )
    n = 3000
    counts = np.zeros(3)
    for sid in range(n):
        istate, _ = sampler.sample(_key(shot_id=sid, frame_id=-1))
        counts[istate.frame_id] += 1
    p = np.array([0.1, 0.2, 0.7])
    assert np.all(np.abs(counts - n * p) < 4 * np.sqrt(n * p * (1 - p)))


class _NaNFirstVelocities(EnsembleFrameSampler):
    """Velocity draws for attempts < `n_bad` are NaN (a velocity-level
    failure), and records which frame each attempt used."""

    def __init__(self, *a, n_bad=2, **kw):
        super().__init__(*a, **kw)
        self.n_bad = n_bad
        self.calls = 0
        self.draws = []

    def _draw_velocities(self, rng, shape):
        v = super()._draw_velocities(rng, shape)
        self.draws.append(v)
        self.calls += 1
        if self.calls <= self.n_bad:
            return np.full(shape, np.nan)
        return v


def test_k3_velocity_failure_redraws_velocities_only_and_keeps_reasons():
    """K3 + D-I4: a velocity-level failure redraws velocities for the SAME
    frame, and the per-attempt reasons survive into the accepted report."""
    frames = [_lj_frame(0, _well_separated_atoms(2)), _lj_frame(1, _well_separated_atoms(2), time=2.5)]
    sampler = _NaNFirstVelocities(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0,
        backend=_FakeBackend(), energy_window=None, min_pair_dist=None, n_bad=2,
    )
    istate, report = sampler.sample(_key(frame_id=1))
    assert report.ok
    assert istate.frame_id == 1
    assert report.n_redraws == 2
    assert istate.meta["n_redraws"] == 2
    assert report.rejected_attempts == [
        {"attempt": 0, "frame_id": 1, "reasons": ["nonfinite_v"]},
        {"attempt": 1, "frame_id": 1, "reasons": ["nonfinite_v"]},
    ]
    assert report.reasons == []
    # each attempt draws fresh velocities from its own substream (D-I9 M10)
    assert len(sampler.draws) == 3
    assert not np.array_equal(sampler.draws[0], sampler.draws[2])
    np.testing.assert_array_equal(istate.state.v, sampler.draws[2])


def test_5_5b_velocity_exhaustion_counts_every_attempt():
    """D-I9 M9: the attempt count must be exactly max_redraws + 1."""
    frames = [_lj_frame(0, _well_separated_atoms(2))]
    max_redraws = 4
    sampler = _NaNFirstVelocities(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0,
        backend=_FakeBackend(), energy_window=None, min_pair_dist=None,
        max_redraws=max_redraws, n_bad=10**9,
    )
    with pytest.raises(ICRejectedError) as excinfo:
        sampler.sample(_key())
    err = excinfo.value
    assert len(err.reasons) == max_redraws + 1
    assert len(err.attempts) == max_redraws + 1
    assert [a["attempt"] for a in err.attempts] == list(range(max_redraws + 1))
    assert err.level == "velocity"
    assert sampler.calls == max_redraws + 1


def test_ic_rejected_error_survives_pickling():
    """run_batch's worker processes re-raise ICRejectedError in the parent.
    Default exception pickling calls ICRejectedError(message), which parsed
    the message as the reasons list and garbled str(err)."""
    import pickle

    err = ICRejectedError(["energy_window"], attempts=[{"attempt": 0, "frame_id": 3, "reasons": ["energy_window"]}],
                          frame_id=3, level="coordinate")
    back = pickle.loads(pickle.dumps(err))
    assert back.reasons == ["energy_window"]
    assert back.attempts == err.attempts
    assert back.frame_id == 3 and back.level == "coordinate"
    assert str(back) == str(err)


# --- 5.5: all frames invalid -> ICRejectedError with all reasons --------


def test_5_5_all_frames_invalid_raises_with_reasons():
    coords1 = _well_separated_atoms(2)
    coords1[0, 0] = np.nan
    coords2 = np.array(coords1)  # also NaN
    pool = EnsembleFramePool(
        [_lj_frame(0, coords1, weight=1.0), _lj_frame(1, coords2, weight=1.0)]
    )
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(2),
        kT=1.0,
        backend=_lj_backend(2),
        energy_window=(-10.0, 10.0),
        min_pair_dist=0.1,
        max_redraws=5,
    )
    for fid in (-1, 0, 1):
        with pytest.raises(ICRejectedError) as excinfo:
            sampler.sample(_key(shot_id=0, frame_id=fid))
        assert excinfo.value.reasons  # non-empty
        assert all(r == "nonfinite_x" for r in excinfo.value.reasons)
        assert "nonfinite_x" in str(excinfo.value)


# --- K1 / C1: the shot clock starts at 0 ------------------------------------


def test_k1_frame_time_is_provenance_not_the_shot_clock():
    """D-C1 / A-C1 (p1_frame_time.py): frame.time used to become MDState.t."""
    frame = _lj_frame(0, _well_separated_atoms(2), time=500.0)
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool([frame]), masses=np.ones(2), kT=1.0,
        backend=_lj_backend(2), energy_window=None, min_pair_dist=None,
    )
    istate, _ = sampler.sample(_key())
    assert istate.state.t == 0.0
    assert istate.meta["frame_time"] == 500.0


def test_c1_run_shot_from_a_late_frame_runs_the_full_lag():
    """D-C1 acceptance: a frame with time=500 through run_shot + FixedLag(tau)
    must run tau/dt_obs chunks, not stop at the first observation."""
    from cytherea.engine.shot import ObsSpec, run_shot
    from cytherea.observe.events import FixedLag

    backend = AnalyticBackend(DoubleWell1D(barrier=2.0, x0=1.0), integrator="baoab",
                              dt=0.01, kT=1.0, gamma=0.05, mass=1.0)
    frame = EnsembleFrame(coordinates=np.array([0.3]), box=None, topology_ref="toy1d",
                          temperature=1.0, weight=1.0, source_id="s", frame_id=0, time=500.0)
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool([frame]), masses=np.array([1.0]), kT=1.0, backend=backend,
        energy_window=None, min_pair_dist=None, remove_com_momentum=False,
    )
    obs = ObsSpec(fns={"x": lambda s: float(s.x[0])}, dt_obs=0.05, store_stride=1)
    rec = run_shot(_key(frame_id=0), sampler, backend, FixedLag(tau=0.2), obs, None)
    assert rec.stop_reason == "fixed_lag"
    assert len(rec.observables["t"]) == 5  # t = 0, 0.05, ..., 0.2
    assert rec.observables["t"][0] == 0.0
    assert rec.event_time == pytest.approx(0.2)


# --- K2: meta keys --------------------------------------------------------


def test_k2_meta_has_exactly_the_contract_keys():
    frames = [_lj_frame(4, _well_separated_atoms(2), weight=0.25, source_id="eq7",
                        time=12.5, topology_ref="lj2")]
    pool = EnsembleFramePool(frames, state_of=lambda f: "A")
    sampler = EnsembleFrameSampler(
        pool=pool, masses=np.ones(2), kT=1.0, backend=_lj_backend(2),
        energy_window=None, min_pair_dist=None,
    )
    istate, report = sampler.sample(_key(frame_id=4))
    assert set(istate.meta) == {
        "frame_id", "frame_time", "frame_weight", "source_id", "topology_ref", "state", "n_redraws",
    }
    assert istate.meta == {
        "frame_id": 4, "frame_time": 12.5, "frame_weight": 0.25, "source_id": "eq7",
        "topology_ref": "lj2", "state": "A", "n_redraws": 0,
    }
    # no state partition -> state is None
    sampler2 = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames), masses=np.ones(2), kT=1.0, backend=_lj_backend(2),
        energy_window=None, min_pair_dist=None,
    )
    assert sampler2.sample(_key(frame_id=4))[0].meta["state"] is None


# --- I1: non-finite energy / forces are always rejected --------------------


def _one_frame_sampler(backend, n=12, energy_window=None, **kw):
    x = np.array([[1.0 * i, 0.0, 0.0] for i in range(n)])
    fr = _lj_frame(0, x)
    return x, EnsembleFrameSampler(EnsembleFramePool([fr]), np.ones(n), 1.0, backend,
                                   energy_window=energy_window, min_pair_dist=None, **kw)


@pytest.mark.parametrize("E", [float("nan"), float("inf"), -float("inf")])
def test_i1_nonfinite_energy_rejected_without_energy_window(E):
    """D-I1(a) (p3 (a)): with energy_window=None the energy was never
    evaluated, so a NaN/inf-energy IC was accepted."""
    x, sampler = _one_frame_sampler(_FakeBackend(E=E))
    rep = sampler.validate(_istate(x, np.zeros_like(x)))
    assert rep.ok is False
    assert "nonfinite_energy" in rep.reasons
    with pytest.raises(ICRejectedError) as excinfo:
        sampler.sample(_key())
    assert excinfo.value.reasons == ["nonfinite_energy"]
    assert excinfo.value.level == "coordinate"


def test_i1_nonfinite_energy_rejected_with_energy_window():
    """D-I1(c): M3 (`E < lo or E > hi` lets NaN through) must die."""
    x, sampler = _one_frame_sampler(_FakeBackend(E=float("nan")), energy_window=(-1.0, 1.0))
    rep = sampler.validate(_istate(x, np.zeros_like(x)))
    assert rep.ok is False
    assert "nonfinite_energy" in rep.reasons


def test_i1_nonfinite_forces_rejected():
    """D-I1(b) (p3 (b)): finite E inside the window with NaN forces was
    accepted (forces were discarded)."""
    def nan_force(x):
        F = np.zeros_like(x)
        F[3, 1] = np.nan
        return F
    for window in (None, (-1.0, 1.0)):
        x, sampler = _one_frame_sampler(_FakeBackend(E=0.0, F=nan_force), energy_window=window)
        rep = sampler.validate(_istate(x, np.zeros_like(x)))
        assert rep.ok is False
        assert "nonfinite_energy" in rep.reasons


def test_i1_energy_recorded_in_checks_even_without_window():
    x, sampler = _one_frame_sampler(_FakeBackend(E=-3.5))
    rep = sampler.validate(_istate(x, np.zeros_like(x)))
    assert rep.ok
    assert rep.checks["energy"] == -3.5


# --- I2: constraints --------------------------------------------------------


def _diatomics(nmol, bond=1.0, spacing=3.0):
    N = 2 * nmol
    x = np.zeros((N, 3))
    x[0::2, 0] = np.arange(nmol) * spacing
    x[1::2, 0] = x[0::2, 0] + bond
    pairs = np.stack([np.arange(0, N, 2), np.arange(1, N, 2)], axis=1)
    return x, pairs


def test_i2_old_style_constraints_callable_is_rejected():
    def old_callable(x, v):
        return x, v, 0.0
    x = _well_separated_atoms(2)
    with pytest.raises(TypeError):
        EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(2), 1.0,
                             _FakeBackend(), None, None, constraints=old_callable)


def test_i2_validate_measures_the_residual_of_the_given_state():
    """D-I2(a) (p3 (c)): validate() used to trust the residual of the
    *projected* state, so a violating state passed (bond 3.0 vs 1.0)."""
    x, pairs = _diatomics(1)
    masses = np.ones(2)
    cons = S.DistanceConstraints(pairs, [1.0], masses)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), masses, 1.0,
                                   _FakeBackend(), None, None, constraints=cons)
    bad = x.copy()
    bad[1] = [3.0, 0.0, 0.0]
    rep = sampler.validate(_istate(bad, np.zeros((2, 3))))
    assert rep.ok is False
    assert "constraint_residual" in rep.reasons
    assert rep.checks["constraint_residual"] == pytest.approx(2.0)

    # velocity along the rigid bond violates the velocity constraint
    v = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    rep = sampler.validate(_istate(x, v))
    assert "constraint_residual" in rep.reasons

    # a satisfying state (velocity perpendicular to the bond) passes
    v = np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    rep = sampler.validate(_istate(x, v))
    assert rep.ok, rep.reasons


def test_i2_constraint_tolerance_is_relative_and_matches_openmm():
    """D-I2(c): the hard-coded absolute 1e-8 made any OpenMM-backed state
    (tolerance 1e-5, relative) fail."""
    assert S.CONSTRAINT_TOLERANCE == 1e-5
    ob = pytest.importorskip("cytherea.backends.openmm_backend")
    assert S.CONSTRAINT_TOLERANCE == ob.CONSTRAINT_TOLERANCE

    x, pairs = _diatomics(1, bond=2.0)
    cons = S.DistanceConstraints(pairs, [2.0], np.ones(2))
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(2), 1.0,
                                   _FakeBackend(), None, None, constraints=cons)
    assert sampler.constraint_tolerance == S.CONSTRAINT_TOLERANCE
    for rel_err, ok in [(5e-6, True), (5e-5, False)]:
        xe = x.copy()
        xe[1, 0] = 2.0 * (1.0 + rel_err)
        rep = sampler.validate(_istate(xe, np.zeros((2, 3))))
        assert rep.ok is ok, (rel_err, rep.reasons, rep.checks)


def test_i2_distance_constraints_projection():
    """The reference `DistanceConstraints` projects coupled constraints (a
    rigid water-like triangle) to 1e-10, conserves momentum, and makes
    velocities satisfy the velocity constraints."""
    rng = np.random.default_rng(0)
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.03, 0.09, 0.0],
                  [2.0, 0.0, 0.0], [2.1, 0.02, 0.0]])
    masses = np.array([16.0, 1.0, 1.0, 12.0, 1.0])
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    dists = [0.09572, 0.09572, 0.15139, 0.1090]
    cons = S.DistanceConstraints(pairs, dists, masses)
    assert cons.n_constraints == 4
    v = rng.normal(size=x.shape) / np.sqrt(masses)[:, None]
    p0 = (masses[:, None] * v).sum(axis=0)
    xp, vp = cons.project(x, v)
    for (i, j), d in zip(pairs, dists):
        assert abs(np.linalg.norm(xp[j] - xp[i]) / d - 1) < 1e-10
        rij = xp[j] - xp[i]
        assert abs(np.dot(rij, vp[j] - vp[i])) / (np.linalg.norm(rij) * np.linalg.norm(vp[j] - vp[i])) < 1e-10
    assert np.allclose((masses[:, None] * vp).sum(axis=0), p0, atol=1e-12)
    assert np.allclose((masses[:, None] * xp).sum(axis=0), (masses[:, None] * x).sum(axis=0), atol=1e-12)
    assert cons.residual(xp, vp) < 1e-10
    # idempotent on a satisfying state
    xq, vq = cons.project(xp, vp)
    assert np.allclose(xq, xp, atol=1e-12) and np.allclose(vq, vp, atol=1e-12)


def test_i2_dof_subtracts_constraints_and_rigid_diatomics_pass():
    """D-I2(b) (p3 (d)): with correct MB-projected velocities, 1000 rigid
    diatomics were rejected on every attempt (T_inst/kT = 0.83 against the
    band) because dof ignored the constraints."""
    nmol = 1000
    x, pairs = _diatomics(nmol)
    N = 2 * nmol
    masses = np.where(np.arange(N) % 2 == 0, 12.0, 1.0)
    cons = S.DistanceConstraints(pairs, np.ones(nmol), masses)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), masses, 1.0,
                                   _FakeBackend(), None, None, max_redraws=0, constraints=cons)
    istate, rep = sampler.sample(_key())
    assert rep.ok, rep.reasons
    assert rep.checks["temperature_dof"] == 3 * N - nmol - 3
    assert cons.residual(istate.state.x, istate.state.v) < 1e-9
    # COM momentum stays removed through the projection
    assert np.max(np.abs((masses[:, None] * istate.state.v).sum(axis=0))) < 1e-9


def test_i2_constrained_mb_draws_have_the_right_mean_temperature():
    """The projected MB draw has <2 KE / (3N - Nc - 3)> = kT: checks both
    the projection (mass-weighted) and the dof formula physically."""
    nmol = 100
    x, pairs = _diatomics(nmol)
    N = 2 * nmol
    masses = np.where(np.arange(N) % 2 == 0, 16.0, 1.0)
    cons = S.DistanceConstraints(pairs, np.ones(nmol), masses)
    kT = 2.5
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), masses, kT,
                                   _FakeBackend(), None, None, constraints=cons)
    dof = 3 * N - nmol - 3
    temps = []
    for sid in range(200):
        istate, rep = sampler.sample(_key(shot_id=sid))
        temps.append(rep.checks["temperature"])
    mean_T = float(np.mean(temps))
    sigma = kT * math.sqrt(2.0 / dof) / math.sqrt(len(temps))
    assert abs(mean_T - kT) < 4 * sigma


def test_i2_constraints_are_applied_in_sample():
    """D-I9 M5: a frame slightly off the constraint manifold must be
    projected by sample() (otherwise the gate rejects it)."""
    x, pairs = _diatomics(1)
    x[1, 0] += 1e-3
    cons = S.DistanceConstraints(pairs, [1.0], np.ones(2))
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(2), 1.0,
                                   _FakeBackend(), None, None, constraints=cons)
    istate, rep = sampler.sample(_key())
    assert rep.ok
    assert abs(np.linalg.norm(istate.state.x[1] - istate.state.x[0]) - 1.0) < 1e-10
    assert rep.checks["constraint_residual_input"] == pytest.approx(1e-3, rel=1e-6)


# --- I4: provenance of rejections -------------------------------------------


def test_i4_prevalidate_pool_reports_rejected_weight():
    backend = _lj_backend(2)
    _bad, _good, pool = _bad_good_pool(w_bad=0.7, w_good=0.3)
    sampler = EnsembleFrameSampler(pool=pool, masses=np.ones(2), kT=1.0, backend=backend,
                                   energy_window=(-10.0, 10.0), min_pair_dist=None)
    pv = sampler.prevalidate_pool()
    assert pv.n_frames == 2
    assert pv.n_rejected == 1
    assert pv.rejected_weight_fraction == pytest.approx(0.7)
    assert pv.rejected == [{"frame_id": 0, "weight": 0.7, "reasons": ["energy_window"]}]
    assert pv.to_dict()["rejected_weight_fraction"] == pytest.approx(0.7)


# --- I5: topology consistency ------------------------------------------------


def test_i5_topology_ref_must_be_consistent():
    x = _well_separated_atoms(2)
    mixed = EnsembleFramePool([_lj_frame(0, x, topology_ref="L-ab42"), _lj_frame(1, x, topology_ref="D-ab42")])
    with pytest.raises(ValueError):
        EnsembleFrameSampler(mixed, np.ones(2), 1.0, _FakeBackend(), None, None)
    pool = EnsembleFramePool([_lj_frame(0, x, topology_ref="L-ab42")])
    with pytest.raises(ValueError):
        EnsembleFrameSampler(pool, np.ones(2), 1.0, _FakeBackend(), None, None, topology_ref="D-ab42")
    EnsembleFrameSampler(pool, np.ones(2), 1.0, _FakeBackend(), None, None, topology_ref="L-ab42")


# --- I8: min_pair_dist: O(N log N), periodic ---------------------------------


def test_i8_min_pair_dist_uses_minimum_image():
    """D-I8: raw distances overestimate the minimum-image distance, so a
    clash across the periodic boundary was accepted."""
    L = 3.0
    box = np.diag([L, L, L])
    x = np.array([[0.05, 1.0, 1.0], [2.95, 1.0, 1.0], [1.5, 2.0, 1.0]])
    fr = _lj_frame(0, x, box=box)
    sampler = EnsembleFrameSampler(EnsembleFramePool([fr]), np.ones(3), 1.0, _FakeBackend(),
                                   None, min_pair_dist=0.5)
    rep = sampler.validate(_istate(x, np.zeros_like(x), box=box))
    assert rep.ok is False
    assert "min_pair_dist" in rep.reasons
    assert rep.checks["min_pair_dist"] == pytest.approx(0.1)

    # coordinates far outside the primary cell wrap correctly
    xs = x + np.array([[-3 * L, 0, 0], [2 * L, L, -L], [0, 0, 5 * L]])
    rep = sampler.validate(_istate(xs, np.zeros_like(x), box=box))
    assert rep.checks["min_pair_dist"] == pytest.approx(0.1)


def test_i8_min_pair_dist_matches_brute_force():
    rng = np.random.default_rng(3)
    L = np.array([2.0, 2.5, 3.0])
    for box in (None, np.diag(L)):
        for _ in range(5):
            x = rng.uniform(-1.0, 4.0, size=(150, 3))
            d = x[:, None, :] - x[None, :, :]
            if box is not None:
                d -= L * np.round(d / L)
            dist = np.sqrt((d ** 2).sum(-1))
            brute = dist[np.triu_indices(len(x), 1)].min()
            fr = _lj_frame(0, x, box=box)
            sampler = EnsembleFrameSampler(EnsembleFramePool([fr]), np.ones(150), 1.0,
                                           _FakeBackend(), None, min_pair_dist=1e-6)
            rep = sampler.validate(_istate(x, np.zeros_like(x), box=box))
            assert rep.checks["min_pair_dist"] == pytest.approx(brute, rel=1e-12)


def test_i8_min_pair_dist_rejects_triclinic_and_oversized_threshold():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    tric = np.array([[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    with pytest.raises(ValueError):
        EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x, box=tric)]), np.ones(2), 1.0,
                             _FakeBackend(), None, min_pair_dist=0.1)
    ortho = np.diag([3.0, 3.0, 3.0])
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x, box=ortho)]), np.ones(2), 1.0,
                                   _FakeBackend(), None, min_pair_dist=0.1)
    with pytest.raises(ValueError):
        sampler.validate(_istate(x, np.zeros_like(x), box=tric))
    with pytest.raises(ValueError):
        EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x, box=ortho)]), np.ones(2), 1.0,
                             _FakeBackend(), None, min_pair_dist=1.6)


def test_i8_min_pair_dist_memory_is_not_quadratic():
    """D-I8 (p6_minpair_mem.py): the dense check needed ~55 N^2 bytes
    (N=5000 -> ~1.4 GB). A KD-tree needs O(N)."""
    N = 5000
    x = np.random.default_rng(0).uniform(0, 10, size=(N, 3))
    fr = _lj_frame(0, x)
    sampler = EnsembleFrameSampler(EnsembleFramePool([fr]), np.ones(N), 1.0, _FakeBackend(),
                                   None, 1e-6, remove_com_momentum=False)
    ist = _istate(x, np.zeros_like(x))
    tracemalloc.start()
    try:
        sampler.validate(ist)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 50e6, peak


# --- invalid box (M-c) ---------------------------------------------------------


@pytest.mark.parametrize("box", [
    np.diag([np.nan, 3.0, 3.0]),
    np.diag([3.0, -3.0, 3.0]),
    np.zeros((3, 3)),
])
def test_mc_invalid_box_is_rejected(box):
    x = _well_separated_atoms(2)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(2), 1.0,
                                   _FakeBackend(), None, None)
    rep = sampler.validate(_istate(x, np.zeros_like(x), box=box))
    assert rep.ok is False
    assert "invalid_box" in rep.reasons


def test_mc_velocity_shape_mismatch_raises():
    x = _well_separated_atoms(4)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(4), 1.0,
                                   _FakeBackend(), None, None)
    with pytest.raises(ValueError):
        sampler.validate(_istate(x, np.zeros((1, 3))))


# --- M-a, M-b: constructor validation ----------------------------------------


def test_ma_frame_temperature_checked_against_kT_when_kB_given():
    kB = S.KB_KJ_PER_MOL_K
    kT = kB * 300.0
    x = _well_separated_atoms(2)
    pool = EnsembleFramePool([_lj_frame(0, x, temperature=300.0)])
    s = EnsembleFrameSampler(pool, np.ones(2), kT, _FakeBackend(), None, None, boltzmann_constant=kB)
    _, rep = s.sample(_key())
    assert rep.checks["frame_temperature"] == 300.0
    # kT given in K by mistake (Review Focus 4: unit mix-up)
    with pytest.raises(ValueError):
        EnsembleFrameSampler(pool, np.ones(2), 300.0, _FakeBackend(), None, None, boltzmann_constant=kB)
    pool310 = EnsembleFramePool([_lj_frame(0, x, temperature=310.0)])
    with pytest.raises(ValueError):
        EnsembleFrameSampler(pool310, np.ones(2), kT, _FakeBackend(), None, None, boltzmann_constant=kB)


@pytest.mark.parametrize("kw", [
    {"kT": 0.0}, {"kT": -1.0}, {"kT": float("nan")},
    {"masses": np.array([1.0, 0.0])}, {"masses": np.array([1.0, np.nan])},
    {"masses": np.array([1.0, -2.0])}, {"max_redraws": -1},
])
def test_mb_constructor_validates_parameters(kw):
    x = _well_separated_atoms(2)
    args = dict(pool=EnsembleFramePool([_lj_frame(0, x)]), masses=np.ones(2), kT=1.0,
                backend=_FakeBackend(), energy_window=None, min_pair_dist=None)
    args.update(kw)
    with pytest.raises(ValueError):
        EnsembleFrameSampler(**args)


# --- M-d, M-e: pool ------------------------------------------------------------


def test_md_state_of_is_evaluated_once_per_frame():
    calls = []

    def state_of(f):
        calls.append(f.frame_id)
        return f.frame_id % 2

    frames = [_lj_frame(i, _well_separated_atoms(2)) for i in range(6)]
    pool = EnsembleFramePool(frames, state_of=state_of)
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert pool.choose(rng, state=1).frame_id % 2 == 1
        pool.choose(rng, state=0)
    assert sorted(calls) == list(range(6))


def test_md_zero_weight_state_raises_named_error():
    frames = [_lj_frame(0, _well_separated_atoms(2), weight=1.0), _lj_frame(1, _well_separated_atoms(2), weight=0.0)]
    pool = EnsembleFramePool(frames, state_of=lambda f: f.frame_id)
    with pytest.raises(ValueError, match="zero weight"):
        pool.choose(np.random.default_rng(0), state=1)


def test_me_pool_rejects_duplicates_and_is_immutable():
    x = _well_separated_atoms(2)
    with pytest.raises(ValueError):
        EnsembleFramePool([_lj_frame(0, x), _lj_frame(0, x)])
    with pytest.raises(ValueError):
        EnsembleFramePool([_lj_frame(-1, x)])
    pool = EnsembleFramePool([_lj_frame(0, x), _lj_frame(1, x)])
    with pytest.raises(AttributeError):
        pool.frames.append(_lj_frame(2, x))
    with pytest.raises(ValueError):
        pool.weights[0] = 5.0
    assert pool.get(1).frame_id == 1


@pytest.mark.parametrize("w0,w1", [(1e-320, 3e-320), (0.5e308, 1.5e308)])
def test_me_extreme_weights_keep_their_ratio(w0, w1):
    """M-e: weights are normalised by max(w) first, so subnormal weights
    keep their ratio and huge weights (whose sum overflows to inf) still
    give valid probabilities."""
    x = _well_separated_atoms(2)
    pool = EnsembleFramePool([_lj_frame(0, x, weight=w0), _lj_frame(1, x, weight=w1)])
    assert pool.weight_fraction([1]) == pytest.approx(0.75, rel=1e-3)
    rng = np.random.default_rng(1)
    n = 20000
    c1 = sum(pool.choose(rng).frame_id == 1 for _ in range(n))
    assert abs(c1 - 0.75 * n) < 4 * math.sqrt(n * 0.75 * 0.25)


# --- M-g: reasons vocabulary --------------------------------------------------


def test_mg_reasons_vocabulary_is_partitioned_and_used():
    assert S.REASONS == S.COORDINATE_REASONS | S.VELOCITY_REASONS
    assert not (S.COORDINATE_REASONS & S.VELOCITY_REASONS)
    assert S.COORDINATE_REASONS == {
        "nonfinite_x", "invalid_box", "nonfinite_energy", "energy_window",
        "min_pair_dist", "constraint_residual", "constraint_input", "energy_error",
    }
    assert S.VELOCITY_REASONS == {"nonfinite_v", "temperature"}
    x = _well_separated_atoms(12)
    x[0, 0] = np.nan
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, _well_separated_atoms(12))]),
                                   np.ones(12), 1.0, _FakeBackend(), (-1.0, 1.0), 0.1)
    rep = sampler.validate(_istate(x, np.full_like(x, np.inf)))
    assert set(rep.reasons) <= S.REASONS


# --- 5.6: MB velocities + COM momentum removal ---------------------------


def test_5_6_mb_velocity_statistics_and_com_momentum():
    # Two separate checks per the brief, deliberately using two draw modes:
    #
    #   (a) <1/2 m v^2> == 1/2 kT +- 3 sigma per *raw* MB degree of freedom.
    #       This must use remove_com_momentum=False: removing the COM
    #       momentum from an (n_atoms, 3) draw genuinely shifts the mean
    #       per-component KE below kT/2 (3 fewer independent dof than
    #       components: true equipartition is
    #       <KE_total> = ((3n-3)/2) kT, not (3n/2) kT), which is correct
    #       physics, not something this check should be fooled by.
    #   (b) after COM removal, total momentum per draw is ~0 to machine
    #       precision (brief: < 1e-12) -- this one needs
    #       remove_com_momentum=True.
    n_atoms = 50
    kT = 2.0
    masses = np.ones(n_atoms)
    frame = _lj_frame(0, _well_separated_atoms(n_atoms), weight=1.0)
    pool = EnsembleFramePool([frame])
    shape = frame.coordinates.shape

    sampler_raw = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=kT,
        backend=_lj_backend(n_atoms, kT=kT),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=False,
    )
    sampler_com = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=kT,
        backend=_lj_backend(n_atoms, kT=kT),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=True,
    )

    n_samples = 10_000
    ke_per_dof = []
    max_com = 0.0
    for i in range(n_samples):
        rng = np.random.default_rng(1000 + i)  # statistics only, not an IC key
        v_raw = sampler_raw._draw_velocities(rng, shape)
        ke = 0.5 * masses[:, None] * v_raw**2
        ke_per_dof.append(ke.ravel())

        rng2 = np.random.default_rng(2000 + i)
        v_com = sampler_com._draw_velocities(rng2, shape)
        total_p = np.sum(masses[:, None] * v_com, axis=0)
        max_com = max(max_com, float(np.max(np.abs(total_p))))

    ke_per_dof = np.concatenate(ke_per_dof)
    mean_ke = ke_per_dof.mean()
    # single-dof KE = 0.5*kT*z^2, z~N(0,1): mean=kT/2, var=kT^2/2
    sigma_mean = math.sqrt((kT**2 / 2.0) / ke_per_dof.size)
    assert abs(mean_ke - 0.5 * kT) < 3 * sigma_mean

    # after COM removal, total momentum per draw must be ~0 to machine
    # precision (brief: < 1e-12)
    assert max_com < 1e-12


def test_5_6b_mb_statistics_with_unequal_masses():
    """D-I9 M1/M2: every 5.6 check used masses=1, so sigma=sqrt(kT*m) and an
    unweighted COM removal both survived test_ic."""
    masses = np.repeat(np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 8)
    n_atoms = masses.size
    kT = 1.5
    frame = _lj_frame(0, _well_separated_atoms(n_atoms))
    pool = EnsembleFramePool([frame])
    raw = EnsembleFrameSampler(pool, masses, kT, _FakeBackend(), None, None, remove_com_momentum=False)
    com = EnsembleFrameSampler(pool, masses, kT, _FakeBackend(), None, None, remove_com_momentum=True)
    n = 4000
    vs = np.stack([raw._draw_velocities(np.random.default_rng(10 + i), (n_atoms, 3)) for i in range(n)])
    # per-atom variance of each component must be kT/m
    var = vs.var(axis=(0, 2))
    rel_sigma = math.sqrt(2.0 / (3 * n))
    assert np.all(np.abs(var * masses / kT - 1.0) < 5 * rel_sigma)
    for i in range(50):
        v = com._draw_velocities(np.random.default_rng(99 + i), (n_atoms, 3))
        p = (masses[:, None] * v).sum(axis=0)
        assert np.max(np.abs(p)) < 1e-12


def test_single_particle_com_removal_zeroes_velocity():
    # Controller decision: COM removal is only meaningful for (n_atoms, 3)
    # coordinates, but for n_atoms == 1 it legitimately sets v == 0 exactly
    # (not an error -- just a documented quirk of a single-particle toy).
    frame = _lj_frame(0, [[0.0, 0.0, 0.0]], weight=1.0)
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(1),
        kT=1.0,
        backend=_lj_backend(1),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=True,
    )
    rng = np.random.default_rng(0)
    v = sampler._draw_velocities(rng, frame.coordinates.shape)
    assert np.allclose(v, 0.0)


# --- 5.7: state-restricted frame choice ----------------------------------


def test_5_7_state_restricted_choice():
    frames = [
        _lj_frame(0, _well_separated_atoms(2), weight=1.0),
        _lj_frame(1, _well_separated_atoms(2), weight=1.0),
        _lj_frame(2, _well_separated_atoms(2), weight=1.0),
    ]
    state_of = {0: 0, 1: 1, 2: 1}
    pool = EnsembleFramePool(frames, state_of=lambda f: state_of[f.frame_id])

    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(2),
        kT=1.0,
        backend=_lj_backend(2),
        energy_window=None,
        min_pair_dist=None,
        state=1,
    )
    seen = set()
    for shot_id in range(200):
        istate, report = sampler.sample(_key(shot_id=shot_id, frame_id=-1))
        assert report.ok
        seen.add(istate.frame_id)
    assert seen <= {1, 2}
    assert 0 not in seen


def test_5_7b_state_restricted_choice_is_weighted_within_state():
    """D-I9 M6: within-state weights misaligned (`weights[:len(idxs)]`)
    survived because 5.7 used equal weights."""
    frames = [
        _lj_frame(0, _well_separated_atoms(2), weight=5.0),
        _lj_frame(1, _well_separated_atoms(2), weight=1.0),
        _lj_frame(2, _well_separated_atoms(2), weight=3.0),
    ]
    pool = EnsembleFramePool(frames, state_of=lambda f: 0 if f.frame_id == 0 else 1)
    rng = np.random.default_rng(7)
    n = 20000
    c2 = sum(pool.choose(rng, state=1).frame_id == 2 for _ in range(n))
    assert abs(c2 - 0.75 * n) < 4 * math.sqrt(n * 0.75 * 0.25)
    assert pool.weight_fraction([2], state=1) == pytest.approx(0.75)


# --- decision: low-dof temperature check is skipped, never rejects ------


def test_temperature_check_skipped_for_low_dof():
    # 5 atoms * 3 - 3 (COM) = 12 dof < 30 -> must be skipped, not rejected.
    # remove_com_momentum=True is the sampler's default, and sample() itself
    # always actually removes it, so the data-derived dof (R17b) is exactly
    # the 12 computed here, not just a config-flag assumption.
    n_atoms = 5
    frame = _lj_frame(0, _well_separated_atoms(n_atoms), weight=1.0)
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(n_atoms),
        kT=1.0,
        backend=_lj_backend(n_atoms),
        energy_window=None,
        min_pair_dist=None,
    )
    istate, report = sampler.sample(_key(shot_id=0))
    assert report.ok is True
    assert report.checks["temperature"] == "skipped_low_dof"
    assert report.checks["temperature_dof"] == pytest.approx(3 * n_atoms - 3)


def test_temperature_check_applied_for_high_dof_and_data_derived_dof():
    # 11 atoms -> 33 raw velocity components. R17b: validate() must derive
    # dof from whether *this v* actually has ~zero total momentum, not from
    # the sampler's static remove_com_momentum=True config flag.
    n_atoms = 11
    kT = 1.0
    masses = np.ones(n_atoms)
    frame = _lj_frame(0, _well_separated_atoms(n_atoms), weight=1.0)
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=kT,
        backend=_lj_backend(n_atoms, kT=kT),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=True,
    )

    raw_v = np.random.default_rng(0).normal(0.0, math.sqrt(kT), size=(n_atoms, 3))
    total_p = np.sum(masses[:, None] * raw_v, axis=0)
    com_removed_v = raw_v - total_p / masses.sum()

    # (a) actually COM-removed, correctly-scaled velocities -> data-derived
    # dof is 33 - 3 = 30 (at the threshold), and it should pass.
    good_report = sampler.validate(
        InitialState(
            state=MDState(x=frame.coordinates, v=com_removed_v, t=0.0),
            frame_id=0,
            meta={},
        )
    )
    assert good_report.checks["temperature_dof"] == pytest.approx(30.0)
    assert isinstance(good_report.checks["temperature"], float)
    assert "temperature" not in good_report.reasons

    # (b) same COM-removed velocities, 100x too hot: scaling a ~zero total
    # momentum by a constant keeps it ~zero, so dof is still 30 -- but this
    # must now be rejected on "temperature".
    hot_report = sampler.validate(
        InitialState(
            state=MDState(x=frame.coordinates, v=com_removed_v * 100.0, t=0.0),
            frame_id=0,
            meta={},
        )
    )
    assert hot_report.checks["temperature_dof"] == pytest.approx(30.0)
    assert hot_report.ok is False
    assert "temperature" in hot_report.reasons

    # (c) the actual Important-2 regression case: the sampler is configured
    # with remove_com_momentum=True, but *this* v was never actually
    # COM-corrected (its total momentum is not ~0 -- it's raw MB noise).
    # dof must reflect that: 33, not 30, regardless of the config flag.
    not_removed_report = sampler.validate(
        InitialState(
            state=MDState(x=frame.coordinates, v=raw_v, t=0.0),
            frame_id=0,
            meta={},
        )
    )
    assert not_removed_report.checks["temperature_dof"] == pytest.approx(33.0)


def test_temperature_band_edge_is_five_sigma():
    """D-I9 M7: sigma_T = kT sqrt(2/dof); 4.9 sigma passes, 5.1 sigma fails."""
    n_atoms = 200
    kT = 1.0
    masses = np.ones(n_atoms)
    x = _well_separated_atoms(n_atoms)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), masses, kT,
                                   _FakeBackend(), None, None, remove_com_momentum=False)
    dof = 3 * n_atoms
    v0 = np.random.default_rng(5).normal(size=(n_atoms, 3))
    T0 = float(np.sum(v0 ** 2)) / dof
    sigma = kT * math.sqrt(2.0 / dof)
    for n_sigma, ok in [(4.9, True), (-4.9, True), (5.1, False), (-5.1, False)]:
        target = kT + n_sigma * sigma
        v = v0 * math.sqrt(target / T0)
        rep = sampler.validate(_istate(x, v))
        assert rep.checks["temperature_dof"] == dof
        assert rep.ok is ok, (n_sigma, rep.checks["temperature"])


# --- decision: remove_com_momentum only meaningful for (n_atoms, 3) -----


def test_remove_com_momentum_raises_for_non_atom_shapes():
    # R17a: this must now raise eagerly at *construction*, against the
    # pool's frame shapes -- not lazily, only once someone happens to draw
    # velocities.
    frame = EnsembleFrame(
        coordinates=np.array([0.5]),
        box=None,
        topology_ref="toy1d",
        temperature=1.0,
        weight=1.0,
        source_id="s",
        frame_id=0,
        time=0.0,
    )
    pool = EnsembleFramePool([frame])
    backend = AnalyticBackend(DoubleWell1D(barrier=5.0), integrator="baoab", dt=0.001, kT=1.0, gamma=0.1)
    with pytest.raises(ValueError):
        EnsembleFrameSampler(
            pool=pool,
            masses=np.ones(1),
            kT=1.0,
            backend=backend,
            energy_window=None,
            min_pair_dist=None,
            remove_com_momentum=True,
        )

    # With remove_com_momentum=False it must work fine, both to construct
    # and to draw from.
    sampler2 = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(1),
        kT=1.0,
        backend=backend,
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=False,
    )
    v = sampler2._draw_velocities(np.random.default_rng(0), frame.coordinates.shape)
    assert v.shape == (1,)


def test_min_pair_dist_raises_for_non_atom_shapes_at_construction():
    # R17a: min_pair_dist set against non-(n_atoms>=2, 3) frames must also
    # raise eagerly at construction.
    frame = EnsembleFrame(
        coordinates=np.array([0.5]),
        box=None,
        topology_ref="toy1d",
        temperature=1.0,
        weight=1.0,
        source_id="s",
        frame_id=0,
        time=0.0,
    )
    pool = EnsembleFramePool([frame])
    backend = AnalyticBackend(DoubleWell1D(barrier=5.0), integrator="baoab", dt=0.001, kT=1.0, gamma=0.1)
    with pytest.raises(ValueError):
        EnsembleFrameSampler(
            pool=pool,
            masses=np.ones(1),
            kT=1.0,
            backend=backend,
            energy_window=None,
            min_pair_dist=0.1,
            remove_com_momentum=False,
        )

    # A single-atom (1, 3) frame is also too few atoms for min_pair_dist.
    single_atom_frame = _lj_frame(0, [[0.0, 0.0, 0.0]])
    single_atom_pool = EnsembleFramePool([single_atom_frame])
    with pytest.raises(ValueError):
        EnsembleFrameSampler(
            pool=single_atom_pool,
            masses=np.ones(1),
            kT=1.0,
            backend=_lj_backend(1),
            energy_window=None,
            min_pair_dist=0.1,
            remove_com_momentum=False,
        )


def test_com_removal_raises_for_anisotropic_per_atom_mass_array():
    # R18 #3: an explicit (n_atoms, 3) mass array with different mass per
    # component for the same atom has no well-defined "atom momentum" to
    # remove -- must raise rather than silently use only one column's mass.
    n_atoms = 2
    frame = _lj_frame(0, _well_separated_atoms(n_atoms), weight=1.0)
    pool = EnsembleFramePool([frame])
    anisotropic_masses = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=anisotropic_masses,
        kT=1.0,
        backend=_lj_backend(n_atoms),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=True,
    )
    with pytest.raises(ValueError):
        sampler._draw_velocities(np.random.default_rng(0), frame.coordinates.shape)

    # An isotropic (n_atoms, 3) mass array (same mass repeated across each
    # atom's row) is fine.
    isotropic_masses = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    sampler2 = EnsembleFrameSampler(
        pool=pool,
        masses=isotropic_masses,
        kT=1.0,
        backend=_lj_backend(n_atoms),
        energy_window=None,
        min_pair_dist=None,
        remove_com_momentum=True,
    )
    v = sampler2._draw_velocities(np.random.default_rng(0), frame.coordinates.shape)
    total_p = np.sum(isotropic_masses * v, axis=0)
    assert np.max(np.abs(total_p)) < 1e-10


# --- min_pair_dist gate ----------------------------------------------------


def test_min_pair_dist_rejects_close_atoms():
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    frame = _lj_frame(0, coords)
    pool = EnsembleFramePool([frame])
    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=np.ones(2),
        kT=1.0,
        backend=_lj_backend(2),
        energy_window=None,
        min_pair_dist=0.5,
    )
    istate = InitialState(
        state=MDState(x=coords, v=np.zeros((2, 3)), t=0.0),
        frame_id=0,
        meta={},
    )
    report = sampler.validate(istate)
    assert report.ok is False
    assert "min_pair_dist" in report.reasons
    assert report.checks["min_pair_dist"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Fix wave 2, package L4 (fixreview-p2 N-I1, m1, m6; from_openmm_system)
# ---------------------------------------------------------------------------


def _diatomic_sampler(x, pairs, nmol, backend=None, **kw):
    N = 2 * nmol
    masses = np.where(np.arange(N) % 2 == 0, 16.0, 1.0)
    cons = S.DistanceConstraints(pairs, np.ones(nmol), masses)
    return EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), masses, 1.0,
                                backend or _FakeBackend(), None, None, constraints=cons, **kw), cons


@pytest.mark.parametrize("rel_off", [0.3, 0.05])
def test_n_i1_frame_far_off_the_constraint_manifold_is_rejected_not_repaired(rel_off):
    """p2-N-I1: sample() projected every frame and gated only the projected
    state, so a frame 30 % off its constraints was silently moved."""
    x, pairs = _diatomics(10)
    x[1, 0] += rel_off  # bond 0: 1.0 -> 1 + rel_off
    sampler, _ = _diatomic_sampler(x, pairs, 10)
    with pytest.raises(ICRejectedError) as ei:
        sampler.sample(_key())
    assert ei.value.reasons == ["constraint_input"] and ei.value.level == "coordinate"
    assert ei.value.attempts[0]["reasons"] == ["constraint_input"]
    assert "constraint_input" in S.COORDINATE_REASONS


def test_n_i1_per_atom_wrapped_periodic_frame_is_rejected():
    """A water-like molecule broken across the periodic edge (per-atom
    wrapping): OpenMM constrains raw coordinates, so it must be refused."""
    L = 4.0
    pairs = np.array([[0, 1], [2, 3]])
    # molecule 0 straddles x = L: its H (whole position 4.3) is wrapped to 0.3
    x = np.array([[3.8, 1.0, 1.0], [0.3, 1.0, 1.0], [1.5, 2.5, 2.0], [2.5, 2.5, 2.0]])
    frame = _lj_frame(0, x, box=np.diag([L] * 3))
    masses = np.array([16.0, 1.0, 16.0, 1.0])
    cons = S.DistanceConstraints(pairs, np.ones(2), masses)
    sampler = EnsembleFrameSampler(EnsembleFramePool([frame]), masses, 1.0, _FakeBackend(), None, None,
                                   constraints=cons)
    with pytest.raises(ICRejectedError, match="constraint_input"):
        sampler.sample(_key())


def test_n_i1_dcd_like_input_residual_still_passes_and_is_projected():
    x, pairs = _diatomics(10)
    x[1::2, 0] += 2e-6  # float32 DCD rounding + 1e-5 integrator tolerance
    sampler, cons = _diatomic_sampler(x, pairs, 10)
    istate, rep = sampler.sample(_key())
    assert rep.ok and rep.checks["constraint_residual_input"] == pytest.approx(2e-6, rel=1e-3)
    assert cons.residual(istate.state.x, istate.state.v) < 1e-9


def test_n_i1_input_tolerance_is_validated_and_configurable():
    x, pairs = _diatomics(2)
    x[1, 0] += 0.05
    with pytest.raises(ValueError, match="input_constraint_tolerance"):
        _diatomic_sampler(x, pairs, 2, input_constraint_tolerance=0.0)
    sampler, _ = _diatomic_sampler(x, pairs, 2, input_constraint_tolerance=0.1)
    _istate_ok, rep = sampler.sample(_key())
    assert rep.ok


def test_n_i1_prevalidate_pool_flags_far_off_frames_and_projects_near_ones():
    """p2 m1 (R3): prevalidate_pool must project a frame that is only
    slightly off the manifold (1e-4, between the gate's 1e-5 and the input
    tolerance) instead of rejecting it -- and reject a far-off one."""
    near, pairs = _diatomics(3)
    near[1, 0] += 1e-4
    far = near.copy()
    far[3, 0] += 0.2
    masses = np.where(np.arange(6) % 2 == 0, 16.0, 1.0)
    cons = S.DistanceConstraints(pairs, np.ones(3), masses)
    pool = EnsembleFramePool([_lj_frame(0, near), _lj_frame(1, far)])
    sampler = EnsembleFrameSampler(pool, masses, 1.0, _FakeBackend(), None, None, constraints=cons)
    pv = sampler.prevalidate_pool()
    assert [r["frame_id"] for r in pv.rejected] == [1]
    assert pv.rejected[0]["reasons"] == ["constraint_input"]


def test_m1_prevalidate_pool_respects_the_sampler_state():
    """p2 m1 (R10)."""
    good = _well_separated_atoms(4)
    bad = good.copy()
    bad[1] = bad[0]  # coincident atoms
    pool = EnsembleFramePool([_lj_frame(0, good, source_id="A"), _lj_frame(1, bad, source_id="B")],
                             state_of=lambda f: f.source_id)
    sampler = EnsembleFrameSampler(pool, np.ones(4), 1.0, _FakeBackend(), None, 0.5, state="A")
    pv = sampler.prevalidate_pool()
    assert pv.n_frames == 1 and pv.n_rejected == 0


class _BoxRecordingBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.boxes = []

    def energy_forces(self, x, box=None):
        self.boxes.append(None if box is None else np.array(box))
        return super().energy_forces(x, box)


def test_m1_validate_evaluates_energy_in_the_frames_own_box():
    """p2 m1 (R11, ruling R20): NPT pools need the frame's box, not the
    backend's default cell."""
    x = _well_separated_atoms(3, spacing=1.0)
    box = np.diag([3.3, 3.1, 3.7])
    backend = _BoxRecordingBackend()
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x, box=box)]), np.ones(3), 1.0,
                                   backend, None, None)
    sampler.sample(_key())
    assert backend.boxes and all(b is not None and np.array_equal(b, box) for b in backend.boxes)


class _NaNVelocityOnce(EnsembleFrameSampler):
    def _draw_velocities(self, rng, shape):
        v = super()._draw_velocities(rng, shape)
        return np.full(shape, np.nan) if not getattr(self, "_drawn", False) and not setattr(self, "_drawn", True) else v


def test_m1_an_attempt_with_coordinate_and_velocity_reasons_raises_at_coordinate_level():
    """p2 m1 (R2: any -> all): one coordinate-level reason is enough to
    stop, even when the same attempt also failed on velocities."""
    x = _well_separated_atoms(3)
    sampler = _NaNVelocityOnce(EnsembleFramePool([_lj_frame(0, x)]), np.ones(3), 1.0,
                               _FakeBackend(E=1e9), (-1.0, 1.0), None, max_redraws=5)
    with pytest.raises(ICRejectedError) as ei:
        sampler.sample(_key())
    assert ei.value.level == "coordinate"
    assert set(ei.value.attempts[0]["reasons"]) == {"nonfinite_v", "energy_window"}
    assert len(ei.value.attempts) == 1


class _RaisingEnergyBackend(_FakeBackend):
    def energy_forces(self, x, box=None):
        raise ValueError("box is smaller than twice the nonbonded cutoff")


def test_m6_backend_energy_error_is_a_coordinate_level_rejection():
    """p2 m6: a deterministic per-frame backend error (e.g. OpenMM refusing
    a box below 2x the cutoff) must be an ICRejectedError, not abort the
    batch."""
    x = _well_separated_atoms(3)
    sampler = EnsembleFrameSampler(EnsembleFramePool([_lj_frame(0, x)]), np.ones(3), 1.0,
                                   _RaisingEnergyBackend(), None, None)
    with pytest.raises(ICRejectedError) as ei:
        sampler.sample(_key())
    assert ei.value.reasons == ["energy_error"] and ei.value.level == "coordinate"
    rep = sampler.validate(_istate(x, np.zeros_like(x)))
    assert "twice the nonbonded cutoff" in rep.checks["energy_error"]


def test_from_openmm_system_matches_the_systems_constraints():
    openmm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")
    system = openmm.System()
    for m in (16.0, 1.0, 1.0):
        system.addParticle(m * unit.dalton)
    system.addConstraint(0, 1, 0.09572 * unit.nanometer)
    system.addConstraint(0, 2, 0.09572)
    system.addConstraint(1, 2, 0.15139 * unit.nanometer)
    cons = S.DistanceConstraints.from_openmm_system(system)
    assert cons.n_constraints == 3 and cons.projection_tolerance == 1e-10
    assert cons.pairs.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert np.allclose(cons.distances, [0.09572, 0.09572, 0.15139])
    assert np.allclose(cons.masses, [16.0, 1.0, 1.0])
    system.addParticle(0.0)  # a virtual site
    with pytest.raises(ValueError, match="mass"):
        S.DistanceConstraints.from_openmm_system(system)


def test_load_frames_reads_files_written_before_the_rename(tmp_path):
    """Frame files written as venus-ng (format tag ``venus-ng-frames/1``, e.g.
    runs/a3/frames) still load; save_frames writes the current tag; any other
    tag is refused."""
    from cytherea.ic.frames import load_frames, save_frames

    frames = [EnsembleFrame(coordinates=np.full((2, 3), float(k)), box=None, topology_ref="t",
                            temperature=300.0, weight=1.0, source_id="s", frame_id=k, time=float(k))
              for k in range(2)]
    save_frames(tmp_path / "new.npz", frames)
    with np.load(tmp_path / "new.npz") as z:
        assert str(z["format"]) == "cytherea-frames/1"
        arrays = {k: z[k] for k in z.files}
    for tag, ok in (("venus-ng-frames/1", True), ("other-frames/1", False)):
        np.savez(tmp_path / "f.npz", **dict(arrays, format=np.array(tag)))
        if ok:
            got = load_frames(tmp_path / "f.npz")
            assert [f.frame_id for f in got] == [0, 1] and np.array_equal(got[1].coordinates, frames[1].coordinates)
        else:
            with pytest.raises(ValueError, match="cytherea-frames/1"):
                load_frames(tmp_path / "f.npz")
