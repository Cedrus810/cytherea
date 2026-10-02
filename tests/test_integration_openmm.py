"""INT1 end-to-end integration test: `run_shot`/`run_batch` with the real
`OpenMMBackend` on alanine dipeptide in vacuum (openmmtools
`AlanineDipeptideVacuum`), driving `EnsembleFrameSampler` (MB velocities +
the IC gate), `FixedLag`, and phi/psi observables -- everything below this
test is unit-tested against a test double or the analytic backend; this is
the one place all of it is wired together against a real molecular System.

Design (per task-INT1-brief.md / global-constraints.md):
- Reference platform, double precision (bitwise-reproducible on this
  platform, and GPU is not needed for this task).
- `AlanineDipeptideVacuum()`'s default System uses HBonds constraints (12
  H-bond constraints, 22 atoms, non-periodic) -- confirmed interactively
  and mirrored in `PhysicsConfig.constraints="hbonds"`, `rigid_water=False`.
  The System is non-periodic, so every `EnsembleFrame` in the pool carries
  `box=None` (a periodic frame would be a configuration error for this
  System -- `OpenMMPropagator.set_state` rejects a non-None box against a
  non-periodic System).
- Two pool frames come from a short Langevin equilibration run (gamma=1/ps,
  purpose="equilibration" -- allowed any friction per design section 9);
  the measurement dynamics use Verlet, purpose="measurement".
- phi/psi are the standard alanine-dipeptide backbone dihedrals, computed
  from `MDState.x` with a small local dihedral helper (no dihedral utility
  exists elsewhere in the codebase yet).

Kept fast (well under the ~20s budget: Reference platform on a 22-atom
vacuum system executes thousands of steps per second) so it runs in the
default `pytest -q`, not gated behind `slow`.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import openmm.unit as u
import pytest

from cytherea.backends.base import MDState, PhysicsConfig
from cytherea.backends.openmm_backend import OpenMMBackend
from cytherea.engine.shot import ObsSpec, SpecLabeler, protocol_hash, run_shot
from cytherea.exec.batch import run_batch
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import DistanceConstraints, EnsembleFrameSampler
from cytherea.keys import ShotKey, derive_rng, key_digest
from cytherea.observe.events import AbsorbingAB, FixedLag, offline_replay, spec_region
from cytherea.store import ShotRecord, Store, config_hash

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from openmmtools import testsystems

# Standard alanine-dipeptide backbone atom indices for this topology (ACE
# residue 0: atoms 0-5; ALA residue 1: atoms 6-15; NME residue 2: atoms
# 16-21 -- confirmed interactively against openmmtools' topology):
#   phi = C(ACE,4) - N(ALA,6)  - CA(ALA,8)  - C(ALA,14)
#   psi = N(ALA,6) - CA(ALA,8) - C(ALA,14)  - N(NME,16)
_PHI_IDX = (4, 6, 8, 14)
_PSI_IDX = (6, 8, 14, 16)


def _dihedral(x: np.ndarray, idx: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = (x[i] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _phi(state: MDState) -> float:
    return _dihedral(state.x, _PHI_IDX)


def _psi(state: MDState) -> float:
    return _dihedral(state.x, _PSI_IDX)


def _kT_kJ_per_mol(T: float) -> float:
    return (u.MOLAR_GAS_CONSTANT_R * T * u.kelvin).value_in_unit(u.kilojoule_per_mole)


def _key(shot_id: int = 0, stage: str = "int1") -> ShotKey:
    return ShotKey(global_seed=42, frame_id=0, shot_id=shot_id, stage=stage)


def _ala():
    t = testsystems.AlanineDipeptideVacuum()  # default: HBonds, non-periodic
    x0 = np.asarray(t.positions.value_in_unit(u.nanometer), dtype=np.float64)
    return t.system, t.topology, x0


def _masses(system) -> np.ndarray:
    return np.array(
        [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())]
    )


def _mb_velocities(masses: np.ndarray, T: float, key: ShotKey) -> np.ndarray:
    kT = _kT_kJ_per_mol(T)
    rng = derive_rng(key, "velocities")
    return rng.normal(size=(len(masses), 3)) * np.sqrt(kT / masses)[:, None]


def _measurement_cfg(**overrides) -> PhysicsConfig:
    base = dict(
        integrator="verlet",
        dt_ps=0.001,
        temperature_K=300.0,
        friction_per_ps=0.0,
        constraints="hbonds",
        rigid_water=False,
        platform="Reference",
        precision="double",
        deterministic_forces=True,
        purpose="measurement",
    )
    base.update(overrides)
    return PhysicsConfig(**base)


def _make_pool_and_sampler(system, topology, x0, masses, measurement_backend):
    """Two pool frames from a short Langevin equilibration (gamma=1/ps,
    purpose="equilibration"), fed into an EnsembleFrameSampler gated by the
    *measurement* backend's energy_forces (as run_shot's IC gate would be)."""
    eq_cfg = PhysicsConfig(
        integrator="langevin_middle",
        dt_ps=0.001,
        temperature_K=300.0,
        friction_per_ps=1.0,
        constraints="hbonds",
        rigid_water=False,
        platform="Reference",
        precision="double",
        deterministic_forces=True,
        purpose="equilibration",
    )
    eq_backend = OpenMMBackend(system, topology, eq_cfg)
    v0 = _mb_velocities(masses, 300.0, _key(stage="eq_v0"))
    prop = eq_backend.build(MDState(x=x0, v=v0, t=0.0), None, _key(stage="eq"))
    prop.run(500)
    s1 = prop.get_state()
    prop.run(500)
    s2 = prop.get_state()
    assert s1.box is None and s2.box is None  # vacuum system: no periodic box

    # Contract K1: whatever a frame's time, each shot's clock starts at t=0
    # (the frame time is provenance in InitialState.meta["frame_time"]; the
    # solvated INT2 test below uses nonzero frame times).
    frame1 = EnsembleFrame(
        coordinates=s1.x, box=None, topology_ref="ala_vacuum", temperature=300.0,
        weight=1.0, source_id="eq", frame_id=0, time=0.0,
    )
    frame2 = EnsembleFrame(
        coordinates=s2.x, box=None, topology_ref="ala_vacuum", temperature=300.0,
        weight=1.0, source_id="eq", frame_id=1, time=0.0,
    )
    pool = EnsembleFramePool([frame1, frame2])

    e1, _ = measurement_backend.energy_forces(frame1.coordinates)
    e2, _ = measurement_backend.energy_forces(frame2.coordinates)
    lo, hi = min(e1, e2) - 200.0, max(e1, e2) + 200.0

    sampler = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=_kT_kJ_per_mol(300.0),
        backend=measurement_backend,
        energy_window=(lo, hi),
        min_pair_dist=0.05,
        max_redraws=10,
    )
    return pool, sampler


def _make_obs_stop():
    obs = ObsSpec(fns={"phi": _phi, "psi": _psi}, dt_obs=0.005, store_stride=1)
    stop = FixedLag(tau=0.02)
    return obs, stop


# ---------------------------------------------------------------------------
# The main end-to-end test: run_shot with store=None, records complete and
# reproducible bitwise across two identical runs (Reference platform).
# ---------------------------------------------------------------------------


def test_int1_run_shot_end_to_end_complete_and_reproducible():
    system, topology, x0 = _ala()
    masses = _masses(system)
    cfg = _measurement_cfg()
    backend = OpenMMBackend(system, topology, cfg)
    _pool, sampler = _make_pool_and_sampler(system, topology, x0, masses, backend)
    obs, stop = _make_obs_stop()
    key = _key(shot_id=0, stage="measure")

    rec1 = run_shot(key, sampler, backend, stop, obs, cfg, store=None)
    rec2 = run_shot(key, sampler, backend, stop, obs, cfg, store=None)

    # -- completeness --
    assert rec1.kind == "shot"
    assert rec1.frame_id in (0, 1)
    assert rec1.ic_validity["ok"] is True
    assert rec1.stop_rule_kind == "fixed_lag"
    assert rec1.stop_reason == "fixed_lag"
    assert rec1.event_time is not None
    assert rec1.physics_config_hash == config_hash(backend.effective_config(cfg))  # K9
    assert rec1.backend_provenance["kind"] == "openmm"
    assert rec1.backend_provenance["platform"] == "Reference"
    assert set(rec1.observables) == {"t", "phi", "psi"}
    n = len(rec1.observables["t"])
    assert n >= 2
    assert len(rec1.observables["phi"]) == n and len(rec1.observables["psi"]) == n
    assert all(np.isfinite(v) for v in rec1.observables["phi"])
    assert all(np.isfinite(v) for v in rec1.observables["psi"])

    # -- bitwise reproducibility for the same key on the Reference platform --
    assert rec1 == rec2


def test_int1_store_none_does_not_write_then_run_batch_writes(tmp_path):
    system, topology, x0 = _ala()
    masses = _masses(system)
    cfg = _measurement_cfg()
    backend = OpenMMBackend(system, topology, cfg)
    _pool, sampler = _make_pool_and_sampler(system, topology, x0, masses, backend)
    obs, stop = _make_obs_stop()

    store = Store(tmp_path / "store.sqlite")
    key0 = _key(shot_id=0, stage="measure")

    rec_direct = run_shot(key0, sampler, backend, stop, obs, cfg, store=None)

    assert not store.has(key_digest(key0))  # store=None: nothing written

    shot_fn = functools.partial(
        run_shot, sampler=sampler, backend=backend, stop=stop, obs=obs,
        physics_cfg=cfg, store=None,
    )
    keys = [_key(shot_id=i, stage="measure") for i in range(3)]
    results = run_batch(keys, shot_fn, store, n_workers=1)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, ShotRecord)
    assert results[0] == rec_direct
    for key, result in zip(keys, results):
        assert store.has(key_digest(key))
        assert store.get(key_digest(key)) == result


def test_int1_dt_obs_not_multiple_of_propagator_dt_rejected(tmp_path):
    system, topology, x0 = _ala()
    masses = _masses(system)
    cfg = _measurement_cfg()  # propagator.dt == cfg.dt_ps == 0.001
    backend = OpenMMBackend(system, topology, cfg)
    _pool, sampler = _make_pool_and_sampler(system, topology, x0, masses, backend)
    _obs, stop = _make_obs_stop()
    bad_obs = ObsSpec(fns={"phi": _phi, "psi": _psi}, dt_obs=0.0017, store_stride=1)
    store = Store(tmp_path / "store.sqlite")
    key = _key(shot_id=0, stage="baddtobs")

    with pytest.raises(ValueError):
        run_shot(key, sampler, backend, stop, bad_obs, cfg, store=store)
    assert not store.has(key_digest(key))


def test_int1_nose_hoover_measurement_rejected():
    system, topology, _x0 = _ala()
    with pytest.raises(ValueError, match="equilibration"):
        OpenMMBackend(
            system,
            topology,
            PhysicsConfig(
                integrator="nose_hoover",
                dt_ps=0.001,
                temperature_K=300.0,
                friction_per_ps=1.0,
                constraints="hbonds",
                rigid_water=False,
                platform="Reference",
                precision="double",
                deterministic_forces=True,
                purpose="measurement",
            ),
        )


def test_int1_provenance_describes_physics_cfg_not_backend_cfg():
    """Review C-I2 / contract K8: when run_shot passes a physics_cfg that
    differs from the backend's constructor cfg, the record's
    backend_provenance must describe the dynamics that actually ran (here
    measurement Verlet), not the constructor cfg (equilibration Langevin)."""
    system, topology, x0 = _ala()
    masses = _masses(system)
    cfg = _measurement_cfg()
    eq_cfg = _measurement_cfg(
        integrator="langevin_middle", friction_per_ps=1.0, purpose="equilibration", dt_ps=0.002
    )
    backend = OpenMMBackend(system, topology, eq_cfg)
    _pool, sampler = _make_pool_and_sampler(system, topology, x0, masses, backend)
    obs, stop = _make_obs_stop()

    rec = run_shot(_key(shot_id=0, stage="k8"), sampler, backend, stop, obs, cfg, store=None)

    prov = rec.backend_provenance
    assert prov["integrator"] == "verlet"
    assert prov["purpose"] == "measurement"
    assert prov["friction_per_ps"] == 0.0
    assert prov["dt_ps"] == cfg.dt_ps
    assert prov["effective_config"] == backend.effective_config(cfg)
    assert rec.physics_config_hash == config_hash(backend.effective_config(cfg))  # K9


# ---------------------------------------------------------------------------
# INT2: sampler + engine + OpenMM backend on a solvated PERIODIC system (CPU)
#
# A small rigid-water box (openmmtools WaterBox, 50 waters, 150 constraints,
# PME, 1.2 nm box), CPU platform / mixed precision (Threads=1, deterministic
# forces). Two pool frames with a periodic box and NONZERO frame times come
# from a short equilibration run; the IC sampler projects onto the System's
# own constraints (DistanceConstraints built from getConstraintParameters),
# its min_pair_dist check is minimum-image, and the energy gate evaluates at
# the frame's box. Contracts exercised: K1 (clock from 0), K2 (ic_meta), K3
# (explicit frame_id selects that frame), K4 (nonfinite), K6 (thinning flag
# and offline replay of a stored record), K8 (cfg=None ->
# effective_config hash) and R39 (protocol_hash, spec regions).
# ---------------------------------------------------------------------------

_WB_FRAME_TIMES = (0.1, 0.2)  # ps: provenance only, never the shot clock


def _wb_cfg(**overrides) -> PhysicsConfig:
    base = dict(
        integrator="verlet", dt_ps=0.002, temperature_K=300.0, friction_per_ps=0.0,
        constraints="none", rigid_water=True, platform="CPU", precision="mixed",
        deterministic_forces=True, purpose="measurement",
    )
    base.update(overrides)
    return PhysicsConfig(**base)


def _wb_key(shot_id=0, frame_id=0, stage="int2") -> ShotKey:
    return ShotKey(global_seed=7, frame_id=frame_id, shot_id=shot_id, stage=stage)


@pytest.fixture(scope="module")
def waterbox_setup():
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    system = wb.system
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    box0 = np.array(
        [[v[i].value_in_unit(u.nanometer) for i in range(3)] for v in system.getDefaultPeriodicBoxVectors()]
    )
    masses = _masses(system)
    backend = OpenMMBackend(system, wb.topology, _wb_cfg())
    eq = OpenMMBackend(
        system, wb.topology,
        _wb_cfg(integrator="langevin_middle", friction_per_ps=1.0, purpose="equilibration"),
    )
    prop = eq.build(MDState(x0, np.zeros_like(x0), 0.0, box0), None, _wb_key(stage="eq"))
    states = []
    for _ in _WB_FRAME_TIMES:
        prop.run(50)
        states.append(prop.get_state())
    frames = [
        EnsembleFrame(coordinates=s.x, box=s.box, topology_ref="waterbox50", temperature=300.0,
                      weight=w, source_id="eq", frame_id=i, time=t)
        for i, (s, t, w) in enumerate(zip(states, _WB_FRAME_TIMES, (1.0, 3.0)))
    ]
    pairs, dist = [], []
    for k in range(system.getNumConstraints()):
        i, j, d = system.getConstraintParameters(k)
        pairs.append((i, j))
        dist.append(d.value_in_unit(u.nanometer))
    energies = [backend.energy_forces(f.coordinates, f.box)[0] for f in frames]
    sampler = EnsembleFrameSampler(
        pool=EnsembleFramePool(frames), masses=masses, kT=_kT_kJ_per_mol(300.0), backend=backend,
        energy_window=(min(energies) - 500.0, max(energies) + 500.0), min_pair_dist=0.05,
        max_redraws=10, constraints=DistanceConstraints(pairs, dist, masses),
        topology_ref="waterbox50",
    )
    return dict(system=system, backend=backend, frames=frames, sampler=sampler, n_constraints=len(pairs))


def _wb_obs(store_stride=1, extra=None) -> ObsSpec:
    fns = {"o0_z": lambda s: float(s.x[0, 2])}
    fns.update(extra or {})
    return ObsSpec(fns=fns, dt_obs=0.01, store_stride=store_stride)


def test_int2_solvated_periodic_shot_end_to_end_cpu(waterbox_setup, tmp_path):
    ws = waterbox_setup
    backend, sampler, frames = ws["backend"], ws["sampler"], ws["frames"]
    assert all(f.box is not None and f.time > 0.0 for f in frames)
    obs, stop = _wb_obs(), FixedLag(0.05)

    rec = run_shot(_wb_key(frame_id=1), sampler, backend, stop, obs, None, store=None)

    # K1: the clock starts at 0 whatever the frame time, t = step_index * dt
    t = rec.observables["t"]
    assert t[0] == 0.0
    assert t == [(5 * k) * 0.002 for k in range(6)]  # step_index * dt, multiplied not accumulated
    assert rec.stop_reason == "fixed_lag" and rec.event_time == pytest.approx(0.05)
    assert rec.warnings == []  # nothing rebased: the sampler hands over t = 0
    # K2/K3: meta of the explicitly selected frame, recorded as ic_meta
    assert rec.frame_id == 1
    assert rec.ic_meta["frame_id"] == 1
    assert rec.ic_meta["frame_time"] == _WB_FRAME_TIMES[1]
    assert rec.ic_meta["frame_weight"] == 3.0  # the frame's own weight, as given to the pool
    assert rec.ic_meta["topology_ref"] == "waterbox50" and rec.ic_meta["n_redraws"] == 0
    assert set(rec.ic_meta) >= {"frame_id", "frame_time", "frame_weight", "source_id", "topology_ref",
                                "state", "n_redraws"}
    # IC gate agrees with the System: dof = 3N - n_constraints - 3, periodic box
    checks = rec.ic_validity["checks"]
    n_atoms = ws["system"].getNumParticles()
    assert checks["temperature_dof"] == 3 * n_atoms - ws["n_constraints"] - 3
    assert checks["box_volume"] == pytest.approx(1.2**3)
    assert checks["constraint_residual"] < 1e-8
    # R39 and K8
    assert isinstance(rec.protocol_hash, str) and len(rec.protocol_hash) == 64
    assert rec.protocol_hash == protocol_hash(stop, obs, sampler)
    assert rec.physics_config_hash == config_hash(backend.effective_config(None))
    assert rec.backend_provenance["platform"] == "CPU"
    # K6: stride 1 -> unthinned; the stored record replays to the online decision
    assert rec.observables_thinned is False
    store = Store(tmp_path / "s.sqlite")
    store.append(rec)
    stored = store.get(rec.key_digest)
    assert stored == rec
    replay = offline_replay(FixedLag(0.05), stored)
    assert (replay.reason, replay.event_time) == (rec.stop_reason, rec.event_time)
    # same key -> identical record (CPU, Threads=1, deterministic forces)
    assert run_shot(_wb_key(frame_id=1), sampler, backend, stop, obs, None, store=None) == rec


def test_int2_sampler_ic_is_what_the_openmm_propagator_runs(waterbox_setup):
    """The sampler's constraint projection and OpenMM's agree: building the
    propagator from the sampled IC leaves x and v unchanged to OpenMM's
    constraint tolerance (no hidden re-projection that would change the
    ensemble the gate certified), and the IC carries the frame's box."""
    ws = waterbox_setup
    istate, validity = ws["sampler"].sample(_wb_key(frame_id=0))
    assert validity.ok and istate.state.t == 0.0
    assert np.array_equal(istate.state.box, ws["frames"][0].box)
    prop = ws["backend"].build(istate.state, None, _wb_key(frame_id=0))
    s0 = prop.get_state()
    assert np.max(np.abs(s0.x - istate.state.x)) < 1e-6
    assert np.max(np.abs(s0.v - istate.state.v)) / np.max(np.abs(istate.state.v)) < 1e-4
    assert np.array_equal(s0.box, istate.state.box)


def test_int2_thinned_record_is_flagged_and_refused_by_replay(waterbox_setup):
    ws = waterbox_setup
    rec = run_shot(_wb_key(frame_id=0), ws["sampler"], ws["backend"], FixedLag(0.04), _wb_obs(store_stride=2),
                   None, store=None)
    assert rec.observables_thinned is True
    assert rec.observables["t"][0] == 0.0
    with pytest.raises(ValueError, match="thinned"):
        offline_replay(FixedLag(0.04), rec)


def test_int2_nonfinite_observable_stops_the_solvated_shot(waterbox_setup, tmp_path):
    """K4 end to end: an observable that turns NaN at the third observation
    (t = 0.02 ps) stops the shot as "nonfinite" (not a timeout, no final
    label) under an AbsorbingAB rule with spec regions (R39), and the
    record survives the store round trip."""
    ws = waterbox_setup
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        return float("nan") if calls["n"] >= 3 else 0.0

    obs = _wb_obs(extra={"q": flaky})
    stop = AbsorbingAB(
        spec_region("A", lambda o: o["q"] < -1.0, "q < -1"),
        spec_region("B", lambda o: o["q"] > 1.0, "q > 1"),
        tau_persist=0.01, t_max=1.0,
    )
    rec = run_shot(_wb_key(frame_id=0, stage="int2_nan"), ws["sampler"], ws["backend"], stop, obs, None,
                   store=None, labeler=SpecLabeler(lambda o: "A", "always A"))
    assert rec.stop_reason == "nonfinite"
    assert rec.event_time == pytest.approx(0.02)
    assert rec.final_state_label is None
    assert rec.observables["t"][0] == 0.0 and len(rec.observables["t"]) == 3
    assert np.isnan(rec.observables["q"][-1])
    assert rec.ic_meta["frame_time"] == _WB_FRAME_TIMES[0]
    assert isinstance(rec.protocol_hash, str) and rec.observables_thinned is False
    store = Store(tmp_path / "nan.sqlite")
    store.append(rec)
    back = store.get(rec.key_digest)
    assert back.stop_reason == "nonfinite" and np.isnan(back.observables["q"][-1])
