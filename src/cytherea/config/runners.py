"""Mode dispatch and the runners behind the CLI (Task 13).

`dispatch(cfg, resume=False)` returns a zero-argument callable that runs
the config's mode and returns a small summary dict.

Store bookkeeping: next to ``store_path`` a sidecar ``<store_path>.config.json``
holds the normalized config and its `RunConfig.config_hash`. ``run`` refuses
an existing store (use resume); ``resume`` requires the store and a sidecar
with the same config hash -- on top of `run_batch`'s own checks of code
version, physics and protocol hashes -- and then only runs the keys that are
not in the store yet (contract of `run_batch`, test 7.4). A WE run cannot be
resumed (its records hold no coordinates; p7 M-7): ``resume`` refuses it.

Modes: ``shoot.ensemble`` and ``shoot.surface`` share one runner (keys
``ShotKey(seed, frame_id, shot_id, stage)`` over ``budget.frames`` x
``budget.shots_per_frame`` through `run_batch`, or `run_we` with a WE
resampler); ``prepare`` minimises, equilibrates and writes frames;
``shoot.encounter`` shoots from the b sphere (`EncounterSampler`); its IC
rejections (e.g. ``encounter_clash``) are logged as failures by `run_batch`.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import json
import os
from typing import Callable

import numpy as np

from cytherea.config import build as B
from cytherea.config.schema import RunConfig
from cytherea.exec.batch import ShotFailure, run_batch
from cytherea.keys import SegmentKey, ShotKey
from cytherea.observe.events import SpecPredicate
from cytherea.store import Store


class ResumeError(RuntimeError):
    """``run`` over an existing store, or ``resume`` that does not match it."""


def sidecar_path(store_path: str) -> str:
    return f"{store_path}.config.json"


def _write_sidecar(cfg: RunConfig) -> None:
    payload = {"config_hash": cfg.config_hash(), "mode": cfg.mode, "config": cfg.normalized()}
    tmp = sidecar_path(cfg.store_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.replace(tmp, sidecar_path(cfg.store_path))


def _open_store(cfg: RunConfig, resume: bool) -> Store:
    path, side = cfg.store_path, sidecar_path(cfg.store_path)
    if not resume:
        if os.path.exists(path) or os.path.exists(side):
            raise ResumeError(f"{path} already exists: use `cytherea resume` (or another store_path)")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        store = Store(path)
        _write_sidecar(cfg)
        return store
    if not os.path.exists(path):
        raise ResumeError(f"{path} does not exist: nothing to resume (use `cytherea run`)")
    if not os.path.exists(side):
        raise ResumeError(f"{side} is missing: cannot check that the config is the one the store was run with")
    with open(side, encoding="utf-8") as fh:
        recorded = json.load(fh)
    if recorded.get("config_hash") != cfg.config_hash():
        raise ResumeError(
            f"the config differs from the one {path} was run with (config_hash {recorded.get('config_hash')} "
            f"vs {cfg.config_hash()}); see {side}"
        )
    return Store(path)


def _summary(results) -> dict:
    reasons = collections.Counter()
    n_fail = 0
    for r in results:
        if isinstance(r, ShotFailure):
            n_fail += 1
        else:
            reasons[r.stop_reason] += 1
    return {"n_keys": len(results), "n_failures": n_fail, "stop_reasons": dict(sorted(reasons.items()))}


# ------------------------------------------------------------------ shoot


def run_shoot(cfg: RunConfig, resume: bool = False) -> dict:
    if cfg.resampler.kind == "we":
        if resume:
            raise ResumeError("a WE run cannot be resumed from its store (records hold no coordinates; p7 M-7)")
        return _run_we(cfg)
    shot_fn, pool = B.build_shot_fn(cfg)
    keys = B.shot_keys(cfg, pool)
    store = _open_store(cfg, resume)
    results = run_batch(keys, shot_fn, store, n_workers=cfg.budget.n_workers, return_records=False)
    return {"mode": cfg.mode, "store": cfg.store_path, **_summary(results)}


@dataclasses.dataclass(frozen=True)
class _Progress:
    fn: object

    def __call__(self, state) -> np.ndarray:
        return np.array([self.fn(state)])

    @property
    def spec(self):
        return {"progress": self.fn.spec}


@dataclasses.dataclass(frozen=True)
class _BinOf:
    edges: tuple[float, ...]

    def __call__(self, z) -> int:
        return int(np.searchsorted(self.edges, z[0], side="right"))


def _run_we(cfg: RunConfig) -> dict:
    from cytherea.resample.we import BinnedWE, Walker, run_we

    w = cfg.resampler
    backend = B.build_backend(cfg)
    pool = B.build_pool(cfg)
    sampler = B.build_sampler(cfg, backend, pool)
    obs = B.build_obs(cfg)
    z_fn = _Progress(obs.fns[w.progress])
    fids = B.frame_ids(cfg, pool)
    fw = np.array([pool.get(f).weight for f in fids], dtype=float)
    fw = fw / fw.sum()
    init, i = [], 0
    for fid, wf in zip(fids, fw):
        for j in range(w.walkers_per_frame):
            ist, _ = sampler.sample(ShotKey(cfg.seed, fid, j, f"{cfg.budget.stage}:we-init"))
            init.append(Walker(SegmentKey(cfg.seed, cfg.budget.stage, 0, i), None, (0, 0),
                               float(wf / w.walkers_per_frame), z_fn(ist.state), ist.state))
            i += 1
    total = sum(x.weight for x in init)
    init = [dataclasses.replace(x, weight=x.weight / total) for x in init]
    store = _open_store(cfg, resume=False)
    edges = tuple(float(e) for e in w.bin_edges)
    res = run_we(init, backend, BinnedWE(SpecPredicate(_BinOf(edges), {"edges": list(edges)}), w.target_per_bin),
                 B.build_stop(cfg), z_fn, w.n_iter, w.tau_seg, store, cfg.seed, cfg.budget.stage, None,
                 dt_obs=obs.dt_obs, observables=obs.fns)
    return {"mode": cfg.mode, "store": cfg.store_path, "resampler": "we", "n_walkers": int(res.n_walkers.sum()),
            "absorbed": {k: float(v.sum()) for k, v in sorted(res.absorbed.items())},
            "final_weight": res.final_weight, "valid": res.valid}


# ---------------------------------------------------------------- prepare


def run_prepare(cfg: RunConfig, resume: bool = False) -> dict:
    """Minimise (optional), equilibrate ``budget.equilibrate``, then save
    ``budget.n_frames`` frames every ``budget.frame_interval``. Velocities
    come from the IC sampler with key ``ShotKey(seed, 0, 0, stage)``."""
    import openmm
    import openmm.unit as u

    from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool, save_frames

    if resume:
        raise ResumeError("prepare is not resumable: run it again")
    out = cfg.budget.output_frames
    if os.path.exists(out):
        raise ResumeError(f"{out} already exists")
    system, topology, x = B._cached_openmm(cfg.system.system_xml, cfg.system.topology_pdb)
    backend = B.build_backend(cfg)
    periodic = system.usesPeriodicBoundaryConditions()
    box = np.array([[v.value_in_unit(u.nanometer) for v in vec] for vec in system.getDefaultPeriodicBoxVectors()]) \
        if periodic else None
    e_min = None
    if cfg.budget.minimize:
        platform = openmm.Platform.getPlatformByName(cfg.physics.platform)
        ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
        ctx.setPositions(x)
        if box is not None:
            ctx.setPeriodicBoxVectors(*box)
        openmm.LocalEnergyMinimizer.minimize(ctx)
        st = ctx.getState(getPositions=True, getEnergy=True)
        x = np.asarray(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=np.float64)
        e_min = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        del ctx
    ref = f"sha256:{backend.topology_sha256}"
    start = EnsembleFrame(coordinates=x, box=box, topology_ref=ref, temperature=cfg.physics.temperature,
                          weight=1.0, source_id="structure", frame_id=0, time=0.0)
    sampler = B.build_sampler(cfg, backend, EnsembleFramePool([start]))
    key = ShotKey(cfg.seed, 0, 0, cfg.budget.stage)
    ist, _ = sampler.sample(key)
    prop = backend.build(ist.state, None, key)
    dt = float(prop.dt)
    if cfg.budget.equilibrate:
        prop.run(int(round(cfg.budget.equilibrate / dt)))
    n_int = int(round(cfg.budget.frame_interval / dt))
    if n_int < 1:
        raise ValueError("budget.frame_interval is shorter than one step")
    frames = []
    for k in range(cfg.budget.n_frames):
        prop.run(n_int)
        s = prop.get_state()
        frames.append(EnsembleFrame(coordinates=np.asarray(s.x, dtype=np.float64), box=s.box, topology_ref=ref,
                                    temperature=cfg.physics.temperature, weight=1.0,
                                    source_id=f"prepare:{cfg.budget.stage}", frame_id=k, time=float(s.t)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_frames(out, frames)
    return {"mode": cfg.mode, "output_frames": out, "n_frames": len(frames), "minimized_energy_kJ_mol": e_min}


def run_encounter(cfg: RunConfig, resume: bool = False) -> dict:
    """``budget.shots_per_frame`` shots from the b sphere, keys ``ShotKey(seed, -1, k, stage)``."""
    if cfg.resampler.kind != "none":
        raise ValueError("shoot.encounter with a resampler is not wired through the config yet (use run_we)")
    from cytherea.engine.shot import run_shot

    backend = B.build_backend(cfg)
    sampler = B.build_encounter_sampler(cfg, backend)
    shot_fn = functools.partial(run_shot, sampler=sampler, backend=backend, stop=B.build_stop(cfg),
                                obs=B.build_obs(cfg), physics_cfg=None)
    keys = [ShotKey(cfg.seed, -1, k, cfg.budget.stage) for k in range(cfg.budget.shots_per_frame)]
    store = _open_store(cfg, resume)
    results = run_batch(keys, shot_fn, store, n_workers=cfg.budget.n_workers, return_records=False)
    return {"mode": cfg.mode, "store": cfg.store_path, **_summary(results)}


RUNNERS: dict[str, Callable[..., dict]] = {
    "prepare": run_prepare,
    "shoot.ensemble": run_shoot,
    "shoot.surface": run_shoot,
    "shoot.encounter": run_encounter,
}


def dispatch(cfg: RunConfig, resume: bool = False) -> Callable[[], dict]:
    """The runner of ``cfg.mode``, bound to ``cfg`` (and ``resume``)."""
    return functools.partial(RUNNERS[cfg.mode], cfg, resume)


# ----------------------------------------------------------------- report


def report(store_path: str) -> dict:
    """Summary of a store: counts by kind / stage / stop reason, failures,
    distinct hashes and code versions, and the sidecar's config hash."""
    if not os.path.exists(store_path):
        raise FileNotFoundError(store_path)
    store = Store(store_path)
    n, kinds, stages, reasons = 0, collections.Counter(), collections.Counter(), collections.Counter()
    phys, proto, code = set(), set(), set()
    for r in store.iter():
        n += 1
        kinds[r.kind] += 1
        stages[str(r.key.get("stage", r.key.get("run_id")))] += 1
        reasons[r.stop_reason] += 1
        phys.add(r.physics_config_hash)
        proto.add(r.protocol_hash)
        code.add(r.code_version)
    out = {
        "store": store_path, "owner_host": store.owner_host(), "n_records": n,
        "kinds": dict(sorted(kinds.items())), "stages": dict(sorted(stages.items())),
        "stop_reasons": dict(sorted(reasons.items())), "n_failures": len(store.failures()),
        "physics_config_hashes": sorted(p for p in phys if p), "protocol_hashes": sorted(p for p in proto if p),
        "code_versions": sorted(c for c in code if c),
    }
    side = sidecar_path(store_path)
    if os.path.exists(side):
        with open(side, encoding="utf-8") as fh:
            meta = json.load(fh)
        out["config_hash"], out["mode"] = meta.get("config_hash"), meta.get("mode")
    return out
