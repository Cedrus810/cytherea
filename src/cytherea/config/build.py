"""Engine objects from a `RunConfig` (Task 13).

Everything built here is picklable (module-level dataclasses, no lambdas),
so ``budget.n_workers > 1`` works for analytic backends, and carries a
``spec`` so it enters the protocol hash (ruling R39). The protocol hash
covers observable *names* only (`cytherea.engine.shot.protocol_description`);
their definitions (indices, atoms, periodicity) are covered by the config
hash that the CLI checks on resume.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

from cytherea.backends.base import MDState
from cytherea.config.schema import RegionSpec, RunConfig
from cytherea.engine.shot import ObsSpec, run_shot
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool, load_frames
from cytherea.ic.sampler import DistanceConstraints, EnsembleFrameSampler
from cytherea.keys import ShotKey
from cytherea.observe.events import AbsorbingAB, BSurface, FixedLag, spec_region

# ------------------------------------------------------------ observables


@dataclasses.dataclass(frozen=True)
class Coord:
    index: int

    def __call__(self, state: MDState) -> float:
        return float(np.asarray(state.x).reshape(-1)[self.index])

    @property
    def spec(self) -> dict:
        return {"coord": self.index}


def _min_image(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    frac = np.linalg.solve(box.T, d)
    frac -= np.round(frac)
    return box.T @ frac


@dataclasses.dataclass(frozen=True)
class Distance:
    i: int
    j: int
    periodic: bool = True

    def __call__(self, state: MDState) -> float:
        x = np.asarray(state.x)
        d = x[self.j] - x[self.i]
        if self.periodic and state.box is not None:
            d = _min_image(d, np.asarray(state.box, dtype=float))
        return float(np.linalg.norm(d))

    @property
    def spec(self) -> dict:
        return {"distance": [self.i, self.j], "periodic": self.periodic}


@dataclasses.dataclass(frozen=True)
class Dihedral:
    """IUPAC dihedral in radians, (-pi, pi]; positions as stored (molecules whole)."""

    a: int
    b: int
    c: int
    d: int

    def __call__(self, state: MDState) -> float:
        x = np.asarray(state.x, dtype=float)
        p0, p1, p2, p3 = x[self.a], x[self.b], x[self.c], x[self.d]
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))

    @property
    def spec(self) -> dict:
        return {"dihedral": [self.a, self.b, self.c, self.d]}


@dataclasses.dataclass(frozen=True)
class ComDistance:
    """Distance between the mass-weighted centres of two atom-index ranges."""

    a: tuple[int, int]
    b: tuple[int, int]
    masses_a: tuple[float, ...]
    masses_b: tuple[float, ...]
    periodic: bool = True

    def __call__(self, state: MDState) -> float:
        x = np.asarray(state.x, dtype=float)
        ma, mb = np.asarray(self.masses_a), np.asarray(self.masses_b)
        ca = (ma[:, None] * x[self.a[0]:self.a[1]]).sum(0) / ma.sum()
        cb = (mb[:, None] * x[self.b[0]:self.b[1]]).sum(0) / mb.sum()
        d = cb - ca
        if self.periodic and state.box is not None:
            d = _min_image(d, np.asarray(state.box, dtype=float))
        return float(np.linalg.norm(d))

    @property
    def spec(self) -> dict:
        import hashlib

        m = np.ascontiguousarray(np.concatenate([self.masses_a, self.masses_b]), dtype=np.float64)
        return {"com_distance": [list(self.a), list(self.b)], "periodic": self.periodic,
                "masses_sha256": hashlib.sha256(m.tobytes()).hexdigest()}


def build_obs(cfg: RunConfig) -> ObsSpec:
    fns = {}
    for o in cfg.observables.items:
        if o.kind == "coord":
            fns[o.name] = Coord(o.index)
        elif o.kind == "distance":
            fns[o.name] = Distance(*o.atoms, periodic=o.periodic)
        elif o.kind == "com_distance":
            (a0, a1), (b0, b1) = o.ranges
            m = masses_of(cfg, max(a1, b1))
            ma = m[a0:a1] if m.size > 1 else np.full(a1 - a0, m[0])
            mb = m[b0:b1] if m.size > 1 else np.full(b1 - b0, m[0])
            fns[o.name] = ComDistance((a0, a1), (b0, b1), tuple(map(float, ma)), tuple(map(float, mb)),
                                      periodic=o.periodic)
        else:
            fns[o.name] = Dihedral(*o.atoms)
    return ObsSpec(fns=fns, dt_obs=cfg.observables.dt_obs, store_stride=cfg.observables.store_stride)


# ----------------------------------------------------------------- regions


@dataclasses.dataclass(frozen=True)
class AllOf:
    """Picklable region predicate: every ``(observable, op, value)`` holds."""

    conditions: tuple[tuple[str, str, float], ...]

    def __call__(self, obs) -> bool:
        for name, op, value in self.conditions:
            x = obs[name]
            if not (x <= value if op == "le" else x >= value):
                return False
        return True


def build_region(r: RegionSpec):
    conds = tuple((c.observable, "le" if c.le is not None else "ge", float(c.le if c.le is not None else c.ge))
                  for c in r.all_of)
    return spec_region(r.name, AllOf(conds), {"all_of": [list(c) for c in conds]})


def build_stop(cfg: RunConfig):
    s = cfg.stop
    if s.kind == "fixed_lag":
        return FixedLag(s.tau)
    if s.kind == "absorbing_AB":
        return AbsorbingAB(build_region(s.A), build_region(s.B), tau_persist=s.tau_persist, t_max=s.t_max)
    return BSurface(build_region(s.reaction), s.r_observable, s.q, tau_persist=s.tau_persist, t_max=s.t_max)


# ----------------------------------------------------------------- backend


def _potential(cfg: RunConfig):
    from cytherea.backends import analytic

    p = cfg.system.potential
    cls = getattr(analytic, p.name)
    try:
        return cls(**p.params)
    except TypeError as exc:
        raise ValueError(f"system.potential: {p.name}({p.params}): {exc}") from None


def load_openmm(cfg: RunConfig):
    """(System, Topology, positions in nm) of an OpenMM system spec."""
    import openmm
    import openmm.app as app
    import openmm.unit as u

    with open(cfg.system.system_xml, encoding="utf-8") as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(cfg.system.topology_pdb)
    x = np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=np.float64)
    if x.shape[0] != system.getNumParticles():
        raise ValueError(f"{cfg.system.topology_pdb}: {x.shape[0]} atoms, System has {system.getNumParticles()}")
    return system, pdb.topology, x


@functools.lru_cache(maxsize=None)
def _cached_openmm(xml: str, pdb: str):
    import types

    return load_openmm(types.SimpleNamespace(system=types.SimpleNamespace(system_xml=xml, topology_pdb=pdb)))


def build_backend(cfg: RunConfig):
    ph = cfg.physics
    if cfg.system.kind == "analytic":
        from cytherea.backends.analytic import AnalyticBackend

        return AnalyticBackend(_potential(cfg), ph.integrator, ph.dt, ph.kT, gamma=ph.gamma, mass=ph.mass)
    from cytherea.backends.openmm_backend import OpenMMBackend

    system, topology, _ = _cached_openmm(cfg.system.system_xml, cfg.system.topology_pdb)
    return OpenMMBackend(system, topology, ph.to_physics_config())


def kT_of(cfg: RunConfig) -> float:
    if cfg.physics.kind == "analytic":
        return cfg.physics.kT
    import openmm.unit as u

    return (u.MOLAR_GAS_CONSTANT_R * cfg.physics.temperature * u.kelvin).value_in_unit(u.kilojoule_per_mole)


def masses_of(cfg: RunConfig, n_particles: int) -> np.ndarray:
    if cfg.physics.kind == "analytic":
        return np.array([cfg.physics.mass])
    import openmm.unit as u

    system = _cached_openmm(cfg.system.system_xml, cfg.system.topology_pdb)[0]
    return np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(n_particles)])


# --------------------------------------------------------------------- IC


def build_pool(cfg: RunConfig) -> EnsembleFramePool:
    ic = cfg.ic
    if ic.kind == "points":
        frames = [
            EnsembleFrame(coordinates=np.asarray(p, dtype=float).reshape(-1), box=None,
                          topology_ref=f"analytic:{cfg.system.potential.name}", temperature=cfg.physics.kT,
                          weight=1.0, source_id=f"point{i}", frame_id=i, time=0.0)
            for i, p in enumerate(ic.points)
        ]
        return EnsembleFramePool(frames)
    if ic.kind == "frames":
        return EnsembleFramePool(load_frames(ic.path))
    raise ValueError(f"ic.kind {ic.kind!r} has no frame pool")


def build_sampler(cfg: RunConfig, backend, pool: EnsembleFramePool | None = None) -> EnsembleFrameSampler:
    ic = cfg.ic
    pool = pool if pool is not None else build_pool(cfg)
    if cfg.system.kind == "analytic":
        return EnsembleFrameSampler(pool=pool, masses=masses_of(cfg, 1), kT=kT_of(cfg), backend=backend,
                                    energy_window=getattr(ic, "energy_window", None),
                                    min_pair_dist=getattr(ic, "min_pair_dist", None),
                                    remove_com_momentum=ic.kind == "frames" and ic.remove_com_momentum,
                                    max_redraws=getattr(ic, "max_redraws", 20))
    system = _cached_openmm(cfg.system.system_xml, cfg.system.topology_pdb)[0]
    n = system.getNumParticles()
    constraints = DistanceConstraints.from_openmm_system(system) if (
        ic.kind != "frames" or ic.constraints == "from_system") and system.getNumConstraints() else None
    return EnsembleFrameSampler(
        pool=pool, masses=masses_of(cfg, n), kT=kT_of(cfg), backend=backend,
        energy_window=getattr(ic, "energy_window", None), min_pair_dist=getattr(ic, "min_pair_dist", None),
        max_redraws=getattr(ic, "max_redraws", 20), constraints=constraints,
        remove_com_momentum=getattr(ic, "remove_com_momentum", True),
    )


def build_encounter_sampler(cfg: RunConfig, backend):
    from cytherea.ic.encounter import EncounterSampler

    ic = cfg.ic
    pool_A, pool_B = EnsembleFramePool(load_frames(ic.frames_A)), EnsembleFramePool(load_frames(ic.frames_B))
    n = len(pool_A.frames[0].coordinates) + len(pool_B.frames[0].coordinates)
    constraints = None
    if cfg.system.kind == "openmm" and ic.constraints == "from_system":
        system = _cached_openmm(cfg.system.system_xml, cfg.system.topology_pdb)[0]
        if system.getNumConstraints():
            constraints = DistanceConstraints.from_openmm_system(system)
    return EncounterSampler(pool_A, pool_B, ic.b, masses_of(cfg, n) if cfg.system.kind == "openmm"
                            else np.full(n, cfg.physics.mass), kT_of(cfg), backend, ic.min_pair_dist, ic.label,
                            energy_window=ic.energy_window, constraints=constraints, max_redraws=ic.max_redraws)


def frame_ids(cfg: RunConfig, pool: EnsembleFramePool) -> list[int]:
    ids = [f.frame_id for f in pool.frames]
    if cfg.budget.frames == "all":
        return ids
    missing = sorted(set(cfg.budget.frames) - set(ids))
    if missing:
        raise ValueError(f"budget.frames: frame ids {missing} are not in the pool")
    return list(cfg.budget.frames)


def shot_keys(cfg: RunConfig, pool: EnsembleFramePool) -> list[ShotKey]:
    return [ShotKey(global_seed=cfg.seed, frame_id=fid, shot_id=k, stage=cfg.budget.stage)
            for fid in frame_ids(cfg, pool) for k in range(cfg.budget.shots_per_frame)]


def build_shot_fn(cfg: RunConfig):
    """``partial(run_shot, ...)`` for `run_batch` (no store: run_batch writes)."""
    backend = build_backend(cfg)
    pool = build_pool(cfg)
    sampler = build_sampler(cfg, backend, pool)
    fn = functools.partial(run_shot, sampler=sampler, backend=backend, stop=build_stop(cfg), obs=build_obs(cfg),
                           physics_cfg=None)
    return fn, pool
