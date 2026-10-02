#!/usr/bin/env python
"""A3 encounter shots: barnase-barstar from the NAM b sphere (Task 16.3 / 16.5).

  prepare  partner frames: each partner alone in GBn2 (same force field and
           salt as the complex), LangevinMiddle gamma = 1/ps (equilibration),
           --equil-ps of burn-in then --n-frames frames every --frame-ps;
           written with cytherea.ic.frames.save_frames
  shoot    --n shots from b through cytherea (EncounterSampler, run_shot,
           run_batch; resumable: only missing keys run) with the measurement
           dynamics: LangevinMiddle gamma = 0.1/ps, 2 fs, HBonds, 300 K,
           CUDA mixed + DeterministicForces. Observables every 1 ps: r (COM
           distance) and Q (fraction of native interface contacts). Stop:
           BSurface -- reaction Q >= Q_BOUND held for TAU_PERSIST, escape
           r >= q (= Q2_FACTOR * b), timeout at T_MAX.
  analyze  beta at q2 (the shots' own escape surface) and at q1 = 2 b by
           offline replay of every record with the q1 rule (a shot that
           reaches q1 first escapes there), both through estimate_kon
           (Jeffreys interval, timeouts / non-finite stops counted, validity
           flag); the interval of beta_inf(q2) - beta_inf(q1) from the
           Jeffreys posterior of the paired outcomes; stop times.

Native contacts: build_system.py's heavy-atom pairs within 0.5 nm in the
minimised complex; a pair counts as formed when d < 1.25 d_native.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

B_NM = 5.0          # b sphere (> sum of the partners' maximal COM extents, 4.64 nm)
Q2_FACTOR = 3.0     # shots escape at q2 = 3 b; q1 = 2 b by offline replay
Q_BOUND = 0.3       # reaction: >= 30 % of the native contacts ...
TAU_PERSIST = 20.0  # ... held for 20 ps
T_MAX = 50_000.0    # ps
DT_OBS = 1.0        # ps
MIN_AB_DIST = 0.3   # nm, contact check of the placement (never binding at b = 5 nm)


@dataclasses.dataclass(frozen=True)
class NativeContacts:
    """Fraction of native pairs (i, j, d0) with |x_i - x_j| < factor * d0."""

    i: tuple[int, ...]
    j: tuple[int, ...]
    d0: tuple[float, ...]
    factor: float = 1.25

    def __call__(self, state) -> float:
        x = np.asarray(state.x)
        d = np.linalg.norm(x[list(self.i)] - x[list(self.j)], axis=1)
        return float(np.mean(d < self.factor * np.asarray(self.d0)))

    @property
    def spec(self) -> dict:
        h = hashlib.sha256(np.array([self.i, self.j], dtype=np.int64).tobytes()
                           + np.array(self.d0, dtype=np.float64).tobytes()).hexdigest()
        return {"native_contacts_sha256": h, "n_pairs": len(self.i), "factor": self.factor}


def _load(sysdir: Path):
    import openmm
    import openmm.app as app

    build = json.loads((sysdir / "build.json").read_text())
    system = openmm.XmlSerializer.deserialize((sysdir / "complex_system.xml").read_text())
    pdb = app.PDBFile(str(sysdir / "complex.pdb"))
    return build, system, pdb


def _masses(system) -> np.ndarray:
    import openmm.unit as u

    return np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])


def cmd_prepare(a) -> None:
    import openmm
    import openmm.app as app
    import openmm.unit as u

    from cytherea.ic.frames import EnsembleFrame, save_frames

    build, _, _ = _load(a.system_dir)
    ff = app.ForceField(*build["forcefield"])
    kappa = build["implicit_kappa_per_nm"]
    a.out.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(("barnase", "barstar")):
        pdb = app.PDBFile(str(a.system_dir / f"{name}.pdb"))
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                                 implicitSolventKappa=kappa / u.nanometer)
        integ = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picosecond)
        integ.setRandomNumberSeed(a.seed + k)
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(a.platform),
                             {"Precision": "mixed"} if a.platform == "CUDA" else {})
        ctx.setPositions(pdb.positions)
        ctx.setVelocitiesToTemperature(300 * u.kelvin, a.seed + k)
        integ.step(int(round(a.equil_ps / 0.002)))
        frames = []
        for f in range(a.n_frames):
            integ.step(int(round(a.frame_ps / 0.002)))
            x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
            frames.append(EnsembleFrame(coordinates=np.asarray(x, dtype=np.float64), box=None,
                                        topology_ref=f"{name}:{build['pdb_sha256'][:16]}", temperature=300.0,
                                        weight=1.0, source_id=f"{name}-equil", frame_id=f,
                                        time=a.equil_ps + (f + 1) * a.frame_ps))
        save_frames(a.out / f"{name}.npz", frames)
        print(f"{name}: {len(frames)} frames -> {a.out / f'{name}.npz'}")


def make_shot(sysdir: Path, frames_dir: Path, q: float, platform: str = "CUDA"):
    from cytherea.backends.base import PhysicsConfig
    from cytherea.backends.openmm_backend import OpenMMBackend
    from cytherea.config.build import ComDistance
    from cytherea.engine.shot import ObsSpec, run_shot
    from cytherea.ic.encounter import EncounterSampler
    from cytherea.ic.frames import EnsembleFramePool, load_frames
    from cytherea.ic.sampler import KB_KJ_PER_MOL_K, DistanceConstraints

    build, system, pdb = _load(sysdir)
    n_a = build["n_atoms"]["barnase"]
    n = system.getNumParticles()
    m = _masses(system)
    cfg = PhysicsConfig(integrator="langevin_middle", dt_ps=0.002, temperature_K=300.0, friction_per_ps=0.1,
                        constraints="hbonds", rigid_water=False, platform=platform,
                        precision="mixed" if platform == "CUDA" else "double", deterministic_forces=True,
                        purpose="measurement")
    backend = OpenMMBackend(system, pdb.topology, cfg)
    kT = KB_KJ_PER_MOL_K * 300.0
    sampler = EncounterSampler(EnsembleFramePool(load_frames(frames_dir / "barnase.npz")),
                               EnsembleFramePool(load_frames(frames_dir / "barstar.npz")), B_NM, m, kT, backend,
                               MIN_AB_DIST, (0, 0), constraints=DistanceConstraints.from_openmm_system(system),
                               boltzmann_constant=KB_KJ_PER_MOL_K)
    pairs = json.loads((sysdir / "native_contacts.json").read_text())["pairs"]
    Qfn = NativeContacts(tuple(int(p[0]) for p in pairs), tuple(int(p[1]) for p in pairs),
                         tuple(float(p[2]) for p in pairs))
    rfn = ComDistance((0, n_a), (n_a, n), tuple(m[:n_a]), tuple(m[n_a:]), periodic=False)
    obs = ObsSpec(fns={"r": rfn, "Q": Qfn}, dt_obs=DT_OBS, store_stride=1)
    stop = make_stop(q)
    return functools.partial(run_shot, sampler=sampler, backend=backend, stop=stop, obs=obs, physics_cfg=None)


def make_stop(q: float):
    from cytherea.config.build import AllOf
    from cytherea.observe.events import BSurface, spec_region

    bound = spec_region("bound", AllOf((("Q", "ge", Q_BOUND),)), {"all_of": [["Q", "ge", Q_BOUND]]})
    return BSurface(bound, "r", q, tau_persist=TAU_PERSIST, t_max=T_MAX)


def cmd_shoot(a) -> None:
    from cytherea.exec.batch import run_batch
    from cytherea.keys import ShotKey, key_digest
    from cytherea.store import Store

    shot = make_shot(a.system_dir, a.frames_dir, Q2_FACTOR * B_NM, a.platform)
    keys = [ShotKey(a.seed, -1, k, a.stage) for k in range(a.n)]
    store = Store(a.store)
    t0 = time.perf_counter()
    done = sum(1 for k in keys if store.has(key_digest(k)))
    for i in range(0, len(keys), a.chunk):  # chunks: progress lines and bounded memory
        res = run_batch(keys[i:i + a.chunk], shot, store, n_workers=1, return_records=False)
        reasons = {}
        for r in res:
            s = getattr(r, "stop_reason", "ic_rejected")
            reasons[s] = reasons.get(s, 0) + 1
        print(f"{time.strftime('%H:%M:%S')} keys {i}-{i + len(res) - 1}: {reasons}  "
              f"wall {time.perf_counter() - t0:.0f} s (skipped {done} already done)", flush=True)


MIN_OUTCOME = 5  # fewer reactions or escapes than this: the difference interval is prior-dominated
_CELLS = ("reaction", "escape", "other")


def _cell(reason: str) -> int:
    return _CELLS.index(reason) if reason in _CELLS[:2] else 2


def diff_interval(o1, o2, q1: float, q2: float, seed: int = 1, n_draw: int = 20000) -> tuple[float, float]:
    """95 % interval of beta_inf(q2) - beta_inf(q1) from the Jeffreys posterior of
    the paired outcomes: each shot is one cell of (outcome at q1) x (outcome at
    q2), cells reaction / escape / other (timeout, non-finite), the cell
    probabilities ~ Dirichlet(counts + 1/2). The pairing is kept (a shot can
    escape at q1 and still react at q2), and unlike a bootstrap the interval
    never collapses to a point on few shots."""
    from cytherea.estimate.association import nam_beta_inf

    counts = np.zeros((3, 3))
    for a, b in zip(o1, o2):
        counts[_cell(a), _cell(b)] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    p = rng.dirichlet(counts.ravel() + 0.5, size=n_draw).reshape(n_draw, 3, 3)
    r1, e1 = p[:, 0, :].sum(1), p[:, 1, :].sum(1)
    r2, e2 = p[:, :, 0].sum(1), p[:, :, 1].sum(1)
    b1, b2 = r1 / (r1 + e1), r2 / (r2 + e2)
    d = np.array([nam_beta_inf(y, B_NM, q2) - nam_beta_inf(x, B_NM, q1) for x, y in zip(b1, b2)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_records(recs, seed: int = 1, n_draw: int = 20000) -> dict:
    """beta / beta_inf at q2 (the records' own outcomes) and at q1 = 2 b (offline
    replay), each through `estimate_kon` (weighted Jeffreys interval; timeouts
    and non-finite stops counted, ``valid`` False on any non-finite stop or
    > 5 % timeouts, K4), and the interval of beta_inf(q2) - beta_inf(q1) from
    `diff_interval`. ``valid`` is False when either estimate is invalid;
    ``diff_ci_note`` warns when the interval is dominated by the prior (fewer
    than MIN_OUTCOME reactions or escapes at one q). Shot weights are 1
    (b-sphere shots drawn by weight)."""
    import dataclasses as dc

    from cytherea.estimate.association import estimate_kon
    from cytherea.observe.events import offline_replay

    recs = list(recs)
    if not recs:
        raise ValueError("no records")
    q2, q1 = Q2_FACTOR * B_NM, 2.0 * B_NM
    at_q1, t_stop = [], []
    for r in recs:
        t_stop.append(r.event_time if r.event_time is not None else r.observables["t"][-1])
        d = offline_replay(make_stop(q1), r)
        reason = "timeout" if d is None else d.reason
        at_q1.append(dc.replace(r, stop_reason=reason, event_time=None if d is None else d.event_time))
    est = {"q2": estimate_kon(recs, B_NM, q2, 1.0), "q1": estimate_kon(at_q1, B_NM, q1, 1.0)}

    def summary(e):
        return {"beta": e.beta, "beta_ci": list(e.beta_ci), "beta_inf": e.beta_inf,
                "beta_inf_ci": list(e.beta_inf_ci), "n_reaction": e.n_reaction, "n_escape": e.n_escape,
                "n_timeout": e.n_timeout, "n_nonfinite": e.n_nonfinite, "valid": e.valid}

    prior_dominated = any(min(e.n_reaction, e.n_escape) < MIN_OUTCOME for e in est.values())
    return {
        "n_shots": len(recs), "b_nm": B_NM, "q1_nm": q1, "q2_nm": q2,
        "q2": summary(est["q2"]), "q1": summary(est["q1"]),
        "valid": bool(est["q1"].valid and est["q2"].valid),
        "beta_inf_q2_minus_q1": est["q2"].beta_inf - est["q1"].beta_inf,
        "diff_ci95": list(diff_interval([r.stop_reason for r in at_q1], [r.stop_reason for r in recs],
                                        q1, q2, seed=seed, n_draw=n_draw)),
        "diff_ci_method": "Jeffreys-Dirichlet posterior of the paired (q1, q2) outcomes",
        "diff_ci_note": f"fewer than {MIN_OUTCOME} reactions or escapes at one q: the interval is "
                        "dominated by the prior" if prior_dominated else None,
        "stop_time_ps": {"median": float(np.median(t_stop)), "mean": float(np.mean(t_stop)),
                         "p90": float(np.percentile(t_stop, 90)), "total_ns": float(np.sum(t_stop)) / 1000},
    }


def cmd_analyze(a) -> None:
    from cytherea.store import Store

    recs = [r for r in Store(a.store).iter(kind="shot") if r.key.get("stage") == a.stage]
    res = analyze_records(recs, seed=a.seed)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--system-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--equil-ps", type=float, default=200.0)
    s.add_argument("--n-frames", type=int, default=20)
    s.add_argument("--frame-ps", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=20261002)
    s.add_argument("--platform", default="CUDA")
    s = sub.add_parser("shoot")
    s.add_argument("--system-dir", type=Path, required=True)
    s.add_argument("--frames-dir", type=Path, required=True)
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--stage", default="a3")
    s.add_argument("--seed", type=int, default=20261002)
    s.add_argument("--chunk", type=int, default=5)
    s.add_argument("--platform", default="CUDA")
    s = sub.add_parser("analyze")
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--stage", default="a3")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    {"prepare": cmd_prepare, "shoot": cmd_shoot, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
