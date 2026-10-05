#!/usr/bin/env python
"""A1 shooting (Task 14b): long fixed-lag shots of alanine dipeptide from core
frames of the 14a reference, analysed as T(tau), its implied timescales and
the shots' own Chapman-Kolmogorov test.

  frames   choose N_PER_STATE core-interior DCD frames per state from the
           reference runs (burn-in skipped; systematic draw with a seeded
           random start over the time-ordered candidates of each state, so
           the frames spread over runs and visits), read them, shift every
           molecule back into the box as a whole; write frames.npz
           (cytherea frame format) and frames.json (run, DCD frame, time,
           phi/psi, state, core visit of every frame)
  configs  N shard configs (mode shoot.ensemble, FixedLag(K_MAX tau), phi/psi
           every 1 ps, SHOTS_PER_FRAME shots per frame, the reference
           propagator), frames dealt round-robin, one store per shard: run
           them in parallel with `cytherea run` under MPS + core pinning
  export   on each owning host, export completed shard stores to JSONL;
           analyze can read these exports from either host
  analyze  read every shard store: shot end states at k tau (label_shot),
           row-normalised core-start T(tau) clustered by frame and, for
           comparison, by core visit; ITS (reversible MLE, timescales only);
           the shots' own CK (ck_test_shots, the reference's k list);
           element-wise comparison with the reference contract T; IC
           rejection report; frame/visit spread
  pes      14.1: pes_consistency_suite on start frames (CUDA mixed, sampled
           mode, finite differences on the solute)

Every shot's first tau is a tau shot (14.2); the whole shot gives T(k tau)
for k up to K_MAX (14.3). See the plan's Task 14 for the protocol.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ala2_common as C  # noqa: E402

TAU_PS = 100.0          # 14b lag = the reference's shoot_lag_ps (analyze checks the contract)
K_MAX = 55              # shots run K_MAX tau = 5.5 ns ~ 2 t2 (CK horizon)
N_PER_STATE = 50
SHOTS_PER_FRAME = 10
SKIP_PS = 10_000.0      # burn-in of every reference run (as the 14a analysis)
STAGE = "a1_long"
SEED = 20261003

REF_RUNS = ["runs/ala2_ref"] + [f"runs/ala2_par/r{i:02d}" for i in range(1, 9)]


# --------------------------------------------------------------------------- frames

def molecules(topology) -> list[np.ndarray]:
    """Atom indices of every molecule (connected component of the bond graph)."""
    n = topology.getNumAtoms()
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in topology.bonds():
        ra, rb = root(a.index), root(b.index)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [np.array(g, dtype=np.int64) for g in sorted(groups.values(), key=lambda g: g[0])]


def wrap_molecules(x: np.ndarray, box_lengths: np.ndarray, mols: list[np.ndarray]) -> np.ndarray:
    """Shift every molecule by whole box vectors so that its first atom lies in
    [0, L) (orthorhombic box); molecules stay whole."""
    x = np.array(x, dtype=np.float64)
    L = np.asarray(box_lengths, dtype=np.float64)
    for m in mols:
        x[m] -= np.floor(x[m[0]] / L) * L
    return x


def _visits(core: np.ndarray) -> np.ndarray:
    """Core-visit index of every record: TBA label runs numbered from 0; -1
    before the first core entry."""
    tba, n_drop = C.transition_based_assignment(core)
    out = np.full(core.size, -1, dtype=np.int64)
    if tba.size:
        out[n_drop:] = np.concatenate([[0], np.cumsum(tba[1:] != tba[:-1])])
    return out


def select_frames(runs, skip_ps: float, n_per_state: int, seed: int, dcd_counts: dict | None = None) -> list[dict]:
    """The start frames (module docstring). ``dcd_counts`` (run dir name -> number
    of DCD frames) replaces reading the DCD headers (tests)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    cand: dict[int, list[dict]] = {s: [] for s in range(C.N_STATES)}
    for run in map(Path, runs):
        fi = C.FrameIndex.from_run(run)
        C.check_obs_interval(fi.phipsi_interval_steps * fi.timestep_ps)
        rec = C.load_phipsi(run / "phipsi.bin")
        core = C.core_labels(rec["phi"], rec["psi"])
        visit = _visits(core)
        n_dcd = dcd_counts[run.name] if dcd_counts is not None else C.dcd_n_frames(run / "traj.dcd")
        k = np.arange(n_dcd)
        r = fi.dcd_frame_to_record(k)
        ok = r < rec.size
        k, r = k[ok], r[ok]
        t = fi.step_to_time_ps(fi.record_to_step(r))
        keep = (t >= skip_ps) & (core[r] >= 0)
        for kk, rr, tt in zip(k[keep], r[keep], t[keep]):
            s = int(core[rr])
            cand[s].append({"run": run.name, "run_dir": str(run), "dcd_frame": int(kk), "record": int(rr),
                            "time_ps": float(tt), "phi": float(rec["phi"][rr]), "psi": float(rec["psi"][rr]),
                            "state": s, "state_name": C.STATE_NAMES[s], "visit": int(visit[rr])})
    out = []
    for s in range(C.N_STATES):
        c = cand[s]
        if len(c) < n_per_state:
            raise ValueError(f"state {s} ({C.STATE_NAMES[s]}) has only {len(c)} core frames after the burn-in, "
                             f"{n_per_state} requested")
        step = len(c) / n_per_state
        u = rng.random() * step
        for i in range(n_per_state):
            out.append(dict(c[int(math.floor(u + i * step))], frame_id=len(out)))
    return out


def build_frames(selection: list[dict], topology_pdb: str, temperature_K: float):
    """EnsembleFrames (nm, (n, 3); box (3, 3)) of the selected DCD frames."""
    import openmm.app as app

    from cytherea.ic.frames import EnsembleFrame

    top = app.PDBFile(str(topology_pdb)).topology
    mols = molecules(top)
    ref = f"pdb-sha256:{C.sha256_file(topology_pdb)}"
    frames = []
    for f in selection:
        x, L = C.read_dcd_frame(Path(f["run_dir"]) / "traj.dcd", f["dcd_frame"])
        if L is None:
            raise ValueError(f"{f['run_dir']}: DCD without a unit cell")
        if x.shape[0] != top.getNumAtoms():
            raise ValueError(f"{f['run_dir']}: {x.shape[0]} atoms in the DCD, {top.getNumAtoms()} in the topology")
        frames.append(EnsembleFrame(coordinates=wrap_molecules(x, L, mols), box=np.diag(L), topology_ref=ref,
                                    temperature=temperature_K, weight=1.0,
                                    source_id=f"{f['run']}:dcd{f['dcd_frame']}", frame_id=f["frame_id"],
                                    time=f["time_ps"]))
    return frames


# --------------------------------------------------------------------------- analysis

def end_states(phi_rad, psi_rad, start_label: int, ks, tau_ps: float, dt_obs_ps: float) -> np.ndarray:
    """State of a shot at t = k tau for every k of `ks`: `ala2_common.label_shot`
    (TBA seeded with the start core, phi/psi every 1 ps; element 0 = start)."""
    lab = C.label_shot(np.degrees(phi_rad), np.degrees(psi_rad), start_label, dt_obs_ps)
    idx = [int(round(k * tau_ps / dt_obs_ps)) for k in ks]
    if max(idx) >= lab.size:
        raise ValueError(f"shot too short: {lab.size} observations, k tau = {max(ks) * tau_ps} ps needs {max(idx) + 1}")
    return lab[idx]


def _t_summary(est) -> dict:
    return {"matrix": est.T.tolist(), "ci95_lo": est.ci_low.tolist(), "ci95_hi": est.ci_high.tolist(),
            "n_eff": None if est.n_eff is None else np.asarray(est.n_eff).tolist()}


def analyze_shots(start, ends, ks, tau_ps: float, frame_ids, visit_ids, n_boot: int = 500, seed: int = 1,
                  tau_only: dict | None = None) -> dict:
    """T(tau) (row-normalised, frame clusters; also visit clusters), ITS from the
    reversible MLE and from the row-normalised T, and the shots' own CK.

    ``start, ends, frame_ids, visit_ids``: the long shots (state at every k of
    `ks`). ``tau_only`` (optional): shots that only ran tau, as a dict with
    ``start, end, frame_ids, visit_ids``. T(tau), the ITS and the CK's T(tau)
    use every shot's first tau; the CK's T(k tau), k >= 2, the long shots only
    (user decision 2026-10-04, A1 14.3). With ``tau_only`` the T(tau) of each
    part is also reported on its own (``T_parts``; e.g. GPU long shots vs CPU
    tau shots)."""
    from cytherea.estimate import ck_test_shots, estimate_T

    start, ends = np.asarray(start), np.asarray(ends)
    ks = [int(k) for k in ks]
    long_start, long_end1, long_fids = start, ends[:, ks.index(1)], np.asarray(frame_ids)
    rng = np.random.Generator(np.random.PCG64(seed))
    n = C.N_STATES
    parts = {}
    if tau_only is not None:
        t_start, t_end = np.asarray(tau_only["start"]), np.asarray(tau_only["end"])
        parts["long_first_tau"] = (long_start, long_end1, long_fids)
        parts["tau_only"] = (t_start, t_end, np.asarray(tau_only["frame_ids"]))
        start1 = np.concatenate([long_start, t_start])
        end1 = np.concatenate([long_end1, t_end])
        fids1 = np.concatenate([long_fids, parts["tau_only"][2]])
        vids1 = np.concatenate([np.asarray(visit_ids), np.asarray(tau_only["visit_ids"])])
    else:
        start1, end1, fids1, vids1 = long_start, long_end1, long_fids, np.asarray(visit_ids)
    w = np.ones(start1.size)
    est = estimate_T(start1, end1, w, n, tau_ps, reversible=False, n_boot=n_boot, rng=rng, frame_ids=fids1)
    by_visit = estimate_T(start1, end1, w, n, tau_ps, reversible=False, n_boot=n_boot, rng=rng, frame_ids=vids1)
    rev = estimate_T(start1, end1, w, n, tau_ps, reversible=True, n_boot=n_boot, rng=rng, frame_ids=fids1)
    T_parts = {name: _t_summary(estimate_T(a, b, np.ones(a.size), n, tau_ps, reversible=False, n_boot=n_boot,
                                           rng=rng, frame_ids=c))
               for name, (a, b, c) in parts.items()}
    ck_tau = None if tau_only is None else {"start": parts["tau_only"][0], "end": parts["tau_only"][1],
                                            "frame_ids": parts["tau_only"][2]}
    ck = ck_test_shots(start, ends, ks, n, n_boot, rng, frame_ids=frame_ids, tau_only=ck_tau)
    k1 = ks.index(1)
    rows = []
    for m, k in enumerate(ks):
        Tk = np.zeros((n, n))
        np.add.at(Tk, (start, ends[:, m]), 1.0)
        Tk /= Tk.sum(axis=1, keepdims=True)
        T1 = np.zeros((n, n))
        np.add.at(T1, (start1, end1), 1.0)
        T1 /= T1.sum(axis=1, keepdims=True)
        rows.append({"k": k, "lag_ps": k * tau_ps,
                     "predicted_stay": np.diag(np.linalg.matrix_power(T1, k)).tolist(),
                     "estimated_stay": np.diag(Tk).tolist()})
    assert rows[k1]["k"] == 1
    return {
        "n_shots": int(start1.size), "n_long_shots": int(start.size), "n_frames": int(np.unique(fids1).size),
        "n_visits": int(np.unique(vids1).size), "tau_ps": tau_ps,
        "shots_per_state": np.bincount(start1, minlength=n).tolist(),
        "T": _t_summary(est),
        "T_parts": T_parts,
        "T_by_visit_cluster": _t_summary(by_visit),
        "its_ps": {"t2": float(rev.its[0]), "t3": float(rev.its[1]),
                   "t2_ci95": [float(rev.its_ci_low[0]), float(rev.its_ci_high[0])],
                   "t3_ci95": [float(rev.its_ci_low[1]), float(rev.its_ci_high[1])],
                   "estimator": "reversible MLE of the core-start counts (timescales only)",
                   "row_normalised": [float(x) for x in est.its]},
        "ck": {"passed": bool(ck.passed), "max_dev": float(ck.max_dev), "ks": ks, "horizon_ps": max(ks) * tau_ps,
               "n_boot": ck.n_boot, "T_tau_from": "all shots" if tau_only is not None else "long shots",
               "test": "ck_test_shots, frame-clustered bootstrap stratified by start state",
               "rows": rows},
    }


def compare_with_reference(res: dict, ref: dict) -> dict:
    """Element-wise comparison with the reference contract T and the 14.2 check."""
    c = ref["contract_14b"]
    msm = ref["core_start"]["msm"]
    if abs(c["shoot_lag_ps"] - res["tau_ps"]) > 1e-9 or abs(c["obs_interval_ps"] - C.OBS_INTERVAL_PS) > 1e-9:
        raise ValueError(f"the reference contract (tau {c['shoot_lag_ps']} ps) does not match these shots")
    T_ref = np.array(msm["transition_matrix_row_normalised"])
    lo_ref = np.array(msm["transition_matrix_row_normalised_ci95_lo"])
    hi_ref = np.array(msm["transition_matrix_row_normalised_ci95_hi"])
    T, lo, hi = (np.array(res["T"][k]) for k in ("matrix", "ci95_lo", "ci95_hi"))
    its_ref = next(r for r in ref["core_start"]["implied_timescales"] if abs(r["lag_ps"] - res["tau_ps"]) < 1e-9)
    t2_lo, t2_hi = its_ref["ci95_lo_ps"][0], its_ref["ci95_hi_ps"][0]
    t2 = res["its_ps"]["t2"]
    return {
        "T_ref": T_ref.tolist(), "T_ref_ci95_lo": lo_ref.tolist(), "T_ref_ci95_hi": hi_ref.tolist(),
        "T_shoot_minus_ref": (T - T_ref).tolist(),
        "intervals_overlap": ((lo <= hi_ref) & (lo_ref <= hi)).tolist(),
        "ref_in_shoot_ci": ((lo <= T_ref) & (T_ref <= hi)).tolist(),
        "t2_ref_ps": its_ref["timescales_ps"][0], "t2_ref_ci95_ps": [t2_lo, t2_hi],
        "t3_ref_ps": its_ref["timescales_ps"][1],
        "t3_ref_ci95_ps": [its_ref["ci95_lo_ps"][1], its_ref["ci95_hi_ps"][1]],
        "acceptance_14_2": {"passed": bool(t2_lo <= t2 <= t2_hi), "t2_shoot_ps": t2,
                            "rule": "shooting t2 inside the reference core-start t2 95% CI at tau"},
    }


def ic_report(records) -> dict:
    """14.5: IC rejections (accepted shots' rejected attempts) and their reasons."""
    reasons: dict[str, int] = {}
    n_rej = 0
    for r in records:
        for a in r.ic_validity.get("rejected_attempts", []) or []:
            n_rej += 1
            for why in a.get("reasons", []):
                reasons[why] = reasons.get(why, 0) + 1
    n = len(records)
    return {"n_accepted": n, "n_rejected_attempts": n_rej,
            "rejection_rate": n_rej / (n + n_rej) if n + n_rej else float("nan"),
            "reasons": reasons, "acceptance_14_5": {"passed": bool(n and n_rej / (n + n_rej) < 0.01)}}


def spread_report(meta: list[dict]) -> dict:
    out = {}
    for s in range(C.N_STATES):
        fs = [f for f in meta if f["state"] == s]
        out[C.STATE_NAMES[s]] = {"n_frames": len(fs), "n_runs": len({f["run"] for f in fs}),
                                 "n_visits": len({(f["run"], f["visit"]) for f in fs})}
    return out


# --------------------------------------------------------------------------- commands

def cmd_frames(a) -> None:
    from cytherea.ic.frames import save_frames

    sel = select_frames(a.runs, a.skip_ps, a.n_per_state, a.seed)
    frames = build_frames(sel, a.topology, C.TEMPERATURE_K)
    a.out.mkdir(parents=True, exist_ok=True)
    save_frames(a.out / "frames.npz", frames)
    meta = {"runs": [str(Path(r).resolve()) for r in a.runs], "skip_ps": a.skip_ps, "n_per_state": a.n_per_state,
            "seed": a.seed, "topology": str(Path(a.topology).resolve()), "frames": sel,
            "spread": spread_report(sel)}
    (a.out / "frames.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["spread"], indent=1))


def shard_config(frames_npz: Path, system_dir: Path, frame_ids: list[int], store: Path, phi_idx, psi_idx,
                 k_max: int, shots_per_frame: int, seed: int, stage: str) -> dict:
    return {
        "mode": "shoot.ensemble", "units": "openmm",
        "system": {"kind": "openmm", "system_xml": str(system_dir / "system.xml"),
                   "topology_pdb": str(system_dir / "topology.pdb")},
        "physics": {"kind": "openmm", "integrator": "langevin_middle", "dt": "2 fs", "temperature": "300 K",
                    "friction": "0.1 /ps", "constraints": "hbonds", "rigid_water": True, "platform": "CUDA",
                    "precision": "mixed", "purpose": "measurement"},
        "ic": {"kind": "frames", "path": str(frames_npz), "min_pair_dist": "0.5 angstrom"},
        "stop": {"kind": "fixed_lag", "tau": f"{k_max * TAU_PS:g} ps"},
        "observables": {"dt_obs": "1 ps", "items": [{"name": "phi", "kind": "dihedral", "atoms": list(phi_idx)},
                                                    {"name": "psi", "kind": "dihedral", "atoms": list(psi_idx)}]},
        "budget": {"shots_per_frame": shots_per_frame, "frames": frame_ids, "stage": stage},
        "seed": seed, "store_path": str(store),
    }


def cmd_configs(a) -> None:
    import openmm.app as app
    import yaml

    meta = json.loads((a.frames_dir / "frames.json").read_text())
    top = app.PDBFile(meta["topology"]).topology
    phi_idx, psi_idx = C.phi_psi_indices(top)
    ids = [f["frame_id"] for f in meta["frames"]]
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(a.shards):
        cfg = shard_config((a.frames_dir / "frames.npz").resolve(), a.system_dir.resolve(), ids[i::a.shards],
                           (a.out / f"shard{i:02d}.sqlite").resolve(), phi_idx, psi_idx, a.k_max,
                           a.shots_per_frame, a.seed, a.stage)
        (a.out / f"shard{i:02d}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{a.shards} configs in {a.out}: {len(ids)} frames x {a.shots_per_frame} shots x {a.k_max * TAU_PS / 1000:g} ns "
          f"= {len(ids) * a.shots_per_frame * a.k_max * TAU_PS / 1e6:.2f} us")


def iter_shots(path):
    """Read owned SQLite stores or portable, full-record JSONL exports."""
    from cytherea.store import ShotRecord, Store

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open() as fh:
            for line in fh:
                rec = ShotRecord(**json.loads(line))
                if rec.kind == "shot":
                    yield rec
    elif path.suffix == ".sqlite":
        yield from Store(path).iter(kind="shot")
    else:
        raise ValueError(f"unsupported shot file: {path}")


def export_records(store, out: Path, stage: str) -> int:
    """Run on the store's owning host after its writer finishes.

    Replace the export only after every record has been written successfully.
    Keep all ShotRecord fields, including provenance and nonfinite observables.
    """
    out = Path(out)
    if Path(store).suffix != ".sqlite" or out.suffix != ".jsonl":
        raise ValueError("export requires a .sqlite source and .jsonl destination")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    n = 0
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=out.parent, delete=False) as fh:
            temporary = Path(fh.name)
            for rec in iter_shots(store):
                if rec.key.get("stage") == stage:
                    fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                    n += 1
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return n


def cmd_export(a) -> None:
    stems = [p.stem for p in a.stores]
    if len(stems) != len(set(stems)):
        raise ValueError("store filenames must have distinct stems in one export directory")
    counts = {str(p): export_records(p, a.out / (p.stem + ".jsonl"), a.stage) for p in a.stores}
    print(json.dumps({"stage": a.stage, "n_shots": sum(counts.values()), "stores": counts}, indent=1))


def load_records(stores, stage: str):
    out = []
    seen = set()
    for p in stores:
        for r in iter_shots(p):
            if r.key.get("stage") != stage:
                continue
            if r.key_digest in seen:
                raise ValueError(f"duplicate shot {r.key_digest} in {p}; use each shard only once")
            seen.add(r.key_digest)
            out.append(r)
    return out


def shot_files(directory):
    return sorted(Path(directory).glob("*.sqlite")) + sorted(Path(directory).glob("*.jsonl"))


def _shot_arrays(recs, by_id, visit_id, ks):
    start, ends, fids, vids = [], [], [], []
    for r in sorted(recs, key=lambda r: r.key_digest):
        f = by_id[r.frame_id]
        o = r.observables
        start.append(f["state"])
        ends.append(end_states(o["phi"], o["psi"], f["state"], ks, TAU_PS, C.OBS_INTERVAL_PS))
        fids.append(r.frame_id)
        vids.append(visit_id[(f["run"], f["visit"])])
    return np.array(start), np.array(ends), np.array(fids), np.array(vids)


def cmd_analyze(a) -> None:
    meta = json.loads((a.frames_dir / "frames.json").read_text())
    by_id = {f["frame_id"]: f for f in meta["frames"]}
    visit_id = {v: i for i, v in enumerate(sorted({(f["run"], f["visit"]) for f in meta["frames"]}))}
    stores = shot_files(a.shots_dir)
    recs = load_records(stores, a.stage)
    tau_stores = shot_files(a.tau_dir) if a.tau_dir else []
    tau_recs = load_records(tau_stores, a.tau_stage) if tau_stores else []
    every = recs + tau_recs
    nonfinite = [r for r in every if r.stop_reason == "nonfinite"]
    if any(r.stop_reason not in ("fixed_lag", "nonfinite") for r in every):
        raise ValueError("unexpected stop reasons: " + str({r.stop_reason for r in every}))
    ks = sorted(set(a.ck_ks))
    start, ends, fids, vids = _shot_arrays([r for r in recs if r.stop_reason == "fixed_lag"], by_id, visit_id, ks)
    tau_only = None
    if tau_recs:
        ts, te, tf, tv = _shot_arrays([r for r in tau_recs if r.stop_reason == "fixed_lag"], by_id, visit_id, [1])
        tau_only = {"start": ts, "end": te[:, 0], "frame_ids": tf, "visit_ids": tv}
    res = analyze_shots(start, ends, ks, TAU_PS, fids, vids, n_boot=a.n_boot, seed=a.seed, tau_only=tau_only)
    res["n_nonfinite"] = len(nonfinite)
    res["valid"] = not nonfinite
    res["stores"] = {"long": [str(p) for p in stores], "tau_only": [str(p) for p in tau_stores]}
    res["spread"] = meta["spread"]
    res["ic"] = ic_report(every)
    res["acceptance_14_3"] = {"passed": res["ck"]["passed"]}
    if a.reference:
        res["reference"] = compare_with_reference(res, json.loads(Path(a.reference).read_text()))
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)


def cmd_pes(a) -> None:
    import openmm
    import openmm.app as app

    from cytherea.backends.openmm_backend import OpenMMBackend
    from cytherea.backends.base import PhysicsConfig
    from cytherea.backends.pes_suite import pes_consistency_suite
    from cytherea.ic.frames import load_frames

    system = openmm.XmlSerializer.deserialize((a.system_dir / "system.xml").read_text())
    top = app.PDBFile(str(a.system_dir / "topology.pdb")).topology
    cfg = PhysicsConfig(integrator="langevin_middle", dt_ps=0.002, temperature_K=300.0, friction_per_ps=0.1,
                        constraints="hbonds", rigid_water=True, platform="CUDA", precision="mixed",
                        deterministic_forces=True, purpose="measurement")
    backend = OpenMMBackend(system, top, cfg)
    frames = load_frames(a.frames_dir / "frames.npz")
    probes = [np.asarray(frames[i].coordinates) for i in np.linspace(0, len(frames) - 1, a.n_probes).astype(int)]
    solute = [atom.index for atom in top.atoms() if atom.residue.name not in ("HOH", "WAT", "NA", "CL")]
    rep = pes_consistency_suite(backend, probes, mode="sampled", fd_atom_groups={"solute": solute}, precision="mixed",
                                check_invariance=False)
    out = {"passed": rep.passed, "reasons": rep.reasons, "fd_max_rel_err": rep.fd_max_rel_err,
           "fd_noise_rel": rep.fd_noise_rel, "repeat_bitwise": rep.repeat_bitwise, "tolerances": rep.tolerances,
           "fd_max_atom_rel_err": rep.fd_max_atom_rel_err, "fd_noise_floor_rel": rep.fd_noise_floor_rel,
           "repeat_max_rel_err": rep.repeat_max_rel_err, "fd_skipped_cutoff": rep.fd_skipped_cutoff,
           "fd_cutoffs": rep.fd_cutoffs, "precision": rep.precision, "mode": rep.mode,
           "n_probes": len(probes), "n_fd_coords": rep.n_fd_coords, "solute_atoms": solute}
    print(json.dumps(out, indent=1, default=str))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("frames")
    s.add_argument("--runs", nargs="+", default=REF_RUNS)
    s.add_argument("--topology", default="runs/ala2_system/topology.pdb")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--skip-ps", type=float, default=SKIP_PS)
    s.add_argument("--n-per-state", type=int, default=N_PER_STATE)
    s.add_argument("--seed", type=int, default=SEED)
    s = sub.add_parser("configs")
    s.add_argument("--frames-dir", type=Path, required=True)
    s.add_argument("--system-dir", type=Path, default=Path("runs/ala2_system"))
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shards", type=int, default=8)
    s.add_argument("--k-max", type=int, default=K_MAX)
    s.add_argument("--shots-per-frame", type=int, default=SHOTS_PER_FRAME)
    s.add_argument("--stage", default=STAGE)
    s.add_argument("--seed", type=int, default=SEED)
    s = sub.add_parser("export", help="export completed stores on their owning host")
    s.add_argument("--stores", type=Path, nargs="+", required=True,
                   help="explicit list of SQLite shards owned by this host")
    s.add_argument("--out", type=Path, required=True, help="directory for portable JSONL exports")
    s.add_argument("--stage", default=STAGE)
    s = sub.add_parser("analyze")
    s.add_argument("--frames-dir", type=Path, required=True)
    s.add_argument("--shots-dir", type=Path, required=True, help="long shots (*.sqlite or exported *.jsonl)")
    s.add_argument("--tau-dir", type=Path, default=None, help="tau-only shots (*.sqlite or exported *.jsonl), optional")
    s.add_argument("--reference", default="runs/ala2_par/analysis/analysis.json")
    s.add_argument("--stage", default=STAGE)
    s.add_argument("--tau-stage", default="a1_tau")
    s.add_argument("--ck-ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 9, 13, 18, 27, 38, 55])
    s.add_argument("--n-boot", type=int, default=500)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("pes")
    s.add_argument("--frames-dir", type=Path, required=True)
    s.add_argument("--system-dir", type=Path, default=Path("runs/ala2_system"))
    s.add_argument("--n-probes", type=int, default=5)
    s.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    {"frames": cmd_frames, "configs": cmd_configs, "export": cmd_export,
     "analyze": cmd_analyze, "pes": cmd_pes}[a.cmd](a)


if __name__ == "__main__":
    main()
