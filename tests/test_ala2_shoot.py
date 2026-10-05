"""Task 14b (A1 shooting) helpers: DCD frame reading, molecule wrapping,
core-frame selection from the reference runs, shot end-state labelling and
the analysis of multi-lag shots (T(tau), ITS, the shots' own CK).
Everything here is fast and synthetic; the campaign itself runs on the GPU."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "alanine_dipeptide"
sys.path.insert(0, str(EX))

import ala2_common as C  # noqa: E402
import shoot_a1 as S  # noqa: E402

# a phi/psi point inside each core (degrees) and one outside all cores
CORE_CENTRE = {0: (-80.0, 150.0), 1: (-80.0, -45.0), 2: (60.0, 45.0)}
BRIDGE = (-80.0, 50.0)


def _toy_topology(n_waters=3):
    import openmm.app as app

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ALA", chain)
    atoms = [top.addAtom(f"C{i}", app.element.carbon, res) for i in range(3)]
    top.addBond(atoms[0], atoms[1])
    top.addBond(atoms[1], atoms[2])
    for _ in range(n_waters):
        r = top.addResidue("HOH", chain)
        o = top.addAtom("O", app.element.oxygen, r)
        for name in ("H1", "H2"):
            top.addBond(o, top.addAtom(name, app.element.hydrogen, r))
    return top


def test_read_dcd_frame_round_trip(tmp_path):
    import openmm.app as app
    import openmm.unit as u
    from openmm import Vec3

    top = _toy_topology(1)
    box = (Vec3(2.0, 0, 0), Vec3(0, 2.5, 0), Vec3(0, 0, 3.0)) * u.nanometer
    top.setPeriodicBoxVectors(box)
    rng = np.random.Generator(np.random.PCG64(0))
    frames = [rng.normal(scale=20.0, size=(6, 3)) for _ in range(3)]  # nm, far outside the box
    with open(tmp_path / "t.dcd", "wb") as fh:
        dcd = app.DCDFile(fh, top, 0.002, firstStep=5000, interval=5000)
        for x in frames:
            dcd.writeModel([Vec3(*p) for p in x] * u.nanometer, periodicBoxVectors=box)
    assert C.dcd_n_frames(tmp_path / "t.dcd") == 3
    for k, x in enumerate(frames):
        xyz, lengths = C.read_dcd_frame(tmp_path / "t.dcd", k)
        assert xyz.dtype == np.float64 and xyz.shape == (6, 3)
        assert np.allclose(xyz, x, rtol=1e-6, atol=1e-5)  # float32 Angstrom on disk
        assert np.allclose(lengths, [2.0, 2.5, 3.0])
    with pytest.raises(IndexError):
        C.read_dcd_frame(tmp_path / "t.dcd", 3)


def test_wrap_molecules_keeps_molecules_whole():
    top = _toy_topology(3)
    mols = S.molecules(top)
    assert sorted(len(m) for m in mols) == [3, 3, 3, 3] and sorted(np.concatenate(mols).tolist()) == list(range(12))
    rng = np.random.Generator(np.random.PCG64(1))
    box = np.array([2.0, 2.5, 3.0])
    x = np.concatenate([rng.normal(size=3) * 30.0 + rng.normal(scale=0.1, size=(3, 3)) for _ in mols])
    w = S.wrap_molecules(x, box, mols)
    for m in mols:
        assert np.allclose(np.diff(w[m], axis=0), np.diff(x[m], axis=0), atol=1e-9)  # rigid shift
        assert np.all((w[m[0]] >= 0) & (w[m[0]] < box))
        shift = (w[m] - x[m]) / box
        assert np.allclose(shift, np.round(shift[0]), atol=1e-9)  # whole box vectors only


def _write_run(path: Path, states: list[int], dcd_every: int = 10):
    """A ref_long-style run dir: phipsi.bin (1 ps records), run.json; no DCD."""
    path.mkdir()
    phi = np.array([CORE_CENTRE[s][0] if s >= 0 else BRIDGE[0] for s in states], dtype=np.float32)
    psi = np.array([CORE_CENTRE[s][1] if s >= 0 else BRIDGE[1] for s in states], dtype=np.float32)
    rec = np.zeros(len(states), dtype=C.PHIPSI_DTYPE)
    rec["step"] = np.arange(len(states)) * 500
    rec["phi"], rec["psi"] = phi, psi
    rec.tofile(path / "phipsi.bin")
    (path / "run.json").write_text(json.dumps({"phipsi_interval_steps": 500, "dcd_interval_steps": 500 * dcd_every,
                                               "timestep_ps": 0.002}))


def test_select_frames_core_only_spread_and_reproducible(tmp_path):
    # run a: 0 for 300 ps, bridge 20 ps, 1 for 300 ps, 0 again 300 ps; run b: 2 for 400, 0 for 200
    a = [0] * 300 + [-1] * 20 + [1] * 300 + [0] * 300
    b = [2] * 400 + [0] * 200
    _write_run(tmp_path / "a", a)
    _write_run(tmp_path / "b", b)
    runs = [tmp_path / "a", tmp_path / "b"]
    sel = S.select_frames(runs, skip_ps=100.0, n_per_state=6, seed=3, dcd_counts={"a": 91, "b": 59})
    assert sorted({f["frame_id"] for f in sel}) == list(range(18))
    for f in sel:
        run_states = a if f["run"] == "a" else b
        r = f["record"]
        assert r == 10 * (f["dcd_frame"] + 1) and f["time_ps"] == r * 1.0 >= 100.0
        assert run_states[r] == f["state"] >= 0  # raw core label of the frame's own record
        assert C.core_labels(f["phi"], f["psi"]) == f["state"]
    by_state = {s: [f for f in sel if f["state"] == s] for s in range(3)}
    assert all(len(v) == 6 for v in by_state.values())
    # state 0 lives in two runs and in two visits of run a: the systematic draw reaches all of them
    assert {(f["run"], f["visit"]) for f in by_state[0]} == {("a", 0), ("a", 2), ("b", 1)}
    assert S.select_frames(runs, 100.0, 6, 3, dcd_counts={"a": 91, "b": 59}) == sel
    assert S.select_frames(runs, 100.0, 6, 4, dcd_counts={"a": 91, "b": 59}) != sel
    with pytest.raises(ValueError, match="state 1"):
        S.select_frames(runs, 100.0, 40, 3, dcd_counts={"a": 91, "b": 59})  # only 30 core frames of state 1


def test_end_states_at_multiples_of_tau():
    # start in 0, bridge at 150-160 ps, enter 1 at 160 ps, back to 0 at 320 ps (TBA keeps 1 on the bridge)
    states = [0] * 150 + [-1] * 10 + [1] * 160 + [0] * 181
    phi = np.radians([CORE_CENTRE[s][0] if s >= 0 else BRIDGE[0] for s in states])
    psi = np.radians([CORE_CENTRE[s][1] if s >= 0 else BRIDGE[1] for s in states])
    got = S.end_states(phi, psi, start_label=0, ks=(1, 2, 3, 5), tau_ps=100.0, dt_obs_ps=1.0)
    assert got.tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError, match="1 ps"):
        S.end_states(phi[::2], psi[::2], 0, (1,), 100.0, 2.0)
    with pytest.raises(ValueError, match="too short"):
        S.end_states(phi, psi, 0, (1, 6), 100.0, 1.0)


def _chain_paths(P, n_frames, shots, n_steps, rng):
    """Synthetic shots: states every tau-step of a Markov chain (states 0..2),
    converted to phi/psi series at 1 ps with tau = 10 ps (core centres only)."""
    cum = np.cumsum(P, axis=1)
    out = []
    fid = 0
    for s in range(3):
        for _ in range(n_frames):
            for _ in range(shots):
                x, path = s, [s]
                for _ in range(n_steps):
                    x = int((rng.random() > cum[x]).sum())
                    path.append(x)
                fine = np.repeat(path, 10)[: 10 * n_steps + 1]
                phi = np.radians([CORE_CENTRE[q][0] for q in fine])
                psi = np.radians([CORE_CENTRE[q][1] for q in fine])
                out.append((fid, s, phi, psi))
            fid += 1
    return out


def test_analyze_recovers_T_and_passes_ck_on_a_markov_chain():
    # reversible: pi = (0.6, 0.3, 0.1), symmetric fluxes 0.03 / 0.004 / 0.002
    P = np.array([[0.94 + 1 / 300, 0.05, 0.004 / 0.6], [0.10, 0.90 - 0.002 / 0.3, 0.002 / 0.3], [0.04, 0.02, 0.94]])
    rng = np.random.Generator(np.random.PCG64(9))
    shots = _chain_paths(P, 40, 5, 8, rng)
    ks = (1, 2, 4, 8)
    start = np.array([s for _, s, _, _ in shots])
    ends = np.array([S.end_states(phi, psi, s, ks, 10.0, 1.0) for _, s, phi, psi in shots])
    fids = np.array([f for f, _, _, _ in shots])
    res = S.analyze_shots(start, ends, ks, tau_ps=10.0, frame_ids=fids, visit_ids=fids // 4, n_boot=200, seed=1)
    T = np.array(res["T"]["matrix"])
    lo, hi = np.array(res["T"]["ci95_lo"]), np.array(res["T"]["ci95_hi"])
    assert np.all((lo <= P + 0.02) & (P - 0.02 <= hi))
    assert np.allclose(T.sum(axis=1), 1.0)
    assert res["ck"]["passed"] is True and res["ck"]["ks"] == list(ks)
    t2_true = -10.0 / math.log(sorted(np.abs(np.linalg.eigvals(P)))[-2])
    assert res["its_ps"]["t2_ci95"][0] <= t2_true <= res["its_ps"]["t2_ci95"][1]
    assert res["n_shots"] == 600 and res["n_frames"] == 120 and len(res["T"]["n_eff"]) == 3
    assert np.array(res["T_by_visit_cluster"]["matrix"]) == pytest.approx(T)  # same counts, other clusters


def test_analyze_combines_long_and_tau_only_shots():
    """T(tau) and ITS use every shot's first tau, the CK only the long shots;
    each part's T(tau) is reported on its own."""
    P = np.array([[0.94 + 1 / 300, 0.05, 0.004 / 0.6], [0.10, 0.90 - 0.002 / 0.3, 0.002 / 0.3], [0.04, 0.02, 0.94]])
    rng = np.random.Generator(np.random.PCG64(12))
    ks = (1, 2, 4, 8)
    long = _chain_paths(P, 20, 4, 8, rng)
    short = _chain_paths(P, 20, 6, 1, rng)
    ls = np.array([s for _, s, _, _ in long])
    le = np.array([S.end_states(phi, psi, s, ks, 10.0, 1.0) for _, s, phi, psi in long])
    lf = np.array([f for f, _, _, _ in long])
    ss = np.array([s for _, s, _, _ in short])
    se = np.array([S.end_states(phi, psi, s, (1,), 10.0, 1.0)[0] for _, s, phi, psi in short])
    sf = np.array([f for f, _, _, _ in short])
    res = S.analyze_shots(ls, le, ks, 10.0, lf, lf, n_boot=100, seed=2,
                          tau_only={"start": ss, "end": se, "frame_ids": sf, "visit_ids": sf})
    assert res["n_shots"] == 240 + 360 and res["n_long_shots"] == 240 and res["n_frames"] == 60
    assert set(res["T_parts"]) == {"long_first_tau", "tau_only"}
    C1 = np.zeros((3, 3))
    np.add.at(C1, (np.concatenate([ls, ss]), np.concatenate([le[:, 0], se])), 1.0)
    assert np.allclose(res["T"]["matrix"], C1 / C1.sum(axis=1, keepdims=True))
    assert res["ck"]["passed"] is True


def _export_record(digest="a", stage="a1_long", nonfinite=False):
    from cytherea.store import ShotRecord

    return ShotRecord(
        key_digest=digest * 64, key={"stage": stage, "shot_id": 0}, kind="shot",
        frame_id=0, origin_label=(0, 0), ic_validity={"ok": True},
        stop_rule_kind="fixed_lag", stop_reason="nonfinite" if nonfinite else "fixed_lag",
        event_time=None, physics_config_hash="physics", protocol_hash="protocol",
        backend_provenance={"platform": "CUDA"}, code_version="source-version",
        observables={"phi": [0.1, float("nan") if nonfinite else 0.2]},
        final_state_label=None, ic_meta={"frame_id": 0}, observables_thinned=False,
        warnings=["test warning"],
    )


def test_exports_from_two_hosts_can_be_analyzed_without_opening_sqlite(tmp_path, monkeypatch):
    import cytherea.store as store_module

    paths = []
    records = [_export_record("a"), _export_record("b")]
    for host, rec in zip(("host-a", "host-b"), records):
        monkeypatch.setattr(store_module, "_current_hostname", lambda host=host: host)
        db = tmp_path / f"{host}.sqlite"
        store = store_module.Store(db)
        store.append(rec)
        store.append(_export_record("c", stage="a1_tau"))
        out = tmp_path / "exports" / f"{host}.jsonl"
        assert S.export_records(db, out, "a1_long") == 1
        paths.append(out)

    def forbid_sqlite(*args, **kwargs):
        raise AssertionError("portable analysis must not open a Store")

    monkeypatch.setattr(store_module, "Store", forbid_sqlite)
    assert S.shot_files(tmp_path / "exports") == paths
    assert S.load_records(paths, "a1_long") == records


def test_export_rejects_foreign_store_and_preserves_previous_export(tmp_path, monkeypatch):
    import cytherea.store as store_module

    monkeypatch.setattr(store_module, "_current_hostname", lambda: "owner")
    db = tmp_path / "shard.sqlite"
    store_module.Store(db).append(_export_record())
    out = tmp_path / "shard.jsonl"
    S.export_records(db, out, "a1_long")
    previous = out.read_bytes()
    monkeypatch.setattr(store_module, "_current_hostname", lambda: "other-host")
    with pytest.raises(store_module.StoreOwnershipError):
        S.export_records(db, out, "a1_long")
    assert out.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard.jsonl", "shard.sqlite"]


def test_export_keeps_nonfinite_and_rejects_duplicate_shots(tmp_path):
    from cytherea.store import Store

    db = tmp_path / "shard.sqlite"
    rec = _export_record(nonfinite=True)
    Store(db).append(rec)
    out = tmp_path / "shard.jsonl"
    S.export_records(db, out, "a1_long")
    got = S.load_records([out], "a1_long")[0]
    assert got.stop_reason == "nonfinite"
    assert math.isnan(got.observables["phi"][1])
    assert got.physics_config_hash == rec.physics_config_hash
    assert got.protocol_hash == rec.protocol_hash
    with pytest.raises(ValueError, match="duplicate shot"):
        S.load_records([db, out], "a1_long")


def test_missing_export_source_does_not_create_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.export_records(tmp_path / "missing.sqlite", tmp_path / "out.jsonl", "a1_long")
    assert not list(tmp_path.iterdir())
