"""Task 13: config schema, units, mode dispatch and the CLI.

13.1 minimal configs of the four modes load and dispatch to their runner;
13.2 a missing field is a pydantic error naming its path; 13.3 units
("20 angstrom" -> 2.0 nm, a bare number under units: openmm -> error);
13.4 `cytherea resume` after a crash between shots behaves as test 7.4
(only missing keys run; records identical to an uninterrupted run); 13.5
key order (and unit spelling) does not change the config hash. Plus end-to-
end runs of the analytic shoot modes (plain and WE), prepare -> frames ->
shoot.ensemble on alanine dipeptide in vacuum (Reference platform), the
store/sidecar guards, `report`, and loading every example config.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import os
import signal
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from cytherea import cli
from cytherea.config import RunConfig, dispatch, load_config, parse_quantity, report
from cytherea.config import runners as R
from cytherea.config.runners import ResumeError
from cytherea.store import Store

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ units


def test_parse_quantity_units_and_errors():
    assert parse_quantity("20 angstrom", "length", "openmm") == pytest.approx(2.0)
    assert parse_quantity("2 fs", "time", "openmm") == pytest.approx(0.002)
    assert parse_quantity("0.1 /ps", "inverse_time", "openmm") == pytest.approx(0.1)
    assert parse_quantity("60 deg", "angle", "openmm") == pytest.approx(math.pi / 3)
    assert parse_quantity("1 kcal/mol", "energy", "openmm") == pytest.approx(4.184)
    assert parse_quantity("1.5e3 ps", "time", "openmm") == pytest.approx(1500.0)
    assert parse_quantity(0.5, "length", "reduced") == 0.5
    with pytest.raises(ValueError, match="explicit unit"):
        parse_quantity(20, "length", "openmm")
    with pytest.raises(ValueError, match="not a length unit"):
        parse_quantity("20 ps", "length", "openmm")
    with pytest.raises(ValueError, match="plain numbers"):
        parse_quantity("1 nm", "length", "reduced")
    with pytest.raises(ValueError):
        parse_quantity(True, "length", "reduced")
    with pytest.raises(ValueError, match="<number> <unit>"):
        parse_quantity("nm", "length", "openmm")


# ---------------------------------------------------------------- configs


def _analytic(tmp_path, **over) -> dict:
    cfg = {
        "mode": "shoot.ensemble",
        "units": "reduced",
        "system": {"kind": "analytic", "potential": {"name": "DoubleWell1D", "params": {"barrier": 2.0}}},
        "physics": {"kind": "analytic", "integrator": "overdamped", "dt": 1e-3, "kT": 1.0, "gamma": 1.0},
        "ic": {"kind": "points", "points": [[-0.3], [0.0], [0.3]]},
        "stop": {"kind": "absorbing_AB",
                 "A": {"name": "A", "all_of": [{"observable": "x", "le": -1.0}]},
                 "B": {"name": "B", "all_of": [{"observable": "x", "ge": 1.0}]},
                 "tau_persist": 0.0, "t_max": 50.0},
        "observables": {"dt_obs": 1e-3, "items": [{"name": "x", "kind": "coord", "index": 0}]},
        "budget": {"shots_per_frame": 4},
        "seed": 7,
        "store_path": str(tmp_path / "store.sqlite"),
    }
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def ala2_files(tmp_path_factory):
    import openmm
    import openmm.app as app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from openmmtools import testsystems
    t = testsystems.AlanineDipeptideVacuum()
    d = tmp_path_factory.mktemp("ala2")
    (d / "system.xml").write_text(openmm.XmlSerializer.serialize(t.system))
    with open(d / "ala2.pdb", "w") as fh:
        app.PDBFile.writeFile(t.topology, t.positions, fh)
    return d


def _openmm(files, tmp_path, **over) -> dict:
    cfg = {
        "mode": "shoot.ensemble",
        "units": "openmm",
        "system": {"kind": "openmm", "system_xml": str(files / "system.xml"), "topology_pdb": str(files / "ala2.pdb")},
        "physics": {"kind": "openmm", "integrator": "verlet", "dt": "1 fs", "temperature": "300 K",
                    "friction": "0 /ps", "constraints": "hbonds", "rigid_water": False, "platform": "Reference",
                    "precision": "double", "purpose": "measurement"},
        "ic": {"kind": "frames", "path": str(tmp_path / "frames.npz")},
        "stop": {"kind": "fixed_lag", "tau": "20 fs"},
        "observables": {"dt_obs": "10 fs", "items": [
            {"name": "phi", "kind": "dihedral", "atoms": [4, 6, 8, 14]},
            {"name": "psi", "kind": "dihedral", "atoms": [6, 8, 14, 16]}]},
        "budget": {"shots_per_frame": 1},
        "seed": 11,
        "store_path": str(tmp_path / "ala2.sqlite"),
    }
    cfg.update(over)
    return cfg


def _prepare(files, tmp_path) -> dict:
    cfg = _openmm(files, tmp_path, mode="prepare", ic={"kind": "structure"}, stop=None, observables=None,
                  store_path=None,
                  budget={"minimize": True, "equilibrate": "50 fs", "n_frames": 3, "frame_interval": "20 fs",
                          "output_frames": str(tmp_path / "frames.npz")})
    cfg["physics"] = dict(cfg["physics"], integrator="langevin_middle", friction="1 /ps", purpose="equilibration")
    return {k: v for k, v in cfg.items() if v is not None}


def _encounter(files, tmp_path, b="20 angstrom") -> dict:
    cfg = _openmm(files, tmp_path, mode="shoot.encounter",
                  ic={"kind": "encounter", "frames_A": "a.npz", "frames_B": "b.npz", "b": b,
                      "min_pair_dist": "3 angstrom"})
    cfg["observables"]["items"].append({"name": "r", "kind": "distance", "atoms": [0, 21]})
    cfg["stop"] = {"kind": "b_surface", "reaction": {"name": "bound", "all_of": [{"observable": "r", "le": "5 angstrom"}]},
                   "r_observable": "r", "q": "8 nm", "tau_persist": "1 ps", "t_max": "1 ns"}
    return cfg


def test_13_1_four_modes_load_and_dispatch(ala2_files, tmp_path):
    configs = {
        "shoot.ensemble": _analytic(tmp_path),
        "shoot.surface": _analytic(tmp_path, mode="shoot.surface"),
        "prepare": _prepare(ala2_files, tmp_path),
        "shoot.encounter": _encounter(ala2_files, tmp_path),
    }
    want = {"shoot.ensemble": R.run_shoot, "shoot.surface": R.run_shoot, "prepare": R.run_prepare,
            "shoot.encounter": R.run_encounter}
    for mode, data in configs.items():
        cfg = RunConfig.from_dict(data)
        assert cfg.mode == mode
        fn = dispatch(cfg)
        assert fn.func is want[mode] and fn.args == (cfg, False)


def test_13_2_missing_field_names_its_path(tmp_path):
    data = _analytic(tmp_path)
    del data["physics"]["dt"]
    with pytest.raises(ValidationError) as ei:
        RunConfig.from_dict(data)
    locs = [e["loc"] for e in ei.value.errors()]
    assert ("physics", "analytic", "dt") in locs
    assert "physics.analytic.dt" in str(ei.value)
    del data["seed"]
    with pytest.raises(ValidationError, match="seed"):
        RunConfig.from_dict(data)
    data = _analytic(tmp_path)
    data["budget"]["shots_per_frmae"] = 3  # typo: unknown keys are errors
    with pytest.raises(ValidationError, match="shots_per_frmae"):
        RunConfig.from_dict(data)


def test_13_3_units(ala2_files, tmp_path):
    cfg = RunConfig.from_dict(_encounter(ala2_files, tmp_path))
    assert cfg.ic.b == pytest.approx(2.0)
    assert cfg.stop.reaction.all_of[0].le == pytest.approx(0.5)  # threshold in the observable's dimension
    assert cfg.physics.dt == pytest.approx(0.001) and cfg.physics.temperature == 300.0
    with pytest.raises(ValidationError) as ei:
        RunConfig.from_dict(_encounter(ala2_files, tmp_path, b=20))
    assert ("ic", "encounter", "b") in [e["loc"] for e in ei.value.errors()]
    assert "explicit unit" in str(ei.value)
    # a threshold with the wrong dimension, and a bare number under units: openmm
    bad = _openmm(ala2_files, tmp_path, stop={"kind": "absorbing_AB",
                  "A": {"name": "A", "all_of": [{"observable": "phi", "le": "1 nm"}]},
                  "B": {"name": "B", "all_of": [{"observable": "phi", "ge": "60 deg"}]},
                  "tau_persist": "0 ps", "t_max": "1 ps"})
    with pytest.raises(ValidationError, match=r"stop\.A\.all_of\[0\]\.le.*not a angle unit"):
        RunConfig.from_dict(bad)
    bad["stop"]["A"]["all_of"][0]["le"] = -1.0
    with pytest.raises(ValidationError, match="explicit unit"):
        RunConfig.from_dict(bad)
    bad["stop"]["A"]["all_of"][0]["le"] = "-60 deg"
    ok = RunConfig.from_dict(bad)
    assert ok.stop.A.all_of[0].le == pytest.approx(-math.pi / 3)
    # reduced configs take plain numbers only
    data = _analytic(tmp_path)
    data["physics"]["dt"] = "1 fs"
    with pytest.raises(ValidationError, match="plain numbers"):
        RunConfig.from_dict(data)


def test_13_5_config_hash_ignores_key_order_and_unit_spelling(ala2_files, tmp_path):
    a = _encounter(ala2_files, tmp_path)
    b = {k: a[k] for k in reversed(list(a))}
    b["physics"] = {k: a["physics"][k] for k in reversed(list(a["physics"]))}
    p1, p2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    p1.write_text(yaml.safe_dump(a, sort_keys=False))
    p2.write_text(yaml.safe_dump(b, sort_keys=False))
    assert p1.read_text() != p2.read_text()
    h = load_config(p1).config_hash()
    assert load_config(p2).config_hash() == h
    c = copy.deepcopy(a)
    c["ic"]["b"], c["physics"]["dt"] = "2 nm", "0.001 ps"
    assert RunConfig.from_dict(c, base_dir=tmp_path).config_hash() == h
    c["ic"]["b"] = "21 angstrom"
    assert RunConfig.from_dict(c, base_dir=tmp_path).config_hash() != h


def test_mode_rules(ala2_files, tmp_path):
    # an analytic config under units: openmm fails on its bare numbers first ...
    with pytest.raises(ValidationError, match="explicit unit"):
        RunConfig.from_dict(_analytic(tmp_path, units="openmm"))
    # ... and on the system kind when every quantity carries a unit
    mixed = _openmm(ala2_files, tmp_path)
    mixed["system"] = _analytic(tmp_path)["system"]
    with pytest.raises(ValidationError, match="analytic system needs units: reduced"):
        RunConfig.from_dict(mixed)
    with pytest.raises(ValidationError, match="shoot.surface takes stop.kind"):
        RunConfig.from_dict(_analytic(tmp_path, mode="shoot.surface",
                                      stop={"kind": "fixed_lag", "tau": 1.0}))
    meas = _prepare(ala2_files, tmp_path)
    meas["physics"]["purpose"] = "measurement"
    with pytest.raises(ValidationError, match="purpose: equilibration"):
        RunConfig.from_dict(meas)
    eq = _analytic(tmp_path)
    eq["physics"]["purpose"] = "equilibration"
    with pytest.raises(ValidationError, match="purpose must be measurement"):
        RunConfig.from_dict(eq)
    with pytest.raises(ValidationError, match="unknown observable 'y'"):
        RunConfig.from_dict(_analytic(tmp_path, stop={"kind": "absorbing_AB",
                            "A": {"name": "A", "all_of": [{"observable": "y", "le": -1.0}]},
                            "B": {"name": "B", "all_of": [{"observable": "x", "ge": 1.0}]},
                            "tau_persist": 0.0, "t_max": 5.0}))
    with pytest.raises(ValidationError, match="the unit system is unknown"):
        RunConfig.model_validate(_analytic(tmp_path))


# ------------------------------------------------------------ end to end


def test_shoot_ensemble_analytic_run_resume_report(tmp_path):
    cfg = RunConfig.from_dict(_analytic(tmp_path))
    out = dispatch(cfg)()
    assert out["n_keys"] == 12 and out["n_failures"] == 0
    assert set(out["stop_reasons"]) <= {"A", "B"} and sum(out["stop_reasons"].values()) == 12
    side = json.loads(Path(R.sidecar_path(cfg.store_path)).read_text())
    assert side["config_hash"] == cfg.config_hash()
    with pytest.raises(ResumeError, match="use `cytherea resume`"):
        dispatch(cfg)()
    before = {r.key_digest: r for r in Store(cfg.store_path).iter()}
    again = dispatch(cfg, resume=True)()
    assert again["n_keys"] == 12
    assert {r.key_digest: r for r in Store(cfg.store_path).iter()} == before
    changed = RunConfig.from_dict(_analytic(tmp_path, seed=8))
    with pytest.raises(ResumeError, match="config differs"):
        dispatch(changed, resume=True)()
    rep = report(cfg.store_path)
    assert rep["n_records"] == 12 and rep["kinds"] == {"shot": 12} and rep["stages"] == {"main": 12}
    assert rep["config_hash"] == cfg.config_hash() and len(rep["protocol_hashes"]) == 1
    with pytest.raises(ResumeError, match="does not exist"):
        dispatch(RunConfig.from_dict(_analytic(tmp_path, store_path=str(tmp_path / "x.sqlite"))), resume=True)()


def test_shoot_ensemble_analytic_parallel_matches_serial(tmp_path):
    a = RunConfig.from_dict(_analytic(tmp_path, store_path=str(tmp_path / "s1.sqlite")))
    b = RunConfig.from_dict(_analytic(tmp_path, store_path=str(tmp_path / "s2.sqlite"),
                                      budget={"shots_per_frame": 4, "n_workers": 3}))
    dispatch(a)()
    dispatch(b)()
    ra = {r.key_digest: r for r in Store(a.store_path).iter()}
    rb = {r.key_digest: r for r in Store(b.store_path).iter()}
    assert ra == rb and len(ra) == 12


def test_shoot_surface_with_we(tmp_path):
    data = _analytic(tmp_path, mode="shoot.surface", ic={"kind": "points", "points": [[-0.2], [0.2]]})
    data["stop"]["t_max"] = 0.1
    data["resampler"] = {"kind": "we", "progress": "x", "bin_edges": [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
                         "target_per_bin": 2, "tau_seg": 0.05, "n_iter": 20, "walkers_per_frame": 2}
    data["budget"] = {}
    cfg = RunConfig.from_dict(data)
    out = dispatch(cfg)()
    assert out["resampler"] == "we" and out["valid"] is True
    total = sum(out["absorbed"].values()) + out["final_weight"]
    assert total == pytest.approx(1.0, rel=1e-12)
    recs = list(Store(cfg.store_path).iter(kind="segment"))
    assert recs and {r.key["run_id"] for r in recs} == {"main"}
    with pytest.raises(ResumeError, match="cannot be resumed"):
        dispatch(cfg, resume=True)()


def test_shoot_encounter_analytic_end_to_end(tmp_path):
    from cytherea.ic.frames import EnsembleFrame, save_frames

    tri = np.array([[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [0.56, 0.97, 0.0]])
    for name, geom in (("a", tri), ("b", tri[:2])):
        save_frames(tmp_path / f"{name}.npz", [EnsembleFrame(coordinates=geom + 0.01 * k, box=None,
                    topology_ref=name, temperature=1.0, weight=1.0, source_id=name, frame_id=k, time=0.0)
                    for k in range(2)])
    data = {
        "mode": "shoot.encounter", "units": "reduced",
        "system": {"kind": "analytic", "potential": {"name": "LJCluster", "params": {"n_atoms": 5}}},
        "physics": {"kind": "analytic", "integrator": "baoab", "dt": 1e-3, "kT": 1.0, "gamma": 1.0},
        "ic": {"kind": "encounter", "frames_A": "a.npz", "frames_B": "b.npz", "b": 3.0, "min_pair_dist": 0.9,
               "label": [1, 2]},
        "stop": {"kind": "b_surface", "reaction": {"name": "bound", "all_of": [{"observable": "r", "le": 1.5}]},
                 "r_observable": "r", "q": 4.0, "tau_persist": 0.02, "t_max": 0.3},
        "observables": {"dt_obs": 0.01, "items": [{"name": "r", "kind": "com_distance", "ranges": [[0, 3], [3, 5]]}]},
        "budget": {"shots_per_frame": 6, "stage": "enc"},
        "seed": 4, "store_path": "enc.sqlite",
    }
    (tmp_path / "enc.yaml").write_text(yaml.safe_dump(data))
    cfg = load_config(tmp_path / "enc.yaml")
    out = dispatch(cfg)()
    assert out["n_keys"] == 6 and out["n_failures"] == 0
    recs = list(Store(cfg.store_path).iter())
    assert len(recs) == 6 and all(r.frame_id == -1 and r.ic_meta["b"] == 3.0 for r in recs)
    assert all(abs(r.observables["r"][0] - 3.0) < 1e-9 for r in recs)
    assert all(r.stop_reason in {"reaction", "escape", "timeout"} for r in recs)
    assert {r.ic_meta["state"] == [1, 2] for r in recs} == {True}


def test_prepare_then_shoot_alanine_dipeptide(ala2_files, tmp_path):
    prep = RunConfig.from_dict(_prepare(ala2_files, tmp_path))
    out = dispatch(prep)()
    assert out["n_frames"] == 3 and Path(prep.budget.output_frames).exists()
    from cytherea.ic.frames import load_frames

    frames = load_frames(prep.budget.output_frames)
    assert [f.frame_id for f in frames] == [0, 1, 2] and frames[0].box is None
    assert frames[0].coordinates.shape == (22, 3)
    with pytest.raises(ResumeError, match="already exists"):
        dispatch(prep)()
    shoot = RunConfig.from_dict(_openmm(ala2_files, tmp_path))
    res = dispatch(shoot)()
    assert res["n_keys"] == 3 and res["stop_reasons"] == {"fixed_lag": 3}
    rec = next(Store(shoot.store_path).iter())
    assert len(rec.observables["phi"]) == 3 and all(-math.pi <= v <= math.pi for v in rec.observables["phi"])


# ---------------------------------------------------------------- 13.4


def _crash_run(cfg_path: str, kill_at: int) -> None:
    """Spawned child: `cytherea run` whose run_batch kills the whole process
    group the moment shot `kill_at` is computed but not yet appended (as 7.4a)."""
    os.setsid()
    real = R.run_batch

    def crashing(keys, shot_fn, store, **kw):
        def hook(rec):
            if rec.key["frame_id"] * 1000 + rec.key["shot_id"] == kill_at:
                os.killpg(0, signal.SIGKILL)

        return real(keys, shot_fn, store, on_before_append=hook, **kw)

    R.run_batch = crashing
    cli.main(["run", cfg_path])


@pytest.mark.parametrize("n_workers", [1, 3])
def test_13_4_cli_resume_after_crash_matches_uninterrupted_run(tmp_path, n_workers):
    crash = _analytic(tmp_path, store_path=str(tmp_path / "crash.sqlite"),
                      budget={"shots_per_frame": 4, "n_workers": n_workers})
    clean = dict(crash, store_path=str(tmp_path / "clean.sqlite"))
    p_crash, p_clean = tmp_path / "crash.yaml", tmp_path / "clean.yaml"
    p_crash.write_text(yaml.safe_dump(crash))
    p_clean.write_text(yaml.safe_dump(clean))
    kill_at = 1 * 1000 + 2  # frame 1, shot 2: keys are frame-major, so 6 records are committed
    proc = multiprocessing.get_context("spawn").Process(target=_crash_run, args=(str(p_crash), kill_at))
    proc.start()
    proc.join(120)
    assert proc.exitcode == -signal.SIGKILL
    partial = {r.key_digest for r in Store(crash["store_path"]).iter()}
    assert len(partial) == 6
    assert cli.main(["resume", str(p_crash)]) == 0
    assert cli.main(["run", str(p_clean)]) == 0
    a = {r.key_digest: r for r in Store(crash["store_path"]).iter()}
    b = {r.key_digest: r for r in Store(clean["store_path"]).iter()}
    assert len(a) == 12 and a == b


# ---------------------------------------------------------------- CLI


def test_cli_exit_codes_and_report(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_analytic(tmp_path)))
    assert cli.main(["run", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["n_keys"] == 12
    assert cli.main(["run", str(good)]) == 2
    assert "cytherea resume" in capsys.readouterr().err
    assert cli.main(["report", str(tmp_path / "store.sqlite")]) == 0
    assert json.loads(capsys.readouterr().out)["n_records"] == 12
    bad = _analytic(tmp_path)
    del bad["physics"]["kT"]
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert cli.main(["run", str(tmp_path / "bad.yaml")]) == 2
    assert "physics.analytic.kT" in capsys.readouterr().err
    assert cli.main(["report", str(tmp_path / "none.sqlite")]) == 2


def test_relative_paths_are_relative_to_the_config_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    data = _analytic(tmp_path, store_path="out/store.sqlite")
    (d / "c.yaml").write_text(yaml.safe_dump(data))
    assert load_config(d / "c.yaml").store_path == str(d / "out" / "store.sqlite")


def test_example_configs_load():
    paths = sorted(REPO.glob("examples/*/config*.yaml"))
    assert paths
    for p in paths:
        cfg = load_config(p)
        assert dispatch(cfg).func is R.RUNNERS[cfg.mode]
