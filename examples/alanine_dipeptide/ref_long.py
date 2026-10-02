#!/usr/bin/env python
"""Long reference trajectory for A1 (plain OpenMM, independent of cytherea).

Measurement dynamics (design §9): LangevinMiddle, gamma = 0.1/ps, 300 K,
dt = 2 fs, fixed box (NVT), seeded (--seed, nonzero).  CUDA runs use mixed
precision with DeterministicForces=true; CPU is meant for smoke tests only
(there is no automatic fallback: the platform is whatever --platform says).

Outputs in --out (see README.md):
  run.json          run parameters (checked on --resume), sha256 of the input
                    system.xml/state.xml/topology.pdb (verified on --resume when
                    present) and the scripts' git commit
  phipsi.bin        phi/psi every --phipsi-ps (1 ps), 16-byte records
                    (int64 step, float32 phi_deg, float32 psi_deg); record 0 = step 0
  phipsi.json       format description
  traj.dcd          full-system DCD every --dcd-ps (10 ps), first frame at step dcd
  checkpoint.chk    OpenMM checkpoint every --checkpoint-ns (10 ns) and at the end
  checkpoint.prev.chk  the previous checkpoint (automatic fallback on --resume if
                    checkpoint.chk is missing or unreadable)
  checkpoint.json   step/time/record counts of the latest checkpoint (informational)
  progress.log      ns done, ns/day, PE, T, ETA (flushed every line)

Resume (--resume): the checkpoint's own step count S is authoritative.  phi/psi
records after S and DCD frames after S are truncated, then the run continues
and appends -- no duplicated or missing frames.  --total-ns may be increased on
resume to extend a finished run.  SIGINT/SIGTERM trigger a clean stop: the run
continues to the next DCD boundary (<= --dcd-ps), writes a checkpoint and exits 0.
A hard kill (SIGKILL, node crash) loses at most --checkpoint-ns of work.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ala2_common as C  # noqa: E402

GAMMA_PER_PS = 0.1

_STOP = {"flag": False, "signal": None}


def _on_signal(signum, frame):  # pragma: no cover - exercised via subprocess
    _STOP["flag"] = True
    _STOP["signal"] = signum


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--total-ns", required=True, type=float)
    p.add_argument("--seed", required=True, type=int, help="integrator seed, in [1, 2^31-1]")
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL", "Reference"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--system-dir", type=Path, default=None,
                   help="dir with system.xml/state.xml/topology.pdb from build_system.py "
                        "(default: /home/ruigengji/cytherea/runs/ala2_system)")
    p.add_argument("--phipsi-ps", type=float, default=C.OBS_INTERVAL_PS,
                   help="phi/psi interval (default = ala2_common.OBS_INTERVAL_PS, the state contract's 1 ps)")
    p.add_argument("--dcd-ps", type=float, default=10.0)
    p.add_argument("--checkpoint-ns", type=float, default=10.0)
    p.add_argument("--flush-ps", type=float, default=100.0, help="phi/psi buffer flush interval")
    p.add_argument("--report-ps", type=float, default=1000.0, help="progress.log interval")
    p.add_argument("--stop-after-ps", type=float, default=None,
                   help="testing aid: stop cleanly (checkpoint + exit 0) after this many ps in THIS process")
    p.add_argument("--crash-at-ps", type=float, default=None,
                   help="testing aid: flush everything written so far, then SIGKILL this process "
                        "once the trajectory reaches this time (simulates a hard crash between checkpoints)")
    return p.parse_args(argv)


def ps_to_steps(ps: float) -> int:
    n = int(round(ps / C.TIMESTEP_PS))
    if n <= 0 or abs(n * C.TIMESTEP_PS - ps) > 1e-9 * max(1.0, ps):
        raise SystemExit(f"{ps} ps is not a positive whole number of {C.TIMESTEP_PS} ps steps")
    return n


INPUT_FILES = ("system.xml", "state.xml", "topology.pdb")


def input_hashes(sysdir: Path) -> dict:
    return {name: C.sha256_file(sysdir / name) for name in INPUT_FILES}


def scripts_git() -> dict | None:
    """git commit of the scripts directory (+ whether examples/alanine_dipeptide is dirty)."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=30, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=30, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty), "dir": str(here)}
    except Exception as exc:  # pragma: no cover - git missing / not a repo
        return {"commit": None, "error": str(exc), "dir": str(here)}


class Logger:
    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        print(line, flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (0 < args.seed < 2**31):
        raise SystemExit("--seed must be in [1, 2^31-1] (0 means 'random' in OpenMM)")
    out: Path = args.out
    sysdir = args.system_dir or Path("/home/ruigengji/cytherea/runs/ala2_system")

    phipsi_steps = ps_to_steps(args.phipsi_ps)
    dcd_steps = ps_to_steps(args.dcd_ps)
    ckpt_steps = ps_to_steps(args.checkpoint_ns * 1000.0)
    flush_steps = ps_to_steps(args.flush_ps)
    report_steps = ps_to_steps(args.report_ps)
    for name, n in (("dcd", dcd_steps), ("checkpoint", ckpt_steps), ("flush", flush_steps), ("report", report_steps)):
        if n % phipsi_steps:
            raise SystemExit(f"--{name} interval must be a multiple of --phipsi-ps")
    if ckpt_steps % dcd_steps:
        raise SystemExit("--checkpoint-ns must be a multiple of --dcd-ps")
    total_steps = ps_to_steps(args.total_ns * 1000.0)
    if total_steps % dcd_steps:
        raise SystemExit("--total-ns must be a multiple of --dcd-ps")

    run_json = out / "run.json"
    params = {
        "seed": args.seed,
        "platform": args.platform,
        "timestep_ps": C.TIMESTEP_PS,
        "temperature_K": C.TEMPERATURE_K,
        "gamma_per_ps": GAMMA_PER_PS,
        "integrator": "LangevinMiddleIntegrator",
        "phipsi_interval_steps": phipsi_steps,
        "dcd_interval_steps": dcd_steps,
        "checkpoint_interval_steps": ckpt_steps,
        "system_dir": str(sysdir.resolve()),
    }
    if args.resume:
        if not run_json.exists() or not ((out / "checkpoint.chk").exists()
                                         or (out / "checkpoint.prev.chk").exists()):
            raise SystemExit(f"--resume: no run.json or no checkpoint(.prev).chk in {out}")
        old = json.loads(run_json.read_text())
        for k, v in params.items():
            if old.get(k) != v:
                raise SystemExit(f"--resume: parameter {k} differs from run.json ({old.get(k)!r} != {v!r})")
        # Input provenance: verified when recorded; runs created before this field
        # existed (e.g. the live 1 us reference) have none and stay resumable.
        resume_notes = []
        if "input_sha256" in old:
            now = input_hashes(sysdir)
            bad = [k for k in INPUT_FILES if old["input_sha256"].get(k) != now.get(k)]
            if bad:
                raise SystemExit(f"--resume: {', '.join(bad)} in {sysdir} changed since the run started "
                                 f"(sha256 mismatch); refusing to resume")
            resume_notes.append("input sha256 verified")
        else:
            resume_notes.append("run.json has no input_sha256 (older run): inputs NOT verified")
        old["total_steps"] = max(old.get("total_steps", 0), total_steps)
        run_meta = old
    else:
        if run_json.exists() or (out / "phipsi.bin").exists():
            raise SystemExit(f"{out} already holds a run; use --resume or a new --out")
        out.mkdir(parents=True, exist_ok=True)
        run_meta = dict(params)
        run_meta.update({
            "total_steps": total_steps,
            "openmm_version": openmm.__version__,
            "host": socket.gethostname(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cmdline": sys.argv,
            "input_sha256": input_hashes(sysdir),
            "scripts_git": scripts_git(),
        })
    run_meta.setdefault("segments", [])

    log = Logger(out / "progress.log")
    system = openmm.XmlSerializer.deserialize((sysdir / "system.xml").read_text())
    pdb = app.PDBFile(str(sysdir / "topology.pdb"))
    top = pdb.topology
    phi_idx, psi_idx = C.phi_psi_indices(top)
    for f in system.getForces():
        if isinstance(f, (openmm.MonteCarloBarostat,)):
            raise SystemExit("system.xml contains a barostat; production must be NVT")

    integrator = openmm.LangevinMiddleIntegrator(C.TEMPERATURE_K * unit.kelvin,
                                                 GAMMA_PER_PS / unit.picosecond,
                                                 C.TIMESTEP_PS * unit.picosecond)
    integrator.setRandomNumberSeed(args.seed)
    plat, props = C.make_platform(args.platform)
    context = openmm.Context(system, integrator, plat, props)
    device = {k: plat.getPropertyValue(context, k) for k in plat.getPropertyNames()
              if k in ("DeviceName", "DeviceIndex", "Precision", "DeterministicForces", "Threads")}

    phipsi_path = out / "phipsi.bin"
    dcd_path = out / "traj.dcd"
    ckpt_path = out / "checkpoint.chk"

    if args.resume:
        prev_path = out / "checkpoint.prev.chk"
        loaded = None
        unreadable = []
        for cand in (ckpt_path, prev_path):
            if not cand.exists():
                log(f"RESUME: {cand.name} missing, trying the fallback")
                continue
            try:
                context.loadCheckpoint(cand.read_bytes())
                loaded = cand
                break
            except Exception as exc:
                unreadable.append(cand)
                log(f"RESUME: could not load {cand.name} ({exc}); trying the fallback")
        if loaded is None:
            raise SystemExit("--resume: no loadable checkpoint")
        if loaded != ckpt_path:
            log(f"RESUME: using FALLBACK {loaded.name}")
            if ckpt_path in unreadable:
                # otherwise the next rotation would copy the unreadable file over the
                # good checkpoint.prev.chk we just resumed from
                bad = out / f"checkpoint.bad-{time.strftime('%Y%m%dT%H%M%S')}.chk"
                os.replace(ckpt_path, bad)
                C.fsync_dir(out)
                log(f"RESUME: moved the unreadable checkpoint.chk to {bad.name}")
        for note in resume_notes:
            log(f"RESUME: {note}")
        step = int(context.getStepCount())
        if step % dcd_steps:
            raise SystemExit(f"checkpoint step {step} not on a DCD boundary")
        n_rec_expected = step // phipsi_steps + 1
        recs = C.load_phipsi(phipsi_path)
        if len(recs) < n_rec_expected:
            raise SystemExit(f"phipsi.bin has {len(recs)} records < {n_rec_expected} required by checkpoint "
                             f"at step {step}: data lost, refusing to resume")
        exp_steps = np.arange(n_rec_expected, dtype=np.int64) * phipsi_steps
        if not np.array_equal(recs["step"][:n_rec_expected], exp_steps):
            raise SystemExit("phipsi.bin step column inconsistent with the checkpoint")
        dropped_rec = len(recs) - n_rec_expected
        C.truncate_phipsi(phipsi_path, n_rec_expected)
        n_dcd_expected = step // dcd_steps
        n_dcd = C.dcd_n_frames(dcd_path) if dcd_path.exists() else 0
        if n_dcd < n_dcd_expected:
            raise SystemExit(f"traj.dcd has {n_dcd} frames < {n_dcd_expected} required: refusing to resume")
        if dcd_path.exists():
            C.truncate_dcd(dcd_path, n_dcd_expected)
        log(f"RESUME from checkpoint step {step} ({step * C.TIMESTEP_PS / 1000:.3f} ns): "
            f"dropped {dropped_rec} phi/psi records and {n_dcd - n_dcd_expected} DCD frames past the checkpoint")
        dcd_fh = open(dcd_path, "r+b")
        dcd = app.DCDFile(dcd_fh, top, C.TIMESTEP_PS, dcd_steps, dcd_steps, append=True)
    else:
        state = openmm.XmlSerializer.deserialize((sysdir / "state.xml").read_text())
        context.setState(state)
        context.setTime(0.0)
        context.setStepCount(0)
        step = 0
        C.write_phipsi_meta(out / "phipsi.json", phi_idx, psi_idx, phipsi_steps)
        dcd_fh = open(dcd_path, "wb")
        dcd = app.DCDFile(dcd_fh, top, C.TIMESTEP_PS, dcd_steps, dcd_steps)
        # record 0 = the starting frame
        xyz0 = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        rec0 = np.array([(0, C.dihedral_deg(xyz0, phi_idx), C.dihedral_deg(xyz0, psi_idx))], dtype=C.PHIPSI_DTYPE)
        with open(phipsi_path, "wb") as fh:
            rec0.tofile(fh)
        log(f"START seed={args.seed} platform={args.platform} {device} gamma={GAMMA_PER_PS}/ps "
            f"dt={C.TIMESTEP_PS} ps total={total_steps * C.TIMESTEP_PS / 1000:.3f} ns")

    total_steps = run_meta["total_steps"]
    run_meta["segments"].append({"start_step": step, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "pid": os.getpid(), "host": socket.gethostname(), "device": device,
                                 "scripts_git": scripts_git()})
    C.atomic_write_text(run_json, json.dumps(run_meta, indent=2) + "\n")

    phipsi_fh = open(phipsi_path, "ab")
    buf: list[tuple[int, float, float]] = []
    n_dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    def flush_phipsi():
        if buf:
            np.array(buf, dtype=C.PHIPSI_DTYPE).tofile(phipsi_fh)
            buf.clear()
        phipsi_fh.flush()

    def checkpoint(reason: str):
        flush_phipsi()
        os.fsync(phipsi_fh.fileno())
        dcd_fh.flush()
        os.fsync(dcd_fh.fileno())
        data = context.createCheckpoint()
        # new checkpoint fully on disk first, then rotate: there is never a moment
        # without a complete checkpoint.chk
        tmp = out / "checkpoint.chk.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if ckpt_path.exists():
            prev_tmp = out / "checkpoint.prev.chk.tmp"
            shutil.copyfile(ckpt_path, prev_tmp)
            C.fsync_path(prev_tmp)
            os.replace(prev_tmp, out / "checkpoint.prev.chk")
        os.replace(tmp, ckpt_path)
        C.fsync_dir(out)
        info = {"step": step, "time_ps": step * C.TIMESTEP_PS,
                "n_phipsi_records": step // phipsi_steps + 1, "n_dcd_frames": step // dcd_steps,
                "reason": reason, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        C.atomic_write_text(out / "checkpoint.json", json.dumps(info, indent=2) + "\n")
        log(f"CHECKPOINT step {step} ({step * C.TIMESTEP_PS / 1000:.3f} ns) [{reason}]")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    stop_after_steps = ps_to_steps(args.stop_after_ps) if args.stop_after_ps else None
    crash_at_step = ps_to_steps(args.crash_at_ps) if args.crash_at_ps else None

    if not args.resume:
        checkpoint("initial")

    seg_start_step = step
    t0 = time.time()
    t_last, s_last = t0, step
    while step < total_steps:
        integrator.step(phipsi_steps)
        step += phipsi_steps
        st = context.getState(getPositions=True)
        xyz = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        phi = C.dihedral_deg(xyz, phi_idx)
        psi = C.dihedral_deg(xyz, psi_idx)
        if not (np.isfinite(phi) and np.isfinite(psi)):
            flush_phipsi()
            log(f"ABORT non-finite phi/psi at step {step}; last good checkpoint kept")
            return 2
        buf.append((step, phi, psi))
        if step % dcd_steps == 0:
            dcd.writeModel(st.getPositions(), periodicBoxVectors=st.getPeriodicBoxVectors())
        if step % flush_steps == 0:
            flush_phipsi()
        if step % report_steps == 0 or step == total_steps:
            est = context.getState(getEnergy=True)
            pe = est.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = est.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            if not np.isfinite(pe):
                flush_phipsi()
                log(f"ABORT non-finite potential energy at step {step}")
                return 2
            now = time.time()
            ns_day_recent = (step - s_last) * C.TIMESTEP_PS / 1000 / max(now - t_last, 1e-9) * 86400
            ns_day_seg = (step - seg_start_step) * C.TIMESTEP_PS / 1000 / max(now - t0, 1e-9) * 86400
            eta_h = (total_steps - step) * C.TIMESTEP_PS / 1000 / max(ns_day_seg, 1e-9) * 24
            log(f"PROGRESS {step * C.TIMESTEP_PS / 1000:.3f}/{total_steps * C.TIMESTEP_PS / 1000:.3f} ns  "
                f"{ns_day_recent:.1f} ns/day (recent)  {ns_day_seg:.1f} ns/day (this process)  "
                f"PE {pe:.1f} kJ/mol  T {2 * ke / (n_dof * kB):.1f} K  ETA {eta_h:.2f} h")
            t_last, s_last = now, step
        if crash_at_step is not None and step >= crash_at_step:
            flush_phipsi()
            dcd_fh.flush()
            log(f"CRASH-TEST: SIGKILL self at step {step}")
            os.kill(os.getpid(), signal.SIGKILL)
        # checkpoints only ever sit on DCD boundaries, so resume bookkeeping stays exact
        if step % ckpt_steps == 0 or step == total_steps:
            checkpoint("final" if step == total_steps else "periodic")
        want_stop = _STOP["flag"] or (stop_after_steps is not None and step - seg_start_step >= stop_after_steps)
        if want_stop and step % dcd_steps == 0 and step < total_steps:
            if step % ckpt_steps:
                checkpoint(f"stop (signal {_STOP['signal']})" if _STOP["flag"] else "stop-after-ps")
            log(f"STOPPED cleanly at step {step}; continue with --resume")
            return 0
    flush_phipsi()
    log(f"DONE {step * C.TIMESTEP_PS / 1000:.3f} ns")
    phipsi_fh.close()
    dcd_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
