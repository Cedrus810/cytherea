#!/usr/bin/env python
"""ns/day of the A3 measurement dynamics (Task 16.5): barnase-barstar, GBn2,
NoCutoff, LangevinMiddle gamma = 0.1/ps, 2 fs, HBonds, CUDA mixed with
DeterministicForces, partners 5 nm apart. Run only on an idle GPU (check
nvidia-smi first, in a separate command)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--system-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--deterministic", default="true")
    a = p.parse_args()
    system = openmm.XmlSerializer.deserialize((a.system_dir / "complex_system.xml").read_text())
    pdb = app.PDBFile(str(a.system_dir / "complex.pdb"))
    n_a = json.loads((a.system_dir / "build.json").read_text())["n_atoms"]["barnase"]
    x = pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    x[n_a:] += np.array([5.0, 0.0, 0.0])
    integ = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 0.1 / u.picosecond, 0.002 * u.picosecond)
    integ.setRandomNumberSeed(1)
    plat = openmm.Platform.getPlatformByName("CUDA")
    ctx = openmm.Context(system, integ, plat, {"Precision": "mixed", "DeterministicForces": a.deterministic})
    ctx.setPositions(x)
    ctx.setVelocitiesToTemperature(300 * u.kelvin, 1)
    integ.step(1000)
    ctx.getState(getEnergy=True)
    t0 = time.perf_counter()
    for _ in range(a.steps // 500):
        integ.step(500)
        ctx.getState(getPositions=True)  # one observation per ps, as the shots do
    el = time.perf_counter() - t0
    ns = a.steps * 0.002 / 1000
    print(json.dumps({"n_atoms": system.getNumParticles(), "steps": a.steps, "wall_s": el,
                      "ns_per_day": ns / el * 86400, "deterministic_forces": a.deterministic,
                      "gpu": plat.getPropertyValue(ctx, "DeviceName")}, indent=1))


if __name__ == "__main__":
    main()
