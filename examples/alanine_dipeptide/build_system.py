#!/usr/bin/env python
"""Build and equilibrate solvated alanine dipeptide (A1 reference, plain OpenMM).

Pipeline (all seeded, see --seed):
  1. ACE-ALA-NME from openmmtools' bundled ``alanine-dipeptide-gbsa`` PDB
     (22 atoms, vacuum geometry; no download).
  2. amber14-all + amber14/tip3pfb, cubic box with >= 1.0 nm between the
     solute's bounding sphere and every box face (edge = 2 r + 2.0 nm), neutral
     (the peptide is already neutral: no ions are added), PME 0.9 nm,
     HBonds constraints, rigid water.
  3. Energy minimisation.
  4. NVT  --nvt-ps  (default 100 ps), LangevinMiddle gamma = 1/ps, 300 K, 2 fs.
  5. NPT  --npt-ps  (default 500 ps), + MonteCarloBarostat 1 bar / 300 K.
     Box volume sampled every 1 ps; the first --npt-discard-ps (default 100 ps)
     are discarded; the rest are averaged.
  6. Box fixed at that average volume: the last NPT configuration is rescaled
     isotropically (molecule centres scaled, molecules kept rigid, as the MC
     barostat does), then NVT --relax-ps (default 20 ps, gamma = 1/ps) at the
     fixed box.  gamma = 1/ps is used ONLY here (preparation, design §9).

Outputs in --out:
  system.xml      final System (no barostat; default box = fixed production box)
  topology.pdb    topology + final coordinates + CRYST1 box
  state.xml       equilibrated State (positions, velocities, box), time reset to 0
  build.json      provenance: versions, settings, energies, NPT volume statistics
  equil_volume.csv  NPT volume trace (ps, nm^3)
"""
from __future__ import annotations

import argparse
import json
import platform as _pyplatform
import sys
import time
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ala2_common as C  # noqa: E402


def source_pdb() -> Path:
    import openmmtools

    pdb = Path(openmmtools.__file__).parent / "data" / "alanine-dipeptide-gbsa" / "alanine-dipeptide.pdb"
    if not pdb.exists():
        raise FileNotFoundError(pdb)
    return pdb


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU", "Reference"])
    p.add_argument("--seed", type=int, default=20260930, help="nonzero; integrator/barostat/velocity seed")
    p.add_argument("--nvt-ps", type=float, default=100.0)
    p.add_argument("--npt-ps", type=float, default=500.0)
    p.add_argument("--npt-discard-ps", type=float, default=100.0)
    p.add_argument("--relax-ps", type=float, default=20.0)
    p.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    return p.parse_args(argv)


def nsteps(ps: float) -> int:
    n = int(round(ps / C.TIMESTEP_PS))
    if n < 0:
        raise ValueError("negative duration")
    return n


def rescale_to_volume(context: openmm.Context, target_volume_nm3: float) -> float:
    """Isotropically scale box + molecule centres so the box volume equals target."""
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    a, b, c = (v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors())
    a, b, c = np.array(a), np.array(b), np.array(c)
    vol = float(np.dot(a, np.cross(b, c)))
    s = (target_volume_nm3 / vol) ** (1.0 / 3.0)
    new = pos.copy()
    for mol in context.getMolecules():
        mol = list(mol)
        centre = pos[mol].mean(axis=0)
        new[mol] = pos[mol] + (s - 1.0) * centre
    context.setPeriodicBoxVectors(a * s, b * s, c * s)
    context.setPositions(new)
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed == 0 or not (0 < args.seed < 2**31):
        raise SystemExit("--seed must be in [1, 2^31-1] (0 means 'random' in OpenMM)")
    out: Path = args.out
    if out.exists() and (out / "state.xml").exists() and not args.force:
        raise SystemExit(f"{out} already contains state.xml (use --force to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()

    temperature = C.TEMPERATURE_K * unit.kelvin
    dt = C.TIMESTEP_PS * unit.picosecond

    pdb_path = source_pdb()
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(*C.FORCEFIELD_FILES)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    xyz_solute = np.array(pdb.positions.value_in_unit(unit.nanometer))
    centre = 0.5 * (xyz_solute.min(axis=0) + xyz_solute.max(axis=0))
    radius = float(np.linalg.norm(xyz_solute - centre, axis=1).max())
    edge = 2 * radius + 2 * C.PADDING_NM
    modeller.addSolvent(ff, model="tip3p", boxSize=openmm.Vec3(edge, edge, edge) * unit.nanometer,
                        neutralize=True, ionicStrength=0 * unit.molar)
    top = modeller.topology
    n_water = sum(1 for r in top.residues() if r.name == "HOH")
    n_ions = sum(1 for r in top.residues() if r.name in ("NA", "CL", "Na+", "Cl-"))

    def make_system():
        return ff.createSystem(top, nonbondedMethod=app.PME,
                               nonbondedCutoff=C.NONBONDED_CUTOFF_NM * unit.nanometer,
                               constraints=app.HBonds, rigidWater=True,
                               ewaldErrorTolerance=5e-4)

    system = make_system()
    charge = sum(
        f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
        for i in range(f.getNumParticles())
    )
    barostat = openmm.MonteCarloBarostat(C.PRESSURE_BAR * unit.bar, temperature, 25)
    barostat.setRandomNumberSeed(args.seed)
    barostat.setFrequency(0)  # inactive during minimisation + NVT
    system.addForce(barostat)

    integrator = openmm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(args.seed)
    plat, props = C.make_platform(args.platform)
    context = openmm.Context(system, integrator, plat, props)
    context.setPositions(modeller.positions)

    def pe():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    log = {}
    log["pe_initial_kJmol"] = pe()
    openmm.LocalEnergyMinimizer.minimize(context, 10.0, 0)
    log["pe_minimized_kJmol"] = pe()
    print(f"minimised: PE {log['pe_initial_kJmol']:.1f} -> {log['pe_minimized_kJmol']:.1f} kJ/mol", flush=True)

    context.setVelocitiesToTemperature(temperature, args.seed)
    integrator.step(nsteps(args.nvt_ps))
    log["pe_after_nvt_kJmol"] = pe()
    print(f"NVT {args.nvt_ps} ps done: PE {log['pe_after_nvt_kJmol']:.1f}", flush=True)

    # NPT
    barostat.setFrequency(25)
    context.reinitialize(preserveState=True)
    vols = []
    n_npt_ps = int(round(args.npt_ps))
    steps_per_ps = nsteps(1.0)
    for k in range(1, n_npt_ps + 1):
        integrator.step(steps_per_ps)
        box = context.getState().getPeriodicBoxVectors()
        a, b, c = (np.array(v.value_in_unit(unit.nanometer)) for v in box)
        vols.append((float(k), float(np.dot(a, np.cross(b, c)))))
    vols = np.array(vols)
    (out / "equil_volume.csv").write_text(
        "time_ps,volume_nm3\n" + "".join(f"{t:.1f},{v:.6f}\n" for t, v in vols))
    keep = vols[vols[:, 0] > args.npt_discard_ps, 1]
    if keep.size == 0:
        keep = vols[-max(1, len(vols) // 2):, 1]
    v_avg = float(keep.mean())
    v_sem_naive = float(keep.std(ddof=1) / np.sqrt(keep.size)) if keep.size > 1 else float("nan")
    print(f"NPT {args.npt_ps} ps done: <V> = {v_avg:.4f} nm^3 over {keep.size} samples "
          f"(std {keep.std():.4f})", flush=True)

    # Fix the box at <V>: final system has no barostat
    scale = rescale_to_volume(context, v_avg)
    final_system = make_system()
    box_vecs = context.getState().getPeriodicBoxVectors()
    final_system.setDefaultPeriodicBoxVectors(*box_vecs)
    state_mid = context.getState(getPositions=True, getVelocities=True)
    del context, integrator

    integrator = openmm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(args.seed + 1)
    context = openmm.Context(final_system, integrator, plat, props)
    context.setPeriodicBoxVectors(*box_vecs)
    context.setPositions(state_mid.getPositions())
    context.setVelocities(state_mid.getVelocities())
    context.applyConstraints(1e-6)
    context.applyVelocityConstraints(1e-6)
    integrator.step(nsteps(args.relax_ps))
    context.setTime(0.0)
    context.setStepCount(0)
    final = context.getState(getPositions=True, getVelocities=True, getEnergy=True, getParameters=True)
    log["pe_final_kJmol"] = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ke = final.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    n_dof = 3 * final_system.getNumParticles() - final_system.getNumConstraints() - 3
    t_inst = 2 * ke / (n_dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin))
    if not np.isfinite(log["pe_final_kJmol"]):
        raise SystemExit("non-finite final energy")

    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(final_system))
    (out / "state.xml").write_text(openmm.XmlSerializer.serialize(final))
    with open(out / "topology.pdb", "w") as fh:
        top.setPeriodicBoxVectors(box_vecs)
        app.PDBFile.writeFile(top, final.getPositions(), fh, keepIds=True)

    box_nm = [[float(x) for x in v.value_in_unit(unit.nanometer)] for v in box_vecs]
    mass_amu = sum(final_system.getParticleMass(i).value_in_unit(unit.dalton)
                   for i in range(final_system.getNumParticles()))
    density = mass_amu * 1.66053906660e-27 / (v_avg * 1e-27) / 1000.0  # g/cm^3
    phi_idx, psi_idx = C.phi_psi_indices(top)
    xyz = final.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    info = {
        "source_pdb": str(pdb_path),
        "openmm_version": openmm.__version__,
        "python": _pyplatform.python_version(),
        "platform": args.platform,
        "platform_properties": props,
        "seed": args.seed,
        "forcefield": list(C.FORCEFIELD_FILES),
        "water_model": "TIP3P-FB (amber14/tip3pfb.xml), rigid",
        "nonbonded": {"method": "PME", "cutoff_nm": C.NONBONDED_CUTOFF_NM, "ewaldErrorTolerance": 5e-4,
                      "dispersion_correction": True},
        "constraints": "HBonds",
        "remove_cm_motion": True,
        "padding_nm": C.PADDING_NM,
        "padding_meaning": "min distance solute bounding sphere -> box face; edge = 2 r + 2 padding",
        "solute_bounding_radius_nm": radius,
        "initial_box_edge_nm": edge,
        "n_atoms": final_system.getNumParticles(),
        "n_water": n_water,
        "n_ions": n_ions,
        "ionic_strength_M": 0.0,
        "net_charge_e": round(charge, 6),
        "protonation": "n/a (ACE-ALA-NME has no titratable groups)",
        "n_constraints": final_system.getNumConstraints(),
        "n_dof": n_dof,
        "temperature_K": C.TEMPERATURE_K,
        "timestep_ps": C.TIMESTEP_PS,
        "protocol": {
            "minimize": "LocalEnergyMinimizer tol 10 kJ/mol/nm",
            "nvt_ps": args.nvt_ps, "npt_ps": args.npt_ps, "npt_discard_ps": args.npt_discard_ps,
            "relax_fixed_box_ps": args.relax_ps,
            "equilibration_integrator": "LangevinMiddle gamma=1/ps (preparation only)",
            "barostat": "MonteCarloBarostat 1 bar, every 25 steps",
        },
        "npt_volume_nm3": {"mean": v_avg, "std": float(keep.std(ddof=1)) if keep.size > 1 else None,
                           "n_samples": int(keep.size), "naive_sem": v_sem_naive,
                           "initial_modeller": float(vols[0, 1]) if len(vols) else None},
        "rescale_factor_last_npt_frame": scale,
        "box_vectors_nm": box_nm,
        "density_g_cm3": density,
        "energies_kJmol": log,
        "final_instantaneous_T_K": t_inst,
        "final_phi_psi_deg": [C.dihedral_deg(xyz, phi_idx), C.dihedral_deg(xyz, psi_idx)],
        "wall_s": time.time() - t_wall0,
    }
    (out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out}: {info['n_atoms']} atoms ({n_water} waters), box {box_nm[0][0]:.4f} nm, "
          f"density {density:.4f} g/cm3, T_inst {t_inst:.1f} K, wall {info['wall_s']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
