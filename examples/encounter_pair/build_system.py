#!/usr/bin/env python
"""Barnase-barstar in GBn2 implicit solvent for the A3 encounter test (Task 16).

Plain OpenMM + pdbfixer (no cytherea): PDB 1BRS, chain A (barnase) and chain
D (barstar), crystal waters and heterogens removed, missing atoms added,
hydrogens at pH 7. amber14-all + implicit/gbn2.xml with 0.15 M implicit
salt (Debye length ~0.8 nm), NoCutoff, HBonds constraints, 300 K.

Why the salt: the NAM conversion beta -> beta_inf uses Omega = b/q, i.e. free
diffusion beyond b. Without salt GBn2 has no screening, and the barnase (+2)
/ barstar (-6) Coulomb energy is still ~ -1.7 kT at 5 nm and -0.6 kT at
15 nm, so beta_inf would depend on q for physical reasons (test 16.3).

Writes to --out:
  complex.pdb / complex_system.xml   the complex (atoms: barnase, then barstar)
  barnase.pdb / barstar.pdb          the partners alone (same atom order)
  native_contacts.json               interface heavy-atom pairs (complex
                                     indices) within --contact-cutoff in the
                                     minimised complex
  build.json                         provenance (sha256 of 1BRS.pdb, counts)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

FF = ("amber14-all.xml", "implicit/gbn2.xml")
URL = "https://files.rcsb.org/download/1BRS.pdb"


def fixed_chain(raw: Path, chain_id: str):
    from pdbfixer import PDBFixer

    fx = PDBFixer(filename=str(raw))
    fx.removeChains([i for i, c in enumerate(fx.topology.chains()) if c.id != chain_id])
    fx.removeHeterogens(keepWater=False)
    fx.findMissingResidues()
    fx.missingResidues = {}  # no loop building: terminal gaps stay gaps
    fx.findNonstandardResidues()
    fx.replaceNonstandardResidues()
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    fx.addMissingHydrogens(7.0)
    return fx.topology, fx.positions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--contact-cutoff-nm", type=float, default=0.5)
    p.add_argument("--salt-molar", type=float, default=0.15)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw = a.out / "1BRS.pdb"
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    ff = app.ForceField(*FF)

    parts = {}
    for name, chain in (("barnase", "A"), ("barstar", "D")):
        top, pos = fixed_chain(raw, chain)
        parts[name] = (top, pos)
    modeller = app.Modeller(*parts["barnase"])
    modeller.add(*parts["barstar"])
    n_a = parts["barnase"][0].getNumAtoms()
    # Debye-Hueckel kappa as OpenMM's AmberPrmtopFile computes it (1/Angstrom), solvent eps 78.5, 300 K
    kappa_per_nm = 10.0 * 50.33355 * (a.salt_molar / (78.5 * 300.0)) ** 0.5
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             implicitSolventKappa=kappa_per_nm / u.nanometer)

    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(a.platform))
    ctx.setPositions(modeller.positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=10.0)
    st = ctx.getState(getPositions=True, getEnergy=True)
    x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    e1 = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

    with open(a.out / "complex.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, x * u.nanometer, fh, keepIds=True)
    (a.out / "complex_system.xml").write_text(openmm.XmlSerializer.serialize(system))
    for name, sl in (("barnase", slice(0, n_a)), ("barstar", slice(n_a, None))):
        with open(a.out / f"{name}.pdb", "w") as fh:
            app.PDBFile.writeFile(parts[name][0], x[sl] * u.nanometer, fh, keepIds=True)

    atoms = list(modeller.topology.atoms())
    heavy_a = [i for i in range(n_a) if atoms[i].element.symbol != "H"]
    heavy_b = [i for i in range(n_a, len(atoms)) if atoms[i].element.symbol != "H"]
    xa, xb = np.asarray(x)[heavy_a], np.asarray(x)[heavy_b]
    d = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
    ia, ib = np.nonzero(d < a.contact_cutoff_nm)
    pairs = [[heavy_a[i], heavy_b[j], float(d[i, j])] for i, j in zip(ia, ib)]
    masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
    com = lambda sl: (masses[sl, None] * np.asarray(x)[sl]).sum(0) / masses[sl].sum()  # noqa: E731
    (a.out / "native_contacts.json").write_text(json.dumps(
        {"cutoff_nm": a.contact_cutoff_nm, "pairs": pairs, "n_pairs": len(pairs)}, indent=1))
    build = {
        "pdb": "1BRS", "chains": {"barnase": "A", "barstar": "D"}, "forcefield": list(FF),
        "implicit_salt_molar": a.salt_molar, "implicit_kappa_per_nm": kappa_per_nm,
        "debye_length_nm": 1.0 / kappa_per_nm if kappa_per_nm > 0 else None,
        "pdb_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "n_atoms": {"barnase": n_a, "barstar": len(atoms) - n_a, "complex": len(atoms)},
        "n_constraints": system.getNumConstraints(),
        "energy_kJ_mol": {"before_min": e0, "after_min": e1},
        "native_com_distance_nm": float(np.linalg.norm(com(slice(n_a, None)) - com(slice(0, n_a)))),
        "radius_of_gyration_nm": {
            name: float(np.sqrt((masses[sl] * ((np.asarray(x)[sl] - com(sl)) ** 2).sum(1)).sum() / masses[sl].sum()))
            for name, sl in (("barnase", slice(0, n_a)), ("barstar", slice(n_a, None)))},
        "max_extent_from_com_nm": {
            name: float(np.max(np.linalg.norm(np.asarray(x)[sl] - com(sl), axis=1)))
            for name, sl in (("barnase", slice(0, n_a)), ("barstar", slice(n_a, None)))},
        "n_native_contacts": len(pairs),
        "openmm": openmm.__version__,
    }
    (a.out / "build.json").write_text(json.dumps(build, indent=1))
    print(json.dumps(build, indent=1))


if __name__ == "__main__":
    main()
