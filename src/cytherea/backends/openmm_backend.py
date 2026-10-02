"""The `openmm` PotentialBackend: the production propagation path.

Everything the engine validated on analytic toys must carry over here, so
this module is mostly about getting the physics settings and the
reproducibility plumbing right, not about clever code.

Design decisions (task-12 brief + controller rulings):

- Units are OpenMM's: positions nm, velocities nm/ps, time ps, energies
  kJ/mol, forces kJ/mol/nm, masses dalton. Arrays are float64, shape
  (n_atoms, 3).
- No OpenMM reporters (ruling R3). `OpenMMPropagator.run(n)` only advances n
  steps; the engine computes observables from `get_state()` between chunks.
- Integrators: "verlet" -> `openmm.VerletIntegrator`, "langevin_middle" ->
  `openmm.LangevinMiddleIntegrator`, "nose_hoover" ->
  `openmm.NoseHooverIntegrator` (cfg.friction_per_ps is used as its
  collision frequency, which is a thermostat coupling constant, not a
  friction, so the section-9 friction rule does not apply to it).
  `nose_hoover` is allowed only with purpose="equilibration" in Phase A
  (ruling R21): MDState carries no thermostat-chain variables, so it is not
  a complete Nose-Hoover checkpoint, and the on-step velocity conversion
  below is only approximate for it.
- Velocity convention (ruling R21): `MDState.v` is the on-step velocity
  v(t). All three OpenMM integrators internally hold the staggered v(t-dt/2),
  so the propagator converts at its boundary with one force evaluation per
  call (never per step): get_state returns v = v_half + F(x) dt/(2m);
  set_state stores v_half = v - F(x) dt/(2m) and then applies the velocity
  constraints (`Context.applyVelocityConstraints(tol)`). For a constrained
  System get_state also projects v onto the velocity constraints (ruling
  R28: temporarily set v on the Context, applyVelocityConstraints with
  `PROJECTION_TOLERANCE` (1e-10), read it back, restore the original
  v_half -- skipped when there are no
  constraints), so get_state().v satisfies the constraints and a set/get
  round trip is exact to rounding. Massless (fixed)
  particles get no correction. For Verlet this is exactly the time-centred
  velocity OpenMM itself uses for `State.getKineticEnergy()`. Because
  set_state re-derives v_half, a run resumed from an MDState agrees with
  the uninterrupted run to rounding level, not bitwise (resuming twice from
  the same MDState is bitwise identical, and get_state alone never
  perturbs the trajectory).
- Integrator seed: a stochastic integrator's seed is derived from
  `derive_rng(rng_key, "openmm_seed")` as an int in [1, 2**31 - 1] -- never
  0, which OpenMM interprets as "pick a random seed".
- Physics rule (design section 9): `purpose == "measurement"` with a
  Langevin friction > 0.1 / ps is rejected (ValueError) at construction (and
  for any cfg passed to `build`). Equilibration may use any friction.
- Constraints: `cfg.constraints` / `cfg.rigid_water` describe how the given
  System was built; this backend cannot change them. They are checked
  against the System when a Topology is available: with water residues
  recognised as described below, "rigid water" is recognised by an H-H constraint inside a
  water residue, and every remaining constraint must be exactly the set of
  topology bonds implied by the mode (none: no bonds; hbonds: bonds with a
  hydrogen; allbonds: all bonds; plus the water O-H bonds when rigid_water).
  Without a Topology only the "none + not rigid_water => zero constraints"
  rule is checkable.
- Platforms: Reference is double-only (cfg.precision must be "double"); the
  CPU platform has fixed native arithmetic and no double mode (cfg.precision
  "double" is rejected; "mixed"/"single" are recorded as requested);
  CUDA gets Precision=cfg.precision. When cfg.deterministic_forces, CUDA and
  CPU get DeterministicForces="true" and CPU additionally Threads="1".
  CUDA measurement runs must set deterministic_forces (ruling R22).
  precision="single" is rejected for purpose="measurement" on every platform
  (review C-M7: NVE energy conservation is poor in single precision and the
  1e-10 velocity projection of R28 is unreachable there; use "mixed").
- Constraint tolerance: every integrator gets an explicit
  `CONSTRAINT_TOLERANCE` (1e-5, OpenMM's default), recorded in provenance.
- Water (constraint check) is recognised by residue name (HOH/WAT) or by
  composition -- one O, two H, and at most two massless extra points -- so
  CHARMM "TIP3", GROMACS "SOL", "T3P", "SPC", TIP4P-style waters are all
  checked (review C-M5).
- System-resident stochastic forces (review C-I1): any Force in the System
  that carries a random-number seed (MonteCarloBarostat,
  MonteCarloAnisotropicBarostat, MonteCarloMembraneBarostat,
  MonteCarloFlexibleBarostat, AndersenThermostat, ... -- detected
  generically by `setRandomNumberSeed`) acts on the dynamics with its own
  RNG. purpose="measurement" rejects such a System (ValueError; design
  section 9 allows only NVE / low-friction Langevin / Nose-Hoover, and a
  seed-0 force would be unseeded randomness). For purpose="equilibration"
  each build deep-copies the System and sets force i's seed to
  `derive_rng(rng_key, "openmm_force_seed/<i>")` as an int in
  [1, 2**31 - 1] (never 0) on the copy -- the caller's System is never
  mutated; the seeds are on `propagator.force_seeds` and in provenance. The
  single-point Context of `energy_forces` uses a copy of the System without
  those forces (they never contribute energy, and it never steps).
- OpenMM's process-global RNG (review C-M1, fixreview-p5 N1): on Reference
  the LangevinMiddle kernel, on Reference and CPU the AndersenThermostat
  kernel, and on *every* platform (CUDA included) the MonteCarlo barostats
  (`MonteCarlo{,Anisotropic,Membrane,Flexible}Barostat`, and the RPMD one;
  their Impl is platform-independent core code that seeds and draws from
  `SimTKOpenMMUtilities`) use one process-global RNG that is re-seeded
  whenever such a Context is created. A propagator built with any of them
  therefore refuses to `run()` (RuntimeError) once another such Context has
  been created by this module after it -- it would silently continue on the
  other key's noise. Contexts created outside this module cannot be
  tracked. CPU/CUDA LangevinMiddle and CUDA Andersen are per-Context and
  need no guard. The single-point Context of `energy_forces` never holds a
  stochastic force, so it never re-seeds.
- Numerical blow-ups (contract K10): the CPU and CUDA platforms raise
  ``OpenMMException("Particle coordinate is NaN ...")`` from a step or a
  state read instead of returning a NaN state (Reference returns it). The
  propagator re-raises any OpenMMException whose message reports NaN or an
  infinite value as `cytherea.backends.base.NumericalInstabilityError`
  (chained), which the engine records as a ``"nonfinite"`` stop; every
  other OpenMMException propagates unchanged.
- `set_state` validates everything before the first mutation: shapes,
  finiteness of x, v, t and box (review C-M3), and -- for Verlet
  measurement in a System with a CMMotionRemover -- that the IC has no net
  momentum (|sum m v| <= 1e-3 * sqrt(sum (m v)^2); review C-M6: the
  remover would otherwise change the IC at the first step, and velocity
  reversal would miss at the percent level; the IC sampler removes the COM
  momentum by default). `set_state(box=None)` on a periodic System means
  the System's default box, exactly as for `energy_forces` (review C-M2).
  If the System has virtual sites they are recomputed from the real atoms
  after every `setPositions` (set_state and energy_forces; review C-M4), so
  get_state().x of a virtual site may differ from the x that was set.
- `energy_forces(x)` uses one dedicated Context (created on first use and
  reused -- it never steps) on the *configured* platform with the same
  properties, so e.g. CUDA force reproducibility is measured on CUDA.
  `energy_forces(x, box)` evaluates in `box` (pass `state.box` for a
  propagated state; ruling R20); box=None means the System's default box,
  re-applied on every call so no box leaks from a previous call.
- Provenance (review C-I2, contract K8): `build` records (a copy of) the cfg
  it actually used, its integrator seed and force seeds.
  `provenance(cfg=None)` describes `cfg`, or with cfg=None the cfg of the
  most recent build (the constructor cfg before any build); the seeds are
  included when the most recent build used that cfg. `effective_config(cfg=
  None)` (None -> the constructor cfg, i.e. what `build(s, None, key)` runs)
  is a plain, JSON-serialisable, key-independent dict of the settings in
  effect: integrator, dt, temperature and friction (None / 0.0 for Verlet,
  which has neither), constraints, rigid water, platform, precision,
  deterministic forces, platform properties, purpose, constraint tolerance,
  and the seeding rule of the integrator and of every System stochastic
  force (the seed values themselves are per key, so they live in
  provenance, not in this config description). Contract K9: it also holds
  `system_sha256` (sha256 of the serialised System as given to the
  constructor; see `system_sha256()`) and `topology_sha256` (canonical
  atoms/residues/bonds; None without a Topology), both computed once at
  construction (about 0.6 s at 1e5 atoms; `backend.identity_hash_s`), so
  `physics_config_hash` changes with the force field, water model or any
  System parameter. Provenance repeats both digests.
- Build cost (review C-I3): `build` creates a fresh Context per call. Context
  reuse is deliberately NOT implemented: LangevinMiddle's (and any System
  force's) seed is read only when a Context is created or reinitialize()d,
  so a reused Context could not honour the per-shot derive_rng key and the
  keyed-RNG reproducibility contract would break; for Verlet reuse would be
  safe but needs the step count and CMMotionRemover phase reset too. The
  CUDA Context-creation cost (estimated 0.3-2 s, possibly dominating 1-2 ps
  A1 shots) must first be measured on an idle GPU (a later task) before
  deciding. What does not depend on the key is computed once per backend
  and cached: inverse masses, the virtual-site / CMMotionRemover /
  stochastic-force scans, and the constraint-consistency verdict per
  (constraints, rigid_water). Every build records wall times in
  `propagator.build_timing` and `backend.last_build_timing` (validate_s,
  system_prep_s, context_creation_s, set_state_s, total_s) -- deliberately
  not in provenance(), whose content must be identical for identical keys.
"""


from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import math
import re
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
import openmm
import openmm.unit as u

from cytherea.backends.base import MDState, NumericalInstabilityError, PhysicsConfig
from cytherea.keys import Key, derive_rng

_INTEGRATORS = ("verlet", "langevin_middle", "nose_hoover")
_CONSTRAINTS = ("none", "hbonds", "allbonds")
_PLATFORMS = ("CUDA", "CPU", "Reference")
_PRECISIONS = ("mixed", "double", "single")
_PURPOSES = ("equilibration", "measurement")

# Design section 9: measurement dynamics must be NVE, Nose-Hoover, or
# low-friction Langevin with gamma <= 0.1 / ps.
MAX_MEASUREMENT_FRICTION_PER_PS = 0.1

_WATER_RESIDUES = frozenset({"HOH", "WAT"})
# at most this many massless extra points (TIP4P: 1, TIP5P: 2) in a water
_MAX_WATER_EXTRA_POINTS = 2

# Explicit (== OpenMM's default) constraint tolerance for every integrator.
CONSTRAINT_TOLERANCE = 1e-5

# Tolerance for the velocity projection done in get_state (R28). Much
# tighter than CONSTRAINT_TOLERANCE so that get_state().v satisfies the
# constraints to near rounding and a set/get round trip is exact to ~1e-10;
# it is one solve per get_state call, never per step, so the extra
# iterations cost nothing measurable.
PROJECTION_TOLERANCE = 1e-10

# set_state, Verlet measurement + CMMotionRemover: largest accepted
# |sum m v| / sqrt(sum (m v)^2). Independent Maxwell-Boltzmann draws give
# O(1) for any N (E|sum p|^2 = sum E|p_i|^2; fixreview-p5 m2);
# states produced by dynamics with a CMMotionRemover give ~1e-5 (PME force
# non-conservation in single precision, measured on the A1 system) or
# rounding (~1e-15, vacuum / double).
MAX_NET_MOMENTUM_RATIO = 1e-3

# com_distance: a group whose unwrapped extent along any box axis exceeds
# this fraction of the box edge cannot be reliably made whole.
_MAX_GROUP_EXTENT_FRACTION = 0.45

_SEED_MAX = 2**31 - 1


def _derived_seed(rng_key: Key, substream: str) -> int:
    rng = derive_rng(rng_key, substream)
    return int(rng.integers(1, _SEED_MAX, endpoint=True))


def integrator_seed(rng_key: Key) -> int:
    """OpenMM integrator seed for `rng_key`: an int in [1, 2**31 - 1].

    Never 0 -- OpenMM treats a seed of 0 as "choose a random seed", which
    would silently break reproducibility.
    """
    return _derived_seed(rng_key, "openmm_seed")


def force_seed_substream(index: int) -> str:
    """derive_rng substream for the seed of System force number `index`."""
    return f"openmm_force_seed/{int(index)}"


def force_seed(rng_key: Key, index: int) -> int:
    """Seed (in [1, 2**31 - 1], never 0) for the stochastic System force
    with force index `index`, derived from `rng_key` (review C-I1)."""
    return _derived_seed(rng_key, force_seed_substream(index))


def stochastic_forces(system: openmm.System) -> list[tuple[int, str]]:
    """(force index, class name) of every Force in `system` that carries a
    random-number seed (barostats, AndersenThermostat, ...)."""
    out = []
    for i, f in enumerate(system.getForces()):
        if hasattr(f, "setRandomNumberSeed") and hasattr(f, "getRandomNumberSeed"):
            out.append((i, type(f).__name__))
    return out


def _check_stochastic_forces(stoch: list[tuple[int, str]], cfg: PhysicsConfig) -> None:
    if stoch and cfg.purpose == "measurement":
        listed = ", ".join(f"{name} (force index {i})" for i, name in stoch)
        raise ValueError(
            f"the System contains stochastic force(s) {listed}; purpose='measurement' "
            "dynamics must be NVE, Nose-Hoover or low-friction Langevin at fixed volume "
            "(design section 9) -- remove them from the System, or use "
            "purpose='equilibration' (their seeds are then derived from the rng key)"
        )


def validate_physics_config(cfg: PhysicsConfig) -> None:
    """Raise ValueError for a PhysicsConfig this backend must not run."""
    if cfg.integrator not in _INTEGRATORS:
        raise ValueError(f"unknown integrator {cfg.integrator!r}; expected one of {_INTEGRATORS}")
    if cfg.constraints not in _CONSTRAINTS:
        raise ValueError(f"unknown constraints {cfg.constraints!r}; expected one of {_CONSTRAINTS}")
    if cfg.platform not in _PLATFORMS:
        raise ValueError(f"unknown platform {cfg.platform!r}; expected one of {_PLATFORMS}")
    if cfg.precision not in _PRECISIONS:
        raise ValueError(f"unknown precision {cfg.precision!r}; expected one of {_PRECISIONS}")
    if cfg.purpose not in _PURPOSES:
        raise ValueError(f"unknown purpose {cfg.purpose!r}; expected one of {_PURPOSES}")
    if not (math.isfinite(cfg.dt_ps) and cfg.dt_ps > 0.0):
        raise ValueError(f"dt_ps must be finite and > 0, got {cfg.dt_ps!r}")
    if cfg.integrator in ("langevin_middle", "nose_hoover"):
        if not (math.isfinite(cfg.temperature_K) and cfg.temperature_K > 0.0):
            raise ValueError(
                f"{cfg.integrator} needs temperature_K > 0, got {cfg.temperature_K!r}"
            )
        if not (math.isfinite(cfg.friction_per_ps) and cfg.friction_per_ps >= 0.0):
            raise ValueError(
                f"friction_per_ps must be finite and >= 0, got {cfg.friction_per_ps!r}"
            )
        if cfg.integrator == "nose_hoover" and cfg.friction_per_ps == 0.0:
            raise ValueError(
                "nose_hoover uses friction_per_ps as its collision frequency, which must be > 0"
            )
    if cfg.integrator == "nose_hoover" and cfg.purpose != "equilibration":
        raise ValueError(
            "nose_hoover is allowed only for purpose='equilibration' in Phase A: "
            "MDState is not a complete Nose-Hoover checkpoint (no thermostat "
            "chain state) and its on-step velocity conversion is approximate"
        )
    if cfg.platform == "CUDA" and cfg.purpose == "measurement" and not cfg.deterministic_forces:
        raise ValueError(
            "CUDA measurement runs require deterministic_forces=True "
            "(design section 7: DeterministicForces=true for best-effort reproducibility)"
        )
    if (
        cfg.purpose == "measurement"
        and cfg.integrator == "langevin_middle"
        and cfg.friction_per_ps > MAX_MEASUREMENT_FRICTION_PER_PS
    ):
        raise ValueError(
            f"measurement dynamics with Langevin friction_per_ps={cfg.friction_per_ps} "
            f"> {MAX_MEASUREMENT_FRICTION_PER_PS} is forbidden (design section 9: "
            "gamma = 1/ps is for equilibration only; use NVE, Nose-Hoover, or "
            "gamma <= 0.1/ps)"
        )
    if cfg.platform == "Reference" and cfg.precision != "double":
        raise ValueError(
            f"the Reference platform is double precision only; got precision={cfg.precision!r}"
        )
    if cfg.platform == "CPU" and cfg.precision == "double":
        raise ValueError(
            "the CPU platform has no double-precision mode; use Reference for double"
        )
    if cfg.purpose == "measurement" and cfg.precision == "single":
        raise ValueError(
            "precision='single' is not allowed for purpose='measurement': NVE energy "
            "conservation is poor in single precision and the get_state velocity "
            "projection (tolerance 1e-10) cannot converge in it; use 'mixed'"
        )


def _is_h(atom) -> bool:
    return atom.element is not None and atom.element.symbol == "H"


def _is_water_residue(residue) -> bool:
    """Water by name (HOH/WAT) or by composition: exactly one O and two H
    atoms plus at most two element-less extra points (review C-M5)."""
    if residue.name in _WATER_RESIDUES:
        return True
    symbols = []
    n_extra = 0
    for a in residue.atoms():
        if a.element is None:
            n_extra += 1
        else:
            symbols.append(a.element.symbol)
    return sorted(symbols) == ["H", "H", "O"] and n_extra <= _MAX_WATER_EXTRA_POINTS


def _check_constraints(system: openmm.System, topology, cfg: PhysicsConfig) -> None:
    """Check cfg.constraints / cfg.rigid_water against the System (see
    module docstring for the rules). Raises ValueError on mismatch."""
    pairs = set()
    for i in range(system.getNumConstraints()):
        a, b, _ = system.getConstraintParameters(i)
        pairs.add(frozenset((a, b)))

    if topology is None:
        if cfg.constraints == "none" and not cfg.rigid_water and pairs:
            raise ValueError(
                f"cfg.constraints='none' and rigid_water=False, but the System has "
                f"{len(pairs)} constraints"
            )
        return

    atoms = list(topology.atoms())
    if len(atoms) != system.getNumParticles():
        raise ValueError(
            f"topology has {len(atoms)} atoms but the System has "
            f"{system.getNumParticles()} particles"
        )

    waters = [r for r in topology.residues() if _is_water_residue(r)]
    water_ids = {r.index for r in waters}

    def is_water(atom) -> bool:
        return atom.residue.index in water_ids

    rigid_waters = 0
    for r in waters:
        hs = [a.index for a in r.atoms() if _is_h(a)]
        if len(hs) == 2 and frozenset(hs) in pairs:
            rigid_waters += 1
    if cfg.rigid_water and waters and rigid_waters != len(waters):
        raise ValueError(
            f"cfg.rigid_water=True but only {rigid_waters}/{len(waters)} water "
            "residues carry an H-H constraint (System built with flexible water?)"
        )
    if not cfg.rigid_water and rigid_waters:
        raise ValueError(
            f"cfg.rigid_water=False but {rigid_waters} water residues are rigid "
            "(H-H constrained) in the System"
        )

    expected = set()
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        water_bond = is_water(a) and is_water(b) and a.residue is b.residue
        if water_bond and cfg.rigid_water:
            expected.add(frozenset((a.index, b.index)))
        elif cfg.constraints == "allbonds":
            expected.add(frozenset((a.index, b.index)))
        elif cfg.constraints == "hbonds" and (_is_h(a) or _is_h(b)):
            expected.add(frozenset((a.index, b.index)))
    actual = set()
    for p in pairs:
        i, j = tuple(p)
        # H-H pair inside a water residue: the rigid-water constraint, handled above
        if is_water(atoms[i]) and is_water(atoms[j]) and _is_h(atoms[i]) and _is_h(atoms[j]):
            continue
        actual.add(p)
    if actual != expected:
        raise ValueError(
            f"System constraints are inconsistent with cfg.constraints="
            f"{cfg.constraints!r}, rigid_water={cfg.rigid_water}: expected "
            f"{len(expected)} bond constraints, System has {len(actual)} "
            f"({len(actual - expected)} unexpected, {len(expected - actual)} missing)"
        )


def _platform_properties(cfg: PhysicsConfig) -> dict[str, str]:
    """The platform properties this backend requests for `cfg` (pure)."""
    props: dict[str, str] = {}
    if cfg.platform == "CUDA":
        props["Precision"] = cfg.precision
        if cfg.deterministic_forces:
            props["DeterministicForces"] = "true"
    elif cfg.platform == "CPU":
        if cfg.deterministic_forces:
            props["DeterministicForces"] = "true"
            props["Threads"] = "1"
    return props


def _platform_and_properties(cfg: PhysicsConfig) -> tuple[openmm.Platform, dict[str, str]]:
    return openmm.Platform.getPlatformByName(cfg.platform), _platform_properties(cfg)


def _make_integrator(cfg: PhysicsConfig, rng_key: Key) -> tuple[openmm.Integrator, int | None]:
    dt = cfg.dt_ps * u.picosecond
    seed = None
    if cfg.integrator == "verlet":
        integ = openmm.VerletIntegrator(dt)
    elif cfg.integrator == "langevin_middle":
        integ = openmm.LangevinMiddleIntegrator(
            cfg.temperature_K * u.kelvin, cfg.friction_per_ps / u.picosecond, dt
        )
        seed = integrator_seed(rng_key)
        integ.setRandomNumberSeed(seed)
    else:  # nose_hoover: deterministic, no seed
        integ = openmm.NoseHooverIntegrator(
            cfg.temperature_K * u.kelvin, cfg.friction_per_ps / u.picosecond, dt
        )
    integ.setConstraintTolerance(CONSTRAINT_TOLERANCE)
    return integ, seed


def _masses(system: openmm.System) -> np.ndarray:
    return np.array(
        [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())],
        dtype=np.float64,
    )


def _inverse_masses(system: openmm.System) -> np.ndarray:
    """(n, 1) array of 1/m in 1/dalton; 0 for massless (fixed) particles."""
    m = _masses(system)
    inv = np.zeros_like(m)
    inv[m > 0.0] = 1.0 / m[m > 0.0]
    return inv[:, None]


def _has_virtual_sites(system: openmm.System) -> bool:
    return any(system.isVirtualSite(i) for i in range(system.getNumParticles()))


def _box_vec3(box: np.ndarray) -> tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3]:
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (3, 3):
        raise ValueError(f"box must have shape (3, 3), got {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("box must be finite")
    return openmm.Vec3(*b[0]), openmm.Vec3(*b[1]), openmm.Vec3(*b[2])


def _as_nx3(a: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3), got {a.shape}")
    return a


def _restore_or_note(exc: BaseException, restore: Callable[[], None]) -> None:
    """Run `restore` after `exc`; if the restore itself fails, attach that
    to `exc` as a note instead of letting it replace `exc` (review C-M11)."""
    try:
        restore()
    except Exception as restore_exc:  # pragma: no cover - exercised via fault injection
        exc.add_note(
            f"restoring the Context after this failure also failed ({restore_exc!r}); "
            "the Context state is undefined -- rebuild the propagator"
        )


# -- OpenMM's process-global RNG (review C-M1) --------------------------------
# Incremented whenever this module creates a Context whose kernels (re-)seed
# OpenMM's process-global RNG.
_global_rng_generation = 0


def _uses_process_global_rng(cfg: PhysicsConfig, stochastic_types: set[str]) -> bool:
    if cfg.platform == "Reference" and cfg.integrator == "langevin_middle":
        return True
    if any("MonteCarlo" in name and "Barostat" in name for name in stochastic_types):
        return True  # every platform (fixreview-p5 N1)
    return cfg.platform in ("Reference", "CPU") and "AndersenThermostat" in stochastic_types


_INSTABILITY_MESSAGE = re.compile(r"\bNaN\b|\binfinite\b", re.IGNORECASE)


@contextlib.contextmanager
def _numerical_instability_as_k10():
    """Re-raise OpenMM's NaN/infinity exceptions as NumericalInstabilityError
    (contract K10); anything else propagates unchanged."""
    try:
        yield
    except openmm.OpenMMException as exc:
        if _INSTABILITY_MESSAGE.search(str(exc)):
            raise NumericalInstabilityError(str(exc)) from exc
        raise


def _next_global_rng_generation() -> int:
    global _global_rng_generation
    _global_rng_generation += 1
    return _global_rng_generation


class OpenMMPropagator:
    """A Context + integrator pair. `run(n)` just steps; state is read and
    written through `MDState` (nm, nm/ps, ps, box nm), converting between
    OpenMM's staggered v(t - dt/2) and the on-step v(t) (module docstring).

    Built by `OpenMMBackend.build`, which also sets `cfg` (the PhysicsConfig
    this propagator runs), `seed` (integrator seed or None), `force_seeds`
    ({force index: seed} of System stochastic forces) and `build_timing`.
    """

    def __init__(
        self,
        context: openmm.Context,
        integrator: openmm.Integrator,
        seed: int | None,
        periodic: bool,
        *,
        cfg: PhysicsConfig | None = None,
        inv_m: np.ndarray | None = None,
        has_virtual_sites: bool | None = None,
        net_momentum_masses: np.ndarray | None = None,
        force_seeds: dict[int, int] | None = None,
        global_rng_generation: int | None = None,
    ) -> None:
        self.context = context
        self.integrator = integrator
        self.seed = seed
        self.cfg = cfg
        self.force_seeds: dict[int, int] = dict(force_seeds or {})
        self.build_timing: dict[str, float] = {}
        self._periodic = periodic
        system = context.getSystem()
        self._n = system.getNumParticles()
        self._inv_m = _inverse_masses(system) if inv_m is None else inv_m
        self._has_vsites = _has_virtual_sites(system) if has_virtual_sites is None else has_virtual_sites
        self._momentum_masses = net_momentum_masses
        self._default_box = system.getDefaultPeriodicBoxVectors() if periodic else None
        self._constrained = system.getNumConstraints() > 0
        self._dt = float(integrator.getStepSize().value_in_unit(u.picosecond))
        self._half_dt = 0.5 * self._dt
        self._global_rng_generation = global_rng_generation

    @property
    def dt(self) -> float:
        """Read-only: the integrator's own step size in ps, as resolved by
        the `PhysicsConfig` actually passed to `OpenMMBackend.build()`
        (controller ruling R34). Fixed for this propagator's lifetime;
        there is no setter.
        """
        return self._dt

    def run(self, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if (
            self._global_rng_generation is not None
            and self._global_rng_generation != _global_rng_generation
        ):
            raise RuntimeError(
                "another Context that draws from OpenMM's process-global RNG (Reference "
                "LangevinMiddle, AndersenThermostat on Reference/CPU, or a MonteCarlo "
                "barostat on any platform) was created after "
                "this propagator and re-seeded that global RNG; continuing would silently "
                "use the other key's noise stream. Build a new propagator instead."
            )
        if n_steps:
            with _numerical_instability_as_k10():
                self.integrator.step(int(n_steps))

    def get_state(self) -> MDState:
        with _numerical_instability_as_k10():
            st = self.context.getState(getPositions=True, getVelocities=True, getForces=True)
        x = np.array(st.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=np.float64)
        v_half = np.array(
            st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond),
            dtype=np.float64,
        )
        F = np.asarray(
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer),
            dtype=np.float64,
        )
        v = v_half + F * self._inv_m * self._half_dt
        if self._constrained:
            # Project the on-step velocity onto the velocity constraints
            # (ruling R28): let OpenMM do it on the Context, read the result,
            # then restore the untouched v_half so the trajectory continues
            # bitwise unchanged -- also when the projection raises (the
            # original exception propagates after v_half is restored).
            try:
                self.context.setVelocities(v)
                self.context.applyVelocityConstraints(PROJECTION_TOLERANCE)
                v = np.array(
                    self.context.getState(getVelocities=True)
                    .getVelocities(asNumpy=True)
                    .value_in_unit(u.nanometer / u.picosecond),
                    dtype=np.float64,
                )
            except BaseException as exc:
                _restore_or_note(exc, lambda: self.context.setVelocities(v_half))
                raise
            self.context.setVelocities(v_half)
        box = None
        if self._periodic:
            box = np.array(
                st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer),
                dtype=np.float64,
            )
        return MDState(x=x, v=v, t=float(st.getTime().value_in_unit(u.picosecond)), box=box)

    def set_state(self, s: MDState) -> None:
        """Load `s` into the Context. Atomic: all inputs are validated before
        the first mutation (shapes, finiteness, net momentum where checked),
        and if anything fails afterwards the Context's previous positions,
        velocities, box and time are restored before the original exception
        is re-raised. `s.box=None` on a periodic System means the System's
        default box."""
        if s.box is not None:
            if not self._periodic:
                raise ValueError("MDState has a box but the System is not periodic")
            box = _box_vec3(s.box)
        else:
            box = self._default_box  # None for a non-periodic System
        x = _as_nx3(s.x, self._n, "x")
        v = _as_nx3(s.v, self._n, "v")
        t = float(s.t)
        if not np.all(np.isfinite(x)):
            raise ValueError("MDState.x must be finite")
        if not np.all(np.isfinite(v)):
            raise ValueError("MDState.v must be finite")
        if not math.isfinite(t):
            raise ValueError(f"MDState.t must be finite, got {t!r}")
        if self._momentum_masses is not None:
            p = self._momentum_masses[:, None] * v
            scale = float(np.sqrt(np.sum(p * p)))
            net = float(np.linalg.norm(p.sum(axis=0)))
            if scale > 0.0 and net > MAX_NET_MOMENTUM_RATIO * scale:
                raise ValueError(
                    f"the initial velocities carry a net momentum (|sum m v| / "
                    f"sqrt(sum (m v)^2) = {net / scale:.3e} > {MAX_NET_MOMENTUM_RATIO}) but "
                    "the System has a CMMotionRemover, which would silently change them at "
                    "the first step of this Verlet measurement run (and break velocity "
                    "reversal); remove the COM momentum from the IC"
                )

        prev = self.context.getState(getPositions=True, getVelocities=True)

        def restore() -> None:
            self.context.setPeriodicBoxVectors(*prev.getPeriodicBoxVectors())
            self.context.setPositions(prev.getPositions(asNumpy=True))
            self.context.setVelocities(prev.getVelocities(asNumpy=True))
            self.context.setTime(prev.getTime())

        try:
            if box is not None:
                self.context.setPeriodicBoxVectors(*box)
            self.context.setPositions(x)
            if self._has_vsites:
                self.context.computeVirtualSites()
            F = np.asarray(
                self.context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(u.kilojoule_per_mole / u.nanometer),
                dtype=np.float64,
            )
            self.context.setVelocities(v - F * self._inv_m * self._half_dt)
            self.context.applyVelocityConstraints(self.integrator.getConstraintTolerance())
            self.context.setTime(t)
        except BaseException as exc:
            _restore_or_note(exc, restore)
            raise


_OPENMM_VERSION_ATTR = re.compile(r' openmmVersion="[^"]*"')


def system_sha256(system: openmm.System) -> str:
    """sha256 of ``XmlSerializer.serialize(system)`` (contract K9): every
    particle mass, constraint, Force and parameter (charges, LJ sigma/epsilon,
    cutoffs, PME settings, ...) and the default box. Two normalisations keep
    it a pure description of the physics: the ``openmmVersion`` attribute of
    the root element is removed (the version is in provenance; a serializer
    *format* change across OpenMM versions still changes the digest -- the
    safe direction, a resume refusal), and the random-number seed of every
    System stochastic force is set to 0 on a copy before serialising (each
    build re-seeds those from the key, so the caller's seed value never
    drives the dynamics). Deterministic across processes and hosts for the
    same OpenMM version. Costs one serialisation (about 0.6 s for 1e5
    atoms), so `OpenMMBackend` computes it once, at construction."""
    stochastic = stochastic_forces(system)
    if stochastic:
        system = copy.deepcopy(system)
        for i, _ in stochastic:
            system.getForce(i).setRandomNumberSeed(0)
    xml = openmm.XmlSerializer.serialize(system)
    xml = _OPENMM_VERSION_ATTR.sub("", xml, count=1)
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()


def topology_sha256(topology) -> str | None:
    """sha256 of a canonical description of an `openmm.app.Topology`
    (contract K9): per atom, in index order, its chain index and id, residue
    name, id and insertion code, atom name and element symbol; then every
    bond as a sorted (i, j) index pair, in sorted order. Topology-level box
    vectors and bond types/orders are not included (the System carries the
    box; bond orders do not enter the dynamics). None -> None."""
    if topology is None:
        return None
    h = hashlib.sha256()
    for ci, chain in enumerate(topology.chains()):
        for res in chain.residues():
            for atom in res.atoms():
                el = atom.element.symbol if atom.element is not None else ""
                fields = (
                    str(atom.index), str(ci), str(chain.id), str(res.name), str(res.id),
                    str(getattr(res, "insertionCode", "")), str(atom.name), el,
                )
                h.update("\x1f".join(fields).encode("utf-8") + b"\x1e")
    h.update(b"bonds\x1e")
    bonds = sorted(tuple(sorted((b[0].index, b[1].index))) for b in topology.bonds())
    for i, j in bonds:
        h.update(f"{i}-{j}\x1e".encode())
    return h.hexdigest()


def _cfg_key(cfg: PhysicsConfig) -> tuple:
    return (cfg.platform, cfg.precision, bool(cfg.deterministic_forces))


class OpenMMBackend:
    """`PotentialBackend` over an existing `openmm.System` (see module
    docstring). `topology` (an `openmm.app.Topology`) is used only for the
    constraint-consistency check and may be None for toy Systems. The
    System must not be modified after the backend is constructed (masses,
    constraints and forces are scanned once and cached)."""

    kind: Literal["openmm"] = "openmm"
    gpu_resident: bool = False  # per instance: True only for platform="CUDA"

    def __init__(self, system: openmm.System, topology, cfg: PhysicsConfig) -> None:
        validate_physics_config(cfg)
        self._stochastic = stochastic_forces(system)
        _check_stochastic_forces(self._stochastic, cfg)
        _check_constraints(system, topology, cfg)
        self._constraints_ok = {(cfg.constraints, bool(cfg.rigid_water))}
        self.system = system
        self.topology = topology
        # contract K9: the identity of the System/Topology as given (before
        # any build), computed once per backend -- never per shot
        t_hash = time.perf_counter()
        self.system_sha256 = system_sha256(system)
        self.topology_sha256 = topology_sha256(topology)
        self.identity_hash_s = time.perf_counter() - t_hash
        self.cfg = dataclasses.replace(cfg)
        self.gpu_resident = cfg.platform == "CUDA"
        self._n = system.getNumParticles()
        self._periodic = bool(system.usesPeriodicBoundaryConditions())
        self._masses = _masses(system)
        inv = np.zeros_like(self._masses)
        inv[self._masses > 0.0] = 1.0 / self._masses[self._masses > 0.0]
        self._inv_m = inv[:, None]
        self._has_vsites = _has_virtual_sites(system)
        self._has_cmm_remover = any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces())
        if self._stochastic:
            # single-point evaluation never steps: drop the stochastic forces
            # (no energy contribution; keeps the global RNG untouched)
            self._energy_system = copy.deepcopy(system)
            for i, _ in sorted(self._stochastic, reverse=True):
                self._energy_system.removeForce(i)
        else:
            self._energy_system = system
        self._energy_context: openmm.Context | None = None
        self._energy_integrator: openmm.Integrator | None = None
        self._platform_values: dict[tuple, tuple[str, dict[str, str]]] = {}
        # (cfg copy, integrator seed, force seeds) of the most recent build
        self._last_build: tuple[PhysicsConfig, int | None, dict[int, int]] | None = None
        self.last_build_timing: dict[str, float] | None = None

    # -- cfg resolution -------------------------------------------------------

    def _resolve_cfg(self, cfg: PhysicsConfig | None) -> PhysicsConfig:
        """`cfg` validated for this System (None -> the constructor cfg)."""
        if cfg is None:
            return self.cfg
        validate_physics_config(cfg)
        _check_stochastic_forces(self._stochastic, cfg)
        ck = (cfg.constraints, bool(cfg.rigid_water))
        if ck not in self._constraints_ok:
            _check_constraints(self.system, self.topology, cfg)
            self._constraints_ok.add(ck)
        return cfg

    # -- propagation ---------------------------------------------------------

    def _system_for_build(self, rng_key: Key) -> tuple[openmm.System, dict[int, int]]:
        if not self._stochastic:
            return self.system, {}
        system = copy.deepcopy(self.system)
        seeds = {}
        for i, _ in self._stochastic:
            seeds[i] = force_seed(rng_key, i)
            system.getForce(i).setRandomNumberSeed(seeds[i])
        return system, seeds

    def build(self, s: MDState, cfg: PhysicsConfig | None, rng_key: Key) -> OpenMMPropagator:
        """Fresh Context initialised from `s`. `cfg=None` means the
        constructor's cfg; any other cfg is validated the same way. The cfg
        used, its seeds and the build wall times are recorded (module
        docstring: provenance, build cost)."""
        t0 = time.perf_counter()
        cfg = dataclasses.replace(self._resolve_cfg(cfg))
        t1 = time.perf_counter()
        system, force_seeds = self._system_for_build(rng_key)
        integrator, seed = _make_integrator(cfg, rng_key)
        platform, props = _platform_and_properties(cfg)
        t2 = time.perf_counter()
        context = openmm.Context(system, integrator, platform, props)
        t3 = time.perf_counter()
        generation = None
        if _uses_process_global_rng(cfg, {name for _, name in self._stochastic}):
            generation = _next_global_rng_generation()
        self._remember_platform_values(cfg, context)
        check_momentum = (
            cfg.integrator == "verlet" and cfg.purpose == "measurement" and self._has_cmm_remover
        )
        prop = OpenMMPropagator(
            context,
            integrator,
            seed,
            self._periodic,
            cfg=cfg,
            inv_m=self._inv_m,
            has_virtual_sites=self._has_vsites,
            net_momentum_masses=self._masses if check_momentum else None,
            force_seeds=force_seeds,
            global_rng_generation=generation,
        )
        prop.set_state(s)
        t4 = time.perf_counter()
        timing = {
            "validate_s": t1 - t0,
            "system_prep_s": t2 - t1,
            "context_creation_s": t3 - t2,
            "set_state_s": t4 - t3,
            "total_s": t4 - t0,
        }
        prop.build_timing = timing
        self.last_build_timing = dict(timing)
        self._last_build = (cfg, seed, dict(force_seeds))
        return prop

    # -- single-point evaluation ---------------------------------------------

    def _ctx(self) -> openmm.Context:
        if self._energy_context is None:
            platform, props = _platform_and_properties(self.cfg)
            # never stepped; only used for single-point energies/forces
            self._energy_integrator = openmm.VerletIntegrator(self.cfg.dt_ps * u.picosecond)
            self._energy_integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
            self._energy_context = openmm.Context(
                self._energy_system, self._energy_integrator, platform, props
            )
            self._remember_platform_values(self.cfg, self._energy_context)
        return self._energy_context

    def energy_forces(
        self, x: np.ndarray, box: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        """(E kJ/mol, F kJ/mol/nm with shape (n_atoms, 3), float64) at `x`
        (nm), in periodic cell `box` (nm; None -> the System's default).
        Virtual sites are recomputed from the real atoms first."""
        if box is not None and not self._periodic:
            raise ValueError("a box was given but the System is not periodic")
        x = _as_nx3(x, self._n, "x")
        vecs = _box_vec3(box) if box is not None else None
        ctx = self._ctx()
        if self._periodic:
            ctx.setPeriodicBoxVectors(*(vecs or self.system.getDefaultPeriodicBoxVectors()))
        ctx.setPositions(x)
        if self._has_vsites:
            ctx.computeVirtualSites()
        st = ctx.getState(getEnergy=True, getForces=True)
        E = float(st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        F = np.array(
            st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer),
            dtype=np.float64,
        )
        return E, F

    # -- provenance ------------------------------------------------------------

    def _remember_platform_values(self, cfg: PhysicsConfig, ctx: openmm.Context) -> None:
        key = _cfg_key(cfg)
        if key in self._platform_values:
            return
        platform = ctx.getPlatform()
        values = {}
        for name in platform.getPropertyNames():
            try:
                values[name] = platform.getPropertyValue(ctx, name)
            except Exception:  # pragma: no cover - property not readable on this Context
                pass
        self._platform_values[key] = (platform.getName(), values)

    def _platform_values_for(self, cfg: PhysicsConfig) -> tuple[str, dict[str, str]]:
        key = _cfg_key(cfg)
        if key not in self._platform_values:
            if key == _cfg_key(self.cfg):
                self._ctx()
            else:  # a cfg that was never built: read the values off a throwaway Context
                platform, props = _platform_and_properties(cfg)
                integ = openmm.VerletIntegrator(cfg.dt_ps * u.picosecond)
                ctx = openmm.Context(self._energy_system, integ, platform, props)
                self._remember_platform_values(cfg, ctx)
                del ctx
        return self._platform_values[key]

    def effective_config(self, cfg: PhysicsConfig | None = None) -> dict:
        """Plain, JSON-serialisable description of the settings in effect for
        `cfg` (None -> the constructor cfg, which `build(s, None, key)` uses).
        Key-independent: seeds appear as their derivation rule (contract K8;
        module docstring). Contract K9: it also identifies the System and
        Topology the dynamics run on (`system_sha256`, `topology_sha256`,
        computed once at construction), so `physics_config_hash` changes
        with the force field / water model; the platform and precision are
        the resolved ones. Equal for ``None`` and an equal explicit cfg."""
        cfg = self._resolve_cfg(cfg)
        verlet = cfg.integrator == "verlet"
        return {
            "backend": self.kind,
            "system_sha256": self.system_sha256,
            "topology_sha256": self.topology_sha256,
            "integrator": cfg.integrator,
            "dt_ps": float(cfg.dt_ps),
            "temperature_K": None if verlet else float(cfg.temperature_K),
            "friction_per_ps": 0.0 if verlet else float(cfg.friction_per_ps),
            "constraints": cfg.constraints,
            "rigid_water": bool(cfg.rigid_water),
            "platform": cfg.platform,
            "precision": "double" if cfg.platform == "Reference" else cfg.precision,
            "deterministic_forces": True if cfg.platform == "Reference" else bool(cfg.deterministic_forces),
            "platform_properties": _platform_properties(cfg),
            "purpose": cfg.purpose,
            "constraint_tolerance": CONSTRAINT_TOLERANCE,
            "integrator_seed": (
                "derive_rng(rng_key, 'openmm_seed')" if cfg.integrator == "langevin_middle" else None
            ),
            "system_stochastic_forces": [
                {"index": i, "type": name, "seed": f"derive_rng(rng_key, '{force_seed_substream(i)}')"}
                for i, name in self._stochastic
            ],
        }

    def provenance(self, cfg: PhysicsConfig | None = None) -> dict:
        """Provenance of the dynamics run with `cfg`; None -> the cfg of the
        most recent `build` (the constructor cfg before any build). The
        integrator / force seeds are those of the most recent build if it
        used this cfg, else None (contract K8)."""
        if cfg is None:
            cfg = self._last_build[0] if self._last_build is not None else self.cfg
        cfg = self._resolve_cfg(cfg)
        platform_name, values = self._platform_values_for(cfg)

        def prop(name: str) -> str | None:
            return values.get(name)

        if cfg.platform == "Reference":
            precision = "double"
            deterministic = "true"  # serial double-precision reference code
        else:
            precision = prop("Precision") if cfg.platform == "CUDA" else cfg.precision
            deterministic = prop("DeterministicForces")
        last = self._last_build if (self._last_build is not None and self._last_build[0] == cfg) else None
        verlet = cfg.integrator == "verlet"
        return {
            "kind": self.kind,
            "openmm_version": openmm.version.full_version,
            "openmm_git_revision": openmm.version.git_revision,
            "platform": platform_name,
            "precision": precision,
            "deterministic_forces": deterministic,
            "gpu_name": prop("DeviceName") if cfg.platform == "CUDA" else None,
            "cpu_threads": prop("Threads") if cfg.platform == "CPU" else None,
            "system_sha256": self.system_sha256,
            "topology_sha256": self.topology_sha256,
            "num_particles": self._n,
            "num_constraints": self.system.getNumConstraints(),
            "constraint_tolerance": CONSTRAINT_TOLERANCE,
            "periodic": self._periodic,
            "integrator": cfg.integrator,
            "friction_per_ps": 0.0 if verlet else cfg.friction_per_ps,
            "dt_ps": cfg.dt_ps,
            "temperature_K": None if verlet else cfg.temperature_K,
            "constraints": cfg.constraints,
            "rigid_water": cfg.rigid_water,
            "purpose": cfg.purpose,
            "integrator_seed": last[1] if last is not None else None,
            "stochastic_force_seeds": (
                {str(i): sd for i, sd in last[2].items()} if last is not None else None
            ),
            "effective_config": self.effective_config(cfg),
        }


# -- com_distance -----------------------------------------------------------------

# lattice shifts {-1, 0, 1}^3 for the triclinic minimum-image refinement
_SHIFTS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.float64)


def _group_indices(idx, n: int, name: str) -> np.ndarray:
    a = np.asarray(idx)
    if a.size == 0:
        raise ValueError("atom groups must be non-empty")
    if a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer atom indices, got dtype {a.dtype}")
    a = a.astype(np.intp).ravel()
    if np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"{name}: atom index out of range [0, {n})")
    return a


def _reduced_box(box: np.ndarray) -> np.ndarray:
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (3, 3):
        raise ValueError(f"box must have shape (3, 3), got {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("box must be finite")
    ax, by, cz = b[0, 0], b[1, 1], b[2, 2]
    tol = 1e-12 * max(ax, by, cz, 1.0)
    reduced = (
        b[0, 1] == 0.0 and b[0, 2] == 0.0 and b[1, 2] == 0.0
        and ax > 0.0 and by > 0.0 and cz > 0.0
        and abs(b[1, 0]) <= 0.5 * ax + tol
        and abs(b[2, 0]) <= 0.5 * ax + tol
        and abs(b[2, 1]) <= 0.5 * by + tol
    )
    if not reduced:
        raise ValueError(
            "com_distance needs box vectors in OpenMM's reduced form (a=(ax,0,0), "
            "b=(bx,by,0), c=(cx,cy,cz), |bx|,|cx| <= ax/2, |cy| <= by/2)"
        )
    return b


def com_distance(
    state: MDState, idx_a: np.ndarray, idx_b: np.ndarray, masses: np.ndarray
) -> float:
    """Minimum-image distance (nm) between the mass-weighted centres of mass
    of atom groups `idx_a` and `idx_b`.

    `masses` is the full per-atom mass array (n_atoms,), indexed by the
    groups; each group needs a positive total mass, indices must be integers
    in [0, n_atoms). Each group is first made whole by unwrapping every atom
    relative to the group's first atom with the minimum-image convention;
    the COM-COM vector then gets the minimum-image convention too.
    Orthorhombic and triclinic boxes are supported; a triclinic box must be
    in OpenMM's reduced form (as OpenMM itself requires), and its minimum
    image is exact (reduction plus a search over the 27 neighbouring
    images). Guard against groups that cannot be reliably made whole
    (ValueError): orthorhombic -- a made-whole group whose extent along any
    axis exceeds 0.45 L; triclinic -- an atom farther than 0.45 x the
    shortest lattice vector from the group's first atom. OpenMM keeps
    molecules whole in its positions, so the make-whole step matters only
    for foreign (per-atom wrapped) frames. `state.box is None` -> plain
    Euclidean.
    """
    x = np.asarray(state.x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"state.x must have shape (n, 3), got {x.shape}")
    n = x.shape[0]
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (n,):
        raise ValueError(f"masses must have shape ({n},), got {m.shape}")
    ia = _group_indices(idx_a, n, "idx_a")
    ib = _group_indices(idx_b, n, "idx_b")

    box = None
    ortho = True
    lam1 = None
    if state.box is not None:
        box = _reduced_box(state.box)
        ortho = bool(np.all(box[~np.eye(3, dtype=bool)] == 0.0))
        if not ortho:
            lattice = _SHIFTS @ box
            norms = np.linalg.norm(lattice, axis=1)
            lam1 = float(norms[norms > 0.0].min())

    def min_image(d: np.ndarray) -> np.ndarray:
        """Minimum image of the rows of d (k, 3)."""
        if box is None:
            return d
        if ortho:
            L = np.diag(box)
            return d - L * np.round(d / L)
        d = d - np.round(d[:, 2] / box[2, 2])[:, None] * box[2]
        d = d - np.round(d[:, 1] / box[1, 1])[:, None] * box[1]
        d = d - np.round(d[:, 0] / box[0, 0])[:, None] * box[0]
        cand = d[:, None, :] + (_SHIFTS @ box)[None, :, :]
        best = np.argmin(np.einsum("kij,kij->ki", cand, cand), axis=1)
        return cand[np.arange(len(d)), best]

    def com(idx: np.ndarray) -> np.ndarray:
        w = m[idx]
        if not np.all(np.isfinite(w)) or np.any(w < 0.0) or not w.sum() > 0.0:
            raise ValueError(
                "each group needs finite, non-negative masses with a positive total mass "
                f"(got total {w.sum()!r}; virtual sites have mass 0)"
            )
        xs = x[idx]
        ref = xs[0]
        whole = ref + min_image(xs - ref)
        if box is not None:
            if ortho:
                L = np.diag(box)
                extent = whole.max(axis=0) - whole.min(axis=0)
                if np.any(extent > _MAX_GROUP_EXTENT_FRACTION * L):
                    raise ValueError(
                        f"group extent {extent} nm exceeds {_MAX_GROUP_EXTENT_FRACTION} x box "
                        f"{L} nm; cannot reliably make the group whole"
                    )
            else:
                reach = float(np.max(np.linalg.norm(whole - ref, axis=1)))
                if reach > _MAX_GROUP_EXTENT_FRACTION * lam1:
                    raise ValueError(
                        f"group extent (max distance {reach:.4f} nm from its first atom) exceeds "
                        f"{_MAX_GROUP_EXTENT_FRACTION} x the shortest lattice vector {lam1:.4f} nm; "
                        "cannot reliably make the group whole"
                    )
        return (w[:, None] * whole).sum(axis=0) / w.sum()

    d = min_image((com(ia) - com(ib))[None, :])[0]
    return float(np.linalg.norm(d))
