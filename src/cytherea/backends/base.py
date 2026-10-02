"""The protocol every trajectory-propagation backend implements.

`PotentialBackend` is the seam between "how a trajectory advances" (analytic
toy dynamics for A0 acceptance tests, OpenMM for real molecular systems,
OpenMM+ML, OpenMM+QM later) and everything above it (shooting, committor
estimation, weighted ensemble, ...), which only ever talks to this protocol.

This task (Task 3) defines the protocol, `MDState`, and `PhysicsConfig`, and
implements the `analytic` backend against it. `PhysicsConfig` is defined here
(not in the later OpenMM task) per controller ruling R1: something has to
define it first, and the analytic backend doesn't need any of its fields
(it takes its own parameters from its constructor and ignores `cfg`).
`PhysicsConfig` carries no validation logic here -- e.g. the rule that
measurement dynamics may not use `friction_per_ps == 1.0` (design doc
section 9: gamma=1/ps is for equilibration only) belongs to the OpenMM
backend that actually builds an OpenMM `Integrator` from these fields, not to
this plain data container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from cytherea.keys import Key


@dataclass
class MDState:
    """The minimal state needed to resume a trajectory: positions, velocities,
    simulation time, and (for periodic systems) the box vectors.

    `v` is the ON-STEP velocity at time `t` -- synchronous with `x` -- for
    every backend (ruling R21). Backends whose integrator internally keeps a
    staggered velocity (e.g. OpenMM's leapfrog-style integrators, which hold
    v(t - dt/2)) convert at the Propagator boundary, so kinetic energies,
    velocity reversal and Maxwell-Boltzmann initial velocities mean the same
    thing on every backend. For constrained systems `v` satisfies the
    velocity constraints.

    Caveat for temperature observables: for Langevin (BAOAB-like /
    LangevinMiddle) integrators the configurational sampling is accurate but
    the on-step velocity is not exactly Maxwell-Boltzmann -- the
    instantaneous temperature computed from `v` is biased low by O(dt^2)
    (review measurement on alanine dipeptide: about -2.7% at 1 fs without
    constraints, about -4.7% at 2 fs with HBonds). Temperature observables
    and IC-gate temperature windows built from `v` must account for this.

    Checkpoint caveat: `MDState` is NOT a complete checkpoint for a
    stochastic propagator (Langevin/BAOAB, overdamped). It carries no RNG
    state, so `set_state(s)` on a fresh `build(...)` of the same key replays
    the noise stream from its start rather than continuing it. Resuming a
    partially run trajectory from an `MDState` in a new process is therefore
    not bitwise identical to the uninterrupted run. This is harmless for the
    current per-shot resume granularity (a shot is rerun from its IC), but
    anything that resumes *within* a trajectory must not rely on it.
    """

    x: np.ndarray
    v: np.ndarray
    t: float
    box: np.ndarray | None = None


class Propagator(Protocol):
    """A stateful object that can advance an `MDState` and be checkpointed.

    `dt` (controller ruling R34) is the propagator's *own* integration
    timestep, in whatever time unit the backend that built it uses (reduced
    units for the analytic backend, ps for OpenMM) -- read-only: it reflects
    however `dt` was actually resolved at `build()` time (for a backend
    whose effective step size depends on the `PhysicsConfig` passed to
    `build`, `dt` is the value that config actually produced, not some
    backend-level default), and callers must treat it as fixed for the
    propagator's lifetime. This is what callers needing the *effective*
    step size (e.g. `run_shot`, to convert an observation cadence into a
    step count) should read, instead of guessing at a same-named attribute
    on the `PotentialBackend` itself.

    A trajectory that blows up numerically either comes back from
    `get_state()` with non-finite x/v (the analytic backend, OpenMM
    Reference), or makes `run()` / `get_state()` raise
    `NumericalInstabilityError` (OpenMM CPU/CUDA, which refuse to return a
    NaN state). The engine treats both alike: the shot or segment stops as
    ``"nonfinite"`` and is recorded (contracts K4, K10). Any other exception
    is a real error and propagates.
    """

    dt: float

    def run(self, n_steps: int) -> None: ...

    def get_state(self) -> MDState: ...

    def set_state(self, s: MDState) -> None: ...


class NumericalInstabilityError(RuntimeError):
    """The dynamics of a propagator became non-finite (NaN/inf coordinates,
    velocities or forces) and the backend cannot return that state (contract
    K10; e.g. OpenMM's "Particle coordinate is NaN" on the CPU and CUDA
    platforms). `run_shot` / `run_segment` record it as a ``"nonfinite"``
    stop; backends raise it only for numerical blow-ups, never for other
    failures. The propagator is unusable afterwards."""


@dataclass
class PhysicsConfig:
    """Dynamics configuration for a `PotentialBackend.build(...)` call.

    Exact fields per controller ruling R1 (copied verbatim from the later
    OpenMM task so that type exists before anything needs to reference it).
    No validation is performed here -- see module docstring.
    """

    integrator: Literal["verlet", "langevin_middle", "nose_hoover"]
    dt_ps: float
    temperature_K: float
    friction_per_ps: float
    constraints: Literal["none", "hbonds", "allbonds"]
    rigid_water: bool
    platform: Literal["CUDA", "CPU", "Reference"]
    precision: Literal["mixed", "double", "single"]
    deterministic_forces: bool
    purpose: Literal["equilibration", "measurement"]


@runtime_checkable
class PotentialBackend(Protocol):
    """Anything that can compute forces and build a `Propagator` for an
    `MDState`. `kind` and `gpu_resident` are plain data attributes (not
    methods) so callers can branch on backend capabilities without calling
    into the backend.

    Protocol additions (fix wave 2026-10-01, contract K8):
    - `effective_config(cfg=None)`: a plain, JSON-serialisable dict of every
      parameter that actually drives the dynamics of a propagator built with
      `build(s, cfg, key)`; `cfg=None` means the backend-level defaults
      (e.g. OpenMM's constructor cfg), i.e. what `build(s, None, key)` runs.
      `run_shot` / `run_segment` / `run_batch` record
      `config_hash(effective_config(cfg))` as `physics_config_hash`, for
      `cfg=None` and an explicit cfg alike (contract K9,
      `cytherea.engine.shot.resolve_physics_config`). It must therefore
      cover *every* input that determines the dynamics -- for an analytic
      backend the potential's type and all its parameters (raise TypeError
      rather than fall back to a repr), integrator, dt, gamma, kT, masses;
      for OpenMM every cfg field plus digests of the System and Topology,
      the platform and the precision -- and must be cheap: anything
      expensive (serialising a large System) is computed once per backend,
      since it is called for every shot.
      Backends that ignore `cfg` (analytic) accept and ignore it.
      The dict must be deterministic: no dependence on call history or on
      any rng key (per-key seeds belong in provenance, not here).
    - `provenance(cfg=None)`: describes the backend and the configuration
      in effect for `cfg` (for OpenMM, cfg=None means the cfg of the most
      recent build). Backends that ignore `cfg` (analytic) accept and ignore
      it.
    """

    kind: Literal["analytic", "openmm", "openmm+ml", "openmm+qm"]
    gpu_resident: bool

    def build(
        self, s: MDState, cfg: PhysicsConfig | None, rng_key: Key
    ) -> Propagator: ...

    def energy_forces(
        self, x: np.ndarray, box: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        """Single-point energy and forces at `x`. `box` (3x3, rows = box
        vectors) is the periodic cell to evaluate in -- pass `state.box` when
        evaluating a propagated state; None means the backend's default cell
        (ignored by non-periodic backends)."""
        ...

    def effective_config(self, cfg: PhysicsConfig | None = None) -> dict:
        """Plain dict of the parameters that drive `build(s, cfg, key)`'s
        dynamics (cfg=None -> backend defaults; K8). See the class
        docstring."""
        ...

    def provenance(self, cfg: PhysicsConfig | None = None) -> dict:
        """Description of the backend and of the configuration in effect for
        `cfg` (K8; see the class docstring for what cfg=None means)."""
        ...
