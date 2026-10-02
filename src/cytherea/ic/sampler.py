"""The initial-condition sampler and validity gate (Task 5; full-review fix P2).

`EnsembleFrameSampler.sample(key)` picks a frame, draws Maxwell-Boltzmann
velocities for it, applies the holonomic constraints (if any), and runs the
result through the gate `validate`.

Frame selection (contract K3)
-----------------------------
- `key.frame_id >= 0`: exactly that frame (`EnsembleFramePool.get`). An
  unknown id is a `KeyError`; a frame outside the sampler's `state` is a
  `ValueError`. This is the enumerated design: every shot of frame k is
  launched from frame k, and estimators weight shots by the recorded
  `meta["frame_weight"]` (e.g. w_k / K_k when frame k has K_k shots).
- `key.frame_id == -1`: one weighted draw, probability proportional to
  `frame.weight` within `state`, from `derive_rng(key, "ic/frame")`. This is
  the importance-sampled design: frames are already drawn in proportion to
  w, so every shot carries weight 1. Do NOT weight these shots by
  `frame_weight` again, or w is counted twice.
- Any other value is a `ValueError`. Which design produced a record is
  visible from its key (`frame_id == -1` or not).

Rejection and redraw (contract K3)
----------------------------------
Every failure reason is either coordinate-level or velocity-level
(`COORDINATE_REASONS`, `VELOCITY_REASONS`):
- A coordinate-level failure (non-finite x, invalid box, non-finite energy
  or forces, energy window, `min_pair_dist`, constraint residual) is
  deterministic for the frame. `sample` never switches to another frame,
  because that would silently renormalise the frame weights over the frames
  that happen to pass. It raises `ICRejectedError` at once, with
  `level="coordinate"`, the frame id, and the reasons.
- A velocity-level failure (non-finite v, instantaneous temperature) redraws
  only the velocities, for the same frame, from
  `derive_rng(key, f"ic/velocities/{k}")` for attempt k = 0, 1, ...,
  `max_redraws`. When every attempt fails, `ICRejectedError` is raised with
  `level="velocity"`.
Every rejected attempt is kept, as `{"attempt", "frame_id", "reasons"}`, in
`ValidityReport.rejected_attempts` (on success) or `ICRejectedError.attempts`
(on failure), so no rejection reason is thrown away.
`prevalidate_pool()` runs the coordinate-level checks once per frame and
reports the fraction of pool weight that can never pass. Record it in the
run's provenance: those frames' weight is excluded from every estimate.

Initial state (contracts K1, K2)
--------------------------------
`InitialState.state.t` is always 0.0: the shot clock starts at the IC.
`EnsembleFrame.time` is provenance and goes to `meta["frame_time"]`.
`InitialState.meta` has exactly the keys `frame_id`, `frame_time`,
`frame_weight`, `source_id`, `topology_ref`, `state` (the pool's
`state_of(frame)`, None without a state partition) and `n_redraws`.

The gate (`validate`)
---------------------
Design doc section 7 calls out the 1996 VENUS/h2oleps bug: a NaN initial
condition passed a numeric gate silently, because `NaN >= tol` evaluates to
`False`. A comparison-based gate that only rejects on true comparisons
accepts NaN by default. `validate` checks `np.isfinite` first, and every
numeric comparison is written so that NaN fails it.

`validate` always evaluates `backend.energy_forces(x, box)` once, whether or
not an `energy_window` is configured, and rejects non-finite energy or forces
(`nonfinite_energy`). The cost is one energy+force evaluation per validated
state, i.e. per attempt; that is negligible next to the shot it gates. The
energy is recorded in `checks["energy"]`. Because accept/reject can depend on
`energy_forces` near a window edge, a non-deterministic energy (CUDA without
DeterministicForces) can flip a decision and change the IC; measurement runs
must use deterministic forces (ruling R22 #6).

`min_pair_dist` uses `scipy.spatial.cKDTree`: O(N log N) time and O(N)
memory. For a periodic state (`state.box` given) it uses the minimum-image
convention for orthorhombic boxes (coordinates wrapped into [0, L)). A
triclinic box with `min_pair_dist` set is a `ValueError`, as is a threshold
of at least half the shortest box edge.

Constraints
-----------
`constraints` is an object implementing the `Constraints` protocol:
- `n_constraints: int`: the number of independent holonomic constraints;
- `project(x, v) -> (x', v')`: positions projected onto the constraint
  manifold, and velocities projected onto its tangent space. The projection
  must be mass-weighted (as in SHAKE/RATTLE), so that projecting a
  Maxwell-Boltzmann draw gives the constrained Maxwell-Boltzmann
  distribution, and it must conserve total momentum;
- `residual(x, v) -> float`: the dimensionless, relative residual of the
  given state itself (not of its projection). `DistanceConstraints` uses
  max_k max(| |r_k| / d_k - 1 |, |r_k . v_k| / (|r_k| |v_k|)), where
  r_k and v_k are the relative position and velocity of the constrained
  pair: the relative bond-length error (OpenMM's constraint tolerance
  semantics) and the cosine between the relative velocity and the bond.
`validate` computes `constraints.residual` on the state it is validating and
rejects when it exceeds `constraint_tolerance`, which defaults to
`CONSTRAINT_TOLERANCE` = 1e-5 (OpenMM's default constraint tolerance, the
value `OpenMMBackend` sets on every integrator). `sample` projects every
attempt before validating it; `checks["constraint_residual_input"]` records
the residual of the frame's own coordinates before projection. A frame
whose own residual exceeds `input_constraint_tolerance` (default 1e-3,
about 100x what a float32 DCD frame of a 1e-5-tolerance run shows) is
rejected *before* projection with the coordinate-level reason
``"constraint_input"`` (fixreview-p2 N-I1): projecting it would silently
replace the frame with another configuration -- e.g. a water broken across
the periodic edge by per-atom wrapping (OpenMM constrains raw coordinates,
no minimum image) or a frame from a different water geometry.
`prevalidate_pool` applies the same input check. The ``constraint_residual``
of the gate contains a velocity part (the cosine) too; it is still a
coordinate-level reason, since the exact RATTLE velocity solve of `project`
cannot leave a velocity excess.
`DistanceConstraints` is a reference implementation for pair-distance
constraints; `DistanceConstraints.from_openmm_system(system)` builds it from
an OpenMM System's constraints and masses (it projects to 1e-10, so the
gate's 1e-5 has a wide margin). A projection done through an OpenMM Context
instead must use a tolerance near 1e-10 too (as `get_state` does, R31): at
the integrator's 1e-5, CCMA leaves residuals of 9.6-9.9e-6, within 4 % of
the gate (fixreview-p2 m2).

Energy errors and `min_pair_dist`
-----------------------------------
An exception raised by `backend.energy_forces` for a frame (e.g. OpenMM
refusing a box below twice the cutoff) is deterministic for that frame, so
it is the coordinate-level reason ``"energy_error"`` (message in
``checks["energy_error"]``) rather than an error that aborts a batch
(fixreview-p2 m6). `min_pair_dist` is the smallest distance over *all* atom
pairs, bonded and constrained ones included, so it is always at most the
shortest bond (0.0957 nm O-H in TIP3P, 0.1011 nm in TIP3P-FB): use it only
as a near-coincidence detector (thresholds below about 0.09 nm), and rely
on the energy window for clashes. It supports orthorhombic boxes only (a
triclinic box raises at construction; use ``min_pair_dist=None`` there).

Temperature check
-----------------
The instantaneous temperature uses dof = n_velocity_components -
n_constraints - 3 (the 3 only when the velocities' total momentum is
actually zero, R17b). The +-5 sigma band uses sigma_T = kT sqrt(2 / dof),
which is exact for the Gamma-distributed kinetic energy of an MB draw. Below
30 dof the check is skipped (it is statistically meaningless there).
The check can only catch RNG or mass-array bugs, since the velocities are
drawn at `kT`. To catch a kT unit mix-up (e.g. 300 passed as kT), pass
`boltzmann_constant` (e.g. `KB_KJ_PER_MOL_K`): the constructor then requires
kB * frame.temperature == kT (relative 1e-4) for every frame.
`checks["frame_temperature"]` always records the frame's temperature.

Reproducibility (`derive_rng`, see `cytherea.keys`): the frame draw and each
velocity attempt use their own keyed substreams, so the same `ShotKey`
always gives the same attempts and the same accepted state, bit for bit.
Masses must be finite and positive; massless particles (OpenMM virtual
sites) are rejected at construction.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from cytherea.backends.base import MDState, PotentialBackend
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.keys import Key, derive_rng

# Below this many velocity degrees of freedom, the instantaneous-temperature
# check is statistically meaningless (the +-5 sigma band it would use is
# derived asymptotically) and, worse, would *bias* the MB distribution by
# preferentially rejecting draws in the tails of a genuinely-correct sampler
# for small systems. See controller decision in task-5 brief addendum.
_MIN_DOF_FOR_TEMPERATURE_CHECK = 30
_TEMPERATURE_SIGMA_MULTIPLE = 5.0

# Relative tolerance of the constraint residual accepted by the gate. Equal to
# OpenMM's default constraint tolerance and to
# `cytherea.backends.openmm_backend.CONSTRAINT_TOLERANCE` (not imported, so
# this module does not need OpenMM).
CONSTRAINT_TOLERANCE = 1e-5

# Largest relative constraint residual of a frame's own coordinates that
# `sample` projects away (larger: "constraint_input"; fixreview-p2 N-I1).
INPUT_CONSTRAINT_TOLERANCE = 1e-3

# Molar gas constant in kJ/(mol K) (CODATA 2018, the value OpenMM uses), for
# `boltzmann_constant` when kT is in kJ/mol.
KB_KJ_PER_MOL_K = 0.00831446261815324
_FRAME_TEMPERATURE_RTOL = 1e-4

# The machine-readable rejection reasons, partitioned by what a redraw can
# change (contract K3).
COORDINATE_REASONS = frozenset(
    {
        "nonfinite_x",
        "invalid_box",
        "nonfinite_energy",
        "energy_window",
        "min_pair_dist",
        "constraint_residual",
        "constraint_input",
        "energy_error",
    }
)
VELOCITY_REASONS = frozenset({"nonfinite_v", "temperature"})
REASONS = COORDINATE_REASONS | VELOCITY_REASONS


@dataclasses.dataclass
class InitialState:
    """``origin_label``: the (i, j) origin label a sampler assigns (the
    `EncounterSampler`'s state pair); `run_shot` writes it to
    ``ShotRecord.origin_label``. None for plain ensemble frames."""

    state: MDState
    frame_id: int
    meta: dict
    origin_label: tuple[int, int] | None = None


@dataclasses.dataclass
class ValidityReport:
    """Result of `validate` (and of a successful `sample`).

    `checks` holds the measured values (floats; `"skipped_low_dof"` for a
    skipped temperature check). `rejected_attempts` lists every attempt
    rejected before the accepted one, as `{"attempt", "frame_id",
    "reasons"}`; `n_redraws == len(rejected_attempts)` after `sample`.
    """

    ok: bool
    reasons: list[str]
    checks: dict[str, float | str]
    n_redraws: int = 0
    rejected_attempts: list[dict] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "n_redraws": self.n_redraws,
            "checks": dict(self.checks),
            "rejected_attempts": [dict(a) for a in self.rejected_attempts],
        }


class ICRejectedError(Exception):
    """Raised by `EnsembleFrameSampler.sample` when the IC for a key cannot
    be accepted.

    `level` is `"coordinate"` (the frame itself fails a coordinate-level
    check; raised after the first attempt, the frame is never switched) or
    `"velocity"` (all `max_redraws + 1` velocity draws failed). `frame_id` is
    the frame that was tried. `attempts` lists every attempt as `{"attempt",
    "frame_id", "reasons"}`, and `reasons` is the flat list of all their
    reasons (not deduplicated).
    """

    def __init__(
        self,
        reasons: list[str],
        attempts: list[dict] | None = None,
        frame_id: int | None = None,
        level: str | None = None,
    ) -> None:
        self.reasons = list(reasons)
        self.attempts = [dict(a) for a in attempts] if attempts is not None else []
        self.frame_id = frame_id
        self.level = level
        n = len(self.attempts) if attempts is not None else len(self.reasons)
        super().__init__(
            f"IC rejected ({level or 'unspecified'}-level, frame_id={frame_id}) "
            f"after {n} failed attempt(s); reasons: {self.reasons}"
        )

    def __reduce__(self):
        # run_batch's worker processes re-raise this in the parent. Default
        # exception pickling would call ICRejectedError(message), which
        # parses the message string as the reasons list and garbles str().
        return (type(self), (self.reasons, self.attempts, self.frame_id, self.level))


@runtime_checkable
class Constraints(Protocol):
    """Holonomic constraints for the IC gate (see the module docstring)."""

    n_constraints: int

    def project(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def residual(self, x: np.ndarray, v: np.ndarray) -> float: ...


@dataclasses.dataclass
class PoolValidation:
    """Result of `EnsembleFrameSampler.prevalidate_pool`: coordinate-level
    checks evaluated once per frame (within the sampler's `state`)."""

    state: Hashable | None
    n_frames: int
    n_rejected: int
    rejected_weight_fraction: float
    rejected: list[dict]

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "n_frames": self.n_frames,
            "n_rejected": self.n_rejected,
            "rejected_weight_fraction": self.rejected_weight_fraction,
            "rejected": [dict(r) for r in self.rejected],
        }


class DistanceConstraints:
    """Pair-distance constraints |x_j - x_i| = d_k with a mass-weighted
    SHAKE/RATTLE-style projection (a reference `Constraints` implementation).

    `pairs`: (n_constraints, 2) atom indices; `distances`: (n_constraints,)
    target lengths; `masses`: (n_atoms,) positive masses. `project` solves
    the coupled constraints with Newton iterations along the constraint
    gradients at the input positions (sparse linear solves) until every
    relative bond-length error is at most `projection_tolerance`, then
    removes the velocity components along the constraint gradients with one
    exact mass-weighted solve. Both steps conserve total momentum and the
    centre of mass. If the position iteration does not converge within
    `max_iterations`, the returned state keeps its residual and the gate
    rejects it.
    """

    def __init__(
        self,
        pairs,
        distances,
        masses,
        projection_tolerance: float = 1e-10,
        max_iterations: int = 50,
    ) -> None:
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.size == 0:
            pairs = pairs.reshape(0, 2)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"pairs must have shape (n_constraints, 2), got {pairs.shape}")
        d = np.asarray(distances, dtype=float).reshape(-1)
        if d.shape[0] != pairs.shape[0]:
            raise ValueError("distances must have one entry per pair")
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("constraint distances must be finite and positive")
        m = np.asarray(masses, dtype=float).reshape(-1)
        if not np.all(np.isfinite(m)) or np.any(m <= 0):
            raise ValueError("masses must be finite and positive")
        n = m.shape[0]
        if pairs.size and (pairs.min() < 0 or pairs.max() >= n):
            raise ValueError("constraint atom index out of range")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("a constraint must join two distinct atoms")
        key = np.sort(pairs, axis=1)
        if len(np.unique(key, axis=0)) != len(key):
            raise ValueError("duplicate constraint pair")
        if not (math.isfinite(projection_tolerance) and projection_tolerance > 0):
            raise ValueError("projection_tolerance must be finite and positive")
        self.pairs = pairs
        self.distances = d
        self.masses = m
        self.n_atoms = n
        self.n_constraints = int(pairs.shape[0])
        self.projection_tolerance = float(projection_tolerance)
        self.max_iterations = int(max_iterations)
        self._i = pairs[:, 0]
        self._j = pairs[:, 1]
        nc = self.n_constraints
        self._rows = np.repeat(np.arange(nc), 6)
        self._cols = np.concatenate(
            [3 * self._i[:, None] + np.arange(3), 3 * self._j[:, None] + np.arange(3)], axis=1
        ).ravel()
        self._minv3 = np.repeat(1.0 / m, 3)

    @classmethod
    def from_openmm_system(cls, system, projection_tolerance: float = 1e-10, max_iterations: int = 50):
        """The constraints of an `openmm.System` (``getConstraintParameters``,
        distances in nm) with its particle masses (Da). A massless particle
        (virtual site) raises ValueError, as the IC sampler cannot handle
        one. Duck-typed: OpenMM itself is not imported here."""
        def strip(q, unit_name):
            if hasattr(q, "value_in_unit"):
                import openmm.unit as u

                return float(q.value_in_unit(getattr(u, unit_name)))
            return float(q)

        masses = np.array(
            [strip(system.getParticleMass(i), "dalton") for i in range(system.getNumParticles())]
        )
        if np.any(masses <= 0):
            bad = np.flatnonzero(masses <= 0)[:5].tolist()
            raise ValueError(
                f"System particles {bad} have zero mass (virtual sites); masses must be positive"
            )
        pairs, dists = [], []
        for k in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(k)
            pairs.append((int(i), int(j)))
            dists.append(strip(d, "nanometer"))
        return cls(pairs, dists, masses, projection_tolerance=projection_tolerance,
                   max_iterations=max_iterations)

    def _check(self, a: np.ndarray, name: str) -> np.ndarray:
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (self.n_atoms, 3):
            raise ValueError(f"{name} must have shape ({self.n_atoms}, 3), got {a.shape}")
        return a

    def _jacobian(self, x: np.ndarray):
        """Sparse (n_constraints, 3N) gradient of g_k = (|r_k|^2 - d_k^2)/2."""
        from scipy.sparse import csr_matrix

        r = x[self._j] - x[self._i]
        data = np.concatenate([-r, r], axis=1).ravel()
        return csr_matrix((data, (self._rows, self._cols)), shape=(self.n_constraints, 3 * self.n_atoms))

    def residual(self, x: np.ndarray, v: np.ndarray) -> float:
        if self.n_constraints == 0:
            return 0.0
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        r = x[self._j] - x[self._i]
        length = np.linalg.norm(r, axis=1)
        pos = np.abs(length / self.distances - 1.0)
        vr = v[self._j] - v[self._i]
        nv = np.linalg.norm(vr, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = np.where(nv > 0, np.abs(np.sum(r * vr, axis=1)) / (length * nv), 0.0)
        res = float(max(np.max(pos), np.max(vel)))
        return res if math.isfinite(res) else float("nan")

    def project(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        x = self._check(x, "x")
        v = self._check(v, "v")
        if self.n_constraints == 0:
            return x, v
        minv = diags(self._minv3)
        step = (self._jacobian(x) @ minv).T.tocsr()  # M^-1 J(x_input)^T
        d2 = self.distances**2
        for _ in range(self.max_iterations):
            r = x[self._j] - x[self._i]
            l2 = np.sum(r * r, axis=1)
            if np.max(np.abs(np.sqrt(l2) / self.distances - 1.0)) <= self.projection_tolerance:
                break
            a = (self._jacobian(x) @ step).tocsc()
            lam = np.atleast_1d(spsolve(a, 0.5 * (l2 - d2)))
            x = x - (step @ lam).reshape(-1, 3)
        jac = self._jacobian(x)
        mjt = (jac @ minv).T.tocsr()
        a = (jac @ mjt).tocsc()
        mu = np.atleast_1d(spsolve(a, jac @ v.reshape(-1)))
        v = v - (mjt @ mu).reshape(-1, 3)
        return x, v


def _mass_array(masses: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast `masses` to `shape` under the documented conventions:
    exact shape match, a scalar, or (per potential's n_atoms,) masses for
    (n_atoms, 3) coordinates (one mass per atom, applied to all 3 of that
    atom's velocity components).
    """
    masses = np.asarray(masses, dtype=float)
    if masses.shape == shape:
        return masses
    if masses.size == 1:
        return np.full(shape, float(masses.reshape(())))
    if masses.ndim == len(shape) - 1 and masses.shape == tuple(shape[:-1]):
        return masses.reshape(masses.shape + (1,) * (len(shape) - masses.ndim))
    raise ValueError(
        f"masses of shape {masses.shape} are not broadcastable to "
        f"coordinates of shape {shape} under the (n_atoms,)-for-(n_atoms,3) "
        "convention"
    )


def _box_lengths_if_orthorhombic(box: np.ndarray) -> np.ndarray | None:
    """Edge lengths of an orthorhombic (3, 3) box, or None if triclinic."""
    diag = np.diag(box).copy()
    off = box - np.diag(diag)
    if np.any(np.abs(off) > 1e-12 * float(np.max(np.abs(diag)))):
        return None
    return diag


def _min_pair_distance(x: np.ndarray, box: np.ndarray | None) -> float:
    """Smallest interatomic distance, with the minimum-image convention for
    an orthorhombic `box`. O(N log N) time, O(N) memory (KD-tree)."""
    from scipy.spatial import cKDTree

    if box is None:
        pts = x
        tree = cKDTree(pts)
    else:
        lengths = _box_lengths_if_orthorhombic(box)
        if lengths is None:
            raise ValueError(
                "min_pair_dist supports only orthorhombic periodic boxes; got a "
                "triclinic box"
            )
        pts = np.mod(x, lengths)
        pts = np.where(pts >= lengths, 0.0, pts)  # mod may round up to L
        tree = cKDTree(pts, boxsize=lengths)
    dist, _ = tree.query(pts, k=2)
    return float(np.min(dist[:, 1]))


class EnsembleFrameSampler:
    def __init__(
        self,
        pool: EnsembleFramePool,
        masses: np.ndarray,
        kT: float,
        backend: PotentialBackend,
        energy_window: tuple[float, float] | None,
        min_pair_dist: float | None,
        state: Hashable | None = None,
        max_redraws: int = 20,
        constraints: Constraints | None = None,
        remove_com_momentum: bool = True,
        *,
        constraint_tolerance: float = CONSTRAINT_TOLERANCE,
        input_constraint_tolerance: float = INPUT_CONSTRAINT_TOLERANCE,
        topology_ref: str | None = None,
        boltzmann_constant: float | None = None,
    ) -> None:
        self.pool = pool
        self.masses = np.asarray(masses, dtype=float)
        self.kT = float(kT)
        self.backend = backend
        self.energy_window = energy_window
        self.min_pair_dist = min_pair_dist
        self.state = state
        self.max_redraws = max_redraws
        self.constraints = constraints
        self.remove_com_momentum = remove_com_momentum
        self.constraint_tolerance = float(constraint_tolerance)
        self.input_constraint_tolerance = float(input_constraint_tolerance)
        self.topology_ref = topology_ref
        self.boltzmann_constant = boltzmann_constant

        self._check_parameters()
        # R17a: fail loudly and eagerly (at construction) when a configured
        # check is structurally incompatible with the pool's frames, rather
        # than silently skipping it every time validate() runs. The per-call
        # guards in _draw_velocities/validate are defensive backstops.
        self._check_frame_compatibility()

    # -- construction-time validation -----------------------------------------

    def _check_parameters(self) -> None:
        if not (math.isfinite(self.kT) and self.kT > 0):
            raise ValueError(f"kT must be finite and positive, got {self.kT}")
        if self.masses.size == 0 or not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise ValueError(
                "masses must be finite and positive (massless particles such as "
                "virtual sites are not supported by the IC sampler)"
            )
        if isinstance(self.max_redraws, bool) or not isinstance(self.max_redraws, (int, np.integer)) or self.max_redraws < 0:
            raise ValueError(f"max_redraws must be an int >= 0, got {self.max_redraws!r}")
        if self.energy_window is not None:
            lo, hi = (float(b) for b in self.energy_window)
            if math.isnan(lo) or math.isnan(hi) or lo > hi:
                raise ValueError(f"energy_window must be (lo, hi) with lo <= hi, got {self.energy_window}")
        if self.min_pair_dist is not None:
            if not (math.isfinite(self.min_pair_dist) and self.min_pair_dist > 0):
                raise ValueError(f"min_pair_dist must be finite and positive, got {self.min_pair_dist}")
        if not (math.isfinite(self.constraint_tolerance) and self.constraint_tolerance > 0):
            raise ValueError("constraint_tolerance must be finite and positive")
        if not (
            math.isfinite(self.input_constraint_tolerance)
            and self.input_constraint_tolerance >= self.constraint_tolerance
        ):
            raise ValueError(
                "input_constraint_tolerance must be finite and >= constraint_tolerance, got "
                f"{self.input_constraint_tolerance!r}"
            )
        if self.constraints is not None:
            c = self.constraints
            if not (
                callable(getattr(c, "project", None))
                and callable(getattr(c, "residual", None))
                and isinstance(getattr(c, "n_constraints", None), (int, np.integer))
                and not isinstance(getattr(c, "n_constraints", None), bool)
            ):
                raise TypeError(
                    "constraints must provide project(x, v) -> (x, v), "
                    "residual(x, v) -> float and an int n_constraints "
                    "(see cytherea.ic.sampler.Constraints); a plain callable "
                    "is no longer accepted"
                )
            if c.n_constraints < 0:
                raise ValueError("constraints.n_constraints must be >= 0")
        if self.state is not None:
            self.pool.weight_fraction([], state=self.state)  # raises if empty / zero weight

    def _check_frame_compatibility(self) -> None:
        frames = self.pool.frames
        shape0 = tuple(np.asarray(frames[0].coordinates).shape)
        refs = {f.topology_ref for f in frames}
        if len(refs) != 1:
            raise ValueError(
                f"all pool frames must share one topology_ref (one backend); got {sorted(map(str, refs))}"
            )
        if self.topology_ref is not None and self.topology_ref not in refs:
            raise ValueError(
                f"pool topology_ref {refs.pop()!r} does not match the sampler's "
                f"topology_ref {self.topology_ref!r}"
            )
        for frame in frames:
            shape = tuple(np.asarray(frame.coordinates).shape)
            if shape != shape0:
                raise ValueError(
                    f"all pool frames must have the same coordinate shape; frame_id="
                    f"{frame.frame_id} has {shape}, expected {shape0}"
                )
        is_atoms = len(shape0) == 2 and shape0[1] == 3
        _mass_array(self.masses, shape0)  # raises if not broadcastable
        if self.remove_com_momentum and not is_atoms:
            raise ValueError(
                "remove_com_momentum=True requires every pool frame to "
                f"have (n_atoms, 3) coordinates; got shape {shape0}. Pass "
                "remove_com_momentum=False for non-atomic coordinate spaces."
            )
        if self.min_pair_dist is not None and not (is_atoms and shape0[0] >= 2):
            raise ValueError(
                f"min_pair_dist is set but the pool frames have shape {shape0}, "
                "which is not (n_atoms>=2, 3); min_pair_dist only applies to "
                "atomic coordinate spaces with at least 2 atoms."
            )
        if self.min_pair_dist is not None:
            for frame in frames:
                if frame.box is not None:
                    self._check_box_for_min_pair_dist(np.asarray(frame.box, dtype=float), frame.frame_id)
        if self.boltzmann_constant is not None:
            kB = float(self.boltzmann_constant)
            for frame in frames:
                kT_frame = kB * float(frame.temperature)
                if not abs(kT_frame - self.kT) <= _FRAME_TEMPERATURE_RTOL * self.kT:
                    raise ValueError(
                        f"frame_id={frame.frame_id}: boltzmann_constant * temperature "
                        f"= {kT_frame} does not match kT = {self.kT} (units mix-up, or "
                        "a frame from a different temperature)"
                    )

    def _check_box_for_min_pair_dist(self, box: np.ndarray, frame_id: Any = None) -> np.ndarray | None:
        if box.shape != (3, 3) or not np.all(np.isfinite(box)):
            return None  # an invalid box is rejected by validate ("invalid_box")
        lengths = _box_lengths_if_orthorhombic(box)
        where = "" if frame_id is None else f"frame_id={frame_id}: "
        if lengths is None:
            raise ValueError(
                f"{where}min_pair_dist supports only orthorhombic periodic boxes; "
                "got a triclinic box"
            )
        if self.min_pair_dist >= 0.5 * float(np.min(np.abs(lengths))):
            raise ValueError(
                f"{where}min_pair_dist={self.min_pair_dist} must be smaller than half "
                f"the shortest box edge ({float(np.min(lengths))})"
            )
        return lengths

    # -- velocity draw ----------------------------------------------------

    def _draw_velocities(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        """v_i ~ N(0, kT/m_i) per coordinate, optionally with center-of-mass
        momentum removed. COM removal is only meaningful for (n_atoms, 3)
        coordinates -- for anything else it would silently zero out an
        arbitrary component of a non-atomic coordinate vector, so it is
        rejected with a clear error instead. (`EnsembleFrameSampler.__init__`
        already rejects this configuration eagerly against the pool's frame
        shapes; this is a defensive backstop for direct calls with some
        other shape.)
        """
        is_atoms = len(shape) == 2 and shape[1] == 3
        if self.remove_com_momentum and not is_atoms:
            raise ValueError(
                "remove_com_momentum=True requires (n_atoms, 3) coordinates "
                f"(one mass per atom); got coordinate shape {shape}. Pass "
                "remove_com_momentum=False for non-atomic coordinate spaces."
            )
        m = _mass_array(self.masses, shape)
        sigma = np.sqrt(self.kT / m)
        v = rng.normal(0.0, 1.0, size=shape) * sigma
        if self.remove_com_momentum:
            per_atom_mass = self._per_atom_mass_for_com(m, shape)
            total_p = np.sum(m * v, axis=0)  # shape (3,)
            total_mass = float(np.sum(per_atom_mass))
            v = v - total_p / total_mass
        return v

    @staticmethod
    def _per_atom_mass_for_com(m: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Reduce a broadcast mass array `m` (already shaped per `shape`) to
        one scalar mass per atom, for use in center-of-mass removal.

        `_mass_array` returns either an (n_atoms, 1) array (the documented
        (n_atoms,)-masses-for-(n_atoms,3)-coordinates convention -- always
        isotropic by construction) or, if the caller passed a full
        (n_atoms, 3) mass array explicitly, that array unchanged. The latter
        only has a well-defined "COM removal" if each atom's mass is the
        same across its x/y/z rows (R18 #3) -- an anisotropic per-atom mass
        would make "the atom's momentum" ambiguous (removing it component-
        wise, as the previous implementation silently did by using only the
        x-column's mass for every component, is *wrong* physics, not a
        matter of convention), so that case raises instead.
        """
        if m.ndim == 2 and m.shape == shape and shape[-1] == 3:
            if not (np.allclose(m[:, 0], m[:, 1]) and np.allclose(m[:, 0], m[:, 2])):
                raise ValueError(
                    "remove_com_momentum=True requires an isotropic mass "
                    "per atom (the same mass for x/y/z); got a full "
                    "(n_atoms, 3) mass array with anisotropic rows"
                )
            return m[:, 0]
        if m.ndim == 2:
            return m[:, 0]
        return m

    # -- frame selection ----------------------------------------------------

    def _select_frame(self, key: Key) -> EnsembleFrame:
        fid = getattr(key, "frame_id", None)
        if fid is None or isinstance(fid, bool) or not isinstance(fid, (int, np.integer)):
            raise TypeError(
                "EnsembleFrameSampler.sample needs a key with an int frame_id "
                f"(a ShotKey); got {key!r}"
            )
        fid = int(fid)
        if fid == -1:
            return self.pool.choose(derive_rng(key, "ic/frame"), state=self.state)
        if fid < -1:
            raise ValueError(f"key.frame_id must be >= 0 or -1 (weighted draw), got {fid}")
        frame = self.pool.get(fid)
        if self.state is not None and self.pool.state_label(frame) != self.state:
            raise ValueError(
                f"key.frame_id={fid} is in state {self.pool.state_label(frame)!r}, "
                f"not the sampler's state {self.state!r}"
            )
        return frame

    # -- sampling with redraw ---------------------------------------------

    def sample(self, key: Key) -> tuple[InitialState, ValidityReport]:
        frame = self._select_frame(key)
        fid = int(frame.frame_id)
        x0 = np.array(frame.coordinates, dtype=float, copy=True)
        box = None if frame.box is None else np.array(frame.box, dtype=float, copy=True)
        finite_x0 = bool(np.all(np.isfinite(x0)))

        input_residual = None
        if self.constraints is not None and finite_x0:
            input_residual = float(self.constraints.residual(x0, np.zeros_like(x0)))
            if not (input_residual <= self.input_constraint_tolerance):
                # N-I1: never "repair" a frame that is far off the manifold
                attempts0 = [{"attempt": 0, "frame_id": fid, "reasons": ["constraint_input"],
                              "constraint_residual_input": input_residual}]
                raise ICRejectedError(
                    ["constraint_input"], attempts=attempts0, frame_id=fid, level="coordinate"
                )

        attempts: list[dict] = []
        all_reasons: list[str] = []
        for k in range(self.max_redraws + 1):
            rng = derive_rng(key, f"ic/velocities/{k}")
            v = self._draw_velocities(rng, x0.shape)
            x = x0.copy()
            if self.constraints is not None and finite_x0 and np.all(np.isfinite(v)):
                x, v = self.constraints.project(x0, v)
                x = np.asarray(x, dtype=float)
                v = np.asarray(v, dtype=float)
                if x.shape != x0.shape or v.shape != x0.shape:
                    raise ValueError("constraints.project returned arrays of the wrong shape")

            meta = {
                "frame_id": fid,
                "frame_time": float(frame.time),
                "frame_weight": float(frame.weight),
                "source_id": frame.source_id,
                "topology_ref": frame.topology_ref,
                "state": self.pool.state_label(frame),
                "n_redraws": k,
            }
            istate = InitialState(
                state=MDState(x=x, v=v, t=0.0, box=None if box is None else box.copy()),
                frame_id=fid,
                meta=meta,
            )
            report = self.validate(istate)
            report.checks["frame_temperature"] = float(frame.temperature)
            if input_residual is not None:
                report.checks["constraint_residual_input"] = input_residual
            if report.ok:
                report.n_redraws = k
                report.rejected_attempts = attempts
                return istate, report

            attempts.append({"attempt": k, "frame_id": fid, "reasons": list(report.reasons)})
            all_reasons.extend(report.reasons)
            if any(r in COORDINATE_REASONS for r in report.reasons):
                raise ICRejectedError(all_reasons, attempts=attempts, frame_id=fid, level="coordinate")

        raise ICRejectedError(all_reasons, attempts=attempts, frame_id=fid, level="velocity")

    # -- pool pre-validation --------------------------------------------------

    def prevalidate_pool(self) -> PoolValidation:
        """Evaluate the coordinate-level checks once per frame (within the
        sampler's `state`), at zero velocity after constraint projection.
        Returns the frames that can never pass and the fraction of pool
        weight they carry; record it in the run provenance."""
        if self.state is None:
            frames = list(self.pool.frames)
        else:
            frames = [f for f in self.pool.frames if self.pool.state_label(f) == self.state]
        rejected: list[dict] = []
        for frame in frames:
            x = np.array(frame.coordinates, dtype=float, copy=True)
            v = np.zeros_like(x)
            if self.constraints is not None and np.all(np.isfinite(x)):
                res_in = float(self.constraints.residual(x, v))
                if not (res_in <= self.input_constraint_tolerance):
                    rejected.append({"frame_id": int(frame.frame_id), "weight": float(frame.weight),
                                     "reasons": ["constraint_input"]})
                    continue
                x, v = self.constraints.project(x, v)
            box = None if frame.box is None else np.array(frame.box, dtype=float, copy=True)
            rep = self.validate(
                InitialState(state=MDState(x=np.asarray(x, float), v=np.asarray(v, float), t=0.0, box=box),
                             frame_id=int(frame.frame_id), meta={})
            )
            coord = [r for r in rep.reasons if r in COORDINATE_REASONS]
            if coord:
                rejected.append({"frame_id": int(frame.frame_id), "weight": float(frame.weight), "reasons": coord})
        fraction = self.pool.weight_fraction([r["frame_id"] for r in rejected], state=self.state)
        return PoolValidation(
            state=self.state,
            n_frames=len(frames),
            n_rejected=len(rejected),
            rejected_weight_fraction=fraction,
            rejected=rejected,
        )

    # -- validity gate ------------------------------------------------------

    def validate(self, s: InitialState) -> ValidityReport:
        x = np.asarray(s.state.x, dtype=float)
        v = np.asarray(s.state.v, dtype=float)
        if v.shape != x.shape:
            raise ValueError(f"velocity shape {v.shape} does not match coordinate shape {x.shape}")
        reasons: list[str] = []
        checks: dict[str, float | str] = {}

        finite_x = bool(np.all(np.isfinite(x)))
        finite_v = bool(np.all(np.isfinite(v)))
        checks["nonfinite_x"] = 0.0 if finite_x else 1.0
        checks["nonfinite_v"] = 0.0 if finite_v else 1.0
        if not finite_x:
            reasons.append("nonfinite_x")
        if not finite_v:
            reasons.append("nonfinite_v")

        box = s.state.box
        box_ok = True
        if box is not None:
            b = np.asarray(box, dtype=float)
            vol = float(np.linalg.det(b)) if b.shape == (3, 3) and np.all(np.isfinite(b)) else float("nan")
            checks["box_volume"] = vol
            if not (math.isfinite(vol) and vol > 0):
                reasons.append("invalid_box")
                box_ok = False
            box = b
        coords_usable = finite_x and box_ok

        # Energy and forces: always evaluated once (module docstring, I1).
        energy = None
        if coords_usable:
            # R20: evaluate at the InitialState's own box, never the
            # backend's default cell.
            try:
                energy = self.backend.energy_forces(x, box=box)
            except Exception as exc:  # p2 m6: deterministic for this frame
                checks["energy_error"] = f"{type(exc).__name__}: {exc}"[:500]
                checks["max_abs_force"] = float("nan")
                reasons.append("energy_error")
        if energy is not None:
            E = float(energy[0])
            F = np.asarray(energy[1], dtype=float)
            checks["energy"] = E
            if self.energy_window is not None:
                checks["energy_window"] = E
            finite_F = bool(F.size == x.size and np.all(np.isfinite(F)))
            checks["max_abs_force"] = float(np.max(np.abs(F))) if finite_F and F.size else float("nan")
            if not math.isfinite(E) or not finite_F:
                reasons.append("nonfinite_energy")
            elif self.energy_window is not None:
                lo, hi = self.energy_window
                if not (lo <= E <= hi):
                    reasons.append("energy_window")
        else:
            # nonfinite_x / invalid_box / energy_error already fail the gate;
            # do not claim the energy was checked when it could not be.
            checks["energy"] = float("nan")
            if self.energy_window is not None:
                checks["energy_window"] = float("nan")

        if self.min_pair_dist is not None:
            if coords_usable and x.ndim == 2 and x.shape[1] == 3 and x.shape[0] >= 2:
                if box is not None:
                    self._check_box_for_min_pair_dist(box)
                min_d = _min_pair_distance(x, box)
                checks["min_pair_dist"] = min_d
                if not (min_d >= self.min_pair_dist):
                    reasons.append("min_pair_dist")
            # else: not applicable (not (n_atoms, 3), fewer than 2 atoms,
            # or coordinates already rejected).

        n_constraints = 0
        if self.constraints is not None:
            n_constraints = int(self.constraints.n_constraints)
            if finite_x and finite_v:
                residual = float(self.constraints.residual(x, v))
                checks["constraint_residual"] = residual
                if not (residual <= self.constraint_tolerance):
                    reasons.append("constraint_residual")
            else:
                checks["constraint_residual"] = float("nan")
        checks["n_constraints"] = float(n_constraints)

        # R17b: the COM part of the dof comes from the *actual* v (total
        # momentum numerically zero), not from the remove_com_momentum flag.
        # Holonomic constraints each remove one more dof.
        is_atoms = x.ndim == 2 and x.shape[-1] == 3
        dof = v.size - n_constraints
        com_removed = False
        if is_atoms and finite_v:
            m_com = _mass_array(self.masses, x.shape)
            total_p = np.sum(m_com * v, axis=0)  # shape (3,)
            norm_p = float(np.linalg.norm(total_p))
            denom = float(np.sum(m_com * np.abs(v))) + 1e-300
            com_removed = norm_p <= 1e-10 * denom
        if com_removed:
            dof -= 3
        checks["temperature_dof"] = float(dof)

        if finite_v and dof >= _MIN_DOF_FOR_TEMPERATURE_CHECK:
            m = _mass_array(self.masses, x.shape)
            KE = 0.5 * float(np.sum(m * v**2))
            # instantaneous "temperature" in kT units: KE = (dof/2)*kT_inst
            T_inst = 2.0 * KE / dof
            # sigma of T_inst: KE ~ Gamma(dof/2, kT) exactly under the null
            # (MB) hypothesis, so Var(KE) = (dof/2)*kT**2 and
            # T_inst = 2*KE/dof -> Var(T_inst) = 2*kT**2/dof
            sigma_T = self.kT * math.sqrt(2.0 / dof)
            checks["temperature"] = T_inst
            if not (abs(T_inst - self.kT) <= _TEMPERATURE_SIGMA_MULTIPLE * sigma_T):
                reasons.append("temperature")
        elif finite_v:
            checks["temperature"] = "skipped_low_dof"
        else:
            checks["temperature"] = float("nan")

        assert set(reasons) <= REASONS, reasons
        ok = len(reasons) == 0
        return ValidityReport(ok=ok, reasons=reasons, checks=checks)
