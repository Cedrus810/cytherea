"""PES consistency suite: the checks every `PotentialBackend` (analytic,
OpenMM, OpenMM+ML, ...) must pass before shooting or committor work trusts it
(design §3.5, spec change S2 of 2026-10-01).

Two modes
=========

``mode="strict"`` is for analytic backends and the OpenMM Reference/double
platform. It runs a central finite difference on *every* coordinate of every
probe, and its tolerance defaults come from the ``"double"`` row of
`DEFAULT_TOLERANCES`.

``mode="sampled"`` is for production platforms (mixed/single precision) and
protein-scale systems.
- The finite difference covers, per probe (fixreview-p4 I-1):
  ``n_fd_atoms`` atoms drawn uniformly without replacement (default
  `DEFAULT_N_FD_ATOMS`); every atom of ``fd_atoms_include``; ``n_fd_atoms``
  atoms drawn from each group of ``fd_atom_groups`` (the whole group if it
  is smaller) -- pass the solute, an ML/QM region or any other localised
  term here, since a uniform draw over a solvated box almost never touches
  them; and the `_N_TOP_FORCE_ATOMS` atoms with the largest force. The
  Generator comes from ``derive_rng(_PES_SUITE_KEY, "fd_atoms")`` for
  ``fd_seed=0`` (else ``"fd_atoms/<seed>"``), so the subset is
  deterministic; vary ``fd_seed`` to check other atoms. ``PESReport.fd_atoms``
  lists the atoms checked per probe.
- ``fd_step`` defaults to ``1e-4`` (strict: ``1e-5``): in mixed precision at
  1e4-1e5 atoms the roundoff of a 1e-5 nm difference is large enough to
  hide a 10 % force defect (fixreview-p4 I-2, measured on a 34k-atom water
  box), and 1e-4 was better at every size measured.
- Tolerance defaults come from the `DEFAULT_TOLERANCES` row for the
  backend's precision. The precision is resolved in this order:
  1. the ``precision`` argument;
  2. ``backend.effective_config()``, where platform ``"Reference"`` means
     ``"double"``, otherwise its ``"precision"`` entry is used;
  3. ``"double"``, the strictest row (fail-loud, never silently lenient).

An "atom" is a row of ``x`` when ``x.ndim >= 2`` (for example an
``(n_atoms, 3)`` molecule), and a single coordinate when ``x`` is 1-D (the
analytic toys).

Default tolerances (`DEFAULT_TOLERANCES`)
=========================================

========  =========  =========  =========  ===========
row       fd_rtol    inv_rtol   nve_rtol   repeat_rtol
========  =========  =========  =========  ===========
double    1e-4       1e-10      1e-3       0 (bitwise)
mixed     5e-3       1e-4       1e-2       1e-3
single    2e-2       1e-3       1e-2       5e-3
========  =========  =========  =========  ===========

Any tolerance passed explicitly (``fd_rtol``, ``inv_rtol``, ``nve_rtol``,
``repeat_rtol``) overrides its row entry. The mixed ``repeat_rtol`` comes
from the A1 system on CUDA mixed with DeterministicForces (2026-10-04): the
first evaluation at a position differs from later ones by up to 3e-4 in
the repeat metric (atom reordering changes the summation order); double
precision repeats to 7e-10.

Rules shared by both modes
==========================

- **Any NaN/inf fails.** Every energy and force the suite evaluates
  (probes, FD displacements, transformed probes, repeat evaluations, NVE
  states) and every propagated x/v must be finite. Otherwise
  ``all_finite=False`` and ``passed=False``, with a reason in ``reasons``.
  All aggregates are nan-propagating (`np.max` over arrays, never Python
  ``max()``, which silently skips a NaN that isn't first).
- **Normalisations are independent of the energy zero.**
  - Force scale ``F_s``: the suite-wide max |force component| over all
    probes. It is suite-wide, not per probe, so a minimised structure among
    other probes does not blow up its own ratio.
  - Force errors are divided by ``F_s``.
  - An energy difference caused by a displacement ``d`` is divided by
    ``F_s * ||d||_1``, the first-order bound on ``|dE|``.
  - The NVE drift is divided by the kinetic-energy scale ``n_dof * kT / 2``.
  - Adding a constant to the energy changes none of these numbers.
- **The NVE check starts from a thermally perturbed state:**
  Maxwell-Boltzmann velocities at ``kT`` from
  ``derive_rng(_PES_SUITE_KEY, "nve_v")``. It never starts at rest, where
  any propagator trivially "conserves" energy at a minimum.

Checks
======

Finite differences (FD)
-----------------------

The FD force is ``F_fd(h) = -(E(x+h) - E(x-h)) / 2h``. It uses only the
backend's *energy*, so it is an independent check on the reported force
``F``. The check also evaluates ``F_fd(h/2)``. The per-probe FD resolution
``N_p = max |F_fd(h) - F_fd(h/2)|`` measures the truncation and roundoff
noise of the difference itself.

Each coordinate's *excess* error is ``max(|F_fd(h) - F| - A_c, 0)``, the
error that FD noise cannot explain, with the per-coordinate allowance
(fixreview-p4 I-2: the earlier ``3 * N_p``, the probe-wide maximum, let a
defect of up to ~4 x fd_rtol pass and neutralised the per-atom metric)

    A_c = max(_FD_TRUNCATION_FACTOR * n_c, _FD_NOISE_FACTOR * median_p(n))

where ``n_c = |F_fd(h) - F_fd(h/2)|`` for that coordinate (its own
truncation error is ~(4/3) n_c) and ``median_p(n)`` is the median of n_c
over the probe's checked coordinates (the roundoff level, which a single
coordinate's n_c can under-estimate by chance). The following are reported:

- ``fd_max_rel_err = max excess / F_s``
- ``fd_max_atom_rel_err = max excess / max(|F_atom|, F_rms)``: a per-atom
  metric, so a defect confined to low-force atoms is not masked by the
  global max (``F_rms`` is the suite-wide RMS force component).
- ``fd_raw_rel_err = max |F_fd(h) - F| / F_s``
- ``fd_noise_rel = max N_p / F_s``
- ``fd_allowance_rel = max A_c / F_s`` and ``fd_noise_floor_rel = max
  _FD_NOISE_FACTOR * median_p(n) / F_s``

The FD check passes when ``fd_max_rel_err <= fd_rtol``,
``fd_max_atom_rel_err <= fd_rtol`` and ``fd_noise_floor_rel <= fd_rtol``.
The last bounds the effective tolerance of every coordinate whose own FD
resolution is typical by 2 x fd_rtol (a larger defect there always fails);
a coordinate with an unusually large ``n_c`` is allowed its own
``1.5 n_c`` on top. If it fails (reason ``fd_unresolved``), the FD noise is
too large for the check to have power at ``fd_rtol``. Typical causes are a probe set where
every probe is stationary (no force signal), or an ``fd_step`` that is too
small for the precision.

Cutoff crossings
----------------

A plain (unshifted) cutoff makes the energy jump where a pair crosses r_c
(OpenMM PME: the erfc direct-space term and LJ without a switching
function), so a finite difference whose stencil moves a pair across r_c
measures that jump, not the force. A backend that declares its cutoffs
through an optional ``energy_cutoffs()`` -> ``{"cutoffs_nm": [...],
"box_lengths_nm": [Lx, Ly, Lz] or None}`` (orthorhombic, the box
``energy_forces(x)`` uses without an explicit box) gets every FD coordinate
skipped whose stencil (+-fd_step) can move some pair across a cutoff:
``|r - r_c| <= h |d_c| / r + h^2 / (r - h)`` with the minimum-image pair
vector d (a bound on the change of r for a displacement h along c). Skipped
coordinates enter neither the error nor the noise floor; their number is
``PESReport.fd_skipped_cutoff`` and the cutoffs ``fd_cutoffs``. If every
coordinate is skipped the check has no power (``fd_unresolved``). The guard
needs ``(n_atoms, 3)`` probes. On the A1 system (2026-10-04) 27 of 186
coordinates crossed; the others agreed to 2.6e-6 (double) / 3.8e-4 (mixed)
of the force scale, the crossing ones were off by 28-56 kJ/mol/nm.

Repeatability
-------------

Each probe is re-evaluated after all the FD and invariance evaluations at
other positions, so the check can see history-dependent nondeterminism
(neighbour lists, caches). The re-evaluation is compared against the
probe's first evaluation.
- ``repeat_rtol=0.0`` requires bitwise equality; the default is the
  precision row's ``repeat_rtol`` (bitwise for double).
- Otherwise the check requires ``repeat_max_rel_err <= repeat_rtol``, where
  ``repeat_max_rel_err = max(|dE| / (F_s * fd_step), max|dF| / F_s)``. That
  is the relative force error the nondeterminism would inject into the FD
  check.

Translation/rotation invariance (optional)
------------------------------------------

This requires x of shape ``(n_atoms, 3)`` and a backend without PBC. It
applies a fixed translation and a fixed proper rotation. It compares ``E``
by ``|dE| / (F_s * ||d||_1)`` and the (rotated) force by ``max|dF| / F_s``,
against ``inv_rtol``.

Short NVE run (optional, ``nve_steps > 0``)
--------------------------------------------

1. Builds a propagator with ``build(state0, nve_cfg, _PES_SUITE_KEY)``.
   Pass an NVE ``PhysicsConfig`` as ``nve_cfg`` for backends that honour
   one. The analytic backend ignores ``cfg``, so it must be constructed as
   BAOAB with gamma=0.
2. ``state0`` is ``probes[0]`` with Maxwell-Boltzmann velocities at ``kT``,
   or the ``nve_state`` supplied by the caller.
3. Takes ``E_tot(0)`` from ``propagator.get_state()``, so it reflects any
   constraint projection done at build.
4. Samples ``E_tot = PE(x; box) + 1/2 sum(m v^2)`` over up to 20 chunks.
5. Reports ``nve_rel_drift = max_t |E_tot(t) - E_tot(0)| / (n_dof*kT/2)``,
   the drift as a fraction of the thermal kinetic energy.

- ``nve_abs_drift_kT = max_t |E_tot(t) - E_tot(0)| / kT`` is reported too:
  the relative drift dilutes a localised energy injection as n_dof grows.
- A thermostatted or dissipative propagator fails this for small and
  medium systems, but not reliably at protein scale (fixreview-p4 m1): a
  thermostat's energy random walk grows like sqrt(n_dof) while the
  normalisation grows like n_dof (BAOAB gamma t = 0.5: drift 0.30 at 100
  dof, 1.9e-2 at 1e4, 5.9e-3 -- a pass of the mixed 1e-2 -- at 1e5). Use an
  NVE ``nve_cfg`` (Verlet, no friction); this check cannot prove that.
- If the energy is quantised below ``fd_step`` (E(x+h) bitwise equal to
  E(x-h)), the FD noise reads 0 and a resulting failure is reported as
  ``fd`` rather than ``fd_unresolved`` (fixreview-p4 m4; never a false
  pass). The invariance energy bounds grow with N and with the distance
  from the origin; at large N the force side of that check carries it.
- ``kT`` is required unless ``nve_state`` is given. In that case ``kT``
  defaults to ``2*KE(nve_state)/n_dof``, the state's instantaneous kinetic
  temperature.
- For ``(n_atoms, 3)`` probes the mass-weighted COM velocity is removed
  from the draw by default (``remove_com``), as OpenMM Systems usually carry
  a CMMotionRemover.
- ``n_dof`` defaults to the number of coordinates with nonzero mass, minus 3
  when the COM velocity was removed. Pass it explicitly to also subtract
  constraints.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Literal

import numpy as np

from cytherea.backends.base import MDState, PhysicsConfig, PotentialBackend
from cytherea.keys import ShotKey, derive_rng

# Fixed key for every random draw the suite makes (the FD atom subset, the NVE
# Maxwell-Boltzmann velocities, and the NVE propagator's own stream). Each use
# has its own substream label.
_PES_SUITE_KEY = ShotKey(
    global_seed=0, frame_id=0, shot_id=0, stage="pes_consistency_suite"
)

# Multiple of the probe's median FD resolution median|F_fd(h) - F_fd(h/2)|
# that is the roundoff floor of the allowance (fixreview-p4 I-2). With
# roundoff of sd s in F_fd(h) (2s in F_fd(h/2)), |F_fd(h) - F_fd(h/2)| has
# median ~1.5 s, so 2 x median ~ 3 s covers the largest roundoff error of a
# few hundred checked coordinates.
_FD_NOISE_FACTOR = 2.0
# Multiple of a coordinate's own |F_fd(h) - F_fd(h/2)| allowed for its
# truncation error (pure central-difference truncation: 4/3).
_FD_TRUNCATION_FACTOR = 1.5
# Sampled mode always checks the atoms with the largest forces too.
_N_TOP_FORCE_ATOMS = 4

DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "double": {"fd_rtol": 1e-4, "inv_rtol": 1e-10, "nve_rtol": 1e-3, "repeat_rtol": 0.0},
    "mixed": {"fd_rtol": 5e-3, "inv_rtol": 1e-4, "nve_rtol": 1e-2, "repeat_rtol": 1e-3},
    "single": {"fd_rtol": 2e-2, "inv_rtol": 1e-3, "nve_rtol": 1e-2, "repeat_rtol": 5e-3},
}
DEFAULT_N_FD_ATOMS = 16

# Fixed translation vector and rotation matrix for the invariance checks. They
# describe which geometric transform to apply, not a sampled quantity, so they
# are plain constants.
_TRANSLATION = np.array([1.3, -0.7, 2.1])


def _fixed_rotation_matrix() -> np.ndarray:
    """A fixed proper rotation (det +1, not the identity), built with
    Rodrigues' formula about the (1,1,1) axis."""
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    theta = 0.7
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclasses.dataclass
class PESReport:
    """Result of `pes_consistency_suite`. ``passed`` is True iff ``reasons``
    is empty. Each failed gate adds one human-readable reason whose leading
    token is one of ``nonfinite``, ``fd``, ``fd_atom``, ``fd_unresolved``,
    ``repeat``, ``invariance_translation``, ``invariance_rotation``, ``nve``
    or ``nve_nonfinite``.
    """

    passed: bool
    fd_max_rel_err: float
    translation_err: float | None
    rotation_err: float | None
    repeat_bitwise: bool
    nve_rel_drift: float | None
    mode: str = "strict"
    precision: str = "double"
    all_finite: bool = True
    reasons: list[str] = dataclasses.field(default_factory=list)
    force_scale: float = float("nan")
    fd_raw_rel_err: float = float("nan")
    fd_noise_rel: float = float("nan")
    fd_max_atom_rel_err: float = float("nan")
    n_fd_coords: int = 0
    repeat_max_rel_err: float = float("nan")
    tolerances: dict = dataclasses.field(default_factory=dict)
    kT: float | None = None
    n_dof: int | None = None
    fd_allowance_rel: float = float("nan")
    fd_noise_floor_rel: float = float("nan")
    fd_atoms: list[list[int]] = dataclasses.field(default_factory=list)
    fd_step: float = float("nan")
    nve_abs_drift_kT: float | None = None
    fd_skipped_cutoff: int = 0
    fd_cutoffs: list[float] = dataclasses.field(default_factory=list)


def _ratio(num: float, den: float) -> float:
    """num/den that propagates NaN, and maps 0/0 to 0 and x/0 (x>0) to inf."""
    num, den = float(num), float(den)
    if np.isnan(num) or np.isnan(den):
        return float("nan")
    if den > 0.0:
        return num / den
    return 0.0 if num == 0.0 else float("inf")


def _nanmax(values) -> float:
    """Max that propagates NaN (unlike Python's max()); empty -> 0.0."""
    arr = np.asarray(list(values), dtype=float)
    return float(np.max(arr)) if arr.size else 0.0


class _Evaluator:
    """Wraps ``backend.energy_forces``: checks the force shape, and records
    the first non-finite result it sees (with a context label)."""

    def __init__(self, backend: PotentialBackend) -> None:
        self._backend = backend
        self.n_calls = 0
        self.nonfinite: str | None = None

    def __call__(
        self, x: np.ndarray, context: str, box: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        self.n_calls += 1
        if box is None:
            E, F = self._backend.energy_forces(x)
        else:
            E, F = self._backend.energy_forces(x, box=box)
        E = float(E)
        F = np.asarray(F, dtype=float)
        if F.shape != np.shape(x):
            raise ValueError(
                f"backend.energy_forces returned forces of shape {F.shape} "
                f"for x of shape {np.shape(x)} ({context})"
            )
        if self.nonfinite is None and not (
            np.isfinite(E) and np.all(np.isfinite(F))
        ):
            self.nonfinite = f"non-finite energy/force at {context}"
        return E, F


def _atom_coords(shape: tuple[int, ...], atom: int) -> list[tuple[int, ...]]:
    """Coordinate indices belonging to one "atom" (a row when ndim >= 2, a
    single coordinate for 1-D x)."""
    if len(shape) >= 2:
        return [(atom, *rest) for rest in np.ndindex(shape[1:])]
    return [(atom,)]


def _n_atoms(shape: tuple[int, ...]) -> int:
    return shape[0] if len(shape) >= 2 else int(np.prod(shape, dtype=int))


def _atom_index_array(values, n_atoms: int, name: str) -> np.ndarray:
    a = np.unique(np.asarray(list(values), dtype=np.int64))
    if a.size and (a.min() < 0 or a.max() >= n_atoms):
        raise ValueError(f"{name}: atom index out of range [0, {n_atoms})")
    return a


def _cutoff_guard(backend: PotentialBackend, shape: tuple[int, ...]):
    """(cutoffs, box lengths or None) declared by ``backend.energy_cutoffs()``,
    or ``([], None)`` without the hook (module docstring, "Cutoff crossings")."""
    hook = getattr(backend, "energy_cutoffs", None)
    decl = hook() if callable(hook) else None
    if not decl or not decl.get("cutoffs_nm"):
        return [], None
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"the cutoff-crossing guard needs probes of shape (n_atoms, 3), got {shape}")
    cutoffs = sorted({float(c) for c in decl["cutoffs_nm"]})
    if not all(np.isfinite(c) and c > 0 for c in cutoffs):
        raise ValueError(f"energy_cutoffs(): cutoffs must be finite and > 0, got {cutoffs}")
    L = decl.get("box_lengths_nm")
    if L is not None:
        L = np.asarray(L, dtype=float)
        if L.shape != (3,) or not np.all(np.isfinite(L) & (L > 0)):
            raise ValueError(f"energy_cutoffs(): box_lengths_nm must be 3 positive lengths, got {L}")
    return cutoffs, L


def _crosses_cutoff(x: np.ndarray, atom: int, comp: int, h: float, cutoffs, L) -> bool:
    """Can moving `atom` by up to +-h along `comp` move some pair across a cutoff?"""
    d = np.delete(x - x[atom], atom, axis=0)
    if L is not None:
        d -= np.round(d / L) * L
    r = np.linalg.norm(d, axis=1)
    slack = h * np.abs(d[:, comp]) / np.maximum(r, 1e-300) + h * h / np.maximum(r - h, 1e-300)
    return any(bool(np.any(np.abs(r - rc) <= slack)) for rc in cutoffs)


def _fd_component(
    ev: _Evaluator, x: np.ndarray, idx: tuple[int, ...], h: float, label: str
) -> float:
    xp = x.copy()
    xp[idx] += h
    xm = x.copy()
    xm[idx] -= h
    Ep, _ = ev(xp, label)
    Em, _ = ev(xm, label)
    return -(Ep - Em) / (2.0 * h)


def _broadcast_masses(masses, shape: tuple[int, ...]) -> np.ndarray:
    """Validate ``masses`` against the coordinate shape and broadcast it to
    that shape.

    Accepted: a scalar; an array of exactly ``shape``; or ``(n_atoms, 1)``
    when ``len(shape) == 2``. A 1-D ``(n_atoms,)`` array with 2-D x is
    rejected. It is ambiguous: for 3 atoms, numpy would silently apply atom
    j's mass to Cartesian component j of every atom. Pass ``masses[:, None]``
    instead.
    """
    shape = tuple(shape)
    m = np.asarray(masses, dtype=float)
    ok = m.shape == () or m.shape == shape
    if len(shape) == 2 and m.shape == (shape[0], 1):
        ok = True
    if not ok:
        allowed = f"a scalar or shape {shape}"
        if len(shape) == 2:
            allowed += f" or ({shape[0]}, 1) -- for per-atom masses pass masses[:, None]"
        raise ValueError(
            f"masses of shape {m.shape} is ambiguous/incompatible for "
            f"coordinates of shape {shape}; use {allowed}"
        )
    if not np.all(np.isfinite(m)) or np.any(m < 0.0) or not np.any(m > 0.0):
        raise ValueError("masses must be finite, >= 0, and not all zero")
    return np.broadcast_to(m, shape).astype(float)


def _resolve_precision(
    backend: PotentialBackend, precision: str | None, mode: str
) -> str:
    if precision is not None:
        if precision not in DEFAULT_TOLERANCES:
            raise ValueError(
                f"precision must be one of {sorted(DEFAULT_TOLERANCES)}, "
                f"got {precision!r}"
            )
        if mode == "strict" and precision != "double":
            raise ValueError(
                "mode='strict' is for double precision (analytic, OpenMM "
                f"Reference); got precision={precision!r} -- use "
                "mode='sampled' for mixed/single-precision platforms"
            )
        return precision
    eff = getattr(backend, "effective_config", None)
    if callable(eff):
        try:
            cfg = eff()
        except Exception:  # noqa: BLE001 -- a broken hook just falls back
            cfg = None
        if isinstance(cfg, dict):
            if cfg.get("platform") == "Reference":
                return "double"
            if cfg.get("precision") in DEFAULT_TOLERANCES:
                return str(cfg["precision"])
    return "double"


def pes_consistency_suite(
    backend: PotentialBackend,
    probes: list[np.ndarray],
    *,
    mode: Literal["strict", "sampled"] = "strict",
    fd_step: float | None = None,
    n_fd_atoms: int | None = None,
    fd_atoms_include=(),
    fd_atom_groups=None,
    fd_seed: int = 0,
    fd_rtol: float | None = None,
    precision: Literal["double", "mixed", "single"] | None = None,
    check_invariance: bool = False,
    inv_rtol: float | None = None,
    repeat_rtol: float | None = None,
    nve_steps: int = 0,
    masses: np.ndarray | float | None = None,
    kT: float | None = None,
    nve_rtol: float | None = None,
    nve_cfg: PhysicsConfig | None = None,
    nve_state: MDState | None = None,
    n_dof: int | None = None,
    remove_com: bool | None = None,
    rtol: float | None = None,
) -> PESReport:
    """Run the PES consistency suite. See the module docstring for the full
    definition of every check, the default tolerances, and the
    normalisations.

    Keyword arguments (all keyword-only):
    - ``mode``: ``"strict"`` (full FD, double-precision tolerances) or
      ``"sampled"`` (FD on ``n_fd_atoms`` random atoms per probe,
      precision-scaled tolerances).
    - ``n_fd_atoms``, ``fd_atoms_include``, ``fd_atom_groups``,
      ``fd_seed``: sampled mode only (module docstring); passing them in
      strict mode is a ``ValueError``.
    - ``fd_step``: default 1e-5 (strict) / 1e-4 (sampled).
    - ``fd_rtol`` / ``inv_rtol`` / ``nve_rtol``: override the
      `DEFAULT_TOLERANCES` row. ``rtol`` is a deprecated alias for
      ``fd_rtol``.
    - ``precision``: selects the tolerance row in sampled mode. If omitted,
      it is read from ``backend.effective_config()``.
    - ``repeat_rtol``: ``0.0`` means bitwise repeatability is required;
      ``> 0`` is a declared tolerance (design §3.5); ``None`` (default) takes
      the precision row's value (bitwise for double).
    - ``masses``: required when ``nve_steps > 0``. A scalar, an array of
      x's shape, or ``(n_atoms, 1)``; ``(n_atoms,)`` with 2-D x is a
      ``ValueError``.
    - ``kT``: required when ``nve_steps > 0``, unless ``nve_state`` is
      given, in which case it is derived as ``2*KE/n_dof``.
    - ``nve_cfg``: passed to ``backend.build`` for the NVE run.
    - ``nve_state``: optional explicit NVE start state. Its KE must be > 0.
    - ``n_dof``: kinetic degrees of freedom for the KE scale. Defaults to
      the number of coordinates with nonzero mass, minus 3 when the COM
      velocity is removed.
    - ``remove_com``: remove the mass-weighted centre-of-mass velocity from
      the Maxwell-Boltzmann draw. Default: True for ``(n_atoms, 3)`` probes
      with ``n_atoms >= 2`` (OpenMM Systems usually carry a CMMotionRemover,
      which would otherwise strip the COM kinetic energy at the first step
      and read as an NVE "drift"), False otherwise (1-D toy coordinates).
    """
    # --- argument validation --------------------------------------------
    if mode not in ("strict", "sampled"):
        raise ValueError(f"mode must be 'strict' or 'sampled', got {mode!r}")
    if rtol is not None:
        if fd_rtol is not None:
            raise ValueError("pass fd_rtol, not both fd_rtol and rtol")
        warnings.warn(
            "pes_consistency_suite(rtol=...) is deprecated; use fd_rtol=",
            DeprecationWarning,
            stacklevel=2,
        )
        fd_rtol = rtol
    probes = [np.array(p, dtype=float, copy=True) for p in probes]
    if not probes:
        raise ValueError("pes_consistency_suite needs at least one probe")
    shape = probes[0].shape
    if any(p.shape != shape for p in probes):
        raise ValueError("all probes must have the same shape")
    if fd_step is None:
        fd_step = 1e-5 if mode == "strict" else 1e-4
    if not fd_step > 0.0:
        raise ValueError(f"fd_step must be > 0, got {fd_step!r}")
    sampled_only = {
        "n_fd_atoms": n_fd_atoms is not None,
        "fd_atoms_include": len(list(fd_atoms_include)) > 0,
        "fd_atom_groups": fd_atom_groups is not None,
        "fd_seed": fd_seed != 0,
    }
    if mode == "strict" and any(sampled_only.values()):
        bad = [k for k, v in sampled_only.items() if v]
        raise ValueError(
            f"{', '.join(bad)} only meaningful in mode='sampled' (strict = FD on "
            "every coordinate)"
        )
    if mode == "sampled":
        n_fd_atoms = DEFAULT_N_FD_ATOMS if n_fd_atoms is None else n_fd_atoms
        if int(n_fd_atoms) != n_fd_atoms or n_fd_atoms < 1:
            raise ValueError(f"n_fd_atoms must be an int >= 1, got {n_fd_atoms!r}")
        n_fd_atoms = int(n_fd_atoms)
    if repeat_rtol is not None and not repeat_rtol >= 0.0:
        raise ValueError(f"repeat_rtol must be >= 0, got {repeat_rtol!r}")
    if check_invariance and (len(shape) != 2 or shape[1] != 3):
        raise ValueError(
            f"check_invariance needs probes of shape (n_atoms, 3), got {shape}"
        )
    if nve_steps < 0:
        raise ValueError(f"nve_steps must be >= 0, got {nve_steps!r}")
    if nve_steps > 0 and masses is None:
        raise ValueError(
            "pes_consistency_suite(..., nve_steps>0) requires `masses` to "
            "compute the total mechanical energy PE + KE (potential energy "
            "alone is not conserved by a correct integrator)."
        )

    precision_row = _resolve_precision(backend, precision, mode)
    tol = dict(DEFAULT_TOLERANCES["double" if mode == "strict" else precision_row])
    for name, value in (
        ("fd_rtol", fd_rtol),
        ("inv_rtol", inv_rtol),
        ("nve_rtol", nve_rtol),
        ("repeat_rtol", repeat_rtol),
    ):
        if value is not None:
            if not value >= 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            tol[name] = float(value)

    reasons: list[str] = []
    ev = _Evaluator(backend)

    # --- first evaluation at every probe; suite-wide force scale ---------
    first = [ev(x, f"probe {i}") for i, x in enumerate(probes)]
    F_all = np.stack([F for _, F in first])
    force_scale = float(np.max(np.abs(F_all)))  # NaN-propagating
    f_rms = float(np.sqrt(np.mean(F_all**2)))

    # --- finite differences ----------------------------------------------
    n_atoms = _n_atoms(shape)
    atom_rng = None
    include = np.zeros(0, dtype=np.int64)
    groups: list[np.ndarray] = []
    if mode == "sampled":
        if isinstance(fd_seed, bool) or int(fd_seed) != fd_seed or fd_seed < 0:
            raise ValueError(f"fd_seed must be an int >= 0, got {fd_seed!r}")
        atom_rng = derive_rng(_PES_SUITE_KEY, "fd_atoms" if fd_seed == 0 else f"fd_atoms/{int(fd_seed)}")
        include = _atom_index_array(fd_atoms_include, n_atoms, "fd_atoms_include")
        for name, members in dict(fd_atom_groups or {}).items():
            g = _atom_index_array(members, n_atoms, f"fd_atom_groups[{name!r}]")
            if g.size == 0:
                raise ValueError(f"fd_atom_groups[{name!r}] is empty")
            groups.append(g)
    cutoffs, box_L = _cutoff_guard(backend, shape)
    n_skipped = 0
    fd_atoms_checked: list[list[int]] = []
    excess_max: list[float] = []
    raw_max: list[float] = []
    noise_max: list[float] = []
    atom_rel: list[float] = []
    allowance_max: list[float] = []
    floor_max: list[float] = []
    n_fd_coords = 0
    for i, (x, (_, F0)) in enumerate(zip(probes, first)):
        if atom_rng is None:
            atoms = np.arange(n_atoms)
        else:
            k = min(n_fd_atoms, n_atoms)
            chosen = [atom_rng.choice(n_atoms, size=k, replace=False), include]
            for g in groups:
                chosen.append(atom_rng.choice(g, size=min(n_fd_atoms, g.size), replace=False))
            f_atom_all = np.abs(np.asarray(F0, dtype=float)).reshape(n_atoms, -1).max(axis=1)
            if np.all(np.isfinite(f_atom_all)):
                top = min(_N_TOP_FORCE_ATOMS, n_atoms)
                chosen.append(np.argsort(-f_atom_all, kind="stable")[:top])
            atoms = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in chosen]))
        fd_atoms_checked.append([int(a) for a in atoms])
        errs: dict[int, list[float]] = {}
        noise_of: dict[int, list[float]] = {}
        noises: list[float] = []
        for a in atoms:
            for idx in _atom_coords(shape, int(a)):
                if cutoffs and _crosses_cutoff(x, int(a), idx[1], fd_step, cutoffs, box_L):
                    n_skipped += 1
                    continue
                label = f"FD probe {i} coord {idx}"
                f_h = _fd_component(ev, x, idx, fd_step, label)
                f_h2 = _fd_component(ev, x, idx, 0.5 * fd_step, label)
                errs.setdefault(int(a), []).append(abs(f_h - F0[idx]))
                noise_of.setdefault(int(a), []).append(abs(f_h - f_h2))
                noises.append(abs(f_h - f_h2))
                n_fd_coords += 1
        N_p = _nanmax(noises)
        noise_max.append(N_p)
        arr = np.asarray(noises, dtype=float)
        floor = _FD_NOISE_FACTOR * float(np.median(arr)) if arr.size and not np.isnan(arr).any() else float("nan")
        floor_max.append(floor if arr.size else 0.0)
        for a, e_list in errs.items():
            e = np.asarray(e_list)
            allowance = np.maximum(_FD_TRUNCATION_FACTOR * np.asarray(noise_of[a]), floor)
            exc = np.where(np.isnan(e) | np.isnan(allowance), np.nan, np.maximum(e - allowance, 0.0))
            raw_max.append(_nanmax(e))
            excess_max.append(_nanmax(exc))
            allowance_max.append(_nanmax(allowance))
            f_atom = float(np.max(np.abs(F0[a])))
            atom_rel.append(_ratio(_nanmax(exc), max(f_atom, f_rms)))
    fd_max_rel_err = _ratio(_nanmax(excess_max), force_scale)
    fd_raw_rel_err = _ratio(_nanmax(raw_max), force_scale)
    fd_noise_rel = _ratio(_nanmax(noise_max), force_scale)
    fd_allowance_rel = _ratio(_nanmax(allowance_max), force_scale)
    fd_floor_rel = _ratio(_nanmax(floor_max), force_scale)
    fd_max_atom_rel_err = _nanmax(atom_rel)
    if not fd_max_rel_err <= tol["fd_rtol"]:
        reasons.append(
            f"fd: fd_max_rel_err={fd_max_rel_err:.3e} > fd_rtol={tol['fd_rtol']:.1e}"
        )
    if not fd_max_atom_rel_err <= tol["fd_rtol"]:
        reasons.append(
            f"fd_atom: fd_max_atom_rel_err={fd_max_atom_rel_err:.3e} > "
            f"fd_rtol={tol['fd_rtol']:.1e}"
        )
    if n_fd_coords == 0:
        reasons.append(
            f"fd_unresolved: every FD coordinate ({n_skipped}) was skipped because its "
            "stencil crosses a cutoff; the FD check has no data (use other probes or fd_seed)"
        )
    if not fd_floor_rel <= tol["fd_rtol"]:
        reasons.append(
            f"fd_unresolved: FD noise floor {fd_floor_rel:.3e} (relative to force "
            f"scale {force_scale:.3e}) > fd_rtol={tol['fd_rtol']:.1e}; the FD "
            "check has no power here (all probes stationary, or fd_step "
            "unsuited to the precision)"
        )

    # --- translation / rotation invariance (opt-in) ------------------------
    translation_err: float | None = None
    rotation_err: float | None = None
    if check_invariance:
        R = _fixed_rotation_matrix()
        t_errs: list[float] = []
        r_errs: list[float] = []
        for i, (x, (E0, F0)) in enumerate(zip(probes, first)):
            x_t = x + _TRANSLATION
            Et, Ft = ev(x_t, f"probe {i} translated")
            d_t = float(np.sum(np.abs(x_t - x)))
            t_errs.append(_ratio(abs(Et - E0), force_scale * d_t))
            t_errs.append(_ratio(float(np.max(np.abs(Ft - F0))), force_scale))

            x_r = x @ R.T
            Er, Fr = ev(x_r, f"probe {i} rotated")
            d_r = float(np.sum(np.abs(x_r - x)))
            r_errs.append(_ratio(abs(Er - E0), force_scale * d_r))
            r_errs.append(
                _ratio(float(np.max(np.abs(Fr - F0 @ R.T))), force_scale)
            )
        translation_err = _nanmax(t_errs)
        rotation_err = _nanmax(r_errs)
        if not translation_err <= tol["inv_rtol"]:
            reasons.append(
                f"invariance_translation: {translation_err:.3e} > "
                f"inv_rtol={tol['inv_rtol']:.1e}"
            )
        if not rotation_err <= tol["inv_rtol"]:
            reasons.append(
                f"invariance_rotation: {rotation_err:.3e} > "
                f"inv_rtol={tol['inv_rtol']:.1e}"
            )

    # --- repeatability (after many evaluations at other positions) --------
    repeat_bitwise = True
    rep_errs: list[float] = []
    for i, (x, (E0, F0)) in enumerate(zip(probes, first)):
        E2, F2 = ev(x, f"probe {i} repeat")
        if E2 != E0 or not np.array_equal(F2, F0):
            repeat_bitwise = False
        rep_errs.append(_ratio(abs(E2 - E0), force_scale * fd_step))
        rep_errs.append(_ratio(float(np.max(np.abs(F2 - F0))), force_scale))
    repeat_max_rel_err = _nanmax(rep_errs)
    repeat_rtol = tol["repeat_rtol"]
    if repeat_rtol == 0.0:
        if not repeat_bitwise:
            reasons.append(
                "repeat: energy_forces is not bitwise repeatable "
                f"(repeat_max_rel_err={repeat_max_rel_err:.3e}; pass "
                "repeat_rtol>0 to declare a tolerance)"
            )
    elif not repeat_max_rel_err <= repeat_rtol:
        reasons.append(
            f"repeat: repeat_max_rel_err={repeat_max_rel_err:.3e} > "
            f"repeat_rtol={repeat_rtol:.1e}"
        )

    # --- short NVE run from a thermally perturbed state (opt-in) ----------
    nve_rel_drift: float | None = None
    kT_used: float | None = None
    n_dof_used: int | None = None
    if nve_steps > 0:
        nve_rel_drift, kT_used, n_dof_used = _nve_drift(
            backend, ev, probes[0], masses, kT, nve_steps, nve_cfg, nve_state,
            n_dof, remove_com, reasons,
        )
        if not np.isnan(nve_rel_drift) and not nve_rel_drift <= tol["nve_rtol"]:
            reasons.append(
                f"nve: nve_rel_drift={nve_rel_drift:.3e} (of n_dof*kT/2) > "
                f"nve_rtol={tol['nve_rtol']:.1e}"
            )

    all_finite = ev.nonfinite is None
    if not all_finite:
        reasons.insert(0, f"nonfinite: {ev.nonfinite}")

    return PESReport(
        passed=not reasons,
        fd_max_rel_err=fd_max_rel_err,
        translation_err=translation_err,
        rotation_err=rotation_err,
        repeat_bitwise=repeat_bitwise,
        nve_rel_drift=nve_rel_drift,
        nve_abs_drift_kT=(
            None if nve_rel_drift is None or kT_used is None or n_dof_used is None
            else float(nve_rel_drift) * n_dof_used / 2.0
        ),
        mode=mode,
        precision=precision_row,
        all_finite=all_finite,
        reasons=reasons,
        force_scale=force_scale,
        fd_raw_rel_err=fd_raw_rel_err,
        fd_noise_rel=fd_noise_rel,
        fd_allowance_rel=fd_allowance_rel,
        fd_noise_floor_rel=fd_floor_rel,
        fd_atoms=fd_atoms_checked,
        fd_step=float(fd_step),
        fd_max_atom_rel_err=fd_max_atom_rel_err,
        n_fd_coords=n_fd_coords,
        repeat_max_rel_err=repeat_max_rel_err,
        tolerances=tol,
        kT=kT_used,
        n_dof=n_dof_used,
        fd_skipped_cutoff=n_skipped,
        fd_cutoffs=list(cutoffs),
    )


def _nve_drift(
    backend: PotentialBackend,
    ev: _Evaluator,
    x_probe: np.ndarray,
    masses,
    kT: float | None,
    nve_steps: int,
    nve_cfg: PhysicsConfig | None,
    nve_state: MDState | None,
    n_dof: int | None,
    remove_com: bool | None,
    reasons: list[str],
) -> tuple[float, float, int]:
    """Run the NVE check. Returns (nve_rel_drift, kT, n_dof).

    NaN drift means the run went non-finite (``nve_nonfinite`` reason added).
    """
    if nve_state is not None:
        x0 = np.array(nve_state.x, dtype=float, copy=True)
        v0 = np.array(nve_state.v, dtype=float, copy=True)
        if v0.shape != x0.shape:
            raise ValueError(
                f"nve_state.v shape {v0.shape} != nve_state.x shape {x0.shape}"
            )
        start = MDState(x=x0, v=v0, t=float(nve_state.t), box=nve_state.box)
    else:
        x0 = np.array(x_probe, dtype=float, copy=True)
        start = None
    m = _broadcast_masses(masses, x0.shape)
    moving = m > 0.0
    atomic = x0.ndim == 2 and x0.shape[1] == 3 and x0.shape[0] >= 2
    if remove_com is None:
        remove_com = start is None and atomic
    if remove_com and not atomic:
        raise ValueError("remove_com needs (n_atoms >= 2, 3) coordinates")
    if remove_com and start is not None:
        raise ValueError("remove_com applies to the suite's own MB draw, not nve_state")
    if n_dof is None:
        n_dof = int(np.count_nonzero(moving)) - (3 if remove_com else 0)
    if not (int(n_dof) == n_dof and n_dof >= 1):
        raise ValueError(f"n_dof must be an int >= 1, got {n_dof!r}")
    n_dof = int(n_dof)

    def kinetic(v: np.ndarray) -> float:
        return 0.5 * float(np.sum(m * v**2))

    if start is None:
        if kT is None:
            raise ValueError(
                "nve_steps>0 needs kT (to draw thermal Maxwell-Boltzmann "
                "start velocities and to normalise the drift by n_dof*kT/2), "
                "or an explicit nve_state"
            )
        if not (np.isfinite(kT) and kT > 0.0):
            raise ValueError(f"kT must be finite and > 0, got {kT!r}")
        rng = derive_rng(_PES_SUITE_KEY, "nve_v")
        xi = rng.standard_normal(size=x0.shape)
        v0 = np.zeros_like(x0)
        v0[moving] = xi[moving] * np.sqrt(kT / m[moving])
        if remove_com:
            p_com = np.sum(m * v0, axis=0)
            v0 = np.where(moving, v0 - p_com / np.sum(m, axis=0), 0.0)
        start = MDState(x=x0, v=v0, t=0.0)
    else:
        ke0 = kinetic(start.v)
        if not ke0 > 0.0:
            raise ValueError(
                "nve_state has zero kinetic energy: an NVE check started at "
                "rest is vacuous at a minimum; give it thermal velocities"
            )
        if kT is None:
            kT = 2.0 * ke0 / n_dof
        elif not (np.isfinite(kT) and kT > 0.0):
            raise ValueError(f"kT must be finite and > 0, got {kT!r}")
    ke_scale = 0.5 * n_dof * float(kT)

    propagator = backend.build(start, nve_cfg, _PES_SUITE_KEY)

    def total_energy(s: MDState, label: str) -> float:
        if not (np.all(np.isfinite(s.x)) and np.all(np.isfinite(s.v))):
            if ev.nonfinite is None:
                ev.nonfinite = f"non-finite propagated state at {label}"
            return float("nan")
        PE, _ = ev(np.asarray(s.x, dtype=float), label, box=s.box)
        return PE + kinetic(np.asarray(s.v, dtype=float))

    E0 = total_energy(propagator.get_state(), "NVE t=0")
    energies = [E0]
    n_chunks = min(nve_steps, 20)
    base_steps, remainder = divmod(nve_steps, n_chunks)
    for i in range(n_chunks):
        if not np.isfinite(energies[-1]):
            break
        propagator.run(base_steps + (1 if i < remainder else 0))
        energies.append(total_energy(propagator.get_state(), f"NVE chunk {i + 1}"))
    E = np.asarray(energies)
    if not np.all(np.isfinite(E)):
        bad = int(np.argmax(~np.isfinite(E)))
        reasons.append(
            f"nve_nonfinite: total energy / state non-finite at NVE sample {bad}"
        )
        return float("nan"), float(kT), n_dof
    return float(np.max(np.abs(E - E0))) / ke_scale, float(kT), n_dof
