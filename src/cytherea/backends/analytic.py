"""Analytic toy potentials and the `analytic` PotentialBackend.

Every potential shares one interface, `energy_grad(x) -> (E, dE/dx)`, with
`dE/dx` the *same shape* as `x` (a gradient, not yet negated into a force --
`AnalyticBackend.energy_forces` does that negation once, in one place, so
every potential class only ever has to get dV/dx right).

Units are reduced/dimensionless throughout this module (design doc: any
config using these must declare `units: reduced` explicitly) -- these are
toy potentials for A0 acceptance tests, not physical molecular systems.

Coordinate-shape convention per potential:
- FreeParticle(dim), Harmonic(k, dim): x has shape (dim,).
- DoubleWell1D: x has shape (1,) (a single reduced coordinate).
- DoubleWell2D, MullerBrown, ChannelDoubleWell2D: x has shape (2,), read as
  (x, y).
- LJCluster(n_atoms, ...): x has shape (n_atoms, 3) -- one row per atom,
  *not* flattened to (3*n_atoms,). Chosen because pairwise distances and the
  translation/rotation-invariance checks in the PES suite are far more
  direct to express and verify against a (n_atoms, 3) array of position
  vectors than against a flat vector callers would have to reshape anyway.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
from typing import Literal, Protocol

import numpy as np
from scipy.special import expit

from cytherea.backends.base import MDState, PhysicsConfig, Propagator
from cytherea.keys import Key, derive_rng


class Potential(Protocol):
    """Structural interface every analytic potential implements.

    Its parameters must be describable for the physics hash (contract K9):
    a dataclass, or plain-data instance attributes, or an optional
    ``params() -> dict`` method (see `_potential_params`)."""

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]: ...


def _check_shape(x: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if x.shape != shape:
        raise ValueError(f"{name} expects x of shape {shape}, got {x.shape}")


@dataclasses.dataclass(frozen=True)
class FreeParticle:
    """V(x) = 0 for all x (shape (dim,)). Used as a trivial sanity backend."""

    dim: int

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (self.dim,), "FreeParticle")
        return 0.0, np.zeros_like(x)


@dataclasses.dataclass(frozen=True)
class DoubleWell1D:
    """V(x) = barrier * ((x/x0)**2 - 1)**2. Minima at x=+-x0 (V=0), barrier
    top at x=0 (V=barrier). x has shape (1,).
    """

    barrier: float
    x0: float = 1.0

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (1,), "DoubleWell1D")
        u = (x / self.x0) ** 2 - 1.0
        E = float(self.barrier * u[0] ** 2)
        grad = self.barrier * 4.0 * x * u / (self.x0**2)
        return E, grad


@dataclasses.dataclass(frozen=True)
class DoubleWell2D:
    """V(x, y) = barrier * (x**2 - 1)**2 + 0.5 * ky * y**2. x has shape (2,)."""

    barrier: float
    ky: float

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (2,), "DoubleWell2D")
        xx, yy = x[0], x[1]
        E = self.barrier * (xx**2 - 1.0) ** 2 + 0.5 * self.ky * yy**2
        grad = np.array(
            [4.0 * self.barrier * xx * (xx**2 - 1.0), self.ky * yy]
        )
        return float(E), grad


# Standard Müller-Brown parameters (four Gaussian-like terms).
_MB_A = (-200.0, -100.0, -170.0, 15.0)
_MB_a = (-1.0, -1.0, -6.5, 0.7)
_MB_b = (0.0, 0.0, 11.0, 0.6)
_MB_c = (-10.0, -10.0, -6.5, 0.7)
_MB_X0 = (1.0, 0.0, -0.5, -1.0)
_MB_Y0 = (0.0, 0.5, 1.5, 1.0)


@dataclasses.dataclass(frozen=True)
class MullerBrown:
    """Standard Müller-Brown surface, scaled by `scale`:

    V(x, y) = scale * sum_k A_k * exp(a_k*(x-x0_k)**2 + b_k*(x-x0_k)*(y-y0_k)
                                       + c_k*(y-y0_k)**2)

    with the four-term parameter set (A, a, b, c, x0, y0) fixed to the
    literature values. x has shape (2,).
    """

    scale: float = 1.0

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (2,), "MullerBrown")
        xx, yy = x[0], x[1]
        E = 0.0
        dEdx = 0.0
        dEdy = 0.0
        for A, a, b, c, x0, y0 in zip(_MB_A, _MB_a, _MB_b, _MB_c, _MB_X0, _MB_Y0):
            dx = xx - x0
            dy = yy - y0
            term = A * np.exp(a * dx**2 + b * dx * dy + c * dy**2)
            E += term
            dEdx += term * (2.0 * a * dx + b * dy)
            dEdy += term * (b * dx + 2.0 * c * dy)
        E *= self.scale
        grad = self.scale * np.array([dEdx, dEdy])
        return float(E), grad


@dataclasses.dataclass(frozen=True)
class ChannelDoubleWell2D:
    """Two parallel x-direction double-well channels (y>0 and y<0) separated
    by a wall at y=0, for Task 11's memory-effect counterexample.

        V(x, y) = B(y) * (x**2 - 1)**2 + W(y)

    where:
        B(y) = barrier_minus + (barrier_plus - barrier_minus) * sigmoid(y/wb)
            smoothly interpolates the channel's x-barrier height from
            barrier_minus (y << 0) to barrier_plus (y >> 0); wb is a small
            fixed transition width, so B(y) behaves like a step at y=0 while
            remaining everywhere differentiable.
        W(y) = wall * exp(-(y/ww)**2)
            a smooth bump centered at y=0, independent of x; ww is a fixed
            width.
        C(y) = 0.5 * k_confine * max(|y| - y_confine, 0)**2
            a weak y-confinement outside |y| > y_confine (C^1: value and
            slope vanish at |y| = y_confine), so the Boltzmann density is
            normalisable and y cannot diffuse off to infinity. It is exactly
            zero for |y| <= y_confine, so both channels (|y| ~ 1) and the
            wall region are untouched.

    The full potential is V = B(y)*(x**2-1)**2 + W(y) + C(y).

    `transition_width` (wb, default 0.1) and `wall_width` (ww, default 0.25)
    are dataclass fields so they appear in provenance; the defaults keep the
    brief's 3-argument constructor `(barrier_plus, barrier_minus, wall)`.

    Because W and C do not depend on x, V(+-1, y) = W(y) + C(y) identically (the
    (x**2-1)**2 factor vanishes at x=+-1) -- so the wall term never perturbs
    either channel's double-well shape, and the x-barrier height at any
    fixed y, V(0, y) - V(+-1, y), is exactly B(y), independent of `wall`.
    `wall` should be chosen >> kT so a trajectory confined to one channel
    cannot cross y=0 into the other.

    x has shape (2,).
    """

    barrier_plus: float
    barrier_minus: float
    wall: float
    transition_width: float = 0.1
    wall_width: float = 0.25
    y_confine: float = 2.0
    k_confine: float = 10.0

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (2,), "ChannelDoubleWell2D")
        xx, yy = x[0], x[1]
        wb = self.transition_width
        ww = self.wall_width

        z = yy / wb
        sig = float(expit(z))  # overflow-free logistic sigmoid
        dsig_dy = sig * (1.0 - sig) / wb
        B = self.barrier_minus + (self.barrier_plus - self.barrier_minus) * sig
        dB_dy = (self.barrier_plus - self.barrier_minus) * dsig_dy

        W = self.wall * np.exp(-((yy / ww) ** 2))
        dW_dy = W * (-2.0 * yy / (ww**2))

        excess = max(abs(yy) - self.y_confine, 0.0)
        C = 0.5 * self.k_confine * excess**2
        dC_dy = self.k_confine * excess * np.sign(yy)

        u = xx**2 - 1.0
        E = B * u**2 + W + C
        dEdx = B * 4.0 * xx * u
        dEdy = dB_dy * u**2 + dW_dy + dC_dy
        return float(E), np.array([dEdx, dEdy])


@dataclasses.dataclass(frozen=True)
class Harmonic:
    """V(x) = 0.5 * k * |x|**2. x has shape (dim,)."""

    k: float
    dim: int

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (self.dim,), "Harmonic")
        E = 0.5 * self.k * float(np.sum(x**2))
        grad = self.k * x
        return E, grad


@dataclasses.dataclass(frozen=True)
class LJCluster:
    """n_atoms particles under pairwise standard 12-6 Lennard-Jones:

        V = sum_{i<j} 4*epsilon*((sigma/r_ij)**12 - (sigma/r_ij)**6)

    x has shape (n_atoms, 3) -- see module docstring for the shape
    convention rationale.
    """

    n_atoms: int
    epsilon: float = 1.0
    sigma: float = 1.0

    def energy_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        _check_shape(x, (self.n_atoms, 3), "LJCluster")
        eps, sig = self.epsilon, self.sigma
        E = 0.0
        grad = np.zeros_like(x)
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                rij = x[i] - x[j]
                r2 = float(np.dot(rij, rij))
                r = np.sqrt(r2)
                sr6 = (sig / r) ** 6
                sr12 = sr6**2
                E += 4.0 * eps * (sr12 - sr6)
                # dE/dr = -24*eps*(2*sr12 - sr6)/r
                dEdr = -24.0 * eps * (2.0 * sr12 - sr6) / r
                fij = dEdr * rij / r  # dE/d(rij), rij = x[i]-x[j]
                grad[i] += fij
                grad[j] -= fij
        return float(E), grad


class _AnalyticPropagator:
    """`Propagator` returned by `AnalyticBackend.build` (Task 4).

    Two integrators, both driven entirely by one `numpy.random.Generator`
    created once (in `AnalyticBackend.build`, via
    ``derive_rng(rng_key, "propagate")``) and stored for the propagator's
    whole lifetime -- `set_state` below rebinds `x`/`v`/`t` only and never
    touches `_rng`, so resuming from a checkpoint continues drawing from
    where that Generator's stream left off rather than restarting it.

    ``overdamped``
        Euler-Maruyama for overdamped (Brownian) dynamics::

            x_{n+1} = x_n - (1/(mass*gamma)) * dV/dx(x_n) * dt
                          + sqrt(2*D*dt) * xi_n

        with ``D = kT / (mass*gamma)`` and mobility ``1/(mass*gamma)``.
        Per controller ruling R14 (fix round 1), `gamma` is a *rate*
        (units 1/time, matching `PhysicsConfig.friction_per_ps`) for
        *both* integrators -- the same role it plays in BAOAB below. This
        is exactly BAOAB's overdamped (high-friction) limit: dropping the
        inertial term from `dv = (F/mass - gamma*v) dt + noise` and
        solving `0 = F/mass - gamma*v + noise/mass` for `v` gives
        `v = F/(mass*gamma) + noise/(mass*gamma)`, i.e. mobility
        `1/(mass*gamma)` and `D = kT/(mass*gamma)` -- identical to the
        task-4 brief's "D = kT/gamma" convention when `mass=1`. (Before
        this ruling, overdamped used `D = kT/gamma` with no mass
        dependence at all, which silently disagreed with BAOAB's `gamma`
        role whenever `mass != 1`; this is what the fix-round-1 review
        caught.) `v` in every `MDState` this integrator returns is
        all-zeros: overdamped dynamics has no velocity degree of freedom,
        so there is nothing meaningful to report; this is a documented
        convention, not an oversight.

    ``baoab``
        Leimkuhler-Matthews BAOAB for underdamped Langevin dynamics::

            dx = v dt
            dv = (F(x)/mass - gamma*v) dt + sqrt(2*gamma*kT/mass) dW

        via the B-A-O-A-B splitting (B = half-kick using the force at the
        current position, A = half-drift, O = one *exact* Ornstein-
        Uhlenbeck update of v with decay ``c = exp(-gamma*dt)`` and
        stationary variance ``kT/mass``). At `gamma=0`, `c=1` and the O
        step's noise amplitude is exactly zero, so BAOAB degenerates
        exactly (not approximately) to velocity Verlet -- the NVE limit
        Test 4.5 relies on.
    """

    def __init__(
        self,
        potential: Potential,
        integrator: Literal["overdamped", "baoab"],
        dt: float,
        kT: float,
        gamma: float,
        mass: float,
        state: MDState,
        rng: np.random.Generator,
    ) -> None:
        if integrator not in ("overdamped", "baoab"):
            raise ValueError(f"unknown integrator {integrator!r}")
        self._potential = potential
        self._integrator = integrator
        self._dt = float(dt)
        self._kT = float(kT)
        self._gamma = float(gamma)
        self._mass = float(mass)
        self._rng = rng
        self._set_state_arrays(state)

        if integrator == "baoab":
            # Exact-OU decay/noise amplitude for the O step (Leimkuhler-
            # Matthews); c=1, noise=0 at gamma=0 recovers velocity Verlet.
            self._c = float(np.exp(-self._gamma * self._dt))
            self._noise_scale = float(
                np.sqrt(max(0.0, 1.0 - self._c**2) * self._kT / self._mass)
            )
            self._refresh_cached_force()

    def _set_state_arrays(self, state: MDState) -> None:
        x = np.array(state.x, dtype=float, copy=True)
        v = np.array(state.v, dtype=float, copy=True)
        if v.shape != x.shape:
            raise ValueError(
                f"MDState.v has shape {v.shape} but x has shape {x.shape}; "
                "they must match exactly (no broadcasting)"
            )
        self._x = x
        # Overdamped dynamics has no velocity degree of freedom: v is
        # all-zeros in *every* MDState it returns, including right after
        # build()/set_state() (not only after run()).
        self._v = np.zeros_like(x) if self._integrator == "overdamped" else v
        # Contract K1: time is t0 + step_index * dt (multiplied, never
        # accumulated), so a long run has no O(n * eps) clock drift.
        self._t0 = float(state.t)
        self._n_steps = 0
        self._box = state.box

    @property
    def _t(self) -> float:
        return self._t0 + self._n_steps * self._dt

    @staticmethod
    def _check_n_steps(n_steps: int) -> int:
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 0:
            raise ValueError(f"n_steps must be an int >= 0, got {n_steps!r}")
        return int(n_steps)

    def _refresh_cached_force(self) -> None:
        _, grad = self._potential.energy_grad(self._x)
        self._force = -grad

    def run(self, n_steps: int) -> None:
        n_steps = self._check_n_steps(n_steps)
        if self._integrator == "overdamped":
            self._run_overdamped(n_steps)
        else:
            self._run_baoab(n_steps)

    def _run_overdamped(self, n_steps: int) -> None:
        # D = kT/(mass*gamma), mobility = 1/(mass*gamma) (R14: gamma is a
        # rate for both integrators). Both are constant across steps for
        # fixed dt/kT/gamma/mass, so hoist them out of the loop.
        mobility = 1.0 / (self._mass * self._gamma)
        D = self._kT * mobility
        noise_scale = np.sqrt(2.0 * D * self._dt)
        x = self._x
        for _ in range(n_steps):
            _, grad = self._potential.energy_grad(x)
            xi = self._rng.standard_normal(size=x.shape)
            x = x - (mobility * grad) * self._dt + noise_scale * xi
            self._n_steps += 1
        self._x = x
        self._v = np.zeros_like(x)

    def _run_baoab(self, n_steps: int) -> None:
        dt_half = 0.5 * self._dt
        x, v, force = self._x, self._v, self._force
        for _ in range(n_steps):
            # B: half kick with the force at the current position.
            v = v + dt_half * force / self._mass
            # A: half drift.
            x = x + dt_half * v
            # O: exact Ornstein-Uhlenbeck step (identity at gamma=0).
            if self._gamma > 0.0:
                xi = self._rng.standard_normal(size=v.shape)
                v = self._c * v + self._noise_scale * xi
            # A: half drift.
            x = x + dt_half * v
            # B: half kick with the force at the new position.
            _, grad = self._potential.energy_grad(x)
            force = -grad
            v = v + dt_half * force / self._mass
            self._n_steps += 1
        self._x, self._v, self._force = x, v, force

    @property
    def dt(self) -> float:
        """Read-only: the propagator's own integration timestep (controller
        ruling R34), in this backend's reduced units. Fixed at `build()`
        time; there is no setter.
        """
        return self._dt

    def get_state(self) -> MDState:
        return MDState(
            x=self._x.copy(), v=self._v.copy(), t=self._t, box=self._box
        )

    def set_state(self, s: MDState) -> None:
        self._set_state_arrays(s)
        if self._integrator == "baoab":
            self._refresh_cached_force()


class AnalyticBackend:
    """`PotentialBackend` wrapping one analytic `Potential`.

    All dynamics parameters come from the constructor; `build()`'s `cfg`
    argument is accepted (per the shared `PotentialBackend` protocol) but
    ignored -- `cfg` may be `None`.
    """

    kind: Literal["analytic"] = "analytic"
    gpu_resident: bool = False

    def __init__(
        self,
        potential: Potential,
        integrator: Literal["overdamped", "baoab"],
        dt: float,
        kT: float,
        gamma: float,
        mass: float = 1.0,
    ) -> None:
        # Fail fast on construction (controller ruling R15 #4) rather than
        # letting an invalid combination silently produce NaN/inf inside a
        # propagator's step loop, or an ambiguous-looking numerical result
        # much later. overdamped's mobility/D are 1/(mass*gamma) (R14):
        # gamma=0 there is a division by zero, and gamma<0 would reverse
        # the drift's sign -- both physically meaningless. BAOAB tolerates
        # gamma=0 (Test 4.5's exact NVE/velocity-Verlet limit) but not
        # gamma<0 (a growing, not decaying, OU process).
        if integrator not in ("overdamped", "baoab"):
            raise ValueError(f"unknown integrator {integrator!r}")
        if integrator == "overdamped" and not gamma > 0.0:
            raise ValueError(
                f"overdamped requires gamma > 0 (got {gamma!r}); "
                "D = kT/(mass*gamma) is undefined at gamma=0 and the drift "
                "direction is wrong at gamma<0"
            )
        if integrator == "baoab" and gamma < 0.0:
            raise ValueError(f"baoab requires gamma >= 0 (got {gamma!r})")
        if not dt > 0.0:
            raise ValueError(f"dt must be > 0 (got {dt!r})")
        if kT < 0.0:
            raise ValueError(f"kT must be >= 0 (got {kT!r})")
        if not mass > 0.0:
            raise ValueError(f"mass must be > 0 (got {mass!r})")

        self.potential = potential
        self.integrator = integrator
        self.dt = dt
        self.kT = kT
        self.gamma = gamma
        self.mass = mass

    def build(
        self, s: MDState, cfg: PhysicsConfig | None, rng_key: Key
    ) -> Propagator:
        # Created exactly once per build(): the propagator's whole lifetime
        # (including across set_state() checkpoint resumes) draws from this
        # one Generator's stream -- see _AnalyticPropagator's docstring.
        rng = derive_rng(rng_key, "propagate")
        return _AnalyticPropagator(
            potential=self.potential,
            integrator=self.integrator,
            dt=self.dt,
            kT=self.kT,
            gamma=self.gamma,
            mass=self.mass,
            state=s,
            rng=rng,
        )

    def energy_forces(self, x: np.ndarray, box: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        E, dEdx = self.potential.energy_grad(np.asarray(x, dtype=float))
        return E, -dEdx

    def effective_config(self, cfg: PhysicsConfig | None = None) -> dict:
        """Plain, JSON-serialisable dict of everything that drives this
        backend's dynamics (contracts K8, K9): the potential (class and
        every parameter -- see `_potential_params`; a potential that cannot
        be described stably raises TypeError), the integrator and its
        dt/kT/gamma/mass, and ``units: "reduced"``.

        The analytic backend ignores any `PhysicsConfig` passed to `build`,
        so the result is the same for every `cfg` (the argument exists only
        for signature symmetry with the OpenMM backend). Hence the physics
        hash of a run with an explicit cfg equals that of ``cfg=None``, and
        both change whenever the backend's own dynamics change (K9).
        """
        return {
            "backend": self.kind,
            "units": "reduced",
            "potential": type(self.potential).__qualname__,
            "potential_params": _potential_params(self.potential),
            "integrator": self.integrator,
            "dt": float(self.dt),
            "kT": float(self.kT),
            "gamma": float(self.gamma),
            "mass": float(self.mass),
        }

    def provenance(self, cfg: PhysicsConfig | None = None) -> dict:
        """Backend description. `cfg` is accepted for protocol symmetry and
        ignored (see `effective_config`). Works for any `Potential`, not only
        dataclass ones."""
        prov = self.effective_config()
        prov["kind"] = prov.pop("backend")
        return prov


# Arrays with more elements than this are described by digest, not listed.
_MAX_LISTED_ARRAY = 64
_MAX_PARAM_DEPTH = 16


def _param_error(where: str, what: str) -> TypeError:
    return TypeError(
        f"effective_config: cannot describe potential parameter {where} ({what}) "
        "stably; give the potential a params() method returning a plain dict of "
        "everything that determines its energy (contract K9: no repr(), no "
        "identity-based hashing)"
    )


def _plain(value, where: str = "potential", depth: int = 0):
    """Stable, plain-JSON description of a potential parameter (contract K9,
    fixreview-p4 m2): numbers, strings, bools and None as themselves; numpy
    scalars as Python numbers; arrays as lists (more than
    `_MAX_LISTED_ARRAY` elements: ``{"ndarray_sha256", "shape", "dtype"}``);
    lists/tuples and str-keyed dicts element-wise; nested dataclasses and
    plain objects as ``{"type", "params"}`` (see `_potential_params`).
    Callables, classes and anything else raise TypeError -- never repr(),
    whose memory addresses differ between processes."""
    if depth > _MAX_PARAM_DEPTH:
        raise _param_error(where, "nested too deeply (a reference cycle?)")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise _param_error(where, f"array of dtype {value.dtype}")
        if value.size > _MAX_LISTED_ARRAY:
            arr = np.ascontiguousarray(value)
            return {
                "ndarray_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
        return [_plain(v, where, depth + 1) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{where}[{i}]", depth + 1) for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise _param_error(where, f"non-str dict key {k!r}")
            out[k] = _plain(v, f"{where}[{k!r}]", depth + 1)
        return out
    if isinstance(value, type) or inspect.isroutine(value) or isinstance(value, functools.partial):
        raise _param_error(where, f"a {type(value).__name__}")
    if (dataclasses.is_dataclass(value) or callable(getattr(value, "params", None))
            or hasattr(value, "__dict__") or hasattr(type(value), "__slots__")):
        return {
            "type": type(value).__qualname__,
            "params": _potential_params(value, where, depth + 1),
        }
    raise _param_error(where, f"type {type(value).__name__}")


def _slot_names(obj) -> list[str]:
    names: list[str] = []
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(n for n in slots if n not in ("__dict__", "__weakref__"))
    return names


def _potential_params(potential, where: str = "potential", depth: int = 0) -> dict:
    """Parameters of a `Potential` (contract K9, fixreview-p4 m2), in order
    of preference:

    1. ``potential.params()`` if defined -- a plain dict of everything that
       determines the energy; use it when the instance also holds caches or
       other call-history state that must not enter the hash;
    2. its dataclass fields, when it is a dataclass;
    3. else *all* its instance attributes, private ones included (``self._k``
       is a parameter like any other), plus any ``__slots__`` values.

    Every value goes through `_plain`, which raises TypeError for anything
    it cannot describe stably (a lambda, an opaque handle, ...). A potential
    with no instance state at all has ``{}`` (its class identifies it)."""
    hook = getattr(potential, "params", None)
    if callable(hook) and not isinstance(potential, type):
        got = hook()
        if not isinstance(got, dict):
            raise _param_error(f"{where}.params()", f"returned {type(got).__name__}, not a dict")
        return _plain(got, f"{where}.params()", depth)
    if dataclasses.is_dataclass(potential) and not isinstance(potential, type):
        return {
            f.name: _plain(getattr(potential, f.name), f"{where}.{f.name}", depth)
            for f in dataclasses.fields(potential)
        }
    attrs = dict(getattr(potential, "__dict__", None) or {})
    for name in _slot_names(potential):
        if hasattr(potential, name):
            attrs[name] = getattr(potential, name)
    return {k: _plain(v, f"{where}.{k}", depth) for k, v in sorted(attrs.items())}
