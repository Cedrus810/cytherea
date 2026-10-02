"""Tests for cytherea.backends.analytic: the analytic toy potentials used
by every A0 acceptance test (task-3-brief.md, "必测用例" 3.1-3.3, 3.6, plus
one extra test for ChannelDoubleWell2D's barrier/wall geometry which the
controller ruling asked for explicitly).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from cytherea.backends.analytic import (
    AnalyticBackend,
    ChannelDoubleWell2D,
    DoubleWell1D,
    DoubleWell2D,
    FreeParticle,
    Harmonic,
    LJCluster,
    MullerBrown,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _central_diff_grad(potential, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Reference finite-difference gradient of E w.r.t. x, computed directly
    against `potential.energy_grad` (independent of pes_suite.py, which is
    tested separately in test_pes_suite.py -- this file only pins down the
    potentials themselves).
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy()
        xp[idx] += h
        xm = x.copy()
        xm[idx] -= h
        Ep, _ = potential.energy_grad(xp)
        Em, _ = potential.energy_grad(xm)
        grad[idx] = (Ep - Em) / (2.0 * h)
    return grad


def _rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), floor))


# ---------------------------------------------------------------------------
# 3.1: analytic gradient matches central-difference gradient for every
# potential, 20 random probes each, relative error < 1e-6.
# ---------------------------------------------------------------------------

def _rng():
    # Deterministic, local-only RNG for test-probe generation (not sampled
    # physics -- no Key/derive_rng needed for picking test coordinates).
    return np.random.default_rng(12345)


@pytest.mark.parametrize(
    "potential, sample_x",
    [
        (FreeParticle(dim=3), lambda rng: rng.uniform(-5, 5, size=(3,))),
        (DoubleWell1D(barrier=5.0), lambda rng: rng.uniform(-3, 3, size=(1,))),
        (DoubleWell2D(barrier=3.0, ky=2.0), lambda rng: rng.uniform(-3, 3, size=(2,))),
        (MullerBrown(), lambda rng: rng.uniform([-1.5, -0.5], [1.2, 2.2])),
        (
            ChannelDoubleWell2D(barrier_plus=4.0, barrier_minus=8.0, wall=50.0),
            # avoid the immediate vicinity of the y=0 transition/wall, where
            # curvature is deliberately sharp (see class docstring) and a
            # central difference with h=1e-6 would need a much smaller step
            # to resolve; |y| >= 0.3 is >> the transition width (0.1).
            lambda rng: np.array(
                [rng.uniform(-2, 2), rng.uniform(0.3, 2.0) * rng.choice([-1.0, 1.0])]
            ),
        ),
        (Harmonic(k=2.5, dim=4), lambda rng: rng.uniform(-4, 4, size=(4,))),
        (
            LJCluster(n_atoms=4, epsilon=1.2, sigma=0.9),
            # keep pairwise distances well away from the r->0 singularity
            lambda rng: rng.uniform(-1.5, 1.5, size=(4, 3))
            + np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        ),
    ],
    ids=[
        "FreeParticle",
        "DoubleWell1D",
        "DoubleWell2D",
        "MullerBrown",
        "ChannelDoubleWell2D",
        "Harmonic",
        "LJCluster",
    ],
)
def test_force_equals_minus_gradient_20_random_probes(potential, sample_x):
    rng = _rng()
    for _ in range(20):
        x = sample_x(rng)
        E, dEdx = potential.energy_grad(x)
        assert np.all(np.isfinite(dEdx))
        F_fd = -_central_diff_grad(potential, x)
        F = -dEdx
        assert _rel_err(F_fd, F) < 1e-6


# ---------------------------------------------------------------------------
# 3.2: DoubleWell1D(5.0) minima at x=+-1, barrier top at x=0, height 5.0
# ---------------------------------------------------------------------------

def test_double_well_1d_minima_and_barrier():
    pot = DoubleWell1D(barrier=5.0)

    E_plus, g_plus = pot.energy_grad(np.array([1.0]))
    E_minus, g_minus = pot.energy_grad(np.array([-1.0]))
    E_top, g_top = pot.energy_grad(np.array([0.0]))

    assert E_plus == pytest.approx(0.0, abs=1e-12)
    assert E_minus == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(g_plus, 0.0, atol=1e-12)
    assert np.allclose(g_minus, 0.0, atol=1e-12)

    assert E_top == pytest.approx(5.0)
    # x=0 is a stationary point (top of the barrier): gradient is zero there
    assert np.allclose(g_top, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3.3: MullerBrown() minima match the literature coordinates (verified by
# numerical minimization, not hard-coded), tolerance 1e-3.
# ---------------------------------------------------------------------------

_MB_LITERATURE_MINIMA = [
    (-0.558, 1.442),
    (0.623, 0.028),
    (-0.050, 0.467),
]


def test_muller_brown_minima_match_literature():
    pot = MullerBrown()

    def energy_only(xy):
        E, _ = pot.energy_grad(np.asarray(xy, dtype=float))
        return E

    def grad_only(xy):
        _, g = pot.energy_grad(np.asarray(xy, dtype=float))
        return g

    for x0, y0 in _MB_LITERATURE_MINIMA:
        result = minimize(energy_only, x0=[x0, y0], jac=grad_only, method="BFGS")
        # BFGS on this surface can report a "precision loss" status once it
        # has already converged to the minimum (the line search can't find
        # further improvement at double precision) -- so the acceptance
        # criterion is the actual converged position and near-zero gradient,
        # not scipy's own `success` flag.
        assert np.max(np.abs(result.jac)) < 1e-4
        assert result.x[0] == pytest.approx(x0, abs=1e-3)
        assert result.x[1] == pytest.approx(y0, abs=1e-3)


# ---------------------------------------------------------------------------
# ChannelDoubleWell2D: barrier heights per channel + wall height (ruling
# R2-adjacent test the brief explicitly asks for: "add a small test that
# measures barrier heights along y=+1 and y=-1 and the wall height at
# x=+-1, y=0").
# ---------------------------------------------------------------------------

def test_channel_double_well_barrier_and_wall_heights():
    pot = ChannelDoubleWell2D(barrier_plus=4.0, barrier_minus=9.0, wall=50.0)

    def E(x, y):
        val, _ = pot.energy_grad(np.array([x, y]))
        return val

    # x-barrier height along y=+1 (top of barrier at x=0 minus well floor at x=1)
    barrier_top_plus = E(0.0, 1.0)
    well_plus = E(1.0, 1.0)
    assert (barrier_top_plus - well_plus) == pytest.approx(4.0, rel=1e-3)

    # x-barrier height along y=-1
    barrier_top_minus = E(0.0, -1.0)
    well_minus = E(-1.0, -1.0)
    assert (barrier_top_minus - well_minus) == pytest.approx(9.0, rel=1e-3)

    # wall height at x=+-1, y=0 (x-double-well term vanishes there, so this
    # isolates the wall term exactly)
    assert E(1.0, 0.0) == pytest.approx(50.0, rel=1e-9)
    assert E(-1.0, 0.0) == pytest.approx(50.0, rel=1e-9)

    # the wall must be tall compared to the channel barriers so the two
    # channels cannot mix at accessible energies
    assert E(1.0, 0.0) > 3.0 * max(4.0, 9.0)


# ---------------------------------------------------------------------------
# AnalyticBackend: energy_forces() sign convention and provenance
# ---------------------------------------------------------------------------

def test_analytic_backend_energy_forces_matches_potential_negated_gradient():
    pot = DoubleWell2D(barrier=3.0, ky=1.0)
    backend = AnalyticBackend(
        potential=pot, integrator="baoab", dt=0.001, kT=1.0, gamma=1.0
    )
    x = np.array([0.3, -0.7])
    E_expected, g_expected = pot.energy_grad(x)
    E, F = backend.energy_forces(x)
    assert E == pytest.approx(E_expected)
    assert np.allclose(F, -g_expected)


def test_analytic_backend_build_returns_working_propagator():
    # Superseded by task-4-brief.md's tests 4.1-4.5 (tests/test_analytic_
    # dynamics.py), which exercise build()'s actual overdamped/baoab
    # dynamics in depth. This is just a smoke test that build() no longer
    # raises NotImplementedError (as it did in Task 3) and returns
    # something implementing the Propagator protocol.
    from cytherea.backends.base import MDState
    from cytherea.keys import ShotKey

    pot = FreeParticle(dim=1)
    backend = AnalyticBackend(
        potential=pot, integrator="overdamped", dt=0.001, kT=1.0, gamma=1.0
    )
    key = ShotKey(global_seed=0, frame_id=0, shot_id=0, stage="smoke")
    state = MDState(x=np.zeros(1), v=np.zeros(1), t=0.0)
    prop = backend.build(state, None, key)
    prop.run(3)
    out = prop.get_state()
    assert isinstance(out, MDState)
    assert out.t == 3 * 0.001  # t = step_index * dt (contract K1)
    prop.set_state(state)
    assert prop.get_state().t == 0.0


def test_analytic_backend_provenance_identifies_potential_and_params():
    pot = Harmonic(k=2.0, dim=3)
    backend = AnalyticBackend(
        potential=pot, integrator="baoab", dt=0.002, kT=0.5, gamma=0.1
    )
    prov = backend.provenance()
    assert prov["kind"] == "analytic"
    assert "Harmonic" in prov["potential"]
    assert prov["dt"] == 0.002
    assert prov["kT"] == 0.5


# ===========================================================================
# Fix wave 2026-10-01 (fullreview B-analytic Minors 9-11, contract K8).
# ===========================================================================

import json  # noqa: E402
import warnings  # noqa: E402


# --- Minor 11: point values pin each formula (FD self-consistency alone
# cannot see a dropped 1/2, a wrong x0 scaling, or an ignored `scale`) -----


def test_point_values_double_well_1d_with_x0():
    pot = DoubleWell1D(barrier=2.0, x0=1.5)
    # V = B*((x/x0)^2 - 1)^2 ; x=3: (4-1)^2 = 9 -> 18 ; minima at +-x0
    assert pot.energy_grad(np.array([3.0]))[0] == pytest.approx(18.0)
    assert pot.energy_grad(np.array([1.5]))[0] == pytest.approx(0.0, abs=1e-14)
    assert pot.energy_grad(np.array([0.0]))[0] == pytest.approx(2.0)
    # dV/dx = 4*B*x*u/x0^2 at x=3: 4*2*3*3/2.25 = 32
    assert pot.energy_grad(np.array([3.0]))[1][0] == pytest.approx(32.0)


def test_point_values_double_well_2d():
    pot = DoubleWell2D(barrier=3.0, ky=2.0)
    # 3*(4-1)^2 + 0.5*2*9 = 27 + 9
    E, g = pot.energy_grad(np.array([2.0, 3.0]))
    assert E == pytest.approx(36.0)
    assert np.allclose(g, [4 * 3.0 * 2.0 * 3.0, 2.0 * 3.0])


def test_point_values_harmonic():
    E, g = Harmonic(k=4.0, dim=2).energy_grad(np.array([1.0, -2.0]))
    assert E == pytest.approx(0.5 * 4.0 * 5.0)
    assert np.allclose(g, [4.0, -8.0])


def test_point_values_muller_brown_and_scale():
    # Literature value at the deepest minimum (-0.558, 1.442): V ~ -146.70.
    x = np.array([-0.558224, 1.441726])
    E1, _ = MullerBrown().energy_grad(x)
    assert E1 == pytest.approx(-146.6995, abs=1e-3)
    E2, g2 = MullerBrown(scale=0.1).energy_grad(np.array([0.3, 0.7]))
    E3, g3 = MullerBrown().energy_grad(np.array([0.3, 0.7]))
    assert E2 == pytest.approx(0.1 * E3)
    assert np.allclose(g2, 0.1 * g3)


def test_point_values_lj_pair():
    lj = LJCluster(n_atoms=2, epsilon=1.7, sigma=0.8)
    at_sigma = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    assert lj.energy_grad(at_sigma)[0] == pytest.approx(0.0, abs=1e-12)
    rmin = 2.0 ** (1.0 / 6.0) * 0.8
    at_min = np.array([[0.0, 0.0, 0.0], [0.0, rmin, 0.0]])
    E, g = lj.energy_grad(at_min)
    assert E == pytest.approx(-1.7)
    assert np.allclose(g, 0.0, atol=1e-12)


def test_point_values_channel_double_well():
    pot = ChannelDoubleWell2D(barrier_plus=4.0, barrier_minus=8.0, wall=50.0)
    # far inside the y>0 channel: B -> barrier_plus, W -> 0, no confinement
    E, _ = pot.energy_grad(np.array([0.0, 1.5]))
    sig = 1.0 / (1.0 + np.exp(-15.0))
    B = 8.0 + (4.0 - 8.0) * sig
    assert E == pytest.approx(B + 50.0 * np.exp(-((1.5 / 0.25) ** 2)))
    # confinement: 0.5*k*(|y|-2)^2 beyond |y| = 2
    E3, g3 = pot.energy_grad(np.array([1.0, -3.0]))
    assert E3 == pytest.approx(0.5 * 10.0 * 1.0, rel=1e-9)
    assert g3[1] == pytest.approx(-10.0, rel=1e-9)


# --- Minor 9: ChannelDoubleWell2D confinement / overflow / widths ---------


def test_channel_double_well_has_no_overflow_far_out_and_is_confined():
    pot = ChannelDoubleWell2D(barrier_plus=4.0, barrier_minus=8.0, wall=50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # probe2.py P5: RuntimeWarning at y=-80
        E, g = pot.energy_grad(np.array([0.5, -80.0]))
    assert np.isfinite(E) and np.all(np.isfinite(g))
    # restoring force in y beyond the confinement radius (was ~0: free y)
    assert pot.energy_grad(np.array([1.0, 3.0]))[1][1] > 1.0
    assert pot.energy_grad(np.array([1.0, -3.0]))[1][1] < -1.0
    # and nothing changes inside |y| <= y_confine
    assert pot.energy_grad(np.array([1.0, 1.9]))[1][1] == pytest.approx(
        0.0, abs=1e-12
    )


def test_channel_double_well_widths_are_in_provenance():
    backend = AnalyticBackend(
        potential=ChannelDoubleWell2D(4.0, 8.0, 50.0),
        integrator="baoab", dt=0.01, kT=1.0, gamma=1.0,
    )
    params = backend.provenance()["potential_params"]
    assert params["transition_width"] == 0.1
    assert params["wall_width"] == 0.25
    assert params["y_confine"] == 2.0 and params["k_confine"] == 10.0


# --- Minor 10 + K8: provenance for any Potential; effective_config ---------


class _PlainPotential:
    """Not a dataclass -- the Potential protocol only needs energy_grad."""

    def __init__(self, k):
        self.k = k

    def energy_grad(self, x):
        x = np.asarray(x, float)
        return 0.5 * self.k * float(np.sum(x**2)), self.k * x


class _SlotsPotential:
    __slots__ = ()

    def energy_grad(self, x):
        x = np.asarray(x, float)
        return 0.0, np.zeros_like(x)


def test_provenance_works_for_non_dataclass_potentials():
    b = AnalyticBackend(_PlainPotential(3.0), "baoab", 0.01, 1.0, 1.0)
    prov = b.provenance()
    assert prov["potential"] == "_PlainPotential"
    assert prov["potential_params"] == {"k": 3.0}
    b2 = AnalyticBackend(_SlotsPotential(), "baoab", 0.01, 1.0, 1.0)
    # no instance state: {} (K9 -- never a repr with a memory address)
    assert b2.provenance()["potential_params"] == {}


def test_provenance_accepts_cfg_argument():
    b = AnalyticBackend(Harmonic(1.0, 1), "baoab", 0.01, 1.0, 1.0)
    assert b.provenance(None) == b.provenance()
    # the analytic backend ignores cfg on both K8 methods (documented)
    from cytherea.backends.base import PhysicsConfig

    cfg = PhysicsConfig(
        integrator="verlet", dt_ps=0.002, temperature_K=300.0,
        friction_per_ps=0.0, constraints="none", rigid_water=False,
        platform="Reference", precision="double", deterministic_forces=True,
        purpose="measurement",
    )
    assert b.provenance(cfg) == b.provenance()
    assert b.effective_config(cfg) == b.effective_config(None) == b.effective_config()


def test_effective_config_is_plain_and_reflects_dynamics_parameters():
    def make(**kw):
        args = dict(integrator="baoab", dt=0.01, kT=0.7, gamma=0.5, mass=2.0)
        args.update(kw)
        return AnalyticBackend(potential=Harmonic(k=2.0, dim=3), **args)

    cfg = make().effective_config()
    assert json.loads(json.dumps(cfg)) == cfg  # plain / JSON-serialisable
    assert cfg == {
        "backend": "analytic",
        "units": "reduced",
        "potential": "Harmonic",
        "potential_params": {"k": 2.0, "dim": 3},
        "integrator": "baoab",
        "dt": 0.01,
        "kT": 0.7,
        "gamma": 0.5,
        "mass": 2.0,
    }
    assert make().effective_config() == cfg
    for kw in (dict(gamma=0.0), dict(dt=0.02), dict(kT=1.0), dict(mass=1.0),
               dict(integrator="overdamped")):
        assert make(**kw).effective_config() != cfg, kw
    other_pot = AnalyticBackend(Harmonic(k=3.0, dim=3), "baoab", 0.01, 0.7, 0.5, 2.0)
    assert other_pot.effective_config() != cfg


# ===========================================================================
# Fix wave 2, package L1 (contract K9; fixreview-p4 m2): non-dataclass
# potentials get a stable, complete parameter description -- or an error.
# ===========================================================================

import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

from cytherea.store import config_hash  # noqa: E402


class _PrivK:
    """Parameters only in private attributes."""

    def __init__(self, k):
        self._k = k

    def energy_grad(self, x):
        x = np.asarray(x, float)
        return 0.5 * self._k * float(x @ x), self._k * x


class _WithLambda:
    def __init__(self):
        self.f = lambda x: x

    def energy_grad(self, x):
        return 0.0, np.zeros_like(np.asarray(x, float))


class _Opaque:
    """An attribute that is neither plain data nor describable."""

    def __init__(self):
        self.handle = object()

    def energy_grad(self, x):
        return 0.0, np.zeros_like(np.asarray(x, float))


def _b(pot):
    return AnalyticBackend(pot, "overdamped", 0.01, 1.0, 1.0)


def test_k9_private_parameters_enter_the_hash():
    a = _b(_PrivK(1.0)).effective_config()
    b = _b(_PrivK(50.0)).effective_config()
    assert config_hash(a) != config_hash(b)
    assert a["potential_params"] == {"_k": 1.0}


@pytest.mark.parametrize("pot", [_WithLambda(), _Opaque()])
def test_k9_unrepresentable_potential_raises_instead_of_repr(pot):
    with pytest.raises(TypeError, match="params()"):
        _b(pot).effective_config()


def test_k9_params_hook_is_used_verbatim():
    class _Hooked:
        def __init__(self, k):
            self._k = k
            self._cache = {}  # call-history state: excluded by the hook

        def params(self):
            return {"k": self._k}

        def energy_grad(self, x):
            x = np.asarray(x, float)
            self._cache[len(self._cache)] = 1
            return 0.5 * self._k * float(x @ x), self._k * x

    b = _b(_Hooked(2.0))
    before = b.effective_config()
    b.energy_forces(np.ones(2))
    assert b.effective_config() == before
    assert before["potential_params"] == {"k": 2.0}


_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from cytherea.backends.analytic import AnalyticBackend, Harmonic
    from cytherea.store import config_hash

    class Inner:
        def __init__(self):
            self.k = 2.0
    class Composite:
        def __init__(self):
            self.inner = Inner()
            self.sub = Harmonic(1.0, 2)
            self._centres = np.arange(200.0)
            self.names = ("a", "b")
        def energy_grad(self, x):
            return 0.0, np.zeros_like(x)
    b = AnalyticBackend(Composite(), "baoab", 0.01, 1.0, 0.1)
    print(config_hash(b.effective_config()))
    """
)


def test_k9_effective_config_hash_is_identical_across_processes():
    """No memory address (repr) may reach the hash: two interpreter runs of
    the same construction give the same digest."""
    digests = {
        subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_SCRIPT], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(digests) == 1 and len(next(iter(digests))) == 64
