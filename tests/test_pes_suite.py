"""Tests for cytherea.backends.pes_suite.pes_consistency_suite (task-3-brief.md
"必测用例" 3.4-3.6, plus determinism / floor-relative-error edge cases and the
NVE total-energy drift check the controller rulings called out).

Note on the NVE path (ruling R11): at the time this file was originally
written (Task 3), `AnalyticBackend.build()` still raised
`NotImplementedError`, so `pes_consistency_suite`'s `nve_steps>0` path
could only be exercised against a small test-local exact harmonic-
oscillator `Propagator` (closed-form x(t)=x0*cos(wt), v(t)=-x0*w*sin(wt))
wrapped in a fake backend -- kept below because it pins down the total-
mechanical-energy drift computation itself (PE + KE, not PE alone)
independently of any particular integrator. Task 4 has since implemented
`AnalyticBackend.build()` for real (overdamped/baoab); see
`test_nve_steps_positive_calls_build_and_runs_now_that_task4_exists`
below, and `tests/test_analytic_dynamics.py` for the real BAOAB-gamma=0
NVE check (task-4-brief.md test 4.5).
"""

from __future__ import annotations

import numpy as np
import pytest

from cytherea.backends.analytic import (
    AnalyticBackend,
    DoubleWell2D,
    Harmonic,
    LJCluster,
)
from cytherea.backends.base import MDState
from cytherea.backends.pes_suite import PESReport, pes_consistency_suite


def _make_backend(potential=None):
    potential = potential if potential is not None else DoubleWell2D(barrier=3.0, ky=2.0)
    return AnalyticBackend(
        potential=potential, integrator="baoab", dt=0.001, kT=1.0, gamma=1.0
    )


def _probes_2d(n=20, seed=7):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-3, 3, size=(2,)) for _ in range(n)]


# ---------------------------------------------------------------------------
# 3.4: a deliberately-broken backend (forces * 1.01) must fail the suite.
# ---------------------------------------------------------------------------

class _BrokenBackend:
    """Wraps a real backend but multiplies reported forces by 1.01."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, inner):
        self._inner = inner

    def build(self, s, cfg, rng_key):
        return self._inner.build(s, cfg, rng_key)

    def energy_forces(self, x):
        E, F = self._inner.energy_forces(x)
        return E, F * 1.01

    def effective_config(self, cfg=None):
        return self._inner.effective_config(cfg)

    def provenance(self, cfg=None):
        return self._inner.provenance(cfg)


def test_broken_backend_fails_pes_consistency_suite():
    good = _make_backend()
    broken = _BrokenBackend(good)
    probes = _probes_2d()

    good_report = pes_consistency_suite(good, probes)
    broken_report = pes_consistency_suite(broken, probes)

    assert good_report.passed is True
    assert broken_report.passed is False
    # the 1% force error should show up as an fd relative error well above
    # the default rtol (1e-4), not get lost in floor/noise
    assert broken_report.fd_max_rel_err > 1e-3


def test_good_backend_passes_pes_consistency_suite():
    backend = _make_backend()
    report = pes_consistency_suite(backend, _probes_2d())
    assert isinstance(report, PESReport)
    assert report.passed is True
    assert report.fd_max_rel_err < 1e-4
    assert report.repeat_bitwise is True
    assert report.translation_err is None
    assert report.rotation_err is None
    assert report.nve_rel_drift is None


# ---------------------------------------------------------------------------
# 3.5: repeated calls to energy_forces on the same x are bitwise identical.
# ---------------------------------------------------------------------------

def test_repeat_bitwise_true_for_deterministic_backend():
    backend = _make_backend()
    report = pes_consistency_suite(backend, _probes_2d(n=5))
    assert report.repeat_bitwise is True


def test_repeat_bitwise_false_when_backend_is_nondeterministic():
    class _FlakyBackend:
        kind = "analytic"
        gpu_resident = False

        def __init__(self, inner):
            self._inner = inner
            self._calls = 0

        def build(self, s, cfg, rng_key):
            return self._inner.build(s, cfg, rng_key)

        def energy_forces(self, x):
            E, F = self._inner.energy_forces(x)
            self._calls += 1
            # perturb every second call so repeated calls on the same x
            # disagree
            if self._calls % 2 == 0:
                F = F + 1e-3
            return E, F

        def effective_config(self, cfg=None):
            return self._inner.effective_config(cfg)

        def provenance(self, cfg=None):
            return self._inner.provenance(cfg)

    backend = _FlakyBackend(_make_backend())
    report = pes_consistency_suite(backend, _probes_2d(n=3))
    assert report.repeat_bitwise is False
    assert report.passed is False


# ---------------------------------------------------------------------------
# 3.6: check_invariance=True with a multi-atom LJ toy: translation and
# rotation invariance errors < 1e-10.
# ---------------------------------------------------------------------------

def _lj_probes(n=6, seed=3):
    rng = np.random.default_rng(seed)
    probes = []
    base = np.array([[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0.5, 0.5, 1.1]], dtype=float)
    for _ in range(n):
        probes.append(base + rng.uniform(-0.05, 0.05, size=base.shape))
    return probes


def test_lj_cluster_translation_and_rotation_invariance():
    backend = _make_backend(LJCluster(n_atoms=4, epsilon=1.0, sigma=1.0))
    report = pes_consistency_suite(
        backend, _lj_probes(), check_invariance=True
    )
    assert report.translation_err is not None
    assert report.rotation_err is not None
    assert report.translation_err < 1e-10
    assert report.rotation_err < 1e-10
    assert report.passed is True


def test_invariance_errors_are_relative_to_energy_and_force_scale():
    """Ruling R22 #4: translation/rotation errors are relative
    (|dE|/|E0|, max|dF|/max|F0|), so scaling the whole potential by 1e8
    leaves them at roundoff level instead of blowing past an absolute 1e-10."""
    backend = _make_backend(LJCluster(n_atoms=4, epsilon=1e8, sigma=1.0))
    report = pes_consistency_suite(backend, _lj_probes(), check_invariance=True)
    assert report.translation_err < 1e-10
    assert report.rotation_err < 1e-10
    assert report.passed is True
    strict = pes_consistency_suite(
        backend, _lj_probes(), check_invariance=True, inv_rtol=0.0
    )
    assert strict.passed is False or (
        strict.translation_err == 0.0 and strict.rotation_err == 0.0
    )


class _AnisotropicallyBrokenBackend:
    """Scales only the x-component of every reported force. A *uniform*
    force scaling (see `_BrokenBackend`) actually stays rotation-equivariant
    (scaling commutes with rotation), so it doesn't exercise the rotation
    check; an axis-dependent scaling does not commute with an arbitrary
    rotation, so it is a genuine rotation-covariance violation while still
    leaving translation invariance intact (the scaling doesn't depend on
    position at all).
    """

    kind = "analytic"
    gpu_resident = False

    def __init__(self, inner):
        self._inner = inner

    def build(self, s, cfg, rng_key):
        return self._inner.build(s, cfg, rng_key)

    def energy_forces(self, x):
        E, F = self._inner.energy_forces(x)
        F = F.copy()
        F[..., 0] *= 1.05
        return E, F

    def effective_config(self, cfg=None):
        return self._inner.effective_config(cfg)

    def provenance(self, cfg=None):
        return self._inner.provenance(cfg)


def test_check_invariance_detects_broken_backend():
    good = _make_backend(LJCluster(n_atoms=4, epsilon=1.0, sigma=1.0))
    broken = _AnisotropicallyBrokenBackend(good)
    report = pes_consistency_suite(broken, _lj_probes(), check_invariance=True)
    assert report.passed is False
    # translation invariance is untouched by an axis-dependent force scale...
    assert report.translation_err < 1e-10
    # ...but rotation covariance is broken (scaling one axis doesn't commute
    # with an arbitrary rotation)
    assert report.rotation_err > 1e-6


# ---------------------------------------------------------------------------
# nve_steps: the ==0 path must not touch backend.build at all; the >0 path
# must call it, requires `masses`, and must compare *total* mechanical
# energy (PE + KE), not potential energy alone (fix round 1 / ruling R11).
# ---------------------------------------------------------------------------

def test_nve_steps_zero_never_calls_build():
    calls = []

    class _NoBuildBackend:
        kind = "analytic"
        gpu_resident = False

        def __init__(self, inner):
            self._inner = inner

        def build(self, s, cfg, rng_key):
            calls.append((s, cfg, rng_key))
            raise AssertionError("build() must not be called when nve_steps=0")

        def energy_forces(self, x):
            return self._inner.energy_forces(x)

        def effective_config(self, cfg=None):
            return self._inner.effective_config(cfg)

        def provenance(self, cfg=None):
            return self._inner.provenance(cfg)

    backend = _NoBuildBackend(_make_backend())
    report = pes_consistency_suite(backend, _probes_2d(n=3), nve_steps=0)
    assert calls == []
    assert report.nve_rel_drift is None


def test_nve_steps_positive_calls_build_and_runs_now_that_task4_exists():
    # Task 3 pinned this down as "surfaces NotImplementedError" because
    # AnalyticBackend.build() didn't exist yet. Task 4 implemented it, so
    # this now runs for real: _make_backend() uses integrator="baoab" with
    # gamma=1.0 (non-zero friction, i.e. genuine Langevin, not NVE), so the
    # resulting nve_rel_drift need not be tiny -- this only checks that the
    # path executes end-to-end and returns a real number.
    backend = _make_backend()
    report = pes_consistency_suite(
        backend, _probes_2d(n=1), nve_steps=10, masses=np.array([1.0, 1.0]),
        kT=1.0,
    )
    assert report.nve_rel_drift is not None
    assert np.isfinite(report.nve_rel_drift)
    # p4 m1: the absolute drift in kT is reported next to the relative one
    assert report.nve_abs_drift_kT == pytest.approx(report.nve_rel_drift * report.n_dof / 2.0)


def test_nve_steps_positive_without_masses_raises_value_error():
    backend = _make_backend()
    with pytest.raises(ValueError, match="masses"):
        pes_consistency_suite(backend, _probes_2d(n=1), nve_steps=10)


# --- test-local exact harmonic-oscillator propagator, used to exercise the
# NVE total-energy drift computation itself, independent of Task 4's real
# (not-yet-implemented) AnalyticBackend propagators.

class _ExactHarmonicPropagator:
    """Exact closed-form solution for a 1D harmonic oscillator started at
    (x0, v0): x(t) = x0*cos(w*t) + (v0/w)*sin(w*t), v = dx/dt, w = sqrt(k/m).
    A correct energy-conserving propagator, so PE+KE is exactly constant
    (up to floating-point rounding) along this trajectory.
    """

    def __init__(self, x0: np.ndarray, m: float, k: float, dt: float, v0=None):
        self.x0 = float(x0[0])
        self.v0 = 0.0 if v0 is None else float(np.asarray(v0)[0])
        self.m = m
        self.omega = np.sqrt(k / m)
        self.dt = dt
        self.t = 0.0

    def run(self, n_steps: int) -> None:
        self.t += n_steps * self.dt

    def get_state(self) -> MDState:
        w, t = self.omega, self.t
        x = self.x0 * np.cos(w * t) + self.v0 / w * np.sin(w * t)
        v = -self.x0 * w * np.sin(w * t) + self.v0 * np.cos(w * t)
        return MDState(x=np.array([x]), v=np.array([v]), t=self.t)

    def set_state(self, s: MDState) -> None:
        self.t = s.t


class _EnergyInjectingHarmonicPropagator(_ExactHarmonicPropagator):
    """Same exact x(t) as `_ExactHarmonicPropagator`, but scales the reported
    velocity up by 1.01 every step it runs -- injecting kinetic energy the
    real dynamics never had, without touching positions. Stands in for a
    buggy integrator (e.g. a sign or units error in a thermostat) that a
    PE-only drift check could never catch.
    """

    def __init__(self, x0: np.ndarray, m: float, k: float, dt: float, v0=None):
        super().__init__(x0, m, k, dt, v0)
        self._v_scale = 1.0

    def run(self, n_steps: int) -> None:
        super().run(n_steps)
        self._v_scale *= 1.01**n_steps

    def get_state(self) -> MDState:
        s = super().get_state()
        return MDState(x=s.x, v=s.v * self._v_scale, t=s.t)


class _HarmonicOscillatorBackend:
    """Fake PotentialBackend around Harmonic(k, dim=1) whose build() returns
    one of the two propagators above instead of raising NotImplementedError
    (unlike the real AnalyticBackend, which is Task 4's job).
    """

    kind = "analytic"
    gpu_resident = False

    def __init__(self, k: float, m: float, dt: float, propagator_cls, offset=0.0):
        self.potential = Harmonic(k=k, dim=1)
        self.offset = offset
        self.k = k
        self.m = m
        self.dt = dt
        self._propagator_cls = propagator_cls

    def build(self, s, cfg, rng_key):
        return self._propagator_cls(x0=s.x, m=self.m, k=self.k, dt=self.dt, v0=s.v)

    def energy_forces(self, x):
        E, dEdx = self.potential.energy_grad(np.asarray(x, dtype=float))
        return E + self.offset, -dEdx

    def effective_config(self, cfg=None):
        return {"backend": "fake-harmonic", "k": self.k, "m": self.m, "dt": self.dt}

    def provenance(self, cfg=None):
        return {"kind": self.kind, "potential": "Harmonic", "k": self.k}


def test_nve_exact_energy_conserving_propagator_passes_with_tiny_drift():
    backend = _HarmonicOscillatorBackend(
        k=3.0, m=2.0, dt=0.01, propagator_cls=_ExactHarmonicPropagator
    )
    report = pes_consistency_suite(
        backend,
        probes=[np.array([1.5])],
        nve_steps=50,
        masses=np.array([2.0]),
        kT=0.8,
    )
    assert report.nve_rel_drift is not None
    assert report.nve_rel_drift < 1e-12
    assert report.passed is True


def test_nve_energy_injecting_propagator_fails():
    backend = _HarmonicOscillatorBackend(
        k=3.0, m=2.0, dt=0.01, propagator_cls=_EnergyInjectingHarmonicPropagator
    )
    report = pes_consistency_suite(
        backend,
        probes=[np.array([1.5])],
        nve_steps=50,
        masses=np.array([2.0]),
        kT=0.8,
    )
    assert report.nve_rel_drift is not None
    assert report.nve_rel_drift > 1e-4
    assert report.passed is False


# ---------------------------------------------------------------------------
# floor in the relative-error definition: a potential with exactly zero
# force everywhere (FreeParticle) must not blow up fd_max_rel_err.
# ---------------------------------------------------------------------------

def test_free_particle_zero_force_does_not_blow_up_relative_error():
    from cytherea.backends.analytic import FreeParticle

    backend = _make_backend(FreeParticle(dim=3))
    probes = [np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.0, 0.0])]
    report = pes_consistency_suite(backend, probes)
    assert report.fd_max_rel_err == pytest.approx(0.0, abs=1e-12)
    assert report.passed is True


# ===========================================================================
# Fix wave 2026-10-01 (fullreview B-analytic I1-I3 + Minors, spec S2).
# ===========================================================================

import hashlib  # noqa: E402
import warnings  # noqa: E402

from cytherea.backends import pes_suite as _ps  # noqa: E402
from cytherea.backends.pes_suite import DEFAULT_TOLERANCES  # noqa: E402
from cytherea.keys import derive_rng  # noqa: E402


class _Offset:
    """Potential wrapper adding a constant to the energy (gradient unchanged):
    the physics is identical, only the arbitrary energy zero moves."""

    def __init__(self, inner, c):
        self.inner, self.c = inner, c

    def energy_grad(self, x):
        E, g = self.inner.energy_grad(x)
        return E + self.c, g


class _InjectingBackend(AnalyticBackend):
    """BAOAB (gamma=0: velocity Verlet) whose propagator multiplies v by 1.001
    after every step -- an energy-injecting integrator bug."""

    def build(self, s, cfg, key):
        p = super().build(s, cfg, key)
        orig = p._run_baoab

        def run(n):
            for _ in range(n):
                orig(1)
                p._v = p._v * 1.001

        p.run = run
        return p


_LJ_RMIN = 2.0 ** (1.0 / 6.0)
_LJ_TRIANGLE = np.array(
    [[0, 0, 0], [_LJ_RMIN, 0, 0], [_LJ_RMIN / 2, _LJ_RMIN * np.sqrt(3) / 2, 0]],
    dtype=float,
) + np.array([0.3, -0.2, 0.1])


# --- I1: NaN / inf anywhere must fail -------------------------------------


class _NaNAfterFirstChunkPropagator:
    """Correct at t=0, NaN after the first run() (a blown-up integrator)."""

    dt = 0.01

    def __init__(self, s):
        self.s, self.n = s, 0

    def run(self, n):
        self.n += n

    def get_state(self):
        if self.n == 0:
            return MDState(x=self.s.x, v=self.s.v, t=0.0)
        nan = np.full_like(self.s.x, np.nan)
        return MDState(x=nan, v=nan, t=self.n * self.dt)

    def set_state(self, s):
        self.s = s


class _NaNNVEBackend:
    kind = "analytic"
    gpu_resident = False

    def __init__(self):
        self.pot = Harmonic(1.0, 1)

    def build(self, s, cfg, key):
        return _NaNAfterFirstChunkPropagator(s)

    def energy_forces(self, x, box=None):
        E, g = self.pot.energy_grad(np.asarray(x, float))
        return E, -g

    def effective_config(self, cfg=None):
        return {"backend": "fake_NaNNVEBackend"}

    def provenance(self, cfg=None):
        return {"kind": self.kind}


def test_nve_nan_blowup_after_first_chunk_fails():
    # probe_nan_and_scale.py P1: before the fix passed=True, drift=0.0.
    r = pes_consistency_suite(
        _NaNNVEBackend(), [np.array([1.0])], nve_steps=100, masses=1.0, kT=1.0
    )
    assert r.passed is False
    assert r.all_finite is False
    assert np.isnan(r.nve_rel_drift)
    assert any(x.startswith("nve_nonfinite") for x in r.reasons)


class _NaNAtSecondProbeTranslated(AnalyticBackend):
    def energy_forces(self, x, box=None):
        x = np.asarray(x, float)
        E, F = super().energy_forces(x)
        if abs(x[0, 0] - (0.01 + _ps._TRANSLATION[0])) < 1e-9:
            return float("nan"), F * np.nan
        return E, F


def test_invariance_nan_at_second_probe_translated_fails():
    # probe2.py P1b': before the fix passed=True, translation_err=4.9e-15.
    base = np.array(
        [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0.5, 0.5, 1.1]], dtype=float
    )
    b = _NaNAtSecondProbeTranslated(LJCluster(4), "baoab", 0.001, 1.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = pes_consistency_suite(b, [base, base + 0.01], check_invariance=True)
    assert r.passed is False
    assert r.all_finite is False
    assert np.isnan(r.translation_err)
    assert r.reasons[0].startswith("nonfinite")


class _NaNAtOneFDDisplacement(AnalyticBackend):
    def energy_forces(self, x, box=None):
        x = np.asarray(x, float)
        E, F = super().energy_forces(x)
        if x[1] > 0.5 + 0.9e-5:  # only the +h displacement of coordinate 1
            return float("nan"), F
        return E, F


def test_nan_at_a_single_fd_displacement_fails_as_nonfinite():
    b = _NaNAtOneFDDisplacement(DoubleWell2D(3.0, 2.0), "baoab", 0.001, 1.0, 1.0)
    r = pes_consistency_suite(b, [np.array([0.2, 0.5])])
    assert r.passed is False
    assert r.all_finite is False
    assert r.reasons[0].startswith("nonfinite")
    assert r.repeat_bitwise is True  # not misattributed to repeatability


def test_inf_force_at_a_probe_fails():
    class _InfForce(AnalyticBackend):
        def energy_forces(self, x, box=None):
            E, F = super().energy_forces(x)
            if np.asarray(x)[0] > 2.0:
                F = F.copy()
                F[0] = np.inf
            return E, F

    b = _InfForce(DoubleWell2D(3.0, 2.0), "baoab", 0.001, 1.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = pes_consistency_suite(b, [np.array([0.2, 0.5]), np.array([2.5, 0.1])])
    assert r.passed is False and r.all_finite is False


# --- I2: energy-zero independence, thermal NVE start, stationary probes ----


def _ke_of_suite_mb_draw(x0, masses, kT):
    xi = derive_rng(_ps._PES_SUITE_KEY, "nve_v").standard_normal(size=x0.shape)
    v = xi * np.sqrt(kT / masses)
    return 0.5 * float(np.sum(masses * v**2))


def test_nve_drift_is_independent_of_energy_offset():
    # probe_nan_and_scale.py P2a: before the fix the same correct velocity-
    # Verlet trajectory gave drift 6.2e-4 / 4.2e+04 / 4.2e-10 for offsets
    # 0 / E_tot(0)~0 / 1e6. Now: identical drift, all pass.
    x0 = np.array([1.0, 0.5, -0.3])
    kT = 1.0
    e_tot0 = 0.5 * float(np.sum(x0**2)) + _ke_of_suite_mb_draw(x0, 1.0, kT)
    drifts = []
    for c in (0.0, -e_tot0, 1e6):
        be = AnalyticBackend(_Offset(Harmonic(1.0, 3), c), "baoab", 0.05, kT, 0.0)
        r = pes_consistency_suite(be, [x0], nve_steps=2000, masses=1.0, kT=kT)
        assert r.passed, (c, r.reasons)
        drifts.append(r.nve_rel_drift)
    assert drifts[0] > 0.0
    assert drifts[1] == pytest.approx(drifts[0], rel=1e-6)
    assert drifts[2] == pytest.approx(drifts[0], rel=1e-3)  # 1e6 costs ~1e-10 roundoff


def test_nve_energy_injecting_propagator_fails_despite_large_offset():
    # probe_nan_and_scale.py P2b: before the fix, offset 1e4 -> 1.4e-5, pass.
    for c in (0.0, 1e4):
        be = _InjectingBackend(_Offset(Harmonic(1.0, 3), c), "baoab", 0.05, 1.0, 0.0)
        r = pes_consistency_suite(
            be, [np.array([1.0, 0.5, -0.3])], nve_steps=200, masses=1.0, kT=1.0
        )
        assert r.passed is False, c
        assert r.nve_rel_drift > 0.1


def test_nve_energy_injecting_propagator_started_at_minimum_fails():
    # probe_nan_and_scale.py P2c: before the fix, at rest at the minimum the
    # drift was 0.0 and the injecting propagator passed.
    be = _InjectingBackend(Harmonic(1.0, 3), "baoab", 0.05, 1.0, 0.0)
    r = pes_consistency_suite(be, [np.zeros(3)], nve_steps=200, masses=1.0, kT=1.0)
    assert r.passed is False
    assert r.nve_rel_drift > 0.1


def test_nve_dissipative_propagator_fails():
    # probe_nan_and_scale.py P3: a strongly damped Langevin run measured as
    # "NVE" must not pass.
    be = AnalyticBackend(Harmonic(1.0, 3), "baoab", 0.05, 1e-12, 5.0)
    r = pes_consistency_suite(
        be, [np.array([1.0, 0.5, -0.3])], nve_steps=2000, masses=1.0, kT=1.0
    )
    assert r.passed is False
    assert r.nve_rel_drift > 0.1


def test_nve_requires_kT_or_explicit_state_with_kinetic_energy():
    be = AnalyticBackend(Harmonic(1.0, 3), "baoab", 0.05, 1.0, 0.0)
    x0 = np.array([1.0, 0.5, -0.3])
    with pytest.raises(ValueError, match="kT"):
        pes_consistency_suite(be, [x0], nve_steps=10, masses=1.0)
    with pytest.raises(ValueError, match="kT"):
        pes_consistency_suite(be, [x0], nve_steps=10, masses=1.0, kT=0.0)
    at_rest = MDState(x=x0, v=np.zeros(3), t=0.0)
    with pytest.raises(ValueError, match="rest"):
        pes_consistency_suite(be, [x0], nve_steps=10, masses=1.0, nve_state=at_rest)


def test_nve_state_supplied_derives_kT_from_its_kinetic_energy():
    be = AnalyticBackend(Harmonic(1.0, 3), "baoab", 0.01, 1.0, 0.0)
    x0 = np.array([1.0, 0.5, -0.3])
    v0 = np.array([0.3, -0.6, 0.9])  # KE = 0.63 -> kT = 2*0.63/3 = 0.42
    r = pes_consistency_suite(
        be, [x0], nve_steps=500, masses=1.0,
        nve_state=MDState(x=x0, v=v0, t=0.0),
    )
    assert r.kT == pytest.approx(0.42)
    assert r.n_dof == 3
    assert r.passed, r.reasons


class _RecordingPropagator:
    """Projects x[0], v[0] -> 0 at build (a 'constraint'), then is exact NVE
    for a free particle."""

    dt = 0.1

    def __init__(self, s):
        x = np.array(s.x, float)
        v = np.array(s.v, float)
        x[0] = 0.0
        v[0] = 0.0  # velocity constraint too
        self.x, self.v, self.t, self.box = x, v, 0.0, np.eye(3) * 7.0

    def run(self, n):
        self.x = self.x + n * self.dt * self.v
        self.t += n * self.dt

    def get_state(self):
        return MDState(x=self.x.copy(), v=self.v.copy(), t=self.t, box=self.box)

    def set_state(self, s):
        raise AssertionError


class _RecordingBackend:
    """E = sum(x[0]**2) * 100 (x[0] is 'constrained' to 0 by the propagator)
    + box[0,0] (so evaluating without the state's box would show up)."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self):
        self.build_cfgs = []
        self.boxes = []

    def build(self, s, cfg, key):
        self.build_cfgs.append(cfg)
        return _RecordingPropagator(s)

    def energy_forces(self, x, box=None):
        self.boxes.append(box)
        x = np.asarray(x, float)
        E = 100.0 * float(x[0] ** 2) + (0.0 if box is None else float(box[0, 0]))
        F = np.zeros_like(x)
        F[0] = -200.0 * x[0]
        return E, F

    def effective_config(self, cfg=None):
        return {"backend": "fake_RecordingBackend"}

    def provenance(self, cfg=None):
        return {"kind": self.kind}


def test_nve_uses_nve_cfg_projected_start_and_state_box():
    # Minor 5: nve_cfg is passed to build, E_tot(0) comes from get_state()
    # (after the build-time projection), and PE uses the state's box.
    be = _RecordingBackend()
    sentinel = object()
    r = pes_consistency_suite(
        be, [np.array([0.3, 1.0, 2.0])], nve_steps=5, masses=1.0, kT=1.0,
        nve_cfg=sentinel,
    )
    assert be.build_cfgs == [sentinel]
    assert any(b is not None and b[0, 0] == 7.0 for b in be.boxes)
    assert r.nve_rel_drift == pytest.approx(0.0, abs=1e-12), r.reasons


def test_correct_lj_at_minimum_among_other_probes_passes():
    # probe3.py: a correct LJ cluster at its exact minimum (max|F| = 4e-15)
    # failed FD (2.2) and rotation (1.3e-6) with the per-probe normalisation.
    # With the suite-wide force scale a minimised structure among ordinary
    # probes passes.
    b = AnalyticBackend(LJCluster(3), "baoab", 0.001, 1.0, 0.0)
    rng = np.random.default_rng(1)
    probes = [_LJ_TRIANGLE] + [
        _LJ_TRIANGLE + rng.uniform(-0.05, 0.05, size=(3, 3)) for _ in range(4)
    ]
    rep = pes_consistency_suite(b, probes, check_invariance=True)
    assert rep.passed, rep.reasons
    assert rep.rotation_err < 1e-10 and rep.translation_err < 1e-10


def test_stationary_only_probe_set_fails_loudly_as_unresolved():
    # With no force signal anywhere the suite cannot certify anything; it
    # must say so (fd_unresolved) instead of passing or failing silently.
    b = AnalyticBackend(LJCluster(3), "baoab", 0.001, 1.0, 0.0)
    rep = pes_consistency_suite(b, [_LJ_TRIANGLE])
    assert rep.passed is False
    assert any(x.startswith("fd_unresolved") for x in rep.reasons)
    assert rep.fd_max_rel_err == 0.0  # the forces themselves are not wrong


def test_invariance_and_fd_are_independent_of_energy_offset():
    probes = [
        np.array([[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0.5, 0.5, 1.1]], float)
        + d
        for d in (0.0, 0.02, -0.03)
    ]
    E0, _ = LJCluster(4).energy_grad(probes[0])
    results = []
    for c in (0.0, -E0, 1e6):
        b = AnalyticBackend(_Offset(LJCluster(4), c), "baoab", 0.001, 1.0, 0.0)
        rep = pes_consistency_suite(b, probes, check_invariance=True)
        assert rep.passed, (c, rep.reasons)
        results.append(rep)
    # E0 -> 0 used to blow the relative energy error up by 1/floor
    assert results[1].translation_err < 1e-10


def test_broken_force_detected_at_every_offset():
    for c in (0.0, 1e3):
        good = AnalyticBackend(_Offset(DoubleWell2D(3.0, 2.0), c), "baoab", 0.001, 1.0, 1.0)
        rep = pes_consistency_suite(_BrokenBackend(good), _probes_2d())
        assert rep.passed is False
        assert rep.fd_max_rel_err > 1e-3


# --- I3: sampled mode ------------------------------------------------------


class _CountingBackend:
    kind = "analytic"
    gpu_resident = False

    def __init__(self, inner, force_scale=1.0, noise=0.0, effective=None):
        self._inner = inner
        self.calls = 0
        self._force_scale = force_scale
        self._noise = noise
        self._effective = effective

    def build(self, s, cfg, rng_key):
        return self._inner.build(s, cfg, rng_key)

    def energy_forces(self, x, box=None):
        self.calls += 1
        x = np.asarray(x, float)
        E, F = self._inner.energy_forces(x)
        if self._noise:
            # deterministic "precision noise": a pure function of x, so the
            # backend stays bitwise repeatable, as a mixed-precision GPU
            # kernel with deterministic reductions would be
            h = hashlib.sha256(x.tobytes()).digest()
            u = int.from_bytes(h[:8], "little") / 2.0**64 - 0.5
            E = E + self._noise * u
        return E, F * self._force_scale

    def effective_config(self, cfg=None):
        if self._effective is None:
            raise AttributeError
        return self._effective

    def provenance(self, cfg=None):
        return self._inner.provenance(cfg)


def _lj_cluster_probes(n_atoms, n_probes=3, seed=5):
    rng = np.random.default_rng(seed)
    grid = np.array(
        [[i % 3, (i // 3) % 3, i // 9] for i in range(n_atoms)], float
    ) * 1.12
    return [grid + rng.uniform(-0.04, 0.04, size=grid.shape) for _ in range(n_probes)]


def test_sampled_mode_evaluates_only_n_fd_atoms():
    inner = AnalyticBackend(LJCluster(12), "baoab", 0.001, 1.0, 0.0)
    probes = _lj_cluster_probes(12)
    strict_b = _CountingBackend(inner)
    strict = pes_consistency_suite(strict_b, probes)
    sampled_b = _CountingBackend(inner)
    sampled = pes_consistency_suite(sampled_b, probes, mode="sampled", n_fd_atoms=2)
    assert strict.n_fd_coords == 3 * 12 * 3
    # n_fd_atoms uniform atoms + the _N_TOP_FORCE_ATOMS largest forces (L7, p4 I-1)
    assert sampled.n_fd_coords == 3 * sum(len(a) for a in sampled.fd_atoms)
    assert all(2 <= len(a) <= 2 + pes_mod._N_TOP_FORCE_ATOMS for a in sampled.fd_atoms)
    assert sampled.n_fd_coords < strict.n_fd_coords
    # 4 FD energies per coordinate (+-h, +-h/2) + first + repeat per probe
    assert strict_b.calls == 4 * strict.n_fd_coords + 2 * 3
    assert sampled_b.calls == 4 * sampled.n_fd_coords + 2 * 3
    assert strict.passed and sampled.passed, (strict.reasons, sampled.reasons)
    assert sampled.mode == "sampled"


def test_sampled_mode_subset_is_deterministic():
    inner = AnalyticBackend(LJCluster(12), "baoab", 0.001, 1.0, 0.0)
    probes = _lj_cluster_probes(12)
    r1 = pes_consistency_suite(inner, probes, mode="sampled", n_fd_atoms=3)
    r2 = pes_consistency_suite(inner, probes, mode="sampled", n_fd_atoms=3)
    assert r1 == r2


def test_sampled_mixed_passes_noisy_correct_backend_that_strict_rejects():
    # Energy noise ~ what mixed precision does to E: FD noise ~1e-3 of the
    # force scale. strict (double tolerances) must refuse (fd_unresolved);
    # sampled at mixed tolerances must pass a correct backend and still
    # reject a 5% force error.
    inner = AnalyticBackend(LJCluster(12), "baoab", 0.001, 1.0, 0.0)
    probes = _lj_cluster_probes(12)
    F_s = max(float(np.max(np.abs(inner.energy_forces(p)[1]))) for p in probes)
    noise = 1e-3 * F_s * 1e-5
    good = _CountingBackend(inner, noise=noise)
    strict = pes_consistency_suite(good, probes)
    assert strict.passed is False
    assert any(x.startswith("fd_unresolved") for x in strict.reasons)
    sampled = pes_consistency_suite(good, probes, mode="sampled", precision="mixed")
    assert sampled.passed, sampled.reasons
    assert sampled.tolerances == DEFAULT_TOLERANCES["mixed"]
    bad = _CountingBackend(inner, noise=noise, force_scale=1.05)
    r_bad = pes_consistency_suite(bad, probes, mode="sampled", precision="mixed")
    assert r_bad.passed is False
    assert r_bad.fd_max_rel_err > DEFAULT_TOLERANCES["mixed"]["fd_rtol"]


def test_sampled_precision_resolved_from_effective_config():
    inner = AnalyticBackend(LJCluster(4), "baoab", 0.001, 1.0, 0.0)
    probes = _lj_probes()
    for eff, expected in (
        ({"precision": "single", "platform": "CUDA"}, "single"),
        ({"precision": "mixed", "platform": "Reference"}, "double"),
        (None, "double"),  # unknown -> strictest row
    ):
        r = pes_consistency_suite(
            _CountingBackend(inner, effective=eff), probes, mode="sampled"
        )
        assert r.precision == expected
        assert r.tolerances == DEFAULT_TOLERANCES[expected]


def test_strict_mode_uses_double_row_and_rejects_sampled_only_options():
    b = _make_backend()
    r = pes_consistency_suite(b, _probes_2d(n=2))
    assert r.tolerances == DEFAULT_TOLERANCES["double"]
    assert DEFAULT_TOLERANCES["double"]["fd_rtol"] == 1e-4  # design §3.5
    with pytest.raises(ValueError, match="n_fd_atoms"):
        pes_consistency_suite(b, _probes_2d(n=2), n_fd_atoms=1)
    with pytest.raises(ValueError, match="strict"):
        pes_consistency_suite(b, _probes_2d(n=2), precision="mixed")
    with pytest.raises(ValueError, match="mode"):
        pes_consistency_suite(b, _probes_2d(n=2), mode="fast")


def test_explicit_tolerances_override_row():
    r = pes_consistency_suite(
        _make_backend(), _probes_2d(n=2), mode="sampled", precision="single",
        fd_rtol=1e-6, inv_rtol=1e-7, nve_rtol=1e-8,
    )
    assert r.tolerances == {"fd_rtol": 1e-6, "inv_rtol": 1e-7, "nve_rtol": 1e-8,
                            "repeat_rtol": DEFAULT_TOLERANCES["single"]["repeat_rtol"]}


def test_rtol_is_a_deprecated_alias_for_fd_rtol():
    with pytest.warns(DeprecationWarning):
        r = pes_consistency_suite(_make_backend(), _probes_2d(n=2), rtol=1e-6)
    assert r.tolerances["fd_rtol"] == 1e-6
    with pytest.raises(ValueError):
        pes_consistency_suite(_make_backend(), _probes_2d(n=2), rtol=1e-6, fd_rtol=1e-6)


# --- Minors ---------------------------------------------------------------


def test_masses_of_shape_n_atoms_is_rejected_as_ambiguous():
    # Minor 4: masses.shape == (3,) with x.shape == (3, 3) used to broadcast
    # atom j's mass onto Cartesian component j of every atom, silently.
    b = AnalyticBackend(LJCluster(3), "baoab", 0.001, 1.0, 0.0)
    x0 = _LJ_TRIANGLE.copy()
    x0[0, 0] += 0.05
    with pytest.raises(ValueError, match="masses"):
        pes_consistency_suite(
            b, [x0], nve_steps=10, masses=np.array([1.0, 2.0, 3.0]), kT=0.1
        )
    for m in (1.0, np.ones((3, 1)), np.ones((3, 3))):
        r = pes_consistency_suite(b, [x0], nve_steps=200, masses=m, kT=0.1)
        assert r.passed, r.reasons


def test_empty_probe_list_is_an_error():
    # Minor 6: used to pass vacuously (or IndexError with nve_steps>0).
    with pytest.raises(ValueError, match="probe"):
        pes_consistency_suite(_make_backend(), [])
    with pytest.raises(ValueError, match="probe"):
        pes_consistency_suite(_make_backend(), [], nve_steps=5, masses=1.0, kT=1.0)


class _StaleNeighbourListBackend:
    """Like an MD code that rebuilds its neighbour list / reorders atoms only
    every 7th call: the result depends on where the list was last built
    (``ref``), so it is identical for back-to-back calls at the same x but
    differs after other positions were visited -- history-dependent
    nondeterminism."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, inner):
        self._inner = inner
        self._ref = None
        self._calls = 0

    def build(self, s, cfg, rng_key):
        return self._inner.build(s, cfg, rng_key)

    def energy_forces(self, x, box=None):
        x = np.asarray(x, float)
        if self._calls % 7 == 0:
            self._ref = x.copy()
        self._calls += 1
        E, F = self._inner.energy_forces(x)
        return E, F + 1e-3 * float(np.sum(self._ref - x))

    def effective_config(self, cfg=None):
        return {"backend": "fake_StaleNeighbourListBackend"}

    def provenance(self, cfg=None):
        return {"kind": self.kind}


def test_repeat_check_sees_history_dependent_nondeterminism():
    # Minor 7: the old back-to-back pair (calls 6 and 7 here, same list)
    # could not see this; comparing against the first evaluation can.
    b = _StaleNeighbourListBackend(_make_backend())
    r = pes_consistency_suite(b, [np.array([0.2, 0.3])])
    assert r.repeat_bitwise is False
    assert r.passed is False
    assert any(x.startswith("repeat") for x in r.reasons)


def test_repeat_rtol_declares_a_tolerance():
    probes = [np.array([0.2, 0.3])]
    r = pes_consistency_suite(
        _StaleNeighbourListBackend(_make_backend()), probes, repeat_rtol=1e-6
    )
    assert r.repeat_bitwise is False
    assert 0.0 < r.repeat_max_rel_err < 1e-6
    assert r.passed, r.reasons
    r_tight = pes_consistency_suite(
        _StaleNeighbourListBackend(_make_backend()), probes, repeat_rtol=1e-12
    )
    assert r_tight.passed is False


class _StiffSoftHarmonic:
    """V = 0.5 * sum k_i x_i^2 with one stiff coordinate (k=400) and 99 soft
    ones (k=1)."""

    def __init__(self):
        self.k = np.ones(100)
        self.k[0] = 400.0

    def energy_grad(self, x):
        return 0.5 * float(np.sum(self.k * x**2)), self.k * x


class _SoftCoordinateBroken(AnalyticBackend):
    def energy_forces(self, x, box=None):
        E, F = super().energy_forces(x)
        F = F.copy()
        F[7] *= 1.01  # 1% error on one low-force coordinate
        return E, F


def test_per_atom_metric_catches_defect_on_low_force_atom():
    # Minor 8: 1% error on a coordinate whose force is 1/400 of the max is
    # 2.5e-5 of the global scale (passes a global 1e-4 gate); relative to
    # max(|F_i|, F_rms ~ 40) it is 2.5e-4 and must fail.
    b = _SoftCoordinateBroken(_StiffSoftHarmonic(), "baoab", 0.001, 1.0, 1.0)
    r = pes_consistency_suite(b, [np.ones(100)])
    assert r.fd_max_rel_err < 1e-4
    assert r.fd_max_atom_rel_err > 1e-4
    assert r.passed is False
    assert any(x.startswith("fd_atom") for x in r.reasons)
    good = AnalyticBackend(_StiffSoftHarmonic(), "baoab", 0.001, 1.0, 1.0)
    assert pes_consistency_suite(good, [np.ones(100)]).passed


def test_invariance_requires_atom_by_3_probes():
    with pytest.raises(ValueError, match="n_atoms, 3"):
        pes_consistency_suite(_make_backend(), _probes_2d(n=2), check_invariance=True)


class _CMMotionRemoverBackend(AnalyticBackend):
    """Velocity Verlet plus an OpenMM-style CMMotionRemover (subtracts the
    COM velocity after every step)."""

    def build(self, s, cfg, key):
        p = super().build(s, cfg, key)
        orig = p._run_baoab

        def run(n):
            for _ in range(n):
                orig(1)
                p._v = p._v - p._v.mean(axis=0)  # equal masses

        p.run = run
        return p


def test_nve_mb_draw_removes_com_velocity_for_atomic_systems():
    # Found running the suite on OpenMM ala2 (which has a CMMotionRemover):
    # an MB draw with COM momentum loses that KE at the first step, a
    # spurious 7% "drift". Default: COM removed, n_dof = 3N - 3.
    b = _CMMotionRemoverBackend(LJCluster(3), "baoab", 0.001, 1.0, 0.0)
    x0 = _LJ_TRIANGLE.copy()
    x0[0, 0] += 0.05
    r = pes_consistency_suite(b, [x0], nve_steps=200, masses=1.0, kT=0.1)
    assert r.n_dof == 6
    assert r.passed, r.reasons
    r_keep = pes_consistency_suite(
        b, [x0], nve_steps=200, masses=1.0, kT=0.1, remove_com=False
    )
    assert r_keep.n_dof == 9
    assert r_keep.passed is False


# ---------------------------------------------------------------------------
# Fix wave 2, package L7 (fixreview-p4 I-1, I-2)
# ---------------------------------------------------------------------------

import cytherea.backends.pes_suite as pes_mod  # noqa: E402


class _Springs:
    """Independent anisotropic springs per atom; force on atoms < n_bad scaled."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, n, n_bad, factor):
        rng = np.random.default_rng(1)
        self.k = rng.uniform(0.5, 2.0, size=(n, 3))
        self.n_bad, self.f = n_bad, factor

    def energy_forces(self, x, box=None):
        E = 0.5 * float(np.sum(self.k * x**2))
        F = -self.k * x
        F[: self.n_bad] *= self.f
        return E, F

    def build(self, *a):
        raise NotImplementedError

    def effective_config(self, cfg=None):
        return {"platform": "CUDA", "precision": "mixed"}

    def provenance(self, cfg=None):
        return {}


def _spring_probes(n=3000, k=5):
    rng = np.random.default_rng(2)
    return [rng.normal(size=(n, 3)) for _ in range(k)]


def test_l7_i1_localised_defect_is_caught_when_its_group_is_given():
    """p4 I-1: a uniform 16-atom draw over 3000 atoms never touched a defect
    confined to 60 atoms (here a 50 % *reduction*, so the defective atoms
    are not among the largest forces either)."""
    b, probes = _Springs(3000, 60, 0.5), _spring_probes()
    blind = pes_consistency_suite(b, probes, mode="sampled")
    assert blind.passed  # the uniform draw alone is blind to it ...
    seen = pes_consistency_suite(b, probes, mode="sampled", fd_atom_groups={"solute": range(60)})
    assert not seen.passed and any(r.startswith("fd") for r in seen.reasons)  # ... the group is not
    assert all(len(set(a) & set(range(60))) >= 16 for a in seen.fd_atoms)
    inc = pes_consistency_suite(b, probes, mode="sampled", fd_atoms_include=[7])
    assert not inc.passed and all(7 in a for a in inc.fd_atoms)


def test_l7_i1_top_force_atoms_seed_and_reporting():
    b, probes = _Springs(3000, 60, 1.5), _spring_probes()  # amplified: among the largest forces
    rep = pes_consistency_suite(b, probes, mode="sampled")
    assert not rep.passed  # the top-|F| atoms are always checked
    assert len(rep.fd_atoms) == len(probes) and all(len(a) >= 16 for a in rep.fd_atoms)
    other = pes_consistency_suite(_Springs(3000, 0, 1.0), probes, mode="sampled", fd_seed=3)
    base = pes_consistency_suite(_Springs(3000, 0, 1.0), probes, mode="sampled")
    assert other.passed and base.passed and other.fd_atoms != base.fd_atoms
    assert base.fd_step == 1e-4  # sampled-mode default step
    for kw in ({"fd_atoms_include": [1]}, {"fd_atom_groups": {"g": [1]}}, {"fd_seed": 2}):
        with pytest.raises(ValueError, match="sampled"):
            pes_consistency_suite(_Springs(10, 0, 1.0), [np.ones((10, 3))], **kw)
    with pytest.raises(ValueError, match="out of range"):
        pes_consistency_suite(_Springs(10, 0, 1.0), [np.ones((10, 3))], mode="sampled",
                              fd_atoms_include=[10])


class _Noisy:
    """Springs plus deterministic 'roundoff' in the energy (p4 repro)."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, n, a, delta=0.0):
        rng = np.random.default_rng(1)
        self.k = rng.uniform(0.5, 2.0, size=(n, 3))
        self.w = rng.normal(size=(n, 3))
        self.a, self.delta = a, delta

    def energy_forces(self, x, box=None):
        E = 0.5 * float(np.sum(self.k * x**2)) + self.a * np.sin(1e9 * float(np.sum(self.w * x)))
        return E, -self.k * x * (1.0 + self.delta)

    def build(self, *a):
        raise NotImplementedError

    def effective_config(self, cfg=None):
        return {"platform": "CUDA", "precision": "mixed"}

    def provenance(self, cfg=None):
        return {}


@pytest.mark.parametrize("delta,ok", [(0.0, True), (0.0075, True), (0.01, False), (0.015, False)])
def test_l7_i2_noise_allowance_cannot_hide_a_defect_of_twice_the_tolerance(delta, ok):
    """p4 I-2: with FD noise just under the gate (mixed, fd_rtol 5e-3), the
    probe-wide 3 x max allowance let a 1.5 % global defect pass. The
    per-coordinate allowance bounds the effective tolerance by ~2 x fd_rtol.
    The parametrisation also pins _FD_NOISE_FACTOR (3 -> fd_unresolved at
    delta=0; 1 -> delta=0.0075 fails)."""
    n = 40
    rng = np.random.default_rng(3)
    probes = [rng.normal(size=(n, 3)) for _ in range(3)]
    rep = pes_consistency_suite(_Noisy(n, 8e-8, delta), probes, mode="sampled", n_fd_atoms=n, fd_step=1e-5)
    assert rep.tolerances["fd_rtol"] == 5e-3
    assert rep.passed is ok, rep.reasons
    assert rep.fd_noise_floor_rel < 5e-3


def test_l7_i2_too_much_noise_is_fd_unresolved_not_a_pass():
    n = 40
    rng = np.random.default_rng(3)
    probes = [rng.normal(size=(n, 3)) for _ in range(3)]
    rep = pes_consistency_suite(_Noisy(n, 1e-7, 0.0), probes, mode="sampled", n_fd_atoms=n, fd_step=1e-5)
    assert not rep.passed and any(r.startswith("fd_unresolved") for r in rep.reasons)


# ---------------------------------------------------------------------------
# Cutoff crossings (A1 14.1, 2026-10-04): a plain cutoff makes E jump at r_c,
# so a finite difference whose stencil moves a pair across r_c measures the
# jump, not the force. Backends that declare their cutoffs
# (`energy_cutoffs()`) get those coordinates skipped and counted.
# ---------------------------------------------------------------------------

class _TruncatedPairBackend:
    """E = sum over pairs with r < rc of eps (sig/r)^6 -- unshifted, so E jumps
    by eps (sig/rc)^6 where a pair crosses rc. Optional orthorhombic box (min
    image). ``declare`` toggles the energy_cutoffs() hook; ``bad_atom`` scales
    that atom's force by 1.05 (a defect the FD check must still see)."""

    kind = "analytic"
    gpu_resident = False

    def __init__(self, rc=0.9, box=None, declare=True, bad_atom=None, eps=100.0, sig=0.3, precision="mixed"):
        self.rc, self.box, self.eps, self.sig = rc, box, eps, sig
        self.bad_atom, self.precision = bad_atom, precision
        if declare:
            self.energy_cutoffs = lambda: {"cutoffs_nm": [rc],
                                           "box_lengths_nm": None if box is None else list(box)}

    def _d(self, x):
        d = x[None, :, :] - x[:, None, :]
        if self.box is not None:
            L = np.asarray(self.box)
            d -= np.round(d / L) * L
        return d

    def energy_forces(self, x, box=None):
        x = np.asarray(x, float)
        d = self._d(x)
        r = np.linalg.norm(d, axis=-1)
        n = x.shape[0]
        iu = np.triu_indices(n, 1)
        on = np.zeros((n, n), bool)
        on[iu] = r[iu] < self.rc
        E = float(np.sum(self.eps * (self.sig / r[on]) ** 6))
        F = np.zeros_like(x)
        for i, j in zip(*np.nonzero(on)):
            g = 6 * self.eps * self.sig**6 / r[i, j] ** 8 * d[i, j]  # -dE/dx_j along d = x_j - x_i
            F[j] += g
            F[i] -= g
        if self.bad_atom is not None:
            F[self.bad_atom] *= 1.05
        return E, F

    def build(self, s, cfg, rng_key):
        raise NotImplementedError

    def effective_config(self, cfg=None):
        return {"backend": "fake_truncated_pair", "precision": self.precision}

    def provenance(self, cfg=None):
        return {"kind": self.kind}


def _pair_probe(gap, h=1e-4):
    """6 atoms: 0-1 at rc + gap*h (inside the FD stencil when |gap| < 1), the
    rest well inside the cutoff of their neighbours and far from rc."""
    x = np.array([[0.0, 0.0, 0.0], [0.9 + gap * h, 0.0, 0.0], [0.35, 0.33, 0.0],
                  [0.45, -0.32, 0.05], [0.3, 0.0, 0.36], [0.6, 0.05, -0.34]])
    return x


def _two_crossing_probe(h=1e-4):
    """Atom 0 moving along x: pair 0-1 enters the cutoff at +0.3 h (both FD
    stencils), pair 0-2 leaves it at -~0.84 h (only the h stencil), so
    F_fd(h) and F_fd(h/2) carry the same jump error and the per-coordinate
    FD allowance cannot absorb it -- as on the solvated A1 system."""
    r2 = 0.9 - 0.7 * h
    return np.array([[0.0, 0.0, 0.0], [0.9 + 0.3 * h, 0.0, 0.0], [np.sqrt(r2**2 - 0.25), 0.5, 0.0],
                     [0.3, -0.35, 0.1], [0.25, 0.05, 0.4], [0.5, -0.1, -0.35]])


def test_cutoff_crossing_fails_fd_without_a_declared_cutoff():
    r = pes_consistency_suite(_TruncatedPairBackend(declare=False), [_two_crossing_probe()], mode="sampled")
    assert not r.passed and any(s.startswith("fd") for s in r.reasons), r.reasons
    assert r.fd_skipped_cutoff == 0
    ok = pes_consistency_suite(_TruncatedPairBackend(), [_two_crossing_probe()], mode="sampled")
    assert ok.passed, ok.reasons
    assert ok.fd_skipped_cutoff >= 1


def test_declared_cutoff_skips_crossing_coordinates_and_reports_them():
    r = pes_consistency_suite(_TruncatedPairBackend(), [_pair_probe(0.3)], mode="sampled")
    assert r.passed, r.reasons
    assert r.fd_skipped_cutoff >= 1 and r.n_fd_coords == 18 - r.fd_skipped_cutoff
    assert r.fd_cutoffs == [0.9]
    # far from the cutoff nothing is skipped
    far = _pair_probe(0.0)
    far[1, 0] = 0.7
    r2 = pes_consistency_suite(_TruncatedPairBackend(), [far], mode="sampled")
    assert r2.passed and r2.fd_skipped_cutoff == 0 and r2.n_fd_coords == 18


def test_declared_cutoff_still_catches_a_force_defect():
    r = pes_consistency_suite(_TruncatedPairBackend(bad_atom=3), [_pair_probe(0.3)], mode="sampled")
    assert not r.passed and any(s.startswith("fd") for s in r.reasons)


def test_cutoff_crossing_through_a_periodic_image():
    x = _pair_probe(0.0)
    x[1] = [-0.9 - 0.3e-4 + 2.0, 0.0, 0.0]  # 0-1 at 1.1 nm directly, rc + 0.3 h through the image
    r = pes_consistency_suite(_TruncatedPairBackend(box=(2.0, 2.0, 2.0)), [x], mode="sampled")
    assert r.passed, r.reasons
    assert r.fd_skipped_cutoff == 2  # atom 0 and atom 1, x only
    nobox = pes_consistency_suite(_TruncatedPairBackend(), [x], mode="sampled")  # no image: 1.1 nm, no crossing
    assert nobox.fd_skipped_cutoff == 0


def test_every_coordinate_skipped_is_unresolved():
    x = np.array([[0.0, 0.0, 0.0], [0.9 + 0.3e-4, 0.0, 0.0]])
    x[1] = (0.9 + 0.3e-4) * np.ones(3) / np.sqrt(3.0)  # every component moves r
    r = pes_consistency_suite(_TruncatedPairBackend(), [x], mode="sampled")
    assert not r.passed
    assert any(s.startswith("fd_unresolved") for s in r.reasons)
    assert r.n_fd_coords == 0 and r.fd_skipped_cutoff == 6


def test_cutoff_guard_needs_atom_rows():
    class _Flat(_TruncatedPairBackend):
        def energy_forces(self, x, box=None):
            E, F = super().energy_forces(np.asarray(x).reshape(-1, 3))
            return E, F.reshape(-1)

    with pytest.raises(ValueError, match="n_atoms, 3"):
        pes_consistency_suite(_Flat(), [_pair_probe(5.0).reshape(-1)], mode="sampled")


class _JitterBackend(_TruncatedPairBackend):
    """First evaluation at a position differs from later ones by `rel` of the
    force scale (like OpenMM CUDA mixed reordering atoms after its first call)."""

    def __init__(self, rel, **kw):
        super().__init__(**kw)
        self.rel, self._seen = rel, set()

    def energy_forces(self, x, box=None):
        E, F = super().energy_forces(x)
        k = np.asarray(x).tobytes()
        if k not in self._seen:
            self._seen.add(k)
            F = F + self.rel * np.max(np.abs(F))
        return E, F


def test_repeat_tolerance_defaults_follow_the_precision_row():
    far = _pair_probe(0.0)
    far[1, 0] = 0.7
    r = pes_consistency_suite(_JitterBackend(3e-4), [far], mode="sampled", precision="mixed")
    assert not r.repeat_bitwise and r.passed, r.reasons
    assert r.tolerances["repeat_rtol"] == DEFAULT_TOLERANCES["mixed"]["repeat_rtol"]
    strict0 = pes_consistency_suite(_JitterBackend(3e-4), [far], mode="sampled", precision="mixed", repeat_rtol=0.0)
    assert not strict0.passed and any(s.startswith("repeat") for s in strict0.reasons)
    big = pes_consistency_suite(_JitterBackend(5e-3), [far], mode="sampled", precision="mixed")
    assert not big.passed and any(s.startswith("repeat") for s in big.reasons)
    assert DEFAULT_TOLERANCES["double"]["repeat_rtol"] == 0.0
