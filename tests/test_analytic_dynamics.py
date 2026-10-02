"""Tests for `AnalyticBackend.build`'s overdamped/BAOAB propagators
(task-4-brief.md, "必测用例" 4.1-4.5), plus the LJCluster NVE
mass-broadcasting case and overdamped-vs-baoab determinism checks the
controller ruling asked for explicitly.

Parameter/statistic choices are documented inline at each test and
summarized in task-4-report.md (dt choices, thinning/autocorrelation
justification for 4.2, the two sigma conventions used in 4.1).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cytherea.backends.analytic import (
    AnalyticBackend,
    DoubleWell1D,
    FreeParticle,
    Harmonic,
    LJCluster,
)
from cytherea.backends.base import MDState
from cytherea.backends.pes_suite import pes_consistency_suite
from cytherea.keys import ShotKey, derive_rng


def _key(shot_id: int, stage: str) -> ShotKey:
    return ShotKey(global_seed=0, frame_id=0, shot_id=shot_id, stage=stage)


# ---------------------------------------------------------------------------
# 4.1: overdamped FreeParticle(3) MSD/(6*D*t) == 1 +- 3*sigma, N=2000, t=10.
#
# Fix round 1 / Important 3: uses non-unit kT, gamma, mass (kT=0.7,
# gamma=3.0, mass=2.5) rather than all-1.0, per the review finding that
# unit parameters can hide bugs that only show up when kT/gamma/mass
# actually appear with distinct values (e.g. a missing mass factor in D).
# D = kT/(mass*gamma) per ruling R14 (gamma is a rate for both
# integrators); the MSD ratio itself is dimensionless and D cancels out of
# it algebraically, so this is mostly a "does the test even reference
# mass/gamma correctly" regression guard rather than a new physics check
# -- the point is that 4.1 no longer *could* pass by accident if mass had
# silently dropped out of the diffusion constant.
#
# A sign error in the drift term would NOT be caught here (FreeParticle's
# force is identically zero, so overdamped reduces to pure Brownian motion
# regardless of drift sign) -- only a wrong noise amplitude can fail this
# test. Test 4.2 below (non-zero force) is what catches a drift sign error;
# see task-4-report.md's self-review section.
# ---------------------------------------------------------------------------


def test_overdamped_free_particle_msd_matches_diffusion_law():
    dim = 3
    kT = 0.7
    gamma = 3.0
    mass = 2.5
    D = kT / (mass * gamma)
    dt = 0.1
    n_steps = 100
    t_final = dt * n_steps
    N = 2000

    backend = AnalyticBackend(
        potential=FreeParticle(dim=dim),
        integrator="overdamped",
        dt=dt,
        kT=kT,
        gamma=gamma,
        mass=mass,
    )

    sq_disp = np.empty(N)
    for i in range(N):
        x0 = np.zeros(dim)
        state = MDState(x=x0, v=np.zeros(dim), t=0.0)
        prop = backend.build(state, None, _key(i, "msd"))
        prop.run(n_steps)
        xf = prop.get_state().x
        sq_disp[i] = np.sum((xf - x0) ** 2)

    ratio = sq_disp.mean() / (6.0 * D * t_final)
    # Analytic standard error of the mean of |dx|**2/(6*D*t) for dim=3:
    # each dx_i ~ N(0, 2*D*t), so |dx|**2/(6*D*t) has variance 2/3 (see
    # task-4-report.md derivation), giving sigma = sqrt(24)/(6*sqrt(N))
    # exactly as quoted in the task brief.
    sigma = np.sqrt(24.0) / (6.0 * np.sqrt(N))
    assert abs(ratio - 1.0) < 3 * sigma, (ratio, sigma)


# ---------------------------------------------------------------------------
# 4.2: BAOAB DoubleWell1D(3.0), >=1e6 total steps, KS test of the position
# histogram against the exact Boltzmann CDF (numerical integration).
#
# dt=0.005: the harmonic frequency at either well minimum is
# omega = sqrt(V''(x0)/mass) = sqrt(8*barrier/x0**2) = sqrt(24) ~ 4.90, so
# omega*dt ~ 0.0245 and BAOAB's O(dt**2) configurational bias
# (~(omega*dt)**2/8 ~ 7.5e-5 relative) is negligible next to the KS test's
# sensitivity.
#
# gamma=1.0, kT=1.0: barrier=3*kT gives a moderate (not vanishing, not
# instantaneous) inter-well hopping rate, empirically ~1 crossing per ~30
# time units (see task-4-report.md prototyping).
#
# Thinning: burn-in of 20_000 steps (100 time units) discards the initial-
# condition bias of starting exactly at a well minimum; samples are then
# taken every `stride=14_000` steps (70 time units, ~2x the empirically
# measured integrated autocorrelation time of ~33.5 time units / 6707
# steps -- see report). The test verifies this stride actually decorrelates
# by asserting the *measured* lag-1 autocorrelation of the thinned samples
# is small, rather than just trusting the a priori choice.
#
# This is marked slow (>=1e6 steps takes ~25s wall time with this pure-
# Python single-trajectory propagator) -- run explicitly with
# `pytest -m slow -k test_baoab_double_well_matches_boltzmann_distribution`.
# ---------------------------------------------------------------------------


def _boltzmann_cdf(potential_1d, kT: float, xmin: float, xmax: float, n_grid: int = 200_001):
    grid = np.linspace(xmin, xmax, n_grid)
    V = np.array([potential_1d.energy_grad(np.array([xi]))[0] for xi in grid])
    unnorm = np.exp(-(V - V.min()) / kT)
    cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (unnorm[1:] + unnorm[:-1]) * np.diff(grid)))
    )
    cdf /= cdf[-1]

    def cdf_func(x):
        return np.interp(x, grid, cdf)

    return cdf_func


@pytest.mark.slow
def test_baoab_double_well_matches_boltzmann_distribution():
    barrier, kT, gamma, dt, mass = 3.0, 1.0, 1.0, 0.005, 1.0
    n_traj = 8
    n_steps = 250_000
    burn_in = 20_000
    stride = 14_000

    potential = DoubleWell1D(barrier)
    backend = AnalyticBackend(
        potential=potential, integrator="baoab", dt=dt, kT=kT, gamma=gamma, mass=mass
    )

    all_samples = []
    per_traj_seqs = []
    for i in range(n_traj):
        state = MDState(x=np.array([1.0]), v=np.array([0.0]), t=0.0)
        prop = backend.build(state, None, _key(i, "dw_boltzmann"))
        prop.run(burn_in)
        seq = []
        n_samples = (n_steps - burn_in) // stride
        for _ in range(n_samples):
            prop.run(stride)
            seq.append(prop.get_state().x[0])
        per_traj_seqs.append(seq)
        all_samples.extend(seq)

    all_samples = np.array(all_samples)
    assert len(all_samples) >= 100
    assert n_traj * n_steps >= 1_000_000

    # Justify the thinning stride: pooled lag-1 autocorrelation of the
    # thinned samples must be small (an untinned or under-thinned series
    # from a slowly-hopping bistable potential would show strong lag-1
    # correlation).
    a, b = [], []
    for seq in per_traj_seqs:
        seq = np.asarray(seq)
        a.extend(seq[:-1])
        b.extend(seq[1:])
    lag1_corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(lag1_corr) < 0.3, lag1_corr

    cdf_func = _boltzmann_cdf(potential, kT, xmin=-3.0, xmax=3.0)
    ks = stats.kstest(all_samples, cdf_func)
    assert ks.pvalue > 0.01, (ks.statistic, ks.pvalue)


# ---------------------------------------------------------------------------
# 4.3: same rng_key + same initial state, run twice -> bitwise identical.
# Checked for both integrators (controller ruling).
# ---------------------------------------------------------------------------


def _run_and_collect(backend, state, key, n_steps, chunk=1):
    prop = backend.build(state, None, key)
    xs, vs, ts = [], [], []
    for _ in range(n_steps // chunk):
        prop.run(chunk)
        s = prop.get_state()
        xs.append(s.x.copy())
        vs.append(s.v.copy())
        ts.append(s.t)
    return np.array(xs), np.array(vs), np.array(ts)


def test_baoab_determinism_same_key_bitwise_identical():
    backend = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator="baoab", dt=0.01, kT=1.0, gamma=1.0
    )
    state = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)
    key = _key(0, "determinism")

    xs1, vs1, ts1 = _run_and_collect(backend, state, key, n_steps=500)
    xs2, vs2, ts2 = _run_and_collect(backend, state, key, n_steps=500)

    assert np.array_equal(xs1, xs2)
    assert np.array_equal(vs1, vs2)
    assert np.array_equal(ts1, ts2)


def test_overdamped_determinism_same_key_bitwise_identical():
    backend = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator="overdamped", dt=0.01, kT=1.0, gamma=1.0
    )
    state = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)
    key = _key(0, "determinism")

    xs1, vs1, ts1 = _run_and_collect(backend, state, key, n_steps=500)
    xs2, vs2, ts2 = _run_and_collect(backend, state, key, n_steps=500)

    assert np.array_equal(xs1, xs2)
    assert np.array_equal(vs1, vs2)
    assert np.array_equal(ts1, ts2)


# ---------------------------------------------------------------------------
# 4.4: two distinct rng_keys, same initial state -> trajectories have
# already diverged by step 10. Checked for both integrators.
# ---------------------------------------------------------------------------


def test_baoab_distinct_keys_diverge_by_step_10():
    backend = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator="baoab", dt=0.01, kT=1.0, gamma=1.0
    )
    state = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)

    prop1 = backend.build(state, None, _key(1, "diverge"))
    prop2 = backend.build(state, None, _key(2, "diverge"))
    prop1.run(10)
    prop2.run(10)

    assert not np.array_equal(prop1.get_state().x, prop2.get_state().x)


def test_overdamped_distinct_keys_diverge_by_step_10():
    backend = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator="overdamped", dt=0.01, kT=1.0, gamma=1.0
    )
    state = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)

    prop1 = backend.build(state, None, _key(1, "diverge"))
    prop2 = backend.build(state, None, _key(2, "diverge"))
    prop1.run(10)
    prop2.run(10)

    assert not np.array_equal(prop1.get_state().x, prop2.get_state().x)


# ---------------------------------------------------------------------------
# 4.5: BAOAB with gamma=0 reduces to velocity Verlet -> NVE energy drift
# < 1e-4 for a harmonic oscillator, via pes_consistency_suite. omega*dt is
# kept small (<=0.02) so the O(dt**2) velocity-Verlet drift
# (~(omega*dt)**2/8) holds well under the 1e-4 tolerance with margin:
# k=1.0, mass=1.0 -> omega=1.0; dt=0.01 -> omega*dt=0.01,
# (omega*dt)**2/8 ~= 1.25e-5.
# ---------------------------------------------------------------------------


def test_baoab_gamma_zero_harmonic_nve_energy_drift_below_tolerance():
    backend = AnalyticBackend(
        potential=Harmonic(k=1.0, dim=3),
        integrator="baoab",
        dt=0.01,
        kT=1.0,
        gamma=0.0,
        mass=1.0,
    )
    probes = [np.array([1.0, 0.5, -0.3])]
    # Spec S2: the suite starts from Maxwell-Boltzmann velocities at kT and
    # normalises the drift by the kinetic-energy scale n_dof*kT/2 (not by
    # |E_tot(0)|, which depends on the energy zero).
    report = pes_consistency_suite(
        backend, probes, nve_steps=100_000, masses=1.0, kT=1.0
    )
    assert report.nve_rel_drift is not None
    assert report.nve_rel_drift < 1e-4
    assert report.passed


# ---------------------------------------------------------------------------
# LJCluster NVE with masses shaped (n_atoms, 1) for (n_atoms, 3)
# broadcasting -- Task 3 review flagged this path as untested. 3 atoms
# started near (not exactly at, so there is real PE<->KE exchange to
# conserve) the equilateral-triangle LJ minimum (side = 2**(1/6)*sigma).
# ---------------------------------------------------------------------------


def test_baoab_gamma_zero_lj_cluster_nve_energy_drift_below_tolerance():
    r_min = 2.0 ** (1.0 / 6.0)
    x0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_min, 0.0, 0.0],
            [r_min / 2.0, r_min * np.sqrt(3.0) / 2.0, 0.0],
        ]
    )
    x0[0, 0] += 0.05  # perturb off the exact minimum so PE/KE actually trade off

    backend = AnalyticBackend(
        potential=LJCluster(n_atoms=3, epsilon=1.0, sigma=1.0),
        integrator="baoab",
        dt=0.001,
        kT=1.0,
        gamma=0.0,
        mass=1.0,
    )
    masses = np.full((3, 1), 1.0)  # broadcasts against x/v of shape (3, 3)
    # kT=0.1 (<< epsilon): thermal velocities that keep the cluster bound.
    report = pes_consistency_suite(
        backend, [x0], nve_steps=5000, masses=masses, kT=0.1
    )
    assert report.nve_rel_drift is not None
    assert report.nve_rel_drift < 1e-4
    assert report.passed


# ===========================================================================
# Fix round 1 (task-4 review): the sections below were added because the
# review found the *integrator formulas* correct but several *tests*
# structurally unable to fail for the bugs they were meant to catch. See
# task-4-report.md's "Fix round 1" section for the full review findings and
# negative-control evidence (RED-phase reruns against deliberately-broken
# code, and non-committed scratch negative controls).
# ===========================================================================


# ---------------------------------------------------------------------------
# Ruling R15 #4: AnalyticBackend.__init__ validates its parameters instead
# of silently producing NaN/inf (overdamped gamma<=0), a sign-reversed drift
# (overdamped/baoab gamma<0), a frozen trajectory (dt<=0), or other
# nonsensical configurations deep inside a propagator's step loop.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(integrator="overdamped", dt=0.01, kT=1.0, gamma=0.0),
        dict(integrator="overdamped", dt=0.01, kT=1.0, gamma=-1.0),
        dict(integrator="baoab", dt=0.01, kT=1.0, gamma=-1.0),
        dict(integrator="overdamped", dt=0.0, kT=1.0, gamma=1.0),
        dict(integrator="overdamped", dt=-0.01, kT=1.0, gamma=1.0),
        dict(integrator="baoab", dt=0.01, kT=-1.0, gamma=1.0),
        dict(integrator="overdamped", dt=0.01, kT=1.0, gamma=1.0, mass=0.0),
        dict(integrator="overdamped", dt=0.01, kT=1.0, gamma=1.0, mass=-2.0),
    ],
)
def test_analytic_backend_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AnalyticBackend(potential=FreeParticle(dim=1), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(integrator="baoab", dt=0.01, kT=1.0, gamma=0.0),  # 4.5's NVE limit
        dict(integrator="overdamped", dt=0.01, kT=1.0, gamma=1.0),
        dict(integrator="baoab", dt=0.01, kT=0.0, gamma=1.0),  # kT=0 is valid (T=0)
    ],
)
def test_analytic_backend_accepts_valid_boundary_parameters(kwargs):
    AnalyticBackend(potential=FreeParticle(dim=1), **kwargs)  # must not raise


# ---------------------------------------------------------------------------
# Ruling R15 #6: set_state() must not reset the RNG (checkpoint resume
# continues the same random stream rather than replaying it), and run(n)
# must be bitwise identical to n calls of run(1) (chunking is purely a
# caller convenience, not part of the physics).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrator", ["baoab", "overdamped"])
def test_set_state_does_not_reset_rng(integrator):
    backend = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator=integrator, dt=0.01, kT=1.0, gamma=1.0
    )
    s0 = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)
    prop = backend.build(s0, None, _key(0, "rng_persist"))

    prop.run(50)
    first_segment_end = prop.get_state().x.copy()

    prop.set_state(s0)  # rewind x/v/t to the start...
    prop.run(50)  # ...but the RNG stream must have kept advancing
    second_segment_end = prop.get_state().x.copy()

    # If set_state() had reset the RNG, this second 50-step run from the
    # same s0 would reproduce first_segment_end exactly (that would be the
    # *correct* behavior for a fresh build() with the same key, which
    # test_*_determinism_same_key_bitwise_identical above already checks --
    # this test specifically wants the *opposite* outcome for set_state()).
    assert not np.array_equal(first_segment_end, second_segment_end)


@pytest.mark.parametrize("integrator", ["baoab", "overdamped"])
def test_run_chunking_is_bitwise_irrelevant(integrator):
    dt, kT, gamma = 0.01, 1.0, 1.0
    key = _key(0, "chunking")
    s0 = MDState(x=np.array([0.3]), v=np.array([0.0]), t=0.0)
    n = 37  # deliberately not a round number / power of 2

    backend_a = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator=integrator, dt=dt, kT=kT, gamma=gamma
    )
    prop_a = backend_a.build(s0, None, key)
    prop_a.run(n)

    backend_b = AnalyticBackend(
        potential=DoubleWell1D(3.0), integrator=integrator, dt=dt, kT=kT, gamma=gamma
    )
    prop_b = backend_b.build(s0, None, key)
    for _ in range(n):
        prop_b.run(1)

    sa, sb = prop_a.get_state(), prop_b.get_state()
    assert np.array_equal(sa.x, sb.x)
    assert np.array_equal(sa.v, sb.v)
    assert sa.t == sb.t


# ---------------------------------------------------------------------------
# Important 2 (fix round 1): the overdamped drift term (and, per ruling
# R14, its mobility 1/(mass*gamma)) had no test that could fail for a wrong
# drift sign or a wrong/missing mobility factor -- Harmonic's force is the
# only thing FreeParticle's zero-force 4.1 can't exercise, and nothing
# before this checked overdamped's *drift* at all. Two complementary tests,
# both using a non-unit (kT, gamma, mass, k) tuple (Important 3):
#
#   kT=0.7, mass=2.5, gamma=3.0, k=2.0 (Harmonic's own spring constant,
#   deliberately distinct from kT/gamma/mass so no two parameters can
#   accidentally cancel each other's bugs).
#
#   mobility = 1/(mass*gamma) = 0.13333, lambda = k*mobility = 0.26667
#   (relaxation rate), D = kT*mobility = 0.093333.
#
# (a) stationary-moment test: <x^2> = kT/k and <x*V'(x)> = kT. For an
#     overdamped harmonic oscillator this is exactly the fluctuation-
#     dissipation *ratio* D/lambda = kT/k, which is invariant to an
#     overall mobility rescaling (scaling mobility scales both D and
#     lambda by the same factor, canceling in the ratio) -- so on its own
#     this test can catch e.g. an inconsistent noise/drift relationship,
#     but NOT a missing-mass-in-mobility bug (verified empirically in
#     task-4-report.md: this test's ratio is unchanged between the old
#     buggy D=kT/gamma and the fixed D=kT/(mass*gamma) formula when run
#     with mass != 1). That is exactly why the OU mean-relaxation test (b)
#     below is required in addition, not as an alternative.
# (b) OU mean-relaxation test: <x(t)> = x0*exp(-lambda*t) across an
#     ensemble of independent trajectories (the stochastic term averages
#     to zero, leaving only the deterministic drift). This *does* pin the
#     absolute mobility factor, since lambda = k/(mass*gamma) directly.
# ---------------------------------------------------------------------------


def test_overdamped_harmonic_stationary_moments_match_boltzmann():
    kT, mass, gamma, k = 0.7, 2.5, 3.0, 2.0
    dt = 0.01
    burn_steps = 1875  # ~5/lambda: initial-condition transient e^-5 negligible
    stride = 750  # ~2/lambda: decorrelates x^2 (lag-1 corr ~ e^-2 ~ 0.14)
    n_chains = 4
    samples_per_chain = 300

    backend = AnalyticBackend(
        potential=Harmonic(k=k, dim=1), integrator="overdamped",
        dt=dt, kT=kT, gamma=gamma, mass=mass,
    )

    all_x = []
    per_chain = []
    for i in range(n_chains):
        state = MDState(x=np.array([1.0]), v=np.array([0.0]), t=0.0)
        prop = backend.build(state, None, _key(i, "overdamped_moments"))
        prop.run(burn_steps)
        seq = []
        for _ in range(samples_per_chain):
            prop.run(stride)
            seq.append(prop.get_state().x[0])
        per_chain.append(seq)
        all_x.extend(seq)

    xs = np.array(all_x)
    n = len(xs)

    # Justify the thinning stride: pooled lag-1 autocorrelation of x^2 (the
    # quantity actually averaged below) must be small.
    a, b = [], []
    for seq in per_chain:
        seq2 = np.asarray(seq) ** 2
        a.extend(seq2[:-1])
        b.extend(seq2[1:])
    lag1_corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(lag1_corr) < 0.3, lag1_corr

    x2 = xs**2
    x2_mean = x2.mean()
    x2_sem = x2.std(ddof=1) / np.sqrt(n)
    expected_x2 = kT / k
    assert abs(x2_mean - expected_x2) < 3 * x2_sem, (x2_mean, x2_sem, expected_x2)

    xgrad = xs * (k * xs)  # x * V'(x), V'(x) = k*x
    xgrad_mean = xgrad.mean()
    xgrad_sem = xgrad.std(ddof=1) / np.sqrt(n)
    assert abs(xgrad_mean - kT) < 3 * xgrad_sem, (xgrad_mean, xgrad_sem, kT)


def test_overdamped_harmonic_ou_mean_relaxation_pins_mobility():
    kT, mass, gamma, k = 0.7, 2.5, 3.0, 2.0
    mobility = 1.0 / (mass * gamma)
    lam = k * mobility  # relaxation rate = k/(mass*gamma), ruling R14
    dt = 0.01
    x0 = 5.0
    t_check = 2.0 / lam  # e^-2 decay: strong signal, still >> thermal noise
    n_steps = int(round(t_check / dt))
    N = 1000

    backend = AnalyticBackend(
        potential=Harmonic(k=k, dim=1), integrator="overdamped",
        dt=dt, kT=kT, gamma=gamma, mass=mass,
    )

    xs = np.empty(N)
    for i in range(N):
        state = MDState(x=np.array([x0]), v=np.array([0.0]), t=0.0)
        prop = backend.build(state, None, _key(i, "ou_relax"))
        prop.run(n_steps)
        xs[i] = prop.get_state().x[0]

    expected = x0 * np.exp(-lam * t_check)
    mean = xs.mean()
    sem = xs.std(ddof=1) / np.sqrt(N)
    assert abs(mean - expected) < 3 * sem, (mean, sem, expected)


# ---------------------------------------------------------------------------
# Important 1 (fix round 1): the brief-literal 4.2 KS test (kept above,
# unmodified) is a *weak* alternative-specific test -- against a gross
# temperature error (noise scaled by a constant factor in the O-step), the
# double well's exact left-right symmetry keeps well populations at 50/50
# at any temperature, so the largest CDF gap stays tiny and the plain KS
# test has very low power to detect it at this sample size. Two
# complementary tests, both using a non-unit (kT, gamma, mass) tuple
# distinct from the plain-KS test's (kT=1, gamma=1, mass=1):
#
#   kT=0.7, gamma=3.0, mass=2.5 (same tuple as the overdamped moment tests
#   above, reused for consistency -- not required to match, just
#   convenient).
#
# Both exploit that DoubleWell1D's V(x) = barrier*((x/x0)^2-1)^2 is an
# *even* function, so V'(x) is odd and x*V'(x) is even, and the
# distribution of |x| is identical between the two wells (mirror images of
# each other) -- meaning neither check needs the trajectory to actually
# hop between wells to be unbiased, only to sample its own well's local
# equilibrium. This is what makes both checks cheap: unlike the brief's
# literal 4.2 KS test, they do not depend on the rare (Kramers) inter-well
# hopping rate at all, only on the fast intra-well vibrational relaxation.
#
# (a) configurational + kinetic temperature: <x*V'(x)> = kT and
#     <mass*v^2> = kT, sampled from a single trajectory after a short
#     burn-in, thinned by a stride verified (via measured lag-1
#     autocorrelation of the actual sampled quantities, x^2 and v^2 -- NOT
#     of raw x, which stays strongly correlated across the rare inter-well
#     hops for hundreds of time units and would give a wildly misleading
#     "still correlated" reading for a quantity nobody is actually
#     averaging) to already fully decorrelate these two quantities.
# (b) KS test on |x| against the folded exact CDF, 2*F(|x|)-1 -- thousands
#     of samples at the same cheap thinning stride as (a).
# ---------------------------------------------------------------------------


def _double_well_local_samples(potential, kT, gamma, mass, dt, burn_steps, stride, n_samples, key_stage):
    backend = AnalyticBackend(
        potential=potential, integrator="baoab", dt=dt, kT=kT, gamma=gamma, mass=mass
    )
    state = MDState(x=np.array([1.0]), v=np.array([0.0]), t=0.0)
    prop = backend.build(state, None, _key(0, key_stage))
    prop.run(burn_steps)
    xs = np.empty(n_samples)
    vs = np.empty(n_samples)
    for i in range(n_samples):
        prop.run(stride)
        s = prop.get_state()
        xs[i] = s.x[0]
        vs[i] = s.v[0]
    return xs, vs


def test_baoab_double_well_configurational_and_kinetic_temperature():
    barrier, kT, gamma, mass, dt = 3.0, 0.7, 3.0, 2.5, 0.005
    burn_steps = 1500
    stride = 300  # see module comment: decorrelates x^2/v^2, not raw x
    M = 2500
    potential = DoubleWell1D(barrier)

    xs, vs = _double_well_local_samples(
        potential, kT, gamma, mass, dt, burn_steps, stride, M, "dw_temp"
    )

    # Justify the stride against the quantities actually averaged, not raw x.
    lag1_x2 = float(np.corrcoef((xs**2)[:-1], (xs**2)[1:])[0, 1])
    lag1_v2 = float(np.corrcoef((vs**2)[:-1], (vs**2)[1:])[0, 1])
    assert abs(lag1_x2) < 0.3, lag1_x2
    assert abs(lag1_v2) < 0.3, lag1_v2

    def gradV(x):
        u = (x / potential.x0) ** 2 - 1.0
        return barrier * 4.0 * x * u / (potential.x0**2)

    xgrad = xs * gradV(xs)
    xgrad_mean, xgrad_sem = xgrad.mean(), xgrad.std(ddof=1) / np.sqrt(M)
    assert abs(xgrad_mean - kT) < 3 * xgrad_sem, (xgrad_mean, xgrad_sem, kT)

    mv2 = mass * vs**2
    mv2_mean, mv2_sem = mv2.mean(), mv2.std(ddof=1) / np.sqrt(M)
    assert abs(mv2_mean - kT) < 3 * mv2_sem, (mv2_mean, mv2_sem, kT)


def test_baoab_double_well_folded_ks_against_boltzmann():
    barrier, kT, gamma, mass, dt = 3.0, 0.7, 3.0, 2.5, 0.005
    burn_steps = 1500
    stride = 300
    M = 2500
    potential = DoubleWell1D(barrier)

    xs, _vs = _double_well_local_samples(
        potential, kT, gamma, mass, dt, burn_steps, stride, M, "dw_folded_ks"
    )

    cdf_func = _boltzmann_cdf(potential, kT, xmin=-3.0, xmax=3.0)

    def folded_cdf(y):
        return np.clip(2.0 * cdf_func(y) - 1.0, 0.0, 1.0)

    ks = stats.kstest(np.abs(xs), folded_cdf)
    assert ks.pvalue > 0.01, (ks.statistic, ks.pvalue)


# ===========================================================================
# Fix wave 2026-10-01 (fullreview B-analytic Minors 1, 2, 3, 12).
# ===========================================================================


def test_baoab_free_particle_velocity_autocorrelation_pins_gamma_as_rate():
    """Minor 1: gamma is a RATE (ruling R14): <v(t)v(0)>/<v(0)^2> =
    exp(-gamma*t), independent of the mass. A mutant using
    c = exp(-gamma*dt/mass) (gamma as a friction coefficient) leaves the
    stationary distribution unchanged and passed every other fast test
    (probe_friction_mutant.py); with mass=2.5 it gives exp(-0.4*gamma*t).

    For a free particle BAOAB's velocity evolves only through the exact OU
    step, so the regression slope of v(t) on v(0) is exactly c**n in
    expectation; 4000 independent components of one FreeParticle give
    SE ~ 0.93/sqrt(4000) ~ 0.015 at gamma*t = 1 (expected 0.368; the
    coefficient mutant would give 0.670, ~20 SE away).
    """
    kT, gamma, mass, dt = 0.7, 3.0, 2.5, 0.01
    dim = 4000
    n_steps = int(round(1.0 / (gamma * dt)))  # gamma * t = 1 (to rounding)
    backend = AnalyticBackend(
        potential=FreeParticle(dim=dim), integrator="baoab",
        dt=dt, kT=kT, gamma=gamma, mass=mass,
    )
    v0 = derive_rng(_key(0, "vacf_v0"), "mb").standard_normal(dim) * np.sqrt(kT / mass)
    prop = backend.build(MDState(x=np.zeros(dim), v=v0, t=0.0), None, _key(0, "vacf"))
    prop.run(n_steps)
    vt = prop.get_state().v
    slope = float(np.dot(v0, vt) / np.dot(v0, v0))
    expected = np.exp(-gamma * n_steps * dt)
    se = np.sqrt((1.0 - expected**2) / dim)
    assert abs(slope - expected) < 4 * se, (slope, expected, se)
    # and the stationary variance kT/m is kept (noise consistent with rate)
    assert np.var(vt) * mass / kT == pytest.approx(1.0, abs=4 * np.sqrt(2.0 / dim))


def test_overdamped_get_state_before_first_run_has_zero_velocity():
    """Minor 2: overdamped v is all-zeros in EVERY returned MDState,
    including right after build()/set_state()."""
    backend = AnalyticBackend(
        potential=Harmonic(k=1.0, dim=2), integrator="overdamped",
        dt=0.01, kT=1.0, gamma=1.0,
    )
    s = MDState(x=np.array([0.3, -0.2]), v=np.array([1.5, -0.7]), t=0.0)
    prop = backend.build(s, None, _key(0, "od_v"))
    assert np.array_equal(prop.get_state().v, np.zeros(2))
    prop.run(3)
    prop.set_state(s)
    assert np.array_equal(prop.get_state().v, np.zeros(2))


@pytest.mark.parametrize("integrator", ["baoab", "overdamped"])
def test_time_is_t0_plus_step_index_times_dt(integrator):
    """Minor 3 / contract K1: t = t0 + n*dt (multiplied), not accumulated.
    Ten accumulated 0.1 steps give 0.9999999999999999; 10*0.1 == 1.0."""
    backend = AnalyticBackend(
        potential=FreeParticle(dim=1), integrator=integrator,
        dt=0.1, kT=1.0, gamma=1.0,
    )
    prop = backend.build(MDState(x=np.zeros(1), v=np.zeros(1), t=0.0), None, _key(0, "clock"))
    for _ in range(10):
        prop.run(1)
    assert prop.get_state().t == 1.0
    prop.run(90)
    assert prop.get_state().t == 100 * 0.1
    prop.set_state(MDState(x=np.zeros(1), v=np.zeros(1), t=5.0))
    prop.run(7)
    assert prop.get_state().t == 5.0 + 7 * 0.1


@pytest.mark.parametrize("integrator", ["baoab", "overdamped"])
def test_velocity_shape_must_match_positions(integrator):
    """Minor 12: v of shape (3,) used to broadcast silently into (n, 3)."""
    backend = AnalyticBackend(
        potential=LJCluster(n_atoms=2), integrator=integrator,
        dt=0.001, kT=1.0, gamma=1.0,
    )
    x = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    bad = MDState(x=x, v=np.zeros(3), t=0.0)
    with pytest.raises(ValueError, match="shape"):
        backend.build(bad, None, _key(0, "vshape"))
    prop = backend.build(MDState(x=x, v=np.zeros_like(x), t=0.0), None, _key(0, "vshape"))
    with pytest.raises(ValueError, match="shape"):
        prop.set_state(bad)


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_run_rejects_negative_or_non_integer_step_counts(n):
    backend = AnalyticBackend(
        potential=FreeParticle(dim=1), integrator="baoab", dt=0.1, kT=1.0, gamma=1.0
    )
    prop = backend.build(MDState(x=np.zeros(1), v=np.zeros(1), t=0.0), None, _key(0, "nsteps"))
    with pytest.raises(ValueError):
        prop.run(n)
    prop.run(0)  # zero is a valid no-op
    assert prop.get_state().t == 0.0
