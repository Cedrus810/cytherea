"""Tests for cytherea.backends.openmm_backend (task-12-brief.md 12.1-12.6 plus
the controller decisions: integrator seed never 0, the design-section-9
friction rule, and PhysicsConfig-vs-System constraint consistency).

Platforms: everything in the default (not-slow) run uses the Reference or
single-thread CPU platform. The CUDA tests (12.2 and the best-effort CUDA
trajectory check) are marked `slow` so that a plain `pytest -q` never touches
the shared GPU; they additionally skip themselves unless CUDA is available
AND `nvidia-smi` reports no running compute process (global constraint:
never run on the GPU while someone else is using it).

`openmmtools` is only used as a source of a ready-made small molecular
System (alanine dipeptide in vacuum -- Amber ff96 per its docstring; the
plan says "amber14" but the force-field identity is irrelevant for a PES
consistency test) and a tiny water box (for the rigid-water check). Its
import emits a JAX-plugin RuntimeWarning (environment noise, unrelated to
this code), which is suppressed at import time below.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u
import pytest

from cytherea.backends.analytic import AnalyticBackend, DoubleWell2D
from cytherea.backends.base import MDState, NumericalInstabilityError, PhysicsConfig, PotentialBackend
from cytherea.backends.openmm_backend import (
    OpenMMBackend,
    com_distance,
    integrator_seed,
)
from cytherea.backends.pes_suite import pes_consistency_suite
from cytherea.ic.frames import EnsembleFrame, EnsembleFramePool
from cytherea.ic.sampler import EnsembleFrameSampler, InitialState
from cytherea.keys import ShotKey, derive_rng

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from openmmtools import testsystems


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> PhysicsConfig:
    base = dict(
        integrator="verlet",
        dt_ps=0.001,
        temperature_K=300.0,
        friction_per_ps=0.0,
        constraints="none",
        rigid_water=False,
        platform="Reference",
        precision="double",
        deterministic_forces=True,
        purpose="measurement",
    )
    base.update(overrides)
    return PhysicsConfig(**base)


def _ala(constraints=None):
    t = testsystems.AlanineDipeptideVacuum(constraints=constraints)
    x0 = np.asarray(t.positions.value_in_unit(u.nanometer), dtype=np.float64)
    return t.system, t.topology, x0


def _masses(system) -> np.ndarray:
    return np.array(
        [system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())]
    )


def _key(shot_id=0, stage="t12"):
    return ShotKey(global_seed=12, frame_id=0, shot_id=shot_id, stage=stage)


def _ala_probes(x0, n=20):
    """20 probe configurations: the crd geometry plus small deterministic
    Gaussian perturbations (sigma 0.005 nm), drawn via derive_rng."""
    rng = derive_rng(_key(stage="t12_probes"), "probes")
    return [x0] + [x0 + rng.normal(0.0, 0.005, size=x0.shape) for _ in range(n - 1)]


def _mb_velocities(masses, T, key, remove_com=True):
    """Maxwell-Boltzmann velocities (nm/ps) drawn via derive_rng; by default
    with the net momentum removed, as the IC sampler does (a Verlet
    measurement run in a System with a CMMotionRemover rejects ICs with net
    momentum -- review C-M6)."""
    kT = (u.MOLAR_GAS_CONSTANT_R * T * u.kelvin).value_in_unit(u.kilojoule_per_mole)
    rng = derive_rng(key, "velocities")
    m = np.asarray(masses, dtype=np.float64)
    safe = np.where(m > 0.0, m, 1.0)
    v = rng.normal(size=(len(m), 3)) * np.sqrt(kT / safe)[:, None]
    v[m == 0.0] = 0.0
    if remove_com:
        v -= (m[:, None] * v).sum(axis=0) / m.sum()
        v[m == 0.0] = 0.0
    return v


def _gpu_idle_reason() -> str | None:
    """None if the CUDA platform exists and nvidia-smi shows no compute
    process; otherwise a human-readable skip reason."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        return f"CUDA_VISIBLE_DEVICES={visible!r} hides every GPU"
    try:
        openmm.Platform.getPlatformByName("CUDA")
    except Exception:
        return "OpenMM CUDA platform not available"
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found; cannot confirm the GPU is idle"
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return f"nvidia-smi failed: {out.stderr.strip()}"
    if out.stdout.strip():
        return f"GPU busy (compute processes: {out.stdout.strip()})"
    return None


def _double_well_2d_system(pot: DoubleWell2D, kz: float = 1.0e6, mass: float = 1.0):
    """One particle whose (x, y) see DoubleWell2D exactly and whose z is
    pinned by a stiff harmonic term 0.5*kz*z^2 (exactly zero energy and
    force at z=0, so the (x, y) comparison is untouched). Units: the
    analytic numbers are read as kJ/mol and nm."""
    system = openmm.System()
    system.addParticle(mass)
    f = openmm.CustomExternalForce("barrier*(x^2-1)^2 + 0.5*ky*y^2 + 0.5*kz*z^2")
    f.addGlobalParameter("barrier", pot.barrier)
    f.addGlobalParameter("ky", pot.ky)
    f.addGlobalParameter("kz", kz)
    f.addParticle(0, [])
    system.addForce(f)
    return system


# ---------------------------------------------------------------------------
# protocol / basic shape
# ---------------------------------------------------------------------------


def test_backend_satisfies_protocol_and_shapes():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    assert isinstance(b, PotentialBackend)
    assert b.kind == "openmm"
    # review C-M10: only a CUDA backend is GPU-resident
    assert b.gpu_resident is False
    assert OpenMMBackend(
        system, top, _cfg(platform="CUDA", precision="mixed", deterministic_forces=True)
    ).gpu_resident is True
    E, F = b.energy_forces(x0)
    assert isinstance(E, float) and np.isfinite(E)
    assert F.shape == (22, 3) and F.dtype == np.float64


# ---------------------------------------------------------------------------
# R34 follow-up: OpenMMPropagator.dt is the integrator's own step size (ps),
# read-only, resolved from the PhysicsConfig actually passed to build().
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "integrator,extra",
    [
        ("verlet", {}),
        ("langevin_middle", {"friction_per_ps": 0.05}),
        ("nose_hoover", {"friction_per_ps": 1.0, "purpose": "equilibration"}),
    ],
)
def test_r34_propagator_dt_equals_cfg_dt_ps(integrator, extra):
    system, top, x0 = _ala()
    cfg = _cfg(integrator=integrator, dt_ps=0.0025, **extra)
    b = OpenMMBackend(system, top, cfg)
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), None, _key())
    assert isinstance(p.dt, float)
    assert p.dt == pytest.approx(cfg.dt_ps, abs=1e-12)


def test_r34_propagator_dt_reflects_build_time_cfg_not_backend_default():
    """dt is the value the cfg passed to *this* build() call actually
    produced, not some fixed backend-level default (R34)."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(dt_ps=0.001))
    p_default = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), None, _key())
    other_cfg = _cfg(dt_ps=0.002)
    p_other = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), other_cfg, _key())
    assert p_default.dt == pytest.approx(0.001, abs=1e-12)
    assert p_other.dt == pytest.approx(0.002, abs=1e-12)


# ---------------------------------------------------------------------------
# R20 follow-up: EnsembleFrameSampler's IC gate must evaluate the energy
# window at the frame's own box, not the backend's default box.
# ---------------------------------------------------------------------------


def test_r20_ic_gate_energy_evaluated_at_frame_box_not_default():
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    b = OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=True))
    masses = _masses(wb.system)
    box_1_3 = np.diag([1.3, 1.3, 1.3])

    E_1_3, _ = b.energy_forces(x0, box_1_3)
    E_default, _ = b.energy_forces(x0)
    assert abs(E_1_3 - E_default) > 1.0  # sanity: the box genuinely matters here

    v0 = _mb_velocities(masses, 300.0, _key(stage="v_r20"))
    istate = InitialState(
        state=MDState(x=x0, v=v0, t=0.0, box=box_1_3), frame_id=0, meta={}
    )

    frame = EnsembleFrame(
        coordinates=x0,
        box=box_1_3,
        topology_ref="waterbox",
        temperature=300.0,
        weight=1.0,
        source_id="s",
        frame_id=0,
        time=0.0,
    )
    pool = EnsembleFramePool([frame])
    kT = (u.MOLAR_GAS_CONSTANT_R * 300.0 * u.kelvin).value_in_unit(u.kilojoule_per_mole)

    # Window brackets the energy AT box_1_3 but excludes the default-box
    # energy -- correct only if the gate evaluates at the frame's own box.
    lo, hi = E_1_3 - 5.0, E_1_3 + 5.0
    assert not (lo <= E_default <= hi)
    sampler_accepts = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=kT,
        backend=b,
        energy_window=(lo, hi),
        min_pair_dist=None,
        max_redraws=0,
    )
    report = sampler_accepts.validate(istate)
    assert report.ok, report.reasons
    assert report.checks["energy_window"] == pytest.approx(E_1_3, abs=1e-6)

    # Window brackets the default-box energy but excludes the frame's own
    # (1.3 nm) box energy -- must now be rejected, not silently accepted via
    # a stale default-box evaluation.
    lo2, hi2 = E_default - 5.0, E_default + 5.0
    assert not (lo2 <= E_1_3 <= hi2)
    sampler_rejects = EnsembleFrameSampler(
        pool=pool,
        masses=masses,
        kT=kT,
        backend=b,
        energy_window=(lo2, hi2),
        min_pair_dist=None,
        max_redraws=0,
    )
    report2 = sampler_rejects.validate(istate)
    assert not report2.ok
    assert "energy_window" in report2.reasons
    assert report2.checks["energy_window"] == pytest.approx(E_1_3, abs=1e-6)


# ---------------------------------------------------------------------------
# 12.1 PES consistency suite on alanine dipeptide, Reference/double
# ---------------------------------------------------------------------------


def test_12_1_alanine_dipeptide_pes_suite_reference_double():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    probes = _ala_probes(x0)
    assert len(probes) == 20
    # fd_step 1e-5 nm (see task report for the step-size sweep): central
    # differences have truncation error ~h^2 and roundoff ~eps*|E|/h; with
    # |E| ~ 1e2 kJ/mol and max|F| ~ 1e3 kJ/mol/nm both are far below 1e-4.
    rep = pes_consistency_suite(b, probes, fd_step=1e-5, fd_rtol=1e-4)
    print(f"12.1 fd_max_rel_err={rep.fd_max_rel_err:.3e} repeat_bitwise={rep.repeat_bitwise}")
    assert rep.repeat_bitwise
    assert rep.fd_max_rel_err < 1e-4
    assert rep.passed


def test_12_1_alanine_dipeptide_translation_rotation_invariance():
    """Vacuum, no cutoff: E and F must be translation/rotation invariant.
    The suite's invariance errors are relative (|dE|/|E0|, max|dF|/max|F0|,
    ruling R22 #4), so the default inv_rtol=1e-10 applies to kJ/mol-scale
    molecules as well as O(1) toys."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    rep = pes_consistency_suite(b, _ala_probes(x0), fd_step=1e-5, check_invariance=True)
    print(f"12.1 rel translation_err={rep.translation_err:.3e} rel rotation_err={rep.rotation_err:.3e}")
    assert rep.translation_err < 1e-10
    assert rep.rotation_err < 1e-10
    assert rep.passed


def test_12_1b_alanine_dipeptide_nve_drift_suite_reference():
    """pes_consistency_suite NVE path (spec S2) at the suite's default
    strict/double nve_rtol=1e-3 (Verlet, dt=0.1 fs, 2000 steps = 0.2 ps).
    The start is thermal (MB velocities at kT = 300 K, COM removed) and the
    drift is max|E_tot(t) - E_tot(0)| / (n_dof*kT/2), independent of the
    energy zero (P4 measured 1.08e-4 here, n_dof = 63). MDState.v is the
    on-step velocity (ruling R21), so the suite's E_tot = PE + KE is the
    genuine O(dt^2) shadow-energy fluctuation of Verlet, not a half-step
    offset."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(dt_ps=0.0001))
    m = _masses(system)[:, None]
    rep = pes_consistency_suite(
        b, [x0], fd_step=1e-5, nve_steps=2000, masses=m, kT=0.0083144626 * 300.0
    )
    print(f"12.1b suite nve_rel_drift={rep.nve_rel_drift:.3e} n_dof={rep.n_dof}")
    assert rep.passed, rep


# ---------------------------------------------------------------------------
# 12.1s spec S2 sampled mode on the production-precision CPU platform (mixed)
# ---------------------------------------------------------------------------


class _ScaledForceAtom:
    """Negative control: delegates to an OpenMM backend but scales the force
    on one atom (a defect the per-atom FD metric must catch)."""

    def __init__(self, backend, atom, factor):
        self._b, self._atom, self._factor = backend, atom, factor

    def energy_forces(self, x, box=None):
        E, F = self._b.energy_forces(x, box)
        F = np.array(F, copy=True)
        F[self._atom] *= self._factor
        return E, F

    def effective_config(self, cfg=None):
        return self._b.effective_config(cfg)

    def __getattr__(self, name):
        return getattr(self._b, name)


def test_12_1s_sampled_mode_cpu_mixed_passes_and_precision_comes_from_backend():
    """S2: sampled mode on CPU/mixed at fd_step 1e-5 passes FD, invariance and
    the bitwise repeat check. The tolerance row is resolved from
    effective_config() (mixed: fd_rtol 5e-3, inv_rtol 1e-4), not passed in.
    P4 measured noise 4.1e-4, translation 2.9e-7, rotation 1.5e-7 here."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(platform="CPU", precision="mixed"))
    assert b.effective_config()["precision"] == "mixed"
    rep = pes_consistency_suite(b, _ala_probes(x0), mode="sampled", fd_step=1e-5, check_invariance=True)
    print(
        f"12.1s sampled CPU/mixed fd={rep.fd_max_rel_err:.2e} noise={rep.fd_noise_rel:.2e} "
        f"tr={rep.translation_err:.2e} rot={rep.rotation_err:.2e}"
    )
    assert rep.mode == "sampled" and rep.precision == "mixed"
    assert rep.tolerances["fd_rtol"] == 5e-3 and rep.tolerances["inv_rtol"] == 1e-4
    # 20 probes x (16 sampled + up to 4 largest-force atoms) x 3 coordinates
    assert rep.n_fd_coords == 3 * sum(len(a) for a in rep.fd_atoms)
    assert all(16 <= len(a) <= 20 for a in rep.fd_atoms) and len(rep.fd_atoms) == 20
    assert rep.all_finite and rep.repeat_bitwise
    assert rep.passed, rep


def test_12_1s_strict_mode_on_cpu_mixed_refuses_as_unresolved():
    """S2: strict mode always uses the double row; on a mixed platform at
    fd_step 1e-5 the FD noise exceeds it, and the suite must say so
    (fd_unresolved) rather than pass or blame the forces."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(platform="CPU", precision="mixed"))
    rep = pes_consistency_suite(b, _ala_probes(x0), mode="strict", fd_step=1e-5)
    assert not rep.passed
    assert any(r.startswith("fd_unresolved") for r in rep.reasons), rep.reasons
    with pytest.raises(ValueError):
        pes_consistency_suite(b, _ala_probes(x0), mode="strict", precision="mixed")


def test_12_1s_sampled_mode_cpu_mixed_catches_a_five_percent_force_defect():
    """S2 negative control: the relaxed mixed tolerances still catch a 5 %
    error on one atom's force (all 22 atoms sampled so the defect is hit)."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(platform="CPU", precision="mixed"))
    rep = pes_consistency_suite(
        _ScaledForceAtom(b, atom=3, factor=1.05), _ala_probes(x0), mode="sampled", n_fd_atoms=22
    )
    assert rep.precision == "mixed"
    assert not rep.passed
    assert any(r.startswith("fd") for r in rep.reasons), rep.reasons


def test_12_1s_sampled_mode_cpu_mixed_nve_drift():
    """S2 NVE on CPU/mixed: thermal start at 300 K, drift normalised by
    n_dof*kT/2 against the mixed nve_rtol=1e-2 (dt 0.1 fs, 2000 steps;
    measured 1.08e-4, n_dof 63)."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(platform="CPU", precision="mixed", dt_ps=0.0001))
    rep = pes_consistency_suite(
        b, _ala_probes(x0), mode="sampled", nve_steps=2000,
        masses=_masses(system)[:, None], kT=0.0083144626 * 300.0,
    )
    print(f"12.1s sampled CPU/mixed nve_rel_drift={rep.nve_rel_drift:.3e}")
    assert rep.n_dof == 3 * 22 - 3
    assert rep.nve_rel_drift < 1e-3
    assert rep.passed, rep


def _energy_error(b, x0, masses, dt, integrator="verlet", t_total=0.2, n_samples=100):
    """max |E_tot(t) - E_tot(0)| along an NVE run, KE straight from
    get_state().v (the on-step velocity)."""
    cfg = _cfg(dt_ps=dt, integrator=integrator, friction_per_ps=0.0)
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), cfg, _key())
    E0, _ = b.energy_forces(x0)
    n = int(round(t_total / dt))
    err = 0.0
    for _ in range(n_samples):
        p.run(n // n_samples)
        s = p.get_state()
        PE, _ = b.energy_forces(s.x)
        err = max(err, abs(PE + 0.5 * np.sum(masses[:, None] * s.v**2) - E0))
    return err, abs(E0)


@pytest.mark.parametrize("integrator", ["verlet", "langevin_middle"])
def test_12_1c_nve_energy_conservation_second_order(integrator):
    """With on-step velocities the energy error of Verlet is O(dt^2):
    halving dt divides it by ~4 (a half-step velocity would give ~2).
    LangevinMiddle at friction 0 is the same leapfrog scheme, so the same
    boundary conversion must make it second order too."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    m = _masses(system)
    e2, E0 = _energy_error(b, x0, m, 0.0002, integrator)
    e1, _ = _energy_error(b, x0, m, 0.0001, integrator)
    print(f"12.1c {integrator} |dE|: dt=0.2fs {e2:.3e}, dt=0.1fs {e1:.3e}, ratio {e2 / e1:.2f}, |E0|={E0:.2f}")
    assert e1 / E0 < 1e-4
    assert 3.0 < e2 / e1 < 5.0


# ---------------------------------------------------------------------------
# 12.2 CUDA + DeterministicForces: bitwise repeat forces; provenance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_12_2_cuda_deterministic_forces_bitwise_and_provenance():
    reason = _gpu_idle_reason()
    if reason is not None:
        pytest.skip(reason)
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(platform="CUDA", precision="mixed", deterministic_forces=True))
    x = _ala_probes(x0)[3]
    E0, F0 = b.energy_forces(x)
    for _ in range(10):
        E, F = b.energy_forces(x)
        assert E == E0
        assert np.array_equal(F, F0)
    # a second, independent backend/Context on the same GPU also agrees bitwise
    b2 = OpenMMBackend(system, top, _cfg(platform="CUDA", precision="mixed", deterministic_forces=True))
    E2, F2 = b2.energy_forces(x)
    assert E2 == E0 and np.array_equal(F2, F0)
    prov = b.provenance()
    print(f"12.2 provenance={prov}")
    assert prov["platform"] == "CUDA"
    assert prov["precision"] == "mixed"
    assert prov["deterministic_forces"] == "true"
    assert prov["openmm_version"] == openmm.version.full_version
    assert isinstance(prov["gpu_name"], str) and prov["gpu_name"]


@pytest.mark.slow
def test_12_2b_cuda_langevin_trajectory_bitwise_best_effort():
    """Design section 7: trajectory-level reproducibility on CUDA is best
    effort (DeterministicForces=true, same GPU); checked here for a short run."""
    reason = _gpu_idle_reason()
    if reason is not None:
        pytest.skip(reason)
    system, top, x0 = _ala()
    cfg = _cfg(
        integrator="langevin_middle", friction_per_ps=0.1, platform="CUDA",
        precision="mixed", deterministic_forces=True,
    )
    b = OpenMMBackend(system, top, cfg)
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    runs = []
    for _ in range(2):
        p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=5))
        p.run(1000)
        runs.append(p.get_state())
    assert np.array_equal(runs[0].x, runs[1].x)
    assert np.array_equal(runs[0].v, runs[1].v)


# ---------------------------------------------------------------------------
# 12.3 friction rule (design section 9)
# ---------------------------------------------------------------------------


def test_12_3_measurement_with_friction_1_rejected_at_construction():
    system, top, _ = _ala()
    with pytest.raises(ValueError, match="friction"):
        OpenMMBackend(
            system, top,
            _cfg(integrator="langevin_middle", friction_per_ps=1.0, purpose="measurement"),
        )


@pytest.mark.parametrize(
    "integrator,friction,purpose",
    [
        ("langevin_middle", 0.1, "measurement"),  # boundary: gamma <= 0.1 allowed
        ("langevin_middle", 0.0, "measurement"),
        ("langevin_middle", 1.0, "equilibration"),
        ("langevin_middle", 5.0, "equilibration"),
        ("verlet", 0.0, "measurement"),
        ("nose_hoover", 1.0, "equilibration"),  # collision frequency, not a friction
    ],
)
def test_friction_rule_allowed_cases(integrator, friction, purpose):
    system, top, _ = _ala()
    OpenMMBackend(system, top, _cfg(integrator=integrator, friction_per_ps=friction, purpose=purpose))


def test_friction_rule_also_enforced_on_build_cfg_override():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    bad = _cfg(integrator="langevin_middle", friction_per_ps=0.5, purpose="measurement")
    with pytest.raises(ValueError, match="friction"):
        b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), bad, _key())


@pytest.mark.parametrize(
    "overrides",
    [
        dict(dt_ps=0.0),
        dict(dt_ps=-0.001),
        dict(integrator="brownian"),
        dict(platform="OpenCL"),
        dict(purpose="production"),
        dict(integrator="langevin_middle", friction_per_ps=-0.1),
        dict(integrator="langevin_middle", temperature_K=0.0, friction_per_ps=0.1),
        dict(integrator="nose_hoover", friction_per_ps=0.0, purpose="equilibration"),
        dict(integrator="nose_hoover", friction_per_ps=1.0, purpose="measurement"),  # R21
        dict(platform="CUDA", precision="mixed", deterministic_forces=False),  # R22 #6
        dict(platform="Reference", precision="mixed"),  # Reference is double-only
        dict(platform="CPU", precision="double"),  # CPU cannot provide double
        # review C-M7: single precision is not accurate enough for NVE
        # measurement (and PROJECTION_TOLERANCE 1e-10 is unreachable in it)
        dict(platform="CUDA", precision="single", deterministic_forces=True),
        dict(platform="CPU", precision="single"),
    ],
)
def test_invalid_config_rejected(overrides):
    system, top, _ = _ala()
    with pytest.raises(ValueError):
        OpenMMBackend(system, top, _cfg(**overrides))


# ---------------------------------------------------------------------------
# constraint consistency (PhysicsConfig describes how the System was built)
# ---------------------------------------------------------------------------


def test_constraints_none_but_system_constrained_rejected():
    system, top, _ = _ala(constraints=app.HBonds)
    assert system.getNumConstraints() > 0
    with pytest.raises(ValueError, match="constraint"):
        OpenMMBackend(system, top, _cfg(constraints="none"))


def test_constraints_hbonds_matches_hbond_system():
    system, top, _ = _ala(constraints=app.HBonds)
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds"))
    assert b.provenance()["num_constraints"] == system.getNumConstraints() == 12


@pytest.mark.parametrize("claimed", ["hbonds", "allbonds"])
def test_constraints_claimed_but_system_unconstrained_rejected(claimed):
    system, top, _ = _ala(constraints=None)
    with pytest.raises(ValueError, match="constraint"):
        OpenMMBackend(system, top, _cfg(constraints=claimed))


def test_constraints_allbonds_vs_hbonds_mismatch_rejected():
    system, top, _ = _ala(constraints=app.AllBonds)
    with pytest.raises(ValueError, match="constraint"):
        OpenMMBackend(system, top, _cfg(constraints="hbonds"))
    OpenMMBackend(system, top, _cfg(constraints="allbonds"))


def test_rigid_water_consistency():
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    with pytest.raises(ValueError, match="rigid"):
        OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=False))
    OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=True))
    wb_flex = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=False)
    with pytest.raises(ValueError, match="rigid"):
        OpenMMBackend(wb_flex.system, wb_flex.topology, _cfg(constraints="none", rigid_water=True))


# ---------------------------------------------------------------------------
# 12.4 DoubleWell2D as an OpenMM CustomExternalForce vs AnalyticBackend
# ---------------------------------------------------------------------------


def test_12_4_double_well_2d_customexternalforce_matches_analytic():
    pot = DoubleWell2D(barrier=5.0, ky=3.0)
    ana = AnalyticBackend(pot, integrator="baoab", dt=0.001, kT=1.0, gamma=1.0)
    omm = OpenMMBackend(_double_well_2d_system(pot), None, _cfg())
    rng = derive_rng(_key(stage="t12_dw2d"), "probes")
    max_dE = 0.0
    max_dF = 0.0
    for xy in rng.uniform(-2.0, 2.0, size=(50, 2)):
        Ea, Fa = ana.energy_forces(xy)
        Eo, Fo = omm.energy_forces(np.array([[xy[0], xy[1], 0.0]]))
        max_dE = max(max_dE, abs(Eo - Ea))
        max_dF = max(max_dF, float(np.max(np.abs(Fo[0, :2] - Fa))))
        assert Fo[0, 2] == 0.0
    print(f"12.4 max|dE|={max_dE:.3e} max|dF|={max_dF:.3e}")
    assert max_dE < 1e-10
    assert max_dF < 1e-10


def test_12_4b_double_well_openmm_passes_pes_suite():
    pot = DoubleWell2D(barrier=5.0, ky=3.0)
    omm = OpenMMBackend(_double_well_2d_system(pot, kz=10.0), None, _cfg())
    rng = derive_rng(_key(stage="t12_dw2d"), "suite")
    probes = [np.array([[a, b_, c]]) for a, b_, c in rng.uniform(-2.0, 2.0, size=(20, 3))]
    rep = pes_consistency_suite(omm, probes, fd_step=1e-5, fd_rtol=1e-6)
    assert rep.passed, rep


# ---------------------------------------------------------------------------
# 12.5 same rng_key -> bitwise identical Langevin trajectory (Reference, CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,precision", [("Reference", "double"), ("CPU", "mixed")]
)
def test_12_5_langevin_same_key_bitwise_identical(platform, precision):
    system, top, x0 = _ala()
    cfg = _cfg(
        integrator="langevin_middle", friction_per_ps=0.1, platform=platform,
        precision=precision, deterministic_forces=True,
    )
    b = OpenMMBackend(system, top, cfg)
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    s0 = MDState(x=x0, v=v0, t=0.0)

    def traj(key):
        p = b.build(s0, None, key)
        frames = []
        for _ in range(5):
            p.run(100)
            frames.append(p.get_state())
        return frames

    a = traj(_key(shot_id=1))
    c = traj(_key(shot_id=1))
    d = traj(_key(shot_id=2))
    for fa, fc in zip(a, c):
        assert np.array_equal(fa.x, fc.x)
        assert np.array_equal(fa.v, fc.v)
        assert fa.t == fc.t
    # the key actually matters (the noise is seeded from it)
    assert not np.array_equal(a[-1].x, d[-1].x)
    assert a[-1].t == pytest.approx(500 * 0.001)


def test_seed_derived_from_key_and_never_zero():
    seeds = [integrator_seed(_key(shot_id=i)) for i in range(2000)]
    assert all(1 <= s <= 2**31 - 1 for s in seeds)
    assert len(set(seeds)) > 1990  # essentially all distinct
    assert integrator_seed(_key(shot_id=7)) == integrator_seed(_key(shot_id=7))
    # and it is what actually lands on the integrator
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1))
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), None, _key(shot_id=7))
    assert p.integrator.getRandomNumberSeed() == integrator_seed(_key(shot_id=7)) != 0
    assert p.seed == integrator_seed(_key(shot_id=7))


def test_integrator_types():
    system, top, x0 = _ala()
    s = MDState(x=x0, v=np.zeros_like(x0), t=0.0)
    for name, cls in [
        ("verlet", openmm.VerletIntegrator),
        ("langevin_middle", openmm.LangevinMiddleIntegrator),
        ("nose_hoover", openmm.NoseHooverIntegrator),
    ]:
        b = OpenMMBackend(
            system, top, _cfg(integrator=name, friction_per_ps=0.1, purpose="equilibration")
        )
        p = b.build(s, None, _key())
        assert isinstance(p.integrator, cls)
        assert p.integrator.getStepSize().value_in_unit(u.picosecond) == 0.001
        assert b.provenance()["integrator"] == name


def test_set_then_get_state_roundtrip_velocity():
    """R21: set_state(v) stores v_half = v - F dt/2m, get_state returns
    v_half + F dt/2m; an immediate round trip returns the same on-step v."""
    system, top, x0 = _ala()
    m = _masses(system)
    for integ in ["verlet", "langevin_middle"]:
        b = OpenMMBackend(system, top, _cfg(integrator=integ, friction_per_ps=0.1))
        for i, x in enumerate(_ala_probes(x0, n=5)):
            v0 = _mb_velocities(m, 300.0, _key(shot_id=i, stage="v"))
            p = b.build(MDState(x=x, v=v0, t=0.25), None, _key())
            s = p.get_state()
            rel = np.max(np.abs(s.v - v0)) / np.max(np.abs(v0))
            print(f"roundtrip {integ} probe {i}: max rel |dv| = {rel:.3e}")
            assert rel < 1e-10
            assert np.array_equal(s.x, x) and s.t == 0.25 and s.box is None


def test_verlet_kinetic_energy_matches_openmm_time_centred():
    """OpenMM's State.getKineticEnergy() for leapfrog Verlet is the
    time-centred KE; the KE from get_state().v must match it."""
    system, top, x0 = _ala()
    m = _masses(system)
    b = OpenMMBackend(system, top, _cfg(integrator="verlet"))
    v0 = _mb_velocities(m, 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key())
    worst = 0.0
    for _ in range(10):
        p.run(20)
        s = p.get_state()
        ke_ours = 0.5 * float(np.sum(m[:, None] * s.v**2))
        ke_omm = p.context.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            u.kilojoule_per_mole
        )
        worst = max(worst, abs(ke_ours - ke_omm) / ke_omm)
    print(f"KE get_state().v vs OpenMM getKineticEnergy: max rel diff = {worst:.3e}")
    assert worst < 1e-10


def test_get_state_does_not_perturb_trajectory():
    """Observables are read via get_state between run() chunks (R3); that
    must leave the trajectory bitwise unchanged."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=3))
    p.run(300)
    one = p.get_state()
    q = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=3))
    for _ in range(3):
        q.run(100)
        q.get_state()
    three = q.get_state()
    assert np.array_equal(one.x, three.x) and np.array_equal(one.v, three.v)


def test_checkpoint_resume_close_and_deterministic():
    """Restoring through MDState re-derives v_half = v - F dt/2m, which can
    differ from the original v_half by rounding, so a resumed run matches
    the continuous one to rounding level (not bitwise); resuming twice from
    the same MDState is bitwise identical."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(integrator="verlet"))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.25), None, _key())
    p.run(50)
    mid = p.get_state()
    p.run(50)
    end = p.get_state()
    ends = []
    for _ in range(2):
        q = b.build(MDState(x=x0, v=v0, t=0.0), None, _key())
        q.set_state(mid)
        q.run(50)
        ends.append(q.get_state())
    assert np.array_equal(ends[0].x, ends[1].x) and np.array_equal(ends[0].v, ends[1].v)
    dx = np.max(np.abs(ends[0].x - end.x))
    dv = np.max(np.abs(ends[0].v - end.v)) / np.max(np.abs(end.v))
    print(f"resume vs continuous: max|dx|={dx:.3e} nm, max rel|dv|={dv:.3e}")
    assert dx < 1e-10 and dv < 1e-9
    assert ends[0].t == pytest.approx(0.25 + 0.1)


def test_constrained_set_state_applies_velocity_constraints():
    system, top, x0 = _ala(constraints=app.HBonds)
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds", integrator="verlet"))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key())
    st = p.context.getState(getVelocities=True, getPositions=True)
    v = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
    x = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    worst = 0.0
    for k in range(system.getNumConstraints()):
        i, j, _ = system.getConstraintParameters(k)
        r = x[i] - x[j]
        worst = max(worst, abs(float(np.dot(r, v[i] - v[j]))) / float(np.linalg.norm(r)))
    assert worst < 1e-4  # relative-velocity component along each constrained bond ~ 0


def _constraint_violation(system, x, v) -> float:
    """max over constraints of |r_ij . (v_i - v_j)| / |r_ij| (nm/ps): the
    relative velocity along each constrained bond, which must be ~0."""
    worst = 0.0
    for k in range(system.getNumConstraints()):
        i, j, _ = system.getConstraintParameters(k)
        r = x[i] - x[j]
        worst = max(worst, abs(float(np.dot(r, v[i] - v[j]))) / float(np.linalg.norm(r)))
    return worst


def _equilibrated_constrained_state(b, system, x0, n_steps=50):
    """Run a short constrained Verlet segment so x satisfies the
    constraints, and return (propagator, state)."""
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=4))
    p.run(n_steps)
    return p, p.get_state()


@pytest.mark.parametrize("integrator,dt", [("verlet", 0.001), ("verlet", 0.002), ("langevin_middle", 0.002)])
def test_constrained_roundtrip_and_get_state_satisfies_constraints(integrator, dt):
    """R28: with HBonds, get_state().v is projected onto the velocity
    constraints, so (a) it satisfies them to solver tolerance and (b) a
    set_state/get_state round trip of a constraint-satisfying v is exact to
    rounding (the projection is idempotent and linear)."""
    system, top, x0 = _ala(constraints=app.HBonds)
    b = OpenMMBackend(
        system, top, _cfg(constraints="hbonds", integrator=integrator, dt_ps=dt, friction_per_ps=0.1)
    )
    _, s = _equilibrated_constrained_state(b, system, x0)
    viol = _constraint_violation(system, s.x, s.v)
    q = b.build(s, None, _key(shot_id=9))
    s2 = q.get_state()
    rel = np.max(np.abs(s2.v - s.v)) / np.max(np.abs(s.v))
    print(f"R28 {integrator} dt={dt}: get_state().v constraint violation={viol:.3e} nm/ps, "
          f"round-trip max rel |dv|={rel:.3e}")
    # get_state projects with PROJECTION_TOLERANCE=1e-10; thermal |v| is a few
    # nm/ps, so tolerance level is ~1e-10..1e-9 nm/ps (was 0.07-0.16 nm/ps
    # without the projection)
    assert viol < 1e-9
    assert rel < 1e-10
    assert np.array_equal(s2.x, s.x)


@pytest.mark.parametrize("dt", [0.001, 0.002])
def test_constrained_verlet_kinetic_energy_matches_openmm(dt):
    """OpenMM's Verlet getKineticEnergy() is the time-centred KE of the
    constrained velocities v_half + F dt/2m projected onto the constraints;
    get_state().v is the same quantity, so the two agree to the
    constraint-solver tolerance. OpenMM projects its KE velocities with the
    integrator's CONSTRAINT_TOLERANCE (1e-5), so agreement is only
    guaranteed at that level; we bound the relative KE difference by 1e-6
    (the solver on Reference converges far below tol: measured ~1e-12,
    vs 3e-4 / 1.4e-3 at 1 / 2 fs without the projection)."""
    system, top, x0 = _ala(constraints=app.HBonds)
    m = _masses(system)
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds", integrator="verlet", dt_ps=dt))
    p, _ = _equilibrated_constrained_state(b, system, x0)
    worst = 0.0
    for _ in range(10):
        p.run(20)
        s = p.get_state()
        ke_ours = 0.5 * float(np.sum(m[:, None] * s.v**2))
        ke_omm = p.context.getState(getEnergy=True).getKineticEnergy().value_in_unit(
            u.kilojoule_per_mole
        )
        worst = max(worst, abs(ke_ours - ke_omm) / ke_omm)
    print(f"R28 constrained Verlet dt={dt}: KE get_state().v vs OpenMM max rel diff = {worst:.3e}")
    assert worst < 1e-6


@pytest.mark.parametrize("integrator", ["verlet", "langevin_middle"])
def test_constrained_get_state_does_not_perturb_trajectory(integrator):
    """The projection in get_state temporarily overwrites the Context's
    velocities and restores v_half; the continuing trajectory must be
    bitwise unchanged."""
    system, top, x0 = _ala(constraints=app.HBonds)
    b = OpenMMBackend(
        system, top, _cfg(constraints="hbonds", integrator=integrator, dt_ps=0.002, friction_per_ps=0.1)
    )
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=3))
    p.run(300)
    one = p.get_state()
    q = b.build(MDState(x=x0, v=v0, t=0.0), None, _key(shot_id=3))
    for _ in range(3):
        q.run(100)
        q.get_state()
        q.get_state()
    three = q.get_state()
    assert np.array_equal(one.x, three.x) and np.array_equal(one.v, three.v)


class _FailOnceContext:
    """Thin proxy around an openmm.Context that delegates everything but
    raises RuntimeError on the first call of `fail_method` (a transient
    OpenMM failure). Records the call sequence so a test can prove the
    failure happened after the Context had already been mutated."""

    def __init__(self, ctx, fail_method):
        self._ctx = ctx
        self._fail_method = fail_method
        self.calls = []
        self.failed = False

    def __getattr__(self, name):
        attr = getattr(self._ctx, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            self.calls.append(name)
            if name == self._fail_method and not self.failed:
                self.failed = True
                raise RuntimeError(f"injected failure in {name}")
            return attr(*args, **kwargs)

        return wrapper


def _raw_context_state(ctx):
    """Positions, velocities (raw, as stored), box and time straight from
    the Context, for bitwise before/after comparison."""
    st = ctx.getState(getPositions=True, getVelocities=True)
    return (
        np.array(st.getPositions(asNumpy=True).value_in_unit(u.nanometer)),
        np.array(st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)),
        np.array(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)),
        st.getTime().value_in_unit(u.picosecond),
    )


def _assert_raw_equal(a, b):
    for ai, bi in zip(a[:3], b[:3]):
        assert np.array_equal(ai, bi)
    assert a[3] == b[3]


def test_get_state_projection_failure_restores_context():
    """Round 3: if applyVelocityConstraints raises inside get_state's
    projection, the Context's v_half is restored (bitwise) and the original
    exception propagates."""
    system, top, x0 = _ala(constraints=app.HBonds)
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds", integrator="verlet"))
    p, _ = _equilibrated_constrained_state(b, system, x0)
    real = p.context
    before = _raw_context_state(real)
    proxy = _FailOnceContext(real, "applyVelocityConstraints")
    p.context = proxy
    with pytest.raises(RuntimeError, match="injected failure"):
        p.get_state()
    p.context = real
    assert "setVelocities" in proxy.calls[: proxy.calls.index("applyVelocityConstraints")]
    _assert_raw_equal(_raw_context_state(real), before)
    # and the propagator is still fully usable afterwards
    p.get_state()
    _assert_raw_equal(_raw_context_state(real), before)


@pytest.mark.parametrize("system_name", ["waterbox_periodic_rigid", "ala_nonperiodic"])
@pytest.mark.parametrize("fail_method", ["applyVelocityConstraints", "setVelocities", "setTime"])
def test_set_state_failure_restores_context(fail_method, system_name):
    """Round 3: set_state is atomic -- a failure after the first Context
    mutation restores the previous positions, velocities, box and time
    (bitwise) and re-raises."""
    if system_name == "waterbox_periodic_rigid":
        wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
        system, top = wb.system, wb.topology
        x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
        b = OpenMMBackend(system, top, _cfg(constraints="none", rigid_water=True))
        new_box = np.diag([1.3, 1.3, 1.3])
    else:
        system, top, x0 = _ala()
        b = OpenMMBackend(system, top, _cfg())
        new_box = None
    m = _masses(system)
    p = b.build(MDState(x=x0, v=_mb_velocities(m, 300.0, _key(stage="v")), t=0.5), None, _key())
    p.run(10)
    real = p.context
    before = _raw_context_state(real)
    new = MDState(
        x=x0 + 0.01,
        v=_mb_velocities(m, 300.0, _key(shot_id=1, stage="v")),
        t=7.0,
        box=new_box,
    )
    proxy = _FailOnceContext(real, fail_method)
    p.context = proxy
    with pytest.raises(RuntimeError, match="injected failure"):
        p.set_state(new)
    p.context = real
    first_fail = proxy.calls.index(fail_method)
    assert "setPositions" in proxy.calls[:first_fail]  # the Context had been mutated
    _assert_raw_equal(_raw_context_state(real), before)


def test_energy_forces_uses_given_box():
    """R20: energy_forces(x, box) evaluates in that box, equal to the
    propagator's PE/forces for a state with a non-default box; box=None
    means the System default (no stale box from a previous call)."""
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    b = OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=True))
    box = np.diag([1.3, 1.3, 1.3])
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0, box=box), None, _key())
    s = p.get_state()
    st = p.context.getState(getEnergy=True, getForces=True)
    pe_prop = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    f_prop = st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)
    E_box, F_box = b.energy_forces(s.x, s.box)
    E_def, _ = b.energy_forces(s.x)
    print(f"box 1.3 nm: propagator PE={pe_prop:.4f}, energy_forces(x, box)={E_box:.4f}, "
          f"energy_forces(x) [default 1.2 nm box]={E_def:.4f}")
    assert abs(E_box - pe_prop) <= 1e-12 * abs(pe_prop)
    assert np.max(np.abs(F_box - f_prop)) <= 1e-12 * np.max(np.abs(f_prop))
    assert abs(E_def - pe_prop) > 1.0  # the box genuinely matters here
    assert b.energy_forces(s.x)[0] == E_def  # box=None -> default again, not stale


def test_energy_forces_box_on_nonperiodic_system_rejected():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    with pytest.raises(ValueError, match="periodic"):
        b.energy_forces(x0, np.diag([3.0, 3.0, 3.0]))


def test_analytic_backend_accepts_box_argument():
    ana = AnalyticBackend(DoubleWell2D(5.0, 3.0), integrator="baoab", dt=0.001, kT=1.0, gamma=1.0)
    x = np.array([0.3, -0.2])
    assert ana.energy_forces(x, None)[0] == ana.energy_forces(x)[0]
    assert ana.energy_forces(x, box=np.eye(3))[0] == ana.energy_forces(x)[0]


def test_periodic_state_carries_box():
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    b = OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=True))
    box = np.diag([1.3, 1.3, 1.3])
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0, box=box), None, _key())
    s = p.get_state()
    assert s.box.shape == (3, 3) and np.allclose(s.box, box)


def test_provenance_fields_reference():
    system, top, _ = _ala()
    prov = OpenMMBackend(
        system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1, dt_ps=0.002)
    ).provenance()
    for k in [
        "openmm_version", "platform", "precision", "deterministic_forces", "gpu_name",
        "num_constraints", "integrator", "friction_per_ps", "dt_ps", "temperature_K",
        "constraints", "rigid_water", "purpose",
    ]:
        assert k in prov, k
    assert prov["platform"] == "Reference" and prov["precision"] == "double"
    assert prov["gpu_name"] is None
    assert prov["friction_per_ps"] == 0.1 and prov["dt_ps"] == 0.002
    assert prov["num_constraints"] == 0
    assert prov["constraint_tolerance"] == 1e-5


def test_constraint_tolerance_set_on_integrators():
    system, top, x0 = _ala(constraints=app.HBonds)
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds", integrator="langevin_middle", friction_per_ps=0.1))
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0), None, _key())
    assert p.integrator.getConstraintTolerance() == b.provenance()["constraint_tolerance"] == 1e-5


def test_cpu_deterministic_uses_single_thread():
    system, top, _ = _ala()
    prov = OpenMMBackend(
        system, top, _cfg(platform="CPU", precision="mixed", deterministic_forces=True)
    ).provenance()
    assert prov["platform"] == "CPU"
    assert prov["cpu_threads"] == "1"
    assert prov["deterministic_forces"] == "true"


# ---------------------------------------------------------------------------
# 12.6 com_distance across a periodic boundary
# ---------------------------------------------------------------------------


def _wrap_per_atom(x, L):
    """Wrap every atom independently into [0, L) -- this splits a molecule
    that straddles a face, which is exactly what com_distance must undo."""
    return x - L * np.floor(x / L)


def test_12_6_com_distance_continuous_across_boundary():
    L = 3.0
    box = np.diag([L, L, L])
    # molecule A: 3 atoms with extent ~0.3 nm; molecule B: 2 atoms, fixed
    mol_a = np.array([[0.0, 0.0, 0.0], [0.15, 0.05, 0.0], [0.3, -0.05, 0.02]])
    mol_b = np.array([[1.0, 1.5, 1.5], [1.1, 1.5, 1.5]])
    masses = np.array([12.0, 1.0, 16.0, 14.0, 14.0])
    idx_a, idx_b = np.array([0, 1, 2]), np.array([3, 4])

    def ref_dist(x):
        ca = (masses[idx_a, None] * x[idx_a]).sum(0) / masses[idx_a].sum()
        cb = (masses[idx_b, None] * x[idx_b]).sum(0) / masses[idx_b].sum()
        d = ca - cb
        d -= L * np.round(d / L)
        return float(np.linalg.norm(d))

    shifts = np.linspace(2.5, 3.2, 701)  # A slides in +x through the face x=L
    d_prev = None
    naive_jumps = []
    max_err = 0.0
    max_jump_excess = 0.0
    for s in shifts:
        x = np.vstack([mol_a + np.array([s, 1.4, 1.5]), mol_b])
        xw = _wrap_per_atom(x, L)
        d = com_distance(MDState(x=xw, v=np.zeros_like(xw), t=0.0, box=box), idx_a, idx_b, masses)
        max_err = max(max_err, abs(d - ref_dist(x)))
        if d_prev is not None:
            # physical motion per frame is 1e-3 nm; anything beyond that is a jump
            max_jump_excess = max(max_jump_excess, abs(d - d_prev) - 1.0e-3)
        d_prev = d
        # naive (no make-whole, no min-image) COM for the negative control
        ca = (masses[idx_a, None] * xw[idx_a]).sum(0) / masses[idx_a].sum()
        cb = (masses[idx_b, None] * xw[idx_b]).sum(0) / masses[idx_b].sum()
        naive_jumps.append(float(np.linalg.norm(ca - cb)))
    print(f"12.6 max|d-ref|={max_err:.3e} max jump excess={max_jump_excess:.3e}")
    assert max_err < 1e-6
    assert max_jump_excess < 1e-6
    # negative control: the naive per-atom-wrapped COM does jump
    assert np.max(np.abs(np.diff(naive_jumps))) > 0.1

    # tight crossing: atom 0 just below vs just above the face
    eps = 1e-9
    before = np.vstack([mol_a + np.array([L - eps, 1.4, 1.5]), mol_b])
    after = np.vstack([mol_a + np.array([L + eps, 1.4, 1.5]), mol_b])
    d0 = com_distance(MDState(_wrap_per_atom(before, L), np.zeros((5, 3)), 0.0, box), idx_a, idx_b, masses)
    d1 = com_distance(MDState(_wrap_per_atom(after, L), np.zeros((5, 3)), 0.0, box), idx_a, idx_b, masses)
    assert abs(d1 - d0) < 1e-6


def test_com_distance_min_image_between_coms_and_no_box():
    L = 2.0
    box = np.diag([L, L, L])
    x = np.array([[0.1, 1.0, 1.0], [1.9, 1.0, 1.0]])
    m = np.array([1.0, 1.0])
    s = MDState(x=x, v=np.zeros_like(x), t=0.0, box=box)
    assert com_distance(s, np.array([0]), np.array([1]), m) == pytest.approx(0.2, abs=1e-12)
    s_nobox = MDState(x=x, v=np.zeros_like(x), t=0.0, box=None)
    assert com_distance(s_nobox, np.array([0]), np.array([1]), m) == pytest.approx(1.8, abs=1e-12)


def test_com_distance_group_too_large_for_box_rejected():
    L = 2.0
    box = np.diag([L, L, L])
    m = np.ones(4)
    # group A: extent 0.8 nm = 0.4 L along x -> fine
    x_ok = np.array([[0.1, 1.0, 1.0], [0.9, 1.0, 1.0], [1.5, 1.5, 1.5], [1.6, 1.5, 1.5]])
    com_distance(MDState(x_ok, np.zeros((4, 3)), 0.0, box), np.array([0, 1]), np.array([2, 3]), m)
    # group A: a chain of 3 atoms spanning 0.95 nm = 0.475 L after unwrapping -> reject
    x_bad = np.array([[0.1, 1.0, 1.0], [0.6, 1.0, 1.0], [1.05, 1.0, 1.0], [1.6, 1.5, 1.5]])
    with pytest.raises(ValueError, match="extent"):
        com_distance(MDState(x_bad, np.zeros((4, 3)), 0.0, box), np.array([0, 1, 2]), np.array([3]), m)


def test_com_distance_triclinic_reduced_form_accepted_other_rejected():
    """Review C-M8: triclinic boxes are supported in OpenMM's reduced form
    (was: every triclinic box rejected); anything else is rejected."""
    x = np.array([[0.1, 0.1, 0.1], [1.9, 0.1, 0.1]])
    box = np.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]])
    assert com_distance(MDState(x, x, 0.0, box), np.array([0]), np.array([1]), np.ones(2)) == pytest.approx(0.2)
    bad = np.array([[2.0, 0.0, 0.0], [1.5, 2.0, 0.0], [0.0, 0.0, 2.0]])  # |bx| > ax/2
    with pytest.raises(ValueError, match="reduced"):
        com_distance(MDState(x, x, 0.0, bad), np.array([0]), np.array([1]), np.ones(2))


def test_com_distance_on_openmm_state():
    """com_distance works on a Propagator's get_state() output (nm, with box)."""
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=True)
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    m = _masses(wb.system)
    b = OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=True))
    vecs = wb.system.getDefaultPeriodicBoxVectors()
    box = np.array([[v[i].value_in_unit(u.nanometer) for i in range(3)] for v in vecs])
    p = b.build(MDState(x=x0, v=np.zeros_like(x0), t=0.0, box=box), None, _key())
    s = p.get_state()
    d = com_distance(s, np.array([0, 1, 2]), np.array([3, 4, 5]), m)
    assert np.isfinite(d) and 0.0 < d <= np.sqrt(3) * box[0, 0] / 2


# ---------------------------------------------------------------------------
# Full-review fix wave (fullreview-C-openmm.md): I1 System-resident
# stochastic forces, I2/K8 provenance of the cfg actually built, I3 build
# timing/caching, Minors M1-M11.
# ---------------------------------------------------------------------------


def _waterbox(edge=1.8, cutoff=0.7, constrained=True):
    w = testsystems.WaterBox(box_edge=edge * u.nanometer, cutoff=cutoff * u.nanometer, constrained=constrained)
    return w.system, w.topology, np.asarray(w.positions.value_in_unit(u.nanometer))


def _stochastic_force(kind):
    T = 300.0 * u.kelvin
    if kind == "MonteCarloBarostat":
        return openmm.MonteCarloBarostat(1.0 * u.bar, T, 5)
    if kind == "MonteCarloAnisotropicBarostat":
        return openmm.MonteCarloAnisotropicBarostat(openmm.Vec3(1.0, 1.0, 1.0) * u.bar, T, True, True, True, 5)
    if kind == "MonteCarloMembraneBarostat":
        return openmm.MonteCarloMembraneBarostat(
            1.0 * u.bar, 0.0 * u.bar * u.nanometer, T,
            openmm.MonteCarloMembraneBarostat.XYIsotropic, openmm.MonteCarloMembraneBarostat.ZFree, 5,
        )
    if kind == "MonteCarloFlexibleBarostat":
        return openmm.MonteCarloFlexibleBarostat(1.0 * u.bar, T, 5)
    if kind == "AndersenThermostat":
        return openmm.AndersenThermostat(T, 50.0 / u.picosecond)
    raise AssertionError(kind)


_STOCHASTIC_KINDS = [
    "MonteCarloBarostat",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
    "MonteCarloFlexibleBarostat",
    "AndersenThermostat",
]


@pytest.mark.parametrize("kind", _STOCHASTIC_KINDS)
def test_i1_stochastic_system_force_rejected_for_measurement(kind):
    """C-I1: a barostat / Andersen thermostat in the System acts on the
    dynamics and carries its own RNG; purpose='measurement' must reject it
    (design section 9), at construction and for a build-time cfg."""
    system, top, x0 = _waterbox()
    system.addForce(_stochastic_force(kind))
    with pytest.raises(ValueError, match=kind):
        OpenMMBackend(system, top, _cfg(rigid_water=True))
    eq = OpenMMBackend(system, top, _cfg(rigid_water=True, purpose="equilibration"))
    with pytest.raises(ValueError, match=kind):
        eq.build(MDState(x0, np.zeros_like(x0), 0.0, np.diag([1.8] * 3)), _cfg(rigid_water=True), _key())
    with pytest.raises(ValueError, match=kind):
        eq.effective_config(_cfg(rigid_water=True))


def _barostat_run(b, x0, v0, key, n=100, edge=1.5):
    p = b.build(MDState(x0, v0, 0.0, np.diag([edge] * 3)), None, key)
    p.run(n)
    return p, p.get_state()


def test_i1_equilibration_barostat_seeded_from_key_reproducible():
    """C-I1 reproducer (p2 (a), smaller box for speed): WaterBox +
    MonteCarloBarostat(freq 5), same key twice -> identical box and x (p2
    measured 1.8425 vs 1.8582 nm before the fix); the seed is
    derived from derive_rng(key, 'openmm_force_seed/<index>'), nonzero, set
    on a copy of the System (the caller's System keeps seed 0)."""
    system, top, x0 = _waterbox(edge=1.5, cutoff=0.6)
    idx = system.addForce(_stochastic_force("MonteCarloBarostat"))
    b = OpenMMBackend(
        system, top,
        _cfg(rigid_water=True, integrator="langevin_middle", friction_per_ps=1.0, purpose="equilibration"),
    )
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v_i1"))
    p1, a = _barostat_run(b, x0, v0, _key(shot_id=1))
    _, c = _barostat_run(b, x0, v0, _key(shot_id=1))
    _, d = _barostat_run(b, x0, v0, _key(shot_id=2))
    assert np.array_equal(a.box, c.box) and np.array_equal(a.x, c.x)
    assert not np.array_equal(a.box, d.box)  # the MC moves did happen and are keyed
    expected = int(derive_rng(_key(shot_id=1), f"openmm_force_seed/{idx}").integers(1, 2**31 - 1, endpoint=True))
    assert p1.force_seeds == {idx: expected} and expected != 0
    assert p1.context.getSystem().getForce(idx).getRandomNumberSeed() == expected
    assert system.getForce(idx).getRandomNumberSeed() == 0  # caller's System untouched
    assert p1.context.getSystem() is not system
    prov = b.provenance()  # describes the most recent build (key shot_id=2)
    expected2 = int(derive_rng(_key(shot_id=2), f"openmm_force_seed/{idx}").integers(1, 2**31 - 1, endpoint=True))
    assert prov["stochastic_force_seeds"] == {str(idx): expected2} and expected2 != expected
    assert prov["effective_config"]["system_stochastic_forces"] == [
        {"index": idx, "type": "MonteCarloBarostat", "seed": f"derive_rng(rng_key, 'openmm_force_seed/{idx}')"}
    ]


def test_i1_equilibration_andersen_seeded_from_key_reproducible():
    """C-I1 reproducer (p2 (b)) for equilibration: Andersen + Verlet on
    Reference, same key twice -> bitwise identical x; other key differs."""
    system, top, x0 = _waterbox(edge=1.5, cutoff=0.6)
    idx = system.addForce(_stochastic_force("AndersenThermostat"))
    b = OpenMMBackend(system, top, _cfg(rigid_water=True, purpose="equilibration"))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v_i1"))
    runs = [_barostat_run(b, x0, v0, _key(shot_id=k), n=50)[1] for k in (1, 1, 2)]
    assert np.array_equal(runs[0].x, runs[1].x) and np.array_equal(runs[0].v, runs[1].v)
    assert not np.array_equal(runs[0].x, runs[2].x)
    assert system.getForce(idx).getRandomNumberSeed() == 0


def test_i1_energy_forces_ignores_stochastic_forces():
    """The single-point Context never steps, so the barostat is irrelevant
    to it; energies equal those of the System without the barostat."""
    system, top, x0 = _waterbox()
    E_plain, F_plain = OpenMMBackend(system, top, _cfg(rigid_water=True)).energy_forces(x0)
    system.addForce(_stochastic_force("MonteCarloBarostat"))
    b = OpenMMBackend(system, top, _cfg(rigid_water=True, purpose="equilibration"))
    E, F = b.energy_forces(x0)
    assert E == E_plain and np.array_equal(F, F_plain)


# -- I2 / K8 -----------------------------------------------------------------


def test_k8_provenance_reflects_build_cfg_not_constructor_cfg():
    """C-I2: provenance must describe the cfg the propagator ran with.
    provenance(None) -> the cfg of the latest build; provenance(cfg) -> cfg."""
    system, top, x0 = _ala()
    eq_cfg = _cfg(integrator="langevin_middle", friction_per_ps=1.0, purpose="equilibration", dt_ps=0.002)
    meas_cfg = _cfg(integrator="verlet", purpose="measurement", dt_ps=0.001)
    b = OpenMMBackend(system, top, eq_cfg)
    assert b.provenance()["integrator"] == "langevin_middle"  # nothing built yet: constructor cfg
    p = b.build(MDState(x0, np.zeros_like(x0), 0.0), meas_cfg, _key())
    assert p.cfg == meas_cfg
    prov = b.provenance()
    assert prov["integrator"] == "verlet" and prov["purpose"] == "measurement"
    assert prov["friction_per_ps"] == 0.0 and prov["dt_ps"] == 0.001
    assert prov["effective_config"] == b.effective_config(meas_cfg)
    assert prov["integrator_seed"] is None
    prov_eq = b.provenance(eq_cfg)
    assert prov_eq["integrator"] == "langevin_middle" and prov_eq["purpose"] == "equilibration"
    assert prov_eq["friction_per_ps"] == 1.0 and prov_eq["dt_ps"] == 0.002
    assert prov_eq["integrator_seed"] is None  # the latest build did not use eq_cfg
    b.build(MDState(x0, np.zeros_like(x0), 0.0), None, _key(shot_id=3))
    prov2 = b.provenance()
    assert prov2["integrator"] == "langevin_middle"
    assert prov2["integrator_seed"] == integrator_seed(_key(shot_id=3))
    with pytest.raises(ValueError, match="friction"):
        b.provenance(_cfg(integrator="langevin_middle", friction_per_ps=1.0, purpose="measurement"))


def test_k8_build_records_a_copy_of_its_cfg():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    cfg = _cfg(dt_ps=0.002)
    b.build(MDState(x0, np.zeros_like(x0), 0.0), cfg, _key())
    cfg.dt_ps = 0.004  # caller mutates its object afterwards
    assert b.provenance()["dt_ps"] == 0.002


def test_k8_effective_config_plain_dict():
    import json

    system, top, _ = _ala(constraints=app.HBonds)
    cfg = _cfg(integrator="langevin_middle", friction_per_ps=0.1, constraints="hbonds", dt_ps=0.002)
    b = OpenMMBackend(system, top, cfg)
    ec = b.effective_config()
    json.dumps(ec)  # plain, JSON-serialisable
    assert ec == b.effective_config(cfg) == b.effective_config()  # deterministic, None -> constructor cfg
    assert ec["integrator"] == "langevin_middle" and ec["dt_ps"] == 0.002
    assert ec["temperature_K"] == 300.0 and ec["friction_per_ps"] == 0.1
    assert ec["constraints"] == "hbonds" and ec["rigid_water"] is False
    assert ec["platform"] == "Reference" and ec["precision"] == "double"
    assert ec["deterministic_forces"] is True and ec["purpose"] == "measurement"
    assert ec["constraint_tolerance"] == 1e-5
    assert ec["integrator_seed"] == "derive_rng(rng_key, 'openmm_seed')"
    assert ec["system_stochastic_forces"] == []
    v = b.effective_config(_cfg(constraints="hbonds"))  # Verlet: no thermostat -> T/friction not in effect
    assert v["temperature_K"] is None and v["friction_per_ps"] == 0.0 and v["integrator_seed"] is None
    cpu = b.effective_config(_cfg(platform="CPU", precision="mixed", constraints="hbonds"))
    assert cpu["platform_properties"] == {"DeterministicForces": "true", "Threads": "1"}


# -- I3: build timing + cached per-build Python work ------------------------


def test_i3_build_timing_recorded_not_in_provenance(monkeypatch):
    import cytherea.backends.openmm_backend as ob

    system, top, x0 = _ala(constraints=app.HBonds)
    calls = []
    real = ob._check_constraints
    monkeypatch.setattr(ob, "_check_constraints", lambda *a: (calls.append(1), real(*a))[1])
    b = OpenMMBackend(system, top, _cfg(constraints="hbonds"))
    assert len(calls) == 1
    s = MDState(x0, np.zeros_like(x0), 0.0)
    for dt in (0.001, 0.002):
        p = b.build(s, _cfg(constraints="hbonds", dt_ps=dt), _key())
    assert len(calls) == 1  # same (constraints, rigid_water): verdict cached, not re-checked
    t = p.build_timing
    for k in ("validate_s", "system_prep_s", "context_creation_s", "set_state_s", "total_s"):
        assert isinstance(t[k], float) and t[k] >= 0.0, k
    assert t["total_s"] >= t["context_creation_s"]
    assert b.last_build_timing == t
    assert not any("_s" == k[-2:] for k in b.provenance())  # wall times would break record equality


# -- M1: OpenMM's process-global RNG (Reference Langevin, Reference/CPU Andersen)


def test_m1_reference_langevin_interleaved_contexts_raise():
    """p2 (c): on Reference, creating a second LangevinMiddle Context
    re-seeds the process-global RNG, so the first propagator would silently
    continue on the second key's noise. run() must refuse."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1))
    s = MDState(x0, _mb_velocities(_masses(system), 300.0, _key(stage="v")), 0.0)
    p1 = b.build(s, None, _key(shot_id=1))
    p1.run(10)
    p2 = b.build(s, None, _key(shot_id=2))
    with pytest.raises(RuntimeError, match="global RNG"):
        p1.run(10)
    p2.run(10)  # the most recent one is fine
    b.energy_forces(x0)  # a Verlet single-point Context does not re-seed
    p2.run(10)


def test_m1_cpu_langevin_contexts_independent():
    system, top, x0 = _ala()
    b = OpenMMBackend(
        system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1, platform="CPU", precision="mixed")
    )
    s = MDState(x0, _mb_velocities(_masses(system), 300.0, _key(stage="v")), 0.0)
    p = b.build(s, None, _key(shot_id=1))
    p.run(50)
    alone = p.get_state()
    p1 = b.build(s, None, _key(shot_id=1))
    b.build(s, None, _key(shot_id=2))
    p1.run(50)  # per-Context RNG on CPU: no guard, and genuinely unaffected
    assert np.array_equal(p1.get_state().x, alone.x)


# -- M2 / M3: set_state box=None and finiteness ------------------------------


def test_m2_set_state_box_none_means_system_default_box():
    system, top, x0 = _waterbox()
    b = OpenMMBackend(system, top, _cfg(rigid_water=True))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    p = b.build(MDState(x0, v0, 0.0, np.diag([2.0] * 3)), None, _key())
    p.set_state(MDState(x0, v0, 0.0, None))
    assert np.allclose(p.get_state().box, np.diag([1.8] * 3), atol=1e-12)
    pe = p.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    assert pe == pytest.approx(b.energy_forces(x0)[0], rel=1e-12)


@pytest.mark.parametrize("field", ["x", "v", "t", "box"])
def test_m3_set_state_rejects_nonfinite_before_mutation(field):
    system, top, x0 = _waterbox()
    b = OpenMMBackend(system, top, _cfg(rigid_water=True))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    box = np.diag([1.8] * 3)
    p = b.build(MDState(x0, v0, 0.0, box), None, _key())
    before = _raw_context_state(p.context)
    x, v, t, bx = x0.copy(), v0.copy(), 0.5, box.copy()
    if field == "x":
        x[3, 1] = np.inf
    elif field == "v":
        v[0, 0] = np.nan
    elif field == "t":
        t = float("nan")
    else:
        bx[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        p.set_state(MDState(x, v, t, bx))
    _assert_raw_equal(_raw_context_state(p.context), before)
    with pytest.raises(ValueError, match="finite"):
        b.build(MDState(x, v, t, bx), None, _key())


# -- M4: virtual sites ---------------------------------------------------------


def test_m4_virtual_sites_recomputed_in_energy_forces_and_set_state():
    """p4 (b): TIP4P-Ew; stale extra-point positions gave -14111 vs -3635
    kJ/mol. energy_forces and set_state now recompute the virtual sites."""
    w = testsystems.WaterBox(box_edge=1.5 * u.nanometer, cutoff=0.6 * u.nanometer, model="tip4pew")
    system, top = w.system, w.topology
    x0 = np.asarray(w.positions.value_in_unit(u.nanometer))
    m = _masses(system)
    vs = np.array([system.isVirtualSite(i) for i in range(system.getNumParticles())])
    assert vs.any()
    b = OpenMMBackend(system, top, _cfg(rigid_water=True))
    E_true, F_true = b.energy_forces(x0)
    xs = x0.copy()
    xs[vs] += 0.05
    E_stale, F_stale = b.energy_forces(xs)
    assert E_stale == pytest.approx(E_true, rel=1e-9)
    assert np.allclose(F_stale, F_true, rtol=0, atol=1e-6 * np.max(np.abs(F_true)))
    v0 = _mb_velocities(m, 300.0, _key(stage="v"))
    box = np.diag([1.5] * 3)
    ref = b.build(MDState(x0, v0, 0.0, box), None, _key()).get_state()
    p = b.build(MDState(xs, v0, 0.0, box), None, _key())
    s = p.get_state()
    assert np.max(np.abs(s.x[vs] - xs[vs])) > 0.04  # the displaced sites did not stick
    assert np.allclose(s.x[vs], ref.x[vs], atol=1e-12)  # recomputed from the real atoms
    assert np.array_equal(s.x[~vs], xs[~vs])
    pe = p.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    assert pe == pytest.approx(E_true, rel=1e-9)


# -- M5: water recognised by composition ------------------------------------


@pytest.mark.parametrize("resname", ["TIP3", "SOL", "T3P", "SPC"])
def test_m5_water_recognised_by_composition(resname):
    for constrained, claim in [(True, False), (False, True)]:
        system, top, _ = _waterbox(edge=1.2, cutoff=0.5, constrained=constrained)
        for r in top.residues():
            r.name = resname
        with pytest.raises(ValueError, match="rigid"):
            OpenMMBackend(system, top, _cfg(rigid_water=claim))
        OpenMMBackend(system, top, _cfg(rigid_water=not claim))


# -- M6: CMMotionRemover vs an IC with net momentum -------------------------


def test_m6_verlet_measurement_rejects_net_momentum_with_cmmotionremover():
    system, top, x0 = _ala()
    assert any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces())
    m = _masses(system)
    raw = _mb_velocities(m, 300.0, _key(stage="v"), remove_com=False)
    b = OpenMMBackend(system, top, _cfg())
    with pytest.raises(ValueError, match="momentum"):
        b.build(MDState(x0, raw, 0.0), None, _key())
    b.build(MDState(x0, _mb_velocities(m, 300.0, _key(stage="v")), 0.0), None, _key())
    # Langevin measurement / equilibration: not checked (no reversal symmetry anyway)
    b.build(MDState(x0, raw, 0.0), _cfg(integrator="langevin_middle", friction_per_ps=0.1), _key())
    b.build(MDState(x0, raw, 0.0), _cfg(purpose="equilibration"), _key())
    # no CMMotionRemover: nothing alters the IC, so nothing to check
    system2, top2, _ = _ala()
    for i in reversed(range(system2.getNumForces())):
        if isinstance(system2.getForce(i), openmm.CMMotionRemover):
            system2.removeForce(i)
    OpenMMBackend(system2, top2, _cfg()).build(MDState(x0, raw, 0.0), None, _key())


# -- M8: com_distance edge cases + triclinic ----------------------------------


def test_m8_com_distance_rejects_bad_groups():
    box = np.diag([2.0] * 3)
    x = np.array([[0.1, 1.0, 1.0], [0.3, 1.0, 1.0], [1.5, 1.5, 1.5]])
    s = MDState(x, np.zeros_like(x), 0.0, box)
    m = np.array([12.0, 0.0, 16.0])
    with pytest.raises(ValueError, match="mass"):
        com_distance(s, np.array([1]), np.array([2]), m)
    with pytest.raises(ValueError, match="index"):
        com_distance(s, np.array([-1]), np.array([2]), m)
    with pytest.raises(ValueError, match="index"):
        com_distance(s, np.array([0]), np.array([3]), m)
    with pytest.raises(ValueError, match="integer"):
        com_distance(s, np.array([0.0]), np.array([2]), m)
    with pytest.raises(ValueError, match="masses"):
        com_distance(s, np.array([0]), np.array([2]), m[:2])


def _brute_min_image(d, box):
    """Independent reference: fold d into the cell by fractional rounding,
    then search all images within +-2 lattice vectors."""
    f = d @ np.linalg.inv(box)
    d = (f - np.round(f)) @ box
    best = None
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                c = d + i * box[0] + j * box[1] + k * box[2]
                if best is None or np.linalg.norm(c) < np.linalg.norm(best):
                    best = c
    return best


def _truncated_octahedron(d):
    # OpenMM reduced form (as produced by app.Modeller for boxShape='octahedron')
    return np.array([
        [d, 0.0, 0.0],
        [d / 3.0, 2.0 * np.sqrt(2.0) * d / 3.0, 0.0],
        [-d / 3.0, np.sqrt(2.0) * d / 3.0, np.sqrt(6.0) * d / 3.0],
    ])


def test_m8_com_distance_triclinic_min_image_and_continuity():
    box = _truncated_octahedron(4.0)
    rng = derive_rng(_key(stage="m8"), "tri")
    m = np.array([12.0, 1.0, 16.0, 14.0, 14.0])
    mol_a = np.array([[0.0, 0.0, 0.0], [0.15, 0.05, 0.0], [0.3, -0.05, 0.02]])
    mol_b = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.05]])
    ia, ib = np.array([0, 1, 2]), np.array([3, 4])
    inv = np.linalg.inv(box)
    for _ in range(200):
        ca, cb = rng.uniform(-6, 6, size=3), rng.uniform(-6, 6, size=3)
        x = np.vstack([mol_a + ca, mol_b + cb])
        # wrap every atom independently into the unit cell (splits molecules)
        f = x @ inv
        xw = (f - np.floor(f)) @ box
        com_a = (m[ia, None] * x[ia]).sum(0) / m[ia].sum()
        com_b = (m[ib, None] * x[ib]).sum(0) / m[ib].sum()
        ref = np.linalg.norm(_brute_min_image(com_a - com_b, box))
        d = com_distance(MDState(xw, np.zeros_like(xw), 0.0, box), ia, ib, m)
        assert d == pytest.approx(ref, abs=1e-9)


def test_m8_com_distance_rejects_non_reduced_triclinic():
    box = np.array([[2.0, 0.3, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])  # a has a y component
    x = np.zeros((2, 3))
    with pytest.raises(ValueError, match="reduced"):
        com_distance(MDState(x, x, 0.0, box), np.array([0]), np.array([1]), np.ones(2))
    # triclinic make-whole guard: a chain reaching 0.95 nm > 0.45 x lambda_1 (= d = 2 nm)
    tri = _truncated_octahedron(2.0)
    chain = np.array([[0.1, 0.5, 0.5], [0.4, 0.5, 0.5], [0.7, 0.5, 0.5], [1.05, 0.5, 0.5], [1.5, 1.2, 1.0]])
    m = np.ones(5)
    with pytest.raises(ValueError, match="extent"):
        com_distance(MDState(chain, chain, 0.0, tri), np.arange(4), np.array([4]), m)
    com_distance(MDState(chain, chain, 0.0, tri), np.arange(3), np.array([4]), m)  # 0.6 nm: fine


def test_m8_com_distance_on_openmm_state_matches_independent_calc():
    system, top, x0 = _waterbox(edge=1.2, cutoff=0.5)
    m = _masses(system)
    b = OpenMMBackend(system, top, _cfg(rigid_water=True))
    box = np.diag([1.2] * 3)
    p = b.build(MDState(x0, _mb_velocities(m, 300.0, _key(stage="v")), 0.0, box), None, _key())
    p.run(20)
    s = p.get_state()
    for ia, ib in [(np.arange(0, 3), np.arange(3, 6)), (np.arange(6, 9), np.arange(30, 33))]:
        # OpenMM keeps molecules whole, so plain COMs + one min-image are exact
        ca = (m[ia, None] * s.x[ia]).sum(0) / m[ia].sum()
        cb = (m[ib, None] * s.x[ib]).sum(0) / m[ib].sum()
        d = ca - cb
        d -= 1.2 * np.round(d / 1.2)
        assert com_distance(s, ia, ib, m) == pytest.approx(float(np.linalg.norm(d)), abs=1e-12)


# -- M9: velocity reversal; solvated PME + HBonds + rigid water ---------------


def _reversal_error(b, x0, v0, n, key):
    p = b.build(MDState(x0, v0, 0.0), None, key)
    p.run(n)
    s = p.get_state()
    q = b.build(MDState(s.x, -s.v, 0.0, s.box), None, key)
    q.run(n)
    r = q.get_state()
    return float(np.max(np.abs(r.x - x0))), float(np.max(np.abs(-r.v - v0)) / np.max(np.abs(v0)))


def test_m9_velocity_reversal_retraces_trajectory():
    """Two-sided shooting relies on it (p3): forward n steps, reverse the
    on-step v, n steps -> back at x0 with -v0 (to rounding, unconstrained)."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg(dt_ps=0.001))
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v"))
    dx, dv = _reversal_error(b, x0, v0, 500, _key())
    print(f"reversal (Verlet, unconstrained, 500 steps): max|dx|={dx:.3e} nm rel|dv|={dv:.3e}")
    assert dx < 1e-10 and dv < 1e-9
    # negative control: reversing OpenMM's raw half-step velocity misses badly
    p = b.build(MDState(x0, v0, 0.0), None, _key())
    p.run(500)
    vh = p.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
    x_end = p.get_state().x
    ctx = p.context
    ctx.setVelocities(-vh)
    p.integrator.step(500)
    x_back = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    assert np.max(np.abs(x_back - x0)) > 1e3 * dx and np.isfinite(x_end).all()


def test_m9_solvated_pme_hbonds_rigid_water_propagation_cpu():
    """The A1-like case in the default suite: explicit-solvent alanine
    dipeptide (PME, HBonds, rigid water, CMMotionRemover), CPU platform."""
    t = testsystems.AlanineDipeptideExplicit()
    system, top = t.system, t.topology
    x0 = np.asarray(t.positions.value_in_unit(u.nanometer))
    box = np.array([[v[i].value_in_unit(u.nanometer) for i in range(3)] for v in system.getDefaultPeriodicBoxVectors()])
    cfg = _cfg(constraints="hbonds", rigid_water=True, platform="CPU", precision="mixed", dt_ps=0.002)
    b = OpenMMBackend(system, top, cfg)
    m = _masses(system)
    # a constraint-satisfying start: a short equilibration first
    eq = b.build(MDState(x0, np.zeros_like(x0), 0.0, box), _cfg(
        constraints="hbonds", rigid_water=True, platform="CPU", precision="mixed", dt_ps=0.002,
        integrator="langevin_middle", friction_per_ps=5.0, purpose="equilibration"), _key(stage="eq"))
    eq.run(50)
    s_eq = eq.get_state()
    v0 = _mb_velocities(m, 300.0, _key(stage="v"))
    p = b.build(MDState(s_eq.x, v0, 0.0, s_eq.box), None, _key())
    s0 = p.get_state()
    assert _constraint_violation(system, s0.x, s0.v) < 1e-6
    q = b.build(s0, None, _key())
    assert np.max(np.abs(q.get_state().v - s0.v)) / np.max(np.abs(s0.v)) < 1e-5
    p.run(50)
    s = p.get_state()
    rq = b.build(MDState(s.x, -s.v, 0.0, s.box), None, _key())
    rq.run(50)
    back = rq.get_state()
    err = float(np.max(np.abs(back.x - s0.x)))
    print(f"solvated PME/HBonds/rigid water: reversal over 50 steps max|dx|={err:.3e} nm, "
          f"build {p.build_timing['total_s']:.3f} s")
    assert np.isfinite(s.x).all() and np.isfinite(s.v).all()
    assert err < 1e-5  # constraint-tolerance level (measured 5e-7 nm, CPU mixed)


# -- M11: a failing restore does not mask the original exception ------------


class _FailPlanContext:
    """Proxy that raises on the n-th call of a method, per `plan`
    ({method: {call numbers}}, 1-based) -- e.g. a failure in the forward
    path and another one in the restore."""

    def __init__(self, ctx, plan):
        self._ctx = ctx
        self._plan = plan
        self._count = {}

    def __getattr__(self, name):
        attr = getattr(self._ctx, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            k = self._count[name] = self._count.get(name, 0) + 1
            if k in self._plan.get(name, ()):
                raise RuntimeError(f"{'injected failure' if k == 1 else 'restore failure'} in {name}")
            return attr(*args, **kwargs)

        return wrapper


def test_m11_failing_restore_keeps_original_exception():
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cfg())
    m = _masses(system)
    p = b.build(MDState(x0, _mb_velocities(m, 300.0, _key(stage="v")), 0.0), None, _key())
    real = p.context
    # setTime fails in the forward path; the restore's setPositions (2nd call) fails too
    p.context = _FailPlanContext(real, {"setTime": {1}, "setPositions": {2}})
    new = MDState(x0 + 0.01, _mb_velocities(m, 300.0, _key(shot_id=1, stage="v")), 7.0)
    with pytest.raises(RuntimeError, match="injected failure in setTime") as ei:
        p.set_state(new)
    assert any("restore" in n for n in getattr(ei.value, "__notes__", []))
    p.context = real


# ===========================================================================
# Fix wave 2, package L1: contract K9 -- the OpenMM System and Topology are
# part of the physics identity (fixreview-int2 I2 = p1-N-I1 = p5-m3).
# ===========================================================================

import functools as _functools  # noqa: E402
import sys as _sys  # noqa: E402
import textwrap as _textwrap  # noqa: E402

from cytherea.engine.shot import ObsSpec, resolve_physics_config, run_shot  # noqa: E402
from cytherea.exec.batch import ResumeConfigMismatchError, run_batch  # noqa: E402
from cytherea.observe.events import FixedLag  # noqa: E402
from cytherea.store import Store  # noqa: E402

# Shared, verbatim, by this process and a fresh interpreter (cross-process test).
_AR4_SRC = _textwrap.dedent(
    """
    import openmm
    from openmm import app

    def make_ar4(eps=0.996, atom_name="Ar"):
        system = openmm.System()
        top = app.Topology()
        res = top.addResidue("AR", top.addChain())
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        for _ in range(4):
            system.addParticle(39.948)
            nb.addParticle(0.0, 0.34, eps)
            top.addAtom(atom_name, app.element.argon, res)
        system.addForce(nb)
        return system, top
    """
)
_ar4_ns: dict = {}
exec(_AR4_SRC, _ar4_ns)  # noqa: S102 -- the same source runs in the subprocess test
_make_ar4 = _ar4_ns["make_ar4"]
_AR4_X = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])


def _ar4_shot_fn(eps, cfg):
    system, top = _make_ar4(eps)
    backend = OpenMMBackend(system, top, _cfg())
    pool = EnsembleFramePool([EnsembleFrame(_AR4_X, None, "ar4", 300.0, 1.0, "eq", 0, 0.0)])
    sampler = EnsembleFrameSampler(pool, np.full(4, 39.948), 0.0083144626 * 300.0, backend, None, None)
    obs = ObsSpec({"x0": lambda s: float(s.x[0, 0])}, 0.002, 1)
    return _functools.partial(run_shot, sampler=sampler, backend=backend, stop=FixedLag(0.004),
                              obs=obs, physics_cfg=cfg, store=None)


@pytest.mark.parametrize("explicit", [False, True])
def test_k9_one_lj_epsilon_changes_the_physics_hash_and_refuses_resume(tmp_path, explicit):
    cfg = _cfg() if explicit else None
    hashes = {}
    for eps in (0.996, 0.5):
        system, top = _make_ar4(eps)
        hashes[eps] = resolve_physics_config(OpenMMBackend(system, top, _cfg()), cfg)[1]
    assert hashes[0.996] != hashes[0.5]
    keys = [_key(shot_id=i, stage="k9") for i in range(3)]
    store = Store(tmp_path / "s.sqlite")
    run_batch(keys[:2], _ar4_shot_fn(0.996, cfg), store)
    with pytest.raises(ResumeConfigMismatchError, match="physics_config_hash"):
        run_batch(keys, _ar4_shot_fn(0.5, cfg), store)
    assert sum(1 for _ in store.iter()) == 2
    run_batch(keys, _ar4_shot_fn(0.996, cfg), store)  # the same System rebuilt: accepted
    assert sum(1 for _ in store.iter()) == 3


def test_k9_rebuilt_system_hash_is_identical_in_another_process():
    system, top = _make_ar4()
    here = resolve_physics_config(OpenMMBackend(system, top, _cfg()), None)[1]
    system2, top2 = _make_ar4()
    assert resolve_physics_config(OpenMMBackend(system2, top2, _cfg()), None)[1] == here
    script = _AR4_SRC + _textwrap.dedent(
        """
        from cytherea.backends.base import PhysicsConfig
        from cytherea.backends.openmm_backend import OpenMMBackend
        from cytherea.engine.shot import resolve_physics_config
        cfg = PhysicsConfig(integrator="verlet", dt_ps=0.001, temperature_K=300.0,
                            friction_per_ps=0.0, constraints="none", rigid_water=False,
                            platform="Reference", precision="double",
                            deterministic_forces=True, purpose="measurement")
        system, top = make_ar4()
        print(resolve_physics_config(OpenMMBackend(system, top, cfg), None)[1])
        """
    )
    out = subprocess.run([_sys.executable, "-c", script], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert out.stdout.strip() == here


def test_k9_openmm_none_and_equivalent_explicit_cfg_hash_equal():
    system, top = _make_ar4()
    b = OpenMMBackend(system, top, _cfg())
    assert resolve_physics_config(b, None)[1] == resolve_physics_config(b, _cfg())[1]
    assert resolve_physics_config(b, _cfg(dt_ps=0.002))[1] != resolve_physics_config(b, None)[1]


def test_k9_effective_config_and_provenance_carry_system_and_topology_digests():
    system, top = _make_ar4()
    b = OpenMMBackend(system, top, _cfg(platform="CPU", precision="mixed"))
    ec = b.effective_config()
    assert len(ec["system_sha256"]) == 64 and len(ec["topology_sha256"]) == 64
    assert ec["platform"] == "CPU" and ec["precision"] == "mixed"
    prov = b.provenance()
    assert prov["system_sha256"] == ec["system_sha256"]
    assert prov["topology_sha256"] == ec["topology_sha256"]
    # a topology change alone (an atom name) changes only the topology digest
    s2, t2 = _make_ar4(atom_name="AR1")
    ec2 = OpenMMBackend(s2, t2, _cfg(platform="CPU", precision="mixed")).effective_config()
    assert ec2["system_sha256"] == ec["system_sha256"]
    assert ec2["topology_sha256"] != ec["topology_sha256"]
    # no Topology: recorded as None, the System digest still identifies it
    ec3 = OpenMMBackend(s2, None, _cfg(platform="CPU", precision="mixed")).effective_config()
    assert ec3["topology_sha256"] is None and ec3["system_sha256"] == ec["system_sha256"]


def test_k9_system_is_serialised_once_per_backend_not_per_shot(monkeypatch):
    """The System digest is computed at construction; resolving the physics
    hash (done for every shot) never re-serialises."""
    calls = {"n": 0}
    real = openmm.XmlSerializer.serialize

    def counting(obj):
        calls["n"] += 1
        return real(obj)

    monkeypatch.setattr(openmm.XmlSerializer, "serialize", staticmethod(counting))
    system, top = _make_ar4()
    b = OpenMMBackend(system, top, _cfg())
    assert calls["n"] == 1 and b.identity_hash_s >= 0.0
    fn = _ar4_shot_fn(0.996, None)  # constructs a second backend: one more
    assert calls["n"] == 2
    for i in range(3):
        resolve_physics_config(b, None)
        b.provenance(_cfg())
        fn(_key(shot_id=i, stage="k9once"))
    assert calls["n"] == 2


def test_k9_stochastic_force_seed_value_is_not_part_of_the_system_digest():
    from cytherea.backends.openmm_backend import system_sha256

    def with_barostat(seed, pressure=1.0):
        system, _top, _ = _waterbox(edge=1.5, cutoff=0.6)
        baro = openmm.MonteCarloBarostat(pressure * u.bar, 300.0 * u.kelvin, 5)
        baro.setRandomNumberSeed(seed)
        system.addForce(baro)
        return system

    s7 = with_barostat(7)
    assert system_sha256(s7) == system_sha256(with_barostat(9))  # re-seeded per build anyway
    assert system_sha256(s7) != system_sha256(with_barostat(7, pressure=2.0))
    baro = next(f for f in s7.getForces() if isinstance(f, openmm.MonteCarloBarostat))
    assert baro.getRandomNumberSeed() == 7  # the caller's System is not mutated


# ---------------------------------------------------------------------------
# Fix wave 2, package L2 (fixreview-p1 N-I3, fixreview-p5 N1/m1/m2)
# ---------------------------------------------------------------------------


def _cpu_blowup_cfg():
    # 20 fs Verlet without constraints: ala2 vacuum blows up within a few
    # hundred steps (p1 probe_openmm_blowup_k4.py)
    return _cfg(dt_ps=0.02, platform="CPU", precision="mixed")


def test_k10_cpu_blowup_raises_numerical_instability_error():
    """N-I3: on CPU (and CUDA) OpenMM raises 'Particle coordinate is NaN'
    instead of returning a NaN state; the propagator must translate that
    into NumericalInstabilityError (contract K10)."""
    system, top, x0 = _ala()
    b = OpenMMBackend(system, top, _cpu_blowup_cfg())
    v0 = _mb_velocities(_masses(system), 3000.0, _key(stage="v_k10"))
    p = b.build(MDState(x0, v0, 0.0), None, _key())
    with pytest.raises(NumericalInstabilityError, match="NaN") as ei:
        for _ in range(500):
            p.run(10)
            p.get_state()
    assert isinstance(ei.value.__cause__, openmm.OpenMMException)


def test_k10_cpu_blowup_is_a_nonfinite_record_in_a_default_batch(tmp_path):
    """N-I3 reproducer end to end: run_batch with the default
    on_error='raise' must not abort, and the blown-up shots must be stored
    nonfinite records that estimators see."""
    import functools

    from cytherea.engine.shot import ObsSpec, run_shot
    from cytherea.exec.batch import run_batch
    from cytherea.observe.events import FixedLag
    from cytherea.store import Store

    system, top, x0 = _ala()
    masses = _masses(system)
    b = OpenMMBackend(system, top, _cpu_blowup_cfg())
    pool = EnsembleFramePool([EnsembleFrame(x0, None, "ala2", 300.0, 1.0, "s", 0, 0.0)])
    sampler = EnsembleFrameSampler(pool=pool, masses=masses, kT=2.494 * 20, backend=b,
                                   energy_window=None, min_pair_dist=None, remove_com_momentum=True)
    obs = ObsSpec(fns={"x0": lambda s: float(s.x[0, 0])}, dt_obs=0.2, store_stride=1)
    fn = functools.partial(run_shot, sampler=sampler, backend=b, stop=FixedLag(200.0), obs=obs,
                           physics_cfg=None)
    store = Store(tmp_path / "s.sqlite")
    keys = [ShotKey(3, 0, i, "ic") for i in range(3)]
    out = run_batch(keys, fn, store)
    assert [r.stop_reason for r in out] == ["nonfinite"] * 3
    assert sum(1 for _ in store.iter(stop_reason="nonfinite")) == 3
    assert all(any("coordinate is NaN" in w for w in r.warnings) for r in out)


class _StubIntegrator:
    def __init__(self, inner, exc):
        self._inner, self._exc = inner, exc

    def step(self, n):
        raise self._exc

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.parametrize(
    "msg,translated",
    [
        ("Particle coordinate is NaN.  For more information, see https://github.com/openmm", True),
        ("Energy or force at minimization starting point is infinite or NaN.", True),
        ("Called setPositions() on a Context with the wrong number of positions", False),
    ],
)
def test_k10_only_nan_exceptions_are_translated(msg, translated):
    system, top, x0 = _ala()
    p = OpenMMBackend(system, top, _cfg()).build(MDState(x0, np.zeros_like(x0), 0.0), None, _key())
    p.integrator = _StubIntegrator(p.integrator, openmm.OpenMMException(msg))
    expected = NumericalInstabilityError if translated else openmm.OpenMMException
    with pytest.raises(expected) as ei:
        p.run(1)
    if not translated:
        assert not isinstance(ei.value, NumericalInstabilityError)


@pytest.mark.parametrize("platform,precision", [("Reference", "double"), ("CPU", "mixed")])
def test_n1_interleaved_barostat_contexts_raise(platform, precision):
    """p5-N1: every MonteCarlo barostat seeds and draws from OpenMM's
    process-global RNG on every platform; creating a second barostat
    Context re-seeds it, so the first propagator must refuse to run."""
    system, top, x0 = _waterbox(edge=1.5, cutoff=0.6)
    system.addForce(_stochastic_force("MonteCarloBarostat"))
    cfg = _cfg(rigid_water=True, integrator="langevin_middle", friction_per_ps=1.0,
               purpose="equilibration", platform=platform, precision=precision)
    b = OpenMMBackend(system, top, cfg)
    v0 = _mb_velocities(_masses(system), 300.0, _key(stage="v_n1"))
    s = MDState(x0, v0, 0.0, np.diag([1.5] * 3))
    p1 = b.build(s, None, _key(shot_id=1))
    p1.run(5)
    p2 = b.build(s, None, _key(shot_id=1))  # even the same key re-seeds the global stream
    with pytest.raises(RuntimeError, match="global RNG"):
        p1.run(5)
    p2.run(5)


def test_n1_barostat_build_invalidates_a_reference_langevin_propagator():
    system, top, x0 = _ala()
    lb = OpenMMBackend(system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1))
    p = lb.build(MDState(x0, _mb_velocities(_masses(system), 300.0, _key(stage="v")), 0.0), None, _key())
    p.run(5)
    wsys, wtop, wx0 = _waterbox(edge=1.5, cutoff=0.6)
    wsys.addForce(_stochastic_force("MonteCarloBarostat"))
    wb = OpenMMBackend(wsys, wtop, _cfg(rigid_water=True, integrator="langevin_middle",
                                        friction_per_ps=1.0, purpose="equilibration", platform="CPU",
                                        precision="mixed"))
    wb.build(MDState(wx0, np.zeros_like(wx0), 0.0, np.diag([1.5] * 3)), None, _key())
    with pytest.raises(RuntimeError, match="global RNG"):
        p.run(5)


def test_m1_energy_forces_of_a_barostat_backend_leaves_global_rng_alone():
    """p5-m1: the single-point Context omits stochastic forces, so the first
    energy_forces call of a barostat backend must not re-seed the global RNG
    under a live Reference Langevin propagator: its trajectory stays bitwise
    identical to an undisturbed run."""
    system, top, x0 = _ala()
    lb = OpenMMBackend(system, top, _cfg(integrator="langevin_middle", friction_per_ps=0.1))
    s = MDState(x0, _mb_velocities(_masses(system), 300.0, _key(stage="v")), 0.0)

    def traj(disturb):
        p = lb.build(s, None, _key(shot_id=7))
        p.run(20)
        if disturb:
            wsys, wtop, wx0 = _waterbox(edge=1.5, cutoff=0.6)
            wsys.addForce(_stochastic_force("MonteCarloBarostat"))
            wb = OpenMMBackend(wsys, wtop, _cfg(rigid_water=True, integrator="langevin_middle",
                                                friction_per_ps=1.0, purpose="equilibration"))
            wb.energy_forces(wx0, np.diag([1.5] * 3))
        p.run(20)
        return p.get_state()

    a, c = traj(False), traj(True)
    assert np.array_equal(a.x, c.x) and np.array_equal(a.v, c.v)


def _velocities_with_momentum_ratio(m, ratio, key):
    """COM-free Maxwell-Boltzmann velocities plus a uniform drift tuned (by
    bisection) so that |sum m v| / sqrt(sum (m v)^2) == ratio."""
    base = _mb_velocities(m, 300.0, key)
    e = np.array([1.0, 0.0, 0.0])

    def r(c):
        p = m[:, None] * (base + c * e)
        return np.linalg.norm(p.sum(axis=0)) / np.sqrt(np.sum(p * p))

    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if r(mid) < ratio else (lo, mid)
    v = base + 0.5 * (lo + hi) * e
    assert r(0.5 * (lo + hi)) == pytest.approx(ratio, rel=1e-6)
    return v


def test_m2_net_momentum_threshold_is_pinned():
    """p5-m2: ratio 2e-3 is rejected and 5e-4 accepted (threshold 1e-3)."""
    system, top, x0 = _ala()
    m = _masses(system)
    b = OpenMMBackend(system, top, _cfg())
    with pytest.raises(ValueError, match="momentum"):
        b.build(MDState(x0, _velocities_with_momentum_ratio(m, 2e-3, _key(stage="m2")), 0.0), None, _key())
    b.build(MDState(x0, _velocities_with_momentum_ratio(m, 5e-4, _key(stage="m2")), 0.0), None, _key())


# ---------------------------------------------------------------------------
# A1 14.1 (2026-10-04): the backend declares its pair cutoffs so the PES
# suite can skip FD coordinates whose stencil crosses one (pes_suite,
# "Cutoff crossings").
# ---------------------------------------------------------------------------


def test_energy_cutoffs_declares_pair_cutoffs_and_default_box():
    from cytherea.backends.pes_suite import pes_consistency_suite

    system, top, _ = _ala()
    assert OpenMMBackend(system, top, _cfg()).energy_cutoffs() is None  # vacuum, NoCutoff
    wb = testsystems.WaterBox(box_edge=1.2 * u.nanometer, cutoff=0.5 * u.nanometer, constrained=False)
    b = OpenMMBackend(wb.system, wb.topology, _cfg(constraints="none", rigid_water=False))
    dec = b.energy_cutoffs()
    assert dec["cutoffs_nm"] == pytest.approx([0.5])
    assert dec["box_lengths_nm"] == pytest.approx([1.2, 1.2, 1.2])
    x0 = np.asarray(wb.positions.value_in_unit(u.nanometer))
    rep = pes_consistency_suite(b, [x0], mode="sampled", precision="double", n_fd_atoms=60)
    assert rep.fd_cutoffs == pytest.approx([0.5])
    assert rep.passed, rep.reasons
