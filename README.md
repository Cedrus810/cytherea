# Cytherea

[English](README.md) | 简体中文

**Aimed-shooting and weighted-ensemble kinetics for proteins in solution, built on OpenMM.**

Cytherea rebuilds the chemical-dynamics idea of [VENUS96](https://doi.org/10.1016/0010-4655(96)00042-4)
(Hase group, classical chemical-dynamics trajectories) for solution-phase biomolecular
simulation: rare conformational transitions are sampled with aimed shooting from
milestone regions, propagated as ensembles of short trajectories, and turned into
transition matrices, committors and rates with calibrated uncertainty. The name is an
epithet of Aphrodite — Venus under another name.

> **Status:** research code, Phase A (`0.1.0`). Interfaces may change without notice.

## What is in the box

- **Deterministic by construction** — every shot, segment and weighted-ensemble
  iteration derives its own RNG substream from hierarchical keys; resume is guarded by
  config, protocol and code-identity hashes (package source plus OpenMM
  System/Topology digests).
- **Two backends, one contract** — analytic potentials and propagators (Euler–Maruyama,
  BAOAB, Verlet, overdamped) and OpenMM share a single protocol: on-step velocities,
  `energy_forces(x, box)`, observation every `dt_obs`, and `NaN` translated into a
  `NumericalInstabilityError` instead of silent corruption.
- **PES consistency suite** — NVE total-energy drift, integrator invariants and
  finite-difference force checks with a cutoff-crossing guard (coordinates whose
  stencil moves a pair across a nonbonded cutoff are skipped and reported), in
  strict and sampled (stratified atom-group) modes.
- **Ensemble initial conditions** — `EnsembleFramePool` with structural validity gates
  (energy window, minimum interatomic distance, COM momentum, temperature
  degrees of freedom) and constraint projection.
- **Estimators with calibrated uncertainty** — cluster-bootstrap transition matrices,
  state-fixed hierarchical bootstrap, committor and k_on with
  n_eff = min(Korn–Graubard, Kish), a global null-centred bootstrap
  Chapman–Kolmogorov test that stays calibrated on sparse Markov chains, and a
  shot-based variant (`ck_test_shots`) that tests fixed-lag shooting data at
  k·τ straight from the shots, τ-only shots included.
- **In-house weighted ensemble** — `BinnedWE` with label constraints, weight-aware
  recycling, and exact offline replay of segment lineages.
- **Milestone absorbing networks** — core-set milestoning, absorption probabilities,
  and Markov tests stratified by (origin label, milestone).
- **Config + CLI** — pydantic schemas with dimension-checked unit parsing,
  `cytherea run / resume / report`, and sidecar config-hash resume guards.

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/Cedrus810/cytherea.git
cd cytherea
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `deeptime`, `pydantic`, `pyyaml`) are pulled in
automatically. The OpenMM backend additionally requires
[OpenMM](https://openmm.org), best installed from conda-forge into the same
environment.

## Quick start

An analytic 1D double well — committor from fixed starting points, 200 shots per
frame, runs on CPU in seconds:

```bash
cytherea run examples/toy_doublewell/config.yaml
```

`examples/` also contains weighted-ensemble and milestone-network toys, the
alanine-dipeptide explicit-solvent reference and shooting pipelines (OpenMM), and a
barnase–barstar encounter-sampling pair.

## Repository map

| path | contents |
|---|---|
| `src/cytherea/` | the library: keys, store, backends, engine, ic, estimate, network, resample, config, cli |
| `examples/` | runnable campaigns, from 1D toys to solvated peptides |
| `docs/design/` | design document (VENUS96 → Aβ42, v2 authoritative) |
| `docs/reports/` | acceptance-campaign reports (A0 toys, A1 alanine dipeptide, A3 encounter pilot) and review handoffs |
| `results/` | small machine-readable artifacts (analysis JSONs) backing `docs/reports/` |
| `docs/STATUS.md` | living status and handoff notes |
| `CHANGELOG.md` | what changed, when |

## Lineage

The architecture consciously rebuilds VENUS96 — classical trajectories turned into
kinetics — for proteins in solution. No VENUS96 or VENUSpy source code is used or
included; Cytherea is an independent implementation.

## License

[MIT](LICENSE)
