# A1 alanine dipeptide: reference (Task 14a) and shooting (Task 14b)

The 14a scripts produce the **ground-truth reference** for acceptance gate A1:
1037 ns in 10 independent trajectories (947 ns after burn-in; the user accepted
this in place of one ≥ 1 µs trajectory on 2026-10-03) and its MSM. They use
**plain OpenMM + deeptime** and do not import `cytherea`, so the reference cannot
share a bug with the engine it is checking. `shoot_a1.py` is the shooting half
(section 4); it runs through `cytherea`.

| file | role |
|---|---|
| `build_system.py` | solvate + minimise + equilibrate; writes the fixed-box production system |
| `ref_long.py` | long NVT reference, γ = 0.1 ps⁻¹, resumable |
| `analyze_ref.py` | core-set / transition-based states, reversible MSMs, ITS with block-bootstrap CI, CK test |
| `shoot_a1.py` | 14b: start frames from the reference, shard configs, analysis of the long shots, PES check |
| `ala2_common.py` | shared constants, φ/ψ, file formats, **the state definition** (core boxes, `core_labels`, `transition_based_assignment`, `OBS_INTERVAL_PS` = 1 ps, `label_shot`) and `FrameIndex`; no `cytherea` import |

Environment: `mamba activate openmm_dev` (see global constraints). Run the
commands from the repository root.

## 1. Build (about 1 min on the 2080 Ti)

```bash
python examples/alanine_dipeptide/build_system.py --out /home/ruigengji/cytherea/runs/ala2_system
```

* Source: openmmtools' bundled `data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb` (ACE-ALA-NME, no download needed).
* amber14-all + amber14/tip3pfb, rigid water, HBonds constraints, PME with a 0.9 nm cutoff, dispersion correction, CMMotionRemover.
* Cubic box with ≥ 1.0 nm between the solute's bounding sphere and every box face (edge = 2r + 2.0 nm). OpenMM's own `padding=1.0 nm` would give a 2.0 nm box, so the script passes an explicit box size instead. The peptide is neutral, so no ions are added.
* Minimise, then run 100 ps NVT and 500 ps NPT (MonteCarloBarostat, 1 bar). Both use LangevinMiddle γ = 1 ps⁻¹, which is allowed for preparation only. ⟨V⟩ is averaged over the last 400 ps. The box is then fixed at ⟨V⟩ by rescaling the molecule centres, followed by a 20 ps NVT relaxation at that box.
* Outputs: `system.xml` (no barostat), `topology.pdb`, `state.xml` (positions, velocities, box; time 0), `build.json` (provenance), `equil_volume.csv`.
* Result of the reference build (seed 20260930): 2461 atoms (813 waters), box edge 2.9116 nm, ρ = 0.9951 g cm⁻³.

## 2. Production reference (≈ 1450 ns/day on an idle 2080 Ti, so 1 µs takes ≈ 17 h)

```bash
python examples/alanine_dipeptide/ref_long.py \
    --out /home/ruigengji/cytherea/runs/ala2_ref --total-ns 1000 --seed 20261001 \
    --platform CUDA --system-dir /home/ruigengji/cytherea/runs/ala2_system
```

* Integrator: LangevinMiddle, γ = 0.1 ps⁻¹, 300 K, dt = 2 fs, fixed box. CUDA uses mixed precision with `DeterministicForces=true`. There is no automatic fallback: `--platform CPU` is meant for smoke tests.
* `phipsi.bin` stores φ/ψ every 1 ps (`ala2_common.OBS_INTERVAL_PS`, part of the state contract) as raw 16-byte records (`<i8` step, `<f4` φ°, `<f4` ψ°). Record 0 is step 0, and the layout is described in `phipsi.json`. Load it with `ala2_common.load_phipsi` or `np.fromfile(p, dtype=[('step','<i8'),('phi','<f4'),('psi','<f4')])`. φ = C(ACE)–N–CA–C and ψ = N–CA–C–N(NME); the values agree with `mdtraj.compute_phi/psi`.
* `traj.dcd` holds a full-system frame every 10 ps (100 000 frames ≈ 3 GB per µs).
* `checkpoint.chk` is written every 10 ns and at the end. The new file is completely on disk before rotation, so a complete `checkpoint.chk` exists at every moment. The previous checkpoint is kept as `checkpoint.prev.chk`; `--resume` falls back to it automatically, and logs this, if `checkpoint.chk` is missing or unreadable. An unreadable `checkpoint.chk` is first renamed to `checkpoint.bad-<timestamp>.chk`, so the next rotation cannot copy it over the good `.prev`. Renames are followed by a best-effort directory fsync. `checkpoint.json` is informational.
* `progress.log` gets a line every 1 ns (ns done, ns/day, PE, T, ETA), flushed each time. `run.json` records the parameters, the sha256 of the input `system.xml`/`state.xml`/`topology.pdb`, the scripts' git commit, and one entry per process segment. On `--resume` the hashes are verified when present. Runs created before the hashes were added (the live 1 µs reference started 2026-09-30 21:35) have none; they resume normally and the log says the inputs were not verified.

### Resume / extend

Re-run **the same command with `--resume`**. The step count in the checkpoint is
authoritative. φ/ψ records and DCD frames written after that step are truncated
before the run continues, so the output has no duplicated or missing frames.
Any parameter change (seed, platform, intervals, system dir) is refused. To extend
a finished run, pass a larger `--total-ns` together with `--resume`.

* **SIGINT/SIGTERM** (Ctrl-C, `kill`): the run continues to the next 10 ps boundary, writes a checkpoint and exits 0. Nothing is lost.
* **Hard crash** (SIGKILL, node or GPU failure): at most 10 ns since the last checkpoint is lost and then re-run.

Checkpoints are platform- and GPU-specific (they come from OpenMM `createCheckpoint`), so resume on the same kind of device. In testing on CUDA, a run that was killed and resumed matched an uninterrupted run with the same seed **bit for bit**, for both φ/ψ and DCD.

### Frame bookkeeping (DCD ↔ step ↔ φ/ψ record)

φ/ψ record r is at step 500·r, so record 0 is step 0. DCD frame k, counted from 0, is at step 5000·(k+1); there is no DCD frame at step 0.

```python
import ala2_common as C
fi = C.FrameIndex.from_run("/home/ruigengji/cytherea/runs/ala2_ref")   # reads run.json intervals
r = fi.dcd_frame_to_record(k)             # phi/psi record of DCD frame k  (= 10*(k+1))
rec = C.load_phipsi(run / "phipsi.bin")[r]
label = C.core_labels(rec["phi"], rec["psi"])   # raw core label of that frame (-1 = non-core)
```

14b shoots only from frames with a raw core label ≥ 0. Each shot is labelled with `C.label_shot(phi, psi, start_label, interval_ps)`, which is `C.transition_based_assignment(C.core_labels(phi, psi), initial_label=start_label)`. It refuses any φ/ψ sampling interval other than `C.OBS_INTERVAL_PS` = 1 ps, and it refuses a `start_label` that disagrees with the start frame's own core label.

DCD frames hold **unwrapped** coordinates, written without `enforcePeriodicBox`, as float32 Å. At 19.5 ns they already span −35 … 32 nm, and they grow roughly as √t: about 10² nm at 1 µs, where the float32 spacing is ≈ 1–2·10⁻⁴ Å. The 14b frame loader must therefore:

* wrap molecules back into the box (the box vectors are stored with each frame);
* re-apply the constraints before the IC gate;
* use a minimum-image clash check (INT1 deferred item).

`FrameIndex` does index arithmetic only; it does not load frames. This run's format is kept for continuity.

## 3. Analysis (seconds to about a minute)

```bash
python examples/alanine_dipeptide/analyze_ref.py --run /home/ruigengji/cytherea/runs/ala2_ref
```

Core sets, in degrees:

| state | φ | ψ |
|---|---|---|
| C7eq/C5 (β + PII) | [−180, −30] | [100, 180] ∪ [−180, −160] |
| αR | [−180, −30] | [−80, −10] |
| αL | [30, 100] | [0, 90] |

Frames outside the cores keep the label of the last core visited (transition-based
assignment). This covers the ψ ≈ 0–100° bridge at φ < 0, C7ax and the seams.

* MSMs are reversible MLE on the largest connected set, using sliding counts, for the lags 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 and 2000 ps. Implied timescales are t = −τ / ln|λ|.
* 95% CIs come from 200 bootstrap resamples of 20 contiguous blocks (50 ns each for 1 µs). The bootstrap RNG is an explicit `PCG64(--seed)`.
* **Convergence rule.** A lag τᵢ is converged when, for **every** larger lag τⱼ in the list, at least one of two conditions holds:
  * the 95% percentile CI of the **paired** per-replicate ratio t2(τᵢ)[b] / t2(τⱼ)[b] contains 1. The bootstrap resamples the same blocks at every lag, so this ratio is much tighter than the marginal CIs. Comparing against a marginal CI accepted, for example, 30 ps on a dense lag list although t2(30)/t2(70) = 1.21 [1.12, 1.33].
  * the point estimates differ by at most `--its-tol` (10%).

  Lags with τ > t2 are kept in the comparison and listed in `lag_exceeds_t2_ps`, so a rising ITS is not hidden. Larger lags whose t2 CI is invalid are left out of the comparison; they are listed. A candidate lag needs:
  * a finite t2 with a valid CI;
  * the full visited state set as its active set;
  * at least one larger lag to compare with.

  `shoot_lag.tol_only_lag_ps` reports the same rule without the paired-ratio clause.
* **Two lags are reported:**
  * `tba_converged_lag_ps` measures the quality of the reference MSM. It uses the rule above on the TBA ITS, and `--msm-lag-ps` overrides it. `msm` and `ck_test` are evaluated at this lag.
  * `shoot_lag_ps` is **the 14b FixedLag τ**. It uses the rule above on the **core-start** ITS, restricted to τ ≥ `--min-shoot-lag-ps`, which defaults to 10 ps (10 observation intervals), **and** it requires the core-start CK test at that lag to pass (`pass = true`; inconclusive counts as a failure). Below about 10 ps, shots from core frames almost never reach another core, because transitions pass through the non-core bridge, so T(τ) measures the short-τ bias rather than the barrier crossing. The CK condition exists because t2 alone is not enough. Once αL is visited, t2 is the αL process, whose core-start ITS is flat from 10 ps on. The faster C7eq ↔ αR process (t3) is still in the bridge bias at that lag, and the CK, which 14b repeats on the shots (14.3), fails on its diagonal. On the full data, core-start t3 is 316 ps against a TBA value of 168 ps at 10 ps, 201 vs 177 at 50 ps, and 187 vs 176 at 100 ps. `shoot_lag.t2_only_lag_ps` reports the t2 rule without the CK condition, and `shoot_lag.ck_scan` lists the CK result of every lag that was tried. `--shoot-lag-ps` overrides the choice, and the result is flagged `below_min_lag` if the override is under the bound. `core_start.msm` and `core_start.ck_test` are evaluated at this lag and **never** at the TBA lag. If no lag qualifies, `shoot_lag_ps` is null and so is `core_start.msm`, and the reason is given.
* **Degenerate summaries.** A summary whose largest strongly connected set misses a state that the trajectory visited is reported as `null` with a `msm_null_reason`. On the real data this happens for core-start at τ = 1 ps, where the counts are [[14126, 0], [1, 3101]]. States never visited at all, such as αL in the first 19.5 ns, are listed in `never_visited_states`, and the summary's `complete_state_set` is then false.
* **CK test.** The script compares the diagonal of T(τ)ᵏ against T_est(kτ), inside the bootstrap interval ± 0.01. It does this for both modes, using a geometric k list up to k_max = max(`--ck-k`, ⌈`--ck-horizon` · t2/τ⌉), where the horizon defaults to 2·t2. The k list is capped by the block length. If the cap keeps k_max·τ below t2, the test is inconclusive: `pass = null` and `horizon_reaches_t2 = false`. A CK spanning only a few τ ≪ t2 says nothing about Markovianity on the slow time scale.
* **Bootstrap replicates** are excluded from the CIs when their largest connected set differs from the full estimate's, for example a resample with no αL visit, whose "t2" would be a different process. The number excluded is reported for each lag (`n_boot_excluded_active_set_mismatch`). Replicates with an infinite timescale are dropped (`n_boot_valid`). Because the resulting CI is conditional, it is flagged `ci_valid = false` when more than 5% of the replicates are lost (`--ci-max-excluded-frac`).
* There are **two counting modes**:
  * `implied_timescales` / `msm` / `ck_test` use TBA counts: C_ij = #{t: TBA[t]=i, TBA[t+τ]=j}.
  * `core_start` uses C_ij = #{t: raw core[t]=i, TBA[t+τ]=j}, which counts only windows that start *inside* a core. This is the quantity that 14b FixedLag shots from core frames estimate. It gives ITS with bootstrap CIs and, at `shoot_lag_ps`, T with CIs (`core_start.msm`). At short τ its t2 is biased **high** relative to TBA, because transitions pass through non-core bridge frames and the windows that start there are excluded; those windows are over-represented in committed crossings. On the 19.5 ns data it is 1397 vs 181 ps at τ = 2 ps and 320 vs 185 ps at τ = 10 ps. The bias shrinks as τ grows. Compare 14b shots against this mode.
* Outputs: `analysis.json` and `its.png`. The figure has four panels: Ramachandran with cores; ITS vs lag, including core-start t2 and t3, the TBA lag and the shooting lag; CK for TBA; CK for core-start. `analysis.json` also carries a `contract_14b` block that summarises the next section.

### 14b contract (what the shooting half must match)

The reference quantity is `core_start.msm.transition_matrix_row_normalised`: the **row-normalised** core-start T_ij(τ) = C_ij / Σ_j C_ij at τ = `shoot_lag_ps`, with block-bootstrap CIs (`…_ci95_lo/hi`). It is only comparable if 14b uses the same:

1. **States and labelling**: `ala2_common.label_shot` (core boxes plus TBA seeded with the start core), from φ/ψ sampled every `OBS_INTERVAL_PS` = 1 ps. Labelling at 5 or 10 ps misses short visits to other cores. On these data it lowers T₀₁(100 ps) from 0.103 to 0.099 and 0.093 (−10%).
2. **Start frames**: frames with raw core label ≥ 0, drawn from the reference trajectory. The loader caveats are in §2.
3. **Lag**: FixedLag τ = `shoot_lag_ps`, chosen from the **core-start** ITS (not the TBA ITS) with a passing core-start CK, **on the contract lag grid** `ala2_common.SHOOT_LAG_GRID_PS` = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000 ps (ruling R40). This grid is the `--lags-ps` default. The chosen lag is always a grid point, so it moves with grid granularity; a `shoot_lag_ps` from any other grid (for example a dense one) is diagnostic only, and `analysis.json` then says `contract_14b.lag_grid_matches_contract: false`. See the note on the 19.5 ns value below.
4. **Estimator**: 14b compares the **row-normalised** T, which is `estimate_T(..., reversible=False)` in cytherea. The row-normalised T does not depend on how many shots start in each state. The reversible MLE, reported as `transition_matrix`, couples the rows through detailed balance. Because the 3-state core-start counts are not symmetric in expectation, the reversible T depends on the per-state row totals, which is the shooting design. Equal shots per state instead of the reference's row weights shift the αL-related elements by 3.5–6%. The reversible T is kept for ITS and populations only.
5. **Propagator**: LangevinMiddle, γ = 0.1 ps⁻¹, 300 K, dt = 2 fs, fixed box, as `ref_long.py`. It must not be NVE and must not be γ = 1.

On the first 19.53 ns, `shoot_lag_ps` is 100 ps with the default lag list and 70 ps with a denser list (…, 30, 50, 70, 100, 150, 200, 300). The two differ only by grid granularity: 70 ps is not in the default list. Under R40 the default-grid value is the one 14b uses; the dense-grid 70 ps is diagnostic. 50 ps is rejected by a hair, because the paired ratio t2(50)/t2(100) = 1.16 has the CI [1.0008, 1.35]. At 100 ps, core-start t2 = 185 ps, and T₀₁ = 0.103 [0.071, 0.164], T₁₀ = 0.316 [0.210, 0.497]. These numbers are superseded by the following.

**Full reference (2026-10-02), the 14b τ.** The data are `runs/ala2_ref` (237 ns, as 2 pieces) plus `runs/ala2_par/r01–r08` (100 ns each), with 10 ns burn-in skipped per run: 947 ns in 10 trajectories and 20 blocks of 45 ns. TBA fractions are C7eq/C5 0.700, αR 0.263, αL 0.037. There are 1100/1096 C7eq ↔ αR transitions, but only 11 C7eq → αL and 12 αL → C7eq, so t2, the αL exchange, rests on about 12 events. The TBA results: converged lag 1 ps, t2 = 2.79 ns [1.72, 4.06], t3 = 164 ps [159, 170], and the CK passes (horizon 5.6 ns). The t2-only core-start rule gives 10 ps, but the core-start CK fails at 10, 20 and 50 ps. It passes at 100 and 200 ps, so **`shoot_lag_ps` = 100 ps**. At 100 ps, core-start t2 = 2.72 ns [1.51, 3.96] and t3 = 187 ps, and the row-normalised T is:

| from \ to | C7eq/C5 | αR | αL |
|---|---|---|---|
| C7eq/C5 | 0.8861 [0.8797, 0.8926] | 0.1125 [0.1062, 0.1190] | 0.0014 [0.0007, 0.0020] |
| αR | 0.3013 [0.2912, 0.3114] | 0.6979 [0.6882, 0.7084] | 0.0008 [0, 0.0018] |
| αL | 0.0316 [0.0209, 0.0584] | 0.0034 [0, 0.0084] | 0.9651 [0.9367, 0.9763] |

The output is in `runs/ala2_par/analysis/`.

## 4. Shooting (Task 14b)

Protocol (plan Task 14, revised 2026-10-03): 50 core-interior reference frames per
state, **10 shots per frame, each `FixedLag(5500 ps)`** = 55 τ ≈ 2·t2, φ/ψ every
1 ps, the reference propagator. The first τ = 100 ps of every shot is a τ shot
(14.2); the whole shot gives the core-start T(kτ) up to k = 55, so the CK test
(14.3) runs on the shots' own data. 150 × 10 × 5.5 ns = 8.25 µs.

```bash
python examples/alanine_dipeptide/shoot_a1.py frames  --out runs/ala2_shoot/frames
python examples/alanine_dipeptide/shoot_a1.py configs --frames-dir runs/ala2_shoot/frames --out runs/ala2_shoot/long --shards 8
cytherea run runs/ala2_shoot/long/shard00.yaml      # one per shard, in parallel under MPS, each pinned to a core
# on each host, for the stores it owns (a store opens only on its owner host, spec S3):
python examples/alanine_dipeptide/shoot_a1.py export --stores runs/ala2_shoot/gpu_long/g0[0-3].sqlite \
       --out runs/ala2_shoot/exports/long --stage a1_long
python examples/alanine_dipeptide/shoot_a1.py analyze --frames-dir runs/ala2_shoot/frames \
       --shots-dir runs/ala2_shoot/exports/long --tau-dir runs/ala2_shoot/exports/tau \
       --out runs/ala2_shoot/analysis_14b.json
python examples/alanine_dipeptide/shoot_a1.py pes --frames-dir runs/ala2_shoot/frames   # 14.1
```

### Combining shards from different hosts

After the shard writers finish, run `export` **on each shard's owning host**,
listing only that host's stores. Each export retains the complete shot record,
including provenance and nonfinite outcomes. SQLite ownership checks still apply.
For example, on the host owning GPU shards g00–g03:

```bash
python examples/alanine_dipeptide/shoot_a1.py export \
       --stores runs/ala2_shoot/gpu_long/g0{0,1,2,3}.sqlite \
       --out runs/ala2_shoot/exports/long --stage a1_long
```

Export the other GPU shards on their owning host into the same export directory
(each shard has a distinct filename). Export each host's completed CPU shards
with `--stage a1_tau --out runs/ala2_shoot/exports/tau`, listing the appropriate
`cpu_tau/cNN.sqlite` files explicitly. The JSONL exports can be transferred or
read from either host; do not copy live SQLite files or take over production stores.

```bash
python examples/alanine_dipeptide/shoot_a1.py analyze \
       --frames-dir runs/ala2_shoot/frames \
       --shots-dir runs/ala2_shoot/exports/long \
       --tau-dir runs/ala2_shoot/exports/tau \
       --out runs/ala2_shoot/analysis_14b.json
```

Analyze accepts both owned `*.sqlite` stores and exported `*.jsonl` records.
Include each shard once: mixing a store with its export raises a duplicate-shot
error rather than counting the same trajectory twice. Re-running export replaces
that shard's JSONL only after a successful export; partial exports are not published.

* **Frames.** Candidates are the DCD frames (every 10 ps) whose own φ/ψ record has
  a raw core label ≥ 0, after 10 ns of burn-in per run. Per state, a systematic
  draw with a seeded random start over the time-ordered candidates spreads the
  frames over runs and core visits. Every molecule is shifted back into the box
  as a whole (float64); the IC gate re-projects the constraints, checks
  `min_pair_dist` with minimum images and draws Maxwell–Boltzmann velocities.
  `frames.json` records run, DCD frame, time, φ/ψ, state and core visit of each
  frame. On the reference, αL's 50 frames come from 10 visits in 8 runs, so
  frames are far from independent there; `analyze` also reports T with core
  visits as the clusters.
* **Analysis.** End states at kτ from `label_shot`. T(τ) is the row-normalised
  core-start estimate (`estimate_T(..., reversible=False)`, frame clusters);
  ITS come from the reversible MLE (timescales only). The CK test is
  `cytherea.estimate.ck_test_shots` on the reference's k list (1 … 55). The
  output compares T element-wise with the 14b contract below and checks 14.2
  (shooting t2 inside the reference core-start t2 CI) and 14.5 (IC rejections).

## Tests

* `pytest -q tests/test_ala2_reference.py` (fast) checks:
  * core assignment, exact core-start counts and the 1 ps contract;
  * analytic implied timescales on a synthetic 3-state chain;
  * a bridge-mediated chain: the core-start bias is high, it is degenerate at 1 ps, and the summary is taken at the shooting lag;
  * the convergence rule: rising ITS are detected and τ > t2 lags are kept;
  * a two-bridge chain (slow αL-like t2, fast bridged t3): the t2-only lag fails the core-start CK, and the shooting lag is the first lag that passes it;
  * a lumped chain with hidden memory, which the long-horizon CK test must fail;
  * DCD truncation.
* `pytest -q tests/test_ala2_shoot.py` (fast): DCD frame reading, molecule wrapping, frame selection, shot end states, and T / ITS / CK recovered from synthetic multi-lag shots.
* `pytest -q -m slow tests/test_ala2_reference.py` (≈ 3.5 min, CPU): builds the system, runs 12 ps with a simulated crash and resumes. It also covers the missing-checkpoint and unreadable-checkpoint fallbacks, checks the formats and runs the analysis.
