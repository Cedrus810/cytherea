# Cytherea Phase A（VENUS 现代化本体）施工计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **本计划只给规格**：文件、接口签名、行为约定、必测用例（输入 → 期望）和验收阈值。实现由施工方完成，计划中不写实现代码。

**Goal:** 建成与蛋白种类无关的 Cytherea 引擎，并在解析 toy、丙氨酸二肽、chignolin 和一对小蛋白的 encounter 上逐级通过验收。

**Architecture:** Python 包 `cytherea`，由以下部件组成：
- 按键派生的 RNG；
- 只追加写入的 SQLite 记录库；
- 可插拔的 `PotentialBackend`（Phase A 只实现 analytic / openmm）；
- 采样器与 IC 门禁；
- 在线停止判据；
- `run_shot` 与批执行器；
- `Resampler`（WE，带标签约束）；
- 吸收网络；
- 估计量。

蛋白规模的传播完全在 OpenMM 内完成，每步不回 Python。

**Tech Stack:** Python 3.12，mamba 环境 `openmm_dev`（OpenMM 8.5.2、openmmml 1.6、numpy 2.4.3、scipy 1.17.1、deeptime 0.4.5、mdtraj 1.11.1、pydantic 2.13、PyYAML、pytest 9.1、pytest-xdist），标准库 `sqlite3`。

**Spec:** `docs/design/VENUS96_AB42_IDP_design_v2.md`（下文写作“设计 §x”）。执行者必须同时阅读本计划和设计文档。

## Global Constraints

- 代码仓库：新建 `/home/ruigengji/cytherea/`，git 初始化，源码放在 `src/cytherea/`，测试放在 `tests/`，示例放在 `examples/`。
- 环境激活：`export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba; export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3; source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh; mamba activate openmm_dev`。
- **不新增依赖**。确实需要时（例如 wepy），先由 Task 17 出具评估报告，经用户批准后再加。
- **不拷贝、不改写 VENUSpy 的任何代码**（设计 §2.5：它没有 LICENSE）；F77 代码也不移植。
- 内部单位与 OpenMM 一致：nm、ps、kJ/mol、K。解析 toy 允许使用约化单位，但必须在配置里显式声明 `units: reduced`。凡是出现 Å 的输入，必须显式转换。
- 随机性全部来自 `derive_rng(key)`。禁止使用 `np.random.*` 的全局状态、Python 的 `random` 模块，以及不带种子的 OpenMM 积分器（设计 §7）。
- IC 必须逐位可复现；轨迹逐位可复现只做到尽力而为（CUDA 下 `DeterministicForces=true`）。
- IC 门禁不得静默放行：任何非有限值、越出窗口的样本都必须带原因拒绝，并计入记录（设计 §7，h2oleps 教训）。
- 动力学测量用的传播器只能是 NVE、低摩擦 Langevin（γ ≤ 0.1 ps⁻¹）或 Nosé–Hoover；γ = 1 ps⁻¹ 只允许用于平衡准备（设计 §9）。
- GPU 作业一次只跑一个，放在前台，开跑前先看 `nvidia-smi`。GPU 被占用时不跑性能测试。不开后台等待循环。
- 每个任务完成后 `pytest -q` 必须全绿，然后 commit。commit 信息用英文 conventional 格式。

## Review Focus

以下 5 类输入或故障是设计暗含、但容易漏测的，每一条都已经在对应任务里加了测试：

1. **WE 权重下溢**：多次 split 之后权重可能 < 1e-300。期望：权重下限触发同标签内的强制 merge，Σw 仍守恒；绝不能出现 0 或 NaN 权重。→ Task 10 用例 10.7。
2. **PBC 下的距离类 observable**：分子跨越盒边界时，COM 距离必须按最小镜像计算，并且先把分子整体拼回完整。期望：分子跨边界前后，observable 连续。→ Task 12 用例 12.6。
3. **崩溃后续算**：批处理进行到一半时进程被杀，库里只能出现完整记录，不能有半条。期望：续算只补跑缺失的 key，结果与一次跑完完全一致。→ Task 7 用例 7.4。
4. **单位混淆**：配置里写成 Å 或缺少单位。期望：配置加载时直接报错，而不是悄悄按 nm 解释。→ Task 13 用例 13.3。
5. **停止判据永不触发**：轨迹到达 t_max 也没有命中任何吸收区。期望：记为 `timeout`，在 committor 中单独计数，超过 5% 时估计标记为无效。→ Task 6 用例 6.5、Task 8 用例 8.3。

---

## 文件结构

```text
cytherea/
  pyproject.toml                 # 包元数据，console script: cytherea
  src/cytherea/
    keys.py                      # ShotKey / SegmentKey / IterKey、derive_rng、key_digest
    store.py                     # ShotRecord、Store（SQLite，只追加）、config_hash
    backends/base.py             # MDState、Propagator、PotentialBackend 协议
    backends/analytic.py         # 解析势 + overdamped/BAOAB 传播器
    backends/pes_suite.py        # pes_consistency_suite
    backends/openmm_backend.py   # OpenMMBackend
    ic/frames.py                 # EnsembleFrame、EnsembleFramePool
    ic/sampler.py                # EnsembleFrameSampler、MB 速度、IC 门禁
    ic/encounter.py              # EncounterSampler（b 面、SO(3)×S²）
    observe/events.py            # Region、StopRule 及其实现、offline_replay
    engine/shot.py               # run_shot
    exec/batch.py                # run_batch（可续算，与执行顺序无关）
    resample/we.py               # Walker、Resampler、BinnedWE、run_segment、run_we
    network/absorb.py            # StageNetwork、build_transitions、solve_absorption、markov_test
    estimate/msm.py              # estimate_T（deeptime 封装）、CK、implied timescales
    estimate/committor.py        # estimate_committor
    estimate/association.py      # nam_beta_inf、estimate_kon
    estimate/decompose.py        # decompose、hierarchical_bootstrap
    config/schema.py             # pydantic 配置模型、load_config
    cli.py                       # cytherea run / resume / report
  tests/                         # 与源码一一对应；参考解放在 tests/reference/
  examples/toy_*/  alanine_dipeptide/  chignolin/  encounter_pair/
  docs/reports/                  # 各阶段验收报告
```

---

### Task 1: 仓库骨架 + 按键派生 RNG

**Files:** Create `pyproject.toml`、`src/cytherea/__init__.py`、`src/cytherea/keys.py`、`tests/test_keys.py`、`.gitignore`（忽略 `*.sqlite`、`runs/`、`examples/**/out/`）

**Interfaces — Produces:**

```python
@dataclass(frozen=True)
class ShotKey:   global_seed: int; frame_id: int; shot_id: int; stage: str
@dataclass(frozen=True)
class SegmentKey: run_id: str; iteration: int; walker_id: int
@dataclass(frozen=True)
class IterKey:   global_seed: int; run_id: str; iteration: int
Key = ShotKey | SegmentKey | IterKey
def key_digest(key: Key) -> str                   # 规范编码后的 sha256 十六进制串
def derive_rng(key: Key, substream: str = "") -> numpy.random.Generator   # PCG64(SeedSequence)，由 digest 派生
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 1.1 | 同一个 key 调用两次 `derive_rng(k).random(8)` | 两次结果逐位相同 |
| 1.2 | 只有 `shot_id` 不同，或只有 `stage` 不同 | 前 8 个数全都不同 |
| 1.3 | 先按 key 顺序 A,B,C 生成，再按 C,A,B 生成 | 每个 key 的结果与顺序无关 |
| 1.4 | 在 `multiprocessing` 子进程里生成 | 与主进程结果逐位相同 |
| 1.5 | `ShotKey(1,2,3,"ic")` 的 digest | 等于写死在测试里的黄金值（首次实现后固定下来，以后改动编码会立即被抓到） |
| 1.6 | `substream="a"` 与 `substream="b"` | 两条流不同 |

- [ ] 按上表写测试 → 运行确认失败 → 实现 → 全部通过 → `git init` 并 commit `feat: keyed RNG and repo skeleton`

---

### Task 2: 记录库（只追加）与 provenance

**Files:** Create `src/cytherea/store.py`、`tests/test_store.py`

**Interfaces — Consumes:** Task 1 的 `Key`、`key_digest`。**Produces:**

```python
@dataclass
class ShotRecord:
    key_digest: str; key: dict; kind: Literal["shot", "segment"]
    frame_id: int | None; origin_label: tuple[int, int] | None
    ic_validity: dict                        # {"ok": bool, "reasons": [...], "n_redraws": int}
    stop_rule_kind: str; stop_reason: str; event_time: float | None
    physics_config_hash: str; backend_provenance: dict; code_version: str
    observables: dict[str, list[float]]      # 抽稀后的时间序列，含 "t"
    final_state_label: str | None; weight: float = 1.0
    parent_digest: str | None = None
class DuplicateKeyError(Exception): ...
class Store:
    def __init__(self, path: str | os.PathLike): ...
    def append(self, rec: ShotRecord) -> None     # 单个事务写入；重复 key 抛 DuplicateKeyError
    def get(self, digest: str) -> ShotRecord
    def has(self, digest: str) -> bool
    def iter(self, **filters) -> Iterator[ShotRecord]   # 可按 kind、stop_reason、origin_label 过滤
def config_hash(obj: Mapping) -> str                  # 规范 JSON（键排序、浮点用 repr）后取 sha256
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 2.1 | 写入一条记录后读出 | 所有字段相等，浮点逐位相同 |
| 2.2 | 同一个 digest 写两次 | 抛 `DuplicateKeyError`，库里仍只有一条 |
| 2.3 | 库文件不提供任何更新或删除接口 | 通过 `Store` 做不到覆盖（API 层面） |
| 2.4 | 两个 dict 内容相同、键顺序不同 | `config_hash` 相同 |
| 2.5 | `{"dt": 0.002}` 与 `{"dt": 0.0020000001}` | `config_hash` 不同 |
| 2.6 | 写到一半时模拟异常（事务内 raise） | 库中不出现半条记录（配合 Review Focus 3） |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: append-only shot store`

---

### Task 3: 后端协议 + 解析势 + PES 一致性测试

**Files:** Create `backends/base.py`、`backends/analytic.py`、`backends/pes_suite.py`、`tests/test_analytic_pes.py`、`tests/test_pes_suite.py`

**Interfaces — Produces:**

```python
@dataclass
class MDState: x: ndarray; v: ndarray; t: float; box: ndarray | None = None
class Propagator(Protocol):
    def run(self, n_steps: int) -> None
    def get_state(self) -> MDState
    def set_state(self, s: MDState) -> None
class PotentialBackend(Protocol):
    kind: Literal["analytic", "openmm", "openmm+ml", "openmm+qm"]
    gpu_resident: bool
    def build(self, s: MDState, cfg: "PhysicsConfig", rng_key: Key) -> Propagator
    def energy_forces(self, x: ndarray) -> tuple[float, ndarray]
    def provenance(self) -> dict
# analytic.py 中的势函数，统一接口 energy_grad(x) -> (E, dE/dx)，约化单位：
class FreeParticle(dim: int)
class DoubleWell1D(barrier: float, x0: float = 1.0)         # V = barrier·((x/x0)² − 1)²
class DoubleWell2D(barrier: float, ky: float)                # V = barrier·(x² − 1)² + ½·ky·y²
class MullerBrown(scale: float = 1.0)                        # 标准四项参数
class ChannelDoubleWell2D(barrier_plus: float, barrier_minus: float, wall: float)
    # y>0 与 y<0 两个通道，中间墙高 wall（取 ≫ kT，不能互通），x 方向势垒高度依通道而定；用于 Task 11 的带记忆反例
class AnalyticBackend(PotentialBackend):
    def __init__(self, potential, integrator: Literal["overdamped", "baoab"], dt: float, kT: float, gamma: float, mass: float = 1.0): ...
@dataclass
class PESReport: passed: bool; fd_max_rel_err: float; translation_err: float | None; rotation_err: float | None; repeat_bitwise: bool; nve_rel_drift: float | None
def pes_consistency_suite(backend: PotentialBackend, probes: list[ndarray], fd_step: float = 1e-5, rtol: float = 1e-4, check_invariance: bool = False, nve_steps: int = 0) -> PESReport
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 3.1 | 每个解析势，20 个随机探针点 | 力等于 −∇E，中心差分相对误差 < 1e-6 |
| 3.2 | `DoubleWell1D(5.0)` | 极小点在 x = ±1，势垒顶在 x = 0，高度为 5.0 |
| 3.3 | `MullerBrown()` | 三个极小点与文献坐标一致，误差 < 1e-3 |
| 3.4 | 把一个后端故意改坏（力乘以 1.01） | `pes_consistency_suite(...).passed is False` |
| 3.5 | 同一个 x 重复调用 `energy_forces` | `repeat_bitwise is True` |
| 3.6 | `check_invariance=True`，用多原子 LJ toy | 平移、旋转不变误差 < 1e-10 |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: backend protocol, analytic potentials, PES suite`

---

### Task 4: 解析传播器（overdamped Langevin / BAOAB）

**Files:** Modify `backends/analytic.py`；Create `tests/test_analytic_dynamics.py`

**Interfaces:** 延续 Task 3 的 `AnalyticBackend.build(...) -> Propagator`。随机力全部来自 `derive_rng(rng_key, "propagate")`。

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 4.1 | `FreeParticle(3)`，overdamped，D = kT/γ，N = 2000 条，t = 10 | MSD / (6Dt) = 1 ± 3σ |
| 4.2 | `DoubleWell1D(3.0)`，BAOAB，长轨迹 1e6 步 | x 的直方图与 Boltzmann 分布的 KS 检验 p > 0.01 |
| 4.3 | 同一个 `rng_key`、同一初态，跑两次 | 轨迹逐位相同 |
| 4.4 | 两个不同的 `rng_key` | 第 10 步起轨迹就已不同 |
| 4.5 | BAOAB 取 γ = 0，谐振子，1e5 步 | 相对能量漂移 < 1e-4（NVE 极限） |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: analytic Langevin propagators`

---

### Task 5: 系综帧、采样器与 IC 门禁

**Files:** Create `ic/frames.py`、`ic/sampler.py`、`tests/test_ic.py`

**Interfaces — Consumes:** Task 1 的 `ShotKey`、`derive_rng`；Task 3 的 `PotentialBackend.energy_forces`。**Produces:**

```python
@dataclass
class EnsembleFrame:
    coordinates: ndarray; box: ndarray | None; topology_ref: str
    temperature: float; weight: float; source_id: str; frame_id: int; time: float
class EnsembleFramePool:
    def __init__(self, frames: Sequence[EnsembleFrame], state_of: Callable[[EnsembleFrame], int] | None = None): ...
    def choose(self, rng, state: int | None = None) -> EnsembleFrame   # 按 weight 抽取，可限定在某个状态内
@dataclass
class InitialState: state: MDState; frame_id: int; meta: dict
@dataclass
class ValidityReport: ok: bool; reasons: list[str]; checks: dict[str, float]; n_redraws: int = 0
class EnsembleFrameSampler:
    def __init__(self, pool, masses: ndarray, kT: float, backend: PotentialBackend,
                 energy_window: tuple[float, float] | None, min_pair_dist: float | None,
                 state: int | None = None, max_redraws: int = 20, constraints: Callable | None = None): ...
    def sample(self, key: ShotKey) -> tuple[InitialState, ValidityReport]
        # 第 k 次重抽使用 derive_rng(key, f"ic/redraw{k}")；超过 max_redraws 抛 ICRejectedError（带全部原因）
    def validate(self, s: InitialState) -> ValidityReport
        # 检查：坐标与速度有限、约束残差、最近原子对距离、E_pot 在窗口内、瞬时温度在 kT 的 ±5σ 内
class ICRejectedError(Exception): ...
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 5.1 | 3 帧，权重 1:2:7，抽 1e5 次 | 各帧频率与权重之比的偏差在 3σ 内 |
| 5.2 | 同一个 `ShotKey` 采样两次 | `InitialState` 逐位相同 |
| 5.3 | 帧坐标里含 NaN | `ok=False`，reasons 中含 `"nonfinite_x"`，**绝不返回 ok=True**（h2oleps 回归测试） |
| 5.4 | 势能在窗口外 | 拒绝；自动重抽；`n_redraws ≥ 1` 并记录在案 |
| 5.5 | 所有帧都非法 | 抛 `ICRejectedError`，其中列出所有原因 |
| 5.6 | MB 速度，1e4 个样本，每个自由度 | ⟨½mv²⟩ = ½kT ± 3σ；去掉 COM 动量后总动量 < 1e-12 |
| 5.7 | 限定 `state=s` | 只会抽到 `state_of(f) == s` 的帧 |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: ensemble sampler with IC gate`

---

### Task 6: Observable、区域与在线停止判据

**Files:** Create `observe/events.py`、`tests/test_events.py`

**Interfaces — Produces:**

```python
Observables = dict[str, float]
@dataclass(frozen=True)
class Region: name: str; predicate: Callable[[Observables], bool]
@dataclass
class StopDecision:
    reason: Literal["fixed_lag", "A", "B", "reaction", "escape", "timeout", "pes_uncertain"]
    event_time: float | None
class StopRule(Protocol):
    kind: Literal["fixed_lag", "absorbing_AB", "b_surface"]
    def update(self, obs: Observables, t: float) -> StopDecision | None
    def reset(self) -> None
class FixedLag(StopRule):    def __init__(self, tau: float)
class AbsorbingAB(StopRule): def __init__(self, A: Region, B: Region, tau_persist: float, t_max: float)
class BSurface(StopRule):    def __init__(self, reaction: Region, r_name: str, q: float, tau_persist: float, t_max: float)
def offline_replay(rule: StopRule, series: dict[str, ndarray]) -> StopDecision | None   # series 中含 "t"
```

**语义（锁定）：**
- 区域内的停留时间要 ≥ `tau_persist` 才算事件；`event_time` 取**进入区域的时刻**。
- 在满足持久性之前离开区域，计时清零。
- 逃逸（`r ≥ q`）不需要持久性。
- 到 `t_max` 时，返回 `timeout`。

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 6.1 | 进入 B 后停留 0.8·τ_p，离开，再进入并停留 1.2·τ_p | 事件发生，`event_time` 等于第二次进入的时刻 |
| 6.2 | 在 A 和 B 之间快速抖动，每次停留都 < τ_p | 在 t_max 之前不触发 |
| 6.3 | 同一条序列分别在线和离线回放 | 两者的 `StopDecision` 完全相同 |
| 6.4 | `BSurface`，r 越过 q | `escape`，不等待持久性 |
| 6.5 | 始终不进入任何区域 | 在 t_max 返回 `timeout`（Review Focus 5） |
| 6.6 | `FixedLag(τ)` | 恰好在 t = τ 时返回 `fixed_lag` |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: persistent-event stop rules`

---

### Task 7: `run_shot` 与批执行器

**Files:** Create `engine/shot.py`、`exec/batch.py`、`tests/test_engine.py`

**Interfaces — Consumes:** Task 1、2、3、5、6。**Produces:**

```python
@dataclass
class ObsSpec: fns: dict[str, Callable[[MDState], float]]; dt_obs: float; store_stride: int
def run_shot(key: ShotKey, sampler, backend: PotentialBackend, stop: StopRule, obs: ObsSpec,
             physics_cfg: "PhysicsConfig", store: Store, labeler: Callable[[Observables], str] | None = None) -> ShotRecord
    # 流程：sample（经过门禁）→ build → 每 dt_obs 推进一次并更新 stop → 生成记录 → store.append
def run_batch(keys: Sequence[ShotKey], shot_fn: Callable[[ShotKey], ShotRecord], store: Store,
              n_workers: int = 1) -> list[ShotRecord]
    # 跳过 store.has(digest) 为真的 key；返回结果按 keys 的顺序排列
```

**必测用例（全部用 Task 3 的解析后端）：**

| # | 输入 | 期望 |
|---|---|---|
| 7.1 | 100 个 key，分别用 `n_workers=1` 和 `n_workers=4` | 两次得到的记录逐字段相同 |
| 7.2 | key 顺序打乱后再跑 | 每个 key 对应的记录不变 |
| 7.3 | 某条 IC 被拒绝，但重抽后成功 | 记录里 `ic_validity.n_redraws ≥ 1` |
| 7.4 | 跑到第 50 个 key 时杀掉进程，然后 `run_batch` 续算 | 最终库与一次跑完的库逐条相同（Review Focus 3） |
| 7.5 | 记录的 `physics_config_hash` | 与 `config_hash(physics_cfg)` 相等 |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: shot runner and resumable batch executor`

---

### Task 8: 估计量与分解

**Files:** Create `estimate/msm.py`、`estimate/committor.py`、`estimate/association.py`、`estimate/decompose.py`、`tests/test_estimators.py`

**Interfaces — Produces:**

```python
@dataclass
class TEstimate: T: ndarray; ci_low: ndarray; ci_high: ndarray; its: ndarray; ck_passed: bool; ck_max_dev: float
def estimate_T(start_states: ndarray, end_states: ndarray, weights: ndarray, n_states: int, lag: float,
               reversible: bool, n_boot: int, rng) -> TEstimate                          # 设计 §4.1；reversible 调用 deeptime
def ck_test(dtrajs: list[ndarray], lag_steps: int, ks: Sequence[int], n_states: int, n_boot: int, rng) -> tuple[bool, float]
@dataclass
class CommittorEstimate: q: float; ci: tuple[float, float]; n_A: int; n_B: int; n_timeout: int; timeout_frac: float; valid: bool
def estimate_committor(records: Iterable[ShotRecord]) -> CommittorEstimate   # 设计 §4.2；timeout_frac > 0.05 时 valid=False
def nam_beta_inf(beta: float, b: float, q: float) -> float                   # 设计 §4.3：β/(1 − (1 − β)·b/q)
@dataclass
class KonEstimate: beta: float; beta_inf: float; kon: float; ci: tuple[float, float]
def estimate_kon(records, b: float, q: float, D_AB: float) -> KonEstimate    # k_D(b) = 4π·D_AB·b
@dataclass
class Decomposition: delta: float; population: float; dynamical: float
def decompose(W: ndarray, Wp: ndarray, A: ndarray, Ap: ndarray) -> Decomposition   # 设计 §4.4 的中点公式
def hierarchical_bootstrap(groups: Mapping[int, Mapping[int, Sequence[float]]], stat: Callable, n_boot: int, rng) -> ndarray
    # 三层重抽：状态 → 帧 → shot
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 8.1 | 已知 3 态 Markov 链生成的数据 | `estimate_T` 与真值一致（在 CI 内）；CK 检验通过 |
| 8.2 | 隐藏态链（观测是两个隐态的混合，不满足 Markov） | CK 检验失败 |
| 8.3 | 100 条记录，其中 6 条 timeout | `valid=False`，`timeout_frac=0.06`（Review Focus 5） |
| 8.4 | `nam_beta_inf(1, b, q)` | 等于 1 |
| 8.5 | `nam_beta_inf(β, b, q→∞)` | 趋于 β |
| 8.6 | 随机的 W、W′、A、A′ | `population + dynamical == delta`，误差 < 1e-12 |
| 8.7 | 合成的分层数据，组间方差已知 | bootstrap 给出的方差与解析值一致（±10%） |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: T, committor, NAM, decomposition estimators`

---

### Task 9: A0 解析验收之一（自由扩散 + 双阱 committor）

**Files:** Create `examples/toy_diffusion/`、`examples/toy_doublewell/`、`tests/reference/committor_ref.py`（参考解求解器）、`tests/test_acceptance_a0.py`（标记为 `@pytest.mark.slow`）、`docs/reports/A0_part1.md`

**参考解（锁定）：**
- 一维 overdamped：\(q(x)=\int_a^x e^{\beta V}\,dy\big/\int_a^b e^{\beta V}\,dy\)，用数值积分求解。
- 二维：在网格上用有限差分求解 backward Kolmogorov 方程 \(L q=0\)（边界条件 \(q|_A=0\)、\(q|_B=1\)），并做网格收敛检验。

**验收：**

| # | 内容 | 阈值 |
|---|---|---|
| 9.1 | 三维自由扩散：吸收球半径 a，起点在半径 b = 2a 的球面，逃逸面 q = 8a；N = 4000 | 用 \(\beta\) 经 `nam_beta_inf` 换算后与 a/b 的差 ≤ 3σ；把 dt 减半后结果不变（离散化偏差检验） |
| 9.2 | 一维双阱（势垒 5 kT），20 个起点 × 400 shot | committor 与参考解的 RMSE < 0.03 |
| 9.3 | 二维 Müller–Brown，30 个起点 × 400 shot | RMSE < 0.03 |
| 9.4 | 用 `offline_replay` 重算全部记录 | 与在线判定 100% 一致 |

- [ ] 写参考解与验收测试 → 运行 → 写报告（数值 + 图）→ commit `test: A0 acceptance part 1`

---

### Task 10: `Resampler` 与带标签约束的 WE

**Files:** Create `resample/we.py`、`tests/test_we.py`

**Interfaces — Consumes:** Task 1 的 `IterKey`/`SegmentKey`；Task 3 的后端；Task 6 的停止判据；Task 2 的记录库（`kind="segment"`）。**Produces:**

```python
@dataclass
class Walker:
    segment_key: SegmentKey; parent: SegmentKey | None; origin_label: tuple[int, int]
    weight: float; z: ndarray; state: MDState
class Resampler(Protocol):
    kind: Literal["none", "uniform", "adaptive", "we", "revo"]
    def resample(self, walkers: list[Walker], it: int, key: IterKey) -> list[Walker]
class BinnedWE(Resampler):
    def __init__(self, bin_of: Callable[[ndarray], int], target_per_bin: int,
                 allow_cross_label_merge: bool = False, min_weight: float = 1e-250): ...
    # split/merge 采用 Huber–Kim 方式；merge 只在 (bin, origin_label) 相同的 walker 之间进行
@dataclass
class WERun: run_id: str; flux_to_sink: ndarray; n_eff: ndarray; n_walkers: ndarray; weights_sum: ndarray
def run_segment(w: Walker, backend, tau_seg: float, stop: StopRule, z_fn: Callable[[MDState], ndarray],
                store: Store, rng_key: SegmentKey) -> tuple[Walker, StopDecision | None]
def run_we(init: list[Walker], backend, resampler: Resampler, stop: StopRule, z_fn, n_iter: int, tau_seg: float,
           store: Store, global_seed: int, run_id: str, recycle_to: list[Walker] | None) -> WERun
    # recycle_to 非空时为稳态模式：进入 sink 的权重按原权重回收到 source
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 10.1 | 任意一轮 resample | 前后 Σw 之差 < 1e-12 |
| 10.2 | 两种标签混在同一个 bin 里 | 不发生跨标签 merge；`allow_cross_label_merge=True` 时才允许 |
| 10.3 | 同一个 `IterKey`，重跑 resample | 结果逐位相同 |
| 10.4 | 一维双阱（势垒 8 kT），稳态 WE | 通量 = 1/MFPT，与精确 MFPT 积分公式相比落在 5 次独立 WE 重复的 95% CI 内 |
| 10.5 | 两个标签分别从两个势阱出发 | 各标签的条件吸收概率与参考解一致（RMSE < 0.03） |
| 10.6 | `n_eff` 的计算 | 等于 \(1/\sum w^2\)（权重归一化后） |
| 10.7 | 连续强制 split 200 轮 | 没有 0、NaN 或低于 `min_weight` 的权重；Σw 守恒（Review Focus 1） |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: label-constrained weighted ensemble`

---

### Task 11: 吸收网络与 Markov 性检验

**Files:** Create `network/absorb.py`、`tests/test_network.py`、`docs/reports/A0_part2.md`

**Interfaces — Consumes:** Task 10 的 segment 记录。**Produces:**

```python
@dataclass
class StageNetwork: transient: list[str]; absorbing: list[str]; augmented: bool
def build_transitions(records: Iterable[ShotRecord], net: StageNetwork, milestone_of: Callable[[Observables], str | None],
                      label: tuple[int, int] | None) -> tuple[ndarray, ndarray]     # (Q, R)，由加权穿越计数估计
def solve_absorption(Q: ndarray, R: ndarray) -> ndarray                              # 𝓑 = (I − Q)⁻¹ R
@dataclass
class MarkovReport: passed: bool; max_dev: float; ci: tuple[float, float]
def markov_test(train, heldout, net: StageNetwork, milestone_of, label) -> MarkovReport
    # 用网络预测的吸收概率，对比留出数据中直接统计的吸收概率
```

**必测用例与验收：**

| # | 输入 | 期望 |
|---|---|---|
| 11.1 | 手工构造的 Q、R（已知解析解） | `solve_absorption` 误差 < 1e-12；各行之和为 1 |
| 11.2 | 二维双阱，在 x 方向铺 12 个 milestone | 网络给出的 committor 与 Task 9 的参考解相比 RMSE < 0.03 |
| 11.3 | `ChannelDoubleWell2D`（两个通道的势垒不同，墙 ≫ kT），标签 = y₀ 的符号，**所有标签合并**到一个网络 | `markov_test.passed is False` |
| 11.4 | 同一个体系，`augmented=True`，按标签分别建网络 | `passed is True`；各标签的 committor 与参考解相比 RMSE < 0.03 |

- [ ] 写测试 → 失败 → 实现 → 通过 → 写报告 → commit `feat: absorbing network with Markov test`

---

### Task 12: OpenMM 后端

**Files:** Create `backends/openmm_backend.py`、`tests/test_openmm_backend.py`

**Interfaces — Produces：** 实现 Task 3 的 `PotentialBackend`，另外：

```python
@dataclass
class PhysicsConfig:
    integrator: Literal["verlet", "langevin_middle", "nose_hoover"]; dt_ps: float; temperature_K: float
    friction_per_ps: float; constraints: Literal["none", "hbonds", "allbonds"]; rigid_water: bool
    platform: Literal["CUDA", "CPU", "Reference"]; precision: Literal["mixed", "double", "single"]
    deterministic_forces: bool; purpose: Literal["equilibration", "measurement"]
class OpenMMBackend(PotentialBackend):
    def __init__(self, system: openmm.System, topology, cfg: PhysicsConfig): ...
    # kind="openmm"，gpu_resident=True；积分器种子由 derive_rng(rng_key) 派生；observable 通过 reporter 以 dt_obs 为间隔在线计算
def com_distance(state: MDState, idx_a: ndarray, idx_b: ndarray, masses: ndarray) -> float   # 先拼回完整分子，再按最小镜像计算
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 12.1 | 丙氨酸二肽（真空，amber14），20 个探针构型 | `pes_consistency_suite` 在 Reference/double 下通过（rtol 1e-4） |
| 12.2 | 同一个构型，CUDA + `DeterministicForces=true`，重复算力 | 逐位相同；provenance 中记录了 GPU 型号、精度和 OpenMM 版本 |
| 12.3 | `purpose="measurement"` 且 `friction_per_ps=1.0` | 构造时报错（设计 §9） |
| 12.4 | 解析势（Task 3 的 `DoubleWell2D`）写成 OpenMM `CustomExternalForce`，在 Reference 平台上运行 | 能量和力与 `AnalyticBackend` 的差 < 1e-10。这证明 toy 上验证过的引擎逻辑，放到 OpenMM 这条生产路径上同样成立 |
| 12.5 | 同一个 `rng_key` 的 Langevin 积分，跑两次（CPU/Reference 平台） | 轨迹逐位相同 |
| 12.6 | 两个分子，其中一个跨越周期边界 | 跨越前后 `com_distance` 连续，跳变 < 1e-6 nm（Review Focus 2） |

- [ ] 写测试 → 失败 → 实现 → 通过（GPU 用例放前台运行，开跑前看 `nvidia-smi`）→ commit `feat: OpenMM backend`

---

### Task 13: 配置、mode 分发与 CLI

**Files:** Create `config/schema.py`、`cli.py`、`tests/test_config.py`、`examples/*/config.yaml`

**Interfaces — Produces:**

```python
class RunConfig(pydantic.BaseModel):
    mode: Literal["prepare", "shoot.ensemble", "shoot.encounter", "shoot.surface"]
    units: Literal["openmm", "reduced"]; system: SystemSpec; physics: PhysicsConfig; ic: ICSpec
    stop: StopSpec; observables: ObsSpecModel; resampler: ResamplerSpec; budget: BudgetSpec; seed: int
    store_path: str
def load_config(path: str) -> RunConfig          # 所有带量纲的字段都必须写成 "<值> <单位>" 的字符串，或显式带单位的字段
def dispatch(cfg: RunConfig) -> Callable[[], None]
# CLI：cytherea run <config> | cytherea resume <config> | cytherea report <store>
```

**必测用例：**

| # | 输入 | 期望 |
|---|---|---|
| 13.1 | 4 种 mode 的最小合法配置 | 加载成功，并分发到对应的 runner |
| 13.2 | 缺少必填字段 | pydantic 报错，信息中指出字段路径 |
| 13.3 | `b: "20 angstrom"` | 转换为 2.0 nm；`b: 20`（没有单位）→ 报错（Review Focus 4） |
| 13.4 | `cytherea resume` 中断后续算 | 行为与用例 7.4 一致 |
| 13.5 | 键顺序不同的两份配置 | 规范化后的 `config_hash` 相同 |

- [ ] 写测试 → 失败 → 实现 → 通过 → commit `feat: config schema, mode dispatch, CLI`

---

### Task 14: A1 丙氨酸二肽（显式水）

**Files:** `examples/alanine_dipeptide/`（14a 参考脚本已完成；14b 新增 `shoot_a1.py`：`frames` / `configs` / `analyze` / `pes` 子命令）、`cytherea.estimate.msm.ck_test_shots`、`docs/reports/A1.md`

**协议（锁定；2026-10-03 按用户决定修订，解决 review_handoff DOC-01）：**
- 力场：amber14-all + TIP3P-FB；300 K；约束 HBonds，刚性水；dt = 2 fs；测量阶段 Langevin(Middle) γ = 0.1 ps⁻¹（与参考相同）。
- **参考**（14a，已完成）：`runs/ala2_ref`（237 ns，2 段）+ `runs/ala2_par/r01–r08`（各 100 ns），原始 1037 ns，每条去掉 10 ns 预平衡后 947 ns、10 条独立轨迹。**用户 2026-10-03 确认按此验收**（多条独立轨迹可以代替“一条 ≥1 μs”）。状态 C7eq/C5、αR、αL（`ala2_common` 的 core 定义），14b 的 τ = `shoot_lag_ps` = 100 ps。
- **起始帧**：只取参考 DCD（每 10 ps 一帧）中 raw core label ≥ 0 的帧（跳过每条 10 ns 预平衡），每个状态 50 帧；在每个状态的候选帧（按轨迹、时间排序）上做带随机起点的等距抽样，使帧分散到各轨迹和时间段。每帧记录：来源轨迹、DCD 帧号、时间、φ/ψ、raw core label、所属 core 驻留段（visit）编号、box。坐标处理：按分子整体平移回盒内（float64），IC 门禁重新投影约束、用最小镜像的 `min_pair_dist`；速度由 IC 采样器按 Maxwell–Boltzmann 重抽。
- **射击（长 shot，用户 2026-10-03 选定 14.3 的方案 A）**：每帧 **10 发**，`FixedLag(5500 ps)`（= 55 τ ≈ 2·t2），φ/ψ 每 1 ps 记录。每发的前 100 ps 就是 τ shot（供 14.2），整条给出 k = 1…55 的 T(kτ)（供 14.3）。总积分量 150 × 10 × 5.5 ns = 8.25 μs；开跑前在 GPU 空闲时实测吞吐（单进程 / MPS 并行），把预算报用户。分片：同一个帧文件，按 `budget.frames` 分成若干配置、各写自己的 store，用 MPS + 绑核并行。
- **标注**：`ala2_common.label_shot`（带初始标签的 TBA，1 ps）；t = kτ 处的标签即终态。
- **估计量**：比较**行归一化**的 core-start T（`estimate_T(..., reversible=False)`，按帧 cluster）；ITS 用 reversible 估计，只用于时间尺度。另报以 core 驻留段为 cluster 的区间（αL 只有约 12 段，帧数 ≠ 独立样本数）。
- **surface**（14.4）：在 αR ↔ C7eq 分界面附近取 30 帧，每帧 100 shot，`AbsorbingAB`；单独一批，预算另报。

**接口：**
- `ck_test_shots(start_states, end_states_by_k, ks, n_states, n_boot, rng, frame_ids=None) -> CKResult`：shot 版 CK。`end_states_by_k[:, m]` 是每发在 t = ks[m]·τ 的状态，`ks` 必须含 1。T(kτ) 和 T(τ) 都只用 t = 0 出发的窗口（core-start），行归一化。统计量与 `ck_test` 相同：D = max_{k,ij} |T(τ)^k − T(kτ)|；零分布用按起始状态分层的 bootstrap（有 `frame_ids` 时以帧为单位），两项在同一重抽样上重估、以观测偏差为中心；D ≤ 95 分位为通过。
- `shoot_a1.py frames`：输出 `frames.npz`（cytherea 帧格式）和 `frames.json`（上面列的每帧元数据）。
- `shoot_a1.py analyze`：读全部分片 store，输出 `analysis_14b.json`：T(τ)（元素、Jeffreys 区间、n_eff）、与参考 contract T 的逐元素对照、ITS（t2、t3 及 bootstrap CI）、CK 结果、IC 拒绝率与原因、失败记录、帧与驻留段的相关性统计。

**测试：**
- `ck_test_shots`：3 态 Markov 链合成 shot 通过；带隐藏记忆的 lumped 链在长 horizon 上不通过；`ks` 不含 1、形状不符、某起始状态无 shot 时报错；结果只依赖 shot 集合，与顺序无关。
- DCD 读帧：与 OpenMM 写出的小 DCD 往返一致（坐标、box）；分子整体回盒后分子内距离不变。
- 帧选择：只选 core 内部帧、跳过预平衡、每状态数目正确、可复现（同 seed 同结果）。
- analyze：合成记录（已知 T 的 Markov 链生成 φ/ψ 序列）恢复 T 并通过 CK。

**验收：**

| # | 内容 | 阈值 |
|---|---|---|
| 14.1 | `pes_consistency_suite`（溶剂化体系，CUDA mixed，sampled 模式，`fd_atom_groups` = 溶质；后端声明截断，FD 跨截断的坐标跳过并报告数目） | 通过（mixed 精度的容差行，含 `repeat_rtol` 1e-3） |
| 14.2 | 射击 T(τ) 的最慢 ITS t2 | 落在参考 core-start t2 的 95% CI 内（[1.51, 3.96] ns）；同时报告 t3 与逐元素 T 对照。t2 只有约 12 个参考事件支撑，是宽松检验 |
| 14.3 | 射击数据自身的 CK（`ck_test_shots`，k = 1…55；T(τ) 用全部 shot 的第一个 τ，T(kτ) 用长 shot，2026-10-04） | 通过 |
| 14.4 | 分界面 committor 分布 | 峰值在 0.5 附近，并如实报告直方图（诊断） |
| 14.5 | 所有记录的 `ic_validity` | 拒绝率 < 1%，全部原因已汇总 |

- [ ] `ck_test_shots` + 帧工具（测试先行）→ 实测吞吐、报预算 → 跑长 shot → 14.1 → analyze → 14.4 → 写 `docs/reports/A1.md`

### Task 15: A2 chignolin

**Files:** Create `examples/chignolin/`、`docs/reports/A2.md`

**协议：**
- 力场：CHARMM36m + TIP3P（CHARMM 修正版）；340 K；约束 HBonds；dt = 2 fs。开工前先确认 OpenMM 8.5.2 自带的 charmm36 xml 是否包含 36m 修正，并写进报告；如果不包含，报告给用户，不要自行替换力场。
- **参考**：总计 ≥ 10 μs 的长轨迹。可以拆成多条，要求双向各有 ≥ 10 次折叠/去折叠事件。
- 状态：用 RMSD（相对 NMR 结构）加上 CA–CA 接触的 TICA，聚为 folded / unfolded / misfolded。

**验收：**

| # | 内容 | 阈值 |
|---|---|---|
| 15.1 | 射击 \(T(\tau)\) 给出的折叠与去折叠时间 | 落在自跑参考 MSM 的 95% CI 内；与文献同一数量级（只报告，不作为门槛） |
| 15.2 | 用 `AbsorbingAB` 算 folded / unfolded 之间的 committor | 与参考 MSM 的 committor 相比 RMSE < 0.1 |
| 15.3 | 算力记录 | 报告 ns/day 和 GPU·天，作为 Phase C pilot 的外推基准 |

**预算门**：先在 2080 Ti 上实测 ns/day，据此估算参考轨迹需要的天数，**报用户确认后再跑长轨迹**。

- [ ] 实测吞吐 → 报用户 → 跑参考 → 射击 → 写报告 → commit `test: A2 chignolin acceptance`

---

### Task 16: A3 encounter 功能测试（隐式溶剂，b 面）

**Files:** Create `ic/encounter.py`、`tests/test_encounter.py`、`examples/encounter_pair/`、`docs/reports/A3.md`

**Interfaces — Produces:**

```python
class EncounterSampler:
    def __init__(self, pool_A: EnsembleFramePool, pool_B: EnsembleFramePool, b: float, masses: ndarray,
                 kT: float, backend: PotentialBackend, min_pair_dist: float, label: tuple[int, int]): ...
    def sample(self, key: ShotKey) -> tuple[InitialState, ValidityReport]
        # A 取自状态 i，B 取自状态 j；B 做 SO(3) 均匀旋转，COM 放在 S² 上的均匀方向、距离 b 处；随后经 IC 门禁
```

**体系**：barnase–barstar，OpenMM GBn2 隐式溶剂，300 K，Langevin γ = 0.1 ps⁻¹（用于测量），结合判据用接触数加持久性判据。本任务只验证流程，**不与实验 \(k_{\rm on}\) 做定量比较**（设计 §12 A3）。

**必测用例与验收：**

| # | 输入 | 期望 |
|---|---|---|
| 16.1 | 1e4 次 `EncounterSampler.sample` | COM 距离恒等于 b（误差 < 1e-9 nm）；取向四元数的分布与 Haar 测度的 KS 检验 p > 0.01 |
| 16.2 | 同一个 key，采样两次 | 逐位相同 |
| 16.3 | 分别取 q₁ = 2b 和 q₂ = 3b，暴力 b 面射击 | 两者经 NAM 修正后的 \(\beta_\infty\) 在 CI 内一致 |
| 16.4 | 同一个体系用 WE（Task 10，进展坐标 = COM 距离 + 接触数） | β 与暴力射击的结果在 CI 内一致 |
| 16.5 | 报告 | 实测 ns/day、每条 shot 的中位停止时间、WE 相对暴力射击的算力比 |

- [ ] 写 16.1、16.2 的测试 → 失败 → 实现 → 通过 → 前台跑 16.3、16.4 → 写报告 → commit `test: A3 encounter functional test`

---

### Task 17: WE 引擎选型评估（wepy / WESTPA）

**Files:** Create `docs/reports/we_engine_evaluation.md`

**要求：** 这个任务**只评估，不安装进 `openmm_dev`**。需要试用时，另建一个临时环境 `cytherea_eval`，评估结束后删除。

报告必须回答以下问题：
1. 许可证：原文、能否作为依赖使用。
2. 能否实现 Task 10 的标签约束 merge（通过自定义 resampler 实现）。
3. 能否使用我们的 `derive_rng(IterKey)`，保证 resample 逐位可复现。
4. 与 Task 12 的 OpenMM 后端对接的方式和每步开销。
5. 在用例 10.4 上与自研 `BinnedWE` 的结果和速度对比。

最后给出结论：保持自研，或者在 Phase C 前换用外部引擎，并通过适配器接到 `Resampler` 接口后面。

- [ ] 评估 → 写报告 → 删掉临时环境 → commit `docs: WE engine evaluation`，**等用户决定**

---

## 不在本计划内（后续计划处理）

- 远场 BD、首次击中分布、BD/MD 界面检验（设计 §4.3，Phase C 之前另立计划，BD 工具选型同时进行）。
- REVO、MAB 这类自适应分箱（Task 17 结论出来后再定）。
- `openmm+ml`、`openmm+qm`（openmm-orca / openmm-pyscf）的 QM/MM（Phase F）。本项目不接 ASE。
- Aβ42 的一切内容（Phase B0 起）。

## 自查记录（写计划时做过的）

- **设计覆盖**：
  - §1 的模块映射对应 Task 1–13；
  - §3.5 的后端对应 Task 3、12；
  - §4.1–4.5 对应 Task 8、10、11；
  - §6 对应 Task 6；
  - §7 对应 Task 1、5、12；
  - §9 对应 Task 12（12.3）、14、15；
  - §12 的 A0–A3 对应 Task 9–11、14–16；
  - §3 的 WE 选型对应 Task 17。
  - 未覆盖的部分已列入上面“不在本计划内”。
- **类型一致性**：`ShotKey`、`SegmentKey`、`IterKey`、`MDState`、`InitialState`、`ValidityReport`、`StopDecision`、`ShotRecord`、`Walker`、`PhysicsConfig` 在各任务里的名称和字段一致。
- **没有占位符**：没有 TBD，也没有“同上”。
