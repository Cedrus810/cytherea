# Cytherea：面向蛋白质的 VENUS96 现代化
## 设计 v2（取代 v1 `VENUS96_AB42_IDP_design.md` 中的软件路线；v1 的科学问题全部保留）

> 本文只写规格：边界、接口签名、锁定的数学定义、必测用例、验收阈值，不写实现。

---

# 0. 决策记录（2026-09-30）

1. **F77 代码整体放弃**：不移植，不包装，也不作为 oracle 保留。
2. **首要目标是“VENUS96 现代化 for 蛋白质”**：用现代技术栈（Python + OpenMM）重建 VENUS96 的功能结构，目标体系是溶液中的蛋白质，不再是气相小分子。
3. **Aβ42 是第一个应用**，放在 `examples/` 下，不进入核心代码。
4. **小分子气相 RRHO/QCT 不重复实现**：这部分已由 VENUSpy（JCTC 2025，第 2.5 节）覆盖。v1 中的 `LegacyRRHOQCTSampler`、local RRHO、ZPE 模块从核心中删除。
5. **PES 层必须有**（2026-09-30 补充）：引擎不能只绑死一种经典力场。第 3.5 节定义分层的 PES 后端，**全部走 OpenMM 这一个引擎**：ML 势用 openmm-ml，QM/MM 用 openmm-orca / openmm-pyscf。**不接 ASE**（2026-09-30 决定）。

与 v1 相比的主要改动：

| # | v1 | v2 |
|---|----|----|
| 1 | “重构 VENUS96”，Phase 0 解耦 F77 | 不重构 F77：按第 1 节的**模块映射**用现代栈重建 |
| 2 | 固定 τ 的 \(P(B\mid X)\) 被称为 “committor-like” | 拆成三个互不替代的估计量：\(T_{ij}(\tau)\)、committor \(q\)、结合概率 \(\beta_\infty\)（第 4 节） |
| 3 | \(P_{\rm assoc}(i,j)\) 没有给出定义 | 用 NAM b-surface 形式定义，同时给出 \(k_{\rm on}\)，对应 VENUS 的 \(b\)、\(\sigma\)（4.3 节） |
| 4 | mutation 分解只有定性描述 | 精确的两项中点分解 + 分层 bootstrap；要求公共状态空间（4.4 节、第 5 节） |
| 5 | 没有提力场、水模型、恒温器 | 列为一等设计参数（第 9 节） |
| 6 | 没有算力估计 | 成本公式 + pilot 门 + 多保真度 + 自适应分配（第 10 节） |
| 7 | 没有项目终止判据 | 时间尺度分离前提检验（第 8 节） |
| 8 | 可复现性只提到种子 | IC 级逐位可复现 + IC 有效性门禁，源自 venus96 实测暴露的问题（第 7 节） |
| 9 | dimer：从远处开始，均匀射击 | 分阶段（扩散 → encounter → 成熟）+ 带谱系约束的 WE；`Resampler` 升为一等接口；第一篇论文只做相对 \(p_\beta\)（4.3、4.5 节） |

---

# 1. “现代化 VENUS96 for 蛋白质”：模块映射

Cytherea 在功能上与 VENUS96 一一对应，所以它是 VENUS 的现代化，而不是另起炉灶的通用工具。下表左列取自对 venus96.f 的实测审阅（2026-09-29）。

| VENUS96 模块（F77） | 职责 | Cytherea 对应物（蛋白） | 变化 |
|---|---|---|---|
| 主程序输入 deck：21 个整数 + 各类势参数 | 定义体系与势能 | `config.yaml` + OpenMM `System`（拓扑 + 标准力场） | 势能不再手写 |
| `NSELT` 运行模式 | 选择计算类型 | `mode:` 字段（见下一张表） | 保留“一个程序、多种实验”的结构 |
| `NACT` 采样器（`ORTHAN`、`INITQP` 简正、`LMEXCT` 局域模、`THRMAN` Boltzmann） | 单分子初始条件 | `EnsembleFrameSampler`：从外部系综按权重取帧，再做 MB 速度重采样 | RRHO 换成系综采样 |
| 双分子采样：`NOB`/`BMAX` 碰撞参数、`EREL`/平动 Boltzmann、`ROTATE` 随机取向、初始间距 | A+B 碰撞初态 | `EncounterSampler`：b-surface 放置、SO(3)×S² 均匀取向、热化速度 | 气相弹道碰撞改为溶液扩散碰撞（4.3 节） |
| `DVDQ`/`ENERGY` + 约 20 种解析势 | 力与能量 | `PotentialBackend`：第一版只有 OpenMM | 由外部力场提供 |
| `RUNGEK` 启动 + `ADAMSM` 6 阶 PECE，NVE | 传播 | OpenMM 积分器（Verlet NVE 或低摩擦 Langevin），带约束 | 第 9 节锁定 |
| `NSELT=1` 能量极小 | 结构准备 | OpenMM 极小化 + 平衡化协议 | — |
| `TEST`：`RBAR`/`RMAX` 事件、`NPATHS` 反应通道、`NAST` 状态 | 在线事件检测、终止轨迹 | `StopRule` + `OutcomeClassifier`（多通道，带持久性） | 在线执行（第 6 节） |
| `FINAL`/`GFINAL`：产物的转动/振动能、角动量、相对能 | 终态分析 | 终态 observable：接触、\(R_g\)、SASA、二级结构、状态标签 | 按蛋白体系重新定义 |
| `ENMODE`/`EBOND`（`MPLOT`）模式能量监测 | 轨迹过程中的监测 | observable 时间序列（按步长抽稀存储） | — |
| unit 50 checkpoint（`NCHKP`） | 断点续算 | 每条 shot 一条只追加的记录；长 shot 另存 OpenMM checkpoint；续算 = 补跑缺失的 `ShotKey` | 不再依赖整体状态文件 |
| `RANDST`/`RAND0`/`RAND1`：一条全局随机数流 | 随机数 | 按键派生的种子：`SeedSequence(global, frame, shot, stage)` | **关键改动**：旧设计中第 n 条轨迹的初态依赖前 n−1 条消耗的随机数，因此无法并行 |
| `DO` 循环里依次跑 NT 条轨迹（`GOTO 451`） | 轨迹批处理 | 批执行器：多 GPU、多 context，彼此独立 | embarrassingly parallel |
| unit 6 格式化文本输出 | 结果 | 结构化记录（Parquet 或 SQLite）+ 估计量报告 | 可以完整重分析 |
| `SYBMOL` unit 8 坐标 | 可视化 | DCD/XTC 轨迹文件（可选、按步长抽稀） | — |
| `NMODE`（`NSELT=-1`）、`MPATH`（`NSELT=-2`） | 简正分析、反应路径 | **第一版不迁移** | 蛋白体系上意义有限 |

`mode:` 字段与 `NSELT` 的对应关系：

| `mode` | 对应 NSELT | 用途 |
|---|---|---|
| `prepare` | 1 | 极小化与平衡化 |
| `shoot.ensemble` | 0 / 2（单分子） | 从系综帧出发射击：得到 \(T(\tau)\) 或 committor |
| `shoot.encounter` | 2（A+B） | 双分子 encounter：得到 \(\beta_\infty\)、\(k_{\rm on}\) |
| `shoot.surface` | 3（从势垒出发） | 从分界面或过渡态系综出发：committor、committor 分布检验 |

---

# 2. 为什么不能把 F77 放大

用数字说明为什么这一方向被否决，免得日后被重新提起：

- `ND1` 从 100 调到约 5×10⁴ 后，`COMMON/CHEMAC` 中 `CA`、`CB(3*ND1,3*ND1)` 两个矩阵约需 360 GB，全对距离数组 `R(ND1*(ND1+1)/2)` 约需 10 GB。
- F77 代码没有周期边界、没有 PME、没有约束、没有恒温器；NVE 多步积分器处理柔性水时，步长只能取约 0.5 fs。
- 全局状态放在 71 个 COMMON 块里，一次只能跑一条轨迹。

---

# 2.5 先行工作：VENUSpy，以及我们的边界

K. Fujioka, R. Richard, J. Waldrop, Y. Luo, T. L. Windus, R. Sun（夏威夷大学、爱荷华州立大学），*VENUSpy: A Chemical Dynamics Simulation Program in the Era of Machine Learning*，JCTC 2025, 21, 10679；代码在 github.com/kaka-zuumi/VENUSpy，pip 包名 `venuspython`。

2026-09-30 看了它的仓库（HEAD d4e38a4，2025-12-20）：

| 维度 | VENUSpy | Cytherea（本项目） |
|---|---|---|
| 目标体系 | 气相小分子反应动力学（例子：B+C₂H₂、CH+SH₂、HBr+HCl、O+O₂ 等） | 溶液中的蛋白 / IDP |
| 规模与底座 | 约 6k 行 Python，基于 ASE；ASE `VelocityVerlet`，每步在 Python 里调用 calculator | GPU 常驻传播（OpenMM），每步不回 Python |
| 初始条件 | 完整继承 VENUS：简正 / thermal / EBK 双原子 / 碰撞能 + 碰撞参数 + 随机取向 | 从系综取帧 + MB 速度；溶液 b-surface encounter |
| PES | 很丰富：ChemPotPy 解析 PES、sGDML、PhysNet、SchNet、xTB、MOPAC、Psi4、GAMESS（QCEngine）、BAGEL、NWChem、NWChemEx | 经典力场为主；ML 势和 QM/MM 通过 OpenMM 插件接入（第 3.5 节），不接 ASE |
| 特色 | ML/ab initio 混合动力学：`smoothedMD` 在检测到能量跳变时，用即时重训的 sGDML 接管，之后再切回 | 三个估计量（T、committor、k_on）、mutation 分解、provenance、按键派生的并行种子 |
| 周期边界、溶剂、恒温器、蛋白 | 无（只有 xtb/physnet 的 calculator 内部带 PBC 开关） | 核心需求 |

**结论**：

- “VENUS 的 Python 现代化”这件事**已经有人做了，而且发表了**，所以本项目不能以此作为卖点。
- 我们的差异在于**凝聚相与蛋白**：系综初态、溶液 encounter、GPU 常驻传播、动力学估计量。
- 这两者是互补关系，不是竞争关系。论文中必须引用 VENUSpy，并在 introduction 里讲清楚这条谱系：VENUS96 → VENUS/NWChem → VENUSpy（气相 + ML/QM）→ Cytherea（凝聚相 + 蛋白）。

**许可证风险（阻断项）**：VENUSpy 仓库里**没有 LICENSE 文件**，pyproject 里也没有 license 字段；只有从 tblite 拷来的 `xtbcalc.py` 带 LGPL 头。没有许可证就意味着默认保留全部权利。

- **不拷贝、不改写 VENUSpy 的任何代码**。
- 不与 VENUSpy 互通，因为我们不做气相小分子。
- 用户已决定（2026-10-01）：不联系作者询问许可证，此事不再跟进。

---

# 3. 系统组成

```text
cytherea/
  config/      YAML schema、mode 分发
  ic/          EnsembleFrameSampler、EncounterSampler、SurfaceSampler、速度重采样
  engine/      Shot 生命周期：构建 → IC 门禁 → 传播 → 在线停止判据 → 记录
  backends/    PotentialBackend：openmm（主干）、openmm+ml（openmm-ml/openmm-torch）、openmm+qm（openmm-orca / openmm-pyscf）、analytic（toy）
  observe/     在线 observable、事件与持久性、吸收边界
  estimate/    T(τ)、committor、β∞/k_on、mutation 分解、bootstrap
  store/       ShotRecord 与 provenance（只追加写入）
  exec/        批执行器（单机多 GPU；集群适配器留接口）
examples/
  toy_*/  alanine_dipeptide/  chignolin/  abeta42_monomer/  abeta42_dimer/
```

**复用而不重写**：

- 传播：OpenMM；
- 特征：MDTraj 或 MDAnalysis；
- TICA / VAMP / MSM：deeptime；
- WE：先评估 wepy（Python 原生、自带 OpenMM runner 和 REVO、支持自定义 resampler，便于实现 4.5 节的标签约束），WESTPA 作为备选。无论用哪个，外部都只暴露 `Resampler` 接口（第 11 节）。这个选型在 A0 结束前定下来，**先核实许可证**。

本项目自己写的只有：mode 分发、shot 编排、provenance、停止判据与分类器、估计量、mutation 分解，以及第 3.5 节的 PES 适配层。

---

# 3.5 PES 层（分层后端）

**硬约束**：蛋白规模的传播必须完全留在 GPU 上。OpenMM-Het 项目实测过（2026-09-29，RTX 5090）：每步 GPU↔CPU 往返的固定开销足以抵消任何收益。所以“每步回 Python 取一次力”的做法（也就是 ASE/VENUSpy 的模式）不进入本项目。唯一的例外是 QM/MM：QM 力本来就很贵，每 k 步回一次 CPU 的开销相对可以忽略（见下表）。

| 后端 | PES 来源 | 传播位置 | 适用规模 | 用在哪里 |
|---|---|---|---|---|
| `analytic` | 解析 toy 势（双阱、自由扩散、Müller–Brown） | NumPy/JAX，或 OpenMM `CustomExternalForce` | 1–10² 维 | A0 验收 |
| `openmm` | 经典力场（AMBER/CHARMM + 水模型） | OpenMM GPU | 10⁴–10⁶ 原子 | 主干，Phase A1 起 |
| `openmm+ml` | ML 势（openmm-ml 支持的 ANI、MACE、AIMNet2 等，或自训 TorchScript 模型） | 通过 openmm-torch `TorchForce` 留在 GPU | ML 全体系可到 10²–10⁴；ML/MM 可更大 | 局部反应区、力场精修（Phase F，可选） |
| `openmm+qm`（QM/MM） | openmm-orca（ORCA，OPI；电子嵌入、H link、ONIOM）或 openmm-pyscf（`PythonForce`） | OpenMM 传播；QM 力可以每步算，也可以用 MTS 每 k 步算一次 | 蛋白 + 10–10² 原子的 QM 区 | Phase F，可选。**依赖项**：openmm-orca 目前是 v0.3.0，只支持非周期体系；周期性 MM + cutoff 嵌入计划在 v0.4 实现。溶剂化蛋白的 QM/MM 要等 v0.4 |

**接口要求**：

- 所有后端都实现同一个 `PotentialBackend`（第 11 节），引擎代码不区分后端。
- 每个后端都必须通过 **PES 一致性测试**。测试分两层（2026-10-01 修订：原来“所有后端同一阈值”的要求在 mixed/single 精度和蛋白规模下无法满足）：
  - `strict`（解析后端，以及 Reference/double 平台）：对全部坐标做有限差分，严格容差（相对误差 < 1e-4）。
  - `sampled`（生产平台，mixed/single）：只对随机抽取的一部分原子做有限差分，容差按精度放宽，默认值写在文档里。
  - 两层都必须满足：出现任何 NaN/inf 都判失败；所有归一化都与能量零点无关（力按力的尺度，能量差按“力尺度 × 位移”，NVE 漂移按动能尺度）；NVE 检查从热化扰动后的状态出发，不能从极小点静止出发。
  - 其余检查不变：平移和旋转不变性（无 PBC 时）；同一构型重复调用，结果逐位相同，或差异在声明的容差内。
- **不接 ASE**（2026-09-30 决定）：ASE 能提供的只是气相小分子的 calculator 生态，而那正是 VENUSpy 做、我们不做的方向。多一个后端就多一套传播路径、一套单位和一套测试，却没有哪个阶段真正需要。`PotentialBackend` 协议保持开放，以后确有需要时再加。

## 3.6 混合 PES 策略：和 VENUSpy 不同（Phase F 才启用）

VENUSpy 的做法是：检测到能量跳变，就用即时重训的 sGDML 接管。这个做法**不适合**本项目，原因有三：

1. **规模不行**：sGDML 是面向小分子整体描述符的核方法，训练代价随原子数急剧上升。蛋白或几十到上百原子的反应区，没法在轨迹中途即时重训。
2. **会破坏 shot 独立性**：第 n 条轨迹的模型取决于前 n−1 条轨迹收集到的训练数据，结果随执行顺序而变。这正是 VENUS96 全局随机数流的问题（第 1 节）以另一种形式重现，违反第 7 节的原则。
3. **会破坏估计量的前提**：一条轨迹中途换了哈密顿量，它就不再是单一 PES 上的样本，第 4 节的三个估计量都不再有定义。

VENUSpy 要解决的是 SCF 不收敛这类 PES 故障，这个问题主干（经典力场）上根本不存在。所以混合 PES 只在 Phase F 的局部反应区（ML/MM）才出现，锁定规则如下：

- **轮内冻结**：一轮射击内，ML 模型固定不变。模型版本（权重哈希）写进 `ShotKey` 和 provenance。
- **用不确定度触发，而不是用能量跳变触发**：用 committee（≥4 个模型）的力分歧 \(\sigma_F\) 作为指标，每 \(\Delta t_{\rm obs}\) 在线检查一次。
- **超出置信域就停，不换势**：\(\sigma_F>\sigma_{\max}\) 时终止该 shot，并标记为 `stop_reason = "pes_uncertain"`。这样的 shot 不计入任何估计量，但它的比例必须报告；比例超过 5% 时，该轮估计无效。
- **在轮与轮之间做主动学习**：把被标记的构型拿去做 QM 标注，重训后进入下一轮。每一轮都是一个独立、可复现的统计实验。这正好利用了 VENUS 射击范式本身的批次结构。
- **起点用预训练模型**（如 MACE-OFF、AIMNet2 这类通用模型），在此基础上做局部微调，不从零开始学。

---

# 4. 三个估计量（锁定定义）

记号：
- \(S_i\)：公共状态空间中的状态（第 5 节）；
- 一次 shot =（帧 \(X_k\)，速度样本 \(p\)，停止规则）；
- \(w_k\)：帧的平衡统计权重，由外部 ensemble 模块提供。

## 4.1 转移矩阵 \(T_{ij}(\tau)\)：固定 lag

- **停止规则**：固定时长 \(\tau\)，不设吸收边界。
- **估计**：\(\hat T_{ij}(\tau)=\sum_{k\in S_i} w_k\,\hat n_{k\to j}/\sum_{k\in S_i} w_k\)。
- **前提**：帧按态内平衡分布 \(\propto w_k\) 抽取；需要细致平衡时用 deeptime 的 reversible 估计器。
- **必做检验**：implied timescales 随 \(\tau\) 收敛；Chapman–Kolmogorov 检验 \(T(k\tau)\approx T(\tau)^k\)，偏差在 bootstrap 95% CI 内。

## 4.2 Committor \(q_B(X)\)：首达，两端吸收

- **停止规则**：先到达 A 或 B 即停止（A、B 均带持久性判据）；最长时长为 \(t_{\max}\)，超时的 shot 单独计数。
- **估计**：\(\hat q_B=N_B/(N_A+N_B)\)。超时比例大于 5% 时，该估计无效。
- \(T_{ij}(\tau)\) 不是 committor。v1 §3.2 中的 \(P(B\mid X_k)\) 属于 4.1 节，而不是本节。

## 4.3 双分子结合：分阶段（`shoot.encounter`，2026-09-30 修订）

**主算法不再是“均匀射击，直到自己撞成稳定 dimer”**。过程拆成三段，每段用不同层级的方法：

\[
\text{扩散 encounter}\;\xrightarrow{\ \text{BD/解析}\ }\;\Sigma_{\rm enc}\;\xrightarrow{\ \text{Stage A: WE}\ }\;E_{\rm long}\;\xrightarrow{\ \text{Stage B: WE}\ }\;D_\beta
\]

- **显式溶剂里不模拟远距离扩散**。原子模拟从 encounter 面 \(\Sigma_{\rm enc}\) 开始。
- **\(\Sigma_{\rm enc}\) 上的入口分布必须按通量加权**（锁定）。只有当 \(\Sigma\) 取在相互作用已经各向同性的 NAM b 面时，均匀随机取向才成立。如果 \(\Sigma\) 取得很近（例如 \(r_{\min}\sim8\)–15 Å），Aβ42 的电荷分布会让进入 \(\Sigma\) 的取向和构象强烈偏向某些方向，这时均匀取 \(\Omega\) 是错的。入口分布要取外层 BD 模拟在 \(\Sigma\) 上的首次击中点分布（SEEKR2 式的 BD/MD 衔接）。二选一，在 Phase C 开工前锁定：
  - (i) \(\Sigma=b\) 面：均匀取向成立，代价是多模拟一段扩散；
  - (ii) 近处 \(\Sigma\) + BD 击中分布：成本更低，但需要 BD 外层。
- **显式溶剂初态**：每个入口构型都要先重新溶剂化，再做限制性平衡（溶质位置限制，逐步放开），然后释放速度。这一步的成本和偏差都要写入 provenance。
- **Stage A（encounter 发现）**：吸收边界是 \(\{r>r_{\rm esc}\}\) 和 \(E_{\rm long}\)（持久性判据）。大多数轨迹会很快逃逸，early termination 在这一阶段非常有效。
- **Stage B（dimer 成熟）**：初态**按 Stage A 进入 \(E_{\rm long}\) 的通量加权抽取**，不能挑选。吸收边界是回到 \(\Sigma\) 或逃逸，以及 \(D_\beta\)。
- **两阶段的拼接**：两阶段之间允许回流（B 中解离回到 A 的轨迹），所以总概率**不是**两段概率的简单乘积。要按 milestoning / Markov 网络求吸收概率：以各阶段的吸收计数作为边，用首达分析解 \(p_\beta\)。

**锁定的数据流**：

\[
P_iP_j\;\to\;\text{远场 BD}\;\to\;\rho_\Sigma(z\mid\ell)\;\to\;\text{带 origin label 的显式溶剂 WE}\;\to\;\text{吸收网络}\;\to\;p_\beta(\ell)\;\to\;P_\beta=\sum_{ij}P_iP_jp_\beta(i,j),\qquad \ell=(i,j)
\]

**锁定的定义**：

- 入口分布是 BD 的首次击中通量，不能在近场再做一次随机取向：
  \[
  \rho_\Sigma(z\mid\ell)=J_{b\to\Sigma}(z\mid\ell)\Big/\!\int_\Sigma J_{b\to\Sigma}(z'\mid\ell)\,dz'
  \]
- 各阶段的入口同理：
  \[
  \rho_B(z\mid\ell)=J_{A\to B}(z\mid\ell)\Big/\!\int J_{A\to B}\,dz'
  \]
- 吸收概率用 transient 转移矩阵 \(Q\) 和吸收转移矩阵 \(R\) 求解，二者都由 WE 的加权穿越计数估计：
  \[
  \mathcal B=(I-Q)^{-1}R
  \]

**必须补上的两个前提**（它们决定分解是否真的干净）：

1. **标签必须属于 Markov 状态本身。** 普通 milestoning 的核心假设是：轨迹一旦击中 milestone，就忘掉此前的历史，**这里面也包括 origin label**。如果把所有标签的穿越计数合并成一个 \(Q\)，网络就会在第一个 milestone 上把条件信息抹掉，\(p_\beta(\ell)\) 只剩入口分布那一点差别。因此网络定义在增广状态 \((m,\ell)\) 上（\(m\) 是 milestone 或 cell），每个 \(\ell\) 各自估计 \(Q^{(\ell)}\)、\(R^{(\ell)}\)，然后
   \[
   \mathcal B^{(\ell)}=(I-Q^{(\ell)})^{-1}R^{(\ell)}
   \]
   **Markov 性检验（必做）**：留出一部分 WE 数据，从中直接统计吸收概率，与网络预测值比较，差异要在 CI 内。检验不通过说明 milestone 太粗。IDP 链的构象就是典型的正交慢变量，这种情况下要加密 milestone，或者把构象特征并进 \(m\)。
2. **远场 BD 对 IDP 只是近似。** BD 把每个构象当作刚体，而 IDP 在 \(b\to\Sigma\) 的扩散过程中仍在变构象。对策：
   - 每个状态取多帧，BD 通量按帧加权平均；
   - \(\Sigma\) 不要放得太近，让构象和结合耦合明显的区域落在显式溶剂那一侧；
   - **BD/MD 界面一致性检验（必做）**：在 \(\Sigma\) 之外再设一个重叠 milestone，BD 和显式 MD 分别给出它的穿越通量与击中分布，两者在 CI 内一致；
   - BD 用的连续介质静电与显式溶剂不同，这一差异也由这个检验一并覆盖。

**第一篇论文的目标量：相对 \(p_\beta\)，不做绝对 \(k_{\rm on}\)**

\[
p_\beta(i,j)=P(D_\beta\text{ 先于逃逸}\mid \Sigma_{\rm enc},X_i,X_j),\qquad
P_\beta^{(v)}=\sum_{ij}P_i^{(v)}P_j^{(v)}\,p_\beta^{(v)}(i,j).
\]

这样做的理由是：A2T、A2V 以及各手性变体都**不改变净电荷**，所以远程的扩散 encounter 通量 \(k_{\rm enc}\) 近似与 variant 无关，比较 variant 时可以约掉。这个近似要验证：对每个 variant 做一次廉价的 BD 计算 \(k_{\rm enc}\)，差异 < 10% 才能约掉；否则把 \(k_{\rm enc}\) 保留在比较量里。4.4 节的分解中，\(A_{ij}\) 取 \(p_\beta(i,j)\)。

**绝对 \(k_{\rm on}\)（后续论文，保留定义）**：\(k_{\rm on}=k_{\rm enc}\cdot p_\beta^{\infty}\)，其中 \(p_\beta^{\infty}\) 已按下面的 NAM 公式做过回碰修正。另外还需要浓度 / 标准态修正、有限盒扩散修正，以及长寿命 dimer 的定义。

### NAM b-surface 公式（选项 (i)，以及绝对 \(k_{\rm on}\) 所需）

- **初态**：
  - A、B 两个分子分别取构象 \(X_i^A\)、\(X_j^B\)，放在 COM 距离 \(b\) 处；
  - 相对取向在 SO(3)×S² 上均匀抽样；
  - 在 \(r\ge b\) 区域，相互作用可近似为各向同性。
- **停止规则**：满足结合判据（带持久性）记为“反应”；COM 距离达到 \(q>b\) 记为“逃逸”。
- **锁定公式**（\(\beta\) 为从 \(b\) 出发、先反应后逃逸的概率）：

\[
\beta_\infty=\frac{\beta}{1-(1-\beta)\,\Omega},\qquad
\Omega=\frac{k_D(b)}{k_D(q)},\qquad
k_D(r)=4\pi D_{AB}\,r,
\]

\[
k_{\rm on}(i,j)=k_D(b)\,\beta_\infty(i,j),\qquad P_{\rm assoc}(i,j)\equiv\beta_\infty(i,j).
\]

- **要求**：
  - \(D_{AB}\) 从模拟中实测，并做 PBC 有限尺寸修正（Yeh–Hummer）；粘度校正暂不做（见第 9 节）；
  - \(q\) 必须明显小于盒子半宽。
- **与 VENUS 的对应**：\(b_{\max}\) 对应 b-surface；\(\sigma(E)=\pi b_{\max}^2 P_r\) 对应 \(k_{\rm on}=k_D(b)\beta_\infty\)。

## 4.5 Weighted Ensemble 与重采样（`Resampler`，一等接口）

WE 的轨迹段本身是**无偏 MD**，改变的只是“下一批算力投给谁”。clone 和 merge 精确守恒统计权重，所以动力学是正确的。Stage A 和 Stage B 都用 WE。

- **进展坐标是多维的**：
  \[
  z=(d_{\rm inter},N_{\rm contact},N_{\beta\text{-contact}},Q_{\rm CHC},Q_{\rm C\text{-term}},\Delta{\rm SASA}_{\rm buried},\ldots)
  \]
  不能只用 \(d_{\rm COM}\)：两条 Aβ42 碰上很容易，难的是碰上之后的重排。高维空间不用规则网格分箱，改用无网格或自适应方案：MAB（minimal adaptive binning）、Voronoi，或 REVO（在构型空间中 clone 稀缺的 walker，merge 过密的 walker）。
- **谱系约束**（锁定，这是本项目的特有要求）：\(p_\beta(i,j)\) 以初态 \((i,j)\) 为条件，所以每个 walker 都携带**起源标签** \((i,j)\)，**merge 只允许在同一标签内进行**。跨标签 merge 会抹掉条件信息，从而破坏 4.4 节的分解。
  - 代价是 WE 效率下降。标签粒度取公共状态空间的**状态**（\(K^2\) 类），不取单个帧。
  - 诊断量：起源标签与终态结局之间的互信息 \(I(\text{label};\text{outcome})\)。它 ≈ 0 就说明初态已被遗忘，与第 8 节的 kill 判据等价。这时允许跨标签 merge，但 4.4 节的分解要改为以 encounter 过程中的构象为条件。
- **误差条**：来自 ≥ 5 次**独立的 WE 重复**。walker 之间彼此相关，不能直接做 bootstrap。
- **速率与收敛**：用稳态 WE（sink 到 source 回收），或 RED 方案（用预稳态数据估计速率），并报告通量随迭代的收敛曲线。
- **有效样本数** \(N_{\rm eff}=1/\sum_k w_k^2\)（权重已归一化）：每次迭代都记录，它过低说明权重退化。
- **已知局限**：不在进展坐标里的慢自由度（IDP 链的构象弛豫正是这一类）会拖慢通量收敛。WE 缓解不了这个问题，只能靠第 10 节的 pilot 去量化。

## 4.4 Mutation 分解：精确，无残差

记 \(W_{ij}=P_iP_j\)，\(A_{ij}=\beta_\infty(i,j)\)（或 \(k_{\rm on}(i,j)\)），变体量加 ′，\(\bar x=(x+x')/2\)。则有

\[
\Delta\langle A\rangle=\underbrace{\sum_{ij}\Delta W_{ij}\,\bar A_{ij}}_{\text{population}}+\underbrace{\sum_{ij}\bar W_{ij}\,\Delta A_{ij}}_{\text{dynamical}}
\]

- 这是恒等式，不含交叉项。
- 误差用分层 bootstrap 估计：**状态固定，只在状态内重抽帧，每帧的 shot 整体保留**（2026-10-01 修订：原来的“帧 → shot”两层方案对二值结果会把方差放大约 1.7 倍，现保留为可选项）。两项各给出 95% CI。

---

# 5. 公共状态空间

- 所有 variant 的 ensemble 合并后统一做特征、降维（TICA/VAMP）和聚类，只得到**一套** \(\{S_i\}\)，各 variant 只重新估计 \(P_i\)。
- 手性变体中 D 残基的 φ/ψ 是镜像的，二选一：
  - 特征中去掉被修改的片段，只用其余部分 + 全局量；
  - 或做显式映射 \((\phi,\psi)\to(-\phi,-\psi)\)。

  选定后写入 provenance。
- 状态质量以 VAMP-2 分数和 CK 检验为准；PC1/PC2 只用于画图。

---

# 6. 事件与停止判据（在线）

- 判据形式为阈值 + 持久性 \(\tau_{\rm persist}\)；事件时间记为进入该区域的时刻。
- 判据以 OpenMM reporter 在线运行，采样间隔 \(\Delta t_{\rm obs}\)。离线复算必须与在线结果一致，作为验收项。
- 多通道分类沿用 v1 §9 的 7 个通道（对应 VENUS 的 `NPATHS`）。计算 \(\beta\) 时只用其中锁定的一个反应判据，其余通道作为附加标签。
- Early termination 只允许用于吸收型估计量（4.2、4.3 节）；用于 4.1 节会引入偏差，禁止。

---

# 7. 可复现性与 IC 门禁

venus96.f 实测（2026-09-29，gfortran-14，32 个回归例）暴露了两个问题：

- 简正本征向量没有相位约定：换编译器后符号翻转，同一个种子得到不同的初态。
- NaN 被静默接受：`ESEL<0` 时 `DSQRT` 得到 NaN，而 `NaN.GE.0.001` 为假，于是 NaN 被当成已收敛，整条轨迹全是 NaN，退出码仍为 0。

据此锁定：

1. **IC 级逐位可复现（强制）**：
   - `ic = f(global_seed, frame_id, shot_id, stage)`，与并行度和执行顺序无关；
   - 所有存在任意性的量（本征向量符号、简并基、取向参数化）都要有显式约定。
2. **IC 有效性门禁（强制）**：
   - 检查项：坐标和速度有限、约束满足、无原子重叠、初始能量和温度落在窗口内；
   - 不通过时记录原因并重抽，绝不静默放行；
   - 拒绝率写入 provenance。
3. **轨迹级可复现只做到尽力而为**：
   - CUDA 设 `DeterministicForces=true`，并固定 platform、精度和 GPU 型号，这些都写入记录；
   - 混沌体系不要求逐位一致。实测 venus96 al3：初态吻合到 1e-8，5 万步后动能完全不同，而总能量守恒到 1e-8。

4. **存储与主机（2026-10-01）**：共享盘是 NFS（`nolock,local_lock=all,nocto`，没有跨主机锁，属性缓存很长）。因此每个 run 的 SQLite 记录库只归一台主机所有：库里写有 owner hostname，其他主机打开时直接报错，除非显式接管；日志模式用 rollback journal（`journal_mode=DELETE`、`synchronous=FULL`），不用 WAL。跨主机续算的做法：先停掉旧主机上的写入，在旧主机上用 `Store.backup_to(copy)`（SQLite backup API）做一份一致的拷贝，把拷贝移到新主机，再用 `takeover=True` 打开。**不要手工拷库文件**：写入方在提交途中被杀时会留下 hot journal，只拷库文件得到的是损坏的库。每次写入都在事务内复核 owner，被接管后旧主机上残留的 Store 对象写入会直接报错。

5. **续算一致性与数值失稳（2026-10-01 第二轮复审）**：
   - `physics_config_hash` 一律是后端实际生效配置的 hash（K9）：解析后端含势及其全部参数、积分器、dt、γ、kT、质量；OpenMM 后端含序列化 System 的 sha256、拓扑的 sha256、平台和精度。换力场、水模型或 γ，续算一律拒绝，除非显式 `allow_config_change=True`（只放行代码变化用 `allow_code_change=True`）。终态打标签函数（labeler）和 WE 的进度坐标也进 protocol hash。
   - OpenMM CPU/CUDA 在坐标出现 NaN 时抛异常，后端把它转成 `NumericalInstabilityError`，引擎记为 `"nonfinite"` 记录（K10），与 K4 一致，估计量因此判为无效，而不是让这一发消失在失败日志里。
   - 帧权重由 `shot_weights` 统一换算成每发的权重（K11）：枚举设计（指定 frame_id）每发权重为帧权重除以该帧的发数；按权重抽帧（frame_id = −1）时不再乘帧权重。池化多帧的估计量默认用它。

---

# 8. 前提检验：时间尺度分离（应用层 kill criterion）

\(P_{\rm assoc}(i,j)\) 以初始构象为条件。只有满足

\[
t_{\rm relax}(S_i)\gg t_{\rm enc}
\]

时它才有意义。其中 \(t_{\rm relax}\) 取自 4.1 节的 implied timescales，\(t_{\rm enc}\) 取自 pilot 中 \(b\to\)（反应 | 逃逸）首达时间的中位数。

若所有 metastable state 都有 \(t_{\rm relax}\lesssim t_{\rm enc}\)，则项目转向：

- 改为以 encounter 过程中的构象为条件；
- 或放弃 dimer 分解，只做 monomer 动力学。

---

# 9. 物理设定（一等设计参数）

| 参数 | 要求 |
|---|---|
| 力场 + 水模型 | **已定（用户 2026-09-30）：Aβ42 默认 CHARMM36m + 修正 TIP3P**。力场用 OpenMM 自带的 `charmm36_2024.xml`（由 `par_all36m_prot.prm` 生成，含 C36m D 型氨基酸参数，可直接支持 Phase E）。水模型用 CHARMM TIP3P，但把水 H 原子的 LJ ε 改为 −0.1 kcal/mol，即 C36m 原文（Huang et al. 2017）为 IDP 提出的修正，用于增强蛋白–水色散、防止过度塌缩；Phase B 开工前核对原文的具体参数。a99SB-disp、TIP4P-Ew、TIP4P-D 不用。标准 CHARMM TIP3P 只用于 A2 chignolin（折叠蛋白）。粘度问题见下一行（暂不校正）。验收：\(R_g\)（SAXS）和 NMR 观测量对照文献 |
| 粘度 / 时间尺度 | **暂不做粘度校正**（用户 2026-09-30）。理由：Langevin 积分器自带摩擦，也贡献有效粘度，按水模型粘度比缩放时间尺度并不成立。只报告原始值，同时写明水模型和 γ |
| shot 传播器 | NVE，或低摩擦 Langevin（\(\gamma\le0.1\ \mathrm{ps^{-1}}\)），或 Nosé–Hoover。测动力学时禁止使用 \(\gamma=1\ \mathrm{ps^{-1}}\) |
| HMR / 4 fs | 要么不用，要么在 toy 体系上量化它对动力学的偏差并报告 |
| 溶液条件 | 离子强度与 His 质子化态写入 provenance |
| 速度重采样 | 按约束体系的 MB 分布抽样，包含水分子；去除 COM 动量；温度检查纳入 IC 门禁 |

## 9.5 为什么主干保持经典（不做全局 QCT/ZPE）

**判据**：记 \(x=\hbar\omega/k_BT\)。310 K 时 \(k_BT\approx 216\ \mathrm{cm^{-1}}\)。单个谐振模式的量子自由能修正为

\[
\Delta F=k_BT\left[\ln\!\left(2\sinh\tfrac{x}{2}\right)-\ln x\right]\approx k_BT\,\frac{x^2}{24}\quad(x\ll1).
\]

| 自由度 | 典型 ω (cm⁻¹) | x | ΔF |
|---|---|---|---|
| 扭转、链塌缩、接触重排 | 10–200 | ≤ 0.9 | ≤ 0.04 kT，可忽略 |
| X–H 伸缩 | ~3000 | ~14 | ~4 kT，不可忽略 |

构象势垒穿越的 Wigner 隧穿因子 \(\kappa\approx1+x_\ddagger^2/24\)；对 ~100 cm⁻¹ 的虚频，\(\kappa\approx1.01\)，可忽略。

**三条锁定结论**：

1. **X–H 伸缩用约束冻结**（SHAKE/SETTLE），不做量子修正。高频模式在 310 K 基本处在基态，冻结比“经典柔性振动”更接近量子行为；它也是第 9 节约束设置的物理依据。
2. **不能在经验力场上再叠加显式 NQE**。a99SB-disp、CHARMM36m、TIP4P-D 这类力场都是对着室温实验数据拟合的，平均意义上的核量子效应已经吸收在参数里。再叠加 PIMD 等方法会重复计入。如果将来确实要显式 NQE，必须换成专为 PIMD 参数化的模型（如 q-TIP4P/F 这一类）。
3. **手性变体（Phase E）不需要任何量子修正**。D/L 残基是不同的立体异构体，差别来自构象和位阻，属于经典效应，用镜像参数的经典力场就能完整描述。宇称破缺能量约 10⁻¹⁷ kT，可忽略。v1 §11.2 中“与 chirality 相关而需要控制局部振动态”一条**删除**。

**真正需要量子处理的情形**，只在 Phase F 做，属于化学问题而不是构象问题：

- Cu/Zn 与 His6/13/14 的配位和氧化还原；
- Met35 氧化；
- 质子转移反应。

生理 pH 下的质子化态**不属于**这一类，用固定质子化态或恒 pH MD 处理即可。

---

# 10. 算力：成本模型与多保真度

**成本模型**：

\[
C=\sum_{(i,j)} M_{ij}\,\mathbb E[t_{\rm stop}(i,j)]\big/\text{throughput}(N_{\rm atoms})
\]

**量级估计（需用 pilot 实测替换）**：

- 体系规模：monomer 约 3–5×10⁴ 原子；dimer 盒边约 12–14 nm，约 2–3×10⁵ 原子。
- 算力：以 \(K=10\) 为例，55 对 × \(M=50\) × 100 ns = 275 μs，单卡需要数千 GPU·天。均匀射击不可行。

**被忽略的一项：输入系综本身的成本（Phase B0）**。这是 shooting 之前的前置成本。每个 variant 都要一套收敛的显式溶剂 REMD/REST2 系综，数量级是“几十个副本 × 每个副本 μs 级”；WT、A2T、A2V 加上手性变体共约 5–6 套。这很可能与 dimer 射击同一量级，必须在预算中单列。最省的办法是拿到已发表工作的系综数据，再按第 9 节验收力场的一致性。

**结论（2026-09-30 修订）**：数千 GPU·天这个数字不是一个需要优化的问题，它说明**均匀 dimer 协议本身就不该进入正式方案**。dimer 部分改为 4.3 节的分阶段方案 + 4.5 节的 WE，成本从

\[
C_{\rm brute}\sim N_X^2N_\Omega N_v\langle t\rangle
\quad\text{变为}\quad
C_{\rm WE}\sim \sum_{\text{stage}}N_{\rm walker}N_{\rm iter}\tau_{\rm seg}\times N_{\rm rep},
\]

不再随预先离散出来的 \(X_i\times X_j\times\Omega\) 笛卡尔积爆炸。参照量级：Saglam & Chong（Chem. Sci. 2019）在显式溶剂中用 WE 模拟 barnase–barstar，累计 **18 μs**，得到 203 条独立结合路径，\(k_{\rm on}\) 与实验在误差内一致。Aβ42 没有唯一的天然结合构型，还有标签约束（4.5 节），预期会比这个更难；但量级应该是几十到几百 μs，而不是几千 GPU·天的笛卡尔积。

**应对措施**：

1. **Pilot 门**：不能只测平均停止时间 \(\langle t_{\rm stop}\rangle\)。要在目标 GPU 上实测 ns/day，并对 3 个标签类各做一次短 WE，报告以下量：
   - \(f_{\rm escape}\)：在 \(\tau_{\rm short}\) 内逃逸的比例；
   - \(f_{\rm metastable}\)：进入 \(E_{\rm long}\) 的比例，以及它们的停留时间分布（有没有 μs 级的“黏糊”复合物）；
   - \(\tau_{\rm decor}\)：clone 之后两条子轨迹去相关所需的时间，它决定 \(\tau_{\rm seg}\) 的下限；
   - \(N_{\rm eff}\) 随迭代的走势，以及 \(I(\text{label};\text{outcome})\) 的初步估计。

   由这些量外推出 \(C_{\rm WE}\)，经用户确认后才进入生产。
2. **多保真度**：Tier 0 只负责**高召回**，不要求定量准确。
   - Tier 0 由 GB 隐式溶剂、CG、极短的显式溶剂模拟和几何取向搜索组成，任务只是产生候选 \(\{X_i,X_j,\Omega\}\) 和有用的进展坐标；
   - **所有概率和动力学量都由 Tier 1 的显式溶剂 WE 重新计算**；
   - 需要验证的只有假阴性率：从 Tier 0 拒掉的样本中随机抽 ≥ 60 个放到 Tier 1 跑，若 0 个是 productive，按 rule of three 可以得出 FN < 5%（95% 置信）。估计 \(P_\beta\) 时把被拒样本的 \(p_\beta\) 记为 0，它引入的偏差上界 = FN 率 × 被拒样本的权重质量，需要报告。
3. **自适应分配**：在 WE 的不同标签类之间，按 \(p_\beta\) 的后验方差分配 walker 预算，目标是最小化 4.4 节两项的 CI 宽度。

---

# 11. 接口规格（只列签名）

```python
class EnsembleFrame(Protocol):
    coordinates: ArrayNx3; box: Array3x3 | None
    topology_ref: str          # 拓扑内容哈希
    temperature: float; weight: float
    source_id: str; frame_id: int; time: float

class InitialConditionSampler(Protocol):
    def sample(self, key: ShotKey) -> InitialState: ...        # 纯函数：同一个 key 必须给出逐位相同的结果
    def validate(self, s: InitialState) -> ValidityReport: ...

class StopRule(Protocol):
    kind: Literal["fixed_lag", "absorbing_AB", "b_surface"]
    def update(self, obs: Observables, t: float) -> StopDecision | None: ...

class PotentialBackend(Protocol):
    kind: Literal["analytic", "openmm", "openmm+ml", "openmm+qm"]
    gpu_resident: bool                      # False 时引擎拒绝对超过 cfg.max_atoms_host 的体系开生产
    def build(self, s: InitialState, cfg: PhysicsConfig) -> Propagator: ...
    def energy_forces(self, x: ArrayNx3) -> tuple[float, ArrayNx3]: ...   # 一致性测试与 IC 门禁用
    def provenance(self) -> dict: ...

def pes_consistency_suite(backend: PotentialBackend, probes) -> PESReport: ...   # 第 3.5 节，所有后端必须通过

class Resampler(Protocol):                   # 与 InitialConditionSampler、PotentialBackend 同级
    kind: Literal["none", "uniform", "adaptive", "we", "revo"]   # none = 经典 VENUS：一条轨迹跑到底
    def resample(self, walkers: list[Walker], it: int, key: IterKey) -> list[Walker]: ...
    # 约束：Σ w 精确守恒（误差 ≤ 1e-12）；merge 只在同一 origin_label 内进行（可以按 4.5 节的诊断结果显式放开）
    # clone/merge 的随机性由 IterKey = (global_seed, run_id, iteration) 派生，与并行度无关

@dataclass
class Walker:
    segment_key: SegmentKey                  # (run_id, iteration, walker_id)
    parent: SegmentKey | None
    origin_label: tuple[int, int]            # (i, j) 状态标签
    weight: float
    z: Array                                 # 进展坐标
    checkpoint_ref: str

def run_shot(key: ShotKey, sampler, backend, stop: StopRule, store) -> ShotRecord: ...
def run_segment(walker: Walker, backend, tau_seg: float, stop: StopRule, store) -> Walker: ...   # WE 的基本单元
def run_we(init: list[Walker], backend, resampler: Resampler, stop: StopRule, n_iter: int, store) -> WERun: ...
def solve_staged(stage_runs: dict[str, WERun], network: StageNetwork) -> PBetaEstimate: ...      # 4.3 节的拼接
def estimate_T(records, states, lag) -> TEstimate                  # 含 CI、CK 检验
def estimate_committor(records, A, B) -> CommittorEstimate         # 含超时比例
def estimate_kon(records, b, q, D_AB) -> KonEstimate               # 4.3 节
def decompose(wt, var, P_wt, P_var) -> Decomposition               # 4.4 节
```

`config.yaml` 顶层键：`mode`、`system`（拓扑、力场、水模型、离子）、`physics`（积分器、步长、恒温器、约束、platform）、`ic`（sampler 及其参数）、`stop`、`observables`、`budget`（shot 数或自适应目标）、`seed`。

`ShotRecord` 字段：v1 §14 的全部字段，另加 `shot_key`、`ic_validity`（含重抽次数）、`stop_rule.kind`、`stop_reason`、`event_time`、`physics_config_hash`、`backend.provenance`、`code_version` 和抽稀后的 observable 时间序列。

---

# 12. 路线图与验收门

**Phase A：VENUS 现代化本体（与蛋白种类无关），这是第一个交付物。**

| 阶段 | 内容 | 验收 |
|---|---|---|
| A0 引擎 + 解析 toy + PES 层 | 引擎、四种 mode、三个估计量、IC 门禁、provenance；`analytic`、`openmm` 两个后端 | ⓪ 两个后端都通过 `pes_consistency_suite`；⓪′ WE 验收：在一维或二维双阱 Langevin 中，WE 给出的速率与暴力长轨迹的参考速率在 CI 内一致；每次迭代的 Σw 守恒到 1e-12；带标签约束的 WE 能精确复现各标签的条件概率；⓪″ 吸收网络验收：在二维双阱上铺 milestone，\(\mathcal B=(I-Q)^{-1}R\) 给出的 committor 与数值精确解相比 RMSE < 0.03；再构造一个人为带记忆的 toy 体系，标签合并的网络必须失败，而增广 \((m,\ell)\) 的网络必须通过，证明 Markov 性检验能识别问题；① 自由扩散：从 \(b\) 出发，击中半径 \(a\) 的概率 \(=a/b\)，3σ 内；② 一维/二维双阱 Langevin 的 committor 与数值精确解相比 RMSE < 0.03；③ 同一个 ShotKey 重跑得到逐位相同的 IC；④ 注入的非法 IC 被 100% 拦截；⑤ 在线与离线事件判定一致 |
| A1 丙氨酸二肽（显式水） | 在 OpenMM 真实体系上跑通 `shoot.ensemble`、`shoot.surface` | implied timescales 与 ≥1 μs 长轨迹 MSM 的参考值在 CI 内一致；CK 检验通过 |
| A2 小蛋白（如 chignolin） | 蛋白体系上的 \(T(\tau)\) 与 committor | 折叠/去折叠时间与本项目自跑的长轨迹参考一致，与文献同数量级 |
| A3 `shoot.encounter` 功能测试 | 用隐式溶剂下的一对小蛋白走通 b-surface 流程 | 在 \(\Omega\) 取不同 \(q\) 时，\(\beta_\infty\) 在 CI 内保持不变（检验 4.3 节公式的自洽性）；与实验 \(k_{\rm on}\) 的定量对比**不作为**这一阶段的门槛 |

**Phase B 及以后：Aβ42 应用**

| 阶段 | 内容 | 验收 |
|---|---|---|
| B0 输入系综 | **生产或获取 Aβ42 平衡系综**。系综是 shooting 的输入，本计划不负责生成方法，但必须有人把它产出来：自跑 REMD/REST2，或向文献作者（如 Zhu & Yu）索取数据 | 各 variant 的系综在目标温度下收敛（分块统计量稳定，副本交换充分），带权重，并满足第 11 节的 `EnsembleFrame` 规格。**成本单独立项**，见第 10 节 |
| B 单体 | 力场验收 → 公共状态空间 → \(T(\tau)\) | \(R_g\)、NMR 观测量在文献误差内；CK 检验通过；给出各态的 \(t_{\rm relax}\) |
| C Dimer | 锁定 \(\Sigma\) 选项 → Tier 0 高召回筛选 + FN 抽检 → 分阶段 WE pilot（第 10 节的四个量）→ 第 8 节 / 4.5 节的 kill 诊断 → 生产 | pilot 报告经用户确认；FN < 5%（rule of three）；\(I(\text{label};\text{outcome})\) 显著大于 0；≥ 5 次独立 WE 重复得到的 \(p_\beta(i,j)\) 带 CI；BD 给出的 \(k_{\rm enc}\) 在各 variant 间差异 < 10%；增广网络通过 Markov 性检验；BD/MD 重叠 milestone 的通量与击中分布一致 |
| D WT / A2T / A2V | 4.4 节分解 | 两项各自给出 95% CI；Tier 0 与 Tier 1 交叉验证 |
| E 手性变体 | 开工前锁定 D 残基的特征映射 | 同 D |

**Phase F（可选，在 D 之后）：局部反应 PES**

| 阶段 | 内容 | 验收 |
|---|---|---|
| F | `openmm+ml`（ML/MM），以及 `openmm+qm`（openmm-orca ≥ v0.4 的周期性 QM/MM，或 MTS）；按 3.6 节的混合策略执行 | 通过 PES 一致性测试；NVE 下 MTS 的能量漂移不超过单时间步参考的 2 倍；ML/MM 边界处的力连续；`pes_uncertain` 比例 ≤ 5%；同一模型版本加同一个 ShotKey，重跑结果一致（在第 7 节的容差内） |

**第一版不做**：v1 §20 的全部条目；小分子气相 QCT/RRHO/ZPE（由 VENUSpy 覆盖）；`NMODE`/`MPATH` 的迁移。

---

# 13. 新意定位

- **“现代化 VENUS”已有 VENUSpy（JCTC 2025）**，而且它做的是气相 + ML/QM PES（第 2.5 节）。本项目唯一可以主张的软件新意是“凝聚相 / 蛋白 + GPU 常驻 + 动力学估计量”，绝不能写成“第一个 Python 版 VENUS”。
- **方法层面已有成熟工具**：固定 lag 射击 + MSM（deeptime/PyEMMA）、aimless shooting/TPS（OpenPathSampling）、WE（WESTPA）、BD 结合速率（Browndye/SDA）。“VENUS 风格射击”本身不构成贡献。
- **真正的贡献**：
  1. 构象条件化的 \(k_{\rm on}(i,j)\)，附带时间尺度分离检验；
  2. 在公共状态空间上，对变体效应做精确的 population / dynamical 分解并给出不确定度。
- **VENUS 的定位**：适合写在引言里，说明为什么把碰撞参数语言（\(b,\Omega,\sigma\leftrightarrow k_{\rm on}\)）带进 IDP 研究是自然的。

---

# 14. 待用户决定

1. ~~力场 + 水模型~~ → 已定：Aβ42 用 CHARMM36m + 修正 TIP3P（H 原子 ε = −0.1 kcal/mol），第 9 节。
2. 算力预算上限（在 Phase C pilot 之后确认）。
3. A2 用哪个小蛋白（默认 chignolin）。
4. 是否联系 VENUSpy 作者（R. Sun 组）：询问许可证，或讨论合作 / 联合谱系。未解决之前，第 2.5 节的“不碰其代码”规则一直有效。

---

## 参考

- Q. Zhu, H. Yu, *N-terminal Chirality and Sequence Variations Modulate the Conformational Landscape of Amyloid-beta 42*，bioRxiv 10.64898/2026.03.19.713039。v1 中所列 JCIM 卷期、页码和 DOI 尚未核实。
- Northrup, Allison, McCammon, *J. Chem. Phys.* 1984：BD b-surface 结合速率方法。
- K. Fujioka, R. Richard, J. Waldrop, Y. Luo, T. L. Windus, R. Sun, *VENUSpy: A Chemical Dynamics Simulation Program in the Era of Machine Learning*, J. Chem. Theory Comput. 2025, 21, 10679（doi:10.1021/acs.jctc.5c01408；预印本 chemrxiv 10.26434/chemrxiv.15000804）；代码 github.com/kaka-zuumi/VENUSpy，无 LICENSE。
- A. S. Saglam, L. T. Chong, *Protein–protein binding pathways and calculations of rate constants using fully-continuous, explicit-solvent simulations*, Chem. Sci. 2019（barnase–barstar WE，累计 18 μs，203 条路径）。
- WESTPA 2.0, *J. Chem. Theory Comput.* 2022（doi:10.1021/acs.jctc.1c01154）；MAB 分箱；RED 速率估计方案。
- REVO（Dickson 组，wepy）；SEEKR2（MD/BD milestoning）；Plattner et al., *Nat. Chem.* 2017（barnase–barstar 的完整缔合动力学，MSM）。
- ChemPotPy：解析 PES 的 Python 库，*J. Phys. Chem. A*（doi:10.1021/acs.jpca.3c05899），VENUSpy 的解析 PES 来源。
- venus96.f 实测审阅（2026-09-29）：模块结构、71 个 COMMON 块、32 例回归结果、h2oleps NaN 的根因。
