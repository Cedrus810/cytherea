# VENUS96 → Aβ42 / IDP Ensemble-Trajectory Engine
## 面向蛋白与无序蛋白的重构设计草案

**目标**：在保留 VENUS96 “initial ensemble → many independent trajectories → outcome classification → statistics” 核心哲学的前提下，将其从以小分子 RRHO/QCT 为中心的程序，重构为可同时支持：

1. 传统小分子气相 QCT；
2. 蛋白 / IDP 构象动力学；
3. Aβ42 单体构象系综；
4. Aβ42–Aβ42 encounter / dimerization shooting；
5. 后续 reactive MLP / QM/MM 局部反应动力学。

这里不尝试把传统 VENUS96 的 RRHO/QCT **直接放大到整个蛋白**。相反，RRHO 被降级为一个可插拔的局部 initial-condition sampler；对于 IDP，主分布来自 MD/REMD 等显式构象采样。

---

# 1. 核心原则

VENUS96 真正值得继承的不是某一种积分器或某一种势函数，而是：

\[
\boxed{
\text{Initial Ensemble}
\rightarrow
\text{Independent Trajectories}
\rightarrow
\text{Outcome Classification}
\rightarrow
\text{Statistics}
}
\]

传统小分子 QCT 中：

\[
(v,j,E_{\rm coll},b,\Omega)
\rightarrow
\Gamma_0
\rightarrow
\Gamma(t)
\rightarrow
\text{product state}
\]

Aβ42 / IDP 中改写为：

\[
(X_{\rm conf},X_{\rm solv},p,\Omega,r,\ldots)
\rightarrow
\Gamma_0
\rightarrow
\Gamma(t)
\rightarrow
\text{conformational / association outcome}
\]

其中 \(X_{\rm conf}\) 不再是围绕单一最低点的 RRHO 扰动，而是从真实的 IDP 构象系综中抽样。

---

# 2. 为什么 Aβ42 不能使用全局 RRHO

对于小分子，RRHO 可以近似：

\[
Q \approx
Q_{\rm trans} Q_{\rm rot}
\prod_i Q_{{\rm vib},i}
\]

并假设：

\[
V(q_i)\approx \frac12\omega_i^2 q_i^2
\]

但 Aβ42 是 intrinsically disordered peptide，主要自由度包含：

- 大量 backbone \(\phi/\psi\) 转动；
- side-chain rotamer；
- transient helix / β / turn；
- chain collapse / expansion；
- transient long-range contacts；
- hydration-shell rearrangement；
- 多个彼此可互换的 metastable basins。

因此不存在一个具有代表性的全局参考结构 \(X_0\)，使得：

\[
X \approx X_0+\delta X
\]

在整个热力学相关区域内成立。

对 Aβ42，应使用：

\[
\boxed{
X_{\rm peptide},X_{\rm water}
\sim
P_{\rm ensemble}(X)
}
\]

而不是：

\[
X \sim P_{\rm RRHO}(X|X_0)
\]

2026 年 Zhu 与 Yu 对 Aβ42 WT、A2T、A2V 以及 N 端 chirality variants 的工作本身就是这一逻辑：利用 temperature-REMD 描述 rugged conformational landscape，并研究 mutation/chirality 如何重排 ensemble population，而不是把 Aβ42 看成围绕一个固定结构的小振动体系。

参考：

- Q. Zhu, H. Yu, *N-Terminal Chirality and Sequence Variations Modulate the Conformational Landscape of Amyloid-Beta 42*, J. Chem. Inf. Model. **2026**, 66, 10156–10171. DOI: 10.1021/acs.jcim.6c01001

---

# 3. 不删除 RRHO：改成 Hierarchical Initial-Condition Model

建议把 VENUS96 原来的 initial-condition 逻辑抽象成：

```text
InitialConditionSampler
├── LegacyRRHOQCTSampler
├── MaxwellBoltzmannSampler
├── MDEnsembleSampler
├── REMDEnsembleSampler
├── LocalRRHOSampler
├── HinderedRotorSampler
├── WignerSampler
└── CompositeSampler
```

## 3.1 Legacy 模式

对小分子保持原始 VENUS96 逻辑：

\[
E_i = \left(v_i+\frac12\right)\hbar\omega_i
\]

支持：

- normal-mode sampling；
- rotational state；
- collision energy；
- impact parameter；
- random molecular orientation；
- classical / quasiclassical trajectory。

必须保证重构后可回归原 VENUS96 benchmark。

---

## 3.2 Protein / IDP 默认模式

主构象来自：

\[
X_k\sim P_{\rm MD/REMD}(X)
\]

速度来自：

\[
p\sim P_{\rm MB}(p|T)
\]

因此一个初态为：

\[
\Gamma_0=(X_k,p_j)
\]

同一个构象可以重复 velocity shooting：

\[
(X_k,p_1),\ldots,(X_k,p_M)
\]

由此估计条件转移概率：

\[
P(B|X_k)
\simeq
\frac{N_B(X_k)}{M}
\]

这已经自然接近 committor：

\[
p_B(X)=P(\text{reach B before A}|X)
\]

---

# 4. RRHO 在蛋白模式中的正确位置

RRHO 不应全局使用，但仍可作为 **局部快速自由度模型**。

将体系划分：

\[
q =
(q_{\rm slow}, q_{\rm local}, q_{\rm bath})
\]

其中：

### Slow conformational DOFs

由 MD/REMD 直接采样：

- backbone torsions；
- large-amplitude collective motion；
- chain collapse；
- transient β-hairpin；
- long-range contacts。

### Local stiff DOFs

可选择使用 local RRHO / Wigner：

- 某些键伸缩；
- 局部刚性基团；
- 特定研究问题中的高频局域振动。

### Bath

溶剂和其余经典自由度由实际 snapshot + thermal velocity 描述。

于是：

\[
P_{\rm init}
=
P_{\rm ensemble}(X_{\rm slow},X_{\rm bath})
P_{\rm local}(q_{\rm local},p_{\rm local}|X)
P_{\rm MB}(p_{\rm bath})
\]

这是 Protein-compatible RRHO 的核心定义。

---

# 5. Aβ42 第一阶段：单体 Ensemble → Shooting

## 5.1 目的

不是再次证明 Aβ42 有很多构象，而是获得：

\[
\boxed{
P(X_i\rightarrow X_j|\tau)
}
\]

以及：

\[
\boxed{
P(\text{aggregation-prone outcome}|X_i)
}
\]

最终寻找：

- 哪些 monomer basins 是 kinetic hubs；
- 哪些状态容易形成 β-rich contacts；
- 哪些状态只是热力学上常见，却不是 oligomerization-competent；
- A2T / A2V / chirality modification 改的是 basin population，还是每个 basin 的动力学命运。

---

## 5.2 构象池

第一版建议直接接受外部 trajectory：

```text
trajectory.xtc / dcd / nc
topology
temperature
weights (optional)
```

来源可以是：

- conventional MD；
- REMD；
- REST2；
- MetaD；
- AWH；
- Weighted Ensemble；
- diffusion / generative structures + MD relaxation。

核心代码不应该绑定某一种 enhanced sampling 方法。

---

## 5.3 Ensemble reweighting

若输入来自 canonical MD：

\[
w_k = 1/N
\]

若来自 REMD，取目标温度对应样本或正确重权。

若来自带 bias 的模拟：

\[
w_k
\propto
e^{+\beta V_{\rm bias}(X_k)}
\]

具体重权算法应由独立模块处理，而不是写死在 trajectory engine。

接口：

```text
EnsembleFrame {
    coordinates
    box
    topology_state
    temperature
    statistical_weight
    source_id
    time
}
```

---

# 6. 不要直接 PCA 二维分盆：State Representation

Aβ42 高维、无序，PC1/PC2 可以用于画图，但不建议作为唯一 state definition。

第一版 feature set 建议：

### Global

- \(R_g\)
- end-to-end distance
- SASA
- total intramolecular contacts
- secondary-structure fractions

### Residue-resolved

- backbone \(\phi,\psi\)
- residue-residue contact map
- H-bond map
- solvent exposure
- CHC contacts

特别关注：

- N-terminal region；
- central hydrophobic core (CHC)；
- C-terminal hydrophobic region；
- D23–K28 / E22–K28 等可能的关键接触。

状态识别可依次尝试：

```text
features
   ↓
TICA / VAMP / learned latent representation
   ↓
clustering
   ↓
MSM / metastable states
```

原则：

> state 的定义应首先满足 kinetic usefulness，而不是二维图看起来漂亮。

---

# 7. VENUS-style Shooting Layer

这是整个项目最核心的新层。

对每一个代表状态 \(S_i\)：

1. 按统计权重选择若干 frames；
2. 对每个 frame 重采 \(M\) 组 velocities；
3. 运行固定长度 \(\tau\) 的 unbiased trajectories；
4. 自动分类终态；
5. 记录 transition counts。

例如：

```text
State S17
├── frame 001
│   ├── shot 001 → S17
│   ├── shot 002 → S03
│   └── shot 003 → S21
├── frame 002
│   ├── shot 001 → S17
│   ├── shot 002 → S21
│   └── shot 003 → S21
└── ...
```

得到：

\[
T_{ij}(\tau)
=
P(S_j,t+\tau|S_i,t)
\]

这一步与传统 VENUS：

\[
N_{\rm product}/N_{\rm total}
\]

在统计哲学上完全一致。

---

# 8. 第二阶段：Aβ42–Aβ42 Encounter Ensemble

这是比单体 FEL 更有价值的阶段。

构造两个 monomer：

\[
X_i^{(A)},X_j^{(B)}
\]

再采样：

- relative orientation \(\Omega\)；
- initial separation \(r\)；
- relative translational velocity；
- solvent environment。

初态：

\[
\Gamma_0=
(X_i^{A},X_j^{B},\Omega,r,p)
\]

这就是蛋白版的：

\[
(v,j,E_{\rm coll},b,\Omega)
\]

---

# 9. Aβ42 dimer trajectory outcomes

不要一开始只定义 “bound / unbound”。

建议至少使用多通道 classification：

```text
UNBOUND
ENCOUNTER
COMPACT_NON_BETA
BETA_CONTACT
BETA_HAIRPIN_ASSOCIATED
LONG_LIVED_DIMER
OTHER
```

后续可以通过数据重新合并状态。

自动 event detector 需要支持：

### Geometric observables

- intermolecular minimum distance；
- COM distance；
- number of heavy-atom contacts；
- residue-residue contact map；
- intermolecular H-bonds；
- β-strand registry；
- buried SASA。

### Temporal persistence

一个接触只有持续超过：

\[
\tau_{\rm persist}
\]

才定义为稳定事件，避免把瞬时碰撞误判成结合。

---

# 10. 我们真正想计算的量

传统 VENUS：

\[
\sigma(E)
\]

Aβ42 对应的核心量变成：

## 10.1 Monomer-conditioned association probability

\[
P_{\rm assoc}(X_i,X_j)
\]

回答：

> 哪些 monomer conformations 真正具备 association competence？

---

## 10.2 Conformational committor

\[
p_{\rm assoc}(X_i)
=
\sum_j
P(X_j)
P_{\rm assoc}(X_i,X_j)
\]

这比单纯说：

> “这个 basin 比较 compact”

有意义得多。

---

## 10.3 Mutation decomposition

对于 WT、A2T、A2V：

\[
P_{\rm assoc}
=
\sum_{ij}
P_iP_jP_{\rm assoc}(i,j)
\]

mutation 可以通过两种机制改变最终行为：

### Population effect

\[
P_i^{\rm mutant}\neq P_i^{\rm WT}
\]

即 mutation 改变 monomer ensemble。

### Dynamical effect

\[
P_{\rm assoc}^{\rm mutant}(i,j)
\neq
P_{\rm assoc}^{\rm WT}(i,j)
\]

即即使处于相似构象，mutation 仍改变 encounter fate。

这个 decomposition 很重要，因为它可以把：

> “mutation 改了 ensemble”

和：

> “mutation 改了同一类状态的动力学”

分开。

---

# 11. RRHO / ZPE 在 Aβ42 项目中的处理

## 11.1 默认情况

如果研究：

- monomer conformational transitions；
- dimer encounter；
- early aggregation；
- β-contact formation；

则主体自由度是低频热运动。

因此：

\[
\boxed{\text{默认不启用 global QCT / global ZPE}}
\]

而使用经典 MD ensemble。

---

## 11.2 Local RRHO

只有明确需要时，对选定局部子空间使用：

\[
H_{\rm local}
=
P^THP
\]

例如：

- 特定局部高频 vibration；
- isotope effect；
- 特定 chemical reaction；
- 与 chirality 相关而需要控制局部振动态的模型实验。

---

## 11.3 ZPE leakage

若未来引入 reactive QCT，不要用 whole-protein normal-mode ZPE protection。

设计：

```text
ZPEProtection
├── None
├── ProductFilter
├── GaussianBinning
├── LocalPairProtection
└── AdaptiveLocalProtection
```

`AdaptiveLocalProtection` 只保护稳定的高频 spectator bonds，进入反应区时平滑关闭保护。

Aβ42 aggregation-only 第一版不需要这部分进入核心路径。

---

# 12. 力场与动力学 backend

Trajectory engine 与力场彻底解耦。

接口：

```text
PotentialBackend
├── OpenMMForceFieldBackend
├── GromacsExternalBackend
├── AmberExternalBackend
├── MLPotentialBackend
├── QMBackend
└── QMMMBackend
```

对于第一版 Aβ42：

\[
\boxed{\text{OpenMM backend 最适合做 shooting runtime}}
\]

原因：

- Python/C++ API 方便；
- 可以直接批量启动 independent replicas；
- checkpoint / restart 简单；
- 容易接自定义 collective variables；
- 后续容易接 ML potential。

不要求第一阶段就把整个传播器重写成 JAX。

---

# 13. VENUS96 重构的软件边界

建议不要直接把蛋白功能硬塞进原来的 input deck。

重构成三层：

```text
venus-core/
    trajectory propagation
    random number management
    trajectory lifecycle
    checkpoint
    statistics

venus-ic/
    legacy rrho
    qct
    md ensemble
    local rrho
    hindered rotor
    velocity resampling
    encounter sampling

venus-analysis/
    event detection
    state classification
    binning
    committor
    transition matrix
```

然后：

```text
venus-backends/
    legacy analytic PES
    OpenMM
    ML potential
    QM/MM
```

Aβ42 是一个应用层：

```text
examples/
    abeta42_monomer/
    abeta42_dimer/
```

而不是把 Aβ42-specific code 写进核心。

---

# 14. 推荐的数据结构

每一条 trajectory 都应该是一个独立实验记录：

```yaml
trajectory_id: ...
parent_ensemble_frame: ...
system_variant: WT
state_initial: S17
temperature: 310
random_seed: ...
velocity_seed: ...
backend: openmm
potential: ...
start_time: ...
stop_reason: ...
final_state: ...
trajectory_weight: ...
```

额外存：

```yaml
observables:
  rg: ...
  contacts_total: ...
  beta_fraction: ...
  inter_contacts: ...
  buried_sasa: ...
```

这样后续可以完整重分析，而不用重新跑 MD。

---

# 15. 随机数与可复现性

VENUS-style massive shooting 非常依赖随机采样，因此必须把 RNG 作为一等公民。

推荐：

\[
seed =
H(
global\_seed,
frame\_id,
shot\_id,
stage
)
\]

避免并行 job 数量变化导致随机序列变化。

同一个：

```text
frame_id + shot_id + global_seed
```

必须始终产生相同初态。

---

# 16. HPC 执行模型

Aβ42 shooting 天然 embarrassingly parallel。

最简单结构：

```text
coordinator
   │
   ├── trajectory batch 0001
   ├── trajectory batch 0002
   ├── trajectory batch 0003
   └── ...
```

每批：

- 1 GPU；
- 多个 independent replicas；
- 不需要 replica-to-replica 通信。

与 REMD 不同，production shooting 可以完全避免频繁交换。

后续可优化：

- CUDA graphs；
- multi-context GPU packing；
- batched ML potential evaluation；
- trajectory early termination；
- adaptive allocation。

---

# 17. Early termination

这是把 VENUS 思想应用到大体系时非常重要的优化。

若 trajectory 已进入明确 basin：

\[
S_j
\]

并持续：

\[
t > \tau_{\rm commit}
\]

则提前结束。

例如 dimerization：

```text
if COM_distance > r_escape for 5 ns:
    outcome = UNBOUND
    stop()

if beta_contacts >= N and persistence >= tau:
    outcome = BETA_ASSOCIATED
    stop()
```

这样没必要所有 shot 都跑同样长度。

---

# 18. Adaptive shooting

第一版 uniform shooting：

\[
M_i=M
\]

第二版应按统计不确定度动态分配 trajectories。

若：

\[
\hat p_i =
\frac{N_i}{M_i}
\]

binomial uncertainty：

\[
\sigma_i
\simeq
\sqrt{\frac{p_i(1-p_i)}{M_i}}
\]

优先给：

- \(p\sim0.5\)；
- transition region；
- rare but important basin；
- statistical uncertainty 高的 state；

增加 shots。

这就是把传统 VENUS 固定轨迹数升级成现代 active trajectory allocation。

---

# 19. 第一版 MVP

## Phase 0 — VENUS96 preservation

必须先保证：

- 原始 test cases 可跑；
- RRHO/QCT results 不因重构改变；
- legacy input 可以转换成新 internal representation。

**这一阶段只做软件解耦，不改物理。**

---

## Phase 1 — Generic Ensemble Input

实现：

```text
trajectory/topology
        ↓
EnsembleFrame pool
        ↓
frame selection
        ↓
velocity resampling
        ↓
independent short MD
```

验收：

- 任意普通蛋白 trajectory 都可作为初态池；
- 同一 frame 可 reproducibly shoot 100+ trajectories；
- trajectory metadata 完整。

---

## Phase 2 — Aβ42 Monomer

输入 WT Aβ42 ensemble。

实现：

- feature extraction；
- clustering / state assignment；
- repeated shooting；
- \(T_{ij}(\tau)\)；
- basic committor-like statistics。

第一版无需 mutation。

---

## Phase 3 — Aβ42 Dimer Encounter

实现：

- two-monomer conformer selection；
- random relative orientation；
- initial separation；
- solvent preparation；
- encounter shooting；
- multi-outcome classification。

得到：

\[
P_{\rm assoc}(i,j)
\]

---

## Phase 4 — WT / A2T / A2V

重复同一协议，比较：

\[
P_i
\]

和：

\[
P_{\rm assoc}(i,j)
\]

把 mutation effect 分解为：

\[
\text{ensemble redistribution}
+
\text{conditional dynamical change}
\]

---

## Phase 5 — Chirality variants

加入：

- WT\(_{1-6D}\)
- A2V\(_{1-6D}\)
- A2T Cβ stereochemical variants

与 2026 JCIM 工作直接形成可比较层：

原工作主要回答：

\[
\text{variant}
\rightarrow
\text{monomer ensemble}
\]

新框架回答：

\[
\text{variant}
\rightarrow
\text{monomer ensemble}
\rightarrow
\text{encounter outcome / kinetic competence}
\]

---

# 20. 不建议第一版做的东西

为了避免项目爆炸，第一版明确不做：

- whole-protein Hessian；
- whole-protein RRHO；
- every-mode QCT；
- 全局 ZPE enforcement；
- proton tunneling；
- reactive MLP；
- fibril-scale 10+mer；
- end-to-end differentiable trajectory engine；
- 一上来训练神经网络 reaction coordinate。

这些都可以以后加。

第一版只验证一个核心问题：

\[
\boxed{
\text{VENUS-style ensemble shooting 是否能从 Aβ42 monomer ensemble 中识别 kinetically distinct states？}
}
\]

然后再进入 dimerization。

---

# 21. 最关键的科学问题

这个项目真正值得回答的不是：

> Aβ42 有哪些构象？

这个问题已经有大量文献。

而是：

\[
\boxed{
\text{哪些构象真正决定后续动力学命运？}
}
\]

具体来说：

1. 高 population basin 是否一定具有高 association probability？
2. rare conformers 是否反而是 aggregation-competent states？
3. A2T 的 protective effect 是减少这些 states 的 population，还是降低这些 states 的 association competence？
4. A2V 是否相反？
5. N-terminal chirality perturbation 如何通过 ensemble redistribution 改变 CHC-mediated association？
6. 对同一类 monomer conformer，variant identity 是否仍显著改变 encounter outcome？

---

# 22. 总体架构

```text
                   ┌───────────────────────┐
                   │ Legacy VENUS96 QCT    │
                   │ RRHO / collision IC   │
                   └───────────┬───────────┘
                               │
                               │
                     InitialCondition API
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
 Legacy small molecules                         Protein / IDP
                                                      │
                                  ┌───────────────────┴──────────────────┐
                                  │                                      │
                           MD/REMD ensemble                        Local RRHO
                                  │                                (optional)
                                  └──────────────────┬───────────────────┘
                                                     │
                                              initial states
                                                     │
                                                     ▼
                                         Independent trajectories
                                                     │
                    ┌────────────────────────────────┼──────────────────────┐
                    │                                │                      │
                    ▼                                ▼                      ▼
               monomer state                 encounter state          reaction event
                    │                                │                      │
                    └────────────────────────────────┼──────────────────────┘
                                                     │
                                                     ▼
                                             Outcome classifier
                                                     │
                                                     ▼
                                    transition / committor statistics
```

---

# 23. 一句话的软件设计结论

**不要把 protein 强行兼容到 RRHO。**

应该让 VENUS 重构为：

\[
\boxed{
\text{Trajectory engine}
+
\text{pluggable initial-condition distributions}
+
\text{pluggable outcome classifiers}
}
\]

其中：

- 小分子：`RRHO/QCT sampler`
- folded protein：`MD ensemble + optional local RRHO`
- IDP/Aβ42：`enhanced-sampling ensemble`
- bimolecular protein encounter：`two-ensemble + orientation/separation sampler`
- reactive enzyme：`protein ensemble + local reactive sampler`

如此才能同时保持 VENUS96 的传统物理用途，又真正支持大分子体系。

---

# 24. 推荐的第一条开发主线

```text
VENUS96 source
    ↓
抽离 trajectory lifecycle
    ↓
抽离 InitialConditionSampler
    ↓
LegacyRRHOQCTSampler 回归
    ↓
加入 MDEnsembleSampler
    ↓
OpenMM backend
    ↓
Aβ42 monomer shooting
    ↓
state transition statistics
    ↓
Aβ42 dimer encounter sampler
```

如果这条路线跑通，后续才值得加：

```text
local RRHO
→ hindered rotor
→ local ZPE protection
→ reactive MLP
→ QM/MM
→ protein reaction trajectories
```

---

## 参考文献 / 起点

1. Zhu Q.; Yu H. **N-Terminal Chirality and Sequence Variations Modulate the Conformational Landscape of Amyloid-Beta 42.** *J. Chem. Inf. Model.* **2026**, 66, 10156–10171. DOI: `10.1021/acs.jcim.6c01001`.

2. 可继续参考 Aβ42 heterogeneous monomer/hydration ensemble、Aβ oligomer REMD、MSM/TPT/committor 与 transition-path sampling 文献，用于后续 state definition 和 kinetic validation。

---

## 当前建议

**Aβ42 第一版不要碰 QCT。**

先把 VENUS96 的“ensemble experiment”抽出来：

\[
\boxed{
\text{REMD/MD ensemble}
\rightarrow
\text{many short unbiased shots}
\rightarrow
\text{state/outcome statistics}
}
\]

验证这个框架后，再将 `LocalRRHO / QCT / ZPE` 作为局部、高频、反应型问题的扩展模块。

这样整个重构不会被传统 QCT 的假设绑死，同时又保留 VENUS96 最有价值的设计思想。
