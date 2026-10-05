# Cytherea 状态与交接（2026-10-03）

2026-10-02 项目由 venus-ng 更名 **Cytherea**，本仓库是更名后在 yayoi 上重新播种的仓库；此前的提交历史只留在 kasuga180 的 `/home/kasuga/gitdirs/venus-ng.git`。

`phase-a`，第二轮修复（L1–L7）已全部提交。快速套件 863 passed（`pytest -q -n 12`，183，2026-10-04）。工作方式：用户要求**线性开发**，由主会话直接实现，一次一件，不派 subagent，不另做审查轮次。

2026-10-03 的外部审查交接见 `docs/reports/review_handoff_2026-10-03.md`（BUG-01..03、DOC-01/02、TODO-01..07），按其顺序推进；进度见下方“下一步”。

## Phase A 进度

| 任务 | 状态 |
|---|---|
| T1 keys / T2 store / T3 后端协议 + 解析势 + PES 测试 / T4 解析传播器 | 完成 |
| T5 IC 门禁 / T6 停止判据 / T7 run_shot + 批执行器 / T8 估计量 | 完成 |
| T10 WE（BinnedWE，带标签约束） | 完成（作为修复包 P7 合并） |
| T12 OpenMM 后端 / INT1 集成 | 完成 |
| T14a A1 丙氨酸二肽的参考脚本和分析 | **完成**：947 ns 参考（10 条轨迹），TBA 与 core-start CK 都通过；**14b 的 τ = 100 ps**（结果在 `runs/ala2_par/analysis/`，数字见 README 的 14b contract 一节） |
| T17 WE 引擎评估 | 完成；**用户已定（2026-10-01）：继续用自研 BinnedWE**（`docs/reports/we_engine_evaluation.md`） |
| **全量 Opus 复查 + 修复** | 7 路审查共找出 4 个 Critical、35 个 Important，分 8 个修复包 P1–P8 修复，由 INT2 集成，**已全部合并** |
| **修复包复审 + 第二轮修复** | 7 路 Opus 复审（`fixreview-*.md`）共 12 个 Important，按 `fixplan2.md` 分 L1–L7 线性修完（新增约定 K9 实际配置 hash、K10 数值失稳、K11 帧权重） |
| T9 A0 解析验收 part 1 | **完成**（9.1–9.4 全部通过；9.2/9.3 每点 shot 数由 400 提到 1600，理由见 `docs/reports/A0_part1.md`） |
| T11 吸收网络 | **完成**：`cytherea.network`；11.1–11.4 全部通过（`docs/reports/A0_part2.md`；WE run 数由 24+24 提到 48+48，理由见报告）。Markov 检验按 (origin label, milestone) 分层比较 |
| T13 配置 / CLI | **完成**：`cytherea.config`（pydantic 模型、单位解析、mode 分发）+ `cytherea run/resume/report`；13.1–13.5 全部通过。依赖新增 pydantic≥2、pyyaml（用户 2026-10-02 批准） |
| T14b A1 射击 | **进行中**：协议按用户 2026-10-03 决定定稿（计划 Task 14 已改写：每帧 10 发、行归一化 T、947 ns 参考照用、14.3 用 5.5 ns 长 shot）。`ck_test_shots`（核心估计量）、`shoot_a1.py`（frames/configs/analyze/pes）及测试已完成；起始帧已选（`runs/ala2_shoot/frames`，αL 50 帧来自 10 次访问）；8 个分片配置在 `runs/ala2_shoot/long`；GPU 冒烟通过。待：吞吐实测 → 报预算 → 开跑 |
| T15 A2 chignolin | **未完成**：WIP 在 `t15` 分支；预算估计还没出（C36m 文件是 `charmm36_2024.xml`） |
| T16 A3 encounter | **进行中**：`EncounterSampler`（16.1/16.2 通过）、config 接入 `shoot.encounter`、barnase–barstar GBn2 体系（0.15 M 隐式盐，`runs/encounter_system/`）、基准 369 ns/day（2080 Ti）。30 发 pilot：前 14 发完成后因更名中断，2026-10-03 已续跑（见下）；16.3/16.4 的正式发数等 pilot 结果再定（用户 2026-10-02 选的“先 pilot”） |

## 下一步（按优先级）
1. **A3 pilot 已跑完，结论 `docs/reports/A3.md`**（30 发，180，2026-10-04 14:47 结束；`runs/a3/pilot_analysis.json`）：0 reaction、23 escape、7 timeout（23%），β（q2）95% 上界 0.10、β∞ 上界 0.15，**valid = False**（timeout > 5%）。7 发 timeout 里 6 发是 r ≈ 2.2–2.8 nm、Q ≤ 0.17 的非天然紧密复合物（50 ns 内不解离也不转天然），1 发（shot 26）在 r ≈ 3.2 nm 附近徘徊。共 519 ns，停止时间中位数 7.4 ns、p90 = t_max。核对、版本迁移和 30 发结果见 `docs/reports/A3_pilot_status_2026-10-03.md`（第 7 节）。**正式协议待用户定**（见“待用户决定”）。
2. **A1 射击（14b）已跑完并分析，报告 `docs/reports/A1.md`**（2026-10-03 20:20 → 10-04 约 10:00；用户批准的 12 h 方案：181/183 两块 2080 Ti 跑 600 发 5.5 ns 长 shot（每帧 4 发，`runs/ala2_shoot/gpu_long`），两台 CPU 跑 900 发 100 ps τ shot（每帧 6 发，`cpu_tau`，CPU 平台）；导出到 `runs/ala2_shoot/exports/{long,tau}`（`shoot_a1.py export`，用户加的），结果 `runs/ala2_shoot/analysis_14b.json`）。
   - 14.2 通过：t2 = 3.63 ns（shot CI [2.37, ∞)），参考 core-start CI [1.51, 3.96]；t3 = 211 [178, 255] ps vs 参考 187 [179, 194]。T(τ) 的 9 个元素参考值全在 shot 的 95% 区间内。
   - 14.3 CK **通过**：D = 0.125（k = 1…55）。T(τ) 用全部 1500 发（用户 2026-10-04 决定；`ck_test_shots(..., tau_only=...)`），T(kτ) 用 600 发长 shot；最大偏差在 αL 长 lag（预测 0.26 vs 实测 0.15 @ 5.5 ns，实测贴着参考）。
   - 14.5 通过：1500 发 0 拒绝。
   - 14.1 **通过**（2026-10-04，183 CUDA mixed，`runs/ala2_shoot/pes_14_1.json`）：FD 最大误差 1.7e-4、单原子 3.4e-4（阈值 5e-3），重复 3.0e-4（mixed 默认 1e-3），470 个坐标检查、58 个跨 0.9 nm 截断的坐标被跳过。此前不通过的原因：PME + 直接截断的能量在截断处跳变，FD 跨截断的坐标误差 28–56 kJ/mol/nm（不跨截断的 double 2.6e-6、mixed 3.8e-4，`pes_cutoff_diag_*.json`）。为此 `pes_suite` 加了截断跨越守卫（后端可选接口 `energy_cutoffs()`，OpenMM 后端已实现）和按精度的 `repeat_rtol` 默认值（double 逐位，mixed 1e-3，single 5e-3）。
   - 温控（2026-10-04，183，`runs/ala2_shoot/thermo/`）：起始速度 GPU 299.9 K、CPU 299.4 K；运行中（自由度 4929）CUDA 298.9 ± 0.2 K、CPU 298.8 ± 0.5 K，dt 改 1 fs 后 CUDA 299.8 ± 0.2 K：约 1 K 的偏低随 dt² 缩小，是 LangevinMiddle 半步速度读数的偏差，不是温控问题；两平台一致。
   - GPU/CPU 按帧配对置换检验：整体 p = 0.14，无平台效应证据（`paired_platform.json`）。
   - 未做：14.4（分界面 committor）。
3. Phase B 之前：WE 的汇目前写死为 `"B"`，还不能续算崩溃的 WE（p7 M-2、M-7）；sampled 模式的 PES 检查要传 `fd_atom_groups`（溶质）。

## A1 参考分析的结论（2026-10-02）
- 数据：`runs/ala2_ref`（按 2 段用）加 `r01–r08`，每条去掉 10 ns 预平衡，共 947 ns。命令和完整日志在 `runs/ala2_par/analysis.log`。
- 237 ns 时 CK 不通过，原因已查清。TBA 那次是数据不够，到 947 ns 时已通过（τ = 1 ps，horizon 5.6 ns ≈ 2·t2）。core-start 那次是**选 lag 的规则有缺陷**：规则只看 t2，αL 被访问到以后 t2 是 αL 过程（约 2.8 ns），core-start t2 从 10 ps 起就平了；而 C7eq↔αR 过程（t3）的 core-start 值在 10 ps 时是 316 ps，TBA 是 168 ps，要到 100 ps 左右才收敛，CK 正是在 C7eq、αR 两个对角元上失败。
- 修复：shoot lag 现在还要求 core-start CK 通过（`analyze_ref.py`；新增合成用例 `test_shooting_lag_requires_the_core_start_ck_to_pass`）。CK 在 10、20、50 ps 不通过，100、200 ps 通过，所以 τ = 100 ps。计划 Task 14 的写法已同步修改。
- 统计上的弱点：C7eq↔αL 方向一共只有约 12 次事件，t2 = 2.79 ns，CI 是 [1.72, 4.06]。14.2（射击得到的 t2 落在参考 CI 内）因此是一个宽松的检验。

## 待用户决定
- A3 正式协议（pilot 跑完后定）。2026-10-04 的分析：30 发里 6 发在前几 ns 进入 r < 3 nm，之后几乎不再分开（整个 pilot 只有 1 次离开；单指数寿命 MLE 270 ns，95% 下界 49 ns）；23 发 escape 全程 r ≥ 3.26 nm。所以按天然接触判据（Q ≥ 0.3）既得不到反应，也压不住 timeout；延长 t_max 每发被困的 shot 要几百 ns。按设计 4.3 Stage A 的“持久 encounter”判据离线回放（只用已存的 r）：r ≤ 3.5 nm 持续 1 ns → q₂：7/23/0，β∞ = 0.31 [0.16, 0.51]；q₁：6/24/0，β∞ = 0.33 [0.16, 0.54]；平均停止 6.6 ns。建议改用该判据（Task 16 只验流程，不对实验 k_on），待用户定。
- chignolin 那条 ≥10 µs 参考轨迹的预算（T15 先要把估计做出来）。

## 注意
- 181（kasuga01）上有别的项目在跑（pars），本项目的重计算放 180（`canna@192.168.0.180`，hostname kasuga，16 核，2080 Ti，同一个 NFS home）；快速套件也可以在 180 上跑（`pytest -q -n 6 -p no:cacheprovider`）。
- NFS（`nocto`）的属性缓存很长：集群写的文件，在本机可能看到旧的长度，甚至目录列表也不全。读这类文件前，先用 `dd iflag=direct` 拷到本地，并按 `checkpoint.json` 的记录数核对（2026-10-01 的 A1 数据就是这样拷到 scratch 的）。
- 一张 GPU 上并行跑多个小体系，必须开 MPS（`CUDA_MPS_PIPE_DIRECTORY` 和 `CUDA_MPS_LOG_DIRECTORY` 都要放在节点本地的 /tmp）。不开 MPS 时，8 条加起来只有约 1260 ns/day，比单条跑还慢；开了以后每条约 940 ns/day。

## 已知的延后项（精选）
- T13：通过 config 跑 WE 时，所有 walker 的 origin label 都是 (0,0)，还不能按标签起跑；protocol hash 只含 observable 的名字，不含其定义，CLI 续算靠 sidecar 里的 config hash 把关（`<store>.config.json`）。
- `run_batch` 在 n_workers>1 时不能用 OpenMM 后端，因为 Context 无法 pickle。
- 每次 build 都新建 Context；复用会破坏按 key 派生的 RNG，暂不做。
- reversible bootstrap 的覆盖率略低（0.90–0.945）；帧数很少时，只重抽帧的方案会低估方差，偏差因子为 (F−1)/F。
- 完整清单见 ledger 的 `progress.md`：以 `minor (deferred)` 开头的行，以及所有 `Ruling R1–R41`。
