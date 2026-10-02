# Cytherea 状态与交接（2026-10-02）

2026-10-02 项目由 venus-ng 更名 **Cytherea**，本仓库是更名后在 yayoi 上重新播种的仓库；此前的提交历史只留在 kasuga180 的 `/home/kasuga/gitdirs/venus-ng.git`。

`phase-a`，第二轮修复（L1–L7）已全部提交。快速套件 828 passed（`pytest -q -n 6`）。工作方式：用户要求**线性开发**，由主会话直接实现、自审，一次一件，不派 subagent。

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
| T14b A1 射击 | 未开始（参考已齐，τ = 100 ps；协议见计划 Task 14 的修订版；比较的是行归一化的 T） |
| T15 A2 chignolin | **未完成**：WIP 在 `t15` 分支；预算估计还没出（C36m 文件是 `charmm36_2024.xml`） |
| T16 A3 encounter | **进行中**：`EncounterSampler`（16.1/16.2 通过）、config 接入 `shoot.encounter`、barnase–barstar GBn2 体系（0.15 M 隐式盐，`runs/encounter_system/`）、基准 369 ns/day（2080 Ti）。**30 发 pilot 正在 180 上跑**（见下）；16.3/16.4 的正式发数等 pilot 结果再定（用户 2026-10-02 选的“先 pilot”） |

## 下一步（按优先级）
1. **A3 pilot（180 上，nohup，PID 14736，2026-10-02 11:27 开跑）**：`runs/a3/pilot.sqlite`，日志 `runs/a3/pilot.log`（每 5 发一行）。库归 kasuga（180）所有，分析也在 180 上跑：`python examples/encounter_pair/shoot.py analyze --store runs/a3/pilot.sqlite --stage pilot --out runs/a3/pilot_analysis.json`。中断了就用同一条 `shoot ... --n 30 --stage pilot` 续算（只补缺的 key）。每发约 30–60 min，30 发约 22–30 h。拿到 β 和停止时间后，估 16.3/16.4 的预算，交给用户定。
   注意：pilot 是在 review 修复之前启动的，记录里 `origin_label` 为 None（不影响 β）。
2. 然后 14b（A1 射击；先定 14.3 的做法，见“待用户决定”）或 T15。
2. 14b 之前：
   - IC 约束用 `DistanceConstraints.from_openmm_system(system)`，不需要另做基于 Context 的约束对象；
   - 在 GPU 上实测每次 build 新建 Context 的开销；
   - brief 写明：每个状态至少 50 帧、每帧最多 10 发，并报告设计效应（`TEstimate.n_eff`）；
   - 用 `records_to_transitions` 生成 `estimate_T` 的输入；
   - 14.3（射击数据的 CK）怎么做还没定，见“待用户决定”。
3. Phase B 之前：WE 的汇目前写死为 `"B"`，还不能续算崩溃的 WE（p7 M-2、M-7）；sampled 模式的 PES 检查要传 `fd_atom_groups`（溶质）。

## A1 参考分析的结论（2026-10-02）
- 数据：`runs/ala2_ref`（按 2 段用）加 `r01–r08`，每条去掉 10 ns 预平衡，共 947 ns。命令和完整日志在 `runs/ala2_par/analysis.log`。
- 237 ns 时 CK 不通过，原因已查清。TBA 那次是数据不够，到 947 ns 时已通过（τ = 1 ps，horizon 5.6 ns ≈ 2·t2）。core-start 那次是**选 lag 的规则有缺陷**：规则只看 t2，αL 被访问到以后 t2 是 αL 过程（约 2.8 ns），core-start t2 从 10 ps 起就平了；而 C7eq↔αR 过程（t3）的 core-start 值在 10 ps 时是 316 ps，TBA 是 168 ps，要到 100 ps 左右才收敛，CK 正是在 C7eq、αR 两个对角元上失败。
- 修复：shoot lag 现在还要求 core-start CK 通过（`analyze_ref.py`；新增合成用例 `test_shooting_lag_requires_the_core_start_ck_to_pass`）。CK 在 10、20、50 ps 不通过，100、200 ps 通过，所以 τ = 100 ps。计划 Task 14 的写法已同步修改。
- 统计上的弱点：C7eq↔αL 方向一共只有约 12 次事件，t2 = 2.79 ns，CI 是 [1.72, 4.06]。14.2（射击得到的 t2 落在参考 CI 内）因此是一个宽松的检验。

## 待用户决定
- chignolin 那条 ≥10 µs 参考轨迹的预算（T15 先要把估计做出来）。
- 14.3（射击数据的 CK）的做法。CK 的 horizon 必须达到约 2·t2 ≈ 5.5 ns。要从射击数据直接估 T(kτ)，就得打长度达到 kτ 的 shot：150 帧 × 10 发 × 5.5 ns ≈ 8 µs，单卡要好几天。便宜的替代方案是用射击得到的 T(τ)^k 去对参考的 T(kτ)，这样就是“射击 vs 参考”，而不是射击数据自身的 Markov 检验。另一个折中是只把 horizon 做到 t3 的量级。要在写 14b brief 前定。

## 注意
- 181（kasuga01）上有别的项目在跑（pars），本项目的重计算放 180（`canna@192.168.0.180`，hostname kasuga，16 核，2080 Ti，同一个 NFS home）；快速套件也可以在 180 上跑（`pytest -q -n 6 -p no:cacheprovider`）。
- NFS（`nocto`）的属性缓存很长：集群写的文件，在本机可能看到旧的长度，甚至目录列表也不全。读这类文件前，先用 `dd iflag=direct` 拷到本地，并按 `checkpoint.json` 的记录数核对（2026-10-01 的 A1 数据就是这样拷到 scratch 的）。
- 一张 GPU 上并行跑多个小体系，必须开 MPS（`CUDA_MPS_PIPE_DIRECTORY` 和 `CUDA_MPS_LOG_DIRECTORY` 都要放在节点本地的 /tmp）。不开 MPS 时，8 条加起来只有约 1260 ns/day，比单条跑还慢；开了以后每条约 940 ns/day。

## 已知的延后项（精选）
- T13：`shoot.encounter` 的 runner 要等 Task 16 的 encounter 采样器（现在直接报 NotImplementedError）；通过 config 跑 WE 时，所有 walker 的 origin label 都是 (0,0)，还不能按标签起跑；protocol hash 只含 observable 的名字，不含其定义，CLI 续算靠 sidecar 里的 config hash 把关（`<store>.config.json`）。
- `run_batch` 在 n_workers>1 时不能用 OpenMM 后端，因为 Context 无法 pickle。
- 每次 build 都新建 Context；复用会破坏按 key 派生的 RNG，暂不做。
- reversible bootstrap 的覆盖率略低（0.90–0.945）；帧数很少时，只重抽帧的方案会低估方差，偏差因子为 (F−1)/F。
- 完整清单见 ledger 的 `progress.md`：以 `minor (deferred)` 开头的行，以及所有 `Ruling R1–R41`。
