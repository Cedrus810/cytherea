# Cytherea：Bug 与 TODO 交接

日期：2026-10-03（Asia/Tokyo）  
审查基线：`bbe9a71`（`docs: add Chinese README`）  
当前目录：`/home/ruigengji/cytherea`；项目原名、原目录为 `venus-ng`。  
范围：记录审查发现、证据、优先级和后续交付；本交接不表示相关修复或生产作业已经执行。

## 1. 当前判断

核心架构和测试基础已经比较扎实。下一阶段优先完成真实分子上的验证闭环：先核对 A3 pilot 故障与存量数据，再实现 A1 射击及参考对照。A3 的正式采样、A2 长参考轨迹和 Aβ42 应用按验证结果与预算逐步推进。

目前不能把 A3 标记为完成，也不能把 A1 参考分析完成等同于 A1 射击验收完成。

## 2. 已修复并复核的问题

以下四项来自前两轮审查，属于已修复事项，不应重新列为待修 bug。

| 编号 | 原问题 | 复核结果 |
|---|---|---|
| FIX-01 | `records_to_transitions` 排序记录后未同步重排外部权重 | 权重随记录一起重排；原复现算例得到正确转移概率 0.9 |
| FIX-02 | encounter 的 `origin_label` 被 `run_shot` 写成 `None` | `InitialState.origin_label` 传入记录；复现得到 `(0, 1)` |
| FIX-03 | EncounterSampler 未检查伙伴 B 的温度一致性 | 两个池的温度不一致会拒绝；启用 kB 检查时同时检查 kB·T 与 kT |
| FIX-04 | A3 分析忽略 `nonfinite`，小样本区间退化为 `[1, 1]` | 使用核心估计量，报告 `n_nonfinite` 与 `valid`；失稳算例得到 `valid=False` 和非退化区间 |

验证记录：2026-10-02 在更名前的 checkout 对 `test_estimators.py`、`test_encounter.py`、`test_engine.py` 运行了相关测试，结果为 **225 passed、2 deselected**。更早一次快速全套结果为 **824 passed、15 deselected**。当前 `docs/STATUS.md` 记载 **828 passed**，本次交接没有重新执行更名后仓库的全量或 slow 验收；不要把历史测试结果写成本次新验证。

## 3. 当前 Bug 与运行故障

### BUG-01 / P1：A3 pilot 写库失败，存量数据和运行状态待核对

**状态：日志确认故障；根因和当前进程状态尚未确认。**

来源：[pilot.log](../../runs/a3/pilot.log)。2026-10-03 审查时使用 direct I/O 读取日志，避免只依赖 NFS 属性缓存。

日志中已报告：

- keys 0–4：4 次 escape、1 次 timeout，累计 wall 18851 s。
- keys 5–9：4 次 escape、1 次 timeout，累计 wall 36980 s，约 10.3 h。
- 随后在 `Store.append` → `_raw_connect` 中出现 `sqlite3.OperationalError: unable to open database file`。
- traceback 中的脚本与源码路径仍为 `/home/ruigengji/venus-ng/...`。

**影响：** 文档仍称“30 发 pilot 正在跑”，但日志已经记录异常退出；不能据此认定 30 发已经完成。日志按每 5 发汇总，实际提交记录数可能多于已打印的 10 发，需要查库确认。

**待核实：** 更名或目录迁移使旧路径失效是一个可能原因，日志本身不足以证明。还需检查原作业的 store 参数、工作目录、路径可达性和权限。

**TODO / 验收：**

- [ ] 在数据库所属主机核对当前进程、已提交 ShotKey、失败记录及最后一次成功写入。
- [ ] 查明失败原因，记录实际使用的命令、路径、代码版本和库归属。
- [ ] 确定续算或迁移方案，检查重复 key 不新增、缺失 key 可补跑、版本差异会被检测。
- [ ] 更新 `docs/STATUS.md`，明确“已确认完成数 / 未完成数 / 当前运行状态”。

数据迁移遵循既有 Store 契约：停止旧主机写入后，在库所属主机用 `Store.backup_to` 生成一致副本，再按约定接管；不手工复制运行中的 SQLite 文件。

### BUG-02 / P2：旧 pilot 的“直接续算”说明与版本保护不一致

**状态：已复现版本拒绝；实际旧 pilot 的所有 hash 仍需所属主机核对。**

来源：[STATUS.md](../STATUS.md)、[batch.py](../../src/cytherea/exec/batch.py)、[shot.py](../../examples/encounter_pair/shoot.py)。

pilot 在审查修复前启动，之后源码已改变，且项目发生包名迁移。`run_batch` 默认比较源码身份、物理配置和协议 hash；示例 `cmd_shoot` 没有显式的旧版本迁移流程。审查用“相同物理 / 协议、不同代码身份”的存量记录复现了 `ResumeConfigMismatchError`。

**TODO / 验收：**

- [ ] 明确旧 pilot 是固定旧版本继续完成，还是经核对迁移为新 run；不要把“同一命令”当作跨版本续算保证。
- [ ] 若接受版本差异，先逐项确认改动对动力学、采样、记录和分析的影响，保留迁移 provenance。
- [ ] 不用宽泛的 `allow_config_change=True` 掩盖未经检查的物理或协议变化。
- [ ] 为文档中的实际续算流程提供可重复的验证结果。

### BUG-03 / P2：A3 差值后验给不可能的配对结果分配概率

**状态：当前代码仍存在。**

来源：[shoot.py](../../examples/encounter_pair/shoot.py) 的 `diff_interval`：对全部 3×3 单元使用 `Dirichlet(counts + 0.5)`。

在当前分析协议下，q1 < q2，两个停止规则具有相同的反应判据、持久时间和 timeout，并回放同一条轨迹。如果 q1 判为 reaction，q2 也应判为 reaction。因此 `(reaction, escape)` 等单元是结构性不可能事件，不能仅把它们当作“尚未观察到的事件”加伪计数。

此前复现使用 30 发合成数据（3 对 reaction/reaction、27 对 escape/escape）；当前模型给四个不可能单元分配的后验平均概率质量合计约 **5.8%**。该数字是合成示例，不是实际 pilot 的测量值。

**TODO / 验收：**

- [ ] 按当前停止规则枚举可达配对结果；不可能单元保持零概率，异常输入明确拒绝。
- [ ] 核对两侧边际概率与差值估计所用的先验及条件化方式，解释它们与单侧估计量的关系。
- [ ] 补充稀有反应、零反应、相同结果、escape/reaction、timeout 和 nonfinite 的合成检验。
- [ ] 用已知真值的重复模拟评估区间覆盖和判定表现，报告先验敏感性；不能只验证“区间不塌缩”。
- [ ] 区分后验可信区间与频率学置信区间；存在失稳或过多 timeout 时，仍保持整体无效标记。

## 4. 协议与文档待同步

### DOC-01：A1 射击协议存在冲突

来源：[施工计划 Task 14](../superpowers/plans/2026-09-30-cytherea-phase-a.md)、[A1 README](../../examples/alanine_dipeptide/README.md)、[STATUS.md](../STATUS.md)。

| 项目 | 施工计划中的写法 | 当前 README / 状态约定 |
|---|---|---|
| 每帧发数 | 20 发 | 每帧最多 10 发，并报告设计效应 |
| T 的比较 | reversible estimator | 比较行归一化 core-start T；使用 `estimate_T(..., reversible=False)` |
| 参考数据 | 一条 ≥1 μs | 当前为 10 条轨迹 / 947 ns 有效参考数据 |
| 14.3 CK | 射击数据自身 CK 通过 | 实施方式和预算仍待确定 |

- [ ] 开始 14b 前统一协议，明确参考数据是否满足最终验收约定。
- [ ] 保持 τ=100 ps、φ/ψ 每 1 ps 标注、与参考一致的 core/TBA 定义及传播物理。
- [ ] 不把射击预测与参考数据的交叉比较写成“射击数据自身 CK”。

### DOC-02：交接状态有过期内容

- [ ] A3 的运行状态按 BUG-01 核对结果更新。
- [ ] 删除“shoot.encounter runner 尚未接入 / 直接报 NotImplementedError”的过期描述；当前 runner 已接入。
- [ ] 统一更名前后路径、版本和续算说明；原 PID、时间戳不能替代当前运行状态检查。

## 5. 下一步 TODO：按顺序执行

### TODO-01：收尾 A3 故障与数据盘点

先执行 BUG-01、BUG-02 的核对，交付一份准确的 pilot 状态记录。已打印的前 10 发没有 reaction，timeout 为 2/10；这是部分样本的诊断，不能据此断言真实结合概率为零，也不能宣布整个 30 发 campaign 无效或完成。

### TODO-02：修正 A3 差值分析

完成 BUG-03，并用修正后的分析方法评估存量数据。正式扩大发数前，给出事件率、timeout、停止时间和预算的不确定性。

### TODO-03：实现 A1 射击 runner 与参考对照（下一条主线）

- [ ] 实现参考帧加载：core 内部帧、来源轨迹/时间、frame_id、温度、权重与 box 信息可追溯。
- [ ] 处理已有 DCD 的精度与展开坐标：按 README 恢复分子到盒内、投影约束，并使用最小镜像的 clash 检查。
- [ ] 起始帧尽量分布在不同轨迹和时间块，报告相关性；帧数不等于独立样本数。
- [ ] 用一致的状态定义和带初始标签的 TBA 标注 shot；φ/ψ 观测间隔为 1 ps。
- [ ] 用 `records_to_transitions` 生成估计量输入，比较完整行归一化 T，报告 n_eff、拒绝率和全部失败原因。
- [ ] 同时报告 t2、t3。αL 相关参考跃迁只有约 12 次，t2 的宽 CI 是较弱检验，不能单独证明全部动力学正确。
- [ ] 完成适合生产精度的 PES 检查及现有 Task 14 的其余验收。
- [ ] 交付 `docs/reports/A1.md`：协议、命令、数据标识、T、时间尺度、CI、成本和未完成项。

**初始规模建议，尚非生产运行批准：** 3 个状态 × 50 帧 × 10 发 × 100 ps = **150 ns** 积分量。实际墙钟还包括 Context 构建、IC 检查、观测和写库；应按相关性、有效事件数和精度目标调整发数。

### TODO-04：单独明确 A1 CK 的验收方案和预算

现有设计要求 horizon 达到约 2·t2 ≈ 5.5 ns。若上述 1500 发全部延长到 5.5 ns，积分量约 **8.25 μs**，与初始短射击预算不同。

- [ ] 给出可审阅的 CK 方案：自有长 shot、射击预测对参考数据的交叉验证，或分阶段诊断；明确每种方式证明了什么。
- [ ] 跨数据交叉比较不替代仍未完成的自有数据 CK 验收。
- [ ] 所需长作业预算确定后再执行，不默认启动约 8 μs campaign。

### TODO-05：A3 做小预算的 WE / 暴力射击对照

- [ ] 根据 pilot 决定 q、t_max、反应判据与正式样本需求；协议变化使用可追溯的新 run。
- [ ] 为真实 OpenMM encounter 构建小预算 WE campaign，与暴力射击使用一致初始分布和物理。
- [ ] 处理现有 WE sink 固定为 `B` 与 BSurface `reaction` 的衔接，核对来源标签及退出事件。
- [ ] 对多个独立 WE run 估计不确定度；不把有共同谱系的 walker 当作独立样本。
- [ ] 比较 β/β∞、有效事件数、ns/day、每 GPU 小时产出和区间精度；实测后判断是否值得扩大。
- [ ] 完成 Task 16.3–16.5 并交付 `docs/reports/A3.md`。

### TODO-06：A2 chignolin 先过预算门

已有 `runs/chignolin_ref_bench/checkpoint.json` 只显示 600 ps、因 signal 15 停止，不能当作长参考轨迹已完成。

- [ ] 在 GPU 空闲时测吞吐，给出 ≥10 μs 参考轨迹的 GPU·天预算。
- [ ] 核对旧 `t15` WIP 的可获取性和接口兼容性；不直接按旧接口继续。
- [ ] 长参考轨迹预算确认后，完成折叠/去折叠、T 和 committor 对照，交付 `docs/reports/A2.md`。

### TODO-07：Phase B 前补齐运行基础

- [ ] 评估并实现适当的 WE checkpoint，明确坐标、速度、谱系、RNG 和迭代边界的恢复契约。
- [ ] 确认 sampled PES 检查使用需要的 `fd_atom_groups`。
- [ ] 核对 config 驱动 WE 的来源标签能力，以及 OpenMM 批执行器的进程/Context 限制。
- [ ] 在更名后的干净环境运行安装、CLI smoke 和必要测试，记录包名/路径迁移影响。

## 6. 后续研究方向

完成 Phase A 的验证门后，先推进 Aβ42 单体的输入系综、力场验收和公共状态空间，再推进构象条件化的 encounter / dimer 过程。

最终目标是量化变体效应中两项各自的贡献：构象占比变化，以及同一构象的结合倾向变化。dimer 分解前必须检验时间尺度分离；若初始构象在 encounter 过程中快速失去记忆，应按既有设计调整条件变量或研究范围，而不是强行使用初始标签解释结果。

## 7. 执行约定

遵循当前 `CLAUDE.md` 和 `docs/STATUS.md`：线性开发、一次一件；既有参考轨迹只读；计时与 benchmark 前单独确认 GPU 空闲；跨主机 SQLite 按 Store 所有权与 backup 契约迁移。长作业的范围、协议和预算先形成可审阅的方案。

本交接的下一件具体工作：**核对 A3 已提交记录和路径故障，修正差值分析，然后开始 A1 射击 runner 与验收报告。**
