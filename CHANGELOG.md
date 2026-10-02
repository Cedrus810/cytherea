# 更新日志

格式参照 [Keep a Changelog](https://keepachangelog.com/);版本号遵循 SemVer,0.x 阶段接口可能调整。

## [0.1.0] - 2026-10-02

Phase A 首个快照:T1–T14a/T17 完成,两轮修复(全分支 7 路复审:4 Critical + 35 Important → 修复包 P1–P8;修复包复审:12 Important → L1–L7)全部合并,快速套件 829 passed(`pytest -q -n 6`)。A1 射击(14b)、A2 chignolin(15)、A3 encounter 正式发数(16.3/16.4)未完,进行中。

本版本同时记录:项目由 **venus-ng 更名 Cytherea**,仓库在 yayoi 上重新播种;更早的提交历史只存于 kasuga180 本地盘,本仓库的分组提交是快照的展示性重排,不代表各提交点可独立构建。

### Added
- **keys [T1]**:ShotKey / SegmentKey / IterKey 确定性派生,`derive_rng` 子流,`key_digest`。
- **store [T2]**:SQLite 运行记录库——所有者主机检查、协议/配置 hash 续算把关(R38/R39)、`backup_to` 备份 API(可从崩溃的 hot journal 恢复)、`summaries_many` 快速摘要。[L3]
- **backends [T3–T4]**:PotentialBackend / Propagator 协议(`build`、`energy_forces(x, box)`、on-step 速度约定、`dt` 属性);解析势(双阱、Müller–Brown、ChannelDoubleWell2D、LJ cluster、谐振子)与解析传播器(Euler–Maruyama、BAOAB、Verlet、过阻尼);PES 一致性套件(NVE 总能量漂移、不变量、有限差分力检查,采样模式按原子组分层)。[R11/R14/R20/R22, P4, L7]
- **ic [T5]**:EnsembleFrameSampler / EnsembleFramePool——能量窗、最小原子间距、COM 动量、温度自由度等结构检查;约束投影(输入容差 1e-3,OpenMM CCMA 投影 1e-10);`from_openmm_system`。[R17/R18, P2, L4]
- **encounter [T16a]**:EncounterSampler(barnase–barstar 结合对的起跑采样,GBn2 隐式盐)。
- **observe [T6]**:事件语义与停止判据(FixedLag、吸收、persistence、timeout),浮点容差统一,消除相位相关的伪超时。[R16, P3]
- **estimate [T8]**:`estimate_T`(状态内帧簇 bootstrap)、`hierarchical_bootstrap`(状态固定分层,即 S1)、committor / k_on(n_eff = min(KG, Kish),永不反保守)、`ck_test`(全局零中心 bootstrap D 统计量,稀疏链不误报)。[R19/R23–R30, P6, L5, K11]
- **we [T10]**:自研 BinnedWE(带标签约束、按权重 recycle、segment own-rows / lineage 回放)。WE 引擎评估(wepy / WESTPA 对比)后用户拍板:继续自研。[T17]
- **engine [T7]**:`run_shot` 生命周期(停止判据、NaN 观测行、数值失稳 `NumericalInstabilityError`)、`run_batch` 并行执行(仅父进程写库、崩溃续算、`on_before_append` 钩子)。[R33–R35, K10]
- **openmm [T12]**:OpenMM 后端——每 dt_obs 分块回 CPU 计算 observable;on-step 速度换算与速度约束投影(R21/R28/R31);barostat/Andersen 的进程级 RNG 防护;OpenMMException 中 NaN → `NumericalInstabilityError`;代码身份 hash(K9:包源码 + System/Topology sha256)。[P1, P5, L1, L2]
- **network [T11]**:`StageNetwork` / `build_transitions` / `solve_absorption` / `markov_test`——core-set milestoning、直接吸收概率估计(经 split/merge 无偏)、按 (origin label, milestone) 分层的 Markov 检验、以 WE run 为单位的零中心 bootstrap。
- **config / CLI [T13]**:pydantic schema(extra=forbid、判别联合、量纲检查)、单位解析("`<number> <unit>`")、`cytherea run/resume/report`、sidecar config hash 续算把关、`save_frames` / `load_frames`(npz)。
- **A0 解析验收 [T9, T11]**:9.1–9.4 与 11.2–11.4 全部通过(β∞ 0.4984±0.0080;committor RMSE 0.0141 / 0.0121;11.2 RMSE 0.013;11.4 通过,RMSE 0.010 / 0.0016)。报告:`docs/reports/A0_part1.md`、`A0_part2.md`。
- **A1 参考 [T14a]**:ala2 显式溶剂 947 ns(10 条轨迹)参考与分析管线;TBA 与 core-start CK 双通过后 shoot lag 定为 100 ps(lag 规则同时要求两个过程过 CK);`analyze_ref.py` 支持多段拼接。
- **examples**:toy_doublewell / toy_diffusion / toy_network / alanine_dipeptide / encounter_pair;`cytherea.zsh` 为 PBS 作业模板。

### Fixed
- 全分支 7 路 Opus 复审:4 Critical(frame.time 作 shot 时钟、NFS 上 SQLite WAL 不安全、SegmentKey 缺 global_seed、WE stop rule 逐段重置)+ 35 Important → 修复包 P1–P8,INT2 集成合并。
- 修复包 7 路复审再修 12 Important → L1–L7:代码身份 K9、数值失稳 K10、帧权重 K11、`backup_to`、输入约束容差、`shot_weights` 枚举、有限差分采样模式与功率门。

### Changed
- 2026-10-02:项目 venus-ng → **Cytherea**;提交信息不再带 AI 署名。
- 已定的科学选择(详见 CLAUDE.md):Aβ42 用 CHARMM36m(`charmm36_2024.xml`)+ 修正 TIP3P(Huang 2017,Phase B 前核对原文);暂不做粘度校正,只报告原始值并注明水模型与 γ;bootstrap 默认状态固定、状态内重抽帧(S1);PES 一致性检查分 strict / sampled 两层(S2)。
