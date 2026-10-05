# results/：当前结果的原始产物

`docs/reports/` 各报告引用的小型机器可读产物（分析 JSON 等）收录在这里；
图在 [`docs/reports/figures/`](../docs/reports/figures/)。轨迹、checkpoint、
SQLite store 等大数据留在 `runs/`（gitignore，共享 NFS），不进 git。

目录结构与 `runs/` 中的来源一致。这些文件是快照，重跑后随报告一起更新；
最后更新 2026-10-04。

## ala2_ref/ — 14a 参考分析

| 文件 | 内容 | 引用 |
|---|---|---|
| `analysis.json` | 947 ns 参考（10 条独立轨迹）的 MSM 分析：状态定义、ITS、CK 检验、shoot lag 选择（τ = 100 ps 的依据）与 14b contract T | `docs/reports/A1.md` 的 T(τ)/ITS/CK 节；STATUS「A1 参考分析的结论」 |

## ala2_shoot/ — 14b 射击（A1 验收）

| 文件 | 内容 | 引用 |
|---|---|---|
| `analysis_14b.json` | 主分析：T(τ)（元素、Jeffreys 区间、n_eff）、与参考的逐元素对照、ITS（14.2）、`ck_test_shots` 的 CK（14.3）、IC 拒绝率（14.5） | `docs/reports/A1.md` 全文 |
| `pes_14_1.json` | 14.1 PES 一致性结论（183，CUDA mixed，sampled 模式） | `docs/reports/A1.md`「PES（14.1）」 |
| `pes_diag.json`、`pes_cutoff_diag*.json` | 截断诊断：FD 跨 0.9 nm 截断 vs 不跨坐标的误差（守卫的依据） | 同上 |
| `paired_platform.json` | GPU/CPU 按帧配对置换检验（p = 0.14，无平台效应证据） | `docs/reports/A1.md`「温控」 |
| `thermo/*.json` | 温控检查原始数据：CPU/CUDA × 4，dt = 2 fs 与 1 fs（LangevinMiddle 半步速度读数偏差的诊断） | 同上；图 `A1_thermo.png` |
| `frames.json` | 起始帧元数据：来源轨迹、DCD 帧号、时间、φ/ψ、state、core visit、seed | `docs/reports/A1.md`「实际跑的方案」；图 `A1_frames.png` |

## a3/ — A3 encounter pilot（Task 16）

| 文件 | 内容 | 引用 |
|---|---|---|
| `pilot_analysis.json` | 30 发 pilot：β、β∞、停止时间统计、valid = False（timeout > 5%） | `docs/reports/A3.md`「pilot 数据」 |
| `replay_encounter.json` | 持久 encounter 判据（r ≤ 3.5 nm 持续 1 ns）的离线回放：q₁/q₂ 的 β∞ 与平均停止时间 | `docs/reports/A3.md`「持久 encounter 判据的离线回放」 |
