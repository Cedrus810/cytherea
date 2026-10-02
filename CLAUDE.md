# Cytherea：给 Claude 的项目说明

Cytherea 用 Python + OpenMM 重建 VENUS96 的功能结构，目标是溶液中的蛋白质。VENUS96 是 Hase 组的经典轨迹程序。第一个应用是 Aβ42 IDP，只放在 `examples/` 下，不进核心代码。
**开工前先读 `docs/STATUS.md`**，里面有当前状态、待办、已做的决定，以及还在等用户拍板的事项。

## 权威文档
- 设计：`docs/design/VENUS96_AB42_IDP_design_v2.md`（v1 原稿放在同一目录下，只作参考）
- Phase A 施工计划：`docs/superpowers/plans/2026-09-30-cytherea-phase-a.md`
- SDD ledger 及所有审查、修复报告：`.superpowers/sdd/2026-09-30-cytherea-phase-a/`（已被 gitignore，只存在于本机）

`/home/ruigengji/venus96/` 只作为 F77 原始代码的只读存档。那里的 v2 设计稿和计划是**过期副本**，不要再用。

## 环境与仓库
- 环境：`export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba; export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3; source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh; mamba activate openmm_dev`
- 包以 editable 方式从本 checkout 安装，所以直接 `pytest -q` 就是快速套件。slow 测试要显式用 `-m slow` 跑。在 git worktree 里跑测试必须加 `PYTHONPATH=src`。
- **git 仓库就在树内**（`/home/ruigengji/cytherea/.git`；yayoi 的 /home 是本地 xfs 盘，可直接写）。2026-10-02 项目由 venus-ng 更名 Cytherea 后重新播种，此前的历史（T1–T13、修复包 P1–P8/L1–L7 的提交）只存在于 kasuga180 本地盘的 `/home/kasuga/gitdirs/venus-ng.git`，yayoi 够不着；在 kasuga 的 NFS home 上写 git 对象会报权限错误，那是旧仓库用独立 gitdir 的原因。**现有提交是快照按模块/任务的展示性分组，不代表各提交点可独立构建，别拿来做 bisect。**
- 工作分支是 `phase-a`（种子提交后的工作分支）；`main` 与它同点起步。旧仓库的 `t9`、`t15` WIP 分支没有搬过来（在 180 的 gitdir 里），要用得先取回，而且写的时候用的是修复前的接口，拿来用之前要先改。
- commit 信息**不加任何 AI 署名**（用户 2026-10-02 明确要求：贡献者里没有 Claude）。
- 不新增依赖。scipy、deeptime、pydantic（≥2）、pyyaml 已经声明（后两个 2026-10-02 为 T13 加入）。

## 工作规则（来自用户的明确要求）
- **subagent 的模型**：所有审查、复审一律用 opus；实现可以用 sonnet，但后面必须接 opus 审查；不用 haiku。
- **施工计划只写规格**：接口、测试、验收，不写实现代码。
- **GPU**：做正确性检查或生产运行时可以和别人共用 GPU。计时和 benchmark 必须在 GPU 空闲时跑：先单独执行一次 `nvidia-smi` 确认空闲，再另起一条命令运行，两步不要写进同一条命令。
- **长时间作业**：用 harness 的后台任务运行，不写 `sleep` 等待循环，杀进程时按 PID 精确地杀。
- **`runs/` 目录**：已被 gitignore，放在共享 NFS 上。`runs/ala2_ref`（237 ns）和 `runs/ala2_par/r01–r08`（各 100 ns）是 A1 的参考轨迹，已经跑完。除非明确要续跑，否则**只读**。
- **SQLite 记录库只归一台主机**（S3）。跨主机续算：在旧主机上用 `Store.backup_to(copy)` 拷贝，再在新主机上用 `takeover=True` 打开。不要手工拷库文件。
- 不拷贝 VENUSpy 的代码（它没有 LICENSE；用户 2026-10-01 决定不联系作者询问许可证，此事不再提），不移植也不包装 F77，不接 ASE。

## 用户已定的科学选择
- Aβ42 用 CHARMM36m（`charmm36_2024.xml`）加修正 TIP3P（水 H 原子 LJ ε = −0.1 kcal/mol，按 Huang 2017，Phase B 开始前要核对原文）。A2 的 chignolin 用标准 CHARMM TIP3P。A1 用 amber14 + TIP3P-FB。
- 暂不做粘度校正：只报告原始值，同时写明所用的水模型和 γ。
- bootstrap 默认的做法是：状态固定，只在状态内重抽帧（S1）。PES 一致性测试分为 strict 和 sampled 两层（S2）。
