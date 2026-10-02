# WE 引擎选型评估：wepy vs WESTPA vs 自研 `BinnedWE`

**这份报告只提供事实和建议，最终选型由用户决定。**

- 评估对象：
  - `wepy`，GitHub `ADicksonLab/wepy`，克隆时 HEAD `944f63f03a1155811291f4536177e4c975c91ece`（2026-01-22）；另外从 PyPI 拉取了 `wepy==1.2`（`__about__.py` 内写的是 `1.1.0`，PyPI 的版本号与仓库内 versioneer 字符串不一致，但代码结构一致）用于可导入性测试。
  - `WESTPA`，GitHub `westpa/westpa`，克隆时 HEAD `eb63b46fcd662447a7e8f08111c5809dbf5d81f5`（2026-09-24）。
- 方法：按 controller ruling R8，没有装进 `openmm_dev`。两个仓库都 `git clone --depth 1` 到 scratch 目录只读评估；另外 `pip install --no-deps --target <scratch>/site wepy` 连同它缺失的纯 Python 依赖（`dill`、`geomm`、`tabulate`、`multiprocessing_logging`，`numpy/h5py/networkx/pandas/click/scipy/jinja2/pint/mdtraj/tables/tqdm` 在 `openmm_dev` 里已有）到同一个 scratch 目录，通过 `PYTHONPATH=<scratch>/site` 叠加在 `openmm_dev` 之上做真实 `import` 和一次小规模计时实验。`openmm_dev` 本身没有任何改动。WESTPA 额外缺 `pyzmq`、`blessings`，没有为它做同样的 import 测试（见问题 4/5 的说明）。

---

## 1. 许可证

**wepy**：仓库根目录 `LICENSE`，原文：

```
MIT License

Copyright (c) 2017, 2020 ADicksonLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. ...
```

`pyproject.toml` 里 `license = "MIT"`，与 LICENSE 文件一致。

**WESTPA**：仓库根目录 `LICENSE`，原文：

```
MIT License

Copyright (c) 2013 WESTPA Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, ...
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, ...
```

`pyproject.toml` 里 `license = "MIT"`、`license-files = ["LICEN[CS]E*"]`，一致。

**结论**：两者都是 MIT，都**可以**作为依赖使用，没有 copyleft、没有 VENUSpy 那种“没有 LICENSE 文件”的阻断问题。这一条不构成选型障碍。

---

## 2. 能否实现 Task 10 的标签约束 merge

Task 10 的要求：`Walker.origin_label: tuple[int, int]`，merge 只允许在同一 `(bin, origin_label)` 内进行，`allow_cross_label_merge=True` 时才放开。

**wepy**：`wepy.resampling.resamplers.resampler.Resampler` 是一个几乎不设限的抽象基类（`src/wepy/resampling/resamplers/resampler.py`），真正的约束只有 `min_num_walkers`/`max_num_walkers`；`resample(self, walkers, debug_mode=False)` 要自己实现全部逻辑，返回 `(resampled_walkers, resampling_data, resampler_data)`。内置的 `REVOResampler`、`WExploreResampler` 都不认识“标签”这个概念，也没有子分组机制。要做标签约束，必须**整个 `resample()` 方法自己写**：自己维护 walker 到 origin_label 的映射（wepy 的 `Walker`/`WalkerState` 本身没有这个字段，需要在自定义 `WalkerState` 子类里加），自己实现 split/merge 决策，只在同标签内选 merge 候选。技术上完全可行——wepy 在这一层没有任何东西会主动阻止，但也没有任何东西会帮忙，等于是在 wepy 的框架壳子里重写一遍 Task 10 的 `BinnedWE`。

另外一个隐患：如果打算复用 wepy 自带的 `CloneMergeDecision.action()`（`src/wepy/resampling/decisions/clone_merge.py`）去真正执行 clone/merge（而不是只产生决策记录），它调用的是 `wepy.walker.split()` 和 `wepy.walker.keep_merge()`；`keep_merge()` 内部用 `random.choices()`（Python 全局 `random` 模块，`walker.py:113`）选哪个被合并 walker 的状态被保留。这与标签约束无关，但意味着"自己写 resample() 决策，再交给 wepy 默认的 action 去执行"这条路径不能直接用（见问题 3）。

**WESTPA**：`WEDriver._run_we()`（`core/we_driver.py:650`）原生支持按“子分组”（subgroup）独立做 split/merge：

```python
subgroups = self.subgroup_function(self, ibin, **self.subgroup_function_kwargs)
```

`subgroup_function` 是一个可配置回调（默认 `_group_walkers_identity`，`we_driver.py:125`），对每个 bin 调用一次，把 bin 内的 walker 划分成若干子组；随后的 `_split_by_weight`/`_merge_by_weight`/`_merge_by_threshold` 都是**逐子组**执行的（`we_driver.py:668-712`），子组之间不会因为这几个函数而 merge。把 `subgroup_function` 换成“按 `origin_label` 分组”，能在**大多数情况下**不改 WESTPA 核心代码就拿到“merge 只在同一 `(bin, origin_label)` 内进行”；这是 WESTPA 已有的、为 MAB（minimal adaptive binning）等场景设计的机制（`core/binning/binless.py` 里 `group_function` 是同一思路的另一处用法），不是纯粹臆测。

**但这个隔离不是无条件成立的，有一个具体的失效场景（2026-10 复审发现，已用 westpa @ `eb63b46f` 核实）**：`_run_we()` 里，当一个 bin 内的子组数 `len(subgroups)` **大于** `target_count = self.bin_target_counts[ibin]` 时（`we_driver.py:677` 的 `>=` 分支先把每个子组各自合并成 1 个 walker，得到 `len(subgroups)` 个 walker；随后 `we_driver.py:691-692` 无条件地在 `len(subgroups) > target_count` 时调用 `self._adjust_count(bin, subgroups, target_count)` 去把 walker 数进一步砍到 `target_count`），`_adjust_count` 内部在这种情况下会**放弃子组边界，把整个 bin 池化成一个单一分组**：

```python
# we_driver.py:558-561
if len(subgroups) > target_count:
    sorted_subgroups = [set()]
    for i in bin:
        sorted_subgroups[0].add(i)
```

之后的 merge 循环就是在这个池化后的单一分组里挑“权重最低的两个 walker”合并（`we_driver.py:582-595`），完全不检查这两个 walker 原本属于哪个子组——也就是说，只要一个 bin 里出现的**不同 `origin_label` 数量超过该 bin 的 `target_count`**，就会**静默地跨标签 merge**，直接违反 Task 10 的约束。这不是边角情形：标签数是 `K^2` 类（设计 §4.5），如果 `target_per_bin` 设得比某个 bin 里出现的标签种类少（高维/稀疏区域尤其容易发生），隔离就会失效，而且失效时不会报错或警告。

**隔离成立的前提**：每个 bin 的 `bin_target_counts[ibin]` 必须 ≥ 该 bin 内实际出现的 distinct `origin_label` 数（即 `len(subgroups) <= target_count`，触发的是 `we_driver.py:562-563` 的 else 分支，子组边界保留）。这不是引擎自动保证的，需要我们的适配器负责：要么在构造 `bin_target_counts` 时**强制保证**这个不等式（例如按“每个可能标签至少分配 1 个目标名额”的方式设定分箱/目标计数），要么放弃依赖 WESTPA 原生的 `_adjust_count`，改成对 `_run_we`/`_adjust_count` 打补丁或整个重写这一段逻辑。上一版报告说“不改 WESTPA 一行核心代码”、“不是我们臆测的可行性”是过于绝对的表述，这里更正：**只在上述前提满足、且我们不依赖超配（`len(subgroups) > target_count`）路径时**才成立；一旦触发超配路径，就必须靠适配器的前提保证或对 `_adjust_count` 的改写/打补丁来堵住，不能只靠配置 `subgroup_function`。

**结论**：两者都能实现，但 WESTPA 有现成的“子分组隔离 merge”接口，wepy 需要从零写整个 resample 逻辑（等价于重新实现一遍 `BinnedWE`）。

---

## 3. 能否使用 `derive_rng(IterKey)`，保证 resample 逐位可复现

这是两个引擎都没有为我们准备好的地方，但程度不同。

**wepy**：随机性分散在至少三处全局状态里：
- `wepy/walker.py:34` `import random as rand`；`keep_merge()`（同文件 113 行）用 `rand.choices(walkers, weights=weights)` 决定 merge 后保留哪个子历史——这是 Python **全局** `random` 模块，设计文档明令禁止。
- 内置 resampler（`resamplers/wexplore.py`、`resamplers/revo.py`）里也有 `import random as rand` 和裸的 `np.random.choice(...)`（`wexplore.py:1615`），同样是全局状态，且构造函数里的 `random_seed` 参数注释写着"If None the system (random) one will be used"（`revo.py:185`），不是按 key 派生。
- 内置 `OpenMMRunner.run_segment()`（`runners/openmm.py:448`）每次都 `new_integrator.setRandomNumberSeed(0)`——OpenMM 里 `0` 是特殊值，意味着"由 OpenMM 自己挑一个随机种子"，即**显式放弃确定性**，与 Task 12 "积分器种子由 `derive_rng(rng_key)` 派生"直接冲突。

如果我们完全不用 wepy 内置的 resampler 和 runner，只用它的 `Resampler`/`Runner` 抽象基类和 `sim_manager.Manager` 编排壳子，自己在自定义 `resample()`、自定义 `Runner.run_segment()` 里各自调用 `derive_rng(IterKey(...))`/`derive_rng(SegmentKey(...))`，是可以做到逐位可复现的——但前提是完全绕开上面三处全局状态（不能用 `keep_merge`、不能用内置 resampler、不能用内置 `OpenMMRunner`），等于只借用了 wepy 的类型骨架，没有复用它的算法或执行路径。

**WESTPA**：`WEDriver.rng`（`we_driver.py:122`）和 `WESimManager.rng`（`sim_manager.py:99`）都是 `numpy.random.Generator(MT19937())` 实例属性，**不是全局状态**，也没有散落在别处的裸随机调用——检索了 `core/` 下的随机性使用，只有这两处 `self.rng`。真正用到它的地方是 `_merge_walkers()`（`we_driver.py:483`）：

```python
iparent = np.digitize((self.rng.uniform(0, glom.weight),), cumul_weight)[0]
```

这是一次按权重加权的随机选择，决定 merge 后保留哪个子历史。因为 `self.rng` 是公开的实例属性，且构造时用的是默认无种子的 `Generator(MT19937())`（代码里没有暴露种子配置项，也没有在 checkpoint/restart 时保存/恢复 `rng` 的 `bit_generator.state`），要做到逐位可复现，需要**在每次迭代调用 `_run_we()`/`construct_next()` 之前，把 `driver.rng` 整个替换成 `derive_rng(IterKey(global_seed, run_id, iteration))` 派生出的新 `Generator`**。这是覆盖一个公开属性，不是打补丁改内部逻辑，风险比 wepy 那边小得多；但依赖的是"当前这份源码里随机性只经过这一个属性"这个事实，不是 WESTPA 公开承诺的 API 契约，跨版本可能失效，需要每次升级 WESTPA 时重新审查。

**结论**：WESTPA 的随机性集中在一个可覆盖的公开属性上，改造成本和风险都明显低于 wepy（wepy 的全局 `random`/`np.random` 调用分散在至少 3 个文件里，且内置 OpenMM runner 直接放弃确定性）。两者都不能"开箱即用"满足 `derive_rng`，都需要我们自己接管随机源。

---

## 4. 与 Task 12 OpenMM 后端对接的方式和每步开销

**wepy**：是一个进程内的 Python 库，`sim_manager.Manager` 直接在同一个 Python 进程里轮流调用 `Runner.run_segment()` 和 `Resampler.resample()`，没有强制的 IPC/文件层，`work_mapper.Mapper`（串行）足够我们保持单进程单 GPU context。但它内置的 `OpenMMRunner.run_segment()`（`runners/openmm.py:377-513`）**每次调用都重新构造一个 `openmm.app.Simulation`**（`simulation = omma.Simulation(self.topology, self.system, new_integrator, platform)`，第 503/510 行），也就是每个 walker 每一次 WE 迭代都重建一次 `Context`（重新上传体系、重新 JIT 编译 CUDA kernel），而不是复用一个持久 `Context` 只做 `setState`/`step`。这正是设计文档 §3.5 明令禁止的模式的一个变体——不是"每步回 Python"，而是"每个 WE 迭代重建一次 GPU 常驻对象"，对蛋白规模、迭代数动辄成百上千的 WE 跑法，这个固定开销会显著侵蚀 GPU 常驻传播本该有的收益。要避免这个问题，必须**不使用 wepy 内置的 `OpenMMRunner`**，而是自己写一个 `Runner` 子类，内部持有 Task 12 的 `OpenMMBackend`/`Propagator`（每个 walker 一个持久 `Context`，`run_segment` 只做 `set_state → run(n_steps) → get_state`），只借用 wepy 的 `Runner` 协议签名。

**WESTPA**：架构比 wepy 重得多——它是一个围绕 HDF5 数据仓库（`west.h5`）、`west.cfg` 配置、`System` 驱动类和 work manager（serial/processes/zeromq/mpi）的完整生产系统，传播器要么是外部可执行文件（`core/propagators/executable.py`，每段调用一次子进程），要么继承 `core/propagators/__init__.py` 里的 `WESTPropagator` 基类自己写 Python 实现（`propagate(segments)` 等方法，理论上可以在其中持有一个常驻 `OpenMMBackend`/`Propagator`，配合串行 work manager 保持单 GPU 进程）。可行，但比 wepy 多一层：每次迭代驱动主循环都会通过 `w_run` 走一遍完整的 bin 分配 → 传播 → HDF5 落盘的流程，落盘和数据仓库管理是 WESTPA 的核心卖点之一，但对我们这种"引擎已经有自己的 append-only 记录库（Task 2）"的项目来说是重复建设，需要要么放弃 WESTPA 自带的 HDF5 存储只用它的 `WEDriver`，要么接受两套记录系统并存。

**开销数量级**：没有对 Task 12 的 `OpenMMBackend` 做直接测试（Task 12 尚未完成，属于并行任务）。已确认的是 wepy 内置 runner 的架构性缺陷（每迭代重建 Context）和 WESTPA 的架构性重量（HDF5 落盘 + 独立数据仓库），这两点足以判断：**无论选哪个引擎，都不能直接用它们各自内置的 propagator/runner，必须让 Task 12 的 `OpenMMBackend` 挂在它们的 `Runner`/`WESTPropagator` 协议后面自己维护持久 `Context`**。

---

## 5. 与自研 `BinnedWE` 的对比（用例 10.4）

`BinnedWE` 还没有实现（正在另一个任务里构建），所以按 controller ruling R8 分两步处理：

### (a) 一旦 `BinnedWE` 存在，如何跑这个对比

**输入**（与用例 10.4 一致）：
- 体系：一维双阱解析势（`AnalyticBackend`/`DoubleWell2D` 退化到 1D 或专门的 1D 双阱），势垒高度 8 kT，Langevin 传播，`γ ≤ 0.1 ps⁻¹`。
- WE 配置：`target_per_bin`、bin 宽度（沿 x 的规则分箱，或退化情形不需要标签约束，`origin_label` 全部相同）、`tau_seg`、`n_iter`、`n_walkers` 初始值、稳态模式（`recycle_to` 非空，sink→source 回收）。
- 随机性：`global_seed` 取 5 个不同值，对应 5 次独立 WE 重复（设计 §4.5 的误差条要求）。
- 参考值：解析 MFPT 积分公式给出的精确通量 `1/MFPT`。

**跑法**：
1. 用 `BinnedWE`（Task 10 产物）跑 5 次独立重复，每次记录每次迭代的 `WERun.flux_to_sink`、`n_eff`、`n_walkers`、`weights_sum`，以及每次迭代的墙钟时间（resample 部分和传播部分分开计时）。
2. 用同一个 1D 双阱、同一个 `tau_seg`/`target_per_bin`，在 wepy 或 WESTPA 里各写一个"移植版"的标签约束 resampler（wepy：自定义 `Resampler` 子类；WESTPA：自定义 `subgroup_function` + 覆盖 `driver.rng`），也跑 5 次独立重复。
3. 传播段本身两边必须用同一个 `PotentialBackend`/`Propagator`（Task 3 的 `AnalyticBackend` 或 Task 12 的 `OpenMMBackend`），保证差异只来自 resample 逻辑和框架开销，不是传播器本身的差异。

**度量**（对齐用例 10.4 的验收 + 报告需要的信息）：
- **正确性**：5 次重复给出的通量的均值和 95% CI 是否落在解析 `1/MFPT` 附近（验收阈值同 10.4：95% CI 内）；`Σw` 守恒误差（应 < 1e-12）；`n_eff` 随迭代的轨迹是否合理（不应长期退化到 1）。
- **速度**：每次迭代的墙钟时间中位数、resample 部分单独的墙钟时间中位数（区分"框架开销" vs "传播时间"），换算成同样通量精度（例如 CI 宽度收窄到某个阈值）所需的总墙钟时间，才是公平的"速度"比较，不能只比单次迭代的 resample 耗时。
- **可复现性**：同一个 `global_seed`，两次独立跑，通量曲线和每步的 walker 数是否逐位相同（`BinnedWE` 应满足；外部引擎接上 `derive_rng` 后也应满足，需实测验证覆盖 `rng` 属性/自定义随机源是否真的截断了引擎自身残留的随机性，例如 wepy 的 `keep_merge` 是否被完全绕开）。

### (b) 15 分钟预算内的实测：wepy 自带 toy 的每次迭代开销

WESTPA 需要 `west.cfg` + `System` 驱动类 + work manager 才能跑起来（且额外缺 `pyzmq`、`blessings` 两个依赖），在 15 分钟预算内不可行，**没有跑**，原因是架构重量本身（见问题 4），不是许可证或技术障碍。

wepy 自带 `wepy.runners.randomwalk.RandomWalkRunner`（1D/N 维随机游走 toy）和 `wepy.resampling.resamplers.revo.REVOResampler`，两者都在 pip 包里，不需要额外依赖。写了一个不经过 `sim_manager.Manager`（避免额外编排开销）、直接循环调用 `runner.run_segment()` 和 `resampler.resample()` 的最小计时脚本（`N_WALKERS=48`，1 维，每段 10 步随机游走，200 次迭代），实测结果：

```
n_walkers=48 n_cycles=200 dim=1
mean segment-propagation time per cycle: 0.7555 ms
mean resample() time per cycle:          68.5749 ms  (min 29.5, max 147.3)
resample() overhead fraction of cycle:    98.9 %
sum(weight) after 200 cycles: 1.0（守恒，符合预期）
```

**解读**：这测的是 wepy **自己的** REVO resampler（全对全距离矩阵 + 方差优化选 clone/merge 候选，纯 Python/NumPy），不是我们会写的标签约束 `BinnedWE` 逻辑，数量级仅供参考。48 个 walker、1 维时，resample 本身平均约 69 ms/次，比 10 步 toy 传播（0.76 ms）贵接近 100 倍——如果传播段真的只有几毫秒（toy 体系），resample 开销会主导墙钟时间；但对真实 OpenMM 体系，`tau_seg` 通常对应秒到分钟级传播时间，这个量级的 resample 开销会降到可忽略。这也提示：`BinnedWE`（bin 查找 + 同标签内 split/merge，不需要全对全距离矩阵）预期比 REVO 便宜得多，但要用真实数字验收，仍需按 5(a) 的方案在 `BinnedWE` 完成后实测。

---

## 结论与建议

**建议：保持自研 `BinnedWE`，不在 Phase C 之前引入 wepy 或 WESTPA。**

理由：
1. 标签约束是本项目的核心特有需求，WESTPA 虽然有现成的 `subgroup_function` 机制天然契合，但 wepy 完全没有对应机制，两者都不能"配置一下就用"，最省事的也要写一个完整的自定义 resampler/子分组函数——工作量和自己实现 `BinnedWE` 的核心逻辑相当。
2. 两个引擎的随机性都不满足 `derive_rng` 契约：wepy 的全局 `random`/`np.random` 状态散落在 `walker.py`、`wexplore.py`、`revo.py`、`runners/openmm.py` 至少 4 个文件里，且内置 OpenMM runner **主动放弃**确定性（`setRandomNumberSeed(0)`）；WESTPA 的 `self.rng` 虽然可覆盖，但覆盖的正确性依赖于"当前版本随机性只经过这一个属性"这个未被文档承诺的事实。两者都需要绕开或替换引擎的默认执行路径才能达到设计文档 §7 的逐位可复现要求，而不是简单的适配层。
3. Task 12 的 `OpenMMBackend`（GPU 常驻 `Context`）如果接到 wepy 内置 `OpenMMRunner` 后面会被迫退化成"每次 WE 迭代重建一次 Context"，正是设计文档 §3.5 要避免的模式；接到 WESTPA 后面则要么放弃它自带的 HDF5 数据仓库（与我们已有的 Task 2 记录库重复），要么维护两套记录系统。两种情况都要求我们自己写 `Runner`/`WESTPropagator` 的适配层来保护 Task 12 的持久 `Context`，引擎本身在这一层反而是负担而不是帮助。
4. 用 wepy 自带 REVO resampler 在 toy 体系上实测，resample 本身的纯 Python 开销（48 walker、1 维，约 69 ms/次）不算小；`BinnedWE` 的分箱 + 同标签内 split/merge 逻辑比全对全距离矩阵的 REVO 简单得多，没有理由认为自研版本在速度上会吃亏。
5. 两个引擎都是 MIT 协议，`Resampler`/`Runner`（或 `subgroup_function`/`WESTPropagator`）接口都verified 可以在需要时再接，不存在"现在不选就以后用不了"的沉没成本问题。第 6 节的适配器草图说明了这条退路怎么留。

**什么时候值得重新评估**：如果 A2/A3（chignolin、encounter）阶段发现 `BinnedWE` 在高维进展坐标、REVO/MAB 这类自适应分箱上遇到明显的工程瓶颈（比如需要 Voronoi 树、并行子分组这类 WESTPA/wepy 已经踩过坑的基础设施），或者需要 WESTPA 成熟的多副本/多节点 work manager 来扩展到几十个并发 WE 迭代时，值得重新用本报告的第 6 节适配器把 WESTPA（优先于 wepy，因为它的 `subgroup_function` 和集中式 `rng` 更契合我们的约束）接进来评估实测吞吐和正确性,而不是现在就切换。

---

## 6. 适配器草图（接口层面，不含实现）

无论以后接哪一个，外部引擎都必须完全藏在 Task 10 的 `Resampler` Protocol 后面，`run_we()`/估计量代码不能感知底层是 `BinnedWE` 还是外部引擎：

```python
class ExternalWEAdapter(Resampler):
    """Wraps an external WE engine's native resampler/driver behind our
    Resampler protocol. Owns the translation between our Walker/IterKey
    and the external engine's native walker/segment objects; the rest of
    cytherea never imports the external engine directly.
    """
    kind: Literal["we"] = "we"

    def __init__(
        self,
        bin_of: Callable[[np.ndarray], int],
        target_per_bin: int,
        allow_cross_label_merge: bool = False,
        min_weight: float = 1e-250,
        # native engine handle is constructed here, not passed in, so that
        # nothing outside this module holds a reference to wepy/WESTPA types
    ): ...

    def resample(self, walkers: list[Walker], it: int, key: IterKey) -> list[Walker]:
        # 1. translate our Walker list -> native walker/segment objects
        #    (carrying origin_label as either a WalkerState field [wepy]
        #    or via the subgroup_function closure [WESTPA])
        # 2. override the engine's RNG entry point with
        #    derive_rng(key, substream="we_resample") BEFORE calling into
        #    the engine (wepy: pass our own Generator into our custom
        #    Resampler subclass; WESTPA: driver.rng = derive_rng(key, ...))
        # 3. invoke the engine's native resample/_run_we with our
        #    subgroup/label logic (custom Resampler subclass for wepy,
        #    custom subgroup_function for WESTPA) -- never the engine's
        #    built-in REVO/WExplore/default binning, since those don't
        #    know about origin_label and (for wepy) touch global random
        # 4. translate native walkers back to our Walker list, re-deriving
        #    segment_key/parent from the engine's lineage records
        # 5. assert conservation: abs(sum(w.weight for w in out) - 1.0) < 1e-12
        #    before returning (same invariant BinnedWE must satisfy)
        # 6. assert label isolation: for every merged walker in the output,
        #    all of its pre-merge parents (from the engine's own lineage
        #    records) must share one origin_label -- fail loudly (raise,
        #    do not warn-and-continue) if not. This is not redundant with
        #    step 5: WESTPA's own subgroup_function isolation silently
        #    breaks when a bin holds more distinct origin_labels than its
        #    bin_target_counts (we_driver.py:558-561, reached whenever
        #    len(subgroups) > target_count via :677/:691-692), pooling all
        #    subgroups into one before picking merge pairs -- see the
        #    section 2 caveat above. The adapter must catch that case itself; the engine
        #    will not.
        ...
```

关键约束（无论选哪个引擎都要满足）：
- 适配器是唯一导入 `wepy`/`westpa` 的模块；`pyproject.toml` 里把它列为可选依赖（extra），`resampler.kind="we"` 时才需要装。
- 适配器自己实现标签约束（wepy：自定义 `Resampler` 子类 + 自定义 `WalkerState`；WESTPA：自定义 `subgroup_function`），不使用引擎内置的 REVO/WExplore/默认分箱。
- 适配器在每次调用引擎的 resample 入口之前，必须把引擎的随机源替换/传入为 `derive_rng(IterKey(...))` 派生的 `Generator`，且必须有一条测试（对应用例 10.3 的逐位可复现要求）验证这一步真的截断了引擎自身的残留随机性。
- 底层传播仍然用 Task 12 的 `OpenMMBackend`/`Propagator`（持久 `Context`），适配器只把它包成引擎要求的 `Runner`/`WESTPropagator` 形状，不能让引擎重建 `Context`。
- **（新增，2026-10 复审）** 若选 WESTPA：适配器必须要么在构造 `bin_target_counts` 时保证每个 bin 的 target 名额 ≥ 该 bin 内出现的 distinct `origin_label` 数，要么对 `_adjust_count`（`we_driver.py:555-601`）打补丁/整体重写，防止 `len(subgroups) > target_count` 时的池化分支跨标签 merge；无论选了哪条路，适配器都必须落地上面 `resample()` 步骤 6 的事后检查（合并后逐个核对 parent 的 `origin_label` 是否一致），作为不依赖引擎内部实现细节的最后一道防线。
- `Σw` 守恒断言、`min_weight` 强制 merge（用例 10.7）由适配器在把结果交还给 `run_we()` 之前自己检查，不依赖引擎自己的守恒实现是否精确到 1e-12。

---

## Scratch 清理

评估用的 scratch 目录 `/tmp/claude-1000/-home-ruigengji-venus96/98bd9a0b-e9ea-41c8-8a0e-06287606e8dc/scratchpad/we_eval`（`wepy_src/`、`westpa_src/` 只读克隆，`site/` 里 `pip install --no-deps --target` 装的 wepy 1.2 及其 4 个缺失依赖，`toy_timing.py` 计时脚本）已在完成本报告后删除；`openmm_dev` 环境全程未被改动，没有新增任何已装包。
