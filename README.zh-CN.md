# Cytherea

English | [简体中文](README.zh-CN.md)

**面向溶液蛋白的瞄准发射 + 加权集合动力学框架,基于 OpenMM。**

Cytherea 把 [VENUS96](https://doi.org/10.1016/0010-4655(96)00042-4)(Hase 组的经典化学动力学轨迹程序)的化学动力学思想重建到溶液相生物分子模拟上:从 milestone 区域出发做 aimed shooting(瞄准发射),以短轨迹集合的方式传播,再转成带校准不确定度的转移矩阵、committor 与速率。名字取自 Cythera——阿佛洛狄忒(维纳斯)的别称。

> **状态:**研究代码,Phase A(`0.1.0`)。接口可能随时调整。

## 里面有什么

- **构造上确定**——每发 shot、每个 segment、每轮 WE 迭代都从分层 key 派生自己的 RNG 子流;续算由 config、协议与代码身份 hash(包源码 + OpenMM System/Topology 摘要)三重把关。
- **两套后端,一份契约**——解析势与传播器(Euler–Maruyama、BAOAB、Verlet、过阻尼)和 OpenMM 共享同一协议:on-step 速度、`energy_forces(x, box)`、每 `dt_obs` 回读观测,`NaN` 翻译成 `NumericalInstabilityError` 而不是静默污染。
- **PES 一致性套件**——NVE 总能量漂移、积分器不变量、有限差分力检查带截断跨越守卫(抽样点会跨过非键截断的坐标跳过并报告数目),分 strict 与 sampled(按原子组分层的抽样)两种模式。
- **系综初条件**——`EnsembleFramePool` 带结构门禁(能量窗、最小原子间距、COM 动量、温度自由度)与约束投影。
- **不确定度校准过的估计量**——簇 bootstrap 转移矩阵、状态固定分层 bootstrap、committor 与 k_on(n_eff = min(Korn–Graubard, Kish)),全局零中心 bootstrap Chapman–Kolmogorov 检验(稀疏 Markov 链上不误报),以及直接在定长射击数据上做 k·τ 检验的 shot 版 CK(`ck_test_shots`,支持仅 τ 的短 shot)。
- **自研加权集合**——`BinnedWE`,带标签约束、按权重 recycle、segment 谱系可精确离线回放。
- **milestone 吸收网络**——core-set milestoning、吸收概率,以及按 (origin label, milestone) 分层的 Markov 检验。
- **配置 + CLI**——pydantic schema 带量纲检查的单位解析,`cytherea run / resume / report`,sidecar config hash 续算把关。

## 安装

Python ≥ 3.10。

```bash
git clone https://github.com/Cedrus810/cytherea.git
cd cytherea
pip install -e .
```

核心依赖(`numpy`、`scipy`、`deeptime`、`pydantic`、`pyyaml`)自动安装。OpenMM 后端另需 [OpenMM](https://openmm.org),建议用 conda-forge 装进同一环境。

## 快速上手

解析 1D 双阱——从固定起点算 committor,每帧 200 发,CPU 上几秒跑完:

```bash
cytherea run examples/toy_doublewell/config.yaml
```

`examples/` 里还有加权集合与 milestone 网络的 toy 例子、丙氨酸二肽显式溶剂参考与射击管线(OpenMM)、barnase–barstar 结合对的 encounter 采样。

## 仓库地图

| 路径 | 内容 |
|---|---|
| `src/cytherea/` | 库本体:keys、store、backends、engine、ic、estimate、network、resample、config、cli |
| `examples/` | 可运行算例,从 1D toy 到显式溶剂多肽 |
| `docs/design/` | 设计文档(VENUS96 → Aβ42,v2 为准) |
| `docs/reports/` | 验收报告(A0 toy、A1 丙氨酸二肽、A3 encounter pilot)与审查交接 |
| `docs/STATUS.md` | 活页状态与交接记录 |
| `CHANGELOG.md` | 变更日志 |

## 血统

架构上有意重建 VENUS96——把经典轨迹变成动力学——用于溶液中的蛋白质。未使用、未包含任何 VENUS96 或 VENUSpy 源码;Cytherea 是独立实现。

## 许可证

[MIT](LICENSE)
