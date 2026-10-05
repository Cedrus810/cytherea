# Cytherea

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語

**OpenMM ベースの、溶液中タンパク質を対象とした aimed shooting + weighted ensemble 動力学フレームワーク。**

Cytherea は、[VENUS96](https://doi.org/10.1016/0010-4655(96)00042-4)（Hase グループの古典化学動力学トラジェクトリープログラム）の化学動力学の考えを、溶液相の生体分子シミュレーションへと再構築したものです。milestone 領域から aimed shooting で出発し、短いトラジェクトリのアンサンブルとして伝播させ、較正された不確かさ付きの遷移行列・committor・速度定数へと変換します。名前は Cythera——アプロディーテー（ヴィーナス）の別称に由来します。

> **ステータス:** 研究用コード、Phase A（`0.1.0`）。インターフェースは予告なく変更されることがあります。

## 実装しているもの

- **構造的に決定論的** — すべての shot・segment・weighted ensemble 反復が、階層的 key から専用の RNG サブストリームを導出します。再開は config・プロトコル・コード ID（パッケージソース + OpenMM System/Topology ダイジェスト）の三重ハッシュで保護されます。
- **2 つのバックエンド、1 つの契約** — 解析ポテンシャルと伝播器（Euler–Maruyama、BAOAB、Verlet、過減衰）と OpenMM が同一のプロトコルを共有します：on-step 速度、`energy_forces(x, box)`、`dt_obs` ごとの観測、`NaN` は黙って汚染するのではなく `NumericalInstabilityError` に変換。
- **PES 一貫性スイート** — NVE 全エネルギードリフト、積分器の不変量、有限差分力チェック（カットオフ横断ガード付き：非結合カットオフをまたぎうる座標はスキップして報告）。strict モードと sampled モード（原子群で層化）の 2 種類。
- **アンサンブル初期条件** — `EnsembleFramePool`。構造ゲート（エネルギー窓、最小原子間距離、COM 運動量、温度自由度）と拘束射影付き。
- **不確かさを較正した推定量** — クラスタ bootstrap 遷移行列、状態固定階層 bootstrap、n_eff = min(Korn–Graubard, Kish) による committor と k_on、疎な Markov 連鎖でも較正が保たれる大域的零中心 bootstrap Chapman–Kolmogorov 検定、さらに固定 lag の shooting データそのものから k·τ を検定する shot 版 CK（`ck_test_shots`、τ のみの短 shot にも対応）。
- **自前の weighted ensemble** — ラベル制約、重みを考慮した recycle、segment の系譜を正確にオフライン再生できる `BinnedWE`。
- **milestone 吸収ネットワーク** — core-set milestoning、吸収確率、(origin label, milestone) で層化した Markov 検定。
- **config + CLI** — 次元チェック付きの単位解析を持つ pydantic スキーマ、`cytherea run / resume / report`、サイドカー config ハッシュによる再開保護。

## インストール

Python ≥ 3.10。

```bash
git clone https://github.com/Cedrus810/cytherea.git
cd cytherea
pip install -e .
```

主要な依存パッケージ（`numpy`、`scipy`、`deeptime`、`pydantic`、`pyyaml`）は自動でインストールされます。OpenMM バックエンドには別途 [OpenMM](https://openmm.org) が必要です。同じ環境に conda-forge で入れるのがおすすめです。

## クイックスタート

解析的な 1 次元二重井戸 — 固定した始点から committor を計算。フレームごとに 200 shot、CPU 上で数秒で終わります：

```bash
cytherea run examples/toy_doublewell/config.yaml
```

`examples/` にはこのほか、weighted ensemble と milestone ネットワークの toy 例、アラニンジペプチドの明示溶媒 reference・shooting パイプライン（OpenMM）、barnase–barstar 結合対の encounter サンプリングがあります。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `src/cytherea/` | ライブラリ本体：keys、store、backends、engine、ic、estimate、network、resample、config、cli |
| `examples/` | 実行可能な計算例：1D toy から溶媒和ペプチドまで |
| `docs/design/` | 設計ドキュメント（VENUS96 → Aβ42、v2 が正） |
| `docs/reports/` | 受け入れ検証レポート（A0 toy、A1 アラニンジペプチド、A3 encounter pilot）とレビュー引き継ぎ |
| `results/` | `docs/reports/` を裏付ける小型の機械可読成果物（分析 JSON など） |
| `docs/STATUS.md` | 随時更新される現状と引き継ぎメモ |
| `CHANGELOG.md` | 変更履歴 |

## ライセンス

[MIT](LICENSE)
