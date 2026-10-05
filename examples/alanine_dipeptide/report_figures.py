#!/usr/bin/env python
"""Figures of docs/reports/A1.md (frames, T(tau), CK, thermostat) and of the A3 pilot
report (r(t) of every shot). Inputs are local copies of the analysis outputs
(files written on other hosts: copy them with `dd iflag=direct` first, see
docs/STATUS.md); every figure goes to docs/reports/figures/.

  python examples/alanine_dipeptide/report_figures.py --a1 DIR --thermo DIR --a3-store FILE

DIR for --a1 holds analysis_14b.json and paired_platform.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ala2_common as C  # noqa: E402

OUT = Path("docs/reports/figures")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
STATE_COLOR = [BLUE, ORANGE, AQUA]
STATE_MARK = ["o", "^", "s"]
LABEL = ["C7eq/C5", r"$\alpha_R$", r"$\alpha_L$"]


def style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def axes_style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_frames():
    """Reference Ramachandran density, core boxes, the 150 start frames."""
    meta = json.load(open("runs/ala2_shoot/frames/frames.json"))
    phi, psi = [], []
    for run in meta["runs"]:
        rec = C.load_phipsi(Path(run) / "phipsi.bin")[int(meta["skip_ps"]):]
        phi.append(rec["phi"])
        psi.append(rec["psi"])
    phi, psi = np.concatenate(phi), np.concatenate(psi)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=SURF)
    axes_style(ax)
    h, xe, ye = np.histogram2d(phi, psi, bins=180, range=[[-180, 180], [-180, 180]])
    ax.pcolormesh(xe, ye, np.log10(h.T + 1), cmap="Greys", vmin=0, vmax=np.log10(h.max() + 1) * 1.15, rasterized=True)
    for s, name in enumerate(C.STATE_NAMES):
        box = C.CORE_BOXES_DEG[name]
        for p0, p1 in box["phi"]:
            for q0, q1 in box["psi"]:
                ax.add_patch(Rectangle((p0, q0), p1 - p0, q1 - q0, fill=False, ec=STATE_COLOR[s], lw=1.6))
    for s in range(3):
        fs = [f for f in meta["frames"] if f["state"] == s]
        sp = meta["spread"][C.STATE_NAMES[s]]
        ax.scatter([f["phi"] for f in fs], [f["psi"] for f in fs], s=26, marker=STATE_MARK[s], color=STATE_COLOR[s],
                   edgecolor=SURF, lw=0.7, zorder=3,
                   label=f"{LABEL[s]}: {sp['n_frames']} frames, {sp['n_runs']} runs, {sp['n_visits']} core visits")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel(r"$\phi$ (deg)"); ax.set_ylabel(r"$\psi$ (deg)")
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID)
    ax.set_title(f"reference 947 ns (grey, log density), core boxes and the 150 start frames",
                 color=INK, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "A1_frames.png", dpi=150, facecolor=SURF)


def fig_T(a1: Path):
    """T(tau) element by element: reference, all shots, GPU long shots' first tau, CPU tau shots."""
    r = json.load(open(a1 / "analysis_14b.json"))
    pp = json.load(open(a1 / "paired_platform.json"))
    ref = r["reference"]
    series = [("reference (947 ns)", ref["T_ref"], ref["T_ref_ci95_lo"], ref["T_ref_ci95_hi"], INK2, "D"),
              ("shots, all 1500", r["T"]["matrix"], r["T"]["ci95_lo"], r["T"]["ci95_hi"], BLUE, "o"),
              ("GPU: long shots, first tau (600)", *(r["T_parts"]["long_first_tau"][k] for k in ("matrix", "ci95_lo", "ci95_hi")), ORANGE, "^"),
              ("CPU: tau shots (900)", *(r["T_parts"]["tau_only"][k] for k in ("matrix", "ci95_lo", "ci95_hi")), AQUA, "s")]
    fig, axs = plt.subplots(3, 3, figsize=(11, 7.2), facecolor=SURF)
    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            axes_style(ax)
            ax.grid(axis="y", visible=False)
            for k, (name, T, lo, hi, col, mk) in enumerate(series):
                v, a, b = T[i][j], lo[i][j], hi[i][j]
                ax.errorbar([v], [3 - k], xerr=[[v - a], [b - v]], fmt=mk, ms=6, color=col, mec=SURF, mew=0.7,
                            elinewidth=1.6, capsize=0, label=name)
            ax.set_yticks([]); ax.set_ylim(-0.6, 3.6)
            p = pp["elements"][f"{C.STATE_NAMES[i]}->{C.STATE_NAMES[j]}"]["p"]
            ax.set_title(f"{LABEL[i]} $\\rightarrow$ {LABEL[j]}   (GPU vs CPU, paired p = {p:.2g})",
                         color=INK, fontsize=9, loc="left")
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"A1 T($\\tau$ = 100 ps), row-normalised core-start, 95% intervals; "
                 f"GPU vs CPU paired by frame: global p = {pp['global_p']:.2f}",
                 color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUT / "A1_T.png", dpi=150, facecolor=SURF)


def fig_ck(a1: Path):
    """14.3: T(tau)^k from all 1500 shots vs T(k tau) measured on the 600 long shots and the reference T(k tau)."""
    from scipy import stats

    r = json.load(open(a1 / "analysis_14b.json"))
    ref = json.load(open("runs/ala2_par/analysis/analysis.json"))["core_start"]["ck_test"]
    rows = r["ck"]["rows"]
    k = np.array([x["k"] for x in rows])
    pred = np.array([x["predicted_stay"] for x in rows])
    est = np.array([x["estimated_stay"] for x in rows])
    rk = np.array([x["k"] for x in ref["rows"]])
    rest = np.array([x["estimated_stay"] for x in ref["rows"]])
    n_long = r["n_long_shots"] // 3
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURF, sharex=True)
    for s, ax in enumerate(axs):
        axes_style(ax)
        ax.plot(rk * 0.1, rest[:, s], color=INK2, lw=2, label="reference T(k$\\tau$), 947 ns", zorder=2)
        ax.plot(k * 0.1, pred[:, s], color=BLUE, lw=2, label=f"shots: T($\\tau$)$^k$, T($\\tau$) from all {r['n_shots']} shots", zorder=3)
        m = k > 1
        x = est[m, s] * n_long
        lo = stats.beta.ppf(0.025, x + 0.5, n_long - x + 0.5)
        hi = stats.beta.ppf(0.975, x + 0.5, n_long - x + 0.5)
        ax.errorbar(k[m] * 0.1, est[m, s], yerr=[est[m, s] - lo, hi - est[m, s]], fmt="o", ms=5, color=ORANGE, mec=SURF,
                    mew=0.8, elinewidth=1.2, capsize=0,
                    label=f"shots: T(k$\\tau$) measured on the {r['n_long_shots']} long shots (Jeffreys 95%, per shot)", zorder=4)
        ax.set_title(f"stay in {LABEL[s]}", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("lag k$\\tau$ (ns)")
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("probability to be in the start state")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    verdict = "passed" if r["ck"]["passed"] else "failed"
    fig.suptitle(f"A1 14.3 Chapman-Kolmogorov from the shots ($\\tau$ = 100 ps, k = 1-55): {verdict}, "
                 f"D = {r['ck']['max_dev']:.3f}", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUT / "A1_ck.png", dpi=150, facecolor=SURF)


def fig_thermo(tdir: Path):
    """Mean kinetic temperature per run (after 20 ps) with block errors, three conditions."""
    conds = [("CUDA mixed, 2 fs", "cuda_[0-3].json", BLUE), ("CPU, 2 fs", "cpu_[0-3].json", AQUA),
             ("CUDA mixed, 1 fs", "cuda_dt1_[0-3].json", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURF)
    axes_style(ax)
    ax.axhline(300.0, color=INK2, lw=1.2, ls="--", zorder=1)
    ax.text(2.45, 300.15, "300 K", color=INK2, fontsize=8, ha="right", va="bottom")
    for c, (name, pat, col) in enumerate(conds):
        means, sems = [], []
        for f in sorted(glob.glob(str(tdir / pat))):
            T = np.array(json.load(open(f))["T"])[40:]
            nb = T.size // 20
            blocks = T[: nb * 20].reshape(nb, 20).mean(1)
            means.append(T.mean())
            sems.append(blocks.std(ddof=1) / np.sqrt(nb))
        xs = c + np.linspace(-0.18, 0.18, len(means))
        ax.errorbar(xs, means, yerr=sems, fmt="o", ms=5, color=col, mec=SURF, mew=0.7, elinewidth=1.2, capsize=0, zorder=3)
        m, e = np.mean(means), np.sqrt(np.sum(np.square(sems))) / len(sems)
        ax.hlines(m, c - 0.3, c + 0.3, color=col, lw=2.5, zorder=2)
        ax.text(c + 0.32, m, f"{m:.1f} $\\pm$ {e:.1f} K", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), [c[0] for c in conds])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("mean kinetic temperature (K)")
    ax.set_title("A1 propagator, 4 runs per condition (bars: 10 ps block errors); "
                 "the 2 fs deficit shrinks ~dt$^2$", color=INK, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "A1_thermo.png", dpi=150, facecolor=SURF)


def fig_a3(store: Path):
    """r(t) of the 30 pilot shots by outcome, and the timeouts' last 20 ns (r and Q)."""
    c = sqlite3.connect(f"file:{store}?mode=ro&immutable=1", uri=True)
    recs = sorted((json.loads(p) for (p,) in c.execute("select payload from records")), key=lambda p: p["key"]["shot_id"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (a1, a2):
        axes_style(ax)
    for p in recs:
        o = p["observables"]
        t, r = np.asarray(o["t"]) / 1000.0, np.asarray(o["r"])
        if p["stop_reason"] == "escape":
            a1.plot(t, r, color=BLUE, lw=0.8, alpha=0.55, zorder=2)
        else:
            a1.plot(t, r, color=ORANGE, lw=1.0, zorder=3)
    for y, name in ((15.0, "q2 = 15 nm (escape)"), (10.0, "q1 = 10 nm (offline replay)"), (5.0, "b = 5 nm (start)"),
                    (2.35, "native complex COM 2.35 nm")):
        a1.axhline(y, color=INK2, lw=1, ls="--", zorder=1)
        a1.text(50.5, y, name, va="center", fontsize=8, color=INK2)
    n_esc = sum(p["stop_reason"] == "escape" for p in recs)
    a1.plot([], [], color=BLUE, lw=1.5, label=f"escape ({n_esc})")
    a1.plot([], [], color=ORANGE, lw=1.5, label=f"timeout at 50 ns ({len(recs) - n_esc})")
    a1.legend(loc="center right", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9)
    a1.set_xlim(0, 50); a1.set_ylim(1.5, 16.5)
    a1.set_xlabel("t (ns)"); a1.set_ylabel("COM distance r (nm)")
    a1.set_title("A3 pilot, 30 shots from the b sphere: no reaction (Q $\\geq$ 0.3 for 20 ps)", color=INK,
                 fontsize=10, loc="left")
    touts = []
    for p in recs:
        if p["stop_reason"] != "timeout":
            continue
        o = p["observables"]
        t, r, q = np.asarray(o["t"]) / 1000.0, np.asarray(o["r"]), np.asarray(o["Q"])
        m = t >= 30.0
        touts.append((r[m].mean(), r[m].min(), r[m].max(), q.max(), p["key"]["shot_id"]))
    touts.sort()
    for y, (mean, lo, hi, qmax, sid) in enumerate(touts):
        a2.hlines(y, lo, hi, color=ORANGE, lw=6, alpha=0.45, zorder=2)
        a2.scatter([mean], [y], s=40, color=ORANGE, edgecolor=SURF, lw=0.8, zorder=3)
        a2.text(hi + 0.04, y, f"Q$_{{max}}$ = {qmax:.2f}", va="center", fontsize=8, color=INK2)
    a2.axvline(2.35, color=INK2, lw=1, ls="--")
    a2.text(2.37, len(touts) - 0.4, "native\nCOM", fontsize=8, color=INK2, va="top")
    a2.set_yticks(range(len(touts)), [f"shot {t[4]}" for t in touts])
    a2.set_ylim(-0.7, len(touts) - 0.3)
    a2.set_xlim(1.9, 4.0)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("r over the last 20 ns (dot: mean, bar: range; nm)")
    a2.set_title("the 7 timeouts: compact, Q far below 0.3", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "A3_pilot.png", dpi=150, facecolor=SURF)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--a1", type=Path, required=True)
    p.add_argument("--thermo", type=Path, required=True)
    p.add_argument("--a3-store", type=Path, required=True)
    a = p.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_frames()
    fig_T(a.a1)
    fig_ck(a.a1)
    fig_thermo(a.thermo)
    fig_a3(a.a3_store)


if __name__ == "__main__":
    main()
