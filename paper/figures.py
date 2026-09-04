"""Figures for the CMS paper. Every value is read from a stored artefact.

Inputs (all produced by the pipeline, nothing hand-entered):
    data/raw/dataset_<sys>.csv          generate_data.py
    data/raw/<sys>_probe.json           probe_eligible.py
    models/results_heads_<sys>.json     train_mlp.py / rerun_heads.py
    models/results_baselines_<sys>.json train_baselines.py
    models/results_a23.json             train_mlp.py (fecrni extras)
    models/width{64,128,256}.json       train_mlp.py --hidden
    models/loss_ablation_{mse,mae}.json train_mlp.py --loss
    models/residue_sweep_<sys>.json     train_mlp.py --residue-sweep
    models/block_holdout_<sys>.json   holdout_eval.py
    models/pred_<sys>_<tag>_s<seed>.npz train_mlp.py / train_baselines.py

Usage: py -3.12 paper/figures.py
"""
import json
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")
D = os.path.join(ROOT, "data", "raw")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

CM = 1 / 2.54
COLW, FULLW = 8.4 * CM, 17.4 * CM
SEEDS = [42, 123, 2024]
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
LBL = {"fecrni": "Fe-Cr-Ni", "fecrmn": "Fe-Cr-Mn", "fecrmo": "Fe-Cr-Mo",
       "fecrv": "Fe-Cr-V", "femnni": "Fe-Mn-Ni"}
X3 = {"fecrni": "Ni", "fecrmn": "Mn", "fecrmo": "Mo", "fecrv": "V", "femnni": "Ni"}

HEADS = [("mlp_sigmoid", "sigmoid"), ("mlp_residue", "residue"),
         ("mlp_sparsemax", "sparsemax"), ("mlp_softmax", "softmax"),
         ("mlp_sig_norm", "sig-norm"), ("mlp_renorm", "renorm")]
BASES = [("ridge_renorm", "ridge"), ("knn_renorm", "$k$-NN"),
         ("xgb_renorm", "XGBoost"), ("rf_renorm", "random forest")]

C = {"sigmoid": "#8c8c8c", "residue": "#c44e52", "sparsemax": "#dd8452",
     "softmax": "#4c72b0", "sig-norm": "#55a868", "renorm": "#2a6f3f",
     "ridge": "#b8b8b8", "$k$-NN": "#9d7fb0", "XGBoost": "#d9a03c",
     "random forest": "#7a4f9c", "mlp": "#2a6f3f", "rf": "#7a4f9c"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "lines.linewidth": 1.0, "savefig.dpi": 400, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def colorbar(fig, mappable, ax, label):
    """Colorbar tagged so the figure audit does not treat it as a data panel."""
    cb = fig.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(label, fontsize=6.6)
    cb.ax.tick_params(labelsize=6.2)
    cb.ax._is_colorbar = True
    return cb


def legend_below(ax, ncol, y=-0.30, fontsize=6.4):
    """Legend under the axes: never collides with data, never clipped."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     fontsize=fontsize, columnspacing=0.9, handlelength=1.1,
                     handletextpad=0.35, borderpad=0.3, frameon=False)


def guide(artist):
    """Mark a reference line/band as part of the coordinate frame, not data.

    The figure audit checks that text never lands on data. Reference guides
    span a whole axis by construction, so counting them as data would flag
    every in-panel annotation and push correctly placed labels somewhere
    worse.
    """
    artist._is_guide = True
    return artist


def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left")


def results(system):
    out = {}
    for fn in [f"results_heads_{system}.json", f"results_baselines_{system}.json"]:
        p = os.path.join(M, fn)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    if system == "fecrni":
        p = os.path.join(M, "results_a23.json")
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def agg(d, prefix, field="mean_mae"):
    v = [d[f"{prefix}_s{s}"][field] for s in SEEDS if f"{prefix}_s{s}" in d]
    return (float(np.mean(v)), float(np.std(v)), v) if v else (np.nan, np.nan, [])


def phases(system):
    probe = json.load(open(os.path.join(D, f"{system}_probe.json")))
    names = sorted(probe["active_counts"].keys())
    return names, [f"NP_{p}" for p in names]


def pred(system, tag, seed=42):
    for suffix in ("192x192x192_huber", "std"):
        p = os.path.join(M, f"pred_{system}_{tag}_{suffix}_s{seed}.npz")
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"pred_{system}_{tag}_*_s{seed}.npz")


# --------------------------------------------------------------------------
def fig1_dataset():
    """Design space, sampling strategies, phase statistics, target simplex."""
    fig = plt.figure(figsize=(FULLW, 0.52 * FULLW))
    gs = fig.add_gridspec(2, 3)

    df = pd.read_csv(os.path.join(D, "dataset_fecrni.csv"))
    ax = fig.add_subplot(gs[0, 0])
    # Gibbs triangle in (Cr, X) coordinates; Fe = 1 - Cr - X
    sc = ax.scatter(df["Cr"], df["Ni"], c=df["T"], s=0.7, cmap="viridis",
                    linewidths=0, rasterized=True)
    guide(ax.plot([0, 1], [1, 0], color="k", lw=0.6)[0])
    colorbar(fig, sc, ax, "$T$ (K)")
    ax.set_xlabel("$x_{\\mathrm{Cr}}$")
    ax.set_ylabel("$x_{\\mathrm{Ni}}$")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    panel(ax, "a", "Fe-Cr-Ni design points")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df["T"], bins=52, color="#4c72b0", edgecolor="none")
    guide(ax.axvspan(1300, 2000, color="#dd8452", alpha=0.16, lw=0))
    ax.set_xlabel("$T$ (K)")
    ax.set_ylabel("design points")
    panel(ax, "b", "temperature density")

    ax = fig.add_subplot(gs[0, 2])
    box = df["in_stainless_box"].astype(bool)
    ax.scatter(df.loc[~box, "Cr"], df.loc[~box, "T"], s=0.7, c="#c9c9c9",
               linewidths=0, rasterized=True, label="full simplex")
    ax.scatter(df.loc[box, "Cr"], df.loc[box, "T"], s=0.7, c="#2a6f3f",
               linewidths=0, rasterized=True, label="stainless box")
    ax.set_xlabel("$x_{\\mathrm{Cr}}$")
    ax.set_ylabel("$T$ (K)")
    legend_below(ax, 2, y=-0.22)
    for h in ax.get_legend().legend_handles:
        h.set_sizes([9])
    panel(ax, "c", f"stainless subset ({100 * box.mean():.0f}% of rows)")

    # phase occurrence per system
    ax = fig.add_subplot(gs[1, :2])
    xpos, labels, colors = [], [], []
    vals = []
    cyc = plt.get_cmap("tab20").colors
    pos = 0.0
    for i, s in enumerate(SYS):
        names, cols = phases(s)
        d = pd.read_csv(os.path.join(D, f"dataset_{s}.csv"))
        for j, (nm, c) in enumerate(zip(names, cols)):
            vals.append(100 * float((d[c] > 1e-3).mean()))
            xpos.append(pos)
            labels.append(nm.replace("_PHASE", "").replace("_A12", "")
                          .replace("_A1", "").replace("_A2", ""))
            colors.append(cyc[j % 20])
            pos += 1.0
        pos += 1.2
    ax.bar(xpos, vals, width=0.86, color=colors, edgecolor="none")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.0)
    # One label per phase is the content of this panel, not tick crowding:
    # 31 rotated 6 pt labels over ~11.6 cm leaves ~3.7 mm each.
    ax._dense_ticks_ok = True
    ax.set_ylabel("rows with $N^\\phi > 10^{-3}$ (%)")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 400)
    # system separators and labels, inside the axes above the bars
    start = 0.0
    for s in SYS:
        n = len(phases(s)[0])
        ax.text(start + (n - 1) / 2, 190, LBL[s], ha="center", fontsize=6.6)
        start += n + 1.2
    panel(ax, "d", "phase occurrence (log scale)")

    ax = fig.add_subplot(gs[1, 2])
    devs = []
    for s in SYS:
        _, cols = phases(s)
        d = pd.read_csv(os.path.join(D, f"dataset_{s}.csv"))
        devs.append(np.abs(d[cols].values.sum(axis=1) - 1.0))
    ax.boxplot([np.log10(np.clip(x, 1e-18, None)) for x in devs],
               tick_labels=[LBL[s].replace("Fe-", "") for s in SYS],
               widths=0.6, flierprops=dict(ms=1.2, alpha=0.35))
    guide(ax.axhline(np.log10(1e-6), color="#c44e52", ls="--", lw=0.8))
    ax.set_ylabel("$\\log_{10}|\\sum_k y_k - 1|$")
    ax.set_ylim(-18, -4)
    ax.annotate("acceptance $10^{-6}$", xy=(0.5, -5.2),
                xycoords=("axes fraction", "data"), ha="center",
                fontsize=6.2, color="#c44e52")
    ax.tick_params(axis="x", labelsize=6.2, rotation=45)
    panel(ax, "e", "target simplex closure")

    save(fig, "fig1_dataset")


# --------------------------------------------------------------------------
def fig2_heads():
    """Main comparison: MAE by head and system, plus the constraint gain."""
    fig = plt.figure(figsize=(FULLW, 0.44 * FULLW))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    n = len(HEADS)
    w = 0.8 / n
    for i, (key, lab) in enumerate(HEADS):
        m = [agg(results(s), key)[0] for s in SYS]
        e = [agg(results(s), key)[1] for s in SYS]
        ax.bar(np.arange(len(SYS)) + (i - (n - 1) / 2) * w, m, width=w,
               yerr=e, capsize=1.4, color=C[lab], label=lab,
               error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(SYS)))
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=20, fontsize=6.8)
    ax.set_ylabel("test MAE")
    ax.set_ylim(0, 0.075)
    legend_below(ax, 3, y=-0.20)
    panel(ax, "a", "neural heads (mean $\\pm$ s.d., 3 seeds)")

    ax = fig.add_subplot(gs[0, 1])
    best_mlp, rf = [], []
    for s in SYS:
        d = results(s)
        cand = [agg(d, k)[0] for k in ["mlp_renorm", "mlp_sig_norm", "mlp_softmax"]]
        best_mlp.append(min(cand))
        rf.append(agg(d, "rf_renorm")[0])
    xs = np.arange(len(SYS))
    ax.bar(xs - 0.2, best_mlp, width=0.4, color=C["mlp"], label="best constrained MLP")
    ax.bar(xs + 0.2, rf, width=0.4, color=C["rf"], label="random forest")
    for i, (a, b) in enumerate(zip(best_mlp, rf)):
        ax.text(i, max(a, b) + 0.0008, f"{b / a:.2f}$\\times$", ha="center",
                fontsize=6.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=20, fontsize=6.8)
    ax.set_ylabel("test MAE")
    ax.set_ylim(0, 0.023)
    legend_below(ax, 1, y=-0.20)
    panel(ax, "b", "MLP vs trees")

    ax = fig.add_subplot(gs[0, 2])
    keys = [("mlp_sigmoid", "sigmoid"), ("rf_raw", "RF raw"),
            ("xgb_raw", "XGB raw"), ("mlp_softmax", "softmax"),
            ("mlp_renorm", "renorm"), ("rf_renorm", "RF renorm")]
    for i, (key, lab) in enumerate(keys):
        v = [agg(results(s), key, "consistency")[0] for s in SYS]
        v = [max(x, 1e-18) for x in v]
        ax.scatter([i] * len(v), v, s=13, color="#4c72b0" if i < 3 else "#2a6f3f",
                   zorder=3, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 1e-1)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[1] for k in keys], rotation=45, ha="right", fontsize=6.4)
    ax.set_ylabel("mean $|\\sum_k \\hat y_k - 1|$")
    guide(ax.axhspan(1e-19, 1e-6, color="#2a6f3f", alpha=0.09, lw=0))
    # Corridor between the constrained heads (1e-8) and the renormalised
    # trees (1e-17); no model lands here.
    ax.annotate("on-simplex", xy=(0.02, 3e-13), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=6.2, color="#2a6f3f")
    panel(ax, "c", "simplex closure")

    save(fig, "fig2_heads")


# --------------------------------------------------------------------------
def fig3_fields():
    """Predicted phase fields vs the CALPHAD oracle, and where error sits."""
    system = "fecrni"
    names, _ = phases(system)
    z = pred(system, "mlp_renorm")
    f, yt, yp = z["features"], z["y_true"], z["y_pred"]
    cr, T = f[:, 1], f[:, 3]

    fig = plt.figure(figsize=(FULLW, 0.66 * FULLW))
    gs = fig.add_gridspec(3, len(names) + 1, width_ratios=[1] * len(names) + [0.055])
    letters = "abcdefghijkl"
    sc_val = sc_err = None
    for j, nm in enumerate(names):
        for row, (v, tag) in enumerate([(yt[:, j], "CALPHAD"), (yp[:, j], "surrogate")]):
            ax = fig.add_subplot(gs[row, j])
            sc_val = ax.scatter(cr, T, c=v, s=1.4, cmap="magma", vmin=0, vmax=1,
                                linewidths=0, rasterized=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(700, 2000)
            if row == 0:
                ax.set_title(f"({letters[j]})  {nm}", loc="left", fontsize=7.4)
            if j == 0:
                ax.set_ylabel(f"{tag}\n$T$ (K)", fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax = fig.add_subplot(gs[2, j])
        err = np.abs(yp[:, j] - yt[:, j])
        sc_err = ax.scatter(cr, T, c=err, s=1.4, cmap="viridis", vmin=0,
                            vmax=0.15, linewidths=0, rasterized=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(700, 2000)
        ax.set_xlabel("$x_{\\mathrm{Cr}}$")
        if j == 0:
            ax.set_ylabel("$|$error$|$\n$T$ (K)", fontsize=7)
        else:
            ax.set_yticklabels([])
    cax = fig.add_subplot(gs[0:2, len(names)])
    cb = fig.colorbar(sc_val, cax=cax)
    cb.set_label("$N^\\phi$", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.2)
    cb.ax._is_colorbar = True
    cax = fig.add_subplot(gs[2, len(names)])
    cb = fig.colorbar(sc_err, cax=cax)
    cb.set_label("$|\\hat y - y|$", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.2)
    cb.ax._is_colorbar = True
    save(fig, "fig3_fields")


# --------------------------------------------------------------------------
def fig4_perphase():
    """Per-phase accuracy and the detection/regression divergence."""
    fig = plt.figure(figsize=(FULLW, 0.46 * FULLW))
    gs = fig.add_gridspec(1, 3)

    # (a) parity for the hardest and easiest phase of fecrni
    ax = fig.add_subplot(gs[0, 0])
    z = pred("fecrni", "mlp_renorm")
    names, _ = phases("fecrni")
    yt, yp = z["y_true"], z["y_pred"]
    mae = np.mean(np.abs(yp - yt), axis=0)
    for j, mk, col in [(int(np.argmax(mae)), "o", "#c44e52"),
                       (int(np.argmin(mae)), "^", "#4c72b0")]:
        ax.scatter(yt[:, j], yp[:, j], s=2.2, marker=mk, alpha=0.4,
                   color=col, linewidths=0, rasterized=True,
                   label=f"{names[j]} (MAE {mae[j]:.4f})")
    guide(ax.plot([0, 1], [0, 1], color="k", lw=0.7, ls="--")[0])
    ax.set_xlabel("CALPHAD $N^\\phi$")
    ax.set_ylabel("surrogate $\\hat N^\\phi$")
    ax.set_aspect("equal")
    leg = legend_below(ax, 1, y=-0.22)
    for h in leg.legend_handles:
        h.set_sizes([9])
        h.set_alpha(1.0)
    panel(ax, "a", "Fe-Cr-Ni, MLP renorm")

    # (b) per-phase MAE vs phase occurrence, all systems
    ax = fig.add_subplot(gs[0, 1])
    for s, mk in zip(SYS, ["o", "s", "^", "D", "v"]):
        d = results(s)
        names, cols = phases(s)
        best = min(["mlp_renorm", "mlp_sig_norm", "mlp_softmax"],
                   key=lambda k: agg(d, k)[0])
        per = np.mean([d[f"{best}_s{k}"]["mae"] for k in SEEDS], axis=0)
        raw = pd.read_csv(os.path.join(D, f"dataset_{s}.csv"))
        occ = [100 * float((raw[c] > 1e-3).mean()) for c in cols]
        ax.scatter(occ, per, s=15, marker=mk, label=LBL[s], alpha=0.85,
                   edgecolors="white", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("phase occurrence (% of rows)")
    ax.set_ylabel("per-phase MAE")
    legend_below(ax, 3, y=-0.22, fontsize=6.2)
    panel(ax, "b", "rare phases are cheap in MAE")

    # (c) MAE ratio vs F1 ratio, MLP over RF
    ax = fig.add_subplot(gs[0, 2])
    offsets = {"fecrni": (5, 4), "fecrmn": (5, -9), "fecrmo": (-30, 4),
               "fecrv": (5, 4), "femnni": (-30, -9)}
    for s, mk in zip(SYS, ["o", "s", "^", "D", "v"]):
        d = results(s)
        best = min(["mlp_renorm", "mlp_sig_norm", "mlp_softmax"],
                   key=lambda k: agg(d, k)[0])
        rmae = agg(d, best)[0] / agg(d, "rf_renorm")[0]
        rf1 = agg(d, best, "mean_f1")[0] / agg(d, "rf_renorm", "mean_f1")[0]
        ax.scatter(rmae, rf1, s=28, marker=mk, edgecolors="white", linewidths=0.4)
        ax.annotate(LBL[s].replace("Fe-", ""), (rmae, rf1), fontsize=6.0,
                    textcoords="offset points", xytext=offsets[s])
    guide(ax.axvline(1, color="k", lw=0.7, ls="--"))
    guide(ax.axhline(1, color="k", lw=0.7, ls="--"))
    ax.set_xlabel("MAE ratio  MLP / RF")
    ax.set_ylabel("$F_1$ ratio  MLP / RF")
    ax.set_xlim(0.35, 2.9)
    ax.set_ylim(0.5, 1.30)
    panel(ax, "c", "regressor $\\neq$ detector")

    save(fig, "fig4_perphase")


# --------------------------------------------------------------------------
def fig5_holdout():
    """Contiguous-band holdout penalties and the ablations."""
    fig = plt.figure(figsize=(FULLW, 0.46 * FULLW))
    gs = fig.add_gridspec(1, 3)

    bh = {}
    for s in SYS:
        p = os.path.join(M, f"block_holdout_{s}.json")
        if os.path.exists(p):
            bh[s] = json.load(open(p))

    def bagg(s, tag):
        dd = bh.get(s, {})
        v = [dd[f"{tag}_s{k}"]["mean_mae"] for k in SEEDS if f"{tag}_s{k}" in dd]
        return float(np.mean(v)) if v else np.nan

    d = results("fecrni")

    ax = fig.add_subplot(gs[0, 0])
    hmodels = [("mlp_renorm", "MLP", "o", C["renorm"]),
               ("rf_renorm", "RF", "s", C["random forest"]),
               ("xgb_renorm", "XGB", "^", C["XGBoost"])]
    xs = np.arange(len(SYS))
    for blk, blab, fill in [("T_band", "$T$ band", True),
                            ("X2_band", "$x_2$ band", False)]:
        for mi, (key, mlab, mk, col) in enumerate(hmodels):
            ratios = [bagg(s, f"{blk}_{key}") / bagg(s, f"random_ctrl_{key}")
                      for s in SYS]
            ax.plot(xs + (mi - 1) * 0.22, ratios, marker=mk, ms=3.6, ls="none",
                    color=col, markerfacecolor=col if fill else "none",
                    markeredgewidth=0.9, label=f"{mlab}, {blab}")
    guide(ax.axhline(1.0, color="0.5", lw=0.8, ls="--"))
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=18, fontsize=6.4)
    ax.set_ylabel("MAE ratio vs. random control")
    ax.set_ylim(0.03, 7.5)
    legend_below(ax, 2, y=-0.34)
    panel(ax, "a", "contiguous-band penalty ratios")

    ax = fig.add_subplot(gs[0, 1])
    widths = [64, 128, 256]
    for key, lab in [("mlp_renorm", "renorm"), ("mlp_sig_norm", "sig-norm"),
                     ("mlp_softmax", "softmax"), ("mlp_sigmoid", "sigmoid")]:
        ys, es = [], []
        for w in widths:
            dd = json.load(open(os.path.join(M, f"width{w}.json")))
            v = [dd[f"{key}_s{s}"]["mean_mae"] for s in SEEDS if f"{key}_s{s}" in dd]
            ys.append(np.mean(v))
            es.append(np.std(v))
        ys.append(agg(d, key)[0])
        es.append(agg(d, key)[1])
        ax.errorbar(widths + [192], ys, yerr=es, marker="o", ms=3, capsize=1.4,
                    lw=1.0, label=lab, color=C[lab])
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 192, 256])
    ax.set_xticklabels(["64", "128", "192", "256"])
    ax.set_xlabel("hidden width")
    ax.set_ylabel("test MAE")
    ax.set_ylim(0.008, 0.030)
    legend_below(ax, 4, y=-0.20)
    panel(ax, "b", "capacity")

    ax = fig.add_subplot(gs[0, 2])
    losses = [("huber", "Huber $\\delta{=}0.01$"), ("mae", "MAE"), ("mse", "MSE")]
    for i, (key, lab) in enumerate([("mlp_renorm", "renorm"),
                                    ("mlp_sig_norm", "sig-norm")]):
        ys, es = [], []
        for lk, _ in losses:
            if lk == "huber":
                mm, ee, _ = agg(d, key)
            else:
                dd = json.load(open(os.path.join(M, f"loss_ablation_{lk}.json")))
                v = [dd[f"{key}_s{s}"]["mean_mae"] for s in SEEDS
                     if f"{key}_s{s}" in dd]
                mm, ee = np.mean(v), np.std(v)
            ys.append(mm)
            es.append(ee)
        ax.bar(np.arange(3) + (i - 0.5) * 0.36, ys, width=0.36, yerr=es,
               capsize=1.4, label=lab, color=C[lab], error_kw=dict(lw=0.6))
    ax.set_xticks(range(3))
    ax.set_xticklabels([l[1] for l in losses], fontsize=6.4)
    ax.set_ylabel("test MAE")
    ax.set_ylim(0, 0.021)
    legend_below(ax, 2, y=-0.20)
    panel(ax, "c", "loss function")

    save(fig, "fig5_ablation")

# --------------------------------------------------------------------------
def fig7_extrap():
    """Strict out-of-range extrapolation: penalty ratios vs matched controls."""
    ex = {}
    for s in SYS:
        p = os.path.join(M, f"holdout_extrap_{s}.json")
        if os.path.exists(p):
            ex[s] = json.load(open(p))
    if not ex:
        print("  holdout_extrap_<sys>.json not present -- skipping fig7")
        return

    def xagg(s, tag):
        dd = ex.get(s, {})
        v = [dd[f"{tag}_s{k}"]["mean_mae"] for k in SEEDS if f"{tag}_s{k}" in dd]
        return float(np.mean(v)) if v else np.nan

    fig = plt.figure(figsize=(FULLW, 0.40 * FULLW))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    hmodels = [("mlp_renorm", "MLP", "o", C["renorm"]),
               ("rf_renorm", "RF", "s", C["random forest"]),
               ("xgb_renorm", "XGB", "^", C["XGBoost"])]
    xs = np.arange(len(SYS))
    for blk, blab, fill in [("X2_extrap", "$x_2$ extrapolation", False),
                            ("T_extrap", "$T$ extrapolation", True)]:
        for mi, (key, mlab, mk, col) in enumerate(hmodels):
            ratios = [xagg(s, f"{blk}_{key}") / xagg(s, f"{blk}_ctrl_{key}")
                      for s in SYS]
            ax.plot(xs + (mi - 1) * 0.22, ratios, marker=mk, ms=3.6, ls="none",
                    color=col, markerfacecolor=col if fill else "none",
                    markeredgewidth=0.9, label=f"{mlab}, {blab}")
    guide(ax.axhline(1.0, color="0.5", lw=0.8, ls="--"))
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=18, fontsize=6.4)
    ax.set_ylabel("MAE ratio vs. matched random control")
    ax.set_ylim(0.5, 40)
    legend_below(ax, 2, y=-0.34)
    panel(ax, "a", "strict extrapolation penalty ratios")

    ax = fig.add_subplot(gs[0, 1])
    for blk, blab, fill in [("X2_extrap", "$x_2$ extrapolation", False),
                            ("T_extrap", "$T$ extrapolation", True)]:
        for mi, (key, mlab, mk, col) in enumerate(hmodels):
            vals = [xagg(s, f"{blk}_{key}") for s in SYS]
            ax.plot(xs + (mi - 1) * 0.22, vals, marker=mk, ms=3.6, ls="none",
                    color=col, markerfacecolor=col if fill else "none",
                    markeredgewidth=0.9, label=f"{mlab}, {blab}")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=18, fontsize=6.4)
    ax.set_ylabel("test MAE on the far side")
    ax.set_ylim(0.01, 0.6)
    legend_below(ax, 2, y=-0.34)
    panel(ax, "b", "absolute MAE beyond the training range")

    save(fig, "fig7_extrap")


# --------------------------------------------------------------------------
def fig6_failures():
    """The three negative results: sparsemax, penalty, residue."""
    fig = plt.figure(figsize=(FULLW, 0.44 * FULLW))
    gs = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    for i, s in enumerate(SYS):
        d = results(s)
        _, _, sm = agg(d, "mlp_sparsemax")
        _, _, rn = agg(d, "mlp_renorm")
        ax.scatter([i - 0.14] * len(sm), sm, s=16, color="#dd8452", zorder=3,
                   label="sparsemax" if i == 0 else None)
        ax.scatter([i + 0.14] * len(rn), rn, s=16, color="#2a6f3f", zorder=3,
                   marker="^", label="renorm" if i == 0 else None)
        ax.plot([i - 0.14] * 2, [min(sm), max(sm)], color="#dd8452", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(SYS)))
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=20, fontsize=6.6)
    ax.set_ylabel("test MAE (per seed)")
    legend_below(ax, 2, y=-0.20)
    panel(ax, "a", "sparsemax: seed lottery")

    ax = fig.add_subplot(gs[0, 1])
    d = results("fecrni")
    labs = ["sigmoid\n(no penalty)", "$\\lambda=1$", "$\\lambda=10$"]
    keys = ["mlp_sigmoid", "mlp_penalty_1", "mlp_penalty_10"]
    m = [agg(d, k)[0] for k in keys]
    e = [agg(d, k)[1] for k in keys]
    cons = [agg(d, k, "consistency")[0] for k in keys]
    ax.bar(range(3), m, yerr=e, capsize=1.6, width=0.62,
           color=["#8c8c8c", "#dd8452", "#c44e52"], error_kw=dict(lw=0.6))
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=6.4)
    ax.set_ylabel("test MAE")
    ax2 = ax.twinx()
    ax2.plot(range(3), cons, marker="o", ms=3.4, color="#4c72b0", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean $|\\sum_k \\hat y_k - 1|$", color="#4c72b0")
    ax2.tick_params(axis="y", colors="#4c72b0")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    panel(ax, "b", "penalty: buys closure, loses fit")

    ax = fig.add_subplot(gs[0, 2])
    for i, s in enumerate(SYS):
        sw = json.load(open(os.path.join(M, f"residue_sweep_{s}.json")))
        vals = sorted(v["mean_mae"] for v in sw.values())
        d = results(s)
        best = min(agg(d, k)[0] for k in ["mlp_renorm", "mlp_sig_norm", "mlp_softmax"])
        ax.scatter([i] * len(vals), vals, s=13, color="#c44e52", zorder=3,
                   label="residue, each dropped phase" if i == 0 else None)
        ax.scatter([i], [best], s=26, marker="*", color="#2a6f3f", zorder=4,
                   label="best constrained head" if i == 0 else None)
        ax.plot([i, i], [min(vals), max(vals)], color="#c44e52", lw=0.8)
    ax.set_xticks(range(len(SYS)))
    ax.set_xticklabels([LBL[s] for s in SYS], rotation=20, fontsize=6.6)
    ax.set_ylabel("test MAE (seed 42)")
    ax.set_ylim(0, 0.075)
    legend_below(ax, 1, y=-0.20, fontsize=6.2)
    panel(ax, "c", "residue: no good choice")

    save(fig, "fig6_failures")


if __name__ == "__main__":
    print("figures ->", FIG)
    fig1_dataset()
    fig2_heads()
    fig3_fields()
    fig4_perphase()
    fig5_holdout()
    fig6_failures()
    fig7_extrap()
