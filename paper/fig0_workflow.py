"""Figure 0: study workflow schematic.

Single-column (89 mm) technical schematic in the style of a journal method
figure: five stacked stage blocks on a warm-to-cool progression, each with a
drawn vector glyph (T-x grid, cluster-stratified split triangle, MLP with
simplex output, temperature-band holdout, phase-fraction screen), black
labelled arrows naming the data object transferred, and a left elbow marking
the held-out rows. Purely schematic -- no result metrics appear in the figure.

Usage: py -3.12 paper/fig0_workflow.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0,
    "figure.dpi": 200,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.unicode_minus": False,
})

INK = "#33404D"
W, H = 3.50, 5.52

CX, BW, BH, GAP = 1.50, 2.44, 0.72, 0.44
LEFT, RIGHT = CX - BW / 2, CX + BW / 2
CYS = [H - 0.44 - i * (BH + GAP) for i in range(5)]

GLYPH_X0, GLYPH_X1 = 0.36, 0.94
TEXT_X0, TEXT_X1 = 1.00, 2.68
FS_TITLE, FS_SUB, FS_ARROW, FS_SIDE = 7.0, 5.6, 5.0, 5.0

# warm-to-cool stage progression (hue family sampled from the reference style,
# deepened so that white bold text stays legible in print)
STAGE_FC = ["#D9535B", "#E08B4C", "#C7921F", "#7FA845", "#4A8FBF"]
CLUSTER = ["#C0504D", "#E8A33D", "#4E8FBF"]

LW = 0.7
GLW = 0.28


def _measure(scratch, s, fs, weight="normal"):
    t = scratch.text(0.5, 0.5, s, fontsize=fs, ha="center", va="center",
                     weight=weight, linespacing=1.30)
    scratch.canvas.draw()
    bb = t.get_window_extent(renderer=scratch.canvas.get_renderer())
    t.remove()
    return bb.width / scratch.dpi, bb.height / scratch.dpi


def g_grid(ax, cy, warns):
    """T-x sampling grid with an equilibrium envelope."""
    x0, x1, y0, y1 = 0.38, 0.92, cy - 0.19, cy + 0.25
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=INK, linewidth=LW, zorder=6))
    for f in (1 / 3, 2 / 3):
        ax.plot([x0, x1], [y0 + (y1 - y0) * f] * 2, color=INK, linewidth=0.35,
                linestyle=(0, (2, 2)), zorder=5)
        ax.plot([x0 + (x1 - x0) * f] * 2, [y0, y1], color=INK, linewidth=0.35,
                linestyle=(0, (2, 2)), zorder=5)
    xs = np.array([0.425, 0.545, 0.665, 0.785, 0.885])
    ys = y0 + (y1 - y0) * np.array([0.16, 0.34, 0.58, 0.80, 0.92])
    ax.plot(xs, ys, color=INK, linewidth=0.85, zorder=6)
    ax.plot(xs, ys, "o", ms=1.7, mfc=INK, mec="none", zorder=7)
    if min(ys) < y0 or max(ys) > y1:
        warns.append("glyph grid: envelope outside frame")


def g_split(ax, cy, warns):
    """Composition triangle with three KMeans clusters."""
    rng = np.random.default_rng(7)
    cx, top, half = 0.65, cy + 0.27, 0.26
    hh = half * np.sqrt(3) / 2
    A, B, C = (cx, top), (cx - half, top - hh), (cx + half, top - hh)
    ax.add_patch(Polygon([A, B, C], closed=True, fill=False, edgecolor=INK,
                         linewidth=LW, zorder=6))
    for k, col in enumerate(CLUSTER):
        for _ in range(5):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            px = u * A[0] + v * B[0] + w * C[0]
            py = u * A[1] + v * B[1] + w * C[1]
            ax.add_patch(Circle((px, py), 0.019, fc=col, ec="white",
                                linewidth=0.3, zorder=7))
    if not (GLYPH_X0 <= cx - half and cx + half <= GLYPH_X1):
        warns.append("glyph split: triangle wider than glyph lane")


def g_mlp(ax, cy, warns):
    """Feed-forward surrogate with an explicit simplex output bracket."""
    cols_x = [0.395, 0.565, 0.735, 0.885]
    counts = [3, 4, 4, 2]
    nodes = {}
    for x, n in zip(cols_x, counts):
        nodes[x] = [cy + (i - (n - 1) / 2) * 0.125 for i in range(n)]
    for a, b in zip(cols_x[:-1], cols_x[1:]):
        for ya in nodes[a]:
            for yb in nodes[b]:
                ax.plot([a, b], [ya, yb], color=INK, linewidth=GLW, zorder=4)
    for x, ys in nodes.items():
        for y in ys:
            ax.add_patch(Circle((x, y), 0.0245, fc="white", ec=INK,
                                linewidth=0.75, zorder=6))
    ax.plot([0.915, 0.933, 0.933, 0.915],
            [cy - 0.0625, cy - 0.0625, cy + 0.0625, cy + 0.0625],
            color=INK, linewidth=0.6, zorder=6)
    if 0.933 > GLYPH_X1:
        warns.append("glyph mlp: bracket outside glyph lane")


def g_bands(ax, cy, warns):
    """Envelope over a temperature axis with held-out bands."""
    x0, x1, y0, y1 = 0.38, 0.92, cy - 0.19, cy + 0.25
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=INK, linewidth=LW, zorder=6))
    w = x1 - x0
    for a, b in ((0.16, 0.30), (0.56, 0.70)):
        ax.add_patch(Rectangle((x0 + w * a, y0), w * (b - a), y1 - y0,
                               fc=INK, alpha=0.11, ec="none", zorder=4))
        for e in (a, b):
            ax.plot([x0 + w * e] * 2, [y0, y1], color=INK, linewidth=0.5,
                    linestyle=(0, (2, 1.6)), zorder=5)
    xs = np.linspace(x0 + 0.015, x1 - 0.015, 60)
    ys = y0 + (y1 - y0) * (0.30 + 0.52 * np.abs(np.sin((xs - x0) * 8.5)))
    ax.plot(xs, ys, color=INK, linewidth=0.85, zorder=6)
    if ys.min() < y0 or ys.max() > y1:
        warns.append("glyph bands: envelope outside frame")


def g_screen(ax, cy, warns):
    """Query arrow resolving to a stack of phase-fraction bars."""
    ax.add_patch(FancyArrowPatch((0.335, cy), (0.435, cy), arrowstyle="-|>",
                                 mutation_scale=4.5, linewidth=0.7,
                                 color=INK, shrinkA=0, shrinkB=0, zorder=6))
    rows = [(1.00, ), (0.66, ), (0.42, )]
    y = cy + 0.175
    for (frac,) in rows:
        ax.add_patch(Rectangle((0.475, y), 0.40 * frac, 0.082,
                               fc="#EEF2F5", ec=INK, linewidth=0.55,
                               zorder=6))
        y -= 0.125
    if 0.475 + 0.40 > GLYPH_X1:
        warns.append("glyph screen: bar outside glyph lane")


GLYPH = [g_grid, g_split, g_mlp, g_bands, g_screen]

STAGES = [
    ("data", "CALPHAD MODELING",
     "MatCalc steel database\n9 Fe-base alloy systems"),
    ("split", "STRATIFIED SPLIT",
     "cluster-stratified 64 / 16 / 20\nKMeans k = 6, 3 seeds"),
    ("train", "SURROGATE TRAINING",
     "simplex-constrained MLP + gated head\nridge, k-NN, XGBoost, forest"),
    ("eval", "EVALUATION",
     "interpolation, band holdout,\nextrapolation; ideal-form control"),
    ("deploy", "DEPLOYMENT",
     "experimental anchor and\ntwo high-throughput screens"),
]

LINKS = ["phase fractions", "train / val / test", "fitted surrogates",
         "phase fractions at (x, T)"]


def build():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    scratch = plt.figure(figsize=(4, 4))
    warns = []

    for i, ((name, title, sub), cy, fc) in enumerate(zip(STAGES, CYS, STAGE_FC)):
        ax.add_patch(FancyBboxPatch(
            (LEFT, cy - BH / 2), BW, BH,
            boxstyle="round,pad=0,rounding_size=0.055",
            facecolor=fc, edgecolor=fc, linewidth=0, zorder=3))
        GLYPH[i](ax, cy, warns)
        ax.text(TEXT_X0, cy + 0.135, title, ha="left", va="center",
                fontsize=FS_TITLE, color="white", weight="bold", zorder=6)
        ax.text(TEXT_X0, cy - 0.115, sub, ha="left", va="center",
                fontsize=FS_SUB, color="white", linespacing=1.30, zorder=6)
        tw, _ = _measure(scratch, title, FS_TITLE, "bold")
        sw, _ = _measure(scratch, sub, FS_SUB)
        if tw > TEXT_X1 - TEXT_X0 - 0.02:
            warns.append(f"TITLE-FIT {name}: {tw:.3f}in > "
                         f"{TEXT_X1 - TEXT_X0:.3f}in")
        if sw > TEXT_X1 - TEXT_X0 - 0.02:
            warns.append(f"SUB-FIT {name}: {sw:.3f}in > "
                         f"{TEXT_X1 - TEXT_X0:.3f}in")
        if cy - BH / 2 < 0 or cy + BH / 2 > H:
            warns.append(f"BOX-OUT {name}")

    for i, lab in enumerate(LINKS):
        y0, y1 = CYS[i] - BH / 2, CYS[i + 1] + BH / 2
        ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), arrowstyle="-|>",
                                     mutation_scale=6.0, linewidth=0.9,
                                     color=INK, shrinkA=0, shrinkB=0, zorder=4))
        lw_, _ = _measure(scratch, lab, FS_ARROW)
        ax.text(CX + 0.075, (y0 + y1) / 2, lab, ha="left", va="center",
                fontsize=FS_ARROW, color=INK, zorder=4)
        if CX + 0.075 + lw_ > W - 0.02:
            warns.append(f"ARROW-FIT {lab}: right edge {CX + 0.075 + lw_:.3f}in")

    ex = LEFT - 0.10
    y_top, y_bot = CYS[1] - BH / 2, CYS[3] + BH / 2
    ax.plot([ex, ex], [y_bot, y_top], color=INK, linewidth=0.9, zorder=4)
    for yy in (y_top, y_bot):
        ax.plot([ex, ex + 0.055], [yy, yy], color=INK, linewidth=0.9, zorder=4)
    ax.text(ex - 0.055, (y_top + y_bot) / 2, "held-out rows", ha="center",
            va="center", rotation=90, fontsize=FS_SIDE, color=INK, zorder=4)

    ax.add_patch(FancyArrowPatch((RIGHT + 0.30, CYS[2] + 0.02),
                                 (RIGHT - 0.02, CYS[2] + 0.02),
                                 arrowstyle="-|>", mutation_scale=5.0,
                                 linewidth=0.8, color=STAGE_FC[4],
                                 shrinkA=0, shrinkB=0, zorder=6))
    ax.text(RIGHT + 0.33, CYS[2] + 0.02, r"$\Sigma f_i = 1$", ha="left",
            va="center", fontsize=5.4, color=STAGE_FC[4], zorder=6)
    if RIGHT + 0.33 + 0.34 > W:
        warns.append("CALLOUT: simplex label overflows canvas")

    plt.close(scratch)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig0_workflow.{ext}")
    plt.close(fig)
    for wmsg in warns:
        print("  QC:", wmsg)
    print("wrote fig0_workflow.pdf/.png")
    return warns


if __name__ == "__main__":
    build()