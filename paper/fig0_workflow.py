"""Figure 0: study workflow for the introduction.

A Stoco-style single-flow schematic: five solid color blocks with white
bold titles (data -> split -> train -> evaluate -> deploy), one labeled
arrow between consecutive blocks, and small side annotations. No bands,
no parallel rails, no elbows. Purely schematic -- no numbers are read
from artefacts beyond the fixed protocol constants stated in the text.

Usage: py -3.12 paper/fig0_workflow.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10.0,
    "figure.dpi": 150,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "axes.unicode_minus": False,
})

# palette: one neutral base + semantic accents
INK = "#1f1f1f"
DATA = "#2d6a9e"; DATA_LT = "#dcebf5"     # CALPHAD / data stage
MODEL = "#b56a2e"; MODEL_LT = "#f8eddf"   # model families
EVAL = "#4c7a3d"; EVAL_LT = "#e9f0e4"     # evaluation protocols
GATE = "#b5862a"; GATE_LT = "#fdf3dd"     # decision / acceptance gate
BASE = "#fbfbfa"; BASE_EC = "#9a9a8e"
SEC = "#777777"                           # secondary-analysis arrows

W = 6.40
H = 8.60
BAND_L = 0.30
BAND_R = 11.30

FS_HEAD = 12.0
FS_BODY = 10.0
FS_SMALL = 8.6

PAD_H = 0.16
PAD_V = 0.10


def _measure(scratch, text, fs, weight="normal"):
    t = scratch.text(0.5, 0.5, text, fontsize=fs, ha="center", va="center",
                     weight=weight, linespacing=1.25)
    scratch.canvas.draw()
    bb = t.get_window_extent(renderer=scratch.canvas.get_renderer())
    t.remove()
    return bb.width / scratch.dpi, bb.height / scratch.dpi


class Layout:
    def __init__(self):
        self.scratch = plt.figure(figsize=(4, 4))
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.boxes = []
        self.arrows = []
        self.warns = []

    # -- primitives ---------------------------------------------------------
    def rbox(self, cx, cy, w, h, fc, ec, lw=1.2, radius=0.08, zorder=3):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                         boxstyle=f"round,pad=0,rounding_size={radius}",
                                         facecolor=fc, edgecolor=ec,
                                         linewidth=lw, zorder=zorder))

    def diamond(self, cx, cy, w, h, fc, ec, lw=1.3):
        self.ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                                   (cx, cy - h / 2), (cx - w / 2, cy)],
                                  closed=True, facecolor=fc, edgecolor=ec,
                                  linewidth=lw, zorder=3))

    def band(self, y0, y1, header, color):
        self.rbox((BAND_L + BAND_R) / 2, (y0 + y1) / 2, BAND_R - BAND_L,
                  y1 - y0, "#fafaf7", "#c9c9bd", lw=1.0, radius=0.12,
                  zorder=1)
        # filled header strip with white bold text (modern journal look);
        # the strip sits above everything (zorder 5) so connector drops
        # pass behind it instead of crossing the header text
        strip_h = 0.46
        self.ax.add_patch(FancyBboxPatch(
            (BAND_L + 0.02, y1 - strip_h), BAND_R - BAND_L - 0.04, strip_h - 0.06,
            boxstyle="round,pad=0,rounding_size=0.09",
            facecolor=color, edgecolor=color, linewidth=0, zorder=5))
        self.ax.text(0.80, y1 - strip_h / 2 - 0.01, header, ha="left",
                     va="center", fontsize=FS_HEAD, color="white",
                     weight="bold", zorder=6)

    def text(self, x, y, s, fs=FS_BODY, color=INK, weight="normal",
             ha="center", va="center"):
        return self.ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                            weight=weight, linespacing=1.25, zorder=4)

    # -- interior size for a shape -----------------------------------------
    def _inner(self, shape, w, h):
        if shape == "diamond":
            return (w - 0.22) * 0.72, h * 0.62
        return w, h

    def _fit(self, shape, w, h, tw, th):
        for _ in range(8):
            iw, ih = self._inner(shape, w, h)
            dw = (tw + 2 * PAD_H) - iw
            dh = (th + 2 * PAD_V) - ih
            if dw <= 0 and dh <= 0:
                break
            w += max(dw, 0.0) * 1.15
            h += max(dh, 0.0) * 1.15
        return w, h

    def box(self, name, cx, cy, w, h, lines, shape="rbox",
            fc=BASE, ec=BASE_EC, lw=1.2, fs=FS_BODY, weight="normal"):
        tw, th = _measure(self.scratch, lines, fs, weight)
        w, h = self._fit(shape, w, h, tw, th)
        if shape == "diamond":
            self.diamond(cx, cy, w, h, fc, ec, lw)
        else:
            self.rbox(cx, cy, w, h, fc, ec, lw)
        self.text(cx, cy, lines, fs=fs, weight=weight)
        self.boxes.append((name, cx, cy, w, h, shape))
        iw, ih = self._inner(shape, w, h)
        if tw > iw - 0.10 or th > ih - 0.05:
            self.warns.append(f"TEXT-FIT {name}: text {tw:.2f}x{th:.2f} "
                              f"> inner {iw - 0.10:.2f}x{ih - 0.05:.2f}")

    def edge(self, name, side):
        b = next(b for b in self.boxes if b[0] == name)
        if side in ("l", "r"):
            return b[1] + (b[3] / 2 if side == "r" else -b[3] / 2), b[2]
        return b[1], b[2] + (b[4] / 2 if side == "t" else -b[4] / 2)

    def top_at(self, name, x):
        """Point on the top edge of a box at a given x."""
        b = next(b for b in self.boxes if b[0] == name)
        return x, b[2] + b[4] / 2

    def arrow(self, pts, color="#444444", lw=1.3, ls="-", scale=13,
              name=None, orig=None, dest=None):
        # arrows are drawn above the boxes (zorder 4) so the heads are
        # never hidden by the box they point into
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            if i == len(pts) - 2:
                self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                                  mutation_scale=scale,
                                                  linewidth=lw, color=color,
                                                  linestyle=ls, shrinkA=0,
                                                  shrinkB=0, zorder=4))
            else:
                self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color,
                             lw=lw, ls=ls, solid_capstyle="round", zorder=4)
        if name:
            self.arrows.append((pts, name, orig, dest))

    # -- QC -----------------------------------------------------------------
    def qc(self):
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                ax0, ay0 = a[1] - a[3] / 2, a[2] - a[4] / 2
                ax1, ay1 = a[1] + a[3] / 2, a[2] + a[4] / 2
                bx0, by0 = b[1] - b[3] / 2, b[2] - b[4] / 2
                bx1, by1 = b[1] + b[3] / 2, b[2] + b[4] / 2
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > 0.02 and oy > 0.02:
                    self.warns.append(f"OVERLAP {a[0]} x {b[0]}: "
                                      f"{ox:.2f}x{oy:.2f}")
        for b in self.boxes:
            x0, y0 = b[1] - b[3] / 2, b[2] - b[4] / 2
            if (x0 < 0.005 or x0 + b[3] > W - 0.005
                    or y0 < 0.005 or y0 + b[4] > H - 0.005):
                self.warns.append(f"BOX-OUT {b[0]} at {x0:.2f},{y0:.2f}")
        for pts, name, orig, dest in self.arrows:
            skip = {orig, dest}
            for b in self.boxes:
                if b[0] in skip:
                    continue
                bx0, by0 = b[1] - b[3] / 2, b[2] - b[4] / 2
                bx1, by1 = b[1] + b[3] / 2, b[2] + b[4] / 2
                for i in range(len(pts) - 1):
                    x0, y0 = pts[i]
                    x1, y1 = pts[i + 1]
                    if x0 == x1:
                        if bx0 < x0 < bx1 and min(y0, y1) < by1 and max(y0, y1) > by0:
                            self.warns.append(f"ARROW-THRU {name} -> {b[0]}")
                    elif y0 == y1:
                        if by0 < y0 < by1 and min(x0, x1) < bx1 and max(x0, x1) > bx0:
                            self.warns.append(f"ARROW-THRU {name} -> {b[0]}")
                    else:
                        self.warns.append(f"DIAGONAL {name}: "
                                          f"{x0},{y0} -> {x1},{y1}")


def build():
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"]})
    L = Layout()
    CX = 3.20  # centred single column

    stages = [
        ("data", "#3b82c4", "CALPHAD DATA",
         "MatCalc steel database", "9 systems \u00b7 88,542 equilibria"),
        ("split", "#e08a2e", "SPLIT",
         "cluster-stratified 64 / 16 / 20", "KMeans k = 6 \u00b7 3 seeds"),
        ("train", "#d14f4f", "TRAIN MODELS",
         "constrained MLP + gated head", "ridge \u00b7 k-NN \u00b7 XGBoost \u00b7 random forest"),
        ("eval", "#62a33c", "EVALUATE",
         "interpolation \u00b7 band holdout \u00b7 extrapolation",
         "ideal-form control (H/I) \u00b7 U1 triage"),
        ("deploy", "#7c6bc0", "DEPLOY",
         "anchor + 2 validated screens", "precision 1.00 and 0.98"),
    ]
    links = ["mass-balanced equilibria", "train \u00b7 val \u00b7 test",
             "trained surrogates", "where to trust"]

    BW, BH, GAP = 4.30, 1.02, 0.55
    cys = [H - 0.95 - i * (BH + GAP) for i in range(len(stages))]
    for (name, color, title, l1, l2), cy in zip(stages, cys):
        L.rbox(CX, cy, BW, BH, color, color, lw=1.4, radius=0.14)
        L.text(CX, cy + 0.24, title, fs=13.0, color="white", weight="bold")
        L.text(CX, cy - 0.10, l1, fs=10.0, color="white")
        L.text(CX, cy - 0.33, l2, fs=10.0, color="white")
        L.boxes.append((name, CX, cy, BW, BH, "rbox"))

    for i, lab in enumerate(links):
        y0 = cys[i] - BH / 2
        y1 = cys[i + 1] + BH / 2
        L.arrow([(CX, y0), (CX, y1)], color="#444444", lw=1.6, scale=15,
                name="flow%d" % i, orig=stages[i][0], dest=stages[i + 1][0])
        L.ax.text(CX + 0.30, (y0 + y1) / 2, lab, ha="left", va="center",
                  fontsize=8.5, color="#555555", zorder=4)

    # left bracket: held-out test rows (split -> evaluate), Stoco-style
    bx = CX - BW / 2 - 0.45
    y_top, y_bot = cys[1], cys[3]
    L.ax.plot([bx, bx], [y_bot, y_top], color="#444444", lw=1.3, zorder=4)
    L.ax.plot([bx, bx + 0.12], [y_top, y_top], color="#444444", lw=1.3,
              zorder=4)
    L.ax.plot([bx, bx + 0.12], [y_bot, y_bot], color="#444444", lw=1.3,
              zorder=4)
    L.ax.text(bx - 0.14, (y_top + y_bot) / 2, "held-out test rows",
              ha="center", va="center", rotation=90, fontsize=8.5,
              color="#555555", zorder=4)

    L.qc()
    for wmsg in L.warns:
        print("  QC:", wmsg)

    for ext in ("pdf", "png"):
        L.fig.savefig(FIG / f"fig0_workflow.{ext}")
    plt.close(L.fig)
    plt.close(L.scratch)
    print("wrote fig0_workflow.pdf/.png")
    return L.warns


if __name__ == "__main__":
    build()
