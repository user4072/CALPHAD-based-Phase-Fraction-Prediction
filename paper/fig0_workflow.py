"""Figure 0: study workflow for the introduction.

A schematic of the pipeline only (no results): CALPHAD data generation,
the cluster-stratified split, the parallel model families, and the three
evaluation protocols (each spatial one with a size-matched random control),
plus the two secondary analyses. Purely schematic -- no numbers are read
from artefacts beyond the fixed protocol constants stated in the text.

Layout: three horizontal bands, all connectors are arrows with heads (no
headless lines). The two model families converge into a single "trained
surrogates" node, which fans out to the three evaluation protocols; the two
secondary analyses (ablations, boundary-error analysis) are fed by dashed
arrows.

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

W = 11.6
H = 11.5
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
        self.ax.text(0.80, y1 - 0.10, header, ha="left", va="center",
                     fontsize=FS_HEAD, color=color, weight="bold", zorder=4)

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
    L = Layout()
    CX = (BAND_L + BAND_R) / 2          # centre x

    # columns: the three protocols; secondary analyses sit in the reserved
    # corridors between columns so no vertical drop crosses any box
    X_L, X_M, X_R = 2.05, CX, 9.50      # interp / band-holdout / extrapolation
    X_A, X_B = 3.85, 7.72               # ablate, boundary

    # ============================================================ BAND 1
    b1y0, b1y1 = 7.05, 11.20
    L.band(b1y0, b1y1, "1  CALPHAD DATA GENERATION  (pycalphad + MatCalc)", DATA)

    L.box("db", CX, 10.45, 4.6, 0.62,
          "MatCalc steel database  mc_fe_v2.062\n"
          "custom parser \u2192 five Fe-based ternary subsystems",
          fc=DATA_LT, ec=DATA)
    L.box("probe", CX, 9.59, 5.4, 0.62,
          "Probe-driven phase sets\n"
          "2,000 probe points \u00b7 retain phases ever active \u00b7 K = 4\u20139 targets",
          fc=DATA_LT, ec=DATA)
    L.box("sample", CX, 8.73, 5.6, 0.66,
          "Structured sampling \u2014 five strategies, 11,220 candidates\n"
          "Fe-weighted uniform \u00b7 sigma-focused \u00b7 isothermal grid\n"
          "liquidus-zone \u00b7 near-pure-Fe draws",
          fc=DATA_LT, ec=DATA, fs=FS_SMALL)
    L.box("gate", CX, 7.80, 3.4, 0.86,
          "mass-balance gate\n$|\\Sigma N^\\phi - 1| < 10^{-6}$",
          shape="diamond", fc=GATE_LT, ec=GATE, fs=FS_SMALL)

    L.arrow([L.edge("db", "b"), L.edge("probe", "t")],
            name="db->probe", orig="db", dest="probe")
    L.arrow([L.edge("probe", "b"), L.edge("sample", "t")],
            name="probe->sample", orig="probe", dest="sample")
    L.arrow([L.edge("sample", "b"), L.edge("gate", "t")],
            name="sample->gate", orig="sample", dest="gate")

    # ============================================================ BAND 2
    b2y0, b2y1 = 3.05, 6.80
    L.band(b2y0, b2y1, "2  SPLIT  +  MODEL FAMILIES", MODEL)

    L.box("split", CX, 6.12, 6.4, 0.62,
          "Cluster-stratified split  64 / 16 / 20  (train / val / test)\n"
          "KMeans k = 6 on (x\u2082, x\u2083, T) \u00b7 seeds 42, 123, 2024 \u00b7 "
          "test never touched in fitting",
          fc=MODEL_LT, ec=MODEL)

    # two parallel model branches
    L.box("mlp", CX - 2.75, 4.825, 4.6, 0.86,
          "Constrained MLP  (3 \u00d7 192)\n"
          "six heads: sigmoid \u00b7 residue \u00b7 sparsemax \u00b7\n"
          "softmax \u00b7 sigmoid/\u03a3 \u00b7 renorm\n"
          "+ probes: power-norm \u00b7 sum-to-one penalty",
          fc=MODEL_LT, ec=MODEL, fs=FS_SMALL)
    L.box("base", CX + 2.75, 4.90, 4.6, 0.72,
          "Classical baselines\n"
          "ridge \u00b7 k-NN \u00b7 XGBoost \u00b7 random forest\n"
          "each raw and after post-hoc renormalization",
          fc=MODEL_LT, ec=MODEL, fs=FS_SMALL)

    # convergence node: everything that gets fitted, before evaluation
    L.box("trained", CX, 3.57, 4.6, 0.55,
          "Trained surrogates\n"
          "six heads \u00b7 four baselines \u00b7 three seeds",
          fc=MODEL_LT, ec=MODEL, fs=FS_SMALL)

    L.arrow([L.edge("gate", "b"), L.edge("split", "t")],
            name="gate->split", orig="gate", dest="split")

    # fork: split -> mlp / base (elbow arrows)
    sy = L.edge("split", "b")[1]
    jy = sy - 0.25
    mx, bx = L.edge("mlp", "t")[0], L.edge("base", "t")[0]
    L.arrow([(CX, sy), (CX, jy), (mx, jy), L.edge("mlp", "t")],
            name="split->mlp", orig="split", dest="mlp")
    L.arrow([(CX, sy), (CX, jy), (bx, jy), L.edge("base", "t")],
            name="split->base", orig="split", dest="base")

    # converge: mlp / base -> trained (elbow arrows into the top edge)
    ny = L.edge("mlp", "b")[1] - 0.25
    tx_l, tx_r = CX - 1.40, CX + 1.40
    L.arrow([L.edge("mlp", "b"), (mx, ny), (tx_l, ny), (tx_l, L.top_at("trained", tx_l)[1])],
            name="mlp->trained", orig="mlp", dest="trained")
    L.arrow([L.edge("base", "b"), (bx, ny), (tx_r, ny), (tx_r, L.top_at("trained", tx_r)[1])],
            name="base->trained", orig="base", dest="trained")

    # ============================================================ BAND 3
    b3y0, b3y1 = 0.30, 2.80
    L.band(b3y0, b3y1, "3  EVALUATION PROTOCOLS", EVAL)

    # three evaluation protocols (row 1 of band 3)
    L.box("interp", X_L, 1.77, 3.3, 0.66,
          "Interpolation test\n"
          "MAE \u00b7 closure \u00b7 detection \u00b7 admissibility\n"
          "paired bootstrap",
          fc=EVAL_LT, ec=EVAL, fs=FS_SMALL)
    L.box("bandhold", X_M, 1.72, 3.5, 0.66,
          "Contiguous interior-band holdout\n"
          "T \u2208 [1200,1400] K \u00b7 x\u2082 \u2208 [0.25,0.35]\n"
          "+ size-matched control \u00b7 region-matched reanalysis",
          fc=EVAL_LT, ec=EVAL, fs=FS_SMALL)
    L.box("extrap", X_R, 1.72, 3.4, 0.66,
          "Strict one-sided extrapolation\n"
          "x\u2082: \u2264 0.25 \u2192 > 0.35 \u00b7 T: \u2264 1200 K \u2192 > 1400 K\n"
          "+ size-matched random control",
          fc=EVAL_LT, ec=EVAL, fs=FS_SMALL)

    # fan-out: every evaluation box gets its own arrow, each one starting

    # on the bottom edge of "Trained surrogates". The start x's follow the
    # left-to-right order of the targets, so no two arrows cross.

    tb = L.edge("trained", "b")[1]

    JY = 2.48                              # elbow height for the outer arrows

    x_i = L.edge("trained", "l")[0] + 0.10
    x_e = L.edge("trained", "r")[0] - 0.10
    L.arrow([(x_i, tb), (x_i, JY), (X_L, JY), L.edge("interp", "t")],

            name="trained->interp", orig="trained", dest="interp")

    L.arrow([(CX, tb), L.edge("bandhold", "t")],

            name="trained->band", orig="trained", dest="bandhold")

    L.arrow([(x_e, tb), (x_e, JY), (X_R, JY), L.edge("extrap", "t")],

            name="trained->extrap", orig="trained", dest="extrap")

    # secondary analyses (row 2 of band 3), dashed = derived analyses
    L.box("ablate", X_A, 0.77, 3.3, 0.52,
          "Ablations: width \u00b7 loss \u00b7 penalty\n"
          "presence-threshold sensitivity 10\u207b\u2074\u201310\u207b\u00b2",
          fc=EVAL_LT, ec=EVAL, fs=FS_SMALL)
    L.box("boundary", X_B, 0.77, 3.3, 0.52,
          "Boundary-error analysis\n"
          "MAE vs. distance to nearest active-phase-set shift",
          fc=EVAL_LT, ec=EVAL, fs=FS_SMALL)

    L.arrow([(X_A, tb), L.edge("ablate", "t")], color=SEC, ls="--",

            name="trained->ablate", orig="trained", dest="ablate")

    L.arrow([(X_B, tb), L.edge("boundary", "t")], color=SEC, ls="--",

            name="trained->boundary", orig="trained", dest="boundary")

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
