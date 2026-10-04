"""Build the graphical abstract for the Fe-based ternary CALPHAD
surrogate benchmark paper.

Reuses the visual language of Graphical_Abstract_v2.pptx [Repaired].pptx
(13.33 x 7.5 in slide, Arial, three numbered panels with coloured header
bars, light-tinted sub-cards, small horizontal bar charts) but replaces
all content with the present study:

  Panel 1  CALPHAD data + constrained output heads (closure vs. accuracy)
  Panel 2  Spatial generalization with size-matched random controls
  Panel 3  Regression vs. phase-presence detection

Every number is taken from the paper's generated tables (tab/*.tex);
nothing is invented.

Usage: py -3.12 paper/make_graphical_abstract.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Graphical_Abstract_fe_surrogate.pptx"

# ---------------------------------------------------------------- palette
# panel accents follow the v2 file: blue / green / purple
BLUE, BLUE_DK = RGBColor(0x2F, 0x55, 0x97), RGBColor(0x24, 0x42, 0x76)
GREEN, GREEN_DK = RGBColor(0x2F, 0x6B, 0x4F), RGBColor(0x24, 0x52, 0x3C)
PURPLE, PURPLE_DK = RGBColor(0x5B, 0x3F, 0x8C), RGBColor(0x46, 0x30, 0x6C)
INK = RGBColor(0x2B, 0x2B, 0x2B)
GREY = RGBColor(0x5B, 0x65, 0x73)
LIGHT = RGBColor(0x8A, 0x93, 0xA0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAR_BG = RGBColor(0xE8, 0xEC, 0xF0)
BAR_BLUE = RGBColor(0x3F, 0x7C, 0xAC)
BAR_GREEN = RGBColor(0x4C, 0x8C, 0x6A)
BAR_PURPLE = RGBColor(0x7B, 0x5E, 0xA8)
BAR_RED = RGBColor(0xB0, 0x4A, 0x3A)
BAR_ORANGE = RGBColor(0xC0, 0x7A, 0x2E)
PH_BCC = RGBColor(0x3F, 0x7C, 0xAC)
PH_FCC = RGBColor(0xC0, 0x7A, 0x2E)
PH_LIQ = RGBColor(0x4A, 0x9B, 0x9B)
PH_SIG = RGBColor(0x7B, 0x5E, 0xA8)
GREEN_OK = RGBColor(0x2E, 0x7D, 0x4F)
RED_BAD = RGBColor(0xB0, 0x4A, 0x3A)
TINT_BLUE = RGBColor(0xF9, 0xFB, 0xFE)
TINT_GREEN = RGBColor(0xF7, 0xFB, 0xF8)
TINT_GREEN2 = RGBColor(0xED, 0xF7, 0xF0)
TINT_PURPLE = RGBColor(0xF7, 0xF5, 0xF9)
TINT_AMBER = RGBColor(0xFD, 0xF6, 0xE8)
LINE_SOFT = RGBColor(0xD6, 0xDC, 0xE2)

FONT = "Arial"

EMU_IN = 914400


def _set_font(run, size, color=INK, bold=False, italic=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color


class Board:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])

    # ------------------------------------------------------------- helpers
    def rect(self, x, y, w, h, fill, line=None, line_w=0.75, round_=False):
        shp = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = line
            shp.line.width = Pt(line_w)
        shp.shadow.inherit = False
        if round_:
            try:
                shp.adjustments[0] = 0.08
            except Exception:
                pass
        return shp

    def oval(self, x, y, w, h, fill, line=None):
        shp = self.slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x),
                                          Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = line
            shp.line.width = Pt(0.75)
        shp.shadow.inherit = False
        return shp

    def text(self, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line_spacing=1.0):
        """runs: list of paragraphs; each paragraph is a list of
        (text, size, color, bold) tuples."""
        tb = self.slide.shapes.add_textbox(Inches(x), Inches(y),
                                           Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, para in enumerate(runs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = line_spacing
            for (t, size, color, bold) in para:
                r = p.add_run()
                r.text = t
                _set_font(r, size, color, bold)
        return tb

    def arrow(self, x0, y0, x1, y1, color=GREY, width=1.5):
        conn = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                               Inches(x0), Inches(y0),
                                               Inches(x1), Inches(y1))
        conn.line.color.rgb = color
        conn.line.width = Pt(width)
        ln = conn.line._get_or_add_ln()
        from pptx.oxml.ns import qn
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle",
                                                "w": "med", "len": "med"})
        ln.append(tail)
        return conn

    def hbar(self, x, y, w, h, frac, color):
        """horizontal bar: grey track + coloured fill."""
        self.rect(x, y, w, h, BAR_BG)
        self.rect(x, y, max(w * frac, 0.012), h, color)

    def stackbar(self, x, y, w, h, fracs, colors, total=1.0):
        """adjacent coloured segments; returns the total drawn width."""
        xx = x
        for frac, col in zip(fracs, colors):
            ww = w * frac / total
            self.rect(xx, y, ww, h, col)
            xx += ww
        return xx - x

    def glyph(self, x, y, ch, color, size=11, w=0.20, h=0.18):
        self.text(x, y, w, h, [[(ch, size, color, True)]],
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def header(b: Board):
    b.text(0.55, 0.13, 12.25, 0.40,
           [[("Validated High-Throughput Screening", 20,
              BLUE_DK, True)]])
    b.text(0.55, 0.52, 12.25, 0.24,
           [[("with simplex-constrained surrogates for CALPHAD "
              "phase-fraction prediction in nine alloy systems", 10,
              GREY, False)]])
    ln = b.rect(0.55, 0.80, 12.25, 0.012, LINE_SOFT)
    return ln


def panel(b: Board, x, accent, number, title):
    w = 4.00
    b.rect(x, 0.98, w, 6.02, WHITE, line=accent, line_w=1.0)
    b.rect(x, 0.98, w, 0.46, accent)
    b.oval(x + 0.12, 1.07, 0.27, 0.27, WHITE)
    b.text(x + 0.12, 1.075, 0.27, 0.27, [[(number, 14, accent, True)]],
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    b.text(x + 0.47, 1.06, w - 0.60, 0.30, [[(title, 14, WHITE, True)]],
           anchor=MSO_ANCHOR.MIDDLE)
    return w


def panel1(b: Board, x):
    panel(b, x, BLUE, "1", "CALPHAD data \u2192 constrained heads")

    # -- data card
    b.rect(x + 0.15, 1.56, 3.70, 1.06, TINT_BLUE, line=BLUE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 1.63, 2.55, 0.20,
           [[("Five Fe-based ternary systems", 10.5, INK, True)]])
    b.text(x + 0.28, 1.84, 2.55, 0.30,
           [[("Fe\u2013Cr\u2013Ni \u00b7 Fe\u2013Cr\u2013Mn \u00b7 "
              "Fe\u2013Cr\u2013Mo", 8, GREY, False)],
            [("Fe\u2013Cr\u2013V \u00b7 Fe\u2013Mn\u2013Ni", 8, GREY,
              False)]], line_spacing=1.05)
    b.text(x + 0.28, 2.14, 2.55, 0.16,
           [[("pycalphad + MatCalc steel DB", 8, GREY, False)]])
    b.text(x + 0.28, 2.29, 2.55, 0.16,
           [[("44,397 validated equilibria \u00b7 3 seeds", 8, GREY, False)]])
    b.text(x + 0.28, 2.44, 2.55, 0.16,
           [[("targets: N\u1d4f \u2265 0,  \u03a3 N\u1d4f = 1", 8, BLUE_DK,
              True)]])

    # mini composition simplex (equilateral triangle, Fe apex at top)
    tx, ty = x + 3.30, 2.10          # triangle centre
    r = 0.40
    A = (tx, ty - r * 0.52)                       # Fe apex (top)
    Bp = (tx - r * 0.95, ty + r * 0.62)           # x2 corner (bottom left)
    Cp = (tx + r * 0.95, ty + r * 0.62)           # x3 corner (bottom right)
    tri = b.slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                   Inches(Bp[0]), Inches(A[1]),
                                   Inches(Cp[0] - Bp[0]),
                                   Inches(Bp[1] - A[1]))
    tri.fill.solid()
    tri.fill.fore_color.rgb = WHITE
    tri.line.color.rgb = BLUE
    tri.line.width = Pt(0.75)
    tri.shadow.inherit = False
    # three sample points inside the simplex (barycentric)
    for (wA, wB, wC) in ((0.50, 0.30, 0.20), (0.30, 0.20, 0.50),
                         (0.60, 0.15, 0.25)):
        px_ = wA * A[0] + wB * Bp[0] + wC * Cp[0]
        py_ = wA * A[1] + wB * Bp[1] + wC * Cp[1]
        b.oval(px_ - 0.022, py_ - 0.022, 0.044, 0.044, BAR_BLUE)
    b.text(tx - 0.20, A[1] - 0.17, 0.40, 0.14,
           [[("Fe", 6.5, GREY, True)]], align=PP_ALIGN.CENTER)
    b.text(Bp[0] - 0.30, Bp[1] + 0.01, 0.28, 0.14,
           [[("x\u2082", 6.5, GREY, True)]], align=PP_ALIGN.RIGHT)
    b.text(Cp[0] - 0.14, Cp[1] + 0.01, 0.28, 0.14,
           [[("x\u2083", 6.5, GREY, True)]], align=PP_ALIGN.CENTER)

    # -- model card
    b.rect(x + 0.15, 2.74, 3.70, 0.94, TINT_BLUE, line=BLUE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 2.81, 3.44, 0.20,
           [[("One fixed protocol, three seeds", 10.5, INK, True)]])
    b.text(x + 0.28, 3.03, 3.44, 0.16,
           [[("constrained MLP (3 \u00d7 192): six output heads", 8, GREY, False)]])
    b.text(x + 0.28, 3.20, 3.44, 0.32,
           [[("sigmoid \u00b7 residue \u00b7 sparsemax \u00b7 softmax \u00b7",
              8, GREY, False)],
            [("sigmoid/\u03a3 \u00b7 renorm", 8, GREY, False)]],
           line_spacing=1.05)
    b.text(x + 0.28, 3.50, 3.44, 0.16,
           [[("baselines: ridge \u00b7 k-NN \u00b7 XGBoost \u00b7 random forest",
              8, GREY, False)]])

    # -- closure card: stacked phase-fraction bars
    b.rect(x + 0.15, 3.80, 3.70, 1.38, TINT_BLUE, line=BLUE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 3.87, 3.44, 0.20,
           [[("Simplex closure: predicted phase fractions", 10.5, INK,
              True)]])
    b.text(x + 0.28, 4.07, 1.60, 0.14,
           [[("one test point", 7, GREY, False)]])
    # phase-colour legend for the stacked bars
    lx = x + 2.05
    for lab, col in [("BCC", PH_BCC), ("FCC", PH_FCC), ("LIQ", PH_LIQ),
                     ("\u03c3", PH_SIG)]:
        b.rect(lx, 4.105, 0.09, 0.07, col)
        lw = 0.12 if lab == "\u03c3" else 0.20
        b.text(lx + 0.11, 4.065, lw, 0.14, [[(lab, 6.5, GREY, False)]])
        lx += 0.11 + lw + 0.09
    segs = [(0.55, PH_BCC), (0.30, PH_FCC), (0.10, PH_LIQ), (0.05, PH_SIG)]
    bar_x, bar_w, bar_h = x + 1.38, 1.75, 0.13
    rows = [
        ("on-simplex heads", segs, "\u03a3 = 1", GREEN_OK),
        ("post-hoc renorm", segs, "\u03a3 = 1", GREEN_OK),
        ("unconstrained sigmoid",
         [(0.59, PH_BCC), (0.32, PH_FCC), (0.11, PH_LIQ), (0.05, PH_SIG)],
         "\u03a3 \u2248 1.07", RED_BAD),
    ]
    yy = 4.26
    for lab, s, sumlab, col in rows:
        b.text(x + 0.28, yy - 0.005, 1.05, 0.16, [[(lab, 7, GREY, False)]])
        b.stackbar(bar_x, yy, bar_w, bar_h, [f for f, _ in s],
                   [c for _, c in s])
        b.text(x + 3.30, yy - 0.005, 0.50, 0.16,
               [[(sumlab, 7.5, col, True)]])
        yy += 0.19
    b.text(x + 0.28, yy + 0.01, 3.44, 0.30,
           [[("closure \u2260 simplex validity: the residue head closes the "
              "sum yet emits negative fractions on up to 1.2 % of cells",
              7.5, GREY, False)]], line_spacing=1.05)

    # -- takeaway with verdict glyphs
    b.rect(x + 0.15, 5.24, 3.70, 0.62, TINT_BLUE, line=BLUE, line_w=0.75,
           round_=True)
    b.glyph(x + 0.24, 5.30, "\u2713", GREEN_OK, size=12)
    b.text(x + 0.46, 5.30, 3.30, 0.24,
           [[("structural constraints: \u03a3 \u0177 = 1 at no accuracy cost",
              8.5, BLUE_DK, True)]])
    b.glyph(x + 0.24, 5.56, "\u2717", RED_BAD, size=12)
    b.text(x + 0.46, 5.56, 3.30, 0.24,
           [[("no single architecture wins all five systems", 8.5, INK,
              True)]])

    # -- accuracy mini-bars (best constrained MLP vs random forest)
    b.text(x + 0.28, 5.96, 2.20, 0.16,
           [[("Test MAE, best MLP vs random forest", 8, GREY, True)]])
    # legend
    b.rect(x + 2.52, 6.00, 0.14, 0.07, BAR_BLUE)
    b.text(x + 2.68, 5.96, 0.42, 0.14, [[("MLP", 6.5, GREY, False)]])
    b.rect(x + 3.08, 6.00, 0.14, 0.07, BAR_ORANGE)
    b.text(x + 3.24, 5.96, 0.55, 0.14, [[("forest", 6.5, GREY, False)]])
    rows = [
        ("Fe\u2013Cr\u2013Ni", 0.0106, 0.0180, "MLP wins"),
        ("Fe\u2013Cr\u2013Mn", 0.0074, 0.0133, "MLP wins"),
        ("Fe\u2013Cr\u2013Mo", 0.0126, 0.0123, "tie"),
        ("Fe\u2013Cr\u2013V", 0.0149, 0.0137, "tie"),
        ("Fe\u2013Mn\u2013Ni", 0.0140, 0.0058, "RF wins"),
    ]
    yy = 6.15
    vmax = 0.0180
    for lab, mlp, rf, verdict in rows:
        b.text(x + 0.28, yy, 0.85, 0.14, [[(lab, 7, GREY, False)]])
        b.hbar(x + 1.16, yy + 0.02, 1.90, 0.05, mlp / vmax, BAR_BLUE)
        b.hbar(x + 1.16, yy + 0.085, 1.90, 0.05, rf / vmax, BAR_ORANGE)
        b.text(x + 3.12, yy + 0.02, 0.72, 0.14, [[(verdict, 7, GREY, True)]])
        yy += 0.165


def panel2(b: Board, x):
    panel(b, x, GREEN, "2", "Spatial holdout vs. controls")

    # -- protocol card with domain strips
    b.rect(x + 0.15, 1.56, 3.70, 1.30, TINT_GREEN, line=GREEN, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 1.63, 3.44, 0.20,
           [[("Two spatial protocols + controls", 10.5, INK, True)]])

    # strip 1: contiguous interior-band holdout (band removed from the
    # middle; training data remain on both sides)
    b.text(x + 0.28, 1.86, 3.44, 0.14,
           [[("contiguous interior-band holdout", 7.5, GREY, True)]])
    sx, sw = x + 0.28, 2.30
    b.rect(sx, 2.02, sw, 0.10, GREEN)
    b.rect(sx + sw * 0.40, 2.02, sw * 0.20, 0.10, WHITE, line=RED_BAD,
           line_w=0.75)
    b.text(sx + sw + 0.06, 1.99, 1.10, 0.16,
           [[("T \u2208 [1200, 1400] K", 6.5, GREY, False)]])

    # strip 2: strict one-sided extrapolation (training only on the left)
    b.text(x + 0.28, 2.20, 3.44, 0.14,
           [[("strict one-sided extrapolation", 7.5, GREY, True)]])
    b.rect(sx, 2.36, sw * 0.55, 0.10, GREEN)
    b.rect(sx + sw * 0.55, 2.36, sw * 0.45, 0.10, WHITE, line=RED_BAD,
           line_w=0.75)
    b.text(sx + sw + 0.06, 2.33, 1.10, 0.16,
           [[("x\u2082: \u2264 0.25 \u2192 > 0.35", 6.5, GREY, False)]])

    # legend for the strips
    b.rect(sx, 2.56, 0.12, 0.07, GREEN)
    b.text(sx + 0.16, 2.52, 0.45, 0.14, [[("train", 6.5, GREY, False)]])
    b.rect(sx + 0.72, 2.56, 0.12, 0.07, WHITE, line=RED_BAD, line_w=0.5)
    b.text(sx + 0.88, 2.52, 1.20, 0.14,
           [[("held out / far side", 6.5, GREY, False)]])
    b.text(sx + 2.05, 2.52, 1.40, 0.14,
           [[("+ size-matched random control", 6.5, GREEN_DK, True)]],
           align=PP_ALIGN.RIGHT)

    # -- penalty card (composition band, Fe-Cr-Ni)
    b.rect(x + 0.15, 2.98, 3.70, 1.42, TINT_GREEN2, line=GREEN, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 3.05, 3.44, 0.20,
           [[("Composition-band penalty, Fe\u2013Cr\u2013Ni", 10.5, INK, True)]])
    b.text(x + 0.28, 3.26, 3.44, 0.16,
           [[("band MAE \u00f7 matched random-control MAE", 7.5, GREY, False)]])
    rows = [
        ("constrained MLP", 1.61, BAR_GREEN),
        ("XGBoost", 2.24, BAR_ORANGE),
        ("random forest", 3.11, BAR_RED),
    ]
    yy = 3.46
    for lab, val, col in rows:
        b.text(x + 0.28, yy, 1.30, 0.16, [[(lab, 8, GREY, False)]])
        b.hbar(x + 1.62, yy + 0.03, 1.55, 0.10, val / 3.2, col)
        b.text(x + 3.22, yy, 0.55, 0.16, [[(f"{val:.2f}\u00d7", 8, INK, True)]])
        yy += 0.21
    b.text(x + 0.28, yy + 0.01, 3.44, 0.30,
           [[("held-out band lies 99.3\u201399.4 % inside the training convex "
              "hull \u2014 interior generalization, not extrapolation",
              7.5, GREY, False)]], line_spacing=1.05)

    # -- extrapolation card with degradation sketch
    b.rect(x + 0.15, 4.52, 3.70, 1.40, TINT_GREEN, line=GREEN, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 4.59, 3.44, 0.20,
           [[("Strict one-sided extrapolation", 10.5, INK, True)]])

    # schematic: MAE vs. the extrapolated coordinate; all families rise
    # sharply past the training range, the MLP least of all
    ax0, ay0 = x + 0.34, 5.44          # axis origin (bottom-left)
    aw, ah = 1.95, 0.58
    b.arrow(ax0, ay0, ax0, ay0 - ah - 0.03, color=GREY, width=1.0)
    b.arrow(ax0, ay0, ax0 + aw + 0.10, ay0, color=GREY, width=1.0)
    # training-range boundary
    bx = ax0 + aw * 0.55
    b.rect(bx, ay0 - ah, 0.008, ah, LIGHT)

    def curve(pts, color, width=1.4):
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            conn = b.slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0),
                Inches(x1), Inches(y1))
            conn.line.color.rgb = color
            conn.line.width = Pt(width)
            conn.shadow.inherit = False

    def to_px(fx, fy):
        return (ax0 + fx * aw, ay0 - fy * ah)

    # MLP: gentle rise; trees: steep rise
    mlp = [to_px(0.05, 0.10), to_px(0.30, 0.12), to_px(0.55, 0.16),
           to_px(0.75, 0.30), to_px(0.97, 0.52)]
    tree = [to_px(0.05, 0.14), to_px(0.30, 0.16), to_px(0.55, 0.26),
            to_px(0.75, 0.58), to_px(0.97, 0.95)]
    curve(mlp, BAR_GREEN)
    curve(tree, BAR_RED)
    b.text(ax0 + aw + 0.12, ay0 - 0.95 * ah - 0.04, 0.62, 0.14,
           [[("trees", 6.5, RED_BAD, True)]], align=PP_ALIGN.LEFT)
    b.text(ax0 + aw + 0.12, ay0 - 0.52 * ah - 0.04, 0.62, 0.14,
           [[("MLP", 6.5, GREEN_DK, True)]], align=PP_ALIGN.LEFT)
    # axis annotations below the plot
    b.text(x + 0.29, ay0 + 0.03, 0.30, 0.14,
           [[("MAE", 6.5, GREY, False)]], align=PP_ALIGN.LEFT)
    b.text(bx - 0.55, ay0 + 0.03, 1.10, 0.14,
           [[("training range edge", 6, GREY, False)]],
           align=PP_ALIGN.CENTER)
    b.text(x + 2.50, ay0 + 0.03, 0.90, 0.14,
           [[("coordinate", 6.5, GREY, False)]], align=PP_ALIGN.LEFT)

    b.text(x + 0.28, 5.64, 3.44, 0.28,
           [[("all evaluated families degrade sharply beyond the training "
              "range; the MLP degrades least \u2014 yet none extrapolates "
              "reliably", 7, GREEN_DK, True)]], line_spacing=1.05)

    # -- takeaway
    b.rect(x + 0.15, 5.98, 3.70, 0.88, TINT_GREEN2, line=GREEN, line_w=0.75,
           round_=True)
    b.text(x + 0.28, 6.06, 3.44, 0.72,
           [[("Constrained MLPs degrade more gracefully than trees under "
              "contiguous region holdout: the region-matched penalty is "
              "smaller for MLPs in 9 of 10 band \u00d7 system "
              "combinations. Graceful degradation is not reliable "
              "extrapolation.", 8.5, GREEN_DK, True)]], line_spacing=1.1)


def panel3(b: Board, x):
    panel(b, x, PURPLE, "3", "Regression vs. phase detection")

    # -- concept card: the same prediction viewed two ways
    b.rect(x + 0.15, 1.56, 3.70, 1.10, TINT_PURPLE, line=PURPLE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 1.63, 3.44, 0.20,
           [[("Two distinct tasks, one prediction", 10.5, INK, True)]])
    # regression side: continuous stacked fractions
    b.text(x + 0.28, 1.88, 1.60, 0.14,
           [[("regression (MAE)", 7, GREY, True)]])
    b.stackbar(x + 0.28, 2.04, 1.55, 0.12,
               [0.55, 0.30, 0.10, 0.05],
               [PH_BCC, PH_FCC, PH_LIQ, PH_SIG])
    # phase-colour legend under the regression bar
    lx = x + 0.28
    for lab, col in [("BCC", PH_BCC), ("FCC", PH_FCC), ("LIQ", PH_LIQ),
                     ("\u03c3", PH_SIG)]:
        b.rect(lx, 2.225, 0.08, 0.06, col)
        lw = 0.10 if lab == "\u03c3" else 0.19
        b.text(lx + 0.10, 2.185, lw, 0.13, [[(lab, 6, GREY, False)]])
        lx += 0.10 + lw + 0.08
    # detection side: binary presence calls
    b.text(x + 2.10, 1.88, 1.60, 0.14,
           [[("detection (F\u2081)", 7, GREY, True)]])
    for i, (col, on) in enumerate([(PH_BCC, True), (PH_FCC, True),
                                   (PH_LIQ, True), (PH_SIG, False)]):
        cx_ = x + 2.10 + i * 0.24
        b.oval(cx_, 2.03, 0.14, 0.14, col if on else WHITE,
               line=None if on else LIGHT)
        if not on:
            b.text(cx_ - 0.02, 2.015, 0.18, 0.16,
                   [[("\u00d7", 8, LIGHT, True)]], align=PP_ALIGN.CENTER,
                   anchor=MSO_ANCHOR.MIDDLE)
    # phase labels under the detection circles (same colour order)
    for i, lab in enumerate(["BCC", "FCC", "LIQ", "\u03c3"]):
        cx_ = x + 2.10 + i * 0.24
        b.text(cx_ + 0.07 - 0.12, 2.185, 0.24, 0.13,
               [[(lab, 6, GREY, False)]], align=PP_ALIGN.CENTER)
    b.text(x + 0.28, 2.35, 3.44, 0.30,
           [[("rare phases are cheap in aggregate MAE \u2014 zero-fraction "
              "rows contribute almost no error", 8, PURPLE_DK, True)]],
           line_spacing=1.05)

    # -- Fe-Mn-Ni case card: the ranking reverses
    b.rect(x + 0.15, 2.78, 3.70, 1.62, TINT_PURPLE, line=PURPLE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 2.85, 3.44, 0.20,
           [[("Fe\u2013Mn\u2013Ni: the ranking reverses", 10.5, INK, True)]])

    # regression row: MAE bars (lower is better)
    b.text(x + 0.28, 3.10, 3.44, 0.16,
           [[("test MAE (lower is better)", 7.5, GREY, True)]])
    b.text(x + 0.28, 3.28, 0.40, 0.14, [[("RF", 7, GREY, False)]])
    b.hbar(x + 0.70, 3.30, 1.90, 0.07, 0.0058 / 0.0140, BAR_ORANGE)
    b.text(x + 2.66, 3.27, 0.60, 0.14, [[("0.0058", 7, INK, True)]])
    b.text(x + 0.28, 3.44, 0.40, 0.14, [[("MLP", 7, GREY, False)]])
    b.hbar(x + 0.70, 3.46, 1.90, 0.07, 1.0, BAR_BLUE)
    b.text(x + 2.66, 3.43, 0.60, 0.14, [[("0.0140", 7, INK, True)]])
    b.text(x + 3.28, 3.28, 0.55, 0.30,
           [[("forest\nwins", 7, GREY, True)]], align=PP_ALIGN.CENTER,
           line_spacing=1.0)

    # detection row: rare-phase presence calls
    b.text(x + 0.28, 3.66, 3.44, 0.16,
           [[("rare-phase detection (MNNI \u00b7 MNNI2 \u00b7 ALPHA_MN)",
              7.5, GREY, True)]])
    b.text(x + 0.28, 3.86, 0.40, 0.14, [[("RF", 7, GREY, False)]])
    for i, on in enumerate([True, True, False]):
        b.oval(x + 0.72 + i * 0.24, 3.85, 0.14, 0.14,
               BAR_PURPLE if on else WHITE,
               line=None if on else LIGHT)
    b.text(x + 1.50, 3.85, 1.80, 0.14,
           [[("resolves 2 of 3", 7, GREY, False)]])
    b.text(x + 0.28, 4.04, 0.40, 0.14, [[("MLP", 7, GREY, False)]])
    for i in range(3):
        b.oval(x + 0.72 + i * 0.24, 4.03, 0.14, 0.14, WHITE, line=LIGHT)
        b.text(x + 0.70 + i * 0.24, 4.015, 0.18, 0.16,
               [[("\u00d7", 8, LIGHT, True)]], align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE)
    b.text(x + 1.50, 4.03, 1.80, 0.14,
           [[("detects none", 7, GREY, False)]])
    b.text(x + 3.28, 3.86, 0.55, 0.30,
           [[("forest\nwins", 7, GREY, True)]], align=PP_ALIGN.CENTER,
           line_spacing=1.0)

    b.text(x + 0.28, 4.22, 3.44, 0.16,
           [[("the best regressor need not be the best detector", 8,
              PURPLE_DK, True)]])

    # -- threshold card with threshold chips
    b.rect(x + 0.15, 4.52, 3.70, 1.06, TINT_PURPLE, line=PURPLE, line_w=0.5,
           round_=True)
    b.text(x + 0.28, 4.59, 3.44, 0.20,
           [[("Robust to the presence threshold", 10.5, INK, True)]])
    for i, thr in enumerate(["10\u207b\u2074", "10\u207b\u00b3",
                             "10\u207b\u00b2"]):
        cx_ = x + 0.30 + i * 0.52
        chip = b.rect(cx_, 4.84, 0.44, 0.20, WHITE, line=PURPLE, line_w=0.75,
                      round_=True)
        b.text(cx_, 4.855, 0.44, 0.17, [[(thr, 8, PURPLE_DK, True)]],
               align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    b.text(x + 1.90, 4.84, 1.85, 0.20,
           [[("gap persists at all three", 7.5, GREY, False)]],
           anchor=MSO_ANCHOR.MIDDLE)
    b.text(x + 0.28, 5.12, 3.44, 0.30,
           [[("although the detector ranking flips with the threshold on "
              "Fe\u2013Cr\u2013Mo", 8, GREY, False)]], line_spacing=1.05)

    # -- takeaway
    b.rect(x + 0.15, 5.70, 3.70, 1.16, TINT_AMBER, line=PURPLE, line_w=0.75,
           round_=True)
    b.text(x + 0.28, 5.78, 3.44, 1.00,
           [[("Report both. ", 9, PURPLE_DK, True),
             ("Low phase-fraction MAE does not imply good phase-presence "
              "detection \u2014 for alloy design, detecting a rare phase "
              "matters as much as estimating its fraction.", 9, INK, False)]],
           line_spacing=1.1)


def footer(b: Board):
    b.text(0.55, 7.12, 12.25, 0.24,
           [[("Fixed-protocol benchmark \u00b7 6 heads + 2 mechanism probes "
              "\u00b7 4 baselines \u00b7 5 ternary + 4 extension systems "
              "\u00b7 88,542 equilibria "
              "\u00b7 seeds 42 / 123 / 2024 \u00b7 negative results reported "
              "in full", 8, LIGHT, False)]],
           align=PP_ALIGN.CENTER)


def main():
    b = Board()
    header(b)
    panel1(b, 0.28)
    panel2(b, 4.67)
    panel3(b, 9.06)
    # inter-panel arrows
    b.arrow(4.30, 3.99, 4.65, 3.99, color=GREY, width=1.75)
    b.arrow(8.69, 3.99, 9.04, 3.99, color=GREY, width=1.75)
    footer(b)
    b.prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
