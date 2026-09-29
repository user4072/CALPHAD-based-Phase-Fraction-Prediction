"""Geometric QC for the graphical abstract PPTX.

Estimates rendered text size with DejaVu Sans (slightly wider than Arial,
so a conservative overflow detector) and checks:
  1. every text box's content fits its box (with wrap estimation);
  2. no two non-text shapes overlap unless one is contained in the other;
  3. text boxes do not spill outside their parent card/panel;
  4. everything stays inside the slide with a margin.

Usage: py -3.12 paper/qc_graphical_abstract.py
"""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent.parent
PPTX = ROOT / "Graphical_Abstract_fe_surrogate.pptx"

FONT_PATH = None
# prefer Arial (the actual rendering font); fall back to DejaVu Sans
for cand in (r"C:\Windows\Fonts\arial.ttf",):
    if Path(cand).exists():
        FONT_PATH = cand
        break
if FONT_PATH is None:
    for f in font_manager.fontManager.ttflist:
        if f.name == "DejaVu Sans" and f.style == "normal":
            FONT_PATH = f.fname
            break

_cache = {}


def text_w_in(s: str, pt: float) -> float:
    """Rendered width of a string at a given point size, in inches."""
    key = int(pt * 4)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONT_PATH, key)  # px at 288 dpi/4
    font = _cache[key]
    px = font.getlength(s)
    return px / 288.0  # 288 px per inch at this scale


def para_lines(text: str, pt: float, box_w_in: float) -> int:
    if not text:
        return 1
    w = text_w_in(text, pt)
    return max(1, int(w // box_w_in) + (1 if w % box_w_in > 1e-6 else 0))


def main():
    prs = Presentation(PPTX)
    slide = prs.slides[0]
    W = prs.slide_width / 914400
    H = prs.slide_height / 914400
    warns = []

    rects, texts = [], []
    for sh in slide.shapes:
        x = sh.left / 914400
        y = sh.top / 914400
        w = sh.width / 914400
        h = sh.height / 914400
        rec = (sh.shape_id, sh.name, x, y, w, h)
        if sh.has_text_frame and sh.text_frame.text.strip():
            texts.append((rec, sh.text_frame))
        else:
            rects.append(rec)

    # 1. slide bounds
    for sid, name, x, y, w, h in rects + [t[0] for t in texts]:
        if x < 0.02 or y < 0.02 or x + w > W - 0.02 or y + h > H - 0.02:
            warns.append(f"OUT-OF-SLIDE {name} at ({x:.2f},{y:.2f}) "
                         f"size ({w:.2f}x{h:.2f})")

    # 2. text fit
    for (sid, name, x, y, w, h), tf in texts:
        total_h = 0.0
        for para in tf.paragraphs:
            runs = [(r.text, r.font.size.pt if r.font.size else 10)
                    for r in para.runs]
            line = "".join(t for t, _ in runs)
            pt = max((p for _, p in runs), default=10)
            ls = para.line_spacing if para.line_spacing else 1.0
            n = para_lines(line, pt, w) if line else 1
            total_h += n * (pt / 72.0) * 1.22 * ls
        if total_h > h + 0.03:
            warns.append(f"TEXT-OVERFLOW {name}: est {total_h:.2f}in > box "
                         f"{h:.2f}in at ({x:.2f},{y:.2f})")

    # 3. text boxes spilling outside their container card
    for (sid, name, x, y, w, h), tf in texts:
        # find the smallest rect that contains the text origin
        best = None
        for rid, rname, rx, ry, rw, rh in rects:
            if rx <= x + 0.01 and ry <= y + 0.01 and \
               rx + rw >= x + 0.05 and ry + rh >= y + 0.05:
                if best is None or rw * rh < best[3] * best[4]:
                    best = (rname, rx, ry, rw, rh)
        if best is None:
            continue
        rname, rx, ry, rw, rh = best
        # estimate content height
        total_h = 0.0
        for para in tf.paragraphs:
            line = "".join(r.text for r in para.runs)
            pt = max((r.font.size.pt for r in para.runs if r.font.size),
                     default=10)
            ls = para.line_spacing if para.line_spacing else 1.0
            n = para_lines(line, pt, w) if line else 1
            total_h += n * (pt / 72.0) * 1.22 * ls
        if y + total_h > ry + rh + 0.02:
            warns.append(f"SPILL {name} below {rname}: text ends "
                         f"{y + total_h:.2f} > card bottom {ry + rh:.2f}")

    # 4. text boxes overlapping rects they are not contained in
    GLYPHS = set("\u00d7\u2713\u2717")
    for (sid, name, x, y, w, h), tf in texts:
        # intentional glyph marks (x / check / cross) drawn on shapes
        if set(tf.text.strip()) <= GLYPHS:
            continue
        for rid, rname, rx, ry, rw, rh in rects:
            ox = min(x + w, rx + rw) - max(x, rx)
            oy = min(y + h, ry + rh) - max(y, ry)
            if ox > 0.03 and oy > 0.03:
                inside = (x >= rx - 0.01 and y >= ry - 0.01 and
                          x + w <= rx + rw + 0.01 and y + h <= ry + rh + 0.01)
                if not inside:
                    warns.append(f"TEXT-ON-RECT {name} x {rname}: "
                                 f"{ox:.2f}x{oy:.2f}")

    # 5. card/card overlap (rects of similar size, non-contained);
    # connectors are lines/curves whose adjacent segments always share
    # endpoints, so their bounding boxes overlap by design -- skip them
    boxes = [r for r in rects if not r[1].startswith("Connector")]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ax0, ay0, ax1, ay1 = a[2], a[3], a[2] + a[4], a[3] + a[5]
            bx0, by0, bx1, by1 = b[2], b[3], b[2] + b[4], b[3] + b[5]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > 0.03 and oy > 0.03:
                contained = (ax0 >= bx0 - 0.01 and ay0 >= by0 - 0.01 and
                             ax1 <= bx1 + 0.01 and ay1 <= by1 + 0.01) or \
                            (bx0 >= ax0 - 0.01 and by0 >= ay0 - 0.01 and
                             bx1 <= ax1 + 0.01 and by1 <= ay1 + 0.01)
                if not contained:
                    warns.append(f"RECT-OVERLAP {a[1]} x {b[1]}: "
                                 f"{ox:.2f}x{oy:.2f}")

    if warns:
        for wmsg in warns:
            print("QC:", wmsg)
    else:
        print("QC clean")
    print(f"({len(rects)} rects, {len(texts)} text boxes)")


if __name__ == "__main__":
    main()
