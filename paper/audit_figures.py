"""Quality gate for the paper figures. Non-zero exit = defect remains.

Imports paper/figures.py, monkey-patches `save` so figures stay open, then
checks each axes for defects that are invisible in a thumbnail but obvious in
print: text drawn over data, legends covering data, artists outside the axes,
fonts below the journal floor, panels too small to read, and colliding tick
labels.

Text-over-data is tested against actual marker and vertex positions, not
against the bounding hull of a collection. Testing the hull reports a
collision for any annotation placed inside a panel whose points happen to
span it -- which is most log-scale panels -- and that false positive is worse
than no check at all, because the natural response is to move a correctly
placed label somewhere worse.

Usage: py -3.12 paper/audit_figures.py
"""
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figures as F

FONT_FLOOR = 6.0            # Elsevier: below ~6 pt nothing survives reduction
MIN_AXES_IN = (1.05, 0.72)  # drawable area, inches
MAX_TICKS = 14
PAD = 1.5                   # pixels of clearance demanded around text

problems = []
open_figs = []


def _capture(fig, stem):
    open_figs.append((stem, fig))


F.save = _capture


def bbox_overlap(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if dx <= 0 or dy <= 0:
        return 0.0
    smaller = min(a.width * a.height, b.width * b.height)
    return (dx * dy) / smaller if smaller > 0 else 0.0


def data_marks(ax, renderer):
    """Return (points, patch_bboxes): pixel positions of markers and vertices,
    and bounding boxes of solid patches (bars, wedges) where the hull is the
    mark."""
    pts = []
    for coll in ax.collections:
        if getattr(coll, "_is_guide", False):
            continue
        try:
            off = coll.get_offsets()
        except Exception:
            continue
        if off is None or len(off) == 0:
            continue
        p = ax.transData.transform(np.asarray(off))
        pts.append(p[np.isfinite(p).all(axis=1)])
    for line in ax.lines:
        if getattr(line, "_is_guide", False):
            continue
        xy = line.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        p = ax.transData.transform(np.asarray(xy))
        p = p[np.isfinite(p).all(axis=1)]
        if len(p) > 1:
            # densify so a long segment counts along its length, not just at
            # its ends
            seg = np.linspace(0, 1, 24)[:, None]
            for a, b in zip(p[:-1], p[1:]):
                pts.append(a + seg * (b - a))
        elif len(p):
            pts.append(p)
    boxes = []
    for patch in ax.patches:
        if getattr(patch, "_is_guide", False):
            continue
        try:
            bb = patch.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append(bb)
    points = np.vstack(pts) if pts else np.empty((0, 2))
    return points, boxes


def hits(bb, points, boxes, frac=0.55):
    """True if a text bbox lands on a marker/vertex or substantially on a bar."""
    if len(points):
        inside = ((points[:, 0] > bb.x0 - PAD) & (points[:, 0] < bb.x1 + PAD)
                  & (points[:, 1] > bb.y0 - PAD) & (points[:, 1] < bb.y1 + PAD))
        if inside.any():
            return True
    return any(bbox_overlap(bb, b) > frac for b in boxes)


def audit(stem, fig):
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.get_window_extent(renderer)
    for ai, ax in enumerate(fig.get_axes()):
        if getattr(ax, "_is_colorbar", False):
            continue
        where = f"{stem} axes[{ai}]"

        ax_bb = ax.get_window_extent(renderer)
        w_in, h_in = ax_bb.width / fig.dpi, ax_bb.height / fig.dpi
        if w_in < MIN_AXES_IN[0] or h_in < MIN_AXES_IN[1]:
            problems.append(f"{where}: drawable area {w_in:.2f}x{h_in:.2f} in "
                            f"below {MIN_AXES_IN[0]}x{MIN_AXES_IN[1]}")

        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            labels = [t for t in axis.get_ticklabels() if t.get_text().strip()]
            if len(labels) > MAX_TICKS and not getattr(ax, "_dense_ticks_ok", False):
                problems.append(f"{where}: {len(labels)} {name} tick labels "
                                f"(max {MAX_TICKS})")
            # Axis-aligned bboxes of rotated text overlap even when the glyphs
            # do not, so only upright labels are collision-checked.
            if axis is ax.xaxis and labels and all(
                    abs(t.get_rotation() % 180) < 1e-6 for t in labels):
                boxes = []
                for t in labels:
                    try:
                        boxes.append(t.get_window_extent(renderer))
                    except Exception:
                        pass
                boxes.sort(key=lambda b: b.x0)
                if any(b.x0 < a.x1 - 0.5 for a, b in zip(boxes, boxes[1:])):
                    problems.append(f"{where}: x tick labels overlap")

        texts = [t for t in ax.texts if t.get_text().strip()]
        for t in texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text().strip() and t.get_fontsize() < FONT_FLOOR:
                problems.append(f"{where}: {t.get_fontsize():.1f} pt text "
                                f"{t.get_text()[:28]!r} below {FONT_FLOOR} pt floor")
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_ticklabels():
                if t.get_text().strip() and t.get_fontsize() < FONT_FLOOR:
                    problems.append(f"{where}: {t.get_fontsize():.1f} pt tick label "
                                    f"below {FONT_FLOOR} pt floor")

        points, boxes = data_marks(ax, renderer)
        for t in texts:
            try:
                tb = t.get_window_extent(renderer)
            except Exception:
                continue
            if hits(tb, points, boxes):
                problems.append(f"{where}: text {t.get_text()[:28]!r} sits on data")
            if tb.x0 < ax_bb.x0 - 1 or tb.x1 > ax_bb.x1 + 1 \
                    or tb.y0 < ax_bb.y0 - 1 or tb.y1 > ax_bb.y1 + 1:
                problems.append(f"{where}: text {t.get_text()[:28]!r} outside axes")

        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(renderer)
            outside = lb.y1 <= ax_bb.y0 + 1 or lb.y0 >= ax_bb.y1 - 1
            if not outside and hits(lb, points, boxes, frac=0.42):
                problems.append(f"{where}: legend sits on data")
            if lb.x0 < fig_bb.x0 - 1 or lb.x1 > fig_bb.x1 + 1:
                problems.append(f"{where}: legend clipped by the figure edge")


def main():
    for name in ["fig1_dataset", "fig2_heads", "fig3_fields",
                 "fig4_perphase", "fig5_holdout", "fig6_failures",
                 "fig7_extrap"]:
        getattr(F, name)()
    for stem, fig in open_figs:
        fig.canvas.draw()
        audit(stem, fig)
        plt.close(fig)
    if problems:
        uniq = sorted(set(problems))
        print(f"{len(uniq)} distinct figure defect(s):")
        for p in uniq:
            print(f"  - {p}")
        return 1
    print(f"{len(open_figs)} figures audited, no defects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
