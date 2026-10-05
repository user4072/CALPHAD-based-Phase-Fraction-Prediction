"""Build paper/Manuscript (5).docx for CMS submission.

Copies paper/Manuscript (4).docx (NEVER modified) into a new file and applies,
in the same simple writing style:
  - new Validated title, new abstract (<=250 words), extended keywords, Highlights
  - Stoco related-work paragraph + [33]; pycalphad [34] and MatCalc [35] citations
  - scope-extension datasets (methods), gated-head methods, region/ideal protocol
  - new Results subsections: gated remedy (Table 11, Fig 9), uncertainty triage,
    region-matched + ideal-form (Table 12), extensions (Table 13),
    experimental anchor (Fig 10), screening (Tables 14-15, Figs 11-13)
  - discussion appends, conclusions appends, references [33]-[37]
  - Figure 1 image replaced with the current pipeline figure + new caption

Tables are parsed from paper/tab/*.tex (generated, audit-checked numbers).
Usage: py -3.12 paper/make_manuscript5.py
"""
import os
import re
import shutil
import sys
from copy import deepcopy

from docx import Document
from docx.shared import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
SRC = os.path.join(PAPER, "Manuscript (4).docx")
DST = os.path.join(PAPER, "Manuscript (5).docx")
FIG = os.path.join(PAPER, "figures")

from manuscript5_text import (NEW_TITLE, NEW_ABSTRACT, HIGHLIGHTS, NEW_KEYWORDS,
    SCOPE_PARA, CONTRIB_PARA, STOCO_PARA, EXT_DATA_PARA, GATED_METHOD_PARA,
    SPATIAL_APPEND, FIG1_CAPTION, GATED_H2, GATED_PS, TABLE11_CAP, FIG9_CAP,
    UNC_H2, UNC_P1, UNC_P2, REGION_H2, REGION_P1, REGION_P2, TABLE12_CAP,
    EXT_H2, EXT_PS, TABLE13_CAP, ANCHOR_H2, ANCHOR_P, FIG10_CAP, SCREEN_H2,
    SCREEN_PS, TABLE14_CAP, FIG11_CAP, TABLE15_CAP, FIG13_CAP, FIG12_CAP,
    DISCUSS_ADD1, DISCUSS_ADD2, LIMIT_ADD, CONC_ADD, NEW_REFS, PROJ_PARA,
    TABLE_SHIFT_CAP)


# --------------------------------------------------------------------------
# LaTeX table parsing
# --------------------------------------------------------------------------
SUP = {"-9": "\u207b\u2079", "-8": "\u207b\u2078", "-3": "\u207b\u00b3",
       "-4": "\u207b\u2074", "-2": "\u207b\u2072", "-1": "\u207b\u00b9",
       "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
       "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
       "8": "\u2078", "9": "\u2079"}
SUB = {"2": "\u2082", "3": "\u2083"}


def clean_cell(s):
    s = re.sub(r"\bXGB\b", "XGBoost", s)
    s = s.strip()
    bold = False
    m = re.fullmatch(r"\\textbf\{(.+)\}", s)
    if m:
        bold = True
        s = m.group(1)
    s = s.replace("\\times", "\u00d7").replace("\\le", "\u2264").replace(
        "\\ge", "\u2265").replace("\\Sigma", "\u03a3").replace(
        "\\Delta", "\u0394").replace("\\max", "max").replace(
        "\\sim", "~").replace("\\%", "%").replace("\\&", "&").replace("\\!", "").replace(
        "~", " ")
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = re.sub(r"\^\{(-?\d+)\}",
               lambda m: SUP.get(m.group(1), "^" + m.group(1)), s)
    s = re.sub(r"\^(\d+)",
               lambda m: "".join(SUP.get(d, "^" + d) for d in m.group(1))
               if all(d in SUP for d in m.group(1)) else "^" + m.group(1), s)
    s = re.sub(r"_(\d)",
               lambda m: SUB.get(m.group(1), "_" + m.group(1)), s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text(rm|bf|it)\{([^}]*)\}", r"\2", s)
    s = s.replace("$", "")
    s = re.sub(r"\\[a-zA-Z]+\s?", "", s)
    s = re.sub(r"[{}]", "", s)
    s = s.replace("(see Section sec:remedy)", "(see text)")
    return re.sub(r"\s+", " ", s).strip(), bold


def parse_tab(fname):
    """Parse a generated tab/*.tex into a list of blocks.

    Returns list of dicts: {kind: 'header'|'rows'|'note', cells: [...],
    spans: [...], bolds: [...]}, in file order.
    """
    raw = open(os.path.join(PAPER, "tab", fname), encoding="utf-8").read()
    lines = [ln for ln in raw.split("\n")
             if ln.strip() and not ln.strip().startswith("%")]
    blocks = []
    cur = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("\\begin{tabular") or s.startswith("\\footnotesize") or \
                s.startswith("\\scriptsize") or s.startswith("\\setlength") or \
                s.startswith("\\resizebox") or s.startswith("\\end{tabular") or \
                s.startswith("\\vspace"):
            continue
        if s.startswith("\\toprule") or s.startswith("\\midrule") or \
                s.startswith("\\bottomrule") or s.startswith("\\cmidrule"):
            if cur:
                blocks.append(cur)
                cur = None
            continue
        if s.startswith("\\addlinespace"):
            if cur:
                blocks.append(cur)
                cur = None
            continue
        if "\\\\" not in s:
            continue
        row = s.split("\\\\")[0]
        parts = [p.strip() for p in row.split("&")]
        note = None
        cells, spans, bolds = [], [], []
        for p in parts:
            mm = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", p)
            if mm:
                t, b = clean_cell(mm.group(2))
                cells.append(t)
                spans.append(int(mm.group(1)))
                bolds.append(b)
            else:
                t, b = clean_cell(p)
                cells.append(t)
                spans.append(1)
                bolds.append(b)
        if cur is None:
            cur = {"cells": [], "spans": [], "bolds": []}
        # extend column-wise: blocks accumulate rows sharing column count
        if cur["cells"] and len(cells) != len(cur["cells"][0]):
            blocks.append(cur)
            cur = {"cells": [], "spans": [], "bolds": []}
        cur["cells"].append(cells)
        cur["spans"].append(spans)
        cur["bolds"].append(bolds)
    if cur:
        blocks.append(cur)
    return blocks


# --------------------------------------------------------------------------
# docx helpers
# --------------------------------------------------------------------------
def find_idx(doc, start):
    for i, p in enumerate(doc.paragraphs):
        if p.text.startswith(start):
            return i
    raise RuntimeError("anchor not found: %r" % start[:60])


def insert_para(doc, idx, text, style="Normal", after=True, before=False):
    p = doc.add_paragraph(style=style)
    p.text = text
    ref = doc.paragraphs[idx]._p
    if before:
        ref.addprevious(p._p)
    elif after:
        ref.addnext(p._p)
    else:
        ref.addprevious(p._p)
    return p


def set_cell(cell, text, bold=False, size=9):
    cell.text = ""
    par = cell.paragraphs[0]
    run = par.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)


def make_table(doc, rows, header_bold=True, size=9, style="Table Grid"):
    """Build a table element (not placed). rows: list of (cells, spans, bolds)."""
    ncols = max(sum(sp) for _, sp, _ in rows)
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.style = style
    t.autofit = True
    for i, (cells, spans, bolds) in enumerate(rows):
        ci = 0
        for txt, sp, b in zip(cells, spans, bolds):
            if sp > 1:
                first = t.rows[i].cells[ci]
                last = t.rows[i].cells[ci + sp - 1]
                first.merge(last)
                for k in range(ci + 1, ci + sp):
                    t.rows[i].cells[k].text = ""
                set_cell(first, txt, bold=(b or (header_bold and i == 0)),
                         size=size)
            else:
                set_cell(t.rows[i].cells[ci], txt,
                         bold=(b or (header_bold and i == 0)), size=size)
            ci += sp
    doc.element.body.remove(t._tbl)  # detach; caller places it
    return t


def insert_table(doc, idx, rows, header_bold=True, size=9, style="Table Grid",
                 after=True):
    """rows: list of (cells, spans, bolds). Returns table."""
    t = make_table(doc, rows, header_bold, size, style)
    ref = doc.paragraphs[idx]._p
    if after:
        ref.addnext(t._tbl)
    else:
        ref.addprevious(t._tbl)
    return t


def add_figure(doc, idx, img, caption, width=6.0):
    p_img = insert_para(doc, idx, "", after=True)
    p_img.add_run().add_picture(img, width=Inches(width))
    p_cap = insert_para(doc, idx + 1, caption, after=True)
    return p_cap


def renumber_captions_and_refs(doc):
    """Renumber Table/Figure captions in physical order and remap all refs."""
    tmap, fmap = {}, {}
    tn = fn = 0
    for p in doc.paragraphs:
        m = re.match(r"^(Table|Figure) (\d+)\.", p.text.strip())
        if m:
            if m.group(1) == "Table":
                tn += 1
                tmap[int(m.group(2))] = tn
            else:
                fn += 1
                fmap[int(m.group(2))] = fn

    def sub(m):
        word, num = m.group(1), int(m.group(2))
        mp = tmap if word.startswith("Table") else fmap
        return "%s %d" % (word, mp.get(num, num))

    for p in doc.paragraphs:
        t = p.text.strip()
        if re.match(r"^\d+\.\t", t):
            continue  # reference list
        new = re.sub(r"\b(Tables?|Figures?) (\d+)", sub, p.text)
        if new != p.text:
            p.text = new
    return tmap, fmap


def main():
    assert os.path.exists(SRC), SRC
    shutil.copyfile(SRC, DST)
    doc = Document(DST)

    # ---- 1. title / abstract / keywords / highlights ----
    if not doc.paragraphs[0].text.startswith("Simplex Constraints"):
        raise RuntimeError("unexpected title paragraph")
    doc.paragraphs[0].text = NEW_TITLE

    ai = find_idx(doc, "Abstract: ")
    doc.paragraphs[ai].text = NEW_ABSTRACT
    assert len(NEW_ABSTRACT.split()) <= 250, len(NEW_ABSTRACT.split())
    # (4) stores the abstract body in the following paragraph: remove the stale
    # leftover so the old abstract does not survive next to the new one
    stale = find_idx(doc, "Machine-learning surrogates massively accelerate")
    assert stale == ai + 1, (stale, ai)
    p_el = doc.paragraphs[stale]._p
    p_el.getparent().remove(p_el)

    ki = find_idx(doc, "Keywords:")
    doc.paragraphs[ki].text = NEW_KEYWORDS
    for j, h in enumerate(HIGHLIGHTS):
        assert len(h) <= 85, h
    insert_para(doc, ki, "Highlights:", after=True)
    for j, h in enumerate(reversed(HIGHLIGHTS)):
        insert_para(doc, ki + 1, "\u2022  " + h, after=True)

    # ---- 2. intro: scope + contributions + Stoco ----
    si = find_idx(doc, "To address these interconnected challenges")
    doc.paragraphs[si].text = SCOPE_PARA

    ci = find_idx(doc, "The specific contributions")
    doc.paragraphs[ci].text = CONTRIB_PARA

    sec = find_idx(doc, "A secondary challenge in materials informatics")
    insert_para(doc, sec, STOCO_PARA, before=True)

    # ---- 3. methods: citations ----
    mi = find_idx(doc, "All thermodynamic equilibrium calculations")
    t = doc.paragraphs[mi].text
    assert "integrated with pycalphad to extract" in t
    t = t.replace("open MatCalc steel database (mc_fe_v2.062)",
                  "open MatCalc steel database [35] (mc_fe_v2.062)")
    t = t.replace("integrated with pycalphad to extract",
                  "integrated with pycalphad [34] to extract")
    doc.paragraphs[mi].text = t

    # ---- 4. Figure 1: replace image + caption ----
    shape = doc.inline_shapes[0]
    rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    with open(os.path.join(FIG, "fig0_workflow.png"), "rb") as f:
        doc.part.related_parts[rid]._blob = f.read()
    f1 = find_idx(doc, "Figure 1. Study pipeline.")
    doc.paragraphs[f1].text = FIG1_CAPTION

    # ---- 5. scope-extension datasets (after Figure 2 caption) ----
    f2 = find_idx(doc, "Figure 2. Statistical overview")
    insert_para(doc, f2, "Scope-Extension Datasets", style="Heading 3",
                after=True)
    insert_para(doc, f2 + 1, EXT_DATA_PARA, after=True)

    # ---- 6. gated-head methods (before Baseline Models) ----
    bidx = next(i for i, p in enumerate(doc.paragraphs)
                if p.text.strip() == "Baseline Models"
                and "Heading" in p.style.name)
    insert_para(doc, bidx, GATED_METHOD_PARA, before=True)
    insert_para(doc, bidx, "Presence-Gated Two-Stage Head", style="Heading 3",
                before=True)

    # ---- 7. spatial protocols: region-matched + ideal-form ----
    sp = find_idx(doc, "A fundamental challenge in measuring spatial")
    doc.paragraphs[sp].text = doc.paragraphs[sp].text + SPATIAL_APPEND

    # ---- 8. gated remedy results (after Figure 4 caption) ----
    f4 = find_idx(doc, "Figure 4. Analysis of the regressor")
    insert_para(doc, f4, GATED_H2, style="Heading 2", after=True)
    for j, par in enumerate(GATED_PS):
        insert_para(doc, f4 + 1 + j, par, after=True)
    k = f4 + 1 + len(GATED_PS)
    insert_para(doc, k, TABLE_SHIFT_CAP, after=True)
    gs = [b for b in parse_tab("gatedshift.tex") if b["cells"]]
    ghead, grows, gnotes = [], [], []
    for b in gs:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            if len(r) == 6 and r[0] == "System":
                ghead.append((r, s_, bb))
            elif len(r) == 6:
                grows.append((r, s_, bb))
            elif len(r) == 1:
                gnotes.append((r, s_, bb))
    assert len(grows) == 8, len(grows)
    insert_table(doc, k + 1, ghead + grows + gnotes, after=True)
    insert_para(doc, k + 1, TABLE11_CAP, after=True)
    rem = parse_tab("remedy.tex")
    blocks = [b for b in rem if b["cells"]]
    head_rows = []
    data_rows = []
    note_rows = []
    for b in blocks:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            if len(r) == 3 and any("macro-AUPRC" in c for c in r):
                head_rows.append(
                    (["", "macro-AUPRC (3 seeds)", "gated/MLP MAE ratio"],
                     [1, 6, 1], [False, True, True]))
            elif len(r) == 8 and r[0].startswith("System"):
                head_rows.append((r, s_, bb))
            elif len(r) == 8 and (r[0].startswith("Fe") or r[0].startswith("Cr")):
                data_rows.append((r, s_, bb))
            elif "Ratio $>1$" in str(r) or "gated head costs" in str(r):
                note_rows.append((["Ratio >1: the gated head costs MAE "
                                   "relative to MLP renorm."],
                                  [8], [False]))
    table11 = head_rows + data_rows + note_rows
    assert len(data_rows) == 7, len(data_rows)
    insert_table(doc, k + 2, table11, after=True)
    add_figure(doc, k + 3, os.path.join(FIG, "fig8_remedy.png"), FIG9_CAP)

    # ---- 8b. projection result (end of closure section) ----
    gh = next(i for i, p in enumerate(doc.paragraphs)
              if p.text.strip() == "The Regressor" + chr(8211) + "Detector Gap"
              and "Heading" in p.style.name)
    insert_para(doc, gh, PROJ_PARA, before=True)

    # ---- 9. uncertainty triage (before Spatial Generalization heading) ----
    sidx = next(i for i, p in enumerate(doc.paragraphs)
                if p.text.strip() == "Spatial Generalization and Extrapolation Limits"
                and "Heading" in p.style.name)
    insert_para(doc, sidx, UNC_H2, style="Heading 2", before=True)
    insert_para(doc, sidx, UNC_P2, before=True)
    insert_para(doc, sidx, UNC_P1, before=True)

    # ---- 10. region-matched + ideal-form (after Figure 6 caption) ----
    f6 = find_idx(doc, "Figure 6. Strict one-sided extrapolation evaluation.")
    insert_para(doc, f6, REGION_H2, style="Heading 2", after=True)
    insert_para(doc, f6 + 1, REGION_P1, after=True)
    insert_para(doc, f6 + 2, REGION_P2, after=True)
    insert_para(doc, f6 + 3, TABLE12_CAP, after=True)
    reg = [b for b in parse_tab("region.tex") if b["cells"]]
    # panel A: header + 10 rows; then notes; then panel B 10 rows
    ahead, arows, bhead, brows, notes = [], [], [], [], []
    seen_b = False
    for b in reg:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            if len(r) == 5 and r[0] == "Band":
                ahead.append((r, s_, bb))
            elif len(r) == 5 and ("band" in r[0] or r[0] == ""):
                (brows if seen_b else arows).append((r, s_, bb))
            elif "Ideal-form size-matched" in str(r):
                seen_b = True
                notes.append((["Ideal-form size-matched control (identical test "
                               "rows, equal training size): band-in/band-out "
                               "ratio H/I, training-size effect I/main in "
                               "parentheses."],
                              [5], [False]))
            elif "Region-matched ratio" in str(r):
                notes.insert(0, (["Region-matched ratio of mean MAEs: holdout "
                                  "mean MAE on band rows divided by "
                                  "in-distribution mean MAE on the identical "
                                  "rows; per-seed ratios in parentheses "
                                  "(seeds 42/123/2024)."],
                                 [5], [False]))
    assert len(arows) == 10 and len(brows) == 10, (len(arows), len(brows))
    table12 = (ahead[:1]
               + [(["(A) Region-matched ratios"], [5], [True])]
               + arows
               + [(["(B) Ideal-form control: H/I (I/main)"], [5], [True])]
               + brows + notes)
    insert_table(doc, f6 + 4, table12, after=True)

    # ---- 11. extensions + anchor + screening (end of Results) ----
    # forward-chained cursor: each emission goes right after the previous one,
    # starting after the Figure 8 caption (tables are not paragraphs, so the
    # cursor is the raw lxml element).
    f8 = find_idx(doc, "Figure 8. Ground-truth and predicted phase-fraction")
    disc = next(i for i, p in enumerate(doc.paragraphs)
                if p.text.strip() == "Discussion" and "Heading" in p.style.name)
    assert f8 < disc
    cur_el = doc.paragraphs[f8]._p

    def emit_para(text, style="Normal"):
        nonlocal cur_el
        p = doc.add_paragraph(style=style)
        p.text = text
        cur_el.addnext(p._p)
        cur_el = p._p
        return p

    def emit_table(rows):
        nonlocal cur_el
        t = make_table(doc, rows)
        cur_el.addnext(t._tbl)
        cur_el = t._tbl
        return t

    def emit_fig(img, caption, width=6.0):
        nonlocal cur_el
        p_img = doc.add_paragraph()
        p_img.add_run().add_picture(img, width=Inches(width))
        cur_el.addnext(p_img._p)
        cur_el = p_img._p
        p_cap = doc.add_paragraph()
        p_cap.text = caption
        cur_el.addnext(p_cap._p)
        cur_el = p_cap._p
    emit_para(EXT_H2, style="Heading 2")
    for par in EXT_PS:
        emit_para(par)
    emit_para(TABLE13_CAP)
    ext = [b for b in parse_tab("extension.tex") if b["cells"]]
    panels, names = [], ["(A) System and dataset statistics",
                         "(B) Test MAE, mean over 3 seeds",
                         "(C) Spatial holdout and extrapolation",
                         "(D) Phase-set validation"]
    # flatten rows; a new panel starts at each in-tex (X) label row
    flat = []
    for b in ext:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            flat.append((r, s_, bb))
    cur_panel = []
    for r, s_, bb in flat:
        if len(r) == 1 and re.match(r"^\([ABCD]\)", r[0]):
            if cur_panel:
                panels.append(cur_panel)
            cur_panel = []
            continue
        cur_panel.append((r, s_, bb))
    if cur_panel:
        panels.append(cur_panel)
    assert len(panels) == 4, len(panels)
    for i, prows in enumerate(panels):
        emit_table([([names[i]], [len(prows[0][0])], [True])] + prows)

    emit_para(ANCHOR_H2, style="Heading 2")
    emit_para(ANCHOR_P)
    emit_fig(os.path.join(FIG, "fig9_anchor.png"), FIG10_CAP)

    emit_para(SCREEN_H2, style="Heading 2")
    emit_para(SCREEN_PS[0])
    emit_fig(os.path.join(FIG, "fig10_screen.png"), FIG12_CAP)
    for par in SCREEN_PS[1:]:
        emit_para(par)
    emit_para(TABLE14_CAP)
    scr = [b for b in parse_tab("screen.tex") if b["cells"]]
    srows = []
    for b in scr:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            if len(r) == 2 and r[0] != "Quantity":
                srows.append((r, s_, bb))
    shead = (["Quantity", "Value"], [1, 1], [True, True])
    emit_table([shead] + srows)
    emit_fig(os.path.join(FIG, "fig11_operating.png"), FIG11_CAP)
    emit_para(TABLE15_CAP)
    sq = [b for b in parse_tab("screenq.tex") if b["cells"]]
    qrows = []
    for b in sq:
        for r, s_, bb in zip(b["cells"], b["spans"], b["bolds"]):
            if len(r) == 2 and r[0] != "Quantity":
                qrows.append((r, s_, bb))
    emit_table([shead] + qrows)
    emit_fig(os.path.join(FIG, "fig12_screenq.png"), FIG13_CAP)

    # ---- 12. discussion appends ----
    di = find_idx(doc, "However, this graceful in-hull degradation must not be")
    doc.paragraphs[di].text = doc.paragraphs[di].text + DISCUSS_ADD1

    fi = find_idx(doc, "This distinction emphasizes the role of surrogates")
    doc.paragraphs[fi].text = doc.paragraphs[fi].text + DISCUSS_ADD2

    li = find_idx(doc, "Another significant limitation of our current")
    doc.paragraphs[li].text = doc.paragraphs[li].text + LIMIT_ADD

    # ---- 13. conclusions ----
    ex = find_idx(doc, "Extrapolation Limits:")
    for j, par in enumerate(CONC_ADD):
        insert_para(doc, ex + j, par, after=True)

    # ---- 14b. renumber tables/figures in physical order ----
    tmap, fmap = renumber_captions_and_refs(doc)
    print("table map:", tmap)
    print("figure map:", fmap)
    # inherited twin-ref the regex cannot see ("Figures 5b and 5c")
    for p in doc.paragraphs:
        if "Figures 6b and 5c" in p.text:
            p.text = p.text.replace("Figures 6b and 5c", "Figures 6b and 6c")

    # ---- 14. references ----
    ri = find_idx(doc, "32.\tAditya")
    for j, ref in enumerate(NEW_REFS):
        insert_para(doc, ri + j, ref, after=True)

    doc.save(DST)
    print("wrote", DST)

    # ---- verification ----
    d2 = Document(DST)
    full = "\n".join(p.text for p in d2.paragraphs)
    words = len(full.split())
    print("words (prose, excl. tables):", words)
    seq = []
    for p in d2.paragraphs:
        m = re.match(r"^(Table|Figure) (\d+)\.", p.text.strip())
        if m:
            seq.append((m.group(1), int(m.group(2))))
    print("caption order:", seq)
    assert [n for k, n in seq if k == "Table"] == list(range(1, 17)), seq
    assert [n for k, n in seq if k == "Figure"] == list(range(1, 14)), seq
    for tag in ["[33]", "[34]", "[35]", "[36]", "[37]",
                "88,542", "501,501", "330,000", "0.91\u20130.99",
                "mean 2.01", "Highlights"]:
        assert tag in full, tag
    assert "Trustworthy high-throughput" not in full
    assert "44,397 equilibria from five" not in full
    assert len(d2.inline_shapes) == 13, len(d2.inline_shapes)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()