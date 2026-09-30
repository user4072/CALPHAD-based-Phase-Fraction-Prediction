"""Generate the Word (.docx) version of the CMS manuscript.

Rationale (same as the reference project): neither pandoc nor LibreOffice is
available in this environment, so the document is written directly with
python-docx. Tables become native Word tables, figures are embedded from
paper/figures/*.png, and mathematics is rendered as Unicode.

Every numeric value is read from the stored result artefacts -- the same
files paper/tables.py and paper/figures.py use -- so this document and the
LaTeX version cannot drift apart.

Usage: <python with python-docx> paper/make_docx.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")
D = os.path.join(ROOT, "data", "raw")
FIGDIR = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(ROOT, "paper", "paper_cms.docx")

SEEDS = [42, 123, 2024]
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
LBL = {"fecrni": "Fe\u2013Cr\u2013Ni", "fecrmn": "Fe\u2013Cr\u2013Mn",
       "fecrmo": "Fe\u2013Cr\u2013Mo", "fecrv": "Fe\u2013Cr\u2013V",
       "femnni": "Fe\u2013Mn\u2013Ni"}

# Unicode shorthands
PM, TIMES, LEQ, APPROX, MINUS = "\u00b1", "\u00d7", "\u2264", "\u2248", "\u2212"
GEQ, IN, LAMBDA, TAU, ALPHA, DELTA = "\u2265", "\u2208", "\u03bb", "\u03c4", "\u03b1", "\u0394"
SIG, PHI, SIGMA_PH = "\u03a3", "\u03c6", "\u03c3"
SUP = {str(i): c for i, c in enumerate("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")}
SUP["-"] = "\u207b"
SUP["k"] = "\u1d4f"
SUB = {str(i): c for i, c in enumerate("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")}
SUB.update({"k": "\u2096", "i": "\u1d62", "m": "\u2098", "n": "\u2099",
            "r": "\u1d63", "s": "\u209b", "t": "\u209c"})


def sup(s: str) -> str:
    return "".join(SUP[c] for c in s)


def sub(s: str) -> str:
    return "".join(SUB[c] for c in s)


def sci(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "\u2014"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    return f"{mant:.{digits}f}{TIMES}10{sup(str(e))}"


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
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def phases(system):
    probe = json.load(open(os.path.join(D, f"{system}_probe.json")))
    names = sorted(probe["active_counts"].keys())
    return names, [f"NP_{p}" for p in names], probe


BH = {}
for s in SYS:
    p = os.path.join(M, f"block_holdout_{s}.json")
    if os.path.exists(p):
        BH[s] = json.load(open(p))


def bagg(s, tag, field="mean_mae"):
    d = BH.get(s, {})
    v = [d[f"{tag}_s{k}"][field] for k in SEEDS if f"{tag}_s{k}" in d]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def mae_cell(m, s, bold=False):
    if np.isnan(m):
        return ("\u2014", False)
    return (f"{m:.4f} ({int(round(s * 1e4)):d})", bold)


# ── document scaffolding ─────────────────────────────────────────────────────
def new_doc() -> Document:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.30
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
    for name, size, bold in (("Heading 1", 14, True), ("Heading 2", 12, True),
                             ("Heading 3", 11, True)):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(size)
        s.font.bold = bold
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.space_before = Pt(12)
        s.paragraph_format.space_after = Pt(4)
    return doc


def para(doc, text="", *, align=None, italic=False, bold=False, size=None,
         space_after=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.italic = italic
        r.bold = bold
        if size:
            r.font.size = Pt(size)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def body(doc, text: str):
    return para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)


def finding(doc, head: str, text: str):
    """Run-in paragraph head, mirroring \\paragraph{...}."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(head + ". ")
    r.bold = True
    p.add_run(text)
    return p


def equation(doc, text: str, number: str):
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    t.columns[0].width = Inches(5.4)
    t.columns[1].width = Inches(0.6)
    c0, c1 = t.rows[0].cells
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p0.add_run(text)
    r.italic = True
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.add_run(number)
    for row in t.rows:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.space_before = Pt(4)
            cell.paragraphs[0].paragraph_format.space_after = Pt(4)
    return t


def shade(cell, hexfill="F2F2F2"):
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hexfill)
    cell._tc.get_or_add_tcPr().append(el)


def table(doc, headers, rows, caption, *, note=None, bold_cells=None,
          left_cols=(0,)):
    """Native Word table. bold_cells: set of (row, col) to bold (0-based,
    excluding the header row)."""
    para(doc, caption, size=10, space_after=3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        shade(hdr[i])
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i in left_cols
                           else WD_ALIGN_PARAGRAPH.CENTER)
            r = p.add_run(str(v))
            r.font.size = Pt(9)
            if bold_cells and (ri, i) in bold_cells:
                r.bold = True
    if note:
        para(doc, note, size=8.5, space_after=10)
    return t


def figure(doc, stem: str, caption: str, width_in: float = 6.4):
    png = os.path.join(FIGDIR, f"{stem}.png")
    if not os.path.exists(png):
        raise SystemExit(f"missing figure: {png}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(png, width=Inches(width_in))
    cap = para(doc, caption, size=9.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    cap.paragraph_format.space_after = Pt(12)
    return cap


# ── content ──────────────────────────────────────────────────────────────────
def build() -> None:
    doc = new_doc()

    # title block
    para(doc, "Simplex Constraints, Spatial Generalization, and the "
              "Regressor\u2013Detector Gap in Machine-Learning Surrogates "
              "for CALPHAD Phase-Fraction Prediction",
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=15)
    para(doc, "Anonymous", align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
    para(doc, "Submitted to Computational Materials Science",
         align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=10, space_after=14)

    # abstract
    doc.add_heading("Abstract", level=1)
    body(doc,
         f"Machine-learning surrogates can accelerate CALPHAD equilibrium "
         f"calculations by several orders of magnitude. Surrogate models that "
         f"map composition and temperature to equilibrium "
         f"phase fractions must respect the simplex constraint {SIG}{SUB['k']} "
         f"N{sup('k')} = 1, yet the literature offers no consensus on how that "
         f"constraint should be imposed. We compare six output heads \u2014 "
         f"sigmoid, softmax, sigmoid/{SIG}, clip-then-renormalise, sparsemax, "
         f"and a residue closure head \u2014 against four classical baselines "
         f"across five Fe-based ternary systems "
         f"({', '.join(LBL[s] for s in SYS)}) solved with pycalphad from the "
         f"open MatCalc steel database, totalling 44,397 mass-balance-validated "
         f"equilibria; power normalisation and a sum-to-one loss penalty are "
         f"evaluated additionally as mechanism probes on {LBL['fecrni']}. "
         f"Structurally constrained heads satisfy sum-to-one closure to "
         f"{LEQ}5{TIMES}10{sup('-8')} at no accuracy cost; the unconstrained "
         f"sigmoid violates it by 6\u20138%, and the residue head, which "
         f"enforces closure but not non-negativity, produces negative "
         f"fractions on up to 1.2% of cells. No single evaluated architecture "
         f"consistently dominates across the five systems: constrained "
         f"multilayer perceptrons lead on systems with dense smooth "
         f"coexistence ({LBL['fecrni']}, {LBL['fecrmn']}), while random "
         f"forests win where sharp phase boundaries dominate "
          f"({LBL['femnni']}), and paired-bootstrap tests "
          f"confirm statistical ties in the remaining systems "
          f"({LBL['fecrmo']}, {LBL['fecrv']}). Two spatial protocols, each "
          f"with a size-matched random control and a region-matched "
          f"reanalysis, separate where points are held out "
          f"from the penalty of having less data. On contiguous interior "
         f"holdout bands \u2014 which a Delaunay check shows to lie "
         f"99.3\u201399.4% inside the convex hull of the remaining training "
         f"points \u2014 the random forest pays a 3.11{TIMES} penalty against "
         f"1.61{TIMES} for the constrained MLP on the {LBL['fecrni']} "
         f"composition band, whereas on the temperature band both pay "
         f"1.31{TIMES}: constrained MLPs degrade more gracefully than trees "
         f"under contiguous composition-region holdout. Under strict "
         f"one-sided out-of-range extrapolation along a single coordinate "
         f"(train x\u2082 {LEQ} 0.25, test x\u2082 > 0.35; train T {LEQ} "
          f"1200 K, test T > 1400 K) all evaluated model families degrade "
          f"by roughly an order of magnitude, and the temperature protocol "
          f"is additionally limited by the closed phase basis, as liquid is "
          f"absent from the near-side training data in four of the five "
          f"systems; the constrained MLP achieves the lowest absolute "
          f"far-side error in all five systems, although absolute "
          f"extrapolation performance remains poor. Phase-presence detection "
         f"degrades faster than regression under distribution shift: "
         f"F{SUB['1']} collapses on the held-out temperature band for all "
         f"evaluated models even where MAE barely moves, and the "
         f"qualitative regressor\u2013detector gap is stable across presence "
         f"thresholds 10{sup('-4')}\u201310{sup('-2')}, although the "
         f"detector ranking flips with the threshold on {LBL['fecrmo']}. "
         f"Three negative results are "
         f"reported in full: under the evaluated configuration sparsemax "
         f"exhibits severe seed sensitivity; a sum-to-one penalty degrades "
         f"MAE by 8\u201316{TIMES}; and the residue head is "
         f"1.3\u20132.5{TIMES} worse than the best constrained head in all five "
         f"systems.")
    para(doc, "Keywords: CALPHAD; surrogate model; phase fraction; simplex "
              "constraint; spatial generalisation; iron alloys",
         size=10, italic=True, space_after=12)

    # 1. Introduction
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Phase fractions as a regression target", level=2)
    body(doc,
         "The CALPHAD method predicts equilibrium phase assemblages from "
         "assessed thermodynamic databases (Saunders and Miodownik, 1998; "
         "Lukas et al., 2007). Given a database, the equilibrium at fixed "
         "composition, temperature and pressure follows from constrained "
         "Gibbs-energy minimisation, and the output of interest in alloy "
         "design is frequently the vector of phase fractions y = (N\u00b9, "
         "\u2026, N\u1d37), where K is the number of phases that can appear. "
         "This vector lies on the probability simplex,")
    equation(doc,
             f"y {IN} {DELTA}\u1d37\u207b\u00b9,    "
             f"{SIG}{{k=1..K}} N{sup('k')} = 1,    N{sup('k')} {GEQ} 0", "(1)")
    body(doc,
         "by the mass balance of the equilibrium solver. Any surrogate that "
         "predicts y should respect Eq. (1); a prediction that violates it is "
         "not merely inaccurate but physically inadmissible.")
    body(doc,
         "Machine-learning surrogates replace the per-point minimisation with "
         "a forward pass (Hart et al., 2021). Surrogates of phase-equilibrium "
         "and quasi-equilibrium calculations have been developed for process "
         "optimisation and phase-field coupling (Nentwich and Engell, 2019; "
         "Jiang et al., 2019), deep-learning surrogates now reproduce CALPHAD "
         "phase fractions fast enough to screen composition spaces spanning "
         "more than ten elements (Tahkola et al., 2026), machine-learning "
         "models of phase formation and transformation temperatures are "
         "widely used in alloy design (Feng et al., 2021; Lu et al., 2024), "
         "and uncertainty quantification for CALPHAD-based workflows is an "
         "active companion topic (Wang and Xiong, 2020). For phase fractions "
         "the modelling question decomposes into two: how to represent the "
         "output so that Eq. (1) holds, and whether the representation costs "
         "accuracy. Existing studies adopt a particular representation \u2014 "
         "typically softmax or post-hoc renormalisation \u2014 but to our "
         "knowledge no study has systematically compared simplex-constraint "
         "mechanisms across multiple alloy systems; the present benchmark "
         "addresses that gap.")
    body(doc,
         "One distinction frames everything that follows. A surrogate learns "
         "the map (x, T) \u2192 y(CALPHAD): it reproduces what the database's "
         "equilibrium routine returns, not what physical truth would return. "
         "\u201cValidated against CALPHAD\u201d therefore means \u201creproduces "
         "the database/solver outputs to solver precision\u201d, not "
         "\u201cindependently validates the thermodynamic database\u201d. Every "
         "accuracy claim in this paper is a statement about emulation "
         "fidelity, not thermodynamic truth.")

    doc.add_heading("1.2 Generalisation beyond the training split, and the "
                    "control problem", level=2)
    body(doc,
         "A second question concerns generalisation. The standard train/test "
         "split \u2014 random or cluster-stratified \u2014 tests interpolation: "
         "every test point has training neighbours in (x, T). Alloy design, "
         "however, queries points far from the computed ones, in two distinct "
         "senses. First, a contiguous interior region may be missing from the "
         "training data \u2014 a hole in the middle of the sampled domain, "
         "with training points on all sides. Second, a query may lie outside "
         "the range of one of the training coordinates altogether \u2014 "
         "strict, one-sided extrapolation along that coordinate. The two are "
         "different tests: in the first the test "
         "points typically remain inside the convex hull of the training "
         "set, in the second they do not. Both protocols share a confound: "
         "removing the held-out region also removes training data, so a "
         "degradation in held-out accuracy conflates \u201ccannot generalise "
         "there\u201d with \u201chad less to learn from\u201d. We resolve this "
         "with size-matched random controls \u2014 the same number of rows "
         "removed at random \u2014 which isolate the spatial penalty from the "
         "data-volume penalty. Because the control rows differ from the "
         "held-out rows, those ratios alone still conflate region difficulty "
         "with distribution shift; a region-matched reanalysis, which "
         "compares the same held-out rows with and without regional training "
         "data, provides the complementary reference (Section 7.1).")

    doc.add_heading("1.3 Contributions", level=2)
    body(doc,
         "The study is deliberately scoped to ternary systems, where "
         "ground-truth generation is inexpensive (a few CPU-hours per "
         "system), so that every protocol decision can be validated "
         "exhaustively; the methodological questions it isolates \u2014 "
         "constraint enforcement, generalisation protocols, and detection "
         "\u2014 arise identically in the higher-dimensional design spaces "
         "where surrogate speed becomes a practical necessity.")
    for i, c in enumerate([
        "A systematic comparison of simplex-constraint mechanisms for "
        "neural-network phase-fraction prediction \u2014 six heads across all "
        "five systems plus two mechanism probes \u2014 against four classical "
        "baselines (Section 4).",
        "Two spatial generalisation protocols with size-matched random "
        "controls and a region-matched reanalysis (Section 7): a contiguous "
        "interior-band holdout, which a "
        "convex-hull check shows to be an in-hull distribution shift rather "
        "than extrapolation, and a strict one-sided out-of-range "
        "extrapolation protocol (one coordinate at a time); together these "
        "separate spatial penalties from data-volume reductions. "
        "Constrained MLPs "
        "degrade more gracefully than trees under composition-region holdout "
        "and under the tested strict extrapolation protocols; the "
        "temperature-band penalty is largely a data effect.",
        f"A regressor\u2013detector gap: the model with the lowest MAE is not "
        f"necessarily the best phase-presence detector, detection degrades "
        f"faster than regression under distribution shift, and the gap is "
        f"stable across presence thresholds (Section 6).",
        "Three negative results reported in full \u2014 sparsemax, sum-to-one "
        "penalty, and residue closure \u2014 each informative about why the "
        "constraint mechanism matters (Section 9).",
    ], start=1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(c)

    body(doc,
         "Figure 1 summarises the pipeline end to end: the CALPHAD data "
         "generation, the fixed-protocol split and model families, and the "
         "three evaluation protocols they are subjected to.")
    figure(doc, "fig0_workflow",
           "Figure 1. Study pipeline. (1) CALPHAD data generation: five "
           "Fe-based ternary subsystems are extracted from the open MatCalc "
           "steel database, probe-driven phase sets fix the target "
           "dimensionality, structured sampling draws 11,220 raw design "
           "points per system (8,880 of which fall inside the valid "
           "sub-simplex and are submitted to the solver), and the "
           "equilibrium convergence and mass-balance gate accepts 44,397 "
           "equilibria in total. (2) A cluster-stratified 64/16/20 split "
           "feeds a constrained MLP with six output heads (plus two "
           "mechanism probes) and four classical baselines, all fitted "
           "under one fixed protocol and three seeds. (3) Three evaluation "
           f"protocols {MINUS} interpolation, contiguous interior-band "
           "holdout, and strict one-sided extrapolation, the two spatial "
           "ones each with a size-matched random control \u2014 plus two "
           "secondary analyses (ablations and boundary-error analysis, "
           "dashed arrows).")

    # 2. Data generation
    doc.add_heading("2. Data generation", level=1)
    doc.add_heading("2.1 Thermodynamic description", level=2)
    body(doc,
         "The source database is the open MatCalc steel database mc_fe_v2.062 "
         "(MatCalc, 2023), which cannot be parsed by pycalphad (Otis and Liu, "
         "2017) in its raw form. We parse it with a lenient custom parser, "
         "extract the five three-element subsystems, and apply minimal syntax "
         "repairs (reference-element blocks and malformed function ranges). "
         "Both a pruned database (constituents restricted to the system "
         "elements) and an unpruned database are built. Equivalence validation "
         f"on 100 re-solved points per system confirms the pruning is "
         f"bit-for-bit inert: maximum |{DELTA}N^{PHI}| = 0 and "
         f"|{DELTA}G{sub('m')}| = 0 J/mol for all five systems. "
         "All equilibrium calculations are performed with the open-source "
         "CALPHAD implementation pycalphad (Otis and Liu, 2017) against this "
         "database, so the surrogates emulate this specific solver\u2013database "
         "combination; their accuracy is correspondingly bounded by the "
         "database's assessments and the solver's convergence behaviour.")
    body(doc,
         "Eligible phase counts are 28 (Fe\u2013Cr\u2013Ni), 31 "
         "(Fe\u2013Cr\u2013Mn), 29 (Fe\u2013Cr\u2013Mo), 25 (Fe\u2013Cr\u2013V), "
         "and 29 (Fe\u2013Mn\u2013Ni). Pressure is fixed at 101,325 Pa "
         "throughout.")

    doc.add_heading("2.2 Probe-driven phase sets", level=2)
    body(doc,
         "Running equilibria with all eligible phases costs ~2.5 s per point. "
         "To streamline dataset generation, a phase-set pre-screening is "
         "employed: each system is independently probed with the full "
         "eligible set on 2,000 Dirichlet draws (RNG seed 7) to identify "
         "which phases are actively present, and only phases observed active "
         "during this probing phase are retained as regression targets, "
         "reducing the dimensionality (K) from the total eligible phases "
         "(Table 1). The probe points are fresh Dirichlet draws, "
         "independent of the dataset generated afterwards, so no label from "
         "any train/validation/test row enters the phase-set construction. "
         "This global preprocessing step fixes the output dimensionality "
         "for each system prior to any predictive modelling. The fidelity "
         "of this phase-set reduction is validated on an independent set of "
         "1,500 points per system by re-solving every point with the "
         "complete eligible phase set (25\u201331 phases depending on system) "
         "and comparing per-phase amounts and total molar Gibbs energies "
         "against the stored values. In four of the five systems the "
         "reduced-set and full-set equilibria agree to solver precision: "
         f"max |{DELTA}N^{PHI}| {LEQ} 2.5{TIMES}10{sup('-9')} and "
         f"max |{DELTA}G| {LEQ} 7.8{TIMES}10{sup('-5')} J/mol, "
         "and no excluded phase appears at any sampled point. "
         "Fe\u2013Cr\u2013Mo is the exception: at 3 of 1,500 points (0.2%), "
         "all located in the Mo-rich, low-temperature corner (x(Mo) "
         "\u2265 0.42, T \u2248 700 K) where BCC, \u03bc, and Laves phases "
         "compete closely, the full-set solve returns a different "
         f"assemblage involving LAVES_PHASE at up to |{DELTA}N^{PHI}| = 0.71 "
         f"and |{DELTA}G| = 6.3{TIMES}10{sup('2')} J/mol. "
         "Every phase involved in either solution is already in the "
         "probe-active target set, so these discrepancies reflect "
         "near-degenerate solver minima in this region rather than an "
         "incomplete target set: repeated full-set solves can return either "
         "minimum, and the exact failing-point count varies between "
         f"validation runs (1\u20133 of 1,500); no excluded phase exceeds "
         f"the 10{sup('-3')} threshold anywhere. The target vector is the "
         f"full probe-active phase-fraction vector with no occurrence "
         f"threshold, so {SIG}{sub('k')} y{sub('k')} = 1 holds to the solver's "
         f"mass-balance tolerance: |{SIG}{sub('k')} y{sub('k')} {MINUS} 1| "
         f"{LEQ} 3.3{TIMES}10{sup('-8')} across all test rows.")
    body(doc,
         "Phases stable in multiple composition sets (for example the BCC "
         "\u03b1/\u03b1\u2032 miscibility gap) are summed over their "
         "composition-set vertices into a single fraction column, matching "
         "the vertex accumulation performed by the equilibrium solver. The "
         "pre-screening probe and the structured design are different "
         "distributions over the Gibbs triangle: the probe retains any "
         "phase observed at least once under uniform Dirichlet draws, which "
         "is why MNNI2 \u2014 active in only 0.1% of probe draws and never in "
         "the structured design \u2014 enters the Fe\u2013Mn\u2013Ni target "
         "vector with zero active rows. A phase stable in less than roughly "
         "0.05% of the design space could evade both the 2,000-draw probe "
         "and the structured sampling and thus be absent from the closed "
         "target basis.")
    body(doc,
         "The probe does, however, cover the entire sampled domain, including "
         "regions that later become test rows. Phase-set discovery is "
         "therefore a global, transductive preprocessing step: the target "
         "dimensionality K is fixed with knowledge of the whole design space "
         "before any predictive split. This is deliberate \u2014 the "
         "surrogate's task is to predict fractions over a known phase basis, "
         "and discovering that basis is part of building the oracle "
         "interface, not part of the predictive task. It is not label "
         "leakage in the ordinary sense (no test label is seen), but it does "
         "mean that a phase active only in an unprobed region would be "
         "missed, and that the reported accuracies are conditional on this "
         "globally predefined phase basis being complete; the experiments do "
         "not assess discovery of previously unseen phases (Section 11).")

    rows = []
    for s in SYS:
        names, _, probe = phases(s)
        comps = LBL[s].split("\u2013")[1:]
        rows.append([LBL[s], f"{comps[0]}, {comps[1]}",
                     str(len(probe["eligible"])), str(len(names)),
                     ", ".join(names)])
    table(doc, ["System", "x\u2082, x\u2083", "Eligible", "K", "Target phases"],
          rows,
          "Table 1. Systems, probe-active phase counts K, and target phases. "
          "\u201cEligible\u201d is the number of phases in the full database "
          "subset; K is the number ever observed active in the probe.",
          left_cols=(0, 4))

    doc.add_heading("2.3 Sampling and acceptance", level=2)
    body(doc,
         "Design variables per system are (x\u2082, x\u2083, T) with "
         "x(Fe) = 1 \u2212 x\u2082 \u2212 x\u2083, temperature uniform in "
         "[700, 2000] K. Five deterministic sampling strategies (RNG seed 42) "
         "generate 11,220 raw design points: 6,000 Fe-weighted uniform draws, "
         "2,000 sigma-field-focused draws, 720 grid points on five isothermal "
         "slices, 1,500 liquidus-zone draws, and 1,000 near-pure-Fe draws. A "
         f"candidate is accepted only if the composition lies inside the "
         f"defined sub-simplex and the equilibrium satisfies mass balance "
         f"|{SIG}{sub('k')} N{sup('k')} {MINUS} 1| < 10{sup('-6')}. Of the "
         f"11,220 raw design draws per system, 2,340 fell outside the "
         f"sub-simplex and were discarded before any equilibrium solve; of "
         f"the remaining 8,880 valid candidates, 8,877 to 8,880 converged and "
         f"satisfied mass balance, the only solver rejections being three "
         f"non-converged points in Fe\u2013Cr\u2013Mo. The 11,220 to 8,880 "
         f"reduction is therefore a design-space filter rather than a "
         f"convergence filter, and the datasets contain no solver-driven "
         f"holes. Table 2 summarises the datasets.")
    body(doc,
         "The sampling distribution is deliberately structured, not uniform: "
         "the five strategies over-represent the Fe-rich corner, the sigma "
         "field, the liquidus zone, and isothermal grids. This is a design "
         "choice \u2014 phase boundaries and rare-phase regions carry most of "
         "the information a surrogate must learn, and uniform draws would "
         "undersample them. The consequence, which we keep in mind when "
         "interpreting results, is that all reported accuracies are "
         "expectations under this distribution; rankings can shift under a "
         "different prior (Section 11).")

    rows = []
    for s in SYS:
        names, cols, probe = phases(s)
        df = pd.read_csv(os.path.join(D, f"dataset_{s}.csv"))
        dev = np.abs(df[cols].values.sum(axis=1) - 1.0).max()
        occ = sorted(100 * float((df[c] > 1e-3).mean()) for c in cols)
        box = 100 * float(df["in_stainless_box"].mean())
        rows.append([LBL[s], f"{len(df):,}", f"{box:.1f}", sci(dev),
                     str(len(names)), f"{occ[0]:.2f}\u2013{occ[-1]:.1f}"])
    table(doc, ["System", "Rows", "Box (%)", "max|\u03a3y\u22121|", "K",
                "Occurrence range (%)"],
          rows,
          "Table 2. Dataset composition. \u201cBox (%)\u201d is the fraction "
          "of rows inside the industrially relevant stainless-steel "
          "composition box (Fe \u2265 55 at.%, Cr \u2264 30 at.%, "
          "x\u2083 \u2264 30 at.%, with x\u2083 the third element of the "
          "system). \u201cOccurrence range\u201d spans the rarest "
          "and most common target phase.")

    figure(doc, "fig1_dataset",
           "Figure 2. Dataset for Fe\u2013Cr\u2013Ni and summary statistics "
           "across all five systems. (a) Design points in the (x(Cr), x(Ni)) "
           "plane coloured by temperature. (b) Temperature density; the shaded "
           "band marks the region above 1300 K where liquid appears. (c) The "
           "stainless-steel subset (green) occupies 38% of rows. (d) Phase "
           "occurrence across all five systems on a logarithmic scale; rare "
           "phases such as MNNI2 (0.00%) and CR3MN5 (0.03%) are retained as "
           f"targets. (e) Simplex closure of the target vectors: all systems "
           f"satisfy |{SIG}y {MINUS} 1| < 10{sup('-6')}, with the bulk of "
           f"rows at 10{sup('-9')}\u201310{sup('-8')}.")

    # 3. Models and training
    doc.add_heading("3. Models and training", level=1)
    doc.add_heading("3.1 Neural-network heads", level=2)
    body(doc,
         "The base architecture is a multilayer perceptron with three hidden "
         "layers of 192 units (Linear \u2192 LayerNorm \u2192 SiLU \u2192 "
         f"Dropout 0.1), trained with AdamW (learning rate 3{TIMES}10{sup('-4')}, "
         f"weight decay 5{TIMES}10{sup('-4')}, cosine annealing over 300 "
         f"epochs, batch size 128, gradient-norm clipping at 1.0, early "
         f"stopping on validation MAE with patience 40). The loss is Huber "
         f"with {DELTA} = 0.01 by default. Inputs are standardised with a "
         f"StandardScaler fitted on the training split only. Six output "
         f"heads are compared across all five systems (Table 3); two further "
         f"mechanisms, power normalisation and a sum-to-one loss penalty, "
         f"are evaluated on {LBL['fecrni']} only as mechanism probes "
         f"(Sections 8\u20139).")
    body(doc,
         f"The sigmoid head applies {SIGMA_PH}(z) without normalisation and "
         f"serves as the unconstrained control. The softmax head applies "
         f"softmax(z). The sigmoid/{SIG} head computes {SIGMA_PH}(z) / "
         f"{SIG}{sub('k')} {SIGMA_PH}(z{sub('k')}) at train time. The renorm "
         f"head applies {SIGMA_PH}(z) and then clips to [0, 1] and divides by "
         f"the row sum as a post-hoc projection; the projection lies outside "
         f"the training graph and is applied at inference time only. "
         f"The sparsemax head applies "
         f"the sparsemax operator (Martins and Astudillo, 2016) to 4z, "
         f"producing exact zeros. The residue head predicts K{MINUS}1 "
         f"fractions with sigmoids and closes the sum algebraically, "
         f"\u0177(K) = 1 {MINUS} {SIG}{{k<K}} \u0177(k); it enforces the "
         f"sum-to-one equality by construction but does not intrinsically "
         f"guarantee non-negativity, so it is not a fully simplex-valid "
         f"output (Section 9). The two {LBL['fecrni']} probes are the "
         f"power-norm head, {SIGMA_PH}(z){sup('2')} / {SIG}{sub('k')} "
         f"{SIGMA_PH}(z{sub('k')}){sup('2')} \u2014 each sigmoid output "
         f"raised to the power p = 2 and renormalised by its row sum, "
         f"applied inside the training graph \u2014 and the penalty head, "
         f"{SIGMA_PH}(z) plus {LAMBDA}({SIG}{sub('k')} \u0177{sub('k')} "
         f"{MINUS} 1){sup('2')} added to the loss. Non-softmax heads use a "
         f"per-phase two-layer output block (192 \u2192 192 \u2192 1) and a "
         f"learnable per-phase output scale.")
    body(doc,
         "We distinguish two properties that the literature often conflates. "
         "Closure is the equality \u03a3 \u0177(k) = 1 alone. Simplex "
         "membership is Eq. (1) in full: closure and \u0177(k) \u2265 0 for "
         "every k. Softmax, sigmoid/\u03a3, sparsemax, and power-norm "
         "guarantee both; renorm guarantees both after its projection; the "
         "residue head guarantees closure only.")

    doc.add_heading("3.2 Baselines", level=2)
    body(doc,
         "Four classical models are fitted one output channel at a time on the "
         f"raw targets: ridge regression ({ALPHA} = 1), k-nearest neighbours "
         f"(k = 10, distance-weighted), XGBoost (Chen and Guestrin, 2016; 500 "
         f"trees, depth 8, learning rate 0.05, early stopping at 50 rounds), "
         f"and random forest (Breiman, 2001; 500 trees, depth 20). Each is "
         f"evaluated raw and after post-hoc renormalisation (clipped to "
         f"[0, 1], then divided by the row sum; clipping matters only for "
         f"outputs that can leave [0, 1], since sigmoid outputs already lie "
         f"in (0, 1)); the renormalised "
         f"versions are the physically admissible ones reported in the main "
         f"comparison. All hyperparameters \u2014 neural and classical \u2014 "
         f"are fixed for the study: the comparison is intended to isolate "
         f"output-constraint and model-family behaviour under a fixed, "
         f"reproducible protocol rather than to identify globally optimal "
         f"hyperparameters for any model family.")

    doc.add_heading("3.3 Splitting, seeds, and metrics", level=2)
    body(doc,
         "The split is cluster-stratified: KMeans (k = 6, seed-dependent "
         "initialisation) on standardised (x\u2082, x\u2083, T), with each "
         "cluster contributing 64/16/20 train/validation/test by "
         "seed-dependent permutation. The K-means stratification, the "
         "per-cluster shuffle, and hence the split itself are re-drawn for "
         "every seed, so the reported seed-to-seed spread reflects both "
         "split variation and training stochasticity. Six clusters give a "
         "coarse "
         "stratification of the three-dimensional design space while "
         "keeping each cluster large enough (~1,500 rows) for a stable "
         "within-cluster permutation. The validation partition is used "
         "only for early stopping and XGBoost's stopping criterion; the "
         "test partition is never touched during training or model "
         "selection, so all reported test metrics are computed on data "
         "that played no role in fitting. Three seeds (42, 123, 2024) "
         "control all stochastic stages; every reported number is mean "
         "\u00b1 standard deviation over seeds, a measure of seed-to-seed "
         "training stochasticity only, not a full uncertainty "
         "quantification (which would additionally cover sampling, "
         "architecture, phase-set, and database uncertainty). The test "
         "set is capped at 4,000 rows.")
    body(doc, "The primary metric is the mean absolute error averaged over "
              "phases,")
    equation(doc,
             f"MAE = (1/K) {SIG}{{k=1..K}} (1/n) {SIG}{{i=1..n}} "
             f"|\u0177{sub('ik')} {MINUS} y{sub('ik')}|", "(2)")
    body(doc,
         f"Simplex closure is measured as mean |{SIG}{sub('k')} \u0177{sub('k')} "
         f"{MINUS} 1|. Phase-presence detection is assessed by per-phase "
         f"F{SUB['1']} and balanced accuracy (active if N{sup('k')} > "
         f"10{sup('-3')}); the reported mean F{SUB['1']} is a macro-average "
         f"over the K phases of each system. A phase with no active row "
         f"in a given test set (MNNI2, with zero active rows in the whole "
         f"{LBL['femnni']} dataset) has an undefined F{SUB['1']}; we score "
         f"it as F{SUB['1']} = 0 for every model, which keeps the phase in "
         f"the average without favouring any model. The 10{sup('-3')} threshold "
         f"is a practical choice: it sits two orders of magnitude below the "
         f"smallest non-trivial fraction a lever-rule calculation would "
         f"report, and above the solver's numerical noise floor "
         f"(~10{sup('-8')}). Detection conclusions are checked against this "
         f"choice in Section 6, where the same metrics at 10{sup('-4')} and "
         f"10{sup('-2')} leave the qualitative picture unchanged. An "
         f"entropy-weighted MAE weights rows by 1 + H(y)/H(max) with H(max) "
         f"= ln K, up-weighting multi-phase coexistence states, where "
         f"several non-zero fractions must be predicted simultaneously; it "
         f"is an auxiliary reporting metric, not a theoretically privileged "
         f"loss.")

    # 4. Main accuracy
    doc.add_heading("4. Main accuracy", level=1)
    body(doc,
         "Table 3 reports the test MAE for all heads and baselines across the "
         "five systems. Figure 3 visualises the comparison.")

    ROWS = [("mlp_sigmoid", "sigmoid (unconstrained)", "off-simplex"),
            ("mlp_residue", "residue, y(d) = 1\u2212\u03a3", "off-simplex"),
            ("mlp_sparsemax", "sparsemax", "on-simplex"),
            ("mlp_softmax", "softmax", "on-simplex"),
            ("mlp_sig_norm", f"sigmoid/{SIG} (train-time)", "on-simplex"),
            ("mlp_renorm", "sigmoid + renorm (post hoc)", "on-simplex"),
            ("mlp_power_norm", "power-normalisation (\u2020)", "on-simplex"),
            ("ridge_renorm", "ridge", "baselines"),
            ("knn_renorm", "k-NN, k = 10", "baselines"),
            ("xgb_renorm", "XGBoost", "baselines"),
            ("rf_renorm", "random forest", "baselines")]
    best = {}
    RES = {s: results(s) for s in SYS}
    for s in SYS:
        cand = {k: agg(RES[s], k)[0] for k, _, _ in ROWS}
        best[s] = min(cand, key=cand.get)
    rows, bold_cells = [], set()
    for ri, (key, lab, grp) in enumerate(ROWS):
        cells = [grp, lab]
        for ci, s in enumerate(SYS):
            m, sd = agg(RES[s], key)
            txt, b = mae_cell(m, sd, bold=(best[s] == key))
            cells.append(txt)
            if b:
                bold_cells.add((ri, ci + 2))
        rows.append(cells)
    table(doc, ["Group", "Model"] + [LBL[s] for s in SYS], rows,
          "Table 3. Test MAE (mean over three seeds; uncertainty in units of "
          "the last digit). Best per system in bold. Heads are grouped by "
          "whether they satisfy Eq. (1) structurally (on-simplex: softmax, "
          f"sigmoid/{SIG}, sparsemax, renorm after its projection, "
          "power-norm), fail to satisfy it (off-simplex: the unconstrained "
          "sigmoid and the closure-only residue head), or are baselines "
          "(shown after post-hoc renormalisation). Six heads are "
          f"compared across all five systems; the power-normalisation head "
          f"(\u2020) was evaluated on {LBL['fecrni']} only.",
          left_cols=(0, 1), bold_cells=bold_cells)

    figure(doc, "fig2_heads",
           "Figure 3. Main comparison. (a) Test MAE for the six neural heads "
           "across all five systems (mean \u00b1 s.d., three seeds). (b) Best "
           "constrained MLP versus random forest; the annotation gives the "
           "RF/MLP ratio. (c) Simplex closure: constrained heads and "
           f"renormalised baselines sit at or below 10{sup('-8')}; "
           f"unconstrained outputs violate closure by 10{sup('-2')}\u2013"
           f"10{sup('-1')}.")

    rn, sd = agg(RES["fecrni"], "mlp_renorm")
    rfn, _ = agg(RES["fecrni"], "rf_renorm")
    finding(doc, "Constrained MLPs lead on smooth-coexistence systems",
            f"On {LBL['fecrni']} the renorm head achieves {rn:.4f} {PM} "
            f"{sd:.4f} against {rfn:.4f} for the random forest \u2014 a 41% "
            f"reduction, roughly 12{TIMES} the seed standard deviation. On "
            f"{LBL['fecrmn']} the sigmoid/{SIG} head achieves 0.0074 {PM} "
            f"0.0002 against 0.0133 {PM} 0.0005 for the forest. Both systems "
            f"feature extended smooth two- and three-phase coexistence fields "
            f"(sigma with BCC/FCC in {LBL['fecrni']}; the Laves, mu, R and "
            f"chi intermetallic family in {LBL['fecrmo']}), which favour the "
            f"smooth function class of the MLP.")
    finding(doc, "Trees win where boundaries are sharp",
            f"On {LBL['fecrv']} the random forest achieves 0.0137 {PM} 0.0004 "
            f"against 0.0149 {PM} 0.0004 for the best MLP; on {LBL['femnni']} "
            f"the gap is larger, 0.0058 {PM} 0.0003 against 0.0140 {PM} "
            f"0.0013. Both systems are dominated by sharp, essentially binary "
            f"phase boundaries (BCC\u2194SIGMA/FCC in {LBL['fecrv']}; "
            f"FCC\u2194LIQUID in {LBL['femnni']}) with rare phases. "
            f"Axis-aligned tree splits can represent such sharp transitions "
            f"efficiently; on {LBL['fecrv']}, however, the margin is "
            f"statistically indistinguishable from a tie, formally confirmed "
            f"via paired bootstrap resampling over 10,000 iterations "
            f"(Table 3b).")
    finding(doc, "Fe\u2013Cr\u2013Mo is a statistical tie",
            "The random forest (0.0123 \u00b1 0.0002) and sigmoid/\u03a3 "
            "(0.0126 \u00b1 0.0002) differ by 0.0003, comparable to the seed "
            "standard deviation. The system's nine-phase target vector, with "
            "its intermetallic family, provides enough smooth structure for "
            "the MLP to compete but enough sharp boundaries for the forest to "
            "match it.")
    finding(doc, "Paired bootstrap confirms the ranking and the ties",
            "Three seeds give limited statistical power, so we paired the "
            "two models' per-row errors on the identical test rows and "
            "resampled with replacement (10,000 paired bootstrap resamples "
            "of the test rows, three seeds pooled; Table 3b). The 95% "
            "confidence interval of the "
            f"mean-MAE difference excludes zero in favour of the MLP on "
            f"{LBL['fecrni']} and {LBL['fecrmn']} and in favour of the random "
            f"forest on {LBL['femnni']}, while it includes zero on "
            f"{LBL['fecrmo']} and {LBL['fecrv']}: the two near-ties claimed "
            f"above are genuine ties, not under-powered comparisons.")
    PB = json.load(open(os.path.join(M, "paired_bootstrap.json")))
    rows = []
    for s in SYS:
        e = PB[s]
        d_ = e["mean_diff_mlp_minus_rf"]
        lo, hi = e["ci95"]
        verdict = ("tie" if lo <= 0 <= hi else
                   ("MLP better" if hi < 0 else "RF better"))
        head = e["best_mlp"][len("mlp_"):].replace("_", "-")
        rows.append([LBL[s], head, f"{d_:+.5f}", f"[{lo:+.5f}, {hi:+.5f}]",
                     verdict])
    table(doc, ["System", "Best MLP head", "\u0394MAE (MLP\u2212RF)",
                "95% CI", "Verdict"], rows,
          "Table 3b. Paired bootstrap comparison of the best constrained MLP "
          "head and the random forest on the interpolation test set. "
          "\u0394MAE is the mean per-row MAE difference (MLP \u2212 RF); a "
          "negative value favours the MLP. CI: 95% bootstrap confidence "
          "interval (10,000 paired bootstrap resamples of the test rows, "
          "three seeds pooled).", left_cols=(0, 1))
    finding(doc, "Post-hoc renormalisation did not hurt accuracy in any of "
            "the five systems tested",
            "All four evaluated baselines improve or remain unchanged under "
            "renormalisation, by 0.0000\u2013"
            "0.0004 for the random forest and k-NN, 0.0007\u20130.0020 for "
            "XGBoost, and 0.0066\u20130.0163 for ridge. The larger the "
            "simplex violation, the more the projection recovers.")
    finding(doc, "Aggregate cell-averaged MAE understates worst-case "
            "behaviour and phase-rule admissibility",
            "Restricting error to active cells (true fraction above 0.005) "
            "raises the renorm MLP's MAE from 0.006\u20130.016 to "
            "0.023\u20130.074 depending on system (random forest: "
            "0.025\u20130.049), the 95th percentile of the per-row mean "
            "error reaches 0.03\u20130.12, and the single worst cell in "
            "every system is a complete miss (error near 1). Physical "
            "admissibility is, moreover, stricter than simplex membership: "
            "in a ternary at fixed pressure, more than three phases can "
            "coexist only at invariant points, so any prediction assigning "
            f"more than three phases above the 10{sup('-3')} presence "
            f"threshold is physically inadmissible. The ground truth "
            f"contains no such rows, the renorm MLP emits none in four "
            f"systems (3.4% of {LBL['fecrmo']} rows), and the random forest "
            f"reaches 0\u20138.9%; XGBoost is the worst offender at up to "
            f"22.4% in {LBL['fecrmo']}. Simplex-valid outputs therefore do "
            f"not guarantee phase-rule admissibility, and this axis should "
            f"accompany closure in future evaluation protocols.")

    # 5. Simplex closure
    doc.add_heading("5. Simplex closure", level=1)
    body(doc,
         f"Table 4 reports the mean closure violation. All structurally "
         f"constrained heads satisfy {SIG}{sub('k')} \u0177{sub('k')} = 1 to "
         f"{LEQ} 5{TIMES}10{sup('-8')} \u2014 float32 round-off over K terms. "
         f"The unconstrained sigmoid violates closure by 6\u20138% on average. "
         f"Raw tree outputs violate it by 0.9\u20134.7%. Post-hoc "
         f"renormalisation repairs all evaluated models to machine precision "
         f"(~10{sup('-17')}).")
    body(doc,
         "Closure, however, is only half of Eq. (1). The residue head closes "
         "the sum by construction yet still emits negative fractions on "
         "0.0\u20131.2% of cells depending on the system (Section 9), so it "
         "is closure-valid but not simplex-valid. All other constrained "
         "heads and the renormalised baselines satisfy both conditions.")
    CONS = [("none", "mlp_sigmoid", "sigmoid"),
            ("none", "rf_raw", "random forest, raw"),
            ("none", "xgb_raw", "XGBoost, raw"),
            ("structural", "mlp_softmax", "softmax"),
            ("structural", "mlp_sig_norm", f"sigmoid/{SIG}"),
            ("structural", "mlp_sparsemax", "sparsemax"),
            ("structural", "mlp_residue", "residue"),
            ("post hoc", "mlp_renorm", "sigmoid + renorm"),
            ("post hoc", "rf_renorm", "random forest + renorm")]
    rows = [[grp, lab] + [sci(agg(RES[s], key, "consistency")[0]) for s in SYS]
            for grp, key, lab in CONS]
    table(doc, ["Closure", "Model"] + [LBL[s] for s in SYS], rows,
          f"Table 4. Mean closure violation mean |{SIG}{sub('k')} "
          f"\u0177{sub('k')} {MINUS} 1| on the test set (mean over three "
          f"seeds). Structural constraints achieve closure to float32 "
          f"precision; post-hoc renormalisation achieves machine precision; "
          f"unconstrained outputs violate closure by orders of magnitude. "
          f"Closure is the equality constraint only; see the text for "
          f"non-negativity.",
          left_cols=(0, 1))
    body(doc,
         "The closure result has a practical consequence: a surrogate whose "
         "output violates Eq. (1) cannot be used directly in any downstream "
         "calculation that assumes a valid phase assemblage \u2014 lever-rule "
         "property averaging, for instance, or a subsequent kinetic "
         "simulation. The constrained heads and post-hoc renormalisation both "
         "produce admissible outputs; the former do so without a separate "
         "projection step.")

    # 6. Regression versus detection
    doc.add_heading("6. Regression versus detection", level=1)
    body(doc,
         "Table 5 and Figure 4 reveal a gap between regression accuracy and "
         "phase-presence detection. The two are distinct tasks: the model "
         "that best estimates continuous phase fractions need not be the "
         "model that best identifies phase presence, and the model with the "
         "lowest MAE is not always the best detector.")
    rows = []
    for s in SYS:
        d = RES[s]
        bmlp = min(["mlp_renorm", "mlp_sig_norm", "mlp_softmax"],
                   key=lambda k: agg(d, k)[0])
        for i, model in enumerate([bmlp, "rf_renorm"]):
            lab = ("MLP " + bmlp[len("mlp_"):].replace("_", "-")) if i == 0 \
                else "random forest"
            ba = np.mean([np.mean(d[f"{model}_s{k}"]["balanced_acc"])
                          for k in SEEDS if f"{model}_s{k}" in d])
            m, _ = agg(d, model)
            f1 = agg(d, model, "mean_f1")[0]
            en = agg(d, model, "entropy_mae")[0]
            rows.append([LBL[s] if i == 0 else "", lab, f"{m:.4f}",
                         f"{f1:.3f}", f"{ba:.3f}", f"{en:.4f}"])
    table(doc, ["System", "Model", "MAE", "F\u2081", "Bal. acc.",
                "Entropy-w. MAE"], rows,
          "Table 5. Regression versus detection. For each system the best "
          "constrained MLP head and the random forest are compared on MAE, "
          "mean F\u2081, balanced accuracy, and entropy-weighted MAE. The MLP "
          "is the better regressor on two systems (with a near-tie on "
          "Fe\u2013Cr\u2013Mo) but the worse detector on three.",
          left_cols=(0, 1))

    figure(doc, "fig4_perphase",
           "Figure 4. Per-phase accuracy and the regressor\u2013detector gap. "
           "(a) Parity plot for the hardest and easiest phase of "
           "Fe\u2013Cr\u2013Ni under the renorm head. (b) Per-phase MAE "
           "versus phase occurrence across all five systems: rare phases are "
           "cheap in aggregate MAE because the dominant zero-fraction rows "
           "contribute little error. (c) MAE ratio "
           "versus F\u2081 ratio (MLP over random forest): systems below the "
           "diagonal have an MLP that regresses better but detects worse.")

    finding(doc, "Rare phases are cheap in aggregate MAE and hard to detect",
            "Aggregate MAE can underweight rare phases because the dominant "
            "zero-fraction rows contribute almost no error: a phase present "
            "in 0.1% of rows contributes little to the average whether the "
            "model predicts it correctly or not. Low MAE on such a phase is "
            "therefore not evidence that the model represents it; it is an "
            "artefact of the averaging. Detecting its presence \u2014 the "
            "metallurgically important question \u2014 requires the model to "
            f"distinguish a small positive signal from noise, which is what "
            f"the presence metrics measure. On "
            f"{LBL['femnni']} the MLP predicts MNNI, MNNI2, and ALPHA_MN as "
            f"absent (F{SUB['1']} = 0 for all three) while the random forest "
            f"resolves two of them.")
    finding(doc, "The gap is not an artefact of the zero-positive scoring "
            "convention",
            "Excluding zero-positive phases from the macro-average raises "
            f"the {LBL['femnni']} MLP F{SUB['1']} only from 0.312 to 0.364, "
            "versus 0.504 to 0.588 for the random forest. Threshold-free "
            "per-phase average precision (AUPRC) confirms the pattern: "
            "across the ten rare phases with occurrence below 5%, the "
            "renorm MLP's average precision is 0.001\u20130.05 for nine of "
            f"them (for example 0.014 for the 2%-occurrence FCC phase in "
            f"{LBL['fecrv']} and 0.003 for MNNI in {LBL['femnni']}; the "
            f"exception is FCC in {LBL['fecrmo']} at 0.63), whereas the "
            f"random forest retains 0.40\u20130.97 in all of them except the "
            f"two near-zero-occurrence phases (CR3MN5 at 0.03% and "
            f"MU_PHASE_I at 0.1%), where both families fail.")
    finding(doc, "The gap is stable across presence thresholds",
            f"The presence threshold N{sup('k')} > 10{sup('-3')} is a "
            f"modelling decision, so we recomputed F{SUB['1']} at "
            f"10{sup('-4')} and 10{sup('-2')} from the stored predictions "
            f"(Table 5b). Absolute F{SUB['1']} values move \u2014 stricter "
            f"thresholds make detection harder \u2014 but the ordering does "
            f"not on four of five systems: the random forest remains the "
            f"better detector on {LBL['fecrmn']}, {LBL['fecrv']}, and "
            f"{LBL['femnni']} at all three thresholds, and the MLP on "
            f"{LBL['fecrni']}. On {LBL['fecrmo']} the better detector flips "
            f"with the threshold, consistent with that system's overall tie. "
            f"The regressor\u2013detector gap is not an artefact of the "
            f"threshold choice.")
    TS = json.load(open(os.path.join(M, "threshold_sensitivity.json")))
    rows = []
    for s in SYS:
        for i, model in enumerate(["mlp", "rf"]):
            lab = ("best MLP (" + TS[s]["best_mlp"][len("mlp_"):].replace("_", "-") + ")") \
                if model == "mlp" else "random forest"
            cells = []
            for thr in ["0.0001", "0.001", "0.01"]:
                m, sd = TS[s][model][thr]["mean_f1"]
                cells.append(f"{m:.3f} ({int(round(sd * 1e3))})")
            rows.append([LBL[s] if i == 0 else "", lab] + cells)
    table(doc, ["System", "Model", "F\u2081 at 10\u207b\u2074",
                "F\u2081 at 10\u207b\u00b3", "F\u2081 at 10\u207b\u00b2"],
          rows,
          "Table 5b. Sensitivity of phase-presence detection to the presence "
          f"threshold. Mean F{SUB['1']} (three seeds; uncertainty in units of "
          f"the last digit) for the best constrained MLP head and the random "
          f"forest at thresholds 10{sup('-4')}, 10{sup('-3')} (main-study "
          f"default), and 10{sup('-2')}.", left_cols=(0, 1))
    finding(doc, "Detection degrades faster than regression under shift",
            f"This gap widens under distribution shift. On the held-out "
            f"temperature band of {LBL['fecrni']}, the MLP's mean F{SUB['1']} "
            f"drops from 0.93 to 0.64 while its MAE rises only from 0.010 to "
            f"0.013 (Section 7). Across all five systems and both model "
            f"families, macro-F{SUB['1']} falls by roughly 20\u201360% in "
            f"relative terms on the held-out temperature bands, even as "
            f"regression MAE remains comparatively stable, whereas the "
            f"composition bands affect detection more mildly for the MLP "
            f"(2\u201310% relative) than for the random forest "
            f"(7\u201344%). Phase-presence detection is the first "
            f"casualty of leaving the training distribution.")

    # 7. Spatial generalisation
    doc.add_heading("7. Spatial generalisation: contiguous holdout and "
                    "strict extrapolation", level=1)
    doc.add_heading("7.1 Two protocols, each with a random control", level=2)
    body(doc,
         "The cluster-stratified split of Section 3.3 is an interpolation "
         "test. We add two spatial protocols; in both the model is retrained "
         "from scratch (same protocol, three seeds, four models, five "
         "systems).")
    body(doc,
         "Contiguous interior-band holdout. Two blocks are removed from "
         "training and used as the test set: a temperature band "
         "T \u2208 [1200, 1400] K (1,275 rows) and a composition band "
         "x\u2082 \u2208 [0.25, 0.35] (1,676 rows). Because training rows "
         "remain on both sides of each band, this is a missing-region "
         "generalisation test, not strict extrapolation. We verified this "
         "geometrically: a Delaunay triangulation of the remaining training "
         "points in scaled (x\u2082, x\u2083, T) coordinates shows that "
         "99.3\u201399.4% of the held-out band rows lie inside the convex "
         "hull of the training set in all five systems. The held-out region is "
         "a hole in the sampling, not an out-of-support region.")
    body(doc,
         "Strict one-sided out-of-range extrapolation. Training keeps only "
         "the near side of one coordinate and the test set is the far side, "
         "with the interior gap in neither split: train x\u2082 \u2264 0.25, "
         "test x\u2082 > 0.35 "
         f"(2,679 test rows on {LBL['fecrni']}); and train "
         f"T \u2264 1200 K, test T > 1400 K (3,219 test rows). Every test "
         f"point now lies outside the training range of the extrapolated "
         f"coordinate by construction. This is strict out-of-range "
         f"extrapolation along one coordinate at a time; it does not imply "
         f"that the whole multidimensional chemical space of the test rows "
         f"is unseen, since the other coordinates remain inside their "
         f"training ranges.")
    body(doc,
         "The random controls are the key methodological element of both "
         "protocols. Removing a region also removes training data, so a bare "
         "region-versus-interpolation comparison cannot separate \u201ccannot "
         "generalise there\u201d from \u201chad less to learn from\u201d. Each "
         "block therefore has a size-matched control: the same number of "
         "rows removed from training at random, with an equal-size random "
         "test set. Any degradation beyond the control is spatial. Because "
         "the control rows differ from the band rows, these ratios alone "
         "conflate region difficulty with distribution shift. As a "
         "complementary reference we therefore also report a "
         "region-matched reanalysis: for each band, the band-holdout error "
         "is compared with the error of the main-study models on the same "
         "band rows inside the interpolation test set, where band data were "
         "present in training. This ratio isolates the regional shift on "
         "identical rows; its residual confound is a training-set size "
         "difference of roughly 15%, since removing a band also removes "
         "those rows from training. Table 6 "
         "reports the band results and Table 6b the extrapolation results; "
         "Figure 5 panel (a) shows the band penalty ratios and Figure 6 the "
         "extrapolation ones.")

    MODELS = [("mlp_renorm", "MLP renorm"),
              ("mlp_sig_norm", f"MLP sigmoid/{SIG}"),
              ("rf_renorm", "random forest"), ("xgb_renorm", "XGBoost")]
    rows = []
    for blk, blab in [("random_ctrl", "random control"),
                      ("T_band", "T \u2208 [1200, 1400] K"),
                      ("X2_band", "x\u2082 \u2208 [0.25, 0.35]")]:
        for key, mlab in MODELS:
            cells = [blab, mlab]
            for s in SYS:
                m, sd = bagg(s, f"{blk}_{key}")
                cells.append(mae_cell(m, sd)[0])
            rows.append(cells)
    for key, mlab in MODELS:
        cells = ["penalty T / x\u2082", mlab]
        for s in SYS:
            c, _ = bagg(s, f"random_ctrl_{key}")
            t, _ = bagg(s, f"T_band_{key}")
            x, _ = bagg(s, f"X2_band_{key}")
            cells.append("\u2014" if np.isnan(c) else
                         f"{t / c:.2f} / {x / c:.2f}")
        rows.append(cells)
    table(doc, ["Held-out region", "Model"] + [LBL[s] for s in SYS], rows,
          "Table 6. Contiguous interior-band holdout. Top: test MAE on the "
          "random control, the held-out temperature band, and the held-out "
          "composition band (mean over three seeds; uncertainty in units of "
          "the last digit). Bottom: penalty ratios (band MAE / control MAE) "
          "for the temperature and composition bands. The ratios compare "
          "the holdout error against a size-matched random control (an "
          "equal number of rows removed at random and evaluated on those "
          "random rows). This quantifies whether removing a contiguous "
          "band is costlier than removing random rows, but does not by "
          "itself separate region difficulty from distribution shift; a "
          "region-matched reanalysis on identical band rows is reported "
          "in the text.",
          left_cols=(0, 1))

    EX = {}
    for s in SYS:
        p = os.path.join(M, f"holdout_extrap_{s}.json")
        if os.path.exists(p):
            EX[s] = json.load(open(p))

    def xagg(s, tag, field="mean_mae"):
        dd = EX.get(s, {})
        v = [dd[f"{tag}_s{k}"][field] for k in SEEDS
             if f"{tag}_s{k}" in dd]
        return (float(np.mean(v)), float(np.std(v))) if v \
            else (np.nan, np.nan)
    if EX:
        rows = []
        for blk, blab in [
                ("X2_extrap_ctrl", "random control (x\u2082-matched)"),
                ("X2_extrap", "train x\u2082 \u2264 0.25, test x\u2082 > 0.35"),
                ("T_extrap_ctrl", "random control (T-matched)"),
                ("T_extrap", "train T \u2264 1200 K, test T > 1400 K")]:
            for key, mlab in MODELS:
                cells = [blab, mlab]
                for s in SYS:
                    m, sd = xagg(s, f"{blk}_{key}")
                    cells.append(mae_cell(m, sd)[0])
                rows.append(cells)
        for key, mlab in MODELS:
            cells = ["penalty x\u2082 / T", mlab]
            for s in SYS:
                cx, _ = xagg(s, f"X2_extrap_ctrl_{key}")
                x, _ = xagg(s, f"X2_extrap_{key}")
                ct, _ = xagg(s, f"T_extrap_ctrl_{key}")
                t, _ = xagg(s, f"T_extrap_{key}")
                cells.append("\u2014" if np.isnan(cx) or np.isnan(ct) else
                             f"{x / cx:.2f} / {t / ct:.2f}")
            rows.append(cells)
        table(doc, ["Region", "Model"] + [LBL[s] for s in SYS], rows,
              "Table 6b. Strict one-sided out-of-range extrapolation: "
              "training keeps only the near side of one coordinate, the "
              "test set is the far side, and the interior gap is in neither "
              "split. Top: "
              "test MAE on the far side against size-matched random "
              "controls (mean over three seeds; uncertainty in units of "
              "the last digit). Bottom: penalty ratios (far-side MAE / "
              "matched-control MAE) for composition and temperature "
              "extrapolation. All models exhibit sharp error increases "
              "under genuine extrapolation. Standard deviations are "
              "seed-to-seed; the large-looking Fe\u2013Cr\u2013V "
              f"random-control deviation (0.0242 {PM} 0.0073) reflects "
              "ordinary seed variability (per-seed values "
              "0.027/0.014/0.031), not a diverged run. The temperature "
              "protocol additionally tests an out-of-basis target, as "
              "LIQUID is absent from the near-side training data in four "
              "of the five systems.", left_cols=(0, 1))

    figure(doc, "fig5_ablation",
           "Figure 5. (a) Contiguous-band penalty ratios (band MAE / "
           "random-control MAE) across all five systems for the MLP, random "
           "forest, and XGBoost. Filled markers: temperature band; open "
           "markers: composition band. The dashed line at 1.0 marks the "
           "random control. (b) Effect of hidden-layer width on test MAE "
           "for four heads on Fe\u2013Cr\u2013Ni. (c) Effect of the loss "
           "function on test MAE for the renorm and sigmoid/\u03a3 heads.")
    figure(doc, "fig7_extrap",
           "Figure 6. Strict one-sided out-of-range extrapolation. (a) "
           "Penalty ratios "
           "(far-side MAE / matched random-control MAE) for composition "
           "extrapolation (train x\u2082 \u2264 0.25, test x\u2082 > 0.35; "
           "open markers) and temperature extrapolation (train T \u2264 1200 "
           "K, test T > 1400 K; filled markers). (b) Absolute test MAE on "
           "the far side. All evaluated model families degrade sharply "
           "beyond the training range of the extrapolated coordinate; the "
           "constrained MLP achieves the lowest absolute error.")

    doc.add_heading("7.2 Findings: contiguous interior-band holdout", level=2)
    finding(doc, "Two complementary references are needed for the band "
            "results",
            "Results for the missing-region generalisation test (Table 6, "
            "Figure 5a) require two complementary references. The "
            "size-matched random controls ask whether removing a contiguous "
            "band is costlier than removing the same number of random rows, "
            "but they are evaluated on different, average-difficulty rows "
            "and therefore conflate region difficulty with distribution "
            "shift. The region-matched reanalysis (Section 7.1) instead "
            "compares the same band rows with and without regional training "
            "data, and under this measure the picture is consistent: the "
            "in-distribution-to-holdout penalty is smaller for the "
            "constrained MLPs than for the tree ensembles in nine of the "
            f"ten band{TIMES}system combinations. On the composition bands, "
            "which cross steep phase boundaries, the MLP penalty ratios are "
            "1.2\u20132.1 versus 1.4\u20133.0 for the random forest and "
            f"XGBoost in all five systems; on the temperature bands the MLP "
            f"ratios are 0.9\u20132.8 versus 1.3\u20133.9, with "
            f"{LBL['fecrni']} the single exception, where the random-forest "
            f"ratio is lower (2.1{TIMES} versus 2.8{TIMES}) although its "
            f"absolute holdout error remains twice the MLP's. Ratios must "
            f"nonetheless be read alongside absolute errors: a band that is "
            f"easy in distribution, such as the nearly single-phase "
            f"1200\u20131400 K band of {LBL['femnni']}, inflates ratios "
            f"through a small denominator.")
    finding(doc, "The temperature-band penalty is largely a data effect",
            f"On {LBL['fecrni']} the MLP and the random forest pay the same "
            f"1.31{TIMES} penalty once the random control is used as the "
            f"reference. Comparing the temperature band against the "
            f"interpolation split instead would have suggested that trees "
            f"cannot generalise in temperature; the size-matched control "
            f"shows that the temperature band removes 1,275 training rows, "
            f"and the resulting data loss, not the spatial shift, accounts "
            f"for most of the degradation.")
    finding(doc, "The composition band is where model families separate",
            f"On {LBL['fecrni']} the random forest pays 3.11{TIMES} on the "
            f"composition band against 1.61{TIMES} for the constrained MLP "
            f"(2.24{TIMES} for XGBoost). The same ordering holds on "
            f"{LBL['fecrmn']} (1.57{TIMES} versus 0.85{TIMES}) and "
            f"{LBL['fecrmo']} (1.85{TIMES} versus 1.01{TIMES}). Since the "
            f"held-out rows lie inside the training convex hull, this is "
            f"not an extrapolation result in the strict sense; it is a "
            f"result about graceful degradation under a contiguous missing "
            f"region. Tree-based models partition the input space into "
            f"piecewise-constant regions, so predictions inside the hole "
            f"depend on the learned partition boundaries and on how the "
            f"forest aggregates across trees; this can degrade rapidly when "
            f"the hole spans a phase-boundary shift, while the MLP's "
            f"smooth parametrisation degrades more gracefully.")
    finding(doc, "The composition-band ordering is consistent across all "
            "system\u2013seed pairs",
            f"As a statistical check on the spatial protocol, we applied a "
            f"two-sided sign test to the 15 system{TIMES}seed penalty-ratio "
            f"pairs. The MLP's composition-band penalty is lower than the "
            f"random forest's and lower than XGBoost's on all 15 pairs "
            f"(p = 6.1{TIMES}10{sup('-5')} for each comparison). On the "
            f"temperature band the MLP\u2013forest difference is not "
            f"significant (11 of 15 pairs, p = 0.12), consistent with the "
            f"shared 1.31{TIMES} penalty, while the MLP\u2013XGBoost "
            f"difference remains significant (15 of 15, p = "
            f"6.1{TIMES}10{sup('-5')}).")
    finding(doc, "The penalty is system-dependent, not universal",
            f"On {LBL['fecrv']} the random forest degrades 4.92{TIMES} on the "
            f"temperature band \u2014 a phase-boundary-rich region where the "
            f"forest's local averaging fails badly. On {LBL['femnni']} the "
            f"held-out bands are easier than the random control "
            f"(0.04\u20130.73{TIMES}), because the bands sit in the dataset's "
            f"easy, liquid-dominated corner. Generalisation difficulty is a "
            f"property of the system and the region, not of the model alone.")
    finding(doc, "Detection collapses on the temperature band",
            f"Mean F{SUB['1']} drops sharply for all four evaluated models "
            f"on the temperature band, even where MAE barely moves. On "
            f"{LBL['fecrni']} the MLP's F{SUB['1']} falls from 0.93 (control) "
            f"to 0.64 (temperature band); the random forest's from 0.85 to "
            f"0.53. The phase-presence decision is more sensitive to "
            f"distribution shift than the fraction regression.")

    doc.add_heading("7.3 Findings: strict one-sided out-of-range "
                    "extrapolation", level=2)
    finding(doc, "All evaluated model families degrade sharply beyond the "
            "training range",
            f"On {LBL['fecrni']} the composition-extrapolation MAE rises to "
            f"~0.17 for the MLP and ~0.22 for the random forest, against "
            f"~0.013 on the matched control \u2014 an order of magnitude. "
            f"Temperature extrapolation is similarly severe. No architecture "
            f"evaluated in this study extrapolates phase fractions reliably "
            f"beyond the trained composition or temperature range; this is "
            f"the honest headline of the strict protocol.")
    finding(doc, "The temperature protocol is additionally limited by the "
            "closed phase basis",
            "Liquid has no active row in the near-side training data "
            f"(T {LEQ} 1200 K) in four of the five systems "
            f"({LBL['fecrmo']} retains some low-temperature liquid), so the "
            f"far side demands a phase that the target basis cannot express. "
            f"In {LBL['femnni']} this renders the task nearly uninformative: "
            f"all four model families land within 4% of the constant "
            f"near-side-mean predictor (MAE 0.182\u20130.184 versus 0.190), "
            f"and the per-phase errors show that the unseen liquid column "
            f"alone contributes 0.088 of the 7-phase mean, its entire "
            f"far-side mean fraction of 0.613 being missed. In the other "
            f"systems the models still beat the constant predictor "
            f"substantially (for example 0.099\u20130.197 in {LBL['fecrv']} "
            f"versus 0.307 for the constant predictor), indicating partial "
            f"transfer of the seen-phase structure. The composition "
            f"protocol, by contrast, keeps every frequent phase inside the "
            f"training basis (only trace phases such as ALPHA_MN, BETA_MN, "
            f"and MNNI on the Mn-rich far side of {LBL['femnni']} are "
            f"unseen) and therefore measures genuine spatial extrapolation.")
    finding(doc, "The constrained MLP reaches the lowest far-side error",
            f"In absolute terms the constrained MLP's far-side MAE is the "
            f"lowest of the three model families on composition "
            f"extrapolation in all five systems and on temperature "
            f"extrapolation in all five (Figure 6, panel b). Relative to "
            f"its own matched control the MLP's penalty ratio can be higher "
            f"than the trees' \u2014 the trees' controls are themselves "
            f"degraded, so the ratio is reference-dependent \u2014 and we "
            f"report the absolute-error comparison as the cleaner "
            f"statement. A two-sided sign test over the 15 "
            f"system{TIMES}seed pairs agrees with that caveat: the MLP's "
            f"penalty ratio is lower than the trees' on only 9 of 15 pairs "
            f"(p = 0.61), so the ratio comparison is not significant. The "
            f"ordering matches the contiguous-band result: "
            f"smooth parametrisations degrade more gracefully than leaf "
            f"averaging when the query leaves the trained region. The "
            f"difference is one of degree, not of kind \u2014 graceful "
            f"degradation is not reliable extrapolation.")

    # 8. Ablations
    doc.add_heading("8. Ablations", level=1)
    body(doc,
         "Table 7 and Figure 5 panels (b) and (c) report three ablations on "
         f"{LBL['fecrni']}.")
    d = RES["fecrni"]
    rows = []
    for lk, lab in [("huber", "Huber, \u03b4 = 0.01 (default)"),
                    ("mae", "MAE"), ("mse", "MSE")]:
        cells = []
        for key in ["mlp_renorm", "mlp_sig_norm"]:
            if lk == "huber":
                m, sd = agg(d, key)
            else:
                dd = json.load(open(os.path.join(M, f"loss_ablation_{lk}.json")))
                v = [dd[f"{key}_s{k}"]["mean_mae"] for k in SEEDS
                     if f"{key}_s{k}" in dd]
                m, sd = float(np.mean(v)), float(np.std(v))
            cells.append(mae_cell(m, sd)[0])
        rows.append(["loss", lab] + cells)
    for w in [64, 128, 192, 256]:
        cells = []
        for key in ["mlp_renorm", "mlp_sig_norm"]:
            if w == 192:
                m, sd = agg(d, key)
            else:
                dd = json.load(open(os.path.join(M, f"width{w}.json")))
                v = [dd[f"{key}_s{k}"]["mean_mae"] for k in SEEDS
                     if f"{key}_s{k}" in dd]
                m, sd = float(np.mean(v)), float(np.std(v))
            cells.append(mae_cell(m, sd)[0])
        suffix = " (default)" if w == 192 else ""
        rows.append(["width", f"3{TIMES}{w}{suffix}"] + cells)
    for key, lab in [("mlp_sigmoid", "none"), ("mlp_penalty_1", f"{LAMBDA}=1"),
                     ("mlp_penalty_10", f"{LAMBDA}=10")]:
        m, sd = agg(d, key)
        c = agg(d, key, "consistency")[0]
        rows.append(["closure penalty", lab,
                     f"{mae_cell(m, sd)[0]}, closure {sci(c)}", ""])
    table(doc, ["Ablation", "Setting", "renorm", f"sigmoid/{SIG}"], rows,
          f"Table 7. Ablations on {LBL['fecrni']} (mean over three seeds; "
          f"uncertainty in units of the last digit). Loss function, "
          f"hidden-layer width, and closure penalty are varied for the renorm "
          f"and sigmoid/{SIG} heads.", left_cols=(0, 1))
    finding(doc, "Huber and MAE losses are equivalent; MSE is worse",
            f"The Huber loss with {DELTA} = 0.01 and the MAE loss give "
            f"statistically indistinguishable results (0.0106 versus 0.0103 "
            f"for renorm). The MSE loss is 47% worse (0.0156), because "
            f"rare-phase outliers dominate the squared error.")
    finding(doc, "Accuracy improves monotonically with width up to 256",
            "Widening the hidden layers from 64 to 256 units reduces the "
            f"renorm MAE from 0.0137 to 0.0099. The default 3{TIMES}192 "
            f"architecture was fixed before this ablation was run; the "
            f"ablation is a post-hoc diagnostic, and architecture search was "
            f"deliberately out of scope for this study. We therefore keep "
            f"192 as the default in all main comparisons rather than "
            f"retroactively adopting 256, and report the ablation so that "
            f"readers can judge how much accuracy an architecture search "
            f"could still recover. The reported MLP configuration should "
            f"therefore not be interpreted as architecture-optimal.")
    finding(doc, "The closure penalty is strongly detrimental",
            f"Adding {LAMBDA}({SIG}{sub('k')} \u0177{sub('k')} {MINUS} 1)"
            f"{sup('2')} to the loss degrades MAE by 8{TIMES} at {LAMBDA} = 1 "
            f"and 16{TIMES} at {LAMBDA} = 10, while reducing the closure "
            f"violation only from 6.2{TIMES}10{sup('-2')} to "
            f"1.3{TIMES}10{sup('-3')}. For the tested penalty formulation "
            f"and {LAMBDA} values, the soft closure penalty substantially "
            f"degraded MAE and did not achieve the numerical closure "
            f"obtained by structural constraints. The penalty is redundant "
            f"\u2014 the targets already sum to one \u2014 and competes with "
            f"the data loss. The penalty weights were restricted to "
            f"{LAMBDA} {IN} {{1, 10}}: with the Huber regression term "
            f"saturating its per-output gradient at \u03b4 = 0.01, these "
            f"weights make the closure term dominate the regression gradient "
            f"by one to two orders of magnitude, and no smaller {LAMBDA}, "
            f"gradient-rescaled penalty, or warm-up schedule was evaluated. "
            f"The defensible conclusion is therefore that the soft penalty "
            f"adds nothing over the structural heads at the tested weights, "
            f"not that soft penalties are unviable in general.")

    # 9. Negative results
    doc.add_heading("9. Negative results", level=1)
    body(doc,
         "Three constraint mechanisms that are plausible in principle fail in "
         "practice. We report each in full because the failure modes are "
         "informative.")
    doc.add_heading("9.1 Sparsemax: severe seed sensitivity under the "
                    "evaluated configuration", level=2)
    body(doc,
         f"Sparsemax (Martins and Astudillo, 2016) projects the logits onto "
         f"the simplex, producing exact zeros for phases the model deems "
         f"absent. It satisfies closure to {LEQ} 5{TIMES}10{sup('-8')} and "
         f"achieves the sparsity that softmax cannot. But under the "
         f"evaluated training configuration it exhibited severe seed "
         f"sensitivity and substantially worse MAE than the other "
         f"constrained heads: 2\u20134{TIMES} worse than renorm or "
         f"sigmoid/{SIG}, with the three seeds giving 0.018/0.037/0.117 on "
         f"{LBL['fecrmo']}, 0.053/0.067/0.013 on {LBL['fecrni']}, and "
         f"0.019/0.015/0.079 on {LBL['femnni']} \u2014 a seed standard "
         f"deviation of the same order as the mean (Figure 7, panel a).")
    body(doc,
         f"The observed behaviour is consistent with a sparsity trap: once "
         f"the sparsemax projection drives a phase's logit below the "
         f"threshold {TAU}, the gradient at that logit is exactly zero, so "
         f"a phase that starts out clipped can never recover, and whether "
         f"it recovers is decided by initialisation. We present this as a "
         f"mechanistic interpretation of the observed seed sensitivity, not "
         f"as a demonstrated cause; the experiment evaluates one "
         f"implementation and optimisation configuration. Notably, the head "
         f"was trained with a regression loss on its projected outputs "
         f"\u2014 the configuration most exposed to that zero-gradient "
         f"region \u2014 so the dedicated sparsemax loss and "
         f"{ALPHA}-entmax variants remain untested; a fixed-scale entmax "
         f"({ALPHA} < 2) or a warm start from softmax would be the natural "
         f"remedies. We therefore do not claim "
         f"that sparsemax is unsuitable for phase-fraction prediction in "
         f"general \u2014 only that it performed poorly and unreliably under "
         f"the configuration studied here.")
    doc.add_heading("9.2 Sum-to-one penalty: redundant and harmful", level=2)
    body(doc,
         "The penalty head of Section 8 is included here as a negative "
         "result. The sum-to-one constraint is already satisfied by the "
         "targets; adding it as a soft penalty creates a redundant loss term "
         f"that competes with the data fit. At {LAMBDA} = 10 the penalty "
         f"dominates the Huber loss entirely, and the model learns to "
         f"minimise closure violation at the expense of phase-fraction "
         f"accuracy.")
    doc.add_heading("9.3 Residue closure: closure without non-negativity, "
                    "and the error concentrates", level=2)
    body(doc,
         f"The residue head predicts K{MINUS}1 phase fractions and closes "
         f"the sum algebraically: \u0177(K) = 1 {MINUS} {SIG}{{k<K}} "
         f"\u0177(k). This enforces the sum-to-one equality by construction "
         f"but does not intrinsically guarantee non-negativity: if the "
         f"other predicted fractions sum to more than one, the residue is "
         f"negative. In the stored predictions the residue head emits "
         f"negative values on 0.0% ({LBL['fecrv']}), 0.02% "
         f"({LBL['femnni']}), 0.40% ({LBL['fecrni']}), 0.50% "
         f"({LBL['fecrmn']}), and 1.21% ({LBL['fecrmo']}) of cells, with "
         f"the most negative cell at {MINUS}0.59. It is therefore "
         f"closure-valid but not simplex-valid, and any deployment would "
         f"need a projection step after all.")
    body(doc,
         "Beyond admissibility, the head is also the least accurate "
         "constrained variant: even the best residue head is "
         f"1.3\u20132.5{TIMES} worse than the best constrained head in the "
         f"same system, and the choice of which phase to drop matters \u2014 "
         f"a sweep over all K choices (Figure 7, panel c) shows the best "
         f"choice is the dominant phase in four of five systems. The "
         f"observed degradation is consistent with the residual phase "
         f"absorbing the accumulated prediction error of the other "
         f"K{MINUS}1 channels; we note this as an interpretation supported "
         f"by the sweep, not a directly demonstrated mechanism.")

    figure(doc, "fig6_failures",
           "Figure 7. Three negative results. (a) Sparsemax: per-seed test "
           "MAE (orange) versus the renorm head (green) across all five "
           "systems. The seed-to-seed spread of sparsemax is of the same "
           "order as its mean. (b) Sum-to-one penalty on Fe\u2013Cr\u2013Ni: "
           "MAE (bars, left axis) and closure violation (line, right axis) as "
           "\u03bb increases. The penalty buys closure at the cost of the "
           "fit. (c) Residue sweep: per-phase MAE for each choice of dropped "
           "phase (red) versus the best constrained head (green star) across "
           "all five systems.")

    # 10. Predicted phase fields
    doc.add_heading("10. Predicted phase fields", level=1)
    body(doc,
         f"Figure 8 shows the predicted phase-fraction fields for "
         f"{LBL['fecrni']} under the renorm head, alongside the CALPHAD "
         f"oracle, for all four target phases. The surrogate reproduces the "
         f"phase-boundary structure \u2014 the BCC, FCC, liquid and sigma "
         f"fields \u2014 with visible accuracy. The error panel shows that "
         f"errors concentrate at phase boundaries, where the fraction changes "
         f"rapidly, and are small in the interior of single-phase fields.")
    body(doc,
         f"We quantified this concentration on the {LBL['fecrni']} test set. "
         f"For each row we computed the distance, in scaled (x\u2082, "
         f"x\u2083, T) coordinates, to the nearest test row with a different "
         f"active-phase set (a KD-tree boundary proxy; a differing set was "
         f"found within 64 neighbours for 92.9% of rows), and binned the "
         f"per-row mean absolute error by that distance. This proxy "
         f"measures proximity to regions where the active-phase set changes "
         f"in the sampled data; it is not the exact thermodynamic distance "
         f"to a phase boundary. Rows in the nearest "
         f"third to the boundary proxy carry a mean MAE of 0.022, against "
         f"0.008 in the middle third and 0.002 in the farthest third \u2014 "
         f"a boundary-to-interior ratio of ~10{TIMES}. In this system, "
         f"surrogate error is strongly concentrated near regions where the "
         f"active-phase set changes, while the interior of phase fields is "
         f"reproduced to high accuracy.")
    figure(doc, "fig3_fields",
           f"Figure 8. Predicted phase-fraction fields for {LBL['fecrni']} "
           f"under the renorm head (seed 42). Top row: CALPHAD oracle. Middle "
           f"row: surrogate prediction. Bottom row: absolute error. Errors "
           f"concentrate near regions where the active-phase set changes; "
           f"the interior of single-phase fields is reproduced to high "
           f"accuracy.")

    # 11. Discussion
    doc.add_heading("11. Discussion", level=1)
    doc.add_heading("11.1 When do constrained MLPs beat trees?", level=2)
    body(doc,
         f"In the five systems studied here, the answer tracked geometry. "
         f"Systems with extended smooth coexistence fields \u2014 "
         f"{LBL['fecrni']} and {LBL['fecrmn']} \u2014 favoured the smooth "
         f"function class of the MLP, with {LBL['fecrmo']} a statistical "
         f"tie between the families. Among the systems dominated by sharp, "
         f"essentially binary phase boundaries, {LBL['femnni']} favoured "
         f"axis-aligned tree splits, whereas {LBL['fecrv']}, another "
         f"sharp-boundary system, is a statistical tie between the "
         f"families. We present this as an empirical "
         f"observation across the tested systems, not a universal law: with "
         f"only five systems, system identity is confounded with phase "
         f"topology, rare-phase frequency, sampling density, and boundary "
         f"geometry. A surrogate pipeline should evaluate both families and "
         f"select per system; no single evaluated architecture consistently "
         f"dominated across the five systems.")
    doc.add_heading("11.2 The spatial-generalisation result reframes the "
                    "interpolation comparison", level=2)
    body(doc,
         f"The holdout results of Section 7 reframe the main comparison. On "
         f"{LBL['fecrni']} the MLP leads the random forest by 41% on the "
         f"interpolation split; on the contiguous composition band the gap "
         f"widens to 3.11/1.61 {APPROX} 1.9{TIMES} in penalty ratio, and the "
         f"same ordering persists under the tested strict one-sided "
         f"extrapolation protocols. Under the region-matched reanalysis "
         f"(Section 7.1), which compares identical band rows with and "
         f"without regional training data, the in-distribution-to-holdout "
         f"penalty remains smaller for the constrained MLPs than for the "
         f"tree ensembles in nine of the ten band{TIMES}system "
         f"combinations. The MLP's advantage is not merely that "
         f"it fits the training distribution better; within the tested "
         f"protocols it also degrades less severely in absolute error when "
         f"the query point leaves it. We do not claim that the MLP "
         f"extrapolates reliably: all evaluated model families degrade "
         f"sharply in absolute terms, and the MLP's advantage is one of "
         f"relative degradation. Part of the temperature-protocol "
         f"degradation is a closed-basis limitation rather than a "
         f"spatial-generalisation failure: liquid is absent from the "
         f"near-side training data in four of the five systems, and in "
         f"{LBL['femnni']} all four model families come within 4% of the "
         f"constant near-side-mean predictor. Extrapolation conclusions "
         f"should therefore be read as joint failures of basis and "
         f"generalisation. For alloy-design workflows that query "
         f"unseen compositions, this relative comparison is the more "
         f"relevant one.")
    doc.add_heading("11.3 What the surrogate is and is not", level=2)
    body(doc,
         "These results frame the surrogate as a high-speed screening "
         "filter rather than a thermodynamic arbiter: while the MLP "
         "speedups enable the screening of multi-million composition "
         "databases in seconds, any candidate alloy identified during "
         "high-throughput screening must be independently re-validated "
         "using direct CALPHAD calculations and, ultimately, targeted "
         "experimental synthesis. Measured on CPU, the 192-unit MLP forward "
         f"pass costs 2\u20133 \u00b5s per point in batch inference "
         f"(0.4\u20130.6 ms for a single query), against roughly 2.5 s per "
         f"equilibrium on the CALPHAD side of this work \u2014 a speedup of "
         f"about 10{sup('6')} in batch mode and "
         f"4\u20136{TIMES}10{sup('3')} for single queries.")
    body(doc,
         "The surrogate emulates the oracle: for a queried (x, T) it returns "
         "the phase fractions the database's equilibrium routine would "
         "return, fast. It inherits the oracle's assumptions \u2014 the "
         "database's Gibbs energies, the equilibrium assumption itself, and "
         "the solver's convergence behaviour. It is not an independent "
         "thermodynamic statement. \u201cValidated against the database\u201d "
         "here means \u201creproduces the database's answers to solver "
         "precision on the validated points\u201d, not \u201cthermodynamically "
         "true\u201d.")
    doc.add_heading("11.4 Limitations", level=2)
    body(doc,
         "Phase-set sufficiency is validated on 1,500 re-solved points per "
         "system, not "
         "proven for the whole design space; a phase active only inside an "
         "unprobed region would be missed, and the phase basis itself was "
         "fixed with knowledge of the whole sampled domain, so all results "
         "are conditional on this globally predefined phase basis and do "
         "not assess discovery of previously unseen phases (Section 2). The "
         "closed phase set also bounds what the strict extrapolation "
         "protocols can test: liquid lies outside the near-side basis in "
         "four of the five temperature-extrapolation systems, so part of "
         "the far-side error there is a basis limitation. The "
         "contiguous-band holdout is a missing-region test inside the "
         "sampled domain; the strict extrapolation protocol is one-sided "
         "and covers only one coordinate at a time, and behaviour beyond "
         "the sampled simplex remains unvalidated. The sparsemax and "
         "penalty conclusions are scoped to the tested hyperparameters. The "
         "design distribution is deliberately structured, not uniform, so "
         "all reported average errors are expectations under that sampling "
         "distribution; rankings could shift under a different prior. All "
         "MLPs share one fixed architecture and all baselines use fixed "
         "hyperparameters; the comparison is a fixed-protocol benchmark, "
         "not an exhaustive hyperparameter search, and the reported MLP is "
         "not architecture-optimal. The mean \u00b1 standard deviation over "
         "three seeds quantifies seed-to-seed training stochasticity only, "
         "not total predictive uncertainty. Per-experiment seed counts are "
         "small; the sparsemax variance claim in particular rests on three "
         "runs. The phase-boundary analysis "
         "uses an empirical boundary proxy, not the exact thermodynamic "
         "distance to a phase boundary, and the proxy inherits the sampling "
         "density of the design; computing true boundary distances from a "
         "dense CALPHAD grid would decouple the two. Finally, the "
         "comparison set is not "
         "exhaustive over function classes: classical low-dimensional "
         "interpolants such as radial-basis-function or spline fits, or "
         "Gaussian-process regression, might approximate these "
         "three-dimensional phase-fraction fields competitively, and we did "
         "not evaluate them; the k-NN baseline partially covers the "
         "local-smoothness regime. Compositional-data-analysis regressors "
         "such as log-ratio (ALR/CLR/ILR) regression and Dirichlet "
         "regression are likewise not evaluated: the targets contain "
         "structural zeros \u2014 only one to three of the K phases are "
         "active on any given row \u2014 and log-ratio transforms require "
         "strictly positive components, so these families would need ad "
         "hoc zero replacement before fitting. Two-stage or presence-gated "
         "classify-then-regress architectures \u2014 the common remedy for "
         "the regressor\u2013detector gap \u2014 were likewise not "
         "evaluated. All conclusions are furthermore scoped to three-input "
         "ternary systems; whether the MLP/tree crossover, the band-holdout "
         "behaviour, and the regressor\u2013detector gap persist in "
         "higher-dimensional design spaces, where axis-aligned tree "
         "partitions lose their advantage, is untested.")

    # 12. Conclusions
    doc.add_heading("12. Conclusions", level=1)
    body(doc,
         "We compared simplex-constraint mechanisms for neural-network "
         "phase-fraction prediction \u2014 six output heads across all five "
         "systems plus two mechanism probes \u2014 against four classical "
         "baselines across five Fe-based ternary systems, totalling 44,397 "
         "mass-balance-validated equilibria. The contribution is a "
         "systematic empirical benchmark with controls, not a new "
         "architecture.")
    body(doc,
         f"Structurally constrained heads \u2014 softmax, sigmoid/{SIG}, and "
         f"renorm \u2014 satisfy the full simplex constraint (closure and "
         f"non-negativity) to float32 precision at no accuracy cost relative "
         f"to the unconstrained sigmoid, which violates closure by "
         f"6\u20138%. The residue head enforces closure by construction but "
         f"not non-negativity, emitting negative fractions on up to 1.2% of "
         f"cells. Post-hoc renormalisation repairs all evaluated models to machine "
         f"precision and did not hurt accuracy in any of the five systems "
         f"tested.")
    body(doc,
         "No single evaluated architecture consistently dominated across the "
         "five systems; performance is topologically dependent. Within the "
         "five systems studied, constrained MLPs led the two systems "
         "dominated by continuous, smooth phase coexistence fields, the "
         "random forest led the one system defined by sharp, binary phase "
         "boundaries, and the remaining two systems were statistical ties "
         "\u2014 a 2\u20131\u20132 pattern across the five systems \u2014 as "
         "paired-bootstrap tests confirmed.")
    body(doc,
         f"Two spatial protocols with size-matched random controls, "
         f"complemented by a region-matched reanalysis, separated "
         f"the spatial penalty from the data-volume penalty. On contiguous "
         f"interior holdout bands \u2014 which lie 99.3\u201399.4% inside "
         f"the convex hull of the remaining training points, and are "
         f"therefore a distribution-shift test rather than extrapolation "
         f"\u2014 the random forest paid 3.11{TIMES} against 1.61{TIMES} for "
         f"the constrained MLP on the {LBL['fecrni']} composition band, "
         f"whereas on the temperature band both paid 1.31{TIMES}: the "
         f"temperature-band penalty is largely a data effect. Under strict "
         f"one-sided out-of-range extrapolation all evaluated model "
         f"families degraded sharply, and the constrained MLP achieved the "
         f"lowest absolute far-side error in all five systems, although "
         f"absolute extrapolation performance remains poor. The temperature "
         f"protocol is additionally limited by the closed phase basis, as "
         f"liquid is unseen on the near-side training data in four of the "
         f"five systems. Constrained "
         f"MLPs degrade more gracefully than "
         f"trees when queries leave the trained region; graceful degradation "
         f"is not reliable extrapolation. For alloy-design workflows, "
         f"band-limited spatial shifts are tolerable for constrained MLPs, "
         f"whereas one-sided extrapolation should be treated as unsupported "
         f"and re-validated with direct CALPHAD evaluation, and any "
         f"workflow in which rare phases matter should complement the "
         f"regressor with a dedicated presence detector or a presence-gated "
         f"head (not evaluated here).")
    body(doc,
         f"Phase-presence detection degrades faster than regression under "
         f"distribution shift: F{SUB['1']} collapses on the held-out "
         f"temperature band for all evaluated models even where MAE barely "
         f"moves. The model with the lowest MAE is not necessarily the best "
         f"detector \u2014 regression accuracy and phase-presence detection "
         f"are distinct tasks \u2014 and the qualitative gap is stable "
         f"across presence thresholds 10{sup('-4')}\u201310{sup('-2')}, "
         f"although the detector ranking flips with the threshold on "
         f"{LBL['fecrmo']}.")
    body(doc,
         f"Three negative results are informative. Under the evaluated "
         f"configuration sparsemax exhibited severe seed sensitivity, "
         f"consistent with a zero-gradient sparsity trap. For the tested "
         f"formulation and {LAMBDA} values, a sum-to-one penalty was "
         f"redundant and strongly detrimental. The residue head, which "
         f"closes the sum algebraically, is 1.3\u20132.5{TIMES} worse than "
         f"the best constrained head in all five systems, consistent with the "
         f"residual phase absorbing the accumulated error of the other "
         f"channels.")

    # Data availability
    doc.add_heading("Data availability", level=1)
    body(doc,
         "The complete pipeline (TDB extraction, probe, data generation, "
         "training, evaluation, and all analyses reported in this paper, "
         "including the region-matched reanalysis, AUPRC, phase-rule "
         "admissibility, and the reconstruction of the design-space "
         "accounting) is available at [GitHub URL]. The five datasets "
         "(44,397 equilibria with per-phase fractions and thermodynamic "
         "potentials), the stored per-seed model predictions, and all "
         "result files are archived at [Zenodo DOI]. The MatCalc "
         "mc_fe_v2.062 source database is distributed under its own "
         "license and is not redistributed here; scripts reproducing the "
         "ternary extraction from a local copy are provided.")
    body(doc,
         "The generation pipeline validates mass balance at acceptance time, "
         "and all figures and tables are generated programmatically from "
         "stored result artefacts; no reported value is hand-entered. The "
         "recompute audit re-derives every metric from the stored prediction "
         "tensors and confirms agreement to below 10\u207b\u2079 on all seven "
         "summary fields for all 99 MLP runs.")

    # References
    doc.add_heading("References", level=1)
    for ref in [
        "Breiman, L., 2001. Random forests. Mach. Learn. 45, 5\u201332.",
        "Chen, T., Guestrin, C., 2016. XGBoost: A scalable tree boosting "
        "system. Proc. KDD, 785\u2013794.",
        "Feng, S., Fu, H., Zhou, H., Wu, Y., Lu, Z., Dong, H., 2021. A "
        "general and transferable deep learning framework for predicting "
        "phase formation in materials. npj Comput. Mater. 7, 22.",
        "Hart, G.L.W., Mueller, T., Toher, C., Curtarolo, S., 2021. Machine "
        "learning for alloys. Nat. Rev. Mater. 6, 730\u2013755.",
        "Jiang, X., Zhang, R., Zhang, C., Yin, H., Qu, X., 2019. Fast "
        "prediction of the quasi phase equilibrium in phase field model for "
        "multicomponent alloys based on machine learning method. Calphad "
        "66, 101644.",
        "Lu, J., Xu, G., Chen, F., Cui, Y., 2024. Classified dataset, "
        "regression and machine learning modeling for prediction of phase "
        "transformation temperatures in steels. Calphad 87, 102748.",
        "Lukas, H.L., Fries, S.G., Sundman, B., 2007. Computational "
        "Thermodynamics: The Calphad Method. Cambridge University Press.",
        "Martins, A.F.T., Astudillo, R.F., 2016. From softmax to sparsemax: A "
        "sparse model of attention and multi-label classification. Proc. "
        "ICML, 1614\u20131623.",
        "MatCalc, 2023. Open thermodynamic databases. https://www.matcalc.at/. "
        "Accessed 2024.",
        "Nentwich, C., Engell, S., 2019. Surrogate modeling of phase "
        "equilibrium calculations using adaptive sampling. Comput. Chem. "
        "Eng. 126, 204\u2013217.",
        "Otis, R., Liu, Z.-K., 2017. pycalphad: CALPHAD-based computational "
        "thermodynamics in Python. J. Open Res. Softw. 5, 1.",
        "Saunders, N., Miodownik, A.P., 1998. CALPHAD: A Comprehensive Guide. "
        "Pergamon, Oxford.",
        "Tahkola, M., Linnala, L., Blackburn, T., Savukoski, S., Gagneur, "
        "V., Kaipainen, J., Ma, K., Knowles, A.J., Laukkanen, A., Pinomaa, "
        "T., 2026. Accelerated discovery of Cr-based A2+B2 superalloys "
        "across 11 elements with a deep-learning CALPHAD surrogate. npj "
        "Comput. Mater. https://doi.org/10.1038/s41524-026-02113-x.",
        "Wang, X., Xiong, W., 2020. Uncertainty quantification and "
        "composition optimization for alloy additive manufacturing through "
        "a CALPHAD-based ICME framework. npj Comput. Mater. 6, 188.",
    ]:
        para(doc, ref, size=10, space_after=4)

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
