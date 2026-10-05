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

NEW_TITLE = ("Validated high-throughput screening with simplex-constrained "
             "surrogates for CALPHAD phase-fraction prediction")

NEW_ABSTRACT = (
    "Abstract: Machine-learning surrogates accelerate CALPHAD equilibrium "
    "calculations by orders of magnitude, but deployment needs three answers: "
    "how to predict phase fractions under the simplex constraint, where the "
    "predictions can be trusted, and whether screening actually saves cost. We "
    "benchmark six neural-network output heads and a presence-gated two-stage "
    "head against four classical baselines on 88,542 mass-balance-validated "
    "equilibria from nine systems: five Fe-based ternaries, a quaternary "
    "Fe\u2013Cr\u2013Ni\u2013C system, a carbide Fe\u2013Cr\u2013C system, and two Fe-free "
    "ternaries. Structurally constrained heads close the simplex to machine "
    "precision at no accuracy cost. A presence-gated head resolves the "
    "regressor\u2013detector gap, lifting macro-AUPRC to 0.91\u20130.99 on all nine "
    "systems while leading the random forest in MAE. Size-matched and "
    "region-matched controls with an ideal-form control show interior-band "
    "penalties are spatial, not volumetric, while strict extrapolation fails "
    "on every axis. Ensemble disagreement provides cheap per-row triage inside "
    "the trained region. Two validated lean-nickel screens (501,501 and "
    "330,000 compositions; shortlist precision 1.00 and 0.98) pay back data "
    "costs several times over. Failed mechanisms (sparsemax, sum-to-one "
    "penalty, residue head) delimit the scope.")

HIGHLIGHTS = [
    "Simplex closure to machine precision costs no accuracy; renorm is enough",
    "Presence supervision, not architecture, resolves sharp phase boundaries",
    "Interior-band penalties are spatial, not volumetric; extrapolation unsupported",
    "Lean-nickel screen: 501,501 compositions in 0.4 s, shortlist precision 1.00",
    "Failed mechanisms delimit the scope: sparsemax, sum-to-one penalty, residue",
]

NEW_KEYWORDS = ("Keywords: CALPHAD; Machine learning surrogate; Simplex constraint; "
                "Spatial generalization; Phase fraction prediction; "
                "High-throughput screening.")

SCOPE_PARA = (
    "To address these interconnected challenges, this study systematically "
    "investigates how neural network architectures can reliably predict "
    "multi-phase equilibria while strictly satisfying physical mass "
    "conservation. Based on 88,542 rigorously validated thermodynamic equilibria "
    "from nine systems \u2014 five Fe-based ternary systems (Fe\u2013Cr\u2013Ni, "
    "Fe\u2013Cr\u2013Mn, Fe\u2013Cr\u2013Mo, Fe\u2013Cr\u2013V, Fe\u2013Mn\u2013Ni), a quaternary "
    "Fe\u2013Cr\u2013Ni\u2013C system, a carbide Fe\u2013Cr\u2013C system, and two Fe-free "
    "ternaries (Cr\u2013Co\u2013Ni, Cr\u2013Ni\u2013Mn) \u2014 we evaluate the limits of deep "
    "learning surrogates under rigorous distribution shifts and demonstrate two "
    "validated high-throughput screens. These benchmark systems cover highly "
    "relevant metallurgical phases, including the industrially vital FCC "
    "solid-solution structures found in stainless steels and nickel-based "
    "superalloys.")

CONTRIB_PARA = (
    "The specific contributions of this work are sixfold: First, we benchmark six "
    "distinct neural network output configurations against four classical "
    "baselines to conclusively resolve how structural outputs affect both "
    "continuous regression accuracy and physical admissibility. Second, we "
    "establish decoupled spatial generalization protocols \u2014 a contiguous "
    "interior-band holdout and a strict one-sided extrapolation task \u2014 utilizing "
    "size-matched random controls plus a region-matched reanalysis with an "
    "ideal-form control to strictly separate spatial penalties from data volume "
    "reductions. Third, we characterize the regressor\u2013detector gap and close "
    "most of it with a presence-gated two-stage head. Fourth, we show ensemble "
    "disagreement provides cheap per-row error triage inside the trained region. "
    "Fifth, we re-run the full protocol on four scope-extension systems "
    "(quaternary, carbide, and two Fe-free chemistries) to test generality. "
    "Finally, we validate the pipeline end to end with two high-throughput "
    "lean-nickel screens and a four-point experimental anchor, and we report "
    "informative negative results for three constraint mechanisms to establish "
    "clear design guidelines.")

STOCO_PARA = (
    "The closest adjacent work predicts single-temperature phase labels rather "
    "than fraction vectors: Stoco et al. [33] classify single-phase FCC/BCC "
    "against multiphase states in a ten-element space, with lever-rule data "
    "augmentation for the minority class, while this work regresses full "
    "fraction vectors across temperature with the simplex enforced structurally. "
    "Their blind gridding left 21.5% of points non-converged where the balance "
    "element vanishes; our probe-driven design space with a mass-balance gate "
    "avoids that failure mode by construction, at the price of the "
    "sampling-coverage confound discussed in the Limitations.")

EXT_DATA_PARA = (
    "Four additional systems test whether the conclusions survive changes of "
    "dimension and chemistry under the identical probe, acceptance, and split "
    "protocol. The quaternary Fe\u2013Cr\u2013Ni\u2013C system adds interstitial carbon "
    "(13,192 accepted rows; 9 active phases among 35 eligible, including M23C6, "
    "M7C3, CEMENTITE, and GRAPHITE). The ternary Fe\u2013Cr\u2013C system (13,199 "
    "rows; 9 active among 32, with SIGMA in only 0.4% of rows) tests "
    "carbide-dominated diagrams. Two Fe-free ternaries reuse the identical "
    "ternary protocol: Cr\u2013Co\u2013Ni (8,874 rows; 5 active among 23; HCP_A3 "
    "rarest at 0.7%) and Cr\u2013Ni\u2013Mn (8,880 rows; 9 active among 30, "
    "including Mn intermetallics). Together the nine datasets contain 88,542 "
    "mass-balance-validated equilibria.")

GATED_METHOD_PARA = (
    "To close the regressor\u2013detector gap we evaluate a presence-gated "
    "two-stage head under the same fixed protocol. A shared 3\u00d7192 trunk feeds "
    "two per-phase heads: a fraction head (per-phase sigmoid outputs with "
    "inverse-prevalence phase-weighted Huber loss) and a presence head "
    "(per-phase logits with class-weighted binary cross-entropy at the 10\u207b\u00b3 "
    "presence threshold). The prediction is the elementwise product of the two "
    "sigmoid outputs, followed by clip-and-renormalize projection at evaluation. "
    "A weighted-only ablation trains just the fraction head to isolate the "
    "contribution of presence supervision.")

SPATIAL_APPEND = (
    " As a complementary reference we also report a region-matched reanalysis: "
    "for each band, the band-holdout error is compared against the main-study "
    "models on the identical band rows inside the interpolation test set. Its "
    "residual confound is a roughly 15% training-size difference, which an "
    "ideal-form control removes by retraining at equal size (band rows kept in "
    "training, an equal number of random non-band rows dropped instead) on "
    "identical test rows.")

FIG1_CAPTION = (
    "Figure 1. Study pipeline. (1) CALPHAD data generation: five Fe-based ternary "
    "subsystems are extracted from the open MatCalc steel database, probe-driven "
    "phase sets fix the target dimensionality, structured sampling draws 11,220 "
    "candidate points per system, and a mass-balance gate accepts 44,397 equilibria "
    "in total; four scope-extension systems (quaternary Fe\u2013Cr\u2013Ni\u2013C, "
    "Fe\u2013Cr\u2013C, and Fe-free Cr\u2013Co\u2013Ni and Cr\u2013Ni\u2013Mn) reuse the same "
    "pipeline and add 44,145 equilibria (88,542 in total). (2) A cluster-stratified "
    "64/16/20 split feeds a constrained MLP with six fraction-only heads and a "
    "presence-gated two-stage head, plus four classical baselines, all fitted under "
    "one fixed protocol and three seeds. (3) Three evaluation protocols \u2014 "
    "interpolation, contiguous interior-band holdout, and strict one-sided "
    "extrapolation, the two spatial ones each with a size-matched random control "
    "\u2014 plus secondary analyses (ablations, boundary-error analysis, "
    "region-matched reanalysis with an ideal-form control, and uncertainty "
    "screening, dashed arrows). (4) Deployment: a four-point experimental "
    "anchor and two validated high-throughput screens (a 501,501-composition "
    "ternary screen and a 330,000-composition quaternary screen), with "
    "ensemble disagreement gating the ternary screen.")

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
# New results prose (same simple style as Manuscript 4)
# --------------------------------------------------------------------------
GATED_H2 = "Closing the Detector Gap with a Presence-Gated Head"
GATED_PS = [
    ("Separating the presence decision from the fraction estimate closes most "
     "of the gap at no regression cost on the ternaries (Table 11, Figure 9). "
     "Across the five ternaries the bare renorm MLP macro-AUPRC spans 0.36\u20130.99 "
     "and the gated head 0.91\u20130.99; the gated head beats the best classical "
     "detector on four of the five systems (Fe\u2013Cr\u2013V is a 0.988 against 0.990 "
     "near-tie) while reducing test MAE on all five, by 6% (Fe\u2013Cr\u2013Ni) to 72% "
     "(Fe\u2013Mn\u2013Ni, 0.0140 to 0.0039). Per-phase scores show where the remedy "
     "matters: on Fe\u2013Cr\u2013Mn ALPHA_MN rises from 0.004 to 0.933 and BETA_MN "
     "from 0.005 to 0.896; on Fe\u2013Mn\u2013Ni MNNI rises from 0.003 to 1.00. One "
     "rare phase resists \u2014 the 0.1%-occurrence MU_PHASE_I in Fe\u2013Cr\u2013Mo stays "
     "below 0.5. The weighted-only ablation is insufficient (0.51\u20130.99, and it "
     "degrades MAE by 28% on Fe\u2013Cr\u2013V): presence supervision, not loss "
     "reweighting, is the operative ingredient."),
    ("The remedy transfers without retuning. On quaternary Fe\u2013Cr\u2013Ni\u2013C the "
     "gated head raises macro-AUPRC from 0.805 to 0.965 (CEMENTITE 0.09 to 0.873, "
     "GRAPHITE 0.58 to 0.926) at a 12% MAE cost; on Fe\u2013Cr\u2013C from 0.559 to "
     "0.922 at a 47% cost. On the Fe-free systems detection comes nearly for free: "
     "Cr\u2013Co\u2013Ni rises from 0.799 to 0.979 (HCP_A3, invisible to the joint head "
     "at 0.02, reaches 0.91) while MAE improves from 0.0055 to 0.0045, and "
     "Cr\u2013Ni\u2013Mn from 0.877 to 0.971 with MAE improving from 0.0040 to 0.0036."),
    ("Under shift the remedy mostly holds except where detection itself collapses. "
     "On contiguous bands the gated head matches or beats the fraction-only MLP "
     "while crushing the forest, with macro-AUPRC at 0.96\u20131.00. Under composition "
     "extrapolation it wins on Fe\u2013Cr\u2013Ni (0.147 against 0.172) but loses on "
     "Fe\u2013Cr\u2013V, where the presence head misfires out of distribution; under "
     "temperature extrapolation all families fail together and gated AUPRC collapses "
     "to 0.59\u20130.63. Two caveats bound this result: the remedy was designed after "
     "seeing the bare regressor fail on these test sets, so its numbers are "
     "post-hoc; and the shift evaluation covers two systems only."),
]
TABLE11_CAP = (
    "Table 11. Presence-gated remedy. Macro-AUPRC (mean over three seeds; "
    "zero-positive phases excluded) for the renorm MLP, the three tree/k-NN "
    "baselines, the phase-weighted ablation, and the gated two-stage head; the "
    "final column is the gated-head test MAE relative to the renorm MLP (below 1 "
    "means the gated head also improves regression). Best AUPRC per row in bold.")
FIG9_CAP = (
    "Figure 9. Macro-AUPRC by system and family (mean over three seeds). The "
    "gated two-stage head (rightmost bar in each group) matches or exceeds the "
    "best classical detector on six of the seven tabulated systems and lifts the "
    "constrained MLP far above its bare regression ranking on the rare-phase "
    "systems.")

UNC_H2 = "Uncertainty Triage with Ensemble Disagreement"
UNC_P1 = (
    "A deployment needs to know when not to trust the surrogate. From artefacts "
    "the protocol already produces we evaluate two per-row scores on the five "
    "ternary systems: ensemble disagreement (U1), the per-row standard deviation "
    "across the three seed models averaged over phases; and input-space distance "
    "(U2), the distance to the nearest training row in standardized coordinates. "
    "U1 carries real signal: its Spearman correlation with per-row error averages "
    "0.707 (0.94 on Fe\u2013Cr\u2013Ni, 0.93 on Fe\u2013Cr\u2013Mn, 0.87 on Fe\u2013Cr\u2013V, 0.69 "
    "on Fe\u2013Cr\u2013Mo, but 0.11 on Fe\u2013Mn\u2013Ni). Retaining the 50% lowest-U1 rows "
    "cuts MAE from 0.0123 to 0.0029 on Fe\u2013Cr\u2013Ni (a factor of 4.3) and by "
    "similar factors on Fe\u2013Cr\u2013Mn and Fe\u2013Cr\u2013Mo \u2014 but the curve is flat or "
    "reversed on Fe\u2013Cr\u2013V and Fe\u2013Mn\u2013Ni, so the gate must be validated once "
    "per system before use. Ensemble averaging itself is only a small refinement "
    "(mean gain 0.35% on fully paired rows).")
UNC_P2 = (
    "Input-space distance has no power (mean AUROC 0.408, below random): under "
    "interpolation every test row is close to training rows, while the hardest rows "
    "sit near phase boundaries. Nor can disagreement replace the spatial protocols: "
    "models trained on the same data can agree confidently and wrongly where "
    "training carries no information. The division of labour is simple: U1 for cheap "
    "per-row triage inside the trained region, holdout protocols or direct CALPHAD "
    "revalidation anywhere else.")

REGION_H2 = "Region-Matched Reanalysis and the Ideal-Form Control"
REGION_P1 = (
    "Because random controls remove different rows than the band, penalty ratios "
    "conflate region difficulty with distribution shift. The region-matched "
    "reanalysis compares band-holdout error against the main-study models on the "
    "identical band rows (Table 12, panel A). Under this measure the constrained "
    "MLP penalty is smaller than the tree ensembles\u2019 in nine of the ten "
    "band\u2013system combinations: on composition bands 1.2\u20132.1 versus 1.4\u20133.0 for "
    "the forest; on temperature bands 0.9\u20132.8 versus 1.3\u20133.9, with Fe\u2013Cr\u2013Ni "
    "the single exception (2.8 against 2.1, though absolute holdout error stays "
    "twice the MLP\u2019s, 0.0132 against 0.0258). One instability sits behind the "
    "headline: on Fe\u2013Mn\u2013Ni the MLP temperature ratios spread 0.9/20.3/0.6 "
    "across seeds from a near-zero denominator, downweighted in the ratio-of-means "
    "headline (1.1).")
REGION_P2 = (
    "The ideal-form control (Table 12, panel B) then confirms the penalty is "
    "spatial, not volumetric: at equal training size and on identical test rows, "
    "the training-size effect stays within 0.97\u20131.12 (mean 1.03) while removing "
    "the band still costs 0.93\u20133.84 (mean 2.01). The MLP degrades least in nine "
    "of ten combinations. Trust tracks where the data are, not how many there are.")
TABLE12_CAP = (
    "Table 12. Region-matched band reanalysis. Panel A: holdout mean MAE on band "
    "rows divided by in-distribution mean MAE on the identical rows (ratio of "
    "means; per-seed ratios in parentheses). Panel B repeats the comparison in "
    "ideal form \u2014 identical test rows and equal training size \u2014 reporting the "
    "band-in/band-out ratio H/I with the training-size effect I/main in parentheses.")

EXT_H2 = "Extension to Quaternary, Carbide, and Fe-Free Systems"
EXT_PS = [
    ("The full protocol re-run on the four extension systems shows the "
     "smooth-coexistence ranking survives four inputs (Table 13). On quaternary "
     "Fe\u2013Cr\u2013Ni\u2013C the renorm head reaches 0.0037 MAE against 0.0075 for the "
     "forest \u2014 a twofold margin, larger than on any ternary \u2014 and on Fe\u2013Cr\u2013C "
     "sigmoid/\u03a3 reaches 0.0029 against 0.0055. Beyond iron, renorm leads "
     "Cr\u2013Co\u2013Ni (0.0055 against 0.0070) and sigmoid/\u03a3 leads nine-phase "
     "Cr\u2013Ni\u2013Mn (0.0036 against 0.0052); sparsemax, which collapses on every Fe "
     "system, competes on dilute nine-phase targets (0.0048)."),
    ("The detector gap widens exactly where phases get rarer, and the gated head "
     "transfers with no retuning: the quaternary bare MLP misses the rarer carbides "
     "and Fe\u2013Cr\u2013C collapses to 0.559 macro-AUPRC, while the gated head reaches "
     "0.965 and 0.922; on the Fe-free systems the bare MLP misses HCP_A3 (0.02) and "
     "MNNI, and the gated head reaches 0.979 and 0.971 at best-or-tied MAE. The "
     "spatial protocols give the same qualitative answers: composition-band holdout "
     "costs about 2\u20133.5 times the control and strict extrapolation fails on every "
     "tested axis, worst in temperature (22\u201324 times on the transfer systems) \u2014 "
     "except Cr\u2013Co\u2013Ni bands, which test easier than the control (0.8 times, "
     "both bands FCC-dominated)."),
]
TABLE13_CAP = (
    "Table 13. Scope-extension systems. (A) System and dataset statistics. "
    "(B) Test MAE (mean over three seeds), best per system in bold. "
    "(C) Contiguous-band and strict-extrapolation holdouts against size-matched "
    "random controls. (D) Phase-set validation against the full eligible set.")

ANCHOR_H2 = "Experimental Anchor"
ANCHOR_P = (
    "Four published DTA transition temperatures bound the surrogate\u2019s "
    "contribution against the database\u2019s own disagreement with experiment "
    "(Figure 10). Three Fe\u2013C\u2013Cr alloy transitions come from Drozdov\u2019a et "
    "al. [36] and one Fe\u2013Cr\u2013Ni liquidus from Yamada et al. [37]. At the "
    "10\u207b\u2074 detection threshold the surrogate matches full CALPHAD within 21 K "
    "(mean 17.75 K) \u2014 the same order as the database\u2019s own mean deviation "
    "from experiment (19 K, median 8 K, driven by a single +59 K outlier). The "
    "surrogate term is larger on three of the four alloys, and with n = 4 neither "
    "term can be declared limiting.")
FIG10_CAP = (
    "Figure 10. Experimental anchor: CALPHAD and surrogate liquid/solid fractions "
    "versus temperature for four alloys with published DTA transition temperatures "
    "(dotted markers). Three Fe\u2013C\u2013Cr alloys from Drozdov\u2019a et al. [36] and "
    "one Fe\u2013Cr\u2013Ni alloy from Yamada et al. [37].")

SCREEN_H2 = "High-Throughput Screening"
SCREEN_PS = [
    ("We screen Fe\u2013Cr\u2013Ni at T = 1000 K \u2014 below liquid appearance, at a "
     "service-relevant temperature \u2014 for a lean-nickel austenitic window: fully "
     "austenitic (FCC \u2265 0.99), sigma-free and melt-free (each \u2264 10\u207b\u00b3), with "
     "nickel capped at 12 mol% (Table 14, Figure 12). The screen evaluates 501,501 "
     "compositions on a 10\u207b\u00b3 simplex grid in 0.4 s, shortlisting 10,936 hits "
     "(2.18%). Every shortlisted alloy checked validates: all 11 Pareto-frontier "
     "points (minimal Ni per Cr bin, falling from 7.5% Ni at 1.5% Cr to 4.8% near "
     "12\u201313% Cr) confirm under the full 28-phase set, as do all 400 points of a "
     "random hit subset \u2014 and a follow-up run validated the remaining 10,525 with "
     "zero misses: 10,936 of 10,936. Treating the in-scope background miss rate "
     "(4 of 76) as representative, recall is about 0.8; the misses cluster at "
     "decision boundaries, and ensemble spread flags the worst one (Figure 11, "
     "panel a; AUROC 0.86; the "
     "clear failure carries ten times the shortlist 95th-percentile spread). The "
     "threshold sweep over the 811-point pool gives precision 1.00 at all nine "
     "FCC\u2013sigma combinations (Figure 11, panel b; Table 14)."),
    ("One screen amortises the pipeline several times over: against a measured "
     "0.41 s per full-set solve and 268 s training, break-even is near 10\u2074 "
     "points \u2014 the 501,501-point screen pays back roughly 7 to 50 times in a "
     "single query. Half the training data buys most of the accuracy (0.0121 at 50% "
     "against 0.0091 full), and commercial-grade projections (AISI 304 correctly "
     "refused on its 3.7% CALPHAD sigma; ferritic 430 refused in agreement) show no "
     "grade-name bias."),
    ("With carbon in the query the screen still works \u2014 and its failures mark "
     "the model\u2019s edge (Table 15, Figure 13). On quaternary Fe\u2013Cr\u2013Ni\u2013C, "
     "330,000 compositions screen in 0.5 s with 14,475 hits (4.39%); the frontier "
     "reaches 2.9% Ni at near-zero Cr, carbon substituting for both Ni and Cr. Of 20 "
     "frontier points 15 confirm; the 5 rejections form one contiguous Cr pocket "
     "where the screen over-smooths a two-phase region at its decision edge. "
     "Combined shortlist precision is 0.98 (14,143/14,475), recall about 0.8, "
     "disagreement gates transfer (AUROC 0.856), and break-even near 10\u2074 points "
     "means the screen pays back roughly 10 to 30 times."),
]
TABLE14_CAP = (
    "Table 14. Application case: lean-Ni austenitic screen at 1000 K on "
    "Fe\u2013Cr\u2013Ni. Break-even is the one-time cost divided by the measured "
    "0.41 s per-point full-set saving.")
FIG11_CAP = (
    "Figure 11. Screen operating characteristics. (a) Shortlist spread "
    "distribution (10,936 points) with the four in-scope misses and the "
    "90/95/99th-percentile gates. (b) Pool precision and recall across the "
    "FCC\u2013sigma threshold grid: precision is 1.00 at all nine combinations.")
TABLE15_CAP = (
    "Table 15. Application case: quaternary lean-Ni screen with carbon at "
    "1000 K on Fe\u2013Cr\u2013Ni\u2013C.")
FIG13_CAP = (
    "Figure 13. Quaternary screen with carbon at 1000 K. (a) Pareto frontier of "
    "minimal Ni versus Cr: 15 of 20 frontier points confirm; the 5 rejections form "
    "one contiguous pocket. (b) Surrogate-versus-CALPHAD FCC parity on the "
    "validated shortlist.")
FIG12_CAP = (
    "Figure 12. Lean-Ni austenitic screen at 1000 K. (a) Gibbs triangle coloured "
    "by surrogate FCC: shortlist, Pareto frontier of minimal Ni per Cr bin (all 11 "
    "full-CALPHAD-confirmed) and the four in-scope misses. (b) The frontier as "
    "minimal qualifying Ni versus Cr. (c) Data efficiency: test MAE versus "
    "training-set size.")

DISCUSS_ADD1 = (
    " The ideal-form control sharpens this picture into an operational rule: at "
    "equal training size, removing a contiguous band still doubles the error on "
    "average (0.93\u20133.84, mean 2.01) while equal-size subsampling elsewhere "
    "changes it by only 3% \u2014 trust tracks where the data are, not how many "
    "there are. For screening loops this means coverage of the query region matters "
    "more than sheer dataset size, and the constrained MLP degrades least in nine "
    "of ten band\u2013system combinations.")
DISCUSS_ADD2 = (
    " The two validated screens quantify the filter economics: break-even near "
    "10\u2074 points against payback of 7\u201350 times (ternary) and 10\u201330 times "
    "(quaternary) in a single query, with shortlist precision 1.00 and 0.98 and "
    "recall about 0.8 on both. The honest boundary stands: one temperature each on "
    "two systems, and trusting the screen still requires spot CALPHAD \u2014 the "
    "screens above spent about 11,000 full-set solves on validation. Screening is "
    "cheap; certification is not free.")
LIMIT_ADD = (
    " Three further bounds apply. Only one solver generated all labels, so solver "
    "disagreement is unexplored. All nine systems come from one assessed steel "
    "database, which isolates model behaviour from database variation but holds "
    "database error fixed. And the gated remedy was evaluated under shift on two "
    "systems only, with one composition-extrapolation reversal.")

CONC_ADD = [
    ("Presence Supervision Closes the Gap: The gated two-stage head lifts "
     "macro-AUPRC to 0.91\u20130.99 on all nine systems (best classical detector "
     "beaten on eight) while leading the forest in MAE \u2014 presence supervision, "
     "not architecture, resolves the sharp-boundary cases."),
    ("Spatial, Not Volumetric: The ideal-form control holds training size fixed and "
     "still doubles the error on band removal (mean 2.01) against a 3% size effect "
     "\u2014 trust tracks coverage, and strict extrapolation remains unsupported on "
     "every axis tested."),
    ("Generality Across Dimension and Chemistry: Rankings, gap, remedy, and "
     "protocols transfer to a quaternary, a carbide ternary, and two Fe-free "
     "ternaries (88,542 equilibria total), with the sparse head competitive only on "
     "dilute nine-phase targets."),
    ("Screening Pays: Two validated lean-nickel screens (501,501 and 330,000 "
     "compositions; precision 1.00 and 0.98) pay back data costs several times over, "
     "with ensemble disagreement as a shortlist triage gate and a four-point "
     "experimental anchor bounding database disagreement."),
]

NEW_REFS = [
    "33.\tStoco, C.B., et al., Accelerating phase prediction via CALPHAD-informed "
    "machine learning and data augmentation. Computational Materials Science, "
    "2026. 272: p. 114834.",
    "34.\tOtis, R. and Z.-K. Liu, pycalphad: CALPHAD-based computational "
    "thermodynamics in Python. Journal of Open Research Software, 2017. 5: p. 1.",
    "35.\tMatCalc, Open thermodynamic databases. 2023. https://www.matcalc.at/.",
    "36.\tDrozdov\u2019a, L\u2019., et al., Temperatures of liquidus, solidus and "
    "peritectic transformation of Fe\u2013C\u2013Cr based alloys. In: METAL 2017 \u2014 26th "
    "International Conference on Metallurgy and Materials, Brno, Czech Republic, "
    "2017. pp. 65\u201370.",
    "37.\tYamada, A., et al., Determination of liquidus and solidus surfaces at "
    "iron-rich region of Fe\u2013Cr\u2013Ni system. Tetsu-to-Hagan\u2019e, 1987. 73: pp. "
    "1676\u20131683.",
]

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
    insert_para(doc, k, TABLE11_CAP, after=True)
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
    insert_table(doc, k + 1, table11, after=True)
    add_figure(doc, k + 2, os.path.join(FIG, "fig8_remedy.png"), FIG9_CAP)

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
    assert [n for k, n in seq if k == "Table"] == list(range(1, 16)), seq
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