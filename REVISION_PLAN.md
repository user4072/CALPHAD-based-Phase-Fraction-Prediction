# Revision Plan / Reviewer-Response Ledger

Mapping of every point from the Q1-level review to its current status.
Statuses: **fixed in text** (manuscript edited), **analysis added** (computed
from stored predictions in `analysis_revision/`, no retraining),
**needs experiment** (requires new compute; proposed protocol given).

## Major issues

| # | Review point | Status | Where / how |
|---|---|---|---|
| 1 | Phase-set reduction validated with wrong criterion | **fixed + analysis added** | `analysis_revision/validate_phase_set_full.py` re-solves 1,500 points/system with the FULL eligible set (28–31 phases) and compares per-phase NP **and** GM. 4/5 systems: max |ΔNP| ≤ 2.5e−9, max |ΔGM| ≤ 7.8e−5 J/mol. Fe–Cr–Mo: 1–3 of 1500 points (varying between repeated runs — the solves are nondeterministic there) in the Mo-rich, <710 K corner differ (max |ΔNP| = 0.71, ΔG up to ~1 kJ/mol) — all phases involved are already probe-active, i.e. near-degenerate solver minima, not an incomplete target set. Manuscript text rewritten accordingly. Composition-set summation (BCC α/α′) now stated. MNNI2 explained (probe distribution ≠ structured design distribution; 0.1% probe rate, 0 rows in design). Probe-detection limitation (<0.05% phases) stated. |
| 2 | Spatial "penalty ratio" conflates region difficulty with shift | **fixed + analysis added** | Confound acknowledged in Methods + Table 8 caption. Region-matched reanalysis added (same band rows with vs without regional training data): MLP penalty < tree penalty in 9/10 band×system combos (composition bands: MLP 1.2–2.1 vs RF/XGB 1.4–3.0 in all five systems). Results + Discussion rewritten with the corrected framing. Ideal protocol (train with band included, N-matched, test on held-out band rows) listed below as the remaining refinement. |
| 3 | Temperature extrapolation partly ill-posed (LIQUID unseen) | **fixed + analysis added** | Verified: LIQUID has zero near-side (T ≤ 1200 K) presence in 4/5 systems. Fe–Mn–Ni: all four model families within 4% of the constant near-side-mean predictor (0.182–0.184 vs 0.190); unseen-LIQUID column contributes 0.088 of the 7-phase mean. Reframed as closed-basis limitation (basis + generalization failure jointly) in Results, Discussion, Table 9 caption, Conclusions. X2-protocol verified to keep all frequent phases in-basis (genuine spatial test). |
| 4 | Soft-penalty failure likely a gradient-scaling artifact | **fixed in text (caveat)** | λ ∈ {1, 10} only — confirmed from stored runs. Caveat added: with Huber δ = 0.01, the closure term dominates the regression gradient by 1–2 orders of magnitude; no smaller λ / rescaled penalty / warm-up evaluated; conclusion scoped to "adds nothing over structural heads at the tested weights". **Needs experiment:** λ sweep ∈ {1e−4 … 10} with penalty normalised to the Huber scale + warm-up. |
| 5 | Sparsemax tested in the known-failing configuration | **fixed in text (caveat)** | Confirmed: Huber regression loss on projected outputs, logits ×4. Caveat added: dedicated sparsemax loss, α-entmax (α = 1.5), and softmax→sparsemax annealing untested; a working sparse head remains desirable for downstream phase-field use (exact zeros). **Needs experiment:** sparsemax-loss + entmax variants. |
| 6 | No hyperparameter search → "topological dependency" weak | **partly fixed** | Fixed-protocol disclaimer already present and retained; Fe–Cr–Mo "MLP-superiority" contradiction fixed (it is a tie; Fe–Cr–V also a tie, now stated); topology explanation now labelled post-hoc. **Needs experiment:** equal-budget HPO (e.g. 50 Optuna trials/family/system) + per-system topology descriptor (multi-phase row fraction, mean Shannon entropy, or gradient-norm estimate) correlated with ΔMAE. |
| 7 | Missing baselines (GP/RBF, CoDA/ALR, Dirichlet head, two-stage) | **fixed in text (limitation); presence-gated head done** | Named explicitly in Limitations with rationale (structural zeros break log-ratio transforms; zero replacement is ad hoc). **Done (cycle 2):** the presence-gated multi-task / two-stage head is implemented and evaluated under the fixed protocol (§6.1, `models/results_remedy*.json`). **Needs experiment:** GP/RBF interpolator, ALR with multiplicative zero replacement, Dirichlet-likelihood head. |
| 8 | Regressor–detector gap needs remedy + fairer evaluation | **done** | Zero-positive phases excluded (MLP 0.312→0.364 vs RF 0.504→0.588 in Fe–Mn–Ni — gap persists); per-phase AUPRC added (MLP rare-phase AP 0.001–0.05 for 9/10 rare phases vs RF 0.40–0.97 except two near-zero-occurrence phases); macro-F1 convention stated; F1-under-shift now reported for all five systems (not single-system evidence). **Mitigation (cycle 2, done):** presence-gated two-stage head, §6.1 + `models/results_remedy.json` — macro-AUPRC 0.36–0.99 → 0.91–0.99 across the five ternaries at a 6–72% MAE cost (and 0.56/0.81 → 0.92/0.97 on the extension systems at 12–47%). |
| 9 | Dimensionality: add quaternary | **fixed in text (scope + limitation); quaternary done, learning curves needs experiment** | Ternary scoping sentence added in Introduction; Limitations now state that the tests run up to the quaternary system; Intro motivation reframed (ground truth for ternaries costs hours; the study is a methodology audit whose questions transfer to high-D). **Done (cycle 2):** quaternary Fe–Cr–Ni–C (13,192 rows) + carbon-rich Fe–Cr–C (13,199) generated and evaluated under the identical full protocol (§12, `tab/extension.tex`, `models/*fecrnic*`, `models/*fecrc*`). **Needs experiment:** learning curves (MAE vs N). |

## Moderate issues

| # | Point | Status | Where / how |
|---|---|---|---|
| Head definitions ambiguous (power-norm undefined; renorm clipping; post-hoc vs in-graph) | **fixed in text** | Power-norm defined (sigmoid^p, p = 2, row-normalised, train-time, Fe–Cr–Ni only, † in Table 3); renorm head = independently trained sigmoid model + inference-time projection outside the training graph; clipping only active for baselines. |
| Speedup asserted, never measured | **analysis added** | Measured on CPU: 2–3 µs/point batch (0.4–0.6 ms single query) vs 2.5 s/point CALPHAD → ~10⁶× (batch) / 4–6×10³× (single). Added in Discussion. |
| Gibbs phase rule admissibility | **analysis added** | >3 phases above 10⁻³: ground truth 0% everywhere; renorm MLP 0% in 4 systems (3.4% Fe–Cr–Mo); RF 0–8.9%; XGBoost up to 22.4% (Fe–Cr–Mo). New paragraph in Results. |
| Error metrics hide tails | **analysis added** | Active-cell MAE (0.023–0.074 for renorm MLP), P95 row error 0.03–0.12, worst cell ≈ complete miss. Added in Results. |
| Uncertainty (ensembles) | **done (with negative result for U2)** | §8 uncertainty screening: U1 three-seed ensemble disagreement as an OOD/disagreement gate — mean AUROC 0.695 (≈0.70) for rows with MAE > 0.05, free accuracy from the seed mean (−2.3% MAE on average), but the coverage–risk curve reverses on 2/5 systems (Fe–Cr–V, Fe–Mn–Ni), so the gate must be validated per system before use. U2 nearest-training-row distance has **no power** (mean AUROC 0.408, below chance). |
| Splits clarification | **fixed in text** | Split is re-drawn per seed (spread = split + training stochasticity); cluster-stratified split ≈ stratified random in effect; 3 seeds thin for the sparsemax variance claim (noted). |
| Fe–Cr–V control anomaly (0.0234 ± 0.0073) | **fixed in text (data)** | Per-seed control values 0.027/0.014/0.031 — no diverged seed; the "(73)" last-digit notation was misread as ±0.070. Clarified in Table 9 caption. |
| Convex-hull check trivial | **accepted (softened)** | Kept as completeness check; Table 8 caption no longer claims the random control "isolates the spatial penalty". |
| Boundary proxy density-dependence | **kept + alternative named** | Existing caveat retained; true-boundary computation from a dense CALPHAD grid listed as future work in Limitations. |
| Threshold sensitivity overgeneralised | **already fixed previously** | Detector ranking changes in Fe–Cr–Mo are already stated; qualitative-gap-vs-exact-ranking distinction retained. |
| Redundancy §3.6/§4.5 | **fixed in text** | Discussion negative-results section condensed to one paragraph referencing Results; caveats retained. |
| "Transductive probing" misnomer | **fixed in text** | Now "phase-set pre-screening" (both occurrences; "transductive architectures" usage in the OOD context kept — correct ML term). |
| Citations [17], [24], [12], [3] | **fixed in docx fields** | [17]→[8] Tahkola swap; [24] removed from both detector-gap contexts; [12] removed (pharmacology); [3] removed from CALPHAD-fundamentals instances. Bibliography regenerates on EndNote update. |
| Missing tool citations (pycalphad, MatCalc, sparsemax, entmax, Aitchison, XGBoost, scikit-learn, AdamW, Lukas et al.) | **user TODO (EndNote)** | Requires adding records to the EndNote library — see list at the end of this file. |
| Eq. (1) numbering | **check pending** | See final checklist. |
| Table 2 "Box (%)" undefined | **fixed in text** | Defined in the Table 2 caption (stainless-steel box). |
| Figures 8/2 redesign | **needs experiment (compute)** | Isothermal-section overlays proposed in the plan below. |
| Conclusions decision guideline | **fixed in text** | "Practical Guidance" paragraph added (defaults; presence head for rare phases; extrapolation unsupported → re-validate with CALPHAD). |
| Code/data availability | **fixed in text** | Availability paragraph added with GitHub/Zenodo placeholders. |

## Tone and over-claiming

Swept: "massively", "rigorously" (×8), "conclusively", "definitively",
"dramatically", "catastrophically", "paramount", "exciting and vital",
"highly comprehensive", "near-perfect" (tree interpretation), "profoundly",
"entirely prohibitive", "superiority/excel/dominate" (abstract + conclusions
now state the 2–1–2 pattern with two ties), "every model" → "all evaluated
model families", "guarantees" (kept only for structural closure, which is
mathematical), "never hurts" (already fixed previously; verified again).

## Remaining experiments (prioritised)

1. **Region-matched controls, ideal form** — retrain with band included and
   N-matched by subsampling elsewhere; test on band rows held out from
   within the band. (Removes the ~15% training-size confound.)
   **DONE (2026-10-04):** `analysis_revision/region_matched_ideal.py` →
   `models/region_matched_ideal.json` (90 records + `__agg__`, 180 seeded
   fits, 5 ternaries × 3 seeds × 2 bands × 3 renorm models × 2 arms).
   Size effect I/main 0.970–1.116 (mean 1.033) → old confound
   second-order; band removal H/I 0.93–3.84 (mean 2.01) → penalty
   spatial; MLP smallest H/I in 9/10 combos (means 1.49 vs 2.25/2.28).
   Panel B of tab:region + findings passage + docx Table 13 ideal rows;
   audit `g_region_ideal`. Note: record keys must embed the system —
   key collision silently overwrote the first system's records (fixed).
2. **Soft-penalty λ sweep** with Huber-scale normalisation + warm-up
   (λ ∈ {1e−4 … 10}).
3. **Sparsemax loss / entmax-1.5 / annealing** variants.
4. **Equal-budget HPO** for MLP, RF, XGBoost (50 Optuna trials per family
   per system) + per-system topology descriptors correlated with ΔMAE.
5. **Missing baselines**: GP/RBF, ALR-CoDA (zero replacement), Dirichlet
   head, two-stage / presence-gated heads (also serves as the
   regressor–detector-gap remedy).
6. **One quaternary system** + learning curves (MAE vs N).
7. **Deep-ensemble uncertainty** as an extrapolation/OOD gate.
8. **True phase-boundary distances** from a dense CALPHAD isothermal grid
   (replaces the sampling-density-dependent proxy).
9. **Figure upgrades**: predicted vs CALPHAD isothermal sections (1073 K,
   1273 K) + one vertical section with boundary contours.

## EndNote TODO (user actions in Word)

1. Open the manuscript in Word → EndNote tab → **Update Citations and
   Bibliography** (regenerates the bibliography after the field edits; the
   visible bibliography currently still lists the removed [3], [12], [17],
   [24] entries).
2. Add and cite (suggested locations):
   - **pycalphad**: Otis, R. & Liu, Z.-K. *pycalphad* (2017) — at "integrated
     with pycalphad" in Thermodynamic Description.
   - **MatCalc database**: the mc_fe_v2.062 source/database citation — at
     "open MatCalc steel database".
   - **CALPHAD fundamentals**: Lukas, Fries & Sundman, *Computational
     Thermodynamics* (2007) or Saunders & Miodownik (1998) — at "The CALPHAD
     method provides…".
   - **sparsemax**: Martins & Astudillo (ICLR 2016) — at the sparsemax head
     definition and/or the sparsemax limitation.
   - **entmax**: Peters et al. (ACL 2019) — at the sparsemax limitation
     (α-entmax).
   - **Aitchison** (1986) — at the CoDA paragraph in Limitations.
   - **XGBoost**: Chen & Guestrin (2016) — at the baseline-models list.
   - **scikit-learn**: Pedregosa et al. (2011) — at the baseline-models list.
   - **AdamW**: Loshchilov & Hutter (2019) — at the optimizer sentence.
3. Fill the availability placeholders `[GitHub URL]` / `[Zenodo DOI]` in the
   Conclusions section once the repository is pushed and the archive is
   minted.

## Revision cycle 2 (current)

Added in this cycle (all numbers verified by `paper/audit_claims.py` —
PASS=373, FAIL=0, NV=7 — and cross-referenced in `paper/claim_ledger.md`):

- **Projection analysis** — post-hoc projection of the stored sigmoid head
  reproduces the renorm head to ≤1.3×10⁻⁴ MAE in every system
  (`paper_cms.tex` §5, `models/projection_ablation.json`).
- **Presence-gated remedy** — gated two-stage head closes most of the
  regressor–detector gap: macro-AUPRC 0.36–0.99 → 0.91–0.99 across the five
  ternaries while *reducing* MAE by 6–72% (§6.1, `models/results_remedy.json`).
- **Uncertainty screening** — ensemble-disagreement gate works as a ranker
  (mean AUROC 0.70) but fails coverage–risk on 2/5 systems; the NN-distance
  gate has no power (AUROC 0.408) (§8).
- **Scope extensions** — quaternary Fe–Cr–Ni–C (13,192 rows) + carbon-rich
  Fe–Cr–C (13,199) run through the identical protocol: gated macro-AUPRC
  0.965 / 0.922 at 12–47% MAE cost, renorm MAE 0.0037 / 0.0029 vs RF
  0.0075 / 0.0055, temperature-extrapolation penalties 15.2× / 20× (§12).
- **Experimental anchor** — 4 digitised DTA transition temperatures
  (Drozdová 2017 A/B/C: 1498/1471/1397 °C; Yamada 1987: 1441 °C): surrogate
  vs full-set CALPHAD mean 17.75 K (max 21 K); CALPHAD vs experiment mean
  19.25 K (max 59 K) (§13, `paper/anchor_eval.py`, `paper/anchor_data/`).
- **Claim-audit infrastructure** — `paper/claim_ledger.md` +
  `paper/audit_claims.py` (373 checks + 7 not-verifiable) tie every tex
  number to an artefact.
- **Seven datasets, 70,788 rows** — five ternaries (44,397) + Fe–Cr–Ni–C
  (13,192) + Fe–Cr–C (13,199) (§2); design-box generation is now
  dimension-generic (6 strategies).

Net effect on the ledger: Major 7 (presence-gated head), Major 8 (remedy),
Major 9 (quaternary), and Moderate "Uncertainty (ensembles)" are now done;
"Remaining experiments" items 5–7 are only partially consumed — GP/RBF, ALR,
Dirichlet head, learning curves, the ideal N-matched region protocol (Major
2 refinement), λ sweep, entmax, HPO, boundary distances, and figure upgrades
remain open.
