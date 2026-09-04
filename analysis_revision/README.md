# Revision analyses

All analyses in this directory are computed **from stored artefacts**
(stored per-seed test predictions in `models/pred_*.npz`, result JSONs in
`models/`, and the raw datasets in `data/raw/`) — no model was retrained.
They back the revisions described in `REVISION_PLAN.md`.

## Files

| File | Purpose |
|---|---|
| `revision_analyses.py` | All stored-prediction analyses (below). Writes `revision_analyses.json` and prints a human-readable log (`analysis_log.txt`). |
| `revision_analyses.json` | Machine-readable results of every analysis (1–11 below). |
| `validate_phase_set_full.py` | Re-runs the phase-set validation (same 1,500 sampled points, same seed 11 as the original protocol) with the **full eligible phase set**, additionally comparing total molar Gibbs energy (GM). Writes `phase_set_validation.json`. |
| `phase_set_validation.json` | Per-system validation summary: n re-solved, max |ΔNP|, max |ΔGM|, failing points, missing phases. |
| `fecrmo_discrepancy.py` | Characterizes the Fe–Cr–Mo validation discrepancies (coordinates, per-phase amounts, ΔG). Output captured in `fecrmo_check.log`. |
| `validation.log` / `fecrmo_check.log` | Run outputs (evidence trail). |
| `analysis_log.txt` | Console log of `revision_analyses.py`. |

## What each analysis answers

1. **Region-matched band controls** — for each contiguous band (T ∈
   [1200, 1400] K; x₂ ∈ [0.25, 0.35]) and each holdout model/system/seed:
   the MAE of the **main-study models on the same band rows inside the
   interpolation test set** (band data present in training) versus the
   **band-holdout MAE** (band removed). Ratio-of-means isolates the regional
   shift on identical rows (residual confound: ~15% training-size
   difference). Result: MLP penalty < tree penalty in 9/10 band × system
   combinations; composition bands MLP 1.2–2.1 vs RF/XGB 1.4–3.0.
2. **Extrapolation seen/unseen decomposition** — near-side (training) and
   far-side phase-presence structure per protocol; per-phase far-side MAE per
   model; the constant near-side-mean predictor as a reference floor.
   Result: LIQUID unseen on the T-protocol near side in 4/5 systems;
   Fe–Mn–Ni T-extrap within 4% of the constant predictor for all four
   families.
3. **Gibbs-phase-rule admissibility** — fraction of test rows with >3 phases
   above the 10⁻³ (and 5×10⁻³) presence thresholds. Ground truth 0% everywhere;
   renorm MLP 0% in four systems (3.4% Fe–Cr–Mo); RF 0–8.9%; XGBoost up to
   22.4%.
4. **Active-cells MAE and tail statistics** — cell-MAE restricted to true
   fraction > 0.005; P95 of per-cell and per-row error; max cell error.
5. **Detection recomputed** — macro-F1 with zero-positive phases excluded;
   per-phase AUPRC (average precision) with the fraction itself as the score.
6. **F1 under the band holdouts** — macro-F1 (interpolation vs T-band vs
   X2-band) for the MLP and RF in all five systems.
7. **Fe–Cr–V random-control audit** — per-seed control MAEs
   (0.027/0.014/0.031): ordinary seed variability, no diverged run.
8. **Sparsemax per-seed spread** — seed-wise MAEs per system (variance on the
   order of the mean, confirming the reported instability).
9. **renorm identity check** — `renorm(sigmoid predictions)` versus the
   stored renorm-model predictions (they differ: renorm is an independently
   initialized/trained sigmoid model + inference-time projection, not the
   same weights).
10. **Rejected-candidate reconstruction** — rebuilds the deterministic
    design (seed 42), diffs against the datasets: 11,220 raw draws → 2,340
    design-space exclusions → 8,880 valid candidates → 8,880 accepted
    (8,877 in Fe–Cr–Mo; 3 non-converged points, coordinates recorded).
11. **Inference timing** — forward-pass latency of the 192×192×192
    architecture (freshly initialized; latency is weight-independent) on
    CPU: 2.1–3.1 µs/row batch, 0.4–0.6 ms single query.

## Reproducing

```bash
py -3.12 -X utf8 analysis_revision/revision_analyses.py
py -3.12 -X utf8 analysis_revision/validate_phase_set_full.py --n 1500   # ~10 min/system
py -3.12 -X utf8 analysis_revision/fecrmo_discrepancy.py                 # ~10 min
```

Requires the pipeline environment (`requirements.txt`) and the raw datasets.
