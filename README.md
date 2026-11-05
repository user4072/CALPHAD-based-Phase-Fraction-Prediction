# **Validated High-Throughput Screening with Simplex-Constrained Surrogates for CALPHAD Phase-Fraction Prediction**

Benchmark and failure-mode analysis of machine-learning surrogates that predict
equilibrium phase fractions across **nine alloy systems** and **88,542
mass-balance-validated equilibria**:

- **Five Fe-based ternaries**: Fe–Cr–Ni, Fe–Cr–Mn, Fe–Cr–Mo, Fe–Cr–V, Fe–Mn–Ni
- **Quaternary**: Fe–Cr–Ni–C (interstitial carbon)
- **Carbide ternary**: Fe–Cr–C
- **Fe-free ternaries**: Cr–Co–Ni, Cr–Ni–Mn

Six neural-network output heads and a presence-gated two-stage head are
benchmarked against four classical baselines under one fixed protocol, with
strict admissibility auditing, rare-phase detection, matched spatial controls,
and costed end-to-end screening.

## Headline results

- Structurally constrained heads close the simplex to machine precision at no
  accuracy cost relative to the unconstrained baseline; a capacity-matched
  control shows the ranking among constrained heads is partly capacity-driven.
- The presence-gated head reaches macro-AUPRC 0.88–0.99 on all nine systems and
  beats the best seed-matched classical detector on eight of nine (confirmed on
  three held-out seeds).
- Interior-band penalties are spatial, not volumetric: removing a band doubles
  the error at equal training size, while equal-size subsampling elsewhere
  changes it by ~3%.
- Strict one-sided extrapolation fails on every axis; temperature is worst
  (15–24× the matched control).
- Two lean-nickel screens (501,501 and 330,000 compositions) run in under a
  second, with shortlist precision 1.00 and 0.98, in-scope recall 0.75
  (95% CI 0.66–0.83) and 0.79 (0.67–0.89), and a measured same-hardware payback
  of 12–140× the one-time pipeline cost.

## What this repository contains

| Path | Contents |
|---|---|
| `databases/` | TDB extraction scripts only. Neither the MatCalc source database `mc_fe_v2.062` nor the derived extractions are redistributed (see License); regenerate them from a licensed MatCalc copy with step 0 below. |
| `data/raw/` | The nine datasets (88,542 mass-balance-validated equilibria with per-phase fractions, GM/HM/SM/CPM), per-system probe sidecars and design checkpoints, plus retemplated robustness variants. |
| `src/fe_surrogate/` | Shared library: system registry, splits, metrics, heads, TDB utilities. |
| `models/` | Per-seed result JSONs and stored test-set predictions (`pred_*.npz`) for every model × system × seed — including the capacity-matched control, the composition-grouped-split runs, and the fresh-seed confirmatory runs; source of every number in the study. |
| `analysis_revision/` | Extended analyses computed from stored predictions (no retraining): region-matched and ideal-form spatial controls, extrapolation seen/unseen decomposition, per-phase AUPRC, Gibbs-phase-rule admissibility, active-cell/tail errors, full-set phase validation, inference timing. |
| `paper/` | Verification and result data: the automated claim audit (`audit_claims.py`, 618 checks tying every reported number to a stored artifact), result figures, screening grids and full-set validation data, table data, and anchor-evaluation results. |
| `*.py` (root) | The pipeline: probing, data generation, training, evaluation, and statistical analyses. |

## Pipeline

Run everything from the repository root with Python 3.12:

```bash
pip install -r requirements.txt
```

0. **TDB extraction** (requires a local copy of the MatCalc database — place it at
   `databases/mc_fe_v2.062.tdb`; the generated `.tdb` files are gitignored and stay local):
   ```bash
   py -3.12 create_ternary_tdb.py        # five ternary extractions
   py -3.12 build_unpruned_tdbs.py       # unpruned references for equivalence checks
   py -3.12 validate_tdbs.py            # syntax/schema validation
   py -3.12 validate_tdb_equivalence.py # pruned vs unpruned equilibrium equivalence
   ```
1. **Phase-set pre-screening** (probe: which phases are ever active):
   ```bash
   py -3.12 probe_eligible.py --system fecrni --n 2000   # per system
   ```
2. **Data generation** (five deterministic sampling strategies; ~2.5 s per
   equilibrium, parallelised over 8 workers; ~1 h per system):
   ```bash
   py -3.12 generate_data.py --system fecrni
   ```
3. **Phase-set validation** (re-solve sampled points with the *full* eligible
   phase set and compare per-phase amounts and GM against the dataset):
   ```bash
   py -3.12 validate_phase_set.py --system fecrni --n 1500
   # the extended version used in the revised study also records max|dGM|:
   py -3.12 -X utf8 analysis_revision/validate_phase_set_full.py --n 1500
   ```
4. **Training** (six MLP output heads + four baselines, three seeds, fixed
   protocol):
   ```bash
   py -3.12 train_mlp.py --system fecrni           # heads incl. renorm/sigmoid identity
   py -3.12 train_baselines.py --system fecrni     # ridge / kNN / RF / XGBoost
   py -3.12 rerun_heads.py --system fecrni         # head re-runs for all systems
   ```
5. **Evaluation protocols** (interpolation; contiguous interior-band holdout;
   strict one-sided extrapolation; each spatial protocol with a size-matched
   random control):
   ```bash
   py -3.12 holdout_eval.py                       # band mode, five ternary systems
   py -3.12 holdout_eval.py --mode extrap         # strict extrapolation mode
   ```
6. **Analyses** (paired bootstrap, threshold sensitivity, boundary-error
   proxy, and the extended analyses):
   ```bash
   py -3.12 paired_bootstrap.py
   py -3.12 threshold_sensitivity.py
   py -3.12 boundary_error.py
   py -3.12 -X utf8 analysis_revision/revision_analyses.py
   ```
7. **Claim audit** (re-derives every reported number from the stored artifacts):
   ```bash
   py -3.12 -X utf8 paper/audit_claims.py
   ```

## Key experimental design decisions

- **Probe-driven phase sets**: model targets are the phases observed active in
  a 2,000-draw probe with the *full* eligible phase set; composition sets of a
  phase (e.g. BCC α/α′) are summed over their vertices.
- **Mass-balance acceptance**: only equilibria with |Σ NP − 1| < 10⁻⁶ are kept
  (8,880 of 11,220 candidates per ternary system — a design-space filter, not
  solver rejections; the carbon systems use bounded design boxes).
- **Fixed-protocol benchmark**: identical MLP backbone (192×192×192) and fixed
  baseline hyperparameters across all systems; three seeds; the split is
  re-drawn per seed (so seed spread includes split variation).
- **Metrics**: cell-averaged MAE, macro-F1 and macro-AUPRC at the 10⁻³ presence
  threshold, closure/simplex violations, active-cell MAE, and Gibbs-phase-rule
  admissibility.

## License

Code: MIT (see `LICENSE`). The MatCalc thermodynamic database
`mc_fe_v2.062.tdb` is **not** redistributed; obtain it from
[MatCalc](https://matcalc.at). The derived extractions are also
**not** redistributed: they inherit the source database's terms, so users
must generate them from their own licensed copy with the included scripts
(step 0 of the pipeline).
