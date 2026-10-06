***Simplex Constraints, Spatial Generalization, and the Regressor–Detector Gap in Machine-Learning Surrogates for CALPHAD method Phase-Fraction Prediction
Benchmark and failure-mode analysis of machine-learning surrogates that***

predict equilibrium phase fractions in five Fe-based ternary systems:
**Fe–Cr–Ni, Fe–Cr–Mn, Fe–Cr–Mo, Fe–Cr–V, Fe–Mn–Ni**.

The full study is described in the manuscript

(*"Simplex Constraints, Spatial Generalization, and the Regressor–Detector Gap
in Machine-Learning Surrogates for CALPHAD Phase-Fraction Prediction"*).

## What this repository contains

| Path | Contents |
|---|---|
| `databases/` | TDB extraction scripts only. Neither the MatCalc source database `mc_fe_v2.062` nor the derived ternary extractions are redistributed (see License); regenerate them from a licensed MatCalc copy with step 0 below. |
| `data/raw/` | The five datasets (44,397 mass-balance-validated equilibria with per-phase fractions, GM/HM/SM/CPM), probe sidecars, and design checkpoints. |
| `src/fe_surrogate/` | Shared library: system registry, splits, metrics, heads, TDB utilities. |
| `models/` | Per-seed result JSONs and stored test-set predictions (`pred_*.npz`) for every model × system × seed; source of every number in the paper. |
| `analysis_revision/` | Revision analyses computed from stored predictions (no retraining): region-matched band controls, extrapolation seen/unseen decomposition, AUPRC, phase-rule (Gibbs) admissibility, active-cell/tail errors, design-space accounting reconstruction, full-set phase validation, inference timing. |
| `paper/` | Manuscript source and figure generation scripts. 
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
   # the extended version used for the revision also records max|dGM|:
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
   py -3.12 holdout_eval.py                       # band mode, all five systems
   py -3.12 holdout_eval.py --mode extrap         # strict extrapolation mode
   ```
6. **Analyses** (paired bootstrap, threshold sensitivity, boundary-error
   proxy, residual sweep, and all revision analyses):
   ```bash
   py -3.12 paired_bootstrap.py
   py -3.12 threshold_sensitivity.py
   py -3.12 boundary_error.py
   py -3.12 -X utf8 analysis_revision/revision_analyses.py
   ```
7. **Paper artefacts** (figures, tables, workflow diagram):
   ```bash
   py -3.12 paper/figures.py
 
   ```

`REPORT.md` documents the pipeline stage by stage, including the per-strategy
sampling accounting and the validation ledger.

## Key experimental design decisions

- **Probe-driven phase sets**: model targets are the phases observed active in
  a 2,000-draw Dirichlet probe with the *full* eligible phase set; composition
  sets of a phase (e.g. BCC α/α′) are summed over their vertices.
- **Mass-balance acceptance**: only equilibria with |Σ NP − 1| < 10⁻⁶ are kept
  (8,880 of 8,880 valid candidates per system; the 11,220 → 8,880 reduction is
  a design-space filter, not solver rejections).
- **Fixed-protocol benchmark**: identical MLP backbone (192×192×192) and fixed
  baseline hyperparameters across all systems; three seeds; the split is
  re-drawn per seed (so seed spread includes split variation).
- **Metrics**: cell-averaged MAE, macro-F1 at the 10⁻³ presence threshold,
  closure/simplex violations, and (revision) AUPRC, active-cell MAE, and
  Gibbs-phase-rule admissibility.

## License

Code: MIT (see `LICENSE`). The MatCalc thermodynamic database
`mc_fe_v2.062.tdb` is **not** redistributed; obtain it from
[MatCalc](https://matcalc.at). The derived ternary extractions are also
**not** redistributed: they inherit the source database's terms, so users
must generate them from their own licensed copy with the included scripts
(step 0 of the pipeline).
