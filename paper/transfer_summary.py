"""Compact transfer summary for the two non-Fe systems (Cr-Co-Ni, Cr-Ni-Mn).

Aggregates the standard battery (heads, baselines, remedy, band holdout)
into models/transfer_summary.json and prints the transfer table, and
additionally emits models/results_{s}_summary.json in the exact schema of
the scope-extension summaries so tab/extension.tex covers the transfer
systems with zero changes to the existing rows. Existing paper result
files are never touched; all inputs are per-system files.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M = os.path.join(ROOT, "models")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fe_surrogate.systems import SYSTEMS as CFG
SEEDS = [42, 123, 2024]
SYSTEMS = ["crconi", "crnimn"]
HEADS = ["mlp_sigmoid", "mlp_softmax", "mlp_residue", "mlp_renorm",
         "mlp_sig_norm", "mlp_sparsemax"]
BASES = ["rf_renorm", "xgb_renorm", "knn_renorm", "ridge_renorm"]
HOLD = ["mlp_renorm", "mlp_sig_norm", "rf_renorm", "xgb_renorm"]
BLOCKS = ["random_ctrl", "T_band", "X2_band"]


def jload(p):
    with open(p) as f:
        return json.load(f)


def bagg(d, tag, field="mean_mae"):
    v = [d[f"{tag}_s{k}"][field] for k in SEEDS if f"{tag}_s{k}" in d]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


out = {}
for s in SYSTEMS:
    heads = jload(os.path.join(M, f"results_heads_{s}.json"))
    base = jload(os.path.join(M, f"results_baselines_{s}.json"))
    rem = jload(os.path.join(M, f"results_remedy_{s}.json"))
    hold = jload(os.path.join(M, f"block_holdout_{s}.json"))
    entry = {"heads": {}, "baselines": {}, "holdout": {}, "remedy": {}}
    for h in HEADS:
        m, sd = bagg(heads, h)
        entry["heads"][h] = {"mean_mae": m, "sd": sd}
    for b in BASES:
        m, sd = bagg(base, b)
        entry["baselines"][b] = {"mean_mae": m, "sd": sd}
    for blk in BLOCKS:
        entry["holdout"][blk] = {}
        for hk in HOLD:
            m, sd = bagg(hold, f"{blk}_{hk}")
            entry["holdout"][blk][hk] = {"mean_mae": m, "sd": sd}
    agg = rem.get("aggregated", {}).get(s, {})
    entry["remedy"] = {
        "weighted_macro_auprc": agg.get("weighted", {}).get("macro_auprc"),
        "gated_macro_auprc": agg.get("gated", {}).get("macro_auprc"),
        "gated_mae": agg.get("gated", {}).get("mae"),
        "comparison": rem.get("comparison", {}).get(s, {}),
    }
    out[s] = entry

with open(os.path.join(M, "transfer_summary.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---- extension-schema summaries (one per system) ----
BASE_KEYS = ["xgb_raw", "xgb_renorm", "rf_raw", "rf_renorm",
             "ridge_raw", "ridge_renorm", "knn_raw", "knn_renorm"]
EXT_BLOCKS = ["T_band", "X2_band", "random_ctrl"]
EXT_XBLOCKS = ["X2_extrap", "X2_extrap_ctrl", "T_extrap", "T_extrap_ctrl"]


def seed_mean(d, tag, field="mean_mae"):
    v = [d[f"{tag}_s{k}"][field] for k in SEEDS if f"{tag}_s{k}" in d]
    return float(np.mean(v)) if v else np.nan


for s in SYSTEMS:
    cfg = CFG[s]
    raw = os.path.join(ROOT, "data", "raw")
    df = pd.read_csv(os.path.join(raw, f"dataset_{s}.csv"))
    probe = jload(os.path.join(raw, f"{s}_probe.json"))
    active = jload(os.path.join(raw, f"{s}_active.json"))
    heads = jload(os.path.join(M, f"results_heads_{s}.json"))
    base = jload(os.path.join(M, f"results_baselines_{s}.json"))
    rem = jload(os.path.join(M, f"results_remedy_{s}.json"))
    band = jload(os.path.join(M, f"block_holdout_{s}.json"))
    ext = jload(os.path.join(M, f"holdout_extrap_{s}.json"))
    psv = jload(os.path.join(ROOT, "analysis_revision",
                             f"phase_set_validation_{s}.json"))[s]
    phase_names = sorted(probe["active_counts"].keys())
    per_nonzero = {p: float((df[f"NP_{p}"] > 0.001).mean())
                   for p in phase_names}
    heads_mean = {h: seed_mean(heads, h) for h in HEADS}
    heads_seed = {}
    for h in HEADS:
        for k in SEEDS:
            e = heads[f"{h}_s{k}"]
            heads_seed[f"{h}_s{k}"] = {
                "mean_mae": e["mean_mae"], "consistency": e["consistency"],
                "stainless_box_mae": e["stainless_box_mae"]}
    base_mean = {b: seed_mean(base, b) for b in BASE_KEYS}

    def block_mean(d, blk):
        vals = [d[f"{blk}_{hk}_s{k}"]["mean_mae"]
                for hk in HOLD for k in SEEDS
                if f"{blk}_{hk}_s{k}" in d]
        return float(np.mean(vals)) if vals else np.nan

    band_pm = {b: block_mean(band, b) for b in EXT_BLOCKS}
    ext_pm = {b: block_mean(ext, b) for b in EXT_XBLOCKS}
    summary = {
        "system": s,
        "elements": [e for e in cfg["elements"] if e != "VA"],
        "generated_from": [f"results_baselines_{s}.json",
                           f"results_heads_{s}.json",
                           f"results_remedy_{s}.json",
                           f"phase_set_validation_{s}.json",
                           f"block_holdout_{s}.json",
                           f"holdout_extrap_{s}.json"],
        "dataset": {
            "rows": int(len(df)),
            "n_phases": len(phase_names),
            "phases": phase_names,
            "max_abs_sum_np_minus_1": float(active["max_sum_deviation"]),
            "in_stainless_box_fraction": float(df["in_stainless_box"].mean()),
            "converged_fraction": float(df["converged"].mean()),
            "per_phase_nonzero": per_nonzero,
            "probe_active_counts": probe["active_counts"],
            "T_range_K": [cfg["t_min"], cfg["t_max"]],
        },
        "baselines_mae_mean_over_seeds": base_mean,
        "heads_mae_mean_over_seeds": heads_mean,
        "heads_per_seed": heads_seed,
        "remedy": {"aggregated": rem.get("aggregated", {}).get(s, {}),
                   "runs": rem.get("runs", {})},
        "phase_set_validation": psv,
        "holdout": {
            "band": {"per_block_mean_mae": band_pm,
                     "overall_mean_mae": float(np.mean(list(band_pm.values()))),
                     "n_entries": 36},
            "extrap": {"per_block_mean_mae": ext_pm,
                       "overall_mean_mae": float(np.mean(list(ext_pm.values()))),
                       "n_entries": 48},
        },
    }
    with open(os.path.join(M, f"results_{s}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved models/results_{s}_summary.json")

for s in SYSTEMS:
    e = out[s]
    print(f"=== {s} ===")
    for h in HEADS:
        d = e["heads"][h]
        print(f"  {h:16s} MAE={d['mean_mae']:.4f}+/-{d['sd']:.4f}")
    for b in BASES:
        d = e["baselines"][b]
        print(f"  {b:16s} MAE={d['mean_mae']:.4f}+/-{d['sd']:.4f}")
    r = e["remedy"]
    print(f"  remedy weighted AUPRC={r['weighted_macro_auprc']} "
          f"gated AUPRC={r['gated_macro_auprc']} gatedMAE={r['gated_mae']}")
    for blk in BLOCKS:
        cells = " ".join(f"{hk.split('_')[0] if '_' not in hk else hk[:6]}="
                          f"{e['holdout'][blk][hk]['mean_mae']:.4f}"
                          for hk in HOLD)
        print(f"  {blk:12s} {cells}")
print("Saved models/transfer_summary.json")
