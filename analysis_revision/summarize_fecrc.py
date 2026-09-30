"""Consolidate all fecrc (Fe-Cr-C) pipeline outputs into one summary.

Reads (whichever exist):
  data/raw/dataset_fecrc.csv + fecrc_probe.json           dataset stats
  models/results_baselines_fecrc.json                     baseline MAE
  models/results_heads_fecrc.json                         MLP head MAE
  models/results_remedy_fecrc.json                        remedy MAE + AUPRC
  analysis_revision/phase_set_validation_fecrc.json       full-phase-set checks
  analysis_revision/detection_fecrc.json                  AUPRC/F1 table
  models/block_holdout_fecrc.json                         band holdout
  models/holdout_extrap_fecrc.json                        Cr/T extrapolation
  models/holdout_extrap_c_fecrc.json                      C extrapolation (opt)

Output: models/results_fecrc_summary.json

Usage: py -3.12 -X utf8 analysis_revision/summarize_fecrc.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

from fe_surrogate.experiment import SEEDS
from fe_surrogate.systems import SYSTEMS

SYSTEM = "fecrc"
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "results_fecrc_summary.json")


def load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def group_mean(res, suffix=""):
    """Group keys 'tag_s{seed}' -> {tag: mean over seeds} of mean_mae."""
    out = {}
    for k, v in res.items():
        if suffix and not k.endswith(suffix):
            continue
        tag = k[: k.rfind("_s")]
        out.setdefault(tag, []).append(v["mean_mae"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def dataset_stats():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "dataset_fecrc.csv"))
    probe = load(os.path.join(ROOT, "data", "raw", "fecrc_probe.json"))
    cols = [c for c in df.columns if c.startswith("NP_")]
    NP = df[cols].values
    occ = (NP > 0).sum(axis=0)
    return {
        "rows": int(len(df)),
        "n_phases": len(cols),
        "phases": [c[3:] for c in cols],
        "max_abs_sum_np_minus_1": float(np.abs(NP.sum(axis=1) - 1).max()),
        "in_stainless_box_fraction": float(df["in_stainless_box"].mean()),
        "converged_fraction": float(df["converged"].mean())
        if "converged" in df.columns else None,
        "per_phase_nonzero": {c[3:]: {"n": int(o), "frac": float(o / len(df)),
                                      "max": float(df[c].max())}
                              for c, o in zip(cols, occ)},
        "probe_active_counts": probe["active_counts"] if probe else None,
        "T_range_K": [float(df["T"].min()), float(df["T"].max())],
        "Cr_range": [float(df["Cr"].min()), float(df["Cr"].max())],
        "C_range": [float(df["C"].min()), float(df["C"].max())],
    }


def holdout_summary(path):
    """Keys look like '<block>_<model>_s<seed>'; split block off the front."""
    d = load(path)
    if d is None:
        return None
    prefixes = ("mlp_", "rf_", "xgb_", "knn_")
    out = {}
    for k, v in d.items():
        tag = k[: k.rfind("_s")]
        pos = [tag.find(p) for p in prefixes if tag.find(p) > 0]
        block = tag[: min(pos) - 1] if pos else tag
        out.setdefault(block, []).append(v["mean_mae"])
    per_block = {b: float(np.mean(vals)) for b, vals in out.items()}
    per_block_all = {k: v["mean_mae"] for k, v in d.items()}
    return {"per_block_mean_mae": per_block,
            "per_block_model_mean_mae": per_block_all,
            "overall_mean_mae": float(np.mean(list(per_block.values()))),
            "n_entries": len(d)}


def main():
    summary = {"system": SYSTEM,
               "elements": SYSTEMS[SYSTEM]["comps"] + ["T"],
               "generated_from": []}

    summary["dataset"] = dataset_stats()

    b = load(os.path.join(MODELS, "results_baselines_fecrc.json"))
    if b:
        summary["baselines_mae_mean_over_seeds"] = group_mean(b)
        summary["generated_from"].append("results_baselines_fecrc.json")

    h = load(os.path.join(MODELS, "results_heads_fecrc.json"))
    if h:
        heads = group_mean(h)
        summary["heads_mae_mean_over_seeds"] = heads
        detail = {}
        for k, v in h.items():
            detail[k] = {"mean_mae": v["mean_mae"],
                         "consistency": v["consistency"],
                         "stainless_box_mae": v["stainless_box_mae"]}
        summary["heads_per_seed"] = detail
        summary["generated_from"].append("results_heads_fecrc.json")

    r = load(os.path.join(MODELS, "results_remedy_fecrc.json"))
    if r:
        agg = r.get("aggregated", {}).get(SYSTEM, {})
        summary["remedy"] = {
            "aggregated": agg,
            "runs": {k: {"mae": v["mae"], "macro_auprc": v["macro_auprc"],
                         "macro_f1_zero_pos_excluded":
                             v["macro_f1_zero_pos_excluded"],
                         "per_phase_auprc": v["per_phase_auprc"]}
                     for k, v in r.get("runs", {}).items()},
        }
        summary["generated_from"].append("results_remedy_fecrc.json")

    ps = load(os.path.join(HERE, "phase_set_validation_fecrc.json"))
    if ps:
        summary["phase_set_validation"] = ps.get(SYSTEM, ps)
        summary["generated_from"].append("phase_set_validation_fecrc.json")

    det = load(os.path.join(HERE, "detection_fecrc.json"))
    if det:
        summary["detection"] = {
            "families": {f: {k: v[k] for k in
                             ["macro_auprc", "macro_auprc_phase_means",
                              "macro_f1_with_zero_pos",
                              "macro_f1_zero_pos_excluded",
                              "carbide_macro_auprc", "per_phase_auprc",
                              "n_seeds", "zero_positive_phases"]}
                         for f, v in det["families"].items()},
            "key_question": det["key_question"],
            "phases": det.get("phases"),
        }
        summary["generated_from"].append("detection_fecrc.json")

    hold = {"band": holdout_summary(
                os.path.join(MODELS, f"block_holdout_{SYSTEM}.json")),
            "extrap": holdout_summary(
                os.path.join(MODELS, f"holdout_extrap_{SYSTEM}.json")),
            "extrap_c": holdout_summary(
                os.path.join(MODELS, f"holdout_extrap_c_{SYSTEM}.json"))}
    summary["holdout"] = {k: (v if v else "still running / not available")
                          for k, v in hold.items()}
    for k, v in hold.items():
        if v:
            summary["generated_from"].append(
                {"band": f"block_holdout_{SYSTEM}.json",
                 "extrap": f"holdout_extrap_{SYSTEM}.json",
                 "extrap_c": f"holdout_extrap_c_{SYSTEM}.json"}[k])

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUT}")
    print(f"sections: {list(summary)}")


if __name__ == "__main__":
    main()
