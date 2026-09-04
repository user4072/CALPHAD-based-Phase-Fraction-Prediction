"""Threshold sensitivity of the regressor--detector gap.

Phase presence is defined as N^phi > 1e-3 in the main study. This script
recomputes per-phase F1 and balanced accuracy for the best constrained MLP
and the random forest at three thresholds (1e-4, 1e-3, 1e-2) from the
stored prediction tensors, so no model is retrained.

Output: models/threshold_sensitivity.json
Usage:  py -3.12 threshold_sensitivity.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

from fe_surrogate.experiment import phase_classification  # noqa: E402

SEEDS = [42, 123, 2024]
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
THRESHOLDS = [1e-4, 1e-3, 1e-2]


def results(system):
    out = {}
    for fn in [f"results_heads_{system}.json", f"results_baselines_{system}.json"]:
        p = os.path.join(M, fn)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def best_mlp(system):
    d = results(system)
    return min(["mlp_renorm", "mlp_sig_norm", "mlp_softmax"],
               key=lambda k: np.mean([d[f"{k}_s{s}"]["mean_mae"] for s in SEEDS]))


def pred_npz(system, tag, seed):
    for suffix in ("192x192x192_huber", "std"):
        p = os.path.join(M, f"pred_{system}_{tag}_{suffix}_s{seed}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["y_true"], z["y_pred"]
    raise FileNotFoundError(f"pred_{system}_{tag}_*_s{seed}.npz")


out = {}
for s in SYS:
    mlp = best_mlp(s)
    out[s] = {"best_mlp": mlp}
    for model, tag in [("mlp", mlp), ("rf", "rf_renorm")]:
        out[s][model] = {}
        for thr in THRESHOLDS:
            f1s, bas = [], []
            for seed in SEEDS:
                yt, yp = pred_npz(s, tag, seed)
                f1, ba = phase_classification(yt, yp, threshold=thr)
                f1s.append(float(np.mean(f1)))
                bas.append(float(np.mean(ba)))
            out[s][model][f"{thr:g}"] = {
                "mean_f1": [float(np.mean(f1s)), float(np.std(f1s))],
                "balanced_acc": [float(np.mean(bas)), float(np.std(bas))],
            }
    print(f"{s}: best MLP = {mlp}")
    for model in ["mlp", "rf"]:
        row = " | ".join(
            f"thr={thr:g}: F1={out[s][model][f'{thr:g}']['mean_f1'][0]:.3f}"
            for thr in THRESHOLDS)
        print(f"  {model:4s} {row}")

with open(os.path.join(M, "threshold_sensitivity.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved models/threshold_sensitivity.json")
