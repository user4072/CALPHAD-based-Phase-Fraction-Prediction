"""Paired bootstrap comparison of the best constrained MLP vs the random
forest on the interpolation test set.

For each seed the two models are evaluated on the SAME test rows (the
cluster-stratified split is seed-dependent), so their per-row absolute
errors can be paired. We resample test rows with replacement (10,000
draws) and report the bootstrap distribution of the mean-MAE difference
(MLP - RF): a 95% CI that excludes zero supports a real per-system gap;
one that includes zero flags a statistical tie. No model is retrained.

Output: models/paired_bootstrap.json
Usage:  py -3.12 paired_bootstrap.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

SEEDS = [42, 123, 2024]
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
N_BOOT = 10_000


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


out = {"n_boot": N_BOOT}
for s in SYS:
    mlp = best_mlp(s)
    diffs = []
    n_test = None
    for seed in SEEDS:
        yt, yp_m = pred_npz(s, mlp, seed)
        _, yp_r = pred_npz(s, "rf_renorm", seed)
        assert yp_m.shape == yp_r.shape
        # per-row mean absolute error across phases, paired by row
        e_m = np.mean(np.abs(yp_m - yt), axis=1)
        e_r = np.mean(np.abs(yp_r - yt), axis=1)
        diffs.append(e_m - e_r)
        n_test = len(e_m)
    diffs = np.concatenate(diffs)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out[s] = {
        "best_mlp": mlp,
        "n_rows": int(len(diffs)),
        "mean_diff_mlp_minus_rf": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "p_mlp_worse": float((boot > 0).mean()),
    }
    verdict = "TIE" if lo <= 0 <= hi else ("MLP better" if hi < 0 else "RF better")
    print(f"{s}: diff {diffs.mean():+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")

with open(os.path.join(M, "paired_bootstrap.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved models/paired_bootstrap.json")
