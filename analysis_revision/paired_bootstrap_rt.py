"""Paired bootstrap best-MLP vs RF on the three re-templated systems.

Same protocol as paired_bootstrap.py (per-row paired MLP-minus-RF MAE
differences pooled across seeds 42/123/2024, 10,000 resamples, 95% CI),
adapted to the re-template file naming (pred_<sys>_rt_<head>_s<seed>.npz,
results_retemplate_<sys>.json). No model is retrained.

Writes ONLY models/paired_bootstrap_rt.json (published
paired_bootstrap.json untouched).

Usage: py -3.12 analysis_revision/paired_bootstrap_rt.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

SEEDS = [42, 123, 2024]
SYS = ["fecrmo_rt", "fecrmn_rt", "fecrv_rt"]
N_BOOT = 10_000
OUT = os.path.join(M, "paired_bootstrap_rt.json")


def results(system):
    base = system.replace("_rt", "")
    p = os.path.join(M, f"results_retemplate_{base}.json")
    return json.load(open(p))["runs"]


def best_mlp(system):
    d = results(system)
    return min(["renorm", "sig_norm", "softmax"],
               key=lambda k: np.mean([d[f"{k}_s{s}"]["mean_mae"]
                                      for s in SEEDS]))


def pred_npz(system, tag, seed):
    p = os.path.join(M, f"pred_{system}_{tag}_s{seed}.npz")
    z = np.load(p)
    return z["y_true"], z["y_pred"]


out = {"n_boot": N_BOOT}
for s in SYS:
    mlp = best_mlp(s)
    diffs = []
    for seed in SEEDS:
        yt, yp_m = pred_npz(s, mlp, seed)
        _, yp_r = pred_npz(s, "rf", seed)
        assert yp_m.shape == yp_r.shape, (s, seed)
        e_m = np.mean(np.abs(yp_m - yt), axis=1)
        e_r = np.mean(np.abs(yp_r - yt), axis=1)
        diffs.append(e_m - e_r)
    diffs = np.concatenate(diffs)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    out[s] = {"best_mlp": mlp, "mean_diff": float(np.mean(diffs)),
              "ci95": [lo, hi], "n_rows": int(len(diffs))}
    print(f"{s}: best={mlp} mean_diff={np.mean(diffs):+.6f} "
          f"ci95=[{lo:+.6f},{hi:+.6f}]")

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {OUT}")
