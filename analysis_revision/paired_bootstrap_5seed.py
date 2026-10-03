"""Paired bootstrap on 5 seeds (42/123/2024 + 7/99) for the tied systems.

Same protocol as paired_bootstrap.py (per-row paired MLP-minus-RF MAE
differences pooled across seeds, 10,000 resamples, 95% CI), except:
  - seeds = [42, 123, 2024, 7, 99]; new seeds read from
    models/extra_<system>_<model>_s<seed>.npz
  - best-MLP head kept from the 3-seed ranking (comparability)
  - all five systems recomputed (ties + clear wins, for completeness)

Writes ONLY models/paired_bootstrap_5seed.json (published
paired_bootstrap.json untouched).

Usage: py -3.12 analysis_revision/paired_bootstrap_5seed.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

SEEDS3 = [42, 123, 2024]
SEEDS5 = [42, 123, 2024, 7, 99]
# extra seeds (7, 99) were trained only for the two tied systems
SYS = ["fecrmo", "fecrv"]
N_BOOT = 10_000
OUT = os.path.join(M, "paired_bootstrap_5seed.json")


def results(system):
    out = {}
    for fn in [f"results_heads_{system}.json",
               f"results_baselines_{system}.json"]:
        p = os.path.join(M, fn)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def best_mlp(system):
    d = results(system)
    return min(["mlp_renorm", "mlp_sig_norm", "mlp_softmax"],
               key=lambda k: np.mean([d[f"{k}_s{s}"]["mean_mae"]
                                      for s in SEEDS3]))


def pred_npz(system, tag, seed):
    if seed in (7, 99):
        # new-seed files use the extra_ prefix (no arch suffix)
        p = os.path.join(M, f"extra_{system}_{tag}_s{seed}.npz")
        z = np.load(p)
        return z["y_true"], z["y_pred"]
    for suffix in ("192x192x192_huber", "std"):
        p = os.path.join(M, f"pred_{system}_{tag}_{suffix}_s{seed}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["y_true"], z["y_pred"]
    raise FileNotFoundError(f"pred_{system}_{tag}_*_s{seed}.npz")


# extra_<system>_<model> tags mirror the stored file naming
# (extra seeds exist only for the two tied systems)

out = {"n_boot": N_BOOT, "seeds": SEEDS5}
for s in SYS:
    mlp = best_mlp(s)
    diffs = []
    for seed in SEEDS5:
        yt, yp_m = pred_npz(s, mlp, seed)
        _, yp_r = pred_npz(s, "rf_renorm", seed)
        assert yp_m.shape == yp_r.shape
        e_m = np.mean(np.abs(yp_m - yt), axis=1)
        e_r = np.mean(np.abs(yp_r - yt), axis=1)
        diffs.append(e_m - e_r)
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
    verdict = "TIE" if lo <= 0 <= hi else ("MLP better" if hi < 0
                                           else "RF better")
    print(f"{s}: diff {diffs.mean():+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]  "
          f"-> {verdict}", flush=True)

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT}")
