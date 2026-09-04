"""Aggregate results across seeds and produce the comparison tables.

For fecrni: merges the A1 run (results_heads.json) with the A2/A3 additions
(results_a23.json). For fecrmn: the B1 run. Prints per-system tables and the
cross-system head ranking.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fe_surrogate.config import MODELS_DIR
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.experiment import active_phases

VARIANT_ORDER = [
    ("mlp_sigmoid", "MLP sigmoid (unconstrained)"),
    ("mlp_renorm", "MLP sigmoid + test-time renormalise"),
    ("mlp_sig_norm", "MLP sigmoid / sum (train-time normalise)"),
    ("mlp_softmax", "MLP softmax (sum=1 by construction)"),
    ("mlp_power_norm", "MLP sigmoid^2 / sum (power norm)"),
    ("mlp_residue", "MLP residue (FCC = 1 - sum)"),
    ("mlp_penalty_1", "MLP sigmoid + penalty lambda=1 (raw)"),
    ("mlp_penalty_10", "MLP sigmoid + penalty lambda=10 (raw)"),
    ("xgb_raw", "XGBoost"),
    ("xgb_renorm", "XGBoost + renormalise"),
    ("rf_raw", "Random forest"),
    ("rf_renorm", "Random forest + renormalise"),
    ("knn_raw", "kNN (k=10)"),
    ("knn_renorm", "kNN + renormalise"),
    ("ridge_raw", "Ridge (linear floor)"),
    ("ridge_renorm", "Ridge + renormalise"),
]


def load_system_results(system):
    files = [f"results_heads_{system}.json", f"results_baselines_{system}.json"]
    if system == "fecrni":
        files.append("results_a23.json")
    merged = {}
    for fn in files:
        path = os.path.join(MODELS_DIR, fn)
        if os.path.exists(path):
            merged.update(json.load(open(path)))
    return merged


def gather(results, prefix):
    out = []
    for seed in [42, 123, 2024]:
        key = f"{prefix}_s{seed}"
        if key in results:
            out.append(results[key])
    return out


def table(results, system):
    _, phase_names = active_phases(system)
    print(f"== {SYSTEMS[system]['name']} ({system}) ==")
    print(f"{'variant':34s} {'mean_MAE':>8s} {'mean_F1':>7s} {'bnd_MAE':>8s} {'bulk_MAE':>8s} {'cons':>6s} {'box':>6s}")
    print("-" * 84)
    agg = {}
    for prefix, label in VARIANT_ORDER:
        rs = gather(results, prefix)
        if not rs:
            continue
        mm = np.array([r["mean_mae"] for r in rs])
        f1 = np.array([r.get("mean_f1", 0.0) for r in rs])
        bd = np.array([r.get("boundary_mae", 0.0) for r in rs])
        bu = np.array([r.get("bulk_mae", 0.0) for r in rs])
        co = np.array([r["consistency"] for r in rs])
        bx = np.array([r["stainless_box_mae"] for r in rs])
        agg[prefix] = mm
        print(f"{label:34s} {mm.mean():8.4f} {f1.mean():7.3f} {bd.mean():8.4f} {bu.mean():8.4f} {co.mean():6.4f} {bx.mean():6.4f}")
    return agg


def main():
    syss = sys.argv[1:] if len(sys.argv) > 1 else sorted(SYSTEMS)
    aggs = {}
    for system in syss:
        results = load_system_results(system)
        aggs[system] = table(results, system)
        print()

    if len(syss) >= 2:
        print("Cross-system head ranking (mean_MAE):")
        header = f"{'variant':40s}" + "".join(f"{SYSTEMS[s]['name'][:10]:>12s}" for s in syss)
        print(header)
        print("-" * (40 + 12 * len(syss)))
        for prefix, label in VARIANT_ORDER:
            if all(prefix in aggs[s] for s in syss):
                row = f"{label:40s}" + "".join(f"{aggs[s][prefix].mean():12.4f}" for s in syss)
                print(row)


if __name__ == "__main__":
    main()