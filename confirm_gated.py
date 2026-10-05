"""Seed-matched confirmatory comparison for the presence-gated head.

The fresh-seed gated runs (1009/20260905/31337) must be compared against
classical detectors fitted on the same seeds, not against the development-seed
baselines that models/results_baselines_<sys>.json holds. This recomputes the
classical detector macro-AUPRC from the fresh-seed prediction npz files and
writes a single summary artifact.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

from train_remedy import detection_metrics  # noqa: E402

SEEDS = [1009, 20260905, 31337]
ORDER = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni",
         "fecrnic", "fecrc", "crconi", "crnimn"]
LBL = {"fecrni": "Fe-Cr-Ni", "fecrmn": "Fe-Cr-Mn", "fecrmo": "Fe-Cr-Mo",
       "fecrv": "Fe-Cr-V", "femnni": "Fe-Mn-Ni", "fecrnic": "Fe-Cr-Ni-C",
       "fecrc": "Fe-Cr-C", "crconi": "Cr-Co-Ni", "crnimn": "Cr-Ni-Mn"}
CLASSICAL = {"rf": "rf_renorm_std", "xgb": "xgb_renorm_std",
             "knn": "knn_renorm_std"}

FRESH_FILES = {"results_remedy_freshseeds.json": ORDER[:5]}
for _s in ORDER[5:]:
    FRESH_FILES[f"results_remedy_freshseeds_{_s}.json"] = [_s]


def names_for(system):
    p = os.path.join(ROOT, "data", "raw", f"{system}_probe.json")
    return sorted(json.load(open(p, encoding="utf-8"))["active_counts"].keys())


gated, gmae, classical = {}, {}, {}
for fn, syss in FRESH_FILES.items():
    d = json.load(open(os.path.join(M, fn), encoding="utf-8"))
    for k, v in d["runs"].items():
        if v.get("variant") == "gated":
            gated.setdefault(v["system"], []).append(
                (v["seed"], v["macro_auprc"], v["mae"]))

for system in ORDER:
    names = names_for(system)
    row = {}
    for m, tag in CLASSICAL.items():
        vals = []
        for s in SEEDS:
            p = os.path.join(M, f"pred_{system}_{tag}_s{s}.npz")
            if not os.path.exists(p):
                continue
            dd = np.load(p)
            dm = detection_metrics(dd["y_true"], dd["y_pred"], names)
            if dm["macro_auprc"] is not None:
                vals.append(dm["macro_auprc"])
        if vals:
            row[m] = float(np.mean(vals))
    classical[system] = row

print(f"{'system':<11}{'gated':>8}{'rf':>8}{'xgb':>8}{'knn':>8}"
      f"{'best classical':>16}{'gated wins?':>14}")
print("-" * 83)
wins = 0
summary = {}
for system in ORDER:
    gs = sorted(gated.get(system, []))
    ga = float(np.mean([r[1] for r in gs]))
    gm = float(np.mean([r[2] for r in gs]))
    row = classical[system]
    best = max(row, key=row.get)
    bv = row[best]
    win = ga >= bv
    wins += win
    summary[system] = {"label": LBL[system], "gated_auprc": ga,
                       "gated_mae": gm, "classical": row,
                       "best_classical": best, "best_classical_auprc": bv,
                       "gated_wins": bool(win)}
    print(f"{LBL[system]:<11}{ga:>8.3f}{row.get('rf', float('nan')):>8.3f}"
          f"{row.get('xgb', float('nan')):>8.3f}"
          f"{row.get('knn', float('nan')):>8.3f}{bv:>10.3f} ({best:<4})"
          f"{('yes' if win else 'NO'):>10}")
print("-" * 83)
allg = [summary[s]["gated_auprc"] for s in ORDER]
print(f"gated macro-AUPRC on fresh seeds: {min(allg):.3f}-{max(allg):.3f}")
print(f"gated beats best classical on {wins} of {len(ORDER)} systems")
out = {"seeds": SEEDS, "per_system": summary, "wins": int(wins),
       "n_systems": len(ORDER)}
with open(os.path.join(M, "gated_confirmatory.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote models/gated_confirmatory.json")