"""Verify the regenerated head results and emit report-ready tables.

Checks
  1. every stored metric recomputes exactly from the saved pred_*.npz
  2. boundary/bulk now follow the entropy rule in experiment.py
  3. main comparison, consistency, phase-level and per-seed tables

Usage: py -3.12 check_heads.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from fe_surrogate.experiment import evaluate, SEEDS
from fe_surrogate.systems import SYSTEMS

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
HEADS = ["mlp_sigmoid", "mlp_softmax", "mlp_residue", "mlp_renorm",
         "mlp_sig_norm", "mlp_sparsemax", "mlp_power_norm",
         "mlp_penalty_1", "mlp_penalty_10"]
BASE = ["xgb_raw", "xgb_renorm", "rf_raw", "rf_renorm",
        "knn_raw", "knn_renorm", "ridge_raw", "ridge_renorm"]


def load(system):
    out = {}
    for fn in [f"results_heads_{system}.json", f"results_baselines_{system}.json"]:
        p = os.path.join(M, fn)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    if system == "fecrni":
        p = os.path.join(M, "results_a23.json")
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def agg(d, prefix, field="mean_mae"):
    v = [d[f"{prefix}_s{s}"][field] for s in SEEDS if f"{prefix}_s{s}" in d]
    return (np.mean(v), np.std(v), len(v)) if v else (None, None, 0)


def nan_or(x, fmt="{:.4f}"):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


print("=" * 78)
print("1. RECOMPUTE CHECK: stored JSON vs metrics recomputed from pred_*.npz")
print("=" * 78)
bad = 0
checked = 0
for s in SYS:
    d = load(s)
    pat = re.compile(rf"^pred_{s}_(?P<tag>.+)_(?P<hid>\d+(?:x\d+)*)_"
                     rf"(?P<loss>huber|mse|mae)_s(?P<seed>\d+)\.npz$")
    for f in sorted(glob.glob(os.path.join(M, f"pred_{s}_mlp_*.npz"))):
        m = pat.match(os.path.basename(f))
        if not m or m.group("hid") != "192x192x192" or m.group("loss") != "huber":
            continue
        key = f"{m.group('tag')}_s{m.group('seed')}"
        if key not in d:
            continue
        z = np.load(f)
        r = evaluate(z["y_true"], z["y_pred"], z["box"].astype(bool))
        st = d[key]
        checked += 1
        for fld in ["mean_mae", "consistency", "mean_f1", "entropy_mae",
                    "stainless_box_mae", "boundary_mae", "bulk_mae"]:
            a, b2 = st[fld], r[fld]
            na = isinstance(a, float) and np.isnan(a)
            nb = isinstance(b2, float) and np.isnan(b2)
            if na and nb:
                continue
            if na != nb or abs(a - b2) > 1e-9:
                print(f"  MISMATCH {s} {key} {fld}: stored={a} recomputed={b2}")
                bad += 1
print(f"  {checked} runs checked, {bad} mismatches")

print()
print("=" * 78)
print("2. ENTROPY-CLASS ROW COUNTS (why boundary/bulk is NaN where it is)")
print("=" * 78)
for s in SYS:
    f = sorted(glob.glob(os.path.join(M, f"pred_{s}_mlp_renorm_*_huber_s42.npz")))
    if not f:
        continue
    z = np.load(f[0])
    y = z["y_true"]
    eps = 1e-12
    h = -np.sum(np.where(y > eps, y * np.log(np.clip(y, eps, None)), 0.0), axis=1)
    hm = np.log(y.shape[1])
    hi = h >= 0.5 * hm
    print(f"  {s:8s} K={y.shape[1]} n={len(y)} H_max={hm:.4f} "
          f"boundary={hi.sum():5d} ({100*hi.mean():5.1f}%) bulk={(~hi).sum():5d} "
          f"max_H={h.max():.4f}")

print()
print("=" * 78)
print("3. MAIN COMPARISON  mean MAE +- std over 3 seeds")
print("=" * 78)
hdr = f"{'model':22s}" + "".join(f"{s:>18s}" for s in SYS)
print(hdr)
print("-" * len(hdr))
for p in HEADS + BASE:
    cells = []
    for s in SYS:
        m, sd, n = agg(load(s), p)
        cells.append(f"{m:.4f}+-{sd:.4f}" if n else "--")
    if any(c != "--" for c in cells):
        print(f"{p:22s}" + "".join(f"{c:>18s}" for c in cells))

print()
print("=" * 78)
print("4. CONSISTENCY  mean |sum yhat - 1|, seed-mean")
print("=" * 78)
print(f"{'model':22s}" + "".join(f"{s:>13s}" for s in SYS))
for p in ["mlp_sigmoid", "mlp_softmax", "mlp_renorm", "mlp_sig_norm",
          "mlp_sparsemax", "mlp_residue", "rf_raw", "xgb_raw", "rf_renorm"]:
    cells = []
    for s in SYS:
        m, sd, n = agg(load(s), p, "consistency")
        cells.append(f"{m:.2e}" if n else "--")
    print(f"{p:22s}" + "".join(f"{c:>13s}" for c in cells))

print()
print("=" * 78)
print("5. PHASE-LEVEL, best MLP vs RF (seed-mean over 3 seeds)")
print("=" * 78)
print(f"{'system':8s} {'model':14s} {'MAE':>8s} {'F1':>7s} {'balacc':>7s} "
      f"{'entMAE':>8s} {'bnd':>8s} {'bulk':>8s} {'box':>8s}")
for s in SYS:
    d = load(s)
    best = min(((h, agg(d, h)[0]) for h in ["mlp_renorm", "mlp_sig_norm", "mlp_softmax"]
                if agg(d, h)[2]), key=lambda t: t[1])[0]
    for model in [best, "rf_renorm"]:
        row = [agg(d, model, f)[0] for f in
               ["mean_mae", "mean_f1", "entropy_mae", "boundary_mae",
                "bulk_mae", "stainless_box_mae"]]
        ba = np.mean([np.mean(d[f"{model}_s{k}"]["balanced_acc"])
                      for k in SEEDS if f"{model}_s{k}" in d])
        print(f"{s:8s} {model:14s} {nan_or(row[0]):>8s} {nan_or(row[1],'{:.3f}'):>7s} "
              f"{ba:7.3f} {nan_or(row[2]):>8s} {nan_or(row[3]):>8s} "
              f"{nan_or(row[4]):>8s} {nan_or(row[5]):>8s}")

print()
print("=" * 78)
print("6. PER-SEED MAE (appendix A)")
print("=" * 78)
print(f"{'system':8s} {'head':14s}" + "".join(f"{'s'+str(k):>10s}" for k in SEEDS))
for s in SYS:
    d = load(s)
    for h in ["mlp_renorm", "mlp_sig_norm", "mlp_softmax", "mlp_sparsemax",
              "mlp_residue", "mlp_sigmoid", "rf_renorm"]:
        v = [d[f"{h}_s{k}"]["mean_mae"] if f"{h}_s{k}" in d else None for k in SEEDS]
        if all(x is None for x in v):
            continue
        print(f"{s:8s} {h:14s}" + "".join(
            f"{x:10.5f}" if x is not None else f"{'--':>10s}" for x in v))

print()
print("=" * 78)
print("7. OLD (v2 backup) vs NEW mean MAE -- confirms the rerun is consistent")
print("=" * 78)
print(f"{'system':8s} {'head':14s} {'old':>9s} {'new':>9s} {'delta':>9s}")
for s in SYS:
    op = os.path.join(M, "_v2_backup", f"results_heads_{s}.json")
    if not os.path.exists(op):
        continue
    old, new = json.load(open(op)), load(s)
    for h in ["mlp_sigmoid", "mlp_softmax", "mlp_residue", "mlp_renorm",
              "mlp_sig_norm", "mlp_sparsemax"]:
        ov = [old[f"{h}_s{k}"]["mean_mae"] for k in SEEDS if f"{h}_s{k}" in old]
        nv = [new[f"{h}_s{k}"]["mean_mae"] for k in SEEDS if f"{h}_s{k}" in new]
        if ov and nv:
            print(f"{s:8s} {h:14s} {np.mean(ov):9.4f} {np.mean(nv):9.4f} "
                  f"{np.mean(nv)-np.mean(ov):+9.4f}")
