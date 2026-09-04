"""Quantify the concentration of surrogate error near phase boundaries.

Figure 3 shows visually that errors sit on phase boundaries; this script
makes that quantitative. For the Fe-Cr-Ni test set (renorm head, seed 42)
each row's distance to the nearest row with a DIFFERENT active-phase set
(the boundary proxy) is computed with a KD-tree in scaled (x2, x3, T)
coordinates, and the per-row mean absolute error is binned by that
distance. No model is retrained.

Output: models/boundary_error.json
Usage:  py -3.12 boundary_error.py
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
M = os.path.join(ROOT, "models")

SYSTEM = "fecrni"
TAG = "mlp_renorm"
SEED = 42
K_NEIGH = 64  # neighbours searched for a differing phase set


def pred_npz(system, tag, seed):
    for suffix in ("192x192x192_huber", "std"):
        p = os.path.join(M, f"pred_{system}_{tag}_{suffix}_s{seed}.npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["features"], z["y_true"], z["y_pred"]
    raise FileNotFoundError(f"pred_{system}_{tag}_*_s{seed}.npz")


feat, yt, yp = pred_npz(SYSTEM, TAG, SEED)
n = len(feat)
# scale coordinates to comparable ranges: x in [0,1], T in [700,2000]
coords = np.column_stack([feat[:, :2], (feat[:, 3] - 700.0) / 1300.0])
active = frozenset
sets = [frozenset(np.where(yt[i] > 1e-3)[0]) for i in range(n)]

tree = cKDTree(coords)
d_boundary = np.full(n, np.nan)
# query enough neighbours that a differing set is almost always found
dists, idxs = tree.query(coords, k=min(K_NEIGH, n), workers=-1)
for i in range(n):
    for d, j in zip(dists[i][1:], idxs[i][1:]):
        if sets[j] != sets[i]:
            d_boundary[i] = d
            break
found = ~np.isnan(d_boundary)
print(f"{n} rows; boundary distance found for {found.sum()} "
      f"({100 * found.mean():.1f}%)")

err = np.mean(np.abs(yp - yt), axis=1)
q33, q66 = np.percentile(d_boundary[found], [33, 66])
bins = [("boundary (d <= p33)", d_boundary <= q33),
        ("near (p33 < d <= p66)", (d_boundary > q33) & (d_boundary <= q66)),
        ("interior (d > p66)", d_boundary > q66)]
out = {"system": SYSTEM, "tag": TAG, "seed": SEED, "n_rows": int(n),
       "p33": float(q33), "p66": float(q66), "bins": {}}
for lab, m in bins:
    out["bins"][lab] = {"n": int(m.sum()),
                        "mean_mae": float(err[m].mean()),
                        "median_mae": float(np.median(err[m]))}
    print(f"  {lab:24s} n={m.sum():5d}  mean MAE {err[m].mean():.5f}  "
          f"median {np.median(err[m]):.5f}")
ratio = out["bins"]["boundary (d <= p33)"]["mean_mae"] / \
    out["bins"]["interior (d > p66)"]["mean_mae"]
out["boundary_over_interior"] = float(ratio)
print(f"boundary/interior mean-MAE ratio: {ratio:.2f}")

with open(os.path.join(M, "boundary_error.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved models/boundary_error.json")
