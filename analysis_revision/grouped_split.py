"""Composition-grouped split (R2.3 control).

Reviewers note that the published cluster-stratified split is row-based:
the SAME alloy composition at different temperatures can fall into train
and test. This script rebuilds the split so that groups of near-duplicate
compositions are atomic:

  1. compositions are rounded to 1e-3 and grouped by exact match,
  2. groups whose composition distance <= radius (default 0.01, L2, in
     compositional coordinates) are merged (union-find),
  3. each group carries ALL its rows (every temperature),
  4. groups are stratified by k-means (k=6) on the group-mean phase
     profile, replicating the 6-cluster phase-mix stratification of
     cluster_split, and allocated to train/val/test at 64/16/20 of ROWS
     (shuffle within cluster, rng salted per cluster).

Models trained on this split (identical protocols to the published ones):
  mlp_renorm  - MLP4 sigmoid head + inference renormalisation (train_mlp)
  gated       - presence-gated head with prevalence-weighted Huber + BCE
                (train_remedy)
  xgb_renorm  - gradient-boosted trees (500, depth 8, lr 0.05)
  rf_renorm   - random forest (500, depth 20) + inference renormalisation

Metrics: evaluate() mean MAE / consistency / in-box MAE (same function and
conventions as the manuscript) and detection_metrics() macro-AUPRC with
the y>1e-3 protocol.

Usage:
  py -3.12 -X utf8 analysis_revision/grouped_split.py \
      [--systems fecrni femnni] [--radius 0.01] [--seeds 42 123 2024]

Outputs:
  models/grouped_split.json
  models/pred_<system>_grouped_<model>_s<seed>.npz
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from train_mlp import MLP4, train_mlp as train_core, DEVICE, EPOCHS
import train_remedy as trm
from fe_surrogate.experiment import load_data, evaluate, renorm, active_phases
from fe_surrogate.systems import SYSTEMS

MODELS_DIR = os.path.join(ROOT, "models")
SEEDS = [42, 123, 2024]


def build_groups(X, radius=0.01, decimals=3):
    """Row -> group id via exact-round composition match + radius union."""
    comp = X[:, :-1]
    keys = np.round(comp, decimals)
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    n = len(uk)
    tree = cKDTree(uk)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    n_pairs = len(pairs)
    if n_pairs:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = csr_matrix((np.ones(2 * n_pairs), (rows, cols)), shape=(n, n))
        _, grp = connected_components(graph, directed=False)
    else:
        grp = np.arange(n)
    _, group_id = np.unique(grp[inv], return_inverse=True)
    return group_id, n_pairs


def grouped_split(X, Y, seed=7, radius=0.01, k_clusters=6, fractions=(0.64, 0.16, 0.20)):
    g, n_pairs = build_groups(X, radius=radius)
    n_groups = int(g.max()) + 1
    prof = np.zeros((n_groups, Y.shape[1]))
    cnt = np.zeros(n_groups)
    np.add.at(prof, g, Y)
    np.add.at(cnt, g, 1)
    prof /= np.maximum(cnt[:, None], 1)
    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=0).fit(prof)
    lab = km.labels_
    rng = np.random.default_rng(seed)
    split_g = np.full(n_groups, -1)
    for c in range(k_clusters):
        gi = np.where(lab == c)[0]
        rows_c = cnt[gi].sum()
        gi_order = gi[cnt[gi].argsort()[::-1]]
        quota = np.array([0.64, 0.16, 0.20]) * rows_c
        assigned = np.array([0.0, 0.0, 0.0])
        for gid in gi_order:
            a = int(np.argmin(assigned - quota))
            split_g[gid] = a
            assigned[a] += cnt[gid]
    tr = np.where(split_g[g] == 0)[0]
    va = np.where(split_g[g] == 1)[0]
    te = np.where(split_g[g] == 2)[0]
    # minimum composition distance between test and train unique comps
    comp = X[:, :-1]
    u_tr = np.unique(np.round(comp[tr], 3), axis=0)
    u_te = np.unique(np.round(comp[te], 3), axis=0)
    dmin = float(cKDTree(u_tr).query(u_te, k=1)[0].min())
    info = {
        "n_groups": n_groups,
        "n_radius_pairs": int(n_pairs),
        "min_dist_te_to_tr_comps": round(dmin, 5),
        "row_counts": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "group_counts": {"train": int((split_g == 0).sum()),
                         "val": int((split_g == 1).sum()),
                         "test": int((split_g == 2).sum())},
    }
    print(f"  grouped split: groups={n_groups} rows {info['row_counts']} | "
          f"min dist te->tr comps={dmin:.4f}")
    return tr, va, te, info


def train_renorm_mlp(X, Y, tr, va, seed, epochs):
    model = MLP4(n_phases=Y.shape[1], head="sigmoid", n_in=X.shape[1]).to(DEVICE)
    model, _, scaler = train_core(model, X, Y, tr, va, seed, epochs)
    model.eval()
    return model, scaler


def pred_renorm_mlp(model, scaler, X, idx):
    with torch.no_grad():
        raw = model(torch.tensor(scaler.transform(X[idx]), dtype=torch.float32,
                                 device=DEVICE)).cpu().numpy()
    return renorm(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["fecrni", "femnni"])
    ap.add_argument("--radius", type=float, default=0.01)
    ap.add_argument("--radius-map", type=str, default="",
                    help="comma list sys=radius overriding --radius, e.g. fecrc=0.002")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", nargs="+",
                    default=["mlp_renorm", "gated", "xgb_renorm", "rf_renorm"])
    args = ap.parse_args()

    radius_map = {}
    for item in filter(None, args.radius_map.split(",")):
        k, v = item.split("=")
        radius_map[k.strip()] = float(v)

    out_path = os.path.join(MODELS_DIR, "grouped_split.json")
    all_out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_out = json.load(f)

    for system in args.systems:
        X, Y, df = load_data(system)
        from fe_surrogate.experiment import active_phases
        _, phase_names = active_phases(system)
        box = df["in_stainless_box"].values.astype(bool)
        radius = radius_map.get(system, args.radius)
        print(f"[{system}] {len(X)} rows, {len(phase_names)} phases, radius={radius}")
        tr, va, te, info = grouped_split(X, Y, radius=radius)
        res_sys = {"split": info, "models": {}}
        for seed in args.seeds:
            for name in args.models:
                t0 = time.time()
                if name == "mlp_renorm":
                    model, scaler = train_renorm_mlp(X, Y, tr, va, seed, args.epochs)
                    pred = pred_renorm_mlp(model, scaler, X, te)
                elif name == "gated":
                    gmodel = trm.GatedMLP(n_phases=Y.shape[1], n_in=X.shape[1]).to(trm.DEVICE)
                    gmodel, scaler, _, _, _, _ = trm.train_variant(
                        gmodel, "gated", X, Y, tr, va, seed, args.epochs)
                    pred = trm.predict(gmodel, "gated", scaler, X, te)
                else:
                    raw = np.zeros((len(te), Y.shape[1]))
                    if name == "xgb_renorm":
                        for j in range(Y.shape[1]):
                            m = XGBRegressor(n_estimators=500, max_depth=8,
                                             learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8,
                                             early_stopping_rounds=50,
                                             eval_metric="mae", random_state=seed)
                            m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])],
                                  verbose=False)
                            raw[:, j] = m.predict(X[te])
                    else:
                        for j in range(Y.shape[1]):
                            m = RandomForestRegressor(n_estimators=500, max_depth=20,
                                                      n_jobs=-1, random_state=seed)
                            m.fit(X[tr], Y[tr, j])
                            raw[:, j] = m.predict(X[te])
                    pred = renorm(raw)
                ev = evaluate(Y[te], pred, box[te], renormalise=False)
                det = trm.detection_metrics(Y[te], pred, phase_names)
                res_sys["models"][f"{name}_s{seed}"] = {
                    "mean_mae": ev["mean_mae"],
                    "consistency": ev["consistency"],
                    "stainless_box_mae": ev["stainless_box_mae"],
                    "per_phase_mae": ev["mae"],
                    "macro_auprc": det["macro_auprc"],
                    "per_phase_auprc": det["per_phase_auprc"],
                    "zero_positive_phases": det["zero_positive_phases"],
                    "time_s": round(time.time() - t0, 1),
                }
                np.savez(os.path.join(MODELS_DIR,
                                      f"pred_{system}_grouped_{name}_s{seed}.npz"),
                         y_true=Y[te], y_pred=pred, box=box[te], te_idx=te,
                         features=X[te], tr_idx=tr, va_idx=va)
                print(f"  [{name} s{seed}] MAE={ev['mean_mae']:.4f} "
                      f"AUPRC={det['macro_auprc']:.4f} | {res_sys['models'][f'{name}_s{seed}']['time_s']}s")
        all_out[system] = res_sys
        with open(out_path, "w") as f:
            json.dump(all_out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
