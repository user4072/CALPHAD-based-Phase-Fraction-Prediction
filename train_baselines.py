"""Conventional non-neural baselines for the Fe-Cr-Ni surrogate study.

  xgb    - gradient-boosted trees (500 trees, depth 8, lr 0.05, early stop)
  rf     - random forest (500 trees, depth 20)
  ridge  - linear floor (standardised features) - quantifies nonlinearity
  knn    - k-nearest neighbours (standardised features) - local smoothness

Each is evaluated raw and with inference-time renormalisation (sum-to-1
projection), the only closure strategy available to non-neural models.
Protocol: 3 paired seeds, same cluster-stratified splits as the MLP arms.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from fe_surrogate.config import MODELS_DIR
from fe_surrogate.experiment import load_data, cluster_split, evaluate, renorm, SEEDS
from fe_surrogate.systems import SYSTEMS


def run_sklearn(model, X, Y, tr, va, te_idx, seed):
    pred = np.zeros((len(te_idx), Y.shape[1]))
    for j in range(Y.shape[1]):
        m = model
        if hasattr(m, "random_state"):
            m.set_params(random_state=seed)
        m.fit(X[tr], Y[tr, j])
        pred[:, j] = m.predict(X[te_idx])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgb", "rf", "ridge", "knn"])
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="fecrni")
    ap.add_argument("--out", type=str, default=os.path.join(MODELS_DIR, "results_baselines.json"))
    args = ap.parse_args()

    X, Y, df = load_data(args.system)
    box = df["in_stainless_box"].values.astype(bool)
    print(f"System {args.system}: {len(X)} rows")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    results = {}
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
        for name in args.models:
            t0 = time.time()
            if name == "xgb":
                def make_xgb():
                    return XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        early_stopping_rounds=50, eval_metric="mae")
                pred = np.zeros((len(te_idx), Y.shape[1]))
                for j in range(Y.shape[1]):
                    m = make_xgb()
                    m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])], verbose=False)
                    pred[:, j] = m.predict(X[te_idx])
            elif name == "rf":
                pred = run_sklearn(RandomForestRegressor(n_estimators=500, max_depth=20,
                                                         n_jobs=-1), X, Y, tr, va, te_idx, seed)
            elif name == "ridge":
                pred = run_sklearn(Ridge(alpha=1.0), Xs, Y, tr, va, te_idx, seed)
            elif name == "knn":
                pred = run_sklearn(KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
                                   Xs, Y, tr, va, te_idx, seed)
            else:
                raise ValueError(name)
            for ren in [False, True]:
                tag = f"{name}_{'renorm' if ren else 'raw'}_s{seed}"
                res = evaluate(Y[te_idx], pred, box[te_idx], renormalise=ren)
                pred_final = renorm(pred) if ren else pred
                np.savez(os.path.join(MODELS_DIR, f"pred_{args.system}_{name}_{'renorm' if ren else 'raw'}_std_s{seed}.npz"),
                         y_true=Y[te_idx], y_pred=pred_final, box=box[te_idx],
                         te_idx=te_idx, features=X[te_idx])
                res["time_s"] = round(time.time() - t0, 1)
                results[tag] = res
                print(f"[{tag}] mean_MAE={res['mean_mae']:.4f} | consistency={res['consistency']:.4f} "
                      f"| box={res['stainless_box_mae']:.4f} | {res['time_s']}s")

    out = args.out
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()