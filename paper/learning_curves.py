"""Case D: data-efficiency (learning curves) for the deployed head.

Retrains the Fe-Cr-Ni renorm head on fractions of the training split to
answer the practitioner's question: how many CALPHAD solves must be paid
before the surrogate works? Fixed protocol otherwise: same
cluster-stratified splits, same validation split (early stopping), same
4000-row test cap, same hyperparameters, 3 seeds.

Outputs (NEW files only):
  paper/screen_data/learning_fecrni_renorm.json

Usage:
    py -3.12 paper/learning_curves.py [--fractions 0.1 0.25 0.5 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from fe_surrogate.experiment import (load_data, cluster_split, evaluate,
                                     renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS
from train_mlp import MLP4, train_mlp, DEVICE

SYSTEM = "fecrni"
OUT = os.path.join(HERE, "screen_data", "learning_fecrni_renorm.json")
EPOCHS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--system", choices=sorted(SYSTEMS), default=SYSTEM)
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON (default screen_data/learning_"
                         "<system>_renorm.json)")
    args = ap.parse_args()
    system = args.system
    out = args.out or os.path.join(
        HERE, "screen_data", f"learning_{system}_renorm.json")

    X, Y, df = load_data(system)
    box = df["in_stainless_box"].values.astype(bool)
    n_phases = Y.shape[1]
    rng_cap = np.random.default_rng(0)
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))

    for frac in args.fractions:
        for seed in SEEDS:
            key = f"frac{frac:g}_s{seed}"
            if key in results and "mean_mae" in results[key]:
                print(f"[{key}] cached: "
                      f"mae={results[key]['mean_mae']:.4f}")
                continue
            tr, va, te = cluster_split(X, Y, seed)
            te_idx = te[rng_cap.choice(len(te), min(4000, len(te)),
                                       replace=False)]
            rng_sub = np.random.default_rng(1000 + seed)
            sub = rng_sub.choice(tr, size=max(1, int(round(frac * len(tr)))),
                                 replace=False)
            sub = np.sort(sub)
            model = MLP4(n_phases=n_phases, head="sigmoid",
                         hidden=(192, 192, 192), n_in=X.shape[1]).to(DEVICE)
            t0 = time.time()
            model, best_mae, scaler = train_mlp(
                model, X, Y, sub, va, seed, EPOCHS)
            t_train = time.time() - t0
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(
                    scaler.transform(X[te_idx]), dtype=torch.float32,
                    device=DEVICE)).cpu().numpy()
            res = evaluate(Y[te_idx], pred, box[te_idx], renormalise=True)
            results[key] = {
                "frac": frac, "seed": seed,
                "n_train": int(len(sub)), "n_train_full": int(len(tr)),
                "mean_mae": float(res["mean_mae"]),
                "val_mae": float(best_mae), "train_s": round(t_train, 1),
            }
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{key}] n={len(sub)} mae={res['mean_mae']:.4f} "
                  f"val={best_mae:.4f} train={t_train:.0f}s")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
