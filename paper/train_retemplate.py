"""Train + evaluate heads and RF on the re-templated Fe-Cr-Ni dataset.

Fixed protocol identical to the paper (cluster_split per seed, Huber,
early stopping 300 epochs, 4000-row test cap, clip-renorm projection for
the renorm/RF variants). Writes ONLY new files:
  models/results_retemplate.json
  models/pred_fecrni_rt_<tag>_s<seed>.npz

Usage: py -3.12 paper/train_retemplate.py [--heads renorm sig_norm softmax rf]
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
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from fe_surrogate.experiment import cluster_split, evaluate, renorm, SEEDS
from train_mlp import MLP4, train_mlp, DEVICE
from train_baselines import run_sklearn

SYSTEM = "fecrni_rt"
DATA = os.path.join(ROOT, "data", "raw", "dataset_fecrni_rt.csv")
PROBE = os.path.join(ROOT, "data", "raw", "fecrni_probe.json")
OUT = os.path.join(ROOT, "models", "results_retemplate.json")
EPOCHS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+",
                    default=["renorm", "sig_norm", "softmax", "rf"])
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    probe = json.load(open(PROBE))
    names = sorted(probe["active_counts"].keys())
    cols = [f"NP_{p}" for p in names]
    X = df[["Fe", "Cr", "Ni", "T"]].values.astype(np.float64)
    Y = df[cols].values.astype(np.float32)
    box = df["in_stainless_box"].values.astype(bool)
    print(f"[{SYSTEM}] {len(X)} rows, {len(names)} phases", flush=True)

    results = {"config": {"system": SYSTEM, "template_seed": 1007,
                          "seeds": SEEDS, "heads": args.heads},
               "runs": {}}
    if os.path.exists(OUT):
        results["runs"] = json.load(open(OUT)).get("runs", {})
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
        for head in args.heads:
            key = f"{head}_s{seed}"
            if key in results["runs"]:
                print(f"  skip {key} (exists)", flush=True)
                continue
            t0 = time.time()
            if head == "rf":
                pred = run_sklearn(
                    RandomForestRegressor(n_estimators=500, max_depth=20,
                                          n_jobs=-1),
                    X, Y, tr, va, te_idx, seed)
                pred = renorm(pred)
            else:
                model_head = "sigmoid" if head == "renorm" else head
                model = MLP4(n_phases=len(names), head=model_head,
                             hidden=(192, 192, 192),
                             n_in=X.shape[1]).to(DEVICE)
                model, _, scaler = train_mlp(model, X, Y, tr, va, seed,
                                             EPOCHS)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(
                        scaler.transform(X[te_idx]), dtype=torch.float32,
                        device=DEVICE)).cpu().numpy()
                if head == "renorm":
                    pred = renorm(pred)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dt = time.time() - t0
            res = evaluate(Y[te_idx], pred, box[te_idx])
            np.savez(os.path.join(ROOT, "models",
                                  f"pred_{SYSTEM}_{head}_s{seed}.npz"),
                     y_true=Y[te_idx], y_pred=pred, box=box[te_idx],
                     te_idx=te_idx)
            results["runs"][key] = {"mean_mae": res["mean_mae"],
                                    "time_s": round(dt, 1)}
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{key}] mae={res['mean_mae']:.4f} ({dt:.0f}s)",
                  flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
