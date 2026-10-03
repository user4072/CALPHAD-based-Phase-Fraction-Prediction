"""M6: tuning-budget sensitivity + extra seeds on the contended cells.

M6a (tuning): does the head ranking survive a modest tuning budget?
  Fe-Cr-Ni, 3 published seeds, fixed protocol except:
    MLP renorm + softmax x lr {1e-4, 1e-3} (default 3e-4 already stored)
    RF x {(trees 500, depth None), (trees 200, depth 20)}
    (default RF 500/depth-20 already stored)
M6b (seeds): do the Mo/V ties persist with 2 more seeds?
  fecrmo: sig_norm (its best head) + RF; fecrv: renorm + RF; seeds {7, 99}.
  Per-row errors are stored so the paired bootstrap can be re-run on 5.

Writes ONLY new files (published results JSONs are never touched):
  models/tuning_sensitivity.json
  models/extra_seeds.json
  models/extra_<system>_<modelkey>_s<seed>.npz (y_true, y_pred, te_idx)
Resumable: completed keys are skipped.

Usage: py -3.12 paper/tuning_extra.py [--m6a] [--m6b]  (default: both)
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
from sklearn.ensemble import RandomForestRegressor

import train_mlp
from train_mlp import MLP4, train_mlp, DEVICE
from train_baselines import run_sklearn
from fe_surrogate.experiment import (load_data, cluster_split, evaluate,
                                     renorm, SEEDS)

TUNE_JSON = os.path.join(ROOT, "models", "tuning_sensitivity.json")
EXTRA_JSON = os.path.join(ROOT, "models", "extra_seeds.json")
NEW_SEEDS = [7, 99]
LR_GRID = [1e-4, 1e-3]
RF_GRID = [(500, None), (200, 20)]  # (n_estimators, max_depth)


def test_idx(te):
    rng = np.random.default_rng(0)
    return te[rng.choice(len(te), min(4000, len(te)), replace=False)]


def set_lr(lr):
    # NOTE: `from train_mlp import train_mlp` rebinds the name to the
    # function, so the module global must be set via sys.modules.
    sys.modules["train_mlp"].LR = lr


def run_mlp(system, head, seed, lr, tag, epochs=300):
    X, Y, df = load_data(system)
    box = df["in_stainless_box"].values.astype(bool)
    tr, va, te = cluster_split(X, Y, seed)
    te_idx = test_idx(te)
    set_lr(lr)  # module-global lookup inside train_mlp()
    model = MLP4(n_phases=Y.shape[1], head=head,
                 hidden=(192, 192, 192), n_in=X.shape[1]).to(DEVICE)
    t0 = time.time()
    model, best_mae, scaler = train_mlp(model, X, Y, tr, va, seed, epochs)
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(scaler.transform(X[te_idx]),
                                  dtype=torch.float32,
                                  device=DEVICE)).cpu().numpy()
    if head in ("sigmoid",):
        pass  # renorm handled by caller flag
    return X, Y, te_idx, box, pred, dt


def save_npz(system, modelkey, seed, y_true, y_pred, te_idx):
    np.savez(os.path.join(ROOT, "models",
                          f"extra_{system}_{modelkey}_s{seed}.npz"),
             y_true=y_true, y_pred=y_pred, te_idx=te_idx)


def m6a():
    out = {"config": {"system": "fecrni", "seeds": SEEDS,
                      "lr_grid": LR_GRID, "rf_grid": RF_GRID,
                      "note": "fixed protocol except lr/RF grid"},
           "runs": {}}
    if os.path.exists(TUNE_JSON):
        out["runs"] = json.load(open(TUNE_JSON)).get("runs", {})
    X, Y, df = load_data("fecrni")
    box = df["in_stainless_box"].values.astype(bool)
    # --- MLP lr sweep (renorm=sigmoid+proj, softmax, sig_norm) ---
    for head, ren in [("sigmoid", True), ("softmax", False),
                      ("sig_norm", False)]:
        for lr in LR_GRID:
            for seed in SEEDS:
                key = f"mlp_{head}_lr{lr:g}_s{seed}"
                if key in out["runs"]:
                    print(f"  skip {key} (exists)", flush=True)
                    continue
                X, Y, te_idx, box, pred, dt = run_mlp(
                    "fecrni", head, seed, lr, key)
                if ren:
                    pred = renorm(pred)
                res = evaluate(Y[te_idx], pred, box[te_idx])
                out["runs"][key] = {
                    "head": head, "lr": lr, "seed": seed,
                    "mean_mae": res["mean_mae"], "time_s": round(dt, 1)}
                with open(TUNE_JSON, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"  [{key}] mae={res['mean_mae']:.4f} "
                      f"({dt:.0f}s)", flush=True)
    # --- RF grid ---
    for n_est, depth in RF_GRID:
        for seed in SEEDS:
            key = f"rf_t{n_est}_d{depth}_s{seed}"
            if key in out["runs"]:
                print(f"  skip {key} (exists)", flush=True)
                continue
            tr, va, te = cluster_split(X, Y, seed)
            te_idx = test_idx(te)
            t0 = time.time()
            pred = run_sklearn(
                RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                      n_jobs=-1),
                X, Y, tr, va, te_idx, seed)
            dt = time.time() - t0
            res = evaluate(Y[te_idx], pred, box[te_idx], renormalise=True)
            out["runs"][key] = {
                "n_estimators": n_est, "max_depth": depth, "seed": seed,
                "mean_mae": res["mean_mae"], "time_s": round(dt, 1)}
            with open(TUNE_JSON, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  [{key}] mae={res['mean_mae']:.4f} "
                  f"({dt:.0f}s)", flush=True)
    set_lr(3e-4)
    print(f"wrote {TUNE_JSON}")


def m6b():
    out = {"config": {"seeds": NEW_SEEDS, "note": "contended cells only"},
           "runs": {}}
    if os.path.exists(EXTRA_JSON):
        out["runs"] = json.load(open(EXTRA_JSON)).get("runs", {})
    jobs = [("fecrmo", "mlp_sig_norm", "sig_norm", True),
            ("fecrmo", "rf_renorm", None, True),
            ("fecrv", "mlp_renorm", "sigmoid", True),
            ("fecrv", "rf_renorm", None, True)]
    for system, modelkey, head, do_ren in jobs:
        X, Y, df = load_data(system)
        box = df["in_stainless_box"].values.astype(bool)
        for seed in NEW_SEEDS:
            key = f"{modelkey}_{system}_s{seed}"
            if key in out["runs"]:
                print(f"  skip {key} (exists)", flush=True)
                continue
            tr, va, te = cluster_split(X, Y, seed)
            te_idx = test_idx(te)
            t0 = time.time()
            if head is None:
                pred = run_sklearn(
                    RandomForestRegressor(n_estimators=500, max_depth=20,
                                          n_jobs=-1),
                    X, Y, tr, va, te_idx, seed)
            else:
                set_lr(3e-4)
                model = MLP4(n_phases=Y.shape[1], head=head,
                             hidden=(192, 192, 192),
                             n_in=X.shape[1]).to(DEVICE)
                model, _, scaler = train_mlp(model, X, Y, tr, va, seed,
                                             300)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(
                        scaler.transform(X[te_idx]), dtype=torch.float32,
                        device=DEVICE)).cpu().numpy()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dt = time.time() - t0
            if do_ren:
                pred = renorm(pred)
            res = evaluate(Y[te_idx], pred, box[te_idx])
            save_npz(system, modelkey, seed, Y[te_idx], pred, te_idx)
            out["runs"][key] = {
                "system": system, "model": modelkey, "seed": seed,
                "mean_mae": res["mean_mae"], "time_s": round(dt, 1)}
            with open(EXTRA_JSON, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  [{key}] mae={res['mean_mae']:.4f} "
                  f"({dt:.0f}s)", flush=True)
    set_lr(3e-4)
    print(f"wrote {EXTRA_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m6a", action="store_true")
    ap.add_argument("--m6b", action="store_true")
    args = ap.parse_args()
    do_a, do_b = args.m6a, args.m6b
    if not (do_a or do_b):
        do_a = do_b = True
    if do_a:
        print("[m6a] tuning sweep", flush=True)
        m6a()
    if do_b:
        print("[m6b] extra seeds", flush=True)
        m6b()


if __name__ == "__main__":
    main()
