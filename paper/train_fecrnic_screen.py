"""Train Fe-Cr-Ni-C renorm heads for the application screen (A4).

Same fixed protocol as the paper (cluster_split per seed, Huber,
early stopping, 300 epochs, 4000-row test cap, clip-renorm projection at
evaluation) but writes ONLY to paper/screen_data/ckpts_fecrnic/ -- the
published models/results_heads_fecrnic.json is never touched.

Output per seed: ckpts_fecrnic/fecrnic_renorm_s<seed>.pt with the same
bundle layout as the anchor checkpoints (model_head/state_dict/scaler/
phase_names/val/test MAE/time).

Usage: py -3.12 paper/train_fecrnic_screen.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from fe_surrogate.experiment import (load_data, cluster_split, renorm,
                                     active_phases, SEEDS)
from train_mlp import MLP4, train_mlp, EPOCHS, DEVICE

SYSTEM = "fecrnic"
CKPT_DIR = os.path.join(HERE, "screen_data", "ckpts_fecrnic")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    X, Y, df = load_data(SYSTEM)
    cols, names = active_phases(SYSTEM)
    assert list(cols) == sorted(cols), "phase order must match probe sort"
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        path = os.path.join(CKPT_DIR, f"{SYSTEM}_renorm_s{seed}.pt")
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[{SYSTEM} s{seed}] cached: test={ck['test_mae']:.4f}")
            continue
        tr, va, te = cluster_split(X, Y, seed)
        te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
        model = MLP4(n_phases=len(names), head="sigmoid",
                     hidden=(192, 192, 192), n_in=X.shape[1]).to(DEVICE)
        t0 = time.time()
        model, val_mae, scaler = train_mlp(model, X, Y, tr, va, seed,
                                           EPOCHS)
        t_train = time.time() - t0
        model.eval()
        with torch.no_grad():
            p = renorm(model(torch.tensor(
                scaler.transform(X[te_idx]), dtype=torch.float32,
                device=DEVICE)).cpu().numpy())
        test_mae = float(np.mean(np.abs(p - Y[te_idx])))
        ck = {"system": SYSTEM, "head": "renorm", "model_head": "sigmoid",
              "seed": seed, "n_phases": len(names), "phase_names": names,
              "n_in": int(X.shape[1]), "hidden": [192, 192, 192],
              "state_dict": {k: v.detach().cpu()
                             for k, v in model.state_dict().items()},
              "scaler_mean": torch.tensor(scaler.mean_),
              "scaler_scale": torch.tensor(scaler.scale_),
              "val_mae": float(val_mae), "test_mae": test_mae,
              "time_s": round(t_train, 1)}
        torch.save(ck, path)
        print(f"[{SYSTEM} s{seed}] test={test_mae:.4f} val={val_mae:.4f} "
              f"({t_train:.0f}s) -> {path}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
