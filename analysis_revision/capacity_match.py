"""Capacity-matched control for the simplex-head comparison (R1.1/R2.2).

The published Table 3 softmax/sparsemax arms use a single Linear map to K
outputs, while every other head uses a per-phase block (Linear-192 + SiLU
+ Linear-1) plus a learnable per-phase positive scale. This script trains
softmax/sparsemax heads built from the SAME per-phase blocks (identical
parameter count to the sigmoid family), then softmax / sparsemax(4z)
normalisation. Everything else (trunk, Huber loss delta 0.01, AdamW +
cosine, batch 128, early-stop patience 40 on val MAE, 3 paired seeds,
cluster-stratified split, test subsample cap 4000 with rng 0) mirrors
train_mlp.py exactly.

Usage:
  py -3.12 -X utf8 analysis_revision/capacity_match.py \
      [--systems fecrni ...] [--heads softmax sparsemax] [--epochs 300]

Outputs:
  models/results_capacity_<system>.json (+ models/results_capacity_matched.json)
  models/pred_<system>_capmlp_<head>_s<seed>.npz
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

import train_mlp as tm
from train_mlp import MLP4, train_mlp as train_core, DEVICE, EPOCHS, sparsemax
from fe_surrogate.experiment import load_data, cluster_split, evaluate, active_phases
from fe_surrogate.systems import SYSTEMS

SEEDS = [42, 123, 2024]
MODELS_DIR = os.path.join(ROOT, "models")


class MLP4CapMatch(MLP4):
    """Per-phase blocks -> positive scale -> logits -> (sparse)max over K.

    Parameter count identical to the sigmoid/renorm head family: the same
    per-phase two-layer block plus learnable scale, followed by a simplex
    map instead of independent sigmoids.
    """

    def __init__(self, n_phases=4, head="softmax", hidden=(192, 192, 192), n_in=4):
        super().__init__(n_phases=n_phases, head="sigmoid", hidden=hidden, n_in=n_in)
        self.cap_head = head

    def forward(self, x):
        h = self.net(x)
        logits = []
        for i, hd in enumerate(self.heads_out):
            scale = torch.sigmoid(self.output_scales[i])
            logits.append(scale * hd(h))
        z = torch.cat(logits, dim=-1)
        if self.cap_head == "softmax":
            return torch.softmax(z, dim=-1)
        return sparsemax(4.0 * z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=[
        "fecrni", "fecrmn", "fecrmo", "fecrv", "femnni",
        "fecrnic", "fecrc", "crconi", "crnimn"])
    ap.add_argument("--heads", nargs="+", default=["softmax"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    for system in args.systems:
        X, Y, df = load_data(system)
        _, phase_names = active_phases(system)
        n_phases = len(phase_names)
        box = df["in_stainless_box"].values.astype(bool)
        results = {}
        n_single = sum(p.numel() for p in
                       MLP4(n_phases=n_phases, head="softmax", n_in=X.shape[1]).parameters())
        n_deep = sum(p.numel() for p in
                     MLP4CapMatch(n_phases=n_phases, head="softmax", n_in=X.shape[1]).parameters())
        print(f"[{system}] {len(X)} rows, {n_phases} phases | params softmax "
              f"single-map={n_single:,} deep={n_deep:,} (deep == sigmoid-family plumbing)")

        rng = np.random.default_rng(0)
        for head in args.heads:
            for seed in args.seeds:
                tr, va, te = cluster_split(X, Y, seed)
                te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
                t0 = time.time()
                model = MLP4CapMatch(n_phases=n_phases, head=head,
                                     n_in=X.shape[1]).to(DEVICE)
                model, _, scaler = train_core(model, X, Y, tr, va, seed, args.epochs)
                model.eval()
                Xte = scaler.transform(X[te_idx])
                with torch.no_grad():
                    pred = model(torch.tensor(Xte, dtype=torch.float32,
                                              device=DEVICE)).cpu().numpy()
                res = evaluate(Y[te_idx], pred, box[te_idx])
                tag = f"capmlp_{head}"
                np.savez(os.path.join(MODELS_DIR, f"pred_{system}_{tag}_s{seed}.npz"),
                         y_true=Y[te_idx], y_pred=pred, box=box[te_idx],
                         te_idx=te_idx, features=X[te_idx])
                res["time_s"] = round(time.time() - t0, 1)
                res["n_params_deep"] = int(n_deep)
                res["n_params_single_map"] = int(n_single)
                results[f"{tag}_s{seed}"] = res
                print(f"[{system} {tag} s{seed}] mean_MAE={res['mean_mae']:.4f} "
                      f"| consistency={res['consistency']:.4f} "
                      f"| box={res['stainless_box_mae']:.4f} | {res['time_s']}s")
        out = os.path.join(MODELS_DIR, f"results_capacity_{system}.json")
        if os.path.exists(out):
            with open(out) as f:
                prior = json.load(f)
            prior.update(results)
            results = prior
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {out}")

    combined = {}
    for system in args.systems:
        p = os.path.join(MODELS_DIR, f"results_capacity_{system}.json")
        if os.path.exists(p):
            with open(p) as f:
                combined[system] = json.load(f)
    with open(os.path.join(MODELS_DIR, "results_capacity_matched.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print("Saved models/results_capacity_matched.json")


if __name__ == "__main__":
    main()
