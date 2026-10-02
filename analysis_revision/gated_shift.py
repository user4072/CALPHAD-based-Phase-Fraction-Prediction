"""M2: gated presence-supervised head under the spatial protocols.

The paper's headline (presence supervision resolves sharp-boundary cases)
was never tested under distribution shift. This script closes that gap:
it retrains the exact GatedMLP recipe (train_remedy.train_variant,
variant="gated") on the exact holdout splits (holdout_eval.blocks_for /
extrap_blocks_for: train = cluster-train minus excluded rows, val =
cluster-val minus excluded rows, test = held rows), then scores test MAE
(after the clip-and-renorm projection, as in the paper) and macro-AUPRC
(detection_metrics, same protocol as the remedy).

Comparison needs no recomputation: per-block renorm-MLP and RF MAE come
from the stored models/block_holdout_<sys>.json and
models/holdout_extrap_<sys>.json.

Resumable: results accumulate in models/gated_shift.json keyed
"gated_<system>_<mode>_<block>_s<seed>"; completed keys are skipped.

Usage:
    py -3.12 analysis_revision/gated_shift.py [--systems fecrni fecrv]
        [--modes band extrap] [--epochs 300]

Output: models/gated_shift.json
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

from fe_surrogate.experiment import (load_data, cluster_split, renorm,
                                     active_phases, SEEDS)
from fe_surrogate.systems import SYSTEMS
from holdout_eval import (blocks_for, extrap_blocks_for, X2_TRAIN_HI,
                             X2_TEST_LO, T_TRAIN_HI, T_TEST_LO)
from train_mlp import EPOCHS
from train_remedy import (GatedMLP, train_variant, predict,
                          detection_metrics, DEVICE)

OUT = os.path.join(ROOT, "models", "gated_shift.json")


def block_mae_lookup(system, mode):
    """Stored {block: {seed: (mlp_mae, rf_mae)}} from the main benchmark."""
    fn = (f"block_holdout_{system}.json" if mode == "band"
          else f"holdout_extrap_{system}.json")
    d = json.load(open(os.path.join(ROOT, "models", fn)))
    out = {}
    for key, v in d.items():
        # keys look like "<block>_<model>_s<seed>"
        for model in ("mlp_renorm", "rf_renorm"):
            pre = f"_{model}_s"
            if pre in key:
                block, seed = key.split(pre)
                out.setdefault(block, {}).setdefault(
                    model, {})[int(seed)] = v["mean_mae"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["fecrni", "fecrv"])
    ap.add_argument("--modes", nargs="+", default=["band", "extrap"],
                    choices=["band", "extrap"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    results = {"config": {"arch": "GatedMLP 3x192 (train_remedy recipe)",
                          "epochs": args.epochs, "seeds": SEEDS,
                          "systems": args.systems, "modes": args.modes,
                          "projection": "clip-renorm at evaluation",
                          "splits": "holdout_eval blocks (train/val minus "
                          "excluded, test = held rows)"},
               "runs": {}}
    if os.path.exists(OUT):
        results["runs"] = json.load(open(OUT)).get("runs", {})
    print(f"[gated-shift] {len(results['runs'])} runs already stored | "
          f"device {DEVICE}", flush=True)

    t_all = time.time()
    for system in args.systems:
        X, Y, df = load_data(system)
        _, names = active_phases(system)
        lookup = {m: block_mae_lookup(system, m) for m in args.modes}
        for mode in args.modes:
            cfg = SYSTEMS[system]
            x_test_lo = None
            if mode == "extrap":
                # same adaptive fallback as run_system_extrap
                x2 = cfg["comps"][1]
                x_test_lo = X2_TEST_LO
                if not (df[x2] > X2_TEST_LO).any():
                    x_test_lo = 0.5 * (X2_TRAIN_HI + float(df[x2].max()))
            for seed in SEEDS:
                if mode == "band":
                    blk = {k: (v, v) for k, v in
                           blocks_for(df, cfg, seed).items()}
                else:
                    blk = extrap_blocks_for(df, cfg, seed,
                                            x_test_lo=x_test_lo)
                tr_all, va_all, _ = cluster_split(X, Y, seed)
                rng = np.random.default_rng(seed)
                for block, masks in blk.items():
                    held, excl = masks[:2]
                    key = f"gated_{system}_{mode}_{block}_s{seed}"
                    if key in results["runs"]:
                        continue
                    te_idx = np.where(held)[0]
                    tr = tr_all[~excl[tr_all]]
                    va = va_all[~excl[va_all]]
                    tr = tr[rng.permutation(len(tr))]
                    t0 = time.time()
                    model = GatedMLP(n_phases=len(names),
                                     n_in=X.shape[1]).to(DEVICE)
                    model, scaler, _, ep_run, _, _ = train_variant(
                        model, "gated", X, Y, tr, va, seed, args.epochs)
                    raw = predict(model, "gated", scaler, X, te_idx)
                    pred = renorm(raw)
                    mae = float(np.mean(np.abs(pred - Y[te_idx])))
                    det = detection_metrics(Y[te_idx], pred, names)
                    ref = lookup[mode].get(block, {})
                    mlp = ref.get("mlp_renorm", {}).get(seed)
                    rf = ref.get("rf_renorm", {}).get(seed)
                    results["runs"][key] = {
                        "system": system, "mode": mode, "block": block,
                        "seed": seed, "n_train": int(len(tr)),
                        "n_test": int(len(te_idx)), "mae": mae,
                        "macro_auprc": det["macro_auprc"],
                        "per_phase_auprc": det["per_phase_auprc"],
                        "ref_mlp_renorm_mae": mlp, "ref_rf_renorm_mae": rf,
                        "epochs_run": ep_run,
                        "time_s": round(time.time() - t0, 1),
                    }
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"  [{key}] n_tr={len(tr)} n_te={len(te_idx)} "
                          f"mae={mae:.4f} auprc={det['macro_auprc']} "
                          f"(ref mlp={mlp} rf={rf}) "
                          f"{time.time() - t0:.0f}s", flush=True)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    print(f"[gated-shift] done {len(results['runs'])} runs total "
          f"({(time.time() - t_all) / 60:.0f} min this session)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
