"""Block (spatial) holdout: contiguous-region generalisation and strict
out-of-range extrapolation.

The cluster-stratified split of the main study is an interpolation test --
every test point has training neighbours in (x, T). Two spatial protocols
build on it:

mode=band (contiguous-region holdout). Entire contiguous bands of the
design space are removed from training and used as the test set. Because
training data remain on BOTH sides of each band, the held-out rows lie
almost entirely inside the convex hull of the remaining training points
(99.3--99.4% by a Delaunay test), so this protocol measures spatial
distribution shift / missing-region generalisation, NOT strict
extrapolation.

Blocks (per system):
  T_band      T in [1200, 1400] K
  X2_band     x2 in [0.25, 0.35], where x2 is the second component
              (Cr for fecr*, Mn for femnni)
  random_ctrl a RANDOM subset of the same size as X2_band

mode=extrap (strict out-of-range extrapolation). Training keeps only the
near side of one axis and the test set is the far side, with the interior
gap in neither split, so every test point lies outside the training range
of that coordinate by construction:

  X2_extrap   train x2 <= 0.25, test x2 > 0.35
  T_extrap    train T <= 1200 K, test T > 1400 K

Each extrapolation block has its own size-matched random control
(<block>_ctrl): the same number of rows removed from training and an
equal-size random test set, so the spatial penalty can again be separated
from the data-volume penalty.

The random controls are the point of these scripts. Removing a region also
removes training data, so a bare region-versus-interpolation comparison
cannot separate "cannot generalise there" from "had less to learn from".
Each control removes an identical number of rows at random: any
degradation beyond the control is spatial. The band control is sized to
the larger of the two bands, so it is a conservative reference for the
smaller T band (which leaves more training data).

Models: MLP renorm, MLP sig_norm, RF + renorm, XGBoost + renorm, 3 seeds each.
Test set = every row in the block (no cap). Training protocol, preprocessing
and early stopping are identical to the main study.

Usage: py -3.12 holdout_eval.py                          # band mode, all five
       py -3.12 holdout_eval.py --system fecrni
       py -3.12 holdout_eval.py --mode extrap            # extrapolation mode
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from fe_surrogate.config import MODELS_DIR
from fe_surrogate.experiment import (load_data, active_phases, cluster_split,
                                     evaluate, renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS
from train_mlp import MLP4, train_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_LO, T_HI = 1200.0, 1400.0
X2_LO, X2_HI = 0.25, 0.35
# Strict extrapolation thresholds: train on the near side, test on the far
# side, leave the gap in neither.
X2_TRAIN_HI, X2_TEST_LO = 0.25, 0.35
T_TRAIN_HI, T_TEST_LO = 1200.0, 1400.0
MODELS = ["mlp_renorm", "mlp_sig_norm", "rf_renorm", "xgb_renorm"]


def blocks_for(df, cfg, seed):
    """Boolean held-out masks. The random control matches X2_band in size."""
    x2 = cfg["comps"][1]
    t_mask = (df["T"] >= T_LO) & (df["T"] <= T_HI)
    x_mask = (df[x2] >= X2_LO) & (df[x2] <= X2_HI)
    rng = np.random.default_rng(1000 + seed)
    r_mask = np.zeros(len(df), dtype=bool)
    r_mask[rng.choice(len(df), int(x_mask.sum()), replace=False)] = True
    return {"T_band": t_mask.values, "X2_band": x_mask.values,
            "random_ctrl": r_mask}


def extrap_blocks_for(df, cfg, seed):
    """Strict out-of-range extrapolation blocks.

    Returns {name: (held_mask, exclude_mask)} where exclude_mask is every
    row removed from training/validation (held rows plus, for the real
    blocks, the interior gap). Each block is paired with a size-matched
    random control: the same number of rows excluded from training and an
    equal-size random test set drawn from within the excluded set.
    """
    x2 = cfg["comps"][1]
    n = len(df)
    x_held = (df[x2] > X2_TEST_LO).values
    x_excl = (df[x2] > X2_TRAIN_HI).values  # held + gap (0.25, 0.35]
    t_held = (df["T"] > T_TEST_LO).values
    t_excl = (df["T"] > T_TRAIN_HI).values  # held + gap (1200, 1400]
    rng = np.random.default_rng(2000 + seed)
    blocks = {}
    for tag, held, excl in [("X2_extrap", x_held, x_excl),
                            ("T_extrap", t_held, t_excl)]:
        blocks[tag] = (held, excl)
        e_idx = rng.choice(n, int(excl.sum()), replace=False)
        e_mask = np.zeros(n, dtype=bool)
        e_mask[e_idx] = True
        h_mask = np.zeros(n, dtype=bool)
        h_mask[rng.choice(e_idx, int(held.sum()), replace=False)] = True
        blocks[f"{tag}_ctrl"] = (h_mask, e_mask)
    return blocks


def fit_tree(kind, X, Y, tr, va, te, seed):
    pred = np.zeros((len(te), Y.shape[1]))
    for j in range(Y.shape[1]):
        if kind == "rf":
            m = RandomForestRegressor(n_estimators=500, max_depth=20,
                                      random_state=seed, n_jobs=-1)
            m.fit(X[tr], Y[tr, j])
        else:
            m = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=50, eval_metric="mae",
                             random_state=seed)
            m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])], verbose=False)
        pred[:, j] = m.predict(X[te])
    return pred


def fit_and_eval(system, blocks_fn, out_name, header):
    """Train all models on each block of one system and dump the metrics.

    blocks_fn(seed) maps block name -> (held_mask, exclude_mask).
    """
    cfg = SYSTEMS[system]
    X, Y, df = load_data(system)
    _, names = active_phases(system)
    box = df["in_stainless_box"].values.astype(bool)
    print(f"\n{'=' * 70}\n{system}: {len(X)} rows, {len(names)} phases {names}\n"
          f"  {header} | device {DEVICE}\n{'=' * 70}")

    results = {}
    for seed_ in SEEDS:
        blk = blocks_fn(seed_)
        tr_all, va_all, _ = cluster_split(X, Y, seed_)
        for block, (held, excl) in blk.items():
            te_idx = np.where(held)[0]
            tr = tr_all[~excl[tr_all]]
            va = va_all[~excl[va_all]]
            rng = np.random.default_rng(seed_)
            tr = tr[rng.permutation(len(tr))]
            if seed_ == SEEDS[0]:
                print(f"  {block:14s} test {len(te_idx):5d} | train {len(tr):5d} "
                      f"| val {len(va):4d}")
            for model in MODELS:
                t0 = time.time()
                if model.startswith("mlp"):
                    head = model[len("mlp_"):]
                    net = MLP4(n_phases=len(names), head=head).to(DEVICE)
                    net, _, scaler = train_mlp(net, X, Y, tr, va, seed_)
                    net.eval()
                    with torch.no_grad():
                        pred = net(torch.tensor(scaler.transform(X[te_idx]),
                                                dtype=torch.float32,
                                                device=DEVICE)).cpu().numpy()
                    if head == "renorm":
                        pred = renorm(pred)
                else:
                    pred = renorm(fit_tree(model.split("_")[0], X, Y, tr, va,
                                           te_idx, seed_))
                res = evaluate(Y[te_idx], pred, box[te_idx])
                res.update(n_test=int(len(te_idx)), n_train=int(len(tr)),
                           time_s=round(time.time() - t0, 1))
                results[f"{block}_{model}_s{seed_}"] = res
                print(f"    [{block}_{model}_s{seed_}] MAE={res['mean_mae']:.4f} "
                      f"F1={res['mean_f1']:.3f} cons={res['consistency']:.2e} "
                      f"| {res['time_s']:.0f}s", flush=True)

    out = os.path.join(MODELS_DIR, out_name)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved {out}")


def run_system(system):
    cfg = SYSTEMS[system]
    df = load_data(system)[2]
    x2 = cfg["comps"][1]
    header = (f"bands: T in [{T_LO:.0f},{T_HI:.0f}] K | "
              f"{x2} in [{X2_LO},{X2_HI}]")
    fit_and_eval(system,
                 lambda seed: {k: (v, v) for k, v in
                               blocks_for(df, cfg, seed).items()},
                 f"block_holdout_{system}.json", header)


def run_system_extrap(system):
    cfg = SYSTEMS[system]
    x2 = cfg["comps"][1]
    df = load_data(system)[2]
    # Sanity: the far-side test rows must exceed the near-side training
    # range on the extrapolated coordinate, making the split strictly
    # out-of-range by construction.
    near_x = df.loc[df[x2] <= X2_TRAIN_HI, x2]
    far_x = df.loc[df[x2] > X2_TEST_LO, x2]
    near_t = df.loc[df["T"] <= T_TRAIN_HI, "T"]
    far_t = df.loc[df["T"] > T_TEST_LO, "T"]
    assert near_x.max() <= X2_TRAIN_HI and far_x.min() > X2_TEST_LO
    assert near_t.max() <= T_TRAIN_HI and far_t.min() > T_TEST_LO
    header = (f"strict extrapolation: train {x2} <= {X2_TRAIN_HI} -> "
              f"test {x2} > {X2_TEST_LO} | train T <= {T_TRAIN_HI:.0f} -> "
              f"test T > {T_TEST_LO:.0f}")
    fit_and_eval(system,
                 lambda seed: extrap_blocks_for(df, cfg, seed),
                 f"holdout_extrap_{system}.json", header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), action="append",
                    help="repeatable; default = all five")
    ap.add_argument("--mode", choices=["band", "extrap"], default="band",
                    help="band = contiguous-region holdout; "
                         "extrap = strict out-of-range extrapolation")
    args = ap.parse_args()
    systems = args.system or ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
    runner = run_system if args.mode == "band" else run_system_extrap
    for system in systems:
        t0 = time.time()
        runner(system)
        print(f"  {system} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
