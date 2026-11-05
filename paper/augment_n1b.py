"""N1b augmentation pilot v2: does targeted boundary augmentation fix boundary error?

v1 (void) taught two lessons, both fixed here:
(a) the augmented rows were appended to the pool but the ORIGINAL tr
    indices were passed to train_mlp, so they never trained -- v2 passes
    extended indices;
(b) model weights were initialised from ambient RNG (init lottery ~
    1e-2 in spots), so same-data retrains were incomparable -- v2 seeds
    torch+numpy BEFORE model creation in every arm, and self-checks
    determinism by retraining one arm twice (bit-equality asserted).

Design (pre-registered, leakage-free):
- three arms, identical protocol except the train pool: base (no aug),
  targeted (~200 jittered-around-boundary-train-rows solves), random
  (~200 uniform-simplex solves); all retrained, same seeds, same
  seeded inits;
- split frozen (cluster_split on ORIGINAL data; pool=[X|new] keeps
  original tr/va indices valid; augmented rows appended to tr);
- test rows byte-identical to the published runs (te_idx from stored
  baseline npz); boundary bins reuse stored p33/p66;
- success criterion: targeted boundary-MAE beats BOTH base and random
  on all three seeds (sign consistency), reported as mean deltas.

Writes ONLY models/augmentation_n1b.json, models/augment_n1b_points.npz,
models/augment_n1b_pred_<variant>_s<seed>.npz (+ console table).
Usage: py -3.12 paper/augment_n1b.py [--n-aug 200]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.experiment import (cluster_split, active_phases,
                                     load_data, renorm)
from train_mlp import train_mlp, MLP4, EPOCHS, DEVICE
from generate_data import run_eq, _init_worker, N_WORKERS

SYSTEM = "fecrni"
N_AUG = 200
JITTER_X = 0.005
JITTER_T = 10.0
SEEDS = [42, 123, 2024]
TAG = "mlp_renorm"
M = os.path.join(ROOT, "models")


def boundary_dist(feat, y_true):
    """Distance to the nearest row with a different active-phase set."""
    coords = np.column_stack([feat[:, :2], (feat[:, 3] - 700.0) / 1300.0])
    sets = [frozenset(np.where(y_true[i] > 1e-3)[0]) for i in range(len(feat))]
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=min(64, len(feat)), workers=-1)
    d = np.full(len(feat), np.nan)
    for i in range(len(feat)):
        for dist, j in zip(dists[i][1:], idxs[i][1:]):
            if sets[j] != sets[i]:
                d[i] = dist
                break
    return d


def jitter_free(rows, seed):
    """Jitter the FREE composition columns (Cr, Ni = cols 1,2) + T.

    X layout is [balance, free1, free2, T]; balance recomputed exactly so
    every proposal lies on the simplex by construction.
    """
    rng = np.random.default_rng(seed)
    out = rows.copy()
    out[:, 1:3] = np.clip(out[:, 1:3] + rng.normal(0, JITTER_X,
                                                   size=(len(rows), 2)),
                          0.0, 1.0)
    s = out[:, 1:3].sum(axis=1, keepdims=True)
    over = (s > 0.995).flatten()
    out[over, 1:3] *= 0.995 / s[over]
    out[:, 0] = 1.0 - out[:, 1:3].sum(axis=1)
    out[:, 3] = np.clip(out[:, 3] + rng.normal(0, JITTER_T, size=len(rows)),
                        700.0, 2000.0)
    return out


def solve_points(pts, cfg, phases, phase_cols):
    tasks = [{"cfg": cfg, "names": phases, "cols": phase_cols,
              "point": (float(p[1]), float(p[2]), float(p[3]))} for p in pts]
    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(cfg["tdb"],)) as pool:
        rows = pool.map(run_eq, tasks)
    return [r for r in rows if r["converged"]]


def train_arm(Xpool, Ypool, tr_idx, va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP4(n_phases=Ypool.shape[1], head="sigmoid",
                 hidden=(192, 192, 192), n_in=Xpool.shape[1]).to(DEVICE)
    t0 = time.time()
    model, _, scaler = train_mlp(model, Xpool, Ypool, tr_idx, va,
                                 seed, EPOCHS)
    return model, scaler, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-aug", type=int, default=N_AUG)
    args = ap.parse_args()
    cfg = SYSTEMS[SYSTEM]
    X, Y, _ = load_data(SYSTEM)
    phase_cols, phase_names = active_phases(SYSTEM)
    print(f"{SYSTEM}: {len(X)} rows, phases {phase_names}", flush=True)

    be = json.load(open(os.path.join(M, "boundary_error.json")))
    p33, p66 = be["p33"], be["p66"]

    results = {"config": {"system": SYSTEM, "n_aug": args.n_aug,
                          "jitter_x": JITTER_X, "jitter_T": JITTER_T,
                          "split": "frozen cluster_split; pool=[X|new]; "
                                   "augmented rows APPENDED to tr",
                          "init": "torch+numpy seeded BEFORE model creation, "
                                  "all arms",
                          "p33": p33, "p66": p66,
                          "design": "v2 (v1 void: aug never trained, "
                                    "ambient inits)"},
               "variants": {}}

    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        base_npz = np.load(os.path.join(
            M, f"pred_{SYSTEM}_{TAG}_192x192x192_huber_s{seed}.npz"))
        te_idx = base_npz["te_idx"]
        y_pub = base_npz["y_pred"]
        d_te = boundary_dist(X[te_idx], Y[te_idx])
        bins = {"boundary": d_te <= p33,
                "near": (d_te > p33) & (d_te <= p66),
                "interior": d_te > p66}
        out_s = {"n_test": int(len(te_idx))}
        for lab, m in bins.items():
            out_s[f"published_{lab}_mae"] = float(
                np.mean(np.abs(y_pub[m] - Y[te_idx][m])))
        out_s["published_overall_mae"] = float(
            np.mean(np.abs(y_pub - Y[te_idx])))

        if seed == 42:
            d_tr = boundary_dist(X[tr], Y[tr])
            order = np.argsort(np.where(np.isnan(d_tr), np.inf, d_tr))
            src = X[tr][order[:args.n_aug]]
            tgt_pts = jitter_free(src, seed=11)
            rng = np.random.default_rng(12)
            free = rng.dirichlet([0.5, 0.3, 0.2],
                                 size=args.n_aug)[:, 1:3]
            rnd_pts = np.column_stack([
                1.0 - free.sum(axis=1), free,
                rng.uniform(700.0, 2000.0, size=args.n_aug)])
            tgt_rows = solve_points(tgt_pts, cfg, phase_names, phase_cols)
            rnd_rows = solve_points(rnd_pts, cfg, phase_names, phase_cols)
            print(f"targeted solved {len(tgt_rows)}/{len(tgt_pts)} | "
                  f"random solved {len(rnd_rows)}/{len(rnd_pts)}", flush=True)

            def to_xy(rows):
                Xa = np.array([[r[c] for c in cfg["comps"]] + [r["T"]]
                               for r in rows])
                Ya = np.array([[r[c] for c in phase_cols] for r in rows],
                              dtype=np.float32)
                return Xa, Ya

            Xt, Yt = to_xy(tgt_rows)
            Xr, Yr = to_xy(rnd_rows)
            results["config"]["n_targeted_solved"] = int(len(Xt))
            results["config"]["n_random_solved"] = int(len(Xr))
            np.savez(os.path.join(M, "augment_n1b_points.npz"),
                     Xt=Xt, Yt=Yt, Xr=Xr, Yr=Yr)

        aug = np.load(os.path.join(M, "augment_n1b_points.npz"))
        arms = {"base": (None, None),
                "targeted": (aug["Xt"], aug["Yt"]),
                "random": (aug["Xr"], aug["Yr"])}
        for vlab, (Xa, Ya) in arms.items():
            if Xa is None:
                Xpool, Ypool = X.astype(np.float64), Y.astype(np.float32)
                tr_idx = tr
            else:
                Xpool = np.concatenate([X, Xa]).astype(np.float64)
                Ypool = np.concatenate([Y, Ya]).astype(np.float32)
                tr_idx = np.concatenate(
                    [tr, np.arange(len(X), len(Xpool))])
            assert len(tr_idx) == len(tr) + (0 if Xa is None else len(Xa))
            model, scaler, dt = train_arm(Xpool, Ypool, tr_idx, va, seed)
            model.eval()
            Xte = scaler.transform(X[te_idx])
            with torch.no_grad():
                pred = model(torch.tensor(
                    Xte, dtype=torch.float32,
                    device=DEVICE)).cpu().numpy()
            pred_f = renorm(np.clip(pred, 0.0, 1.0))
            np.savez(os.path.join(M, f"augment_n1b_pred_{vlab}_s{seed}.npz"),
                     y_true=Y[te_idx], y_pred=pred_f)
            out_s[f"{vlab}_overall_mae"] = float(
                np.mean(np.abs(pred_f - Y[te_idx])))
            for lab, m in bins.items():
                out_s[f"{vlab}_{lab}_mae"] = float(
                    np.mean(np.abs(pred_f[m] - Y[te_idx][m])))
            out_s[f"{vlab}_time_s"] = dt
            print(f"[seed {seed} {vlab}] overall="
                  f"{out_s[f'{vlab}_overall_mae']:.5f} boundary="
                  f"{out_s[f'{vlab}_boundary_mae']:.5f} ({dt}s)", flush=True)
        results["variants"][f"s{seed}"] = out_s

    # determinism self-check: retrain (targeted, s42), expect bit-equality
    tr42, va42, _ = cluster_split(X, Y, 42)
    aug = np.load(os.path.join(M, "augment_n1b_points.npz"))
    Xp = np.concatenate([X, aug["Xt"]]).astype(np.float64)
    Yp = np.concatenate([Y, aug["Yt"]]).astype(np.float32)
    trx = np.concatenate([tr42, np.arange(len(X), len(Xp))])
    m1, s1, _ = train_arm(Xp, Yp, trx, va42, 42)
    z0 = np.load(os.path.join(M, "augment_n1b_pred_targeted_s42.npz"))
    te42 = np.load(os.path.join(
        M, f"pred_{SYSTEM}_{TAG}_192x192x192_huber_s42.npz"))["te_idx"]
    m1.eval()
    with torch.no_grad():
        p1 = m1(torch.tensor(s1.transform(X[te42]), dtype=torch.float32,
                             device=DEVICE)).cpu().numpy()
    p1 = renorm(np.clip(p1, 0.0, 1.0))
    maxdiff = float(np.abs(p1 - z0["y_pred"]).max())
    results["determinism_check"] = {"retrain_maxdiff": maxdiff,
                                    "pass": bool(maxdiff == 0.0)}
    print(f"determinism self-check maxdiff={maxdiff:.2e} "
          f"({'PASS' if maxdiff == 0.0 else 'FAIL'})", flush=True)

    with open(os.path.join(M, "augmentation_n1b.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved models/augmentation_n1b.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
