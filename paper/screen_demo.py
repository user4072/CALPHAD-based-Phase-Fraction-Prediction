"""Application case A: high-throughput stainless-steel screening demo (Fe-Cr-Ni).

Query (fixed T = 1000 K, a service-relevant temperature below the ~1300 K
liquid appearance): fully austenitic, sigma-free solvus window --
    FCC_A1 >= 0.99, SIGMA <= 1e-3, LIQUID <= 1e-3
on the 3-seed ensemble mean of projected fraction vectors, using the
stored anchor renorm checkpoints (paper/anchor_data/anchor_ckpts/).

Stages:
  screen   : dense simplex grid at fixed T, batch inference, shortlist.
             --step sets the (x_Cr, x_Ni) grid spacing (pilot: 0.01).
  (validation with full-set CALPHAD lives in screen_validate.py)

Outputs (NEW files only):
  paper/screen_data/screen_<system>_T<int>T_step<step>.npz
  paper/screen_data/screen_<system>_T<int>T_step<step>.json  (summary)

Usage:
    py -3.12 paper/screen_demo.py --step 0.01   # pilot (~5k points)
    py -3.12 paper/screen_demo.py --step 0.001  # full (~500k points)
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

from fe_surrogate.experiment import SEEDS, renorm
from train_mlp import MLP4

SYSTEM = "fecrni"
HEAD = "renorm"
CKPT_DIR = os.path.join(HERE, "anchor_data", "anchor_ckpts")
OUT_DIR = os.path.join(HERE, "screen_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T_FIXED_K = 1000.0
Q_FCC_MIN, Q_SIGMA_MAX, Q_LIQ_MAX = 0.99, 1e-3, 1e-3
Q_NI_MAX = 0.12  # lean-Ni: minimise the expensive austenite stabiliser


def load_bundle(seed: int):
    path = os.path.join(CKPT_DIR, f"{SYSTEM}_{HEAD}_s{seed}.pt")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MLP4(n_phases=ck["n_phases"], head=ck["model_head"],
                 hidden=tuple(ck["hidden"]), n_in=ck["n_in"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["scaler_mean"].cpu().numpy(), \
        ck["scaler_scale"].cpu().numpy(), ck["phase_names"]


def simplex_grid(step: float):
    """Uniform (x_Cr, x_Ni) grid over the ternary simplex, Fe = balance."""
    n = int(round(1.0 / step))
    cr, ni = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            cr.append(i * step)
            ni.append(j * step)
    cr = np.array(cr)
    ni = np.array(ni)
    fe = 1.0 - cr - ni
    X = np.column_stack([fe, cr, ni,
                         np.full_like(fe, T_FIXED_K)]).astype(np.float64)
    return X


@torch.no_grad()
def predict(model, Xs: np.ndarray, batch: int = 65536) -> np.ndarray:
    outs = []
    for a in range(0, len(Xs), batch):
        xb = torch.tensor(Xs[a:a + batch], dtype=torch.float32,
                          device=DEVICE)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=65536)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{SYSTEM}_T{int(T_FIXED_K)}K_step{args.step:g}"
    X = simplex_grid(args.step)
    print(f"[screen] {SYSTEM} T={T_FIXED_K:.0f}K step={args.step:g} "
          f"points={len(X):,} device={DEVICE}")

    bundles = [load_bundle(s) for s in SEEDS]
    names = bundles[0][3]
    assert [b[3] for b in bundles][1:] == [names] * (len(bundles) - 1), \
        "phase order differs across seeds"
    i_fcc = names.index("FCC_A1")
    i_sig = names.index("SIGMA")
    i_liq = names.index("LIQUID")
    print(f"[screen] phases={names}")

    t0 = time.time()
    proj = []
    for (model, mu, sd, _), s in zip(bundles, SEEDS):
        Xs = (X - mu) / sd
        proj.append(renorm(predict(model, Xs, args.batch)))
    t_inf = time.time() - t0
    ens = np.mean(proj, axis=0)
    spread = np.std(proj, axis=0).mean(axis=1)  # mean over phases
    print(f"[screen] inference {len(X):,} pts x {len(SEEDS)} seeds "
          f"in {t_inf:.1f}s "
          f"({1e6 * t_inf / (len(X) * len(SEEDS)):.2f} us/point/seed)")

    fcc, sig, liq = ens[:, i_fcc], ens[:, i_sig], ens[:, i_liq]
    ni = X[:, 2]
    hit = ((fcc >= Q_FCC_MIN) & (sig <= Q_SIGMA_MAX) &
           (liq <= Q_LIQ_MAX) & (ni <= Q_NI_MAX))
    idx = np.where(hit)[0]
    print(f"[screen] query FCC>={Q_FCC_MIN} SIGMA<={Q_SIGMA_MAX} "
          f"LIQUID<={Q_LIQ_MAX} Ni<={Q_NI_MAX}: {hit.sum():,} hits "
          f"({100 * hit.mean():.3f}%)")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"screen_{tag}.npz"),
        X=X, ensemble=ens.astype(np.float32),
        spread=spread.astype(np.float32), hit_idx=idx,
        phases=np.array(names),
        q=np.array([Q_FCC_MIN, Q_SIGMA_MAX, Q_LIQ_MAX, Q_NI_MAX]))
    summary = {
        "system": SYSTEM, "head": HEAD, "seeds": SEEDS,
        "T_K": T_FIXED_K, "step": args.step, "n_points": int(len(X)),
        "query": {"FCC_min": Q_FCC_MIN, "SIGMA_max": Q_SIGMA_MAX,
                  "LIQUID_max": Q_LIQ_MAX, "Ni_max": Q_NI_MAX},
        "n_hits": int(hit.sum()),
        "inference_s": round(t_inf, 2),
        "us_per_point_per_seed": round(1e6 * t_inf / (len(X) * len(SEEDS)), 3),
        "device": DEVICE,
        "hit_FeCrNi_ranges": {
            "Fe": [round(float(X[idx, 0].min()), 4),
                   round(float(X[idx, 0].max()), 4)] if len(idx) else None,
            "Cr": [round(float(X[idx, 1].min()), 4),
                   round(float(X[idx, 1].max()), 4)] if len(idx) else None,
            "Ni": [round(float(X[idx, 2].min()), 4),
                   round(float(X[idx, 2].max()), 4)] if len(idx) else None,
        },
    }
    with open(os.path.join(OUT_DIR, f"screen_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[screen] wrote screen_{tag}.npz/.json")
    if len(idx):
        print("[screen] first hits (Fe, Cr, Ni, FCC, spread):")
        for k in idx[:10]:
            print(f"   {X[k, 0]:.3f} {X[k, 1]:.3f} {X[k, 2]:.3f} "
                  f"FCC={fcc[k]:.4f} U1={spread[k]:.2e}")


if __name__ == "__main__":
    main()
