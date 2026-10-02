"""Application case A4: quaternary lean-Ni screen WITH carbon (Fe-Cr-Ni-C).

Query at T = 1000 K over the steel design box
(Cr 0.001--0.35, Ni 0.001--0.30, C 0.001--0.05, Fe balance):
high-austenite, carbide-controlled, sigma-free, lean-Ni --
    FCC_A1 >= 0.90, total carbides <= 0.05, SIGMA <= 1e-3,
    LIQUID <= 1e-3, Ni <= 0.12
on the 3-seed ensemble mean of projected fraction vectors, using
paper/screen_data/ckpts_fecrnic/ (trained by train_fecrnic_screen.py
under the fixed protocol).

Outputs (NEW files only):
  paper/screen_data/screen_fecrnic_T1000K_box.npz + .json

Usage:
    py -3.12 paper/screen_demo_q.py [--cr-step 0.004 --ni-step 0.004
                                     --c-step 0.001]
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

SYSTEM = "fecrnic"
HEAD = "renorm"
CKPT_DIR = os.path.join(HERE, "screen_data", "ckpts_fecrnic")
OUT_DIR = os.path.join(HERE, "screen_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T_FIXED_K = 1000.0
CR_LO, CR_HI = 0.001, 0.35
NI_LO, NI_HI = 0.001, 0.30
C_LO, C_HI = 0.001, 0.05
Q_FCC_MIN, Q_CARB_MAX = 0.90, 0.05
Q_SIGMA_MAX, Q_LIQ_MAX, Q_NI_MAX = 1e-3, 1e-3, 0.12
CARBIDES = ("M23C6", "M7C3", "CEMENTITE", "GRAPHITE")


def load_bundle(seed: int):
    path = os.path.join(CKPT_DIR, f"{SYSTEM}_{HEAD}_s{seed}.pt")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MLP4(n_phases=ck["n_phases"], head=ck["model_head"],
                 hidden=tuple(ck["hidden"]), n_in=ck["n_in"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["scaler_mean"].cpu().numpy(), \
        ck["scaler_scale"].cpu().numpy(), ck["phase_names"]


def box_grid(cr_step, ni_step, c_step):
    cr = np.arange(CR_LO, CR_HI + 1e-12, cr_step)
    ni = np.arange(NI_LO, NI_HI + 1e-12, ni_step)
    cc = np.arange(C_LO, C_HI + 1e-12, c_step)
    CR, NI, CC = np.meshgrid(cr, ni, cc, indexing="ij")
    cr, ni, cc = CR.ravel(), NI.ravel(), CC.ravel()
    fe = 1.0 - cr - ni - cc
    ok = fe > 0
    X = np.column_stack([fe[ok], cr[ok], ni[ok], cc[ok],
                         np.full(ok.sum(), T_FIXED_K)]).astype(np.float64)
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
    ap.add_argument("--cr-step", type=float, default=0.004)
    ap.add_argument("--ni-step", type=float, default=0.004)
    ap.add_argument("--c-step", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=65536)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{SYSTEM}_T{int(T_FIXED_K)}K_box"
    X = box_grid(args.cr_step, args.ni_step, args.c_step)
    print(f"[screen-q] {SYSTEM} T={T_FIXED_K:.0f}K box grid "
          f"points={len(X):,} device={DEVICE}")

    bundles = [load_bundle(s) for s in SEEDS]
    names = bundles[0][3]
    assert [b[3] for b in bundles][1:] == [names] * (len(bundles) - 1)
    i_fcc = names.index("FCC_A1")
    i_sig = names.index("SIGMA")
    i_liq = names.index("LIQUID")
    i_carb = [names.index(p) for p in CARBIDES]
    print(f"[screen-q] phases={names}")

    t0 = time.time()
    proj = []
    for (model, mu, sd, _), s in zip(bundles, SEEDS):
        proj.append(renorm(predict(model, (X - mu) / sd, args.batch)))
    t_inf = time.time() - t0
    ens = np.mean(proj, axis=0)
    spread = np.std(proj, axis=0).mean(axis=1)
    print(f"[screen-q] inference {len(X):,} pts x {len(SEEDS)} seeds "
          f"in {t_inf:.1f}s "
          f"({1e6 * t_inf / (len(X) * len(SEEDS)):.2f} us/point/seed)")

    fcc = ens[:, i_fcc]
    carb = ens[:, i_carb].sum(axis=1)
    sig, liq, ni = ens[:, i_sig], ens[:, i_liq], X[:, 2]
    hit = ((fcc >= Q_FCC_MIN) & (carb <= Q_CARB_MAX)
           & (sig <= Q_SIGMA_MAX) & (liq <= Q_LIQ_MAX) & (ni <= Q_NI_MAX))
    idx = np.where(hit)[0]
    print(f"[screen-q] query FCC>={Q_FCC_MIN} CARB<={Q_CARB_MAX} "
          f"SIGMA<={Q_SIGMA_MAX} LIQUID<={Q_LIQ_MAX} Ni<={Q_NI_MAX}: "
          f"{hit.sum():,} hits ({100 * hit.mean():.2f}%)")

    np.savez_compressed(
        os.path.join(OUT_DIR, f"screen_{tag}.npz"),
        X=X, ensemble=ens.astype(np.float32),
        spread=spread.astype(np.float32), hit_idx=idx,
        phases=np.array(names),
        q=np.array([Q_FCC_MIN, Q_CARB_MAX, Q_SIGMA_MAX, Q_LIQ_MAX,
                    Q_NI_MAX]))
    with open(os.path.join(OUT_DIR, f"screen_{tag}.json"), "w") as f:
        json.dump({
            "system": SYSTEM, "head": HEAD, "seeds": SEEDS,
            "T_K": T_FIXED_K,
            "grid": {"cr_step": args.cr_step, "ni_step": args.ni_step,
                     "c_step": args.c_step},
            "n_points": int(len(X)),
            "query": {"FCC_min": Q_FCC_MIN, "CARB_max": Q_CARB_MAX,
                      "SIGMA_max": Q_SIGMA_MAX, "LIQUID_max": Q_LIQ_MAX,
                      "Ni_max": Q_NI_MAX},
            "n_hits": int(hit.sum()),
            "inference_s": round(t_inf, 2),
            "us_per_point_per_seed": round(
                1e6 * t_inf / (len(X) * len(SEEDS)), 3),
            "device": DEVICE,
        }, f, indent=2)
    print(f"[screen-q] wrote screen_{tag}.npz/.json")
    if len(idx):
        print("[screen-q] leanest hits (Fe, Cr, Ni, C, FCC, carb):")
        o = np.argsort(X[idx, 2])[:10]
        for k in idx[o]:
            print(f"   {X[k, 0]:.3f} {X[k, 1]:.3f} {X[k, 2]:.4f} "
                  f"{X[k, 3]:.4f} FCC={fcc[k]:.4f} carb={carb[k]:.4f}")


if __name__ == "__main__":
    main()
