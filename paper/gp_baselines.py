"""GP/RBF baseline pilot (REVISION_PLAN Major-7 "needs experiment").

Fits independent Gaussian-process regressors (RBF + WhiteKernel,
normalize_y) per phase on a documented N=2000-row train subset -- full
5681-row exact GPR is O(n^3) and out of scope for a baseline -- and
scores the identical frozen test rows as the published heads with the
same clip-renorm projection the tree baselines receive.

Writes ONLY models/gp_baselines.json (+ console table).
Usage: py -3.12 paper/gp_baselines.py [--n-fit 2000] [--system fecrni]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from fe_surrogate.experiment import (cluster_split, active_phases,
                                     load_data, renorm)

M = os.path.join(ROOT, "models")
SEEDS = [42, 123, 2024]
TAG = "mlp_renorm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--system", type=str, default="fecrni")
    args = ap.parse_args()
    X, Y, _ = load_data(args.system)
    phase_cols, phase_names = active_phases(args.system)
    print(f"{args.system}: {len(X)} rows, {len(phase_names)} phases",
          flush=True)
    out = {"config": {"system": args.system, "n_fit": args.n_fit,
                      "kernel": "RBF + WhiteKernel, normalize_y=True",
                      "protocol": "independent GP per phase; clip-renorm "
                                  "at eval (same as tree baselines); "
                      "test rows byte-identical to published runs"},
           "runs": {}}
    for seed in SEEDS:
        tr, _, _ = cluster_split(X, Y, seed)
        base_npz = np.load(os.path.join(
            M, f"pred_{args.system}_{TAG}_192x192x192_huber_s{seed}.npz"))
        te_idx = base_npz["te_idx"]
        rng = np.random.default_rng(seed)
        fit_idx = rng.choice(tr, min(args.n_fit, len(tr)), replace=False)
        t0 = time.time()
        scaler = StandardScaler().fit(X[fit_idx])
        Xf = scaler.transform(X[fit_idx])
        preds = []
        for j in range(Y.shape[1]):
            kern = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                          random_state=seed)
            gp.fit(Xf, Y[fit_idx][:, j])
            preds.append(gp.predict(scaler.transform(X[te_idx])))
        pred = renorm(np.clip(np.column_stack(preds), 0.0, 1.0))
        mae = float(np.mean(np.abs(pred - Y[te_idx])))
        per = [float(np.mean(np.abs(pred[:, j] - Y[te_idx][:, j])))
               for j in range(Y.shape[1])]
        cons = float(np.mean(np.abs(np.column_stack(preds).sum(axis=1)
                                    - 1.0)))
        out["runs"][f"s{seed}"] = {"mae": mae, "per_phase_mae": per,
                                   "raw_consistency": cons,
                                   "ref_renorm_mae": float(np.mean(np.abs(
                                       base_npz["y_pred"] - Y[te_idx]))),
                                   "time_s": round(time.time() - t0, 1)}
        print(f"[seed {seed}] GP mae={mae:.4f} "
              f"(ref renorm={out['runs'][f's{seed}']['ref_renorm_mae']:.4f}) "
              f"raw_cons={cons:.4f} ({out['runs'][f's{seed}']['time_s']}s)",
              flush=True)
    with open(os.path.join(M, "gp_baselines.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved models/gp_baselines.json")


if __name__ == "__main__":
    main()
