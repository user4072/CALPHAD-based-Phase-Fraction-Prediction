"""Coverage-robustness: does the family ranking survive region
stratification (C) and boundary-matched comparison (D')?

Background (see ledger): the five ternaries share one seed-42 sampling
template, so raw design-space coverage is IDENTICAL across systems and
matching on it is vacuous. What differs is metallurgical coverage --
boundary density, prevalence -- induced by each system's phase fields.
These tests use only stored test predictions (no new solves):

C:  per system x region (box/nonbox/boundary/bulk) x {best-MLP, RF},
    mean over 3 seeds + per-seed agreement. Boundary = row entropy
    H(y) >= 0.5 ln K (the paper's own boundary definition).
D': subsample every system's test rows to a COMMON boundary-row
    fraction, recompute best-MLP vs RF. If 2-1-2 persists at matched
    boundary density, topology (not coverage) drives it.

Best head per system (Table 3): Ni renorm, Mn sig_norm, Mo sig_norm,
V renorm, MnNi renorm.

Output (NEW): models/coverage_robustness.json

Usage: py -3.12 analysis_revision/coverage_robust.py [--frac 0.3333]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

MODELS_DIR = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS_DIR, "coverage_robustness.json")
SEEDS = [42, 123, 2024]
SYS = ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"]
BEST = {"fecrni": "mlp_renorm_192x192x192_huber",
        "fecrmn": "mlp_sig_norm_192x192x192_huber",
        "fecrmo": "mlp_sig_norm_192x192x192_huber",
        "fecrv": "mlp_renorm_192x192x192_huber",
        "femnni": "mlp_renorm_192x192x192_huber"}
RF = "rf_renorm_std"
RNG = np.random.default_rng(12345)


def load(system, tag, seed):
    d = np.load(os.path.join(
        MODELS_DIR, f"pred_{system}_{tag}_s{seed}.npz"))
    return (np.asarray(d["y_true"], dtype=np.float64),
            np.asarray(d["y_pred"], dtype=np.float64),
            np.asarray(d["box"]).astype(bool))


def row_mae(yt, yp):
    return np.mean(np.abs(yt - yp), axis=1)


def entropy(yt):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(yt > 0, yt * np.log(np.where(yt > 0, yt, 1.0)), 0.0)
    return -np.sum(t, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=None,
                    help="target boundary-row fraction for D' (default: "
                    "min across systems, no upsampling)")
    ap.add_argument("--thr", type=float, default=0.25,
                    help="boundary = H(y) >= thr * ln K (paper uses 0.5)")
    args = ap.parse_args()
    THR = args.thr
    out = {"config": {"best_head": {s: BEST[s].split("_")[1]
                                    for s in SYS},
                       "thr_values": [0.5, THR],
                       "matched_frac": "min boundary share (no upsampling)",
                       "rng": 12345},
           "by_thr": {}}
    for thr in (0.5, THR) if THR != 0.5 else (0.5,):
        # pass 1: boundary shares -> common fraction (min, no upsampling)
        fracs = {}
        for system in SYS:
            for seed in SEEDS:
                yt, _, _ = load(system, BEST[system], seed)
                K = yt.shape[1]
                b = entropy(yt) >= thr * np.log(K)
                fracs.setdefault(system, []).append(float(b.mean()))
        f_common = min(np.mean(v) for v in fracs.values())
        out["by_thr"][str(thr)] = {
            "boundary_fracs": {s: [round(x, 4) for x in v]
                               for s, v in fracs.items()},
            "matched_frac": round(float(f_common), 4),
            "regions": {}, "matched": {}}
    for system in SYS:
        T = out["by_thr"]
        for thr in list(T):
            T[thr]["regions"][system] = {}
            T[thr]["matched"][system] = {}
        per_seed = {thr: {r: {"mlp": [], "rf": []}
                          for r in ("box", "nonbox", "boundary", "bulk")}
                    for thr in T}
        match_seed = {thr: {"mlp": [], "rf": []} for thr in T}
        for seed in SEEDS:
            yt, yp_m, box = load(system, BEST[system], seed)
            _, yp_r, _ = load(system, RF, seed)
            assert yp_m.shape == yp_r.shape == yt.shape
            K = yt.shape[1]
            mm, mr = row_mae(yt, yp_m), row_mae(yt, yp_r)
            for thr in T:
                bnd = entropy(yt) >= float(thr) * np.log(K)
                masks = {"box": box, "nonbox": ~box, "boundary": bnd,
                         "bulk": ~bnd}
                for r, m in masks.items():
                    if m.sum() == 0:
                        continue
                    per_seed[thr][r]["mlp"].append(float(mm[m].mean()))
                    per_seed[thr][r]["rf"].append(float(mr[m].mean()))
                f_common = T[thr]["matched_frac"]
                ib, iu = np.where(bnd)[0], np.where(~bnd)[0]
                n = len(yt)
                n_b = min(int(round(f_common * n)), len(ib))
                n_u = n - n_b
                if n_u > len(iu):
                    n_u, n_b = len(iu), n - len(iu)
                sel = np.concatenate(
                    [RNG.choice(ib, n_b, replace=False),
                     RNG.choice(iu, n_u, replace=False)])
                match_seed[thr]["mlp"].append(float(mm[sel].mean()))
                match_seed[thr]["rf"].append(float(mr[sel].mean()))
        for thr in T:
            for r in per_seed[thr]:
                T[thr]["regions"][system][r] = {
                    m: {"mean": float(np.mean(v)),
                        "per_seed": [round(x, 5) for x in v]}
                    for m, v in per_seed[thr][r].items() if v}
            T[thr]["matched"][system] = {
                m: {"mean": float(np.mean(v)),
                    "per_seed": [round(x, 5) for x in v]}
                for m, v in match_seed[thr].items()}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    # console summary per threshold: winner per region + matched
    for thr in out["by_thr"]:
        T = out["by_thr"][thr]
        print(f"--- thr={thr} matched_frac={T['matched_frac']} ---")
        for system in SYS:
            cells = []
            for r in ("box", "nonbox", "boundary", "bulk"):
                d = T["regions"][system].get(r)
                if not d:
                    cells.append(f"{r}:n/a")
                    continue
                pm, pr = d["mlp"]["per_seed"], d["rf"]["per_seed"]
                w = "MLP" if d["mlp"]["mean"] < d["rf"]["mean"] else "RF"
                ag = sum(a < b for a, b in zip(pm, pr))
                cells.append(f"{r}:{w}{ag}/3")
            m = T["matched"][system]
            w = "MLP" if m["mlp"]["mean"] < m["rf"]["mean"] else "RF"
            print(f"{system} bfrac={np.mean(T['boundary_fracs'][system]):.3f} "
                  + " ".join(cells)
                  + f" | matched:{w} "
                  f"({m['mlp']['mean']:.4f} vs {m['rf']['mean']:.4f})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
