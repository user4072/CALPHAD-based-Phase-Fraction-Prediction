"""Application analyses A3/A5/A6 from stored data (no new compute).

A3  U1-gate: do the in-scope misses carry higher ensemble spread (U1)
    than confirmed hits? Gate operating points (hits retained vs misses
    excluded).
A5  Query-threshold sensitivity: precision/recall on the 811-point
    validation pool over FCC_min x SIGMA_max grids (Ni<=0.12 scope and
    LIQUID<=1e-3 fixed), plus population shortlist sizes from the full grid.
A6  Miss autopsy: training-set density (standardized k-NN distances, the
    U2 convention) around each miss vs its nearest validated hit.

Inputs (all existing):
  paper/screen_data/screen_fecrni_T1000K_step0.001.npz
  paper/screen_data/validate_fecrni_T1000K_step0.001.npz
  data/raw/dataset_fecrni.csv
Output (NEW): paper/screen_data/screen_analysis.json

Usage: py -3.12 paper/screen_analyze.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

TAG = "fecrni_T1000K_step0.001"
SD = os.path.join(HERE, "screen_data")
OUT = os.path.join(SD, "screen_analysis.json")

FCC_GRID = (0.97, 0.99, 0.995)
SIG_GRID = (1e-4, 1e-3, 1e-2)
NI_MAX, LIQ_MAX = 0.12, 1e-3


def main():
    scr = np.load(os.path.join(SD, f"screen_{TAG}.npz"))
    val = np.load(os.path.join(SD, f"validate_{TAG}.npz"))
    X, ens, spread = scr["X"], scr["ensemble"], scr["spread"]
    names = list(scr["phases"])
    i_fcc, i_sig, i_liq = (names.index(p) for p in
                           ("FCC_A1", "SIGMA", "LIQUID"))
    hit_idx = scr["hit_idx"]
    order = val["order"]
    kinds = np.array([k for k in val["kinds"]])
    F, S, L, ok = val["FCC"], val["SIGMA"], val["LIQUID"], val["ok"]
    ni = X[order, 2]
    out = {"tag": TAG}

    # ---------------- A3: U1-gate ----------------
    sp_hits = spread[hit_idx]
    q = {p: float(np.percentile(sp_hits, p)) for p in (50, 90, 95, 99)}
    m_scope = (kinds == "bg") & (ni <= NI_MAX) & val["confirmed"] & ok
    sp_miss = spread[order[m_scope]]
    m_bg = (kinds == "bg") & (ni <= NI_MAX) & ok & ~val["confirmed"]
    sp_bg_no = spread[order[m_bg]]
    y_true = np.array([1] * int(m_scope.sum()) + [0] * int((kinds == "hit").sum()))
    y_score = np.concatenate([sp_miss, spread[order[kinds == "hit"]]])
    auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None
    gates = {}
    for g, thr in (("q90", q[90]), ("q95", q[95]), ("q99", q[99])):
        gates[g] = {
            "thr": thr,
            "miss_excluded": int((sp_miss > thr).sum()),
            "miss_total": int(m_scope.sum()),
            "hits_retained_frac": round(float((sp_hits <= thr).mean()), 4),
        }
    out["A3_u1_gate"] = {
        "hits_spread_p50_p90_p99_max": [
            q[50], q[90], q[95],
            float(np.percentile(sp_hits, 99)), float(sp_hits.max())],
        "miss_spreads": [round(float(v), 6) for v in sorted(sp_miss)],
        "bg_noscope_spread_median": round(float(np.median(sp_bg_no)), 6),
        "auroc_miss_vs_hits": auroc,
        "gates": gates,
    }
    print("[A3] hits spread p50/p90/p95/p99:",
          f"{q[50]:.2e}/{q[90]:.2e}/{q[95]:.2e}/"
          f"{float(np.percentile(sp_hits, 99)):.2e}")
    print("[A3] miss spreads:", [f"{v:.2e}" for v in sorted(sp_miss)])
    print("[A3] AUROC(miss vs hits):", round(auroc, 3) if auroc else None)
    for g, d in gates.items():
        print(f"[A3] gate {g}: excludes {d['miss_excluded']}/"
              f"{d['miss_total']} misses, retains "
              f"{100 * d['hits_retained_frac']:.1f}% of hits")

    # ---------------- A5: threshold sweep ----------------
    fcc_e, sig_e = ens[order, i_fcc], ens[order, i_sig]
    liq_e = ens[order, i_liq]
    ni_all = X[order, 2]
    sweep = {}
    for fmin in FCC_GRID:
        for smax in SIG_GRID:
            sq = ((fcc_e >= fmin) & (sig_e <= smax)
                  & (liq_e <= LIQ_MAX) & (ni_all <= NI_MAX))
            cq = ok & (F >= fmin) & (S <= smax) & (L <= LIQ_MAX) & (ni_all <= NI_MAX)
            tp = int((sq & cq).sum())
            sweep[f"F{fmin:g}_S{smax:g}"] = {
                "surr_pos": int(sq.sum()), "cal_pos": int(cq.sum()),
                "precision_pool": round(tp / int(sq.sum()), 4)
                if sq.sum() else None,
                "recall_pool": round(tp / int(cq.sum()), 4)
                if cq.sum() else None,
            }
    fcc_g, sig_g = ens[:, i_fcc], ens[:, i_sig]
    pop = {}
    for fmin in FCC_GRID:
        for smax in SIG_GRID:
            h = ((fcc_g >= fmin) & (sig_g <= smax)
                 & (ens[:, i_liq] <= LIQ_MAX) & (X[:, 2] <= NI_MAX))
            pop[f"F{fmin:g}_S{smax:g}"] = int(h.sum())
    out["A5_threshold_sweep"] = {"pool_811": sweep, "population_hits": pop,
                                 "note": "pool = 11 frontier + 400 hits + "
                                 "400 bg (boundary-enriched stress pool); "
                                 "population from full 501,501 grid"}
    print("[A5] pool precision/recall (FCC_min x SIGMA_max):")
    for k, d in sweep.items():
        print(f"   {k}: P={d['precision_pool']} R={d['recall_pool']} "
              f"(surr {d['surr_pos']}, cal {d['cal_pos']}, "
              f"pop {pop[k]:,})")

    # ---------------- A6: miss autopsy ----------------
    df = pd.read_csv(os.path.join(ROOT, "data", "raw",
                                  "dataset_fecrni.csv"))
    Xtr = df[["Fe", "Cr", "Ni", "T"]].values.astype(np.float64)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    tree = cKDTree(Xtr / std)
    Xs = X / std
    aut = []
    confirmed_hits = order[(kinds == "hit") & val["confirmed"] & ok]
    for j in np.where(m_scope)[0]:
        i = int(order[j])
        d1, _ = tree.query(Xs[i], k=1)
        d5, _ = tree.query(Xs[i], k=5)
        # nearest validated hit in standardized coords
        dh, _ = cKDTree(Xs[confirmed_hits]).query(Xs[i], k=1)
        aut.append({
            "comp": [round(float(X[i, c]), 4) for c in range(3)],
            "d1_train": round(float(np.atleast_1d(d1)[0]), 4),
            "d5_train_mean": round(float(np.mean(np.atleast_1d(d5))), 4),
            "d_nearest_hit": round(float(np.atleast_1d(dh)[0]), 4),
            "surr_FCC": round(float(ens[i, i_fcc]), 4),
            "cal_FCC": round(float(F[j]), 4),
        })
    # reference: median d1 over validated hits
    dh_all, _ = tree.query(Xs[confirmed_hits], k=1)
    out["A6_miss_autopsy"] = {
        "misses": aut,
        "hits_median_d1_train": round(float(np.median(dh_all)), 4),
        "note": "standardized (x,T)/std coords (U2 convention); "
                "d1 = nearest training row",
    }
    print("[A6] hits median d1:", round(float(np.median(dh_all)), 4))
    for a in aut:
        print(f"[A6] miss {a['comp']}: d1={a['d1_train']} "
              f"d5={a['d5_train_mean']} d_hit={a['d_nearest_hit']} "
              f"surr={a['surr_FCC']} cal={a['cal_FCC']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
