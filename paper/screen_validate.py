"""Application case A (validation): full-set CALPHAD check of the screen.

Validates, with ALL eligible phases of the pruned Fe-Cr-Ni TDB (28 phases,
same set as the anchor reference):
  1. every Pareto-frontier shortlist point (minimal Ni per Cr bin),
  2. a uniform random subset of the remaining hits (precision),
  3. a uniform random background sample of non-hits (miss rate / recall).

Outputs (NEW files only):
  paper/screen_data/validate_<tag>.npz + .json

Usage:
    py -3.12 paper/screen_validate.py --tag fecrni_T1000K_step0.001 \\
        --n-hits 400 --n-bg 400 --workers 8
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
sys.path.insert(0, HERE)

import numpy as np

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from anchor_eval import _cal_run, _cal_init

SYSTEM = "fecrni"
Q_FCC_MIN, Q_SIGMA_MAX, Q_LIQ_MAX = 0.99, 1e-3, 1e-3
N_CR_BINS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--n-hits", type=int, default=400)
    ap.add_argument("--n-bg", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe-timing-n", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "screen_data"), exist_ok=True)
    d = np.load(os.path.join(HERE, "screen_data", f"screen_{args.tag}.npz"))
    X, ens, hit_idx = d["X"], d["ensemble"], d["hit_idx"]
    names = list(d["phases"])
    i_fcc, i_sig, i_liq = (names.index(p) for p in
                           ("FCC_A1", "SIGMA", "LIQUID"))
    T_K = float(X[0, 3])
    rng = np.random.default_rng(7)

    # --- Pareto frontier: minimal Ni per Cr bin among hits ---
    cr = X[hit_idx, 1]
    bins = np.floor(cr * N_CR_BINS).astype(int)
    frontier = []
    for b in sorted(set(bins.tolist())):
        sel = hit_idx[bins == b]
        frontier.append(sel[np.argmin(X[sel, 2])])
    frontier = np.array(sorted(set(int(f) for f in frontier)))
    rest = np.setdiff1d(hit_idx, frontier)
    sub_hits = rng.choice(rest, size=min(args.n_hits, len(rest)),
                          replace=False) if len(rest) else np.array([], int)
    nonhit = np.setdiff1d(np.arange(len(X)), hit_idx)
    sub_bg = rng.choice(nonhit, size=min(args.n_bg, len(nonhit)),
                        replace=False)
    print(f"[validate] hits={len(hit_idx):,} frontier={len(frontier)} "
          f"sub-hits={len(sub_hits)} background={len(sub_bg)}")

    phases = sorted(eligible_phases(SYSTEMS[SYSTEM]["tdb"],
                                    SYSTEMS[SYSTEM]["comps"]))
    print(f"[validate] full set: {len(phases)} eligible phases, "
          f"{args.workers} workers")

    # CalphadRunner's audited single-point solver (_cal_run), issued over
    # many compositions at one T through a worker pool.
    from anchor_eval import _cal_init
    import multiprocessing as mp
    cfg = SYSTEMS[SYSTEM]
    pool = mp.Pool(args.workers, initializer=_cal_init,
                   initargs=(cfg["tdb"], cfg["elements"],
                             cfg["comps_species"], phases))

    def run_points(sel):
        tasks = [((float(X[i, 1]), float(X[i, 2])), T_K) for i in sel]
        t0 = time.time()
        out = pool.map(_cal_run, tasks, chunksize=4)
        return out, time.time() - t0

    order = np.concatenate([frontier, sub_hits, sub_bg])
    kinds = (["frontier"] * len(frontier) + ["hit"] * len(sub_hits)
             + ["bg"] * len(sub_bg))
    res, t_cal = run_points(order)
    print(f"[validate] {len(order)} full-set solves in {t_cal:.0f}s "
          f"({t_cal / len(order):.2f} s/point wall, "
          f"{t_cal * args.workers / len(order):.2f} s/point CPU)")

    def frac(r, phase):
        return r["amounts"].get(phase, 0.0) if r["ok"] else np.nan

    F = np.array([frac(r, "FCC_A1") for r in res])
    S = np.array([frac(r, "SIGMA") for r in res])
    L = np.array([frac(r, "LIQUID") for r in res])
    ok = np.array([r["ok"] for r in res])
    confirmed = ok & (F >= Q_FCC_MIN) & (S <= Q_SIGMA_MAX) & (L <= Q_LIQ_MAX)

    out = {}
    for k in ("frontier", "hit", "bg"):
        m = np.array(kinds) == k
        out[k] = {
            "n": int(m.sum()),
            "n_ok": int((ok[m]).sum()),
            "n_confirmed": int((confirmed[m]).sum()),
            "precision": (round(float(confirmed[m].sum() / ok[m].sum()), 4)
                          if ok[m].sum() else None),
        }
    out["calphad_wall_s"] = round(t_cal, 1)
    out["calphad_s_per_point_wall"] = round(t_cal / len(order), 3)
    out["calphad_s_per_point_cpu"] = round(t_cal * args.workers / len(order), 3)

    # --- probe-set timing (honest as-run denominator for break-even) ---
    from fe_surrogate.experiment import active_phases as probe_phases
    cols, _ = probe_phases(SYSTEM)
    probe_set = sorted(c[len("NP_"):] for c in cols)
    pool2 = mp.Pool(args.workers, initializer=_cal_init,
                    initargs=(cfg["tdb"], cfg["elements"],
                              cfg["comps_species"], probe_set))
    sample = rng.choice(np.arange(len(X)),
                        size=min(args.probe_timing_n, len(X)), replace=False)
    tasks = [((float(X[i, 1]), float(X[i, 2])), T_K) for i in sample]
    t0 = time.time()
    out2 = pool2.map(_cal_run, tasks, chunksize=4)
    t_probe = time.time() - t0
    out["probe_set"] = {
        "phases": probe_set, "n": len(sample),
        "s_per_point_wall": round(t_probe / len(sample), 3),
        "s_per_point_cpu": round(t_probe * args.workers / len(sample), 3),
        "n_ok": int(sum(r["ok"] for r in out2)),
    }
    print(f"[validate] probe-set ({len(probe_set)} phases): "
          f"{t_probe / len(sample):.2f} s/point wall")

    pool.close()
    pool2.close()

    np.savez_compressed(
        os.path.join(HERE, "screen_data", f"validate_{args.tag}.npz"),
        order=order, kinds=np.array(kinds),
        FCC=F, SIGMA=S, LIQUID=L, ok=ok, confirmed=confirmed)
    with open(os.path.join(HERE, "screen_data",
                           f"validate_{args.tag}.json"), "w") as f:
        json.dump({"tag": args.tag, "T_K": T_K,
                   "query": {"FCC_min": Q_FCC_MIN, "SIGMA_max": Q_SIGMA_MAX,
                             "LIQUID_max": Q_LIQ_MAX},
                   **out}, f, indent=2)
    print("[validate] precision:", out["hit"]["precision"],
          "| frontier:", out["frontier"]["precision"],
          "| bg miss rate:", round(1 - (out["bg"]["precision"] or 0), 4)
          if out["bg"]["precision"] is not None else None)
    print(f"[validate] wrote validate_{args.tag}.npz/.json")


if __name__ == "__main__":
    main()
