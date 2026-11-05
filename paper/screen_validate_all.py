"""A2: validate EVERY shortlisted hit with full-set CALPHAD (28 phases).

Validates the shortlist points not covered by screen_validate.py
(11 frontier + 400 subset already done): full eligible phase set,
single-T (1000 K) solves, mass-balance gate. Chunked with per-chunk
saves -> resumable: re-running continues from the partial file.

Outputs (NEW files only):
  paper/screen_data/validate_all_hits_<tag>_partial.npz  (scratch/resume)
  paper/screen_data/validate_all_hits_<tag>.npz + .json  (final)

Usage:
    py -3.12 paper/screen_validate_all.py --tag fecrni_T1000K_step0.001 \\
        --workers 8 --chunks 8
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
from anchor_eval import _cal_init, _cal_run
import multiprocessing as mp

SYSTEM = "fecrni"
Q_FCC_MIN, Q_SIGMA_MAX, Q_LIQ_MAX = 0.99, 1e-3, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()

    SD = os.path.join(HERE, "screen_data")
    scr = np.load(os.path.join(SD, f"screen_{args.tag}.npz"))
    val = np.load(os.path.join(SD, f"validate_{args.tag}.npz"))
    X = scr["X"]
    T_K = float(X[0, 3])
    hit_idx = np.array(scr["hit_idx"])
    done_order = np.array(val["order"])[np.array([k for k in val["kinds"]]) != "bg"]
    todo = np.setdiff1d(hit_idx, done_order)
    print(f"[all] hits={len(hit_idx):,} already validated={len(done_order)} "
          f"remaining={len(todo):,}")

    partial = os.path.join(SD, f"validate_all_hits_{args.tag}_partial.npz")
    got = {}
    if os.path.exists(partial):
        p = np.load(partial, allow_pickle=True)
        for i, r in zip(p["order"], p["res"].tolist()):
            got[int(i)] = r
        print(f"[all] resumed {len(got)} completed points")
    todo = np.array([i for i in todo if int(i) not in got])
    n_pool = len(todo) + len(got)
    print(f"[all] to run: {len(todo):,}")

    cfg = SYSTEMS[SYSTEM]
    phases = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    pool = mp.Pool(args.workers, initializer=_cal_init,
                   initargs=(cfg["tdb"], cfg["elements"],
                             cfg["comps_species"], phases))
    t_all = time.time()
    try:
        for ci, blk in enumerate(np.array_split(todo, args.chunks)):
            if not len(blk):
                continue
            tasks = [((float(X[i, 1]), float(X[i, 2])), T_K) for i in blk]
            t0 = time.time()
            out = pool.map(_cal_run, tasks, chunksize=4)
            for i, r in zip(blk, out):
                got[int(i)] = {"ok": bool(r["ok"]),
                               "amounts": r["amounts"],
                               "total": r["total"], "err": r["err"]}
            np.savez_compressed(
                partial, order=np.array(sorted(got)),
                res=np.array([got[i] for i in sorted(got)], dtype=object))
            dt = time.time() - t0
            print(f"[all] chunk {ci + 1}: {len(blk)} pts in {dt:.0f}s "
                  f"({dt / len(blk):.2f} s/pt wall), "
                  f"total done {len(got)}/{n_pool}", flush=True)
    finally:
        pool.close()

    order = np.array(sorted(got))
    F = np.array([got[i]["amounts"].get("FCC_A1", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    S = np.array([got[i]["amounts"].get("SIGMA", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    L = np.array([got[i]["amounts"].get("LIQUID", 0.0)
                  if got[i]["ok"] else np.nan for i in order])
    ok = np.array([got[i]["ok"] for i in order])
    conf = ok & (F >= Q_FCC_MIN) & (S <= Q_SIGMA_MAX) & (L <= Q_LIQ_MAX)
    t_wall = time.time() - t_all
    summary = {
        "tag": args.tag, "n": int(len(order)), "n_ok": int(ok.sum()),
        "n_confirmed": int(conf.sum()),
        "precision": round(float(conf.sum() / ok.sum()), 4)
        if ok.sum() else None,
        "query": {"FCC_min": Q_FCC_MIN, "SIGMA_max": Q_SIGMA_MAX,
                  "LIQUID_max": Q_LIQ_MAX},
        "wall_s": round(t_wall, 1),
        "s_per_point_wall": round(t_wall / max(len(order), 1), 3),
    }
    np.savez_compressed(os.path.join(SD, f"validate_all_hits_{args.tag}.npz"),
                        order=order, FCC=F, SIGMA=S, LIQUID=L,
                        ok=ok, confirmed=conf)
    with open(os.path.join(SD, f"validate_all_hits_{args.tag}.json"),
              "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[all] precision {summary['precision']} "
          f"({summary['n_confirmed']}/{summary['n_ok']}) "
          f"in {t_wall:.0f}s wall")
    print(f"[all] wrote validate_all_hits_{args.tag}.npz/.json")


if __name__ == "__main__":
    main()
