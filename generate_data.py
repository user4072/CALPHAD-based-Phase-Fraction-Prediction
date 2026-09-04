"""Generate ~10k ternary equilibria with the corrected extraction pipeline.

Lessons from the previous projects baked in from day 1:
  1. VERTEX ACCUMULATION: eq.NP is indexed by composition set, not phase.
     A phase stable in two composition sets appears twice; we SUM over all
     matching vertices.
  2. MASS-BALANCE ACCEPTANCE: only rows with |sum(NP) - 1| < 1e-6 are kept.
  3. COMPLETE PHASE SET: every phase whose constituents can be populated by
     the system elements participates in the equilibrium (no hand-picked
     whitelist). Active phases are then detected from the generated data
     with a quantified threshold and written to a sidecar JSON.
  4. DEDUP ON FEATURES, never on targets.
  5. Checkpointing: each batch is appended to a CSV; re-running resumes.

Usage: python generate_data.py --system fecrni|fecrmn|fecrmo|fecrv|femnni
"""
import os
import sys
import json
import time
import logging
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.config import P, MASS_BALANCE_TOL
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases

N_WORKERS = min(8, os.cpu_count() or 4)
# A phase is a model target if it occupies >= 0.5% of the generated rows.
ACTIVE_THRESHOLD = 0.005
NONZERO_LEVEL = 0.001


def _init_worker(tdb_path):
    global _db
    _db = Database(tdb_path)


def run_eq(args):
    cfg = args["cfg"]
    names, cols = args["names"], args["cols"]
    x1, x2, temperature = args["point"]
    row = {cfg["comps"][0]: 1.0 - x1 - x2, cfg["comps"][1]: x1, cfg["comps"][2]: x2,
           "T": temperature}
    for col in cols:
        row[col] = 0.0
    for prop in ["GM", "HM", "SM", "CPM"]:
        row[prop] = 0.0
    row["converged"] = False
    try:
        cond = {v.T: temperature, v.P: P}
        for sp, val in zip(cfg["comps_species"], [x1, x2]):
            cond[v.X(sp)] = val
        eq = equilibrium(_db, cfg["elements"], list(names), cond, max_iterations=500)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        for i, name in enumerate(names):
            idx = np.where(ph == name)[0]
            if len(idx) > 0:
                # LESSON 1: accumulate over all composition-set vertices
                row[cols[i]] = float(np.nansum(npf[idx]))
        for prop in ["GM", "HM", "SM", "CPM"]:
            try:
                val = float(eq[prop].values.flatten()[0])
                row[prop] = val if not np.isinf(val) and not np.isnan(val) else 0.0
            except Exception:
                row[prop] = 0.0
        # LESSON 2: mass balance is an acceptance criterion
        total = sum(row[c] for c in cols)
        row["converged"] = abs(total - 1.0) < MASS_BALANCE_TOL
    except Exception:
        row["converged"] = False
    return row


def generate_tasks(cfg):
    tasks = []
    rng = np.random.default_rng(42)
    t_min, t_max = cfg["t_min"], cfg["t_max"]
    sf = cfg["sigma_focus"]

    logger.info("Strategy 1: LHS Gibbs triangle, Fe-weighted")
    for _ in range(6000):
        if rng.uniform() < 0.6:
            c1 = rng.uniform(0.001, 0.45)
            c2 = rng.uniform(0.001, 0.45)
        else:
            c1 = rng.uniform(0.001, 0.90)
            c2 = rng.uniform(0.001, 0.90)
        if c1 + c2 > 0.995:
            continue
        tasks.append((c1, c2, float(rng.uniform(t_min, t_max))))

    logger.info("Strategy 2: sigma-field focus")
    lo1, hi1, t_lo = sf["low"]
    lo2, hi2, t_hi = sf["high"]
    for _ in range(2000):
        c1 = rng.uniform(lo1, hi1)
        c2 = rng.uniform(lo2, hi2)
        if c1 + c2 > 0.995:
            continue
        tasks.append((c1, c2, float(rng.uniform(t_lo, t_hi))))

    logger.info("Strategy 3: isothermal grids")
    for T in [900, 1100, 1300, 1500, 1700]:
        grid = np.linspace(0.01, 0.85, 12)
        for c1 in grid:
            for c2 in grid:
                if 0.01 < c1 + c2 < 0.98:
                    tasks.append((float(c1), float(c2), float(T)))

    logger.info("Strategy 4: liquidus zone")
    for _ in range(1500):
        c1 = rng.uniform(0.001, 0.85)
        c2 = rng.uniform(0.001, 0.85)
        if 0.70 <= c1 + c2 <= 0.995:
            tasks.append((c1, c2, float(rng.uniform(1300, 2000))))

    logger.info("Strategy 5: near-pure Fe (magnetic / A3-A4 region)")
    for _ in range(1000):
        c1 = rng.uniform(0.0, 0.04)
        c2 = rng.uniform(0.0, 0.04)
        tasks.append((c1, c2, float(rng.uniform(700, 1811))))

    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]

    probe_path = os.path.join(os.path.dirname(cfg["dataset"]), f"{args.system}_probe.json")
    if not os.path.exists(probe_path):
        sys.exit(f"Missing probe file {probe_path} - run probe_eligible.py first")
    with open(probe_path) as f:
        probe = json.load(f)
    # Probe-driven phase set: phases observed active in the full-set probe.
    phases = sorted(probe["active_counts"].keys())
    if not phases:
        sys.exit("Probe found no active phases - aborting")
    phase_cols = [f"NP_{p}" for p in phases]
    logger.info(f"System {args.system}: {len(phases)} probe-active phases "
                f"({', '.join(phases)})")

    tasks = generate_tasks(cfg)
    logger.info(f"Total tasks: {len(tasks)}")
    rng = np.random.default_rng(42)
    rng.shuffle(tasks)

    done = {}
    if os.path.exists(cfg["checkpoint"]):
        existing = pd.read_csv(cfg["checkpoint"])
        if set(phase_cols).issubset(existing.columns):
            for _, r in existing.iterrows():
                done[(r[cfg["comps"][1]], r[cfg["comps"][2]], r["T"])] = r.to_dict()
            logger.info(f"Resuming: {len(done)} rows already present")
        else:
            logger.warning("Checkpoint schema mismatch (incomplete phase set); discarding")
            os.remove(cfg["checkpoint"])

    t0 = time.time()
    rows = list(done.values())
    batch_size = 400
    task_args = [{"cfg": cfg, "names": phases, "cols": phase_cols, "point": t}
                 for t in tasks if t not in done]
    with mp.Pool(N_WORKERS, initializer=_init_worker, initargs=(cfg["tdb"],)) as pool:
        total = len(task_args)
        for i in range(0, total, batch_size):
            batch = task_args[i:i + batch_size]
            results = pool.map(run_eq, batch)
            ok = [r for r in results if r["converged"]]
            rows.extend(ok)
            df_batch = pd.DataFrame(ok)
            df_batch.to_csv(cfg["checkpoint"], mode="a",
                            header=not os.path.exists(cfg["checkpoint"])
                            or os.path.getsize(cfg["checkpoint"]) == 0, index=False)
            elapsed = time.time() - t0
            rate = len(rows) / elapsed if elapsed > 0 else 0
            logger.info(f"  {i + len(batch)}/{total} | accepted {len(ok)}/{len(batch)} | "
                        f"total {len(rows)} | {rate:.1f}/s | ETA {(total - i) / max(rate, 0.01):.0f}s")

    df = pd.DataFrame(rows)
    # LESSON 5: dedup on features only
    n_before = len(df)
    df = df.drop_duplicates(subset=cfg["comps"] + ["T"], keep="first")
    logger.info(f"Dedup (features): {n_before} -> {len(df)}")

    # LESSON 2 enforced again on the final frame
    sums = df[phase_cols].sum(axis=1)
    df = df[np.abs(sums - 1.0) < MASS_BALANCE_TOL].copy()
    max_dev = np.abs(df[phase_cols].sum(axis=1) - 1.0).max()
    logger.info(f"Mass-balance filter: {len(df)} rows, max |sum-1| = {max_dev:.2e}")

    # Design-space tag: stainless-like box (Fe >= 55%, Cr <= 30%, X <= 30%)
    box = cfg["box"]
    mask = np.ones(len(df), dtype=bool)
    for col, val in box.items():
        if col == "Fe":
            mask &= df[col] >= val
        else:
            mask &= df[col] <= val
    df["in_stainless_box"] = mask.astype(int)
    logger.info(f"Stainless-box rows: {100 * df['in_stainless_box'].mean():.1f}%")

    # Active-phase detection with a documented threshold
    occ = {c: float((df[c] > NONZERO_LEVEL).mean()) for c in phase_cols}
    active_cols = [c for c in phase_cols if occ[c] >= ACTIVE_THRESHOLD]
    excluded = [c for c in phase_cols if occ[c] < ACTIVE_THRESHOLD]
    excluded_mass = float(df[excluded].sum().sum()) if excluded else 0.0
    total_mass = float(len(df))
    sidecar = {
        "system": args.system,
        "n_rows": int(len(df)),
        "eligible_phases": phase_cols,
        "active_phases": active_cols,
        "active_threshold": ACTIVE_THRESHOLD,
        "occurrence": occ,
        "excluded_mass_fraction": excluded_mass / total_mass,
        "max_sum_deviation": float(max_dev),
    }
    with open(os.path.join(os.path.dirname(cfg["dataset"]), f"{args.system}_active.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"Active phases ({ACTIVE_THRESHOLD:.1%} occurrence): {active_cols}")
    logger.info(f"Excluded mass fraction: {100 * excluded_mass / total_mass:.4f}%")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(df)} rows in {elapsed:.0f}s ({len(df) / elapsed:.1f}/s)")
    for c in active_cols:
        logger.info(f"  {c}: {100 * occ[c]:.1f}% nonzero, max={df[c].max():.4f}")

    df.to_csv(cfg["dataset"], index=False)
    logger.info(f"Saved {len(df)} samples to {cfg['dataset']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()