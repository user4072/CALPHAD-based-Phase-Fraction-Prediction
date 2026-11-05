"""EXPERIMENTAL ANCHOR evaluation: surrogate vs full-set CALPHAD vs published DTA.

Pipeline (all outputs are NEW files; nothing existing is modified):

  1. Read paper/anchor_data/anchors.csv, pull the published DTA anchor values
     (Drozdova 2017 Fe-Cr-C alloys A/B/C, Yamada 1987 Fe-Cr-Ni 21.0Cr-16.6Ni)
     and convert their wt% compositions to mole fractions (Fe balance).
  2. Train (or reload from the paper/anchor_data/anchor_ckpts/ cache) the MLP
     heads of train_mlp.py over seeds 42/123/2024:
        fecrc  -> [sig_norm, renorm]      fecrni -> [renorm]
     Predictions are the ensemble mean (over the 3 seeds) of the projected
     fraction vectors.
  3. Temperature sweeps 600-1900 degC on a 5 K grid (+1 K refinement +-5 K
     around every detected transition, range extension if a transition is not
     covered) evaluated twice on the identical grid:
        (a) FULL-SET pycalphad reference - ALL eligible phases of the pruned
            system TDB (same eligible set as databases/mc_fe_v2.062.tdb, cf.
            analysis_revision/validate_phase_set_full.py), 8-worker pool;
        (b) the surrogate (instant).
  4. Transition extraction on heating:
        T_first_liquid = smallest T with NP_LIQUID > 1e-4  (start of melting;
                         published T_S / T_P quantity)
        T_last_solid   = largest  T with any solid > 1e-4   (liquidus;
                         published T_L quantity)
     plus the phase identities at both transitions (L+delta check for B).
  5. Comparisons: surrogate-vs-CALPHAD (model contribution), CALPHAD-vs-DTA
     (database contribution, for the PUBLISHED quantity of each alloy), and
     surrogate-vs-DTA (total), plus max |NP_sur - NP_CAL| over the whole
     sweep (figure of merit).

Usage:
    $env:PYTHONIOENCODING="utf-8"
    py -3.12 paper/anchor_eval.py [--train-workers 4] [--skip-pe]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import multiprocessing as mp
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from pycalphad import Database, equilibrium, variables as v

from fe_surrogate.config import MASS_BALANCE_TOL, P
from fe_surrogate.experiment import SEEDS, active_phases, cluster_split, load_data, renorm
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import eligible_phases
from train_mlp import MLP4, train_mlp

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ATOMIC_WEIGHTS = {"Fe": 55.845, "Cr": 51.996, "C": 12.011, "Ni": 58.693}

ANCHORS_CSV = os.path.join(HERE, "anchor_data", "anchors.csv")
OUT_JSON = os.path.join(HERE, "anchor_data", "anchor_eval.json")
CKPT_DIR = os.path.join(HERE, "anchor_data", "anchor_ckpts")

COARSE_T_MIN_C, COARSE_T_MAX_C, COARSE_STEP_C = 600.0, 1900.0, 5.0
EXTEND_DOWN_C, EXTEND_UP_C = 400.0, 2400.0
REFINE_HALF_WIDTH_K, REFINE_STEP_C = 5, 1.0
LIQUID_FRAC_THR = 1e-4
# Sensitivity of the detected transitions to the detection threshold: the
# surrogate never predicts exactly zero, so the 1e-4 crossing sits on a tail.
THR_SENSITIVITY = (1e-4, 1e-3, 1e-2)
NEAR_TRANSITION_WINDOW_K = 15.0
N_CALPHAD_WORKERS = min(8, os.cpu_count() or 4)
HIDDEN = (192, 192, 192)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEADS = {"fecrc": ["sig_norm", "renorm"], "fecrni": ["renorm"]}
PRIMARY_HEAD = {"fecrc": "sig_norm", "fecrni": "renorm"}
# train_mlp trains "renorm" as a plain sigmoid model projected at inference.
MODEL_HEAD_ARG = {"sig_norm": "sig_norm", "renorm": "sigmoid"}

# Published quantity -> which of the two computed transitions it corresponds to.
TRANSITION_OF_PUBLISHED = {
    "T_L": ("T_last_solid_C",
            "T_L (liquidus) on heating is the highest temperature that still "
            "contains solid, i.e. where the last solid fraction disappears: "
            "T_last_solid."),
    "T_S": ("T_first_liquid_C",
            "T_S (solidus / start of melting) on heating is the first "
            "temperature at which a liquid fraction appears: T_first_liquid."),
    "T_P": ("T_first_liquid_C",
            "T_P (start of the peritectic transformation on heating) is the "
            "first appearance of liquid; L+delta coexistence at that point is "
            "the peritectic-like first-melting signature: T_first_liquid."),
}

ANCHOR_SPECS = [
    {"label": "A", "source_id": "drozdova2017", "alloy_id": "A",
     "system": "fecrc", "published_qty": "T_L"},
    {"label": "B", "source_id": "drozdova2017", "alloy_id": "B",
     "system": "fecrc", "published_qty": "T_P"},
    {"label": "C", "source_id": "drozdova2017", "alloy_id": "C",
     "system": "fecrc", "published_qty": "T_S"},
    {"label": "Y_21.0Cr-16.6Ni", "source_id": "yamada1987",
     "alloy_id": "DTA_21.0Cr-16.6Ni", "system": "fecrni",
     "published_qty": "T_L"},
]

# Optional: Yamada peritectic-eutectic transition point (composition, not a
# temperature) - evaluated qualitatively (which solid coexists with the first
# liquid: delta = peritectic-like, gamma = eutectic-like).
PE_SPEC = {"label": "PE_15Cr-10Ni", "source_id": "yamada1987",
           "alloy_id": "PE_transition_point", "system": "fecrni",
           "published_qty": None}

WT_COLS = [("C", "wt_pct_C"), ("Cr", "wt_pct_Cr"), ("Ni", "wt_pct_Ni"),
           ("Mo", "wt_pct_Mo"), ("Mn", "wt_pct_Mn"), ("Si", "wt_pct_Si"),
           ("N", "wt_pct_N")]


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Anchors: composition + published values
# --------------------------------------------------------------------------- #
def wt_to_mole(wt_pct: dict) -> tuple[dict, dict, float]:
    """wt% -> (mole fractions, moles per 100 g, total moles per 100 g)."""
    moles = {el: w / ATOMIC_WEIGHTS[el] for el, w in wt_pct.items()}
    total = sum(moles.values())
    x = {el: m / total for el, m in moles.items()}
    return x, moles, total


def pick_published(df: pd.DataFrame, source_id: str, alloy_id: str, qty: str) -> dict:
    q = df[(df["source_id"] == source_id) & (df["alloy_id"] == alloy_id)
           & (df["quantity"] == qty)]
    if q.empty:
        raise RuntimeError(f"no rows for {source_id}/{alloy_id}/{qty}")
    dta = q[q["method"].str.contains("DTA", na=False)
            & ~q["method"].str.contains("calculated", case=False, na=False)]
    if dta.empty:
        raise RuntimeError(f"no DTA row for {source_id}/{alloy_id}/{qty}")
    r = dta.iloc[0]
    return {"value_C": float(r["value"]), "unit": r["unit"],
            "method": r["method"], "uncertainty": r["uncertainty"],
            "page_ref": r["page_ref"], "notes": r["notes"]}


def composition_from_rows(rows: pd.DataFrame) -> dict:
    """Reported non-Fe elements (wt%); Fe is the balance. NaN -> not reported."""
    wt = {}
    for el, col in WT_COLS:
        vals = rows[col].dropna()
        if len(vals) == 0:
            continue
        if not np.allclose(vals.values.astype(float), vals.values.astype(float)[0]):
            raise RuntimeError(f"inconsistent {col} within alloy rows")
        val = float(vals.values[0])
        if val != 0.0:
            wt[el] = val
    wt["Fe"] = 100.0 - sum(wt.values())
    return wt


def build_target(spec: dict, df: pd.DataFrame) -> dict:
    rows = df[(df["source_id"] == spec["source_id"])
              & (df["alloy_id"] == spec["alloy_id"])]
    if rows.empty:
        raise RuntimeError(f"no rows for {spec['source_id']}/{spec['alloy_id']}")
    wt = composition_from_rows(rows)
    x, moles, total = wt_to_mole(wt)
    sanity = {
        "moles_per_100g": moles,
        "total_moles_per_100g": total,
        "mole_sum": float(sum(x.values())),
        "fe_balance_wt_pct": wt["Fe"],
        "mole_sum_error": float(abs(sum(x.values()) - 1.0)),
        "fe_balance_error": float(abs(x["Fe"] - (1.0 - sum(v for k, v in x.items()
                                                           if k != "Fe")))),
    }
    t = {"label": spec["label"], "system": spec["system"],
         "kind": "anchor" if spec["published_qty"] else "qualitative",
         "source_id": spec["source_id"], "alloy_id": spec["alloy_id"],
         "wt_pct": wt, "mole_frac": x, "conversion_sanity": sanity,
         "published_qty": spec["published_qty"], "published": None}
    if spec["published_qty"]:
        t["published"] = pick_published(df, spec["source_id"], spec["alloy_id"],
                                        spec["published_qty"])
    return t


def design_box_check(system: str, x: dict) -> dict:
    """Is the anchor inside the dataset design box? (flagged either way)."""
    cfg = SYSTEMS[system]
    df = pd.read_csv(cfg["dataset"], usecols=cfg["comps"])
    checks = {}
    if cfg.get("box_ranges"):
        for el, (lo, hi) in cfg["box_ranges"].items():
            checks[f"{el}_in_[{lo},{hi}]"] = bool(lo <= x[el] <= hi)
    box = cfg["box"]
    for el, lim in box.items():
        if el == "Fe":
            checks[f"Fe_stainless_box_>= {lim}"] = bool(x[el] >= lim)
        else:
            checks[f"{el}_stainless_box_<= {lim}"] = bool(x[el] <= lim)
    for el in cfg["comps"]:
        lo, hi = float(df[el].min()), float(df[el].max())
        checks[f"{el}_in_dataset_[{lo:.4f},{hi:.4f}]"] = bool(lo <= x[el] <= hi)
    # observed-occurrence counts for exact anchor rows
    mask = np.ones(len(df), dtype=bool)
    for el in cfg["comps"]:
        mask &= np.isclose(df[el].values, x[el], atol=1e-6)
    return {"in_design_box": bool(all(checks.values())), "checks": checks,
            "n_dataset_rows_within_1e-4": int(mask.sum())}


# --------------------------------------------------------------------------- #
# Models: train (or load cached) one (system, head, seed)
# --------------------------------------------------------------------------- #
def ckpt_path(system: str, head: str, seed: int) -> str:
    return os.path.join(CKPT_DIR, f"{system}_{head}_s{seed}.pt")


def _train_job(job) -> dict:
    system, head, seed, epochs = job
    path = ckpt_path(system, head, seed)
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return {"system": system, "head": head, "seed": seed, "cached": True,
                "val_mae": ck["val_mae"], "test_mae": ck["test_mae"],
                "time_s": ck.get("time_s", None)}
    t0 = time.time()
    X, Y, _df = load_data(system)
    _cols, names = active_phases(system)
    tr, va, te = cluster_split(X, Y, seed)
    model = MLP4(n_phases=len(names), head=MODEL_HEAD_ARG[head], hidden=HIDDEN,
                 n_in=X.shape[1]).to(DEVICE)
    model, val_mae, scaler = train_mlp(model, X, Y, tr, va, seed, epochs)
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(scaler.transform(X[te]), dtype=torch.float32,
                               device=DEVICE)).cpu().numpy()
    if head == "renorm":
        p = renorm(p)
    test_mae = float(np.mean(np.abs(p - Y[te])))
    ck = {"system": system, "head": head, "model_head": MODEL_HEAD_ARG[head],
          "seed": seed, "n_phases": len(names), "phase_names": names,
          "n_in": int(X.shape[1]), "hidden": list(HIDDEN),
          "state_dict": {k: v_.detach().cpu() for k, v_ in model.state_dict().items()},
          "scaler_mean": torch.tensor(scaler.mean_),
          "scaler_scale": torch.tensor(scaler.scale_),
          "val_mae": float(val_mae), "test_mae": test_mae,
          "time_s": round(time.time() - t0, 1)}
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(ck, path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"system": system, "head": head, "seed": seed, "cached": False,
            "val_mae": ck["val_mae"], "test_mae": test_mae, "time_s": ck["time_s"]}


def ensure_models(systems, workers: int, epochs: int, use_cache: bool) -> list:
    jobs = [(s, h, se, epochs) for s in systems for h in HEADS[s] for se in SEEDS]
    if not use_cache:
        for s, h, se, _e in jobs:
            p = ckpt_path(s, h, se)
            if os.path.exists(p):
                os.remove(p)
    pending = [j for j in jobs if not os.path.exists(ckpt_path(j[0], j[1], j[2]))]
    log(f"[train] {len(jobs)} models ({len(pending)} to train, "
        f"{len(jobs) - len(pending)} cached) | device {DEVICE} | workers {workers}")
    summaries = []
    if pending:
        if workers <= 1:
            for j in pending:
                r = _train_job(j)
                summaries.append(r)
                log(f"  trained {r['system']}/{r['head']}/s{r['seed']}: "
                    f"val={r['val_mae']:.4f} test={r['test_mae']:.4f} "
                    f"({r['time_s']}s)")
        else:
            with mp.Pool(workers) as pool:
                for r in pool.imap_unordered(_train_job, pending):
                    summaries.append(r)
                    log(f"  trained {r['system']}/{r['head']}/s{r['seed']}: "
                        f"val={r['val_mae']:.4f} test={r['test_mae']:.4f} "
                        f"({r['time_s']}s)")
    for s, h, se, _e in jobs:
        if all(not (r["system"] == s and r["head"] == h and r["seed"] == se)
               for r in summaries):
            ck = torch.load(ckpt_path(s, h, se), map_location="cpu",
                            weights_only=False)
            summaries.append({"system": s, "head": h, "seed": se, "cached": True,
                              "val_mae": ck["val_mae"], "test_mae": ck["test_mae"],
                              "time_s": ck.get("time_s")})
    return sorted(summaries, key=lambda r: (r["system"], r["head"], r["seed"]))


def load_model_bundle(system: str, head: str, seed: int):
    ck = torch.load(ckpt_path(system, head, seed), map_location=DEVICE,
                    weights_only=False)
    model = MLP4(n_phases=ck["n_phases"], head=ck["model_head"],
                 hidden=tuple(ck["hidden"]), n_in=ck["n_in"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["scaler_mean"].cpu().numpy(), \
        ck["scaler_scale"].cpu().numpy(), ck["phase_names"]


# --------------------------------------------------------------------------- #
# Full-set pycalphad sweep
# --------------------------------------------------------------------------- #
_CAL = {"db": None, "elems": None, "species": None, "phases": None}
_DB_CACHE: dict = {}


def _cal_init(tdb_path, elems, species, phases):
    warnings.filterwarnings("ignore")
    if tdb_path not in _DB_CACHE:
        _DB_CACHE[tdb_path] = Database(tdb_path)
    _CAL["db"] = _DB_CACHE[tdb_path]
    _CAL["elems"] = elems
    _CAL["species"] = species
    _CAL["phases"] = phases


def _cal_run(task) -> dict:
    free, t_k = task
    try:
        cond = {v.T: t_k, v.P: P}
        for sp, val in zip(_CAL["species"], free):
            cond[v.X(sp)] = val
        eq = equilibrium(_CAL["db"], _CAL["elems"], list(_CAL["phases"]), cond)
        ph = np.asarray(eq.Phase.values).flatten()
        npf = np.asarray(eq.NP.values).flatten()
        amounts = {}
        for p in _CAL["phases"]:
            idx = np.where(ph == p)[0]
            amounts[p] = float(np.nansum(npf[idx])) if len(idx) else 0.0
        total = float(sum(amounts.values()))
        ok = abs(total - 1.0) < MASS_BALANCE_TOL
        return {"ok": ok, "amounts": amounts if ok else None, "total": total,
                "err": None if ok else f"mass balance {total:.3e}"}
    except Exception as exc:  # solver failure -> logged
        return {"ok": False, "amounts": None, "total": None,
                "err": repr(exc)[:300]}


class CalphadRunner:
    """One worker pool per target (system TDB loaded once per worker)."""

    def __init__(self, system: str, phases: list):
        cfg = SYSTEMS[system]
        self.cfg = cfg
        self.phases = phases
        self.failures = []
        self.pool = mp.Pool(N_CALPHAD_WORKERS, initializer=_cal_init,
                             initargs=(cfg["tdb"], cfg["elements"],
                                       cfg["comps_species"], phases))

    def run_missing(self, free, t_list, cache: dict) -> list:
        missing = [t for t in t_list if round(t, 6) not in cache]
        if missing:
            tasks = [(tuple(free), t + 273.15) for t in missing]
            res = self.pool.map(_cal_run, tasks, chunksize=2)
            for t, r in zip(missing, res):
                cache[round(t, 6)] = r
        return [cache[round(t, 6)] for t in t_list]

    def close(self):
        self.pool.close()
        self.pool.join()


def species_to_element(sp: str) -> str:
    return sp.capitalize()


def results_to_matrix(results, phases):
    n = len(results)
    F = np.full((n, len(phases)), np.nan)
    ok = np.zeros(n, dtype=bool)
    fails = []
    for i, r in enumerate(results):
        if r["ok"] and r["amounts"] is not None:
            F[i] = [r["amounts"].get(p, 0.0) for p in phases]
            ok[i] = True
        else:
            fails.append((i, r.get("err")))
    return F, ok, fails


# --------------------------------------------------------------------------- #
# Transition extraction
# --------------------------------------------------------------------------- #
def extract_transitions(t_c, F, ok, phases, thr=LIQUID_FRAC_THR) -> dict:
    """On heating (ascending T): first liquid appearance + last solid."""
    t_c = np.asarray(t_c, dtype=float)
    names = list(phases)
    liq_idx = names.index("LIQUID") if "LIQUID" in names else None

    def phases_at(i, minimum=thr):
        return [names[j] for j in range(len(names))
                if np.isfinite(F[i, j]) and F[i, j] > minimum]

    out = {"T_first_liquid_C": None, "T_last_solid_C": None,
           "phases_at_first_liquid": [], "solids_at_first_liquid": [],
           "assemblage_below_first_liquid": [],
           "liquid_frac_at_first_liquid": None,
           "phases_at_last_solid": [], "solids_at_last_solid": [],
           "assemblage_above_last_solid": [],
           "solid_frac_at_last_solid": None,
           "T_first_liquid_sustained_C": None,
           "spurious_low_T_liquid_points": [],
           "T_first_liquid_plus_BCC_A2_C": None,
           "solids_at_first_liquid_plus_BCC_A2": [],
           "coverage_ok": False}
    if not ok.any():
        return out
    liq = F[:, liq_idx] if liq_idx is not None else np.zeros(len(t_c))
    solid = np.array([np.nansum(np.delete(F[i], liq_idx))
                      if liq_idx is not None else np.nansum(F[i])
                      for i in range(len(t_c))])
    liq = np.where(ok, liq, np.nan)
    solid = np.where(ok, solid, np.nan)

    first = [i for i in range(len(t_c)) if ok[i] and liq[i] > thr]
    last = [i for i in range(len(t_c)) if ok[i] and solid[i] > thr]
    if first:
        i = first[0]
        out["T_first_liquid_C"] = float(t_c[i])
        out["phases_at_first_liquid"] = phases_at(i)
        out["solids_at_first_liquid"] = [p for p in out["phases_at_first_liquid"]
                                         if p != "LIQUID"]
        out["liquid_frac_at_first_liquid"] = float(liq[i])
        if i > 0:
            out["assemblage_below_first_liquid"] = phases_at(i - 1)
    if last:
        i = last[-1]
        out["T_last_solid_C"] = float(t_c[i])
        out["phases_at_last_solid"] = phases_at(i)
        out["solids_at_last_solid"] = [p for p in out["phases_at_last_solid"]
                                       if p != "LIQUID"]
        out["solid_frac_at_last_solid"] = float(solid[i])
        if i + 1 < len(t_c):
            out["assemblage_above_last_solid"] = phases_at(i + 1)
    t_min, t_max = float(t_c[ok].min()), float(t_c[ok].max())

    # Noise-robust (sustained) onset of melting: first index from which the
    # liquid fraction never again falls to/below thr (failed points ignored).
    above = ok & (liq > thr)
    tail = np.zeros(len(t_c), dtype=bool)
    acc = True
    for i in range(len(t_c) - 1, -1, -1):
        acc = (bool(above[i]) and acc) if ok[i] else acc
        tail[i] = acc
    sus = int(np.argmax(tail)) if tail.any() else None
    if sus is not None:
        out["T_first_liquid_sustained_C"] = float(t_c[sus])
        if first and first[0] < sus:
            out["spurious_low_T_liquid_points"] = [
                {"T_C": float(t_c[i]), "liquid_frac": float(liq[i])}
                for i in first if i < sus]
    elif first:
        out["spurious_low_T_liquid_points"] = [
            {"T_C": float(t_c[i]), "liquid_frac": float(liq[i])}
            for i in first]

    # First coexistence of liquid + delta ferrite (peritectic-like melting),
    # scanned from the sustained melting branch only (raw 1e-4 patches are
    # numerical tails, not melting).
    if "BCC_A2" in names:
        bcc = names.index("BCC_A2")
        start = sus if sus is not None else (first[0] if first else 0)
        both = [i for i in range(start, len(t_c))
                if ok[i] and liq[i] > thr and F[i, bcc] > thr]
        if both:
            out["T_first_liquid_plus_BCC_A2_C"] = float(t_c[both[0]])
            out["solids_at_first_liquid_plus_BCC_A2"] = \
                [names[j] for j in range(len(names))
                 if j != liq_idx and np.isfinite(F[both[0], j])
                 and F[both[0], j] > thr]

    out["coverage_ok"] = (out["T_first_liquid_C"] is not None
                          and out["T_last_solid_C"] is not None
                          and out["T_first_liquid_C"] > t_min
                          and out["T_last_solid_C"] < t_max)
    out["_first_at_min"] = out["T_first_liquid_C"] is not None and \
        out["T_first_liquid_C"] <= t_min + 1e-9
    out["_last_at_max"] = out["T_last_solid_C"] is not None and \
        out["T_last_solid_C"] >= t_max - 1e-9
    out["_first_missing"] = out["T_first_liquid_C"] is None
    out["_last_missing"] = out["T_last_solid_C"] is None
    return out


def refine_grid(transitions) -> set:
    pts = set()
    for key in ("T_first_liquid_C", "T_last_solid_C",
                "T_first_liquid_sustained_C"):
        t = transitions.get(key)
        if t is None:
            continue
        base = float(np.round(t))
        for dt in range(-REFINE_HALF_WIDTH_K, REFINE_HALF_WIDTH_K + 1):
            pts.add(base + dt * REFINE_STEP_C)
    return pts


def calphad_sweep(system, x, label, log_prefix=""):
    """Coarse 5 K sweep + range extension + 1 K refinement -> final grid."""
    cfg = SYSTEMS[system]
    free = [x[species_to_element(sp)] for sp in cfg["comps_species"]]
    phases = sorted(eligible_phases(cfg["tdb"], cfg["comps"]))
    cache: dict = {}
    grid = set(float(t) for t in
               np.arange(COARSE_T_MIN_C, COARSE_T_MAX_C + 1e-9, COARSE_STEP_C))
    runner = CalphadRunner(system, phases)
    failures = []
    extensions = []
    trans = {}
    try:
        for _rnd in range(3):
            ts = sorted(grid)
            res = runner.run_missing(free, ts, cache)
            F, ok, fails = results_to_matrix(res, phases)
            for i, err in fails:
                failures.append({"T_C": ts[i], "err": err})
            trans = extract_transitions(ts, F, ok, phases)
            add = set()
            reason = []
            if trans["_first_missing"] or trans["_last_at_max"]:
                add |= {float(t) for t in
                        np.arange(max(grid) + COARSE_STEP_C,
                                  EXTEND_UP_C + 1e-9, COARSE_STEP_C)}
                reason.append("no first-liquid or solid persists at range top")
            if trans["_last_missing"] or trans["_first_at_min"]:
                add |= {float(t) for t in
                        np.arange(EXTEND_DOWN_C, min(grid) - 1e-9,
                                  COARSE_STEP_C)}
                reason.append("no last-solid or liquid already at range bottom")
            add -= grid
            if not add:
                break
            extensions.append({"added_points": len(add),
                               "range_C": [min(add), max(add)],
                               "reason": "; ".join(reason)})
            log(f"  {log_prefix}extending sweep by {len(add)} points "
                f"({min(add):.0f}-{max(add):.0f} degC): {'; '.join(reason)}")
            grid |= add
        ref = refine_grid(trans)
        if ref:
            extensions.append({"added_points": len(ref - grid),
                               "range_C": [min(ref), max(ref)],
                               "reason": "1 K refinement +-5 K around "
                                         "detected transitions"})
        grid |= ref
        ts = sorted(grid)
        res = runner.run_missing(free, ts, cache)
        F, ok, fails = results_to_matrix(res, phases)
        for i, err in fails:
            failures.append({"T_C": ts[i], "err": err})
        trans = extract_transitions(ts, F, ok, phases)
    finally:
        runner.close()
    coverage = {k: trans.pop(k) for k in
                ("_first_at_min", "_last_at_max", "_first_missing",
                 "_last_missing") if k in trans}
    log(f"  {log_prefix}CALPHAD grid {len(ts)} pts, ok {int(ok.sum())}, "
        f"fail {len(fails)} | T_first_liquid={trans['T_first_liquid_C']} "
        f"T_last_solid={trans['T_last_solid_C']} degC")
    return {"T_C": ts, "F": F, "ok": ok, "phases": phases,
            "transitions": trans, "failures": failures,
            "extensions": extensions, "coverage": coverage}


# --------------------------------------------------------------------------- #
# Surrogate sweeps
# --------------------------------------------------------------------------- #
def make_features(x, comps, t_c) -> np.ndarray:
    X = np.empty((len(t_c), len(comps) + 1), dtype=np.float64)
    for j, c in enumerate(comps):
        X[:, j] = x[c]
    X[:, -1] = np.asarray(t_c, dtype=float) + 273.15
    return X


def predict_curve(model, mean, scale, X):
    with torch.no_grad():
        raw = model(torch.tensor((X - mean) / scale, dtype=torch.float32,
                                 device=DEVICE)).cpu().numpy()
    return raw


def surrogate_sweep(system, head, bundles, x, t_c):
    """Per-seed + ensemble fraction curves on the CALPHAD grid."""
    cfg = SYSTEMS[system]
    X = make_features(x, cfg["comps"], t_c)
    per_seed, seed_trans, sums = {}, {}, {}
    raw_sum_dev = None
    ensemble = None
    for seed in SEEDS:
        model, mean, scale, names = bundles[(system, head, seed)]
        raw = predict_curve(model, mean, scale, X)
        if raw_sum_dev is None:
            raw_sum_dev = float(np.abs(raw.sum(axis=1) - 1.0).max())
        pred = renorm(raw) if head == "renorm" else raw
        per_seed[seed] = pred
        sums[seed] = float(np.abs(pred.sum(axis=1) - 1.0).max())
        ok = np.ones(len(t_c), dtype=bool)
        seed_trans[seed] = extract_transitions(t_c, pred, ok, names)
        ensemble = pred if ensemble is None else ensemble + pred
    ensemble = ensemble / len(SEEDS)
    ens_sum_dev = float(np.abs(ensemble.sum(axis=1) - 1.0).max())
    ok = np.ones(len(t_c), dtype=bool)
    ens_trans = extract_transitions(t_c, ensemble, ok, names)
    for k in ("_first_at_min", "_last_at_max", "_first_missing", "_last_missing"):
        ens_trans.pop(k, None)
        for s in seed_trans.values():
            s.pop(k, None)
    return {"phase_names": names, "per_seed": per_seed,
            "seed_transitions": seed_trans, "ensemble": ensemble,
            "ensemble_transitions": ens_trans,
            "max_sum_dev_per_seed": max(sums.values()),
            "max_sum_dev_ensemble": ens_sum_dev,
            "max_raw_sum_dev_before_projection": raw_sum_dev}


def fraction_deviation(cal, sur):
    """max |NP_sur - NP_CAL| over all sweep points and all phases."""
    cal_phases, sur_phases = cal["phases"], sur["phase_names"]
    all_phases = sorted(set(cal_phases) | set(sur_phases))
    n = len(cal["T_C"])
    cal_col = {p: j for j, p in enumerate(cal_phases)}
    ci = {p: j for j, p in enumerate(all_phases)}
    C = np.zeros((n, len(all_phases)))
    S = np.zeros((n, len(all_phases)))
    for p, j in cal_col.items():
        C[:, ci[p]] = np.where(cal["ok"], cal["F"][:, j], 0.0)
    for j, p in enumerate(sur_phases):
        S[:, ci[p]] = sur["ensemble"][:, j]
    dev = np.abs(S - C)[cal["ok"]]
    active_idx = [ci[p] for p in sur_phases]
    per_seed = {}
    for seed, pred in sur["per_seed"].items():
        Ss = np.zeros_like(S)
        for j, p in enumerate(sur_phases):
            Ss[:, ci[p]] = pred[:, j]
        per_seed[str(seed)] = float(np.abs(Ss - C)[cal["ok"]].max())
    extra = []
    for p in all_phases:
        if p in sur_phases:
            continue
        col = np.where(cal["ok"], cal["F"][:, cal_col[p]], 0.0)
        if float(np.nanmax(col)) > 1e-4:
            extra.append({"phase": p, "max_NP_calphad": float(np.nanmax(col))})

    # where does the maximum occur, and how does it look away from melting?
    ok_idx = np.flatnonzero(cal["ok"])
    D = np.abs(S - C)
    D[~cal["ok"]] = 0.0

    def _loc(i, j):
        if i is None:
            return None
        return {"T_C": float(cal["T_C"][i]), "phase": all_phases[j],
                "NP_calphad": float(C[i, j]), "NP_surrogate": float(S[i, j]),
                "abs_dNP": float(D[i, j])}

    flat = np.argwhere(D > 0.0)
    i_max, j_max = max(flat, key=lambda ij: D[ij[0], ij[1]]) if flat.size else (None, None)
    loc = _loc(i_max, j_max)
    t_c = np.asarray(cal["T_C"], dtype=float)[ok_idx]
    # the melting interval itself (solidus +- window .. liquidus +- window) is
    # where a few-K offset between a smooth regressor and a sharp CALPHAD jump
    # necessarily produces large fraction differences; report both in and out
    fl = cal["transitions"].get("T_first_liquid_C")
    ls = cal["transitions"].get("T_last_solid_C")
    window = None
    if fl is not None and ls is not None and ls >= fl:
        window = [float(fl) - NEAR_TRANSITION_WINDOW_K,
                  float(ls) + NEAR_TRANSITION_WINDOW_K]
    in_melting = (np.zeros(t_c.shape, dtype=bool) if window is None
                  else (t_c >= window[0]) & (t_c <= window[1]))
    dev_ok = dev if dev.shape == t_c.shape else D[ok_idx]
    outside = dev_ok[~in_melting]
    outside_loc = None
    if outside.size:
        rows = np.flatnonzero(~in_melting)
        a, b = np.unravel_index(int(np.argmax(dev_ok[rows])),
                                dev_ok[rows].shape)
        outside_loc = _loc(int(ok_idx[rows[a]]), int(b))
    return {"max_abs_dNP_all_phases": float(dev.max()),
            "max_abs_dNP_surrogate_phases": float(dev[:, active_idx].max()),
            "max_abs_dNP_per_seed": per_seed,
            "max_abs_dNP_location": loc,
            "max_abs_dNP_outside_melting_region": (
                float(outside.max()) if outside.size else None),
            "max_abs_dNP_outside_melting_region_location": outside_loc,
            "mean_abs_dNP_all_phases": float(dev.mean()),
            "median_abs_dNP_all_phases": float(np.median(dev)),
            "melting_region_window_C": window,
            "n_points_in_melting_region": int(in_melting.sum()),
            "n_points_outside_melting_region": int((~in_melting).sum()),
            "n_points_compared": int(cal["ok"].sum()),
            "phases_nonzero_in_calphad_not_predicted": extra}


def trans_value(trans, key):
    return trans.get(key)


def delta(a, b):
    return None if (a is None or b is None) else float(a - b)


def transition_at_threshold(t_c, liq, thr):
    """(first liquid, last solid) in degC from a liquid-fraction curve."""
    t_c = np.asarray(t_c, dtype=float)
    first = last = None
    for i, v in enumerate(liq):
        if np.isfinite(v) and v > thr:
            first = float(t_c[i])
            break
    for i in range(len(t_c) - 1, -1, -1):
        v = liq[i]
        if np.isfinite(v) and (1.0 - v) > thr:
            last = float(t_c[i])
            break
    return first, last


def threshold_sensitivity(t_c, liq_cal, liq_sur):
    """Detected transitions as a function of the detection threshold.

    The surrogate is a smooth regressor that never predicts exactly zero, so
    its 1e-4 crossing sits on a decaying tail whereas CALPHAD jumps from exact
    zero. Scanning a few thresholds shows how much of the delta is that tail.
    """
    out = {}
    for thr in THR_SENSITIVITY:
        fc, lc = transition_at_threshold(t_c, liq_cal, thr)
        fs, ls = transition_at_threshold(t_c, liq_sur, thr)
        out[f"{thr:g}"] = {
            "T_first_liquid_C": {"calphad": fc, "surrogate": fs,
                                 "sur_minus_cal": delta(fs, fc)},
            "T_last_solid_C": {"calphad": lc, "surrogate": ls,
                               "sur_minus_cal": delta(ls, lc)},
        }
    return out


def _dict_diff(a: dict, b: dict):
    """max |a-b| over the union of phases (missing = 0) and where it occurs."""
    keys = sorted(set(a) | set(b))
    diffs = [(abs(a.get(k, 0.0) - b.get(k, 0.0)), k) for k in keys]
    d, k = max(diffs, default=(0.0, None))
    return float(d), k


def phase_set_spot_check(system, x, cal, sur, loc):
    """Re-evaluate the point of max deviation with the probe-active phase set.

    The surrogate was trained on probe-active-set data, so a disagreement with
    the full-set CALPHAD reference can come either from the phase set itself
    (reference-side) or from the model. One equilibrium with the probe-active
    phases at that single temperature separates the two.
    """
    if loc is None:
        return None
    cfg = SYSTEMS[system]
    t0 = float(loc["T_C"])
    idx = int(np.argmin(np.abs(np.asarray(cal["T_C"], dtype=float) - t0)))
    if not np.isclose(float(cal["T_C"][idx]), t0) or not cal["ok"][idx]:
        return None
    _, probe_names = active_phases(system)
    probe_names = sorted(probe_names)
    full = {p: float(cal["F"][idx, j]) for j, p in enumerate(cal["phases"])
            if np.isfinite(cal["F"][idx, j])}
    free = [x[species_to_element(sp)] for sp in cfg["comps_species"]]
    _cal_init(cfg["tdb"], cfg["elements"], cfg["comps_species"], probe_names)
    res = _cal_run((tuple(free), t0 + 273.15))
    out = {"T_C": t0,
           "trigger": loc,
           "probe_active_phases": probe_names,
           "n_full_phases": len(cal["phases"]),
           "probe_set_ok": bool(res["ok"]),
           "probe_set_err": res["err"]}
    if not res["ok"]:
        return out
    probe = res["amounts"]
    svals = {p: float(sur["ensemble"][idx, k])
             for k, p in enumerate(sur["phase_names"])}
    dev_fs, ph_fs = _dict_diff(full, svals)
    dev_ps, ph_ps = _dict_diff(probe, svals)
    dev_fp, ph_fp = _dict_diff(full, probe)
    out.update({
        "probe_amounts": probe,
        "full_amounts": full,
        "surrogate_amounts": svals,
        "max_abs_dev_full_vs_surrogate": dev_fs,
        "phase_max_dev_full_vs_surrogate": ph_fs,
        "max_abs_dev_probe_vs_surrogate": dev_ps,
        "phase_max_dev_probe_vs_surrogate": ph_ps,
        "max_abs_dev_full_vs_probe": dev_fp,
        "phase_max_dev_full_vs_probe": ph_fp,
    })
    if dev_fp <= 1e-3:
        verdict = ("model_error: probe-active and full phase sets agree at "
                   "this point")
    elif dev_ps < 0.5 * dev_fs:
        verdict = ("phase_set_effect: surrogate matches the probe-active set "
                   "much better than the full set")
    elif dev_ps < dev_fs:
        verdict = "mixed: phase-set and model contributions"
    else:
        verdict = "model_error_dominates"
    out["attribution"] = verdict
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def evaluate_target(target, bundles, failures):
    system = target["system"]
    log(f"[{target['label']}] system {system} | sweeping 600-1900 degC "
        f"(5 K) + refinement")
    cal = calphad_sweep(system, target["mole_frac"], target["label"],
                        log_prefix=f"[{target['label']}] ")
    failures.extend([{"alloy": target["label"], "system": system, **f}
                     for f in cal["failures"]])
    heads = {}
    for head in HEADS[system]:
        sur = surrogate_sweep(system, head, bundles, target["mole_frac"],
                              cal["T_C"])
        dev = fraction_deviation(cal, sur)
        heads[head] = {"surrogate": sur, "deviation": dev}
    return cal, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true",
                    help="retrain all models even if checkpoints exist")
    ap.add_argument("--skip-pe", action="store_true",
                    help="skip the optional Yamada P-E point evaluation")
    ap.add_argument("--out", type=str, default=OUT_JSON)
    args = ap.parse_args()

    t_start = time.time()
    log("=" * 100)
    log("EXPERIMENTAL ANCHOR EVALUATION (surrogate vs full-set CALPHAD vs DTA)")
    log("=" * 100)

    # ---------------- 1. anchors + mole conversion ---------------- #
    df = pd.read_csv(ANCHORS_CSV)
    specs = list(ANCHOR_SPECS) + ([] if args.skip_pe else [PE_SPEC])
    targets = [build_target(s, df) for s in specs]
    for t in targets:
        x = t["mole_frac"]
        log(f"[{t['label']}] wt% { {k: round(v, 4) for k, v in t['wt_pct'].items()} }"
            f" -> mole { {k: round(v, 6) for k, v in x.items()} } "
            f"(sum err {t['conversion_sanity']['mole_sum_error']:.2e})")
        if t["published"]:
            log(f"    published {t['published_qty']} = "
                f"{t['published']['value_C']} {t['published']['unit']} "
                f"(DTA, {t['published']['page_ref']})")
        t["design_box"] = design_box_check(t["system"], x)
        flag = "INSIDE" if t["design_box"]["in_design_box"] else "OUTSIDE"
        log(f"    design box: {flag}")

    # ---------------- 2. models ---------------- #
    systems = sorted({t["system"] for t in targets})
    summaries = ensure_models(systems, args.train_workers, args.epochs,
                              use_cache=not args.no_cache)
    for r in summaries:
        log(f"    {r['system']}/{r['head']}/s{r['seed']}: val MAE "
            f"{r['val_mae']:.4f} | test MAE {r['test_mae']:.4f}"
            f"{' | cached' if r.get('cached') else ''}")
    bundles = {}
    for s in systems:
        for h in HEADS[s]:
            for seed in SEEDS:
                bundles[(s, h, seed)] = load_model_bundle(s, h, seed)
    meta_t_range = {}
    for s in systems:
        tt = pd.read_csv(SYSTEMS[s]["dataset"], usecols=["T"])["T"]
        meta_t_range[s] = [float(tt.min()), float(tt.max())]

    # ---------------- 3. sweeps ---------------- #
    failures = []
    results = {}
    for t in targets:
        cal, heads = evaluate_target(t, bundles, failures)
        results[t["label"]] = {"target": t, "cal": cal, "heads": heads}

    # ---------------- 4. comparisons / JSON ---------------- #
    alloys = {}
    pe_block = None
    summary_rows = []
    for t in targets:
        label = t["label"]
        cal, heads = results[label]["cal"], results[label]["heads"]
        primary = PRIMARY_HEAD[t["system"]]
        t_range = meta_t_range[t["system"]]
        t_grid = np.asarray(cal["T_C"], dtype=float)
        entry = {
            "system": t["system"],
            "source_id": t["source_id"],
            "alloy_id": t["alloy_id"],
            "wt_pct": t["wt_pct"],
            "mole_frac": t["mole_frac"],
            "conversion_sanity": t["conversion_sanity"],
            "design_box": t["design_box"],
            "calphad": {k: v for k, v in cal["transitions"].items()},
            "calphad_n_points": len(cal["T_C"]),
            "calphad_n_failed": len(cal["failures"]),
            "calphad_failures": cal["failures"],
            "calphad_coverage": cal["coverage"],
            "sweep_range_flags": {
                "range_C": [float(t_grid.min()), float(t_grid.max())],
                "range_K": [float(t_grid.min()) + 273.15,
                            float(t_grid.max()) + 273.15],
                "dataset_T_range_K": t_range,
                "n_points_outside_dataset_T_range": int(
                    np.sum((t_grid + 273.15 < t_range[0])
                           | (t_grid + 273.15 > t_range[1]))),
                "transitions_inside_dataset_T_range": bool(
                    all(z is None or t_range[0] <= z + 273.15 <= t_range[1]
                        for z in (cal["transitions"]["T_first_liquid_C"],
                                  cal["transitions"]["T_last_solid_C"]))),
                "range_extended": cal["extensions"],
            },
            "surrogate": {},
            "deltas_K": {},
            "max_fraction_deviation": {},
            "sweep": {"T_C": [float(z) for z in cal["T_C"]],
                      "NP_LIQUID_calphad": [
                          float(z) if np.isfinite(z) else None
                          for z in (cal["F"][:, cal["phases"].index("LIQUID")]
                                    if "LIQUID" in cal["phases"]
                                    else np.full(len(cal["T_C"]), np.nan))],
                      "solid_calphad": [
                          float(np.nansum(np.delete(cal["F"][i],
                                                    cal["phases"].index("LIQUID")))
                                if "LIQUID" in cal["phases"] else np.nan)
                          if cal["ok"][i] else None
                          for i in range(len(cal["T_C"]))],
                      },
        }
        # surrogate blocks
        for head, hd in heads.items():
            sur = hd["surrogate"]
            per_seed = {str(s): {k: v for k, v in tr.items()}
                        for s, tr in sur["seed_transitions"].items()}
            entry["surrogate"][head] = {
                "phase_names": sur["phase_names"],
                "per_seed_transitions": per_seed,
                "ensemble_transitions": sur["ensemble_transitions"],
                "max_sum_dev_per_seed": sur["max_sum_dev_per_seed"],
                "max_sum_dev_ensemble": sur["max_sum_dev_ensemble"],
                "max_raw_sum_dev_before_projection":
                    sur["max_raw_sum_dev_before_projection"],
            }
            entry["max_fraction_deviation"][head] = hd["deviation"]
            cal_t = cal["transitions"]
            sur_t = sur["ensemble_transitions"]
            entry["deltas_K"][head] = {
                "sur_minus_cal": {
                    "T_first_liquid": delta(sur_t["T_first_liquid_C"],
                                            cal_t["T_first_liquid_C"]),
                    "T_last_solid": delta(sur_t["T_last_solid_C"],
                                          cal_t["T_last_solid_C"])},
                "per_seed_sur_minus_cal": {
                    str(s): {
                        "T_first_liquid": delta(tr["T_first_liquid_C"],
                                                cal_t["T_first_liquid_C"]),
                        "T_last_solid": delta(tr["T_last_solid_C"],
                                              cal_t["T_last_solid_C"])}
                    for s, tr in sur["seed_transitions"].items()},
                "per_seed_sur_minus_cal_sustained": {
                    str(s): {
                        "T_first_liquid": delta(
                            tr.get("T_first_liquid_sustained_C"),
                            cal_t.get("T_first_liquid_sustained_C")),
                        "T_last_solid": delta(tr.get("T_last_solid_C"),
                                              cal_t.get("T_last_solid_C"))}
                    for s, tr in sur["seed_transitions"].items()},
            }
            entry["sweep"][f"NP_LIQUID_sur_ens_{head}"] = [
                float(z) for z in
                sur["ensemble"][:, sur["phase_names"].index("LIQUID")]]
            # noise-robust onset of melting + tail-artifact flags
            d = entry["deltas_K"][head]
            d["sur_minus_cal_T_first_liquid_sustained"] = delta(
                sur_t.get("T_first_liquid_sustained_C"),
                cal_t.get("T_first_liquid_sustained_C"))
            g_min, g_max = float(t_grid.min()), float(t_grid.max())
            entry["surrogate"][head]["first_liquid_artifact"] = {
                "raw_first_liquid_C": sur_t.get("T_first_liquid_C"),
                "sustained_first_liquid_C":
                    sur_t.get("T_first_liquid_sustained_C"),
                "spurious_points_below_sustained":
                    sur_t.get("spurious_low_T_liquid_points", []),
                "is_spurious": bool(
                    sur_t.get("spurious_low_T_liquid_points")),
            }
            edge = []
            for s_, tr_ in sur["seed_transitions"].items():
                flags = []
                if tr_.get("T_first_liquid_C") is not None and \
                        tr_["T_first_liquid_C"] <= g_min:
                    flags.append("T_first_liquid at grid min")
                if tr_.get("T_last_solid_C") is not None and \
                        tr_["T_last_solid_C"] >= g_max:
                    flags.append("T_last_solid at grid max")
                if flags:
                    edge.append({"seed": s_, "flags": flags})
            entry["surrogate"][head]["edge_transition_flags"] = edge
            if t["published"]:
                key = TRANSITION_OF_PUBLISHED[t["published_qty"]][0]
                d = entry["deltas_K"][head]
                d["published_transition_key"] = key
                d["published_value_C"] = t["published"]["value_C"]
                d["cal_minus_exp"] = delta(cal_t[key],
                                           t["published"]["value_C"])
                d["sur_minus_exp"] = delta(sur_t[key],
                                           t["published"]["value_C"])
                if key == "T_first_liquid_C":
                    d["sur_minus_exp_sustained"] = delta(
                        sur_t.get("T_first_liquid_sustained_C"),
                        t["published"]["value_C"])
                    d["sur_minus_cal_sustained"] = delta(
                        sur_t.get("T_first_liquid_sustained_C"),
                        cal_t.get("T_first_liquid_sustained_C"))
        # threshold sensitivity + phase-set attribution (primary head only)
        p_sur = heads[primary]["surrogate"]
        liq_cal = (cal["F"][:, cal["phases"].index("LIQUID")]
                   if "LIQUID" in cal["phases"]
                   else np.full(len(cal["T_C"]), np.nan))
        liq_sur = p_sur["ensemble"][:, p_sur["phase_names"].index("LIQUID")]
        # truncate the surrogate liquid curve below its sustained crossing so
        # the threshold scan measures tail shape, not isolated 1e-4 patches
        sus_c = p_sur["ensemble_transitions"].get("T_first_liquid_sustained_C")
        liq_sur_scan = (np.where(t_grid < sus_c, 0.0, liq_sur)
                        if sus_c is not None else liq_sur)
        entry["threshold_sensitivity_primary_head"] = threshold_sensitivity(
            cal["T_C"], liq_cal, liq_sur_scan)
        dev_p = entry["max_fraction_deviation"][primary]
        entry["phase_set_attribution_primary_head"] = phase_set_spot_check(
            t["system"], t["mole_frac"], cal, p_sur,
            dev_p["max_abs_dNP_location"])
        out_loc = dev_p["max_abs_dNP_outside_melting_region_location"]
        same = (out_loc is not None and dev_p["max_abs_dNP_location"] is not None
                and out_loc["T_C"] == dev_p["max_abs_dNP_location"]["T_C"]
                and out_loc["phase"] == dev_p["max_abs_dNP_location"]["phase"])
        entry["phase_set_attribution_outside_melting_region"] = (
            entry["phase_set_attribution_primary_head"] if same or out_loc is None
            else phase_set_spot_check(t["system"], t["mole_frac"], cal, p_sur,
                                      out_loc))
        if t["published"]:
            key, why = TRANSITION_OF_PUBLISHED[t["published_qty"]]
            entry["published"] = dict(t["published"])
            entry["published"]["quantity"] = t["published_qty"]
            entry["published"]["corresponds_to_computed_transition"] = key
            entry["published"]["why"] = why
            cal_t = cal["transitions"]
            sur_t = heads[primary]["surrogate"]["ensemble_transitions"]
            d = entry["deltas_K"][primary]
            summary_rows.append({
                "alloy": label, "system": t["system"],
                "published_qty": t["published_qty"],
                "published_C": t["published"]["value_C"],
                "cal_first_liquid": cal_t["T_first_liquid_C"],
                "cal_last_solid": cal_t["T_last_solid_C"],
                "sur_first_liquid": sur_t["T_first_liquid_C"],
                "sur_first_liquid_sustained":
                    sur_t.get("T_first_liquid_sustained_C"),
                "sur_first_liquid_spurious": bool(
                    entry["surrogate"][primary]["first_liquid_artifact"]
                    ["is_spurious"]),
                "sur_last_solid": sur_t["T_last_solid_C"],
                "pub_transition": key,
                "cal_pub": trans_value(cal_t, key),
                "sur_pub": trans_value(sur_t, key),
                "d_sur_cal": d["sur_minus_cal"][
                    "T_first_liquid" if key == "T_first_liquid_C" else "T_last_solid"],
                "d_cal_exp": d["cal_minus_exp"],
                "d_sur_exp": d["sur_minus_exp"],
                "max_dnp": entry["max_fraction_deviation"][primary][
                    "max_abs_dNP_all_phases"],
            })
            alloys[label] = entry
        else:
            sur_t = heads[primary]["surrogate"]["ensemble_transitions"]
            solids = sur_t["solids_at_first_liquid"]
            cal_solids = cal["transitions"]["solids_at_first_liquid"]
            pe_block = {
                **entry,
                "note": ("Yamada peritectic-eutectic transition point "
                         "(15 wt% Cr - 10 wt% Ni): no published temperature; "
                         "evaluated qualitatively from the solid coexisting "
                         "with the first liquid."),
                "interpretation": {
                    "calphad_solids_at_first_liquid": cal_solids,
                    "surrogate_solids_at_first_liquid": solids,
                    "calphad_L_plus_delta": "BCC_A2" in cal_solids,
                    "surrogate_L_plus_delta": "BCC_A2" in solids,
                    "calphad_L_plus_gamma": "FCC_A1" in cal_solids,
                    "surrogate_L_plus_gamma": "FCC_A1" in solids,
                },
            }

    # ---------------- summary ---------------- #
    def agg(rows, field):
        vals = [abs(r[field]) for r in rows if r[field] is not None]
        if not vals:
            return {"mean": None, "max": None, "n": 0}
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
                "n": len(vals)}

    # all transitions (2 per anchor alloy) with the primary head
    all_trans_d = []
    excluded_spurious = []
    for label, entry in alloys.items():
        primary = PRIMARY_HEAD[entry["system"]]
        cal_t, sur_t = entry["calphad"], \
            entry["surrogate"][primary]["ensemble_transitions"]
        for key in ("T_first_liquid_C", "T_last_solid_C"):
            d = delta(sur_t[key], cal_t[key])
            if d is None:
                continue
            if key == "T_first_liquid_C" and entry["surrogate"][primary][
                    "first_liquid_artifact"]["is_spurious"]:
                excluded_spurious.append({
                    "alloy": label, "transition": key,
                    "raw_surrogate_C": sur_t[key],
                    "sustained_surrogate_C":
                        sur_t.get("T_first_liquid_sustained_C"),
                    "calphad_C": cal_t[key],
                    "spurious_points": entry["surrogate"][primary]
                        ["first_liquid_artifact"]["spurious_points_below_sustained"],
                    "reason": "surrogate liquid fraction exceeds "
                              f"{LIQUID_FRAC_THR} only in isolated low-T "
                              "patches, not the sustained melting branch"})
                continue
            all_trans_d.append({"alloy": label, "transition": key,
                                "abs_delta": abs(d), "delta": d})

    # how much of the delta is just the 1e-4 detection threshold on a tail?
    thr_agg = {}
    for thr in THR_SENSITIVITY:
        for key in ("T_first_liquid_C", "T_last_solid_C"):
            vals = [abs(v[key]["sur_minus_cal"]) for v in
                    (e["threshold_sensitivity_primary_head"][f"{thr:g}"]
                     for e in alloys.values())
                    if v[key]["sur_minus_cal"] is not None]
            thr_agg.setdefault(f"{thr:g}", {})[key] = (
                {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
                 "n": len(vals)} if vals else {"mean": None, "max": None,
                                               "n": 0})

    summary = {
        "n_anchor_alloys": len(alloys),
        "published_quantities": [r["published_qty"] for r in summary_rows],
        "sur_minus_cal_K_published": agg(summary_rows, "d_sur_cal"),
        "cal_minus_exp_K_published": agg(summary_rows, "d_cal_exp"),
        "sur_minus_exp_K_published": agg(summary_rows, "d_sur_exp"),
        "sur_minus_cal_K_all_transitions": (
            {"mean": float(np.mean([r["abs_delta"] for r in all_trans_d])),
             "max": float(np.max([r["abs_delta"] for r in all_trans_d])),
             "n": len(all_trans_d)} if all_trans_d else
            {"mean": None, "max": None, "n": 0}),
        "excluded_spurious_first_liquid": excluded_spurious,
        "sur_minus_cal_K_by_detection_threshold": thr_agg,
        "phase_set_attribution_at_max_deviation": {
            label: (entry["phase_set_attribution_primary_head"] or {})
                .get("attribution")
            for label, entry in alloys.items()},
        "phase_set_attribution_outside_melting_region": {
            label: (entry["phase_set_attribution_outside_melting_region"]
                    or {}).get("attribution")
            for label, entry in alloys.items()},
        "mean_abs_dNP_per_alloy_primary_head": {
            r["alloy"]: entry["max_fraction_deviation"][
                PRIMARY_HEAD[entry["system"]]]["mean_abs_dNP_all_phases"]
            for r in summary_rows
            for entry in [alloys[r["alloy"]]]},
        "max_fraction_deviation_per_alloy_primary_head": {
            r["alloy"]: r["max_dnp"] for r in summary_rows},
        "design_box_flags": {t["label"]: t["design_box"]["in_design_box"]
                             for t in targets},
        "calphad_failures_total": len(failures),
        "all_models_sum_to_one": all(
            e["surrogate"][h]["max_sum_dev_ensemble"] < 1e-6
            for e in list(alloys.values()) + ([pe_block] if pe_block else [])
            for h in e["surrogate"]),
        "elapsed_s": round(time.time() - t_start, 1),
    }

    out = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "anchors_csv": os.path.relpath(ANCHORS_CSV, ROOT),
            "atomic_weights": ATOMIC_WEIGHTS,
            "seeds": SEEDS,
            "heads": HEADS,
            "primary_head": PRIMARY_HEAD,
            "device": DEVICE,
            "hidden": list(HIDDEN),
            "calphad": {
                "tdb": {s: os.path.relpath(SYSTEMS[s]["tdb"], ROOT)
                        for s in systems},
                "phase_set": "ALL eligible phases of the pruned system TDB "
                             "(same set as databases/mc_fe_v2.062.tdb)",
                "n_eligible": {s: len(eligible_phases(SYSTEMS[s]["tdb"],
                                                      SYSTEMS[s]["comps"]))
                               for s in systems},
                "n_workers": N_CALPHAD_WORKERS,
                "mass_balance_tol": MASS_BALANCE_TOL,
            },
            "sweep": {
                "coarse_T_C": [COARSE_T_MIN_C, COARSE_T_MAX_C, COARSE_STEP_C],
                "refine": f"+-{REFINE_HALF_WIDTH_K} K at {REFINE_STEP_C} K "
                          "around each detected transition",
                "extend_range_if_uncovered": [EXTEND_DOWN_C, EXTEND_UP_C],
                "dataset_T_range_K": meta_t_range,
            },
            "transition_definitions": {
                "T_first_liquid_C": "smallest T with NP_LIQUID > "
                                    f"{LIQUID_FRAC_THR} (start of melting; "
                                    "published T_S/T_P)",
                "T_last_solid_C": "largest T with any solid > "
                                  f"{LIQUID_FRAC_THR} (liquidus; published T_L)",
                "T_first_liquid_sustained_C": "smallest T after which "
                                              f"NP_LIQUID > {LIQUID_FRAC_THR} "
                                              "at every higher T (noise-robust "
                                              "onset of melting)",
                "T_first_liquid_plus_BCC_A2_C": "smallest T with both liquid "
                                                "and delta ferrite above "
                                                f"{LIQUID_FRAC_THR} "
                                                "(peritectic-like melting)",
            },
            "threshold_sensitivity": {
                "thresholds": list(THR_SENSITIVITY),
                "note": "the surrogate never predicts exactly zero, so its "
                        "first-liquid/last-solid crossing depends on the "
                        "detection threshold; scanning thresholds separates "
                        "tail effects from real curve offsets",
                "truncation": "the surrogate liquid curve is set to zero "
                              "below its sustained first-liquid crossing so "
                              "the scan is not dominated by isolated ~1e-4 "
                              "low-T patches",
            },
            "melting_region_window_K": NEAR_TRANSITION_WINDOW_K,
            "fraction_deviation_notes": {
                "max_abs_dNP_outside_melting_region":
                    "max |NP_sur - NP_cal| over points outside "
                    f"[T_first_liquid - {NEAR_TRANSITION_WINDOW_K:g} K, "
                    f"T_last_solid + {NEAR_TRANSITION_WINDOW_K:g} K]",
                "phase_set_attribution":
                    "the point of max deviation is re-evaluated with the "
                    "probe-active phase set (the surrogate's training phase "
                    "set): if both phase sets give the same equilibrium, the "
                    "deviation is model error, not a reference-side phase-set "
                    "effect",
            },
            "training": summaries,
        },
        "alloys": alloys,
        "optional_pe_point": pe_block,
        "summary": summary,
        "failures": failures,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved {args.out}")

    # ---------------- compact table ---------------- #
    log("")
    hdr = (f"{'alloy':<18} {'published':<12} {'CAL T_S/T_L':>16} "
           f"{'SUR T_S/T_L':>16} {'dT(sur-cal)':>11} {'dT(cal-exp)':>11} "
           f"{'dT(sur-exp)':>11} {'max|dNP|':>9}")
    log(hdr)
    log("-" * len(hdr))
    for r in summary_rows:
        fmt = lambda z: "  --  " if z is None else f"{z:.0f}"
        fmt_d = lambda z: "   --   " if z is None else f"{z:+8.1f}"

        def _sfirst(rr):
            if rr.get("sur_first_liquid_spurious"):
                v = rr.get("sur_first_liquid_sustained")
                return "--   *" if v is None else f"{v:.0f}*"
            return fmt(rr["sur_first_liquid"])
        log(f"{r['alloy']:<18} "
            f"{r['published_qty'] + '=' + format(r['published_C'], '.0f'):<12} "
            f"{fmt(r['cal_first_liquid']) + '/' + fmt(r['cal_last_solid']):>16} "
            f"{_sfirst(r) + '/' + fmt(r['sur_last_solid']):>16} "
            f"{fmt_d(r['d_sur_cal'])} {fmt_d(r['d_cal_exp'])} "
            f"{fmt_d(r['d_sur_exp'])} {r['max_dnp']:9.4f}")
    log("* dT columns refer to the PUBLISHED quantity of each alloy "
        "(A: T_L = last solid; B: T_P = first liquid; C: T_S = first liquid; "
        "Yamada: T_L = last solid).")
    if any(r.get("sur_first_liquid_spurious") for r in summary_rows):
        log("* star: raw 1e-4 first-liquid crossing is a spurious low-T tail "
            "patch; the sustained (noise-robust) crossing is shown instead "
            "(both are in the JSON).")
    log(f"* dT(sur-...) uses the primary head "
        f"(fecrc: {PRIMARY_HEAD['fecrc']}, fecrni: {PRIMARY_HEAD['fecrni']}), "
        f"ensemble mean over seeds {SEEDS}.")
    log("")

    def _mm(d):
        if d["mean"] is None:
            return "-- / --"
        return f"{d['mean']:.1f} / {d['max']:.1f}"

    def _short_att(att):
        return {"model_error: probe-active and full phase sets agree at "
                "this point": "phase-set=eq",
                "phase_set_effect: surrogate matches the probe-active set "
                "much better than the full set": "phase-set",
                "mixed: phase-set and model contributions": "mixed",
                "model_error_dominates": "model"}.get(att, att)

    log(f"summary | |dT(sur-cal)| mean/max (published) = "
        f"{_mm(summary['sur_minus_cal_K_published'])} K | "
        f"|dT(cal-exp)| mean/max = "
        f"{_mm(summary['cal_minus_exp_K_published'])} K | "
        f"|dT(sur-exp)| mean/max = "
        f"{_mm(summary['sur_minus_exp_K_published'])} K")
    for t in targets:
        e = alloys.get(t["label"]) or pe_block
        primary = PRIMARY_HEAD[t["system"]]
        d = e["max_fraction_deviation"][primary]
        art = e["surrogate"][primary]["first_liquid_artifact"]
        att = (e["phase_set_attribution_primary_head"] or {}).get(
            "attribution", "--")
        att_s = _short_att(att)
        loc = d["max_abs_dNP_location"]
        outside = d["max_abs_dNP_outside_melting_region"]
        log(f"  {t['label']:<18} max|dNP| {d['max_abs_dNP_all_phases']:.4f} "
            f"(mean {d['mean_abs_dNP_all_phases']:.5f}, outside melting region "
            f"{('--' if outside is None else f'{outside:.4f}')} ) "
            f"| box {'IN' if t['design_box']['in_design_box'] else 'OUT'} "
            f"| dev@{att_s}{' | spurious low-T liquid' if art['is_spurious'] else ''}")
        if loc:
            log(f"  {'':<18}   max at {loc['T_C']:.0f} degC {loc['phase']}: "
                f"cal {loc['NP_calphad']:.4f} vs sur {loc['NP_surrogate']:.4f}")
        out_l = d["max_abs_dNP_outside_melting_region_location"]
        if out_l and (loc is None or (out_l["T_C"], out_l["phase"])
                      != (loc["T_C"], loc["phase"])):
            att2 = _short_att((e["phase_set_attribution_outside_melting_region"]
                               or {}).get("attribution", "--"))
            log(f"  {'':<18}   outside-region max at {out_l['T_C']:.0f} degC "
                f"{out_l['phase']}: cal {out_l['NP_calphad']:.4f} vs sur "
                f"{out_l['NP_surrogate']:.4f} | dev@{att2}")
        if art["is_spurious"]:
            sus = art["sustained_first_liquid_C"]
            log(f"  {'':<18}   raw first-liquid "
                f"{art['raw_first_liquid_C']:.0f} degC -> sustained "
                f"{('none' if sus is None else f'{sus:.0f} degC')}")
    log(f"  |dT(sur-cal)| by detection threshold (mean/max, all "
        f"transitions, primary head):")
    for thr in THR_SENSITIVITY:
        a = thr_agg[f"{thr:g}"]

        def _tk(block):
            return (f"{block['mean']:.1f}/{block['max']:.1f}"
                    if block["mean"] is not None else "--/--")
        log(f"      thr={thr:<7g} T_first_liquid "
            f"{_tk(a['T_first_liquid_C'])} K ({a['T_first_liquid_C']['n']}) | "
            f"T_last_solid {_tk(a['T_last_solid_C'])} K "
            f"({a['T_last_solid_C']['n']})")
    if excluded_spurious:
        log(f"  excluded {len(excluded_spurious)} spurious surrogate "
            f"first-liquid value(s) from the all-transition stats:")
        for xs in excluded_spurious:
            sus = xs["sustained_surrogate_C"]
            log(f"      {xs['alloy']}: raw {xs['raw_surrogate_C']:.0f} degC "
                f"-> sustained {('none' if sus is None else f'{sus:.0f} degC')} "
                f"(CALPHAD {xs['calphad_C']:.0f} degC)")
    if pe_block:
        ip = pe_block["interpretation"]
        log(f"  P-E point 15Cr-10Ni: CALPHAD solids at first liquid "
            f"{ip['calphad_solids_at_first_liquid']} (L+delta="
            f"{ip['calphad_L_plus_delta']}); surrogate "
            f"{ip['surrogate_solids_at_first_liquid']} (L+delta="
            f"{ip['surrogate_L_plus_delta']})")
    b = alloys.get("B")
    if b:
        primary = PRIMARY_HEAD["fecrc"]
        s = b["surrogate"][primary]["ensemble_transitions"]
        cal_ld = b["calphad"].get("T_first_liquid_plus_BCC_A2_C")
        sur_ld = s.get("T_first_liquid_plus_BCC_A2_C")
        log(f"  B first melting: CALPHAD solids "
            f"{b['calphad']['solids_at_first_liquid']} (assemblage below: "
            f"{b['calphad']['assemblage_below_first_liquid']}) | surrogate "
            f"{s['solids_at_first_liquid']} | L+delta at first liquid = "
            f"{'YES' if 'BCC_A2' in b['calphad']['solids_at_first_liquid'] else 'NO'}")
        log(f"  B first liquid+delta (peritectic-like) coexistence: CALPHAD "
            f"{cal_ld} degC, surrogate {sur_ld} degC "
            f"(solidus alone is L+gamma)")
    if failures:
        log(f"  FAILURES ({len(failures)}):")
        for f_ in failures[:20]:
            log(f"    {f_['alloy']} T={f_['T_C']:.1f} degC: {f_['err']}")
    else:
        log("  no CALPHAD sweep failures")
    log(f"  output: {args.out}")
    log(f"  elapsed {summary['elapsed_s']} s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
