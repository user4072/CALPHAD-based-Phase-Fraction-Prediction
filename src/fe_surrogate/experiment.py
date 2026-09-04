"""Shared experiment infrastructure for the ternary surrogate studies."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from fe_surrogate.config import DATA_DIR
from fe_surrogate.systems import SYSTEMS

SEEDS = [42, 123, 2024]


def active_phases(system):
    """Phase columns/names used as model targets.

    Targets are ALL probe-active phases (observed stable in the full-phase
    probe), so the target vector lies exactly on the simplex:
    sum_i y_i = 1 - (mass-balance tolerance ~1e-8) by construction of the
    equilibrium solver. No occurrence threshold is applied.
    """
    cfg = SYSTEMS[system]
    path = os.path.join(os.path.dirname(cfg["dataset"]), f"{system}_probe.json")
    with open(path) as f:
        probe = json.load(f)
    cols = sorted(f"NP_{p}" for p in probe["active_counts"].keys())
    names = [c[len("NP_"):] for c in cols]
    return cols, names


def load_data(system="fecrni"):
    cfg = SYSTEMS[system]
    df = pd.read_csv(cfg["dataset"])
    phase_cols, _ = active_phases(system)
    X = df[cfg["comps"] + ["T"]].values.astype(np.float64)
    Y = df[phase_cols].values.astype(np.float32)
    return X, Y, df


def cluster_split(X, Y, seed):
    """64/16/20 split stratified by phase-assemblage cluster (k-means, k=6)."""
    coords = np.column_stack([X[:, :3], (X[:, 3] - 700) / 1300])
    km = KMeans(n_clusters=6, random_state=seed, n_init=10).fit(coords)
    idx = np.arange(len(X))
    tr, va, te = [], [], []
    for c in range(6):
        ci = idx[km.labels_ == c]
        rng = np.random.default_rng(seed + c)
        rng.shuffle(ci)
        n = len(ci)
        tr.append(ci[: int(0.64 * n)])
        va.append(ci[int(0.64 * n): int(0.80 * n)])
        te.append(ci[int(0.80 * n):])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    cons = np.mean(np.abs(y_pred.sum(axis=1) - 1.0))
    mad = np.mean(np.abs(y_true - y_true.mean(axis=0)), axis=0)
    nmae = mae / (mad + 1e-12)
    return mae, r2, cons, rmse, nmae


def phase_classification(y_true, y_pred, threshold=1e-3):
    """Per-phase presence/absence classification (phase amount > threshold)."""
    f1, bal_acc = [], []
    for k in range(y_true.shape[1]):
        tp = np.sum((y_true[:, k] > threshold) & (y_pred[:, k] > threshold))
        fp = np.sum((y_true[:, k] <= threshold) & (y_pred[:, k] > threshold))
        fn = np.sum((y_true[:, k] > threshold) & (y_pred[:, k] <= threshold))
        tn = np.sum((y_true[:, k] <= threshold) & (y_pred[:, k] <= threshold))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1.append(2 * prec * rec / (prec + rec + 1e-12))
        bal_acc.append(0.5 * (tp / (tp + fn + 1e-12) + tn / (tn + fp + 1e-12)))
    return np.array(f1), np.array(bal_acc)


def renorm(pred):
    """Inference-time projection: clip to [0,1], then renormalise to sum 1.

    This is the projection applied to all "renorm" variants (MLP sigmoid,
    RF, XGB). It differs from the Euclidean projection onto the probability
    simplex in that it does not reorder/soft-threshold negative entries;
    in practice tree outputs exceed [0,1] rarely, so both agree on almost
    all rows. The alternative (Euclidean simplex projection) is not used.
    """
    p = np.clip(pred, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    return p / np.where(s > 1e-12, s, 1.0)


def evaluate(y_true, y_pred, box_mask, renormalise=False):
    if renormalise:
        y_pred = renorm(y_pred)
    mae, r2, cons, rmse, nmae = metrics(y_true, y_pred)
    f1, bal_acc = phase_classification(y_true, y_pred)
    box_mae = np.mean(np.abs(y_pred[box_mask] - y_true[box_mask])) if box_mask.sum() > 0 else float("nan")
    eps = 1e-12
    h_true = -np.sum(np.where(y_true > eps, y_true * np.log(np.clip(y_true, eps, None)), 0.0), axis=1)
    h_max = np.log(y_true.shape[1])
    w = 1.0 + h_true / (h_max + eps)
    entropy_mae = np.mean(w * np.mean(np.abs(y_pred - y_true), axis=1))
    # Boundary rows = at least half of the max entropy (genuinely multi-phase);
    # bulk rows = the rest (single-phase dominated). Robust for any phase count.
    hi = h_true >= 0.5 * h_max
    boundary_mae = np.mean(np.abs(y_pred[hi] - y_true[hi])) if hi.sum() > 0 else float("nan")
    bulk_mae = np.mean(np.abs(y_pred[~hi] - y_true[~hi])) if (~hi).sum() > 0 else float("nan")
    return {
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "rmse": rmse.tolist(),
        "nmae": nmae.tolist(),
        "mean_mae": float(np.mean(mae)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_nmae": float(np.mean(nmae)),
        "f1": f1.tolist(),
        "balanced_acc": bal_acc.tolist(),
        "mean_f1": float(np.mean(f1)),
        "consistency": float(cons),
        "entropy_mae": float(entropy_mae),
        "boundary_mae": float(boundary_mae),
        "bulk_mae": float(bulk_mae),
        "stainless_box_mae": float(box_mae),
    }