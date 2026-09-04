"""Train GATv2, flat MLP-80 and XGBoost on the Fe-Cr-Ni dataset.

Comparison protocol carried over from the pycalphad1 revision work:
  - capacity-matched arms (GATv2 ~525K, MLP-80 ~155K, XGBoost trees)
  - paired statistics across shared seeds (3 seeds)
  - stratified split by phase-assemblage cluster (64/16/20)
  - no test-set touch during model selection (early stopping on val only)
  - Huber loss delta=0.01, AdamW, cosine annealing, early stop patience 40
  - report per-phase MAE/R2, thermodynamic consistency (mean |sum(pred)-1|),
    and stainless-box error
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

from fe_surrogate.config import DATA_DIR, MODELS_DIR, PHASE_COLS, PHASE_NAMES
from fe_surrogate.graphs import build_graphs, build_flat_80
from fe_surrogate.models import GATv2Net, MLP80

SEEDS = [42, 123, 2024]
EPOCHS = 300
PATIENCE = 40
BATCH = 128
LR = 3e-4
HUBER_DELTA = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_fecrni.csv"))
    X = df[["Fe", "Cr", "Ni", "T"]].values.astype(np.float64)
    Y = df[PHASE_COLS].values.astype(np.float32)
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
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    te = np.concatenate(te)
    return tr, va, te


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    cons = np.mean(np.abs(y_pred.sum(axis=1) - 1.0))
    return mae, r2, cons


def train_nn(model, X, Y, tr, va, seed, epochs=EPOCHS, flat=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if flat:
        ftr = torch.tensor(build_flat_80(X[tr]), dtype=torch.float32, device=DEVICE)
        fva = torch.tensor(build_flat_80(X[va]), dtype=torch.float32, device=DEVICE)
        ytr = torch.tensor(Y[tr], dtype=torch.float32, device=DEVICE)
        yva = torch.tensor(Y[va], dtype=torch.float32, device=DEVICE)
        n = len(tr)
        loader_tr = [(ftr[i:i + BATCH], ytr[i:i + BATCH]) for i in range(0, n, BATCH)]
        loader_va = [(fva, yva)]
    else:
        graphs_tr = build_graphs(X[tr], Y[tr])
        graphs_va = build_graphs(X[va], Y[va])
        loader_tr = PyGDataLoader(graphs_tr, batch_size=BATCH, shuffle=True)
        loader_va = PyGDataLoader(graphs_va, batch_size=1024, shuffle=False)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    best_mae = float("inf")
    best_state = None
    patience = 0
    for ep in range(epochs):
        model.train()
        for batch in loader_tr:
            opt.zero_grad()
            if flat:
                x, y = batch
                out = model(x)
            else:
                g = batch.to(DEVICE)
                out = model(g)
                y = g.y
            loss = loss_fn(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader_va:
                if flat:
                    x, y = batch
                    preds.append(model(x).cpu().numpy())
                else:
                    g = batch.to(DEVICE)
                    preds.append(model(g).cpu().numpy())
                    y = g.y
                trues.append(y.cpu().numpy())
        p = np.concatenate(preds)
        t = np.concatenate(trues)
        mae = np.mean(np.abs(p - t))
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_mae


def train_xgb(X, Y, tr, va, seed):
    preds = np.zeros((len(va), Y.shape[1]))
    for j in range(Y.shape[1]):
        m = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=50, random_state=seed,
                         eval_metric="mae")
        m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])], verbose=False)
        preds[:, j] = m.predict(X[va])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gatv2", "mlp", "xgb"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    epochs = args.epochs

    X, Y, df = load_data()
    print(f"Data: {len(X)} rows | device {DEVICE} | epochs {epochs}")

    results = {"dataset": len(X), "phases": PHASE_NAMES}
    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        rng = np.random.default_rng(seed)
        subset = rng.choice(len(te), min(4000, len(te)), replace=False)
        te_idx = te[subset]
        for name in args.models:
            tag = f"{name}_s{seed}"
            t0 = time.time()
            if name == "gatv2":
                model = GATv2Net().to(DEVICE)
                model, _ = train_nn(model, X, Y, tr, va, seed)
                model.eval()
                graphs_te = build_graphs(X[te_idx], None)
                loader = PyGDataLoader(graphs_te, batch_size=1024, shuffle=False)
                preds = []
                with torch.no_grad():
                    for g in loader:
                        preds.append(model(g.to(DEVICE)).cpu().numpy())
                pred = np.concatenate(preds)
                os.makedirs(os.path.join(MODELS_DIR, name), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, name, f"seed{seed}.pt"))
            elif name == "mlp":
                model = MLP80().to(DEVICE)
                model, _ = train_nn(model, X, Y, tr, va, seed, epochs, flat=True)
                model.eval()
                f80 = build_flat_80(X[te_idx])
                pred = model(torch.tensor(f80, device=DEVICE)).cpu().detach().numpy()
                os.makedirs(os.path.join(MODELS_DIR, name), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, name, f"seed{seed}.pt"))
            elif name == "xgb":
                pred = np.zeros((len(te_idx), Y.shape[1]))
                for j in range(Y.shape[1]):
                    m = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     early_stopping_rounds=50, random_state=seed,
                                     eval_metric="mae")
                    m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])], verbose=False)
                    pred[:, j] = m.predict(X[te_idx])
            else:
                raise ValueError(name)
            mae, r2, cons = metrics(Y[te_idx], pred)
            box = df["in_stainless_box"].values[te_idx] == 1
            box_mae = np.mean(np.abs(pred[box] - Y[te_idx][box])) if box.sum() > 0 else float("nan")
            results[tag] = {
                "time_s": round(time.time() - t0, 1),
                "mae": mae.tolist(),
                "r2": r2.tolist(),
                "mean_mae": float(np.mean(mae)),
                "consistency": float(cons),
                "stainless_box_mae": float(box_mae),
            }
            print(f"[{tag}] mean_MAE={np.mean(mae):.4f} | R2={r2} | "
                  f"consistency={cons:.4f} | box_MAE={box_mae:.4f} | {time.time()-t0:.0f}s")

    out = os.path.join(MODELS_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()