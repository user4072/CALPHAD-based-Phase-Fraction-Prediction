"""MLP-4 head-variant comparison (the simplex-constraint study).

Head variants on identical MLP-4 backbones (raw Fe,Cr,Ni,T inputs):
  sigmoid    - independent per-phase sigmoid x scale (baseline)
  softmax    - single 4-logit head, sum = 1 by construction
  residue    - 3 sigmoid heads, 4th phase = 1 - sum (clip only at eval)
  renorm     - sigmoid model, renormalised at test time (inference projection)
  sig_norm   - sigmoid outputs, divided by their sum (train-time, diff'able)
  power_norm - sigmoid outputs ^ p, divided by their sum (p=2, diff'able)
  penalty    - sigmoid model + lambda * mean((sum - 1)^2) in the loss,
               evaluated RAW (no projection) to isolate the penalty's effect

Protocol: 3 paired seeds, cluster-stratified 64/16/20 split, Huber loss,
AdamW + cosine annealing, early stopping on val, test subset capped at 4000.
Residue-drop sensitivity check: --residue-sweep runs 1 seed per dropped phase.
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fe_surrogate.config import MODELS_DIR
from fe_surrogate.experiment import (load_data, cluster_split, evaluate,
                                     active_phases, renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
PATIENCE = 40
BATCH = 128
LR = 3e-4
HUBER_DELTA = 0.01
HIDDEN = 192


def sparsemax(z):
    """Euclidean projection of logits z onto the probability simplex.

    Produces exact zeros (unlike softmax), matching the sparse structure of
    equilibrium phase fractions. Differentiable almost everywhere.
    """
    zs = z.sort(descending=True, dim=-1).values
    csum = torch.cumsum(zs, dim=-1) - 1.0
    k = torch.arange(1, zs.size(-1) + 1, device=z.device)
    cond = zs - csum / k > 0
    k_eff = cond.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (torch.gather(csum, -1, k_eff - 1) / k_eff).squeeze(-1)
    return torch.clamp(z - tau.unsqueeze(-1), min=0.0)


class MLP4(nn.Module):
    """3 hidden layers of width 192, LayerNorm + SiLU + Dropout."""

    def __init__(self, n_phases=4, head="sigmoid", residue_drop=None, power=2.0,
                 hidden=(192, 192, 192)):
        super().__init__()
        self.head = head
        self.n_phases = n_phases
        self.residue_drop = residue_drop
        self.power = power
        layers = [nn.Linear(4, hidden[0]), nn.LayerNorm(hidden[0]), nn.SiLU(), nn.Dropout(0.1)]
        prev = hidden[0]
        for h in hidden[1:]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(0.1)]
            prev = h
        self.net = nn.Sequential(*layers)
        last_h = prev
        n_out = n_phases - 1 if head == "residue" else n_phases
        if head in ("softmax", "sparsemax"):
            self.head_out = nn.Linear(last_h, n_phases)
        else:
            self.heads_out = nn.ModuleList(
                [nn.Sequential(nn.Linear(last_h, last_h), nn.SiLU(), nn.Linear(last_h, 1))
                 for _ in range(n_out)])
            self.output_scales = nn.Parameter(torch.ones(n_out))

    def forward(self, x):
        h = self.net(x)
        if self.head == "softmax":
            return torch.softmax(self.head_out(h), dim=-1)
        if self.head == "sparsemax":
            return sparsemax(self.head_out(h) * 4.0)
        out = []
        for i, head in enumerate(self.heads_out):
            scale = torch.sigmoid(self.output_scales[i])
            out.append(head(h).sigmoid() * scale)
        if self.head == "residue":
            pred = torch.cat(out, dim=-1)
            res = 1.0 - pred.sum(dim=-1, keepdim=True)
            return torch.cat([pred[:, :self.residue_drop],
                              res,
                              pred[:, self.residue_drop:]], dim=-1)
        pred = torch.cat(out, dim=-1)
        if self.head == "sig_norm":
            return pred / pred.sum(dim=-1, keepdim=True)
        if self.head == "power_norm":
            p = pred.pow(self.power)
            return p / p.sum(dim=-1, keepdim=True)
        return pred


def train_mlp(model, X, Y, tr, va, seed, epochs=EPOCHS, penalty_lambda=0.0,
              loss="huber"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr])
    Xva = scaler.transform(X[va])
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(Y[tr], dtype=torch.float32, device=DEVICE)
    xva = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(Y[va], dtype=torch.float32, device=DEVICE)
    scaler_te = scaler
    loader_tr = [(xtr[i:i + BATCH], ytr[i:i + BATCH]) for i in range(0, len(tr), BATCH)]

    opt = AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    if loss == "mse":
        loss_fn = nn.MSELoss()
    elif loss == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    best_mae = float("inf")
    best_state = None
    patience = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader_tr:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if penalty_lambda > 0:
                dev = pred.sum(dim=-1) - 1.0
                loss = loss + penalty_lambda * dev.pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = model(xva).cpu().numpy()
        mae = np.mean(np.abs(p - yva.cpu().numpy()))
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_mae, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+",
                    default=["sigmoid", "softmax", "residue", "renorm"])
    ap.add_argument("--loss", choices=["huber", "mse", "mae"], default="huber")
    ap.add_argument("--hidden", type=int, nargs="+", default=[192, 192, 192],
                    help="hidden layer widths, e.g. 64 64 64")
    ap.add_argument("--penalty-lambdas", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--power", type=float, default=2.0, help="exponent for power_norm")
    ap.add_argument("--residue-drop", type=int, default=None,
                    help="phase index predicted as 1-sum (default: dominant phase "
                         "by mean amount)")
    ap.add_argument("--residue-sweep", action="store_true",
                    help="1-seed sensitivity check over each dropped phase")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=str, default=os.path.join(MODELS_DIR, "results_heads.json"))
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="fecrni")
    args = ap.parse_args()

    cfg = SYSTEMS[args.system]
    X, Y, df = load_data(args.system)
    _, phase_names = active_phases(args.system)
    n_phases = len(phase_names)
    box = df["in_stainless_box"].values.astype(bool)
    print(f"System {args.system}: {len(X)} rows, {n_phases} active phases "
          f"{phase_names} | device {DEVICE}")

    results = {}
    rng = np.random.default_rng(0)
    if args.residue_drop is None:
        args.residue_drop = int(np.argmax(Y.mean(axis=0)))
        print(f"residue_drop = {args.residue_drop} ({phase_names[args.residue_drop]}, "
              f"dominant phase)")

    if args.residue_sweep:
        print("== residue-drop sensitivity (seed 42, one architecture) ==")
        tr, va, te = cluster_split(X, Y, 42)
        te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
        for drop in range(n_phases):
            model = MLP4(n_phases=n_phases, head="residue", residue_drop=drop,
                         hidden=tuple(args.hidden)).to(DEVICE)
            model, _, scaler = train_mlp(model, X, Y, tr, va, 42, args.epochs,
                                         loss=args.loss)
            model.eval()
            Xte = scaler.transform(X[te_idx])
            with torch.no_grad():
                pred = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            res = evaluate(Y[te_idx], pred, box[te_idx])
            print(f"  drop {phase_names[drop]:8s}: mean_MAE={res['mean_mae']:.4f} "
                  f"| consistency={res['consistency']:.4f} | per-phase={np.round(res['mae'], 4)}")
            results[f"sweep_drop_{phase_names[drop]}"] = res
        sweep_out = os.path.join(MODELS_DIR, f"residue_sweep_{args.system}.json")
        with open(sweep_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {sweep_out}")
        return

    for seed in SEEDS:
        tr, va, te = cluster_split(X, Y, seed)
        te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
        runs = []
        for head in args.heads:
            if head == "penalty":
                runs += [(f"mlp_penalty_{lam:g}", "sigmoid", lam) for lam in args.penalty_lambdas]
            else:
                runs.append((f"mlp_{head}", "sigmoid" if head == "renorm" else head, 0.0))
        for tag, model_head, lam in runs:
            t0 = time.time()
            model = MLP4(n_phases=n_phases, head=model_head, residue_drop=args.residue_drop,
                         power=args.power, hidden=tuple(args.hidden)).to(DEVICE)
            model, _, scaler = train_mlp(model, X, Y, tr, va, seed, args.epochs,
                                         penalty_lambda=lam, loss=args.loss)
            model.eval()
            Xte = scaler.transform(X[te_idx])
            with torch.no_grad():
                pred = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            res = evaluate(Y[te_idx], pred, box[te_idx],
                           renormalise=(model_head == "sigmoid" and tag.endswith("renorm")))
            pred_final = renorm(pred) if (model_head == "sigmoid" and tag.endswith("renorm")) else pred
            tag_safe = tag.replace(" ", "_")
            hid = "x".join(str(h) for h in args.hidden)
            np.savez(os.path.join(MODELS_DIR, f"pred_{args.system}_{tag_safe}_{hid}_{args.loss}_s{seed}.npz"),
                     y_true=Y[te_idx], y_pred=pred_final, box=box[te_idx],
                     te_idx=te_idx, features=X[te_idx])
            res["time_s"] = round(time.time() - t0, 1)
            results[f"{tag}_s{seed}"] = res
            print(f"[{tag}_s{seed}] mean_MAE={res['mean_mae']:.4f} | R2={np.round(res['r2'], 3)} "
                  f"| consistency={res['consistency']:.4f} | box={res['stainless_box_mae']:.4f} | {res['time_s']}s")

    out = args.out
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()