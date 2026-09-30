"""Rare-phase detection remedies for the sigmoid MLP surrogate.

The plain MLP (sigmoid head, Huber, 192x192x192) has fine aggregate MAE but
cannot rank rare phases (Fe-Cr-Mn ALPHA_MN AUPRC ~ random). Two remedies,
same backbone / cluster split / early stopping as train_mlp.py:

  weighted - per-phase inverse-prevalence weighted Huber:
             w_k = clip(1/prevalence_k, 1, 100), prevalence_k = fraction of
             TRAIN rows with y_k > 1e-3, then normalised to mean 1 over phases.
  gated    - two heads on the same trunk:
             * fraction head: per-phase sigmoid outputs, weighted Huber
             * presence head: per-phase logits, class-weighted BCE
               (pos_weight_k = clip(n_neg/n_pos, 1, 100), label y > 1e-3)
             output = sigmoid(frac) * sigmoid(presence);
             projection (clip + row renorm) applied at evaluation.

Evaluation protocol is copied verbatim from
analysis_revision/revision_analyses.py::detection_recomputed so the numbers
are directly comparable with analysis_revision/revision_analyses.json
(positive label y_true > 1e-3, average_precision_score per phase,
zero-positive phases excluded, macro over phases, mean over SEEDS).

Outputs (new files only):
  models/pred_{system}_{variant}_{arch}_s{seed}.npz
  models/results_remedy.json   (per-run + aggregated + embedded baselines)

Usage:
  py -3.12 -X utf8 train_remedy.py --variants weighted --systems fecrmn fecrni fecrmo fecrv femnni
  py -3.12 -X utf8 train_remedy.py --variants gated --systems fecrmn fecrni
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fe_surrogate.config import MODELS_DIR
from fe_surrogate.experiment import (load_data, cluster_split, evaluate,
                                     active_phases, phase_classification,
                                     renorm, SEEDS)
from fe_surrogate.systems import SYSTEMS

from train_mlp import MLP4, EPOCHS, PATIENCE, BATCH, LR, HUBER_DELTA, HIDDEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    EPOCHS = 120
HIDDEN = (HIDDEN, HIDDEN, HIDDEN) if isinstance(HIDDEN, int) else tuple(HIDDEN)
ARCH = "x".join(str(h) for h in HIDDEN)
POS_THRESH = 1e-3
RESULTS_PATH = os.path.join(MODELS_DIR, "results_remedy.json")
REVISION_JSON = os.path.join(HERE, "analysis_revision", "revision_analyses.json")
SYSTEM_ORDER = ["fecrmn", "fecrni", "fecrmo", "fecrv", "femnni"]
BASELINE_TAGS = ["mlp_renorm", "rf_renorm", "xgb_renorm", "knn_renorm"]


class GatedMLP(nn.Module):
    """MLP4-style trunk with two per-phase heads (fraction + presence)."""

    def __init__(self, n_phases=4, hidden=HIDDEN, n_in=4):
        super().__init__()
        self.n_phases = n_phases
        layers = [nn.Linear(n_in, hidden[0]), nn.LayerNorm(hidden[0]), nn.SiLU(),
                  nn.Dropout(0.1)]
        prev = hidden[0]
        for h in hidden[1:]:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU(),
                       nn.Dropout(0.1)]
            prev = h
        self.net = nn.Sequential(*layers)
        head = lambda: nn.Sequential(nn.Linear(prev, prev), nn.SiLU(),
                                     nn.Linear(prev, 1))
        self.frac_heads = nn.ModuleList([head() for _ in range(n_phases)])
        self.pres_heads = nn.ModuleList([head() for _ in range(n_phases)])

    def forward(self, x):
        h = self.net(x)
        frac = torch.cat([hd(h) for hd in self.frac_heads], dim=-1)
        pres = torch.cat([hd(h) for hd in self.pres_heads], dim=-1)
        return frac, pres

    def output(self, x):
        frac, pres = self.forward(x)
        return torch.sigmoid(frac) * torch.sigmoid(pres)


def phase_weights(Y, tr):
    """w_k = clip(1/prevalence_k, 1, 100), normalised to mean 1 over phases."""
    prev = (Y[tr] > POS_THRESH).mean(axis=0).astype(np.float64)
    prev = np.maximum(prev, 1e-6)
    w = np.clip(1.0 / prev, 1.0, 100.0)
    return (w / w.mean()).astype(np.float32), prev


def presence_pos_weight(Y, tr):
    pos = (Y[tr] > POS_THRESH).sum(axis=0).astype(np.float64)
    neg = len(tr) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        pw = np.where(pos > 0, neg / np.maximum(pos, 1.0), 100.0)
    return np.clip(pw, 1.0, 100.0).astype(np.float32)


def train_variant(model, variant, X, Y, tr, va, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(X[tr])
    xtr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(Y[tr], dtype=torch.float32, device=DEVICE)
    xva = torch.tensor(scaler.transform(X[va]), dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(Y[va], dtype=torch.float32, device=DEVICE)
    loader = [(xtr[i:i + BATCH], ytr[i:i + BATCH])
              for i in range(0, len(tr), BATCH)]

    w_ph, prev = phase_weights(Y, tr)
    w_t = torch.tensor(w_ph, device=DEVICE)
    huber = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")
    bce = None
    if variant == "gated":
        pw_t = torch.tensor(presence_pos_weight(Y, tr), device=DEVICE)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw_t, reduction="none")

    opt = AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best_mae, best_state, patience, epochs_run = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            if variant == "weighted":
                pred = model(xb)
                loss = (huber(pred, yb) * w_t).mean()
            else:
                frac, pres = model(xb)
                loss = ((huber(torch.sigmoid(frac), yb) * w_t).mean()
                        + bce(pres, (yb > POS_THRESH).float()).mean())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        epochs_run = ep + 1
        model.eval()
        with torch.no_grad():
            if variant == "weighted":
                p = model(xva)
            else:
                p = model.output(xva)
            mae = float(torch.mean(torch.abs(p - yva)))
        if mae < best_mae:
            best_mae, best_state, patience = mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, scaler, best_mae, epochs_run, prev, w_ph


def predict(model, variant, scaler, X, idx):
    model.eval()
    xt = torch.tensor(scaler.transform(X[idx]), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        raw = (model(xt) if variant == "weighted" else model.output(xt))
    return raw.cpu().numpy().astype(np.float32)


def detection_metrics(y_t, y_p, names):
    """Exact protocol of revision_analyses.detection_recomputed."""
    f1, _ = phase_classification(y_t, y_p)
    pos = y_t.sum(axis=0) > 0
    per = {n: float(average_precision_score((y_t[:, i] > POS_THRESH).astype(int),
                                            y_p[:, i]))
           for i, n in enumerate(names) if pos[i]}
    return {
        "macro_auprc": float(np.mean(list(per.values()))) if per else None,
        "per_phase_auprc": per,
        "macro_f1_with_zero_pos": float(f1.mean()),
        "macro_f1_zero_pos_excluded": (float(f1[pos].mean())
                                       if pos.any() else None),
        "per_phase_f1": {n: float(f1[i]) for i, n in enumerate(names)},
        "zero_positive_phases": [names[i] for i in range(len(names)) if not pos[i]],
    }


BASELINE_NPZ_TAGS = {
    "mlp": "mlp_renorm_192x192x192_huber",
    "rf": "rf_renorm_std",
    "xgb": "xgb_renorm_std",
    "knn": "knn_renorm_std",
}


def detection_baseline_from_npz(system, names):
    """Same protocol as analysis_revision/detection_fecrnic.py, built from the
    stored prediction npz files. Used for systems absent from
    revision_analyses.json (fecrc and anything newer)."""
    out = {}
    for _m, tag in BASELINE_NPZ_TAGS.items():
        per_phase = {n: [] for n in names}
        macros = []
        n_seeds = 0
        for seed in SEEDS:
            p = os.path.join(MODELS_DIR, f"pred_{system}_{tag}_s{seed}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            det_m = detection_metrics(d["y_true"], d["y_pred"], names)
            if det_m["macro_auprc"] is not None:
                macros.append(det_m["macro_auprc"])
            for n, v in det_m["per_phase_auprc"].items():
                per_phase[n].append(v)
            n_seeds += 1
        if not n_seeds:
            continue
        out[f"{tag}"] = {
            "n_seeds": n_seeds,
            "macro_AUPRC": float(np.mean(macros)) if macros else None,
            "per_phase_AUPRC": {n: (float(np.mean(v)) if v else None)
                                for n, v in per_phase.items()},
        }
    return out


def load_baselines(systems):
    with open(REVISION_JSON, encoding="utf-8") as f:
        det = json.load(f)["detection_recomputed"]
    mae = {}
    for system in SYSTEMS:
        row = {}
        for model, fname, tag in [
                ("mlp", f"results_heads_{system}.json", "mlp_renorm"),
                ("rf", f"results_baselines_{system}.json", "rf_renorm"),
                ("xgb", f"results_baselines_{system}.json", "xgb_renorm"),
                ("knn", f"results_baselines_{system}.json", "knn_renorm")]:
            path = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(path):
                row[model] = None
                continue
            src = json.load(open(path, encoding="utf-8"))
            vals = [src[f"{tag}_s{s}"]["mean_mae"] for s in SEEDS
                    if f"{tag}_s{s}" in src]
            row[model] = float(np.mean(vals)) if vals else None
        mae[system] = row
    for system in systems:
        if system in det and det[system]:
            continue
        _, names = active_phases(system)
        built = detection_baseline_from_npz(system, names)
        if built:
            det[system] = built
            print(f"[{system}] detection baselines recomputed from npz: "
                  f"{sorted(built)}", flush=True)
        else:
            print(f"[{system}] no npz baselines found; comparison left empty",
                  flush=True)
    return det, mae


def aggregate(runs, det, base_mae):
    agg, comp = {}, {}
    groups = {}
    for r in runs.values():
        groups.setdefault((r["system"], r["variant"]), []).append(r)
    for (system, variant), rs in groups.items():
        names = [n for n in rs[0]["per_phase_auprc"]]
        per = {}
        for n in rs[0]["per_phase_auprc"]:
            vals = [r["per_phase_auprc"][n] for r in rs if n in r["per_phase_auprc"]]
            per[n] = float(np.mean(vals))
        sys_row = agg.setdefault(system, {})
        sys_row[variant] = {
            "n_seeds": len(rs),
            "macro_auprc": float(np.mean([r["macro_auprc"] for r in rs])),
            "macro_auprc_std": float(np.std([r["macro_auprc"] for r in rs])),
            "per_phase_auprc": per,
            "mae": float(np.mean([r["mae"] for r in rs])),
            "mae_std": float(np.std([r["mae"] for r in rs])),
            "max_sum_err": float(np.max([r["max_sum_err_proj"] for r in rs])),
            "max_sum_err_raw": float(np.max([r["max_sum_err_raw"] for r in rs])),
            "macro_f1_with_zero_pos": float(np.mean(
                [r["macro_f1_with_zero_pos"] for r in rs])),
            "macro_f1_zero_pos_excluded": (float(np.mean(
                [r["macro_f1_zero_pos_excluded"] for r in rs
                 if r["macro_f1_zero_pos_excluded"] is not None]))
                if any(r["macro_f1_zero_pos_excluded"] is not None for r in rs)
                else None),
            "per_phase_f1": {n: float(np.mean(
                [r["per_phase_f1"][n] for r in rs]))
                for n in rs[0]["per_phase_f1"]},
            "mean_epochs": float(np.mean([r["epochs_run"] for r in rs])),
            "mean_time_s": float(np.mean([r["time_s"] for r in rs])),
        }
    # comparison vs stored baselines
    for system, sys_row in agg.items():
        b = det.get(system, {})
        b_mlp = b.get("mlp_renorm_192x192x192_huber", {})
        base_auprc = {}
        for m, tag in [("mlp", "mlp_renorm_192x192x192_huber"),
                       ("rf", "rf_renorm_std"), ("xgb", "xgb_renorm_std"),
                       ("knn", "knn_renorm_std")]:
            pp = b.get(tag, {}).get("per_phase_AUPRC", {})
            vals = [v for v in pp.values() if v is not None]
            base_auprc[m] = float(np.mean(vals)) if vals else None
        rare = [n for n, v in b_mlp.get("per_phase_AUPRC", {}).items()
                if v is not None and v < 0.5]
        row = {"baseline_macro_auprc": base_auprc,
               "baseline_mae": base_mae.get(system),
               "rare_phases_mlp_auprc_below_0.5": rare,
               "remedy": {}}
        for variant, vr in sys_row.items():
            still = [n for n in rare
                     if vr["per_phase_auprc"].get(n, 0.0) < 0.5]
            row["remedy"][variant] = {
                "macro_auprc": vr["macro_auprc"],
                "mae": vr["mae"],
                "mae_ratio_vs_mlp": (vr["mae"] / base_mae[system]["mlp"]
                                     if base_mae.get(system, {}).get("mlp") else None),
                "n_rare_phases": len(rare),
                "rare_phases_still_below_0.5": still,
                "all_rare_phases_ge_0.5": len(still) == 0,
            }
        comp[system] = row
    return agg, comp


def save(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


def main():
    global RESULTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["weighted", "gated"],
                    choices=["weighted", "gated"])
    ap.add_argument("--systems", nargs="+", default=SYSTEM_ORDER)
    ap.add_argument("--system", choices=sorted(SYSTEMS), default=None,
                    help="single system; overrides --systems (e.g. fecrnic)")
    ap.add_argument("--out", type=str, default=None,
                    help="results JSON (default models/results_remedy.json); "
                         "use a system-specific path for new systems")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.system:
        args.systems = [args.system]
    if args.out:
        RESULTS_PATH = os.path.abspath(args.out)

    results = {"config": {"arch": ARCH, "hidden": list(HIDDEN),
                          "epochs": args.epochs, "patience": PATIENCE,
                          "batch": BATCH, "lr": LR, "huber_delta": HUBER_DELTA,
                          "loss": "huber", "pos_thresh": POS_THRESH,
                          "device": DEVICE, "seeds": args.seeds,
                          "variants": args.variants, "systems": args.systems,
                          "split": "cluster_split 64/16/20, SEEDS",
                          "protocol": "revision_analyses.detection_recomputed"}}
    det, base_mae = load_baselines(args.systems)
    if os.path.exists(RESULTS_PATH):
        prev = json.load(open(RESULTS_PATH, encoding="utf-8"))
        results["runs"] = prev.get("runs", {})
    else:
        results["runs"] = {}
    results["baselines"] = det
    results["baselines_mae"] = base_mae

    t_all = time.time()
    for system in args.systems:
        cfg = SYSTEMS[system]
        X, Y, df = load_data(system)
        _, names = active_phases(system)
        box = df["in_stainless_box"].values.astype(bool)
        print(f"[{system}] {len(X)} rows, {len(names)} phases {names} "
              f"| device {DEVICE}", flush=True)
        rng = np.random.default_rng(0)
        for seed in args.seeds:
            tr, va, te = cluster_split(X, Y, seed)
            te_idx = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
            for variant in args.variants:
                key = f"{variant}_{system}_s{seed}"
                npz = os.path.join(MODELS_DIR, f"pred_{system}_{variant}_{ARCH}_s{seed}.npz")
                if key in results["runs"] and os.path.exists(npz) and not args.force:
                    print(f"  skip {key} (exists)", flush=True)
                    continue
                t0 = time.time()
                if variant == "weighted":
                    model = MLP4(n_phases=len(names), head="sigmoid",
                                 hidden=HIDDEN, n_in=X.shape[1]).to(DEVICE)
                else:
                    model = GatedMLP(n_phases=len(names), hidden=HIDDEN,
                                     n_in=X.shape[1]).to(DEVICE)
                model, scaler, best_val, ep_run, prev_k, w_k = train_variant(
                    model, variant, X, Y, tr, va, seed, args.epochs)
                raw = predict(model, variant, scaler, X, te_idx)
                proj = renorm(raw)
                y_t = Y[te_idx]
                res = evaluate(y_t, proj, box[te_idx])
                det_m = detection_metrics(y_t, proj, names)
                np.savez(npz, y_true=y_t, y_pred=proj, box=box[te_idx],
                         te_idx=te_idx, features=X[te_idx])
                rec = {
                    "system": system, "variant": variant, "seed": int(seed),
                    "mae": float(res["mean_mae"]),
                    "per_phase_mae": [float(v) for v in res["mae"]],
                    "r2": [float(v) for v in res["r2"]],
                    "consistency": float(res["consistency"]),
                    "max_sum_err_raw": float(np.max(np.abs(raw.sum(axis=1) - 1.0))),
                    "max_sum_err_proj": float(np.max(np.abs(proj.sum(axis=1) - 1.0))),
                    "best_val_mae": float(best_val),
                    "epochs_run": int(ep_run),
                    "time_s": round(time.time() - t0, 1),
                    "train_prevalence": {n: float(prev_k[i]) for i, n in enumerate(names)},
                    "phase_weights": {n: float(w_k[i]) for i, n in enumerate(names)},
                }
                rec.update(det_m)
                results["runs"][key] = rec
                agg, comp = aggregate(results["runs"], det, base_mae)
                results["aggregated"] = agg
                results["comparison"] = comp
                save(results)
                print(f"  {key}: macroAUPRC={det_m['macro_auprc']:.4f} "
                      f"MAE={rec['mae']:.4f} F1excl="
                      f"{det_m['macro_f1_zero_pos_excluded']} "
                      f"eps={ep_run} t={rec['time_s']}s", flush=True)
                print(f"      AUPRC { {k: round(v, 3) for k, v in det_m['per_phase_auprc'].items()} }",
                      flush=True)
        del X, Y
    results["config"]["total_time_s"] = round(time.time() - t_all, 1)
    save(results)
    print(f"Saved {RESULTS_PATH} ({results['config']['total_time_s']}s)", flush=True)


if __name__ == "__main__":
    main()

