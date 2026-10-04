"""Assemble the Zenodo data release (data + results + audit trail, no code).

Collects datasets, probes, result JSONs, stored prediction tensors,
checkpoints, screen/anchor/learning artifacts, and the claim audit;
verifies headline row counts (44,397 + 13,192 + 13,199 + 8,874 + 8,880
= 88,542) before
zipping; writes manifest.json (sha256 per file) into the archive.

Excluded by design: solver checkpoints (redundant), *.err/*.log,
licensed MatCalc TDBs (scripts regenerate from a local copy), legacy
model dirs, LaTeX build artifacts.

Usage:
    py -3.12 paper/make_release.py [--out fe_surrogate_release_v1.zip]

The uploader then: creates a Zenodo record, uploads the zip (+ fills
title/authors from .zenodo.json), publishes, and pastes the DOI over the
[Zenodo DOI] placeholder in paper/paper_cms.tex (+ docx twin).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_ROWS = {
    "fecrni": 8880, "fecrmn": 8880, "fecrmo": 8877, "fecrv": 8880,
    "femnni": 8880, "fecrnic": 13192, "fecrc": 13199,
    "crconi": 8874, "crnimn": 8880,
}
EXPECTED_TERNARY = 44397
EXPECTED_TOTAL = 88542

GLOBS = [
    "data/raw/dataset_*.csv",
    "data/raw/*_probe.json",
    "data/raw/*_active.json",
    "models/results_*.json",
    "models/tuning_sensitivity.json",
    "models/extra_seeds.json",
    "models/extra_*.npz",
    "models/paired_bootstrap.json",
    "models/paired_bootstrap_5seed.json",
    "models/gated_shift.json",
    "models/coverage_robustness.json",
    "models/ensemble_gate.json",
    "models/block_holdout_*.json",
    "models/holdout_extrap*.json",
    "models/loss_ablation_*.json",
    "models/width*.json",
    "models/residue_sweep_*.json",
    "models/boundary_error.json",
    "models/threshold_sensitivity.json",
    "analysis_revision/revision_analyses.json",
    "analysis_revision/detection_*.json",
    "models/pred_*.npz",
    "paper/anchor_data/anchor_ckpts/*.pt",
    "paper/screen_data/ckpts_fecrnic/*.pt",
    "paper/anchor_data/anchors.csv",
    "paper/anchor_data/anchors.md",
    "paper/anchor_data/anchor_eval.json",
    "paper/screen_data/*.json",
    "paper/screen_data/*.npz",
    "paper/audit_claims.py",
    "paper/claim_ledger.md",
    "LICENSE",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def count_rows(csv_path: str) -> int:
    with open(csv_path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fe_surrogate_release_v1.zip")
    args = ap.parse_args()

    files = []
    for pat in GLOBS:
        got = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not got:
            print(f"WARNING: pattern matched nothing: {pat}")
        files.extend(got)
    files = sorted(set(files))
    rel = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in files]
    print(f"collected {len(files)} files")

    # ---- headline verification (abort on mismatch) ----
    rows = {}
    for sys, n_exp in EXPECTED_ROWS.items():
        p = os.path.join(ROOT, "data", "raw", f"dataset_{sys}.csv")
        n = count_rows(p)
        rows[sys] = n
        assert n == n_exp, f"{sys}: got {n}, expected {n_exp}"
    tern = sum(rows[s] for s in
               ["fecrni", "fecrmn", "fecrmo", "fecrv", "femnni"])
    assert tern == EXPECTED_TERNARY, tern
    assert (tern + rows["fecrnic"] + rows["fecrc"] + rows["crconi"]
            + rows["crnimn"] == EXPECTED_TOTAL)
    print(f"row counts OK: ternary={tern} + {rows['fecrnic']} + "
          f"{rows['fecrc']} + {rows['crconi']} + {rows['crnimn']} = "
          f"{EXPECTED_TOTAL}")

    manifest = {
        "title": "fe_surrogate data release v1",
        "row_counts": rows,
        "total_equilibria": EXPECTED_TOTAL,
        "files": [],
    }
    total = 0
    for full, r in zip(files, rel):
        n = os.path.getsize(full)
        total += n
        manifest["files"].append(
            {"path": r, "bytes": n, "sha256": sha256(full)})
    print(f"payload {total / 1e6:.1f} MB")

    out = os.path.join(ROOT, args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for full, r in zip(files, rel):
            z.write(full, r)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB zipped)")
    print()
    print("UPLOAD CHECKLIST (manual, ~10 min):")
    print("  1. zenodo.org -> New upload -> drop", args.out)
    print("  2. Fill title/creators/description from .zenodo.json; "
          "confirm author list + ORCIDs + data license.")
    print("  3. Publish -> copy the DOI.")
    print("  4. Replace [Zenodo DOI] in paper/paper_cms.tex (+ rebuild "
          "docx: py -3.12 paper/make_docx.py), re-run audit, commit.")


if __name__ == "__main__":
    main()
