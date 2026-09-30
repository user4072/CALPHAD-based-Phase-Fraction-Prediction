"""Strict out-of-range extrapolation on the CARBON axis for fecrc.

holdout_eval.py's extrap mode only builds blocks on x2 (the second
component, Cr for fecrc) and T, and adapts x_test_lo automatically when the
design box stops short of the default (Cr <= 0.30 -> train Cr <= 0.25,
test Cr > 0.275).  The carbon axis is the interesting one for the Fe-Cr-C
carbide system, but holdout_eval.py has no per-system C block -- so this
script adds it as a new analysis, reusing holdout_eval.fit_and_eval
(models, split, preprocessing, early stopping, metrics: all identical).

Block (same structure as X2_extrap in holdout_eval.py):
  C_extrap     train C <= 0.045, test C > 0.045   (gap-free, strictly
               out-of-range: every test row exceeds the training range)
  C_extrap_ctrl size-matched random control: the same number of rows
               excluded from training, equal-size random test set

Usage: py -3.12 -X utf8 analysis_revision/holdout_extrap_c_fecrc.py
Output: models/holdout_extrap_c_fecrc.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from fe_surrogate.experiment import load_data
from holdout_eval import fit_and_eval

SYSTEM = "fecrc"
C_TRAIN_HI = 0.045   # near side kept in training
C_TEST_LO = 0.045    # far side used as test (strictly out of range)


def c_blocks_for(df, seed):
    """{name: (held_mask, exclude_mask)} for the carbon axis."""
    n = len(df)
    c_held = (df["C"] > C_TEST_LO).values
    c_excl = (df["C"] > C_TRAIN_HI).values  # held (+ empty gap)
    assert c_excl.sum() >= c_held.sum() > 0, \
        f"no rows with C > {C_TEST_LO} (max {df['C'].max():.4f})"
    rng = np.random.default_rng(3000 + seed)
    blocks = {"C_extrap": (c_held, c_excl)}
    e_idx = rng.choice(n, int(c_excl.sum()), replace=False)
    e_mask = np.zeros(n, dtype=bool)
    e_mask[e_idx] = True
    h_mask = np.zeros(n, dtype=bool)
    h_mask[rng.choice(e_idx, int(c_held.sum()), replace=False)] = True
    blocks["C_extrap_ctrl"] = (h_mask, e_mask)
    return blocks


def main():
    df = load_data(SYSTEM)[2]
    # Strictness check, same style as holdout_eval.run_system_extrap.
    near = df.loc[df["C"] <= C_TRAIN_HI, "C"]
    far = df.loc[df["C"] > C_TEST_LO, "C"]
    assert len(near) > 0 and len(far) > 0
    assert near.max() <= C_TRAIN_HI and far.min() > C_TEST_LO
    header = (f"strict carbon extrapolation: train C <= {C_TRAIN_HI} -> "
              f"test C > {C_TEST_LO} | near {len(near)} rows, far {len(far)} rows")
    print(header, flush=True)
    fit_and_eval(SYSTEM,
                 lambda seed: c_blocks_for(df, seed),
                 "holdout_extrap_c_fecrc.json", header)


if __name__ == "__main__":
    main()
