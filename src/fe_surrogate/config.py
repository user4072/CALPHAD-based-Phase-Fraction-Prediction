"""Configuration for the Fe-Cr-Ni surrogate project."""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(HERE, "databases", "fecrni_ternary.tdb")
DATA_DIR = os.path.join(HERE, "data", "raw")
MODELS_DIR = os.path.join(HERE, "models")

ELEMENTS = ["Fe", "Cr", "Ni"]
PHASE_COLS = ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"]
PHASE_NAMES = ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"]
PHASE_LABELS = ["Liquid", "gamma (FCC)", "alpha (BCC)", "sigma (TCP)"]

T_MIN, T_MAX = 700.0, 2000.0
P = 101325.0
MAX_ITERATIONS = 500
MASS_BALANCE_TOL = 1e-6