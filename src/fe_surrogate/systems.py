"""System registry for the Fe-Cr-Ni / Fe-Cr-Mn surrogate studies.

Each entry carries everything the pipeline needs for one ternary system
extracted from the open MatCalc steel database mc_fe_v2.062.tdb.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEMS = {
    "fecrni": {
        "name": "Fe-Cr-Ni",
        "tdb": os.path.join(HERE, "databases", "fecrni_ternary.tdb"),
        "dataset": os.path.join(HERE, "data", "raw", "dataset_fecrni.csv"),
        "checkpoint": os.path.join(HERE, "data", "raw", "checkpoint_fecrni.csv"),
        "elements": ["FE", "CR", "NI", "VA"],          # pycalphad species
        "comps": ["Fe", "Cr", "Ni"],                   # feature column names
        "comps_species": ["CR", "NI"],                 # independent X() variables
        "phases": ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"],
        "phase_cols": ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"],
        "box": {"Fe": 0.55, "Cr": 0.30, "Ni": 0.30},   # stainless-like region
        "sigma_focus": {"low": [0.15, 0.50, 700.0], "high": [0.02, 0.25, 1250.0]},
        "t_min": 700.0, "t_max": 2000.0,
    },
    "fecrmn": {
        "name": "Fe-Cr-Mn",
        "tdb": os.path.join(HERE, "databases", "fecrmn_ternary.tdb"),
        "dataset": os.path.join(HERE, "data", "raw", "dataset_fecrmn.csv"),
        "checkpoint": os.path.join(HERE, "data", "raw", "checkpoint_fecrmn.csv"),
        "elements": ["FE", "CR", "MN", "VA"],
        "comps": ["Fe", "Cr", "Mn"],
        "comps_species": ["CR", "MN"],
        "phases": ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"],
        "phase_cols": ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"],
        "box": {"Fe": 0.55, "Cr": 0.30, "Mn": 0.30},
        "sigma_focus": {"low": [0.15, 0.50, 700.0], "high": [0.02, 0.25, 1250.0]},
        "t_min": 700.0, "t_max": 2000.0,
    },
    "fecrmo": {
        "name": "Fe-Cr-Mo",
        "tdb": os.path.join(HERE, "databases", "fecrmo_ternary.tdb"),
        "dataset": os.path.join(HERE, "data", "raw", "dataset_fecrmo.csv"),
        "checkpoint": os.path.join(HERE, "data", "raw", "checkpoint_fecrmo.csv"),
        "elements": ["FE", "CR", "MO", "VA"],
        "comps": ["Fe", "Cr", "Mo"],
        "comps_species": ["CR", "MO"],
        "phases": ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"],
        "phase_cols": ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"],
        "box": {"Fe": 0.55, "Cr": 0.30, "Mo": 0.30},
        "sigma_focus": {"low": [0.15, 0.50, 700.0], "high": [0.02, 0.25, 1250.0]},
        "t_min": 700.0, "t_max": 2000.0,
    },
    "fecrv": {
        "name": "Fe-Cr-V",
        "tdb": os.path.join(HERE, "databases", "fecrv_ternary.tdb"),
        "dataset": os.path.join(HERE, "data", "raw", "dataset_fecrv.csv"),
        "checkpoint": os.path.join(HERE, "data", "raw", "checkpoint_fecrv.csv"),
        "elements": ["FE", "CR", "V", "VA"],
        "comps": ["Fe", "Cr", "V"],
        "comps_species": ["CR", "V"],
        "phases": ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"],
        "phase_cols": ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"],
        "box": {"Fe": 0.55, "Cr": 0.30, "V": 0.30},
        "sigma_focus": {"low": [0.15, 0.50, 700.0], "high": [0.02, 0.25, 1250.0]},
        "t_min": 700.0, "t_max": 2000.0,
    },
    "femnni": {
        "name": "Fe-Mn-Ni",
        "tdb": os.path.join(HERE, "databases", "femnni_ternary.tdb"),
        "dataset": os.path.join(HERE, "data", "raw", "dataset_femnni.csv"),
        "checkpoint": os.path.join(HERE, "data", "raw", "checkpoint_femnni.csv"),
        "elements": ["FE", "MN", "NI", "VA"],
        "comps": ["Fe", "Mn", "Ni"],
        "comps_species": ["MN", "NI"],
        "phases": ["LIQUID", "FCC_A1", "BCC_A2", "SIGMA"],
        "phase_cols": ["NP_LIQUID", "NP_FCC_A1", "NP_BCC_A2", "NP_SIGMA"],
        "box": {"Fe": 0.55, "Mn": 0.30, "Ni": 0.30},
        "sigma_focus": {"low": [0.15, 0.50, 700.0], "high": [0.02, 0.25, 1250.0]},
        "t_min": 700.0, "t_max": 2000.0,
    },
}