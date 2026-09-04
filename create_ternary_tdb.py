"""Extract a pycalphad-clean ternary TDB from the open MatCalc steel database.

The phase set is data-driven: every phase whose constituents can all be
populated by the system's elements is kept (see fe_surrogate/tdb_utils.py).
No hand-picked whitelist, so no stable phase can be silently dropped.

Usage: python create_ternary_tdb.py --system fecrmo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import extract_ternary_tdb

HERE = os.path.dirname(os.path.abspath(__file__))
FULL_TDB = os.path.join(HERE, "databases", "mc_fe_v2.062.tdb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    args = ap.parse_args()
    cfg = SYSTEMS[args.system]
    n_elig = extract_ternary_tdb(FULL_TDB, cfg["tdb"], cfg["comps"],
                                 f"{cfg['name']} ternary")
    print(f"Created {cfg['tdb']} with {n_elig} eligible phases")


if __name__ == "__main__":
    main()