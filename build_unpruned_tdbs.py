"""Build unpruned (syntax-repaired only) TDBs for the equivalence test.

The raw mc_fe_v2.062.tdb cannot be parsed by pycalphad at all (parse error on
REFERENCE_ELEMENT), so "original vs extracted" is not a valid comparison.
Instead we compare the syntax-repaired-but-unpruned TDB against the final
pruned product: both load in pycalphad and differ only by constituent-array
pruning and out-of-system phase removal.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fe_surrogate.systems import SYSTEMS
from fe_surrogate.tdb_utils import extract_ternary_tdb

FULL_TDB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "databases", "mc_fe_v2.062.tdb")

for s in SYSTEMS:
    cfg = SYSTEMS[s]
    dst = os.path.join(os.path.dirname(__file__), "databases", f"{s}_unpruned.tdb")
    extract_ternary_tdb(FULL_TDB, dst, cfg["comps"], f"extracted (unpruned) {s}",
                        prune=False)
    print(f"built {os.path.basename(dst)}")
print("done")