"""Validate that every extracted ternary TDB loads cleanly in pycalphad."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pycalphad import Database
from fe_surrogate.systems import SYSTEMS

ok = True
for name, cfg in SYSTEMS.items():
    try:
        db = Database(cfg["tdb"])
        print(f"{name}: OK ({len(db.phases)} phases)")
    except Exception as e:
        ok = False
        print(f"{name}: FAIL - {type(e).__name__}: {str(e)[:200]}")
print("ALL OK" if ok else "SOME FAILED")