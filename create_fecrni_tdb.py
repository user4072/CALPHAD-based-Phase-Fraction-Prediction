"""Create a clean Fe-Cr-Ni ternary TDB from the open MatCalc steel database.

Source: mc_fe_v2.062.tdb (MatCalc, TU Wien, ODbL license).
Strategy (same as create_alniti_tdb.py): keep ALL element definitions + ALL
functions so cross-references work. Only filter PHASE/CONSTITUENT/PARAMETER
to the phases relevant to the Fe-Cr-Ni system.

Fe-Cr-Ni is the only candidate with genuine ternary interaction parameters
in all key phases (LIQUID, FCC_A1, BCC_A2, CEMENTITE, M7C3), making it the
physically richest ternary in this database.
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "databases" / "mc_fe_v2.062.tdb"
DST = HERE / "databases" / "fecrni_ternary.tdb"

PHASES_OF_INTEREST = {
    "LIQUID", "FCC_A1", "BCC_A2", "HCP_A3",
    "BCC_B2", "SIGMA", "CHI_A12",
}

BLOCK_START_RE = re.compile(
    r'^\s*(TYPE_|DEFINE_SYSTEM|DEFAULT_COMMAND|ELEMENT\s|FUNCTION\s|'
    r'PHASE\s|CONSTITUENT\s|CONST\s|PARAMETER\s)')
PHASE_START_RE = re.compile(r'^\s*PHASE\s+(\S+)')
# MatCalc-specific commands that pycalphad's TDB parser cannot read.
# They only add composition sets for elements outside the Fe-Cr-Ni system
# (C, N, Cu, Nb, Ta, Ti, V, Mo, W, Si, Mn, Co), so dropping them is safe.
MATCALC_CMD_RE = re.compile(
    r'^\s*(REFERENCE_ELEMENT|ADD_COMPOSITION_SET|CREATE_NEW_PHASE|'
    r'ATTACH_CONTRIBUTION|DETACH_CONTRIBUTION)\b')


def is_comment(line):
    return line.strip().startswith('$')


def keep_line(line):
    return not MATCALC_CMD_RE.match(line)


def extract_fecrni_tdb():
    with open(SRC, encoding='utf-8', errors='replace') as f:
        text = f.read()

    lines = text.split('\n')
    blocks = []
    current_block = []

    def flush_block():
        if current_block:
            blocks.append('\n'.join(current_block))
            current_block.clear()

    for line in lines:
        if is_comment(line):
            continue
        if not keep_line(line):
            continue
        if BLOCK_START_RE.match(line):
            flush_block()
            current_block.append(line)
        elif current_block:
            current_block.append(line)

    flush_block()

    out_blocks = []
    phase_active = False

    header = """$ Fe-Cr-Ni ternary thermodynamic database
$ Clean extract from the open MatCalc steel database mc_fe_v2.062.tdb
$ (TU Wien, Open Database License 1.0).
$ Fe-Cr-Ni has assessed ternary interaction parameters in LIQUID, FCC_A1,
$ BCC_A2, CEMENTITE and M7C3 - the stainless-steel backbone system.
$
"""
    out_blocks.append(header)

    for block in blocks:
        first_line = block.split('\n')[0].strip()

        if first_line.startswith("TYPE_") or first_line.startswith("DEFINE_SYSTEM") or first_line.startswith("DEFAULT_COMMAND"):
            out_blocks.append(block)
            continue

        if first_line.startswith("ELEMENT"):
            out_blocks.append(block)
            continue

        if first_line.startswith("FUNCTION"):
            out_blocks.append(block)
            continue

        m = PHASE_START_RE.match(first_line)
        if m:
            phase_name = m.group(1)
            phase_active = phase_name in PHASES_OF_INTEREST
            if phase_active:
                out_blocks.append(block)
            continue

        if first_line.startswith("CONSTITUENT") or first_line.startswith("CONST") or first_line.startswith("PARAMETER"):
            if phase_active:
                out_blocks.append(block)
            continue

    out_text = '\n'.join(out_blocks)

    # Repair a typo present in the original mc_fe source (two SIGMA parameters
    # carry "6000.00.00" instead of "6000.00"; MatCalc tolerates it, pycalphad
    # does not).
    out_text = out_text.replace('6000.00.00', '6000.00')

    with open(DST, 'w', encoding='utf-8') as f:
        f.write(out_text)

    print(f"Created {DST}")
    print(f"  Input lines: {len(lines)}")
    print(f"  Output lines: {out_text.count(chr(10)) + 1}")


if __name__ == "__main__":
    extract_fecrni_tdb()