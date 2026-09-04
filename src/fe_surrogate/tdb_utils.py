"""TDB utilities shared by the pipeline: eligible-phase detection and
pycalphad-clean extraction from the open MatCalc steel database."""
from __future__ import annotations

import re

CONSTITUENT_RE = re.compile(r'^\s*CONSTITUENT\s+(\S+)\s*:\s*(.*?)\s*:\s*!$', re.S)

BLOCK_START_RE = re.compile(
    r'^\s*(TYPE_|DEFINE_SYSTEM|DEFAULT_COMMAND|ELEMENT\s|FUNCTION\s|'
    r'PHASE\s|CONSTITUENT\s|CONST\s|PARAMETER\s)')
PHASE_START_RE = re.compile(r'^\s*PHASE\s+(\S+)')
# MatCalc-specific commands that pycalphad's TDB parser cannot read. They only
# add composition sets for elements outside the selected systems, so dropping
# them is safe.
MATCALC_CMD_RE = re.compile(
    r'^\s*(REFERENCE_ELEMENT|ADD_COMPOSITION_SET|CREATE_NEW_PHASE|'
    r'ATTACH_CONTRIBUTION|DETACH_CONTRIBUTION)\b')

# Site-ratio marker that terminates a CONSTITUENT block (":!" or " : !").
END_MARKER_RE = re.compile(r":\s*!\s*$")


def eligible_phases(tdb_path, system_elems):
    """Return {phase: (sublattice, ...)} for phases whose constituents can
    all be populated by the system elements (VA allowed as a pseudo-element).
    Handles constituent definitions that wrap across multiple lines and end
    with the ':!' site-ratio marker. Element symbols are compared uppercase
    (TDB convention)."""
    system_elems = {e.upper() for e in system_elems} | {"VA"}
    phases = {}
    with open(tdb_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("CONSTITUENT"):
            i += 1
            continue
        body = line
        while not END_MARKER_RE.search(body) and i + 1 < len(lines):
            i += 1
            body += " " + lines[i].strip()
        m = CONSTITUENT_RE.match(body)
        if m:
            name = m.group(1)
            subs = [s.strip() for s in m.group(2).split(":") if s.strip()]
            ok = True
            for sub in subs:
                tokens = {t.strip().rstrip("%") for t in sub.split(",") if t.strip()}
                if not (tokens & system_elems):
                    ok = False
                    break
            if ok:
                phases[name] = subs
        i += 1
    return phases


def _prune_constituents(text, system_elems):
    """Rewrite CONSTITUENT blocks keeping only system elements + VA.

    Phases whose constituents include elements outside the system never
    populate them (their mole fraction is identically zero), so restricting
    the arrays is physically exact for the system's equilibria and shrinks
    the solver problem enormously. Phases left with an empty sublattice
    (e.g. carbides in a C-free system) cannot form and are dropped.
    """
    keep = {e.upper() for e in system_elems} | {"VA"}
    out = []
    for block in text.split("\n\n"):
        if not block.lstrip().startswith("CONSTITUENT"):
            out.append(block)
            continue
        m = CONSTITUENT_RE.match(block)
        if not m:
            out.append(block)
            continue
        name = m.group(1)
        subs = []
        drop = False
        for sub in m.group(2).split(":"):
            tokens = [t.strip().rstrip("%").upper()
                      for t in sub.replace("\n", " ").split(",") if t.strip()]
            kept = [t for t in tokens if t in keep]
            if not kept:
                drop = True
                break
            subs.append(",".join(kept))
        if drop:
            continue
        out.append(f"CONSTITUENT {name}  : {':'.join(subs)} :!")
    return "\n\n".join(out)


def extract_ternary_tdb(src_path, dst_path, system_elems, title, prune=True):
    """Extract a pycalphad-clean TDB keeping all ELEMENT/FUNCTION definitions
    but only the phases eligible for the system. Repairs the '6000.00.00'
    typo in the mc_fe source (MatCalc tolerates it, pycalphad does not)."""
    eligible = set(eligible_phases(src_path, system_elems).keys())

    with open(src_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.split("\n")

    blocks = []
    current_block = []

    def flush_block():
        if current_block:
            blocks.append("\n".join(current_block))
            current_block.clear()

    for line in lines:
        if line.strip().startswith("$"):
            continue
        if MATCALC_CMD_RE.match(line):
            continue
        if BLOCK_START_RE.match(line):
            flush_block()
            current_block.append(line)
        elif current_block:
            current_block.append(line)
    flush_block()

    out_blocks = [f"$ {title} thermodynamic database\n"
                  f"$ Clean extract from the open MatCalc steel database "
                  f"mc_fe_v2.062.tdb\n$ (TU Wien, Open Database License 1.0).\n$\n"]
    phase_active = False
    for block in blocks:
        first = block.split("\n")[0].strip()
        if first.startswith(("TYPE_", "DEFINE_SYSTEM", "DEFAULT_COMMAND",
                             "ELEMENT", "FUNCTION")):
            out_blocks.append(block)
            continue
        m = PHASE_START_RE.match(first)
        if m:
            phase_active = m.group(1) in eligible
            if phase_active:
                out_blocks.append(block)
            continue
        if first.startswith(("CONSTITUENT", "CONST", "PARAMETER")) and phase_active:
            out_blocks.append(block)

    out_text = "\n".join(out_blocks)
    if prune:
        out_text = _prune_constituents(out_text, system_elems)
    out_text = out_text.replace("6000.00.00", "6000.00")
    # Two mc_fe parameters (PDMN_B2) carry a duplicated trailing temperature
    # range ("...N ; 6000.00 N"); MatCalc tolerates it, pycalphad does not.
    out_text = re.sub(r"\bN\s*;\s*\d+(?:\.\d+)?\s+N\b", "N", out_text)
    # REF keys in mc_fe may contain spaces ("REF:TEST KOZE10"); pycalphad's
    # reference_key grammar allows only alphanumerics, ':', '_', '-'.
    out_text = re.sub(r"REF:([^!\n]*?)(?=!|$)",
                      lambda m: "REF:" + re.sub(r"\s+", "_", m.group(1).strip()),
                      out_text)

    # pycalphad's TDB reader splits commands on '!' and joins lines. Two
    # statements with no '!' between them (e.g. adjacent PARAMETER lines
    # sharing one REF:...!) become one unparseable command. Insert a '!'
    # terminator whenever a new top-level statement follows a line that was
    # not already terminated.
    top_keyword = re.compile(
        r'^\s*(PARAMETER|ELEMENT|FUNCTION|PHASE|CONSTITUENT|CONST\b|'
        r'TYPE_|DEFINE_SYSTEM|DEFAULT_COMMAND)\b')
    out_lines = out_text.split("\n")
    fixed = []
    for line in out_lines:
        if fixed and top_keyword.match(line) and not fixed[-1].rstrip().endswith("!"):
            fixed[-1] = fixed[-1].rstrip() + " !"
        fixed.append(line)
    out_text = "\n".join(fixed)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    return len(eligible)