#!/usr/bin/env python3
"""Extract IAU 2000A nutation tables from ERFA nut00a.c and write Rust arrays."""
import re

with open("erfa/src/nut00a.c") as f:
    src = f.read()

# ──────────────────────────────────────────────
# 1. Luni-solar table (xls[])
# ──────────────────────────────────────────────
ls_match = re.search(r'xls\[\]\s*=\s*\{(.+?)\};', src, re.DOTALL)
ls_block = ls_match.group(1)

# Each row: {nl,nlp,nf,nd,nom, sp,spt,cp, ce,cet,se} = 11 fields
ls_rows = re.findall(r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,'
                     r'\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,'
                     r'\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\}',
                     ls_block)
print(f"Luni-solar terms: {len(ls_rows)}")

# ──────────────────────────────────────────────
# 2. Planetary table (xpl[])
# ──────────────────────────────────────────────
pl_match = re.search(r'xpl\[\]\s*=\s*\{(.+?)\};', src, re.DOTALL)
pl_block = pl_match.group(1)

# Each row: {nl,nf,nd,nom, nme,nve,nea,nma, nju,nsa,nur,nne, npa, sp,cp,se,ce} = 17 fields
pl_rows = re.findall(r'\{\s*' + r'\s*,\s*'.join([r'(-?\d+)'] * 17) + r'\s*\}', pl_block)
print(f"Planetary terms: {len(pl_rows)}")

# ──────────────────────────────────────────────
# 3. Write Rust luni-solar table
# ──────────────────────────────────────────────
with open("/tmp/nut00a_ls.rs", "w") as out:
    out.write(f"/// IAU 2000A luni-solar nutation series ({len(ls_rows)} terms).\n")
    out.write("/// Columns: nl, nlp, nf, nd, nom, sp, spt, cp, ce, cet, se\n")
    out.write("/// Coefficients in units of 0.1 µas (and 0.1 µas/century for t-terms).\n")
    out.write("#[rustfmt::skip]\n")
    out.write(f"pub(crate) const NUT00A_LS: [[f64; 11]; {len(ls_rows)}] = [\n")
    for i, r in enumerate(ls_rows):
        vals = list(r)
        # First 5 are ints, rest are doubles
        ints = ", ".join(f"{int(v):3}" for v in vals[:5])
        floats = ", ".join(f"{float(v)}" for v in vals[5:])
        out.write(f"    [{ints}, {floats}],\n")
    out.write("];\n")

print("Wrote /tmp/nut00a_ls.rs")

# ──────────────────────────────────────────────
# 4. Write Rust planetary table
# ──────────────────────────────────────────────
with open("/tmp/nut00a_pl.rs", "w") as out:
    out.write(f"/// IAU 2000A planetary nutation series ({len(pl_rows)} terms).\n")
    out.write("/// Columns: nl, nf, nd, nom, nme, nve, nea, nma, nju, nsa, nur, nne, npa, sp, cp, se, ce\n")
    out.write("/// Note: planetary Delaunay args are l, F, D, Ω (no l').\n")  
    out.write("/// Coefficients sp, cp, se, ce in integer units of 0.1 µas.\n")
    out.write("#[rustfmt::skip]\n")
    out.write(f"pub(crate) const NUT00A_PL: [[i32; 17]; {len(pl_rows)}] = [\n")
    for i, r in enumerate(pl_rows):
        vals = ", ".join(f"{int(v):4}" for v in r)
        out.write(f"    [{vals}],\n")
    out.write("];\n")

print("Wrote /tmp/nut00a_pl.rs")

# ──────────────────────────────────────────────
# 5. Extract fundamental argument functions
# ──────────────────────────────────────────────
# Find inline polynomials used in nut00a for Delaunay args
# and planetary longitudes
# The luni-solar section uses eraFal03, inline l', eraFaf03, inline D, eraFaom03
# The planetary section uses inline Delaunay + eraFame03 etc.

# Let's also extract the fundamental argument function implementations
fa_files = ['fal03.c', 'falp03.c', 'faf03.c', 'fad03.c', 'faom03.c',
            'fame03.c', 'fave03.c', 'fae03.c', 'fama03.c',
            'faju03.c', 'fasa03.c', 'faur03.c', 'fapa03.c']

import os
for fname in fa_files:
    path = f"erfa/src/{fname}"
    if os.path.exists(path):
        with open(path) as ff:
            content = ff.read()
        # Extract the body between { and the first return 
        ret = re.search(r'return\s+(\w+)\s*;', content)
        if ret:
            var = ret.group(1)
            # Find the assignment to that variable
            assign = re.search(rf'{var}\s*=\s*(.+?);', content, re.DOTALL)
            if assign:
                expr = assign.group(1).strip()
                expr = re.sub(r'\s+', ' ', expr)
                print(f"{fname}: {var} = {expr[:150]}")
            else:
                print(f"{fname}: return {var} (no assignment found)")
        else:
            print(f"{fname}: no return found")
    else:
        print(f"{fname}: not found")

# Also extract the inline arguments from nut00a.c itself for the luni-solar section
# and the planetary section
print("\n--- Inline luni-solar Delaunay arguments in nut00a.c ---")

# Find the luni-solar Delaunay argument block
ls_args_block = re.search(r'/\* Fundamental \(Delaunay\) arguments from Simon.*?(?=\n/\*)', src, re.DOTALL)
if ls_args_block:
    print(ls_args_block.group()[:500])

# Find the planetary argument block  
print("\n--- Inline planetary arguments in nut00a.c ---")
pl_args_block = re.search(r'Planetary nutation.*?alme\s*=.*?apa\s*=[^;]+;', src, re.DOTALL)
if pl_args_block:
    print(pl_args_block.group()[:800])
