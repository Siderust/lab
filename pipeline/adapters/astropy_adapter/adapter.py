#!/usr/bin/env python3
"""
Astropy Adapter for the Siderust Lab.

Reads experiment + inputs from stdin, runs Astropy transformations,
writes JSON results to stdout.

Requires: pip install astropy numpy

Supported experiments:
  frame_rotation_bpn — Bias-Precession-Nutation matrix (GCRS → CIRS)
    Input per line: jd_tt  vx vy vz
    Output: transformed direction + matrix elements

  gmst_era — Greenwich Mean Sidereal Time and Earth Rotation Angle
    Input per line: jd_ut1  jd_tt
    Output: GMST (rad), ERA (rad), GAST (rad)
"""

import sys
import json
import math
import time
import numpy as np

def normalize3(v):
    r = np.linalg.norm(v)
    if r > 0:
        return v / r
    return v

def ang_sep(a, b):
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.acos(dot)

def run_frame_rotation_bpn(lines_iter):
    """
    Compute the Bias-Precession-Nutation matrix using Astropy's ERFA bindings.

    Astropy internally uses ERFA's eraPnm06a (IAU 2006/2000A) for the BPN matrix.
    We call it directly via erfa to get the exact same result as the ERFA C adapter.
    """
    try:
        import erfa
    except ImportError:
        # astropy bundles erfa
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    cases = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jd_tt = float(parts[0])
        vin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        vin = normalize3(vin)

        # Split JD for ERFA precision
        date1 = 2451545.0
        date2 = jd_tt - 2451545.0

        # BPN matrix (GCRS → CIRS), IAU 2006/2000A
        rnpb = erfa.pnm06a(date1, date2)

        vout = normalize3(rnpb @ vin)

        # Closure test
        vinv = normalize3(rnpb.T @ vout)
        closure_rad = ang_sep(vin, vinv)

        cases.append({
            "jd_tt": jd_tt,
            "input": vin.tolist(),
            "output": vout.tolist(),
            "closure_rad": closure_rad,
            "matrix": rnpb.tolist(),
        })

    result = {
        "experiment": "frame_rotation_bpn",
        "library": "astropy",
        "model": "IAU_2006_2000A (via erfa.pnm06a)",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_gmst_era(lines_iter):
    """Compute GMST, ERA, and GAST via Astropy's ERFA bindings."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    cases = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jd_ut1 = float(parts[0])
        jd_tt = float(parts[1])

        ut1_hi = 2451545.0
        ut1_lo = jd_ut1 - 2451545.0
        tt_hi = 2451545.0
        tt_lo = jd_tt - 2451545.0

        gmst = erfa.gmst06(ut1_hi, ut1_lo, tt_hi, tt_lo)
        era = erfa.era00(ut1_hi, ut1_lo)
        gast = erfa.gst06a(ut1_hi, ut1_lo, tt_hi, tt_lo)

        cases.append({
            "jd_ut1": jd_ut1,
            "jd_tt": jd_tt,
            "gmst_rad": float(gmst),
            "era_rad": float(era),
            "gast_rad": float(gast),
        })

    result = {
        "experiment": "gmst_era",
        "library": "astropy",
        "model": "GMST=IAU2006, ERA=IAU2000 (via erfa)",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_frame_rotation_bpn_perf(lines_iter):
    """Performance measurement for BPN matrix computation."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    jds = []
    vecs = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jds.append(float(parts[0]))
        v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        vecs.append(normalize3(v))

    # Warm-up
    for i in range(min(n, 100)):
        erfa.pnm06a(2451545.0, jds[i] - 2451545.0)

    # Timed run
    t0 = time.perf_counter_ns()
    sink = np.zeros(3)
    for i in range(n):
        rnpb = erfa.pnm06a(2451545.0, jds[i] - 2451545.0)
        sink = rnpb @ vecs[i]
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "frame_rotation_bpn_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink[0]),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def main():
    lines_iter = iter(sys.stdin)
    experiment = next(lines_iter).strip()

    dispatch = {
        "frame_rotation_bpn": run_frame_rotation_bpn,
        "gmst_era": run_gmst_era,
        "frame_rotation_bpn_perf": run_frame_rotation_bpn_perf,
    }

    if experiment not in dispatch:
        print(f"Unknown experiment: {experiment}", file=sys.stderr)
        sys.exit(1)

    dispatch[experiment](lines_iter)


if __name__ == "__main__":
    main()
