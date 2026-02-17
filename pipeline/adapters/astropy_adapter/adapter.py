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

def solve_kepler_newton(M_rad, e, max_iter=100, tol=1e-15):
    """Solve M = E - e*sin(E) with a shared Newton setup for parity."""
    E = M_rad
    converged = True
    iters = 0
    for iters in range(max_iter):
        f = E - e * math.sin(E) - M_rad
        fp = 1.0 - e * math.cos(E)
        if abs(fp) < 1e-30:
            converged = False
            break
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    else:
        converged = False
    return E, iters, converged

def sun_ra_dec_dist_from_epv00(jd_tt, erfa_mod):
    """Compute Sun geocentric RA/Dec/dist from ERFA epv00."""
    date1, date2 = 2451545.0, jd_tt - 2451545.0
    pvh, _pvb = erfa_mod.epv00(date1, date2)
    sx, sy, sz = -pvh[0][0], -pvh[0][1], -pvh[0][2]
    dist_au = math.sqrt(sx*sx + sy*sy + sz*sz)
    ra = math.atan2(sy, sx) % (2 * math.pi)
    dec = math.asin(sz / dist_au)
    return ra, dec, dist_au

def moon_ra_dec_dist_meeus(jd_tt):
    """Moon geocentric RA/Dec via simplified Meeus Ch.47."""
    T = (jd_tt - 2451545.0) / 36525.0

    Lp = (218.3164477 + 481267.88123421*T - 0.0015786*T**2
          + T**3/538841.0 - T**4/65194000.0) % 360.0
    D  = (297.8501921 + 445267.1114034*T - 0.0018819*T**2
          + T**3/545868.0 - T**4/113065000.0) % 360.0
    M  = (357.5291092 + 35999.0502909*T - 0.0001536*T**2
          + T**3/24490000.0) % 360.0
    Mp = (134.9633964 + 477198.8675055*T + 0.0087414*T**2
          + T**3/69699.0 - T**4/14712000.0) % 360.0
    F  = (93.2720950 + 483202.0175233*T - 0.0036539*T**2
          - T**3/3526000.0 + T**4/863310000.0) % 360.0

    Lp_r, D_r, M_r, Mp_r, F_r = [x * math.pi / 180 for x in [Lp, D, M, Mp, F]]

    sum_l = (6288774*math.sin(Mp_r) + 1274027*math.sin(2*D_r - Mp_r)
            + 658314*math.sin(2*D_r) + 213618*math.sin(2*Mp_r)
            - 185116*math.sin(M_r) - 114332*math.sin(2*F_r))

    sum_b = (5128122*math.sin(F_r) + 280602*math.sin(Mp_r + F_r)
            + 277693*math.sin(Mp_r - F_r) + 173237*math.sin(2*D_r - F_r))

    sum_r = (-20905355*math.cos(Mp_r) - 3699111*math.cos(2*D_r - Mp_r)
            - 2955968*math.cos(2*D_r) - 569925*math.cos(2*Mp_r))

    ecl_lon = (Lp + sum_l / 1e6) * math.pi / 180
    ecl_lat = (sum_b / 1e6) * math.pi / 180
    dist_km = 385000.56 + sum_r / 1000.0

    eps = (23.4392911 - 0.01300417*T - 1.638e-7*T**2) * math.pi / 180
    ce, se = math.cos(eps), math.sin(eps)
    ra = math.atan2(math.sin(ecl_lon)*ce - math.tan(ecl_lat)*se, math.cos(ecl_lon))
    ra = ra % (2 * math.pi)
    dec = math.asin(math.sin(ecl_lat)*ce + math.cos(ecl_lat)*se*math.sin(ecl_lon))
    return ra, dec, dist_km

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


# -----------------------------------------------------------------------
# New experiments
# -----------------------------------------------------------------------

def run_equ_ecl(lines_iter):
    """Equatorial ↔ Ecliptic via erfa.eqec06 / erfa.eceq06 (IAU 2006)."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())
    cases = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jd_tt = float(parts[0])
        ra_rad = float(parts[1])
        dec_rad = float(parts[2])

        date1 = 2451545.0
        date2 = jd_tt - 2451545.0

        ecl_lon, ecl_lat = erfa.eqec06(date1, date2, ra_rad, dec_rad)
        ra_back, dec_back = erfa.eceq06(date1, date2, ecl_lon, ecl_lat)

        v_in = np.array([math.cos(dec_rad)*math.cos(ra_rad),
                         math.cos(dec_rad)*math.sin(ra_rad),
                         math.sin(dec_rad)])
        v_bk = np.array([math.cos(dec_back)*math.cos(ra_back),
                         math.cos(dec_back)*math.sin(ra_back),
                         math.sin(dec_back)])
        closure_rad = ang_sep(v_in, v_bk)

        cases.append({
            "jd_tt": jd_tt,
            "ra_rad": ra_rad,
            "dec_rad": dec_rad,
            "ecl_lon_rad": float(ecl_lon),
            "ecl_lat_rad": float(ecl_lat),
            "closure_rad": closure_rad,
        })

    result = {
        "experiment": "equ_ecl",
        "library": "astropy",
        "model": "IAU_2006_ecliptic (via erfa)",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_equ_horizontal(lines_iter):
    """Equatorial → Horizontal via erfa.hd2ae + GAST."""
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
        ra_rad = float(parts[2])
        dec_rad = float(parts[3])
        obs_lon = float(parts[4])
        obs_lat = float(parts[5])

        ut1_hi, ut1_lo = 2451545.0, jd_ut1 - 2451545.0
        tt_hi, tt_lo = 2451545.0, jd_tt - 2451545.0

        gast = erfa.gst06a(ut1_hi, ut1_lo, tt_hi, tt_lo)
        last = gast + obs_lon
        ha = last - ra_rad

        az, alt = erfa.hd2ae(ha, dec_rad, obs_lat)

        ha_back, dec_back = erfa.ae2hd(az, alt, obs_lat)
        ra_back = last - ha_back

        v_in = np.array([math.cos(dec_rad)*math.cos(ra_rad),
                         math.cos(dec_rad)*math.sin(ra_rad),
                         math.sin(dec_rad)])
        v_bk = np.array([math.cos(dec_back)*math.cos(ra_back),
                         math.cos(dec_back)*math.sin(ra_back),
                         math.sin(dec_back)])
        closure_rad = ang_sep(v_in, v_bk)

        cases.append({
            "jd_ut1": jd_ut1,
            "jd_tt": jd_tt,
            "ra_rad": ra_rad,
            "dec_rad": dec_rad,
            "obs_lon_rad": obs_lon,
            "obs_lat_rad": obs_lat,
            "az_rad": float(az),
            "alt_rad": float(alt),
            "closure_rad": closure_rad,
        })

    result = {
        "experiment": "equ_horizontal",
        "library": "astropy",
        "model": "eraHd2ae_GAST (via erfa)",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_solar_position(lines_iter):
    """Sun geocentric RA/Dec via erfa.epv00."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())
    cases = []
    for i in range(n):
        jd_tt = float(next(lines_iter).strip())
        ra, dec, dist_au = sun_ra_dec_dist_from_epv00(jd_tt, erfa)

        cases.append({
            "jd_tt": jd_tt,
            "ra_rad": ra,
            "dec_rad": dec,
            "dist_au": dist_au,
        })

    result = {
        "experiment": "solar_position",
        "library": "astropy",
        "model": "ERFA_epv00_analytic (via erfa)",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_lunar_position(lines_iter):
    """Moon geocentric RA/Dec — simplified Meeus Ch.47."""
    n = int(next(lines_iter).strip())
    cases = []
    for i in range(n):
        jd_tt = float(next(lines_iter).strip())
        ra, dec, dist_km = moon_ra_dec_dist_meeus(jd_tt)

        cases.append({
            "jd_tt": jd_tt,
            "ra_rad": ra,
            "dec_rad": dec,
            "dist_km": dist_km,
        })

    result = {
        "experiment": "lunar_position",
        "library": "astropy",
        "model": "Meeus_Ch47_simplified",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_kepler_solver(lines_iter):
    """Kepler equation solver — Newton-Raphson."""
    n = int(next(lines_iter).strip())
    cases = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        M_rad = float(parts[0])
        e = float(parts[1])

        E, iters, converged = solve_kepler_newton(M_rad, e, max_iter=100, tol=1e-15)

        nu = 2.0 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2),
            math.sqrt(1 - e) * math.cos(E / 2),
        )
        residual = abs(E - e * math.sin(E) - M_rad)

        cases.append({
            "M_rad": M_rad,
            "e": e,
            "E_rad": E,
            "nu_rad": nu,
            "residual_rad": residual,
            "iters": iters,
            "converged": converged,
        })

    result = {
        "experiment": "kepler_solver",
        "library": "astropy",
        "model": "Newton_Raphson",
        "count": n,
        "cases": cases,
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_gmst_era_perf(lines_iter):
    """Performance measurement for GMST/ERA computation."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    jd_ut1_arr = []
    jd_tt_arr = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jd_ut1_arr.append(float(parts[0]))
        jd_tt_arr.append(float(parts[1]))

    # Warm-up
    for i in range(min(n, 100)):
        erfa.gmst06(2451545.0, jd_ut1_arr[i] - 2451545.0,
                    2451545.0, jd_tt_arr[i] - 2451545.0)

    # Timed run
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        gmst = erfa.gmst06(2451545.0, jd_ut1_arr[i] - 2451545.0,
                          2451545.0, jd_tt_arr[i] - 2451545.0)
        sink += gmst
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "gmst_era_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_equ_ecl_perf(lines_iter):
    """Performance measurement for equatorial-ecliptic transform."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    jds = []
    ras = []
    decs = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        jds.append(float(parts[0]))
        ras.append(float(parts[1]))
        decs.append(float(parts[2]))

    # Warm-up: match measured operation to functional path (eqec06).
    for i in range(min(n, 100)):
        erfa.eqec06(2451545.0, jds[i] - 2451545.0, ras[i], decs[i])

    # Timed run
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        ecl_lon, _ecl_lat = erfa.eqec06(2451545.0, jds[i] - 2451545.0, ras[i], decs[i])
        sink += ecl_lon
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "equ_ecl_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_equ_horizontal_perf(lines_iter):
    """Performance measurement for equatorial-horizontal transform."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    params = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        params.append((
            float(parts[0]), float(parts[1]), float(parts[2]),
            float(parts[3]), float(parts[4]), float(parts[5])
        ))

    # Warm-up
    for i in range(min(n, 100)):
        jd_ut1, jd_tt, ra, dec, lon, lat = params[i]
        gast = erfa.gst06a(2451545.0, jd_ut1 - 2451545.0,
                           2451545.0, jd_tt - 2451545.0)
        ha = (gast + lon - ra) % (2 * math.pi)
        az, el = erfa.hd2ae(ha, dec, lat)

    # Timed run
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        jd_ut1, jd_tt, ra, dec, lon, lat = params[i]
        gast = erfa.gst06a(2451545.0, jd_ut1 - 2451545.0,
                           2451545.0, jd_tt - 2451545.0)
        ha = (gast + lon - ra) % (2 * math.pi)
        az, el = erfa.hd2ae(ha, dec, lat)
        sink += az
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "equ_horizontal_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_solar_position_perf(lines_iter):
    """Performance measurement for solar position computation."""
    try:
        import erfa
    except ImportError:
        import astropy._erfa as erfa

    n = int(next(lines_iter).strip())

    jds = []
    for i in range(n):
        jds.append(float(next(lines_iter).strip()))

    # Warm-up
    for i in range(min(n, 100)):
        sun_ra_dec_dist_from_epv00(jds[i], erfa)

    # Timed run — perf contract: compute RA + Dec + distance
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        ra, dec, dist = sun_ra_dec_dist_from_epv00(jds[i], erfa)
        sink += ra + dec + dist
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "solar_position_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_lunar_position_perf(lines_iter):
    """Performance measurement for lunar position computation."""
    n = int(next(lines_iter).strip())

    jds = []
    for i in range(n):
        jds.append(float(next(lines_iter).strip()))

    # Warm-up
    for i in range(min(n, 100)):
        moon_ra_dec_dist_meeus(jds[i])

    # Timed run — perf contract: compute RA + Dec + distance
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        ra, dec, dist_km = moon_ra_dec_dist_meeus(jds[i])
        sink += ra + dec + dist_km
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "lunar_position_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def run_kepler_solver_perf(lines_iter):
    """Performance measurement for Kepler solver."""
    n = int(next(lines_iter).strip())

    m_arr = []
    e_arr = []
    for i in range(n):
        parts = next(lines_iter).strip().split()
        m_arr.append(float(parts[0]))
        e_arr.append(float(parts[1]))

    # Warm-up
    for i in range(min(n, 100)):
        solve_kepler_newton(m_arr[i], e_arr[i], max_iter=100, tol=1e-15)

    # Timed run
    t0 = time.perf_counter_ns()
    sink = 0.0
    for i in range(n):
        E, _iters, _conv = solve_kepler_newton(m_arr[i], e_arr[i], max_iter=100, tol=1e-15)
        sink += E
    elapsed_ns = time.perf_counter_ns() - t0

    result = {
        "experiment": "kepler_solver_perf",
        "library": "astropy",
        "count": n,
        "total_ns": elapsed_ns,
        "per_op_ns": elapsed_ns / n,
        "throughput_ops_s": n / (elapsed_ns * 1e-9),
        "_sink": float(sink),
    }
    json.dump(result, sys.stdout, indent=None)
    print()


def main():
    lines_iter = iter(sys.stdin)
    experiment = next(lines_iter).strip()

    dispatch = {
        "frame_rotation_bpn": run_frame_rotation_bpn,
        "gmst_era": run_gmst_era,
        "equ_ecl": run_equ_ecl,
        "equ_horizontal": run_equ_horizontal,
        "solar_position": run_solar_position,
        "lunar_position": run_lunar_position,
        "kepler_solver": run_kepler_solver,
        "frame_rotation_bpn_perf": run_frame_rotation_bpn_perf,
        "gmst_era_perf": run_gmst_era_perf,
        "equ_ecl_perf": run_equ_ecl_perf,
        "equ_horizontal_perf": run_equ_horizontal_perf,
        "solar_position_perf": run_solar_position_perf,
        "lunar_position_perf": run_lunar_position_perf,
        "kepler_solver_perf": run_kepler_solver_perf,
    }

    if experiment not in dispatch:
        print(f"Unknown experiment: {experiment}", file=sys.stderr)
        sys.exit(1)

    dispatch[experiment](lines_iter)


if __name__ == "__main__":
    main()
