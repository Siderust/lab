#!/usr/bin/env python3
"""
Siderust Lab Orchestrator
=========================

Generates inputs, runs adapters (ERFA, Siderust, Astropy, libnova), collects results,
computes accuracy & performance metrics, and writes structured output.

Usage:
    python3 pipeline/orchestrator.py [--experiment frame_rotation_bpn] [--n 1000] [--seed 42]

Experiments:
    frame_rotation_bpn  — Bias-Precession-Nutation direction transform (ICRS → TrueOfDate/CIRS)
    gmst_era            — Greenwich Mean Sidereal Time & Earth Rotation Angle
    equ_ecl             — Equatorial ↔ Ecliptic coordinate transform
    equ_horizontal      — Equatorial → Horizontal (AltAz)
    solar_position      — Sun geocentric RA/Dec
    lunar_position      — Moon geocentric RA/Dec
    kepler_solver       — Kepler equation M→E→ν self-consistency
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
ERFA_BIN = LAB_ROOT / "pipeline" / "adapters" / "erfa_adapter" / "build" / "erfa_adapter"
SIDERUST_BIN = LAB_ROOT / "pipeline" / "adapters" / "siderust_adapter" / "target" / "release" / "siderust-adapter"
ASTROPY_SCRIPT = LAB_ROOT / "pipeline" / "adapters" / "astropy_adapter" / "adapter.py"
LIBNOVA_BIN = LAB_ROOT / "pipeline" / "adapters" / "libnova_adapter" / "build" / "libnova_adapter"

RAD_TO_MAS = 180.0 / math.pi * 3600.0 * 1000.0  # radians → milli-arcseconds
RAD_TO_ARCSEC = 180.0 / math.pi * 3600.0

# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def generate_frame_rotation_inputs(n: int, seed: int):
    """
    Generate test inputs for the frame_rotation_bpn experiment.

    Returns:
        epochs: array of JD(TT) values
        directions: Nx3 array of unit vectors in ICRS
        case_labels: list of human-readable case labels
    """
    rng = np.random.default_rng(seed)

    # --- Epochs ---
    # Typical dates: 2000–2030
    typical_epochs = 2451545.0 + rng.uniform(0, 30 * 365.25, size=max(1, n - 10))

    # Edge-case epochs
    edge_epochs = np.array([
        2451545.0,            # J2000.0 exactly
        2451545.5,            # J2000.0 + 0.5 day
        2451179.5,            # 1999-01-01 (near J2000)
        2415020.0,            # ~1900 (wide range)
        2488069.5,            # ~2100 (wide range)
        2457754.5,            # 2017-01-01 (leap second on 2016-12-31)
        2453736.5,            # 2006-01-01 (IAU 2006 precession epoch)
        2459580.5,            # 2022-02-01 (recent)
        2460310.5,            # 2024-01-01
        2460676.5,            # 2025-01-01
    ])

    epochs = np.concatenate([typical_epochs, edge_epochs])[:n]
    np.random.default_rng(seed).shuffle(epochs)

    # --- Directions ---
    # Random unit vectors on the sphere
    random_dirs = rng.standard_normal((max(1, n - 6), 3))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    # Named directions (edge cases)
    edge_dirs = np.array([
        [1.0, 0.0, 0.0],            # +X (vernal equinox direction)
        [0.0, 1.0, 0.0],            # +Y
        [0.0, 0.0, 1.0],            # +Z (north celestial pole)
        [0.0, 0.0, -1.0],           # -Z (south celestial pole)
        [1.0/math.sqrt(3)] * 3,     # diagonal
        [-1.0/math.sqrt(3)] * 3,    # anti-diagonal
    ])

    directions = np.vstack([random_dirs, edge_dirs])[:n]

    # Labels
    labels = [f"random_{i}" for i in range(max(0, n - 6))]
    labels += ["equinox_+X", "+Y", "north_pole_+Z", "south_pole_-Z", "diagonal", "anti_diagonal"]
    labels = labels[:n]

    return epochs, directions, labels


def generate_gmst_era_inputs(n: int, seed: int):
    """Generate test inputs for the gmst_era experiment."""
    rng = np.random.default_rng(seed)

    # JD(UT1) values: spread over 2000–2030
    typical_ut1 = 2451545.0 + rng.uniform(0, 30 * 365.25, size=max(1, n - 5))

    edge_ut1 = np.array([
        2451545.0,   # J2000.0
        2451179.5,   # 1999-01-01
        2415020.0,   # ~1900
        2488069.5,   # ~2100
        2457754.5,   # 2017-01-01
    ])

    jd_ut1 = np.concatenate([typical_ut1, edge_ut1])[:n]

    # TT ≈ UT1 + ΔT; for simplicity use ΔT ≈ 69.184s / 86400 ≈ 0.0008 days
    delta_t_days = 69.184 / 86400.0
    jd_tt = jd_ut1 + delta_t_days

    return jd_ut1, jd_tt


def generate_equ_ecl_inputs(n: int, seed: int):
    """Generate inputs for equatorial ↔ ecliptic transform experiment.

    Returns (epochs, ra, dec) — all in radians.
    """
    rng = np.random.default_rng(seed)

    # Epochs: 2000–2100
    typical = 2451545.0 + rng.uniform(0, 100 * 365.25, size=max(1, n - 6))
    edge = np.array([
        2451545.0,    # J2000
        2415020.0,    # ~1900
        2488069.5,    # ~2100
        2460676.5,    # 2025
        2453736.5,    # 2006 (IAU 2006 epoch)
        2460310.5,    # 2024
    ])
    epochs = np.concatenate([typical, edge])[:n]

    # RA ∈ [0, 2π), Dec ∈ [-π/2, π/2]
    ra = rng.uniform(0, 2 * np.pi, size=n)
    dec = np.arcsin(rng.uniform(-1, 1, size=n))  # uniform on sphere

    # Inject edge cases in the first few slots
    if n >= 4:
        ra[0], dec[0] = 0.0, 0.0        # vernal equinox
        ra[1], dec[1] = np.pi, 0.0       # autumnal equinox
        ra[2], dec[2] = 0.0, np.pi / 2   # north pole
        ra[3], dec[3] = 0.0, -np.pi / 2  # south pole

    return epochs, ra, dec


def generate_equ_horizontal_inputs(n: int, seed: int):
    """Generate inputs for equatorial → horizontal coordinate transform.

    Returns (jd_ut1, jd_tt, ra, dec, observer_lon, observer_lat) — all radians.
    """
    rng = np.random.default_rng(seed)

    jd_ut1 = 2451545.0 + rng.uniform(0, 30 * 365.25, size=n)
    delta_t_days = 69.184 / 86400.0
    jd_tt = jd_ut1 + delta_t_days

    ra = rng.uniform(0, 2 * np.pi, size=n)
    dec = np.arcsin(rng.uniform(-1, 1, size=n))

    # Observer locations: lon ∈ [-π, π], lat ∈ [-π/2, π/2]
    lon = rng.uniform(-np.pi, np.pi, size=n)
    lat = np.arcsin(rng.uniform(-1, 1, size=n))

    # Inject known observatory locations (radians)
    if n >= 3:
        # Greenwich: lon=0, lat=51.4769°
        lon[0], lat[0] = 0.0, np.deg2rad(51.4769)
        # Cerro Paranal: lon=-70.4042°, lat=-24.6272°
        lon[1], lat[1] = np.deg2rad(-70.4042), np.deg2rad(-24.6272)
        # North pole observer
        lon[2], lat[2] = 0.0, np.pi / 2

    return jd_ut1, jd_tt, ra, dec, lon, lat


def generate_solar_position_inputs(n: int, seed: int):
    """Generate epoch inputs for Sun position experiment.

    Returns array of JD(TT) values spanning 1900–2100.
    """
    rng = np.random.default_rng(seed)

    # JD range: ~1900 (2415020) to ~2100 (2488069)
    typical = rng.uniform(2415020.0, 2488069.5, size=max(1, n - 6))
    edge = np.array([
        2451545.0,    # J2000
        2451179.5,    # 1999-01-01
        2460310.5,    # 2024-01-01
        2451625.0,    # ~2000 March equinox
        2451716.0,    # ~2000 June solstice
        2451900.0,    # ~2000 Dec solstice
    ])
    return np.concatenate([typical, edge])[:n]


def generate_lunar_position_inputs(n: int, seed: int):
    """Generate epoch inputs for Moon position experiment.

    Returns array of JD(TT) values spanning 1900–2100.
    """
    rng = np.random.default_rng(seed)

    typical = rng.uniform(2415020.0, 2488069.5, size=max(1, n - 6))
    edge = np.array([
        2451545.0,    # J2000
        2451550.0,    # near J2000
        2460310.5,    # 2024-01-01
        2459580.5,    # 2022-02-01
        2451179.5,    # 1999-01-01
        2488069.5,    # ~2100
    ])
    return np.concatenate([typical, edge])[:n]


def generate_kepler_inputs(n: int, seed: int):
    """Generate inputs for Kepler's equation experiment.

    Returns (M_array, e_array) — M in radians, e in [0, 1).
    """
    rng = np.random.default_rng(seed)

    # Specific eccentricities to probe accuracy across the range
    fixed_ecc = np.array([0.0, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.999999])

    M_list = []
    e_list = []

    # For each fixed eccentricity, generate several M values
    per_ecc = max(1, (n - 10) // len(fixed_ecc))
    for ecc in fixed_ecc:
        M_vals = rng.uniform(0, 2 * np.pi, size=per_ecc)
        M_list.extend(M_vals)
        e_list.extend([ecc] * per_ecc)

    # Fill remaining with random e ∈ [0, 1)
    remaining = max(0, n - len(M_list))
    if remaining > 0:
        M_list.extend(rng.uniform(0, 2 * np.pi, size=remaining))
        e_list.extend(rng.uniform(0, 0.999, size=remaining))

    M_arr = np.array(M_list[:n])
    e_arr = np.array(e_list[:n])

    # Edge cases: M = 0, M = π
    if n >= 2:
        M_arr[0], e_arr[0] = 0.0, 0.5
        M_arr[1], e_arr[1] = np.pi, 0.5

    return M_arr, e_arr


# ---------------------------------------------------------------------------
# Adapter runners
# ---------------------------------------------------------------------------

def run_adapter(cmd, input_text: str, label: str) -> dict:
    """Run an adapter process, return parsed JSON output."""
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print(f"  ⚠ {label}: binary not found ({cmd[0]}), skipping.", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠ {label}: timed out after 120s, skipping.", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  ⚠ {label}: exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"     stderr: {result.stderr[:500]}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  ⚠ {label}: JSON parse error: {e}", file=sys.stderr)
        print(f"     stdout[:500]: {result.stdout[:500]}", file=sys.stderr)
        return None


def format_bpn_input(epochs, directions):
    """Format input for the frame_rotation_bpn experiment."""
    lines = ["frame_rotation_bpn", str(len(epochs))]
    for jd, d in zip(epochs, directions):
        lines.append(f"{jd:.15f} {d[0]:.17e} {d[1]:.17e} {d[2]:.17e}")
    return "\n".join(lines) + "\n"


def format_gmst_input(jd_ut1, jd_tt):
    """Format input for the gmst_era experiment."""
    lines = ["gmst_era", str(len(jd_ut1))]
    for u, t in zip(jd_ut1, jd_tt):
        lines.append(f"{u:.15f} {t:.15f}")
    return "\n".join(lines) + "\n"


def format_bpn_perf_input(epochs, directions):
    """Format input for the BPN performance experiment."""
    lines = ["frame_rotation_bpn_perf", str(len(epochs))]
    for jd, d in zip(epochs, directions):
        lines.append(f"{jd:.15f} {d[0]:.17e} {d[1]:.17e} {d[2]:.17e}")
    return "\n".join(lines) + "\n"


def format_gmst_perf_input(jd_ut1, jd_tt):
    """Format input for the GMST/ERA performance experiment."""
    lines = ["gmst_era_perf", str(len(jd_ut1))]
    for u, t in zip(jd_ut1, jd_tt):
        lines.append(f"{u:.15f} {t:.15f}")
    return "\n".join(lines) + "\n"


def format_equ_ecl_perf_input(epochs, ra, dec):
    """Format input for equatorial-ecliptic performance experiment."""
    lines = ["equ_ecl_perf", str(len(epochs))]
    for jd, r, d in zip(epochs, ra, dec):
        lines.append(f"{jd:.15f} {r:.17e} {d:.17e}")
    return "\n".join(lines) + "\n"


def format_equ_horizontal_perf_input(jd_ut1, jd_tt, ra, dec, lon, lat):
    """Format input for equatorial-horizontal performance experiment."""
    lines = ["equ_horizontal_perf", str(len(jd_ut1))]
    for u, t, r, d, lo, la in zip(jd_ut1, jd_tt, ra, dec, lon, lat):
        lines.append(f"{u:.15f} {t:.15f} {r:.17e} {d:.17e} {lo:.17e} {la:.17e}")
    return "\n".join(lines) + "\n"


def format_solar_position_perf_input(epochs):
    """Format input for solar position performance experiment."""
    lines = ["solar_position_perf", str(len(epochs))]
    for jd in epochs:
        lines.append(f"{jd:.15f}")
    return "\n".join(lines) + "\n"


def format_lunar_position_perf_input(epochs):
    """Format input for lunar position performance experiment."""
    lines = ["lunar_position_perf", str(len(epochs))]
    for jd in epochs:
        lines.append(f"{jd:.15f}")
    return "\n".join(lines) + "\n"


def format_kepler_perf_input(M_arr, e_arr):
    """Format input for Kepler solver performance experiment."""
    lines = ["kepler_solver_perf", str(len(M_arr))]
    for m, e in zip(M_arr, e_arr):
        lines.append(f"{m:.17e} {e:.17e}")
    return "\n".join(lines) + "\n"


def format_equ_ecl_input(epochs, ra, dec):
    """Format input for equ_ecl experiment: jd_tt ra dec per line."""
    lines = ["equ_ecl", str(len(epochs))]
    for jd, r, d in zip(epochs, ra, dec):
        lines.append(f"{jd:.15f} {r:.17e} {d:.17e}")
    return "\n".join(lines) + "\n"


def format_equ_horizontal_input(jd_ut1, jd_tt, ra, dec, lon, lat):
    """Format input for equ_horizontal experiment: jd_ut1 jd_tt ra dec lon lat per line."""
    lines = ["equ_horizontal", str(len(jd_ut1))]
    for u, t, r, d, lo, la in zip(jd_ut1, jd_tt, ra, dec, lon, lat):
        lines.append(f"{u:.15f} {t:.15f} {r:.17e} {d:.17e} {lo:.17e} {la:.17e}")
    return "\n".join(lines) + "\n"


def format_solar_position_input(epochs):
    """Format input for solar_position experiment: jd_tt per line."""
    lines = ["solar_position", str(len(epochs))]
    for jd in epochs:
        lines.append(f"{jd:.15f}")
    return "\n".join(lines) + "\n"


def format_lunar_position_input(epochs):
    """Format input for lunar_position experiment: jd_tt per line."""
    lines = ["lunar_position", str(len(epochs))]
    for jd in epochs:
        lines.append(f"{jd:.15f}")
    return "\n".join(lines) + "\n"


def format_kepler_input(M_arr, e_arr):
    """Format input for kepler_solver experiment: M e per line."""
    lines = ["kepler_solver", str(len(M_arr))]
    for m, e in zip(M_arr, e_arr):
        lines.append(f"{m:.17e} {e:.17e}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_accuracy_metrics(ref_cases, cand_cases, ref_label, cand_label):
    """
    Compare candidate adapter results to reference adapter results.

    Metrics:
        angular_error_mas: angular separation between output directions
        closure_error_rad: self-consistency (A→B→A) error
        matrix_frobenius:  Frobenius norm of matrix difference

    Returns a dict of computed metrics.
    """
    angular_errors_mas = []
    closure_errors_rad = []
    matrix_frob_norms = []
    nan_count = 0
    inf_count = 0
    worst_cases = []

    for ref_c, cand_c in zip(ref_cases, cand_cases):
        ref_out = np.array(ref_c["output"])
        cand_out = np.array(cand_c["output"])

        # Check for NaN/Inf
        if np.any(np.isnan(cand_out)):
            nan_count += 1
            continue
        if np.any(np.isinf(cand_out)):
            inf_count += 1
            continue

        # Angular error
        dot = np.clip(np.dot(ref_out, cand_out), -1.0, 1.0)
        ang_err_rad = math.acos(dot)
        ang_err_mas = ang_err_rad * RAD_TO_MAS
        angular_errors_mas.append(ang_err_mas)

        # Closure error from candidate
        closure_errors_rad.append(cand_c.get("closure_rad", 0.0))

        # Matrix difference (if both have non-null matrices)
        if ref_c.get("matrix") is not None and cand_c.get("matrix") is not None:
            ref_m = np.array(ref_c["matrix"])
            cand_m = np.array(cand_c["matrix"])
            frob = np.linalg.norm(ref_m - cand_m)
            matrix_frob_norms.append(frob)

        worst_cases.append({
            "jd_tt": cand_c["jd_tt"],
            "angular_error_mas": ang_err_mas,
        })

    angular_errors_mas = np.array(angular_errors_mas)
    closure_errors_rad = np.array(closure_errors_rad)

    # Sort worst cases
    worst_cases.sort(key=lambda x: x["angular_error_mas"], reverse=True)

    def percentiles(arr):
        if len(arr) == 0:
            return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None, "min": None, "mean": None, "rms": None}
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "rms": float(np.sqrt(np.mean(arr**2))),
        }

    return {
        "reference": ref_label,
        "candidate": cand_label,
        "angular_error_mas": percentiles(angular_errors_mas),
        "closure_error_rad": percentiles(closure_errors_rad),
        "matrix_frobenius": percentiles(np.array(matrix_frob_norms)) if matrix_frob_norms else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "worst_cases": worst_cases[:10],
    }


def compute_gmst_accuracy(ref_cases, cand_cases, ref_label, cand_label):
    """Compare GMST/ERA values between reference and candidate."""
    gmst_errors_rad = []
    era_errors_rad = []

    for ref_c, cand_c in zip(ref_cases, cand_cases):
        if "gmst_rad" in ref_c and "gmst_rad" in cand_c:
            err = abs(ref_c["gmst_rad"] - cand_c["gmst_rad"])
            gmst_errors_rad.append(err)

        if "era_rad" in ref_c and "era_rad" in cand_c:
            err = abs(ref_c["era_rad"] - cand_c["era_rad"])
            era_errors_rad.append(err)

    def percentiles(arr):
        if len(arr) == 0:
            return {"p50": None, "p90": None, "p99": None, "max": None}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "rms": float(np.sqrt(np.mean(a**2))),
        }

    return {
        "reference": ref_label,
        "candidate": cand_label,
        "gmst_error_rad": percentiles(gmst_errors_rad),
        "gmst_error_arcsec": percentiles([e * RAD_TO_ARCSEC for e in gmst_errors_rad]),
        "era_error_rad": percentiles(era_errors_rad),
    }


def angular_separation(ra1, dec1, ra2, dec2):
    """Compute angular separation between two (RA, Dec) pairs in radians."""
    cos_sep = (math.sin(dec1) * math.sin(dec2)
               + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    return math.acos(max(-1.0, min(1.0, cos_sep)))


def compute_angular_accuracy(ref_cases, cand_cases, ref_label, cand_label,
                             ra_key="ra_rad", dec_key="dec_rad",
                             extra_keys=None):
    """Compare RA/Dec angular positions between reference and candidate.

    Works for equ_ecl (lon/lat), solar/lunar position, etc.
    extra_keys: optional list of keys to also compute absolute difference stats.
    """
    sep_errors_arcsec = []
    signed_ra_errors = []
    signed_dec_errors = []
    extra_diffs = {k: [] for k in (extra_keys or [])}
    nan_count = 0

    for ref_c, cand_c in zip(ref_cases, cand_cases):
        r_ra = ref_c.get(ra_key)
        r_dec = ref_c.get(dec_key)
        c_ra = cand_c.get(ra_key)
        c_dec = cand_c.get(dec_key)

        if any(v is None for v in [r_ra, r_dec, c_ra, c_dec]):
            nan_count += 1
            continue
        if any(math.isnan(v) or math.isinf(v) for v in [c_ra, c_dec]):
            nan_count += 1
            continue

        sep = angular_separation(r_ra, r_dec, c_ra, c_dec)
        sep_errors_arcsec.append(sep * RAD_TO_ARCSEC)

        # Signed errors for bias detection
        dra = c_ra - r_ra
        # wrap RA difference to [-π, π]
        if dra > math.pi:
            dra -= 2 * math.pi
        elif dra < -math.pi:
            dra += 2 * math.pi
        signed_ra_errors.append(dra * RAD_TO_ARCSEC)
        signed_dec_errors.append((c_dec - r_dec) * RAD_TO_ARCSEC)

        for k in (extra_keys or []):
            rv = ref_c.get(k)
            cv = cand_c.get(k)
            if rv is not None and cv is not None:
                extra_diffs[k].append(cv - rv)

    def percentiles(arr):
        if len(arr) == 0:
            return {"p50": None, "p90": None, "p99": None, "max": None, "mean": None, "rms": None}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(np.abs(a), 50)),
            "p90": float(np.percentile(np.abs(a), 90)),
            "p99": float(np.percentile(np.abs(a), 99)),
            "max": float(np.max(np.abs(a))),
            "mean": float(np.mean(a)),
            "rms": float(np.sqrt(np.mean(a**2))),
        }

    result = {
        "reference": ref_label,
        "candidate": cand_label,
        "angular_sep_arcsec": percentiles(sep_errors_arcsec),
        "signed_ra_error_arcsec": percentiles(signed_ra_errors),
        "signed_dec_error_arcsec": percentiles(signed_dec_errors),
        "nan_count": nan_count,
    }

    for k in (extra_keys or []):
        result[f"{k}_diff"] = percentiles(extra_diffs[k])

    return result


def compute_kepler_accuracy(ref_cases, cand_cases, ref_label, cand_label):
    """Compare Kepler solver results: E and ν residuals, plus self-consistency."""
    def wrap_to_pi(angle_rad: float) -> float:
        """Wrap an angle to [-pi, pi] in constant time."""
        return ((angle_rad + math.pi) % (2.0 * math.pi)) - math.pi

    E_errors_rad = []
    nu_errors_rad = []
    consistency_errors = []
    nan_count = 0

    for ref_c, cand_c in zip(ref_cases, cand_cases):
        r_E = ref_c.get("E_rad")
        c_E = cand_c.get("E_rad")
        r_nu = ref_c.get("nu_rad")
        c_nu = cand_c.get("nu_rad")

        if any(v is None for v in [r_E, c_E, r_nu, c_nu]):
            nan_count += 1
            continue
        if any(not math.isfinite(v) for v in [r_E, c_E, r_nu, c_nu]):
            nan_count += 1
            continue

        # Wrap differences to [-π, π] since E and ν are defined mod 2π.
        E_errors_rad.append(wrap_to_pi(c_E - r_E))

        nu_errors_rad.append(wrap_to_pi(c_nu - r_nu))

        # Self-consistency: M = E - e*sin(E) should hold
        M_input = cand_c.get("M_rad", ref_c.get("M_rad", 0.0))
        e = cand_c.get("e", ref_c.get("e", 0.0))
        M_recon = c_E - e * math.sin(c_E)
        # Wrap to [0, 2π)
        M_input_w = M_input % (2 * math.pi)
        M_recon_w = M_recon % (2 * math.pi)
        consistency = abs(M_input_w - M_recon_w)
        if consistency > math.pi:
            consistency = 2 * math.pi - consistency
        consistency_errors.append(consistency)

    def percentiles(arr):
        if len(arr) == 0:
            return {"p50": None, "p90": None, "p99": None, "max": None, "mean": None, "rms": None}
        a = np.array(arr)
        return {
            "p50": float(np.percentile(np.abs(a), 50)),
            "p90": float(np.percentile(np.abs(a), 90)),
            "p99": float(np.percentile(np.abs(a), 99)),
            "max": float(np.max(np.abs(a))),
            "mean": float(np.mean(a)),
            "rms": float(np.sqrt(np.mean(a**2))),
        }

    return {
        "reference": ref_label,
        "candidate": cand_label,
        "E_error_rad": percentiles(E_errors_rad),
        "nu_error_rad": percentiles(nu_errors_rad),
        "consistency_error_rad": percentiles(consistency_errors),
        "nan_count": nan_count,
    }


# ---------------------------------------------------------------------------
# Alignment checklist
# ---------------------------------------------------------------------------

def alignment_checklist(experiment: str, mode: str = "common_denominator"):
    """Return the alignment checklist for this run."""
    base = {
        "units": {
            "angles": "radians (internal), mas for error reporting",
            "distances": "meters",
            "float_type": "f64",
        },
        "time_input": "JD (Julian Date), TT scale for precession/nutation, UT1 for sidereal time",
        "time_scales": "TT for BPN matrix; UT1≈TT-69.184s simplified",
        "leap_seconds": "not applicable (JD input, no UTC conversion in this experiment)",
        "earth_orientation": {
            "ut1_minus_utc": "not used (JD(TT) input)",
            "polar_motion_xp_yp": "zero (not applied)",
            "eop_mode": "disabled",
        },
        "geodesy": "not applicable (direction-only experiment)",
        "refraction": "disabled",
        "ephemeris_source": "not applicable (no aberration/parallax)",
    }

    if experiment == "frame_rotation_bpn":
        base["models"] = {
            "erfa": "IAU 2006/2000A bias-precession-nutation (eraPnm06a)",
            "siderust": "IERS 2003 frame bias + Meeus precession (ζ,z,θ) + IAU 1980 nutation (63 terms)",
            "astropy": "IAU 2006/2000A via bundled ERFA (erfa.pnm06a)",
            "libnova": "Meeus precession (ζ,z,θ Equ 20.3) + IAU 1980 nutation (63-term Table 21A), applied as RA/Dec corrections (no BPN matrix)",
        }
        base["mode"] = mode
        base["note"] = (
            "ERFA and Astropy use the same IAU 2006/2000A model (reference). "
            "Siderust uses Meeus precession + IAU 1980 nutation (lower fidelity). "
            "libnova uses Meeus precession + IAU 1980 nutation via coordinate-level API (no rotation matrix). "
            "Differences measure the model gap, not implementation bugs."
        )
    elif experiment == "gmst_era":
        base["models"] = {
            "erfa": "GMST=IAU2006 (eraGmst06), ERA=IAU2000 (eraEra00)",
            "siderust": "GST polynomial (IAU 2006 coefficients), ERA from IERS definition",
            "astropy": "GMST=IAU2006, ERA=IAU2000 via bundled ERFA",
            "libnova": "GMST=Meeus Formula 11.4, GAST=MST+nutation correction (no ERA)",
        }
        base["mode"] = mode

    elif experiment == "equ_ecl":
        base["models"] = {
            "erfa": "IAU 2006 obliquity-based transform (eraEqec06 / eraEceq06)",
            "siderust": "IAU 2006 obliquity, Ecliptic frame via Transform trait",
            "astropy": "IAU 2006 via bundled ERFA (erfa.eqec06 / erfa.eceq06)",
            "libnova": "Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ / ln_get_equ_from_ecl",
        }
        base["note"] = (
            "ERFA and Astropy share the same IAU 2006 obliquity model (reference). "
            "Siderust uses IAU 2006 obliquity with its own Ecliptic frame implementation. "
            "libnova uses Meeus obliquity polynomial — expect ~arcsec-level differences."
        )

    elif experiment == "equ_horizontal":
        base["models"] = {
            "erfa": "Spherical trig via eraHd2ae / eraAe2hd; GAST via eraGst06a; no refraction",
            "siderust": "Manual spherical trig matching ERFA formula; GAST from GST polynomial",
            "astropy": "eraHd2ae / eraAe2hd via bundled ERFA; GAST via eraGst06a",
            "libnova": "ln_get_hrz_from_equ / ln_get_equ_from_hrz; convention fix: az_erfa = (360 - az_ln + 180) % 360",
        }
        base["note"] = (
            "Azimuth convention: ERFA 0°=North CW; libnova 0°=South. "
            "All adapters use the same spherical trig, differences arise from GAST model. "
            "No atmospheric refraction applied."
        )
        base["refraction"] = "disabled"

    elif experiment == "solar_position":
        base["models"] = {
            "erfa": "VSOP87 via eraEpv00: heliocentric Earth → geocentric Sun (negate); BCRS equatorial output",
            "siderust": "VSOP87 via Sun::get_apparent_geocentric_equ (includes aberration + FK5)",
            "astropy": "VSOP87 via erfa.epv00 (same as ERFA)",
            "libnova": "VSOP87 via ln_get_solar_equ_coords (different truncation/corrections)",
        }
        base["ephemeris_source"] = "VSOP87 (analytic, all libraries)"
        base["note"] = (
            "ERFA epv00 returns BCRS-aligned equatorial (no obliquity rotation). "
            "Differences reflect VSOP87 truncation levels and aberration correction details."
        )

    elif experiment == "lunar_position":
        base["models"] = {
            "erfa": "Simplified Meeus Ch.47 (major terms only, ~10' accuracy)",
            "siderust": "ELP 2000 via Moon::get_apparent_topocentric_equ with site at (0,0,0)",
            "astropy": "Simplified Meeus Ch.47 (same algorithm as ERFA adapter)",
            "libnova": "ELP 2000-82B via ln_get_lunar_equ_coords (full model)",
        }
        base["note"] = (
            "ERFA and Astropy use simplified Meeus (~10' accuracy) — for benchmarking only. "
            "Siderust and libnova use full ELP 2000 — expect ~arcmin-level differences vs reference. "
            "No dedicated ERFA Moon ephemeris exists; cross-library comparison is the primary metric."
        )
        base["ephemeris_source"] = "Meeus/ELP 2000 (varies by library)"

    elif experiment == "kepler_solver":
        base["models"] = {
            "erfa": "Newton-Raphson iteration (100 iters, tol 1e-15)",
            "siderust": "solve_keplers_equation (internal algorithm)",
            "astropy": "Newton-Raphson iteration in Python (100 iters, tol 1e-15)",
            "libnova": "Sinnott bisection via ln_solve_kepler (internal convergence ~1e-6 deg)",
        }
        base["note"] = (
            "Kepler's equation M = E - e*sin(E) is solved for E given (M, e). "
            "Self-consistency M_reconstructed = E - e*sin(E) is the primary metric. "
            "libnova uses a bisection method with lower convergence tolerance."
        )

    return base


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def get_git_sha(repo_path: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_metadata():
    """Collect environment metadata for reproducibility."""
    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_shas": {
            "lab": get_git_sha(LAB_ROOT),
            "siderust": get_git_sha(LAB_ROOT / "siderust"),
            "erfa": get_git_sha(LAB_ROOT / "erfa"),
            "libnova": get_git_sha(LAB_ROOT / "libnova"),
        },
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "toolchain": {
            "python": platform.python_version(),
        },
    }

    # Add rustc and cc versions if available
    for tool, cmd in [("rustc", ["rustc", "--version"]), ("cc", ["gcc", "--version"])]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                meta["toolchain"][tool] = r.stdout.strip().split("\n")[0]
        except Exception:
            pass

    return meta


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary_table(all_results: list) -> str:
    """Generate a Markdown summary table from all results."""

    def fmt(v, precision=2):
        if v is None:
            return "—"
        if precision == 'e':
            return f"{v:.2e}"
        return f"{v:.{precision}f}"

    # Separate by experiment type
    bpn_results = [r for r in all_results if r.get("experiment") == "frame_rotation_bpn"]
    gmst_results = [r for r in all_results if r.get("experiment") == "gmst_era"]

    lines = []

    if bpn_results:
        lines.append("### Frame Rotation (BPN: ICRS → True-of-Date)")
        lines.append("")
        lines.append("| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Matrix Frob p50 | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |")
        lines.append("|---------|---------------|---------------|---------------|-----------------|-------------------|--------------|----------------|")

        for r in bpn_results:
            lib = r.get("candidate_library", "?")
            acc = r.get("accuracy", {})
            ang = acc.get("angular_error_mas", {})
            clo = acc.get("closure_error_rad", {})
            mfr = acc.get("matrix_frobenius", {}) or {}
            perf = r.get("performance", {})
            ref_perf = r.get("reference_performance", {})
            speedup = "—"
            if perf.get("per_op_ns") and ref_perf.get("per_op_ns"):
                speedup = f"{ref_perf['per_op_ns'] / perf['per_op_ns']:.1f}×"

            lines.append(
                f"| {lib} "
                f"| {fmt(ang.get('p50'))} "
                f"| {fmt(ang.get('p99'))} "
                f"| {fmt(ang.get('max'))} "
                f"| {fmt(mfr.get('p50'), 2)} "
                f"| {fmt(clo.get('p99'), 2)} "
                f"| {fmt(perf.get('per_op_ns'), 1)} "
                f"| {speedup} |"
            )
        lines.append("")

    if gmst_results:
        lines.append("### GMST / ERA (Time Scales)")
        lines.append("")
        lines.append("| Library | GMST p50 (arcsec) | GMST p99 (arcsec) | GMST max (arcsec) | ERA p50 (rad) | ERA max (rad) |")
        lines.append("|---------|-------------------|-------------------|-------------------|---------------|---------------|")

        for r in gmst_results:
            lib = r.get("candidate_library", "?")
            acc = r.get("accuracy", {})
            gmst = acc.get("gmst_error_arcsec", {})
            era = acc.get("era_error_rad", {})

            lines.append(
                f"| {lib} "
                f"| {fmt(gmst.get('p50'), 6)} "
                f"| {fmt(gmst.get('p99'), 6)} "
                f"| {fmt(gmst.get('max'), 6)} "
                f"| {fmt(era.get('p50'), 10)} "
                f"| {fmt(era.get('max'), 10)} |"
            )
        lines.append("")

    # --- Angular experiments (equ_ecl, equ_horizontal, solar_position, lunar_position) ---
    for exp_name, title in [
        ("equ_ecl", "Equatorial ↔ Ecliptic Transform"),
        ("equ_horizontal", "Equatorial → Horizontal (AltAz)"),
        ("solar_position", "Sun Geocentric Position"),
        ("lunar_position", "Moon Geocentric Position"),
    ]:
        exp_results = [r for r in all_results if r.get("experiment") == exp_name]
        if not exp_results:
            continue

        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Library | Sep p50 (arcsec) | Sep p99 (arcsec) | Sep max (arcsec) | RA bias (arcsec) | Dec bias (arcsec) |")
        lines.append("|---------|------------------|------------------|------------------|------------------|-------------------|")

        for r in exp_results:
            lib = r.get("candidate_library", "?")
            acc = r.get("accuracy", {})
            sep = acc.get("angular_sep_arcsec", {})
            ra_b = acc.get("signed_ra_error_arcsec", {})
            dec_b = acc.get("signed_dec_error_arcsec", {})

            lines.append(
                f"| {lib} "
                f"| {fmt(sep.get('p50'), 4)} "
                f"| {fmt(sep.get('p99'), 4)} "
                f"| {fmt(sep.get('max'), 4)} "
                f"| {fmt(ra_b.get('mean'), 4)} "
                f"| {fmt(dec_b.get('mean'), 4)} |"
            )
        lines.append("")

    # --- Kepler solver ---
    kepler_results = [r for r in all_results if r.get("experiment") == "kepler_solver"]
    if kepler_results:
        lines.append("### Kepler Solver (M→E→ν)")
        lines.append("")
        lines.append("| Library | E p50 (rad) | E max (rad) | ν p50 (rad) | ν max (rad) | Consistency max (rad) |")
        lines.append("|---------|-------------|-------------|-------------|-------------|-----------------------|")

        for r in kepler_results:
            lib = r.get("candidate_library", "?")
            acc = r.get("accuracy", {})
            E_err = acc.get("E_error_rad", {})
            nu_err = acc.get("nu_error_rad", {})
            con = acc.get("consistency_error_rad", {})

            lines.append(
                f"| {lib} "
                f"| {fmt(E_err.get('p50'), 'e')} "
                f"| {fmt(E_err.get('max'), 'e')} "
                f"| {fmt(nu_err.get('p50'), 'e')} "
                f"| {fmt(nu_err.get('max'), 'e')} "
                f"| {fmt(con.get('max'), 'e')} |"
            )
        lines.append("")

    # --- Feature / Model Parity Matrix ---
    lines.append("### Feature / Model Parity Matrix")
    lines.append("")
    experiments_seen = sorted(set(r.get("experiment", "") for r in all_results))
    libs_seen = sorted(set(r.get("candidate_library", "") for r in all_results))
    all_libs = ["erfa"] + libs_seen  # erfa is always reference

    header = "| Experiment | " + " | ".join(all_libs) + " |"
    separator = "|------------|" + "|".join(["---"] * len(all_libs)) + "|"
    lines.append(header)
    lines.append(separator)

    # Model descriptions per (experiment, library) from alignment checklists
    for exp in experiments_seen:
        exp_r = [r for r in all_results if r.get("experiment") == exp]
        if not exp_r:
            continue
        alignment = exp_r[0].get("alignment", {})
        models = alignment.get("models", {})
        cells = []
        for lib in all_libs:
            model_str = models.get(lib, "—")
            # Truncate for table readability
            if len(model_str) > 50:
                model_str = model_str[:47] + "..."
            cells.append(model_str)
        lines.append(f"| {exp} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main experiment runners
# ---------------------------------------------------------------------------

def run_experiment_frame_rotation_bpn(n: int, seed: int, run_perf: bool = True):
    """
    Run the frame_rotation_bpn experiment end-to-end.

    Reference: ERFA (IAU 2006/2000A BPN matrix)
    Candidates: Siderust, Astropy, libnova
    """
    print(f"\n{'='*70}")
    print(f" Experiment: frame_rotation_bpn (N={n}, seed={seed})")
    print(f"{'='*70}")

    # 1) Generate inputs
    print("  Generating inputs...")
    epochs, directions, labels = generate_frame_rotation_inputs(n, seed)
    input_text = format_bpn_input(epochs, directions)

    # 2) Run adapters
    adapters = {}

    print("  Running ERFA adapter (reference)...")
    adapters["erfa"] = run_adapter([str(ERFA_BIN)], input_text, "erfa")

    print("  Running Siderust adapter...")
    adapters["siderust"] = run_adapter([str(SIDERUST_BIN)], input_text, "siderust")

    print("  Running Astropy adapter...")
    adapters["astropy"] = run_adapter(
        [sys.executable, str(ASTROPY_SCRIPT)], input_text, "astropy"
    )

    print("  Running libnova adapter...")
    adapters["libnova"] = run_adapter([str(LIBNOVA_BIN)], input_text, "libnova")

    # 3) Compute accuracy metrics
    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_accuracy_metrics(
            ref_data["cases"], cand_data["cases"], "erfa", lib
        )

        result = {
            "experiment": "frame_rotation_bpn",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("frame_rotation_bpn"),
            "inputs": {"count": n, "seed": seed,
                       "epoch_range": [f"JD {min(epochs):.1f}", f"JD {max(epochs):.1f}"]},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        }
        results.append(result)

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_bpn_perf_input(epochs[:perf_n], directions[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                # Find or create the result entry for this library
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Also time the reference (erfa) by itself
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_gmst_era(n: int, seed: int, run_perf: bool = True):
    """
    Run the gmst_era experiment end-to-end.

    Reference: ERFA (IAU 2006 GMST, IAU 2000 ERA)
    Candidates: Siderust, Astropy, libnova
    """
    print(f"\n{'='*70}")
    print(f" Experiment: gmst_era (N={n}, seed={seed})")
    print(f"{'='*70}")

    # 1) Generate inputs
    print("  Generating inputs...")
    jd_ut1, jd_tt = generate_gmst_era_inputs(n, seed)
    input_text = format_gmst_input(jd_ut1, jd_tt)

    # 2) Run adapters
    adapters = {}

    print("  Running ERFA adapter (reference)...")
    adapters["erfa"] = run_adapter([str(ERFA_BIN)], input_text, "erfa")

    print("  Running Siderust adapter...")
    adapters["siderust"] = run_adapter([str(SIDERUST_BIN)], input_text, "siderust")

    print("  Running Astropy adapter...")
    adapters["astropy"] = run_adapter(
        [sys.executable, str(ASTROPY_SCRIPT)], input_text, "astropy"
    )

    print("  Running libnova adapter...")
    adapters["libnova"] = run_adapter([str(LIBNOVA_BIN)], input_text, "libnova")

    # 3) Compute accuracy
    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_gmst_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib
        )

        result = {
            "experiment": "gmst_era",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("gmst_era"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        }
        results.append(result)

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_gmst_perf_input(jd_ut1[:perf_n], jd_tt[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_equ_ecl(n: int, seed: int, run_perf: bool = True):
    """
    Run the equ_ecl experiment: equatorial ↔ ecliptic coordinate transform.

    Reference: ERFA (IAU 2006 obliquity)
    Candidates: Siderust, Astropy, libnova
    """
    print(f"\n{'='*70}")
    print(f" Experiment: equ_ecl (N={n}, seed={seed})")
    print(f"{'='*70}")

    print("  Generating inputs...")
    epochs, ra, dec = generate_equ_ecl_inputs(n, seed)
    input_text = format_equ_ecl_input(epochs, ra, dec)

    adapters = {}
    for lib, cmd, label in [
        ("erfa", [str(ERFA_BIN)], "erfa"),
        ("siderust", [str(SIDERUST_BIN)], "siderust"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "astropy"),
        ("libnova", [str(LIBNOVA_BIN)], "libnova"),
    ]:
        print(f"  Running {label} adapter...")
        adapters[lib] = run_adapter(cmd, input_text, label)

    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_angular_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib,
            ra_key="ecl_lon_rad", dec_key="ecl_lat_rad",
        )

        results.append({
            "experiment": "equ_ecl",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("equ_ecl"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        })

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_equ_ecl_perf_input(epochs[:perf_n], ra[:perf_n], dec[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_equ_horizontal(n: int, seed: int, run_perf: bool = True):
    """
    Run the equ_horizontal experiment: equatorial → horizontal (AltAz).

    Reference: ERFA (eraHd2ae + eraGst06a)
    Candidates: Siderust, Astropy, libnova
    """
    print(f"\n{'='*70}")
    print(f" Experiment: equ_horizontal (N={n}, seed={seed})")
    print(f"{'='*70}")

    print("  Generating inputs...")
    jd_ut1, jd_tt, ra, dec, lon, lat = generate_equ_horizontal_inputs(n, seed)
    input_text = format_equ_horizontal_input(jd_ut1, jd_tt, ra, dec, lon, lat)

    adapters = {}
    for lib, cmd, label in [
        ("erfa", [str(ERFA_BIN)], "erfa"),
        ("siderust", [str(SIDERUST_BIN)], "siderust"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "astropy"),
        ("libnova", [str(LIBNOVA_BIN)], "libnova"),
    ]:
        print(f"  Running {label} adapter...")
        adapters[lib] = run_adapter(cmd, input_text, label)

    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_angular_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib,
            ra_key="az_rad", dec_key="alt_rad",
        )

        results.append({
            "experiment": "equ_horizontal",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("equ_horizontal"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        })

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_equ_horizontal_perf_input(
            jd_ut1[:perf_n], jd_tt[:perf_n], ra[:perf_n], dec[:perf_n],
            lon[:perf_n], lat[:perf_n]
        )

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_solar_position(n: int, seed: int, run_perf: bool = True):
    """
    Run the solar_position experiment: geocentric Sun RA/Dec.

    Reference: ERFA (VSOP87 via epv00)
    Candidates: Siderust, Astropy, libnova
    """
    print(f"\n{'='*70}")
    print(f" Experiment: solar_position (N={n}, seed={seed})")
    print(f"{'='*70}")

    print("  Generating inputs...")
    epochs = generate_solar_position_inputs(n, seed)
    input_text = format_solar_position_input(epochs)

    adapters = {}
    for lib, cmd, label in [
        ("erfa", [str(ERFA_BIN)], "erfa"),
        ("siderust", [str(SIDERUST_BIN)], "siderust"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "astropy"),
        ("libnova", [str(LIBNOVA_BIN)], "libnova"),
    ]:
        print(f"  Running {label} adapter...")
        adapters[lib] = run_adapter(cmd, input_text, label)

    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_angular_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib,
            ra_key="ra_rad", dec_key="dec_rad",
            extra_keys=["dist_au"],
        )

        results.append({
            "experiment": "solar_position",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("solar_position"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        })

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_solar_position_perf_input(epochs[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_lunar_position(n: int, seed: int, run_perf: bool = True):
    """
    Run the lunar_position experiment: geocentric Moon RA/Dec.

    Note: ERFA/Astropy use simplified Meeus (~10' accuracy).
    Siderust/libnova use full ELP 2000. Cross-lib comparison is the metric.
    """
    print(f"\n{'='*70}")
    print(f" Experiment: lunar_position (N={n}, seed={seed})")
    print(f"{'='*70}")

    print("  Generating inputs...")
    epochs = generate_lunar_position_inputs(n, seed)
    input_text = format_lunar_position_input(epochs)

    adapters = {}
    for lib, cmd, label in [
        ("erfa", [str(ERFA_BIN)], "erfa"),
        ("siderust", [str(SIDERUST_BIN)], "siderust"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "astropy"),
        ("libnova", [str(LIBNOVA_BIN)], "libnova"),
    ]:
        print(f"  Running {label} adapter...")
        adapters[lib] = run_adapter(cmd, input_text, label)

    # For Moon, use ERFA as reference but note it's simplified Meeus
    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_angular_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib,
            ra_key="ra_rad", dec_key="dec_rad",
            extra_keys=["dist_km"],
        )

        results.append({
            "experiment": "lunar_position",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("lunar_position"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        })

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_lunar_position_perf_input(epochs[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


def run_experiment_kepler_solver(n: int, seed: int, run_perf: bool = True):
    """
    Run the kepler_solver experiment: Kepler's equation M→E→ν.

    All libraries compared against each other (ERFA as reference).
    """
    print(f"\n{'='*70}")
    print(f" Experiment: kepler_solver (N={n}, seed={seed})")
    print(f"{'='*70}")

    print("  Generating inputs...")
    M_arr, e_arr = generate_kepler_inputs(n, seed)
    input_text = format_kepler_input(M_arr, e_arr)

    adapters = {}
    for lib, cmd, label in [
        ("erfa", [str(ERFA_BIN)], "erfa"),
        ("siderust", [str(SIDERUST_BIN)], "siderust"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "astropy"),
        ("libnova", [str(LIBNOVA_BIN)], "libnova"),
    ]:
        print(f"  Running {label} adapter...")
        adapters[lib] = run_adapter(cmd, input_text, label)

    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        print("  ✗ ERFA adapter failed — cannot compute accuracy.", file=sys.stderr)
        return results

    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        print(f"  Computing accuracy: {lib} vs erfa...")
        accuracy = compute_kepler_accuracy(
            ref_data["cases"], cand_data["cases"], "erfa", lib,
        )

        results.append({
            "experiment": "kepler_solver",
            "candidate_library": lib,
            "reference_library": "erfa",
            "alignment": alignment_checklist("kepler_solver"),
            "inputs": {"count": n, "seed": seed},
            "accuracy": accuracy,
            "performance": {},
            "run_metadata": run_metadata(),
        })

    # 4) Performance measurement
    if run_perf:
        perf_n = min(n, 10000)
        print(f"  Running performance tests (N={perf_n})...")
        perf_input = format_kepler_perf_input(M_arr[:perf_n], e_arr[:perf_n])

        for lib, cmd in [
            ("erfa", [str(ERFA_BIN)]),
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            print(f"    Timing {lib}...")
            perf_data = run_adapter(cmd, perf_input, f"{lib}_perf")
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = {
                            "per_op_ns": perf_data.get("per_op_ns"),
                            "throughput_ops_s": perf_data.get("throughput_ops_s"),
                            "total_ns": perf_data.get("total_ns"),
                            "batch_size": perf_data.get("count"),
                        }

        # Reference performance
        perf_erfa = run_adapter([str(ERFA_BIN)], perf_input, "erfa_perf")
        if perf_erfa:
            for r in results:
                r["reference_performance"] = {
                    "per_op_ns": perf_erfa.get("per_op_ns"),
                    "throughput_ops_s": perf_erfa.get("throughput_ops_s"),
                }

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: list, experiment: str):
    """Write result JSON files and summary table."""
    # Use timestamp with seconds precision to prevent collisions
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_DIR / timestamp_str / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        lib = r.get("candidate_library", "unknown")
        path = out_dir / f"{lib}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  ✓ Wrote {path.relative_to(LAB_ROOT)}")

    # Summary table
    summary = generate_summary_table(results)
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# {experiment} — Summary\n\n")
        f.write(f"Timestamp: {timestamp_str}\n\n")
        f.write(summary)
        f.write("\n## Alignment Checklist\n\n")
        if results:
            f.write("```json\n")
            json.dump(results[0].get("alignment", {}), f, indent=2)
            f.write("\n```\n")
    print(f"  ✓ Wrote {summary_path.relative_to(LAB_ROOT)}")

    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_experiments = [
        "frame_rotation_bpn", "gmst_era", "equ_ecl", "equ_horizontal",
        "solar_position", "lunar_position", "kepler_solver",
    ]

    parser = argparse.ArgumentParser(description="Siderust Lab Orchestrator")
    parser.add_argument("--experiment", default=None,
                        choices=all_experiments + ["all"],
                        help="(deprecated, use --experiments) Which experiment to run")
    parser.add_argument("--experiments", default=None,
                        help="Comma-separated list of experiments to run, or 'all'")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for input generation")
    parser.add_argument("--no-perf", action="store_true",
                        help="Skip performance tests")
    args = parser.parse_args()

    # Resolve which experiments to run (--experiments takes precedence)
    raw = args.experiments or args.experiment or "all"
    if raw == "all":
        experiments_to_run = all_experiments
    else:
        experiments_to_run = [e.strip() for e in raw.split(",") if e.strip()]

    all_results = []

    dispatch = {
        "frame_rotation_bpn": lambda: run_experiment_frame_rotation_bpn(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "gmst_era": lambda: run_experiment_gmst_era(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "equ_ecl": lambda: run_experiment_equ_ecl(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "equ_horizontal": lambda: run_experiment_equ_horizontal(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "solar_position": lambda: run_experiment_solar_position(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "lunar_position": lambda: run_experiment_lunar_position(
            args.n, args.seed, run_perf=not args.no_perf
        ),
        "kepler_solver": lambda: run_experiment_kepler_solver(
            args.n, args.seed, run_perf=not args.no_perf
        ),
    }

    for exp in experiments_to_run:
        runner = dispatch.get(exp)
        if runner is None:
            print(f"Unknown experiment: {exp}", file=sys.stderr)
            continue

        results = runner()

        if results:
            write_results(results, exp)
            all_results.extend(results)

            # Print summary
            print(f"\n{'─'*70}")
            print(f" Results Summary: {exp}")
            print(f"{'─'*70}")
            for r in results:
                lib = r.get("candidate_library", "?")
                acc = r.get("accuracy", {})

                ang = acc.get("angular_error_mas", {})
                if ang.get("p50") is not None:
                    print(f"  {lib} vs erfa:")
                    print(f"    Angular error (mas): p50={ang['p50']:.3f}  p99={ang['p99']:.3f}  max={ang['max']:.3f}")

                gmst = acc.get("gmst_error_arcsec", {})
                if gmst.get("p50") is not None:
                    print(f"  {lib} vs erfa:")
                    print(f"    GMST error (arcsec): p50={gmst['p50']:.6f}  p99={gmst['p99']:.6f}  max={gmst['max']:.6f}")

                sep = acc.get("angular_sep_arcsec", {})
                if sep.get("p50") is not None:
                    print(f"  {lib} vs erfa:")
                    print(f"    Angular sep (arcsec): p50={sep['p50']:.4f}  p99={sep['p99']:.4f}  max={sep['max']:.4f}")

                E_err = acc.get("E_error_rad", {})
                if E_err.get("p50") is not None:
                    con = acc.get("consistency_error_rad", {})
                    print(f"  {lib} vs erfa:")
                    print(f"    E error (rad): p50={E_err['p50']:.2e}  max={E_err['max']:.2e}  consistency max={con.get('max', 0):.2e}")

                perf = r.get("performance", {})
                if perf.get("per_op_ns"):
                    print(f"    Performance: {perf['per_op_ns']:.0f} ns/op, {perf['throughput_ops_s']:.0f} ops/s")

    if all_results:
        # Write combined summary
        summary = generate_summary_table(all_results)
        print(f"\n{'='*70}")
        print(" Combined Summary Table")
        print(f"{'='*70}")
        print(summary)


if __name__ == "__main__":
    main()
