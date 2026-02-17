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
import time
import statistics
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

# Performance benchmarking defaults
DEFAULT_PERF_ROUNDS = 10        # Number of separate timing rounds (more rounds = lower CV)
DEFAULT_PERF_WARMUP = 100       # Warmup iterations before each round (adapters use min(N, 100))
MIN_MEASURABLE_NS = 10.0        # Warn if per-op time is below this threshold
MIN_PERF_N = 50000              # Minimum batch size for perf — ensures cheap ops accumulate enough time

# ---------------------------------------------------------------------------
# Experiment descriptions (for non-expert users)
# ---------------------------------------------------------------------------
EXPERIMENT_DESCRIPTIONS = {
    "frame_rotation_bpn": {
        "title": "Frame Rotation (Bias-Precession-Nutation)",
        "what": "Rotates a direction vector from the ICRS celestial reference frame to the "
                "True-of-Date frame using the Bias-Precession-Nutation (BPN) matrix.",
        "why": "This is the fundamental coordinate transformation used in astrometry. "
               "Differences indicate how well each library models Earth's axis wobble.",
        "units": "Angular error in milli-arcseconds (mas). 1 mas = 1/3,600,000 of a degree.",
        "interpret": "Lower error = closer to ERFA's IAU 2006/2000A model. Siderust uses "
                     "simpler Meeus precession + IAU 1980 nutation, so some offset is expected.",
    },
    "gmst_era": {
        "title": "Greenwich Mean Sidereal Time & Earth Rotation Angle",
        "what": "Computes GMST (how far the Earth has rotated relative to the stars) and ERA "
                "(the raw rotation angle) at given epochs.",
        "why": "Time-scale conversions underpin all ground-based astronomical observations. "
               "Small errors here compound in coordinate transforms.",
        "units": "GMST error in arcseconds; ERA error in radians.",
        "interpret": "Lower error = better agreement with ERFA's IAU 2006 polynomial. "
                     "Libnova uses Meeus formula which differs at the arcsecond level.",
    },
    "equ_ecl": {
        "title": "Equatorial to Ecliptic Coordinate Transform",
        "what": "Converts sky positions from RA/Dec (equatorial) to ecliptic longitude/latitude, "
                "using the obliquity of the ecliptic.",
        "why": "Ecliptic coordinates are natural for solar system objects. The transform "
               "depends on the obliquity model used.",
        "units": "Angular separation in arcseconds between reference and candidate ecliptic positions.",
        "interpret": "Lower separation = better agreement with ERFA's IAU 2006 obliquity model.",
    },
    "equ_horizontal": {
        "title": "Equatorial to Horizontal (Alt-Az) Transform",
        "what": "Converts celestial RA/Dec to local azimuth/altitude for a ground observer, "
                "using sidereal time and spherical trigonometry.",
        "why": "This is the 'where do I point my telescope?' calculation. Accuracy depends "
               "on the GAST (sidereal time) model used.",
        "units": "Angular separation in arcseconds between reference and candidate az/alt positions.",
        "interpret": "Lower separation = better. Differences mainly arise from GAST model choice.",
    },
    "solar_position": {
        "title": "Sun Geocentric Position",
        "what": "Computes the apparent geocentric RA/Dec of the Sun at given epochs using "
                "VSOP87 planetary theory.",
        "why": "Sun position is needed for solar observations, shadow calculations, and "
               "as input to other transforms.",
        "units": "Angular separation in arcseconds between reference and candidate Sun positions.",
        "interpret": "ERFA uses direct VSOP87 via epv00. Siderust uses its own VSOP87 "
                     "implementation with aberration+FK5 corrections.",
    },
    "lunar_position": {
        "title": "Moon Geocentric Position",
        "what": "Computes geocentric RA/Dec of the Moon. Note: ERFA/Astropy use simplified "
                "Meeus (~10 arcmin accuracy) while Siderust/libnova use full ELP 2000.",
        "why": "Moon position accuracy varies significantly between libraries because no "
               "standard 'reference' implementation exists in ERFA.",
        "units": "Angular separation in arcseconds. Expect ~arcminute differences due to model gap.",
        "interpret": "Large differences (arcminutes) reflect different ephemeris models, not bugs. "
                     "Cross-library comparison is more meaningful than ERFA-as-truth here.",
    },
    "kepler_solver": {
        "title": "Kepler Equation Solver (M → E → ν)",
        "what": "Solves Kepler's equation M = E - e·sin(E) for the eccentric anomaly E, "
                "then computes the true anomaly ν. Tests convergence across eccentricities.",
        "why": "Kepler's equation is fundamental to orbital mechanics. Different solvers "
               "(Newton-Raphson vs bisection) have different convergence properties.",
        "units": "E and ν errors in radians. Self-consistency residual in radians.",
        "interpret": "Lower residual = better convergence. Libnova's bisection method converges "
                     "to ~1e-6 deg, while Newton-Raphson methods reach ~1e-15 rad.",
    },
    "frame_bias": {
        "title": "Frame Bias (ICRS → Mean J2000)",
        "what": "Applies the ICRS-to-mean-J2000 frame bias rotation to a direction vector. "
                "This isolates the ~17 mas frame bias from precession/nutation.",
        "why": "Frame bias is a small but constant rotation. Testing it separately verifies "
               "the bias component of the full BPN matrix.",
        "units": "Angular error in mas (milliarcseconds).",
        "interpret": "All libraries implementing IAU frame bias should agree to sub-mas level.",
    },
    "precession": {
        "title": "Precession (Mean J2000 → Mean of Date)",
        "what": "Applies the IAU precession matrix to rotate from mean J2000 to mean equator "
                "and equinox of date.",
        "why": "Precession is the largest component of the BPN matrix. This isolates it from "
               "nutation and frame bias to pinpoint model accuracy.",
        "units": "Angular error in mas (milliarcseconds).",
        "interpret": "ERFA/Astropy use IAU 2006 precession. Siderust uses the same model. "
                     "libnova uses Meeus — expect arcsec-level differences.",
    },
    "nutation": {
        "title": "Nutation (Mean of Date → True of Date)",
        "what": "Applies the nutation matrix to rotate from mean equator/equinox of date "
                "to true equator/equinox of date.",
        "why": "Nutation oscillates ±9 arcsec. Isolating it reveals IAU 2000A vs 2000B differences.",
        "units": "Angular error in mas (milliarcseconds).",
        "interpret": "ERFA/Astropy use IAU 2000A (1365 terms). Siderust uses IAU 2000B (77 terms). "
                     "libnova uses IAU 1980 (69 terms). Expect ~1 mas difference (2000A vs 2000B).",
    },
    "icrs_ecl_j2000": {
        "title": "ICRS → Ecliptic J2000",
        "what": "Transforms an ICRS direction vector to ecliptic coordinates at the J2000 epoch. "
                "This is a time-independent rotation by the mean obliquity at J2000.",
        "why": "Tests the obliquity constant and ecliptic frame rotation without time-dependent terms.",
        "units": "Angular error in mas (milliarcseconds).",
        "interpret": "Time-independent transform — all IAU-based libraries should agree to µas level.",
    },
    "icrs_ecl_tod": {
        "title": "ICRS → Ecliptic of Date",
        "what": "Transforms equatorial RA/Dec to ecliptic longitude/latitude at the date of observation. "
                "Combines precession, nutation, and obliquity.",
        "why": "End-to-end ecliptic transform that exercises the full precession-nutation chain plus obliquity.",
        "units": "Angular separation in arcseconds.",
        "interpret": "ERFA/Astropy share IAU 2006 obliquity. libnova uses Meeus — expect arcsec differences.",
    },
    "horiz_to_equ": {
        "title": "Horizontal → Equatorial (AltAz → RA/Dec)",
        "what": "Converts horizontal (azimuth, altitude) coordinates to equatorial RA/Dec via hour angle "
                "and GAST computation.",
        "why": "Reverse of equ_horizontal. Tests the inverse spherical trig path and GAST model.",
        "units": "Angular separation in arcseconds.",
        "interpret": "Same spherical trig across all libraries. Differences arise from GAST model only.",
    },
    # 13 new matrix experiments (inverse / composed / obliquity transforms)
    "inv_frame_bias": {
        "title": "Inverse Frame Bias (Mean J2000 → ICRS)",
        "what": "Reverses the ICRS-to-J2000 frame bias by applying the transposed bias matrix.",
        "units": "Angular error in mas.", "interpret": "All IAU-based libraries agree to sub-mas.",
    },
    "inv_precession": {
        "title": "Inverse Precession (Mean of Date → Mean J2000)",
        "what": "Reverses IAU precession by applying the transposed precession matrix.",
        "units": "Angular error in mas.", "interpret": "libnova uses Meeus — arcsec differences expected.",
    },
    "inv_nutation": {
        "title": "Inverse Nutation (True of Date → Mean of Date)",
        "what": "Reverses nutation by transposing the nutation matrix (or by approximate correction in libnova).",
        "units": "Angular error in mas.", "interpret": "libnova uses approximate ΔRA/ΔDec subtraction.",
    },
    "inv_bpn": {
        "title": "Inverse BPN (True of Date → ICRS)",
        "what": "Reverses the full BPN chain (transpose of eraPnm06a matrix).",
        "units": "Angular error in mas.", "interpret": "libnova skipped (no frame bias concept).",
    },
    "inv_icrs_ecl_j2000": {
        "title": "Ecliptic J2000 → ICRS",
        "what": "Reverse of ICRS → EclipticMeanJ2000 via transposed ecliptic rotation matrix.",
        "units": "Angular error in mas.", "interpret": "Time-independent — µas agreement expected.",
    },
    "obliquity": {
        "title": "Obliquity (Ecliptic J2000 → Eq Mean J2000)",
        "what": "Pure obliquity rotation from ecliptic to equatorial at J2000 (Rx(+ε₀)).",
        "units": "Angular error in mas.", "interpret": "Time-independent obliquity constant.",
    },
    "inv_obliquity": {
        "title": "Inverse Obliquity (Eq Mean J2000 → Ecliptic J2000)",
        "what": "Pure obliquity rotation from equatorial to ecliptic at J2000 (Rx(−ε₀)).",
        "units": "Angular error in mas.", "interpret": "Time-independent obliquity constant.",
    },
    "bias_precession": {
        "title": "Bias + Precession (ICRS → Mean of Date)",
        "what": "Combined frame bias + precession: ICRS → EquatorialMeanOfDate.",
        "units": "Angular error in mas.", "interpret": "libnova skipped (no frame bias).",
    },
    "inv_bias_precession": {
        "title": "Inverse Bias+Precession (Mean of Date → ICRS)",
        "what": "Reverse of ICRS → MeanOfDate via transposed RBP matrix.",
        "units": "Angular error in mas.", "interpret": "libnova skipped (no frame bias).",
    },
    "precession_nutation": {
        "title": "Precession + Nutation (Mean J2000 → True of Date)",
        "what": "Combined precession + nutation: EquatorialMeanJ2000 → EquatorialTrueOfDate.",
        "units": "Angular error in mas.", "interpret": "libnova uses Meeus prec + IAU 1980 nut.",
    },
    "inv_precession_nutation": {
        "title": "Inverse Prec+Nut (True of Date → Mean J2000)",
        "what": "Reverse of EqMeanJ2000 → EqTrueOfDate via transposed composed matrix.",
        "units": "Angular error in mas.", "interpret": "libnova uses approximate inverses.",
    },
    "inv_icrs_ecl_tod": {
        "title": "Ecliptic of Date → ICRS",
        "what": "Reverse of ICRS → EclipticTrueOfDate (transpose of eraEcm06).",
        "units": "Angular error in mas.", "interpret": "libnova chains ecl→eq(date)→prec→J2000.",
    },
    "inv_equ_ecl": {
        "title": "Ecliptic of Date → Eq Mean of Date",
        "what": "Reverse of EqMeanOfDate → EclipticTrueOfDate.",
        "units": "Angular error in mas.", "interpret": "libnova uses ln_get_equ_from_ecl at date.",
    },
}

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def progress(msg: str, experiment: str = "", step: str = "", total_steps: int = 0, current_step: int = 0):
    """Print a structured progress message that the web app can parse.
    
    Format: [PROGRESS] experiment=<name> step=<step> current=<n> total=<m> | <message>
    """
    parts = ["[PROGRESS]"]
    if experiment:
        parts.append(f"experiment={experiment}")
    if step:
        parts.append(f"step={step}")
    if total_steps > 0:
        parts.append(f"current={current_step}")
        parts.append(f"total={total_steps}")
    parts.append(f"| {msg}")
    print(" ".join(parts), flush=True)


def dataset_fingerprint(data: dict) -> str:
    """Compute a SHA-256 fingerprint of the input dataset for reproducibility."""
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def ensure_siderust_adapter_built() -> None:
    """Build siderust adapter so runs use current local siderust sources."""
    manifest = LAB_ROOT / "pipeline" / "adapters" / "siderust_adapter" / "Cargo.toml"
    cmd = ["cargo", "build", "--release", "--manifest-path", str(manifest)]
    print("Ensuring siderust adapter is up to date...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(LAB_ROOT),
    )
    if result.returncode != 0:
        print("✗ Failed to build siderust adapter.", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-1000:], file=sys.stderr)
        raise SystemExit(2)
    print("✓ Siderust adapter build complete.")

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


def generate_direction_vector_inputs(n: int, seed: int):
    """Generate random JD TT epochs + unit direction vectors for frame transform tests."""
    rng = np.random.default_rng(seed)
    # 100-year range centered on J2000 (JD 2451545.0 = 2000-01-01.5 TT)
    jd_tt = rng.uniform(2451545.0, 2488070.0, size=n)
    # Random unit vectors on the sphere
    phi = rng.uniform(0, 2 * np.pi, size=n)
    cos_theta = rng.uniform(-1, 1, size=n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    directions = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ])
    return jd_tt, directions


def generate_horiz_to_equ_inputs(n: int, seed: int):
    """Generate random horizontal coordinates + observer locations for horiz_to_equ tests."""
    rng = np.random.default_rng(seed + 7)  # offset seed to avoid duplicate data
    jd_tt = rng.uniform(2451545.0, 2488070.0, size=n)
    jd_ut1 = jd_tt - 69.184 / 86400.0  # simplified UT1 ≈ TT - 69.184s
    # Azimuth 0..2π, altitude 5°..89° (avoid horizon and zenith singularities)
    az = rng.uniform(0, 2 * np.pi, size=n)
    alt = rng.uniform(np.radians(5), np.radians(89), size=n)
    # Random observer locations
    obs_lon = rng.uniform(-np.pi, np.pi, size=n)
    obs_lat = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
    return jd_ut1, jd_tt, az, alt, obs_lon, obs_lat


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


def run_multi_sample_perf(cmd, input_text: str, label: str,
                          rounds: int = DEFAULT_PERF_ROUNDS) -> dict | None:
    """Run performance adapter multiple rounds, compute statistical summary.

    Returns a dict with per_op_ns stats: mean, median, std_dev, min, max, ci95, samples.
    """
    samples_per_op = []
    samples_total = []

    for r in range(rounds):
        result = run_adapter(cmd, input_text, f"{label}_round{r}")
        if result is None:
            return None

        per_op = result.get("per_op_ns")
        total = result.get("total_ns")
        count = result.get("count", 0)

        if per_op is not None:
            samples_per_op.append(per_op)
        if total is not None:
            samples_total.append(total)

    if not samples_per_op:
        return None

    mean_ns = statistics.mean(samples_per_op)
    median_ns = statistics.median(samples_per_op)
    std_dev = statistics.stdev(samples_per_op) if len(samples_per_op) > 1 else 0.0
    min_ns = min(samples_per_op)
    max_ns = max(samples_per_op)

    # 95% confidence interval (assumes normal distribution)
    n = len(samples_per_op)
    ci95_half = 1.96 * std_dev / math.sqrt(n) if n > 1 else 0.0

    # Coefficient of variation (stability indicator)
    cv = (std_dev / mean_ns * 100) if mean_ns > 0 else 0.0

    # "Too fast to measure" warning
    warnings = []
    if median_ns < MIN_MEASURABLE_NS:
        warnings.append(
            f"Per-op time ({median_ns:.1f} ns) is below measurable threshold "
            f"({MIN_MEASURABLE_NS} ns). Results may be dominated by measurement overhead."
        )
    if cv > 20:
        warnings.append(
            f"High coefficient of variation ({cv:.1f}%). Results are unstable. "
            "Consider increasing sample count or reducing system load."
        )

    return {
        "per_op_ns": median_ns,  # Use median as primary (robust to outliers)
        "per_op_ns_mean": mean_ns,
        "per_op_ns_median": median_ns,
        "per_op_ns_std_dev": std_dev,
        "per_op_ns_min": min_ns,
        "per_op_ns_max": max_ns,
        "per_op_ns_ci95": [mean_ns - ci95_half, mean_ns + ci95_half],
        "per_op_ns_cv_pct": cv,
        "throughput_ops_s": 1e9 / median_ns if median_ns > 0 else 0,
        "total_ns_median": statistics.median(samples_total) if samples_total else None,
        "batch_size": result.get("count"),
        "rounds": rounds,
        "samples": samples_per_op,
        "warnings": warnings,
    }


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


# --- Direction-vector format helpers (frame_bias, precession, nutation, icrs_ecl_j2000) ---

def _format_direction_vector_input(exp_name, epochs, directions):
    """Generic formatter for direction-vector experiments."""
    lines = [exp_name, str(len(epochs))]
    for jd, d in zip(epochs, directions):
        lines.append(f"{jd:.15f} {d[0]:.17e} {d[1]:.17e} {d[2]:.17e}")
    return "\n".join(lines) + "\n"


def format_frame_bias_input(epochs, directions):
    return _format_direction_vector_input("frame_bias", epochs, directions)

def format_frame_bias_perf_input(epochs, directions):
    return _format_direction_vector_input("frame_bias_perf", epochs, directions)

def format_precession_input(epochs, directions):
    return _format_direction_vector_input("precession", epochs, directions)

def format_precession_perf_input(epochs, directions):
    return _format_direction_vector_input("precession_perf", epochs, directions)

def format_nutation_input(epochs, directions):
    return _format_direction_vector_input("nutation", epochs, directions)

def format_nutation_perf_input(epochs, directions):
    return _format_direction_vector_input("nutation_perf", epochs, directions)

def format_icrs_ecl_j2000_input(epochs, directions):
    return _format_direction_vector_input("icrs_ecl_j2000", epochs, directions)

def format_icrs_ecl_j2000_perf_input(epochs, directions):
    return _format_direction_vector_input("icrs_ecl_j2000_perf", epochs, directions)

# 13 new experiments — all use direction-vector format
def _make_dir_format_pair(exp_name):
    """Factory for format functions for direction-vector experiments."""
    def fmt(epochs, directions):
        return _format_direction_vector_input(exp_name, epochs, directions)
    def fmt_perf(epochs, directions):
        return _format_direction_vector_input(f"{exp_name}_perf", epochs, directions)
    return fmt, fmt_perf

format_inv_frame_bias_input, format_inv_frame_bias_perf_input = _make_dir_format_pair("inv_frame_bias")
format_inv_precession_input, format_inv_precession_perf_input = _make_dir_format_pair("inv_precession")
format_inv_nutation_input, format_inv_nutation_perf_input = _make_dir_format_pair("inv_nutation")
format_inv_bpn_input, format_inv_bpn_perf_input = _make_dir_format_pair("inv_bpn")
format_inv_icrs_ecl_j2000_input, format_inv_icrs_ecl_j2000_perf_input = _make_dir_format_pair("inv_icrs_ecl_j2000")
format_obliquity_input, format_obliquity_perf_input = _make_dir_format_pair("obliquity")
format_inv_obliquity_input, format_inv_obliquity_perf_input = _make_dir_format_pair("inv_obliquity")
format_bias_precession_input, format_bias_precession_perf_input = _make_dir_format_pair("bias_precession")
format_inv_bias_precession_input, format_inv_bias_precession_perf_input = _make_dir_format_pair("inv_bias_precession")
format_precession_nutation_input, format_precession_nutation_perf_input = _make_dir_format_pair("precession_nutation")
format_inv_precession_nutation_input, format_inv_precession_nutation_perf_input = _make_dir_format_pair("inv_precession_nutation")
format_inv_icrs_ecl_tod_dir_input, format_inv_icrs_ecl_tod_dir_perf_input = _make_dir_format_pair("inv_icrs_ecl_tod")
format_inv_equ_ecl_dir_input, format_inv_equ_ecl_dir_perf_input = _make_dir_format_pair("inv_equ_ecl")


def format_icrs_ecl_tod_input(epochs, ra, dec):
    """Format input for icrs_ecl_tod experiment: jd_tt ra dec per line."""
    lines = ["icrs_ecl_tod", str(len(epochs))]
    for jd, r, d in zip(epochs, ra, dec):
        lines.append(f"{jd:.15f} {r:.17e} {d:.17e}")
    return "\n".join(lines) + "\n"


def format_icrs_ecl_tod_perf_input(epochs, ra, dec):
    """Format input for icrs_ecl_tod performance experiment."""
    lines = ["icrs_ecl_tod_perf", str(len(epochs))]
    for jd, r, d in zip(epochs, ra, dec):
        lines.append(f"{jd:.15f} {r:.17e} {d:.17e}")
    return "\n".join(lines) + "\n"


def format_horiz_to_equ_input(jd_ut1, jd_tt, az, alt, lon, lat):
    """Format input for horiz_to_equ experiment: jd_ut1 jd_tt az alt lon lat per line."""
    lines = ["horiz_to_equ", str(len(jd_ut1))]
    for u, t, a, al, lo, la in zip(jd_ut1, jd_tt, az, alt, lon, lat):
        lines.append(f"{u:.15f} {t:.15f} {a:.17e} {al:.17e} {lo:.17e} {la:.17e}")
    return "\n".join(lines) + "\n"


def format_horiz_to_equ_perf_input(jd_ut1, jd_tt, az, alt, lon, lat):
    """Format input for horiz_to_equ performance experiment."""
    lines = ["horiz_to_equ_perf", str(len(jd_ut1))]
    for u, t, a, al, lo, la in zip(jd_ut1, jd_tt, az, alt, lon, lat):
        lines.append(f"{u:.15f} {t:.15f} {a:.17e} {al:.17e} {lo:.17e} {la:.17e}")
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

        # Skip cases where the reference itself has poor self-consistency
        # (e.g., Newton-Raphson diverged for extreme eccentricity).
        r_residual = ref_c.get("residual_rad", 0.0)
        if not math.isfinite(r_residual) or abs(r_residual) > 1e-10:
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
        "library_notes": {
            "astropy": (
                "The 'astropy' adapter calls ERFA/pyerfa C kernels directly from Python. "
                "Accuracy results are identical to ERFA. Performance results measure "
                "Python-loop + ERFA-kernel overhead, NOT the high-level astropy.coordinates stack."
            ),
        },
    }

    if experiment == "frame_rotation_bpn":
        base["models"] = {
            "erfa": "IAU 2006/2000A bias-precession-nutation (eraPnm06a)",
            "siderust": "IERS 2003 frame bias + IAU 2006 precession + IAU 2000B nutation (frame_rotation provider)",
            "astropy": "IAU 2006/2000A via bundled ERFA (erfa.pnm06a)",
            "libnova": "Meeus precession (ζ,z,θ Equ 20.3) + IAU 1980 nutation (63-term Table 21A), applied as RA/Dec corrections (no BPN matrix)",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (models differ across libraries)"
        base["mode"] = mode
        base["note"] = (
            "ERFA and Astropy use the same IAU 2006/2000A model (reference). "
            "Siderust uses IAU 2006 precession + IAU 2000B nutation (close to ERFA, with 2000B vs 2000A differences). "
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
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (libnova uses Meeus, no ERA)"
        base["mode"] = mode

    elif experiment == "equ_ecl":
        base["models"] = {
            "erfa": "IAU 2006 obliquity-based transform (eraEqec06 / eraEceq06)",
            "siderust": "IAU 2006 ecliptic-of-date via precession matrix + mean obliquity",
            "astropy": "IAU 2006 via bundled ERFA (erfa.eqec06 / erfa.eceq06)",
            "libnova": "Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ / ln_get_equ_from_ecl",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (libnova Meeus obliquity differs)"
        base["note"] = (
            "ERFA and Astropy share the same IAU 2006 obliquity model (reference). "
            "Siderust uses an explicit IAU 2006 equatorial/ecliptic-of-date transform path. "
            "libnova uses Meeus obliquity polynomial — expect ~arcsec-level differences."
        )

    elif experiment == "equ_horizontal":
        base["models"] = {
            "erfa": "Spherical trig via eraHd2ae / eraAe2hd; GAST via eraGst06a; no refraction",
            "siderust": "Spherical trig matching ERFA formulas; GAST IAU 2006 via siderust astro path",
            "astropy": "eraHd2ae / eraAe2hd via bundled ERFA; GAST via eraGst06a",
            "libnova": "ln_get_hrz_from_equ / ln_get_equ_from_hrz; convention fix: az_erfa = (360 - az_ln + 180) % 360",
        }
        base["model_parity_class"] = "model-parity"
        base["accuracy_interpretation"] = "accuracy vs ERFA reference (same spherical trig model)"
        base["note"] = (
            "Azimuth convention: ERFA 0°=North CW; libnova 0°=South. "
            "All adapters use the same spherical trig, differences arise from GAST model. "
            "No atmospheric refraction applied."
        )
        base["refraction"] = "disabled"

    elif experiment == "solar_position":
        base["models"] = {
            "erfa": "VSOP87 via eraEpv00: heliocentric Earth → geocentric Sun (negate); BCRS equatorial output",
            "siderust": "Geometric heliocentric-ecliptic center transformed to geocentric ICRS (no aberration)",
            "astropy": "VSOP87 via erfa.epv00 (same as ERFA)",
            "libnova": "VSOP87 via ln_get_solar_equ_coords (different truncation/corrections)",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (VSOP87 truncation levels differ)"
        base["ephemeris_source"] = "VSOP87 (analytic, all libraries)"
        base["note"] = (
            "ERFA epv00 returns BCRS-aligned equatorial (no obliquity rotation). "
            "Differences reflect VSOP87 truncation levels and aberration correction details."
        )

    elif experiment == "lunar_position":
        base["models"] = {
            "erfa": "Simplified Meeus Ch.47 (major terms only, ~10' accuracy)",
            "siderust": "Simplified Meeus Ch.47 (major terms only), centralized in siderust astro module",
            "astropy": "Simplified Meeus Ch.47 (same algorithm as ERFA adapter)",
            "libnova": "ELP 2000-82B via ln_get_lunar_equ_coords (full model)",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = (
            "agreement with ERFA baseline (ERFA uses simplified Meeus, "
            "libnova uses full ELP 2000 — arcminute-level differences are expected)"
        )
        base["note"] = (
            "ERFA, Astropy, and Siderust use the same simplified Meeus benchmark model (~10' accuracy). "
            "libnova uses full ELP 2000, so arcminute-level differences vs reference are expected. "
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
        base["model_parity_class"] = "model-parity"
        base["accuracy_interpretation"] = "accuracy vs ERFA reference (same equation, different solvers)"
        base["note"] = (
            "Kepler's equation M = E - e*sin(E) is solved for E given (M, e). "
            "Self-consistency M_reconstructed = E - e*sin(E) is the primary metric. "
            "libnova uses a bisection method with lower convergence tolerance."
        )

    elif experiment == "frame_bias":
        base["models"] = {
            "erfa": "IAU 2006 frame bias matrix component from eraBp06",
            "siderust": "IERS 2003 frame bias via frame rotation provider (ICRS → EquatorialMeanJ2000)",
            "astropy": "IAU 2006 frame bias via bundled ERFA (erfa.bp06)",
            "libnova": "Not available (no frame bias concept in libnova)",
        }
        base["model_parity_class"] = "model-parity"
        base["accuracy_interpretation"] = "accuracy vs ERFA reference (IAU frame bias is a fixed rotation)"
        base["note"] = (
            "Frame bias is a small (~17 mas) time-independent rotation between ICRS and mean J2000. "
            "libnova has no equivalent — its results are skipped."
        )

    elif experiment == "precession":
        base["models"] = {
            "erfa": "IAU 2006 precession matrix from eraPmat06",
            "siderust": "IAU 2006 precession via frame rotation provider (EquatorialMeanJ2000 → EquatorialMeanOfDate)",
            "astropy": "IAU 2006 precession via bundled ERFA (erfa.pmat06)",
            "libnova": "Meeus precession (ζ,z,θ Equ 20.3) via ln_get_equ_prec2",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (libnova uses Meeus model)"
        base["note"] = (
            "ERFA, Astropy, and Siderust all use IAU 2006 precession. "
            "libnova uses Meeus precession formulae — expect arcsec-level differences."
        )

    elif experiment == "nutation":
        base["models"] = {
            "erfa": "IAU 2000A nutation (1365 terms) via eraNum06a",
            "siderust": "IAU 2000B nutation (77 terms) via frame rotation provider",
            "astropy": "IAU 2000A nutation via bundled ERFA (erfa.num06a)",
            "libnova": "IAU 1980 nutation (69 terms) via ln_get_equ_nut / ln_nutation",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (nutation models differ in term count)"
        base["note"] = (
            "IAU 2000A (ERFA/Astropy) has 1365 terms, 2000B (Siderust) has 77 terms (~1 mas difference), "
            "IAU 1980 (libnova) has 69 terms (~tens of mas difference)."
        )

    elif experiment == "icrs_ecl_j2000":
        base["models"] = {
            "erfa": "IAU 2006 ecliptic rotation at J2000 epoch via eraEcm06",
            "siderust": "ICRS → EclipticMeanJ2000 via frame rotation (mean obliquity at J2000)",
            "astropy": "IAU 2006 ecliptic rotation via bundled ERFA (erfa.ecm06)",
            "libnova": "Meeus obliquity (Eq 22.2) applied at J2000 epoch",
        }
        base["model_parity_class"] = "model-parity"
        base["accuracy_interpretation"] = "accuracy vs ERFA reference (time-independent obliquity)"
        base["note"] = (
            "Time-independent rotation by the mean obliquity at J2000. "
            "All IAU-based libraries should agree to µas level. "
            "libnova uses Meeus obliquity which is close but not identical."
        )

    elif experiment == "icrs_ecl_tod":
        base["models"] = {
            "erfa": "IAU 2006 equatorial → ecliptic of date via eraEqec06",
            "siderust": "ICRS → ecliptic of date via DirectionAstroExt::to_ecliptic_of_date",
            "astropy": "IAU 2006 equatorial → ecliptic via bundled ERFA (erfa.eqec06)",
            "libnova": "Meeus obliquity (Eq 22.2) via ln_get_ecl_from_equ",
        }
        base["model_parity_class"] = "model-mismatch"
        base["accuracy_interpretation"] = "agreement with ERFA baseline (libnova Meeus obliquity differs)"
        base["note"] = (
            "Similar to equ_ecl but explicitly identified as ICRS → ecliptic-of-date transform. "
            "ERFA/Astropy share IAU 2006 obliquity model. libnova uses Meeus."
        )

    elif experiment == "horiz_to_equ":
        base["models"] = {
            "erfa": "Spherical trig via eraAe2hd; GAST via eraGst06a; no refraction",
            "siderust": "Spherical trig via FromHorizontal::to_equatorial; GAST IAU 2006",
            "astropy": "eraAe2hd via bundled ERFA; GAST via eraGst06a",
            "libnova": "ln_get_equ_from_hrz; convention fix: az = (input_az - 180) % 360",
        }
        base["model_parity_class"] = "model-parity"
        base["accuracy_interpretation"] = "accuracy vs ERFA reference (same spherical trig model)"
        base["note"] = (
            "Inverse of equ_horizontal. Same spherical trig, same GAST dependencies. "
            "Azimuth convention: ERFA 0°=North CW; libnova 0°=South. "
            "No atmospheric refraction applied."
        )
        base["refraction"] = "disabled"

    # 13 new matrix experiments — alignment checklists
    elif experiment in ("inv_frame_bias",):
        base["models"] = {
            "erfa": "Transpose of IAU 2006 frame bias matrix (eraBp06 → rb^T)",
            "siderust": "EquatorialMeanJ2000 → ICRS via frame rotation provider inverse",
            "astropy": "Transpose of IAU 2006 frame bias via bundled ERFA",
            "libnova": "Not available (no frame bias concept)",
        }
        base["model_parity_class"] = "model-parity"

    elif experiment in ("inv_precession",):
        base["models"] = {
            "erfa": "Transpose of IAU 2006 precession matrix (eraPmat06 → rp^T)",
            "siderust": "EquatorialMeanOfDate → EquatorialMeanJ2000 via frame rotation inverse",
            "astropy": "Transpose of IAU 2006 precession via bundled ERFA",
            "libnova": "Meeus inverse precession via ln_get_equ_prec2(date→J2000)",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("inv_nutation",):
        base["models"] = {
            "erfa": "Transpose of IAU 2000A nutation matrix (eraNum06a → rn^T)",
            "siderust": "EquatorialTrueOfDate → EquatorialMeanOfDate via frame rotation inverse",
            "astropy": "Transpose of IAU 2000A nutation via bundled ERFA",
            "libnova": "Approximate inverse via ΔRA/ΔDec subtraction (IAU 1980)",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("inv_bpn",):
        base["models"] = {
            "erfa": "Transpose of IAU 2006 BPN matrix (eraPnm06a → rnpb^T)",
            "siderust": "EquatorialTrueOfDate → ICRS via frame rotation inverse",
            "astropy": "Transpose of IAU 2006 BPN via bundled ERFA",
            "libnova": "Not available (no ICRS/frame bias concept)",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("inv_icrs_ecl_j2000",):
        base["models"] = {
            "erfa": "Transpose of eraEcm06(J2000) matrix",
            "siderust": "EclipticMeanJ2000 → ICRS via TransformFrame inverse",
            "astropy": "Transpose of erfa.ecm06(J2000)",
            "libnova": "ln_get_equ_from_ecl at J2000 (Meeus obliquity)",
        }
        base["model_parity_class"] = "model-parity"

    elif experiment in ("obliquity", "inv_obliquity"):
        base["models"] = {
            "erfa": "Pure Rx(±ε₀) rotation using eraObl06(J2000)",
            "siderust": "EclipticMeanJ2000 ↔ EquatorialMeanJ2000 via TransformFrame",
            "astropy": "Pure Rx(±ε₀) using erfa.obl06(J2000)",
            "libnova": "ln_get_equ_from_ecl / ln_get_ecl_from_equ at J2000 (Meeus obliquity)",
        }
        base["model_parity_class"] = "model-parity"

    elif experiment in ("bias_precession", "inv_bias_precession"):
        base["models"] = {
            "erfa": "IAU 2006 bias+precession product (eraBp06 → rbp / rbp^T)",
            "siderust": "ICRS ↔ EquatorialMeanOfDate via composed frame rotation",
            "astropy": "IAU 2006 bias+precession via bundled ERFA",
            "libnova": "Not available (no frame bias concept)",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("precession_nutation", "inv_precession_nutation"):
        base["models"] = {
            "erfa": "N×P composed matrix (eraPmat06 × eraNum06a / transpose)",
            "siderust": "EquatorialMeanJ2000 ↔ EquatorialTrueOfDate via frame rotation",
            "astropy": "N×P composed via bundled ERFA",
            "libnova": "ln_get_equ_prec + ln_get_equ_nut sequenced / approximate inverse",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("inv_icrs_ecl_tod",):
        base["models"] = {
            "erfa": "Transpose of eraEcm06(date) matrix",
            "siderust": "EclipticTrueOfDate → ICRS via FromEclipticTrueOfDate::to_icrs",
            "astropy": "Transpose of erfa.ecm06(date)",
            "libnova": "ln_get_equ_from_ecl(date) + ln_get_equ_prec2(date→J2000)",
        }
        base["model_parity_class"] = "model-mismatch"

    elif experiment in ("inv_equ_ecl",):
        base["models"] = {
            "erfa": "RBP × ECM06^T composed matrix",
            "siderust": "EclipticTrueOfDate → EquatorialMeanOfDate via FromEclipticTrueOfDate",
            "astropy": "rbp × ecm06^T composed via bundled ERFA",
            "libnova": "ln_get_equ_from_ecl at date (mean-of-date output)",
        }
        base["model_parity_class"] = "model-mismatch"

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
        "git_branch": _get_git_branch(LAB_ROOT),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "platform_detail": platform.platform(),
        "toolchain": {
            "python": platform.python_version(),
            "numpy": np.__version__,
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

    # Try to get CPU frequency info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    meta["cpu_model"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    # Try to get pyerfa version
    try:
        import erfa
        meta["toolchain"]["pyerfa"] = erfa.__version__
    except Exception:
        pass

    return meta


def _get_git_branch(repo_path: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


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
    # --- Direction-vector transform experiments (BPN-style metrics) ---
    for exp_name, title in [
        ("frame_bias", "Frame Bias (ICRS → Mean J2000)"),
        ("precession", "Precession (Mean J2000 → Mean of Date)"),
        ("nutation", "Nutation (Mean of Date → True of Date)"),
        ("icrs_ecl_j2000", "ICRS → Ecliptic J2000"),
    ]:
        exp_results = [r for r in all_results if r.get("experiment") == exp_name]
        if not exp_results:
            continue

        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Library | Ang p50 (mas) | Ang p99 (mas) | Ang max (mas) | Closure p99 (rad) | Perf (ns/op) | Speedup vs ref |")
        lines.append("|---------|---------------|---------------|---------------|-------------------|--------------|----------------|")

        for r in exp_results:
            lib = r.get("candidate_library", "?")
            acc = r.get("accuracy", {})
            ang = acc.get("angular_error_mas", {})
            clo = acc.get("closure_error_rad", {})
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
                f"| {fmt(clo.get('p99'), 2)} "
                f"| {fmt(perf.get('per_op_ns'), 1)} "
                f"| {speedup} |"
            )
        lines.append("")

    # --- Angular experiments (equ_ecl, equ_horizontal, solar_position, etc.) ---
    for exp_name, title in [
        ("equ_ecl", "Equatorial ↔ Ecliptic Transform"),
        ("equ_horizontal", "Equatorial → Horizontal (AltAz)"),
        ("solar_position", "Sun Geocentric Position"),
        ("lunar_position", "Moon Geocentric Position"),
        ("icrs_ecl_tod", "ICRS → Ecliptic of Date"),
        ("horiz_to_equ", "Horizontal → Equatorial (AltAz → RA/Dec)"),
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

def run_experiment_frame_rotation_bpn(n: int, seed: int, run_perf: bool = True,
                                      perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """
    Run the frame_rotation_bpn experiment end-to-end.

    Reference: ERFA (IAU 2006/2000A BPN matrix)
    Candidates: Siderust, Astropy, libnova
    """
    exp_name = "frame_rotation_bpn"
    total_steps = 6 + (8 if run_perf else 0)  # gen + 4 adapters + accuracy + perf
    step = 0

    progress(f"Starting experiment (N={n}, seed={seed})", exp_name, "start", total_steps, step)

    # 1) Generate inputs
    step += 1
    progress("Generating inputs...", exp_name, "generate_inputs", total_steps, step)
    epochs, directions, labels = generate_frame_rotation_inputs(n, seed)
    input_text = format_bpn_input(epochs, directions)

    # Compute dataset fingerprint for reproducibility
    ds_fingerprint = dataset_fingerprint({
        "experiment": exp_name, "n": n, "seed": seed,
        "epochs_hash": hashlib.sha256(epochs.tobytes()).hexdigest()[:12],
    })

    # 2) Run adapters
    adapters = {}
    adapter_list = [
        ("erfa", [str(ERFA_BIN)], "reference"),
        ("siderust", [str(SIDERUST_BIN)], "candidate"),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)], "candidate"),
        ("libnova", [str(LIBNOVA_BIN)], "candidate"),
    ]

    for lib, cmd, role in adapter_list:
        step += 1
        progress(f"Running {lib} adapter ({role})...", exp_name, f"adapter_{lib}", total_steps, step)
        adapters[lib] = run_adapter(cmd, input_text, lib)

    # 3) Compute accuracy metrics
    step += 1
    progress("Computing accuracy metrics...", exp_name, "accuracy", total_steps, step)
    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        progress("ERFA adapter failed — cannot compute accuracy.", exp_name, "error")
        return results

    meta = run_metadata()
    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        accuracy = compute_accuracy_metrics(
            ref_data["cases"], cand_data["cases"], "erfa", lib
        )

        result = {
            "experiment": exp_name,
            "candidate_library": lib,
            "reference_library": "erfa",
            "description": EXPERIMENT_DESCRIPTIONS.get(exp_name, {}),
            "alignment": alignment_checklist(exp_name),
            "inputs": {
                "count": n, "seed": seed,
                "epoch_range": [f"JD {min(epochs):.1f}", f"JD {max(epochs):.1f}"],
                "dataset_fingerprint": ds_fingerprint,
            },
            "accuracy": accuracy,
            "performance": {},
            "reference_performance": {},
            "benchmark_config": {
                "perf_rounds": perf_rounds if run_perf else 0,
                "perf_warmup": DEFAULT_PERF_WARMUP,
                "perf_enabled": run_perf,
            },
            "run_metadata": meta,
        }
        results.append(result)

    # 4) Performance measurement (multi-sample)
    if run_perf:
        perf_n = max(MIN_PERF_N, min(n, 10000))
        progress(f"Running performance tests (N={perf_n}, {perf_rounds} rounds)...",
                 exp_name, "performance", total_steps, step + 1)

        # Regenerate inputs if perf_n > n
        if perf_n <= n:
            perf_epochs, perf_dirs = epochs[:perf_n], directions[:perf_n]
        else:
            perf_epochs, perf_dirs, _ = generate_frame_rotation_inputs(perf_n, seed)
        perf_input = format_bpn_perf_input(perf_epochs, perf_dirs)

        for lib, cmd in [
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            step += 1
            progress(f"Timing {lib} ({perf_rounds} rounds)...", exp_name, f"perf_{lib}", total_steps, step)
            perf_data = run_multi_sample_perf(cmd, perf_input, f"{lib}_perf", rounds=perf_rounds)
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = perf_data

        # Time the reference (erfa) with multi-sample
        step += 1
        progress(f"Timing erfa reference ({perf_rounds} rounds)...", exp_name, "perf_erfa", total_steps, step)
        perf_erfa = run_multi_sample_perf([str(ERFA_BIN)], perf_input, "erfa_perf", rounds=perf_rounds)
        if perf_erfa:
            for r in results:
                r["reference_performance"] = perf_erfa

    progress("Experiment complete.", exp_name, "done", total_steps, total_steps)
    return results


def _run_generic_experiment(exp_name: str, n: int, seed: int, run_perf: bool = True,
                             perf_rounds: int = DEFAULT_PERF_ROUNDS,
                             input_gen_fn=None, input_fmt_fn=None, perf_fmt_fn=None,
                             accuracy_fn=None, accuracy_kwargs=None):
    """Generic experiment runner with progress tracking, multi-sample perf, and metadata.

    This consolidates the common pattern shared by all experiments.
    """
    if accuracy_kwargs is None:
        accuracy_kwargs = {}

    total_steps = 6 + (5 if run_perf else 0)
    step = 0

    progress(f"Starting experiment (N={n}, seed={seed})", exp_name, "start", total_steps, step)

    # 1) Generate inputs
    step += 1
    progress("Generating inputs...", exp_name, "generate_inputs", total_steps, step)
    inputs = input_gen_fn(n, seed)
    input_text = input_fmt_fn(*inputs) if not isinstance(inputs, str) else inputs

    # Dataset fingerprint
    ds_fp_data = {"experiment": exp_name, "n": n, "seed": seed}
    if isinstance(inputs, tuple) and len(inputs) > 0:
        for i, inp in enumerate(inputs):
            if isinstance(inp, np.ndarray):
                ds_fp_data[f"input_{i}_hash"] = hashlib.sha256(inp.tobytes()).hexdigest()[:12]
    ds_fingerprint = dataset_fingerprint(ds_fp_data)

    # 2) Run adapters
    adapters = {}
    adapter_list = [
        ("erfa", [str(ERFA_BIN)]),
        ("siderust", [str(SIDERUST_BIN)]),
        ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
        ("libnova", [str(LIBNOVA_BIN)]),
    ]

    for lib, cmd in adapter_list:
        step += 1
        role = "reference" if lib == "erfa" else "candidate"
        progress(f"Running {lib} adapter ({role})...", exp_name, f"adapter_{lib}", total_steps, step)
        adapters[lib] = run_adapter(cmd, input_text, lib)

    # 3) Compute accuracy
    step += 1
    progress("Computing accuracy metrics...", exp_name, "accuracy", total_steps, step)
    results = []
    ref_data = adapters.get("erfa")
    if ref_data is None:
        progress("ERFA adapter failed — cannot compute accuracy.", exp_name, "error")
        return results

    meta = run_metadata()
    for lib in ["siderust", "astropy", "libnova"]:
        cand_data = adapters.get(lib)
        if cand_data is None:
            continue

        # Skip libraries that reported "skipped" (e.g., libnova for frame_bias)
        if isinstance(cand_data, dict) and cand_data.get("skipped"):
            progress(f"{lib}: skipped ({cand_data.get('reason', 'no reason')})", exp_name, f"skip_{lib}")
            continue

        accuracy = accuracy_fn(ref_data["cases"], cand_data["cases"], "erfa", lib, **accuracy_kwargs)

        result = {
            "experiment": exp_name,
            "candidate_library": lib,
            "reference_library": "erfa",
            "description": EXPERIMENT_DESCRIPTIONS.get(exp_name, {}),
            "alignment": alignment_checklist(exp_name),
            "inputs": {
                "count": n, "seed": seed,
                "dataset_fingerprint": ds_fingerprint,
            },
            "accuracy": accuracy,
            "performance": {},
            "reference_performance": {},
            "benchmark_config": {
                "perf_rounds": perf_rounds if run_perf else 0,
                "perf_warmup": DEFAULT_PERF_WARMUP,
                "perf_enabled": run_perf,
            },
            "run_metadata": meta,
        }
        results.append(result)

    # 4) Performance measurement (multi-sample)
    if run_perf and perf_fmt_fn is not None:
        perf_n = max(MIN_PERF_N, min(n, 10000))
        progress(f"Running performance tests (N={perf_n}, {perf_rounds} rounds)...",
                 exp_name, "performance", total_steps, step + 1)

        # Generate perf input
        perf_inputs = input_gen_fn(perf_n, seed)  # Regenerate for perf_n
        perf_input = perf_fmt_fn(*perf_inputs) if not isinstance(perf_inputs, str) else perf_inputs

        for lib, cmd in [
            ("siderust", [str(SIDERUST_BIN)]),
            ("astropy", [sys.executable, str(ASTROPY_SCRIPT)]),
            ("libnova", [str(LIBNOVA_BIN)]),
        ]:
            step += 1
            progress(f"Timing {lib} ({perf_rounds} rounds)...", exp_name, f"perf_{lib}", total_steps, step)
            perf_data = run_multi_sample_perf(cmd, perf_input, f"{lib}_perf", rounds=perf_rounds)
            if perf_data:
                for r in results:
                    if r["candidate_library"] == lib:
                        r["performance"] = perf_data

        step += 1
        progress(f"Timing erfa reference ({perf_rounds} rounds)...", exp_name, "perf_erfa", total_steps, step)
        perf_erfa = run_multi_sample_perf([str(ERFA_BIN)], perf_input, "erfa_perf", rounds=perf_rounds)
        if perf_erfa:
            for r in results:
                r["reference_performance"] = perf_erfa

    progress("Experiment complete.", exp_name, "done", total_steps, total_steps)
    return results


def run_experiment_gmst_era(n: int, seed: int, run_perf: bool = True,
                            perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the gmst_era experiment end-to-end."""
    return _run_generic_experiment(
        exp_name="gmst_era", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_gmst_era_inputs,
        input_fmt_fn=format_gmst_input,
        perf_fmt_fn=format_gmst_perf_input,
        accuracy_fn=compute_gmst_accuracy,
    )


def run_experiment_equ_ecl(n: int, seed: int, run_perf: bool = True,
                           perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the equ_ecl experiment: equatorial ↔ ecliptic coordinate transform."""
    return _run_generic_experiment(
        exp_name="equ_ecl", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_equ_ecl_inputs,
        input_fmt_fn=format_equ_ecl_input,
        perf_fmt_fn=format_equ_ecl_perf_input,
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "ecl_lon_rad", "dec_key": "ecl_lat_rad"},
    )


def run_experiment_equ_horizontal(n: int, seed: int, run_perf: bool = True,
                                   perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the equ_horizontal experiment: equatorial → horizontal (AltAz)."""
    return _run_generic_experiment(
        exp_name="equ_horizontal", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_equ_horizontal_inputs,
        input_fmt_fn=format_equ_horizontal_input,
        perf_fmt_fn=format_equ_horizontal_perf_input,
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "az_rad", "dec_key": "alt_rad"},
    )


def run_experiment_solar_position(n: int, seed: int, run_perf: bool = True,
                                   perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the solar_position experiment: geocentric Sun RA/Dec."""
    def _solar_gen(n, seed):
        epochs = generate_solar_position_inputs(n, seed)
        return (epochs,)

    return _run_generic_experiment(
        exp_name="solar_position", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=_solar_gen,
        input_fmt_fn=lambda epochs: format_solar_position_input(epochs),
        perf_fmt_fn=lambda epochs: format_solar_position_perf_input(epochs),
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "ra_rad", "dec_key": "dec_rad", "extra_keys": ["dist_au"]},
    )


def run_experiment_lunar_position(n: int, seed: int, run_perf: bool = True,
                                   perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the lunar_position experiment: geocentric Moon RA/Dec."""
    def _lunar_gen(n, seed):
        epochs = generate_lunar_position_inputs(n, seed)
        return (epochs,)

    return _run_generic_experiment(
        exp_name="lunar_position", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=_lunar_gen,
        input_fmt_fn=lambda epochs: format_lunar_position_input(epochs),
        perf_fmt_fn=lambda epochs: format_lunar_position_perf_input(epochs),
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "ra_rad", "dec_key": "dec_rad", "extra_keys": ["dist_km"]},
    )


def run_experiment_kepler_solver(n: int, seed: int, run_perf: bool = True,
                                  perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the kepler_solver experiment: Kepler's equation M→E→ν."""
    return _run_generic_experiment(
        exp_name="kepler_solver", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_kepler_inputs,
        input_fmt_fn=format_kepler_input,
        perf_fmt_fn=format_kepler_perf_input,
        accuracy_fn=compute_kepler_accuracy,
    )


def run_experiment_frame_bias(n: int, seed: int, run_perf: bool = True,
                               perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the frame_bias experiment: ICRS → Mean J2000 frame bias."""
    return _run_generic_experiment(
        exp_name="frame_bias", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_direction_vector_inputs,
        input_fmt_fn=format_frame_bias_input,
        perf_fmt_fn=format_frame_bias_perf_input,
        accuracy_fn=compute_accuracy_metrics,
    )


def run_experiment_precession(n: int, seed: int, run_perf: bool = True,
                               perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the precession experiment: Mean J2000 → Mean of Date."""
    return _run_generic_experiment(
        exp_name="precession", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_direction_vector_inputs,
        input_fmt_fn=format_precession_input,
        perf_fmt_fn=format_precession_perf_input,
        accuracy_fn=compute_accuracy_metrics,
    )


def run_experiment_nutation(n: int, seed: int, run_perf: bool = True,
                             perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the nutation experiment: Mean of Date → True of Date."""
    return _run_generic_experiment(
        exp_name="nutation", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_direction_vector_inputs,
        input_fmt_fn=format_nutation_input,
        perf_fmt_fn=format_nutation_perf_input,
        accuracy_fn=compute_accuracy_metrics,
    )


def run_experiment_icrs_ecl_j2000(n: int, seed: int, run_perf: bool = True,
                                    perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the icrs_ecl_j2000 experiment: ICRS → Ecliptic J2000."""
    return _run_generic_experiment(
        exp_name="icrs_ecl_j2000", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_direction_vector_inputs,
        input_fmt_fn=format_icrs_ecl_j2000_input,
        perf_fmt_fn=format_icrs_ecl_j2000_perf_input,
        accuracy_fn=compute_accuracy_metrics,
    )


def run_experiment_icrs_ecl_tod(n: int, seed: int, run_perf: bool = True,
                                 perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the icrs_ecl_tod experiment: ICRS → Ecliptic of Date."""
    return _run_generic_experiment(
        exp_name="icrs_ecl_tod", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_equ_ecl_inputs,
        input_fmt_fn=format_icrs_ecl_tod_input,
        perf_fmt_fn=format_icrs_ecl_tod_perf_input,
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "ecl_lon_rad", "dec_key": "ecl_lat_rad"},
    )


def run_experiment_horiz_to_equ(n: int, seed: int, run_perf: bool = True,
                                 perf_rounds: int = DEFAULT_PERF_ROUNDS):
    """Run the horiz_to_equ experiment: Horizontal → Equatorial."""
    return _run_generic_experiment(
        exp_name="horiz_to_equ", n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
        input_gen_fn=generate_horiz_to_equ_inputs,
        input_fmt_fn=format_horiz_to_equ_input,
        perf_fmt_fn=format_horiz_to_equ_perf_input,
        accuracy_fn=compute_angular_accuracy,
        accuracy_kwargs={"ra_key": "ra_rad", "dec_key": "dec_rad"},
    )


# 13 new matrix experiment runner functions
def _make_dir_experiment_runner(exp_name, fmt_input, fmt_perf_input):
    """Factory for direction-vector experiment runners."""
    def runner(n, seed, run_perf=True, perf_rounds=DEFAULT_PERF_ROUNDS):
        return _run_generic_experiment(
            exp_name=exp_name, n=n, seed=seed, run_perf=run_perf, perf_rounds=perf_rounds,
            input_gen_fn=generate_direction_vector_inputs,
            input_fmt_fn=fmt_input,
            perf_fmt_fn=fmt_perf_input,
            accuracy_fn=compute_accuracy_metrics,
        )
    return runner

run_experiment_inv_frame_bias = _make_dir_experiment_runner("inv_frame_bias", format_inv_frame_bias_input, format_inv_frame_bias_perf_input)
run_experiment_inv_precession = _make_dir_experiment_runner("inv_precession", format_inv_precession_input, format_inv_precession_perf_input)
run_experiment_inv_nutation = _make_dir_experiment_runner("inv_nutation", format_inv_nutation_input, format_inv_nutation_perf_input)
run_experiment_inv_bpn = _make_dir_experiment_runner("inv_bpn", format_inv_bpn_input, format_inv_bpn_perf_input)
run_experiment_inv_icrs_ecl_j2000 = _make_dir_experiment_runner("inv_icrs_ecl_j2000", format_inv_icrs_ecl_j2000_input, format_inv_icrs_ecl_j2000_perf_input)
run_experiment_obliquity = _make_dir_experiment_runner("obliquity", format_obliquity_input, format_obliquity_perf_input)
run_experiment_inv_obliquity = _make_dir_experiment_runner("inv_obliquity", format_inv_obliquity_input, format_inv_obliquity_perf_input)
run_experiment_bias_precession = _make_dir_experiment_runner("bias_precession", format_bias_precession_input, format_bias_precession_perf_input)
run_experiment_inv_bias_precession = _make_dir_experiment_runner("inv_bias_precession", format_inv_bias_precession_input, format_inv_bias_precession_perf_input)
run_experiment_precession_nutation = _make_dir_experiment_runner("precession_nutation", format_precession_nutation_input, format_precession_nutation_perf_input)
run_experiment_inv_precession_nutation = _make_dir_experiment_runner("inv_precession_nutation", format_inv_precession_nutation_input, format_inv_precession_nutation_perf_input)
run_experiment_inv_icrs_ecl_tod_dir = _make_dir_experiment_runner("inv_icrs_ecl_tod", format_inv_icrs_ecl_tod_dir_input, format_inv_icrs_ecl_tod_dir_perf_input)
run_experiment_inv_equ_ecl_dir = _make_dir_experiment_runner("inv_equ_ecl", format_inv_equ_ecl_dir_input, format_inv_equ_ecl_dir_perf_input)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: list, experiment: str, timestamp_str: str | None = None):
    """Write result JSON files and summary table."""
    # Use provided timestamp or generate new one (for backward compatibility)
    if timestamp_str is None:
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
        "frame_bias", "precession", "nutation", "icrs_ecl_j2000",
        "icrs_ecl_tod", "horiz_to_equ",
        # 13 new matrix experiments
        "inv_frame_bias", "inv_precession", "inv_nutation", "inv_bpn",
        "inv_icrs_ecl_j2000", "obliquity", "inv_obliquity",
        "bias_precession", "inv_bias_precession",
        "precession_nutation", "inv_precession_nutation",
        "inv_icrs_ecl_tod", "inv_equ_ecl",
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
    parser.add_argument("--perf-rounds", type=int, default=DEFAULT_PERF_ROUNDS,
                        help=f"Number of performance timing rounds (default: {DEFAULT_PERF_ROUNDS})")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: fewer rounds, smaller N for faster execution")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip automatic rebuild of siderust adapter")
    args = parser.parse_args()

    # CI mode overrides
    if args.ci:
        if args.n == 1000:  # Only override if default
            args.n = 100
        if args.perf_rounds == DEFAULT_PERF_ROUNDS:
            args.perf_rounds = 2

    if not args.no_build:
        ensure_siderust_adapter_built()

    # Resolve which experiments to run (--experiments takes precedence)
    raw = args.experiments or args.experiment or "all"
    if raw == "all":
        experiments_to_run = all_experiments
    else:
        experiments_to_run = [e.strip() for e in raw.split(",") if e.strip()]

    # Generate single timestamp for this entire run
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    # Print run configuration banner
    run_perf = not args.no_perf
    print(f"\n{'='*70}")
    print(f" Siderust Lab — Benchmark Run")
    print(f"{'='*70}")
    print(f"  Timestamp:     {run_timestamp}")
    print(f"  Experiments:   {', '.join(experiments_to_run)}")
    print(f"  N (cases):     {args.n}")
    print(f"  Seed:          {args.seed}")
    print(f"  Performance:   {'enabled (' + str(args.perf_rounds) + ' rounds)' if run_perf else 'disabled'}")
    print(f"  CI mode:       {'yes' if args.ci else 'no'}")
    print(f"{'='*70}\n")

    all_results = []

    dispatch = {
        "frame_rotation_bpn": lambda: run_experiment_frame_rotation_bpn(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "gmst_era": lambda: run_experiment_gmst_era(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "equ_ecl": lambda: run_experiment_equ_ecl(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "equ_horizontal": lambda: run_experiment_equ_horizontal(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "solar_position": lambda: run_experiment_solar_position(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "lunar_position": lambda: run_experiment_lunar_position(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "kepler_solver": lambda: run_experiment_kepler_solver(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "frame_bias": lambda: run_experiment_frame_bias(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "precession": lambda: run_experiment_precession(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "nutation": lambda: run_experiment_nutation(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "icrs_ecl_j2000": lambda: run_experiment_icrs_ecl_j2000(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "icrs_ecl_tod": lambda: run_experiment_icrs_ecl_tod(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        "horiz_to_equ": lambda: run_experiment_horiz_to_equ(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds
        ),
        # 13 new matrix experiments
        "inv_frame_bias": lambda: run_experiment_inv_frame_bias(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_precession": lambda: run_experiment_inv_precession(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_nutation": lambda: run_experiment_inv_nutation(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_bpn": lambda: run_experiment_inv_bpn(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_icrs_ecl_j2000": lambda: run_experiment_inv_icrs_ecl_j2000(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "obliquity": lambda: run_experiment_obliquity(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_obliquity": lambda: run_experiment_inv_obliquity(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "bias_precession": lambda: run_experiment_bias_precession(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_bias_precession": lambda: run_experiment_inv_bias_precession(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "precession_nutation": lambda: run_experiment_precession_nutation(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_precession_nutation": lambda: run_experiment_inv_precession_nutation(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_icrs_ecl_tod": lambda: run_experiment_inv_icrs_ecl_tod_dir(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
        "inv_equ_ecl": lambda: run_experiment_inv_equ_ecl_dir(
            args.n, args.seed, run_perf=run_perf, perf_rounds=args.perf_rounds),
    }

    total_experiments = len(experiments_to_run)
    for exp_idx, exp in enumerate(experiments_to_run, 1):
        progress(f"Experiment {exp_idx}/{total_experiments}: {exp}",
                 step="experiment_start", current_step=exp_idx, total_steps=total_experiments)

        runner_fn = dispatch.get(exp)
        if runner_fn is None:
            print(f"Unknown experiment: {exp}", file=sys.stderr)
            continue

        results = runner_fn()

        if results:
            write_results(results, exp, run_timestamp)
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
                    ns = perf["per_op_ns"]
                    ops = perf.get("throughput_ops_s", 0)
                    cv = perf.get("per_op_ns_cv_pct", 0)
                    rounds = perf.get("rounds", 1)
                    print(f"    Performance: {ns:.0f} ns/op (median, {rounds} rounds, CV={cv:.1f}%)")
                    if perf.get("warnings"):
                        for w in perf["warnings"]:
                            print(f"    ⚠ {w}")

    if all_results:
        # Write combined summary
        summary = generate_summary_table(all_results)
        print(f"\n{'='*70}")
        print(" Combined Summary Table")
        print(f"{'='*70}")
        print(summary)

        # Write run manifest
        manifest = {
            "run_id": run_timestamp,
            "config": {
                "experiments": experiments_to_run,
                "n": args.n,
                "seed": args.seed,
                "perf_enabled": run_perf,
                "perf_rounds": args.perf_rounds,
                "ci_mode": args.ci,
            },
            "metadata": run_metadata(),
            "experiment_count": len(set(r["experiment"] for r in all_results)),
            "total_results": len(all_results),
        }
        manifest_path = RESULTS_DIR / run_timestamp / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  ✓ Run manifest: {manifest_path.relative_to(LAB_ROOT)}")


if __name__ == "__main__":
    main()
