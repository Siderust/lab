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


def run_experiment_gmst_era(n: int, seed: int):
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
            "run_metadata": run_metadata(),
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: list, experiment: str):
    """Write result JSON files and summary table."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = RESULTS_DIR / date_str / experiment
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
        f.write(f"Date: {date_str}\n\n")
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
    parser = argparse.ArgumentParser(description="Siderust Lab Orchestrator")
    parser.add_argument("--experiment", default="frame_rotation_bpn",
                        choices=["frame_rotation_bpn", "gmst_era", "all"],
                        help="Which experiment to run")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for input generation")
    parser.add_argument("--no-perf", action="store_true",
                        help="Skip performance tests")
    args = parser.parse_args()

    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = ["frame_rotation_bpn", "gmst_era"]
    else:
        experiments_to_run = [args.experiment]

    all_results = []

    for exp in experiments_to_run:
        if exp == "frame_rotation_bpn":
            results = run_experiment_frame_rotation_bpn(
                args.n, args.seed, run_perf=not args.no_perf
            )
        elif exp == "gmst_era":
            results = run_experiment_gmst_era(args.n, args.seed)
        else:
            print(f"Unknown experiment: {exp}", file=sys.stderr)
            continue

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
