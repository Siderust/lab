"""Performance Matrix API — 2D FROM×TO grid for coordinate transforms.

Provides two matrices (performance + accuracy) for each comparison library.
Rows = source frames, columns = target frames, cells = experiment data.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["performance-matrix"])

# ------------------------------------------------------------------
# Frame definitions (matrix axes)
# ------------------------------------------------------------------

# Core 7 frames (have benchmark implementations)
CORE_FRAMES = [
    "ICRS",
    "EqMeanJ2000",
    "EqMeanOfDate",
    "EqTrueOfDate",
    "EclMeanJ2000",
    "EclTrueOfDate",
    "Horizontal",
]

# Extended frames (shown on axes but marked "Not implemented")
EXTENDED_FRAMES = [
    "GCRS", "CIRS", "TIRS", "ITRF", "Galactic",
]

ALL_FRAMES = CORE_FRAMES + EXTENDED_FRAMES

# ------------------------------------------------------------------
# Experiment → matrix cell mapping
# Maps experiment_id → (source_frame, target_frame)
# ------------------------------------------------------------------

EXPERIMENT_CELL_MAP: dict[str, tuple[str, str]] = {
    # Original experiments (forward directions)
    "frame_bias":           ("ICRS",            "EqMeanJ2000"),
    "precession":           ("EqMeanJ2000",     "EqMeanOfDate"),
    "nutation":             ("EqMeanOfDate",     "EqTrueOfDate"),
    "frame_rotation_bpn":   ("ICRS",            "EqTrueOfDate"),
    "icrs_ecl_j2000":       ("ICRS",            "EclMeanJ2000"),
    "icrs_ecl_tod":         ("ICRS",            "EclTrueOfDate"),
    "equ_ecl":              ("EqMeanOfDate",     "EclTrueOfDate"),
    "equ_horizontal":       ("EqTrueOfDate",     "Horizontal"),
    "horiz_to_equ":         ("Horizontal",       "EqTrueOfDate"),
    # 13 new inverse/composed/obliquity experiments
    "inv_frame_bias":       ("EqMeanJ2000",     "ICRS"),
    "inv_precession":       ("EqMeanOfDate",     "EqMeanJ2000"),
    "inv_nutation":         ("EqTrueOfDate",     "EqMeanOfDate"),
    "inv_bpn":              ("EqTrueOfDate",     "ICRS"),
    "inv_icrs_ecl_j2000":   ("EclMeanJ2000",    "ICRS"),
    "obliquity":            ("EclMeanJ2000",    "EqMeanJ2000"),
    "inv_obliquity":        ("EqMeanJ2000",     "EclMeanJ2000"),
    "bias_precession":      ("ICRS",            "EqMeanOfDate"),
    "inv_bias_precession":  ("EqMeanOfDate",     "ICRS"),
    "precession_nutation":  ("EqMeanJ2000",     "EqTrueOfDate"),
    "inv_precession_nutation": ("EqTrueOfDate",  "EqMeanJ2000"),
    "inv_icrs_ecl_tod":     ("EclTrueOfDate",   "ICRS"),
    "inv_equ_ecl":          ("EclTrueOfDate",   "EqMeanOfDate"),
}

# Reverse map: (source, target) → experiment_id
CELL_EXPERIMENT_MAP: dict[tuple[str, str], str] = {v: k for k, v in EXPERIMENT_CELL_MAP.items()}

# All coordinate-transform experiment IDs
ALL_COORD_EXPERIMENTS = list(EXPERIMENT_CELL_MAP.keys())


def _extract_per_op_ns(perf) -> float | None:
    """Extract per_op_ns from either a Pydantic model or a dict."""
    if perf is None:
        return None
    if isinstance(perf, dict):
        return perf.get("per_op_ns")
    return getattr(perf, "per_op_ns", None)


def _extract_accuracy_metric(accuracy) -> dict | None:
    """Extract the primary accuracy metric (angular_error_mas or angular_sep_arcsec).

    Returns a dict with p50, p99, max in mas (milliarcseconds).
    """
    if accuracy is None:
        return None

    # Handle both dict and Pydantic model
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # Try angular_error_mas first (direction-vector experiments)
    ang = _get(accuracy, "angular_error_mas")
    if ang is not None:
        p50 = _get(ang, "p50")
        if p50 is not None:
            return {
                "p50_mas": _get(ang, "p50"),
                "p99_mas": _get(ang, "p99"),
                "max_mas": _get(ang, "max"),
                "metric_type": "angular_error_mas",
            }

    # Try angular_sep_arcsec (RA/Dec experiments) — convert to mas
    sep = _get(accuracy, "angular_sep_arcsec")
    if sep is not None:
        p50 = _get(sep, "p50")
        if p50 is not None:
            return {
                "p50_mas": _get(sep, "p50") * 1000.0,
                "p99_mas": _get(sep, "p99") * 1000.0,
                "max_mas": _get(sep, "max") * 1000.0,
                "metric_type": "angular_sep_arcsec_to_mas",
            }

    return None


@router.get("/runs/{run_id}/performance-matrix")
async def get_performance_matrix(run_id: str):
    """Return 2D FROM×TO performance & accuracy matrices.

    Response shape:
    {
        run_id, frames, experiment_map,
        perf_matrix: { siderust_vs_erfa: { "ICRS→EqMeanJ2000": {speedup, siderust_ns, other_ns}, ... }, ... },
        accuracy_matrix: { siderust_vs_erfa: { "ICRS→EqMeanJ2000": {p50_mas, ...}, ... }, ... },
    }
    """
    from ..main import loader

    detail = loader.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Collect all data per experiment per library
    exp_perf: dict[str, dict[str, float]] = {}      # exp → lib → ns
    exp_accuracy: dict[str, dict[str, dict]] = {}    # exp → lib → accuracy_dict

    for exp_id in ALL_COORD_EXPERIMENTS:
        results = detail.experiments.get(exp_id, [])
        exp_perf[exp_id] = {}
        exp_accuracy[exp_id] = {}

        for r in results:
            lib = r.candidate_library

            # Performance
            ns = _extract_per_op_ns(r.performance)
            if ns:
                exp_perf[exp_id][lib] = ns
            ref_ns = _extract_per_op_ns(r.reference_performance)
            if ref_ns and "erfa" not in exp_perf[exp_id]:
                exp_perf[exp_id]["erfa"] = ref_ns

            # Accuracy
            acc = _extract_accuracy_metric(r.accuracy)
            if acc:
                exp_accuracy[exp_id][lib] = acc

    # Build 2D grid for each comparison library
    comparison_libs = ["erfa", "astropy", "libnova", "anise"]
    perf_matrix: dict[str, dict[str, dict]] = {}
    accuracy_matrix: dict[str, dict[str, dict]] = {}

    for other_lib in comparison_libs:
        perf_cells: dict[str, dict] = {}
        acc_cells: dict[str, dict] = {}

        for src in ALL_FRAMES:
            for dst in ALL_FRAMES:
                if src == dst:
                    continue  # diagonal — skip

                cell_key = f"{src}→{dst}"
                exp_id = CELL_EXPERIMENT_MAP.get((src, dst))

                # Check if this cell is in the extended region
                is_extended = src in EXTENDED_FRAMES or dst in EXTENDED_FRAMES

                if is_extended:
                    perf_cells[cell_key] = {"status": "not_implemented"}
                    acc_cells[cell_key] = {"status": "not_implemented"}
                    continue

                if exp_id is None:
                    # No experiment maps to this cell (e.g., EqMeanJ2000→Horizontal)
                    perf_cells[cell_key] = {"status": "no_experiment"}
                    acc_cells[cell_key] = {"status": "no_experiment"}
                    continue

                # Build performance cell
                sr_ns = exp_perf.get(exp_id, {}).get("siderust")
                other_ns = exp_perf.get(exp_id, {}).get(other_lib)

                if sr_ns and other_ns:
                    perf_cells[cell_key] = {
                        "status": "available",
                        "experiment": exp_id,
                        "siderust_ns": round(sr_ns, 1),
                        "other_ns": round(other_ns, 1),
                        "speedup": round(other_ns / sr_ns, 2),
                    }
                else:
                    # Check if experiment was skipped by this lib
                    has_results = exp_id in detail.experiments
                    perf_cells[cell_key] = {
                        "status": "skipped" if has_results else "no_data",
                        "experiment": exp_id,
                    }

                # Build accuracy cell
                acc_data = exp_accuracy.get(exp_id, {}).get(other_lib if other_lib != "erfa" else "siderust")
                if acc_data:
                    acc_cells[cell_key] = {
                        "status": "available",
                        "experiment": exp_id,
                        **acc_data,
                    }
                else:
                    has_results = exp_id in detail.experiments
                    acc_cells[cell_key] = {
                        "status": "skipped" if has_results else "no_data",
                        "experiment": exp_id,
                    }

        perf_matrix[f"siderust_vs_{other_lib}"] = perf_cells
        accuracy_matrix[f"siderust_vs_{other_lib}"] = acc_cells

    # Build experiment map for UI reference
    experiment_map = {
        exp_id: {"from": src, "to": dst}
        for exp_id, (src, dst) in EXPERIMENT_CELL_MAP.items()
    }

    return {
        "run_id": run_id,
        "frames": ALL_FRAMES,
        "core_frames": CORE_FRAMES,
        "extended_frames": EXTENDED_FRAMES,
        "experiment_map": experiment_map,
        "cell_experiment_map": {f"{src}→{dst}": exp for (src, dst), exp in CELL_EXPERIMENT_MAP.items()},
        "perf_matrix": perf_matrix,
        "accuracy_matrix": accuracy_matrix,
    }
