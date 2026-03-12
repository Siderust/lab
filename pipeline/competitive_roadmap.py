#!/usr/bin/env python3
"""
Competitive roadmap for the lab's latest benchmark results.

Reads `latest_results/` or a specific run directory and answers:
  - Which expected experiments are missing?
  - Where is Siderust not the most accurate candidate?
  - Where is Siderust not the fastest candidate?
  - What concrete work is implied by those gaps?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


EXPECTED_EXPERIMENTS = [
    "frame_rotation_bpn",
    "gmst_era",
    "equ_ecl",
    "equ_horizontal",
    "solar_position",
    "lunar_position",
    "mercury_position",
    "venus_position",
    "mars_position",
    "jupiter_position",
    "saturn_position",
    "uranus_position",
    "neptune_position",
    "kepler_solver",
    "frame_bias",
    "precession",
    "nutation",
    "icrs_ecl_j2000",
    "icrs_ecl_tod",
    "horiz_to_equ",
    "inv_frame_bias",
    "inv_precession",
    "inv_nutation",
    "inv_bpn",
    "inv_icrs_ecl_j2000",
    "obliquity",
    "inv_obliquity",
    "bias_precession",
    "inv_bias_precession",
    "precession_nutation",
    "inv_precession_nutation",
    "inv_icrs_ecl_tod",
    "inv_equ_ecl",
]

POSITION_EXPERIMENTS = {
    "solar_position",
    "lunar_position",
    "mercury_position",
    "venus_position",
    "mars_position",
    "jupiter_position",
    "saturn_position",
    "uranus_position",
    "neptune_position",
}

ACCURACY_PRIORITY = [
    "angular_error_mas",
    "angular_sep_arcsec",
    "gmst_error_arcsec",
    "E_error_rad",
    "consistency_error_rad",
    "nu_error_rad",
]

PERF_PRIORITY = [
    "per_op_ns_median",
    "per_op_ns",
    "per_op_ns_mean",
]


@dataclass
class Gap:
    experiment: str
    detail: str
    action: str


def load_results(root: Path) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        exp_results = []
        for json_path in sorted(child.glob("*.json")):
            try:
                data = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("candidate_library"):
                exp_results.append(data)
        results[child.name] = exp_results
    return results


def performance_value(result: dict) -> float | None:
    perf = result.get("performance")
    if not isinstance(perf, dict):
        return None
    if perf.get("valid") is False:
        return None
    for key in PERF_PRIORITY:
        value = perf.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def accuracy_value(result: dict) -> tuple[str, str, float] | None:
    accuracy = result.get("accuracy")
    if not isinstance(accuracy, dict):
        return None

    for metric_name in ACCURACY_PRIORITY:
        stats = accuracy.get(metric_name)
        if not isinstance(stats, dict):
            continue
        for field in ("p99", "max", "mean", "p50"):
            value = stats.get(field)
            if isinstance(value, (int, float)):
                return metric_name, field, abs(float(value))
    return None


def coverage_action(experiment: str) -> str:
    if experiment in POSITION_EXPERIMENTS:
        return (
            "Run and publish the JPL Horizons-backed ephemeris benchmark for this body, "
            "then use the result to choose between analytic and JPL-backed Siderust modes."
        )
    return (
        "Include this experiment in the latest benchmark run and keep latest_results in sync "
        "so regressions are visible immediately."
    )


def performance_action(experiment: str) -> str:
    if experiment in {"frame_rotation_bpn", "nutation", "precession_nutation"}:
        return (
            "Profile the frame-rotation hot path, cache per-epoch matrices for batched runs, "
            "and remove avoidable cartesian/spherical conversion overhead."
        )
    if experiment in {"equ_ecl", "icrs_ecl_tod", "equ_horizontal", "horiz_to_equ"}:
        return (
            "Add a direct benchmark kernel for this transform instead of routing through the "
            "most generic context-heavy path."
        )
    if experiment in POSITION_EXPERIMENTS:
        return (
            "Separate fast analytic mode from high-accuracy mode and remove repeated ephemeris "
            "object construction from the planetary hot path."
        )
    if experiment == "kepler_solver":
        return (
            "Tune solver branching by eccentricity and avoid fallback work on easy low-e cases."
        )
    return "Profile allocations and dispatch in the measured path, then specialize the hottest code."


def accuracy_action(experiment: str) -> str:
    if experiment == "lunar_position":
        return (
            "Replace the simplified Meeus lunar model with a higher-fidelity path, ideally a "
            "JPL/DE-backed mode, and compare geometric vs astrometric settings explicitly."
        )
    if experiment == "solar_position":
        return (
            "Add a JPL-backed precision mode for the Sun or apply the same astrometric corrections "
            "used by Horizons so the benchmark can target reference-grade accuracy."
        )
    if experiment in POSITION_EXPERIMENTS:
        return (
            "Benchmark Siderust's planetary model against JPL per planet and add a higher-fidelity "
            "ephemeris mode where analytic VSOP/Meeus accuracy is not enough."
        )
    if experiment in {"frame_rotation_bpn", "precession", "nutation", "precession_nutation"}:
        return (
            "Verify matrix composition order and constants against SOFA/IAU reference formulas, "
            "then add regression tests at the worst offending epochs."
        )
    return (
        "Align the implementation exactly to the IAU/SOFA reference model and pin the worst cases "
        "in targeted regression tests."
    )


def summarize(root: Path) -> tuple[list[Gap], list[Gap], list[Gap], dict[str, list[dict]]]:
    results = load_results(root)
    coverage_gaps: list[Gap] = []
    accuracy_gaps: list[Gap] = []
    performance_gaps: list[Gap] = []

    for experiment in EXPECTED_EXPERIMENTS:
        exp_results = results.get(experiment, [])
        if not exp_results:
            coverage_gaps.append(
                Gap(
                    experiment=experiment,
                    detail="No results published for this experiment in the selected run.",
                    action=coverage_action(experiment),
                )
            )
            continue

        siderust_result = next(
            (r for r in exp_results if r.get("candidate_library") == "siderust"),
            None,
        )
        if siderust_result is None:
            coverage_gaps.append(
                Gap(
                    experiment=experiment,
                    detail="Latest run has no Siderust result for this experiment.",
                    action=coverage_action(experiment),
                )
            )
            continue

        perf_candidates = []
        for result in exp_results:
            value = performance_value(result)
            if value is not None:
                perf_candidates.append((result["candidate_library"], value))

        if perf_candidates:
            best_lib, best_ns = min(perf_candidates, key=lambda item: item[1])
            siderust_ns = next(
                (value for lib, value in perf_candidates if lib == "siderust"),
                None,
            )
            if siderust_ns is None:
                coverage_gaps.append(
                    Gap(
                        experiment=experiment,
                        detail="Latest run has no Siderust performance data for this experiment.",
                        action=performance_action(experiment),
                    )
                )
            elif best_lib != "siderust":
                slowdown = siderust_ns / best_ns if best_ns > 0 else float("inf")
                performance_gaps.append(
                    Gap(
                        experiment=experiment,
                        detail=(
                            f"Siderust is slower than {best_lib}: "
                            f"{siderust_ns:.1f} ns/op vs {best_ns:.1f} ns/op "
                            f"({slowdown:.2f}x slower)."
                        ),
                        action=performance_action(experiment),
                    )
                )

        acc_candidates = []
        for result in exp_results:
            metric = accuracy_value(result)
            if metric is not None:
                metric_name, field_name, value = metric
                acc_candidates.append(
                    (result["candidate_library"], metric_name, field_name, value)
                )

        if acc_candidates:
            best_lib, metric_name, field_name, best_value = min(
                acc_candidates, key=lambda item: item[3]
            )
            siderust_metric = next(
                (
                    (metric, field, value)
                    for lib, metric, field, value in acc_candidates
                    if lib == "siderust"
                ),
                None,
            )
            if siderust_metric is None:
                coverage_gaps.append(
                    Gap(
                        experiment=experiment,
                        detail="Latest run has no Siderust accuracy data for this experiment.",
                        action=accuracy_action(experiment),
                    )
                )
            elif best_lib != "siderust":
                _, siderust_field, siderust_value = siderust_metric
                if math.isclose(siderust_value, best_value, rel_tol=1e-9, abs_tol=1e-15):
                    continue
                if best_value > 0:
                    ratio = siderust_value / best_value
                    ratio_text = f"{ratio:.2f}x worse"
                else:
                    ratio_text = "non-zero gap over a zero-error best result"
                accuracy_gaps.append(
                    Gap(
                        experiment=experiment,
                        detail=(
                            f"Siderust trails {best_lib} on {metric_name}.{field_name}: "
                            f"{siderust_value:.3e} vs {best_value:.3e} ({ratio_text})."
                        ),
                        action=accuracy_action(experiment),
                    )
                )

    performance_gaps.sort(
        key=lambda gap: float(gap.detail.split("(")[-1].split("x")[0])
        if "x slower" in gap.detail
        else 0.0,
        reverse=True,
    )
    accuracy_gaps.sort(key=lambda gap: gap.experiment)
    return coverage_gaps, accuracy_gaps, performance_gaps, results


def measured_wins(results: dict[str, list[dict]]) -> tuple[int, int, int]:
    measured = 0
    perf_wins = 0
    acc_wins = 0

    for experiment in EXPECTED_EXPERIMENTS:
        exp_results = results.get(experiment, [])
        if not exp_results:
            continue
        siderust_result = next(
            (r for r in exp_results if r.get("candidate_library") == "siderust"),
            None,
        )
        if siderust_result is None:
            continue

        measured += 1

        perf_candidates = []
        for result in exp_results:
            value = performance_value(result)
            if value is not None:
                perf_candidates.append((result["candidate_library"], value))
        if perf_candidates and min(perf_candidates, key=lambda item: item[1])[0] == "siderust":
            perf_wins += 1

        acc_candidates = []
        for result in exp_results:
            metric = accuracy_value(result)
            if metric is not None:
                _, _, value = metric
                acc_candidates.append((result["candidate_library"], value))
        if acc_candidates and min(acc_candidates, key=lambda item: item[1])[0] == "siderust":
            acc_wins += 1

    return measured, perf_wins, acc_wins


def print_markdown(root: Path, coverage: list[Gap], accuracy: list[Gap], performance: list[Gap], results: dict[str, list[dict]]) -> None:
    measured, perf_wins, acc_wins = measured_wins(results)
    covered = sum(1 for exp in EXPECTED_EXPERIMENTS if results.get(exp))

    print(f"# Competitive Roadmap: {root}")
    print()
    print("## Scorecard")
    print(
        f"- Coverage: {covered}/{len(EXPECTED_EXPERIMENTS)} expected experiments have published results."
    )
    print(f"- Measured Siderust experiments: {measured}")
    print(f"- Performance wins: {perf_wins}/{measured}")
    print(f"- Accuracy wins: {acc_wins}/{measured}")
    print()

    print("## Coverage Gaps")
    if coverage:
        for gap in coverage:
            print(f"- `{gap.experiment}`: {gap.detail} Action: {gap.action}")
    else:
        print("- None.")
    print()

    print("## Accuracy Gaps")
    if accuracy:
        for gap in accuracy:
            print(f"- `{gap.experiment}`: {gap.detail} Action: {gap.action}")
    else:
        print("- Siderust is currently the most accurate measured candidate in every published experiment.")
    print()

    print("## Performance Gaps")
    if performance:
        for gap in performance:
            print(f"- `{gap.experiment}`: {gap.detail} Action: {gap.action}")
    else:
        print("- Siderust is currently the fastest measured candidate in every published experiment.")
    print()

    print("## Priority Order")
    priorities = coverage[:]
    priorities.extend(performance[:5])
    priorities.extend(accuracy[:5])
    if priorities:
        for idx, gap in enumerate(priorities, start=1):
            print(f"{idx}. `{gap.experiment}`: {gap.action}")
    else:
        print("1. Keep the benchmark suite green and re-run the roadmap after each significant Siderust change.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent / "latest_results"),
        help="Path to latest_results/ or a specific results/<run_id> directory.",
    )
    args = parser.parse_args()

    root = Path(args.results_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory not found: {root}")

    coverage, accuracy, performance, results = summarize(root)
    print_markdown(root, coverage, accuracy, performance, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
