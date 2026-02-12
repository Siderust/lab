"""
Results Loader
==============

Discovers, parses, and normalises the JSON result files produced by the
orchestrator (``results/<date>/<experiment>/<library>.json``).

Public API
----------
- ``discover_runs(results_dir)``  → list of ``RunInfo``
- ``load_run(run_info)``          → list of ``ExperimentResult``
- ``ExperimentResult``            — dataclass holding one result file's data
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
RAD_TO_MAS = 180.0 / math.pi * 3600.0 * 1000.0
RAD_TO_ARCSEC = 180.0 / math.pi * 3600.0
NS_PER_US = 1000.0


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PercentileStats:
    """A bag of percentile/summary statistics for a single metric."""

    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None
    min: float | None = None
    mean: float | None = None
    rms: float | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "PercentileStats":
        if d is None:
            return cls()
        return cls(
            p50=d.get("p50"),
            p90=d.get("p90"),
            p95=d.get("p95"),
            p99=d.get("p99"),
            max=d.get("max"),
            min=d.get("min"),
            mean=d.get("mean"),
            rms=d.get("rms"),
        )

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())


@dataclass
class WorstCase:
    case_id: str = ""
    jd_tt: float | None = None
    angular_error_mas: float | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict, idx: int = 0) -> "WorstCase":
        return cls(
            case_id=d.get("case_id", f"case_{idx}"),
            jd_tt=d.get("jd_tt"),
            angular_error_mas=d.get("angular_error_mas"),
            notes=d.get("notes", ""),
        )


@dataclass
class Performance:
    per_op_ns: float | None = None
    throughput_ops_s: float | None = None
    total_ns: float | None = None
    batch_size: int | None = None

    @property
    def per_op_us(self) -> float | None:
        return self.per_op_ns / NS_PER_US if self.per_op_ns is not None else None

    @classmethod
    def from_dict(cls, d: dict | None) -> "Performance":
        if d is None:
            return cls()
        return cls(
            per_op_ns=d.get("per_op_ns"),
            throughput_ops_s=d.get("throughput_ops_s"),
            total_ns=d.get("total_ns"),
            batch_size=d.get("batch_size"),
        )

    def is_empty(self) -> bool:
        return self.per_op_ns is None and self.throughput_ops_s is None


@dataclass
class ExperimentResult:
    """One parsed result JSON (one library × one experiment × one run)."""

    # Identity
    experiment: str = ""
    candidate_library: str = ""
    reference_library: str = ""
    mode: str = "common_denominator"

    # Alignment / assumptions
    alignment: dict = field(default_factory=dict)

    # Inputs
    input_count: int = 0
    seed: int | None = None
    epoch_range: list[str] = field(default_factory=list)

    # Accuracy — frame_rotation_bpn
    angular_error_mas: PercentileStats = field(default_factory=PercentileStats)
    closure_error_rad: PercentileStats = field(default_factory=PercentileStats)
    matrix_frobenius: PercentileStats = field(default_factory=PercentileStats)
    nan_count: int = 0
    inf_count: int = 0
    worst_cases: list[WorstCase] = field(default_factory=list)

    # Accuracy — gmst_era
    gmst_error_rad: PercentileStats = field(default_factory=PercentileStats)
    gmst_error_arcsec: PercentileStats = field(default_factory=PercentileStats)
    era_error_rad: PercentileStats = field(default_factory=PercentileStats)

    # Performance
    performance: Performance = field(default_factory=Performance)
    reference_performance: Performance = field(default_factory=Performance)

    # Run metadata
    run_metadata: dict = field(default_factory=dict)

    # Source
    source_path: str = ""
    run_date: str = ""

    @property
    def speedup_vs_ref(self) -> float | None:
        ref = self.reference_performance.per_op_ns
        cand = self.performance.per_op_ns
        if ref and cand and cand > 0:
            return ref / cand
        return None

    @property
    def has_performance(self) -> bool:
        return not self.performance.is_empty()

    @property
    def has_accuracy(self) -> bool:
        return (
            not self.angular_error_mas.is_empty()
            or not self.gmst_error_arcsec.is_empty()
        )

    @property
    def primary_error_label(self) -> str:
        """Human-readable label for the primary accuracy metric."""
        if self.experiment == "frame_rotation_bpn":
            return "Angular error (mas)"
        elif self.experiment == "gmst_era":
            return "GMST error (arcsec)"
        return "Error"

    @property
    def primary_error_stats(self) -> PercentileStats:
        """The primary accuracy metric for this experiment type."""
        if self.experiment == "frame_rotation_bpn":
            return self.angular_error_mas
        elif self.experiment == "gmst_era":
            return self.gmst_error_arcsec
        return PercentileStats()


def _parse_result(data: dict, source_path: str, run_date: str) -> ExperimentResult:
    """Parse a single result JSON dict into an ExperimentResult."""
    acc = data.get("accuracy", {})
    inp = data.get("inputs", {})
    alignment = data.get("alignment", {})

    return ExperimentResult(
        experiment=data.get("experiment", ""),
        candidate_library=data.get("candidate_library", ""),
        reference_library=data.get("reference_library", ""),
        mode=alignment.get("mode", "common_denominator"),
        alignment=alignment,
        input_count=inp.get("count", 0),
        seed=inp.get("seed"),
        epoch_range=inp.get("epoch_range", []),
        # BPN accuracy
        angular_error_mas=PercentileStats.from_dict(
            acc.get("angular_error_mas")
        ),
        closure_error_rad=PercentileStats.from_dict(
            acc.get("closure_error_rad")
        ),
        matrix_frobenius=PercentileStats.from_dict(
            acc.get("matrix_frobenius")
        ),
        nan_count=acc.get("nan_count", 0),
        inf_count=acc.get("inf_count", 0),
        worst_cases=[
            WorstCase.from_dict(w, i)
            for i, w in enumerate(acc.get("worst_cases", []))
        ],
        # GMST/ERA accuracy
        gmst_error_rad=PercentileStats.from_dict(acc.get("gmst_error_rad")),
        gmst_error_arcsec=PercentileStats.from_dict(
            acc.get("gmst_error_arcsec")
        ),
        era_error_rad=PercentileStats.from_dict(acc.get("era_error_rad")),
        # Performance
        performance=Performance.from_dict(data.get("performance")),
        reference_performance=Performance.from_dict(
            data.get("reference_performance")
        ),
        # Metadata
        run_metadata=data.get("run_metadata", {}),
        source_path=source_path,
        run_date=run_date,
    )


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RunInfo:
    """Identifies a single run folder: results/<date>/<experiment>."""

    date: str
    experiment: str
    path: Path
    libraries: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.date} / {self.experiment}"


def discover_runs(results_dir: Path) -> list[RunInfo]:
    """Scan ``results/`` and return a list of discovered runs."""
    runs: list[RunInfo] = []
    if not results_dir.is_dir():
        return runs

    for date_dir in sorted(results_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for exp_dir in sorted(date_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            libs = sorted(
                p.stem
                for p in exp_dir.glob("*.json")
                if p.stem != "summary"
            )
            if libs:
                runs.append(
                    RunInfo(
                        date=date_dir.name,
                        experiment=exp_dir.name,
                        path=exp_dir,
                        libraries=libs,
                    )
                )
    return runs


def load_results(run: RunInfo, libraries: list[str] | None = None) -> list[ExperimentResult]:
    """Load result JSONs for the given run.  Optionally filter by library names."""
    results: list[ExperimentResult] = []
    for json_path in sorted(run.path.glob("*.json")):
        if json_path.stem == "summary":
            continue
        if libraries and json_path.stem not in libraries:
            continue
        try:
            data = json.loads(json_path.read_text())
            results.append(
                _parse_result(data, str(json_path), run.date)
            )
        except (json.JSONDecodeError, KeyError) as exc:
            # Gracefully skip malformed files; caller can check len(results)
            print(f"⚠ Skipping {json_path}: {exc}")
    return results


# ──────────────────────────────────────────────────────────────────────
# Convenience: flatten all runs into one list
# ──────────────────────────────────────────────────────────────────────
def load_all(results_dir: Path) -> list[ExperimentResult]:
    """Load every result JSON under ``results_dir``."""
    all_results: list[ExperimentResult] = []
    for run in discover_runs(results_dir):
        all_results.extend(load_results(run))
    return all_results
