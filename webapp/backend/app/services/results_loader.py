"""Scan the results/ and reports/ directories, parse JSON, and index runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..models.schemas import (
    ExperimentResult,
    RunDetail,
    RunSummary,
)

# Matches YYYY-MM-DD directory names
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ResultsLoader:
    """Reads and caches benchmark results from the filesystem."""

    def __init__(self, lab_root: Path) -> None:
        self.lab_root = lab_root
        self.results_dir = lab_root / "results"
        self.reports_dir = lab_root / "reports"
        # run_id -> { experiment -> [ExperimentResult, …] }
        self._cache: dict[str, dict[str, list[ExperimentResult]]] = {}
        self._report_index: dict[str, dict[str, bool]] = {}  # run_id -> {exp: True}
        self.reload()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-scan the filesystem and rebuild the cache."""
        self._cache.clear()
        self._report_index.clear()
        self._scan_results()
        self._scan_reports()

    def list_runs(self) -> list[RunSummary]:
        """Return lightweight summaries for every discovered run."""
        summaries: list[RunSummary] = []
        for run_id in sorted(self._cache.keys(), reverse=True):
            experiments = self._cache[run_id]
            all_results = [r for results in experiments.values() for r in results]

            # Derive metadata from the first result that has it
            git_shas: dict[str, str] = {}
            machine: str | None = None
            timestamp: str | None = None
            libraries: set[str] = set()

            for r in all_results:
                libraries.add(r.candidate_library)
                if r.run_metadata:
                    if not timestamp:
                        timestamp = r.run_metadata.date
                    if not git_shas:
                        git_shas = r.run_metadata.git_shas
                    if not machine:
                        parts = []
                        if r.run_metadata.cpu:
                            parts.append(r.run_metadata.cpu)
                        if r.run_metadata.os:
                            parts.append(r.run_metadata.os)
                        machine = " / ".join(parts) if parts else None

            summaries.append(
                RunSummary(
                    id=run_id,
                    timestamp=timestamp,
                    git_shas=git_shas,
                    machine=machine,
                    experiments=sorted(experiments.keys()),
                    libraries=sorted(libraries),
                    result_count=len(all_results),
                    has_reports=run_id in self._report_index,
                )
            )
        return summaries

    def get_run(self, run_id: str) -> RunDetail | None:
        """Return the full run (all experiments, all libraries)."""
        experiments = self._cache.get(run_id)
        if experiments is None:
            return None

        all_results = [r for results in experiments.values() for r in results]
        git_shas: dict[str, str] = {}
        machine: str | None = None
        timestamp: str | None = None

        for r in all_results:
            if r.run_metadata:
                if not timestamp:
                    timestamp = r.run_metadata.date
                if not git_shas:
                    git_shas = r.run_metadata.git_shas
                if not machine:
                    parts = []
                    if r.run_metadata.cpu:
                        parts.append(r.run_metadata.cpu)
                    if r.run_metadata.os:
                        parts.append(r.run_metadata.os)
                    machine = " / ".join(parts) if parts else None

        return RunDetail(
            id=run_id,
            timestamp=timestamp,
            git_shas=git_shas,
            machine=machine,
            experiments=experiments,
            has_reports=self._report_index.get(run_id, {}),
        )

    def get_experiment(
        self, run_id: str, experiment: str
    ) -> list[ExperimentResult] | None:
        """Return all library results for one experiment in a run."""
        experiments = self._cache.get(run_id)
        if experiments is None:
            return None
        return experiments.get(experiment)

    def get_run_ids(self) -> list[str]:
        return sorted(self._cache.keys(), reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_results(self) -> None:
        if not self.results_dir.is_dir():
            return
        for date_dir in sorted(self.results_dir.iterdir()):
            if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
                continue
            run_id = date_dir.name
            self._cache[run_id] = {}
            for exp_dir in sorted(date_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                experiment = exp_dir.name
                results: list[ExperimentResult] = []
                for json_file in sorted(exp_dir.glob("*.json")):
                    try:
                        data = json.loads(json_file.read_text())
                        results.append(ExperimentResult(**data))
                    except Exception:
                        continue
                if results:
                    self._cache[run_id][experiment] = results

    def _scan_reports(self) -> None:
        if not self.reports_dir.is_dir():
            return
        for date_dir in sorted(self.reports_dir.iterdir()):
            if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
                continue
            run_id = date_dir.name
            self._report_index[run_id] = {}
            for exp_dir in sorted(date_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                # Consider a report available if index.html or report.md exists
                if (exp_dir / "index.html").exists() or (exp_dir / "report.md").exists():
                    self._report_index[run_id][exp_dir.name] = True


def _extract_metric_value(accuracy: dict[str, Any], path: list[str]) -> float | None:
    """Walk into nested dicts to get a numeric value."""
    obj: Any = accuracy
    for key in path:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    if isinstance(obj, (int, float)) and obj is not None:
        return float(obj)
    return None


def compute_comparison(
    loader: ResultsLoader, run_a: str, run_b: str
) -> list[dict[str, Any]]:
    """Compare two runs and return metric deltas."""
    from ..models.schemas import MetricDelta

    detail_a = loader.get_run(run_a)
    detail_b = loader.get_run(run_b)
    if detail_a is None or detail_b is None:
        return []

    # Metric paths to compare — (human label, [dict path])
    metric_paths: list[tuple[str, list[str]]] = [
        # BPN
        ("angular_error_mas.p99", ["angular_error_mas", "p99"]),
        ("angular_error_mas.max", ["angular_error_mas", "max"]),
        # GMST
        ("gmst_error_arcsec.p99", ["gmst_error_arcsec", "p99"]),
        ("gmst_error_arcsec.max", ["gmst_error_arcsec", "max"]),
        # Angular (equ_ecl, equ_horizontal, solar, lunar)
        ("angular_sep_arcsec.p99", ["angular_sep_arcsec", "p99"]),
        ("angular_sep_arcsec.max", ["angular_sep_arcsec", "max"]),
        # Kepler
        ("E_error_rad.max", ["E_error_rad", "max"]),
        ("consistency_error_rad.max", ["consistency_error_rad", "max"]),
    ]

    deltas: list[dict[str, Any]] = []

    all_experiments = set(detail_a.experiments.keys()) | set(detail_b.experiments.keys())
    for exp in sorted(all_experiments):
        results_a = {r.candidate_library: r for r in detail_a.experiments.get(exp, [])}
        results_b = {r.candidate_library: r for r in detail_b.experiments.get(exp, [])}
        all_libs = set(results_a.keys()) | set(results_b.keys())

        for lib in sorted(all_libs):
            ra = results_a.get(lib)
            rb = results_b.get(lib)
            acc_a = ra.accuracy if ra else {}
            acc_b = rb.accuracy if rb else {}

            for label, path in metric_paths:
                va = _extract_metric_value(acc_a, path)
                vb = _extract_metric_value(acc_b, path)
                if va is None and vb is None:
                    continue

                delta = None
                delta_pct = None
                regression = False
                if va is not None and vb is not None:
                    delta = vb - va
                    if va != 0:
                        delta_pct = (delta / abs(va)) * 100.0
                    # For error metrics, increase = regression
                    regression = delta is not None and delta > 0

                deltas.append(
                    MetricDelta(
                        experiment=exp,
                        library=lib,
                        metric=label,
                        value_a=va,
                        value_b=vb,
                        delta=delta,
                        delta_pct=delta_pct,
                        regression=regression,
                    ).model_dump()
                )

    return deltas
