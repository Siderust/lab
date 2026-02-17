"""Pydantic models matching the JSON artifacts in results/."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reusable stat blocks
# ---------------------------------------------------------------------------

class PercentileStats(BaseModel):
    """Percentile summary produced by the orchestrator."""

    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = Field(None, alias="max")
    min: float | None = None
    mean: float | None = None
    rms: float | None = None


# ---------------------------------------------------------------------------
# Accuracy blocks (per-experiment flavour)
# ---------------------------------------------------------------------------

class WorstCase(BaseModel):
    jd_tt: float | None = None
    angular_error_mas: float | None = None


class BpnAccuracy(BaseModel):
    reference: str | None = None
    candidate: str | None = None
    angular_error_mas: PercentileStats | None = None
    closure_error_rad: PercentileStats | None = None
    matrix_frobenius: PercentileStats | None = None
    nan_count: int = 0
    inf_count: int = 0
    worst_cases: list[WorstCase] = []


class GmstAccuracy(BaseModel):
    reference: str | None = None
    candidate: str | None = None
    gmst_error_rad: PercentileStats | None = None
    gmst_error_arcsec: PercentileStats | None = None
    era_error_rad: PercentileStats | None = None


class AngularAccuracy(BaseModel):
    """Shared by equ_ecl, equ_horizontal, solar_position, lunar_position."""

    reference: str | None = None
    candidate: str | None = None
    angular_sep_arcsec: PercentileStats | None = None
    signed_ra_error_arcsec: PercentileStats | None = None
    signed_dec_error_arcsec: PercentileStats | None = None
    nan_count: int = 0
    # Optional extra keys (dist_au_diff, dist_km_diff, …)
    dist_au_diff: PercentileStats | None = None
    dist_km_diff: PercentileStats | None = None


class KeplerAccuracy(BaseModel):
    reference: str | None = None
    candidate: str | None = None
    E_error_rad: PercentileStats | None = None
    nu_error_rad: PercentileStats | None = None
    consistency_error_rad: PercentileStats | None = None
    nan_count: int = 0


# ---------------------------------------------------------------------------
# Performance (enhanced with multi-sample statistics)
# ---------------------------------------------------------------------------

class PerformanceData(BaseModel):
    """Performance data — supports both old (single-pass) and new (multi-sample) formats."""
    per_op_ns: float | None = None
    per_op_ns_mean: float | None = None
    per_op_ns_median: float | None = None
    per_op_ns_std_dev: float | None = None
    per_op_ns_min: float | None = None
    per_op_ns_max: float | None = None
    per_op_ns_ci95: list[float] | None = None
    per_op_ns_cv_pct: float | None = None
    throughput_ops_s: float | None = None
    total_ns: float | None = None
    total_ns_median: float | None = None
    batch_size: int | None = None
    rounds: int | None = None
    samples: list[float] | None = None
    warnings: list[str] | None = None


# ---------------------------------------------------------------------------
# Experiment description (for non-expert users)
# ---------------------------------------------------------------------------

class ExperimentDescription(BaseModel, extra="allow"):
    """Human-readable description of what an experiment measures."""
    title: str | None = None
    what: str | None = None
    why: str | None = None
    units: str | None = None
    interpret: str | None = None


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration used for this benchmark run."""
    perf_rounds: int = 0
    perf_warmup: int = 0
    perf_enabled: bool = False


# ---------------------------------------------------------------------------
# Alignment / parity
# ---------------------------------------------------------------------------

class AlignmentChecklist(BaseModel, extra="allow"):
    """Alignment checklist — intentionally loose to accept any experiment."""

    units: dict[str, str] | None = None
    time_input: str | None = None
    time_scales: str | None = None
    leap_seconds: str | None = None
    earth_orientation: dict[str, str] | None = None
    geodesy: str | None = None
    refraction: str | None = None
    ephemeris_source: str | None = None
    models: dict[str, str] | None = None
    mode: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

class RunMetadata(BaseModel):
    date: str | None = None
    git_shas: dict[str, str] = {}
    git_branch: str | None = None
    cpu: str | None = None
    cpu_model: str | None = None
    cpu_count: int | None = None
    os: str | None = None
    platform_detail: str | None = None
    toolchain: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Top-level result per (experiment, candidate_library) pair
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel, extra="allow"):
    """One JSON file from results/<date>/<experiment>/<library>.json."""

    experiment: str
    candidate_library: str
    reference_library: str
    description: ExperimentDescription | dict[str, Any] = {}
    alignment: AlignmentChecklist | None = None
    inputs: dict[str, Any] = {}
    accuracy: dict[str, Any] = {}
    performance: PerformanceData | dict[str, Any] = {}
    reference_performance: PerformanceData | dict[str, Any] = {}
    benchmark_config: BenchmarkConfig | dict[str, Any] = {}
    run_metadata: RunMetadata | None = None


# ---------------------------------------------------------------------------
# Run = collection of experiment results for one date
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    """Lightweight object returned by GET /api/runs."""

    id: str  # date string, e.g. "2026-02-12"
    timestamp: str | None = None
    git_shas: dict[str, str] = {}
    machine: str | None = None
    experiments: list[str] = []
    libraries: list[str] = []
    result_count: int = 0
    has_reports: bool = False


class RunDetail(BaseModel):
    """Full run returned by GET /api/runs/{id}."""

    id: str
    timestamp: str | None = None
    git_shas: dict[str, str] = {}
    machine: str | None = None
    experiments: dict[str, list[ExperimentResult]] = {}
    has_reports: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class MetricDelta(BaseModel):
    experiment: str
    library: str
    metric: str
    value_a: float | None = None
    value_b: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    regression: bool = False


class CompareResult(BaseModel):
    run_a: str
    run_b: str
    deltas: list[MetricDelta] = []


# ---------------------------------------------------------------------------
# Benchmark job
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    experiments: list[str] = ["all"]
    n: int = 1000
    seed: int = 42
    no_perf: bool = False
    perf_rounds: int = 5
    ci_mode: bool = False
    notes: str = ""


class BenchmarkStatus(BaseModel):
    job_id: str
    status: str = "pending"  # pending | running | completed | failed | cancelled
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    config: BenchmarkRequest | None = None
    current_experiment: str | None = None
    current_step: str | None = None
    progress_pct: float | None = None
