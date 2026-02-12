"""
Sanity checks for results_loader, report_generator, and theme.

Run with:  python -m pytest ui/test_loader.py -v
"""

import json
from pathlib import Path

import pytest

from ui.results_loader import (
    ExperimentResult,
    PercentileStats,
    Performance,
    RunInfo,
    WorstCase,
    discover_runs,
    load_results,
)
from ui.report_generator import (
    fmt,
    _fmt,
    color,
    _color,
    one_line_summary,
    generate_markdown_report,
)


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

SAMPLE_BPN = {
    "experiment": "frame_rotation_bpn",
    "candidate_library": "siderust",
    "reference_library": "erfa",
    "alignment": {"mode": "common_denominator", "units": {"angles": "radians"}},
    "inputs": {"count": 100, "seed": 42, "epoch_range": ["JD 2451545.0", "JD 2460000.0"]},
    "accuracy": {
        "reference": "erfa",
        "candidate": "siderust",
        "angular_error_mas": {
            "p50": 44.0, "p90": 72.0, "p95": 76.0, "p99": 83.5,
            "max": 230.5, "min": 0.0, "mean": 47.0, "rms": 50.4,
        },
        "closure_error_rad": {
            "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 1.49e-08,
            "max": 2.1e-08, "min": 0.0, "mean": 6e-10, "rms": 3e-09,
        },
        "matrix_frobenius": {
            "p50": 3.9e-07, "p90": 5.4e-07, "p95": 5.6e-07, "p99": 6.1e-07,
            "max": 2.2e-06, "min": 2.9e-07, "mean": 4.1e-07, "rms": 4.2e-07,
        },
        "nan_count": 0,
        "inf_count": 0,
        "worst_cases": [
            {"jd_tt": 2415020.0, "angular_error_mas": 230.5},
            {"jd_tt": 2488069.5, "angular_error_mas": 166.5},
        ],
    },
    "performance": {
        "per_op_ns": 1219.3,
        "throughput_ops_s": 820127,
        "total_ns": 12193228,
        "batch_size": 10000,
    },
    "reference_performance": {
        "per_op_ns": 46260.6,
        "throughput_ops_s": 21617,
    },
    "run_metadata": {
        "date": "2026-02-12T17:55:18Z",
        "git_shas": {"lab": "abc123", "siderust": "def456"},
        "cpu": "x86_64",
        "os": "Linux 6.17",
        "toolchain": {"python": "3.12.3"},
    },
}

SAMPLE_GMST = {
    "experiment": "gmst_era",
    "candidate_library": "siderust",
    "reference_library": "erfa",
    "alignment": {"mode": "common_denominator"},
    "inputs": {"count": 100, "seed": 42},
    "accuracy": {
        "reference": "erfa",
        "candidate": "siderust",
        "gmst_error_rad": {"p50": 1.3e-07, "p90": 2.9e-07, "p99": 3.3e-07, "max": 1.4e-06, "mean": 1.4e-07, "rms": 1.8e-07},
        "gmst_error_arcsec": {"p50": 0.027, "p90": 0.060, "p99": 0.068, "max": 0.286, "mean": 0.029, "rms": 0.036},
        "era_error_rad": {"p50": 2.1e-12, "p90": 1.1e-11, "p99": 1.8e-11, "max": 2.6e-11, "mean": 4.1e-12, "rms": 6.1e-12},
    },
    "run_metadata": {"date": "2026-02-12T17:55:21Z"},
}


@pytest.fixture
def tmp_results(tmp_path):
    """Create a temporary results directory with sample data."""
    bpn_dir = tmp_path / "2026-02-12" / "frame_rotation_bpn"
    bpn_dir.mkdir(parents=True)
    (bpn_dir / "siderust.json").write_text(json.dumps(SAMPLE_BPN))

    gmst_dir = tmp_path / "2026-02-12" / "gmst_era"
    gmst_dir.mkdir(parents=True)
    (gmst_dir / "siderust.json").write_text(json.dumps(SAMPLE_GMST))

    return tmp_path


@pytest.fixture
def multi_date_results(tmp_path):
    """Results across two dates for trend tests."""
    for date in ("2026-02-10", "2026-02-12"):
        bpn_dir = tmp_path / date / "frame_rotation_bpn"
        bpn_dir.mkdir(parents=True)
        data = dict(SAMPLE_BPN)
        # Slightly vary the data for the earlier date
        if date == "2026-02-10":
            data = json.loads(json.dumps(SAMPLE_BPN))
            data["accuracy"]["angular_error_mas"]["p99"] = 90.0
        (bpn_dir / "siderust.json").write_text(json.dumps(data))
    return tmp_path


# ──────────────────────────────────────────────────────────
# Tests: PercentileStats
# ──────────────────────────────────────────────────────────

def test_percentile_stats_from_dict():
    s = PercentileStats.from_dict({"p50": 1.5, "p99": 10.0, "max": 20.0})
    assert s.p50 == 1.5
    assert s.p99 == 10.0
    assert s.max == 20.0
    assert s.p90 is None  # missing key


def test_percentile_stats_empty():
    s = PercentileStats.from_dict(None)
    assert s.is_empty()


def test_percentile_stats_as_dict():
    s = PercentileStats(p50=1.0, max=5.0)
    d = s.as_dict()
    assert d == {"p50": 1.0, "max": 5.0}


# ──────────────────────────────────────────────────────────
# Tests: Performance
# ──────────────────────────────────────────────────────────

def test_performance_per_op_us():
    p = Performance(per_op_ns=1500.0)
    assert p.per_op_us == 1.5


def test_performance_empty():
    p = Performance()
    assert p.is_empty()
    assert p.per_op_us is None


# ──────────────────────────────────────────────────────────
# Tests: Discovery & Loading
# ──────────────────────────────────────────────────────────

def test_discover_runs(tmp_results):
    runs = discover_runs(tmp_results)
    assert len(runs) == 2
    labels = {r.experiment for r in runs}
    assert "frame_rotation_bpn" in labels
    assert "gmst_era" in labels


def test_discover_runs_empty(tmp_path):
    runs = discover_runs(tmp_path)
    assert len(runs) == 0


def test_load_results_bpn(tmp_results):
    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    results = load_results(bpn_run)
    assert len(results) == 1

    r = results[0]
    assert r.experiment == "frame_rotation_bpn"
    assert r.candidate_library == "siderust"
    assert r.angular_error_mas.p50 == 44.0
    assert r.angular_error_mas.max == 230.5
    assert r.nan_count == 0
    assert len(r.worst_cases) == 2
    assert r.worst_cases[0].angular_error_mas == 230.5
    assert r.performance.per_op_ns == 1219.3
    assert r.speedup_vs_ref == pytest.approx(37.94, rel=0.01)


def test_load_results_gmst(tmp_results):
    runs = discover_runs(tmp_results)
    gmst_run = next(r for r in runs if r.experiment == "gmst_era")
    results = load_results(gmst_run)
    assert len(results) == 1

    r = results[0]
    assert r.experiment == "gmst_era"
    assert r.gmst_error_arcsec.p50 == 0.027
    assert r.era_error_rad.max == 2.6e-11


def test_load_results_filter(tmp_results):
    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    # Filter for a library that doesn't exist
    results = load_results(bpn_run, libraries=["nonexistent"])
    assert len(results) == 0


# ──────────────────────────────────────────────────────────
# Tests: ExperimentResult properties
# ──────────────────────────────────────────────────────────

def test_primary_error_bpn(tmp_results):
    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    r = load_results(bpn_run)[0]
    assert r.primary_error_label == "Angular error (mas)"
    assert r.primary_error_stats.p50 == 44.0


def test_primary_error_gmst(tmp_results):
    runs = discover_runs(tmp_results)
    gmst_run = next(r for r in runs if r.experiment == "gmst_era")
    r = load_results(gmst_run)[0]
    assert r.primary_error_label == "GMST error (arcsec)"
    assert r.primary_error_stats.p50 == 0.027


# ──────────────────────────────────────────────────────────
# Tests: Formatting and summaries
# ──────────────────────────────────────────────────────────

def test_fmt():
    assert fmt(None) == "—"
    assert fmt(3.14159, 2) == "3.14"
    assert fmt(0.0, 6) == "0.000000"


def test_fmt_alias():
    """_fmt is a back-compat alias for fmt."""
    assert _fmt(1.5, 1) == fmt(1.5, 1)


def test_color():
    assert color("siderust") == "#3b82f6"
    assert color("unknown_lib") == "#6b7280"


def test_color_alias():
    assert _color("siderust") == color("siderust")


def test_one_line_summary(tmp_results):
    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    results = load_results(bpn_run)
    summary = one_line_summary(results)
    assert "siderust" in summary
    assert "83.50" in summary  # p99


def test_markdown_report(tmp_results):
    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    results = load_results(bpn_run)
    md = generate_markdown_report(results)
    assert "# Lab Report" in md
    assert "Alignment Checklist" in md
    assert "Run Metadata" in md
    assert "Reproduction" in md
    assert "siderust" in md


# ──────────────────────────────────────────────────────────
# Tests: Graceful handling of missing/malformed data
# ──────────────────────────────────────────────────────────

def test_missing_fields_graceful(tmp_path):
    """An older JSON missing some fields should still parse."""
    exp_dir = tmp_path / "2025-01-01" / "old_experiment"
    exp_dir.mkdir(parents=True)
    minimal = {
        "experiment": "old_experiment",
        "candidate_library": "siderust",
        "reference_library": "erfa",
    }
    (exp_dir / "siderust.json").write_text(json.dumps(minimal))

    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    results = load_results(runs[0])
    assert len(results) == 1
    r = results[0]
    assert r.experiment == "old_experiment"
    assert r.angular_error_mas.is_empty()
    assert r.performance.is_empty()
    assert r.nan_count == 0


def test_malformed_json_skipped(tmp_path):
    """Malformed JSON files should be skipped gracefully."""
    exp_dir = tmp_path / "2025-01-01" / "bad"
    exp_dir.mkdir(parents=True)
    (exp_dir / "broken.json").write_text("{invalid json")

    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    results = load_results(runs[0])
    assert len(results) == 0  # Skipped


# ──────────────────────────────────────────────────────────
# Tests: Multi-date discovery (for trends)
# ──────────────────────────────────────────────────────────

def test_multi_date_discovery(multi_date_results):
    runs = discover_runs(multi_date_results)
    bpn_runs = [r for r in runs if r.experiment == "frame_rotation_bpn"]
    assert len(bpn_runs) == 2
    dates = {r.date for r in bpn_runs}
    assert "2026-02-10" in dates
    assert "2026-02-12" in dates


def test_multi_date_load(multi_date_results):
    runs = discover_runs(multi_date_results)
    bpn_runs = sorted(
        [r for r in runs if r.experiment == "frame_rotation_bpn"],
        key=lambda r: r.date,
    )
    r1 = load_results(bpn_runs[0])[0]  # 2026-02-10
    r2 = load_results(bpn_runs[1])[0]  # 2026-02-12
    assert r1.angular_error_mas.p99 == 90.0
    assert r2.angular_error_mas.p99 == 83.5


# ──────────────────────────────────────────────────────────
# Tests: Box plot has visible IQR (regression test)
# ──────────────────────────────────────────────────────────

def test_box_plot_visible_iqr(tmp_results):
    """Box plot Q1 and median must differ (non-zero height)."""
    from ui.report_generator import make_accuracy_box

    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    results = load_results(bpn_run)
    fig = make_accuracy_box(results)

    # The first (only) box trace
    trace = fig.data[0]
    q1 = trace.q1[0]
    median = trace.median[0]
    q3 = trace.q3[0]
    # Q1 should be p50 (44.0), median should be p90 (72.0), Q3 should be p95 (76.0)
    assert q1 != median, "Q1 and median should differ for a visible box"
    assert median != q3, "Median and Q3 should differ"
    assert q1 == 44.0
    assert median == 72.0
    assert q3 == 76.0


# ──────────────────────────────────────────────────────────
# Tests: Repro block uses real path (not placeholder)
# ──────────────────────────────────────────────────────────

def test_repro_block_real_path(tmp_results):
    from ui.report_generator import _repro_block

    runs = discover_runs(tmp_results)
    bpn_run = next(r for r in runs if r.experiment == "frame_rotation_bpn")
    r = load_results(bpn_run)[0]
    block = _repro_block(r)
    assert "/path/to/" not in block, "Placeholder path should be replaced"
    assert "cd " in block
