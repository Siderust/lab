"""Trends tab — cross-run accuracy & latency trends over time."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from ui.report_generator import fmt, make_trend_chart
from ui.results_loader import discover_runs, load_results
from ui.theme import is_dark

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def render(
    results: list["ExperimentResult"],
    experiment: str,
    results_dir: Path,
) -> None:
    dark = is_dark()

    st.markdown(
        "Track how accuracy and latency evolve across pipeline runs.  "
        "Each point is a different run date."
    )

    # ── Gather data across all dates for this experiment ──
    runs = discover_runs(results_dir)
    experiment_runs = [r for r in runs if r.experiment == experiment]
    experiment_runs.sort(key=lambda r: r.date)

    if len(experiment_runs) <= 1:
        st.info(
            "Only one run date available — trends will appear once you have "
            "multiple runs.  Current values are shown below."
        )
        _show_current(results, experiment)
        return

    trend_data: list[dict] = []
    for run in experiment_runs:
        run_results = load_results(run)
        for r in run_results:
            s = r.primary_error_stats
            trend_data.append({
                "date": run.date,
                "library": r.candidate_library,
                "p50": s.p50,
                "p90": s.p90,
                "p99": s.p99,
                "max": s.max,
                "latency_ns": r.performance.per_op_ns,
            })

    if not trend_data:
        st.warning("No trend data could be loaded.")
        return

    # Metric selector
    metric = st.selectbox(
        "Accuracy metric to track",
        ["p99", "p50", "p90", "max"],
        index=0,
        key="trend_metric",
    )

    fig = make_trend_chart(trend_data, metric=metric, dark=dark)
    st.plotly_chart(fig, width="stretch", key="trend_chart")

    # Delta table
    st.markdown('<div class="section-header">Run-over-run deltas</div>', unsafe_allow_html=True)
    _delta_table(trend_data, metric)


def _show_current(results: list["ExperimentResult"], experiment: str) -> None:
    """Fallback: show current values when only 1 date exists."""
    rows = []
    for r in results:
        s = r.primary_error_stats
        rows.append({
            "Library": r.candidate_library,
            "p50": fmt(s.p50, 3),
            "p99": fmt(s.p99, 3),
            "max": fmt(s.max, 3),
            "Latency (ns)": f"{r.performance.per_op_ns:,.0f}" if r.performance.per_op_ns else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _delta_table(trend_data: list[dict], metric: str) -> None:
    """Show date-over-date changes."""
    # Group by library
    libs: dict[str, list[dict]] = {}
    for d in trend_data:
        libs.setdefault(d["library"], []).append(d)

    rows = []
    for lib, points in libs.items():
        points.sort(key=lambda p: p["date"])
        for i in range(1, len(points)):
            prev, curr = points[i - 1], points[i]
            prev_v = prev.get(metric)
            curr_v = curr.get(metric)
            if prev_v is not None and curr_v is not None and prev_v != 0:
                delta_pct = ((curr_v - prev_v) / abs(prev_v)) * 100
                direction = "⬇ improved" if delta_pct < 0 else "⬆ regressed" if delta_pct > 0 else "— unchanged"
            else:
                delta_pct = None
                direction = "—"

            rows.append({
                "Library": lib,
                "From": prev["date"],
                "To": curr["date"],
                f"{metric} (prev)": fmt(prev_v, 3),
                f"{metric} (curr)": fmt(curr_v, 3),
                "Δ%": f"{delta_pct:+.1f}%" if delta_pct is not None else "—",
                "Status": direction,
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("Not enough data points for delta computation.")
