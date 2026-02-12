"""Accuracy tab — bar charts, box plots, ECDF, closure / Frobenius tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.report_generator import (
    color,
    fmt,
    make_accuracy_bar,
    make_accuracy_box,
    make_gmst_era_bar,
)
from ui.theme import is_dark, plotly_layout

if TYPE_CHECKING:
    from ui.results_loader import ExperimentResult


def _approx_ecdf(results: list["ExperimentResult"], dark: bool) -> go.Figure:
    """Build an approximate ECDF from available percentiles."""
    fig = go.Figure()
    for r in results:
        s = r.angular_error_mas
        n = max(r.input_count, 1)
        points: list[tuple[float, float]] = []
        # Use 1/n (not 0) for the min — proper ECDF
        if s.min is not None:
            points.append((s.min, 1.0 / n))
        if s.p50 is not None:
            points.append((s.p50, 0.50))
        if s.p90 is not None:
            points.append((s.p90, 0.90))
        if s.p95 is not None:
            points.append((s.p95, 0.95))
        if s.p99 is not None:
            points.append((s.p99, 0.99))
        if s.max is not None:
            points.append((s.max, 1.0))

        if points:
            xs, ys = zip(*sorted(points))
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    name=r.candidate_library,
                    line=dict(color=color(r.candidate_library), width=2),
                    marker=dict(size=7),
                )
            )

    fig.update_layout(
        title="Approximate ECDF — Angular Error (mas)",
        xaxis_title="Angular error (mas)",
        yaxis_title="Cumulative fraction",
        **plotly_layout(),
    )
    return fig


def render(results: list["ExperimentResult"], experiment: str) -> None:
    dark = is_dark()

    if experiment == "frame_rotation_bpn":
        # Bar chart
        fig_bar = make_accuracy_bar(results, title="Angular Error — Percentile Summary", dark=dark)
        st.plotly_chart(fig_bar, width="stretch", key="acc_bar")

        # Box plot (now with visible IQR)
        fig_box = make_accuracy_box(results, dark=dark)
        st.plotly_chart(fig_box, width="stretch", key="acc_box")

        # ECDF
        st.markdown('<div class="section-header">Approximate ECDF</div>', unsafe_allow_html=True)
        fig_ecdf = _approx_ecdf(results, dark)
        st.plotly_chart(fig_ecdf, width="stretch", key="acc_ecdf")

        # Closure error
        st.markdown('<div class="section-header">Closure Error (round-trip A→B→A)</div>', unsafe_allow_html=True)
        closure_data = []
        for r in results:
            s = r.closure_error_rad
            if not s.is_empty():
                closure_data.append({
                    "Library": r.candidate_library,
                    "p50 (rad)": fmt(s.p50, 12),
                    "p99 (rad)": fmt(s.p99, 12),
                    "max (rad)": fmt(s.max, 12),
                    "RMS (rad)": fmt(s.rms, 12),
                })
        if closure_data:
            st.dataframe(pd.DataFrame(closure_data), width="stretch", hide_index=True)

        # Matrix Frobenius
        st.markdown('<div class="section-header">Matrix Frobenius Norm</div>', unsafe_allow_html=True)
        frob_data = []
        for r in results:
            s = r.matrix_frobenius
            if not s.is_empty():
                frob_data.append({
                    "Library": r.candidate_library,
                    "p50": f"{s.p50:.2e}" if s.p50 is not None else "—",
                    "p99": f"{s.p99:.2e}" if s.p99 is not None else "—",
                    "max": f"{s.max:.2e}" if s.max is not None else "—",
                })
        if frob_data:
            st.dataframe(pd.DataFrame(frob_data), width="stretch", hide_index=True)

    elif experiment == "gmst_era":
        fig_gmst = make_gmst_era_bar(results, dark=dark)
        st.plotly_chart(fig_gmst, width="stretch", key="gmst_bar")

        # Box plot for GMST too
        fig_box = make_accuracy_box(results, dark=dark)
        st.plotly_chart(fig_box, width="stretch", key="gmst_box")

        # Detailed table
        st.markdown('<div class="section-header">Detailed GMST / ERA Error</div>', unsafe_allow_html=True)
        detail_data = []
        for r in results:
            gs = r.gmst_error_arcsec
            es = r.era_error_rad
            detail_data.append({
                "Library": r.candidate_library,
                "GMST p50 (arcsec)": fmt(gs.p50, 6),
                "GMST p99 (arcsec)": fmt(gs.p99, 6),
                "GMST max (arcsec)": fmt(gs.max, 6),
                "ERA p50 (rad)": f"{es.p50:.2e}" if es.p50 is not None else "—",
                "ERA max (rad)": f"{es.max:.2e}" if es.max is not None else "—",
            })
        st.dataframe(pd.DataFrame(detail_data), width="stretch", hide_index=True)

    else:
        st.info(f"No specialised accuracy view for experiment `{experiment}` yet.")

    # ── CSV export ──────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    csv_rows = []
    for r in results:
        s = r.primary_error_stats
        csv_rows.append({
            "library": r.candidate_library,
            "experiment": r.experiment,
            "p50": s.p50, "p90": s.p90, "p95": s.p95, "p99": s.p99,
            "max": s.max, "min": s.min, "mean": s.mean, "rms": s.rms,
            "nan_count": r.nan_count, "inf_count": r.inf_count,
        })
    if csv_rows:
        df_export = pd.DataFrame(csv_rows)
        csv_str = df_export.to_csv(index=False)
        st.download_button(
            "⬇ Download accuracy table (CSV)",
            data=csv_str,
            file_name=f"{experiment}_accuracy.csv",
            mime="text/csv",
        )
